#!/usr/bin/env python3
"""NAS 원본 클립 라벨링 - gui_server.py 의 [NAS 라벨링] 탭 백엔드.

NAS 데이터셋의 클립을 하나씩 보며 사람이 직접 라벨을 매기는 기능이다.
여러 라벨러가 동시에 쓰므로 계정(로그인), 작업 중 표시, 수정 충돌 검사,
수정 이력을 같이 둔다. gui_server 와 같은 이유로 표준 라이브러리만 쓴다.

저장 구조 (nas_labels/):
  users.json                  계정. 비밀번호는 PBKDF2 해시만 둔다. git 제외.
  sessions.json               로그인 세션. 토큰의 sha256 만 둔다. git 제외.
  <dataset_id>/history.jsonl  모든 생성/수정/삭제 기록 - 정본(append-only)
  <dataset_id>/labels.json    현재 라벨. test_label_333_diff3.json 과 같은
                              양식이라 evaluate_labels.py 에 그대로 넣을 수 있다.
  <dataset_id>/assignments.jsonl  청크 분배 기록 - 정본(append-only)
  <dataset_id>/assignments.json   라벨러별 현재 할당 청크(보기용 결과물)

여러 라벨러가 써도 파일은 데이터셋당 한 벌이다. 라벨러는 각자 PC 의
브라우저로 이 서버(gui_server.py --host 0.0.0.0)에 접속할 뿐이고, 쓰는 것은
이 서버 프로세스 하나다. 다른 PC 에서 gui_server.py 를 따로 띄우면 그
PC 에 별도 파일이 생기므로 그렇게 쓰지 않는다.

분배: 관리자가 청크(약 100클립) 단위로 라벨러에게 나눠 준다. 라벨러의
[다음 미라벨]은 자기 할당분 안에서만 움직인다. 강제는 아니라서 남의
클립도 열어 고칠 수 있다(경고만 보인다).

정본이 history.jsonl 인 이유:
  "누가 언제 어떻게 고쳤나"는 이력에만 있고, 라벨 파일과 이력을 따로 쓰면
  둘 사이에서 프로세스가 죽었을 때 서로 어긋난다. 이력을 먼저 쓰고(fsync)
  라벨 파일은 이력을 다시 읽어 만든 결과물로 둔다. 서버를 띄울 때 이력으로
  라벨을 재구성해서 labels.json 과 다르면 labels.json 을 백업한 뒤 고쳐 쓴다
  (손으로 labels.json 을 고쳐도 반영되지 않는다 - 고칠 것은 GUI 로 고친다).

계정 관리(CLI, 첫 관리자 계정은 이걸로 만든다):
  python3 nas_labeling.py useradd <id> [--name 표시이름] [--admin]
  python3 nas_labeling.py passwd <id>
  python3 nas_labeling.py userdel <id>
  python3 nas_labeling.py users
기존 라벨 파일 가져오기(가져온 사람/시각이 이력에 남는다):
  python3 nas_labeling.py import <dataset_id> <label.json> --user <id>
"""
from __future__ import annotations

import argparse
import bisect
import getpass
import hashlib
import hmac
import http.cookies
import json
import os
import re
import secrets
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS_JSON = Path(os.environ.get("NAS_DATASETS_JSON") or ROOT / "nas_datasets.json")
# 저장 위치. 테스트 때는 환경변수로 다른 곳을 가리켜 실제 라벨을 건드리지 않는다.
STORE = Path(os.environ.get("NAS_LABEL_STORE") or ROOT / "nas_labels")

LOCAL_ADDRS = ("127.0.0.1", "::1", "::ffff:127.0.0.1")
COOKIE = "nl_session"
SESSION_DAYS = 14
PBKDF2_ITER = 200_000
USER_RE = re.compile(r"^[A-Za-z0-9_.-]{2,32}$")

# "작업 중" 표시 유지 시간. 브라우저가 30초마다 갱신하므로 창을 닫거나
# 네트워크가 끊기면 이 시간 뒤에 저절로 풀린다.
CLAIM_TTL = 90

WEATHER_AXES = ("illumination", "precipitation", "road_surface",
                "atmospheric_obscurants")
RSF = ("rain", "snow", "fog")
NOTE_MAX = 4000


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _chunk_no(path) -> int:
    m = re.search(r"chunk_(\d+)", Path(path).name)
    return int(m.group(1)) if m else -1


def _atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Conflict(Exception):
    """다른 사람이 먼저 고쳐서 기준 rev 가 맞지 않는다."""

    def __init__(self, info: dict):
        super().__init__("conflict")
        self.info = info


class AuthError(Exception):
    pass


class AssignConflict(Exception):
    """이미 다른 라벨러에게 할당된 청크를 덮어쓰려 했다."""

    def __init__(self, info: dict):
        super().__init__("assign conflict")
        self.info = info


def parse_chunks(spec) -> list[int]:
    """'0-9, 15, 20-29' -> [0..9, 15, 20..29]. 리스트를 받으면 그대로 정수화."""
    if isinstance(spec, (list, tuple)):
        return sorted({int(x) for x in spec})
    out = set()
    for part in re.split(r"[,\s]+", str(spec or "").strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ValueError(f"청크 범위를 읽을 수 없습니다: {part!r} (예: 0-9, 15, 20-29)")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if b < a:
            a, b = b, a
        if b - a > 100_000:
            raise ValueError(f"범위가 너무 넓습니다: {part}")
        out.update(range(a, b + 1))
    if not out:
        raise ValueError("청크를 하나 이상 지정하세요")
    return sorted(out)


def fmt_chunks(chunks) -> str:
    """[0,1,2,5,7,8] -> '0-2, 5, 7-8'"""
    cs = sorted(chunks)
    out, i = [], 0
    while i < len(cs):
        j = i
        while j + 1 < len(cs) and cs[j + 1] == cs[j] + 1:
            j += 1
        out.append(str(cs[i]) if i == j else f"{cs[i]}-{cs[j]}")
        i = j + 1
    return ", ".join(out)


# ---------------------------------------------------------------------------
# 계정 / 세션
# ---------------------------------------------------------------------------
class Auth:
    """계정은 파일에서 매번 읽는다 - CLI 로 추가/삭제한 것이 재시작 없이 반영된다."""

    def __init__(self, store: Path):
        self.users_path = store / "users.json"
        self.sessions_path = store / "sessions.json"
        self.lock = threading.Lock()
        store.mkdir(parents=True, exist_ok=True)

    # --- 계정 ---
    def users(self) -> dict:
        try:
            return json.loads(self.users_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}

    def _save_users(self, users: dict):
        _atomic_write(self.users_path, json.dumps(users, ensure_ascii=False, indent=2))
        os.chmod(self.users_path, 0o600)

    @staticmethod
    def _hash(pw: str, salt: bytes) -> str:
        return hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt,
                                   PBKDF2_ITER).hex()

    def set_user(self, uid: str, pw: str, name: str | None = None,
                 admin: bool | None = None, create: bool = False):
        if not USER_RE.match(uid or ""):
            raise ValueError("아이디는 영문/숫자/_.- 2~32자입니다")
        if len(pw or "") < 4:
            raise ValueError("비밀번호는 4자 이상이어야 합니다")
        with self.lock:
            users = self.users()
            if create and uid in users:
                raise ValueError(f"이미 있는 아이디: {uid}")
            if not create and uid not in users:
                raise ValueError(f"없는 아이디: {uid}")
            u = users.get(uid, {"created": now_iso()})
            salt = secrets.token_bytes(16)
            u.update(salt=salt.hex(), iter=PBKDF2_ITER, hash=self._hash(pw, salt))
            if name is not None:
                u["name"] = name
            u.setdefault("name", uid)
            if admin is not None:
                u["admin"] = bool(admin)
            u.setdefault("admin", False)
            users[uid] = u
            self._save_users(users)

    def del_user(self, uid: str):
        with self.lock:
            users = self.users()
            if uid not in users:
                raise ValueError(f"없는 아이디: {uid}")
            del users[uid]
            self._save_users(users)

    def check(self, uid: str, pw: str) -> bool:
        u = self.users().get(uid)
        if not u:
            # 없는 아이디도 같은 시간이 걸리게 - 응답 시간으로 아이디 유무를 못 알게.
            self._hash(pw or "", b"x" * 16)
            return False
        got = hashlib.pbkdf2_hmac("sha256", (pw or "").encode("utf-8"),
                                  bytes.fromhex(u["salt"]), u.get("iter", PBKDF2_ITER)).hex()
        return hmac.compare_digest(got, u["hash"])

    @staticmethod
    def public(uid: str, u: dict) -> dict:
        return {"user": uid, "name": u.get("name", uid), "admin": bool(u.get("admin"))}

    # --- 세션 ---
    def _sessions(self) -> dict:
        try:
            return json.loads(self.sessions_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_sessions(self, s: dict):
        now = time.time()
        s = {k: v for k, v in s.items() if v.get("exp", 0) > now}
        _atomic_write(self.sessions_path, json.dumps(s))
        os.chmod(self.sessions_path, 0o600)

    @staticmethod
    def _tk(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def login(self, uid: str, pw: str) -> str | None:
        if not self.check(uid, pw):
            time.sleep(0.5)          # 비밀번호 대입을 느리게
            return None
        token = secrets.token_urlsafe(32)
        with self.lock:
            s = self._sessions()
            s[self._tk(token)] = {"user": uid, "created": now_iso(),
                                  "exp": time.time() + SESSION_DAYS * 86400}
            self._save_sessions(s)
        return token

    def logout(self, token: str):
        with self.lock:
            s = self._sessions()
            if s.pop(self._tk(token), None) is not None:
                self._save_sessions(s)

    def user_for(self, token: str | None) -> dict | None:
        if not token:
            return None
        sess = self._sessions().get(self._tk(token))
        if not sess or sess.get("exp", 0) < time.time():
            return None
        u = self.users().get(sess["user"])
        return self.public(sess["user"], u) if u else None   # 삭제된 계정은 무효


# ---------------------------------------------------------------------------
# 라벨 양식
# ---------------------------------------------------------------------------
class Schema:
    """test_label_333_diff3.json 의 클립 한 건 양식.

    {"categories": {이름: 1~4}, "<묶음>": 0~4 ..., "rain_snow_fog": [...],
     "weather": {4축: 0~4}, "note": str}
    묶음 점수는 보통 그 묶음 카테고리 점수의 최댓값이지만 기존 라벨에도
    다르게 매긴 건이 있어(333건 중 15건) 강제하지 않는다 - 화면이 최댓값을
    채워 주고, 사람이 바꿀 수 있다.

    매길 수 있는 점수는 constrained_tier 의 기준표 눈금이다(Driving environment
    는 0, 1, 3). 눈금이 바뀌기 전에 저장한 값은 그 클립에서 그대로 둘 때만
    통과시킨다 - 그래야 note 만 고치는 저장이 옛 값 때문에 막히지 않는다.
    """

    def __init__(self, taxonomy_path: Path):
        import constrained_tier as CT
        scene = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        self.file = taxonomy_path.name
        self.groups: dict[str, list[str]] = {}
        for scen in scene.get("special", {}).get("scenarios", []):
            self.groups[scen["name"]] = [c["name"] for c in scen.get("categories", [])]
        self.cat_group = {c: g for g, cs in self.groups.items() for c in cs}
        self.group_values = {g: list(CT.group_values(g)) if g in CT.GROUP_RUBRICS
                             else [0, 1, 2, 3, 4] for g in self.groups}
        # 카테고리는 붙어 있으면 1 이상이다. 눈금의 0 은 "없음" 칸이 대신한다.
        self.cat_values = {
            c: (sorted(v for v in CT.rubric_of(c if c in CT.CATEGORY_RUBRICS else g)[1] if v)
                if c in CT.CATEGORY_RUBRICS or g in CT.GROUP_RUBRICS else [1, 2, 3, 4])
            for c, g in self.cat_group.items()}

    def normalize(self, raw: dict, prev: dict | None) -> dict:
        if not isinstance(raw, dict):
            raise ValueError("라벨 형식이 아닙니다")

        def iv(v, lo, hi, what):
            try:
                x = int(v)
            except (TypeError, ValueError):
                raise ValueError(f"{what}: 정수가 아닙니다 ({v!r})")
            if x != v and not (isinstance(v, str) and v.strip() == str(x)):
                raise ValueError(f"{what}: 정수가 아닙니다 ({v!r})")
            if not lo <= x <= hi:
                raise ValueError(f"{what}: {lo}~{hi} 범위 밖 ({x})")
            return x

        def score(v, lo, allowed, old, what):
            x = iv(v, lo, 4, what)
            if x not in allowed and x != old:
                raise ValueError(f"{what}: {x}점은 기준표에 없습니다 "
                                 f"(가능: {', '.join(map(str, allowed))})")
            return x

        cr = raw.get("categories") or {}
        if not isinstance(cr, dict):
            raise ValueError("categories 는 {이름: 점수} 여야 합니다")
        # 분류 체계에서 빠진 옛 카테고리는 원래 붙어 있던 클립에서만 유지를 허용한다.
        known = set(self.cat_group) | set((prev or {}).get("categories", {}))
        bad = [k for k in cr if k not in known]
        if bad:
            raise ValueError("알 수 없는 카테고리: " + ", ".join(bad))
        order = [c for cs in self.groups.values() for c in cs]
        order += [c for c in cr if c not in order]
        was = prev or {}
        cats = {c: score(cr[c], 1, self.cat_values.get(c, (1, 2, 3, 4)),
                         was.get("categories", {}).get(c), f"categories.{c}")
                for c in order if c in cr}

        out = {"categories": cats}
        for g in self.groups:
            out[g] = score(raw.get(g, 0), 0, self.group_values[g], was.get(g), g)

        rsf = raw.get("rain_snow_fog") or []
        if not isinstance(rsf, list) or any(x not in RSF for x in rsf):
            raise ValueError("rain_snow_fog 는 rain/snow/fog 중에서 고릅니다")
        out["rain_snow_fog"] = [x for x in RSF if x in rsf]

        w = raw.get("weather") or {}
        if not isinstance(w, dict):
            raise ValueError("weather 형식이 아닙니다")
        out["weather"] = {k: iv(w.get(k, 0), 0, 4, f"weather.{k}") for k in WEATHER_AXES}

        note = raw.get("note") or ""
        if not isinstance(note, str):
            raise ValueError("note 는 문자열입니다")
        if len(note) > NOTE_MAX:
            raise ValueError(f"note 가 너무 깁니다({len(note)}자, 최대 {NOTE_MAX})")
        # 공백도 손대지 않는다 - 가져온 라벨이 원본과 한 글자라도 달라지면 안 된다.
        out["note"] = note
        return out

    @staticmethod
    def flat(lab: dict | None) -> dict:
        """비교용 평탄화. 없는 카테고리는 키가 없다(= 해제)."""
        if not lab:
            return {}
        f = {}
        for k, v in lab.items():
            if k == "categories":
                f.update({f"categories.{c}": s for c, s in v.items()})
            elif k == "weather":
                f.update({f"weather.{a}": s for a, s in v.items()})
            elif k == "rain_snow_fog":
                f[k] = ", ".join(v)
            else:
                f[k] = v
        return f

    @classmethod
    def diff(cls, before: dict | None, after: dict | None) -> list[dict]:
        a, b = cls.flat(before), cls.flat(after)
        keys = list(a) + [k for k in b if k not in a]
        return [{"field": k, "from": a.get(k), "to": b.get(k)}
                for k in keys if a.get(k) != b.get(k)]


# ---------------------------------------------------------------------------
# 데이터셋 하나 = 클립 목록 + 라벨 저장소
# ---------------------------------------------------------------------------
class Dataset:
    def __init__(self, cfg: dict, mounts: dict, schema: Schema, store: Path,
                 shared_sources=()):
        self.id = cfg["id"]
        if not re.match(r"^[A-Za-z0-9_.-]+$", self.id):
            raise ValueError(f"데이터셋 id 에 쓸 수 없는 문자: {self.id}")
        self.name = cfg.get("name", self.id)
        self.nas_path = cfg["nas_path"]
        self.view = cfg.get("view", "camera_front_wide_120fov")
        self.local = self._resolve(self.nas_path, mounts)
        self.camera_dir = self.local / "camera"
        self.schema = schema
        self.dir = store / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hist_path = self.dir / "history.jsonl"
        self.labels_path = self.dir / "labels.json"
        # 청크 단위 분배. 이력과 같은 이유로 append-only 로그가 정본이고
        # assignments.json 은 사람이 보기 위한 결과물이다.
        self.assign_log_path = self.dir / "assignments.jsonl"
        self.assign_path = self.dir / "assignments.json"
        self.assign: dict[int, str] = {}           # chunk -> user
        self.assign_events: list[dict] = []
        self.assign_ver = 0

        self.lock = threading.RLock()
        self.labels: dict[str, dict] = {}
        self.audit: dict[str, dict] = {}
        self.hist: dict[str, list] = {}
        self.events: list[tuple[int, str]] = []    # (seq, uuid) - 폴링용
        self.seq = 0
        self.claims: dict[str, dict] = {}

        # 클립 목록(인덱스)은 NAS 를 훑어야 해서 느릴 수 있다 - 백그라운드로 만든다.
        self._shared = [s for s in shared_sources if s is not None]
        self.source = None
        self.order: list[str] | None = None
        self.chunk: dict[str, int] = {}
        self.chunk_clips: dict[int, list[str]] = {}
        self.pos: dict[str, int] = {}
        self.index_state = "idle"          # idle | building | ready | error
        self.index_error = ""
        self._index_lock = threading.Lock()
        self._load_history()
        self._load_assign()

    @staticmethod
    def _resolve(nas_path: str, mounts: dict) -> Path:
        from clip_source import resolve_nas_path
        return resolve_nas_path(nas_path, mounts)

    # --- 클립 목록 ---
    def start_index(self):
        with self._index_lock:
            if self.index_state in ("building", "ready"):
                return
            self.index_state = "building"
        threading.Thread(target=self._build_index, daemon=True,
                         name=f"nl-index-{self.id}").start()

    def _build_index(self):
        try:
            if not self.camera_dir.is_dir():
                raise FileNotFoundError(
                    f"{self.camera_dir} 가 없습니다 (NAS 마운트 확인: {self.nas_path})")
            from clip_source import ZipSource, LocalSource
            src = None
            for s in self._shared:
                if Path(getattr(s, "camera_dir", "")).resolve() == self.camera_dir.resolve():
                    src = s
            if src is None:
                src = ZipSource(self.camera_dir)
            if isinstance(src, LocalSource):
                uu = src.uuids(self.view)
                chunk = {u: -1 for u in uu}
            else:
                pre = self.view + "/"
                chunk = {k[len(pre):]: _chunk_no(v[0])
                         for k, v in src.index.items() if k.startswith(pre)}
            order = sorted(chunk, key=lambda u: (chunk[u], u))
            cc: dict[int, list[str]] = {}
            for u in order:
                cc.setdefault(chunk[u], []).append(u)
            with self.lock:
                self.source, self.chunk, self.order = src, chunk, order
                self.chunk_clips = cc
                self.pos = {u: i for i, u in enumerate(order)}
                self.index_state = "ready"
            print(f"[nas_label] {self.id}: 클립 {len(order):,}개", flush=True)
        except Exception as e:                       # noqa: BLE001
            self.index_state, self.index_error = "error", f"{type(e).__name__}: {e}"
            print(f"[nas_label] {self.id}: 인덱스 실패 - {self.index_error}", flush=True)

    def require_index(self):
        if self.index_state != "ready":
            self.start_index()
            raise RuntimeError(
                "클립 목록 준비 중입니다" if self.index_state == "building"
                else f"클립 목록을 만들지 못했습니다: {self.index_error}")

    # --- 이력 ---
    def _apply(self, ev: dict):
        u = ev["uuid"]
        a = self.audit.setdefault(u, {"rev": 0, "users": []})
        a["rev"] = ev["rev"]
        if ev["action"] == "delete":
            self.labels.pop(u, None)
        else:
            self.labels[u] = ev["after"]
            if "created_at" not in a or a.get("deleted"):
                a["created_by"], a["created_at"] = ev["user"], ev["ts"]
        a["deleted"] = ev["action"] == "delete"
        a["updated_by"], a["updated_at"] = ev["user"], ev["ts"]
        if ev["user"] not in a["users"]:
            a["users"].append(ev["user"])
        self.hist.setdefault(u, []).append(ev)
        self.seq = ev["seq"]
        self.events.append((ev["seq"], u))

    def _hist_stamp(self):
        try:
            st = self.hist_path.stat()
            return (st.st_size, st.st_mtime_ns)
        except FileNotFoundError:
            return (0, 0)

    def _sync(self):
        """다른 프로세스(import CLI 등)가 이력에 쓴 것을 따라잡는다.

        메모리 상태로 seq/rev 를 매기므로, 파일이 바뀐 줄 모르고 쓰면 번호가
        겹친다. 쓰기 직전과 폴링 때 크기/mtime 을 보고 바뀌었으면 다시 읽는다.
        """
        if self._hist_stamp() == self._stamp:
            return
        self.labels, self.audit, self.hist, self.events, self.seq = {}, {}, {}, [], 0
        self._load_history()

    def _load_history(self):
        if self.hist_path.exists():
            with open(self.hist_path, encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._apply(json.loads(line))
                    except (json.JSONDecodeError, KeyError) as e:
                        # 마지막 줄이 쓰다 끊긴 경우 - 그 건은 저장되지 않은 것으로 본다.
                        print(f"[nas_label] {self.hist_path}:{n} 읽기 실패({e}) - 건너뜀",
                              flush=True)
        want = self._labels_doc()
        try:
            have = json.loads(self.labels_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            have = None
        if have is None or have.get("clips") != want["clips"]:
            if have is not None:
                bak = self.labels_path.with_name(
                    f"labels.json.mismatch-{datetime.now():%Y%m%d_%H%M%S}")
                shutil.copy2(self.labels_path, bak)
                print(f"[nas_label] {self.labels_path} 가 이력과 달라 다시 씁니다 "
                      f"(기존 파일: {bak.name})", flush=True)
            self._write_labels()
        self._stamp = self._hist_stamp()

    def _labels_doc(self) -> dict:
        who = sorted({a["updated_by"] for u, a in self.audit.items() if u in self.labels})
        last = max((a["updated_at"] for a in self.audit.values()), default="")
        return {"_meta": {"labeled_by": ", ".join(who), "date": last[:10],
                          "source_run": "", "guide": "",
                          "dataset": self.nas_path, "view": self.view,
                          "taxonomy": self.schema.file,
                          "generated_from": "history.jsonl (nas_labeling.py)"},
                "clips": self.labels}

    def _write_labels(self):
        _atomic_write(self.labels_path,
                      json.dumps(self._labels_doc(), ensure_ascii=False, indent=2))

    def _record(self, uuid: str, user: str, action: str, before, after,
                ts: str | None = None, extra: dict | None = None) -> dict:
        ev = {"seq": self.seq + 1, "ts": ts or now_iso(), "user": user,
              "uuid": uuid, "action": action,
              "rev": self.audit.get(uuid, {}).get("rev", 0) + 1,
              "before": before, "after": after,
              "changes": Schema.diff(before, after)}
        if extra:
            ev.update(extra)
        line = json.dumps(ev, ensure_ascii=False) + "\n"
        with open(self.hist_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        self._apply(ev)
        self._stamp = self._hist_stamp()
        return ev

    # --- 작업 중 표시 ---
    def active_claims(self) -> dict[str, str]:
        now = time.time()
        with self.lock:
            for u in [u for u, c in self.claims.items() if c["until"] < now]:
                del self.claims[u]
            return {u: c["user"] for u, c in self.claims.items()}

    def claim(self, uuid: str, user: str):
        with self.lock:
            # 한 사람은 한 번에 한 클립만 잡는다 - 다른 클립으로 넘어가면 앞 것은 푼다.
            for u in [u for u, c in self.claims.items() if c["user"] == user and u != uuid]:
                del self.claims[u]
            self.claims[uuid] = {"user": user, "until": time.time() + CLAIM_TTL}

    def release(self, uuid: str, user: str):
        with self.lock:
            if self.claims.get(uuid, {}).get("user") == user:
                del self.claims[uuid]

    # --- 분배 (청크 -> 라벨러) ---
    def _load_assign(self):
        if not self.assign_log_path.exists():
            return
        with open(self.assign_log_path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self._apply_assign(json.loads(line))
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"[nas_label] {self.assign_log_path}:{n} 읽기 실패({e}) - 건너뜀",
                          flush=True)

    def _apply_assign(self, ev: dict):
        chunks = parse_chunks(ev["chunks"]) if ev.get("chunks") else []
        if ev.get("replace") and ev.get("user"):
            # 이 라벨러의 할당 전체를 chunks 로 바꾼다(기존 것은 해제).
            for c in [c for c, x in self.assign.items() if x == ev["user"]]:
                del self.assign[c]
        for c in chunks:
            if ev.get("user"):
                self.assign[c] = ev["user"]
            else:
                self.assign.pop(c, None)
        self.assign_events.append(ev)
        self.assign_ver += 1

    def _record_assign(self, by: str, user: str | None, chunks: list[int],
                       replaced: dict | None = None, mode: str = "manual",
                       replace: bool = False, removed: list[int] | None = None):
        ev = {"ts": now_iso(), "by": by, "user": user or None,
              "action": "assign" if user else "unassign", "mode": mode,
              "chunks": fmt_chunks(chunks), "n": len(chunks)}
        if replace:
            ev["replace"] = True
        if removed:
            ev["removed"] = fmt_chunks(removed)
        if replaced:
            ev["replaced"] = replaced
        with open(self.assign_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._apply_assign(ev)
        by_user: dict[str, list[int]] = {}
        for c, x in self.assign.items():
            by_user.setdefault(x, []).append(c)
        _atomic_write(self.assign_path, json.dumps(
            {"_meta": {"generated_from": self.assign_log_path.name, "updated_at": ev["ts"]},
             "by_user": {x: fmt_chunks(cs) for x, cs in sorted(by_user.items())}},
            ensure_ascii=False, indent=2))
        return ev

    def assignee(self, u: str) -> str | None:
        return self.assign.get(self.chunk.get(u, -2))

    def assign_chunks(self, spec, user: str | None, by: str, force: bool = False) -> dict:
        """user 의 할당을 spec 청크로 바꾼다(기존 할당 중 spec 에 없는 것은 해제).

        user 가 없으면 spec 청크를 할당 해제한다. 남에게 할당된 청크를 가져오는
        것은 force 일 때만 한다.
        """
        self.require_index()
        chunks = parse_chunks(spec)
        bad = [c for c in chunks if c not in self.chunk_clips]
        if bad:
            raise ValueError(f"데이터셋에 없는 청크 {len(bad)}개: {fmt_chunks(bad)} "
                             f"(있는 범위: {min(self.chunk_clips)}-{max(self.chunk_clips)})")
        with self.lock:
            if not user:
                change = [c for c in chunks if c in self.assign]
                if not change:
                    return {"changed": 0, "added": 0, "removed": 0}
                was: dict[str, int] = {}
                for c in change:
                    was[self.assign[c]] = was.get(self.assign[c], 0) + 1
                self._record_assign(by, None, change, was)
                return {"changed": len(change), "added": 0, "removed": len(change)}
            taken: dict[str, list[int]] = {}
            for c in chunks:
                x = self.assign.get(c)
                if x and x != user:
                    taken.setdefault(x, []).append(c)
            if taken and not force:
                raise AssignConflict({x: fmt_chunks(cs) for x, cs in taken.items()})
            old = {c for c, x in self.assign.items() if x == user}
            new = set(chunks)
            if old == new:
                return {"changed": 0, "added": 0, "removed": 0}
            added, removed = sorted(new - old), sorted(old - new)
            self._record_assign(by, user, chunks,
                                {x: len(cs) for x, cs in taken.items()} or None,
                                replace=True, removed=removed)
            return {"changed": len(added) + len(removed), "added": len(added),
                    "removed": len(removed), "n_chunks": len(chunks)}

    def auto_assign(self, users: list[str], per_user: int, by: str) -> dict:
        """선택한 라벨러들의 할당을 각각 새 청크 per_user 개로 바꾼다.

        후보는 미라벨 클립이 남은 청크 중 아무에게도 할당되지 않았거나 이번에
        고른 라벨러들이 갖고 있던 것(바뀌면서 풀리므로)을 앞 번호부터.
        """
        self.require_index()
        users = list(dict.fromkeys(users))
        if not users:
            raise ValueError("분배할 라벨러를 고르세요")
        if not 1 <= per_user <= 10_000:
            raise ValueError("1인당 청크 수는 1 이상이어야 합니다")
        with self.lock:
            sel = set(users)
            free = [c for c in sorted(self.chunk_clips)
                    if self.assign.get(c) in (None, *sel)
                    and any(u not in self.labels for u in self.chunk_clips[c])]
            got, i = {}, 0
            for user in users:
                take = free[i:i + per_user]
                i += len(take)
                old = {c for c, x in self.assign.items() if x == user}
                if set(take) != old:
                    self._record_assign(by, user, take, mode="auto", replace=True,
                                        removed=sorted(old - set(take)))
                got[user] = fmt_chunks(take)
            return {"assigned": got, "left": max(0, len(free) - i)}

    def assign_summary(self) -> dict:
        self.require_index()
        with self.lock:
            per: dict[str, list[int]] = {}
            for c, x in self.assign.items():
                per.setdefault(x, []).append(c)
            out = {}
            for x, cs in per.items():
                clips = sum(len(self.chunk_clips.get(c, [])) for c in cs)
                done = sum(1 for c in cs for u in self.chunk_clips.get(c, []) if u in self.labels)
                out[x] = {"chunks": fmt_chunks(cs), "n_chunks": len(cs),
                          "clips": clips, "labeled": done}
            free = [c for c in self.chunk_clips if c not in self.assign]
            return {"by_user": out, "n_chunks": len(self.chunk_clips),
                    "clips": len(self.order),
                    "unassigned": {"n_chunks": len(free),
                                   "clips": sum(len(self.chunk_clips[c]) for c in free)},
                    "log": list(reversed(self.assign_events[-30:])),
                    "assign_ver": self.assign_ver}

    def _mine(self, user: str) -> tuple[int, int]:
        cs = [c for c, x in self.assign.items() if x == user]
        total = sum(len(self.chunk_clips.get(c, [])) for c in cs)
        done = sum(1 for c in cs for u in self.chunk_clips.get(c, []) if u in self.labels)
        return total, done

    # --- 조회 ---
    def row(self, u: str, claims: dict) -> dict:
        lab = self.labels.get(u)
        a = self.audit.get(u, {})
        r = {"uuid": u, "chunk": self.chunk.get(u), "labeled": lab is not None,
             "rev": a.get("rev", 0), "claimed_by": claims.get(u),
             "assignee": self.assignee(u)}
        if lab is not None:
            r.update(created_by=a.get("created_by"), created_at=a.get("created_at"),
                     updated_by=a.get("updated_by"), updated_at=a.get("updated_at"),
                     users=a.get("users", []), categories=lab.get("categories", {}),
                     groups={g: lab.get(g) for g in self.schema.groups})
        return r

    def counts(self, user: str, claims: dict) -> dict:
        total = len(self.order or [])
        labeled = sum(1 for u in self.labels if u in self.pos)
        by = {}
        for u in self.labels:
            for x in self.audit[u].get("users", []):
                by[x] = by.get(x, 0) + 1
        at, ad = self._mine(user)
        return {"total": total, "labeled": labeled, "unlabeled": total - labeled,
                "claimed": len(claims),
                "mine": by.get(user, 0), "labelers": by,
                "assigned": at, "assigned_labeled": ad}

    def list(self, q: dict, user: str) -> dict:
        self.require_index()
        claims = self.active_claims()
        status = q.get("status", "all")
        needle = (q.get("q") or "").strip().lower()
        labeler = q.get("labeler") or ""
        sort = q.get("sort", "order")
        with self.lock:
            if status == "unlabeled":
                cand = [u for u in self.order if u not in self.labels]
            elif status == "claimed":
                cand = [u for u in self.order if u in claims]
            elif status in ("labeled", "mine"):
                who = user if status == "mine" else labeler
                cand = [u for u in self.order if u in self.labels
                        and (not who or who in self.audit[u].get("users", []))]
            else:
                cand = self.order
            who = q.get("assignee") or ""
            if who:
                want = None if who == "@none" else (user if who == "@me" else who)
                cand = [u for u in cand if self.assign.get(self.chunk.get(u, -2)) == want]
            if needle:
                cand = [u for u in cand if needle in u]
            if q.get("chunk", "") != "":
                try:
                    c = int(q["chunk"])
                    cand = [u for u in cand if self.chunk.get(u) == c]
                except ValueError:
                    pass
            if sort == "recent":
                cand = sorted(cand, key=lambda u: self.audit.get(u, {}).get("updated_at", ""),
                              reverse=True)
            total = len(cand)
            off = max(0, int(q.get("offset", 0) or 0))
            lim = min(500, max(1, int(q.get("limit", 100) or 100)))
            rows = [dict(self.row(u, claims), idx=self.pos[u]) for u in cand[off:off + lim]]
            return {"total": total, "offset": off, "rows": rows, "seq": self.seq,
                    "counts": self.counts(user, claims)}

    def detail(self, uuid: str) -> dict:
        self.require_index()
        if uuid not in self.pos:
            raise KeyError(uuid)
        claims = self.active_claims()
        with self.lock:
            return {"row": self.row(uuid, claims), "label": self.labels.get(uuid),
                    "idx": self.pos[uuid], "history": list(reversed(self.hist.get(uuid, [])))}

    def next_clip(self, after: str | None, user: str, step: int = 1,
                  only_unlabeled: bool = True, scope: str = "any") -> str | None:
        """after 다음(이전)의 미라벨 클립. 남이 작업 중인 클립은 건너뛴다.

        scope: mine = 내게 할당된 청크만, free = 내 것 + 미할당 청크(남의 할당은
        건너뜀), any = 전부.
        """
        self.require_index()
        claims = self.active_claims()
        if scope == "mine":
            cs = sorted(c for c, x in self.assign.items() if x == user)
            seq = [u for c in cs for u in self.chunk_clips.get(c, [])]
        else:
            seq = self.order
        n = len(seq)
        if not n:
            return None
        if scope == "mine":
            # 목록 순서(order)상 after 다음에 오는 내 클립부터.
            p = self.pos.get(after, -1)
            ps = [self.pos[u] for u in seq]
            i = (bisect.bisect_right(ps, p) - 1) if step > 0 else bisect.bisect_left(ps, p)
        else:
            i = self.pos.get(after, -1 if step > 0 else 0)
        for k in range(1, n + 1):
            u = seq[(i + step * k) % n]
            if not only_unlabeled:
                return u
            if u in self.labels:
                continue
            if claims.get(u) not in (None, user):
                continue
            if scope == "free" and self.assign.get(self.chunk.get(u, -2)) not in (None, user):
                continue
            return u
        return None

    def poll(self, since: int, user: str) -> dict:
        claims = self.active_claims()
        with self.lock:
            self._sync()
            changed, seen = [], set()
            for s, u in reversed(self.events):
                if s <= since:
                    break
                if u not in seen:
                    seen.add(u)
                    changed.append(self.row(u, claims))
            return {"seq": self.seq, "changed": changed, "claims": claims,
                    "counts": self.counts(user, claims) if self.order is not None else None,
                    "index_state": self.index_state, "assign_ver": self.assign_ver}

    # --- 쓰기 ---
    def save(self, uuid: str, raw: dict, user: str, base_rev) -> dict:
        self.require_index()
        if uuid not in self.pos:
            raise KeyError(uuid)
        with self.lock:
            self._sync()
            prev = self.labels.get(uuid)
            rev = self.audit.get(uuid, {}).get("rev", 0)
            if base_rev is not None and int(base_rev) != rev:
                a = self.audit.get(uuid, {})
                raise Conflict({"rev": rev, "by": a.get("updated_by"),
                                "at": a.get("updated_at")})
            new = self.schema.normalize(raw, prev)
            if prev is not None and not Schema.diff(prev, new):
                return {"unchanged": True, "rev": rev, "label": prev}
            ev = self._record(uuid, user, "create" if prev is None else "update", prev, new)
            self._write_labels()
            return {"saved": True, "rev": ev["rev"], "label": new, "event": ev}

    def delete(self, uuid: str, user: str, base_rev) -> dict:
        with self.lock:
            self._sync()
            prev = self.labels.get(uuid)
            if prev is None:
                raise KeyError(uuid)
            rev = self.audit.get(uuid, {}).get("rev", 0)
            if base_rev is not None and int(base_rev) != rev:
                a = self.audit.get(uuid, {})
                raise Conflict({"rev": rev, "by": a.get("updated_by"),
                                "at": a.get("updated_at")})
            ev = self._record(uuid, user, "delete", prev, None)
            self._write_labels()
            return {"deleted": True, "rev": ev["rev"]}

    def info(self) -> dict:
        return {"id": self.id, "name": self.name, "nas_path": self.nas_path,
                "local_path": str(self.local), "view": self.view,
                "labels_file": (str(self.labels_path.relative_to(ROOT))
                                if self.labels_path.is_relative_to(ROOT)
                                else str(self.labels_path)),
                "index_state": self.index_state, "index_error": self.index_error,
                "clips": len(self.order) if self.order is not None else None}


# ---------------------------------------------------------------------------
# 서비스 (gui_server 가 쓰는 입구)
# ---------------------------------------------------------------------------
def load_config() -> dict:
    return json.loads(DATASETS_JSON.read_text(encoding="utf-8"))


class LabelingService:
    def __init__(self, shared_sources=(), store: Path = STORE, warm: bool = True):
        cfg = load_config()
        self.schema = Schema(ROOT / cfg.get("taxonomy", "scene_category_3.2.json"))
        self.auth = Auth(store)
        self.datasets: dict[str, Dataset] = {}
        for d in cfg.get("datasets", []):
            ds = Dataset(d, cfg.get("mounts", {}), self.schema, store, shared_sources)
            self.datasets[ds.id] = ds
            if warm:
                ds.start_index()

    def schema_info(self) -> dict:
        import constrained_tier as CT
        try:
            from prompts import WEATHER_SCALES, WEATHER_SCALES_KOR
        except ImportError:
            WEATHER_SCALES, WEATHER_SCALES_KOR = {}, {}
        rub = {g: {"title": GR[0], "scale": GR[1]} for g, GR in CT.GROUP_RUBRICS.items()}
        cat_rub = {c: {"group": v[0], "title": v[1], "scale": v[2]}
                   for c, v in CT.CATEGORY_RUBRICS.items()}
        # [평가 기준] 블록. 영어 표는 모델 프롬프트에 들어가는 것과 같은 표이고,
        # 한국어 표는 사람이 읽을 번역이다.
        kor = {"Dynamic object": CT.DIFFICULTY_RUBRIC_FOR_DYNAMIC_OBJECT_KOR,
               "Driving environment": CT.DIFFICULTY_RUBRIC_FOR_DRIVING_ENVIRONMENT_KOR,
               "Unpaved road": CT.UNPAVED_RUBRIC_KOR}
        rubrics = [{"name": n, "title": v["title"], "en": v["scale"], "ko": kor[n]}
                   for n, v in (*rub.items(), *cat_rub.items())]
        return {"taxonomy": self.schema.file, "groups": self.schema.groups,
                "group_rubrics": rub, "category_rubrics": cat_rub, "rubrics": rubrics,
                "group_values": self.schema.group_values,
                "category_values": self.schema.cat_values,
                "weather_axes": list(WEATHER_AXES), "weather_scales": WEATHER_SCALES,
                "weather_scales_ko": WEATHER_SCALES_KOR,
                "rain_snow_fog": list(RSF), "tiers": CT.TIER_LABELS}

    # --- HTTP 입구. h 는 gui_server.Handler ---
    @staticmethod
    def _token(h) -> str | None:
        c = http.cookies.SimpleCookie()
        try:
            c.load(h.headers.get("Cookie") or "")
        except http.cookies.CookieError:
            return None
        m = c.get(COOKIE)
        return m.value if m else None

    @staticmethod
    def _is_local(h) -> bool:
        return h.client_address[0] in LOCAL_ADDRS

    def is_admin_request(self, h) -> bool:
        """사내망 접속자에게 다른 탭(추론 실행 등)을 열어 줄지 - 관리자 로그인만."""
        me = self.auth.user_for(self._token(h))
        return bool(me and me.get("admin"))

    def _ds(self, q) -> Dataset:
        ds = self.datasets.get(q.get("ds") or "")
        if ds is None:
            raise KeyError("데이터셋 없음: " + str(q.get("ds")))
        return ds

    def handle(self, h, method: str, p: str, q: dict, b: dict):
        ep = p[len("/api/nl/"):]
        if ep == "login" and method == "POST":
            token = self.auth.login((b.get("user") or "").strip(), b.get("password") or "")
            if not token:
                return h._err("아이디 또는 비밀번호가 맞지 않습니다", 401)
            me = self.auth.user_for(token)
            ck = (f"{COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; "
                  f"Max-Age={SESSION_DAYS * 86400}")
            return h._json({"me": me}, headers={"Set-Cookie": ck})
        token = self._token(h)
        me = self.auth.user_for(token)
        if ep == "me":
            return h._json({"me": me, "local": self._is_local(h)})
        if me is None:
            return h._err("로그인이 필요합니다", 401)
        if ep == "logout":
            self.auth.logout(token)
            return h._json({"ok": True}, headers={
                "Set-Cookie": f"{COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"})
        uid = me["user"]
        try:
            return self._route(h, method, ep, q, b, me, uid)
        except Conflict as e:
            return h._json({"error": "다른 사용자가 먼저 수정했습니다", "conflict": e.info}, 409)
        except AssignConflict as e:
            who = ", ".join(f"{x}: {cs}" for x, cs in e.info.items())
            return h._json({"error": f"이미 다른 라벨러에게 할당된 청크가 있습니다 ({who})",
                            "assign_conflict": e.info}, 409)
        except KeyError as e:
            return h._err(f"찾을 수 없음: {e.args[0] if e.args else ''}", 404)
        except RuntimeError as e:
            return h._json({"error": str(e), "indexing": True}, 503)
        except (ValueError, AuthError) as e:
            return h._err(str(e), 400)

    def _route(self, h, method, ep, q, b, me, uid):
        if method == "GET":
            if ep == "bootstrap":
                return h._json({"me": me, "schema": self.schema_info(),
                                "datasets": [d.info() for d in self.datasets.values()]})
            if ep == "clips":
                return h._json(self._ds(q).list(q, uid))
            if ep == "clip":
                return h._json(self._ds(q).detail(q.get("uuid", "")))
            if ep == "next":
                ds = self._ds(q)
                scope = q.get("scope") if q.get("scope") in ("mine", "free") else "any"
                u = ds.next_clip(q.get("after") or None, uid, -1 if q.get("dir") == "prev" else 1,
                                 q.get("any") != "1", scope)
                return h._json({"uuid": u, "idx": ds.pos.get(u) if u else None})
            if ep == "assignments":
                ds = self._ds(q)
                return h._json(dict(ds.assign_summary(), users=[
                    {"user": k, "name": v.get("name", k), "admin": bool(v.get("admin"))}
                    for k, v in sorted(self.auth.users().items())]))
            if ep == "poll":
                return h._json(self._ds(q).poll(int(q.get("since") or 0), uid))
            if ep == "video":
                ds = self._ds(q)
                ds.require_index()
                u = q.get("uuid", "")
                if u not in ds.pos:
                    return h._err("not found", 404)
                src = ds.source
                if not hasattr(src, "size_of"):          # 로컬 mp4
                    return h._send_file(Path(src.open_video(ds.view, u)))
                return h._stream(lambda: src.open_video(ds.view, u),
                                 src.size_of(ds.view, u), "video/mp4", None)
            if ep == "users" and me["admin"]:
                return h._json({"users": [self.auth.public(k, v)
                                          for k, v in sorted(self.auth.users().items())]})
            return h._err("unknown endpoint: " + ep, 404)

        if ep == "save":
            ds = self._ds(b)
            return h._json(ds.save(b.get("uuid", ""), b.get("label") or {}, uid, b.get("base_rev")))
        if ep == "delete":
            return h._json(self._ds(b).delete(b.get("uuid", ""), uid, b.get("base_rev")))
        if ep == "claim":
            ds = self._ds(b)
            if b.get("release"):
                ds.release(b.get("uuid", ""), uid)
            else:
                ds.claim(b.get("uuid", ""), uid)
            return h._json({"claims": ds.active_claims()})
        if ep in ("assign", "assign_auto"):
            if not me["admin"]:
                return h._err("관리자만 분배할 수 있습니다", 403)
            ds, known = self._ds(b), self.auth.users()
            if ep == "assign":
                target = (b.get("user") or "").strip() or None
                if target and target not in known:
                    raise ValueError(f"없는 계정: {target}")
                return h._json(ds.assign_chunks(b.get("chunks"), target, uid,
                                                bool(b.get("force"))))
            users = [x for x in (b.get("users") or []) if x]
            unknown = [x for x in users if x not in known]
            if unknown:
                raise ValueError("없는 계정: " + ", ".join(unknown))
            return h._json(ds.auto_assign(users, int(b.get("per_user") or 0), uid))
        if ep == "passwd":
            if not self.auth.check(uid, b.get("old") or ""):
                return h._err("현재 비밀번호가 맞지 않습니다", 400)
            self.auth.set_user(uid, b.get("new") or "")
            return h._json({"ok": True})
        if ep == "useradd":
            if not me["admin"]:
                return h._err("관리자만 할 수 있습니다", 403)
            self.auth.set_user((b.get("user") or "").strip(), b.get("password") or "",
                               name=(b.get("name") or "").strip() or None,
                               admin=bool(b.get("admin")), create=not b.get("reset"))
            return h._json({"ok": True})
        return h._err("unknown endpoint: " + ep, 404)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _ask_pw() -> str:
    a = getpass.getpass("비밀번호: ")
    if a != getpass.getpass("비밀번호 확인: "):
        sys.exit("비밀번호가 서로 다릅니다")
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("useradd")
    a.add_argument("user")
    a.add_argument("--name")
    a.add_argument("--admin", action="store_true")
    sub.add_parser("passwd").add_argument("user")
    sub.add_parser("userdel").add_argument("user")
    sub.add_parser("users")
    i = sub.add_parser("import")
    i.add_argument("dataset")
    i.add_argument("file")
    i.add_argument("--user", required=True, help="이력에 '가져온 사람'으로 남을 계정")
    i.add_argument("--overwrite", action="store_true",
                   help="이미 라벨된 클립도 덮어쓴다(기본은 건너뜀)")
    args = ap.parse_args()

    auth = Auth(STORE)
    try:
        if args.cmd == "useradd":
            auth.set_user(args.user, _ask_pw(), name=args.name, admin=args.admin, create=True)
            print(f"추가: {args.user}{' (관리자)' if args.admin else ''}")
        elif args.cmd == "passwd":
            auth.set_user(args.user, _ask_pw())
            print(f"변경: {args.user}")
        elif args.cmd == "userdel":
            auth.del_user(args.user)
            print(f"삭제: {args.user}")
        elif args.cmd == "users":
            for k, v in sorted(auth.users().items()):
                print(f"{k:20s} {v.get('name', ''):16s} {'admin' if v.get('admin') else ''}")
        elif args.cmd == "import":
            if args.user not in auth.users():
                sys.exit(f"없는 계정: {args.user}")
            svc = LabelingService(warm=False)
            ds = svc.datasets.get(args.dataset) or sys.exit(f"없는 데이터셋: {args.dataset}")
            ds._build_index()
            if ds.index_state != "ready":
                sys.exit(ds.index_error)
            data = json.loads(Path(args.file).read_text(encoding="utf-8"))
            clips = data.get("clips", data)
            n = skip = miss = 0
            with ds.lock:
                ds._sync()
                for u, lab in clips.items():
                    if u not in ds.pos:
                        miss += 1
                        continue
                    prev = ds.labels.get(u)
                    if prev is not None and not args.overwrite:
                        skip += 1
                        continue
                    new = ds.schema.normalize(lab, lab)
                    if prev is not None and not Schema.diff(prev, new):
                        skip += 1
                        continue
                    ds._record(u, args.user, "create" if prev is None else "update",
                               prev, new, extra={"import": Path(args.file).name})
                    n += 1
                ds._write_labels()
            print(f"가져옴 {n}건, 건너뜀 {skip}건, 데이터셋에 없음 {miss}건 -> {ds.labels_path}")
            print("떠 있는 GUI 서버는 다음 폴링(수 초) 때 이력을 다시 읽어 반영합니다.")
    except ValueError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""edge-case 마이닝 GUI 백엔드 - 표준 라이브러리만 쓰는 로컬 웹 서버.

왜 Flask/FastAPI 를 안 쓰나:
  이 서버의 base 환경은 pip 설치 한 번으로 torch 가 CUDA 13 판으로 갈려
  GPU 8장이 전부 먹통이 된 적이 있다(20260824). 추론 파이프라인이 얹혀
  있는 환경에 웹 프레임워크를 새로 넣을 이유가 없다 - http.server 로
  충분하고, 새 의존성이 0 이면 그 사고가 반복될 여지도 0 이다.

제공 기능(요청 3가지):
  1) 추론 실행 + 성능 리포트 + 카테고리별 TP/FP/FN 목록 + 영상 다운로드
  2) 씬(카테고리 정의)과 라벨 추가/수정
  3) 씬 검색 - uuid / 카테고리 / safety / rarity 필터

채점은 evaluate_labels.py 의 함수를 그대로 불러 쓴다. 지표 계산을 여기서
다시 구현하면 CLI 와 GUI 가 서로 다른 숫자를 내놓게 되고, 그건 이 프로젝트가
이미 한 번 겪은 사고다(집계는 D 기준, 채점은 C 기준으로 갈렸던 건).

실행:
  python3 gui_server.py                 # http://127.0.0.1:8000
  python3 gui_server.py --port 8080 --host 0.0.0.0
원격 서버면 SSH 터널을 권한다:
  ssh -L 8000:127.0.0.1:8000 etri@<server>
"""
from __future__ import annotations

import argparse
import glob
import json
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import config
import evaluate_labels as EV
from constrained_tier import TIER_VALUES

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
GUI_DIR = ROOT / "gui"

# 프레임 소스. None 이면 pav_sample 의 mp4 파일을 직접 서빙한다.
# --data nas 로 켜면 NAS 청크 zip 안의 mp4 를 /clip/<uuid> 로 흘려보낸다.
CLIP_SOURCE = None

# 추론 작업은 몇 시간씩 걸린다(27B 기준 클립당 40초). 브라우저 요청 안에서
# 돌릴 수 없으므로 백그라운드 프로세스로 띄우고 상태만 폴링하게 한다.
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def _safe_under(base: Path, target: Path) -> bool:
    """target 이 base 안에 있나. 미디어 서빙에서 ../ 탈출을 막는다."""
    try:
        Path(target).resolve().relative_to(Path(base).resolve())
        return True
    except (ValueError, OSError):
        return False


def list_label_files() -> list[str]:
    return sorted(p.name for p in ROOT.glob("test_label_*.json"))


def is_v2_scene(path: Path) -> bool:
    """2.0 스키마인가 - special.scenarios[].categories[] 에 templates 키가 있나.

    1.x 는 같은 자리에 synonyms/prompt_templates(_candidates) 를 쓴다. 그 파일을
    GUI 에서 열면 모든 값이 0/빈칸으로 보이고, 저장하면 2.0 필드로 덮어써
    원본을 망가뜨린다. 그래서 목록 단계에서 걸러낸다. 파일명(2.0/1.5)이 아니라
    내용을 보는 이유는, 이름 규칙이 바뀌어도(3.0, _E 등) 계속 맞기 때문이다.
    """
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        for scen in d.get("special", {}).get("scenarios", []):
            for cat in scen.get("categories", []):
                return "templates" in cat
    except (json.JSONDecodeError, OSError, AttributeError):
        pass
    return False


def list_scene_files() -> list[str]:
    """편집 가능한(2.0 스키마) 분류 체계 파일만."""
    return sorted(p.name for p in ROOT.glob("scene_category_*.json")
                  if is_v2_scene(p))


def taxonomy_categories(scene_path: Path) -> list[dict]:
    """[{scenario, name, templates, template_candidates}] - 프롬프트 메뉴와 같은 순서."""
    scene = json.loads(Path(scene_path).read_text(encoding="utf-8"))
    out = []
    for scen in scene.get("special", {}).get("scenarios", []):
        for cat in scen.get("categories", []):
            out.append({
                "scenario": scen["name"],
                "name": cat["name"],
                "templates": cat.get("templates", []),
                "template_candidates": cat.get("template_candidates", []),
            })
    return out


# 실행 결과는 성격에 따라 나눠 담는다.
#   labeld/    - 라벨된 로컬 클립 추론 + 채점 (평가 결과가 있다)
#   unlabeled/ - NAS 데이터 마이닝 (대조할 정답이 없어 채점 불가)
# 평가/검색 탭은 채점된 실행만 골라야 하므로 이 구분이 필요하다.
LABELED_DIR = "labeld"
UNLABELED_DIR = "unlabeled"


def run_kind(rel: str) -> str:
    """실행 이름(RESULTS 기준 상대경로)이 어느 갈래인지. 옛 평면 구조는 이름으로 본다."""
    head = rel.split("/")[0]
    if head in (LABELED_DIR, UNLABELED_DIR):
        return "labeled" if head == LABELED_DIR else "unlabeled"
    # 분리 이전 실행: _eval 로 끝나면 채점된 것이다(run_labeled_*.sh 규칙).
    return "labeled" if "_eval" in rel else "unlabeled"


# 난이도 4축 - aggregate_clip.log 의 DIFFICULTY SUMMARY 와 같은 순서/이름.
# 종합 난이도(driving_difficulty)는 뺐다 - 요인과 따로 매기면 모델이 둘을
# 무관하게 찍어 모순된 값이 나왔다(prompts.py 의 DIFFICULTY_AXES 주석).
DIFF_AXES = [
    ("illumination", "조도 (illumination)"),
    ("precipitation", "강수 (precipitation)"),
    ("road_surface", "노면 (road_surface)"),
    ("atmospheric_obscurants", "대기 가림 (atmospheric_obscurants)"),
]


def _dist(vals: list[int], lo: int, hi: int) -> dict:
    """점수 분포 + 요약 통계. aggregate_clip.log 의 표 하나에 해당한다."""
    counts = {k: 0 for k in range(lo, hi + 1)}
    for v in vals:
        if v in counts:
            counts[v] += 1
    n = len(vals)
    out = {"n": n, "counts": [{"score": k, "count": counts[k],
                               "pct": (counts[k] / n * 100) if n else 0.0}
                              for k in range(lo, hi + 1)]}
    if n:
        srt = sorted(vals)
        mean = sum(vals) / n
        out.update({
            "mean": mean,
            "var": sum((x - mean) ** 2 for x in vals) / n,
            "median": srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2,
            "min": srt[0], "max": srt[-1],
        })
    return out


def aggregate_run(run_dir: Path) -> dict:
    """NAS 마이닝 실행의 집계.

    aggregate_clip.log 를 파싱하지 않고 clip_results*.csv 에서 직접 센다.
    로그는 사람이 읽으라고 만든 고정폭 텍스트라 서식이 조금만 바뀌어도
    파서가 깨진다. 같은 수치를 원본에서 다시 계산하는 편이 안전하다.
    """
    files = sorted(glob.glob(str(run_dir / "clip_results*.csv")))
    if not files:
        return {"error": f"no clip_results*.csv in {run_dir.name}"}

    import csv as _csv
    rows, seen = [], set()
    for f in files:
        with open(f, encoding="utf-8", newline="") as fh:
            for r in _csv.DictReader(fh):
                u = (r.get("uuid") or "").strip()
                if u and u not in seen:
                    seen.add(u)
                    rows.append(r)

    def as_int(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    total = len(rows)
    cat_of = []
    for r in rows:
        raw = (r.get("categories") or "").strip()
        cat_of.append([c for c in raw.split("|") if c.strip()] if raw else [])
    edge = sum(1 for c in cat_of if c)

    # 카테고리 빈도 - 분류 체계 순서를 유지해 0건도 남긴다(있는데 0인 것과
    # 아예 정의에 없는 것은 다르다).
    # 실행에 기록된 분류 체계를 쓰되, 그 파일이 사라졌으면(이름이 바뀌거나
    # 지워진 옛 실행) 현재 기본값으로 대신한다. 없으면 시나리오 묶음이 전부
    # "(분류 체계 밖)" 으로 빠져 표가 쓸모없어진다.
    scene_json = run_config_of(run_dir).get("key", {}).get("scene_json")
    sp = None
    if scene_json:
        sp = Path(scene_json)
        if not sp.is_absolute():
            sp = ROOT / sp
    if sp is None or not sp.exists():
        sp = Path(config.SCENE_JSON)
    tax = taxonomy_categories(sp) if sp.exists() else []
    order = [(c["scenario"], c["name"]) for c in tax]
    seen_names = {n for _, n in order}
    for cats in cat_of:
        for c in cats:
            if c not in seen_names:
                seen_names.add(c)
                order.append(("(분류 체계 밖)", c))

    freq = []
    for scen, name in order:
        n = sum(1 for cats in cat_of if name in cats)
        freq.append({"scenario": scen, "name": name, "clips": n,
                     "pct_all": (n / total * 100) if total else 0.0,
                     "pct_edge": (n / edge * 100) if edge else 0.0})
    occurrences = sum(len(c) for c in cat_of)

    # 클립당 카테고리 개수 분포
    per_clip = {}
    for cats in cat_of:
        if cats:
            per_clip[len(cats)] = per_clip.get(len(cats), 0) + 1

    tiers = {}
    for key in ("safety_tier", "rarity_tier"):
        allv = [v for v in (as_int(r.get(key)) for r in rows) if v is not None]
        edgev = [v for v, cats in ((as_int(r.get(key)), c)
                                   for r, c in zip(rows, cat_of)) if v is not None and cats]
        tiers[key.replace("_tier", "")] = {
            "all": _dist(allv, min(TIER_VALUES), max(TIER_VALUES)),
            "edge": _dist(edgev, min(TIER_VALUES), max(TIER_VALUES))}

    diff = []
    for key, label in DIFF_AXES:
        vals = [v for v in (as_int(r.get(key)) for r in rows) if v is not None]
        d = _dist(vals, 0, 4)
        d.update({"key": key, "label": label})
        diff.append(d)

    parse_ok = sum(1 for r in rows
                   if str(r.get("parse_ok", "")).strip().lower() in ("true", "1", "yes"))
    cfg = run_config_of(run_dir).get("key", {})
    return {
        "run": rel_run_name(run_dir),
        "clips": {"total": total, "edge": edge, "normal": total - edge,
                  "edge_pct": (edge / total * 100) if total else 0.0},
        "categories": freq,
        "occurrences": occurrences,
        "avg_per_edge": (occurrences / edge) if edge else 0.0,
        "per_clip": [{"k": k, "n": per_clip[k]} for k in sorted(per_clip)],
        "tiers": tiers,
        "difficulty": diff,
        "parse_ok": parse_ok,
        "config": cfg,
        "has_log": (run_dir / "aggregate_clip.log").exists(),
    }


def run_dirs() -> list[Path]:
    """실행 폴더들. labeld/ · unlabeled/ 아래 한 겹과, 분리 이전의 평면 폴더 모두.

    반환값을 RESULTS 기준 상대경로로 쓰는 곳이 많아(=드롭다운 값) 경로는
    그대로 두고, 이름만 rel_run_name() 으로 만든다.
    """
    if not RESULTS.exists():
        return []
    out = []
    for d in RESULTS.iterdir():
        if not d.is_dir():
            continue
        if d.name in (LABELED_DIR, UNLABELED_DIR):
            out.extend(x for x in d.iterdir() if x.is_dir())
        else:
            out.append(d)
    # 폴더명이 아니라 실행 시각(폴더 이름 앞의 타임스탬프)으로 정렬한다.
    # 상대경로로 정렬하면 labeld/* 와 unlabeled/* 가 갈래별로 뭉쳐서, 최근
    # 실행이 목록 위에 오지 않는다.
    return sorted(out, key=lambda p: (p.name, rel_run_name(p)), reverse=True)


def rel_run_name(d: Path) -> str:
    """RESULTS 기준 상대경로 - 'labeld/20260821_110448_eval' 처럼."""
    try:
        return d.relative_to(RESULTS).as_posix()
    except ValueError:
        return d.name


def run_config_of(d: Path) -> dict:
    p = d / "run_config.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def labels_path_for_run(d: Path) -> Path:
    """이 실행을 채점할 때 쓸 라벨 파일.

    run_config.json 에 gt_labels 가 있으면 그걸, 없으면 config.py 기본값.
    실행마다 라벨이 다를 수 있으므로 기본값으로 뭉뚱그리면 안 된다.
    """
    key = run_config_of(d).get("key", {})
    for k in ("gt_labels", "labels"):
        v = key.get(k)
        if v:
            p = Path(v)
            if not p.is_absolute():
                p = ROOT / p
            if p.exists():
                return p
    return Path(config.LABELS_JSON)


def find_viz(run_dir: Path, uuid: str) -> dict:
    """이 실행이 남긴 uuid 의 시각화 파일들. 없으면 빈 dict."""
    out = {}
    for pat in (f"viz/**/{uuid}/clip.mp4", f"viz/{uuid}/clip.mp4"):
        hits = glob.glob(str(run_dir / pat), recursive=True)
        if hits:
            out["video"] = str(Path(hits[0]).relative_to(ROOT))
            rj = Path(hits[0]).with_name("result.json")
            if rj.exists():
                out["result_json"] = str(rj.relative_to(ROOT))
            break
    return out


# ---------------------------------------------------------------------------
# 채점 - evaluate_labels.py 의 함수를 그대로 쓴다
# ---------------------------------------------------------------------------
def evaluate_run(run_dir: Path, labels_path: Path | None = None) -> dict:
    """한 실행의 성능 리포트 + 카테고리별 TP/FP/FN uuid 목록."""
    labels_path = Path(labels_path) if labels_path else labels_path_for_run(run_dir)
    if not labels_path.exists():
        return {"error": f"labels not found: {labels_path}"}

    truth, meta, skipped = EV.load_labels(labels_path)
    pred = EV.load_results(run_dir)
    if not pred:
        return {"error": f"no clip_results*.csv in {run_dir.name}"}

    common = sorted(set(truth) & set(pred))
    if not common:
        return {"error": "라벨과 결과에 겹치는 uuid 가 없습니다"}
    missing = sorted(set(truth) - set(pred))

    # --- 1) Verdict: 카테고리가 하나라도 있으면 Special ---
    v_tp = v_fp = v_fn = v_tn = 0
    for u in common:
        t, p = bool(truth[u]["categories"]), bool(pred[u]["categories"])
        if t and p:
            v_tp += 1
        elif p and not t:
            v_fp += 1
        elif t and not p:
            v_fn += 1
        else:
            v_tn += 1
    v_p, v_r, v_f = EV.prf(v_tp, v_fp, v_fn)
    n = len(common)

    # --- 2) 카테고리별 (멀티라벨) ---
    # 분류 체계에 있는 이름은 GT/예측이 0건이어도 표에 남긴다 - 빠지면
    # "모델이 못 맞춘 것"과 "애초에 없는 카테고리"가 구분되지 않는다.
    scene_json = run_config_of(run_dir).get("key", {}).get("scene_json")
    tax_names = []
    if scene_json:
        sp = Path(scene_json)
        if not sp.is_absolute():
            sp = ROOT / sp
        if sp.exists():
            tax_names = [c["name"] for c in taxonomy_categories(sp)]
    names = sorted(set(tax_names)
                   | {c for u in common for c in truth[u]["categories"]}
                   | {c for u in common for c in pred[u]["categories"]})

    per_cat = []
    micro_tp = micro_fp = micro_fn = 0
    for name in names:
        tp = [u for u in common if name in truth[u]["categories"] and name in pred[u]["categories"]]
        fp = [u for u in common if name not in truth[u]["categories"] and name in pred[u]["categories"]]
        fn = [u for u in common if name in truth[u]["categories"] and name not in pred[u]["categories"]]
        p, r, f = EV.prf(len(tp), len(fp), len(fn))
        support = len(tp) + len(fn)
        micro_tp += len(tp); micro_fp += len(fp); micro_fn += len(fn)
        per_cat.append({
            "name": name, "support": support,
            "precision": p, "recall": r, "f1": f,
            "tp": tp, "fp": fp, "fn": fn,
            "n_tp": len(tp), "n_fp": len(fp), "n_fn": len(fn),
            # 지지수가 적으면 P/R/F1 이 한 건에 크게 흔들려 신뢰하기 어렵다
            "reliable": support >= 3,
        })
    micro_f1 = EV.prf(micro_tp, micro_fp, micro_fn)[2]
    solid = [c for c in per_cat if c["reliable"]]
    macro_f1 = sum(c["f1"] for c in solid) / len(solid) if solid else 0.0
    per_clip = sum(EV.f1_set(truth[u]["categories"], pred[u]["categories"])
                   for u in common) / n

    # --- 3) safety / rarity ---
    tiers = {}
    for key in ("safety", "rarity"):
        pairs = [(truth[u][key], pred[u][key]) for u in common
                 if pred[u].get(key) is not None]
        if pairs:
            mae = sum(abs(a - b) for a, b in pairs) / len(pairs)
            exact = sum(a == b for a, b in pairs) / len(pairs)
            within1 = sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs)
            tiers[key] = {"n": len(pairs), "mae": mae, "exact": exact,
                          "within1": within1}
        else:
            tiers[key] = {"n": 0}

    return {
        "run": run_dir.name,
        "labels": str(labels_path),
        "labels_name": labels_path.name,
        "n_labels": len(truth),
        "n_results": len(pred),
        "n_matched": n,
        "missing": missing[:50],
        "n_missing": len(missing),
        "skipped": skipped,
        "verdict": {
            "tp": v_tp, "fp": v_fp, "fn": v_fn, "tn": v_tn,
            "accuracy": (v_tp + v_tn) / n,
            "precision": v_p, "recall": v_r, "f1": v_f,
            "n_special_truth": sum(1 for u in common if truth[u]["categories"]),
            "n_normal_truth": sum(1 for u in common if not truth[u]["categories"]),
        },
        "categories": per_cat,
        "micro_f1": micro_f1, "macro_f1": macro_f1, "per_clip_f1": per_clip,
        "tiers": tiers,
        "config": run_config_of(run_dir).get("key", {}),
        "meta": meta,
    }


def clip_detail(uuid: str, run_dir: Path | None, labels_path: Path) -> dict:
    """클립 하나의 GT / 예측 / 시각화 경로."""
    out = {"uuid": uuid}
    truth, _, _ = EV.load_labels(labels_path)
    if uuid in truth:
        t = truth[uuid]
        out["truth"] = {"categories": sorted(t["categories"]),
                        "safety": t["safety"], "rarity": t["rarity"],
                        "note": t["note"]}
    if run_dir is not None:
        rows = EV.load_results(run_dir)
        if uuid in rows:
            r = rows[uuid]
            out["pred"] = {"categories": sorted(r["categories"]),
                           "safety": r["safety"], "rarity": r["rarity"]}
        out.update(find_viz(run_dir, uuid))
        rj = out.get("result_json")
        if rj:
            try:
                out["reasoning"] = json.loads((ROOT / rj).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    # 원본 클립(시각화가 없어도 영상은 볼 수 있어야 한다)
    view = "camera_front_wide_120fov"
    if CLIP_SOURCE is not None:
        # NAS 청크 zip 은 mp4 가 파일로 없다. /clip/<uuid> 로 서빙한다.
        if CLIP_SOURCE.exists(view, uuid):
            out["source_video"] = f"clip/{uuid}"
    else:
        src = ROOT / "pav_sample/camera" / view / f"{uuid}.{view}.mp4"
        if src.exists():
            out["source_video"] = str(src.relative_to(ROOT))
    return out


# ---------------------------------------------------------------------------
# 라벨 편집
# ---------------------------------------------------------------------------
LABEL_FIELDS = ("categories", "influenced_ego", "safety_criticality",
                "rarity", "weather", "is_night", "note")


def read_labels_file(name: str) -> dict:
    p = ROOT / name
    if not p.exists() or not p.name.startswith("test_label_"):
        raise FileNotFoundError(name)
    return json.loads(p.read_text(encoding="utf-8"))


def write_labels_file(name: str, data: dict):
    """원자적 저장 + 1회 백업.

    라벨 파일은 손으로 만든 정답이라 날리면 복구가 안 된다. 임시 파일에
    다 쓴 뒤 os.replace 로 갈아끼워, 쓰다 죽어도 반쪽짜리 파일이 남지 않게
    한다.
    """
    p = ROOT / name
    if not p.name.startswith("test_label_"):
        raise ValueError("허용되지 않은 파일: " + name)
    if p.exists():
        shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, p)


def upsert_label(file_name: str, uuid: str, patch: dict) -> dict:
    data = read_labels_file(file_name)
    clips = data.setdefault("clips", {})
    cur = clips.get(uuid, {"categories": [], "influenced_ego": False,
                           "safety_criticality": min(TIER_VALUES),
                           "rarity": min(TIER_VALUES),
                           "weather": [], "is_night": False, "note": ""})
    for k in LABEL_FIELDS:
        if k in patch:
            cur[k] = patch[k]
    clips[uuid] = cur
    write_labels_file(file_name, data)
    return cur


def delete_label(file_name: str, uuid: str) -> bool:
    data = read_labels_file(file_name)
    clips = data.get("clips", {})
    if uuid not in clips:
        return False
    del clips[uuid]
    write_labels_file(file_name, data)
    return True


# ---------------------------------------------------------------------------
# 분류 체계(씬) 편집
# ---------------------------------------------------------------------------
def write_scene_file(name: str, data: dict):
    p = ROOT / name
    if not p.name.startswith("scene_category_"):
        raise ValueError("허용되지 않은 파일: " + name)
    if p.exists():
        shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, p)


def upsert_category(scene_name: str, scenario: str, cat: dict,
                    old_name: str | None = None) -> dict:
    """카테고리 추가/수정. old_name 을 주면 이름 변경으로 처리한다.

    카테고리 이름은 프롬프트 메뉴와 정답 라벨이 문자열로 맞춰 쓰는 키라서,
    이름을 바꾸면 기존 라벨이 매칭되지 않는다 - 그래서 이름 변경은
    호출한 쪽이 old_name 을 명시했을 때만 한다.
    """
    p = ROOT / scene_name
    data = json.loads(p.read_text(encoding="utf-8"))
    scenarios = data.setdefault("special", {}).setdefault("scenarios", [])
    target = next((s for s in scenarios if s["name"] == scenario), None)
    if target is None:
        target = {"name": scenario, "categories": []}
        scenarios.append(target)

    entry = {
        "name": cat["name"],
        "templates": cat.get("templates", []),
        "template_candidates": cat.get("template_candidates", []),
    }
    look = old_name or cat["name"]
    for s in scenarios:
        for i, c in enumerate(s.get("categories", [])):
            if c["name"] == look:
                if s is target:
                    target["categories"][i] = entry
                else:      # 시나리오가 바뀐 경우 - 옮긴다
                    del s["categories"][i]
                    target["categories"].append(entry)
                write_scene_file(scene_name, data)
                return entry
    target["categories"].append(entry)
    write_scene_file(scene_name, data)
    return entry


def delete_category(scene_name: str, name: str) -> bool:
    p = ROOT / scene_name
    data = json.loads(p.read_text(encoding="utf-8"))
    for s in data.get("special", {}).get("scenarios", []):
        for i, c in enumerate(s.get("categories", [])):
            if c["name"] == name:
                del s["categories"][i]
                write_scene_file(scene_name, data)
                return True
    return False


# ---------------------------------------------------------------------------
# 씬 검색
# ---------------------------------------------------------------------------
# 난이도 4축 - 검색 필터가 쓰는 키 목록. aggregate 쪽 DIFF_AXES 와 같은 순서.
DIFF_KEYS = [k for k, _ in DIFF_AXES]


def load_pred_rows(run_dir: Path) -> dict:
    """실행 CSV -> {uuid: {...}}. 난이도 4축과 근거 텍스트까지 들고 온다.

    EV.load_results 는 채점에 필요한 categories/safety/rarity 만 준다.
    씬 검색은 난이도로도 거르고 팝업에 근거를 보여줘야 해서 원본 행이
    통째로 필요하다.
    """
    files = sorted(glob.glob(str(run_dir / "clip_results*.csv")))
    if not files:
        return {}
    import csv as _csv
    out = {}
    for f in files:
        with open(f, encoding="utf-8", newline="") as fh:
            for r in _csv.DictReader(fh):
                u = (r.get("uuid") or "").strip()
                if not u or u in out:
                    continue
                raw = (r.get("categories") or "").strip()
                out[u] = {
                    "uuid": u,
                    "categories": [c for c in raw.split("|") if c.strip()],
                    "safety": _as_int(r.get("safety_tier")),
                    "rarity": _as_int(r.get("rarity_tier")),
                    "difficulty": {k: _as_int(r.get(k)) for k in DIFF_KEYS},
                    # 축마다 근거 문장이 따로 있다(<축>_reason). 점수만 보면
                    # 왜 그렇게 매겼는지 알 수 없어 검토가 안 된다.
                    "difficulty_reason": {
                        k: r.get(k + "_reason", "") for k in DIFF_KEYS},
                    "observation": r.get("observation", ""),
                    "ego_behavior": r.get("ego_behavior", ""),
                    "unusual_elements": r.get("unusual_elements", ""),
                    "safety_reason": r.get("safety_reason", ""),
                    "rarity_reason": r.get("rarity_reason", ""),
                    "verdict": r.get("verdict", ""),
                }
    return out


def _as_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _diff_ok(d: dict, q: dict) -> bool:
    """난이도 4축 범위 필터.

    조건을 건 축에 값이 없는 클립은 뺀다. 통과시키면 "종합 난이도 3 이상"
    같은 조건에 값 없는 클립이 전부 딸려 나와(실측 163,084 중 136,060개가
    난이도 미기록) 필터가 고장난 것처럼 보인다. 조건을 걸지 않은 축은
    당연히 아무것도 거르지 않으므로, 옛 실행이라고 결과가 0건이 되지는
    않는다.
    """
    for k in DIFF_KEYS:
        lo, hi = q.get(k + "_min"), q.get(k + "_max")
        if lo is None and hi is None:
            continue
        v = (d or {}).get(k)
        if v is None:
            return False
        if lo is not None and v < int(lo):
            return False
        if hi is not None and v > int(hi):
            return False
    return True


def search_nas_clips(q: dict) -> dict:
    """NAS 마이닝 실행(unlabeled/)의 예측 자체를 검색한다.

    라벨 검색과 달리 대조할 정답이 없다. 그래서 모델이 낸 카테고리·점수·
    난이도를 그대로 조건으로 쓴다.
    """
    run = (q.get("run") or "").strip()
    if not run:
        return {"error": "실행을 선택하세요", "total": 0, "rows": []}
    d = RESULTS / run
    if not d.exists():
        return {"error": f"no such run: {run}", "total": 0, "rows": []}
    preds = load_pred_rows(d)
    if not preds:
        return {"error": f"no clip_results*.csv in {run}", "total": 0, "rows": []}

    uuid_q = (q.get("uuid") or "").strip().lower()
    text_q = (q.get("text") or "").strip().lower()
    cats = [c for c in (q.get("categories") or "").split("|") if c]
    cat_mode = q.get("cat_mode", "any")
    smin = int(q.get("safety_min", min(TIER_VALUES)))
    smax = int(q.get("safety_max", max(TIER_VALUES)))
    rmin = int(q.get("rarity_min", min(TIER_VALUES)))
    rmax = int(q.get("rarity_max", max(TIER_VALUES)))
    special = q.get("special", "")

    rows = []
    for u, r in preds.items():
        if uuid_q and uuid_q not in u.lower():
            continue
        pc = set(r["categories"])
        if cats:
            hit = set(cats) & pc
            if cat_mode == "any" and not hit:
                continue
            if cat_mode == "all" and not set(cats) <= pc:
                continue
            if cat_mode == "none" and hit:
                continue
        if special == "special" and not pc:
            continue
        if special == "normal" and pc:
            continue
        # 점수가 없는 클립(파싱 실패 등)은 범위 조건을 좁혔을 때만 뺀다.
        if r["safety"] is not None and not (smin <= r["safety"] <= smax):
            continue
        if r["rarity"] is not None and not (rmin <= r["rarity"] <= rmax):
            continue
        if not _diff_ok(r["difficulty"], q):
            continue
        if text_q and text_q not in " ".join([
                r["observation"], r["ego_behavior"], r["unusual_elements"],
                r["safety_reason"], r["rarity_reason"]]).lower():
            continue
        rows.append(r)

    rows.sort(key=lambda r: (-(r["safety"] or 0), -(r["rarity"] or 0), r["uuid"]))
    total = len(rows)
    limit = int(q.get("limit", 500))
    return {"total": total, "rows": rows[:limit], "run": run,
            "has_difficulty": any(
                v is not None for r in list(preds.values())[:50]
                for v in r["difficulty"].values())}


def search_clips(q: dict) -> dict:
    """라벨 기준 검색. uuid / 카테고리 / safety / rarity 필터.

    run 을 주면 그 실행의 예측도 함께 붙여, 검색 결과에서 바로 맞았는지
    틀렸는지 볼 수 있게 한다.
    """
    labels_name = q.get("labels") or Path(config.LABELS_JSON).name
    truth, _, _ = EV.load_labels(ROOT / labels_name)

    run = q.get("run")
    pred, pred_rows = {}, {}
    if run:
        d = RESULTS / run
        if d.exists():
            pred = EV.load_results(d)
            # 난이도는 정답 라벨에 없는 값이라 실행 CSV 에서 가져온다.
            pred_rows = load_pred_rows(d)

    uuid_q = (q.get("uuid") or "").strip().lower()
    cats = [c for c in (q.get("categories") or "").split("|") if c]
    cat_mode = q.get("cat_mode", "any")          # any | all | none
    smin = int(q.get("safety_min", min(TIER_VALUES)))
    smax = int(q.get("safety_max", max(TIER_VALUES)))
    rmin = int(q.get("rarity_min", min(TIER_VALUES)))
    rmax = int(q.get("rarity_max", max(TIER_VALUES)))
    special = q.get("special", "")               # "" | special | normal
    note_q = (q.get("note") or "").strip().lower()
    only = q.get("only", "")                     # "" | tp | fp | fn (run 필요)
    # 난이도 필터는 예측값 기준이다. 실행을 안 골랐으면 적용할 수 없으므로
    # 조용히 무시한다 - 조건을 걸었는데 전부 탈락하는 것보다 낫다.
    want_diff = any(q.get(k + "_min") is not None or q.get(k + "_max") is not None
                    for k in DIFF_KEYS)

    rows = []
    for u, t in truth.items():
        if uuid_q and uuid_q not in u.lower():
            continue
        tc = t["categories"]
        if cats:
            hit = set(cats) & tc
            if cat_mode == "any" and not hit:
                continue
            if cat_mode == "all" and not set(cats) <= tc:
                continue
            if cat_mode == "none" and hit:
                continue
        if not (smin <= t["safety"] <= smax):
            continue
        if not (rmin <= t["rarity"] <= rmax):
            continue
        if special == "special" and not tc:
            continue
        if special == "normal" and tc:
            continue
        if note_q and note_q not in (t.get("note") or "").lower():
            continue

        if want_diff and pred_rows:
            if not _diff_ok((pred_rows.get(u) or {}).get("difficulty"), q):
                continue

        row = {"uuid": u, "categories": sorted(tc), "safety": t["safety"],
               "rarity": t["rarity"], "note": t.get("note", "")}
        if u in pred_rows:
            row["difficulty"] = pred_rows[u]["difficulty"]
        if u in pred:
            pc = pred[u]["categories"]
            row["pred_categories"] = sorted(pc)
            row["pred_safety"] = pred[u]["safety"]
            row["pred_rarity"] = pred[u]["rarity"]
            row["correct"] = (pc == tc)
            if only == "tp" and not (pc and tc):
                continue
            if only == "fp" and not (pc - tc):
                continue
            if only == "fn" and not (tc - pc):
                continue
        elif only:
            continue
        rows.append(row)

    rows.sort(key=lambda r: (-r["safety"], -r["rarity"], r["uuid"]))
    total = len(rows)
    limit = int(q.get("limit", 500))
    return {"total": total, "rows": rows[:limit], "labels": labels_name,
            "has_difficulty": any(
                v is not None for r in list(pred_rows.values())[:50]
                for v in r["difficulty"].values())}


# ---------------------------------------------------------------------------
# 추론 작업 실행
# ---------------------------------------------------------------------------
# 모델마다 두 갈래 스크립트를 든다.
#   labeled : 로컬 pav_sample 의 라벨된 클립 -> 추론 + 채점
#   nas     : NAS 청크 zip 전체         -> 추론만 (대조할 정답이 없다)
# GUI 의 "로컬 샘플 추론&평가" / "NAS 전체 추론" 탭이 각각을 고른다.
# NAS 마운트 명령. 서버가 대신 실행하지는 않는다 - 그러려면 NAS 비밀번호를
# 파일에 두거나 sudo 를 NOPASSWD 로 풀어야 하는데, 이 GUI 는 인증이 없어서
# 포트에 닿는 누구나 그 권한을 쓰게 된다. 명령만 보여주고 실행은 사람이 한다.
NAS_SHARE = "//10.254.92.171/OPEN_DATASET"
NAS_MOUNT = "/mnt/nas"
NAS_USER = "E2E_DATA"


def nas_mount_command() -> str:
    return (f"sudo mount -t cifs {NAS_SHARE} {NAS_MOUNT} "
            f"-o username={NAS_USER},vers=3.0,uid=$(id -u),gid=$(id -g),"
            "cache=loose,nounix,serverino,iocharset=utf8")


def nas_status() -> dict:
    """NAS 가 실제로 읽히는지. 마운트 여부만 보면 안 된다 - 마운트가 남아
    있는데 서버가 죽어 I/O 만 막히는 상태도 있다. 그래서 디렉터리를 한 번
    열어 본다.
    """
    from clip_source import NAS_CAMERA_DIR
    d = Path(NAS_CAMERA_DIR)
    ok, reason = False, ""
    try:
        ok = d.is_dir() and any(d.iterdir())
        if not ok:
            reason = "마운트되어 있지 않거나 비어 있습니다"
    except OSError as e:
        reason = f"{type(e).__name__}: {e}"
    return {
        "ok": ok,
        "reason": reason,
        "path": str(d),
        "command": nas_mount_command(),
        # 이 GUI 가 NAS 를 쓰도록 떠 있는가(--data nas). local 이면 NAS 가
        # 없어도 정상이므로 경고를 띄우지 않는다.
        "needed": CLIP_SOURCE is not None and getattr(CLIP_SOURCE, "kind", "") == "zip",
    }


def nas_clip_count():
    """NAS 청크에 든 고유 클립 수. 모르면 None.

    clip_source 가 만들어 둔 인덱스 캐시만 읽는다 - 여기서 인덱스를 새로
    만들면 GUI 부팅이 몇 분씩 멈춘다. 캐시가 없으면 화면에서 숫자를 빼고
    "전체"라고만 쓴다.
    """
    try:
        from clip_source import NAS_CAMERA_DIR
        cache = Path(NAS_CAMERA_DIR) / ".clip_index.json"
        if not cache.exists():
            return None
        blob = json.loads(cache.read_text())
        idx = blob.get("index") if isinstance(blob, dict) else blob
        return len(idx) if idx else None
    except (OSError, ValueError, ImportError):
        return None


MODELS = {
    "Qwen/Qwen3-VL-8B-Instruct": {
        "script": "run_labeled_8b.sh", "nas_script": "run_nas_nvidia_8b.sh",
        "label": "Qwen3-VL-8B (8샤드 병렬, 빠름)",
    },
    "Qwen/Qwen3.8-27B": {
        "script": "run_labeled_27b.sh", "nas_script": "run_nas_nvidia_27b.sh",
        "label": "Qwen3.8-27B (GPU 3장, 느림)",
    },
    "Qwen/Qwen3-VL-32B-Instruct": {
        "script": "run_labeled_8b.sh", "nas_script": "run_nas_nvidia_8b.sh",
        "label": "Qwen3-VL-32B",
    },
}


# 추론 스크립트를 어떤 파이썬으로 돌릴 것인가.
#   run_labeled_8b.sh 안의 python3 는 PATH 를 따라간다. 이 GUI 를 mmdet3d(3.8) 처럼
#   엉뚱한 conda 환경에서 띄우면 그 PATH 가 자식에게 그대로 상속되고, 샤드
#   8개가 전부 "'type' object is not subscriptable" 로 즉사한다
#   (실측 20260831_132629_eval - 3시간 뒤에야 결과 0건인 걸 알게 된다).
#   그래서 파이프라인 환경의 bin 을 PATH 맨 앞에 붙여 넘긴다.
#   27B 는 run_labeled_27b.sh 가 자기 PYBIN(qwen38)을 못박아 쓰므로 영향 없다.
PIPELINE_BIN = "/home/etri/miniconda3/bin"


def job_env() -> dict:
    env = dict(os.environ)
    env["PATH"] = PIPELINE_BIN + os.pathsep + env.get("PATH", "")
    # 활성화돼 있던 다른 환경의 흔적은 지운다. 남겨두면 파이썬이 남의
    # site-packages 를 먼저 집는다.
    for k in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONHOME", "PYTHONPATH"):
        env.pop(k, None)
    return env


# 단일 클립 조회로 만든 uuid 목록을 두는 곳. run 폴더가 생기기 전에 써야
# 하므로(스크립트가 만든다) gui/ 아래에 job id 로 남긴다.
SINGLE_DIR = GUI_DIR / "single"


def write_uuid_file(uuid: str, jid: str) -> Path:
    """--only-uuids 에 넘길 한 줄짜리 파일을 만든다."""
    SINGLE_DIR.mkdir(parents=True, exist_ok=True)
    p = SINGLE_DIR / f"{jid}.txt"
    p.write_text(uuid.strip() + "\n", encoding="utf-8")
    return p


def build_command(opts: dict, jid: str = "adhoc") -> list[str]:
    model = opts.get("model", "Qwen/Qwen3-VL-8B-Instruct")
    spec = MODELS.get(model)
    if spec is None:
        raise ValueError("모르는 모델: " + model)
    # mode: "labeled"(기본) 는 로컬 라벨 클립 추론+채점, "nas" 는 NAS 청크
    # 전체를 훑는 마이닝. 스크립트가 갈리고 받는 인자도 다르다.
    nas = opts.get("mode") == "nas"
    script = spec["nas_script"] if nas else spec["script"]
    cmd = ["bash", script]

    # 단일 클립 조회: uuid 하나만 돌려 시각화를 바로 본다.
    # 샤드를 8개로 두면 7개는 할 일이 없는데도 모델을 올리느라 GPU 를 물고
    # 몇 분을 버린다 - 1개로 못박는다. 시각화는 이 기능의 목적이므로 강제.
    single = (opts.get("single_uuid") or "").strip()

    is27 = "27b" in script
    if is27:
        if opts.get("gpus"):
            cmd += ["--gpus", "1" if single else str(opts["gpus"])]
        elif single:
            cmd += ["--gpus", "1"]
    elif single:
        cmd += ["--num-shards", "1"]
    elif opts.get("shards"):
        cmd += ["--num-shards", str(opts["shards"])]
    # 네 스크립트 모두 --model 을 받는다. 안 넘기면 8B 기본값으로 조용히
    # 돌아버려서, 몇 시간 뒤 결과를 보고서야 다른 모델이었다는 걸 알게 된다.
    cmd += ["--model", model]

    # 라벨은 채점용이라 NAS 마이닝에는 없다(대조할 정답이 없다).
    if not nas and opts.get("labels"):
        cmd += ["--labels", str(ROOT / opts["labels"])]
    # 반대로 --limit-clips 는 마이닝 전용 - 라벨 클립은 uuid 로 흩어져 있어
    # 앞에서 N개를 자를 수 없다.
    if nas and opts.get("limit_clips") and not single:
        cmd += ["--limit-clips", str(opts["limit_clips"])]
    if opts.get("scene_json"):
        cmd += ["--scene-json", str(ROOT / opts["scene_json"])]
    if opts.get("use_egomotion"):
        cmd += ["--use-egomotion"]
    if opts.get("traj"):
        cmd += ["--traj", opts["traj"]]
    # 시각화 기본값이 스크립트마다 반대다: labeled_* 는 off(--viz 로 켬),
    # nas_* 는 on(--no-viz 로 끔). 켤 때는 --viz-normal/--viz-special 로
    # 어느 쪽이든 명시적으로 켜지고, 끌 때는 nas_* 에만 --no-viz 가 필요하다.
    # 세 갈래다: 전부 생성 / 카테고리별 N개만 / 생성 안 함.
    if single:
        cmd += ["--viz-normal", "--viz-special"]
    elif opts.get("viz_per_category"):
        cmd += ["--viz-per-category", str(opts["viz_per_category"])]
    elif opts.get("viz"):
        cmd += ["--viz-normal", "--viz-special"]
    elif nas:
        cmd += ["--no-viz"]
    # 등급(4·5단계)을 프롬프트에서 빼는 스위치. 두 개를 다 끄면 스크립트에
    # --no-score-tiers 별칭이 있지만, 굳이 쓰지 않는다. run.log 에 남는 명령이
    # 어느 축을 껐는지 그대로 읽히는 편이 나중에 실행끼리 비교할 때 낫다.
    if opts.get("no_safety_tier"):
        cmd += ["--no-safety-tier"]
    if opts.get("no_rarity_tier"):
        cmd += ["--no-rarity-tier"]
    # 난이도 4축은 등급과 별개 축이라 기본 off - 켤 때만 넘긴다.
    if opts.get("difficulty"):
        cmd += ["--difficulty"]
    # 프레임 소스. 스크립트마다 기본값이 다르므로(labeled=local, nas=nas)
    # 기본과 같을 때만 생략한다 - run.log 에 남는 명령이 짧을수록 낫다.
    data = opts.get("data")
    if data and data != ("nas" if nas else "local"):
        cmd += ["--data", data]
    if opts.get("memo"):
        cmd += ["--memo", opts["memo"]]
    # 맨 뒤에 붙인다 - run.log 한 줄에서 "어느 클립이었나" 가 눈에 띄게.
    if single:
        cmd += ["--only-uuids", str(write_uuid_file(single, jid))]
    return cmd


def start_job(opts: dict) -> dict:
    jid = datetime.now().strftime("%Y%m%d_%H%M%S")
    cmd = build_command(opts, jid)
    log_path = ROOT / "gui" / "jobs" / f"{jid}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "w", encoding="utf-8")
    fh.write(f"$ {' '.join(cmd)}\n\n")
    fh.flush()
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh,
                            stderr=subprocess.STDOUT, env=job_env(),
                            start_new_session=True)
    job = {"id": jid, "cmd": cmd, "pid": proc.pid, "log": str(log_path),
           "started": time.time(), "status": "running", "opts": opts,
           # 단일 클립 조회면 프런트가 끝난 뒤 이 uuid 로 시각화를 찾는다.
           "single_uuid": (opts.get("single_uuid") or "").strip() or None}
    with JOBS_LOCK:
        JOBS[jid] = dict(job, _proc=proc)
    return job


def job_view(j: dict) -> dict:
    proc = j.get("_proc")
    if proc is not None and j["status"] == "running":
        rc = proc.poll()
        if rc is not None:
            j["status"] = "done" if rc == 0 else f"failed (exit {rc})"
            j["returncode"] = rc
    out = {k: v for k, v in j.items() if not k.startswith("_")}
    # 진행률: tqdm 이 \r 로 덮어쓰므로 마지막 조각만 꺼내 보여준다
    try:
        txt = Path(j["log"]).read_text(encoding="utf-8", errors="replace")
        tail = txt.replace("\r", "\n").strip().split("\n")
        out["tail"] = [l for l in tail if l.strip()][-25:]
        m = None
        for line in reversed(tail):
            m = re.search(r"(\d+)/(\d+)\s*\[", line) or re.search(
                r"\[progress\]\s*(\d+)/(\d+)", line)
            if m:
                break
        if m:
            out["progress"] = {"done": int(m.group(1)), "total": int(m.group(2))}
        # 이 작업이 만든 결과 폴더를 찾아 링크해준다
        mm = re.search(r"run dir\s*:\s*(\S+)", txt)
        if mm:
            # 로그의 "results/labeld/<ts>_eval" 에서 RESULTS 아래 상대경로를
            # 그대로 남긴다. .name 만 쓰면 labeld/ 가 떨어져 나가 프런트가
            # RESULTS/<이름> 으로 찾을 때 없는 경로가 된다.
            rd = Path(mm.group(1))
            try:
                out["run_dir"] = rd.resolve().relative_to(RESULTS.resolve()).as_posix()
            except ValueError:
                out["run_dir"] = rd.name
    except OSError:
        out["tail"] = []
    return out


def stop_job(jid: str) -> bool:
    with JOBS_LOCK:
        j = JOBS.get(jid)
    if not j or j["status"] != "running":
        return False
    proc = j.get("_proc")
    if proc is None:
        return False
    # start_new_session=True 로 띄웠으므로 프로세스 그룹째 정리한다.
    # 8샤드 실행은 자식이 8개라 부모만 죽이면 GPU 를 붙든 채 남는다.
    try:
        os.killpg(os.getpgid(proc.pid), 15)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    j["status"] = "stopped"
    return True


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "EdgeCaseGUI/1.0"

    def log_message(self, fmt, *args):        # 요청마다 찍히면 시끄럽다
        pass

    # --- 응답 헬퍼 ---
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, msg, code=400):
        self._json({"error": msg}, code)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def _send_clip(self, uuid: str, view: str = "camera_front_wide_120fov"):
        """NAS 청크 zip 안의 mp4 를 Range 지원해서 그대로 흘려보낸다.

        zip 이 무압축(store)이라 내부 mp4 는 연속 바이트 구간이다. 그래서
        구간만 열어주면(open_video) 일반 파일과 똑같이 seek 이 된다.
        """
        if CLIP_SOURCE is None or not CLIP_SOURCE.exists(view, uuid):
            return self._err("not found: " + uuid, 404)
        size = CLIP_SOURCE.size_of(view, uuid)
        self._stream(lambda: CLIP_SOURCE.open_video(view, uuid), size,
                     "video/mp4", f"{uuid}.{view}.mp4")

    def _send_file(self, path: Path, download=False):
        """정적 파일. mp4 는 Range 를 지원해야 브라우저에서 탐색이 된다."""
        # 경로 검사를 존재 확인보다 먼저 한다 - 순서가 반대면 404/403 차이로
        # 프로젝트 밖 파일의 존재 여부를 알아낼 수 있다.
        if not _safe_under(ROOT, path):
            return self._err("forbidden", 403)
        if not path.exists() or not path.is_file():
            return self._err("not found: " + str(path), 404)

        size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return self._stream(lambda: open(path, "rb"), size, ctype,
                            path.name if download else None)

    def _stream(self, opener, size: int, ctype: str, download_name=None):
        """Range 를 처리하며 opener() 가 준 file-like 을 흘려보낸다."""
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                code = 206

        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download_name:
            self.send_header("Content-Disposition",
                             f'attachment; filename="{download_name}"')
        self.end_headers()
        with opener() as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return            # 사용자가 영상 재생을 끊은 것뿐
                remaining -= len(chunk)

    # --- 라우팅 ---
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        p = u.path
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            return self._get(p, q)
        except Exception as e:                    # 서버가 죽으면 GUI 가 먹통
            return self._err(f"{type(e).__name__}: {e}", 500)

    def _get(self, p, q):
        if p in ("/", "/index.html"):
            return self._send_file(GUI_DIR / "index.html")

        if p == "/api/nas_status":
            return self._json(nas_status())

        if p == "/api/bootstrap":
            return self._json({
                "labels": list_label_files(),
                "scenes": list_scene_files(),
                "default_labels": Path(config.LABELS_JSON).name,
                "default_scene": Path(config.SCENE_JSON).name,
                "models": [{"id": k, **v} for k, v in MODELS.items()],
                "nas_clips": nas_clip_count(),
                "runs": [{
                    "name": rel_run_name(d),
                    "kind": run_kind(rel_run_name(d)),
                    "mtime": d.stat().st_mtime,
                    "has_results": bool(glob.glob(str(d / "clip_results*.csv"))),
                    "memo": run_config_of(d).get("key", {}).get("memo", ""),
                    "model": run_config_of(d).get("key", {}).get("model", ""),
                } for d in run_dirs()],
            })

        if p == "/api/aggregate":
            d = RESULTS / q.get("name", "")
            if not d.exists():
                return self._err("no such run", 404)
            return self._json(aggregate_run(d))

        if p == "/api/aggregate_log":
            d = RESULTS / q.get("name", "")
            lp = d / "aggregate_clip.log"
            if not lp.exists():
                return self._err("no aggregate_clip.log", 404)
            return self._json({"text": lp.read_text(encoding="utf-8",
                                                    errors="replace")})

        if p == "/api/run":
            d = RESULTS / q.get("name", "")
            if not d.exists():
                return self._err("no such run", 404)
            lp = ROOT / q["labels"] if q.get("labels") else None
            return self._json(evaluate_run(d, lp))

        if p.startswith("/clip/"):
            return self._send_clip(p[len("/clip/"):])

        if p == "/api/clip":
            uuid = q.get("uuid", "")
            d = RESULTS / q["run"] if q.get("run") else None
            lp = ROOT / (q.get("labels") or Path(config.LABELS_JSON).name)
            return self._json(clip_detail(uuid, d if d and d.exists() else None, lp))

        if p == "/api/search":
            return self._json(search_clips(q))

        if p == "/api/nas_search":
            return self._json(search_nas_clips(q))

        if p == "/api/taxonomy":
            name = q.get("scene") or Path(config.SCENE_JSON).name
            return self._json({"scene": name,
                               "categories": taxonomy_categories(ROOT / name)})

        if p == "/api/labels":
            name = q.get("labels") or Path(config.LABELS_JSON).name
            data = read_labels_file(name)
            clips = data.get("clips", {})
            return self._json({"labels": name, "meta": data.get("_meta", {}),
                               "count": len(clips)})

        if p == "/api/jobs":
            with JOBS_LOCK:
                js = list(JOBS.values())
            return self._json({"jobs": [job_view(j) for j in
                                        sorted(js, key=lambda x: x["started"],
                                               reverse=True)]})

        if p == "/media":
            rel = q.get("path", "")
            return self._send_file(ROOT / rel, download=q.get("dl") == "1")

        if p == "/api/evaluation_log":
            d = RESULTS / q.get("name", "")
            f = d / "evaluation.log"
            if not f.exists():
                return self._err("no evaluation.log", 404)
            return self._json({"text": f.read_text(encoding="utf-8",
                                                   errors="replace")})

        return self._err("unknown endpoint: " + p, 404)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        try:
            return self._post(u.path, self._body())
        except Exception as e:
            return self._err(f"{type(e).__name__}: {e}", 500)

    def _post(self, p, b):
        if p == "/api/label":
            name = b.get("labels") or Path(config.LABELS_JSON).name
            uuid = (b.get("uuid") or "").strip()
            if not uuid:
                return self._err("uuid 가 비었습니다")
            if b.get("delete"):
                ok = delete_label(name, uuid)
                return self._json({"deleted": ok})
            return self._json({"saved": upsert_label(name, uuid, b.get("patch", {}))})

        if p == "/api/category":
            scene = b.get("scene") or Path(config.SCENE_JSON).name
            if b.get("delete"):
                return self._json({"deleted": delete_category(scene, b["name"])})
            cat = {"name": (b.get("name") or "").strip(),
                   "templates": b.get("templates", []),
                   "template_candidates": b.get("template_candidates", [])}
            if not cat["name"]:
                return self._err("카테고리 이름이 비었습니다")
            return self._json({"saved": upsert_category(
                scene, b.get("scenario") or "Dynamic object", cat,
                b.get("old_name"))})

        if p == "/api/job":
            if b.get("stop"):
                return self._json({"stopped": stop_job(b["id"])})
            return self._json(start_job(b))

        return self._err("unknown endpoint: " + p, 404)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="기본 127.0.0.1 (외부 공개하려면 0.0.0.0)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data", default="local",
                    help="영상 소스. local(기본, pav_sample) | nas(NAS 청크 "
                         "zip) | 임의 경로.")
    args = ap.parse_args()

    if args.data not in (None, "local"):
        global CLIP_SOURCE
        from clip_source import make_source
        CLIP_SOURCE = make_source(args.data, root=ROOT)
        print(f"[gui] data={args.data} -> {CLIP_SOURCE.kind}", flush=True)

    GUI_DIR.mkdir(exist_ok=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[gui] http://{args.host}:{args.port}")
    print(f"[gui] 원격이면: ssh -L {args.port}:127.0.0.1:{args.port} <user>@<server>")
    print("[gui] Ctrl-C 로 종료")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[gui] 종료")


if __name__ == "__main__":
    main()

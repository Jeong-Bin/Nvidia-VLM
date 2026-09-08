"""클립 소스 추상화 - 로컬 mp4 / NAS zip 중 어디서 읽든 같은 API 를 준다.

edge_case_mining 의 _read_frames_at 은 mp4 경로 문자열을 받는다. NAS 는
청크 zip 안에 mp4 가 무압축(store)으로 들어 있어 경로가 존재하지 않으므로,
"프레임을 읽어주는 객체"를 한 겹 두고 호출부는 그대로 둔다.

무압축이라 zip 내부 mp4 는 연속된 바이트 구간이다. 그 구간만 파일처럼
노출하면(_ZipSlice) 전체를 메모리로 복사하지 않고 PyAV 가 직접 seek 한다
- 실측 0.36s/클립으로 로컬 cv2(0.44s)보다 오히려 빠르다.
"""
from __future__ import annotations

import io
import json
import os
import re
import struct
import time
import zipfile
from pathlib import Path

CHUNK_GLOB = "*.chunk_*.zip"
# NAS 스냅샷 경로. 날짜 폴더로 버전을 나눠 두었다 - 최신을 가리킨다.
NAS_CAMERA_DIR = "/mnt/nas/NVIDIA_DATASET/20260901/camera"
BUILD_WAIT_S = 900      # 남이 만드는 인덱스를 기다릴 최대 시간
STALE_LOCK_S = 1800     # 이보다 오래된 락은 죽은 프로세스의 것으로 본다


def _chunk_no(path) -> int:
    """파일명에서 청크 번호를 뽑는다. 못 읽으면 -1(맨 앞으로)."""
    m = re.search(r"chunk_(\d+)", Path(path).name)
    return int(m.group(1)) if m else -1


class _ZipSlice(io.RawIOBase):
    """zip 안의 한 구간을 독립된 파일처럼 보여준다 (store 방식 전제)."""

    def __init__(self, path: Path, offset: int, size: int):
        self._f = open(path, "rb")
        self._off = offset
        self._size = size
        self._pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, off, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = off
        elif whence == io.SEEK_CUR:
            self._pos += off
        else:
            self._pos = self._size + off
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def tell(self):
        return self._pos

    def readinto(self, b):
        left = self._size - self._pos
        if left <= 0:
            return 0
        self._f.seek(self._off + self._pos)
        n = self._f.readinto(memoryview(b)[:min(len(b), left)])
        self._pos += n or 0
        return n or 0

    def close(self):
        try:
            self._f.close()
        finally:
            super().close()


def _data_offset(fh, info: zipfile.ZipInfo) -> int:
    """로컬 헤더를 건너뛴 실제 데이터 시작 오프셋.

    fh 는 이미 열린 파일 핸들. SMB 너머에서는 zip 마다 다시 여는 비용이
    커서(왕복 지연이 청크 수만큼 쌓인다) 핸들을 재사용한다.
    """
    fh.seek(info.header_offset + 26)
    nlen, elen = struct.unpack("<HH", fh.read(4))
    return info.header_offset + 30 + nlen + elen


class LocalSource:
    """pav_sample 처럼 mp4 가 파일로 풀려 있는 경우."""

    kind = "local"

    def __init__(self, camera_dir: Path):
        self.camera_dir = Path(camera_dir)

    def path(self, view: str, uuid: str) -> Path:
        return self.camera_dir / view / f"{uuid}.{view}.mp4"

    def exists(self, view: str, uuid: str) -> bool:
        return self.path(view, uuid).exists()

    def open_video(self, view: str, uuid: str):
        """PyAV/cv2 에 넘길 수 있는 것을 준다 (여기서는 경로 문자열)."""
        return str(self.path(view, uuid))

    def uuids(self, view: str):
        d = self.camera_dir / view
        return sorted(p.name.split(".")[0] for p in d.glob("*.mp4"))

    def locate_uuid(self, view: str, uuid: str) -> bool:
        """uuid 하나가 있는지만 본다. 로컬은 stat 한 번이라 exists 와 같다."""
        return self.exists(view, uuid)


class ZipSource:
    """NAS 처럼 청크 zip 안에 mp4 가 들어 있는 경우.

    uuid -> (zip 경로, 오프셋, 길이) 인덱스를 한 번 만들어 캐시에 저장한다.
    383개 zip 인덱싱에 ~80s 걸리므로 캐시가 없으면 첫 실행만 느리다.
    """

    kind = "zip"

    def __init__(self, camera_dir: Path, cache: Path | None = None):
        self.camera_dir = Path(camera_dir)
        self.cache = Path(cache) if cache else self.camera_dir / ".clip_index.json"
        self._idx = None

    @staticmethod
    def _scan_one(args):
        """zip 하나에서 (키, 이름) 목록을 뽑는다.

        데이터 오프셋은 여기서 구하지 않는다. 오프셋을 알려면 파일마다
        로컬 헤더로 seek 해야 하는데, SMB 에서는 그 왕복이 쌓여 청크당
        12.8s 가 걸린다(중앙 디렉터리만 읽으면 0.9s). 오프셋은 그 클립을
        실제로 열 때 한 번만 구하면 된다.
        """
        view, zp = args
        out = []
        try:
            with zipfile.ZipFile(zp) as z:
                for info in z.infolist():
                    if not info.filename.endswith(".mp4"):
                        continue
                    uuid = info.filename.split(".")[0]
                    out.append((f"{view}/{uuid}", [str(zp), info.filename]))
        except (zipfile.BadZipFile, OSError):
            pass  # 다운로드 중이라 아직 불완전한 청크일 수 있다
        return out

    def _build(self, progress: bool = True):
        """청크 zip 을 훑어 uuid -> 위치 인덱스를 만든다.

        직렬로 읽는다. SMB 는 동시 요청을 늘리면 오히려 느려진다 - 스레드
        8개로 겹쳐봤더니 0.78s/zip 이 4.5s/zip 으로 5배 나빠졌다(다운로드가
        같은 회선을 쓰는 중이라 경합이 더 심했다).
        """
        jobs = []
        for view_dir in sorted(self.camera_dir.iterdir()):
            if not view_dir.is_dir():
                continue
            # 청크 번호 오름차순. 파일명 정렬로도 같은 순서가 나오지만,
            # 자리수가 다른 이름이 섞이면 어긋난다 - 중복 해소가 이 순서에
            # 달려 있으므로(아래 idx 주석) 숫자로 못박는다.
            jobs += [(view_dir.name, zp)
                     for zp in sorted(view_dir.glob(CHUNK_GLOB),
                                      key=_chunk_no)]

        # uuid -> 위치. 같은 uuid 가 여러 청크에 들어 있는 경우가 있어
        # (원본 저장소에 1,180건의 중복 사본이 있다 - 바이트 단위로 동일)
        # 뒤에 스캔한 것이 앞을 덮어쓴다. 청크 번호 오름차순으로 훑으므로
        # 항상 "가장 높은 번호"가 남고, 이는 clip_index.parquet 이 정본으로
        # 지정한 쪽과 일치한다(1,180건 전부 확인). 덕분에 추론 대상 목록에
        # 같은 클립이 두 번 들어가지 않는다.
        idx = {}
        if not jobs:
            return idx
        t0 = time.time()
        for done, job in enumerate(jobs, 1):
            idx.update(self._scan_one(job))
            if progress and (done % 50 == 0 or done == len(jobs)):
                el = time.time() - t0
                eta = el / done * (len(jobs) - done)
                print(f"  [clip_source] 인덱싱 {done}/{len(jobs)} "
                      f"({el:.0f}s, 남은 ~{eta:.0f}s)", flush=True)
        return idx

    def _fingerprint(self):
        """청크 구성이 바뀌었는지 판별할 값 - (zip 개수, 최신 mtime).

        전체를 열어보지 않고 디렉터리만 훑으므로 1초 이내다. 청크가
        추가/삭제되거나 다시 받아지면 둘 중 하나는 반드시 달라진다.
        """
        n, newest = 0, 0.0
        for view_dir in sorted(self.camera_dir.iterdir()):
            if not view_dir.is_dir():
                continue
            for zp in view_dir.glob(CHUNK_GLOB):
                n += 1
                try:
                    newest = max(newest, zp.stat().st_mtime)
                except OSError:
                    pass
        return {"zips": n, "newest": round(newest, 3)}

    @property
    def index(self):
        if self._idx is not None:
            return self._idx

        fp = self._fingerprint()
        if self.cache.exists():
            try:
                blob = json.loads(self.cache.read_text())
            except (json.JSONDecodeError, OSError):
                blob = None
            # 옛 형식(맵만 저장)은 지문이 없다 - 새로 만든다.
            if isinstance(blob, dict) and blob.get("fingerprint") == fp:
                self._idx = blob["index"]
                return self._idx
            if blob is not None:
                old_n = (blob.get("fingerprint") or {}).get("zips", "?")
                print(f"  [clip_source] 청크 구성이 바뀌었습니다 "
                      f"({old_n} -> {fp['zips']}개). 인덱스를 다시 만듭니다.",
                      flush=True)

        # 샤드 8개가 동시에 시작하면 전부 "캐시 없음"을 보고 각자 1000개
        # zip 을 훑는다 - 같은 일을 8중으로 하면서 SMB 를 서로 막아 8배가
        # 아니라 수십 배 느려진다. 락을 잡은 하나만 만들고 나머지는 기다린다.
        lock = self.cache.with_suffix(".lock")
        if self._acquire(lock):
            try:
                self._idx = self._build()
                try:
                    self.cache.write_text(json.dumps(
                        {"fingerprint": fp, "index": self._idx}))
                except OSError:
                    pass  # 읽기전용 마운트여도 동작은 해야 한다
            finally:
                lock.unlink(missing_ok=True)
            return self._idx

        print("  [clip_source] 다른 프로세스가 인덱스를 만드는 중 - 기다립니다",
              flush=True)
        for _ in range(BUILD_WAIT_S):
            time.sleep(1)
            if self.cache.exists():
                try:
                    blob = json.loads(self.cache.read_text())
                except (json.JSONDecodeError, OSError):
                    continue  # 아직 쓰는 중
                if isinstance(blob, dict) and blob.get("fingerprint") == fp:
                    self._idx = blob["index"]
                    return self._idx
            if not lock.exists():
                break  # 만들던 쪽이 죽었다 - 직접 만든다
        self._idx = self._build()
        return self._idx

    @staticmethod
    def _acquire(lock: Path) -> bool:
        """락을 잡으면 True. 남의 락이 오래됐으면 뺏는다(죽은 프로세스)."""
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > STALE_LOCK_S:
                    lock.unlink(missing_ok=True)
                    return ZipSource._acquire(lock)
            except OSError:
                pass
            return False
        except OSError:
            return True  # 락을 못 만드는 환경이면 그냥 직접 만든다

    def exists(self, view: str, uuid: str) -> bool:
        return f"{view}/{uuid}" in self.index

    def open_video(self, view: str, uuid: str):
        """PyAV 가 바로 받는 file-like. cv2 는 이걸 못 받는다."""
        zp, name = self.index[f"{view}/{uuid}"]
        with open(zp, "rb") as fh, zipfile.ZipFile(fh) as z:
            info = z.getinfo(name)
            off = _data_offset(fh, info)
        return io.BufferedReader(_ZipSlice(Path(zp), off, info.file_size))

    def size_of(self, view: str, uuid: str) -> int:
        """mp4 바이트 크기. Range 응답에 필요하다."""
        zp, name = self.index[f"{view}/{uuid}"]
        with zipfile.ZipFile(zp) as z:
            return z.getinfo(name).file_size

    def uuids(self, view: str):
        pre = f"{view}/"
        return sorted(k[len(pre):] for k in self.index if k.startswith(pre))

    def locate_uuid(self, view: str, uuid: str) -> bool:
        """uuid 하나를 인덱스 없이 찾아 _idx 에 심는다 (찾으면 True).

        단일 클립 조회 때문에 전체 인덱스(3145 zip, ~1시간)를 만들 수는 없다.
        uuid 를 이미 아는 경우엔 zip 을 하나씩 열어 중앙 디렉터리만 보고
        찾는 즉시 멈추면 된다 - 평균 절반만 보므로 실측 30초 안쪽이고,
        캐시가 이미 유효하면 그것부터 쓰므로 즉시 끝난다.

        찾은 항목만 self._idx 에 넣어 두면 open_video/size_of 가 그대로
        동작한다. 전체 인덱스인 척하지 않는 게 중요하다 - uuids() 는
        여전히 index 프로퍼티를 타서 정상적으로 전체를 만든다.
        """
        key = f"{view}/{uuid}"
        if self._idx is not None and key in self._idx:
            return True

        # 유효한 캐시가 있으면 그게 가장 빠르다.
        try:
            if self.cache.exists():
                blob = json.loads(self.cache.read_text())
                if (isinstance(blob, dict)
                        and blob.get("fingerprint") == self._fingerprint()
                        and key in blob.get("index", {})):
                    self._idx = blob["index"]
                    return True
        except (json.JSONDecodeError, OSError):
            pass

        vd = self.camera_dir / view
        if not vd.is_dir():
            return False
        for zp in sorted(vd.glob(CHUNK_GLOB)):
            for k, v in self._scan_one((view, zp)):
                if k == key:
                    if self._idx is None:
                        self._idx = {}
                    self._idx[k] = v
                    return True
        return False


def make_source(spec: str | None = None, *, root: Path | None = None):
    """--data 플래그 한 개로 소스를 고른다.

    spec:
      "local" 또는 None  -> ROOT/pav_sample/camera
      "nas"              -> NAS 기본 경로의 청크 zip
      그 밖의 문자열      -> 경로로 해석, mp4 가 보이면 local, zip 이면 zip
    """
    root = Path(root or Path(__file__).resolve().parent)
    if spec in (None, "local"):
        return LocalSource(root / "pav_sample" / "camera")
    if spec == "nas":
        return ZipSource(Path(NAS_CAMERA_DIR))
    p = Path(spec)
    if not p.exists():
        raise FileNotFoundError(f"데이터 경로 없음: {p}")
    if any(p.glob(f"*/{CHUNK_GLOB}")) or any(p.glob(CHUNK_GLOB)):
        return ZipSource(p)
    return LocalSource(p)

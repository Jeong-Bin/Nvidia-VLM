#!/usr/bin/env python
"""camera_front_wide_120fov 청크를 2단계로 NAS에 적재한다.

1단계: 로컬 NVMe에 병렬로 내려받는다 (워커 4개가 최적 - 측정치 30MB/s).
2단계: 배치가 차면 rsync로 NAS에 순차 전송하고 로컬 사본을 지운다.

CIFS에 직접 쓰면 xet 백엔드의 랜덤 오프셋 쓰기 때문에 느려지므로
배치를 로컬에 모은 뒤 순차로 밀어넣는다. 중단 후 재실행하면
NAS에 이미 있는 청크는 건너뛴다.
"""
import argparse
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
CAM = "camera_front_wide_120fov"
REL_DIR = f"camera/{CAM}"
NAS_DIR = Path("/mnt/nas/NVIDIA_DATASET")
STAGE_DIR = Path("/home/etri/Jeongbin/Nvidia-VLM/.stage")

_print_lock = threading.Lock()
_batch_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def verify(path):
    """zip 헤더와 목록을 읽어 온전한 파일인지 확인한다."""
    try:
        with zipfile.ZipFile(path) as z:
            return len(z.namelist()) > 0
    except (zipfile.BadZipFile, OSError):
        return False


def flush_batch(batch):
    """배치를 NAS로 옮기고 로컬에서 지운다. 호출자가 락을 쥔다."""
    if not batch:
        return 0
    dest = NAS_DIR / REL_DIR
    dest.mkdir(parents=True, exist_ok=True)
    gb = sum(p.stat().st_size for p in batch) / 2**30
    log(f"  -> NAS 전송 {len(batch)}개 ({gb:.1f} GB) ...")
    t0 = time.time()
    r = subprocess.run(
        ["rsync", "-a", "--no-perms", "--no-owner", "--no-group",
         "--remove-source-files", *[str(p) for p in batch], str(dest) + "/"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        log(f"  !! rsync 실패 (rc={r.returncode}): {r.stderr[:300]}")
        return 0
    dt = time.time() - t0
    log(f"  -> 전송 완료 {gb*1024/dt:.0f} MB/s, 로컬 정리됨")
    return len(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=363)
    ap.add_argument("--end", type=int, default=999)
    ap.add_argument("--workers", type=int, default=4,
                    help="동시 다운로드 수 (측정상 4가 최적, 6은 오히려 느림)")
    ap.add_argument("--batch", type=int, default=20,
                    help="로컬에 모았다가 한 번에 옮길 청크 수")
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    remote = set(HfApi().list_repo_files(REPO, repo_type="dataset"))

    todo = []
    for n in range(args.start, args.end + 1):
        rel = f"{REL_DIR}/{CAM}.chunk_{n:04d}.zip"
        if rel in remote and not (NAS_DIR / rel).exists():
            todo.append((n, rel))

    log(f"대상 {len(todo)}개 (범위 {args.start}~{args.end}, "
        f"워커 {args.workers}, 배치 {args.batch})")

    state = {"batch": [], "done": 0, "got": 0}
    failed = []
    t0 = time.time()

    def work(item):
        n, rel = item
        name = f"{CAM}.chunk_{n:04d}.zip"
        local = STAGE_DIR / name
        for attempt in range(1, args.retries + 1):
            try:
                p = hf_hub_download(REPO, rel, repo_type="dataset",
                                    local_dir=STAGE_DIR / "_dl")
                if not verify(p):
                    raise OSError("zip 검증 실패")
                shutil.move(str(p), local)
                break
            except (HfHubHTTPError, OSError) as e:
                log(f"재시도 {attempt}/{args.retries} chunk_{n:04d}: "
                    f"{type(e).__name__}")
                time.sleep(min(2 ** attempt, 30))
        else:
            failed.append(rel)
            log(f"실패 chunk_{n:04d}")
            return

        with _batch_lock:
            state["got"] += 1
            state["batch"].append(local)
            i, total = state["got"], len(todo)
            el = time.time() - t0
            rate = state["got"] / el * 3600 if el else 0
            eta = (total - i) / rate if rate else 0
            log(f"[{i}/{total}] 받음 chunk_{n:04d} "
                f"({local.stat().st_size/2**30:.2f} GB) "
                f"| {rate:.0f} chunk/h, 남은 시간 ~{eta:.1f}h")
            if len(state["batch"]) >= args.batch:
                state["done"] += flush_batch(state["batch"])
                state["batch"] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))

    with _batch_lock:
        state["done"] += flush_batch(state["batch"])
        state["batch"] = []

    el = time.time() - t0
    log(f"\n완료: NAS 적재 {state['done']}개, 실패 {len(failed)}개, "
        f"소요 {el/3600:.1f}시간")
    for f in failed:
        log(f"  {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

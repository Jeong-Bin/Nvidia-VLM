#!/usr/bin/env python3
"""NVIDIA PhysicalAI-Autonomous-Vehicles 에서 클립을 추가로 내려받아
기존 pav_sample/ 트리에 합친다.

기존 로컬 데이터 = clip_index.parquet 기준 chunk 0~4 (500 클립).
기본 설정은 chunk 5~19 (1,497 클립)를 추가로 받아 총 1,997 클립을 만든다.

받는 것:
  - camera/{front_wide, cross_left, cross_right}_...  (전방 3뷰만)
  - labels/egomotion, labels/obstacle.offline

각 청크는 zip 하나이고 내부는 flat 구조(`{uuid}.{view}.mp4` 등)라
기존 폴더에 그대로 풀면 병합된다.

진행 상황은 stdout 과 --log 파일에 동시에 기록한다(청크 단위).
중단 후 재실행하면 이미 받은 청크는 건너뛴다(HF 캐시 + 추출 완료 마커).
"""
import argparse
import json
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
ROOT = Path(__file__).resolve().parent

# 전방 3뷰만 (rear/tele/lidar/radar 는 받지 않는다)
CAMERA_VIEWS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
]
LABEL_KINDS = ["egomotion", "obstacle.offline"]


def repo_path(kind: str, chunk: int) -> str:
    """HF 저장소 안에서의 파일 경로."""
    if kind in CAMERA_VIEWS:
        return f"camera/{kind}/{kind}.chunk_{chunk:04d}.zip"
    return f"labels/{kind}/{kind}.chunk_{chunk:04d}.zip"


def dest_dir(kind: str) -> Path:
    """로컬에서 풀어놓을 위치 (기존 트리와 동일)."""
    if kind in CAMERA_VIEWS:
        return ROOT / "pav_sample" / "camera" / kind
    return ROOT / "pav_sample" / "labels" / kind


class Logger:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, msg: str):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")


def human(n: float) -> str:
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-chunk", type=int, default=5)
    ap.add_argument("--end-chunk", type=int, default=19, help="inclusive")
    ap.add_argument("--cache-dir", default=str(ROOT / ".hf_download_cache"),
                    help="zip 임시 저장 위치 (추출 후 --keep-zip 없으면 삭제)")
    ap.add_argument("--keep-zip", action="store_true",
                    help="추출 후에도 zip 을 남긴다 (디스크 2배 필요)")
    ap.add_argument("--log", default=str(ROOT / "download.log"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log = Logger(Path(args.log))
    chunks = list(range(args.start_chunk, args.end_chunk + 1))
    kinds = CAMERA_VIEWS + LABEL_KINDS
    tasks = [(k, c) for c in chunks for k in kinds]

    state_path = ROOT / ".download_state.json"
    done = set()
    if state_path.exists():
        done = set(tuple(x) for x in json.loads(state_path.read_text()))

    log("=" * 70)
    log(f"chunks {args.start_chunk}..{args.end_chunk} ({len(chunks)} chunks), "
        f"kinds={kinds}")
    log(f"total tasks: {len(tasks)}, already done: {len(done)}")
    log(f"cache: {args.cache_dir}  keep_zip={args.keep_zip}")
    if args.dry_run:
        for k, c in tasks[:10]:
            log(f"  would fetch {repo_path(k, c)} -> {dest_dir(k)}")
        log("dry-run, exiting")
        return

    for d in kinds:
        dest_dir(d).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_done = 0
    total_bytes = 0
    for i, (kind, chunk) in enumerate(tasks, 1):
        key = (kind, chunk)
        if key in done:
            continue
        rp = repo_path(kind, chunk)
        try:
            zp = hf_hub_download(REPO, rp, repo_type="dataset",
                                 cache_dir=args.cache_dir)
            size = os.path.getsize(zp)
            with zipfile.ZipFile(zp) as z:
                members = z.namelist()
                z.extractall(dest_dir(kind))
            total_bytes += size
            n_done += 1
            if not args.keep_zip:
                # HF 캐시의 blob 실체를 지운다 (symlink 경유)
                real = os.path.realpath(zp)
                for p in {real, zp}:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            el = time.time() - t0
            rate = total_bytes / el if el else 0
            remain = len(tasks) - i
            eta = remain * (el / n_done) if n_done else 0
            log(f"[{i}/{len(tasks)}] {kind} chunk_{chunk:04d}  "
                f"{human(size)}  {len(members)} files  "
                f"| {human(rate)}/s  elapsed {el/60:.1f}m  ETA {eta/60:.1f}m")
            done.add(key)
            state_path.write_text(json.dumps([list(x) for x in done]))
        except Exception as e:
            log(f"[{i}/{len(tasks)}] !! FAILED {rp}: {type(e).__name__}: {e}")

    log(f"finished: {n_done} new chunks, {human(total_bytes)} downloaded, "
        f"{(time.time()-t0)/60:.1f} min")

    # 최종 클립 수 확인
    for v in CAMERA_VIEWS:
        n = len(list(dest_dir(v).glob("*.mp4")))
        log(f"  {v}: {n} clips")
    for k in LABEL_KINDS:
        n = len(list(dest_dir(k).glob("*.parquet")))
        log(f"  {k}: {n} parquet")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""camera_front_wide_120fov 청크를 지정 범위만큼 NAS로 내려받는다.

이미 있는 파일은 건너뛰므로 중단 후 재실행하면 이어받기가 된다.
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
CAM = "camera_front_wide_120fov"
LOCAL_DIR = Path("/mnt/nas/NVIDIA_DATASET")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=342)
    ap.add_argument("--end", type=int, default=1000)
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()

    from huggingface_hub import HfApi
    remote = set(HfApi().list_repo_files(REPO, repo_type="dataset"))

    todo = []
    for n in range(args.start, args.end + 1):
        rel = f"camera/{CAM}/{CAM}.chunk_{n:04d}.zip"
        if rel not in remote:
            continue  # 청크 번호는 연속이 아니다
        if (LOCAL_DIR / rel).exists():
            continue
        todo.append(rel)

    print(f"대상 {len(todo)}개 (범위 {args.start}~{args.end})", flush=True)

    failed = []
    for i, rel in enumerate(todo, 1):
        for attempt in range(1, args.retries + 1):
            try:
                hf_hub_download(
                    REPO, rel, repo_type="dataset",
                    local_dir=LOCAL_DIR,
                )
                size = (LOCAL_DIR / rel).stat().st_size / 2**30
                print(f"[{i}/{len(todo)}] OK {rel} ({size:.2f} GB)", flush=True)
                break
            except (HfHubHTTPError, OSError) as e:
                print(f"[{i}/{len(todo)}] 재시도 {attempt}/{args.retries} "
                      f"{rel}: {type(e).__name__}", flush=True)
        else:
            failed.append(rel)
            print(f"[{i}/{len(todo)}] 실패 {rel}", flush=True)

    print(f"\n완료. 실패 {len(failed)}개", flush=True)
    for f in failed:
        print("  ", f, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

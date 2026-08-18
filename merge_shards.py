#!/usr/bin/env python3
"""샤드 CSV 를 하나로 합치고 원본을 지운다.

왜 샤드로 나눠 쓰다가 합치는가:
  8개 프로세스가 한 CSV 에 동시에 쓰면 행이 섞여 깨진다. 그래서 추론 중에는
  각자 자기 파일에 쓰고(락 불필요), 끝난 뒤 여기서 합친다.

왜 원본을 지우는가:
  실행 폴더마다 CSV 가 9개씩 쌓이면 어느 것이 최종본인지 헷갈린다. 병합이
  성공한 뒤에만 지우므로, 중간에 실패하면 샤드 파일이 남아 복구할 수 있다.

Usage:
  python merge_shards.py --run-dir results/20260813_171524_eval
  python merge_shards.py --run-dir <dir> --keep-shards
"""
import argparse
import glob
from pathlib import Path

import pandas as pd

MERGED_NAME = "clip_results_all.csv"
SHARD_GLOB = "clip_results_shard_*.csv"


def merge_shards(run_dir: Path, keep_shards: bool = False, quiet: bool = False):
    """샤드 CSV -> clip_results_all.csv. (병합본 경로, 행 수) 를 돌려준다.

    이미 병합본만 있고 샤드가 없으면 아무것도 하지 않는다 - 같은 폴더에
    두 번 돌려도 안전하다.
    """
    run_dir = Path(run_dir)
    shards = sorted(glob.glob(str(run_dir / SHARD_GLOB)))
    merged = run_dir / MERGED_NAME

    if not shards:
        if merged.exists():
            n = len(pd.read_csv(merged))
            if not quiet:
                print(f"[merge] already merged: {merged.name} ({n} rows)")
            return merged, n
        if not quiet:
            print(f"[merge] no {SHARD_GLOB} in {run_dir}")
        return None, 0

    frames = [pd.read_csv(f) for f in shards]
    df = pd.concat(frames, ignore_index=True)

    # 샤드는 서로 겹치지 않아야 한다. 겹치면 같은 클립이 두 번 세어지므로
    # 조용히 넘기지 않고 알린 뒤 중복을 제거한다.
    n_raw = len(df)
    if "uuid" in df.columns:
        df = df.drop_duplicates(subset="uuid", keep="first")
        if len(df) != n_raw and not quiet:
            print(f"[merge] warn: {n_raw - len(df)} duplicate uuid(s) dropped")
        df = df.sort_values("uuid", ignore_index=True)

    df.to_csv(merged, index=False)
    if not quiet:
        print(f"[merge] {len(shards)} shards -> {merged.name} ({len(df)} rows)")

    if not keep_shards:
        # 병합본이 실제로 쓰인 뒤에만 지운다.
        for f in shards:
            Path(f).unlink()
        if not quiet:
            print(f"[merge] removed {len(shards)} shard file(s)")
    return merged, len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--keep-shards", action="store_true",
                    help="병합 후에도 샤드 CSV 를 남긴다 (기본은 삭제)")
    args = ap.parse_args()
    merge_shards(Path(args.run_dir), keep_shards=args.keep_shards)


if __name__ == "__main__":
    main()

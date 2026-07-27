#!/usr/bin/env python3
"""8개 shard의 진행 상황(각 results_shard_N.csv 의 행 수)을 합산해
하나의 통합 프로그레스 바로 보여준다.

각 shard 프로세스는 여전히 독립적으로 자기 CSV/로그에 쓰지만(안전),
이 스크립트는 그 파일들을 주기적으로 읽어 전체 진행률만 계산한다 -
shard 프로세스에 개입하지 않는 순수 관찰자.

모든 shard 의 PID 가 종료되면 자동으로 멈춘다.

Usage:
  python monitor_progress.py --run-dir results/20260724_104830 \
      --num-shards 8 --total-units 5000 --pids 111 222 333 ...
"""
import argparse
import csv
import time
from pathlib import Path

from tqdm import tqdm


def count_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            return max(0, sum(1 for _ in csv.reader(f)) - 1)  # 헤더 제외
    except Exception:
        return 0


def pids_alive(pids):
    alive = []
    for pid in pids:
        try:
            import os
            os.kill(pid, 0)
        except OSError:
            continue
        alive.append(pid)
    return alive


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--total-units", type=int, required=True)
    ap.add_argument("--pids", type=int, nargs="+", required=True,
                    help="감시할 shard 프로세스 PID 목록 (전부 종료되면 모니터도 종료)")
    ap.add_argument("--poll-interval", type=float, default=2.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    csv_paths = [run_dir / f"results_shard_{g}.csv" for g in range(args.num_shards)]

    pbar = tqdm(total=args.total_units, unit="unit", dynamic_ncols=True,
               desc="[all shards]", mininterval=1.0, smoothing=0.1)
    last_done = 0
    while True:
        done = sum(count_rows(p) for p in csv_paths)
        if done > last_done:
            pbar.update(done - last_done)
            last_done = done

        alive = pids_alive(args.pids)
        if not alive:
            break
        if last_done >= args.total_units:
            break
        time.sleep(args.poll_interval)

    # 마지막 한번 더 집계 (프로세스 종료 직후 파일 flush 반영)
    done = sum(count_rows(p) for p in csv_paths)
    if done > last_done:
        pbar.update(done - last_done)
    pbar.close()

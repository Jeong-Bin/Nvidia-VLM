#!/usr/bin/env python3
"""shard별 결과 CSV들을 합쳐 최종 카테고리 집계를 만든다 (top-1 기준).

Usage:
  python aggregate.py                       # results/ 아래 가장 최근 실행 폴더 자동 사용
  python aggregate.py --run-dir results/20260724_104830
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def latest_run_dir():
    candidates = sorted((ROOT / "results").glob("[0-9]" * 8 + "_" + "[0-9]" * 6))
    return candidates[-1] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="results/<timestamp> 실행 폴더. 생략 시 가장 최근 실행 자동 사용")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    if run_dir is None or not run_dir.exists():
        print("no run directory found under results/ (run run_all.sh first, or pass --run-dir)")
        return
    print(f"[info] aggregating run: {run_dir}\n")

    files = sorted(glob.glob(str(run_dir / "results_shard_*.csv")))
    if not files:
        print(f"no results_shard_*.csv found in {run_dir}")
        return
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    out_csv = run_dir / "edge_case_results_all.csv"
    df.to_csv(out_csv, index=False)

    # 라벨 순서 로드 (배열 스키마, normal/special 모두 동일 구조)
    scene = json.loads((ROOT / "scene_category.json").read_text(encoding="utf-8"))
    normal_cats, special_cats = [], []
    for scen in scene["normal"]["scenarios"]:
        for cat in scen["categories"]:
            normal_cats.append((scen["name"], cat["name"]))
    for scen in scene["special"]["scenarios"]:
        for cat in scen["categories"]:
            special_cats.append((scen["name"], cat["name"]))

    total = len(df)
    n_clips = df["uuid"].nunique() if "uuid" in df.columns else None
    counts = Counter(df["top1_category"].fillna("OOD").replace("", "OOD"))

    def pct(n):
        return 100 * n / total if total else 0.0

    if n_clips:
        print(f"총 판정 단위 수: {total}  ({n_clips} clips x ~{total // n_clips} timestamps)\n")
    else:
        print(f"총 판정 단위 수: {total}\n")
    print(f"{'SCENARIO':<38}{'CATEGORY':<26}{'COUNT':>7}{'%':>8}")
    print("-" * 79)
    special_total = 0
    for scenario, c in special_cats:
        n = counts.get(c, 0)
        special_total += n
        print(f"{scenario:<38}{c:<26}{n:>7}{pct(n):>7.1f}%")
    print("-" * 79)
    print(f"{'(special edge-case 합계, top-1 기준)':<64}{special_total:>7}{pct(special_total):>7.1f}%\n")

    normal_total = 0
    for scenario, c in normal_cats:
        n = counts.get(c, 0)
        normal_total += n
        print(f"{scenario:<38}{c:<26}{n:>7}{pct(n):>7.1f}%")
    print(f"{'(normal 합계)':<64}{normal_total:>7}{pct(normal_total):>7.1f}%")
    ood_n = counts.get('OOD', 0)
    print(f"{'OOD':<64}{ood_n:>7}{pct(ood_n):>7.1f}%")

    known = {c for _, c in special_cats} | {c for _, c in normal_cats} | {"OOD"}
    others = {k: v for k, v in counts.items() if k not in known}
    if others:
        others_n = sum(others.values())
        print(f"{'(기타/미분류)':<64}{others_n:>7}{pct(others_n):>7.1f}%  {others}")

    print(f"\n[저장] {out_csv}")
    print(f"[시각화] {run_dir}/<category>/ (special/OOD 로 분류된 판정 단위만, "
          f"파일명: {{uuid}}_f{{frame_idx}}.png)")


if __name__ == "__main__":
    main()

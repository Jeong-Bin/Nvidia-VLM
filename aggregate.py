#!/usr/bin/env python3
"""shard별 결과 CSV들을 합쳐 집계한다 (멀티라벨).

두 가지를 집계한다:
  1) verdict 버킷 - Special / Normal_but / Normal (서로 배타적, 합계 100%)
  2) special 카테고리별 출현 빈도 - 멀티라벨이라 한 판정 단위가 여러 카테고리에
     동시에 잡힐 수 있으므로 합계가 100%를 넘을 수 있다.

집계 결과는 화면과 <run_dir>/aggregate.log 에 함께 기록한다.

Usage:
  python aggregate.py                       # results/ 아래 가장 최근 실행 폴더 자동 사용
  python aggregate.py --run-dir results/20260728_105752
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SCENE_JSON = ROOT / "scene_category_B.json"

BUCKETS = ["Special", "Normal_but", "Normal"]


def latest_run_dir():
    candidates = sorted((ROOT / "results").glob("[0-9]" * 8 + "_" + "[0-9]" * 6))
    return candidates[-1] if candidates else None


def load_special_categories(scene_json: Path):
    """[(scenario, category), ...] - 집계 표의 행 순서를 json 정의 순서로 고정."""
    scene = json.loads(scene_json.read_text(encoding="utf-8"))
    out = []
    for scen in scene["special"]["scenarios"]:
        for cat in scen["categories"]:
            out.append((scen["name"], cat["name"]))
    return out


def split_categories(cell) -> list[str]:
    """CSV 의 categories 칸("A|B") -> ["A", "B"]. 비어있으면 []."""
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [c for c in (p.strip() for p in cell.split("|")) if c]


class Tee:
    """화면과 로그 파일에 동시에 쓴다."""

    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, line: str = ""):
        print(line)
        self.f.write(line + "\n")

    def close(self):
        self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="results/<timestamp> 실행 폴더. 생략 시 가장 최근 실행 자동 사용")
    ap.add_argument("--log-name", default="aggregate.log",
                    help="집계 로그 파일명 (실행 폴더 안에 저장)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    if run_dir is None or not run_dir.exists():
        print("no run directory found under results/ (run run_all.sh first, or pass --run-dir)")
        return

    files = sorted(glob.glob(str(run_dir / "results_shard_*.csv")))
    if not files:
        # 단일 프로세스 실행이면 shard 없이 results.csv 하나만 있을 수 있다
        files = sorted(glob.glob(str(run_dir / "results*.csv")))
    if not files:
        print(f"no results CSV found in {run_dir}")
        return

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out_csv = run_dir / "edge_case_results_all.csv"
    df.to_csv(out_csv, index=False)

    log = Tee(run_dir / args.log_name)
    log(f"[info] run dir : {run_dir}")
    log(f"[info] shards   : {len(files)} CSV")

    # 구 스키마(top1_category 기반, 2단계 파이프라인) CSV 를 넣으면 새 컬럼이
    # 없어 전부 Normal 로 보이므로, 조용히 0 을 내지 말고 분명히 알린다.
    missing = [c for c in ("bucket", "categories") if c not in df.columns]
    if missing:
        log(f"[error] this CSV is missing {missing} - it looks like output from the "
            f"old two-stage pipeline (top1_category), which this script no longer "
            f"aggregates. Re-run the inference to get the new schema.")
        log.close()
        return

    total = len(df)
    n_clips = df["uuid"].nunique() if "uuid" in df.columns else 0
    if total == 0:
        log("[warn] no rows")
        log.close()
        return

    def pct(n):
        return 100 * n / total

    log(f"[info] units    : {total}"
        + (f"  ({n_clips} clips)" if n_clips else ""))
    if "parse_ok" in df.columns:
        n_fail = int((df["parse_ok"] == 0).sum())
        if n_fail:
            log(f"[warn] JSON parse failed on {n_fail} units ({pct(n_fail):.1f}%)")

    # --- 1) verdict 버킷 (배타적, 합계 100%) ---
    bucket = df.get("bucket")
    buckets = (bucket.fillna("").replace("", "Normal") if bucket is not None
               else pd.Series(["Normal"] * total))
    bcounts = Counter(buckets)

    log("")
    log("=" * 62)
    log("VERDICT BUCKETS  (mutually exclusive)")
    log("=" * 62)
    log(f"{'BUCKET':<24}{'COUNT':>10}{'%':>10}")
    log("-" * 62)
    for b in BUCKETS:
        log(f"{b:<24}{bcounts.get(b, 0):>10}{pct(bcounts.get(b, 0)):>9.1f}%")
    log("-" * 62)
    log(f"{'TOTAL':<24}{total:>10}{100.0:>9.1f}%")

    reviewable = bcounts.get("Special", 0) + bcounts.get("Normal_but", 0)
    log(f"{'(review candidates)':<24}{reviewable:>10}{pct(reviewable):>9.1f}%")

    # --- 2) special 카테고리 출현 빈도 (멀티라벨, 합계 100% 초과 가능) ---
    cat_lists = df["categories"].apply(split_categories) if "categories" in df else []
    ccounts = Counter()
    for lst in cat_lists:
        ccounts.update(lst)
    n_labels = sum(ccounts.values())
    n_labeled_units = int(sum(1 for lst in cat_lists if lst))

    log("")
    log("=" * 62)
    log("SPECIAL CATEGORY FREQUENCY  (multi-label: % may exceed 100)")
    log("=" * 62)
    log(f"{'SCENARIO':<22}{'CATEGORY':<24}{'COUNT':>8}{'%':>8}")
    log("-" * 62)
    known = set()
    for scenario, cat in load_special_categories(SCENE_JSON):
        known.add(cat)
        n = ccounts.get(cat, 0)
        log(f"{scenario:<22}{cat:<24}{n:>8}{pct(n):>7.1f}%")

    # json 에 없는 이름이 섞였다면(파서가 걸렀어야 하는 것) 별도로 보여준다
    unknown = {k: v for k, v in ccounts.items() if k not in known}
    if unknown:
        log("-" * 62)
        for cat, n in sorted(unknown.items(), key=lambda x: -x[1]):
            log(f"{'(unknown)':<22}{cat:<24}{n:>8}{pct(n):>7.1f}%")

    log("-" * 62)
    log(f"{'TOTAL label occurrences':<46}{n_labels:>8}{pct(n_labels):>7.1f}%")
    log(f"{'units with >=1 label':<46}{n_labeled_units:>8}"
        f"{pct(n_labeled_units):>7.1f}%")
    if n_labeled_units:
        log(f"{'avg labels per labeled unit':<46}"
            f"{n_labels / n_labeled_units:>8.2f}")

    # 멀티라벨이 실제로 얼마나 나오는지 - 라벨 개수 분포
    size_dist = Counter(len(lst) for lst in cat_lists)
    log("")
    log("labels per unit: " + ", ".join(
        f"{k}:{size_dist[k]}" for k in sorted(size_dist)))

    log("")
    log(f"[saved] merged CSV -> {out_csv}")
    log(f"[saved] log        -> {run_dir / args.log_name}")
    log(f"[viz]   {run_dir}/{{Special,Normal_but}}/<uuid>_f<idx>/"
        f"{{card.png,result.json}}")
    log.close()


if __name__ == "__main__":
    main()

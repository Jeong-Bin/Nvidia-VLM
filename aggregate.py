#!/usr/bin/env python3
"""shard별 결과 CSV들을 합쳐 집계한다 (멀티라벨).

집계 내용:
  1) 판정 단위 개요 - 카테고리가 붙은 단위 수와 그 안의 blocking yes/no 분포
  2) special 카테고리별 출현 빈도를 세 벌로 - 전체 / blocking_yes / blocking_no.
     멀티라벨이라 한 단위가 여러 카테고리에 동시에 잡힐 수 있으므로 카테고리
     합계는 100%를 넘을 수 있다.

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

    # 구 스키마 CSV 를 넣으면 새 컬럼이 없어 조용히 0 이 나오므로 분명히 알린다.
    missing = [c for c in ("categories", "blocks_path") if c not in df.columns]
    if missing:
        log(f"[error] this CSV is missing {missing} - it predates the current "
            f"schema (categories + blocks_path). Re-run the inference.")
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

    cat_lists = df["categories"].apply(split_categories)
    blocks = df["blocks_path"].astype(str).str.strip().str.lower().isin(
        ("yes", "y", "true", "1"))
    labeled = cat_lists.apply(bool)

    n_labeled = int(labeled.sum())
    n_block_yes = int((labeled & blocks).sum())
    n_block_no = int((labeled & ~blocks).sum())

    # --- 1) 판정 단위 개요 ---
    log("")
    log("=" * 64)
    log("UNITS")
    log("=" * 64)
    log(f"{'WHAT':<40}{'COUNT':>10}{'%':>10}")
    log("-" * 64)
    log(f"{'total units':<40}{total:>10}{100.0:>9.1f}%")
    log(f"{'with >=1 category':<40}{n_labeled:>10}{pct(n_labeled):>9.1f}%")
    log(f"{'  blocking_yes':<40}{n_block_yes:>10}{pct(n_block_yes):>9.1f}%")
    log(f"{'  blocking_no':<40}{n_block_no:>10}{pct(n_block_no):>9.1f}%")
    log(f"{'no category (not visualised)':<40}"
        f"{total - n_labeled:>10}{pct(total - n_labeled):>9.1f}%")

    # --- 2) 카테고리 빈도: 전체 / blocking_yes / blocking_no ---
    specials = load_special_categories(SCENE_JSON)
    known = {c for _, c in specials}

    def counts_for(mask):
        c = Counter()
        for lst in cat_lists[mask]:
            c.update(lst)
        return c

    all_counts = counts_for(labeled)
    yes_counts = counts_for(labeled & blocks)
    no_counts = counts_for(labeled & ~blocks)

    log("")
    log("=" * 64)
    log("SPECIAL CATEGORY FREQUENCY  (multi-label: a unit can be in several)")
    log("=" * 64)
    log(f"{'SCENARIO':<20}{'CATEGORY':<22}{'ALL':>7}{'BLOCK':>7}{'NO-BLK':>8}"
        f"{'% ALL':>8}")
    log("-" * 64)
    for scenario, cat in specials:
        n = all_counts.get(cat, 0)
        log(f"{scenario:<20}{cat:<22}{n:>7}{yes_counts.get(cat, 0):>7}"
            f"{no_counts.get(cat, 0):>8}{pct(n):>7.1f}%")

    # json 에 없는 이름이 섞였다면(파서가 걸렀어야 하는 것) 별도로 보여준다
    unknown = {k: v for k, v in all_counts.items() if k not in known}
    if unknown:
        log("-" * 64)
        for cat, n in sorted(unknown.items(), key=lambda x: -x[1]):
            log(f"{'(unknown)':<20}{cat:<22}{n:>7}{yes_counts.get(cat, 0):>7}"
                f"{no_counts.get(cat, 0):>8}{pct(n):>7.1f}%")

    n_labels = sum(all_counts.values())
    log("-" * 64)
    log(f"{'TOTAL label occurrences':<42}{n_labels:>7}"
        f"{sum(yes_counts.values()):>7}{sum(no_counts.values()):>8}"
        f"{pct(n_labels):>7.1f}%")
    if n_labeled:
        log(f"{'avg labels per labeled unit':<42}{n_labels / n_labeled:>7.2f}")

    # 멀티라벨이 실제로 얼마나 나오는지 - 라벨 개수 분포
    size_dist = Counter(cat_lists.apply(len))
    log("")
    log("labels per unit: " + ", ".join(
        f"{k}:{size_dist[k]}" for k in sorted(size_dist)))

    # --- 3) Q3 교차검증 (--check-path 로 돌린 실행에만 있음) ---
    if "path3d_blocked" in df.columns:
        chk = df["path3d_blocked"].astype(str).str.strip()
        has = chk.isin(("Yes", "No"))
        n_chk = int(has.sum())
        if n_chk:
            geo = chk.eq("Yes")
            both = has & blocks & geo
            neither = has & ~blocks & ~geo
            only_model = has & blocks & ~geo
            only_geo = has & ~blocks & geo
            n_ok = int(both.sum() + neither.sum())

            log("")
            log("=" * 64)
            log("Q3 (model) vs 3D GEOMETRY")
            log("=" * 64)
            log(f"{'WHAT':<40}{'COUNT':>10}{'%':>10}")
            log("-" * 64)
            log(f"{'units with 3D labels':<40}{n_chk:>10}{pct(n_chk):>9.1f}%")
            log(f"{'agree':<40}{n_ok:>10}{100*n_ok/n_chk:>9.1f}%")
            log(f"{'  both say blocked':<40}{int(both.sum()):>10}"
                f"{100*both.sum()/n_chk:>9.1f}%")
            log(f"{'  both say clear':<40}{int(neither.sum()):>10}"
                f"{100*neither.sum()/n_chk:>9.1f}%")
            log(f"{'disagree':<40}{n_chk - n_ok:>10}"
                f"{100*(n_chk-n_ok)/n_chk:>9.1f}%")
            log(f"{'  model Yes / geometry No':<40}{int(only_model.sum()):>10}"
                f"{100*only_model.sum()/n_chk:>9.1f}%")
            log(f"{'  model No / geometry Yes':<40}{int(only_geo.sum()):>10}"
                f"{100*only_geo.sum()/n_chk:>9.1f}%")
            log("(disagreements are the review-priority units)")

    log("")
    log(f"[saved] merged CSV -> {out_csv}")
    log(f"[saved] log        -> {run_dir / args.log_name}")
    log(f"[viz]   {run_dir}/blocking_{{yes,no}}/<category>/<uuid>_f<idx>/"
        f"{{card.png,result.json}}")
    log.close()


if __name__ == "__main__":
    main()

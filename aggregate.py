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

from config import SCENE_JSON as CONFIG_SCENE_JSON

ROOT = Path(__file__).resolve().parent
# 기본값은 config.py 한 곳에서만 정한다. 예전에는 여기서 직접
# scene_category_B.json 을 들고 있었는데, config.py 가 E 를 가리키는 동안에도
# 이 파일만 B 로 남아 있었다 - 그러면 멀쩡한 카테고리가 전부 0 으로 찍혀서
# 모델이 못 맞춘 것인지 이름이 안 맞은 것인지 구분할 수 없게 된다.
# 다만 집계는 "이 실행이 실제로 쓴" 정의를 우선하므로, run_config.json 이
# 있으면 아래 scene_json_for_run() 이 그쪽을 쓴다 (aggregate_clip.py 와 동일).
SCENE_JSON = CONFIG_SCENE_JSON


def scene_json_for_run(run_dir):
    """이 실행이 실제로 쓴 카테고리 정의. 없으면 config.py 기본값.

    edge_case_mining.py 샤드 0 이 남긴 run_config.json 을 본다. 집계 기준이
    실행 기준과 어긋나면 멀쩡한 카테고리가 전부 0 으로 찍힌다.
    """
    cfg_path = Path(run_dir) / "run_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            name = cfg.get("key", {}).get("scene_json")
            if name:
                p = Path(name)
                return p if p.is_absolute() else ROOT / p
        except (json.JSONDecodeError, OSError):
            pass
    return SCENE_JSON


def latest_run_dir():
    # results/labeld/<ts>, results/unlabeled/<ts> 처럼 한 겹 아래도 본다.
    stamp = "[0-9]" * 8 + "_" + "[0-9]" * 6
    candidates = sorted((ROOT / "results").glob(stamp)
                        + list((ROOT / "results").glob("*/" + stamp)))
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
    labeled = cat_lists.apply(bool)
    n_labeled = int(labeled.sum())

    # blocks_path 가 전부 비어 있으면 --no-blocking 으로 돌린 실행이다.
    # 그런 실행에서는 blocking 구분을 아예 출력하지 않는다.
    bp = df["blocks_path"].astype(str).str.strip()
    has_blocking = bool(bp.isin(("Yes", "No", "yes", "no")).any())
    blocks = bp.str.lower().isin(("yes", "y", "true", "1"))

    # --- 1) 판정 단위 개요 ---
    log("")
    log("=" * 64)
    log("UNITS")
    log("=" * 64)
    log(f"{'WHAT':<40}{'COUNT':>10}{'%':>10}")
    log("-" * 64)
    log(f"{'total units':<40}{total:>10}{100.0:>9.1f}%")
    # SPECIAL / NORMAL 은 장면(판정 단위) 단위로 배타적으로 센다. 카테고리가
    # 여러 개 붙은 장면도 SPECIAL 1건이다 - 아래 카테고리 표는 멀티라벨이라
    # 합계가 이 값을 넘으므로 헷갈리지 않게 여기서 분명히 구분한다.
    n_special = n_labeled
    n_normal = total - n_labeled
    log(f"{'SPECIAL (>=1 category)':<40}{n_special:>10}{pct(n_special):>9.1f}%")
    if has_blocking:
        n_block_yes = int((labeled & blocks).sum())
        n_block_no = int((labeled & ~blocks).sum())
        log(f"{'  blocking_yes':<40}{n_block_yes:>10}{pct(n_block_yes):>9.1f}%")
        log(f"{'  blocking_no':<40}{n_block_no:>10}{pct(n_block_no):>9.1f}%")
    log(f"{'NORMAL (no category)':<40}{n_normal:>10}{pct(n_normal):>9.1f}%")
    log("-" * 64)
    log(f"{'TOTAL':<40}{total:>10}{100.0:>9.1f}%")
    log("(a scene with several categories counts once as SPECIAL)")

    # --- 2) 카테고리 빈도 (blocking 을 물었으면 yes/no 로 쪼개서도) ---
    specials = load_special_categories(scene_json_for_run(run_dir))
    known = {c for _, c in specials}

    def counts_for(mask):
        c = Counter()
        for lst in cat_lists[mask]:
            c.update(lst)
        return c

    all_counts = counts_for(labeled)
    yes_counts = counts_for(labeled & blocks) if has_blocking else Counter()
    no_counts = counts_for(labeled & ~blocks) if has_blocking else Counter()

    def row(left, cat, n):
        if has_blocking:
            return (f"{left:<20}{cat:<22}{n:>7}{yes_counts.get(cat, 0):>7}"
                    f"{no_counts.get(cat, 0):>8}{pct(n):>7.1f}%")
        return f"{left:<20}{cat:<22}{n:>7}{pct(n):>8.1f}%"

    log("")
    log("=" * 64)
    log("SPECIAL CATEGORY FREQUENCY  (multi-label: a unit can be in several)")
    log("=" * 64)
    if has_blocking:
        log(f"{'SCENARIO':<20}{'CATEGORY':<22}{'ALL':>7}{'BLOCK':>7}{'NO-BLK':>8}"
            f"{'% ALL':>8}")
    else:
        log(f"{'SCENARIO':<20}{'CATEGORY':<22}{'COUNT':>7}{'%':>8}")
    log("-" * 64)
    for scenario, cat in specials:
        log(row(scenario, cat, all_counts.get(cat, 0)))

    # json 에 없는 이름이 섞였다면(파서가 걸렀어야 하는 것) 별도로 보여준다
    unknown = {k: v for k, v in all_counts.items() if k not in known}
    if unknown:
        log("-" * 64)
        for cat, n in sorted(unknown.items(), key=lambda x: -x[1]):
            log(row("(unknown)", cat, n))

    n_labels = sum(all_counts.values())
    log("-" * 64)
    if has_blocking:
        log(f"{'TOTAL label occurrences':<42}{n_labels:>7}"
            f"{sum(yes_counts.values()):>7}{sum(no_counts.values()):>8}"
            f"{pct(n_labels):>7.1f}%")
    else:
        log(f"{'TOTAL label occurrences':<42}{n_labels:>7}{pct(n_labels):>8.1f}%")
    if n_labeled:
        log(f"{'avg labels per labeled unit':<42}{n_labels / n_labeled:>7.2f}")

    # 멀티라벨이 실제로 얼마나 나오는지 - 라벨 개수 분포
    size_dist = Counter(cat_lists.apply(len))
    log("")
    log("labels per unit: " + ", ".join(
        f"{k}:{size_dist[k]}" for k in sorted(size_dist)))

    # --- 3) Q3 교차검증 (--check-path 로 돌린 실행에만 있음) ---
    # Q3 를 묻지 않았다면 대조할 모델 답이 없으므로 이 섹션 자체를 건너뛴다.
    if "path3d_blocked" in df.columns and has_blocking:
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
    layout = "blocking_{yes,no}/<category>" if has_blocking else "<category>"
    log(f"[viz]   {run_dir}/{layout}/<uuid>_f<idx>/{{card.png,result.json}}")
    log.close()


if __name__ == "__main__":
    main()

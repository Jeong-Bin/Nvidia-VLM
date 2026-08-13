#!/usr/bin/env python3
"""클립 모드 결과(8개 GPU 샤드)를 합쳐 카테고리별로 집계한다.

aggregate.py 와 판정 단위가 달라 파일을 나눴다:
  aggregate.py      - (uuid, frame_idx) 단위. blocks_path/Q3 축이 있다.
  aggregate_clip.py - uuid(클립) 하나가 한 행. Q3 도 점수도 없다.

집계 내용:
  1) 클립 개요 - EDGE-CASE(카테고리 1개 이상) / NORMAL
  2) 카테고리별 클립 수와 비율. 멀티라벨이라 한 클립이 여러 카테고리에
     동시에 잡히면 그 카테고리 각각에 1 씩 반영한다 - 따라서 카테고리
     합계는 클립 수를 넘고 비율 합도 100% 를 넘을 수 있다.
  3) 동시 출현 - 한 클립에 몇 개가 같이 붙는지, 어떤 쌍이 자주 겹치는지

결과는 화면과 <run_dir>/aggregate_clip.log 에 함께 기록한다.

Usage:
  python aggregate_clip.py                          # results/ 아래 최근 실행 자동
  python aggregate_clip.py --run-dir results/full
  python aggregate_clip.py --run-dir results/full --min-clip-frames 1
"""
import argparse
import glob
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SCENE_JSON = ROOT / "scene_category_C.json"

# 클립 모드 CSV 를 알아보는 표지. aggregate.py 용 CSV 를 잘못 넣으면
# 조용히 0 이 나오는 대신 분명히 알린다.
REQUIRED_COLS = ("uuid", "categories", "n_categories")


def latest_run_dir():
    """results/ 아래에서 클립 모드 CSV 를 가진 가장 최근 폴더."""
    cands = []
    for d in (ROOT / "results").glob("*"):
        if d.is_dir() and (list(d.glob("clip_results*.csv"))
                           or list(d.glob("*clip*.csv"))):
            cands.append(d)
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def load_categories(scene_json: Path):
    """[(scenario, category), ...] - 표의 행 순서를 json 정의 순서로 고정."""
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
                    help="results/<name> 실행 폴더. 생략 시 가장 최근 자동 사용")
    ap.add_argument("--log-name", default="aggregate_clip.log")
    ap.add_argument("--pattern", default="clip_results*.csv",
                    help="합칠 샤드 CSV 패턴 (기본 clip_results*.csv)")
    ap.add_argument("--top-pairs", type=int, default=10,
                    help="함께 등장하는 카테고리 쌍을 몇 개까지 보여줄지")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    if run_dir is None or not run_dir.exists():
        print("no run directory found under results/ (pass --run-dir)")
        return

    files = sorted(glob.glob(str(run_dir / args.pattern)))
    if not files:
        print(f"no CSV matching {args.pattern} in {run_dir}")
        return

    frames, per_file = [], []
    for f in files:
        d = pd.read_csv(f)
        per_file.append((Path(f).name, len(d)))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    log = Tee(run_dir / args.log_name)
    log(f"[info] run dir : {run_dir}")
    log(f"[info] shards  : {len(files)} CSV")
    for name, n in per_file:
        log(f"         {name:34} {n:6d} clips")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        log(f"[error] CSV is missing {missing} - this does not look like a "
            f"clip-mode run. For (uuid, frame_idx) results use aggregate.py.")
        log.close()
        return

    # 샤드는 서로 겹치지 않아야 한다. 겹치면 같은 클립이 두 번 세어지므로
    # 조용히 넘기지 않고 알린 뒤 중복을 제거한다.
    n_raw = len(df)
    dup = int(df["uuid"].duplicated().sum())
    if dup:
        log(f"[warn] {dup} duplicate uuid(s) across shards - keeping first")
        df = df.drop_duplicates(subset="uuid", keep="first")

    total = len(df)
    if total == 0:
        log("[warn] no rows")
        log.close()
        return

    merged = run_dir / "clip_results_all.csv"
    df.to_csv(merged, index=False)

    def pct(n):
        return 100 * n / total

    if "parse_ok" in df.columns:
        n_fail = int((df["parse_ok"] == 0).sum())
        if n_fail:
            log(f"[warn] JSON parse failed on {n_fail} clips ({pct(n_fail):.1f}%)")

    cat_lists = df["categories"].apply(split_categories)
    labeled = cat_lists.apply(bool)
    n_edge = int(labeled.sum())

    # --- 1) 클립 개요 ---
    log("")
    log("=" * 64)
    log("CLIPS")
    log("=" * 64)
    log(f"{'WHAT':<40}{'COUNT':>10}{'%':>10}")
    log("-" * 64)
    log(f"{'total clips':<40}{total:>10}{100.0:>9.1f}%")
    log(f"{'EDGE-CASE (>=1 category)':<40}{n_edge:>10}{pct(n_edge):>9.1f}%")
    log(f"{'NORMAL    (no category)':<40}{total-n_edge:>10}"
        f"{pct(total-n_edge):>9.1f}%")
    log("-" * 64)
    log(f"{'TOTAL':<40}{total:>10}{100.0:>9.1f}%")
    log("(a clip with several categories counts once as EDGE-CASE)")

    # --- 2) 카테고리 빈도 (멀티라벨: 클립 하나가 여러 곳에 반영된다) ---
    counts = Counter()
    for lst in cat_lists:
        counts.update(lst)          # 한 클립의 카테고리 각각에 1
    known = load_categories(SCENE_JSON)
    known_names = {c for _, c in known}

    log("")
    log("=" * 64)
    log("CATEGORY FREQUENCY  (clips per category; multi-label)")
    log("=" * 64)
    log(f"{'SCENARIO':<22}{'CATEGORY':<32}{'CLIPS':>6}{'% ALL':>8}{'% EDGE':>8}")
    log("-" * 64)
    for scenario, cat in known:
        n = counts.get(cat, 0)
        edge_pct = (100 * n / n_edge) if n_edge else 0.0
        log(f"{scenario:<22}{cat:<32}{n:>6}{pct(n):>7.1f}%{edge_pct:>7.1f}%")

    unknown = {k: v for k, v in counts.items() if k not in known_names}
    if unknown:
        log("-" * 64)
        for cat, n in sorted(unknown.items(), key=lambda x: -x[1]):
            log(f"{'(unknown)':<22}{cat:<32}{n:>6}{pct(n):>7.1f}%")

    n_labels = sum(counts.values())
    log("-" * 64)
    log(f"{'TOTAL label occurrences':<54}{n_labels:>6}{pct(n_labels):>7.1f}%")
    if n_edge:
        log(f"{'avg categories per edge-case clip':<54}"
            f"{n_labels / n_edge:>6.2f}")
    log("(% ALL = share of all clips, % EDGE = share of edge-case clips;")
    log(" a clip with 2 categories is counted in both, so sums exceed 100%)")

    # --- 3) 동시 출현 ---
    size_dist = Counter(cat_lists.apply(len))
    log("")
    log("categories per clip: " + ", ".join(
        f"{k}:{size_dist[k]}" for k in sorted(size_dist)))

    pairs = Counter()
    for lst in cat_lists:
        if len(lst) > 1:
            pairs.update(combinations(sorted(set(lst)), 2))
    if pairs:
        log("")
        log(f"most frequent co-occurring pairs (top {args.top_pairs}):")
        for (a, b), n in pairs.most_common(args.top_pairs):
            log(f"   {n:4d}  {a} + {b}")

    log("")
    log(f"[saved] merged CSV -> {merged}  ({total} clips"
        + (f", {n_raw - total} duplicates dropped)" if dup else ")"))
    log(f"[saved] log        -> {run_dir / args.log_name}")
    log.close()


if __name__ == "__main__":
    main()

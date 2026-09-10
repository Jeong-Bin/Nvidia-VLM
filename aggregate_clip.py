#!/usr/bin/env python3
"""클립 모드 결과(8개 GPU 샤드)를 합쳐 카테고리별로 집계한다.

aggregate.py 와 판정 단위가 달라 파일을 나눴다:
  aggregate.py      - (uuid, frame_idx) 단위. blocks_path/Q3 축이 있다.
  aggregate_clip.py - uuid(클립) 하나가 한 행. Q3 도 점수도 없다.

집계 내용:
  1) 클립 개요 - SPECIAL(카테고리 1개 이상) / NORMAL
  2) 카테고리별 클립 수와 비율. 멀티라벨이라 한 클립이 여러 카테고리에
     동시에 잡히면 그 카테고리 각각에 1 씩 반영한다 - 따라서 카테고리
     합계는 클립 수를 넘고 비율 합도 100% 를 넘을 수 있다.
  3) 동시 출현 - 한 클립에 몇 개가 같이 붙는지, 어떤 쌍이 자주 겹치는지
  4) 등급 분포 - safety/rarity 의 값별 개수와 비율, 평균/분산/중앙값.
     정답 라벨이 없는 실행(NAS 신규 청크 등)에서는 evaluate_labels.py 를
     못 돌리므로, 모델 출력 자체의 분포를 보는 것이 여기서 유일한 점검
     수단이다. edge-case 만 따로 한 번 더 낸다 - normal 클립이 거의 전부
     1 이라 전체 평균에 섞으면 special 안에서의 분포가 묻힌다.
  5) 주행 난이도 분포 - --difficulty 를 켠 실행에서만. 전체 난이도와 요인
     4개(조도/강수/노면/대기가림)를 0~4 눈금으로 각각 집계하고, 마지막에
     다섯 축을 나란히 놓은 요약표를 낸다. 등급과 달리 edge-case 여부와
     무관한 축이라 SPECIAL 만 따로 내지는 않는다.

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

from config import SCENE_JSON as CONFIG_SCENE_JSON
from constrained_tier import TIER_VALUES, tier_label
from prompts import DIFFICULTY_AXES, DIFFICULTY_MIN, DIFFICULTY_MAX

ROOT = Path(__file__).resolve().parent
# 기본값은 config.py 한 곳에서만 정한다. 다만 집계는 "이 실행이 실제로 쓴"
# 정의를 우선하므로, run_config.json 이 있으면 아래 scene_json_for_run() 이
# 그쪽을 쓴다 - 기본값으로 집계하면 다른 체계로 돌린 결과의 카테고리가
# 통째로 0 으로 나온다.
SCENE_JSON = CONFIG_SCENE_JSON

# 클립 모드 CSV 를 알아보는 표지. aggregate.py 용 CSV 를 잘못 넣으면
# 조용히 0 이 나오는 대신 분명히 알린다.
REQUIRED_COLS = ("uuid", "categories", "n_categories")


def scene_json_for_run(run_dir):
    """이 실행이 실제로 쓴 카테고리 정의. 없으면 config.py 기본값.

    edge_case_mining.py 샤드 0 이 남긴 run_config.json 을 본다. 집계 기준이
    실행 기준과 어긋나면 멀쩡한 카테고리가 전부 0 으로 찍혀서, 모델이 못
    맞춘 것인지 이름이 안 맞은 것인지 구분할 수 없게 된다.
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
    """results/ 아래에서 클립 모드 CSV 를 가진 가장 최근 폴더."""
    # 실행 폴더는 results/labeld/<ts> 처럼 한 겹 아래에도 있다(라벨/비라벨 분리).
    # 옛 실행은 results/<ts> 에 그대로 있으므로 두 깊이를 모두 본다.
    cands = []
    for pat in ("*", "*/*"):
        for d in (ROOT / "results").glob(pat):
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


def tier_series(df, col):
    """CSV 의 등급 칸 -> 정수 리스트. 비었거나 못 읽는 값은 뺀다.

    parse 실패한 클립은 이 칸이 빈 문자열이라 float NaN 으로 읽힌다.
    그걸 0 으로 치면 평균이 통째로 내려앉으므로 아예 분모에서 뺀다 -
    대신 아래에서 "미기록 N건"으로 따로 알린다.
    """
    if col not in df.columns:
        return []
    out = []
    for v in df[col]:
        try:
            if pd.isna(v):
                continue
            out.append(int(float(v)))
        except (TypeError, ValueError):
            continue
    return out


def dist_block(log, title, values, total, scale, labels=None,
               note=""):
    """점수 한 축의 분포 - 값별 개수/비율과 평균/분산/중앙값.

    safety/rarity(0~4)와 주행 난이도 4축(0~4)이 같이 쓴다. 눈금이 같아졌지만
    여전히 scale 로 받고, 라벨이 있는 축(None/Low/Moderate/...)만 labels 를 준다.

    라벨이 없는 데이터에서는 정답과 대조할 수 없으므로(evaluate_labels.py 의
    MAE/혼동행렬을 못 쓴다) 모델 출력 자체의 분포만 본다. 그래도 쓸모가
    있는 이유는, 한쪽 등급으로 쏠렸는지가 여기서 바로 보이기 때문이다 -
    실측으로 safety 가 2 에 몰리는 경향이 있어 그 쏠림을 보는 것이 이
    표의 주된 목적이다.

    분산은 모분산(N 으로 나눔)이다. 표본에서 모집단을 추정하는 상황이
    아니라 "이번 실행이 낸 출력의 퍼짐"을 그대로 기술하는 값이라서다.
    """
    log("")
    log("=" * 64)
    log(f"{title}  (model output distribution)")
    log("=" * 64)
    if note:
        log(f"  {note}")
    n = len(values)
    if not n:
        log("  (no rows - 이 칸이 비어 있다. 프롬프트에서 이 축을 껐거나"
            " JSON parse 가 전부 실패한 경우다)")
        return

    counts = Counter(values)
    log(f"{'SCORE':<28}{'COUNT':>10}{'% RATED':>10}{'% ALL':>10}")
    log("-" * 64)
    # 눈금에 정의된 값은 0 건이어도 모두 찍는다 - 빠져 있으면 "한 번도 안
    # 나온 점수"와 "눈금에 없는 점수"가 구분되지 않는다.
    for v in scale:
        c = counts.get(v, 0)
        name = f"{v} = {labels[v]}" if labels else str(v)
        log(f"{name:<28}{c:>10}{100*c/n:>9.1f}%"
            f"{100*c/total:>9.1f}%")
    # 정의 밖의 값이 나오면 조용히 버리지 않는다 (파서가 샌 경우).
    extra = sorted(set(counts) - set(scale))
    if extra:
        log("-" * 64)
        for v in extra:
            c = counts[v]
            log(f"{f'{v} = (out of range)':<28}{c:>10}{100*c/n:>9.1f}%"
                f"{100*c/total:>9.1f}%")
    log("-" * 64)
    log(f"{'TOTAL rated':<28}{n:>10}{100.0:>9.1f}%{100*n/total:>9.1f}%")
    if n < total:
        log(f"{'unrated (blank/parse fail)':<28}{total-n:>10}{'':>10}"
            f"{100*(total-n)/total:>9.1f}%")

    s = sorted(values)
    mean = sum(s) / n
    # 모분산 - 표본분산이 아니다 (위 docstring 참고)
    var = sum((x - mean) ** 2 for x in s) / n
    median = (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)
    log("")
    log(f"  mean {mean:.3f}   var {var:.3f}   std {var ** 0.5:.3f}   "
        f"median {median:g}   min {s[0]}   max {s[-1]}")


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
    log(f"{'SPECIAL (>=1 category)':<40}{n_edge:>10}{pct(n_edge):>9.1f}%")
    log(f"{'NORMAL    (no category)':<40}{total-n_edge:>10}"
        f"{pct(total-n_edge):>9.1f}%")
    log("-" * 64)
    log(f"{'TOTAL':<40}{total:>10}{100.0:>9.1f}%")
    log("(a clip with several categories counts once as SPECIAL)")

    # --- 2) 카테고리 빈도 (멀티라벨: 클립 하나가 여러 곳에 반영된다) ---
    counts = Counter()
    for lst in cat_lists:
        counts.update(lst)          # 한 클립의 카테고리 각각에 1
    scene_json = scene_json_for_run(run_dir)
    known = load_categories(scene_json)
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

    # --- 4) 등급 분포 ---
    # 라벨 없는 실행에서는 evaluate_labels.py 를 못 돌리므로, 등급을 여기서
    # 본다. edge-case 만 따로 다시 내는 이유는 normal 클립이 거의 전부 1 이라
    # 전체 평균이 1 쪽으로 눌려 special 안에서의 분포가 안 보이기 때문이다.
    edge_df = df[labeled]
    tier_labels = {v: tier_label(v) for v in TIER_VALUES}
    for col, title in (("safety_tier", "SAFETY CRITICALITY"),
                       ("rarity_tier", "RARITY")):
        dist_block(log, f"{title} - ALL clips", tier_series(df, col), total,
                   TIER_VALUES, tier_labels)
        if n_edge:
            dist_block(log, f"{title} - SPECIAL clips only",
                       tier_series(edge_df, col), n_edge,
                       TIER_VALUES, tier_labels)

    # --- 5) 주행 난이도 분포 ---
    # 등급과 달리 edge-case 여부와 무관한 축이다 - 평범한 클립도 비가 오면
    # 높다. 그래서 SPECIAL 만 따로 내지 않고 전체만 낸다. 대신 축이
    # 다섯이라, 어느 요인이 전체 난이도를 끌어올리는지 나란히 놓고 본다.
    diff_scale = list(range(DIFFICULTY_MIN, DIFFICULTY_MAX + 1))
    diff_series = {k: tier_series(df, k) for k, _ in DIFFICULTY_AXES}
    if any(diff_series.values()):
        for key, name in DIFFICULTY_AXES:
            dist_block(log, f"{name} ({key})", diff_series[key], total,
                       diff_scale)

        # 다섯 축 요약표. 위 표를 다섯 번 읽지 않고도 "무엇이 높은가"를
        # 한눈에 보려는 것이다.
        log("")
        log("=" * 64)
        log("DIFFICULTY SUMMARY  (all 5 axes side by side)")
        log("=" * 64)
        log(f"  {'AXIS':<26}{'N':>6}{'MEAN':>8}{'VAR':>8}{'MEDIAN':>8}"
            f"{'MIN':>5}{'MAX':>5}")
        log("  " + "-" * 62)
        for key, name in DIFFICULTY_AXES:
            v = sorted(diff_series[key])
            if not v:
                log(f"  {name:<26}{0:>6}{'-':>8}{'-':>8}{'-':>8}{'-':>5}{'-':>5}")
                continue
            m = len(v)
            mean = sum(v) / m
            var = sum((x - mean) ** 2 for x in v) / m
            med = v[m // 2] if m % 2 else (v[m // 2 - 1] + v[m // 2]) / 2
            log(f"  {name:<26}{m:>6}{mean:>8.3f}{var:>8.3f}{med:>8g}"
                f"{v[0]:>5}{v[-1]:>5}")
        log("  " + "-" * 62)
        log(f"  scale {DIFFICULTY_MIN}-{DIFFICULTY_MAX}; factors are judged "
            f"independently, so they need not sum to the overall rating.")

    log("")
    log(f"[saved] merged CSV -> {merged}  ({total} clips"
        + (f", {n_raw - total} duplicates dropped)" if dup else ")"))
    log(f"[saved] log        -> {run_dir / args.log_name}")
    log.close()


if __name__ == "__main__":
    main()

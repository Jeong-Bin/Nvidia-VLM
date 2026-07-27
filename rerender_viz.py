#!/usr/bin/env python3
"""results/<run_dir>/results_shard_*.csv (또는 지정한 CSV들)로부터 시각화 카드만 다시 생성한다.

추론을 재실행하지 않고, CSV에 저장된 caption/top1~3 매칭 결과 + 원본 mp4에서
다시 뽑은 프레임으로 카드 이미지를 그린다. visualize.py 의 레이아웃/뷰 순서를
바꾼 뒤 이미지만 재생성하고 싶을 때 사용.

Usage:
  python rerender_viz.py                                  # results/ 아래 가장 최근 실행 폴더 자동 사용
  python rerender_viz.py --run-dir results/20260724_104830
  python rerender_viz.py --csv path/to/results_shard_0.csv
"""
import argparse
import csv
import glob
from pathlib import Path

from tqdm import tqdm

from edge_case_mining import sample_unit_frames, category_slug, ROOT
from visualize import render_scene_card
from aggregate import latest_run_dir


def load_rows(csv_paths):
    rows = []
    for p in csv_paths:
        with open(p, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def row_to_matches(row):
    matches = []
    for j in (1, 2, 3):
        cat = row.get(f"top{j}_category", "")
        if not cat:
            continue
        matches.append({
            "scenario": row[f"top{j}_scenario"],
            "category": cat,
            "confidence": float(row[f"top{j}_confidence"]),
        })
    return matches


def is_normal_row(row):
    """top1이 normal 카테고리("Normal Driving"/"Normal Stop")인지."""
    cat = row.get("top1_category", "")
    return "Normal" in cat if cat else False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="results/<timestamp> 실행 폴더. 생략 시 가장 최근 실행 자동 사용")
    ap.add_argument("--csv", nargs="+", default=None,
                    help="대상 CSV 경로(들) 직접 지정. --run-dir 대신 사용 가능")
    ap.add_argument("--viz-dir", default=None,
                    help="시각화 저장 폴더. 생략 시 --run-dir(또는 자동탐색된 실행 폴더)와 동일")
    args = ap.parse_args()

    if args.csv:
        csv_paths = args.csv
        run_dir = Path(args.csv[0]).parent
    else:
        run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
        if run_dir is None or not run_dir.exists():
            print("no run directory found under results/ (run run_all.sh first, or pass --run-dir/--csv)")
            raise SystemExit(1)
        csv_paths = sorted(glob.glob(str(run_dir / "results_shard_*.csv")))

    if not csv_paths:
        print(f"no CSV found in {run_dir}")
        raise SystemExit(1)
    print(f"[info] run dir: {run_dir}")
    print(f"[info] reading {len(csv_paths)} CSV file(s): {[Path(p).name for p in csv_paths]}")

    rows = load_rows(csv_paths)
    # special 카테고리 + OOD (top1_category 빈 문자열) 모두 대상, normal 만 제외
    target_rows = [r for r in rows if not is_normal_row(r)]
    print(f"[info] {len(rows)} total rows, {len(target_rows)} special/OOD (top1) rows to render")

    viz_path = Path(args.viz_dir) if args.viz_dir else run_dir
    viz_path.mkdir(parents=True, exist_ok=True)

    for row in tqdm(target_rows, unit="img", dynamic_ncols=True):
        uuid = row["uuid"]
        frame_idx = int(row["frame_idx"])
        matches = row_to_matches(row)
        top1 = matches[0]["category"] if matches else "OOD"

        unit_frames = sample_unit_frames(uuid, frame_idx)
        cat_dir = viz_path / category_slug(top1)
        cat_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{uuid}_f{frame_idx:04d}.png"
        render_scene_card(uuid, frame_idx, unit_frames["cur"], row["caption"], matches,
                          out_path=cat_dir / out_name)

    print(f"[done] re-rendered {len(target_rows)} images -> {viz_path}/<category>/")

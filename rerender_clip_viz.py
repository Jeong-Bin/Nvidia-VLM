#!/usr/bin/env python3
"""클립 모드 실행의 시각화(viz/)를 추론 없이 다시 만든다.

CSV(1차 결과)와 <run>/attn/*.npz(--attn-timeline 기록)만으로 영상 + result.json 을
다시 그린다. 시각화 코드(visualize_clip.py)나 어텐션 곡선 계산(attn_timeline.py)을
바꾼 뒤 이미 끝난 실행에 적용할 때 쓴다.

어느 클립을 그리나: 실행 때와 같은 규칙을 따른다(run_config.json 의 설정).
  --viz-per-category N 이었으면 장르마다 N개 - viz/<장르>/<uuid>/
    이 정원은 샤드(GPU)마다 따로 센다. 실행 때 샤드 k 가 eval_uuids.txt 순서의
    k, k+n, k+2n... 번째 클립을 맡아 제 몫의 정원을 채웠으므로, 같은 순서와
    같은 샤드 나누기로 다시 세야 원래와 같은 클립이 나온다.
  아니면 special 이고 묶음 점수 최댓값이 2 이상인 클립 - viz/score_<N>/<uuid>/

어느 CSV 를 쓰나: 기본은 2차 확인(recheck) 전 결과다. 실행 중에 만들어지던 원래
시각화가 1차 결과로 그려졌고, 어텐션 곡선도 1차 답의 토큰에서 나온다(2차 확인이
더한 장르는 1차 답에 없어 곡선이 없다). --final 이면 2차 확인 후 결과로 그린다.

몰림 토큰(attention sink)을 빼기 전에 기록한 npz(sink_removed 표시 없음)는
첫 칸(0~1초)을 빼고 곡선을 만든다 - 기록에 몰림 토큰이 섞여 있어 그 칸 값이
내용이 아니기 때문이다. 시각화에서는 그 칸이 회색("no value")으로 나온다.

Usage:
  python rerender_clip_viz.py --run-dir results/labeld/<run>
  python rerender_clip_viz.py --run-dir results/labeld/<run> --final --out-dir viz_final
  python rerender_clip_viz.py --run-dir results/labeld/<run> --uuids <uuid> ... --video-name clip2.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

import edge_case_mining as E
import visualize_clip as V
from attn_timeline import curves_for, default_layers, load_spatial
from constrained_tier import group_key, score_fields, score_dirname
from prompts import DIFFICULTY_AXES

ROOT = Path(__file__).resolve().parent


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def row_to_result(r: dict) -> dict:
    """CSV 한 행 -> render_clip_result 가 받는 result dict."""
    cats = [c for c in (r.get("categories") or "").split("|") if c]
    gs = {}
    for part in (r.get("group_scores") or "").split("|"):
        k, _, v = part.rpartition("=")
        if k and _int(v) is not None:
            gs[k] = _int(v)
    res = {
        "verdict": r.get("verdict") or ("Special" if cats else "Normal"),
        "categories": cats,
        "parse_ok": r.get("parse_ok") == "1",
        "observation": r.get("observation", ""),
        "ego_behavior": r.get("ego_behavior", ""),
        "unusual_elements": r.get("unusual_elements", ""),
        "group_scores": gs,
        "score_reason": r.get("score_reason", ""),
        "tier_score": max(gs.values()) if gs else None,
    }
    for k, _ in DIFFICULTY_AXES:
        res[k] = _int(r.get(k))
        res[f"{k}_reason"] = r.get(f"{k}_reason", "")
    if r.get("time_spans"):
        try:
            res["time_spans"] = json.loads(r["time_spans"])
        except ValueError:
            pass
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--final", action="store_true",
                    help="2차 확인(recheck) 후 결과로 그린다 (기본: 2차 확인 전)")
    ap.add_argument("--out-dir", default="viz",
                    help="run-dir 아래 저장 폴더 이름 (기본 viz)")
    ap.add_argument("--width", type=int, default=V.VIZ_WIDTH)
    ap.add_argument("--uuids", nargs="+", default=None,
                    help="이 클립만 다시 그린다 (기본: 실행 때 규칙대로 전부)")
    ap.add_argument("--video-name", default="clip.mp4",
                    help="영상 파일 이름. clip.mp4 가 아니면 원래 영상과 "
                         "result.json 은 그대로 두고 영상만 이 이름으로 더한다")
    args = ap.parse_args()

    run = Path(args.run_dir)
    cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    argv = cfg.get("argv", {})
    key = cfg.get("key", {})

    csv_path = run / "before_recheck_clip_results_all.csv"
    if args.final or not csv_path.exists():
        csv_path = run / "clip_results_all.csv"
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    print(f"[info] csv  : {csv_path.name} ({len(rows)} clips)")

    scene = Path(key.get("scene_json") or argv.get("scene_json") or E.SCENE_JSON)
    scene = scene if scene.is_absolute() else ROOT / scene
    V.load_scenario_map(scene)
    labels = E.load_labels(scene)
    group = key.get("group") or "all"
    gk = {n: group_key(n) for n, g in score_fields(labels)
          if group == "all" or group_key(g) == group}

    gt_path = argv.get("gt_labels")
    if gt_path and not Path(gt_path).is_absolute():
        gt_path = ROOT / gt_path
    gt = E._load_gt_labels(gt_path) if gt_path else None

    tok = None
    attn_dir = run / "attn"
    if attn_dir.exists():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(key.get("model") or argv.get("model"))

    per_cat = argv.get("viz_per_category")
    view = "camera_front_wide_120fov"
    out_root = run / args.out_dir
    counts = Counter()
    n = 0
    # 실행 때와 같은 처리 순서와 샤드 나누기(edge_case_mining: uuids[k::n]).
    by_uuid = {r["uuid"]: r for r in rows}
    order_file = run / "eval_uuids.txt"
    order = ([u.strip() for u in order_file.read_text(encoding="utf-8").splitlines()
              if u.strip() in by_uuid] if order_file.exists() else list(by_uuid))
    n_shards = int(key.get("num_shards") or argv.get("num_shards") or 1)
    shard_of = {u: i % n_shards for i, u in enumerate(order)}
    print(f"[info] shards: {n_shards} (장르별 정원을 샤드마다 센다)")
    only = set(args.uuids or ())
    for u in order:
        if only and u not in only:
            continue
        r = by_uuid[u]
        sh = shard_of[u]
        if _int(r.get("n_frames")) in (None, 0):
            continue
        res = row_to_result(r)
        cats = res["categories"]
        if not cats:
            continue
        if per_cat:
            targets = [c for c in cats if counts[(sh, c)] < per_cat]
            if only:
                # 고른 클립만 돌면 정원을 셀 앞 클립이 없어 원래 없던 장르
                # 폴더까지 생긴다 - 이미 그려 둔 장르 폴더가 있으면 거기만.
                had = [c for c in cats
                       if (out_root / V.safe_dirname(c) / r["uuid"]).is_dir()]
                targets = had or targets
            if not targets:
                continue
            out_dirs = [out_root / V.safe_dirname(c) / r["uuid"] for c in targets]
        else:
            top = res["tier_score"]
            if argv.get("not_save_low", True) and top is not None and top <= 1:
                continue
            targets = None
            out_dirs = [out_root / score_dirname(top) / r["uuid"]]

        # 어텐션 곡선 - 1차 답의 토큰에서 다시 계산한다.
        npz = attn_dir / f"{r['uuid']}.npz"
        if tok is not None and npz.exists():
            z = np.load(npz)
            rows_ = z["rows"].astype(np.float32)
            old = "sink_removed" not in z.files
            cv = curves_for(rows_, z["gen_ids"].tolist(), tok, cats, gk,
                            drop_first=old)
            cv["times"] = z["times"].tolist()
            cv["layers"] = default_layers(rows_.shape[1])
            res["attn_curves"] = cv
            sp = load_spatial(z)
            if sp is not None:
                res["attn_spatial"] = {**sp, "times": z["times"].tolist()}
        if key.get("time_spans"):
            res["_time_spans_on"] = True

        extra = {"ego_behavior_measured": r.get("ego_behavior_measured", ""),
                 "n_frames_seen": _int(r.get("n_frames")),
                 "ego_speed_kmh": r.get("ego_speed_kmh") or None}
        for i, od in enumerate(out_dirs):
            V.render_clip_result(r["uuid"], E.clip_path(view, r["uuid"]), res, od,
                                 max_width=args.width,
                                 attn_focus=(targets[i] if targets else None),
                                 gt=(gt or {}).get(r["uuid"]), extra=extra,
                                 video_name=args.video_name,
                                 write_json=args.video_name == "clip.mp4")
            n += 1
            print(f"[viz] {od.relative_to(run) / args.video_name}")
        if targets is not None:
            counts.update((sh, c) for c in targets)
    print(f"[done] {n} videos -> {out_root}")


if __name__ == "__main__":
    main()

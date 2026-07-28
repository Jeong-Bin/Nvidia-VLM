#!/usr/bin/env python3
"""obstacle.offline 라벨 조회 - 판정 단위(uuid, frame_idx) 주변의 3D 객체 요약.

NVIDIA PhysicalAI-AV 는 클립마다 obstacle.offline 파일을 제공한다:
  timestamp_us, track_id, center_x/y/z, size_x/y/z, orientation_*, label_class

label_class 분포(전체의 약 89%가 automobile):
  automobile, person, heavy_truck, trailer, bus, rider, other_vehicle,
  protruding_object, stroller, animal, train_or_tram_car

용도: VLM 프롬프트에 "주변에 무엇이 있는지"를 사실로 알려준다. 단, 프레임당
40~55개 객체를 모두 나열하면 노이즈가 되므로 요약해서 넣는다:
  - 흔한 automobile 은 개수만
  - 드물고 위험한 클래스(animal, person, protruding_object 등)는 최근접 거리까지

라벨 자체가 pseudo(autolabels v2)이고 14개 카테고리 중 Road Construction /
Tunnel / Pothole 같은 환경 카테고리는 전혀 커버하지 못한다는 점에 유의.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from egomotion import _read_label, frame_timestamps

# 카메라 프레임 시각 기준 앞뒤로 이 범위 안의 검출만 본다 (obstacle 은
# 검출마다 timestamp 가 조금씩 다르다)
WINDOW_US = 50_000
# 이 거리 밖의 객체는 무시 (m)
MAX_DIST_M = 60.0

# automobile 은 너무 흔해서 정보량이 낮다 - 개수만 세고, 나머지는 거리까지 알린다
COMMON_CLASSES = {"automobile"}
# 사람이 봐도 자연스러운 표기로 변환
CLASS_LABEL = {
    "automobile": "car",
    "person": "pedestrian",
    "heavy_truck": "heavy truck",
    "other_vehicle": "other vehicle",
    "protruding_object": "protruding object",
    "train_or_tram_car": "train/tram",
}


@lru_cache(maxsize=1024)
def _obstacle_arrays(uuid: str):
    """(timestamp_us, dist, label_class) 배열. 없으면 None."""
    df = _read_label("obstacle.offline", uuid)
    if df is None or len(df) == 0:
        return None
    t = df["timestamp_us"].to_numpy(dtype=np.float64)
    dist = np.hypot(df["center_x"].to_numpy(), df["center_y"].to_numpy())
    cls = df["label_class"].to_numpy(dtype=object)
    return t, dist, cls


def obstacle_summary(uuid: str, frame_idx: int,
                     window_us: int = WINDOW_US,
                     max_dist: float = MAX_DIST_M) -> dict | None:
    """판정 단위 주변 객체 요약.

    반환: {"counts": {class: n}, "nearest": {class: dist_m}, "total": n}
          라벨이 없으면 None.
    """
    ts = frame_timestamps(uuid)
    ob = _obstacle_arrays(uuid)
    if ts is None or ob is None:
        return None
    if not (0 <= frame_idx < len(ts)):
        return None

    t_us = float(ts[frame_idx])
    t, dist, cls = ob
    sel = (np.abs(t - t_us) <= window_us) & (dist <= max_dist)
    if not sel.any():
        return {"counts": {}, "nearest": {}, "total": 0}

    d_sel, c_sel = dist[sel], cls[sel]
    counts, nearest = {}, {}
    for c in set(c_sel):
        m = c_sel == c
        counts[c] = int(m.sum())
        nearest[c] = float(d_sel[m].min())
    return {"counts": counts, "nearest": nearest, "total": int(sel.sum())}


def describe_obstacles(summary: dict | None, max_classes: int = 6) -> str:
    """obstacle_summary -> 프롬프트에 넣을 한 줄 사실 문구.

    드문 클래스를 앞에 두고(정보량이 높다), automobile 은 개수만 밝힌다.
    """
    if not summary or not summary["counts"]:
        return ""
    counts, nearest = summary["counts"], summary["nearest"]

    # 흔한 클래스는 뒤로, 그 안에서는 가까운 것부터
    def sort_key(c):
        return (c in COMMON_CLASSES, nearest[c])

    parts = []
    for c in sorted(counts, key=sort_key)[:max_classes]:
        name = CLASS_LABEL.get(c, c.replace("_", " "))
        n = counts[c]
        label = f"{n} {name}" + ("s" if n > 1 and not name.endswith("s") else "")
        if c not in COMMON_CLASSES:
            label += f" (nearest {nearest[c]:.0f} m)"
        parts.append(label)
    return "3D sensor labels detect nearby: " + ", ".join(parts) + "."


def has_obstacle(uuid: str) -> bool:
    from egomotion import _zip_index
    return uuid in _zip_index("obstacle.offline")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from egomotion import CAMERA_DIR, REF_VIEW

    ap = argparse.ArgumentParser(description="obstacle 조회 점검")
    ap.add_argument("--uuid")
    ap.add_argument("--limit-clips", type=int, default=3)
    ap.add_argument("--timestamps-per-clip", type=int, default=5)
    args = ap.parse_args()

    uuids = ([args.uuid] if args.uuid else
             sorted(p.name.split(".")[0]
                    for p in (CAMERA_DIR / REF_VIEW).glob("*.mp4"))[:args.limit_clips])
    for u in uuids:
        ts = frame_timestamps(u)
        if ts is None:
            print(f"{u}: no camera timestamps")
            continue
        print(f"\n=== {u} ===")
        for fi in np.linspace(0, len(ts) - 1, args.timestamps_per_clip, dtype=int):
            s = obstacle_summary(u, int(fi))
            if s is None:
                print(f"  f{fi:4d}  (no obstacle label)")
                continue
            print(f"  f{fi:4d}  n={s['total']:3d}  {describe_obstacles(s)}")

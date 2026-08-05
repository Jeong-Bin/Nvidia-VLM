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

# --- rig 좌표계 규약 (20260803 실측으로 확정) ---
# center_x = 전방(+가 앞), center_y = 횡방향, center_z = 상방.
# 검증: 20초 클립에서 30프레임 이상 지속된 82개 트랙에 대해
#   d(center_x)/dt 를 자차 속도로 나눈 값의 중앙값이 정확히 -1.00 이었다
#   (정지 객체는 자차가 전진한 만큼 x 가 줄어든다). 마주 오는 차량은 -1 보다
#   큰 값이 나와 물리적으로도 일관됨.
# size_x/size_y 는 객체 자기 좌표계의 길이/폭 (승용차 4.29 x 1.93 m,
#   대형트럭 10.47 x 3.08 m) 이라 yaw 로 회전시켜야 rig 축에 투영된다.
# y 의 좌/우 부호는 확정하지 못했으나, 경로 침범 판정은 |y| 만 쓰므로 무관.

# 자차 주행 통로의 반폭 (m). 차폭 약 2m 에 여유를 더한 값.
EGO_HALF_WIDTH_M = 1.5
# 전방 몇 m 까지를 "주행 경로"로 볼지
LOOKAHEAD_M = 30.0

# --- 카메라 화각 (20260805 sensor_extrinsics 실측) ---
# 광축을 rig 로 회전시켜 얻은 yaw:
#   front_wide  -0.9도, cross_left +67.0도, cross_right -65.9도  (모두 120도 FOV)
# obstacle 라벨은 360도 전방위라 그대로 쓰면 카메라에 안 보이는 뒤쪽 객체까지
# 프롬프트에 들어간다 - 40클립 표본에서 검출의 47.9% 가 자차 뒤(x<0)였고,
# 이는 모델이 이미지로 확인할 수 없는 것을 "있다"고 알려주는 셈이라
# 과탐(특히 person/rider 계열)의 직접적 원인이 된다. 그래서 시야 밖은 버린다.
FOV_HALF_DEG = 60.0                      # 120도 FOV 의 절반
CROSS_CAM_YAW_DEG = 67.0                 # cross 카메라 광축 yaw (좌우 대칭 가정)
# 전방 3뷰 합산 시야: 67 + 60 = 127도. front-wide 단독이면 60도.
VIEW_FOV_HALF_DEG = {
    1: FOV_HALF_DEG,                              # front-wide 단독
    3: CROSS_CAM_YAW_DEG + FOV_HALF_DEG,          # 전방 3뷰
}

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
    """(timestamp_us, dist, bearing_deg, label_class) 배열. 없으면 None.

    bearing_deg 는 자차 정면(+x)에서 잰 방위각의 절댓값 (0=정면, 90=바로 옆,
    180=바로 뒤). 좌우 대칭이라 부호는 필요 없다.
    """
    df = _read_label("obstacle.offline", uuid)
    if df is None or len(df) == 0:
        return None
    t = df["timestamp_us"].to_numpy(dtype=np.float64)
    x = df["center_x"].to_numpy(dtype=np.float64)
    y = df["center_y"].to_numpy(dtype=np.float64)
    dist = np.hypot(x, y)
    bearing = np.degrees(np.arctan2(np.abs(y), x))
    cls = df["label_class"].to_numpy(dtype=object)
    return t, dist, bearing, cls


def obstacle_summary(uuid: str, frame_idx: int,
                     window_us: int = WINDOW_US,
                     max_dist: float = MAX_DIST_M,
                     n_views: int = 3) -> dict | None:
    """판정 단위 주변 객체 요약.

    n_views 로 카메라 시야를 정해 그 밖의 객체는 버린다 (3=전방 3뷰 127도,
    1=front-wide 단독 60도). 안 보이는 것을 프롬프트에 넣으면 모델이 확인할
    방법이 없어 과탐으로 이어지므로, 화각은 실제 사용한 뷰와 맞춰야 한다.

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
    t, dist, bearing, cls = ob
    fov_half = VIEW_FOV_HALF_DEG.get(n_views, VIEW_FOV_HALF_DEG[3])
    sel = ((np.abs(t - t_us) <= window_us) & (dist <= max_dist)
           & (bearing <= fov_half))
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
    머리말은 edge_case_mining 이 egomotion 문구와 구분하는 데 쓰므로
    OBSTACLE_FACT_PREFIX 와 일치해야 한다.
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


# ---------------------------------------------------------------------------
# 주행 경로 침범 판정 (Q3 교차검증용)
#
# VLM 의 Q3("경로를 막는가")는 이미지만 보고 눈대중으로 내리는 판단인데, 이건
# 본질적으로 기하 문제다 - 객체가 자차 진행 방향의 좁은 통로 안에 있는가.
# 3D 라벨이 있으면 계산으로 풀 수 있으므로, 모델 답과 대조해 불일치 건을
# 검수 우선순위로 올린다. 프롬프트에는 넣지 않는다 (모델 동작을 건드리면
# 카테고리 나열이 죽는 것을 확인했으므로 - build_vlm_prompt 위 주석 참고).
#
# 한계: 통로를 직선으로 본다. 자차가 선회 중이면 실제 경로는 휘지만,
# egomotion 의 curvature 부호와 rig y 축 부호의 대응을 확정하지 못해
# 반대로 휘게 만들 위험이 있어 넣지 않았다. 선회 구간에서는 판정이
# 보수적으로(덜 잡히게) 틀릴 수 있다.
# ---------------------------------------------------------------------------
def _yaw_from_quat(qx, qy, qz, qw):
    """쿼터니언 -> rig 평면상의 yaw (rad)."""
    return np.arctan2(2 * (qw * qz + qx * qy),
                      1 - 2 * (qy ** 2 + qz ** 2))


@lru_cache(maxsize=1024)
def _obstacle_boxes(uuid: str):
    """(t, x, y, half_extent_y, cls) - 경로 판정에 필요한 최소 배열."""
    df = _read_label("obstacle.offline", uuid)
    if df is None or len(df) == 0:
        return None
    t = df["timestamp_us"].to_numpy(dtype=np.float64)
    x = df["center_x"].to_numpy(dtype=np.float64)
    y = df["center_y"].to_numpy(dtype=np.float64)
    yaw = _yaw_from_quat(df["orientation_x"].to_numpy(),
                         df["orientation_y"].to_numpy(),
                         df["orientation_z"].to_numpy(),
                         df["orientation_w"].to_numpy())
    # 회전한 직사각형을 rig 축에 투영했을 때의 y 방향 반폭
    half_y = 0.5 * (np.abs(df["size_x"].to_numpy() * np.sin(yaw))
                    + np.abs(df["size_y"].to_numpy() * np.cos(yaw)))
    cls = df["label_class"].to_numpy(dtype=object)
    return t, x, y, half_y, cls


def path_intrusion(uuid: str, frame_idx: int,
                   half_width: float = EGO_HALF_WIDTH_M,
                   lookahead: float = LOOKAHEAD_M,
                   window_us: int = WINDOW_US) -> dict | None:
    """자차 전방 통로를 침범하는 객체가 있는지 3D 라벨로 판정.

    통로 = 0 < x < lookahead 이고 |y| - (객체 반폭) < half_width 인 영역.
    객체의 실제 크기를 고려하므로, 중심은 통로 밖이어도 차체가 걸치면 잡는다.

    반환: {"blocked", "n_in_path", "nearest_m", "nearest_class", "objects"}
          라벨이 없으면 None.
    """
    ts = frame_timestamps(uuid)
    boxes = _obstacle_boxes(uuid)
    if ts is None or boxes is None:
        return None
    if not (0 <= frame_idx < len(ts)):
        return None

    t_us = float(ts[frame_idx])
    t, x, y, half_y, cls = boxes
    sel = ((np.abs(t - t_us) <= window_us)
           & (x > 0) & (x <= lookahead)
           & (np.abs(y) - half_y < half_width))
    if not sel.any():
        return {"blocked": False, "n_in_path": 0,
                "nearest_m": None, "nearest_class": None, "objects": []}

    xs, ys, cs = x[sel], y[sel], cls[sel]
    order = np.argsort(xs)
    objs = [{"class": str(cs[i]), "x_m": round(float(xs[i]), 1),
             "y_m": round(float(ys[i]), 1)} for i in order]
    return {"blocked": True, "n_in_path": int(sel.sum()),
            "nearest_m": round(float(xs[order[0]]), 1),
            "nearest_class": str(cs[order[0]]),
            "objects": objs}


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from egomotion import CAMERA_DIR, REF_VIEW

    ap = argparse.ArgumentParser(description="obstacle 조회 점검")
    ap.add_argument("--uuid")
    ap.add_argument("--limit-clips", type=int, default=3)
    ap.add_argument("--timestamps-per-clip", type=int, default=5)
    ap.add_argument("--n-views", type=int, default=3, choices=[1, 3],
                    help="카메라 화각 필터: 3=전방 3뷰(127도), 1=front-wide(60도)")
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
            s = obstacle_summary(u, int(fi), n_views=args.n_views)
            if s is None:
                print(f"  f{fi:4d}  (no obstacle label)")
                continue
            p = path_intrusion(u, int(fi))
            if p and p["blocked"]:
                tag = (f"PATH BLOCKED by {p['nearest_class']} @{p['nearest_m']}m "
                       f"({p['n_in_path']} in path)")
            else:
                tag = "path clear"
            print(f"  f{fi:4d}  n={s['total']:3d}  {tag}")
            print(f"          {describe_obstacles(s)}")

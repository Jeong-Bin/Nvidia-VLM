#!/usr/bin/env python3
"""자차 미래 궤적을 카메라 이미지에 투영해 그린다.

왜 이미지에 그리는가:
  자차 운동을 텍스트로 주는 시도는 세 번 실패했다 - 3D bbox 요약(과탐 폭증),
  ego-track 시계열(F1 -5%p), timeline 서술(이득 없음). 반면 요약 한 문장은
  효과가 있었다(+6.8%p). 모델이 짧은 서술은 쓰지만 긴 수치·구조화된 텍스트는
  영상과 결합하지 못한다는 뜻이다. 그렇다면 시공간 정보는 텍스트가 아니라
  픽셀로 줘야 한다 - 궤적이 도로 위에 그려지면 "이 보행자가 내 경로 위에
  있는가"가 별도 추론 없이 보인다.

투영 모델:
  NVIDIA PhysicalAI-AV 카메라는 ftheta(어안 다항식)이라 pinhole 이 안 맞는다.
  calibration 이 angle_to_pixeldist_poly 를 주므로 그대로 쓴다:
    rig 좌표 -> extrinsic 역변환 -> 카메라 좌표
    -> theta = atan2(hypot(x,y), z)        광축 대비 각도
    -> r = poly(theta)                     각도 -> 픽셀 반경
    -> (u,v) = principal_point + r*(cos,sin)

  검산(합성 점): 정면 10m 지면점이 화면 중앙(x=960)에 오고, 5->50m 로 멀어지면
  y 가 922->565 로 지평선에 수렴하며, 좌우 3m 가 중심 기준 대칭(633/1269)이다.

궤적 길이:
  기본 5초. egomotion 이 카메라보다 먼저 끝나는 클립이 3.8% 있어(76/1998)
  그런 구간에서는 남은 만큼만 짧게 그린다 - 없는 미래를 외삽하지 않는다.
  실측: 전체 1,208,545 프레임 중 99.20%가 5초를 온전히 확보한다.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from egomotion import _read_label, frame_timestamps

ROOT = Path(__file__).resolve().parent
CALIB_DIR = ROOT / "pav_sample" / "calibration"

# 미래 몇 초를 그릴지. 5초면 시속 40km 에서 약 55m 앞까지다.
TRAJ_HORIZON_S = 5.0
# 차폭(m). 2줄 모드에서 좌우 간격. NVIDIA 승용차 실측 폭 약 1.9m.
VEHICLE_WIDTH_M = 1.9
# 궤적 선 색 (BGR). 순수 초록은 도로/차량 색과 겹치지 않아 눈에 띈다.
TRAJ_COLOR = (0, 255, 0)
TRAJ_THICK_RATIO = 0.006          # 이미지 폭 대비. 896px 에서 약 5px
# 불투명도 (0~1). 1.0 이면 완전히 덮어쓴다.
#
# 왜 조절이 필요한가: 궤적은 자차 진행 경로 위에 그려지므로, 하필 그 경로에
# 있는 객체(우리가 찾으려는 바로 그 edge-case)를 가린다. 완전 불투명이면
# 모델이 보행자 대신 초록 선을 보게 된다. 반투명이면 경로도 보이고 그 위의
# 객체도 함께 보인다.
TRAJ_ALPHA = 1.0


@lru_cache(maxsize=1)
def _calib():
    """(intrinsics, extrinsics) DataFrame. 없으면 (None, None)."""
    fi, fe = CALIB_DIR / "camera_intrinsics.parquet", CALIB_DIR / "sensor_extrinsics.parquet"
    if not (fi.exists() and fe.exists()):
        return None, None
    return pd.read_parquet(fi), pd.read_parquet(fe)


def has_calibration(uuid: str, cam: str = "camera_front_wide_120fov") -> bool:
    intr, extr = _calib()
    if intr is None:
        return False
    return (uuid, cam) in intr.index and (uuid, cam) in extr.index


def _quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


@lru_cache(maxsize=512)
def _cam_model(uuid: str, cam: str):
    """(principal_point, poly, max_angle, (W,H), R, t). 없으면 None."""
    intr, extr = _calib()
    if intr is None or (uuid, cam) not in intr.index or (uuid, cam) not in extr.index:
        return None
    mp = json.loads(intr.loc[(uuid, cam)].model_parameters)
    e = extr.loc[(uuid, cam)]
    return (np.asarray(mp["principal_point"], dtype=float),
            np.asarray(mp["angle_to_pixeldist_poly"], dtype=float)[::-1],
            float(mp["max_angle"]), tuple(mp["resolution"]),
            _quat_to_R(e.qx, e.qy, e.qz, e.qw),
            np.array([e.x, e.y, e.z], dtype=float))


def project_rig(uuid: str, pts_rig: np.ndarray, cam: str = "camera_front_wide_120fov"):
    """rig 좌표 (N,3) -> (픽셀 (N,2), 유효 마스크, (W,H)). 캘리브 없으면 None."""
    m = _cam_model(uuid, cam)
    if m is None:
        return None
    pp, poly, max_ang, (W, H), R, t = m
    pc = (np.asarray(pts_rig, dtype=float) - t) @ R
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    theta = np.arctan2(np.hypot(x, y), z)
    rd = np.polyval(poly, theta)
    ang = np.arctan2(y, x)
    uv = np.stack([pp[0] + rd * np.cos(ang), pp[1] + rd * np.sin(ang)], 1)
    ok = (theta <= max_ang) & (z > 0) & \
         (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv, ok, (W, H)


@lru_cache(maxsize=256)
def _ego_pose(uuid: str):
    """(t, pos(N,3), quat(N,4)). 없으면 None."""
    df = _read_label("egomotion", uuid)
    if df is None or len(df) == 0:
        return None
    df = df.sort_values("timestamp")
    return (df["timestamp"].to_numpy(dtype=np.float64),
            df[["x", "y", "z"]].to_numpy(dtype=np.float64),
            df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64))


def future_path_rig(uuid: str, frame_idx: int, horizon_s: float = TRAJ_HORIZON_S):
    """frame_idx 시점 기준 미래 horizon_s 초의 자차 위치를 그 시점 rig 좌표로.

    미래가 모자라면 있는 만큼만 돌려준다 (외삽하지 않는다). 클립 끝처럼
    남은 구간이 거의 없으면 점이 2개 미만이 되고, 그 경우 None 이다.
    """
    ts = frame_timestamps(uuid)
    pose = _ego_pose(uuid)
    if ts is None or pose is None or not (0 <= frame_idx < len(ts)):
        return None
    te, pos, quat = pose
    t_now = float(ts[frame_idx])
    if t_now < te[0] or t_now > te[-1]:
        return None
    sel = (te >= t_now) & (te <= t_now + horizon_s * 1e6)
    if sel.sum() < 2:
        return None
    i0 = int(np.clip(np.searchsorted(te, t_now), 0, len(te) - 1))
    # world -> 현재 rig
    return (pos[sel] - pos[i0]) @ _quat_to_R(*quat[i0])


def _offset_lines(path_rig: np.ndarray, half_w: float):
    """중심 궤적을 좌우로 half_w 만큼 민 두 줄. 진행방향 법선을 쓴다."""
    d = np.gradient(path_rig[:, :2], axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = np.divide(d, n, out=np.zeros_like(d), where=n > 1e-9)
    normal = np.stack([-d[:, 1], d[:, 0]], 1)          # 좌측 법선
    out = []
    for s in (+1, -1):
        p = path_rig.copy()
        p[:, :2] = p[:, :2] + s * half_w * normal
        out.append(p)
    return out


def draw_trajectory(img_bgr, uuid: str, frame_idx: int,
                    cam: str = "camera_front_wide_120fov",
                    horizon_s: float = TRAJ_HORIZON_S,
                    mode: str = "center", color=TRAJ_COLOR,
                    thickness: int | None = None,
                    alpha: float = TRAJ_ALPHA) -> bool:
    """img_bgr 위에 궤적을 그린다 (제자리 수정). 그렸으면 True.

    mode  : "center" 차량 중심 1줄 | "width" 차폭 2줄
    alpha : 불투명도 0~1. 1 미만이면 별도 레이어에 그린 뒤 합성해서,
            겹치는 선끼리 두 번 섞여 진해지는 일이 없게 한다.
    이미지 크기는 원본(캘리브 해상도)과 달라도 된다 - 픽셀 좌표를 이미지
    크기에 맞춰 비례 축소하므로 입력 해상도에 자동으로 맞는다.
    """
    path = future_path_rig(uuid, frame_idx, horizon_s)
    if path is None:
        return False
    lines = ([path] if mode == "center"
             else _offset_lines(path, VEHICLE_WIDTH_M / 2.0))

    h, w = img_bgr.shape[:2]
    if thickness is None:
        thickness = max(2, int(round(w * TRAJ_THICK_RATIO)))

    # 반투명이면 선을 빈 레이어에 모아 그린 뒤 한 번에 합성한다. 이미지에
    # 직접 반투명으로 그리면 두 줄이 교차하는 지점만 두 번 섞여 진해진다.
    alpha = float(np.clip(alpha, 0.0, 1.0))
    canvas = img_bgr if alpha >= 1.0 else np.zeros_like(img_bgr)
    mask = None if alpha >= 1.0 else np.zeros(img_bgr.shape[:2], np.uint8)

    drew = False
    for line in lines:
        pr = project_rig(uuid, line, cam)
        if pr is None:
            return False
        uv, ok, (W, H) = pr
        if ok.sum() < 2:
            continue
        # 캘리브 해상도 -> 실제 이미지 해상도
        pts = uv[ok] * np.array([w / W, h / H])
        pts = pts.astype(np.int32)
        # 화면 밖 구간에서 끊긴 점들을 직선으로 이으면 화면을 가로지르는
        # 가짜 선이 생긴다. 연속된 구간끼리만 잇는다.
        idx = np.where(ok)[0]
        brk = np.where(np.diff(idx) > 1)[0] + 1
        for seg in np.split(np.arange(len(pts)), brk):
            if len(seg) >= 2:
                poly = [pts[seg].reshape(-1, 1, 2)]
                cv2.polylines(canvas, poly, False, color, thickness,
                              cv2.LINE_AA)
                if mask is not None:
                    cv2.polylines(mask, poly, False, 255, thickness,
                                  cv2.LINE_AA)
                drew = True

    if drew and mask is not None:
        m = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
        np.copyto(img_bgr,
                  (img_bgr.astype(np.float32) * (1 - m)
                   + canvas.astype(np.float32) * m).astype(np.uint8))
    return drew

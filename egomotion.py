#!/usr/bin/env python3
"""egomotion 라벨 조회 - 판정 단위(uuid, frame_idx)의 자차 운동 상태를 알아낸다.

NVIDIA PhysicalAI-AV 는 클립마다 egomotion 파일을 제공한다(100Hz):
  timestamp(us), qx..qw, x,y,z, vx,vy,vz, ax,ay,az, curvature

이걸 카메라의 timestamps.parquet(frame_index -> timestamp us)와 같은 시계로
조인하면, 각 프레임 시점의 정확한 속도/가속도/곡률을 얻는다.

용도:
  1) 1단계 캡션 프롬프트에 "사실"로 주입 - 모델이 6장 이미지로 정지/주행을
     추측할 필요가 없어지고, 남는 여력을 특이사항 서술에 쓴다.
  2) 2단계 결과 후처리 - 속도 0인데 Normal Driving 으로 분류된 건 등을 교정.
  3) 후보 선별 - 급제동/장시간정지/큰 곡률은 그 자체로 edge-case 후보.

egomotion 라벨은 zip 청크 안에 들어있으므로(파일 수 = 클립 수), 압축을
풀지 않고 zip 에서 직접 읽되 uuid -> zip 위치를 한 번만 인덱싱해 캐시한다.
"""
from __future__ import annotations

import io
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LABELS_DIR = ROOT / "pav_sample" / "labels"
CAMERA_DIR = ROOT / "pav_sample" / "camera"
REF_VIEW = "camera_front_wide_120fov"

# 정지 판정 임계값 (m/s). 0.5 m/s = 1.8 km/h - 센서 노이즈보다 크고
# "기어가는(creeping)" 상태보다는 작게 잡는다.
STOP_SPEED = 0.5
# 서행(creeping) 상한 (m/s). 2.0 m/s = 7.2 km/h
CREEP_SPEED = 2.0
# 급제동 판정 (m/s^2)
HARD_BRAKE_AX = -3.0
# 유의미한 곡률 (1/m). 0.02 => 반경 50m
TURN_CURVATURE = 0.02


# ---------------------------------------------------------------------------
# uuid -> (zip 경로, zip 내부 파일명) 인덱스
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _zip_index(kind: str) -> dict[str, tuple[str, str]]:
    """kind='egomotion' | 'obstacle.offline' 의 uuid -> (zip, member) 매핑.

    라벨이 zip 이 아니라 이미 풀린 parquet 로 존재하면 (parquet 경로, "") 로 둔다.
    """
    d = LABELS_DIR / kind
    idx: dict[str, tuple[str, str]] = {}
    if not d.exists():
        return idx

    # 이미 풀려 있는 parquet 우선
    for p in d.glob(f"*.{kind}.parquet"):
        idx[p.name.split(".")[0]] = (str(p), "")

    for zp in sorted(d.glob(f"{kind}.chunk_*.zip")):
        try:
            with zipfile.ZipFile(zp) as z:
                for name in z.namelist():
                    uuid = name.split(".")[0]
                    idx.setdefault(uuid, (str(zp), name))
        except zipfile.BadZipFile:
            continue
    return idx


def _read_label(kind: str, uuid: str) -> pd.DataFrame | None:
    loc = _zip_index(kind).get(uuid)
    if loc is None:
        return None
    path, member = loc
    if not member:
        return pd.read_parquet(path)
    with zipfile.ZipFile(path) as z:
        return pd.read_parquet(io.BytesIO(z.read(member)))


# ---------------------------------------------------------------------------
# 프레임 시점 -> 타임스탬프
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4096)
def frame_timestamps(uuid: str, view: str = REF_VIEW) -> np.ndarray | None:
    """frame_index -> timestamp(us) 배열. 없으면 None."""
    p = CAMERA_DIR / view / f"{uuid}.{view}.timestamps.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    return df[col].to_numpy()


@lru_cache(maxsize=2048)
def _ego_arrays(uuid: str):
    """(t, speed, ax_long, curvature) 배열 튜플. 없으면 None.

    ax_long 은 차량 전진방향 가속도(ax)를 그대로 쓴다 - egomotion 은
    이미 차량 좌표계 기준이다.
    """
    df = _read_label("egomotion", uuid)
    if df is None or len(df) == 0:
        return None
    df = df.sort_values("timestamp")
    t = df["timestamp"].to_numpy(dtype=np.float64)
    speed = np.hypot(df["vx"].to_numpy(), df["vy"].to_numpy())
    ax = df["ax"].to_numpy(dtype=np.float64)
    curv = df["curvature"].to_numpy(dtype=np.float64)
    return t, speed, ax, curv


def ego_state(uuid: str, frame_idx: int) -> dict | None:
    """판정 단위 (uuid, frame_idx) 의 자차 운동 상태.

    반환: {speed_ms, speed_kmh, ax, curvature, motion, is_stopped, ...}
          라벨이 없으면 None (파이프라인은 이 경우 기존처럼 동작해야 한다).
    """
    ts = frame_timestamps(uuid)
    ego = _ego_arrays(uuid)
    if ts is None or ego is None:
        return None
    if not (0 <= frame_idx < len(ts)):
        return None

    t_us = float(ts[frame_idx])
    t, speed, ax, curv = ego
    # 카메라 시점이 egomotion 구간을 벗어나면 신뢰할 수 없다
    if t_us < t[0] - 1e6 or t_us > t[-1] + 1e6:
        return None

    sp = float(np.interp(t_us, t, speed))
    a = float(np.interp(t_us, t, ax))
    c = float(np.interp(t_us, t, curv))

    if sp < STOP_SPEED:
        motion = "stopped"
    elif sp < CREEP_SPEED:
        motion = "creeping forward very slowly"
    else:
        motion = "driving"

    return {
        "speed_ms": sp,
        "speed_kmh": sp * 3.6,
        "ax": a,
        "curvature": c,
        "motion": motion,
        "is_stopped": sp < STOP_SPEED,
        "is_hard_braking": a < HARD_BRAKE_AX,
        "is_turning": abs(c) > TURN_CURVATURE,
        "turn_dir": ("left" if c > 0 else "right") if abs(c) > TURN_CURVATURE else None,
    }


def describe_ego(state: dict | None) -> str:
    """ego_state -> 1단계 프롬프트에 넣을 짧은 사실 문구.

    캡션이 이 문구를 그대로 베끼지 않도록 최대한 건조하게, 한 줄로 쓴다.
    """
    if state is None:
        return ""
    sp = state["speed_kmh"]
    if state["is_stopped"]:
        s = "The ego-vehicle is STOPPED (0 km/h)."
    else:
        s = f"The ego-vehicle is MOVING at about {sp:.0f} km/h."
    if state["is_hard_braking"]:
        s += f" It is braking hard ({state['ax']:.1f} m/s^2)."
    elif state["ax"] < -1.0:
        s += " It is slowing down."
    elif state["ax"] > 1.0:
        s += " It is accelerating."
    if state["is_turning"]:
        s += f" It is turning {state['turn_dir']}."
    return s


def has_egomotion(uuid: str) -> bool:
    return uuid in _zip_index("egomotion")


if __name__ == "__main__":
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser(description="egomotion 조회 점검")
    ap.add_argument("--uuid")
    ap.add_argument("--limit-clips", type=int, default=5)
    ap.add_argument("--timestamps-per-clip", type=int, default=10)
    ap.add_argument("--coverage", action="store_true",
                    help="전체 클립의 egomotion/obstacle 라벨 커버리지 집계")
    args = ap.parse_args()

    cam_uuids = sorted(p.name.split(".")[0]
                       for p in (CAMERA_DIR / REF_VIEW).glob("*.mp4"))

    if args.coverage:
        ego_idx, ob_idx = _zip_index("egomotion"), _zip_index("obstacle.offline")
        n = len(cam_uuids)
        ne = sum(u in ego_idx for u in cam_uuids)
        no = sum(u in ob_idx for u in cam_uuids)
        print(f"camera clips        : {n}")
        print(f"with egomotion      : {ne}  ({100*ne/n:.1f}%)")
        print(f"with obstacle       : {no}  ({100*no/n:.1f}%)")
        raise SystemExit

    uuids = [args.uuid] if args.uuid else cam_uuids[:args.limit_clips]
    motions = Counter()
    for u in uuids:
        ts = frame_timestamps(u)
        if ts is None:
            print(f"{u}: no camera timestamps")
            continue
        idxs = np.linspace(0, len(ts) - 1, args.timestamps_per_clip, dtype=int)
        print(f"\n=== {u} ({len(ts)} frames) ===")
        for fi in idxs:
            st = ego_state(u, int(fi))
            if st is None:
                print(f"  f{fi:4d}  (no egomotion)")
                continue
            motions[st["motion"]] += 1
            print(f"  f{fi:4d}  {st['speed_kmh']:6.1f} km/h  ax={st['ax']:6.2f}  "
                  f"curv={st['curvature']:8.4f}  {st['motion']:12s} "
                  f"| {describe_ego(st)}")
    print("\nmotion distribution:", dict(motions))

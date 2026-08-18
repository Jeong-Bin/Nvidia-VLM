#!/usr/bin/env python3
"""egomotion 라벨 조회 - 자차의 속도/가속도/조향을 라벨에서 직접 읽는다.

조회 단위가 두 가지다. 어느 쪽을 쓸지는 모델에 무엇을 보여주는지로 정한다:
  프레임 단위 (uuid, frame_idx) - ego_state / ego_behavior_change
      한 순간의 스냅샷과 그 앞 5초. run_all.sh 의 프레임 단위 판정용.
  클립 단위  (uuid)             - ego_clip_behavior
      20초 클립 전 구간 요약. 클립 모드(영상 통째 입력)용.


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


# ---------------------------------------------------------------------------
# 행동 변화 (nuReasoning 2단계: Ego Behavior Summary)
# ---------------------------------------------------------------------------
# nuReasoning 의 마이닝 원칙은 "특이 요소가 자차 행동을 실제로 바꿨는가" 이다
# (Fig. S1: "The mere presence of unusual or critical objects is not
# sufficient"). ego_state() 는 한 시점의 스냅샷이라 이 질문에 답할 수 없다.
# 여기서는 프레임 주변 구간을 통째로 보고 "무엇이 어떻게 변했는지"를 뽑는다.
#
# 영상 20초를 VLM 에 넣어 추측하게 하는 대신 100Hz 라벨에서 계산한다.
# 토큰은 한 줄이고 값은 센서 그대라 더 정확하다. 단, 이건 자차 행동만
# 답한다 - "왜" 그랬는지(공사장/보행자/터널)는 여전히 이미지 몫이다.

# 되돌아볼 구간 (초). 감속-정지 같은 반응은 대개 3~5초 안에 끝난다.
BEHAVIOR_WINDOW_S = 5.0
# 유의미한 속도 변화 (m/s). 2 m/s = 7.2 km/h.
# 구간 순변화가 이 값을 넘으면 그것만으로 감속/가속으로 본다.
SPEED_CHANGE_MS = 2.0
# 유의미한 감속/가속 (m/s^2) - 구간 중 최저/최고 순간가속도 기준.
# 순간값은 노이즈로 한 번씩 튀므로 단독 근거로 쓰지 않는다: 순변화가 같은
# 부호로 최소 SPEED_MIN_MS 이상 있을 때만 보조 근거로 인정한다.
DECEL_MS2 = -1.5
ACCEL_MS2 = 1.5
# 순간가속도를 근거로 쓸 때 요구하는 최소 순변화 (m/s). 0.5 m/s = 1.8 km/h
SPEED_MIN_MS = 0.5
# 구간 내 누적 heading 변화가 이보다 크면 조향한 것으로 본다 (deg)
HEADING_CHANGE_DEG = 15.0

# --- notable 판정 임계값 -----------------------------------------------------
# 실측(1,080 units, 120 클립): 5초 구간의 |속도변화| 중앙값이 6.3 km/h 라
# "조금이라도 변했는가"(changed)는 65% 에서 참이 되어 선별력이 없다.
# nuReasoning 이 요구하는 것은 "clear, non-trivial change" 이므로, 흔한
# 주행 중 속도 흔들림과 실제 반응을 가르는 별도의 (더 엄격한) 선을 둔다.
# 아래 값들은 각각 상위 ~5-13% 에서만 참이 된다.
NOTABLE_DECEL_MS2 = -3.0        # 급제동    (실측 12.7%)
NOTABLE_DROP_KMH = -15.0        # 큰 속도 하락 (실측 12.0%)
NOTABLE_HEADING_DEG = 30.0      # 뚜렷한 조향 (실측 9.8%)


def _yaw_deg(qx, qy, qz, qw) -> np.ndarray:
    """쿼터니언 -> yaw(deg). 차량 heading."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.degrees(np.arctan2(siny, cosy))


@lru_cache(maxsize=2048)
def _ego_arrays_full(uuid: str):
    """(t, speed, ax, curvature, yaw_deg). _ego_arrays 에 heading 을 더한 것."""
    df = _read_label("egomotion", uuid)
    if df is None or len(df) == 0:
        return None
    df = df.sort_values("timestamp")
    t = df["timestamp"].to_numpy(dtype=np.float64)
    speed = np.hypot(df["vx"].to_numpy(), df["vy"].to_numpy())
    ax = df["ax"].to_numpy(dtype=np.float64)
    curv = df["curvature"].to_numpy(dtype=np.float64)
    yaw = _yaw_deg(df["qx"].to_numpy(), df["qy"].to_numpy(),
                   df["qz"].to_numpy(), df["qw"].to_numpy())
    return t, speed, ax, curv, yaw


def ego_behavior_change(uuid: str, frame_idx: int,
                        window_s: float = BEHAVIOR_WINDOW_S) -> dict | None:
    """프레임 시점까지 window_s 초 동안 자차 행동이 어떻게 변했는지.

    egomotion 은 카메라 클립보다 뒤로는 길게(중앙값 ~108초) 이어지지만
    앞쪽 여유는 거의 없다(중앙값 0.1초). 따라서 클립 앞부분 프레임에서는
    되돌아볼 구간이 모자란다 - 그 경우 실제 확보된 길이를 span_s 로 알리고,
    너무 짧으면(<1초) None 을 반환해 호출부가 조용히 건너뛰게 한다.

    반환:
      span_s          실제로 관찰한 구간 길이 (요청한 window_s 보다 짧을 수 있음)
      speed_start/end 구간 시작/끝 속도 (km/h)
      speed_delta     end - start (km/h). 음수면 감속
      min_ax, max_ax  구간 내 최저/최고 순간 종가속도 (m/s^2)
      heading_delta   구간 누적 heading 변화 (deg). +면 좌회전
      came_to_stop    움직이다가 멈췄는가
      started_moving  멈춰있다가 출발했는가
      decelerating / accelerating / steering  유의미한 변화 여부
      changed         위 중 하나라도 참 - "행동이 바뀌었다"
    """
    ts = frame_timestamps(uuid)
    ego = _ego_arrays_full(uuid)
    if ts is None or ego is None:
        return None
    if not (0 <= frame_idx < len(ts)):
        return None

    t, speed, ax, curv, yaw = ego
    t_end = float(ts[frame_idx])
    if t_end < t[0] or t_end > t[-1]:
        return None
    t_start = max(t_end - window_s * 1e6, float(t[0]))

    span_s = (t_end - t_start) / 1e6
    if span_s < 1.0:            # 구간이 1초도 안 되면 변화를 논할 수 없다
        return None

    sel = (t >= t_start) & (t <= t_end)
    if sel.sum() < 2:
        return None

    sp_w, ax_w, yaw_w = speed[sel], ax[sel], yaw[sel]
    v0, v1 = float(sp_w[0]), float(sp_w[-1])

    # heading 은 ±180 에서 튀므로 unwrap 후 누적 변화를 본다
    yaw_u = np.unwrap(np.radians(yaw_w))
    heading_delta = float(np.degrees(yaw_u[-1] - yaw_u[0]))

    min_ax, max_ax = float(ax_w.min()), float(ax_w.max())
    dv = v1 - v0
    came_to_stop = v0 >= CREEP_SPEED and v1 < STOP_SPEED
    started_moving = v0 < STOP_SPEED and v1 >= CREEP_SPEED
    # 순변화가 1차 근거. 순간가속도는 순변화가 같은 방향일 때만 보조로 쓴다 -
    # 그렇지 않으면 감속 구간에서 튄 +ax 하나로 "가속"이 되어버린다.
    decelerating = dv <= -SPEED_CHANGE_MS or (min_ax <= DECEL_MS2
                                              and dv <= -SPEED_MIN_MS)
    accelerating = dv >= SPEED_CHANGE_MS or (max_ax >= ACCEL_MS2
                                             and dv >= SPEED_MIN_MS)
    steering = abs(heading_delta) >= HEADING_CHANGE_DEG

    return {
        "span_s": span_s,
        "speed_start": v0 * 3.6,
        "speed_end": v1 * 3.6,
        "speed_delta": (v1 - v0) * 3.6,
        "min_ax": min_ax,
        "max_ax": max_ax,
        "heading_delta": heading_delta,
        "came_to_stop": came_to_stop,
        "started_moving": started_moving,
        "decelerating": decelerating,
        "accelerating": accelerating,
        "steering": steering,
        "is_hard_braking": min_ax < HARD_BRAKE_AX,
        # 조금이라도 변했는가. 흔하다(실측 65%) - 필터로 쓰지 말 것.
        "changed": bool(came_to_stop or started_moving or decelerating
                        or accelerating or steering),
        # "clear, non-trivial change" 인가. 드물다(실측 ~25%) - 이쪽이 필터용.
        "notable": bool(came_to_stop or started_moving
                        or min_ax <= NOTABLE_DECEL_MS2
                        or (dv * 3.6) <= NOTABLE_DROP_KMH
                        or abs(heading_delta) >= NOTABLE_HEADING_DEG),
    }


def ego_clip_behavior(uuid: str) -> dict | None:
    """클립 전체(약 20초) 동안의 자차 행동 요약.

    ego_behavior_change() 와의 차이 - 그쪽은 한 프레임에서 5초를 되돌아본다.
    프레임 단위로 판정하던 시절에는 그게 맞았지만, 클립 모드는 20초 영상을
    통째로 넣으므로 마지막 5초만 서술하면 시간축이 어긋난다.

    실측(랜덤 250클립): 마지막 5초는 클립 속도 변동폭의 중앙값 30% 밖에
    담지 못한다. 클립 중 급제동(<-3 m/s^2)이 있었는데 마지막 5초에는 없는
    경우가 24%, 정차했는데 마지막 5초엔 아닌 경우가 15% 다. 즉 급제동 클립
    4개 중 1개는 "일정 속도를 유지했다"로 잘못 서술된다.

    구간 요약이라 시작/끝 값만으로는 부족하다 - 중간에 있었던 최저속도와
    급제동 시점을 함께 돌려준다. 시각은 클립 시작(첫 프레임)부터 잰 초.

    반환 (ego_behavior_change 의 키를 모두 포함하고 아래를 더한다):
      speed_min/max       구간 최저/최고 속도 (km/h)
      t_min_ax            최대 감속이 일어난 시각 (s, 클립 시작 기준)
      t_speed_min         최저 속도 시각 (s)
      stopped_s           정차해 있던 총 시간 (s)
      ends_stopped        클립 끝에 멈춰 있는가
    """
    ts = frame_timestamps(uuid)
    ego = _ego_arrays_full(uuid)
    if ts is None or ego is None or len(ts) < 2:
        return None

    t, speed, ax, curv, yaw = ego
    t_beg, t_end = float(ts[0]), float(ts[-1])
    # 카메라 구간이 egomotion 밖이면 신뢰할 수 없다
    if t_end < t[0] or t_beg > t[-1]:
        return None
    t_beg = max(t_beg, float(t[0]))
    t_end = min(t_end, float(t[-1]))

    span_s = (t_end - t_beg) / 1e6
    if span_s < 1.0:
        return None

    sel = (t >= t_beg) & (t <= t_end)
    if sel.sum() < 2:
        return None

    t_w = t[sel]
    sp_w, ax_w, yaw_w = speed[sel], ax[sel], yaw[sel]
    v0, v1 = float(sp_w[0]), float(sp_w[-1])

    yaw_u = np.unwrap(np.radians(yaw_w))
    heading_delta = float(np.degrees(yaw_u[-1] - yaw_u[0]))

    i_min_ax = int(np.argmin(ax_w))
    i_sp_min = int(np.argmin(sp_w))
    min_ax, max_ax = float(ax_w[i_min_ax]), float(ax_w.max())
    dv = v1 - v0

    # 샘플 간격이 일정하다고 보고 정차 시간을 센다 (100Hz 라벨)
    dt_s = span_s / max(len(sp_w) - 1, 1)
    stopped_s = float((sp_w < STOP_SPEED).sum() * dt_s)

    came_to_stop = v0 >= CREEP_SPEED and v1 < STOP_SPEED
    started_moving = v0 < STOP_SPEED and v1 >= CREEP_SPEED
    decelerating = dv <= -SPEED_CHANGE_MS or (min_ax <= DECEL_MS2
                                              and dv <= -SPEED_MIN_MS)
    accelerating = dv >= SPEED_CHANGE_MS or (max_ax >= ACCEL_MS2
                                             and dv >= SPEED_MIN_MS)
    steering = abs(heading_delta) >= HEADING_CHANGE_DEG

    return {
        "span_s": span_s,
        "speed_start": v0 * 3.6,
        "speed_end": v1 * 3.6,
        "speed_delta": dv * 3.6,
        "speed_min": float(sp_w.min()) * 3.6,
        "speed_max": float(sp_w.max()) * 3.6,
        "min_ax": min_ax,
        "max_ax": max_ax,
        "t_min_ax": float(t_w[i_min_ax] - t_beg) / 1e6,
        "t_speed_min": float(t_w[i_sp_min] - t_beg) / 1e6,
        "stopped_s": stopped_s,
        "ends_stopped": v1 < STOP_SPEED,
        "heading_delta": heading_delta,
        "came_to_stop": came_to_stop,
        "started_moving": started_moving,
        "decelerating": decelerating,
        "accelerating": accelerating,
        "steering": steering,
        "is_hard_braking": min_ax < HARD_BRAKE_AX,
        "changed": bool(came_to_stop or started_moving or decelerating
                        or accelerating or steering),
        "notable": bool(came_to_stop or started_moving
                        or min_ax <= NOTABLE_DECEL_MS2
                        or (dv * 3.6) <= NOTABLE_DROP_KMH
                        or abs(heading_delta) >= NOTABLE_HEADING_DEG),
    }


def describe_clip_behavior(ch: dict | None) -> str:
    """ego_clip_behavior -> 프롬프트에 넣을 1~2 문장.

    describe_behavior() 와 방침은 같되(건조하게, 숫자는 센서 그대로) 구간
    요약이므로 "언제" 를 함께 적는다. 모델이 영상에서 본 사건과 이 수치를
    시각으로 맞춰볼 수 있어야 시간 정렬이 의미를 갖는다.
    """
    if ch is None:
        return ""
    w = f"Over these {ch['span_s']:.0f} seconds"

    lo, hi = ch["speed_min"], ch["speed_max"]
    if ch["came_to_stop"]:
        core = (f"the ego-vehicle slowed from {ch['speed_start']:.0f} km/h "
                f"and came to a stop")
    elif ch["started_moving"]:
        core = (f"the ego-vehicle pulled away from a stop to "
                f"{ch['speed_end']:.0f} km/h")
    elif hi - lo < 5.0:
        core = f"the ego-vehicle held a steady {hi:.0f} km/h"
    else:
        core = (f"the ego-vehicle went from {ch['speed_start']:.0f} to "
                f"{ch['speed_end']:.0f} km/h "
                f"(range {lo:.0f}-{hi:.0f} km/h)")

    s = f"{w} {core}"
    # "hard braking" 이라는 표현을 쓰지 않고 측정값만 적는다.
    #
    # 실측(100클립 A/B, 20260818): 이 문구가 들어간 실행에서 safety 정확도가
    # 급제동 클립에 한해 66% -> 37% 로 무너졌다 (급제동이 없는 클립은 72% ->
    # 69% 로 거의 그대로). 그런데 정답 라벨을 보면 급제동 클립 35건의 GT
    # safety 평균은 1.49 로, 급제동이 없는 65건의 1.45 와 사실상 같다. 즉 이
    # 데이터셋에서 급제동은 대개 신호/정체/교차로 때문이지 위험 반응이
    # 아닌데도, 모델이 "hard braking" 이라는 표현만 보고 등급을 올렸다.
    #
    # 감속 사실 자체는 2단계(Ego Behavior)에 필요하므로 버리지 않는다.
    # 판단은 영상을 보고 하라는 뜻에서, 해석이 담긴 말 대신 수치만 남긴다.
    if ch["is_hard_braking"]:
        s += (f", with a peak deceleration of {abs(ch['min_ax']):.1f} m/s^2 "
              f"about {ch['t_min_ax']:.0f} s in")
    if ch["steering"]:
        side = "left" if ch["heading_delta"] > 0 else "right"
        s += f", and turned {side} by {abs(ch['heading_delta']):.0f} degrees"
    s += "."

    # 중간에 멈춰 있었던 시간은 위 문장이 못 담는다 - 정차가 길면 따로 적는다.
    # came_to_stop 은 끝에 멈춘 경우라 이미 서술됐으므로 제외한다.
    if ch["stopped_s"] >= 2.0 and not ch["came_to_stop"]:
        s += f" It was stationary for about {ch['stopped_s']:.0f} s of that."
    return s


def describe_behavior(ch: dict | None) -> str:
    """ego_behavior_change -> 프롬프트에 넣을 한 문장.

    describe_ego() 와 같은 방침으로 건조하게 쓰되, 숫자는 변화량만 남긴다.
    현재 속도는 describe_ego() 가 이미 말하므로 여기서 반복하지 않는다.
    """
    if ch is None:
        return ""
    w = f"Over the past {ch['span_s']:.0f} seconds"

    if ch["came_to_stop"]:
        core = (f"the ego-vehicle slowed from {ch['speed_start']:.0f} km/h "
                f"and came to a stop")
    elif ch["started_moving"]:
        core = (f"the ego-vehicle pulled away from a stop to "
                f"{ch['speed_end']:.0f} km/h")
    elif ch["decelerating"] and ch["speed_delta"] < 0:
        core = (f"the ego-vehicle slowed from {ch['speed_start']:.0f} to "
                f"{ch['speed_end']:.0f} km/h")
    elif ch["accelerating"] and ch["speed_delta"] > 0:
        core = (f"the ego-vehicle sped up from {ch['speed_start']:.0f} to "
                f"{ch['speed_end']:.0f} km/h")
    else:
        # 속도 자체는 크게 안 변했는데 조향만 한 경우도 여기로 온다
        core = "the ego-vehicle held a steady speed"

    s = f"{w} {core}"
    if ch["is_hard_braking"]:
        s += f", braking hard ({ch['min_ax']:.1f} m/s^2)"
    if ch["steering"]:
        side = "left" if ch["heading_delta"] > 0 else "right"
        s += f", and turned {side} by {abs(ch['heading_delta']):.0f} degrees"
    return s + "."


if __name__ == "__main__":
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser(description="egomotion 조회 점검")
    ap.add_argument("--uuid")
    ap.add_argument("--limit-clips", type=int, default=5)
    ap.add_argument("--timestamps-per-clip", type=int, default=10)
    ap.add_argument("--coverage", action="store_true",
                    help="전체 클립의 egomotion/obstacle 라벨 커버리지 집계")
    ap.add_argument("--behavior", action="store_true",
                    help="행동 변화 추출(ego_behavior_change) 점검")
    ap.add_argument("--window", type=float, default=BEHAVIOR_WINDOW_S)
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

    if args.behavior:
        uuids = [args.uuid] if args.uuid else cam_uuids[:args.limit_clips]
        flags, n_none, n_units = Counter(), 0, 0
        for u in uuids:
            ts = frame_timestamps(u)
            if ts is None:
                continue
            idxs = np.linspace(0, len(ts) - 1, args.timestamps_per_clip,
                               dtype=int)
            print(f"\n=== {u} ({len(ts)} frames) ===")
            for fi in idxs:
                n_units += 1
                ch = ego_behavior_change(u, int(fi), window_s=args.window)
                if ch is None:
                    n_none += 1
                    print(f"  f{fi:4d}  (no window)")
                    continue
                for k in ("came_to_stop", "started_moving", "decelerating",
                          "accelerating", "steering"):
                    if ch[k]:
                        flags[k] += 1
                flags["changed" if ch["changed"] else "steady"] += 1
                print(f"  f{fi:4d}  span={ch['span_s']:.1f}s "
                      f"{ch['speed_start']:5.1f}->{ch['speed_end']:5.1f} km/h "
                      f"ax[{ch['min_ax']:5.1f},{ch['max_ax']:5.1f}] "
                      f"hdg={ch['heading_delta']:6.1f} "
                      f"| {describe_behavior(ch)}")
        print(f"\nunits={n_units}  no-window={n_none}")
        print("flags:", dict(flags))
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

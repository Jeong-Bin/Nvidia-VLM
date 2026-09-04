#!/usr/bin/env python3
"""
VLM 기반 반자동 edge-case (long-tail) mining.

판정 단위는 클립(uuid) 전체가 아니라 "클립 내 특정 순간"(uuid, frame_idx)
이다. 각 20초 클립에서 균등하게 timestamps_per_clip 개의 순간을 뽑고, 매
순간마다 전방 3뷰(front_wide, cross_left, cross_right)의 직전/현재 프레임
6장을 Qwen2.5-VL 에 한 번에 보여줘 JSON 으로 답을 받는다:

  {"verdict": "Normal" | "Special",
   "categories": [scene_category_B.json 의 special 카테고리명, ...],
   "evidence": "실제로 본 것을 요약한 짧은 구절"}

멀티라벨이므로 한 장면에 여러 카테고리가 동시에 붙을 수 있다
(예: 공사장에서 수신호 -> Road Construction + Manual Traffic Control).

Usage:
  # 스모크 테스트 (클립 2개 x 10 timestamp = 20 단위)
  python edge_case_mining.py --limit-clips 2

  # 센서 라벨을 사실로 함께 넣기 (기본은 둘 다 off)
  python edge_case_mining.py --use-egomotion --use-3dbbox

  # 전체
  python edge_case_mining.py
"""
import argparse
import json
import re
import time
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

# PyAV 9 에서 av.AVError 가 없어졌다(현재 18.0.0). 이름만 사라진 게 아니라
# except 절에서 av.AVError 를 평가하는 순간 AttributeError 가 터지므로,
# "열기 실패하면 0 을 돌려준다" 는 방어 코드가 도리어 프로세스를 죽인다 -
# 실측으로 NAS 첫 실행 때 8샤드가 전부 여기서 죽었다.
# FFmpegError 는 OSError 를 상속하지 않으므로 따로 잡아야 한다.
AV_ERROR = getattr(av, "AVError", None) or av.FFmpegError

from config import SCENE_JSON as CONFIG_SCENE_JSON, LABELS_JSON
from trajectory import (TRAJ_HORIZON_S as _TRAJ_HORIZON_S,
                        TRAJ_ALPHA as _TRAJ_ALPHA)
from constrained_tier import (TIER_LABELS, TIER_VALUES, tier_label,
                              tier_menu, tier_score,
                              safety_rubric_text, rarity_rubric_text,
                              contrast_text)
from prompts import (DIFFICULTY_AXES, DIFFICULTY_MIN, DIFFICULTY_MAX,
                     difficulty_block)

ROOT = Path(__file__).resolve().parent
CAMERA_DIR = ROOT / "pav_sample" / "camera"

# 프레임 소스. None 이면 CAMERA_DIR 아래의 mp4 파일을 직접 연다.
# --data 로 NAS 청크 zip 을 고르면 set_clip_source() 가 여기에 꽂는다.
CLIP_SOURCE = None
# 카테고리 정의와 정답 라벨의 기본값은 config.py 한 곳에서만 정한다.
# (예전에는 여기/evaluate_labels/aggregate_clip/셸 스크립트가 각자
#  다른 값을 들고 있어 같은 실행을 서로 다른 체계로 해석한 적이 있다.)
SCENE_JSON = CONFIG_SCENE_JSON

FRONT_VIEWS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
]
# front_wide 단독 모드에서 쓰는 뷰. FRONT_VIEWS[0] 과 같아야 한다
# (clip_frame_count / list_scene_uuids 가 [0] 을 기준 뷰로 쓰므로).
SINGLE_VIEW = [FRONT_VIEWS[0]]


def views_for(single_view: bool = False) -> list[str]:
    """이 실행에서 쓸 카메라 뷰 목록."""
    return list(SINGLE_VIEW if single_view else FRONT_VIEWS)


def category_slug(category: str) -> str:
    """카테고리명을 폴더명으로 쓰기 안전한 형태로 변환 (공백/하이픈 -> 언더스코어).

    OOD 도 그대로 "OOD" 폴더가 된다.
    """
    return category.strip().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# 라벨 로딩 (배열 스키마: {normal,special}.scenarios[].categories[])
#
# normal 도 special 과 완전히 동일한 구조를 쓴다 (예: "Normal Driving",
# "Normal Stop" 카테고리). 2단계 매칭에서는 이 둘을 special 카테고리와
# 동등한 후보로 취급한다 - is_normal 플래그로만 구분.
# ---------------------------------------------------------------------------
def load_labels(scene_json: Path):
    """scene_category.json -> [{scenario, category, synonyms, prompt_templates, is_normal}, ...]"""
    data = json.loads(scene_json.read_text(encoding="utf-8"))

    labels = []
    for section, is_normal in (("normal", True), ("special", False)):
        for scen in data.get(section, {}).get("scenarios", []):
            scenario_name = scen["name"]
            for cat in scen.get("categories", []):
                labels.append(
                    {
                        "scenario": scenario_name,
                        "category": cat["name"],
                        "synonyms": cat.get("synonyms", []),
                        "prompt_templates": cat.get("prompt_templates", []),
                        # 이 카테고리가 "아닌" 경우. 긍정 예시만으로는 경계가
                        # 안 잡히는 카테고리에만 쓴다 (없으면 빈 리스트).
                        "excludes": cat.get("excludes", []),
                        "is_normal": is_normal,
                    }
                )
    return labels


def _examples_for(lab, example_source: str, num_examples: int | None = None):
    """카테고리 하나에서 예시로 쓸 문자열 리스트를 뽑는다.

    example_source: "synonyms" (단순 객체/키워드 나열) 또는
                     "prompt_templates" (유의미한 상황을 서술하는 문장).
    num_examples: 앞에서부터 몇 개만 쓸지 (None 이면 전체).
    synonyms 가 없는 카테고리(예: normal)는 example_source 와 무관하게
    prompt_templates 로 대체한다.
    """
    if example_source == "synonyms" and lab["synonyms"]:
        items = lab["synonyms"]
    else:
        items = lab["prompt_templates"]
    return items[:num_examples] if num_examples else items


def build_category_menu(labels, example_source: str = "synonyms",
                        num_examples: int = 2):
    """프롬프트에 넣을 special 카테고리 목록.

    형식: `- Road Construction(roadwork, traffic cone)`
    카테고리명과 예시를 한 덩어리로 묶어 제시하므로, 모델이 별도의 매칭 단계
    없이 정확한 카테고리명으로 바로 답할 수 있다.

    예시 개수는 2개가 기본. 카테고리당 예시를 전부 나열하면 판별이 경직되어
    다양성이 떨어진다는 관찰(20260724)에 따라 소수만 노출한다.
    normal 은 제외 - Q1(Normal/Special)이 그 역할을 하고, Q2 는 special
    카테고리만 나열하는 자리이기 때문.
    """
    lines = []
    for lab in labels:
        if lab["is_normal"]:
            continue
        ex = _examples_for(lab, example_source, num_examples)
        lines.append(f"- {lab['category']}({', '.join(ex)})" if ex
                     else f"- {lab['category']}")
        # 부정 조건은 별도 줄로. 실측(20260818, 100클립): Jaywalking 은 긍정
        # 예시에 "where there is no crosswalk" 가 있는데도 횡단보도를 정상
        # 통행하는 보행자를 20건 오탐했다 (FP 20 = 전체 FP 38 의 53%).
        # 모델이 "crossing the street" 자체를 카테고리로 읽고 있어서,
        # 긍정 예시를 늘리는 대신 무엇이 아닌지를 못박는다.
        for ex_line in lab["excludes"]:
            lines.append(f"    NOT this category: {ex_line}")
    return "\n".join(lines)


def special_categories(labels):
    """special 카테고리명 리스트 (출력 검증용)."""
    return [l["category"] for l in labels if not l["is_normal"]]


# ---------------------------------------------------------------------------
# 프레임 샘플링
#
# 판정 단위는 "클립(uuid)"이 아니라 "클립 내 특정 순간(frame_idx)" 이다.
# 20초 클립에서 균등하게 N개 timestamp 를 뽑고, 각 timestamp 마다 3뷰
# (front_wide/cross_left/cross_right)의 같은 frame_idx 프레임을 묶어
# 하나의 판정 단위로 쓴다 (뷰 간 프레임 수는 동일함이 확인됨; index 를
# 그대로 매칭 - 촬영 시각 오차는 최대 약 2프레임 수준으로 무시).
# ---------------------------------------------------------------------------
def _resize_to_pil(bgr, max_long_side: int):
    h, w = bgr.shape[:2]
    scale = max_long_side / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _read_frames_at(src, indices, max_long_side: int = 896):
    """비디오에서 여러 frame index 를 읽어 {idx: PIL.Image} 로 반환 (한 번의 open).

    src 는 경로 문자열이거나 file-like(zip 내부 구간). NAS 청크 zip 은 mp4 가
    파일로 존재하지 않으므로 cv2 대신 PyAV 를 쓴다 - cv2 는 경로만 받는다.
    PyAV 순차 디코딩이 cv2 의 CAP_PROP_POS_FRAMES 와 픽셀 단위로 일치함을
    확인했다(평균차 0.00).

    실패한 인덱스는 결과 dict 에서 빠진다.
    """
    want = sorted(set(int(i) for i in indices))
    if not want:
        return {}
    out = {}
    try:
        container = av.open(src)
    except (AV_ERROR, OSError):
        return {}
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        last = want[-1]
        remain = set(want)
        for n, frame in enumerate(container.decode(video=0)):
            if n in remain:
                rgb = frame.to_ndarray(format="rgb24")
                out[n] = _resize_to_pil(rgb[:, :, ::-1], max_long_side)
                remain.discard(n)
            if n >= last or not remain:
                break
    except AV_ERROR:
        pass
    finally:
        container.close()
    return out


def clip_path(view: str, uuid: str) -> Path:
    return CAMERA_DIR / view / f"{uuid}.{view}.mp4"


def _mirror_clip_source():
    """이 모듈이 __main__ 으로 돌 때, import 된 쪽 사본에도 같은 값을 심는다.

    이 파일을 스크립트로 실행하면 파이썬은 그것을 __main__ 으로 올린다.
    그런데 qwen_runner 는 `from edge_case_mining import sample_clip_frames`
    로 같은 파일을 한 번 더 - 이번엔 edge_case_mining 이라는 별개의 모듈
    객체로 - 올린다. 그래서 __main__ 쪽에서 CLIP_SOURCE 를 아무리 채워도
    qwen_runner 가 부르는 sample_clip_frames 는 CLIP_SOURCE=None 인 사본을
    보고 로컬 pav_sample 경로를 열려 든다.

    실측: --data nas 첫 실행에서 8샤드 전부가 NAS uuid 를 로컬 경로로 찾다
    FileNotFoundError 로 죽었다. uuid 목록은 __main__ 이 만들어 NAS 것이
    맞았기에, 목록은 NAS / 읽기는 로컬이라는 어긋난 상태였다.
    """
    if __name__ != "__main__":
        return
    import sys
    other = sys.modules.get("edge_case_mining")
    if other is not None and other is not sys.modules[__name__]:
        other.CLIP_SOURCE = CLIP_SOURCE
        other.CAMERA_DIR = CAMERA_DIR


def set_clip_source(spec):
    """--data 값으로 프레임 소스를 정한다. None/"local" 이면 기존 동작."""
    global CLIP_SOURCE, CAMERA_DIR
    if spec in (None, "local"):
        CLIP_SOURCE = None
        _mirror_clip_source()
        return None
    from clip_source import make_source
    CLIP_SOURCE = make_source(spec, root=ROOT)
    if getattr(CLIP_SOURCE, "kind", None) == "local":
        CAMERA_DIR = CLIP_SOURCE.camera_dir
    _mirror_clip_source()
    return CLIP_SOURCE


def clip_open(view: str, uuid: str):
    """프레임 소스를 연다. CLIP_SOURCE 가 설정돼 있으면 그쪽에 위임한다."""
    if CLIP_SOURCE is not None:
        return CLIP_SOURCE.open_video(view, uuid)
    return str(clip_path(view, uuid))


def clip_frame_count(uuid: str) -> int:
    """front_wide 뷰 기준 클립의 총 프레임 수 (3뷰 모두 동일함이 확인됨)."""
    try:
        container = av.open(clip_open(FRONT_VIEWS[0], uuid))
    except (AV_ERROR, OSError):
        return 0
    try:
        stream = container.streams.video[0]
        total = stream.frames
        if not total:  # 컨테이너가 프레임 수를 안 들고 있으면 duration 으로
            total = int(float(stream.duration * stream.time_base)
                        * float(stream.average_rate)) if stream.duration else 0
        return int(total)
    except (AV_ERROR, TypeError):
        return 0
    finally:
        container.close()


def sample_timestamp_indices(uuid: str, n_timestamps: int = 10):
    """클립 하나에서 균등 간격으로 n_timestamps 개의 frame_idx 를 뽑는다."""
    total = clip_frame_count(uuid)
    if total <= 0:
        return []
    idxs = np.linspace(0, total - 1, num=min(n_timestamps, total), dtype=int)
    return [int(i) for i in idxs]


def list_scene_uuids(limit: int | None = None):
    """front_wide 뷰 기준 클립 uuid 리스트 (정렬, 선택적 limit).

    세 뷰 모두 같은 uuid 집합을 공유함 (500 클립 x 3앵글).
    """
    if CLIP_SOURCE is not None:
        uuids = CLIP_SOURCE.uuids(FRONT_VIEWS[0])
    else:
        d = CAMERA_DIR / FRONT_VIEWS[0]
        uuids = sorted(p.name.split(".")[0] for p in d.glob("*.mp4"))
    if limit:
        uuids = uuids[:limit]
    return uuids


def list_frame_units(uuids, n_timestamps: int = 10):
    """[(uuid, frame_idx), ...] 형태의 전체 판정 단위 리스트를 만든다.

    클립당 n_timestamps 개 => 총 len(uuids) * n_timestamps 개 단위.
    """
    units = []
    for uuid in uuids:
        for idx in sample_timestamp_indices(uuid, n_timestamps):
            units.append((uuid, idx))
    return units


# temporal modeling: 각 판정 시점 t 에 대해 t-DELTA(직전 프레임)도 함께 본다.
# 두 시점의 위치 변화로 "정차/주행/이동방향" 을 추론 (30fps 기준 30 = 약 1초 전).
TEMPORAL_DELTA = 30


def sample_unit_frames(uuid: str, frame_idx: int, max_long_side: int = 896,
                       temporal_delta: int = TEMPORAL_DELTA,
                       views: list[str] | None = None):
    """판정 단위(uuid, frame_idx) 하나에 대해 각 뷰의 (직전, 현재) 프레임을 읽어
    {"prev": {view: PIL.Image}, "cur": {view: PIL.Image}} 로 반환.

    - cur  = frame_idx 시점
    - prev = frame_idx - temporal_delta 시점 (< 0 이면 프레임 0 으로 대체)
    - views = None 이면 전방 3뷰 전체 (단독 모드는 SINGLE_VIEW 를 넘긴다)
    시각화는 cur 만, 모델 입력은 prev+cur 을 순서대로 사용.
    """
    prev_idx = max(0, frame_idx - temporal_delta)
    prev, cur = {}, {}
    for view in (views or FRONT_VIEWS):
        frames = _read_frames_at(clip_open(view, uuid), [prev_idx, frame_idx],
                                 max_long_side=max_long_side)
        prev[view] = frames.get(prev_idx)
        cur[view] = frames.get(frame_idx)
    return {"prev": prev, "cur": cur, "prev_idx": prev_idx, "cur_idx": frame_idx}


# ---------------------------------------------------------------------------
# 클립 모드 - 판정 단위가 (uuid, frame_idx) 가 아니라 클립(uuid) 전체
#
# 위의 2프레임 방식은 "1초 전 -> 현재" 변화만 본다. cut-in, 갑작스러운 보행자
# 진입처럼 몇 초에 걸쳐 전개되는 것은 두 장으로 잡히지 않는다(scene_category_C
# 의 Abnormal Vehicle Behavior 가 정확히 여기 걸린다). 클립 모드는 20초를
# 1fps 로 훑어 그 시간축을 모델에게 직접 보여준다.
#
# 비용 때문에 기본값이 다르다 (실측 30fps, 1920x1080 원본):
#   - 뷰: front_wide 1개. 3뷰면 토큰이 3배가 되고, 논문(nuReasoning)도 마이닝
#     단계는 front 카메라 하나만 쓴다.
#   - 해상도: 640x360. 프레임당 294 토큰. 2프레임 모드의 896x504(576 토큰)보다
#     낮지만, 클립 모드가 답하려는 것은 "무엇이 시간에 걸쳐 변했나"라 프레임
#     한 장의 선명도는 덜 중요하다.
#   - 20장 x 294 = 5,878 토큰. KV 캐시 약 1.0GB 로 24GB 안에 여유 있게 들어간다.
# ---------------------------------------------------------------------------
CLIP_FPS = 2.0              # 초당 몇 장 뽑을지
CLIP_MAX_FRAMES = 40        # 클립당 최대 장수 (20초 x 1fps)
CLIP_MAX_LONG_SIDE = 896    # 클립 모드 프레임의 긴 변 (896 x 504, 640 x 320, 448 x 252)
SOURCE_FPS = 30.0           # 실측: 200 클립 중앙값 30.000 fps (33.30 ms)


def sample_clip_indices(uuid: str, fps: float = CLIP_FPS,
                        max_frames: int = CLIP_MAX_FRAMES,
                        source_fps: float = SOURCE_FPS):
    """클립 전체에서 fps 간격으로 frame_idx 를 뽑는다 (최대 max_frames 장).

    30fps 소스에서 1fps 면 30 프레임마다 한 장. 클립이 max_frames 초보다 길면
    앞에서부터 자르지 않고 균등 간격으로 다시 뽑아 20초 전체를 덮는다 -
    뒷부분을 버리면 클립 후반의 상황을 통째로 놓치기 때문.
    """
    total = clip_frame_count(uuid)
    if total <= 0:
        return []
    step = max(1, int(round(source_fps / fps)))
    idxs = list(range(0, total, step))
    if len(idxs) > max_frames:
        idxs = [int(i) for i in np.linspace(0, total - 1, num=max_frames)]
    return [int(i) for i in idxs]


def sample_clip_frames(uuid: str, fps: float = CLIP_FPS,
                       max_frames: int = CLIP_MAX_FRAMES,
                       max_long_side: int = CLIP_MAX_LONG_SIDE,
                       views: list[str] | None = None,
                       traj: str | None = None):
    """클립 하나를 1fps 로 훑어 시간순 프레임 목록을 만든다.

    반환: {"frames": [(frame_idx, view, PIL.Image), ...] 시간순,
           "indices": [frame_idx, ...], "views": [...]}
    뷰가 여러 개면 같은 시점의 뷰들이 연달아 오도록 정렬한다(t0의 3뷰,
    t1의 3뷰, ...) - 시간 순서가 뷰 순서보다 중요하기 때문.
    """
    views = views or SINGLE_VIEW
    idxs = sample_clip_indices(uuid, fps, max_frames)
    if not idxs:
        return {"frames": [], "indices": [], "views": views}

    per_view = {}
    for view in views:
        per_view[view] = _read_frames_at(clip_open(view, uuid), idxs,
                                         max_long_side=max_long_side)
    # 자차 미래 궤적을 프레임 위에 그린다. 텍스트로 주던 시공간 정보를
    # 픽셀로 옮기는 것 - 리사이즈 후에 그려야 선 두께가 입력 해상도에
    # 맞고, draw_trajectory 가 캘리브 해상도와의 배율을 알아서 맞춘다.
    if traj:
        from trajectory import draw_trajectory
        for view, imgs in per_view.items():
            for i, im in list(imgs.items()):
                bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
                if draw_trajectory(bgr, uuid, i, cam=view, mode=traj):
                    imgs[i] = Image.fromarray(
                        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    frames = []
    for i in idxs:
        for view in views:
            im = per_view[view].get(i)
            if im is not None:
                frames.append((i, view, im))
    return {"frames": frames, "indices": idxs, "views": views}


# ---------------------------------------------------------------------------
# 프롬프트 - 단일 호출로 Q1(Normal/Special) + Q2(해당 카테고리 전부) 동시 응답
#
# 2단계로 나누던 예전 구조(캡션 -> 캡션 텍스트만으로 매칭)는 캡션->카테고리
# 번역 손실이 있었다(동일 캡션이 다른 카테고리로 뒤집히는 문제). 여기서는
# 모델이 이미지를 직접 보고 카테고리명을 바로 답하므로 그 손실이 없다.
# 2단계를 분리했던 원래 이유("2단계에 이미지를 주면 캡션을 무시하고 픽셀에서
# 재판단한다")는 보호할 캡션이 없어졌으므로 더 이상 해당하지 않는다.
#
# Q1 이 Normal 이어도 Q2 를 건너뛰지 않는다 - 건너뛰면 Q1 오판이 복구 불가능한
# 누락이 되기 때문. 실제로 모델은 거의 항상 verdict="Normal" 을 주므로, 검수
# 대상은 verdict 가 아니라 "categories 가 비어있지 않은가"로 정한다.
#
# Q3(blocks_path)은 Q2 의 필터가 아니라 장면 전체에 대한 독립 질문이다. 이게
# 중요한 이유는 아래 실험 기록 참고 - 카테고리 나열에 조건을 걸면 나열 자체가
# 죽는다. Q3 는 나열된 것을 걸러내는 용도가 아니라, 사후 검수 우선순위를
# 매기는 별도 축으로만 쓴다.
#
# 프롬프트 문구 실험 (20260728, 공사장 클립 3프레임 + 평범한 6프레임으로 검증):
#   - "Mere presence is not enough ..." 같은 억제 규칙을 넣으면 탐지가 0 이 된다.
#     명백한 공사장에서도 categories 가 비어버림.
#   - verdict 와 categories 를 묶으면("카테고리를 하나라도 적으면 Special")
#     역시 탐지가 0 이 된다. 모델이 Special 선언에 보수적이라, 나열이 Special 을
#     강제하는 구조에서는 아예 나열을 포기한다.
#   - 카테고리를 먼저 묻고 verdict 를 나중에 물어도 탐지가 0. evidence 에는
#     "Construction site ..." 라고 쓰면서 categories 는 비우는 모순이 나타난다.
#   => 채택: Q1(verdict), Q2(categories), Q3(blocks_path)를 서로 독립으로 두고
#      "앞 답과 무관하게 각각 답하라"고 명시. 조건을 거는 순간 탐지가 죽는다.
# ---------------------------------------------------------------------------
# describe_obstacles() 가 내는 문장의 머리말. 이걸로 egomotion 줄과 구분한다.
OBSTACLE_FACT_PREFIX = "3D sensor labels detect nearby:"


def _split_sensor_facts(sensor_facts: str):
    """합쳐진 센서 문구를 (egomotion 줄들, obstacle 줄들) 로 나눈다.

    build_sensor_facts() 가 줄 단위로 이어 붙인 것을 되돌리는 것이라 단순
    접두사 매칭으로 충분하다.
    """
    ego, obs = [], []
    for line in (sensor_facts or "").splitlines():
        if not line.strip():
            continue
        (obs if line.startswith(OBSTACLE_FACT_PREFIX) else ego).append(line)
    return "\n".join(ego), "\n".join(obs)


def build_vlm_prompt(category_menu: str, sensor_facts: str = "",
                     ask_blocking: bool = True,
                     single_view: bool = False) -> str:
    """이미지 + 카테고리 메뉴 -> JSON 한 덩어리를 요구하는 프롬프트.

    sensor_facts: egomotion/obstacle 라벨에서 뽑은 사실 문구(옵션). 비어 있으면
                  해당 블록 자체가 빠진다.
    ask_blocking: False 면 Q3 를 아예 묻지 않는다 (JSON 스키마에서도 빠진다).
    single_view:  True 면 front-wide 만 쓰므로 이미지가 6장이 아니라 2장이다.
                  프롬프트가 장수/뷰 구성을 실제와 다르게 말하면 안 되므로
                  도입부 문구를 함께 바꾼다.
    """
    # egomotion 과 obstacle 은 신뢰도의 성격이 달라 블록을 나눈다.
    #
    # egomotion(속도/가속도)은 모델이 이미지로 추측하던 것을 대체하는 진짜
    # 사실이라 "이미지보다 이쪽을 믿으라"가 맞다.
    #
    # obstacle(주변 객체 목록)은 다르다. 무엇이 있는지만 알려줄 뿐 그것이
    # 특이상황인지는 여전히 이미지로 판단해야 한다. 그런데 같은 "ground truth,
    # trust these over your own guess" 문구를 쓰자 모델이 존재 자체를 근거로
    # 카테고리를 찍어버렸다 - 20260804 실측에서 obstacle 을 켜자 3D 라벨에
    # 대응 클래스가 있는 카테고리만 폭증했다(Animal 21->215 로 10배,
    # Jaywalking 183->553, cyclist 123->258). 대응 클래스가 없는
    # Road Construction 은 723->896 로 거의 그대로였다.
    # 그래서 obstacle 은 "참고용이고 판단은 이미지로 하라"고 명시한다.
    ego_facts, obstacle_facts = _split_sensor_facts(sensor_facts)

    fact_block = ""
    if ego_facts:
        fact_block += f"""
KNOWN FACTS about the ego-vehicle at the CURRENT moment (from vehicle sensors -
ground truth, trust these over your own guess from the images):
{ego_facts}
"""
    if obstacle_facts:
        fact_block += f"""
FOR REFERENCE, a 3D sensor lists objects it detected around the vehicle:
{obstacle_facts}
This only tells you that those objects exist somewhere in the scene. It does
NOT tell you whether any of them is unusual or affects driving - ordinary
traffic and pedestrians going about their business are detected too. Judge from
the IMAGES whether anything is actually noteworthy, and do not report a
category just because an object of that kind appears in this list.
"""

    if ask_blocking:
        n_q = "THREE"
        q3_block = """
Q3. Is anything actually blocking or intruding into the ego-vehicle's driving path right now? "Yes" or "No".
"""
        independence = """The three questions are INDEPENDENT - answer each one on its own:
- Answer Q2 in full even when the verdict is "Normal".
- Q3 does NOT filter Q2. Still list a category in Q2 even if it sits off to the
  side and the answer to Q3 is "No"."""
        schema_q3 = '\n "blocks_path": "Yes" or "No",'
    else:
        n_q = "TWO"
        q3_block = ""
        independence = ("The two questions are INDEPENDENT - answer Q2 in full "
                        "even when the verdict is \"Normal\".")
        schema_q3 = ""

    if single_view:
        intro = """You are an autonomous-driving scene analyst. You are shown TWO images
in order, both from the vehicle's front-wide camera: the FIRST is from about
1 second EARLIER, the SECOND is the CURRENT moment."""
    else:
        intro = """You are an autonomous-driving scene analyst. You are shown SIX images
in order: the FIRST three are from about 1 second EARLIER, the LAST three are
the CURRENT moment. Each group of three is synchronized camera views
(front-wide, cross-left, cross-right) of the same vehicle."""

    return f"""{intro}
{fact_block}
Compare the earlier frames to the current ones to judge motion, then answer {n_q} questions about the CURRENT moment.

Q1. Ordinary driving, or a SPECIAL edge-case situation that interfere with driving? "Normal" or "Special".
Q2. Which categories below are present in the CURRENT scene? List EVERY one that applies. Copy the category names EXACTLY as written:

{category_menu}
{q3_block}
{independence}

Respond with ONLY a JSON object, no other text:
{{"verdict": "Normal" or "Special",
 "categories": ["<exact category name>", ...],{schema_q3}
 "evidence": "<one short phrase describing what you actually see>"}}"""


# ---------------------------------------------------------------------------
# 프롬프트 (nuReasoning 방식) - 위 build_vlm_prompt 과 병행하는 별도 경로
#
# nuReasoning(arXiv 2605.31572) Fig. S1/S2 의 마이닝 프롬프트에서 추론 절차만
# 가져온 것. 위의 Q1/Q2/Q3 방식과 무엇이 다른가:
#
#   - 단계별 chain-of-thought 를 거친다. 각 단계가 JSON 필드로 남으므로
#     "왜 그렇게 봤는지"를 사후 검수할 수 있다. Q1/Q2/Q3 는 evidence 한 줄이
#     전부라 근거가 남지 않는다.
#   - 3단계에서 이상 요소마다 "자차 행동을 바꿨는가"를 함께 적게 한다.
#     검수자가 읽을 정보이지 탐지를 거르는 조건이 아니다.
#
# 논문과 다른 점: 난이도 점수(Final Assessment, 1~10)를 쓰지 않는다.
# 우리 과제는 "이 클립에 edge-case 요소가 있는가"를 가리는 것이지 롱테일
# 가치를 서열화하는 것이 아니다. 그래서 6단계와 채점 rubric, 그리고 점수를
# 낮추라는 취지의 behavior-centric 억제 문구를 모두 뺐다.
#
# 그 억제 문구를 빼는 것은 실측 근거도 있다: 위 Q1/Q2/Q3 주석의 20260728
# 실험에서 "Mere presence is not enough" 류의 규칙을 넣자 명백한 공사장에서도
# 탐지가 0 이 됐다. 판정은 "이상 요소를 나열했는가"로만 정한다 -
# scenario_types 가 비어 있지 않으면 Special.
#
# 자차 행동 변화는 egomotion.ego_behavior_change() 가 100Hz 라벨에서 계산해
# 사실로 넣어준다. 논문은 30초 영상을 보여주고 모델이 추측하게 했지만, 우리는
# 라벨이 있으므로 추측시킬 이유가 없다.
# ---------------------------------------------------------------------------


def clip_intro(n_frames: int, n_views: int, fps: float = CLIP_FPS,
               as_video: bool = False) -> str:
    """클립 모드 도입부. 실제로 넣은 것과 어긋나면 안 되므로 계산해서 쓴다.

    as_video=True 면 프레임을 낱장이 아니라 비디오 한 편으로 넘긴 경우다.
    이때는 프로세서가 프레임마다 타임스탬프를 직접 박아주므로 "몇 장을 몇 초
    간격으로 보여준다"는 설명이 오히려 실제와 어긋난다(시간축 병합 때문에
    모델이 보는 시점 수는 프레임 수의 절반이다). 그래서 장수를 말하지 않고
    영상이라는 사실과 길이만 알려준다.
    """
    span = n_frames / max(fps, 1e-6) / max(n_views, 1)
    if as_video:
        return (f"""You are an autonomous-driving scene analyst. You are shown a
{span:.0f}-second video from the vehicle's front-wide camera, sampled at about
{fps:g} frame per second. Each frame is tagged with its timestamp, and the LAST
frame is the most recent moment. Read it as a sequence: what changes over time
tells you how the scene and the ego-vehicle evolved.""")

    if n_views == 1:
        what = (f"{n_frames} images from the vehicle's front-wide camera, "
                f"in time order, about {1/fps:.0f} second apart, covering "
                f"roughly {span:.0f} seconds of driving")
    else:
        what = (f"{n_frames} images in time order from {n_views} synchronized "
                f"camera views, about {1/fps:.0f} second apart per view, "
                f"covering roughly {span:.0f} seconds of driving")
    return (f"""You are an autonomous-driving scene analyst. You are shown {what}.
The LAST image is the most recent moment. Read them as a sequence: what changes
from one image to the next tells you how the scene and the ego-vehicle evolved.""")


def build_nureasoning_prompt(category_menu: str, sensor_facts: str = "",
                             single_view: bool = False,
                             behavior_facts: str = "",
                             intro: str | None = None,
                             timeline: bool = False,
                             force_behavior_hint: bool = False,
                             header_style: str = "v1",
                             safety_tiers: bool = True,
                             rarity_tiers: bool = True,
                             difficulty: bool = False) -> str:
    """nuReasoning 6단계 CoT + 1~10 점수를 요구하는 프롬프트.

    --traj 로 궤적을 그려도 프롬프트에는 그 사실을 알리지 않는다. 알려주는
    편이 나을 것 같지만 실측은 반대였다(115클립, 설정/라벨 동일, 프롬프트만
    다름):

      설명 없음  Verdict F1 90.5  micro 0.777  safety MAE 0.296
      설명 있음  Verdict F1 83.0  micro 0.685  safety MAE 0.339

    6개 축이 모두 나빠졌다. 원인은 "궤적으로 어떤 도로 사용자와 실제로
    상호작용하는지 보라" 는 문구가 필터로 읽힌 것이다 - 경로 밖의 보행자를
    특이 요소에서 통째로 빼버려, 놓친 클립 6건 모두 unusual_elements 가
    비었다. 이 프롬프트는 이미 "지나쳤더라도 카테고리에 넣으라" 고 지시하고
    있어 정면으로 충돌한다. 짧게 줄여도 설명 없는 쪽이 계속 나았다.

    설명을 빼도 모델이 초록 선을 객체로 오신고하는 일은 없었다(실측 0건).

    build_vlm_prompt() 과 인자 구성을 최대한 맞춰 호출부에서 갈아끼우기 쉽게
    했다. 다른 점은 behavior_facts 하나 - egomotion 에서 뽑은 "지난 5초간
    자차가 어떻게 변했는가" 문장이며, 논문 2단계(Ego Behavior Summary)의
    근거로 쓰인다. 비어 있으면 그 블록이 통째로 빠지고, 모델은 이미지만으로
    행동 변화를 추정하게 된다(정확도는 떨어지지만 동작은 한다).

    Q3(blocks_path)에 해당하는 별도 질문은 없다 - 논문에서는 "행동에 영향을
    줬는가"는 3단계 안에서 요소별로 서술된다.
    """
    ego_facts, obstacle_facts = _split_sensor_facts(sensor_facts)

    fact_block = ""
    if ego_facts:
        fact_block += f"""
KNOWN FACTS about the ego-vehicle at the CURRENT moment (from vehicle sensors -
ground truth, trust these over your own guess from the images):
{ego_facts}
"""
    if behavior_facts:
        # 헤더 문구가 이 프롬프트에서 가장 큰 단일 기여자다.
        #
        # 실측(115클립, 20260824) 4-way 분해:
        #   A 아무것도 없음        79.1%
        #   D hint 만              82.6%   (+3.5%p)
        #   C 헤더+placebo+hint    87.0%   (+4.4%p 추가)
        #   B 헤더+센서수치+hint   87.0%   (+0.0%p) <- 수치는 기여하지 않는다
        # 즉 이득의 56%가 이 헤더에서 나오는데, 정작 그것이 소개하는 센서
        # 수치는 0%p 다. 헤더가 사실 공급이 아니라 "2단계와 3단계에서 자차
        # 영향을 따로 판단하라" 는 절차 지시로 작동했다는 뜻이다.
        #
        # 그래서 v2 는 없는 사실을 소개하는 대신 그 절차만 직접 말한다.
        # v1 을 남겨두는 이유: 위 수치는 실행 간 변동폭(±5%p) 과 같은
        # 크기라 v2 가 더 낫다는 보장이 없다. 되돌릴 수 있어야 한다.
        if header_style == "v2":
            fact_block += f"""
EGO-VEHICLE BEHAVIOUR - judge this separately from the scene. In step 2 state
what the ego-vehicle did (speed profile, lateral behaviour, right-of-way), and
in step 3 decide for EACH unusual element whether that element is what made the
ego-vehicle behave that way. An element can be unusual without influencing the
ego, and the ego can slow or stop for reasons that are not in this list at all.
{behavior_facts}
"""
        else:
            fact_block += f"""
HOW THE EGO-VEHICLE'S BEHAVIOUR CHANGED (measured from vehicle sensors, not a
guess - use this for step 2 and for judging ego influence in step 3):
{behavior_facts}
"""
    if obstacle_facts:
        fact_block += f"""
FOR REFERENCE, a 3D sensor lists objects it detected around the vehicle:
{obstacle_facts}
This only tells you that those objects exist somewhere in the scene. It does
NOT tell you whether any of them is unusual or affects driving - ordinary
traffic and pedestrians going about their business are detected too. Judge from
the IMAGES whether anything is actually noteworthy, and do not report a
scenario type just because an object of that kind appears in this list.
"""

    # intro 를 넘기면(클립 모드) 그걸 쓰고, 아니면 2프레임 모드 문구를 쓴다.
    if intro is None:
        if single_view:
            intro = """You are an autonomous-driving scene analyst. You are shown TWO images
in order, both from the vehicle's front-wide camera: the FIRST is from about
1 second EARLIER, the SECOND is the CURRENT moment."""
        else:
            intro = """You are an autonomous-driving scene analyst. You are shown SIX images
in order: the FIRST three are from about 1 second EARLIER, the LAST three are
the CURRENT moment. Each group of three is synchronized camera views
(front-wide, cross-left, cross-right) of the same vehicle."""

    tier_scale = tier_menu()
    tier_min, tier_max = min(TIER_VALUES), max(TIER_VALUES)
    safety_rubric = safety_rubric_text()
    rarity_rubric = rarity_rubric_text()
    contrasts = contrast_text()

    # egomotion 을 켠 실행에서는 감속/조향이 100Hz 라벨로 이미 계산돼 있다.
    # 예전에는 이 값을 "급제동은 3등급을 뒷받침한다" 는 식으로 3등급의
    # 객관적 근거라고 지정했는데, 실측에서 그게 틀렸다.
    #
    # 100클립 A/B(20260818): 정답 라벨 기준으로 급제동 클립 35건의 GT safety
    # 평균은 1.49 로 급제동이 없는 65건(1.45)과 사실상 같다. 이 데이터셋의
    # 급제동은 대개 신호/정체/교차로 때문이지 위험 반응이 아니다. 그런데도
    # 위 문장 때문에 모델이 감속만 보고 등급을 올려, 급제동 클립의 safety
    # 정확도가 66% -> 37% 로 무너졌다 (급제동 없는 클립은 72% -> 69%).
    #
    # 그래서 감속을 근거로 "지정" 하지 않고, 원인을 영상에서 확인하라고만
    # 한다. 실제로 위험이 보이는 클립은 여전히 높게 나와야 한다.
    #
    # --use-egomotion-d 는 이 지시문만 남기고 센서 문장(②)과 헤더(①)를 뺀다.
    # 그때는 위 문장의 "above" 가 가리킬 대상이 없으므로, 참조를 지운 판을
    # 쓴다. 내용은 같고 문장이 홀로 성립하도록만 고쳤다.
    if force_behavior_hint and not behavior_facts:
        behavior_hint = (
            "\n   Slowing or stopping is routine (signals, junctions, queues),"
            "\n   so treat the ego-vehicle's own braking or stopping as evidence"
            "\n   only when the video shows what caused it.")
    else:
        behavior_hint = (
            "\n   The measured ego behaviour above says what the vehicle did, not"
            "\n   why. Slowing or stopping is routine (signals, junctions, queues),"
            "\n   so treat it as evidence only when the video shows what caused it."
            if behavior_facts else "")

    # --no-safety-tier / --no-rarity-tier: 등급 단계를 하나씩 끈다.
    # edge-case 마이닝의 본체는 1~3단계(있는 요소를 빠짐없이 찾아 이름
    # 붙이는 것)이고, 등급은 그 위에 얹은 부가 점수다. 등급을 요구하면
    # 모델이 "몇 점을 줄까" 에 예산을 더 쓰게 되므로, 탐지 자체의
    # 재현율/정밀도가 등급 유무로 갈리는지 보려는 A/B 용이다.
    #
    # 둘을 따로 끄므로 단계 번호를 고정할 수 없다. rarity 만 켠 실행에서
    # "5." 로 시작하면 4번이 없는 목록이 되어, 모델이 빠진 단계를 찾으려
    # 하거나 없는 safety_tier 를 지어낸다. 그래서 켜진 것부터 4, 5 로
    # 다시 매긴다.
    steps45 = ""
    # 난이도 단계도 이 번호를 이어받으므로 조건 밖에 둔다 - 등급을 둘 다
    # 끄고 난이도만 켜면 난이도가 4번이 되어야 한다.
    step_no = 4
    if safety_tiers or rarity_tiers:
        n_on = int(safety_tiers) + int(rarity_tiers)
        # 머리말도 켜진 개수에 맞춘다 - 한 단계만 남았는데 "steps 4 and 5"
        # 라고 하면 없는 단계를 가리킨다.
        head_ref = ("steps 4 and 5" if n_on == 2 else f"step {step_no}")
        parts = [f"""   For {head_ref}, rate the SITUATION, never the object by itself. The same
   object is routine or serious depending on what it is doing and where it is:
{contrasts}
   So "there is an animal" or "there is a pedestrian" tells you nothing on its
   own - look at what it is doing relative to the ego-vehicle's path.
"""]
        if safety_tiers:
            parts.append(f"""{step_no}. Safety Criticality: how close this came to needing emergency action.
   A higher number means more dangerous. Pick the integer whose description
   fits best:
{safety_rubric}
   Judge by what the ego-vehicle actually had to DO, not by how much attention
   the scene deserves - almost every scene deserves attention, so "requires
   vigilance" is never a reason to pick 2 or 3.{behavior_hint}
""")
            step_no += 1
        if rarity_tiers:
            # "judged the same way" 는 앞의 safety 단계를 받는 말이라,
            # rarity 만 켜면 가리킬 대상이 없어진다.
            lead = ("how unusual this situation is, judged the same way."
                    if safety_tiers else "how unusual this situation is.")
            parts.append(f"""{step_no}. Rarity: {lead}
   A higher number means more unusual. Pick the integer whose description
   fits best:
{rarity_rubric}
""")
            # 예전에는 rarity 가 마지막이라 올릴 필요가 없었지만, 이제
            # 난이도 단계가 뒤에 붙어 이 번호를 이어받는다.
            step_no += 1
        steps45 = "".join(parts)
    # 난이도(--difficulty)는 등급 다음 단계로 붙는다. 등급과 성격이 다르다:
    # 등급은 "이 클립이 얼마나 위험/희귀한가"(edge-case 여부에 달림)이고,
    # 난이도는 "이 장면의 주행 조건이 얼마나 나쁜가"(평범한 클립도 비 오면
    # 높다)라서, 서로 독립으로 매기게 두어야 한다. 그래서 앞 단계를
    # 참조하는 문구를 넣지 않는다.
    if difficulty:
        steps45 += (f"{step_no}. Driving Difficulty: how hard the driving "
                    f"CONDITIONS are, judged\n   independently of the "
                    f"edge-case decision above - an ordinary clip in heavy "
                    f"rain\n   still scores high, and a rare event on a "
                    f"clear day does not.\n") + difficulty_block()
        step_no += 1
    tier_fields = ""
    if safety_tiers:
        tier_fields += f''' "safety_tier": <integer {tier_min}-{tier_max}>,
 "safety_reason": "<why that safety rating>",
'''
    if rarity_tiers:
        tier_fields += f''' "rarity_tier": <integer {tier_min}-{tier_max}>,
 "rarity_reason": "<why that rarity rating>",
'''
    if difficulty:
        # 요인을 먼저, 전체를 마지막에 - 프롬프트 본문의 순서와 같게 둔다.
        # JSON 필드 순서가 곧 생성 순서라, 전체를 앞에 두면 요인을 세우기
        # 전에 전체 점수를 찍게 되어 둘이 어긋난다.
        for key, name in DIFFICULTY_AXES[1:] + DIFFICULTY_AXES[:1]:
            tier_fields += (
                f''' "{key}": <integer {DIFFICULTY_MIN}-{DIFFICULTY_MAX}>,\n'''
                f''' "{key}_reason": "<one short sentence for {name}>",\n''')
    # 1단계를 시간순 서술로 할지(--timeline) 예전처럼 한 덩어리 요약으로 할지.
    #
    # 왜 분기가 필요한가: 20초 클립에 서로 다른 시점의 사건이 둘 이상 있을 때,
    # 한 덩어리로 요약하면 뒤 사건이 탈락한다. 실측(20260818, 100클립): GT 가
    # 2개인 클립 11건의 재현율이 64% 로 1개짜리(72%)보다 낮았고, 어떤 클립은
    # 1단계에 "crosses a railroad crossing" 이라 써놓고 3단계에서 Railroad 를
    # 빠뜨렸다. 다만 구간마다 뭔가 찾아야 한다는 압력이 없는 사건을 만들어낼
    # 위험도 있어(Jaywalking 과탐이 이미 최대 오류원) 옛 방식을 남겨 A/B 한다.
    if timeline:
        step1 = """1. Scene Description: walk through the clip in time order. A 20-second clip
   often passes through more than one distinct situation, so describe them
   one at a time with the rough timestamp of each, rather than blending the
   whole clip into a single summary - for example "0-6 s: crossing a railroad;
   8-20 s: a pedestrian steps into the lane between parked cars". Cover the
   road environment and the visible agents or objects in each. If a stretch
   has nothing worth noting, say so instead of inventing something.
   Do not describe lighting or weather."""
        step3_head = """3. Unusual Elements and Ego Influence: go back over every situation you listed
   in step 1 and check each one against the scenario types. An element that
   appears in only part of the clip counts exactly as much as one that lasts
   throughout, so a type you named in step 1 must not disappear here. List
   every unusual element, and for each one state whether it changed the
   ego-vehicle's behaviour. Also name the matching scenario types from the
   list above, copying the names EXACTLY."""
        observation_field = ("<the clip in time order, each situation with "
                             "its rough timestamp>")
    else:
        step1 = """1. Scene Description: the road environment and the visible agents or objects.
   Do not describe lighting or weather."""
        step3_head = """3. Unusual Elements and Ego Influence: list every unusual element, and for each
   one state whether it changed the ego-vehicle's behaviour. Also name the
   matching scenario types from the list above, copying the names EXACTLY."""
        observation_field = "<scene description, one or two sentences>"

    return f"""{intro}
{fact_block}
Your job is to decide whether this clip contains any edge-case element - a rare
or unusual road situation - and to name which of the types below it matches.
You are NOT rating how difficult or how valuable the clip is.

SCENARIO TYPES:
{category_menu}

List EVERY type you can actually see. An element still counts even if the
ego-vehicle drove past it without reacting - whether it changed the driving is
recorded separately in step 3, and never a reason to leave a type out. If you
see no unusual element at all, return an empty list.

Work through these steps in order:
{step1}
2. Ego Behaviour Summary: the ego-vehicle's speed profile, lateral behaviour,
   and right-of-way behaviour. State explicitly whether its behaviour is
   unchanged/typical.
{step3_head}
{steps45}
Respond with ONLY a JSON object, no other text:
{{"observation": "{observation_field}",
 "ego_behavior": "<how the ego-vehicle is behaving and whether it changed>",
 "unusual_elements": "<each unusual element and whether it influenced the ego>",
{tier_fields} "scenario_types": ["<exact scenario type name>", ...]}}"""


def _flatten_field(v) -> str:
    """서술 필드를 사람이 읽을 한 줄로 편다.

    "unusual_elements" 는 문장을 요구했는데도 모델이 자주 구조체 배열로 답한다
    (실측: [{"element": "...", "influenced_ego": true, "scenario_type": "..."}]).
    str() 로 그냥 감싸면 파이썬 repr 이 CSV 에 들어가 읽기 어려우므로,
    dict/list 는 값만 뽑아 이어 붙인다.
    """
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, dict):
        parts = []
        for k, x in v.items():
            if isinstance(x, bool):
                parts.append(f"{k}={'yes' if x else 'no'}")
            elif x not in (None, ""):
                parts.append(str(x).strip())
        return "; ".join(parts)
    if isinstance(v, (list, tuple)):
        return " | ".join(p for p in (_flatten_field(x) for x in v) if p)
    return str(v).strip()


def _coerce_difficulty(v) -> int | None:
    """난이도 값을 0~4 정수로 만든다. 못 읽으면 None.

    _coerce_tier 와 따로 두는 이유는 눈금이 다르기 때문이다 - 등급은 1~4,
    난이도는 0~4 다. 0 을 흡수하지 못하면 "조건이 전혀 나쁘지 않다" 는
    가장 흔한 답이 통째로 미기록으로 떨어진다.

    난이도 필드는 constrained decoding 대상이 아니라(제약기가 TIER_VALUES
    에 묶여 있다) 여기 오는 값이 "2 (moderate)" 나 "2/4" 처럼 지저분할 수
    있다. 앞머리 정수만 떼어 쓰고, 범위 밖이면 버린다 - 클램프해서 0 이나
    4 로 밀어 넣으면 분포가 양 끝에 가짜로 쌓인다.
    """
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        n = int(round(v))
        return n if DIFFICULTY_MIN <= n <= DIFFICULTY_MAX else None
    if isinstance(v, str):
        m = re.match(r"\s*(\d+)", v.strip())
        if m:
            n = int(m.group(1))
            return n if DIFFICULTY_MIN <= n <= DIFFICULTY_MAX else None
    return None


def _coerce_tier(v) -> int | None:
    """모델이 낸 등급 값을 1/2/3 정수로 만든다. 못 읽으면 None.

    constrained decoding 을 켜면 여기 오는 값은 이미 1/2/3 이다. 다만
    제약을 끈 실행이나 예전 형식("Low"/"Moderate"/"High" 문자열)도 있어서
    문자열 폴백을 남긴다.
    """
    if isinstance(v, bool):          # True/False 가 int 로 새는 것 방지
        return None
    if isinstance(v, (int, float)):
        n = int(round(v))
        return n if n in TIER_LABELS else None
    if isinstance(v, str):
        t = v.strip()
        if not t:
            return None
        # "2", "2 (Moderate)", "Moderate" 모두 흡수
        m = re.match(r"\s*([1-3])\b", t)
        if m:
            return int(m.group(1))
        label = _extract_tier(t)
        for k, name in TIER_LABELS.items():
            if name == label:
                return k
    return None


def _extract_tier(text: str) -> str:
    """자유 서술에서 Low/Moderate/High 등급만 뽑는다 (구 형식 폴백).

    프롬프트가 정수를 요구하도록 바뀐 뒤로는 주 경로가 아니다. 제약을 끄고
    돌린 실행이나 예전 CSV 를 다시 읽을 때를 위해 남겨둔다.

    프롬프트가 "Low, Moderate, or High 로 시작하고 이유를 덧붙여라"라고
    시키므로 먼저 맨 앞 단어로 판정한다. 그런데 "Low to moderate",
    "Highly critical" 처럼 경계를 흐리는 답도 실측(20260811, 50클립)에서
    나와서, 시작 단어가 애매하면(두 등급이 함께 언급되는 등) 텍스트 전체를
    보고 더 강한 쪽으로 반올림한다 - 필터링(--not-save-low)이 걸러야 할
    것을 놓치는 게, 있는 것을 더 얹는 것보다 나쁘기 때문이다.
    "common, low rarity" 처럼 다른 낱말 뒤에 우연히 "low" 가 붙은 문장을
    "moderate" 로 잘못 올리지 않도록, 상향 판정은 실제로 완화 어구(to/or)가
    있을 때만 적용한다. 등급이 전혀 안 보이면 "Unknown" - 필터는 이를
    Moderate/High 와 동일하게(저장) 취급한다.
    """
    t = (text or "").strip().lower()
    if not t:
        return "Unknown"
    head = t[:40]

    # 시작 단어로 우선 판정 - 프롬프트가 요구한 형식이라 대개 여기서 끝난다.
    if head.startswith("high"):
        return "High"
    if head.startswith("moderate") or head.startswith("medium"):
        return "Moderate"
    if head.startswith("low"):
        # "low to moderate"/"low or high" 처럼 다른 등급 낱말이 바로 뒤에
        # 함께 나오면 더 강한 쪽으로 - 단어 자체("moderate"/"high")로 검사해야
        # "low-risk"의 하이픈 같은 걸 오탐하지 않는다.
        rest = head[3:20]
        if "moderate" in rest or "medium" in rest or "high" in rest:
            return "Moderate" if "high" not in rest else "High"
        return "Low"

    # 시작 단어가 세 등급 중 하나가 아니면("Highly critical" 등) 전체에서
    # 등급 낱말을 찾되, high > moderate > low 순으로 강한 것을 우선한다.
    if "high" in head:
        return "High"
    if "moderate" in head or "medium" in head:
        return "Moderate"
    if "low" in head:
        return "Low"
    return "Unknown"


def parse_nureasoning_output(text: str, labels: list) -> dict:
    """nuReasoning 출력 -> 기존 결과 dict 와 호환되는 형태.

    다운스트림(시각화/집계/CSV)이 categories/verdict/evidence 를 기대하므로
    새 필드를 거기에 매핑해 둔다:
      scenario_types -> categories   (기존 검증 로직 그대로 재사용)
      verdict        -> categories 가 비어 있지 않으면 "Special"
      observation 등 -> evidence 에 요약, 원본 단계별 답도 모두 보존

    safety_tier/rarity_tier 는 1/2/3 정수, safety_label/rarity_label 은 그것을
    사람이 읽는 "Low"/"Moderate"/"High" 로 옮긴 것. 등급을 못 읽으면 정수는
    None, 라벨은 "Unknown" 이고, 필터는 이를 "거르지 않음"으로 취급한다
    (놓치는 것보다 더 보는 쪽이 안전).

    blocks_path 는 항상 None - 이 방식에는 Q3 가 없다. 파싱에 실패하면
    카테고리 없이 Normal 이 되어, 없는 special 을 만들어내는 대신 놓치는
    쪽으로 떨어진다.
    """
    valid = {l["category"] for l in labels if not l["is_normal"]}
    out = {"verdict": "Normal", "categories": [], "blocks_path": None,
           "evidence": "", "parse_ok": False,
           "observation": "", "ego_behavior": "", "unusual_elements": "",
           "safety_assessment": "", "rarity_assessment": "",
           "safety_tier": None, "rarity_tier": None,
           "safety_label": "Unknown", "rarity_label": "Unknown",
           "tier_score": None}
    # 난이도 축은 --difficulty 를 껐으면 모델이 내지 않는다. 그래도 키는
    # 항상 만들어 둔다 - 없으면 CSV 열 개수가 실행마다 달라져 병합이 깨진다.
    for key, _ in DIFFICULTY_AXES:
        out[key] = None
        out[f"{key}_reason"] = ""

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return out
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return out

    out["parse_ok"] = True
    for k in ("observation", "ego_behavior", "unusual_elements"):
        out[k] = _flatten_field(obj.get(k, ""))

    # 등급: 새 스키마는 safety_tier(정수) + safety_reason(서술)로 나뉘어 있다.
    # 구 스키마(safety_assessment 한 필드에 등급+이유)도 계속 읽는다.
    for kind in ("safety", "rarity"):
        tier = _coerce_tier(obj.get(f"{kind}_tier"))
        reason = _flatten_field(obj.get(f"{kind}_reason", ""))
        legacy = _flatten_field(obj.get(f"{kind}_assessment", ""))
        if tier is None and legacy:
            tier = _coerce_tier(legacy)
        out[f"{kind}_tier"] = tier
        out[f"{kind}_label"] = tier_label(tier)
        # 사람이 읽는 서술은 reason 우선, 없으면 구 형식 문장을 쓴다.
        out[f"{kind}_assessment"] = reason or legacy
    out["tier_score"] = tier_score(out["safety_tier"], out["rarity_tier"])

    # 난이도 5축. 값과 근거 문장을 따로 받는다.
    for key, _ in DIFFICULTY_AXES:
        out[key] = _coerce_difficulty(obj.get(key))
        out[f"{key}_reason"] = _flatten_field(obj.get(f"{key}_reason", ""))

    raw_cats = obj.get("scenario_types", [])
    if isinstance(raw_cats, str):
        raw_cats = [raw_cats]
    raw_cats = list(raw_cats)

    # 모델이 scenario_types 를 비워두고 unusual_elements 안에 scenario_type 을
    # 넣어버리는 경우가 있다(실측 20260811). 그대로 두면 카테고리가 통째로
    # 사라지므로 거기서도 걷어온다 - 어차피 아래에서 유효성 검사를 거친다.
    nested = obj.get("unusual_elements", [])
    if isinstance(nested, (list, tuple)):
        for el in nested:
            if not isinstance(el, dict):
                continue
            for key in ("scenario_type", "scenario_types", "category"):
                v = el.get(key)
                if isinstance(v, str):
                    raw_cats.append(v)
                elif isinstance(v, (list, tuple)):
                    raw_cats.extend(v)

    seen = set()
    for c in raw_cats:
        cat = _closest_valid(str(c).strip(), valid)
        if cat and cat not in seen:
            seen.add(cat)
            out["categories"].append(cat)

    # 판정은 오직 "edge-case 요소를 하나라도 나열했는가". 별도 질문을 두지
    # 않는 이유는 20260728 실험 - verdict 를 따로 물어 categories 와 묶으면
    # 모델이 Special 선언에 보수적이라 나열 자체를 포기했다.
    out["verdict"] = "Special" if out["categories"] else "Normal"

    # 카드/CSV 가 한 줄 요약을 기대하므로 관찰 + 이상요소를 합쳐 채운다
    out["evidence"] = " ".join(
        p for p in (out["observation"], out["unusual_elements"]) if p).strip()
    return out


# ---------------------------------------------------------------------------
# 출력 파싱
# ---------------------------------------------------------------------------
def _closest_valid(name: str, valid_names: set) -> str | None:
    if name in valid_names:
        return name
    low = name.lower().strip()
    for v in valid_names:
        if v.lower() == low:
            return v
    return None


def parse_vlm_output(text: str, labels: list,
                     ask_blocking: bool = True) -> dict:
    """모델 출력 -> {"verdict","categories","blocks_path","evidence","parse_ok"}.

    categories 는 scene_category_B.json 에 실제로 있는 special 카테고리명만
    남긴다(대소문자 차이는 흡수). 모델이 만들어낸 이름은 버린다.
    JSON 파싱에 실패하면 parse_ok=False 로 표시하고 verdict 는 Normal 로 둔다
    (없는 special 을 만들어내는 것보다 놓치는 쪽이 사후 검수에 안전).

    blocks_path 는 ask_blocking=False 면 항상 None 이다. "묻지 않았다"와
    "No 라고 답했다"는 다른 상태이므로 False 로 뭉뚱그리지 않는다 - 시각화
    폴더 구조와 집계가 이 구분에 의존한다.
    """
    valid = {l["category"] for l in labels if not l["is_normal"]}
    out = {"verdict": "Normal", "categories": [],
           "blocks_path": False if ask_blocking else None,
           "evidence": "", "parse_ok": False}

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return out
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return out

    out["parse_ok"] = True
    v = str(obj.get("verdict", "")).strip().lower()
    out["verdict"] = "Special" if v.startswith("s") else "Normal"
    out["evidence"] = str(obj.get("evidence", "")).strip()

    # Q3: "Yes"/"No" 를 기대하지만 true/false 로 답하는 경우도 흡수
    if ask_blocking:
        b = obj.get("blocks_path", False)
        if isinstance(b, bool):
            out["blocks_path"] = b
        else:
            out["blocks_path"] = str(b).strip().lower() in ("yes", "y", "true", "1")

    raw_cats = obj.get("categories", [])
    if isinstance(raw_cats, str):
        raw_cats = [raw_cats]
    seen = set()
    for c in raw_cats:
        cat = _closest_valid(str(c).strip(), valid)
        if cat and cat not in seen:
            seen.add(cat)
            out["categories"].append(cat)
    return out


def unit_name(uuid: str, frame_idx: int) -> str:
    """판정 단위의 고유 폴더명: <uuid>_f<frame_idx:04d>"""
    return f"{uuid}_f{frame_idx:04d}"


def blocking_dir(result: dict) -> str | None:
    """Q3 답에 따른 최상위 폴더명. Q3 를 묻지 않았으면 None."""
    b = result.get("blocks_path")
    if b is None:
        return None
    return "blocking_yes" if b else "blocking_no"


def viz_targets(result: dict) -> list[str]:
    """이 판정 단위의 시각화를 저장할 상대 경로들.

    Q3 를 물었으면 blocking_{yes,no}/<category>/, 묻지 않았으면 <category>/ 로
    바로 담는다. 멀티라벨이면 해당하는 모든 카테고리 폴더에 같은 결과를 중복
    저장한다(검수할 때 카테고리 단위로 훑을 수 있어야 하므로).
    카테고리가 하나도 없으면 저장하지 않는다 - verdict 와 무관하게, 볼 것이
    없는 장면이기 때문.
    """
    if not result["categories"]:
        return []
    top = blocking_dir(result)
    prefix = f"{top}/" if top else ""
    return [f"{prefix}{category_slug(c)}" for c in result["categories"]]


RUN_CONFIG_NAME = "run_config.json"


def _file_sha(path, n=10):
    """파일 내용의 짧은 해시. 없거나 못 읽으면 None."""
    if not path:
        return None
    try:
        import hashlib
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:n]
    except OSError:
        return None


def save_run_config(args, run_dir, n_views=1):
    """이번 실행의 설정을 <run_dir>/run_config.json 에 남긴다.

    argparse 네임스페이스를 통째로 저장하되, 나중에 사람이 먼저 보게 될
    핵심 값(fps/frames/해상도/센서 플래그)은 따로 "key" 에 모아 둔다 -
    evaluate_labels.py 가 그것만 골라 로그 머리에 찍는다.

    왜 결과 폴더에 두는가: 실행 폴더를 나중에 열었을 때 어떤 설정으로 낸
    숫자인지 알 방법이 run.log 를 뒤지는 것뿐이면, 로그가 지워지거나
    --eval-only 로 재채점할 때 근거가 사라진다.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "key": {
            # 사람이 붙인 실행 설명. 설정만으로는 구분이 안 되는 실험
            # (예: 라벨 파일 내용을 고쳤을 때)을 나중에 알아보게 해준다.
            "memo": args.memo,
            "model": args.model,
            "scene_json": args.scene_json,
            # 라벨 "내용" 의 지문. 경로는 그대로인데 내용만 고치는 일이
            # 잦아(실측: 20260818 의 130207 과 134742 는 설정이 완전히
            # 같은데 그 사이 Too Close Person 을 Jaywalking 에 병합해
            # 점수가 달라졌다) 경로만으로는 두 실행을 구분할 수 없다.
            "gt_labels_sha": _file_sha(args.gt_labels),
            # 시각화 패널에 GT 를 함께 그릴 때 쓴 정답 라벨. 채점에 쓰는
            # --labels 와 다른 파일일 수 있어(test_label.json vs _D) 따로 남긴다.
            "gt_labels": args.gt_labels,
            "prompt_style": args.prompt_style,
            "clip_fps": args.clip_fps,
            "clip_max_frames": args.clip_max_frames,
            "clip_long_side": args.clip_long_side,
            "n_views": n_views,
            "single_view": bool(args.single_view),
            "video_input": bool(args.clip_video_input),
            "timeline": bool(args.timeline),
            "ego_track": bool(args.ego_track),
            "traj": args.traj,
            # 궤적 스타일은 trajectory.py 상수가 단일 진실 공급원이다.
            # 그 값을 여기 박아두어야 나중에 상수를 바꿔도 과거 실행이
            # 어떤 설정이었는지 되짚을 수 있다.
            **({"traj_horizon": _TRAJ_HORIZON_S,
                "traj_alpha": _TRAJ_ALPHA} if args.traj else {}),
            "use_egomotion": bool(args.use_egomotion),
            "ego_ablation": args.ego_ablation,
            "header_style": args.header_style,
            "margin": bool(args.margin),
            "safety_tiers": bool(args.safety_tiers),
            "rarity_tiers": bool(args.rarity_tiers),
            "use_3dbbox": bool(args.use_obstacle),
            "constrain_tiers": bool(args.constrain_tiers),
            "difficulty": bool(args.difficulty),
            "num_shards": args.num_shards,
        },
        # 위에 없는 옵션까지 전부. 값이 Path 등이면 문자열로 눕힌다.
        "argv": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                     else str(v))
                 for k, v in vars(args).items()},
    }
    path = run_dir / RUN_CONFIG_NAME
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"[info] run config -> {path}")
    return path


def _load_gt_labels(path):
    """test_label.json -> {uuid: {categories, safety, rarity}}. 없으면 None.

    라벨 파일은 safety_criticality(오타로 safty_criticality 가 섞이기도 함)와
    rarity 를 쓰므로, 시각화가 쓰는 이름으로 맞춰 담는다.
    """
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    clips = data.get("clips", data)
    out = {}
    for uuid, v in clips.items():
        out[uuid] = {
            "categories": v.get("categories") or [],
            "safety": v.get("safety_criticality", v.get("safty_criticality")),
            "rarity": v.get("rarity"),
        }
    return out


def clip_uuid(mp4_path: str) -> str:
    return Path(mp4_path).name.split(".")[0]


if __name__ == "__main__":
    import datetime

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-clips", type=int, default=None,
                    help="처리할 클립(uuid) 수 제한 (스모크 테스트용)")
    ap.add_argument("--timestamps-per-clip", type=int, default=10,
                    help="클립당 샘플링할 timestamp(프레임) 수")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct",
                    choices=["Qwen/Qwen2.5-VL-7B-Instruct",
                             "Qwen/Qwen3-VL-8B-Instruct",
                             "Qwen/Qwen3-VL-32B-Instruct",
                             "Qwen/Qwen3.8-27B"],
                    help="사용할 VLM. Qwen3-VL 은 Qwen2.5-VL 과 동일한 "
                         "Qwen2_5_VLForConditionalGeneration 아키텍처가 아니므로 "
                         "qwen_runner.py 가 model_id 를 보고 알맞은 모델/프로세서 "
                         "클래스를 자동으로 고른다. Qwen3.8-27B 는 "
                         "architectures=Qwen3_5ForConditionalGeneration 로 또 "
                         "다르고, bf16 55.6GB 라 GPU 1장에 안 들어가 "
                         "device_map='auto' 로 여러 GPU 에 걸쳐 로드한다 - "
                         "8샤드 병렬(run_video_C.sh)이 아니라 run_video_27b.sh "
                         "(qwen38 conda 환경)로 돌려야 한다. FP8 판(-FP8)은 "
                         "일부러 목록에서 뺐다 - 멀티 GPU 로 로드하면 출력이 "
                         "깨진다(실측, 텍스트 전용 생성도 실패). GPU 1장에 "
                         "FP8(30.9GB)을 통째로 올릴 방법이 없는 한 못 쓴다.")
    # 후보로 검토했던 타 계열: InternVL3-38B/78B(OCR/세밀한 장면 이해가 강해
    # 표지판·차선 마킹에 유리할 수 있음), LLaVA-OneVision-72B(비디오 벤치마크
    # 강점). 다만 지금 프롬프트는 Qwen 특성에 맞춰 튜닝한 것이라 계열을 바꾸면
    # 재검증이 필요하다 - build_vlm_prompt() 위 주석 참고.
    ap.add_argument("--out", default=None,
                    help="결과 CSV 경로. 생략 시 results/<timestamp>/results.csv 자동 생성")
    ap.add_argument("--viz-dir", default=None,
                    help="시각화 이미지 저장 폴더. 생략 시 --out 과 같은 실행 폴더 사용")
    ap.add_argument("--dry-run", action="store_true",
                    help="모델 없이 프레임 샘플링만 검증")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="전체 판정 단위를 몇 등분할지 (GPU 병렬용)")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="이 프로세스가 처리할 shard 인덱스 (0-based)")
    ap.add_argument("--use-egomotion", action="store_true",
                    help="egomotion 라벨(속도/가속도/곡률)을 사실로 프롬프트에 넣는다 (기본 off).")
    # --use-egomotion 이 켜면 프롬프트에 ① 헤더 ② 센서 수치 ③ Safety 루브릭
    # 끝의 hint 가 한꺼번에 들어간다. 아래 두 플래그는 그 셋을 갈라 어느
    # 것이 이득을 냈는지 재기 위한 대조군이다. 둘 다 --use-egomotion 은 꺼진
    # 상태로 동작한다 (켜면 ②가 되살아나 대조가 깨진다).
    ap.add_argument("--use-egomotion-c", dest="ego_c", action="store_true",
                    help="[대조군 C] ①헤더와 ③hint 는 그대로 두고 ②센서 수치만 "
                         "내용 없는 문장으로 바꾼다. B(--use-egomotion)에 "
                         "근접하면 수치가 아니라 프롬프트 구조가 일한 것이다. "
                         "--use-egomotion 과 함께 쓸 수 없다.")
    ap.add_argument("--margin", action="store_true",
                    help="생성 토큰마다 1위-2위 확률차를 재서 CSV 에 남긴다 "
                         "(margin_min/margin_mean/n_close_tokens). 이 값이 "
                         "작은 클립은 의미 없는 프롬프트 섭동(예: 마침표 하나)"
                         "에도 예측이 뒤집힌다 - 실측 10/115 클립이 그랬다. "
                         "A/B 비교 전에 그 차이를 믿어도 되는지 판단할 때 쓴다.")
    ap.add_argument("--header-style", choices=["v1", "v2"], default="v1",
                    help="자차 행동 블록의 헤더 문구. v1=기존('measured from "
                         "vehicle sensors'), v2=센서 언급 없이 2/3단계 절차만 "
                         "지시. 헤더가 이득의 56%%를 내는데 그것이 소개하는 "
                         "센서 수치는 0%%p 라, v2 는 그 절차를 직접 말한다. "
                         "--use-egomotion 또는 -c 와 함께 쓴다 (기본 v1).")
    ap.add_argument("--use-egomotion-d", dest="ego_d", action="store_true",
                    help="[대조군 D] ①②를 모두 빼고 ③hint 만 넣는다 (참조어 "
                         "'above' 를 지운 판). A(아무것도 없음)보다 오르면 "
                         "지시문 단독 효과다. --use-egomotion 과 함께 쓸 수 없다.")
    ap.add_argument("--use-3dbbox", dest="use_obstacle", action="store_true",
                    help="obstacle.offline 3D bbox 라벨(위치/크기/방향)의 주변 "
                         "객체 요약을 프롬프트에 넣는다 (기본 off). 2D 이미지 "
                         "bbox 가 아니라 3D 라벨이다 - 대응 클래스가 있는 "
                         "카테고리(Animal/Jaywalking/cyclist)가 과탐하는 경향이 "
                         "실측됐으니(20260804) 켤 때 결과를 함께 확인할 것.")
    ap.add_argument("--single-view", action="store_true",
                    help="front-wide 카메라만 사용한다 (기본은 전방 3뷰). 이미지가 "
                         "6장에서 2장으로 줄어 추론이 빨라지지만, 측면에서만 보이는 "
                         "상황(Jaywalking, Side Street 등)은 놓칠 수 있다. "
                         "프롬프트 문구와 시각화 레이아웃도 함께 바뀐다.")
    ap.add_argument("--no-blocking", dest="ask_blocking", action="store_false",
                    help="Q3(주행 경로를 막는가)를 묻지 않는다. 이 경우 시각화는 "
                         "blocking_{yes,no} 없이 <category>/ 로 바로 저장되고 "
                         "집계에서도 blocking 구분이 빠진다 (기본은 물음).")
    ap.add_argument("--check-path", action="store_true",
                    help="NVIDIA 3D bbox 로 자차 전방 통로 침범 여부를 기하학적으로 "
                         "계산해 모델의 Q3(blocks_path) 와 대조한다 (기본 off). "
                         "프롬프트에는 넣지 않고 결과만 CSV/JSON 에 남긴다 - "
                         "불일치 건이 검수 우선순위가 된다. --no-blocking 이면 "
                         "대조할 모델 답이 없으므로 기하 계산 결과만 기록한다.")
    ap.add_argument("--example-source", choices=["synonyms", "prompt_templates"],
                    default="synonyms",
                    help="카테고리 예시 소스. synonyms(기본)는 짧은 키워드, "
                         "prompt_templates 는 상황 서술 문장.")
    ap.add_argument("--num-examples", type=int, default=2,
                    help="카테고리당 프롬프트에 넣을 예시 개수 (기본 2). "
                         "많이 넣을수록 프롬프트가 길어지고 판별이 경직될 수 있음.")
    # --scene-json 과 --prompt-style 은 서로 직교한다. C 카테고리를 기존
    # Q1/Q2/Q3 로 돌려보는 것도, B 카테고리를 nuReasoning 으로 돌려보는 것도
    # 각각 유효한 비교라 하나로 묶지 않는다.
    ap.add_argument("--scene-json", default=str(SCENE_JSON),
                    help=f"카테고리 정의 JSON (기본 {SCENE_JSON.name}, "
                         "config.py 에서 정함).")
    ap.add_argument("--prompt-style", choices=["qa", "nureasoning"], default="qa",
                    help="qa(기본): 기존 Q1(verdict)/Q2(categories)/Q3(blocking). "
                         "nureasoning: 단계별 CoT 로 근거를 남기며 edge-case "
                         "요소를 나열한다(난이도 점수는 매기지 않는다). Q3 를 "
                         "쓰지 않으므로 --no-blocking 과 같은 폴더 구조가 되고, "
                         "--use-egomotion 을 켜면 자차 행동 변화 문장이 함께 "
                         "들어간다.")
    ap.add_argument("--only-uuids", default=None,
                    help="이 파일에 적힌 uuid(한 줄에 하나)만 처리한다. "
                         "정답 라벨이 있는 클립만 골라 검증할 때 쓴다.")
    ap.add_argument("--clip-mode", action="store_true",
                    help="판정 단위를 (uuid, frame_idx) 가 아니라 클립 전체로 "
                         "바꾼다. 20초를 --clip-fps 로 훑어 시간순 이미지를 "
                         "한 번에 넣는다. nureasoning 프롬프트 전용이며 "
                         "시각화는 하지 않는다.")
    ap.add_argument("--clip-fps", type=float, default=CLIP_FPS,
                    help=f"클립 모드 샘플링 fps (기본 {CLIP_FPS}). "
                         f"소스는 30fps.")
    ap.add_argument("--clip-max-frames", type=int, default=CLIP_MAX_FRAMES,
                    help=f"클립당 최대 프레임 수 (기본 {CLIP_MAX_FRAMES})")
    ap.add_argument("--clip-long-side", type=int, default=CLIP_MAX_LONG_SIDE,
                    help=f"클립 모드 프레임 긴 변 픽셀 (기본 "
                         f"{CLIP_MAX_LONG_SIDE}; 2프레임 모드는 896)")
    ap.add_argument("--clip-viz", action="store_true",
                    help="클립 모드 시각화: 원본 mp4 아래에 1~5단계 추론을 붙인 "
                         "영상과 result.json 을 클립마다 저장한다.")
    ap.add_argument("--clip-viz-all", action="store_true",
                    help="edge-case 요소가 없는 클립까지 전부 영상으로 만든다. "
                         "기본은 카테고리가 하나 이상 붙은 클립만 - 전량은 "
                         "클립당 약 39MB 라 금방 수십 GB 가 된다.")
    ap.add_argument("--gt-labels", default=str(LABELS_JSON),
                    help=f"정답 라벨 json (기본 {LABELS_JSON.name}, config.py "
                         "에서 정함). 시각화 패널에 GT 와 Pred 를 나란히 "
                         "그린다. 라벨에 없는 클립은 Pred 만 그린다.")
    ap.add_argument("--traj", choices=["center", "width"], default=None,
                    help="자차 미래 궤적을 입력 프레임에 그린다 (기본 off). "
                         "center=차량 중심 1줄, width=차폭 2줄. "
                         "calibration/ 의 intrinsic/extrinsic 이 필요하다.")
    ap.add_argument("--ego-track", action="store_true",
                    help="egomotion 을 요약 문장만이 아니라 1초 간격 시계열로도 "
                         "넘긴다 (--use-egomotion 필요, 기본 off). 요약 한 문장은 "
                         "S자 조향이나 감속-가속 반복을 담지 못한다 - 실측 300클립 "
                         "중 74%%가 속도 방향이 바뀌고 10%%는 좌우 회전이 모두 있다.")
    ap.add_argument("--timeline", action="store_true",
                    help="1단계(Scene Description)를 시간순 서술로 바꾼다 "
                         "(기본 off). 20초 안에 사건이 둘 이상일 때 뒤 사건이 "
                         "요약에서 탈락하는 것을 막으려는 것. 3단계도 함께 "
                         "바뀌어 1단계에서 나열한 상황을 다시 훑게 한다.")
    ap.add_argument("--memo", default="",
                    help="이 실행이 무엇을 시험하는지 한 줄 메모. "
                         "run_config.json 에 저장되고 evaluation.log 머리에 "
                         "찍힌다. 예: --memo \"Too Close Person 제거\"")
    ap.add_argument("--viz-normal", action="store_true",
                    help="카테고리가 없는(Normal) 클립을 시각화한다. "
                         "결과는 <viz-dir>/normal/score_<N>/ 아래.")
    ap.add_argument("--viz-special", action="store_true",
                    help="카테고리가 하나 이상인(Special) 클립을 시각화한다. "
                         "결과는 <viz-dir>/special/score_<N>/ 아래.")
    ap.add_argument("--clip-viz-width", type=int, default=None,
                    help="시각화 영상 폭 (기본 1280). 0 을 주면 원본 해상도.")
    ap.add_argument("--not-save-low", type=lambda s: s not in ("0", "false", "False"),
                    default=True,
                    help="Safety Criticality 와 Rarity 가 둘 다 Low 인 클립은 "
                         "시각화에서 제외한다 (기본 True). 카테고리가 나열됐지만 "
                         "모델 스스로 '영향도 낮고 흔함'으로 판단한 경우다. "
                         "CSV/scenario_types 에는 영향 없음 - 시각화 대상만 "
                         "줄인다. --not-save-low=0 으로 끌 수 있다. "
                         "--no-safety-tier/--no-rarity-tier 로 끈 등급은 값이 "
                         "없어 '둘 다 Low' 가 성립하지 않으므로, 이 필터는 "
                         "저절로 무력화된다(놓치는 것보다 더 보는 쪽).")
    ap.add_argument("--no-safety-tier", dest="safety_tiers",
                    action="store_false",
                    help="Safety Criticality 단계를 프롬프트와 출력 스키마에서 "
                         "뺀다. safety_tier/safety_label/safety_reason 은 CSV "
                         "에서 빈 값이 되고, tier_score(safety+rarity 합)도 빈 "
                         "값이 된다. --no-rarity-tier 와 함께 주면 등급 산정 "
                         "없이 1~3단계(요소 탐지)만 남는다.")
    ap.add_argument("--no-rarity-tier", dest="rarity_tiers",
                    action="store_false",
                    help="Rarity 단계를 프롬프트와 출력 스키마에서 뺀다. "
                         "rarity_tier/rarity_label/rarity_reason 은 CSV 에서 "
                         "빈 값이 되고, tier_score 도 빈 값이 된다. 남은 "
                         "단계는 4번으로 다시 매겨진다.")
    ap.add_argument("--viz-per-category", type=int, default=None,
                    metavar="N",
                    help="카테고리마다 최초 N개 클립만 시각화한다. 폴더를 "
                         "score 가 아니라 카테고리 이름으로 나누고, 한 클립이 "
                         "여러 카테고리를 받으면 해당 폴더 전부에 중복 저장한다 "
                         "(<viz-dir>/<Category>/<uuid>/). Normal 클립은 대상이 "
                         "아니다. 샤드마다 따로 세므로 8샤드면 최대 8N 개가 "
                         "나온다.")
    ap.add_argument("--difficulty", action="store_true",
                    help="주행 난이도 5축(전체 + 조도/강수/노면/대기가림)을 "
                         "0~4 로 함께 매긴다. prompts.py 의 눈금을 쓰며, "
                         "값과 근거 문장이 CSV 열로 나가고 시각화 패널과 "
                         "aggregate_clip.log 분포에 실린다. 기본 off - "
                         "출력 토큰이 늘어 느려지므로 필요한 실행에서만 켠다.")
    ap.add_argument("--no-constrain-tiers", dest="constrain_tiers",
                    action="store_false",
                    help="등급(safety/rarity) 필드를 디코딩 단계에서 1/2/3 으로 "
                         "강제하는 것을 끈다. 기본은 강제 - 프롬프트 지시만으로는 "
                         "'Low to moderate' 같은 모호한 답이 새어나왔다.")
    ap.add_argument("--clip-no-video-input", dest="clip_video_input",
                    action="store_false",
                    help="클립 모드에서 프레임을 비디오가 아니라 낱장 이미지 "
                         "목록으로 넘긴다. 기본은 비디오 - Qwen3-VL 이 인접 "
                         "프레임을 병합하고 타임스탬프를 붙여줘서 토큰이 약 "
                         "절반이 된다(실측 4,449 -> 2,296).")
    ap.add_argument("--data", default="local",
                    help="프레임 소스. local(기본, pav_sample) | nas(NAS 청크 "
                         "zip) | 임의 경로. zip 이 보이면 zip 모드로 자동 판별.")
    args = ap.parse_args()

    # 프레임 소스를 먼저 정한다 - 이후 list_scene_uuids 등이 이걸 본다.
    _src = set_clip_source(args.data)
    if _src is not None:
        print(f"[data] {args.data} -> {_src.kind}", flush=True)

    # 세 모드는 상호 배타다. 함께 켜면 ②가 되살아나거나 hint 가 두 번
    # 정의되어 무엇을 재는 실행인지 알 수 없게 된다 - 몇 시간 돌린 뒤
    # 결과를 못 쓰느니 여기서 멈춘다.
    _ego_modes = [n for n, on in (("--use-egomotion", args.use_egomotion),
                                  ("--use-egomotion-c", args.ego_c),
                                  ("--use-egomotion-d", args.ego_d)) if on]
    if len(_ego_modes) > 1:
        ap.error("동시에 쓸 수 없습니다: " + ", ".join(_ego_modes)
                 + " (대조군 C/D 는 --use-egomotion 이 꺼진 상태로 동작합니다)")
    args.ego_ablation = "c" if args.ego_c else ("d" if args.ego_d else None)

    # v2 는 헤더 문구를 바꾸는 옵션인데, 헤더 자체가 없는 조합에서는 아무
    # 일도 하지 않는다. 조용히 무시하면 v1 과 똑같은 결과가 나오고 그걸
    # "v2 는 효과 없음" 으로 읽게 된다 - 실행 전에 막는다.
    if args.header_style == "v2" and not (args.use_egomotion or args.ego_c):
        ap.error("--header-style v2 는 자차 행동 블록이 있어야 의미가 있습니다. "
                 "--use-egomotion 또는 --use-egomotion-c 와 함께 쓰세요 "
                 "(D 모드와 아무 플래그 없는 실행에는 헤더가 없습니다).")

    SCENE_JSON = Path(args.scene_json)
    if not SCENE_JSON.exists():
        raise SystemExit(f"[error] scene json not found: {SCENE_JSON}")
    # nuReasoning 은 Q3(blocking)를 묻지 않는다 - 행동 영향이 3단계와 score
    # 안으로 흡수되기 때문. 사용자가 --no-blocking 을 안 줬어도 강제로 끈다.
    if args.prompt_style == "nureasoning" and args.ask_blocking:
        args.ask_blocking = False
        print("[info] --prompt-style nureasoning: Q3(blocking) is not part of "
              "this prompt, forcing --no-blocking")
    # 클립 모드는 시간순 시퀀스를 전제로 한 nureasoning 프롬프트에만 맞는다.
    # --viz-normal/--viz-special 은 그 자체가 "시각화하라"는 뜻이므로
    # --clip-viz 를 따로 요구하지 않는다 (빼먹으면 조용히 아무것도 안 나온다).
    if args.viz_normal or args.viz_special:
        args.clip_viz = True
    if args.clip_mode and args.prompt_style != "nureasoning":
        args.prompt_style = "nureasoning"
        args.ask_blocking = False
        print("[info] --clip-mode implies --prompt-style nureasoning")

    if args.out is None:
        run_dir = ROOT / "results" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        args.out = str(run_dir / "results.csv")
        if args.viz_dir is None:
            args.viz_dir = str(run_dir)
    elif args.viz_dir is None:
        args.viz_dir = str(Path(args.out).parent)

    labels = load_labels(SCENE_JSON)
    n_special = sum(1 for l in labels if not l["is_normal"])
    category_menu = build_category_menu(
        labels, example_source=args.example_source, num_examples=args.num_examples)
    print(f"[info] labels: {SCENE_JSON.name} - {n_special} special categories")
    print(f"[info] prompt style: {args.prompt_style}"
          + (" (edge-case presence only, no difficulty score)"
             if args.prompt_style == "nureasoning" else ""))
    print(f"[info] category menu: {args.num_examples} example(s) per category "
          f"from {args.example_source}")
    _ego_state = ("ON" if args.use_egomotion
                  else f"ABLATION-{args.ego_ablation.upper()}"
                  if args.ego_ablation else "OFF")
    print(f"[info] sensor facts: egomotion={_ego_state}, "
          f"obstacle={'ON' if args.use_obstacle else 'OFF'}")
    print(f"[info] Q3 blocking question: {'ON' if args.ask_blocking else 'OFF'}"
          + ("" if args.ask_blocking else " (viz not split by blocking)"))
    print(f"[info] 3D path check: {'ON' if args.check_path else 'OFF'}"
          f" (geometric cross-check of Q3, not fed to the model)")
    _views = views_for(args.single_view)
    print(f"[info] camera views: {len(_views)} "
          f"({'front-wide only' if args.single_view else 'front 3-view'})"
          f" -> {2 * len(_views)} images per unit")

    uuids = list_scene_uuids(args.limit_clips)

    # 특정 클립만 처리 (검증용). 라벨된 uuid 는 데이터셋 전체에 흩어져 있어서
    # --limit-clips 로는 못 뽑는다. 목록에 있지만 데이터셋에 없는 uuid 는
    # 조용히 빠지면 원인을 못 찾으므로 개수를 알린다.
    if args.only_uuids:
        wanted = [l.strip() for l in
                  Path(args.only_uuids).read_text(encoding="utf-8").splitlines()
                  if l.strip()]
        have = set(uuids)
        uuids = [u for u in wanted if u in have]
        n_missing = len(wanted) - len(uuids)
        print(f"[info] --only-uuids: {len(uuids)}/{len(wanted)} found"
              + (f"  ({n_missing} not in dataset)" if n_missing else ""))
        if not uuids:
            raise SystemExit("[error] none of the requested uuids exist")

    # 클립 모드는 판정 단위가 uuid 하나라 아래의 frame-unit 경로를 타지 않는다.
    if args.clip_mode:
        if args.num_shards > 1:
            uuids = uuids[args.shard_id::args.num_shards]
            print(f"[info] shard {args.shard_id}/{args.num_shards}: "
                  f"{len(uuids)} clips")
        n_img = args.clip_max_frames * len(_views)
        print(f"[info] clip mode: {len(uuids)} clips, {args.clip_fps} fps, "
              f"max {args.clip_max_frames} frames/view -> {n_img} images/clip "
              f"@ long side {args.clip_long_side}px")
        # 실행 설정을 폴더에 남긴다 - 몇 주 뒤 결과만 보고 "이건 몇 fps 였지"
        # 를 되짚을 방법이 로그 뒤지기밖에 없으면 A/B 비교를 신뢰할 수 없다.
        # 샤드 0 만 쓴다 (8개가 같은 파일에 동시에 쓰면 깨진다).
        if args.shard_id == 0:
            save_run_config(args, Path(args.out).parent, n_views=len(_views))
        if args.dry_run:
            for u in uuids[:3]:
                c = sample_clip_frames(u, fps=args.clip_fps,
                                       max_frames=args.clip_max_frames,
                                       max_long_side=args.clip_long_side,
                                       views=_views)
                sizes = {im.size for _, _, im in c["frames"]}
                print(f"  {u}: {len(c['frames'])} images, idx "
                      f"{c['indices'][:3]}..{c['indices'][-1]}, sizes {sizes}")
            print("[dry-run] done")
            raise SystemExit(0)
        from qwen_runner import run_clip_inference
        # 이 import 가 edge_case_mining 사본을 만든다 - 프레임 소스를 옮겨 심는다
        _mirror_clip_source()
        run_clip_inference(uuids, labels, category_menu,
                           model_id=args.model, out_csv=args.out,
                           use_egomotion=args.use_egomotion,
                           ego_ablation=args.ego_ablation,
                           header_style=args.header_style,
                           want_margin=args.margin,
                           safety_tiers=args.safety_tiers,
                           rarity_tiers=args.rarity_tiers,
                           difficulty=args.difficulty,
                           use_obstacle=args.use_obstacle,
                           single_view=args.single_view,
                           fps=args.clip_fps,
                           max_frames=args.clip_max_frames,
                           max_long_side=args.clip_long_side,
                           viz_dir=(args.viz_dir if args.clip_viz else None),
                           viz_per_category=args.viz_per_category,
                           viz_only_edge=not args.clip_viz_all,
                           not_save_low=args.not_save_low,
                           gt_labels=_load_gt_labels(args.gt_labels),
                           timeline=args.timeline,
                           ego_track=args.ego_track,
                           traj=args.traj,
                           viz_normal=(args.viz_normal
                                       if (args.viz_normal or args.viz_special)
                                       else None),
                           viz_special=(args.viz_special
                                        if (args.viz_normal or args.viz_special)
                                        else None),
                           video_input=args.clip_video_input,
                           constrain_tiers=args.constrain_tiers,
                           **({"viz_width": (args.clip_viz_width or None)}
                              if args.clip_viz_width is not None else {}))
        raise SystemExit(0)

    print(f"[info] clips: {len(uuids)}  x  {args.timestamps_per_clip} timestamps/clip")

    units = list_frame_units(uuids, args.timestamps_per_clip)
    print(f"[info] total (clip, timestamp) units: {len(units)}")

    if args.num_shards > 1:
        units = units[args.shard_id::args.num_shards]
        print(f"[info] shard {args.shard_id}/{args.num_shards}: {len(units)} units")

    if args.dry_run:
        for uuid, idx in units[:3]:
            uf = sample_unit_frames(uuid, idx, views=_views)
            prev_ok = {v: (im.size if im else None) for v, im in uf["prev"].items()}
            cur_ok = {v: (im.size if im else None) for v, im in uf["cur"].items()}
            print(f"  {uuid} cur@{uf['cur_idx']} prev@{uf['prev_idx']}")
            print(f"     prev: {prev_ok}")
            print(f"     cur : {cur_ok}")
        print("[dry-run] done")
        raise SystemExit(0)

    from qwen_runner import run_inference
    _mirror_clip_source()
    run_inference(units, labels, category_menu,
                  model_id=args.model, out_csv=args.out, viz_dir=args.viz_dir,
                  use_egomotion=args.use_egomotion, use_obstacle=args.use_obstacle,
                  check_path=args.check_path, ask_blocking=args.ask_blocking,
                  single_view=args.single_view,
                  prompt_style=args.prompt_style)

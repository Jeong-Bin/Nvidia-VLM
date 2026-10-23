#!/usr/bin/env python3
"""
VLM 기반 반자동 edge-case (long-tail) mining - 클립 단위.

판정 단위는 클립(uuid) 하나다. 20초 front-wide 영상을 2fps 로 뽑아 비디오
한 편으로 Qwen3-VL 에 넣고, nuReasoning 방식의 단계별 JSON 답(장면 설명 ->
자차 행동 -> 특이 요소 -> scenario_types -> 묶음 점수 ...)을 받는다.
추론 루프는 qwen_runner.run_clip_inference 에 있다.

멀티라벨이므로 한 장면에 여러 카테고리가 동시에 붙을 수 있다
(예: 공사장에서 수신호 -> Road Construction + Manual Traffic Control).

Usage (보통은 run_labeled_8b.sh / run_nas_nvidia_8b.sh 가 샤드를 나눠 부른다):
  python edge_case_mining.py --only-uuids uuids.txt --use-egomotion \
      --out results/<run>/clip_results_shard_0.csv
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
import visualize_clip as VZ
from constrained_tier import (TIER_LABELS, tier_score,
                              group_rubric_blocks, group_key,
                              score_groups, GROUP_RUBRICS)
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


# ---------------------------------------------------------------------------
# 라벨 로딩 (배열 스키마: {normal,special}.scenarios[].categories[])
#
# normal 도 special 과 완전히 동일한 구조를 쓴다 (예: "Normal Driving",
# "Normal Stop" 카테고리). 2단계 매칭에서는 이 둘을 special 카테고리와
# 동등한 후보로 취급한다 - is_normal 플래그로만 구분.
# ---------------------------------------------------------------------------
def load_labels(scene_json: Path):
    """scene_category.json -> [{scenario, category, templates, template_candidates, is_normal}, ...]"""
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
                        "templates": cat.get("templates", []),
                        "template_candidates": cat.get("template_candidates", []),
                        # 이 카테고리가 "아닌" 경우. 긍정 예시만으로는 경계가
                        # 안 잡히는 카테고리에만 쓴다 (없으면 빈 리스트).
                        "excludes": cat.get("excludes", []),
                        # [[상황, 점수], ...] - 4단계 rubric 표 아래에 싣는
                        # 예시(constrained_tier.group_rubric_blocks). 없으면 빈 리스트.
                        "score_examples": cat.get("score_examples", []),
                        # "unknown": true 인 카테고리(예상 못한 새 종류)는
                        # --unknown-categories 를 줄 때만 메뉴에 넣는다.
                        "unknown": bool(cat.get("unknown")),
                        "is_normal": is_normal,
                    }
                )
    return labels


def _examples_for(lab):
    """카테고리 하나에서 예시로 쓸 문자열 리스트 - templates 전부.

    templates 가 빈 카테고리는 template_candidates(상황 서술 문장)로 대체한다.
    """
    return lab["templates"] or lab["template_candidates"]


def build_category_menu(labels, notes=()):
    """프롬프트에 넣을 special 카테고리 목록.

    형식: `- Road Construction(roadwork, traffic cone)`
    카테고리명과 예시를 한 덩어리로 묶어 제시하므로, 모델이 별도의 매칭 단계
    없이 정확한 카테고리명으로 바로 답할 수 있다.

    templates 는 전부 노출하는 것이 기본이다. 예전에는 상위 2개만 썼는데
    (전부 나열하면 판별이 경직된다는 20260724 관찰), scene_category_2.0
    부터 templates 를 그 자체로 판별 기준이 되도록 다듬었으므로 잘라내지
    않는다.
    normal 은 제외 - Q1(Normal/Special)이 그 역할을 하고, Q2 는 special
    카테고리만 나열하는 자리이기 때문.
    """
    lines = []
    for lab in labels:
        if lab["is_normal"]:
            continue
        ex = _examples_for(lab)
        lines.append(f"- {lab['category']}({', '.join(ex)})" if ex
                     else f"- {lab['category']}")
        # 부정 조건은 별도 줄로. 실측(20260818, 100클립): Jaywalking 은 긍정
        # 예시에 "where there is no crosswalk" 가 있는데도 횡단보도를 정상
        # 통행하는 보행자를 20건 오탐했다 (FP 20 = 전체 FP 38 의 53%).
        # 모델이 "crossing the street" 자체를 카테고리로 읽고 있어서,
        # 긍정 예시를 늘리는 대신 무엇이 아닌지를 못박는다.
        for ex_line in lab["excludes"]:
            lines.append(f"    NOT this category: {ex_line}")
    # 여러 카테고리에 공통인 규칙(scene_category.json 의 special.menu_notes)은
    # 메뉴 끝에 한 번만 싣는다. 예전에는 "중앙분리대/갓길/주차 차로는 도로
    # 밖이 아니다" 같은 문장이 Pedestrian/Cyclist/Emergency Vehicle 의
    # excludes 마다 반복돼 메뉴가 길어졌다(프롬프트가 길수록 탐지가 떨어진
    # 실측: 5,585 -> 6,389자에서 F1 80.4 -> 74.0).
    for note in notes:
        lines.append(f"\n{note}")
    return "\n".join(lines)


def load_menu_notes(scene_json) -> list[str]:
    """scene_category.json 의 special.menu_notes - 카테고리 공통 규칙 문장들."""
    data = json.loads(Path(scene_json).read_text(encoding="utf-8"))
    return list(data.get("special", {}).get("menu_notes", []))


# ---------------------------------------------------------------------------
# 프레임 읽기
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

    실패한 인덱스는 결과 dict 에서 빠진다. 그러므로 돌려받은 개수가 요청한
    개수보다 적을 수 있다 - 호출부(run_clip_inference)가 개수를 대조해
    모자라면 추론하지 않고 실패로 기록한다.

    읽는 도중의 OSError 도 여기서 잡는다. 파일을 여는 단계의 OSError 는
    예전부터 잡았지만 디코딩 중에 NAS 가 끊기면 나는 것(Errno 112 Host is
    down)은 잡지 않아, 그 예외가 샤드 프로세스를 통째로 죽였다(실측
    20260904_124936_video8b 샤드 2). 여는 단계와 똑같이 "못 읽음"으로
    돌려주면 프로세스는 계속 돌고 그 클립만 실패 행으로 남아, 나중에
    --resume 으로 다시 돌릴 수 있다.
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
    except (AV_ERROR, OSError):
        pass
    finally:
        # 연결이 끊긴 파일은 닫다가도 OSError 를 낸다. finally 에서 새는
        # 예외는 위에서 잡은 것을 덮어쓰고 호출부까지 올라간다.
        try:
            container.close()
        except (AV_ERROR, OSError):
            pass
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


def list_scene_uuids(limit: int | None = None):
    """front_wide 뷰 기준 클립 uuid 리스트 (정렬, 선택적 limit).

    세 뷰 모두 같은 uuid 집합을 공유함 (500 클립 x 3앵글).
    """
    if CLIP_SOURCE is not None:
        uuids = CLIP_SOURCE.uuids(FRONT_VIEWS[0])
    else:
        d = CAMERA_DIR / FRONT_VIEWS[0]
        # set 으로 한 번 걸러 같은 uuid 를 두 번 돌리지 않는다. NAS 청크에는
        # 원본 저장소의 중복 사본이 1,180건 있고(ZipSource 가 걸러낸다),
        # 로컬도 재다운로드나 수동 복사로 같은 상황이 생길 수 있다.
        uuids = sorted({p.name.split(".")[0] for p in d.glob("*.mp4")})
    if limit:
        uuids = uuids[:limit]
    return uuids


# ---------------------------------------------------------------------------
# 클립 샘플링 - 20초를 일정 간격으로 훑어 시간축을 모델에게 직접 보여준다.
# cut-in, 갑작스러운 보행자 진입처럼 몇 초에 걸쳐 전개되는 것을 잡기 위해서다.
# 뷰는 front_wide 하나가 기본이다(--single-view) - 3뷰면 토큰이 3배가 되고,
# 논문(nuReasoning)도 마이닝 단계는 front 카메라 하나만 쓴다.
# ---------------------------------------------------------------------------
CLIP_FPS = 2.0              # 초당 몇 장 뽑을지
CLIP_MAX_FRAMES = 80        # 클립당 최대 장수 (20초 x 1fps)
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
        # clip_open 도 NAS 에서는 zip 을 열다가 OSError 를 낼 수 있다. 여기서
        # 새면 샤드가 죽으므로 "이 뷰는 한 장도 못 읽음"으로 둔다 - 호출부가
        # 개수를 대조해 실패로 기록한다.
        try:
            src = clip_open(view, uuid)
        except OSError:
            per_view[view] = {}
            continue
        per_view[view] = _read_frames_at(src, idxs,
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
# 프롬프트 문구 실험 (20260728, 공사장 클립 3프레임 + 평범한 6프레임으로 검증):
#   - "Mere presence is not enough ..." 같은 억제 규칙을 넣으면 탐지가 0 이 된다.
#     명백한 공사장에서도 categories 가 비어버림.
#   - verdict 와 categories 를 묶으면("카테고리를 하나라도 적으면 Special")
#     역시 탐지가 0 이 된다. 모델이 Special 선언에 보수적이라, 나열이 Special 을
#     강제하는 구조에서는 아예 나열을 포기한다.
#   - 카테고리를 먼저 묻고 verdict 를 나중에 물어도 탐지가 0. evidence 에는
#     "Construction site ..." 라고 쓰면서 categories 는 비우는 모순이 나타난다.
#   => 판정은 "이상 요소를 나열했는가"로만 정한다. 조건을 거는 순간 탐지가 죽는다.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 프롬프트 (nuReasoning 방식)
#
# nuReasoning(arXiv 2605.31572) Fig. S1/S2 의 마이닝 프롬프트에서 추론 절차만
# 가져온 것:
#
#   - 단계별 chain-of-thought 를 거친다. 각 단계가 JSON 필드로 남으므로
#     "왜 그렇게 봤는지"를 사후 검수할 수 있다.
#   - 3단계에서 이상 요소마다 "자차 행동을 바꿨는가"를 함께 적게 한다.
#     검수자가 읽을 정보이지 탐지를 거르는 조건이 아니다.
#
# 논문과 다른 점: 난이도 점수(Final Assessment, 1~10)를 쓰지 않는다.
# 우리 과제는 "이 클립에 edge-case 요소가 있는가"를 가리는 것이지 롱테일
# 가치를 서열화하는 것이 아니다. 그래서 6단계와 채점 rubric, 그리고 점수를
# 낮추라는 취지의 behavior-centric 억제 문구를 모두 뺐다.
#
# 그 억제 문구를 빼는 것은 실측 근거도 있다: 위 20260728 실험에서 "Mere presence is not enough" 류의 규칙을 넣자 명백한 공사장에서도
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


def build_nureasoning_prompt(category_menu: str,
                             labels: list | None = None,
                             behavior_facts: str = "",
                             intro: str | None = None,
                             timeline: bool = False,
                             score_categories: bool = True,
                             difficulty: bool = False,
                             difficulty_only: bool = False,
                             explain_traj: bool = False,
                             time_spans: bool = False) -> str:
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

    intro 는 clip_intro() 가 만든 도입부. behavior_facts 는 egomotion 에서
    뽑은 "클립 동안 자차가 어떻게 움직였는가" 문장이며, 논문 2단계(Ego
    Behavior Summary)의 근거로 쓰인다. 비어 있으면 그 블록이 통째로 빠지고,
    모델은 영상만으로 행동 변화를 추정하게 된다.
    """
    if intro is None:
        raise ValueError("intro 가 필요합니다 (clip_intro)")

    fact_block = ""
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
        # (이 분해에 쓴 대조군 C/D 와, 절차만 말하는 v2 헤더 옵션은 쓰지 않아
        # 2026-10-23 에 지웠다.)
        # 3단계에서 영향 여부를 빼서 그 판단은 5단계 점수에서만 한다.
        behavior_use = ("for the scores in step 5"
                        if score_categories and score_groups(labels or [])
                        else "for nothing else")
        fact_block += f"""
HOW THE EGO-VEHICLE'S BEHAVIOUR CHANGED (measured from vehicle sensors, not a
guess - use this for step 2 and {behavior_use}):
{behavior_facts}
"""

    # --explain-traj: 프레임에 그려진 초록 선이 무엇인지 한 문장으로 알린다.
    #
    # 기본은 알리지 않는 것이다. 예전에 알려 봤다가 6개 축이 모두 나빠졌기
    # 때문이다(실측 115클립, 위 docstring 참고) - "궤적으로 어떤 도로
    # 사용자와 실제로 상호작용하는지 보라"는 식으로 쓰자 그 문구가 필터로
    # 읽혀, 경로 밖 보행자를 특이 요소에서 통째로 빼버렸다.
    #
    # 그래서 이 문장은 선이 무엇인지만 말하고, 그것으로 무엇을 하라는 말은
    # 하지 않는다. 판단 지시를 넣는 순간 그때의 실패가 재현된다.
    traj_note = ("\nA green line is drawn on the frames: it marks the "
                 f"ego-vehicle's own path over the next {_TRAJ_HORIZON_S:g} "
                 "seconds. It is an overlay, not an object in the scene."
                 if explain_traj else "")

    # 묶음(Dynamic object / Driving environment)마다 점수 하나. 묶음과 그
    # 기준표는 labels(scene_category.json)와 constrained_tier.GROUP_RUBRICS
    # 에서 오므로 카테고리 파일을 바꾸면 여기도 따라간다. 기준표 칸이 비어
    # 있으면 group_rubric_blocks 가 멈춘다.
    groups = score_groups(labels or []) if score_categories else []
    rubric_body = group_rubric_blocks(labels or []) if groups else ""

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
    behavior_hint = (
        "\n   The measured ego behaviour above says what the vehicle did, not"
        "\n   why. Slowing or stopping is routine (signals, junctions, queues),"
        "\n   so treat it as evidence only when the video shows what caused it."
        if behavior_facts else "")

    # 4단계: 클립에 나온 카테고리 이름을 명시적으로 고른다(scenario_types).
    #
    # 예전에는 3단계 끝의 "Also name the matching scenario types" 한 문장과
    # JSON 맨 끝의 scenario_types 로 이름을 받았다. 점수가 카테고리가 아니라
    # 묶음에 붙게 되면서, 어느 카테고리가 나왔는지는 이 목록만 말해 준다
    # (GUI 카테고리 검색과 탐지 채점이 이것을 쓴다). 그래서 단계로 올리고,
    # JSON 에서도 점수보다 앞에 둔다 - 5단계의 "이름을 댄 묶음은 1 이상,
    # 대지 않은 묶음은 0" 규칙이 생성 순서상 이미 적은 목록을 참조하게.
    #
    # 5단계: 묶음마다 0~4 점수 하나와 그 근거.
    #
    # 카테고리마다 점수를 매기던 방식에서 바꾼 이유는 새 카테고리가 생겨도
    # 기준표를 늘리지 않고 한 통합 기준으로 재기 위해서다. 대가는 요소가
    # 여럿일 때 무엇이 점수를 정했는지 숫자만으로 알 수 없다는 것 - 그래서
    # 근거에 "어느 종류가, 무엇을 해서" 그 점수인지 적게 한다.
    #
    # 가장 어려웠던 요소(또는 상황)로 매기고 평균은 금지한다 - 예전
    # safety/rarity 에서 모델이 평균을 냈다(실측 20260904, 27,024클립: 요소 수와
    # 무관하게 rarity 평균 2.00 고정, safety 는 요소 2개 1.53 -> 3개 1.36).
    # "여러 요소가 있으면 HIGHEST" 로만 쓰면 요소 수가 점수와 관련 있다는
    # 인상을 주므로, 하나로도 4 가 될 수 있고 개수 자체는 점수를 정하지
    # 않는다고 못박는다. 다만 여러 도로 사용자를 "동시에" 상대해야 했던
    # 장면(인파 사이를 조율하며 서행)은 개별 최고점보다 어렵다 - 그 경우만
    # 하나의 상황으로 묶어 매기게 한다. Dynamic object 3 점 문구와 맞물리는
    # 규칙이라 그 문구를 고칠 때 함께 본다.
    #
    # 제외된 인스턴스(excludes)는 점수를 정하지 못한다는 문장도 남긴다 -
    # 인도 위 보행자에 점수를 주던 오답이 이 문장으로 13 -> 8건이 됐다
    # (20261006_143241 -> 163205).
    #
    # 근거를 점수보다 먼저 쓰게 하는 것도 그대로다(JSON 순서 = 생성 순서).
    # 점수가 앞이면 숫자를 먼저 정하고 근거는 사후 설명이 됐다(실측
    # 20261002_175140: 근거는 1점 문장인데 숫자는 2인 경우 17건 -> 순서를
    # 바꾼 뒤 3건).
    step_no = 4
    steps45 = f"""{step_no}. Scenario Types: name every scenario type from the list above that
   matches an element in step 3, copying the names EXACTLY. An element that a
   type's "NOT this category" lines exclude is not named. If none matches,
   give an empty list.
"""
    step_no += 1
    if groups:
        _gnames = " and ".join(groups) if len(groups) <= 2 else (
            ", ".join(groups[:-1]) + " and " + groups[-1])
        _who = (f"each group - {_gnames} -" if len(groups) > 1
                else f"the group {_gnames}")
        steps45 += f"""{step_no}. Difficulty Scores: give {_who}
   one score from 0 to 4 for how much the types of that group made driving
   harder for the ego-vehicle in this clip. A group none of whose types you named in step
   {step_no - 1} scores 0; a group whose types you named scores at least 1.
   Score what each element is DOING, never its name alone - the same object is
   routine or serious depending on what it is doing and where it is.
   Score the element, or the situation, that made driving hardest. A single
   element can earn any score, up to 4 - the number of elements never sets the
   score by itself. When there are several, never average them: a harmless one
   never lowers the score. Several elements count together only when the
   vehicle had to deal with them at the same moment - weaving through a crowd,
   say - and then that situation is scored as one. An element that its type's
   "NOT this category" lines exclude never sets the score.
   These scores are INDEPENDENT of weather, lighting, or road/atmospheric
   conditions - rain, nighttime, fog, or a wet road do not by themselves raise
   any score. Score what actually happened in the scene, not how hard the
   conditions were to drive in.
   Before each score, give its reason: name the type that decided it and say
   what that element did, or what the situation was, that earned the score.
   Use the scale listed for each group:
{rubric_body}{behavior_hint}
"""
        step_no += 1
    # --time-spans: 이름을 댄 종류마다 "언제" 자차에 위험했는지 [시작, 끝] 초를
    # 받는다. 시각화 영상의 재생 바 위에 구간 띠로 그리기 위한 것이다.
    #
    # 원본 프롬프트에 단계와 JSON 칸이 늘어나므로 장르/점수 판단도 흔들릴 수
    # 있다(문구 몇 줄로 F1 이 ±5~10%p 움직여 왔다). 그래서 기본은 끄고,
    # 켰을 때만 이 단계가 붙는다 - 끈 실행의 프롬프트는 글자 하나 다르지 않다.
    #
    # 자리는 점수 바로 뒤다. JSON 순서가 곧 생성 순서라 앞(장르, 점수)에 두면
    # 그 판단이 시간 칸에 끌려갈 수 있고, 점수 근거를 쓴 직후라 "무엇이
    # 점수를 정했나"와 같은 요소를 가리키기 쉽다. 날씨는 이 판단과 무관하므로
    # 뒤로 밀린다.
    #
    # 시각은 영상에 박힌 타임스탬프(Qwen3-VL 이 프레임 묶음마다 붙이는
    # "<3.5 seconds>")를 그대로 읽게 한다. 원본 mp4 의 0 초가 첫 프레임이라
    # (sample_clip_indices 가 0 프레임부터 15 프레임 간격으로 뽑는다) 답한
    # 초를 시각화 영상의 시간에 바로 쓸 수 있다.
    #
    # 구간마다 0~4 점수를 붙인다(A 방식: 구간 경계는 모델이 정한다).
    # 점수는 5단계와 같은 기준표로 매기게 한다 - 따로 척도를 주면 같은 클립의
    # 점수와 구간 점수가 서로 다른 잣대가 된다. 구간 점수가 5단계 점수를
    # 넘지 않게 해 두 값이 어긋나지 않게 한다.
    #
    # 최소 길이 1초: 모델이 구분하는 시점이 1초에 하나다(Qwen3-VL 이 2fps
    # 프레임을 2장씩 묶는다). 그보다 짧은 구간은 볼 수 없는 차이를 지어낸
    # 것이다. 구간 길이는 정하지 않는다 - 고정하면 초마다 점수를 매기는 꼴이
    # 되어 긴 숫자 배열의 위치를 세다 밀리는 문제가 생긴다.
    #
    # 점수 단계(5단계)를 끈 실행에는 기준표가 없으므로 구간만 받는다.
    #
    # "두 단계로 묻기(보인 시간 -> 점수 구간)" + "통째 구간 금지" 문구를 넣어 본
    # 적이 있다(20261014_175954 inline / 20261019_141710 followup). 잠깐 나온
    # 요소의 0~20초 통째 구간은 줄지 않았고(21% -> 24%, 23% -> 27%), 점수
    # 구간이 보인 시간을 그대로 베껴(105/106) 구간 안의 점수 변화가 사라졌다
    # (구간을 나눈 답 17 -> 1). 그래서 이 문구로 되돌렸다.
    if time_spans and not difficulty_only:
        if groups:
            steps45 += f"""{step_no}. Hazard Timing: for each type you named in step 4, split the time its
   element was in the clip into stretches, and score each stretch from 0 to 4
   on the scale of step 5 - how much that element made driving harder for the
   ego-vehicle during that stretch. Start a new stretch wherever the score
   changes; each stretch lasts at least 1 second, and its length is otherwise
   up to you. No stretch scores higher than the score its group got in step 5.
   Read the times from the timestamps of the video, in seconds from its start.
   If you named no type, give an empty object.
"""
        else:
            steps45 += f"""{step_no}. Hazard Timing: for each type you named in step 4, give when its element
   was a hazard to the ego-vehicle - the stretch of the clip in which it was
   in or coming into the vehicle's way, or otherwise made driving harder. Read
   the times from the timestamps of the video, in seconds from its start, and
   give one or more [start, end] spans per type. If the element never made
   driving harder, give the stretch in which it stood out most. If you named
   no type, give an empty object.
"""
        step_no += 1
    # 날씨(--weather)는 점수 다음 단계로 붙는다. 점수와 성격이 다르다:
    # 등급은 "이 클립이 얼마나 위험/희귀한가"(edge-case 여부에 달림)이고,
    # 난이도는 "이 장면의 주행 조건이 얼마나 나쁜가"(평범한 클립도 비 오면
    # 높다)라서, 서로 독립으로 매기게 두어야 한다. 그래서 앞 단계를
    # 참조하는 문구를 넣지 않는다.
    #
    # 단계 이름을 "Driving Difficulty" 에서 "Weather" 로 바꿨다. 이건 주석이
    # 아니라 모델이 읽는 문장이므로 추론에 영향을 줄 수 있다 - 옛 실행과
    # 숫자를 직접 비교하기 전에 같은 이름으로 한 번 더 돌려봐야 한다.
    # 뒤의 "how hard the driving CONDITIONS are" 가 실제 판단 기준을 말하고
    # 있어 제목만으로 척도가 바뀌지는 않을 것으로 본다.
    if difficulty:
        steps45 += (f"{step_no}. Weather: how hard the driving "
                    f"CONDITIONS are, judged\n   independently of the "
                    f"edge-case decision above - an ordinary clip in heavy "
                    f"rain\n   still scores high, and a rare event on a "
                    f"clear day does not.\n") + difficulty_block()
        step_no += 1
    tier_fields = ""
    # weather-only 는 카테고리를 아예 다루지 않는다. 그 모드에서 이 필드를
    # 남기면 스키마가 프롬프트 본문에 없는 scenario type 을 요구하게 되어,
    # 모델이 카테고리를 지어내거나 빈 딕셔너리를 채우느라 예산을 쓴다.
    # main() 이 --weather-only 에 score_categories=False 를 함의시키지만,
    # 함수를 직접 부르는 경로(테스트/다른 호출부)도 있어 여기서 막는다.
    # 묶음마다 근거 -> 점수 순서. 키는 묶음 이름에서 만든다
    # (constrained_tier.group_key: "Dynamic object" -> "dynamic_object").
    for g in groups:
        k = group_key(g)
        tier_fields += (
            f' "{k}_reason": "<which {g} type decided the score, and what '
            f'it did or what the situation was>",\n'
            f' "{k}": <integer 0-4>,\n')
    if time_spans and not difficulty_only:
        tier_fields += (' "time_spans": {"<exact scenario type name>": '
                        + ('[[<start seconds>, <end seconds>, <score 0-4>], '
                           '...], ...},\n' if groups else
                           '[[<start seconds>, <end seconds>], ...], ...},\n'))
    if difficulty:
        # 프롬프트 본문과 같은 순서. JSON 필드 순서가 곧 생성 순서다.
        for key, name in DIFFICULTY_AXES:
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
    # 3단계 요소마다 붙이던 칸을 뺀다 - 영향 여부(influenced_ego)도, 장르 후보도.
    #
    # 영향 여부는 5단계 점수와 정보가 겹치고, 점수도 그 값을 따르지 않았다
    # (실측 20261013_101558, 2차 확인 전: yes 99건 중 0~1점 19건 - 빨간불에
    # 서서 기다림처럼 기준표상 1점, no 38건 중 2점 4건 - 근거 문장이 바로잡음).
    #
    # 그 칸에 장르 후보(Pre-Categories, "어느 장르로 보이는가, 없으면 none")를
    # 적게 해 봤더니 4단계가 거르지 못하고 후보를 그대로 옮겨 적었다
    # (실측 20261013_110400, 2차 확인 전 TP/FP/FN 85/36/30 -> 109/95/6,
    # F1 72.0 -> 68.3). 인도 위 보행자에도 "Pedestrian on Road" 를 붙이는 등
    # 요소마다 이름을 대게 하는 압력이 그대로 탐지 압력이 됐다.
    #
    # 그래서 칸 없이 요소만 나열한다. 칸이 영향 없는 요소를 목록에 살려
    # 두던 효과가 사라질 수 있다 - 이 판에서도 성능이 떨어지면 영향 여부를
    # 되살린다(커밋 3722a95 의 문구).
    if timeline:
        step1 = """1. Scene Description: walk through the clip in time order. A 20-second clip
   often passes through more than one distinct situation, so describe them
   one at a time with the rough timestamp of each, rather than blending the
   whole clip into a single summary - for example "0-6 s: crossing a railroad;
   8-20 s: a pedestrian steps into the lane between parked cars". Cover the
   road environment and the visible agents or objects in each. If a stretch
   has nothing worth noting, say so instead of inventing something.
   Do not describe lighting or weather."""
        step3_head = """3. Unusual Elements: go back over every situation you listed in step 1
   and check each one against the scenario types. An element that appears in
   only part of the clip counts exactly as much as one that lasts throughout,
   so a type you named in step 1 must not disappear here. List every unusual
   element."""
        observation_field = ("<the clip in time order, each situation with "
                             "its rough timestamp>")
    else:
        step1 = """1. Scene Description: the road environment and the visible agents or objects.
   Do not describe lighting or weather."""
        step3_head = """3. Unusual Elements: list every unusual element."""
        observation_field = "<scene description, one or two sentences>"

    # 노면도 요소로 센다. "unusual element" 를 모델이 물체로만 읽어, 1단계에
    # "unpaved dirt road" 라고 써 놓고 3단계에서 빠뜨렸다(실측
    # 20261002_175140: Unpaved road FN 11건 중 8건이 1단계에 unpaved/dirt 를
    # 적었다). 한 줄로만 둔다 - 3단계 지시가 길어질 때마다 다른 요소를
    # 빠뜨리는 쪽으로 무너졌다(5585 -> 6389자에서 FN 10 -> 35).
    step3_head += """
   The road itself counts as an element when its surface is unusual - an
   unpaved, dirt or gravel road is listed here like any object."""

    elements_field = "<each unusual element>"

    # ------------------------------------------------------------------
    # --weather-only: 주행 조건(날씨) 4축만 묻는다.
    #
    # edge-case 탐지 문구를 남긴 채 단계만 끄면 프롬프트가 자기모순에 빠진다:
    #   (1) 1단계가 "Do not describe lighting or weather" 로 조도/날씨 서술을
    #       금지하는데, 4단계는 바로 그것을 점수로 매기라고 요구한다. 난이도가
    #       유일한 과제가 되면 이 금지문이 근거 관찰 자체를 막는다.
    #   (2) 머리말의 "You are NOT rating how difficult ... the clip is" 가
    #       이 모드에서는 정반대 지시가 된다.
    #   (3) 4단계의 "judged independently of the edge-case decision above" 가
    #       사라진 단계를 가리킨다(rarity 단독 실행에서 겪은 것과 같은 문제).
    # 그래서 문구를 덧대지 않고 과제 선언부터 다시 쓴다. 관찰 지시는 금지가
    # 아니라 요구로 뒤집어, 점수의 근거를 먼저 보게 한다.
    # 센서 사실(속도/가속도/주변 객체)은 넣지 않는다. 조도·강수·노면·대기가림은
    # 영상에서만 읽히는 값이라 자차 속도가 근거가 될 수 없고, 남겨두면 그 블록이
    # "use this for step 2 and step 3" 처럼 사라진 단계를 가리킨다((3)과 같은 문제).
    if difficulty_only:
        return f"""{intro}

Your job is to rate the DRIVING CONDITIONS of this clip - the light, the
weather, and the state of the road surface. You are not looking for unusual
events or rare situations, and nothing in this task depends on whether the
scene is ordinary: most clips are ordinary, and an ordinary clip in good
conditions should be rated {DIFFICULTY_MIN} on every axis.

Work through these steps in order:
1. Condition Description: describe what the scene looks like in terms of
   lighting, weather and road surface - time of day, how well lit the scene is,
   whether anything is falling (rain or snow), whether the road is dry, wet,
   snow-covered or unpaved, and how far you can see through the air. Describe
   only what you can actually see; if something is not visible, say so rather
   than guessing.
2. Weather: rate each factor from what you described in step 1.
{difficulty_block()}
Respond with ONLY a JSON object, no other text:
{{"observation": "<what the lighting, weather and road surface look like>",
{tier_fields.rstrip().rstrip(',')}}}"""

    # 카테고리 메뉴 바로 아래의 "무엇을 넣는가" 지시.
    #
    # 예전에는 "자차가 반응하지 않았어도 넣어라 - 빼는 이유가 될 수 없다"였다.
    # 모델이 영향 없는 요소를 "적을 필요 없다"며 빼는 경향이 실측으로 확인돼
    # 넣은 장치다(점수를 끄자 3단계가 통째로 빈 클립 41 -> 52). 그런데 이제
    # 경로 밖 객체는 excludes 로 빼므로 "절대 빼지 마라"가 excludes 와 정면으로
    # 부딪친다.
    #
    # "경로 밖이면 빼라"로 바꾸지도 않는다 - 이 문장은 모든 카테고리에 걸리는데,
    # 도로 밖 공사(CONSTRUCTION 1)나 자차가 안 가는 길의 차단기(GATE 1)는 경로
    # 밖이어도 카테고리다. 그래서 빼는 기준은 카테고리 설명과 "NOT this
    # category" 줄에 맡기고, 반응 여부만은 기준이 아니라고 남긴다. 새 DYNAMIC
    # 에도 자차가 반응하지 않은 칸(2, 3 의 "그대로 주행")이 있어 이 보호 장치는
    # 여전히 필요하다.
    #
    # "scored in step 5" 는 점수 단계가 있을 때만 붙인다 - 없는데 붙이면 없는
    # 단계를 가리키거나, 날씨 단계를 점수 단계로 읽게 된다.
    # 3단계에서 영향 여부를 빼면서 "recorded separately in step 3" 도 고쳤다.
    influence_ref = ("; how much it changed the driving is scored\n"
                     "separately in step 5" if groups else "")
    # 머리말의 "You are NOT rating how difficult" 는 점수 단계가 있으면 5단계와
    # 정면으로 모순된다. 탐지는 난이도와 무관하게 한다는 뜻만 남긴다.
    if groups:
        job_note = ("Whether an element counts never depends on how difficult "
                    "it made the\ndriving - that is scored separately, after "
                    "the types are named.")
    else:
        job_note = "You are NOT rating how difficult or how valuable the clip is."

    return f"""{intro}{traj_note}
{fact_block}
Your job is to decide whether this clip contains any edge-case element - a rare
or unusual road situation - and to name which of the types below it matches.
{job_note}

SCENARIO TYPES:
{category_menu}

List EVERY type you can actually see. What decides whether an element counts is
the type's description and its "NOT this category" lines - never whether the
ego-vehicle reacted to it. An element the ego-vehicle drove past without slowing
or steering can still count{influence_ref}. If you see no unusual element at all, return
an empty list.

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
 "unusual_elements": "{elements_field}",
 "scenario_types": ["<exact scenario type name>", ...],
{tier_fields.rstrip().rstrip(',')}}}"""


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

    _coerce_tier 와 따로 두는 이유는 폴백 대상이 다르기 때문이다 - 난이도는
    "Low"/"Moderate" 같은 문자열 라벨로 나올 일이 없고, 숫자만 뽑으면 된다.
    0 을 흡수하지 못하면 "조건이 전혀 나쁘지 않다" 는 가장 흔한 답이 통째로
    미기록으로 떨어진다.

    값이 "2 (moderate)" 나 "2/4" 처럼 지저분할 수 있다. 앞머리 정수만 떼어
    쓰고, 범위 밖이면 버린다 - 클램프해서 0 이나 4 로 밀어 넣으면 분포가 양
    끝에 가짜로 쌓인다.
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
    """모델이 낸 등급 값을 0~4 정수로 만든다. 못 읽으면 None.

    값이 "2 (Moderate)" 처럼 지저분하거나 예전 형식("Low"/"Moderate"/"High"
    문자열)일 수 있어서 문자열 폴백을 남긴다.
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
        m = re.match(r"\s*(\d+)\b", t)
        if m:
            n = int(m.group(1))
            return n if n in TIER_LABELS else None
        label = _extract_tier(t)
        for k, name in TIER_LABELS.items():
            if name == label:
                return k
    return None


def _extract_tier(text: str) -> str:
    """자유 서술에서 Low/Moderate/High 등급만 뽑는다 (구 형식 폴백).

    프롬프트가 정수를 요구하도록 바뀐 뒤로는 주 경로가 아니다. 모델이 그래도
    문자열로 답할 때를 위해 남겨둔다.

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
    """nuReasoning 출력 -> 결과 dict (CSV 한 행과 시각화의 재료).

      scenario_types -> categories   (메뉴에 있는 이름만, 대소문자 차이는 흡수)
      verdict        -> categories 가 비어 있지 않으면 "Special"
      observation 등 -> 단계별 답을 그대로 보존

    group_scores 는 {묶음 이름: 0~4 정수} - Dynamic object / Driving
    environment 처럼 scene_category.json 의 묶음마다 하나다. GT 라벨의 같은
    이름 키와 바로 비교된다. score_reason 은 묶음별 근거를
    "Dynamic object: ... | Driving environment: ..." 로 이은 것.

    tier_score 는 그중 최댓값 - 시각화 폴더를 나누는 데만 쓴다.

    파싱에 실패하면 카테고리 없이 Normal 이 되어, 없는 special 을 만들어내는 대신 놓치는
    쪽으로 떨어진다.
    """
    valid = {l["category"] for l in labels if not l["is_normal"]}
    out = {"verdict": "Normal", "categories": [], "parse_ok": False,
           "observation": "", "ego_behavior": "", "unusual_elements": "",
           "group_scores": {}, "score_reason": "",
           "tier_score": None, "time_spans": {}, "time_spans_raw": {},
           "time_seen": {}}
    # 날씨 축은 --weather 를 껐으면 모델이 내지 않는다. 그래도 키는
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


    # 난이도 4축. 값과 근거 문장을 따로 받는다.
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

    # 묶음 점수(0~4). 묶음마다 "<키>_reason" 과 "<키>" 를 받는다
    # (constrained_tier.group_key). 0 도 그대로 담는다 - 이제 0 은 "그 묶음의
    # 특수 카테고리 없음"이라는 기준표의 칸이다.
    #
    # 점수와 카테고리 목록이 어긋나도(이름을 댄 묶음이 0, 대지 않은 묶음이
    # 1 이상) 여기서 고치지 않는다. 고치면 모델의 실제 판단이 결과에서
    # 사라진다 - 어긋남은 채점(evaluate_labels)이 따로 센다.
    reasons = []
    for g in score_groups(labels):
        k = group_key(g)
        if k not in obj:
            continue
        n = _coerce_tier(obj.get(k))
        if n is not None:
            out["group_scores"][g] = n
        why = _flatten_field(obj.get(f"{k}_reason", ""))
        if why:
            reasons.append(f"{g}: {why}")
    out["score_reason"] = " | ".join(reasons)
    out["tier_score"] = tier_score(out["group_scores"])

    # --time-spans 를 켠 실행에서만 나오는 칸. 이름을 댄 장르의 것만 남긴다 -
    # 장르 목록에 없는 이름의 구간은 시각화에서 무엇의 띠인지 설명할 수 없다.
    (out["time_spans"], out["time_spans_raw"],
     out["time_seen"]) = _parse_time_spans(obj.get("time_spans"), valid,
                                           set(out["categories"]))

    # 판정은 오직 "edge-case 요소를 하나라도 나열했는가". 별도 질문을 두지
    # 않는 이유는 20260728 실험 - verdict 를 따로 물어 categories 와 묶으면
    # 모델이 Special 선언에 보수적이라 나열 자체를 포기했다.
    out["verdict"] = "Special" if out["categories"] else "Normal"
    return out


def _seconds(v) -> float | None:
    """구간 끝점 하나를 초(float)로. "7.5", "7.5 s", "0:07.5" 도 받는다."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.match(r"\s*(?:(\d+):)?(\d+(?:\.\d+)?)", v)
        if m:
            return int(m.group(1) or 0) * 60 + float(m.group(2))
    return None


def _score_of(v) -> int | None:
    """구간 점수 0~4. 못 읽거나 범위 밖이면 None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, str):
        m = re.match(r"\s*(\d+)", v)
        v = int(m.group(1)) if m else None
    if isinstance(v, (int, float)) and 0 <= int(round(v)) <= 4:
        return int(round(v))
    return None


def _spans_of(v) -> list[list]:
    """구간 표기 -> [[시작, 끝, 점수|None], ...] (모델이 답한 값 그대로).

    받는 꼴: [s, e, score] / [s, e] / [[s, e, score], ...] / "3-5 s" /
    {"start": s, "end": e, "score": n}. 모델이 스키마를 그대로 따르지 않는
    경우가 많아 흔한 꼴은 모두 받는다. 끝이 시작보다 앞이면 뒤집고, 음수는
    버린다. 1초 단위 정리는 _snap_spans 가 한다.
    """
    out = []
    if isinstance(v, dict):
        v = [v.get("start"), v.get("end"),
             v.get("score", v.get("level", v.get("risk")))]
    if isinstance(v, str):
        nums = re.findall(r"\d+(?::\d+)?(?:\.\d+)?", v)
        v = nums[:2] if len(nums) >= 2 else ([nums[0], nums[0]] if nums else [])
    if not isinstance(v, (list, tuple)) or not v:
        return out
    if any(isinstance(x, (list, tuple, dict, str)) and not _is_num_str(x)
           for x in v):
        for x in v:
            out += _spans_of(x)
        return out
    # 평평한 숫자 목록: [s, e] / [s, e, n] / [s, e, n, s, e, n, ...] / [s, e, s, e]
    n = 3 if (len(v) == 3 or (len(v) % 3 == 0 and len(v) % 2 != 0)
              or (len(v) % 6 == 0)) else 2
    for k in range(0, len(v) - n + 1, n):
        s0, s1 = _seconds(v[k]), _seconds(v[k + 1])
        sc = _score_of(v[k + 2]) if n == 3 else None
        if s0 is None or s1 is None or s0 < 0 or s1 < 0:
            continue
        out.append([min(s0, s1), max(s0, s1), sc])
    return out


def _is_num_str(x) -> bool:
    return isinstance(x, str) and re.fullmatch(r"\s*\d+(?::\d+)?(?:\.\d+)?\s*s?\s*", x) is not None


def _snap_spans(spans: list[list], window=None) -> list[list]:
    """모델이 답한 구간을 1초 단위로 정리한다.

    모델이 구분하는 시점은 1초에 하나다 - 영상의 "<7.2 seconds>" 묶음은
    7.0 초와 7.5 초 프레임이라 [7, 8) 을 덮는다. 그래서 시작은 내리고 끝은
    올려 정수 초에 맞추고, 시작과 끝이 같은 묶음이면 1초로 늘린다.
    그다음 초마다 가장 높은 점수를 남겨(겹치면 높은 쪽) 같은 점수끼리 이어
    붙인다. 점수를 못 읽은 구간(None)은 0 보다 낮게 친다.

    window([처음, 마지막] 초)를 주면 그 밖의 초는 버린다 - 모델이 "보인 시간"
    밖에 점수 구간을 그린 것은 스스로 한 말과 어긋난 것이다. window 도 같은
    규칙(내림/올림, 최소 1초)으로 정수 초에 맞춘다.
    """
    import math
    lo = hi = None
    if window:
        lo = int(math.floor(window[0] + 1e-6))
        hi = max(int(math.ceil(window[1] - 1e-6)), lo + 1)
    per_sec = {}
    for s0, s1, sc in spans:
        a = int(math.floor(s0 + 1e-6))
        b = max(int(math.ceil(s1 - 1e-6)), a + 1)
        if lo is not None:
            a, b = max(a, lo), min(b, hi)
        for t in range(a, b):
            cur = per_sec.get(t, "missing")
            if cur == "missing" or (sc if sc is not None else -1) > \
                    (cur if cur is not None else -1):
                per_sec[t] = sc
    out = []
    for t in sorted(per_sec):
        sc = per_sec[t]
        if out and out[-1][1] == t and out[-1][2] == sc:
            out[-1][1] = t + 1
        else:
            out.append([t, t + 1, sc])
    return out


def _parse_time_spans(raw, valid: set, named: set) -> tuple[dict, dict, dict]:
    """(정리된 구간, 모델이 답한 구간, 보인 시간).

    구간은 {"장르": [[s, e, 점수], ...]}, 보인 시간은 {"장르": [처음, 마지막]}.
    장르마다 {"seen": [...], "stretches": [...]} 꼴(두 단계 질문)과 구간 목록만
    있는 예전 꼴을 모두 받는다. 이름을 댄 장르(named)의 것만 남기고, 이름은
    정식 표기로 바꾼다. 정리된 쪽은 정수 초(_snap_spans, 보인 시간 밖은 버림)
    이고 시각화와 CSV 가 쓴다. 답한 쪽과 보인 시간은 result.json 에 남긴다.
    """
    items = []
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, (list, tuple)):
        # [{"type": "...", "spans": [...]}, ...] 꼴
        for el in raw:
            if isinstance(el, dict):
                name = (el.get("type") or el.get("scenario_type")
                        or el.get("category") or el.get("name"))
                spans = (el.get("spans") or el.get("span") or el.get("time")
                         or el.get("stretches") or el)
                if name:
                    items.append((name, spans))
    given, seen = {}, {}
    for name, spans in items:
        cat = _closest_valid(str(name).strip(), valid)
        if not cat or cat not in named:
            continue
        if isinstance(spans, dict) and ("seen" in spans or "stretches" in spans):
            sw = _spans_of(spans.get("seen"))
            if sw:
                seen[cat] = [sw[0][0], sw[0][1]]
            spans = spans.get("stretches")
            # 보인 시간만 있고 구간이 없으면 그 시간을 점수 없는 구간으로 둔다.
            if not spans and sw:
                spans = [[sw[0][0], sw[0][1]]]
        sp = _spans_of(spans)
        if sp:
            given.setdefault(cat, []).extend(sp)
    snapped = {c: _snap_spans(sp, seen.get(c)) for c, sp in given.items()}
    return snapped, given, seen


def followup_time_spans_question(scored: bool) -> str:
    """--time-spans followup: 1차 답변이 끝난 대화에 덧붙이는 질문.

    1차 프롬프트에 붙이는 Hazard Timing 단계(inline)와 같은 지시를 같은 형식으로
    묻는다 - 두 방식의 차이가 "언제 묻느냐"뿐이어야 결과를 비교할 수 있다.
    1차 답변을 본 뒤에 묻는 것이라 "step 4" 대신 답변의 "scenario_types" 를
    가리킨다. 점수 단계(5단계)를 끈 실행에는 기준표가 없으므로 구간만 묻는다.
    """
    if scored:
        return """For each scenario type you named in "scenario_types", split the time its
element was in the clip into stretches, and score each stretch from 0 to 4 on
the scale of step 5 - how much that element made driving harder for the
ego-vehicle during that stretch. Start a new stretch wherever the score
changes; each stretch lasts at least 1 second, and its length is otherwise up
to you. No stretch scores higher than the score its group got in step 5. Read
the times from the timestamps of the video, in seconds from its start.

Respond with ONLY a JSON object, no other text:
{"time_spans": {"<exact scenario type name>": [[<start seconds>, <end seconds>, <score 0-4>], ...], ...}}"""
    return """For each scenario type you named in "scenario_types", give when its element
was a hazard to the ego-vehicle - the stretch of the clip in which it was in or
coming into the vehicle's way, or otherwise made driving harder. Read the times
from the timestamps of the video, in seconds from its start, and give one or
more [start, end] spans per type. If the element never made driving harder,
give the stretch in which it stood out most.

Respond with ONLY a JSON object, no other text:
{"time_spans": {"<exact scenario type name>": [[<start seconds>, <end seconds>], ...], ...}}"""


def parse_time_spans_answer(text: str, labels: list, named: set) -> tuple[dict, dict, dict]:
    """이어 묻기 답 -> (정리된 구간, 모델이 답한 구간, 보인 시간). 못 읽으면 빈 dict 셋.

    {"time_spans": {...}} 로 감싸지 않고 장르 딕셔너리만 내는 경우도 받는다.
    """
    valid = {l["category"] for l in labels if not l["is_normal"]}
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {}, {}, {}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {}, {}, {}
    raw = obj.get("time_spans", obj) if isinstance(obj, dict) else obj
    return _parse_time_spans(raw, valid, named)


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


def _config_key(args, n_views=1, prompt_sha=None) -> dict:
    """run_config.json 의 "key" - 이 실행의 결과를 결정하는 설정.

    save_run_config 와 이어받기 대조(_check_resume)가 같은 함수를 써야,
    기록한 값과 대조하는 값이 어긋나지 않는다.
    """
    return {
        # 사람이 붙인 실행 설명. 설정만으로는 구분이 안 되는 실험
        # (예: 라벨 파일 내용을 고쳤을 때)을 나중에 알아보게 해준다.
        "memo": args.memo,
        "model": args.model,
        "scene_json": args.scene_json,
        # 카테고리 파일 "내용" 의 지문. 경로가 같아도 templates 나 excludes
        # 를 고치면 프롬프트가 달라진다 - 이어받기가 그것을 알아채야 한다.
        "scene_json_sha": _file_sha(args.scene_json),
        # 프롬프트 틀의 지문. rubric 문장이나 단계 지시를 코드에서 고치면
        # 설정 플래그는 그대로인데 모델이 읽는 글이 바뀐다. 이 저장소에서
        # 가장 자주 바뀌는 것이 그것이라, 이어받기가 옛 프롬프트로 만든 행
        # 뒤에 새 프롬프트 행을 덧붙이는 것을 막으려면 이 값이 필요하다.
        "prompt_sha": prompt_sha,
        # 라벨 "내용" 의 지문. 경로는 그대로인데 내용만 고치는 일이
        # 잦아(실측: 20260818 의 130207 과 134742 는 설정이 완전히
        # 같은데 그 사이 Too Close Person 을 Jaywalking 에 병합해
        # 점수가 달라졌다) 경로만으로는 두 실행을 구분할 수 없다.
        "gt_labels_sha": _file_sha(args.gt_labels),
        # 시각화 패널에 GT 를 함께 그릴 때 쓴 정답 라벨. 채점에 쓰는
        # --labels 와 다른 파일일 수 있어(test_label.json vs _D) 따로 남긴다.
        "gt_labels": args.gt_labels,
        # 프롬프트 방식. 예전 프레임 모드의 qa 와 구분하던 키라 값은 하나뿐이지만,
        # 옛 실행과 --resume 대조가 맞도록 남긴다.
        "prompt_style": "nureasoning",
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
        # ego_ablation / header_style / use_3dbbox / tiers_elements 는 옵션을
        # 지워(2026-10-23) 값이 하나뿐이다. 옛 실행과 --resume 대조가 맞도록
        # 그 값으로 남긴다.
        "ego_ablation": None,
        "header_style": "v1",
        "margin": bool(args.margin),
        "score_categories": bool(args.score_categories),
        "use_3dbbox": False,
        "constrain_tiers": bool(args.constrain_tiers),
        "difficulty": bool(args.difficulty),
        # 키 이름은 difficulty_only 로 둔다 - 기존 실행 폴더의
        # run_config.json 과 같은 이름이어야 evaluate_labels 가 옛 실행도
        # 읽는다. 화면 표기만 weather 로 바뀌었다.
        "difficulty_only": bool(args.difficulty_only),
        "tiers_elements": False,
        "explain_traj": bool(args.explain_traj),
        # 끔 = False, inline = True 로 둔다 - 이 옵션이 on/off 였던 때의
        # run_config 와 같은 값이어야 --resume 대조가 통과한다.
        "time_spans": {None: False, "inline": True}.get(args.time_spans,
                                                         args.time_spans),
        "unknown_categories": bool(args.unknown_categories),
        "group": args.group,
        "num_shards": args.num_shards,
    }


def save_run_config(args, run_dir, n_views=1, prompt_sha=None):
    """이번 실행의 설정을 <run_dir>/run_config.json 에 남긴다.

    argparse 네임스페이스를 통째로 저장하되, 나중에 사람이 먼저 보게 될
    핵심 값(fps/frames/해상도/센서 플래그)은 따로 "key" 에 모아 둔다 -
    evaluate_labels.py 가 그것만 골라 로그 머리에 찍는다.

    왜 결과 폴더에 두는가: 실행 폴더를 나중에 열었을 때 어떤 설정으로 낸
    숫자인지 알 방법이 run.log 를 뒤지는 것뿐이면, 로그가 지워지거나
    --eval-only 로 재채점할 때 근거가 사라진다.

    --resume 으로 이어받는 실행은 원래 설정을 덮어쓰지 않는다 - 대조를
    통과했으니 결과를 결정하는 값은 같고, 덮어쓰면 처음 시작한 시각이
    사라진다. 대신 "resumes" 에 이어받은 기록을 한 줄 덧붙인다.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RUN_CONFIG_NAME
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    if getattr(args, "resume", False) and path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
        cfg.setdefault("resumes", []).append({
            "timestamp": now,
            "num_shards": args.num_shards,
            "memo": args.memo,
        })
        # 옛 실행에는 지문이 없다. 이번에 대조한 값으로 채워 두면 다음
        # 이어받기부터는 프롬프트까지 대조된다.
        key = cfg.setdefault("key", {})
        for k in ("scene_json_sha", "prompt_sha"):
            if key.get(k) is None:
                key[k] = _config_key(args, n_views, prompt_sha)[k]
    else:
        cfg = {
            "timestamp": now,
            "key": _config_key(args, n_views, prompt_sha),
            # 위에 없는 옵션까지 전부. 값이 Path 등이면 문자열로 눕힌다.
            "argv": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                         else str(v))
                     for k, v in vars(args).items()},
        }
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"[info] run config -> {path}")
    return path


# 이어받을 때 원래 실행과 달라도 되는 설정. 결과 행의 내용을 바꾸지 않는다.
#   memo           사람이 붙인 설명
#   num_shards     클립을 몇 갈래로 나누는가 - 어느 클립이 끝났는지는
#                  CSV 로 판정하므로 갈래 수가 바뀌어도 빠지거나 겹치지 않는다
#   gt_labels*     시각화 패널에 GT 를 곁들일 때만 쓴다
RESUME_IGNORED_KEYS = {"memo", "num_shards", "gt_labels", "gt_labels_sha"}


def _prompt_fingerprint(args, labels, category_menu, n_views) -> str:
    """모델이 읽는 프롬프트 틀의 해시.

    클립마다 달라지는 부분(프레임 수, 자차 행동 문장)은 자리표시자로 둔다 -
    지문이 잡아야 하는 것은 rubric, 단계 지시, 카테고리 메뉴처럼 코드와
    설정 파일에서 오는 부분이다.
    """
    import hashlib
    text = build_nureasoning_prompt(
        category_menu, labels,
        behavior_facts="<BEHAVIOR>" if args.use_egomotion else "",
        intro=f"<INTRO views={n_views} fps={args.clip_fps} "
              f"frames={args.clip_max_frames} video={args.clip_video_input}>",
        timeline=args.timeline,
        score_categories=args.score_categories,
        difficulty=args.difficulty,
        difficulty_only=args.difficulty_only,
        explain_traj=args.explain_traj,
        time_spans=(args.time_spans == "inline"))
    # followup 은 1차 프롬프트를 바꾸지 않지만 결과(time_spans 열)를 정하므로
    # 이어 묻는 질문도 지문에 넣는다.
    if args.time_spans == "followup":
        text += "\n<FOLLOWUP>\n" + followup_time_spans_question(
            bool(args.score_categories and score_groups(labels or [])))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _check_resume(args, run_dir, n_views, prompt_sha):
    """이어받아도 되는 실행인지 대조한다. 안 되면 SystemExit.

    결과를 결정하는 설정이 하나라도 다르면, 한 CSV 안에 서로 다른 조건으로
    만든 행이 섞인다. 그 뒤의 집계는 둘을 구분할 방법이 없어 조용히
    오염된다. 그래서 다르면 멈추고 무엇이 다른지 보여준다.

    옛 run_config 에 없는 키(예: prompt_sha 는 나중에 추가됐다)는 대조할 수
    없으므로 경고만 하고 넘어간다. 열 구성이 바뀐 옛 실행은 그와 별개로
    CSV 헤더 대조에서 걸린다.
    """
    run_dir = Path(run_dir)
    path = run_dir / RUN_CONFIG_NAME
    if not path.exists():
        raise SystemExit(f"[error] --resume: {path} 가 없습니다 - 원래 실행의 "
                         f"설정을 알 수 없어 이어받을 수 없습니다.")
    old = json.loads(path.read_text(encoding="utf-8")).get("key", {})
    new = _config_key(args, n_views, prompt_sha)

    def norm(k, v):
        # 경로는 적는 방식(상대/절대)이 달라도 같은 파일이면 같다.
        if k == "scene_json" and v:
            return str(Path(v).resolve())
        return v

    diff, unverified = [], []
    for k, v in new.items():
        if k in RESUME_IGNORED_KEYS:
            continue
        if k not in old or old[k] is None:
            if v is not None:
                unverified.append(k)
            continue
        if norm(k, old[k]) != norm(k, v):
            diff.append(f"    {k:18} 원래 {old[k]!r}\n{'':22} 지금 {v!r}")
    if diff:
        hint = ""
        log = run_dir / "run.log"
        if log.exists():
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("[info] opts"):
                    hint = f"\n  원래 실행의 옵션 ({log.name}):\n    {line}"
                    break
        raise SystemExit(
            "[error] --resume: 원래 실행과 설정이 다릅니다. 같은 옵션으로 "
            "다시 실행하세요.\n" + "\n".join(diff) + hint)
    if unverified:
        print(f"[warn] --resume: 원래 실행 설정에 없는 항목이라 대조하지 못함: "
              f"{', '.join(unverified)}")


def _load_gt_labels(path):
    """test_label.json -> {uuid: {categories, scores}}. 없으면 None.

    categories 는 이름 리스트다(옛 라벨의 {이름: 점수} 딕셔너리도 이름만
    받는다). scores 는 묶음 점수 {묶음 이름: 0~4} - 라벨의 최상위
    "Dynamic object" / "Driving environment" 키에서 읽는다. 아직 안 매긴
    묶음(키 없음, null, -1)은 빠진다.
    """
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    clips = data.get("clips", data)
    out = {}
    for uuid, v in clips.items():
        if not isinstance(v, dict):
            continue
        names = list(v.get("categories") or [])
        scores = {g: v[g] for g in GROUP_RUBRICS
                  if isinstance(v.get(g), int) and not isinstance(v.get(g), bool)
                  and 0 <= v[g] <= 4}
        # 날씨 4축 GT - 시각화 패널이 "Weather score" 로 나란히 적는다.
        w = v.get("weather") if isinstance(v.get("weather"), dict) else {}
        weather = {k: w[k] for k, _ in DIFFICULTY_AXES
                   if isinstance(w.get(k), int) and not isinstance(w.get(k), bool)}
        out[uuid] = {"categories": names, "scores": scores, "weather": weather}
    return out


if __name__ == "__main__":
    import datetime

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-clips", type=int, default=None,
                    help="처리할 클립(uuid) 수 제한 (스모크 테스트용)")
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
                         "8샤드 병렬(run_labeled_8b.sh)이 아니라 run_labeled_27b.sh "
                         "(qwen38 conda 환경)로 돌려야 한다. FP8 판(-FP8)은 "
                         "일부러 목록에서 뺐다 - 멀티 GPU 로 로드하면 출력이 "
                         "깨진다(실측, 텍스트 전용 생성도 실패). GPU 1장에 "
                         "FP8(30.9GB)을 통째로 올릴 방법이 없는 한 못 쓴다.")
    # 후보로 검토했던 타 계열: InternVL3-38B/78B(OCR/세밀한 장면 이해가 강해
    # 표지판·차선 마킹에 유리할 수 있음), LLaVA-OneVision-72B(비디오 벤치마크
    # 강점). 다만 지금 프롬프트는 Qwen 특성에 맞춰 튜닝한 것이라 계열을 바꾸면
    # 재검증이 필요하다 - "프롬프트 문구 실험 (20260728)" 주석 참고.
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
    ap.add_argument("--margin", action="store_true",
                    help="생성 토큰마다 1위-2위 확률차를 재서 CSV 에 남긴다 "
                         "(margin_min/margin_mean/n_close_tokens). 이 값이 "
                         "작은 클립은 의미 없는 프롬프트 섭동(예: 마침표 하나)"
                         "에도 예측이 뒤집힌다 - 실측 10/115 클립이 그랬다. "
                         "A/B 비교 전에 그 차이를 믿어도 되는지 판단할 때 쓴다.")
    ap.add_argument("--single-view", action="store_true",
                    help="front-wide 카메라만 사용한다 (기본은 전방 3뷰). 이미지가 "
                         "6장에서 2장으로 줄어 추론이 빨라지지만, 측면에서만 보이는 "
                         "상황(Jaywalking, Side Street 등)은 놓칠 수 있다. "
                         "프롬프트 문구와 시각화 레이아웃도 함께 바뀐다.")
    ap.add_argument("--scene-json", default=str(SCENE_JSON),
                    help=f"카테고리 정의 JSON (기본 {SCENE_JSON.name}, "
                         "config.py 에서 정함).")
    ap.add_argument("--only-uuids", default=None,
                    help="이 파일에 적힌 uuid(한 줄에 하나)만 처리한다. "
                         "정답 라벨이 있는 클립만 골라 검증할 때 쓴다.")
    ap.add_argument("--resume", action="store_true",
                    help="--out 이 가리키는 실행 폴더를 이어받는다. 폴더의 "
                         "clip_results*.csv 에서 끝난 클립(n_frames > 0)을 "
                         "빼고 나머지만 돌려 같은 파일 뒤에 덧붙인다. 영상을 "
                         "못 읽어 n_frames=0 으로 기록된 클립은 실패로 보고 "
                         "다시 돌린다 - 그 옛 행은 이어받기 전에 백업 후 "
                         "지워야 중복이 생기지 않는다(resume_run.py, 셸 "
                         "스크립트의 --resume 이 자동으로 한다). 설정(모델/"
                         "카테고리/프롬프트/fps 등)이 원래 실행과 다르면 "
                         "거부한다. 클립 모드 전용.")
    ap.add_argument("--resume-prepare", action="store_true",
                    help="--resume 의 준비 단계만 한다: 설정 대조, 열 구성 "
                         "대조, 실패 행 백업 후 삭제. 모델을 올리지 않고 "
                         "끝난다. 여러 샤드가 동시에 정리하면 병합본을 함께 "
                         "다시 쓰게 되므로, 셸 스크립트가 샤드를 띄우기 전에 "
                         "한 번만 부른다.")
    # 예전에는 판정 단위를 (uuid, frame_idx) 로 하는 프레임 모드가 기본이었고
    # 이 옵션으로 클립 모드를 골랐다. 이제 클립 모드만 남아 아무 일도 하지
    # 않지만, 옛 명령(run.log, --resume)이 그대로 돌도록 받아만 둔다.
    ap.add_argument("--clip-mode", action="store_true", help=argparse.SUPPRESS)
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
                    help="묶음 점수가 전부 0/1 인 Special 클립은 시각화에서 "
                         "제외한다 (기본 True). 카테고리가 나열됐지만 모델 "
                         "스스로 '자차에 영향 없음'으로 판단한 경우다. "
                         "CSV/scenario_types 에는 영향 없음 - 시각화 대상만 "
                         "줄인다. --not-save-low=0 으로 끌 수 있다. "
                         "점수 단계가 없는 실행(--weather-only)은 값이 없어 "
                         "이 조건이 성립하지 "
                         "않으므로, 필터는 저절로 무력화된다"
                         "(놓치는 것보다 더 보는 쪽).")
    # 상위 카테고리 점수(5단계)는 항상 묻는다 - 이제 이것이 기본 평가다.
    # 예전의 --no-category-scores(와 옛 이름 --no-safety-tier 등)는 없앴다.
    # 점수 단계가 빠지는 경우는 --weather-only(탐지까지 뺌)뿐이고, 아래에서
    # score_categories 를 끈다.
    ap.set_defaults(score_categories=True)
    ap.add_argument("--viz-per-category", type=int, default=None,
                    metavar="N",
                    help="카테고리마다 최초 N개 클립만 시각화한다. 폴더를 "
                         "score 가 아니라 카테고리 이름으로 나누고, 한 클립이 "
                         "여러 카테고리를 받으면 해당 폴더 전부에 중복 저장한다 "
                         "(<viz-dir>/<Category>/<uuid>/). Normal 클립은 대상이 "
                         "아니다. 샤드마다 따로 세므로 8샤드면 최대 8N 개가 "
                         "나온다.")
    ap.add_argument("--weather", "--difficulty", dest="difficulty",
                    action="store_true",
                    help="날씨 4축(조도/강수/노면/대기가림)을 "
                         "0~4 로 함께 매긴다. prompts.py 의 눈금을 쓰며, "
                         "값과 근거 문장이 CSV 열로 나가고 시각화 패널과 "
                         "aggregate_clip.log 분포에 실린다. 기본 off - "
                         "출력 토큰이 늘어 느려지므로 필요한 실행에서만 켠다.")
    ap.add_argument("--group", default="all",
                    help="다룰 상위 카테고리: all(기본) / dynamic_object / "
                         "driving_environment (scene_category.json 의 special "
                         "scenario 이름을 소문자_밑줄로 쓴 것). 하나를 고르면 "
                         "그 상위 카테고리의 카테고리만 메뉴에 넣고, 점수도 그 "
                         "상위 카테고리 하나만 매긴다. 채점은 GT 를 같은 범위로 "
                         "줄여서 한다.")
    ap.add_argument("--unknown-categories", action="store_true",
                    help="scene_category.json 에서 \"unknown\": true 인 카테고리"
                         "(예상 못한 동적 객체 / 주행 환경)를 카테고리 메뉴에 "
                         "넣는다. 기본 off - "
                         "넣었다 뺐다 하며 효과를 보는 ablation 용이다. 메뉴가 "
                         "길어지면 다른 카테고리 탐지가 떨어진 전례가 있어 "
                         "(5,585 -> 6,389자에서 F1 80.4 -> 74.0) 기본은 뺀다. "
                         "점수는 그 카테고리가 속한 상위 카테고리(Dynamic object "
                         "/ Driving environment) 기준표로 잰다.")
    ap.add_argument("--explain-traj", action="store_true",
                    help="--traj 로 그린 초록 선이 자차의 미래 궤적임을 "
                         "프롬프트에 한 문장으로 알린다. 기본 off - 예전에 "
                         "알려 봤다가 6개 축이 모두 나빠졌다(실측 115클립: "
                         "Verdict F1 90.5->83.0). 그때는 '궤적으로 상호작용을 "
                         "보라'는 판단 지시가 붙어 필터로 읽힌 것이 원인이라, "
                         "여기서는 선이 무엇인지만 말하고 무엇을 하라고는 "
                         "하지 않는다. --traj 없이 주면 의미가 없다.")
    ap.add_argument("--time-spans", nargs="?", const="inline", default=None,
                    choices=["inline", "followup"],
                    help="[클립 모드] 이름을 댄 장르마다 시간 구간을 나눠 "
                         "구간별 위험도 0~4(5단계와 같은 기준표)를 받는다. "
                         "구간 경계는 모델이 정하고(최소 1초), 정수 초로 정리해 "
                         "시각화 영상의 재생 바 위에 점수별 색 띠로 그린다. "
                         "inline(값 생략 시): 1차 답변 JSON 에 칸을 넣는다 - "
                         "원본 프롬프트가 바뀐다. followup: 1차 답변이 끝난 "
                         "대화에 질문을 이어 붙인다 - 1차 프롬프트/결과는 그대로, "
                         "영상 계산을 다시 쓰므로 짧은 질문과 답만 계산한다. "
                         "기본 off.")
    ap.add_argument("--attn-timeline", action="store_true",
                    help="[클립 모드] 1차 생성 중 출력 토큰이 영상의 어느 시점(1초 "
                         "칸)을 봤는지 어텐션을 기록한다. 장르 이름/점수/점수 "
                         "근거 토큰의 시간 곡선을 시각화 영상 재생 바에 히트맵으로 "
                         "그리고, 원본 행렬은 <run>/attn/<uuid>.npz 로 남긴다. "
                         "프롬프트와 출력은 바뀌지 않는다(기록용 계산을 옆에서 "
                         "따로 한다). 기본 off.")
    ap.add_argument("--weather-only", "--difficulty-only",
                    dest="difficulty_only", action="store_true",
                    help="날씨 4축만 추론한다. edge-case 탐지"
                         "(VERDICT/CATEGORIES)와 카테고리 점수를 모두 "
                         "빼고 프롬프트를 날씨 전용으로 다시 쓴다 - 단계만 "
                         "끄면 '조도/날씨를 서술하지 마라'(1단계)와 '조도/날씨를 "
                         "점수로 매겨라'(4단계)가 충돌한다. --weather 를 "
                         "자동으로 켜며, 점수/카테고리 CSV 열은 빈 값이 된다"
                         "(열 구성 자체는 유지해 실행 간 비교가 깨지지 않는다).")
    ap.add_argument("--no-constrain-tiers", dest="constrain_tiers",
                    action="store_false",
                    help=argparse.SUPPRESS)
    # 예전 제약 디코딩(등급 필드를 정수로 강제)을 끄던 옵션. 제약기를 지워
    # 아무 효과가 없지만, run_config key(constrain_tiers)와 옛 명령이 그대로
    # 맞도록 받아만 둔다.
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

    SCENE_JSON = Path(args.scene_json)
    if not SCENE_JSON.exists():
        raise SystemExit(f"[error] scene json not found: {SCENE_JSON}")
    # --viz-normal/--viz-special 은 그 자체가 "시각화하라"는 뜻이므로
    # --clip-viz 를 따로 요구하지 않는다 (빼먹으면 조용히 아무것도 안 나온다).
    if args.viz_normal or args.viz_special:
        args.clip_viz = True

    # --weather-only 는 날씨 4축을 켜고 나머지 판정을 끈다. 사용자가
    # --weather 를 같이 적지 않아도 되게 하고, 점수를 켠 채로 들어와도
    # 프롬프트와 어긋나지 않도록 여기서 한 번에 맞춘다.
    if args.difficulty_only:
        args.difficulty = True
        args.score_categories = False
        print("[info] --weather-only: 날씨 4축만 추론한다 "
              "(verdict/categories/scores off)")

    # --time-spans: 영상 입력을 끄면 타임스탬프가 박히지 않아 모델이 장 번호로
    # 시각을 어림해야 하므로 경고만 한다. --weather-only 는 장르를 묻지 않아 끈다.
    if args.time_spans:
        if args.difficulty_only:
            args.time_spans = None
            print("[info] --weather-only: 장르를 묻지 않으므로 --time-spans 끔")
        elif args.time_spans == "followup":
            print("[info] --time-spans followup: 1차 답변 뒤 같은 대화에 장르별 "
                  "구간 위험도를 이어 묻는다 (1차 프롬프트 그대로)")
        else:
            print("[info] --time-spans inline: 장르별 구간 위험도를 1차 답변에 "
                  "함께 받는다 (원본 프롬프트 변경)")
            if not args.clip_video_input:
                print("[warn] --clip-no-video-input 이면 프레임에 타임스탬프가 "
                      "없어 구간이 부정확할 수 있다")

    # --resume 은 이어받을 폴더를 --out 으로 받는다. 없으면 아래에서 새 실행
    # 폴더가 만들어져, 이어받는 대신 처음부터 도는 실행이 조용히 생긴다.
    if (args.resume or args.resume_prepare) and args.out is None:
        raise SystemExit("[error] --resume 은 --out <이어받을 폴더>/"
                         "clip_results_shard_N.csv 가 필요합니다")
    if args.resume_prepare:
        args.resume = True

    if args.out is None:
        run_dir = ROOT / "results" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        args.out = str(run_dir / "results.csv")
        if args.viz_dir is None:
            args.viz_dir = str(run_dir)
    elif args.viz_dir is None:
        args.viz_dir = str(Path(args.out).parent)

    labels = load_labels(SCENE_JSON)
    # Unknown 카테고리(scene_category.json 에서 "unknown": true - 예상 못한
    # 동적 객체 / 주행 환경)는 기본으로 메뉴에서 뺀다. 넣었다 뺐다 하며 효과를
    # 보는 ablation 용이다. 빼는 위치가 여기여야 메뉴, 기준표 머리의 카테고리
    # 목록, 파싱의 유효 이름이 모두 함께 따라간다.
    if not args.unknown_categories:
        labels = [l for l in labels if not l.get("unknown")]
    # --group: 상위 카테고리 하나만 다룬다(all / dynamic_object /
    # driving_environment). 다른 상위 카테고리의 카테고리는 메뉴에서 빠지고,
    # 5단계도 그 상위 카테고리 점수 하나만 묻는다 - 아직 기준표를 다 채우지
    # 않은 상위 카테고리를 빼고 돌릴 수 있다. 채점(evaluate_labels)도
    # run_config 의 group 을 보고 GT 를 같은 범위로 줄인다.
    if args.group != "all":
        _known = {group_key(l["scenario"]) for l in labels if not l["is_normal"]}
        if args.group not in _known:
            raise SystemExit(f"[error] --group {args.group}: {SCENE_JSON.name} "
                             f"에 없는 상위 카테고리입니다 (가능: all, "
                             f"{', '.join(sorted(_known))})")
        labels = [l for l in labels
                  if l["is_normal"] or group_key(l["scenario"]) == args.group]
    # 시각화 패널이 카테고리마다 쓰는 rubric 의 상한(분모)을 알아야 하므로,
    # 카테고리->묶음 표를 여기서 한 번 넘겨 둔다.
    VZ.load_scenario_map(SCENE_JSON)
    n_special = sum(1 for l in labels if not l["is_normal"])
    category_menu = build_category_menu(labels, notes=load_menu_notes(SCENE_JSON))
    print(f"[info] labels: {SCENE_JSON.name} - {n_special} special categories")
    print(f"[info] sensor facts: egomotion={'ON' if args.use_egomotion else 'OFF'}")
    _views = views_for(args.single_view)
    print(f"[info] camera views: {len(_views)} "
          f"({'front-wide only' if args.single_view else 'front 3-view'})")

    prompt_sha = _prompt_fingerprint(args, labels, category_menu, len(_views))

    # --resume: 원래 실행과 대조한 뒤 끝난 클립 목록을 얻는다.
    #
    # 순서가 중요하다. (1) 설정 대조와 (2) CSV 열 구성 대조를 둘 다 통과한
    # 뒤에야 (3) 실패 행을 지운다 - 이어받을 수 없는 실행을 발견하기 전에
    # 파일부터 고쳐 두면 되돌리기 번거롭다. 전부 클립 목록 만들기 전에 한다.
    # NAS 는 목록에 한 시간이 걸릴 수 있어, 거부될 실행이 그만큼 기다리게
    # 하지 않는다.
    resume_done = None
    if args.resume:
        import resume_run as RR
        # 이 import 는 torch 를 올리지만 어차피 아래에서 올린다. 열 구성의
        # 진실 공급원이 qwen_runner 라 여기서 가져와야 한다.
        from qwen_runner import CLIP_CSV_COLUMNS
        run_dir = Path(args.out).parent
        if not RR.result_files(run_dir):
            raise SystemExit(f"[error] --resume: {run_dir} 에 이어받을 "
                             f"{RR.RESULT_GLOB} 가 없습니다")
        _check_resume(args, run_dir, len(_views), prompt_sha)
        try:
            if args.resume_prepare:
                info = RR.prepare(run_dir, CLIP_CSV_COLUMNS)
                print(f"[resume] 준비 완료 - 끝난 클립 {len(info['done']):,}개")
                raise SystemExit(0)
            info = RR.scan(run_dir, CLIP_CSV_COLUMNS)
            if info["n_drop"]:
                # 샤드가 여럿이면 여기서 정리하면 안 된다 - 같은 병합본을
                # 여러 프로세스가 동시에 다시 쓰게 된다. 정리하지 않고
                # 진행하면 실패 행과 새 행이 같은 uuid 로 겹친다.
                if args.num_shards > 1:
                    raise SystemExit(
                        f"[error] --resume: 지워야 할 실패 행이 "
                        f"{info['n_drop']:,}개 남아 있습니다. 샤드를 띄우기 "
                        f"전에 한 번만 정리하세요:\n"
                        f"  python resume_run.py --run-dir {run_dir} --apply")
                info = RR.prepare(run_dir, CLIP_CSV_COLUMNS)
        except RR.ResumeError as e:
            raise SystemExit(f"[error] --resume: {e}")
        resume_done = info["done"]
        print(f"[resume] {run_dir}: 끝난 클립 {len(resume_done):,}개는 건너뛴다")

    # 특정 클립만 처리 (검증용). 라벨된 uuid 는 데이터셋 전체에 흩어져 있어서
    # --limit-clips 로는 못 뽑는다. 목록에 있지만 데이터셋에 없는 uuid 는
    # 조용히 빠지면 원인을 못 찾으므로 개수를 알린다.
    #
    # 목록을 먼저 처리한다. list_scene_uuids() 는 NAS zip 에서 전체 인덱스
    # (3145개 zip, ~1시간)를 만드는데, uuid 를 이미 아는 마당에 그걸 기다릴
    # 이유가 없다. locate_uuid 로 하나씩 직접 찾고, 전부 찾으면 전체 인덱스는
    # 건너뛴다. 하나라도 못 찾을 때만 원래 경로로 내려가 전체와 대조한다
    # (그래야 "데이터셋에 없음" 과 "아직 인덱싱 안 됨" 이 구분된다).
    if args.only_uuids:
        wanted = [l.strip() for l in
                  Path(args.only_uuids).read_text(encoding="utf-8").splitlines()
                  if l.strip()]
        found, missing = [], []
        if CLIP_SOURCE is not None and hasattr(CLIP_SOURCE, "locate_uuid"):
            for u in wanted:
                (found if CLIP_SOURCE.locate_uuid(FRONT_VIEWS[0], u)
                 else missing).append(u)
        else:
            missing = list(wanted)

        if missing:
            have = set(list_scene_uuids(args.limit_clips))
            found = [u for u in wanted if u in have or u in found]
        uuids = found
        n_missing = len(wanted) - len(uuids)
        print(f"[info] --only-uuids: {len(uuids)}/{len(wanted)} found"
              + (f"  ({n_missing} not in dataset)" if n_missing else ""))
        if not uuids:
            raise SystemExit("[error] none of the requested uuids exist")
    else:
        uuids = list_scene_uuids(args.limit_clips)

    if args.num_shards > 1:
        uuids = uuids[args.shard_id::args.num_shards]
        print(f"[info] shard {args.shard_id}/{args.num_shards}: "
              f"{len(uuids)} clips")
    # 끝난 클립은 샤드로 나눈 '뒤에' 뺀다. 먼저 빼고 나누면, 샤드마다
    # 끝난 목록을 읽은 시점이 조금씩 다를 때(먼저 시작한 샤드가 이미
    # 행을 덧붙인 뒤) 나머지 샤드의 몫이 밀려 어떤 클립은 두 번, 어떤
    # 클립은 한 번도 안 돌게 된다. 나눈 뒤에 빼면 샤드끼리 몫이 겹치지
    # 않으므로 그 시점 차이가 결과를 바꾸지 못한다.
    if resume_done is not None:
        n_all = len(uuids)
        uuids = [u for u in uuids if u not in resume_done]
        print(f"[resume] {n_all - len(uuids):,}개 끝남 -> "
              f"남은 {len(uuids):,}개를 돌린다")
        if not uuids:
            print("[resume] 이 샤드에 남은 클립이 없습니다")
    n_img = args.clip_max_frames * len(_views)
    print(f"[info] clip mode: {len(uuids)} clips, {args.clip_fps} fps, "
          f"max {args.clip_max_frames} frames/view -> {n_img} images/clip "
          f"@ long side {args.clip_long_side}px")
    # 실행 설정을 폴더에 남긴다 - 몇 주 뒤 결과만 보고 "이건 몇 fps 였지"
    # 를 되짚을 방법이 로그 뒤지기밖에 없으면 A/B 비교를 신뢰할 수 없다.
    # 샤드 0 만 쓴다 (8개가 같은 파일에 동시에 쓰면 깨진다).
    if args.shard_id == 0:
        save_run_config(args, Path(args.out).parent, n_views=len(_views),
                        prompt_sha=prompt_sha)
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
    # 이어받았는데 이 샤드 몫이 전부 끝났으면 모델을 올리지 않는다 -
    # 8B 도 올리는 데 수십 초가 걸리고, 할 일 없이 GPU 만 잡는다.
    if resume_done is not None and not uuids:
        raise SystemExit(0)
    from qwen_runner import run_clip_inference
    # 이 import 가 edge_case_mining 사본을 만든다 - 프레임 소스를 옮겨 심는다
    _mirror_clip_source()
    run_clip_inference(uuids, labels, category_menu,
                       model_id=args.model, out_csv=args.out,
                       use_egomotion=args.use_egomotion,
                       want_margin=args.margin,
                       score_categories=args.score_categories,
                       difficulty=args.difficulty,
                       difficulty_only=args.difficulty_only,
                       explain_traj=args.explain_traj,
                       time_spans=args.time_spans,
                       attn_timeline=args.attn_timeline,
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
                       resume=args.resume,
                       **({"viz_width": (args.clip_viz_width or None)}
                          if args.clip_viz_width is not None else {}))

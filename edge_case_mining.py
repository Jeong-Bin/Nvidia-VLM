#!/usr/bin/env python3
"""
VLM 기반 반자동 edge-case (long-tail) mining.

판정 단위는 클립(uuid) 전체가 아니라 "클립 내 특정 순간"(uuid, frame_idx)
이다. 각 20초 클립에서 균등하게 timestamps_per_clip 개의 순간을 뽑고, 매
순간마다 전방 3뷰(front_wide, cross_left, cross_right)의 같은 프레임을
함께 Qwen2.5-VL 에 보여줘 2단계로 분류한다:

  1단계 (caption): 그 순간을 한 문장으로 서술
  2단계 (match)  : 그 문장이 scene_category.json 의 어느 (scenario, category)와
                    가장 잘 맞는지 top-3 후보를 뽑는다 (normal 포함, 없으면 OOD)

500 클립 x 10 timestamps/clip (기본값) = 5,000 개 판정 단위.

Usage:
  # 스모크 테스트 (클립 2개 x 10 timestamp = 20 단위)
  python edge_case_mining.py --limit-clips 2

  # 전체 (500 클립 x 10 timestamp = 5,000 단위)
  python edge_case_mining.py
"""
import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
CAMERA_DIR = ROOT / "pav_sample" / "camera"
SCENE_JSON = ROOT / "scene_category.json"

FRONT_VIEWS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
]


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


def build_label_menu(labels, example_source: str = "synonyms", num_examples: int | None = None):
    """모델에게 보여줄 (scenario, category) 목록 텍스트를 만든다 (2단계 매칭용).

    example_source="synonyms"(기본값, 20260723 방식): 각 카테고리를 synonym 키워드로
    제시. prompt_templates 를 쓰면 normal 카테고리의 풍부한 "정지/횡단" 예시가
    special 카테고리를 눌러버려(예: "turkey crossing" 을 Animal 대신 Normal Stop 으로)
    다양성이 붕괴되는 것을 확인했으므로 기본은 synonyms.
    synonyms 가 없는 카테고리(normal)는 자동으로 prompt_templates 로 대체된다.
    num_examples=None 이면 전체 사용.
    """
    lines = []
    for i, lab in enumerate(labels, 1):
        examples = ", ".join(f'"{e}"' for e in _examples_for(lab, example_source, num_examples))
        lines.append(
            f'{i}. scenario="{lab["scenario"]}", category="{lab["category"]}" '
            f'(e.g. {examples})'
        )
    return "\n".join(lines)


def build_caption_hint(labels, example_source: str = "synonyms", num_examples: int = 1):
    """1단계 캡션 프롬프트에 넣을 special 카테고리 체크리스트.

    카테고리당 example_source 에서 앞 num_examples 개만 예시로 넣는다.
    기본값(synonyms, 1개)은 초기 20260723 실행 방식으로, 예시를 최소화해 모델의
    관찰 자유도를 높인다 - 예시를 전부 나열하면 오히려 판별이 경직되어 다양성이
    떨어진다는 관찰에 따른 것.
    example_source="prompt_templates": 유의미한 상황을 서술하는 문장으로 힌트를 줌
    (예: "An electric scooter is popping out"), "존재 vs 상황" 을 구분시키고 싶을 때.
    캡션이 이 예시를 그대로 베끼지 않도록 프롬프트 본문에서 별도로 지시한다.
    normal 은 제외 (특이사항이 없을 때의 기본값이라 체크리스트에 부적절).
    """
    lines = []
    for i, lab in enumerate((l for l in labels if not l["is_normal"]), 1):
        examples = ", ".join(
            f'"{e}"' for e in _examples_for(lab, example_source, num_examples)
        )
        lines.append(f'{i}. {lab["category"]} (e.g. {examples})')
    return "\n".join(lines)


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


def _read_frames_at(mp4_path: str, indices, max_long_side: int = 896):
    """비디오에서 여러 frame index 를 읽어 {idx: PIL.Image} 로 반환 (한 번의 open).

    실패한 인덱스는 결과 dict 에서 빠진다.
    """
    cap = cv2.VideoCapture(mp4_path)
    out = {}
    for idx in sorted(set(int(i) for i in indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if ok:
            out[idx] = _resize_to_pil(bgr, max_long_side)
    cap.release()
    return out


def clip_path(view: str, uuid: str) -> Path:
    return CAMERA_DIR / view / f"{uuid}.{view}.mp4"


def clip_frame_count(uuid: str) -> int:
    """front_wide 뷰 기준 클립의 총 프레임 수 (3뷰 모두 동일함이 확인됨)."""
    cap = cv2.VideoCapture(str(clip_path(FRONT_VIEWS[0], uuid)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    return total


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
                       temporal_delta: int = TEMPORAL_DELTA):
    """판정 단위(uuid, frame_idx) 하나에 대해 3뷰의 (직전, 현재) 프레임을 읽어
    {"prev": {view: PIL.Image}, "cur": {view: PIL.Image}} 로 반환.

    - cur  = frame_idx 시점
    - prev = frame_idx - temporal_delta 시점 (< 0 이면 프레임 0 으로 대체)
    시각화는 cur 만, 모델 1단계 입력은 prev+cur 을 순서대로 사용.
    """
    prev_idx = max(0, frame_idx - temporal_delta)
    prev, cur = {}, {}
    for view in FRONT_VIEWS:
        p = str(clip_path(view, uuid))
        frames = _read_frames_at(p, [prev_idx, frame_idx], max_long_side=max_long_side)
        prev[view] = frames.get(prev_idx)
        cur[view] = frames.get(frame_idx)
    return {"prev": prev, "cur": cur, "prev_idx": prev_idx, "cur_idx": frame_idx}


# ---------------------------------------------------------------------------
# 프롬프트 - 1단계: 캡션
#
# caption_hint 는 special 24개 카테고리의 참고 키워드/예시 목록이다 (기본값:
# 카테고리당 synonym 1개 - 초기 20260723 방식). 예시는 "이런 종류의 것들을
# 놓치지 말고 관찰하라"는 최소 힌트일 뿐이며, 캡션이 예시를 그대로 베끼지 않고
# 실제 관찰을 자유롭게 서술하도록 지시한다 (예시를 과하게 나열하면 오히려
# 판별이 경직되어 다양성이 떨어지는 것을 확인).
# ---------------------------------------------------------------------------
def build_caption_prompt(caption_hint: str) -> str:
    return f"""You are an autonomous-driving scene analyst. You are shown SIX images
in order: the FIRST three are from about 1 second EARLIER, the LAST three are
the CURRENT moment. Each group of three is synchronized camera views
(front-wide, cross-left, cross-right) of the same vehicle.

By comparing the earlier frames to the current ones, judge the MOTION of the
ego-vehicle and of nearby agents (e.g. moving vs stopped, and in which
direction). Then describe the CURRENT moment in exactly ONE concise sentence,
stating the motion (moving / stopped) and anything unusual, hazardous, or
noteworthy for autonomous driving.

When checking for anything noteworthy, keep in mind (non-exhaustive) types of
special situations like these - do not just copy a category name, describe
what you actually see:
{caption_hint}

If it is ordinary driving with nothing noteworthy, describe it plainly as
such, in your own words.

Respond with ONLY the single sentence, no extra text, no quotes."""


# ---------------------------------------------------------------------------
# 프롬프트 - 2단계: caption -> top-3 (scenario, category) 매칭
# ---------------------------------------------------------------------------
def build_match_prompt(caption: str, label_menu: str) -> str:
    return f"""You are matching a one-sentence driving-scene description to a
predefined taxonomy of driving scene categories, including both ordinary
("normal") categories and special "edge-case / long-tail" categories.

Scene description:
"{caption}"

Candidate (scenario, category) pairs:
{label_menu}

Task: pick up to 3 candidates that plausibly match the scene description,
ranked best match first. Only include a candidate if it is genuinely
plausible - do not pad the list to reach 3. If nothing plausibly matches
(neither a special category nor a normal one), return an empty list (OOD).

Respond with ONLY a compact JSON object, no extra text:
{{"matches": [
   {{"scenario": "<scenario name>", "category": "<category name>", "confidence": <float 0-1>}},
   ...
 ]}}
Return between 0 and 3 items in "matches", ordered best first."""


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


def parse_match_output(text: str, labels: list):
    """2단계 모델 출력에서 top-3 매치를 파싱.

    반환: [{"scenario":..., "category":..., "confidence":..., "is_normal":...}, ...]
    (최대 3개, 없으면 [] = OOD)
    """
    valid_cats = {l["category"] for l in labels}
    cat_to_label = {l["category"]: l for l in labels}

    m = re.search(r"\{.*\}", text, re.DOTALL)
    matches = []
    if m:
        try:
            obj = json.loads(m.group(0))
            for item in obj.get("matches", [])[:3]:
                cat = _closest_valid(str(item.get("category", "")).strip(), valid_cats)
                if cat is None:
                    continue
                conf = float(item.get("confidence", 0.0))
                lab = cat_to_label[cat]
                matches.append(
                    {"scenario": lab["scenario"], "category": cat,
                     "confidence": conf, "is_normal": lab["is_normal"]}
                )
        except Exception:
            pass
    return matches


def clip_uuid(mp4_path: str) -> str:
    return Path(mp4_path).name.split(".")[0]


if __name__ == "__main__":
    import datetime

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-clips", type=int, default=None,
                    help="처리할 클립(uuid) 수 제한 (스모크 테스트용)")
    ap.add_argument("--timestamps-per-clip", type=int, default=10,
                    help="클립당 샘플링할 timestamp(프레임) 수")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
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
    ap.add_argument("--caption-example-source", choices=["synonyms", "prompt_templates"],
                    default="synonyms",
                    help="1단계 캡션 힌트에 쓸 예시 소스. synonyms(기본값, 초기 20260723 방식)는 "
                         "단순 객체 키워드; prompt_templates 는 유의미한 상황 서술 문장. "
                         "2단계(매칭) 는 항상 prompt_templates 로 고정됨.")
    ap.add_argument("--caption-num-examples", type=int, default=1,
                    help="1단계 캡션 힌트에서 카테고리당 넣을 예시 개수 (기본값 1). "
                         "예시를 많이 넣을수록 프롬프트가 길어지고 오히려 판별이 경직될 수 있음.")
    args = ap.parse_args()

    if args.out is None:
        run_dir = ROOT / "results" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        args.out = str(run_dir / "results.csv")
        if args.viz_dir is None:
            args.viz_dir = str(run_dir)
    elif args.viz_dir is None:
        args.viz_dir = str(Path(args.out).parent)

    labels = load_labels(SCENE_JSON)
    n_normal = sum(1 for l in labels if l["is_normal"])
    n_special = len(labels) - n_normal
    label_menu = build_label_menu(labels)
    caption_prompt = build_caption_prompt(
        build_caption_hint(labels, example_source=args.caption_example_source,
                           num_examples=args.caption_num_examples)
    )
    print(f"[info] loaded {n_special} special categories, {n_normal} normal categories")
    print(f"[info] caption hint: {args.caption_num_examples} example(s) per category "
          f"from {args.caption_example_source}")

    uuids = list_scene_uuids(args.limit_clips)
    print(f"[info] clips: {len(uuids)}  x  {args.timestamps_per_clip} timestamps/clip")

    units = list_frame_units(uuids, args.timestamps_per_clip)
    print(f"[info] total (clip, timestamp) units: {len(units)}")

    if args.num_shards > 1:
        units = units[args.shard_id::args.num_shards]
        print(f"[info] shard {args.shard_id}/{args.num_shards}: {len(units)} units")

    if args.dry_run:
        for uuid, idx in units[:3]:
            uf = sample_unit_frames(uuid, idx)
            prev_ok = {v: (im.size if im else None) for v, im in uf["prev"].items()}
            cur_ok = {v: (im.size if im else None) for v, im in uf["cur"].items()}
            print(f"  {uuid} cur@{uf['cur_idx']} prev@{uf['prev_idx']}")
            print(f"     prev: {prev_ok}")
            print(f"     cur : {cur_ok}")
        print("[dry-run] done")
        raise SystemExit(0)

    from qwen_runner import run_inference
    run_inference(units, labels, label_menu, caption_prompt,
                  model_id=args.model, out_csv=args.out, viz_dir=args.viz_dir)

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
  python edge_case_mining.py --use-egomotion --use-obstacle

  # 전체
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
SCENE_JSON = ROOT / "scene_category_B.json"

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
def build_vlm_prompt(category_menu: str, sensor_facts: str = "") -> str:
    """6장 이미지 + 카테고리 메뉴 -> JSON 한 덩어리를 요구하는 프롬프트.

    sensor_facts: egomotion/obstacle 라벨에서 뽑은 사실 문구(옵션). 비어 있으면
    해당 블록 자체가 빠진다.
    """
    fact_block = f"""
KNOWN FACTS at the CURRENT moment (from vehicle sensors - ground truth, trust
these over your own guess from the images):
{sensor_facts}
""" if sensor_facts else ""

    return f"""You are an autonomous-driving scene analyst. You are shown SIX images
in order: the FIRST three are from about 1 second EARLIER, the LAST three are
the CURRENT moment. Each group of three is synchronized camera views
(front-wide, cross-left, cross-right) of the same vehicle.
{fact_block}
Compare the earlier frames to the current ones to judge motion, then answer THREE questions about the CURRENT moment.

Q1. Ordinary driving, or a SPECIAL edge-case situation that interfere with driving? "Normal" or "Special".
Q2. Which categories below are present in the CURRENT scene? List EVERY one that applies. Copy the category names EXACTLY as written:

{category_menu}

Q3. Is anything actually blocking or intruding into the ego-vehicle's driving path right now? "Yes" or "No".

The three questions are INDEPENDENT - answer each one on its own:
- Answer Q2 in full even when the verdict is "Normal".
- Q3 does NOT filter Q2. Still list a category in Q2 even if it sits off to the
  side and the answer to Q3 is "No".

Respond with ONLY a JSON object, no other text:
{{"verdict": "Normal" or "Special",
 "categories": ["<exact category name>", ...],
 "blocks_path": "Yes" or "No",
 "evidence": "<one short phrase describing what you actually see>"}}"""


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


def parse_vlm_output(text: str, labels: list) -> dict:
    """모델 출력 -> {"verdict","categories","blocks_path","evidence","parse_ok"}.

    categories 는 scene_category_B.json 에 실제로 있는 special 카테고리명만
    남긴다(대소문자 차이는 흡수). 모델이 만들어낸 이름은 버린다.
    JSON 파싱에 실패하면 parse_ok=False 로 표시하고 verdict 는 Normal,
    blocks_path 는 False 로 둔다 (없는 special 을 만들어내는 것보다 놓치는 쪽이
    사후 검수에 안전).
    """
    valid = {l["category"] for l in labels if not l["is_normal"]}
    out = {"verdict": "Normal", "categories": [], "blocks_path": False,
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


def blocking_dir(result: dict) -> str:
    """Q3 답에 따른 최상위 폴더명."""
    return "blocking_yes" if result["blocks_path"] else "blocking_no"


def viz_targets(result: dict) -> list[str]:
    """이 판정 단위의 시각화를 저장할 상대 경로들.

    blocking_{yes,no}/<category>/ 아래에 카테고리별로 나눠 담는다. 멀티라벨이면
    해당하는 모든 카테고리 폴더에 같은 결과를 중복 저장한다(검수할 때 카테고리
    단위로 훑을 수 있어야 하므로).
    카테고리가 하나도 없으면 저장하지 않는다 - verdict 와 무관하게, 볼 것이
    없는 장면이기 때문.
    """
    if not result["categories"]:
        return []
    top = blocking_dir(result)
    return [f"{top}/{category_slug(c)}" for c in result["categories"]]


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
                             "Qwen/Qwen3-VL-32B-Instruct"],
                    help="사용할 VLM. Qwen3-VL 은 Qwen2.5-VL 과 동일한 "
                         "Qwen2_5_VLForConditionalGeneration 아키텍처가 아니므로 "
                         "qwen_runner.py 가 model_id 를 보고 알맞은 모델/프로세서 "
                         "클래스를 자동으로 고른다.")
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
    ap.add_argument("--use-obstacle", action="store_true",
                    help="obstacle.offline 3D 라벨의 주변 객체 요약을 프롬프트에 넣는다 (기본 off).")
    ap.add_argument("--example-source", choices=["synonyms", "prompt_templates"],
                    default="synonyms",
                    help="카테고리 예시 소스. synonyms(기본)는 짧은 키워드, "
                         "prompt_templates 는 상황 서술 문장.")
    ap.add_argument("--num-examples", type=int, default=2,
                    help="카테고리당 프롬프트에 넣을 예시 개수 (기본 2). "
                         "많이 넣을수록 프롬프트가 길어지고 판별이 경직될 수 있음.")
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
    n_special = sum(1 for l in labels if not l["is_normal"])
    category_menu = build_category_menu(
        labels, example_source=args.example_source, num_examples=args.num_examples)
    print(f"[info] labels: {SCENE_JSON.name} - {n_special} special categories")
    print(f"[info] category menu: {args.num_examples} example(s) per category "
          f"from {args.example_source}")
    print(f"[info] sensor facts: egomotion={'ON' if args.use_egomotion else 'OFF'}, "
          f"obstacle={'ON' if args.use_obstacle else 'OFF'}")

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
    run_inference(units, labels, category_menu,
                  model_id=args.model, out_csv=args.out, viz_dir=args.viz_dir,
                  use_egomotion=args.use_egomotion, use_obstacle=args.use_obstacle)

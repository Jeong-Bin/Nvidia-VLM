#!/usr/bin/env python3
"""Qwen VLM 단일 호출 추론 러너.

--model 로 Qwen2.5-VL-7B, Qwen3-VL-8B/32B 중 선택 가능 (edge_case_mining.py 의
choices 참고). Qwen3-VL 은 Qwen2.5-VL 과 다른 모델 클래스
(Qwen3VLForConditionalGeneration vs Qwen2_5_VLForConditionalGeneration) 를 쓰므로
AutoModelForImageTextToText / AutoProcessor 로 로드해 model_id 에 따라 알맞은
구현이 자동으로 선택되게 한다 - 모델을 바꿔도 이 파일을 고칠 필요가 없다.

판정 단위: (uuid, frame_idx) - 클립 내 특정 순간의 3뷰 프레임 세트.

한 번의 호출로 직전 3뷰 + 현재 3뷰(6장)를 보여주고 JSON 을 받는다:
  {"verdict": "Normal"|"Special", "categories": [...],
   "blocks_path": true|false, "evidence": "..."}

시각화는 "경로를 막는가"(Q3) x "어느 카테고리인가"(Q2) 로 나눠 담는다:

  <viz_dir>/blocking_yes/<Category>/<uuid>_f<idx>/{card.png, result.json}
  <viz_dir>/blocking_no/<Category>/<uuid>_f<idx>/{card.png, result.json}

멀티라벨이면 해당하는 모든 카테고리 폴더에 같은 결과를 중복 저장한다.
카테고리가 하나도 없는 판정 단위는 저장하지 않는다.
"""
import csv
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          LogitsProcessorList)
from transformers.video_utils import VideoMetadata

from edge_case_mining import (
    sample_unit_frames, views_for, clip_path,
    build_vlm_prompt, parse_vlm_output, unit_name, viz_targets, blocking_dir,
    build_nureasoning_prompt, parse_nureasoning_output,
    sample_clip_frames, clip_intro,
    CLIP_FPS, CLIP_MAX_FRAMES, CLIP_MAX_LONG_SIDE,
)
from egomotion import (
    ego_state, describe_ego, ego_behavior_change, describe_behavior,
    ego_clip_behavior, describe_clip_behavior,
    ego_clip_track, describe_clip_track,
)
from obstacle import obstacle_summary, describe_obstacles, path_intrusion
from visualize import render_scene_card
from visualize_clip import render_clip_result, VIZ_WIDTH
from constrained_tier import make_tier_processor, tier_label, score_dirname
from prompts import DIFFICULTY_AXES


# 27B(Qwen3.8, architectures=Qwen3_5ForConditionalGeneration)는 bf16 55.6GB /
# FP8 30.9GB 라 GPU 1장(24GB)에 안 들어간다. 8B 는 GPU 1장에 모델 전체를
# 올려 8샤드를 각자 다른 GPU 에서 동시에 돌리는데(CUDA_VISIBLE_DEVICES 로
# 프로세스마다 GPU 1장만 보여줌), 27B 는 애초에 여러 GPU 에 걸쳐야 해서
# 그 구조를 못 쓴다 - device_map="auto" 로 한 프로세스가 GPU 여러 장을
# 동시에 물어야 한다. 그래서 model_id 로 자동 판별해 로딩 방식을 가른다.
#
# FP8(Qwen/Qwen3.8-27B-FP8)은 목록에 없다 - 여기 넣지 말 것.
# 실측(20260824): 멀티 GPU 로 로드하면 transformers 가 "DeepGEMM 대신
# Triton/grouped_mm 경로로 우회한다" 고 경고하는데, 실제로 그 경로가
# 손상된 값을 낸다. 이미지와 무관하게 "2+2는?" 같은 순수 텍스트 질문도
# 의미 없는 토큰을 반복하다 무한 루프로 깨졌다(비전 문제가 아님을 순수
# 텍스트 생성으로 격리 확인). bf16 원본은 같은 멀티 GPU 배치에서 정상
# 응답했으므로 FP8 양자화 자체가 원인이다. GPU 1장에 FP8 을 통째로 올릴
# 방법이 없는 한(24GB < 30.9GB) 이 경로는 막혀 있다 - bf16 만 쓴다.
MULTI_GPU_MODELS = ("Qwen/Qwen3.8-27B",)

# chat_template.jinja 를 보면 enable_thinking 이 undefined 거나 true 면
# 기본 reasoning_effort="xhigh" 로 사고 과정을 먼저 뱉는다 - 편집 방식 클립
# 분류에는 그 사고 분량이 낭비다. 실측(20260824, 27B): thinking=True 로
# 두면 프레임을 한 장씩 짚어가는 800단어 넘는 서술을 쏟아내다
# NUR_MAX_NEW_TOKENS=800 에 걸려 JSON 을 한 글자도 못 내고 잘렸다
# (parse_ok=False). thinking=False 로 끄면 같은 프롬프트로 곧장 깨끗한
# JSON 을 낸다. 그래서 THINKING_MODELS 는 항상 빈 튜플이다 - MULTI_GPU_MODELS
# 와 절대 같은 값으로 두지 말 것(다시 27B 를 thinking 모델로 자동 분류하는
# 실수를 반복하게 된다). 8B(Qwen3-VL) 의 템플릿에는 이 인자가 아예 없어
# 넘겨도 조용히 무시된다.
THINKING_MODELS = ()


def _safe_dirname(name: str) -> str:
    """카테고리 이름을 폴더명으로 쓸 수 있게 다듬는다.

    현재 분류 체계에는 경로에 위험한 문자가 없지만(공백만 있다), 체계는
    사람이 손으로 고치는 json 이라 슬래시가 들어올 수 있다 - 그때 엉뚱한
    상위 폴더에 쓰지 않도록 여기서 막는다.
    """
    out = re.sub(r'[/\\:*?"<>|]+', "_", str(name)).strip().strip(".")
    return out or "unknown"


def load_model(model_id: str):
    """model_id 에 맞는 모델/프로세서 클래스를 Auto* 로 자동 선택해 로드한다.

    Qwen2.5-VL 은 Qwen2_5_VLForConditionalGeneration, Qwen3-VL(dense 8B/32B)은
    Qwen3VLForConditionalGeneration 으로 클래스가 다르지만, 둘 다
    AutoModelForImageTextToText 로 커버되므로 여기서 분기할 필요가 없다.

    27B 는 device_map="auto" 로 여러 GPU 에 층을 나눠 올린다 - 그러면 이
    프로세스가 CUDA_VISIBLE_DEVICES 로 보이는 GPU 를 전부 쓰게 되므로, 8샤드
    병렬(샤드마다 GPU 1장) 구조로는 못 돌린다. run_video_27b.sh 는 처음부터
    GPU 여러 장을 한 프로세스에 몰아준다.
    """
    print(f"[model] loading {model_id} ...")
    t0 = time.time()
    if model_id in MULTI_GPU_MODELS:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda:0")
    processor = AutoProcessor.from_pretrained(model_id, max_pixels=768 * 768)
    model.eval()
    print(f"[model] loaded in {time.time()-t0:.1f}s")
    return model, processor


# qa 스타일(verdict/categories/evidence 3필드)은 256 토큰이면 충분하지만,
# nureasoning 은 6단계 서술을 모두 뱉으므로 256 에서 문장 중간에 잘린다.
# 실측(20260811, 50클립): 256 으로 두면 12/50(24%)이 JSON 미완성으로 파싱
# 실패했고, 잘린 위치는 모두 마지막 필드(rarity_assessment) 근처였다.
NUR_MAX_NEW_TOKENS = 800

# --difficulty 를 켜면 JSON 에 필드가 8개 늘어난다(4축 x 값+근거 문장).
# 800 은 그 여유가 없다 - 위 실측에서 이미 마지막 필드 근처에서 잘렸고,
# 난이도 필드는 그보다 더 뒤에 온다. 근거 문장 5개를 짧게 잡아도 약 200
# 토큰이 더 필요하므로 그만큼 올린다. 난이도를 끈 실행의 예산은 건드리지
# 않는다 - 예산이 바뀌면 기존 A/B 결과와 비교할 수 없게 된다.
NUR_MAX_NEW_TOKENS_DIFFICULTY = 1100


# 클립 모드 CSV 의 열. 헤더와 빈 행이 같은 정의를 봐야 어긋나지 않는다.
CLIP_CSV_COLUMNS = [
    "uuid", "n_frames", "verdict", "n_categories",
    "categories", "parse_ok", "observation", "ego_behavior",
    "unusual_elements", "safety_tier", "safety_label",
    "safety_reason", "rarity_tier", "rarity_label", "rarity_reason",
    "tier_score",
    "ego_speed_kmh", "ego_motion", "ego_behavior_measured",
    # 결정 마진 - 이 클립의 판정이 얼마나 아슬아슬했는지.
    # margin_min 이 작을수록 의미 없는 프롬프트 섭동에도 뒤집힌다.
    "margin_min", "margin_mean", "n_close_tokens",
    # 난이도 4축. --difficulty 를 끈 실행에서는 전부 빈 칸이지만 열 자체는
    # 항상 쓴다 - 열 구성이 실행마다 달라지면 샤드 병합과 실행 간 비교가
    # 깨진다.
    *[c for k, _ in DIFFICULTY_AXES for c in (k, f"{k}_reason")],
]


def _lp_list(proc):
    """LogitsProcessor 하나를 generate 가 받는 리스트 형태로 감싼다."""
    if proc is None:
        return None
    return LogitsProcessorList([proc])


# 판정을 담는 JSON 키 - 이 값들이 바뀌면 채점 결과가 바뀐다.
DECISION_KEYS = ("scenario_types", "safety_tier", "rarity_tier", "verdict",
                 "scenario_type", "influenced_ego")


def _decision_steps(gen_ids, tokenizer):
    """생성 토큰 인덱스 중 '판정 값이 시작되는' 자리들의 집합.

    토큰을 하나씩 붙여가며 문자열을 재구성하고, DECISION_KEYS 뒤의 값이
    열리는 지점(따옴표/숫자/true/false 가 시작되는 곳)을 담은 토큰을 고른다.
    """
    import re
    text, spans = "", []
    for t in gen_ids:
        piece = tokenizer.decode([t])
        spans.append((len(text), len(text) + len(piece)))
        text += piece

    targets = []
    for key in DECISION_KEYS:
        for mk in re.finditer(re.escape(f'"{key}"'), text):
            # 키 뒤 ':' 다음에 오는 첫 값 문자
            mv = re.compile(r'\s*:\s*\[?\s*("?)').search(text, mk.end())
            if mv:
                targets.append(mv.end())
    out = set()
    for pos in targets:
        for i, (a, b) in enumerate(spans):
            if a <= pos < b:
                out.add(i)
                break
    return out


def _decision_margin(gen_out, tokenizer):
    """생성 토큰의 1위-2위 확률차를 재되, 판정에 실제로 쓰이는 자리만 본다.

    왜 필요한가: 이 파이프라인은 결정적이다(같은 입력 -> 같은 출력, 실측
    115/115 일치). 그런데 카테고리 메뉴의 마침표 하나를 빼자 10/115 클립의
    예측이 뒤집혀 accuracy 가 2.7%p 움직였다. 토큰 수도 위치도 그대로이고
    바뀐 것은 토큰 하나의 임베딩뿐이었다 - 즉 그 10개는 1위와 2위가 백지
    한 장 차이인 경계선 클립이고, 의미 없는 섭동에도 넘어간다.

    왜 전체 토큰의 최솟값을 쓰면 안 되는가:
      처음에 그렇게 재봤더니 3/3 클립이 margin_min=0.0000 이었다. 0 이
      나온 자리를 열어보니 산문 안의 ',' vs '.', ' in' vs ' with' 같은
      말투 선택이었다. 그런 동점은 어느 쪽이 이겨도 카테고리도 등급도
      바뀌지 않는다 - 지표가 판정이 아니라 문장 표현의 흔들림을 재고
      있었고, 모든 클립이 0 으로 뭉개져 아무것도 구분하지 못했다.

    그래서 JSON 값이 시작되는 자리(따옴표/괄호/숫자 직후)만 센다. 여기서
    갈리면 카테고리 이름이나 등급 숫자가 실제로 바뀐다.

    돌려주는 값:
      margin_min  판정 자리 중 가장 아슬아슬했던 확률차. 클립이 뒤집힐지를
                  좌우하는 것은 평균이 아니라 이 최솟값이다.
      margin_mean 판정 자리들의 평균.
      n_close     확률차가 CLOSE_CALL_P 미만인 판정 자리 수.

    scores 가 없으면(구버전 transformers) None - 마진은 부가 정보라
    없다고 실행을 세울 이유는 없다.
    """
    scores = getattr(gen_out, "scores", None)
    seqs = getattr(gen_out, "sequences", None)
    if not scores or seqs is None:
        return None
    gen_ids = seqs[0, seqs.shape[1] - len(scores):].tolist()

    # 판정 자리 찾기: 생성 텍스트에서 우리가 실제로 파싱하는 키의 값이
    # 시작되는 오프셋을 구하고, 그 오프셋을 담은 토큰을 고른다.
    # 문장 안의 쉼표 같은 자리를 세면 말투의 흔들림을 재게 된다(실측:
    # 그렇게 했더니 ',' vs '.' 동점 때문에 전 클립이 0 으로 뭉개졌다).
    decision_at = _decision_steps(gen_ids, tokenizer)
    if not decision_at:
        return None

    mn, tot, n_close, n = 1.0, 0.0, 0, 0
    for i, step in enumerate(scores):
        # 판정 자리인가: 직전 토큰이 값의 시작을 여는 자리여야 한다.
        # 첫 토큰과 여는 따옴표/괄호/콜론 뒤가 그렇다.
        if i not in decision_at:
            continue
        top2 = torch.topk(torch.softmax(step[0].float(), dim=-1), 2)
        d = float(top2.values[0] - top2.values[1])
        mn = min(mn, d)
        tot += d
        n_close += int(d < CLOSE_CALL_P)
        n += 1
    if n == 0:
        return None
    return {"margin_min": mn, "margin_mean": tot / n, "n_close": n_close,
            "n_tokens": n}


# 1위-2위 확률차가 이 값 미만이면 "아슬아슬한 판정" 으로 센다. 0.1 은
# 임의 기준이지만, 마침표 실험에서 뒤집힌 클립을 가르는 데 쓸 수 있는
# 크기다 (뒤집히려면 섭동이 이 차이를 넘겨야 한다).
CLOSE_CALL_P = 0.1


@torch.inference_mode()
def _generate(model, processor, images, text_prompt, max_new_tokens=256,
              as_video=False, video_fps=None, logits_processor=None,
              want_margin=False, thinking=False):
    """이미지 목록을 넣고 모델 원문 출력을 받는다.

    as_video=True 면 낱장 이미지 N개가 아니라 "비디오 한 편"으로 넘긴다.
    Qwen3-VL 은 비디오를 별도 경로로 처리하므로 같은 프레임이라도 결과가
    다르다 (실측 20260811, 640x360 20프레임):

      낱장 이미지 20개 : 4,449 토큰
      비디오 1편       : 2,296 토큰  (51%)

    토큰이 반으로 주는 것은 temporal_patch_size=2 라 인접 프레임 쌍이 하나로
    병합되기 때문이다. 그리고 프롬프트에 프레임별 타임스탬프("0.5 seconds",
    "2.5 seconds", ...)가 자동으로 박히므로, 모델이 "몇 초 간격인지"를 문장
    설명이 아니라 위치 인코딩으로 알게 된다.

    video_fps 는 우리가 뽑은 샘플링 fps(예: 1.0)를 그대로 준다. 이 값으로
    타임스탬프가 계산되므로 실제 클립 시간과 맞아야 한다.
    """
    if as_video and images:
        return _generate_video(model, processor, images, text_prompt,
                               max_new_tokens=max_new_tokens,
                               video_fps=video_fps or 1.0,
                               logits_processor=logits_processor,
                               want_margin=want_margin, thinking=thinking)

    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": text_prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=thinking,
    )
    proc_kwargs = dict(text=[text], padding=True, return_tensors="pt")
    if images:
        proc_kwargs["images"] = images
    inputs = processor(**proc_kwargs).to(model.device)

    gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=False,
                         logits_processor=_lp_list(logits_processor),
                         return_dict_in_generate=want_margin,
                         output_scores=want_margin)
    margin = _decision_margin(gen, processor.tokenizer) if want_margin else None
    seq = gen.sequences if want_margin else gen
    trimmed = seq[:, inputs.input_ids.shape[1]:]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    del inputs, gen, trimmed
    torch.cuda.empty_cache()
    return out.strip()


@torch.inference_mode()
def _generate_video(model, processor, frames, text_prompt, max_new_tokens=256,
                    video_fps=1.0, logits_processor=None, want_margin=False,
                    thinking=False):
    """이미 뽑아둔 프레임 목록을 비디오 한 편으로 넘겨 생성한다.

    do_sample_frames=False 로 두는 것이 핵심 - 우리가 이미 1fps 로 골라둔
    프레임이므로 프로세서가 다시 샘플링(기본 fps=2)하면 안 된다.

    video_metadata 는 생략하면 안 된다. 없으면 프로세서가 fps=24 로 가정해
    타임스탬프가 실제 시간과 어긋나고(20초 클립이 0.8초로 보인다), Qwen3VL
    구현은 frames_indices 가 없으면 예외를 던진다.
    """
    arr = np.stack([np.asarray(f.convert("RGB")) for f in frames])
    meta = VideoMetadata(
        total_num_frames=len(frames), fps=float(video_fps),
        duration=len(frames) / float(video_fps),
        width=frames[0].size[0], height=frames[0].size[1],
        video_backend="custom", frames_indices=list(range(len(frames))),
    )
    messages = [{"role": "user", "content": [{"type": "video"},
                                             {"type": "text",
                                              "text": text_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=thinking)
    inputs = processor(text=[text], videos=[arr], video_metadata=[meta],
                       do_sample_frames=False, padding=True,
                       return_tensors="pt").to(model.device)

    gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=False,
                         logits_processor=_lp_list(logits_processor),
                         return_dict_in_generate=want_margin,
                         output_scores=want_margin)
    margin = _decision_margin(gen, processor.tokenizer) if want_margin else None
    seq = gen.sequences if want_margin else gen
    trimmed = seq[:, inputs.input_ids.shape[1]:]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    del inputs, gen, trimmed
    torch.cuda.empty_cache()
    return (out.strip(), margin) if want_margin else out.strip()


def build_sensor_facts(uuid, frame_idx, use_egomotion, use_obstacle, n_views=3):
    """프롬프트에 넣을 센서 사실 문구와, 시각화/CSV 용 원본 상태를 함께 반환.

    n_views 는 obstacle 목록을 실제 카메라 화각으로 자르는 데 쓴다 - 뷰 구성과
    어긋나면 화면에 없는 객체를 모델에게 알려주게 된다.
    """
    lines, ego = [], None
    if use_egomotion:
        ego = ego_state(uuid, frame_idx)
        if ego is not None:
            lines.append(describe_ego(ego))
    if use_obstacle:
        s = describe_obstacles(obstacle_summary(uuid, frame_idx, n_views=n_views))
        if s:
            lines.append(s)
    return "\n".join(lines), ego


def build_behavior_facts(uuid, frame_idx, use_egomotion):
    """nuReasoning 2단계용 "지난 N초간 자차 행동 변화" 문장과 원본 dict.

    egomotion 라벨을 쓰지 않는 실행에서는 빈 문자열 - 그 경우 모델이 이미지
    만으로 행동 변화를 추정하게 되고, 프롬프트에서 해당 블록이 통째로 빠진다.
    클립 앞부분처럼 되돌아볼 구간이 없으면 ego_behavior_change 가 None 을
    주므로 그 경우도 빈 문자열이다.
    """
    if not use_egomotion:
        return "", None
    ch = ego_behavior_change(uuid, frame_idx)
    return (describe_behavior(ch) if ch else ""), ch


# --use-egomotion-c 가 ② 자리에 넣는 내용 없는 문장.
#
# 왜 이런 게 필요한가: --use-egomotion 을 켜면 프롬프트에 세 가지가 한꺼번에
# 들어간다 - ① "HOW THE EGO-VEHICLE'S BEHAVIOUR CHANGED" 헤더, ② 센서 수치
# 문장, ③ Safety 루브릭 끝의 behavior_hint. 셋 다 behavior_facts 가 비었는지
# 하나로 갈리므로, A(전부 off)/B(전부 on) 두 점만으로는 +7.9%p 가 어디서
# 왔는지 알 수 없다.
#
# C 는 ①③ 을 그대로 둔 채 ② 의 정보만 뺀다. 문장 구조와 길이는 유지하되
# 클립마다 달라지는 값이 없으므로, B 에 근접하면 센서 수치는 기여하지 않고
# 프롬프트 구조가 일을 한 것이다.
EGO_PLACEBO_SENTENCE = "This clip covers about 20 seconds of continuous driving."


def build_clip_ego_facts(uuid, use_egomotion, ego_track=False,
                         frame_indices=None, ego_ablation=None):
    """클립 모드용 egomotion 사실. (프롬프트 문구, 요약 dict) 를 돌려준다.

    build_sensor_facts + build_behavior_facts 의 클립 판이다. 저 둘은
    한 프레임 시점의 스냅샷(속도 순간값)과 그 앞 5초를 말하는데, 클립
    모드는 20초 영상을 통째로 넣으므로 시점이 어긋난다. 여기서는 클립
    전 구간을 한 번에 요약해 그 어긋남을 없앤다.

    describe_ego() 의 "is MOVING at 42 km/h" 를 여기서 쓰지 않는 이유:
    현재형 문장이 20초 영상 전체를 가리키는 것처럼 읽히지만 실제로는
    마지막 프레임의 값이다. 구간 요약이 속도 범위를 이미 말하므로
    중복이기도 하다.
    """
    # 두 플래그는 독립이다. 세 조합을 각각 A/B 할 수 있어야 "요약 문장이
    # 기여하는가" 와 "원시 수치가 기여하는가" 를 분리할 수 있다:
    #   use_egomotion 만  -> 요약 한 문장 (해석이 들어간 서술)
    #   ego_track 만      -> 1초 간격 수치만 (해석어 없음)
    #   둘 다             -> 요약 + 수치
    # C: 센서 수치 대신 내용 없는 문장. ch 는 CSV/시각화용으로 계속 만든다.
    if ego_ablation == "c":
        return EGO_PLACEBO_SENTENCE, ego_clip_behavior(uuid)
    # D: 프롬프트에는 아무것도 넣지 않는다 (hint 만 별도로 켜진다).
    if ego_ablation == "d":
        return "", ego_clip_behavior(uuid)

    if not (use_egomotion or ego_track):
        return "", None

    # ch 는 CSV/시각화가 쓰는 원본 dict 라 ego_track 단독일 때도 만들어 둔다
    # (프롬프트에 넣지 않을 뿐, 결과 파일의 ego_speed_kmh 등이 비면 곤란하다).
    ch = ego_clip_behavior(uuid)

    parts = []
    if use_egomotion and ch:
        parts.append(describe_clip_behavior(ch))
    if ego_track:
        tr = describe_clip_track(ego_clip_track(uuid, frame_indices))
        if tr:
            parts.append(tr)
    return "\n".join(parts), ch


def classify_unit(model, processor, unit_frames, prompt, views,
                  max_new_tokens=256):
    """직전 + 현재 프레임을 뷰 순서대로 한 번에 넣어 모델 원문 출력을 받는다.

    3뷰면 6장, front-wide 단독이면 2장.
    """
    prev, cur = unit_frames["prev"], unit_frames["cur"]
    images = ([prev[v] for v in views if prev.get(v) is not None]
              + [cur[v] for v in views if cur.get(v) is not None])
    return _generate(model, processor, images, prompt,
                     max_new_tokens=max_new_tokens)


def run_clip_inference(uuids, labels, category_menu,
                       model_id="Qwen/Qwen3-VL-8B-Instruct",
                       out_csv="clip_results.csv",
                       use_egomotion=False, use_obstacle=False,
                       single_view=True,
                       fps=CLIP_FPS, max_frames=CLIP_MAX_FRAMES,
                       max_long_side=CLIP_MAX_LONG_SIDE,
                       viz_dir=None, viz_only_edge=True, not_save_low=True,
                       viz_width=VIZ_WIDTH, video_input=True,
                       constrain_tiers=True,
                       viz_normal=None, viz_special=None,
                       gt_labels=None, timeline=False, ego_track=False,
                       ego_ablation=None, header_style="v1",
                       want_margin=False, safety_tiers=True, rarity_tiers=True,
                       difficulty=False, difficulty_only=False, traj=None,
                       viz_per_category=None):
    """클립 전체(20초)를 1fps 로 넣어 클립 단위로 판정한다.

    run_inference 와 판정 단위가 다르다 - 저쪽은 (uuid, frame_idx) 이고
    여기는 uuid 하나가 한 행이다. 그래서 루프를 공유하지 않고 따로 뒀다.
    프롬프트는 nuReasoning 방식만 쓴다: 20장을 시간순으로 읽는 것이 이 모드의
    목적이고, Q1/Q2/Q3 는 "현재 순간"을 전제로 쓰인 문구라 시퀀스 입력과
    맞지 않는다.

    센서 사실은 클립의 마지막 프레임 기준으로 만든다 - 모델이 보는 마지막
    이미지와 같은 시점이어야 "지금 자차 상태"라는 말이 맞기 때문.

    viz_dir 을 주면 클립마다 <viz_dir>/<uuid>/{clip.mp4,result.json} 을 남긴다.
    영상은 1fps 추론 입력이 아니라 원본 mp4 를 그대로 쓰고, 그 아래에 1~5단계
    추론 내용을 붙인다. viz_only_edge=True(기본)면 edge-case 요소가 하나라도
    잡힌 클립만 만든다 - 전량은 클립당 약 39MB 라 금방 수십 GB 가 된다.

    gt_labels({uuid: {categories, safety, rarity}})를 주면 시각화 패널에
    GT 와 Pred 를 나란히 그린다 - 검수자가 라벨과 예측을 한 화면에서
    비교할 수 있게 한다. 없는 uuid 는 그냥 Pred 만 그린다.

    viz_normal / viz_special 을 주면 그 둘로만 대상을 정하고, 결과를
    <viz_dir>/{normal,special}/score_<N>/<uuid>/ 로 나눠 담는다. 둘 다 None
    이면 예전 방식(viz_only_edge + not_save_low)으로 동작한다.

    not_save_low=True(기본)면 거기서 한 번 더 거른다: Safety Criticality 와
    Rarity 가 둘 다 "Low" 인 클립은 저장하지 않는다 - 카테고리는 나열됐지만
    자차에 영향도 없고(Q3 폐지 대신 4단계가 이 역할) 희귀하지도 않다고 모델
    스스로 판단한 경우다. 탐지(scenario_types, CSV)는 건드리지 않고 시각화
    대상만 줄인다 - 이 필터가 틀려도 재추론 없이 CSV 로 다시 뽑을 수 있다.

    safety_tiers / rarity_tiers 는 4/5단계를 하나씩 끈다. 끈 등급은 프롬프트와
    출력 스키마에서 빠지고 CSV 에서 빈 칸이 되며, tier_score(합계)도 비게
    된다. 값이 없으면 not_save_low 의 "둘 다 Low" 조건이 성립하지 않아 그
    필터는 저절로 무력화된다.
    """
    model, processor = load_model(model_id)
    views = views_for(single_view)

    # 등급 필드를 1/2/3 정수로만 나오게 디코딩 단계에서 막는다.
    # 프롬프트 지시만으로는 "Low to moderate" 류가 새어나왔다(실측 20260811).
    # 끈 등급은 애초에 생성되지 않으므로 제약 대상에서도 뺀다 - 둘 다 끄면
    # 제약할 필드가 없어 프로세서 자체를 만들지 않는다.
    tier_field_names = tuple(
        n for n, on in (("safety_tier", safety_tiers),
                        ("rarity_tier", rarity_tiers)) if on)
    tier_proc = (make_tier_processor(processor.tokenizer,
                                     fields=tier_field_names)
                 if constrain_tiers and tier_field_names else None)

    n_special = sum(1 for l in labels if not l["is_normal"])
    print(f"[clip] {len(uuids)} clips | {len(views)} view(s) | {fps}fps "
          f"max {max_frames} frames @ {max_long_side}px | {n_special} categories")
    print(f"[clip] input mode: {'VIDEO (temporal merge + timestamps)' if video_input else 'image list'}")
    print(f"[clip] tiers: safety={'on' if safety_tiers else 'off'}  "
          f"rarity={'on' if rarity_tiers else 'off'}")
    print("[clip] tier constraint: "
          + (f"ON (1/2/3 enforced at decode: {', '.join(tier_field_names)})"
             if tier_proc else "OFF"))

    cat_counts = Counter()
    # --viz-per-category N: 카테고리마다 최초 N개 클립만 시각화한다.
    # 폴더는 score 가 아니라 카테고리 이름으로 나누고, 한 클립이 여러
    # 카테고리를 받으면 해당 폴더 전부에 중복 저장한다.
    viz_cat_counts = Counter()
    n_parse_fail = n_viz = n_edge = 0
    viz_path = Path(viz_dir) if viz_dir else None
    if viz_path:
        viz_path.mkdir(parents=True, exist_ok=True)
        if viz_normal is not None or viz_special is not None:
            which = [n for n, on in (("normal", viz_normal),
                                     ("special", viz_special)) if on]
            print(f"[clip] viz -> {viz_path}/{{{','.join(which) or 'none'}}}"
                  f"/score_<N>/<uuid>/{{clip.mp4,result.json}}"
                  f"  width {viz_width or 'original'}")
        else:
            print(f"[clip] viz -> {viz_path}/score_<N>/<uuid>/{{clip.mp4,result.json}}"
                  + ("  (clips with >=1 category only)" if viz_only_edge
                     else "  (all clips)")
                  + f"  width {viz_width or 'original'}")
    t_start = time.time()

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CLIP_CSV_COLUMNS)

        pbar = tqdm(uuids, total=len(uuids), unit="clip", dynamic_ncols=True,
                    mininterval=1.0, smoothing=0.1)
        for uuid in pbar:
            clip = sample_clip_frames(uuid, fps=fps, max_frames=max_frames,
                                      max_long_side=max_long_side, views=views,
                                      traj=traj)
            images = [im for _, _, im in clip["frames"]]
            if not images:
                # 빈 칸 개수를 손으로 세지 않는다 - 열이 늘 때마다 어긋난다.
                row = [uuid, 0, "Normal", 0, "", 0, "(no frames)"]
                writer.writerow(row + [""] * (len(CLIP_CSV_COLUMNS) - len(row)))
                n_parse_fail += 1
                continue

            # egomotion 은 클립 전 구간을 요약한다 (시간축 정렬).
            behavior, ego = build_clip_ego_facts(uuid, use_egomotion,
                                                 ego_track=ego_track,
                                                 frame_indices=clip["indices"],
                                                 ego_ablation=ego_ablation)

            # 3D bbox 는 아직 마지막 프레임 ±0.05초 스냅샷이다. 20초 영상에
            # 붙이기엔 시간축이 어긋나 있어(실측: 과탐 감소 대신 recall -6%p)
            # 기본 off 이고, 시간축 요약은 후속 작업으로 남겨둔다.
            last_idx = clip["indices"][-1]
            facts = ""
            if use_obstacle:
                facts = describe_obstacles(
                    obstacle_summary(uuid, last_idx, n_views=len(views)))

            prompt = build_nureasoning_prompt(
                category_menu, facts, behavior_facts=behavior,
                intro=clip_intro(len(images), len(views), fps=fps,
                                 as_video=video_input),
                timeline=timeline,
                # D 는 behavior_facts 가 비어도 hint 를 켠다.
                force_behavior_hint=(ego_ablation == "d"),
                header_style=header_style,
                safety_tiers=safety_tiers,
                rarity_tiers=rarity_tiers,
                difficulty=difficulty,
                difficulty_only=difficulty_only)
            gen_out = _generate(model, processor, images, prompt,
                                max_new_tokens=(NUR_MAX_NEW_TOKENS_DIFFICULTY
                                                if difficulty
                                                else NUR_MAX_NEW_TOKENS),
                                as_video=video_input, video_fps=fps,
                                logits_processor=tier_proc,
                                want_margin=want_margin,
                                thinking=(model_id in THINKING_MODELS))
            raw, margin = gen_out if want_margin else (gen_out, None)
            result = parse_nureasoning_output(raw, labels)

            cats = result["categories"]
            cat_counts.update(cats)
            n_edge += bool(cats)
            n_parse_fail += not result["parse_ok"]

            writer.writerow([
                uuid, len(images), result["verdict"],
                len(cats), "|".join(cats), int(result["parse_ok"]),
                result.get("observation", ""), result.get("ego_behavior", ""),
                result.get("unusual_elements", ""),
                result.get("safety_tier", ""), result.get("safety_label", ""),
                result.get("safety_assessment", ""),
                result.get("rarity_tier", ""), result.get("rarity_label", ""),
                result.get("rarity_assessment", ""),
                result.get("tier_score", ""),
                # 클립 요약이므로 한 시점의 속도가 아니라 구간 범위를 적는다
                f"{ego['speed_min']:.0f}-{ego['speed_max']:.0f}" if ego else "",
                ("stopped" if ego and ego["stopped_s"] >= ego["span_s"] - 0.5
                 else "moving" if ego else ""),
                behavior,
                f"{margin['margin_min']:.4f}" if margin else "",
                f"{margin['margin_mean']:.4f}" if margin else "",
                margin["n_close"] if margin else "",
                # 난이도: 못 읽은 축은 None 이라 빈 칸으로 나간다.
                *[v for k, _ in DIFFICULTY_AXES
                  for v in (result.get(k) if result.get(k) is not None else "",
                            result.get(f"{k}_reason", ""))],
            ])
            f.flush()
            pbar.set_postfix_str(
                f"{'EDGE' if cats else '----'} cats={len(cats)}")

            # 시각화 대상 결정.
            #
            # viz_normal / viz_special 은 서로 독립이다. 둘 다 켜면 전부,
            # 하나만 켜면 그쪽만, 둘 다 끄면(그리고 viz_only_edge 도 아니면)
            # 아무것도 만들지 않는다. viz_only_edge 는 예전 인자로, 켜져 있으면
            # special 만 만든다는 뜻이라 viz_special 과 같은 의미다.
            is_special = bool(cats)
            if viz_normal is None and viz_special is None:
                # 예전 방식으로 호출된 경우 - 기존 동작을 그대로 유지한다.
                want_viz = is_special or not viz_only_edge
                if want_viz and not_save_low and is_special:
                    want_viz = not (result.get("safety_tier") == 1
                                    and result.get("rarity_tier") == 1)
            else:
                want_viz = (viz_special if is_special else viz_normal) or False

            # 카테고리별 N개 모드는 special 클립만, 아직 정원이 안 찬
            # 카테고리에 한해 만든다.
            targets = None
            if viz_per_category:
                targets = [c for c in cats
                           if viz_cat_counts[c] < viz_per_category]
                want_viz = bool(targets)

            if viz_path and want_viz:
                if targets is not None:
                    # <viz_dir>/<Category>/<uuid>/ - 여러 카테고리면 전부에
                    # 같은 클립을 넣는다(중복 저장이 의도된 동작).
                    out_dirs = [viz_path / _safe_dirname(c) / uuid
                                for c in targets]
                else:
                    # <viz_dir>/{normal,special}/score_<N>/<uuid>/ 로 나눈다.
                    # 점수는 safety+rarity 합계(2~8), 못 읽으면 score_unknown.
                    bucket = "special" if is_special else "normal"
                    score_dir = score_dirname(result.get("tier_score"))
                    out_dirs = [viz_path / bucket / score_dir / uuid
                                if (viz_normal is not None or viz_special is not None)
                                else viz_path / score_dir / uuid]
                for out_dir in out_dirs:
                    render_clip_result(
                        uuid, clip_path(views[0], uuid), result,
                        out_dir, max_width=viz_width,
                        gt=(gt_labels or {}).get(uuid),
                        traj=traj, traj_view=views[0],
                        extra={"ego_behavior_measured": behavior,
                               "n_frames_seen": len(images),
                               # 클립 요약이라 한 시점의 속도가 없다 - 구간 범위를 준다
                               "ego_speed_kmh": (
                                   f"{ego['speed_min']:.0f}-{ego['speed_max']:.0f}"
                                   if ego else None)})
                    n_viz += 1
                if targets is not None:
                    viz_cat_counts.update(targets)

    dt = time.time() - t_start
    n = max(len(uuids), 1)
    print(f"\n[clip] done: {len(uuids)} clips in {dt/60:.1f} min "
          f"({dt/n:.1f} s/clip)")
    if n_parse_fail:
        print(f"[warn] JSON parse failed on {n_parse_fail} clips")
    print(f"[clip] SPECIAL (>=1 category): {n_edge}/{n} ({100*n_edge/n:.1f}%)")
    print(f"[clip] NORMAL   (no category)  : {n-n_edge}/{n} "
          f"({100*(n-n_edge)/n:.1f}%)")
    print("[clip] categories:")
    for c, v in cat_counts.most_common():
        print(f"   {c:32} {v:4d}  ({100*v/n:5.1f}%)")
    if viz_path:
        print(f"[clip] videos written: {n_viz}")
    print(f"[saved] {out_csv}")


def run_inference(units, labels, category_menu,
                  model_id="Qwen/Qwen3-VL-8B-Instruct",
                  out_csv="edge_case_results.csv", viz_dir="viz",
                  use_egomotion=False, use_obstacle=False, check_path=False,
                  ask_blocking=True, single_view=False,
                  prompt_style="qa"):
    """units: [(uuid, frame_idx), ...]

    check_path=True 면 3D 라벨로 전방 통로 침범을 계산해 모델의 Q3 와 대조한다.
    프롬프트에는 넣지 않으므로 모델 출력 자체는 달라지지 않는다.
    ask_blocking=False 면 Q3 를 묻지 않고, 시각화도 blocking 으로 나누지 않는다.
    single_view=True 면 front-wide 만 써서 이미지가 6장이 아니라 2장이 된다.

    prompt_style="nureasoning" 이면 6단계 CoT + 1~10 점수 프롬프트로 갈아끼운다.
    이때 --use-egomotion 이 켜져 있으면 자차 행동 변화 문장이 함께 들어간다
    (논문 2단계의 근거). 난이도 점수는 매기지 않는다 - 판정은 edge-case 요소를
    하나라도 나열했는가로만 정한다.
    """
    nur = prompt_style == "nureasoning"
    if nur:
        # 이 프롬프트에는 Q3 가 없다. 호출부가 이미 강제하지만, run_inference 를
        # 직접 쓰는 경로에서도 폴더 구조가 어긋나지 않도록 여기서도 막는다.
        ask_blocking = False
    model, processor = load_model(model_id)
    viz_path = Path(viz_dir)
    viz_path.mkdir(parents=True, exist_ok=True)

    views = views_for(single_view)

    # 프롬프트 생성/파싱을 스타일에 따라 한 쌍으로 골라둔다. 아래 루프는
    # 어느 스타일인지 신경 쓰지 않는다.
    def _make_prompt(facts="", behavior=""):
        if nur:
            return build_nureasoning_prompt(category_menu, facts,
                                            single_view=single_view,
                                            behavior_facts=behavior)
        return build_vlm_prompt(category_menu, facts,
                                ask_blocking=ask_blocking,
                                single_view=single_view)

    def _parse(raw):
        if nur:
            return parse_nureasoning_output(raw, labels)
        return parse_vlm_output(raw, labels, ask_blocking=ask_blocking)

    # 센서 사실을 안 쓰면 프롬프트가 매번 같으므로 한 번만 만든다
    base_prompt = _make_prompt()
    use_sensors = use_egomotion or use_obstacle

    # 멀티라벨이라 카테고리 합계는 판정 단위 수를 넘을 수 있다.
    # blocking 여부로 한 번 더 쪼개서 센다.
    cat_counts = Counter()
    cat_counts_by_block = {"blocking_yes": Counter(), "blocking_no": Counter()}
    block_counts = Counter()   # 카테고리가 붙은 단위의 blocking yes/no
    n_labeled = 0
    n_parse_fail = 0
    # Q3 교차검증: 모델 답 x 기하 계산의 2x2 (검증을 켠 단위만 집계)
    agree = Counter()
    t_start = time.time()

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "uuid", "frame_idx", "verdict", "blocks_path", "n_categories",
            "categories", "evidence", "parse_ok", "ego_speed_kmh", "ego_motion",
            "path3d_blocked", "path3d_nearest_m", "path3d_nearest_class",
            "path3d_agree",
        ])

        pbar = tqdm(units, total=len(units), unit="unit", dynamic_ncols=True,
                    mininterval=1.0, smoothing=0.1)
        for uuid, frame_idx in pbar:
            unit_frames = sample_unit_frames(uuid, frame_idx, views=views)

            # behavior 는 아래 payload 에서도 읽으므로 프레임이 없는 분기에서도
            # 정의돼 있어야 한다.
            ego, behavior = None, ""
            if not any(unit_frames["cur"].values()):
                result = {"verdict": "Normal", "categories": [],
                          "blocks_path": False if ask_blocking else None,
                          "evidence": "(no frame)", "parse_ok": False}
            else:
                facts = ""
                if use_sensors:
                    facts, ego = build_sensor_facts(
                        uuid, frame_idx, use_egomotion, use_obstacle,
                        n_views=len(views))
                if nur:
                    behavior, _ = build_behavior_facts(
                        uuid, frame_idx, use_egomotion)
                prompt = (_make_prompt(facts, behavior)
                          if (facts or behavior) else base_prompt)
                raw = classify_unit(
                    model, processor, unit_frames, prompt, views,
                    max_new_tokens=NUR_MAX_NEW_TOKENS if nur else 256)
                result = _parse(raw)

            cats = result["categories"]
            block_key = blocking_dir(result)          # Q3 off 면 None
            cat_counts.update(cats)
            n_parse_fail += not result["parse_ok"]
            if cats:
                n_labeled += 1
                if block_key:
                    block_counts[block_key] += 1
                    cat_counts_by_block[block_key].update(cats)

            # Q3 교차검증 (프롬프트에는 들어가지 않았으므로 모델 답과 독립).
            # Q3 를 안 물었으면 대조할 모델 답이 없어 기하 결과만 남긴다.
            path = path_intrusion(uuid, frame_idx) if check_path else None
            if path is not None and ask_blocking:
                agree[(result["blocks_path"], path["blocked"])] += 1

            writer.writerow([
                uuid, frame_idx, result["verdict"],
                "" if block_key is None else ("Yes" if result["blocks_path"] else "No"),
                len(cats), "|".join(cats), result["evidence"],
                int(result["parse_ok"]),
                f"{ego['speed_kmh']:.1f}" if ego else "",
                ego["motion"] if ego else "",
                ("Yes" if path["blocked"] else "No") if path else "",
                path["nearest_m"] if path and path["nearest_m"] is not None else "",
                path["nearest_class"] or "" if path else "",
                (int(result["blocks_path"] == path["blocked"])
                 if (path and ask_blocking) else ""),
            ])
            f.flush()

            # Q3 를 물었으면 blocking_{yes,no}/<category>/, 아니면 <category>/
            # 아래에 저장. 멀티라벨이면 해당하는 모든 카테고리 폴더에 중복 저장.
            targets = viz_targets(result)
            if targets:
                payload = {
                    "uuid": uuid, "frame_idx": frame_idx,
                    "verdict": result["verdict"],
                    "categories": cats,
                    "evidence": result["evidence"],
                }
                if ask_blocking:
                    payload["blocks_path"] = result["blocks_path"]
                if nur:
                    # 단계별 답을 그대로 남긴다 - 왜 그렇게 봤는지 사후
                    # 검수할 때 이게 유일한 근거다. evidence 는 이 중 두 개를
                    # 이어 붙인 요약이라 원본을 대신하지 못한다.
                    payload["reasoning"] = {
                        k: result.get(k, "") for k in
                        ("observation", "ego_behavior", "unusual_elements",
                         "safety_assessment", "rarity_assessment")
                    }
                    if behavior:
                        payload["ego_behavior_measured"] = behavior
                if ego:
                    payload["ego"] = {"speed_kmh": round(ego["speed_kmh"], 1),
                                      "motion": ego["motion"]}
                if path is not None:
                    payload["path_3d"] = {
                        "blocked": path["blocked"],
                        "n_in_path": path["n_in_path"],
                        "nearest_m": path["nearest_m"],
                        "nearest_class": path["nearest_class"],
                    }
                    if ask_blocking:
                        payload["path_3d"]["agrees_with_model"] = (
                            result["blocks_path"] == path["blocked"])
                payload_txt = json.dumps(payload, ensure_ascii=False, indent=2)

                card_src = None
                for rel in targets:
                    out_dir = viz_path / rel / unit_name(uuid, frame_idx)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / "result.json").write_text(payload_txt,
                                                         encoding="utf-8")
                    dst = out_dir / "card.png"
                    if card_src is None:
                        # 카드는 한 번만 그리고, 나머지 폴더에는 복사해 넣는다
                        render_scene_card(uuid, frame_idx, unit_frames["cur"],
                                          result, out_path=dst)
                        card_src = dst
                    else:
                        shutil.copyfile(card_src, dst)

            blk = "" if block_key is None else (
                f" block={'Y' if result['blocks_path'] else 'N'}")
            pbar.set_postfix_str(f"cats={len(cats)}{blk} labeled={n_labeled}")

    total = len(units)

    def pct(n):
        return 100 * n / total if total else 0.0

    # SPECIAL / NORMAL 은 장면 단위 배타 집계 - 카테고리가 여러 개 붙어도 1건이다
    print("\n===== UNITS =====")
    print(f"  {'SPECIAL (>=1 category)':28s} : {n_labeled:5d}  ({pct(n_labeled):5.1f}%)")
    if ask_blocking:
        for k in ("blocking_yes", "blocking_no"):
            print(f"    {k:26s} : {block_counts[k]:5d}  ({pct(block_counts[k]):5.1f}%)")
    print(f"  {'NORMAL (no category)':28s} : {total-n_labeled:5d}  "
          f"({pct(total-n_labeled):5.1f}%)")
    print(f"  {'TOTAL':28s} : {total:5d}  (100.0%)")

    def dump_categories(title, counter):
        print(f"\n===== {title} (multi-label) =====")
        if not counter:
            print("  (none)")
            return
        for cat, c in counter.most_common():
            print(f"  {cat:28s} : {c:5d}  ({pct(c):5.1f}% of units)")

    dump_categories("CATEGORY OCCURRENCES"
                    + (" - ALL" if ask_blocking else ""), cat_counts)
    if ask_blocking:
        dump_categories("CATEGORY OCCURRENCES - blocking_yes",
                        cat_counts_by_block["blocking_yes"])
        dump_categories("CATEGORY OCCURRENCES - blocking_no",
                        cat_counts_by_block["blocking_no"])

    if check_path and agree:
        n_chk = sum(agree.values())
        n_ok = agree[(True, True)] + agree[(False, False)]
        print("\n===== Q3 vs 3D GEOMETRY =====")
        print(f"  {'checked units':28s} : {n_chk:5d}  ({100*n_chk/total:5.1f}% have 3D labels)")
        print(f"  {'agree':28s} : {n_ok:5d}  ({100*n_ok/n_chk:5.1f}%)")
        print(f"  {'model Yes / geometry No':28s} : {agree[(True, False)]:5d}")
        print(f"  {'model No / geometry Yes':28s} : {agree[(False, True)]:5d}")
        print("  (disagreements are the review-priority units)")
    elif check_path and not ask_blocking:
        print("\n[info] 3D path check recorded, but Q3 is off so there is no "
              "model answer to compare against")

    if n_parse_fail:
        print(f"\n[warn] JSON parse failed on {n_parse_fail}/{total} units")

    print(f"\n[done] results -> {out_csv}  ({time.time()-t_start:.1f}s)")
    layout = "blocking_{yes,no}/<category>/" if ask_blocking else "<category>/"
    print(f"[done] visualizations -> {viz_path}/{layout}")
    return cat_counts

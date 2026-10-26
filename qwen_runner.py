#!/usr/bin/env python3
"""Qwen VLM 클립 추론 러너 (edge_case_mining.py 가 부른다).

--model 로 Qwen2.5-VL-7B, Qwen3-VL-8B/32B, Qwen3.8-27B 중 선택 가능
(edge_case_mining.py 의 choices 참고). 모델마다 클래스가 다르지만
AutoModelForImageTextToText / AutoProcessor 로 로드해 model_id 에 따라 알맞은
구현이 자동으로 선택되게 한다 - 모델을 바꿔도 이 파일을 고칠 필요가 없다.

판정 단위: 클립(uuid) 하나. 20초를 일정 fps 로 뽑은 프레임을 비디오 한 편으로
넣고 nuReasoning 단계별 JSON 을 받아 CSV 한 행으로 쓴다(run_clip_inference).
시각화는 <viz_dir>/.../<uuid>/{clip.mp4,result.json} (visualize_clip.py).
"""
import csv
import os
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import VideoMetadata

from edge_case_mining import (
    views_for, clip_path,
    build_nureasoning_prompt, parse_nureasoning_output,
    followup_time_spans_question, parse_time_spans_answer,
    sample_clip_frames, clip_intro,
    CLIP_FPS, CLIP_MAX_FRAMES, CLIP_MAX_LONG_SIDE,
)
from egomotion import (
    ego_clip_behavior, describe_clip_behavior,
)
from visualize_clip import render_clip_result, safe_dirname, VIZ_WIDTH
from constrained_tier import score_dirname, score_groups, group_key
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


def load_model(model_id: str):
    """model_id 에 맞는 모델/프로세서 클래스를 Auto* 로 자동 선택해 로드한다.

    Qwen2.5-VL 은 Qwen2_5_VLForConditionalGeneration, Qwen3-VL(dense 8B/32B)은
    Qwen3VLForConditionalGeneration 으로 클래스가 다르지만, 둘 다
    AutoModelForImageTextToText 로 커버되므로 여기서 분기할 필요가 없다.

    27B 는 device_map="auto" 로 여러 GPU 에 층을 나눠 올린다 - 그러면 이
    프로세스가 CUDA_VISIBLE_DEVICES 로 보이는 GPU 를 전부 쓰게 되므로, 8샤드
    병렬(샤드마다 GPU 1장) 구조로는 못 돌린다. run_labeled_27b.sh 는 처음부터
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


# nureasoning 은 6단계 서술을 모두 뱉으므로 256 토큰에서는 문장 중간에 잘린다.
# 실측(20260811, 50클립): 256 으로 두면 12/50(24%)이 JSON 미완성으로 파싱
# 실패했고, 잘린 위치는 모두 마지막 필드(rarity_assessment) 근처였다.
NUR_MAX_NEW_TOKENS = 800

# --difficulty 를 켜면 JSON 에 필드가 8개 늘어난다(4축 x 값+근거 문장).
# 800 은 그 여유가 없다 - 위 실측에서 이미 마지막 필드 근처에서 잘렸고,
# 난이도 필드는 그보다 더 뒤에 온다. 근거 문장 5개를 짧게 잡아도 약 200
# 토큰이 더 필요하므로 그만큼 올린다. 난이도를 끈 실행의 예산은 건드리지
# 않는다 - 예산이 바뀌면 기존 A/B 결과와 비교할 수 없게 된다.
NUR_MAX_NEW_TOKENS_DIFFICULTY = 1100
# --time-spans 가 붙이는 칸. 장르 2~3개에 구간 몇 개씩([시작, 끝, 점수])이면
# 100 토큰 안팎이지만, 모자라 JSON 이 잘리면 그 클립은 통째로 파싱 실패
# (Normal)가 되므로 넉넉히 둔다.
NUR_TIME_SPANS_EXTRA_TOKENS = 250
# --time-spans followup 의 이어 묻기 답 한도. 답이 time_spans JSON 하나뿐이라
# inline 의 추가분과 같은 크기면 된다.
NUR_FOLLOWUP_MAX_NEW_TOKENS = 300


# 클립 모드 CSV 의 열. 헤더와 빈 행이 같은 정의를 봐야 어긋나지 않는다.
CLIP_CSV_COLUMNS = [
    "uuid", "n_frames", "verdict", "n_categories",
    "categories", "parse_ok", "observation", "ego_behavior",
    "unusual_elements",
    # 묶음 점수(0~4). "Dynamic object=2|Driving environment=0" 형태로 한 칸에
    # 담는다 - 묶음마다 열을 만들면 scene_category.json 의 묶음을 고칠 때마다
    # 열 구성이 달라져 실행 간 병합이 깨진다. score_reason 은 묶음별 근거를
    # "Dynamic object: ... | Driving environment: ..." 로 이은 것.
    "group_scores", "score_reason",
    # 그중 최댓값. 시각화 폴더를 나누는 데만 쓴다.
    "tier_score",
    "ego_speed_kmh", "ego_motion", "ego_behavior_measured",
    # 결정 마진 - 이 클립의 판정이 얼마나 아슬아슬했는지.
    # margin_min 이 작을수록 의미 없는 프롬프트 섭동에도 뒤집힌다.
    "margin_min", "margin_mean", "n_close_tokens",
    # 난이도 4축. --difficulty 를 끈 실행에서는 전부 빈 칸이지만 열 자체는
    # 항상 쓴다 - 열 구성이 실행마다 달라지면 샤드 병합과 실행 간 비교가
    # 깨진다.
    *[c for k, _ in DIFFICULTY_AXES for c in (k, f"{k}_reason")],
    # --time-spans: 장르별 위험 구간 {"장르": [[시작, 끝, 점수], ...]} 의 JSON
    # (정수 초로 정리한 값 - edge_case_mining._snap_spans).
    # 끈 실행은 빈 칸. 맨 뒤에 붙인다 - 이 열이 생기기 전 실행의 CSV 는 헤더가
    # 달라 --resume 으로 이어 쓸 수 없다(qwen_runner 가 헤더를 대조해 멈춘다).
    "time_spans",
]


@torch.inference_mode()
def _continue_chat(model, processor, state, question, max_new_tokens=300):
    """생성이 끝난 대화(state)에 사용자 질문 하나를 이어 붙여 답을 받는다.

    1차 생성의 KV 캐시를 그대로 넘기므로 영상과 1차 프롬프트/답변은 다시
    계산하지 않고, 덧붙인 질문과 새 답만 계산한다(--time-spans followup).
    Qwen3-VL 의 3차원 위치(M-RoPE)는 1차 때 저장된 rope_deltas 로 이어진다 -
    transformers 가 캐시가 있는 생성을 "이어 생성"으로 보고 그 값을 쓴다.

    대화 틀은 chat template 과 같게 손으로 잇는다: 1차 답이 <|im_end|> 로
    끝났으면 그 뒤에, 출력 한도에 걸려 끊겼으면 <|im_end|> 를 먼저 닫는다.
    """
    tok = processor.tokenizer
    seq = state["sequences"]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    head = "" if int(seq[0, -1]) == im_end else "<|im_end|>"
    text = (head + "\n<|im_start|>user\n" + question
            + "<|im_end|>\n<|im_start|>assistant\n")
    add = tok(text, add_special_tokens=False,
              return_tensors="pt").input_ids.to(seq.device)
    ids = torch.cat([seq, add], dim=1)
    gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                         past_key_values=state["cache"],
                         max_new_tokens=max_new_tokens, do_sample=False)
    out = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
    del gen, ids, add
    return out.strip()


# 판정을 담는 JSON 키 - 이 값들이 바뀌면 채점 결과가 바뀐다.
DECISION_KEYS = ("scenario_types", "dynamic_object", "driving_environment",
                 "verdict", "scenario_type", "influenced_ego")


def _decision_steps(gen_ids, tokenizer):
    """생성 토큰 인덱스 중 '판정 값이 시작되는' 자리들의 집합.

    토큰을 하나씩 붙여가며 문자열을 재구성하고, DECISION_KEYS 뒤의 값이
    열리는 지점(따옴표/숫자/true/false 가 시작되는 곳)을 담은 토큰을 고른다.
    """
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
              as_video=False, video_fps=None,
              want_margin=False, thinking=False, state_out=None,
              attn_rec=None, attn_out=None):
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

    state_out 에 빈 dict 를 주면 생성이 끝난 대화(토큰 열 + KV 캐시)를 담아
    돌려준다 - _continue_chat 이 같은 대화에 질문을 이어 붙일 때 쓴다.
    출력 자체는 주지 않을 때와 같다(같은 탐욕 생성이고, 반환 형태만 다르다).
    """
    if as_video and images:
        return _generate_video(model, processor, images, text_prompt,
                               max_new_tokens=max_new_tokens,
                               video_fps=video_fps or 1.0,
                               want_margin=want_margin, thinking=thinking,
                               state_out=state_out, attn_rec=attn_rec,
                               attn_out=attn_out)

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
                         return_dict_in_generate=(want_margin
                                                  or state_out is not None),
                         output_scores=want_margin)
    margin = _decision_margin(gen, processor.tokenizer) if want_margin else None
    seq = gen.sequences if (want_margin or state_out is not None) else gen
    if state_out is not None:
        state_out["sequences"] = seq
        state_out["cache"] = gen.past_key_values
    trimmed = seq[:, inputs.input_ids.shape[1]:]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    del inputs, gen, trimmed
    torch.cuda.empty_cache()
    return (out.strip(), margin) if want_margin else out.strip()


@torch.inference_mode()
def _generate_video(model, processor, frames, text_prompt, max_new_tokens=256,
                    video_fps=1.0, want_margin=False,
                    thinking=False, state_out=None, attn_rec=None,
                    attn_out=None):
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

    # --attn-timeline: 이 generate 동안만 기록한다(이어 묻기 등 뒤의 생성은 뺀다).
    if attn_rec is not None:
        attn_rec.begin(inputs.input_ids,
                       processor.tokenizer.convert_tokens_to_ids("<|video_pad|>"),
                       float(video_fps))
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=False,
                         return_dict_in_generate=(want_margin
                                                  or state_out is not None),
                         output_scores=want_margin)
    margin = _decision_margin(gen, processor.tokenizer) if want_margin else None
    seq = gen.sequences if (want_margin or state_out is not None) else gen
    if state_out is not None:
        state_out["sequences"] = seq
        state_out["cache"] = gen.past_key_values
    trimmed = seq[:, inputs.input_ids.shape[1]:]
    if attn_rec is not None:
        rows = attn_rec.end()
        if attn_out is not None and rows is not None:
            attn_out.update(rows=rows, gen_ids=trimmed[0].tolist(),
                            times=attn_rec.times,
                            sink=attn_rec.sink_summary())
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    del inputs, gen, trimmed
    torch.cuda.empty_cache()
    return (out.strip(), margin) if want_margin else out.strip()


def build_clip_ego_facts(uuid, use_egomotion):
    """클립 모드용 egomotion 사실. (프롬프트 문구, 요약 dict) 를 돌려준다.

    한 프레임 시점의 스냅샷(속도 순간값)이 아니라 클립 전 구간을 한 번에
    요약한다 - 20초 영상을 통째로 넣으므로 그래야 시점이 어긋나지 않는다.

    describe_ego() 의 "is MOVING at 42 km/h" 를 여기서 쓰지 않는 이유:
    현재형 문장이 20초 영상 전체를 가리키는 것처럼 읽히지만 실제로는
    마지막 프레임의 값이다. 구간 요약이 속도 범위를 이미 말하므로
    중복이기도 하다.

    요약 한 문장만 넣는다. 1초 간격 속도/방향 시계열을 덧붙이던 --ego-track 은
    성능을 떨어뜨려 지웠다 (prompt_progress.md 참고).
    """
    if not use_egomotion:
        return "", None
    ch = ego_clip_behavior(uuid)
    return describe_clip_behavior(ch), ch


def run_clip_inference(uuids, labels, category_menu,
                       model_id="Qwen/Qwen3-VL-8B-Instruct",
                       out_csv="clip_results.csv",
                       use_egomotion=False,
                       single_view=True,
                       fps=CLIP_FPS, max_frames=CLIP_MAX_FRAMES,
                       max_long_side=CLIP_MAX_LONG_SIDE,
                       viz_dir=None, viz_only_edge=True, not_save_low=True,
                       viz_width=VIZ_WIDTH, video_input=True,
                       viz_normal=None, viz_special=None,
                       gt_labels=None, timeline=False,
                       want_margin=False, score_categories=True,
                       resume=False,
                       difficulty=False, difficulty_only=False,
                       explain_traj=False, traj=None,
                       viz_per_category=None, time_spans=None,
                       attn_timeline=False):
    """클립 전체(20초)를 fps 간격으로 넣어 클립 단위로 판정한다 (uuid 하나 = CSV 한 행).

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

    not_save_low=True(기본)면 거기서 한 번 더 거른다: 묶음 점수가 전부
    1 이하인 Special 클립은 저장하지 않는다 - 카테고리는 나열됐지만 자차에 영향이
    없다고 모델 스스로 판단한 경우다. 탐지(scenario_types, CSV)는 건드리지
    않고 시각화 대상만 줄인다 - 이 필터가 틀려도 재추론 없이 CSV 로 다시
    뽑을 수 있다.

    score_categories 는 5단계(묶음 점수)를 끈다. 끄면 프롬프트와 출력
    스키마에서 빠지고 CSV 에서 빈 칸이 되며 tier_score 도 비게 된다. 값이
    없으면 not_save_low 조건이 성립하지 않아 그 필터는 저절로 무력화된다.
    """
    model, processor = load_model(model_id)
    views = views_for(single_view)

    # --attn-timeline: 생성 토큰이 영상의 어느 시점을 봤는지 기록한다
    # (attn_timeline.py). 영상 입력일 때만 시간 칸을 나눌 수 있다.
    attn_rec = None
    if attn_timeline:
        if not video_input:
            print("[warn] --attn-timeline 은 영상 입력(--clip-no-video-input 아님)"
                  "에서만 동작한다 - 끔")
        else:
            from attn_timeline import AttnRecorder
            attn_rec = AttnRecorder(model)
            if not attn_rec.n_layers:
                print("[warn] --attn-timeline: 이 모델에서 텍스트 어텐션 층을 "
                      "찾지 못했다 - 끔")
                attn_rec.close()
                attn_rec = None
            else:
                print(f"[info] --attn-timeline: 어텐션 층 {attn_rec.n_layers}개에 "
                      f"기록 훅을 달았다")
    attn_dir = Path(out_csv).parent / "attn"

    n_special = sum(1 for l in labels if not l["is_normal"])
    print(f"[clip] {len(uuids)} clips | {len(views)} view(s) | {fps}fps "
          f"max {max_frames} frames @ {max_long_side}px | {n_special} categories")
    print(f"[clip] input mode: {'VIDEO (temporal merge + timestamps)' if video_input else 'image list'}")
    print(f"[clip] category scores: {'on' if score_categories else 'off'}")

    cat_counts = Counter()
    # --viz-per-category N: 카테고리마다 최초 N개 클립만 시각화한다.
    # 폴더는 score 가 아니라 카테고리 이름으로 나누고, 한 클립이 여러
    # 카테고리를 받으면 해당 폴더 전부에 중복 저장한다.
    viz_cat_counts = Counter()
    n_parse_fail = n_viz = n_edge = 0
    # 영상을 못 읽어 모델에 넣지 않은 클립. 파싱 실패와 따로 센다 - 둘은
    # 원인도 대처도 다르다(이쪽은 --resume 으로 다시 돌릴 대상이다).
    n_no_frames = n_partial = 0
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

    # resume 이면 기존 파일 뒤에 덧붙인다. "w" 로 열면 여는 순간 그때까지의
    # 결과가 지워진다 - 이어받기의 목적 자체가 사라진다.
    #
    # 헤더는 파일이 없거나 비었을 때만 쓴다. 이미 있는 파일의 헤더가 지금
    # 열 구성과 다르면 쓰기 전에 멈춘다 - 덧붙이면 한 파일 안에서 같은
    # 칸의 뜻이 행마다 달라진다. (main 이 먼저 대조하지만, 이 함수를
    # 직접 부르는 경로가 있어 여기서도 막는다.)
    append = False
    if resume and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        with open(out_csv, newline="", encoding="utf-8") as fh:
            head = next(csv.reader(fh), [])
        if head != CLIP_CSV_COLUMNS:
            raise RuntimeError(
                f"{out_csv} 의 열 구성이 지금 코드와 달라 이어 쓸 수 없습니다 "
                f"({len(head)}열 vs {len(CLIP_CSV_COLUMNS)}열)")
        append = True
        # 쓰다가 죽은 프로세스가 마지막 줄을 줄바꿈 없이 남겼으면, 그 뒤에
        # 바로 붙여 쓰면 두 행이 한 줄로 합쳐진다. 준비 단계가 잘린 행을
        # 지우지만, 끝에 줄바꿈이 없는 경우까지 여기서 한 번 더 막는다.
        with open(out_csv, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            ends_nl = fh.read(1) in (b"\n", b"\r")
        if not ends_nl:
            with open(out_csv, "a", newline="", encoding="utf-8") as fh:
                fh.write("\r\n")

    with open(out_csv, "a" if append else "w", newline="",
              encoding="utf-8") as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow(CLIP_CSV_COLUMNS)

        pbar = tqdm(uuids, total=len(uuids), unit="clip", dynamic_ncols=True,
                    mininterval=1.0, smoothing=0.1)
        for uuid in pbar:
            clip = sample_clip_frames(uuid, fps=fps, max_frames=max_frames,
                                      max_long_side=max_long_side, views=views,
                                      traj=traj)
            images = [im for _, _, im in clip["frames"]]
            # 요청한 장수를 다 읽지 못했으면 모델에 넣지 않는다.
            #
            # 디코딩이 중간에 실패하면 그때까지 읽은 프레임만 돌아온다. 그대로
            # 넣으면 모델은 클립 뒷부분을 못 본 채 판정하고, CSV 에는 그 장수만
            # 적혀 짧은 클립(19.3초 -> 39장, 정상)과 구분되지 않는다. 요청
            # 장수는 클립 길이에서 나오므로(sample_clip_indices) 그것과 대조하면
            # 짧은 클립은 통과하고 끊긴 읽기만 걸린다. 실측(NAS 실행 두 개):
            # 40 이 아닌 149건이 전부 짧은 클립이었고 끊긴 읽기는 0건이었다 -
            # 지금까지는 없었지만 생겨도 알 방법이 없었다.
            #
            # 둘 다 n_frames=0 으로 적는다. --resume 은 n_frames > 0 만 끝난
            # 것으로 보므로(resume_run.py) 따로 규칙을 두지 않아도 다시 돌릴
            # 대상이 된다. 사유는 observation 칸에 남겨 열 구성을 바꾸지 않는다.
            expected = len(clip["indices"]) * len(views)
            if len(images) < expected:
                why = ("(no frames)" if not images else
                       f"(partial frames {len(images)}/{expected})")
                # 빈 칸 개수를 손으로 세지 않는다 - 열이 늘 때마다 어긋난다.
                row = [uuid, 0, "Normal", 0, "", 0, why]
                writer.writerow(row + [""] * (len(CLIP_CSV_COLUMNS) - len(row)))
                f.flush()
                if images:
                    n_partial += 1
                    tqdm.write(f"[warn] {uuid}: {why} - 추론하지 않고 실패로 기록")
                else:
                    n_no_frames += 1
                continue

            # egomotion 은 클립 전 구간을 요약한다 (시간축 정렬).
            behavior, ego = build_clip_ego_facts(uuid, use_egomotion)

            prompt = build_nureasoning_prompt(
                category_menu, labels, behavior_facts=behavior,
                intro=clip_intro(len(images), len(views), fps=fps,
                                 as_video=video_input),
                timeline=timeline,
                score_categories=score_categories,
                difficulty=difficulty,
                difficulty_only=difficulty_only,
                explain_traj=explain_traj,
                time_spans=(time_spans == "inline"))
            # followup 은 1차 대화(토큰 열 + KV 캐시)를 받아 두었다가 질문을
            # 이어 붙인다. 1차 생성 자체는 받지 않을 때와 같다.
            fu_state = {} if time_spans == "followup" else None
            gen_out = _generate(model, processor, images, prompt,
                                max_new_tokens=(NUR_MAX_NEW_TOKENS_DIFFICULTY
                                                if difficulty
                                                else NUR_MAX_NEW_TOKENS)
                                + (NUR_TIME_SPANS_EXTRA_TOKENS
                                   if time_spans == "inline" else 0),
                                as_video=video_input, video_fps=fps,
                                want_margin=want_margin,
                                thinking=(model_id in THINKING_MODELS),
                                state_out=fu_state,
                                attn_rec=attn_rec,
                                attn_out=(attn_out := {}))
            raw, margin = gen_out if want_margin else (gen_out, None)
            result = parse_nureasoning_output(raw, labels)
            # 장르를 댄 클립만 곡선을 만든다(관심 토큰이 장르 이름과 점수다).
            if attn_out.get("rows") is not None and result["categories"]:
                from attn_timeline import curves_for, save_npz, default_layers
                rows_ = attn_out["rows"]
                gk = {g: group_key(g) for g in (score_groups(labels)
                                                if score_categories else [])}
                curves = curves_for(rows_, attn_out["gen_ids"],
                                    processor.tokenizer, result["categories"], gk)
                curves["times"] = attn_out["times"]
                curves["layers"] = default_layers(rows_.shape[1])
                result["attn_curves"] = curves
                save_npz(attn_dir / f"{uuid}.npz", rows_, attn_out["gen_ids"],
                         attn_out["times"], curves["layers"],
                         sink=attn_out.get("sink"))
            if fu_state is not None:
                # 장르를 하나도 대지 않았으면 물을 대상이 없다.
                if result["categories"] and fu_state:
                    q = followup_time_spans_question(
                        bool(score_categories and score_groups(labels)))
                    ans = _continue_chat(model, processor, fu_state, q,
                                         max_new_tokens=NUR_FOLLOWUP_MAX_NEW_TOKENS)
                    (result["time_spans"], result["time_spans_raw"],
                     result["time_seen"]) = parse_time_spans_answer(
                        ans, labels, set(result["categories"]))
                    result["time_spans_answer"] = ans
                fu_state.clear()
                torch.cuda.empty_cache()

            cats = result["categories"]
            cat_counts.update(cats)
            n_edge += bool(cats)
            n_parse_fail += not result["parse_ok"]

            writer.writerow([
                uuid, len(images), result["verdict"],
                len(cats), "|".join(cats), int(result["parse_ok"]),
                result.get("observation", ""), result.get("ego_behavior", ""),
                result.get("unusual_elements", ""),
                # {"Dynamic object": 2} -> "Dynamic object=2"
                "|".join(f"{k}={v}" for k, v in
                         (result.get("group_scores") or {}).items()),
                result.get("score_reason", ""),
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
                (json.dumps(result["time_spans"], ensure_ascii=False)
                 if result.get("time_spans") else ""),
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
                    # 묶음 점수가 전부 1 이하면 모델이 "카테고리는 있지만 자차에
                    # 영향 없음"으로 본 것이다. 점수를 끈 실행에서는 tier_score
                    # 가 None 이라 이 조건이 성립하지 않아 필터가 저절로
                    # 무력화된다.
                    top = result.get("tier_score")
                    want_viz = top is None or top > 1
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
                    out_dirs = [viz_path / safe_dirname(c) / uuid
                                for c in targets]
                else:
                    # <viz_dir>/{normal,special}/score_<N>/<uuid>/ 로 나눈다.
                    # 묶음 점수의 최댓값(0~4). 점수를 끈 실행이면 score_unknown.
                    bucket = "special" if is_special else "normal"
                    score_dir = score_dirname(result.get("tier_score"))
                    out_dirs = [viz_path / bucket / score_dir / uuid
                                if (viz_normal is not None or viz_special is not None)
                                else viz_path / score_dir / uuid]
                # 시각화가 "구간을 물었는데 못 받은 장르"를 빈 줄로 그리게 한다.
                result["_time_spans_on"] = bool(time_spans)
                result["_time_spans_mode"] = time_spans
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
    if n_no_frames or n_partial:
        print(f"[warn] 영상을 다 읽지 못해 추론하지 않은 클립: "
              f"프레임 없음 {n_no_frames} / 일부만 읽힘 {n_partial} - "
              f"CSV 에 n_frames=0 으로 남았고 --resume 으로 다시 돌릴 수 있다")
    print(f"[clip] SPECIAL (>=1 category): {n_edge}/{n} ({100*n_edge/n:.1f}%)")
    print(f"[clip] NORMAL   (no category)  : {n-n_edge}/{n} "
          f"({100*(n-n_edge)/n:.1f}%)")
    print("[clip] categories:")
    for c, v in cat_counts.most_common():
        print(f"   {c:32} {v:4d}  ({100*v/n:5.1f}%)")
    if viz_path:
        print(f"[clip] videos written: {n_viz}")
    print(f"[saved] {out_csv}")


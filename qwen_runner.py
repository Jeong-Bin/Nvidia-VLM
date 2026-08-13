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
from egomotion import ego_state, describe_ego, ego_behavior_change, describe_behavior
from obstacle import obstacle_summary, describe_obstacles, path_intrusion
from visualize import render_scene_card
from visualize_clip import render_clip_result, VIZ_WIDTH
from constrained_tier import make_tier_processor, tier_label, score_dirname


def load_model(model_id: str):
    """model_id 에 맞는 모델/프로세서 클래스를 Auto* 로 자동 선택해 로드한다.

    Qwen2.5-VL 은 Qwen2_5_VLForConditionalGeneration, Qwen3-VL(dense 8B/32B)은
    Qwen3VLForConditionalGeneration 으로 클래스가 다르지만, 둘 다
    AutoModelForImageTextToText 로 커버되므로 여기서 분기할 필요가 없다.
    """
    print(f"[model] loading {model_id} ...")
    t0 = time.time()
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


def _lp_list(proc):
    """LogitsProcessor 하나를 generate 가 받는 리스트 형태로 감싼다."""
    if proc is None:
        return None
    return LogitsProcessorList([proc])


@torch.inference_mode()
def _generate(model, processor, images, text_prompt, max_new_tokens=256,
              as_video=False, video_fps=None, logits_processor=None):
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
                               logits_processor=logits_processor)

    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": text_prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    proc_kwargs = dict(text=[text], padding=True, return_tensors="pt")
    if images:
        proc_kwargs["images"] = images
    inputs = processor(**proc_kwargs).to(model.device)

    gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=False,
                         logits_processor=_lp_list(logits_processor))
    trimmed = gen[:, inputs.input_ids.shape[1]:]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    del inputs, gen, trimmed
    torch.cuda.empty_cache()
    return out.strip()


@torch.inference_mode()
def _generate_video(model, processor, frames, text_prompt, max_new_tokens=256,
                    video_fps=1.0, logits_processor=None):
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
                                         add_generation_prompt=True)
    inputs = processor(text=[text], videos=[arr], video_metadata=[meta],
                       do_sample_frames=False, padding=True,
                       return_tensors="pt").to(model.device)

    gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=False,
                         logits_processor=_lp_list(logits_processor))
    trimmed = gen[:, inputs.input_ids.shape[1]:]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    del inputs, gen, trimmed
    torch.cuda.empty_cache()
    return out.strip()


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
                       constrain_tiers=True):
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

    not_save_low=True(기본)면 거기서 한 번 더 거른다: Safety Criticality 와
    Rarity 가 둘 다 "Low" 인 클립은 저장하지 않는다 - 카테고리는 나열됐지만
    자차에 영향도 없고(Q3 폐지 대신 4단계가 이 역할) 희귀하지도 않다고 모델
    스스로 판단한 경우다. 탐지(scenario_types, CSV)는 건드리지 않고 시각화
    대상만 줄인다 - 이 필터가 틀려도 재추론 없이 CSV 로 다시 뽑을 수 있다.
    """
    model, processor = load_model(model_id)
    views = views_for(single_view)

    # 등급 필드를 1/2/3 정수로만 나오게 디코딩 단계에서 막는다.
    # 프롬프트 지시만으로는 "Low to moderate" 류가 새어나왔다(실측 20260811).
    tier_proc = (make_tier_processor(processor.tokenizer)
                 if constrain_tiers else None)

    n_special = sum(1 for l in labels if not l["is_normal"])
    print(f"[clip] {len(uuids)} clips | {len(views)} view(s) | {fps}fps "
          f"max {max_frames} frames @ {max_long_side}px | {n_special} categories")
    print(f"[clip] input mode: {'VIDEO (temporal merge + timestamps)' if video_input else 'image list'}")
    print(f"[clip] tier constraint: {'ON (1/2/3 enforced at decode)' if constrain_tiers else 'OFF'}")

    cat_counts = Counter()
    n_parse_fail = n_viz = n_edge = 0
    viz_path = Path(viz_dir) if viz_dir else None
    if viz_path:
        viz_path.mkdir(parents=True, exist_ok=True)
        print(f"[clip] viz -> {viz_path}/score_<N>/<uuid>/{{clip.mp4,result.json}}"
              + ("  (clips with >=1 category only)" if viz_only_edge
                 else "  (all clips)")
              + f"  width {viz_width or 'original'}")
    t_start = time.time()

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "uuid", "n_frames", "verdict", "n_categories",
            "categories", "parse_ok", "observation", "ego_behavior",
            "unusual_elements", "safety_tier", "safety_label",
            "safety_reason", "rarity_tier", "rarity_label", "rarity_reason",
            "tier_score",
            "ego_speed_kmh", "ego_motion", "ego_behavior_measured",
        ])

        pbar = tqdm(uuids, total=len(uuids), unit="clip", dynamic_ncols=True,
                    mininterval=1.0, smoothing=0.1)
        for uuid in pbar:
            clip = sample_clip_frames(uuid, fps=fps, max_frames=max_frames,
                                      max_long_side=max_long_side, views=views)
            images = [im for _, _, im in clip["frames"]]
            if not images:
                writer.writerow([uuid, 0, "Normal", 0, "", 0,
                                 "(no frames)", "", "", "", "", "", "",
                                 "", "", "", "", "", ""])
                n_parse_fail += 1
                continue

            # 마지막 프레임 시점 = 모델이 보는 가장 최근 순간
            last_idx = clip["indices"][-1]
            facts, ego = build_sensor_facts(uuid, last_idx, use_egomotion,
                                            use_obstacle, n_views=len(views))
            behavior, _ = build_behavior_facts(uuid, last_idx, use_egomotion)

            prompt = build_nureasoning_prompt(
                category_menu, facts, behavior_facts=behavior,
                intro=clip_intro(len(images), len(views), fps=fps,
                                 as_video=video_input))
            raw = _generate(model, processor, images, prompt,
                            max_new_tokens=NUR_MAX_NEW_TOKENS,
                            as_video=video_input, video_fps=fps,
                            logits_processor=tier_proc)
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
                f"{ego['speed_kmh']:.1f}" if ego else "",
                ego["motion"] if ego else "",
                behavior,
            ])
            f.flush()
            pbar.set_postfix_str(
                f"{'EDGE' if cats else '----'} cats={len(cats)}")

            want_viz = cats or not viz_only_edge
            if want_viz and not_save_low and cats:
                # 카테고리가 없는 클립(viz_only_edge=False 로 전량 저장할 때)은
                # safety/rarity 도 의미가 없으므로 이 필터를 적용하지 않는다.
                both_low = (result.get("safety_tier") == 1
                           and result.get("rarity_tier") == 1)
                want_viz = not both_low
            if viz_path and want_viz:
                # 검수 편의를 위해 safety+rarity 합계로 폴더를 나눈다
                # (2~6, 못 읽으면 score_unknown).
                score_dir = score_dirname(result.get("tier_score"))
                render_clip_result(
                    uuid, clip_path(views[0], uuid), result,
                    viz_path / score_dir / uuid, max_width=viz_width,
                    extra={"ego_behavior_measured": behavior,
                           "n_frames_seen": len(images),
                           "ego_speed_kmh": (round(ego["speed_kmh"], 1)
                                             if ego else None)})
                n_viz += 1

    dt = time.time() - t_start
    n = max(len(uuids), 1)
    print(f"\n[clip] done: {len(uuids)} clips in {dt/60:.1f} min "
          f"({dt/n:.1f} s/clip)")
    if n_parse_fail:
        print(f"[warn] JSON parse failed on {n_parse_fail} clips")
    print(f"[clip] EDGE-CASE (>=1 category): {n_edge}/{n} ({100*n_edge/n:.1f}%)")
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

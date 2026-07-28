#!/usr/bin/env python3
"""Qwen2.5-VL 2단계 추론 러너.

판정 단위: (uuid, frame_idx) - 클립 내 특정 순간의 3뷰 프레임 세트.

1단계: 그 순간(3뷰 프레임) -> 한 문장 캡션
2단계: 캡션 -> top-3 (scenario, category) 매칭 (없으면 OOD)

결과를 CSV로 저장하고, special 로 분류된 순간은 시각화 이미지도 만든다.
"""
import csv
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from edge_case_mining import (
    sample_unit_frames, FRONT_VIEWS, category_slug,
    build_match_prompt, parse_match_output,
    build_caption_prompt, apply_ego_correction,
)
from egomotion import ego_state, describe_ego
from visualize import render_scene_card


def load_model(model_id: str):
    print(f"[model] loading {model_id} ...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda:0")
    processor = AutoProcessor.from_pretrained(model_id, max_pixels=768 * 768)
    model.eval()
    print(f"[model] loaded in {time.time()-t0:.1f}s")
    return model, processor


@torch.inference_mode()
def _generate(model, processor, images, text_prompt, max_new_tokens=128):
    """images 가 비어있으면 순수 텍스트 프롬프트로 생성 (2단계 매칭용)."""
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

    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = gen[:, inputs.input_ids.shape[1]:]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    del inputs, gen, trimmed
    torch.cuda.empty_cache()
    return out.strip()


def classify_unit(model, processor, unit_frames, caption_prompt, label_menu):
    """판정 단위 하나(직전 3뷰 + 현재 3뷰 = 6장)에 대해 (caption, match_raw) 반환.

    1단계: prev(3장) + cur(3장) 순서로 넣어 움직임 포함 캡션 생성.
    2단계: 그 캡션 텍스트만으로 매칭 (이미지를 다시 넣으면 모델이 캡션을
           무시하고 이미지에서 재판단해 버림).
    """
    prev, cur = unit_frames["prev"], unit_frames["cur"]
    images = ([prev[v] for v in FRONT_VIEWS if prev.get(v) is not None]
              + [cur[v] for v in FRONT_VIEWS if cur.get(v) is not None])

    caption = _generate(model, processor, images, caption_prompt, max_new_tokens=90)

    match_prompt = build_match_prompt(caption, label_menu)
    match_raw = _generate(model, processor, [], match_prompt, max_new_tokens=256)

    return caption, match_raw


def run_inference(units, labels, label_menu, caption_hint,
                  model_id="Qwen/Qwen2.5-VL-7B-Instruct",
                  out_csv="edge_case_results.csv", viz_dir="viz",
                  use_egomotion=True):
    """units: [(uuid, frame_idx), ...]

    시각화는 viz_dir/<category_slug>/ 하위에 카테고리별로 분리 저장한다.
    top1 이 special 카테고리 또는 OOD 인 경우에만 저장하고, normal 은 저장하지 않는다.

    use_egomotion=True 면 판정 단위마다 egomotion 라벨에서 자차 운동 상태를 읽어
    (a) 1단계 캡션 프롬프트에 사실로 주입하고 (b) 2단계 normal 오분류를 교정한다.
    라벨이 없는 클립은 자동으로 기존 동작(이미지만으로 추론)으로 넘어간다.
    """
    model, processor = load_model(model_id)
    viz_path = Path(viz_dir)
    viz_path.mkdir(parents=True, exist_ok=True)

    # egomotion 을 안 쓰면 프롬프트가 매번 같으므로 한 번만 만든다
    base_prompt = build_caption_prompt(caption_hint)

    counts = Counter()  # top-1 category 기준 집계
    n_special_hits = 0
    n_ego = 0          # egomotion 사실을 주입한 단위 수
    n_corrected = 0    # normal 카테고리가 교정된 단위 수
    t_start = time.time()
    n = len(units)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "uuid", "frame_idx", "caption",
            "top1_scenario", "top1_category", "top1_confidence",
            "top2_scenario", "top2_category", "top2_confidence",
            "top3_scenario", "top3_category", "top3_confidence",
            "ego_speed_kmh", "ego_motion", "ego_corrected",
        ])

        pbar = tqdm(units, total=n, unit="unit", dynamic_ncols=True,
                   mininterval=1.0, smoothing=0.1)
        for uuid, frame_idx in pbar:
            unit_frames = sample_unit_frames(uuid, frame_idx)

            ego = ego_state(uuid, frame_idx) if use_egomotion else None
            if ego is not None:
                n_ego += 1
                prompt = build_caption_prompt(caption_hint, describe_ego(ego))
            else:
                prompt = base_prompt

            corrected = False
            if not any(unit_frames["cur"].values()):
                caption, matches = "(no frame)", []
            else:
                caption, match_raw = classify_unit(
                    model, processor, unit_frames, prompt, label_menu
                )
                matches = parse_match_output(match_raw, labels)
                matches, corrected = apply_ego_correction(matches, ego)
                n_corrected += corrected

            top1 = matches[0]["category"] if matches else "OOD"
            top1_is_special = bool(matches) and not matches[0]["is_normal"]
            top1_is_ood = not matches
            counts[top1] += 1
            if top1_is_special or top1_is_ood:
                n_special_hits += 1

            row = [uuid, frame_idx, caption]
            for j in range(3):
                if j < len(matches):
                    m = matches[j]
                    row += [m["scenario"], m["category"], f"{m['confidence']:.2f}"]
                else:
                    row += ["", "", ""]
            row += [f"{ego['speed_kmh']:.1f}" if ego else "",
                    ego["motion"] if ego else "",
                    "1" if corrected else ""]
            writer.writerow(row)
            f.flush()

            # top1이 special 카테고리 또는 OOD 이면 시각화 카드 생성
            # (현재 시점 cur 3뷰를 표시; 움직임 정보는 캡션에 녹아 있음)
            # 카테고리별 하위 폴더에 분리 저장: viz_dir/<category_slug>/{uuid}_f{frame_idx}.png
            if top1_is_special or top1_is_ood:
                cat_dir = viz_path / category_slug(top1)
                cat_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"{uuid}_f{frame_idx:04d}.png"
                render_scene_card(uuid, frame_idx, unit_frames["cur"], caption, matches,
                                  out_path=cat_dir / out_name)

            pbar.set_postfix_str(f"last={top1[:20]}  special={n_special_hits}")

    total = sum(counts.values())
    print("\n===== TOP-1 CATEGORY COUNTS =====")
    for cat, c in counts.most_common():
        pct = 100 * c / total if total else 0.0
        print(f"  {cat:28s} : {c:5d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':28s} : {total:5d}  (100.0%)")
    if use_egomotion:
        print(f"\n[egomotion] fact injected on {n_ego}/{total} units "
              f"({100*n_ego/total if total else 0:.1f}%), "
              f"normal category corrected on {n_corrected} units")
    print(f"\n[done] results -> {out_csv}  ({time.time()-t_start:.1f}s)")
    print(f"[done] visualizations -> {viz_path}/")
    return counts

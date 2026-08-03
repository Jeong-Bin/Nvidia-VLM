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

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from edge_case_mining import (
    sample_unit_frames, FRONT_VIEWS,
    build_vlm_prompt, parse_vlm_output, unit_name, viz_targets, blocking_dir,
)
from egomotion import ego_state, describe_ego
from obstacle import obstacle_summary, describe_obstacles
from visualize import render_scene_card


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


@torch.inference_mode()
def _generate(model, processor, images, text_prompt, max_new_tokens=256):
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


def build_sensor_facts(uuid, frame_idx, use_egomotion, use_obstacle):
    """프롬프트에 넣을 센서 사실 문구와, 시각화/CSV 용 원본 상태를 함께 반환."""
    lines, ego = [], None
    if use_egomotion:
        ego = ego_state(uuid, frame_idx)
        if ego is not None:
            lines.append(describe_ego(ego))
    if use_obstacle:
        s = describe_obstacles(obstacle_summary(uuid, frame_idx))
        if s:
            lines.append(s)
    return "\n".join(lines), ego


def classify_unit(model, processor, unit_frames, prompt):
    """직전 3뷰 + 현재 3뷰(6장)를 한 번에 넣어 모델 원문 출력을 받는다."""
    prev, cur = unit_frames["prev"], unit_frames["cur"]
    images = ([prev[v] for v in FRONT_VIEWS if prev.get(v) is not None]
              + [cur[v] for v in FRONT_VIEWS if cur.get(v) is not None])
    return _generate(model, processor, images, prompt)


def run_inference(units, labels, category_menu,
                  model_id="Qwen/Qwen2.5-VL-7B-Instruct",
                  out_csv="edge_case_results.csv", viz_dir="viz",
                  use_egomotion=False, use_obstacle=False):
    """units: [(uuid, frame_idx), ...]"""
    model, processor = load_model(model_id)
    viz_path = Path(viz_dir)
    viz_path.mkdir(parents=True, exist_ok=True)

    # 센서 사실을 안 쓰면 프롬프트가 매번 같으므로 한 번만 만든다
    base_prompt = build_vlm_prompt(category_menu)
    use_sensors = use_egomotion or use_obstacle

    # 멀티라벨이라 카테고리 합계는 판정 단위 수를 넘을 수 있다.
    # blocking 여부로 한 번 더 쪼개서 센다.
    cat_counts = Counter()
    cat_counts_by_block = {"blocking_yes": Counter(), "blocking_no": Counter()}
    block_counts = Counter()   # 카테고리가 붙은 단위의 blocking yes/no
    n_labeled = 0
    n_parse_fail = 0
    t_start = time.time()

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "uuid", "frame_idx", "verdict", "blocks_path", "n_categories",
            "categories", "evidence", "parse_ok", "ego_speed_kmh", "ego_motion",
        ])

        pbar = tqdm(units, total=len(units), unit="unit", dynamic_ncols=True,
                    mininterval=1.0, smoothing=0.1)
        for uuid, frame_idx in pbar:
            unit_frames = sample_unit_frames(uuid, frame_idx)

            ego = None
            if not any(unit_frames["cur"].values()):
                result = {"verdict": "Normal", "categories": [],
                          "blocks_path": False, "evidence": "(no frame)",
                          "parse_ok": False}
            else:
                if use_sensors:
                    facts, ego = build_sensor_facts(
                        uuid, frame_idx, use_egomotion, use_obstacle)
                    prompt = build_vlm_prompt(category_menu, facts) if facts else base_prompt
                else:
                    prompt = base_prompt
                raw = classify_unit(model, processor, unit_frames, prompt)
                result = parse_vlm_output(raw, labels)

            cats = result["categories"]
            block_key = blocking_dir(result)
            cat_counts.update(cats)
            n_parse_fail += not result["parse_ok"]
            if cats:
                n_labeled += 1
                block_counts[block_key] += 1
                cat_counts_by_block[block_key].update(cats)

            writer.writerow([
                uuid, frame_idx, result["verdict"],
                "Yes" if result["blocks_path"] else "No",
                len(cats), "|".join(cats), result["evidence"],
                int(result["parse_ok"]),
                f"{ego['speed_kmh']:.1f}" if ego else "",
                ego["motion"] if ego else "",
            ])
            f.flush()

            # blocking_{yes,no}/<category>/<uuid>_f<idx>/ 아래에 저장.
            # 멀티라벨이면 해당하는 모든 카테고리 폴더에 같은 내용을 중복 저장한다.
            targets = viz_targets(result)
            if targets:
                payload = {
                    "uuid": uuid, "frame_idx": frame_idx,
                    "verdict": result["verdict"],
                    "categories": cats,
                    "blocks_path": result["blocks_path"],
                    "evidence": result["evidence"],
                }
                if ego:
                    payload["ego"] = {"speed_kmh": round(ego["speed_kmh"], 1),
                                      "motion": ego["motion"]}
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

            pbar.set_postfix_str(
                f"cats={len(cats)} block={'Y' if result['blocks_path'] else 'N'} "
                f"labeled={n_labeled}")

    total = len(units)

    def pct(n):
        return 100 * n / total if total else 0.0

    print("\n===== UNITS =====")
    print(f"  {'total':28s} : {total:5d}  (100.0%)")
    print(f"  {'with >=1 category':28s} : {n_labeled:5d}  ({pct(n_labeled):5.1f}%)")
    for k in ("blocking_yes", "blocking_no"):
        print(f"    {k:26s} : {block_counts[k]:5d}  ({pct(block_counts[k]):5.1f}%)")

    def dump_categories(title, counter):
        print(f"\n===== {title} (multi-label) =====")
        if not counter:
            print("  (none)")
            return
        for cat, c in counter.most_common():
            print(f"  {cat:28s} : {c:5d}  ({pct(c):5.1f}% of units)")

    dump_categories("CATEGORY OCCURRENCES - ALL", cat_counts)
    dump_categories("CATEGORY OCCURRENCES - blocking_yes",
                    cat_counts_by_block["blocking_yes"])
    dump_categories("CATEGORY OCCURRENCES - blocking_no",
                    cat_counts_by_block["blocking_no"])

    if n_parse_fail:
        print(f"\n[warn] JSON parse failed on {n_parse_fail}/{total} units")

    print(f"\n[done] results -> {out_csv}  ({time.time()-t_start:.1f}s)")
    print(f"[done] visualizations -> {viz_path}/blocking_{{yes,no}}/<category>/")
    return cat_counts

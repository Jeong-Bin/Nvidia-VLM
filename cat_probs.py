#!/usr/bin/env python3
"""1차 답의 scenario_types 자리에서 장르마다 "그 이름이 나올 확률"을 잰다 (--cat-probs).

목적: recheck 후보를 키워드 목록(recheck_keywords) 대신 모델 자신의 확신도로
고르기 위해서다. 키워드 방식은 1차 장면 설명에 그 단어가 나와야만 후보가 되어,
프롬프트가 바뀌어 설명 표현이 달라지면 후보가 흔들렸다(실측 20261021_142430
inline: 설명에서 "pedestrian" 이 빠진 클립 2개는 recheck 가 보행자를 묻지도
않았다). 장르가 늘 때마다 키워드를 손으로 관리해야 하는 것도 문제였다.

어떻게 재나 (1차 출력은 바꾸지 않는다):
  1차 생성이 끝난 뒤 그 대화의 KV 캐시를 scenario_types 목록의 각 "자리"까지
  되돌리고(crop), 그 자리에서 장르 이름이 이어질 확률을 teacher forcing 으로
  계산한다.
    자리 = 목록 처음(`[` 바로 뒤) + 이미 적은 항목 각각의 뒤
    목록 처음:   ... "scenario_types": [ + "<장르>"
    항목 뒤:     ... "Animal" + , "<장르>"
  장르 이름은 여러 토큰이라 이름 전체의 확률(토큰 확률의 곱)을 쓴다 - 첫
  토큰만 보면 "Unknown but abnormal moving object" 와 "... driving environment"
  처럼 첫 토큰이 같은 장르를 가를 수 없다.
  장르의 확신도 = 여러 자리 중 가장 큰 확률. 이미 목록에 적은 장르는 1.0.

  토큰 경계: 모델이 `["` 나 `",` 처럼 괄호/따옴표/쉼표를 한 토큰으로 냈을 수
  있어, 자리 글자 위치가 토큰 중간일 수 있다. 그 자리 앞의 마지막 토큰 경계까지
  캐시를 되돌리고, 경계부터 자리까지의 글자(carry)를 이어 붙일 글 앞에 함께
  넣어 확률을 잰다.

비용: 장르 수 x 자리 수(보통 1~3) 번의 짧은 forward(10 토큰 안팎). 영상은 다시
계산하지 않는다. 클립당 1초 미만.

해석: 이 확률도 모델이 이미 쓴 장면 설명/요소를 보고 정해진다. 설명에 아예
언급되지 않은 요소는 확률도 거의 0 일 수 있다 - 그런 놓침은 이 방식으로도
후보에 오르지 않는다.
"""
from __future__ import annotations

import bisect
import json
import math
import re
from pathlib import Path

import torch

from attn_timeline import token_char_spans


def _crop(cache, n: int):
    for layer in cache.layers:
        layer.crop(n)


@torch.inference_mode()
def category_probs(model, tokenizer, state: dict, prompt_len: int,
                   categories: list[str]) -> dict | None:
    """{"probs": {장르: 확률}, "slots": 자리 수}. scenario_types 를 못 찾으면 None.

    state: qwen_runner._generate(state_out=...) 가 담은 {"sequences", "cache"}.
    이 함수는 캐시를 잘라 쓰므로 부른 뒤에는 그 캐시로 생성을 이어 갈 수 없다
    (이어 묻기 등은 먼저 끝내야 한다).
    """
    seq = state["sequences"][0]
    cache = state["cache"]
    gen = seq[prompt_len:].tolist()
    text, spans = token_char_spans(gen, tokenizer)
    m = re.search(r'"scenario_types"\s*:\s*\[', text)
    if not m:
        return None
    lb = m.end()                                   # '[' 바로 뒤
    rb = text.find("]", lb)
    if rb < 0:
        return None
    region = text[lb:rb]
    # 이미 적은 항목과 각 항목의 닫는 따옴표 뒤 위치
    items = [(mm.group(1), lb + mm.end()) for mm in re.finditer(r'"([^"]*)"', region)]
    named = {n.strip().lower() for n, _ in items}
    slots = [(lb, False)] + [(end, True) for _, end in items]

    named_keep = {c for c in categories if c.lower() in named}
    todo = [c for c in categories if c not in named_keep]
    logs = {c: -math.inf for c in todo}
    # 뒤 자리부터 처리한다 - 캐시를 앞쪽으로만 자르면 되므로.
    for pos, after_item in sorted(slots, key=lambda s: -s[0]):
        # 자리 앞에서 끝나는 토큰 수. 토큰 끝 위치는 앞에서부터 줄지 않으므로
        # 끝이 pos 이하인 마지막 토큰까지가 경계다.
        k = bisect.bisect_right([e for _, e in spans], pos)
        bound_char = spans[k - 1][1] if k > 0 else 0
        carry = text[bound_char:pos]
        n_prefix = prompt_len + k                  # 캐시에 남길 길이 + 마지막 토큰
        last_tok = seq[n_prefix - 1:n_prefix]
        for c in todo:
            cont = carry + (', "' if after_item else '"') + c + '"'
            ids = tokenizer(cont, add_special_tokens=False,
                            return_tensors="pt").input_ids.to(seq.device)[0]
            _crop(cache, n_prefix - 1)
            inp = torch.cat([last_tok, ids]).unsqueeze(0)
            past = n_prefix - 1
            out = model(input_ids=inp, past_key_values=cache, use_cache=True,
                        cache_position=torch.arange(past, past + inp.shape[1],
                                                    device=seq.device))
            logp = torch.log_softmax(out.logits[0].float(), dim=-1)
            # logits[j] 는 inp[j+1] 을 예측한다
            lp = float(sum(logp[j, ids[j]] for j in range(len(ids))))
            logs[c] = max(logs[c], lp)
            del out, logp
    # 확률이 0/1 로 크게 갈리므로(이름을 안 댄 장르는 대개 1e-6 미만) 순위를
    # 보려면 log 값이 필요하다. probs 는 그대로의 값, log10 은 log10 확률.
    return {"probs": {c: (1.0 if c in named_keep else math.exp(logs[c]))
                      for c in categories},
            "log10": {c: (0.0 if c in named_keep else round(logs[c] / math.log(10), 3))
                      for c in categories},
            "slots": len(slots)}


def load_probs(run_dir: Path) -> dict:
    """<run>/cat_probs_*.jsonl -> {uuid: {장르: 확률}} (샤드 파일을 모두 합친다)."""
    out = {}
    for f in sorted(Path(run_dir).glob("cat_probs_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                j = json.loads(line)
                out[j["uuid"]] = j.get("probs") or {}
    return out

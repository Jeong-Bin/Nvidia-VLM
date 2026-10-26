#!/usr/bin/env python3
"""1차 추론 중 "출력 토큰이 영상의 어느 시점을 봤는가"를 기록한다 (--attn-timeline).

목적: 모델이 장르 이름이나 점수를 쓸 때 영상의 몇 초를 보고 있었는지를,
모델에게 시간을 말하게 하지 않고 내부 신호에서 읽는다. 시간을 글로 답하게 한
--time-spans 는 8B 가 영상의 시각 표시를 읽지 못하고 0~20초로 어림해 한계가
분명했다(실측 20261014_162702 / 175954 / 20261019_141710: 끝점의 96~99% 가 정수,
시각 표시의 ".2" 값 0건, 잠깐 나온 요소도 21~27% 가 통째 구간).

어떻게 재나:
  모델의 어텐션 계산(sdpa)은 손대지 않는다 - 계산 방식을 eager 로 바꾸면 수치가
  미세하게 달라져 출력이 바뀔 수 있다. 대신 각 어텐션 층의 forward 가 끝난 뒤
  훅에서, 방금 처리한 마지막 위치의 query 를 같은 가중치로 다시 만들고(RoPE
  포함) 캐시에 쌓인 key 전체와 곱해 그 한 줄의 어텐션을 직접 계산한다. 출력에는
  아무것도 되돌려 주지 않으므로 생성 결과는 그대로다.

  영상 토큰은 Qwen3-VL 이 프레임 2장을 한 묶음으로 넣는다(temporal_patch_size=2).
  묶음마다 "<t seconds>" 표시 뒤에 <|video_pad|> 가 이어 붙으므로, 연속된
  video_pad 한 덩어리가 한 시간 칸이다. 2fps 면 칸 하나가 1초([g, g+1) 초)다.
  한 줄의 어텐션을 칸마다 더하고 head 평균을 낸 값이 그 층·그 토큰의 시간 곡선이다.

  행(row) k 는 "생성 토큰 k 를 낸 forward" 의 마지막 위치 어텐션이다 - 첫 토큰은
  prefill 의 마지막 위치가, 이후 토큰은 그 앞 토큰을 넣은 decode 단계가 낸다.
  그래서 행 수 = 생성 토큰 수이고, 토큰 k 를 쓸 때 본 곳이 행 k 다.

몰림 토큰(attention sink) 빼기 - 기록할 때(AttnRecorder):
  영상 토큰 중 어텐션이 습관적으로 몰리는 토큰이 있다. 장면 내용과 무관하게
  "볼 곳이 없을 때 남는 어텐션을 버리는 자리"다. 실측(076fce0d, 뒤쪽 층):
  첫 칸(0~1초) 448개 토큰 중 한 개(440번)가 그 칸 어텐션의 95% 를 가져갔고,
  1초 칸 전체의 40배였다. 그대로 두면 모든 토큰의 곡선에서 첫 칸이 최댓값이
  되어 시점 정보를 덮는다. 첫 칸을 통째로 빼면 0~1초에 실제로 있던 것도
  사라지고, 평균과 비교해 보정해도 몰리는 양이 토큰마다 1.5~76배로 달라 내용이
  남지 않았다. 그래서 몰림 토큰만 찾아서 뺀다: 각 층의 첫 forward(prefill)에서
  영상 어텐션의 SINK_FRAC 이상을 혼자 가져가는 토큰을 그 층의 몰림 토큰으로
  정하고, 그 클립의 생성이 끝날 때까지 칸 합계에서 제외한다.

곡선 만들기(curves_for):
  1) (몰림 토큰은 기록할 때 이미 빠져 있다)
  2) 토큰·층마다 칸의 합이 1 이 되게 나눈다 - "영상을 얼마나 봤나"가
     아니라 "영상을 볼 때 어느 시점을 봤나"만 남긴다. 결정 토큰은 영상보다
     글(메뉴/기준표)을 더 보므로, 나누지 않으면 영상 쪽 값이 전부 작아진다.
  3) 뒤쪽 층(마지막 1/4)을 평균한다 - 3클립에서 사건 시점과 가장 잘 맞았다
     (개가 나온 클립: 근거 문장 11~15초 정점, 모델이 따로 답한 구간 10~15초).
  4) 답 전체 토큰의 평균 분포로 나눈 비율(rel)로 본다 - 그 토큰을 쓸 때
     평소보다 더 본 시점만 남긴다. 1 보다 크면 평균보다 더 본 것.

공간 지도(spatial, 시각화의 히트맵 영상):
  같은 훅에서 칸으로 더하기 전의 토큰별 영상 어텐션도 모은다. 생성 토큰마다
  뒤쪽 층(default_layers)의 분포(층마다 합 1)를 평균해 한 줄씩 쌓고, 생성이
  끝난 뒤 spatial_maps 가 장르별 토큰(curves_for 의 tokens - 이름 토큰 + 그 장르의
  근거 문장)만 골라 평균한다. 칸 하나는 Qwen3-VL 이 2x2 로 합친 패치 격자
  (896x512 입력이면 16x28, 토큰 하나가 32x32px)이고, 토큰은 그 격자의 행 우선
  순서다. 시각화는 시간 곡선처럼 답 전체 평균 대비 비율(rel)로 그린다 - 장르를
  쓸 때 평소보다 더 본 곳만 남고, 늘 보는 자리(첫 칸, 모서리)는 지워진다.

해석할 때 주의:
  어텐션은 "어디를 봤나"이지 "그것 때문에 판단했나"가 아니다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

# 곡선을 만들 때 평균 낼 층 - 마지막 1/4. 3클립 비교(앞 0~8 / 가운데 9~26 /
# 뒤 27~35)에서 뒤쪽이 사건 시점과 가장 잘 맞고 정점도 뚜렷했다. 표본이 작으니
# 원본 행렬(모든 층)을 npz 로 남겨 나중에 다른 층으로 다시 계산할 수 있게 한다.
def default_layers(n_layers: int) -> list[int]:
    return list(range((3 * n_layers) // 4, n_layers))


# 첫 칸(0~1초)을 통째로 뺄지. 몰림 토큰을 기록 단계에서 빼므로 기본은 끈다.
# 몰림 토큰을 빼기 전에 기록한 npz(sink_removed 가 없는 것)는 curves_for 가
# 알아서 첫 칸을 뺀다.
DROP_FIRST_BUCKET = False

# 몰림 토큰 기준: 한 층의 영상 어텐션 중 이 비율 이상을 혼자 가져가는 토큰.
# 실측 몰림 토큰은 칸 하나의 95%, 영상 전체의 약 60% 였고, 보통 토큰 하나는
# 0.1% 안팎이라 20% 면 둘을 확실히 가른다.
SINK_FRAC = 0.2


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class AttnRecorder:
    """Qwen3-VL 텍스트 어텐션 층에 훅을 달아 시간 칸별 어텐션을 모은다."""

    def __init__(self, model):
        self.layers = []
        for _, m in model.named_modules():
            if (type(m).__name__.endswith("TextAttention")
                    and hasattr(m, "q_proj") and hasattr(m, "q_norm")
                    and hasattr(m, "layer_idx")):
                self.layers.append(m)
        self.layers.sort(key=lambda m: m.layer_idx)
        self.n_layers = len(self.layers)
        self.enabled = False
        self._buf = []
        self._handles = [m.register_forward_hook(self._hook, with_kwargs=True)
                         for m in self.layers]

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # ------------------------------------------------------------------
    def begin(self, input_ids: torch.Tensor, video_token_id: int, fps: float,
              grid=None):
        """한 번의 generate 직전에 부른다. 영상 토큰 위치를 시간 칸으로 묶는다.

        grid: (칸 수, 행, 열) - 병합 후 영상 토큰 격자. 주면 공간 지도도 모은다.
        """
        ids = input_ids[0]
        pos = torch.nonzero(ids == video_token_id).flatten()
        if pos.numel() == 0:
            self.enabled = False
            return False
        # 연속된 덩어리마다 새 칸
        new = torch.ones_like(pos, dtype=torch.bool)
        new[1:] = pos[1:] != pos[:-1] + 1
        bucket = torch.cumsum(new.long(), 0) - 1
        self.vid_pos = pos
        self.bucket = bucket
        self.n_buckets = int(bucket[-1]) + 1
        # 칸 g 는 프레임 2g, 2g+1 -> [2g/fps, (2g+2)/fps) 초
        self.times = [[2 * g / fps, (2 * g + 2) / fps] for g in range(self.n_buckets)]
        self._buf = []
        self.sink = {}          # layer_idx -> 영상 토큰 중 몰림 토큰 표시(bool)
        # 공간 지도: 격자가 영상 토큰 수와 맞을 때만 모은다.
        self.grid = (tuple(int(x) for x in grid)
                     if grid is not None and int(np.prod(grid)) == pos.numel()
                     else None)
        self._sp_layers = set(default_layers(self.n_layers))
        self._sp_last = max(self._sp_layers)
        self._sp_rows = []      # 생성 토큰마다 [영상 토큰 수] (뒤쪽 층 평균 분포)
        self._sp_cur = None
        self.enabled = True
        return True

    def sink_summary(self) -> dict:
        """층별 몰림 토큰 수와 그 토큰이 속한 칸 - npz 에 남긴다."""
        out = {}
        for li, m in sorted(self.sink.items()):
            idx = torch.nonzero(m).flatten()
            out[li] = {"n": int(idx.numel()),
                       "buckets": self.bucket[idx].tolist(),
                       "video_index": idx.tolist()}
        return out

    def end(self) -> np.ndarray | None:
        """generate 직후에 부른다. [생성 토큰 수, 층 수, 칸 수] (float32)."""
        self.enabled = False
        if not self._buf:
            return None
        rows = torch.stack(self._buf).float().cpu().numpy()
        self._buf = []
        n = rows.shape[0] // self.n_layers
        return rows[: n * self.n_layers].reshape(n, self.n_layers, self.n_buckets)

    def spatial_rows(self) -> np.ndarray | None:
        """생성 토큰마다의 공간 분포 [생성 토큰 수, 영상 토큰 수] (float32). 없으면 None.

        행 k 는 rows 의 행 k 와 같은 토큰이다(생성 토큰 k 를 낸 forward).
        """
        if self.grid is None or not self._sp_rows:
            return None
        out = torch.stack(self._sp_rows).float().cpu().numpy()
        self._sp_rows = []
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _hook(self, module, args, kwargs, output):
        if not self.enabled:
            return
        h = args[0] if args else kwargs["hidden_states"]
        cos, sin = kwargs.get("position_embeddings") or args[1]
        cache = kwargs.get("past_key_values")
        if cache is None:
            return
        keys = cache.layers[module.layer_idx].keys          # [1, kvh, L, hd]
        hd = module.head_dim
        q = module.q_norm(module.q_proj(h[:, -1:]).view(1, 1, -1, hd)).transpose(1, 2)
        c, s = cos[:, -1:].unsqueeze(1), sin[:, -1:].unsqueeze(1)
        q = (q * c) + (_rotate_half(q) * s)                  # [1, H, 1, hd]
        kvh = keys.shape[1]
        rep = q.shape[1] // kvh
        q = q.view(1, kvh, rep, hd).float()
        att = torch.matmul(q, keys.float().transpose(-1, -2)) * module.scaling
        att = torch.softmax(att, dim=-1)                     # [1, kvh, rep, L]
        att = att.reshape(-1, att.shape[-1]).mean(0)         # head 평균 [L]
        v = att.index_select(0, self.vid_pos)
        # 이 층의 첫 forward(prefill)에서 몰림 토큰을 정하고 이후 계속 뺀다.
        li = module.layer_idx
        if li not in self.sink:
            self.sink[li] = v > SINK_FRAC * v.sum()
        v = v.masked_fill(self.sink[li], 0.0)
        if self.grid is not None and li in self._sp_layers:
            x = v.float() / (v.sum().float() + 1e-12)
            self._sp_cur = x if self._sp_cur is None else self._sp_cur + x
            if li == self._sp_last:             # 이 forward 의 뒤쪽 층을 다 봤다
                self._sp_rows.append((self._sp_cur / len(self._sp_layers)).half())
                self._sp_cur = None
        out = torch.zeros(self.n_buckets, device=v.device, dtype=v.dtype)
        out.index_add_(0, self.bucket, v)
        self._buf.append(out)


# ----------------------------------------------------------------------
# 관심 토큰 고르기
# ----------------------------------------------------------------------
def token_char_spans(gen_ids, tokenizer) -> tuple[str, list[tuple[int, int]]]:
    """생성 토큰을 한 개씩 이어 붙여 각 토큰이 차지하는 글자 범위를 구한다."""
    text, spans = "", []
    for i in range(len(gen_ids)):
        nxt = tokenizer.decode(gen_ids[: i + 1], skip_special_tokens=True)
        spans.append((len(text), len(nxt)))
        text = nxt
    return text, spans


def _tok_at(spans, ch: int) -> int | None:
    for k, (a, b) in enumerate(spans):
        if a <= ch < b:
            return k
    return None


def _toks_in(spans, a: int, b: int) -> list[int]:
    return [k for k, (x, y) in enumerate(spans) if x < b and y > a and y > x]


def curves_for(rows: np.ndarray, gen_ids, tokenizer, categories: list[str],
               group_keys: dict[str, str], layers: list[int] | None = None,
               drop_first: bool | None = None) -> dict:
    """관심 토큰들의 시간 곡선(rel). 칸마다 하나, 뺀 칸은 None.

    categories : 1차 답의 scenario_types 에서 이름을 댄 장르
    group_keys : {"Dynamic object": "dynamic_object", ...} - JSON 키
    반환:
      "categories": {장르: 곡선} - 시각화가 장르 줄에 그리는 것. 그 장르 이름을
                    고른 토큰(scenario_types 안의 첫 토큰)과 점수 근거 문장 중
                    그 장르를 다룬 부분("Animal: ...")의 토큰을 합쳐 평균한다.
      "name":       {장르: 곡선} - 이름 토큰만
      "score":      {묶음: 곡선} - 점수 숫자 토큰만
      "reason":     {묶음: 곡선} - 근거 문장 전체
      "tokens":     {이름: [행 번호...]} - 어느 토큰을 썼는지
    """
    text, spans = token_char_spans(gen_ids, tokenizer)
    n = min(len(spans), rows.shape[0])
    spans = spans[:n]
    layers = layers or default_layers(rows.shape[1])
    a = rows[:n][:, layers, :].astype(np.float64)
    first = 1 if (DROP_FIRST_BUCKET if drop_first is None else drop_first) else 0
    a = a[..., first:]
    p = (a / (a.sum(-1, keepdims=True) + 1e-12)).mean(axis=1)   # [n, buckets]
    base = p.mean(axis=0) + 1e-12

    def rel(ks):
        ks = sorted(set(k for k in ks if k is not None and k < n))
        if not ks:
            return None
        r = (p[ks].mean(axis=0) / base).round(4).tolist()
        return [None] * first + r

    out = {"categories": {}, "name": {}, "score": {}, "reason": {}, "tokens": {}}
    name_tok = {}
    m = re.search(r'"scenario_types"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if m:
        region_a, region = m.start(1), m.group(1)
        for c in categories:
            i = region.lower().find(c.lower())
            k = _tok_at(spans, region_a + i) if i >= 0 else None
            if k is not None:
                name_tok[c] = k
                out["name"][c] = rel([k])
    seg_toks = {c: [] for c in categories}
    for g, key in group_keys.items():
        ms = re.search(r'"%s"\s*:\s*(\d)' % re.escape(key), text)
        if ms:
            k = _tok_at(spans, ms.start(1))
            if k is not None:
                out["score"][g] = rel([k])
                out["tokens"][f"{g} score"] = [k]
        mr = re.search(r'"%s_reason"\s*:\s*"(.*?)"' % re.escape(key), text, re.DOTALL)
        if mr and mr.end(1) > mr.start(1):
            ra, rtext = mr.start(1), mr.group(1)
            ks = _toks_in(spans, ra, mr.end(1))
            if ks:
                out["reason"][g] = rel(ks)
                out["tokens"][f"{g} reason"] = ks
            # 근거 문장을 장르 이름이 나오는 자리에서 끊어 장르별로 나눈다.
            # 이름이 하나도 안 나오고 장르가 하나뿐이면 문장 전체가 그 장르 몫이다.
            hits = sorted((rtext.lower().find(c.lower()), c) for c in categories
                          if rtext.lower().find(c.lower()) >= 0)
            if hits:
                for (pos, c), nxt in zip(hits, hits[1:] + [(len(rtext), None)]):
                    seg_toks[c] += _toks_in(spans, ra + pos, ra + nxt[0])
            elif len(categories) == 1 and ks:
                seg_toks[categories[0]] += ks
    for c in categories:
        ks = ([name_tok[c]] if c in name_tok else []) + seg_toks[c]
        r = rel(ks)
        if r is not None:
            out["categories"][c] = r
            out["tokens"][c] = sorted(set(ks))
    return out


def spatial_maps(sp_rows: np.ndarray, token_sets: dict, grid) -> dict:
    """장르별 공간 지도. {"mean": [칸,행,열], "maps": {이름: [칸,행,열]}}.

    sp_rows    : AttnRecorder.spatial_rows() [생성 토큰 수, 영상 토큰 수]
    token_sets : {이름: [행 번호...]} - curves_for 의 tokens 에서 장르 것만
    mean 은 답 전체 토큰의 평균(시각화의 비교 기준), maps 는 그 이름의 토큰 평균.
    """
    n = sp_rows.shape[0]
    out = {"mean": sp_rows.mean(axis=0).reshape(grid), "maps": {}}
    for name, ks in token_sets.items():
        ks = sorted(set(k for k in ks if k is not None and k < n))
        if ks:
            out["maps"][name] = sp_rows[ks].mean(axis=0).reshape(grid)
    return out


def save_npz(path: Path, rows: np.ndarray, gen_ids, times, layers,
             sink: dict | None = None, spatial: dict | None = None):
    """원본 행렬(모든 층)과 토큰 id 를 남긴다 - 층/토큰을 바꿔 다시 계산할 수 있게.

    rows 는 몰림 토큰을 뺀 값이다(sink_removed=1). sink 는 층별로 뺀 토큰 정보.
    spatial 은 spatial_maps() 의 결과 (있을 때만) - spatial_mean, spatial_names,
    spatial_maps([이름 수, 칸, 행, 열]) 로 남긴다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if spatial and spatial["maps"]:
        names = list(spatial["maps"])
        extra = {"spatial_mean": spatial["mean"].astype(np.float32),
                 "spatial_names": json.dumps(names, ensure_ascii=False),
                 "spatial_maps": np.stack([spatial["maps"][k] for k in names]
                                          ).astype(np.float32)}
    np.savez_compressed(path, **extra, rows=rows.astype(np.float16),
                        gen_ids=np.asarray(gen_ids, dtype=np.int64),
                        times=np.asarray(times, dtype=np.float32),
                        default_layers=np.asarray(layers, dtype=np.int16),
                        sink_removed=np.int8(1),
                        sink=json.dumps(sink or {}))


def load_spatial(z) -> dict | None:
    """save_npz 로 남긴 공간 지도를 spatial_maps() 형태로 되돌린다. 없으면 None."""
    if "spatial_maps" not in z.files:
        return None
    names = json.loads(str(z["spatial_names"]))
    return {"mean": z["spatial_mean"],
            "maps": {k: m for k, m in zip(names, z["spatial_maps"])}}

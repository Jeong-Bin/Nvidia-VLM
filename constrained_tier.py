#!/usr/bin/env python3
"""JSON 출력에서 등급 필드를 정수 0~4 로만 나오게 강제한다.

배경 - 왜 이게 필요한가:
  프롬프트로 'Low, Moderate, High 중 하나만 써라'라고 지시해도 모델이
  "Low to moderate", "Highly critical", "Common, low rarity" 처럼 경계를
  흐리는 답을 낸다(실측 20260811, 50클립). 사후 파싱(_extract_tier)으로
  정규화할 수는 있지만, 그건 이미 나온 모호함을 추측으로 메우는 것이다.

왜 문자열이 아니라 정수인가:
  Qwen3-VL 토크나이저 실측 -
    "Low"      -> ['Low']            1토큰
    "Moderate" -> ['Moder', 'ate']   2토큰   <- 상태 추적이 배로 복잡해진다
    "High"     -> ['High']           1토큰
    1 / 2 / 3  -> 각각 1토큰
  여러 토큰짜리 후보는 "지금 Moder 까지 냈으니 다음엔 ate 만 허용" 같은
  중간 상태를 다 관리해야 한다. 정수는 한 스텝이면 끝난다. 그리고 정수에는
  "1.5" 나 "1 to 2" 에 해당하는 자연스러운 표현이 없어서, 언어적으로
  모호함을 만들 여지 자체가 사라진다.

동작 방식:
  생성된 토큰열 끝이 `"safety_tier":` 같은 마커와 일치하면, 그 다음부터
  숫자가 나올 때까지 공백만 허용하고 숫자 자리에서는 TIER_VALUES(0~4) 로 막는다.
  모델이 JSON 을 `"x": 1` 로 쓸지 `"x":1` 로 쓸지 미리 알 수 없으므로
  공백 허용 단계를 반드시 둬야 한다(실측: ': 1' 은 ['Ġ','1'] 2토큰,
  ':1' 은 ['1'] 1토큰).

주의:
  prefix_allowed_tokens_fn 이 빈 리스트를 반환하면 transformers 가 예외를
  던진다. 어떤 분기에서도 빈 리스트가 나오지 않도록 아래에서 항상 최소 한
  개 이상을 돌려준다.
"""
from __future__ import annotations

from functools import lru_cache

# 등급 정의. 프롬프트/파서/집계가 모두 이 한 곳을 참조한다.
#
# 4 는 데이터셋에 사실상 없는 극단값(충돌 임박/발생, 차가 본 적 없을 법한
# 물체)이다. 일부러 넣어둔 이유는 척도 압축을 풀기 위해서다 - 상한이 3 이면
# 모델이 3 을 "최악"으로 취급해 아끼고 2 로 몰린다(실측 20260813: 정답 3인
# 20건 중 18건을 2로 예측). 위에 더 극단적인 칸을 두면 3 이 "최악"이 아니라
# "심각한 편"이 되어 쓰기 쉬워진다. 4 를 실제로 찾는 것은 목표가 아니다.
#
# 0 은 "위협 요소가 아예 없다/전혀 특이하지 않다"를 1(Low)과 분리하기 위해
# 추가했다 - 1 은 "요소는 있지만 쉽게 처리됨"이고 0 은 "그런 요소 자체가
# 없음"이라 서로 다른 사실을 가리킨다. 이 둘을 합쳐두면 정상 주행 클립과
# 경미한 요소가 있는 클립이 같은 칸에 몰려 변별력이 없어진다.
TIER_LABELS = {0: "None", 1: "Low", 2: "Moderate", 3: "High", 4: "Extreme"}
TIER_VALUES = tuple(sorted(TIER_LABELS))          # (0, 1, 2, 3, 4)

# 강제 대상 필드. JSON 키 이름 그대로 쓴다.
TIER_FIELDS = ("safety_tier", "rarity_tier")

# 마커 뒤에서 숫자가 나오기 전까지 허용할 최대 토큰 수. JSON 이면
# 공백 한두 개가 전부라 3 이면 충분하고, 이 값을 넘으면 제약을 푼다
# (모델이 예상 밖 형식으로 갔을 때 생성이 막히는 것을 방지).
MAX_GAP_TOKENS = 3


def tier_label(value) -> str:
    """0/1/2/3/4 -> "None"/"Low"/"Moderate"/"High"/"Extreme". 알 수 없으면 "Unknown"."""
    try:
        return TIER_LABELS[int(value)]
    except (TypeError, ValueError, KeyError):
        return "Unknown"


def tier_menu() -> str:
    """프롬프트에 넣을 등급 정의 문구."""
    return ", ".join(f"{v} = {TIER_LABELS[v]}" for v in TIER_VALUES)


# ---------------------------------------------------------------------------
# 등급 rubric
#
# "Low/Moderate/High" 세 낱말만 주면 기준점이 없어 매번 다르게 해석된다.
# 실측(20260812, 375 edge 클립):
#   - safety=2 가 73% (274/375) 로 중앙에 몰림
#   - rarity=3 이 0건. 척도가 사실상 2단계로 줄어듦
#   - Jaywalking 230건이 1/2/3 에 흩어졌는데 이유 문장은 서로 구분이 안 됨
#     (2: "requires the ego-vehicle to remain vigilant",
#      3: "requires the vehicle to remain stationary and vigilant")
#
# 원인은 모델이 "주의가 필요한가"로 판단한 것 - 그건 거의 모든 장면에서
# 참이라 변별력이 없다. 그래서 형용사 대신 관찰 가능한 기준으로 다시 쓴다:
#   safety -> 자차가 실제로 무엇을 했는가 (조정 없음 / 여유 있는 조정 / 급한 회피)
#   rarity -> 1000개 클립 중 몇 개에 나오는가 (구체적 척도를 줘야 3이 쓰인다)
#
# safety/rarity 는 난이도(illumination/precipitation/road_surface/
# atmospheric_obscurants)와 독립이다 - 비가 오거나 야간이라는 사실 자체는
# 등급을 올리는 근거가 아니다. 실제로 자차가 무엇을 했는지, 그 요소가 얼마나
# 드문지가 기준이다. 이 지시는 프롬프트 본문에도 명시한다
# (build_nureasoning_prompt 의 steps45 조립부 참고).
SAFETY_RUBRIC = {
    0: "It is a scene of peaceful driving, with no elements on the road that threaten safety.",
    1: "There are objects requiring the ego-vehicle's attention on the road, "
       "but they are located away from the vehicle's driving path or are sufficiently distant, "
       "so neither deceleration nor a change in steering is necessary.",
    2: "The ego-vehicle had to give way - slow, yield, wait, or steer around something "
       "- but with plenty of time and space to do it.",
    3: "The ego-vehicle had to act urgently, or a small mistake by anyone "
       "would have caused a collision: hard braking, evasive steering, or "
       "something entering its path at close range.",
    4: "This is a highly dangerous situation. A collision is either imminent or has already occurred. "
       "Urgent evasive action or emergency braking is required, "
       "yet there is no guarantee that such measures will prevent the accident."
}

RARITY_RUBRIC = {
    0: "It is a monotonous scene typical of everyday driving. "
       "Nothing out of the ordinary is visible, apart from the usual vehicles, pedestrians on the sidewalk, or empty roads.",
    1: "These are elements you can frequently see while driving. "
       "For example, pedestrians or cyclists crossing a crosswalk, or traffic cones guiding the lanes.",
    2: "These are elements or situations occasionally encountered while driving. "
       "For example, jaywalkers crossing outside of crosswalks, "
       "cyclists in dangerously close proximity to the ego-vehicle "
       "or lanes completely altered due to construction.",
    3: "These are critical edge cases that autonomous vehicles must not overlook. "
       "It can happen on the road on very rare occasions. "
       "For example, a person wearing a mascot costume, wildlife crossing the road, "
       "a fallen tree blocking the road, traffic accidents or fire, or a road completely submerged by the flood.",
    4: "This is a rare situation—the kind one might not see even once in a lifetime. "
       "For example, a plane making an emergency landing on the road, "
       "a road destroyed by a natural disaster, "
       "or the very moment a major traffic accident occurs.",
}

# 등급은 "무엇이 있는가"가 아니라 "그것이 무엇을 하는가"로 갈린다.
#
# 실측(20260812)에서 이걸 안 가르치면 모델이 객체 이름으로 패턴 매칭한다:
# Animal 7건이 내용과 무관하게 전부 rarity=2 였다 - 길 위의 새(흔함)와
# 도로를 건너는 칠면조, 말을 탄 기마경찰(매우 드묾)이 같은 등급이었다.
# Jaywalking 도 횡단보도를 언급하지 않은 189건 중 134건(71%)이 rarity=1 로,
# 정상 횡단과 무단횡단을 구분하지 못했다.
#
# rubric 에 객체 목록을 주면 그 목록을 외우므로, 대신 "같은 객체 x 다른
# 행동 = 다른 등급" 대비쌍을 준다.
CONTRAST_EXAMPLES = [
    ("a pedestrian using a crosswalk with the signal",
     "a pedestrian stepping into the lane from between parked cars"),
    ("a dog on a leash walking beside its owner on the sidewalk",
     "a loose animal wandering into the roadway"),
    ("a car parked at the kerb",
     "a car stopped across the driving lane"),
    ("a cyclist riding in a bike lane",
     "a cyclist swerving into the traffic lane"),
]


def contrast_text() -> str:
    """대비쌍을 프롬프트 문구로. 왼쪽이 1, 오른쪽이 3 쪽으로 간다."""
    return "\n".join(f"     {low}  ->  low;   {high}  ->  high"
                     for low, high in CONTRAST_EXAMPLES)


def _fmt_rubric(rubric: dict) -> str:
    return "\n".join(f"     {v} ({TIER_LABELS[v]}) = {rubric[v]}"
                     for v in TIER_VALUES)


def safety_rubric_text() -> str:
    return _fmt_rubric(SAFETY_RUBRIC)


def rarity_rubric_text() -> str:
    return _fmt_rubric(RARITY_RUBRIC)


# 시각화 폴더를 나누는 점수 = safety + rarity. 둘 다 0~4 이므로 0~8.
# 이건 nuReasoning 의 1~10 난이도 점수와 다르다 - 그건 "얼마나 가치 있는
# 롱테일인가"를 모델이 직접 매기게 한 것이고(우리는 폐기했다), 이건 이미
# 받아둔 두 등급을 검수 편의를 위해 더한 것뿐이다. 모델에게 묻지 않는다.
SCORE_MIN, SCORE_MAX = 2 * min(TIER_VALUES), 2 * max(TIER_VALUES)


def tier_score(safety_tier, rarity_tier) -> int | None:
    """safety + rarity 합계. 둘 중 하나라도 못 읽었으면 None."""
    if safety_tier is None or rarity_tier is None:
        return None
    try:
        s, r = int(safety_tier), int(rarity_tier)
    except (TypeError, ValueError):
        return None
    if s not in TIER_LABELS or r not in TIER_LABELS:
        return None
    return s + r


def score_dirname(score) -> str:
    """점수 -> 시각화 하위 폴더명. 못 읽은 건 따로 모은다."""
    return f"score_{score}" if score is not None else "score_unknown"


# 마커를 "토큰 id 열"이 아니라 "디코딩된 문자열"로 맞춘다.
#
# BPE 는 앞 문맥에 따라 병합이 달라진다. 실측(Qwen3-VL):
#   '"safety_tier":' 단독      -> ['"s','afety','_t','ier','":']
#   JSON 중간에 같은 키가 올 때 -> ['Ġ"','s','afety','_t','ier','":']
# 앞 토큰이 '"s' 하나로 붙느냐 'Ġ"'+'s' 로 갈리느냐가 문맥에 따라 바뀌므로,
# 토큰 id 열을 접미사 비교하면 실제 생성에서 매칭이 실패한다.
# 꼬리 몇 토큰만 디코딩해 문자열로 비교하면 이 문제가 사라진다.
_TAIL_TOKENS = 12          # 마커 + 여유 공백을 덮기에 충분한 길이


@lru_cache(maxsize=8)
def _build_tables(tokenizer, fields: tuple[str, ...]):
    """(허용 숫자 토큰, 공백 토큰) 집합을 만든다.

    토크나이저마다 결과가 다르므로 하드코딩하지 않고 실제로 인코딩해 본다.
    """
    def enc(s):
        return tuple(tokenizer.encode(s, add_special_tokens=False))

    # 숫자 토큰: 앞에 공백이 있는 형태와 없는 형태 둘 다 확인해 숫자만 담는다.
    digits = set()
    for v in TIER_VALUES:
        for pat in (str(v), f" {v}"):
            ids = enc(pat)
            if len(ids) == 1:
                digits.add(ids[0])
            elif len(ids) == 2:
                # ['Ġ', '1'] 처럼 쪼개지면 뒤쪽(숫자)만 담는다.
                digits.add(ids[1])

    # 공백류 토큰: 마커와 숫자 사이에 낄 수 있는 것들.
    spaces = set()
    for pat in (" ", "  "):
        for tid in enc(pat):
            spaces.add(tid)

    return frozenset(digits), frozenset(spaces)


def _marker_patterns(fields) -> tuple[str, ...]:
    """생성 텍스트 꼬리에서 찾을 마커 문자열들."""
    pats = []
    for f in fields:
        pats.extend((f'"{f}":', f'"{f}" :', f"'{f}':"))
    return tuple(pats)


def _allowed_at(tokenizer, seq, patterns):
    """지금이 등급 자리인가? 맞으면 허용 토큰 집합, 아니면 None(제약 없음)."""
    tail_ids = seq[-_TAIL_TOKENS:]
    if not tail_ids:
        return None
    tail = tokenizer.decode(tail_ids)
    for pat in patterns:
        pos = tail.rfind(pat)
        if pos < 0:
            continue
        after = tail[pos + len(pat):]
        if after == "" or after.strip() == "":   # 마커 직후 또는 공백만 흘렀다
            return True
        break                                    # 이미 값이 나왔다 -> 해제
    return None


class TierLogitsProcessor:
    """등급 자리에서만 로짓을 {공백, 1, 2, 3} 으로 제한하는 LogitsProcessor.

    transformers 의 prefix_allowed_tokens_fn(PrefixConstrainedLogitsProcessor)
    을 쓰지 않는 이유:
      그 구현은 매 스텝 `mask[row, allowed] = 0` 을 하는데, 제약이 없는
      스텝에서는 allowed 가 vocab 전체(Qwen3-VL 기준 151,669개) 리스트가
      된다. 즉 생성 토큰 하나마다 15만 개짜리 파이썬 리스트를 만들어 GPU
      인덱싱에 넘긴다. 단일 프로세스 소규모 테스트는 통과했지만, 8개
      샤드를 동시에 돌리자 8샤드 전부
      "CUDA error: unspecified launch failure" 로 죽었다
      (실측 20260811, 1,998 클립 중 142개만 처리하고 중단).

    여기서는 제약이 있는 스텝에서만 텐서를 건드리고, 없는 스텝은 로짓을
    그대로 통과시킨다. 인덱싱 대상도 4~5개뿐이라 부하가 없다.
    """

    def __init__(self, tokenizer, fields=TIER_FIELDS):
        digits, spaces = _build_tables(tokenizer, tuple(fields))
        self.tokenizer = tokenizer
        self.patterns = _marker_patterns(fields)
        self.allowed = sorted(spaces | digits)   # 공백 또는 바로 숫자
        self._idx = None                         # 디바이스별 캐시

    def __call__(self, input_ids, scores):
        import torch

        rows = []
        for b in range(input_ids.shape[0]):
            seq = input_ids[b].tolist()
            if _allowed_at(self.tokenizer, seq, self.patterns):
                rows.append(b)
        if not rows:
            return scores                        # 제약 없는 스텝: 그대로 통과

        if self._idx is None or self._idx.device != scores.device:
            self._idx = torch.tensor(self.allowed, device=scores.device,
                                     dtype=torch.long)
        # 해당 행만 -inf 로 덮고 허용 토큰 자리에 원래 점수를 되돌린다.
        for b in rows:
            keep = scores[b, self._idx].clone()
            scores[b].fill_(float("-inf"))
            scores[b, self._idx] = keep
        return scores


def make_tier_processor(tokenizer, fields=TIER_FIELDS):
    """model.generate(logits_processor=...) 에 넣을 프로세서를 만든다."""
    return TierLogitsProcessor(tokenizer, fields)


def make_tier_prefix_fn(tokenizer, fields=TIER_FIELDS, vocab_size=None):
    """구버전 호환용 prefix_allowed_tokens_fn.

    쓰지 말 것 - 제약 없는 스텝마다 vocab 전체 리스트를 만들어 8샤드 동시
    실행에서 CUDA 오류를 냈다. make_tier_processor 를 쓴다.
    """
    digits, spaces = _build_tables(tokenizer, tuple(fields))
    allow_gap = sorted(spaces | digits)
    patterns = _marker_patterns(fields)
    all_tokens = list(range(vocab_size or len(tokenizer)))

    def prefix_fn(batch_id, input_ids):
        seq = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        return allow_gap if _allowed_at(tokenizer, seq, patterns) else all_tokens

    return prefix_fn

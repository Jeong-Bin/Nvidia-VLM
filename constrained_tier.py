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
    3: "An object suddenly appeared, the ego-vehicle took emergency actions "
       "such as hard braking and evasive steering. "
       "This is a situation that arises more suddenly compared to the two-point criterion.",
    4: "A collision occurred between the ego-vehicle and another object. "
       "UThe moment of the accident was captured on camera."
       
}

RARITY_RUBRIC = {
    0: "It is a monotonous scene typical of everyday driving. "
       "Nothing out of the ordinary is visible, apart from the usual vehicles, pedestrians on the sidewalk, or empty roads.",
    1: "These are elements you can frequently see while driving. "
       "For example, pedestrians or cyclists crossing a crosswalk.",
    2: "These are elements or situations occasionally encountered while driving. "
       "For example, jaywalkers crossing outside of crosswalks, "
       "cyclists in dangerously close proximity to the ego-vehicle "
       "or there are traffic cones but they do not affect ego-vehicle's driving.",
    3: "These are critical edge cases that can rarely occur on the road."
       "For example, a person wearing a mascot costume, wildlife crossing the road, "
       "a fallen tree blocking the road, an accident that has already occurred, a road completely submerged by the flood, "
       "or construction work and traffic cones have completely altered the ego-vehicle's driving path.",
    4: "This is a super rare situation—the kind one might not see even once in a lifetime. "
       "For example, a road destroyed by a natural disaster, "
       "the very moment a major traffic accident occurs. "
       "Furthermore, various exceptional situations that do not fit the context of a road environment.",
}

IMPACT_RUBRIC = {
    0: "It is a scene of normal driving, with no special object or "
       "environmental element present.",
    1: "It is a special object or environmental element present in the scene, "
       "but positioned away from the ego-vehicle's driving path or far enough "
       "that the vehicle did not need to slow or steer for it.",
    2: "It had a minor impact on driving - the ego-vehicle slowed gradually, "
       "briefly stopped, or made a slight lateral adjustment within its lane "
       "to avoid it, with plenty of time and space to do so.",
    3: "It had a moderate impact on driving - the ego-vehicle had to leave its "
       "lane, make a wide detour, or cross the center line to avoid it, but "
       "still had enough time to do so without urgency.",
    4: "It had a severe impact on driving - the ego-vehicle had to perform an "
       "emergency stop or emergency evasive maneuver with little to no time to react.",
}


# IMPACT_RUBRIC 의 대안판. 위쪽은 "자차가 실제로 무엇을 했는가" 하나만 보는데,
# 그러면 자차가 반응하지 '못한' 장면이 1점으로 떨어진다 - 예: 자전거가 바로
# 옆에 바짝 붙어 달리는데 egomotion 에는 변화가 없는 클립. 마이닝 목적에서는
# 그런 클립이야말로 검수 대상인데 1점이 되면 걸러진다.
#
# 그래서 이 판은 판정 기준을 세 축으로 나눈다:
#   침범 - 이것이 자차의 주행 경로를 얼마나 침범했는가 (도로 밖 / 다른 차선 /
#          경로 옆 / 경로 안)
#   여유 - 자차가 대응할 시간이 있었는가 (속도와 거리를 함께 본다)
#   대응 - 자차가 실제로 무엇을 했는가
#
# 세 축을 단순 합산하지 않는 이유:
#   합산은 축이 서로 독립일 때만 맞는데 여기서는 곱셈에 가깝다. 실제로 계산해
#   보면 "보도 위 보행자 + 자차 50km/h + 근접"이 침범 0 인데도 합계가 7/10 이
#   되어 3점을 받는다. 침범이 0 이면 속도가 얼마든 영향은 0 이어야 하므로,
#   침범을 먼저 게이트로 두고 그 안에서 여유/대응이 강도를 가르게 한다.
#
#   속도와 거리를 따로 더하지 않는 것도 같은 이유다. 둘은 사실상 같은 것을
#   다르게 잰 값이고(50km/h 30m = 2.2초, 10km/h 10m = 3.6초), 따로 더하면
#   두 상황이 같은 점수가 된다. 그래서 '여유'라는 한 축으로 합쳐 시간으로
#   말한다.
#
# 라벨링용 보조 기준(사람이 GT 를 일관되게 매길 때 쓰는 내부 척도):
#   침범 0 도로 밖(보도/갓길 너머) | 1 도로 위지만 다른 차선
#        2 경로 바로 옆 또는 진입 중 | 3 경로 안 정면
#   여유 0 정차 중이거나 서행(10km/h 이하) | 1 30m 이상 또는 저속
#        2 10~30m 중속 | 3 10m 이내 또는 50km/h 이상
# 이 보조 척도는 프롬프트에 넣지 않는다 - 모델에게 축 3개를 따로 재고 합치게
# 하면 그 예산이 탐지에서 빠진다(실측: position 축을 세분화한 v2 에서 FN 이
# 10 -> 35 로 늘고 F1 이 80.4% -> 74.0% 로 떨어졌다). 모델에게는 아래 완성된
# 0~4 문장만 보여준다.
IMPACT_RUBRIC_2 = {
    0: "It is not present in the scene, or it stays entirely off the roadway - "
       "on the pavement, behind a barrier, or beyond the far kerb - and never "
       "moves toward the road. How fast the ego-vehicle is driving does not "
       "matter here: if it is off the roadway and stays there, this is 0.",
    1: "It is on the roadway but in another lane, or it is off the roadway and "
       "merely close to the ego-vehicle's path. The ego-vehicle keeps its speed "
       "and its line, and would have driven the same way had it not been there.",
    2: "It is in or beside the lane the ego-vehicle is driving through, and the "
       "vehicle had room to deal with it - it eased off, waited, or shifted "
       "slightly within its own lane, with several seconds of margin. A slow or "
       "stopped ego-vehicle that simply lets it pass belongs here.",
    3: "It is in the ego-vehicle's path, or so close alongside that the vehicle "
       "could not hold its line - it had to leave its lane, swing wide, or cross "
       "the centre line for it, though still without panic. An element riding or "
       "walking right beside the vehicle at speed belongs here even if the "
       "recorded motion barely changed: the margin was gone, whether or not the "
       "vehicle managed to use it.",
    4: "It is in the ego-vehicle's path with no margin left - closing fast, or "
       "appearing so near that only an emergency stop or a hard swerve could "
       "answer it. A collision, or a near miss that was avoided only by such a "
       "manoeuvre, belongs here.",
}

GATE_RUBRIC = {
    0: "A barrier or level crossing is visible but not on the ego-vehicle's "
       "route - it controls a side entrance, the opposite carriageway, or a "
       "way the vehicle never takes. The vehicle held its speed and its line.",
    1: "It controls the way the ego-vehicle is taking, but it was open, so the "
       "vehicle drove straight through without stopping or slowing for it.",
    2: "It closed the ego-vehicle's way - the barrier was down, or, where there "
       "is no barrier, a red light or flashing signal held traffic back - so "
       "the vehicle came to a stop and waited for the way to clear before "
       "going on.",
    3: "It closed as the ego-vehicle was about to pass, so the vehicle had "
       "to stop sharply.",
}


# 공사 구역 전용. Dynamic object 6종과 달리 공사는 움직이지 않고 도로 구조
# 자체를 바꾸므로, IMPACT 계열의 "무엇이 다가왔는가" 대신 "차선이 얼마나
# 먹혔는가"가 등급을 가른다.
#
# 0 과 1 을 위치로만 가르지 않는 이유: 위치는 연속량이라 어디서 잘라도
# 경계가 생긴다. "옆 차선까지 1점" 으로 좁히면 편도 4차선에서 자차 1차선 /
# 공사 4차선 이 어느 칸에도 안 들어가고, "반대 차선까지 1점" 으로 넓히면
# 왕복 8차선 반대편 끝 공사가 바로 옆 차선 공사와 같은 등급이 된다.
# 그래서 자차가 그 옆을 실제로 지나가는지로 가른다 - 차로 수를 세지 않아도
# 되고, 교차로에서 돌아나가 공사 쪽으로 아예 가지 않는 클립이 0 으로 빠진다.
#
# 등급 폭이 GATE_RUBRIC 과 같은 0~3 인 것은 의도된 것이다. 공사에는 IMPACT
# 4점(긴급 회피)에 해당하는 칸이 없다 - 공사 구역은 예고되고 유도되므로
# 급제동/급조향으로만 답할 수 있는 상황이 아니다. 빈 칸을 만들어 두면 모델이
# 그 칸을 채우려 하므로(실측: Animal 7건 전부 rarity=2) 아예 두지 않는다.
CONSTRUCTION_RUBRIC = {
    0: "The works lie away from where the ego-vehicle is going - off the "
       "roadway altogether, beyond a central reservation or a crash barrier, "
       "or somewhere the vehicle never draws level with because it turns off "
       "or leaves them behind. The vehicle held its speed and its line.",
    1: "The ego-vehicle drives past the works. They are on the roadway but not "
       "in its lane, so it carried on through at the same speed and on the "
       "same line.",
    2: "The works or their traffic cones take up part of the lane the "
       "ego-vehicle is driving in. The vehicle edged across to the far side of "
       "its own lane to get by, without leaving the lane.",
    3: "The lane the ego-vehicle was in is closed off - cones or barricades "
       "block it and guide traffic onto another way. The vehicle had to give "
       "up that lane and move into the next one or onto a temporary lane laid "
       "out for it.",
}


# 비포장 도로 전용. 여기서 어려운 것은 진동이 아니라 주행 가능 영역이
# 어디까지인지가 불확실하다는 점이므로, 경계의 선명도를 주축으로 삼는다.
#
# 두 조건(경계/노면)을 2x2 교차표로 늘어놓지 않는다. 0~3 은 순서 척도이고
# 평가는 MSE 라 칸의 대소가 의미를 가져야 하는데, 축이 둘이면 "경계는
# 뚜렷한데 심하게 파인 길" 과 "경계는 흐린데 노면은 매끈한 길" 중 무엇이
# 위인지 정해지지 않는다. 그래서 노면은 경계가 읽히는 구간(0~1)에서만
# 칸을 가르고, 경계가 무너진 2~3 에서는 쓰지 않는다 - 같은 조건을 두 곳에서
# 재사용하면 2 와 3 의 차이가 노면뿐이 되어 주축이 무의미해진다.
#
# 풀이 무성한 것은 2~3 의 근거가 아니다. 풀줄기는 "여기부터 길이 아니다" 를
# 보여주는 표시라 오히려 경계가 읽힌다는 뜻이다. 경계가 실제로 사라지는
# 것은 노면과 그 바깥이 같은 재질일 때다.
#
# 노면 상태를 젖음/눈/웅덩이로 서술하지 않는 이유: difficulty 의
# road_surface 축이 이미 그것을 재고 있다(1=unpaved or dusty road,
# 3=standing water/puddles, 4=deep snow-covered). 같은 어휘를 프롬프트 두
# 곳에 두면 모델이 한쪽 판단을 다른 쪽으로 복사한다 - DIFFICULTY_ONLY 가
# 존재하는 이유가 그것이다. 여기서는 지형의 요철(파임/자갈)만 쓰고, 3 은
# 원인을 적지 않고 결과만 말한다. 눈 때문에 경계가 사라진 클립도 그 문장에
# 그대로 해당하므로 적용 범위는 줄지 않는다.
UNPAVED_RUBRIC = {
    0: "The surface is unpaved, but it is clear how far the road reaches - the "
       "track and the ground beside it part cleanly in colour or in material, "
       "and the surface is reasonably even. The ego-vehicle held its speed and "
       "its line.",
    1: "The edges of the track are still clear, but the surface is uneven - "
       "rutted, or loose with coarse gravel. The ego-vehicle slowed down as it "
       "went over it.",
    2: "The track runs on into the ground beside it in the same material, so "
       "one of its edges cannot be made out. The other edge, or the wheel "
       "tracks left by whoever went before, still shows which way the road "
       "goes.",
    3: "The track and the ground around it read as one, so neither the width "
       "of the road nor its direction can be told from the terrain.",
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
#
# 요소가 여럿이면 최댓값을 쓴다(프롬프트 steps45 헤더에 명시). rubric 문장이
# 전부 단수 주어라 그냥 두면 모델이 장면을 하나로 뭉뚱그려 평균을 낸다 -
# 실측(20260904, 27,024클립): 특이요소가 1개든 3개든 rarity 평균이 2.00 으로
# 고정이고, safety 는 요소 3개 그룹이 오히려 낮았다(2개 1.53 -> 3개 1.36).
# 최댓값 규칙이 작동하면 요소가 늘수록 상한을 칠 확률이 올라가 평균이
# 올라가야 하므로, 이 고정은 평균내기의 흔적이다.
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

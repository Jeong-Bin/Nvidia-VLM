#!/usr/bin/env python3
"""JSON 출력에서 등급 필드를 정수로만 나오게 강제한다.

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
  (점수가 카테고리별로 바뀌면서 마커로 쓸 고정 키가 없어져 지금은 TIER_FIELDS
  가 비어 있고, 이 제약기는 실제로 쓰이지 않는다.)
  모델이 JSON 을 `"x": 1` 로 쓸지 `"x":1` 로 쓸지 미리 알 수 없으므로
  공백 허용 단계를 반드시 둬야 한다(실측: ': 1' 은 ['Ġ','1'] 2토큰,
  ':1' 은 ['1'] 1토큰).

주의:
  prefix_allowed_tokens_fn 이 빈 리스트를 반환하면 transformers 가 예외를
  던진다. 어떤 분기에서도 빈 리스트가 나오지 않도록 아래에서 항상 최소 한
  개 이상을 돌려준다.
"""
from __future__ import annotations

import re
from functools import lru_cache

# 등급 정의. 프롬프트/파서/집계가 모두 이 한 곳을 참조한다.
#
# 4 는 데이터셋에 사실상 없는 극단값(충돌 임박/발생, 차가 본 적 없을 법한
# 물체)이다. 일부러 넣어둔 이유는 척도 압축을 풀기 위해서다 - 상한이 3 이면
# 모델이 3 을 "최악"으로 취급해 아끼고 2 로 몰린다(실측 20260813: 정답 3인
# 20건 중 18건을 2로 예측). 위에 더 극단적인 칸을 두면 3 이 "최악"이 아니라
# "심각한 편"이 되어 쓰기 쉬워진다. 4 를 실제로 찾는 것은 목표가 아니다.
#
# 0 은 rubric 에 없다. 모든 rubric 이 1 에서 시작하고, 0 은 "그 카테고리가
# 이 클립에 없다" 는 뜻으로만 남는다 - 없는 카테고리는 애초에 점수를 받지
# 않으므로 모델이 0 을 낼 일도 없다. 값 자체는 계속 허용한다: GT 라벨이
# 채점에서 0 을 쓸 수 있고, 제약 디코딩도 0 을 막지 않는다.
TIER_LABELS = {0: "Absent", 1: "Low", 2: "Moderate", 3: "High", 4: "Extreme"}
TIER_VALUES = tuple(sorted(TIER_LABELS))          # (0, 1, 2, 3, 4)

# 강제 대상 필드. JSON 키 이름 그대로 쓴다.
#
# 점수가 클립당 두 개(safety/rarity)에서 "영상에 등장한 카테고리마다 하나"로
# 바뀌면서, 마커로 삼을 고정 키가 없어졌다 - 모델이 내는 키는 카테고리
# 이름이고 그것은 클립마다 다르다. 제약기는 그래서 이 모드에서 쓰지 않는다
# (make_tier_processor 에 빈 fields 를 주면 None 이 된다). 값이 지저분하게
# 나오면 _coerce_tier 가 흡수한다.
TIER_FIELDS = ()

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

# 눈금이 0 이 아니라 1 에서 시작한다.
#
# 0 은 원래 "그 요소가 장면에 아예 없다" 를 위한 칸이었는데, 없는 요소는
# 애초에 categories 에 적히지 않으므로 점수를 받을 일이 없다. 빈 칸을 두면
# 모델이 그것을 채우려 하므로(실측 20260812: Animal 7건이 내용과 무관하게
# 전부 rarity=2) 아예 없앤다.
# IMPACT_RUBRIC_0 = {
#     1: "It is on the roadway, but in another lane well away from the one the "
#        "ego-vehicle is driving in. The vehicle keeps its speed and its line, "
#        "and would have driven the same way had it not been there.",
#     2: "It is right beside the lane the ego-vehicle is driving in. Carrying "
#        "straight on would have been fine, but to keep a safe gap the vehicle "
#        "eased off a little or edged slightly to one side. This case applies "
#        "even if the recorded movement of the ego-vehicle barely changed "
#        "when this is located right next to a moving ego-vehicle,"
#        "regardless of whether the vehicle utilized that safe distance.",
#     3: "It is in the ego-vehicle's path ahead, or crosses in front of it - the "
#        "vehicle slowed, stopped, or steered aside to let it pass, or had to "
#        "leave its lane or cross the centre line for a moment. There was enough "
#        "time and distance to do so.",
#     4: "It is in the ego-vehicle's path with no margin left - closing fast, or "
#        "appearing so near that only an emergency stop or a hard swerve could "
#        "answer it. A collision, or a near miss that was avoided only by such a "
#        "manoeuvre, belongs here.",
# }


# for Animal on Road, Pedestrian on Road, Cyclist on Road
# (Emergency Vehicle 은 탐지 전용으로 뺐다 - RUBRIC_BY_CATEGORY 주석)


# DYNAMIC_RUBRIC = {
#     1: "The ego-vehicle was already stopped - for a red light, a queue or a "
#        "stop line - or barely moving at walking pace or slower, as when "
#        "creeping forward in a queue, before the pedestrian, cyclist or animal "
#        "came into the path ahead or passed close alongside. The vehicle "
#        "simply waited for it to pass.",
#     2: "The ego-vehicle was still moving when it saw the pedestrian, cyclist "
#        "or animal coming into the path ahead, and slowed down or came to a "
#        "stop for it in good time, or steered around it - even if the light "
#        "also turned amber or red at that moment. Or the vehicle kept its "
#        "speed and its line because the pedestrian, cyclist or animal moved "
#        "out of the path on its own or yielded the way first.",
#     3: "The pedestrian, cyclist or animal and the moving ego-vehicle came close "
#        "alongside each other - it moved past the vehicle, or it stood still at "
#        "the edge of the vehicle's lane while the vehicle went by. The vehicle "
#        "steered away from it - or, with no room to do so, kept its speed and "
#        "its line and went by within a very short distance of it.",
#     4: "The pedestrian, cyclist or animal got into the path of the moving "
#        "ego-vehicle without warning - it stepped, ran or darted out, came out "
#        "from behind something or out of the dark, or suddenly turned or "
#        "swerved in front of or beside the vehicle - so there was no chance to "
#        "see it coming. This counts whatever the vehicle then did, even if the "
#        "vehicle only slowed down. A collision, at any speed, belongs here too.",
# }

# SAFETY_FOR = {
#     0: "It is a scene of peaceful driving, with no elements on the road that threaten safety.",
#     1: "There are factors that could affect the ego-vehicle's driving, but it can handle the situation with "
#        "relative ease—without needing to change lanes or decelerate—or the objects are completely clear of "
#        "the ego-vehicle's driving path.",
#     2: "The ego-vehicle had to give way - slow, yield, wait, or steer around something "
#        "- but with plenty of time and space to do it.",
#     3: "The ego-vehicle had to act urgently, or a small mistake by anyone "
#        "would have caused a collision: hard braking, evasive steering, or "
#        "something entering its path at close range.",
# }

# 묶음 단위 난이도 기준표 (0~4).
#
# 점수를 카테고리마다 매기지 않고 scene_category.json 의 묶음(special
# scenario: Dynamic object / Driving environment)마다 하나씩 매긴다. 묶음
# 안의 카테고리는 한 기준표로 함께 잰다 - 새 카테고리가 묶음에 추가돼도
# 기준표를 새로 만들지 않는다.
#
# 0 은 모델에게 보여 주는 칸이다: "그 묶음의 특수 카테고리가 클립에 없음".
# 예전 safety 가 Normal 클립에도 0 을 줬던 것과 같다. 그래서 4단계에서
# 이름을 댄 묶음은 1 이상, 이름을 대지 않은 묶음은 0 이어야 하고, 어긋나면
# 채점(evaluate_labels)이 따로 센다.
#
# 빈 문자열인 칸이 있으면 프롬프트를 만들 때 멈춘다(check_group_rubrics).


# Dynamic object 기준표.
#
# 축은 "자차에게 남은 여유"(시간 + 공간)다. 거리만 쓰면 정차 중 앞을 지나는
# 보행자가 높게 나오고, 속도만 쓰면 인파 사이 서행이 낮게 나온다. 각 칸은
# 앞 칸보다 시간이나 공간 여유가 줄기만 하도록 잡았다(0 없음 -> 1 요구 없음
# -> 2 둘 다 충분 -> 3 공간 부족 -> 4 시간 없음). 다만 모델에게는 여유를 재라고
# 하지 않고, 그 여유를 드러내는 관찰 가능한 사실로만 적는다 - 단안 영상으로는
# 거리도 시간 여유도 재기 어렵고, "risk"/"average" 같은 판단어를 넣을 때마다
# 성능이 떨어졌다(급제동을 근거로 지정 시 66% -> 37%).
#
#   1 옆 차로의 트램처럼 경로가 겹치지 않는 평범한 교통, 정차 중 대기.
#     도로 밖 객체는 excludes 로 카테고리에서 빠지므로 여기 오지 않는다.
#   3 자전거처럼 옆으로 벗어날 수 있는 상대가 바로 옆에 있으면 공간 여유가
#     겉보기 간격보다 작다("could swerve or wobble"). 여러 도로 사용자를
#     동시에 상대한 장면도 여기다(5단계의 "situation is scored as one").
#   4 예고 없는 등장 - 자차 반응의 세기는 묻지 않는다. 실제 도로의 GT4 는
#     감속 1.7~3.7 m/s^2 로 일상 수준이었다(실측 20261006_143241).
#
# 객체는 "it" 이 아니라 "the road user" 로 부른다 - 앞 문장 주어가 자차면 it
# 이 자차로 읽힌다.
DIFFICULTY_RUBRIC_FOR_DYNAMIC_OBJECT = {
    0: "None of the road users listed for this group is in the clip. This is "
       "ordinary driving.",
       
    1: "A road user of this group is there, but it asked nothing of the "
       "ego-vehicle. It kept to its own lane, track or path at an ordinary "
       "distance, as normal traffic does, and never came into the vehicle's "
       "way - or the ego-vehicle was already stopped (for a red light, a queue "
       "or a stop line) or barely moving at walking pace, and simply waited "
       "for the road user to pass.", 
       # 특수 객체 있지만 자차에 영향 없음. 
       # 혹은 자차가 이미 정지해 있음.
       
    2: "The ego-vehicle was moving and saw the road user coming into its path "
       "ahead, with time and room to spare. It slowed down, stopped, gave way "
       "or steered around the road user in good time - even if a light also "
       "turned amber or red at that moment - or kept its speed and line "
       "because the road user moved aside or yielded first.", 
       # 자차는 주행 중. 피하거나 감속 시간 충분.
       # 혹은 상대가 먼저 비켜줘서 속도와 차선 유지한 경우.
       
    3: "The ego-vehicle saw the road user coming, but there was little room. "
       "The road user moved past, or stood at the edge of the vehicle's lane, "
       "within a short distance of the moving vehicle; or rode or walked right "
       "beside it where it could swerve or wobble into the vehicle's way; or "
       "the vehicle had to squeeze past, pull aside, or work its way through "
       "several road users at the same moment.",
       # 다가오는 상대를 발견했지만 공간 부족.
       # 자차 바로 옆을 지나가거나 차선 가장자리에 서 있음
       # 갑자기 방향을 틀거나 비틀거릴 가능성 있음
       # 또는 자차가 여러 객체들 사이를 비집고 지나가거나 갓길에서 그 사이를 지나가야 함
       
    4: "The road user got into the path of the moving ego-vehicle without "
       "warning - it stepped, ran, darted or pulled out, came out from behind "
       "something or out of the dark, or suddenly turned or swerved in front "
       "of or beside the vehicle - so there was no chance to see it coming. "
       "This counts whatever the vehicle then did, even if it only slowed "
       "down. A near miss or a collision, at any speed, belongs here too.",
       # 아무 예고 없이 갑자기 자차 앞으로 뛰어듦
       # 아슬아슬한 충돌 혹은 진짜 충돌
}

DIFFICULTY_RUBRIC_FOR_DRIVING_ENVIRONMENT = {
    0: "This is a normal driving situation. Nothing about the road or its "
       "surroundings - works, barriers, crossings, the road surface or objects "
       "on it - hinders driving.",
    1: "",
    2: "",
    3: "",
    4: "",
}


# ---------------------------------------------------------------------------
# 이하 카테고리별 기준표(1~4)는 묶음 단위로 바꾸기 전의 것이다. 지금은
# 프롬프트에 싣지 않는다 - 통합 기준표 문구를 쓸 때 참고하려고 남긴다.
# ---------------------------------------------------------------------------
DYNAMIC_RUBRIC = {
    1: "The ego-vehicle was already stopped - for a red light, a queue or a "
       "stop line - or barely moving at walking pace or slower, as when "
       "creeping forward in a queue, before it came into the path ahead or "
       "passed close alongside. The vehicle simply waited for it to pass.",
    2: "The ego-vehicle was still moving when it saw it coming into the path "
       "ahead, and slowed down or came to a stop for it in good time, or "
       "steered around it - even if the light also turned amber or red at "
       "that moment. Or the vehicle kept its speed and its line because it "
       "moved out of the path on its own or yielded the way first.",
    3: "It and the moving ego-vehicle came close alongside each other - it moved "
       "past the vehicle, or it stood still at the edge of the vehicle's lane while "
       "the vehicle went by. The vehicle steered away from it - or, with no room to "
       "do so, kept its speed and its line and went by within a very short distance of it.",
    4: "It got into the path of the moving ego-vehicle without warning - it "
       "stepped, ran or darted out, came out from behind something or out of "
       "the dark, or suddenly turned or swerved in front of or beside the "
       "vehicle - so there was no chance to see it coming. This counts "
       "whatever the vehicle then did, even if it only slowed down. A "
       "collision, at any speed, belongs here too.",
}


# for Road Construction
#
# 칸은 "공사가 어디까지 들어왔나"로 가른다 - 모델이 화면에서 확인할 수 있는
# 위치이고, 회피 정도나 "주의해서" 같은 판단어를 쓰지 않는다.
#   2/3 경계: 공사 옆이 차가 달릴 수 있는 곳인가. 예전 3("그 방향으로 차로를
#     바꿀 수 없다")은 옆 차로가 없는 좁은 도로에서 늘 참이라 2 를 3 으로
#     끌어갔다(실측 20261002_175140: GT2->P3 4건 중 메모 "Narrow road" 다수).
#   3/4 경계: 자차 차로 안으로 일부라도 들어왔는가. 예전 4 는 "차로가 닫혀
#     옮겨야 했다"만 받아, 콘이 차로에 일부 들어와 감속/정지만 한 경우를
#     모델이 3 으로 읽었다(GT4->P3 9건 중 "cones directly in the lane" 3건).
CONSTRUCTION_RUBRIC = {
    1: "A construction site, traffic cone, or traffic barricade is located very far from the ego-vehicle's lane. "
       "Therefore, the ego-vehicle maintained its speed and route regardless of its presence.",
    2: "The construction site or traffic cone is located next to the ego-vehicle. "
       "However, since this is an area where the ego-vehicle is normally unable to go "
       "—such as a pedestrian walkway, parking area, or the first lane of the opposite direction—there is no direct impact.",
    3: "The works sit right beside the ego-vehicle's lane, on a lane or road "
       "surface the ego-vehicle itself could otherwise drive on, such as a "
       "neighbouring lane going the same way. None of them - no cone, "
       "equipment or work area - comes into the ego-vehicle's own lane. The "
       "vehicle stayed in its lane and went past alongside them.",
    4: "Cones, equipment or the work area come into the ego-vehicle's own "
       "lane, even partly. Because of that the vehicle had to steer around "
       "them, slow down or stop, or follow the cones into another lane or onto "
       "a temporary lane laid out for it.",
}

# for Manual Traffic Control, Railway crossing, Barrier arm
GATE_RUBRIC = {
    1: "A barrier, a level crossing, or a person controlling traffic is visible "
       "but does not control the ego-vehicle's way - it controls a side "
       "entrance, the opposite carriageway, cross traffic, or a way the "
       "vehicle never takes. The vehicle held its speed and its line.",
    2: "It controls the way the ego-vehicle is taking, but that way was open - "
       "the barrier was up, or the person controlling traffic let the vehicle "
       "through, whether with a wave, by showing a sign that lets traffic go "
       "(such as SLOW), or simply by standing aside without stopping it - so "
       "the vehicle drove straight through without stopping or slowing for it.",
    3: "It closed the ego-vehicle's way - the barrier was down; where there is "
       "no barrier, a red light or flashing signal held traffic back; or the "
       "person controlling traffic signalled it to stop or sent it another way - "
       "so the vehicle came to a stop and waited for the way to clear, or turned "
       "off onto the way it was directed to.",
    4: "It closed as the ego-vehicle was about to pass - the barrier came "
       "down, the signal changed, or the person controlling traffic suddenly "
       "signalled it to stop - so the vehicle had to stop sharply.",
}

# for Unpaved road
#
# 한 축 - 길의 가장자리(와 차선)가 얼마나 읽히는가 - 로만 가른다. 노면의
# 요철은 쓰지 않는다: 날씨의 road_surface 축이 이미 노면 상태를 재고 있어
# (1 = unpaved or dusty road), 여기서 또 재면 두 축이 같은 것을 센다.
#
# 1 은 실제로는 포장 도로다. 흙먼지 때문에 모델이 비포장으로 잡는 일이 있어,
# 그 경우를 가장 낮은 칸으로 받아 둔다 - 난이도는 포장 도로와 같다.
UNPAVED_RUBRIC = {
    1: "The road is actually paved, but a thin layer of dirt or dust makes it "
       "look unpaved. Its lanes and edges stay clearly visible from start to "
       "finish.",
    2: "The road is unpaved, but its edges - and its lanes, if it has any - "
       "stay clearly visible from start to finish.",
    3: "The road is unpaved, and for most of the clip its edges and lanes "
       "cannot be made out, but trees, obstacles or wheel tracks along it "
       "still show roughly where it runs.",
    4: "The road is unpaved, and for most of the clip its edges and lanes are "
       "very hard to make out - nothing along it shows clearly where it runs.",
}


# for Obstacle on Road
OBSTACLE_RUBRIC = {
    1: "A small object - a branch, a bit of debris, a scrap of litter - lies "
       "in the ego-vehicle's path, small enough to drive over. The vehicle "
       "drove over it or past it without changing its speed or its line.",
    2: "An object lies right beside the ego-vehicle's path, partly over the "
       "edge of its lane. To keep a safe gap the vehicle eased off a little or "
       "edged slightly away from it.",
    3: "An object too large to drive over lies in the ego-vehicle's path "
       "ahead. The vehicle saw it from a distance and steered around it or "
       "came gradually to a stop.",
    4: "An object too large to drive over fell into the ego-vehicle's path, "
       "or came into view suddenly from behind the vehicle ahead. The vehicle "
       "had to brake hard, stop, or swerve urgently.",
}




# 묶음 -> (제목, 기준표). 키는 scene_category.json 의 special scenario 이름이다.
GROUP_RUBRICS = {
    "Dynamic object": (
        "how much the road users made driving harder for the ego-vehicle",
        DIFFICULTY_RUBRIC_FOR_DYNAMIC_OBJECT),
    "Driving environment": (
        "how much the road and its surroundings made driving harder for the "
        "ego-vehicle",
        DIFFICULTY_RUBRIC_FOR_DRIVING_ENVIRONMENT),
}


def group_key(group: str) -> str:
    """묶음 이름 -> 모델 출력 JSON 키. "Dynamic object" -> "dynamic_object"."""
    return re.sub(r"[^a-z0-9]+", "_", group.lower()).strip("_")


def score_groups(labels) -> list[str]:
    """점수를 매길 묶음 이름들, 메뉴 순서대로. labels 는 load_labels 결과."""
    out = []
    for lab in labels:
        if not lab.get("is_normal") and lab["scenario"] not in out:
            out.append(lab["scenario"])
    return out


def group_of(labels) -> dict:
    """{카테고리 이름: 묶음 이름}."""
    return {lab["category"]: lab["scenario"] for lab in labels
            if not lab.get("is_normal")}


def group_values(group: str) -> tuple[int, ...]:
    """그 묶음 기준표의 눈금(0~4)."""
    return tuple(sorted(GROUP_RUBRICS[group][1]))


def check_group_rubrics(labels) -> None:
    """프롬프트에 실을 기준표가 다 채워졌는지. 아니면 ValueError.

    빈 칸이 조용히 프롬프트에 들어가면 모델에게 설명 없는 눈금을 주게 된다.
    """
    problems = []
    for g in score_groups(labels):
        if g not in GROUP_RUBRICS:
            problems.append(f"{g!r}: GROUP_RUBRICS 에 기준표가 없음")
            continue
        empty = [v for v, t in sorted(GROUP_RUBRICS[g][1].items())
                 if not str(t).strip()]
        if empty:
            problems.append(f"{g!r}: {empty} 점 문구가 비어 있음")
    if problems:
        raise ValueError("묶음 기준표를 먼저 채워야 합니다 - "
                         + "; ".join(problems))


def group_rubric_blocks(labels) -> str:
    """프롬프트 5단계에 넣을 묶음별 기준표.

    각 표 머리에 그 묶음에 속한 카테고리를 적어, 모델이 어느 표로 어느
    종류를 재는지 알게 한다. 카테고리별 score_examples(scene_category.json)는
    그 카테고리가 속한 묶음 표 아래에 붙인다 - 점수는 묶음 눈금(0~4)이다.
    """
    check_group_rubrics(labels)
    members = {}
    examples = {}
    for lab in labels:
        if lab.get("is_normal"):
            continue
        g = lab["scenario"]
        members.setdefault(g, []).append(lab["category"])
        for ex in lab.get("score_examples") or []:
            situation, score = ex
            if score not in GROUP_RUBRICS[g][1]:
                # 없는 칸을 가르치는 예시는 실행 전에 막는다.
                raise ValueError(
                    f"score_examples of {lab['category']!r}: {score!r} is not "
                    f"a score of the {g!r} rubric")
            examples.setdefault(g, []).append((situation, score))

    out = []
    for g in score_groups(labels):
        title, rubric = GROUP_RUBRICS[g]
        vals = group_values(g)
        block = (f"   {g} ({', '.join(members[g])}) - {title} "
                 f"({min(vals)}-{max(vals)}):\n"
                 + "\n".join(f"     {v} = {rubric[v]}" for v in vals))
        if examples.get(g):
            block += "\n     For example:\n" + "\n".join(
                f"       {sit} -> {sc}" for sit, sc in examples[g])
        out.append(block)
    return "\n".join(out)


# 점수 예시는 코드가 아니라 scene_category.json 의 카테고리별
# "score_examples" 에서 읽는다 - group_rubric_blocks 가 그 카테고리가 속한
# 묶음 표 아래에 싣는다.
#
# 예전에는 여기 CONTRAST_EXAMPLES 4쌍("... -> low / ... -> high")을 두었다.
# safety 등급 시절 예시라 카테고리별 기준표와 충돌해 지웠다(실측
# 20261006_134008: 근거에 "-> low/high" 를 쓴 7건이 전부 틀렸다).


# 시각화 폴더를 나누는 점수 - 묶음 점수 중 최댓값.
#
# "이 클립에서 가장 어려웠던 묶음이 몇 점인가"가 검수 우선순위다. 0 이
# 정상 값이므로 Normal 클립은 score_0 으로 간다.
SCORE_MIN, SCORE_MAX = min(TIER_VALUES), max(TIER_VALUES)


def tier_score(group_scores) -> int | None:
    """묶음 점수 중 최댓값. 읽을 수 있는 값이 하나도 없으면 None.

    group_scores 는 {묶음 이름: 0~4} 딕셔너리다. 점수를 끈 실행처럼 값이
    아예 없을 때만 None 이다 - 0 은 "특수 카테고리 없음"이라는 정상 값이다.
    """
    if not isinstance(group_scores, dict):
        return None
    vals = []
    for v in group_scores.values():
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n in TIER_LABELS:
            vals.append(n)
    return max(vals) if vals else None


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

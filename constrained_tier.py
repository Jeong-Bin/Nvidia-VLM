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
# DYNAMIC_RUBRIC_0 = {
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
#
# 이 점수는 edge-case 인지가 아니라 그 객체가 주행 난이도를 얼마나 올렸는지를
# 잰다. 그래서 자차 경로에 들어오지도 않고 옆을 가깝게 스쳐 가지도 않는 객체
# (먼 차선, 반대편, 인도, 울타리 너머)는 점수를 매기지 않고 카테고리에서
# 뺀다 - scene_category 의 excludes. "장면에 없음"과 "있지만 멀리 있음"이 둘
# 다 난이도 0 이라 한 규칙으로 합쳤고, 다른 rubric 들도 1 부터 시작하므로
# 눈금이 통일된다.
#
# 칸은 자차 상태와 여유로 가른다(횡단보도 여부는 쓰지 않는다):
#   1 자차가 서 있거나 걷는 속도 이하로 기어가고 있었다(앞이든 옆이든)
#     - egomotion 의 정지 판정(1.8 km/h 미만)만 정지로 치면 정체 속 2~5 km/h
#       서행이 "주행 중"으로 읽혀 2 로 간다. 난이도는 정차와 같으므로 묶는다.
#   2 달리는 중, 앞쪽인데 미리 감속/정지하거나 비켜 갔다
#   3 달리는 중, 옆을 가깝게
#   4 예고 없이 경로에 들어왔다(자차가 어떻게 대응했든)
#
# 1 과 2 는 "원래 서 있었나(already stopped)"와 "달리다가 그것 때문에
# 섰나"로 가른다. 예전 1 은 "서 있는 동안 들어왔다"는 시점만 말해, 보행자를
# 보고 미리 감속해 멈춘 뒤 보행자가 건넌 장면도 "선 뒤에 건넜다"로 읽혀
# 1 이 됐다(실측 20261006_104238: GT2->P1 10건 중 근거가 "while the
# vehicle was stopped" 인 것 7건, 그중 d1248bb6 등은 보행자 때문에 선 것).
# 신호가 바뀌어 섰는데 보행자도 함께 원인이면 2 로 둔다 - 라벨이 그 기준이다.
# 신호는 덧붙이는 조건으로만 적는다: 2 의 근거가 "신호에 섰다" 자체가 되면
# 멀리 있는 자전거/동물도 빨간불 정차만으로 2 가 된다.
#
# 그 덧붙임을 처음에 "even when a light turning amber or red also made the
# vehicle stop" 으로 썼더니 반대로 무너졌다(실측 20261006_125459: DYNAMIC
# GT1->P2 7건, 근거가 "was stopped at the red light while cyclists crossed").
# 이미 빨간불에 서 있던 장면까지 2 로 끌어온 것이다. 그래서 1 에 "원래 왜
# 서 있었나(red light/queue/stop line)"를 적고, 2 의 신호 문장은 "달리다가
# 그것을 보고 섰는데 '그 순간' 신호도 바뀐 경우"로 좁혔다.
#
# 2 와 4 는 "다가오는 것을 미리 볼 수 있었나" 로 가른다 - 자차 대응의 세기가
# 아니다. 예전 4 는 "brake hard, stop, or swerve urgently" 를 요구했는데,
# 라벨의 4 는 실제 도로에서 반응이 일상 수준인 "예고 없는 등장"이었다
# (무단횡단, 눈길의 사슴, 뛰어든 개, 밤길 자전거 - 최대 감속 1.7~3.7 m/s^2).
# 모델은 egomotion 의 3 m/s^2 대 감속을 보고 "in good time" 이라며 2 를
# 줬다(실측 20261006_134008/143241/152641: DYNAMIC 4 예측 0건). 감속이 큰
# GT4 3건(9.5~11.6 m/s^2)은 인형과 천천히 부딪히는 모의 실험 영상의 충격
# 값이라 제동 신호로 쓰지 않는다 - 대신 충돌은 속도와 상관없이 4 로 둔다.
#
# 급제동을 근거로 쓰지 않는 원칙은 그대로다 - 이 데이터의 급제동은 대개
# 신호/정체 때문이라 급제동만 보고 4 를 주면 틀린다(실측: 급제동을
# 근거로 지정하자 정확도 66% -> 37%). 단안 카메라로는 절대 거리도, 시간
# 여유(거리/속도)도 재기 어려워 거리로 가르지도 않는다.
#
# 무단횡단을 4 의 조건으로 적지 않는다. 예전에 Jaywalking 과탐이 가장 큰
# 오류원이었다 - 건널목 밖에서 건너도 멀리서부터 보였으면 2 다.
# 대상은 "it" 대신 "the pedestrian, cyclist or animal" 로 적는다. 1 칸의
# "The ego-vehicle was already stopped ... before it came into the path" 처럼
# 앞 문장 주어가 자차면 it 이 문법상 자차를 가리키고, 4 칸의 "even if it only
# slowed down" 은 실제로 자차를 뜻해 같은 표 안에서 it 의 대상이 섞였다.
# "object" 는 쓰지 않는다 - 사람/동물에 어색하고 Obstacle on Road 의
# "Unknown object" 와 겹친다. 이 rubric 을 쓰는 카테고리가 정확히 이 셋이다.

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




# 점수 예시는 코드가 아니라 scene_category.json 의 카테고리별
# "score_examples" 에서 읽는다 - rubric_blocks 가 그 rubric 표 바로 아래에 싣는다.
#
# 예전에는 여기 CONTRAST_EXAMPLES 4쌍("... -> low / ... -> high")을 두고 4단계
# 머리에 실었다. 원래 목적은 객체 이름으로 점수를 정하는 패턴 매칭을 막는
# 것이었다(실측 20260812: Animal 7건이 내용과 무관하게 전부 rarity=2). 그런데
# safety 등급 시절의 예시라 지금 rubric 과 정면으로 충돌했다 - 4쌍 중 3쌍
# (인도 위 개, 자전거 도로 위 자전거, 신호 받고 건너는 보행자 -> low)이 이제는
# excludes 로 빠지거나 다른 점수여야 하는 장면이고, "low/high" 라는 말은 어느
# rubric 에도 없다. 실측(20261006_134008): DYNAMIC 근거에 "-> low/high" 를 쓴
# 7건이 전부 틀렸고, sidewalk/bike lane 을 쓴 21건 중 13건, "stepped into"
# 를 쓴 12건 중 8건이 틀렸다.
#
# 예시를 카테고리 정의 파일로 옮긴 이유: 예시는 rubric 을 고칠 때마다 같이
# 고쳐야 하는 데이터이고, 파일을 바꿔 끼우는 것만으로 "예시 없음"(파일에
# 없으면 블록이 빠진다)과 "새 예시"를 A/B 할 수 있다.


def _fmt_rubric(rubric: dict) -> str:
    """rubric 의 실제 눈금만 돈다 - 표가 1 에서 시작하므로 TIER_VALUES(0 포함)
    를 그대로 돌면 KeyError 가 난다."""
    return "\n".join(f"     {v} ({TIER_LABELS[v]}) = {rubric[v]}"
                     for v in sorted(rubric))


# 카테고리 -> rubric 배정.
#
# scene_category.json 의 scenario 이름이 그대로 묶음 단위다 - Dynamic object
# 6종은 "무엇이 다가왔는가"라 DYNAMIC 하나를 공유하고, Driving environment 는
# 카테고리마다 성격이 달라 따로 준다.
#
# 이름이 아니라 묶음으로 배정하는 이유: 카테고리마다 rubric 을 따로 주면
# 모델이 rubric 문장이 아니라 카테고리 이름으로 패턴 매칭한다(실측 20260812:
# Animal 7건이 내용과 무관하게 전부 rarity=2). 같은 rubric 을 공유하면
# "같은 객체 x 다른 행동 = 다른 점수" 가 유지된다.
RUBRIC_BY_SCENARIO = {
    "Dynamic object": "dynamic",
}
RUBRIC_BY_CATEGORY = {
    "Road Construction": "construction",
    "Railway crossing": "gate",
    "Barrier arm": "gate",
    # Dynamic object 묶음이지만 DYNAMIC 가 아니라 GATE 를 쓴다(GATE_RUBRIC
    # 위 주석). 카테고리 이름 배정이 묶음 배정보다 먼저 적용된다.
    "Manual Traffic Control": "gate",
    "Unpaved road": "unpaved",
    "Obstacle on Road": "obstacle",
    # 탐지 전용 - 찾기만 하고 점수는 매기지 않는다(None).
    #
    # 이 둘은 VLM 이 아니면 찾을 방법이 없어 탐지할 가치는 있지만, 이
    # 데이터에서 주행 난이도를 따로 올리지는 않는다. 긴급차량은 3D 라벨에
    # 클래스가 없고(automobile/heavy_truck 으로 들어간다), 트램은
    # train_or_tram_car 가 있지만 3D 라벨이 1,954클립뿐이다. 그런데 실제
    # 장면은 길가에 선 차량이거나 지나가는 차량이라(평가 20260922_151404:
    # 긴급차량 예측 7건 모두 ego=unaffected), DYNAMIC 로 재면 같은 자리의
    # 택배 트럭과 같은 점수가 나온다 - 그 점수는 이 카테고리에 대한 것이
    # 아니다. 길을 비켜 주는 장면이 데이터에서 나오면 그때 전용 rubric 을
    # 만든다.
    #
    # 여기서 정하는 이유: 탐지 전용 여부는 클립마다 고르는 것이 아니라
    # 카테고리의 성질이다. GT 라벨에는 {"Emergency Vehicle": null} 로
    # 적고(-1 은 "아직 안 매김"이라 뜻이 다르다), 읽는 쪽이 이 표와 어긋나는
    # 라벨을 경고한다.
    "Emergency Vehicle": None,
    # 2.5 부터 "Tram vehicle". "Tram" 한 단어면 모델이 "tram tracks" 를 쓰는
    # 순간 이름으로 매칭해 선로만 있는 장면을 올렸다(실측 20261006_143241:
    # 28d7fbd1, c7928d92, dc257402 - excludes 에 "tracks alone" 이 있는데도).
    # 옛 이름은 2.4 이하 파일을 계속 돌릴 수 있게 남긴다.
    "Tram vehicle": None,
    "Tram": None,
}

# rubric 본문. 이름 -> (제목, 표).
#
# 네 rubric 모두 1~4 로 폭이 같다. 시작이 1 인 이유는 DYNAMIC_RUBRIC 위
# 주석에 적었다 - 0 은 "카테고리가 없음"이고, 없는 카테고리는 점수를 받지
# 않는다.
RUBRICS = {
    "dynamic":       ("how much it affected the ego-vehicle", DYNAMIC_RUBRIC),
    "gate":         ("how much the barrier, crossing or person controlling "
                     "traffic held the ego-vehicle up",
                     GATE_RUBRIC),
    "construction": ("how far the works reached into the ego-vehicle's lane",
                     CONSTRUCTION_RUBRIC),
    "unpaved":      ("how clearly the edges of the road can be made out",
                     UNPAVED_RUBRIC),
    "obstacle":     ("how much the object got in the ego-vehicle's way",
                     OBSTACLE_RUBRIC),
}


def rubric_name_for(category: str, scenario: str = "") -> str | None:
    """이 카테고리가 쓸 rubric 이름. 모르는 카테고리는 dynamic 으로 떨어뜨린다.
    탐지 전용 카테고리는 None."""
    if category in RUBRIC_BY_CATEGORY:
        return RUBRIC_BY_CATEGORY[category]
    return RUBRIC_BY_SCENARIO.get(scenario, "dynamic")


def is_scored(category: str) -> bool:
    """점수를 매기는 카테고리인가. 탐지 전용(RUBRIC_BY_CATEGORY 에서 None)만
    False 다 - 모르는 이름도 dynamic 으로 떨어지므로 True."""
    return RUBRIC_BY_CATEGORY.get(category, "") is not None


def detect_only_categories(labels) -> list[str]:
    """labels(load_labels 결과) 중 탐지 전용 카테고리 이름, 메뉴 순서대로."""
    return [lab["category"] for lab in labels
            if not lab.get("is_normal") and not is_scored(lab["category"])]


def rubric_values(name: str) -> tuple[int, ...]:
    """그 rubric 이 실제로 쓰는 눈금. 폭이 rubric 마다 다르다."""
    return tuple(sorted(RUBRICS[name][1]))


def rubric_text(name: str) -> str:
    """rubric 하나를 프롬프트 문구로."""
    return _fmt_rubric(RUBRICS[name][1])


def rubric_blocks(labels) -> str:
    """프롬프트에 넣을 rubric 전체.

    labels(load_labels 결과)에 실제로 들어 있는 카테고리만 훑어 필요한
    rubric 만 싣는다 - scene_category.json 에서 카테고리를 빼면 그 rubric 도
    저절로 빠진다. 각 rubric 아래에 그것을 쓰는 카테고리를 적어, 모델이
    어느 표를 봐야 하는지 이름으로 찾게 한다.
    """
    used = {}
    for lab in labels:
        if lab.get("is_normal"):
            continue
        name = rubric_name_for(lab["category"], lab.get("scenario", ""))
        if name is None:
            continue
        used.setdefault(name, []).append(lab["category"])

    # 카테고리별 점수 예시(scene_category.json 의 score_examples). 그
    # 카테고리가 쓰는 rubric 표 아래에 붙인다 - 숫자가 어느 눈금의 것인지
    # 모델이 헷갈리지 않게.
    examples = {}
    for lab in labels:
        if lab.get("is_normal"):
            continue
        name = rubric_name_for(lab["category"], lab.get("scenario", ""))
        for ex in lab.get("score_examples") or []:
            situation, score = ex
            if name is None or score not in RUBRICS[name][1]:
                # 잘못된 예시가 조용히 프롬프트에 들어가면 모델에게 없는 칸을
                # 가르치게 된다 - 실행 전에 멈춘다.
                raise ValueError(
                    f"score_examples of {lab['category']!r}: {score!r} is not "
                    f"a score of the {name!r} rubric")
            examples.setdefault(name, []).append((situation, score))

    out = []
    # RUBRICS 에 등록된 순서대로 돈다. 이름을 여기 따로 적어 두면 새 rubric
    # 을 등록하고도 이 목록에 빠뜨렸을 때 프롬프트에서 조용히 사라진다.
    for name in RUBRICS:
        if name not in used:
            continue
        title, rubric = RUBRICS[name]
        vals = tuple(sorted(rubric))
        block = (f"   For {', '.join(used[name])} - {title} "
                 f"({min(vals)}-{max(vals)}):\n"
                 + "\n".join(f"     {v} = {rubric[v]}" for v in vals))
        if examples.get(name):
            block += "\n     For example:\n" + "\n".join(
                f"       {sit} -> {sc}" for sit, sc in examples[name])
        out.append(block)
    return "\n".join(out)


# 시각화 폴더를 나누는 점수.
#
# 예전에는 safety + rarity 합(0~8)이었다. 이제 점수가 카테고리마다 따로
# 붙으므로 합이 카테고리 수에 따라 달라져 폴더 이름으로 못 쓴다. 대신 그
# 클립에서 가장 높은 점수 하나를 쓴다 - "이 클립에서 가장 심한 요소가
# 몇 점인가"가 검수 우선순위이고, 요소가 여럿이면 최댓값을 쓴다는 프롬프트
# 규칙과도 같은 기준이다.
SCORE_MIN, SCORE_MAX = min(TIER_VALUES), max(TIER_VALUES)


def tier_score(category_scores) -> int | None:
    """카테고리별 점수 중 최댓값. 읽을 수 있는 값이 하나도 없으면 None.

    category_scores 는 {카테고리명: 점수} 딕셔너리다. 빈 딕셔너리(=특이
    요소 없음)도 None 이다 - "점수가 0" 과 "매길 대상이 없음" 은 다르다.
    """
    if not isinstance(category_scores, dict):
        return None
    vals = []
    for v in category_scores.values():
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

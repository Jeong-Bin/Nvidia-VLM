

DEFAULT_PROMPT = (
    "You are assessing how difficult this driving scene is for an autonomous "
    "vehicle, based on this front-facing camera image.\n"
    "Rate the DRIVING DIFFICULTY on an integer scale from 0 to 4:\n"
    "  0 = very easy (clear, bright, empty road)\n"
    "  1 = easy\n"
    "  2 = moderate\n"
    "  3 = hard\n"
    "  4 = very hard (severe adverse conditions / very complex)\n"
    "Consider factors such as scene brightness / low light, rain, snow, fog, "
    "glare, road clutter, and traffic density.\n"
    "Answer with the single difficulty integer on the first line, then one short "
    "sentence justifying it."
)


ILLUMINATION_PROMPT = (
    "You are assessing the ILLUMINATION difficulty of this driving scene for an "
    "autonomous vehicle (time of day / ambient light), based on this front-facing "
    "camera image.\n"
    "Rate the ILLUMINATION difficulty on an integer scale from 0 to 4:\n"
    "  0 = full daylight, evenly lit, scene clearly readable everywhere\n"
    "  1 = overcast or flat daylight; reduced contrast but full visibility, or a brief moment shading caused by an overpass\n"
    "  2 = twilight (dusk/dawn) or well-lit night with dense street lighting or city Lights, or tunnel, or strong contrast caused by backlighting\n"
    "  3 = night with partial or dim street lights, or the street lights make the road visible but the surrounding areas are dark\n"
    "  4 = only the ego-vehicle's headlights illuminate the road, or near-total darkness (unlit road) where much of the scene is not resolvable\n"
    "Judge ONLY illumination / ambient light, not weather or road condition.\n"
    "Answer with the single difficulty integer on the first line, then one short "
    "sentence justifying it."
)


PRECIPITATION_PROMPT = (
    "You are assessing the PRECIPITATION difficulty of this driving scene for an "
    "autonomous vehicle (falling weather), based on this front-facing camera "
    "image.\n"
    "Rate the PRECIPITATION difficulty on an integer scale from 0 to 4:\n"
    "  0 = none / clear\n"
    "  1 = drizzle or very light rain or snowfall; occasional drops on lens\n"
    "  2 = moderate rain or snowfall, raindrops or snow partially obscuring the camera\n"
    "  3 = heavy rain or snowfall, raindrops or snow largely obscuring the camera\n"
    "  4 = visibility is severely restricted because the camera is completely covered by raindrops or snow\n"
    "Judge ONLY falling precipitation, not ambient light, road surface, or fog.\n"
    "Answer with the single difficulty integer on the first line, then one short "
    "sentence justifying it."
)


ROAD_SURFACE_PROMPT = (
    "You are assessing the ROAD SURFACE STATE difficulty of this driving scene "
    "for an autonomous vehicle, based on this front-facing camera image.\n"
    "Rate the ROAD SURFACE difficulty on an integer scale from 0 to 4:\n"
    "  0 = dry\n"
    "  1 = damp, no standing water, or snow only off the roadway, unpaved or dusty road\n"
    "  2 = clearly wet and reflective; light spray; or thin snow on the roadway but lane markings still discernible\n"
    "  3 = standing water, puddles, slush, or snow covers most of the roadway or the lane markings are faint\n"
    "  4 = deep snow-covered, icy, or flooded; lane markings entirely obscured\n"
    "Judge ONLY the state of the road surface, not the falling weather or sky.\n"
    "Answer with the single difficulty integer on the first line, then one short "
    "sentence justifying it."
)


ATMOSPHERIC_OBSCURANTS_PROMPT = (
    "You are assessing the ATMOSPHERIC OBSCURANTS difficulty of this driving "
    "scene for an autonomous vehicle (fog / haze / spray), based on this "
    "front-facing camera image.\n"
    "Rate the ATMOSPHERIC OBSCURANTS difficulty on an integer scale from 0 to 4:\n"
    "  0 = clear, long sight line\n"
    "  1 = slight haze\n"
    "  2 = moderate fog/haze/spray; distant objects blurred\n"
    "  3 = dense fog/spray; only the near field visible\n"
    "  4 = very dense; minimal visibility beyond the immediate foreground\n"
    "Judge ONLY airborne obscurants (fog/haze/spray), not darkness or rain "
    "intensity itself.\n"
    "Answer with the single difficulty integer on the first line, then one short "
    "sentence justifying it."
)


# Factor key -> prompt, for iterating over the four difficulty factors and
# weighted-summing their scores into a final difficulty level.
FACTOR_PROMPTS = {
    "illumination": ILLUMINATION_PROMPT,
    "precipitation": PRECIPITATION_PROMPT,
    "road_surface": ROAD_SURFACE_PROMPT,
    "atmospheric_obscurants": ATMOSPHERIC_OBSCURANTS_PROMPT,
}

# ---------------------------------------------------------------------------
# 클립 모드 통합용.
#
# 위 프롬프트들은 "이미지 한 장 + 정수 하나 + 문장 하나"를 전제로 따로
# 물어보게 쓰여 있다. 클립 모드에 그대로 붙이면 한 클립에 모델을 5번 더
# 부르게 되어 비용이 6배가 된다. 그래서 문구는 그대로 두되(단독 호출도
# 계속 되도록) 같은 기준을 nuReasoning JSON 한 번에 얹는 형태로 옮긴다.
#
# 축 순서는 시각화/집계 표의 행 순서이기도 하다.
DIFFICULTY_MIN, DIFFICULTY_MAX = 0, 4

# (JSON 키, 화면에 쓸 이름). DEFAULT_PROMPT(전체 난이도)는 더 이상 모델에게
# 묻지 않는다 - 요인 4개와 따로 매기게 하면 서로 모순되는 답이 나오고(네 요인이
# 0인데 전체가 3인 식), 어떤 가중치가 옳은지도 근거가 없었다. 필요하면 요인
# 값으로 나중에 계산한다. 단독 프롬프트 문구(DEFAULT_PROMPT)는 남겨 둔다.
DIFFICULTY_OVERALL = ("driving_difficulty", "DRIVING DIFFICULTY")
DIFFICULTY_FACTORS = [
    ("illumination", "Illumination"),
    ("precipitation", "Precipitation"),
    ("road_surface", "Road_surface"),
    ("atmospheric_obscurants", "Atmospheric_obscurants"),
]
# 집계/시각화가 도는 축 = 요인 4개. 전체 난이도는 빠졌다(위 주석).
DIFFICULTY_AXES = list(DIFFICULTY_FACTORS)


# 각 축의 0~4 눈금. 위 단독 프롬프트의 눈금을 그대로 옮긴 것이라 둘이
# 어긋나면 안 된다 - 한쪽만 고치는 일이 없도록 여기 한 곳에만 적는다.
DIFFICULTY_SCALES = {
    "driving_difficulty": (
        "   0 = very easy (clear, bright, empty road)\n"
        "   1 = easy\n"
        "   2 = moderate\n"
        "   3 = hard\n"
        "   4 = very hard (severe adverse conditions / very complex)"),
    "illumination": (
        "   0 = full daylight, evenly lit, scene clearly readable everywhere\n"
        "   1 = overcast or flat daylight; reduced contrast but full visibility, or a brief moment shading caused by an overpass\n"
        "   2 = twilight (dusk/dawn) or well-lit night with dense street lighting or city Lights, or tunnel, or strong contrast caused by backlighting\n"
        "   3 = night with partial or dim street lights, or the street lights make the road visible but the surrounding areas are dark\n"
        "   4 = only the ego-vehicle's headlights illuminate the road, or near-total darkness (unlit road) where much of the scene is not resolvable"),
    "precipitation": (
        "   0 = none / clear\n"
        "   1 = drizzle or very light rain or snowfall; occasional drops on lens\n"
        "   2 = moderate rain or snowfall, raindrops or snow partially obscuring the camera\n"
        "   3 = heavy rain or snowfall, raindrops or snow largely obscuring the camera\n"
        "   4 = visibility is severely restricted because the camera is completely covered by raindrops or snow"),
    "road_surface": (
        "   0 = dry\n"
        "   1 = damp, no standing water, or snow only off the roadway, unpaved or dusty road\n"
        "   2 = clearly wet and reflective; light spray; or thin snow on the roadway but lane markings still discernible\n"
        "   3 = standing water, puddles, slush, or snow covers most of the roadway or the lane markings are faint\n"
        "   4 = deep snow-covered, icy, or flooded; lane markings entirely obscured"),
    "atmospheric_obscurants": (
        "   0 = clear, long sight line\n"
        "   1 = slight haze\n"
        "   2 = moderate fog/haze/spray; distant objects blurred\n"
        "   3 = dense fog/spray; only the near field visible\n"
        "   4 = very dense; minimal visibility beyond the immediate foreground"),
}

# 각 요인에서 "무엇만 보라"는 한정. 단독 프롬프트의 "Judge ONLY ..." 줄을
# 옮긴 것으로, 축끼리 서로 번지는 것을 막는 유일한 장치다.
DIFFICULTY_ONLY = {
    "illumination": "ambient light / time of day only - not weather or road state",
    "precipitation": "falling precipitation only - not ambient light, road surface, or fog",
    "road_surface": "the road surface state only - not the falling weather or sky",
    "atmospheric_obscurants": "airborne fog/haze/spray only - not darkness or rain intensity",
}


def difficulty_block() -> str:
    """nuReasoning 프롬프트에 얹을 난이도 단계 본문.

    요인 4개만 묻는다. 전체 난이도(DEFAULT_PROMPT)는 빼기로 했다 - 요인과
    따로 매기게 하면 모델이 둘을 무관하게 찍어 모순된 답이 나왔고(네 요인이
    모두 0인데 전체가 3인 식), 요인을 가중합할 옳은 식도 근거가 없었다.
    넷 중 가장 나쁜 것이 지배하는 장면(짙은 안개)과 여럿이 겹쳐야 어려워지는
    장면(야간 + 젖은 노면)이 섞여 있어 한 식으로 누르기 어렵다. 전체 난이도가
    필요해지면 요인 값에서 계산한다.
    """
    lines = []
    for key, name in DIFFICULTY_FACTORS:
        lines.append(f"   {name} ({key}) - judge {DIFFICULTY_ONLY[key]}:\n"
                     f"{DIFFICULTY_SCALES[key]}")
    factors = "\n".join(lines)
    return f"""   Rate the driving conditions of this scene on an integer scale from
   {DIFFICULTY_MIN} to {DIFFICULTY_MAX}. Rate each factor on its own, judging ONLY what that
   factor names and ignoring the others:
{factors}
   Give each rating one short sentence saying what you saw that justifies it.
   These ratings describe conditions, not edge-cases: a clear empty road at
   noon is {DIFFICULTY_MIN} everywhere, and rating it so is the correct answer.
"""

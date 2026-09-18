

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
    
    "  1 = overcast or flat daylight; reduced contrast but full visibility\n"
    
    "  2 = twilight (dusk/dawn) or well-lit night with dense street lighting; readable but dimmer\n"
    
    "  3 = night with partial or intermittent lighting; large dark regions, reliance on headlights\n"
    
    "  4 = near-total darkness (unlit road) where much of the scene is not resolvable\n"
    
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
    "  1 = drizzle or very light rain; occasional drops on lens\n"
    
    "  2 = steady moderate rain, or light snow falling\n"
    
    "  3 = heavy rain or moderate snowfall that visibly cuts sight distance\n"
    
    "  4 = downpour / heavy snow / blizzard; visibility severely reduced\n"
    
    "Judge ONLY falling precipitation, not ambient light, road surface, or fog.\n"
    "Answer with the single difficulty integer on the first line, then one short "
    "sentence justifying it."
)


ROAD_SURFACE_PROMPT = (
    "You are assessing the ROAD SURFACE STATE difficulty of this driving scene "
    "for an autonomous vehicle, based on this front-facing camera image.\n"
    "Rate the ROAD SURFACE difficulty on an integer scale from 0 to 4:\n"
    "  0 = dry\n"
    "  1 = damp, no standing water\n"
    
    "  2 = clearly wet and reflective; light spray\n"
    
    "  3 = standing water, puddles, slush, or partial snow cover\n"
    
    "  4 = snow-covered, icy, or flooded; lane markings obscured\n"
    
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

# (JSON 키, 화면에 쓸 이름). DEFAULT_PROMPT 는 전체 난이도라 따로 둔다.
DIFFICULTY_OVERALL = ("driving_difficulty", "DRIVING DIFFICULTY")
DIFFICULTY_FACTORS = [
    ("illumination", "Illumination"),
    ("precipitation", "Precipitation"),
    ("road_surface", "Road_surface"),
    ("atmospheric_obscurants", "Atmospheric_obscurants"),
]
# 집계/시각화가 도는 전체 축 (전체 난이도 + 4개 요인).
DIFFICULTY_AXES = [DIFFICULTY_OVERALL] + DIFFICULTY_FACTORS


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
            환한 대낮, 고른 조명, 화면 전체가 명확하게 식별됨
        "   1 = overcast or flat daylight; reduced contrast but full visibility or partial shading caused by an overpass\n"
            흐리거나 평이한 자연광; 대비는 낮지만 시야는 온전히 확보됨, 고가도로 그늘
        "   2 = twilight (dusk/dawn) or well-lit night with dense street lighting or city Lights; readable but dimmer, or strong contrast caused by backlighting\n"
            황혼(해 질 녘/동틀 녘) 또는 가로등이 촘촘하여 조명이 밝은 야간(식별은 가능하나 다소 어두움), 터널, 강한 대비
        "   3 = night with partial, intermittent lighting, dim street lights; large dark regions, reliance on headlights\n"
            부분적 또는 간헐적 조명이 있는 야간; 넓고 어두운 구역, 전조등 의존
        "   4 = near-total darkness (unlit road) where much of the scene is not resolvable"),
            장면의 상당 부분을 식별할 수 없는, 거의 완전한 어둠(조명이 없는 도로)
    "precipitation": (
        "   0 = none / clear\n"
        "   1 = drizzle or very light rain or snowfall; occasional drops on lens\n"
            이슬비 또는 아주 약한 비; 간헐적으로 렌즈에 떨어지는 빗방울
        "   2 = moderate rain or snowfall, raindrops or snow partially obscuring the camera\n"
            꾸준히 내리는 적당한 비 또는 가벼운 눈, 빗방울이 카메라를 부분적으로 가림
        "   3 = heavy rain or snowfall, raindrops or snow largely obscuring the camera\n"
            시야를 눈에 띄게 제한하는 폭우 또는 강설, 빗방울이 카메라를 대부분 가림
        "   4 = visibility is severely restricted because the camera is completely covered by raindrops or snow."),
            폭우 / 폭설 / 눈보라; 시야가 극도로 제한됨
    "road_surface": (
        "   0 = dry\n"
        "   1 = damp, no standing water, snow only off the roadway, unpaved or dusty road\n"
            축축함, 고인 물 없음, 도로 밖에만 눈 차도는 깨끗
        "   2 = clearly wet and reflective; light spray; or thin snow cover with wheel tracks worn through\n"
            확연히 젖어 있고 빛을 반사함; 가벼운 분무, 도로에 눈이 얇게 깔려 있으나 차선 표시는 여전히 식별 가능함
        "   3 = standing water, puddles, slush, or snow covers most of the roadway or the lane markings are faint\n"
            고인 물, 물웅덩이, 질척이는 눈(슬러시) 또는 도로 대부분이 눈으로 덮여 있거나 차선 희미함
        "   4 = deep snow-covered, icy, or flooded; lane markings obscured"),
            눈이나 얼음으로 덮여 있거나 침수됨, 차선 표시가 가려짐
    "atmospheric_obscurants": (
        "   0 = clear, long sight line\n"
        "   1 = slight haze\n"
            약간의 안개
        "   2 = moderate fog/haze/spray; distant objects blurred\n"
            보통 정도의 안개/연무/물보라; 먼 곳의 물체가 흐릿하게 보임
        "   3 = dense fog/spray; only the near field visible\n"
            짙은 안개/물보라; 근거리만 보임
        "   4 = very dense; minimal visibility beyond the immediate foreground"),
            매우 빽빽함; 바로 앞쪽을 제외하고는 시야가 거의 확보되지 않음
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

    전체 난이도(DEFAULT_PROMPT)를 요인 4개와 함께 묻되, 요인을 먼저 매기고
    전체를 마지막에 두었다. 단독 프롬프트 5개를 그냥 이어붙이면 모델이
    전체 점수를 요인과 무관하게 찍어 서로 모순되는 답이 나온다 - 예를 들어
    네 요인이 모두 0인데 전체가 3 인 식이다. 요인을 먼저 세우고 그것을
    근거로 삼게 하면 그 모순이 줄어든다.

    가중합을 쓰지 않는 이유: 어떤 가중치가 맞는지 아직 근거가 없다. 넷 중
    가장 나쁜 것이 난이도를 지배하는 장면(짙은 안개)도 있고, 여럿이 겹쳐야
    어려워지는 장면(야간 + 젖은 노면)도 있어 한 식으로 눌러 담기 어렵다.
    지금은 모델이 판단하게 두고, 분포를 aggregate_clip.log 로 보면서
    필요하면 나중에 식을 넣는다.
    """
    lines = []
    for key, name in DIFFICULTY_FACTORS:
        lines.append(f"   {name} ({key}) - judge {DIFFICULTY_ONLY[key]}:\n"
                     f"{DIFFICULTY_SCALES[key]}")
    factors = "\n".join(lines)
    return f"""   Rate how hard this scene is to drive, on an integer scale from
   {DIFFICULTY_MIN} to {DIFFICULTY_MAX}. First rate each factor on its own, judging ONLY what that
   factor names and ignoring the others:
{factors}
   Then give the overall driving difficulty, taking the factors above together
   with road clutter and traffic density:
{DIFFICULTY_SCALES['driving_difficulty']}
   The overall rating must be consistent with the factors - if every factor is
   {DIFFICULTY_MIN} the scene is not hard unless traffic or clutter makes it so, and a factor
   rated {DIFFICULTY_MAX} should not leave the overall rating at {DIFFICULTY_MIN}.
   Give each rating one short sentence saying what you saw that justifies it.
   These ratings describe conditions, not edge-cases: a clear empty road at
   noon is {DIFFICULTY_MIN} everywhere, and rating it so is the correct answer.
"""

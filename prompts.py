"""날씨/주행 조건 4축(--weather) 프롬프트 조각.

처음에는 축마다 "이미지 한 장 + 정수 하나 + 문장 하나"를 따로 묻는 단독
프롬프트였다. 클립 모드에 그대로 붙이면 한 클립에 모델을 5번 더 부르게 되어
비용이 6배가 되므로, 같은 기준을 nuReasoning JSON 한 번에 얹는 형태로 옮겼다.
"""
# 축 순서는 시각화/집계 표의 행 순서이기도 하다.
WEATHER_MIN, WEATHER_MAX = 0, 4

# (JSON 키, 화면에 쓸 이름). 전체 난이도(driving_difficulty)는 더 이상 모델에게
# 묻지 않는다 - 요인 4개와 따로 매기게 하면 서로 모순되는 답이 나오고(네 요인이
# 0인데 전체가 3인 식), 어떤 가중치가 옳은지도 근거가 없었다. 필요하면 요인
# 값으로 나중에 계산한다.
WEATHER_FACTORS = [
    ("illumination", "Illumination"),
    ("precipitation", "Precipitation"),
    ("road_surface", "Road_surface"),
    ("atmospheric_obscurants", "Atmospheric_obscurants"),
]
# 집계/시각화가 도는 축 = 요인 4개. 전체 난이도는 빠졌다(위 주석).
WEATHER_AXES = list(WEATHER_FACTORS)


# 각 축의 0~4 눈금. 한쪽만 고치는 일이 없도록 여기 한 곳에만 적는다.
WEATHER_SCALES = {
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
        "   0 = dry and clean\n"
        "   1 = damp, no standing water, or snow only off the roadway, dusty but flat road\n"
        "   2 = surface of soil and gravel; or clearly wet and reflective; light spray; or thin snow on the roadway but lane markings still discernible\n"
        "   3 = standing water, puddles, slush, or snow covers most of the roadway or the lane markings are faint, or very rough due to mud or coarse gravel\n"
        "   4 = deep snow-covered, icy, or flooded; lane markings entirely obscured"),
    "atmospheric_obscurants": (
        "   0 = clear, long sight line\n"
        "   1 = slight fog/haze\n"
        "   2 = moderate fog/haze/spray; distant objects blurred\n"
        "   3 = dense fog/spray; only the near field visible\n"
        "   4 = very dense; minimal visibility beyond the immediate foreground"),
}

# WEATHER_SCALES 의 한국어판. [NAS 라벨링] 탭에서 사람이 읽는 번역이고 모델
# 프롬프트에는 들어가지 않는다. 위 눈금을 고치면 여기도 같이 고친다.
WEATHER_SCALES_KOR = {
    "illumination": (
        "   0 = 완전한 주간, 고르게 밝고 장면 전체가 선명하게 보임\n"
        "   1 = 흐리거나 밋밋한 주간 빛; 대비는 낮지만 시야는 온전함, 또는 고가도로로 인해 잠깐 그늘이 짐\n"
        "   2 = 황혼(해질녘/새벽), 또는 가로등이나 도시 불빛이 촘촘해 밝은 야간, 또는 터널, 또는 역광으로 인한 강한 명암 대비\n"
        "   3 = 가로등이 일부만 있거나 어두운 야간, 또는 가로등으로 도로는 보이지만 주변은 어두움\n"
        "   4 = 자차 전조등만 도로를 비춤, 또는 거의 완전한 어둠(조명 없는 도로)이라 장면 대부분을 식별할 수 없음"),
    "precipitation": (
        "   0 = 없음 / 맑음\n"
        "   1 = 이슬비 또는 아주 약한 비나 눈; 렌즈에 가끔 물방울이 맺힘\n"
        "   2 = 보통 세기의 비나 눈; 빗방울이나 눈이 카메라를 일부 가림\n"
        "   3 = 강한 비나 눈; 빗방울이나 눈이 카메라를 대부분 가림\n"
        "   4 = 빗방울이나 눈이 카메라를 완전히 덮어 시야가 심하게 제한됨"),
    "road_surface": (
        "   0 = 건조하고 깨끗함\n"
        "   1 = 축축하지만 고인 물은 없음, 또는 눈이 차도 밖에만 있음, 또는 먼지가 있지만 평탄한 도로\n"
        "   2 = 흙과 자갈 노면; 또는 확연히 젖어 빛이 반사됨; 물보라가 약간 튐; 또는 차도에 눈이 얇게 쌓였지만 차선은 구분됨\n"
        "   3 = 고인 물, 웅덩이, 진창눈, 또는 눈이 차도 대부분을 덮거나 차선이 희미함, 또는 진흙이나 굵은 자갈로 매우 거침\n"
        "   4 = 눈이 깊게 쌓였거나, 빙판이거나, 침수됨; 차선이 완전히 가려짐"),
    "atmospheric_obscurants": (
        "   0 = 맑고 먼 곳까지 보임\n"
        "   1 = 옅은 안개/연무\n"
        "   2 = 보통 수준의 안개/연무/물보라; 먼 물체가 흐릿함\n"
        "   3 = 짙은 안개/물보라; 가까운 곳만 보임\n"
        "   4 = 매우 짙음; 바로 앞 너머로는 거의 보이지 않음"),
}

# 각 요인에서 "무엇만 보라"는 한정. 축끼리 서로 번지는 것을 막는 유일한 장치다.
WEATHER_ONLY = {
    "illumination": "ambient light / time of day only - not weather or road state",
    "precipitation": "falling precipitation only - not ambient light, road surface, or fog",
    "road_surface": "the road surface state only - not the falling weather or sky",
    "atmospheric_obscurants": "airborne fog/haze/spray only - not darkness or rain intensity",
}


def weather_block() -> str:
    """nuReasoning 프롬프트에 얹을 난이도 단계 본문.

    요인 4개만 묻는다. 전체 난이도는 빼기로 했다 - 요인과
    따로 매기게 하면 모델이 둘을 무관하게 찍어 모순된 답이 나왔고(네 요인이
    모두 0인데 전체가 3인 식), 요인을 가중합할 옳은 식도 근거가 없었다.
    넷 중 가장 나쁜 것이 지배하는 장면(짙은 안개)과 여럿이 겹쳐야 어려워지는
    장면(야간 + 젖은 노면)이 섞여 있어 한 식으로 누르기 어렵다. 전체 난이도가
    필요해지면 요인 값에서 계산한다.
    """
    lines = []
    for key, name in WEATHER_FACTORS:
        lines.append(f"   {name} ({key}) - judge {WEATHER_ONLY[key]}:\n"
                     f"{WEATHER_SCALES[key]}")
    factors = "\n".join(lines)
    return f"""   Rate the driving conditions of this scene on an integer scale from
   {WEATHER_MIN} to {WEATHER_MAX}. Rate each factor on its own, judging ONLY what that
   factor names and ignoring the others:
{factors}
   Give each rating one short sentence saying what you saw that justifies it.
   Name only the condition you actually saw - do not copy the wording of a
   level when it does not describe this scene.
   These ratings describe conditions, not edge-cases: a clear empty road at
   noon is {WEATHER_MIN} everywhere, and rating it so is the correct answer.
"""

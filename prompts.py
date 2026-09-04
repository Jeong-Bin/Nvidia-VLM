

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
    "  2 = twilight (dusk/dawn) or well-lit night with dense street lighting; "
    "readable but dimmer\n"
    "  3 = night with partial or intermittent lighting; large dark regions, "
    "reliance on headlights\n"
    "  4 = near-total darkness (unlit road) where much of the scene is not "
    "resolvable\n"
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
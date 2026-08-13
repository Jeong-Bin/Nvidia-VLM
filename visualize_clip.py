#!/usr/bin/env python3
"""클립 단위 판정 결과를 mp4 영상 + json 으로 저장한다.

visualize.py(카드 한 장)와 나란히 두는 별도 경로다. 카드는 "한 순간"을
전제로 만들어져서 클립 모드에 맞지 않는다:
  - 클립 모드는 20초를 보고 판단하므로 정지 프레임 한 장으로는 근거를
    확인할 수 없다 ("안개 때문에 급제동" 같은 판단은 시간축이 있어야 보인다).
  - nuReasoning 6단계 답변(관찰/자차행동/이상요소/안전성/희귀성)은 카드
    하단에 다 들어가지 않는다.

레이아웃:
  상단 - 원본 mp4 를 그대로 재생 (1fps 로 뽑은 추론 입력이 아니라 30fps 원본)
  하단 - 1~5단계 추론 내용을 텍스트로 고정 표시

같은 폴더에 result.json 으로 추론 원문도 함께 남긴다.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 하단 패널에 표시할 단계. (result 의 키, 화면에 쓸 제목) 순서가 곧 표시 순서다.
PANEL_STEPS = [
    ("observation", "1. Scene Description"),
    ("ego_behavior", "2. Ego Behavior Summary"),
    ("unusual_elements", "3. Unusual Element & Ego Influence"),
    ("safety_assessment", "4. Safety Criticality"),
    ("rarity_assessment", "5. Rarity"),
]

# 4/5단계는 등급(1/2/3)과 이유가 따로 오므로, 패널에는
# "Moderate (2) - 이유" 형태로 합쳐 보여준다. 라벨만 쓰면 폴더명이 되는
# 점수(합계)와 눈으로 대조할 수 없어서 정수도 함께 적는다.
# result 에서 (라벨 키, 정수 키) 를 어디서 읽을지 매핑.
STEP_TIER_KEY = {
    "safety_assessment": ("safety_label", "safety_tier"),
    "rarity_assessment": ("rarity_label", "rarity_tier"),
}

FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

BG = (255, 255, 255)
TEXT_DARK = (26, 26, 26)
TEXT_MID = (51, 51, 51)
TEXT_MUTED = (120, 120, 120)
ACCENT = (179, 84, 30)      # 제목/카테고리 강조 (visualize.py 와 같은 주황)
RULE = (221, 221, 221)

# 인코더 선택.
#
# OpenCV 빌드에는 libx264 가 없어 cv2.VideoWriter 로는 mp4v(MPEG-4 Part 2)밖에
# 못 쓴다. mp4v 는 VLC 같은 데스크톱 플레이어에서는 재생되지만 브라우저 계열
# (VS Code 내장 미리보기 포함)에서는 재생되지 않는다 - 실제로 VS Code 에서
# 재생 실패를 겪었다.
#
# PyAV(18.0)에 libx264 가 들어 있으므로 그쪽을 우선 쓴다. 추가 설치가 필요
# 없고, H.264 + yuv420p 조합은 VS Code/Chrome/Safari 어디서나 재생된다.
# 압축률도 훨씬 좋다(실측 39MB -> 4MB 수준).
#
# PyAV 가 없거나 libx264 가 빠진 환경을 위해 cv2/mp4v 경로도 남겨둔다.
FOURCC = "mp4v"          # cv2 폴백에서만 쓴다

# yuv420p 로 고정한다 - 브라우저는 yuv444p/yuvj420p 를 못 읽는 경우가 많다.
# profile=main 은 High 보다 호환 범위가 넓고 화질 차이는 이 용도에서 무시할
# 수준이다. faststart 로 moov atom 을 앞으로 옮겨야 스트리밍 재생이 된다.
X264_OPTS = {"crf": "23", "preset": "veryfast", "profile": "main"}


def _have_pyav() -> bool:
    try:
        import av
        av.codec.Codec("libx264", "w")
        return True
    except Exception:
        return False


HAVE_PYAV = _have_pyav()

# 출력 영상 폭. 원본은 1920 이지만 mp4v 는 압축률이 낮아 원본 그대로면
# 클립당 약 44MB 가 된다(실측). 1280 이면 약 20MB 로 줄면서 검수에 필요한
# 디테일은 남는다. None 을 주면 원본 해상도를 그대로 쓴다.
VIZ_WIDTH = 1280


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _wrap(text, width_chars):
    """빈 줄을 유지하면서 줄바꿈. 값이 비면 em dash 한 줄."""
    text = (text or "").strip()
    if not text:
        return ["—"]
    lines = []
    for para in text.splitlines():
        lines.extend(textwrap.wrap(para, width=width_chars) or [""])
    return lines


def build_text_panel(result: dict, width: int, scale: float = 1.0) -> np.ndarray:
    """1~5단계 추론 내용을 담은 패널 이미지(RGB ndarray)를 만든다.

    높이는 내용에 맞춰 늘어난다 - 항목마다 길이가 제각각이라 고정하면
    잘리거나 빈 공간이 남는다. 두 번 그린다: 한 번은 높이를 재려고,
    한 번은 실제로.
    """
    pad = int(28 * scale)
    fs_head = max(11, int(21 * scale))
    fs_title = max(9, int(15 * scale))
    fs_body = max(8, int(14 * scale))
    line_h = int(fs_body * 1.55)
    title_h = int(fs_title * 1.9)
    gap = int(14 * scale)

    f_head = _font(FONT_BOLD, fs_head)
    f_title = _font(FONT_BOLD, fs_title)
    f_body = _font(FONT_REG, fs_body)

    # 폭에서 대략 몇 글자가 들어가는지 - DejaVu Sans 는 평균 자폭이
    # 폰트 크기의 약 0.55 배다.
    width_chars = max(40, int((width - 2 * pad) / (fs_body * 0.55)))

    cats = result.get("categories") or []
    cat_line = ", ".join(cats) if cats else "— none —"

    blocks = []
    for key, title in PANEL_STEPS:
        body = result.get(key, "")
        label_key, tier_key = STEP_TIER_KEY.get(key, (None, None))
        if label_key:
            label = result.get(label_key, "")
            tier = result.get(tier_key)
            if label and label != "Unknown":
                head = f"{label} ({tier})" if tier is not None else label
                body = f"{head} - {body}" if body else head
        blocks.append((title, _wrap(body, width_chars)))

    # --- 높이 계산 ---
    h = pad
    h += int(fs_head * 1.6) + int(6 * scale)        # uuid 헤드라인
    h += int(fs_body * 1.6) + gap                   # 카테고리 줄
    for _, lines in blocks:
        h += title_h + len(lines) * line_h + gap
    h += pad

    img = Image.new("RGB", (width, h), BG)
    d = ImageDraw.Draw(img)

    y = pad
    head = result.get("_headline", "")
    d.text((pad, y), head, font=f_head, fill=TEXT_DARK)
    y += int(fs_head * 1.6) + int(6 * scale)

    d.text((pad, y), f"Categories ({len(cats)}): ", font=f_title, fill=TEXT_DARK)
    cw = d.textlength(f"Categories ({len(cats)}): ", font=f_title)
    d.text((pad + cw, y), cat_line, font=f_title,
           fill=ACCENT if cats else TEXT_MUTED)
    y += int(fs_body * 1.6) + gap

    for title, lines in blocks:
        d.line([(pad, y - int(4 * scale)), (width - pad, y - int(4 * scale))],
               fill=RULE, width=1)
        d.text((pad, y), title, font=f_title, fill=ACCENT)
        y += title_h
        for ln in lines:
            d.text((pad, y), ln, font=f_body, fill=TEXT_MID)
            y += line_h
        y += gap

    return np.asarray(img)


def render_clip_video(src_mp4, result: dict, out_path,
                      max_width: int | None = None,
                      fourcc: str = FOURCC,
                      max_frames: int | None = None) -> Path | None:
    """원본 mp4 위에 추론 텍스트 패널을 붙여 새 mp4 로 저장.

    src_mp4    : 원본 클립 경로 (1fps 로 뽑은 추론 입력이 아니라 원본 그대로)
    result     : parse_nureasoning_output 결과 (+ "_headline" 을 넣어두면 제목에 쓴다)
    max_width  : 지정하면 그 폭으로 줄인다. None 이면 원본 해상도 유지.
    max_frames : 디버그용 - 앞에서 N 프레임만 쓴다.

    반환: 저장 경로. 원본을 못 열면 None.
    """
    src_mp4 = str(src_mp4)
    cap = cv2.VideoCapture(src_mp4)
    if not cap.isOpened():
        return None

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if src_w <= 0 or src_h <= 0:
        cap.release()
        return None

    if max_width and src_w > max_width:
        out_w = int(max_width)
        out_h = int(round(src_h * out_w / src_w))
    else:
        out_w, out_h = src_w, src_h
    # H.264 가 아니어도 짝수 폭/높이를 요구하는 플레이어가 있어 맞춰둔다
    out_w -= out_w % 2
    out_h -= out_h % 2

    panel = build_text_panel(result, out_w, scale=out_w / 1280.0)
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    total_h = out_h + panel.shape[0]
    total_h -= total_h % 2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((total_h, out_w, 3), 255, dtype=np.uint8)
    # 패널은 클립 내내 같으므로 한 번만 붙이고, 매 프레임 영상 영역만 갈아끼운다
    ph = min(panel_bgr.shape[0], total_h - out_h)
    canvas[out_h:out_h + ph] = panel_bgr[:ph]

    def frames_iter():
        """원본에서 프레임을 읽어 패널 위에 얹은 canvas(BGR)를 흘려준다."""
        n = 0
        limit = max_frames or n_total or 10 ** 9
        while n < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                frame = cv2.resize(frame, (out_w, out_h),
                                   interpolation=cv2.INTER_AREA)
            canvas[:out_h] = frame
            yield canvas
            n += 1

    try:
        if HAVE_PYAV:
            n = _write_h264(out_path, frames_iter(), out_w, total_h, fps)
        else:
            n = _write_cv2(out_path, frames_iter(), out_w, total_h, fps, fourcc)
    finally:
        cap.release()
    return out_path if n else None


def _write_h264(out_path: Path, frames, width: int, height: int, fps: float):
    """PyAV/libx264 로 쓴다. VS Code·브라우저에서 바로 재생되는 조합."""
    import av

    container = av.open(str(out_path), "w",
                        options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=int(round(fps)) or 30)
    stream.width, stream.height = width, height
    stream.pix_fmt = "yuv420p"
    stream.options = dict(X264_OPTS)

    n = 0
    for bgr in frames:
        frame = av.VideoFrame.from_ndarray(bgr[:, :, ::-1].copy(),
                                           format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        n += 1
    for packet in stream.encode():      # 인코더에 남은 프레임 flush
        container.mux(packet)
    container.close()
    return n


def _write_cv2(out_path: Path, frames, width: int, height: int, fps: float,
               fourcc: str):
    """PyAV 가 없는 환경용 폴백. mp4v 라 브라우저 재생은 안 될 수 있다."""
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc),
                             fps, (width, height))
    if not writer.isOpened():
        return 0
    n = 0
    for bgr in frames:
        writer.write(bgr)
        n += 1
    writer.release()
    return n


def save_clip_json(result: dict, out_path, extra: dict | None = None) -> Path:
    """추론 결과 원문을 json 으로 저장 (영상과 같은 폴더).

    verdict(edge-case 요소가 하나라도 있었는가)도 함께 남긴다 - 영상은
    읽기용이고 json 은 재집계용이라 목적이 다르다.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "uuid": result.get("_uuid", ""),
        "verdict": result.get("verdict"),
        "categories": result.get("categories", []),
        "reasoning": {key: result.get(key, "") for key, _ in PANEL_STEPS},
        "safety_tier": result.get("safety_tier"),
        "safety_label": result.get("safety_label", "Unknown"),
        "rarity_tier": result.get("rarity_tier"),
        "rarity_label": result.get("rarity_label", "Unknown"),
        "tier_score": result.get("tier_score"),
        "parse_ok": bool(result.get("parse_ok", False)),
    }
    if extra:
        payload.update(extra)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return out_path


def render_clip_result(uuid: str, src_mp4, result: dict, out_dir,
                       max_width: int | None = VIZ_WIDTH,
                       extra: dict | None = None,
                       max_frames: int | None = None):
    """클립 하나의 영상 + json 을 같은 폴더에 저장하고 (video, json) 을 반환."""
    out_dir = Path(out_dir)
    r = dict(result)
    r["_uuid"] = uuid
    r["_headline"] = uuid
    video = render_clip_video(src_mp4, r, out_dir / "clip.mp4",
                              max_width=max_width, max_frames=max_frames)
    js = save_clip_json(r, out_dir / "result.json", extra=extra)
    return video, js

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
import os
import tempfile
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from constrained_tier import GROUP_RUBRICS, group_values
from prompts import DIFFICULTY_AXES, DIFFICULTY_MAX

# 하단 패널에 표시할 단계. (result 의 키들, 화면에 쓸 제목) 순서가 곧 표시
# 순서다.
#
# 1단계와 2단계를 한 칸에 합친 이유: 둘 다 "무엇이 보였나"를 서술하는 문단
# 이라 따로 두면 패널 높이만 늘고 읽는 순서는 그대로다. 검수자가 실제로
# 대조하는 것은 아래 카테고리별 점수이므로, 서술은 짧게 묶어 위에 둔다.
PANEL_STEPS = [
    (("observation", "ego_behavior"), "1. Scene Description & Ego Behavior"),
    (("unusual_elements",), "2. Unusual Element"),
]


FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

BG = (255, 255, 255)
TEXT_DARK = (26, 26, 26)
TEXT_MID = (51, 51, 51)
TEXT_MUTED = (120, 120, 120)
ACCENT = (179, 84, 30)      # 제목/카테고리 강조 (visualize.py 와 같은 주황)
RULE = (221, 221, 221)

# --time-spans 구간 띠 색(RGB). 장르가 아니라 구간 위험도(0~4)로 칠한다.
# 점수를 못 읽은 구간(None)은 회색 빗금 대신 중간 회색으로 둔다.
SCORE_COLORS = {0: (230, 230, 230), 1: (158, 255, 15), 2: (255, 245, 0),
                3: (255, 143, 31), 4: (255, 0, 0)}
SCORE_UNKNOWN = (180, 180, 180)
# 구간이 없는 시간(그 요소가 없었음)의 바탕. 0 점 색과 구분되게 더 밝게 둔다.
TRACK_BG = (248, 248, 248)
TRACK_EDGE = (215, 215, 215)
PLAYHEAD = (40, 40, 40)

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


def _num(v):
    """점수 표기. 값이 없으면(라벨 없음, 그 축을 안 잰 실행) "-"."""
    return "-" if v is None else str(v)


def difficulty_lines(result: dict, gt: dict | None = None) -> list[str]:
    """날씨 4축을 패널에 넣을 문자열 줄로 만든다. 점수마다 GT 와 Pred 를
    이름을 붙여 나란히 적는다.

        • Illumination (GT:1, Pred:2): <근거 문장>
        • Precipitation (GT:0, Pred:0): <근거 문장>
        ...

    GT 가 없으면(라벨 없는 실행) GT 자리는 "-" 다. 축이 하나도 없으면(날씨를
    끈 실행) 빈 리스트를 돌려주고, 호출부는 블록 자체를 만들지 않는다.
    """
    if all(result.get(k) is None for k, _ in DIFFICULTY_AXES):
        return []
    gw = (gt or {}).get("weather") or {}
    lines = []
    for key, name in DIFFICULTY_AXES:
        why = (result.get(f"{key}_reason") or "").strip()
        lines.append(f"• {name} (GT:{_num(gw.get(key))}, Pred:{_num(result.get(key))})"
                     + (f": {why}" if why else ""))
    return lines

# 카테고리 -> 묶음 이름.
#
# scene_category.json 을 읽어 채우되, 못 읽으면 빈 채로 둔다. 시각화가 씬
# 파일을 못 찾았다고 해서 영상 생성이 실패하면 안 된다.
SCENARIO_OF = {}


def load_scenario_map(scene_json) -> None:
    """scene_category.json 에서 카테고리->묶음 표를 읽어 둔다."""
    try:
        data = json.loads(Path(scene_json).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    SCENARIO_OF.clear()
    for sc in data.get("special", {}).get("scenarios", []):
        for c in sc.get("categories", []):
            SCENARIO_OF[c["name"]] = sc["name"]


def _sorted_names(names) -> list[str]:
    """카테고리 이름을 알파벳 오름차순(대소문자 무시)으로."""
    return sorted(names or [], key=str.lower)


def category_score_lines(result: dict, gt: dict | None) -> list[str]:
    """카테고리와 상위 카테고리 점수를 패널에 넣을 문자열 줄로 만든다.
    점수마다 GT 와 Pred 를 이름을 붙여 나란히 적는다.

        • GT   : Cyclist on Road, Pedestrian on Road
        • Pred : Cyclist on Road, Pedestrian on Road
        • Dynamic object (GT:2, Pred:3)
        • Driving environment (GT:0, Pred:0)
        Dynamic object: <근거> | Driving environment: <근거>

    상위 카테고리 줄은 GT 나 Pred 어느 한쪽이라도 값이 있는 것만 낸다 -
    상위 카테고리 하나만 돌린 실행(--group)에 빈 줄이 붙지 않게. GT 가
    없으면(라벨 없는 실행) GT 줄을 빼고 점수의 GT 자리는 "-" 다.
    카테고리는 GT/Pred 모두 알파벳 오름차순(대소문자 무시)으로 적는다 -
    모델이 낸 순서대로 두면 같은 집합도 순서가 달라 눈으로 맞대기 어렵다.
    """
    pred_cats = result.get("categories") or []
    pred_scores = result.get("group_scores") or {}
    gt_scores = (gt or {}).get("scores") or {}
    if gt is None and not pred_cats and not pred_scores:
        return []
    lines = []
    if gt is not None:
        lines.append("• GT   : " + (", ".join(_sorted_names(gt.get("categories")))
                                   or "None"))
    lines.append("• Pred : " + (", ".join(_sorted_names(pred_cats)) or "None"))
    for g in GROUP_RUBRICS:
        gv, pv = gt_scores.get(g), pred_scores.get(g)
        if gv is None and pv is None:
            continue
        lines.append(f"• {g} (GT:{_num(gv)}, Pred:{_num(pv)})")
    why = (result.get("score_reason") or "").strip()
    if why:
        lines.append(why)
    return lines


def build_timeline_strip(result: dict, duration: float, width: int,
                         scale: float = 1.0):
    """--time-spans 결과를 재생 바 모양의 띠로 그린다. 없으면 None.

    장르마다 한 줄(lane)이고, 줄 안에 모델이 답한 구간을 위험도(0~4) 색으로
    칠한다(SCORE_COLORS). 구간 위에는 "시작-끝s" 만 적는다 - 점수는 색과
    범례로 읽는다. 구간은 이미 정수 초로 정리돼 있다
    (edge_case_mining._snap_spans). 아래에 초 눈금과
    점수 범례를 둔다. 재생 위치(세로선)는 프레임마다 바뀌므로 여기서 그리지
    않고, 그릴 자리(geom)를 함께 돌려준다.

    구간을 하나도 못 받은 장르도 빈 줄로 남긴다 - 줄이 아예 없으면
    "모델이 구간을 안 줬다"와 "그 장르가 없었다"가 구분되지 않는다.
    반환: (RGB ndarray, {"x0", "x1", "y0", "y1", "duration"}) 또는 None.
    """
    spans = result.get("time_spans") or {}
    cats = _sorted_names(result.get("categories"))
    if not cats or not (spans or result.get("_time_spans_on")):
        return None
    duration = max(float(duration or 0), max(
        (sp[1] for v in spans.values() for sp in v), default=0.0), 1e-3)

    pad = int(28 * scale)
    fs = max(8, int(13 * scale))
    f_lab = _font(FONT_BOLD, fs)
    fs_txt = max(7, int(11 * scale))
    f_txt = _font(FONT_REG, fs_txt)
    lane_h = max(12, int(22 * scale))
    lane_gap = max(3, int(6 * scale))
    top = int(12 * scale)
    axis_h = int(20 * scale)
    legend_h = int(20 * scale)
    label_w = int(width * 0.24)
    x0, x1 = pad + label_w, width - pad
    h = (top + len(cats) * (lane_h + lane_gap) - lane_gap + axis_h + legend_h
         + int(8 * scale))

    img = Image.new("RGB", (width, h), BG)
    d = ImageDraw.Draw(img)
    to_x = lambda t: x0 + (min(max(t, 0.0), duration) / duration) * (x1 - x0)
    ty = lambda y: y + (lane_h - fs_txt) // 2 - 1

    y = top
    for c in cats:
        d.text((pad, y + (lane_h - fs) // 2), c, font=f_lab, fill=TEXT_DARK)
        d.rectangle([x0, y, x1, y + lane_h], fill=TRACK_BG, outline=TRACK_EDGE)
        cs = spans.get(c) or []
        if not cs:
            d.text((x0 + int(6 * scale), ty(y)), "no time span given",
                   font=f_txt, fill=TEXT_MUTED)
        for sp in cs:
            s0, s1 = sp[0], sp[1]
            sc = sp[2] if len(sp) > 2 else None
            a, b = to_x(s0), to_x(s1)
            b = max(b, a + 2)
            d.rectangle([a, y, b, y + lane_h],
                        fill=SCORE_COLORS.get(sc, SCORE_UNKNOWN))
            # 점수는 색과 아래 범례로 읽으므로 글자는 시간만 적는다.
            # 다 안 들어가는 좁은 구간은 비운다.
            fg = (255, 255, 255) if sc == 4 else TEXT_DARK
            tag = f"{s0:.0f}-{s1:.0f}s"
            if d.textlength(tag, font=f_txt) + 6 <= b - a:
                d.text((a + 3, ty(y)), tag, font=f_txt, fill=fg)
            # 붙어 있는 구간끼리 경계가 보이게 얇은 선을 긋는다.
            d.line([(a, y), (a, y + lane_h)], fill=BG, width=1)
        y += lane_h + lane_gap
    y_end = y - lane_gap

    # 초 눈금. 20 초 클립이면 5 초마다.
    step = 5 if duration > 10 else 2
    t = 0
    while t <= duration + 1e-6:
        x = to_x(t)
        d.line([(x, y_end), (x, y_end + int(5 * scale))], fill=TEXT_MUTED)
        lab = f"{t:g}s"
        d.text((x - d.textlength(lab, font=f_txt) / 2, y_end + int(6 * scale)),
               lab, font=f_txt, fill=TEXT_MUTED)
        t += step

    # 점수 범례
    ly = y_end + axis_h
    lx = pad
    d.text((lx, ly), "Risk score", font=f_txt, fill=TEXT_MUTED)
    lx += d.textlength("Risk score", font=f_txt) + int(10 * scale)
    box = max(8, int(12 * scale))
    for n in range(5):
        d.rectangle([lx, ly + 1, lx + box, ly + 1 + box],
                    fill=SCORE_COLORS[n], outline=TRACK_EDGE)
        d.text((lx + box + int(4 * scale), ly), str(n), font=f_txt,
               fill=TEXT_MID)
        lx += box + int(22 * scale)
    return np.asarray(img), {"x0": x0, "x1": x1, "y0": top, "y1": y_end,
                             "duration": duration}


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
    # 정답 라벨(gt)이 함께 넘어오면 3번 칸이 GT 와 Pred 를 한 줄씩 짝지어
    # 보여준다(category_score_lines).
    gt = result.get("_gt")

    blocks = []
    for keys, title in PANEL_STEPS:
        # 여러 키를 묶은 칸은 한 문단으로 이어 붙인다(1단계 + 2단계).
        body = " ".join(str(result.get(k) or "").strip() for k in keys).strip()
        blocks.append((title, _wrap(body, width_chars), None))

    # 카테고리별 점수. GT 와 Pred 를 한 줄씩 짝지어 놓는다 - 검수자가 실제로
    # 대조하는 것이 이 값이고, 분모를 함께 적어야 2 가 중간인지 상한인지
    # 읽힌다(rubric 마다 폭이 달라질 수 있다).
    score_lines = category_score_lines(result, gt)
    if score_lines:
        wrapped = []
        for ln in score_lines:
            wrapped.extend(textwrap.wrap(ln, width=width_chars,
                                         subsequent_indent="    ") or [ln])
        blocks.append((f"{len(blocks) + 1}. Categories & Difficulty score",
                       wrapped, None))

    # 난이도는 등급 단계 다음에 자기 블록으로 붙는다. PANEL_STEPS 에 넣지
    # 않는 이유는 형식이 다르기 때문이다 - 저쪽은 "제목 + 문단" 한 덩어리고,
    # 이건 다섯 줄이 각자 점수를 달고 있어 줄 단위로 유지해야 한다.
    diff = difficulty_lines(result, gt)
    if diff:
        wrapped = []
        for ln in diff:
            # 이어지는 줄은 들여쓴다 - 안 그러면 다음 축의 머리줄과 구분이
            # 안 되어 어느 점수의 근거인지 읽을 수 없다.
            parts = textwrap.wrap(ln, width=width_chars,
                                  subsequent_indent="    ") or [ln]
            wrapped.extend(parts)
        # 번호는 앞서 실제로 담긴 블록 수를 이어받는다 - 등급을 끈
        # 실행에서는 4/5단계가 없으므로 "6." 이라고 쓰면 없는 단계를 센다.
        blocks.append((f"{len(blocks) + 1}. Weather score", wrapped, None))

    # --- 높이 계산 ---
    h = pad
    h += int(fs_head * 1.6) + int(6 * scale)        # uuid 헤드라인
    h += int(fs_body * 1.6) + gap                   # 카테고리 줄
    for _, lines, gt_line in blocks:
        h += title_h + len(lines) * line_h + gap
        if gt_line:
            h += line_h
    h += pad

    img = Image.new("RGB", (width, h), BG)
    d = ImageDraw.Draw(img)

    def draw_segments(x, y, segments, font):
        """(글자, 색) 조각들을 한 줄로 이어 그린다. 끝 x 를 돌려준다."""
        for text, color in segments:
            d.text((x, y), text, font=font, fill=color)
            x += d.textlength(text, font=font)
        return x

    y = pad
    head = result.get("_headline", "")
    d.text((pad, y), head, font=f_head, fill=TEXT_DARK)
    y += int(fs_head * 1.6) + int(6 * scale)

    # 카테고리 이름은 3번 칸이 점수와 함께 찍으므로 여기서는 판정과 개수만
    # 적는다 - 같은 목록을 두 번 쓰면 패널만 길어지고 읽는 것은 아래 칸이다.
    verdict = result.get("verdict") or ("Special" if cats else "Normal")
    d.text((pad, y), f"{verdict}  ", font=f_title,
           fill=ACCENT if cats else TEXT_MUTED)
    d.text((pad + d.textlength(f"{verdict}  ", font=f_title), y),
           f"({len(cats)} categor{'y' if len(cats) == 1 else 'ies'})",
           font=f_title, fill=TEXT_MUTED)
    y += int(fs_body * 1.6) + gap

    for title, lines, gt_line in blocks:
        d.line([(pad, y - int(4 * scale)), (width - pad, y - int(4 * scale))],
               fill=RULE, width=1)
        d.text((pad, y), title, font=f_title, fill=ACCENT)
        y += title_h
        if gt_line:
            # "GT:" 는 검정, 등급 값은 강조색
            head_txt, _, val = gt_line.partition(" ")
            draw_segments(pad, y, [(head_txt + " ", TEXT_DARK),
                                   (val, ACCENT if val != "—" else TEXT_MUTED)],
                          f_body)
            y += line_h
        for ln in lines:
            if ln.startswith("Pred: "):
                draw_segments(pad, y, [("Pred: ", TEXT_DARK),
                                       (ln[6:], TEXT_MID)], f_body)
            else:
                d.text((pad, y), ln, font=f_body, fill=TEXT_MID)
            y += line_h
        y += gap

    return np.asarray(img)


def _extract_to_temp(view: str, uuid: str):
    """클립 소스에서 원본 mp4 바이트를 받아 임시 파일로 푼다. 실패하면 None.

    NAS 실행 전용 갈래다. edge_case_mining.clip_open 이 zip 안의 구간을
    파일처럼 열어주지만 cv2 는 그런 객체를 못 받으므로, 여기서 한 번
    실제 파일로 떨어뜨린다. 부르는 쪽이 반드시 _cleanup_temp 로 지운다.
    """
    try:
        from edge_case_mining import CLIP_SOURCE, clip_open
    except ImportError:
        return None
    if CLIP_SOURCE is None:          # 로컬 실행이면 애초에 이 갈래를 안 탄다
        return None
    try:
        fh = clip_open(view, uuid)
        data = fh.read() if hasattr(fh, "read") else Path(str(fh)).read_bytes()
        if hasattr(fh, "close"):
            fh.close()
        fd, name = tempfile.mkstemp(suffix=".mp4", prefix=f"viz_{uuid[:8]}_")
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        return Path(name)
    except Exception:
        # 시각화는 부가 산출물이다 - 여기서 실행을 세우지 않는다.
        return None


def _cleanup_temp(path):
    if path:
        try:
            Path(path).unlink()
        except OSError:
            pass


def render_clip_video(src_mp4, result: dict, out_path,
                      max_width: int | None = None,
                      fourcc: str = FOURCC,
                      max_frames: int | None = None,
                      traj: str | None = None, traj_uuid: str | None = None,
                      traj_view: str = "camera_front_wide_120fov",
                      ) -> Path | None:
    """원본 mp4 위에 추론 텍스트 패널을 붙여 새 mp4 로 저장.

    src_mp4    : 원본 클립 경로 (1fps 로 뽑은 추론 입력이 아니라 원본 그대로)
    result     : parse_nureasoning_output 결과 (+ "_headline" 을 넣어두면 제목에 쓴다)
    max_width  : 지정하면 그 폭으로 줄인다. None 이면 원본 해상도 유지.
    max_frames : 디버그용 - 앞에서 N 프레임만 쓴다.
    traj       : "center"|"width" 를 주면 원본 프레임마다 자차 미래 궤적을
                 그린다. 추론 입력(2fps 40장)과 달리 여기는 원본 605프레임을
                 모두 쓰므로, 프레임 번호를 그대로 궤적 계산에 넘긴다.

    반환: 저장 경로. 원본을 못 열면 None.
    """
    # cv2.VideoCapture 는 실제 파일 경로만 받는다. NAS 실행에서는 원본
    # mp4 가 청크 zip 안에 있어 그 경로가 존재하지 않으므로, 열리지 않으면
    # 소스에서 바이트를 받아 임시 파일로 풀어 놓고 그것을 연다.
    # (zip 안의 mp4 는 무압축이라 복사 비용이 곧 읽기 비용이다.)
    #
    # 이 갈래가 없으면 NAS 실행은 result.json 만 남고 clip.mp4 가 조용히
    # 빠진다 - 실측으로 그렇게 나왔다.
    src_mp4 = str(src_mp4)
    tmp_path = None
    cap = cv2.VideoCapture(src_mp4)
    if not cap.isOpened() and traj_uuid:
        tmp_path = _extract_to_temp(traj_view, traj_uuid)
        if tmp_path:
            cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        _cleanup_temp(tmp_path)
        return None

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if src_w <= 0 or src_h <= 0:
        cap.release()
        _cleanup_temp(tmp_path)
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
    # --time-spans 구간 띠는 영상 바로 아래, 텍스트 패널 위에 둔다.
    strip = build_timeline_strip(result, n_total / fps if n_total else 0.0,
                                 out_w, scale=out_w / 1280.0)
    strip_bgr, geom = ((cv2.cvtColor(strip[0], cv2.COLOR_RGB2BGR), strip[1])
                       if strip else (None, None))
    sh = strip_bgr.shape[0] if strip_bgr is not None else 0
    total_h = out_h + sh + panel.shape[0]
    total_h -= total_h % 2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((total_h, out_w, 3), 255, dtype=np.uint8)
    # 패널은 클립 내내 같으므로 한 번만 붙이고, 매 프레임 영상 영역만 갈아끼운다
    ph = min(panel_bgr.shape[0], total_h - out_h - sh)
    canvas[out_h + sh:out_h + sh + ph] = panel_bgr[:ph]

    def frames_iter():
        """원본에서 프레임을 읽어 패널 위에 얹은 canvas(BGR)를 흘려준다."""
        n = 0
        limit = max_frames or n_total or 10 ** 9
        draw = None
        if traj and traj_uuid:
            try:
                from trajectory import draw_trajectory
                draw = draw_trajectory
            except ImportError:
                draw = None
        while n < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                frame = cv2.resize(frame, (out_w, out_h),
                                   interpolation=cv2.INTER_AREA)
            if draw is not None:
                # 리사이즈 후에 그린다 - 선 두께가 출력 해상도에 맞고,
                # draw_trajectory 가 캘리브 해상도와의 배율을 알아서 맞춘다.
                draw(frame, traj_uuid, n, cam=traj_view, mode=traj)
            canvas[:out_h] = frame
            if strip_bgr is not None:
                # 띠를 다시 깔고 재생 위치에 세로선을 긋는다.
                canvas[out_h:out_h + sh] = strip_bgr
                t = n / fps
                x = int(round(geom["x0"] + min(t / geom["duration"], 1.0)
                              * (geom["x1"] - geom["x0"])))
                sc = out_w / 1280.0
                color = PLAYHEAD[::-1]          # RGB -> BGR
                cv2.line(canvas, (x, out_h + geom["y0"] - int(4 * sc)),
                         (x, out_h + geom["y1"] + int(2 * sc)), color,
                         max(1, int(round(2 * sc))))
                tri = np.array([[x - int(5 * sc), out_h + geom["y0"] - int(9 * sc)],
                                [x + int(5 * sc), out_h + geom["y0"] - int(9 * sc)],
                                [x, out_h + geom["y0"] - int(3 * sc)]], np.int32)
                cv2.fillPoly(canvas, [tri], color)
            yield canvas
            n += 1

    try:
        if HAVE_PYAV:
            n = _write_h264(out_path, frames_iter(), out_w, total_h, fps)
        else:
            n = _write_cv2(out_path, frames_iter(), out_w, total_h, fps, fourcc)
    finally:
        cap.release()
        _cleanup_temp(tmp_path)
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
        "reasoning": {k: result.get(k, "")
                      for keys, _ in PANEL_STEPS for k in keys},
        "group_scores": result.get("group_scores") or {},
        "score_reason": result.get("score_reason", ""),
        "tier_score": result.get("tier_score"),
        "parse_ok": bool(result.get("parse_ok", False)),
    }
    # --time-spans 를 켠 실행에서만 넣는다. time_spans 는 정수 초로 정리한
    # [시작, 끝, 점수], time_spans_raw 는 모델이 답한 값 그대로.
    if result.get("time_spans") or result.get("_time_spans_on"):
        payload["time_spans"] = result.get("time_spans") or {}
        payload["time_spans_raw"] = result.get("time_spans_raw") or {}
    # 난이도는 켠 실행에서만 넣는다 - 끈 실행의 json 에 전부 null 인 키가
    # 열 개 붙으면 재집계할 때 "쟀는데 못 읽음"과 구분이 안 된다.
    if any(result.get(k) is not None for k, _ in DIFFICULTY_AXES):
        payload["difficulty"] = {
            k: {"score": result.get(k), "max": DIFFICULTY_MAX,
                "reason": result.get(f"{k}_reason", "")}
            for k, _ in DIFFICULTY_AXES}
    if result.get("_gt") is not None:
        payload["ground_truth"] = result["_gt"]
    if extra:
        payload.update(extra)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return out_path


def render_clip_result(uuid: str, src_mp4, result: dict, out_dir,
                       max_width: int | None = VIZ_WIDTH,
                       extra: dict | None = None,
                       max_frames: int | None = None,
                       gt: dict | None = None,
                       traj: str | None = None,
                       traj_view: str = "camera_front_wide_120fov",
                       ):
    """클립 하나의 영상 + json 을 같은 폴더에 저장하고 (video, json) 을 반환.

    gt 를 주면(정답 라벨이 있는 클립) 패널에 GT 와 Pred 를 나란히 그린다.
    """
    out_dir = Path(out_dir)
    r = dict(result)
    r["_uuid"] = uuid
    if gt is not None:
        r["_gt"] = gt
    r["_headline"] = uuid
    video = render_clip_video(src_mp4, r, out_dir / "clip.mp4",
                              max_width=max_width, max_frames=max_frames,
                              traj=traj, traj_uuid=uuid, traj_view=traj_view,
                              )
    js = save_clip_json(r, out_dir / "result.json", extra=extra)
    return video, js

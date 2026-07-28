#!/usr/bin/env python3
"""판정 단위(클립 내 특정 순간)를 한 장의 카드 이미지로 시각화.

레이아웃:
  상단: 전방 3뷰(cross-left, front-wide, cross-right)의 같은 순간 정지 프레임
  하단: VLM 이 낸 JSON 결과 - verdict / categories(멀티라벨) / evidence
"""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIEW_LABELS = {
    "camera_cross_left_120fov": "Cross Left",
    "camera_front_wide_120fov": "Front Wide",
    "camera_cross_right_120fov": "Cross Right",
}
VIEW_ORDER = list(VIEW_LABELS.keys())

# verdict 별 강조색
VERDICT_COLORS = {
    "Special": "#B3541E",     # 주황 - 검토 대상
    "Normal_but": "#7A6A1F",  # 머스터드 - Normal 인데 카테고리가 붙은 애매한 건
    "Normal": "#6E7781",      # 회색
}
CHIP_BG = "#F5EDE4"
TEXT_DARK = "#1A1A1A"
TEXT_MID = "#333333"
TEXT_MUTED = "#666666"


def render_scene_card(uuid, frame_idx, frames_by_view, result, out_path):
    """판정 단위 하나를 카드 이미지로 렌더링해 out_path 에 저장.

    frames_by_view: {view_name: PIL.Image or None} (같은 순간의 3뷰 정지 프레임)
    result: {"verdict","categories","evidence"} (parse_vlm_output 결과)
    """
    out_path = Path(out_path)
    verdict = result.get("verdict", "Normal")
    cats = result.get("categories", []) or []
    evidence = result.get("evidence", "") or ""

    # Normal 인데 카테고리가 붙은 건 별도 색으로 구분해 눈에 띄게 한다
    key = "Normal_but" if (verdict == "Normal" and cats) else verdict
    accent = VERDICT_COLORS.get(key, VERDICT_COLORS["Normal"])

    fig = plt.figure(figsize=(15, 7.0), dpi=130, facecolor="white")
    gs = fig.add_gridspec(
        nrows=2, ncols=len(VIEW_ORDER),
        height_ratios=[3, 1.7],
        hspace=0.10, wspace=0.04,
        left=0.02, right=0.98, top=0.86, bottom=0.04,
    )

    fig.suptitle(
        f"{uuid}  (frame {frame_idx})",
        fontsize=14, fontweight="bold", color=TEXT_DARK, x=0.02, ha="left", y=0.975,
    )
    fig.text(0.02, 0.915, key.replace("_", " "), fontsize=12.5,
             fontweight="bold", color=accent, ha="left", va="center")

    # --- 상단: 3개 뷰 ---
    for i, view in enumerate(VIEW_ORDER):
        ax = fig.add_subplot(gs[0, i])
        frame = frames_by_view.get(view)
        if frame is not None:
            ax.imshow(frame)
        else:
            ax.text(0.5, 0.5, "no frame", ha="center", va="center",
                    transform=ax.transAxes, color="#999")
        ax.set_title(VIEW_LABELS[view], fontsize=11, color=TEXT_MID, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#DDDDDD")

    # --- 하단: JSON 결과 ---
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")

    ax.text(0.0, 1.0, f"Categories ({len(cats)}):", fontsize=11,
            fontweight="bold", color=TEXT_DARK, transform=ax.transAxes, va="top")

    if cats:
        # 카테고리를 칩 형태로 나열 (멀티라벨이라 개수가 가변)
        x, y = 0.0, 0.74
        for c in cats:
            t = ax.text(x, y, c, fontsize=10.5, color=accent, fontweight="bold",
                        transform=ax.transAxes, va="top",
                        bbox=dict(boxstyle="round,pad=0.34", facecolor=CHIP_BG,
                                  edgecolor=accent, linewidth=0.8))
            fig.canvas.draw()
            w = t.get_window_extent().transformed(ax.transAxes.inverted()).width
            x += w + 0.018
            if x > 0.88:          # 줄바꿈
                x, y = 0.0, y - 0.22
    else:
        ax.text(0.0, 0.74, "— none —", fontsize=10.5, color=TEXT_MUTED,
                transform=ax.transAxes, va="top")

    ax.text(0.0, 0.30, "Evidence:", fontsize=11, fontweight="bold",
            color=TEXT_DARK, transform=ax.transAxes, va="top")
    ax.text(0.0, 0.15, "\n".join(textwrap.wrap(evidence, width=140)) or "—",
            fontsize=10.5, color=TEXT_MID, style="italic",
            transform=ax.transAxes, va="top")

    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return out_path

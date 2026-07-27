#!/usr/bin/env python3
"""special 로 분류된 판정 단위(클립 내 특정 순간)를 한 장의 카드 이미지로 시각화.

레이아웃:
  상단: 전방 3뷰(front_wide, cross_left, cross_right)의 같은 순간 정지 프레임 3장
  하단: 1단계 캡션 + 2단계 top-3 (scenario, category, confidence)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap

VIEW_LABELS = {
    "camera_cross_left_120fov": "Cross Left",
    "camera_front_wide_120fov": "Front Wide",
    "camera_cross_right_120fov": "Cross Right",
}
VIEW_ORDER = list(VIEW_LABELS.keys())

# dataviz 팔레트 기준 카테고리 강조색 (top1/2/3 순위 표시용)
RANK_COLORS = ["#B3541E", "#6E7781", "#8A8F98"]  # 1st 강조, 2nd/3rd 보조


def render_scene_card(uuid, frame_idx, frames_by_view, caption, matches, out_path):
    """판정 단위(uuid, frame_idx) 하나를 카드 이미지로 렌더링해 out_path에 저장.

    frames_by_view: {view_name: PIL.Image or None} (같은 순간의 3뷰 정지 프레임)
    matches: [{"scenario":..., "category":..., "confidence":...}, ...] (최대 3개)
    """
    out_path = Path(out_path)
    n_views = len(VIEW_ORDER)

    fig = plt.figure(figsize=(15, 7.0), dpi=130, facecolor="white")
    gs = fig.add_gridspec(
        nrows=2, ncols=n_views,
        height_ratios=[3, 1.7],
        hspace=0.10, wspace=0.04,
        left=0.02, right=0.98, top=0.86, bottom=0.04,
    )

    top1_cat = matches[0]["category"] if matches else "OOD"
    fig.suptitle(
        f"{uuid}  (frame {frame_idx})   —   classified as: {top1_cat}",
        fontsize=15, fontweight="bold", color="#1A1A1A", x=0.02, ha="left", y=0.97,
    )

    # --- 상단: 3개 뷰, 같은 순간의 정지 프레임 ---
    for i, view in enumerate(VIEW_ORDER):
        ax = fig.add_subplot(gs[0, i])
        frame = frames_by_view.get(view)
        if frame is not None:
            ax.imshow(frame)
        else:
            ax.text(0.5, 0.5, "no frame", ha="center", va="center",
                    transform=ax.transAxes, color="#999")
        ax.set_title(VIEW_LABELS[view], fontsize=11, color="#333333", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#DDDDDD")

    # --- 하단: 캡션 + top-3 테이블 (전체 폭 합쳐서 사용) ---
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")

    wrapped_caption = "\n".join(textwrap.wrap(caption, width=140))
    ax_text.text(
        0.0, 1.0, "Stage 1 — Scene caption:",
        fontsize=11, fontweight="bold", color="#1A1A1A",
        transform=ax_text.transAxes, va="top",
    )
    ax_text.text(
        0.0, 0.80, wrapped_caption,
        fontsize=10.5, color="#333333", style="italic",
        transform=ax_text.transAxes, va="top",
    )

    table_top = 0.42
    ax_text.text(
        0.0, table_top + 0.14, "Stage 2 — Top-3 category matches:",
        fontsize=11, fontweight="bold", color="#1A1A1A",
        transform=ax_text.transAxes, va="top",
    )

    col_x = [0.0, 0.06, 0.34, 0.62, 0.80]
    headers = ["#", "Scenario", "Category", "Confidence"]
    header_x = [col_x[1], col_x[2], col_x[3], col_x[4]]
    for hx, htext in zip(header_x, headers):
        ax_text.text(hx, table_top, htext, fontsize=9.5, fontweight="bold",
                    color="#666666", transform=ax_text.transAxes, va="top")

    row_h = 0.11
    for rank in range(3):
        y = table_top - row_h * (rank + 1)
        color = RANK_COLORS[rank]
        if rank < len(matches):
            m = matches[rank]
            vals = [str(rank + 1), m["scenario"], m["category"], f"{m['confidence']:.2f}"]
        else:
            vals = [str(rank + 1), "—", "—", "—"]
        for hx, v in zip(header_x, vals):
            weight = "bold" if rank == 0 else "normal"
            ax_text.text(hx, y, v, fontsize=10, color=color, fontweight=weight,
                        transform=ax_text.transAxes, va="top")

    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return out_path

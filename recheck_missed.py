#!/usr/bin/env python3
"""놓친 카테고리를 후보만 골라 2차로 다시 묻는다 (recheck).

왜 필요한가:
  1차 추론에서 놓치는 클립의 대부분은 "1단계 장면 설명에는 적고 3단계 요소
  나열로 옮기지 못한" 경우다. 실측(20261012_165207, Dynamic object 만):
  GT 에 동적 카테고리가 있는데 하나도 못 찾은 30개 중 18개가 장면 설명에
  그 객체를 적어 두었다("…cyclists cross the intersection…" 인데 3단계 빈칸).
  그렇다고 장면 설명에 키워드가 있으면 무조건 올리면, GT 에 동적 카테고리가
  없는 225개 중 35개도 같은 키워드가 있어(그중 15개는 인도 위) 오탐이 더
  많이 생긴다. 그래서 키워드로는 "후보"만 고르고, 후보마다 영상과 함께 그
  카테고리 하나만 짧게 다시 물어 확인한다.

1차 프롬프트는 건드리지 않는다 - 지금까지 맞춰 온 탐지/점수 동작을 그대로
두고, 이 단계의 효과만 따로 재기 위해서다.

후보:
  1차에서 카테고리를 하나도 못 찾은(Normal) 클립 중, 장면 설명/요소 나열에
  어떤 카테고리의 recheck_keywords(scene_category.json)가 나온 클립.
  카테고리 범위는 1차 실행과 같다(run_config 의 group, unknown_categories).

질문 하나에 카테고리 하나: 그 카테고리의 메뉴 항목(설명 + NOT 줄)만 보여 주고
"이 클립에 이것으로 칠 요소가 있나"를 묻는다. 있으면 그 카테고리가 속한
상위 카테고리 기준표(1~4)로 점수도 받는다 - 1차에서 그 상위 카테고리는
0 이었으므로(아무것도 못 찾음) 이 점수로 채운다.

쓰는 법 (GPU 필요):
  python recheck_missed.py --run-dir results/labeld/<실행>              # 후보 추론
  python recheck_missed.py --run-dir results/labeld/<실행> --list-only  # 후보만 보기
  python recheck_missed.py --run-dir results/labeld/<실행> --apply      # 결과 반영
  샤드로 나눌 때: --num-shards N --shard-id k (샤드마다 GPU 하나), 끝나면 --apply

  --apply 는 clip_results_all.csv 를 고친다. 원본은 같은 폴더의
  before_recheck_clip_results_all.csv 로 남긴다(이름이 clip_results 로
  시작하지 않아 채점이 읽지 않는다). 다시 --apply 하면 원본에서 새로 만든다.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECHECK_DIR = "recheck"
BACKUP_NAME = "before_recheck_clip_results_all.csv"
MERGED_NAME = "clip_results_all.csv"

csv.field_size_limit(10**9)


# ---------------------------------------------------------------------------
# 실행 설정 / 카테고리
# ---------------------------------------------------------------------------
def load_run_config(run_dir: Path) -> dict:
    p = run_dir / "run_config.json"
    if not p.exists():
        raise SystemExit(f"[error] {p} 가 없습니다 - 1차 실행 설정을 알 수 없습니다")
    return json.loads(p.read_text(encoding="utf-8"))


def run_labels(cfg: dict):
    """1차 실행과 같은 범위의 카테고리 목록(load_labels 결과)과 scene 경로."""
    import edge_case_mining as E
    from constrained_tier import group_key
    key = cfg.get("key", {})
    scene = Path(key.get("scene_json") or E.SCENE_JSON)
    if not scene.is_absolute():
        scene = ROOT / scene
    labels = E.load_labels(scene)
    if not key.get("unknown_categories"):
        labels = [l for l in labels if not l.get("unknown")]
    group = key.get("group") or "all"
    if group != "all":
        labels = [l for l in labels
                  if l["is_normal"] or group_key(l["scenario"]) == group]
    # recheck_keywords 는 load_labels 가 싣지 않으므로 scene 에서 따로 읽는다.
    data = json.loads(scene.read_text(encoding="utf-8"))
    kw = {c["name"]: c.get("recheck_keywords", [])
          for s in data["special"]["scenarios"] for c in s["categories"]}
    for l in labels:
        l["recheck_keywords"] = kw.get(l["category"], [])
    return labels, scene


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


# ---------------------------------------------------------------------------
# 후보
# ---------------------------------------------------------------------------
def find_candidates(rows: list[dict], labels: list) -> list[tuple[str, str]]:
    """[(uuid, 카테고리)] - 1차에서 Normal 인데 서술에 키워드가 나온 것."""
    pats = {l["category"]: [re.compile(k, re.I) for k in l["recheck_keywords"]]
            for l in labels if not l["is_normal"] and l["recheck_keywords"]}
    out = []
    for r in rows:
        try:
            if int(r.get("n_frames") or 0) <= 0 or r.get("parse_ok") != "1":
                continue
        except ValueError:
            continue
        if (r.get("categories") or "").strip():
            continue
        text = " ".join(r.get(k) or "" for k in
                        ("observation", "unusual_elements", "ego_behavior"))
        for cat, ps in pats.items():
            if any(p.search(text) for p in ps):
                out.append((r["uuid"], cat))
    return out


# ---------------------------------------------------------------------------
# 질문
# ---------------------------------------------------------------------------
def build_question(intro: str, behavior: str, observation: str, lab: dict) -> str:
    import edge_case_mining as E
    from constrained_tier import CATEGORY_RUBRICS, rubric_of, values_text
    entry = E.build_category_menu([lab])
    # 자기 기준표로 따로 재는 카테고리(Unpaved road)는 그 표로 묻는다.
    group = (lab["category"] if lab["category"] in CATEGORY_RUBRICS
             else lab["scenario"])
    title, rubric = rubric_of(group)
    scale = "\n".join(f"     {v} = {rubric[v]}" for v in sorted(rubric) if v >= 1)
    present = values_text([v for v in rubric if v >= 1])
    facts = (f"\nHOW THE EGO-VEHICLE'S BEHAVIOUR CHANGED (measured from vehicle "
             f"sensors, not a guess):\n{behavior}\n" if behavior else "")
    return f"""{intro}
{facts}
An earlier look at this clip described it as: "{observation}"

Check ONE scenario type only:
{entry}

Does this clip contain an element that counts as this type? Decide only by the
type's description and its "NOT this category" lines - never by whether the
ego-vehicle reacted to it. If it counts, also score how much it made driving
harder for the ego-vehicle ({group} - {title}):
{scale}
Slowing or stopping is routine (signals, junctions, queues), so treat the
ego-vehicle's own braking as evidence only when the video shows what caused it.

Respond with ONLY a JSON object, no other text:
{{"reason": "<what you see of this type and what it did>",
 "present": <true or false>,
 "score": <integer {present} if present, otherwise 0>}}"""


def parse_answer(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    out = {"present": False, "score": 0, "reason": "", "parse_ok": False}
    if not m:
        return out
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return out
    pres = obj.get("present")
    if isinstance(pres, str):
        pres = pres.strip().lower() in ("true", "yes")
    out["present"] = bool(pres)
    try:
        n = int(obj.get("score") or 0)
    except (TypeError, ValueError):
        n = 0
    # 있다고 했는데 점수가 0~4 밖이거나 0 이면 1 로 둔다 - 4단계 규칙과 같이
    # "이름을 댄 상위 카테고리는 최소 1" 이다.
    out["score"] = min(max(n, 1), 4) if out["present"] else 0
    out["reason"] = str(obj.get("reason") or "").strip()
    out["parse_ok"] = True
    return out


def _answers(run_dir: Path) -> dict:
    """{(uuid, 카테고리): 답} - 모든 샤드 파일에서 읽는다. 같은 질문이 여럿이면
    나중 것을 쓴다(같은 프롬프트라 답도 같다)."""
    out = {}
    for p in sorted((run_dir / RECHECK_DIR).glob("recheck_shard_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                out[(d["uuid"], d["category"])] = d
            except Exception:
                pass
    return out


def skip_reason(cfg: dict) -> str | None:
    """2차 확인을 하지 않을 실행이면 그 이유. 날씨만 추론한 실행은 카테고리를
    찾지 않았으므로 놓친 것을 다시 물을 대상이 없다."""
    if (cfg.get("key") or {}).get("difficulty_only"):
        return "날씨만 추론한 실행(--weather-only)이라 카테고리가 없음"
    return None


def run_shard(run_dir: Path, num_shards: int, shard_id: int):
    import edge_case_mining as E
    import qwen_runner as Q
    cfg = load_run_config(run_dir)
    if skip_reason(cfg):
        print(f"[recheck] 건너뜀: {skip_reason(cfg)}")
        return
    key, argv = cfg.get("key", {}), cfg.get("argv", {})
    labels, _ = run_labels(cfg)
    by_name = {l["category"]: l for l in labels}
    src = run_dir / (BACKUP_NAME if (run_dir / BACKUP_NAME).exists() else MERGED_NAME)
    _, rows = read_rows(src)
    rows_by_uuid = {r["uuid"]: r for r in rows}
    cands = sorted(find_candidates(rows, labels))[shard_id::num_shards]
    out_dir = run_dir / RECHECK_DIR
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"recheck_shard_{shard_id}.jsonl"
    # 이미 답한 질문은 건너뛴다(끊겼다가 다시 돌릴 때). 자기 샤드 파일만 보면
    # 샤드 수를 바꿔 다시 돌릴 때 몫이 달라져 같은 질문을 또 묻는다(실측
    # 20261013_recheck_test: 6샤드 -> 8샤드 재실행에서 71건 중 53건 중복) -
    # 그래서 모든 샤드 파일을 본다.
    done = set(_answers(run_dir))
    todo = [c for c in cands if c not in done]
    print(f"[recheck] shard {shard_id}/{num_shards}: 후보 {len(cands)}건, "
          f"남은 {len(todo)}건 (source {src.name})")
    if not todo:
        return

    E.set_clip_source(argv.get("data") or key.get("data") or "local")
    views = E.views_for(bool(key.get("single_view", True)))
    fps = float(key.get("clip_fps") or E.CLIP_FPS)
    max_frames = int(key.get("clip_max_frames") or E.CLIP_MAX_FRAMES)
    long_side = int(key.get("clip_long_side") or E.CLIP_MAX_LONG_SIDE)
    as_video = bool(key.get("video_input", True))
    traj = key.get("traj")
    model_id = key.get("model") or argv.get("model")
    model, processor = Q.load_model(model_id)

    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, (uuid, cat) in enumerate(todo, 1):
            clip = E.sample_clip_frames(uuid, fps=fps, max_frames=max_frames,
                                        max_long_side=long_side, views=views,
                                        traj=traj)
            images = [im for _, _, im in clip["frames"]]
            if len(images) < len(clip["indices"]) * len(views):
                print(f"[warn] {uuid}: 프레임을 다 읽지 못해 건너뜀")
                continue
            behavior, _ = Q.build_clip_ego_facts(uuid, bool(key.get("use_egomotion")))
            intro = E.clip_intro(len(images), len(views), fps=fps,
                                 as_video=as_video)
            q = build_question(intro, behavior,
                               rows_by_uuid[uuid].get("observation", ""),
                               by_name[cat])
            raw = Q._generate(model, processor, images, q, max_new_tokens=160,
                              as_video=as_video, video_fps=fps,
                              thinking=(model_id in Q.THINKING_MODELS))
            ans = parse_answer(raw)
            f.write(json.dumps({"uuid": uuid, "category": cat,
                                "group": by_name[cat]["scenario"], **ans,
                                "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[recheck] {i}/{len(todo)} {uuid[:8]} {cat}: "
                  f"{'YES ' + str(ans['score']) if ans['present'] else 'no'}"
                  f"  ({(time.time() - t0) / i:.1f}s/q)", flush=True)


# ---------------------------------------------------------------------------
# 반영
# ---------------------------------------------------------------------------
def _parse_scores(raw: str) -> dict:
    out = {}
    for part in (raw or "").split("|"):
        if "=" in part:
            k, _, v = part.rpartition("=")
            try:
                out[k.strip()] = int(v)
            except ValueError:
                pass
    return out


def apply(run_dir: Path, min_score: int = 2):
    """recheck 결과를 clip_results_all.csv 에 반영한다(원본은 백업에서 다시 만든다).

    min_score: YES 이면서 점수가 이 값 이상인 답만 반영한다. 점수 1 은 기준표에서
    "있지만 자차에 아무것도 요구하지 않음"인데, 1차가 Normal 로 본 클립에서
    recheck 가 점수 1 로 YES 한 것은 대부분 오탐이었다(152848: 8건 중 6건,
    162702: 6건 중 6건 - 인도 위 보행자, 길가에 주차된 구급차/경찰차, 멀리 보이는
    트램 등). 2 로 두면 영향을 준 놓친 요소만 되살린다 - 기본값(2026-10-21):
    152848 판정 F1 82.7 -> 83.2, macro 0.763 -> 0.778 / 162702 82.8 -> 83.7,
    macro 0.731 -> 0.760. 점수 단계가 없던 실행은 점수가 없으므로 이 조건을
    보지 않는다.
    """
    from constrained_tier import tier_score
    cfg0 = load_run_config(run_dir)
    if skip_reason(cfg0):
        print(f"[recheck] 반영 안 함: {skip_reason(cfg0)}")
        return
    # 점수 단계가 없던 실행(옛 --no-tiers-elements)이면 카테고리만 추가하고
    # 점수 칸은 건드리지 않는다 - 1차 결과에 없던 점수가 일부 클립에만 생긴다.
    with_scores = bool((cfg0.get("key") or {}).get("score_categories", True))
    merged, backup = run_dir / MERGED_NAME, run_dir / BACKUP_NAME
    if not backup.exists():
        if not merged.exists():
            raise SystemExit(f"[error] {merged} 가 없습니다 - merge_shards.py 먼저")
        backup.write_bytes(merged.read_bytes())
        print(f"[recheck] 원본 보관 -> {backup.name}")
    cols, rows = read_rows(backup)
    answers = list(_answers(run_dir).values())
    yes, n_low = {}, 0
    for a in answers:
        if not a.get("present"):
            continue
        if with_scores and min_score > 1 and (a.get("score") or 0) < min_score:
            n_low += 1
            continue
        yes.setdefault(a["uuid"], []).append(a)
    n_clip = 0
    for r in rows:
        hits = yes.get(r["uuid"])
        if not hits:
            continue
        cats = [c for c in (r.get("categories") or "").split("|") if c]
        scores = _parse_scores(r.get("group_scores", ""))
        reasons = [r.get("score_reason") or ""]
        for a in hits:
            if a["category"] not in cats:
                cats.append(a["category"])
            if with_scores:
                scores[a["group"]] = max(scores.get(a["group"], 0), a["score"])
            reasons.append(f"recheck {a['category']}: {a['reason']}")
        r["categories"] = "|".join(cats)
        r["n_categories"] = str(len(cats))
        r["verdict"] = "Special"
        if with_scores:
            r["group_scores"] = "|".join(f"{k}={v}" for k, v in scores.items())
        r["score_reason"] = " | ".join(x for x in reasons if x)
        if with_scores:
            r["tier_score"] = str(tier_score(scores) if scores else "")
        n_clip += 1
    tmp = merged.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(merged)
    n_yes = sum(len(v) for v in yes.values())
    print(f"[recheck] 질문 {len(answers)}건 중 YES {n_yes}건 -> "
          f"{n_clip}개 클립을 Special 로 바꿔 {merged.name} 에 반영"
          + (f" (점수 {min_score} 미만 YES {n_low}건은 버림)" if n_low else ""))
    # 실행 설정에 기록을 남긴다 - 나중에 이 폴더의 결과가 recheck 를 거쳤는지
    # 알 수 있게.
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["recheck"] = {"applied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "questions": len(answers), "yes": n_yes,
                          "clips_changed": n_clip, "min_score": min_score,
                          "yes_dropped_low_score": n_low}
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                            encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--list-only", action="store_true",
                    help="후보만 출력하고 끝낸다 (GPU 불필요)")
    ap.add_argument("--apply", action="store_true",
                    help="recheck 결과를 clip_results_all.csv 에 반영한다 (GPU 불필요)")
    ap.add_argument("--min-score", type=int, default=2,
                    help="--apply 때 YES 이면서 점수가 이 값 이상인 답만 반영 "
                         "(기본 2 = '있지만 영향 없음(1점)' YES 를 버린다. "
                         "1 이면 YES 전부 - 예전 동작)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if args.apply:
        apply(run_dir, min_score=args.min_score)
        return
    if args.list_only:
        cfg = load_run_config(run_dir)
        labels, scene = run_labels(cfg)
        src = run_dir / (BACKUP_NAME if (run_dir / BACKUP_NAME).exists() else MERGED_NAME)
        _, rows = read_rows(src)
        cands = find_candidates(rows, labels)
        print(f"[recheck] {src.name}: 후보 {len(cands)}건 "
              f"({len({u for u, _ in cands})}개 클립), 범위 group="
              f"{cfg.get('key', {}).get('group', 'all')}, scene={scene.name}")
        for u, c in cands:
            print(f"  {u}  {c}")
        return
    run_shard(run_dir, args.num_shards, args.shard_id)


if __name__ == "__main__":
    sys.exit(main())

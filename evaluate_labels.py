#!/usr/bin/env python3
"""사람이 만든 정답 라벨(test_label.json)과 모델 결과를 대조해 성능을 잰다.

네 축을 따로 본다 - 하나로 뭉치면 무엇이 문제인지 안 보인다:

  1) VERDICT     Normal/Special 이진 판정. categories 가 비었는가로 파생되며,
                 파이프라인의 판정 규칙과 정확히 같다.
  2) CATEGORIES  멀티라벨 F1. "정답 n개 중 m개" (m/n) 를 쓰지 않는 이유는
                 그 식의 분모가 정답 쪽이라 오탐에 벌점이 없기 때문이다 -
                 정답이 [Jaywalking] 인데 모델이 4개를 다 찍어도 1.0 이 되어,
                 "전부 찍기"가 최적 전략이 되어버린다. F1 은 precision 을
                 함께 보므로 그 문제가 없다.
  3) SAFETY      1~4 정수. MAE + 정확일치 + 혼동행렬.
  4) RARITY      동일.

유병률 보정:
  정답 표본은 normal:special = 50:50 이지만 실제 데이터는 약 81:19 다.
  precision 은 표본에 normal 이 몇 개 들어있느냐에 직접 영향을 받으므로,
  50:50 표본의 precision 을 그대로 보고하면 실제보다 크게 부풀려진다.
  recall 과 specificity 는 표본 비율과 무관하므로, 그 둘로 실제 유병률에서의
  precision 을 다시 계산해 함께 보여준다.

기준선(baseline):
  등급은 "항상 2" 로 찍어도 MAE 가 꽤 낮게 나온다(실측 safety=2 가 73%).
  그래서 최빈값 예측의 MAE 를 함께 내고, 모델이 그걸 이기는지 본다.
  이기지 못하면 모델이 등급을 실제로 판단하는 것이 아니다.

Usage:
  python evaluate_labels.py --run-dir results/20260812_135056_videoC
  python evaluate_labels.py --run-dir <dir> --labels test_label.json
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from constrained_tier import TIER_VALUES

ROOT = Path(__file__).resolve().parent
DEFAULT_LABELS = ROOT / "test_label.json"
SCENE_JSON = ROOT / "scene_category_C.json"

# 실제 데이터셋의 special 비율. 유병률 보정에 쓴다.
# 실측: 1,998 클립 중 375 건이 edge-case (20260812 전량 실행).
DEFAULT_PREVALENCE = 375 / 1998


def load_labels(path: Path):
    """정답 라벨 -> {uuid: {categories, safety, rarity, note}}.

    safety 키는 'safety_criticality' 가 정식이지만 손으로 쓰다 보면
    'safty_criticality' 오타가 섞인다(실측 7/100). 조용히 0 으로 처리하면
    점수가 왜곡되므로 둘 다 받아들이고, 아예 없으면 그 클립을 건너뛴다.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    clips = data.get("clips", data)
    out, skipped = {}, []
    for uuid, v in clips.items():
        s = v.get("safety_criticality", v.get("safty_criticality"))
        r = v.get("rarity")
        if s is None or r is None:
            skipped.append(uuid)
            continue
        out[uuid] = {
            "categories": set(v.get("categories") or []),
            "safety": int(s),
            "rarity": int(r),
            "note": v.get("note", ""),
        }
    return out, data.get("_meta", {}), skipped


def load_results(run_dir: Path):
    """모델 결과 CSV(샤드 병합) -> {uuid: {categories, safety, rarity}}."""
    files = sorted(glob.glob(str(run_dir / "clip_results*.csv")))
    if not files:
        return {}
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset="uuid", keep="first")
    out = {}
    for _, r in df.iterrows():
        cats = r.get("categories")
        cats = set(str(cats).split("|")) if isinstance(cats, str) and cats.strip() else set()
        out[str(r["uuid"])] = {
            "categories": cats,
            "safety": _int_or_none(r.get("safety_tier")),
            "rarity": _int_or_none(r.get("rarity_tier")),
        }
    return out


def _int_or_none(v):
    try:
        if pd.isna(v):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def f1_set(truth: set, pred: set) -> float:
    """멀티라벨 한 클립의 F1. 둘 다 비어 있으면 1.0 (정답을 맞힌 것)."""
    if not truth and not pred:
        return 1.0
    tp = len(truth & pred)
    return prf(tp, len(pred - truth), len(truth - pred))[2]


def confusion(pairs, values=None):
    """[(truth, pred), ...] -> 문자열 혼동행렬 (행 합계 total 포함).

    values 를 안 주면 TIER_VALUES 를 쓴다 - 등급 개수가 바뀌어도(1~3 -> 1~4)
    표가 따라 넓어지도록 한 곳만 보게 한다.
    행 합계를 같이 내는 이유는 "정답 3인 20건 중 몇 건을 2로 봤나" 같은
    질문이 이 표를 읽는 주된 목적이라, 분모가 옆에 있어야 바로 읽히기 때문.
    """
    if values is None:
        values = TIER_VALUES
    c = Counter(pairs)
    w = 9                                    # 열 너비
    head = "       " + "".join(f"{'pred'+str(v):>{w}}" for v in values)
    lines = [head + f"{'total':>{w}}"]
    for t in values:
        row = "".join(f"{c.get((t, p), 0):>{w}}" for p in values)
        total = sum(c.get((t, p), 0) for p in values)
        lines.append(f"  true{t}{row}{total:>{w}}")
    return "\n".join(lines)


def tier_block(name, pairs, log):
    """등급 한 축(safety/rarity)의 MAE / 정확일치 / 혼동행렬 / 기준선."""
    if not pairs:
        log(f"  (no comparable rows)")
        return
    n = len(pairs)
    mae = sum(abs(t - p) for t, p in pairs) / n
    exact = sum(t == p for t, p in pairs) / n
    # 기준선: 정답에서 가장 흔한 등급으로 전부 찍었을 때
    mode = Counter(t for t, _ in pairs).most_common(1)[0][0]
    base_mae = sum(abs(t - mode) for t, _ in pairs) / n
    base_exact = sum(t == mode for t, _ in pairs) / n

    log(f"  n              : {n}")
    log(f"  MAE            : {mae:.3f}      baseline(always {mode}): {base_mae:.3f}"
        f"   {'BEATS' if mae < base_mae else 'DOES NOT BEAT'} baseline")
    log(f"  exact match    : {exact:6.1%}    baseline: {base_exact:6.1%}")
    log(f"  within +-1     : {sum(abs(t-p)<=1 for t,p in pairs)/n:6.1%}")
    log("")
    log(confusion(pairs))


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8") if path else None

    def __call__(self, line=""):
        print(line)
        if self.f:
            self.f.write(line + "\n")

    def close(self):
        if self.f:
            self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="모델 결과 폴더 (clip_results*.csv 가 있는 곳)")
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--prevalence", type=float, default=DEFAULT_PREVALENCE,
                    help="실제 데이터의 special 비율 (precision 보정용). "
                         f"기본 {DEFAULT_PREVALENCE:.3f} = 375/1998")
    ap.add_argument("--out", default=None,
                    help="결과를 저장할 로그 파일. 생략 시 <run-dir>/evaluation.log")
    ap.add_argument("--min-support", type=int, default=3,
                    help="카테고리별 지표를 따로 보여줄 최소 정답 건수 (기본 3)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    labels, meta, skipped = load_labels(Path(args.labels))
    results = load_results(run_dir)
    if not results:
        print(f"[error] no clip_results*.csv in {run_dir}")
        return

    log = Tee(Path(args.out) if args.out else run_dir / "evaluation.log")
    log(f"[info] labels  : {args.labels}  ({len(labels)} clips)")
    log(f"[info] results : {run_dir}  ({len(results)} clips)")
    if meta:
        log(f"[info] labeled_by={meta.get('labeled_by','?')} "
            f"date={meta.get('date','?')}")
    if skipped:
        log(f"[warn] {len(skipped)} label(s) missing safety/rarity - skipped")

    common = [u for u in labels if u in results]
    missing = [u for u in labels if u not in results]
    log(f"[info] matched : {len(common)}/{len(labels)}")
    if missing:
        log(f"[warn] {len(missing)} labelled clip(s) not in results, e.g. "
            f"{missing[:3]}")

    # 카테고리 이름이 taxonomy 와 어긋나면 그 카테고리의 F1 이 0 으로 나온다.
    # 실제 성능이 아니라 버전 불일치인데 숫자만 보면 구분이 안 되므로, 채점
    # 전에 짚어준다. 실측 사례: 예전 실행이 소문자 'cyclist' 를 쓰던 시절의
    # 결과를 대문자 'Cyclist' 라벨로 채점하자 F1=0.00 이 나왔다.
    if SCENE_JSON.exists():
        scene = json.loads(SCENE_JSON.read_text(encoding="utf-8"))
        taxonomy = {c["name"] for s in scene["special"]["scenarios"]
                    for c in s["categories"]}
        lab_names = {c for u in common for c in labels[u]["categories"]}
        res_names = {c for u in common for c in results[u]["categories"]}
        for tag, names in (("labels", lab_names), ("results", res_names)):
            bad = names - taxonomy
            if bad:
                log(f"[warn] {tag} use category names not in "
                    f"{SCENE_JSON.name}: {sorted(bad)}")
        only_lab = sorted(lab_names - res_names)
        only_res = sorted(res_names - lab_names)
        if only_lab and only_res:
            log(f"[warn] names present only in labels {only_lab} vs only in "
                f"results {only_res} - if these look like the same category "
                f"spelled differently, the run predates the current taxonomy "
                f"and its scores below are meaningless for those rows.")
    if not common:
        log("[error] no overlap between labels and results")
        log.close()
        return

    # ---------------- 1) VERDICT ----------------
    tp = fp = fn = tn = 0
    for u in common:
        t, p = bool(labels[u]["categories"]), bool(results[u]["categories"])
        tp += t and p
        fp += (not t) and p
        fn += t and (not p)
        tn += (not t) and (not p)
    P, R, F = prf(tp, fp, fn)
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    log("")
    log("=" * 68)
    log("1) VERDICT  (Special = has >=1 category)")
    log("=" * 68)
    log(f"  truth: special={tp+fn}  normal={tn+fp}")
    log(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    log(f"  accuracy    : {(tp+tn)/len(common):6.1%}")
    log(f"  precision   : {P:6.1%}   recall: {R:6.1%}   F1: {F:6.1%}")
    log(f"  specificity : {spec:6.1%}")
    # 유병률 보정 - recall/specificity 는 표본 비율과 무관하다는 성질을 쓴다
    pv = args.prevalence
    tp_r, fp_r = R * pv, (1 - spec) * (1 - pv)
    P_adj = tp_r / (tp_r + fp_r) if (tp_r + fp_r) else 0.0
    F_adj = 2 * P_adj * R / (P_adj + R) if (P_adj + R) else 0.0
    log("")
    log(f"  adjusted to real prevalence ({pv:.1%} special):")
    log(f"    precision : {P_adj:6.1%}   recall: {R:6.1%}   F1: {F_adj:6.1%}")
    log("    (label set is 50:50, so the raw precision above is optimistic;")
    log("     this row estimates what you would get on all 1,998 clips)")

    # ---------------- 2) CATEGORIES ----------------
    per_clip = [f1_set(labels[u]["categories"], results[u]["categories"])
                for u in common]
    sp = [u for u in common if labels[u]["categories"]]
    per_clip_sp = [f1_set(labels[u]["categories"], results[u]["categories"])
                   for u in sp]

    names = sorted({c for u in common for c in labels[u]["categories"]}
                   | {c for u in common for c in results[u]["categories"]})
    rows, micro = [], [0, 0, 0]
    for c in names:
        t = sum(c in labels[u]["categories"] for u in common)
        ctp = sum(c in labels[u]["categories"] and c in results[u]["categories"]
                  for u in common)
        cfp = sum(c not in labels[u]["categories"] and c in results[u]["categories"]
                  for u in common)
        cfn = t - ctp
        micro[0] += ctp; micro[1] += cfp; micro[2] += cfn
        rows.append((c, t, *prf(ctp, cfp, cfn), ctp, cfp, cfn))

    log("")
    log("=" * 68)
    log("2) CATEGORIES  (multi-label)")
    log("=" * 68)
    log(f"  per-clip F1, all {len(common)} clips  : {sum(per_clip)/len(per_clip):.3f}")
    if per_clip_sp:
        log(f"  per-clip F1, {len(sp)} special only : "
            f"{sum(per_clip_sp)/len(per_clip_sp):.3f}")
    log(f"  micro F1  : {prf(*micro)[2]:.3f}")
    scored = [r for r in rows if r[1] >= args.min_support]
    if scored:
        log(f"  macro F1  : {sum(r[4] for r in scored)/len(scored):.3f}"
            f"   (over {len(scored)} categories with >={args.min_support} labels)")
    log("")
    log(f"  {'category':<32}{'n':>4}{'P':>8}{'R':>8}{'F1':>8}"
        f"{'TP':>5}{'FP':>5}{'FN':>5}")
    log("  " + "-" * 66)
    for c, t, p, r, f, ctp, cfp, cfn in sorted(rows, key=lambda x: -x[1]):
        mark = " " if t >= args.min_support else "*"
        log(f" {mark}{c:<32}{t:>4}{p:>8.2f}{r:>8.2f}{f:>8.2f}"
            f"{ctp:>5}{cfp:>5}{cfn:>5}")
    log("  * = too few labels to read the numbers reliably")

    # ---------------- 3/4) SAFETY, RARITY ----------------
    for key, title in (("safety", "3) SAFETY CRITICALITY"),
                       ("rarity", "4) RARITY")):
        log("")
        log("=" * 68)
        log(f"{title}  ({min(TIER_VALUES)}-{max(TIER_VALUES)})")
        log("=" * 68)
        pairs_all = [(labels[u][key], results[u][key]) for u in common
                     if results[u][key] is not None]
        log(" ALL clips")
        tier_block(key, pairs_all, log)
        pairs_sp = [(labels[u][key], results[u][key]) for u in sp
                    if results[u][key] is not None]
        if pairs_sp and len(pairs_sp) != len(pairs_all):
            log("")
            log(" SPECIAL clips only  (normal clips are almost all 1, which"
                " inflates the scores above)")
            tier_block(key, pairs_sp, log)

    log("")
    log(f"[saved] {args.out or run_dir / 'evaluation.log'}")
    log.close()


if __name__ == "__main__":
    main()

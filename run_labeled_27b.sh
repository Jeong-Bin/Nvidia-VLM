#!/usr/bin/env bash
# 라벨 평가 (Qwen3.8-27B) - 로컬 pav_sample 의 라벨된 클립만 추론하고 채점한다.
#
# 네 스크립트의 역할 분담:
#   run_labeled_8b.sh      8B + 8샤드,  로컬 라벨 클립,  추론 + 평가
#   run_labeled_27b.sh     27B 파이프라인 병렬, 로컬 라벨 클립, 추론 + 평가  <- 이 파일
#   run_nas_nvidia_8b.sh   8B + 8샤드,  NAS 청크 zip,   추론만
#   run_nas_nvidia_27b.sh  27B 파이프라인 병렬, NAS 청크 zip, 추론만
#
# 27B 는 bf16 55.6GB 라 4090 한 장(24GB)에 안 들어간다. device_map="auto" 로
# 한 프로세스가 GPU 여러 장에 층을 나눠 얹는 파이프라인 병렬을 쓴다. 그래서
# 8B 처럼 샤드를 쪼개 병렬로 돌리지 못하고 --num-shards 1 로 고정한다.
#
# 라벨된 uuid 만 --only-uuids 로 골라 처리하고, 끝나면 evaluate_labels.py 로
# evaluation.log 를 남긴다. 라벨 없이 데이터를 훑는 마이닝은 이 파일이 아니라
# run_nas_nvidia_27b.sh 를 쓸 것.
#
PYBIN="/home/etri/miniconda3/envs/qwen38/bin/python3"
MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
GPUS="${GPUS:-0,1,2}"
# 로컬 pav_sample 이 대상이다. NAS 를 보려면 --data nas 로 명시한다.
DATA="${DATA:-local}"
SCENE_JSON="${SCENE_JSON:-}"
# --eval 일 때만 쓴다. run_labeled_8b.sh 와 같은 기본값 원칙 - 여기서 값을 들고
# 있으면 config.py 를 고쳐도 반영이 안 된다. 미지정이면 아래에서 실제
# 경로를 구해 셸도 uuid 목록을 뽑을 수 있게 한다.
LABELS="${LABELS:-}"

usage() {
  cat <<'USAGE'
Usage: bash run_labeled_27b.sh [options]

27B(bf16, 55.6GB)는 GPU 1장(24GB)에 안 들어가므로 GPU 여러 장에 걸쳐 단일
프로세스로 돈다(8B 처럼 8샤드 병렬이 아니다) - 그만큼 느리다. FP8 판은
멀티 GPU 에서 출력이 깨지는 게 확인되어 쓰지 않는다(스크립트 상단 주석).

Options (환경변수로도 지정 가능 - 명령행이 우선):
  --gpus 0,1,2             사용할 GPU 목록 (기본 0,1,2 - bf16 55.6GB 기준) [GPUS]
  --model ID               모델 (기본 Qwen/Qwen3.8-27B)                   [MODEL]
  --data SRC               프레임 소스 local|nas|경로 (기본 local)        [DATA]
  --scene-json PATH        카테고리 정의 (기본: config.py 의 SCENE_JSON) [SCENE_JSON]
  --no-viz                 시각화 mp4 를 만들지 않는다                   [CLIP_VIZ=0]
  --viz-all                edge-case 가 아닌 클립까지 전부 시각화        [VIZ_ALL=1]
  --viz-normal             Normal(카테고리 없음) 클립도 시각화           [VIZ_NORMAL=1]
  --viz-special            Special(카테고리 있음) 클립을 시각화          [VIZ_SPECIAL=1]
  --use-egomotion          egomotion 사실을 주입 (기본 off)              [USE_EGOMOTION=1]
  --traj center|width      자차 미래 궤적을 프레임에 그린다 (기본 off)   [TRAJ]
  --no-safety-tier         Safety Criticality(4단계)만 프롬프트에서 끈다 [SAFETY_TIER=0]
  --no-rarity-tier         Rarity(5단계)만 프롬프트에서 끈다             [RARITY_TIER=0]
  --no-score-tiers         위 둘을 한꺼번에 끄는 별칭                    [SCORE_TIERS=0]
  --difficulty             주행 난이도 5축을 0~4 로 함께 매긴다         [DIFFICULTY=1]
  --viz-per-category N     카테고리마다 최초 N개 클립만 시각화           [VIZ_PER_CAT]
  --clip-fps F              초당 몇 장 뽑을지 (기본 edge_case_mining.py) [CLIP_FPS]
  --clip-max-frames N       클립당 최대 프레임                          [CLIP_MAX_FRAMES]
  --memo "TEXT"             이 실행이 무엇을 시험하는지 한 줄 메모       [MEMO]
  --labels PATH             평가용 정답 라벨 json                       [LABELS]
                           (기본: config.py 의 LABELS_JSON)
  --eval-only DIR           추론을 건너뛰고 그 폴더의 결과만 채점한다
  -h, --help                이 도움말

Examples:
  bash run_labeled_27b.sh --difficulty           # 난이도 5축도 함께
  bash run_nas_nvidia_27b.sh --gpus 0,1,2,3           # 여유있게 4장에 나눠 올림
  bash run_labeled_27b.sh --use-egomotion        # 라벨 클립 평가
  bash run_labeled_27b.sh --eval-only results/20260824_..._video27b_eval
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --gpus=*)             GPUS="${1#*=}" ;;
    --gpus)                shift; GPUS="${1:-0,1,2}" ;;
    --model=*)             MODEL="${1#*=}" ;;
    --data=*)            DATA="${1#*=}" ;;
    --data)              shift; DATA="${1:-}" ;;
    --model)                shift; MODEL="${1:-}" ;;
    --limit-clips=*)       LIMIT_CLIPS="${1#*=}" ;;
    --limit-clips)          shift; LIMIT_CLIPS="${1:-}" ;;
    --scene-json=*)         SCENE_JSON="${1#*=}" ;;
    --scene-json)            shift; SCENE_JSON="${1:-}" ;;
    --no-viz)               CLIP_VIZ=0 ;;
    --viz-all)               VIZ_ALL=1 ;;
    --viz-normal)            VIZ_NORMAL=1 ;;
    --viz-special)           VIZ_SPECIAL=1 ;;
    --use-egomotion)         USE_EGOMOTION=1 ;;
    --no-safety-tier)        SAFETY_TIER=0 ;;
    --no-rarity-tier)        RARITY_TIER=0 ;;
    --no-score-tiers)        SAFETY_TIER=0; RARITY_TIER=0 ;;
    --difficulty)            DIFFICULTY=1 ;;
    --viz-per-category=*)  VIZ_PER_CAT="${1#*=}" ;;
    --viz-per-category)    shift; VIZ_PER_CAT="${1:-}" ;;
    --traj=*)                TRAJ="${1#*=}" ;;
    --traj)                   shift; TRAJ="${1:-center}" ;;
    --clip-fps=*)            CLIP_FPS="${1#*=}" ;;
    --clip-fps)               shift; CLIP_FPS="${1:-}" ;;
    --clip-max-frames=*)     CLIP_MAX_FRAMES="${1#*=}" ;;
    --clip-max-frames)        shift; CLIP_MAX_FRAMES="${1:-}" ;;
    --memo=*)                 MEMO="${1#*=}" ;;
    --memo)                    shift; MEMO="${1:-}" ;;
    --labels=*)                LABELS="${1#*=}" ;;
    --labels)                  shift; LABELS="${1:-}" ;;
    --eval-only=*)              EVAL_ONLY="${1#*=}" ;;
    --eval-only)                 shift; EVAL_ONLY="${1:-}" ;;
    -h|--help)               usage; exit 0 ;;
    *)
      echo "[error] unknown argument: $1" >&2
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [ -n "$SCENE_JSON" ] && [ ! -f "$SCENE_JSON" ]; then
  echo "[error] scene json not found: $SCENE_JSON" >&2
  exit 2
fi
# 이 스크립트는 평가 전용이다. 끌 수 없다 - 마이닝은 run_nas_nvidia_27b.sh.
EVAL_MODE=1

if [ ! -x "$PYBIN" ]; then
  echo "[error] qwen38 conda 환경이 없습니다: $PYBIN" >&2
  echo "        conda create -n qwen38 python=3.12 로 먼저 만드세요." >&2
  exit 2
fi
if [ -n "${LIMIT_CLIPS:-}" ]; then
  echo "[error] --limit-clips 는 이 스크립트에서 못 씁니다 - 라벨된 클립 전체가" >&2
  echo "        대상입니다(uuid 로 흩어져 있어 앞에서 N개를 자를 수 없음)." >&2
  echo "        일부만 훑어보려면 run_nas_nvidia_27b.sh --limit-clips N 을 쓰세요." >&2
  exit 2
fi

# 추론을 건너뛰고 기존 결과만 채점하는 경로 (run_labeled_8b.sh 와 동일)
if [ -n "${EVAL_ONLY:-}" ]; then
  [ -d "$EVAL_ONLY" ] || { echo "[error] no such dir: $EVAL_ONLY" >&2; exit 2; }
  LABELS="${LABELS:-$("$PYBIN" -c 'import config; print(config.LABELS_JSON)')}"
  [ -f "$LABELS" ] || { echo "[error] labels not found: $LABELS" >&2; exit 2; }
  "$PYBIN" -u evaluate_labels.py --run-dir "$EVAL_ONLY" --labels "$LABELS"
  exit $?
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="results/${RUN_TS}_labeled27b"
mkdir -p "$RUN_DIR"

OPTS="--clip-mode --single-view --num-shards 1 --shard-id 0 --model $MODEL"
[ -n "${DATA:-}" ]              && OPTS="$OPTS --data $DATA"
[ -n "$SCENE_JSON" ]            && OPTS="$OPTS --scene-json $SCENE_JSON"
[ "${USE_EGOMOTION:-0}" = "1" ] && OPTS="$OPTS --use-egomotion"
# SCORE_TIERS=0 은 옛 이름 - 둘 다 끄는 뜻으로 계속 받아준다.
[ "${SCORE_TIERS:-1}" = "0" ]   && { SAFETY_TIER=0; RARITY_TIER=0; }
[ "${SAFETY_TIER:-1}" = "0" ]   && OPTS="$OPTS --no-safety-tier"
[ "${RARITY_TIER:-1}" = "0" ]   && OPTS="$OPTS --no-rarity-tier"
[ "${DIFFICULTY:-0}" = "1" ]    && OPTS="$OPTS --difficulty"
[ -n "${VIZ_PER_CAT:-}" ]        && OPTS="$OPTS --viz-per-category $VIZ_PER_CAT"
[ -n "${TRAJ:-}" ]              && OPTS="$OPTS --traj $TRAJ"
# --viz-per-category 는 그 자체가 "시각화하라"는 뜻이다 - 로컬 스크립트는
# 시각화가 기본 off 라, 이걸 안 켜주면 아무것도 저장되지 않는다.
[ -n "${VIZ_PER_CAT:-}" ] && CLIP_VIZ=1
[ "${CLIP_VIZ:-1}" = "1" ]      && OPTS="$OPTS --clip-viz"
[ "${VIZ_ALL:-0}" = "1" ]       && OPTS="$OPTS --clip-viz-all"
[ "${VIZ_NORMAL:-0}" = "1" ]    && OPTS="$OPTS --viz-normal"
[ "${VIZ_SPECIAL:-0}" = "1" ]   && OPTS="$OPTS --viz-special"
[ -n "${CLIP_FPS:-}" ]          && OPTS="$OPTS --clip-fps $CLIP_FPS"
[ -n "${CLIP_MAX_FRAMES:-}" ]   && OPTS="$OPTS --clip-max-frames $CLIP_MAX_FRAMES"
[ -n "${LIMIT_CLIPS:-}" ]       && OPTS="$OPTS --limit-clips $LIMIT_CLIPS"

# --eval: run_labeled_8b.sh 와 같은 방식 - 라벨된 uuid 만 --only-uuids 로 골라
# 처리하고, 시각화가 켜져 있으면 GT 도 함께 넘겨 패널에 GT/Pred 를 나란히
# 그리게 한다. 라벨된 클립은 uuid 로 흩어져 있어 --limit-clips 로는 못
# 뽑는다 - 그래서 위에서 --eval 과 --limit-clips 동시 사용을 막는다.
if [ "${EVAL_MODE:-0}" = "1" ]; then
  LABELS="${LABELS:-$("$PYBIN" -c 'import config; print(config.LABELS_JSON)')}"
  [ -f "$LABELS" ] || { echo "[error] labels not found: $LABELS" >&2; exit 2; }

  UUID_FILE="${RUN_DIR}/eval_uuids.txt"
  "$PYBIN" - "$LABELS" "$UUID_FILE" <<'PY'
import json, sys
labels = json.load(open(sys.argv[1], encoding="utf-8"))
clips = labels.get("clips", labels)
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for u in clips:
        f.write(u + "\n")
print(f"[info] {len(clips)} labelled uuids -> {sys.argv[2]}")
PY
  TOTAL_CLIPS="$(wc -l < "$UUID_FILE")"
  OPTS="$OPTS --only-uuids $UUID_FILE"
  if [ "${VIZ_NORMAL:-0}" = "1" ] || [ "${VIZ_SPECIAL:-0}" = "1" ] \
     || [ "${CLIP_VIZ:-1}" = "1" ]; then
    OPTS="$OPTS --gt-labels $LABELS"
  fi
fi

{
  echo "[info] run dir     : $RUN_DIR"
  echo "[info] model       : $MODEL"
  echo "[info] gpus        : $GPUS (단일 프로세스, device_map=auto)"
  echo "[info] python      : $PYBIN (conda env: qwen38)"
  [ "${EVAL_MODE:-0}" = "1" ] && echo "[info] labels      : $LABELS  ($TOTAL_CLIPS clips)"
  echo "[info] opts        : $OPTS"
  [ -n "${MEMO:-}" ] && echo "[info] memo        : $MEMO"
  echo

  CUDA_VISIBLE_DEVICES="$GPUS" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYBIN" -u edge_case_mining.py \
      $OPTS \
      --memo "${MEMO:-}" \
      --out "${RUN_DIR}/clip_results_shard_0.csv" \
      --viz-dir "${RUN_DIR}/viz" \
      2>&1 | tee "${RUN_DIR}/run_shard_0.log"

  echo "[info] run done."
  "$PYBIN" -u merge_shards.py --run-dir "$RUN_DIR" 2>&1 || \
    cp "${RUN_DIR}/clip_results_shard_0.csv" "${RUN_DIR}/clip_results_all.csv"

  if [ "${EVAL_MODE:-0}" = "1" ]; then
    "$PYBIN" -u evaluate_labels.py --run-dir "$RUN_DIR" --labels "$LABELS"
  else
    "$PYBIN" -u aggregate_clip.py --run-dir "$RUN_DIR" 2>&1
  fi

  echo "[info] results saved in: ${RUN_DIR}/"
} 2>&1 | tee "${RUN_DIR}/run.log"

#!/usr/bin/env bash
# 클립 단위 edge-case mining - Qwen3.8-27B(bf16), 단일 프로세스.
#
# run_video_C.sh 와 무엇이 다른가:
#   run_video_C.sh   8B. GPU 1장에 모델 전체(~18.5GB)가 들어가므로 8개
#                    프로세스를 GPU 0~7 에 하나씩 배정해 병렬로 돈다.
#   run_nas_nvidia_27b.sh 27B 는 bf16 55.6GB 라 GPU 1장(24GB)에 안 들어간다.
#                    device_map="auto" 로 한 프로세스가 GPU 여러 장에 층을
#                    나눠 올려야 하므로, 8샤드 병렬 구조를 못 쓰고 프로세스
#                    하나가 지정된 GPU 를 전부 물고 순차로 처리한다.
#
# 왜 bf16 이고 FP8 이 아닌가:
#   FP8 판(Qwen/Qwen3.8-27B-FP8, 30.9GB)이면 GPU 2장으로 될 것 같지만,
#   실측(20260824)에서 멀티 GPU 로 FP8 을 로드하면 출력이 깨졌다 - 이미지와
#   무관하게 "2+2는?" 같은 순수 텍스트 질문도 의미 없는 토큰을 반복하다
#   무한 루프에 빠졌다(비전 문제가 아님을 텍스트 전용 생성으로 격리 확인).
#   로드 시 "DeepGEMM 대신 Triton/grouped_mm 경로로 우회한다"는 경고가
#   뜨는데, 그 경로가 실제로 손상된 값을 낸다. bf16 원본은 같은 멀티 GPU
#   배치에서 정상 응답했으므로 FP8 양자화 자체가 원인이다. GPU 1장에 FP8
#   을 통째로 올릴 방법이 없는 한(24GB < 30.9GB) FP8 은 못 쓴다.
#
# 실행 환경: base 가 아니라 별도 conda 환경 qwen38 을 쓴다. 27B 계열
# (architectures=Qwen3_5ForConditionalGeneration)이 요구하는 최신
# transformers/torch 를 base 에 직접 설치하면, 의존성이 이 서버 드라이버
# (12.8)가 지원하지 않는 CUDA 13 계열을 끌어와 GPU 8장이 전부 CUDA 를
# 못 쓰는 상태까지 간 적이 있다(compressed-tensors 설치 중 실제 발생).
# qwen38 환경은 torch==2.10.0+cu128 로 고정해서 base 와 완전히 분리해 둔다.
#
# 결과: results/<YYYYMMDD_HHMMSS>_video27b/  (run_video_C.sh 와 같은 형식)
#   --eval 을 주면 results/<YYYYMMDD_HHMMSS>_video27b_eval/ 로, run_labeled_8b.sh 와
#   같은 방식(라벨 있는 클립만 --only-uuids 로 골라 처리 + evaluate_labels.py)
#   으로 evaluation.log 까지 남긴다.
set -u
cd /home/etri/Jeongbin/Nvidia-VLM

PYBIN="/home/etri/miniconda3/envs/qwen38/bin/python3"
MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
GPUS="${GPUS:-0,1,2}"
# NAS 청크 zip 이 이 스크립트의 대상이다 - 기본을 nas 로 둔다.
DATA="${DATA:-nas}"
SCENE_JSON="${SCENE_JSON:-}"
# --eval 일 때만 쓴다. run_labeled_8b.sh 와 같은 기본값 원칙 - 여기서 값을 들고
# 있으면 config.py 를 고쳐도 반영이 안 된다. 미지정이면 아래에서 실제
# 경로를 구해 셸도 uuid 목록을 뽑을 수 있게 한다.
LABELS="${LABELS:-}"

usage() {
  cat <<'USAGE'
Usage: bash run_nas_nvidia_27b.sh [options]

27B(bf16, 55.6GB)는 GPU 1장(24GB)에 안 들어가므로 GPU 여러 장에 걸쳐 단일
프로세스로 돈다(8B 처럼 8샤드 병렬이 아니다) - 그만큼 느리다. FP8 판은
멀티 GPU 에서 출력이 깨지는 게 확인되어 쓰지 않는다(스크립트 상단 주석).

Options (환경변수로도 지정 가능 - 명령행이 우선):
  --gpus 0,1,2             사용할 GPU 목록 (기본 0,1,2 - bf16 55.6GB 기준) [GPUS]
  --model ID               모델 (기본 Qwen/Qwen3.8-27B)                   [MODEL]
  --data SRC               프레임 소스 local|nas|경로 (기본 nas)          [DATA]
  --limit-clips N          처리할 클립 수 제한 (기본: 데이터셋 전체)     [LIMIT_CLIPS]
  --only-uuids FILE        이 파일에 적힌 uuid 만 처리 (한 줄에 하나)    [ONLY_UUIDS]
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
  --difficulty             주행 난이도 4축을 0~4 로 함께 매긴다         [DIFFICULTY=1]
  --difficulty-only        난이도 4축만 추론한다 (탐지/등급 전부 off)  [DIFFICULTY_ONLY=1]
  --viz-per-category N     카테고리마다 최초 N개 클립만 시각화           [VIZ_PER_CAT]
  --clip-fps F              초당 몇 장 뽑을지 (기본 edge_case_mining.py) [CLIP_FPS]
  --clip-max-frames N       클립당 최대 프레임                          [CLIP_MAX_FRAMES]
  --memo "TEXT"             이 실행이 무엇을 시험하는지 한 줄 메모       [MEMO]
  --eval                   라벨 있는 클립만 처리하고 채점한다           [EVAL_MODE=1]
                           (평가가 주 목적이면 run_labeled_27b.sh 를 쓸 것)
  --no-eval                (기본) 라벨 없이 데이터셋 전체를 훑는다      [EVAL_MODE=0]
  --labels PATH             평가용 정답 라벨 json                       [LABELS]
                           (기본: config.py 의 LABELS_JSON)
  --eval-only DIR           추론을 건너뛰고 그 폴더의 결과만 채점한다
  -h, --help                이 도움말

Examples:
  bash run_nas_nvidia_27b.sh --limit-clips 20         # 마이닝 시험(eval 자동 off)
  bash run_nas_nvidia_27b.sh --gpus 0,1,2,3           # 여유있게 4장에 나눠 올림
  bash run_nas_nvidia_27b.sh --use-egomotion          # 라벨 클립 평가(기본)
  bash run_nas_nvidia_27b.sh --eval-only results/20260824_..._video27b_eval
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
    --only-uuids=*)        ONLY_UUIDS="${1#*=}" ;;
    --only-uuids)          shift; ONLY_UUIDS="${1:-}" ;;
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
    --difficulty-only)       DIFFICULTY_ONLY=1 ;;
    --no-tiers-elements)     TIERS_ELEMENTS=1 ;;
    --explain-traj)          EXPLAIN_TRAJ=1 ;;
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
    --eval)                   EVAL_MODE=1 ;;
    --no-eval)                EVAL_MODE=0 ;;
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
# 마이닝이 기본이다. 라벨 대조 평가는 run_labeled_27b.sh 가 맡는다.
EVAL_MODE="${EVAL_MODE:-0}"

if [ ! -x "$PYBIN" ]; then
  echo "[error] qwen38 conda 환경이 없습니다: $PYBIN" >&2
  echo "        conda create -n qwen38 python=3.12 로 먼저 만드세요." >&2
  exit 2
fi
if [ "${EVAL_MODE:-0}" = "1" ] && [ -n "${LIMIT_CLIPS:-}" ]; then
  echo "[error] --eval 과 --limit-clips 는 함께 쓸 수 없습니다 (라벨 클립 전체가 대상)" >&2
  exit 2
fi
# --eval 은 라벨된 uuid 로 자기 --only-uuids 를 만든다. 둘 다 주면 같은 플래그가
# 두 번 붙어 뒤엣것만 먹는다 - 조용히 엉뚱한 대상을 도는 대신 여기서 막는다.
if [ "${EVAL_MODE:-0}" = "1" ] && [ -n "${ONLY_UUIDS:-}" ]; then
  echo "[error] --eval 과 --only-uuids 는 함께 쓸 수 없습니다" >&2
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
if [ "${EVAL_MODE:-0}" = "1" ]; then
  RUN_DIR="results/labeld/${RUN_TS}_video27b_eval"
else
  RUN_DIR="results/unlabeled/${RUN_TS}_video27b"
fi
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
[ "${DIFFICULTY_ONLY:-0}" = "1" ] && OPTS="$OPTS --difficulty-only"
[ "${TIERS_ELEMENTS:-0}" = "1" ] && OPTS="$OPTS --no-tiers-elements"
[ "${EXPLAIN_TRAJ:-0}" = "1" ] && OPTS="$OPTS --explain-traj"
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
# 특정 uuid 만 돌린다(GUI 단일 클립 조회). 전체 인덱스를 안 만들고 그 클립이
# 든 zip 만 찾아 열므로, NAS 에서도 몇 초~30초 안에 시작한다.
# --eval 은 자기 UUID_FILE 을 따로 만들어 쓰므로 아래에서 충돌을 막는다.
[ -n "${ONLY_UUIDS:-}" ]        && OPTS="$OPTS --only-uuids $ONLY_UUIDS"

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

#!/usr/bin/env bash
# test_label.json 의 100개 클립만 추론하고, 정답 라벨과 대조해 성능을 잰다.
#
# run_video_C.sh 와 다른 점:
#   run_video_C.sh  데이터셋 전체(또는 앞에서 N개)를 처리한다.
#   run_eval.sh     라벨이 있는 클립만 골라 처리하고 evaluate_labels.py 까지 돌린다.
#
# 라벨된 클립은 uuid 로 흩어져 있어 --limit-clips 로는 뽑을 수 없다. 그래서
# uuid 목록을 --only-uuids 파일로 넘겨 그 클립만 처리하게 한다.
#
# 결과: results/<YYYYMMDD_HHMMSS>_eval/
#   clip_results_shard_N.csv   샤드별 추론 결과
#   evaluation.log             4개 축 성능 리포트
#   run.log                    전체 로그
set -u
cd /home/etri/Jeongbin/Nvidia-VLM

NSHARDS=8
# 카테고리 정의와 정답 라벨의 기본값은 config.py 가 단일 진실 공급원이다.
# 여기서 기본값을 들고 있으면 안 된다 - 예전에는 이 스크립트가 자기 값을
# 항상 명령행으로 넘겨서, config/argparse 쪽 기본값을 아무리 고쳐도
# 반영되지 않았다. 미지정이면 아래에서 플래그 자체를 생략한다.
SCENE_JSON="${SCENE_JSON:-}"
# --labels 만은 uuid 목록을 뽑아야 해서 셸도 실제 경로를 알아야 한다.
LABELS="${LABELS:-$(python3 -c 'import config; print(config.LABELS_JSON)')}"

usage() {
  cat <<'USAGE'
Usage: bash run_eval.sh [options]

Options (환경변수로도 지정 가능 - 명령행이 우선):
  --labels PATH          정답 라벨 json (기본: config.py 의 LABELS_JSON)  [LABELS]
  --scene-json PATH      카테고리 정의 (기본: config.py 의 SCENE_JSON)    [SCENE_JSON]
  --num-shards N         GPU/shard 개수 (기본 8)                     [NSHARDS]
  --viz                  시각화 mp4 도 만든다 (기본 안 만듦)          [CLIP_VIZ=1]
  --viz-normal           Normal 클립을 시각화 -> viz/normal/score_N/    [VIZ_NORMAL=1]
  --viz-special          Special 클립을 시각화 -> viz/special/score_N/  [VIZ_SPECIAL=1]
  --memo "TEXT"          이 실행이 무엇을 시험하는지 한 줄 메모.
                         evaluation.log 머리에 [info] MEMO 로 찍힌다      [MEMO]
  --use-egomotion        egomotion 사실(자차 행동 요약)을 주입 (기본 off) [USE_EGOMOTION=1]
  --use-3dbbox           obstacle.offline 3D bbox 라벨을 프롬프트에 주입
                         (기본 off, Animal/Jaywalking/cyclist 과탐 경향 실측됨)  [USE_3DBBOX=1]
  --no-video-input       프레임을 비디오가 아니라 낱장으로 넘긴다     [VIDEO_INPUT=0]
  --clip-fps F           초당 몇 장 뽑을지 (기본 1.0)                 [CLIP_FPS]
  --clip-max-frames N    클립당 최대 프레임 (기본 20)                 [CLIP_MAX_FRAMES]
  --eval-only DIR        추론을 건너뛰고 그 폴더의 결과만 채점한다
  -h, --help             이 도움말

Examples:
  bash run_eval.sh                                  # 추론 + 채점
  bash run_eval.sh --eval-only results/20260812_135056_videoC
  bash run_eval.sh --viz-normal --viz-special   # 둘 다 시각화
  bash run_eval.sh --viz-special                # special 만
  bash run_eval.sh --use-3dbbox                 # 3D bbox 라벨도 프롬프트에 주입
  bash run_eval.sh --clip-fps 2.0 --clip-max-frames 40
USAGE
}

EVAL_ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --labels=*)          LABELS="${1#*=}" ;;
    --labels)            shift; LABELS="${1:-}" ;;
    --scene-json=*)      SCENE_JSON="${1#*=}" ;;
    --scene-json)        shift; SCENE_JSON="${1:-}" ;;
    --num-shards=*)      NSHARDS="${1#*=}" ;;
    --num-shards)        shift; NSHARDS="${1:-8}" ;;
    --viz)               CLIP_VIZ=1 ;;
    --viz-normal)        VIZ_NORMAL=1 ;;
    --viz-special)       VIZ_SPECIAL=1 ;;
    --memo=*)            MEMO="${1#*=}" ;;
    --memo)              shift; MEMO="${1:-}" ;;
    --use-egomotion)     USE_EGOMOTION=1 ;;
    --no-egomotion)      USE_EGOMOTION=0 ;;   # 옛 이름 - 이제 기본이 off 라 무의미하지만 받아준다
    --use-3dbbox)        USE_3DBBOX=1 ;;
    --no-video-input)    VIDEO_INPUT=0 ;;
    --clip-fps=*)        CLIP_FPS="${1#*=}" ;;
    --clip-fps)          shift; CLIP_FPS="${1:-}" ;;
    --clip-max-frames=*) CLIP_MAX_FRAMES="${1#*=}" ;;
    --clip-max-frames)   shift; CLIP_MAX_FRAMES="${1:-}" ;;
    --eval-only=*)       EVAL_ONLY="${1#*=}" ;;
    --eval-only)         shift; EVAL_ONLY="${1:-}" ;;
    -h|--help)           usage; exit 0 ;;
    *)
      echo "[error] unknown argument: $1" >&2
      echo >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[ -f "$LABELS" ] || { echo "[error] labels not found: $LABELS" >&2; exit 2; }

# 추론을 건너뛰고 기존 결과만 채점하는 경로
if [ -n "$EVAL_ONLY" ]; then
  [ -d "$EVAL_ONLY" ] || { echo "[error] no such dir: $EVAL_ONLY" >&2; exit 2; }
  python3 -u evaluate_labels.py --run-dir "$EVAL_ONLY" --labels "$LABELS"
  exit $?
fi

if [ -n "$SCENE_JSON" ] && [ ! -f "$SCENE_JSON" ]; then
  echo "[error] scene json not found: $SCENE_JSON" >&2; exit 2
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="results/${RUN_TS}_eval"
mkdir -p "$RUN_DIR"

# 라벨된 uuid 만 뽑아 파일로 넘긴다 (한 줄에 하나)
UUID_FILE="${RUN_DIR}/eval_uuids.txt"
python3 - "$LABELS" "$UUID_FILE" <<'PY'
import json, sys
labels = json.load(open(sys.argv[1], encoding="utf-8"))
clips = labels.get("clips", labels)
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for u in clips:
        f.write(u + "\n")
print(f"[info] {len(clips)} labelled uuids -> {sys.argv[2]}")
PY

TOTAL_CLIPS="$(wc -l < "$UUID_FILE")"

OPTS="--clip-mode --single-view --only-uuids $UUID_FILE"
# 명시했을 때만 넘긴다 - 안 넘기면 config.py 기본값이 실제로 쓰인다
[ -n "$SCENE_JSON" ] && OPTS="$OPTS --scene-json $SCENE_JSON"
[ "${USE_EGOMOTION:-0}" = "1" ] && OPTS="$OPTS --use-egomotion"
[ "${USE_3DBBOX:-0}" = "1" ]    && OPTS="$OPTS --use-3dbbox"
[ "${VIDEO_INPUT:-1}" = "0" ]   && OPTS="$OPTS --clip-no-video-input"
[ "${CLIP_VIZ:-0}" = "1" ]      && OPTS="$OPTS --clip-viz"
# --viz-normal/--viz-special 은 각각 독립이고, 하나라도 주면 시각화가 켜진다.
# 이 스크립트는 정답 라벨이 있는 클립만 돌리므로, 시각화할 때는 GT 를 항상
# 함께 넘겨 패널에 GT/Pred 를 나란히 그리게 한다.
[ "${VIZ_NORMAL:-0}" = "1" ]    && OPTS="$OPTS --viz-normal"
[ "${VIZ_SPECIAL:-0}" = "1" ]   && OPTS="$OPTS --viz-special"
if [ "${VIZ_NORMAL:-0}" = "1" ] || [ "${VIZ_SPECIAL:-0}" = "1" ] \
   || [ "${CLIP_VIZ:-0}" = "1" ]; then
  OPTS="$OPTS --gt-labels $LABELS"
fi
[ -n "${CLIP_FPS:-}" ]          && OPTS="$OPTS --clip-fps $CLIP_FPS"
[ -n "${CLIP_MAX_FRAMES:-}" ]   && OPTS="$OPTS --clip-max-frames $CLIP_MAX_FRAMES"

{
  echo "[info] run dir     : $RUN_DIR"
  echo "[info] labels      : $LABELS  ($TOTAL_CLIPS clips)"
  echo "[info] categories  : ${SCENE_JSON:-$(python3 -c 'import config; print(config.SCENE_JSON.name)') (config.py)}"
  echo "[info] opts        : $OPTS"
  [ -n "${MEMO:-}" ] && echo "[info] memo        : $MEMO"
  echo

  pids=()
  for g in $(seq 0 $((NSHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      nohup python3 -u edge_case_mining.py \
        --num-shards "$NSHARDS" --shard-id "$g" \
        $OPTS \
        --memo "${MEMO:-}" \
        --out "${RUN_DIR}/clip_results_shard_${g}.csv" \
        --viz-dir "${RUN_DIR}/viz" \
        > "${RUN_DIR}/run_shard_${g}.log" 2>&1 &
    pids+=($!)
  done
  echo "[info] launched ${#pids[@]} shards, PIDs: ${pids[*]}"

  while :; do
    alive=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
    done_n=0
    for g in $(seq 0 $((NSHARDS-1))); do
      f="${RUN_DIR}/clip_results_shard_${g}.csv"
      [ -f "$f" ] && done_n=$((done_n + $(($(wc -l < "$f") - 1))))
    done
    [ "$done_n" -lt 0 ] && done_n=0
    printf "\r[progress] %d/%d clips  (%d shard(s) running)   " \
      "$done_n" "$TOTAL_CLIPS" "$alive"
    [ "$alive" -eq 0 ] && break
    sleep 15
  done
  echo

  wait
  echo "[info] all shards done."
  echo

  # 샤드 CSV 를 clip_results_all.csv 로 합치고 원본은 지운다
  python3 -u merge_shards.py --run-dir "$RUN_DIR"
  echo

  python3 -u evaluate_labels.py --run-dir "$RUN_DIR" --labels "$LABELS"

  echo
  echo "[info] results saved in: ${RUN_DIR}/"
} 2>&1 | tee "${RUN_DIR}/run.log"

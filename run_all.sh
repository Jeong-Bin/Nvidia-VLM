#!/usr/bin/env bash
# 전체 판정 단위(클립 x 10 timestamp, 전방 3뷰 동기화)를
# 8개 shard로 나눠 GPU 0~7 에 병렬 배정.
# 각 프로세스는 자기 GPU 1장에 VLM 전체를 로드한다.
#
# 옵션은 명령행 플래그와 환경변수 둘 다 받는다 (명령행 우선):
#   bash run_all.sh --use-egomotion --use-obstacle --check-path
#   USE_EGOMOTION=1 USE_OBSTACLE=1 bash run_all.sh
# 모르는 인자는 조용히 무시하지 않고 즉시 에러를 낸다 - 예전에 명령행 플래그를
# 안 받던 시절 `bash run_all.sh --use-obstacle=1` 이 통째로 무시된 채 2시간
# 돌아버린 적이 있어서, 설정이 실제로 먹었는지 아래 [info] 줄로 확인할 것.
#
# 결과는 results/<YYYYMMDD_HHMMSS>/ 아래에 실행 1회 단위로 모아 저장한다.
# 각 shard 는 여전히 독립적으로 자기 CSV/로그(results_shard_N.csv, run_shard_N.log)에
# 쓰지만(안전, 락 불필요), 진행 상황은 monitor_progress.py 가 그 파일들을 모아
# 하나의 통합 프로그레스 바로 보여주고, 전체 완료 후 결과를 하나의 run.log 와
# edge_case_results_all.csv 로 병합한다.
set -u
cd /home/etri/Jeongbin/Nvidia-VLM

NSHARDS=8
TIMESTAMPS_PER_CLIP=10

usage() {
  cat <<'USAGE'
Usage: bash run_all.sh [options]

Options (환경변수로도 지정 가능 - 명령행이 우선):
  --use-egomotion[=0|1]     egomotion 속도/가속도를 프롬프트에 사실로 주입   [USE_EGOMOTION]
  --use-obstacle[=0|1]      obstacle 3D 라벨의 주변 객체 요약을 주입         [USE_OBSTACLE]
  --check-path[=0|1]        3D bbox 로 Q3 를 기하 검증 (프롬프트엔 안 들어감) [CHECK_PATH]
  --no-blocking             Q3(경로 차단) 질문 자체를 끈다                    [ASK_BLOCKING=0]
  --limit-clips N           처리할 클립 수 제한 (기본: 데이터셋 전체)         [LIMIT_CLIPS]
  --example-source S        synonyms | prompt_templates                       [EXAMPLE_SOURCE]
  --num-examples N          카테고리당 예시 개수                              [NUM_EXAMPLES]
  -h, --help                이 도움말

Examples:
  bash run_all.sh --use-egomotion --use-obstacle --check-path
  USE_EGOMOTION=1 USE_OBSTACLE=1 bash run_all.sh
  bash run_all.sh --limit-clips 500 --no-blocking
USAGE
}

# 명령행 인자 -> 환경변수와 같은 변수에 채운다. 명령행이 환경변수를 덮어쓴다.
# 불린 플래그는 `--flag` 와 `--flag=1` / `--flag=0` 을 모두 받는다.
_bool_val() {  # $1=인자 전체, $2=기본값(값 없이 왔을 때)
  case "$1" in
    *=*) printf '%s' "${1#*=}" ;;
    *)   printf '%s' "$2" ;;
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    --use-egomotion|--use-egomotion=*) USE_EGOMOTION="$(_bool_val "$1" 1)" ;;
    --use-obstacle|--use-obstacle=*)   USE_OBSTACLE="$(_bool_val "$1" 1)" ;;
    --check-path|--check-path=*)       CHECK_PATH="$(_bool_val "$1" 1)" ;;
    --no-blocking|--no-blocking=*)     ASK_BLOCKING=0 ;;
    --limit-clips=*)     LIMIT_CLIPS="${1#*=}" ;;
    --limit-clips)       shift; LIMIT_CLIPS="${1:-}" ;;
    --example-source=*)  EXAMPLE_SOURCE="${1#*=}" ;;
    --example-source)    shift; EXAMPLE_SOURCE="${1:-}" ;;
    --num-examples=*)    NUM_EXAMPLES="${1#*=}" ;;
    --num-examples)      shift; NUM_EXAMPLES="${1:-}" ;;
    -h|--help)           usage; exit 0 ;;
    *)
      echo "[error] unknown argument: $1" >&2
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

# 처리할 클립 수. 미지정이면 현재 데이터셋 전체(front_wide 뷰의 mp4 개수).
TOTAL_CLIPS="${LIMIT_CLIPS:-$(ls pav_sample/camera/camera_front_wide_120fov/*.mp4 2>/dev/null | wc -l)}"
TOTAL_UNITS=$((TOTAL_CLIPS * TIMESTAMPS_PER_CLIP))

LIMIT_OPTS=""
[ -n "${LIMIT_CLIPS:-}" ] && LIMIT_OPTS="--limit-clips $LIMIT_CLIPS"

# 센서 라벨을 프롬프트에 사실로 넣을지 (둘 다 기본 off)
SENSOR_OPTS=""
[ "${USE_EGOMOTION:-0}" = "1" ] && SENSOR_OPTS="$SENSOR_OPTS --use-egomotion"
[ "${USE_OBSTACLE:-0}" = "1" ] && SENSOR_OPTS="$SENSOR_OPTS --use-obstacle"
# 3D bbox 로 Q3 를 기하 검증 (프롬프트에는 안 들어감)
[ "${CHECK_PATH:-0}" = "1" ] && SENSOR_OPTS="$SENSOR_OPTS --check-path"
# Q3(주행 경로 차단 여부) 질문 자체를 끄면 시각화도 blocking 으로 안 나눈다
[ "${ASK_BLOCKING:-1}" = "0" ] && SENSOR_OPTS="$SENSOR_OPTS --no-blocking"
# 캡션 힌트 설정은 지정했을 때만 넘긴다 - 기본값은 edge_case_mining.py 가 단일 진실 공급원
CAPTION_OPTS=""
[ -n "${EXAMPLE_SOURCE:-}" ] && CAPTION_OPTS="$CAPTION_OPTS --example-source $EXAMPLE_SOURCE"
[ -n "${NUM_EXAMPLES:-}" ] && CAPTION_OPTS="$CAPTION_OPTS --num-examples $NUM_EXAMPLES"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="results/${RUN_TS}"
mkdir -p "$RUN_DIR"

{
  echo "[info] run dir: $RUN_DIR"
  echo "[info] total units: $TOTAL_UNITS ($TOTAL_CLIPS clips x $TIMESTAMPS_PER_CLIP timestamps), $NSHARDS shards"
  echo "[info] caption opts: ${CAPTION_OPTS:-<edge_case_mining.py defaults: synonyms x1>}"
  echo "[info] sensor facts: egomotion=${USE_EGOMOTION:-0} obstacle=${USE_OBSTACLE:-0} (1=on, 0=off)"
  echo "[info] 3D path check: ${CHECK_PATH:-0} (1=on, 0=off)"
  echo "[info] Q3 blocking question: ${ASK_BLOCKING:-1} (1=on, 0=off)"

  pids=()
  for g in $(seq 0 $((NSHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      nohup python3 -u edge_case_mining.py \
        --num-shards $NSHARDS --shard-id $g \
        --timestamps-per-clip $TIMESTAMPS_PER_CLIP \
        $LIMIT_OPTS $SENSOR_OPTS $CAPTION_OPTS \
        --out "${RUN_DIR}/results_shard_${g}.csv" \
        --viz-dir "${RUN_DIR}" \
        > "${RUN_DIR}/run_shard_${g}.log" 2>&1 &
    pids+=($!)
  done
  echo "[info] launched shards, PIDs: ${pids[*]}"

  # 통합 진행률 표시 (모든 shard 가 종료될 때까지 대기)
  python3 -u monitor_progress.py \
    --run-dir "$RUN_DIR" --num-shards $NSHARDS --total-units $TOTAL_UNITS \
    --pids "${pids[@]}"

  wait
  echo "[info] all shards done."

  # CSV 병합 + 최종 카테고리 집계를 같은 통합 로그에 기록
  python3 -u aggregate.py --run-dir "$RUN_DIR"

  echo "[info] results saved in: ${RUN_DIR}/"
} 2>&1 | tee "${RUN_DIR}/run.log"

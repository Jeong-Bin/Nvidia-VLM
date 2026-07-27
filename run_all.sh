#!/usr/bin/env bash
# 5,000개 판정 단위(500클립 x 10 timestamp, 전방 3뷰 동기화)를
# 8개 shard로 나눠 GPU 0~7 에 병렬 배정.
# 각 프로세스는 자기 GPU 1장에 Qwen2.5-VL 전체를 로드한다.
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
TOTAL_CLIPS=500
TOTAL_UNITS=$((TOTAL_CLIPS * TIMESTAMPS_PER_CLIP))
# 1단계 캡션 힌트 설정은 edge_case_mining.py 의 기본값
# (synonyms, 카테고리당 1개 = 20260723 방식)을 그대로 따른다 - 단일 진실 공급원.
# 실험적으로 바꾸고 싶을 때만 환경변수로 오버라이드:
#   CAPTION_EXAMPLE_SOURCE=prompt_templates CAPTION_NUM_EXAMPLES=3 bash run_all.sh
CAPTION_OPTS=""
[ -n "${CAPTION_EXAMPLE_SOURCE:-}" ] && CAPTION_OPTS="$CAPTION_OPTS --caption-example-source $CAPTION_EXAMPLE_SOURCE"
[ -n "${CAPTION_NUM_EXAMPLES:-}" ] && CAPTION_OPTS="$CAPTION_OPTS --caption-num-examples $CAPTION_NUM_EXAMPLES"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="results/${RUN_TS}"
mkdir -p "$RUN_DIR"

{
  echo "[info] run dir: $RUN_DIR"
  echo "[info] total units: $TOTAL_UNITS ($TOTAL_CLIPS clips x $TIMESTAMPS_PER_CLIP timestamps), $NSHARDS shards"
  echo "[info] caption opts: ${CAPTION_OPTS:-<edge_case_mining.py defaults: synonyms x1>}"

  pids=()
  for g in $(seq 0 $((NSHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      nohup python3 -u edge_case_mining.py \
        --num-shards $NSHARDS --shard-id $g \
        --timestamps-per-clip $TIMESTAMPS_PER_CLIP \
        $CAPTION_OPTS \
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

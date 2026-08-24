#!/usr/bin/env bash
# 클립 단위 edge-case mining - 비디오 입력 + 전방 1뷰 + scene_category_C.json.
#
# run_all.sh 와 무엇이 다른가:
#   run_all.sh      판정 단위 = (uuid, frame_idx). 클립당 10 timestamp x 2프레임
#                   x 3뷰. Q1/Q2/Q3 프롬프트.
#   run_video_C.sh  판정 단위 = 클립 하나. 20초를 1fps 20장으로 훑어 비디오
#                   한 편으로 넣는다. nuReasoning 단계별 프롬프트.
#
# 클립 1,998개를 8개 shard 로 나눠 GPU 0~7 에 배정한다. 각 프로세스가 자기
# GPU 1장에 모델 전체를 올린다(실측 약 18.5GB / 24GB).
#
# 결과: results/<YYYYMMDD_HHMMSS>_videoC/
#   clip_results_shard_N.csv   샤드별 결과
#   clip_results_all.csv       병합본 (aggregate_clip.py 가 생성)
#   aggregate_clip.log         카테고리 집계
#   viz/<uuid>/{clip.mp4,result.json}   edge-case 클립 시각화
#   run_shard_N.log            샤드별 원본 로그
set -u
cd /home/etri/Jeongbin/Nvidia-VLM

NSHARDS=8
# 기본값은 config.py 가 단일 진실 공급원 - 미지정이면 플래그를 생략한다
SCENE_JSON="${SCENE_JSON:-}"

usage() {
  cat <<'USAGE'
Usage: bash run_video_C.sh [options]

Options (환경변수로도 지정 가능 - 명령행이 우선):
  --limit-clips N        처리할 클립 수 제한 (기본: 데이터셋 전체)     [LIMIT_CLIPS]
  --num-shards N         GPU/shard 개수 (기본 8)                       [NSHARDS]
  --scene-json PATH      카테고리 정의 (기본: config.py 의 SCENE_JSON)  [SCENE_JSON]
  --no-viz               시각화 mp4 를 만들지 않는다                    [CLIP_VIZ=0]
  --viz-all              edge-case 가 아닌 클립까지 전부 시각화         [VIZ_ALL=1]
  --viz-width N          시각화 영상 폭 (기본 1280, 0=원본)            [VIZ_WIDTH]
  --save-low             Safety/Rarity 가 둘 다 Low 인 클립도 시각화     [NOT_SAVE_LOW=0]
  --timeline             1단계를 시간순 서술로 (기본 off)            [TIMELINE=1]
  --ego-track            egomotion 을 1초 간격 시계열로도 제공        [EGO_TRACK=1]
  --traj center|width    자차 미래 궤적을 프레임에 그린다 (기본 off)   [TRAJ]
  --memo "TEXT"          이 실행이 무엇을 시험하는지 한 줄 메모           [MEMO]
  --use-egomotion        egomotion 사실(자차 행동 요약)을 주입 (기본 off)  [USE_EGOMOTION=1]
  --use-egomotion-c      [대조군C] 헤더/hint 유지, 센서 수치만 제거          [EGO_ABLATION=c]
  --use-egomotion-d      [대조군D] hint 만 남기고 헤더/수치 제거             [EGO_ABLATION=d]
  --header-style v1|v2   자차 행동 블록 헤더 문구 (기본 v1)              [HEADER_STYLE]
  --no-score-tiers        Safety/Rarity(4/5단계)를 프롬프트에서 통째로 끈다 [SCORE_TIERS=0]
  --no-video-input       프레임을 비디오가 아니라 낱장으로 넘긴다       [VIDEO_INPUT=0]
  --clip-fps F           초당 몇 장 뽑을지 (기본 1.0)                   [CLIP_FPS]
  --clip-max-frames N    클립당 최대 프레임 (기본 20)                   [CLIP_MAX_FRAMES]
  -h, --help             이 도움말

Examples:
  bash run_video_C.sh                       # 전체, 기본 설정
  bash run_video_C.sh --limit-clips 50      # 시험
  bash run_video_C.sh --no-viz              # CSV 만
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --limit-clips=*)     LIMIT_CLIPS="${1#*=}" ;;
    --limit-clips)       shift; LIMIT_CLIPS="${1:-}" ;;
    --num-shards=*)      NSHARDS="${1#*=}" ;;
    --num-shards)        shift; NSHARDS="${1:-8}" ;;
    --scene-json=*)      SCENE_JSON="${1#*=}" ;;
    --scene-json)        shift; SCENE_JSON="${1:-}" ;;
    --no-viz)            CLIP_VIZ=0 ;;
    --viz-all)           VIZ_ALL=1 ;;
    --viz-width=*)       VIZ_WIDTH="${1#*=}" ;;
    --viz-width)         shift; VIZ_WIDTH="${1:-}" ;;
    --save-low)          NOT_SAVE_LOW=0 ;;
    --timeline)          TIMELINE=1 ;;
    --ego-track)         EGO_TRACK=1 ;;
    --traj=*)            TRAJ="${1#*=}" ;;
    --traj)              shift; TRAJ="${1:-center}" ;;
    --memo=*)            MEMO="${1#*=}" ;;
    --memo)              shift; MEMO="${1:-}" ;;
    --use-egomotion)     USE_EGOMOTION=1 ;;
    --use-egomotion-c)   EGO_ABLATION=c ;;
    --use-egomotion-d)   EGO_ABLATION=d ;;
    --header-style=*)    HEADER_STYLE="${1#*=}" ;;
    --header-style)      shift; HEADER_STYLE="${1:-v1}" ;;
    --no-score-tiers)    SCORE_TIERS=0 ;;
    --no-egomotion)      USE_EGOMOTION=0 ;;   # 옛 이름 - 이제 기본이 off 라 무의미하지만 받아준다
    --no-video-input)    VIDEO_INPUT=0 ;;
    --clip-fps=*)        CLIP_FPS="${1#*=}" ;;
    --clip-fps)          shift; CLIP_FPS="${1:-}" ;;
    --clip-max-frames=*) CLIP_MAX_FRAMES="${1#*=}" ;;
    --clip-max-frames)   shift; CLIP_MAX_FRAMES="${1:-}" ;;
    -h|--help)           usage; exit 0 ;;
    *)
      # 모르는 인자를 조용히 무시하면 설정이 안 먹은 채로 몇 시간 돌아간다.
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

TOTAL_CLIPS="${LIMIT_CLIPS:-$(ls pav_sample/camera/camera_front_wide_120fov/*.mp4 2>/dev/null | wc -l)}"
if [ "$TOTAL_CLIPS" -eq 0 ]; then
  echo "[error] no clips found under pav_sample/camera/camera_front_wide_120fov/" >&2
  exit 2
fi

# 기본값은 edge_case_mining.py 를 단일 진실 공급원으로 두고, 여기서는
# 지정했을 때만 넘긴다 (양쪽에 기본값을 두면 언젠가 어긋난다).
OPTS="--clip-mode --single-view"
[ -n "$SCENE_JSON" ] && OPTS="$OPTS --scene-json $SCENE_JSON"
[ "${TIMELINE:-0}" = "1" ]      && OPTS="$OPTS --timeline"
[ "${EGO_TRACK:-0}" = "1" ]     && OPTS="$OPTS --ego-track"
[ -n "${TRAJ:-}" ]              && OPTS="$OPTS --traj $TRAJ"
[ "${USE_EGOMOTION:-0}" = "1" ] && OPTS="$OPTS --use-egomotion"
[ -n "${EGO_ABLATION:-}" ]      && OPTS="$OPTS --use-egomotion-${EGO_ABLATION}"
[ -n "${HEADER_STYLE:-}" ]      && OPTS="$OPTS --header-style $HEADER_STYLE"
[ "${SCORE_TIERS:-1}" = "0" ]   && OPTS="$OPTS --no-score-tiers"
[ "${VIDEO_INPUT:-1}" = "0" ]   && OPTS="$OPTS --clip-no-video-input"
[ "${CLIP_VIZ:-1}" = "1" ]      && OPTS="$OPTS --clip-viz"
[ "${VIZ_ALL:-0}" = "1" ]       && OPTS="$OPTS --clip-viz-all"
[ -n "${VIZ_WIDTH:-}" ]         && OPTS="$OPTS --clip-viz-width $VIZ_WIDTH"
[ "${NOT_SAVE_LOW:-1}" = "0" ]  && OPTS="$OPTS --not-save-low=0"
[ -n "${CLIP_FPS:-}" ]          && OPTS="$OPTS --clip-fps $CLIP_FPS"
[ -n "${CLIP_MAX_FRAMES:-}" ]   && OPTS="$OPTS --clip-max-frames $CLIP_MAX_FRAMES"
[ -n "${LIMIT_CLIPS:-}" ]       && OPTS="$OPTS --limit-clips $LIMIT_CLIPS"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="results/${RUN_TS}_videoC"
mkdir -p "$RUN_DIR"

{
  echo "[info] run dir     : $RUN_DIR"
  echo "[info] clips       : $TOTAL_CLIPS  ($NSHARDS shards, GPU 0-$((NSHARDS-1)))"
  echo "[info] categories  : ${SCENE_JSON:-$(python3 -c 'import config; print(config.SCENE_JSON.name)') (config.py)}"
  echo "[info] input       : clip mode, front-wide only, video input=${VIDEO_INPUT:-1}"
  echo "[info] egomotion   : ${USE_EGOMOTION:-0} (1=on, 0=off)"
  echo "[info] viz         : ${CLIP_VIZ:-1} (all=${VIZ_ALL:-0})"
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

  # 진행 상황: 샤드 CSV 의 행 수를 세어 보여준다. monitor_progress.py 는
  # (uuid, frame_idx) 단위를 전제로 해서 여기서는 쓰지 않는다.
  while :; do
    alive=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
    done_n=0
    for g in $(seq 0 $((NSHARDS-1))); do
      f="${RUN_DIR}/clip_results_shard_${g}.csv"
      # wc -l 로 세면 안 된다. 모델 출력이나 센서 시계열(--ego-track)에
      # 줄바꿈이 들어가면 한 클립이 CSV 여러 줄을 차지해(실측 115클립 ->
      # 460줄) 진행률이 총 개수를 넘어간다. CSV 규격대로 따옴표를 이해하는
      # 파서로 레코드 수를 센다.
      [ -f "$f" ] && done_n=$((done_n + $(python3 -c "
import csv,sys
try:
    with open(sys.argv[1], newline='', encoding='utf-8') as fh:
        print(max(sum(1 for _ in csv.reader(fh)) - 1, 0))
except Exception:
    print(0)
" "$f")))
    done
    [ "$done_n" -lt 0 ] && done_n=0
    printf "\r[progress] %d/%d clips  (%d shard(s) running)   " \
      "$done_n" "$TOTAL_CLIPS" "$alive"
    [ "$alive" -eq 0 ] && break
    sleep 20
  done
  echo

  wait
  echo "[info] all shards done."

  # 샤드 CSV 를 clip_results_all.csv 로 합치고 원본은 지운다
  python3 -u merge_shards.py --run-dir "$RUN_DIR"

  # 카테고리별 집계 (멀티라벨은 각 카테고리에 반영)
  python3 -u aggregate_clip.py --run-dir "$RUN_DIR"

  echo "[info] results saved in: ${RUN_DIR}/"
} 2>&1 | tee "${RUN_DIR}/run.log"

#!/usr/bin/env bash
# 클립 단위 edge-case mining (Qwen3-8B 전용) - 비디오 입력 + 전방 1뷰.
#
# 정답 라벨이 없는 데이터를 그냥 훑는 용도다. 라벨이 있는 100클립만 골라
# 채점까지 하려면 run_labeled_8b.sh 를, GPU 여러 장이 필요한 27B 는
# run_nas_nvidia_27b.sh 를 쓸 것.
#
# run_all.sh 와 무엇이 다른가:
#   run_all.sh       판정 단위 = (uuid, frame_idx). 클립당 10 timestamp x 2프레임
#                    x 3뷰. Q1/Q2/Q3 프롬프트.
#   run_nas_nvidia_8b.sh  판정 단위 = 클립 하나. 20초를 1fps 20장으로 훑어 비디오
#                    한 편으로 넣는다. nuReasoning 단계별 프롬프트.
#
# 클립을 8개 shard 로 나눠 GPU 0~7 에 배정한다. 각 프로세스가 자기 GPU
# 1장에 모델 전체를 올린다(실측 약 18.5GB / 24GB).
#
# 프레임 소스는 --data 로 고른다. NAS 에 새로 받은 청크를 돌리려면
# --data nas (또는 청크 zip 이 있는 경로)를 주면 된다 - 압축을 풀 필요는
# 없다. zip 인덱스는 첫 실행에서 한 번 만들어 캐시된다.
#
# 결과: results/<YYYYMMDD_HHMMSS>_video8b/
#   clip_results_shard_N.csv   샤드별 결과
#   clip_results_all.csv       병합본 (aggregate_clip.py 가 생성)
#   aggregate_clip.log         카테고리 + 등급 분포 집계
#   viz/<uuid>/{clip.mp4,result.json}   edge-case 클립 시각화
#   run_shard_N.log            샤드별 원본 로그
set -u
cd /home/etri/Jeongbin/Nvidia-VLM

NSHARDS=8
# 기본값은 config.py 가 단일 진실 공급원 - 미지정이면 플래그를 생략한다
SCENE_JSON="${SCENE_JSON:-}"
# NAS 청크 zip 이 이 스크립트의 대상이다 - 기본을 nas 로 둔다.
DATA="${DATA:-nas}"

# 어떤 파이썬으로 돌 것인가. run_labeled_8b.sh 와 같은 이유로 PATH 의 python3 를
# 그냥 믿으면 안 된다 - GUI/cron 처럼 conda 가 활성화되지 않은 셸에서 부르면
# /usr/bin/python3(3.8) 이 잡혀 샤드 8개가 list[str] 표기에서 전부 즉사하고,
# 그런데도 병합/집계 단계는 임포트가 돼서 "정상 완료"로 끝나 버린다.
pyok() {
  [ -x "$1" ] || return 1
  "$1" - >/dev/null 2>&1 <<'PYCHK'
import sys, importlib.util as u
assert sys.version_info >= (3, 9)           # list[str] 표기가 되는가
# 파이프라인 의존성이 있는가. av 를 빼먹으면 안 된다 - 20260831 사고와
# 똑같은 모양으로 20260918_200000 / 20260919_010000 이 또 죽었다.
# nvidia-vlm 환경은 torch/transformers/cv2 를 다 갖고 있어 이 검사를
# 통과했지만 av 가 없어서, 샤드 8개가 edge_case_mining.py 의 import av
# 에서 전멸했다. 여기 목록은 실제 import 되는 것과 맞춰 둔다.
for m in ("torch", "transformers", "cv2", "av", "pandas", "numpy", "PIL", "tqdm"):
    assert u.find_spec(m), m
PYCHK
}
PYBIN="${PYBIN:-}"
if [ -z "$PYBIN" ]; then
  for cand in "$(command -v python3 || true)" /home/etri/miniconda3/bin/python3; do
    if [ -n "$cand" ] && pyok "$cand"; then PYBIN="$cand"; break; fi
  done
fi
if [ -z "$PYBIN" ] || ! pyok "$PYBIN"; then
  echo "[error] 쓸 수 있는 파이썬이 없습니다 (python 3.9+ 와 torch 가 필요)." >&2
  echo "        시도한 인터프리터   : ${PYBIN:-$(command -v python3 || echo 없음)}" >&2
  echo "        conda activate base 후 다시 돌리거나 PYBIN=<경로> 로 지정하세요." >&2
  exit 2
fi

usage() {
  cat <<'USAGE'
Usage: bash run_nas_nvidia_8b.sh [options]

정답 라벨이 없는 데이터를 추론하고, 결과 분포를 aggregate_clip.log 에 남긴다.
(라벨과 대조해 채점하려면 run_labeled_8b.sh 를 쓸 것)

Options (환경변수로도 지정 가능 - 명령행이 우선):
  --model ID             사용할 VLM (기본: edge_case_mining.py 의 8B)      [MODEL]
  --data SRC             프레임 소스 local|nas|경로 (기본 nas)         [DATA]
                         nas = NAS 청크 zip 을 직접 읽는다(압축 해제 불필요)
  --limit-clips N        처리할 클립 수 제한 (기본: 데이터셋 전체)     [LIMIT_CLIPS]
  --only-uuids FILE      이 파일에 적힌 uuid 만 처리 (한 줄에 하나)    [ONLY_UUIDS]
  --num-shards N         GPU/shard 개수 (기본 8)                       [NSHARDS]
  --scene-json PATH      카테고리 정의 (기본: config.py 의 SCENE_JSON)  [SCENE_JSON]
  --no-viz               시각화 mp4 를 만들지 않는다                    [CLIP_VIZ=0]
  --viz-all              edge-case 가 아닌 클립까지 전부 시각화         [VIZ_ALL=1]
  --viz-normal           Normal(카테고리 없음) 클립도 시각화            [VIZ_NORMAL=1]
  --viz-special          Special(카테고리 있음) 클립을 시각화           [VIZ_SPECIAL=1]
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
  --no-category-scores    카테고리별 점수(4단계)를 프롬프트에서 끈다  [CAT_SCORES=0]
  --weather              날씨 4축(조도/강수/노면/대기가림)을
                         0~4 로 함께 매긴다. 시각화 패널과
                         aggregate_clip.log 분포에 실린다        [WEATHER=1]
  --weather-only         날씨 4축만 추론한다. edge-case 탐지와 점수를
                         모두 빼고 프롬프트를 날씨 전용으로 바꾼다
                         (--weather 를 자동으로 켠다)         [WEATHER_ONLY=1]
  --viz-per-category N   카테고리마다 최초 N개 클립만 시각화한다. 폴더를
                         score 대신 카테고리 이름으로 나눈다      [VIZ_PER_CAT]
  --no-video-input       프레임을 비디오가 아니라 낱장으로 넘긴다       [VIDEO_INPUT=0]
  --clip-fps F           초당 몇 장 뽑을지 (기본 1.0)                   [CLIP_FPS]
  --clip-max-frames N    클립당 최대 프레임 (기본 20)                   [CLIP_MAX_FRAMES]
  -h, --help             이 도움말

Examples:
  bash run_nas_nvidia_8b.sh                          # 로컬 pav_sample 전체
  bash run_nas_nvidia_8b.sh --data nas               # NAS 청크 전체 (라벨 없는 신규 데이터)
  bash run_nas_nvidia_8b.sh --data nas --limit-clips 50   # 먼저 50개로 시험
  bash run_nas_nvidia_8b.sh --data /mnt/nas/NVIDIA_DATASET/camera  # 경로 직접 지정
  bash run_nas_nvidia_8b.sh --no-viz                 # CSV 만
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data=*)            DATA="${1#*=}" ;;
    --data)              shift; DATA="${1:-local}" ;;
    --limit-clips=*)     LIMIT_CLIPS="${1#*=}" ;;
    --limit-clips)       shift; LIMIT_CLIPS="${1:-}" ;;
    --only-uuids=*)      ONLY_UUIDS="${1#*=}" ;;
    --only-uuids)        shift; ONLY_UUIDS="${1:-}" ;;
    --model=*)           MODEL="${1#*=}" ;;
    --model)             shift; MODEL="${1:-}" ;;
    --num-shards=*)      NSHARDS="${1#*=}" ;;
    --num-shards)        shift; NSHARDS="${1:-8}" ;;
    --scene-json=*)      SCENE_JSON="${1#*=}" ;;
    --scene-json)        shift; SCENE_JSON="${1:-}" ;;
    --no-viz)            CLIP_VIZ=0 ;;
    --viz-all)           VIZ_ALL=1 ;;
    --viz-normal)        VIZ_NORMAL=1 ;;
    --viz-special)       VIZ_SPECIAL=1 ;;
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
    --no-category-scores|--no-safety-tier|--no-rarity-tier|--no-score-tiers)
                         CAT_SCORES=0 ;;
    --weather|--difficulty)
                         WEATHER=1 ;;
    --weather-only|--difficulty-only)
                         WEATHER_ONLY=1 ;;
    --no-tiers-elements) TIERS_ELEMENTS=1 ;;
    --explain-traj)      EXPLAIN_TRAJ=1 ;;
    --viz-per-category=*) VIZ_PER_CAT="${1#*=}" ;;
    --viz-per-category)  shift; VIZ_PER_CAT="${1:-}" ;;
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

# 클립 개수는 진행률 표시의 분모로만 쓴다. 로컬 경로를 직접 세면 --data nas
# 일 때 0 이 나와 "클립 없음"으로 오판하므로, 실제 소스에 물어본다.
# NAS 는 여기서 zip 인덱스를 만들며(첫 실행 ~80s) 그 캐시를 샤드 8개가
# 나눠 쓴다 - 샤드들이 동시에 인덱싱을 시작하는 것을 피하는 효과도 있다.
TOTAL_CLIPS="${LIMIT_CLIPS:-}"
if [ -z "$TOTAL_CLIPS" ]; then
  TOTAL_CLIPS="$("$PYBIN" - "$DATA" <<'PY'
import sys
try:
    from clip_source import make_source
    # 기준 뷰는 edge_case_mining 이 실제로 쓰는 것과 같아야 한다. 여기에
    # 이름을 또 적어두면 뷰가 바뀔 때 진행률 분모만 조용히 어긋난다.
    from edge_case_mining import FRONT_VIEWS
    print(len(make_source(sys.argv[1]).uuids(FRONT_VIEWS[0])))
except Exception as e:
    print(f"[error] 클립 목록을 못 읽었습니다: {e}", file=sys.stderr)
    print(0)
PY
)"
fi
if [ "$TOTAL_CLIPS" -eq 0 ]; then
  echo "[error] no clips found for --data $DATA" >&2
  echo "        local 이면 pav_sample/camera/ 아래를, nas 면 청크 zip 을 확인하세요." >&2
  exit 2
fi

# 기본값은 edge_case_mining.py 를 단일 진실 공급원으로 두고, 여기서는
# 지정했을 때만 넘긴다 (양쪽에 기본값을 두면 언젠가 어긋난다).
# 이 스크립트는 GPU 1장 = 샤드 1개 구조라 GPU 여러 장에 걸쳐야 하는 27B 는
# 못 돌린다. 조용히 8B 로 돌면 몇 시간 뒤에야 알게 되므로 여기서 막는다.
case "${MODEL:-}" in
  *27B*) echo "[error] $MODEL 은 GPU 여러 장이 필요합니다." >&2
         echo "        bash run_nas_nvidia_27b.sh --model $MODEL 을 쓰세요." >&2
         exit 2 ;;
esac

OPTS="--clip-mode --single-view --data $DATA"
[ -n "${MODEL:-}" ] && OPTS="$OPTS --model $MODEL"
[ -n "$SCENE_JSON" ] && OPTS="$OPTS --scene-json $SCENE_JSON"
[ "${TIMELINE:-0}" = "1" ]      && OPTS="$OPTS --timeline"
[ "${EGO_TRACK:-0}" = "1" ]     && OPTS="$OPTS --ego-track"
[ -n "${TRAJ:-}" ]              && OPTS="$OPTS --traj $TRAJ"
[ "${USE_EGOMOTION:-0}" = "1" ] && OPTS="$OPTS --use-egomotion"
[ -n "${EGO_ABLATION:-}" ]      && OPTS="$OPTS --use-egomotion-${EGO_ABLATION}"
[ -n "${HEADER_STYLE:-}" ]      && OPTS="$OPTS --header-style $HEADER_STYLE"
# SCORE_TIERS=0 은 옛 이름 - 둘 다 끄는 뜻으로 계속 받아준다.
# 옛 환경변수 이름. 점수가 두 축에서 카테고리별 하나로 바뀌었어도
# 스크립트를 부르는 쪽이 아직 이 이름을 쓸 수 있어 받아 준다.
[ "${SCORE_TIERS:-1}" = "0" ] || [ "${SAFETY_TIER:-1}" = "0" ] \
  || [ "${RARITY_TIER:-1}" = "0" ] && CAT_SCORES=0
[ "${CAT_SCORES:-1}" = "0" ]    && OPTS="$OPTS --no-category-scores"
# 옛 환경변수 이름(DIFFICULTY/DIFFICULTY_ONLY)도 계속 받는다.
[ "${DIFFICULTY:-0}" = "1" ]      && WEATHER=1
[ "${DIFFICULTY_ONLY:-0}" = "1" ] && WEATHER_ONLY=1
[ "${WEATHER:-0}" = "1" ]         && OPTS="$OPTS --weather"
[ "${WEATHER_ONLY:-0}" = "1" ]    && OPTS="$OPTS --weather-only"
[ "${TIERS_ELEMENTS:-0}" = "1" ] && OPTS="$OPTS --no-tiers-elements"
[ "${EXPLAIN_TRAJ:-0}" = "1" ] && OPTS="$OPTS --explain-traj"
[ -n "${VIZ_PER_CAT:-}" ]        && OPTS="$OPTS --viz-per-category $VIZ_PER_CAT"
[ "${VIDEO_INPUT:-1}" = "0" ]   && OPTS="$OPTS --clip-no-video-input"
# --viz-per-category 는 그 자체가 "시각화하라"는 뜻이다 - 로컬 스크립트는
# 시각화가 기본 off 라, 이걸 안 켜주면 아무것도 저장되지 않는다.
[ -n "${VIZ_PER_CAT:-}" ] && CLIP_VIZ=1
[ "${CLIP_VIZ:-1}" = "1" ]      && OPTS="$OPTS --clip-viz"
[ "${VIZ_ALL:-0}" = "1" ]       && OPTS="$OPTS --clip-viz-all"
[ "${VIZ_NORMAL:-0}" = "1" ]    && OPTS="$OPTS --viz-normal"
[ "${VIZ_SPECIAL:-0}" = "1" ]   && OPTS="$OPTS --viz-special"
[ -n "${VIZ_WIDTH:-}" ]         && OPTS="$OPTS --clip-viz-width $VIZ_WIDTH"
[ "${NOT_SAVE_LOW:-1}" = "0" ]  && OPTS="$OPTS --not-save-low=0"
[ -n "${CLIP_FPS:-}" ]          && OPTS="$OPTS --clip-fps $CLIP_FPS"
[ -n "${CLIP_MAX_FRAMES:-}" ]   && OPTS="$OPTS --clip-max-frames $CLIP_MAX_FRAMES"
[ -n "${LIMIT_CLIPS:-}" ]       && OPTS="$OPTS --limit-clips $LIMIT_CLIPS"
# 특정 uuid 만 돌린다(GUI 단일 클립 조회). 전체 인덱스를 안 만들고 그 클립이
# 든 zip 만 찾아 열므로, NAS 에서도 몇 초~30초 안에 시작한다.
[ -n "${ONLY_UUIDS:-}" ]        && OPTS="$OPTS --only-uuids $ONLY_UUIDS"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="results/unlabeled/${RUN_TS}_video8b"
mkdir -p "$RUN_DIR"

{
  echo "[info] run dir     : $RUN_DIR"
  echo "[info] data        : $DATA  (no ground-truth labels - 분포만 집계한다)"
  echo "[info] clips       : $TOTAL_CLIPS  ($NSHARDS shards, GPU 0-$((NSHARDS-1)))"
  echo "[info] categories  : ${SCENE_JSON:-$("$PYBIN" -c 'import config; print(config.SCENE_JSON.name)') (config.py)}"
  echo "[info] python      : $PYBIN"
  echo "[info] input       : clip mode, front-wide only, video input=${VIDEO_INPUT:-1}"
  echo "[info] egomotion   : ${USE_EGOMOTION:-0} (1=on, 0=off)"
  echo "[info] viz         : ${CLIP_VIZ:-1} (all=${VIZ_ALL:-0})"
  echo "[info] opts        : $OPTS"
  [ -n "${MEMO:-}" ] && echo "[info] memo        : $MEMO"
  echo

  pids=()
  for g in $(seq 0 $((NSHARDS-1))); do
    CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      nohup "$PYBIN" -u edge_case_mining.py \
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
      [ -f "$f" ] && done_n=$((done_n + $("$PYBIN" -c "
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
  "$PYBIN" -u merge_shards.py --run-dir "$RUN_DIR"

  # 결과 분포 집계 -> <run-dir>/aggregate_clip.log
  #   - Normal / Special 개수와 비율
  #   - 카테고리별 개수, 전체 대비 비율, special 대비 비율
  #   - safety/rarity 등급별 개수/비율과 평균/분산/중앙값
  # 라벨이 없으니 evaluate_labels.py 는 돌리지 않는다 - 대조할 정답이 없다.
  "$PYBIN" -u aggregate_clip.py --run-dir "$RUN_DIR"

  echo "[info] results saved in: ${RUN_DIR}/"
} 2>&1 | tee "${RUN_DIR}/run.log"

# tee 로 파이프하면 종료 코드가 tee 의 것(항상 0)이 된다. 그래서 샤드가
# 전부 죽어도 GUI/cron 은 "정상 완료"로 표시한다. 결과 CSV 유무로 판정한다.
if ! ls "${RUN_DIR}"/clip_results*.csv >/dev/null 2>&1; then
  echo "[error] 결과 CSV 가 없습니다 - 샤드가 전부 실패했습니다." >&2
  echo "        원인: ${RUN_DIR}/run_shard_0.log" >&2
  exit 1
fi

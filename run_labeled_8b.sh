#!/usr/bin/env bash
# test_label.json 의 100개 클립만 추론하고, 정답 라벨과 대조해 성능을 잰다.
#
# run_video_C.sh 와 다른 점:
#   run_video_C.sh  데이터셋 전체(또는 앞에서 N개)를 처리한다.
#   run_labeled_8b.sh     라벨이 있는 클립만 골라 처리하고 evaluate_labels.py 까지 돌린다.
#
# 라벨된 클립은 uuid 로 흩어져 있어 --limit-clips 로는 뽑을 수 없다. 그래서
# uuid 목록을 --only-uuids 파일로 넘겨 그 클립만 처리하게 한다.
#
# 결과: results/<YYYYMMDD_HHMMSS>_eval/
#   clip_results_shard_N.csv   샤드별 추론 결과
#   evaluation.log             4개 축 성능 리포트 (정답 대조)
#   aggregate_clip.log         모델 출력 분포 (카테고리/등급/난이도)
#   run.log                    전체 로그
set -u
cd /home/etri/Jeongbin/Nvidia-VLM

NSHARDS=8
# 로컬 pav_sample 이 대상이다. NAS 를 보려면 --data nas 로 명시한다.
DATA="${DATA:-local}"

# 어떤 파이썬으로 돌 것인가.
#   PATH 의 python3 를 그냥 믿으면 안 된다. GUI(gui_server.py)나 cron 처럼
#   conda 가 활성화되지 않은 셸에서 이 스크립트를 부르면 /usr/bin/python3
#   (3.8) 이 잡히고, 샤드 8개가 전부 edge_case_mining.py 의 list[str] 표기에서
#   "'type' object is not subscriptable" 로 즉사한다(실측 20260831_132629_eval).
#   고약한 건 merge/evaluate 단계는 3.8 에서도 임포트가 돼서 스크립트가 끝까지
#   "정상" 진행한다는 점이다 - 결과가 0건이라는 사실만 남는다.
# 그래서 PATH 의 python3 가 요건(3.9+, torch)을 만족하면 그대로 쓰고,
# 아니면 conda base 로 넘어가고, 그것도 안 되면 시작 전에 멈춘다.
pyok() {
  [ -x "$1" ] || return 1
  "$1" - >/dev/null 2>&1 <<'PYCHK'
import sys, importlib.util as u
assert sys.version_info >= (3, 9)           # list[str] 표기가 되는가
for m in ("torch", "transformers", "cv2"):  # 파이프라인 의존성이 있는가
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
# 카테고리 정의와 정답 라벨의 기본값은 config.py 가 단일 진실 공급원이다.
# 여기서 기본값을 들고 있으면 안 된다 - 예전에는 이 스크립트가 자기 값을
# 항상 명령행으로 넘겨서, config/argparse 쪽 기본값을 아무리 고쳐도
# 반영되지 않았다. 미지정이면 아래에서 플래그 자체를 생략한다.
SCENE_JSON="${SCENE_JSON:-}"
# --labels 만은 uuid 목록을 뽑아야 해서 셸도 실제 경로를 알아야 한다.
LABELS="${LABELS:-$("$PYBIN" -c 'import config; print(config.LABELS_JSON)')}"

usage() {
  cat <<'USAGE'
Usage: bash run_labeled_8b.sh [options]

Options (환경변수로도 지정 가능 - 명령행이 우선):
  --labels PATH          정답 라벨 json (기본: config.py 의 LABELS_JSON)  [LABELS]
  --scene-json PATH      카테고리 정의 (기본: config.py 의 SCENE_JSON)    [SCENE_JSON]
  --num-shards N         GPU/shard 개수 (기본 8)                     [NSHARDS]
  --model ID             사용할 VLM (기본: edge_case_mining.py 의 8B)      [MODEL]
  --data SRC             프레임 소스 local|nas|경로 (기본 local)          [DATA]
                         27B 는 GPU 여러 장이 필요해 여기서 못 돌린다 -
                         run_labeled_27b.sh 를 쓸 것.
  --viz                  시각화 mp4 도 만든다 (기본 안 만듦)          [CLIP_VIZ=1]
  --viz-normal           Normal 클립을 시각화 -> viz/normal/score_N/    [VIZ_NORMAL=1]
  --viz-special          Special 클립을 시각화 -> viz/special/score_N/  [VIZ_SPECIAL=1]
  --timeline             1단계를 시간순 서술로 (기본 off)            [TIMELINE=1]
  --ego-track            egomotion 을 1초 간격 시계열로도 제공        [EGO_TRACK=1]
  --traj center|width    자차 미래 궤적을 프레임에 그린다 (기본 off)   [TRAJ]
  --memo "TEXT"          이 실행이 무엇을 시험하는지 한 줄 메모.
                         evaluation.log 머리에 [info] MEMO 로 찍힌다      [MEMO]
  --use-egomotion        egomotion 사실(자차 행동 요약)을 주입 (기본 off) [USE_EGOMOTION=1]
  --use-egomotion-c      [대조군C] 헤더/hint 유지, 센서 수치만 제거         [EGO_ABLATION=c]
  --use-egomotion-d      [대조군D] hint 만 남기고 헤더/수치 제거            [EGO_ABLATION=d]
  --header-style v1|v2   자차 행동 블록 헤더 문구 (기본 v1)              [HEADER_STYLE]
  --no-safety-tier        Safety Criticality(4단계)만 프롬프트에서 끈다  [SAFETY_TIER=0]
  --no-rarity-tier        Rarity(5단계)만 프롬프트에서 끈다              [RARITY_TIER=0]
  --no-score-tiers        위 둘을 한꺼번에 끄는 별칭                     [SCORE_TIERS=0]
  --difficulty           주행 난이도 4축(조도/강수/노면/대기가림)을
                         0~4 로 함께 매긴다. 시각화 패널과
                         aggregate_clip.log 분포에 실린다        [DIFFICULTY=1]
  --viz-per-category N   카테고리마다 최초 N개 클립만 시각화한다. 폴더를
                         score 대신 카테고리 이름으로 나눈다      [VIZ_PER_CAT]
  --use-3dbbox           obstacle.offline 3D bbox 라벨을 프롬프트에 주입
                         (기본 off, Animal/Jaywalking/cyclist 과탐 경향 실측됨)  [USE_3DBBOX=1]
  --no-video-input       프레임을 비디오가 아니라 낱장으로 넘긴다     [VIDEO_INPUT=0]
  --clip-fps F           초당 몇 장 뽑을지 (기본 1.0)                 [CLIP_FPS]
  --clip-max-frames N    클립당 최대 프레임 (기본 20)                 [CLIP_MAX_FRAMES]
  --eval-only DIR        추론을 건너뛰고 그 폴더의 결과만 채점한다
  -h, --help             이 도움말

Examples:
  bash run_labeled_8b.sh                                  # 추론 + 채점
  bash run_labeled_8b.sh --eval-only results/20260812_135056_videoC
  bash run_labeled_8b.sh --viz-normal --viz-special   # 둘 다 시각화
  bash run_labeled_8b.sh --viz-special                # special 만
  bash run_labeled_8b.sh --use-3dbbox                 # 3D bbox 라벨도 프롬프트에 주입
  bash run_labeled_8b.sh --clip-fps 2.0 --clip-max-frames 40
USAGE
}

EVAL_ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --labels=*)          LABELS="${1#*=}" ;;
    --labels)            shift; LABELS="${1:-}" ;;
    --only-uuids=*)      ONLY_UUIDS="${1#*=}" ;;
    --only-uuids)        shift; ONLY_UUIDS="${1:-}" ;;
    --scene-json=*)      SCENE_JSON="${1#*=}" ;;
    --scene-json)        shift; SCENE_JSON="${1:-}" ;;
    --num-shards=*)      NSHARDS="${1#*=}" ;;
    --num-shards)        shift; NSHARDS="${1:-8}" ;;
    --data=*)            DATA="${1#*=}" ;;
    --data)              shift; DATA="${1:-}" ;;
    --model=*)           MODEL="${1#*=}" ;;
    --model)             shift; MODEL="${1:-}" ;;
    --viz)               CLIP_VIZ=1 ;;
    --viz-normal)        VIZ_NORMAL=1 ;;
    --viz-special)       VIZ_SPECIAL=1 ;;
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
    --no-safety-tier)    SAFETY_TIER=0 ;;
    --no-rarity-tier)    RARITY_TIER=0 ;;
    --no-score-tiers)    SAFETY_TIER=0; RARITY_TIER=0 ;;
    --difficulty)        DIFFICULTY=1 ;;
    --viz-per-category=*) VIZ_PER_CAT="${1#*=}" ;;
    --viz-per-category)  shift; VIZ_PER_CAT="${1:-}" ;;
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
  "$PYBIN" -u evaluate_labels.py --run-dir "$EVAL_ONLY" --labels "$LABELS"
  rc=$?
  # 채점과 분포는 짝이다 - 재채점 경로에서만 분포가 빠지면, 같은 폴더인데
  # 어떤 경로로 만들었느냐에 따라 산출물이 달라진다.
  "$PYBIN" -u aggregate_clip.py --run-dir "$EVAL_ONLY"
  exit $rc
fi

if [ -n "$SCENE_JSON" ] && [ ! -f "$SCENE_JSON" ]; then
  echo "[error] scene json not found: $SCENE_JSON" >&2; exit 2
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="results/labeld/${RUN_TS}_eval"
mkdir -p "$RUN_DIR"

# 라벨된 uuid 만 뽑아 파일로 넘긴다 (한 줄에 하나)
# GUI 단일 클립 조회는 uuid 파일을 직접 넘긴다 - 그때는 라벨 전체를 뽑지 않는다.
if [ -n "${ONLY_UUIDS:-}" ]; then
  UUID_FILE="$ONLY_UUIDS"
  echo "[info] --only-uuids: $UUID_FILE ($(wc -l < "$UUID_FILE") uuid)"
else
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
fi

TOTAL_CLIPS="$(wc -l < "$UUID_FILE")"

OPTS="--clip-mode --single-view --only-uuids $UUID_FILE"
# 이 스크립트는 GPU 1장 = 샤드 1개 구조라 GPU 여러 장에 걸쳐야 하는 27B 는
# 못 돌린다. 조용히 8B 로 돌면 몇 시간 뒤에야 알게 되므로 여기서 막는다.
case "${MODEL:-}" in
  *27B*) echo "[error] $MODEL 은 GPU 여러 장이 필요합니다." >&2
         echo "        bash run_labeled_27b.sh --model $MODEL 을 쓰세요." >&2
         exit 2 ;;
esac
[ -n "${MODEL:-}" ] && OPTS="$OPTS --model $MODEL"
# 프레임 소스. nas 면 NAS 청크 zip 을 직접 읽는다(압축 해제 불필요).
[ -n "${DATA:-}" ] && OPTS="$OPTS --data $DATA"
# 명시했을 때만 넘긴다 - 안 넘기면 config.py 기본값이 실제로 쓰인다
[ -n "$SCENE_JSON" ] && OPTS="$OPTS --scene-json $SCENE_JSON"
[ "${TIMELINE:-0}" = "1" ]      && OPTS="$OPTS --timeline"
[ "${EGO_TRACK:-0}" = "1" ]     && OPTS="$OPTS --ego-track"
[ -n "${TRAJ:-}" ]              && OPTS="$OPTS --traj $TRAJ"
[ "${USE_EGOMOTION:-0}" = "1" ] && OPTS="$OPTS --use-egomotion"
[ -n "${EGO_ABLATION:-}" ]      && OPTS="$OPTS --use-egomotion-${EGO_ABLATION}"
[ -n "${HEADER_STYLE:-}" ]      && OPTS="$OPTS --header-style $HEADER_STYLE"
# SCORE_TIERS=0 은 옛 이름 - 둘 다 끄는 뜻으로 계속 받아준다.
[ "${SCORE_TIERS:-1}" = "0" ]   && { SAFETY_TIER=0; RARITY_TIER=0; }
[ "${SAFETY_TIER:-1}" = "0" ]   && OPTS="$OPTS --no-safety-tier"
[ "${RARITY_TIER:-1}" = "0" ]   && OPTS="$OPTS --no-rarity-tier"
[ "${DIFFICULTY:-0}" = "1" ]     && OPTS="$OPTS --difficulty"
[ -n "${VIZ_PER_CAT:-}" ]        && OPTS="$OPTS --viz-per-category $VIZ_PER_CAT"
[ "${USE_3DBBOX:-0}" = "1" ]    && OPTS="$OPTS --use-3dbbox"
[ "${VIDEO_INPUT:-1}" = "0" ]   && OPTS="$OPTS --clip-no-video-input"
# --viz-per-category 는 그 자체가 "시각화하라"는 뜻이다 - 로컬 스크립트는
# 시각화가 기본 off 라, 이걸 안 켜주면 아무것도 저장되지 않는다.
[ -n "${VIZ_PER_CAT:-}" ] && CLIP_VIZ=1
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
  echo "[info] categories  : ${SCENE_JSON:-$("$PYBIN" -c 'import config; print(config.SCENE_JSON.name)') (config.py)}"
  echo "[info] opts        : $OPTS"
  echo "[info] python      : $PYBIN"
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
    sleep 15
  done
  echo

  wait
  echo "[info] all shards done."
  echo

  # 샤드 CSV 를 clip_results_all.csv 로 합치고 원본은 지운다
  "$PYBIN" -u merge_shards.py --run-dir "$RUN_DIR"
  echo

  "$PYBIN" -u evaluate_labels.py --run-dir "$RUN_DIR" --labels "$LABELS"
  echo

  # 정답과 대조하는 채점(evaluation.log)과 별개로, 모델 출력 자체의 분포도
  # 남긴다(aggregate_clip.log). 둘은 답하는 질문이 다르다 - 채점은 "정답을
  # 맞혔나", 분포는 "모델이 무엇을 얼마나 냈나"다. 난이도 4축은 정답 라벨이
  # 없어 채점 대상이 아니므로, 분포를 보는 것이 유일한 확인 수단이다.
  "$PYBIN" -u aggregate_clip.py --run-dir "$RUN_DIR"

  echo
  echo "[info] results saved in: ${RUN_DIR}/"
} 2>&1 | tee "${RUN_DIR}/run.log"

# tee 로 파이프하면 종료 코드가 tee 의 것(항상 0)이 된다. 그래서 샤드가 8개
# 전부 죽어도 GUI/cron 은 "정상 완료"로 표시한다. 결과 CSV 유무로 판정한다.
if ! ls "${RUN_DIR}"/clip_results*.csv >/dev/null 2>&1; then
  echo "[error] 결과 CSV 가 없습니다 - 샤드가 전부 실패했습니다." >&2
  echo "        원인: ${RUN_DIR}/run_shard_0.log" >&2
  exit 1
fi

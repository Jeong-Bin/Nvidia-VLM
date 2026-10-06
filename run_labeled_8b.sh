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
# 카테고리 정의와 정답 라벨의 기본값은 config.py 가 단일 진실 공급원이다.
# 여기서 기본값을 들고 있으면 안 된다 - 예전에는 이 스크립트가 자기 값을
# 항상 명령행으로 넘겨서, config/argparse 쪽 기본값을 아무리 고쳐도
# 반영되지 않았다. 미지정이면 아래에서 플래그 자체를 생략한다.
SCENE_JSON="${SCENE_JSON:-}"
# --labels 만은 uuid 목록을 뽑아야 해서 셸도 실제 경로를 알아야 한다.
# 명시했는지를 기억해 둔다 - --resume 은 명시하지 않은 것만 원래 실행에서 가져온다.
LABELS_SET="${LABELS:+1}"
LABELS="${LABELS:-$("$PYBIN" -c 'import config; print(config.LABELS_JSON)')}"

usage() {
  cat <<'USAGE'
Usage: bash run_labeled_8b.sh [options]

Options (환경변수로도 지정 가능 - 명령행이 우선):
  --labels PATH          정답 라벨 json (기본: config.py 의 LABELS_JSON)  [LABELS]
  --scene-json PATH      카테고리 정의 (기본: config.py 의 SCENE_JSON)    [SCENE_JSON]
  --num-shards N         GPU/shard 개수 (기본 8)                     [NSHARDS]
  --gpus 1,2,3,4,6,7     쓸 GPU 번호. 샤드 수가 이 개수로 정해지고 샤드 k 는
                         k 번째로 적은 GPU 에서 돈다 (기본: 0..N-1)       [GPUS]
                         고장 난 GPU(Xid 79 로 버스에서 빠진 0/5번 등)를
                         피할 때 쓴다. --resume 과 같이 써도 된다 - 샤드
                         수가 원래 실행과 달라도 끝난 클립은 건너뛴다.
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
  --no-category-scores    카테고리별 점수(4단계)를 프롬프트에서 끈다  [CAT_SCORES=0]
  --weather              날씨 4축(조도/강수/노면/대기가림)을
                         0~4 로 함께 매긴다. 시각화 패널과
                         aggregate_clip.log 분포에 실린다        [WEATHER=1]
  --weather-only         날씨 4축만 추론한다. edge-case 탐지와 점수를
                         모두 빼고 프롬프트를 날씨 전용으로 바꾼다
                         (--weather 를 자동으로 켠다)         [WEATHER_ONLY=1]
  --viz-per-category N   카테고리마다 최초 N개 클립만 시각화한다. 폴더를
                         score 대신 카테고리 이름으로 나눈다      [VIZ_PER_CAT]
  --use-3dbbox           obstacle.offline 3D bbox 라벨을 프롬프트에 주입
                         (기본 off, Animal/Jaywalking/cyclist 과탐 경향 실측됨)  [USE_3DBBOX=1]
  --no-video-input       프레임을 비디오가 아니라 낱장으로 넘긴다     [VIDEO_INPUT=0]
  --clip-fps F           초당 몇 장 뽑을지 (기본 1.0)                 [CLIP_FPS]
  --clip-max-frames N    클립당 최대 프레임 (기본 20)                 [CLIP_MAX_FRAMES]
  --eval-only DIR        추론을 건너뛰고 그 폴더의 결과만 채점한다
  --resume DIR           중단되었거나 일부 샤드가 실패한 실행을 이어받는다.
                         결과가 있는 클립(n_frames>0)은 건너뛰고, 행이
                         없거나 n_frames=0 인 클립만 다시 추론한 뒤 기존
                         결과와 합쳐 전체 클립으로 다시 채점한다
                         (evaluation.log / aggregate_clip.log 를 새로 쓰고,
                         이전 것은 DIR/logs_before_resume_<시각>/ 에 남긴다).
                         원래 실행과 같은 옵션을 줘야 한다 - 모델/카테고리/
                         프롬프트/fps 등이 다르면 시작 전에 거부한다.
                         uuid 목록은 DIR/eval_uuids.txt 를 그대로 쓴다.
                         --labels / --scene-json 은 생략하면 원래 실행이
                         쓴 파일을 쓴다(config.py 기본값이 아니라).
  -h, --help             이 도움말

Examples:
  bash run_labeled_8b.sh                                  # 추론 + 채점
  bash run_labeled_8b.sh --eval-only results/20260812_135056_videoC
  bash run_labeled_8b.sh --viz-normal --viz-special   # 둘 다 시각화
  bash run_labeled_8b.sh --viz-special                # special 만
  bash run_labeled_8b.sh --use-3dbbox                 # 3D bbox 라벨도 프롬프트에 주입
  bash run_labeled_8b.sh --clip-fps 2.0 --clip-max-frames 40
  bash run_labeled_8b.sh --resume results/labeld/<실행폴더> --use-egomotion --viz-normal --viz-special
USAGE
}

EVAL_ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --labels=*)          LABELS="${1#*=}"; LABELS_SET=1 ;;
    --labels)            shift; LABELS="${1:-}"; LABELS_SET=1 ;;
    --only-uuids=*)      ONLY_UUIDS="${1#*=}" ;;
    --only-uuids)        shift; ONLY_UUIDS="${1:-}" ;;
    --scene-json=*)      SCENE_JSON="${1#*=}" ;;
    --scene-json)        shift; SCENE_JSON="${1:-}" ;;
    --num-shards=*)      NSHARDS="${1#*=}"; NSHARDS_SET=1 ;;
    --num-shards)        shift; NSHARDS="${1:-8}"; NSHARDS_SET=1 ;;
    --gpus=*)            GPUS="${1#*=}" ;;
    --gpus)              shift; GPUS="${1:-}" ;;
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
    --use-3dbbox)        USE_3DBBOX=1 ;;
    --no-video-input)    VIDEO_INPUT=0 ;;
    --clip-fps=*)        CLIP_FPS="${1#*=}" ;;
    --clip-fps)          shift; CLIP_FPS="${1:-}" ;;
    --clip-max-frames=*) CLIP_MAX_FRAMES="${1#*=}" ;;
    --clip-max-frames)   shift; CLIP_MAX_FRAMES="${1:-}" ;;
    --eval-only=*)       EVAL_ONLY="${1#*=}" ;;
    --eval-only)         shift; EVAL_ONLY="${1:-}" ;;
    --resume=*)          RESUME_DIR="${1#*=}" ;;
    --resume)            shift; RESUME_DIR="${1:-}" ;;
    -h|--help)           usage; exit 0 ;;
    *)
      echo "[error] unknown argument: $1" >&2
      echo >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# --resume: 정답 라벨과 카테고리 정의는 명시하지 않으면 원래 실행 것을 쓴다.
#
# config.py 의 기본값(라벨/카테고리)은 GUI 가 실제로 넘기는 파일과 다르다.
# 이어받을 때 그 기본값으로 떨어지면 두 가지가 조용히 틀어진다(실측
# 20261002_165800_eval 이어받기): 카테고리 정의는 설정 대조에서 걸리지만,
# 라벨은 대조 대상이 아니라 그대로 진행되어 evaluation.log 가 다른 라벨
# 파일(230클립)로 채점되고, 새로 그린 시각화 82개에 엉뚱한 GT 가 찍혔다.
# 라벨은 run.log 의 첫 "[info] labels" 줄(원래 실행이 쓴 파일)에서 읽는다.
if [ -n "${RESUME_DIR:-}" ] && [ -f "${RESUME_DIR%/}/run_config.json" ]; then
  if [ -z "$LABELS_SET" ]; then
    _orig_labels="$(sed -n 's/^\[info\] labels *: \(.*\)  ([0-9]* clips)$/\1/p' \
                    "${RESUME_DIR%/}/run.log" 2>/dev/null | head -1)"
    if [ -n "$_orig_labels" ]; then
      LABELS="$_orig_labels"
      echo "[resume] labels     : $LABELS  (원래 실행에서 가져옴)"
    else
      echo "[warn] --resume: 원래 실행의 라벨 파일을 run.log 에서 찾지 못해 기본값을 씁니다: $LABELS" >&2
    fi
  fi
  if [ -z "$SCENE_JSON" ]; then
    SCENE_JSON="$("$PYBIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("key",{}).get("scene_json") or "")' \
                  "${RESUME_DIR%/}/run_config.json")"
    [ -n "$SCENE_JSON" ] && echo "[resume] scene-json : $SCENE_JSON  (원래 실행에서 가져옴)"
  fi
fi

[ -f "$LABELS" ] || { echo "[error] labels not found: $LABELS" >&2; exit 2; }

# 샤드 k -> GPU 번호. 예전에는 샤드 번호를 그대로 GPU 번호로 써서, GPU
# 하나가 버스에서 빠지면(Xid 79) 그 샤드가 "No CUDA GPUs are available" 로
# 죽고 몫의 클립이 통째로 빠졌다(실측 20261002_165800_eval: 82/333).
if [ -n "${GPUS:-}" ]; then
  IFS=',' read -r -a GPU_LIST <<< "$GPUS"
  for x in "${GPU_LIST[@]}"; do
    case "$x" in
      ''|*[!0-9]*) echo "[error] --gpus: 숫자를 쉼표로 구분해 주세요: $GPUS" >&2; exit 2 ;;
    esac
  done
  if [ "${NSHARDS_SET:-0}" = "1" ] && [ "$NSHARDS" != "${#GPU_LIST[@]}" ]; then
    echo "[error] --num-shards $NSHARDS 와 --gpus ($GPUS, ${#GPU_LIST[@]}장) 가 맞지 않습니다" >&2
    exit 2
  fi
  NSHARDS="${#GPU_LIST[@]}"
else
  GPU_LIST=($(seq 0 $((NSHARDS-1))))
fi

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

# GPU 점검 - 실행 폴더를 만들기 전에 한다. 한 장이라도 이상하면 여기서 멈춘다
# (빈 결과 폴더가 남지 않게). 기준은 gpu_check.sh 한 곳에만 있다.
source "$(dirname "${BASH_SOURCE[0]}")/gpu_check.sh"
gpu_preflight "${GPU_LIST[*]}" || exit 3

if [ -n "${RESUME_DIR:-}" ]; then
  RUN_DIR="${RESUME_DIR%/}"
  if ! ls "${RUN_DIR}"/clip_results*.csv >/dev/null 2>&1; then
    echo "[error] --resume: ${RUN_DIR} 에 이어받을 clip_results*.csv 가 없습니다" >&2
    exit 2
  fi
else
  RUN_TS="$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="results/labeld/${RUN_TS}_eval"
  mkdir -p "$RUN_DIR"
fi

# 라벨된 uuid 만 뽑아 파일로 넘긴다 (한 줄에 하나)
# GUI 단일 클립 조회는 uuid 파일을 직접 넘긴다 - 그때는 라벨 전체를 뽑지 않는다.
#
# 이어받을 때는 원래 실행의 eval_uuids.txt 를 그대로 쓴다. 라벨 파일에서
# 다시 뽑으면 안 된다 - 샤드 배정은 이 목록의 순서로 정해지므로(uuids[k::N]),
# 그 사이 라벨이 추가/삭제되면 순서가 밀려 어떤 클립은 두 번, 어떤 클립은
# 한 번도 안 돈다. 덮어쓰면 원래 실행의 목록도 사라진다.
if [ -n "${ONLY_UUIDS:-}" ]; then
  UUID_FILE="$ONLY_UUIDS"
  echo "[info] --only-uuids: $UUID_FILE ($(wc -l < "$UUID_FILE") uuid)"
elif [ -n "${RESUME_DIR:-}" ]; then
  UUID_FILE="${RUN_DIR}/eval_uuids.txt"
  if [ ! -f "$UUID_FILE" ]; then
    echo "[error] --resume: ${UUID_FILE} 가 없습니다 - 원래 실행의 uuid 목록을" >&2
    echo "        --only-uuids 로 넘기세요." >&2
    exit 2
  fi
  echo "[info] --resume: $UUID_FILE ($(wc -l < "$UUID_FILE") uuid)"
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

RESUME_FLAG=""
if [ -n "${RESUME_DIR:-}" ]; then
  # 준비 단계는 샤드를 띄우기 전에 한 프로세스로 한 번만 한다. 샤드마다
  # 하면 같은 병합본을 여러 프로세스가 동시에 다시 쓴다. 같은 $OPTS 로
  # 부르므로 원래 실행과 설정이 다르면 여기서 멈춘다 - 파일을 건드리기 전에.
  echo "[resume] ${RUN_DIR} 점검/정리 중..."
  if ! "$PYBIN" -u edge_case_mining.py \
        --num-shards "$NSHARDS" --shard-id 0 $OPTS --memo "${MEMO:-}" \
        --out "${RUN_DIR}/clip_results_shard_0.csv" --viz-dir "${RUN_DIR}/viz" \
        --resume-prepare; then
    echo "[error] --resume 준비 단계에서 멈췄습니다 - 위 메시지를 확인하세요." >&2
    echo "        결과 파일은 바뀌지 않았거나, 바뀌었다면 ${RUN_DIR}/resume_backup_*/ 에 원본이 있습니다." >&2
    exit 1
  fi
  RESUME_FLAG="--resume"
  # 채점 로그는 아래에서 전체 클립 기준으로 새로 쓴다(evaluate_labels.py 가
  # 덮어쓴다). 일부 클립만 반영된 이전 판과 비교할 수 있게 남겨 둔다.
  LOG_BAK="${RUN_DIR}/logs_before_resume_$(date +%Y%m%d_%H%M%S)"
  for f in evaluation.log aggregate_clip.log; do
    if [ -f "${RUN_DIR}/$f" ]; then
      mkdir -p "$LOG_BAK" && cp -p "${RUN_DIR}/$f" "$LOG_BAK/"
    fi
  done
  # 샤드 로그는 복사가 아니라 옮긴다. evaluate_labels.py 는 run_shard_*.log
  # 의 에러를 훑어 "샤드 실패" 를 알리는데, 옛 로그를 남겨 두면 이어받기로
  # 클립이 다 채워진 뒤에도 원래 실행의 실패를 다시 보고한다(샤드 수를
  # 줄여 이어받으면 옛 로그는 덮이지도 않는다).
  for f in "${RUN_DIR}"/run_shard_*.log; do
    [ -f "$f" ] || continue
    mkdir -p "$LOG_BAK" && mv "$f" "$LOG_BAK/"
  done
  [ -d "$LOG_BAK" ] && echo "[resume] 이전 채점 로그 -> $LOG_BAK/"
fi

{
  echo
  echo "[info] run dir     : $RUN_DIR"
  [ -n "$RESUME_FLAG" ] && echo "[info] resume      : $(date '+%Y-%m-%d %H:%M:%S') 이어받기"
  echo "[info] labels      : $LABELS  ($TOTAL_CLIPS clips)"
  echo "[info] categories  : ${SCENE_JSON:-$("$PYBIN" -c 'import config; print(config.SCENE_JSON.name)') (config.py)}"
  echo "[info] opts        : $OPTS"
  echo "[info] shards      : $NSHARDS  (GPU ${GPU_LIST[*]})"
  echo "[info] python      : $PYBIN"
  [ -n "${MEMO:-}" ] && echo "[info] memo        : $MEMO"
  echo

  pids=()
  for g in $(seq 0 $((NSHARDS-1))); do
    # PCI_BUS_ID: --gpus 번호를 nvidia-smi 번호와 같게 한다. 기본값
    # (FASTEST_FIRST)에서는 CUDA 번호가 nvidia-smi 와 다를 수 있어, 피하려던
    # GPU 를 도리어 잡을 수 있다.
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_LIST[$g]} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      nohup "$PYBIN" -u edge_case_mining.py \
        --num-shards "$NSHARDS" --shard-id "$g" \
        $OPTS \
        --memo "${MEMO:-}" \
        --out "${RUN_DIR}/clip_results_shard_${g}.csv" \
        --viz-dir "${RUN_DIR}/viz" $RESUME_FLAG \
        >> "${RUN_DIR}/run_shard_${g}.log" 2>&1 &
    pids+=($!)
  done
  echo "[info] launched ${#pids[@]} shards, PIDs: ${pids[*]}"

  while :; do
    alive=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
    done_n=0
    # 이어받으면 옛 결과는 병합본(clip_results_all.csv)에 있다.
    for f in "${RUN_DIR}/clip_results_all.csv" \
             $(for g in $(seq 0 $((NSHARDS-1))); do echo "${RUN_DIR}/clip_results_shard_${g}.csv"; done); do
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
} 2>&1 | tee -a "${RUN_DIR}/run.log"

# tee 로 파이프하면 종료 코드가 tee 의 것(항상 0)이 된다. 그래서 샤드가 8개
# 전부 죽어도 GUI/cron 은 "정상 완료"로 표시한다. 결과 CSV 유무로 판정한다.
if ! ls "${RUN_DIR}"/clip_results*.csv >/dev/null 2>&1; then
  echo "[error] 결과 CSV 가 없습니다 - 샤드가 전부 실패했습니다." >&2
  echo "        원인: ${RUN_DIR}/run_shard_0.log" >&2
  exit 1
fi

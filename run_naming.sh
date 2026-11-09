# run_naming.sh - 실행 폴더 이름 규칙. 실행 스크립트가 source 한다.
#
#   new_run_dir labeled     # -> results/<데이터셋>/<날짜_시간>[_<메모>]
#
# 데이터셋 폴더는 clip_source.results_group(--data) 가 정한다(GUI 의 데이터
# 선택 이름에서 공백만 _). 채점하는 실행인지(labeled)/아닌지(unlabeled)는
# 폴더 위치가 아니라 실행 폴더의 .run_kind 파일에 적는다 - GUI 가 이것으로
# [라벨 평가 결과] 와 [NAS 추론 결과] 목록을 나눈다.
#
# --memo 가 있으면 폴더 이름 끝에 _<메모> 를 붙인다. 폴더 목록만 보고도 어떤
# 실험인지 알 수 있게 하려는 것이다(전에는 손으로 폴더 이름을 바꿨다).
#
# 메모는 사람이 쓴 문장이라 그대로 붙이면 안 되는 문자가 섞인다:
#   - 공백      : GUI 가 run.log 의 "run dir : (\S+)" 로 경로를 읽어 공백에서 끊긴다 -> _
#   - / \       : 하위 폴더가 생겨 버린다 -> _
#   - : * ? " < > | ' ` $ ; : 셸·윈도우(클라이언트에서 받을 때)에서 문제 -> 제거
# 한글은 그대로 둔다(기존 폴더들도 한글을 쓴다). 너무 길면 60자에서 자른다.
memo_suffix() {
  local m="$1"
  [ -z "$m" ] && return 0
  m=$(printf '%s' "$m" | tr -s '[:space:]/\\' '_' | tr -d ':*?"<>|`$;'"'")
  m="${m#_}"; m="${m%_}"
  m="${m:0:60}"
  [ -n "$m" ] && printf '_%s' "$m"
}

# new_run_dir <labeled|unlabeled> : RUN_DIR 을 정하고 만든다. $PYBIN, $DATA,
# $RUN_TS, $MEMO 를 쓴다. 같은 초에 같은 이름이 있으면 _2, _3 을 붙인다(전에는
# _eval / _video8b 가 갈래를 갈라 줬지만 이제 이름에 없다).
new_run_dir() {
  local kind="$1" group base n=2
  group="$("$PYBIN" -c 'import sys; from clip_source import results_group; print(results_group(sys.argv[1]))' "${DATA:-}")" || return 1
  [ -n "$group" ] || return 1
  base="results/${group}/${RUN_TS}$(memo_suffix "${MEMO:-}")"
  RUN_DIR="$base"
  while [ -e "$RUN_DIR" ]; do RUN_DIR="${base}_$n"; n=$((n + 1)); done
  mkdir -p "$RUN_DIR" && printf '%s\n' "$kind" > "$RUN_DIR/.run_kind"
}

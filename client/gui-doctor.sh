#!/usr/bin/env bash
# 서버에서 실행되는 GUI 자가진단 + 기동 스크립트.
#
# 왜 따로 뺐나:
#   .bat 과 .sh 가 각자 ssh 안에 인라인으로 같은 로직을 품고 있었다. 한쪽만
#   고치면 두 클라이언트의 동작이 갈라지고, 특히 .bat 은 따옴표 이스케이프가
#   겹쳐 조건문을 넣기가 매우 위험하다. 서버에 파일로 두고 양쪽이 똑같이
#   호출하면 로직이 한 벌로 유지된다.
#
# 이 스크립트가 고치는 실제 사고:
#   기존 클라이언트는 `pgrep -f 'gui_server.py'` 로 생사를 판정했는데, 이
#   패턴은 그 문자열을 명령줄에 달고 있는 ssh/bash 자기 자신에게도 걸린다.
#   그래서 GUI 가 죽어 있어도 "이미 실행 중" 으로 보고하고, 클라이언트는
#   그대로 브라우저를 열어 빈 페이지를 띄웠다 - 사용자는 원인을 알 길이
#   없었다(20260911 재부팅 후 3일간 이 상태였다).
#   그래서 여기서는 프로세스 유무가 아니라 "HTTP 가 200 을 주는가" 로
#   판정한다. 그게 사용자가 실제로 겪는 조건이다.
#
# 출력 규약:
#   사람이 읽을 진단문을 그대로 찍는다. 마지막 줄에 STATUS=<코드> 를 남겨
#   클라이언트가 파싱할 수 있게 한다.
#     STATUS=OK        - GUI 응답함. 터널을 열어도 된다.
#     STATUS=STARTED   - 죽어 있어서 새로 띄웠고 응답 확인됨.
#     STATUS=FAIL      - 띄우지 못했다. 위에 이유가 적혀 있다.
set -u

REMOTE_DIR="${1:-/home/etri/Jeongbin/Nvidia-VLM}"
RPORT="${2:-8000}"
DATA="${3:-nas}"

cd "$REMOTE_DIR" 2>/dev/null || {
  echo "  [오류] 서버에 폴더가 없습니다: $REMOTE_DIR"
  echo "STATUS=FAIL"; exit 0
}

# NAS 마운트 상태를 터미널에도 알린다.
#
# GUI 화면에는 경고 배너가 뜨지만, 화면을 열기 전/열지 못할 때는 그걸 볼 수
# 없다. 재부팅마다 GUI 와 NAS 가 함께 끊기므로, GUI 를 살린 직후가 NAS 도
# 같이 점검해 줄 가장 자연스러운 시점이다.
nas_hint() {
  local cam="/mnt/nas/NVIDIA_DATASET/20260901/camera"
  if [ -d "$cam" ] && [ -n "$(ls -A "$cam" 2>/dev/null)" ]; then
    echo "  - NAS 마운트 정상"
    return
  fi
  echo
  echo "  [주의] NAS 가 마운트되어 있지 않습니다 (${cam} 없음/비어 있음)."
  echo "         NAS 탭의 데이터가 보이지 않습니다. 서버에서 아래를 실행하세요:"
  echo
  echo "         sudo mount -t cifs //10.254.92.171/OPEN_DATASET /mnt/nas \\"
  echo "           -o username=E2E_DATA,vers=3.0,uid=\$(id -u),gid=\$(id -g),\\"
  echo "              cache=loose,nounix,serverino,iocharset=utf8"
  echo
  echo "         마운트한 뒤에는 GUI 를 한 번 재시작해야 합니다"
  echo "         (시작 시점에 NAS 인덱스를 읽기 때문)."
}

# HTTP 로 살아있는지 본다. 프로세스 유무가 아니라 이것이 진짜 기준이다.
alive() {
  if command -v curl >/dev/null 2>&1; then
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
         "http://127.0.0.1:${RPORT}/" 2>/dev/null)" = "200" ]
  else
    python3 - "$RPORT" <<'PY' 2>/dev/null
import sys, urllib.request
try:
    urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/", timeout=5)
except Exception:
    sys.exit(1)
PY
  fi
}

if alive; then
  echo "  - GUI 정상 응답 (포트 ${RPORT})"
  nas_hint
  echo "STATUS=OK"; exit 0
fi

# 여기부터는 응답이 없는 상태. 원인을 좁혀서 알려준다.
echo "  - GUI 가 응답하지 않습니다. 원인을 확인합니다..."

# 자기 자신(ssh/bash 명령줄)에 걸리지 않도록 [g] 트릭을 쓴다.
GUIPID="$(pgrep -f '[g]ui_server\.py' | head -1 || true)"
if [ -n "$GUIPID" ]; then
  echo "    · 프로세스는 살아 있으나(PID $GUIPID) 응답이 없습니다 - 기동 중이거나 멈춘 상태."
  echo "    · 최근 로그:"
  tail -5 gui.log 2>/dev/null | sed 's/^/        /' || echo "        (로그 없음)"
  echo "    · 5초 더 기다려 봅니다..."
  sleep 5
  if alive; then
    echo "  - 이제 응답합니다(기동 중이었음)."
    nas_hint
    echo "STATUS=OK"; exit 0
  fi
  echo "    · 여전히 무응답 -> 기존 프로세스를 정리하고 다시 띄웁니다."
  kill "$GUIPID" 2>/dev/null || true
  sleep 2
fi

# 포트를 다른 프로그램이 점유했는지 - 이건 재시작해도 안 풀리는 원인이다.
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":${RPORT} "; then
  echo "    · 포트 ${RPORT} 를 다른 프로그램이 쓰고 있습니다:"
  ss -ltnp 2>/dev/null | grep ":${RPORT} " | sed 's/^/        /'
  echo "    · 해결: 그 프로그램을 끄거나, GUI 를 다른 포트로 띄우세요."
  echo "            Windows:  GUI-접속.bat 10.254.92.108 8000 1024 8001"
  echo "            Linux/Mac: RPORT=8001 ./gui-connect.sh"
  echo "STATUS=FAIL"; exit 0
fi

echo "  - GUI 를 새로 시작합니다..."
setsid nohup python3 gui_server.py --port "${RPORT}" --data "${DATA}" \
  >> gui.log 2>&1 < /dev/null &

for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  sleep 1
  alive && break
done

if alive; then
  echo "  - 시작됨 (로그: ${REMOTE_DIR}/gui.log)"
  nas_hint
  echo "STATUS=STARTED"; exit 0
fi

# 끝내 못 띄웠다. 로그 꼬리가 거의 항상 원인을 담고 있다.
echo "  [오류] GUI 를 시작하지 못했습니다."
echo "  서버 로그 마지막 20줄:"
tail -20 gui.log 2>/dev/null | sed 's/^/      /' || echo "      (gui.log 가 없습니다)"
echo
echo "  서버에서 직접 실행해 오류를 보세요:"
echo "      ssh -p 1024 etri@10.254.92.108"
echo "      cd ${REMOTE_DIR} && python3 gui_server.py --port ${RPORT} --data ${DATA}"
echo "STATUS=FAIL"
exit 0

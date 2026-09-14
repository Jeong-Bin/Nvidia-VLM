#!/usr/bin/env bash
# edge-case 마이닝 GUI - 원클릭 접속 (Linux / macOS)
#
# 하는 일:
#   1) SSH 로 서버에 붙어 gui_server.py 가 떠 있는지 보고, 없으면 띄운다
#   2) 로컬 8000 -> 서버 127.0.0.1:8000 터널을 연다
#   3) 브라우저를 연다
#
# Ctrl-C 로 종료하면 터널만 닫힌다. 서버의 GUI 프로세스는 계속 살아 있어
# 다른 사람이 각자 터널로 같은 화면을 볼 수 있다.
#
# 사용:
#   ./gui-connect.sh                  # 기본값
#   ./gui-connect.sh 10.254.92.108 8080
#   SERVER=other-host SSH_PORT=22 ./gui-connect.sh
set -euo pipefail

USER_NAME="${USER_NAME:-etri}"
SERVER="${SERVER:-${1:-10.254.92.108}}"
REMOTE_DIR="${REMOTE_DIR:-/home/etri/Jeongbin/Nvidia-VLM}"
SSH_PORT="${SSH_PORT:-1024}"          # 이 서버의 sshd 는 22 가 아니라 1024
RPORT="${RPORT:-8000}"
LPORT="${LPORT:-${2:-8000}}"
DATA="${DATA:-local}"

echo
echo "  서버 : ${USER_NAME}@${SERVER}  (ssh 포트 ${SSH_PORT})"
echo

command -v ssh >/dev/null || { echo "[오류] ssh 가 없습니다."; exit 1; }

# 로컬 포트가 이미 쓰이면 비어 있는 포트로 올린다.
port_busy() {
  if command -v ss >/dev/null; then ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN
  else lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; fi
}
while port_busy "$LPORT"; do
  echo "[알림] 로컬 포트 $LPORT 사용 중 -> $((LPORT+1)) 로 옮깁니다."
  LPORT=$((LPORT + 1))
done

echo "[1/3] 서버의 GUI 확인 / 기동..."
# 진단은 서버의 gui-doctor.sh 한 곳에 모아 두고 양쪽 클라이언트가 같이 쓴다.
# 여기서 직접 pgrep 으로 판정하지 않는 이유는 doctor 안에 적어 두었다.
# ssh 접속 자체가 실패한 경우와 doctor 가 FAIL 을 돌려준 경우를 구분한다.
# set -e 가 걸려 있으므로 || 로 받아 실패해도 아래 안내를 찍을 수 있게 한다.
# 이걸 안 하면 ssh 실패 순간 스크립트가 조용히 죽어, 정작 원인을 알려주는
# 메시지가 실행되지 않는다.
DOCTOR_OUT="$(ssh -p "${SSH_PORT}" -o ConnectTimeout=10 "${USER_NAME}@${SERVER}" \
  "bash ${REMOTE_DIR}/client/gui-doctor.sh '${REMOTE_DIR}' '${RPORT}' '${DATA}'" 2>&1)" \
  && SSH_RC=0 || SSH_RC=$?
echo "$DOCTOR_OUT" | grep -v '^STATUS='

if [ $SSH_RC -ne 0 ]; then
  echo
  echo "[오류] 서버에 접속하지 못했습니다 (${USER_NAME}@${SERVER}:${SSH_PORT})."
  echo "  확인할 것:"
  echo "   1) VPN / 사내망에 연결돼 있습니까?"
  echo "   2) 서버가 켜져 있습니까?   ping ${SERVER}"
  echo "   3) SSH 키가 등록돼 있습니까?  ssh -p ${SSH_PORT} ${USER_NAME}@${SERVER}"
  echo "   (이 서버의 sshd 포트는 22 가 아니라 ${SSH_PORT} 입니다)"
  exit 1
fi

if echo "$DOCTOR_OUT" | grep -q '^STATUS=FAIL'; then
  echo
  echo "[중단] GUI 를 띄우지 못해 브라우저를 열지 않습니다. 위 내용을 확인하세요."
  exit 1
fi

echo "[2/3] 터널 여는 중 (로컬 ${LPORT} -> 서버 ${RPORT})..."
ssh -p "${SSH_PORT}" -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L "${LPORT}:127.0.0.1:${RPORT}" "${USER_NAME}@${SERVER}" &
TUNNEL=$!
# 창을 닫거나 Ctrl-C 하면 터널만 정리한다.
trap 'kill $TUNNEL 2>/dev/null || true; echo; echo "[gui] 터널 종료"' EXIT INT TERM

for _ in $(seq 20); do
  port_busy "$LPORT" && break
  sleep 1
done

URL="http://127.0.0.1:${LPORT}"
echo "[3/3] 브라우저 여는 중..."
if command -v xdg-open >/dev/null; then xdg-open "$URL" >/dev/null 2>&1 &
elif command -v open >/dev/null; then open "$URL" &
else echo "  브라우저에서 직접 여세요: $URL"; fi

echo
echo "  ============================================"
echo "   접속됨: $URL"
echo "   Ctrl-C 로 종료"
echo "  ============================================"
wait $TUNNEL

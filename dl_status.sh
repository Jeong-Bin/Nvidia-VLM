#!/usr/bin/env bash
# 청크 다운로드 진행 상황을 한눈에 본다.
LOG="${1:-$HOME/download_1000_3145.log}"
NAS=/mnt/nas/NVIDIA_DATASET/20260901/camera/camera_front_wide_120fov
STAGE=/home/etri/Jeongbin/Nvidia-VLM/.stage

echo "=== 진행 ==="
tail -3 "$LOG" | grep -E "받음|전송" || echo "  (아직 로그 없음)"
echo
DONE=$(grep -c "받음" "$LOG" 2>/dev/null || echo 0)
TOTAL=$(grep -m1 -oP '대상 \K[0-9]+' "$LOG" 2>/dev/null || echo "?")
echo "  다운로드 : $DONE / $TOTAL"
echo "  NAS zip  : $(ls "$NAS"/*.zip 2>/dev/null | wc -l) 개  ($(du -sh "$NAS" 2>/dev/null | cut -f1))"
echo "  대기 중  : $(ls "$STAGE"/*.zip 2>/dev/null | wc -l) 개 (배치 20개 차면 전송)"
echo "  실패     : $(grep -c "실패 chunk\|rsync 실패" "$LOG" 2>/dev/null | head -1) 건"
echo
if pgrep -f download_chunks_2stage >/dev/null; then
  echo "  상태: 실행 중 (PID $(pgrep -f download_chunks_2stage | head -1))"
else
  grep -q "완료: NAS 적재" "$LOG" 2>/dev/null && echo "  상태: 완료" || echo "  상태: 중단됨"
fi

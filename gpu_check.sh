# gpu_check.sh - 추론을 띄우기 전에 GPU 상태를 점검한다. 실행 스크립트가 source 한다.
#
#   source "$(dirname "$0")/gpu_check.sh"
#   gpu_preflight "0,1,2,3,4,5,6,7" || exit 3
#
# 왜 필요한가:
#   이 서버는 RTX 4090 두 장(PCI 01:00, a1:00)이 장시간 추론 중 PCIe 버스에서
#   떨어져 나가는 일(Xid 79)이 반복됐다(2026-09-09 ~ 09-23, 5회). 그 상태에서
#   샤드를 띄우면 일부 샤드만 죽거나, 더 나쁘게는 CUDA 가 남은 GPU 에 번호를
#   다시 매겨서 CUDA_VISIBLE_DEVICES=6 같은 지정이 엉뚱한 GPU 를 가리킨다.
#   몇 시간 뒤 결과 일부가 비어 있는 걸 보고서야 알게 되므로, 시작 전에 막는다.
#
# 검사는 "하나라도 이상하면 전부 거부" 다. 사용할 GPU 만 보지 않는 이유는,
# 한 장이라도 떨어지면 CUDA 장치 번호가 밀려 나머지 지정도 믿을 수 없기 때문이다.

_gpu_err() { echo "[gpu-check] $*" >&2; }

gpu_preflight() {
  local want="${1//,/ }"
  local bad=0

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    _gpu_err "nvidia-smi 를 찾을 수 없습니다 - NVIDIA 드라이버가 올라와 있지 않습니다."
    return 1
  fi

  # 1) 물리 계층: PCIe 에 꽂혀 있는 GPU. 응답 없는 장치는 lspci 에 (rev ff) 로 뜬다.
  local phys dead
  phys=$(lspci -d 10de::0300 2>/dev/null | wc -l)
  dead=$(lspci -d 10de::0300 2>/dev/null | grep -i "rev ff")
  if [ -n "$dead" ]; then
    _gpu_err "PCIe 에서 응답하지 않는 GPU 가 있습니다:"
    echo "$dead" | sed 's/^/             /' >&2
    bad=1
  fi

  # 2) 이번 부팅에서 GPU 가 버스에서 떨어진 기록. 한 번 떨어지면 재부팅 전에는
  #    돌아오지 않으므로, 지금 멀쩡해 보여도 기록이 있으면 거부한다.
  #    (커널 로그를 못 읽는 계정이면 이 단계만 건너뛴다 - 아래 단계가 잡는다)
  local xid
  xid=$(journalctl -k -b 0 --no-pager 2>/dev/null \
        | grep -E "Xid .*: (79|154),|fallen off the bus" | head -4)
  if [ -n "$xid" ]; then
    _gpu_err "이번 부팅에서 GPU 가 버스에서 떨어진 기록이 있습니다 (Xid 79):"
    echo "$xid" | sed -E 's/^.*kernel: /             /' >&2
    bad=1
  fi

  # 3) 드라이버 계층: nvidia-smi 가 모든 GPU 를 오류 없이 읽는가.
  local q rc
  q=$(nvidia-smi --query-gpu=index,pci.bus_id,name,memory.total \
                 --format=csv,noheader 2>&1)
  rc=$?
  if [ $rc -ne 0 ] || echo "$q" | grep -qiE "unknown error|requires reset|unable to determine|ERR!|lost"; then
    _gpu_err "nvidia-smi 가 GPU 를 정상적으로 읽지 못했습니다 (rc=$rc):"
    echo "$q" | sed 's/^/             /' >&2
    bad=1
  fi
  local seen
  seen=$(echo "$q" | grep -cE "^[0-9]+, ")
  if [ "$phys" -gt 0 ] && [ "$seen" -ne "$phys" ]; then
    _gpu_err "꽂혀 있는 GPU 는 ${phys}장인데 드라이버가 ${seen}장만 봅니다."
    bad=1
  fi

  # 4) 요청한 번호가 실제로 있는가.
  local g ok_ids
  ok_ids=$(echo "$q" | grep -oE "^[0-9]+, " | tr -d ', ' | tr '\n' ' ')
  for g in $want; do
    if ! echo "$q" | grep -qE "^${g}, "; then
      _gpu_err "GPU ${g} 을(를) 쓸 수 없습니다 (정상으로 보이는 GPU: ${ok_ids:-없음})"
      bad=1
    fi
  done

  # 5) CUDA 계층: 요청한 GPU 마다 실제로 컨텍스트가 만들어지는가.
  #    nvidia-smi 는 멀쩡한데 CUDA 초기화만 실패하는 경우도 있다. torch 를
  #    부르면 몇 초씩 걸리므로 드라이버 API 를 ctypes 로 직접 친다(1초 남짓).
  if [ $bad -eq 0 ]; then
    local py="${PYBIN:-python3}" out
    out=$(CUDA_VISIBLE_DEVICES="${1// /,}" "$py" - "$want" 2>&1 <<'PY'
import ctypes, sys
want = sys.argv[1].split()
try:
    cu = ctypes.CDLL("libcuda.so.1")
except OSError as e:
    print(f"libcuda.so.1 을 열 수 없음: {e}"); sys.exit(1)
rc = cu.cuInit(0)
if rc:
    print(f"cuInit 실패 (CUresult={rc})"); sys.exit(1)
n = ctypes.c_int()
cu.cuDeviceGetCount(ctypes.byref(n))
if n.value != len(want):
    print(f"CUDA 가 보는 GPU {n.value}장 != 요청 {len(want)}장 ({' '.join(want)})")
    sys.exit(1)
fail = []
for i in range(n.value):
    dev, ctx = ctypes.c_int(), ctypes.c_void_p()
    r = cu.cuDeviceGet(ctypes.byref(dev), i) or cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    if r:
        fail.append(f"GPU {want[i]} (CUresult={r})")
    else:
        cu.cuCtxDestroy_v2(ctx)
if fail:
    print("CUDA 컨텍스트 생성 실패: " + ", ".join(fail)); sys.exit(1)
PY
)
    if [ $? -ne 0 ]; then
      _gpu_err "$out"
      bad=1
    fi
  fi

  if [ $bad -ne 0 ]; then
    _gpu_err "GPU 상태에 문제가 있어 추론을 시작하지 않습니다."
    _gpu_err "버스에서 떨어진 GPU 는 재부팅해야 돌아옵니다. 재부팅 후 'nvidia-smi -L' 로 ${phys:-8}장이 보이는지 확인하세요."
    return 1
  fi
  echo "[gpu-check] OK - GPU $(echo $want | tr ' ' ',') 정상 (전체 ${seen}장)"
  return 0
}

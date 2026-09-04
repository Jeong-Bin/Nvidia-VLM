# edge-case 마이닝 GUI - 클라이언트 접속 패키지

다른 PC에서 GUI 화면에 원격 접속하기 위한 원클릭 실행파일.
SSH 터널을 쓰므로 서버 포트를 사내망에 노출하지 않는다.

## 배포 방법

이 `client/` 폴더만 사용자 PC로 복사하면 된다. (USB, 사내 공유폴더, 메일 등)

| 운영체제 | 실행파일 |
|---|---|
| Windows | `GUI-접속.bat` 더블클릭 |
| Linux / macOS | `./gui-connect.sh` 실행 |

## 사전 준비 (사용자 PC에서 최초 1회)

1. **SSH 클라이언트**
   - Windows 10 1809 이상: 설정 → 앱 → 선택적 기능 → **OpenSSH 클라이언트** 설치
                           또는 윈도우 검색창에 선택적 기능 검색
   - Linux / macOS: 이미 있음

2. **서버 접속 권한** — 아래 둘 중 하나
   - 비밀번호: 실행할 때마다 2번 입력 (기동 확인 1회 + 터널 1회)
   - SSH 키(권장, 비밀번호 없이 접속) — 아래 "SSH 키 등록" 참고

3. **사내망 / VPN** 연결 상태여야 한다.

## SSH 키 등록 (최초 1회, 선택)

키를 등록하면 실행할 때마다 비밀번호를 넣지 않아도 된다.
**등록할 때 1번은 비밀번호를 입력해야 한다** (그게 본인 확인 수단이므로).

### Linux / macOS — 터미널

```bash
ssh-keygen -t ed25519                      # 질문 3개 전부 엔터
ssh-copy-id -p 1024 etri@10.254.92.108     # 비밀번호 1회 입력
```

### Windows — PowerShell

`ssh-copy-id` 는 Windows OpenSSH 에 **없다.** 아래를 쓴다.

```powershell
ssh-keygen -t ed25519                      # 질문 3개 전부 엔터

# 공개키를 서버의 authorized_keys 에 붙인다 (비밀번호 1회 입력)
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub" | ssh -p 1024 etri@10.254.92.108 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

> PowerShell 이 아니라 **명령 프롬프트(cmd)** 라면:
> ```
> type %USERPROFILE%\.ssh\id_ed25519.pub | ssh -p 1024 etri@10.254.92.108 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
> ```

### 확인

비밀번호를 묻지 않고 `key-ok` 가 나오면 성공이다.

```
ssh -p 1024 etri@10.254.92.108 "echo key-ok"
```

`ssh-keygen` 이 `id_ed25519 already exists` 라고 하면 키가 이미 있는 것이니
덮어쓰지 말고(`n`) 등록 명령만 실행한다.

## 동작

실행하면 자동으로:

1. SSH 로 서버에 붙어 `gui_server.py` 가 떠 있는지 확인 — 없으면 띄운다
2. `로컬 8000` → `서버 127.0.0.1:8000` 터널을 연다
3. 브라우저로 `http://127.0.0.1:8000` 을 연다

창을 닫거나 Ctrl-C 하면 **터널만** 끊긴다. 서버의 GUI 프로세스는 계속 살아
있어서, 여러 사람이 각자 터널로 **같은 화면을 동시에** 볼 수 있다.

로컬 8000 포트가 이미 쓰이는 중이면 8001, 8002... 로 자동으로 옮긴다.

## 설정 바꾸기

기본값은 스크립트 상단에 있다.

| 항목 | 기본값 |
|---|---|
| 서버 | `etri@10.254.92.108` |
| SSH 포트 | `1024` (이 서버는 22 가 아니다) |
| GUI 포트 | `8000` |
| 영상 소스 | `local` (`nas` 로 바꾸면 NAS 청크 zip) |

인자로도 덮어쓸 수 있다:

```
GUI-접속.bat 10.254.92.108 8080 1024      # 서버 로컬포트 ssh포트
./gui-connect.sh 10.254.92.108 8080
SERVER=other-host SSH_PORT=22 ./gui-connect.sh
```

## 문제 해결

**`Connection refused`**
→ SSH 포트 확인. 이 서버의 sshd 는 **1024** 를 쓴다 (22 아님).

**`Permission denied (publickey,password)`**
→ 위 "사전 준비 2" 의 키 등록이 안 된 것. 또는 비밀번호 오타.

**브라우저는 열렸는데 화면이 안 뜬다**
→ 터널은 떴지만 서버 GUI 가 죽은 경우. 서버에서 로그 확인:
```
ssh -p 1024 etri@10.254.92.108 "tail -30 /home/etri/Jeongbin/Nvidia-VLM/gui.log"
```

**서버 GUI 를 재시작하고 싶다**
```
ssh -p 1024 etri@10.254.92.108 "pkill -f gui_server.py"
```
그다음 실행파일을 다시 돌리면 새로 뜬다.

## 보안 메모

GUI 서버는 **인증이 없고**, `POST /api/job` 이 서버에서 학습/추론 셸 스크립트를
실행한다. 그래서 `gui_server.py` 는 기본값인 `127.0.0.1` 에만 바인딩하고,
접근은 SSH 터널로만 열어 둔다 — 접속 권한 = SSH 계정 권한.

`--host 0.0.0.0` 으로 직접 노출하면 포트에 닿는 누구나 GPU 작업을 띄우고
라벨을 수정할 수 있으므로 쓰지 않는다.

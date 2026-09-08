@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title edge-case GUI

rem ===========================================================
rem  edge-case 마이닝 GUI - 원클릭 접속 (Windows)
rem
rem  하는 일:
rem    1) SSH 로 서버에 붙어 gui_server.py 가 떠 있는지 보고, 없으면 띄운다
rem    2) 로컬 8000 -> 서버 127.0.0.1:8000 터널을 연다
rem    3) 브라우저를 연다
rem
rem  이 창을 닫으면 터널이 끊긴다. 서버의 GUI 프로세스는 계속 살아 있다.
rem  (여러 사람이 같은 서버 GUI 를 각자 터널로 함께 볼 수 있다)
rem
rem  서버가 바뀌면 아래 SERVER / USER 만 고치면 된다.
rem ===========================================================

set "USER=etri"
set "SERVER=10.254.92.108"
set "REMOTE_DIR=/home/etri/Jeongbin/Nvidia-VLM"
rem 이 서버의 sshd 는 22 가 아니라 1024 를 쓴다.
set "SSH_PORT=1024"
set "RPORT=8000"
set "LPORT=8000"
set "DATA=local"

rem --- 인자로 덮어쓰기: GUI-접속.bat <서버IP> <로컬포트> <ssh포트> ---
if not "%~1"=="" set "SERVER=%~1"
if not "%~2"=="" set "LPORT=%~2"
if not "%~3"=="" set "SSH_PORT=%~3"

echo.
echo   서버   : %USER%@%SERVER%  ^(ssh 포트 %SSH_PORT%^)
echo   주소   : http://127.0.0.1:%LPORT%
echo.

rem --- ssh 가 있는지 ---
where ssh >nul 2>nul
if errorlevel 1 (
  echo [오류] ssh 를 찾을 수 없습니다.
  echo        Windows 설정 ^> 앱 ^> 선택적 기능 에서 "OpenSSH 클라이언트" 를 설치하세요.
  echo        ^(Windows 10 1809 이상 기본 제공^)
  pause
  exit /b 1
)

rem --- 로컬 포트가 이미 쓰이는 중이면 비어 있는 포트를 찾는다 ---
:findport
netstat -ano | findstr /r /c:"LISTENING" | findstr /c:":%LPORT% " >nul 2>nul
if not errorlevel 1 (
  echo [알림] 로컬 포트 %LPORT% 사용 중 -^> 다른 포트를 씁니다.
  set /a LPORT=%LPORT%+1
  goto findport
)

echo [1/3] 서버의 GUI 확인 / 기동...
rem  이미 떠 있으면 그대로 두고, 없을 때만 nohup 으로 띄운다.
rem  로그는 서버의 gui.log 에 쌓인다.
ssh -p %SSH_PORT% -o ConnectTimeout=10 %USER%@%SERVER% ^
  "cd %REMOTE_DIR% && if pgrep -f 'gui_server.py' >/dev/null; then echo '  - 이미 실행 중'; else nohup python3 gui_server.py --port %RPORT% --data %DATA% >> gui.log 2>&1 & sleep 2; echo '  - 새로 시작함 (로그: %REMOTE_DIR%/gui.log)'; fi"
if errorlevel 1 (
  echo.
  echo [오류] 서버 접속 실패. VPN/사내망 연결과 SSH 키를 확인하세요.
  pause
  exit /b 1
)

echo [2/3] 터널 여는 중  ^(로컬 %LPORT% -^> 서버 %RPORT%^)...
rem  터널은 창 없이 띄운다. 예전처럼 start 로 띄우면 창이 하나 더 생기고,
rem  그 창이 이 배치가 끝난 뒤에도 남아 따로 닫아야 했다.
set "PIDFILE=%TEMP%\edge_gui_tunnel_%LPORT%.pid"
rem  창이 사라지면 감시자가 ssh 를 같이 죽이므로 X 로 닫아도 터널이 안 남는다.
rem  감시 대상(이 창)의 PID 는 ps1 이 자기 부모를 따라 올라가 직접 찾는다 -
rem  배치에서 구해 넘기려면 따옴표가 겹쳐 깨지기 쉽다.
if not exist "%~dp0gui-tunnel.ps1" (
  echo [오류] gui-tunnel.ps1 이 없습니다. 이 배치와 같은 폴더에 두세요.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0gui-tunnel.ps1" ^
  -SshPort %SSH_PORT% -LocalPort %LPORT% -RemotePort %RPORT% ^
  -Target %USER%@%SERVER% -PidFile "%PIDFILE%" >nul
if errorlevel 1 (
  echo [경고] 감시 기능을 못 켰습니다 ^(PowerShell 정책^). 창 하나로 대신 엽니다.
  echo        이 경우 창을 X 로 닫으면 터널이 남을 수 있습니다.
  start "edge-case GUI tunnel" /min ssh -p %SSH_PORT% -N ^
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 ^
    -L %LPORT%:127.0.0.1:%RPORT% %USER%@%SERVER%
)

rem 터널이 열릴 때까지 잠깐 기다린다
for /l %%i in (1,1,20) do (
  timeout /t 1 /nobreak >nul
  netstat -ano | findstr /c:"127.0.0.1:%LPORT% " | findstr /c:"LISTENING" >nul 2>nul
  if not errorlevel 1 goto ready
)
echo [경고] 터널 확인 실패. 그래도 브라우저를 열어 봅니다.

:ready
echo [3/3] 브라우저 여는 중...
start "" http://127.0.0.1:%LPORT%

echo.
echo   ============================================
echo    접속됨: http://127.0.0.1:%LPORT%
echo   ============================================
echo.
echo   이 창에서 아무 키나 누르면 터널을 끊고 전부 종료합니다.
pause >nul

call :killold
echo [gui] 터널을 닫았습니다.
endlocal
exit /b 0

:killold
rem  이 배치가 띄운 터널만 PID 로 정확히 죽인다. WINDOWTITLE 로 찾으면
rem  사용자가 따로 쓰던 ssh 세션까지 잡을 수 있어 위험하다.
rem  창을 X 로 닫아 아래 정리가 실행되지 않았더라도, 다음 실행이 여기서
rem  남은 터널을 치우므로 ssh 가 쌓이지 않는다.
if exist "%PIDFILE%" (
  for /f "usebackq delims=" %%p in ("%PIDFILE%") do taskkill /pid %%p /t /f >nul 2>nul
  del "%PIDFILE%" >nul 2>nul
)
exit /b 0

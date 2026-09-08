# edge-case GUI - 터널 기동 + 창 수명 감시 (GUI-접속.bat 이 부른다)
#
# 배치 파일 안에 PowerShell 한 줄로 밀어 넣으면 따옴표가 여러 겹 겹쳐
# TEMP 경로에 공백이나 한글이 있을 때 조용히 깨진다. 스크립트로 빼서
# 인자로만 주고받는다.
param(
  [Parameter(Mandatory)][int]    $SshPort,
  [Parameter(Mandatory)][int]    $LocalPort,
  [Parameter(Mandatory)][int]    $RemotePort,
  [Parameter(Mandatory)][string] $Target,     # user@host
  [Parameter(Mandatory)][string] $PidFile
)

$ErrorActionPreference = 'Stop'

# 감시할 창 = 이 스크립트를 부른 cmd.exe (= 내 부모).
$ConsolePid = (Get-CimInstance Win32_Process -Filter "ProcessId=$PID").ParentProcessId

# 이전 실행이 남긴 터널이 있으면 먼저 치운다.
if (Test-Path -LiteralPath $PidFile) {
  $old = (Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($old) { Stop-Process -Id ([int]$old) -Force -ErrorAction SilentlyContinue }
  Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

$sshArgs = @(
  '-p', "$SshPort", '-N',
  '-o', 'ExitOnForwardFailure=yes',
  '-o', 'ServerAliveInterval=30',
  '-L', "${LocalPort}:127.0.0.1:${RemotePort}",
  $Target
)
$ssh = Start-Process ssh -PassThru -WindowStyle Hidden -ArgumentList $sshArgs
Set-Content -LiteralPath $PidFile -Value $ssh.Id -Encoding ascii

# 감시자: 이 콘솔 창이 사라지면 - 아무 키를 눌러 끝내든, X 로 닫든,
# 작업관리자로 죽이든 - ssh 도 같이 끝낸다. 배치의 정리 코드만 믿으면
# X 로 닫았을 때 그 코드가 실행되지 않아 터널이 그대로 남는다.
# 값을 문자열에 끼워 넣지 않고 인자로 건넨다. TEMP 경로에 $ 나 따옴표가
# 있어도 안전하다.
$watch = {
  param($ConsolePid, $SshPid, $PidFile)
  try { Wait-Process -Id $ConsolePid -ErrorAction Stop } catch {}
  try { Stop-Process -Id $SshPid -Force -ErrorAction SilentlyContinue } catch {}
  Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}
Start-Process powershell -WindowStyle Hidden -ArgumentList @(
  '-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand',
  [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes(
    "& {$watch} $ConsolePid $($ssh.Id) '$($PidFile -replace "'", "''")'"))
) | Out-Null

Write-Output $ssh.Id

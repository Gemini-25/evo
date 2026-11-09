# 설치 프로그램 연기 시험(v3.1 설계 D47·4.12, 윈도우 CI W5). PowerShell 7, 윈도우 전용.
#
#   ./packaging/installer_smoke.ps1 -Setup build\installer\Output\evolab-setup-3.1.0.exe `
#       -OldSetup build\installer\Output\evolab-setup-2.99.0.exe -Version 3.1.0 -Names build\installer\names.json
#
# 순서(D47):
#   1 옛 판(같은 배포본을 2.99.0으로 묶은 것) 조용히 설치 → 옛 _internal에 표시 파일, 데이터 폴더·설정 폴더에 사용자 자료 심기
#   2 이번 판 덮어 설치 → 제거 정보가 HKCU에만 하나·DisplayVersion = 이번 판, 설치 폴더가 %LOCALAPPDATA%\Programs 아래,
#     표시 파일 없음(옛 _internal을 지움), 자료·설정 그대로, 시작 메뉴 둘·바탕 화면 바로 가기와 대상·인수, 두 exe 파일 버전
#   3 설치된 두 exe 자체 점검 + 오류 보고 묶음 명령(시작 메뉴 바로 가기와 같은 명령)
#   4 옛 판으로 내려 설치 → 거절(종료 코드 ≠ 0), 버전 그대로
#   5 프로그램을 띄운 채 조용히 제거 → 뮤텍스(AppMutex) 때문에 제거하지 않음(파일·제거 정보 남음)
#   6 프로그램을 닫고 조용히 제거, 1초마다 최대 120초 기다림 → 프로그램·제거 정보·바로 가기 없음, 자료·설정 남음
# 단계마다 걸린 시간을 찍고(설계 M7) 끝에 표로 모은다. 하나라도 어긋나면 종료 코드 1.
#
# 이름(프로그램·바로 가기·데이터 폴더)은 make_installer.py --names-out이 strings.py에서 뽑은 JSON으로 받는다(한글 이름을 이 파일에
# 두 벌 두지 않고, 콘솔 인코딩을 거치지 않게 \u 이스케이프). 띄우는 프로그램(자체 점검, 제거 중 실행)은 EVOLAB_DATA_DIR·
# EVOLAB_STATE_DIR를 임시 폴더로 바꿔 돌린다(설계 D5): 심어 둔 진짜 데이터·설정 폴더는 설치·제거 프로그램만 만진다.
# 진짜 프로그램을 설치·제거하고 문서 폴더에 표시 파일을 심으므로 CI(GITHUB_ACTIONS·CI)가 아니면 -Force 없이는 돌지 않는다.
#Requires -Version 7.0
param(
  [Parameter(Mandatory = $true)][string]$Setup,
  [Parameter(Mandatory = $true)][string]$OldSetup,
  [Parameter(Mandatory = $true)][string]$Version,
  [Parameter(Mandatory = $true)][string]$Names,
  [string]$LogDir = "installer-logs",
  [int]$UninstallTimeout = 120,
  # 5단계(실행 중 조용한 제거)를 얼마나 엄격히 볼지. strict = 제거 프로그램이 스스로 끝나고 파일이 남음, lenient = 파일만 남으면
  # 됨(시간을 넘긴 제거 프로그램은 끝냄), skip = 건너뜀(닫으라는 상자는 사람이 P4에서 봄). 설계 7.4 되돌릴 지점 7.
  [ValidateSet("strict", "lenient", "skip")][string]$RunningUninstall = "strict",
  [switch]$Force
)
Set-StrictMode -Version 3.0
$ErrorActionPreference = "Stop"

# Write-Error는 $ErrorActionPreference = "Stop"이면 바로 예외가 되어 종료 코드를 고를 수 없다: 알림은 Write-Host로.
if (-not $IsWindows) { Write-Host "error: installer_smoke.ps1 runs only on Windows"; exit 2 }
if (-not ($env:GITHUB_ACTIONS -or $env:CI -or $Force)) {
  Write-Host "error: installer_smoke.ps1 installs and removes the real program and plants files in Documents: run it in CI or pass -Force"
  exit 2
}

$Setup = (Resolve-Path -LiteralPath $Setup).Path
$OldSetup = (Resolve-Path -LiteralPath $OldSetup).Path
$n = Get-Content -Raw -Encoding utf8 -LiteralPath $Names | ConvertFrom-Json
New-Item -ItemType Directory -Force $LogDir | Out-Null
$LogDirFull = (Resolve-Path -LiteralPath $LogDir).Path

# 사용자별 설치(PrivilegesRequired=lowest)의 Inno 상수: {userpf}, {userprograms}\{group}, {userdesktop}, {userdocs}, {localappdata}
$ProgramsRoot = Join-Path $env:LOCALAPPDATA "Programs"
$AppDir = Join-Path $ProgramsRoot $n.app_name
$AppExe = Join-Path $AppDir "evolab.exe"
$CliExe = Join-Path $AppDir "evolab-cli.exe"
$Group = Join-Path ([Environment]::GetFolderPath("Programs")) $n.app_name
$StartLnk = Join-Path $Group "$($n.app_name).lnk"
$ReportLnk = Join-Path $Group "$($n.report_shortcut).lnk"
$DeskLnk = Join-Path ([Environment]::GetFolderPath("DesktopDirectory")) "$($n.app_name).lnk"
$UninstPath = "Software\Microsoft\Windows\CurrentVersion\Uninstall\$($n.app_id)_is1"
$UninstKey = "HKCU:\$UninstPath"
$MachineKeys = @("HKLM:\$UninstPath", "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\$($n.app_id)_is1")
$StateDir = Join-Path $env:LOCALAPPDATA $n.state_dir_name
$DataDir = Join-Path ([Environment]::GetFolderPath("MyDocuments")) $n.data_dir_name
$Stale = Join-Path $AppDir "_internal\stale.txt"
$DataMarker = Join-Path $DataDir "runs\installer-smoke\marker.txt"
$SettingsFile = Join-Path $StateDir "settings.ini"
$LogMarker = Join-Path $StateDir "logs\installer-smoke.txt"
$Sandbox = Join-Path ([IO.Path]::GetTempPath()) ("evolab-smoke-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))

$script:failures = [Collections.Generic.List[string]]::new()
$script:times = [ordered]@{}
$script:stepName = ""
$script:stepWatch = $null

# $Ok에 형식을 붙이지 않는다: [bool] 매개변수는 $null·문자열·배열을 받지 않고 오류를 낸다. 여기서 참·거짓으로 바꾼다.
function Check($Ok, [string]$What) {
  if ([bool]$Ok) { Write-Host "  [ok]   $What" } else { Write-Host "  [FAIL] $What"; $script:failures.Add($What) }
}

function Start-Step([string]$Name) {
  Write-Host ""
  Write-Host "== $Name"
  $script:stepName = $Name
  $script:stepWatch = [Diagnostics.Stopwatch]::StartNew()
}

function Stop-Step {
  $seconds = $script:stepWatch.Elapsed.TotalSeconds
  $script:times[$script:stepName] = $seconds
  Write-Host ("  ({0:N1} s)" -f $seconds)
}

function Wait-Until([scriptblock]$Condition, [int]$Seconds) {
  $deadline = [DateTime]::UtcNow.AddSeconds($Seconds)
  while ($true) {
    if (& $Condition) { return $true }
    if ([DateTime]::UtcNow -ge $deadline) { return $false }
    Start-Sleep -Seconds 1
  }
}

# 다른 프로세스가 쓰는 중인 기록도 읽는다(없거나 못 열면 빈 글).
function Read-Shared([string]$Path) {
  try {
    $share = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
    $fs = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, $share)
    try { return [IO.StreamReader]::new($fs, [Text.Encoding]::UTF8, $true).ReadToEnd() } finally { $fs.Dispose() }
  } catch {
    return ""
  }
}

function Show-LogTail([string]$Path, [int]$Lines = 25) {
  $text = Read-Shared $Path
  if (-not $text) { Write-Host "  (no log at $Path)"; return }
  Write-Host "  --- last lines of $(Split-Path $Path -Leaf)"
  $text -split "`r?`n" | Select-Object -Last $Lines | ForEach-Object { Write-Host "  | $_" }
}

function Get-Uninstall([string]$Name) {
  if (-not (Test-Path -LiteralPath $UninstKey)) { return $null }
  $item = Get-ItemProperty -LiteralPath $UninstKey
  if ($item.PSObject.Properties.Name -contains $Name) { return [string]$item.$Name }
  return $null
}

# 조용한 설치(/CURRENTUSER = 사용자별 설치, 메시지 상자는 기본 답). 종료 코드를 돌려준다(시간 초과면 -1).
function Invoke-Setup([string]$Exe, [string]$LogName, [int]$TimeoutSeconds = 900) {
  $log = Join-Path $LogDirFull $LogName
  $argList = @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER", "/SP-", "/LOG=`"$log`"")
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $p = Start-Process -FilePath $Exe -ArgumentList $argList -PassThru
  # 끝난 뒤에도 ExitCode를 읽으려면 손잡이를 먼저 잡아 둔다.
  $null = $p.Handle
  if (-not $p.WaitForExit($TimeoutSeconds * 1000)) {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    Write-Host "  $(Split-Path $Exe -Leaf): timed out after $TimeoutSeconds s"
    Show-LogTail $log
    return -1
  }
  Write-Host ("  {0}: exit code {1}, {2:N1} s" -f (Split-Path $Exe -Leaf), $p.ExitCode, $sw.Elapsed.TotalSeconds)
  return $p.ExitCode
}

# 제거 프로그램의 임시 사본(명령줄에 /SECONDPHASE=원래 경로). 실제 제거는 이 사본이 한다.
# 함수가 돌려주는 배열은 풀려 나오므로(없으면 $null) 개수는 부르는 쪽에서 @(...)로 감싸 센다.
function Get-UninstallerClones {
  Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match "/SECONDPHASE" }
}

# 조용한 제거. 제거 프로그램은 임시 폴더의 사본으로 다시 떠서 실제 일을 하고, 원래 프로세스는 거의 끝에야 돌아온다(Inno 문서:
# 종료 코드를 받은 순간에도 제거 코드가 아직 돌 수 있음). 그래서 원래 프로세스와 사본이 모두 끝날 때까지 1초마다 보고 기다린다.
# 시간을 넘기면 남은 프로세스를 끝내고 $false.
function Invoke-Uninstall([string]$LogName) {
  $exe = Get-Uninstall "UninstallString"
  if (-not $exe) { Write-Host "  no UninstallString"; return $false }
  $exe = $exe.Trim('"')
  $log = Join-Path $LogDirFull $LogName
  $argList = @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/LOG=`"$log`"")
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $p = Start-Process -FilePath $exe -ArgumentList $argList -PassThru
  $null = $p.Handle
  $exited = $p.WaitForExit($UninstallTimeout * 1000)
  $left = [Math]::Max(5, $UninstallTimeout - [int]$sw.Elapsed.TotalSeconds)
  $finished = $exited -and (Wait-Until { @(Get-UninstallerClones).Count -eq 0 } $left)
  $code = if ($exited) { $p.ExitCode } else { "none (still running)" }
  $closed = (Read-Shared $log) -match "Log closed\."
  Write-Host ("  {0}: exit code {1}, finished: {2}, log closed: {3}, {4:N1} s" -f (Split-Path $exe -Leaf), $code, $finished, $closed, $sw.Elapsed.TotalSeconds)
  if (-not $finished) {
    Get-UninstallerClones | ForEach-Object {
      Write-Host "  stopping $($_.Name) ($($_.ProcessId))"
      Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
    if (-not $exited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
  }
  return $finished
}

# 이름 있는 뮤텍스가 있는가. 손잡이는 바로 닫는다(쥐고 있으면 프로그램을 닫아도 뮤텍스가 남아 제거가 막힌다).
function Test-AppMutex {
  $m = $null
  $exists = [Threading.Mutex]::TryOpenExisting($n.mutex, [ref]$m)
  if ($null -ne $m) { $m.Dispose() }
  return $exists
}

function Use-Sandbox([string]$Tag) {
  $env:EVOLAB_DATA_DIR = Join-Path $Sandbox "$Tag-data"
  $env:EVOLAB_STATE_DIR = Join-Path $Sandbox "$Tag-state"
  $env:EVOLAB_NO_TOUR = "1"
}

function Clear-Sandbox {
  Remove-Item Env:EVOLAB_DATA_DIR, Env:EVOLAB_STATE_DIR, Env:EVOLAB_NO_TOUR -ErrorAction SilentlyContinue
}

function Get-FileVersion([string]$Path) {
  $vi = (Get-Item -LiteralPath $Path).VersionInfo
  return "{0}.{1}.{2}.{3}" -f $vi.FileMajorPart, $vi.FileMinorPart, $vi.FileBuildPart, $vi.FilePrivatePart
}

function Test-Shortcut([string]$Path, [string]$Arguments, [string]$What) {
  $exists = Test-Path -LiteralPath $Path
  Check $exists "$What exists"
  if (-not $exists) { return }
  $lnk = $script:shell.CreateShortcut($Path)
  $target = [string]$lnk.TargetPath
  $sameTarget = $target -and ([IO.Path]::GetFullPath($target) -eq [IO.Path]::GetFullPath($AppExe))
  Check $sameTarget "$What target is evolab.exe ($target)"
  Check ([string]$lnk.Arguments -eq $Arguments) "$What arguments are '$Arguments' ('$($lnk.Arguments)')"
}

Write-Host "setup:      $Setup ($([Math]::Round((Get-Item -LiteralPath $Setup).Length / 1e6, 1)) MB)"
Write-Host "old setup:  $OldSetup ($($n.old_test_version))"
Write-Host "version:    $Version"
Write-Host "app folder: $AppDir"
if ((Test-Path -LiteralPath $UninstKey) -or (Test-Path -LiteralPath $AppDir)) {
  Write-Host "error: the program is already installed on this machine: the smoke test needs a clean first install"
  exit 1
}
$script:shell = New-Object -ComObject WScript.Shell
New-Item -ItemType Directory -Force $Sandbox | Out-Null
$gui = $null

try {
  Start-Step "1 install the old build $($n.old_test_version) (silent, per user)"
  Check ((Invoke-Setup $OldSetup "1-install-old.log") -eq 0) "old setup exit code 0"
  Check ((Get-Uninstall "DisplayVersion") -eq $n.old_test_version) "installed version is $($n.old_test_version)"
  New-Item -ItemType File -Force $Stale | Out-Null
  New-Item -ItemType Directory -Force (Split-Path $DataMarker) | Out-Null
  Set-Content -LiteralPath $DataMarker -Value "user data planted by installer_smoke.ps1" -Encoding utf8
  New-Item -ItemType Directory -Force (Split-Path $LogMarker) | Out-Null
  Set-Content -LiteralPath $LogMarker -Value "log planted by installer_smoke.ps1" -Encoding utf8
  if (Test-Path -LiteralPath $SettingsFile) {
    Write-Host "  settings.ini already exists: it must survive unchanged"
  } else {
    Set-Content -LiteralPath $SettingsFile -Value "[installer_smoke]`r`nplanted=true" -Encoding utf8
  }
  $settingsHash = (Get-FileHash -LiteralPath $SettingsFile -Algorithm SHA256).Hash
  Stop-Step

  Start-Step "2 upgrade to $Version over the old build"
  Check ((Invoke-Setup $Setup "2-upgrade.log") -eq 0) "setup exit code 0"
  Check ((Get-Uninstall "DisplayVersion") -eq $Version) "installed version is $Version"
  Check (Test-Path -LiteralPath $UninstKey) "uninstall entry is under HKCU"
  foreach ($k in $MachineKeys) { Check (-not (Test-Path -LiteralPath $k)) "no uninstall entry at $k" }
  $location = Get-Uninstall "InstallLocation"
  Check ($null -ne $location -and $location.TrimEnd('\') -eq $AppDir) "InstallLocation is $AppDir ($location)"
  Check ($AppDir.StartsWith($ProgramsRoot + "\", [StringComparison]::OrdinalIgnoreCase)) "installed under %LOCALAPPDATA%\Programs"
  foreach ($f in @($AppExe, $CliExe, (Join-Path $AppDir "_internal"))) { Check (Test-Path -LiteralPath $f) "$(Split-Path $f -Leaf) installed" }
  Check (-not (Test-Path -LiteralPath $Stale)) "old _internal was deleted before copying (stale file gone)"
  Check (Test-Path -LiteralPath $DataMarker) "data folder kept"
  Check ((Test-Path -LiteralPath $SettingsFile) -and (Get-FileHash -LiteralPath $SettingsFile -Algorithm SHA256).Hash -eq $settingsHash) "settings.ini kept unchanged"
  Check (Test-Path -LiteralPath $LogMarker) "logs kept"
  Test-Shortcut $StartLnk "" "Start menu shortcut"
  Test-Shortcut $ReportLnk "report --open" "Start menu error-report shortcut"
  Test-Shortcut $DeskLnk "" "Desktop shortcut (task on by default)"
  foreach ($exe in @($AppExe, $CliExe)) {
    if (Test-Path -LiteralPath $exe) {
      $fv = Get-FileVersion $exe
      Check ($fv -eq "$Version.0") "$(Split-Path $exe -Leaf) file version $fv"
      Check ((Get-Item -LiteralPath $exe).VersionInfo.ProductName -eq $n.app_name) "$(Split-Path $exe -Leaf) product name"
    }
  }
  if (Test-Path -LiteralPath $AppDir) {
    $files = Get-ChildItem -LiteralPath $AppDir -Recurse -File | Measure-Object -Property Length -Sum
    Write-Host ("  installed: {0} files, {1:N1} MB" -f $files.Count, ($files.Sum / 1e6))
  }
  Stop-Step

  Start-Step "3 self-test of the installed exes"
  Use-Sandbox "selftest"
  try {
    $cliLog = Join-Path $LogDirFull "3-selftest-cli.log"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $CliExe selftest --log $cliLog
    Check ($LASTEXITCODE -eq 0) ("evolab-cli.exe selftest exit code {0} ({1:N1} s)" -f $LASTEXITCODE, $sw.Elapsed.TotalSeconds)
    $guiLog = Join-Path $LogDirFull "3-selftest-gui.log"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $AppExe -ArgumentList @("selftest", "--log", "`"$guiLog`"") -PassThru
    $null = $p.Handle
    $done = $p.WaitForExit(900000)
    if (-not $done) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    Check ($done -and $p.ExitCode -eq 0) ("evolab.exe selftest exit code {0} ({1:N1} s)" -f $(if ($done) { $p.ExitCode } else { "timeout" }), $sw.Elapsed.TotalSeconds)
    if (-not ($done -and $p.ExitCode -eq 0)) { Show-LogTail $guiLog 40 }
    $zip = Join-Path $Sandbox "report.zip"
    & $CliExe report --out $zip
    Check ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $zip)) "the error-report shortcut's command works (evolab-cli.exe report)"
  } finally {
    Clear-Sandbox
  }
  Stop-Step

  Start-Step "4 downgrade to $($n.old_test_version) is refused (silent setup answers no)"
  $code = Invoke-Setup $OldSetup "4-downgrade.log"
  Check ($code -ne 0) "old setup exit code is not 0 ($code)"
  Check ((Get-Uninstall "DisplayVersion") -eq $Version) "installed version is still $Version"
  # 두 설치 파일은 같은 배포본을 묶어 exe 파일 버전으로는 가를 수 없다: 우리 [Code]가 남긴 줄로 거절한 곳을 본다.
  $downgradeLog = Join-Path $LogDirFull "4-downgrade.log"
  Check ((Read-Shared $downgradeLog) -match "Downgrade refused\.") "setup log says the downgrade was refused by InitializeSetup"
  Check (Test-Path -LiteralPath $AppExe) "evolab.exe is still installed"
  Show-LogTail $downgradeLog 8
  Stop-Step

  Start-Step "5 silent uninstall while the program runs does not remove it (AppMutex, -RunningUninstall $RunningUninstall)"
  if ($RunningUninstall -eq "skip") {
    Write-Host "  skipped (design 7.4 rollback 7): a person checks the close-the-program box in P4"
  } else {
    Use-Sandbox "running"
    try { $gui = Start-Process -FilePath $AppExe -PassThru } finally { Clear-Sandbox }
    $null = $gui.Handle
    $seen = Wait-Until { Test-AppMutex } 90
    Check $seen "the running program holds the mutex '$($n.mutex)'"
    if ($seen) {
      $finished = Invoke-Uninstall "5-uninstall-running.log"
      if ($RunningUninstall -eq "strict") {
        Check $finished "the uninstaller finished on its own"
      } else {
        Write-Host "  the uninstaller finished on its own: $finished (not checked with -RunningUninstall lenient)"
      }
      Check (Test-Path -LiteralPath $AppExe) "evolab.exe is still installed"
      Check (Test-Path -LiteralPath $UninstKey) "uninstall entry is still there"
      Show-LogTail (Join-Path $LogDirFull "5-uninstall-running.log") 12
    }
    if (-not $gui.HasExited) { Stop-Process -Id $gui.Id -Force -ErrorAction SilentlyContinue }
    $null = $gui.WaitForExit(30000)
    Check (Wait-Until { -not (Test-AppMutex) } 30) "the mutex is gone after the program is closed"
    $gui = $null
  }
  Stop-Step

  Start-Step "6 silent uninstall (wait up to $UninstallTimeout s, checking every second)"
  $finished = Invoke-Uninstall "6-uninstall.log"
  Check $finished "the uninstaller finished"
  $gone = Wait-Until { -not (Test-Path -LiteralPath $AppExe) -and -not (Test-Path -LiteralPath $UninstKey) } $UninstallTimeout
  Check $gone "program files and uninstall entry removed"
  foreach ($f in @($AppExe, $CliExe, (Join-Path $AppDir "_internal"), (Join-Path $AppDir "unins000.exe"))) {
    Check (-not (Test-Path -LiteralPath $f)) "$(Split-Path $f -Leaf) removed"
  }
  Check (-not (Test-Path -LiteralPath $UninstKey)) "uninstall entry removed"
  foreach ($l in @($StartLnk, $ReportLnk, $DeskLnk)) { Check (-not (Test-Path -LiteralPath $l)) "$(Split-Path $l -Leaf) removed" }
  Check (-not (Test-Path -LiteralPath $Group)) "Start menu folder removed"
  if (Test-Path -LiteralPath $AppDir) {
    Write-Host "  note: the program folder is still there with:"
    Get-ChildItem -LiteralPath $AppDir -Recurse -Force | ForEach-Object { Write-Host "    $($_.FullName)" }
  }
  Check (Test-Path -LiteralPath $DataMarker) "data folder kept (never deleted)"
  Check ((Test-Path -LiteralPath $SettingsFile) -and (Get-FileHash -LiteralPath $SettingsFile -Algorithm SHA256).Hash -eq $settingsHash) "settings.ini kept (silent uninstall does not ask)"
  Check (Test-Path -LiteralPath $LogMarker) "logs kept (silent uninstall does not ask)"
  Show-LogTail (Join-Path $LogDirFull "6-uninstall.log") 8
  Stop-Step
} catch {
  Write-Host "  [FAIL] unexpected error: $($_.Exception.Message)"
  Write-Host $_.ScriptStackTrace
  $script:failures.Add("unexpected error: $($_.Exception.Message)")
} finally {
  if ($null -ne $gui -and -not $gui.HasExited) { Stop-Process -Id $gui.Id -Force -ErrorAction SilentlyContinue }
  Remove-Item -LiteralPath $Sandbox -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "step times (s):"
foreach ($entry in $script:times.GetEnumerator()) { Write-Host ("  {0,7:N1}  {1}" -f $entry.Value, $entry.Key) }
if ($script:failures.Count -gt 0) {
  Write-Host ""
  Write-Host "$($script:failures.Count) check(s) failed:"
  $script:failures | ForEach-Object { Write-Host "  - $_" }
  exit 1
}
Write-Host "installer smoke test passed"
exit 0

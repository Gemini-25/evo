# 코드 서명(v3.1 설계 D46, docs/signing.md). 인증서 비밀 값이 있을 때만 CI가 부른다(없으면 단계가 건너뜀, W11).
#
#   ./packaging/installer/sign.ps1 -Dist dist\evolab    # PyInstaller 뒤·ISCC 전: 우리 exe 2개 + 서명 없는 PE 전부
#   ./packaging/installer/sign.ps1 -Path <파일>         # Inno Setup SignTool=evolabsign이 설치 파일·제거 프로그램마다 부름
#
# 환경 변수(CI 비밀 값, 단계 env로만 받는다):
#   SIGN_CERT_BASE64    코드 서명 인증서 PFX를 base64로(필수)
#   SIGN_CERT_PASSWORD  PFX 암호(없으면 빈 암호)
#   SIGN_TIMESTAMP_URL  RFC 3161 시각 도장 서버(없으면 http://timestamp.digicert.com)
# 2023년 6월부터 공개 CA의 코드 서명 개인키는 하드웨어(HSM·토큰)에 있어야 해서 PFX 파일을 받지 못할 수 있다. 그때는 클라우드
# 서명 서비스의 도구로 이 스크립트의 Invoke-Sign만 바꾼다(docs/signing.md 4절).
#Requires -Version 7.0
[CmdletBinding(DefaultParameterSetName = "Dist")]
param(
  [Parameter(Mandatory = $true, ParameterSetName = "Dist")][string]$Dist,
  [Parameter(Mandatory = $true, ParameterSetName = "File")][string]$Path
)
Set-StrictMode -Version 3.0
$ErrorActionPreference = "Stop"

if (-not $env:SIGN_CERT_BASE64) { throw "SIGN_CERT_BASE64 is not set (sign.ps1 runs only when a certificate is configured)" }
$timestamp = if ($env:SIGN_TIMESTAMP_URL) { $env:SIGN_TIMESTAMP_URL } else { "http://timestamp.digicert.com" }

function Find-SignTool {
  $kits = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"
  $found = Get-ChildItem -Path $kits -Filter signtool.exe -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Directory.Name -eq "x64" } |
    Sort-Object { $_.Directory.Parent.Name } -Descending |
    Select-Object -First 1
  if (-not $found) { throw "signtool.exe not found under $kits" }
  return $found.FullName
}

# PFX는 러너 임시 폴더에 한 번만 풀고(Inno가 파일마다 이 스크립트를 다시 부름) job이 끝나면 러너와 함께 사라진다.
$tempRoot = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() }
$pfx = Join-Path $tempRoot "evolab-sign.pfx"
if (-not (Test-Path -LiteralPath $pfx)) {
  [IO.File]::WriteAllBytes($pfx, [Convert]::FromBase64String($env:SIGN_CERT_BASE64))
}
$signtool = Find-SignTool

function Invoke-Sign([string[]]$Paths) {
  # signtool은 한 번에 여러 파일을 받는다. 명령줄 길이를 넘지 않게 50개씩.
  for ($i = 0; $i -lt $Paths.Count; $i += 50) {
    $batch = @($Paths[$i..([Math]::Min($i + 49, $Paths.Count - 1))])
    $argv = @("sign", "/f", $pfx, "/fd", "SHA256", "/tr", $timestamp, "/td", "SHA256")
    if ($env:SIGN_CERT_PASSWORD) { $argv += @("/p", $env:SIGN_CERT_PASSWORD) }
    & $signtool @argv @batch
    if ($LASTEXITCODE -ne 0) { throw "signtool sign failed ($LASTEXITCODE)" }
    & $signtool verify /pa /q @batch
    if ($LASTEXITCODE -ne 0) { throw "signtool verify failed ($LASTEXITCODE)" }
  }
}

if ($PSCmdlet.ParameterSetName -eq "File") {
  Invoke-Sign @((Resolve-Path -LiteralPath $Path).Path)
  exit 0
}

$distPath = (Resolve-Path -LiteralPath $Dist).Path
$lister = Join-Path $PSScriptRoot "..\list_unsigned.py"
$unsigned = @(uv run python $lister $distPath --paths | Where-Object { $_.Trim() })
if ($LASTEXITCODE -ne 0) { throw "list_unsigned.py failed ($LASTEXITCODE)" }
# 우리 exe 2개는 서명이 있든 없든 늘 넣는다(목록과 겹치면 하나로).
$ours = @("evolab.exe", "evolab-cli.exe") | ForEach-Object { Join-Path $distPath $_ }
$targets = @($ours + $unsigned | Sort-Object -Unique)
"signing {0} files ({1} unsigned PE + our 2 exe)" -f $targets.Count, $unsigned.Count
Invoke-Sign $targets
uv run python $lister $distPath

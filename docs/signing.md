# 코드 서명 (v3.1)

진화 실험실의 exe와 설치 파일은 아직 **코드 서명이 없습니다**. 인증서가 없기 때문입니다(명세 v3.1: 「코드 서명은 인증서가 있어야
해서 방법만 문서로 남긴다」). 이 문서는 인증서가 생겼을 때 무엇을 어떻게 서명하는지, 그 전까지 사용자에게 무엇을 안내하는지를
적습니다. 실제 비밀 값·인증서 이름은 여기에 쓰지 않습니다.

표시: (문서) = 공개 문서에서 확인한 것, (확인 필요) = 인증서를 살 때 다시 확인할 것. 윈도우 PC에서 직접 재 본 것은 없습니다.

## 1. 서명이 없으면 생기는 일

- **SmartScreen**: 인터넷에서 받은 서명 없는 설치 파일을 처음 열면 「Windows의 PC 보호」 창이 뜹니다. 「추가 정보」를 눌러야
  「실행」 단추가 보입니다. 같은 파일을 받는 사람이 많아져 평판이 쌓이면 줄어들지만, 서명이 없으면 새 버전마다 평판이 처음부터
  다시 쌓입니다(문서).
- **스마트 앱 컨트롤**(윈도우 11): 켜져 있는 PC는 서명이 없고 평판도 없는 프로그램과 DLL을 막을 수 있습니다. 막히면 「추가 정보」
  같은 우회 단추가 없습니다(문서, 확인 필요). exe만 서명하고 배포본 안의 DLL·.pyd를 그대로 두면 프로그램이 뜨다가 막힐 수 있으므로
  배포본의 서명 없는 PE를 모두 서명합니다(2절).
- **백신**: 서명 없는 PyInstaller exe는 오탐이 더 잦다고 알려져 있습니다(사용자 시험 P7에서 Microsoft Defender로 봅니다).
- 프로그램 자체는 서명과 상관없이 똑같이 돕니다. 학습 결과·기록 형식도 바뀌지 않습니다.

## 2. 무엇을 서명하나

| 대상 | 어디서 | 언제 |
| --- | --- | --- |
| `evolab.exe`, `evolab-cli.exe` | `dist\evolab\` | PyInstaller 뒤, ISCC 전 |
| 배포본의 서명 없는 PE(`.exe`·`.dll`·`.pyd`) | `dist\evolab\_internal\` | 같은 때 |
| 설치 파일 `evolab-setup-<버전>.exe` | ISCC가 만들면서 | Inno Setup `SignTool=` |
| 제거 프로그램 `unins000.exe` | ISCC가 만들면서 | Inno Setup `SignedUninstaller=yes` |

- 서명 없는 PE 목록은 빌드마다 `packaging/list_unsigned.py`가 뽑습니다(윈도우 CI의 「Unsigned PE list」 단계가 개수와 이름을 찍음).
  PE 머리의 인증서 표가 있는지만 보며, 서명이 유효한지는 `signtool verify`가 봅니다. 설계 때 잰 CI 배포본은 PE 136개 가운데 25개가
  서명이 없었습니다(파이썬 확장 모듈과 일부 DLL). Qt·파이썬 DLL처럼 이미 서명된 파일은 다시 서명하지 않습니다.
- 휴대용 zip과 Actions artifact도 서명된 `dist\evolab`을 담습니다(서명 단계가 한글 경로 사본·artifact·설치 파일·zip보다 먼저 돎).
- Inno Setup이 설치 파일을 만든 **뒤에** 직접 서명하면 제거 프로그램과 Setup의 임시 사본은 서명되지 않아 보안 프로그램이 막을 수
  있습니다. 그래서 `SignTool=`로 Inno가 서명하게 합니다(Inno Setup 7.1.1 안내, 문서).

## 3. 인증서

- **종류**: 공개 CA의 코드 서명 인증서(OV 또는 EV). 자체 서명 인증서는 SmartScreen 경고를 없애지 못합니다.
- **개인키 보관**: 2023년 6월부터 공개 CA가 내주는 코드 서명 인증서의 개인키는 하드웨어(FIPS 140-2 Level 2 이상 토큰·HSM)나
  클라우드 HSM에만 둘 수 있습니다(CA/브라우저 포럼 코드 서명 기준, 문서). 그래서 예전처럼 `.pfx` 파일 하나를 CI 비밀 값으로 넣는
  방식은 새 인증서에서는 대개 쓸 수 없고, 다음 가운데 하나가 됩니다.
  - **클라우드 서명 서비스**: CA나 클라우드 업체가 HSM을 맡고 CI가 API로 서명을 요청합니다. CI에 맞음.
  - **하드웨어 토큰**: 토큰을 꽂은 PC에서만 서명할 수 있어 GitHub 호스트 러너로는 안 됩니다(자체 러너가 필요).
- **EV와 OV**: EV가 SmartScreen 평판을 바로 준다는 예전 안내는 지금은 맞지 않을 수 있습니다(확인 필요). 값, 발급 자격(개인·단체),
  서비스 이름은 살 때 확인합니다.
- 시각 도장(RFC 3161)을 꼭 넣습니다. 그래야 인증서가 만료된 뒤에도 서명 당시 유효했던 서명이 계속 유효합니다.

## 4. CI에 넣기

윈도우 CI(`.github/workflows/windows.yml`)에는 서명 자리가 이미 있고, 비밀 값이 없으면 건너뜁니다(W11: 단계가 「skipped」).

| 무엇 | 자리 |
| --- | --- |
| 저장소 비밀 값 `SIGN_CERT_BASE64` | 코드 서명 인증서 PFX를 base64로(PFX를 쓸 수 있는 인증서일 때) |
| 저장소 비밀 값 `SIGN_CERT_PASSWORD` | PFX 암호(없으면 비워 둠) |
| 저장소 변수 `SIGN_TIMESTAMP_URL` | 시각 도장 서버(없으면 `sign.ps1`의 기본값) |
| package job `env.SIGN_ENABLED` | `secrets.SIGN_CERT_BASE64 != ''`의 참·거짓. 단계의 `if`는 secrets를 바로 못 보므로 이것을 봄 |
| 「Sign (only when a certificate is configured)」 단계 | `packaging/installer/sign.ps1 -Dist dist\evolab`: 우리 exe 2개 + 서명 없는 PE를 `signtool sign /fd SHA256 /tr <시각 도장> /td SHA256`으로 서명하고 `signtool verify /pa`로 확인 |
| 「Build installers」 단계 | `make_installer.py --sign`이 `.iss`에 `SignTool=evolabsign`·`SignedUninstaller=yes`를 켜고, ISCC에 `/Sevolabsign=pwsh … sign.ps1 -Path $f`를 줌 |

- 비밀 값은 서명하는 두 단계의 `env`로만 넘깁니다. 자체 점검·테스트·오류 보고 묶음을 만드는 단계에는 없습니다.
- 클라우드 서명 서비스를 쓰면 `sign.ps1`의 `Invoke-Sign` 함수만 그 서비스의 도구(예: `signtool sign /dlib …`처럼 서비스가 주는
  방식)로 바꾸고, 필요한 자격 증명을 같은 방식으로 비밀 값에 넣습니다. 워크플로의 단계와 순서는 그대로입니다.
- 서명 순서: PyInstaller → 서명 → 한글 경로 사본·자체 점검 → ISCC(설치 파일·제거 프로그램 서명) → 휴대용 zip·SHA256SUMS.
  체크섬은 서명된 파일로 계산합니다.

## 5. 확인

- `signtool verify /pa /v dist\evolab\evolab.exe` (설치 파일·`unins000.exe`도 같은 방법)
- PowerShell: `Get-AuthenticodeSignature .\evolab-setup-3.1.0.exe | Format-List`
- `uv run python packaging/list_unsigned.py dist\evolab`: 서명한 뒤에는 「unsigned: 0」이어야 합니다(`sign.ps1`이 끝에 찍음).
- 사람 확인: 브라우저로 받은 설치 파일을 열었을 때 게시자 이름이 보이는지(P4).

## 6. 서명 전까지 사용자 안내

도움말 「설치와 제거」 쪽(strings.py `HELP_PAGES["install"]`)과 휴대용 zip의 「설치 없이 쓰기.txt」(`PORTABLE_README`)가
안내합니다. 같은 글이 GitHub 릴리스 본문에도 들어갑니다(`packaging/installer/release_files.py`).

- 설치 파일을 처음 열면 「Windows의 PC 보호」 창이 뜰 수 있다 → 「추가 정보」 → 「실행」.
- zip 판은 풀기 전에 zip 파일 속성에서 「차단 해제」를 켜면 실행할 때마다 나오는 경고가 줄어든다.
- 프로그램은 인터넷으로 새 버전을 확인하지 않는다. 받은 파일이 맞는지는 릴리스의 `SHA256SUMS.txt`와
  `Get-FileHash .\evolab-setup-3.1.0.exe`로 비교할 수 있다.

# 진화 실험실 (evolab)

직접 만든 2D 실험 환경에서 진화 전략(CMA-ES)이 해법을 찾아가는 과정을 보는 윈도우 데스크톱 프로그램입니다. 설계와 진행 단계는 [docs/spec.md](docs/spec.md)에 있습니다.

## 개발 환경에서 실행

[uv](https://docs.astral.sh/uv/)와 파이썬 3.12가 필요합니다.

```powershell
uv sync                  # 의존성 설치
uv run evolab            # 프로그램 창 열기 (메뉴 "직접 조종"에서 착륙선을 몰아 볼 수 있음)
uv run evolab play environments/lander.evoenv            # 착륙선 직접 조종 창
uv run evolab play environments/lander.evoenv --verify 기록.evoplay   # 입력 기록을 다시 돌려 결과 비교
uv run evolab selftest   # 설치 상태 점검 (프로세스, 물리, 학습 라이브러리, 내장 예제, 창)
uv run pytest            # 빠른 테스트
uv run pytest -m slow    # 느린 학습 테스트 (M2부터)
```

## 직접 조종 (착륙선)

- 위쪽 화살표 또는 W: 주 추진기
- 왼쪽 화살표 또는 A / 오른쪽 화살표 또는 D: 측면 추진기 (그쪽으로 밀림)
- R 다시 시작, N 새 시드, Space 일시정지, 마우스 휠 확대
- 두 다리를 패드(초록 선) 위에 대고 속력 0.5m/s 미만으로 1초 버티면 성공입니다.
- "입력 기록 저장"으로 남긴 기록은 "입력 기록 재생"이나 `--verify`로 다시 돌려 같은 결과가 나오는지 확인할 수 있습니다.

## exe 받기

`windows` GitHub Actions가 푸시할 때마다 윈도우에서 테스트하고 exe를 묶습니다. 저장소의 Actions 탭에서 최근 실행을 열고 Artifacts의 `evolab-windows`를 내려받아 압축을 풀면 exe 두 개가 있습니다.

- `evolab.exe`: 더블클릭해서 여는 프로그램 (창 모드)
- `evolab-cli.exe`: 명령줄용. 출력과 종료 코드가 필요한 `selftest`, `train`, `bench`는 이쪽으로 실행합니다

```powershell
.\evolab-cli.exe selftest   # 점검 결과를 화면에 보여 주고 %LOCALAPPDATA%\evolab\logs\selftest.log에도 남깁니다
```

직접 묶으려면 `uv run pyinstaller packaging/evolab.spec --noconfirm` (결과: `dist\evolab\`).

## 폴더

```text
src/evolab/     프로그램 코드 (app: 화면, world: 물리, engine: 학습 엔진, ...)
environments/   내장 예제 환경 파일 (.evoenv)
tests/          테스트
packaging/      exe 묶기 설정
docs/spec.md    설계 명세서
```

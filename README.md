# 진화 실험실 (evolab)

직접 만든 2D 실험 환경에서 진화 전략(CMA-ES)이 해법을 찾아가는 과정을 보는 윈도우 데스크톱 프로그램입니다. 설계와 진행 단계는 [docs/spec.md](docs/spec.md)에 있습니다.

## 개발 환경에서 실행

[uv](https://docs.astral.sh/uv/)와 파이썬 3.12가 필요합니다.

```powershell
uv sync                  # 의존성 설치
uv run evolab            # 프로그램 창 열기
uv run evolab selftest   # 설치 상태 점검 (프로세스, 물리, 학습 라이브러리, 창)
uv run pytest            # 빠른 테스트
uv run pytest -m slow    # 느린 학습 테스트 (M2부터)
```

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

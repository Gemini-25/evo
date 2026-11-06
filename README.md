# 진화 실험실 (evolab)

직접 만든 2D 실험 환경에서 진화 전략(CMA-ES, 유전 알고리즘, 담금질)이 해법을 찾아가는 과정을 보는 윈도우 데스크톱
프로그램입니다. 설계는 [docs/spec.md](docs/spec.md), 마일스톤별 보고와 측정값은 [docs/reports/](docs/reports/)에 있습니다.

## 받아서 쓰기 (파이썬 없이)

1. 저장소 Actions 탭 → 최근 `windows` 실행 → Artifacts의 `evolab-windows`를 내려받아 압축을 풉니다.
2. `evolab.exe`를 더블클릭합니다. 처음 열면 내장 예제가 `문서\진화실험실\environments`로 복사됩니다.
3. **실험실** 탭에서 환경(달 착륙선, 트랙 자동차, 언덕 넘는 자동차)을 고르고 **새로 시작**.

- `evolab.exe`: 창 프로그램. `evolab-cli.exe`: 명령줄용(출력과 종료 코드가 필요한 `selftest`, `train`, `judge`, `bench`,
  `bundle`, `reproduce`, `csv`).
- 노트북은 전원을 연결하고 '최고 성능' 전원 모드에서 학습하기를 권합니다. 평가 워커는 기본으로 물리 코어 − 1개입니다.

## 화면

| 탭 | 하는 일 |
| --- | --- |
| 에디터 | 환경 파일 만들기·고치기: 부품 팔레트, 캔버스, 속성(진화 대상 체크), 아래 탭(세계, 사건·종료, 점수, 랜덤 범위, 생성기, 평가, 뇌, 검사 결과), 되돌리기, 복사·붙여넣기, 직접 조종, 멈춘 실험에 고친 환경 보내기 |
| 실험실 | 학습: 관전(집단 겹쳐 보기, 1·2·4·8배속)/고속(챔피언만 가끔), 멈춤·한 세대만·이어서, 멈춘 상태에서 정지 조건·환경 바꾸기, 세대 타임라인, 1세대 대표와 지금 챔피언 겹쳐 보기, 개체 클릭 들여다보기(센서·뉴런), 통계 대시보드, 영상(GIF) 저장 |
| 비교 | 실행 여러 개를 같은 그래프에(가로축 누적 시험 횟수, 세로축 비교 평가 세트 점수), 반복 실행(시드만 바꿔 N번), 표(최종 점수, 목표 도달까지 시험 횟수, 시간) |
| 기록 | 실행 목록, 열기, 이어하기(예전 체크포인트면 새 실행으로 분기), 복제해서 새 실행, 삭제, 메모 |

메뉴 "직접 조종"에서 키보드로 직접 몰아 볼 수 있습니다(착륙선: ↑/W 주 추진기, ←→/AD 측면 추진기 · 트랙 자동차:
↑↓/WS 가속, ←→/AD 조향 · 언덕차: ←→/AD 바퀴). R 다시 시작, N 새 시드, Space 일시정지.

## 명령줄

```powershell
evolab-cli.exe train <환경.evoenv> --seed 1 --max-time 900        # 화면 없이 15분 학습 (문서\진화실험실\runs 아래 실행 폴더)
evolab-cli.exe train <환경> --method ga                           # 유전 알고리즘 (sa = 담금질)
evolab-cli.exe train --resume <실행 폴더> --max-time 1800          # 이어하기
evolab-cli.exe judge <실행 폴더> [--seeds 10]                      # 챔피언을 판정 시드로 평가(완료 기준)
evolab-cli.exe bundle <실행 폴더>                                  # 재현 묶음(zip)
evolab-cli.exe reproduce <묶음.zip>                                # 다시 돌리고 원래 통계와 비교
evolab-cli.exe csv <실행 폴더>                                     # 세대별 통계 CSV
evolab-cli.exe bench <환경>                                        # 성능 측정
evolab-cli.exe selftest                                            # 설치 상태 점검
```

내장 예제는 exe 폴더의 `_internal\environments\`에 있습니다. 학습 중 Ctrl+C 한 번이면 이번 세대를 마치고
체크포인트를 남기고 멈춥니다.

## 개발 환경

[uv](https://docs.astral.sh/uv/)와 파이썬 3.12가 필요합니다.

```powershell
uv sync                  # 의존성 설치 (uv.lock 고정)
uv run evolab            # 창
uv run evolab train environments/lander.evoenv --max-time 900
uv run pytest            # 빠른 테스트
uv run pytest -m slow    # 느린 학습 테스트
uv run pyinstaller packaging/evolab.spec --noconfirm   # exe 묶기 (dist\evolab\)
```

## 폴더

```text
src/evolab/       프로그램 코드
  app/            화면 (editor, lab, compare, records, play, export, strings.py = 화면 문구 전부)
  engine/         학습 엔진 (Trainer, 워커 풀, 학습 프로세스 세션, 시드, 재생)
  learners/       CMA-ES, 유전 알고리즘, 담금질 (ask/tell)
  world/physics2d 환경 런타임 (pymunk)
  spec/           환경 파일 형식, 검사, 수식 기호
  generators/     지형, 언덕, 트랙 생성기
  runs/           실행 폴더 저장소, 목록, 재현 묶음
environments/     내장 예제 (lander, track_car, hill_car)
tests/            테스트
packaging/        exe 묶기 설정, 아이콘, 라이선스 모으기
docs/             설계 명세서, 마일스톤 보고
```

v6.1(「자라는 뇌」 L 묶음 최종 코드, 6.1.0 작업 트리) 프로그램으로 만든 작은 자라는 뇌 실행 기록과 그 재현 묶음. 다음 판의 프로그램이
v6.1 자라는 실행(학습 방법 "grow", 체크포인트 형식 2 안의 학습 방법 상태 형식 1, 통계 열 brain_neurons·brain_links·grow)을 그대로
열고, 이어 가고, 분기해 다시 돌리고, 재현하는지 검사한다(v6 설계 7.1, 테스트 T55 tests/test_v61_fixture.py).
테스트는 이 폴더를 임시 폴더로 복사해서 쓰고, 여기는 고치지 않는다.

만든 설정: 달 착륙선, CMA-ES · 자라는 뇌(--method grow, 처음 뉴런 2, 3세대마다 두 배), 실행 시드 11, 개체 8, 시험 2, 검증 주기 16
(매 세대), 대각 공분산 문턱 100(켜진 좌표 29·55는 전체 공분산, 107·211은 대각: 체크포인트를 작게), 자동 재시작·예선 끔(새 실행 기본),
체크포인트 주기 10(기본), 워커 0. 6세대까지 학습하고 멈춘 뒤, 이어서 12세대까지 학습했다(사건: start, stop 6, resume 6, stop 12).

들어 있는 것:
- run/        실행 폴더. 만든 그대로이고 폴더 이름만 run으로 바꿨다(원래 이름 「2026-11-13_0943_달 착륙선_CMA-ES · 자라는 뇌」).
              크기를 줄이려고 10세대 체크포인트(주기 체크포인트)만 지웠다. 남은 체크포인트는 6, 12세대(모두 형식 2).
- bundle.zip  run/에서 v6.1 프로그램이 만든 재현 묶음(bundle 명령, 10세대 체크포인트를 지우기 전).

자람은 정확히 세 번: 3·6·9세대 줄의 grow = 1(그 줄의 brain_neurons = 자라기 전 2·4·8), 뉴런 2 → 4 → 8 → 16, 연결 24 → 48 → 96 → 192.
6세대 체크포인트는 두 번째 자람 직후(켜진 뉴런 8, 자란 횟수 2), 12세대 체크포인트는 다 자란 뒤(16, 3). 6세대 체크포인트에서 분기해
12세대까지 다시 돌리면 세 번째 자람(9세대)을 지나간다. 관찰 기록(obs_ 키)은 모든 세대 대표에 있다.

만든 기계: 리눅스 x86_64, python 3.12.3, numpy 2.5.4, pymunk 7.3.1, cma 4.5.0 (run.json의 platform).
숫자 비교(통계 줄 다시 계산, 묶음 재현 compare_stats)는 같은 플랫폼에서만 한다.

만든 명령(저장소 /home/user/evo에서, 작업 트리 = 1e59b26 + v6.1 작업 중 변경):

WORK=/tmp/claude-0/-home-user-evo/6ff254e5-e86e-5068-a687-5ac6a7065af0/scratchpad/v61/fix/fixture_build
FIX=tests/data/v61_grow_run
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 5 .venv/bin/python -m evolab train environments/lander.evoenv --out "$WORK" --method grow --seed 11 --popsize 8 \
    --trials 2 --validate-every 16 --grow-every 3 --diagonal-threshold 100 --max-generations 6 --workers 0
RUN=$(ls -d "$WORK"/*/)
nice -n 5 .venv/bin/python -m evolab train --resume "$RUN" --max-generations 12 --workers 0
mkdir -p "$FIX"
cp -r "$RUN" "$FIX/run"
nice -n 5 .venv/bin/python -m evolab bundle "$FIX/run" --out "$FIX/bundle.zip"
rm "$FIX/run/checkpoints/gen_000010.pkl"

확인(v6.1 통합, 2026-11-10): 최종 코드(L·C 묶음 마무리 뒤, 학습 방법 처음 채우기 규칙 바꾼 뒤)로 위 명령을 새 폴더에 그대로 다시
돌려 비교했다: stats.csv(시간 열 빼고) 12줄, 챔피언·세대 대표 npz 17개(모든 배열), run.json config, 사건 순서, 체크포인트
6·12의 config·학습 상태(elapsed 빼고)·학습 방법 상태(pycma 객체는 logger·timer·_stopdict 말고 모든 속성), 재현 묶음의 파일 목록·
환경·통계가 모두 같아 자료는 바꾸지 않았다.

다시 만듦(v6.1 고침 묶음, 2026-11-13): 자라는 실행의 통계 열 diversity(유전자 표준편차 평균)를 켜진 좌표만의 평균으로 고쳤다
(engine/trainer.py: 꺼진 좌표(늘 0)까지 평균해 자랄 때마다 값이 두 배쯤 뛰던 결함). 통계 열 값이 바뀌므로 고친 코드로 위 명령을
그대로 다시 돌려 run/과 bundle.zip을 통째로 바꿨다. 옛 자료와 비교한 결과: stats.csv는 diversity 열의 1~9세대 줄만 다르고
(1세대 0.0636 → 0.4627 … 9세대 0.1766 → 0.3483. 다 자란 10~12세대 줄은 같다), 다른 열(시간 열 빼고)·챔피언·세대 대표 npz 17개
(파일 바이트까지)·run.json config·사건 순서·체크포인트 6·12의 config·학습 상태·학습 방법 상태는 모두 같다. 달라진 것은 만든
시각(run.json created, events.jsonl time, 체크포인트의 elapsed·pycma 타이머)과 platform.os의 커널 이름뿐이다.

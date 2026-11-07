v1.1(커밋 420c662) 프로그램으로 만든 작은 실행 기록과 그 재현 묶음. v1.2 이후 프로그램이 v1.1 실행(체크포인트
형식 2, 학습 방법 열 learner·restarts·switch 등이 있는 통계)을 그대로 열고, 이어 가고, 재현하는지 검사한다(설계 D33, T12).
테스트는 이 폴더를 임시 폴더로 복사해서 쓰고, 여기는 고치지 않는다.

만든 설정: 달 착륙선, CMA-ES(대각 공분산: --diagonal-threshold 100, 체크포인트를 작게), 실행 시드 11, 개체 8, 시험 2,
검증 주기 16(매 세대), 자동 재시작 48회(처음 상태에서), 예선 없음, 체크포인트 주기 10(기본), 워커 0.
6세대까지 학습하고 멈춘 뒤, 이어서 12세대까지 학습했다(사건: start, stop 6, resume 6, stop 12).

들어 있는 것:
- run/        실행 폴더. 만든 그대로이고 폴더 이름만 run으로 바꿨다. 크기를 줄이려고 테스트에 필요 없는 10세대
              체크포인트(주기 체크포인트, 재시작 뒤)만 지웠다(rm run/checkpoints/gen_000010.pkl). 남은 체크포인트는
              6, 12세대(모두 형식 2).
- bundle.zip  run/에서 v1.1 프로그램이 만든 재현 묶음(bundle 명령).

자동 재시작은 정확히 한 번: 8세대 줄의 switch = 1(처음 상태에서 새로 시작), 9세대부터 learner = 1, restarts = 1.
6세대 체크포인트는 재시작 전(restarts 0, 학습 방법 1개), 12세대 체크포인트는 재시작 뒤(restarts 1, 학습 방법 2개).
6세대 체크포인트에서 분기해 12세대까지 다시 돌리면 재시작을 지나간다.

만든 기계: 리눅스 x86_64, python 3.12.3, numpy 2.5.4, pymunk 7.3.1, cma 4.5.0 (run.json의 platform).
숫자 비교(통계 줄 다시 계산, 묶음 재현 compare_stats)는 같은 플랫폼에서만 한다.

만든 명령(저장소 /home/user/evo에서, SNAP = v1.1 안정 사본, 커밋 420c662):

SNAP=/tmp/claude-0/-home-user-evo/6ff254e5-e86e-5068-a687-5ac6a7065af0/scratchpad/v11/wt_v11
WORK=/tmp/claude-0/-home-user-evo/6ff254e5-e86e-5068-a687-5ac6a7065af0/scratchpad/v12/stage0/fixture_build
FIX=/home/user/evo/tests/data/v11_run_restart
export PYTHONPATH=$SNAP/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -c "import evolab; assert evolab.__file__.startswith('$SNAP/src/'), evolab.__file__"
nice -n 5 .venv/bin/python -m evolab train "$SNAP/environments/lander.evoenv" --out "$WORK" --method cmaes --seed 11 \
    --popsize 8 --trials 2 --validate-every 16 --restart 48 --diagonal-threshold 100 --max-generations 6 --workers 0
RUN=$(ls -d "$WORK"/*/)
nice -n 5 .venv/bin/python -m evolab train --resume "$RUN" --max-generations 12 --workers 0
mkdir -p "$FIX"
mv "$RUN" "$FIX/run"
nice -n 5 .venv/bin/python -m evolab bundle "$FIX/run" --out "$FIX/bundle.zip"
rm "$FIX/run/checkpoints/gen_000010.pkl"

(environments/lander.evoenv는 SNAP과 저장소가 같은 내용이다.)

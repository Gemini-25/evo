"""exe 진입점 (evolab.exe와 evolab-cli.exe 공통).

freeze_support()는 다른 무엇보다 먼저 불러야 한다. 묶인 exe에서 spawn으로 만든 자식
프로세스(학습 프로세스, 워커)는 같은 exe를 다시 실행하는데, 이 호출이 그 경우를 알아보고
자식 일만 하게 만든다. 그 전에 표준 출력·오류가 없는 경우(창 모드 exe)를 먼저 메운다.
"""

import multiprocessing

if __name__ == "__main__":
    from evolab.startup import ensure_std_streams, limit_blas_threads

    ensure_std_streams()
    # 치명적 오류 스택을 logs/faults.log에(v3 설계 D28). freeze_support() 전이라 spawn 자식(학습 프로세스·워커)도 거친다.
    from evolab.diagnostics import enable_fault_log

    enable_fault_log()
    limit_blas_threads()
    multiprocessing.freeze_support()
    from evolab.cli import main

    raise SystemExit(main())

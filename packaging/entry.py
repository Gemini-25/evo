"""exe 진입점.

freeze_support()는 다른 무엇보다 먼저 불러야 한다. 묶인 exe에서 spawn으로 만든 자식
프로세스(학습 프로세스, 워커)는 같은 exe를 다시 실행하는데, 이 호출이 그 경우를 알아보고
자식 일만 하게 만든다.
"""

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from evolab.cli import main

    raise SystemExit(main())

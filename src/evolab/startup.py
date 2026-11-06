"""프로세스 시작 직후 할 일. exe 진입점과 `python -m evolab`가 freeze_support() 전에 부른다."""

from __future__ import annotations

import os
import sys


def ensure_std_streams() -> None:
    """창 없는 exe(또는 pythonw)에서는 sys.stdout/sys.stderr가 None이다.

    그대로 두면 print나 multiprocessing의 오류 출력이 AttributeError를 내고, 자식 프로세스의
    진짜 오류가 사라진다. 표준 출력은 버리고, 표준 오류는 로그 폴더의 stderr.log에 덧붙인다.
    spawn으로 만든 자식도 같은 진입점을 거치므로 모든 프로세스에 적용된다.
    """
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    if sys.stderr is None:
        try:
            from evolab import paths

            log_dir = paths.log_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            sys.stderr = open(log_dir / "stderr.log", "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        except OSError:
            sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115


def limit_blas_threads() -> None:
    """numpy를 불러오기 전에 BLAS 스레드를 1개로 묶는다(명세: 모든 프로세스에서 BLAS 스레드 1개).

    워커끼리 코어를 나눠 쓰고, 화면 프로세스의 재생도 학습과 같은 계산 경로를 타야 하기 때문이다.
    자식 프로세스는 이 환경 변수를 물려받는다. 사용자가 미리 정해 둔 값은 바꾸지 않는다.
    """
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")

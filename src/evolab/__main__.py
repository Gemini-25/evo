"""`python -m evolab` 진입점."""

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

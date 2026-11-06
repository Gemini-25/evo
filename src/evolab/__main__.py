"""`python -m evolab` 진입점."""

import multiprocessing

if __name__ == "__main__":
    from evolab.startup import ensure_std_streams, limit_blas_threads

    ensure_std_streams()
    limit_blas_threads()
    multiprocessing.freeze_support()
    from evolab.cli import main

    raise SystemExit(main())

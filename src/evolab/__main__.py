"""`python -m evolab` 진입점."""

import multiprocessing

if __name__ == "__main__":
    from evolab.startup import ensure_std_streams

    ensure_std_streams()
    multiprocessing.freeze_support()
    from evolab.cli import main

    raise SystemExit(main())

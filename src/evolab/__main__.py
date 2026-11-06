"""`python -m evolab` 진입점."""

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from evolab.cli import main

    raise SystemExit(main())

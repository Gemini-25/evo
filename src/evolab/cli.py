"""명령줄 진입점: `evolab`, `evolab train`, `evolab bench`, `evolab play`, `evolab selftest`.

이 모듈은 맨 위에서 Qt를 불러오지 않는다. spawn 방식의 자식 프로세스가 주 모듈을 다시
불러올 때 워커마다 Qt가 올라오지 않게 하기 위해서다.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from evolab.app import strings as S


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evolab", description=S.CLI_DESCRIPTION)
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help=S.CLI_HELP_TRAIN)
    train.add_argument("env", type=Path, nargs="?", default=None, help=S.CLI_HELP_ENV_FILE)
    train.add_argument("--method", default=None, choices=["cmaes"], help=S.CLI_HELP_METHOD)
    train.add_argument("--seed", type=int, default=None, help=S.CLI_HELP_SEED)
    train.add_argument("--resume", type=Path, default=None, help=S.CLI_HELP_RESUME)
    train.add_argument("--out", type=Path, default=None, help=S.CLI_HELP_OUT)
    train.add_argument("--workers", type=int, default=None, help=S.CLI_HELP_TRAIN_WORKERS)
    train.add_argument("--popsize", type=int, default=None, help=S.CLI_HELP_POPSIZE)
    train.add_argument("--sigma", type=float, default=None, help=S.CLI_HELP_SIGMA)
    train.add_argument("--trials", type=int, default=None, help=S.CLI_HELP_TRIALS)
    train.add_argument("--max-time", type=float, default=None, help=S.CLI_HELP_MAX_TIME)
    train.add_argument("--max-generations", type=int, default=None, help=S.CLI_HELP_MAX_GENS)
    train.add_argument("--max-evaluations", type=int, default=None, help=S.CLI_HELP_MAX_EVALS)
    train.add_argument("--target", type=float, default=None, help=S.CLI_HELP_TARGET)
    train.add_argument("--stagnation", type=int, default=None, help=S.CLI_HELP_STAGNATION)
    train.add_argument("--validate-every", type=int, default=None, help=S.CLI_HELP_VALIDATE_EVERY)

    bench = sub.add_parser("bench", help=S.CLI_HELP_BENCH)
    bench.add_argument("env", type=Path, help=S.CLI_HELP_ENV_FILE)
    bench.add_argument("--workers", default=None, help=S.CLI_HELP_BENCH_WORKERS)
    bench.add_argument("--generations", type=int, default=3, help=S.CLI_HELP_BENCH_GENS)
    bench.add_argument("--popsize", type=int, default=32, help=S.CLI_HELP_POPSIZE)

    judge = sub.add_parser("judge", help=S.CLI_HELP_JUDGE)
    judge.add_argument("run", type=Path, help=S.CLI_HELP_RUN_DIR)
    judge.add_argument("--seeds", type=int, default=100, help=S.CLI_HELP_JUDGE_SEEDS)
    judge.add_argument("--generation", type=int, default=None, help=S.CLI_HELP_JUDGE_GEN)
    judge.add_argument("--workers", type=int, default=None, help=S.CLI_HELP_TRAIN_WORKERS)

    play = sub.add_parser("play", help=S.CLI_HELP_PLAY)
    play.add_argument("env", type=Path, help=S.CLI_HELP_ENV_FILE)
    play.add_argument("--seed", type=int, default=None, help=S.CLI_HELP_PLAY_SEED)
    play.add_argument("--replay", type=Path, default=None, help=S.CLI_HELP_PLAY_REPLAY)
    play.add_argument("--verify", type=Path, default=None, help=S.CLI_HELP_PLAY_VERIFY)

    selftest = sub.add_parser("selftest", help=S.CLI_HELP_SELFTEST)
    selftest.add_argument("--log", type=Path, default=None, help=S.CLI_HELP_LOG)
    selftest.add_argument("--workers", type=int, default=None, help=S.CLI_HELP_WORKERS)
    selftest.add_argument("--no-gui", action="store_true", help=S.CLI_HELP_NO_GUI)
    return parser


_NOT_YET: dict[str, str] = {}


def _configure_streams() -> None:
    """한글 출력 때문에 죽지 않게 한다.

    윈도우에서 출력을 파이프로 받으면 인코딩이 cp1252 같은 로캘 기본값이라 한글을 못 쓴다.
    파이프나 파일이면 UTF-8로 바꾸고, 그래도 못 쓰는 글자는 대체 문자로 바꾼다.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            if stream.isatty():
                reconfigure(errors="replace")
            else:
                reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def main(argv: Sequence[str] | None = None) -> int:
    from evolab.startup import limit_blas_threads

    limit_blas_threads()  # numpy를 불러오기 전에
    _configure_streams()
    args = _build_parser().parse_args(argv)

    if args.command is None:
        from evolab.app.main import run_gui

        return run_gui()

    if args.command == "selftest":
        from evolab.selftest import run_selftest

        return run_selftest(log_path=args.log, workers=args.workers, check_gui=not args.no_gui)

    if args.command == "play":
        return _play(args.env, args.seed, args.replay, args.verify)

    if args.command in ("train", "bench", "judge"):
        from evolab.engine import commands

        return {"train": commands.train_command, "bench": commands.bench_command, "judge": commands.judge_command}[
            args.command
        ](args)

    milestone = _NOT_YET[args.command]
    _print_err(S.CLI_NOT_IMPLEMENTED.format(command=args.command, milestone=milestone))
    return 2


def _play(env_path: Path, seed: int | None, replay_path: Path | None, verify_path: Path | None) -> int:
    from evolab.spec.compiled import EnvFileError
    from evolab.spec.io import env_hash, load_env
    from evolab.world.physics2d.episode import EpisodeBuildError

    try:
        spec = load_env(env_path)
    except EnvFileError as exc:
        _print_err(str(exc))
        return 2

    if verify_path is not None:
        # 종료 코드: 0 같음, 1 다름, 2 기록이나 환경을 쓸 수 없음
        from evolab.app.play.record import verify_record

        try:
            same, result, record = verify_record(spec, verify_path)
        except (OSError, ValueError, EpisodeBuildError) as exc:
            _print_err(S.PLAY_RECORD_BAD.format(reason=exc))
            return 2
        if record.env_hash != env_hash(spec.file):
            _print_err(S.PLAY_REPLAY_ENV_DIFF)
        message = (S.CLI_VERIFY_SAME if same else S.CLI_VERIFY_DIFF).format(
            recorded=record.score, replayed=result.score, steps=result.steps
        )
        if sys.stdout is not None:
            print(message)
        return 0 if same else 1

    from evolab.app.main import create_application
    from evolab.app.play.window import PlayWindow

    app = create_application()
    try:
        window = PlayWindow(spec, seed)
    except EpisodeBuildError as exc:
        _print_err(S.PLAY_BUILD_FAILED.format(seed=seed, reason=exc))
        return 2
    if replay_path is not None:
        window.play_record(replay_path)
    window.show()
    return app.exec()


def _print_err(message: str) -> None:
    # --windowed로 묶은 exe에서는 sys.stderr가 None이다.
    if sys.stderr is not None:
        print(message, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

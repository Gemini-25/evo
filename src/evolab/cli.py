"""명령줄 진입점: `evolab`, `evolab train`, `evolab bench`, `evolab play`, `evolab selftest`.

이 모듈은 맨 위에서 Qt를 불러오지 않는다. spawn 방식의 자식 프로세스가 주 모듈을 다시
불러올 때 워커마다 Qt가 올라오지 않게 하기 위해서다.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, NoReturn

from evolab.app import strings as S

_korean_thread: int | None = None
"""argparse 문장을 한국어로 바꿔 끼운 스레드. 다른 스레드가 그 사이 argparse를 써도 원래 문장을 받는다."""
_saved_gettext: tuple[Callable[[str], str], Callable[[str, str, int], str]] = (
    argparse._, argparse.ngettext)  # type: ignore[attr-defined]


def _korean_gettext(message: str) -> str:
    if threading.get_ident() != _korean_thread:
        return _saved_gettext[0](message)
    return S.RV2_ARGPARSE_TEXT.get(message) or _saved_gettext[0](message)


def _korean_ngettext(singular: str, plural: str, n: int) -> str:
    if threading.get_ident() != _korean_thread:
        return _saved_gettext[1](singular, plural, n)
    return S.RV2_ARGPARSE_TEXT.get(singular if n == 1 else plural) or _saved_gettext[1](singular, plural, n)


@contextlib.contextmanager
def _korean_argparse() -> Iterator[None]:
    """이 블록 안에서 argparse 자신의 문장(사용법, 도움말 제목, 인수 오류)을 strings.py의 한국어로 바꾼다.

    argparse는 문장을 만들 때마다 모듈의 _와 ngettext(gettext)를 찾는다. 우리 파서가 일하는 동안만 바꿔 끼우고
    끝나면(오류로 끝내도) 되돌려서, argparse를 쓰는 다른 코드에는 남지 않는다. 안쪽 호출(하위 명령 파서)은
    바깥 것이 되돌린다.
    """
    global _korean_thread, _saved_gettext
    if argparse._ is _korean_gettext:  # type: ignore[attr-defined]
        yield
        return
    _saved_gettext = (argparse._, argparse.ngettext)  # type: ignore[attr-defined]
    _korean_thread = threading.get_ident()
    argparse._, argparse.ngettext = _korean_gettext, _korean_ngettext  # type: ignore[attr-defined]
    try:
        yield
    finally:
        argparse._, argparse.ngettext = _saved_gettext  # type: ignore[attr-defined]
        _korean_thread = None


class _Parser(argparse.ArgumentParser):
    """인수 오류를 한국어 문장으로 알리고 종료 코드 2로 끝낸다(하위 명령 파서도 이 클래스를 쓴다).

    argparse가 직접 만드는 문장(필요한 인수, 알 수 없는 인수, 고를 수 없는 값, 사용법·도움말 제목)도
    _korean_argparse로 한국어가 된다: 파서를 만들 때(도움말 제목), 읽을 때, 사용법·도움말을 만들 때.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        with _korean_argparse():
            super().__init__(*args, **kwargs)

    def parse_args(self, args: Any = None, namespace: Any = None) -> Any:
        with _korean_argparse():
            return super().parse_args(args, namespace)

    def parse_known_args(self, args: Any = None, namespace: Any = None) -> Any:
        with _korean_argparse():
            return super().parse_known_args(args, namespace)

    def format_usage(self) -> str:
        with _korean_argparse():
            return super().format_usage()

    def format_help(self) -> str:
        with _korean_argparse():
            return super().format_help()

    def _check_value(self, action: argparse.Action, value: Any) -> None:
        """고를 수 없는 값 오류를 파이썬 버전과 상관없이 같은 문장으로(3.12.8·3.13부터 argparse는 고를 수 있는 값을
        따옴표 없이 쓴다). 값과 목록은 repr로 적는다."""
        if action.choices is not None and value not in action.choices:
            choices = ", ".join(repr(c) for c in action.choices)
            text = S.RV2_ARGPARSE_TEXT["invalid choice: %(value)r (choose from %(choices)s)"]
            raise argparse.ArgumentError(action, text % {"value": value, "choices": choices})

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)  # stderr가 None(창 exe)이어도 argparse가 조용히 넘어간다
        self.exit(2, S.RVM_CLI_BAD_ARGS.format(detail=message) + "\n")


def _int_arg(minimum: int | None = None) -> Callable[[str], int]:
    """정수 인수. 정수가 아니거나 minimum보다 작으면 한국어 오류(학습을 시작하기 전에 거른다)."""

    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(S.RVM_CLI_NOT_INT.format(value=text)) from None
        if minimum is not None and value < minimum:
            raise argparse.ArgumentTypeError(S.RVM_CLI_TOO_SMALL.format(minimum=minimum, value=value))
        return value

    return parse


def _float_arg(text: str) -> float:
    """실수 인수(범위는 학습 설정 검사가 본다)."""
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(S.RVM_CLI_NOT_NUMBER.format(value=text)) from None


def _worker_list(text: str) -> list[int]:
    """'1,2,4' 같은 워커 수 목록. 1 이상의 정수가 하나 이상이 아니면 argparse.ArgumentTypeError(한국어)."""
    try:
        values = [int(x) for x in str(text).split(",")]
    except ValueError:
        values = []
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError(S.RVM_CLI_BAD_WORKER_LIST.format(value=text))
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="evolab", description=S.CLI_DESCRIPTION)
    # 이름을 정해 두지 않으면 잘못된 명령 오류에 "command"라는 영어 이름이 나온다
    sub = parser.add_subparsers(dest="command", metavar=S.RV2_CLI_COMMAND_METAVAR)

    train = sub.add_parser("train", help=S.CLI_HELP_TRAIN)
    train.add_argument("env", type=Path, nargs="?", default=None, help=S.CLI_HELP_ENV_FILE)
    train.add_argument("--method", default=None, choices=["cmaes", "ga", "sa"], help=S.CLI_HELP_METHOD)
    train.add_argument("--seed", type=_int_arg(), default=None, help=S.CLI_HELP_SEED)
    train.add_argument("--resume", type=Path, default=None, help=S.CLI_HELP_RESUME)
    train.add_argument("--out", type=Path, default=None, help=S.CLI_HELP_OUT)
    # 범위는 학습 설정 검사(TrainConfig.validate)가 한국어로 알린다. 워커 수 0은 이 프로세스 안에서 평가.
    train.add_argument("--workers", type=_int_arg(0), default=None, help=S.CLI_HELP_TRAIN_WORKERS)
    train.add_argument("--popsize", type=_int_arg(), default=None, help=S.CLI_HELP_POPSIZE)
    train.add_argument("--sigma", type=_float_arg, default=None, help=S.CLI_HELP_SIGMA)
    train.add_argument("--trials", type=_int_arg(), default=None, help=S.CLI_HELP_TRIALS)
    train.add_argument("--max-time", type=_float_arg, default=None, help=S.CLI_HELP_MAX_TIME)
    train.add_argument("--max-generations", type=_int_arg(), default=None, help=S.CLI_HELP_MAX_GENS)
    train.add_argument("--max-evaluations", type=_int_arg(), default=None, help=S.CLI_HELP_MAX_EVALS)
    train.add_argument("--target", type=_float_arg, default=None, help=S.CLI_HELP_TARGET)
    train.add_argument("--stagnation", type=_int_arg(), default=None, help=S.CLI_HELP_STAGNATION)
    train.add_argument("--validate-every", type=_int_arg(), default=None, help=S.CLI_HELP_VALIDATE_EVERY)
    train.add_argument("--diagonal-threshold", type=_int_arg(), default=None, help=S.CLI_HELP_DIAGONAL)
    train.add_argument("--ipop", action="store_true", default=None, help=S.CLI_HELP_IPOP)

    bench = sub.add_parser("bench", help=S.CLI_HELP_BENCH)
    bench.add_argument("env", type=Path, help=S.CLI_HELP_ENV_FILE)
    bench.add_argument("--workers", type=_worker_list, default=None, help=S.CLI_HELP_BENCH_WORKERS)
    bench.add_argument("--generations", type=_int_arg(1), default=3, help=S.CLI_HELP_BENCH_GENS)
    bench.add_argument("--popsize", type=_int_arg(2), default=32, help=S.CLI_HELP_POPSIZE)

    judge = sub.add_parser("judge", help=S.CLI_HELP_JUDGE)
    judge.add_argument("run", type=Path, help=S.CLI_HELP_RUN_DIR)
    judge.add_argument("--seeds", type=_int_arg(1), default=100, help=S.CLI_HELP_JUDGE_SEEDS)
    judge.add_argument("--generation", type=_int_arg(), default=None, help=S.CLI_HELP_JUDGE_GEN)
    judge.add_argument("--workers", type=_int_arg(0), default=None, help=S.CLI_HELP_TRAIN_WORKERS)

    play = sub.add_parser("play", help=S.CLI_HELP_PLAY)
    play.add_argument("env", type=Path, help=S.CLI_HELP_ENV_FILE)
    play.add_argument("--seed", type=_int_arg(), default=None, help=S.CLI_HELP_PLAY_SEED)
    play.add_argument("--replay", type=Path, default=None, help=S.CLI_HELP_PLAY_REPLAY)
    play.add_argument("--verify", type=Path, default=None, help=S.CLI_HELP_PLAY_VERIFY)

    bundle = sub.add_parser("bundle", help=S.CLI_HELP_BUNDLE)
    bundle.add_argument("run", type=Path, help=S.CLI_HELP_RUN_DIR)
    bundle.add_argument("--out", type=Path, default=None, help=S.CLI_HELP_BUNDLE_OUT)

    repro = sub.add_parser("reproduce", help=S.CLI_HELP_REPRODUCE)
    repro.add_argument("bundle", type=Path, help=S.CLI_HELP_BUNDLE_FILE)
    repro.add_argument("--out", type=Path, default=None, help=S.CLI_HELP_OUT)
    repro.add_argument("--workers", type=_int_arg(0), default=None, help=S.CLI_HELP_TRAIN_WORKERS)

    csvp = sub.add_parser("csv", help=S.CLI_HELP_CSV)
    csvp.add_argument("run", type=Path, help=S.CLI_HELP_RUN_DIR)
    csvp.add_argument("--out", type=Path, default=None, help=S.CLI_HELP_CSV_OUT)

    selftest = sub.add_parser("selftest", help=S.CLI_HELP_SELFTEST)
    selftest.add_argument("--log", type=Path, default=None, help=S.CLI_HELP_LOG)
    # 워커 0개면 점검용 워커 풀이 비어 응답을 끝까지(180초) 기다리다 프로세스 구조 실패로 잘못 알린다.
    selftest.add_argument("--workers", type=_int_arg(1), default=None, help=S.CLI_HELP_WORKERS)
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

    if args.command in ("train", "bench", "judge", "bundle", "reproduce", "csv"):
        from evolab.engine import commands

        return {"train": commands.train_command, "bench": commands.bench_command, "judge": commands.judge_command,
                "bundle": commands.bundle_command, "reproduce": commands.reproduce_command,
                "csv": commands.csv_command}[args.command](args)

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

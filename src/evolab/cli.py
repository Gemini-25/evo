"""명령줄 진입점: `evolab`, `evolab train`, `evolab bench`, `evolab play`, `evolab selftest`, `evolab report`,
`evolab export`, `evolab video-probe`, (6.2.0) `evolab preset`.

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
    from evolab.engine.trainer import RESTART_FROM  # numpy 스레드 제한(main 맨 앞) 뒤에 불러온다

    parser = _Parser(prog="evolab", description=S.CLI_DESCRIPTION)
    # 이름을 정해 두지 않으면 잘못된 명령 오류에 "command"라는 영어 이름이 나온다
    sub = parser.add_subparsers(dest="command", metavar=S.RV2_CLI_COMMAND_METAVAR)

    train = sub.add_parser("train", help=S.CLI_HELP_TRAIN)
    train.add_argument("env", type=Path, nargs="?", default=None, help=S.CLI_HELP_ENV_FILE)
    train.add_argument("--method", default=None, choices=["cmaes", "ga", "sa", "grow"], help=S.CLI_HELP_METHOD)
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
    # 학습 안정성(v1.1). 새 실행은 자동 재시작이 기본으로 켜진다. 이어 할 때는 재시작 설정만 바꿀 수 있다.
    restart = train.add_mutually_exclusive_group()
    restart.add_argument("--restart", type=_int_arg(1), default=None, dest="restart", help=S.CLI_HELP_RESTART)
    restart.add_argument("--no-restart", action="store_true", default=False, help=S.CLI_HELP_NO_RESTART)
    train.add_argument("--restart-from", choices=list(RESTART_FROM), default=None, help=S.CLI_HELP_RESTART_FROM)
    train.add_argument("--qualify", type=_int_arg(1), default=None, help=S.CLI_HELP_QUALIFY)
    train.add_argument("--qualify-trials", type=_int_arg(1), default=None, help=S.CLI_HELP_QUALIFY_TRIALS)
    # 구조 뒤집기(v2.1): 유전 알고리즘·담금질 둘 다의 확률(범위는 학습 설정 검사가 본다). 주지 않으면 새 실행 권장값
    # (환경에 있음 유전자가 있으면 0.05, 없으면 0). 학습 방법의 일부라 이어 할 때는 바꿀 수 없다.
    train.add_argument("--structure-flip", type=_float_arg, default=None, dest="structure_flip",
                       help=S.CLI_HELP_STRUCTURE_FLIP)
    # 자라는 뇌(v6.1, --method grow): 범위(1 이상)는 학습 설정 검사가 본다. 학습 방법의 일부라 이어 할 때는 바꿀 수 없다.
    train.add_argument("--grow-start", type=_int_arg(), default=None, dest="grow_start", help=S.CLI_HELP_GROW_START)
    train.add_argument("--grow-every", type=_int_arg(), default=None, dest="grow_every", help=S.CLI_HELP_GROW_EVERY)
    # 그래프용 관찰 기록(v1.2)만 끈다. 학습 설정이 아니라 run.json·재현 묶음에 남지 않는다(학습 결과는 같다).
    train.add_argument("--no-observe", action="store_true", default=False, help=S.CLI_HELP_NO_OBSERVE)

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
    # 처음 실행 안내 점검(v3.0): gen1 = 1세대 재생까지(기본), full = 목표 세대(10)까지 4배속(윈도우 CI W3에서 한 번)
    selftest.add_argument("--tour", choices=["gen1", "full"], default="gen1", help=S.CLI_HELP_SELFTEST_TOUR)

    # 오류 보고 묶음(v3.0): Qt를 불러오지 않는다. 창 모드 exe에서도 출력 없이 끝난다(--open이면 폴더를 연다).
    report = sub.add_parser("report", help=S.CLI_HELP_REPORT)
    report.add_argument("--out", type=Path, default=None, help=S.CLI_HELP_REPORT_OUT)
    report.add_argument("--run", type=Path, default=None, help=S.CLI_HELP_REPORT_RUN)
    report.add_argument("--open", action="store_true", help=S.CLI_HELP_REPORT_OPEN)

    # 내보내기(v3.1, 설계 4.19·D58): 창 없이 이 프로세스에서 다시 시뮬레이션해 MP4·GIF·폰 재생 파일을 만든다.
    export = sub.add_parser("export", help=S.CLI_HELP_EXPORT)
    export.add_argument("run", type=Path, help=S.CLI_HELP_RUN_DIR)
    export.add_argument("--format", required=True, choices=["mp4", "gif", "html"], dest="fmt",
                        help=S.CLI_HELP_EXPORT_FORMAT)
    export.add_argument("--generation", type=_int_arg(1), default=None, help=S.CLI_HELP_EXPORT_GEN)
    export.add_argument("--what", choices=["champion", "versus"], default="champion", help=S.CLI_HELP_EXPORT_WHAT)
    export.add_argument("--size", choices=["small", "medium"], default="medium", help=S.CLI_HELP_EXPORT_SIZE)
    export.add_argument("--camera", choices=["fixed", "follow"], default="fixed", help=S.CLI_HELP_EXPORT_CAMERA)
    export.add_argument("--encoder", choices=["auto", "mf", "builtin"], default="auto",
                        help=S.CLI_HELP_EXPORT_ENCODER)
    export.add_argument("--out", type=Path, required=True, help=S.CLI_HELP_EXPORT_OUT)
    sub.add_parser("video-probe", help=S.CLI_HELP_VIDEO_PROBE)

    # 조리법으로 환경 파일 만들기(6.2.0, 스포츠 설계 4.1 묶음 P): Qt를 불러오지 않는다.
    preset = sub.add_parser("preset", help=S.CLI_HELP_PRESET)
    preset.add_argument("recipe", help=S.CLI_HELP_PRESET_RECIPE)
    preset.add_argument("-o", "--out", type=Path, default=None, help=S.CLI_HELP_PRESET_OUT)
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
    if argv is None and args.command not in (None, "play", "selftest"):
        # 프로세스 진입(exe·`python -m evolab`·`evolab` 명령)에서만 잡히지 않은 예외 훅을 건다(v3 설계 4.9: 기록 + 한 줄 +
        # 종료 코드 1). 인수를 주고 부르는 테스트에는 걸지 않는다. 화면(명령 없음)은 run_gui가, 창을 띄우는 play와 selftest는
        # 걸지 않는다: Qt 슬롯의 예외는 프로세스를 끝내지 않으므로 「멈췄습니다」 한 줄이 틀리고, selftest의 기록은 임시 폴더라
        # 끝나면 지워진다(그 대신 파이썬 기본 훅이 traceback 전체를 stderr에 쓴다).
        from evolab import diagnostics

        diagnostics.install("cli")

    if args.command is None:
        from evolab.app.main import run_gui

        return run_gui()

    if args.command == "selftest":
        from evolab.selftest import run_selftest

        return run_selftest(log_path=args.log, workers=args.workers, check_gui=not args.no_gui, tour=args.tour)

    if args.command == "play":
        return _play(args.env, args.seed, args.replay, args.verify)

    if args.command == "report":
        from evolab.app.report.bundle import report_command

        return report_command(args.out, args.run, args.open)

    if args.command == "export":
        return _export(args)

    if args.command == "video-probe":
        return _video_probe()

    if args.command == "preset":
        return _preset(args.recipe, args.out)

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


def _human_size(n: int) -> str:
    if n < 1024 * 1024:
        return f"{max(1, round(n / 1024))}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def _export(args: argparse.Namespace) -> int:
    """`evolab export`(설계 4.19): 0 성공, 1 실패(실행을 못 읽음, 쓸 수 없는 인코더, 쓰기 실패), 2 인수 오류(확장자가 형식과 다름)."""
    from evolab.app.export import job as export_job

    out = Path(args.out)
    if export_job.format_for_path(out) != args.fmt:
        _print_err(S.RVM_CLI_BAD_ARGS.format(detail=S.EXP_BAD_SUFFIX.format(name=out.name)))
        return 2
    from evolab.app.export import worker
    from evolab.app.export.mp4 import encoder_status
    from evolab.runs.store import RunStoreError
    from evolab.spec.compiled import EnvFileError
    from evolab.world.physics2d.episode import EpisodeBuildError

    try:
        job = export_job.job_from_run(Path(args.run), args.fmt, generation=args.generation, what=args.what,
                                      size=args.size, camera=args.camera, encoder=args.encoder)
    except (RunStoreError, OSError, EnvFileError, EpisodeBuildError, ValueError, KeyError) as exc:
        _print_err(S.EXP_FAILED.format(reason=worker.error_message(exc)))
        return 1
    worker.ensure_qt_app()  # 창 없이 그린다(윈도우 밖에서 QT_QPA_PLATFORM이 없으면 offscreen)
    try:
        done = worker.run_job(job, out, lambda msg: _print_note(msg))
    except Exception as exc:  # noqa: BLE001 - 까닭 한 줄과 종료 코드 1(원문은 잡히지 않은 예외 기록 대신 여기서 짧게)
        _print_err(S.EXP_FAILED.format(reason=worker.error_message(exc)))
        return 1
    kind = str(done.get("encoder", ""))
    if kind == "mf":
        encoder = S.EXP_ENCODER_MF.format(name=encoder_status().detail)
    else:
        encoder = {"builtin": S.EXP_ENCODER_BUILTIN, "gif": S.EXP_ENCODER_GIF, "html": S.EXP_ENCODER_HTML}.get(kind, kind)
    if sys.stdout is not None:
        print(S.CLI_EXPORT_DONE.format(path=done["path"], size=_human_size(int(done["bytes"])),
                                       frames=int(done["frames"]), encoder=encoder))
    return 0


def _print_note(msg: dict[str, Any]) -> bool:
    """내보내기 진행 메시지 중 알림(MF 대체)만 한 줄로. 진행은 찍지 않는다(CI 기록이 짧게)."""
    if msg.get("type") == "note":
        _print_err(str(msg.get("text", "")))
    return True


def _video_probe() -> int:
    """`evolab video-probe`: 쓸 MP4 인코더 한 줄. 인코더가 없어도 종료 코드 0(설계 4.19)."""
    from evolab.app.export.mp4 import encoder_status

    status = encoder_status()
    if sys.stdout is not None:
        print(S.CLI_VIDEO_PROBE_LINE.format(encoder=status.kind, mf=status.detail))
    return 0


def _preset(recipe_arg: str, out: Path | None) -> int:
    """`evolab preset <조리법.json | 내장 이름> [-o 파일]`: 0 성공, 1 만들 수 없음(조리법·틀·쓰기), 2 조리법 파일을 못 읽음."""
    from evolab import paths
    from evolab.presets.recipe import Recipe, RecipeError, build, load_builtin
    from evolab.presets.template import TemplateError
    from evolab.spec.compiled import EnvFileError
    from evolab.spec.io import compile_env, dump_env, feature_version

    source = Path(recipe_arg)
    try:
        if source.suffix.lower() == ".json" or source.exists():
            import json

            try:
                raw = json.loads(source.read_bytes().decode("utf-8-sig"))
            except OSError as exc:
                _print_err(S.SPEC_READ_FAILED.format(path=source, reason=exc.strerror or exc))
                return 2
            except (UnicodeDecodeError, json.JSONDecodeError):
                _print_err(S.RECIPE_ERR_FORMAT.format(reason=S.SPEC_NOT_UTF8))
                return 2
            recipe = Recipe.from_json(raw)
        else:
            recipe = load_builtin(recipe_arg)
        data = build(recipe)
        spec = compile_env(data)
    except (RecipeError, TemplateError, EnvFileError) as exc:
        _print_err(str(exc))
        return 1
    target = out if out is not None else Path(paths.safe_file_stem(recipe.name or source.stem) + ".evoenv")
    text = dump_env(spec.file)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))  # 윈도우에서도 LF 그대로(.gitattributes eol=lf, W1의 SHA-256 비교)
    except OSError as exc:
        _print_err(S.PRESET_SAVE_FAILED.format(reason=exc.strerror or exc))
        return 1
    if sys.stdout is not None:
        print(S.CLI_PRESET_SAVED.format(path=target, version=feature_version(spec.file), inputs=len(spec.inputs),
                                        outputs=len(spec.outputs)))
    return 0


def _print_err(message: str) -> None:
    # --windowed로 묶은 exe에서는 sys.stderr가 None이다.
    if sys.stderr is not None:
        print(message, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

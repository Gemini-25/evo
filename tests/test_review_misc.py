"""검토 수정(직접 조종 기록 위치, 명령줄 인수 검사, 변경 지점 기록)의 회귀 테스트."""

from __future__ import annotations

import argparse
import ast
import json
import queue
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from evolab import cli, paths
from evolab.app import strings as S
from evolab.app.play import record as play_record
from evolab.app.play.record import best_record_score, new_record
from evolab.engine import commands
from evolab.engine.session import PAUSED, Session, SessionRequest
from evolab.runs.store import RunStore
from evolab.spec.io import load_env

LANDER = paths.builtin_env_dir() / "lander.evoenv"
LANDER_TEXT = LANDER.read_text(encoding="utf-8")
NEW_ENV = LANDER_TEXT.replace('"gravity": [0.0, -1.62]', '"gravity": [0.0, -3.0]')
CONFIG = {"popsize": 6, "trials": 2, "validate_every": 24, "checkpoint_every": 2, "run_seed": 5}


@pytest.fixture
def data_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """(문서 폴더, 설정에서 바꾼 데이터 폴더). 진짜 문서 폴더와 paths.ini를 건드리지 않는다."""
    docs = tmp_path / "문서"
    monkeypatch.setattr(paths, "user_state_dir", lambda: tmp_path / "state")
    monkeypatch.setattr(paths, "documents_dir", lambda: docs)
    custom = tmp_path / "내 데이터"
    paths.set_custom_data_dir(custom)
    return docs, custom


def finished_record(score: float, seed: int = 5) -> Any:
    rec = new_record(load_env(LANDER), seed)
    rec.actions = [[0.0] * len(rec.outputs)]
    rec.score, rec.outcome, rec.steps = score, "failure", 1
    return rec


# ---------------------------------------------------------------------------
# 직접 조종 기록은 설정의 데이터 폴더를 따른다
# ---------------------------------------------------------------------------


def test_play_records_follow_custom_data_dir(qtbot: Any, data_dirs: tuple[Path, Path],
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from evolab.app.play.window import PlayWindow, records_dir

    docs, custom = data_dirs
    assert records_dir() == custom / "records" == paths.records_dir()
    assert not (docs / S.DATA_DIR_NAME).exists()  # 옮기기 전 기본 폴더를 다시 만들지 않는다

    window = PlayWindow(load_env(LANDER), seed=5)
    qtbot.addWidget(window)
    window.record = finished_record(12.5)
    seen: dict[str, str] = {}

    def fake_save(parent: Any, title: str, default: str, flt: str) -> tuple[str, str]:
        seen["default"] = default
        return default, flt

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(fake_save))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    window.save_record()
    saved = Path(seen["default"])
    assert saved.parent == custom / "records" and saved.suffix == ".evoplay" and saved.is_file()
    window.close()


def test_play_records_default_location_without_custom_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.play.window import record_dirs, records_dir

    docs = tmp_path / "문서"
    monkeypatch.setattr(paths, "user_state_dir", lambda: tmp_path / "state")
    monkeypatch.setattr(paths, "documents_dir", lambda: docs)
    assert records_dir() == docs / S.DATA_DIR_NAME / "records" == paths.default_records_dir()
    assert record_dirs() == [records_dir()]  # 기본 위치 하나뿐이면 두 번 보지 않는다


def test_best_record_score_reads_new_and_old_record_folders(data_dirs: tuple[Path, Path]) -> None:
    from evolab.app.play.window import record_dirs, records_dir

    docs, custom = data_dirs
    spec = load_env(LANDER)
    assert record_dirs() == [custom / "records"]  # 예전 기본 위치가 없으면 만들지 않는다
    assert best_record_score(spec, record_dirs()) is None

    finished_record(12.5).save(records_dir() / "play_5.evoplay")  # 화면이 저장하는 이름
    assert best_record_score(spec, record_dirs()) == 12.5
    old = paths.default_records_dir()
    old.mkdir(parents=True)
    finished_record(30.0, seed=6).save(old / "옛 기록.json")  # 데이터 폴더를 바꾸기 전 기본 위치의 기록
    (old / "망가짐.evoplay").write_text("{", encoding="utf-8")
    other = finished_record(99.0)
    other.env_hash = "다른 환경"
    other.save(old / "다른 환경.evoplay")
    assert record_dirs() == [custom / "records", old]
    assert best_record_score(spec, record_dirs()) == 30.0
    assert play_record.RECORD_PATTERNS == ("*.evoplay", "*.json")


# ---------------------------------------------------------------------------
# 명령줄 인수: 잘못된 값은 한국어 문장과 종료 코드 2 (트레이스백 없이)
# ---------------------------------------------------------------------------


def _argparse_messages() -> list[str]:
    """argparse(설치된 파이썬)가 gettext로 부르는 원문 문장 전부(_('...'), ngettext('...', '...'))."""
    tree = ast.parse(Path(argparse.__file__).read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("_", "ngettext"):
            found += [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    return found


def _english_phrases() -> list[str]:
    """argparse 문장에서 자리 표시(%s, %(name)s, %r)를 뺀 영어 조각들("invalid choice:", "usage:", "argument" 등)."""
    phrases = set()
    for message in _argparse_messages():
        for part in re.split(r"%(?:\([a-z_]+\))?[sr]", message):
            part = part.strip(" \n()")
            if re.search(r"[A-Za-z]{3,}", part):
                phrases.add(part)
    return sorted(phrases)


ENGLISH_PHRASES = _english_phrases()


def _cli_error(capsys: pytest.CaptureFixture[str], argv: list[str]) -> str:
    """오류 줄(마지막 줄). 사용법 줄을 포함한 stderr 전체에 argparse의 영어 문장이 남지 않았는지도 본다."""
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    left = [p for p in ENGLISH_PHRASES if p in err]
    assert not left, f"영어 문장이 남음 {left}: {err}"
    assert err.startswith(S.RV2_ARGPARSE_TEXT["usage: "])
    return err.strip().splitlines()[-1]


def _argument(name: str, message: str) -> str:
    return S.RV2_ARGPARSE_TEXT["argument %(argument_name)s: %(message)s"] % {"argument_name": name, "message": message}


@pytest.mark.parametrize(("argv", "detail"), [
    (["judge", "run", "--seeds", "0"], S.RVM_CLI_TOO_SMALL.format(minimum=1, value=0)),
    (["judge", "run", "--seeds", "-3"], S.RVM_CLI_TOO_SMALL.format(minimum=1, value=-3)),
    (["judge", "run", "--seeds", "abc"], S.RVM_CLI_NOT_INT.format(value="abc")),
    (["judge", "run", "--workers", "-1"], S.RVM_CLI_TOO_SMALL.format(minimum=0, value=-1)),
    (["bench", "x.evoenv", "--workers", "abc"], S.RVM_CLI_BAD_WORKER_LIST.format(value="abc")),
    (["bench", "x.evoenv", "--workers", "1,,2"], S.RVM_CLI_BAD_WORKER_LIST.format(value="1,,2")),
    (["bench", "x.evoenv", "--workers", "0"], S.RVM_CLI_BAD_WORKER_LIST.format(value="0")),
    (["bench", "x.evoenv", "--generations", "0"], S.RVM_CLI_TOO_SMALL.format(minimum=1, value=0)),
    (["bench", "x.evoenv", "--popsize", "0"], S.RVM_CLI_TOO_SMALL.format(minimum=2, value=0)),
    (["bench", "x.evoenv", "--popsize", "-1"], S.RVM_CLI_TOO_SMALL.format(minimum=2, value=-1)),
    (["train", "x.evoenv", "--workers", "-2"], S.RVM_CLI_TOO_SMALL.format(minimum=0, value=-2)),
    (["train", "x.evoenv", "--popsize", "abc"], S.RVM_CLI_NOT_INT.format(value="abc")),
    (["train", "x.evoenv", "--sigma", "abc"], S.RVM_CLI_NOT_NUMBER.format(value="abc")),
    (["train", "x.evoenv", "--restart", "0"], S.RVM_CLI_TOO_SMALL.format(minimum=1, value=0)),
    (["train", "x.evoenv", "--qualify", "0"], S.RVM_CLI_TOO_SMALL.format(minimum=1, value=0)),
    (["train", "x.evoenv", "--qualify-trials", "abc"], S.RVM_CLI_NOT_INT.format(value="abc")),
    (["reproduce", "x.zip", "--workers", "-1"], S.RVM_CLI_TOO_SMALL.format(minimum=0, value=-1)),
    (["selftest", "--no-gui", "--workers", "0"], S.RVM_CLI_TOO_SMALL.format(minimum=1, value=0)),
    (["selftest", "--no-gui", "--workers", "-2"], S.RVM_CLI_TOO_SMALL.format(minimum=1, value=-2)),
])
def test_cli_rejects_bad_numbers_in_korean(capsys: pytest.CaptureFixture[str], argv: list[str], detail: str) -> None:
    line = _cli_error(capsys, argv)
    assert line == S.RVM_CLI_BAD_ARGS.format(detail=_argument(argv[-2], detail))  # "--seeds 인수: 1 이상이어야…"


_T = S.RV2_ARGPARSE_TEXT
_COMMANDS = ("'train', 'bench', 'judge', 'play', 'bundle', 'reproduce', 'csv', 'selftest', 'report', 'export', "
             "'video-probe'")  # v3.0: report 명령(설계 3.2), v3.1: export·video-probe(설계 4.19)


@pytest.mark.parametrize(("argv", "detail"), [
    (["judge"], _T["the following arguments are required: %s"] % "run"),
    (["bench"], _T["the following arguments are required: %s"] % "env"),
    (["play"], _T["the following arguments are required: %s"] % "env"),
    (["train", "--bogus"], _T["unrecognized arguments: %s"] % "--bogus"),
    (["judge", "run", "더", "있음"], _T["unrecognized arguments: %s"] % "더 있음"),
    (["train", "x.evoenv", "--method", "xx"], _argument("--method", _T[
        "invalid choice: %(value)r (choose from %(choices)s)"] % {"value": "xx",
                                                     "choices": "'cmaes', 'ga', 'sa', 'grow'"})),  # v6.1: grow
    (["nosuchcmd"], _argument(S.RV2_CLI_COMMAND_METAVAR, _T[
        "invalid choice: %(value)r (choose from %(choices)s)"] % {"value": "nosuchcmd", "choices": _COMMANDS})),
    (["없는명령"], _argument(S.RV2_CLI_COMMAND_METAVAR, _T[
        "invalid choice: %(value)r (choose from %(choices)s)"] % {"value": "없는명령", "choices": _COMMANDS})),
    (["train", "x.evoenv", "--seed"], _argument("--seed", _T["expected one argument"])),
    (["train", "x.evoenv", "--seed", "abc"], _argument("--seed", S.RVM_CLI_NOT_INT.format(value="abc"))),
    (["train", "x.evoenv", "--max", "3"], _T["ambiguous option: %(option)s could match %(matches)s"] % {
        "option": "--max", "matches": "--max-time, --max-generations, --max-evaluations"}),
    (["train", "x.evoenv", "--ipop=1"], _argument("--ipop", _T["ignored explicit argument %r"] % "1")),
    # v1.1 자동 재시작: 켜기와 끄기는 함께 줄 수 없고, 새로 시작할 곳은 정해진 값만
    (["train", "x.evoenv", "--restart", "5", "--no-restart"],
     _argument("--no-restart", _T["not allowed with argument %s"] % "--restart")),
    (["train", "x.evoenv", "--restart-from", "xx"], _argument("--restart-from", _T[
        "invalid choice: %(value)r (choose from %(choices)s)"] % {"value": "xx",
                                                                 "choices": "'fresh', 'champion', 'alternate'"})),
])
def test_cli_other_argument_errors_are_korean_too(capsys: pytest.CaptureFixture[str], argv: list[str],
                                                  detail: str) -> None:
    """argparse가 직접 만드는 오류도 한국어(옵션 이름과 값만 그대로). 예전에는 접두어만 한국어였다."""
    assert _cli_error(capsys, argv) == S.RVM_CLI_BAD_ARGS.format(detail=detail)


def test_cli_help_headings_are_korean(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in (["-h"], ["judge", "-h"], ["train", "--help"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert out.startswith(_T["usage: "]) and f"\n{_T['options']}:\n" in out
        assert _T["show this help message and exit"] in out
        assert not [p for p in ENGLISH_PHRASES if p in out], out
    assert f"\n{_T['positional arguments']}:\n  {S.RV2_CLI_COMMAND_METAVAR}\n" in cli._build_parser().format_help()


def test_argparse_translation_keys_match_argparse_and_do_not_leak(capsys: pytest.CaptureFixture[str]) -> None:
    """번역 키는 argparse의 실제 원문이어야 하고(파이썬이 바뀌면 여기서 알린다), 우리 파서 밖에는 남지 않는다."""
    assert set(_T) <= set(_argparse_messages())
    original = argparse._  # type: ignore[attr-defined]
    with pytest.raises(SystemExit):
        cli.main(["judge"])
    capsys.readouterr()
    assert argparse._ is original  # type: ignore[attr-defined]
    plain = argparse.ArgumentParser(prog="plain")
    plain.add_argument("--n", type=int)
    assert plain.format_usage().startswith("usage: ") and "options:" in plain.format_help()
    with pytest.raises(SystemExit):
        plain.parse_args(["--n", "x"])
    assert "argument --n: invalid int value: 'x'" in capsys.readouterr().err


def test_cli_accepts_valid_numbers() -> None:
    args = cli._build_parser().parse_args(["bench", "x.evoenv", "--workers", "2,1", "--generations", "1",
                                           "--popsize", "2"])
    assert args.workers == [2, 1] and args.generations == 1 and args.popsize == 2
    args = cli._build_parser().parse_args(["judge", "run", "--seeds", "1", "--workers", "0", "--generation", "-1"])
    assert (args.seeds, args.workers, args.generation) == (1, 0, -1)
    args = cli._build_parser().parse_args(["train", "x.evoenv", "--seed", "0", "--max-time", "1.5", "--workers", "0"])
    assert (args.seed, args.max_time, args.workers) == (0, 1.5, 0)
    assert cli._build_parser().parse_args(["bench", "x.evoenv"]).workers is None
    args = cli._build_parser().parse_args(["train", "x.evoenv", "--restart", "1", "--restart-from", "alternate",
                                           "--qualify", "8", "--qualify-trials", "1"])
    assert (args.restart, args.no_restart, args.restart_from, args.qualify, args.qualify_trials) == (
        1, False, "alternate", 8, 1)
    args = cli._build_parser().parse_args(["train", "x.evoenv", "--no-restart"])
    assert (args.restart, args.no_restart, args.restart_from, args.qualify) == (None, True, None, None)


def test_commands_guard_bad_values_without_parser(capsys: pytest.CaptureFixture[str]) -> None:
    """명령 함수를 직접 불러도(파서를 거치지 않아도) 0으로 나누기나 빈 중앙값으로 죽지 않는다."""
    assert commands.judge_command(argparse.Namespace(run="없음", seeds=0, generation=None, workers=0)) == 2
    assert S.RVM_CLI_OPTION_TOO_SMALL.format(name="--seeds", minimum=1, value=0) in capsys.readouterr().err
    for over, name, minimum, value in ((dict(popsize=0), "--popsize", 2, 0), (dict(generations=0), "--generations", 1, 0),
                                       (dict(workers=[1, 0]), "--workers", 1, 0)):
        ns = argparse.Namespace(**{"env": LANDER, "workers": [1], "generations": 1, "popsize": 2, **over})
        assert commands.bench_command(ns) == 2
        assert S.RVM_CLI_OPTION_TOO_SMALL.format(name=name, minimum=minimum, value=value) in capsys.readouterr().err


def test_cli_resume_rejects_bad_stop_condition_before_touching_run(tmp_path: Path,
                                                                    capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "runs"
    assert cli.main(["train", str(LANDER), "--seed", "1", "--popsize", "4", "--trials", "1", "--max-generations", "3",
                     "--validate-every", "2", "--workers", "0", "--out", str(out)]) == 0
    run = next(p for p in out.iterdir() if p.is_dir())
    store = RunStore.open(run)
    stats, events, ckpts = store.read_stats(), store.read_events(), store.checkpoints()
    capsys.readouterr()
    for argv, name in ((["--max-generations", "0"], "max_generations"), (["--max-time", "-1"], "max_time"),
                       (["--stagnation", "0"], "stagnation_generations")):
        assert cli.main(["train", "--resume", str(run), "--workers", "0", *argv]) == 2
        err = capsys.readouterr().err
        assert S.TRAIN_CANNOT_START.format(reason="") in err and name in err and "Traceback" not in err
    assert store.read_stats() == stats and store.read_events() == events and store.checkpoints() == ckpts


# ---------------------------------------------------------------------------
# 변경 지점: 다 기록하고 적용하거나, 아무것도 남기지 않는다
# ---------------------------------------------------------------------------


def paused_session(tmp_path: Path) -> Session:
    """학습 프로세스 없이(워커 0) 3세대까지 학습하고 멈춘 세션."""
    req = SessionRequest(kind="new", env_text=LANDER_TEXT, runs_dir=str(tmp_path / "runs"), workers=0, spectate=False,
                         config={**CONFIG, "max_generations": 6})
    sess = Session(None, None, req)
    sess._outbox = queue.Queue()
    sess._open()
    for _ in range(3):
        sess.trainer.step()
    sess._pause()
    assert sess.state == PAUSED and sess.trainer.state.generation == 3
    replies(sess)
    return sess


def replies(sess: Session) -> list[dict[str, Any]]:
    out = []
    while not sess._outbox.empty():
        m = sess._outbox.get_nowait()
        if m is not None and m.get("type") == "applied":
            out.append(m)
    return out


def records(sess: Session) -> tuple[Any, ...]:
    info = sess.store.read_run_info()
    _, ckpt = sess.store.load_checkpoint(3)
    events = [(e["kind"], e["generation"], e.get("version"), e.get("changes")) for e in sess.store.read_events()]
    return (info["env_versions"], info.get("protected_checkpoints", []), events, ckpt["config"], ckpt["env_version"])


def fail_once(monkeypatch: pytest.MonkeyPatch, name: str, exc: BaseException) -> None:
    real = getattr(RunStore, name)
    left = [1]

    def flaky(self: RunStore, *a: Any, **k: Any) -> Any:
        if left[0]:
            left[0] -= 1
            raise exc
        return real(self, *a, **k)

    monkeypatch.setattr(RunStore, name, flaky)


LOCKED = PermissionError(13, "locked")


@pytest.mark.parametrize(("what", "step"), [
    ("env", "save_checkpoint"), ("env", "protect"), ("env", "add_env_version"), ("env", "append_event"),
    ("config", "save_checkpoint"), ("config", "protect"), ("config", "append_event"),
    # save_checkpoint가 파일을 바꿔 넣은 뒤 정리 규칙에서 run.json을 읽다가 실패(2차 검토)
    ("env", "_protected"), ("config", "_protected"),
])
def test_failed_change_leaves_no_record_and_trainer_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              what: str, step: str) -> None:
    sess = paused_session(tmp_path)
    before = records(sess)
    old_version, old_trials = sess.trainer.env_version, sess.trainer.config.trials
    msg = {"type": "env", "text": NEW_ENV} if what == "env" else {"type": "config", "changes": {"trials": 3}}
    exc = LOCKED if step != "append_event" else OSError(28, "No space left on device")
    fail_once(monkeypatch, step, exc)
    sess._handle(msg)
    [reply] = replies(sess)
    assert reply["ok"] is False
    assert sess.trainer.env_text == LANDER_TEXT and sess.trainer.env_version == old_version
    assert sess.trainer.config.trials == old_trials
    assert records(sess) == before  # run.json, events.jsonl, 3세대 체크포인트 모두 바꾸기 전 그대로

    sess._handle(msg)  # 다시 하면(잠금이 풀림) 처음부터 실패하지 않은 것과 똑같이 기록된다
    [reply] = replies(sess)
    assert reply["ok"] is True
    versions, protected, events, ckpt_config, ckpt_version = records(sess)
    assert protected == [3] and ckpt_version == sess.trainer.env_version
    if what == "env":
        assert [(v["id"], v["from_generation"]) for v in versions] == [("v000", 0), ("v001", 3)]
        assert events[-1] == ("env_change", 3, "v001", None) and ckpt_version != old_version
        assert reply["env_text"] == NEW_ENV
    else:
        assert events[-1] == ("config_change", 3, None, {"trials": 3}) and ckpt_config["trials"] == 3
        assert reply["changes"] == {"trials": 3}
    sess.pool.close()


def test_failed_env_change_keeps_resume_and_reproduce_in_agreement(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.engine.pool import WorkerPool
    from evolab.engine.trainer import Trainer
    from evolab.runs import bundle

    sess = paused_session(tmp_path)
    fail_once(monkeypatch, "save_checkpoint", LOCKED)
    sess._handle({"type": "env", "text": NEW_ENV})
    assert replies(sess)[0]["ok"] is False
    sess._handle({"type": "config", "changes": {"max_generations": 5}})
    assert replies(sess)[0]["ok"] is True
    while sess.trainer.check_stop() is None:  # 이어서 학습(바꾸기 전 환경으로)
        sess.trainer.step()
    sess.trainer.checkpoint()
    sess.pool.close()
    store = sess.store
    assert [v["id"] for v in store.read_run_info()["env_versions"]] == ["v000"]
    assert "env_change" not in [e["kind"] for e in store.read_events()]
    stats = store.read_stats()
    assert len(stats) == 5
    zp = bundle.export_bundle(store.root, tmp_path / "묶음.zip")
    again = bundle.reproduce(zp, tmp_path / "다시", workers=0)
    assert bundle.compare_stats(stats, RunStore.open(again).read_stats()) is None
    with WorkerPool(0) as pool:
        for gen in (3, None):
            copy = tmp_path / f"사본 {gen}"
            shutil.copytree(store.root, copy)  # 이어하기는 체크포인트 뒤를 잘라 내므로 사본으로
            resumed = Trainer.resume(RunStore.open(copy), pool, gen)
            assert resumed.env_text == LANDER_TEXT  # 이어하기도 실제로 학습한 환경을 고른다


def test_change_failing_after_checkpoint_replaced_keeps_resume_in_agreement(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """save_checkpoint가 3세대 파일을 바꿔 넣은 뒤(정리 규칙의 run.json 읽기에서) 실패해도, 3세대 체크포인트는
    실제로 학습한 설정을 담는다: 3세대에서 이어 학습하면 기록된 통계와 같다(2차 검토)."""
    from evolab.engine.pool import WorkerPool
    from evolab.engine.trainer import Trainer
    from evolab.runs import bundle

    sess = paused_session(tmp_path)
    fail_once(monkeypatch, "_protected", LOCKED)
    sess._handle({"type": "config", "changes": {"trials": 3}})
    assert replies(sess)[0]["ok"] is False
    assert sess.store.load_checkpoint(3)[1]["config"]["trials"] == 2
    while sess.trainer.check_stop() is None:  # 바꾸지 않은 설정으로 계속(6세대까지)
        sess.trainer.step()
    sess.trainer.checkpoint()
    sess.pool.close()
    stats = sess.store.read_stats()
    copy = tmp_path / "사본"
    shutil.copytree(sess.store.root, copy)
    with WorkerPool(0) as pool:
        resumed = Trainer.resume(RunStore.open(copy), pool, 3)
        assert resumed.config.trials == 2
        while resumed.check_stop() is None:
            resumed.step()
    assert bundle.compare_stats(stats, RunStore.open(copy).read_stats()) is None


def test_undo_restores_only_change_keys_of_run_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """되돌리기는 환경 버전·보호 목록만 되돌린다(그 사이 화면이 쓴 메모 같은 다른 키는 남긴다)."""
    sess = paused_session(tmp_path)
    real = RunStore.append_event

    def memo_then_fail(self: RunStore, kind: str, generation: int, **data: Any) -> None:
        if kind == "env_change":
            self.update_run_info(memo="화면에서 쓴 메모")
            raise OSError(28, "No space left on device")
        real(self, kind, generation, **data)

    monkeypatch.setattr(RunStore, "append_event", memo_then_fail)
    sess._handle({"type": "env", "text": NEW_ENV})
    assert replies(sess)[0]["ok"] is False
    info = json.loads((sess.store.root / "run.json").read_text(encoding="utf-8"))
    assert info["memo"] == "화면에서 쓴 메모"
    assert [v["id"] for v in info["env_versions"]] == ["v000"] and info.get("protected_checkpoints", []) == []
    sess.pool.close()


def test_lab_best_score_line_reads_evoplay_records(data_dirs: tuple[Path, Path]) -> None:
    """실험실의 '내 최고 기록'도 화면이 저장하는 .evoplay와 예전 기본 위치를 본다(검토 20의 실험실 쪽)."""
    from evolab.app.lab.view import my_best_score
    from evolab.app.play.window import records_dir

    spec = load_env(LANDER)
    assert my_best_score(spec) is None
    finished_record(7.25).save(records_dir() / "play_1.evoplay")
    assert my_best_score(spec) == 7.25

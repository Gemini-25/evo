"""v3.0 D 묶음: 진단(diagnostics.py, winapp.py, lab/client.py의 오류 기록) — 설계 T20~T23, T34.

모든 테스트는 EVOLAB_STATE_DIR를 임시 폴더로 둔다(사용자의 진짜 로그 폴더를 건드리지 않음, 설계 D5). 훅·Qt 메시지 처리기를
거는 테스트는 끝에 `diagnostics.uninstall()`로 되돌린다. faulthandler는 이 프로세스에서 켜지 않는다(pytest가 쓰는 것을 빼앗지
않게): 치명적 오류 기록은 spawn 자식에서만 본다(T21).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from evolab import diagnostics as D
from evolab import paths, winapp
from evolab.app import strings as S

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """임시 상태 폴더 + 끝에 훅 되돌리기. 돌려주는 값: 로그 폴더."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    D.uninstall()
    yield tmp_path / "state" / "logs"
    D.uninstall()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [1000.0]
    monkeypatch.setattr(D, "_now", lambda: now[0])
    return now


@pytest.fixture
def quiet_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """훅을 걸기 전의 '앞 훅'을 아무것도 안 하는 것으로(테스트가 일부러 낸 예외를 pytest가 따로 알리지 않게)."""
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)
    monkeypatch.setattr(threading, "excepthook", lambda args: None)
    monkeypatch.setattr(sys, "unraisablehook", lambda args: None)


def _lines(log_dir: Path) -> list[dict[str, Any]]:
    path = log_dir / D.ERRORS_FILE
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def _kinds(log_dir: Path, kind: str) -> list[dict[str, Any]]:
    return [r for r in _lines(log_dir) if r["kind"] == kind]


# ---------------------------------------------------------------------------
# 기록 형식·상한·켜짐 조건 (설계 2.3)
# ---------------------------------------------------------------------------
def test_record_line_format(state: Path) -> None:
    D.record("shown", "파일을 쓸 수 없습니다", "Traceback ...\nPermissionError", where="gui")
    (rec,) = _lines(state)
    assert set(rec) == {"t", "kind", "where", "pid", "version", "message", "detail"}
    assert rec["kind"] == "shown" and rec["where"] == "gui" and rec["pid"] == os.getpid()
    from evolab import __version__

    assert rec["version"] == __version__ and rec["message"] == "파일을 쓸 수 없습니다"
    assert rec["detail"].endswith("PermissionError")
    time.strptime(rec["t"], "%Y-%m-%dT%H:%M:%S")
    # 한글은 이스케이프하지 않는다(묶음의 가림이 사용자 이름을 글자 그대로 찾는다)
    assert "파일을" in (state / D.ERRORS_FILE).read_text(encoding="utf-8")
    assert D.recent(5) == [rec]


def test_record_is_off_without_install_or_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """설치(run_gui·cli.main)도 EVOLAB_STATE_DIR도 없으면 쓰지 않는다: 테스트가 학습 오류를 일으켜도 진짜 로그 폴더는 그대로."""
    D.uninstall()
    monkeypatch.delenv(paths.STATE_DIR_ENV, raising=False)
    monkeypatch.setattr(paths, "log_dir", lambda: tmp_path / "logs")
    D.record("training", "x", "y")
    assert not (tmp_path / "logs").exists()


def test_record_caps_lines_and_bytes(state: Path) -> None:
    for i in range(600):
        D.record("qt", f"warning {i}")
        if i % 97 == 0:
            assert len(_lines(state)) <= D.ERRORS_MAX_LINES
    recs = _lines(state)
    assert len(recs) <= D.ERRORS_MAX_LINES and recs[-1]["message"] == "warning 599"
    big = "x" * 200_000
    for i in range(30):
        D.record("uncaught", f"big {i}", big)
        assert (state / D.ERRORS_FILE).stat().st_size <= D.ERRORS_MAX_BYTES
    recs = _lines(state)
    assert recs[-1]["message"] == "big 29" and len(recs[-1]["detail"]) == D.DETAIL_MAX  # 한 줄 detail 상한


def test_record_survives_unwritable_log_dir(state: Path) -> None:
    state.parent.mkdir(parents=True)
    state.write_text("a file where the folder should be", encoding="utf-8")
    D.record("shown", "x")  # 예외 없이 조용히


# ---------------------------------------------------------------------------
# T20 화면 훅
# ---------------------------------------------------------------------------
def _fire(qtbot: Any, fn: Any, calls: list[Any], n: int) -> None:
    from PySide6.QtCore import QTimer

    QTimer.singleShot(0, fn)
    qtbot.waitUntil(lambda: len(calls) >= n, timeout=5000)


def _raise_a() -> None:
    raise ZeroDivisionError("A")


def _raise_b() -> None:
    raise KeyError("B")


def _raise_c() -> None:
    raise ValueError("C")


def _raise_d() -> None:
    raise RuntimeError("D")


@pytest.mark.qt_no_exception_capture
@pytest.mark.no_qt_log
def test_slot_exceptions_record_box_status(qtbot: Any, state: Path, clock: list[float], quiet_hooks: None) -> None:
    calls: list[tuple[str, str]] = []
    D.install("gui")
    assert D.install_gui(lambda action, line: calls.append((action, line))) is False
    (start,) = _kinds(state, "start")
    assert start["message"].startswith("evolab ") and json.loads(start["detail"])["frozen"] is False

    _fire(qtbot, _raise_a, calls, 1)
    assert calls == [("box", "ZeroDivisionError: A")]
    (rec,) = _kinds(state, "uncaught")
    assert rec["message"] == "ZeroDivisionError: A" and "Traceback" in rec["detail"] and "_raise_a" in rec["detail"]

    clock[0] += 5
    _fire(qtbot, _raise_a, calls, 2)  # 같은 서명 두 번째: 상태 줄, 기록은 그대로 한 줄
    assert calls[1] == ("again", "ZeroDivisionError: A") and len(_kinds(state, "uncaught")) == 1

    _fire(qtbot, _raise_b, calls, 3)
    _fire(qtbot, _raise_c, calls, 4)
    _fire(qtbot, _raise_d, calls, 5)  # 네 번째 다른 예외: 상자 없음
    assert [a for a, _ in calls] == ["box", "again", "box", "box", "more"]
    assert [r["message"] for r in _kinds(state, "uncaught")] == ["ZeroDivisionError: A", "KeyError: 'B'",
                                                                "ValueError: C", "RuntimeError: D"]
    # 이벤트 루프는 계속 돈다
    alive: list[int] = []
    _fire(qtbot, lambda: alive.append(1), alive, 1)


@pytest.mark.qt_no_exception_capture
@pytest.mark.no_qt_log
def test_again_status_is_rate_limited(qtbot: Any, state: Path, clock: list[float], quiet_hooks: None) -> None:
    calls: list[tuple[str, str]] = []
    D.install_gui(lambda action, line: calls.append((action, line)))
    assert D._state.where == "gui"  # install_gui가 install("gui")를 대신 부른다
    from PySide6.QtCore import QTimer

    _fire(qtbot, _raise_a, calls, 1)
    _fire(qtbot, _raise_a, calls, 2)
    QTimer.singleShot(0, _raise_a)  # 2초 안: 알리지 않음
    qtbot.wait(200)
    assert [a for a, _ in calls] == ["box", "again"]
    clock[0] += D.AGAIN_REPEAT_S + 0.1
    _fire(qtbot, _raise_a, calls, 3)
    assert [a for a, _ in calls] == ["box", "again", "again"] and len(_kinds(state, "uncaught")) == 1


@pytest.mark.qt_no_exception_capture
@pytest.mark.no_qt_log
def test_qthread_python_thread_and_del_exceptions_are_recorded(qtbot: Any, state: Path, quiet_hooks: None) -> None:
    from PySide6.QtCore import QThread

    calls: list[tuple[str, str]] = []
    D.install("gui")
    D.install_gui(lambda action, line: calls.append((action, line)))

    class Worker(QThread):
        def run(self) -> None:  # noqa: D102
            raise LookupError("in qthread")

    worker = Worker()
    worker.start()
    worker.wait(5000)
    def target() -> None:
        raise OSError("in thread")

    t = threading.Thread(target=target, name="probe-thread")
    t.start()
    t.join()

    class Bad:
        def __del__(self) -> None:
            raise AttributeError("in del")

    Bad()  # 바로 지워지며 __del__ 예외는 unraisable로
    qtbot.waitUntil(lambda: len(calls) >= 2, timeout=5000)
    threads = _kinds(state, "thread")
    assert {r["message"] for r in threads} == {"LookupError: in qthread", "OSError: in thread"}
    assert any(r["detail"].startswith("thread: probe-thread") for r in threads)
    (unr,) = _kinds(state, "unraisable")
    assert unr["message"] == "AttributeError: in del" and "Bad.__del__" in unr["detail"]
    assert sorted(a for a, _ in calls) == ["box", "box"]  # unraisable은 기록만


@pytest.mark.no_qt_log
def test_qt_warnings_recorded_once_per_minute(qtbot: Any, state: Path, clock: list[float], quiet_hooks: None,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    from PySide6.QtCore import qDebug, qWarning

    D.install_gui(lambda action, line: None)
    qWarning("same warning")
    qWarning("same warning")
    qDebug("debug only")
    clock[0] += 30
    qWarning("same warning")
    mine = {"same warning", "other warning"}
    assert [r["message"] for r in _kinds(state, "qt") if r["message"] in mine] == ["same warning"]
    clock[0] += 31
    qWarning("same warning")
    qWarning("other warning")
    qt = [r for r in _kinds(state, "qt") if r["message"] in mine]
    assert [r["message"] for r in qt] == ["same warning", "same warning", "other warning"]
    assert qt[0]["detail"].startswith("QtWarningMsg")
    err = capsys.readouterr().err
    assert err.count("same warning") == 4 and "debug only" in err  # 원래처럼 표준 오류에도 나간다


def test_uninstall_restores_hooks(state: Path, quiet_hooks: None) -> None:
    before = (sys.excepthook, threading.excepthook, sys.unraisablehook)
    D.install("cli")
    D.install("cli")
    assert sys.excepthook is D._excepthook and threading.excepthook is D._threadhook
    assert sys.unraisablehook is D._unraisablehook
    assert _kinds(state, "start") == []  # 명령줄은 훅만
    D.uninstall()
    assert (sys.excepthook, threading.excepthook, sys.unraisablehook) == before


def test_cli_uncaught_exception_records_and_prints_one_line(tmp_path: Path) -> None:
    env = {**os.environ, paths.STATE_DIR_ENV: str(tmp_path / "state"), "PYTHONIOENCODING": "utf-8"}
    env.pop(D.CI_ENV, None)  # CI(GitHub Actions의 CI=true)에서는 traceback도 쓴다: 아래 테스트
    code = "from evolab import diagnostics as D\nD.install('cli')\nraise RuntimeError('boom in cli')\n"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", env=env,
                         timeout=120)
    assert out.returncode == 1
    log = tmp_path / "state" / "logs"
    (rec,) = _kinds(log, "uncaught")
    assert rec["where"] == "cli" and rec["message"] == "RuntimeError: boom in cli" and "Traceback" in rec["detail"]
    line = S.CLI_UNCAUGHT.format(reason="RuntimeError: boom in cli", path=log / D.ERRORS_FILE)
    assert out.stderr.strip().splitlines() == [line]


def test_cli_uncaught_exception_on_ci_also_prints_traceback(tmp_path: Path) -> None:
    """CI에서는 errors.jsonl을 볼 수 없을 수 있다(검토 수정): 한 줄 앞에 traceback 전체."""
    env = {**os.environ, paths.STATE_DIR_ENV: str(tmp_path / "state"), "PYTHONIOENCODING": "utf-8", D.CI_ENV: "true"}
    code = "from evolab import diagnostics as D\nD.install('cli')\nraise RuntimeError('boom in cli')\n"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", env=env,
                         timeout=120)
    assert out.returncode == 1
    lines = out.stderr.strip().splitlines()
    log = tmp_path / "state" / "logs"
    assert lines[0].startswith("Traceback") and "RuntimeError: boom in cli" in lines
    assert lines[-1] == S.CLI_UNCAUGHT.format(reason="RuntimeError: boom in cli", path=log / D.ERRORS_FILE)


def test_selftest_command_leaves_the_default_exception_hook(monkeypatch: pytest.MonkeyPatch, state: Path) -> None:
    """`evolab selftest`에는 명령줄 훅을 걸지 않는다(검토 수정): 창·안내 점검의 Qt 슬롯 예외는 프로세스를 끝내지 않고, 그
    기록은 임시 폴더라 끝나면 지워진다. 파이썬 기본 훅이 traceback 전체를 쓴다. 다른 명령(report)에는 건다."""
    from evolab import cli, selftest
    from evolab.app.report import bundle

    hooks: list[Any] = []
    monkeypatch.setattr(selftest, "run_selftest", lambda **kw: hooks.append(sys.excepthook) or 0)
    monkeypatch.setattr(bundle, "report_command", lambda *a: hooks.append(sys.excepthook) or 0)
    monkeypatch.setattr(sys, "argv", ["evolab", "selftest", "--no-gui"])
    assert cli.main() == 0
    monkeypatch.setattr(sys, "argv", ["evolab", "report", "--out", str(state / "x.zip")])
    assert cli.main() == 0
    assert hooks[0] is not D._excepthook and hooks[1] is D._excepthook


@pytest.mark.no_qt_log
def test_uninstall_keeps_a_qt_handler_installed_after_ours(qtbot: Any, state: Path, quiet_hooks: None) -> None:
    """uninstall은 우리 처리기가 아직 걸려 있을 때만 앞의 것으로 되돌린다(검토 수정). pytest-qt의 로그 잡기처럼 다른 쪽이 그사이
    처리기를 바꾸고 끝냈으면 그 쪽의 것을 그대로 둔다: 끝난 잡기를 다시 걸면 이후 Qt 메시지가 죽은 곳으로 간다."""
    from PySide6.QtCore import qInstallMessageHandler

    def first(*_a: Any) -> None:
        pass

    def later(*_a: Any) -> None:
        pass

    old = qInstallMessageHandler(first)
    try:
        D.install_gui(lambda action, line: None)
        D.uninstall()  # 우리 것이 걸려 있음: 앞의 것(first)으로
        assert qInstallMessageHandler(first) is first
        D.install_gui(lambda action, line: None)
        qInstallMessageHandler(later)  # 다른 쪽이 바꿈
        D.uninstall()
        assert qInstallMessageHandler(first) is later
    finally:
        qInstallMessageHandler(old)


# ---------------------------------------------------------------------------
# T21 spawn 자식의 치명적 오류가 faults.log에
# ---------------------------------------------------------------------------
def test_spawned_child_segfault_is_logged(state: Path) -> None:
    def markers() -> int:
        text = (state / D.FAULTS_FILE).read_text(encoding="utf-8", errors="replace")
        return sum(text.count(m) for m in D.CRASH_MARKERS)

    ok, detail = D.crash_check(timeout=120)
    text = (state / D.FAULTS_FILE).read_text(encoding="utf-8", errors="replace")
    assert ok, (detail, text[-2000:])
    assert markers() == 1 and "_crash_child" in text
    assert int(detail.split()[0].removeprefix("exit=")) != 0
    ok2, detail2 = D.crash_check(timeout=120)  # 두 번째도 같은 파일에 덧붙인다
    assert ok2, detail2
    assert markers() == 2


def test_entry_points_enable_fault_log_before_freeze_support() -> None:
    for path in (ROOT / "packaging" / "entry.py", ROOT / "src" / "evolab" / "__main__.py"):
        text = path.read_text(encoding="utf-8")
        a, b, c = (text.index(x) for x in ("    ensure_std_streams()\n", "    enable_fault_log()\n", "    multiprocessing.freeze_support()\n"))
        assert a < b < c, path


def test_faults_log_trimmed_at_gui_start(qtbot: Any, state: Path, quiet_hooks: None) -> None:
    state.mkdir(parents=True)
    path = state / D.FAULTS_FILE
    body = b"".join(b"Fatal Python error: line %07d\n" % i for i in range(60_000))  # 약 1.8MB
    path.write_bytes(body)
    D.install_gui(lambda action, line: None)
    data = path.read_bytes()
    assert len(data) <= D.FAULTS_KEEP_BYTES and data.endswith(b"line 0059999\n") and data.startswith(b"Fatal")
    small = b"Fatal Python error: small\n"
    path.write_bytes(small)
    D.uninstall()
    D.mark_clean_exit()
    D.install_gui(lambda action, line: None)
    assert path.read_bytes() == small  # 1MB 이하는 그대로


# ---------------------------------------------------------------------------
# T22 학습 클라이언트의 오류 기록과 종료 코드 풀이
# ---------------------------------------------------------------------------
def test_exit_code_text_table() -> None:
    assert D.exit_code_text(None) == "" and D.exit_code_text(0) == ""
    assert D.exit_code_text(3221225477) == S.EXIT_CODE_TEXT[3221225477] == "메모리 접근 오류"
    assert D.exit_code_text(-1073741819) == S.EXIT_CODE_TEXT[3221225477]  # 부호 있는 32비트로 온 NTSTATUS
    assert D.exit_code_text(-11) == S.EXIT_CODE_SIGNAL.format(name="SIGSEGV")
    assert D.exit_code_text(-15) == S.EXIT_CODE_SIGNAL.format(name="SIGTERM")
    assert D.exit_code_text(-200) == S.EXIT_CODE_SIGNAL.format(name="200")
    assert D.exit_code_text(1) == S.EXIT_CODE_UNKNOWN and D.exit_code_text(12345) == S.EXIT_CODE_UNKNOWN
    for code in S.EXIT_CODE_TEXT:
        assert D.exit_code_text(code) == S.EXIT_CODE_TEXT[code]


class _FakeProc:
    def __init__(self, exitcode: int | None) -> None:
        self.exitcode = exitcode

    def join(self, timeout: float | None = None) -> None:
        return None

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        return None


def _client(qtbot: Any) -> Any:
    from evolab.app.lab.client import TrainingClient

    client = TrainingClient()
    failed: list[tuple[str, bool]] = []
    client.failed.connect(lambda text, fatal: failed.append((text, fatal)))
    client.failures = failed  # type: ignore[attr-defined]
    return client


def test_client_records_training_error_with_traceback(qtbot: Any, state: Path) -> None:
    client = _client(qtbot)
    detail = 'Traceback (most recent call last):\n  File "session.py", line 92\nValueError: bad'
    client._dispatch({"type": "error", "fatal": True, "message": "학습을 시작할 수 없습니다: bad", "detail": detail})
    client._dispatch({"type": "error", "fatal": False, "notice": True, "message": "관찰 기록 요약 실패"})
    recs = _kinds(state, "training")
    assert [(r["message"], r["detail"], r["where"]) for r in recs] == [
        ("학습을 시작할 수 없습니다: bad", detail, "training"), ("관찰 기록 요약 실패", "", "training")]
    assert client.failures == [("학습을 시작할 수 없습니다: bad", True)]


@pytest.mark.parametrize(("code", "meaning"), [
    (3221225477, "메모리 접근 오류"),
    (-11, S.EXIT_CODE_SIGNAL.format(name="SIGSEGV")),
    (7, S.EXIT_CODE_UNKNOWN),
])
def test_client_lost_adds_exit_code_meaning(qtbot: Any, state: Path, code: int, meaning: str) -> None:
    client = _client(qtbot)
    client._proc = _FakeProc(code)
    client._closed_sent = False
    client._lost()
    text = S.SESSION_LOST_MEANING.format(base=S.SESSION_LOST.format(code=code), meaning=meaning)
    assert client.failures == [(text, True)]
    (rec,) = _kinds(state, "lost")
    assert rec["message"] == text


def test_client_lost_with_clean_exit_says_nothing(qtbot: Any, state: Path) -> None:
    client = _client(qtbot)
    client._proc = _FakeProc(0)
    client._closed_sent = False
    client._lost()
    assert client.failures == [] and _kinds(state, "lost") == []


def test_real_training_process_error_detail_reaches_errors_jsonl(qtbot: Any, state: Path, tmp_path: Path) -> None:
    from evolab.engine.session import SessionRequest

    client = _client(qtbot)
    client.start(SessionRequest(kind="new", env_text="{ not json", workers=0, runs_dir=str(tmp_path / "runs")))
    try:
        qtbot.waitUntil(lambda: bool(client.failures), timeout=60000)
    finally:
        client.shutdown(timeout=20)
    (rec,) = _kinds(state, "training")
    assert rec["message"] == client.failures[0][0] and client.failures[0][1] is True
    assert rec["detail"].startswith("Traceback (most recent call last)") and "compile_env" in rec["detail"]


# ---------------------------------------------------------------------------
# T23 갑자기 닫힘 표시
# ---------------------------------------------------------------------------
def _flags(log_dir: Path) -> list[str]:
    return sorted(p.name for p in log_dir.glob("running-*.flag"))


def test_running_flag_and_previous_crash(qtbot: Any, state: Path, quiet_hooks: None) -> None:
    import psutil

    assert D.install_gui(lambda action, line: None) is False
    own = f"running-{os.getpid()}.flag"
    assert _flags(state) == [own]
    D.mark_clean_exit()
    assert _flags(state) == []
    D.mark_clean_exit()  # 두 번 불러도 된다

    D.uninstall()
    dead = state / "running-99999999.flag"
    dead.write_text("", encoding="ascii")
    parent = os.getppid()
    alive = state / f"running-{parent}.flag"
    alive.write_text(repr(psutil.Process(parent).create_time()), encoding="ascii")
    reused = state / f"running-{parent}.flag.tmp"  # 이름이 맞지 않는 파일은 무시
    reused.write_text("", encoding="ascii")
    assert D.install_gui(lambda action, line: None) is True  # 죽은 pid의 표시 → 한 번 참, 지움
    assert not dead.exists() and alive.exists() and reused.exists()
    assert own in _flags(state)
    D.uninstall()
    D.mark_clean_exit()
    assert D.install_gui(lambda action, line: None) is False  # 지운 뒤에는 다시 묻지 않음

    D.uninstall()
    D.mark_clean_exit()
    alive.write_text("12345.0", encoding="ascii")  # 같은 pid, 다른 시작 시각 = pid 재사용 → 죽은 것
    assert D.install_gui(lambda action, line: None) is True and not alive.exists()


# ---------------------------------------------------------------------------
# PC 정보, winapp (T34)
# ---------------------------------------------------------------------------
def test_platform_info_has_no_paths_or_names() -> None:
    info = D.platform_info()
    assert set(info) == {"version", "frozen", "os", "cpu", "cores", "memory", "python", "qt", "screen"}
    assert all(isinstance(v, str) and v for v in info.values())
    text = " ".join(info.values())
    home = str(Path.home())
    assert home not in text and os.sep + "home" + os.sep not in text
    for name in {os.environ.get("USERNAME", ""), os.environ.get("USER", ""), os.environ.get("COMPUTERNAME", "")}:
        if len(name) >= 3:
            assert name not in text


def test_diagnostics_module_imports_no_qt() -> None:
    code = ("import sys\nimport evolab.diagnostics, evolab.winapp, evolab.app.report.bundle\n"
            "print(any(m.split('.')[0] in ('PySide6', 'shiboken6', 'numpy') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "False", out.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="윈도우가 아닐 때의 동작")
def test_winapp_does_nothing_off_windows(capsys: pytest.CaptureFixture[str]) -> None:
    assert winapp.create_app_mutex() is None
    assert winapp.known_folder("desktop") is None and winapp.known_folder("documents") is None
    winapp.message_box("제목", "본문")
    assert capsys.readouterr().err == "제목\n본문\n"


@pytest.mark.skipif(sys.platform != "win32", reason="윈도우 전용(W1)")
def test_winapp_mutex_and_folders_on_windows() -> None:
    import ctypes
    from ctypes import wintypes

    handle = winapp.create_app_mutex()
    assert handle
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenMutexW.restype = wintypes.HANDLE
    kernel32.OpenMutexW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    other = kernel32.OpenMutexW(0x00100000, False, winapp.MUTEX_NAME)  # SYNCHRONIZE
    assert other
    kernel32.CloseHandle(other)
    for name in ("desktop", "documents"):
        folder = winapp.known_folder(name)
        assert folder is not None and folder.is_absolute(), name

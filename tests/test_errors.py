"""T18: 오류 문구 (v3 설계 D26, 4.8, 5.5).

- explain이 4.8 표의 예외마다 한국어 글 + 다음 할 일, 모르는 예외 원문은 「자세히」로.
- show_error 상자: 제목 그대로, 본문, 「다음에 할 일」, [확인] [오류 보고 묶음 만들기…], 자세히, 오류 기록 한 줄, 막지 않음.
- AST 규칙: app/ 안(errors.py 밖) QMessageBox.* 호출에 except 이름·str(그 이름)·그 이름을 넣은 format이 없음.
- 화면마다 바꾼 곳(실험실 7 + 변수 경로 3, 에디터 4, 직접 조종 3)이 show_error 상자를 띄움(lab/view.py의 변수 경로는 손으로 확인).
"""

from __future__ import annotations

import ast
import errno
import json
import os
from pathlib import Path
from typing import Any

import pytest
import shiboken6
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox, QWidget

import evolab.app
from evolab import paths
from evolab.app import errors
from evolab.app import strings as S
from evolab.app.errors import Friendly, explain, reason, show_error, structure_next_step
from evolab.runs.store import RunStoreError
from evolab.spec.compiled import EnvFileError, Issue
from evolab.world.physics2d.episode import EpisodeBuildError

APP = Path(evolab.app.__file__).parent
LANDER = paths.builtin_env_dir() / "lander.evoenv"
MAIN_PENDING = 3
"""main.py(I 묶음 파일)의 날것 예외 상자 수. I 묶음이 show_error로 바꾸면(main.py에 show_error가 나타나면) 0이어야 한다."""


@pytest.fixture(autouse=True)
def _private_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """실제 사용자 데이터·설정 폴더를 건드리지 않는다(설계 D5). 오류 기록(errors.jsonl)도 여기에 쌓인다."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))


@pytest.fixture(autouse=True)
def _close_left_boxes() -> Any:
    """시험이 남긴 오류 상자를 닫는다(창 모달 상자가 다음 시험의 「지금 모달 창」이 되지 않게)."""
    yield
    for box in errors.open_error_boxes():
        box.close()


def raised(exc: BaseException) -> BaseException:
    """traceback이 붙은 예외(실제로 raise한 것처럼)."""
    try:
        raise exc
    except BaseException as caught:  # noqa: BLE001
        return caught


def boxes_of(parent: QWidget) -> list[QMessageBox]:
    return [b for b in errors.open_error_boxes() if b.parent() is parent]


# ---------------------------------------------------------------------------
# explain: 설계 4.8 표
# ---------------------------------------------------------------------------
def test_explain_os_errors_say_what_happened_and_what_to_do() -> None:
    perm = S.ERR_PERMISSION, S.ERR_PERMISSION_NEXT.format(menu=S.MENU_SETTINGS)
    full = S.ERR_DISK_FULL, S.ERR_DISK_FULL_NEXT.format(records=S.TAB_RECORDS)
    win_busy = OSError("busy")
    win_busy.winerror = 32  # type: ignore[attr-defined]  # 윈도우 공유 위반(다른 프로그램이 파일을 쥠)
    win_full = OSError("disk")
    win_full.winerror = 112  # type: ignore[attr-defined]
    cases = [
        (PermissionError(errno.EACCES, "Permission denied", "C:/x/stats.csv"), perm),
        (OSError(errno.EACCES, "Permission denied"), perm),
        (OSError(errno.EROFS, "Read-only file system"), perm),
        (win_busy, perm),
        (OSError(errno.ENOSPC, "No space left on device"), full),
        (win_full, full),
        (MemoryError(), (S.ERR_MEMORY, S.ERR_MEMORY_NEXT)),
    ]
    for exc, (message, next_step) in cases:
        f = explain(raised(exc))
        assert (f.message, f.next_step) == (message, next_step), exc
        assert "Errno" not in f.message and "denied" not in f.message
        assert type(exc).__name__ in f.detail and "Traceback" in f.detail  # 원문과 traceback은 자세히로
    missing = explain(raised(FileNotFoundError(errno.ENOENT, "No such file or directory", "/tmp/없는/a.evoenv")))
    assert missing.message == S.ERR_NOT_FOUND.format(path="/tmp/없는/a.evoenv")
    assert missing.next_step == S.ERR_NOT_FOUND_NEXT.format(records=S.TAB_RECORDS)
    assert explain(FileNotFoundError()).message == S.ERR_NOT_FOUND.format(path="").rstrip().rstrip(":")


def test_explain_our_errors_keep_their_korean_text() -> None:
    env = EnvFileError([Issue("error", "world.dt", "dt는 0보다 커야 합니다")])
    f = explain(raised(env))
    assert f.message == str(env) and f.next_step == S.ERR_ENV_FILE_NEXT.format(issues=S.EDP_TAB_ISSUES)
    build = EpisodeBuildError(S.RT_DEGENERATE_SHAPE.format(body="runner"))
    assert (explain(build).message, explain(build).next_step) == (str(build), S.ERR_BUILD_NEXT)
    store = RunStoreError(S.RUNS_NOT_A_RUN.format(path="x"))
    assert (explain(store).message, explain(store).next_step) == (
        str(store), S.ERR_RUN_STORE_NEXT.format(records=S.TAB_RECORDS))
    korean = ValueError(S.TRAIN_CONFIG_INVALID.format(name="popsize", value=1))  # 설정 검사 등 우리 코드의 한국어 예외
    assert (explain(korean).message, explain(korean).next_step) == (str(korean), S.ERR_UNEXPECTED_NEXT)


def test_explain_unknown_errors_hide_the_raw_text_in_details() -> None:
    for exc in (ValueError("could not convert string to float: 'x'"), KeyError("generation"), ZeroDivisionError("float division by zero"),
                OSError(errno.EINVAL, "Invalid argument")):
        f = explain(raised(exc))
        assert f.message == S.ERR_UNEXPECTED.format(kind=type(exc).__name__)
        assert f.next_step == S.ERR_UNEXPECTED_NEXT
        assert str(exc) in f.detail and "Traceback" in f.detail
    assert explain("이미 만든 알림") == Friendly("이미 만든 알림", S.ERR_UNEXPECTED_NEXT, "")


def test_wrapped_os_error_is_judged_by_the_os_error(tmp_path: Path) -> None:
    """에디터 문서가 파일을 못 읽어 낸 EnvFileError(안에 OSError, raise from None)는 OS 오류로 판단한다(영어 strerror가 본문에 안 섞임)."""
    from evolab.app.editor.document import EnvDocument

    with pytest.raises(EnvFileError) as info:
        EnvDocument().open(tmp_path / "없는 파일.evoenv")
    f = explain(info.value)
    assert f.message == S.ERR_NOT_FOUND.format(path=tmp_path / "없는 파일.evoenv")
    assert "FileNotFoundError" in f.detail and S.SPEC_READ_FAILED.split(":")[0] in f.detail
    assert structure_next_step(info.value) is None  # 다음 할 일은 explain이 고른 OS 오류의 것
    broken = EnvFileError([Issue("error", "", S.SPEC_BAD_JSON.format(line=1, col=1))])
    assert structure_next_step(broken) == S.ERR_ENV_STRUCTURE_NEXT.format(builtin=S.ED_FROM_BUILTIN)
    busy = RunStoreError(S.RUNS_FILE_BUSY.format(path="stats.csv"))
    busy.__cause__ = PermissionError(errno.EACCES, "Permission denied")
    assert explain(busy).message == S.ERR_PERMISSION


def test_reason_is_short_and_korean() -> None:
    assert reason(ValueError(S.PLAY_RECORD_NOT_RECORD)) == S.PLAY_RECORD_NOT_RECORD
    assert reason(PermissionError(errno.EACCES, "Permission denied")) == S.ERR_PERMISSION
    assert reason(json.JSONDecodeError("Expecting value", "x", 0)) == "JSONDecodeError"


# ---------------------------------------------------------------------------
# show_error 상자
# ---------------------------------------------------------------------------
def test_show_error_box_has_next_step_report_button_details_and_log(qtbot: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab import diagnostics
    from evolab.app.report import dialog

    parent = QWidget()
    qtbot.addWidget(parent)
    parent.show()
    opened: list[tuple[Any, str]] = []
    monkeypatch.setattr(dialog, "open_for", lambda p, why: opened.append((p, why)))
    exc = raised(PermissionError(errno.EACCES, "Permission denied", "stats.csv"))
    box = show_error(parent, S.ED_SAVE_FAILED, exc)  # 막지 않는다: 곧바로 돌아온다
    assert box.isVisible() and boxes_of(parent) == [box] and box.objectName() == errors.BOX_NAME
    assert box.windowTitle() == S.ED_SAVE_FAILED and box.text() == S.ERR_PERMISSION
    assert box.informativeText() == f"{S.ERR_NEXT_LABEL}: {S.ERR_PERMISSION_NEXT.format(menu=S.MENU_SETTINGS)}"
    assert "PermissionError" in box.detailedText() and "Traceback" in box.detailedText()
    texts = [b.text() for b in box.buttons()]
    assert S.ERR_REPORT_BUTTON in texts and box.button(QMessageBox.StandardButton.Ok) is not None
    line = json.loads(diagnostics.errors_path().read_text(encoding="utf-8").splitlines()[-1])
    assert line["kind"] == "shown" and line["message"] == f"{S.ED_SAVE_FAILED}: {S.ERR_PERMISSION}"
    assert "PermissionError" in line["detail"]

    box.report_button.click()  # 상자를 닫고 다음 차례에 묶음 대화 상자(까닭 "error")
    qtbot.waitUntil(lambda: bool(opened), timeout=5000)
    assert opened == [(parent, errors.REPORT_REASON)] and boxes_of(parent) == []
    assert not shiboken6.isValid(box)  # 닫힌 상자는 지워진다(WA_DeleteOnClose)

    ok = show_error(parent, S.LAB_ERROR_TITLE, "학습 프로세스가 갑자기 끝났습니다", next_step="다시 해 보세요.")
    assert ok.text() == "학습 프로세스가 갑자기 끝났습니다" and ok.informativeText().endswith("다시 해 보세요.")
    assert ok.detailedText() == ""  # 글만 받았으면 자세히가 없다
    ok.button(QMessageBox.StandardButton.Ok).click()
    qtbot.wait(20)
    assert len(opened) == 1 and boxes_of(parent) == []


def test_error_box_closes_with_its_window(qtbot: Any) -> None:
    """상자를 띄운 창이 닫히면 그 상자도 닫힌다: 보인 창(숨김)과 한 번도 보이지 않은 창(만드는 중 오류, 닫기 사건)."""
    shown, never = QWidget(), QWidget()
    for w in (shown, never):
        qtbot.addWidget(w)
    shown.show()
    qtbot.waitExposed(shown)
    a = show_error(shown, "t", "a")
    b = show_error(never, "t", "b")
    child = QWidget(shown)
    c = show_error(child, "t", "c")  # 창 안 위젯이 띄운 상자는 그 창을 따른다
    assert a.isVisible() and b.isVisible() and c.isVisible()
    shown.close()
    never.close()
    assert boxes_of(shown) == boxes_of(never) == boxes_of(child) == []


def test_show_error_text_is_plain_not_html(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    box = show_error(parent, "t", "<b>x</b> & y", message="a < b <i>c</i>")
    labels = {lab.objectName(): lab for lab in box.findChildren(QWidget) if lab.objectName().startswith("qt_msgbox_")}
    from PySide6.QtCore import Qt

    assert labels["qt_msgbox_label"].textFormat() == Qt.TextFormat.PlainText  # type: ignore[attr-defined]
    assert labels["qt_msgbox_informativelabel"].textFormat() == Qt.TextFormat.PlainText  # type: ignore[attr-defined]
    box.close()


# ---------------------------------------------------------------------------
# AST 규칙
# ---------------------------------------------------------------------------
def _is_message_box(func: ast.expr) -> bool:
    """QMessageBox.<무엇> 또는 ….QMessageBox.<무엇> 호출."""
    if not isinstance(func, ast.Attribute):
        return False
    owner = func.value
    return (isinstance(owner, ast.Name) and owner.id == "QMessageBox") or (
        isinstance(owner, ast.Attribute) and owner.attr == "QMessageBox")


def raw_exception_boxes(source: str) -> list[int]:
    """except 이름을(그대로, str(이름), 이름.속성, format·f-string 안에) QMessageBox 호출 인자에 넣은 줄."""
    bad: set[int] = set()
    for handler in ast.walk(ast.parse(source)):
        if not isinstance(handler, ast.ExceptHandler) or not handler.name:
            continue
        for node in ast.walk(handler):
            if isinstance(node, ast.Call) and _is_message_box(node.func):
                args = [*node.args, *(k.value for k in node.keywords)]
                if any(isinstance(n, ast.Name) and n.id == handler.name for a in args for n in ast.walk(a)):
                    bad.add(node.lineno)
    return sorted(bad)


def test_ast_rule_catches_every_raw_form() -> None:
    src = '''
try:
    f()
except OSError as exc:
    QMessageBox.warning(self, T, str(exc))
    QMessageBox.warning(self, T, exc)
    QMessageBox.critical(self, T, S.X.format(reason=exc))
    QtWidgets.QMessageBox.information(self, T, f"{exc}")
    QMessageBox.warning(self, T, text=exc.strerror)
    QMessageBox.warning(self, T, S.FRIENDLY)
    show_error(self, T, exc)
'''
    assert raw_exception_boxes(src) == [5, 6, 7, 8, 9]


def test_no_raw_exception_message_boxes_in_app() -> None:
    found: dict[str, list[int]] = {}
    for path in sorted(APP.rglob("*.py")):
        if path == APP / "errors.py":
            continue
        lines = raw_exception_boxes(path.read_text(encoding="utf-8"))
        if lines:
            found[path.relative_to(APP).as_posix()] = lines
    main = found.pop("main.py", [])
    assert found == {}, found
    if "show_error" in (APP / "main.py").read_text(encoding="utf-8"):
        assert main == [], main
    else:  # I 묶음이 main.py를 고치기 전: 설계 9.1에 센 세 곳보다 늘지 않게
        assert len(main) <= MAIN_PENDING, main


# ---------------------------------------------------------------------------
# 화면마다 바꾼 곳
# ---------------------------------------------------------------------------
@pytest.fixture()
def lab(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from evolab.app.lab.view import LabView

    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path / "runs")
    view = LabView(settings=QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.show()
    try:
        yield view
    finally:
        view.client.abandon(timeout=60)
        view.shutdown()


def test_lab_error_boxes(lab: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plain: list[Any] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: plain.append(a)))
    assert not lab.load_env(tmp_path / "없음.evoenv")
    assert not lab.use_env_text("{ 깨진", "x", None)
    (missing, broken) = [b.friendly for b in boxes_of(lab)]
    assert missing.message == S.ERR_NOT_FOUND.format(path=tmp_path / "없음.evoenv")
    assert broken.next_step == S.ERR_ENV_FILE_NEXT.format(issues=S.EDP_TAB_ISSUES) and "Traceback" in broken.detail
    for b in boxes_of(lab):
        b.close()

    assert lab.load_env(LANDER)
    lab.popsize.setValue(2)
    lab.method.setCurrentIndex(lab.method.findData("cmaes"))
    import dataclasses

    from evolab.engine.trainer import TrainConfig

    bad = dataclasses.replace(TrainConfig.for_new_run(lab.spec), sigma0=-1.0)
    monkeypatch.setattr(lab, "config_from_form", lambda: bad)
    assert not lab.start_new()
    (cfg,) = [b.friendly for b in boxes_of(lab)]
    assert cfg.message.startswith(S.TRAIN_CONFIG_INVALID.split("{")[0]) and cfg.next_step == S.ERR_SETTINGS_NEXT
    boxes_of(lab)[0].close()

    # 학습 프로세스가 보낸 글(변수 경로 둘)과 다시 시뮬레이션 실패(변수 경로, 설계 9.1의 lab/view.py:1597)
    lab._on_applied({"ok": False, "message": "학습 설정 값이 잘못되었습니다: popsize = 1"})
    lab._on_failed(S.SESSION_LOST.format(code=3221225477), True)
    lab._replay_done((lab._replay_tag, ("timeline", ["x"])), raised(ZeroDivisionError("float division by zero")))
    applied, failed, replay = [b.friendly for b in boxes_of(lab)]
    assert applied.next_step == S.ERR_SETTINGS_NEXT
    assert failed.message == S.SESSION_LOST.format(code=3221225477) and lab.state == "closed"
    assert replay.message == S.ERR_UNEXPECTED.format(kind="ZeroDivisionError") and "float division" in replay.detail
    for b in boxes_of(lab):
        b.close()

    assert not lab.open_run(tmp_path / "없는 실행")
    (run,) = [b.friendly for b in boxes_of(lab)]
    assert run.next_step == S.ERR_RUN_STORE_NEXT.format(records=S.TAB_RECORDS)
    assert plain == []  # 날것의 QMessageBox.warning은 하나도 없었다


def test_editor_error_boxes(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.editor.view import EditorView

    view = EditorView()
    qtbot.addWidget(view)
    view.show()
    bad = tmp_path / "깨진.evoenv"
    bad.write_text("{ 이건 JSON이 아님", encoding="utf-8")
    assert not view.open_file(bad)
    assert not view.open_file(tmp_path / "없음.evoenv")
    assert not view.load_text("[]")
    broken, missing, text = [b.friendly for b in boxes_of(view)]
    assert broken.message.startswith(S.SPEC_INVALID_FILE.format(count=1))
    assert broken.next_step == text.next_step == S.ERR_ENV_STRUCTURE_NEXT.format(builtin=S.ED_FROM_BUILTIN)
    assert missing.message == S.ERR_NOT_FOUND.format(path=tmp_path / "없음.evoenv")
    for b in boxes_of(view):
        b.close()

    view.start_from_builtin("lander.evoenv")
    monkeypatch.setattr(view.doc, "save", lambda *_a: (_ for _ in ()).throw(OSError(errno.ENOSPC, "No space left")))
    assert not view.save_as(tmp_path / "a.evoenv")
    view.doc.path = tmp_path / "a.evoenv"
    assert not view.save()
    assert [b.friendly.message for b in boxes_of(view)] == [S.ERR_DISK_FULL, S.ERR_DISK_FULL]


def test_play_window_error_boxes(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.play import window as play_window
    from evolab.app.play.window import PlayWindow
    from evolab.spec.io import load_env

    w = PlayWindow(load_env(LANDER), seed=5)
    qtbot.addWidget(w)
    err = EpisodeBuildError("generators.ground: 모양을 만들 수 없음")
    monkeypatch.setattr(play_window, "Episode", lambda *_a, **_k: (_ for _ in ()).throw(err))
    assert not w.start(9)
    (built,) = [b.friendly for b in boxes_of(w)]
    assert built.message == w.build_failed_text(9, err) and built.next_step == S.ERR_BUILD_NEXT
    boxes_of(w)[0].close()

    bad = tmp_path / "깨진.evoplay"
    bad.write_text("not json", encoding="utf-8")
    other = tmp_path / "다른.evoplay"
    other.write_text(json.dumps({"format": "다른 것"}), encoding="utf-8")
    w.play_record(bad)
    w.play_record(other)
    garbage, foreign = [b.friendly for b in boxes_of(w)]
    assert garbage.message == S.PLAY_RECORD_BAD.format(reason="JSONDecodeError") and "Expecting value" in garbage.detail
    assert foreign.message == S.PLAY_RECORD_BAD.format(reason=S.PLAY_RECORD_NOT_RECORD)
    assert garbage.next_step == foreign.next_step == S.ERR_RECORD_NEXT
    for b in boxes_of(w):
        b.close()

    monkeypatch.setattr(play_window.QFileDialog, "getSaveFileName", lambda *_a, **_k: (str(tmp_path / "r.evoplay"), ""))
    assert w.record is not None
    w.record.score = 0.0  # 끝난 기록만 저장한다
    monkeypatch.setattr(w.record, "save", lambda _p: (_ for _ in ()).throw(PermissionError(errno.EACCES, "denied")))
    w.save_record()
    (saved,) = [b.friendly for b in boxes_of(w)]
    assert saved.message == S.PLAY_SAVE_FAILED.format(reason=S.ERR_PERMISSION)
    assert saved.next_step == S.ERR_PERMISSION_NEXT.format(menu=S.MENU_SETTINGS)
    assert not any(p.exists() for p in (tmp_path / "r.evoplay",))
    assert os.environ[paths.STATE_DIR_ENV].startswith(str(tmp_path))


def _is_error_wrapper(func: ast.expr) -> bool:
    """화면 자체의 오류 상자 도우미 self.<…error…>(…)(예: records/view.py show_error(text) = QMessageBox.warning)."""
    return isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self" and \
        "error" in func.attr.lower()


def raw_exception_wrapper_calls(source: str) -> list[tuple[str, int]]:
    """except 이름을 self.<…error…>(…) 도우미 인자에 넣은 곳 (함수 이름, 줄). T18 규칙은 QMessageBox.* 직접 호출만 본다."""
    tree = ast.parse(source)
    owner: dict[int, str] = {}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            for node in ast.walk(fn):
                owner.setdefault(id(node), fn.name)  # 바깥 함수부터 돌므로 안쪽 함수가 나중에 덮지 않게 setdefault
    found: list[tuple[str, int]] = []
    for handler in ast.walk(tree):
        if not isinstance(handler, ast.ExceptHandler) or not handler.name:
            continue
        for node in ast.walk(handler):
            if isinstance(node, ast.Call) and _is_error_wrapper(node.func):
                args = [*node.args, *(k.value for k in node.keywords)]
                if any(isinstance(n, ast.Name) and n.id == handler.name for a in args for n in ast.walk(a)):
                    found.append((owner.get(id(node), "?"), node.lineno))
    return found


RAW_WRAPPER_BASELINE: dict[str, int] = {
    "records/view.py": 0, "race/view.py": 0, "compare/repeat.py": 0, "compare/view.py": 0,
}
"""v2.1 그대로 남았던 「화면 자체 show_error(text) 도우미에 예외 글」 자리(검토 수정 때 센 값: 기록 5, 경주 1, 반복 실행 1, 비교 1).
설계 7.1이 이 화면들을 v3.0에서 「바꾸지 않음」으로 두어 v3.0은 고치지 않았고, v3.1(F 묶음)이 도우미 몸통을 errors.show_error로,
까닭을 errors.reason으로 바꿔 모두 0이 됐다. 이 수는 줄기만 해야 한다."""


def test_screen_error_wrappers_with_raw_exception_text_only_go_down() -> None:
    """래퍼 자리 수는 늘지 않는다(T18이 못 보는 틈의 톱니바퀴). 새 화면 코드는 errors.show_error를 쓴다."""
    src = '''
class V:
    def f(self):
        try:
            g()
        except OSError as exc:
            self.show_error(str(exc))
            self.show_error(S.X.format(reason=exc))
            self._error(f"{exc}")
            self.show_error(S.FRIENDLY)
            self.status(str(exc))
'''
    assert raw_exception_wrapper_calls(src) == [("f", 7), ("f", 8), ("f", 9)]
    counts: dict[str, int] = {}
    for path in sorted(APP.rglob("*.py")):
        if path == APP / "errors.py":
            continue
        n = len(raw_exception_wrapper_calls(path.read_text(encoding="utf-8")))
        if n:
            counts[path.relative_to(APP).as_posix()] = n
    grown = {k: v for k, v in counts.items() if v > RAW_WRAPPER_BASELINE.get(k, 0)}
    assert grown == {}, counts

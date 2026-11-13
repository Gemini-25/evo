"""T17: 빈 상태 안내 (v3 설계 D17·D24, 4.3, 5.2).

실험실 시작 안내가 보이고 그 단추가 학습을 시작하며 학습 중에는 숨는다, 대시보드 0·1줄 안내, 에디터 빈 문서 안내,
안내 사각형 밖 클릭(카드 글 위 클릭도)은 아래 위젯으로 간다. 진짜 창 이벤트 경로(QWindow)로 누른다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMainWindow, QWidget

from evolab import paths
from evolab.app import strings as S
from evolab.app.help import hints
from evolab.app.help.hints import HintFrame, hint_of, install_hints


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """실제 데이터·설정 폴더를 건드리지 않는다(설계 D5). 안내는 스스로 뜨지 않게."""
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv("EVOLAB_NO_TOUR", "1")
    return tmp_path


@pytest.fixture()
def window(qtbot: Any, isolated: Path) -> Any:
    from evolab.app.main import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1280, 800)
    w.show()
    qtbot.waitExposed(w)
    install_hints(w)  # I 묶음 help/setup.install이 부르기 전에도 같은 시험이 되게(두 번 불러도 한 번)
    try:
        yield w
    finally:
        w.lab.client._proc = None  # 가짜 「학습 중」을 치운다(창을 닫을 때 학습 프로세스를 기다리지 않게)


class Presses(QObject):
    """그 위젯이 받은 마우스 누름 수."""

    def __init__(self, widget: QWidget) -> None:
        super().__init__(widget)
        self.count = 0
        widget.installEventFilter(self)

    def eventFilter(self, _obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.MouseButtonPress:
            self.count += 1
        return False


def click(window: QWidget, widget: QWidget, local: QPoint) -> None:
    """진짜 창 이벤트 경로(QWindow → 위젯 찾기)로 widget 안 local 자리를 누른다."""
    QTest.mouseClick(window.windowHandle(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     widget.mapTo(window, local))


def lab_hint(window: Any) -> HintFrame:
    frame = hint_of(window.lab.canvas)
    assert frame is not None
    frame.refresh()
    return frame


def test_install_is_idempotent_and_finds_the_three_places(window: Any) -> None:
    install_hints(window)
    frames = [hint_of(w) for w in (window.lab.canvas, window.lab.dashboard, window.editor.canvas)]
    assert all(isinstance(f, HintFrame) for f in frames)
    assert window.evolab_hints == frames
    assert sum(isinstance(c, HintFrame) for c in window.lab.canvas.children()) == 1
    from evolab.app.help import tour

    assert hints.TOUR_STATE_KEY == tour.STATE_KEY


def test_lab_idle_hint_starts_training_and_hides_while_training(window: Any, qtbot: Any,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    lab = window.lab
    frame = lab_hint(window)
    assert frame.isVisible() and frame.title_label.text() == S.HINT_LAB_IDLE_TITLE
    assert frame.text() == S.HINT_LAB_IDLE_ANY.format(start=S.LAB_START, env=lab.spec.file.name,
                                                      pop=lab.popsize.value())
    start, tour_btn = frame.buttons[:2]  # 카드에 단추가 더 생겨도(다듬기 U10) 앞의 둘이 이 둘이다
    assert (start.text(), tour_btn.text()) == (S.LAB_START, S.HINT_LAB_TOUR)
    assert start.isVisible() and start.isEnabled() and tour_btn.isVisible()
    # 카드는 캔버스 가운데, 캔버스 안, 단추 줄은 카드 안
    canvas = lab.canvas
    assert canvas.rect().contains(frame.geometry())
    assert abs(frame.geometry().center().x() - canvas.rect().center().x()) <= 1
    assert frame.geometry().contains(frame.bar.geometry())

    lab.method.setCurrentIndex(lab.method.findData("sa"))  # 방법이 바뀌면 바로 고친다(300ms 확인을 기다리지 않음)
    assert frame.text() == S.HINT_LAB_IDLE_ANY.format(start=S.LAB_START, env=lab.spec.file.name,
                                                      pop=lab.sa_neighbors.value())
    lab.method.setCurrentIndex(lab.method.findData("cmaes"))
    # 환경을 바꾸면 바로 고친다(검토 수정: 「점프 게임 32개가…」 대신 개체 수와 환경 이름)
    from evolab import paths

    jump = paths.builtin_env_dir() / "jump.evoenv"
    assert lab.use_env_text(jump.read_text(encoding="utf-8-sig"), str(jump), None, jump)
    assert frame.text() == S.HINT_LAB_IDLE_ANY.format(start=S.LAB_START, env=lab.spec.file.name,
                                                      pop=lab.popsize.value())
    assert "「" + lab.spec.file.name + "」" in frame.text()

    started: list[Any] = []

    def fake_start(request: Any) -> None:  # 학습 프로세스를 띄우지 않고 「학습 중」으로
        started.append(request)
        lab.client._proc = object()

    monkeypatch.setattr(lab.client, "start", fake_start)
    pressed = Presses(canvas)
    click(window, start, start.rect().center())  # 카드 위 단추(진짜 클릭)
    assert len(started) == 1 and started[0].kind == "new" and pressed.count == 0
    assert not frame.isVisible() and not frame.bar.isVisible()  # 학습 중에는 숨는다
    frame.refresh()
    assert not frame.isVisible()
    lab.client._proc = None
    lab.state = "idle"
    frame.refresh()
    assert frame.isVisible()  # 학습이 끝났고 실행을 열지 않았으면 다시
    lab.run_dir = Path("x")
    frame.refresh()
    assert not frame.isVisible()  # 실행을 열었으면 숨는다
    lab.run_dir = None


def test_lab_hint_clicks_outside_buttons_reach_the_canvas(window: Any) -> None:
    canvas = window.lab.canvas
    frame = lab_hint(window)
    pressed = Presses(canvas)
    click(window, canvas, QPoint(5, 5))  # 카드 사각형 밖
    assert pressed.count == 1
    text = frame.text_label.geometry().center()  # 카드 안 글 위: 카드는 마우스를 받지 않는다
    click(window, canvas, frame.mapTo(canvas, text))
    assert pressed.count == 2
    assert canvas.childAt(frame.mapTo(canvas, text)) is not frame


def test_tour_button_hides_when_done_and_starts_the_tour(window: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.help import tour

    calls: list[Any] = []
    monkeypatch.setattr(tour, "start_tour", lambda w, s, **k: calls.append((w, s)))
    frame = lab_hint(window)
    tour_btn = frame.buttons[1]
    tour_btn.click()
    assert calls == [(window, window.lab.settings)]
    window.lab.settings.setValue(hints.TOUR_STATE_KEY, "done")
    frame.refresh()
    assert tour_btn.isHidden() and frame.buttons[0].isVisible()
    window.lab.settings.setValue(hints.TOUR_STATE_KEY, "skipped")  # 건너뛴 사람에게는 남긴다(설계: done이면 숨김)
    frame.refresh()
    assert tour_btn.isVisible()


def test_lab_hint_without_environment(qtbot: Any, isolated: Path) -> None:
    from evolab.app.lab.view import LabView

    win = QMainWindow()
    qtbot.addWidget(win)
    win.lab = LabView(settings=QSettings(str(isolated / "s.ini"), QSettings.Format.IniFormat))  # type: ignore[attr-defined]
    win.setCentralWidget(win.lab)  # type: ignore[attr-defined]
    win.resize(1000, 700)
    win.show()
    qtbot.waitExposed(win)
    install_hints(win)
    frame = hint_of(win.lab.canvas)  # type: ignore[attr-defined]
    assert frame is not None
    frame.refresh()
    assert frame.isVisible() and frame.text() == S.LAB_ENV_NONE and frame.title_label.isHidden()
    assert not frame.buttons[0].isEnabled()  # 환경이 없으면 시작 단추는 회색(패널 단추와 같음)


def test_dashboard_hint_for_zero_and_one_rows(window: Any) -> None:
    lab = window.lab
    frame = hint_of(lab.dashboard)
    assert frame is not None and frame.passthrough
    frame.refresh()
    assert not frame.isVisible()  # 쉬고 있고 실행을 열지 않았으면(실험실 캔버스 안내가 맡음)
    lab.client._proc = object()  # 학습 중, 통계 0줄
    lab.rows = []
    frame.refresh()
    assert frame.isVisible() and frame.text() == S.HINT_DASH_EMPTY and frame.buttons == []
    lab.rows = [{"generation": 1}]
    frame.refresh()
    assert frame.text() == S.HINT_DASH_ONE
    center = frame.geometry().center()
    assert lab.dashboard.childAt(center) is not frame  # 마우스는 아래 타일로
    lab.rows = [{"generation": 1}, {"generation": 2}]
    frame.refresh()
    assert not frame.isVisible()
    lab.client._proc = None
    lab.run_dir = Path("x")  # 연 실행, 통계 1줄
    lab.rows = [{"generation": 1}]
    frame.refresh()
    assert frame.isVisible() and frame.text() == S.HINT_DASH_ONE
    lab.run_dir, lab.rows = None, []


def test_editor_empty_document_hint(window: Any, qtbot: Any) -> None:
    editor = window.editor
    window.tabs.setCurrentWidget(editor)
    editor.doc.new()
    frame = hint_of(editor.canvas)
    assert frame is not None
    qtbot.waitUntil(frame.isVisible, timeout=2000)  # 문서가 바뀌면 곧바로, 아니어도 300ms마다
    assert frame.text() == S.HINT_EDITOR_EMPTY.format(preset=S.ED_NEW_PRESET, shapes=S.EDC_GROUP_SHAPES,
                                                      builtin=S.ED_FROM_BUILTIN)
    (btn,) = frame.buttons
    assert btn.text() == S.ED_FROM_BUILTIN
    pressed = Presses(editor.canvas)
    click(window, editor.canvas, QPoint(4, 4))
    click(window, editor.canvas, frame.mapTo(editor.canvas, frame.text_label.geometry().center()))
    assert pressed.count == 2  # 카드 밖과 카드 글 위 클릭(그리기 도구 끌기)은 캔버스로
    click(window, btn, btn.rect().center())
    menu = editor.builtin_menu
    qtbot.waitUntil(menu.isVisible, timeout=2000)  # 「내장 예제에서 시작」 메뉴가 펼쳐진다(막지 않는 popup)
    menu.close()
    editor.start_from_builtin("lander.evoenv")
    qtbot.waitUntil(lambda: not frame.isVisible(), timeout=2000)
    editor.doc.new()
    qtbot.waitUntil(frame.isVisible, timeout=2000)


def test_anchor_still_finds_the_toolbar_button(window: Any, qtbot: Any) -> None:
    """에디터 안내의 단추 글이 도구 막대 「내장 예제에서 시작」과 같아도 연결표(editor.builtin)는 도구 막대 단추를 가리킨다."""
    from PySide6.QtWidgets import QToolButton

    from evolab.app.help import anchors

    window.tabs.setCurrentWidget(window.editor)
    window.editor.doc.new()
    frame = hint_of(window.editor.canvas)
    assert frame is not None
    qtbot.waitUntil(frame.isVisible, timeout=2000)
    found = anchors.resolve(window, "editor.builtin")
    assert isinstance(found, QToolButton) and found is not frame.buttons[0]
    assert frame.buttons[0].toolTip() == S.HELP_TIPS["editor.builtin"]
    assert lab_hint(window).buttons[0].toolTip() == S.HELP_TIPS["lab.start"]


def test_hidden_tab_keeps_state_until_shown(window: Any, qtbot: Any) -> None:
    """보이지 않는 탭의 대상은 300ms 확인을 건너뛰고, 탭이 보이면(Show) 다시 본다."""
    editor = window.editor
    window.tabs.setCurrentWidget(window.lab)
    frame = hint_of(editor.canvas)
    assert frame is not None
    editor.doc.new()
    window.tabs.setCurrentWidget(editor)
    qtbot.waitUntil(frame.isVisible, timeout=2000)
    rect = QRect(frame.geometry())
    assert editor.canvas.rect().contains(rect)

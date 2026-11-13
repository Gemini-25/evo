"""6.2.0 화면 다듬기, 묶음 Ha (다듬기 계획 2.5 U32·U33, 2.2 U7의 카드 단추).

- U32 주 창의 틀: 상태 막대는 처음부터 있다(첫 알림 때 가운데가 줄지 않음). 오른쪽 꼬리표(`state_chip`)가 실험실 상태를 보이고
  최소 폭이 1이라 주 창 최소 폭을 넓히지 않는다. 실행이 없으면 제목은 APP_TITLE, 실행이 있으면 「{환경} · {상태} {n}세대 — 진화 실험실」.
  꼬리표·제목은 상태가 바뀔 때와 학습 중에만 도는 1초 타이머로만 고치고, 글이 같으면 다시 적지 않는다. 모서리 「? 도움말」은 보통 단추 모양.
- U33 말풍선 단추: 기본 단추(Enter)만 호박색 채움 + 굵게, 나머지는 종이색. 테마와 무관한 같은 색. 환영 카드는 제목 +2pt, 단계 수 숨김.
- U7 캔버스 카드의 「▶ 학습 시작」은 으뜸 단추(늘 어두운 카드라 어두운 테마 값). 패널 단추가 꺼지면 같은 까닭으로 꺼진다.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtGui import QColor, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QStatusBar

from evolab import paths
from evolab.app import strings as S
from evolab.app import style
from evolab.app.help import coach as C
from evolab.app.help.hints import hint_of


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from evolab.app.help import context
    from evolab.app.help import tooltips as T

    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv("EVOLAB_NO_TOUR", "1")
    yield tmp_path
    app = QApplication.instance()
    if isinstance(app, QApplication):
        context.uninstall_shortcuts(app)
        T.uninstall(app)


@pytest.fixture()
def window(qtbot: Any, isolated: Path) -> Any:
    from evolab.app.main import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1280, 720)
    w.show()
    QTest.qWaitForWindowExposed(w)
    return w


# ---------------------------------------------------------------------------
# U32
# ---------------------------------------------------------------------------
def test_status_bar_exists_from_the_start_and_keeps_the_middle_height(qtbot: Any, isolated: Path) -> None:
    from evolab.app.main import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    assert w.findChild(QStatusBar) is not None  # 만들 때 이미 있다(첫 알림 때 생기지 않음)
    w.resize(1280, 720)
    w.show()
    QTest.qWaitForWindowExposed(w)
    before = w.centralWidget().height()
    w.statusBar().showMessage(S.MAIN_ENV_SENT, 8000)
    QApplication.processEvents()
    assert w.centralWidget().height() == before
    assert w.statusBar().currentMessage() == S.MAIN_ENV_SENT  # 알림은 상태 막대 그대로(기존 테스트가 읽는 자리)
    assert w.minimumSizeHint().height() <= 720 - 40  # 1280×720 작업 영역 안(제목 줄 여유)


def test_chip_does_not_widen_the_window(window: Any) -> None:
    chip = window.state_chip
    assert isinstance(chip, style.StateChip) and chip.objectName() == "state_chip"
    assert chip.minimumWidth() == 1 and chip.isVisible()
    width = window.minimumSizeHint().width()
    chip.set_state("running", "가" * 200)
    QApplication.processEvents()
    assert window.minimumSizeHint().width() == width


def test_title_and_chip_follow_the_lab(window: Any, qtbot: Any) -> None:
    lab = window.lab
    assert window.windowTitle() == S.APP_TITLE  # 실행이 없으면 그대로(tests/test_app.py)
    assert window.state_chip.label.text() == S.LAB_STATE_NAMES["idle"]
    assert window.state_chip.dot.kind == "idle"
    assert not window._state_timer.isActive()
    env = lab.spec.file.name
    lab.run_dir, lab.state, lab.generation = Path("run"), "running", 12
    titles: list[str] = []
    window.windowTitleChanged.connect(titles.append)
    try:
        window.refresh_state()
        name = S.LAB_STATE_NAMES["running"]
        want = S.MAIN_TITLE_RUN.format(env=env, state=name, gen=12, app=S.APP_TITLE)
        assert window.windowTitle() == want and want.endswith(S.APP_TITLE) and "12" in want
        assert window.state_chip.label.text() == S.MAIN_CHIP.format(state=name, gen=12)
        assert window.state_chip.dot.kind == "running"
        assert window._state_timer.isActive()  # 학습 중에만 도는 1초 타이머
        window.refresh_state()
        assert titles == [want]  # 글이 같으면 다시 적지 않는다
        lab.generation = 13  # 세대마다 부르지 않아도 타이머가 1초 안에 맞춘다
        qtbot.waitUntil(lambda: "13" in window.windowTitle(), timeout=3000)
        lab.state = "paused"
        qtbot.waitUntil(lambda: not window._state_timer.isActive(), timeout=3000)  # 학습 중이 아니면 타이머가 멈춘다
        assert window.state_chip.dot.kind == "paused"
        assert window.windowTitle() == S.MAIN_TITLE_RUN.format(env=env, state=S.LAB_STATE_NAMES["paused"], gen=13,
                                                               app=S.APP_TITLE)
    finally:
        lab.run_dir, lab.state, lab.generation = None, "idle", 0
        window.refresh_state()
    assert window.windowTitle() == S.APP_TITLE and not window._state_timer.isActive()


def test_state_changed_signal_refreshes_when_the_lab_has_it(window: Any) -> None:
    lab = window.lab
    signal = getattr(lab, "state_changed", None)
    if signal is None:
        pytest.skip("LabView.state_changed(Hb의 계약 3.3)이 아직 없다: run_started·closed와 타이머로 맞춘다")
    lab.run_dir, lab.state, lab.generation = Path("run"), "paused", 3
    try:
        signal.emit("paused")
        assert window.state_chip.dot.kind == "paused"
        assert window.state_chip.label.text() == S.MAIN_CHIP.format(state=S.LAB_STATE_NAMES["paused"], gen=3)
    finally:
        lab.run_dir, lab.state, lab.generation = None, "idle", 0
        window.refresh_state()


def test_corner_help_button_is_a_normal_button(window: Any) -> None:
    corner = window.tabs.cornerWidget()
    assert isinstance(corner, QPushButton) and corner.text() == S.HELP_CORNER_BUTTON
    assert not corner.isFlat() and corner.icon().isNull()


# ---------------------------------------------------------------------------
# U33
# ---------------------------------------------------------------------------
def _overlay(qtbot: Any, window: Any) -> C.CoachOverlay:
    steps = [
        C.CoachStep("welcome", lambda: "본문", kind="welcome", title=S.ONB_WELCOME_TITLE,
                    buttons=(("a", lambda: None), ("b", lambda: None)), default=0),
        C.CoachStep("two", lambda: "둘", kind="next"),
    ]
    ov = C.CoachOverlay(window, steps)
    ov.start()
    qtbot.waitUntil(lambda: ov.bubble.isVisible(), timeout=3000)
    return ov


def _fill(button: QPushButton) -> QColor:
    """단추의 바탕색: 글이 없는 위쪽 가장자리 안쪽 한 점."""
    img = button.grab().toImage()
    ratio = img.devicePixelRatio() or 1.0
    return img.pixelColor(int(button.width() * 0.5 * ratio), int(3 * ratio))


def test_bubble_buttons_default_is_amber_and_others_are_paper(qtbot: Any, window: Any) -> None:
    ov = _overlay(qtbot, window)
    try:
        assert ov.bubble.styleSheet() == C.bubble_style_sheet()
        buttons = [b for b in ov.bubble.findChildren(QPushButton) if b.isVisible()]
        default = [b for b in buttons if b.property(C.DEFAULT_PROPERTY)]
        assert [b.text() for b in default] == ["a"]
        other = next(b for b in buttons if b.text() == "b")
        assert _fill(default[0]).name() == C.RING_COLOR
        assert _fill(other).name() == C.PAPER_COLOR
        assert default[0].font().bold() or "font-weight: bold" in C.bubble_style_sheet()
        # 환영 카드: 단계 수 숨김, 제목 +2pt
        assert ov.counter.text() == "" and not ov.counter.isVisible()
        assert ov.title_label.font().pointSizeF() >= ov.text_label.font().pointSizeF() + 2
        assert ov.bubble.width() == C.BUBBLE_WIDTH  # 폭은 그대로
        ov.next()
        assert ov.counter.text() == S.ONB_PROGRESS.format(i=2, n=2)  # 카드가 아닌 단계는 단계 수 그대로
        assert ov.title_label.font().pointSizeF() < ov._card_title_font.pointSizeF()
    finally:
        ov.finish("done")


def test_bubble_buttons_look_the_same_in_the_dark_theme(qtbot: Any, window: Any) -> None:
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    old = app.palette()
    pal = QPalette(old)
    pal.setColor(QPalette.ColorRole.Window, QColor("#323232"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#f0efec"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#323232"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#f0efec"))
    app.setPalette(pal)
    try:
        assert style.is_dark()
        ov = _overlay(qtbot, window)
        try:
            buttons = {b.text(): b for b in ov.bubble.findChildren(QPushButton) if b.isVisible()}
            assert _fill(buttons["a"]).name() == C.RING_COLOR
            assert _fill(buttons["b"]).name() == C.PAPER_COLOR  # 크림 종이 위에 어두운 단추가 뜨지 않는다
        finally:
            ov.finish("done")
    finally:
        app.setPalette(old)
        style.refresh_theme()


# ---------------------------------------------------------------------------
# U7: 캔버스 카드의 으뜸 단추
# ---------------------------------------------------------------------------
def test_canvas_card_start_button_is_primary_on_the_dark_card(window: Any, qtbot: Any) -> None:
    lab = window.lab
    window.tabs.setCurrentWidget(lab)
    frame = hint_of(lab.canvas)
    assert frame is not None
    qtbot.waitUntil(frame.isVisible, timeout=3000)
    start = frame.buttons[0]
    assert start.text() == S.LAB_START
    assert start.property(style.ROLE_PROPERTY) == "primary"
    assert style.UI_DARK.accent in start.styleSheet()  # 테마와 무관하게 어두운 카드 값
    assert start in style.primaries(lab)
    # 패널 단추가 꺼지면 카드 단추도 같은 까닭으로 꺼진다(풍선 = 보통 풍선 + 까닭)
    style.set_enabled(lab.btn_start, False, "why-off")
    try:
        frame.refresh()
        assert not start.isEnabled() and style.why_off(start) == "why-off"
        assert style.tip_text(start).endswith("why-off") and start.toolTip() == S.HELP_TIPS["lab.start"]
    finally:
        style.set_enabled(lab.btn_start, True)
        frame.refresh()
    assert start.isEnabled() and style.why_off(start) == "" and style.tip_text(start) == start.toolTip()

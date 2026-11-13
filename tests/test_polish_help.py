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


# ---------------------------------------------------------------------------
# U10: 한 카드 안의 세 상태(하면 좋음)
# ---------------------------------------------------------------------------
def _visible_texts(frame: Any) -> list[str]:
    return [b.text() for b in frame.buttons if not b.isHidden()]


def test_canvas_card_saved_run_state(window: Any, qtbot: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.help import hints as H

    lab = window.lab
    window.tabs.setCurrentWidget(lab)
    frame = hint_of(lab.canvas)
    assert frame is not None
    played: list[int] = []
    monkeypatch.setattr(lab, "play_generation", lambda g: played.append(g) or True)
    lab.run_dir, lab.generation = Path("run"), 7
    try:
        assert H.lab_card_state(lab) == "saved"
        frame.refresh()
        assert frame.isVisible() and frame.title_label.text() == S.HINT_LAB_SAVED_TITLE.format(gen=7)
        assert frame.text() == S.HINT_LAB_SAVED.format(cont=S.LAB_CONTINUE, replay=S.HINT_LAB_REPLAY_CHAMP)
        assert _visible_texts(frame) == [S.HINT_LAB_REPLAY_CHAMP, S.LAB_CONTINUE]
        cont = frame.buttons[3]
        assert cont.property(style.ROLE_PROPERTY) == "primary"  # 패널의 「이어서 학습」과 한 쌍
        assert cont.toolTip() == S.HELP_TIPS["lab.continue"] and frame.buttons[2].toolTip()
        assert cont.isEnabled() == lab.btn_continue.isEnabled()
        frame.buttons[2].click()
        assert played == [7]
        lab.view_mode = "replay"  # 다시 보기가 시작되면 숨는다
        frame.refresh()
        assert not frame.isVisible()
    finally:
        lab.run_dir, lab.generation, lab.view_mode = None, 0, "live"
        frame.refresh()
    assert frame.isVisible() and _visible_texts(frame)[0] == S.LAB_START  # 학습 전 카드로 돌아옴


def test_canvas_card_finished_state(window: Any, qtbot: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.help import hints as H

    lab = window.lab
    window.tabs.setCurrentWidget(lab)
    frame = hint_of(lab.canvas)
    assert frame is not None
    extended: list[int] = []
    monkeypatch.setattr(lab, "extend_and_continue", lambda n=10: extended.append(n) or True, raising=False)
    # 카드는 만들 때 lab.extend_and_continue를 잡아 둔다: 새 카드를 만들어 바꿔치기를 쓰게 한다
    frame.hide()
    frame = H.lab_hint(window)
    lab.client._proc = object()  # 학습 프로세스가 있는 것처럼(학습을 띄우지 않음)
    lab.state, lab._stop_reason, lab.run_dir, lab.generation = "finished", "max_generations", Path("run"), 30
    try:
        assert H.lab_card_state(lab) == "finished"
        frame.refresh()
        assert frame.isVisible()
        assert frame.title_label.text() == S.HINT_LAB_FINISHED_TITLE.format(
            reason=S.TRAIN_STOP_REASONS["max_generations"])
        more = S.HINT_LAB_MORE_GENS.format(n=H.MORE_GENERATIONS)
        assert _visible_texts(frame) == [more, S.HINT_LAB_CHANGE_STOP]
        assert frame.buttons[4].property(style.ROLE_PROPERTY) != "primary"  # 패널의 으뜸(이어서 학습)과 다른 행동
        frame.buttons[4].click()
        assert extended == [H.MORE_GENERATIONS]
        lab._stop_reason = "target"  # 목표 점수로 멈췄으면 「n세대 더」는 숨는다
        frame.refresh()
        assert _visible_texts(frame) == [S.HINT_LAB_CHANGE_STOP]
        frame.buttons[5].click()  # 패널을 정지 조건 묶음으로(예외 없이)
        lab.playing = {"generation": 30}  # 관전 재생 중이면 숨는다
        frame.refresh()
        assert not frame.isVisible()
    finally:
        lab.playing = None
        lab.client._proc = None
        lab.state, lab._stop_reason, lab.run_dir, lab.generation = "idle", "", None, 0
        frame.refresh()
        frame.hide()


# ---------------------------------------------------------------------------
# U9의 「이건 뭐예요?」: 꺼진 단추는 까닭까지(하면 좋음)
# ---------------------------------------------------------------------------
def test_whats_this_adds_the_reason_of_a_disabled_button(window: Any) -> None:
    from evolab.app.help import context

    btn = window.lab.btn_start
    tip = btn.toolTip()
    assert tip
    style.set_enabled(btn, True)
    assert context.whats_this_text(btn) == (btn, tip)
    style.set_enabled(btn, False, "why-off")
    try:
        target, text = context.whats_this_text(btn)
        assert target is btn and text == style.tip_text(btn) and text.startswith(tip) and text.endswith("why-off")
        assert btn.toolTip() == tip  # 풍선 글 자체는 그대로(V1)
    finally:
        style.set_enabled(btn, True)


# ---------------------------------------------------------------------------
# U34 도움말 창의 글 모양(하면 좋음)
# ---------------------------------------------------------------------------
def test_help_text_layout(qtbot: Any, isolated: Path) -> None:
    from PySide6.QtWidgets import QTextEdit

    from evolab.app.help import window as HW

    win = HW.HelpWindow()
    qtbot.addWidget(win)
    win.resize(1400, 700)
    win.show()
    QTest.qWaitForWindowExposed(win)
    try:
        assert win.show_page("preset")
        doc = win.browser.document()
        assert doc.documentMargin() == HW.TEXT_MARGIN
        block, headings, body = doc.begin(), [], []
        while block.isValid():
            fmt = block.blockFormat()
            (headings if fmt.headingLevel() > 0 else body).append(fmt)
            block = block.next()
        assert headings[0].topMargin() == 0 and all(h.bottomMargin() == 8 for h in headings)
        assert all(h.topMargin() >= 16 for h in headings[1:])
        assert all(b.lineHeight() == HW.LINE_HEIGHT_PCT for b in body)
        assert win.browser.palette().color(win.browser.palette().ColorRole.Link).name() == style.ui_for().link
        # 넓은 창: 글 폭은 720(+여백)까지, 좁은 창: 창 폭
        assert win.browser.lineWrapMode() == QTextEdit.LineWrapMode.FixedPixelWidth
        assert win.browser.lineWrapColumnOrWidth() == HW.TEXT_MAX_WIDTH + 2 * HW.TEXT_MARGIN
        win.resize(700, 600)
        QApplication.processEvents()
        assert win.browser.lineWrapMode() == QTextEdit.LineWrapMode.WidgetWidth
        # 제목으로 가는 링크·찾기는 그대로(글은 같다)
        assert win.show_page("editor", S.EDP_TAB_RULES) and win.current_heading_block() == S.EDP_TAB_RULES
    finally:
        win.close()

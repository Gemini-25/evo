"""v3.0 A 묶음 코치 오버레이 (v3 설계 4.2, D7, 5.3, 7.3, 8.1 T5).

진짜 MainWindow 위에서: 어두운 단계의 구멍 클릭은 진짜 창 이벤트(QWindow)로 대상에 닿고 어두운 곳 클릭은 막힌다, 밝은 단계는 말풍선
밖 클릭이 아래 위젯으로 간다, Esc = 건너뛰기, Enter = 다음, 스크롤·창 크기 바뀜 뒤 구멍이 대상을 따라간다, 그리기 ≤ 5ms.
짚기(point): 어둡게 하지 않음, 4초 또는 클릭으로 닫힘, 「도움말에서 더 보기」.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QPushButton, QScrollArea, QVBoxLayout, QWidget

from evolab import paths
from evolab.app import strings as S
from evolab.app.help import coach as C
from evolab.app.help.anchors import show_screen
from evolab.app.help.coach import CoachOverlay, CoachStep, point, scroll_into_view


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """실제 데이터·설정 폴더를 건드리지 않는다(설계 D5)."""
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
    return w


def click(window: QWidget, pos: QPoint) -> None:
    """진짜 창 이벤트 경로(QWindow → 위젯 찾기, 마스크 따름)로 누른다."""
    QTest.mouseClick(window.windowHandle(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)


def rect_in(window: QWidget, widget: QWidget) -> QRect:
    r = widget.visibleRegion().boundingRect()
    return QRect(widget.mapTo(window, r.topLeft()), r.size())


def expected_hole(window: QWidget, widget: QWidget) -> QRect:
    m = C.HOLE_MARGIN
    return rect_in(window, widget).adjusted(-m, -m, m, m).intersected(window.rect())


def tab_center(window: Any, index: int) -> QPoint:
    bar = window.tabs.tabBar()
    return bar.mapTo(window, bar.tabRect(index).center())


# ---------------------------------------------------------------------------
# T5 어두운 단계·밝은 단계·키
# ---------------------------------------------------------------------------
def test_dim_hole_click_reaches_target_and_dark_area_is_blocked(qtbot: Any, window: Any) -> None:
    lab = window.lab
    refresh = window.compare.refresh_button
    refreshed: list[bool] = []
    refresh.clicked.connect(lambda *_a: refreshed.append(True))
    clicked: list[str] = []
    steps = [
        CoachStep("one", lambda: "one", anchor="compare.refresh", kind="click", dim=True,
                  prepare=lambda: show_screen(window, "compare")),
        CoachStep("two", lambda: "two", anchor="lab.speed", kind="next",
                  prepare=lambda: (show_screen(window, "lab"), scroll_into_view(lab.speed_combo))),
    ]
    ov = CoachOverlay(window, steps)
    ov.target_clicked.connect(clicked.append)
    ov.start()
    qtbot.waitUntil(lambda: ov.hole == expected_hole(window, refresh) and not ov.hole.isNull(), timeout=3000)
    assert ov.geometry() == window.rect() and ov.isVisible()
    assert ov.counter.text() == S.ONB_PROGRESS.format(i=1, n=2)
    assert ov.buttons() == [S.ONB_SKIP]  # 누르기 단계: 「다음」 없음, 건너뛰기만
    # 어두운 곳(탭 줄 「기록」) 클릭은 막힌다
    before = window.tabs.currentIndex()
    click(window, tab_center(window, 3))
    qtbot.wait(50)
    assert window.tabs.currentIndex() == before and ov.current.key == "one"
    # 메뉴 막대도 막는다(메뉴가 열리지 않는다)
    click(window, QPoint(30, window.menuBar().height() // 2))
    qtbot.wait(50)
    assert ov.current.key == "one"
    # 구멍 가운데 클릭은 진짜 「새로 고침」 단추를 누르고 다음 단계로
    assert not refreshed
    click(window, ov.hole.center())
    qtbot.waitUntil(lambda: ov.current is not None and ov.current.key == "two", timeout=2000)
    assert refreshed == [True] and clicked == ["one"] and ov.clicks == 1
    assert window.tabs.currentWidget() is lab
    # 밝은 단계: 말풍선 밖 클릭은 아래 위젯으로(탭이 바뀐다)
    qtbot.waitUntil(lambda: not ov.hole.isNull(), timeout=2000)
    assert ov.buttons() == [S.ONB_NEXT, S.ONB_SKIP]
    assert not ov.mask().contains(tab_center(window, 3))
    # 탭 바뀜은 신호로 본다: 대상(배속 칸)이 가려지면 코치가 100ms 확인에서 준비(실험실 탭으로)를 다시 해 탭을 되돌리므로,
    # 지금 탭을 기다리면 그 되돌림과 경주한다(통합 단계에서 고침: 붐비는 기계에서 1/5쯤 실패하던 원인).
    seen_tabs: list[int] = []
    window.tabs.currentChanged.connect(seen_tabs.append)
    click(window, tab_center(window, 3))
    qtbot.waitUntil(lambda: window.tabs.indexOf(window.records) in seen_tabs, timeout=2000)
    window.tabs.currentChanged.disconnect(seen_tabs.append)
    window.tabs.setCurrentWidget(lab)
    # Enter = 「다음」(마지막 단계라 끝)
    finished: list[str] = []
    ov.finished.connect(finished.append)
    QTest.keyClick(window, Qt.Key.Key_Return)
    assert finished == ["done"] and ov.result == "done" and not ov.isVisible()
    assert [k for _t, k in ov.log] == ["one", "two", "finish:done"]


def test_escape_skips_and_enter_needs_enabled_next(qtbot: Any, window: Any) -> None:
    gate = {"open": False}
    steps = [CoachStep("a", lambda: "a", kind="next", until=lambda: gate["open"]),
             CoachStep("b", lambda: "b", kind="next")]
    ov = CoachOverlay(window, steps)
    ov.start()
    nxt = ov.button(S.ONB_NEXT)
    assert nxt is not None and not nxt.isEnabled()
    QTest.keyClick(window, Qt.Key.Key_Return)  # 「다음」이 꺼져 있으면 Enter는 그냥 지나간다
    assert ov.current.key == "a"
    gate["open"] = True
    qtbot.waitUntil(lambda: ov.button(S.ONB_NEXT) is not None and ov.button(S.ONB_NEXT).isEnabled(), timeout=2000)
    assert ov.current.key == "a"  # "next" 종류는 조건이 참이어도 스스로 넘어가지 않는다
    QTest.keyClick(window.lab.canvas, Qt.Key.Key_Return)  # 초점이 창 안 어디 있든
    assert ov.current.key == "b"
    QTest.keyClick(window.lab.canvas, Qt.Key.Key_Escape)
    assert ov.result == "skipped" and not ov.isVisible()


def test_wait_value_steps_advance_on_condition_and_buttons_follow_delay(qtbot: Any, window: Any,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr(C, "CLOCK", lambda: now[0])
    flag = {"v": False}
    extra_on = {"v": False}
    steps = [
        CoachStep("w", lambda: f"w{int(now[0])}", kind="wait", until=lambda: flag["v"]),
        CoachStep("v", lambda: "v", kind="value", until=lambda: flag["v"] is False, skip_step=True, next_after_s=2.0,
                  extra=lambda: (("x", lambda: extra_on.__setitem__("v", True)),) if now[0] >= 140 else ()),
    ]
    ov = CoachOverlay(window, steps)
    ov.start()
    assert ov.text_label.text() == "w100" and ov.buttons() == [S.ONB_SKIP]
    now[0] = 105.0
    qtbot.waitUntil(lambda: ov.text_label.text() == "w105", timeout=2000)  # 글은 100ms마다 다시 읽는다
    flag["v"] = True
    qtbot.waitUntil(lambda: ov.current.key == "v", timeout=2000)
    skip = ov.button(S.ONB_SKIP_STEP)
    assert skip is not None and not skip.isEnabled()
    now[0] = 107.5
    qtbot.waitUntil(lambda: ov.button(S.ONB_SKIP_STEP).isEnabled(), timeout=2000)
    assert ov.button("x") is None
    now[0] = 140.0
    qtbot.waitUntil(lambda: ov.button("x") is not None, timeout=2000)
    # 지운 단추는 레이아웃에서 빠져 숨는다(보이는 단추 = 지금 단추)
    visible = [b.text() for b in ov.bubble.findChildren(QPushButton) if b.isVisible()]
    assert visible == ov.buttons() == [S.ONB_SKIP_STEP, "x", S.ONB_SKIP]
    assert ov.press("x") and extra_on["v"] and ov.clicks == 1
    flag["v"] = False
    qtbot.waitUntil(lambda: ov.result == "done", timeout=2000)


def test_hole_follows_scroll_and_resize(qtbot: Any, window: Any) -> None:
    lab = window.lab
    combo = lab.speed_combo
    area = combo.parentWidget()
    while not isinstance(area, QScrollArea):
        area = area.parentWidget()
    bar = area.verticalScrollBar()
    scroll_into_view(combo, 200)
    qtbot.wait(20)
    ov = CoachOverlay(window, [CoachStep("s", lambda: "speed", anchor="lab.speed", kind="next", dim=True)])
    ov.start()
    qtbot.waitUntil(lambda: ov.hole == expected_hole(window, combo), timeout=2000)
    before = QRect(ov.hole)
    step = 40 if bar.value() + 40 <= bar.maximum() else -40
    bar.setValue(bar.value() + step)
    qtbot.waitUntil(lambda: ov.hole == expected_hole(window, combo) and ov.hole != before, timeout=2000)
    assert ov.hole.top() == before.top() - step
    assert not ov.mask().contains(ov.hole.center()) and ov.mask().contains(QPoint(2, window.height() - 2))
    window.resize(1100, 760)
    qtbot.waitUntil(lambda: ov.geometry() == window.rect() and ov.hole == expected_hole(window, combo), timeout=3000)
    # 말풍선은 창 안, 구멍과 겹치지 않는 자리
    assert window.rect().contains(ov.bubble.geometry())
    assert not ov.bubble.geometry().intersects(ov.hole)
    ov.finish("skipped")


def test_lost_target_centres_bubble_and_offers_way_back(qtbot: Any, window: Any) -> None:
    lab = window.lab
    ov = CoachOverlay(window, [CoachStep("s", lambda: "speed", anchor="lab.speed", kind="next",
                                         prepare=lambda: scroll_into_view(lab.speed_combo))])
    ov.start()
    qtbot.waitUntil(lambda: not ov.hole.isNull(), timeout=2000)
    window.tabs.setCurrentWidget(window.editor)  # 밝은 단계라 사용자가 다른 탭으로 갈 수 있다
    qtbot.waitUntil(lambda: S.ONB_BACK_TO_LAB in ov.buttons(), timeout=3000)
    assert ov.hole.isNull()
    centre = ov.bubble.geometry().center()
    assert abs(centre.x() - window.width() // 2) <= 2 and abs(centre.y() - window.height() // 2) <= 2
    assert ov.press(S.ONB_BACK_TO_LAB)
    assert window.tabs.currentWidget() is lab
    qtbot.waitUntil(lambda: not ov.hole.isNull() and S.ONB_BACK_TO_LAB not in ov.buttons(), timeout=2000)
    ov.finish("done")


def test_missing_anchor_widget_means_centred_card_without_back_button(qtbot: Any, window: Any) -> None:
    ov = CoachOverlay(window, [CoachStep("s", lambda: "x", anchor="play.restart", kind="welcome",
                                         buttons=(("ok", lambda: None),))])  # 직접 조종 창이 없다
    ov.start()
    qtbot.wait(450)
    assert ov.hole.isNull() and ov.buttons() == ["ok"]
    ov.finish("done")


def test_bubble_fits_wrapped_text_and_names_itself(qtbot: Any, window: Any) -> None:
    long_text = S.ONB_STEP_DONE
    ov = CoachOverlay(window, [CoachStep("s", lambda: long_text, title=S.ONB_WELCOME_TITLE, kind="welcome",
                                         buttons=(("ok", lambda: None),), status=lambda: S.ONB_PROGRESS_FLAT)])
    ov.start()
    assert ov.bubble.width() == C.BUBBLE_WIDTH
    for label in (ov.title_label, ov.text_label, ov.status_label):
        assert label.isVisible()
        assert label.height() >= label.heightForWidth(label.width()) - 1, label.objectName()  # 줄바꿈한 글이 잘리지 않는다
    assert ov.bubble.accessibleName() == "\n".join([S.ONB_WELCOME_TITLE, long_text, S.ONB_PROGRESS_FLAT])
    ov.finish("done")


def test_overlay_paint_is_cheap(qtbot: Any, window: Any) -> None:
    """그리기 ≤ 5ms(설계 T5, 원형 0.70ms). 1280×800, 어두운 단계 + 구멍, 말풍선 포함, 50번 중앙값."""
    lab = window.lab
    ov = CoachOverlay(window, [CoachStep("s", lambda: S.ONB_STEP_PRESS_START.format(start=S.LAB_START, pop=32),
                                         anchor="lab.start", kind="click", dim=True,
                                         prepare=lambda: scroll_into_view(lab.btn_start))])
    ov.start()
    qtbot.waitUntil(lambda: not ov.hole.isNull(), timeout=2000)
    img = QImage(window.size(), QImage.Format.Format_ARGB32_Premultiplied)
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        ov.render(img)
        times.append((time.perf_counter() - t0) * 1000.0)
    print(f"overlay render median {statistics.median(times):.3f} ms, max {max(times):.3f} ms")
    assert statistics.median(times) <= 5.0
    ov.finish("skipped")


# ---------------------------------------------------------------------------
# 짚기
# ---------------------------------------------------------------------------
def test_point_highlights_without_dimming_and_opens_more(qtbot: Any, window: Any,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    lab = window.lab
    scroll_into_view(lab.btn_start)
    opened: list[tuple[Any, ...]] = []
    monkeypatch.setattr(C, "open_help_page", lambda *a: opened.append(a) or True)
    ov = point(window, lab.btn_start, S.HELP_TIPS.get("lab.start", "x"), more=("lab", None))
    assert isinstance(ov, CoachOverlay) and ov.point_mode and ov.isVisible()
    qtbot.waitUntil(lambda: ov.hole == expected_hole(window, lab.btn_start), timeout=2000)
    assert ov.buttons() == [S.WHATS_THIS_MORE] and ov.counter.text() == ""
    # 어둡게 하지 않는다: 마스크는 말풍선과 주황 띠뿐
    assert not ov.current.dim and not ov.mask().contains(tab_center(window, 0))
    assert not ov.mask().contains(ov.hole.center())
    assert ov.press(S.WHATS_THIS_MORE)
    assert opened == [("lab", None, window)] and ov.result == "done"


def test_point_closes_on_any_click_or_after_timeout(qtbot: Any, window: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    lab = window.lab
    ov = point(window, lab.canvas, "canvas")
    click(window, tab_center(window, 2))  # 아무 곳이나 누르면 닫히고, 그 클릭은 아래 위젯에 간다
    qtbot.waitUntil(lambda: ov.result == "done", timeout=2000)
    assert window.tabs.currentWidget() is window.compare
    window.tabs.setCurrentWidget(lab)
    monkeypatch.setattr(C, "POINT_SECONDS", 0.2)
    ov2 = point(window, lab.canvas, "canvas")
    ov3 = point(window, lab.dashboard, "dashboard")  # 새 짚기는 앞 짚기를 닫는다
    assert ov2.result == "done" and ov3.running
    qtbot.waitUntil(lambda: ov3.result == "done", timeout=3000)
    ov4 = point(window, lab.canvas, "esc")
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert ov4.result == "done"


def test_point_on_widget_in_another_window(qtbot: Any, window: Any) -> None:
    other = QWidget()
    qtbot.addWidget(other)
    lay = QVBoxLayout(other)
    btn = QPushButton("b")
    lay.addWidget(btn)
    other.resize(400, 300)
    other.show()
    qtbot.waitExposed(other)
    ov = point(window, btn, "other")
    assert ov.parentWidget() is other
    qtbot.waitUntil(lambda: ov.hole == expected_hole(other, btn), timeout=2000)
    ov.finish("done")

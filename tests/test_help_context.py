"""v3.0 B 묶음 화면 도움말 문맥: F1·Shift+F1·모서리 단추 (v3 설계 4.5, D21, D22, 8.1 T13).

- F1: 마우스 아래 위젯에서 위로 올라가며 쪽을 정한다: 대시보드 타일 → (dashboard, 타일 제목), 에디터 아래 탭(「검사 결과 (3)」 상태 포함)
  → (editor, 탭 제목), 직접 조종 → play, 나란히 경주 → race, 반복 실행 → compare, 오류 보고 → trouble, 그 밖은 탭 쪽.
- Shift+F1: 다음 클릭은 위젯에 가지 않고 짚기 말풍선(풍선 글, 없으면 이름표 짝, 없으면 WHATS_THIS_NONE) + 그 부분 도움말 링크. Esc로 끝.
- 모서리 「? 도움말」 단추·「이 화면 도움말」 메뉴: 지금 탭 쪽.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QAction, QCursor, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QMessageBox, QPushButton, QWidget

from evolab import paths
from evolab.app import strings as S
from evolab.app.help import anchors, content, context
from evolab.app.help import tooltips as T
from evolab.app.help import window as HW

DATA = Path(__file__).resolve().parent / "data"
LANDER = paths.builtin_env_dir() / "lander.evoenv"


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv("EVOLAB_NO_TOUR", "1")
    monkeypatch.setattr(HW, "hooks", {})
    yield tmp_path
    win = HW.help_window()
    if win is not None:
        win.close()
        win.deleteLater()
    HW._instance = None
    app = QApplication.instance()
    if isinstance(app, QApplication):  # 앱 전체 필터를 다음 테스트(다른 파일)에 남기지 않는다
        context.uninstall_shortcuts(app)
        T.uninstall(app)


@pytest.fixture()
def points(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, str, Any]]:
    calls: list[tuple[Any, str, Any]] = []
    monkeypatch.setattr(HW, "point_at", lambda widget, text, more=None: calls.append((widget, text, more)))
    return calls


@pytest.fixture()
def main_window(qtbot: Any, isolated: Path) -> Any:
    from evolab.app.main import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1280, 800)
    window.show()
    window.activateWindow()
    QTest.qWaitForWindowExposed(window)
    return window


@pytest.fixture()
def shortcuts(main_window: Any) -> context.HelpShortcuts:
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    context.install_shortcuts(app, main_window)
    flt = context.shortcuts()
    assert flt is not None
    return flt


def _hover(widget: QWidget, at: QPoint | None = None) -> None:
    """마우스를 위젯 위(기본 가운데)에 둔다."""
    QCursor.setPos(widget.mapToGlobal(at if at is not None else widget.rect().center()))
    QApplication.processEvents()


def _press_f1(target: QWidget, shift: bool = False) -> None:
    mod = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    QTest.keyClick(target, Qt.Key.Key_F1, mod)


# ---------------------------------------------------------------------------
# context_page
# ---------------------------------------------------------------------------
def test_every_dashboard_tile_opens_its_section(main_window: Any) -> None:
    from evolab.app.lab.dashboard import ALL_TILE_IDS

    dash = main_window.lab.dashboard
    heads = content.headings("dashboard")
    for tid in ALL_TILE_IDS:
        tile = dash.tile(tid)  # 지금 쪽이 아닌 타일은 부모가 없다(배치에서 빠짐)
        expected = ("dashboard", S.DASH_TILE_TITLES[tid])
        assert context.context_page(tile) == expected, tid
        assert context.context_page(tile.title) == expected, tid  # 타일 안 위젯에서 올라가도
        assert expected[1] in heads
    assert context.context_page(dash.axis_box) == ("dashboard", None)  # 대시보드 안, 타일 밖


def test_editor_bottom_tabs_open_their_sections(main_window: Any) -> None:
    tabs = main_window.editor.tabs
    for attr, name in context.EDITOR_TABS:
        page = getattr(tabs, attr)
        assert context.context_page(page) == ("editor", getattr(S, name)), attr
        child = next((c for c in page.findChildren(QWidget)), None)
        if child is not None:
            assert context.context_page(child) == ("editor", getattr(S, name)), attr
    # 「검사 결과 (3)」처럼 개수가 붙은 탭 글이어도 탭 위젯으로 정한다
    issues = tabs.indexOf(tabs.issues_tab)
    tabs.setTabText(issues, S.EDP_TAB_ISSUES_COUNT.format(n=3))
    assert context.context_page(tabs.issues_tab) == ("editor", S.EDP_TAB_ISSUES)
    assert set(content.headings("editor")) >= {getattr(S, n) for _a, n in context.EDITOR_TABS}
    # 아래 탭 밖(캔버스)은 에디터 쪽
    assert context.context_page(main_window.editor.canvas) == ("editor", None)


def test_tab_bars_use_the_tab_under_the_mouse(main_window: Any) -> None:
    w = main_window
    w.tabs.setCurrentWidget(w.editor)
    bar = w.editor.tabs.tabBar()
    structure = w.editor.tabs.indexOf(w.editor.tabs.structure_tab)
    _hover(bar, bar.tabRect(structure).center())
    assert context.context_page(bar) == ("editor", S.ED_TAB_STRUCTURE)
    main_bar = w.tabs.tabBar()
    _hover(main_bar, main_bar.tabRect(w.tabs.indexOf(w.records)).center())
    assert context.context_page(main_bar) == ("records", None)
    _hover(w.lab)  # 탭 줄 밖이면 지금 탭
    assert context.context_page(main_bar) == ("editor", None)


def test_screens_and_windows(qtbot: Any, main_window: Any, isolated: Path) -> None:
    from evolab.app.compare.repeat import RepeatDialog
    from evolab.app.play.window import PlayWindow
    from evolab.app.race import model as M
    from evolab.app.race.view import show_race, shutdown_race
    from evolab.spec.io import load_env

    w = main_window
    assert context.context_page(w.lab.btn_start) == ("lab", None)
    assert context.context_page(w.compare.refresh_button) == ("compare", None)
    assert context.context_page(w.records.memo_button) == ("records", None)
    assert context.context_page(w.menuBar()) == ("start", None)
    assert context.context_page(None) == ("start", None)
    box = QMessageBox(w.lab)  # 탭 안에서 띄운 상자는 그 탭
    qtbot.addWidget(box)
    assert context.context_page(box) == ("lab", None)

    play = PlayWindow(load_env(LANDER))
    qtbot.addWidget(play)
    assert context.context_page(play) == ("play", None)
    assert context.context_page(play.save_button) == ("play", None)

    repeat = RepeatDialog(runs_dir=isolated / "runs", workers=1)
    qtbot.addWidget(repeat)
    assert context.context_page(repeat.buttons) == ("compare", None)

    runs = []
    for name in ("v10_run_ga", "v11_run_restart"):
        dst = isolated / name
        shutil.copytree(DATA / name / "run", dst)
        runs.append(dst)
    entries, _warn, _n = M.entries_from_runs([(runs[0], "#3987e5"), (runs[1], "#d95926")], "last")
    race = show_race(w, entries, title=S.RACE_TITLE)
    try:
        assert context.context_page(race) == ("race", None)
        assert context.context_page(race.view) == ("race", None)
    finally:
        shutdown_race(race)
        race.close()

    class ReportDialog(QDialog):  # D 묶음 대화 상자와 같은 이름(연결표는 클래스 이름으로 알아본다)
        pass

    class ExportDialog(QDialog):  # v3.1
        pass

    report = ReportDialog(w)
    qtbot.addWidget(report)
    inner = QPushButton(report)
    assert context.context_page(inner) == ("trouble", None)
    export = ExportDialog(w)
    qtbot.addWidget(export)
    assert context.context_page(export) == ("records", S.HELP_HEADING_VIDEO)
    assert S.HELP_HEADING_VIDEO in content.headings("records")


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------
def test_f1_opens_help_for_the_widget_under_the_mouse(main_window: Any, shortcuts: context.HelpShortcuts) -> None:
    w = main_window
    tile = w.lab.dashboard.tile("trend")
    w.tabs.setCurrentWidget(w.lab)
    assert tile.isVisible()
    _hover(tile)
    _press_f1(w.lab.btn_start)  # 초점은 다른 곳이어도 마우스 아래가 이긴다
    assert shortcuts.opened[-1] == ("dashboard", S.DASH_TILE_TITLES["trend"])
    win = HW.help_window()
    assert win is not None and win.isVisible()
    assert win.page == "dashboard" and win.current_heading_block() == S.DASH_TILE_TITLES["trend"]
    # 에디터 아래 탭
    w.tabs.setCurrentWidget(w.editor)
    w.editor.tabs.setCurrentWidget(w.editor.tabs.scoring_tab)
    _hover(w.editor.tabs.scoring_tab)
    _press_f1(w)
    assert shortcuts.opened[-1] == ("editor", S.EDP_TAB_SCORING)
    assert win.page == "editor" and win.heading == S.EDP_TAB_SCORING


def test_f1_falls_back_to_focus_and_beats_shortcuts(main_window: Any, shortcuts: context.HelpShortcuts) -> None:
    w = main_window
    w.tabs.setCurrentWidget(w.compare)
    _hover(w)
    QCursor.setPos(QPoint(-10000, -10000))  # 마우스가 창 밖: 초점 위젯
    QApplication.processEvents()
    w.compare.refresh_button.setFocus()
    triggered: list[bool] = []
    act = QAction(w)
    act.setShortcut(QKeySequence(Qt.Key.Key_F1))
    act.triggered.connect(lambda: triggered.append(True))
    w.addAction(act)
    _press_f1(w.compare.refresh_button)
    assert shortcuts.opened[-1] == ("compare", None)
    assert triggered == []  # F1 지름키가 있어도 문맥 도움말이 이김
    # 진짜 키 입력 길(창 핸들 → 지름키 확인 → 창 → 위젯)에서도 같다
    w.activateWindow()
    w.tabs.setCurrentWidget(w.records)
    w.records.refresh_button.setFocus()
    n = len(shortcuts.opened)
    QTest.keyClick(w.windowHandle(), Qt.Key.Key_F1)
    assert shortcuts.opened[n:] == [("records", None)] and triggered == []
    # 도움말 창 안에서 F1은 아무것도 하지 않음
    win = HW.help_window()
    assert win is not None
    n = len(shortcuts.opened)
    assert shortcuts.help_for(win.browser) is None and len(shortcuts.opened) == n


# ---------------------------------------------------------------------------
# Shift+F1 「이건 뭐예요?」
# ---------------------------------------------------------------------------
def test_shift_f1_next_click_explains_instead_of_acting(main_window: Any, shortcuts: context.HelpShortcuts,
                                                        points: list[Any]) -> None:
    w = main_window
    T.install(QApplication.instance(), w)
    w.tabs.setCurrentWidget(w.lab)
    box = w.lab.ipop
    _ensure_visible(box)
    before = box.isChecked()
    _press_f1(w, shift=True)
    assert shortcuts.whats_this_active
    assert QApplication.overrideCursor() is not None
    assert w.statusBar().currentMessage() == S.WHATS_THIS_HINT
    QTest.mouseClick(box, Qt.MouseButton.LeftButton)
    assert box.isChecked() == before  # 클릭이 위젯에 가지 않음
    assert not shortcuts.whats_this_active and QApplication.overrideCursor() is None
    widget, text, more = points[-1]
    assert widget is box and text == box.toolTip() == S.HELP_TIPS["lab.ipop"]
    assert more == ("lab", None)
    QTest.mouseClick(box, Qt.MouseButton.LeftButton)  # 그다음 클릭은 보통대로
    assert box.isChecked() != before


def test_whats_this_texts(main_window: Any, shortcuts: context.HelpShortcuts, points: list[Any]) -> None:
    w = main_window
    T.install(QApplication.instance(), w)
    # 숫자 칸 안의 글 칸을 눌러도 숫자 칸의 풍선
    seed_edit = w.lab.seed.lineEdit()
    assert context.whats_this_text(seed_edit) == (w.lab.seed, S.HELP_TIPS["lab.seed"])
    # 폼 이름표를 누르면 그 칸의 풍선(이름표에도 같은 글이 붙어 있음)
    label = T.form_label(w.lab.popsize)
    assert isinstance(label, QLabel)
    assert context.whats_this_text(label)[1] == S.HELP_TIPS["lab.popsize"]
    label.setToolTip("")
    assert context.whats_this_text(label)[1] == S.HELP_TIPS["lab.popsize"]
    # 설명이 없는 곳: WHATS_THIS_NONE + 그 화면 쪽 링크
    plain = QLabel("x", w.records)
    plain.show()
    context.whats_this(plain)
    widget, text, more = points[-1]
    assert widget is plain and text == S.WHATS_THIS_NONE.format(page=content.title("records"))
    assert more == ("records", None)


def test_escape_cancels_whats_this(main_window: Any, shortcuts: context.HelpShortcuts, points: list[Any]) -> None:
    w = main_window
    w.tabs.setCurrentWidget(w.lab)
    box = w.lab.ipop
    _ensure_visible(box)
    before = box.isChecked()
    context.start_whats_this(w)  # 메뉴 「이건 뭐예요?」
    assert shortcuts.whats_this_active
    QTest.keyClick(w, Qt.Key.Key_Escape)
    assert not shortcuts.whats_this_active and QApplication.overrideCursor() is None
    assert w.statusBar().currentMessage() != S.WHATS_THIS_HINT
    QTest.mouseClick(box, Qt.MouseButton.LeftButton)
    assert box.isChecked() != before and points == []


def test_start_whats_this_installs_filter_when_missing(main_window: Any, isolated: Path) -> None:
    app = QApplication.instance()
    context.uninstall_shortcuts(app)  # 통합 단계가 주 창을 만들 때 이미 달았으면 떼고 시작
    context.start_whats_this(main_window)
    flt = context.shortcuts()
    assert flt is not None and flt.whats_this_active
    flt.stop_whats_this()
    context.install_shortcuts(QApplication.instance(), main_window)  # 다시 달면 하나만
    assert context.shortcuts() is not flt and not flt.whats_this_active


# ---------------------------------------------------------------------------
# 모서리 단추·「이 화면 도움말」
# ---------------------------------------------------------------------------
def test_screen_help_opens_the_current_tab_page(main_window: Any) -> None:
    w = main_window
    for screen in anchors.TAB_SCREENS:
        w.tabs.setCurrentWidget(getattr(w, screen))
        assert context.screen_help_page(w) == (context.SCREEN_PAGES[screen], None)
        win = context.open_screen_help(w)
        assert isinstance(win, HW.HelpWindow) and win.page == context.SCREEN_PAGES[screen]


def test_corner_button_opens_the_current_tab_page(main_window: Any) -> None:
    w = main_window
    corner = anchors.resolve(w, "main.help_corner")
    assert isinstance(corner, QPushButton) or hasattr(corner, "click")
    w.tabs.setCurrentWidget(w.records)
    corner.click()  # type: ignore[union-attr]
    win = HW.help_window()
    assert win is not None and win.page == "records"


def _ensure_visible(widget: QWidget) -> None:
    HW._ensure_visible(widget)
    QApplication.processEvents()
    assert not widget.visibleRegion().isEmpty()


def test_whats_this_shows_a_real_point_bubble(main_window: Any, shortcuts: context.HelpShortcuts) -> None:
    from evolab.app.help import coach

    w = main_window
    T.install(QApplication.instance(), w)
    w.tabs.setCurrentWidget(w.lab)
    box = w.lab.ipop
    _ensure_visible(box)
    before = box.isChecked()
    _press_f1(w, shift=True)
    QTest.mouseClick(box, Qt.MouseButton.LeftButton)
    assert box.isChecked() == before
    bubbles = [o for o in w.findChildren(coach.CoachOverlay) if getattr(o, "point_mode", False) and o.running]
    assert len(bubbles) == 1
    assert S.WHATS_THIS_MORE in bubbles[0].buttons()  # 「도움말에서 더 보기」
    assert bubbles[0].press(S.WHATS_THIS_MORE)
    win = HW.help_window()
    assert win is not None and win.page == "lab"


# ---------------------------------------------------------------------------
# 검토 수정: 타일 자체 위의 F1·Shift+F1, 모달 상자 위의 F1
# ---------------------------------------------------------------------------
def _own_pixel(widget: QWidget, step: int = 2) -> QPoint | None:
    """widgetAt이 그 위젯 자신인 점(자식 위젯이 덮지 않은 여백). 없으면 None."""
    for y in range(0, widget.height(), step):
        for x in range(0, widget.width(), step):
            if QApplication.widgetAt(widget.mapToGlobal(QPoint(x, y))) is widget:
                return QPoint(x, y)
    return None


def test_f1_and_whats_this_on_every_tile_itself(main_window: Any, shortcuts: context.HelpShortcuts) -> None:
    """타일 자신(제목 줄 여백 등) 위의 F1·Shift+F1. 「궤적 밀도」 타일은 속성 window(문자열)가 QWidget.window를 가린다:
    도움말 모듈은 QWidget.window(w)로 불러야 한다(예전에는 TypeError로 오류 상자가 떴다)."""
    from evolab.app.help import coach
    from evolab.app.lab.dashboard import ALL_TILE_IDS, PAGE_OF

    w = main_window
    w.tabs.setCurrentWidget(w.lab)
    dash = w.lab.dashboard
    assert isinstance(dash.tile("density").window, str)  # 가린 속성이 그대로 있어야 이 테스트가 뜻이 있다
    checked: list[str] = []
    for tid in ALL_TILE_IDS:
        dash.set_page(PAGE_OF[tid])
        QApplication.processEvents()
        tile = dash.tile(tid)
        assert tile.isVisible(), tid
        at = _own_pixel(tile)
        if at is None:  # 자식 위젯이 다 덮은 타일(마우스가 타일 자신에 닿을 수 없음)
            continue
        checked.append(tid)
        _hover(tile, at)
        assert QApplication.widgetAt(QCursor.pos()) is tile, tid
        _press_f1(w.lab.btn_start)
        assert shortcuts.opened[-1] == ("dashboard", S.DASH_TILE_TITLES[tid]), tid
        _press_f1(w.lab.btn_start, shift=True)
        assert shortcuts.whats_this_active
        QTest.mouseClick(tile, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)
        assert not shortcuts.whats_this_active and shortcuts.pointed[-1] is tile, tid
        bubbles = [o for o in w.findChildren(coach.CoachOverlay) if o.point_mode and o.running]
        assert len(bubbles) == 1 and bubbles[0].fixed_target is tile, tid
        assert S.WHATS_THIS_MORE in bubbles[0].buttons(), tid
        bubbles[0].finish("done")
    assert "density" in checked and len(checked) >= len(ALL_TILE_IDS) // 2, checked


def test_help_window_survives_the_modal_box_it_was_moved_to(qtbot: Any, main_window: Any,
                                                            shortcuts: context.HelpShortcuts) -> None:
    """F1을 오류 상자 위에서 누르면 도움말 창은 그 상자(모달)를 주인으로 옮겨 단다(막히지 않게). 상자가 닫히고 지워져도 도움말
    창은 주 창으로 돌아와 살아 있어야 한다(예전에는 상자와 함께 지워졌다)."""
    from PySide6.QtCore import QCoreApplication, QEvent

    from evolab.app import errors

    w = main_window
    win = HW.open_help("glossary", None, w)
    assert win.parentWidget() is w
    errors.show_error(w.lab, "제목", "가짜 오류")
    box = errors.open_error_boxes()[-1]
    qtbot.waitUntil(lambda: QApplication.activeModalWidget() is box, timeout=2000)
    _press_f1(box)
    assert HW.help_window() is win and win.parentWidget() is box and win.isVisible()
    box.close()
    QApplication.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()
    assert HW.help_window() is win
    assert win.parentWidget() is w and win.isVisible() and win.isWindow()
    # 다시 열면 같은 창을 쓴다
    assert HW.open_help("compare", None, w) is win and win.parentWidget() is w and win.page == "compare"

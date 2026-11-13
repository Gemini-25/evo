"""화면 도움말 열기: F1(마우스 아래 부분의 도움말)과 Shift+F1 「이건 뭐예요?」 (v3 설계 4.5, D21, D22).

- F1: 앱 이벤트 필터가 `KeyPress F1`을 받는다(`ShortcutOverride`도 받아 다른 지름키가 가로채지 못하게 함). 마우스 아래 위젯
  (`QApplication.widgetAt(QCursor.pos())`, 활성 창 안일 때) 또는 초점 위젯에서 위로 올라가며 쪽과 제목을 정한다(`context_page`):
  대시보드 타일 → (dashboard, 그 타일 제목), 에디터 아래 탭 → (editor, 그 탭 제목: 탭 위젯으로 정해 「검사 결과 (3)」 같은 개수
  표시와 무관), 그 밖은 `anchors.screen_of`의 화면 → `SCREEN_PAGES`.
- Shift+F1(또는 도움말 메뉴 「이건 뭐예요?」 → `start_whats_this`): 다음 클릭은 위젯에 가지 않고, 그 위젯의 풍선 글(없으면 이름표
  짝의 풍선)로 짚기 말풍선 + 「도움말에서 더 보기」. Esc로 끝.

필터는 이벤트 종류 하나만 비교하고 넘긴다(측정: 이벤트 하나에 약 0.8µs, 관전 4배속 초당 약 700개).
"""

from __future__ import annotations

from typing import Final

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QCursor, QKeyEvent, QWindow
from PySide6.QtWidgets import QApplication, QFormLayout, QLabel, QMainWindow, QTabBar, QTabWidget, QWidget

from evolab.app import strings as S
from evolab.app.help import anchors, content

SCREEN_PAGES: Final[dict[str, str]] = {
    "editor": "editor", "lab": "lab", "compare": "compare", "records": "records", "play": "play", "race": "race",
    "repeat": "compare", "report": "trouble", "export": "records", "main": "start", "preset": "preset",
}
"""화면(anchors.SCREENS) → 도움말 쪽(설계 4.5)."""

SCREEN_HEADINGS: Final[dict[str, str]] = {"export": "HELP_HEADING_VIDEO"}
"""화면 → 그 쪽 안의 제목(strings 이름). 내보내기 대화 상자(v3.1)는 「기록과 내보내기」 쪽 「영상」 절(설계 D21)."""

EDITOR_TABS: Final[tuple[tuple[str, str], ...]] = (
    ("rules_tab", "EDP_TAB_RULES"), ("world_tab", "EDP_TAB_WORLD"), ("events_tab", "EDP_TAB_EVENTS"), ("scoring_tab", "EDP_TAB_SCORING"),
    ("random_tab", "EDP_TAB_RANDOM"), ("generators_tab", "EDP_TAB_GENERATORS"), ("eval_tab", "EDP_TAB_EVAL"),
    ("brain_tab", "EDP_TAB_BRAIN"), ("structure_tab", "ED_TAB_STRUCTURE"), ("params_tab", "EDP_TAB_PARAMS"),
    ("issues_tab", "EDP_TAB_ISSUES"),
)
"""에디터 아래 탭(BottomTabs 속성) → 탭 제목 상수(= 「에디터」 쪽 절 제목). 탭 글은 개수가 붙을 수 있어 쓰지 않는다."""

EDITOR_TABS_CLASS: Final = "BottomTabs"
TILE_CLASS: Final = "_Tile"
DASHBOARD_CLASS: Final = "Dashboard"

_F1 = Qt.Key.Key_F1
_KEY_EVENTS = (QEvent.Type.KeyPress, QEvent.Type.ShortcutOverride)
_MOUSE_EVENTS = (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease, QEvent.Type.MouseButtonDblClick)

_filter: HelpShortcuts | None = None


def _has_class(widget: object, name: str) -> bool:
    return any(cls.__name__ == name for cls in type(widget).__mro__)


def _parent(widget: QWidget) -> QWidget | None:
    """부모 위젯. 지금 쪽이 아닌 대시보드 타일은 배치에서 빠져 부모가 없으므로 자기 대시보드를 부모로 본다."""
    parent = widget.parentWidget()
    if parent is None and hasattr(widget, "tile_id"):
        dash = getattr(widget, "dash", None)
        if isinstance(dash, QWidget):
            return dash
    return parent


def _tab_under_cursor(bar: QTabBar) -> int:
    """탭 줄에서 마우스 아래 탭 번호(마우스가 탭 줄 밖이면 지금 탭)."""
    index = bar.tabAt(bar.mapFromGlobal(QCursor.pos()))
    return index if index >= 0 else bar.currentIndex()


def editor_tab_heading(tabs: QTabWidget, page: QWidget | None) -> str | None:
    """에디터 아래 탭 페이지 → 「에디터」 쪽 절 제목. 표에 없는 새 탭은 그 탭 글이 쪽 제목이면 그 글."""
    if page is None:
        return None
    for attr, name in EDITOR_TABS:
        if getattr(tabs, attr, None) is page:
            value = getattr(S, name, None)
            return value if isinstance(value, str) else None
    text = tabs.tabText(tabs.indexOf(page))
    return text if text in content.headings("editor") else None


def _main_tab_page(window: QMainWindow, index: int) -> str | None:
    tabs = getattr(window, "tabs", None)
    if not isinstance(tabs, QTabWidget):
        return None
    page = tabs.widget(index)
    for screen in anchors.TAB_SCREENS:
        if getattr(window, screen, None) is page:
            return SCREEN_PAGES[screen]
    return None


def context_page(widget: QWidget | None) -> tuple[str, str | None]:
    """F1이 열 도움말 (쪽, 제목). 위젯에서 위로 올라가며 대시보드 타일, 에디터 아래 탭, 대시보드를 찾고, 없으면 화면 쪽."""
    if widget is None:
        return content.HOME, None
    passed: list[QWidget] = []
    w: QWidget | None = widget
    try:
        while w is not None:
            tile_id = getattr(w, "tile_id", None)
            if isinstance(tile_id, str) and tile_id in S.DASH_TILE_TITLES and _has_class(w, TILE_CLASS):
                return "dashboard", S.DASH_TILE_TITLES[tile_id]
            if isinstance(w, QTabWidget) and _has_class(w, EDITOR_TABS_CLASS):
                page = next((p for p in passed if w.indexOf(p) >= 0), None)
                if page is None:
                    bar = w.tabBar()
                    page = w.widget(_tab_under_cursor(bar)) if any(p is bar for p in passed) else w.currentWidget()
                return "editor", editor_tab_heading(w, page)
            if _has_class(w, DASHBOARD_CLASS):
                return "dashboard", None
            if isinstance(w, QTabBar) and isinstance(w.parentWidget(), QTabWidget):
                owner = w.parentWidget().parentWidget()
                if anchors._is_main_window(owner):  # noqa: SLF001 - 주 창 탭 줄: 마우스 아래 탭의 화면
                    page_id = _main_tab_page(owner, _tab_under_cursor(w))  # type: ignore[arg-type]
                    if page_id is not None:
                        return page_id, None
            passed.append(w)
            w = _parent(w)
    except RuntimeError:  # 이미 지워진 Qt 객체
        return content.HOME, None
    screen = anchors.screen_of(widget)
    if screen is None:
        return content.HOME, None
    heading_name = SCREEN_HEADINGS.get(screen)
    heading = getattr(S, heading_name, None) if heading_name else None
    return SCREEN_PAGES.get(screen, content.HOME), heading if isinstance(heading, str) else None


def screen_help_page(window: QMainWindow) -> tuple[str, str | None]:
    """주 창의 지금 화면(탭) 도움말 쪽: 모서리 「? 도움말」 단추와 메뉴 「이 화면 도움말」."""
    tabs = getattr(window, "tabs", None)
    current = tabs.currentWidget() if isinstance(tabs, QTabWidget) else None
    if current is None:
        return content.HOME, None
    screen = anchors.screen_of(current)
    return SCREEN_PAGES.get(screen or "main", content.HOME), None


def open_screen_help(window: QMainWindow) -> QWidget:
    """주 창의 지금 탭 쪽을 연다(모서리 단추·「이 화면 도움말」 메뉴가 부름)."""
    from evolab.app.help.window import open_help

    page, heading = screen_help_page(window)
    return open_help(page, heading, window)


# ---------------------------------------------------------------------------
# 「이건 뭐예요?」 글
# ---------------------------------------------------------------------------
def _form_pair(widget: QWidget) -> QWidget | None:
    """폼 줄에서 짝: 이름표면 그 칸, 칸이면 그 이름표."""
    parent = widget.parentWidget()
    layout = QWidget.layout(parent) if parent is not None else None  # 속성 layout을 가린 위젯도 있다
    if not isinstance(layout, QFormLayout):
        return None
    if isinstance(widget, QLabel) and widget.buddy() is not None:
        return widget.buddy()
    label = layout.labelForField(widget)
    if label is not None:
        return label
    for row in range(layout.rowCount()):
        lab = layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
        fld = layout.itemAt(row, QFormLayout.ItemRole.FieldRole)
        if lab is not None and lab.widget() is widget and fld is not None:
            return fld.widget()
    return None


WHATS_THIS_DEPTH: Final = 3
"""누른 위젯에 풍선이 없으면 몇 층 위까지 올라가며 찾을지(숫자 칸 안의 글 칸, 콤보 안의 목록 등 부품 속 위젯)."""


def whats_this_text(widget: QWidget) -> tuple[QWidget, str]:
    """누른 위젯의 설명 (짚을 위젯, 글): 자기 풍선, 폼 짝의 풍선, 위 위젯의 풍선 순. 없으면 빈 글."""
    w: QWidget | None = widget
    for _ in range(WHATS_THIS_DEPTH + 1):
        if w is None or w.isWindow():
            break
        if w.toolTip():
            return w, w.toolTip()
        pair = _form_pair(w)
        if pair is not None and pair.toolTip():
            return w, pair.toolTip()
        w = w.parentWidget()
    return widget, ""


def whats_this(widget: QWidget) -> object | None:
    """widget을 「이건 뭐예요?」로 짚는다: 설명 글(없으면 WHATS_THIS_NONE) + 그 부분 도움말 링크."""
    from evolab.app.help.window import point_at

    target, text = whats_this_text(widget)
    page, heading = context_page(widget)
    if not text:
        text = S.WHATS_THIS_NONE.format(page=content.title(page))
    return point_at(target, text, (page, heading))


# ---------------------------------------------------------------------------
# 앱 이벤트 필터
# ---------------------------------------------------------------------------
def _receiver_window(receiver: QObject | None) -> QWidget | None:
    """이벤트를 받은 객체의 창 위젯: 위젯이면 그 창, 창 핸들(QWindow)이면 그 핸들을 가진 창 위젯."""
    if isinstance(receiver, QWidget):
        return QWidget.window(receiver)  # 속성 window를 가린 위젯도 있다(궤적 밀도 타일)
    if isinstance(receiver, QWindow):
        for top in QApplication.topLevelWidgets():
            if top.windowHandle() is receiver:
                return top
    return None


class HelpShortcuts(QObject):
    """F1·Shift+F1과 「이건 뭐예요?」 클릭을 받는 앱 이벤트 필터(install_shortcuts가 하나만 단다)."""

    def __init__(self, window: QMainWindow | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.window = window
        self.whats_this_active = False
        self.opened: list[tuple[str, str | None]] = []
        """F1로 연 (쪽, 제목) 차례(테스트·자체 점검용)."""
        self.pointed: list[QWidget] = []
        """「이건 뭐예요?」로 짚은 위젯 차례(테스트용)."""
        self._eat_release = False

    # -- F1 -------------------------------------------------------------------
    def target_widget(self, receiver: QObject | None = None) -> QWidget | None:
        """F1의 대상: 마우스 아래 위젯(키를 받은 창 안일 때), 아니면 그 창의 초점 위젯, 아니면 그 창.

        키를 받은 창 = 이벤트를 받은 위젯의 창 또는 창 핸들(QWindow)의 위젯. 키는 활성 창으로 오므로 보통 활성 창과 같다.
        """
        window = _receiver_window(receiver) or QApplication.activeWindow()
        under = QApplication.widgetAt(QCursor.pos())
        if under is not None and (window is None or QWidget.window(under) is window):
            return under
        focus = QApplication.focusWidget()
        if focus is not None and (window is None or QWidget.window(focus) is window):
            return focus
        return receiver if isinstance(receiver, QWidget) else window

    def help_for(self, widget: QWidget | None) -> QWidget | None:
        """F1: widget 자리의 도움말을 연다. 도움말 창 안에서 누르면 아무것도 하지 않는다."""
        from evolab.app.help.window import is_help_window, open_help

        if is_help_window(widget):
            return None
        page, heading = context_page(widget)
        self.opened.append((page, heading))
        return open_help(page, heading, widget if widget is not None else self._main())

    def _main(self) -> QMainWindow | None:
        from evolab.app.help.window import _alive

        return self.window if self.window is not None and _alive(self.window) else None

    # -- 「이건 뭐예요?」 -----------------------------------------------------
    def start_whats_this(self) -> None:
        if self.whats_this_active:
            return
        self.whats_this_active = True
        # 나중에 단 앱 필터가 먼저 받는다: 처음 실행 안내의 코치 필터(Esc = 안내 건너뛰기)보다 먼저 Esc·클릭을 받도록 다시 단다
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
            app.installEventFilter(self)
        QApplication.setOverrideCursor(Qt.CursorShape.WhatsThisCursor)
        main = self._main()
        if main is not None:
            main.statusBar().showMessage(S.WHATS_THIS_HINT)

    def stop_whats_this(self) -> None:
        if not self.whats_this_active:
            return
        self.whats_this_active = False
        QApplication.restoreOverrideCursor()
        main = self._main()
        if main is not None and main.statusBar().currentMessage() == S.WHATS_THIS_HINT:
            main.statusBar().clearMessage()

    # -- 필터 -----------------------------------------------------------------
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        kind = event.type()
        if kind in _KEY_EVENTS:
            return self._key(obj, event, kind)
        if self.whats_this_active and kind in _MOUSE_EVENTS and obj.isWidgetType():
            return self._mouse(obj, kind)
        if self._eat_release and kind == QEvent.Type.MouseButtonRelease and obj.isWidgetType():
            self._eat_release = False
            return True
        return False

    def _key(self, obj: QObject, event: QEvent, kind: QEvent.Type) -> bool:
        if not isinstance(event, QKeyEvent):
            return False
        key = event.key()
        if key == Qt.Key.Key_Escape and self.whats_this_active:
            if kind == QEvent.Type.KeyPress:
                self.stop_whats_this()
            event.accept()
            return True
        if key != _F1:
            return False
        if kind == QEvent.Type.ShortcutOverride:
            event.accept()  # 지름키(메뉴 항목 등)가 F1을 가져가지 않게: 아래 KeyPress에서 문맥으로 연다
            return True
        if event.isAutoRepeat():
            return True
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.start_whats_this()
        else:
            self.help_for(self.target_widget(obj))
        return True

    def _mouse(self, obj: QObject, kind: QEvent.Type) -> bool:
        if kind == QEvent.Type.MouseButtonPress:
            widget = obj if isinstance(obj, QWidget) else None
            self.stop_whats_this()
            self._eat_release = True
            if widget is not None:
                self.pointed.append(widget)
                whats_this(widget)
        return True


def shortcuts() -> HelpShortcuts | None:
    """지금 달린 필터(없으면 None)."""
    return _filter


def install_shortcuts(app: QApplication, window: QMainWindow) -> None:
    """앱에 F1·Shift+F1 필터를 단다. 다시 부르면 앞의 필터를 떼고 새 주 창으로 바꾼다(하나만)."""
    global _filter
    if _filter is not None:
        try:
            app.removeEventFilter(_filter)
            _filter.stop_whats_this()
        except RuntimeError:
            pass
    _filter = HelpShortcuts(window, app)
    app.installEventFilter(_filter)


def uninstall_shortcuts(app: QApplication) -> None:
    """F1·Shift+F1 필터를 뗀다(테스트가 다음 테스트에 남기지 않게)."""
    global _filter
    if _filter is not None:
        try:
            _filter.stop_whats_this()
            app.removeEventFilter(_filter)
        except RuntimeError:
            pass
        _filter = None


def start_whats_this(window: QMainWindow) -> None:
    """도움말 메뉴 「이건 뭐예요?」: 다음 클릭을 설명으로 바꾼다(필터가 없으면 단다)."""
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return
    if _filter is None:
        install_shortcuts(app, window)
    assert _filter is not None
    _filter.start_whats_this()


__all__ = ["EDITOR_TABS", "SCREEN_PAGES", "HelpShortcuts", "context_page", "editor_tab_heading", "install_shortcuts",
           "open_screen_help", "screen_help_page", "shortcuts", "start_whats_this", "uninstall_shortcuts", "whats_this",
           "whats_this_text"]

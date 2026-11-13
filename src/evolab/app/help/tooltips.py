"""풍선 도움말 붙이기 (v3 설계 4.6, D24).

화면 파일을 고치지 않고 바깥에서 붙인다: 연결표(`anchors.ANCHORS`)의 그 화면 키 중 strings.py `HELP_TIPS`에 글이 있는 것을 위젯으로
풀어 **풍선이 비어 있을 때만** `setToolTip`(v1·v2가 단 풍선은 그대로). 폼 줄의 이름표(`QFormLayout.labelForField`)에도 같은 글을
단다(이름표에 마우스를 올려도 보이게).

- 주 창: `install(app, window)`가 주 창의 화면(main, 에디터, 실험실, 비교, 기록)에 한 번 붙인다.
- 따로 뜨는 창(직접 조종, 나란히 경주, 반복 실행, 오류 보고, 내보내기): 앱 이벤트 필터가 그 창의 첫 `Show` 이벤트에서 한 번 붙인다.
"""

from __future__ import annotations

from typing import Final

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QApplication, QFormLayout, QMainWindow, QWidget

from evolab.app import strings as S
from evolab.app.help import anchors

MAIN_SCREENS: Final = ("main", *anchors.TAB_SCREENS)
"""주 창이 맨 위인 화면."""

APPLIED_PROPERTY: Final = "evolab_tips_applied"
"""따로 뜨는 창에 이미 붙였다는 표시(Qt 동적 속성)."""

_installer: WindowTips | None = None


def form_label(widget: QWidget) -> QWidget | None:
    """widget이 폼 줄의 칸이면 그 이름표 위젯."""
    parent = widget.parentWidget()
    layout = QWidget.layout(parent) if parent is not None else None  # 속성 layout을 가린 위젯도 있다
    if isinstance(layout, QFormLayout):
        return layout.labelForField(widget)
    return None


def apply_tooltips(root: QWidget, screen: str) -> int:
    """그 화면의 연결표 위젯에 HELP_TIPS 글을 단다(풍선이 비어 있는 위젯만, 폼 이름표도). 글을 단 위젯 수(이름표 빼고)."""
    count = 0
    for key in anchors.keys_for(screen):
        tip = S.HELP_TIPS.get(key)
        if not tip:
            continue
        widget = anchors.resolve(root, key)
        if widget is None:
            continue
        if not widget.toolTip():
            widget.setToolTip(tip)
            count += 1
        label = form_label(widget)
        if label is not None and not label.toolTip():
            label.setToolTip(widget.toolTip())
    return count


class WindowTips(QObject):
    """따로 뜨는 창이 처음 보일 때 그 창 화면의 풍선 도움말을 붙이는 앱 이벤트 필터."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.applied: list[tuple[str, int]] = []
        """(화면, 붙인 수) 차례(테스트용)."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Show and obj.isWidgetType():
            self.window_shown(obj)  # type: ignore[arg-type]
        return False

    def window_shown(self, widget: QWidget) -> None:
        if not widget.isWindow() or widget.property(APPLIED_PROPERTY):
            return
        screen = anchors._window_screen(widget)  # noqa: SLF001 - 연결표와 같은 규칙(클래스 이름)
        if screen is None:
            return
        widget.setProperty(APPLIED_PROPERTY, True)
        self.applied.append((screen, apply_tooltips(widget, screen)))


def install(app: QApplication, window: QMainWindow) -> int:
    """주 창 화면들에 풍선 도움말을 붙이고, 따로 뜨는 창용 필터를 단다(다시 부르면 필터는 하나만). 주 창에 붙인 수."""
    global _installer
    count = sum(apply_tooltips(window, screen) for screen in MAIN_SCREENS)
    if _installer is None:
        _installer = WindowTips(app)
        app.installEventFilter(_installer)
    return count


def window_tips() -> WindowTips | None:
    """달린 필터(없으면 None)."""
    return _installer


def uninstall(app: QApplication) -> None:
    """따로 뜨는 창용 필터를 뗀다(테스트가 다음 테스트에 남기지 않게). 이미 붙인 풍선은 그대로."""
    global _installer
    if _installer is not None:
        try:
            app.removeEventFilter(_installer)
        except RuntimeError:
            pass
        _installer = None


__all__ = ["MAIN_SCREENS", "WindowTips", "apply_tooltips", "form_label", "install", "uninstall", "window_tips"]

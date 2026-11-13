"""풍선 도움말 붙이기 (v3 설계 4.6, D24).

화면 파일을 고치지 않고 바깥에서 붙인다: 연결표(`anchors.ANCHORS`)의 그 화면 키 중 strings.py `HELP_TIPS`에 글이 있는 것을 위젯으로
풀어 **풍선이 비어 있을 때만** `setToolTip`(v1·v2가 단 풍선은 그대로). 폼 줄의 이름표(`QFormLayout.labelForField`)에도 같은 글을
단다(이름표에 마우스를 올려도 보이게).

- 주 창: `install(app, window)`가 주 창의 화면(main, 에디터, 실험실, 비교, 기록)에 한 번 붙인다.
- 따로 뜨는 창(직접 조종, 나란히 경주, 반복 실행, 오류 보고, 내보내기): 앱 이벤트 필터가 그 창의 첫 `Show` 이벤트에서 한 번 붙인다.
- 같은 앱 필터가 풍선을 띄운다(6.2.0 U3): 긴 풍선은 420px 폭으로 꺾고(`style.wrap_tip`), 꺼진 위젯은 「꺼진 까닭」을 붙이며
  (`style.tip_text`), 표·목록·나무 칸의 긴 풍선도 꺾는다. `toolTip()`에 든 글은 바꾸지 않는다.
- 같은 필터가 `ApplicationPaletteChange`에서 `style.refresh_theme()`을 부른다(6.2.0 U1: 꾸민 위젯은 팔레트 바뀜을 받지 못한다).
"""

from __future__ import annotations

from typing import Final

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QHelpEvent
from PySide6.QtWidgets import QAbstractItemView, QApplication, QFormLayout, QMainWindow, QToolTip, QWidget

from evolab.app import strings as S
from evolab.app import style
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
    """앱 이벤트 필터: 따로 뜨는 창이 처음 보일 때 그 창 화면의 풍선 도움말을 붙이고, 풍선을 띄우며(U3), 테마가 바뀌면 다시 입힌다(U1)."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.applied: list[tuple[str, int]] = []
        """(화면, 붙인 수) 차례(테스트용)."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        kind = event.type()
        if kind == QEvent.Type.Show and obj.isWidgetType():
            self.window_shown(obj)  # type: ignore[arg-type]
        elif kind == QEvent.Type.ToolTip and obj.isWidgetType() and isinstance(event, QHelpEvent):
            return self.show_tip(obj, event)  # type: ignore[arg-type]
        elif kind == QEvent.Type.ApplicationPaletteChange:
            style.refresh_theme()
        return False

    def show_tip(self, widget: QWidget, event: QHelpEvent) -> bool:
        """풍선 하나를 띄운다. 참이면 이 필터가 띄웠고, 거짓이면 Qt가 하던 대로(짧은 보통 풍선, 빈 풍선은 부모로 올라감).

        ① 위젯 풍선: style.tip_text(보통 풍선, 꺼져 있으면 + 까닭). 한 줄이 420px을 넘으면 싸서, 아니면 까닭이 붙었을 때만 띄운다.
        ② 표·목록·나무의 뷰포트(위젯 풍선이 빔): 그 칸의 ToolTipRole 글이 길면 싸서 띄운다(칸의 글은 그대로).
        """
        text = style.tip_text(widget)
        if not text:
            view = widget.parentWidget()
            if isinstance(view, QAbstractItemView) and widget is view.viewport():
                index = view.indexAt(event.pos())
                tip = index.data(Qt.ItemDataRole.ToolTipRole) if index.isValid() else None
                wrapped = style.wrap_tip(tip) if isinstance(tip, str) else None
                if wrapped is not None:
                    QToolTip.showText(event.globalPos(), wrapped, widget, view.visualRect(index))
                    return True
            return False
        wrapped = style.wrap_tip(text)
        if wrapped is None and text == widget.toolTip():
            return False
        QToolTip.showText(event.globalPos(), wrapped or text, widget)
        return True

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

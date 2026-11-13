"""화면의 작은 디자인 시스템: 색·간격·글자·단추 위계·끈 까닭·숫자 형식의 한 집 (6.2.0 화면 다듬기 U1).

목표는 새 모양이 아니라 화면마다 흩어져 있던 값을 한 곳에 모으는 것이다. 윈도우 기본 모양(단추·묶음 상자·탭)은 그대로 두고
앱이 직접 정하는 것(강조·보조 글·흐린 글·으뜸 단추·상태 점·숫자)만 여기서 정한다.

- Qt 위젯과 strings.py만 불러온다(pyqtgraph·numpy·화면 모듈을 불러오지 않는다: 에디터·기록·도움말·오류 상자가 색 하나 때문에
  실험실 쪽을 끌어오지 않게). 그래프 색 묶음 `Theme`·`LIGHT`·`DARK`·`theme_for`도 여기 살고 app/lab/dashboard.py가 같은 이름으로 다시
  내보낸다.
- 앱 전체 스타일시트는 쓰지 않는다. 이 모듈의 함수는 **위젯 하나에만** 색·글꼴을 입힌다.
- 테마는 앱 팔레트로 고른다: `is_dark()` = `QApplication.palette()`의 창 색 밝기 < 128. 꾸민 위젯 자신의 팔레트는 보지 않는다
  (스타일시트에 바탕색이 있으면 그 위젯의 창 색이 스타일시트 색으로 바뀌어, 밝은 테마의 으뜸 단추가 「어둡다」고 읽힌다).
- 꾸민 위젯은 앱 팔레트가 바뀌어도 `PaletteChange`를 받지 못한다. 그래서 `set_role`·`tone`이 입힌 위젯을 등록부에 두고, 이미 있는
  앱 이벤트 필터(app/help/tooltips.py의 WindowTips)가 `ApplicationPaletteChange`에서 `refresh_theme()`을 한 번 부른다.
- `set_role`·`tone`·`set_enabled`·`StateDot.set_kind`·`StateChip.set_state`는 값이 지난번과 같으면 아무것도 하지 않고 돌아간다
  (실험실은 이것들을 세대마다 부른다).
- 끈 까닭: `toolTip()`은 늘 「이것이 무엇인가」의 글이다. `set_enabled(대상, 켜짐, 까닭)`은 까닭을 동적 속성 `evoWhyOff`에만 두고,
  풍선을 띄울 때 필터가 `tip_text(위젯)`(= 보통 풍선, 꺼져 있고 까닭이 있으면 + 빈 줄 + 까닭)을 띄운다.
"""

from __future__ import annotations

import datetime as dt
import html
import math
from dataclasses import dataclass
from typing import Final

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics, QPainter, QPalette
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QToolTip,
    QWidget,
)

from evolab.app import strings as S


# ---------------------------------------------------------------------------
# 색
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Theme:
    """그래프 색 묶음(밝은 화면·어두운 화면 각각 고른 값. 범주 색은 정해진 순서로만 쓴다)."""

    surface: str
    ink: str
    ink2: str
    muted: str
    """그림의 흐린 색(뇌 그림의 마디, 유전자 지도의 자취, 정체 띠, 십자선, 「기타」 계열). 글자에는 쓰지 않는다: 글자는 tick."""
    grid: str
    axis: str
    series: tuple[str, ...]
    ramp: tuple[str, ...]  # 순차 색(히트맵): 바탕에서 멀어지는 순서
    tick: str
    """글자 전용 흐린 색(그래프 눈금 글, 덧붙임 글): 모든 바탕에서 4.5:1 이상. = Ui.muted"""


LIGHT: Final = Theme(
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    ramp=("#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"),
    tick="#6b6963",
)
DARK: Final = Theme(
    surface="#1a1a19",
    ink="#f0efec",
    ink2="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    ramp=("#104281", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4", "#cde2fb"),
    tick="#a3a199",
)


@dataclass(frozen=True)
class Ui:
    """화면 글·단추·상태 점의 색(구실별). 새 색을 만들지 않고 그래프 색 묶음의 값을 다시 쓴다(흐린 글 둘만 새로 고른 값)."""

    accent: str
    """강조: 으뜸 단추 바탕."""
    accent_hover: str
    accent_down: str
    on_accent: str
    """강조색 위의 글."""
    link: str
    ink2: str
    """보조 글(안내 줄, 범례, 캡션)."""
    muted: str
    """흐린 글(읽을 수 없는 실행 줄, 묶음 제목, 끈 으뜸 단추의 글)."""
    danger: str
    """오류 글, 위험 단추의 글·테두리."""
    ok: str
    """상태 점: 학습 중."""
    warn: str
    """상태 점: 잠시 멈춤."""
    info: str
    """상태 점: 시작하는 중·조건이 차서 멈춤."""
    idle: str
    """상태 점: 학습 전·저장됨."""
    bad: str
    """상태 점: 실패."""
    line: str
    """가는 테두리."""
    focus: str
    """키보드 초점 테두리(그 테마의 본문 글 색)."""


UI_LIGHT: Final = Ui(
    accent="#256abf", accent_hover="#1c5cab", accent_down="#184f95", on_accent="#ffffff", link="#256abf",
    ink2="#52514e", muted="#6b6963", danger="#b3261e", ok="#008300", warn="#b45f06", info="#256abf", idle="#6b6963",
    bad="#d03b3b", line="#c3c2b7", focus="#0b0b0b",
)
UI_DARK: Final = Ui(
    accent="#5598e7", accent_hover="#6da7ec", accent_down="#3987e5", on_accent="#0b0b0b", link="#86b6ef",
    ink2="#c3c2b7", muted="#a3a199", danger="#ff7b72", ok="#0ca30c", warn="#fab219", info="#5598e7", idle="#a3a199",
    bad="#e66767", line="#383835", focus="#f0efec",
)

LIGHT_SURFACES: Final = ("#efefef", "#f3f3f3", "#ffffff", "#fcfcfb")
"""밝은 테마에서 글이 놓일 수 있는 바탕: Fusion 창, 윈도우 11 창, 칸, 그래프. tests/test_style.py가 이 위의 대비를 지킨다."""
DARK_SURFACES: Final = ("#202020", "#2b2b2b", "#323232", "#191919", "#1a1a19")
"""어두운 테마의 바탕: 윈도우 11 창 둘, Fusion 어두운 창, 칸, 그래프."""

SPACE: Final = (4, 8, 12, 16, 24)
"""앱이 직접 만든 그릇(카드, 알림, 대화 상자 머리, 도구 줄, 도움말 글)에 쓰는 간격(px). Qt 기본 배치의 간격은 건드리지 않는다."""

ROLE_PROPERTY: Final = "evoRole"
"""단추의 급(set_role)을 적는 Qt 동적 속성."""
WHY_PROPERTY: Final = "evoWhyOff"
"""끈 까닭(set_enabled)을 적는 Qt 동적 속성. 위젯이나 QAction에 건다."""
BUTTON_ROLES: Final = ("primary", "danger", "quiet", "normal")
TONES: Final = ("ink2", "muted", "danger", "link")
FONT_ROLES: Final = ("small", "body", "strong", "title", "display", "mono")
MONO_FAMILIES: Final = ("Consolas", "D2Coding", "Malgun Gothic", "monospace")
TIP_WIDTH: Final = 420
"""긴 풍선을 싸는 폭(px). 띄운 풍선 창은 여백을 더해 428px쯤이다."""


def is_dark(widget: QWidget | None = None) -> bool:
    """어두운 테마인가: 앱 팔레트의 창 색 밝기 < 128. 인자는 보지 않는다(부르는 쪽이 그대로이게 받기만 한다)."""
    app = QApplication.instance()
    if app is None:
        return False
    return QApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128


def theme_for(widget: QWidget | None = None) -> Theme:
    """지금 테마(윈도우 밝은/어두운 모드)에 맞는 그래프 색 묶음."""
    return DARK if is_dark() else LIGHT


def ui_for(widget: QWidget | None = None) -> Ui:
    """지금 테마에 맞는 화면 색 묶음."""
    return UI_DARK if is_dark() else UI_LIGHT


# ---------------------------------------------------------------------------
# 등록부: 테마가 바뀌면 다시 입힐 위젯
# ---------------------------------------------------------------------------
_styled: dict[int, tuple[QWidget, str, str, bool]] = {}
"""id(위젯) → (위젯, "role" | "tone", 값, on_dark_card). 위젯이 사라지면 destroyed로 빠진다."""
_applied: dict[int, tuple[str, str, bool]] = {}
"""id(위젯) → 지난번에 입힌 (종류, 값, 어두운가). 같으면 다시 입히지 않는다."""
_last_dark: bool | None = None


def _forget(key: int) -> None:
    _styled.pop(key, None)
    _applied.pop(key, None)


def _register(widget: QWidget, kind: str, value: str, card: bool) -> None:
    key = id(widget)
    if key not in _styled:
        widget.destroyed.connect(lambda _obj=None, k=key: _forget(k))
    _styled[key] = (widget, kind, value, card)


def refresh_theme() -> int:
    """테마(앱 팔레트의 밝기)가 지난번과 달라졌으면 등록부의 위젯을 다시 입힌다. 다시 입힌 수(그대로면 0).

    앱 이벤트 필터가 ApplicationPaletteChange에서 부른다. 필터가 없는 곳(명령줄로만 띄운 창, 필터를 떼는 테스트)에서는 직접 부른다.
    """
    global _last_dark
    dark = is_dark()
    if _last_dark is dark:
        return 0
    first = _last_dark is None
    _last_dark = dark
    count = 0
    for key, (widget, kind, value, card) in list(_styled.items()):
        was = _applied.get(key)
        try:
            if kind == "role":
                set_role(widget, value, on_dark_card=card)  # type: ignore[arg-type]
            else:
                tone(widget, value)
        except RuntimeError:  # 이미 지워진 Qt 객체(destroyed를 놓친 경우)
            _forget(key)
            continue
        if _applied.get(key) != was:
            count += 1
    return 0 if first and count == 0 else count


def _note_theme(dark: bool) -> None:
    """입힐 때의 테마를 적어 둔다(처음 refresh_theme이 「바뀌지 않았다」를 알게)."""
    global _last_dark
    if _last_dark is None:
        _last_dark = dark


# ---------------------------------------------------------------------------
# 글자
# ---------------------------------------------------------------------------
def font(widget: QWidget, role: str) -> QFont:
    """그 위젯 글꼴에 대한 상대값으로 만든 글꼴(입히지는 않는다: `w.setFont(style.font(w, "title"))`).

    small ×0.9(8pt 아래로는 안 내려감) · body 그대로 · strong 굵게 · title +2pt 굵게 · display +4pt 굵게 · mono 고정폭.
    글을 줄이는 수단은 크기가 아니라 색 단계(본문 → 보조 → 흐린)다.
    """
    if role not in FONT_ROLES:
        raise ValueError(f"unknown font role: {role!r}")
    f = QFont(widget.font())
    size = f.pointSizeF()
    if size <= 0:  # 픽셀 크기로 정한 글꼴(드묾): 크기는 그대로 둔다
        size = 0.0
    if role == "small" and size > 0:
        f.setPointSizeF(max(min(size, 8.0), size * 0.9))
    elif role in ("strong", "title", "display"):
        f.setBold(True)
        if size > 0 and role != "strong":
            f.setPointSizeF(size + (2.0 if role == "title" else 4.0))
    elif role == "mono":
        f.setFamilies(list(MONO_FAMILIES))
        f.setStyleHint(QFont.StyleHint.Monospace)
    return f


def tone(widget: QWidget, role: str) -> None:
    """글자색만 입힌다: "ink2"(보조 글) | "muted"(흐린 글) | "danger"(오류 글) | "link". 테마가 바뀌면 다시 입힌다."""
    if role not in TONES:
        raise ValueError(f"unknown tone: {role!r}")
    dark = is_dark()
    key = id(widget)
    if _applied.get(key) == ("tone", role, dark):
        return
    ui = UI_DARK if dark else UI_LIGHT
    widget.setStyleSheet(f"color: {getattr(ui, role)};")
    _register(widget, "tone", role, False)
    _applied[key] = ("tone", role, dark)
    _note_theme(dark)


def note(text: str = "") -> QLabel:
    """줄바꿈을 켠 보조 글 한 줄(「흐린 안내 줄」의 표준). 묶음 안의 설명은 칸 아래에 이것 하나로 둔다."""
    label = QLabel(text)
    label.setWordWrap(True)
    tone(label, "ink2")
    return label


# ---------------------------------------------------------------------------
# 단추 위계
# ---------------------------------------------------------------------------
def _native_height(button: QAbstractButton) -> int:
    """꾸미기 전 그 단추 자신의 높이(스타일시트를 비우고 잰다: 도구 줄 단추는 새 단추와 높이가 다르다)."""
    if button.styleSheet():
        button.setStyleSheet("")
    button.ensurePolished()
    return int(button.sizeHint().height())


def _button_selector(button: QAbstractButton) -> str:
    return "QToolButton" if isinstance(button, QToolButton) else "QPushButton"


def _paddings(button: QAbstractButton, border: int) -> tuple[int, int]:
    """꾸민 단추의 높이가 꾸미기 전과 같아지는 위·아래 안쪽 여백(테두리 border px).

    글꼴 높이로 셈하지 않고 잰다: 여백 0인 같은 스타일시트를 잠깐 입혀 그 높이를 재고 꾸미기 전 높이와의 차를 나눈다
    (스타일시트 단추의 크기는 글꼴 높이가 아니라 줄 간격 등으로 정해져 1px씩 어긋났다).
    """
    native = _native_height(button)
    cls = _button_selector(button)
    button.setStyleSheet(f"{cls} {{ font-weight: bold; border: {border}px solid transparent; padding: 0px; }}")
    button.ensurePolished()
    room = max(0, native - int(button.sizeHint().height()))
    return room // 2, room - room // 2


def _primary_qss(button: QAbstractButton, ui: Ui) -> str:
    cls = _button_selector(button)
    top, bottom = _paddings(button, 2)
    return (
        f"{cls} {{ background: {ui.accent}; color: {ui.on_accent}; font-weight: bold; border: 2px solid {ui.accent}; "
        f"border-radius: 4px; padding: {top}px 12px {bottom}px 12px; }}"
        f"{cls}:hover {{ background: {ui.accent_hover}; border-color: {ui.accent_hover}; }}"
        f"{cls}:pressed {{ background: {ui.accent_down}; border-color: {ui.accent_down}; }}"
        f"{cls}:focus {{ border-color: {ui.focus}; }}"
        f"{cls}:disabled {{ background: transparent; color: {ui.muted}; font-weight: normal; border: 1px solid {ui.line}; "
        f"padding: {top + 1}px 13px {bottom + 1}px 13px; }}"
    )


def _danger_qss(button: QAbstractButton, ui: Ui) -> str:
    cls = _button_selector(button)
    top, bottom = _paddings(button, 2)
    return (
        f"{cls} {{ background: transparent; color: {ui.danger}; border: 1px solid {ui.danger}; border-radius: 4px; "
        f"padding: {top + 1}px 13px {bottom + 1}px 13px; }}"
        f"{cls}:hover {{ border: 2px solid {ui.danger}; padding: {top}px 12px {bottom}px 12px; }}"
        f"{cls}:pressed {{ background: {ui.line}; border: 2px solid {ui.danger}; padding: {top}px 12px {bottom}px 12px; }}"
        f"{cls}:focus {{ border: 2px solid {ui.focus}; padding: {top}px 12px {bottom}px 12px; }}"
        f"{cls}:disabled {{ color: {ui.muted}; border: 1px solid {ui.line}; padding: {top + 1}px 13px {bottom + 1}px 13px; }}"
    )


def _set_quiet(button: QAbstractButton, on: bool) -> None:
    if isinstance(button, QToolButton):
        if button.autoRaise() != on:
            button.setAutoRaise(on)
    elif isinstance(button, QPushButton) and button.isFlat() != on:
        button.setFlat(on)


def set_role(button: QAbstractButton, role: str, *, on_dark_card: bool = False) -> None:
    """단추의 급: "primary"(으뜸) | "danger"(위험) | "quiet"(조용한) | "normal"(윈도우 기본 그대로).

    - 으뜸: 강조색 채움 + 굵은 글. 높이는 꾸미기 전 그 단추의 높이와 같다. 화면의 한 상태에 **켜진** 으뜸은 많아야 하나.
      꺼지면 채움 없이 흐린 글과 가는 테두리(옆의 꺼진 기본 단추와 같은 무게): 까닭은 set_enabled로 풍선에.
    - 위험: 위험색 글과 테두리(채우지 않음). 되돌릴 수 없는 것(삭제).
    - 조용한: 테두리 없는 단추(도구 단추는 autoRaise, 누름 단추는 flat).
    - on_dark_card: 늘 어두운 카드 위의 단추(캔버스 안내 카드)는 테마와 무관하게 어두운 테마 값을 쓴다.

    그 단추 하나에만 스타일시트를 건다. (급, 테마)가 지난번과 같으면 바로 돌아간다. 급은 동적 속성 evoRole에 적힌다.
    """
    if role not in BUTTON_ROLES:
        raise ValueError(f"unknown button role: {role!r}")
    dark = True if on_dark_card else is_dark()
    key = id(button)
    if _applied.get(key) == ("role", role, dark):
        return
    ui = UI_DARK if dark else UI_LIGHT
    was_quiet = button.property(ROLE_PROPERTY) == "quiet"
    if role == "primary":
        button.setStyleSheet(_primary_qss(button, ui))
    elif role == "danger":
        button.setStyleSheet(_danger_qss(button, ui))
    elif button.styleSheet():
        button.setStyleSheet("")
    if role == "quiet" or was_quiet:
        _set_quiet(button, role == "quiet")
    button.setProperty(ROLE_PROPERTY, role)
    _register(button, "role", role, on_dark_card)
    _applied[key] = ("role", role, dark)
    if not on_dark_card:
        _note_theme(dark)


def primaries(root: QWidget) -> list[QAbstractButton]:
    """root 아래에서 보이는 으뜸 단추(켜짐·꺼짐 모두). 「켜진 으뜸은 많아야 하나」는 부르는 쪽이 isEnabled()로 가른다."""
    return [b for b in root.findChildren(QAbstractButton)
            if b.property(ROLE_PROPERTY) == "primary" and b.isVisibleTo(root)]


# ---------------------------------------------------------------------------
# 상태 점과 꼬리표
# ---------------------------------------------------------------------------
class StateDot(QWidget):
    """상태 색 점 하나(글자 한 줄 높이, 초점 없음, 움직이지 않는 그림). 그릴 때 ui_for()를 읽어 테마를 따라간다."""

    KINDS: Final = {"idle": "idle", "closed": "idle", "starting": "info", "running": "ok", "paused": "warn",
                    "finished": "info", "failed": "bad"}
    """상태 → Ui의 색 칸."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = "idle"
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def sizeHint(self) -> QSize:  # noqa: N802
        h = self.fontMetrics().height()
        return QSize(max(6, h - 4), h)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self.sizeHint()

    def set_kind(self, kind: str) -> None:
        """"idle" | "starting" | "running" | "paused" | "finished" | "closed" | "failed". 같으면 돌아간다.

        모르는 상태는 「학습 전」 색으로 그린다(세대마다 불리는 길이라 예외를 던지지 않는다)."""
        if kind != self.kind:
            self.kind = kind
            self.update()

    def color(self) -> str:
        """지금 테마에서 이 점의 색."""
        return str(getattr(ui_for(), self.KINDS.get(self.kind, "idle")))

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self.color()))
        h = self.fontMetrics().height()
        d = max(4, h - 6)
        painter.drawEllipse(0, (h - d) // 2, d, d)
        painter.end()


class StateChip(QWidget):
    """상태 점 + 글 한 줄. 주 창 상태 막대용(실험실 패널에는 쓰지 않는다: 접힘 예산)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACE[0], 0, SPACE[0], 0)
        layout.setSpacing(SPACE[0])
        self.dot = StateDot(self)
        self.label = QLabel(self)
        layout.addWidget(self.dot)
        layout.addWidget(self.label)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_state(self, kind: str, text: str) -> None:
        """점의 상태와 글. 같으면 아무것도 하지 않는다."""
        self.dot.set_kind(kind)
        if self.label.text() != text:
            self.label.setText(text)


# ---------------------------------------------------------------------------
# 끈 까닭과 풍선
# ---------------------------------------------------------------------------
def set_enabled(target: QWidget | QAction, on: bool, why: str = "") -> None:
    """켜짐과 「꺼진 까닭」. toolTip()은 건드리지 않는다: 까닭은 동적 속성 evoWhyOff에 두고 풍선을 띄울 때 합친다.

    도구 줄 단추처럼 QAction을 따라가는 것은 액션에 건다(why_off가 단추의 defaultAction을 본다). 값이 같으면 아무것도 안 한다.
    """
    if target.isEnabled() != on:
        target.setEnabled(on)
    if (target.property(WHY_PROPERTY) or "") != why:
        target.setProperty(WHY_PROPERTY, why)


def why_off(widget: QWidget) -> str:
    """꺼져 있고 까닭이 있으면 그 글(도구 줄 단추는 제 QAction의 까닭도 본다). 켜져 있으면 빈 글."""
    if widget.isEnabled():
        return ""
    why = widget.property(WHY_PROPERTY)
    if not why and isinstance(widget, QToolButton) and widget.defaultAction() is not None:
        why = widget.defaultAction().property(WHY_PROPERTY)
    return str(why or "")


def tip_text(widget: QWidget) -> str:
    """지금 이 위젯에서 뜰 풍선 글: toolTip(), 꺼져 있고 까닭이 있으면 빈 줄 뒤에 까닭. 필터와 테스트가 함께 쓴다."""
    base = widget.toolTip()
    why = why_off(widget)
    if base and why:
        return f"{base}\n\n{why}"
    return why or base


def is_rich(text: str) -> bool:
    """이미 서식이 든 풍선 글인가(싸지 않고 그대로 둔다)."""
    return text.lstrip().startswith("<")


def wrap_tip(text: str, width: int = TIP_WIDTH) -> str | None:
    """한 줄이 width px을 넘는 보통 글 풍선을 그 폭의 표로 싼 서식 글. 짧거나 이미 서식 글이면 None(그대로 두라는 뜻).

    Qt의 보통 글 풍선은 줄바꿈이 없어 긴 글이 화면 폭까지 한 줄로 뜬다. `<table width=…>`는 늘 그 폭으로 꺾인다.
    """
    if not text or is_rich(text):
        return None
    metrics = QFontMetrics(QToolTip.font())
    if max(metrics.horizontalAdvance(line) for line in text.split("\n")) <= width:
        return None
    return f'<table width="{int(width)}"><tr><td>' + html.escape(text).replace("\n", "<br>") + "</td></tr></table>"


def align_form_labels(*forms: QFormLayout) -> None:
    """여러 폼의 이름표 열 폭을 가장 넓은 이름표에 맞춘다(한 패널의 폼들이 한 줄로 서게)."""
    labels: list[QWidget] = []
    for form in forms:
        for row in range(form.rowCount()):
            item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
            widget = item.widget() if item is not None else None
            if widget is not None:
                labels.append(widget)
    if not labels:
        return
    width = max(w.sizeHint().width() for w in labels)
    for w in labels:
        if w.minimumWidth() != width:
            w.setMinimumWidth(width)


# ---------------------------------------------------------------------------
# 숫자 형식
# ---------------------------------------------------------------------------
def _finite(v: float | None) -> bool:
    if v is None or isinstance(v, (str, bytes)):
        return False
    try:
        return math.isfinite(v)
    except (TypeError, ValueError):
        return False


def _no_negative_zero(text: str) -> str:
    """반올림해 0이 된 음수의 부호를 뗀다(「-0.0초」가 아니라 「0.0초」)."""
    head = text.lstrip("-")
    digits = [ch for ch in head if ch.isdigit()]
    return head if text.startswith("-") and digits and all(ch == "0" for ch in digits) else text


def fmt_score(v: float | None) -> str:
    """점수: 절댓값 1,000 이상은 소수 없이 천 단위 구분, 100 이상은 소수 1자리, 1 이상은 2자리, 그 아래는 유효 3자리.

    −0은 0으로, 값이 없으면(None·NaN·무한대) S.CMP_NONE. app/lab/dashboard.py의 fmt_value가 이 함수의 다른 이름이다.
    """
    if not _finite(v):
        return S.CMP_NONE
    x = float(v)  # type: ignore[arg-type]
    a = abs(x)
    if a >= 1000:
        text = f"{x:,.0f}"
    elif a >= 100:
        text = f"{x:.1f}"
    elif a >= 1:
        text = f"{x:.2f}"
    elif a == 0:
        return "0"
    else:
        text = f"{x:.3g}"
    return _no_negative_zero(text)


def fmt_count(n: float) -> str:
    """세는 수: 정수, 천 단위 구분(단위 「회」「세대」「개」는 부르는 쪽 문구가 붙여 쓴다). 값이 없으면 S.CMP_NONE."""
    return f"{round(n):,}" if _finite(n) else S.CMP_NONE


def fmt_seconds(s: float) -> str:
    """짧은 시간: 소수 1자리 + 「초」."""
    return _no_negative_zero(S.CMP_SECONDS.format(s=float(s))) if _finite(s) else S.CMP_NONE


def fmt_duration(s: float) -> str:
    """긴 시간: 60초 미만 「n.n초」, 그 위 「n분 nn초」, 1시간 위 「n시간 nn분」(비교 화면의 format_duration과 같은 규칙)."""
    if not _finite(s):
        return S.CMP_NONE
    if s < 60:
        return _no_negative_zero(S.CMP_SECONDS.format(s=float(s)))
    total = round(s)
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return S.CMP_HOURS.format(h=hours, m=minutes)
    return S.CMP_MINUTES.format(m=minutes, s=seconds)


def fmt_percent(v: float) -> str:
    """비율: 정수 %. 인자는 이미 퍼센트 값이다(85.0 → 「85%」)."""
    return _no_negative_zero(S.CMP_PERCENT.format(v=float(v))) if _finite(v) else S.CMP_NONE


def fmt_when(iso: str, now: dt.datetime | None = None) -> str:
    """ISO 시각 글 → 「11월 13일 12:07」, 해가 지금과 다르면 해를 앞에. 읽을 수 없는 글은 그대로 돌려준다(풍선에는 ISO 글을 둔다)."""
    try:
        when = dt.datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return str(iso)
    year = (now or dt.datetime.now()).year
    template = S.STYLE_WHEN if when.year == year else S.STYLE_WHEN_YEAR
    return template.format(year=when.year, month=when.month, day=when.day, hour=when.hour, minute=when.minute)


__all__ = [
    "BUTTON_ROLES", "DARK", "DARK_SURFACES", "FONT_ROLES", "LIGHT", "LIGHT_SURFACES", "MONO_FAMILIES", "ROLE_PROPERTY",
    "SPACE", "TIP_WIDTH", "TONES", "UI_DARK", "UI_LIGHT", "WHY_PROPERTY", "StateChip", "StateDot", "Theme", "Ui",
    "align_form_labels", "fmt_count", "fmt_duration", "fmt_percent", "fmt_score", "fmt_seconds", "fmt_when", "font",
    "is_dark", "is_rich", "note", "primaries", "refresh_theme", "set_enabled", "set_role", "theme_for", "tip_text", "tone",
    "ui_for", "why_off", "wrap_tip",
]

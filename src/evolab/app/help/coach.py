"""코치 오버레이: 진짜 주 창 위에 얹는 말풍선과 어두운 막 (v3 설계 4.2, D7, 5.3, 7.3).

- 주 창(또는 따로 뜨는 창)의 자식 위젯이 창 전체를 덮는다. 「해 보기」(어두운) 단계는 대상 둘레(+6px)에 구멍을 낸 마스크를 써서
  구멍 안 클릭만 아래의 진짜 위젯으로 간다(setMask). 「보기」(밝은) 단계의 마스크는 말풍선과 대상 둘레 주황 띠뿐이라 나머지
  화면은 그대로 쓴다.
- 대상은 연결표 키(anchors.ANCHORS)로 찾는다. 100ms마다 속성만 읽어(대상 위치, 넘어가는 조건, 글) 구멍이 스크롤·크기 바뀜을
  따라가고, 대상이 안 보이면 단계의 준비(prepare)를 다시 한다. 세 번 연속 안 보이면 말풍선을 가운데로 옮기고, 대상 화면 탭을
  떠난 것이면 「실험실 탭으로 돌아가기」 단추를 단다.
- Esc = 건너뛰기, Enter = 「다음」(또는 카드의 기본 단추). 키는 앱 이벤트 필터가 받으므로 초점이 어디 있든 된다(그 창 안일 때만).
  다만 밝은(어둡게 하지 않는) 단계에서는 화면을 그대로 쓰므로, 글·숫자 칸·목록에 초점이 있거나 실험실 밖 탭(에디터 등)의 키는
  그 위젯에 그대로 간다(`_takes_key`).
- 「짚기」(point): 어둡게 하지 않는 말풍선 하나. 4초가 지나거나 아무 곳이나 누르면 닫힌다(도움말 링크·Shift+F1이 쓴다).

문구는 strings.py에만 있다(ONB_*, WHATS_THIS_MORE). 이 모듈은 실험실·주 창 클래스를 불러오지 않는다.
"""

from __future__ import annotations

import time
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QRegion
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from evolab.app import strings as S
from evolab.app.help import anchors

APP_FILTER_FALLBACK_MS: Final = 1000
"""창이 그려지기 전에 시작한 안내가 이 시간 안에 그려지지 않으면 그때 앱 전체 필터를 단다(start)."""
POLL_MS: Final = 100
"""속성을 읽어 다시 배치하는 간격(설계 4.2: 100ms 확인)."""
HOLE_MARGIN: Final = 6
"""구멍 = 대상 사각형 + 6px."""
RING_WIDTH: Final = 3
RING_PAD: Final = 2
"""구멍 가장자리 주황 띠의 반폭(+여유). 어두운 단계에서는 구멍 안쪽 RING_PAD까지 막에 넣어 띠가 다 보이게, 밝은 단계에서는
가장자리 ±RING_PAD 띠만 마스크에 넣어 띠가 보이게 한다. 구멍이 대상보다 HOLE_MARGIN 크므로 대상은 늘 눌린다."""
RING_RADIUS: Final = 8
DIM_ALPHA: Final = 150
RING_COLOR: Final = "#f5a524"
PAPER_COLOR: Final = "#fffbe6"
INK_COLOR: Final = "#202020"
FAINT_INK_COLOR: Final = "#6b5d3a"
BUBBLE_WIDTH: Final = 360
BUBBLE_GAP: Final = 12
BUBBLE_PAD_X: Final = 16
BUBBLE_PAD_Y: Final = 12
BUBBLE_ROW_GAP: Final = 8
DEFAULT_PROPERTY: Final = "coachDefault"
"""말풍선의 기본 단추(Enter가 누르는 것) 표시(동적 속성). 스타일시트가 이 속성으로 호박색 채움을 고른다(6.2.0 U33)."""


def bubble_style_sheet() -> str:
    """말풍선 스타일시트(만들 때 한 번). 종이 위 단추는 테마와 무관한 고정 색이다(어두운 테마에서 크림 종이 위 어두운 단추가 뜨지 않게):
    기본 단추 = 호박색 채움 + 글 #202020(7.98:1) + 굵게 + 1px 테두리 #6b5d3a, 나머지 = 종이색 + 같은 테두리(종이와 6.21:1).
    올림·키보드 초점은 테두리만 글 색으로(새 색을 만들지 않는다: 계획 1.2)."""
    button = "#coachBubble QPushButton#coachButton"
    return (
        f"#coachBubble {{ background: {PAPER_COLOR}; border: 2px solid {RING_COLOR}; border-radius: 10px; }}"
        f"#coachBubble QLabel {{ color: {INK_COLOR}; background: transparent; }}"
        f"#coachBubble QLabel#coachCounter, #coachBubble QLabel#coachStatus {{ color: {FAINT_INK_COLOR}; }}"
        f"{button} {{ background: {PAPER_COLOR}; color: {INK_COLOR}; border: 1px solid {FAINT_INK_COLOR};"
        f" border-radius: 4px; padding: 3px 12px; }}"
        f"{button}:hover, {button}:focus {{ border: 1px solid {INK_COLOR}; }}"
        f"{button}:disabled {{ color: {FAINT_INK_COLOR}; background: transparent; }}"
        f"{button}[{DEFAULT_PROPERTY}=\"true\"] {{ background: {RING_COLOR}; color: {INK_COLOR}; font-weight: bold; }}"
        f"{button}[{DEFAULT_PROPERTY}=\"true\"]:disabled {{ background: transparent; color: {FAINT_INK_COLOR};"
        f" font-weight: normal; }}"
    )


def is_card(step: CoachStep) -> bool:
    """제목이 있는 환영 종류의 카드(처음 실행 환영, 이어서 하기): 제목 +2pt, 단계 수 숨김(6.2.0 U33)."""
    return step.kind == "welcome" and bool(step.title)
EDGE: Final = 8
POINT_SECONDS: Final = 4.0
MISSES_BEFORE_CENTER: Final = 3
"""대상이 이만큼 연속으로(100ms마다) 안 보이면 말풍선을 가운데로 옮긴다."""

_KEY_PRESS: Final = QEvent.Type.KeyPress
_MOUSE_PRESS: Final = QEvent.Type.MouseButtonPress
_HOST_EVENTS: Final = (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.LayoutRequest)
"""앱 이벤트 필터가 모든 이벤트(초당 약 1,000번)에서 비교하는 종류: 속성 찾기를 한 번만 하게 미리 꺼내 둔다."""

CLOCK: Callable[[], float] = time.monotonic
"""단계 시각을 재는 시계. 새 오버레이가 만들어질 때 읽는다(테스트가 바꿔 끼운다)."""

TYPING_WIDGETS: Final = (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit, QAbstractItemView)
"""밝은 단계에서 Esc·Enter를 가져가지 않는 초점 위젯(글·숫자 칸은 Enter로 값을 정하고, 목록은 Enter·Esc를 쓴다)."""
KEY_SCREENS: Final = (None, "main", "lab")
"""밝은 단계에서 Esc·Enter를 받는 화면(anchors.screen_of). 다른 탭(에디터·비교·기록)의 키는 그 화면 것이다."""

Kind = Literal["welcome", "click", "wait", "next", "value"]
Buttons = tuple[tuple[str, Callable[[], None]], ...]


@dataclass
class CoachStep:
    """안내 한 단계 (설계 7.3 계약 + 기본값이 있는 덧붙인 칸).

    kind:
      - "welcome": 카드. 단계 단추(buttons)만 둔다(「다음」·「건너뛰기」 없음). Enter = buttons[default]. ⑦처럼 끝내는 말풍선도 이 종류.
      - "click": 대상(연결표 키의 단추)을 누르면 다음 단계. until이 참이 되어도 넘어간다.
      - "wait": until이 참이 되면 다음 단계.
      - "next": 「다음」 단추. until이 있으면 참일 때만 「다음」이 켜진다(넘어가지는 않는다).
      - "value": until(값 조건)이 참이 되면 다음 단계.
    next_after_s: 「다음」과 「이 단계 건너뛰기」가 켜지기까지 단계 시작부터 기다리는 초.
    """

    key: str
    text: Callable[[], str]
    anchor: str | None = None
    kind: Kind = "next"
    dim: bool = False
    until: Callable[[], bool] | None = None
    prepare: Callable[[], None] | None = None
    buttons: Buttons = ()
    next_after_s: float = 0.0
    # ---- 계약 밖 덧붙임(기본값이면 계약 그대로) ----
    title: str = ""
    """카드 제목(굵게). 비면 줄이 없다."""
    status: Callable[[], str] | None = None
    """본문 아래 흐린 진행 줄(⑤의 세대·평균 점수). 비면 줄이 없다."""
    extra: Callable[[], Buttons] | None = None
    """100ms마다 다시 묻는 덧단추(③ 90초 뒤 「더 기다리기」, ④ 30초 뒤 「4배로 바꾸기」). 글이 바뀔 때만 단추를 다시 만든다."""
    skip_step: bool = False
    """「이 단계 건너뛰기」 단추(다음 단계로)를 둔다."""
    default: int | None = 0
    """"welcome" 종류에서 Enter가 누르는 buttons 번호(None이면 Enter를 그냥 둔다)."""
    counted: bool = True
    """말풍선 위 「i/n」에 세는 단계인가(이어서 카드·실패 카드는 세지 않는다)."""
    escape: str = "skipped"
    """이 단계에서 Esc를 누르면 끝나는 까닭(⑦ 끝 말풍선은 "done": 다 본 뒤의 Esc는 건너뛰기가 아니다)."""


def _alive(widget: QWidget | None) -> bool:
    if widget is None:
        return False
    try:
        widget.objectName()
    except RuntimeError:  # 이미 지워진 Qt 객체
        return False
    return True


def _exposed(widget: QWidget) -> bool:
    """위젯이 든 창이 화면에 나와 있는가(한 번이라도 그려질 수 있게 됐나). 창 핸들이 없으면 거짓."""
    try:
        handle = widget.window().windowHandle()
    except RuntimeError:
        return False
    return handle is not None and handle.isExposed()


def shown(widget: QWidget | None) -> bool:
    """위젯이 화면에 실제로 보이는가(숨긴 탭·스크롤 밖이면 거짓)."""
    if not _alive(widget):
        return False
    assert widget is not None
    return widget.isVisible() and not widget.visibleRegion().isEmpty()


def scroll_into_view(widget: QWidget | None, margin: int = 20) -> None:
    """위젯을 감싼 스크롤 영역이 있으면 그 위젯이 보이게 스크롤한다."""
    if not _alive(widget):
        return
    assert widget is not None
    parent = widget.parentWidget()
    while parent is not None and not isinstance(parent, QScrollArea):
        parent = parent.parentWidget()
    if isinstance(parent, QScrollArea):
        parent.ensureWidgetVisible(widget, margin, margin)


def open_help_page(page: str, heading: str | None = None, parent: QWidget | None = None) -> bool:
    """도움말 창을 연다(help/window.open_help, 설계 7.3). 열었으면 참."""
    from evolab.app.help.window import open_help

    open_help(page, heading, parent)
    return True


class CoachOverlay(QWidget):
    """창 위 코치 오버레이 (설계 4.2, 7.3)."""

    finished = Signal(str)
    """"done" | "skipped" | "later" | "failed" """
    step_changed = Signal(str)
    target_clicked = Signal(str)
    """click 단계의 대상이 눌렸다(그 단계 키). 다음 단계로 넘어가기 전에 알린다."""

    def __init__(self, window: QWidget, steps: Sequence[CoachStep]) -> None:
        super().__init__(window)
        self.setObjectName("coachOverlay")
        self.host = window
        self.steps: list[CoachStep] = list(steps)
        self.index = -1
        self.result: str | None = None
        self.clock: Callable[[], float] = CLOCK
        self.log: list[tuple[float, str]] = []
        """(시각, 단계 키 또는 "finish:<까닭>") 기록."""
        self.clicks = 0
        """말풍선 단추와 click 단계 대상을 누른 수."""
        self.hole = QRect()
        self.fixed_target: QWidget | None = None
        """point(): 연결표 키 대신 이 위젯을 짚는다."""
        self.point_mode = False
        self.paint_ms: float = 0.0
        self._started = False
        self._step_t0 = 0.0
        self._misses = 0
        self._lost = False
        self._lost_screen: str | None = None
        self._clicked_target: QAbstractButton | None = None
        self._extra_labels: tuple[str, ...] = ()
        self._mask_key: tuple[Any, ...] | None = None
        self._next_btn: QPushButton | None = None
        self._skip_step_btn: QPushButton | None = None
        self._default_btn: QPushButton | None = None
        self._buttons: list[QPushButton] = []
        self._app_filter_pending = False
        """창이 그려지기 전에 시작해 앱 전체 필터를 첫 그리기 뒤로 미뤄 둔 중(start, paintEvent)."""

        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

        self.bubble = QFrame(self)
        self.bubble.setObjectName("coachBubble")
        self.bubble.setFixedWidth(BUBBLE_WIDTH)
        self.bubble.setStyleSheet(bubble_style_sheet())
        font = QApplication.font()
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() + 1)
        elif font.pixelSize() > 0:
            font.setPixelSize(font.pixelSize() + 1)
        self.bubble.setFont(font)
        lay = QVBoxLayout(self.bubble)
        lay.setContentsMargins(BUBBLE_PAD_X, BUBBLE_PAD_Y, BUBBLE_PAD_X, BUBBLE_PAD_Y)  # 6.2.0 간격 표(SPACE): 16·12, 줄 사이 8
        lay.setSpacing(BUBBLE_ROW_GAP)
        self.counter = QLabel()
        self.counter.setObjectName("coachCounter")
        self.title_label = QLabel()
        self.title_label.setObjectName("coachTitle")
        self.title_label.setWordWrap(True)
        bold = self.title_label.font()
        bold.setBold(True)
        self.title_label.setFont(bold)
        self._title_font = QFont(bold)
        self._card_title_font = QFont(bold)  # 환영 카드의 제목: +2pt(6.2.0 U33, style.font의 "title"과 같은 규칙)
        if bold.pointSizeF() > 0:
            self._card_title_font.setPointSizeF(bold.pointSizeF() + 2)
        elif bold.pixelSize() > 0:
            self._card_title_font.setPixelSize(bold.pixelSize() + 2)
        self.text_label = QLabel()
        self.text_label.setObjectName("coachText")
        self.text_label.setWordWrap(True)
        self.status_label = QLabel()
        self.status_label.setObjectName("coachStatus")
        self.status_label.setWordWrap(True)
        for label in (self.counter, self.title_label, self.text_label, self.status_label):
            label.setTextFormat(Qt.TextFormat.PlainText)
            lay.addWidget(label)
        self.button_row = QHBoxLayout()
        self.button_row.setSpacing(BUBBLE_ROW_GAP)
        lay.addLayout(self.button_row)

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)
        self.hide()

    # ------------------------------------------------------------------
    # 단계
    # ------------------------------------------------------------------
    @property
    def current(self) -> CoachStep | None:
        if 0 <= self.index < len(self.steps):
            return self.steps[self.index]
        return None

    @property
    def running(self) -> bool:
        return self._started and self.result is None

    def now(self) -> float:
        return self.clock()

    def step_elapsed(self) -> float:
        """지금 단계가 시작된 뒤 지난 초."""
        return self.now() - self._step_t0

    def start(self, at: str | None = None) -> None:
        """오버레이를 띄우고 at 단계(없으면 첫 단계)부터 시작한다."""
        if self._started:
            return
        self._started = True
        self.setGeometry(self.host.rect())
        self.show()
        self.raise_()
        self.host.installEventFilter(self)
        if _exposed(self.host):
            self._install_app_filter()
        else:
            # 창이 아직 한 번도 그려지지 않았다(run_gui가 창을 띄운 직후의 환영 카드): 키를 받는 앱 전체 필터는 이 오버레이가
            # 처음 그려진 다음 차례에 단다. 그러지 않으면 창의 첫 그리기 이벤트가 모두 파이썬 필터를 거친다(M3: 창 뜨는 시간).
            self._app_filter_pending = True
            QTimer.singleShot(APP_FILTER_FALLBACK_MS, self, self._install_pending_filter)
        self._timer.start()
        self.go(self.position(at) if at is not None else 0)

    def _install_pending_filter(self) -> None:
        if self._app_filter_pending:
            self._install_app_filter()

    def _install_app_filter(self) -> None:
        """Esc·Enter와 짚기 닫기 클릭을 받는 앱 전체 필터를 단다(안내가 끝났으면 달지 않는다)."""
        self._app_filter_pending = False
        if self.result is not None:
            return
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def position(self, key: str) -> int:
        for i, step in enumerate(self.steps):
            if step.key == key:
                return i
        raise KeyError(key)

    def go_to(self, key: str) -> None:
        self.go(self.position(key))

    def next(self) -> None:
        self.go(self.index + 1)

    def skip(self) -> None:
        self.finish("skipped")

    def go(self, i: int) -> None:
        if self.result is not None:
            return
        self._leave_step()
        if i >= len(self.steps):
            self.finish("done")
            return
        self.index = max(0, i)
        step = self.steps[self.index]
        self._step_t0 = self.now()
        self._misses = 0
        self._lost = False
        self._lost_screen = None
        self._mask_key = None
        self.log.append((self._step_t0, step.key))
        if step.prepare is not None:
            step.prepare()
            if self.result is not None or self.current is not step:  # 준비가 안내를 끝내거나 다른 단계로 보냄
                return
        self._connect_target()
        self._build_buttons()
        self._refresh(force=True)
        self.raise_()
        self.update()
        self.step_changed.emit(step.key)
        if self.result is None and self.current is step:
            QTimer.singleShot(0, self, self._poll)  # 조건이 이미 참이면 바로 넘어간다(다음 이벤트 루프 차례에)

    def finish(self, how: str) -> None:
        """안내를 끝낸다(오버레이를 숨기고 필터를 뗀다). 두 번째부터는 아무것도 하지 않는다."""
        if self.result is not None:
            return
        self.result = how
        self._app_filter_pending = False
        self._leave_step()
        self._timer.stop()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        try:
            self.host.removeEventFilter(self)
        except RuntimeError:
            pass
        self.log.append((self.now(), f"finish:{how}"))
        self.hide()
        self.finished.emit(how)

    def button(self, label: str) -> QPushButton | None:
        """말풍선에 있는 그 글의 단추(자동 모드·테스트가 누른다)."""
        for b in self._buttons:
            if _alive(b) and b.text() == label:
                return b
        return None

    def press(self, label: str) -> bool:
        b = self.button(label)
        if b is None or not b.isEnabled():
            return False
        b.click()
        return True

    def buttons(self) -> list[str]:
        """지금 말풍선 단추의 글(왼쪽부터)."""
        return [b.text() for b in self._buttons if _alive(b)]

    # ------------------------------------------------------------------
    # 대상
    # ------------------------------------------------------------------
    def target(self) -> QWidget | None:
        if self.fixed_target is not None:
            return self.fixed_target if _alive(self.fixed_target) else None
        step = self.current
        if step is None or step.anchor is None:
            return None
        return anchors.resolve(self.host, step.anchor)

    def _connect_target(self) -> None:
        step = self.current
        if step is None or step.kind != "click" or self._clicked_target is not None:
            return
        target = self.target()
        if isinstance(target, QAbstractButton):
            target.clicked.connect(self._on_target_clicked)
            self._clicked_target = target

    def _leave_step(self) -> None:
        target, self._clicked_target = self._clicked_target, None
        if target is not None and _alive(target):
            try:
                target.clicked.disconnect(self._on_target_clicked)
            except (RuntimeError, TypeError):
                pass

    def _on_target_clicked(self, *_args: Any) -> None:
        step = self.current
        if step is None or step.kind != "click" or self.result is not None:
            return
        self.clicks += 1
        self.target_clicked.emit(step.key)
        if self.result is None and self.current is step:
            self.next()

    # ------------------------------------------------------------------
    # 말풍선
    # ------------------------------------------------------------------
    def _clear_buttons(self) -> None:
        while self.button_row.count():
            item = self.button_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)  # 레이아웃에서 빼고 숨긴 뒤 지운다(지운 단추가 잠깐 보이는 결함을 막음)
                w.deleteLater()
        self._buttons = []
        self._next_btn = self._skip_step_btn = self._default_btn = None

    def _add_button(self, label: str, callback: Callable[[], None]) -> QPushButton:
        b = QPushButton(label, self.bubble)
        b.setObjectName("coachButton")
        b.setAutoDefault(False)
        b.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        b.clicked.connect(lambda _checked=False, cb=callback: self._pressed(cb))
        self.button_row.addWidget(b)
        b.show()  # 보이는 말풍선의 레이아웃에 넣은 위젯은 다음 이벤트 루프 차례에야 보인다: 바로 보이게(Enter·자동 모드)
        self._buttons.append(b)
        return b

    def _pressed(self, callback: Callable[[], None]) -> None:
        if self.result is not None:
            return
        self.clicks += 1
        callback()

    def _build_buttons(self) -> None:
        self._clear_buttons()
        step = self.current
        if step is None:
            return
        if step.kind == "next" and not self.point_mode:
            self._next_btn = self._add_button(S.ONB_NEXT, self.next)
            self._default_btn = self._next_btn
        for i, (label, callback) in enumerate(step.buttons):
            b = self._add_button(label, callback)
            if step.kind == "welcome" and step.default == i:
                self._default_btn = b
        if step.skip_step:
            self._skip_step_btn = self._add_button(S.ONB_SKIP_STEP, self.next)
        extras = tuple(step.extra()) if step.extra is not None else ()
        self._extra_labels = tuple(label for label, _cb in extras)
        for label, callback in extras:
            self._add_button(label, callback)
        if self._lost and self._lost_screen is not None:
            self._add_button(S.ONB_BACK_TO_LAB, self._back_to_screen)
        self.button_row.addStretch(1)
        if step.kind != "welcome" and not self.point_mode:
            self._add_button(S.ONB_SKIP, self.skip)
        if self._default_btn is not None:  # 기본 단추(Enter)만 호박색 채움(6.2.0 U33). 스타일시트는 말풍선에 한 번 걸려 있다
            self._default_btn.setProperty(DEFAULT_PROPERTY, True)
            self._default_btn.style().unpolish(self._default_btn)
            self._default_btn.style().polish(self._default_btn)
        self._enable_buttons()

    def _enable_buttons(self) -> None:
        step = self.current
        if step is None:
            return
        ready = self.step_elapsed() >= step.next_after_s
        if self._next_btn is not None:
            self._next_btn.setEnabled(ready and (step.until is None or bool(step.until())))
        if self._skip_step_btn is not None:
            self._skip_step_btn.setEnabled(ready)

    def _fit_bubble(self) -> None:
        """말풍선 높이를 고정 너비(360px)에서 줄바꿈한 글에 맞춘다(adjustSize는 이름표가 바라는 너비로 높이를 재어 글이 잘린다)."""
        lay = self.bubble.layout()
        lay.activate()
        height = lay.totalHeightForWidth(BUBBLE_WIDTH) if lay.hasHeightForWidth() else lay.totalSizeHint().height()
        self.bubble.resize(BUBBLE_WIDTH, max(height, lay.totalMinimumSize().height()))

    def _counter_text(self, step: CoachStep) -> str:
        if self.point_mode or not step.counted or is_card(step):
            return ""  # 환영 카드는 단계 수(「1/7」)를 숨긴다(6.2.0 U33)
        counted = [s for s in self.steps if s.counted]
        if step not in counted:
            return ""
        return S.ONB_PROGRESS.format(i=counted.index(step) + 1, n=len(counted))

    def _refresh(self, force: bool = False) -> None:
        """글·진행 줄·덧단추·단추 켜기를 다시 읽는다. 바뀐 것이 있으면 말풍선 크기와 자리를 다시 정한다."""
        step = self.current
        if step is None:
            return
        changed = force
        if force:
            font = self._card_title_font if is_card(step) else self._title_font
            if self.title_label.font() != font:
                self.title_label.setFont(font)
        for label, value in ((self.counter, self._counter_text(step)), (self.title_label, step.title),
                             (self.text_label, step.text()),
                             (self.status_label, step.status() if step.status is not None else "")):
            if label.text() != value or force:
                label.setText(value)
                label.setVisible(bool(value))
                changed = True
        if changed:
            self.bubble.setAccessibleName("\n".join(x for x in (step.title, self.text_label.text(),
                                                                 self.status_label.text()) if x))
        if step.extra is not None:
            labels = tuple(label for label, _cb in step.extra())
            if labels != self._extra_labels:
                self._build_buttons()
                changed = True
        self._enable_buttons()
        if changed:
            self._fit_bubble()
            self.relayout()

    # ------------------------------------------------------------------
    # 100ms 확인
    # ------------------------------------------------------------------
    def _poll(self) -> None:
        step = self.current
        if step is None or self.result is not None:
            return
        if step.until is not None and step.kind in ("wait", "value", "click") and step.until():
            self.next()
            return
        if self.result is not None or self.current is not step:
            return
        self._check_target(step)
        self._connect_target()
        self._refresh()
        self.relayout()

    def _check_target(self, step: CoachStep) -> None:
        """대상이 있는데 안 보이면 준비를 다시 하고, 세 번 연속 안 보이면 가운데로(탭을 떠났으면 돌아가기 단추)."""
        target = self.target()
        if target is None or shown(target):
            if self._lost:
                self._lost = False
                self._lost_screen = None
                self._build_buttons()
            self._misses = 0
            return
        if step.prepare is not None and self._misses < MISSES_BEFORE_CENTER:
            step.prepare()
            if shown(target):
                self._misses = 0
                return
        self._misses += 1
        if self._misses >= MISSES_BEFORE_CENTER and not self._lost:
            self._lost = True
            self._lost_screen = self._left_screen(step)
            self._build_buttons()
            self._fit_bubble()

    def _left_screen(self, step: CoachStep) -> str | None:
        """대상의 탭 화면을 떠나 안 보이는 것이면 그 화면 이름(지금은 「실험실 탭으로 돌아가기」만 문구가 있다)."""
        anchor = anchors.ANCHORS.get(step.anchor or "")
        if anchor is None or anchor.screen != "lab":
            return None
        tabs = getattr(self.host, "tabs", None)
        page = getattr(self.host, anchor.screen, None)
        if tabs is None or page is None or tabs.currentWidget() is page:
            return None
        return anchor.screen

    def _back_to_screen(self) -> None:
        step = self.current
        screen = self._lost_screen
        if screen is not None:
            anchors.show_screen(self.host, screen)  # type: ignore[arg-type]
        if step is not None and step.prepare is not None:
            step.prepare()
        self._misses = 0
        self._lost = False
        self._lost_screen = None
        self._build_buttons()
        self._fit_bubble()
        self.relayout()

    # ------------------------------------------------------------------
    # 배치·그리기
    # ------------------------------------------------------------------
    def _hole_for(self, target: QWidget | None) -> QRect:
        if target is None or self._lost or not shown(target) or \
                QWidget.window(target) is not QWidget.window(self.host):
            return QRect()
        r = target.visibleRegion().boundingRect()
        top_left = target.mapTo(self.host, r.topLeft()) if self.host.isAncestorOf(target) else \
            self.mapFromGlobal(target.mapToGlobal(r.topLeft()))
        hole = QRect(top_left, r.size()).adjusted(-HOLE_MARGIN, -HOLE_MARGIN, HOLE_MARGIN, HOLE_MARGIN)
        return hole.intersected(self.rect())

    def _place(self, hole: QRect) -> QPoint:
        bw, bh = self.bubble.width(), self.bubble.height()
        width, height = self.width(), self.height()

        def clamp(v: int, lo: int, hi: int) -> int:
            return max(lo, min(v, max(lo, hi)))

        if hole.isNull():
            return QPoint(clamp((width - bw) // 2, EDGE, width - bw - EDGE),
                          clamp((height - bh) // 2, EDGE, height - bh - EDGE))
        cx = clamp(hole.center().x() - bw // 2, EDGE, width - bw - EDGE)
        cy = clamp(hole.center().y() - bh // 2, EDGE, height - bh - EDGE)
        # 아래 → 위 → 왼쪽 → 오른쪽 차례로 들어가는 자리(설계 4.2)
        for x, y in ((cx, hole.bottom() + BUBBLE_GAP), (cx, hole.top() - BUBBLE_GAP - bh),
                     (hole.left() - BUBBLE_GAP - bw, cy), (hole.right() + BUBBLE_GAP, cy)):
            if EDGE <= x and x + bw <= width - EDGE and EDGE <= y and y + bh <= height - EDGE:
                return QPoint(x, y)
        # 큰 대상(관전 캔버스 등): 대상 안쪽 위
        return QPoint(cx, clamp(hole.top() + BUBBLE_GAP, EDGE, height - bh - EDGE))

    def relayout(self) -> None:
        """구멍·말풍선 자리·마스크를 지금 대상 위치로 다시 정한다(바뀐 것이 있을 때만 마스크를 바꾸고 다시 그린다)."""
        step = self.current
        if step is None or self.result is not None:
            return
        if self.geometry() != self.host.rect():
            self.setGeometry(self.host.rect())
        hole = self._hole_for(self.target())
        self.bubble.move(self._place(hole))
        key = (hole.getRect(), self.bubble.geometry().getRect(), step.dim, self.size().toTuple())
        if key == self._mask_key:
            return
        self._mask_key = key
        self.hole = hole
        if step.dim:
            region = QRegion(self.rect())
            if not hole.isNull():
                region = region.subtracted(QRegion(hole.adjusted(RING_PAD, RING_PAD, -RING_PAD, -RING_PAD)))
        else:
            region = QRegion(self.bubble.geometry())
            if not hole.isNull():
                band = QRegion(hole.adjusted(-RING_PAD, -RING_PAD, RING_PAD, RING_PAD)).subtracted(
                    QRegion(hole.adjusted(RING_PAD, RING_PAD, -RING_PAD, -RING_PAD)))
                region = region.united(band)
        self.setMask(region)
        self.update()

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        step = self.current
        if step is None:
            return
        t0 = time.perf_counter()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if step.dim:
            path = QPainterPath()
            path.addRect(self.rect())
            if not self.hole.isNull():
                hole_path = QPainterPath()
                hole_path.addRoundedRect(self.hole, RING_RADIUS, RING_RADIUS)
                path = path.subtracted(hole_path)
            p.fillPath(path, QColor(0, 0, 0, DIM_ALPHA))
        if not self.hole.isNull():
            p.setPen(QPen(QColor(RING_COLOR), RING_WIDTH))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(self.hole, RING_RADIUS, RING_RADIUS)
        p.end()
        self.paint_ms = (time.perf_counter() - t0) * 1000.0
        if self._app_filter_pending:  # 첫 그리기가 끝난 다음 이벤트 루프 차례에 앱 전체 필터(start)
            QTimer.singleShot(0, self, self._install_pending_filter)

    # 어두운 곳의 마우스는 먹는다(부모인 주 창으로도 보내지 않는다)
    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        event.accept()

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        event.accept()

    def mouseDoubleClickEvent(self, event: Any) -> None:  # noqa: N802
        event.accept()

    def wheelEvent(self, event: Any) -> None:  # noqa: N802
        event.accept()

    def contextMenuEvent(self, event: Any) -> None:  # noqa: N802
        event.accept()

    # ------------------------------------------------------------------
    # 이벤트 필터: 창 크기, Esc·Enter, 짚기 닫기
    # ------------------------------------------------------------------
    def _in_host(self, obj: QObject) -> bool:
        if not isinstance(obj, QWidget):
            return False
        try:
            return QWidget.window(obj) is QWidget.window(self.host)  # 속성 window를 가린 위젯도 있다
        except RuntimeError:
            return False

    def _takes_key(self, obj: QObject) -> bool:
        """이 키(Esc·Enter)를 안내가 받는가. 어두운 단계와 짚기는 창 안 어디서든 받는다. 밝은 단계는 화면을 그대로 쓰므로 키를
        처음 받은 위젯(초점 위젯, 부모로 올라오는 중이면 그 초점 위젯)이 말풍선 안이면 받고, 글·숫자 칸·목록이거나 실험실 밖 탭이면
        그 위젯에 둔다(에디터 캔버스의 Esc, 숫자 칸의 Enter 등)."""
        step = self.current
        if step is None or step.dim or self.point_mode:
            return True
        origin: QObject = obj
        focus = QApplication.focusWidget()
        if focus is not None and isinstance(obj, QWidget) and (obj is focus or obj.isAncestorOf(focus)):
            origin = focus  # 초점 위젯이 쓰지 않아 부모로 올라온 키: 처음 받은 위젯으로 판단한다
        if not isinstance(origin, QWidget):
            return True
        if origin is self.bubble or self.bubble.isAncestorOf(origin):
            return True
        if isinstance(origin, TYPING_WIDGETS):
            return False
        return anchors.screen_of(origin) in KEY_SCREENS

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        kind = event.type()
        if kind == _KEY_PRESS:
            if self.result is None and self.isVisible() and self._in_host(obj) and self._takes_key(obj):
                return self._key(event)
            return False
        if kind == _MOUSE_PRESS:
            if self.point_mode and self.result is None and isinstance(obj, QWidget) and \
                    obj is not self.bubble and not self.bubble.isAncestorOf(obj):
                QTimer.singleShot(0, self, lambda: self.finish("done"))
            return False
        if obj is self.host and kind in _HOST_EVENTS and self.result is None:
            self.relayout()
        return False

    def _key(self, event: Any) -> bool:
        key = event.key()
        if key == Qt.Key.Key_Escape:
            step = self.current
            if self.point_mode:
                self.finish("done")
            elif step is not None and step.escape != "skipped":
                self.finish(step.escape)
            else:
                self.skip()
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            b = self._default_btn
            if b is not None and _alive(b) and b.isEnabled():
                b.click()
                return True
        return False


# ---------------------------------------------------------------------------
# 짚기
# ---------------------------------------------------------------------------
_POINTS: list[weakref.ReferenceType[CoachOverlay]] = []


def _close_points() -> None:
    for ref in _POINTS:
        ov = ref()
        if ov is not None and _alive(ov) and ov.running:
            ov.finish("done")
    _POINTS.clear()


def point(window: QWidget | None, widget: QWidget | None, text: str,
          more: tuple[str, str | None] | None = None) -> CoachOverlay:
    """위젯 하나를 어둡게 하지 않고 짚는다: 주황 띠 + 말풍선(글, 있으면 「도움말에서 더 보기」). 4초 또는 클릭으로 닫힌다.

    more = (도움말 쪽, 제목 또는 None). 위젯이 window 밖(다른 창)이면 그 위젯의 창 위에 띄운다.
    """
    host: QWidget | None = window
    if _alive(widget):
        assert widget is not None
        if host is None or not _alive(host) or QWidget.window(widget) is not QWidget.window(host):
            host = QWidget.window(widget)  # 속성 window를 가린 위젯도 있다(궤적 밀도 타일)
    if host is None:
        raise ValueError("point() needs a window or a widget")
    _close_points()
    buttons: Buttons = ()
    holder: list[CoachOverlay] = []
    if more is not None:
        page, heading = more

        def open_more() -> None:
            open_help_page(page, heading, host)
            if holder:
                holder[0].finish("done")

        buttons = ((S.WHATS_THIS_MORE, open_more),)
    step = CoachStep(key="point", text=lambda: text, kind="welcome", buttons=buttons, default=None, counted=False)
    overlay = CoachOverlay(host, [step])
    holder.append(overlay)
    overlay.point_mode = True
    overlay.fixed_target = widget if _alive(widget) else None
    overlay.start()
    QTimer.singleShot(int(POINT_SECONDS * 1000), overlay, lambda: overlay.finish("done"))
    _POINTS.append(weakref.ref(overlay))
    return overlay


__all__ = ["CLOCK", "CoachOverlay", "CoachStep", "open_help_page", "point", "scroll_into_view", "shown"]

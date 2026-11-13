"""빈 상태 안내: 화면이 비어 있을 때 그 자리에 무엇을 하면 되는지 띄운다 (v3 설계 D17·D24, 4.3, 5.2).

| 자리 | 보일 때 | 글 | 단추 |
| --- | --- | --- | --- |
| 실험실 캔버스 | 학습 프로세스 없음(`not lab.client.active`), 실행을 열지 않음(`lab.run_dir is None`) | HINT_LAB_IDLE_TITLE, HINT_LAB_IDLE_ANY | 「▶ 학습 시작」(= 패널의 `btn_start`를 누름), HINT_LAB_TOUR(안내 상태가 done이면 숨김) |
| 대시보드 | 학습 중이거나 실행을 열었고 통계 줄이 0 / 1 | HINT_DASH_EMPTY / HINT_DASH_ONE | — (마우스는 아래로) |
| 에디터 캔버스 | 문서에 그려지는 부품이 없음(`anchors.editor_is_empty`) | HINT_EDITOR_EMPTY | 「내장 예제에서 시작」 메뉴를 펼침 |

- `attach_hint`: 대상의 자식 카드(`HintFrame`, QFrame)와 단추 줄(대상의 다른 자식). 카드는 대상 가운데에 놓고 대상 크기가 바뀌면
  다시 놓는다. 조건과 글은 300ms마다(대상이 보일 때만)와 단추를 누른 뒤 다시 본다. passthrough=True면 카드는 마우스를 받지 않아
  카드 위 클릭·끌기·휠도 아래 대상으로 간다. 단추 줄은 단추 둘레만 차지하고 늘 클릭을 받는다. 카드 사각형 밖은 원래 대상 그대로.
- `install_hints(window)`: 주 창에 셋을 단다(여러 번 불러도 한 번). 화면 파일은 고치지 않는다: 위젯은 주 창 속성
  (`lab.canvas`, `lab.dashboard`, `editor.canvas`)으로 찾는다. 안내(tour) 모듈은 「따라 하기」를 누를 때 불러온다(D62).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Final

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from evolab.app import strings as S

CHECK_MS: Final = 300
"""조건을 다시 보는 간격(설계 4.3)."""
MAX_WIDTH: Final = 440
"""카드 최대 너비(px). 대상이 좁으면 대상 너비 − 2 × MARGIN."""
MIN_WIDTH: Final = 160
"""이보다 좁은 대상에는 카드를 띄우지 않는다."""
MARGIN: Final = 12
"""카드와 대상 가장자리 사이(px)."""
TOUR_STATE_KEY: Final = "onboarding/state"
"""처음 실행 안내 상태 키(= tour.STATE_KEY, 테스트가 확인). 이 모듈을 띄울 때 안내 모듈을 불러오지 않으려고 둔다."""
HINT_NAME: Final = "evolabHint"
BAR_NAME: Final = "evolabHintButtons"

STYLE: Final = (
    f"#{HINT_NAME} {{ background: rgba(24, 28, 36, 228); border: 1px solid rgba(255, 255, 255, 70);"
    " border-radius: 8px; }"
    f" #{HINT_NAME} QLabel {{ color: #f2f4f8; background: transparent; }}"
)
"""밝은·어두운 테마 모두에서 읽히는 짙은 카드(실험실 캔버스는 어둡고 에디터 캔버스는 밝다)."""

ButtonSpec = tuple[str, Callable[[], Any]] | tuple[str, Callable[[], Any], Callable[[], bool | None]]
"""(글, 누르면 부를 함수[, 상태]). 상태 함수가 True면 켬, False면 끔(회색), None이면 숨김. 없으면 늘 켬."""


class HintFrame(QFrame):
    """빈 상태 카드 하나(대상의 자식). 단추는 대상의 다른 자식인 단추 줄(`bar`)에 있다."""

    def __init__(self, target: QWidget, text: Callable[[], str], visible: Callable[[], bool],
                 buttons: Sequence[ButtonSpec] = (), passthrough: bool = True,
                 title: Callable[[], str] | None = None) -> None:
        super().__init__(target)
        self.target = target
        self._text = text
        self._visible = visible
        self._title = title
        self.passthrough = passthrough
        self.setObjectName(HINT_NAME)
        self.setStyleSheet(STYLE)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, passthrough)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(6)
        self.title_label = QLabel()
        font = self.title_label.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() + 2)
        self.title_label.setFont(font)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setWordWrap(True)
        self.text_label = QLabel()
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setWordWrap(True)
        for label in (self.title_label, self.text_label):
            label.setTextFormat(Qt.TextFormat.PlainText)
        lay.addWidget(self.title_label)
        lay.addWidget(self.text_label)
        self.title_label.hide()

        # 단추 줄: 카드가 마우스를 받지 않아도 단추는 눌려야 하므로 카드의 자식이 아니라 대상의 자식으로 카드 위에 겹쳐 놓는다
        self.bar = QWidget(target)
        self.bar.setObjectName(BAR_NAME)
        bl = QHBoxLayout(self.bar)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)
        self.buttons: list[QPushButton] = []
        self._states: list[Callable[[], bool | None] | None] = []
        for spec in buttons:
            btn = QPushButton(spec[0], self.bar)
            slot = spec[1]
            btn.clicked.connect(lambda _checked=False, f=slot: self._press(f))
            bl.addWidget(btn)
            self.buttons.append(btn)
            self._states.append(spec[2] if len(spec) > 2 else None)  # type: ignore[misc]
        self._bar_space = QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        if self.buttons:
            lay.addSpacing(4)
            lay.addItem(self._bar_space)
        self.bar.hide()
        self.hide()
        self._placed: tuple[Any, ...] | None = None
        """마지막으로 배치한 (대상 너비, 높이, 제목, 글, 단추 상태)."""

        target.installEventFilter(self)
        self._timer = QTimer(self)
        self._timer.setInterval(CHECK_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    # ------------------------------------------------------------------
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self.target and event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.refresh()
        return False

    def _press(self, slot: Callable[[], Any]) -> None:
        slot()
        self.refresh()

    def text(self) -> str:
        return self.text_label.text()

    def refresh(self) -> None:
        """조건을 보고 카드를 띄우거나 숨긴다. 대상이 보이지 않으면(다른 탭) 그대로 둔다(보이게 될 때 Show로 다시 본다)."""
        if not self.target.isVisible() and not self.target.isHidden():
            return  # 조상이 숨음: 지금 상태를 그대로 둔다
        try:
            show = bool(self._visible())
        except RuntimeError:  # 창을 닫는 중(위젯이 먼저 지워짐)
            show = False
        if not show:
            self.hide()
            self.bar.hide()
            return
        title = self._title() if self._title is not None else ""
        text = self._text()
        if self.title_label.text() != title:
            self.title_label.setText(title)
        self.title_label.setVisible(bool(title))
        if self.text_label.text() != text:
            self.text_label.setText(text)
        states = tuple(True if state is None else state() for state in self._states)
        for btn, on in zip(self.buttons, states, strict=True):
            btn.setVisible(on is not None)
            btn.setEnabled(bool(on))
        self.bar.setVisible(any(on is not None for on in states))
        key = (self.target.width(), self.target.height(), title, text, states)
        if key != self._placed or self.isHidden():  # 바뀐 것이 없으면 다시 배치하지 않는다(300ms마다 배치를 흔들지 않게)
            if not self._place():
                self._placed = None
                self.hide()
                self.bar.hide()
                return
            self._placed = key
        self.show()
        self.raise_()
        if not self.bar.isHidden():
            self.bar.raise_()

    def _place(self) -> bool:
        """대상 가운데에 놓는다. 대상이 너무 좁으면 False."""
        tw, th = self.target.width(), self.target.height()
        room = tw - 2 * MARGIN
        if room < MIN_WIDTH:
            return False
        self.bar.layout().invalidate()  # 숨기고 보인 단추가 크기에 들게
        bar_hint = self.bar.sizeHint() if not self.bar.isHidden() else None
        self._bar_space.changeSize(0, bar_hint.height() if bar_hint is not None else 0,
                                   QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        lay = self.layout()
        lay.invalidate()
        m = lay.contentsMargins()
        fm = self.text_label.fontMetrics()
        longest = max((fm.horizontalAdvance(line) for line in self.text_label.text().splitlines()), default=0)
        tfm = self.title_label.fontMetrics()
        longest = max(longest, tfm.horizontalAdvance(self.title_label.text()) if self._title else 0)
        natural = longest + m.left() + m.right() + 8
        if bar_hint is not None:
            natural = max(natural, bar_hint.width() + m.left() + m.right())
        w = max(MIN_WIDTH, min(MAX_WIDTH, room, natural))
        h = lay.totalHeightForWidth(w) if lay.hasHeightForWidth() else lay.totalSizeHint().height()
        h = max(h, lay.totalMinimumSize().height())
        x = (tw - w) // 2
        y = max(MARGIN // 2, (th - h) // 2)
        self.setGeometry(x, y, w, h)
        if bar_hint is not None:
            bw = min(bar_hint.width(), w - m.left() - m.right())
            self.bar.setGeometry(x + (w - bw) // 2, y + h - m.bottom() - bar_hint.height(), bw, bar_hint.height())
        return True


def attach_hint(widget: QWidget, text: Callable[[], str], visible: Callable[[], bool],
                buttons: Sequence[ButtonSpec] = (), passthrough: bool = True, *,
                title: Callable[[], str] | None = None) -> QWidget:
    """widget 위에 빈 상태 카드를 단다(설계 7.3). visible()이 참일 때만 보이고 글은 text()(제목은 title())."""
    return HintFrame(widget, text, visible, buttons, passthrough, title)


def hint_of(widget: QWidget) -> HintFrame | None:
    """widget에 단 빈 상태 카드(없으면 None)."""
    for child in widget.children():
        if isinstance(child, HintFrame):
            return child
    return None


# ---------------------------------------------------------------------------
# 주 창의 빈 상태 셋
# ---------------------------------------------------------------------------
def _population(lab: Any) -> int:
    """「{pop}개가 동시에 연습」: CMA-ES·GA는 개체 수, 담금질은 세대마다 해 보는 이웃 수."""
    return int(lab.sa_neighbors.value() if lab.method.currentData() == "sa" else lab.popsize.value())


def lab_idle_text(lab: Any) -> str:
    """「▶ 학습 시작을 누르면 개체 32개가 「점프 게임」에서…」: 환경 이름을 셀 수 있는 낱말로 쓰지 않는다(모든 환경에 맞게)."""
    if lab.spec is None:
        return S.LAB_ENV_NONE
    return S.HINT_LAB_IDLE_ANY.format(start=S.LAB_START, env=lab.spec.file.name, pop=_population(lab))


def _tour_done(lab: Any) -> bool:
    settings = getattr(lab, "settings", None)
    if settings is None:
        return False
    value = settings.value(TOUR_STATE_KEY, "")
    return str(value or "").strip() == "done"


def start_tour(window: QMainWindow) -> Any:
    """「처음이라면 따라 하기로 시작」: 처음 실행 안내(A 묶음)를 처음부터 띄운다."""
    from evolab.app.help import tour

    return tour.start_tour(window, tour.tour_settings(window))


def lab_hint(window: QMainWindow) -> HintFrame:
    lab = window.lab  # type: ignore[attr-defined]
    buttons: tuple[ButtonSpec, ...] = (
        # 패널의 진짜 단추를 누른다: 그 신호를 기다리는 곳(안내 등)이 같은 길로 안다
        (S.LAB_START, lambda: lab.btn_start.click(), lambda: lab.btn_start.isEnabled()),
        (S.HINT_LAB_TOUR, lambda: start_tour(window), lambda: None if _tour_done(lab) else True),
    )
    frame = HintFrame(lab.canvas, lambda: lab_idle_text(lab),
                      lambda: not lab.client.active and lab.run_dir is None, buttons, True,
                      lambda: S.HINT_LAB_IDLE_TITLE if lab.spec is not None else "")
    frame.buttons[0].setToolTip(S.HELP_TIPS.get("lab.start", ""))  # 패널의 그 단추와 같은 풍선(이건 뭐예요?도 같은 글)
    signals = [lab.client.started, lab.client.closed, lab.client.state_changed, lab.run_started]
    # 환경·개체 수가 바뀌면 글을 바로 고친다(300ms 확인을 기다리지 않게)
    signals += [getattr(lab, name) for name in ("env_changed",) if hasattr(lab, name)]
    signals += [lab.popsize.valueChanged, lab.sa_neighbors.valueChanged, lab.method.currentIndexChanged]
    for signal in signals:
        signal.connect(lambda *_a, f=frame: f.refresh())
    return frame


def dashboard_hint(window: QMainWindow) -> HintFrame:
    lab = window.lab  # type: ignore[attr-defined]
    frame = HintFrame(lab.dashboard, lambda: S.HINT_DASH_EMPTY if not lab.rows else S.HINT_DASH_ONE,
                      lambda: (lab.client.active or lab.run_dir is not None) and len(lab.rows) <= 1)
    for signal in (lab.client.started, lab.client.generation, lab.client.closed, lab.run_started):
        signal.connect(lambda *_a, f=frame: f.refresh())
    return frame


def editor_hint(window: QMainWindow) -> HintFrame:
    from evolab.app.help import anchors

    editor = window.editor  # type: ignore[attr-defined]
    holder: list[HintFrame] = []

    def popup() -> None:
        # 「내장 예제에서 시작」 메뉴를 그 단추 아래에 펼친다(popup: 막지 않음)
        btn = holder[0].buttons[0]
        editor.builtin_menu.popup(btn.mapToGlobal(QPoint(0, btn.height())))

    text = S.HINT_EDITOR_EMPTY.format(preset=S.ED_NEW_PRESET, shapes=S.EDC_GROUP_SHAPES, builtin=S.ED_FROM_BUILTIN)
    frame = HintFrame(editor.canvas, lambda: text, lambda: anchors.editor_is_empty(window), ((S.ED_FROM_BUILTIN, popup),))
    holder.append(frame)
    frame.buttons[0].setToolTip(S.HELP_TIPS.get("editor.builtin", ""))
    editor.doc.changed.connect(lambda *_a, f=frame: f.refresh())
    return frame


def install_hints(window: QMainWindow) -> None:
    """주 창에 빈 상태 셋(실험실 캔버스, 대시보드, 에디터 캔버스)을 단다. 이미 달았으면 아무것도 안 한다."""
    if getattr(window, "evolab_hints", None):
        return
    hints: list[HintFrame] = []
    if getattr(window, "lab", None) is not None:
        hints += [lab_hint(window), dashboard_hint(window)]
    if getattr(window, "editor", None) is not None:
        hints.append(editor_hint(window))
    window.evolab_hints = hints  # type: ignore[attr-defined]


__all__ = ["HintFrame", "attach_hint", "dashboard_hint", "editor_hint", "hint_of", "install_hints", "lab_hint",
           "lab_idle_text", "start_tour"]

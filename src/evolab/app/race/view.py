"""나란히 경주 창 (명세: 비교 > 나란히 경주, 설계 4.6·5.2·D26~D31).

- RaceView: 위 줄(코스, 다른 코스, 누구를(실행 경주만), 보기), 알림 띠(환경이 다를 때, 뺀 실행, 실패), 칸 격자(2 → 1×2,
  3 → 1×3, 4 → 2×2, 5·6 → 2×3), 결과 줄, 아래 줄(출발/일시 정지, 처음부터, 배속, 시각, 경주 영상 저장).
- 칸: 제목(색 점 + 이름, 둘째 줄) / 관전 캔버스(늘 어두운 바탕, 그 칸 개체를 따라감) / 상태 글(달리는 중 → 끝 → 순위).
- 준비: _RaceThread가 같은 환경 글은 한 번만 컴파일하고 칸마다 replay(spec, layout_for(spec), 유전자, 코스 시험)를
  돌린다(워커와 같은 Episode·순전파라 점수가 워커 점수와 비트 단위로 같다). 코스·누구를 바꾸면 다시 준비하고, 앞
  준비의 결과는 꼬리표로 버린다.
- 시계: QTimer 하나. t += 실제 dt × 배속, 모든 칸 set_time(t). 칸 캔버스의 그리기 시간을 재어 최근 30프레임 중앙값이
  12ms를 넘으면 60fps(16ms) → 30fps(33ms). 모든 칸이 끝나고 1초(실제 시간) 뒤 멈추고 순위를 채운다(점수 내림차순,
  같은 점수는 「공동 n위」). 환경이 섞이면 순위 대신 끝 글만.
- 「한 화면에 겹쳐서」: 캔버스 하나에 모든 칸을 겹쳐 그린다(1세대 vs 지금과 같은 그림). 모든 칸의 환경이 같을 때만.
- RaceDialog: RaceView를 담는 주인 있는 창(연 화면의 window()가 주인이라 주 창 뒤로 숨지 않음, 최대화 가능).
  show_race는 그 주인에 이미 있는 경주 창을 다시 쓴다(내용을 바꾸고 앞으로).
"""

from __future__ import annotations

import datetime as dt
import html
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QColor, QHideEvent, QKeyEvent, QPaintEvent, QShowEvent, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from evolab import paths
from evolab.app import strings as S
from evolab.app import style
from evolab.app.lab.canvas import SpectatorCanvas
from evolab.app.race.model import (
    COURSES,
    MAX_LANES,
    WHO_MODES,
    RaceEntry,
    common_budget,
    compile_cached,
    course_trial,
    entries_from_runs,
    lane_grid,
    next_course,
    ranks,
    same_env,
)
from evolab.engine.replay import Replay, replay
from evolab.policy.genome import layout_for
from evolab.spec.compiled import EnvFileError
from evolab.spec.editor_meta import ViewBox, view_box
from evolab.world.physics2d.episode import EpisodeBuildError, Trial

FAST_MS = 16
"""처음 프레임 간격(약 60fps)."""
SLOW_MS = 33
"""그리기가 느릴 때 프레임 간격(약 30fps)."""
SLOW_PAINT_MS = 12.0
PAINT_WINDOW = 30
"""60→30fps 판단에 쓰는 최근 프레임 수."""
HOLD_SECONDS = 1.0
"""모든 칸이 끝난 뒤 순위를 채우기까지 기다리는 실제 시간(초)."""
SPEEDS = (0.5, 1.0, 2.0, 4.0)
DEFAULT_SPEED = 1.0
VIEW_MODES = ("split", "overlay")
DEFAULT_SIZE = (1280, 800)
LANE_MIN_SIZE = (160, 120)
PAINT_LOG_MAX = 20000
WARN_MARK = "⚠ "
DOT = "●"


def entry_view(entry: RaceEntry) -> ViewBox | None:
    """칸 환경의 기본 보기(editor.view): 칸 캔버스의 처음 카메라(v2.0 설계 D25). 환경 글은 준비 스레드가 이미 컴파일해
    두었으므로(compile_cached) 다시 컴파일하지 않는다. 읽을 수 없으면 None(지금 카메라 규칙)."""
    try:
        return view_box(compile_cached(entry.env_text).file)
    except (EnvFileError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 준비 스레드
class _RaceThread(QThread):
    progress = Signal(object, int, int)
    """(꼬리표, 끝낸 칸 수, 전체 칸 수)"""
    finished_job = Signal(object, object)
    """(꼬리표, Replay 목록 또는 예외)"""

    def __init__(self, tag: Any, jobs: list[tuple[str, np.ndarray, Trial]], cancel: threading.Event,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.tag = tag
        self.jobs = jobs
        self.cancel = cancel

    def run(self) -> None:
        try:
            prepared: dict[str, tuple[Any, Any]] = {}
            out: list[Replay] = []
            for k, (text, genome, trial) in enumerate(self.jobs):
                if self.cancel.is_set():
                    return
                if text not in prepared:  # 같은 환경 글은 한 번만 컴파일
                    spec = compile_cached(text)
                    prepared[text] = (spec, layout_for(spec))
                spec, layout = prepared[text]
                out.append(replay(spec, layout, genome, trial))
                self.progress.emit(self.tag, k + 1, len(self.jobs))
        except Exception as exc:  # 결과를 화면 스레드로 넘겨 알린다
            self.finished_job.emit(self.tag, exc)
            return
        self.finished_job.emit(self.tag, out)


_LIVE_THREADS: set[_RaceThread] = set()
"""도는 중인 준비 스레드(프로그램을 끝낼 때 멈추고 기다린다: wait_threads)."""


def _start_thread(th: _RaceThread) -> None:
    _LIVE_THREADS.add(th)
    th.finished.connect(lambda t=th: _LIVE_THREADS.discard(t))  # deleteLater보다 먼저: 지워진 객체를 붙잡지 않게
    th.finished.connect(th.deleteLater)  # 주인이 프로그램이라 창을 닫아도 도는 중에 지워지지 않는다
    th.start()


def wait_threads() -> None:
    """프로그램을 끝낼 때: 도는 경주 준비를 모두 멈추라고 하고 끝날 때까지 기다린다. 준비 스레드의 주인은 프로그램
    (QApplication)이라, 프로그램이 끝날 때 아직 돌고 있으면 Qt가 프로세스를 끝내 버린다(abort). 칸 하나를 다시
    시뮬레이션하는 동안(길어야 수백 ms)만 기다린다."""
    threads = list(_LIVE_THREADS)
    for th in threads:
        th.cancel.set()
    for th in threads:
        try:
            th.wait()
        except RuntimeError:  # 이미 끝나 지워졌다
            pass
        _LIVE_THREADS.discard(th)


def shutdown_race(dialog: Any) -> None:
    """프로그램을 끝낼 때(실험실·비교 화면의 shutdown): 경주 창의 시계와 준비를 멈추고(경주 창은 주 창이 닫혀도 남아
    있어 닫힘 알림을 받지 않는다) 도는 준비 스레드가 끝날 때까지 기다린다. dialog는 None이거나 이미 지워졌어도 된다."""
    if dialog is not None:
        try:
            dialog.view.shutdown()
        except RuntimeError:  # 창이 이미 지워졌다
            pass
    wait_threads()


# ---------------------------------------------------------------------------
# 칸
class _LaneCanvas(SpectatorCanvas):
    """그리기 시간을 재는 관전 캔버스(60→30fps 규칙과 성능 측정)."""

    def __init__(self, on_paint: Callable[[float], None], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._on_paint = on_paint
        self.setMinimumSize(*LANE_MIN_SIZE)
        self.empty_text = ""  # 준비 중은 칸 아래 글줄(RACE_PREPARING)이 알린다(실험실 캔버스의 빈 화면 글은 맞지 않음)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        t0 = time.perf_counter()
        super().paintEvent(event)
        self._on_paint((time.perf_counter() - t0) * 1000.0)


class _Lane(QFrame):
    """경주 칸 하나: 제목, 캔버스, 상태 글."""

    def __init__(self, entry: RaceEntry, on_paint: Callable[[float], None], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.entry = entry
        self.replay: Replay | None = None
        self.duration = 0.0
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.title = QLabel()
        self.title.setTextFormat(Qt.TextFormat.RichText)
        self.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        text = f'<span style="color:{html.escape(entry.color)}">{DOT}</span>&nbsp;<b>{html.escape(entry.label)}</b>'
        if entry.sub:
            text += f"<br>{html.escape(entry.sub)}"
        self.title.setText(text)
        self.title.setToolTip("\n".join(t for t in (entry.label, entry.sub) if t))
        self.canvas = _LaneCanvas(on_paint)
        self.status = QLabel()
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        # U25: 순위가 정해지면 머리에 순위 칩(1위는 강조색 채움, 나머지는 테두리만), 우승 칸은 칸 색 2px 테두리
        self.rank_chip = QLabel()
        self.rank_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rank_chip.hide()
        self.winner = False
        self.setObjectName("raceLane")
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        head.addWidget(self.title, 1)
        head.addWidget(self.rank_chip, 0, Qt.AlignmentFlag.AlignTop)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 3, 4, 3)
        lay.setSpacing(2)
        lay.addLayout(head)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.status)

    def set_rank(self, rank: int | None, tie: bool = False) -> None:
        """순위 칩과 우승 테두리. None이면 지운다(준비·처음부터·환경이 섞인 경주)."""
        if rank is None:
            if self.rank_chip.isVisibleTo(self) or self.winner:
                self.rank_chip.hide()
                self.rank_chip.clear()
                self.winner = False
                self.setStyleSheet("")
            return
        ui = style.ui_for(self)
        self.rank_chip.setText((S.RACE_RANK_CHIP_TIE if tie else S.RACE_RANK_CHIP).format(rank=rank))
        if rank == 1:
            chip = (f"background: {ui.accent}; color: {ui.on_accent}; font-weight: bold; border: 1px solid {ui.accent}; "
                    "border-radius: 8px; padding: 0px 8px;")
        else:
            chip = f"color: {ui.ink2}; border: 1px solid {ui.line}; border-radius: 8px; padding: 0px 8px;"
        self.rank_chip.setStyleSheet(chip)
        self.rank_chip.show()
        self.winner = rank == 1
        color = html.escape(self.entry.color)
        self.setStyleSheet(f"QFrame#raceLane {{ border: 2px solid {color}; border-radius: 3px; }}" if self.winner else "")

    def set_replay(self, rep: Replay | None) -> None:
        self.replay = rep
        if rep is None:
            self.canvas.set_overlay([])
            self.duration = 0.0
            return
        self.canvas.set_scene(rep.geometry, None, channels=rep.channels, view=entry_view(self.entry))
        self.canvas.set_population([])
        # 이름은 칸 제목에 있으므로 캔버스 범례는 그리지 않는다(빈 이름)
        self.canvas.set_overlay([(rep.trajectory, rep.agent, QColor(self.entry.color), "")])
        self.canvas.set_follow("champion")
        self.duration = rep.trajectory.duration

    def set_status(self, text: str) -> None:
        if self.status.text() != text:
            self.status.setText(text)


# ---------------------------------------------------------------------------
# 화면
class RaceView(QWidget):
    """나란히 경주 화면(창 안의 내용)."""

    prepared = Signal()
    """모든 칸의 재생이 준비됐다(테스트·측정용)."""
    finished_race = Signal()
    """모든 칸이 끝나고 순위를 채웠다."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.entries: list[RaceEntry] = []
        self.warnings: list[str] = []
        self.runs: list[tuple[Path, str]] | None = None
        self.course = 1
        self.who = "last"
        self.mode = "split"
        self.speed = DEFAULT_SPEED
        self.t = 0.0
        self.ready = False
        self.playing = False
        self.finished = False
        self.failure: str | None = None
        self.rank_list: list[tuple[int, bool]] | None = None
        self.lanes: list[_Lane] = []
        self.frame_ms: deque[float] = deque(maxlen=PAINT_WINDOW)
        self.paint_log: list[float] = []
        """프레임마다 그린 시간(ms, 보이는 캔버스 합). 성능 측정용."""
        self._paint_accum = 0.0
        self._hold = 0.0
        self._tag = 0
        self._cancel: threading.Event | None = None
        self._last_tick = 0.0
        self._quiet = False
        self._resume_on_show = False

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(FAST_MS)
        self._timer.timeout.connect(self._tick)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)  # 준비 중 초점 자리(U23). 탭 차례에는 들지 않는다

        # -- 위 줄 --------------------------------------------------------------
        self.course_box = QComboBox()
        for n in range(1, COURSES + 1):
            self.course_box.addItem(S.RACE_COURSE_ITEM.format(n=n), n)
        self.course_box.setToolTip(S.RACE_COURSE_TIP)
        self.course_box.currentIndexChanged.connect(self._on_course_box)
        self.next_button = QPushButton(S.RACE_NEXT_COURSE)
        self.next_button.setToolTip(S.RACE_COURSE_TIP)
        self.next_button.clicked.connect(self.next_course)
        self.who_label = QLabel(S.RACE_WHO_LABEL)
        self.who_box = QComboBox()
        for w in WHO_MODES:
            self.who_box.addItem(S.RACE_WHO[w], w)
        self.who_box.currentIndexChanged.connect(self._on_who_box)
        self.view_box = QComboBox()
        for m in VIEW_MODES:
            self.view_box.addItem(S.RACE_VIEW[m], m)
        self.view_box.currentIndexChanged.connect(self._on_view_box)
        top = QHBoxLayout()
        top.addWidget(QLabel(S.RACE_COURSE_LABEL))
        top.addWidget(self.course_box)
        top.addWidget(self.next_button)
        top.addSpacing(12)
        top.addWidget(self.who_label)
        top.addWidget(self.who_box)
        top.addSpacing(12)
        top.addWidget(QLabel(S.RACE_VIEW_LABEL))
        top.addWidget(self.view_box)
        top.addStretch(1)

        # -- 알림 띠 ------------------------------------------------------------
        self.band = QLabel()
        self.band.setWordWrap(True)
        self.band.setFrameShape(QFrame.Shape.StyledPanel)
        self.band.setContentsMargins(6, 3, 6, 3)
        self.band.hide()

        # -- 칸 격자와 겹쳐 보기 --------------------------------------------------
        self.grid_widget = QWidget()
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(4)
        self.overlay_canvas = _LaneCanvas(self._on_paint)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.grid_widget)
        self.stack.addWidget(self.overlay_canvas)

        # -- 결과, 아래 줄 --------------------------------------------------------
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        self.result_label.setTextFormat(Qt.TextFormat.RichText)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.message.hide()
        self.start_button = QPushButton(S.RACE_START)
        self.start_button.clicked.connect(self.toggle)
        # U23: Enter는 「출발」(기본 단추). 다른 단추의 autoDefault는 그대로 둔다(초점을 둔 단추는 Enter로 그 단추를 누름)
        self.start_button.setDefault(True)
        self.start_button.setToolTip(S.RACE_START_TIP)
        style.set_role(self.start_button, "primary")
        self.restart_button = QPushButton(S.RACE_RESTART)
        self.restart_button.setToolTip(S.RACE_RESTART_TIP)
        self.restart_button.clicked.connect(self.restart)
        self.speed_box = QComboBox()
        for s in SPEEDS:
            self.speed_box.addItem(S.RACE_SPEED_ITEM.format(n=s), s)
        self.speed_box.setCurrentIndex(SPEEDS.index(DEFAULT_SPEED))
        self.speed_box.currentIndexChanged.connect(lambda _i: self.set_speed(float(self.speed_box.currentData())))
        self.hud_label = QLabel()
        self.gif_button = QPushButton(S.RACE_SAVE_GIF)
        self.gif_button.clicked.connect(lambda: self.save_gif())
        bottom = QHBoxLayout()
        bottom.addWidget(self.start_button)
        bottom.addWidget(self.restart_button)
        bottom.addSpacing(12)
        bottom.addWidget(QLabel(S.RACE_SPEED_LABEL))
        bottom.addWidget(self.speed_box)
        bottom.addSpacing(12)
        bottom.addWidget(self.hud_label)
        bottom.addStretch(1)
        bottom.addWidget(self.gif_button)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.addLayout(top)
        root.addWidget(self.band)
        root.addWidget(self.stack, 1)
        root.addWidget(self.result_label)
        root.addWidget(self.message)
        root.addLayout(bottom)
        self._set_who_visible(False)
        self._update_controls()

    # ------------------------------------------------------------------
    # 내용
    # ------------------------------------------------------------------
    def set_race(self, entries: Sequence[RaceEntry], *, warnings: Sequence[str] = (),
                 runs: Sequence[tuple[Path, str]] | None = None) -> None:
        """칸을 바꾸고 다시 준비한다. runs가 있으면(비교 화면) 「누구를」을 보이고 바꿀 때 칸을 다시 만든다."""
        self.runs = [(Path(p), str(c)) for p, c in runs] if runs else None
        self.who = "last"
        self._quiet = True
        try:
            self.who_box.setCurrentIndex(WHO_MODES.index("last"))
        finally:
            self._quiet = False
        self._set_who_visible(self.runs is not None)
        if self.runs is not None:
            self.who_box.setToolTip(S.RACE_WHO_TIP.format(trials=common_budget(self.runs) or 0))
        self.message.hide()
        self._set_entries(entries, warnings)

    def _set_who_visible(self, on: bool) -> None:
        self.who_label.setVisible(on)
        self.who_box.setVisible(on)

    def _set_entries(self, entries: Sequence[RaceEntry], warnings: Sequence[str]) -> None:
        entries = list(entries)
        warnings = list(warnings)
        if len(entries) > MAX_LANES:
            entries = entries[:MAX_LANES]
            warnings.append(S.RACE_TOO_MANY.format(max=MAX_LANES))
        self.entries = entries
        self.warnings = warnings
        self.failure = None
        self._build_lanes()
        overlay_ok = self.overlay_allowed()
        model = self.view_box.model()
        if isinstance(model, QStandardItemModel):
            item = model.item(VIEW_MODES.index("overlay"))
            if item is not None:
                item.setEnabled(overlay_ok)
                item.setToolTip("" if overlay_ok else S.RACE_OVERLAY_OFF_TIP)
        self.view_box.setToolTip("" if overlay_ok else S.RACE_OVERLAY_OFF_TIP)
        if not overlay_ok and self.mode == "overlay":
            self.set_mode("split")
        self._update_band()
        self.prepare()

    def overlay_allowed(self) -> bool:
        """「한 화면에 겹쳐서」를 켤 수 있는가: 칸이 있고 모든 칸의 환경이 같을 때."""
        return bool(self.entries) and same_env(self.entries)

    def _build_lanes(self) -> None:
        for lane in self.lanes:
            self.grid.removeWidget(lane)
            lane.hide()
            lane.deleteLater()
        self.lanes = []
        for r in range(self.grid.rowCount()):
            self.grid.setRowStretch(r, 0)
        for c in range(self.grid.columnCount()):
            self.grid.setColumnStretch(c, 0)
        if not self.entries:
            return
        rows, cols = lane_grid(len(self.entries))
        for k, entry in enumerate(self.entries):
            lane = _Lane(entry, self._on_paint)
            self.grid.addWidget(lane, k // cols, k % cols)
            self.lanes.append(lane)
        for r in range(rows):
            self.grid.setRowStretch(r, 1)
        for c in range(cols):
            self.grid.setColumnStretch(c, 1)

    def _update_band(self) -> None:
        lines = list(self.warnings)
        if self.entries and not same_env(self.entries):
            lines.append(S.RACE_DIFFERENT_ENV)
        if self.failure:
            lines.append(self.failure)
        self.band.setText("\n".join(WARN_MARK + line for line in lines))
        self.band.setVisible(bool(lines))

    def band_lines(self) -> list[str]:
        """알림 띠의 줄들(표시 기호 없이)."""
        text = self.band.text() if self.band.isVisibleTo(self) else ""
        return [line.removeprefix(WARN_MARK) for line in text.split("\n") if line]

    # ------------------------------------------------------------------
    # 코스, 누구를, 보기, 배속
    # ------------------------------------------------------------------
    def set_course(self, n: int) -> None:
        n = int(n)
        if not 1 <= n <= COURSES:
            raise ValueError(f"course must be 1..{COURSES}, got {n}")
        if self.course_box.currentData() != n:
            self._quiet = True
            try:
                self.course_box.setCurrentIndex(n - 1)
            finally:
                self._quiet = False
        if n == self.course:
            return
        self.course = n
        self.prepare()

    def next_course(self) -> None:
        """「다른 코스」: 다음 번호(32 다음 1)."""
        self.set_course(next_course(self.course))

    def _on_course_box(self, _index: int) -> None:
        if not self._quiet:
            self.set_course(int(self.course_box.currentData()))

    def set_who(self, who: str) -> None:
        """누구를 달리게 할지(실행 경주만): 칸을 다시 만들고 다시 준비한다."""
        if who not in WHO_MODES:
            raise ValueError(f"unknown race mode {who!r}")
        if self.who_box.currentData() != who:
            self._quiet = True
            try:
                self.who_box.setCurrentIndex(WHO_MODES.index(who))
            finally:
                self._quiet = False
        if who == self.who or self.runs is None:
            self.who = who
            return
        self.who = who
        entries, warnings, _n = entries_from_runs(self.runs, who)
        self._set_entries(entries, warnings)

    def _on_who_box(self, _index: int) -> None:
        if not self._quiet:
            self.set_who(str(self.who_box.currentData()))

    def set_mode(self, mode: str) -> None:
        """나란히(칸마다 캔버스) 또는 한 화면에 겹쳐서(환경이 모두 같을 때만)."""
        if mode not in VIEW_MODES:
            raise ValueError(f"unknown view mode {mode!r}")
        if mode == "overlay" and not self.overlay_allowed():
            mode = "split"
        if self.view_box.currentData() != mode:
            self._quiet = True
            try:
                self.view_box.setCurrentIndex(VIEW_MODES.index(mode))
            finally:
                self._quiet = False
        if mode == self.mode:
            return
        self.mode = mode
        self.stack.setCurrentWidget(self.overlay_canvas if mode == "overlay" else self.grid_widget)
        self.frame_ms.clear()
        if self.ready:
            self._show_time()

    def _on_view_box(self, _index: int) -> None:
        if not self._quiet:
            self.set_mode(str(self.view_box.currentData()))

    def set_speed(self, speed: float) -> None:
        speed = float(speed)
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}, got {speed}")
        if self.speed_box.currentData() != speed:
            self._quiet = True
            try:
                self.speed_box.setCurrentIndex(SPEEDS.index(speed))
            finally:
                self._quiet = False
        self.speed = speed
        self._update_hud()

    # ------------------------------------------------------------------
    # 준비
    # ------------------------------------------------------------------
    def prepare(self) -> None:
        """지금 코스로 모든 칸을 다시 시뮬레이션한다(배경 스레드). 앞 준비는 버린다."""
        self.stop_clock()
        self._tag += 1
        if self._cancel is not None:
            self._cancel.set()
            self._cancel = None
        self.ready = False
        self.finished = False
        self.rank_list = None
        self.t = 0.0
        self._hold = 0.0
        self._timer.setInterval(FAST_MS)
        self.frame_ms.clear()
        self.result_label.clear()
        self.overlay_canvas.set_overlay([])
        for lane in self.lanes:
            lane.set_replay(None)
            lane.set_rank(None)
            lane.set_status(S.RACE_PREPARING.format(done=0, total=len(self.lanes)))
        self._update_controls()
        if not self.entries:
            return
        trial = course_trial(self.course)
        jobs = [(e.env_text, e.genome, trial) for e in self.entries]
        self._cancel = threading.Event()
        th = _RaceThread(self._tag, jobs, self._cancel, QCoreApplication.instance())
        th.progress.connect(self._on_progress)
        th.finished_job.connect(self._on_ready)
        _start_thread(th)

    def _on_progress(self, tag: Any, done: int, total: int) -> None:
        if tag != self._tag:
            return
        for lane in self.lanes:
            if lane.replay is None:
                lane.set_status(S.RACE_PREPARING.format(done=done, total=total))

    def _on_ready(self, tag: Any, result: Any) -> None:
        if tag != self._tag:
            return  # 더 새로운 준비가 있다
        self._cancel = None
        if isinstance(result, Exception):
            reason = str(result).strip() or type(result).__name__
            if isinstance(result, EpisodeBuildError):  # 그 몸 수치로 조립할 수 없는 칸(v2.0 설계 4.9)
                reason = S.PLAY_BODY_BUILD_FAILED + " " + reason
            self.failure = S.RACE_FAILED.format(reason=reason)
            for lane in self.lanes:
                lane.set_status("")
            self._update_band()
            self._update_controls()
            return
        reps: list[Replay] = result
        for lane, rep in zip(self.lanes, reps, strict=True):
            lane.set_replay(rep)
        if self.overlay_allowed() and reps:
            self.overlay_canvas.set_scene(reps[0].geometry, None, channels=reps[0].channels,
                                          view=entry_view(self.entries[0]))
            self.overlay_canvas.set_population([])
            self.overlay_canvas.set_overlay([(r.trajectory, r.agent, QColor(e.color), e.label)
                                             for r, e in zip(reps, self.entries, strict=True)])
            self.overlay_canvas.set_follow("fit")
        self.ready = True
        self.t = 0.0
        self._show_time()
        self._update_controls()
        self.focus_start()
        self.prepared.emit()
        if self.isVisible():
            self.start()

    @property
    def replays(self) -> list[Replay]:
        return [lane.replay for lane in self.lanes if lane.replay is not None]

    @property
    def scores(self) -> list[float]:
        """칸마다 이 코스의 점수(준비 전에는 빈 목록)."""
        return [float(r.result.score) for r in self.replays] if self.ready else []

    def race_duration(self) -> float:
        """가장 긴 칸의 길이(초)."""
        return max((lane.duration for lane in self.lanes), default=0.0)

    # ------------------------------------------------------------------
    # 시계
    # ------------------------------------------------------------------
    def start(self) -> None:
        """출발(끝난 경주면 처음부터)."""
        if not self.ready:
            return
        if self.finished:
            self.seek(0.0)
        self.playing = True
        self._last_tick = time.perf_counter()
        self._paint_accum = 0.0
        self._timer.start()
        self._update_controls()

    def pause(self) -> None:
        self.stop_clock()
        self._update_controls()

    def stop_clock(self) -> None:
        self._timer.stop()
        self.playing = False

    def toggle(self) -> None:
        if self.playing:
            self.pause()
        else:
            self.start()

    def restart(self) -> None:
        """처음부터 다시 달린다."""
        if not self.ready:
            return
        self.seek(0.0)
        self.start()

    def seek(self, t: float) -> None:
        """경주 시계를 t초로(순위를 지우고 다시 센다)."""
        self.t = max(0.0, min(float(t), self.race_duration()))
        self._hold = 0.0
        self.finished = False
        self.rank_list = None
        self.result_label.clear()
        for lane in self.lanes:
            lane.set_rank(None)
        if self.ready:
            self._show_time()

    def advance(self, dt_real: float) -> None:
        """실제 시간 dt_real초만큼 경주 시계를 움직인다(배속을 곱함). 타이머가 부르고, 테스트가 직접 부른다."""
        if not self.ready or self.finished:
            return
        step = max(0.0, float(dt_real))
        end = self.race_duration()
        if self.t < end:
            self.t = min(end, self.t + step * self.speed)
        else:
            self._hold += step
        self._show_time()
        if self.t >= end and self._hold >= HOLD_SECONDS:
            self._finish()

    def _tick(self) -> None:
        now = time.perf_counter()
        dt_real = now - self._last_tick
        self._last_tick = now
        self._record_paint()
        self.advance(dt_real)

    def _on_paint(self, ms: float) -> None:
        self._paint_accum += ms

    def _record_paint(self) -> None:
        """앞 프레임에 그린 시간을 남기고, 최근 30프레임 중앙값이 12ms를 넘으면 30fps로 낮춘다."""
        if self._paint_accum <= 0.0:
            return
        ms, self._paint_accum = self._paint_accum, 0.0
        self.frame_ms.append(ms)
        if len(self.paint_log) < PAINT_LOG_MAX:
            self.paint_log.append(ms)
        if (self._timer.interval() == FAST_MS and len(self.frame_ms) >= PAINT_WINDOW
                and statistics.median(self.frame_ms) > SLOW_PAINT_MS):
            self._timer.setInterval(SLOW_MS)

    @property
    def interval_ms(self) -> int:
        """지금 프레임 간격(16 = 약 60fps, 33 = 약 30fps)."""
        return int(self._timer.interval())

    def _show_time(self) -> None:
        t = self.t
        if self.mode == "overlay":
            self.overlay_canvas.set_time(t)
            self.overlay_canvas.set_hud(S.RACE_HUD.format(n=self.course, t=t, speed=self.speed))
        else:
            for lane in self.lanes:
                lane.canvas.set_time(t)
        for k, lane in enumerate(self.lanes):
            lane.set_status(self._lane_text(k))
        self._update_hud()

    def _lane_text(self, k: int) -> str:
        lane = self.lanes[k]
        rep = lane.replay
        if rep is None:
            return ""
        res = rep.result
        if self.t < lane.duration:
            return S.RACE_RUNNING.format(t=self.t)
        if self.rank_list is not None:
            rank, tie = self.rank_list[k]
            fmt = S.RACE_LANE_RANK_TIE if tie else S.RACE_LANE_RANK
            return fmt.format(rank=rank, end=res.end_label, t=res.time, score=res.score)
        return S.RACE_LANE_DONE.format(end=res.end_label, t=res.time, score=res.score)

    def _update_hud(self) -> None:
        self.hud_label.setText(S.RACE_HUD.format(n=self.course, t=self.t, speed=self.speed))

    def _finish(self) -> None:
        self.stop_clock()
        self.finished = True
        self.rank_list = ranks(self.entries, self.scores)
        for k, lane in enumerate(self.lanes):
            lane.set_status(self._lane_text(k))
        if self.rank_list is not None:
            for lane, (rank, tie) in zip(self.lanes, self.rank_list, strict=True):
                lane.set_rank(rank, tie)
            order = sorted(range(len(self.entries)), key=lambda k: (self.rank_list[k][0], k))  # type: ignore[index]
            # U25: 이름은 굵게, 항목 사이는 넓게(「·」가 이름 안과 순위 사이에 함께 쓰여 구별되지 않던 문제)
            items = [S.RACE_RESULT_ITEM.format(rank=self.rank_list[k][0], name=f"<b>{html.escape(self.entries[k].label)}</b>")
                     for k in order]
            sep = "&nbsp;&nbsp;" + html.escape(S.RACE_RESULT_SEP.strip()) + "&nbsp;&nbsp;"
            self.result_label.setText(html.escape(S.RACE_RESULT.format(order="\0")).replace("\0", sep.join(items)))
        else:
            self.result_label.clear()
        self._update_controls()
        self.finished_race.emit()

    def _update_controls(self) -> None:
        self.start_button.setText(S.RACE_PAUSE if self.playing else S.RACE_START)
        why = "" if self.ready else (S.RACE_WHY_FAILED if self.failure else S.RACE_WHY_PREPARING)
        for button in (self.start_button, self.restart_button, self.gif_button):
            style.set_enabled(button, self.ready, why)
        self._update_hud()

    # ------------------------------------------------------------------
    # 키 (U23): 창 단위 단축키를 쓰지 않는다. 초점 위젯이 쓰지 않은 키만 여기로 올라온다(단추의 Space는 그 단추,
    # 체크 상자의 Space는 체크, 고르기 칸의 글자는 글자 찾기). Enter는 대화 상자의 기본 단추(「출발」)가 받는다.
    # ------------------------------------------------------------------
    def focus_start(self) -> None:
        """준비가 끝날 때: 초점을 「출발」에 둔다. 사용자가 단추가 아닌 칸(고르기 칸·체크 상자)에 둔 초점은 옮기지 않는다."""
        fw = QApplication.focusWidget()
        if fw is not None and fw is not self and self.isAncestorOf(fw) and not isinstance(fw, QAbstractButton):
            return
        self._focus_default()

    def _focus_default(self) -> None:
        """「출발」에 초점(준비 중이라 꺼져 있으면 화면 자신에: 그동안 Enter·Space는 아무것도 하지 않는다)."""
        if self.start_button.isEnabled():
            self.start_button.setFocus(Qt.FocusReason.OtherFocusReason)
        else:
            self.setFocus(Qt.FocusReason.OtherFocusReason)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        plain = event.modifiers() in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.KeypadModifier)
        if plain and self.ready and event.key() == Qt.Key.Key_Space:
            self.toggle()
            event.accept()
            return
        if plain and self.ready and event.key() == Qt.Key.Key_R:
            self.restart()
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # 경주 영상
    # ------------------------------------------------------------------
    def default_gif_path(self) -> Path:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = S.RACE_GIF_NAME.format(n=self.course, stamp=stamp)
        run = self.entries[0].run if self.entries else None
        folder = (run / "exports") if run is not None else paths.data_dir()
        return folder / name

    def ask_save_path(self, default: Path) -> Path | None:
        """저장할 GIF 경로를 묻는다. 테스트는 이 메서드를 바꿔 끼운다."""
        chosen, _ = QFileDialog.getSaveFileName(self, S.VIDEO_SAVE_TITLE, str(default), S.VIDEO_FILTER)
        return Path(chosen) if chosen else None

    def show_error(self, message: str) -> None:
        """오류 상자(다음 할 일 + 오류 보고 묶음 단추, errors.show_error). 테스트는 이 메서드를 바꿔 끼운다."""
        from evolab.app import errors

        errors.show_error(self, S.VIDEO_SAVE_TITLE, message)

    def save_gif(self, path: Path | None = None) -> Path | None:
        """지금 경주를 내보낸다. 취소하면 None.

        path가 없으면 내보내기 대화 상자(MP4·GIF·폰 재생 파일, 나란히면 칸 격자·겹쳐서면 한 세계, v3.1 설계 D48). path를 주면
        v3.0처럼 이 프로세스에서 GIF 하나(나란히: 칸 격자 320×200, 20fps / 겹쳐서: 기존 겹친 영상)."""
        from PySide6.QtWidgets import QApplication

        from evolab.app.export.video import export_gif, export_race_gif

        if not self.ready:
            return None
        if path is None:
            from evolab.app.export.dialog import choose_and_export

            was_playing = self.playing
            self.pause()
            default = self.default_gif_path()
            try:
                out = choose_and_export(self, kind="race", items=list(self.lanes),
                                        title=S.VIDEO_RACE_TITLE.format(n=self.course), default_name=default.stem,
                                        meta={}, folder=default.parent, overlay=self.mode == "overlay")
            finally:
                if was_playing and not self.finished:
                    self.start()
            if out is not None:
                self.message.setText(S.EXP_DONE.format(path=out))
                self.message.show()
            return out
        path = Path(path)
        if path.suffix.lower() != ".gif":
            path = path.with_suffix(".gif")
        was_playing = self.playing
        self.pause()
        title = S.VIDEO_RACE_TITLE.format(n=self.course)
        items = [(lane.replay, QColor(lane.entry.color), lane.entry.label) for lane in self.lanes
                 if lane.replay is not None]
        dlg = QProgressDialog(S.VIDEO_PROGRESS, S.VIDEO_CANCEL, 0, 100, self)
        dlg.setMinimumDuration(300)

        def progress(done: int, total: int) -> bool:
            dlg.setMaximum(total)
            dlg.setValue(done)
            QApplication.processEvents()
            return not dlg.wasCanceled()

        try:
            if self.mode == "overlay":
                out = export_gif(items, path, title=title, progress=progress)
            else:
                out = export_race_gif(items, path, lane_grid(len(items))[1], title=title, progress=progress)
        except (OSError, ValueError) as exc:
            from evolab.app import errors

            why = errors.reason(exc)  # 날것 예외 글 대신 한국어 까닭(v3.1, v3.0 고침 13)
            self.show_error(S.EXP_FAILED.format(reason=why))
            return None
        except MemoryError:  # 프레임은 쌓지 않지만 메모리가 아주 모자란 PC: 프로그램을 끝내지 않고 알린다
            self.show_error(S.VIDEO_NO_MEMORY)
            return None
        finally:
            dlg.close()
            dlg.deleteLater()
            if was_playing and not self.finished:
                self.start()
        if out is not None:
            self.message.setText(S.VIDEO_DONE.format(path=out))
            self.message.show()
        return out

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """창을 닫을 때: 시계를 멈추고 도는 준비를 버린다."""
        self.stop_clock()
        self._tag += 1
        if self._cancel is not None:
            self._cancel.set()
            self._cancel = None
        self._update_controls()

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        # 창을 내리거나 닫으면 멈추고, 다시 보이면 달리던 경주만 이어 간다(사용자가 멈춘 경주는 그대로)
        self._resume_on_show = self.playing
        self.stop_clock()
        try:
            self._update_controls()
        except RuntimeError:  # 프로그램을 끝낼 때: 단추가 먼저 지워졌다
            pass
        super().hideEvent(event)

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        if self.ready and self._resume_on_show and not self.finished:
            self.start()
        elif not self.ready and self._cancel is None and self.entries and not self.failure:
            self.prepare()  # 준비 중에 닫았다가 다시 연 창
        self._resume_on_show = False
        QTimer.singleShot(0, self._focus_on_show)

    def _focus_on_show(self) -> None:
        """창이 뜰 때: Qt가 처음 칸(「코스」 고르기 칸)에 준 초점 대신 「출발」에."""
        try:
            if self.isVisible():
                self._focus_default()
        except RuntimeError:  # 창이 이미 지워졌다
            pass


class RaceDialog(QDialog):
    """나란히 경주 창. 연 화면의 창이 주인이고, 하나를 다시 쓴다(show_race)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint | Qt.WindowType.WindowSystemMenuHint
                            | Qt.WindowType.WindowMinMaxButtonsHint | Qt.WindowType.WindowCloseButtonHint)
        self.setModal(False)
        self.setWindowTitle(S.RACE_TITLE)
        self.view = RaceView(self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)
        self.resize(*DEFAULT_SIZE)

    def set_race(self, entries: Sequence[RaceEntry], *, title: str, warnings: Sequence[str] = (),
                 runs: Sequence[tuple[Path, str]] | None = None) -> None:
        self.setWindowTitle(title)
        self.view.set_race(entries, warnings=warnings, runs=runs)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.view.shutdown()
        super().closeEvent(event)

    def reject(self) -> None:  # Esc
        self.view.shutdown()
        super().reject()


def show_race(parent: QWidget | None, entries: list[RaceEntry], *, title: str, warnings: Sequence[str] = (),
              runs: Sequence[tuple[Path, str]] | None = None) -> RaceDialog:
    """경주 창을 연다. 주인(parent.window())에 이미 경주 창이 있으면 그 창의 내용을 바꾸고 앞으로 가져온다.

    runs가 있으면(비교 화면) 「누구를」을 보이고, 바꾸면 entries_from_runs(runs, 누구를)로 칸을 다시 만든다.
    """
    owner = parent.window() if parent is not None else None
    dlg: RaceDialog | None = None
    if owner is not None:
        found = owner.findChildren(RaceDialog, options=Qt.FindChildOption.FindDirectChildrenOnly)
        dlg = found[0] if found else None
    if dlg is None:
        dlg = RaceDialog(owner)
    dlg.set_race(entries, title=title, warnings=warnings, runs=runs)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg


__all__ = ["RaceDialog", "RaceView", "show_race", "shutdown_race", "wait_threads"]

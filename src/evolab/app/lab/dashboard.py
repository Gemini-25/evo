"""실험실 아래쪽 통계 대시보드: 세대별 통계 줄을 그래프 타일로 보여 준다.

- 자료는 열마다 늘어나는 numpy 배열에 모은다(용량을 두 배씩 늘려 한 줄 추가가 평균 O(1)).
  세대마다 그래프 항목을 새로 만들지 않고 setData로 바꾼다.
- 통계 줄은 학습 중에 바로 오는 dict(숫자)일 수도, RunStore.read_stats()의 글자 dict일 수도 있다.
  빈 글자와 없는 열은 NaN으로 읽는다.
- 타일(그래프 묶음)은 켜고 끄고 순서를 바꿀 수 있고, 배치는 QSettings의 "dashboard" 묶음에 저장한다.
- 숨긴 타일은 갱신하지 않고, 다시 켤 때 한 번에 그린다.
- v1.1 학습 방법 열(learner, learner_best, qualifying, switch 등)이 있으면 학습 방법 교체 세로선(자동 재시작·예선),
  지금 학습 방법 최고점 선, 예선 띠를 더한다. 이 열이 없는 옛 실행은 NaN으로 읽혀 아무것도 더 그리지 않는다.
- v1.2(설계 D14·D15): 대시보드 위에 쪽 탭 「학습 그래프 | 보는 재미」. 기존 여섯 타일(TILE_IDS)은 「학습 그래프」,
  새 다섯 타일(FUN_TILE_IDS: 유전자 지도, 뇌 들여다보기, 궤적 밀도, 행동 분포, 몸 수치 추이)은 「보는 재미」 쪽이다.
  지금 쪽이 아닌 타일은 숨긴 타일처럼 갱신하지 않는다(dirty).
- 갱신 펌프(D15): 대시보드가 화면에 떠 있으면 자료가 바뀌어도 바로 그리지 않고 타일에 dirty만 표시한 뒤, 화면에 보이는
  (스크롤 밖이 아닌) dirty 타일을 이벤트 루프 차례마다 하나씩 갱신하고 그린다(사이사이 입력을 처리하게). 스크롤 밖 타일은
  보일 때 갱신한다. 떠 있지 않으면(테스트, 다른 화면) 지금처럼 바로 갱신한다. flush()는 지금 쪽의 dirty 타일을 바로 갱신한다.
- 「보는 재미」 자료는 RunHistory(history.py)에 모은다: 실행을 열면 load_history가 배경 스레드로 세대 대표 파일의 관찰
  기록을 읽고, 학습 중에는 add_generation이 세대 메시지의 관찰 기록을 넣는다. 보는 세대(focus)는 실험실 타임라인이 정한다.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QRectF, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QFont,
    QFontMetrics,
    QPaintEvent,
    QPalette,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QStyleOptionTab,
    QStylePainter,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from evolab import paths
from evolab.app import strings as S

if TYPE_CHECKING:
    from evolab.app.lab.history import RunHistory
    from evolab.app.lab.watch_types import BrainSource, WatchMeta
    from evolab.runs.observe import Observation
    from evolab.world.physics2d.episode import WorldGeometry

TILE_IDS: tuple[str, ...] = ("trend", "heatmap", "diversity", "blocks", "ends", "perf")
"""「학습 그래프」 쪽 타일(v1.1까지의 여섯, 이 순서가 기본 배치)."""
FUN_TILE_IDS: tuple[str, ...] = ("genemap", "brain", "density", "actions", "body")
"""「보는 재미」 쪽 타일(v1.2, 이 순서가 기본 배치). brain(BrainTile)은 app/lab/brain_view.py에 있다."""
ALL_TILE_IDS: tuple[str, ...] = TILE_IDS + FUN_TILE_IDS
PAGES: tuple[str, ...] = ("learning", "fun")
TILE_PAGES: dict[str, tuple[str, ...]] = {"learning": TILE_IDS, "fun": FUN_TILE_IDS}
"""쪽 → 그 쪽 타일(기본 순서)."""
PAGE_OF: dict[str, str] = {tid: page for page, ids in TILE_PAGES.items() for tid in ids}
"""타일 → 쪽."""
AXES: tuple[str, ...] = ("generation", "evaluations", "elapsed")
BLOCK_MODES: tuple[str, ...] = ("rep", "mean")
DEFAULT_STAGNATION = 50
DEFAULT_COLUMNS = 2
MAX_COLUMNS = 3
HEATMAP_BINS = 40
MAX_SERIES = 8  # 범주 색 수. 넘으면 뒤쪽을 '기타'로 묶는다
TILE_MIN_HEIGHT = 230
AXIS_WIDTH = 66  # 세로축 칸 폭(눈금 글자 + 축 이름). 타일끼리 맞춘다
SETTINGS_GROUP = "dashboard"
SETTINGS_FILE = "settings.ini"
# v1.1 stats.csv의 switch 열(이 세대 끝에 일어난 학습 방법 교체) → 자동 세로선 종류.
# 0(없음)과 열이 없는 옛 실행은 선을 긋지 않는다
SWITCH_KINDS: dict[int, str] = {1: "restart", 2: "restart", 3: "qualify_next", 4: "qualify_pick"}
SWITCH_RESTART_FROM: dict[int, str] = {1: "fresh", 2: "champion"}  # 새로 시작한 곳
QUALIFY_MARK_KINDS: tuple[str, ...] = ("qualify_next", "qualify_pick")

_NAN = math.nan


# ---------------------------------------------------------------------------
# 색 (밝은 화면·어두운 화면 각각 고른 값. 범주 색은 정해진 순서로만 쓴다)
@dataclass(frozen=True)
class Theme:
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    ramp: tuple[str, ...]  # 순차 색(히트맵): 바탕에서 멀어지는 순서


LIGHT = Theme(
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    ramp=("#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)
DARK = Theme(
    surface="#1a1a19",
    ink="#f0efec",
    ink2="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    ramp=("#104281", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4", "#cde2fb"),
)


def theme_for(widget: QWidget) -> Theme:
    """창 색(윈도우 밝은/어두운 모드)에 맞는 색 묶음."""
    return DARK if widget.palette().color(QPalette.ColorRole.Window).lightness() < 128 else LIGHT


def _qcolor(color: str, alpha: int | None = None) -> QColor:
    c = QColor(color)
    if alpha is not None:
        c.setAlpha(alpha)
    return c


def _blend(color: str, surface: str, amount: float) -> str:
    """반투명 색이 바탕 위에서 보이는 색 (범례 견본용)."""
    a, b = QColor(color), QColor(surface)
    mix = [round(x * amount + y * (1 - amount)) for x, y in ((a.red(), b.red()), (a.green(), b.green()), (a.blue(), b.blue()))]
    return QColor(*mix).name()


def _pen(color: str | QColor, width: float = 1.5, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> QPen:
    pen = pg.mkPen(color=color, width=width)
    pen.setStyle(style)
    return pen


def mark_style(kind: str) -> str:
    """세로선 종류 → 펜 묶음: "restart"(자동 재시작), "qualify"(예선 다음·예선 끝), "event"(사건 기록)."""
    if kind == "restart":
        return "restart"
    if kind in QUALIFY_MARK_KINDS:
        return "qualify"
    return "event"


def mark_pen(theme: Theme, style: str) -> QPen:
    """세로선 펜: 사건은 회색 점선, 자동 재시작은 실선, 예선은 점점선."""
    if style == "restart":
        return _pen(theme.series[7], 1.5)
    if style == "qualify":
        return _pen(theme.series[6], 1.5, Qt.PenStyle.DotLine)
    return _pen(theme.ink2, 1, Qt.PenStyle.DashLine)


def _shade_region(brush: QColor, z: float) -> pg.LinearRegionItem:
    """가로축 구간 음영(테두리 없음, 마우스로 못 옮김)."""
    region = pg.LinearRegionItem(
        values=(0, 1), orientation="vertical", movable=False, brush=brush, pen=pg.mkPen(None), hoverBrush=brush
    )
    for line in region.lines:
        line.setPen(pg.mkPen(None))
        line.setHoverPen(pg.mkPen(None))
    region.setZValue(z)
    return region


# ---------------------------------------------------------------------------
# 값 읽기
def to_float(value: Any) -> float:
    """통계 값 하나: '' 나 None, 읽을 수 없는 글자는 NaN."""
    if value is None:
        return _NAN
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return _NAN
        try:
            return float(text)
        except ValueError:
            return _NAN
    try:
        return float(value)
    except (TypeError, ValueError):
        return _NAN


_EMPTY = np.empty(0, dtype=np.float64)


def parse_scores(value: Any) -> np.ndarray:
    """scores 열(공백으로 나눈 숫자 글자 또는 숫자 목록)을 유한한 값만 담은 배열로."""
    if value is None:
        return _EMPTY
    if isinstance(value, str):
        parts = value.split()
        if not parts:
            return _EMPTY
        try:
            arr = np.array(parts, dtype=np.float64)
        except ValueError:
            arr = np.array([to_float(p) for p in parts], dtype=np.float64)
    else:
        try:
            arr = np.asarray(value, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            return _EMPTY
    return arr[np.isfinite(arr)]


def fmt_value(v: float) -> str:
    """읽기 쉬운 숫자 글자. NaN은 '-'."""
    if not math.isfinite(v):
        return "-"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    if a == 0:
        return "0"
    return f"{v:.3g}"


def _split_setting(value: Any) -> list[str]:
    """QSettings 값(쉼표로 이은 글자 또는 목록)을 글자 목록으로. 이상한 값은 빈 목록."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = [str(x) for x in value]
    else:
        items = str(value).split(",")
    return [x.strip() for x in items if x.strip()]


# ---------------------------------------------------------------------------
# 자료
class StatsTable:
    """세대별 숫자 열. 열마다 늘어나는 float64 배열(용량을 두 배씩 늘림)."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.n = 0
        self._cap = 0
        self._cols: dict[str, np.ndarray] = {}
        self.columns_version = 0  # 열 이름 집합이 바뀔 때마다 늘어난다

    def _grow(self) -> None:
        cap = max(256, self._cap * 2)
        for name, arr in self._cols.items():
            new = np.full(cap, _NAN)
            new[: self.n] = arr[: self.n]
            self._cols[name] = new
        self._cap = cap

    def append(self, values: Mapping[str, float]) -> None:
        if self.n >= self._cap:
            self._grow()
        i = self.n
        cols = self._cols
        for name, v in values.items():
            arr = cols.get(name)
            if arr is None:
                arr = np.full(self._cap, _NAN)
                cols[name] = arr
                self.columns_version += 1
            arr[i] = v
        self.n = i + 1

    def col(self, name: str) -> np.ndarray:
        """앞의 n칸(보기). 없는 열은 NaN."""
        arr = self._cols.get(name)
        if arr is None:
            return np.full(self.n, _NAN)
        return arr[: self.n]

    def has(self, name: str) -> bool:
        return name in self._cols

    def names(self) -> list[str]:
        return list(self._cols)


class ScoreHistogram:
    """점수 분포 지도 자료: 세대(줄) × 점수 구간(칸)의 개체 비율.

    구간은 지금까지 본 점수의 최솟값~최댓값에 여유를 둔 범위를 같은 폭으로 나눈다. 새 점수가 범위를
    벗어나면 범위를 넉넉히(전체 폭의 25%) 넓히고 모아 둔 원래 점수로 전체를 한 번에 다시 센다
    (bincount 한 번이라 2000세대 × 개체 256에서도 몇 ms). 범위 안이면 새 세대 줄 하나만 센다.
    """

    def __init__(self, bins: int = HEATMAP_BINS) -> None:
        self.bins = bins
        self.clear()

    def clear(self) -> None:
        self.n = 0
        self.lo = _NAN
        self.hi = _NAN
        self.rebuilds = 0
        self._frac = np.zeros((0, self.bins), dtype=np.float32)
        self._flat = np.empty(0, dtype=np.float64)
        self._row_of = np.empty(0, dtype=np.int64)
        self._flat_n = 0
        self._count = np.zeros(0, dtype=np.int64)
        self._stale = False

    def _ensure_rows(self, need: int) -> None:
        if need <= self._frac.shape[0]:
            return
        cap = max(256, self._frac.shape[0] * 2, need)
        frac = np.zeros((cap, self.bins), dtype=np.float32)
        frac[: self.n] = self._frac[: self.n]
        self._frac = frac
        count = np.zeros(cap, dtype=np.int64)
        count[: self.n] = self._count[: self.n]
        self._count = count

    def _ensure_flat(self, need: int) -> None:
        if need <= self._flat.shape[0]:
            return
        cap = max(4096, self._flat.shape[0] * 2, need)
        flat = np.empty(cap, dtype=np.float64)
        flat[: self._flat_n] = self._flat[: self._flat_n]
        rows = np.empty(cap, dtype=np.int64)
        rows[: self._flat_n] = self._row_of[: self._flat_n]
        self._flat, self._row_of = flat, rows

    def _bin(self, s: np.ndarray) -> np.ndarray:
        width = (self.hi - self.lo) / self.bins
        return np.clip(((s - self.lo) / width).astype(np.int64), 0, self.bins - 1)

    def _widen(self, smin: float, smax: float) -> bool:
        """범위를 넓혔으면 True."""
        if not math.isfinite(self.lo):
            span = smax - smin
            if span <= 0:
                span = max(abs(smax) * 0.1, 1.0)
                smin, smax = smin - span / 2, smax + span / 2
            pad = span * 0.1
            self.lo, self.hi = smin - pad, smax + pad
            return True
        if smin >= self.lo and smax <= self.hi:
            return False
        lo, hi = min(self.lo, smin), max(self.hi, smax)
        pad = (hi - lo) * 0.25
        if smin < self.lo:
            lo -= pad
        if smax > self.hi:
            hi += pad
        self.lo, self.hi = lo, hi
        return True

    def append(self, scores: np.ndarray, *, update: bool = True) -> None:
        """한 세대의 점수를 더한다. update=False면 원래 점수만 모으고 finish()에서 한 번에 센다."""
        i = self.n
        self._ensure_rows(i + 1)
        k = len(scores)
        self._ensure_flat(self._flat_n + k)
        self._flat[self._flat_n : self._flat_n + k] = scores
        self._row_of[self._flat_n : self._flat_n + k] = i
        self._flat_n += k
        self._count[i] = k
        self._frac[i] = 0.0
        self.n = i + 1
        widened = k > 0 and self._widen(float(scores.min()), float(scores.max()))
        if not update:
            self._stale = True
        elif widened or self._stale:
            self._rebuild()
        elif k:
            self._frac[i] = np.bincount(self._bin(scores), minlength=self.bins) / k

    def finish(self) -> None:
        if self._stale:
            self._rebuild()

    def _rebuild(self) -> None:
        self._stale = False
        self.rebuilds += 1
        n, bins = self.n, self.bins
        self._frac[:n] = 0.0
        if not self._flat_n or not math.isfinite(self.lo):
            return
        s = self._flat[: self._flat_n]
        rows = self._row_of[: self._flat_n]
        counts = np.bincount(rows * bins + self._bin(s), minlength=n * bins).reshape(n, bins)
        self._frac[:n] = counts / np.maximum(self._count[:n], 1)[:, None]

    def matrix(self) -> np.ndarray:
        """(세대 수, 구간 수) 개체 비율 0..1."""
        return self._frac[: self.n]


def stagnation_ranges(gens: np.ndarray, champion: np.ndarray, n_gen: int) -> list[tuple[float, float]]:
    """챔피언 점수가 n_gen세대 넘게 오르지 않은 구간 (마지막으로 오른 세대, 다음에 오른 세대 또는 지금)."""
    if len(gens) == 0:
        return []
    finite = np.isfinite(champion)
    if not finite.any():
        return []
    c = np.where(finite, champion, -np.inf)
    prev = np.concatenate(([-np.inf], np.maximum.accumulate(c)[:-1]))
    imp = np.flatnonzero(c > prev)
    starts = gens[imp]
    ends = np.append(gens[imp[1:]], gens[-1])
    keep = ends - starts >= n_gen
    return [(float(a), float(b)) for a, b in zip(starts[keep], ends[keep], strict=True)]


def _step_edges(gens: np.ndarray) -> np.ndarray:
    """세대마다 막대 하나: 세대 사이 가운데를 경계로 한 len+1개 x."""
    n = len(gens)
    edges = np.empty(n + 1)
    if n == 0:
        return edges
    edges[0] = gens[0] - 0.5
    edges[-1] = gens[-1] + 0.5
    if n > 1:
        edges[1:-1] = (gens[1:] + gens[:-1]) / 2
    return edges


def _fold(series: list[tuple[str, str]], values: np.ndarray) -> tuple[list[tuple[str, str]], np.ndarray]:
    """계열이 범주 색 수보다 많으면 뒤쪽을 '기타' 하나로 묶는다. values: (계열, 세대)."""
    if len(series) <= MAX_SERIES:
        return series, values
    keep = MAX_SERIES - 1
    other = np.nansum(values[keep:], axis=0)
    return [*series[:keep], ("", S.DASH_OTHER)], np.vstack([values[:keep], other[None, :]])


@dataclass
class _Entry:
    """범례 한 칸. text(i)는 i번째 세대의 값 글자(마우스를 올렸을 때)."""

    color: str
    glyph: str
    name: str
    text: Callable[[int], str] | None = None


GLYPH_LINE = "━"
GLYPH_DASH = "╌"
GLYPH_FILL = "■"
GLYPH_EARLIER = "\u25c0"
GLYPH_LATER = "\u25b6"
GLYPH_HIDE = "\u2715"


# ---------------------------------------------------------------------------
# 타일
class _GraphView(pg.GraphicsLayoutWidget):
    """그래프 칸. 기본 권장 크기(600×480)가 크면 스크롤 영역이 타일을 너무 키워서 작게 알린다."""

    def sizeHint(self) -> QSize:
        return QSize(320, 160)

    def wheelEvent(self, ev: QWheelEvent) -> None:
        # 대시보드는 스크롤 영역 안에 있으므로 그냥 휠은 화면을 넘기고, Ctrl+휠로 그래프를 확대한다
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(ev)
        else:
            ev.ignore()


class _Tile(QFrame):
    """그래프 타일 하나: 제목 줄(옮기기·숨기기 단추), 범례(마우스를 올리면 값), 그래프."""

    follows_axis = True  # 가로축 전환을 따르는가 (아니면 언제나 세대)
    mark_labels = False  # 세로선 위에 사건 이름을 적는가

    def __init__(self, dash: Dashboard, tile_id: str) -> None:
        super().__init__()
        self.dash = dash
        self.tile_id = tile_id
        self.theme = dash.theme
        self.dirty = True
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(TILE_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)
        head = QHBoxLayout()
        head.setSpacing(4)
        self.title = QLabel(S.DASH_TILE_TITLES[tile_id])
        font = QFont(self.title.font())
        font.setBold(True)
        self.title.setFont(font)
        head.addWidget(self.title)
        head.addStretch(1)
        self.controls = QHBoxLayout()
        self.controls.setSpacing(4)
        head.addLayout(self.controls)
        # 단추 그림은 기호 글자로(표준 아이콘은 어두운 화면에서 안 보이는 스타일이 있다)
        for glyph, tip, slot in (
            (GLYPH_EARLIER, S.DASH_TILE_MOVE_EARLIER, lambda: dash.move_tile_by(tile_id, -1)),
            (GLYPH_LATER, S.DASH_TILE_MOVE_LATER, lambda: dash.move_tile_by(tile_id, 1)),
            (GLYPH_HIDE, S.DASH_TILE_HIDE, lambda: dash.set_tile_visible(tile_id, False)),
        ):
            btn = QToolButton()
            btn.setAutoRaise(True)
            btn.setText(glyph)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            head.addWidget(btn)
        lay.addLayout(head)

        self.legend = QLabel()
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        self.legend.setWordWrap(True)
        small = QFont(self.legend.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() * 0.9))
        self.legend.setFont(small)
        self.legend.setStyleSheet(f"color: {self.theme.ink2};")
        lay.addWidget(self.legend)

        self.view = _GraphView()
        self.view.setBackground(self.theme.surface)
        self.view.ci.setContentsMargins(2, 2, 6, 2)
        self.view.ci.setSpacing(0)
        self.message = QLabel()
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setWordWrap(True)
        self.message.setStyleSheet(f"color: {self.theme.ink2};")
        self.stack = QStackedWidget()
        self.stack.addWidget(self.view)
        self.stack.addWidget(self.message)
        lay.addWidget(self.stack, 1)

        self.plots: list[pg.PlotItem] = []
        self._fixed_y: dict[int, tuple[float, float]] = {}
        self._mark_pools: list[list[pg.InfiniteLine]] = []
        # j번째 세로선(모든 그래프 공통)에 지금 놓인 (가로축 값, 풍선 도움말, 이름, 펜 묶음). None이면 숨김
        self._mark_applied: list[tuple[float, str, str, str] | None] = []
        self._cross: list[pg.InfiniteLine] = []
        self._mark_key: tuple[bytes, tuple[str, ...], tuple[str, ...]] | None = None
        self._legend_text = ""
        self._hover_index: int | None = None
        self._mouse = pg.SignalProxy(self.view.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse)
        self.view.viewport().installEventFilter(self)

    # -- 그래프 만들기 --------------------------------------------------------
    def add_plot(self, y_label: str, *, x_axis: bool = True) -> pg.PlotItem:
        t = self.theme
        plot = self.view.ci.addPlot(row=len(self.plots), col=0)
        plot.setMenuEnabled(False)
        plot.showGrid(x=False, y=True, alpha=0.35)
        for name in ("left", "bottom"):
            ax = plot.getAxis(name)
            ax.setPen(_pen(t.axis, 1))
            ax.setTextPen(_pen(t.muted, 1))
            ax.enableAutoSIPrefix(False)
        plot.getAxis("left").setWidth(AXIS_WIDTH)
        plot.setLabel("left", y_label, color=t.ink2)
        if x_axis:
            plot.setLabel("bottom", self.x_label(), color=t.ink2)
        else:
            plot.getAxis("bottom").setStyle(showValues=False)
            plot.getAxis("bottom").setHeight(4)
        if self.plots:
            plot.setXLink(self.plots[0])
        cross = pg.InfiniteLine(angle=90, movable=False, pen=_pen(t.muted, 1, Qt.PenStyle.DotLine))
        cross.setZValue(50)
        cross.hide()
        plot.addItem(cross, ignoreBounds=True)
        self._cross.append(cross)
        self._mark_pools.append([])
        self.plots.append(plot)
        return plot

    def x_label(self) -> str:
        return S.DASH_AXIS_NAMES[self.dash.axis if self.follows_axis else "generation"]

    def axis_changed(self) -> None:
        if not self.follows_axis or not self.plots:
            return
        self.plots[-1].setLabel("bottom", self.x_label(), color=self.theme.ink2)
        self.reset_view()
        self._mark_key = None

    def fix_y(self, plot: pg.PlotItem, lo: float, hi: float) -> None:
        """세로 범위를 고정한다(비율 그래프). reset_view에서도 지킨다."""
        self._fixed_y[id(plot)] = (lo, hi)
        plot.enableAutoRange(axis="y", enable=False)
        plot.setYRange(lo, hi, padding=0.05)

    def reset_view(self) -> None:
        """사용자가 확대·이동한 것을 풀고 자료 전체를 보인다."""
        for p in self.plots:
            fixed = self._fixed_y.get(id(p))
            if fixed is None:
                p.enableAutoRange()
            else:
                p.enableAutoRange(axis="x")
                p.setYRange(*fixed, padding=0.05)

    # -- 갱신 -----------------------------------------------------------------
    def refresh(self) -> None:
        """대시보드 자료로 그래프를 다시 채운다(항목은 그대로, setData만)."""
        self.dirty = False
        self.update_data()
        self.sync_marks()
        self.update_legend()

    def update_data(self) -> None:
        raise NotImplementedError

    def entries(self) -> list[_Entry]:
        return []

    def x_values(self) -> np.ndarray:
        return self.dash.x_values(self.follows_axis)

    def sync_marks(self) -> None:
        xs, tips, labels, kinds = self.dash.mark_positions(self.follows_axis)
        key = (xs.tobytes(), tuple(tips), tuple(kinds))
        if key == self._mark_key:
            return
        self._mark_key = key
        t = self.theme
        n = len(xs)
        applied = self._mark_applied
        for k, (plot, pool) in enumerate(zip(self.plots, self._mark_pools, strict=True)):
            labelled = self.mark_labels and k == 0
            while len(pool) < n:
                line = pg.InfiniteLine(
                    angle=90,
                    movable=False,
                    pen=mark_pen(t, "event"),
                    label="" if labelled else None,
                    labelOpts={"position": 0.96, "color": t.ink2, "fill": _qcolor(t.surface, 210), "anchors": [(0, 0), (0, 0)]}
                    if labelled
                    else None,
                )
                line.setZValue(40)
                line.hide()
                plot.addItem(line, ignoreBounds=True)
                pool.append(line)
        while len(applied) < n:
            applied.append(None)
        # 값이 바뀐 선만 고친다(자동 재시작이 잦은 긴 실행에서 새 선 하나 때문에 모든 선을 다시 놓지 않게)
        for j, before in enumerate(applied):
            want = (float(xs[j]), tips[j], labels[j], mark_style(kinds[j])) if j < n and math.isfinite(xs[j]) else None
            if want == before:
                continue
            for k, pool in enumerate(self._mark_pools):
                line = pool[j]
                if want is None:
                    line.hide()
                    continue
                x, tip, label, style = want
                if before is None or before[3] != style:
                    line.setPen(mark_pen(t, style))
                # 먼저 보인다: 이름표(InfLineLabel)는 숨은 동안 글자·위치를 고치지 않고, 선을 다시 보일 때도 고치지 않는다
                # (숨겼던 선을 새로 쓰면 이름표가 비거나 앞 표시의 글자가 남는다)
                line.show()
                line.setPos(x)
                line.setToolTip(tip)
                if self.mark_labels and k == 0 and line.label is not None:
                    # 이름표 글자는 str.format 틀로 쓰이므로 중괄호를 겹친다
                    line.label.setFormat(label.replace("{", "{{").replace("}", "}}"))
            applied[j] = want

    def legend_html(self, i: int | None) -> str:
        parts: list[str] = []
        if i is not None:
            parts.append(self.dash.x_text(i, self.follows_axis))
        for e in self.entries():
            swatch = f'<span style="color:{e.color}">{e.glyph}</span>'
            value = ""
            if i is not None and e.text is not None:
                value = " " + e.text(i)
            parts.append(f"{swatch}&nbsp;{_escape(e.name)}{_escape(value)}")
        return "&nbsp;&nbsp; ".join(parts)

    def update_legend(self) -> None:
        html = self.legend_html(self._hover_index)
        if html != self._legend_text:
            self._legend_text = html
            self.legend.setText(html)

    # -- 마우스를 올리면 그 세대 값 -------------------------------------------
    def _on_mouse(self, args: tuple[Any, ...]) -> None:
        pos = args[0]
        for k, plot in enumerate(self.plots):
            if plot.sceneBoundingRect().contains(pos):
                pt = plot.vb.mapSceneToView(pos)
                self.hover(pt.x(), pt.y(), k)
                return
        self.unhover()

    def hover(self, xv: float, yv: float, plot_index: int) -> None:
        x = self.x_values()
        if len(x) == 0 or not np.isfinite(x).any():
            self.unhover()
            return
        i = int(np.nanargmin(np.abs(x - xv)))
        for line in self._cross:
            line.setPos(float(x[i]))
            line.show()
        self._hover_index = i
        self.update_legend()

    def unhover(self) -> None:
        for line in self._cross:
            line.hide()
        self._hover_index = None
        self.update_legend()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Leave:
            self.unhover()
        return super().eventFilter(watched, event)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _curve(pen: QPen | None = None, **kw: Any) -> pg.PlotCurveItem:
    return pg.PlotCurveItem(pen=pen if pen is not None else pg.mkPen(None), connect="finite", **kw)


def _value_text(arr_fn: Callable[[], np.ndarray], scale: float = 1.0) -> Callable[[int], str]:
    def text(i: int) -> str:
        arr = arr_fn()
        return fmt_value(float(arr[i]) * scale) if 0 <= i < len(arr) else "-"

    return text


class _TrendTile(_Tile):
    """점수 추이: 최고·평균·중앙값·최저, 25~75% 띠, 챔피언 검증 점수(계단), 내 최고 기록, 정체 구간.
    v1.1: 지금 학습 방법 최고 검증 점수(계단, 학습 방법이 한 번이라도 바뀐 실행에서만)와 예선 구간 띠."""

    mark_labels = True
    LINES = (
        ("score_min", S.DASH_TREND_MIN, 4, 1.0),
        ("score_mean", S.DASH_TREND_MEAN, 3, 1.2),
        ("score_median", S.DASH_TREND_MEDIAN, 2, 1.5),
        ("score_best", S.DASH_TREND_BEST, 1, 1.5),
    )

    def __init__(self, dash: Dashboard, tile_id: str) -> None:
        super().__init__(dash, tile_id)
        t = self.theme
        self.plot = self.add_plot(S.DASH_SCORE)
        # 25~75% 띠: 위 경계를 따라가다 아래 경계를 거꾸로 돌아오는 닫힌 곡선 하나로 채운다.
        # (FillBetweenItem은 세대마다 경로를 파이썬에서 이어 붙여 1000세대에서 줄마다 4ms쯤 걸린다)
        self.band = pg.PlotCurveItem(pen=pg.mkPen(None), brush=_qcolor(t.series[2], 60), fillLevel="enclosed")
        self.band.setZValue(-10)
        self.plot.addItem(self.band)
        self.lines: dict[str, pg.PlotCurveItem] = {}
        for key, _name, slot, width in self.LINES:
            curve = _curve(_pen(t.series[slot], width))
            self.plot.addItem(curve)
            self.lines[key] = curve
        self.champion = _curve(_pen(t.series[0], 2.5), stepMode="right")
        self.champion.setZValue(10)
        self.plot.addItem(self.champion)
        # 지금 학습 방법의 검증 최고점: 재시작하면 챔피언 선은 평평하게 남고 이 선이 아래에서 다시 오른다.
        # 학습 방법이 처음 것(learner 0)뿐이면 챔피언 선과 같으므로 숨긴다
        self.learner_line = _curve(_pen(t.series[6], 1.2, Qt.PenStyle.DashLine), stepMode="right")
        self.learner_line.setZValue(12)
        self.learner_line.hide()
        self.plot.addItem(self.learner_line)
        self.qualify_band = _shade_region(_qcolor(t.series[6], 30), -35)  # 예선 세대 구간
        self.qualify_band.hide()
        self.plot.addItem(self.qualify_band, ignoreBounds=True)
        self.my_best = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=_pen(t.series[5], 1.5, Qt.PenStyle.DashLine),
            label=S.DASH_MY_BEST,
            labelOpts={"position": 0.02, "color": t.ink2, "fill": _qcolor(t.surface, 210), "anchors": [(0, 1), (0, 1)]},
        )
        self.my_best.setZValue(20)
        self.my_best.hide()
        self.plot.addItem(self.my_best)
        self._regions: list[pg.LinearRegionItem] = []
        self.ranges: list[tuple[float, float]] = []  # 정체 구간(세대 기준)

    def update_data(self) -> None:
        d = self.dash
        x = self.x_values()
        lo, hi = d.col("score_q25"), d.col("score_q75")
        ok = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
        bx, blo, bhi = (x, lo, hi) if ok.all() else (x[ok], lo[ok], hi[ok])  # 빈 값은 건너뛰고 잇는다
        self.band.setData(np.concatenate((bx, bx[::-1])), np.concatenate((bhi, blo[::-1])))
        for key, curve in self.lines.items():
            curve.setData(x, d.col(key))
        self.champion.setData(x, d.col("champion_score"))
        if d.learner_switched:
            self.learner_line.setData(x, d.col("learner_best"))
            self.learner_line.show()
        else:
            self.learner_line.setData(_EMPTY, _EMPTY)
            self.learner_line.hide()
        self._update_qualify_band(x)
        self._update_stagnation()

    def _update_qualify_band(self, x: np.ndarray) -> None:
        rows = self.dash.qualify_rows
        if rows is not None:
            a, b = float(x[rows[0]]), float(x[rows[1]])
            if math.isfinite(a) and math.isfinite(b):
                if tuple(self.qualify_band.getRegion()) != (a, b):
                    self.qualify_band.setRegion((a, b))
                self.qualify_band.show()
                return
        self.qualify_band.hide()

    def set_my_best(self, score: float | None) -> None:
        if score is None or not math.isfinite(score):
            self.my_best.hide()
        else:
            self.my_best.setPos(float(score))
            self.my_best.show()
        self.update_legend()

    def _update_stagnation(self) -> None:
        d = self.dash
        gens = d.col("generation")
        self.ranges = stagnation_ranges(gens, d.col("champion_score"), d.stagnation_generations)
        brush = _qcolor(self.theme.muted, 45)
        while len(self._regions) < len(self.ranges):
            region = _shade_region(brush, -30)
            self.plot.addItem(region, ignoreBounds=True)
            self._regions.append(region)
        for k, region in enumerate(self._regions):
            if k < len(self.ranges):
                a, b = d.x_at_generations(np.array(self.ranges[k]), self.follows_axis)
                if math.isfinite(a) and math.isfinite(b):
                    if tuple(region.getRegion()) != (a, b):
                        region.setRegion((a, b))
                    region.show()
                    continue
            region.hide()

    def entries(self) -> list[_Entry]:
        t, d = self.theme, self.dash
        out = [_Entry(t.series[0], GLYPH_LINE, S.DASH_TREND_CHAMPION, _value_text(lambda: d.col("champion_score")))]
        if d.learner_switched:
            out.append(_Entry(t.series[6], GLYPH_DASH, S.DASH_TREND_LEARNER, _value_text(lambda: d.col("learner_best"))))
        for key, name, slot, _w in reversed(self.LINES):
            out.append(_Entry(t.series[slot], GLYPH_LINE, name, _value_text(lambda k=key: d.col(k))))

        def band(i: int) -> str:
            lo, hi = d.col("score_q25"), d.col("score_q75")
            return f"{fmt_value(float(lo[i]))}~{fmt_value(float(hi[i]))}" if i < len(lo) else "-"

        out.append(_Entry(_blend(t.series[2], t.surface, 0.35), GLYPH_FILL, S.DASH_TREND_BAND, band))
        mb = self.dash.my_best
        if mb is not None:
            out.append(_Entry(t.series[5], GLYPH_DASH, S.DASH_MY_BEST, lambda _i: fmt_value(mb)))
        if d.qualify_rows is not None:
            out.append(_Entry(_blend(t.series[6], t.surface, 0.3), GLYPH_FILL, S.DASH_QUALIFY_LEGEND))
        if self.ranges:
            out.append(_Entry(t.muted, GLYPH_FILL, S.DASH_STAGNATION_LEGEND.format(n=self.dash.stagnation_generations)))
        return out


class _HeatmapTile(_Tile):
    """점수 분포 지도: 가로 세대 × 세로 점수 구간, 색 = 그 구간에 든 개체 비율."""

    follows_axis = False

    def __init__(self, dash: Dashboard, tile_id: str) -> None:
        super().__init__(dash, tile_id)
        t = self.theme
        self.plot = self.add_plot(S.DASH_SCORE)
        self.plot.showGrid(x=False, y=False)
        self.image = pg.ImageItem(axisOrder="col-major")
        self.plot.addItem(self.image)
        # 비율이 작은 칸(흩어진 초기 집단)도 보이도록 낮은 값 쪽에 색 단계를 몰았다
        pos = [0.0, 0.004, 0.05, 0.15, 0.3, 0.6, 1.0]
        colors = [_qcolor(t.surface), *(_qcolor(c) for c in t.ramp)]
        self.cmap = pg.ColorMap(pos, colors)
        self.bar = pg.ColorBarItem(
            values=(0, 1), width=10, colorMap=self.cmap, label=S.DASH_HEATMAP_FRACTION, interactive=False,
            colorMapMenu=False,
        )
        self.bar.axis.setPen(_pen(t.axis, 1))
        self.bar.axis.setTextPen(_pen(t.muted, 1))
        self.bar.setImageItem(self.image, insert_in=self.plot)
        self._first_gen = _NAN
        self._hover_text = ""

    def update_data(self) -> None:
        h = self.dash.hist
        if h.n == 0 or not math.isfinite(h.lo):
            self.image.clear()
            return
        gens = self.dash.col("generation")
        self._first_gen = float(gens[0])
        self.image.setImage(h.matrix(), autoLevels=False, levels=(0.0, 1.0))
        self.image.setRect(QRectF(self._first_gen - 0.5, h.lo, h.n, h.hi - h.lo))

    def entries(self) -> list[_Entry]:
        return [_Entry(self.theme.ramp[3], GLYPH_FILL, S.DASH_HEATMAP_HINT)]

    def hover(self, xv: float, yv: float, plot_index: int) -> None:
        h = self.dash.hist
        if h.n == 0 or not math.isfinite(h.lo) or not math.isfinite(self._first_gen):
            self.unhover()
            return
        i = math.floor(xv - (self._first_gen - 0.5))
        width = (h.hi - h.lo) / h.bins
        b = math.floor((yv - h.lo) / width)
        if not (0 <= i < h.n and 0 <= b < h.bins):
            self.unhover()
            return
        gens = self.dash.col("generation")
        for line in self._cross:
            line.setPos(float(gens[i]))
            line.show()
        pct = float(h.matrix()[i, b]) * 100
        text = S.DASH_HEATMAP_HOVER.format(
            gen=int(gens[i]), lo=fmt_value(h.lo + b * width), hi=fmt_value(h.lo + (b + 1) * width), pct=pct
        )
        self._hover_index = i
        self._hover_text = text
        self.update_legend()

    def legend_html(self, i: int | None) -> str:
        if i is not None and self._hover_text:
            return _escape(self._hover_text)
        return super().legend_html(None)

    def image_array(self) -> np.ndarray:
        """보이는 그림 자료 (세대 수, 구간 수). 자료가 없으면 빈 배열."""
        img = self.image.image
        return np.zeros((0, self.dash.hist.bins)) if img is None else np.asarray(img)


class _DiversityTile(_Tile):
    """탐색 폭과 다양성: 위는 시그마·유전자 표준편차 평균(같은 단위), 아래는 개체 간 평균 거리."""

    def __init__(self, dash: Dashboard, tile_id: str) -> None:
        super().__init__(dash, tile_id)
        t = self.theme
        top = self.add_plot(S.DASH_SPREAD_AXIS, x_axis=False)
        bottom = self.add_plot(S.DASH_DISTANCE_AXIS)
        self.view.ci.layout.setRowStretchFactor(0, 3)
        self.view.ci.layout.setRowStretchFactor(1, 2)
        self.curves = {
            "sigma": _curve(_pen(t.series[0], 1.5)),
            "diversity": _curve(_pen(t.series[1], 1.5)),
            "distance": _curve(_pen(t.series[2], 1.5)),
        }
        top.addItem(self.curves["sigma"])
        top.addItem(self.curves["diversity"])
        bottom.addItem(self.curves["distance"])

    def update_data(self) -> None:
        x = self.x_values()
        for key, curve in self.curves.items():
            curve.setData(x, self.dash.col(key))

    def entries(self) -> list[_Entry]:
        t, d = self.theme, self.dash
        return [
            _Entry(t.series[0], GLYPH_LINE, S.DASH_SIGMA, _value_text(lambda: d.col("sigma"))),
            _Entry(t.series[1], GLYPH_LINE, S.DASH_DIVERSITY, _value_text(lambda: d.col("diversity"))),
            _Entry(t.series[2], GLYPH_LINE, S.DASH_DISTANCE, _value_text(lambda: d.col("distance"))),
        ]


class _StackTile(_Tile):
    """쌓은 그래프의 공통 부분. 계열마다 0에서 누적값까지 채운 곡선을 겹쳐 그린다
    (작은 누적값이 위에 오도록 z를 주어, 불투명 색으로 쌓은 모양이 된다)."""

    step = False  # True면 세대마다 막대(계단), False면 영역

    def __init__(self, dash: Dashboard, tile_id: str, y_label: str) -> None:
        super().__init__(dash, tile_id)
        self.plot = self.add_plot(y_label)
        self._series: list[tuple[str, str]] = []
        self._pos: list[pg.PlotCurveItem] = []
        self._neg: list[pg.PlotCurveItem] = []
        self._values = np.zeros((0, 0))

    def _rebuild_items(self, count: int, negative: bool) -> None:
        for curve in (*self._pos, *self._neg):
            self.plot.removeItem(curve)
        self._pos, self._neg = [], []
        t = self.theme
        for j in range(count):
            color = t.series[j % len(t.series)]
            for pool, use in ((self._pos, True), (self._neg, negative)):
                if not use:
                    continue
                # stepMode는 setData 때 준다(빈 자료로 만들면 계단 모드 길이 검사에 걸림)
                curve = pg.PlotCurveItem(pen=_pen(t.surface, 1), brush=_qcolor(color), fillLevel=0.0, connect="all")
                curve.setZValue(count - j)
                self.plot.addItem(curve)
                pool.append(curve)

    def set_stack(self, series: list[tuple[str, str]], values: np.ndarray, x: np.ndarray, negative: bool) -> None:
        """series: (id, 이름), values: (계열, 세대) — NaN은 0으로 본다."""
        series, values = _fold(series, values)
        if [s for s, _ in series] != [s for s, _ in self._series] or len(self._neg) != (len(series) if negative else 0):
            self._rebuild_items(len(series), negative)
        self._series = series
        self._values = values
        mode = "center" if self.step else None
        if values.shape[1] == 0:
            empty_x = np.zeros(1 if self.step else 0)
            for curve in (*self._pos, *self._neg):
                curve.setData(empty_x, np.zeros(0), stepMode=mode)
            return
        v = np.nan_to_num(values)
        pos = np.cumsum(np.clip(v, 0.0, None), axis=0)
        for j, curve in enumerate(self._pos):
            curve.setData(x, pos[j], stepMode=mode)
        if negative:
            neg = np.cumsum(np.clip(v, None, 0.0), axis=0)
            for j, curve in enumerate(self._neg):
                curve.setData(x, neg[j], stepMode=mode)

    def stack_entries(self, scale: float = 1.0) -> list[_Entry]:
        t = self.theme
        out = []
        for j, (_sid, name) in enumerate(self._series):

            def text(i: int, j: int = j) -> str:
                v = self._values
                return fmt_value(float(v[j, i]) * scale) if i < v.shape[1] else "-"

            out.append(_Entry(t.series[j % len(t.series)], GLYPH_FILL, name, text))
        return out


class _BlocksTile(_StackTile):
    """점수 블록 기여도: 세대 대표 또는 집단 평균의 가중치 곱한 블록 점수를 세대마다 쌓은 막대.
    양수는 0 위로, 음수는 0 아래로 따로 쌓고, 합계를 선으로 겹친다."""

    follows_axis = False
    step = True

    def __init__(self, dash: Dashboard, tile_id: str) -> None:
        super().__init__(dash, tile_id, S.DASH_BLOCKS_AXIS)
        self.mode_box = QComboBox()
        for mode in BLOCK_MODES:
            self.mode_box.addItem(S.DASH_BLOCKS_MODES[mode], mode)
        self.mode_box.setToolTip(S.DASH_BLOCKS_MODE_TOOLTIP)
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        self.controls.addWidget(self.mode_box)
        self.message.setText(S.DASH_BLOCKS_NONE)
        self.zero = pg.InfiniteLine(pos=0.0, angle=0, movable=False, pen=_pen(self.theme.ink2, 1))
        self.zero.setZValue(30)
        self.plot.addItem(self.zero)
        self.total = _curve(_pen(self.theme.ink, 1.2))
        self.total.setZValue(35)
        self.plot.addItem(self.total)
        self._total = np.zeros(0)

    @property
    def mode(self) -> str:
        return str(self.mode_box.currentData())

    def set_mode(self, mode: str) -> None:
        idx = self.mode_box.findData(mode)
        if idx >= 0:
            self.mode_box.setCurrentIndex(idx)

    def _mode_changed(self, _index: int) -> None:
        self.dash.save_settings()
        self.refresh()

    def block_series(self) -> list[tuple[str, str]]:
        """그릴 블록 (id, 이름): 환경 정보 순서, 그 뒤에 정보에 없던 열."""
        d = self.dash
        prefixes = ("contrib_rep_", "contrib_mean_")
        found: list[str] = []
        for name in d.table.names():
            for p in prefixes:
                if name.startswith(p) and name[len(p) :] not in found:
                    found.append(name[len(p) :])
        labels = dict(d.block_meta)
        ordered = [b for b, _ in d.block_meta if b in found] + [b for b in found if b not in labels]
        return [(b, labels.get(b, b)) for b in ordered]

    def update_data(self) -> None:
        d = self.dash
        series = self.block_series()
        if not series and d.table.n > 0:
            self.stack.setCurrentWidget(self.message)
            self.set_stack([], np.zeros((0, 0)), np.zeros(0), True)
            self.total.setData([], [])
            return
        self.stack.setCurrentWidget(self.view)
        prefix = f"contrib_{self.mode}_"
        gens = d.col("generation")
        values = np.vstack([d.col(prefix + b) for b, _ in series]) if series else np.zeros((0, d.table.n))
        self.set_stack(series, values, _step_edges(gens), True)
        if series:
            all_nan = np.isnan(values).all(axis=0)
            self._total = np.where(all_nan, _NAN, np.nansum(values, axis=0))
        else:
            self._total = np.full(d.table.n, _NAN)
        self.total.setData(gens, self._total)

    def entries(self) -> list[_Entry]:
        out = self.stack_entries()
        if self._series:
            out.append(_Entry(self.theme.ink, GLYPH_LINE, S.DASH_BLOCKS_TOTAL, _value_text(lambda: self._total)))
        return out


class _EndsTile(_StackTile):
    """종료 원인 비율: 환경 파일의 종료 조건 순서대로 시험 비율을 쌓은 영역(%)."""

    def __init__(self, dash: Dashboard, tile_id: str) -> None:
        super().__init__(dash, tile_id, S.DASH_ENDS_AXIS)
        self.fix_y(self.plot, 0, 100)

    def end_series(self) -> list[tuple[str, str]]:
        d = self.dash
        found = [n[4:] for n in d.table.names() if n.startswith("end_")]
        labels = dict(d.end_meta)
        ordered = [e for e, _ in d.end_meta] + [e for e in found if e not in labels]
        return [(e, labels.get(e, e)) for e in ordered]

    def update_data(self) -> None:
        d = self.dash
        series = self.end_series()
        values = np.vstack([d.col("end_" + e) * 100.0 for e, _ in series]) if series else np.zeros((0, d.table.n))
        self.set_stack(series, values, self.x_values(), False)

    def entries(self) -> list[_Entry]:
        out = self.stack_entries()
        for e in out:
            if e.text is not None:
                inner = e.text
                e.text = lambda i, f=inner: f(i) + "%"
        return out


class _PerfTile(_Tile):
    """처리 성능: 세대당 시간, 초당 스텝, 워커 사용률(단위가 달라 그래프를 위아래로 나눔)."""

    SERIES = (
        ("gen_seconds", S.DASH_PERF_GEN_SECONDS, 1.0),
        ("steps_per_second", S.DASH_PERF_STEPS, 1.0),
        ("worker_utilization", S.DASH_PERF_UTIL, 100.0),
    )

    def __init__(self, dash: Dashboard, tile_id: str) -> None:
        super().__init__(dash, tile_id)
        t = self.theme
        self.curves: dict[str, pg.PlotCurveItem] = {}
        for k, (key, _name, _scale) in enumerate(self.SERIES):
            plot = self.add_plot(S.DASH_PERF_AXES[key], x_axis=k == len(self.SERIES) - 1)
            curve = _curve(_pen(t.series[k], 1.5))
            plot.addItem(curve)
            self.curves[key] = curve
        util = self.plots[-1]
        self.fix_y(util, 0, 100)

    def update_data(self) -> None:
        x = self.x_values()
        for key, _name, scale in self.SERIES:
            y = self.dash.col(key)
            self.curves[key].setData(x, y * scale if scale != 1.0 else y)

    def entries(self) -> list[_Entry]:
        t, d = self.theme, self.dash
        return [
            _Entry(t.series[k], GLYPH_LINE, name, _value_text(lambda key=key: d.col(key), scale))
            for k, (key, name, scale) in enumerate(self.SERIES)
        ]


_TILE_CLASSES: dict[str, type[_Tile]] = {
    "trend": _TrendTile,
    "heatmap": _HeatmapTile,
    "diversity": _DiversityTile,
    "blocks": _BlocksTile,
    "ends": _EndsTile,
    "perf": _PerfTile,
}
def tile_class(tile_id: str) -> type[_Tile]:
    """타일 id → 타일 클래스. 「보는 재미」 타일 모듈(watch_tiles, brain_view)은 여기서 늦게 불러온다(그 모듈들이
    _Tile을 상속하려고 이 모듈을 불러오므로 위에서 불러오면 순환한다)."""
    cls = _TILE_CLASSES.get(tile_id)
    if cls is not None:
        return cls
    if tile_id == "brain":
        from evolab.app.lab.brain_view import BrainTile

        return BrainTile
    from evolab.app.lab import watch_tiles

    return watch_tiles.TILE_CLASSES[tile_id]


class _PageTabs(QTabBar):
    """쪽 탭. bold_index 탭(아직 한 번도 고르지 않은 「보는 재미」)의 글자를 굵게 그린다."""

    def __init__(self) -> None:
        super().__init__()
        self.bold_index = -1
        self._bold = QFont(self.font())
        self._bold.setBold(True)

    def set_bold_index(self, index: int) -> None:
        if index != self.bold_index:
            self.bold_index = index
            self.updateGeometry()
            self.update()

    def tabSizeHint(self, index: int) -> QSize:  # noqa: N802
        size = super().tabSizeHint(index)
        if index == self.bold_index:
            text = self.tabText(index)
            extra = QFontMetrics(self._bold).horizontalAdvance(text) - self.fontMetrics().horizontalAdvance(text)
            size.setWidth(size.width() + max(0, extra))
        return size

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        if not 0 <= self.bold_index < self.count():
            super().paintEvent(event)
            return
        painter = QStylePainter(self)
        order = [i for i in range(self.count()) if i != self.currentIndex()] + [self.currentIndex()]
        for i in order:  # 고른 탭을 마지막에(기본 그리기와 같은 순서)
            opt = QStyleOptionTab()
            self.initStyleOption(opt, i)
            painter.setFont(self._bold if i == self.bold_index else self.font())
            painter.drawControl(QStyle.ControlElement.CE_TabBarTab, opt)




# ---------------------------------------------------------------------------
GEOMETRY_CACHE = 8
"""세계 그림(set_world_geometry)을 세계 열쇠로 기억해 두는 수."""


class Dashboard(QWidget):
    """통계 대시보드. set_rows로 실행 하나를 열고, 학습 중에는 add_row로 세대마다 한 줄씩 더한다.

    v1.2 「보는 재미」 쪽(설계 7.3의 계약): set_watch_meta(실행의 이름표) → 실행을 열 때 load_history(배경으로 읽기),
    학습 중에는 세대마다 add_generation. set_focus는 실험실 타임라인이 고른 보는 세대(None = 최신 따라가기).
    궤적 밀도 타일은 geometry_needed로 세계 그림을 청하고 실험실이 set_world_geometry로 준다. 뇌 그림은
    set_brain_source·show_time. exact_genome·rep_genome·champion_at은 지도 점 재생과 뇌 그림이 쓰는 유전자.
    """

    focus_requested = Signal(int)
    """유전자 지도의 잔상 점(앞 세대)을 눌렀다: 그 세대를 보는 세대로(실험실이 타임라인을 옮긴다)."""
    individual_requested = Signal(int, int)
    """유전자 지도 점을 두 번 눌렀다: (세대, 개체 번호). 개체 번호 -1은 집단 기록이 없는 세대의 세대 대표 점이다."""
    geometry_needed = Signal(int, str)
    """궤적 밀도 타일이 세계 그림을 원한다: (세대, 세계 열쇠). 같은 열쇠는 한 번만 청한다."""

    def __init__(self, parent: QWidget | None = None, settings: QSettings | None = None) -> None:
        super().__init__(parent)
        from collections import OrderedDict

        from evolab.app.lab.history import RunHistory

        self.tile_ids: list[str] = list(TILE_IDS)
        """「학습 그래프」 쪽 타일 id(v1.1과 같은 값). 모든 타일은 ALL_TILE_IDS."""
        if settings is None:
            settings = QSettings(str(paths.user_state_dir() / SETTINGS_FILE), QSettings.Format.IniFormat)
        self._settings = settings
        self.theme = theme_for(self)
        self.table = StatsTable()
        self.hist = ScoreHistogram()
        self.axis = "generation"
        self.marks: list[tuple[int, str]] = []
        # 자동 세로선 (세대, 종류, 풍선 도움말 값): 통계 줄의 switch 열에서 읽는다(v1.1, 옛 실행은 없음)
        self.auto_marks: list[tuple[int, str, dict[str, float]]] = []
        self._auto_rows: list[int] = []  # 자동 세로선마다 그 세대의 줄 번호(가로축 값 찾기)
        self._auto_tips: list[str] = []
        self.learner_max = _NAN  # learner 열 최댓값(옛 실행은 NaN)
        self.qualify_rows: tuple[int, int] | None = None  # qualifying == 1인 첫 줄과 마지막 줄 번호
        self.my_best: float | None = None
        self.end_meta: list[tuple[str, str]] = []
        self.block_meta: list[tuple[str, str]] = []
        self.stagnation_generations = DEFAULT_STAGNATION
        self._orders: dict[str, list[str]] = {p: list(ids) for p, ids in TILE_PAGES.items()}
        self._hidden: set[str] = set()
        self._columns = DEFAULT_COLUMNS
        self._loading = True  # 설정을 읽는 중(저장하지 않음)

        # v1.2 「보는 재미」
        self.page = "learning"
        """지금 쪽: "learning" | "fun"."""
        self.fun_seen = False
        """「보는 재미」 쪽을 한 번이라도 골랐는가(고를 때까지 탭 글자가 굵다)."""
        self.watch_meta: WatchMeta | None = None
        self.brain_source: BrainSource | None = None
        self.focus: int | None = None
        """보는 세대(None = 최신 따라가기)."""
        self.play_time = 0.0
        """show_time으로 받은 마지막 재생 시각(초)."""
        self.flipping = False
        """「세대 넘겨 보기」 중(유전자 지도 축을 다시 맞추지 않음)."""
        self.history: RunHistory = RunHistory()
        self.history_loading: tuple[int, int] | None = None
        """배경으로 기록을 읽는 중이면 (읽은 세대 수, 전체 세대 수)."""
        self.pump_steps = 0
        """갱신 펌프가 갱신한 타일 수(측정·테스트용)."""
        self._geometries: OrderedDict[str, WorldGeometry] = OrderedDict()
        self._load_tag = 0
        self._loader: Any = None
        self._pending_load: tuple[Path, list[dict[str, Any]]] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 0, 4, 0)
        self.page_tabs = _PageTabs()
        self.page_tabs.setDrawBase(False)
        self.page_tabs.setExpanding(False)
        for page in PAGES:
            self.page_tabs.addTab(S.DASH_PAGES[page])
        self.page_tabs.currentChanged.connect(self._page_tab_changed)
        bar.addWidget(self.page_tabs)
        bar.addSpacing(8)
        bar.addWidget(QLabel(S.DASH_AXIS_LABEL))
        self.axis_box = QComboBox()
        for a in AXES:
            self.axis_box.addItem(S.DASH_AXIS_NAMES[a], a)
        self.axis_box.setToolTip(S.DASH_AXIS_TOOLTIP)
        self.axis_box.currentIndexChanged.connect(lambda _i: self.set_axis(str(self.axis_box.currentData())))
        bar.addWidget(self.axis_box)
        bar.addStretch(1)
        self.tiles_button = QPushButton(S.DASH_TILES_BUTTON)
        self.tiles_button.setToolTip(S.DASH_TILES_TOOLTIP)
        self.tiles_menu = QMenu(self.tiles_button)
        self.tiles_menu.aboutToShow.connect(self._fill_tiles_menu)
        self.tiles_button.setMenu(self.tiles_menu)
        bar.addWidget(self.tiles_button)
        root.addLayout(bar)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(6)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setWidget(self._grid_host)
        root.addWidget(self.scroll, 1)
        # 스크롤·크기 바뀜·다시 그리기로 dirty 타일이 화면에 들어오면 펌프를 돌린다
        self.scroll.verticalScrollBar().valueChanged.connect(self._schedule_pump)
        self.scroll.horizontalScrollBar().valueChanged.connect(self._schedule_pump)
        self._grid_host.installEventFilter(self)

        self._pump_timer = QTimer(self)
        self._pump_timer.setSingleShot(True)
        self._pump_timer.setInterval(0)
        self._pump_timer.timeout.connect(self._pump_step)

        self._tiles: dict[str, _Tile] = {tid: tile_class(tid)(self, tid) for tid in ALL_TILE_IDS}
        for tid in FUN_TILE_IDS:
            help_text = S.DASH_TILE_HELP.get(tid)
            if help_text:
                self._tiles[tid].title.setToolTip(help_text)
        self._grid_rows = 0
        self._load_settings()
        self._loading = False
        self._sync_page_tabs()
        self._relayout()
        self._refresh_all()

    # -- 자료 넣기 ------------------------------------------------------------
    def set_meta(
        self,
        *,
        end_events: list[tuple[str, str]],
        blocks: list[tuple[str, str]],
        stagnation_generations: int | None = None,
    ) -> None:
        """환경 정보: 종료 사건과 점수 블록의 (id, 한국어 이름), 정체 판단 세대 수(None이면 50)."""
        self.end_meta = [(str(a), str(b)) for a, b in end_events]
        self.block_meta = [(str(a), str(b)) for a, b in blocks]
        n = stagnation_generations if stagnation_generations is not None else DEFAULT_STAGNATION
        self.stagnation_generations = max(1, int(n))
        self._refresh_all()

    def set_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """실행을 열 때: 지금 자료를 버리고 통계 줄 전체로 다시 그린다."""
        self.table.clear()
        self.hist.clear()
        self._clear_learner()
        for row in rows:
            self._append(row, bulk=True)
        self.hist.finish()
        for tile in self._tiles.values():
            tile.reset_view()
        self._refresh_all()

    def add_row(self, row: Mapping[str, Any]) -> None:
        """학습 중 한 세대가 끝날 때마다: 줄 하나를 더하고 보이는 타일을 갱신한다(떠 있으면 펌프로)."""
        self._append(row, bulk=False)
        self._refresh_all()

    def set_marks(self, marks: Sequence[tuple[int, str]]) -> None:
        """세로선을 그을 (세대, 사건 종류): resume, env_change, config_change, branch.
        학습 방법 교체 세로선(auto_marks)은 통계 줄의 switch 열에서 따로 읽으므로 여기에 넣지 않는다."""
        self.marks = [(int(g), str(k)) for g, k in marks]
        self._refresh_all(marks_only=True)

    def set_my_best(self, score: float | None) -> None:
        """직접 조종 최고 점수 가로선(None이면 숨김)."""
        self.my_best = None if score is None or not math.isfinite(float(score)) else float(score)
        tile = self._tiles["trend"]
        assert isinstance(tile, _TrendTile)
        tile.set_my_best(self.my_best)

    def set_axis(self, axis: str) -> None:
        """가로축: "generation" | "evaluations" | "elapsed"."""
        if axis not in AXES:
            raise ValueError(f"unknown axis {axis!r}")
        if self.axis_box.currentData() != axis:
            self.axis_box.blockSignals(True)
            self.axis_box.setCurrentIndex(AXES.index(axis))
            self.axis_box.blockSignals(False)
        if axis == self.axis:
            return
        self.axis = axis
        for tile in self._tiles.values():
            tile.axis_changed()
        self._refresh_all()
        self.save_settings()

    def clear(self) -> None:
        """자료와 세로선을 비운다(환경 정보, 관찰 이름표, 내 최고 기록, 배치는 그대로). 「보는 재미」 기록,
        보는 세대, 뇌 원천도 비운다(새 실행)."""
        self.table.clear()
        self.hist.clear()
        self.marks = []
        self._clear_learner()
        self._clear_history()
        self.focus = None
        self.brain_source = None
        self._refresh_all()

    def _append(self, row: Mapping[str, Any], *, bulk: bool) -> None:
        values = {k: to_float(v) for k, v in row.items() if k != "scores"}
        if not math.isfinite(values.get("generation", _NAN)):
            values["generation"] = float(self.table.n + 1)
        i = self.table.n
        self.table.append(values)
        self.hist.append(parse_scores(row.get("scores")), update=not bulk)
        self._note_learner(i, values)

    # -- 학습 방법 교체(v1.1 자동 재시작·예선) --------------------------------
    @property
    def learner_switched(self) -> bool:
        """학습 방법이 한 번이라도 바뀌었는가(max(learner) > 0). 옛 실행과 바뀐 적 없는 실행은 False."""
        return self.learner_max > 0

    def _clear_learner(self) -> None:
        self.auto_marks = []
        self._auto_rows = []
        self._auto_tips = []
        self.learner_max = _NAN
        self.qualify_rows = None

    def _note_learner(self, i: int, values: Mapping[str, float]) -> None:
        """i번째 줄의 learner·qualifying·switch 열을 읽어 자동 세로선, 예선 구간, learner 최댓값을 늘린다.
        줄마다 O(1)이라 수천 세대 실행에서도 세대마다 부담이 늘지 않는다. 열이 없는 옛 줄은 NaN이라 아무것도 하지 않는다."""
        learner = values.get("learner", _NAN)
        if math.isfinite(learner):
            if not learner <= self.learner_max:  # 첫 값(learner_max가 NaN)이거나 더 큼
                self.learner_max = learner
            # 바로 앞 줄이 예선 끝이면 이 줄을 만든 학습 방법이 통과자다
            if self._auto_rows and self._auto_rows[-1] == i - 1 and self.auto_marks[-1][1] == "qualify_pick":
                gen, kind, info = self.auto_marks[-1]
                info["winner"] = learner
                self._auto_tips[-1] = self._auto_tip(gen, kind, info)
        if values.get("qualifying") == 1.0:
            self.qualify_rows = (i if self.qualify_rows is None else self.qualify_rows[0], i)
        code = values.get("switch", _NAN)
        kind = SWITCH_KINDS.get(int(code)) if math.isfinite(code) and code == int(code) else None
        if kind is None:
            return
        gen = int(values["generation"])
        info = {
            "switch": float(code),
            "learner": learner,
            "learner_stall": values.get("learner_stall", _NAN),
            "learner_best": values.get("learner_best", _NAN),
        }
        self.auto_marks.append((gen, kind, info))
        self._auto_rows.append(i)
        self._auto_tips.append(self._auto_tip(gen, kind, info))

    def _auto_tip(self, gen: int, kind: str, info: Mapping[str, float]) -> str:
        """자동 세로선 풍선 도움말. 필요한 값이 빠진(손상된) 줄은 사건 표시와 같은 짧은 글."""
        best = fmt_value(info.get("learner_best", _NAN))
        learner = info.get("learner", _NAN)
        if kind == "restart":
            stall = info.get("learner_stall", _NAN)
            where = SWITCH_RESTART_FROM.get(int(info.get("switch", 0)))
            if math.isfinite(stall) and where is not None:
                return S.DASH_MARK_TIP_RESTART.format(
                    gen=gen, where=S.RESTART_FROM_NAMES_SHORT[where], stall=int(stall), best=best
                )
        elif kind == "qualify_next":
            if math.isfinite(learner):
                return S.DASH_MARK_TIP_QUALIFY_NEXT.format(gen=gen, n=int(learner) + 1, best=best)
        elif kind == "qualify_pick":
            winner = info.get("winner", _NAN)
            if not math.isfinite(winner):
                winner = self._qualify_leader()
            if math.isfinite(winner):
                return S.DASH_MARK_TIP_QUALIFY_PICK.format(gen=gen, winner=int(winner) + 1)
        return S.DASH_MARK_TOOLTIP.format(gen=gen, label=S.DASH_MARK_LABELS.get(kind, kind))

    def _qualify_leader(self) -> float:
        """다음 줄이 아직 없을 때의 통과자: 마지막 예선 끝 표시까지 이어진 예선 표시(출전마다 마지막 세대)의
        learner_best 가운데 첫 최댓값을 낸 learner(동점이면 앞 번호, 명세 A28). 다음 줄이 오면 그 줄의 learner로 고친다."""
        k = len(self.auto_marks) - 1
        while k > 0 and self.auto_marks[k - 1][1] == "qualify_next":
            k -= 1
        top, leader = -math.inf, _NAN
        for _gen, _kind, info in self.auto_marks[k:]:
            b = info.get("learner_best", _NAN)
            if math.isfinite(b) and b > top:
                top, leader = b, info.get("learner", _NAN)
        return leader

    # -- 타일이 쓰는 자료 -----------------------------------------------------
    def col(self, name: str) -> np.ndarray:
        return self.table.col(name)

    def x_values(self, follows_axis: bool = True) -> np.ndarray:
        if not follows_axis or self.axis == "generation":
            return self.table.col("generation")
        return self.table.col(self.axis)

    def x_at_generations(self, gens: np.ndarray, follows_axis: bool = True) -> np.ndarray:
        """세대 번호 → 지금 가로축 값. 자료 범위 밖 세대는 NaN."""
        g = np.asarray(gens, dtype=np.float64)
        all_gens = self.table.col("generation")
        if len(all_gens) == 0 or len(g) == 0:
            return np.full(len(g), _NAN)
        out = np.where((g >= all_gens[0]) & (g <= all_gens[-1]), g, _NAN)
        if not follows_axis or self.axis == "generation":
            return out
        idx = np.clip(np.searchsorted(all_gens, g), 0, len(all_gens) - 1)
        x = self.table.col(self.axis)[idx]
        return np.where(np.isfinite(out), x, _NAN)

    def row_value(self, generation: int, column: str) -> float:
        """그 세대 통계 줄의 열 값(줄이나 값이 없으면 NaN)."""
        gens = self.table.col("generation")
        i = int(np.searchsorted(gens, float(generation)))
        if i < len(gens) and gens[i] == generation:
            return float(self.table.col(column)[i])
        return _NAN

    def mark_positions(self, follows_axis: bool) -> tuple[np.ndarray, list[str], list[str], list[str]]:
        """세로선 (가로축 값, 풍선 도움말, 이름, 종류). 사건 표시(set_marks) 뒤에 자동 표시(switch 열)를 잇는다.
        자료 범위 밖 세대는 NaN. 자동 표시는 그 세대 줄의 가로축 값(세대 끝의 누적 시험 횟수·경과 시간)에 긋는다."""
        kinds = [k for _, k in self.marks]
        labels = [S.DASH_MARK_LABELS.get(k, k) for k in kinds]
        tips = [S.DASH_MARK_TOOLTIP.format(gen=g, label=label) for (g, _), label in zip(self.marks, labels, strict=True)]
        xs = self.x_at_generations(np.array([g for g, _ in self.marks], dtype=np.float64), follows_axis)
        if self.auto_marks:
            auto_kinds = [k for _, k, _ in self.auto_marks]
            kinds += auto_kinds
            labels += [S.DASH_MARK_LABELS.get(k, k) for k in auto_kinds]
            tips += self._auto_tips
            rows = np.asarray(self._auto_rows, dtype=np.int64)
            xs = np.concatenate((xs, self.x_values(follows_axis)[rows]))
        return xs, tips, labels, kinds

    def x_text(self, i: int, follows_axis: bool) -> str:
        gens = self.table.col("generation")
        if not 0 <= i < len(gens):
            return ""
        axis = self.axis if follows_axis else "generation"
        x = float(self.table.col(axis)[i])
        return S.DASH_HOVER_X[axis].format(gen=int(gens[i]), x=fmt_value(x))

    # -- 「보는 재미」 자료(설계 7.3) ------------------------------------------
    def set_watch_meta(self, meta: WatchMeta) -> None:
        """실행의 이름표(유전자 구조, 몸 수치, 구동기, 신경망 입출력 이름). 유전자 구조가 바뀌면 기록을 비운다.
        load_history가 이름표보다 먼저 왔으면 여기서 읽기를 시작한다."""
        old = self.watch_meta
        self.watch_meta = meta
        if old is None or meta is None or old.layout.signature() != meta.layout.signature():
            pending = self._pending_load
            self._clear_history()
            self._pending_load = pending
        self.history.set_layout(meta.layout if meta is not None else None)
        for tid in FUN_TILE_IDS:
            hook = getattr(self._tiles[tid], "meta_changed", None)
            if callable(hook):
                hook()
        if self._pending_load is not None and meta is not None:
            run_dir, rows = self._pending_load
            self._start_loader(run_dir, rows)
        self._invalidate(FUN_TILE_IDS)

    def load_history(self, run_dir: Path | None, rows: Sequence[Mapping[str, Any]]) -> None:
        """실행을 열 때: 지금 기록을 버리고 run_dir의 세대 대표 파일(관찰 기록)을 배경 스레드로 읽기 시작한다.
        rows는 그 실행의 통계 줄(관찰의 일관성 검사 D10과, 통계보다 뒤인 세대를 거르는 데 쓴다). None이면 비우기만."""
        self._clear_history()
        if run_dir is None:
            self._invalidate(FUN_TILE_IDS)
            return
        rows_copy = [dict(r) for r in rows]
        if self.watch_meta is None:
            self._pending_load = (Path(run_dir), rows_copy)
        else:
            self._start_loader(Path(run_dir), rows_copy)
        self._invalidate(FUN_TILE_IDS)

    def _start_loader(self, run_dir: Path, rows: list[dict[str, Any]]) -> None:
        from evolab.app.lab.history import HistoryLoader

        assert self.watch_meta is not None
        self._pending_load = None
        self.history.run_dir = run_dir
        loader = HistoryLoader(run_dir, rows, self.watch_meta.layout, self._load_tag)
        loader.chunk.connect(self._on_history_chunk)
        loader.champions.connect(self._on_history_champions)
        loader.loaded.connect(self._on_history_loaded)
        self._loader = loader
        self.history_loading = (0, len(rows))
        loader.start()

    def _stop_loader(self) -> None:
        if self._loader is not None:
            self._loader.requestInterruption()
            self._loader = None

    def _clear_history(self) -> None:
        self._load_tag += 1  # 아직 도는 앞 읽기의 결과는 버린다(되감기·다른 실행)
        self._stop_loader()
        self._pending_load = None
        self.history_loading = None
        self.history.clear()
        for tid in FUN_TILE_IDS:
            hook = getattr(self._tiles[tid], "history_reset", None) if hasattr(self, "_tiles") else None
            if callable(hook):
                hook()

    def _on_history_chunk(self, tag: int, entries: list[Any], done: int, total: int) -> None:
        if tag != self._load_tag:
            return
        self.history.add_many(entries)
        self.history_loading = (done, total)
        self._invalidate(FUN_TILE_IDS)

    def _on_history_champions(self, tag: int, champions: list[tuple[int, np.ndarray, float]]) -> None:
        if tag != self._load_tag:
            return
        for gen, genome, score in champions:
            self.history.add_champion(gen, genome, score)
        self._invalidate(FUN_TILE_IDS)

    def _on_history_loaded(self, tag: int, summary: dict[str, int]) -> None:
        if tag != self._load_tag:
            return
        self.history.load_summary = dict(summary)
        self.history_loading = None
        self._loader = None
        self._invalidate(FUN_TILE_IDS)

    def wait_history(self, timeout_ms: int = 10000) -> bool:
        """테스트·측정: 배경 읽기가 끝날 때까지 이벤트를 처리하며 기다린다. 끝났으면 True."""
        from PySide6.QtCore import QCoreApplication, QElapsedTimer

        clock = QElapsedTimer()
        clock.start()
        while self.history_loading is not None or self._pending_load is not None:
            if clock.elapsed() > timeout_ms:
                return False
            QCoreApplication.processEvents()
            if self._loader is not None:
                self._loader.wait(5)
        return True

    def add_generation(
        self,
        gen: int,
        observation: dict[str, np.ndarray] | None,
        population: np.ndarray | None,
        representative: int,
        champion: tuple[int, np.ndarray] | None,
    ) -> None:
        """학습 중 세대 하나: 세대 메시지의 관찰 기록(obs_ 키 사전, 없으면 None), 집단 유전자(float64, 최근 64세대만
        화면 메모리에 둔다), 세대 대표 번호, 바뀐 챔피언 (세대, 유전자) 또는 (세대, 유전자, 검증 점수). 점수를 주지 않으면
        그 세대 통계 줄의 champion_score를 쓴다. 이름표(set_watch_meta)가 없으면 무시한다."""
        from evolab.app.lab.history import live_entry

        meta = self.watch_meta
        if meta is None:
            return
        g = int(gen)
        entry, exact = live_entry(g, observation, population, int(representative), meta.layout,
                                  learner=self.row_value(g, "learner"), rep_score=self.row_value(g, "score_best"))
        if entry is not None:
            self.history.add(entry, exact=exact)
        if champion is not None:
            cg, genome = champion[0], champion[1]
            score = champion[2] if len(champion) > 2 else self.row_value(int(cg), "champion_score")  # type: ignore[misc]
            self.history.add_champion(int(cg), genome, score)
        self._invalidate(FUN_TILE_IDS)

    def set_focus(self, gen: int | None) -> None:
        """보는 세대(실험실 타임라인). None이면 최신 세대를 따라간다."""
        g = None if gen is None else int(gen)
        if g == self.focus:
            return
        self.focus = g
        self._invalidate(FUN_TILE_IDS)

    def view_generation(self) -> int | None:
        """타일이 보여 줄 세대: 보는 세대, 따라가는 중이면 기록의 마지막 세대(없으면 None)."""
        return self.focus if self.focus is not None else self.history.latest

    def set_flipping(self, on: bool) -> None:
        """「세대 넘겨 보기」 중이면 True(그동안 유전자 지도 축을 다시 맞추지 않는다)."""
        on = bool(on)
        if on != self.flipping:
            self.flipping = on
            if not on:
                self._invalidate(("genemap",))

    def set_world_geometry(self, world_key: str, geometry: WorldGeometry) -> None:
        """geometry_needed에 답한 세계 그림(세계 열쇠로 최근 GEOMETRY_CACHE개 기억)."""
        self._geometries[str(world_key)] = geometry
        self._geometries.move_to_end(str(world_key))
        while len(self._geometries) > GEOMETRY_CACHE:
            self._geometries.popitem(last=False)
        self._invalidate(("density",))

    def world_geometry(self, world_key: str) -> WorldGeometry | None:
        return self._geometries.get(world_key)

    def set_brain_source(self, src: BrainSource | None) -> None:
        """뇌 들여다보기 타일이 그릴 개체(설계 D24). None이면 타일이 보는 세대까지의 챔피언을 멈춘 그림으로."""
        self.brain_source = src
        self._invalidate(("brain",))

    def show_time(self, t: float) -> None:
        """재생 시각(초): 뇌 타일이 지금 쪽에 있으면 그 타일의 show_time(t)을 부른다(뉴런 색만 바꿈)."""
        self.play_time = float(t)
        if self._active("brain"):
            fn = getattr(self._tiles["brain"], "show_time", None)
            if callable(fn):
                fn(self.play_time)

    def brain_visible(self) -> bool:
        """뇌 그림을 볼 수 있는가: 대시보드가 화면에 떠 있고 「보는 재미」 쪽이며 뇌 타일을 숨기지 않았다."""
        return self.isVisible() and self.page == "fun" and "brain" not in self._hidden

    def exact_genome(self, gen: int, index: int) -> np.ndarray | None:
        """그 세대 index번 개체의 정확한 유전자(float64 복사본). 학습 중 받은 최근 64세대만, 아니면 None."""
        return self.history.exact_genome(gen, index)

    def rep_genome(self, gen: int) -> np.ndarray | None:
        """그 세대 대표의 정확한 유전자(float64): 화면 메모리, 없으면 세대 대표 파일. 없으면 None."""
        return self.history.rep_genome(gen)

    def rep_index(self, gen: int) -> int | None:
        """그 세대 대표의 개체 번호(관찰 기록이 있는 세대만)."""
        entry = self.history.entry(gen)
        return entry.obs.representative if entry is not None and entry.obs is not None else None

    def champion_at(self, gen: int) -> tuple[int, np.ndarray] | None:
        """gen 이하에서 마지막으로 뽑힌 챔피언 (세대, 유전자 float64)."""
        return self.history.champion_at(gen)

    def observation(self, gen: int) -> Observation | None:
        """그 세대의 관찰 기록(없으면 None). 궤적 밀도의 세계 그림(시험 0번 시드·트랙)을 만들 때 쓴다."""
        entry = self.history.entry(gen)
        return entry.obs if entry is not None else None

    # -- 타일 배치 -----------------------------------------------------------
    def tile(self, tile_id: str) -> _Tile:
        return self._tiles[tile_id]

    def tile_order(self, page: str | None = None) -> list[str]:
        """그 쪽(None이면 지금 쪽)의 배치 순서(숨긴 타일 포함)."""
        return list(self._orders[page or self.page])

    def is_tile_visible(self, tile_id: str) -> bool:
        """사용자가 숨기지 않았는가(쪽과 상관없음)."""
        self._check_tile(tile_id)
        return tile_id not in self._hidden

    @property
    def columns(self) -> int:
        return self._columns

    def set_columns(self, columns: int) -> None:
        self._columns = min(MAX_COLUMNS, max(1, int(columns)))
        self._relayout()
        self.save_settings()

    def set_page(self, page: str) -> None:
        """쪽을 바꾼다: "learning"(학습 그래프) | "fun"(보는 재미). 안 보이는 쪽 타일은 갱신하지 않는다."""
        if page not in PAGES:
            raise ValueError(f"unknown page {page!r}")
        if self.page_tabs.currentIndex() != PAGES.index(page):
            self.page_tabs.blockSignals(True)
            self.page_tabs.setCurrentIndex(PAGES.index(page))
            self.page_tabs.blockSignals(False)
        if page == self.page:
            return
        self.page = page
        if page == "fun":
            self.fun_seen = True
        self._sync_page_tabs()
        self._relayout()
        self._refresh_dirty()
        self.save_settings()

    def _page_tab_changed(self, index: int) -> None:
        if 0 <= index < len(PAGES):
            self.set_page(PAGES[index])

    def _sync_page_tabs(self) -> None:
        tabs = self.page_tabs
        if tabs.currentIndex() != PAGES.index(self.page):
            tabs.blockSignals(True)
            tabs.setCurrentIndex(PAGES.index(self.page))
            tabs.blockSignals(False)
        fun = PAGES.index("fun")
        tabs.set_bold_index(-1 if self.fun_seen else fun)
        for i, page in enumerate(PAGES):
            tip = S.DASH_PAGES_TIP[page]
            if page == "fun" and not self.fun_seen:
                tip = f"{tip}\n{S.DASH_FUN_NEW_TIP}"
            tabs.setTabToolTip(i, tip)

    def set_tile_visible(self, tile_id: str, visible: bool) -> None:
        self._check_tile(tile_id)
        if visible:
            self._hidden.discard(tile_id)
        else:
            self._hidden.add(tile_id)
        self._relayout()
        if visible and self._tiles[tile_id].dirty:
            self._invalidate((tile_id,))
        self.save_settings()

    def move_tile(self, tile_id: str, new_index: int) -> None:
        """그 타일 쪽의 배치 순서(숨긴 타일 포함)에서 new_index 자리로 옮긴다."""
        self._check_tile(tile_id)
        order = self._orders[PAGE_OF[tile_id]]
        order.remove(tile_id)
        order.insert(min(max(0, int(new_index)), len(order)), tile_id)
        self._relayout()
        self.save_settings()

    def move_tile_by(self, tile_id: str, step: int) -> None:
        """같은 쪽의 보이는 타일 사이에서 앞(-1)이나 뒤(+1)로 한 칸."""
        if tile_id not in PAGE_OF:
            return
        order = self._orders[PAGE_OF[tile_id]]
        visible = [t for t in order if t not in self._hidden]
        if tile_id not in visible:
            return
        k = visible.index(tile_id) + step
        if 0 <= k < len(visible):
            self.move_tile(tile_id, order.index(visible[k]))

    def reset_layout(self) -> None:
        """두 쪽 모두 기본 배치로(순서, 숨김, 한 줄 그래프 수)."""
        self._orders = {p: list(ids) for p, ids in TILE_PAGES.items()}
        self._hidden = set()
        self._columns = DEFAULT_COLUMNS
        self._relayout()
        self._refresh_dirty()
        self.save_settings()

    def _check_tile(self, tile_id: str) -> None:
        if tile_id not in self._tiles:
            raise KeyError(tile_id)

    def _active(self, tile_id: str) -> bool:
        """지금 쪽에 있고 숨기지 않은 타일(갱신 대상)."""
        return PAGE_OF.get(tile_id) == self.page and tile_id not in self._hidden

    def _relayout(self) -> None:
        for tile in self._tiles.values():
            self._grid.removeWidget(tile)
        for r in range(self._grid_rows):
            self._grid.setRowStretch(r, 0)
        for c in range(MAX_COLUMNS):
            self._grid.setColumnStretch(c, 1 if c < self._columns else 0)
        visible = [t for t in self._orders[self.page] if t not in self._hidden]
        for k, tid in enumerate(visible):
            self._grid.addWidget(self._tiles[tid], k // self._columns, k % self._columns)
            self._tiles[tid].setVisible(True)
        shown = set(visible)
        for tid, tile in self._tiles.items():
            if tid not in shown:
                tile.setVisible(False)
        self._grid_rows = (len(visible) + self._columns - 1) // self._columns
        for r in range(self._grid_rows):
            self._grid.setRowStretch(r, 1)
        self._schedule_pump()

    # -- 갱신(펌프, 설계 D15) -------------------------------------------------
    def _refresh_all(self, *, marks_only: bool = False) -> None:
        self._invalidate(ALL_TILE_IDS, marks_only=marks_only)

    def _invalidate(self, tile_ids: Iterable[str], *, marks_only: bool = False) -> None:
        """타일 자료가 바뀌었다. 지금 쪽이 아니거나 숨긴 타일은 dirty만. 대시보드가 화면에 떠 있으면 dirty로 두고
        펌프가 보이는 것부터 하나씩 갱신하고, 떠 있지 않으면 바로 갱신한다. marks_only(세로선만 바뀜)면 이미 그린
        타일은 세로선만 고친다."""
        shown = self.isVisible()
        queued = False
        for tid in tile_ids:
            tile = self._tiles.get(tid)
            if tile is None:
                continue
            if not self._active(tid):
                tile.dirty = True
            elif marks_only and not tile.dirty:
                tile.sync_marks()
            elif shown:
                tile.dirty = True
                queued = True
            else:
                tile.refresh()
        if queued:
            self._schedule_pump()

    def _refresh_dirty(self) -> None:
        """지금 쪽의 dirty 타일을 갱신한다(떠 있으면 펌프로)."""
        self._invalidate([t for t in self._orders[self.page] if self._tiles[t].dirty])

    def _has_dirty(self) -> bool:
        return any(self._tiles[t].dirty for t in self._orders[self.page] if t not in self._hidden)

    def _schedule_pump(self, *_args: Any) -> None:
        if not self._pump_timer.isActive():
            self._pump_timer.start()

    @staticmethod
    def _in_view(tile: QWidget) -> bool:
        """타일이 화면에 보이는가(스크롤 영역 밖이거나 가려지면 visibleRegion이 비어 있다)."""
        return tile.isVisible() and not tile.visibleRegion().isEmpty()

    def _pump_step(self) -> None:
        """이벤트 루프 차례 하나에 dirty 타일 하나: 지금 쪽 순서에서 화면에 보이는 첫 dirty 타일을 갱신하고 바로 그린다.
        남은 것이 있으면 다음 차례로 미룬다(사이에 입력을 처리). 스크롤 밖 타일은 보일 때까지 dirty로 둔다."""
        if not self.isVisible():
            return
        for tid in self._orders[self.page]:
            tile = self._tiles[tid]
            if tile.dirty and tid not in self._hidden and self._in_view(tile):
                tile.refresh()
                tile.repaint()
                self.pump_steps += 1
                self._schedule_pump()
                return

    def flush(self) -> None:
        """지금 쪽의 dirty 타일을 모두 바로 갱신한다(스크롤 밖 포함, 테스트·화면 저장용)."""
        for tid in self._orders[self.page]:
            tile = self._tiles[tid]
            if tile.dirty and tid not in self._hidden:
                tile.refresh()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        # 스크롤·크기 바뀜·다시 그리기로 타일이 화면에 들어오면 dirty 타일을 펌프로 갱신한다
        if watched is getattr(self, "_grid_host", None) and event.type() in (
            QEvent.Type.Paint, QEvent.Type.Resize, QEvent.Type.Show
        ) and self._has_dirty():
            self._schedule_pump()
        return super().eventFilter(watched, event)

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        if self._has_dirty():
            self._schedule_pump()

    def _fill_tiles_menu(self) -> None:
        menu = self.tiles_menu
        menu.clear()
        for tid in self._orders[self.page]:
            act = QAction(S.DASH_TILE_TITLES[tid], menu)
            act.setCheckable(True)
            act.setChecked(tid not in self._hidden)
            act.toggled.connect(lambda on, t=tid: self.set_tile_visible(t, on))
            menu.addAction(act)
        menu.addSeparator()
        cols = menu.addMenu(S.DASH_TILES_COLUMNS)
        group = QActionGroup(cols)
        for n in range(1, MAX_COLUMNS + 1):
            act = QAction(S.DASH_TILES_COLUMNS_ITEM.format(n=n), cols)
            act.setCheckable(True)
            act.setChecked(n == self._columns)
            act.triggered.connect(lambda _on=False, n=n: self.set_columns(n))
            group.addAction(act)
            cols.addAction(act)
        menu.addSeparator()
        reset = menu.addAction(S.DASH_TILES_RESET)
        reset.triggered.connect(self.reset_layout)

    # -- 설정 저장 ------------------------------------------------------------
    def _load_settings(self) -> None:
        s = self._settings
        s.beginGroup(SETTINGS_GROUP)
        try:
            order = _split_setting(s.value("order"))
            hidden = _split_setting(s.value("hidden"))
            columns = s.value("columns")
            axis = s.value("axis")
            blocks_mode = s.value("blocks_mode")
            page = s.value("page")
            fun_seen = s.value("fun_seen")
            tile_values: dict[str, Any] = {}
            for tid in FUN_TILE_IDS:
                for key in getattr(self._tiles[tid], "SETTING_KEYS", ()):
                    tile_values[key] = s.value(key)
        finally:
            s.endGroup()
        # 쪽마다: 저장된 순서에서 그 쪽 타일만, 빠진 타일(새 버전에서 생긴 타일)은 기본 순서로 뒤에 붙인다
        for p, ids in TILE_PAGES.items():
            seen: list[str] = []
            for tid in order:
                if tid in ids and tid not in seen:
                    seen.append(tid)
            self._orders[p] = seen + [t for t in ids if t not in seen]
        self._hidden = {t for t in hidden if t in ALL_TILE_IDS}
        try:
            self._columns = min(MAX_COLUMNS, max(1, int(str(columns)))) if columns is not None else DEFAULT_COLUMNS
        except ValueError:
            self._columns = DEFAULT_COLUMNS
        if isinstance(axis, str) and axis in AXES:
            self.set_axis(axis)
        blocks = self._tiles["blocks"]
        if isinstance(blocks, _BlocksTile) and isinstance(blocks_mode, str) and blocks_mode in BLOCK_MODES:
            blocks.set_mode(blocks_mode)
        self.fun_seen = str(fun_seen).strip().lower() in ("true", "1")
        if isinstance(page, str) and page in PAGES:
            self.page = page
            if page == "fun":
                self.fun_seen = True
        for tid in FUN_TILE_IDS:
            load = getattr(self._tiles[tid], "load_settings_values", None)
            if callable(load):
                load(tile_values)

    def save_settings(self) -> None:
        if self._loading:
            return
        blocks = self._tiles["blocks"]
        s = self._settings
        s.beginGroup(SETTINGS_GROUP)
        order = [t for p in PAGES for t in self._orders[p]]
        s.setValue("order", ",".join(order))
        s.setValue("hidden", ",".join(t for t in order if t in self._hidden))
        s.setValue("columns", self._columns)
        s.setValue("axis", self.axis)
        if isinstance(blocks, _BlocksTile):
            s.setValue("blocks_mode", blocks.mode)
        s.setValue("page", self.page)
        s.setValue("fun_seen", "true" if self.fun_seen else "false")
        for tid in FUN_TILE_IDS:
            values = getattr(self._tiles[tid], "settings_values", None)
            if callable(values):
                for key, value in values().items():
                    s.setValue(key, value)
        s.endGroup()
        s.sync()

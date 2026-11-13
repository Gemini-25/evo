"""대시보드 「보는 재미」 쪽 타일 (v1.2 설계 4.1·4.2·4.4·4.5, 5.4): 유전자 지도, 몸 수치 추이, 궤적 밀도, 행동 분포.

자료는 Dashboard.history(RunHistory, history.py)에서, 보는 세대는 Dashboard.view_generation()에서 읽는다. 타일은
dashboard._Tile을 상속하고 갱신은 대시보드의 펌프(보이는 dirty 타일을 하나씩)를 따른다. 색은 대시보드 Theme 토큰만
쓴다(궤적 밀도의 관전 캔버스는 테마와 상관없이 늘 어두운 바탕이라 어두운 테마 DARK의 토큰으로 칠한다).
뇌 들여다보기(BrainTile)는 app/lab/brain_view.py에 있다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pyqtgraph as pg
import pyqtgraph.functions as fn
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPaintEngine,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPicture,
    QTransform,
)
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from evolab.app import strings as S
from evolab.app.lab import history as hs
from evolab.app.lab.canvas import SpectatorCanvas
from evolab.app.lab.dashboard import (
    DARK,
    GLYPH_DASH,
    GLYPH_FILL,
    GLYPH_LINE,
    MAX_SERIES,
    Dashboard,
    Theme,
    _blend,
    _curve,
    _Entry,
    _escape,
    _pen,
    _qcolor,
    _Tile,
    fmt_value,
)

GMAP_COLOR_MODES: tuple[str, ...] = ("score", "success")
PICK_PX = 8.0
"""유전자 지도에서 마우스·클릭이 점을 잡는 거리(화면 픽셀)."""
GHOSTS = 3
GHOST_ALPHA = (60, 40, 20)
LUT_SIZE = 64
REFIT_NOTE_MS = 3000
PCA_BIG_VALUES = 4_000_000
"""큰 차원 무작위 PCA 표본의 값 수 상한(행 × 차원, float64 32MB). 행은 PCA_SAMPLE_ROWS 이하."""
DENSITY_STOPS = ((0.0, 1), (0.5, 3), (1.0, None))
"""궤적 밀도 색표: 값 0..1 → DARK.series[1](주황) → DARK.series[3](노랑) → DARK.ink(흰빛). 값이 0인 칸은 투명."""


def _ramp_map(theme: Theme) -> pg.ColorMap:
    """순차 색(ramp) 색표: 0..1을 ramp 색 순서로 고르게(바탕에서 멀어질수록 큰 값)."""
    colors = [_qcolor(c) for c in theme.ramp]
    return pg.ColorMap(np.linspace(0.0, 1.0, len(colors)), colors)


def _heat_map(theme: Theme) -> pg.ColorMap:
    """비율 히트맵 색표(점수 분포 지도와 같은 꼴): 작은 비율도 보이게 낮은 값 쪽에 단계를 몰았다."""
    pos = [0.0, 0.004, 0.05, 0.15, 0.3, 0.6, 1.0]
    return pg.ColorMap(pos, [_qcolor(theme.surface), *(_qcolor(c) for c in theme.ramp)])


def _style_bar(bar: pg.ColorBarItem, theme: Theme) -> None:
    bar.axis.setPen(_pen(theme.axis, 1))
    bar.axis.setTextPen(_pen(theme.tick, 1))


def density_lut() -> np.ndarray:
    """궤적 밀도 색표(256, uint32 ARGB). 0번은 투명."""
    t = DARK
    stops = [(p, QColor(t.series[i]) if i is not None else QColor(t.ink)) for p, i in DENSITY_STOPS]
    alphas = (150, 215, 255)
    out = np.zeros(256, dtype=np.uint32)
    for k in range(1, 256):
        v = k / 255.0
        j = 0 if v <= stops[1][0] else 1
        (p0, c0), (p1, c1) = stops[j], stops[j + 1]
        f = (v - p0) / (p1 - p0)
        a0, a1 = alphas[j], alphas[j + 1]
        r = round(c0.red() + (c1.red() - c0.red()) * f)
        g = round(c0.green() + (c1.green() - c0.green()) * f)
        b = round(c0.blue() + (c1.blue() - c0.blue()) * f)
        a = round(a0 + (a1 - a0) * f)
        out[k] = (a << 24) | (r << 16) | (g << 8) | b
    return out


_DENSITY_LUT = density_lut()


def density_image(counts: np.ndarray) -> QImage:
    """칸별 횟수 (ny, nx) → QImage(ARGB32). 0행 = 격자 iy 0(세계 y0 쪽; 캔버스가 y를 뒤집어 그린다).
    값 = log(1+n)/log(1+최댓값), 0칸은 투명."""
    c = np.asarray(counts, dtype=np.float64)
    ny, nx = c.shape
    top = float(c.max()) if c.size else 0.0
    if top > 0:
        v = np.log1p(c) / math.log1p(top)
        idx = np.where(c > 0, np.clip((v * 255.0).round().astype(np.int64), 1, 255), 0)
    else:
        idx = np.zeros(c.shape, dtype=np.int64)
    argb = np.ascontiguousarray(_DENSITY_LUT[idx])
    img = QImage(argb.data, nx, ny, 4 * nx, QImage.Format.Format_ARGB32)
    return img.copy()


def hline_pixels(t: QTransform, p1: QPointF, p2: QPointF, pen: QPen) -> tuple[int, int, int] | None:
    """코스메틱 펜(폭 1 이하, 계단 현상 그대로) 가로선 하나를 Qt 래스터 엔진(QCosmeticStroker)이 칠하는 픽셀:
    (장치 행, 첫 열, 끝 열(제외)). 칠하는 픽셀이 없으면 None.

    QCosmeticStroker는 장치 좌표를 26.6 고정소수(64배 후 int로 자름)로 바꾸고, 끝이 납작(FlatCap)이 아니면 양 끝을
    반 픽셀(32) 늘린 뒤 열 [(x1 + 32) >> 6, (x2 + 32) >> 6), 행 y1 >> 6을 칠한다(오른쪽에서 왼쪽으로 그은 선은 두
    끝을 바꿔서). 시험(test_lab_latency_v30)이 무작위 선·변환·자르기 영역에서 drawLine과 픽셀까지 같은지 본다."""
    a, b = t.map(p1), t.map(p2)
    x1, x2, y = int(a.x() * 64.0), int(b.x() * 64.0), int(a.y() * 64.0)
    if x1 > x2:
        x1, x2 = x2, x1
    if x1 == x2:  # 26.6에서 길이 0: Qt도 칠하지 않는다
        return None
    if pen.capStyle() != Qt.PenCapStyle.FlatCap:
        x1 -= 32
        x2 += 32
    xs, xe = (x1 + 32) >> 6, (x2 + 32) >> 6
    return (y >> 6, xs, xe) if xs < xe else None


def _plain_hline(pen: QPen, p1: QPointF, p2: QPointF) -> bool:
    """fillRect로 같은 픽셀을 칠할 수 있는 선(코스메틱 실선 폭 1 이하, 가로, 길이가 0이 아님)인가."""
    return (pen.isCosmetic() and pen.widthF() <= 1.0 and pen.style() == Qt.PenStyle.SolidLine
            and pen.brush().style() == Qt.BrushStyle.SolidPattern and p1.y() == p2.y() and p1.x() != p2.x())


def use_fast_grid(axis: pg.AxisItem) -> None:
    """이 축의 격자선(반투명 가로선)을 같은 픽셀의 1픽셀 높이 사각형 채우기로 그리게 한다(v3.0 지연 고침).

    pyqtgraph 축은 격자선을 반투명 펜의 drawLine으로 그리는데, Qt 래스터 엔진은 불투명하지 않은 코스메틱 선을 픽셀마다
    섞어 칠해 불투명한 선의 15배쯤 느리다: 「몸 수치 추이」(0~100%, 높이 185픽셀이면 격자선 55개)에서 축 그림 한 번이
    5~8ms로 타일 그림의 절반이었다. 같은 픽셀을 같은 색 섞기로 칠하는 fillRect는 40배쯤 빠르다. 그림 차례도 같다(축 선과
    눈금 글을 담은 축 그림 다음에 격자선: 원래도 축 선 → 격자선 → 눈금 글이고 글은 격자선과 겹치지 않는다). 래스터가
    아닌 그리기(SVG 내보내기 등), 회전·기울임 변환, 조건에 맞지 않는 눈금이 섞이면 원래 그리기(drawLine)를 쓴다."""
    axis._fast_grid = None  # type: ignore[attr-defined]  # (그림, 격자선 목록): axis.picture가 None이 되면 다시 만든다
    axis.paint = lambda p, *args: _paint_fast_grid(axis, p, *args)


def _paint_fast_grid(axis: pg.AxisItem, p: QPainter, *args: Any) -> None:
    cached = axis._fast_grid  # type: ignore[attr-defined]
    if axis.picture is None or cached is None or cached[0] is not axis.picture:
        picture = QPicture()
        painter = QPainter(picture)
        grid: list[tuple[QPen, QPointF, QPointF]] = []
        try:
            if axis.style["tickFont"]:
                painter.setFont(axis.style["tickFont"])
            specs = axis.generateDrawSpecs(painter)
            if specs is not None:
                axis_spec, ticks, texts = specs
                if ticks and all(_plain_hline(pen, a, b) for pen, a, b in ticks):
                    grid, ticks = list(ticks), []
                axis.drawPicture(painter, axis_spec, ticks, texts)
        finally:
            painter.end()
        axis.picture = picture
        axis._fast_grid = cached = (picture, grid)  # type: ignore[attr-defined]
    picture, grid = cached
    picture.play(p)
    if not grid:
        return
    t = p.deviceTransform()
    engine = p.paintEngine()
    if (t.type().value > QTransform.TransformationType.TxScale.value or engine is None
            or engine.type() != QPaintEngine.Type.Raster or not t.isInvertible()):
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for pen, a, b in grid:  # 원래 그리기
            p.setPen(pen)
            p.drawLine(a, b)
        p.restore()
        return
    inv = t.inverted()[0]
    for pen, a, b in grid:
        hit = hline_pixels(t, a, b, pen)
        if hit is not None:
            row, xs, xe = hit
            # 장치 픽셀 사각형을 논리 좌표로 되돌려 채운다(래스터 엔진이 다시 장치로 옮겨 반올림하면 같은 정수 사각형)
            p.fillRect(inv.mapRect(QRectF(xs, row, xe - xs, 1)), pen.color())


def _focus_line(theme: Theme) -> pg.InfiniteLine:
    line = pg.InfiniteLine(angle=90, movable=False, pen=_pen(theme.ink2, 1, Qt.PenStyle.DotLine))
    line.setZValue(45)
    line.setToolTip(S.DASH_FOCUS_CURSOR)
    line.hide()
    return line


def _band_xy(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """최소~최대 띠: 위 경계를 따라가다 아래 경계를 거꾸로 돌아오는 닫힌 곡선의 점(점수 추이 띠와 같은 방법).
    세 값이 모두 유한한 세대만 쓴다."""
    ok = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
    bx, blo, bhi = x[ok], lo[ok], hi[ok]
    return np.concatenate((bx, bx[::-1])), np.concatenate((bhi, blo[::-1]))


class _SeriesData:
    """_SeriesLayer 계열 하나의 자료 보기(PlotCurveItem.getData처럼 마지막으로 준 (x, y)를 돌려준다, 시험용)."""

    def __init__(self, layer: _SeriesLayer, index: int) -> None:
        self._layer = layer
        self._index = index

    def getData(self) -> tuple[np.ndarray, np.ndarray]:  # noqa: N802 - PlotCurveItem과 같은 이름
        return self._layer.series_data(self._index)

    def isVisible(self) -> bool:  # noqa: N802
        return self._layer.isVisible() and self._layer.is_shown(self._index)


class _SeriesLayer(pg.GraphicsObject):
    """계열 여러 개를 한 그래프 항목으로: kind "line"은 계열마다 선(PlotCurveItem(pen, connect="finite")와 같은 그림),
    "fill"은 닫힌 곡선 채우기(PlotCurveItem(펜 없음, brush, fillLevel="enclosed")와 같은 그림). 위 계열이 나중에 그려진다.

    「몸 수치 추이」는 몸 수치가 15개면 곡선 항목이 31개라, 그릴 때마다 항목마다 파이썬 그리기·경로 만들기·자료 범위
    계산을 되풀이해 750세대 그림 한 번이 20ms 안팎이었다(v3.0 지연 고침). 이 항목은 자료가 바뀐 뒤 처음 그릴 때 계열마다
    pyqtgraph와 같은 함수로 선분·경로를 한 번 만들고, 그리기는 QPainter 호출만 한다. 그리는 QPainter 호출(렌더 힌트, 펜,
    drawLines/drawPath/fillPath와 그 인수)은 PlotCurveItem.paint와 같아 그림이 픽셀까지 같다. 자동 범위용 dataBounds와
    pixelPadding도 보이는 계열을 합친 값이라 ViewBox가 보는 범위가 같다.
    """

    def __init__(self, kind: str) -> None:
        super().__init__()
        if kind not in ("line", "fill"):
            raise ValueError(kind)
        self.kind = kind
        self._aa = bool(pg.getConfigOption("antialias"))
        self._xy: list[tuple[np.ndarray, np.ndarray]] = []
        self._style: list[QPen | QBrush] = []
        self._shown: list[bool] = []
        self._ops: list[tuple[QPen | QBrush, Any]] | None = None
        self._bounds: dict[Any, tuple[float | None, float | None]] = {}
        self._rect: QRectF | None = None
        self._geometry_changed = False
        self._paint_changed = False

    # -- 자료 ---------------------------------------------------------------
    # set_*는 값만 바꾸고 commit()이 한 번에 알린다(계열마다 알리면 ViewBox 범위 다시 맞추기·장면 색인 고침이 계열 수만큼)
    def _style_of(self, style: QPen | QBrush | QColor) -> QPen | QBrush:
        if self.kind == "fill":
            return style if isinstance(style, QBrush) else QBrush(style)
        return style if isinstance(style, QPen) else QPen(style)

    def set_count(self, n: int, style: QPen | QBrush | QColor) -> None:
        """계열 수를 n으로(새 계열은 빈 자료, 보임, style)."""
        style = self._style_of(style)
        empty = np.zeros(0)
        self._xy = self._xy[:n] + [(empty, empty)] * max(0, n - len(self._xy))
        self._style = self._style[:n] + [style] * max(0, n - len(self._style))
        self._shown = self._shown[:n] + [True] * max(0, n - len(self._shown))
        self._geometry_changed = True

    def set_series(self, j: int, x: np.ndarray, y: np.ndarray) -> None:
        self._xy[j] = (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        self._geometry_changed = True

    def set_style(self, j: int, style: QPen | QBrush | QColor) -> None:
        """선이면 QPen(pg.mkPen), 채우기면 QBrush 또는 QColor."""
        style = self._style_of(style)
        if style != self._style[j]:
            self._style[j] = style
            self._paint_changed = True

    def set_shown(self, j: int, shown: bool) -> None:
        if bool(shown) != self._shown[j]:
            self._shown[j] = bool(shown)
            self._geometry_changed = True

    def commit(self) -> None:
        """바뀐 것을 알린다: 자료·보임이 바뀌었으면 범위(prepareGeometryChange·ViewBox 자동 범위)와 다시 그리기,
        펜·붓만 바뀌었으면 다시 그리기."""
        if self._geometry_changed:
            self._bounds.clear()
            self._rect = None
            self.prepareGeometryChange()
            self.informViewBoundsChanged()
        if self._geometry_changed or self._paint_changed:
            self._ops = None
            self.update()
        self._geometry_changed = self._paint_changed = False

    def series_data(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        return self._xy[j]

    def is_shown(self, j: int) -> bool:
        return self._shown[j]

    @property
    def count(self) -> int:
        return len(self._xy)

    def views(self) -> list[_SeriesData]:
        return [_SeriesData(self, j) for j in range(len(self._xy))]

    # -- 그리기 -------------------------------------------------------------
    def _drawn(self) -> list[tuple[QPen | QBrush, Any]]:
        """보이고 점이 있는 계열마다 (펜 또는 붓, 그릴 것). PlotCurveItem이 그릴 때 만드는 것과 같은 것을 같은 함수로."""
        if self._ops is not None:
            return self._ops
        from pyqtgraph.graphicsItems.PlotCurveItem import arrayToLineSegments

        ops: list[tuple[QPen | QBrush, Any]] = []
        for (x, y), style, shown in zip(self._xy, self._style, self._shown, strict=True):
            if not shown or len(x) == 0:
                continue
            if isinstance(style, QBrush):
                if style.style() == Qt.BrushStyle.NoBrush:
                    continue
                ops.append((style, fn.arrayToQPath(x, y, connect="finite", finiteCheck=True)))
                continue
            if style.style() == Qt.PenStyle.NoPen:
                continue
            if self._segments(style):
                ops.append((style, arrayToLineSegments(x, y, connect="finite", finiteCheck=True)))
            else:
                ops.append((style, fn.arrayToQPath(x, y, connect="finite", finiteCheck=True)))
        self._ops = ops
        return ops

    def _segments(self, pen: QPen) -> bool:
        """PlotCurveItem._shouldUseDrawLineSegments(segmentedLineMode "auto")와 같은 판단."""
        return (pen.widthF() > 1.0 and pen.style() == Qt.PenStyle.SolidLine and pen.isSolid()
                and pen.color().alphaF() == 1.0 and not self._aa)

    def paint(self, p: QPainter, *args: Any) -> None:
        ops = self._drawn()
        if not ops:
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing, self._aa)
        for style, what in ops:
            if isinstance(style, QBrush):
                p.fillPath(what, style)
            else:
                p.setPen(style)
                if isinstance(what, QPainterPath):
                    p.drawPath(what)
                else:
                    p.drawLines(*what.drawargs())

    # -- 범위 ---------------------------------------------------------------
    def dataBounds(self, ax: int, frac: float = 1.0, orthoRange: Any = None) -> tuple[float | None, float | None]:  # noqa: N802, N803
        """보이는 계열마다 PlotCurveItem.dataBounds(frac=1)를 낸 값을 합친 범위(계열 자료 범위의 합)."""
        key = (ax, frac, None if orthoRange is None else tuple(orthoRange))
        hit = self._bounds.get(key)
        if hit is not None:
            return hit
        lo: float | None = None
        hi: float | None = None
        for (x, y), shown in zip(self._xy, self._shown, strict=True):
            if not shown or len(x) == 0:
                continue
            d, d2 = (x, y) if ax == 0 else (y, x)
            if orthoRange is not None:
                d = d[(d2 >= orthoRange[0]) & (d2 <= orthoRange[1])]
            d = d[np.isfinite(d)]
            if not len(d):
                continue
            if frac >= 1.0:
                a, b = float(d.min()), float(d.max())
            else:
                a, b = (float(v) for v in np.percentile(d, [50 * (1 - frac), 50 * (1 + frac)]))
            lo = a if lo is None else min(lo, a)
            hi = b if hi is None else max(hi, b)
        out = (lo, hi)
        self._bounds[key] = out
        return out

    def pixelPadding(self) -> float:  # noqa: N802
        """보이는 선 계열 펜 중 가장 굵은 것(PlotCurveItem: 코스메틱 펜 폭 × 0.7072, 채우기는 0)."""
        if self.kind == "fill":
            return 0.0
        pads = [s.widthF() * 0.7072 for s, shown, (x, _) in zip(self._style, self._shown, self._xy, strict=True)
                if shown and len(x) and isinstance(s, QPen) and s.isCosmetic() and s.style() != Qt.PenStyle.NoPen]
        return max(pads, default=0.0)

    def boundingRect(self) -> QRectF:  # noqa: N802
        """PlotCurveItem.boundingRect와 같은 방법(자료 범위 + 펜 폭만큼의 픽셀)."""
        if self._rect is None:
            xmn, xmx = self.dataBounds(0)
            ymn, ymx = self.dataBounds(1)
            if xmn is None or xmx is None or ymn is None or ymx is None:
                return QRectF()
            px = py = 0.0
            pad = self.pixelPadding()
            if pad > 0:
                vx, vy = self.pixelVectors()
                try:
                    px = 0.0 if vx is None else vx.length()
                except OverflowError:
                    px = 0.0
                try:
                    py = 0.0 if vy is None else vy.length()
                except OverflowError:
                    py = 0.0
                px *= pad
                py *= pad
            self._rect = QRectF(xmn - px, ymn - py, 2 * px + xmx - xmn, 2 * py + ymx - ymn)
        return self._rect

    def viewTransformChanged(self) -> None:  # noqa: N802
        self._rect = None
        self.prepareGeometryChange()


class _WatchTile(_Tile):
    """「보는 재미」 타일 공통: 아래쪽 안내 줄(note), 마우스 글(hover text), 쪽 설정 값."""

    SETTING_KEYS: tuple[str, ...] = ()

    def keep_y(self, plot: pg.PlotItem, lo: float, hi: float) -> None:
        """세로 범위를 고정한다(범위가 바뀔 때만: 사용자가 확대한 것을 세대마다 풀지 않게)."""
        if self._fixed_y.get(id(plot)) != (lo, hi):
            self.fix_y(plot, lo, hi)

    def __init__(self, dash: Dashboard, tile_id: str) -> None:
        super().__init__(dash, tile_id)
        self.note = QLabel()
        self.note.setWordWrap(True)
        small = QFont(self.note.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() * 0.9))
        self.note.setFont(small)
        self.note.setStyleSheet(f"color: {self.theme.ink2};")
        self.note.hide()
        lay = self.layout()
        assert lay is not None
        lay.addWidget(self.note)
        self._hover_text = ""
        self._notes: list[str] = []

    @property
    def history(self) -> hs.RunHistory:
        return self.dash.history

    def set_notes(self, notes: list[str]) -> None:
        notes = [n for n in notes if n]
        if notes != self._notes:
            self._notes = notes
            self.note.setText("\n".join(notes))
            self.note.setVisible(bool(notes))

    def loading_note(self) -> str:
        busy = self.dash.history_loading
        return S.DASH_LOADING.format(done=busy[0], total=busy[1]) if busy is not None else ""

    def show_message(self, text: str, notes: list[str] | None = None) -> None:
        self.message.setText(text)
        self.stack.setCurrentWidget(self.message)
        self.set_notes(notes if notes is not None else [self.loading_note()])

    def legend_html(self, i: int | None) -> str:
        if self._hover_text:
            return _escape(self._hover_text)
        return super().legend_html(None)

    def unhover(self) -> None:
        self._hover_text = ""
        super().unhover()

    def settings_values(self) -> dict[str, str]:
        return {}

    def load_settings_values(self, values: Mapping[str, Any]) -> None:
        pass

    @staticmethod
    def _select(box: QComboBox, value: Any) -> None:
        if isinstance(value, str):
            i = box.findData(value)
            if i >= 0:
                box.setCurrentIndex(i)


# ---------------------------------------------------------------------------
# 유전자 지도
_LIVE_JOBS: set[QThread] = set()
"""도는 중인 배경 PCA 스레드를 끝날 때까지 붙잡아 둔다(타일이 먼저 사라져도 QThread가 돌던 중에 지워지지 않게)."""


class _PCAJob(QThread):
    """큰 차원 유전자 지도 축(D18): 표본 뽑기 + 무작위 PCA를 배경에서."""

    done = Signal(object)

    def __init__(self, blocks: list[np.ndarray], generations: int, epoch: int, max_rows: int,
                 canonical: bool = False) -> None:
        super().__init__()
        self.blocks = blocks
        self.generations = generations
        self.epoch = epoch
        self.max_rows = max_rows
        self.canonical = canonical
        """다시 연 실행을 다 읽은 뒤 모든 세대로 맞추는 축(앞 축의 부호를 따르지 않고 알림도 없음)."""

    def run(self) -> None:
        hs.GUI_PRIORITY.wait(self.isInterruptionRequested)  # 펌프가 그리는 중이면 끝난 뒤에(GIL을 다투지 않게)
        sample = hs.sample_rows(self.blocks, self.max_rows)
        result = hs.randomized_axes(sample) if len(sample) else None
        self.done.emit((self, self.epoch, self.generations, result))


def _forget_job(job: QThread) -> None:
    job.wait(2000)
    _LIVE_JOBS.discard(job)


def wait_jobs() -> None:
    """프로그램을 끝낼 때: 도는 배경 PCA를 끝날 때까지 기다린다(주인 없는 QThread가 돌던 중에 지워지면 abort)."""
    for job in list(_LIVE_JOBS):
        job.wait()
        _LIVE_JOBS.discard(job)


class GeneMapTile(_WatchTile):
    """유전자 지도(4.1): 지금까지 모든 세대의 PCA 두 축에 보는 세대 집단을 찍는다. 아래에서 위로 집단 가운데가 지나온
    길 → 앞 세대 잔상 → 보는 세대 점(점수 색 또는 성공 비율) → 세대 대표 고리 → 그때 챔피언 별 → 출발점 빈 원.
    잔상 점을 누르면 그 세대로(focus_requested), 점을 두 번 누르면 그 개체 재생(individual_requested)."""

    follows_axis = False
    SETTING_KEYS = ("genemap_color",)

    def __init__(self, dash: Dashboard, tile_id: str) -> None:
        super().__init__(dash, tile_id)
        t = self.theme
        self.plot = self.add_plot(S.GMAP_AXIS_Y.format(pct=0.0))
        self.plot.showGrid(x=False, y=False)
        for name in ("left", "bottom"):
            self.plot.getAxis(name).setStyle(showValues=False)
        self.plot.setLabel("bottom", S.GMAP_AXIS_X.format(pct=0.0), color=t.ink2)
        self.plot.setAspectLocked(True)
        self.view.setToolTip(S.GMAP_HINT)

        self.controls.addWidget(QLabel(S.GMAP_COLOR_LABEL))
        self.color_box = QComboBox()
        for mode in GMAP_COLOR_MODES:
            self.color_box.addItem(S.GMAP_COLOR_MODES[mode], mode)
        self.color_box.setToolTip(S.GMAP_COLOR_TIP)
        self.color_box.currentIndexChanged.connect(self._color_changed)
        self.controls.addWidget(self.color_box)
        self.refit_button = QPushButton(S.GMAP_REFIT)
        self.refit_button.setToolTip(S.GMAP_REFIT_TIP)
        self.refit_button.clicked.connect(self.refit)
        self.controls.addWidget(self.refit_button)

        self.trail = _curve(_pen(t.muted, 1))
        self.trail.setZValue(0)
        self.plot.addItem(self.trail)
        self.trail_rest = _curve(pg.mkPen(None))  # 보는 세대 뒤의 길: 그리지 않고 화면 범위에만 넣는다
        self.trail_rest.setZValue(0)
        self.plot.addItem(self.trail_rest)
        self.ghosts: list[pg.ScatterPlotItem] = []
        for k, a in enumerate(GHOST_ALPHA):
            g = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=_qcolor(t.muted, a))
            g.setZValue(3 - k)
            self.plot.addItem(g)
            self.ghosts.append(g)
        self.points = pg.ScatterPlotItem(size=7, pen=pg.mkPen(None), brush=_qcolor(t.muted))
        self.points.setZValue(5)
        self.plot.addItem(self.points)
        self.rep_ring = pg.ScatterPlotItem(size=11, pen=_pen(t.ink, 2), brush=pg.mkBrush(None))
        self.rep_ring.setZValue(6)
        self.plot.addItem(self.rep_ring)
        self.champion = pg.ScatterPlotItem(size=14, symbol="star", pen=_pen(t.surface, 1), brush=_qcolor(t.series[3]))
        self.champion.setZValue(7)
        self.plot.addItem(self.champion)
        self.start = pg.ScatterPlotItem(size=9, pen=_pen(t.muted, 1.5), brush=pg.mkBrush(None))
        self.start.setZValue(4)
        self.plot.addItem(self.start)

        self.cmap = _ramp_map(t)
        self._lut = [QBrush(c) for c in self.cmap.map(np.linspace(0.0, 1.0, LUT_SIZE), mode="qcolor")]
        self._unknown = QBrush(_qcolor(t.muted))
        self.bar = pg.ColorBarItem(values=(0, 1), width=10, colorMap=self.cmap, label=S.GMAP_SCORE_BAR,
                                   interactive=False, colorMapMenu=False)
        _style_bar(self.bar, t)
        self.view.ci.addItem(self.bar, row=0, col=1)
        self.view.scene().sigMouseClicked.connect(self._on_click)
        self.message.setText(S.GMAP_EMPTY)

        self._proj: dict[int, np.ndarray] = {}
        self._trail_proj: dict[int, np.ndarray] = {}
        self._proj_version = -1
        # 그린 점(마우스·클릭): (세대, 좌표 (N, 2), 개체 번호 (N,), 종류)
        self._drawn: list[tuple[int, np.ndarray, np.ndarray, str]] = []
        self._special: list[tuple[str, np.ndarray, str]] = []  # (종류, 좌표 (2,), 풍선 글)
        self.shown_generation: int | None = None
        """지금 점을 찍은 세대(보는 세대에 유전자가 없으면 가장 가까운 기록 세대)."""
        self._job: _PCAJob | None = None
        self._fit_all = False
        """다시 연 실행의 기록을 다 읽었다: 다음 갱신에서 읽은 모든 세대로 축을 맞춘다(_fit canonical). 읽는 동안에는
        첫 축만 임시로 맞추고 두 배 규칙으로 다시 맞추지 않는다: 읽는 동안 어느 쪽을 보고 있었는지에 따라 축이 달라지지
        않게(설계 2.5 '학습 중 그림 = 다시 연 그림', 11.2)."""
        self._refit_note = ""
        self._refit_timer = QTimer(self)
        self._refit_timer.setSingleShot(True)
        self._refit_timer.setInterval(REFIT_NOTE_MS)
        self._refit_timer.timeout.connect(self._clear_refit_note)

    # -- 설정 -------------------------------------------------------------------
    @property
    def color_mode(self) -> str:
        return str(self.color_box.currentData())

    def set_color_mode(self, mode: str) -> None:
        self._select(self.color_box, mode)

    def _color_changed(self, _index: int) -> None:
        self.dash.save_settings()
        self.refresh()

    def settings_values(self) -> dict[str, str]:
        return {"genemap_color": self.color_mode}

    def load_settings_values(self, values: Mapping[str, Any]) -> None:
        self._select(self.color_box, values.get("genemap_color"))

    def history_reset(self) -> None:
        self._proj.clear()
        self._trail_proj.clear()
        self._proj_version = -1
        self._job = None
        self._fit_all = False
        self._clear_refit_note()

    def history_loaded(self) -> None:
        """배경 읽기가 끝났다(Dashboard._on_history_loaded): 다음 갱신에서 읽은 모든 세대로 축을 맞춘다."""
        self._fit_all = True

    def sync_marks(self) -> None:
        pass  # 지도에는 세대 세로선이 없다

    # -- 축 맞추기 ------------------------------------------------------------
    def refit(self) -> None:
        """「지도 다시 맞추기」: 지금까지의 세대 전체로 축을 다시 잡는다."""
        pca = self.history.ensure_pca()
        if pca is None or pca.generations == 0:
            return
        self._fit(pca, force=True)
        self.refresh()

    def _fit(self, pca: hs.GenePCA, *, force: bool = False, canonical: bool = False) -> None:
        """축을 맞춘다. canonical: 다시 연 실행을 다 읽은 뒤 모든 세대로(앞 축의 부호를 따르지 않아 읽는 동안 임시 축을
        맞췄든 아니든 같은 축, 알림 없음)."""
        refitting = pca.axes is not None and not canonical
        if not pca.big:
            if canonical:
                pca.axes = None  # 부호를 처음 맞출 때 규칙으로(임시 축을 따르지 않음)
                self._fit_all = False
            if pca.fit() and refitting:
                self._note_refit(pca.generations)
            return
        if self._job is not None:
            return  # 도는 작업이 끝나면 다시 갱신한다(canonical이면 _fit_all이 남아 그때 맞춘다)
        if canonical:
            self._fit_all = False
        blocks = [b for g in self.history.genome_gens() if (b := self.history.entry(g).map_genomes) is not None]
        rows = max(64, min(hs.PCA_SAMPLE_ROWS, PCA_BIG_VALUES // max(1, pca.dim)))
        job = _PCAJob(blocks, pca.generations, self.history.epoch, rows, canonical)
        job.done.connect(self._big_fit_done)
        job.finished.connect(lambda j=job: _forget_job(j))
        self._job = job
        _LIVE_JOBS.add(job)
        job.start()

    def _big_fit_done(self, payload: tuple[_PCAJob, int, int, Any]) -> None:
        job, epoch, generations, result = payload
        if job is not self._job:
            return  # 앞 기록(되감기·다른 실행)의 작업: 지금 도는 작업을 잊지 않는다
        self._job = None
        pca = self.history.ensure_pca()
        if result is None or pca is None or epoch != self.history.epoch:
            return
        refitting = pca.axes is not None and not job.canonical
        if job.canonical:
            pca.axes = None
        mean, axes, explained = result
        pca.set_axes(mean, axes, explained, generations=generations)
        if refitting:
            self._note_refit(generations)
        self.dash._invalidate(("genemap",))

    def _note_refit(self, generations: int) -> None:
        self._refit_note = S.GMAP_REFITTED.format(n=generations)
        self._refit_timer.start()

    def _clear_refit_note(self) -> None:
        if self._refit_note:
            self._refit_note = ""
            self.dash._invalidate(("genemap",))

    # -- 그리기 ---------------------------------------------------------------
    def _clear_items(self) -> None:
        self.trail.setData([], [])
        self.trail_rest.setData([], [])
        for item in (*self.ghosts, self.points, self.rep_ring, self.champion, self.start):
            item.clear()
        self._drawn = []
        self._special = []
        self.shown_generation = None

    def _projected(self, gen: int, pca: hs.GenePCA) -> np.ndarray | None:
        p = self._proj.get(gen)
        if p is None:
            e = self.history.entry(gen)
            genomes = None if e is None else e.map_genomes
            if genomes is None:
                return None
            p = pca.project(genomes)
            self._proj[gen] = p
            if len(self._proj) > 4 * GHOSTS + 8:  # 보는 세대 둘레만 둔다
                for old in sorted(self._proj, key=lambda g: abs(g - gen))[4 * GHOSTS + 8:]:
                    del self._proj[old]
        return p

    def _brushes(self, gen: int) -> list[QBrush]:
        e = self.history.entry(gen)
        assert e is not None
        if self.color_mode == "success":
            if e.obs is None:
                return [self._unknown] * len(e.scores)
            frac = e.obs.successes() / max(1, e.obs.trials)
        else:
            rng = self.history.score_range()
            if rng is None:
                return [self._unknown] * len(e.scores)
            lo, hi = rng
            frac = (np.asarray(e.scores, dtype=np.float64) - lo) / (hi - lo)
        idx = np.clip(np.nan_to_num(frac, nan=0.0) * (LUT_SIZE - 1), 0, LUT_SIZE - 1).round().astype(np.int64)
        return [self._lut[i] for i in idx.tolist()]

    def update_data(self) -> None:
        h = self.history
        meta = self.dash.watch_meta
        gen = self.dash.view_generation()
        if meta is None or gen is None or not h.genome_gens():
            self._clear_items()
            self.show_message(S.GMAP_EMPTY)
            return
        pca = h.ensure_pca()
        assert pca is not None
        loading = self.dash.history_loading is not None
        if self._fit_all and not loading:
            self._fit(pca, canonical=True)
        elif pca.axes is None or (pca.wants_fit() and not self.dash.flipping and not loading):
            self._fit(pca)
        if pca.axes is None:
            self._clear_items()
            self.show_message(S.GMAP_BIG_DIM.format(dim=pca.dim))
            return
        self.stack.setCurrentWidget(self.view)
        if pca.version != self._proj_version:
            self._proj.clear()
            self._trail_proj.clear()
            self._proj_version = pca.version
        target = h.nearest_genome_gen(gen)
        assert target is not None
        e = h.entry(target)
        assert e is not None
        pts = self._projected(target, pca)
        assert pts is not None
        n = len(pts)
        index = np.arange(n) if e.obs is not None else np.array([-1])
        self.points.setData(pos=pts, brush=self._brushes(target))
        self._drawn = [(target, pts, index, "now")]
        rep_pt = pts[e.map_rep]
        self.rep_ring.setData(pos=rep_pt.reshape(1, 2))
        for item in self.ghosts:
            item.clear()
        for k, g in enumerate(h.previous_genome_gens(target, GHOSTS)):
            gp = self._projected(g, pca)
            ge = h.entry(g)
            if gp is None or ge is None:
                continue
            self.ghosts[k].setData(pos=gp)
            self._drawn.append((g, gp, np.arange(len(gp)) if ge.obs is not None else np.array([-1]), "ghost"))
        self._draw_trail(pca, target)
        self._special = []
        champ = h.champion_entry_at(gen)
        if champ is not None:
            cp = pca.project(champ[1])[0]
            self.champion.setData(pos=cp.reshape(1, 2))
            score = champ[2] if math.isfinite(champ[2]) else self.dash.row_value(champ[0], "champion_score")
            self._special.append(("champion", cp, S.GMAP_CHAMPION_TIP.format(gen=champ[0], score=fmt_value(score))))
        else:
            self.champion.clear()
        sp = pca.project(meta.layout.initial())[0]
        self.start.setData(pos=sp.reshape(1, 2))
        self._special.append(("start", sp, S.GMAP_START_TIP))
        t = self.theme
        ex = pca.explained * 100.0
        self.plot.setLabel("bottom", S.GMAP_AXIS_X.format(pct=float(ex[0])), color=t.ink2)
        self.plot.setLabel("left", S.GMAP_AXIS_Y.format(pct=float(ex[1])), color=t.ink2)
        if self.color_mode == "success":
            self.bar.setLevels(values=(0.0, 1.0))
            self.bar.getAxis("left").setLabel(S.GMAP_SUCCESS_BAR)
        else:
            rng = h.score_range() or (0.0, 1.0)
            self.bar.setLevels(values=rng)
            self.bar.getAxis("left").setLabel(S.GMAP_SCORE_BAR)
        self.shown_generation = target
        notes = []
        focus_entry = h.entry(gen)
        if target != gen and focus_entry is not None and focus_entry.obs is not None:
            notes.append(S.DASH_GENOME_STRIDE.format(k=focus_entry.obs.genome_stride, gen=target))
        first_obs = h.first_observed
        if e.obs is None:
            # 'v1.2보다 먼저'라고 말할 수 있는 것은 v1.0 열 머리(learner 열 없음) 실행의 첫 기록 앞 세대뿐이다. 관찰을 끈
            # 학습(--no-observe), 요약 실패, 검사에서 떨어진 세대는 v1.2 세대라도 기록이 없다
            old = not math.isfinite(self.dash.learner_max) and (first_obs is None or target < first_obs)
            notes.append(S.DASH_OLD_RUN_NOTE if old else S.DASH_NO_OBS_NOTE)
        if first_obs is not None and h.unobserved_count and h.gens and first_obs > h.gens[0]:
            notes.append(S.DASH_OBSERVE_FROM.format(gen=first_obs))
        notes.append(self._refit_note)
        notes.append(self.loading_note())
        self.set_notes(notes)

    def _draw_trail(self, pca: hs.GenePCA, upto: int) -> None:
        """집단 가운데가 지나온 길: upto세대(보이는 점의 세대)까지. 그 뒤 길은 보이지 않는 선으로만 두어(trail_rest) 화면
        범위가 전체 길을 담게 한다: 「세대 넘겨 보기」 동안 길이 자라도 지도가 들썩이지 않는다."""
        trail = self.history.trail()
        missing = [(g, mean) for g, _learner, mean in trail if g not in self._trail_proj]
        if missing:  # 축이 바뀐 뒤 처음이면 모든 세대를 한 번에 투영
            proj = pca.project(np.vstack([m for _g, m in missing]))
            for (g, _m), p in zip(missing, proj, strict=True):
                self._trail_proj[g] = p
        parts: tuple[tuple[list[float], list[float]], tuple[list[float], list[float]]] = (([], []), ([], []))
        prev: int | None = None
        last: tuple[float, float] | None = None
        for g, learner, _mean in trail:
            p = self._trail_proj[g]
            xs, ys = parts[0] if g <= upto else parts[1]
            if g > upto and not xs and last is not None:
                xs.append(last[0])  # 뒤 길은 보이는 길 끝에서 잇는다
                ys.append(last[1])
            if prev is not None and learner is not None and learner != prev:
                xs.append(math.nan)
                ys.append(math.nan)
            if learner is not None:
                prev = learner
            xs.append(float(p[0]))
            ys.append(float(p[1]))
            if g <= upto:
                last = (float(p[0]), float(p[1]))
        self.trail.setData(np.array(parts[0][0]), np.array(parts[0][1]))
        self.trail_rest.setData(np.array(parts[1][0]), np.array(parts[1][1]))

    def entries(self) -> list[_Entry]:
        t = self.theme
        gen = self.shown_generation
        out: list[_Entry] = []
        if gen is not None:
            n = len(self._drawn[0][1]) if self._drawn else 0
            out.append(_Entry(t.ramp[3], GLYPH_FILL, S.GMAP_LEGEND.format(gen=gen, n=n)))
        out += [
            _Entry(t.ink, "◯", S.GMAP_REP),
            _Entry(t.series[3], "★", S.GMAP_CHAMPION),
            _Entry(t.muted, "○", S.GMAP_START),
            _Entry(t.muted, GLYPH_LINE, S.GMAP_TRAIL),
            _Entry(_blend(t.muted, t.surface, 0.4), "●", S.GMAP_GHOST),
        ]
        return out

    # -- 마우스 ---------------------------------------------------------------
    def _nearest(self, xv: float, yv: float) -> tuple[str, int, int, str] | None:
        """(종류, 세대, 개체 번호, 풍선 글) 또는 None. 화면 PICK_PX 안의 가장 가까운 점(위에 그린 것 먼저)."""
        px, py = self.plot.vb.viewPixelSize()
        if not (px > 0 and py > 0):
            return None
        best: tuple[float, tuple[str, int, int, str]] | None = None
        for kind, pos, tip in self._special:
            d = math.hypot((float(pos[0]) - xv) / px, (float(pos[1]) - yv) / py)
            if d <= PICK_PX and (best is None or d < best[0]):
                best = (d - 0.01, (kind, -1, -1, tip))
        for gen, pts, idx, kind in self._drawn:
            if not len(pts):
                continue
            d = np.hypot((pts[:, 0] - xv) / px, (pts[:, 1] - yv) / py)
            j = int(np.argmin(d))
            if d[j] <= PICK_PX and (best is None or d[j] < best[0]):
                best = (float(d[j]), (kind, gen, int(idx[j]), self._point_tip(gen, int(idx[j]))))
            if best is not None and kind == "now":
                break  # 보는 세대 점이 잔상보다 위에 있다
        return None if best is None else best[1]

    def _point_tip(self, gen: int, index: int) -> str:
        e = self.history.entry(gen)
        if e is None:
            return ""
        if e.obs is None or index < 0:
            return S.GMAP_POINT_TIP_OLD.format(gen=gen, score=fmt_value(e.rep_score))
        score = fmt_value(float(e.obs.scores[index]))
        if index == e.obs.representative:
            return S.GMAP_POINT_TIP_REP.format(gen=gen, index=index, score=score)
        return S.GMAP_POINT_TIP.format(gen=gen, index=index, score=score,
                                       success=int(e.obs.successes()[index]), trials=e.obs.trials)

    def hover(self, xv: float, yv: float, plot_index: int) -> None:
        hit = self._nearest(xv, yv)
        self._hover_text = "" if hit is None else hit[3]
        self.update_legend()

    def click_at(self, xv: float, yv: float, *, double: bool) -> bool:
        """지도 좌표의 클릭: 잔상 점을 한 번 누르면 focus_requested(세대), 점을 두 번 누르면
        individual_requested(세대, 개체 번호; 집단 기록이 없는 세대의 대표 점은 -1). 무엇을 했으면 True."""
        hit = self._nearest(xv, yv)
        if hit is None:
            return False
        kind, gen, index, _tip = hit
        if kind not in ("now", "ghost"):
            return False
        if double:
            self.dash.individual_requested.emit(gen, index)
            return True
        if kind == "ghost":
            self.dash.focus_requested.emit(gen)
            return True
        return False

    def _on_click(self, ev: Any) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        pos = ev.scenePos()
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        pt = self.plot.vb.mapSceneToView(pos)
        if self.click_at(pt.x(), pt.y(), double=bool(ev.double())):
            ev.accept()


# ---------------------------------------------------------------------------
# 몸 수치 추이
class BodyTile(_WatchTile):
    """몸 수치 추이(4.2): 수치마다 집단 평균 선과 최소~최대 띠. 「모두(범위 대비 %)」는 진화 범위를 0~100%로 맞춰 함께,
    수치 하나는 물리 단위로 평균·띠·세대 대표 점선·진화 범위 끝 가로 점선. 몸 유전자가 없으면 켜는 법 안내."""

    follows_axis = True

    def __init__(self, dash: Dashboard, tile_id: str) -> None:
        super().__init__(dash, tile_id)
        t = self.theme
        self.plot = self.add_plot(S.BODY_AXIS_ALL)
        self.pick_label = QLabel(S.BODY_PICK_LABEL)
        self.pick_label.hide()
        self.controls.addWidget(self.pick_label)
        self.pick = QComboBox()
        self.pick.setToolTip(S.BODY_PICK_TIP)
        self.pick.currentIndexChanged.connect(lambda _i: self.refresh())
        self.pick.hide()
        self.controls.addWidget(self.pick)
        self.message.setText(S.BODY_EMPTY)
        use_fast_grid(self.plot.getAxis("left"))  # 격자선이 그림 시간의 절반이었다(같은 픽셀로 빠르게)
        self.focus_line = _focus_line(t)
        self.plot.addItem(self.focus_line, ignoreBounds=True)
        self.rep_line = _curve(_pen(t.ink, 1.2, Qt.PenStyle.DashLine))
        self.rep_line.setZValue(20)
        self.plot.addItem(self.rep_line)
        self.limits = [pg.InfiniteLine(angle=0, movable=False, pen=_pen(t.muted, 1, Qt.PenStyle.DashLine))
                       for _ in range(2)]
        for line in self.limits:
            line.setToolTip(S.BODY_LIMITS)
            line.hide()
            self.plot.addItem(line)
        # 수치마다 평균 선과 띠: 곡선 항목을 수치마다 두지 않고 선 항목 하나·띠 항목 하나(_SeriesLayer, v3.0 지연 고침).
        # 띠(z −10)를 모두 그린 뒤 선(z 10)을 수치 순서로 그린다(곡선 항목을 따로 두던 그림과 같은 순서)
        self.band_layer = _SeriesLayer("fill")
        self.band_layer.setZValue(-10)
        self.line_layer = _SeriesLayer("line")
        self.line_layer.setZValue(10)
        self.plot.addItem(self.band_layer)
        self.plot.addItem(self.line_layer)
        self.lines: list[_SeriesData] = []
        """수치마다 평균 선 자료(getData, 시험용)."""
        self.bands: list[_SeriesData] = []
        self._series: hs.BodySeries | None = None
        self._x = np.zeros(0)
        self._colors: list[str] = []
        self._left_label: str | None = None

    @property
    def selected(self) -> int:
        """-1 = 모두(범위 대비 %), 0.. = 수치 하나."""
        data = self.pick.currentData()
        return int(data) if data is not None else -1

    def meta_changed(self) -> None:
        meta = self.dash.watch_meta
        self.pick.blockSignals(True)
        self.pick.clear()
        params = meta.body if meta is not None else ()
        if len(params) > 1:
            self.pick.addItem(S.BODY_ALL, -1)
        for j, p in enumerate(params):
            self.pick.addItem(p.name, j)
        self.pick.blockSignals(False)
        self.pick.setVisible(len(params) > 0)
        self.pick_label.setVisible(len(params) > 0)  # 몸 수치가 없는 환경에서 이름표만 남지 않게
        self._rebuild_items(len(params))

    def _rebuild_items(self, count: int) -> None:
        t = self.theme
        self._colors = [t.series[j] if (count <= MAX_SERIES or j < MAX_SERIES - 1) else t.muted
                        for j in range(count)]
        self.band_layer.set_count(0, QBrush())  # 수치가 바뀌면 계열을 새로(앞 환경의 자료·펜을 남기지 않게)
        self.line_layer.set_count(0, QPen())
        self.band_layer.set_count(count, QBrush())
        self.line_layer.set_count(count, QPen())
        for j, color in enumerate(self._colors):
            self.band_layer.set_style(j, _qcolor(color, 45))
            self.line_layer.set_style(j, _pen(color, 1.5))
        self.band_layer.commit()
        self.line_layer.commit()
        self.lines = self.line_layer.views()
        self.bands = self.band_layer.views()

    def _set_left_label(self, text: str) -> None:
        # setLabel은 같은 글이어도 축 그림을 버리고 다시 만든다(축 하나 2~3ms): 글이 바뀔 때만
        if text != self._left_label:
            self._left_label = text
            self.plot.setLabel("left", text, color=self.theme.ink2)

    def update_data(self) -> None:
        meta = self.dash.watch_meta
        if meta is not None and meta.n_body == 0:
            self.show_message(S.BODY_EMPTY_NONE, [])
            return
        bs = self.history.body_series() if meta is not None else None
        if meta is None or bs is None or bs.mean.shape[1] != meta.n_body:
            self._series = None
            self.show_message(S.BODY_EMPTY)
            return
        if self.line_layer.count != meta.n_body:
            self._rebuild_items(meta.n_body)
        self.stack.setCurrentWidget(self.view)
        self._series = bs
        x = self.dash.x_at_generations(bs.gens, True)
        self._x = x
        sel = self.selected
        t = self.theme
        lines, bands = self.line_layer, self.band_layer
        if sel < 0:
            lows = np.array([p.low for p in meta.body])
            span = np.array([p.high - p.low for p in meta.body])
            # 수치마다 (값 − 최소) ÷ 폭 × 100을 한 번에(열마다 따로 계산한 값과 비트까지 같다)
            pct = (bs.mean - lows) / span * 100.0
            lo = (bs.lo - lows) / span * 100.0
            hi = (bs.hi - lows) / span * 100.0
            for j, color in enumerate(self._colors):
                lines.set_series(j, x, pct[:, j])
                lines.set_style(j, _pen(color, 1.5))  # 「모두」로 돌아오면 원래 색
                lines.set_shown(j, True)
                if color == t.muted:
                    bands.set_series(j, np.zeros(0), np.zeros(0))
                else:
                    bands.set_series(j, *_band_xy(x, lo[:, j], hi[:, j]))
                bands.set_style(j, _qcolor(color, 45))
                bands.set_shown(j, True)
            self.rep_line.setData([], [])
            for line in self.limits:
                line.hide()
            self._set_left_label(S.BODY_AXIS_ALL)
            self.keep_y(self.plot, 0.0, 100.0)
        else:
            p = meta.body[sel]
            for j in range(meta.n_body):
                if j == sel:
                    lines.set_style(j, _pen(t.series[0], 1.5))
                    lines.set_series(j, x, bs.mean[:, j])
                    bands.set_style(j, _qcolor(t.series[0], 45))
                    bands.set_series(j, *_band_xy(x, bs.lo[:, j], bs.hi[:, j]))
                lines.set_shown(j, j == sel)
                bands.set_shown(j, j == sel)
            self.rep_line.setData(x, bs.rep[:, sel])
            for line, v in zip(self.limits, (p.low, p.high), strict=True):
                line.setPos(float(v))
                line.show()
            self._set_left_label(p.name)
            self.keep_y(self.plot, float(p.low), float(p.high))
        bands.commit()
        lines.commit()
        self._place_focus(self.focus_line)
        self.set_notes([S.BODY_REPS_ONLY if not bool(bs.observed.all()) else "", self.loading_note()])

    def _place_focus(self, line: pg.InfiniteLine) -> None:
        focus = self.dash.focus
        if focus is None:
            line.hide()
            return
        x = self.dash.x_at_generations(np.array([float(focus)]), True)[0]
        if math.isfinite(x):
            line.setPos(float(x))
            line.show()
        else:
            line.hide()

    def entries(self) -> list[_Entry]:
        meta = self.dash.watch_meta
        t = self.theme
        if meta is None or not meta.body or self._series is None:
            return []
        sel = self.selected
        if sel >= 0:
            return [_Entry(t.series[0], GLYPH_LINE, S.BODY_MEAN),
                    _Entry(_blend(t.series[0], t.surface, 0.45), GLYPH_FILL, S.BODY_RANGE),
                    _Entry(t.ink, GLYPH_DASH, S.BODY_REP), _Entry(t.muted, GLYPH_DASH, S.BODY_LIMITS)]
        out = []
        for j, p in enumerate(meta.body):
            if self._colors[j] == t.muted:
                out.append(_Entry(t.muted, GLYPH_LINE, S.DASH_OTHER))
                break
            out.append(_Entry(self._colors[j], GLYPH_LINE, p.name))
        out.append(_Entry(_blend(t.series[0], t.surface, 0.45), GLYPH_FILL, S.BODY_RANGE))
        return out

    def hover(self, xv: float, yv: float, plot_index: int) -> None:
        bs, meta = self._series, self.dash.watch_meta
        x = self._x
        if bs is None or meta is None or not len(x) or not np.isfinite(x).any():
            self.unhover()
            return
        i = int(np.nanargmin(np.abs(x - xv)))
        for line in self._cross:
            line.setPos(float(x[i]))
            line.show()
        sel = self.selected
        params = range(len(meta.body)) if sel < 0 else [sel]
        parts = [S.DASH_HOVER_X["generation"].format(gen=int(bs.gens[i]), x="")]
        for j in list(params)[:MAX_SERIES]:
            parts.append(S.BODY_HOVER.format(name=meta.body[j].name, mean=fmt_value(float(bs.mean[i, j])),
                                             lo=fmt_value(float(bs.lo[i, j])), hi=fmt_value(float(bs.hi[i, j])),
                                             rep=fmt_value(float(bs.rep[i, j]))))
        self._hover_text = "  ".join(parts)
        self.update_legend()


# ---------------------------------------------------------------------------
# 행동 분포
class ActionsTile(_WatchTile):
    """행동 분포(4.5): 가로 세대 × 세로 출력 세기 20칸(채널 범위) 히트맵, 색 = 그 세기로 쓴 시간 비율(열 합 100%).
    평균 출력 선, 보는 세대의 「쓰는 시간」. 관찰이 없는 세대는 투명."""

    follows_axis = False
    SETTING_KEYS = ("actions_mode",)

    def __init__(self, dash: Dashboard, tile_id: str) -> None:
        super().__init__(dash, tile_id)
        t = self.theme
        self.plot = self.add_plot(S.ACT_AXIS)
        self.plot.showGrid(x=False, y=False)
        self.image = pg.ImageItem(axisOrder="col-major")
        self.plot.addItem(self.image)
        self.cmap = _heat_map(t)
        self.bar = pg.ColorBarItem(values=(0, 1), width=10, colorMap=self.cmap, label=S.ACT_FRACTION,
                                   interactive=False, colorMapMenu=False)
        _style_bar(self.bar, t)
        self.bar.setImageItem(self.image, insert_in=self.plot)
        self.mean_line = _curve(_pen(t.ink, 1.2))
        self.mean_line.setZValue(30)
        self.plot.addItem(self.mean_line)
        self.focus_line = _focus_line(t)
        self.plot.addItem(self.focus_line, ignoreBounds=True)
        self.controls.addWidget(QLabel(S.ACT_CHANNEL_LABEL))
        self.channel_box = QComboBox()
        self.channel_box.setToolTip(S.ACT_CHANNEL_TIP)
        self.channel_box.currentIndexChanged.connect(lambda _i: self.refresh())
        self.controls.addWidget(self.channel_box)
        self.mode_box = QComboBox()
        for mode in hs.ACTION_MODES:
            self.mode_box.addItem(S.ACT_MODES[mode], mode)
        self.mode_box.setToolTip(S.ACT_MODE_TIP)
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        self.controls.addWidget(self.mode_box)
        self.message.setText(S.ACT_EMPTY)
        self._series: hs.ActionSeries | None = None
        self._range = (0.0, 1.0)
        self.summary = ""

    @property
    def channel(self) -> int:
        data = self.channel_box.currentData()
        return int(data) if data is not None else 0

    @property
    def mode(self) -> str:
        return str(self.mode_box.currentData())

    def set_mode(self, mode: str) -> None:
        self._select(self.mode_box, mode)

    def set_channel(self, channel: int) -> None:
        i = self.channel_box.findData(int(channel))
        if i >= 0:
            self.channel_box.setCurrentIndex(i)

    def _mode_changed(self, _index: int) -> None:
        self.dash.save_settings()
        self.refresh()

    def settings_values(self) -> dict[str, str]:
        return {"actions_mode": self.mode}

    def load_settings_values(self, values: Mapping[str, Any]) -> None:
        self._select(self.mode_box, values.get("actions_mode"))

    def meta_changed(self) -> None:
        meta = self.dash.watch_meta
        self.channel_box.blockSignals(True)
        self.channel_box.clear()
        for k, ch in enumerate(meta.channels if meta is not None else ()):
            self.channel_box.addItem(ch.label, k)
        self.channel_box.blockSignals(False)

    def update_data(self) -> None:
        meta = self.dash.watch_meta
        series = None
        if meta is not None and 0 <= self.channel < meta.n_out:
            series = self.history.action_series(self.channel, self.mode)
        self._series = series
        if meta is None or series is None:
            self.image.clear()
            self.mean_line.setData([], [])
            self.summary = ""
            if meta is not None and meta.n_out == 0:  # 구동기 출력이 없는 환경(뇌 없는 글라이더, v2.0 설계 5.4)
                self.show_message(S.DASH_EMPTY_NO_OUTPUTS, [])
                return
            # 기록은 있는데 행동 기록이 하나도 없으면(v1.2 전 실행, 관찰하지 않는 평가기) 그렇다고 알린다
            self.show_message(S.ACT_NONE if meta is not None and len(self.history) else S.ACT_EMPTY)
            return
        self.stack.setCurrentWidget(self.view)
        ch = meta.channels[self.channel]
        lo, hi = float(ch.low), float(ch.high)
        self._range = (lo, hi)
        self.image.setImage(series.frac, autoLevels=False, levels=(0.0, 1.0))
        self.image.setRect(QRectF(series.first - 0.5, lo, series.frac.shape[0], hi - lo))
        self.mean_line.setData(series.gens, series.mean)
        self.keep_y(self.plot, lo, hi)
        gen = self.dash.view_generation()
        e = self.history.entry(gen)
        if e is not None and e.act_frac is not None and e.act_mean_pop is not None and e.act_mean_rep is not None:
            m = hs.ACTION_MODES.index(self.mode)
            on = hs.on_fraction(e.act_frac[m, self.channel], lo, hi)
            mean = float((e.act_mean_pop if m == 0 else e.act_mean_rep)[self.channel])
            self.summary = S.ACT_SUMMARY.format(name=ch.label, on=on * 100.0, mean=mean, gen=e.generation)
        else:
            self.summary = S.ACT_NONE
        if self.dash.focus is not None:
            self.focus_line.setPos(float(self.dash.focus))
            self.focus_line.show()
        else:
            self.focus_line.hide()
        self.set_notes([self.summary, self.loading_note()])

    def image_array(self) -> np.ndarray:
        """보이는 그림 자료 (세대 수, 20). 자료가 없으면 빈 배열."""
        img = self.image.image
        return np.zeros((0, hs.ob.ACTION_BINS)) if img is None else np.asarray(img)

    def entries(self) -> list[_Entry]:
        if self._series is None:  # 빈 상태(안내 글만)에는 그림 설명도 없다(몸 수치·궤적 밀도와 같게)
            return []
        t = self.theme
        return [_Entry(t.ramp[3], GLYPH_FILL, S.ACT_HINT), _Entry(t.ink, GLYPH_LINE, S.ACT_MEAN_LINE)]

    def hover(self, xv: float, yv: float, plot_index: int) -> None:
        s = self._series
        if s is None:
            self.unhover()
            return
        lo, hi = self._range
        i = math.floor(xv - (s.first - 0.5))
        w = (hi - lo) / hs.ob.ACTION_BINS
        b = math.floor((yv - lo) / w)
        if not (0 <= i < s.frac.shape[0] and 0 <= b < hs.ob.ACTION_BINS) or not math.isfinite(s.frac[i, b]):
            self.unhover()
            return
        for line in self._cross:
            line.setPos(float(s.first + i))
            line.show()
        self._hover_text = S.ACT_HOVER.format(gen=s.first + i, lo=lo + b * w, hi=lo + (b + 1) * w,
                                              pct=float(s.frac[i, b]) * 100.0)
        self.update_legend()


# ---------------------------------------------------------------------------
# 궤적 밀도
class _GradientBar(QWidget):
    """궤적 밀도 색 막대(아래 = 드묾, 위 = 자주)."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedWidth(12)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setToolTip(S.DENS_BAR)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(12, 80)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        r = self.rect().adjusted(2, 4, -2, -4)
        grad = QLinearGradient(QPointF(0, r.bottom()), QPointF(0, r.top()))
        for k in (1, 64, 128, 192, 255):
            v = int(_DENSITY_LUT[k])
            grad.setColorAt(k / 255.0, QColor((v >> 16) & 255, (v >> 8) & 255, v & 255, (v >> 24) & 255))
        p.fillRect(r, QColor(DARK.surface))
        p.fillRect(r, grad)
        p.end()


class DensityTile(_WatchTile):
    """궤적 밀도(4.4): 작은 관전 캔버스(늘 어두운 바탕)에 그 세대 시험 0번 세계 + 밀도 층(로그 색). 「이 세대만 / 최근
    20세대 / 처음부터」는 보는 세대와 세계 열쇠·틀이 같은 세대만 합친다. 세계 그림은 geometry_needed로 청한다."""

    follows_axis = False
    SETTING_KEYS = ("density_window",)

    def __init__(self, dash: Dashboard, tile_id: str) -> None:
        super().__init__(dash, tile_id)
        self.canvas = SpectatorCanvas()
        self.canvas.setMinimumSize(160, 110)
        self.canvas.set_follow("fit")
        self.canvas.set_wheel_needs_ctrl(True)
        self.canvas.set_show_movers(False)  # 여러 시각을 합친 그림: 장애물 줄을 0초 자리에 그리지 않는다
        self.canvas.setMouseTracking(True)
        self.canvas.installEventFilter(self)
        self.gradient = _GradientBar()
        self.host = QWidget()
        hl = QHBoxLayout(self.host)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(2)
        hl.addWidget(self.canvas, 1)
        hl.addWidget(self.gradient)
        self.stack.addWidget(self.host)
        self.controls.addWidget(QLabel(S.DENS_WINDOW_LABEL))
        self.window_box = QComboBox()
        for w in hs.DENSITY_WINDOWS:
            self.window_box.addItem(S.DENS_WINDOWS[w].format(n=hs.DENS_RECENT), w)
        self.window_box.setToolTip(S.DENS_WINDOW_TIP)
        self.window_box.currentIndexChanged.connect(self._window_changed)
        self.controls.addWidget(self.window_box)
        self.message.setText(S.DENS_EMPTY)
        self.density_view: hs.DensityView | None = None
        self._requested: set[str] = set()
        self._scene_key: str | None = None
        self._image_key: tuple[int, str, tuple[int, ...]] | None = None
        self._legend_line = ""
        self._updating = False

    @property
    def window(self) -> str:
        return str(self.window_box.currentData())

    def set_window(self, window: str) -> None:
        self._select(self.window_box, window)

    def _window_changed(self, _index: int) -> None:
        self.dash.save_settings()
        self.refresh()

    def settings_values(self) -> dict[str, str]:
        return {"density_window": self.window}

    def load_settings_values(self, values: Mapping[str, Any]) -> None:
        self._select(self.window_box, values.get("density_window"))

    def history_reset(self) -> None:
        self._requested.clear()
        self._image_key = None
        self.density_view = None

    def sync_marks(self) -> None:
        pass

    def refresh(self) -> None:
        if self._updating:  # geometry_needed에 실험실이 바로 답하면 그 안에서 다시 부를 수 있다
            self.dirty = True
            return
        self._updating = True
        try:
            super().refresh()
        finally:
            self._updating = False

    def update_data(self) -> None:
        h = self.history
        gen = self.dash.view_generation()
        self._legend_line = ""
        if gen is None or not h.gens:
            self.density_view = None
            self.show_message(S.DENS_EMPTY)
            return
        dv = h.density(gen, self.window)
        self.density_view = dv
        if dv is None:
            self.show_message(S.DENS_NONE if h.entry(gen) is not None else S.DASH_NO_RECORD)
            return
        key = dv.world_key
        geometry = self.dash.world_geometry(key)
        if geometry is None and key not in self._requested:
            self._requested.add(key)
            self.dash.geometry_needed.emit(dv.generation, key)
            geometry = self.dash.world_geometry(key)  # 실험실이 바로 답했을 수 있다
        if geometry is None:
            self.show_message(S.DENS_PREPARING)
            return
        # 답을 받았으면 '청함'을 지운다: 대시보드가 기억하는 세계 그림은 최근 GEOMETRY_CACHE개뿐이라, 밀려난 세계를
        # 다시 보게 되면 또 청해야 한다(청함만 남아 있으면 DENS_PREPARING에 멈춘다).
        self._requested.discard(key)
        if self._scene_key != key:
            self.canvas.set_scene(geometry, None, None)
            self.canvas.set_population([])
            self._scene_key = key
            self._image_key = None
        image_key = (h.version, key, dv.merged)
        if image_key != self._image_key or self.canvas.density_layer() is None:
            self.canvas.set_density(density_image(dv.counts), dv.frame.rect, fit=True)
            self._image_key = image_key
        self.stack.setCurrentWidget(self.host)
        if len(dv.merged) > 1:
            text = S.DENS_LEGEND_MERGED.format(first=dv.merged[0], last=dv.merged[-1], count=len(dv.merged),
                                               cell=dv.frame.cell)
        else:
            text = S.DENS_LEGEND.format(gen=dv.generation, n=dv.population, cell=dv.frame.cell)
        samples = dv.samples
        if samples > 0 and dv.outside / samples >= 0.01:
            text += S.DENS_OUTSIDE.format(pct=dv.outside / samples * 100.0)
        self._legend_line = text
        self.set_notes([S.DENS_CANNOT_MERGE if dv.cannot_merge else "", self.loading_note()])

    def entries(self) -> list[_Entry]:
        if not self._legend_line:
            return []
        v = int(_DENSITY_LUT[255])
        return [_Entry(QColor((v >> 16) & 255, (v >> 8) & 255, v & 255).name(), GLYPH_FILL, self._legend_line)]

    def hover_world(self, x: float, y: float) -> None:
        """세계 좌표 (x, y)의 칸 횟수를 범례 줄에 보인다(DENS_HOVER)."""
        dv = self.density_view
        if dv is None:
            return
        cell = dv.frame.cell_at(x, y)
        if cell is None:
            self._hover_text = ""
        else:
            iy, ix = divmod(cell, dv.frame.nx)
            count = float(dv.counts[iy, ix])
            total = dv.samples
            self._hover_text = S.DENS_HOVER.format(x=x, y=y, count=int(count),
                                                   pct=count / total * 100.0 if total > 0 else 0.0)
        self.update_legend()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if watched is self.canvas:
            if event.type() == QEvent.Type.MouseMove:
                x, y = self.canvas.screen_to_world(event.position())  # type: ignore[attr-defined]
                self.hover_world(x, y)
            elif event.type() == QEvent.Type.Leave:
                self.unhover()
        return super().eventFilter(watched, event)


TILE_CLASSES: dict[str, type[_Tile]] = {
    "genemap": GeneMapTile,
    "density": DensityTile,
    "actions": ActionsTile,
    "body": BodyTile,
}

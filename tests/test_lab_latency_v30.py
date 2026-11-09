"""v3.0 「보는 재미」 지연 고침의 회귀 시험 (v1.2 명세 목표: 「보는 재미」를 보는 동안 입력 지연 100ms 이하, 측정은 v1.2
T39 방법: 이벤트 루프 틈 최대 < 100ms, p99 < 50ms).

v2.1 측정에서 「보는 재미」 쪽을 아래(「몸 수치 추이」·「몸 구조」 줄)까지 내린 채 750세대 바퀴 실행을 열면 최대 틈이
131~323ms였다. 원인과 고침:
- 「몸 수치 추이」 그림 한 번이 20ms 안팎: 몸 수치마다 곡선 항목 둘(31개) → 선 항목 하나·띠 항목 하나(_SeriesLayer,
  그림이 픽셀까지 같음), 축 이름을 바뀔 때만, 반투명 격자선을 같은 픽셀의 사각형 채우기로(use_fast_grid).
- 보통 창 다시 그리기 동안 배경 작업(궤적 밀도 세계 만들기 등)이 GIL을 다툼 → 창의 UpdateRequest 처리 동안도
  GUI_PRIORITY를 바쁨으로(_PaintGuard).
- 그림이 바뀌지 않았는지는 고치기 전 항목(PlotCurveItem, pyqtgraph 축 그리기)으로 그린 것과 픽셀을 비교해 본다.
"""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QCoreApplication, QElapsedTimer, QLineF, QPointF, QRect, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen, QRegion
from PySide6.QtWidgets import QVBoxLayout, QWidget
from pytestqt.qtbot import QtBot
from test_dashboard import make_dashboard
from test_history import fake_generation, stats_row

from evolab import paths
from evolab.app.lab import dashboard as D
from evolab.app.lab import history as hs
from evolab.app.lab import watch_tiles as W
from evolab.app.lab.watch_types import WatchMeta
from evolab.policy.genome import layout_for
from evolab.runs import observe as ob
from evolab.spec.io import compile_env

WHEELS = paths.builtin_env_dir() / "wheels.evoenv"
FRAME = ob.GridFrame(-10.0, -5.0, 0.8, 32, 24)
SURFACE = "#fcfcfb"
ROWS = {"top": 0.0, "middle": 0.5, "bottom": 1.0}
"""「보는 재미」 쪽 스크롤 위치(세로 막대 최대의 비율): 1280×800 창에서 위는 1·2줄, 가운데는 세 줄 모두 일부, 아래는
2·3줄(「몸 수치 추이」·「몸 구조」)이 보인다."""


def _pixels(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    ptr = img.constBits()
    return np.frombuffer(ptr, dtype=np.uint32, count=img.width() * img.height()).reshape(img.height(), img.width()).copy()


def _assert_same_picture(a: np.ndarray, b: np.ndarray, what: Any = "") -> None:
    """두 그림이 같다: 이 기계(리눅스, PySide6 6.12)에서는 픽셀까지 같다. 다른 CPU에서 Qt가 색 섞기를 다른 SIMD 길로 하면
    한 색 성분이 1 다를 수 있어 그만큼은 허용한다(칠한 자리가 다르면 선 색과 바탕색 차이라 1보다 훨씬 크다)."""
    assert a.shape == b.shape
    if np.array_equal(a, b):
        return
    ca = a.view(np.uint8).reshape(*a.shape, 4).astype(np.int16)
    cb = b.view(np.uint8).reshape(*b.shape, 4).astype(np.int16)
    worst = int(np.abs(ca - cb).max())
    assert worst <= 1, (what, worst, int(np.count_nonzero(a != b)))


# ---------------------------------------------------------------------------
# 격자선: drawLine과 같은 픽셀의 fillRect
def _draw_fast(p: QPainter, pen: QPen, a: QPointF, b: QPointF) -> None:
    """_paint_fast_grid가 격자선 하나를 칠하는 것과 같은 방법."""
    t = p.deviceTransform()
    hit = W.hline_pixels(t, a, b, pen)
    if hit is not None:
        row, xs, xe = hit
        p.fillRect(t.inverted()[0].mapRect(QRect(xs, row, xe - xs, 1).toRectF()), pen.color())


def test_hline_pixels_match_qt_cosmetic_lines() -> None:
    """반투명 코스메틱 가로선(폭 0·1, 끝 모양 둘)을 무작위 위치·이동·확대·장치 픽셀 비율(1, 1.25, 1.5)·자르기 영역에서
    drawLine으로 그린 것과 fillRect로 칠한 것이 섞인 색까지 같다(배경은 무작위 색)."""
    rng = random.Random(7)
    w, h = 64, 48
    noises = {dpr: (np.random.default_rng(3).integers(0, 2**32, size=(int(h * dpr), int(w * dpr)), dtype=np.uint64)
                    .astype(np.uint32) | np.uint32(0xFF000000)) for dpr in (1.0, 1.25, 1.5)}
    for _ in range(1500):
        dpr = rng.choice([1.0, 1.0, 1.25, 1.5])  # 창의 장치 픽셀 비율(윈도우 화면 배율 125%·150%)
        noise = noises[dpr]
        y = rng.choice([rng.uniform(-2, h + 2), float(rng.randint(0, h)),
                        rng.randint(0, h) + rng.choice([0.5, 0.999999, 1e-7, 0.015625, 0.0078125])])
        xa = rng.choice([rng.uniform(-6, w + 6), float(rng.randint(-3, w)), rng.randint(0, w) + rng.choice([0.5, 1e-7])])
        xb = rng.choice([rng.uniform(-6, w + 6), float(rng.randint(-3, w)), rng.randint(0, w) + 0.25])
        if xa == xb:
            continue  # 길이 0(점)은 빠른 길로 그리지 않는다(_plain_hline)
        tx, ty = rng.choice([0.0, 0.5, rng.uniform(-5, 5), 3.0]), rng.choice([0.0, 0.5, rng.uniform(-5, 5), 2.0])
        scale = rng.choice([1.0, 1.0, 1.25, 1.5, rng.uniform(0.5, 2.0)])
        color = QColor(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), rng.choice([6, 44, 89, 128, 255]))
        pen = QPen(color)
        pen.setCosmetic(True)
        pen.setWidthF(rng.choice([1.0, 0.0]))
        pen.setCapStyle(rng.choice([Qt.PenCapStyle.SquareCap, Qt.PenCapStyle.FlatCap]))
        clip = QRegion(QRect(rng.randint(0, 20), rng.randint(0, 20), rng.randint(5, 40), rng.randint(5, 30))) \
            if rng.random() < 0.3 else None
        a, b = QPointF(xa, y), QPointF(xb, y)
        out = []
        for fast in (False, True):
            img = QImage(noise.tobytes(), noise.shape[1], noise.shape[0], QImage.Format.Format_ARGB32_Premultiplied).copy()
            img.setDevicePixelRatio(dpr)
            p = QPainter(img)
            if clip is not None:
                p.setClipRegion(clip)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            p.translate(tx, ty)
            p.scale(scale, scale)
            if fast:
                _draw_fast(p, pen, a, b)
            else:
                p.setPen(pen)
                p.drawLine(QLineF(a, b))
            p.end()
            out.append(_pixels(img))
        _assert_same_picture(out[0], out[1], (y, xa, xb, tx, ty, scale, dpr, pen.widthF(), pen.capStyle(), clip))


def _grid_plot(qtbot: QtBot, height: int, y_range: tuple[float, float]) -> pg.PlotWidget:
    w = pg.PlotWidget()
    qtbot.addWidget(w)
    w.setBackground(SURFACE)
    w.resize(560, height)
    plot = w.getPlotItem()
    plot.showGrid(x=False, y=True, alpha=0.35)
    for name in ("left", "bottom"):
        ax = plot.getAxis(name)
        ax.setPen(D._pen("#c3c2b7", 1))
        ax.setTextPen(D._pen("#898781", 1))
    plot.setLabel("left", "범위 대비 %")
    plot.addItem(pg.PlotCurveItem(np.linspace(0, 750, 200), 50 + 40 * np.sin(np.linspace(0, 9, 200)),
                                  pen=D._pen("#2a78d6", 1.5)))
    plot.enableAutoRange(axis="y", enable=False)
    plot.setYRange(*y_range, padding=0.05)
    return w


@pytest.mark.parametrize(("height", "y_range"), [(230, (0.0, 100.0)), (300, (0.0, 100.0)), (420, (0.0, 100.0)),
                                                 (260, (-3.5, 12.25)), (180, (0.37, 0.91))])
def test_fast_grid_axis_draws_the_same_picture_as_the_pyqtgraph_axis(qtbot: QtBot, height: int,
                                                                     y_range: tuple[float, float]) -> None:
    w = _grid_plot(qtbot, height, y_range)
    ax = w.getPlotItem().getAxis("left")
    before = _pixels(w.grab().toImage())
    W.use_fast_grid(ax)
    ax.picture = None
    after = _pixels(w.grab().toImage())
    assert ax._fast_grid is not None and len(ax._fast_grid[1]) >= 5  # 격자선이 빠른 길로 그려졌다
    _assert_same_picture(before, after)
    # 범위가 바뀌어 축 그림을 다시 만들어도 같다
    w.getPlotItem().setYRange(y_range[0] - 1.0, y_range[1] * 2 + 1.0, padding=0.05)
    fast = _pixels(w.grab().toImage())
    del ax.paint  # pyqtgraph 원래 그리기
    ax.picture = None
    _assert_same_picture(_pixels(w.grab().toImage()), fast)


# ---------------------------------------------------------------------------
# 계열 항목(_SeriesLayer) = 계열마다 PlotCurveItem
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#898781", "#898781"]


_KEEP: list[Any] = []
"""그린 그래프 칸을 시험이 끝날 때까지 붙잡아 둔다(지워지면 항목도 지워져 범위를 비교할 수 없다)."""


def _render(qtbot: QtBot, items: list[Any], xr: tuple[float, float], yr: tuple[float, float]) -> QImage:
    w = pg.PlotWidget()
    qtbot.addWidget(w)
    _KEEP.append(w)
    w.resize(330, 180)
    w.setBackground(SURFACE)
    w.hideAxis("left")
    w.hideAxis("bottom")
    for it in items:
        w.addItem(it)
    w.plotItem.vb.enableAutoRange(enable=False)
    w.setXRange(*xr, padding=0.0)
    w.setYRange(*yr, padding=0.0)
    return w.grab().toImage()


def _series(rng: np.random.Generator, n: int, k: int) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    x = np.arange(1.0, n + 1.0)
    lo = rng.uniform(0, 40, (k, n))
    hi = lo + rng.uniform(0, 50, (k, n))
    mean = (lo + hi) / 2
    mean[1, 5:9] = np.nan  # 값이 없는 세대(선이 끊김, 띠에서 빠짐)
    lo[2, 3] = np.nan
    return x, list(mean), [W._band_xy(x, lo[j], hi[j]) for j in range(k)]  # type: ignore[misc]


@pytest.mark.parametrize("n", [12, 750])
def test_series_layer_draws_the_same_picture_as_one_curve_item_per_series(qtbot: QtBot, n: int) -> None:
    rng = np.random.default_rng(11)
    k = 9
    x, means, bands = _series(rng, n, k)
    shown = [j != 4 for j in range(k)]  # 숨긴 계열(수치 하나를 고른 모양)
    old: list[Any] = []
    for j in range(k):
        band = pg.PlotCurveItem(pen=pg.mkPen(None), brush=D._qcolor(COLORS[j], 45), fillLevel="enclosed")
        band.setZValue(-10)
        line = D._curve(D._pen(COLORS[j], 1.5))
        line.setZValue(10)
        band.setData(*bands[j]) if j != 8 else band.setData([], [])
        line.setData(x, means[j])
        band.setVisible(shown[j])
        line.setVisible(shown[j])
        old += [band, line]
    bl, ll = W._SeriesLayer("fill"), W._SeriesLayer("line")
    bl.setZValue(-10)
    ll.setZValue(10)
    bl.set_count(k, D._qcolor(COLORS[0], 45))
    ll.set_count(k, D._pen(COLORS[0], 1.5))
    for j in range(k):
        bl.set_style(j, D._qcolor(COLORS[j], 45))
        ll.set_style(j, D._pen(COLORS[j], 1.5))
        bl.set_series(j, *(bands[j] if j != 8 else (np.zeros(0), np.zeros(0))))
        ll.set_series(j, x, means[j])
        bl.set_shown(j, shown[j])
        ll.set_shown(j, shown[j])
    bl.commit()
    ll.commit()
    xr, yr = (0.0, n + 1.0), (-5.0, 95.0)
    a, b = _pixels(_render(qtbot, old, xr, yr)), _pixels(_render(qtbot, [bl, ll], xr, yr))
    assert np.array_equal(a, b)
    # 자동 범위에 쓰는 값: 보이는 계열 PlotCurveItem들의 합
    for ax in (0, 1):
        bounds = [it.dataBounds(ax) for it in old if it.isVisible()]
        finite = [bd for bd in bounds if bd[0] is not None and np.isfinite(bd[0])]
        lo_hi = (min(bd[0] for bd in finite), max(bd[1] for bd in finite))
        got = [layer.dataBounds(ax) for layer in (bl, ll)]
        assert (min(g[0] for g in got), max(g[1] for g in got)) == pytest.approx(lo_hi)
    assert ll.pixelPadding() == pytest.approx(old[1].pixelPadding()) and bl.pixelPadding() == old[0].pixelPadding() == 0
    assert ll.series_data(1)[1] is not None and len(ll.views()) == k and not ll.views()[4].isVisible()
    _KEEP.clear()


# ---------------------------------------------------------------------------
# 「몸 수치 추이」 타일 전체: 고치기 전 항목(곡선 31개, pyqtgraph 축)과 같은 그림
def _wheels_dashboard(qtbot: QtBot, tmp_path: Path, gens: int, pop: int = 8) -> tuple[D.Dashboard, W.BodyTile]:
    spec = compile_env(WHEELS.read_text(encoding="utf-8"))
    dash = make_dashboard(qtbot, tmp_path, "latency.ini")
    dash.set_watch_meta(WatchMeta.from_spec(spec))
    dash.set_page("fun")
    lay = layout_for(spec)
    for g in range(1, gens + 1):
        obs, X = fake_generation(g, lay, pop=pop)
        dash.add_row({**stats_row(obs), "evaluations": str(g * pop * 2), "elapsed": str(g * 1.5)})
        dash.add_generation(g, obs.to_arrays(), X, obs.representative, None)
    dash.resize(1272, 820)
    dash.show()
    qtbot.waitExposed(dash)
    dash.flush()
    tile = dash.tile("body")
    assert isinstance(tile, W.BodyTile)
    return dash, tile


def _old_body_items(tile: W.BodyTile) -> list[pg.PlotCurveItem]:
    """v2.1 BodyTile이 만들던 항목(수치마다 띠 하나·선 하나, 같은 자료·색·보임)."""
    t = tile.theme
    meta = tile.dash.watch_meta
    assert meta is not None and tile._series is not None
    bs, x, sel = tile._series, tile._x, tile.selected
    lows = np.array([p.low for p in meta.body])
    span = np.array([p.high - p.low for p in meta.body])
    out = []
    for j, color in enumerate(tile._colors):
        band = pg.PlotCurveItem(pen=pg.mkPen(None), brush=D._qcolor(color, 45), fillLevel="enclosed")
        band.setZValue(-10)
        line = D._curve(D._pen(color, 1.5))
        line.setZValue(10)
        if sel < 0:
            line.setData(x, (bs.mean[:, j] - lows[j]) / span[j] * 100.0)
            if color == t.muted:
                band.setData([], [])
            else:
                band.setData(*W._band_xy(x, (bs.lo[:, j] - lows[j]) / span[j] * 100.0,
                                         (bs.hi[:, j] - lows[j]) / span[j] * 100.0))
        elif j == sel:
            line.setPen(D._pen(t.series[0], 1.5))
            line.setData(x, bs.mean[:, j])
            band.setBrush(D._qcolor(t.series[0], 45))
            band.setData(*W._band_xy(x, bs.lo[:, j], bs.hi[:, j]))
        else:
            line.hide()
            band.hide()
        out += [band, line]
    return out


@pytest.mark.parametrize("pick", [0, 3])
def test_body_tile_draws_the_same_picture_as_the_v21_curve_items(qtbot: QtBot, tmp_path: Path, pick: int) -> None:
    """수치 15개(바퀴 고르는 언덕차, 7개부터 「기타」 회색), 「모두」와 수치 하나. 새 그림(계열 항목 둘 + 빠른 격자선)과
    고치기 전 그림(PlotCurveItem 31개 + pyqtgraph 축)이 픽셀까지 같고, 자동 범위도 같다."""
    dash, tile = _wheels_dashboard(qtbot, tmp_path, 60)
    tile.pick.setCurrentIndex(pick)  # 0 = 모두, 3 = 수치 하나
    dash.flush()
    tile.prepare_paint()
    assert tile.line_layer.count == tile.band_layer.count == 15
    new_img = _pixels(tile.view.grab().toImage())
    new_range = tile.plot.vb.viewRange()
    old = _old_body_items(tile)
    tile.band_layer.hide()
    tile.line_layer.hide()
    for it in old:
        tile.plot.addItem(it)
    axis = tile.plot.getAxis("left")
    del axis.paint  # 격자선도 pyqtgraph 원래 그리기
    axis.picture = None
    tile.prepare_paint()
    old_img = _pixels(tile.view.grab().toImage())
    got_range = tile.plot.vb.viewRange()
    assert all(a == pytest.approx(b) for a, b in zip(got_range, new_range, strict=True)), (got_range, new_range)
    _assert_same_picture(old_img, new_img)


def test_body_tile_draws_two_series_items_and_keeps_the_axis_picture(qtbot: QtBot, tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    dash, tile = _wheels_dashboard(qtbot, tmp_path, 30)
    curves = [it for it in tile.plot.items if isinstance(it, pg.PlotCurveItem)]
    assert len(curves) == 1  # 세대 대표 점선만(수치마다 곡선 항목을 두지 않는다)
    assert sum(isinstance(it, W._SeriesLayer) for it in tile.plot.items) == 2
    tile.prepare_paint()
    tile.view.grab()
    axis = tile.plot.getAxis("left")
    picture = axis.picture
    assert picture is not None
    labels: list[Any] = []
    real = tile.plot.setLabel
    monkeypatch.setattr(tile.plot, "setLabel", lambda *a, **k: (labels.append(a), real(*a, **k)))
    obs, X = fake_generation(31, dash.watch_meta.layout, pop=8)  # type: ignore[union-attr]
    dash.add_row(stats_row(obs))
    dash.add_generation(31, obs.to_arrays(), X, obs.representative, None)
    dash.flush()
    tile.prepare_paint()
    tile.view.grab()
    assert labels == [] and axis.picture is picture  # 같은 축 이름을 다시 넣어 축 그림을 버리지 않는다
    assert len(tile.lines) == 15 and len(tile.lines[0].getData()[0]) == 31


def test_structure_tile_counts_a_long_history_at_once_like_generation_by_generation(qtbot: QtBot,
                                                                                     tmp_path: Path) -> None:
    """「몸 구조」 처음 세기(한 번에 numpy, _store_counts)가 세대마다 센 것과 같다: 세대마다 구조 비율(독립 계산과 비교)과
    구조 열쇠 번호(처음 나온 차례: 세대 순서, 세대 안에서는 코드 순서)."""
    from test_structure_tile import direct_ratios, feed, struct_dashboard, tile, wheels

    spec = wheels()
    dash = struct_dashboard(qtbot, tmp_path, spec)
    lay = dash.watch_meta.layout  # type: ignore[union-attr]
    data = feed(dash, range(1, 41), lambda g: fake_generation(g, lay, pop=24)[1])
    t = tile(dash)
    t.history_reset()  # 처음부터 한 번에 센다
    t.refresh()
    expect_keys: list[tuple[str, ...]] = []
    counter = t.counter(0)
    assert counter is not None and counter.uses_bits
    for g in range(1, 41):
        body = lay.body_values(data[g])
        assert t.structure_at(g) == pytest.approx(direct_ratios(spec, body))
        for code in sorted(set(counter.codes(body).tolist())):  # 예전 차례: 세대마다 코드 오름차순으로 열쇠를 더함
            key = tuple(b for b in t._keys_order if b in counter.present_of(code))
            if key not in expect_keys:
                expect_keys.append(key)
    assert t._keys == expect_keys and len(expect_keys) >= 3


def test_body_series_rows_equal_vstack() -> None:
    arrays = [np.random.default_rng(g).normal(size=15) for g in range(40)]
    assert np.array_equal(hs._rows(arrays), np.vstack(arrays)) and hs._rows(arrays).dtype == np.float64


# ---------------------------------------------------------------------------
# 창 다시 그리기 동안 배경 작업이 쉰다(_PaintGuard)
class _Probe(QWidget):
    def __init__(self, hold_s: float = 0.0) -> None:
        super().__init__()
        self.hold_s = hold_s
        self.busy_seen: list[bool] = []
        self.painting = threading.Event()
        self.ended = 0.0
        self.setMinimumSize(120, 80)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        self.busy_seen.append(hs.GUI_PRIORITY.is_busy)
        self.painting.set()
        if self.hold_s:
            time.sleep(self.hold_s)  # 그리기가 오래 걸리는 척(GIL은 놓는다)
        self.ended = time.monotonic()
        QPainter(self).fillRect(self.rect(), QColor("#336699"))


def _guarded_window(qtbot: QtBot, tmp_path: Path, hold_s: float = 0.0) -> tuple[QWidget, _Probe]:
    win = QWidget()
    qtbot.addWidget(win)
    lay = QVBoxLayout(win)
    probe = _Probe(hold_s)
    lay.addWidget(probe)
    lay.addWidget(D.Dashboard(settings=QSettings(str(tmp_path / "guard.ini"), QSettings.Format.IniFormat)))
    win.resize(600, 500)
    win.show()
    qtbot.waitExposed(win)
    return win, probe


def test_window_repaints_mark_the_screen_busy_for_background_work(qtbot: QtBot, tmp_path: Path) -> None:
    win, probe = _guarded_window(qtbot, tmp_path)
    qtbot.wait(80)  # 펌프·처음 그림의 조용한 시간이 지나게
    probe.busy_seen.clear()
    probe.update()
    qtbot.waitUntil(lambda: len(probe.busy_seen) > 0, timeout=5000)
    assert probe.busy_seen[-1] is True  # 보통 그림(펌프가 아닌 update)도 바쁨으로 감싼다
    qtbot.waitUntil(lambda: not hs.GUI_PRIORITY.is_busy, timeout=5000)  # 끝나면 곧 풀린다(PAINT_QUIET_S)
    assert sys.getswitchinterval() <= hs.GUI_SWITCH_INTERVAL_S  # 대시보드가 보이면 GIL 넘김 간격을 줄인다
    # 대시보드를 다른 창으로 옮기면 거르개도 옮긴다
    other = QWidget()
    qtbot.addWidget(other)
    dash = win.findChild(D.Dashboard)
    assert dash is not None and dash._guarded_window is win
    dash.setParent(other)
    QVBoxLayout(other).addWidget(dash)
    other.show()
    qtbot.waitExposed(other)
    assert dash._guarded_window is other


def test_background_wait_returns_only_after_the_window_repaint(qtbot: QtBot, tmp_path: Path) -> None:
    win, probe = _guarded_window(qtbot, tmp_path, hold_s=0.15)
    qtbot.wait(80)
    probe.painting.clear()
    got: list[float] = []

    def background() -> None:
        assert probe.painting.wait(10)
        hs.GUI_PRIORITY.wait(limit=5.0)  # 배경 작업이 일 하나를 시작하기 전(gui_first)
        got.append(time.monotonic())

    th = threading.Thread(target=background)
    th.start()
    probe.update()
    qtbot.waitUntil(lambda: bool(got), timeout=10000)
    th.join(5)
    assert got[0] >= probe.ended  # 그리기가 끝난 뒤에야 일을 시작한다


# ---------------------------------------------------------------------------
# 배경 작업이 다 쓴 Episode는 그 배경 스레드에서 거둔다(view.release_episodes, v3.0 ML 고침): 순환 참조인 Episode를 다음
# 자동 순환 GC가 화면 스레드의 타일 그림 한가운데 푸는 일(바퀴 고르는 언덕차 8배 관전, 2세대: 22~196ms)을 없앤다.
@pytest.fixture()
def episode_refs(monkeypatch: pytest.MonkeyPatch) -> Any:
    """이 테스트 동안 만든 Episode의 약한 참조들. 자동 순환 GC는 끈다(명시한 수집만 Episode를 풀게)."""
    import gc
    import weakref

    from evolab.world.physics2d import episode as E

    refs: list[Any] = []
    real_init = E.Episode.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        refs.append(weakref.ref(self))

    monkeypatch.setattr(E.Episode, "__init__", init)
    gc.collect()
    gc.disable()
    try:
        yield refs
    finally:
        gc.enable()


def test_release_episodes_waits_for_the_gui_then_collects(episode_refs: list[Any],
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    import gc

    from evolab.app.lab import view as V
    from evolab.engine import seeds
    from evolab.world.physics2d.episode import Episode

    spec = compile_env(WHEELS.read_text(encoding="utf-8"))
    Episode(spec, None, seeds.display_trial()).geometry()
    assert len(episode_refs) == 1
    order: list[str] = []
    real_collect = gc.collect
    monkeypatch.setattr(V, "gui_first", lambda stop: order.append("gui_first") or stop())
    monkeypatch.setattr(gc, "collect", lambda *a: order.append("collect") or real_collect(*a))
    V.release_episodes(lambda: False)
    assert order == ["gui_first", "collect"]  # 화면이 바쁘면 끝날 때까지 기다린 뒤에 거둔다
    assert episode_refs[0]() is None


def test_replay_and_world_threads_leave_no_episode_for_the_gui_thread(qtbot: QtBot, episode_refs: list[Any]) -> None:
    """다시 시뮬레이션(_ReplayThread)과 궤적 밀도 세계(_WorldThread)는 끝나기 전에 자기가 만든 Episode를 거둔다."""
    from evolab.app.lab import view as V
    from evolab.engine import seeds

    spec = compile_env((paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8"))
    lay = layout_for(spec)
    genome = np.zeros(lay.dim)
    threads = [V._ReplayThread("t", [(spec, lay, genome, seeds.display_trial(), "brain")]),  # noqa: SLF001
               V._WorldThread("w", spec, seeds.display_trial())]  # noqa: SLF001
    got: list[Any] = []
    for th in threads:
        th.finished_job.connect(lambda tag, result: got.append((tag, result)), Qt.ConnectionType.DirectConnection)
        th.start()
        assert th.wait(120_000)
    assert [tag for tag, _ in got] == ["t", "w"] and not any(isinstance(r, Exception) for _, r in got)
    assert len(episode_refs) == 2 and all(ref() is None for ref in episode_refs)


def test_build_scene_waits_for_the_gui_before_each_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    import gc

    from evolab.app.lab import view as V
    from evolab.engine import seeds

    spec = compile_env(WHEELS.read_text(encoding="utf-8"))
    lay = layout_for(spec)
    order: list[str] = []
    real_collect = gc.collect
    monkeypatch.setattr(gc, "collect", lambda *a: order.append("collect") or real_collect(*a))
    world, agents = V.build_scene(spec, lay, np.zeros((3, lay.dim)), seeds.display_trial(),
                                  cancel=lambda: order.append("cancel") and False)
    assert agents is not None and len(agents) == 3 and world is not None
    assert order.count("collect") == 3
    assert all(order[i - 1] == "cancel" for i, x in enumerate(order) if x == "collect")


# ---------------------------------------------------------------------------
# 750세대 바퀴 실행을 「보는 재미」 쪽 줄마다 스크롤한 채 열 때 이벤트 루프 틈 (T39 방법)
def fake_wheels_run(base: Path, gens: int = 750) -> Path:
    """바퀴 고르는 언덕차의 가짜 실행: 관찰 기록이 있는 세대 대표 파일 gens개(개체 32 × 시험 3, 세대 대표 env_version은
    실제 환경 버전)와 그 통계 줄. 몸 수치 15개라 「몸 수치 추이」·「몸 구조」가 가장 많이 그린다."""
    from evolab.engine.runsetup import create_run
    from evolab.engine.trainer import TrainConfig
    from evolab.runs.store import CHAMPIONS_DIR, STATS_FILE

    text = WHEELS.read_text(encoding="utf-8")
    spec = compile_env(text)
    lay = layout_for(spec)
    store = create_run(base, text, spec, TrainConfig(popsize=32, trials=3), 0)
    version = store.read_run_info()["env_versions"][0]["id"]
    folder = store.root / CHAMPIONS_DIR
    folder.mkdir(exist_ok=True)
    rows: list[dict[str, str]] = []
    for g in range(1, gens + 1):
        obs, X = fake_generation(g, lay, pop=32, trials=3, frame=FRAME, world_key=f"world{g:011d}")
        rep = obs.representative
        arrays = {"genome": X[rep], "score": np.asarray(float(obs.scores[rep])), "env_version": np.asarray(version),
                  **obs.to_arrays()}
        with open(folder / f"gen_{g:06d}_representative.npz", "wb") as f:  # fsync 없이(시험 준비를 빠르게)
            np.savez(f, allow_pickle=False, **arrays)
        rows.append({**stats_row(obs), "evaluations": str(g * 96), "elapsed": str(g * 2.5),
                     "champion_score": "", "champion_generation": "-1"})
    with open(store.root / STATS_FILE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    return store.root


def measure_rows(run: Path, settings_dir: Path) -> dict[str, Any]:
    """줄마다 새 실험실 창(1280×800, 「보는 재미」 쪽)을 띄워 스크롤한 뒤 run을 열고, 기록을 다 읽고 펌프가 타일을 그린 뒤
    800ms까지의 이벤트 루프 틈(10ms 탐침, 열기 직전 틈부터)을 잰다(T39 measure_open과 같은 절차)."""
    from PySide6.QtWidgets import QApplication

    from evolab.app.lab.view import LabView

    app = QApplication.instance() or QApplication([])
    out: dict[str, Any] = {}
    for row, frac in ROWS.items():
        view = LabView(settings=QSettings(str(settings_dir / f"settings_{row}.ini"), QSettings.Format.IniFormat))
        try:
            view.resize(1280, 800)
            view.show()
            for _ in range(20):
                app.processEvents()
            assert view.load_env(WHEELS)
            dash = view.dashboard
            dash.set_page("fun")
            for _ in range(5):
                app.processEvents()
            bar = dash.scroll.verticalScrollBar()
            bar.setValue(round(bar.maximum() * frac))
            for _ in range(5):
                app.processEvents()
            visible = [t for t in dash._orders["fun"] if dash._in_view(dash._tiles[t])]
            gaps: list[float] = []
            clock = QElapsedTimer()
            probe = QTimer()
            probe.setInterval(10)
            probe.timeout.connect(lambda gaps=gaps, clock=clock: gaps.append(float(clock.restart())))
            clock.start()
            probe.start()
            warm = QElapsedTimer()
            warm.start()
            while warm.elapsed() < 100:
                QCoreApplication.processEvents()
            before = len(gaps)
            assert view.open_run(run)
            assert dash.wait_history(60000)
            settle = QElapsedTimer()
            settle.start()
            while settle.elapsed() < 800:
                QCoreApplication.processEvents()
            probe.stop()
            g = gaps[before:]
            out[row] = {"worst": max(g), "p99": float(np.percentile(g, 99)), "n": len(g), "visible": visible,
                        "gens": len(dash.history), "observed": dash.history.load_summary.get("observed")}
        finally:
            view.shutdown()
            view.deleteLater()
            for _ in range(5):
                app.processEvents()
    return out


def _measure_rows_in_child(tmp_path: Path) -> dict[str, Any]:
    run = fake_wheels_run(tmp_path / "fake")
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "EVOLAB_DATA_DIR": str(tmp_path / "data"), "EVOLAB_STATE_DIR": str(tmp_path / "state")}
    out = subprocess.run([sys.executable, __file__, "measure-rows", str(run), str(tmp_path)], env=env,
                         capture_output=True, text=True, timeout=600, check=False)
    assert out.returncode == 0, out.stderr[-4000:]
    m: dict[str, Any] = json.loads(out.stdout.strip().splitlines()[-1])
    for row, r in m.items():
        print(f"\n[lab] open 750-generation wheels run scrolled to the {row} row {r['visible']}: event-loop gap "
              f"worst {r['worst']:.0f} ms, p99 {r['p99']:.0f} ms, n={r['n']}")
        assert r["gens"] == 750 and r["observed"] == 750 and r["n"] >= 20
    assert "body" in m["bottom"]["visible"] and "structure" in m["bottom"]["visible"]
    assert "genemap" in m["top"]["visible"]
    return m


def test_opening_a_750_generation_wheels_run_at_each_tile_row_keeps_the_event_loop_responsive(tmp_path: Path) -> None:
    """빠른 시험(너그러운 한계): 공유 기계·윈도우 CI 실행기에서도 지키는 값(v1.2 목표의 5배·3배). 목표 그대로는 아래
    느린 시험(-m slow)이 잰다. v2.1(고치기 전)은 아래 줄에서 최대 131~323ms였다."""
    m = _measure_rows_in_child(tmp_path)
    for row, r in m.items():
        assert r["worst"] < 500 and r["p99"] < 150, (row, r)


@pytest.mark.slow
def test_opening_a_750_generation_wheels_run_at_each_tile_row_meets_the_v12_latency_targets(tmp_path: Path) -> None:
    """v1.2 목표 그대로(최대 틈 < 100ms, p99 < 50ms), 줄마다. 한가한 기계에서 잰다."""
    m = _measure_rows_in_child(tmp_path)
    for row, r in m.items():
        assert r["worst"] < 100 and r["p99"] < 50, (row, r)


if __name__ == "__main__":  # 측정 자식 프로세스
    if len(sys.argv) == 4 and sys.argv[1] == "measure-rows":
        print(json.dumps(measure_rows(Path(sys.argv[2]), Path(sys.argv[3]))))

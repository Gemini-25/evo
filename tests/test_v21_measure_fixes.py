"""v2.1 측정(9.2 M13) 뒤 고침의 회귀 시험.

- 「몸 구조」 타일의 쌓은 그림은 항목 하나(_StackArea)로 그린다: 계열마다 PlotCurveItem(fillLevel=0, 바탕색 1픽셀 펜, 작은
  누적이 위)을 두던 그림과 픽셀이 같고(계단·영역 둘 다), 자동 범위가 자료 가로 범위를 그대로 덮는다.
- 챔피언 몸 그림의 배경 일(_BodyJob)은 일마다(있음 칸을 비운 환경 컴파일, Episode 둘) 앞에서 화면이 바쁜지 본다(v1.2
  gui_first 규칙). 쉬어 가도 그림은 같다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtGui import QImage
from pytestqt.qtbot import QtBot
from test_structure_tile import feed, random_X, struct_dashboard, tile, wheels, wheels_on

from evolab.app.lab import dashboard as D
from evolab.app.lab import history as hs
from evolab.app.lab import structure_tile as ST
from evolab.policy.genome import layout_for

SURFACE = "#ffffff"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def _cum(rng: np.random.Generator, layers: int, points: int) -> np.ndarray:
    r = rng.random((layers, points))
    r[1, ::3] = 0.0  # 비율 0인 칸(윗선이 아래 계열 윗선과 겹침)
    return np.cumsum(r / r.sum(axis=0) * 100.0, axis=0)


def _render(qtbot: QtBot, items: list[Any], xr: tuple[float, float]) -> QImage:
    w = pg.PlotWidget()
    qtbot.addWidget(w)
    w.resize(320, 170)
    w.setBackground(SURFACE)
    w.hideAxis("left")
    w.hideAxis("bottom")
    for it in items:
        w.addItem(it)
    w.plotItem.vb.enableAutoRange(enable=False)
    w.setXRange(*xr, padding=0.0)
    w.setYRange(-5.0, 105.0, padding=0.0)
    return w.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)


def _old_curves(x: np.ndarray, cum: np.ndarray, step: bool) -> list[pg.PlotCurveItem]:
    """고치기 전 그림: 계열마다 PlotCurveItem(바탕색 1픽셀 펜, 0까지 채움, 작은 누적이 위)."""
    out = []
    n = len(cum)
    for j in range(n):
        c = pg.PlotCurveItem(pen=D._pen(SURFACE, 1), brush=D._qcolor(COLORS[j]), fillLevel=0.0, connect="all")
        c.setZValue(n - j)
        c.setData(x, cum[j], stepMode="center" if step else None)
        out.append(c)
    return out


def _pixels(img: QImage) -> np.ndarray:
    ptr = img.constBits()
    return np.frombuffer(ptr, dtype=np.uint32, count=img.width() * img.height()).reshape(img.height(), img.width()).copy()


@pytest.mark.parametrize("step", [True, False])
def test_stack_area_draws_the_same_picture_as_one_filled_curve_per_series(qtbot: QtBot, step: bool) -> None:
    rng = np.random.default_rng(5)
    points = 24 if step else 260
    gens = np.arange(1.0, points + 1.0)
    x = D._step_edges(gens) if step else gens
    cum = _cum(rng, 4, points)
    old = _pixels(_render(qtbot, _old_curves(x, cum, step), (float(x[0]), float(x[-1]))))
    area = ST._StackArea(D._pen(SURFACE, 1))
    area.set_layers(x, cum, COLORS[:4], step=step)
    new = _pixels(_render(qtbot, [area], (float(x[0]), float(x[-1]))))
    assert old.shape == new.shape
    same = float(np.mean(old == new))
    assert same >= 0.995, same  # 채움 경계의 반올림 몇 픽셀만 다를 수 있다
    # 모든 계열 색이 실제로 보인다(쌓는 순서가 뒤집히면 맨 위 계열 하나만 보인다)
    seen = {int(v) & 0xFFFFFF for v in np.unique(new)}
    for c in COLORS[:4]:
        assert int(c[1:], 16) in seen
    assert area.layer_count == 4
    lx, ly = area.layer_data(-1)
    assert np.array_equal(lx, x) and np.allclose(ly, 100.0)
    assert area.dataBounds(0) == (float(x[0]), float(x[-1])) and area.dataBounds(1) == (0.0, float(cum.max()))


def test_stack_area_with_no_data_draws_nothing_and_has_no_bounds(qtbot: QtBot) -> None:
    area = ST._StackArea(D._pen(SURFACE, 1))
    area.set_layers(np.zeros(0), np.zeros((0, 0)), [], step=False)
    assert area.dataBounds(0) == (None, None) and area.boundingRect().isNull()
    area.set_layers(np.array([1.0, 2.0]), np.zeros((2, 0)), COLORS[:2], step=True)  # 칸 수와 경계 수가 안 맞으면 빈 그림
    assert area.dataBounds(1) == (None, None)
    blank = _pixels(_render(qtbot, [area], (0.0, 3.0)))
    assert np.all((blank & 0xFFFFFF) == 0xFFFFFF)


def test_structure_tile_autorange_covers_the_generations(qtbot: QtBot, tmp_path: Path) -> None:
    spec = wheels()
    dash = struct_dashboard(qtbot, tmp_path, spec)
    lay = dash.watch_meta.layout  # type: ignore[union-attr]
    feed(dash, range(1, 13), random_X(lay))
    t = tile(dash)
    t.prepare_paint()
    (x0, x1), (y0, y1) = t.plot.vb.viewRange()
    assert x0 <= 0.5 and x1 >= 12.5 and x1 - x0 < 14.0  # 칸 경계(세대 ± 0.5)를 덮고 여백은 조금
    assert (y0, y1) == pytest.approx((-5.0, 105.0))  # 세로는 고정(0~100% + 여백 5%)
    assert t.area.layer_count == len(t.series) and t.area.scene() is t.view.scene()


def test_champion_body_job_yields_to_the_screen_before_each_step(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = wheels()
    lay = layout_for(spec)
    full = ST.full_structure_spec(spec)
    body14 = lay.body_values(wheels_on(lay, (1, 4)))
    calls: list[int] = []
    plain = ST.champion_bodies(spec, body14, full)
    paused = ST.champion_bodies(spec, body14, full, pause=lambda: calls.append(1))
    assert calls == [1] and paused == plain  # 없는 부품이 있으면 둘째 Episode 앞에서 한 번
    calls.clear()
    every = lay.body_values(wheels_on(lay, (1, 2, 3, 4)))
    assert ST.champion_bodies(spec, every, full, pause=lambda: calls.append(1)) == ST.champion_bodies(spec, every, full)
    assert calls == []  # 모든 부품이 있으면 Episode 하나뿐

    waits: list[int] = []
    monkeypatch.setattr(hs.GUI_PRIORITY, "wait", lambda stop=None, limit=0.0: waits.append(1))
    job = ST._BodyJob(("tag",), spec, body14, None, need_full=True)
    got: list[Any] = []
    job.done.connect(got.append)
    job.run()  # 이 스레드에서 바로(시작 전, 컴파일 뒤, 둘째 Episode 앞)
    assert len(waits) == 3 and len(got) == 1
    _, tag, result, made_full = got[0]
    assert tag == ("tag",) and made_full is not None and result == plain

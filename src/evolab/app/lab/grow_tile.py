"""대시보드 「보는 재미」 쪽 「뇌 크기」 타일 (v6 설계 4.19·5.4, D38, v6.1 자라는 뇌).

- 자료: 대시보드가 이미 읽는 통계 열 brain_neurons·brain_links·grow(history.GROW_COLUMNS, 자라는 뇌 실행만)라 새 파일 읽기가 없다.
  뉴런·연결은 그 세대 후보를 만든 크기라, 자란 세대(grow = 1) 줄은 자라기 전 크기이고 다음 줄부터 새 크기다.
- 그림: 은닉 뉴런 수 계단(왼쪽 눈금, 0..틀 합으로 고정), 연결 수 점선(오른쪽 눈금, 0..다 자란 연결 수로 고정). 계단은 세대 g의
  값이 세대 g − 1 끝 ~ 세대 g 끝 구간을 덮어서(stepMode "left") 자란 세대 끝의 세로선과 계단이 오르는 자리가 같다. 자란 세대
  세로선은 대시보드 자동 세로선(종류 "grow", dashboard.GROW_MARK_KIND)을 다른 타일과 같은 규칙으로 긋는다. 보는 세대 세로선,
  그래프를 누르면 가장 가까운 세대가 보는 세대가 된다(dash.focus_requested, 실험실이 타임라인을 옮긴다). 가로축은 대시보드 가로축
  고르기(세대·누적 시험·경과 시간)를 따른다.
- 빈 상태: 뇌 없는 환경은 v2.0 DASH_EMPTY_NO_BRAIN, 통계가 아직 없으면 GROW_TILE_EMPTY, 통계에 열이 없으면(크기 고정·옛 실행)
  GROW_TILE_FIXED(틀의 은닉 뉴런 합과 연결 수). 은닉층이 없는 뇌면 「자라기」를 고를 수 없으므로 권유 대신 까닭을 적은
  GROW_TILE_FIXED_NO_HIDDEN.
- 오른쪽(연결) 눈금은 늘 왼쪽(뉴런) 눈금 × (다 자란 연결 수 ÷ 틀 뉴런 합)이다(_sync_links_axis): 점선을 뉴런 좌표로 그리므로,
  타일을 끌거나 확대해 왼쪽 눈금이 바뀌면 오른쪽 눈금도 같이 바뀌어야 점선 높이와 눈금이 맞는다.
- 갱신은 대시보드 펌프를 따른다(보일 때만). 갱신 한 번은 열 셋을 읽어 곡선 둘에 setData하는 것뿐이다.

학습 결과와 무관한 화면 전용 코드다.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

from evolab.app import strings as S
from evolab.app.lab import history as hs
from evolab.app.lab.dashboard import (
    AXIS_WIDTH,
    GLYPH_DASH,
    GLYPH_LINE,
    Dashboard,
    _curve,
    _Entry,
    _pen,
)
from evolab.app.lab.watch_tiles import _focus_line, _WatchTile


def frame_totals(brain: Any) -> tuple[int, int]:
    """뇌 구조(BrainLayout) → (은닉 뉴런 합, 편향 뺀 가중치 수). 은닉층이 있으면 policy.frame.BrainFrame.full_links()와 같다."""
    return int(sum(brain.hidden)), int(sum(fi * fo for fi, fo in brain.shapes))


class GrowTile(_WatchTile):
    """「뇌 크기」: 자라는 뇌 실행의 세대마다 은닉 뉴런 수(계단)와 연결 수(점선), 자란 세대 세로선, 보는 세대 세로선."""

    follows_axis = True

    def __init__(self, dash: Dashboard, tile_id: str) -> None:
        super().__init__(dash, tile_id)
        t = self.theme
        self.plot = self.add_plot(S.GROW_TILE_AXIS_NEURONS)
        # 가로 격자를 끈다(몸 구조 타일과 같은 까닭): 격자를 그리는 축 그림이 펌프 한 단계의 대부분이었다(300세대 측정: 격자 켬
        # 약 8ms, 끔 약 2ms). 계단은 몇 단뿐이라 왼쪽 눈금으로 읽힌다
        self.plot.showGrid(x=False, y=False)
        # 연결 수 눈금: 오른쪽 축에 묶은 둘째 ViewBox(가로는 뉴런 그래프와 묶음, 세로 0..다 자란 연결 수). 점선은 그 눈금에 맞게
        # 뉴런 그래프 좌표로 바꿔 뉴런 계단 위에 그린다: 은닉층이 하나면 연결 수가 뉴런 수에 비례해(착륙선 12배) 두 선이 겹치는데,
        # 아래 ViewBox에 그리면 점선이 계단에 가려진다. 마우스(확대·누르기·풍선)는 뉴런 그래프가 받게 둘째 ViewBox는 아래에 둔다
        self.links_view = pg.ViewBox(enableMenu=False)
        self.links_view.setMouseEnabled(x=False, y=False)
        self.links_view.setZValue(-1)
        self.plot.showAxis("right")
        right = self.plot.getAxis("right")
        right.setPen(_pen(t.axis, 1))
        right.setTextPen(_pen(t.muted, 1))
        right.enableAutoSIPrefix(False)
        right.setWidth(AXIS_WIDTH)
        right.setLabel(S.GROW_TILE_AXIS_LINKS, color=t.ink2)
        self.plot.scene().addItem(self.links_view)
        right.linkToView(self.links_view)
        self.links_view.setXLink(self.plot)
        self.plot.vb.sigResized.connect(self._sync_views)
        self.neurons_curve = _curve(_pen(t.series[0], 2.0), stepMode="left")
        self.neurons_curve.setZValue(10)
        self.plot.addItem(self.neurons_curve)
        self.links_curve = _curve(_pen(t.series[1], 1.5, Qt.PenStyle.DotLine), stepMode="left")
        self.links_curve.setZValue(11)
        self.plot.addItem(self.links_curve)
        self.focus_line = _focus_line(t)
        self.plot.addItem(self.focus_line, ignoreBounds=True)
        self.view.scene().sigMouseClicked.connect(self._on_click)
        self.message.setText(S.GROW_TILE_EMPTY)
        self.stack.setCurrentWidget(self.message)

        # 그린 자료(테스트·마우스가 읽는다)
        self.series: hs.GrowSeries | None = None
        """지금 그린 계열(자라는 뇌 실행이 아니면 None)."""
        self.xs = np.zeros(0)
        """계열 줄마다 가로축 값(QWidget.x()를 가리지 않게 xs)."""
        self.gens = np.zeros(0)
        """계열 줄마다 세대."""
        self.total = 0
        """틀의 은닉 뉴런 합(왼쪽 눈금 끝, 풍선 「n/전체」)."""
        self.full_links = 0
        """다 자란 연결 수(오른쪽 눈금 끝)."""
        self._links_range: tuple[float, float] | None = None
        """오른쪽 눈금에 마지막으로 놓은 세로 범위(_sync_links_axis)."""
        # 끌기·확대·fix_y·보기 되돌리기 모두 이 신호를 낸다(위 값들이 생긴 뒤에 잇는다)
        self.plot.vb.sigYRangeChanged.connect(self._sync_links_axis)

    # -- 대시보드 갈고리 --------------------------------------------------------
    def meta_changed(self) -> None:
        pass

    def history_reset(self) -> None:
        pass

    def _sync_views(self) -> None:
        self.links_view.setGeometry(self.plot.vb.sceneBoundingRect())
        self.links_view.linkedViewChanged(self.plot.vb, self.links_view.XAxis)

    def links_per_neuron(self) -> float:
        """오른쪽(연결) 눈금 ÷ 왼쪽(뉴런) 눈금 = 다 자란 연결 수 ÷ 틀 뉴런 합. 점선은 연결 수를 이 값으로 나눠 뉴런 좌표에 그린다."""
        return float(max(1, self.full_links)) / float(max(1, self.total))

    def _sync_links_axis(self, *_args: Any) -> None:
        """오른쪽 눈금의 세로 범위 = 뉴런 그래프의 지금 세로 범위 × links_per_neuron(여유 없이). 사용자가 타일을 끌거나 확대해도,
        틀이 바뀌어도 점선 높이를 오른쪽 눈금으로 읽으면 그 세대의 연결 수다."""
        (_x0, _x1), (y0, y1) = self.plot.vb.viewRange()
        k = self.links_per_neuron()
        want = (float(y0) * k, float(y1) * k)
        if want != self._links_range:  # 갱신마다 부르므로 바뀔 때만
            self._links_range = want
            self.links_view.setYRange(*want, padding=0.0)

    def reset_view(self) -> None:
        super().reset_view()
        self._sync_links_axis()

    # -- 갱신 ------------------------------------------------------------------
    def update_data(self) -> None:
        d = self.dash
        meta = d.watch_meta
        if meta is not None and meta.layout.brain.n_out == 0:  # 뇌 없는 환경(글라이더): v2.0 빈 상태
            self._empty(S.DASH_EMPTY_NO_BRAIN)
            return
        if d.table.n == 0:
            self._empty(S.GROW_TILE_EMPTY)
            return
        series = hs.grow_series({name: d.col(name) for name in hs.GROW_COLUMNS if d.table.has(name)})
        if series is None:
            if meta is None:
                self._empty(S.GROW_TILE_EMPTY)
            else:
                neurons, links = frame_totals(meta.layout.brain)
                # 은닉층이 없는 뇌는 실험실 「뇌」에서 「자라기」를 고를 수 없다(LAB_BRAIN_GROW_NO_HIDDEN): 권유 대신 까닭을
                text = S.GROW_TILE_FIXED if meta.layout.brain.hidden else S.GROW_TILE_FIXED_NO_HIDDEN
                self._empty(text.format(neurons=neurons, links=links))
            return
        if meta is not None:
            self.total, self.full_links = frame_totals(meta.layout.brain)
        else:  # 이름표가 아직 없다: 지금까지의 최댓값으로 눈금만
            self.total = int(np.nanmax(series.neurons))
            finite = series.links[np.isfinite(series.links)]
            self.full_links = int(finite.max()) if finite.size else 0
        self.series = series
        self.xs = self.x_values()[series.rows]
        self.gens = d.col("generation")[series.rows]
        top = float(max(1, self.total))
        self.neurons_curve.setData(self.xs, series.neurons)
        # 오른쪽 눈금(0..다 자란 연결 수)과 왼쪽 눈금(0..틀 뉴런 합)이 같은 높이가 되게 뉴런 좌표로 바꿔 그린다
        self.links_curve.setData(self.xs, series.links / self.links_per_neuron())
        self.keep_y(self.plot, 0.0, top)
        self._sync_links_axis()  # 세로 범위가 그대로여도 틀(눈금 비)이 바뀌었을 수 있다
        self._sync_views()
        self._place_focus()
        self.stack.setCurrentWidget(self.view)
        self.set_notes([])

    def _empty(self, text: str) -> None:
        self.series = None
        self.xs = np.zeros(0)
        self.gens = np.zeros(0)
        self.neurons_curve.setData([], [])
        self.links_curve.setData([], [])
        self.focus_line.hide()
        self.show_message(text, [])

    def _place_focus(self) -> None:
        gen = self.dash.view_generation()
        if gen is None:
            self.focus_line.hide()
            return
        x = self.dash.x_at_generations(np.array([float(gen)]), True)[0]
        if math.isfinite(x):
            self.focus_line.setPos(float(x))
            self.focus_line.show()
        else:
            self.focus_line.hide()

    # -- 범례·마우스 ------------------------------------------------------------
    def entries(self) -> list[_Entry]:
        s = self.series
        if s is None:
            return []
        t = self.theme
        out = [_Entry(t.series[0], GLYPH_LINE, S.GROW_TILE_NEURONS), _Entry(t.series[1], GLYPH_DASH, S.GROW_TILE_LINKS)]
        if len(s.events):
            out.append(_Entry(t.series[3], GLYPH_DASH, S.GROW_TILE_EVENT))
        return out

    def nearest(self, xv: float) -> int | None:
        """가로축 값 xv에 가장 가까운 계열 줄의 번호(계열 안 번호). 계열이 없으면 None."""
        x = self.xs
        if self.series is None or not len(x) or not np.isfinite(x).any():
            return None
        return int(np.nanargmin(np.abs(x - xv)))

    def tip_text(self, i: int) -> str:
        """계열 i번째 줄의 풍선 글(GROW_TILE_TIP): 「n세대: 뉴런 a/전체개, 연결 b개」."""
        s = self.series
        assert s is not None
        links = float(s.links[i])
        return S.GROW_TILE_TIP.format(x=self.dash.x_text(int(s.rows[i]), True), neurons=int(s.neurons[i]),
                                      total=self.total, links=int(links) if math.isfinite(links) else 0)

    def hover(self, xv: float, yv: float, plot_index: int) -> None:
        i = self.nearest(xv)
        if i is None:
            self.unhover()
            return
        for line in self._cross:
            line.setPos(float(self.xs[i]))
            line.show()
        self._hover_text = self.tip_text(i)
        self.update_legend()

    def click_at(self, xv: float) -> int | None:
        """그래프 가로축 값 xv를 눌렀다: 가장 가까운 세대를 보는 세대로(dash.focus_requested). 그 세대, 없으면 None."""
        i = self.nearest(xv)
        if i is None:
            return None
        gen = self.gens[i]
        if not math.isfinite(float(gen)):
            return None
        self.dash.focus_requested.emit(int(gen))
        return int(gen)

    def _on_click(self, ev: Any) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        pos = ev.scenePos()
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        pt = self.plot.vb.mapSceneToView(pos)
        if self.click_at(pt.x()) is not None:
            ev.accept()


__all__ = ["GrowTile", "frame_totals"]

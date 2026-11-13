"""뇌 들여다보기 (v1.2 설계 4.3, D23·D24): 신경망 가중치 그림과 재생 시각의 뉴런 활성화.

- BrainView: 층마다 세로 한 줄(입력 · 은닉층… · 출력)로 뉴런을 놓고 연결선을 그린다.
  - 연결선 층(정적 그림)은 유전자·크기·테마가 바뀔 때만 QImage에 한 번 그린다. 층마다 가중치 절댓값이 큰 것
    EDGE_CAP개까지, 색 = 부호(+ series[0], − series[1]), 굵기·진하기 EDGE_LEVELS단계(층 안 최대 |w| 대비, 진하기는
    바탕과 미리 섞은 불투명 색), 같은 펜끼리
    drawLines로 묶는다. 그릴 선이 BACKGROUND_EDGES개를 넘으면 안티에일리어싱 없이 배경 스레드에서 그린다(QImage는
    다른 스레드에서 그려도 된다). 글자(층 이름, 입출력 이름)는 화면 스레드에서 정적 그림에 더한다.
  - 프레임마다 정적 그림을 붙이고 뉴런 원만 칠한다(색 = 부호, 알파 = |값|). 다시 그리기는 MAX_FPS 이하.
  - 색 범위는 층마다: 입력 = 관측 ÷ OBS_CLIP(관측은 ±OBS_CLIP로 잘림), tanh 층(출력층은 늘 tanh) = ±1, relu 은닉층 =
    그 재생 전체에서 그 층 |값|의 최댓값(최소 RELU_MIN_SCALE). 추적이 없으면 뉴런은 회색.
- BrainTile: 대시보드 「보는 재미」 쪽 타일. dashboard.brain_source(BrainSource)를 그리고, 없으면 보는 세대까지의
  마지막 챔피언을 멈춘 그림으로 그린다. 실험실이 재생 시각을 Dashboard.show_time → show_time(t)으로 넘긴다.
- v6.1 자라는 뇌(v6 설계 4.18·5.3, D37): set_network(..., active=, fresh=). active(은닉층마다 켜진 뉴런 bool)가 None이면 지금
  그림 그대로(크기 고정·옛 실행, 바이트까지 같음). 꺼진 자리는 그 층 반지름의 옅은 점선 원(채움·값 색 없음, 연결선은 가중치가
  0이라 select_edges가 이미 뺌), fresh(최근 FRESH_GENERATIONS세대 안에 켜진 뉴런)는 범주 색 3의 2px 테두리. 자리는 틀(다 자란
  크기) 그대로라 세대를 넘겨 보면 같은 자리가 채워진다. BrainTile은 자라는 뇌 실행(Dashboard.brain_grows: 통계에 brain_neurons
  열)에서만 유전자로 켜짐을 판단한다(policy.frame.BrainFrame.active_neurons): 크기 고정 실행은 GA의 첫 개체가 처음 유전자(뇌 0)라
  유전자만 보면 모든 뉴런이 꺼져 보인다. 꺼진 자리가 있으면 제목 끝에 BRAIN_SIZE_LABEL, 아래 줄에 BRAIN_GROW_NOTE.
  멈춘 그림(원천 없음)은 보는 세대의 뇌 크기를 따른다: 그때 챔피언의 켜진 뉴런이 보는 세대 대표와 다르면(챔피언은 검증 때만
  바뀌므로 그 뒤에 뇌가 자랐거나, 자동 재시작으로 작은 뇌에서 다시 시작) 챔피언 대신 보는 세대 대표를 그린다(원천 종류 "size",
  제목 BRAIN_TITLE_REP_SIZE). 그래야 세대를 넘겨 볼 때 모든 크기가 차례로 보이고 「뇌 크기」 타일·통계 줄과 같은 크기다.
  「새로 켜짐」 고리는 멈춘 그림이면 보는 세대 g 기준(g − FRESH_GENERATIONS세대 대표와 비교, 설계 4.18), 원천이 있으면 그 원천의 세대.

학습 결과와 무관한 화면 전용 코드다(유전자를 읽기만 한다).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import QEvent, QLineF, QPointF, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QHelpEvent, QImage, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from evolab.app import strings as S
from evolab.app.lab.dashboard import Dashboard, Theme, _blend, _escape
from evolab.app.lab.history import GUI_PRIORITY
from evolab.app.lab.watch_tiles import _WatchTile
from evolab.policy.brain import BrainLayout, unpack
from evolab.policy.frame import BrainFrame, FrameError
from evolab.world.physics2d.episode import OBS_CLIP

if TYPE_CHECKING:
    from evolab.app.lab.watch_types import BrainSource, WatchMeta

EDGE_CAP = 200
"""층마다 그리는 연결선 수의 상한(가중치 절댓값이 큰 것부터)."""
EDGE_LEVELS = 8
"""연결선 굵기·알파 단계 수(층 안 최대 |w| 대비)."""
BACKGROUND_EDGES = 400
"""그릴 선이 이보다 많으면 배경 스레드에서 안티에일리어싱 없이 그린다."""
MAX_FPS = 30
FRAME_MS = 1000 // MAX_FPS
RESIZE_SETTLE_MS = 80
"""크기를 바꾸는 동안은 앞 그림을 늘여 쓰고, 멈추고 이만큼 지나면 다시 그린다."""
RELU_MIN_SCALE = 1e-6
NAME_CHARS = 12
"""입력·출력 이름을 자르는 글자 수."""
PAD = 6
NODE_RADIUS = (1.5, 8.0)
"""뉴런 원 반지름의 최소·최대(화면 픽셀)."""
ELLIPSIS = "…"
FRESH_GENERATIONS = 5
"""v6.1: 보는 세대 g와 g − FRESH_GENERATIONS세대 대표의 켜짐이 다르면 「새로 켜진 뉴런」 고리(설계 7.3)."""
EMPTY_ALPHA = 150
"""꺼진 자리 점선 원의 진하기(0..255, muted 색)."""
FRESH_WIDTH = 2.0
"""새로 켜진 뉴런 고리 굵기(화면 픽셀)."""
ACTIVE_CACHE = 256
"""BrainTile이 기억하는 세대별 켜짐(g − FRESH_GENERATIONS 세대 대표) 수."""


def short_name(name: str, limit: int = NAME_CHARS) -> str:
    """이름을 limit 글자로 자른다(넘으면 끝에 줄임표)."""
    return name if len(name) <= limit else name[: limit - 1] + ELLIPSIS


def _value_text(v: float | None) -> str:
    return "-" if v is None or not math.isfinite(v) else f"{v:+.2f}"


def select_edges(weights: np.ndarray, cap: int = EDGE_CAP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """층 하나(fan_in, fan_out)에서 그릴 연결선: (입력 번호, 출력 번호, 단계 1..EDGE_LEVELS 부호 붙임).

    |w|가 큰 cap개(같은 |w|는 앞 번호 먼저), |w|가 작은 것부터(센 선이 위에 그려지게). 최대 |w|가 0이면 빈 배열.
    """
    w = np.asarray(weights, dtype=np.float64)
    flat = np.abs(w).ravel()
    empty = np.zeros(0, dtype=np.intp)
    if flat.size == 0:
        return empty, empty, empty
    wmax = float(flat.max())
    if not math.isfinite(wmax) or wmax <= 0.0:
        return empty, empty, empty
    order = np.argsort(-flat, kind="stable")[:cap]  # 큰 것부터 cap개(결정적)
    order = order[::-1]  # 작은 것부터 그린다
    mag = flat[order]
    levels = np.clip(np.ceil(mag / wmax * EDGE_LEVELS), 1, EDGE_LEVELS).astype(np.intp)
    levels = np.where(w.ravel()[order] < 0.0, -levels, levels)
    rows, cols = np.divmod(order, w.shape[1])
    keep = mag > 0.0
    return rows[keep], cols[keep], levels[keep]


def _masks(masks: Sequence[np.ndarray] | None, hidden: tuple[int, ...], what: str) -> tuple[np.ndarray, ...] | None:
    """은닉층마다 (그 층 크기,) bool 사본(읽기 전용). None은 None. 층 수·크기가 다르면 ValueError."""
    if masks is None:
        return None
    out = tuple(np.array(m, dtype=bool).reshape(-1) for m in masks)
    if len(out) != len(hidden) or any(m.shape != (n,) for m, n in zip(out, hidden, strict=False)):
        raise ValueError(f"{what} masks do not match the hidden layers {hidden}")
    for m in out:
        m.flags.writeable = False
    return out


def _same_masks(a: tuple[np.ndarray, ...] | None, b: tuple[np.ndarray, ...] | None) -> bool:
    if a is None or b is None:
        return a is b
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def edge_strength(level: int) -> float:
    """연결선 단계(1..EDGE_LEVELS) → 바탕 위 색의 진하기 0..1(가장 센 단계는 범주 색 그대로)."""
    return min(1.0, (40 + 27 * abs(level)) / 255.0)


def _edge_pen(theme: Theme, level: int, antialias: bool) -> QPen:
    # 반투명 대신 바탕과 미리 섞은 불투명 색: 반투명한 굵은 선은 안티에일리어싱 그리기가 20배 넘게 느리다(9-16-3에서
    # 34ms → 1~2ms). 겹친 선은 나중 선(센 선)이 위에 온다.
    color = QColor(_blend(theme.series[0] if level > 0 else theme.series[1], theme.surface, edge_strength(level)))
    k = abs(level)
    pen = QPen(color)
    pen.setWidthF(0.5 + 0.32 * k if antialias else max(1.0, round(0.5 + 0.32 * k)))
    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    return pen


def paint_edges(image: QImage, groups: Sequence[tuple[int, np.ndarray]], theme: Theme, antialias: bool) -> None:
    """연결선 묶음 (부호 붙인 단계, (m, 4) 화면 좌표 x1 y1 x2 y2)을 image에 그린다. 단계가 작은 묶음부터."""
    p = QPainter(image)
    try:
        p.setRenderHint(QPainter.RenderHint.Antialiasing, antialias)
        for level, segs in sorted(groups, key=lambda g: (abs(g[0]), g[0])):
            p.setPen(_edge_pen(theme, level, antialias))
            p.drawLines([QLineF(float(a), float(b), float(c), float(d)) for a, b, c, d in segs])
    finally:
        p.end()


def _blank(size: QSize, dpr: float) -> QImage:
    img = QImage(max(1, round(size.width() * dpr)), max(1, round(size.height() * dpr)),
                 QImage.Format.Format_ARGB32_Premultiplied)
    img.setDevicePixelRatio(dpr)
    img.fill(Qt.GlobalColor.transparent)
    return img


class _EdgeJob(QThread):
    """큰 신경망의 연결선 층을 배경에서 그린다(안티에일리어싱 없음)."""

    done = Signal(object)
    """(꼬리표, QImage)"""

    def __init__(self, tag: Any, size: QSize, dpr: float, groups: list[tuple[int, np.ndarray]], theme: Theme) -> None:
        super().__init__()
        self.tag = tag
        self.args = (size, dpr, groups, theme)

    def run(self) -> None:  # 배경 스레드
        size, dpr, groups, theme = self.args
        GUI_PRIORITY.wait(self.isInterruptionRequested)  # 펌프가 그리는 중이면 끝난 뒤에(GIL을 다투지 않게)
        img = _blank(size, dpr)
        paint_edges(img, groups, theme, antialias=False)
        self.done.emit((self.tag, img))


_LIVE_JOBS: set[QThread] = set()
"""도는 중인 연결선 스레드를 끝날 때까지 붙잡아 둔다(뷰가 먼저 사라져도 돌던 QThread가 지워지지 않게)."""


def _forget_job(job: QThread) -> None:
    _LIVE_JOBS.discard(job)
    job.deleteLater()


def wait_jobs() -> None:
    """프로그램을 끝낼 때: 도는 연결선 그리기를 끝날 때까지 기다린다(주인 없는 QThread가 돌던 중에 지워지면 abort)."""
    for job in list(_LIVE_JOBS):
        job.wait()


class BrainView(QWidget):
    """신경망 하나의 그림. set_network(구조, 뇌 파라미터, 입출력 이름) → set_trace(추적) → set_time(재생 시각)."""

    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.theme = theme
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(160, 100)
        self.setMouseTracking(True)
        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() * 0.85))
        self._font = font
        self.brain: BrainLayout | None = None
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.inputs: list[tuple[str, str]] = []
        self.outputs: list[tuple[str, str]] = []
        self.active: tuple[np.ndarray, ...] | None = None
        """v6.1: 은닉층마다 켜진 뉴런(bool). None이면 모두 켜진 지금 그림."""
        self.fresh: tuple[np.ndarray, ...] | None = None
        """v6.1: 은닉층마다 최근 FRESH_GENERATIONS세대 안에 켜진 뉴런(bool, 고리). None이면 고리 없음."""
        self._net_key: tuple[Any, ...] | None = None
        self._selected: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.edges_total = 0
        """연결 수(모든 층)."""
        self.edges_drawn = 0
        """그리는 연결선 수(층마다 EDGE_CAP개까지)."""
        self.capped = False
        """어느 층이든 EDGE_CAP개를 넘어 일부만 그렸는가."""
        # 추적
        self.observations: np.ndarray | None = None
        self.activations: list[np.ndarray] | None = None
        self.dt = 1.0 / 60.0
        self.scales: list[float] = []
        """층마다(입력 포함) 색 범위: 값 ÷ scale이 -1..1."""
        self._t = 0.0
        self._k: int | None = None
        # 그림
        self._pos: list[np.ndarray] = []
        self._radius: list[float] = []
        self._layout_key: tuple[Any, ...] | None = None
        self._label_geom: tuple[float, float, float, float, QFontMetrics] = (0.0, 0.0, 0.0, 0.0, QFontMetrics(font))
        self._static: QImage | None = None
        self._static_key: tuple[Any, ...] | None = None
        self._job_key: tuple[Any, ...] | None = None
        self._job_tag = 0
        self.edge_builds = 0
        """연결선 층을 새로 그린 횟수(같은 유전자·크기면 늘지 않는다: 시험·측정용)."""
        self.frames = 0
        """paintEvent 횟수."""
        self.last_paint_ms = 0.0
        self._frame_timer = QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.setInterval(FRAME_MS)
        self._frame_timer.timeout.connect(self._frame_due)
        self._frame_pending = False
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(RESIZE_SETTLE_MS)
        self._resize_timer.timeout.connect(self.update)

    # -- 자료 ------------------------------------------------------------------
    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(320, 180)

    def set_theme(self, theme: Theme) -> None:
        if theme != self.theme:
            self.theme = theme
            self.update()

    def set_network(self, brain: BrainLayout, params: np.ndarray, inputs: Sequence[tuple[str, str]],
                    outputs: Sequence[tuple[str, str]], theme: Theme | None = None, *,
                    active: Sequence[np.ndarray] | None = None, fresh: Sequence[np.ndarray] | None = None) -> None:
        """신경망 하나: 구조와 뇌 파라미터 (n_params,)(layout.brain_params의 한 행), 입력·출력 (이름, 풍선 글).
        구조·파라미터·이름이 같으면 아무것도 하지 않는다(연결선 층을 다시 그리지 않는다).

        v6.1: active·fresh는 은닉층마다 (그 층 크기,) bool(켜진 뉴런, 최근에 켜진 뉴런). None이면 지금 그림 그대로(바이트까지).
        뉴런 원만 바뀌므로 연결선 층은 다시 그리지 않는다. 층 수·크기가 다르면 ValueError."""
        if theme is not None:
            self.set_theme(theme)
        act = _masks(active, brain.hidden, "active")
        ring = _masks(fresh, brain.hidden, "fresh")
        if not (_same_masks(act, self.active) and _same_masks(ring, self.fresh)):
            self.active, self.fresh = act, ring
            self.update()
        flat = np.ascontiguousarray(np.asarray(params, dtype=np.float64).reshape(-1))
        ins = [(str(a), str(b)) for a, b in inputs]
        outs = [(str(a), str(b)) for a, b in outputs]
        key = (brain, flat.tobytes(), tuple(ins), tuple(outs))
        if key == self._net_key:
            return
        net = unpack(brain, flat)
        self.brain = brain
        self.weights = [w[0] for w in net.weights]
        self.biases = [b[0] for b in net.biases]
        self.inputs, self.outputs = ins, outs
        self._net_key = key
        self._selected = [select_edges(w) for w in self.weights]
        self.edges_total = sum(int(w.size) for w in self.weights)
        self.edges_drawn = sum(len(r) for r, _c, _l in self._selected)
        self.capped = any(w.size > EDGE_CAP for w in self.weights)
        self._layout_key = None
        if self.observations is not None and (self.observations.shape[1] != brain.n_in or self.activations is None
                                              or len(self.activations) != len(brain.shapes)):
            self.set_trace(None, None, self.dt)
        self.update()

    def clear_network(self) -> None:
        self.brain = None
        self.active = self.fresh = None
        self.weights, self.biases = [], []
        self._net_key = None
        self._selected = []
        self.edges_total = self.edges_drawn = 0
        self.capped = False
        self._static = None
        self._static_key = None
        self.set_trace(None, None, self.dt)
        self.update()

    @property
    def has_network(self) -> bool:
        return self.brain is not None

    @property
    def empty_slots(self) -> int:
        """v6.1: 꺼진(점선 원) 은닉 뉴런 자리 수."""
        return 0 if self.active is None else int(sum(int((~m).sum()) for m in self.active))

    @property
    def fresh_count(self) -> int:
        """v6.1: 새로 켜진 뉴런 고리 수(꺼진 자리에는 그리지 않음)."""
        if self.fresh is None:
            return 0
        if self.active is None:
            return int(sum(int(m.sum()) for m in self.fresh))
        return int(sum(int((f & a).sum()) for f, a in zip(self.fresh, self.active, strict=True)))

    def node_state(self, layer: int, j: int) -> str:
        """v6.1: 뉴런 하나의 그림 상태(층 번호는 입력 0): "on"(보통), "empty"(꺼진 자리), "fresh"(새로 켜짐 고리)."""
        hidden = layer - 1
        if self.brain is None or not 0 <= hidden < len(self.brain.hidden):
            return "on"
        if self.active is not None and not bool(self.active[hidden][j]):
            return "empty"
        if self.fresh is not None and bool(self.fresh[hidden][j]):
            return "fresh"
        return "on"

    def set_trace(self, observations: np.ndarray | None, activations: Sequence[np.ndarray] | None,
                  dt: float) -> None:
        """재생 추적: 스텝마다 입력 (스텝, n_in)과 층마다 활성화 (스텝, 폭). None이면 뉴런을 회색으로."""
        self.dt = float(dt) if dt and dt > 0 else self.dt
        if observations is None or activations is None or self.brain is None:
            self.observations = None
            self.activations = None
            self.scales = []
        else:
            obs = np.asarray(observations)
            acts = [np.asarray(a) for a in activations]
            if obs.ndim != 2 or obs.shape[1] != self.brain.n_in or len(acts) != len(self.brain.shapes) or \
                    any(a.ndim != 2 or a.shape[0] != obs.shape[0] or a.shape[1] != fo
                        for a, (_fi, fo) in zip(acts, self.brain.shapes, strict=True)):
                raise ValueError("trace does not match the network")
            self.observations = obs
            self.activations = acts
            last = len(acts) - 1
            scales = [float(OBS_CLIP)]
            for k, a in enumerate(acts):
                if self.brain.activation == "relu" and k < last:
                    top = float(np.max(np.abs(a))) if a.size else 0.0
                    scales.append(max(RELU_MIN_SCALE, top if math.isfinite(top) else RELU_MIN_SCALE))
                else:
                    scales.append(1.0)
            self.scales = scales
        self._k = None
        self.set_time(self._t)

    @property
    def traced(self) -> bool:
        return self.observations is not None and self.activations is not None

    @property
    def steps(self) -> int:
        return 0 if self.observations is None else int(self.observations.shape[0])

    @property
    def relu_scaled(self) -> bool:
        """relu 은닉층이 있어 층마다 그 재생의 최댓값으로 색을 맞췄는가(범례에 BRAIN_RELU_NOTE)."""
        return self.brain is not None and self.brain.activation == "relu" and bool(self.brain.hidden)

    def step_index(self, t: float | None = None) -> int | None:
        """재생 시각 t(초)의 스텝 번호: ⌊t/dt⌋를 0..스텝−1로 자른다(InspectPanel 규칙). 추적이 없으면 None."""
        n = self.steps
        if n == 0:
            return None
        tt = self._t if t is None else float(t)
        return int(min(n - 1, max(0, math.floor(tt / self.dt + 1e-9))))

    def set_time(self, t: float) -> None:
        """재생 시각(초). 스텝이 바뀌었을 때만 다시 그린다(초당 MAX_FPS번 이하)."""
        self._t = float(t)
        k = self.step_index()
        if k == self._k:
            return
        self._k = k
        self._request_frame()

    def neuron_values(self) -> list[np.ndarray] | None:
        """지금 스텝의 층마다(입력 포함) 원래 값. 추적이 없으면 None."""
        k = self._k if self._k is not None else self.step_index()
        if k is None or self.observations is None or self.activations is None:
            return None
        return [np.asarray(self.observations[k], dtype=np.float64)] + \
            [np.asarray(a[k], dtype=np.float64) for a in self.activations]

    def neuron_levels(self) -> list[np.ndarray] | None:
        """지금 스텝의 층마다 색 값 -1..1(원래 값 ÷ 층 색 범위). 추적이 없으면 None."""
        values = self.neuron_values()
        if values is None:
            return None
        return [np.clip(v / s, -1.0, 1.0) for v, s in zip(values, self.scales, strict=True)]

    # -- 배치 ------------------------------------------------------------------
    def _ensure_layout(self) -> None:
        brain = self.brain
        key = (self._net_key, self.width(), self.height())
        if brain is None or key == self._layout_key:
            return
        self._layout_key = key
        w, h = float(self.width()), float(self.height())
        fm = QFontMetrics(self._font)
        limit = w * 0.28
        # 이름 칸 = 글 너비 + 여백 + 가장 큰 뉴런 반지름(이름은 원 바깥에서 시작하므로, 빼면 긴 출력 이름 끝이 잘린다)
        edge = PAD * 2 + NODE_RADIUS[1]
        left = min(limit, max((fm.horizontalAdvance(short_name(n)) for n, _ in self.inputs), default=0)) + edge
        right = min(limit, max((fm.horizontalAdvance(short_name(n)) for n, _ in self.outputs), default=0)) + edge
        top = fm.height() + PAD
        bottom = PAD
        sizes = brain.sizes
        ncol = len(sizes)
        span = max(1.0, w - left - right)
        gap = span / max(1, ncol - 1)
        self._pos = []
        self._radius = []
        for k, n in enumerate(sizes):
            x = left + (span * k / (ncol - 1) if ncol > 1 else span / 2)
            spacing = max(1.0, h - top - bottom) / max(1, n)
            r = float(np.clip(min(spacing * 0.38, gap * 0.12), *NODE_RADIUS))
            ys = top + spacing * (np.arange(n, dtype=np.float64) + 0.5)
            self._pos.append(np.column_stack([np.full(n, x), ys]))
            self._radius.append(r)
        self._label_geom = (left, right, top, gap, fm)

    def _headers(self) -> list[str]:
        assert self.brain is not None
        hidden = [S.BRAIN_HIDDEN.format(k=i + 1) for i in range(len(self.brain.hidden))]
        return [S.BRAIN_INPUTS, *hidden, S.BRAIN_OUTPUTS]

    def _edge_groups(self) -> list[tuple[int, np.ndarray]]:
        groups: dict[int, list[np.ndarray]] = {}
        for k, (rows, cols, levels) in enumerate(self._selected):
            if not len(rows):
                continue
            a, b = self._pos[k][rows], self._pos[k + 1][cols]
            segs = np.column_stack([a, b])
            for lev in np.unique(levels):
                groups.setdefault(int(lev), []).append(segs[levels == lev])
        return [(lev, np.concatenate(parts)) for lev, parts in groups.items()]

    def _wanted_key(self) -> tuple[Any, ...]:
        dpr = self.devicePixelRatioF()
        return (self._net_key, self.width(), self.height(), dpr, self.theme)

    def _compose(self, edges: QImage | None, key: tuple[Any, ...]) -> None:
        """정적 그림 = 바탕 + 연결선 층 + 층 이름·입출력 이름(화면 스레드)."""
        dpr = float(key[3])
        img = _blank(self.size(), dpr)
        img.fill(QColor(self.theme.surface))
        if edges is not None:
            p = QPainter(img)
            p.drawImage(QRectF(0, 0, self.width(), self.height()), edges)
            p.end()
        self._paint_labels(img)
        self._static = img
        self._static_key = key

    def _paint_labels(self, img: QImage) -> None:
        if self.brain is None:
            return
        left, right, top, gap, fm = self._label_geom
        p = QPainter(img)
        try:
            p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            p.setFont(self._font)
            p.setPen(QColor(self.theme.ink2))
            col_w = max(20.0, gap - 4)
            for k, text in enumerate(self._headers()):
                x = float(self._pos[k][0, 0]) if len(self._pos[k]) else left
                shown = fm.elidedText(text, Qt.TextElideMode.ElideRight, int(col_w))
                p.drawText(QRectF(x - col_w / 2, 0, col_w, top), Qt.AlignmentFlag.AlignCenter, shown)
            p.setPen(QColor(self.theme.ink))
            h = fm.height()
            for k, names, align, room in ((0, self.inputs, Qt.AlignmentFlag.AlignRight, left),
                                          (len(self._pos) - 1, self.outputs, Qt.AlignmentFlag.AlignLeft, right)):
                r = self._radius[k] if self._radius else 0.0
                for j, (name, _label) in enumerate(names):
                    if j >= len(self._pos[k]):
                        break
                    x, y = float(self._pos[k][j, 0]), float(self._pos[k][j, 1])
                    box_w = max(10.0, room - PAD - r)
                    shown = fm.elidedText(short_name(name), Qt.TextElideMode.ElideRight, int(box_w))
                    if align == Qt.AlignmentFlag.AlignRight:
                        rect = QRectF(x - r - PAD / 2 - box_w, y - h / 2, box_w, h)
                    else:
                        rect = QRectF(x + r + PAD / 2, y - h / 2, box_w, h)
                    p.drawText(rect, align | Qt.AlignmentFlag.AlignVCenter, shown)
        finally:
            p.end()

    def build_static(self) -> bool:
        """연결선 층이 지금 크기·유전자와 맞지 않으면 다시 만든다. 작은 신경망은 바로, 큰 신경망은 배경 스레드로.
        바로 만들었거나 이미 맞으면 True."""
        if self.brain is None:
            return True
        self._ensure_layout()
        key = self._wanted_key()
        if key == self._static_key:
            return True
        groups = self._edge_groups()
        if self.edges_drawn <= BACKGROUND_EDGES:
            dpr = float(key[3])
            edges = _blank(self.size(), dpr)
            paint_edges(edges, groups, self.theme, antialias=True)
            self.edge_builds += 1
            self._compose(edges, key)
            return True
        if key != self._job_key:
            self._job_key = key
            self._job_tag += 1
            job = _EdgeJob((self._job_tag, key), self.size(), float(key[3]), groups, self.theme)
            job.done.connect(self._edges_ready)
            job.finished.connect(lambda j=job: _forget_job(j))
            _LIVE_JOBS.add(job)
            job.start()
        return False

    def _edges_ready(self, payload: tuple[Any, QImage]) -> None:
        (tag, key), img = payload
        if tag != self._job_tag or key != self._wanted_key():
            return  # 그동안 유전자·크기가 바뀌었다
        self._job_key = None
        self.edge_builds += 1
        self._compose(img, key)
        self.update()

    def wait_edges(self, timeout_ms: int = 10000) -> bool:
        """시험·측정: 연결선 층이 지금 크기·유전자로 준비될 때까지 이벤트를 처리하며 기다린다."""
        from PySide6.QtCore import QCoreApplication, QElapsedTimer

        clock = QElapsedTimer()
        clock.start()
        while not self.build_static():
            if clock.elapsed() > timeout_ms:
                return False
            QCoreApplication.processEvents()
            for job in list(_LIVE_JOBS):
                job.wait(5)
        return True

    # -- 그리기 ----------------------------------------------------------------
    def _request_frame(self) -> None:
        if self._frame_timer.isActive():
            self._frame_pending = True
            return
        self.update()
        self._frame_timer.start()

    def _frame_due(self) -> None:
        if self._frame_pending:
            self._frame_pending = False
            self.update()
            self._frame_timer.start()

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        old = event.oldSize()
        if old.isValid() and old != event.size() and self._same_network_static():
            self._resize_timer.start()  # 크기를 바꾸는 동안은 앞 그림을 늘여 쓴다

    def _same_network_static(self) -> bool:
        return self._static is not None and self._static_key is not None and self._static_key[0] == self._net_key

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        from time import perf_counter

        t0 = perf_counter()
        p = QPainter(self)
        try:
            if self.brain is None:
                p.fillRect(self.rect(), QColor(self.theme.surface))
                return
            if self._resize_timer.isActive() and self._same_network_static():
                self._ensure_layout()  # 크기를 바꾸는 중: 앞 그림을 늘여 쓰고 멈춘 뒤 다시 그린다
            else:
                self.build_static()
            static = self._static
            if static is not None and self._same_network_static():
                if self._static_key[1:3] == (self.width(), self.height()):
                    p.drawImage(0, 0, static)
                else:
                    p.drawImage(QRectF(self.rect()), static)
            else:
                p.fillRect(self.rect(), QColor(self.theme.surface))
            self._paint_nodes(p)
        finally:
            p.end()
            self.frames += 1
            self.last_paint_ms = (perf_counter() - t0) * 1000.0

    def _paint_nodes(self, p: QPainter) -> None:
        self._ensure_layout()
        levels = self.neuron_levels()
        t = self.theme
        pos_c, neg_c = QColor(t.series[0]), QColor(t.series[1])
        gray = QColor(t.muted)
        gray.setAlpha(110)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        outline = QPen(QColor(t.muted))
        outline.setWidthF(1.0)
        p.setPen(outline)
        surface = QBrush(QColor(t.surface))
        empty_pen = ring_pen = None
        if self.active is not None or self.fresh is not None:  # v6.1: 꺼진 자리·새 뉴런(없으면 지금 그림 그대로)
            faint = QColor(t.muted)
            faint.setAlpha(EMPTY_ALPHA)
            empty_pen = QPen(faint)
            empty_pen.setWidthF(1.0)
            empty_pen.setStyle(Qt.PenStyle.CustomDashLine)
            empty_pen.setDashPattern([2.0, 2.0])
            ring_pen = QPen(QColor(t.series[3]))
            ring_pen.setWidthF(FRESH_WIDTH)
        last = len(self._pos) - 1
        for k, centers in enumerate(self._pos):
            r = self._radius[k]
            vals = levels[k] if levels is not None else None
            act = self.active[k - 1] if self.active is not None and 0 < k < last else None
            ring = self.fresh[k - 1] if self.fresh is not None and 0 < k < last else None
            for j in range(len(centers)):
                c = QPointF(float(centers[j, 0]), float(centers[j, 1]))
                if act is not None and not act[j]:
                    p.setPen(empty_pen)
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawEllipse(c, r, r)
                    p.setPen(outline)
                    continue
                p.setBrush(surface)
                p.drawEllipse(c, r, r)
                if vals is None:
                    p.setBrush(gray)
                else:
                    v = float(vals[j])
                    col = QColor(pos_c if v >= 0.0 else neg_c)
                    col.setAlpha(round(255 * min(1.0, abs(v))) if math.isfinite(v) else 0)
                    p.setBrush(col)
                p.drawEllipse(c, r, r)
                if ring is not None and ring[j]:
                    p.setPen(ring_pen)
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawEllipse(c, r, r)
                    p.setPen(outline)

    # -- 풍선 도움말 -----------------------------------------------------------
    def node_at(self, x: float, y: float) -> tuple[int, int] | None:
        """화면 좌표의 뉴런 (층 번호(입력 0), 뉴런 번호). 원 둘레 3픽셀 안에서 가장 가까운 것."""
        if self.brain is None:
            return None
        self._ensure_layout()
        best: tuple[float, int, int] | None = None
        for k, centers in enumerate(self._pos):
            if not len(centers):
                continue
            d = np.hypot(centers[:, 0] - x, centers[:, 1] - y)
            j = int(np.argmin(d))
            if d[j] <= self._radius[k] + 3.0 and (best is None or d[j] < best[0]):
                best = (float(d[j]), k, j)
        return None if best is None else (best[1], best[2])

    def node_tip(self, layer: int, j: int) -> str:
        """뉴런 하나의 풍선 글(BRAIN_NODE_*_TIP): 지금 값(추적이 없으면 -)과 편향."""
        values = self.neuron_values()
        value = None if values is None else float(values[layer][j])
        last = len(self._pos) - 1
        if layer == 0:
            name, label = self.inputs[j] if j < len(self.inputs) else (str(j), "")
            return S.BRAIN_NODE_INPUT_TIP.format(name=name, label=label, value=_value_text(value))
        bias = _value_text(float(self.biases[layer - 1][j]))
        if layer == last:
            name, label = self.outputs[j] if j < len(self.outputs) else (str(j), "")
            return S.BRAIN_NODE_OUTPUT_TIP.format(name=name, label=label, value=_value_text(value), bias=bias)
        tip = S.BRAIN_NODE_HIDDEN_TIP.format(k=layer, j=j + 1, value=_value_text(value), bias=bias)
        state = self.node_state(layer, j)  # v6.1: 꺼진 자리·새로 켜진 뉴런은 끝에 한 줄
        if state == "empty":
            return f"{tip}\n{S.BRAIN_NODE_EMPTY}"
        if state == "fresh":
            return f"{tip}\n{S.BRAIN_NODE_NEW.format(gen=FRESH_GENERATIONS)}"
        return tip

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ToolTip and isinstance(event, QHelpEvent):
            hit = self.node_at(float(event.pos().x()), float(event.pos().y()))
            if hit is None:
                QToolTip.hideText()
                event.ignore()
            else:
                QToolTip.showText(event.globalPos(), self.node_tip(*hit), self)
            return True
        return super().event(event)


class BrainTile(_WatchTile):
    """「보는 재미」의 뇌 들여다보기 타일(설계 4.3). 원천은 dashboard.brain_source(D24):

    playing(캔버스가 추적 재생 중인 개체) · live(관전 중 그 세대 대표를 시험 0번으로 따로 다시 돌린 것) · 없으면 보는
    세대까지의 마지막 챔피언 멈춘 그림. 그때까지 챔피언이 없으면(검증 전, 고속 모드의 첫 세대들) 보는 세대 대표의 멈춘
    그림(경주의 RACE_LANE_REP_INSTEAD와 같은 규칙). 연결선 그림은 유전자가 바뀔 때만 다시 만든다.
    v6.1 자라는 뇌 실행: 그때 챔피언의 뇌 크기가 보는 세대와 다르면 보는 세대 대표의 멈춘 그림(원천 "size", _resized_rep).
    """

    follows_axis = False

    def __init__(self, dash: Dashboard, tile_id: str) -> None:
        super().__init__(dash, tile_id)
        self.brain = BrainView(self.theme)
        self.stack.addWidget(self.brain)
        self._title = ""
        self._title_parts: tuple[str, str, float] | None = None
        """(종류, 이름 또는 세대 글, 재생 길이(초)): show_time이 제목의 시각만 바꾼다."""
        self.source_kind: str | None = None
        """지금 그리는 원천: "playing" | "live" | "champion" | "rep"(챔피언이 없어 보는 세대 대표) | "size"(v6.1 자라는 뇌 실행:
        그때 챔피언의 뇌 크기가 보는 세대와 달라 보는 세대 대표) | None(빈 상태)."""
        self.source_generation: int | None = None
        self.champion_generation: int | None = None
        """원천이 "size"일 때 그리지 않은 그때 챔피언의 세대(제목 BRAIN_TITLE_REP_SIZE), 아니면 None."""
        self.size_label = ""
        """v6.1: 제목 끝 BRAIN_SIZE_LABEL(꺼진 자리가 있을 때만, 없으면 빈 글)."""
        self._frame: tuple[Any, BrainFrame | None] | None = None
        """(유전자 배치, 틀 또는 None(은닉층·뇌 없음)): 이름표가 바뀔 때만 다시 만든다."""
        self._active_cache: dict[int, tuple[np.ndarray, ...]] = {}
        """세대 → 그 세대 대표의 켜진 뉴런(새로 켜진 뉴런 고리, 세대를 넘겨 봐도 파일을 다시 읽지 않게)."""
        self._masks_memo: tuple[BrainFrame, tuple[str, tuple[int, ...], bytes], tuple[np.ndarray, ...]] | None = None
        """(틀, (dtype, 모양, 유전자 바이트), 켜진 뉴런): 마지막으로 켜짐을 판단한 유전자 하나(_masks_of)."""
        self.show_message(S.BRAIN_EMPTY, [])

    # -- 대시보드 갈고리 --------------------------------------------------------
    def meta_changed(self) -> None:
        self.brain.clear_network()
        self._frame = None
        self._active_cache.clear()
        self._masks_memo = None

    def history_reset(self) -> None:
        self._active_cache.clear()

    def sync_marks(self) -> None:
        pass

    def legend_html(self, i: int | None) -> str:
        return _escape(self._title)

    def show_time(self, t: float) -> None:
        """재생 시각(초): 뉴런 색과 제목의 시각만 바꾼다(연결선 층은 그대로)."""
        self.brain.set_time(t)
        self._set_title_time(t)

    # -- 갱신 ------------------------------------------------------------------
    def _meta(self) -> WatchMeta | None:
        return self.dash.watch_meta

    def update_data(self) -> None:
        meta = self._meta()
        if meta is None:
            self._empty()
            return
        if meta.layout.brain.n_out == 0:  # 뇌 없는 환경(글라이더): 몸 수치만 진화한다(v2.0 설계 D20)
            self._empty(S.DASH_EMPTY_NO_BRAIN)
            return
        src: BrainSource | None = self.dash.brain_source
        notes: list[str] = []
        lay = meta.layout
        champion_gen: int | None = None
        if src is not None and src.genome.shape == (meta.dim,):
            genome = src.genome
            kind, gen = src.kind, src.generation
            rep = src.replay
            fresh_gen = gen
        else:
            gen_now = self.dash.view_generation()
            champ = self.dash.champion_at(gen_now if gen_now is not None else 2**62)
            resized = self._resized_rep(lay, champ[1], gen_now) if champ is not None and gen_now is not None else None
            if resized is not None:  # v6.1: 그때 챔피언은 다른 크기의 뇌(그 뒤에 자랐거나 다시 시작) → 보는 세대 대표
                assert champ is not None
                gen, genome = gen_now, resized
                kind, rep = "size", None
                champion_gen = champ[0]
            elif champ is not None:
                gen, genome = champ
                kind, rep = "champion", None
            else:
                still = self.dash.rep_genome(gen_now) if gen_now is not None else None
                if still is None:
                    self._empty()
                    return
                gen, genome = gen_now, still
                kind, rep = "rep", None
            fresh_gen = gen_now if gen_now is not None else gen  # 멈춘 그림의 고리는 보는 세대 기준(챔피언을 뽑은 세대가 아님)
        try:
            g64 = np.asarray(genome, dtype=np.float64)
            active, fresh, self.size_label = self._growth_marks(lay, g64, fresh_gen)
            params = lay.brain_params(g64[None, :])[0]
            self.brain.set_network(lay.brain, params, list(meta.inputs), list(meta.outputs), self.theme,
                                   active=active, fresh=fresh)
        except ValueError:  # FrameError 포함
            self._empty()
            return
        traced = rep is not None and rep.observations is not None and rep.activations is not None
        duration = 0.0
        if traced:
            assert rep is not None
            try:
                self.brain.set_trace(rep.observations, rep.activations, rep.trajectory.dt)
                duration = self.brain.steps * self.brain.dt
            except ValueError:
                self.brain.set_trace(None, None, self.brain.dt)
                traced = False
        else:
            self.brain.set_trace(None, None, self.brain.dt)
        self.brain.set_time(self.dash.play_time)
        self.source_kind, self.source_generation = kind, gen
        self.champion_generation = champion_gen
        if kind in ("champion", "rep", "size"):
            self._title_parts = (kind, str(gen), 0.0)
            notes.append(S.BRAIN_LEGEND_STILL)
        else:
            label = src.label if src is not None else ""
            self._title_parts = (kind, label if kind == "playing" else str(gen if gen is not None else ""), duration)
            if not traced:
                notes.append(S.BRAIN_PREPARING)
            notes.append(S.BRAIN_LEGEND if traced else S.BRAIN_LEGEND_STILL)
        if traced and self.brain.relu_scaled:
            notes.append(S.BRAIN_RELU_NOTE)
        if self.brain.capped:
            notes.append(S.BRAIN_EDGES_CAPPED.format(total=self.brain.edges_total, per=EDGE_CAP))
        if self.size_label:  # v6.1: 꺼진 자리(점선 원)가 있을 때만
            notes.append(S.BRAIN_GROW_NOTE)
        self._set_title_time(self.dash.play_time, force=True)
        self.stack.setCurrentWidget(self.brain)
        self.set_notes(notes)

    # -- v6.1 자라는 뇌 -------------------------------------------------------
    def frame_for(self, layout: Any) -> BrainFrame | None:
        """이 유전자 배치의 뇌 틀(은닉층이 없거나 뇌 없는 환경이면 None). 배치가 바뀔 때만 다시 만든다."""
        if self._frame is None or self._frame[0] is not layout:
            try:
                frame: BrainFrame | None = BrainFrame(layout)
            except FrameError:
                frame = None
            self._frame = (layout, frame)
            self._active_cache.clear()
            self._masks_memo = None
        return self._frame[1]

    def _masks_of(self, frame: BrainFrame, genome: np.ndarray) -> tuple[np.ndarray, ...]:
        """유전자의 켜진 뉴런(frame.active_neurons, 은닉층마다 bool). 바로 앞에 물은 유전자와 내용이 같으면(바이트 비교) 다시
        계산하지 않는다: 멈춘 그림은 통계 줄이 올 때마다 그때 챔피언의 켜짐을 두 번(_resized_rep의 크기 비교, _growth_marks의
        그림) 물었고 챔피언은 검증 때만 바뀐다(측정 M12: 펌프마다 켜짐 계산 세 번 → 보는 세대 대표 한 번). 돌려준 배열은
        고치지 않는다."""
        g = np.asarray(genome)
        key = (g.dtype.str, g.shape, g.tobytes())
        memo = self._masks_memo
        if memo is not None and memo[0] is frame and memo[1] == key:
            return memo[2]
        masks = tuple(frame.active_neurons(g))  # 모양이 틀과 다르면 FrameError(기억하지 않음)
        self._masks_memo = (frame, key, masks)
        return masks

    def active_at(self, generation: int) -> tuple[np.ndarray, ...] | None:
        """그 세대 대표의 켜진 뉴런(은닉층마다 bool). 화면 기록의 float16 대표(관찰·세대 대표 점, 파일을 읽지 않음), 없으면
        dash.rep_genome(화면 메모리 또는 세대 대표 파일). 둘 다 없으면 None. 켜짐은 0인지만 보므로 float16으로도 같다(켜진
        좌표가 float16에서 0이 되려면 |값| < 6e-8이어야 한다)."""
        meta = self._meta()
        if meta is None or generation < 1:
            return None
        frame = self.frame_for(meta.layout)
        if frame is None:
            return None
        hit = self._active_cache.get(int(generation))
        if hit is not None:
            return hit
        entry = self.history.entry(int(generation))
        genome: np.ndarray | None = None
        if entry is not None and entry.map_genomes is not None:
            genome = np.asarray(entry.map_genomes[entry.map_rep], dtype=np.float64)
        else:
            genome = self.dash.rep_genome(int(generation))
        if genome is None or genome.shape != (meta.dim,):
            return None
        out = tuple(frame.active_neurons(genome))
        if len(self._active_cache) >= ACTIVE_CACHE:
            self._active_cache.pop(next(iter(self._active_cache)))
        self._active_cache[int(generation)] = out
        return out

    def _resized_rep(self, layout: Any, champion: np.ndarray, generation: int) -> np.ndarray | None:
        """자라는 뇌 실행에서 그때 챔피언의 켜진 뉴런이 보는 세대(generation) 대표의 켜진 뉴런과 다르면 그 세대 대표 유전자
        (float64), 같거나 견줄 수 없으면(크기 고정·옛 실행, 틀 없음, 그 세대 대표를 읽을 수 없음) None.

        챔피언은 검증 때만 바뀐다(새 실행 기본: 학습 시험 1,000회마다, 더 나은 후보가 있을 때만). 그 사이에 뇌가 자라면 챔피언은
        옛 크기의 뇌라, 그것만 그리면 「뇌 크기」 타일·통계 줄과 크기가 어긋나고 세대를 넘겨 볼 때 한 단계가 통째로 빠진다. 자동
        재시작(처음 상태에서)으로 작은 뇌에서 다시 자랄 때도 같다. 켜짐 비교는 active_at(화면 기록의 float16 대표, 기억해 둠)이라
        파일을 다시 읽지 않고, 다를 때만 dash.rep_genome으로 정확한 유전자를 가져온다."""
        if not self.dash.brain_grows:
            return None
        frame = self.frame_for(layout)
        if frame is None:
            return None
        now = self.active_at(int(generation))
        if now is None:
            return None
        try:
            then = self._masks_of(frame, champion)
        except FrameError:
            return None
        if _same_masks(now, then):
            return None
        still = self.dash.rep_genome(int(generation))
        if still is None or still.shape != champion.shape:
            return None
        return still

    def _growth_marks(self, layout: Any, genome: np.ndarray, gen: int | None
                      ) -> tuple[list[np.ndarray] | None, list[np.ndarray] | None, str]:
        """(켜진 뉴런, 새로 켜진 뉴런, 제목 끝 글). 자라는 뇌 실행(dash.brain_grows)만: 크기 고정·옛 실행은 (None, None, "")라
        그림·제목이 v6.0과 같다. 새로 켜짐 = 지금 켜짐 ∧ ¬(gen − FRESH_GENERATIONS세대 대표의 켜짐), 그 대표가 없으면 None."""
        if not self.dash.brain_grows:
            return None, None, ""
        frame = self.frame_for(layout)
        if frame is None:
            return None, None, ""
        active = list(self._masks_of(frame, genome))
        fresh: list[np.ndarray] | None = None
        if gen is not None:
            before = self.active_at(int(gen) - FRESH_GENERATIONS)
            if before is not None:
                fresh = [a & ~b for a, b in zip(active, before, strict=True)]
        label = ""
        if not all(bool(a.all()) for a in active):
            total = int(sum(frame.sizes))
            # 층마다 「마지막 켜진 뉴런 번호 + 1」(하나도 없으면 1) = frame.counts_of(genome). 켜짐을 한 번 더 계산하지 않게
            # 위에서 구한 active로 센다
            counts = [int(np.flatnonzero(a)[-1]) + 1 if a.any() else 1 for a in active]
            label = S.BRAIN_SIZE_LABEL.format(n=int(sum(int(a.sum()) for a in active)), total=total,
                                              links=frame.links(counts))
        return active, fresh, label

    def _empty(self, text: str = S.BRAIN_EMPTY) -> None:
        self.source_kind = self.source_generation = None
        self.champion_generation = None
        self._title_parts = None
        self._title = ""
        self.size_label = ""
        self.brain.clear_network()
        self.show_message(text, [] if text != S.BRAIN_EMPTY else [self.loading_note()])
        self.update_legend()

    def _set_title_time(self, t: float, *, force: bool = False) -> None:
        parts = self._title_parts
        if parts is None:
            return
        kind, name, duration = parts
        shown = min(max(0.0, float(t)), duration) if duration > 0 else max(0.0, float(t))
        if kind == "champion":
            title = S.BRAIN_TITLE_CHAMPION.format(gen=name)
        elif kind == "rep":
            title = S.BRAIN_TITLE_REP.format(gen=name)
        elif kind == "size":  # v6.1: 그때 챔피언과 뇌 크기가 달라 보는 세대 대표를 그림
            title = S.BRAIN_TITLE_REP_SIZE.format(gen=name, champion=self.champion_generation)
        elif kind == "live":
            title = S.BRAIN_TITLE_LIVE.format(gen=name, t=shown)
        else:
            title = S.BRAIN_TITLE_PLAYING.format(label=name, t=shown)
        title += self.size_label  # v6.1: 꺼진 자리가 있으면 「 · 뉴런 n/전체 · 연결 m」
        if force or title != self._title:
            self._title = title
            self.update_legend()

    @property
    def title_text(self) -> str:
        """제목 줄(범례 자리)의 글: BRAIN_TITLE_*."""
        return self._title


__all__ = ["EDGE_CAP", "FRESH_GENERATIONS", "BrainTile", "BrainView", "edge_strength", "paint_edges", "select_edges",
           "short_name"]

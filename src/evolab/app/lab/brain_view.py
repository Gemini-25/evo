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
                    outputs: Sequence[tuple[str, str]], theme: Theme | None = None) -> None:
        """신경망 하나: 구조와 뇌 파라미터 (n_params,)(layout.brain_params의 한 행), 입력·출력 (이름, 풍선 글).
        구조·파라미터·이름이 같으면 아무것도 하지 않는다(연결선 층을 다시 그리지 않는다)."""
        if theme is not None:
            self.set_theme(theme)
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
        for k, centers in enumerate(self._pos):
            r = self._radius[k]
            vals = levels[k] if levels is not None else None
            for j in range(len(centers)):
                c = QPointF(float(centers[j, 0]), float(centers[j, 1]))
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
        return S.BRAIN_NODE_HIDDEN_TIP.format(k=layer, j=j + 1, value=_value_text(value), bias=bias)

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
        """지금 그리는 원천: "playing" | "live" | "champion" | "rep"(챔피언이 없어 보는 세대 대표) | None(빈 상태)."""
        self.source_generation: int | None = None
        self.show_message(S.BRAIN_EMPTY, [])

    # -- 대시보드 갈고리 --------------------------------------------------------
    def meta_changed(self) -> None:
        self.brain.clear_network()

    def history_reset(self) -> None:
        pass

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
        if src is not None and src.genome.shape == (meta.dim,):
            genome = src.genome
            kind, gen = src.kind, src.generation
            rep = src.replay
        else:
            gen_now = self.dash.view_generation()
            champ = self.dash.champion_at(gen_now if gen_now is not None else 2**62)
            if champ is not None:
                gen, genome = champ
                kind, rep = "champion", None
            else:
                still = self.dash.rep_genome(gen_now) if gen_now is not None else None
                if still is None:
                    self._empty()
                    return
                gen, genome = gen_now, still
                kind, rep = "rep", None
        lay = meta.layout
        try:
            params = lay.brain_params(np.asarray(genome, dtype=np.float64)[None, :])[0]
            self.brain.set_network(lay.brain, params, list(meta.inputs), list(meta.outputs), self.theme)
        except ValueError:
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
        if kind in ("champion", "rep"):
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
        self._set_title_time(self.dash.play_time, force=True)
        self.stack.setCurrentWidget(self.brain)
        self.set_notes(notes)

    def _empty(self, text: str = S.BRAIN_EMPTY) -> None:
        self.source_kind = self.source_generation = None
        self._title_parts = None
        self._title = ""
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
        elif kind == "live":
            title = S.BRAIN_TITLE_LIVE.format(gen=name, t=shown)
        else:
            title = S.BRAIN_TITLE_PLAYING.format(label=name, t=shown)
        if force or title != self._title:
            self._title = title
            self.update_legend()

    @property
    def title_text(self) -> str:
        """제목 줄(범례 자리)의 글: BRAIN_TITLE_*."""
        return self._title


__all__ = ["EDGE_CAP", "BrainTile", "BrainView", "edge_strength", "paint_edges", "select_edges", "short_name"]

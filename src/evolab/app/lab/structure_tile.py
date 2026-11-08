"""대시보드 「보는 재미」 쪽 「몸 구조」 타일 (v2 설계 4.13·5.4·D42, v2.1).

- 자료: 세대마다 RunHistory(history.py)의 관찰 기록 obs_body (P, NB) → 개체마다 몸 구조(있는 부품 모음). 구조는 몸 수치만의
  순수 함수라(spec/structure.py, D33) 시뮬레이션 없이 계산한다. 그 세대를 계산한 환경 버전으로 센다(실험실이
  set_env_versions로 실행의 환경 버전 목록을 준다). 관찰 기록이 없는 세대(v1.0·v1.1 세대, --no-observe)는 세대 대표
  하나로 센다(v1.2 D12 규칙).
- 그림: 가로 세대, 세로 개체 비율(%)을 쌓은 그림(STEP_LIMIT세대까지 세대마다 한 칸, 넘으면 영역). 많이 나온 구조
  6개(세대마다의 비율을 모든 세대에 더한 값 순) + 「기타」(회색). 색은 대시보드 Theme 범주 색(새 색 없음). 범례 글은
  개수("바퀴 2개"), 같은 개수가 여러 구조면 물체 이름을 붙인다("바퀴 2개: 바퀴1·바퀴4"). 보는 세대 세로선.
- 아래: 보는 세대까지의 챔피언 몸(Episode(spec, 몸, 전시 시드).geometry()의 에이전트 물체). 없어진 부품은 있음 칸을
  비운 환경으로 만든 모양을 뿌리 물체 기준으로 옮겨 점선 윤곽으로 그린다. Episode 만들기는 배경 스레드(_BodyJob)에서.
- 빈 상태: 없어질 수 있는 부품이 없는 환경(v1·v2.0 환경)은 STRUCT_EMPTY.
- 갱신은 대시보드 펌프를 따른다(보일 때만, 읽는 동안 LOADING_REFRESH_S 간격). 세대마다 센 값은 캐시해 두고 새 세대만
  센다. 세는 일은 numpy 한 번(있음 수식이 "=p.<id>"면 열 비교, 다른 수식은 행마다 작은 사전으로 계산)이다.
"""

from __future__ import annotations

import bisect
import dataclasses
import hashlib
import math
import re
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pyqtgraph as pg
import pyqtgraph.functions as fn
from PySide6.QtCore import QRectF, QSize, Qt, QThread, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPaintEvent, QPen, QPolygonF, QTransform
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from evolab.app import strings as S
from evolab.app.lab import history as hs
from evolab.app.lab.dashboard import GLYPH_FILL, Dashboard, _Entry, _pen, _qcolor, _step_edges
from evolab.app.lab.watch_tiles import _focus_line, _WatchTile
from evolab.scoring.expr import formula_body
from evolab.spec import structure as st
from evolab.spec.compiled import EnvSpec

TOP_STRUCTURES: Final = 6
"""색을 따로 주는 구조 수(나머지는 「기타」 하나)."""
BODY_CACHE: Final = 16
"""챔피언 몸 그림을 기억하는 수(타임라인을 오가도 다시 만들지 않게)."""
STEP_LIMIT: Final = 200
"""이 세대 수까지는 세대마다 한 칸(계단)으로, 넘으면 세대 점을 잇는 영역으로 그린다(점 수가 반: 그리기 시간이 준다,
가로 픽셀보다 세대가 많으면 모양은 같다). 영역일 때 세대가 그림 가로 픽셀보다 많으면 이웃 세대를 묶어 평균한 점만
그린다(_drawn: 긴 실행에서도 그리는 점 수가 그림 폭 정도)."""
PICTURE_HEIGHT: Final = 84
PICTURE_MARGIN: Final = 6.0
OTHER_KEY: Final = None
"""「기타」 계열의 열쇠."""
_P_NAME = re.compile(r"p\.([A-Za-z_][A-Za-z0-9_]*)\Z")
_TRAILING_NUMBER = re.compile(r"[\s_\-]*\d+\Z")
_MAX_BITS: Final = 62
"""있음 칸을 쓰는 물체가 이보다 많으면 비트 코드 대신 행마다 꺼진 물체 모음을 센다(드문 경우, 느림)."""

StructureKey = tuple[str, ...]


# ---------------------------------------------------------------------------
# 구조 세기 (Qt 없음)
class StructureCounter:
    """환경 버전 하나의 몸 구조 계산: 몸 수치 행렬 (R, NB) → 행마다 꺼진 물체 코드, 코드 → 있는 에이전트 물체.

    결과는 structure.present(spec, structure.param_values(env, 행))와 같다: 진화하는 매개변수 값을 [min, max]로 자르고
    (유한하지 않으면 매개변수 값), 있음 칸이 수식이면 그 값, 상수면 상수, 유한하지 않은 수식 값은 0. 0.5 이하면 꺼짐.
    꺼진 물체 모음 → 이어짐 규칙(structure.absent_from)은 코드마다 한 번만 계산한다.
    """

    def __init__(self, spec: EnvSpec) -> None:
        env = spec.file
        self.spec = spec
        self.agents: frozenset[str] = frozenset(b.id for b in env.bodies if b.owner == "agent")
        self.optional: frozenset[str] = st.optional_bodies(env)
        evolved = [p for p in env.params if p.evolve]
        self.n_body = len(evolved)
        base = st.param_values(env)
        self._lo = np.array([p.min if p.min is not None else -math.inf for p in evolved], dtype=np.float64)
        self._hi = np.array([p.max if p.max is not None else math.inf for p in evolved], dtype=np.float64)
        self._default = np.array([float(p.value) for p in evolved], dtype=np.float64)
        column = {f"p.{p.id}": k for k, p in enumerate(evolved)}
        # 있음 칸을 쓰는 에이전트 물체마다 (id, 종류, 자료): const(꺼짐 여부), column(열 번호), formula(수식, 이름 목록)
        self._parts: list[tuple[str, str, Any]] = []
        for b in env.bodies:
            if b.owner != "agent" or b.exists is None:
                continue
            if not isinstance(b.exists, str):
                self._parts.append((b.id, "const", not float(b.exists) > st.EXISTS_THRESHOLD))
                continue
            fn = spec.formulas[(b.exists, st.STRUCTURE_STAGE)]
            hit = _P_NAME.match("".join(formula_body(b.exists).split()))
            if hit is not None and f"p.{hit.group(1)}" in column:
                self._parts.append((b.id, "column", column[f"p.{hit.group(1)}"]))
            elif not any(n in column for n in fn.names):
                # 진화하지 않는 매개변수(와 상수)만 쓰는 수식: 모든 개체에서 같은 값
                self._parts.append((b.id, "const", not st.exists_value(spec, b, base) > st.EXISTS_THRESHOLD))
            else:
                names = sorted(fn.names)
                fixed = {n: base[n] for n in names if n in base and n not in column}
                cols = [(n, column[n]) for n in names if n in column]
                self._parts.append((b.id, "formula", (fn, fixed, cols)))
        self._ids = [pid for pid, _, _ in self._parts]
        self._present: dict[Any, frozenset[str]] = {}

    @property
    def uses_bits(self) -> bool:
        return len(self._parts) <= _MAX_BITS

    def _values(self, body: np.ndarray) -> np.ndarray:
        b = np.asarray(body, dtype=np.float64)
        if b.ndim != 2 or b.shape[1] != self.n_body:
            raise ValueError(f"body values must be (R, {self.n_body}), got {b.shape}")
        if not self.n_body:
            return b
        return np.where(np.isfinite(b), np.clip(b, self._lo, self._hi), self._default)

    def _off(self, values: np.ndarray) -> np.ndarray:
        """(R, 있음 칸 수) bool: 꺼진 물체."""
        rows = values.shape[0]
        off = np.zeros((rows, len(self._parts)), dtype=bool)
        for j, (_pid, kind, data) in enumerate(self._parts):
            if kind == "const":
                off[:, j] = bool(data)
            elif kind == "column":
                off[:, j] = ~(values[:, data] > st.EXISTS_THRESHOLD)
            else:
                fn, fixed, cols = data
                params = dict(fixed)
                col = off[:, j]
                for r in range(rows):
                    for name, k in cols:
                        params[name] = float(values[r, k])
                    x = float(fn(params))
                    col[r] = not (x if math.isfinite(x) else 0.0) > st.EXISTS_THRESHOLD
        return off

    def codes(self, body: np.ndarray) -> np.ndarray:
        """(R, NB) 물리 단위 몸 수치 → (R,) int64 코드(비트 j = 있음 칸 j번째 물체가 꺼짐). uses_bits가 거짓이면
        ValueError(keys를 쓸 것)."""
        if not self.uses_bits:
            raise ValueError("too many bodies with an exists field for bit codes")
        off = self._off(self._values(body))
        code = np.zeros(off.shape[0], dtype=np.int64)
        for j in range(off.shape[1]):
            code |= off[:, j].astype(np.int64) << j
        return code

    def off_sets(self, body: np.ndarray) -> list[frozenset[str]]:
        """행마다 꺼진 물체 모음(코드 대신 쓰는 느린 길)."""
        off = self._off(self._values(body))
        return [frozenset(self._ids[j] for j in np.flatnonzero(row)) for row in off]

    def present_of(self, code: Any) -> frozenset[str]:
        """코드(int) 또는 꺼진 물체 모음(frozenset) → 있는 에이전트 물체."""
        hit = self._present.get(code)
        if hit is None:
            if isinstance(code, frozenset):
                off = code
            else:
                off = frozenset(self._ids[j] for j in range(len(self._ids)) if int(code) >> j & 1)
            hit = self.agents - st.absent_from(self.spec.file, off) if off else self.agents
            self._present[code] = hit
        return hit

    def count(self, blocks: Sequence[np.ndarray]) -> list[tuple[list[Any], np.ndarray]]:
        """세대마다의 몸 수치 (r_i, NB) → 세대마다 (코드 목록, 개체 수). 한 번의 numpy 계산으로 센다."""
        if not blocks:
            return []
        sizes = np.array([len(b) for b in blocks], dtype=np.int64)
        body = np.vstack([np.asarray(b, dtype=np.float64).reshape(len(b), self.n_body) for b in blocks])
        if not self.uses_bits:
            sets = self.off_sets(body)
            out: list[tuple[list[Any], np.ndarray]] = []
            start = 0
            for n in sizes.tolist():
                tally: dict[frozenset[str], int] = {}
                for s in sets[start:start + n]:
                    tally[s] = tally.get(s, 0) + 1
                start += n
                out.append((list(tally), np.array(list(tally.values()), dtype=np.int64)))
            return out
        code = self.codes(body)
        pos = np.repeat(np.arange(len(blocks), dtype=np.int64), sizes)
        order = np.lexsort((code, pos))
        sp, sc = pos[order], code[order]
        total = len(sc)
        if total == 0:
            return [([], np.zeros(0, dtype=np.int64)) for _ in blocks]
        cut = np.flatnonzero((sp[1:] != sp[:-1]) | (sc[1:] != sc[:-1])) + 1
        starts = np.concatenate(([0], cut))
        counts = np.diff(np.concatenate((starts, [total])))
        upos, ucode = sp[starts], sc[starts]
        edges = np.searchsorted(upos, np.arange(len(blocks) + 1))
        return [(ucode[edges[i]:edges[i + 1]].tolist(), counts[edges[i]:edges[i + 1]])
                for i in range(len(blocks))]

    def keys(self, body: np.ndarray, order: Sequence[str]) -> list[StructureKey]:
        """행마다 구조 열쇠: order(없어질 수 있는 물체 순서) 가운데 있는 것(테스트·검사용)."""
        rows = np.asarray(body, dtype=np.float64).reshape(-1, self.n_body)
        codes: list[Any] = self.codes(rows).tolist() if self.uses_bits else list(self.off_sets(rows))
        return [tuple(b for b in order if b in self.present_of(c)) for c in codes]


def base_name(label: str) -> str:
    """범례에 쓰는 부품 이름: 이름 끝의 번호를 뗀 것("바퀴1" → "바퀴", "wheel_2" → "wheel"). 다 떼면 그대로."""
    name = _TRAILING_NUMBER.sub("", label).strip()
    return name or label


def structure_label(key: StructureKey, labels: dict[str, str], groups: Sequence[str], which: bool) -> str:
    """구조 열쇠의 범례 글. groups = 부품 이름(base_name) 순서. 이름이 하나면 STRUCT_COUNT("바퀴 2개"), 여럿이면
    이름마다 붙여 쓴다. which면 있는 물체 이름을 붙인다(STRUCT_COUNT_WHICH). 아무것도 없으면 STRUCT_NONE."""
    if not key:
        return S.STRUCT_NONE
    names = [labels.get(b, b) for b in key]
    tally = {g: 0 for g in groups}
    for n in names:
        g = base_name(n)
        tally[g] = tally.get(g, 0) + 1
    shown = [(g, n) for g, n in tally.items() if n > 0 or len(tally) == 1]
    if len(shown) == 1:
        g, n = shown[0]
        if which:
            return S.STRUCT_COUNT_WHICH.format(name=g, n=n, which="·".join(names))
        return S.STRUCT_COUNT.format(name=g, n=n)
    text = " · ".join(S.STRUCT_COUNT.format(name=g, n=n) for g, n in shown)
    return f"{text}: {'·'.join(names)}" if which else text


# ---------------------------------------------------------------------------
# 챔피언 몸 그림 (Qt 위젯 없음: 배경 스레드에서 부른다)
def full_structure_spec(spec: EnvSpec) -> EnvSpec | None:
    """있음 칸을 모두 비운 환경(모든 부품이 있는 몸): 없어진 부품의 윤곽을 그릴 때 쓴다. 있음 칸이 없거나 만들 수
    없으면 None."""
    from evolab.spec.compiled import EnvFileError
    from evolab.spec.io import compile_env

    env = spec.file
    if all(b.exists is None for b in env.bodies):
        return None
    bodies = [b.model_copy(update={"exists": None}) if b.exists is not None else b for b in env.bodies]
    try:
        return compile_env(env.model_copy(update={"bodies": bodies}))
    except (EnvFileError, ValueError, KeyError):
        return None


def _relative_pose(pose: tuple[float, float, float], root: tuple[float, float, float],
                   target: tuple[float, float, float]) -> tuple[float, float, float]:
    """root 기준 pose를 target 기준으로 옮긴다(pose ∘ root⁻¹ 뒤 target ∘)."""
    dx, dy = pose[0] - root[0], pose[1] - root[1]
    c, s = math.cos(root[2]), math.sin(root[2])
    rx, ry = c * dx + s * dy, -s * dx + c * dy
    ct, sn = math.cos(target[2]), math.sin(target[2])
    return (target[0] + ct * rx - sn * ry, target[1] + sn * rx + ct * ry, target[2] + pose[2] - root[2])


def champion_bodies(spec: EnvSpec, body: np.ndarray | None, full: EnvSpec | None,
                    pause: Callable[[], object] | None = None) -> list[tuple[Any, str]]:
    """챔피언 몸 그림: [(BodyGeom, 역할)]. 역할 "body"(늘 있는 에이전트 물체), "part"(있는 부품), "absent"(없는 부품,
    full로 만든 모양을 뿌리 물체 기준으로 옮긴 것). Episode(spec, body, 전시 시드)를 만든다(EpisodeBuildError는 그대로).
    없는 부품이 있으면 Episode를 하나 더(full로) 만드는데, 그 앞에 pause()를 부른다(배경 스레드가 화면이 바쁜 동안 쉬게)."""
    from evolab.engine import seeds
    from evolab.world.physics2d.episode import Episode, EpisodeBuildError

    trial = seeds.display_trial()
    ep = Episode(spec, body, trial)
    optional = st.optional_bodies(spec.file)
    geo = ep.geometry()
    agent = [b for b in geo.bodies if b.owner == "agent"]
    out: list[tuple[Any, str]] = [(b, "part" if b.id in optional else "body") for b in agent]
    missing = [b.id for b in spec.file.bodies if b.id in ep.absent]
    root = st.root_body(spec.file)
    if not missing or full is None or root is None:
        return out
    if pause is not None:
        pause()
    try:
        whole = Episode(full, body, trial).geometry()
    except (EpisodeBuildError, ValueError, KeyError):
        return out
    by_id = {b.id: b for b in whole.bodies}
    here = {b.id: b for b in agent}
    if root not in by_id or root not in here:
        return out
    r_full, r_here = by_id[root].pose, here[root].pose
    for bid in missing:
        g = by_id.get(bid)
        if g is not None:
            out.append((dataclasses.replace(g, pose=_relative_pose(g.pose, r_full, r_here)), "absent"))
    return out


_LIVE_JOBS: set[QThread] = set()
"""도는 중인 몸 그림 스레드를 끝날 때까지 붙잡아 둔다(타일이 먼저 사라져도 QThread가 돌던 중에 지워지지 않게)."""


class _BodyJob(QThread):
    """챔피언 몸 그림 하나를 배경에서: 있음 칸을 비운 환경 컴파일(처음 한 번) + Episode 하나나 둘."""

    done = Signal(object)

    def __init__(self, tag: tuple[Any, ...], spec: EnvSpec, body: np.ndarray | None, full: EnvSpec | None,
                 need_full: bool) -> None:
        super().__init__()
        self.tag = tag
        self.spec = spec
        self.body = body
        self.full = full
        self.need_full = need_full

    def run(self) -> None:
        # 펌프가 그리는 중이면 끝난 뒤에(GIL을 다투지 않게). 일이 셋(있음 칸을 비운 환경 컴파일, Episode 둘)이라 일마다
        # 앞에서 다시 기다린다(v1.2 gui_first와 같은 규칙: 한 일을 시작한 뒤에는 멈추지 않는다)
        def pause() -> None:
            hs.GUI_PRIORITY.wait(self.isInterruptionRequested)

        pause()
        full = self.full
        try:
            if full is None and self.need_full:
                full = full_structure_spec(self.spec)
                pause()
            result: Any = champion_bodies(self.spec, self.body, full, pause)
        except Exception as exc:  # noqa: BLE001 - 그림만 못 그린다(학습·재생과 무관)
            result = exc
        self.done.emit((self, self.tag, result, full))


def _forget_job(job: QThread) -> None:
    job.wait(2000)
    _LIVE_JOBS.discard(job)


def wait_jobs() -> None:
    """프로그램을 끝낼 때: 도는 몸 그림 스레드를 끝날 때까지 기다린다(주인 없는 QThread가 돌던 중에 지워지면 abort)."""
    for job in list(_LIVE_JOBS):
        job.wait()
        _LIVE_JOBS.discard(job)


def jobs_running() -> bool:
    return any(job.isRunning() for job in _LIVE_JOBS)


# ---------------------------------------------------------------------------
# 위젯
class _BodyPicture(QWidget):
    """챔피언 몸: 물체 모양을 칸에 맞춰(y 위쪽) 그린다. 없는 부품은 점선 윤곽."""

    def __init__(self, tile: StructureTile) -> None:
        super().__init__()
        self.tile = tile
        self.setFixedHeight(PICTURE_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.items: list[tuple[QPainterPath, QPainterPath, str]] = []
        """(채울 모양, 선분, 역할) 세계 좌표."""
        self.part_color = tile.theme.series[0]
        self._bounds = QRectF()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(200, PICTURE_HEIGHT)

    def set_bodies(self, bodies: Sequence[tuple[Any, str]], part_color: str) -> None:
        from evolab.app.play.window import body_path

        items: list[tuple[QPainterPath, QPainterPath, str]] = []
        bounds = QRectF()
        order = {"body": 0, "part": 1, "absent": 2}
        for body, role in sorted(bodies, key=lambda br: order.get(br[1], 3)):
            fill, lines = body_path(body)
            x, y, a = body.pose
            tr = QTransform().translate(float(x), float(y)).rotateRadians(float(a))
            wf, wl = tr.map(fill), tr.map(lines)
            items.append((wf, wl, role))
            for p in (wf, wl):
                if not p.isEmpty():
                    bounds = bounds.united(p.boundingRect()) if not bounds.isNull() else p.boundingRect()
        self.items = items
        self.part_color = part_color
        self._bounds = bounds
        self.update()

    def clear(self) -> None:
        self.items = []
        self._bounds = QRectF()
        self.update()

    def view_transform(self) -> QTransform | None:
        """세계 → 화면(y 위쪽, 가운데 맞춤). 그릴 것이 없으면 None."""
        b = self._bounds
        if not self.items or b.isNull() or b.width() <= 0 and b.height() <= 0:
            return None
        w = max(1.0, self.width() - 2 * PICTURE_MARGIN)
        h = max(1.0, self.height() - 2 * PICTURE_MARGIN)
        s = min(w / max(b.width(), 1e-6), h / max(b.height(), 1e-6))
        cx, cy = b.center().x(), b.center().y()
        return QTransform().translate(self.width() / 2.0, self.height() / 2.0).scale(s, -s).translate(-cx, -cy)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        tr = self.view_transform()
        if tr is None:
            return
        t = self.tile.theme
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setTransform(tr)
        edge = QPen(QColor(t.ink2), 1.2)
        edge.setCosmetic(True)
        dashed = QPen(QColor(t.muted), 1.2, Qt.PenStyle.DashLine)
        dashed.setCosmetic(True)
        for fill, lines, role in self.items:
            if role == "absent":
                p.setPen(dashed)
                p.setBrush(Qt.BrushStyle.NoBrush)
            else:
                p.setPen(edge)
                color = self.part_color if role == "part" else t.ink2
                p.setBrush(QBrush(_qcolor(color, 170 if role == "part" else 70)))
            if not fill.isEmpty():
                p.drawPath(fill)
            if not lines.isEmpty():
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(lines)
        p.end()


def _polygon(xs: np.ndarray, ys: np.ndarray, base: float | None = None) -> QPolygonF:
    """점 (xs, ys)의 QPolygonF(numpy로 한 번에 채움). base가 있으면 끝에서 base 높이로 내려와 처음 x로 돌아가는 두 점을
    더한다(아래까지 채우는 다각형)."""
    n = len(xs)
    poly = fn.create_qpolygonf(n + (2 if base is not None and n else 0))
    arr = fn.ndarray_from_qpolygonf(poly)
    arr[:n, 0] = xs
    arr[:n, 1] = ys
    if base is not None and n:
        arr[n] = (xs[-1], base)
        arr[n + 1] = (xs[0], base)
    return poly


class _StackArea(pg.GraphicsObject):
    """쌓은 비율 그림 한 항목: 계열마다 아래(0)까지 채운 다각형을 위 계열부터 그리고 그 윗선을 바탕색 1픽셀 선으로 긋는다
    (작은 누적이 위에 오는 불투명 색 쌓기, 계열마다 PlotCurveItem(fillLevel=0, 바탕색 펜)을 두던 것과 같은 그림).

    계열마다 PlotCurveItem을 두면 그릴 때마다 계열마다 자료 범위(nanmin·nanmax)·채움 경로를 다시 만들어, 750세대 기록에서
    세대 하나가 올 때 그리기가 5~10ms였다(9.2 M13). 여기서는 자료가 바뀔 때 다각형을 한 번 만들고, 그리기는 채우기와 선만
    한다. 자동 범위는 dataBounds(가로: 처음~끝 x, 세로: 0~맨 위 누적)로 바로 알린다.
    """

    def __init__(self, edge: QPen) -> None:
        super().__init__()
        self._edge = edge
        self._layers: list[tuple[np.ndarray, np.ndarray]] = []
        self._fills: list[tuple[QPolygonF, QBrush]] = []
        self._lines: list[QPolygonF] = []
        self._xr: tuple[float, float] | None = None
        self._yr: tuple[float, float] | None = None
        self._rect = QRectF()

    def set_layers(self, x: np.ndarray, cum: np.ndarray, colors: Sequence[str], step: bool) -> None:
        """x: 계단이면 칸 경계(len = 점 수 + 1, 세대마다 한 칸), 아니면 점의 x. cum: (계열, 점) 누적 %(아래 계열부터),
        colors: 계열 색. 계단은 칸마다 수평선(경계 x에서 세로로 잇는다)."""
        self.prepareGeometryChange()
        x = np.asarray(x, dtype=np.float64)
        cum = np.asarray(cum, dtype=np.float64)
        ok = cum.ndim == 2 and cum.shape[0] == len(colors)
        self._layers = [(x, cum[j]) for j in range(len(colors))] if ok else []
        n = cum.shape[1] if ok else 0
        if not len(colors) or n == 0 or len(x) != n + (1 if step else 0):
            self._fills, self._lines = [], []
            self._xr = self._yr = None
            self._rect = QRectF()
            self.update()
            return
        xs = np.repeat(x, 2)[1:-1] if step else x
        fills: list[tuple[QPolygonF, QBrush]] = []
        lines: list[QPolygonF] = []
        for j in reversed(range(len(colors))):  # 위 계열(큰 누적)부터: 아래 계열이 그 위에 그려진다
            ys = np.repeat(cum[j], 2) if step else cum[j]
            fills.append((_polygon(xs, ys, 0.0), QBrush(_qcolor(colors[j]))))
            if step:  # 계단 선은 양 끝에서 0까지 내려온다(PlotCurveItem stepMode="center" + fillLevel과 같은 선)
                lines.append(_polygon(np.concatenate(([xs[0]], xs, [xs[-1]])), np.concatenate(([0.0], ys, [0.0]))))
            else:
                lines.append(_polygon(xs, ys))
        self._fills, self._lines = fills, lines
        x0, x1 = float(xs[0]), float(xs[-1])
        y0, y1 = min(0.0, float(cum.min())), max(0.0, float(cum.max()))
        self._xr, self._yr = (x0, x1), (y0, y1)
        # 바뀐 곳 알리기(다시 그릴 칸)에만 쓰는 사각형: 1픽셀 선이 가장자리에 걸리지 않게 조금 넓힌다
        px = max(0.5, 0.01 * (x1 - x0))
        py = max(1.0, 0.02 * (y1 - y0))
        self._rect = QRectF(x0 - px, y0 - py, (x1 - x0) + 2 * px, (y1 - y0) + 2 * py)
        self.update()

    @property
    def layer_count(self) -> int:
        return len(self._layers)

    def layer_data(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """j번째 계열에 준 (x, 누적 %) 그대로(계단이면 len(x) = len(y) + 1, 시험용)."""
        return self._layers[j]

    def dataBounds(self, ax: int, frac: float = 1.0, orthoRange: Any = None) -> tuple[float | None, float | None]:  # noqa: N802, N803
        r = self._xr if ax == 0 else self._yr
        return (None, None) if r is None else r

    def pixelPadding(self) -> float:  # noqa: N802
        return self._edge.widthF() * 0.7072  # PlotCurveItem과 같은 여백(코스메틱 펜)

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self._rect

    def paint(self, p: QPainter, *args: Any) -> None:
        if not self._fills:
            return
        for (fill, brush), line in zip(self._fills, self._lines, strict=True):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(brush)
            p.drawPolygon(fill)
            p.setPen(self._edge)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(line)


@dataclass(frozen=True, eq=False)
class _GenCount:
    """세대 하나를 센 값(entry가 바뀌면 다시 센다)."""

    generation: int
    entry: hs.GenEntry
    version: int
    total: int
    kids: np.ndarray
    """(k,) intp: 이 세대에 나온 구조 열쇠 번호(StructureTile._keys), 서로 다름."""
    fracs: np.ndarray
    """(k,) float64: 그 구조의 개체 비율(합 1)."""


class StructureTile(_WatchTile):
    """몸 구조(4.13): 세대마다 구조별 개체 비율을 쌓은 그림과 보는 세대 챔피언의 몸."""

    follows_axis = False

    def __init__(self, dash: Dashboard, tile_id: str) -> None:
        super().__init__(dash, tile_id)
        t = self.theme
        self.plot = self.add_plot(S.STRUCT_AXIS)
        # 불투명하게 쌓은 0~100% 그림이라 가로 격자는 보이지 않는다. 격자를 그리는 축 그림이 타일 그리기의 절반을
        # 넘어서(750세대에서 축 하나 약 5ms, 끄면 약 0.7ms) 끈다
        self.plot.showGrid(x=False, y=False)
        self.fix_y(self.plot, 0.0, 100.0)
        self.focus_line = _focus_line(t)
        self.plot.addItem(self.focus_line, ignoreBounds=True)
        # 그래프 아래에 챔피언 몸: 쌓인 칸(stack)의 그래프 자리를 [그래프, 제목, 몸 그림] 묶음으로 바꾼다
        self.stack.removeWidget(self.view)
        self.host = QWidget()
        hl = QVBoxLayout(self.host)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(1)
        hl.addWidget(self.view, 1)
        self.view.setVisible(True)  # 쌓인 칸에서 뺄 때 숨겨졌다(QStackedLayout.takeAt)
        self.caption = QLabel()
        self.caption.setStyleSheet(f"color: {t.ink2};")
        self.caption.setFont(self.legend.font())
        hl.addWidget(self.caption)
        self.picture = _BodyPicture(self)
        hl.addWidget(self.picture)
        self.stack.addWidget(self.host)
        self.message.setText(S.STRUCT_EMPTY)
        self.stack.setCurrentWidget(self.message)
        # 쌓은 그림은 한 항목(계열마다 곡선 항목을 두면 세대마다 그리기가 느리다, _StackArea)
        self.area = _StackArea(_pen(t.surface, 1))
        self.plot.addItem(self.area)

        self._versions: list[tuple[int, EnvSpec | str]] = []
        self._froms: list[int] = []
        self._counters: dict[int, StructureCounter | None] = {}
        self._full: dict[int, EnvSpec | None] = {}
        self._counts: dict[int, _GenCount] = {}
        self._keys: list[StructureKey] = []
        """나온 구조 열쇠(번호 = 위치). 세기를 비울 때만 비운다."""
        self._key_index: dict[StructureKey, int] = {}
        self._key_of: dict[tuple[int, Any], int] = {}
        """(버전, 코드) → 구조 열쇠 번호."""
        self._keys_order: tuple[str, ...] = ()
        self._epoch = 0
        """버전·기록이 바뀔 때마다 늘어난다(배경 몸 그림 결과가 지금 것인지 보는 꼬리표)."""
        self._agg_key: tuple[Any, ...] | None = None
        self._slots: dict[StructureKey, int] = {}
        """보이는 구조 열쇠 → 색 자리(Theme.series 번호). 열쇠가 많이 나온 6개에 들어올 때 비어 있는 가장 작은 자리를
        받고, 6개에 있는 동안 지킨다(다른 구조가 들고 나도 색이 바뀌지 않게). 6개에서 빠지면 자리를 내놓는다."""
        # 그린 자료(테스트가 읽는다)
        self.gens = np.zeros(0)
        """그린 세대(오름차순, float)."""
        self.series: list[StructureKey | None] = []
        """계열 열쇠(쌓는 순서, 아래부터). None은 「기타」."""
        self.names: list[str] = []
        """계열 범례 글."""
        self.colors: list[str] = []
        self.ratios = np.zeros((0, 0))
        """(계열, 세대) 개체 비율(0..1)."""
        self.observed = np.zeros(0, dtype=bool)
        """(세대,) 집단 기록으로 센 세대(아니면 세대 대표 하나)."""
        self.union_order: list[str] = []
        """없어질 수 있는 물체(모든 환경 버전, 범례 순서)."""
        self.labels: dict[str, str] = {}
        self.key_names: dict[StructureKey, str] = {}
        """보이는 구조 열쇠 → 범례 글(기타에 묶인 구조는 이름이 같은 구조와 겹쳐도 물체 이름 없이)."""
        # 챔피언 몸
        self.champion: tuple[int, StructureKey | None] | None = None
        """그린 챔피언 (뽑힌 세대, 구조 열쇠)."""
        self._bodies: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._job: _BodyJob | None = None
        self._pending: tuple[tuple[Any, ...], int, EnvSpec, np.ndarray | None] | None = None
        self._want: tuple[Any, ...] | None = None

    # -- 환경 버전 ------------------------------------------------------------
    def set_env_versions(self, versions: Sequence[tuple[int, EnvSpec | str]]) -> None:
        """실행의 환경 버전: (적용 시작 세대 from_generation, 환경 EnvSpec 또는 환경 파일 글), run.json의 env_versions
        순서. 세대 g는 from_generation ≤ g − 1인 마지막 버전으로 계산했다(없으면 첫 버전, 실험실 spec_for와 같은 규칙).
        글은 처음 쓸 때 컴파일한다."""
        new = [(int(g), v) for g, v in versions]
        if len(new) == len(self._versions) and all(
                a[0] == b[0] and (a[1] is b[1] or (isinstance(a[1], str) and a[1] == b[1]))
                for a, b in zip(new, self._versions, strict=True)):
            return
        order = sorted(range(len(new)), key=lambda i: new[i][0])  # 안정 정렬: 같은 세대면 나중 기록이 뒤
        self._versions = [new[i] for i in order]
        self._froms = [g for g, _ in self._versions]
        self._counters.clear()
        self._full.clear()
        self._reset_counts()
        self.dash._invalidate((self.tile_id,))

    def add_env_version(self, from_generation: int, env: EnvSpec | str) -> None:
        """학습 중 환경을 바꿨다: 그 뒤 세대(from_generation + 1부터)는 이 환경으로 계산한다."""
        self.set_env_versions([*self._versions, (int(from_generation), env)])

    def version_of(self, generation: int) -> int | None:
        """그 세대를 계산한 환경 버전 번호(없으면 None)."""
        if not self._versions:
            return None
        return max(0, bisect.bisect_right(self._froms, int(generation) - 1) - 1)

    def counter(self, version: int) -> StructureCounter | None:
        if version not in self._counters:
            from evolab.spec.compiled import EnvFileError
            from evolab.spec.io import compile_env

            env = self._versions[version][1]
            same = next((v for v, (_, e) in enumerate(self._versions)
                         if v in self._counters and (e is env or (isinstance(env, str) and e == env))), None)
            if same is not None:  # 예전 버전으로 되돌린 기록: 같은 글은 한 번만 컴파일
                self._counters[version] = self._counters[same]
                return self._counters[version]
            try:
                spec = env if isinstance(env, EnvSpec) else compile_env(env)
                self._counters[version] = StructureCounter(spec)
            except (EnvFileError, ValueError, KeyError):
                self._counters[version] = None
        return self._counters[version]

    def _reset_counts(self) -> None:
        self._counts.clear()
        self._keys.clear()
        self._key_index.clear()
        self._key_of.clear()
        self._agg_key = None
        self._slots.clear()
        self._epoch += 1
        self._pending = None
        self._want = None
        self._bodies.clear()

    # -- 대시보드 걸이(dashboard.py가 「보는 재미」 타일마다 부른다) --------------------------
    def history_reset(self) -> None:
        self._reset_counts()

    def meta_changed(self) -> None:
        self._reset_counts()

    # -- 세기 -----------------------------------------------------------------
    def _union(self) -> tuple[list[str], dict[str, str]]:
        """모든 버전의 없어질 수 있는 물체(마지막 버전의 선언 순서 먼저)와 물체 이름(label, 없으면 id)."""
        order: list[str] = []
        labels: dict[str, str] = {}
        for v in reversed(range(len(self._versions))):
            c = self.counter(v)
            if c is None:
                continue
            for b in c.spec.file.bodies:
                if b.id in c.optional and b.id not in labels:
                    order.append(b.id)
                    labels[b.id] = b.label or b.id
        return order, labels

    def _rows(self, entry: hs.GenEntry, n_body: int) -> tuple[np.ndarray, bool] | None:
        """세대 하나의 몸 수치 행렬과 집단 기록 여부. 관찰 기록이 없으면 세대 대표 한 줄(D12)."""
        obs = entry.obs
        if obs is not None:
            if obs.body is not None:
                return np.asarray(obs.body, dtype=np.float64), True
            return (np.zeros((obs.population, 0)), True) if n_body == 0 else None
        if entry.body_rep is not None:
            return np.asarray(entry.body_rep, dtype=np.float64).reshape(1, -1), False
        if n_body == 0 and entry.rep_genome16 is not None:
            return np.zeros((1, 0)), False
        return None

    def _kid(self, version: int, counter: StructureCounter, code: Any) -> int:
        """(버전, 코드) → 구조 열쇠 번호(처음 나온 열쇠면 더한다)."""
        kid = self._key_of.get((version, code))
        if kid is None:
            here = counter.present_of(code)
            key = tuple(b for b in self._keys_order if b in here)
            kid = self._key_index.get(key)
            if kid is None:
                kid = self._key_index[key] = len(self._keys)
                self._keys.append(key)
            self._key_of[(version, code)] = kid
        return kid

    def _count_missing(self, gens: Sequence[int]) -> None:
        """아직 세지 않았거나 기록이 바뀐 세대를 센다(버전마다 numpy 한 번). 버전이 바뀌면 세기 전체를 비우므로
        (set_env_versions) 이미 센 세대는 entry가 같은지만 본다."""
        h = self.history
        counts = self._counts
        todo: dict[int, list[tuple[int, hs.GenEntry, np.ndarray]]] = {}
        for g in gens:
            entry = h.entry(g)
            have = counts.get(g)
            if entry is None or (have is not None and have.entry is entry):
                continue
            counts.pop(g, None)
            v = self.version_of(g)
            c = self.counter(v) if v is not None else None
            if c is None:
                continue
            rows = self._rows(entry, c.n_body)
            if rows is None or rows[0].shape[1] != c.n_body or not len(rows[0]):
                continue
            todo.setdefault(v, []).append((g, entry, rows[0]))  # type: ignore[arg-type]
        for v, items in todo.items():
            c = self.counter(v)
            assert c is not None
            counted = c.count([rows for _, _, rows in items])
            for (g, entry, rows), (codes, n) in zip(items, counted, strict=True):
                total = int(len(rows))
                merged: dict[int, int] = {}
                for code, k in zip(codes, n.tolist(), strict=True):  # 다른 코드가 같은 구조일 수 있다(이어짐 규칙)
                    kid = self._kid(v, c, code)
                    merged[kid] = merged.get(kid, 0) + int(k)
                counts[g] = _GenCount(g, entry, v, total, np.fromiter(merged, dtype=np.intp, count=len(merged)),
                                      np.array(list(merged.values()), dtype=np.float64) / total)

    def _aggregate(self, gens: list[int]) -> tuple[list[int], list[StructureKey], np.ndarray, np.ndarray]:
        """센 세대 → (세대, 구조 열쇠 목록, (열쇠, 세대) 비율, (세대,) 집단 기록 여부)."""
        counts = self._counts
        gcs = [gc for g in gens if (gc := counts.get(g)) is not None]
        keys = list(self._keys)
        ratios = np.zeros((len(keys), len(gcs)))
        if gcs:
            lens = np.fromiter((len(gc.kids) for gc in gcs), dtype=np.intp, count=len(gcs))
            cols = np.repeat(np.arange(len(gcs), dtype=np.intp), lens)
            ratios[np.concatenate([gc.kids for gc in gcs]), cols] = np.concatenate([gc.fracs for gc in gcs])
        observed = np.fromiter((gc.entry.obs is not None for gc in gcs), dtype=bool, count=len(gcs))
        return [gc.generation for gc in gcs], keys, ratios, observed

    def _choose(self, keys: list[StructureKey], ratios: np.ndarray) -> None:
        """많이 나온 6개 + 기타 → self.series·names·colors·ratios."""
        t = self.theme
        pos = {b: i for i, b in enumerate(self.union_order)}

        def sort_key(k: StructureKey) -> tuple[int, list[int]]:
            return (len(k), [pos[b] for b in k])

        share = ratios.sum(axis=1)
        live = [i for i in range(len(keys)) if share[i] > 0]  # 바뀐 세대에서만 나왔던 열쇠는 뺀다
        ranked = sorted(live, key=lambda i: (-share[i], sort_key(keys[i])))
        top = ranked if len(ranked) <= TOP_STRUCTURES else ranked[:TOP_STRUCTURES]
        rest = [] if len(ranked) <= TOP_STRUCTURES else ranked[TOP_STRUCTURES:]
        shown = sorted(top, key=lambda i: sort_key(keys[i]))
        groups: list[str] = []
        for b in self.union_order:
            g = base_name(self.labels.get(b, b))
            if g not in groups:
                groups.append(g)
        plain = {i: structure_label(keys[i], self.labels, groups, False) for i in live}
        seen: dict[str, int] = {}
        for text in plain.values():
            seen[text] = seen.get(text, 0) + 1
        # 같은 개수인 구조가 여럿이면(기타에 묶인 것 포함) 물체 이름을 붙인다
        self.key_names = {keys[i]: structure_label(keys[i], self.labels, groups, seen[plain[i]] > 1) for i in live}
        self.series = [keys[i] for i in shown]
        self.names = [self.key_names[keys[i]] for i in shown]
        # 색은 순서가 아니라 열쇠마다 지키는 자리로(범례는 부품 수 순서 그대로)
        for k in [k for k in self._slots if k not in self.series]:
            del self._slots[k]
        for k in self.series:
            if k not in self._slots:
                taken = set(self._slots.values())
                self._slots[k] = next(j for j in range(TOP_STRUCTURES + 1) if j not in taken)
        self.colors = [t.series[self._slots[k] % len(t.series)] for k in self.series]
        rows = [ratios[i] for i in shown]
        if rest:
            self.series.append(OTHER_KEY)
            self.names.append(S.STRUCT_OTHER)
            self.colors.append(t.muted)
            rows.append(ratios[rest].sum(axis=0))
        self.ratios = np.vstack(rows) if rows else np.zeros((0, ratios.shape[1]))

    # -- 갱신 -----------------------------------------------------------------
    def update_data(self) -> None:
        if not self._versions:
            self._show_empty(S.STRUCT_EMPTY)
            return
        order, labels = self._union()
        if not order:
            self._show_empty(S.STRUCT_EMPTY)
            return
        h = self.history
        if not h.gens:
            self._show_empty(S.STRUCT_WAIT, [self.loading_note()])
            return
        agg_key = (h.epoch, h.version, len(self._versions), self._epoch)
        if agg_key != self._agg_key:
            if tuple(order) != self._keys_order:  # 없어질 수 있는 물체가 바뀌면 열쇠를 처음부터
                self._reset_counts()
                self._keys_order = tuple(order)
                agg_key = (h.epoch, h.version, len(self._versions), self._epoch)
            self.union_order, self.labels = order, labels
            self._count_missing(h.gens)
            used, keys, ratios, observed = self._aggregate(h.gens)
            if not used:
                self._show_empty(S.STRUCT_WAIT, [self.loading_note()])
                return
            self._choose(keys, ratios)
            self.gens = np.array(used, dtype=np.float64)
            self.observed = observed
            if len(self.gens) <= STEP_LIMIT:
                self.area.set_layers(_step_edges(self.gens), np.cumsum(self.ratios * 100.0, axis=0), self.colors,
                                     step=True)
            else:
                x, cum = self._drawn()
                self.area.set_layers(x, cum, self.colors, step=False)
            self._agg_key = agg_key
        self.stack.setCurrentWidget(self.host)
        self._place_focus()
        self._update_champion()
        self.set_notes([S.STRUCT_REPS_ONLY if len(self.observed) and not bool(self.observed.all()) else "",
                        self.loading_note()])

    def _drawn(self) -> tuple[np.ndarray, np.ndarray]:
        """영역 그림(STEP_LIMIT 넘음)에 넘길 (세대, (계열, 점) 누적 %). 세대가 그림 가로 픽셀보다 많으면 이웃 세대
        k개씩 묶어 평균한다(묶음의 첫 세대 자리, 마지막 세대는 그대로 끝점). 평균의 누적 = 누적의 평균이라 쌓은 모양이
        유지되고, 묶음 하나가 1픽셀 남짓이라 보이는 모양은 같다. self.gens·ratios(테스트·마우스가 읽음)는 그대로 둔다."""
        cum = np.cumsum(self.ratios * 100.0, axis=0)
        n = len(self.gens)
        width = max(STEP_LIMIT, int(self.plot.vb.width()))
        if n <= width:
            return self.gens, cum
        k = math.ceil(n / width)
        starts = np.arange(0, n, k)
        sizes = np.diff(np.append(starts, n)).astype(np.float64)
        mean = np.add.reduceat(cum, starts, axis=1) / sizes
        x = np.append(self.gens[starts], self.gens[-1])
        return x, np.concatenate([mean, cum[:, -1:]], axis=1)

    def _show_empty(self, text: str, notes: list[str] | None = None) -> None:
        self.gens = np.zeros(0)
        self.series, self.names, self.colors = [], [], []
        self.ratios = np.zeros((0, 0))
        self.observed = np.zeros(0, dtype=bool)
        self._agg_key = None
        self.champion = None
        self.picture.clear()
        self.caption.setText("")
        self.show_message(text, notes if notes is not None else [])

    def _place_focus(self) -> None:
        gen = self.dash.view_generation()
        if gen is None:
            self.focus_line.hide()
            return
        self.focus_line.setPos(float(gen))
        self.focus_line.show()

    # -- 챔피언 몸 -----------------------------------------------------------
    def _update_champion(self) -> None:
        meta = self.dash.watch_meta
        gen = self.dash.view_generation()
        champ = self.dash.champion_at(gen) if gen is not None else None
        if champ is None or meta is None:
            self._want = None
            self.champion = None
            self.picture.clear()
            self.caption.setText("")
            return
        cg, genome = champ
        v = self.version_of(cg)
        c = self.counter(v) if v is not None else None
        if c is None:
            return
        try:
            body = meta.layout.body_values(genome) if meta.layout.n_body else None
        except ValueError:
            return
        digest = hashlib.blake2b(np.ascontiguousarray(genome, dtype=np.float64).tobytes(), digest_size=12).digest()
        key = (v, int(cg), digest)
        self._want = key
        if key in self._bodies:
            self._bodies.move_to_end(key)
            self._show_champion(key, cg)
            return
        self._request(key, v, c.spec, body)

    def _show_champion(self, key: tuple[Any, ...], cg: int) -> None:
        bodies = self._bodies[key]
        if bodies is None:  # 이 몸은 만들 수 없다(조립 실패): 그림 없이
            self.picture.clear()
            self.caption.setText(S.STRUCT_CHAMPION.format(gen=cg))
            self.champion = (int(cg), None)
            return
        here = frozenset(b.id for b, role in bodies if role != "absent")
        skey = tuple(b for b in self.union_order if b in here)
        color = self.colors[self.series.index(skey)] if skey in self.series else self.theme.muted
        if self.champion != (int(cg), skey) or self.picture.part_color != color or not self.picture.items:
            self.picture.set_bodies(bodies, color)
        name = self.key_names.get(skey) or structure_label(skey, self.labels, [], False)
        self.caption.setText(f"{S.STRUCT_CHAMPION.format(gen=cg)} · {name}")
        self.champion = (int(cg), skey)

    def _request(self, key: tuple[Any, ...], version: int, spec: EnvSpec, body: np.ndarray | None) -> None:
        if self._job is not None:
            self._pending = (key, version, spec, body)
            return
        job = _BodyJob((self._epoch, key, version), spec, body, self._full.get(version),
                       need_full=version not in self._full)
        job.done.connect(self._job_done)
        job.finished.connect(lambda j=job: _forget_job(j))
        _LIVE_JOBS.add(job)
        self._job = job
        job.start()

    def _job_done(self, payload: tuple[_BodyJob, tuple[Any, ...], Any, EnvSpec | None]) -> None:
        job, (epoch, key, version), result, full = payload
        if self._job is job:
            self._job = None
        if epoch == self._epoch:
            if version not in self._full:
                self._full[version] = full
            self._bodies[key] = None if isinstance(result, Exception) else result
            self._bodies.move_to_end(key)
            while len(self._bodies) > BODY_CACHE:
                self._bodies.popitem(last=False)
        pending, self._pending = self._pending, None
        if pending is not None and pending[0] == self._want and pending[0] not in self._bodies:
            self._request(*pending)
        if epoch == self._epoch and key == self._want:
            self.dash._invalidate((self.tile_id,))

    @property
    def body_busy(self) -> bool:
        """챔피언 몸 그림을 배경에서 만드는 중인가(시험용)."""
        return self._job is not None or self._pending is not None

    # -- 범례·마우스 ------------------------------------------------------------
    def entries(self) -> list[_Entry]:
        return [_Entry(c, GLYPH_FILL, n) for c, n in zip(self.colors, self.names, strict=True)]

    def hover(self, xv: float, yv: float, plot_index: int) -> None:
        x = self.gens
        if not len(x) or not len(self.series):
            self.unhover()
            return
        i = int(np.argmin(np.abs(x - xv)))
        for line in self._cross:
            line.setPos(float(x[i]))
            line.show()
        parts = [S.DASH_HOVER_X["generation"].format(gen=int(x[i]), x="")]
        for j in reversed(range(len(self.series))):  # 위 칸부터(그림과 같은 순서)
            v = float(self.ratios[j, i])
            if v > 0:
                parts.append(f"{self.names[j]} {v * 100.0:.0f}%")
        self._hover_text = "  ".join(parts)
        self.update_legend()

    def structure_at(self, generation: int) -> dict[StructureKey, float]:
        """그 세대의 구조별 개체 비율(센 값 그대로, 기타로 묶기 전). 세지 않은 세대면 빈 사전(시험용)."""
        gc = self._counts.get(int(generation))
        if gc is None:
            return {}
        return {self._keys[int(k)]: float(f) for k, f in zip(gc.kids, gc.fracs, strict=True)}

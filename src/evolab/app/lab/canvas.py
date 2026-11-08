"""관전 화면: 한 세대 집단 전체가 시험 0번을 다시 달리는 모습을 겹쳐 그린다(명세 "실험실 > 관전 화면").

- 집단은 반투명하게, 상위 10%는 진하게, 챔피언은 강조색과 지나온 궤적으로 그린다. 끝난 개체는 마지막
  자세로 남되 조금 흐리게 그린다.
- 겹쳐 보기(1세대 대표와 지금 챔피언, 고속 모드의 챔피언 재생)는 각자 색과 범례 이름으로 그린다.
- 개체 256개를 60fps로 그리려고, 물체 모양은 물체 좌표의 QPainterPath로 한 번만 만들어 두고 프레임마다
  물체별 변환(이동·회전)만 바꿔 그린다. 정적인 세계(지형, 영역, 패드)는 카메라가 그대로인 동안 그려 둔
  그림을 다시 쓴다. 맞춤 카메라는 필요한 범위가 화면 밖으로 나가거나 많이 줄었을 때만 목표를 옮겨서
  대부분의 프레임에서 카메라가 멈춰 있다.
- v1.2: 정적 그림 끝에 궤적 밀도 층(set_density, 대시보드 궤적 밀도 타일)을 그릴 수 있다. 정적 그림에 들어가므로
  프레임마다 드는 비용이 없다. 대시보드 안의 작은 캔버스는 Ctrl+휠일 때만 확대한다(set_wheel_needs_ctrl).
- v2.0(설계 5.3, D7): 정해진 움직임 세계 물체(BodyGeom.motion, 장애물 줄 등)는 궤적에 없고 집단 전체가 같은 움직임을
  보므로, 정적 그림 캐시에서 빼고 따로 한 층으로 두었다가 프레임마다 그 시각의 자세(motion.pose_at)로 한 번만 그린다.
  시각은 보여 주는 프레임의 시각(프레임 간격으로 내림)이라 개체와 장애물이 시뮬레이션한 그대로 맞물려 보인다.
  관전·타임라인·1세대 vs 지금·클릭 검사·궤적 밀도 타일·경주 칸·GIF가 모두 이 캔버스다.
- v2.0 기본 보기(D25): set_scene(view=...)로 환경의 editor.view를 주면 처음 카메라가 그 상자다. 맞춤 모드는 모든 개체와
  목표가 상자 안에 있는 동안 상자를 그대로 보이고, 벗어나면 상자와 필요한 범위를 합친 범위로 넓힌다. 챔피언 따라가기는
  상자 배율로 상자를 보다가 따라갈 물체가 보이는 범위의 가장자리를 넘을 때만 옮긴다(직접 조종 화면과 같은 규칙).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPaintEvent,
    QPen,
    QPixmap,
    QPolygonF,
    QResizeEvent,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from evolab.app import strings as S
from evolab.app.play.window import (
    COLOR_AGENT,
    COLOR_AGENT_EDGE,
    COLOR_BG,
    COLOR_BOUNDS,
    COLOR_FLAME,
    COLOR_HUD,
    COLOR_LEG,
    COLOR_PAD,
    COLOR_REGION,
    COLOR_TERRAIN_FILL,
    COLOR_TERRAIN_LINE,
    COLOR_WORLD_BODY,
    VIEW_SCALE,
    body_path,
    view_camera,
)
from evolab.engine.evaluate import Trajectory
from evolab.spec.editor_meta import ViewBox
from evolab.world.physics2d.episode import BodyGeom, ShapeGeom, WorldGeometry
from evolab.world.physics2d.motion import MotionPlan, pose_at

TOP_FRACTION = 0.10
"""상위 몇 %를 진하게 그릴지."""
FOLLOW_MODES = ("fit", "champion")
PICK_PX = 4.0
"""클릭 판정 여유(화면 픽셀)."""
TIME_EPS = 1e-9

FIT_MARGIN = (6.0, 4.0)
"""맞춤 범위 바깥 여유(m): 가로, 세로. 직접 조종 화면과 같다."""
FIT_SLACK = 0.12
"""맞춤 목표를 옮길 때 더 넓게 잡는 비율(조금 움직일 때마다 카메라가 따라 움직이지 않게)."""
FIT_SHRINK = 0.55
"""필요한 범위가 목표의 이 비율보다 작아지면(가로·세로 모두) 다시 좁힌다."""
SCALE_FIT = (1.0, 30.0)
"""맞춤 모드 배율 범위(px/m)."""
SCALE_FOLLOW = (4.0, 30.0)
"""챔피언 따라가기 배율 범위(px/m). 직접 조종 화면과 같다."""
ZOOM_RANGE = (0.25, 8.0)
EASE_CENTER = 0.15
EASE_SCALE = 0.12
SNAP_PX = 0.25
"""카메라가 목표에 이만큼(px) 가까워지면 목표에 붙인다(그 뒤로는 정적 그림을 다시 쓴다)."""
STATIC_MARGIN = 0.25
"""정적 그림을 화면보다 사방으로 이 비율만큼 넓게 그려 둔다. 배율이 그대로면 카메라가 옮겨 가도 밀어서 쓴다."""
CULL_PX = 3.0

COLOR_CHAMPION = QColor("#ffd166")
COLOR_CHAMPION_LEG = QColor("#f4a261")
COLOR_CHAMPION_EDGE = QColor("#fff1c1")
COLOR_HAZARD = QColor(255, 99, 99, 45)
COLOR_ZONE = QColor(255, 255, 255, 18)
COLOR_HUD_BOX = QColor(11, 14, 20, 190)
COLOR_HUD_EDGE = QColor(255, 255, 255, 40)

ALPHA_REST = 60
ALPHA_REST_END = 28
FADE_TOP_END = 0.5
"""끝난 상위 개체는 배경색 쪽으로 이만큼 섞는다. 진한 개체는 불투명하게 칠해야 빠르다(반투명 부드러운 칠은 4배 느림)."""
FADE_HIGHLIGHT_END = 0.35
ALPHA_TRAIL = 210

_ROLE_COLORS = {"agent": COLOR_AGENT, "leg": COLOR_LEG, "world": COLOR_WORLD_BODY}


def _with_alpha(color: QColor, alpha: int) -> QColor:
    c = QColor(color)
    c.setAlpha(alpha)
    return c


def _faded(color: QColor, amount: float) -> QColor:
    """배경색 쪽으로 섞은 불투명한 색(끝난 개체)."""
    a = max(0.0, min(1.0, amount))
    return QColor(
        round(color.red() + (COLOR_BG.red() - color.red()) * a),
        round(color.green() + (COLOR_BG.green() - color.green()) * a),
        round(color.blue() + (COLOR_BG.blue() - color.blue()) * a),
    )


def _pen(color: QColor, width: float = 1.0) -> QPen:
    q = QPen(color, width)
    q.setCosmetic(True)
    q.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    q.setCapStyle(Qt.PenCapStyle.RoundCap)
    return q


@dataclass
class _Style:
    brush: QBrush
    pen: QPen | Qt.PenStyle
    """다각형·원 테두리. 반투명 집단은 테두리 없이 채우기만 한다."""
    line_pen: QPen
    """선분 모양이 있는 물체의 펜(선분은 펜으로만 보인다)."""


def _style(fill: QColor, edge: QColor | None, edge_width: float = 1.0) -> _Style:
    pen: QPen | Qt.PenStyle = _pen(edge, edge_width) if edge is not None else Qt.PenStyle.NoPen
    return _Style(QBrush(fill), pen, _pen(edge if edge is not None else fill, max(1.5, edge_width)))


@dataclass
class _Shape:
    """움직이는 물체 하나의 모양(물체 좌표)."""

    path: QPainterPath
    """다각형·원은 닫힌 부분 경로, 선분은 열린 부분 경로."""
    pick: QPainterPath
    """클릭 판정용(선분에 두께를 준 것)."""
    role: str
    """agent, leg, world: 색을 고르는 몫."""
    lines: bool
    radius: float
    """물체 원점에서 가장 먼 점까지 거리(클릭 판정 빠른 거르기)."""


@dataclass
class _Thruster:
    body: int
    """물체 번호(이 개체의 움직이는 물체 순서)."""
    channel: int
    """Trajectory.outputs 열 번호."""
    at: tuple[float, float]
    degrees: float
    """불꽃 방향(힘의 반대쪽), 물체 좌표 기준 도."""
    strength: float


@dataclass
class _Agent:
    index: int
    """집단 번호. 겹쳐 보기 개체는 겹쳐 보기 목록 번호."""
    traj: Trajectory
    shapes: list[_Shape]
    layer: str
    """rest, top, champion, overlay"""
    rank: int = 0
    label: str = ""
    styles: dict[tuple[str, bool], _Style] | None = None
    """겹쳐 보기 개체만: 자기 색의 스타일 (몫, 끝남)."""
    color: QColor | None = None
    thrusters: list[_Thruster] = field(default_factory=list)
    trail: list[QPointF] = field(default_factory=list)
    """첫 번째 움직이는 물체가 지나간 점(세계 좌표), 프레임마다 하나."""


def _role(body: BodyGeom) -> str:
    if "leg" in body.tags:
        return "leg"
    return "agent" if body.owner == "agent" else "world"


def _shape_key(body: BodyGeom) -> tuple:
    return (_role(body),) + tuple(
        (s.kind, tuple((float(x), float(y)) for x, y in s.points), float(s.radius)) for s in body.shapes
    )


@dataclass
class _Mover:
    """정해진 움직임 세계 물체 하나(물체 좌표 모양과 움직임)."""

    id: str
    motion: MotionPlan
    fill: QPainterPath
    """다각형·원."""
    lines: QPainterPath
    """선분."""


def _used_rect(image: QImage, rect: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    """밀도 그림에서 투명하지 않은 칸이 있는 범위 → 세계 좌표 (x0, y0, x1, y1), 둘레에 한 칸 여유(격자 안으로 자름).
    그림 0행이 세계 y0 쪽(set_density와 같은 규칙). 모두 투명이면 None."""
    img = image.convertToFormat(QImage.Format.Format_Alpha8)  # 칸마다 알파 한 바이트(바이트 순서와 무관)
    nx, ny = img.width(), img.height()
    if nx <= 0 or ny <= 0:
        return None
    raw = np.frombuffer(img.constBits(), dtype=np.uint8, count=img.sizeInBytes()).reshape(ny, img.bytesPerLine())
    rows, cols = np.nonzero(raw[:, :nx])
    if not len(rows):
        return None
    x0, y0, w, h = rect
    cw, ch = w / nx, h / ny
    i0, i1 = max(int(cols.min()) - 1, 0), min(int(cols.max()) + 2, nx)
    j0, j1 = max(int(rows.min()) - 1, 0), min(int(rows.max()) + 2, ny)
    return (x0 + i0 * cw, y0 + j0 * ch, x0 + i1 * cw, y0 + j1 * ch)


def _mover(body: BodyGeom) -> _Mover:
    assert body.motion is not None
    return _Mover(body.id, body.motion, *body_path(body))


def _build_shape(body: BodyGeom) -> _Shape:
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.WindingFill)
    lines = QPainterPath()
    line_width = 0.0
    radius = 0.0
    for s in body.shapes:
        if s.kind == "poly" and len(s.points) >= 3:
            path.addPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in s.points]))
            path.closeSubpath()
            radius = max(radius, max(math.hypot(x, y) for x, y in s.points))
        elif s.kind == "circle" and s.points:
            cx, cy = s.points[0]
            path.addEllipse(QPointF(float(cx), float(cy)), float(s.radius), float(s.radius))
            radius = max(radius, math.hypot(cx, cy) + s.radius)
        elif s.kind == "segment" and len(s.points) >= 2:
            (ax, ay), (bx, by) = s.points[0], s.points[1]
            lines.moveTo(float(ax), float(ay))
            lines.lineTo(float(bx), float(by))
            line_width = max(line_width, 2.0 * s.radius)
            radius = max(radius, math.hypot(ax, ay) + s.radius, math.hypot(bx, by) + s.radius)
    pick = QPainterPath(path)
    has_lines = not lines.isEmpty()
    if has_lines:
        stroker = QPainterPathStroker()
        stroker.setWidth(max(0.05, line_width))
        pick.addPath(stroker.createStroke(lines))
        path.addPath(lines)
    return _Shape(path, pick, _role(body), has_lines, radius)


class SpectatorCanvas(QWidget):
    """집단 전체의 궤적을 같은 시각으로 맞춰 다시 그리는 관전 화면.

    쓰는 순서: set_scene(세계) -> set_population(궤적들) / set_overlay(...) -> set_time(t)를 프레임마다.
    카메라는 직접 조종 화면처럼 y가 위쪽이고 휠로 확대·축소한다.
    """

    individual_clicked = Signal(int)
    """집단 개체를 클릭했다(set_population 목록 번호)."""
    overlay_clicked = Signal(int)
    """겹쳐 보기 개체를 클릭했다(set_overlay 목록 번호)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(False)
        self.world: WorldGeometry | None = None
        self.follow = "fit"
        self.hud = ""
        self.use_static_cache = True
        """정적인 세계를 그려 둔 그림으로 다시 쓴다(성능 측정에서 끄고 비교할 수 있게)."""
        self.empty_text = S.SPEC_NO_SCENE
        """세계가 없을 때 가운데에 쓰는 글. 준비 중을 다른 곳에서 알리는 화면(나란히 경주 칸)은 빈 글로 둔다."""
        self._t = 0.0
        self._zoom = 1.0
        self._center = (0.0, 0.0)
        self._scale = 22.0
        self._fit_rect: tuple[float, float, float, float] | None = None
        self._camera_ready = False

        self._channels: list[str] | None = None
        self._scene_agents: list[list[BodyGeom]] | None = None
        self._default_bodies: list[BodyGeom] = []
        self._shape_cache: dict[tuple, _Shape] = {}
        self._thruster_cache: dict[tuple[str, ...], list[_Thruster]] = {}
        self._targets = np.zeros((0, 2), dtype=np.float64)
        self._terrain_fill: QPainterPath | None = None
        self._terrain_line: QPainterPath | None = None
        self._static_lines: QPainterPath | None = None
        self._static_fill: QPainterPath | None = None
        self._regions: list[tuple[QPolygonF, QColor]] = []
        self._pads: list[tuple[QPointF, QPointF]] = []
        self._static_pixmap: QPixmap | None = None
        self._static_key: tuple | None = None
        """(너비, 높이, 화소 비율, 배율): 정적 그림을 그린 조건."""
        self._static_origin = (0.0, 0.0)
        """정적 그림을 그릴 때의 세계 원점 화면 위치(ox, oy)."""
        self._static_margin = (0, 0)
        self._last_key: tuple | None = None
        self._density: tuple[QImage, QRectF] | None = None
        """정적 그림 끝에 그리는 밀도 층(그림, 세계 좌표 사각형)."""
        self._density_fit = False
        """맞춤 카메라가 밀도 층 사각형을 보이게 하는가(개체가 없는 밀도 타일)."""
        self._density_used: tuple[float, float, float, float] | None = None
        """밀도 층에서 지나간 칸이 있는 범위(세계 좌표 x0, y0, x1, y1). 없으면 None."""
        self._wheel_needs_ctrl = False
        self._movers: list[_Mover] = []
        """정해진 움직임 세계 물체(정적 그림 캐시 밖, 프레임마다 그 시각의 자세로)."""
        self._show_movers = True
        """정해진 움직임 세계 물체를 그리는가(set_show_movers). 시각이 없는 그림(궤적 밀도 타일)은 끈다."""
        self.mover_poses: list[tuple[float, float, float]] = []
        """마지막으로 그린 정해진 움직임 물체의 자세(_movers 순서). 테스트가 읽는다."""
        self.mover_time = 0.0
        """마지막으로 그린 정해진 움직임 물체의 시각(보여 주는 프레임의 시각)."""
        self._view: ViewBox | None = None
        """기본 보기(editor.view): 처음 카메라. None이면 지금 규칙."""

        self._population: list[_Agent] = []
        self._overlay: list[_Agent] = []
        self._agents: list[_Agent] = []
        """그리는 순서와 무관한 전체 목록(집단 다음 겹쳐 보기). 프레임 배열의 순서."""
        self._layers: list[tuple[str, list[int]]] = []
        """아래에서 위로: (층 이름, self._agents 번호들)."""
        self._has_top = False
        self._has_rest = False
        self._champion_slot: int | None = None
        self._offsets = np.zeros(1, dtype=np.int64)
        self._step_t = np.zeros(0)
        self._frames = np.zeros(0, dtype=np.int64)
        self._last_t = np.zeros(0)
        self._k = np.zeros(0, dtype=np.int64)
        self._ended = np.zeros(0, dtype=bool)
        self._poses = np.zeros((0, 3), dtype=np.float64)
        self._radii = np.zeros(0, dtype=np.float64)
        """물체마다 모양 반지름(프레임 배열 순서)."""

        self._styles = {
            ("rest", False): {r: _style(_with_alpha(c, ALPHA_REST), None) for r, c in _ROLE_COLORS.items()},
            ("rest", True): {r: _style(_with_alpha(c, ALPHA_REST_END), None) for r, c in _ROLE_COLORS.items()},
            ("top", False): {r: _style(QColor(c), COLOR_AGENT_EDGE) for r, c in _ROLE_COLORS.items()},
            ("top", True): {
                r: _style(_faded(c, FADE_TOP_END), _faded(COLOR_AGENT_EDGE, FADE_TOP_END)) for r, c in _ROLE_COLORS.items()
            },
            ("champion", False): self._highlight_styles(COLOR_CHAMPION, COLOR_CHAMPION_LEG, COLOR_CHAMPION_EDGE, 0.0),
            ("champion", True): self._highlight_styles(
                COLOR_CHAMPION, COLOR_CHAMPION_LEG, COLOR_CHAMPION_EDGE, FADE_HIGHLIGHT_END
            ),
        }
        self._font = QFont(self.font())
        self._font.setPointSizeF(max(7.0, self._font.pointSizeF() * 0.9))
        self._font_metrics = QFontMetricsF(self._font)

    @staticmethod
    def _highlight_styles(body: QColor, leg: QColor, edge: QColor, fade: float) -> dict[str, _Style]:
        colors = {"agent": body, "leg": leg, "world": COLOR_WORLD_BODY}
        return {r: _style(_faded(c, fade), _faded(edge, fade), 1.5) for r, c in colors.items()}

    # ------------------------------------------------------------------
    # 장면과 집단
    # ------------------------------------------------------------------

    def set_scene(
        self, world: WorldGeometry, agents: list[list[BodyGeom]] | None = None, channels: list[str] | None = None,
        view: ViewBox | None = None,
    ) -> None:
        """시험 하나의 세계(정적 물체, 영역, 추진기)를 정한다. 집단과 겹쳐 보기는 비운다(다른 세계의 궤적이라).

        agents[i]: 개체 i의 움직이는 물체(Episode.moving_bodies 순서). 몸 유전자가 진화하면 개체마다 모양이
        다를 수 있다. None이면 모두 world의 움직이는 물체 모양을 쓴다.
        channels: Trajectory.outputs 열 순서의 구동기 채널 이름(Episode.actuator_channels()). None이면
        world.thrusters에 처음 나오는 채널 순서로 본다(추진기만 있는 환경에서 맞는 추정).
        view: 환경의 기본 보기 (x0, y0, x1, y1)(spec.editor_meta.view_box). 있으면 처음 카메라가 그 상자다.
        """
        self.world = world
        self._view = view
        self._scene_agents = [list(a) for a in agents] if agents is not None else None
        self._default_bodies = [b for b in world.bodies if b.moving]
        if channels is None:
            channels = list(dict.fromkeys(t.channel for t in world.thrusters))
        self._channels = list(channels)
        self._thruster_cache.clear()
        self._shape_cache.clear()
        pts = [pt for r in world.regions if r.kind in ("pad", "goal") for pt in r.points]
        self._targets = np.array(pts, dtype=np.float64).reshape(-1, 2)
        self._density = None  # 다른 세계의 밀도 층은 버린다
        self._density_fit = False
        self._density_used = None
        self._prepare_static(world)
        self._population = []
        self._overlay = []
        self._rays = None
        self._rebuild()
        self._fit_rect = None
        self._update_camera(snap=True)
        self.update()

    def set_population(
        self, trajectories: list[Trajectory | None], *, ranks: np.ndarray | None = None,
        champion: int | None = None, labels: dict[int, str] | None = None,
    ) -> None:
        """집단의 궤적(시험 0번). None인 개체는 건너뛴다.

        ranks[i]: 개체 i의 순위(0이 최고). 상위 10%는 진하게, 나머지는 반투명하게 그린다(None이면 모두 반투명).
        champion: 강조색과 지나온 궤적, 추진기 불꽃으로 그릴 개체. labels: 개체 옆에 붙일 이름표.
        """
        if self.world is None:
            if not trajectories:
                return  # 비우기는 장면 전에도 된다
            raise RuntimeError("set_scene must be called before set_population")
        n = len(trajectories)
        if ranks is not None:
            ranks = np.asarray(ranks).reshape(-1)
            if len(ranks) != n:
                raise ValueError(f"ranks has {len(ranks)} entries for {n} trajectories")
        if champion is not None and not 0 <= champion < n:
            raise ValueError(f"champion {champion} out of range for {n} trajectories")
        if self._scene_agents is not None and len(self._scene_agents) < n:
            raise ValueError(f"set_scene got bodies for {len(self._scene_agents)} individuals, not {n}")
        n_top = max(1, math.ceil(TOP_FRACTION * n)) if n else 0
        labels = labels or {}
        out: list[_Agent] = []
        for i, tr in enumerate(trajectories):
            if tr is None or tr.frames == 0:
                continue
            bodies = self._scene_agents[i] if self._scene_agents is not None else self._default_bodies
            rank = int(ranks[i]) if ranks is not None else n
            if i == champion:
                layer = "champion"
            elif ranks is not None and rank < n_top:
                layer = "top"
            else:
                layer = "rest"
            agent = _Agent(i, tr, self._shapes_for(bodies, tr), layer, rank, labels.get(i, ""))
            if layer == "champion":
                self._decorate(agent, bodies)
            out.append(agent)
        self._population = out
        self._rebuild()
        self._update_camera(snap=not self._camera_ready)
        self.update()

    def set_overlay(self, items: list[tuple[Trajectory, list[BodyGeom], QColor, str]]) -> None:
        """따로 강조할 개체들(자기 색, 범례 이름, 지나온 궤적, 불꽃). 빈 목록이면 지운다."""
        if self.world is None:
            if not items:
                return  # 비우기는 장면 전에도 된다
            raise RuntimeError("set_scene must be called before set_overlay")
        out: list[_Agent] = []
        for k, (tr, bodies, color, label) in enumerate(items):
            if tr is None or tr.frames == 0:
                continue
            agent = _Agent(k, tr, self._shapes_for(list(bodies), tr), "overlay", 0, label)
            agent.color = QColor(color)
            styles = self._highlight_styles(QColor(color), QColor(color).darker(125), QColor(color).lighter(160), 0.0)
            ended = self._highlight_styles(
                QColor(color), QColor(color).darker(125), QColor(color).lighter(160), FADE_HIGHLIGHT_END
            )
            agent.styles = {(r, False): st for r, st in styles.items()} | {(r, True): st for r, st in ended.items()}
            self._decorate(agent, list(bodies))
            out.append(agent)
        self._overlay = out
        self._rebuild()
        self._update_camera(snap=not self._camera_ready)
        self.update()

    def set_time(self, t: float) -> None:
        """에피소드 시작부터의 시각(초). 개체마다 시각이 t 이하인 마지막 프레임을 보여 준다."""
        self._t = max(0.0, float(t))
        self._compute_frame()
        self._update_camera(snap=False)
        self.update()

    def duration(self) -> float:
        """집단과 겹쳐 보기 궤적 중 가장 긴 것의 길이(초)."""
        return float(self._last_t.max()) if len(self._last_t) else 0.0

    def current_time(self) -> float:
        return self._t

    def set_rays(self, rays: np.ndarray | None, dt: float = 1.0 / 60.0) -> None:
        """들여다보는 개체의 센서 레이(스텝마다). rays: (스텝, 레이 수, 5) = 시작 x, y, 끝 x, y, 맞은 비율(1이면 안 맞음)."""
        self._rays = None if rays is None or not len(rays) else np.asarray(rays, dtype=np.float32)
        self._rays_dt = float(dt)
        self.update()

    def _draw_rays(self, p: QPainter) -> None:
        rays = getattr(self, "_rays", None)
        if rays is None:
            return
        k = int(min(len(rays) - 1, max(0, math.floor(self._t / self._rays_dt + 1e-9))))
        for sx, sy, ex, ey, alpha in rays[k]:
            near = 1.0 - float(alpha)
            color = QColor(int(80 + 175 * near), int(220 - 150 * near), 90, 200)
            pen = QPen(color, 0)
            pen.setCosmetic(True)
            pen.setWidthF(1.5)
            p.setPen(pen)
            p.drawLine(QPointF(float(sx), float(sy)), QPointF(float(ex), float(ey)))
            if alpha < 1.0:
                p.setBrush(color)
                r = 0.25
                p.drawEllipse(QPointF(float(ex), float(ey)), r, r)

    def set_density(
        self, image: QImage | None, rect: tuple[float, float, float, float] | None = None, *, fit: bool = False,
    ) -> None:
        """정적 그림 끝에 밀도 층을 그린다. rect = 세계 좌표 (x0, y0, 너비, 높이), 그림 0행이 y0(아래) 쪽.
        칸은 부드럽게 늘이지 않는다(칸 모양 그대로). fit이면 맞춤 카메라가 개체 대신 이 사각형을 보인다.
        image가 None이면 층을 지운다. 정적 그림을 한 번 다시 그릴 뿐 프레임마다 드는 비용은 없다."""
        if image is None or rect is None or image.isNull():
            self._density = None
            self._density_fit = False
            self._density_used = None
        else:
            x0, y0, w, h = (float(v) for v in rect)
            self._density = (image, QRectF(x0, y0, w, h))
            self._density_fit = bool(fit)
            self._density_used = _used_rect(image, (x0, y0, w, h))
        self._static_key = None
        if self.world is not None:
            self._fit_rect = None
            self._update_camera(snap=True)
        self.update()

    def density_layer(self) -> tuple[QImage, QRectF] | None:
        """지금 밀도 층(그림, 세계 좌표 사각형) 또는 None."""
        return self._density

    def set_show_movers(self, on: bool) -> None:
        """정해진 움직임 세계 물체(장애물 줄 등)를 그릴지. 궤적 밀도 타일처럼 한 시각이 아닌 그림은 끈다: 끄지 않으면
        장애물 줄이 0초 자리에 멈춘 채로 그려져 개체가 닿지 않은 정적 장애물처럼 보인다."""
        self._show_movers = bool(on)
        self.update()

    def set_wheel_needs_ctrl(self, on: bool) -> None:
        """True면 Ctrl+휠일 때만 확대·축소하고 그냥 휠은 부모(대시보드 스크롤)에 넘긴다(대시보드 그래프와 같은 규칙)."""
        self._wheel_needs_ctrl = bool(on)

    def set_follow(self, mode: str) -> None:
        """fit: 모든 개체와 패드·목표가 보이게(기본). champion: 챔피언(없으면 첫 겹쳐 보기)과 목표를 따라간다."""
        if mode not in FOLLOW_MODES:
            raise ValueError(f"unknown follow mode {mode!r}")
        self.follow = mode
        self._fit_rect = None
        self._update_camera(snap=True)
        self.update()

    def set_hud(self, text: str) -> None:
        """구석의 작은 글상자(세대, 시각, 배속 등). 빈 글이면 숨긴다."""
        self.hud = text
        self.update()

    def agent_frame(self, index: int) -> tuple[int, bool] | None:
        """집단 개체 index가 지금 보여 주는 (프레임 번호, 끝났는지). 궤적이 없으면 None."""
        for slot, a in enumerate(self._agents):
            if a.layer != "overlay" and a.index == index:
                return int(self._k[slot]), bool(self._ended[slot])
        return None

    def overlay_frame(self, index: int) -> tuple[int, bool] | None:
        """겹쳐 보기 개체 index가 지금 보여 주는 (프레임 번호, 끝났는지)."""
        for slot, a in enumerate(self._agents):
            if a.layer == "overlay" and a.index == index:
                return int(self._k[slot]), bool(self._ended[slot])
        return None

    # ---- 만들기 ----
    def _shapes_for(self, bodies: list[BodyGeom], tr: Trajectory) -> list[_Shape]:
        if tr.poses.ndim != 3 or tr.poses.shape[1] != len(bodies):
            raise ValueError(f"trajectory has {tr.poses.shape[1:]} poses for {len(bodies)} moving bodies")
        out = []
        for b in bodies:
            key = _shape_key(b)
            shape = self._shape_cache.get(key)
            if shape is None:
                shape = self._shape_cache[key] = _build_shape(b)
            out.append(shape)
        return out

    def _decorate(self, agent: _Agent, bodies: list[BodyGeom]) -> None:
        """강조 개체: 불꽃용 추진기 표와 지나온 궤적 점."""
        ids = tuple(b.id for b in bodies)
        thr = self._thruster_cache.get(ids)
        if thr is None:
            thr = self._thruster_cache[ids] = self._thrusters_for(ids)
        agent.thrusters = [t for t in thr if t.channel < agent.traj.outputs.shape[1]]
        if agent.traj.poses.shape[1]:
            xy = np.nan_to_num(agent.traj.poses[:, 0, :2].astype(np.float64), posinf=0.0, neginf=0.0)
            agent.trail = [QPointF(x, y) for x, y in xy.tolist()]

    def _thrusters_for(self, ids: tuple[str, ...]) -> list[_Thruster]:
        assert self.world is not None
        index = {b: j for j, b in enumerate(ids)}
        channels = {c: k for k, c in enumerate(self._channels or [])}
        out: list[_Thruster] = []
        for t in self.world.thrusters:
            if t.body not in index or t.channel not in channels:
                continue
            out.append(_Thruster(index[t.body], channels[t.channel], (float(t.at[0]), float(t.at[1])),
                                 math.degrees(t.dir + math.pi), min(1.0, t.max_force / 200.0 + 0.3)))
        return out

    def _rebuild(self) -> None:
        """그리는 층과 프레임 배열을 다시 만든다."""
        pop = self._population
        rest = sorted((a for a in pop if a.layer == "rest"), key=lambda a: (-a.rank, a.index))
        top = sorted((a for a in pop if a.layer == "top"), key=lambda a: (-a.rank, a.index))
        champ = [a for a in pop if a.layer == "champion"]
        self._agents = rest + top + champ + list(self._overlay)
        r, t, c = len(rest), len(top), len(champ)
        slots = list(range(len(self._agents)))
        self._layers = [
            ("rest", slots[:r]), ("top", slots[r:r + t]), ("champion", slots[r + t:r + t + c]),
            ("overlay", slots[r + t + c:]),
        ]
        self._has_rest, self._has_top = bool(r), bool(t)
        self._champion_slot = r + t if c else None
        counts = [len(a.shapes) for a in self._agents]
        self._offsets = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(np.int64)
        self._radii = np.array([sh.radius for a in self._agents for sh in a.shapes], dtype=np.float64)
        self._step_t = np.array([max(a.traj.every * a.traj.dt, 1e-12) for a in self._agents], dtype=np.float64)
        self._frames = np.array([a.traj.frames for a in self._agents], dtype=np.int64)
        self._last_t = np.array([a.traj.duration for a in self._agents], dtype=np.float64)
        self._compute_frame()

    def _compute_frame(self) -> None:
        """개체마다 시각 t 이하의 마지막 프레임과 그 자세(모든 물체를 이어 붙인 (물체 수, 3))."""
        n = len(self._agents)
        if not n:
            self._k = np.zeros(0, dtype=np.int64)
            self._ended = np.zeros(0, dtype=bool)
            self._poses = np.zeros((0, 3), dtype=np.float64)
            return
        t = self._t
        ended = t >= self._last_t - TIME_EPS
        # 마지막 프레임 앞은 every 스텝 간격이다. 마지막 프레임은 그 간격보다 가까울 수 있어 따로 본다.
        k = np.floor(t / self._step_t + TIME_EPS).astype(np.int64)
        k = np.where(ended, self._frames - 1, np.minimum(k, self._frames - 2))
        k = np.maximum(k, 0)
        self._k = k
        self._ended = ended
        poses = np.concatenate([a.traj.poses[ki] for a, ki in zip(self._agents, k.tolist(), strict=True)])
        # 물리가 터진 프레임(NaN)이 카메라와 변환을 망가뜨리지 않게
        self._poses = np.nan_to_num(poses.astype(np.float64).reshape(-1, 3), posinf=0.0, neginf=0.0)

    def _prepare_static(self, g: WorldGeometry) -> None:
        terrain_line = QPainterPath()
        terrain_fill = QPainterPath()
        lines = QPainterPath()
        fill = QPainterPath()
        fill.setFillRule(Qt.FillRule.WindingFill)
        bottom = g.bounds[1] - 50.0
        self._movers = []
        for b in g.bodies:
            if b.moving:
                continue
            if b.motion is not None:  # 정해진 움직임: 정적 그림 캐시에 넣지 않고 프레임마다 그 시각의 자세로
                self._movers.append(_mover(b))
                continue
            x0, y0, a0 = b.pose
            c, s = math.cos(a0), math.sin(a0)

            def world(p: tuple[float, float], x0: float = x0, y0: float = y0, c: float = c, s: float = s) -> QPointF:
                return QPointF(x0 + p[0] * c - p[1] * s, y0 + p[0] * s + p[1] * c)

            segs = [sh for sh in b.shapes if sh.kind == "segment" and len(sh.points) >= 2]
            if segs and "terrain" in b.tags:
                # 지형은 이어진 선분 사슬: 선과 그 아래 채우기(직접 조종 화면과 같은 모양)
                pts = [world(p) for sh in segs for p in sh.points[:2]]
                terrain_line.moveTo(pts[0])
                for q in pts[1:]:
                    terrain_line.lineTo(q)
                poly = QPolygonF(pts)
                poly.append(QPointF(pts[-1].x(), bottom))
                poly.append(QPointF(pts[0].x(), bottom))
                terrain_fill.addPolygon(poly)
            else:
                for sh in segs:
                    lines.moveTo(world(sh.points[0]))
                    lines.lineTo(world(sh.points[1]))
            for sh in b.shapes:
                self._add_static_shape(fill, sh, world)
        self._terrain_line = terrain_line
        self._terrain_fill = terrain_fill
        self._static_lines = lines
        self._static_fill = fill
        self._regions = []
        self._pads = []
        for r in g.regions:
            color = {"pad": COLOR_REGION, "goal": COLOR_REGION, "hazard": COLOR_HAZARD}.get(r.kind, COLOR_ZONE)
            if len(r.points) >= 3:
                self._regions.append((QPolygonF([QPointF(float(x), float(y)) for x, y in r.points]), color))
            if r.kind == "pad" and len(r.points) >= 2:
                self._pads.append((QPointF(*r.points[0]), QPointF(*r.points[1])))
        self._static_key = None

    @staticmethod
    def _add_static_shape(fill: QPainterPath, sh: ShapeGeom, world: Callable[[tuple[float, float]], QPointF]) -> None:
        if sh.kind == "poly" and len(sh.points) >= 3:
            fill.addPolygon(QPolygonF([world(p) for p in sh.points]))
            fill.closeSubpath()
        elif sh.kind == "circle" and sh.points:
            fill.addEllipse(world(sh.points[0]), sh.radius, sh.radius)

    # ------------------------------------------------------------------
    # 카메라
    # ------------------------------------------------------------------

    def _base(self) -> tuple[float, float, float]:
        """세계 -> 화면: (배율, x 이동, y 이동). 화면 x = s*x + ox, 화면 y = -s*y + oy."""
        s = self._scale
        cx, cy = self._center
        return s, self.width() / 2.0 - cx * s, self.height() / 2.0 + cy * s

    def world_to_screen(self, x: float, y: float) -> QPointF:
        s, ox, oy = self._base()
        return QPointF(x * s + ox, -y * s + oy)

    def screen_to_world(self, pos: QPointF) -> tuple[float, float]:
        s, ox, oy = self._base()
        return (pos.x() - ox) / s, (oy - pos.y()) / s

    def camera(self) -> tuple[tuple[float, float], float]:
        """지금 카메라: (중심, 배율 px/m)."""
        return self._center, self._scale

    def _follow_slot(self) -> int | None:
        """따라갈 개체: 챔피언, 없으면 첫 겹쳐 보기 개체(움직이는 물체가 있어야 한다)."""
        overlay = self._layers[3][1] if self._layers else []
        for slot in ([self._champion_slot] if self._champion_slot is not None else []) + overlay[:1]:
            if self._agents[slot].shapes:
                return slot
        return None

    def _camera_goal(self) -> tuple[tuple[float, float], float] | None:
        g = self.world
        if g is None:
            return None
        w, h = max(self.width(), 100), max(self.height(), 100)
        mx, my = FIT_MARGIN
        slot = self._follow_slot() if self.follow == "champion" else None
        if slot is not None and self._view is not None and not self._density_fit:
            # 기본 보기: 직접 조종 화면과 같은 규칙(상자 배율로 상자를 보다가 가장자리를 넘을 때만 옮김)
            x, y = self._poses[self._offsets[slot], :2]
            return view_camera(self._view, (float(x), float(y)), w, h, self._zoom)
        if slot is not None:
            # 직접 조종 화면과 같은 카메라: 따라갈 물체와 목표 영역이 함께 보이게
            x, y = self._poses[self._offsets[slot], :2]
            pts = np.vstack([[[x, y]], self._targets])
            x0, y0 = pts.min(axis=0) - (mx, my)
            x1, y1 = pts.max(axis=0) + (mx, my)
            fit = min(w / (x1 - x0), h / (y1 - y0))
            scale = min(SCALE_FOLLOW[1], max(SCALE_FOLLOW[0], fit)) * self._zoom
            return ((x0 + x1) / 2.0, (y0 + y1) / 2.0), scale
        rect = self._fit_target()
        x0, y0, x1, y1 = rect
        fit = min(w / max(x1 - x0, 1e-6), h / max(y1 - y0, 1e-6))
        # 기본 보기가 있으면 사용자가 고른 상자 크기 그대로(상자를 합친 범위도 같은 배율 범위라 넓어질 때 튀지 않음)
        lo, hi = VIEW_SCALE if self._view is not None else SCALE_FIT
        scale = min(hi, max(lo, fit)) * self._zoom
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0), scale

    def _inside_view(self) -> bool:
        """모든 개체(모양 반지름까지)와 패드·목표가 기본 보기 상자 안에 있는가."""
        assert self._view is not None
        x0, y0, x1, y1 = self._view
        t = self._targets
        if len(t) and not ((t[:, 0] >= x0).all() and (t[:, 0] <= x1).all()
                           and (t[:, 1] >= y0).all() and (t[:, 1] <= y1).all()):
            return False
        if not len(self._poses):
            return True
        P, r = self._poses, self._radii
        return bool(((P[:, 0] - r) >= x0).all() and ((P[:, 0] + r) <= x1).all()
                    and ((P[:, 1] - r) >= y0).all() and ((P[:, 1] + r) <= y1).all())

    def _fit_target(self) -> tuple[float, float, float, float]:
        """모든 개체(지금 자세)와 패드·목표가 들어오는 범위. 범위를 벗어나거나 많이 줄 때만 옮긴다.

        기본 보기가 있으면 모두 그 상자 안에 있는 동안 상자 그대로, 벗어나면 상자와 필요한 범위를 합친 범위."""
        g = self.world
        assert g is not None
        if self._density_fit and self._density is not None:
            r = self._density[1]
            full = (r.left(), r.top(), r.right(), r.bottom())  # 세계 좌표라 top이 y0(작은 쪽)
            used = self._density_used
            w, h = max(self.width(), 100), max(self.height(), 100)
            if used is not None and min(w / max(r.width(), 1e-6), h / max(r.height(), 1e-6)) < SCALE_FIT[0]:
                # 격자 전체가 가장 작은 배율로도 다 보이지 않는 넓은 세계(글라이더 1,230m): 가운데만 보이면 지나간 칸이
                # 화면 밖에 남으므로 지나간 칸 범위를 보인다(v1 예제 크기의 격자는 지금처럼 전체)
                return used
            return full
        if self._view is not None and self._inside_view():
            self._fit_rect = self._view
            return self._view
        mx, my = FIT_MARGIN
        bx0, by0, bx1, by1 = g.bounds
        pts = [self._targets]
        if len(self._poses):
            pts.append(self._poses[:, :2])
        allp = np.vstack(pts) if any(len(p) for p in pts) else np.zeros((0, 2))
        if not len(allp):
            need = (bx0, by0, bx1, by1)
        else:
            lo = allp.min(axis=0)
            hi = allp.max(axis=0)
            need = (
                max(lo[0] - mx, bx0 - mx), max(lo[1] - my, by0 - my),
                min(hi[0] + mx, bx1 + mx), min(hi[1] + my, by1 + my),
            )
            if need[2] <= need[0] or need[3] <= need[1]:  # 모두 세계 밖(있을 수 없지만): 그대로 둔다
                need = (lo[0] - mx, lo[1] - my, hi[0] + mx, hi[1] + my)
        if self._view is not None:  # 상자를 벗어난 개체가 있다: 상자와 합친 범위
            vx0, vy0, vx1, vy1 = self._view
            need = (min(need[0], vx0), min(need[1], vy0), max(need[2], vx1), max(need[3], vy1))
        r = self._fit_rect
        nw, nh = need[2] - need[0], need[3] - need[1]
        if r is not None:
            inside = r[0] <= need[0] and r[1] <= need[1] and need[2] <= r[2] and need[3] <= r[3]
            rw, rh = r[2] - r[0], r[3] - r[1]
            if inside and not (nw < FIT_SHRINK * rw and nh < FIT_SHRINK * rh):
                return r
        sx, sy = FIT_SLACK * nw, FIT_SLACK * nh
        self._fit_rect = (need[0] - sx, need[1] - sy, need[2] + sx, need[3] + sy)
        return self._fit_rect

    def _update_camera(self, snap: bool) -> None:
        goal = self._camera_goal()
        if goal is None:
            return
        (gx, gy), gs = goal
        if snap or not self._camera_ready:
            self._center, self._scale = (gx, gy), gs
            # 개체가 하나도 없을 때 맞춘 카메라(패드만)는 첫 집단이 오면 한 번 더 바로 맞춘다
            self._camera_ready = bool(self._agents)
            return
        cx, cy = self._center
        s = self._scale
        cx += (gx - cx) * EASE_CENTER
        cy += (gy - cy) * EASE_CENTER
        s += (gs - s) * EASE_SCALE
        close = abs(gx - cx) * s < SNAP_PX and abs(gy - cy) * s < SNAP_PX
        span = max(self.width(), self.height()) / 2.0
        if close and abs(gs - s) / gs * span < SNAP_PX:
            cx, cy, s = gx, gy, gs
        self._center, self._scale = (cx, cy), s

    # ------------------------------------------------------------------
    # 그리기
    # ------------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.world is None:
            p.fillRect(self.rect(), COLOR_BG)
            if self.empty_text:
                p.setPen(COLOR_HUD)
                p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)
            p.end()
            return
        s, ox, oy = self._base()
        base = QTransform(s, 0.0, 0.0, -s, ox, oy)
        self._paint_static(p, base, s, ox, oy)
        if self._movers and self._show_movers:
            self._draw_movers(p, base)
        if self._agents:
            mats, visible = self._body_transforms(s, ox, oy)
            self._draw_agents(p, mats, visible)
            p.setTransform(base)
            self._draw_trails(p)
            self._draw_flames(p, mats)
            p.resetTransform()
            self._draw_labels(p, mats)
        p.setTransform(base)
        self._draw_rays(p)
        p.resetTransform()
        self._draw_legend(p)
        self._draw_hud(p)
        p.end()

    def _paint_static(self, p: QPainter, base: QTransform, s: float, ox: float, oy: float) -> None:
        """정적인 세계. 그려 둔 그림이 있으면 밀어서 쓰고, 배율이 두 프레임 연속 같을 때만 새로 그려 둔다
        (배율이 계속 바뀌는 동안은 바로 그리는 편이 싸다)."""
        w, h = self.width(), self.height()
        key = (w, h, self.devicePixelRatioF(), s)
        if self.use_static_cache:
            mx, my = self._static_margin
            if self._static_pixmap is not None and key == self._static_key:
                dx, dy = round(ox - self._static_origin[0]), round(oy - self._static_origin[1])
                if abs(dx) <= mx and abs(dy) <= my:
                    self._last_key = key
                    p.drawPixmap(dx - mx, dy - my, self._static_pixmap)
                    return
            if key == self._last_key:
                self._build_static(key, s, ox, oy)
                mx, my = self._static_margin
                p.drawPixmap(-mx, -my, self._static_pixmap)  # type: ignore[arg-type]
                return
        self._last_key = key
        p.fillRect(self.rect(), COLOR_BG)
        p.setTransform(base)
        self._draw_static(p)
        p.resetTransform()

    def _build_static(self, key: tuple, s: float, ox: float, oy: float) -> None:
        dpr = self.devicePixelRatioF()
        mx, my = round(self.width() * STATIC_MARGIN), round(self.height() * STATIC_MARGIN)
        pix = QPixmap(max(1, round((self.width() + 2 * mx) * dpr)), max(1, round((self.height() + 2 * my) * dpr)))
        pix.setDevicePixelRatio(dpr)
        pix.fill(COLOR_BG)
        q = QPainter(pix)
        q.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        q.setTransform(QTransform(s, 0.0, 0.0, -s, ox + mx, oy + my))
        self._draw_static(q)
        q.end()
        self._static_pixmap = pix
        self._static_key = key
        self._static_origin = (ox, oy)
        self._static_margin = (mx, my)

    def _draw_static(self, p: QPainter) -> None:
        g = self.world
        assert g is not None
        x0, y0, x1, y1 = g.bounds
        p.setPen(_pen(COLOR_BOUNDS, 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(QPointF(x0, y0), QPointF(x1, y1)))
        p.setPen(Qt.PenStyle.NoPen)
        for poly, color in self._regions:
            p.setBrush(QBrush(color))
            p.drawPolygon(poly)
        if self._terrain_fill is not None and not self._terrain_fill.isEmpty():
            p.setBrush(QBrush(COLOR_TERRAIN_FILL))
            p.drawPath(self._terrain_fill)
        if self._static_fill is not None and not self._static_fill.isEmpty():
            p.setPen(_pen(COLOR_TERRAIN_LINE, 1.0))
            p.setBrush(QBrush(COLOR_WORLD_BODY))
            p.drawPath(self._static_fill)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(_pen(COLOR_TERRAIN_LINE, 1.5))
        for path in (self._terrain_line, self._static_lines):
            if path is not None and not path.isEmpty():
                p.drawPath(path)
        p.setPen(_pen(COLOR_PAD, 4.0))
        for a, b in self._pads:
            p.drawLine(a, b)
        if self._density is not None:
            image, rect = self._density
            p.save()
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            p.drawImage(rect, image)
            p.restore()

    def frame_time(self) -> float:
        """정해진 움직임 물체를 그릴 시각: 보여 주는 프레임의 시각(지금 시각을 개체 프레임 간격으로 내림). 개체가 없으면
        지금 시각. 개체는 프레임 사이를 잇지 않고 그 시각 이하의 마지막 프레임을 보이므로, 장애물도 같은 시각에 두어야
        시뮬레이션에서처럼 맞물린다(16m/s 장애물이면 프레임 간격 1/30초에 0.5m 차이)."""
        if not len(self._step_t):
            return self._t
        step = float(self._step_t.min())
        return math.floor(self._t / step + TIME_EPS) * step

    def _draw_movers(self, p: QPainter, base: QTransform) -> None:
        """정해진 움직임 세계 물체 층: 그 시각의 자세(pose_at)로 한 번씩(집단 전체가 같은 움직임을 본다)."""
        t = self.frame_time()
        poses = [pose_at(mv.motion, t) for mv in self._movers]
        self.mover_poses = poses
        self.mover_time = t
        fill_pen = _pen(COLOR_TERRAIN_LINE, 1.0)
        line_pen = _pen(COLOR_WORLD_BODY, 2.0)
        brush = QBrush(COLOR_WORLD_BODY)
        for mv, (x, y, ang) in zip(self._movers, poses, strict=True):
            m = QTransform()
            m.translate(x, y)
            m.rotateRadians(ang)
            p.setTransform(m * base)
            if not mv.fill.isEmpty():
                p.setPen(fill_pen)
                p.setBrush(brush)
                p.drawPath(mv.fill)
            if not mv.lines.isEmpty():
                p.setPen(line_pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(mv.lines)
        p.resetTransform()

    def _body_transforms(self, s: float, ox: float, oy: float) -> tuple[list[list[float]], list[bool]]:
        """물체마다 물체 좌표 -> 화면 변환 (m11, m12, m21, m22, dx, dy)과 화면에 걸치는지."""
        P = self._poses
        c = np.cos(P[:, 2]) * s
        sn = np.sin(P[:, 2]) * s
        x = P[:, 0] * s + ox
        y = -P[:, 1] * s + oy
        r = self._radii * s + CULL_PX
        visible = (x > -r) & (x < self.width() + r) & (y > -r) & (y < self.height() + r)
        return np.stack([c, -sn, -sn, -c, x, y], axis=1).tolist(), visible.tolist()

    def _layer_runs(self) -> list[tuple[str, bool, list[int]]]:
        """아래에서 위로 그릴 (층, 끝남, 번호들). 끝난 개체를 같은 층의 진행 중인 개체 아래에 둔다."""
        runs: list[tuple[str, bool, list[int]]] = []
        ended = self._ended
        for name, slots in self._layers:
            if not slots:
                continue
            if name == "overlay":
                runs.extend((name, bool(ended[i]), [i]) for i in slots)
                continue
            done = [i for i in slots if ended[i]]
            live = [i for i in slots if not ended[i]]
            if done:
                runs.append((name, True, done))
            if live:
                runs.append((name, False, live))
        return runs

    def _draw_agents(self, p: QPainter, mats: list[list[float]], visible: list[bool]) -> None:
        agents = self._agents
        offsets = self._offsets.tolist()
        for name, ended, slots in self._layer_runs():
            # 반투명 집단은 계단 현상이 거의 보이지 않는다. 부드럽게 그리기를 끄면 몇 배 빠르다(256개 기준 15 -> 2ms).
            p.setRenderHint(QPainter.RenderHint.Antialiasing, name != "rest")
            styles = self._styles.get((name, ended))
            depth = max(len(agents[i].shapes) for i in slots)
            last: _Style | None = None
            last_lines = False
            # 물체 순서대로(모든 개체의 몸통, 그다음 다리 ...): 붓을 바꾸는 횟수를 줄인다
            for j in range(depth):
                for i in slots:
                    a = agents[i]
                    if j >= len(a.shapes):
                        continue
                    f = offsets[i] + j
                    if not visible[f]:
                        continue
                    shape = a.shapes[j]
                    st = a.styles[(shape.role, ended)] if a.styles is not None else styles[shape.role]  # type: ignore[index]
                    if st is not last or shape.lines != last_lines:
                        p.setBrush(st.brush)
                        p.setPen(st.line_pen if shape.lines else st.pen)
                        last, last_lines = st, shape.lines
                    p.setTransform(QTransform(*mats[f]))
                    p.drawPath(shape.path)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    def _highlight_slots(self) -> list[int]:
        out = [] if self._champion_slot is None else [self._champion_slot]
        return out + (self._layers[3][1] if self._layers else [])

    def _draw_trails(self, p: QPainter) -> None:
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in self._highlight_slots():
            a = self._agents[i]
            k = int(self._k[i])
            if k < 1 or not a.trail:
                continue
            color = a.color if a.color is not None else COLOR_CHAMPION
            p.setPen(_pen(_with_alpha(color, ALPHA_TRAIL), 2.0))
            p.drawPolyline(QPolygonF(a.trail[:k + 1]))

    def _draw_flames(self, p: QPainter, mats: list[list[float]]) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(COLOR_FLAME))
        for i in self._highlight_slots():
            a = self._agents[i]
            if not a.thrusters or self._ended[i]:
                continue
            out = a.traj.outputs[int(self._k[i])]
            off = int(self._offsets[i])
            for t in a.thrusters:
                u = float(out[t.channel])
                if u <= 0.0:
                    continue
                p.setTransform(QTransform(*mats[off + t.body]))
                p.translate(*t.at)
                p.rotate(t.degrees)
                length = 0.3 + 1.7 * min(u, 1.0) * t.strength
                p.drawPolygon(QPolygonF([QPointF(0.0, -0.18), QPointF(length, 0.0), QPointF(0.0, 0.18)]))

    def _draw_labels(self, p: QPainter, mats: list[list[float]]) -> None:
        items = [i for i, a in enumerate(self._agents) if a.label]
        if not items:
            return
        p.setFont(self._font)
        s = self._scale
        for i in items:
            a = self._agents[i]
            if not a.shapes:
                continue
            m = mats[int(self._offsets[i])]
            r = a.shapes[0].radius * s
            color = a.color if a.color is not None else (COLOR_CHAMPION if a.layer == "champion" else COLOR_HUD)
            p.setPen(color)
            p.drawText(QPointF(m[4] + r + 4.0, m[5] - r - 4.0), a.label)

    def _legend_entries(self) -> list[tuple[QColor, str]]:
        out: list[tuple[QColor, str]] = []
        if self._champion_slot is not None:
            out.append((COLOR_CHAMPION, S.SPEC_LEGEND_CHAMPION))
        if self._has_top:
            out.append((COLOR_AGENT, S.SPEC_LEGEND_TOP))
        if self._has_rest:
            out.append((_with_alpha(COLOR_AGENT, 110), S.SPEC_LEGEND_REST))
        for i in self._layers[3][1] if self._layers else []:
            a = self._agents[i]
            out.append((a.color or COLOR_HUD, a.label))
        return out

    def _draw_legend(self, p: QPainter) -> None:
        entries = [(c, t) for c, t in self._legend_entries() if t]
        if not entries:
            return
        fm = self._font_metrics
        p.setFont(self._font)
        line = fm.height() + 2.0
        sw = 10.0
        width = max(fm.horizontalAdvance(t) for _, t in entries) + sw + 22.0
        box = QRectF(self.width() - width - 10.0, 10.0, width, line * len(entries) + 10.0)
        p.setPen(_pen(COLOR_HUD_EDGE, 1.0))
        p.setBrush(QBrush(COLOR_HUD_BOX))
        p.drawRoundedRect(box, 4.0, 4.0)
        y = box.top() + 5.0
        for color, text in entries:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(color))
            p.drawRoundedRect(QRectF(box.left() + 8.0, y + (line - sw) / 2.0, sw, sw), 2.0, 2.0)
            p.setPen(COLOR_HUD)
            p.drawText(QRectF(box.left() + 14.0 + sw, y, width, line), Qt.AlignmentFlag.AlignVCenter, text)
            y += line

    def _draw_hud(self, p: QPainter) -> None:
        if not self.hud:
            return
        p.setFont(self._font)
        fm = self._font_metrics
        rect = fm.boundingRect(QRectF(0, 0, 10000, 10000), int(Qt.AlignmentFlag.AlignLeft), self.hud)
        box = QRectF(10.0, 10.0, rect.width() + 16.0, rect.height() + 10.0)
        p.setPen(_pen(COLOR_HUD_EDGE, 1.0))
        p.setBrush(QBrush(COLOR_HUD_BOX))
        p.drawRoundedRect(box, 4.0, 4.0)
        p.setPen(COLOR_HUD)
        p.drawText(box.adjusted(8.0, 5.0, -8.0, -5.0), int(Qt.AlignmentFlag.AlignLeft), self.hud)

    # ------------------------------------------------------------------
    # 입력
    # ------------------------------------------------------------------

    def hit_test(self, pos: QPointF) -> tuple[str, int] | None:
        """화면 점에서 맨 위에 그려진 개체: ("population", 집단 번호) 또는 ("overlay", 겹쳐 보기 번호)."""
        if not self._agents:
            return None
        wx, wy = self.screen_to_world(pos)
        tol = PICK_PX / max(self._scale, 1e-9)
        P = self._poses
        near = np.hypot(P[:, 0] - wx, P[:, 1] - wy) <= self._radii + tol
        if not near.any():
            return None
        offsets = self._offsets
        for _name, _ended, slots in reversed(self._layer_runs()):
            for i in reversed(slots):
                a = self._agents[i]
                for j in reversed(range(len(a.shapes))):
                    f = int(offsets[i]) + j
                    if not near[f]:
                        continue
                    x, y, ang = P[f]
                    c, s = math.cos(ang), math.sin(ang)
                    dx, dy = wx - x, wy - y
                    bx, by = c * dx + s * dy, -s * dx + c * dy
                    if a.shapes[j].pick.intersects(QRectF(bx - tol, by - tol, 2.0 * tol, 2.0 * tol)):
                        return ("overlay" if a.layer == "overlay" else "population"), a.index
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            hit = self.hit_test(event.position())
            if hit is not None:
                kind, index = hit
                (self.overlay_clicked if kind == "overlay" else self.individual_clicked).emit(index)
                event.accept()
                return
        super().mousePressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._wheel_needs_ctrl and not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            event.ignore()  # 대시보드 스크롤이 받게
            return
        dy = event.angleDelta().y()
        if not dy:
            return
        factor = 1.15 if dy > 0 else 1 / 1.15
        zoom = min(ZOOM_RANGE[1], max(ZOOM_RANGE[0], self._zoom * factor))
        factor = zoom / self._zoom
        self._zoom = zoom
        self._scale = min(400.0, max(0.5, self._scale * factor))
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._static_key = None
        self._update_camera(snap=True)

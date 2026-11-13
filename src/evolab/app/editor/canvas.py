"""에디터 캔버스: 편집 중인 환경을 시험이 시작할 때의 모습으로 그리고, 도구로 부품을 만들고 고친다.

- 미리보기는 런타임과 같은 코드로 만든다: 문서를 컴파일한 EnvSpec으로 전시 시험(display_trial, 트랙 세트가
  있으면 트랙 0)의 Episode를 만들어, 수식과 생성기가 학습 때와 똑같이 계산된 자세·모양을 그린다.
- 검사 오류가 있으면(doc.spec()이 None) 오류가 난 부품을 빼고 다시 컴파일해 본 것으로 그리고, 뺀 부품은 숫자만으로
  (수식은 계산할 수 있으면 계산하고, 못 하면 0) 그린다. 반쯤 만든 파일도 보이게 하려는 것이다.
- 문서가 바뀌면 미리보기를 다시 만든다(그리기는 100ms 모아서, 도구가 자세를 물으면 바로).
- 카메라는 y가 위로. 휠로 커서 둘레 확대·축소, 가운데 단추나 Space + 끌기로 이동, fit()으로 전체 보기.
- 격자 맞춤(snap_enabled, snap_step 기본 0.5m). Alt를 누르고 끌면 맞추지 않는다.
- (v2) 정해진 움직임 물체(kinematic 세계 물체, 장애물 줄)는 시작 자세에 길 점선을 함께 그린다. 미리보기 시각
  (preview_time, 아래 PreviewBar의 슬라이더·▶)을 옮기면 그 시각의 자세(motion.pose_at)로 그린다. 재생 중에는 움직이지
  않는 그림을 이미지로 한 번 그려 두고 움직이는 물체만 덧그린다. 문서가 바뀌면 멈추고 0초로 돌아간다.
- (v2) 기본 보기(editor.view, 해시 밖): 「전체 보기」는 기본 보기가 있으면 그 상자로, 한 번 더 누르면 장면 전체로.
- (v2.1) 없어질 수 있는 물체(spec.structure.optional_bodies)는 점선 테두리와 「?」 표지. 처음 몸에 없는 물체(Episode.absent)와
  거기 붙은 부품은 「없는 부품도 보기」(show_absent)가 켜져 있으면 옅게, 꺼져 있으면 그리지도 누르지도 않는다(꺼진 채 캔버스
  밖에서 숨은 부품을 고르면, 예를 들어 구조 탭 줄, 보기를 다시 켠다: 고른 것은 늘 보인다). 구조 탭에서 고른 줄의 물체와
  함께 사라지는 부품은 점선 강조(set_structure_highlight). 물체 오른쪽 메뉴에 「배열로 늘리기…」(할 수 없으면 흐리게, 이유는
  풍선 도움말).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from PySide6.QtCore import QElapsedTimer, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QContextMenuEvent,
    QFont,
    QFontMetricsF,
    QImage,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPolygonF,
    QResizeEvent,
    QShowEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QMenu, QSizePolicy, QSlider, QToolButton, QWidget

from evolab.app import strings as S
from evolab.app.editor import tools as T
from evolab.app.editor.document import EnvDocument, Ref, where_to_ref
from evolab.app.editor.structure import attached_parts, initial_absent
from evolab.app.play.window import (
    COLOR_AGENT,
    COLOR_AGENT_EDGE,
    COLOR_BG,
    COLOR_FLAME,
    COLOR_HUD,
    COLOR_LEG,
    COLOR_PAD,
    COLOR_TERRAIN_FILL,
    COLOR_TERRAIN_LINE,
    COLOR_WORLD_BODY,
)
from evolab.spec.editor_meta import editor_view_box, with_view_box
from evolab.world.physics2d import geometry as geo
from evolab.world.physics2d.episode import ShapeGeom
from evolab.world.physics2d.motion import MotionPlan, pose_at

Point = tuple[float, float]
Pose = tuple[float, float, float]

PREVIEW_DEBOUNCE_MS = 100
MESSAGE_MS = 5000
PICK_PX = 6.0
HANDLE_PX = 8.0
ROTATE_HANDLE_PX = 26.0
SPAWN_ARROW_PX = 32.0
SCALE_RANGE = (0.5, 2000.0)
BADGE_MIN_BODY_PX = 16.0
"""화면에서 이보다 작은 물체에는 센서·구동기 표지를 그리지 않는다(전체 보기에서 겹치지 않게)."""
DEFAULT_SNAP_STEP = 0.5
PREVIEW_FPS = 30
"""움직임 미리보기 재생 중 다시 그리는 횟수(초당)."""
PREVIEW_STEP = 0.01
"""미리보기 시각 슬라이더 한 칸(초)."""
ROW_WINDOW_S = 15.0
"""장애물 줄은 이 시간(초) 안에 선수 자리(x = 0)에 닿을 장애물만 그린다(나머지는 「→ 장애물 n개」)."""
DEFAULT_MAX_TIME = 20.0
MODE_SYMBOLS = {"once": "\u2192", "loop": "\u21bb", "pingpong": "\u2194"}
OPTIONAL_BADGE = "?"
"""없어질 수 있는 물체 표지(v2.1 설계 5.1)."""
ABSENT_OPACITY = 0.35
"""처음 몸에 없는 부품을 그리는 불투명도(「없는 부품도 보기」)."""
"""길 방식 기호: 한 번(→), 돌기(↻), 왕복(↔)."""

COLOR_GRID = QColor(255, 255, 255, 12)
COLOR_GRID_MAJOR = QColor(255, 255, 255, 28)
COLOR_AXIS = QColor(255, 255, 255, 55)
COLOR_BOUNDS = QColor(255, 255, 255, 70)
COLOR_STATIC = QColor("#59647a")
COLOR_SELECT = QColor("#ffd166")
COLOR_ERROR = QColor("#ff5c5c")
COLOR_WARNING = QColor("#ffa94d")
COLOR_JOINT = QColor("#88c0d0")
COLOR_RAY = QColor(136, 192, 208, 110)
COLOR_TARGET = QColor(180, 142, 173, 190)
COLOR_SPAWN = QColor("#b48ead")
COLOR_CHECKPOINT = QColor(255, 209, 102, 150)
COLOR_CHECKPOINT_GEN = QColor(255, 209, 102, 70)
COLOR_START_LINE = QColor(255, 255, 255, 200)
COLOR_BADGE_SENSOR = QColor(94, 129, 172, 230)
COLOR_BADGE_ACTUATOR = QColor(208, 135, 112, 230)
COLOR_BADGE_TEXT = QColor("#eceff4")
COLOR_LABEL = QColor(229, 233, 240, 150)
COLOR_PATH = QColor(255, 209, 102, 120)
"""움직임 길 점선: 선택 색의 옅은 판(설계 5.7)."""
COLOR_GHOST = QColor(255, 255, 255, 40)
COLOR_WIND = QColor(136, 192, 208, 150)
REGION_FILL = {
    "goal": QColor(61, 220, 132, 45),
    "pad": QColor(61, 220, 132, 45),
    "hazard": QColor(255, 99, 99, 50),
    "zone": QColor(255, 255, 255, 18),
}
REGION_EDGE = {
    "goal": QColor(61, 220, 132, 170),
    "pad": QColor(61, 220, 132, 170),
    "hazard": QColor(255, 99, 99, 170),
    "zone": QColor(255, 255, 255, 80),
}

BADGE_TEXT = {
    "rays": S.EDC_BADGE_RAYS,
    "body_state": S.EDC_BADGE_BODY_STATE,
    "contact": S.EDC_BADGE_CONTACT,
    "target": S.EDC_BADGE_TARGET,
    "joint_angle": S.EDC_BADGE_JOINT_ANGLE,
    "fuel": S.EDC_BADGE_FUEL,
    "time_left": S.EDC_BADGE_TIME_LEFT,
    "fuel_tank": S.EDC_BADGE_TANK,
    "vehicle": S.EDC_BADGE_VEHICLE,
    "motor": S.EDC_BADGE_MOTOR,
    "jump": S.EDC_LABEL_JUMP,
    "aero": S.EDC_TOOL_AERO,
    "move": S.EDC_TOOL_MOVE,
    "balance": S.EDP_VALUE_NAMES["balance"],
}
ACTUATOR_BADGES = frozenset({"fuel_tank", "vehicle", "motor", "jump", "aero", "move", "balance"})
SHAPE_TOOLS = frozenset({"box", "circle", "poly"})
"""새 물체를 그리는 도구(커서 옆에 소유 글자, 6.2.0)."""
COLOR_MOVE = QColor(255, 184, 107, 220)
"""밀기(이동) 표지의 화살표(6.2.0)."""
COLOR_BALANCE = QColor(163, 190, 140, 220)
"""균형 보조 표지(6.2.0): 세울 각도의 선과 양쪽 둥근 화살표."""
MARK_PX = 14.0
"""밀기·균형 보조 표지의 화면 크기(px, 확대와 무관)."""


# ---------------------------------------------------------------------------
# 장면: 그릴 것과 누를 것
# ---------------------------------------------------------------------------


def _bbox(points: Iterable[Point]) -> tuple[float, float, float, float]:
    pts = list(points)
    if not pts:
        return (0.0, 0.0, 0.0, 0.0)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class BodyItem:
    id: str
    ref: Ref
    owner: str
    dynamic: bool
    tags: list[str]
    shapes: list[ShapeGeom]
    """물체 좌표"""
    pose: Pose
    """세계 좌표의 시작 자세"""
    generated: bool = False
    place_by: str | None = None
    placeholder: bool = False
    """모양을 계산할 수 없어 자리 표시만 그린다."""
    kinematic: bool = False
    """정해진 움직임 물체(v2): 세계 물체 type kinematic, 또는 장애물 줄."""
    motion: MotionPlan | None = None
    """정해진 움직임. 시각 t의 자세 = pose_at(motion, t)."""
    path_count: int = 0
    """motion이 길이면 파일의 점 수(끌 손잡이 수. loop의 닫는 구간은 세지 않는다)."""
    row: list[ShapeGeom] | None = None
    """장애물 줄이면 모든 장애물 모양(shapes는 처음 ROW_WINDOW_S초 안에 닿는 것만)."""
    row_hidden: int = 0
    """장애물 줄에서 shapes에 안 넣은(나중에 오는) 장애물 수."""
    wing_les: list[Point] = field(default_factory=list)
    """날개 모양의 앞전(물체 좌표)."""
    optional: bool = False
    """진화로 없어질 수 있는 물체(v2.1, spec.structure.optional_bodies)."""
    absent: bool = False
    """처음 몸(미리보기)에 없는 물체(v2.1, Episode.absent). 모양은 숫자로 계산해 그린다."""
    polys: list[list[Point]] = field(default_factory=list)
    circles: list[tuple[Point, float]] = field(default_factory=list)
    segments: list[tuple[Point, Point, float]] = field(default_factory=list)
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def finish(self) -> BodyItem:
        if self.row is not None and self.motion is not None:
            self.shapes, self.row_hidden = row_window(self.row, self.motion, 0.0)
        self.polys, self.circles, self.segments = shapes_at(self.shapes, self.pose)
        x, y, _a = self.pose
        pts: list[Point] = [(x, y)]
        for poly in self.polys:
            pts += poly
        for c, r in self.circles:
            pts += [(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)]
        for p0, p1, _r in self.segments:
            pts += [p0, p1]
        self.bbox = _bbox(pts)
        return self

    def path_points(self) -> list[Point]:
        """길 움직임의 파일 점들(세계 좌표). 길이 아니면 빈 목록."""
        mo = self.motion
        if mo is None or mo.kind != "path":
            return []
        ox, oy = mo.origin
        return [(ox + x, oy + y) for x, y in mo.points[: self.path_count or len(mo.points)]]

    def contains(self, p: Point, tol: float) -> bool:
        x0, y0, x1, y1 = self.bbox
        if not (x0 - tol <= p[0] <= x1 + tol and y0 - tol <= p[1] <= y1 + tol):
            return False
        for poly in self.polys:
            if geo.point_in_polygon(p, poly):
                return True
            n = len(poly)
            if any(geo.dist_point_segment(p, poly[i], poly[(i + 1) % n]) <= tol for i in range(n)):
                return True
        for c, r in self.circles:
            if math.dist(p, c) <= r + tol:
                return True
        for a, b, r in self.segments:
            if geo.dist_point_segment(p, a, b) <= r + tol:
                return True
        if self.placeholder:
            return math.dist(p, (self.pose[0], self.pose[1])) <= 0.3 + tol
        return False


def shapes_at(shapes: Iterable[ShapeGeom], pose: Pose) -> tuple[list[list[Point]], list[tuple[Point, float]],
                                                               list[tuple[Point, Point, float]]]:
    """물체 좌표 모양들을 자세 pose에 놓은 세계 좌표 (다각형들, 원들, 선분들)."""
    x, y, a = pose
    polys: list[list[Point]] = []
    circles: list[tuple[Point, float]] = []
    segments: list[tuple[Point, Point, float]] = []
    for s in shapes:
        if s.kind == "poly":
            polys.append([geo.transform(p, (x, y), a) for p in s.points])
        elif s.kind == "circle":
            circles.append((geo.transform(s.points[0], (x, y), a), s.radius))
        elif s.kind == "segment":
            segments.append((geo.transform(s.points[0], (x, y), a), geo.transform(s.points[1], (x, y), a), s.radius))
    return polys, circles, segments


def row_window(row: list[ShapeGeom], plan: MotionPlan, t: float) -> tuple[list[ShapeGeom], int]:
    """장애물 줄에서 시각 t의 앞뒤 ROW_WINDOW_S초 안에 선수 자리(세계 x = 0, 생성기 좌표의 0)를 지나는 장애물만.

    줄이 미는 방향 u에 대해 장애물의 "남은 거리" = min(시각 t의 세계 꼭짓점) · (-u)다(0이면 선수 자리에 닿음).
    [-(지난 ROW_WINDOW_S초 동안 간 거리), 다음 ROW_WINDOW_S초 동안 갈 거리] 안의 장애물을 고르고,
    그보다 먼 장애물 수를 함께 돌려준다. 하나도 없으면 가장 가까운 것 하나를 넣는다(멈춘 줄도 보이게).
    """
    if not row:
        return [], 0
    xt, yt, _ = pose_at(plan, t)
    xa, ya, _ = pose_at(plan, t + ROW_WINDOW_S)
    xb, yb, _ = pose_at(plan, max(0.0, t - ROW_WINDOW_S))
    if plan.kind == "path" and len(plan.points) >= 2:
        (px, py), (qx, qy) = plan.points[0], plan.points[-1]
        n = math.hypot(qx - px, qy - py)
        back = ((px - qx) / n, (py - qy) / n) if n > 1e-12 else (1.0, 0.0)
    else:
        back = (1.0, 0.0)
    ahead = math.hypot(xa - xt, ya - yt)
    behind = math.hypot(xt - xb, yt - yb)
    keep: list[ShapeGeom] = []
    later = 0
    nearest: tuple[float, ShapeGeom] | None = None
    for s in row:
        d = min((p[0] + xt) * back[0] + (p[1] + yt) * back[1] for p in s.points)
        if -behind - 1e-9 <= d <= ahead + 1e-9:
            keep.append(s)
        elif d > ahead:
            later += 1
            if nearest is None or d < nearest[0]:
                nearest = (d, s)
    if not keep and nearest is not None:
        keep.append(nearest[1])
        later -= 1
    return keep, later


@dataclass
class RegionItem:
    id: str
    ref: Ref
    kind: str
    points: list[Point]
    generated: bool = False
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        self.bbox = _bbox(self.points)


@dataclass
class CheckpointItem:
    ref: Ref
    set_id: str
    index: int
    a: Point
    b: Point
    editable: bool


@dataclass
class JointItem:
    ref: Ref
    kind: str
    a: str
    b: str
    pa: Point
    pb: Point
    placed: bool


@dataclass
class ThrusterItem:
    ref: Ref
    body: str
    at: Point
    dir: float
    max_force: float


@dataclass
class BadgeItem:
    ref: Ref
    kind: str
    body: str | None
    joint: str | None = None


@dataclass
class RayItem:
    ref: Ref
    body: str
    origin: Point
    lines: list[tuple[Point, Point, float]]


@dataclass
class TargetItem:
    ref: Ref
    body: str
    point: Point


@dataclass
class JumpItem:
    ref: Ref
    body: str
    dir: float
    """뛰는 방향(세계 좌표, 라디안)."""


@dataclass
class MoveItem:
    """밀기(이동) 구동기 표지(6.2.0): 물체 가운데의 화살표(xy 넷, x 좌우 둘, heading 앞 하나와 도는 호)."""

    ref: Ref
    body: str
    mode: str


@dataclass
class BalanceItem:
    """균형 보조 표지(6.2.0): 물체 가운데에서 세울 각도(target, 세계 각도) 쪽으로 선 + 양쪽 둥근 화살표."""

    ref: Ref
    body: str
    target: float


@dataclass
class Scene:
    bounds: tuple[float, float, float, float] = (-50.0, -20.0, 50.0, 60.0)
    bodies: list[BodyItem] = field(default_factory=list)
    regions: list[RegionItem] = field(default_factory=list)
    checkpoints: list[CheckpointItem] = field(default_factory=list)
    joints: list[JointItem] = field(default_factory=list)
    thrusters: list[ThrusterItem] = field(default_factory=list)
    badges: list[BadgeItem] = field(default_factory=list)
    rays: list[RayItem] = field(default_factory=list)
    targets: list[TargetItem] = field(default_factory=list)
    jumps: list[JumpItem] = field(default_factory=list)
    moves: list[MoveItem] = field(default_factory=list)
    balances: list[BalanceItem] = field(default_factory=list)
    aero_bodies: set[str] = field(default_factory=set)
    """공기역학이 달린 물체 id."""
    max_time: float = DEFAULT_MAX_TIME
    """세계 최대 시간(미리보기 슬라이더 끝)."""
    spawn: Pose = (0.0, 0.0, 0.0)
    exact: bool = False
    """문서 전체가 검사를 통과해 런타임 그대로 그렸다."""
    message: str = ""
    optional: frozenset[str] = frozenset()
    """진화로 없어질 수 있는 물체 id(v2.1)."""
    absent: frozenset[str] = frozenset()
    """처음 몸에 없는 물체 id(v2.1)."""
    absent_parts: frozenset[Ref] = frozenset()
    """처음 몸에 없는 물체에 붙어 함께 없는 부품(관절·모터·구동기·센서, v2.1)."""

    def body(self, body_id: str) -> BodyItem | None:
        for b in self.bodies:
            if b.id == body_id:
                return b
        return None

    def region(self, region_id: str) -> RegionItem | None:
        for r in self.regions:
            if r.id == region_id:
                return r
        return None

    def moving(self) -> list[BodyItem]:
        """정해진 움직임이 있는 물체들(그리는 순서)."""
        return [b for b in self.bodies if b.motion is not None]


# ---------------------------------------------------------------------------
# 미리보기 만들기
# ---------------------------------------------------------------------------

_REMOVABLE = ("generators", "bodies", "joints", "actuators", "sensors", "regions", "checkpoints", "randomize",
              "params", "events", "termination", "scoring.blocks")


def _repaired_spec(data: dict[str, Any]) -> Any:
    """오류가 난 부품을 빼 가며 다시 컴파일한다(미리보기 전용 사본). 점수·사건·뇌는 모양과 상관없어 처음부터 뺀다."""
    from evolab.spec.compiled import EnvFileError
    from evolab.spec.io import compile_env

    d = copy.deepcopy(data)
    d["events"] = []
    d["termination"] = []
    d["scoring"] = {"blocks": []}
    d.pop("brain", None)
    for _ in range(10):
        try:
            return compile_env(copy.deepcopy(d))
        except EnvFileError as exc:
            errors = [i for i in exc.issues if i.severity == "error"]
        except Exception:  # noqa: BLE001 (미리보기는 어떤 입력에도 죽지 않는다)
            return None
        doomed: dict[str, set[int]] = {}
        for issue in errors:
            for section in sorted(_REMOVABLE, key=len, reverse=True):
                prefix = section + "["
                if issue.where.startswith(prefix):
                    try:
                        index = int(issue.where[len(prefix):].split("]", 1)[0])
                    except ValueError:
                        continue
                    doomed.setdefault(section, set()).add(index)
                    break
        if not doomed:
            return None
        for section, indexes in doomed.items():
            keys = section.split(".")
            cur: Any = d
            try:
                for k in keys:
                    cur = cur[k]
            except (KeyError, TypeError):
                continue
            if isinstance(cur, list):
                cur[:] = [x for i, x in enumerate(cur) if i not in indexes]
    return None


def preview_trial(index: int = 0) -> Any:
    """미리보기 시험 index번: 검증 세트의 index번 시험(0번 = 전시 시험, 실험실 재생과 같은 랜덤 값)."""
    from evolab.engine import seeds

    return seeds.fixed_trials(seeds.TAG_VALIDATION, index + 1)[index]


def _preview_episode(spec: Any, trial_index: int = 0) -> Any:
    """편집기 미리보기 시험(document.editor_checks와 같은 조건). trial_index는 「미리보기 시험 바꾸기」(v2 설계 5.1)."""
    from evolab.policy.genome import layout_for
    from evolab.world.physics2d.episode import Episode, Trial

    layout = layout_for(spec)
    body = layout.body_values(layout.initial()[None, :])[0] if layout.n_body else None
    trial = preview_trial(trial_index)
    if spec.track_set_size:
        trial = Trial(trial.seed, trial_index % spec.track_set_size)
    return Episode(spec, body, trial)


class _Eval:
    """숫자 칸 계산: 숫자는 그대로, 수식은 알고 있는 값(p.*, r.*, gen.*)으로 계산, 못 하면 기본값."""

    def __init__(self, values: dict[str, float]) -> None:
        self.values = dict(values)
        self._cache: dict[str, Any] = {}

    def set(self, name: str, value: float) -> None:
        self.values[name] = float(value)
        self._cache.clear()

    def opt(self, v: Any) -> float | None:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            x = float(v)
            return x if math.isfinite(x) else None
        if isinstance(v, str) and v.startswith("="):
            fn = self._cache.get(v, False)
            if fn is False:
                fn = self._compile(v)
                self._cache[v] = fn
            if fn is None:
                return None
            try:
                x = float(fn(self.values))
            except Exception:  # noqa: BLE001
                return None
            return x if math.isfinite(x) else None
        return None

    def _compile(self, v: str) -> Any:
        from evolab.scoring.expr import Sym, compile_expr, formula_body

        try:
            symbols = {k: Sym("num", "build") for k in self.values}
            return compile_expr(formula_body(v), symbols, want="num", max_phase="build")
        except Exception:  # noqa: BLE001
            return None

    def num(self, v: Any, default: float = 0.0) -> float:
        x = self.opt(v)
        return default if x is None else x

    def vec(self, v: Any, default: Point = (0.0, 0.0)) -> Point:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (self.num(v[0], default[0]), self.num(v[1], default[1]))
        return default


def _items(data: dict[str, Any], section: str) -> list[dict[str, Any]]:
    v = data.get(section)
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


class _SceneBuilder:
    def __init__(self, data: dict[str, Any], ep: Any, exact: bool, message: str) -> None:
        self.data = data
        self.ep = ep
        self.scene = Scene(exact=exact, message=message)
        self.ev = _Eval(dict(ep._build_env["agent"]) if ep is not None else self._default_values())
        self.ep_gen_ids = {g.id for g in ep.env.generators} if ep is not None else set()
        self.ep_body_ids = {b.id for b in ep.env.bodies} if ep is not None else set()

    def _default_values(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in _items(self.data, "params"):
            if isinstance(p.get("id"), str) and T.is_number(p.get("value")):
                out[f"p.{p['id']}"] = float(p["value"])
        for r in _items(self.data, "randomize"):
            if not isinstance(r.get("id"), str):
                continue
            lo, hi, mean, vals = r.get("low"), r.get("high"), r.get("mean"), r.get("values")
            if T.is_number(lo) and T.is_number(hi):
                x = (float(lo) + float(hi)) / 2.0
                out[f"r.{r['id']}"] = float(math.floor(x)) if r.get("dist") == "int" else x
            elif T.is_number(mean):
                out[f"r.{r['id']}"] = float(mean)
            elif isinstance(vals, list) and vals and T.is_number(vals[0]):
                out[f"r.{r['id']}"] = float(vals[0])
            else:
                out[f"r.{r['id']}"] = 0.0
        return out

    def build(self) -> Scene:
        steps = (self._world_info, self._episode_items, self._generators, self._spawn, self._bodies, self._regions,
                 self._checkpoints, self._joints, self._actuators, self._sensors)
        for step in steps:
            try:
                step()
            except Exception:  # noqa: BLE001 (반쯤 만든 파일에서도 나머지는 그린다)
                continue
        return self.scene

    def _world_info(self) -> None:
        world = self.data.get("world") if isinstance(self.data.get("world"), dict) else {}
        t = self.ev.opt(world.get("max_time", DEFAULT_MAX_TIME))
        self.scene.max_time = t if t is not None and t > 0 else DEFAULT_MAX_TIME

    # ---- 움직임과 날개 (v2) ----
    def _motion(self, b: dict[str, Any]) -> MotionPlan | None:
        """정해진 움직임 세계 물체의 움직임(숫자만으로: 수식은 아는 값으로 계산). 런타임(Episode)과 같은 규칙."""
        mo = b.get("motion")
        if b.get("type") != "kinematic" or b.get("owner", "agent") != "world" or not isinstance(mo, dict):
            return None
        ev = self.ev
        pos, angle = ev.vec(b.get("pos")), ev.num(b.get("angle", 0.0))
        try:
            if mo.get("kind") == "path":
                pts = [ev.vec(q) for q in mo.get("points") or []]
                if not pts:
                    return None
                mode = mo.get("mode", "once")
                return MotionPlan.path(pos, angle, pts, ev.num(mo.get("speed", 1.0), 1.0), ev.num(mo.get("accel", 0.0)),
                                       ev.num(mo.get("start", 0.0)), mode if mode in MODE_SYMBOLS else "once",
                                       ev.num(mo.get("spin", 0.0)))
            if mo.get("kind") == "wave":
                return MotionPlan.wave(pos, angle, ev.vec(mo.get("amplitude", [0.0, 1.0]), (0.0, 1.0)),
                                       ev.num(mo.get("period", 2.0), 2.0), ev.num(mo.get("phase", 0.0)),
                                       ev.num(mo.get("spin", 0.0)))
        except (TypeError, ValueError):
            return None
        return None

    @staticmethod
    def _path_count(b: dict[str, Any]) -> int:
        mo = b.get("motion")
        if isinstance(mo, dict) and mo.get("kind") == "path" and isinstance(mo.get("points"), list):
            return len(mo["points"])
        return 0

    def _wing_les(self, b: dict[str, Any]) -> list[Point]:
        return [self.ev.vec(s.get("le", [0.0, 0.0])) for s in b.get("shapes") or []
                if isinstance(s, dict) and s.get("kind") == "wing"]

    def _decorate(self, item: BodyItem, b: dict[str, Any], motion: MotionPlan | None) -> BodyItem:
        """문서 물체의 v2 정보(정해진 움직임, 날개 앞전)를 붙이고 마무리한다."""
        item.kinematic = b.get("type") == "kinematic"
        item.motion = motion if motion is not None else self._motion(b)
        item.path_count = self._path_count(b) if item.motion is not None else 0
        item.wing_les = self._wing_les(b)
        return item.finish()

    # ---- 런타임 미리보기에 있는 것 ----
    def _episode_items(self) -> None:
        ep = self.ep
        if ep is None:
            b = self.data.get("world", {}).get("bounds") if isinstance(self.data.get("world"), dict) else None
            if isinstance(b, list) and len(b) == 4:
                self.scene.bounds = tuple(self.ev.num(v) for v in b)  # type: ignore[assignment]
            return
        g = ep.geometry()
        self.scene.bounds = tuple(float(v) for v in g.bounds)  # type: ignore[assignment]
        doc_bodies = {str(b.get("id")): b for b in _items(self.data, "bodies")}
        gens = [x.id for x in ep.env.generators]
        gen_kinds = {str(x.get("id")): x.get("kind") for x in _items(self.data, "generators")}
        for bg in g.bodies:
            if bg.id in doc_bodies:
                ref: Ref = ("bodies", bg.id)
                generated = False
            else:
                gid = next((x for x in gens if bg.id in (x, f"{x}.walls")), None)
                if gid is None:
                    continue
                ref, generated = ("generators", gid), True
            place_by = doc_bodies.get(bg.id, {}).get("place_by")
            item = BodyItem(bg.id, ref, bg.owner, bg.moving, list(bg.tags), list(bg.shapes),
                            tuple(float(v) for v in bg.pose), generated,  # type: ignore[arg-type]
                            place_by if isinstance(place_by, str) else None)
            motion = bg.motion  # v2: 런타임이 계산한 정해진 움직임(없으면 None)
            if not generated:
                self.scene.bodies.append(self._decorate(item, doc_bodies[bg.id], motion))
                continue
            if motion is not None:
                item.kinematic, item.motion = True, motion
                if gen_kinds.get(ref[1]) == "obstacles":
                    item.row = list(bg.shapes)
            self.scene.bodies.append(item.finish())
        doc_regions = {str(r.get("id")) for r in _items(self.data, "regions")}
        doc_cps = {str(c.get("id")) for c in _items(self.data, "checkpoints")}
        for rg in g.regions:
            pts = [(float(x), float(y)) for x, y in rg.points]
            if rg.kind == "checkpoint":
                set_id, _, k = rg.id.rpartition(".")
                if set_id in doc_cps:
                    self.scene.checkpoints.append(CheckpointItem(("checkpoints", set_id), set_id, int(k), pts[0],
                                                                 pts[1], True))
                else:
                    gid = next((x for x in gens if set_id == f"{x}.cp"), None)
                    if gid is not None:
                        self.scene.checkpoints.append(CheckpointItem(("generators", gid), set_id, int(k), pts[0],
                                                                     pts[1], False))
                continue
            if rg.id in doc_regions:
                self.scene.regions.append(RegionItem(rg.id, ("regions", rg.id), rg.kind, pts))
            else:
                gid = next((x for x in gens if rg.id == f"{x}.pad"), None)
                if gid is not None:
                    self.scene.regions.append(RegionItem(rg.id, ("generators", gid), rg.kind, pts, True))

    # ---- 숫자만으로 ----
    def _generators(self) -> None:
        for g in _items(self.data, "generators"):
            gid = g.get("id")
            if not isinstance(gid, str) or gid in self.ep_gen_ids:
                continue
            try:
                if g.get("kind") == "heightmap":
                    self._heightmap(gid, g)
                elif g.get("kind") == "track":
                    self._track(gid, g)
                elif g.get("kind") == "hills":
                    self._hills(gid, g)
                elif g.get("kind") == "obstacles":
                    self._obstacles(gid, g)
            except Exception:  # noqa: BLE001
                continue

    def _obstacles(self, gid: str, g: dict[str, Any]) -> None:
        """장애물 줄(숫자만으로, 런타임과 같은 생성기). 검사 오류로 런타임 미리보기에서 빠졌을 때만 쓴다."""
        from evolab.generators.obstacles import generate_obstacles, place_boxes

        ev = self.ev

        def pair(key: str, default: tuple[float, float]) -> tuple[float, float]:
            v = g.get(key)
            if isinstance(v, list) and len(v) == 2:
                return (ev.num(v[0], default[0]), ev.num(v[1], default[1]))
            return default

        speed, accel = max(0.0, ev.num(g.get("speed", 6.0), 6.0)), ev.num(g.get("accel", 0.15), 0.15)
        count = g.get("count", 30)
        row = generate_obstacles(
            int(ev.num(g.get("seed", 0.0))) & 0x7FFFFFFFFFFFFFFF, count=int(count) if T.is_number(count) else 30,
            x=ev.num(g.get("x", 25.0), 25.0), y=ev.num(g.get("y", 0.0)), gap=pair("gap", (1.4, 3.0)),
            width=pair("width", (0.6, 1.4)), height=pair("height", (0.5, 1.2)),
            air_chance=ev.num(g.get("air_chance", 0.25), 0.25), air_y=pair("air_y", (1.3, 1.6)),
            air_height=pair("air_height", (0.5, 0.8)), speed=speed, accel=accel)
        d = ev.num(g.get("direction", math.pi), math.pi)
        shapes = [ShapeGeom("poly", [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                  for x0, y0, x1, y1 in place_boxes(row, d)]
        plan = MotionPlan.path((0.0, 0.0), 0.0, [(0.0, 0.0), (row.length * math.cos(d), row.length * math.sin(d))],
                               speed, accel, 0.0, "once", 0.0)
        self.scene.bodies.append(BodyItem(gid, ("generators", gid), "world", False, ["obstacle"], shapes,
                                          (0.0, 0.0, 0.0), True, kinematic=True, motion=plan, row=shapes).finish())
        for k, v in (("speed", speed), ("accel", accel), ("length", row.length)):
            self.ev.set(f"gen.{gid}.{k}", float(v))

    def _terrain_body(self, gid: str, xs: Any, ys: Any, radius: float) -> None:
        shapes = [ShapeGeom("segment", [(float(xs[i]), float(ys[i])), (float(xs[i + 1]), float(ys[i + 1]))], radius)
                  for i in range(len(xs) - 1)]
        self.scene.bodies.append(BodyItem(gid, ("generators", gid), "world", False, ["terrain"], shapes,
                                          (0.0, 0.0, 0.0), True).finish())
        self.ev.set(f"gen.{gid}.x_min", float(xs[0]))
        self.ev.set(f"gen.{gid}.x_max", float(xs[-1]))

    def _heightmap(self, gid: str, g: dict[str, Any]) -> None:
        from evolab.generators.terrain import generate_heightmap

        ev = self.ev
        xr = g.get("x_range", [-50.0, 50.0])
        pad = g.get("pad") if isinstance(g.get("pad"), dict) else None
        hm = generate_heightmap(
            int(ev.num(g.get("seed", 0.0))) & 0x7FFFFFFFFFFFFFFF, x_min=ev.num(xr[0], -50.0), x_max=ev.num(xr[1], 50.0),
            segment=float(g.get("segment", 1.0)), base_y=ev.num(g.get("base_y", 0.0)),
            amplitude=max(0.0, ev.num(g.get("amplitude", 4.0))),
            roughness=min(1.0, max(0.0, ev.num(g.get("roughness", 0.5)))),
            pad_x=ev.num(pad.get("x", 0.0)) if pad else None,
            pad_width=ev.num(pad.get("width", 8.0), 8.0) if pad else None,
        )
        self._terrain_body(gid, hm.xs, hm.ys, float(g.get("radius", 0.1)))
        if hm.pad is not None and pad is not None:
            left, right, y = hm.pad
            h = float(pad.get("region_height", 3.0))
            pts = [(left, y), (right, y), (right, y + h), (left, y + h)]
            self.scene.regions.append(RegionItem(f"{gid}.pad", ("generators", gid), "pad", pts, True))
            for k, v in (("x", (left + right) / 2.0), ("y", y), ("w", right - left), ("left", left), ("right", right)):
                self.ev.set(f"gen.{gid}.pad.{k}", v)

    def _hills(self, gid: str, g: dict[str, Any]) -> None:
        from evolab.generators import terrain

        ev = self.ev
        xr = g.get("x_range", [-20.0, 400.0])
        xs, ys = terrain.generate_hills(  # type: ignore[attr-defined]
            int(ev.num(g.get("seed", 0.0))) & 0x7FFFFFFFFFFFFFFF,
            x_min=ev.num(xr[0], -20.0), x_max=ev.num(xr[1], 400.0),
            segment=float(g.get("segment", 2.0)), base_y=ev.num(g.get("base_y", 0.0)),
            amplitude=max(0.0, ev.num(g.get("amplitude", 0.5))), growth=max(0.0, ev.num(g.get("growth", 0.01))),
            roughness=min(1.0, max(0.0, ev.num(g.get("roughness", 0.5)))), slope=ev.num(g.get("slope", 0.03)),
            flat_start=max(0.0, ev.num(g.get("flat_start", 15.0))),
        )
        self._terrain_body(gid, xs, ys, float(g.get("radius", 0.1)))

    def _track(self, gid: str, g: dict[str, Any]) -> None:
        from evolab.generators import track as tg
        from evolab.world.physics2d.episode import _cached_track

        ev = self.ev
        width = ev.num(g.get("width", 8.0), 8.0)
        checkpoints = int(g.get("checkpoints", 24))
        if g.get("source", "generated") == "drawn":
            pts = tuple(ev.vec(q) for q in g.get("centerline") or [])
            track = _cached_track(("drawn", pts, width, checkpoints),
                                  lambda: tg.track_from_centerline(pts, width=width, checkpoints=checkpoints))
        else:
            lr = g.get("length_range", [150.0, 250.0])
            params = {"segments": int(g.get("segments", 12)), "curvature": ev.num(g.get("curvature", 0.5), 0.5),
                      "width": width, "length_range": (float(lr[0]), float(lr[1])), "checkpoints": checkpoints}
            frozen = tuple(sorted(params.items()))
            set_seed, set_size = int(g.get("set_seed", 0)), int(g.get("set_size", 5))
            track = _cached_track(("set", set_seed, set_size, frozen),
                                  lambda: tg.track_set(set_seed, set_size, **params))[0]
        radius = float(g.get("wall_radius", 0.1))
        shapes: list[ShapeGeom] = []
        for poly in (track.inner, track.outer):
            pts2 = [(float(x), float(y)) for x, y in poly]
            for i in range(len(pts2)):
                shapes.append(ShapeGeom("segment", [pts2[i], pts2[(i + 1) % len(pts2)]], radius))
        self.scene.bodies.append(BodyItem(f"{gid}.walls", ("generators", gid), "world", False, ["wall"], shapes,
                                          (0.0, 0.0, 0.0), True).finish())
        for k, (a, b) in enumerate(track.checkpoints):
            self.scene.checkpoints.append(CheckpointItem(("generators", gid), f"{gid}.cp", k,
                                                         (float(a[0]), float(a[1])), (float(b[0]), float(b[1])), False))
        x, y, heading = track.start
        for k, v in (("start.x", x), ("start.y", y), ("start.angle", heading), ("length", track.length)):
            self.ev.set(f"gen.{gid}.{k}", float(v))

    def _spawn(self) -> None:
        agent = self.data.get("agent") if isinstance(self.data.get("agent"), dict) else {}
        sp = agent.get("spawn") if isinstance(agent.get("spawn"), dict) else {}
        x, y = self.ev.vec((sp.get("x", 0.0), sp.get("y", 0.0)))
        self.scene.spawn = (x, y, self.ev.num(sp.get("angle", 0.0)))

    def _shapes(self, b: dict[str, Any]) -> list[ShapeGeom]:
        ev = self.ev
        out: list[ShapeGeom] = []
        for s in b.get("shapes") or []:
            if not isinstance(s, dict):
                continue
            kind = s.get("kind")
            try:
                if kind == "box":
                    w, h = ev.opt(s.get("w")), ev.opt(s.get("h"))
                    if w and h and w > 0 and h > 0:
                        out.append(ShapeGeom("poly", geo.box_points(w, h, ev.vec(s.get("offset")),
                                                                    ev.num(s.get("angle", 0.0)))))
                elif kind == "circle":
                    r = ev.opt(s.get("r"))
                    if r and r > 0:
                        out.append(ShapeGeom("circle", [ev.vec(s.get("offset"))], r))
                elif kind == "poly":
                    pts = [ev.vec(p) for p in s.get("points") or []]
                    if len(pts) >= 3:
                        out.append(ShapeGeom("poly", pts))
                elif kind == "radial":
                    radii = [max(0.01, ev.num(r, 0.5)) for r in s.get("radii") or []]
                    if len(radii) >= 3:
                        out.append(ShapeGeom("poly", geo.radial_points(radii, ev.num(s.get("angle", 0.0)))))
                elif kind == "wing":
                    out.append(ShapeGeom("poly", T.wing_outline(
                        ev.vec(s.get("le")), ev.num(s.get("chord", 1.0), 1.0), ev.num(s.get("angle", 0.0)),
                        ev.num(s.get("thick", 0.04), 0.04), ev.num(s.get("camber", 0.0)))))
                elif kind == "chain":
                    pts = [ev.vec(p) for p in s.get("points") or []]
                    pairs = list(zip(pts, pts[1:], strict=False))
                    if s.get("closed") and len(pts) > 2:
                        pairs.append((pts[-1], pts[0]))
                    radius = float(s.get("radius", 0.05)) if T.is_number(s.get("radius", 0.05)) else 0.05
                    out += [ShapeGeom("segment", [a, c], radius) for a, c in pairs]
            except (TypeError, ValueError, IndexError):
                continue
        return out

    def _bodies(self) -> None:
        bodies = {str(b.get("id")): b for b in _items(self.data, "bodies")}
        joints = {str(j.get("id")): j for j in _items(self.data, "joints")}
        known = {b.id: b.pose for b in self.scene.bodies}
        spawn = self.scene.spawn
        ev = self.ev

        def pose(bid: str, depth: int = 0) -> Pose | None:
            if bid in known:
                return known[bid]
            b = bodies.get(bid)
            if b is None or depth > 32:
                return None
            j = joints.get(str(b.get("place_by"))) if b.get("place_by") else None
            result: Pose
            if j is not None and str(j.get("b")) == bid:
                parent = pose(str(j.get("a")), depth + 1)
                fallback: Pose = spawn if b.get("owner", "agent") == "agent" else (0.0, 0.0, 0.0)
                pp: Pose = parent if parent is not None else fallback
                angle = pp[2] + ev.num(j.get("rest_angle", 0.0))
                aw = geo.transform(ev.vec(j.get("anchor_a")), (pp[0], pp[1]), pp[2])
                ab = geo.rotate(ev.vec(j.get("anchor_b")), angle)
                result = (aw[0] - ab[0], aw[1] - ab[1], angle)
            else:
                lp, la = ev.vec(b.get("pos")), ev.num(b.get("angle", 0.0))
                if b.get("owner", "agent") == "agent":
                    wp = geo.transform(lp, (spawn[0], spawn[1]), spawn[2])
                    result = (wp[0], wp[1], la + spawn[2])
                else:
                    result = (lp[0], lp[1], la)
            known[bid] = result
            return result

        for bid, b in bodies.items():
            if bid in self.ep_body_ids and self.scene.body(bid) is not None:
                continue
            try:
                p = pose(bid)
                if p is None:
                    continue
                shapes = self._shapes(b)
                owner = "world" if b.get("owner") == "world" else "agent"
                tags = [t for t in b.get("tags") or [] if isinstance(t, str)]
                place_by = b.get("place_by") if isinstance(b.get("place_by"), str) else None
                motion = self._motion(b)
                if motion is not None:
                    p = pose_at(motion, 0.0)  # 런타임처럼 시작 자세 = 움직임의 0초 자세
                item = BodyItem(bid, ("bodies", bid), owner, b.get("type", "dynamic") == "dynamic", tags, shapes, p,
                                False, place_by, placeholder=not shapes)
                self.scene.bodies.append(self._decorate(item, b, motion))
            except Exception:  # noqa: BLE001
                continue

    def _region_points(self, r: dict[str, Any]) -> list[Point]:
        ev = self.ev
        pos = ev.vec(r.get("pos"))
        s = r.get("shape") if isinstance(r.get("shape"), dict) else {}
        kind = s.get("kind")
        if kind == "box":
            pts = geo.box_points(max(ev.num(s.get("w"), 1.0), 1e-3), max(ev.num(s.get("h"), 1.0), 1e-3),
                                 ev.vec(s.get("offset")), ev.num(s.get("angle", 0.0)))
        elif kind == "circle":
            pts = geo.circle_points(max(ev.num(s.get("r"), 1.0), 1e-3), ev.vec(s.get("offset")))
        elif kind == "poly":
            pts = [ev.vec(p) for p in s.get("points") or []]
        else:
            pts = geo.box_points(1.0, 1.0)
        return [(x + pos[0], y + pos[1]) for x, y in pts]

    def _regions(self) -> None:
        have = {r.id for r in self.scene.regions}
        for r in _items(self.data, "regions"):
            rid = str(r.get("id"))
            if rid in have:
                continue
            try:
                pts = self._region_points(r)
                if len(pts) >= 3:
                    kind = r.get("kind", "zone") if r.get("kind") in REGION_FILL else "zone"
                    self.scene.regions.append(RegionItem(rid, ("regions", rid), kind, pts))
            except Exception:  # noqa: BLE001
                continue

    def _checkpoints(self) -> None:
        have = {c.set_id for c in self.scene.checkpoints}
        for c in _items(self.data, "checkpoints"):
            cid = str(c.get("id"))
            if cid in have:
                continue
            for k, line in enumerate(c.get("lines") or []):
                if isinstance(line, list) and len(line) == 2:
                    self.scene.checkpoints.append(CheckpointItem(("checkpoints", cid), cid, k, self.ev.vec(line[0]),
                                                                 self.ev.vec(line[1]), True))

    def _joints(self) -> None:
        bodies = {str(b.get("id")): b for b in _items(self.data, "bodies")}
        for j in _items(self.data, "joints"):
            jid = str(j.get("id"))
            a, b = self.scene.body(str(j.get("a"))), self.scene.body(str(j.get("b")))
            if a is None and b is None:
                continue
            pa = geo.transform(self.ev.vec(j.get("anchor_a")), (a.pose[0], a.pose[1]), a.pose[2]) if a else None
            pb = geo.transform(self.ev.vec(j.get("anchor_b")), (b.pose[0], b.pose[1]), b.pose[2]) if b else None
            pa = pa or pb
            pb = pb or pa
            assert pa is not None and pb is not None
            placed = bodies.get(str(j.get("b")), {}).get("place_by") == jid
            kind = j.get("kind") if j.get("kind") in ("pivot", "weld", "rot_spring", "spring") else "pivot"
            self.scene.joints.append(JointItem(("joints", jid), kind, str(j.get("a")), str(j.get("b")), pa, pb, placed))

    def _actuators(self) -> None:
        ev = self.ev
        acts = _items(self.data, "actuators")
        agent_bodies = [b.id for b in self.scene.bodies if b.ref[0] == "bodies" and b.owner == "agent"]
        for a in acts:
            aid, kind = str(a.get("id")), a.get("kind")
            ref: Ref = ("actuators", aid)
            if kind == "thruster":
                body = self.scene.body(str(a.get("body")))
                if body is None:
                    continue
                self.scene.thrusters.append(ThrusterItem(ref, body.id, ev.vec(a.get("at")),
                                                         ev.num(a.get("dir"), math.pi / 2),
                                                         max(0.0, ev.num(a.get("max_force"), 0.0))))
            elif kind == "fuel_tank":
                owner = next((str(t.get("body")) for t in acts if t.get("kind") == "thruster" and t.get("tank") == aid),
                             None)
                if owner is None or self.scene.body(owner) is None:
                    owner = agent_bodies[0] if agent_bodies else None
                self.scene.badges.append(BadgeItem(ref, "fuel_tank", owner))
            elif kind == "vehicle":
                body = self.scene.body(str(a.get("body")))
                self.scene.badges.append(BadgeItem(ref, "vehicle", body.id if body else None))
            elif kind == "motor":
                self.scene.badges.append(BadgeItem(ref, "motor", None, str(a.get("joint"))))
            elif kind == "jump":
                body = self.scene.body(str(a.get("body")))
                if body is not None:
                    self.scene.jumps.append(JumpItem(ref, body.id, ev.num(a.get("dir", math.pi / 2), math.pi / 2)))
                self.scene.badges.append(BadgeItem(ref, "jump", body.id if body else None))
            elif kind == "aero":
                body = self.scene.body(str(a.get("body")))
                if body is not None:
                    self.scene.aero_bodies.add(body.id)
                self.scene.badges.append(BadgeItem(ref, "aero", body.id if body else None))
            elif kind == "move":  # 6.2.0 밀기(이동): 엔진이 표지용 기하를 내지 않으므로 문서에서 그린다
                body = self.scene.body(str(a.get("body")))
                mode = a.get("mode", "xy")
                if body is not None:
                    self.scene.moves.append(MoveItem(ref, body.id, mode if mode in ("xy", "x", "heading") else "xy"))
                self.scene.badges.append(BadgeItem(ref, "move", body.id if body else None))
            elif kind == "balance":  # 6.2.0 균형 보조(출력 채널 없음)
                body = self.scene.body(str(a.get("body")))
                if body is not None:
                    self.scene.balances.append(BalanceItem(ref, body.id, ev.num(a.get("target"), 0.0)))
                self.scene.badges.append(BadgeItem(ref, "balance", body.id if body else None))

    def _sensors(self) -> None:
        ep_rays: dict[str, list[float]] = {}
        if self.ep is not None:
            from evolab.spec import model as m

            rays = [s for s in self.ep.env.sensors if isinstance(s, m.Rays)]
            for s, view in zip(rays, self.ep._ray_views, strict=False):
                try:
                    ep_rays[s.id] = [float(x) for x in view[4]()]
                except Exception:  # noqa: BLE001
                    continue
        acts = _items(self.data, "actuators")
        for s in _items(self.data, "sensors"):
            sid, kind = str(s.get("id")), s.get("kind")
            ref: Ref = ("sensors", sid)
            if kind not in BADGE_TEXT:
                continue
            body_id = s.get("body") if isinstance(s.get("body"), str) else None
            body = self.scene.body(body_id) if body_id else None
            if kind == "rays" and body is not None:
                self._rays(ref, sid, s, body, ep_rays.get(sid))
            if kind == "target" and body is not None:
                point = self._target_point(str(s.get("target")))
                if point is not None:
                    self.scene.targets.append(TargetItem(ref, body.id, point))
            if kind == "joint_angle":
                self.scene.badges.append(BadgeItem(ref, kind, None, str(s.get("joint"))))
                continue
            if kind == "fuel":
                tank = str(s.get("tank"))
                owner = next((str(t.get("body")) for t in acts
                              if t.get("kind") == "thruster" and t.get("tank") == tank), None)
                tb = next((b for b in self.scene.badges if b.ref == ("actuators", tank)), None)
                body_id = owner if owner and self.scene.body(owner) else (tb.body if tb else None)
                self.scene.badges.append(BadgeItem(ref, kind, body_id))
                continue
            self.scene.badges.append(BadgeItem(ref, kind, body.id if body else None))

    def _rays(self, ref: Ref, sid: str, s: dict[str, Any], body: BodyItem, alphas: list[float] | None) -> None:
        ev = self.ev
        n = int(s.get("count", 1)) if T.is_number(s.get("count", 1)) else 1
        n = max(1, min(64, n))
        spread, center = ev.num(s.get("spread"), math.pi), ev.num(s.get("center", 0.0))
        length = max(1e-6, ev.num(s.get("length"), 10.0))
        angles = [center] if n == 1 else [center - spread / 2.0 + spread * i / (n - 1) for i in range(n)]
        x, y, a = body.pose
        o = geo.transform(ev.vec(s.get("origin")), (x, y), a)
        if alphas is None or len(alphas) != n:
            alphas = [1.0] * n
        lines = []
        for ang, alpha in zip(angles, alphas, strict=True):
            th = a + ang
            d = alpha * length
            lines.append((o, (o[0] + d * math.cos(th), o[1] + d * math.sin(th)), alpha))
        self.scene.rays.append(RayItem(ref, body.id, o, lines))

    def _target_point(self, target: str) -> Point | None:
        r = self.scene.region(target)
        if r is not None:
            return geo.polygon_center(r.points)
        b = self.scene.body(target)
        if b is not None:
            return (b.pose[0], b.pose[1])
        return None


def build_scene(doc: EnvDocument, trial_index: int = 0) -> Scene:
    """문서의 지금 내용으로 장면을 만든다. 어떤 내용에서도 예외를 내지 않는다. trial_index: 미리보기 시험 번호."""
    data = doc.data if isinstance(doc.data, dict) else {}
    message = ""
    try:
        spec = doc.spec()
    except Exception:  # noqa: BLE001
        spec = None
    exact = spec is not None
    if spec is None:
        spec = _repaired_spec(data)
    ep = None
    if spec is not None:
        try:
            ep = _preview_episode(spec, trial_index)
        except Exception as exc:  # noqa: BLE001 (EpisodeBuildError 등: 숫자만으로 그린다)
            message = S.EDC_PREVIEW_BUILD_FAILED.format(reason=exc)
            exact = False
    if not exact and not message:
        message = S.EDC_PREVIEW_PARTIAL
    try:
        scene = _SceneBuilder(data, ep, exact, message).build()
    except Exception:  # noqa: BLE001
        return Scene(message=message)
    try:
        _mark_structure(scene, doc, ep)
    except Exception:  # noqa: BLE001 (구조 표시는 덤: 실패해도 그림은 그린다)
        pass
    return scene


def _mark_structure(scene: Scene, doc: EnvDocument, ep: Any) -> None:
    """(v2.1) 없어질 수 있는 물체와 처음 몸에 없는 물체를 장면에 적는다. 없는 물체는 런타임이 만들지 않아 _SceneBuilder가
    숫자로 계산해 넣었다(그래서 「없는 부품도 보기」로 그릴 수 있다)."""
    from evolab.spec import structure
    from evolab.spec.compiled import EnvFileError
    from evolab.spec.io import parse_env

    spec = doc.spec()
    try:
        env = spec.file if spec is not None else parse_env(doc.data)
    except (EnvFileError, ValueError, TypeError):
        return
    optional = structure.optional_bodies(env)
    if not optional:
        return
    absent = frozenset(ep.absent) if ep is not None else (initial_absent(doc, env) or frozenset())
    absent &= optional
    scene.optional = optional
    scene.absent = absent
    scene.absent_parts = frozenset(attached_parts(doc.data, absent))
    for b in scene.bodies:
        if b.ref[0] == "bodies":
            b.optional = b.id in optional
            b.absent = b.id in absent


# ---------------------------------------------------------------------------
# 입력 사건, 누른 것
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanvasEvent:
    screen: QPointF
    world: Point
    modifiers: Qt.KeyboardModifier
    force_alt: bool = False

    @property
    def shift(self) -> bool:
        return bool(self.modifiers & Qt.KeyboardModifier.ShiftModifier)

    @property
    def ctrl(self) -> bool:
        return bool(self.modifiers & Qt.KeyboardModifier.ControlModifier)

    @property
    def alt(self) -> bool:
        return self.force_alt or bool(self.modifiers & Qt.KeyboardModifier.AltModifier)

    def with_alt(self, alt: bool) -> CanvasEvent:
        return replace(self, force_alt=self.force_alt or alt)


@dataclass
class Pick:
    ref: Ref | None
    part: str
    """body, gen, region, cp_line, cp_end, joint, thruster, thruster_tip, ray, badge, rotate, spawn, spawn_dir"""
    index: int = 0
    center: Point | None = None


# ---------------------------------------------------------------------------
# 캔버스
# ---------------------------------------------------------------------------


def _pen(color: QColor, width: float = 1.0, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(color, width)
    pen.setStyle(style)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    return pen


class EditorCanvas(QWidget):
    """가운데 캔버스. 도구는 set_tool(도구 id)로 고른다(tools.TOOL_IDS)."""

    coords_changed = Signal(float, float)
    """커서 아래 세계 좌표(m)."""
    status_changed = Signal(str)
    """도구 안내나 할 수 없는 조작의 이유(한국어)."""
    tool_changed = Signal(str)
    preview_changed = Signal(float)
    """미리보기 시각(초)이 바뀌었다."""
    playing_changed = Signal(bool)
    """움직임 미리보기 재생을 시작했다(True)·멈췄다(False)."""
    motion_changed = Signal(bool)
    """장면에 정해진 움직임 물체가 생겼다(True)·없어졌다(False)."""
    preview_trial_changed = Signal(int)
    """미리보기 시험 번호가 바뀌었다(「미리보기 시험 바꾸기」)."""
    replicate_requested = Signal(str)
    """물체 오른쪽 메뉴의 「배열로 늘리기…」(v2.1): 늘릴 물체 id. 늘릴 수 없는 물체면 내지 않고 이유를 상태 줄에."""
    show_absent_changed = Signal(bool)
    """「없는 부품도 보기」가 바뀌었다(v2.1)."""

    def __init__(self, doc: EnvDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)
        self.center: Point = (0.0, 10.0)
        self.scale = 20.0
        """화면 px / m"""
        self.snap_enabled = True
        self.snap_step = DEFAULT_SNAP_STEP
        self.new_owner = "agent"
        """모양 도구로 만드는 새 물체의 소유(agent 또는 world)."""
        self.new_motion = "static"
        """세계 물체를 그릴 때의 움직임(6.2.0 「새로 그리는 것」: static 고정(땅·벽) 또는 dynamic 움직임(공·상자))."""
        self.picker_factory: Any = None
        """틀 보관함 창을 만드는 함수(kind, world, data, parent) -> 대화 상자(exec(), choice()). None이면 app.preset.cards.TemplatePicker.
        테스트가 바꿔 끼운다."""
        self.last_picker: Any = None
        """마지막으로 띄운 틀 보관함 창(테스트용)."""
        self._drop: tuple[Any, dict[str, float] | None, bool] | None = None
        self._drop_needs: list[Any] = []  # 놓을 틀 앞에 함께 넣을 틀(보관함 창의 「…도 함께 넣기」)
        self._swallow_release = False
        self.status_text = ""
        self.cursor_world: Point | None = None
        self._scene: Scene | None = None
        self._dirty = True
        self._live = 0
        self._badge_cache: tuple[Any, list[tuple[BadgeItem, QRectF]]] | None = None
        self._issue_refs: dict[Ref, str] = {}
        self._fit_pending = True
        self._pan: tuple[QPointF, Point] | None = None
        self._space = False
        self._press_alt = False
        self._button_down = False
        self._tools: dict[str, T.Tool] = {}
        self._tool_id = "select"
        self._tool: T.Tool = self._get_tool("select")
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(PREVIEW_DEBOUNCE_MS)
        self._timer.timeout.connect(self.update)
        self._message_timer = QTimer(self)
        self._message_timer.setSingleShot(True)
        self._message_timer.setInterval(MESSAGE_MS)
        self._message_timer.timeout.connect(self._clear_message)
        self.fit_action = QAction(S.EDC_FIT, self)
        self.fit_action.setShortcut(QKeySequence("F"))
        self.fit_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.fit_action.triggered.connect(self.fit)
        self.addAction(self.fit_action)
        # v2: 미리보기 시각, 재생(정적 그림 캐시 + 움직이는 물체 덧그리기), 「전체 보기」 순환
        self.preview_time = 0.0
        self.preview_trial_index = 0
        """미리보기 시험 번호(0 = 전시 시험). 「미리보기 시험 바꾸기」가 늘리고 새 파일을 열면 0. 학습과 저장에는 영향 없음."""
        self._playing = False
        self._play_from = 0.0
        self._play_clock = QElapsedTimer()
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(round(1000 / PREVIEW_FPS))
        self._play_timer.timeout.connect(self._on_play_tick)
        self._static_cache: tuple[Any, QImage] | None = None
        self._has_motion = False
        self._fit_mode = ""
        self._fit_camera: tuple[Point, float] | None = None
        self.drawn_poses: dict[str, Pose] = {}
        """마지막으로 그린 정해진 움직임 물체의 자세(물체 id -> 세계 자세). 테스트와 화면 확인용."""
        # v2.1 몸 구조
        self.show_absent = True
        """「없는 부품도 보기」: 처음 몸에 없는 부품을 옅은 점선으로 그린다(끄면 그리지도 누르지도 않는다)."""
        self.structure_highlight: frozenset[Ref] = frozenset()
        """구조 탭이 강조하라는 부품(고른 물체와 함께 사라지는 부품)."""
        self.drawn_optional: dict[str, str] = {}
        """마지막 정적 그림에서 점선으로 그린 없어질 수 있는 물체(id -> "present" | "absent"). 테스트와 화면 확인용."""
        self._right_cancelled = False
        doc.changed.connect(self._on_doc_changed)
        tool_request = getattr(doc, "tool_request", None)
        if tool_request is not None:
            tool_request.connect(self._on_tool_request)
        doc.selection_changed.connect(self._on_selection_changed)
        doc.checked.connect(self._on_checked)
        doc.file_changed.connect(self._on_file_changed)
        self._on_checked(doc.issues)
        self.setCursor(self._tool.cursor)

    # ------------------------------------------------------------------
    # 문서 신호
    # ------------------------------------------------------------------
    def _on_doc_changed(self) -> None:
        self._dirty = True
        self._static_cache = None
        if self._playing or self.preview_time != 0.0:
            self.stop_preview()  # 문서가 바뀌면 멈추고 0초(설계 D24)
        if self._live:
            self.update()
        else:
            self._timer.start()

    def _on_tool_request(self, tool_id: str, ref: object) -> None:
        """속성 패널 단추가 도구를 켜 달라고 했다(「캔버스에서 길 그리기」)."""
        if tool_id == "path" and isinstance(ref, tuple) and len(ref) == 2:
            self.start_path(str(ref[1]))
        elif tool_id in T.TOOL_IDS:
            self.set_tool(tool_id)
            self.setFocus(Qt.FocusReason.OtherFocusReason)

    def start_path(self, body_id: str) -> bool:
        """길 그리기 도구를 켜고 이 물체의 길을 바로 그리기 시작한다."""
        self.set_tool("path")
        tool = self._tool
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        return isinstance(tool, T.PathTool) and tool.start_with(body_id)

    def _on_selection_changed(self, refs: list[Any]) -> None:
        if not self.show_absent and any(not self.visible_ref(tuple(r)) for r in refs):
            # 「없는 부품도 보기」가 꺼진 채 숨은 부품이 골라졌다(구조 탭 줄, 검사 결과 등 캔버스 밖에서): 보기를 다시
            # 켠다. 보이지 않는 부품이 고른 채로 남아 끌리거나 Delete로 지워지지 않게(set_show_absent가 탭 체크도 맞춘다)
            self.set_show_absent(True)
        self.update()

    def _on_checked(self, issues: list[Any]) -> None:
        out: dict[Ref, str] = {}
        data = self.doc.data
        rank = {"error": 2, "warning": 1}
        for issue in issues:
            ref = where_to_ref(data, issue.where)
            if ref is None:
                continue
            # 강조는 오류 > 경고 > 안내(안내는 색이 없다). 안내가 앞의 경고를 덮지 않게 한다.
            if ref not in out or rank.get(issue.severity, 0) > rank.get(out[ref], 0):
                out[ref] = issue.severity
        self._issue_refs = out
        self.update()

    def _on_file_changed(self) -> None:
        if self.doc.undo_stack.count() == 0:
            self._fit_pending = True
            self._fit_mode = ""  # 새 파일: 「전체 보기」 순환을 처음부터(기본 보기 먼저)
            self.preview_trial_index = 0
            self._dirty = True
            self.update()

    def issue_severity(self, ref: Ref) -> str | None:
        return self._issue_refs.get(ref)

    # ------------------------------------------------------------------
    # v2.1 몸 구조: 없는 부품 보기, 구조 탭 강조, 오른쪽 메뉴
    # ------------------------------------------------------------------
    def set_show_absent(self, on: bool) -> None:
        """「없는 부품도 보기」를 켜고 끈다."""
        on = bool(on)
        if on == self.show_absent:
            return
        self.show_absent = on
        self._badge_cache = None
        self._static_cache = None
        if not on:
            self._drop_hidden_selection()
        self.show_absent_changed.emit(on)
        self.update()

    def _drop_hidden_selection(self) -> None:
        """숨는 부품을 고르고 있었으면 고르기를 푼다(보이지 않는 것을 끌거나 Delete로 지우지 않게)."""
        hidden = [r for r in self.doc.selection if not self.visible_ref(r)]
        if hidden:
            self.doc.select([r for r in self.doc.selection if r not in hidden])

    def set_structure_highlight(self, refs: Iterable[Ref]) -> None:
        """구조 탭이 고른 물체와 함께 사라지는 부품을 점선으로 강조한다(빈 목록이면 끈다)."""
        new = frozenset(tuple(r) for r in refs)  # type: ignore[misc]
        if new != self.structure_highlight:
            self.structure_highlight = new
            self._static_cache = None
            self.update()

    def _absent_ref(self, ref: Ref) -> bool:
        """처음 몸에 없는 부품인가(없는 물체 또는 거기 붙은 부품)."""
        scene = self._scene
        if scene is None or not scene.absent:
            return False
        return (ref[0] == "bodies" and ref[1] in scene.absent) or ref in scene.absent_parts

    def visible_ref(self, ref: Ref) -> bool:
        """그리고 누를 수 있나: 처음 몸에 없는 부품은 「없는 부품도 보기」가 꺼져 있으면 아니다."""
        return self.show_absent or not self._absent_ref(ref)

    def _fade(self, p: QPainter, ref: Ref) -> bool:
        """처음 몸에 없는 부품이면 옅게 그린다(불투명도를 낮춤). 낮췄으면 True(그린 뒤 _unfade)."""
        if self._absent_ref(ref):
            p.setOpacity(ABSENT_OPACITY)
            return True
        return False

    @staticmethod
    def _unfade(p: QPainter, faded: bool) -> None:
        if faded:
            p.setOpacity(1.0)

    def linked(self, ref: Ref) -> bool:
        """구조 탭이 강조하는 부품인가."""
        return ref in self.structure_highlight

    def body_menu(self, body_id: str) -> QMenu:
        """물체 오른쪽 메뉴: 「배열로 늘리기…」."""
        menu = QMenu(self)
        menu.setToolTipsVisible(True)  # 할 수 없는 이유(몸통·세계 물체)를 흐린 항목의 풍선 도움말로
        act = menu.addAction(S.ED_REPLICATE)
        err = T.replicate_error(self.doc, body_id)
        act.setEnabled(err is None)
        act.setToolTip(err or S.ED_REPLICATE_HINT.format(name=body_id))
        act.triggered.connect(lambda _c=False, b=body_id: self.request_replicate(b))
        return menu

    def request_replicate(self, body_id: str) -> bool:
        """「배열로 늘리기…」를 고른 것: 늘릴 수 있으면 replicate_requested를 내고, 아니면 이유를 상태 줄에."""
        err = T.replicate_error(self.doc, body_id)
        if err is not None:
            self.message(err)
            return False
        self.replicate_requested.emit(body_id)
        return True

    def contextMenuEvent(self, ev: QContextMenuEvent) -> None:  # noqa: N802
        if self._right_cancelled:  # 오른쪽 단추로 그리던 것을 취소했다: 메뉴는 띄우지 않는다
            self._right_cancelled = False
            return
        body = None
        if ev.reason() == QContextMenuEvent.Reason.Mouse:
            picked = self.pick_body(self.screen_to_world(QPointF(ev.pos())))
            if picked is not None and not picked.generated and picked.ref[0] == "bodies":
                body = picked.id
                if picked.ref not in self.doc.selection:
                    self.doc.select([picked.ref])
        else:
            body = next((cid for section, cid in self.doc.selection if section == "bodies"), None)
        if body is None:
            return
        self.show_menu(self.body_menu(body), ev.globalPos())

    def show_menu(self, menu: QMenu, global_pos: QPoint) -> None:
        """메뉴를 띄운다(끝날 때까지 기다림). 테스트가 바꿔 끼운다."""
        menu.exec(global_pos)

    # ------------------------------------------------------------------
    # 장면
    # ------------------------------------------------------------------
    def scene(self) -> Scene:
        """지금 문서의 장면(바뀌었으면 다시 만든다)."""
        if self._dirty or self._scene is None:
            self._scene = build_scene(self.doc, self.preview_trial_index)
            self._dirty = False
            self._badge_cache = None
            self._static_cache = None
            has = bool(self._scene.moving())
            if has != self._has_motion:
                self._has_motion = has
                self.motion_changed.emit(has)
            if self._fit_pending and self.isVisible() and self.width() > 0 and self.height() > 0:
                self._fit_pending = False
                self._fit_initial(self._scene)
        return self._scene

    @property
    def has_motion(self) -> bool:
        """장면에 정해진 움직임 물체가 있나(미리보기 줄을 보일지)."""
        self.scene()
        return self._has_motion

    def _paint_scene(self) -> Scene:
        if self._dirty and self._timer.isActive() and self._scene is not None:
            return self._scene  # 모아서 다시 만든다
        return self.scene()

    def begin_live_drag(self) -> None:
        """끄는 동안에는 미리보기를 바로바로 다시 만든다."""
        if not self._live:
            self._live = 1
            self._timer.stop()

    def end_live_drag(self) -> None:
        self._live = 0

    # ------------------------------------------------------------------
    # 카메라와 좌표
    # ------------------------------------------------------------------
    def world_to_screen(self, x: float, y: float) -> QPointF:
        return QPointF(self.width() / 2.0 + (x - self.center[0]) * self.scale,
                       self.height() / 2.0 - (y - self.center[1]) * self.scale)

    def screen_to_world(self, p: QPointF) -> Point:
        return (self.center[0] + (p.x() - self.width() / 2.0) / self.scale,
                self.center[1] - (p.y() - self.height() / 2.0) / self.scale)

    def fit(self) -> None:
        """「전체 보기」: 기본 보기(editor.view)가 있으면 그 상자, 그 상태에서 한 번 더 누르면 장면 전체(순환).
        기본 보기가 없으면 늘 장면 전체."""
        self._fit_pending = False
        scene = self.scene()
        box = self.view_box()
        unchanged = self._fit_camera == (self.center, self.scale)
        if box is not None and not (self._fit_mode == "view" and unchanged):
            self._fit_box(box)
            self._fit_mode = "view"
        else:
            self._fit_to(scene)
            self._fit_mode = "full"
        self._fit_camera = (self.center, self.scale)
        self.update()

    def _fit_initial(self, scene: Scene) -> None:
        """파일을 처음 열었을 때의 카메라: 기본 보기가 있으면 그 상자, 없으면 장면 전체."""
        box = self.view_box()
        if box is not None:
            self._fit_box(box)
            self._fit_mode = "view"
        else:
            self._fit_to(scene)
            self._fit_mode = "full"
        self._fit_camera = (self.center, self.scale)

    def _fit_box(self, box: tuple[float, float, float, float]) -> None:
        x0, y0, x1, y1 = box
        w, h = max(self.width(), 50), max(self.height(), 50)
        self.scale = min(max(min(w / (x1 - x0), h / (y1 - y0)), SCALE_RANGE[0]), SCALE_RANGE[1])
        self.center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        self._badge_cache = None

    # ---- 기본 보기 (editor.view, 해시 밖) ----
    def view_box(self) -> tuple[float, float, float, float] | None:
        """문서의 기본 보기 상자(없거나 잘못되었으면 None)."""
        data = self.doc.data if isinstance(self.doc.data, dict) else {}
        return editor_view_box(data.get("editor"))

    def visible_box(self) -> tuple[float, float, float, float]:
        """지금 화면에 보이는 세계 상자 (x0, y0, x1, y1)."""
        x0, y1 = self.screen_to_world(QPointF(0.0, 0.0))
        x1, y0 = self.screen_to_world(QPointF(float(max(self.width(), 1)), float(max(self.height(), 1))))
        return (x0, y0, x1, y1)

    def save_view_box(self) -> tuple[float, float, float, float]:
        """지금 보기를 기본 보기로 저장한다(되돌리기 한 단계, 환경 해시는 그대로)."""
        box = tuple(round(v, 3) + 0.0 for v in self.visible_box())
        editor = self.doc.data.get("editor") if isinstance(self.doc.data.get("editor"), dict) else {}
        new = with_view_box(editor, box)  # type: ignore[arg-type]
        self.doc.edit(S.ED_VIEW_SAVE, lambda d: d.__setitem__("editor", new))
        self._fit_mode = "view"
        self._fit_camera = (self.center, self.scale)
        self.message(S.ED_VIEW_SAVED)
        return box  # type: ignore[return-value]

    def clear_view_box(self) -> None:
        """기본 보기를 지운다(되돌리기 한 단계). 다른 editor 칸은 그대로 둔다."""
        editor = self.doc.data.get("editor") if isinstance(self.doc.data.get("editor"), dict) else None
        if editor is None or "view" not in editor:
            return
        new = with_view_box(editor, None)

        def mutate(d: dict[str, Any]) -> None:
            if new:
                d["editor"] = new
            else:
                d.pop("editor", None)

        self.doc.edit(S.ED_VIEW_CLEAR, mutate)
        self._fit_mode = ""

    # ---- 미리보기 시각 (v2 D24) ----
    @property
    def playing(self) -> bool:
        return self._playing

    def max_preview_time(self) -> float:
        """미리보기 시각의 끝 = 세계 최대 시간(숫자 칸이면 문서에서 바로, 아니면 장면이 계산한 값)."""
        world = self.doc.data.get("world") if isinstance(self.doc.data, dict) else None
        v = world.get("max_time") if isinstance(world, dict) else None
        if T.is_number(v) and math.isfinite(float(v)) and float(v) > 0:
            return float(v)
        return self._scene.max_time if self._scene is not None else DEFAULT_MAX_TIME

    def set_preview_time(self, t: float) -> None:
        """정해진 움직임 물체를 시각 t(초)의 자세로 그린다(0 ~ 세계 최대 시간). 학습에는 영향 없음."""
        t = min(max(0.0, float(t)), self.max_preview_time())
        if self._playing:
            self._play_from = t
            self._play_clock.restart()
        if t != self.preview_time:
            self.preview_time = t
            self.preview_changed.emit(t)
            self.update()

    def play(self) -> None:
        """움직임 미리보기를 지금 시각부터 재생한다(끝이면 0초부터)."""
        if self._playing or not self.has_motion:
            return
        if self.preview_time >= self.max_preview_time() - 1e-9:
            self.set_preview_time(0.0)
        self._playing = True
        self._play_from = self.preview_time
        self._play_clock.start()
        self._play_timer.start()
        self.playing_changed.emit(True)
        self.update()

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._play_timer.stop()
        self.playing_changed.emit(False)
        self.update()

    def toggle_play(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def stop_preview(self) -> None:
        """멈추고 0초로."""
        self.pause()
        self._static_cache = None
        if self.preview_time != 0.0:
            self.preview_time = 0.0
            self.preview_changed.emit(0.0)
            self.update()

    def next_preview_trial(self) -> int:
        """「미리보기 시험 바꾸기」: 다음 시험의 랜덤 값(장애물 줄 등)으로 미리보기를 다시 만든다(검증 세트를 돎).
        미리보기 시각과 재생은 그대로 둔다. 새 시험 번호를 돌려준다."""
        from evolab.engine import seeds

        self.preview_trial_index = (self.preview_trial_index + 1) % seeds.VALIDATION_SET
        self._dirty = True
        self._static_cache = None
        self.preview_trial_changed.emit(self.preview_trial_index)
        self.update()
        return self.preview_trial_index

    def _on_play_tick(self) -> None:
        if not self._playing:
            return
        t = self._play_from + self._play_clock.elapsed() / 1000.0
        end = self.max_preview_time()
        if t >= end:
            self.preview_time = end
            self.preview_changed.emit(end)
            self.pause()
            return
        self.preview_time = t
        self.preview_changed.emit(t)
        self.update()

    def _fit_to(self, scene: Scene) -> None:
        pts: list[Point] = [(scene.spawn[0], scene.spawn[1])]
        for b in scene.bodies:
            pts += [(b.bbox[0], b.bbox[1]), (b.bbox[2], b.bbox[3])]
        for r in scene.regions:
            pts += [(r.bbox[0], r.bbox[1]), (r.bbox[2], r.bbox[3])]
        for c in scene.checkpoints:
            pts += [c.a, c.b]
        x0, y0, x1, y1 = _bbox(pts)
        if x1 - x0 < 4.0 and y1 - y0 < 4.0:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            x0, y0, x1, y1 = cx - 10.0, cy - 7.0, cx + 10.0, cy + 7.0
        w, h = max(self.width(), 50), max(self.height(), 50)
        mx, my = (x1 - x0) * 0.06 + 0.5, (y1 - y0) * 0.06 + 0.5
        sx, sy = w / (x1 - x0 + 2 * mx), h / (y1 - y0 + 2 * my)
        self.scale = min(max(min(sx, sy), SCALE_RANGE[0]), SCALE_RANGE[1])
        self.center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        self._badge_cache = None

    def zoom_at(self, screen: QPointF, factor: float) -> None:
        w = self.screen_to_world(screen)
        self.scale = min(max(self.scale * factor, SCALE_RANGE[0]), SCALE_RANGE[1])
        self.center = (w[0] - (screen.x() - self.width() / 2.0) / self.scale,
                       w[1] + (screen.y() - self.height() / 2.0) / self.scale)
        self._badge_cache = None
        self.update()

    def owner_frame(self, owner: str) -> Pose:
        """파일 좌표계: 에이전트 물체는 시작 위치 기준, 세계 물체는 세계."""
        return self.scene().spawn if owner == "agent" else (0.0, 0.0, 0.0)

    def thrust_length(self, max_force: float) -> float:
        """추진기 화살표 길이(m): 최대 힘에 비례."""
        return min(4.0, 0.3 + max(0.0, max_force) / 200.0)

    # ---- 격자 맞춤 ----
    def snap_on(self, e: CanvasEvent | None = None) -> bool:
        return self.snap_enabled and self.snap_step > 0 and not (e is not None and e.alt)

    def snap(self, p: Point, e: CanvasEvent | None = None) -> Point:
        if not self.snap_on(e):
            return (T.clean(p[0]), T.clean(p[1]))
        return (T.snap_value(p[0], self.snap_step), T.snap_value(p[1], self.snap_step))

    def snap_length(self, r: float, e: CanvasEvent | None = None) -> float:
        if not self.snap_on(e):
            return T.clean(r)
        return max(self.snap_step, T.snap_value(r, self.snap_step))

    def snap_dir(self, a: float, e: CanvasEvent | None = None, step: float = T.ANGLE_SNAP) -> float:
        if not self.snap_on(e):
            return a
        k = round(a / step)
        a = 0.0 if k == 0 else k * step
        return math.pi if abs(a + math.pi) < 1e-12 else a  # -pi보다 pi로 적는다

    def set_snap(self, enabled: bool, step: float | None = None) -> None:
        self.snap_enabled = bool(enabled)
        if step is not None and step > 0:
            self.snap_step = float(step)
        self.update()

    def set_new_owner(self, owner: str) -> None:
        if owner in ("agent", "world"):
            self.new_owner = owner
            self.update()

    def set_new_motion(self, motion: str) -> None:
        """세계 물체를 그릴 때의 움직임(static·dynamic, 6.2.0). 에이전트 물체는 늘 움직임이다."""
        if motion in T.WORLD_MOTIONS:
            self.new_motion = motion

    def set_owner_mode(self, owner: str, motion: str) -> None:
        """「새로 그리는 것」 두 단추와 고정/움직임 고르기를 한 번에(설계 DU 7.3 E1)."""
        self.set_new_owner(owner)
        self.set_new_motion(motion)

    def owner_color(self) -> QColor:
        """새로 그리는 물체의 소유 색(커서 옆 글자와 고무줄 그림)."""
        return T.OWNER_COLORS.get(self.new_owner, T.OWNER_COLORS["agent"])

    def owner_cursor_text(self) -> str:
        """모양 도구가 켜져 있을 때 커서 옆 글자(「에이전트」/「세계」). 모양 도구가 아니면 빈 글."""
        if self._tool_id not in SHAPE_TOOLS or self._drop is not None:
            return ""
        return S.EDC_CURSOR_OWNER.get(self.new_owner, "")

    # ------------------------------------------------------------------
    # 틀 넣기 (6.2.0, 설계 DU 4.2: 팔레트 「틀」 단추 -> 보관함 -> 놓을 자리 누르기)
    # ------------------------------------------------------------------
    def world_view(self) -> str:
        return T.world_view(self.doc.data)

    def pick_template(self, kind: str) -> bool:
        """보관함 창을 띄워 kind(body·prop)의 틀을 고르게 하고, 고르면 놓을 자리를 기다린다. 고르지 않으면 False."""
        factory = self.picker_factory
        if factory is None:
            from evolab.app.preset.cards import TemplatePicker  # 안내 창 묶음(W)의 보관함 창

            factory = TemplatePicker
        picker = factory(kind, self.world_view(), self.doc.data, self)
        self.last_picker = picker
        if picker.exec() != QDialog.DialogCode.Accepted:
            return False
        choice = picker.choice()
        if not choice:
            return False
        tpl, values, mirror = choice[0], choice[1], bool(choice[2])  # 4번째(진화할 슬라이더)는 6.2.0에 없다(「나중에」)
        needs_add = getattr(picker, "needs_add", None)
        needs = list(needs_add()) if callable(needs_add) else []  # 「…도 함께 넣기」를 켰으면 먼저 넣을 틀(땅 등)
        self.begin_template_drop(tpl, values, mirror, needs)
        return True

    def begin_template_drop(self, tpl: Any, values: dict[str, float] | None = None, mirror: bool = False,
                            needs: list[Any] | None = None) -> None:
        """다음에 캔버스를 누른 자리(격자에 맞춤)에 이 틀을 넣는다. Esc나 오른쪽 단추로 그만둔다.

        needs: 그 앞에 기본 자리((0, 0))에 함께 넣을 틀(보관함 창의 「…도 함께 넣기」). 모두 되돌리기 한 단계."""
        self._tool.cancel()
        self._drop = (tpl, dict(values) if values else None, bool(mirror))
        self._drop_needs = list(needs or [])
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.message(S.TPL_INSERT_HINT.format(name=tpl.name), transient=False)
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    @property
    def template_drop(self) -> Any:
        """놓을 자리를 기다리는 틀(없으면 None)."""
        return self._drop[0] if self._drop is not None else None

    def cancel_template_drop(self) -> None:
        if self._drop is None:
            return
        self._drop = None
        self.setCursor(self._tool.cursor)
        self.message(self._tool.hint(), transient=False)

    def drop_template_at(self, w: Point, e: CanvasEvent | None = None) -> list[Ref]:
        """기다리는 틀을 세계 점 w(격자에 맞춤)에 넣는다(되돌리기 한 단계). 넣은 부품 목록."""
        if self._drop is None:
            return []
        tpl, values, mirror = self._drop
        needs, self._drop_needs = self._drop_needs, []
        self._drop = None
        self.setCursor(self._tool.cursor)
        at = self.snap(w, e)
        from evolab.presets import template as tp

        stack = self.doc.undo_stack
        stack.beginMacro(S.TPL_UNDO_INSERT.format(name=tpl.name))
        try:
            for need in needs:  # 빈자리를 채울 틀(땅 등)을 먼저: 그래야 아래 틀의 빈자리가 그것을 가리킨다
                T.insert_template(self.doc, need, None, at=(0.0, 0.0))
            _slots, missing = tp.resolve_slots(tpl, self.doc.data)
            refs = T.insert_template(self.doc, tpl, values, at=at, mirror=mirror)
        except (tp.TemplateError, ValueError) as exc:
            stack.endMacro()
            if needs:
                stack.undo()  # 반쯤 넣은 것을 남기지 않는다
            self.message(str(exc))
            return []
        stack.endMacro()
        n = T.template_counts(refs)
        text = S.TPL_INSERTED.format(name=tpl.name, bodies=n["bodies"], joints=n["joints"], actuators=n["actuators"],
                                     sensors=n["sensors"])
        for name in missing:  # 장면에 없던 빈자리 대상(땅 등): 무엇을 더 넣어야 하는지 알린다
            slot = tpl.slots[name]
            text += " " + S.TPL_NEEDS.format(what=slot.label or name.lstrip("$"))
        self.message(text)
        return refs

    def _paint_drop(self, p: QPainter) -> None:
        """놓을 자리를 기다리는 틀의 윤곽을 커서 자리에 흐리게."""
        if self._drop is None or self.cursor_world is None:
            return
        from evolab.presets import template as tp

        tpl, values, _mirror = self._drop
        try:
            shapes = tp.outline(tpl, values)
        except (tp.TemplateError, ValueError, KeyError, TypeError):
            return
        at = self.snap(self.cursor_world)
        p.setPen(_pen(COLOR_SELECT, 1.2, Qt.PenStyle.DashLine))
        p.setBrush(QBrush(COLOR_GHOST))
        for kind, _role, pts, r in shapes:
            q = [self._q((x + at[0], y + at[1])) for x, y in pts]
            if kind == "circle" and q:
                p.drawEllipse(q[0], r * self.scale, r * self.scale)
            elif kind == "poly" and len(q) >= 3:
                p.drawPolygon(QPolygonF(q))
            elif len(q) >= 2:
                p.drawPolyline(QPolygonF(q))

    def _paint_owner_cursor(self, p: QPainter) -> None:
        text = self.owner_cursor_text()
        if not text or self.cursor_world is None:
            return
        c = self._q(self.cursor_world)
        f = QFont(self.font())
        f.setPixelSize(11)
        p.setFont(f)
        p.setPen(self.owner_color())
        p.drawText(c + QPointF(14, 24), text)

    # ------------------------------------------------------------------
    # 도구
    # ------------------------------------------------------------------
    @property
    def tool_id(self) -> str:
        return self._tool_id

    @property
    def tool(self) -> T.Tool:
        return self._tool

    def _get_tool(self, tool_id: str) -> T.Tool:
        if tool_id not in self._tools:
            self._tools[tool_id] = T.make_tool(tool_id, self)
        return self._tools[tool_id]

    def set_tool(self, tool_id: str) -> None:
        if tool_id not in T.TOOL_IDS:
            raise KeyError(tool_id)
        if tool_id in T.INSTANT_TOOLS:
            # 부품을 더하고(더한 부품이 선택된다) 선택 도구로 돌아간다.
            T.run_instant(tool_id, self)
            if self._tool_id == "select":
                self.tool_changed.emit("select")  # 팔레트 표시를 맞춘다
            else:
                self.set_tool("select")
            return
        self.cancel_template_drop()
        if tool_id == self._tool_id:
            return
        self._tool.cancel()
        self._tool_id = tool_id
        self._tool = self._get_tool(tool_id)
        self.setCursor(self._tool.cursor)
        self.message(self._tool.hint(), transient=False)
        self.tool_changed.emit(tool_id)
        self.update()

    def message(self, text: str, transient: bool = True) -> None:
        self.status_text = text
        self.status_changed.emit(text)
        if transient and text:
            self._message_timer.start()
        self.update()

    def _clear_message(self) -> None:
        self.status_text = self._tool.hint()
        self.update()

    def set_hover_cursor(self, cursor: Qt.CursorShape) -> None:
        if not self._space and self.cursor().shape() != cursor:
            self.setCursor(cursor)

    # ------------------------------------------------------------------
    # 누르기 판정
    # ------------------------------------------------------------------
    def _tol(self) -> float:
        return PICK_PX / self.scale

    def _near(self, a: Point, b: Point, px: float) -> bool:
        pa, pb = self.world_to_screen(*a), self.world_to_screen(*b)
        return math.hypot(pa.x() - pb.x(), pa.y() - pb.y()) <= px

    def selected_body(self) -> BodyItem | None:
        sel = self.doc.selection
        if len(sel) == 1 and sel[0][0] == "bodies":
            return self.scene().body(sel[0][1])
        return None

    def rotate_handle(self, body: BodyItem) -> Point:
        x, y, a = body.pose
        r = max((math.dist((x, y), p) for p in ((body.bbox[0], body.bbox[1]), (body.bbox[2], body.bbox[3]))),
                default=0.5)
        r = min(r, 1e4) + ROTATE_HANDLE_PX / self.scale
        return (x - r * math.sin(a), y + r * math.cos(a))

    def spawn_handles(self) -> tuple[Point, Point]:
        x, y, a = self.scene().spawn
        d = SPAWN_ARROW_PX / self.scale
        return (x, y), (x + d * math.cos(a), y + d * math.sin(a))

    def thruster_points(self, t: ThrusterItem) -> tuple[Point, Point] | None:
        """(추진기 위치, 꼬리). 화살표는 꼬리에서 추진기 위치로, 힘이 미는 방향을 가리킨다(꼬리 쪽이 분사)."""
        body = self.scene().body(t.body)
        if body is None:
            return None
        x, y, a = body.pose
        base = geo.transform(t.at, (x, y), a)
        length = self.thrust_length(t.max_force)
        return base, (base[0] - length * math.cos(a + t.dir), base[1] - length * math.sin(a + t.dir))

    def _ordered_bodies(self, scene: Scene) -> list[BodyItem]:
        """위에 그린 것부터: 에이전트 물체, 세계 물체, 생성기 물체."""
        agent = [b for b in scene.bodies if b.owner == "agent" and not b.generated and self.visible_ref(b.ref)]
        world = [b for b in scene.bodies if b.owner != "agent" and not b.generated]
        gen = [b for b in scene.bodies if b.generated]
        return agent[::-1] + world[::-1] + gen

    def pick(self, w: Point) -> Pick | None:
        """세계 점 w에서 누른 것. 작은 표시(손잡이, 추진기, 관절, 표지)가 물체보다 먼저다."""
        scene = self.scene()
        tol = self._tol()
        body = self.selected_body()
        if body is not None and not body.place_by:
            if self._near(self.rotate_handle(body), w, HANDLE_PX):
                return Pick(body.ref, "rotate", center=(body.pose[0], body.pose[1]))
        if body is not None and body.motion is not None:
            for k, q in enumerate(body.path_points()):
                if self._near(q, w, HANDLE_PX):
                    return Pick(body.ref, "path_pt", k, center=q)
        sp, sd = self.spawn_handles()
        screen = self.world_to_screen(*w)
        for item, rect in self.badge_rects():
            if rect.contains(screen):
                return Pick(item.ref, "badge")
        for t in scene.thrusters:
            pts = self.thruster_points(t)
            if pts is None or not self.visible_ref(t.ref):
                continue
            base, tip = pts
            if self._near(tip, w, HANDLE_PX):
                return Pick(t.ref, "thruster_tip", center=base)
            if self._near(base, w, HANDLE_PX) or geo.dist_point_segment(w, base, tip) <= tol * 0.7:
                return Pick(t.ref, "thruster", center=base)
        for j in scene.joints:
            if not self.visible_ref(j.ref):
                continue
            if self._near(j.pa, w, HANDLE_PX):
                return Pick(j.ref, "joint", 0, center=j.pa)
            if j.kind == "spring" and self._near(j.pb, w, HANDLE_PX):
                return Pick(j.ref, "joint", 1, center=j.pb)
        for r in scene.rays:
            if self.visible_ref(r.ref) and self._near(r.origin, w, HANDLE_PX * 0.75):
                return Pick(r.ref, "ray", center=r.origin)
        for c in scene.checkpoints:
            if not c.editable:
                continue
            if self._near(c.a, w, HANDLE_PX):
                return Pick(c.ref, "cp_end", c.index * 2)
            if self._near(c.b, w, HANDLE_PX):
                return Pick(c.ref, "cp_end", c.index * 2 + 1)
        if self._near(sd, w, HANDLE_PX):
            return Pick(None, "spawn_dir")
        for c in scene.checkpoints:
            if c.editable and geo.dist_point_segment(w, c.a, c.b) <= tol:
                return Pick(c.ref, "cp_line", c.index)
        for b in self._ordered_bodies(scene):
            if not b.generated and b.contains(w, tol):
                return Pick(b.ref, "body", center=(b.pose[0], b.pose[1]))
        if self._near(sp, w, HANDLE_PX):  # 시작 위치 표시는 물체 아래에 있다
            return Pick(None, "spawn")
        for r in scene.regions[::-1]:
            if not r.generated and geo.point_in_polygon(w, r.points):
                return Pick(r.ref, "region")
        for r in scene.regions[::-1]:
            if r.generated and geo.point_in_polygon(w, r.points):
                return Pick(r.ref, "gen")
        for c in scene.checkpoints:
            if not c.editable and geo.dist_point_segment(w, c.a, c.b) <= tol:
                return Pick(c.ref, "gen")
        for b in scene.bodies:
            if b.generated and b.contains(w, tol):
                return Pick(b.ref, "gen")
        return None

    def pick_body(self, w: Point, exclude: Iterable[str] = ()) -> BodyItem | None:
        """w 아래 물체(에이전트 물체가 먼저, 생성기 물체 포함)."""
        ex = set(exclude)
        tol = self._tol()
        for b in self._ordered_bodies(self.scene()):
            if b.id not in ex and b.contains(w, tol):
                return b
        return None

    def pick_target(self, w: Point, exclude: Iterable[str] = ()) -> str | None:
        """목표 위치 센서의 대상: 움직이는 물체, 아니면 영역(생성기 패드 포함)."""
        ex = set(exclude)
        tol = self._tol()
        scene = self.scene()
        for b in self._ordered_bodies(scene):
            if b.dynamic and not b.generated and b.id not in ex and b.contains(w, tol):
                return b.id
        for r in scene.regions[::-1]:
            if geo.point_in_polygon(w, r.points):
                return r.id
        return None

    def pick_joint(self, w: Point) -> str | None:
        best: tuple[float, str] | None = None
        for j in self.scene().joints:
            if not self.visible_ref(j.ref):
                continue
            for p in (j.pa, j.pb):
                d = math.dist(p, w) * self.scale
                if d <= HANDLE_PX * 1.5 and (best is None or d < best[0]):
                    best = (d, j.ref[1])
        return best[1] if best else None

    def pick_tank(self, w: Point) -> str | None:
        screen = self.world_to_screen(*w)
        for item, rect in self.badge_rects():
            if item.kind == "fuel_tank" and rect.contains(screen):
                return item.ref[1]
        return None

    def refs_in_rect(self, rect: QRectF) -> list[Ref]:
        """화면 사각형 안에 다 들어온 부품(사각형 선택)."""
        scene = self.scene()
        out: list[Ref] = []

        def inside(*pts: Point) -> bool:
            return all(rect.contains(self.world_to_screen(*p)) for p in pts)

        for b in scene.bodies:
            if not b.generated and self.visible_ref(b.ref) and inside((b.bbox[0], b.bbox[1]), (b.bbox[2], b.bbox[3])):
                out.append(b.ref)
        for r in scene.regions:
            if not r.generated and inside((r.bbox[0], r.bbox[1]), (r.bbox[2], r.bbox[3])):
                out.append(r.ref)
        sets: dict[Ref, bool] = {}
        for c in scene.checkpoints:
            if c.editable:
                sets[c.ref] = sets.get(c.ref, True) and inside(c.a, c.b)
        out += [r for r, ok in sets.items() if ok]
        for j in scene.joints:
            if self.visible_ref(j.ref) and inside(j.pa):
                out.append(j.ref)
        for t in scene.thrusters:
            pts = self.thruster_points(t)
            if pts is not None and self.visible_ref(t.ref) and inside(pts[0]):
                out.append(t.ref)
        for item, brect in self.badge_rects():
            if rect.contains(brect) and item.ref not in out:
                out.append(item.ref)
        return list(dict.fromkeys(out))

    # ---- 표지(센서·구동기) 배치 ----
    def _badge_font(self) -> QFont:
        f = QFont(self.font())
        f.setPixelSize(10)
        return f

    def badge_rects(self) -> list[tuple[BadgeItem, QRectF]]:
        scene = self.scene()
        key = (id(scene), self.center, self.scale, self.width(), self.height(), self.show_absent)
        if self._badge_cache is not None and self._badge_cache[0] == key:
            return self._badge_cache[1]
        fm = QFontMetricsF(self._badge_font())
        joints = {j.ref[1]: j for j in scene.joints}
        # 물체 상자 오른쪽 위에 쌓는다. 추진기 화살표가 상자 밖으로 나가면 그 바깥에.
        corner: dict[str, Point] = {}
        for b in scene.bodies:
            corner[b.id] = (b.bbox[2], b.bbox[3])
        for t in scene.thrusters:
            pts = self.thruster_points(t)
            if pts is not None and t.body in corner:
                x, y = corner[t.body]
                corner[t.body] = (max(x, pts[0][0], pts[1][0]), max(y, pts[0][1], pts[1][1]))
        stacks: dict[Any, int] = {}
        out: list[tuple[BadgeItem, QRectF]] = []
        sx, sy, _ = scene.spawn
        for item in scene.badges:
            if not self.visible_ref(item.ref):
                continue
            body = scene.body(item.body) if item.body else None
            if item.joint is not None and item.joint in joints:
                anchor = self.world_to_screen(*joints[item.joint].pa) + QPointF(8, -18)
                group: Any = ("j", item.joint)
            elif body is not None:
                lo = self.world_to_screen(body.bbox[0], body.bbox[1])
                hi = self.world_to_screen(body.bbox[2], body.bbox[3])
                if math.hypot(hi.x() - lo.x(), hi.y() - lo.y()) < BADGE_MIN_BODY_PX:
                    continue  # 화면에서 너무 작은 물체: 확대하면 보인다
                anchor = self.world_to_screen(*corner[body.id]) + QPointF(4, -2)
                group = ("b", body.id)
            else:
                anchor = self.world_to_screen(sx, sy) + QPointF(12, 10)
                group = "spawn"
            k = stacks.get(group, 0)
            stacks[group] = k + 1
            text = BADGE_TEXT.get(item.kind, item.kind)
            w = fm.horizontalAdvance(text) + 8
            out.append((item, QRectF(anchor.x(), anchor.y() + k * 15, w, 14)))
        self._badge_cache = (key, out)
        return out

    # ------------------------------------------------------------------
    # 그리기
    # ------------------------------------------------------------------
    def _q(self, p: Point) -> QPointF:
        return self.world_to_screen(p[0], p[1])

    def _poly(self, pts: Iterable[Point]) -> QPolygonF:
        return QPolygonF([self._q(p) for p in pts])

    def _highlight(self, ref: Ref) -> tuple[QColor | None, bool]:
        """(문제 색, 선택됨)"""
        sev = self._issue_refs.get(ref)
        color = COLOR_ERROR if sev == "error" else (COLOR_WARNING if sev == "warning" else None)
        return color, ref in self.doc.selection

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        scene = self._paint_scene()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._playing or self.preview_time > 0.0:
            # 재생 중이거나 미리보기 시각을 옮겨 둔 동안: 움직이지 않는 그림은 한 번만 그려 두고 다시 쓴다
            p.drawImage(0, 0, self._static_image(scene))
        else:
            self._paint_static(p, scene)
        self._paint_moving(p, scene)
        self._paint_badges(p)  # 표지·손잡이는 움직이는 물체 위에(작고 싸다)
        self._paint_handles(p)
        self._tool.overlay(p)
        self._paint_drop(p)
        self._paint_owner_cursor(p)
        self._paint_hud(p, scene)
        p.end()

    def _paint_static(self, p: QPainter, scene: Scene) -> None:
        """움직이지 않는 것(정해진 움직임 물체는 길 점선만, 표지와 회전 손잡이는 paintEvent가 맨 위에)."""
        p.fillRect(self.rect(), COLOR_BG)
        self._paint_grid(p)
        self._paint_bounds(p, scene)
        self._paint_regions(p, scene)
        self._paint_terrain(p, scene)
        self._paint_checkpoints(p, scene)
        self._paint_paths(p, scene)
        for b in [x for x in scene.bodies if not x.generated and x.owner != "agent" and not self._terrain_like(x)
                  and x.motion is None]:
            self._paint_body(p, b)
        self.drawn_optional = {}
        for b in [x for x in scene.bodies if not x.generated and x.owner == "agent" and self.visible_ref(x.ref)]:
            self._paint_body(p, b)
        self._paint_rays(p, scene)
        self._paint_targets(p, scene)
        self._paint_joints(p, scene)
        self._paint_thrusters(p, scene)
        self._paint_jumps(p, scene)
        self._paint_moves(p, scene)
        self._paint_balances(p, scene)
        self._paint_spawn(p, scene)

    def _static_key(self, scene: Scene) -> Any:
        return (id(scene), self.center, self.scale, self.width(), self.height(), self.devicePixelRatioF(),
                tuple(self.doc.selection), id(self._issue_refs), self._tool_id, self.snap_step, self.show_absent,
                self.structure_highlight)

    def _static_image(self, scene: Scene) -> QImage:
        """움직이지 않는 그림을 그린 이미지(같은 장면·카메라·선택이면 다시 그리지 않는다)."""
        key = self._static_key(scene)
        if self._static_cache is not None and self._static_cache[0] == key:
            return self._static_cache[1]
        dpr = self.devicePixelRatioF()
        img = QImage(max(1, round(self.width() * dpr)), max(1, round(self.height() * dpr)),
                     QImage.Format.Format_ARGB32_Premultiplied)
        img.setDevicePixelRatio(dpr)
        q = QPainter(img)
        q.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        q.setFont(self.font())
        self._paint_static(q, scene)
        q.end()
        self._static_cache = (key, img)
        return img

    def _paint_grid(self, p: QPainter) -> None:
        w, h = self.width(), self.height()
        base = self.snap_step if self.snap_step > 0 else 1.0
        minor = base
        for k in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
            minor = base * k
            if minor * self.scale >= 9:
                break
        major = minor * 5
        x0, y1 = self.screen_to_world(QPointF(0, 0))
        x1, y0 = self.screen_to_world(QPointF(w, h))
        for step, color in ((minor, COLOR_GRID), (major, COLOR_GRID_MAJOR)):
            p.setPen(_pen(color, 1.0))
            i0, i1 = math.floor(x0 / step), math.ceil(x1 / step)
            if i1 - i0 < 400:
                for i in range(i0, i1 + 1):
                    sx = self.world_to_screen(i * step, 0).x()
                    p.drawLine(QPointF(sx, 0), QPointF(sx, h))
            j0, j1 = math.floor(y0 / step), math.ceil(y1 / step)
            if j1 - j0 < 400:
                for j in range(j0, j1 + 1):
                    sy = self.world_to_screen(0, j * step).y()
                    p.drawLine(QPointF(0, sy), QPointF(w, sy))
        o = self.world_to_screen(0.0, 0.0)
        p.setPen(_pen(COLOR_AXIS, 1.0))
        p.drawLine(QPointF(o.x(), 0), QPointF(o.x(), h))
        p.drawLine(QPointF(0, o.y()), QPointF(w, o.y()))

    def _paint_bounds(self, p: QPainter, scene: Scene) -> None:
        x0, y0, x1, y1 = scene.bounds
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            return
        p.setPen(_pen(COLOR_BOUNDS, 1.0, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(self._q((x0, y1)), self._q((x1, y0))))

    def _outline(self, p: QPainter, ref: Ref, draw: Any, under: bool) -> None:
        """문제 윤곽(굵게, 아래)과 선택 윤곽(위). 구조 탭 강조는 선택 색 점선(v2.1)."""
        color, selected = self._highlight(ref)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if under and color is not None:
            p.setPen(_pen(color, 5.0))
            draw()
        if not under and selected:
            p.setPen(_pen(COLOR_SELECT, 2.0))
            draw()
        elif not under and self.linked(ref):
            p.setPen(_pen(COLOR_SELECT, 2.5, Qt.PenStyle.DashLine))
            draw()

    def _paint_regions(self, p: QPainter, scene: Scene) -> None:
        font = self._badge_font()
        for r in scene.regions:
            poly = self._poly(r.points)
            self._outline(p, r.ref, lambda poly=poly: p.drawPolygon(poly), True)
            p.setPen(_pen(REGION_EDGE.get(r.kind, REGION_EDGE["zone"]), 1.0))
            p.setBrush(QBrush(REGION_FILL.get(r.kind, REGION_FILL["zone"])))
            p.drawPolygon(poly)
            if r.kind == "pad" and len(r.points) >= 2:
                p.setPen(_pen(COLOR_PAD, 4.0))
                p.drawLine(self._q(r.points[0]), self._q(r.points[1]))
            self._outline(p, r.ref, lambda poly=poly: p.drawPolygon(poly), False)
            if not r.generated:
                p.setFont(font)
                p.setPen(COLOR_LABEL)
                label = self._label(r.ref)
                p.drawText(self._q((r.bbox[0], r.bbox[3])) + QPointF(3, 12), label)

    def _label(self, ref: Ref) -> str:
        item = self.doc.item(ref) or {}
        return str(item.get("label") or ref[1])

    def _segment_chains(self, b: BodyItem) -> list[list[Point]]:
        chains: list[list[Point]] = []
        for a, c, _r in b.segments:
            if chains and math.dist(chains[-1][-1], a) < 1e-9:
                chains[-1].append(c)
            else:
                chains.append([a, c])
        return chains

    @staticmethod
    def _terrain_like(b: BodyItem) -> bool:
        """생성기 지형·벽, 선(체인)만으로 된 정적 세계 물체: 선으로 그린다(지형 태그면 아래를 채운다)."""
        return b.generated or (b.owner == "world" and not b.dynamic and bool(b.segments) and not b.polys
                               and not b.circles)

    def _paint_terrain(self, p: QPainter, scene: Scene) -> None:
        bottom = scene.bounds[1] - 50.0 if math.isfinite(scene.bounds[1]) else -100.0
        for b in scene.bodies:
            if not self._terrain_like(b) or b.motion is not None:
                continue
            chains = self._segment_chains(b)
            fill = "terrain" in b.tags
            draw_all = lambda chains=chains: [p.drawPolyline(self._poly(ch)) for ch in chains]  # noqa: E731
            self._outline(p, b.ref, draw_all, True)
            if fill:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(COLOR_TERRAIN_FILL))
                for ch in chains:
                    if len(ch) >= 2 and ch[0][0] < ch[-1][0]:
                        p.drawPolygon(self._poly([*ch, (ch[-1][0], bottom), (ch[0][0], bottom)]))
            radius = max((r for _a, _c, r in b.segments), default=0.05)
            p.setPen(_pen(COLOR_TERRAIN_LINE, max(1.5, 2 * radius * self.scale)))
            p.setBrush(Qt.BrushStyle.NoBrush)
            draw_all()
            self._outline(p, b.ref, draw_all, False)

    def _paint_checkpoints(self, p: QPainter, scene: Scene) -> None:
        font = self._badge_font()
        p.setFont(font)
        for c in scene.checkpoints:
            a, b = self._q(c.a), self._q(c.b)
            self._outline(p, c.ref, lambda a=a, b=b: p.drawLine(a, b), True)
            if c.index == 0:
                p.setPen(_pen(COLOR_START_LINE, 2.5))
            else:
                p.setPen(_pen(COLOR_CHECKPOINT if c.editable else COLOR_CHECKPOINT_GEN, 2.0 if c.editable else 1.0))
            p.drawLine(a, b)
            self._outline(p, c.ref, lambda a=a, b=b: p.drawLine(a, b), False)
            if c.editable:
                p.setPen(COLOR_CHECKPOINT)
                p.drawText(a + QPointF(4, -4), str(c.index))
                p.setBrush(QBrush(COLOR_CHECKPOINT))
                p.drawEllipse(a, 2.5, 2.5)
                p.drawEllipse(b, 2.5, 2.5)

    def _paint_body(self, p: QPainter, b: BodyItem) -> None:
        if b.owner == "agent":
            fill = COLOR_LEG if "leg" in b.tags else COLOR_AGENT
        else:
            fill = COLOR_WORLD_BODY if b.dynamic or b.kinematic else COLOR_STATIC
        x, y, a = b.pose

        def outline() -> None:
            for poly in b.polys:
                p.drawPolygon(self._poly(poly))
            for c, r in b.circles:
                p.drawEllipse(self._q(c), r * self.scale, r * self.scale)
            for s0, s1, _r in b.segments:
                p.drawLine(self._q(s0), self._q(s1))
            if b.placeholder:
                p.drawEllipse(self._q((x, y)), 0.3 * self.scale + 4, 0.3 * self.scale + 4)

        faded = self._fade(p, b.ref)
        self._outline(p, b.ref, outline, True)
        if b.optional:  # v2.1 없어질 수 있는 물체: 점선 테두리
            p.setPen(_pen(COLOR_AGENT_EDGE, 1.5, Qt.PenStyle.DashLine))
        else:
            p.setPen(_pen(COLOR_AGENT_EDGE, 1.0))
        p.setBrush(QBrush(fill))
        for poly in b.polys:
            p.drawPolygon(self._poly(poly))
        for c, r in b.circles:
            sc = self._q(c)
            p.drawEllipse(sc, r * self.scale, r * self.scale)
            edge = self._q((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
            p.drawLine(sc, edge)
        if b.segments and not b.generated:
            radius = max((r for _a, _c, r in b.segments), default=0.05)
            p.setPen(_pen(COLOR_TERRAIN_LINE if not b.dynamic else fill, max(1.5, 2 * radius * self.scale)))
            for s0, s1, _r in b.segments:
                p.drawLine(self._q(s0), self._q(s1))
        if b.placeholder:
            p.setPen(_pen(fill, 1.5, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(self._q((x, y)), 0.3 * self.scale + 4, 0.3 * self.scale + 4)
        self._outline(p, b.ref, outline, False)
        if b.wing_les:  # 날개 모양의 앞전에 점
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(COLOR_JOINT))
            for le in b.wing_les:
                p.drawEllipse(self._q(geo.transform(le, (x, y), a)), 2.5, 2.5)
        if b.id in self._scene_aero():
            self._paint_wind(p, b)
        if b.ref in self.doc.selection:
            o = self._q((x, y))
            p.setPen(_pen(COLOR_SELECT, 1.0))
            p.drawLine(o + QPointF(-4, 0), o + QPointF(4, 0))
            p.drawLine(o + QPointF(0, -4), o + QPointF(0, 4))
        if b.optional:
            self._paint_optional_badge(p, b)
            self.drawn_optional[b.id] = "absent" if b.absent else "present"
        self._unfade(p, faded)

    def _paint_optional_badge(self, p: QPainter, b: BodyItem) -> None:
        """없어질 수 있는 물체의 「?」 표지: 물체 상자 왼쪽 위(오른쪽 위는 센서·구동기 표지 자리)."""
        lo = self.world_to_screen(b.bbox[0], b.bbox[1])
        hi = self.world_to_screen(b.bbox[2], b.bbox[3])
        if math.hypot(hi.x() - lo.x(), hi.y() - lo.y()) < BADGE_MIN_BODY_PX:
            return
        c = QPointF(lo.x() - 2, hi.y() - 2)
        p.setPen(_pen(COLOR_BADGE_TEXT, 1.0))
        p.setBrush(QBrush(COLOR_BADGE_SENSOR))
        p.drawEllipse(c, 7.0, 7.0)
        p.setFont(self._badge_font())
        p.drawText(QRectF(c.x() - 7, c.y() - 7, 14, 14), Qt.AlignmentFlag.AlignCenter, OPTIONAL_BADGE)

    def _scene_aero(self) -> set[str]:
        return self._scene.aero_bodies if self._scene is not None else set()

    def _paint_wind(self, p: QPainter, b: BodyItem) -> None:
        """공기역학 물체: 물체 상자 앞(바람이 오는 쪽, 왼쪽)에 작은 바람 줄기 셋."""
        lo, hi = self._q((b.bbox[0], b.bbox[1])), self._q((b.bbox[2], b.bbox[3]))
        cy, x1 = (lo.y() + hi.y()) / 2.0, lo.x() - 5.0
        p.setPen(_pen(COLOR_WIND, 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        for k in (-1, 0, 1):
            y = cy + 6.0 * k
            p.drawPolyline(QPolygonF([QPointF(x1 - 14.0, y), QPointF(x1 - 9.0, y - 1.5), QPointF(x1 - 4.0, y + 1.0),
                                      QPointF(x1, y)]))

    # ---- 정해진 움직임 (v2) ----
    def _paint_paths(self, p: QPainter, scene: Scene) -> None:
        """길 점선(진행 방향 화살촉, 방식 기호)과 흔들기 양 끝 자세(옅게). 움직이지 않으므로 정적 그림에 들어간다."""
        font = self._badge_font()
        for b in scene.moving():
            mo = b.motion
            assert mo is not None
            if b.row is not None:
                continue  # 장애물 줄: 길 대신 「→ 장애물 n개」(움직이는 그림 쪽)
            if mo.kind == "wave":
                period = mo.period
                for frac in (0.25, 0.75):
                    t = (period * (frac - mo.phase)) % period
                    polys, circles, _segs = shapes_at(b.shapes, pose_at(mo, t))
                    p.setPen(_pen(COLOR_GHOST, 1.0, Qt.PenStyle.DashLine))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    for poly in polys:
                        p.drawPolygon(self._poly(poly))
                    for c, r in circles:
                        p.drawEllipse(self._q(c), r * self.scale, r * self.scale)
                continue
            ox, oy = mo.origin
            pts = [self._q((ox + x, oy + y)) for x, y in mo.points]
            if len(pts) < 2:
                continue
            p.setPen(_pen(COLOR_PATH, 1.5, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF(pts))
            seg = next(((a, c) for a, c in zip(pts, pts[1:], strict=False)
                        if math.hypot(c.x() - a.x(), c.y() - a.y()) > 12), None)
            if seg is not None:  # 첫 구간 가운데에 진행 방향 화살촉
                a, c = seg
                mid = QPointF((a.x() + c.x()) / 2, (a.y() + c.y()) / 2)
                ang = math.atan2(c.y() - a.y(), c.x() - a.x())
                p.setPen(_pen(COLOR_PATH, 1.5))
                for side in (-1, 1):
                    a2 = ang + math.pi + side * 0.5
                    p.drawLine(mid, mid + QPointF(7 * math.cos(a2), 7 * math.sin(a2)))
            end = pts[b.path_count - 1] if 0 < b.path_count <= len(pts) else pts[-1]
            p.setFont(font)
            p.setPen(COLOR_PATH)
            p.drawText(end + QPointF(6, -6), MODE_SYMBOLS.get(mo.mode, ""))

    def _paint_moving(self, p: QPainter, scene: Scene) -> None:
        """정해진 움직임 물체를 미리보기 시각의 자세로 그린다(재생 중에는 이것만 다시 그린다)."""
        t = self.preview_time
        poses: dict[str, Pose] = {}
        font = None
        for b in scene.moving():
            mo = b.motion
            assert mo is not None
            pose = pose_at(mo, t)
            poses[b.id] = pose
            shapes = b.shapes
            hidden = b.row_hidden
            if b.row is not None and t > 0.0:
                shapes, hidden = row_window(b.row, mo, t)
            polys, circles, segments = shapes_at(shapes, pose)
            color, selected = self._highlight(b.ref)

            def outline(polys: list[list[Point]] = polys, circles: list[tuple[Point, float]] = circles) -> None:
                for poly in polys:
                    p.drawPolygon(self._poly(poly))
                for c, r in circles:
                    p.drawEllipse(self._q(c), r * self.scale, r * self.scale)

            self._outline(p, b.ref, outline, True)
            if t > 0.0 and b.row is None:  # 시작 자세를 옅게 남긴다(누르고 고치는 자리)
                p.setPen(_pen(COLOR_GHOST, 1.0, Qt.PenStyle.DashLine))
                p.setBrush(Qt.BrushStyle.NoBrush)
                for poly in b.polys:
                    p.drawPolygon(self._poly(poly))
            p.setPen(_pen(COLOR_AGENT_EDGE, 1.0))
            p.setBrush(QBrush(COLOR_WORLD_BODY))
            for poly in polys:
                p.drawPolygon(self._poly(poly))
            for c, r in circles:
                p.drawEllipse(self._q(c), r * self.scale, r * self.scale)
            for s0, s1, r in segments:
                p.setPen(_pen(COLOR_WORLD_BODY, max(1.5, 2 * r * self.scale)))
                p.drawLine(self._q(s0), self._q(s1))
            self._outline(p, b.ref, outline, False)
            if b.row is not None and hidden > 0 and polys:
                font = font or self._badge_font()
                p.setFont(font)
                p.setPen(COLOR_LABEL)
                far = max((q for poly in polys for q in poly), key=lambda q: q[0])
                near = min(q[1] for poly in polys for q in poly)
                p.drawText(self._q((far[0], near)) + QPointF(8, -4), S.EDC_OBSTACLES_MORE.format(count=hidden))
            if selected and b.row is None and self._tool_id == "select":
                p.setPen(_pen(COLOR_SELECT, 1.0))
                p.setBrush(QBrush(COLOR_BG))
                for q in b.path_points():
                    sq = self._q(q)
                    p.drawRect(QRectF(sq.x() - 3.5, sq.y() - 3.5, 7, 7))
        self.drawn_poses = poses

    def _paint_jumps(self, p: QPainter, scene: Scene) -> None:
        """점프 추진기: 물체 위쪽에 뛰는 방향 화살표(이름은 표지에)."""
        for j in scene.jumps:
            body = scene.body(j.body)
            if body is None or not self.visible_ref(j.ref):
                continue
            p.setOpacity(ABSENT_OPACITY if self._absent_ref(j.ref) else 1.0)
            _color, selected = self._highlight(j.ref)
            selected = selected or self.linked(j.ref)
            top = self._q(((body.bbox[0] + body.bbox[2]) / 2.0, body.bbox[3]))
            base = top + QPointF(0, -4)
            tip = base + QPointF(26 * math.cos(j.dir), -26 * math.sin(j.dir))
            main = COLOR_SELECT if selected else COLOR_FLAME
            p.setPen(_pen(main, 2.0))
            p.drawLine(base, tip)
            ang = math.atan2(tip.y() - base.y(), tip.x() - base.x())
            for side in (-1, 1):
                a2 = ang + math.pi + side * 0.5
                p.drawLine(tip, tip + QPointF(7 * math.cos(a2), 7 * math.sin(a2)))
        p.setOpacity(1.0)

    def _arrow(self, p: QPainter, base: QPointF, ang: float, length: float) -> None:
        """화면 각도 ang(라디안, 화면 y 아래) 쪽으로 base에서 length px 화살표."""
        tip = base + QPointF(length * math.cos(ang), length * math.sin(ang))
        p.drawLine(base, tip)
        for side in (-1, 1):
            a2 = ang + math.pi + side * 0.5
            p.drawLine(tip, tip + QPointF(5 * math.cos(a2), 5 * math.sin(a2)))

    def _mark_color(self, ref: Ref, base: QColor) -> QColor:
        color, selected = self._highlight(ref)
        if color is not None:
            return color
        return COLOR_SELECT if selected or self.linked(ref) else base

    def _paint_moves(self, p: QPainter, scene: Scene) -> None:
        """밀기(이동) 표지(6.2.0): xy는 위·아래·왼쪽·오른쪽, x는 왼쪽·오른쪽(세계 축), heading은 물체 앞(+x)과 도는 호."""
        for m in scene.moves:
            body = scene.body(m.body)
            if body is None or not self.visible_ref(m.ref):
                continue
            p.setOpacity(ABSENT_OPACITY if self._absent_ref(m.ref) else 1.0)
            c = self._q((body.pose[0], body.pose[1]))
            p.setPen(_pen(self._mark_color(m.ref, COLOR_MOVE), 2.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            if m.mode == "heading":
                self._arrow(p, c, -body.pose[2], MARK_PX * 1.3)
                p.drawArc(QRectF(c.x() - MARK_PX * 0.7, c.y() - MARK_PX * 0.7, MARK_PX * 1.4, MARK_PX * 1.4),
                          int((math.degrees(body.pose[2]) + 60) * 16), 240 * 16)
            else:
                angles = (0.0, math.pi) if m.mode == "x" else (0.0, math.pi / 2, math.pi, -math.pi / 2)
                for a in angles:
                    self._arrow(p, c + QPointF(3 * math.cos(a), 3 * math.sin(a)), a, MARK_PX)
        p.setOpacity(1.0)

    def _paint_balances(self, p: QPainter, scene: Scene) -> None:
        """균형 보조 표지(6.2.0): 물체가 세워지는 각도(target) 쪽 점선과 그 양쪽의 둥근 화살표(되돌리는 힘)."""
        for b in scene.balances:
            body = scene.body(b.body)
            if body is None or not self.visible_ref(b.ref):
                continue
            p.setOpacity(ABSENT_OPACITY if self._absent_ref(b.ref) else 1.0)
            c = self._q((body.pose[0], body.pose[1]))
            up = -(math.pi / 2 + b.target)  # 화면 각도: 세계 각도 target만큼 기운 위쪽
            color = self._mark_color(b.ref, COLOR_BALANCE)
            p.setPen(_pen(color, 1.5, Qt.PenStyle.DashLine))
            p.drawLine(c, c + QPointF(MARK_PX * 1.8 * math.cos(up), MARK_PX * 1.8 * math.sin(up)))
            p.setPen(_pen(color, 2.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            r = MARK_PX * 1.2
            rect = QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
            deg = math.degrees(math.pi / 2 + b.target)  # Qt 호 각도: 반시계, 3시 방향 0
            for side in (-1, 1):
                p.drawArc(rect, int((deg + side * 18) * 16), int(side * 40 * 16))
                end = math.radians(deg + side * 58)
                tip = c + QPointF(r * math.cos(end), -r * math.sin(end))
                back = -end + (math.pi / 2 if side > 0 else -math.pi / 2)  # 호를 따라 위쪽(target) 쪽으로 되돌아가는 방향
                for k in (-1, 1):
                    a2 = back + k * 0.5
                    p.drawLine(tip, tip + QPointF(5 * math.cos(a2), 5 * math.sin(a2)))
        p.setOpacity(1.0)

    def _paint_joints(self, p: QPainter, scene: Scene) -> None:
        for j in scene.joints:
            if not self.visible_ref(j.ref):
                continue
            p.setOpacity(ABSENT_OPACITY if self._absent_ref(j.ref) else 1.0)
            pa, pb = self._q(j.pa), self._q(j.pb)
            color, selected = self._highlight(j.ref)
            selected = selected or self.linked(j.ref)
            p.setBrush(Qt.BrushStyle.NoBrush)
            thin = QColor(COLOR_JOINT)
            thin.setAlpha(110)
            p.setPen(_pen(thin, 1.0, Qt.PenStyle.DotLine))
            for bid, q in ((j.a, pa), (j.b, pb)):
                body = scene.body(bid)
                if body is not None:
                    p.drawLine(self._q((body.pose[0], body.pose[1])), q)
            if color is not None:
                p.setPen(_pen(color, 3.0))
                p.drawEllipse(pa, 9.0, 9.0)
            main = COLOR_SELECT if selected else COLOR_JOINT
            p.setPen(_pen(main, 2.0 if selected else 1.5))
            if j.kind == "spring":
                self._zigzag(p, pa, pb)
                p.drawEllipse(pa, 3.5, 3.5)
                p.drawEllipse(pb, 3.5, 3.5)
                continue
            if math.hypot(pa.x() - pb.x(), pa.y() - pb.y()) > 1.5:
                p.setPen(_pen(main, 1.0, Qt.PenStyle.DashLine))
                p.drawLine(pa, pb)
                p.drawEllipse(pb, 3.0, 3.0)
                p.setPen(_pen(main, 2.0 if selected else 1.5))
            if j.kind == "weld":
                p.drawRect(QRectF(pa.x() - 4.5, pa.y() - 4.5, 9, 9))
            else:
                p.drawEllipse(pa, 4.5, 4.5)
                p.setBrush(QBrush(main))
                p.drawEllipse(pa, 1.5, 1.5)
                p.setBrush(Qt.BrushStyle.NoBrush)
                if j.kind == "rot_spring":
                    p.drawArc(QRectF(pa.x() - 8, pa.y() - 8, 16, 16), 30 * 16, 240 * 16)
        p.setOpacity(1.0)

    def _zigzag(self, p: QPainter, a: QPointF, b: QPointF) -> None:
        dx, dy = b.x() - a.x(), b.y() - a.y()
        length = math.hypot(dx, dy)
        if length < 4:
            p.drawLine(a, b)
            return
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux
        n = max(4, min(16, int(length / 6)))
        pts = [a]
        for i in range(1, n):
            t = i / n
            s = 4.0 if i % 2 else -4.0
            pts.append(QPointF(a.x() + dx * t + nx * s, a.y() + dy * t + ny * s))
        pts.append(b)
        p.drawPolyline(QPolygonF(pts))

    def _paint_thrusters(self, p: QPainter, scene: Scene) -> None:
        for t in scene.thrusters:
            pts = self.thruster_points(t)
            if pts is None or not self.visible_ref(t.ref):
                continue
            p.setOpacity(ABSENT_OPACITY if self._absent_ref(t.ref) else 1.0)
            base, tail = self._q(pts[0]), self._q(pts[1])
            color, selected = self._highlight(t.ref)
            selected = selected or self.linked(t.ref)
            if color is not None:
                p.setPen(_pen(color, 6.0))
                p.drawLine(base, tail)
            main = COLOR_SELECT if selected else COLOR_FLAME
            p.setPen(_pen(main, 2.5))
            p.drawLine(tail, base)
            ang = math.atan2(base.y() - tail.y(), base.x() - tail.x())
            for side in (-1, 1):
                a2 = ang + math.pi + side * 0.45
                p.drawLine(base, base + QPointF(8 * math.cos(a2), 8 * math.sin(a2)))
            p.setBrush(QBrush(main))
            p.drawEllipse(tail, 3.0, 3.0)
        p.setOpacity(1.0)

    def _paint_rays(self, p: QPainter, scene: Scene) -> None:
        for r in scene.rays:
            if not self.visible_ref(r.ref):
                continue
            p.setOpacity(ABSENT_OPACITY if self._absent_ref(r.ref) else 1.0)
            color, selected = self._highlight(r.ref)
            selected = selected or self.linked(r.ref)
            line = QColor(COLOR_SELECT if selected else COLOR_RAY)
            if selected:
                line.setAlpha(170)
            p.setPen(_pen(line, 1.0))
            for a, b, alpha in r.lines:
                p.drawLine(self._q(a), self._q(b))
                if alpha < 1.0:
                    p.setBrush(QBrush(line))
                    p.drawEllipse(self._q(b), 2.0, 2.0)
            o = self._q(r.origin)
            if color is not None:
                p.setPen(_pen(color, 3.0))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(o, 6.0, 6.0)
            p.setPen(_pen(COLOR_SELECT if selected else COLOR_JOINT, 1.5))
            p.setBrush(QBrush(COLOR_BG))
            p.drawEllipse(o, 3.0, 3.0)
        p.setOpacity(1.0)

    def _paint_targets(self, p: QPainter, scene: Scene) -> None:
        for t in scene.targets:
            body = scene.body(t.body)
            if body is None or not self.visible_ref(t.ref):
                continue
            p.setOpacity(ABSENT_OPACITY if self._absent_ref(t.ref) else 1.0)
            _color, selected = self._highlight(t.ref)
            selected = selected or self.linked(t.ref)
            p.setPen(_pen(COLOR_SELECT if selected else COLOR_TARGET, 1.5, Qt.PenStyle.DashLine))
            a, b = self._q((body.pose[0], body.pose[1])), self._q(t.point)
            p.drawLine(a, b)
            p.setPen(_pen(COLOR_SELECT if selected else COLOR_TARGET, 1.5))
            p.drawLine(b + QPointF(-5, -5), b + QPointF(5, 5))
            p.drawLine(b + QPointF(-5, 5), b + QPointF(5, -5))
        p.setOpacity(1.0)

    def _paint_spawn(self, p: QPainter, scene: Scene) -> None:
        (x, y), d = self.spawn_handles()
        c, tip = self._q((x, y)), self._q(d)
        sp = T.spawn_data(self.doc)
        locked = T.is_formula(sp.get("x")) or T.is_formula(sp.get("y"))
        style = Qt.PenStyle.DashLine if locked else Qt.PenStyle.SolidLine
        p.setPen(_pen(COLOR_SPAWN, 2.0, style))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, 7.0, 7.0)
        p.setPen(_pen(COLOR_SPAWN, 1.0))
        p.drawLine(c + QPointF(-11, 0), c + QPointF(11, 0))
        p.drawLine(c + QPointF(0, -11), c + QPointF(0, 11))
        p.setPen(_pen(COLOR_SPAWN, 2.0))
        p.drawLine(c, tip)
        ang = math.atan2(tip.y() - c.y(), tip.x() - c.x())
        for side in (-1, 1):
            a2 = ang + math.pi + side * 0.5
            p.drawLine(tip, tip + QPointF(7 * math.cos(a2), 7 * math.sin(a2)))
        p.setFont(self._badge_font())
        p.drawText(c + QPointF(-30, 20), S.EDC_SPAWN_LABEL)

    def _paint_badges(self, p: QPainter) -> None:
        p.setFont(self._badge_font())
        for item, rect in self.badge_rects():
            p.setOpacity(ABSENT_OPACITY if self._absent_ref(item.ref) else 1.0)
            color, selected = self._highlight(item.ref)
            selected = selected or self.linked(item.ref)
            bg = COLOR_BADGE_ACTUATOR if item.kind in ACTUATOR_BADGES else COLOR_BADGE_SENSOR
            if color is not None:
                p.setPen(_pen(color, 3.0))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(rect.adjusted(-2, -2, 2, 2), 4, 4)
            p.setPen(_pen(COLOR_SELECT, 1.5) if selected else Qt.PenStyle.NoPen)
            p.setBrush(QBrush(bg))
            p.drawRoundedRect(rect, 3, 3)
            p.setPen(COLOR_BADGE_TEXT)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, BADGE_TEXT.get(item.kind, item.kind))
        p.setOpacity(1.0)

    def _paint_handles(self, p: QPainter) -> None:
        if self._tool_id != "select":
            return
        body = self.selected_body()
        if body is None or body.place_by:
            return
        o, h = self._q((body.pose[0], body.pose[1])), self._q(self.rotate_handle(body))
        item = self.doc.item(body.ref) or {}
        locked = T.is_formula(item.get("angle", 0.0))
        p.setPen(_pen(COLOR_SELECT, 1.0, Qt.PenStyle.DashLine if locked else Qt.PenStyle.SolidLine))
        p.drawLine(o, h)
        p.setBrush(QBrush(COLOR_BG if locked else COLOR_SELECT))
        p.drawEllipse(h, 5.0, 5.0)

    def _paint_hud(self, p: QPainter, scene: Scene) -> None:
        f = QFont(self.font())
        f.setPixelSize(12)
        p.setFont(f)
        if scene.message:
            p.setPen(COLOR_WARNING)
            p.drawText(QRectF(10, 6, self.width() - 20, 40),
                       int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop) | int(Qt.TextFlag.TextWordWrap),
                       scene.message)
        if self.status_text:
            p.setPen(COLOR_HUD)
            p.drawText(QRectF(10, self.height() - 26, self.width() - 20, 20),
                       int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), self.status_text)
        if self.cursor_world is not None:
            x, y = self.cursor_world
            p.setPen(COLOR_LABEL)
            p.drawText(QRectF(10, self.height() - 26, self.width() - 20, 20),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                       S.EDC_COORDS.format(x=x, y=y))

    # ------------------------------------------------------------------
    # 입력
    # ------------------------------------------------------------------
    def _event(self, ev: QMouseEvent) -> CanvasEvent:
        pos = ev.position()
        return CanvasEvent(pos, self.screen_to_world(pos), ev.modifiers(), self._press_alt and self._button_down)

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if ev.button() == Qt.MouseButton.MiddleButton or (ev.button() == Qt.MouseButton.LeftButton and self._space):
            self._pan = (ev.position(), self.center)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if self._drop is not None and ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            if ev.button() == Qt.MouseButton.LeftButton:
                self._swallow_release = True  # 놓기 뒤의 놓음(release)은 도구에 넘기지 않는다
                self.drop_template_at(self.screen_to_world(ev.position()), self._event(ev))
            else:
                self.cancel_template_drop()
                self._right_cancelled = True
            self.update()
            return
        if ev.button() == Qt.MouseButton.RightButton:
            self._right_cancelled = self._tool.busy()
            if self._right_cancelled:
                self._tool.cancel()
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        self._press_alt = bool(ev.modifiers() & Qt.KeyboardModifier.AltModifier)
        self._button_down = True
        self._tool.press(self._event(ev))

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        pos = ev.position()
        if self._pan is not None:
            start, c0 = self._pan
            self.center = (c0[0] - (pos.x() - start.x()) / self.scale, c0[1] + (pos.y() - start.y()) / self.scale)
            self._badge_cache = None
            self.update()
            return
        e = self._event(ev)
        self.cursor_world = e.world
        self.coords_changed.emit(e.world[0], e.world[1])
        self._tool.move(e)
        self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        if self._pan is not None and ev.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._pan = None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._space else self._tool.cursor)
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        if self._swallow_release:
            self._swallow_release = False
            return
        e = self._event(ev)
        self._button_down = False
        self._press_alt = False
        self._tool.release(e)

    def mouseDoubleClickEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        if ev.button() != Qt.MouseButton.LeftButton or self._space:
            return
        self._press_alt = bool(ev.modifiers() & Qt.KeyboardModifier.AltModifier)
        self._button_down = True
        self._tool.double_click(self._event(ev))

    def wheelEvent(self, ev: QWheelEvent) -> None:  # noqa: N802
        dy = ev.angleDelta().y()
        if dy == 0:
            return
        self.zoom_at(ev.position(), 1.2 ** (dy / 120.0))

    def keyPressEvent(self, ev: QKeyEvent) -> None:  # noqa: N802
        if ev.key() == Qt.Key.Key_Space:
            if not ev.isAutoRepeat():
                self._space = True
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if ev.key() == Qt.Key.Key_Escape and self._drop is not None:
            self.cancel_template_drop()
            self.update()
            return
        if self._tool.key(ev):
            self.update()
            return
        if ev.key() == Qt.Key.Key_Escape:
            if self._tool.busy():
                self._tool.cancel()
            else:
                self.set_tool("select")
            return
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev: QKeyEvent) -> None:  # noqa: N802
        if ev.key() == Qt.Key.Key_Space and not ev.isAutoRepeat():
            self._space = False
            self.setCursor(self._tool.cursor)
            return
        super().keyReleaseEvent(ev)

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        self.cursor_world = None
        self.update()
        super().leaveEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        self._badge_cache = None
        super().resizeEvent(event)

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        if self._fit_pending:
            self._dirty = True
        super().showEvent(event)


class PreviewBar(QWidget):
    """캔버스 아래 미리보기 시각 줄(v2 D24): 「미리보기 시각」 슬라이더(0 ~ 세계 최대 시간)와 ▶/⏸.

    정해진 움직임 물체가 있을 때만 보인다(v1 환경 화면은 그대로). 문서가 바뀌면 캔버스가 멈추고 0초로 돌린다.
    """

    def __init__(self, canvas: EditorCanvas, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.canvas = canvas
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(6)
        self.play_button = QToolButton()
        self.play_button.setText("▶")
        self.play_button.setToolTip(S.ED_PREVIEW_PLAY)
        self.play_button.clicked.connect(canvas.toggle_play)
        self.title = QLabel(S.ED_PREVIEW_TIME)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setToolTip(S.ED_PREVIEW_TIME_TIP)
        self.title.setToolTip(S.ED_PREVIEW_TIME_TIP)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider)
        self.time_label = QLabel()
        self.time_label.setMinimumWidth(self.fontMetrics().horizontalAdvance("000.0 / 000.0") + 8)
        self.trial_button = QToolButton()  # 「미리보기 시험 바꾸기」(설계 5.1): 다른 시험의 장애물 줄 등
        self.trial_button.setText(S.ED_PREVIEW_TRIAL.format(n=canvas.preview_trial_index + 1))
        self.trial_button.setToolTip(S.ED_PREVIEW_TRIAL_TIP)
        self.trial_button.clicked.connect(canvas.next_preview_trial)
        lay.addWidget(self.play_button)
        lay.addWidget(self.title)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.time_label)
        lay.addWidget(self.trial_button)
        canvas.preview_changed.connect(self._on_time)
        canvas.playing_changed.connect(self._on_playing)
        canvas.motion_changed.connect(self._on_motion)
        canvas.preview_trial_changed.connect(
            lambda n: self.trial_button.setText(S.ED_PREVIEW_TRIAL.format(n=n + 1)))
        canvas.doc.file_changed.connect(
            lambda: self.trial_button.setText(S.ED_PREVIEW_TRIAL.format(n=canvas.preview_trial_index + 1)))
        canvas.doc.changed.connect(self._sync_range)
        self._sync_range()
        self._on_time(canvas.preview_time)
        self.setVisible(canvas.has_motion)

    def _sync_range(self) -> None:
        end = self.canvas.max_preview_time()
        ticks = max(1, round(end / PREVIEW_STEP))
        if self.slider.maximum() != ticks:
            self.slider.blockSignals(True)
            self.slider.setRange(0, ticks)
            self.slider.blockSignals(False)
        self._on_time(self.canvas.preview_time)

    def _on_slider(self, value: int) -> None:
        self.canvas.set_preview_time(value * PREVIEW_STEP)

    def _on_time(self, t: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(round(t / PREVIEW_STEP))
        self.slider.blockSignals(False)
        self.time_label.setText(f"{t:.1f} / {self.canvas.max_preview_time():.1f}")

    def _on_playing(self, playing: bool) -> None:
        self.play_button.setText("⏸" if playing else "▶")

    def _on_motion(self, has: bool) -> None:
        self._sync_range()
        self.setVisible(has)

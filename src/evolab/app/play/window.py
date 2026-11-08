"""직접 조종 창: 학습 전에 키보드로 환경을 몰아 본다.

학습 워커와 같은 런타임(Episode)을 화면 프로세스 안에서 직접 쓴다(명세의 예외). 사람이 몰아 본 결과와
학습 중 시험 결과가 같은 규칙을 따른다.

v2.0(설계 5.2, D12·D49):
- Esc는 늘 일시정지. Space는 환경이 Space를 쓰지 않을 때만 일시정지(점프 추진기의 키로 쓰면 점프).
- 정해진 움직임 세계 물체(장애물 줄 등)는 궤적에 없으므로 그 시각의 자세(motion.pose_at)로 그린다. 정적 물체의
  다각형·원 모양(점프 게임의 땅)도 그린다. 점프한 스텝(실제 출력 1)에는 물체 아래에 짧은 먼지 표시.
- HUD에 점프 추진기마다 「점프 준비됨 / 공중」(Episode.jump_ready). 출력이 없는 환경(글라이더)은 조종 키 대신
  「조종할 것이 없는 환경」 안내.
- 기본 보기(editor.view)가 있으면 카메라가 그 상자에서 시작하고, 따라갈 물체가 보이는 범위의 가장자리를 넘을 때만
  움직인다(배율은 그대로). 없으면 지금 규칙(따라갈 물체와 목표 영역).

v2.1(몸 구조): 직접 조종은 처음 몸(매개변수 값)으로 한다. 처음 몸에 없는 부품(Episode.absent)이 있으면 그 이름과 「처음 몸은
서툴 수 있고 학습하면 바뀐다」 안내를 보이고(PLAY_STRUCTURE_NOTE), 없는 부품의 구동기 키(Episode.dead_channels)는 도움말에서
뺀다. 센서 표는 없는 부품에 단 센서 옆에 「(없음)」(INSPECT_ABSENT). 몸은 처음 몸 그대로라 입력 기록·「내 최고 기록」은 v2.0과
같은 뜻이다(바퀴 고르는 언덕차의 처음 몸은 설계 D44대로 일부러 바퀴 하나).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QElapsedTimer, QEvent, QPointF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QCloseEvent,
    QColor,
    QFocusEvent,
    QKeyEvent,
    QShowEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from evolab import paths
from evolab.app import strings as S
from evolab.app.play.record import PlayRecord, align_record, new_record
from evolab.spec import model as m
from evolab.spec.compiled import EnvSpec
from evolab.spec.editor_meta import ViewBox, view_box
from evolab.spec.io import env_hash
from evolab.world.physics2d.episode import (
    BodyGeom,
    Episode,
    EpisodeBuildError,
    EpisodeResult,
    FrameState,
    Trial,
    WorldGeometry,
)
from evolab.world.physics2d.motion import MotionPlan, pose_at

TICK_MS = 16
MAX_STEPS_PER_TICK = 5
TABLE_EVERY_TICKS = 4
DUST_SECONDS = 0.35
"""점프 먼지 표시가 남는 시간(초, 시뮬레이션 시각 기준)."""
DUST_PUFFS = 6
VIEW_EDGE = 0.2
"""기본 보기 카메라: 따라갈 물체가 보이는 범위의 가장자리에서 이 비율(반폭·반높이 기준) 안쪽을 넘으면 따라간다."""
VIEW_LEAD = 0.3
"""가로로 따라가기 시작하면 따라갈 물체를 화면 가운데에서 반폭의 이 비율 자리에 둔다(0.3: 화면 폭의 65%·35%). 가는 쪽이
더 보이게 하려는 것으로, 넘은 뒤 잠깐 카메라가 물체보다 빨리 움직여 이 자리로 옮긴다(세로는 가장자리 그대로)."""
VIEW_SCALE = (1.0, 400.0)
"""기본 보기 카메라 배율 범위(px/m)."""

COLOR_BG = QColor("#0b0e14")
COLOR_TERRAIN_FILL = QColor("#2a2f3a")
COLOR_TERRAIN_LINE = QColor("#8a93a6")
COLOR_PAD = QColor("#3ddc84")
COLOR_REGION = QColor(61, 220, 132, 40)
COLOR_BOUNDS = QColor(255, 255, 255, 40)
COLOR_AGENT = QColor("#d8dee9")
COLOR_AGENT_EDGE = QColor("#4c566a")
COLOR_LEG = QColor("#a3be8c")
COLOR_WORLD_BODY = QColor("#81a1c1")
COLOR_FLAME = QColor("#ff9f43")
COLOR_HUD = QColor("#e5e9f0")


def qt_key(name: str) -> int | None:
    """환경 파일의 키 이름(Up, W, Space ...)을 Qt 키 코드로."""
    key = getattr(Qt.Key, f"Key_{name}", None)
    return int(key.value) if key is not None else None


_VK_ARROWS = {0x25: "Left", 0x26: "Up", 0x27: "Right", 0x28: "Down", 0x20: "Space"}


def event_key(event: QKeyEvent) -> int:
    """한글 입력기가 켜져 있어도 같은 키로 보이게 한다(윈도우 가상 키 코드로 보정)."""
    key = int(event.key())
    if key not in (0, int(Qt.Key.Key_unknown.value)):
        return key
    vk = int(event.nativeVirtualKey())
    if 0x41 <= vk <= 0x5A:
        return int(Qt.Key.Key_A.value) + (vk - 0x41)
    if 0x30 <= vk <= 0x39:
        return int(Qt.Key.Key_0.value) + (vk - 0x30)
    name = _VK_ARROWS.get(vk)
    return qt_key(name) or key if name else key


@dataclass
class KeyBinding:
    output_index: int
    keys: list[int]
    label: str
    key_names: list[str]
    value: float = 1.0
    """누르는 동안의 출력. 같은 출력에 묶인 키 값은 더한 뒤 -1..1로 자른다(차량: 앞 +1, 뒤 -1)."""


def key_bindings(spec: EnvSpec) -> list[KeyBinding]:
    """출력 채널마다 키. 추진기는 누르면 1, 차량은 앞/뒤 키가 가속 +1/-1, 왼쪽/오른쪽 키가 조향 +1/-1, 모터는 +1/-1."""
    out: list[KeyBinding] = []
    by_id = {a.id: a for a in spec.file.actuators}

    def add(i: int, names: list[str], label: str, value: float) -> None:
        mapped = [(n, k) for n in names if (k := qt_key(n)) is not None]
        if mapped:
            out.append(KeyBinding(i, [k for _, k in mapped], label, [n for n, _ in mapped], value))

    for i, channel in enumerate(spec.outputs):
        a = by_id.get(spec.actuator_channels[channel].owner)
        if isinstance(a, m.Thruster) and a.keys:
            add(i, a.keys, a.label or a.id, 1.0)
        elif isinstance(a, m.Vehicle):
            name = a.label or a.id
            if channel.endswith(".throttle"):
                add(i, a.keys_forward, S.PLAY_KEY_FORWARD.format(name=name), 1.0)
                add(i, a.keys_back, S.PLAY_KEY_BACK.format(name=name), -1.0)
            else:
                add(i, a.keys_left, S.PLAY_KEY_LEFT.format(name=name), 1.0)
                add(i, a.keys_right, S.PLAY_KEY_RIGHT.format(name=name), -1.0)
        elif isinstance(a, m.Motor):
            name = a.label or a.id
            add(i, a.keys_pos, S.PLAY_KEY_CCW.format(name=name), 1.0)
            add(i, a.keys_neg, S.PLAY_KEY_CW.format(name=name), -1.0)
        elif isinstance(a, m.Jump) and a.keys:
            add(i, a.keys, a.label or a.id, 1.0)  # 누르고 있는 동안 출력 1: 착지할 때마다 다시 뛴다
    return out


def uses_space(bindings: list[KeyBinding]) -> bool:
    """조종 키에 Space가 있는가(점프 추진기만 쓸 수 있다). 그러면 Space는 일시정지가 아니라 조종 키다(D12)."""
    space = qt_key("Space")
    return any(space in kb.keys for kb in bindings)


def _follow_axis(c: float, p: float, edge: float, hold: float) -> float:
    """카메라 한 축의 중심. 점 p가 c ± edge 안이면 c. 넘으면 넘은 거리 e만큼 옮기고, 가장자리보다 안쪽 자리(hold < edge)에
    두려면 e가 0..D(= 2(edge − hold))인 동안 그 차이를 부드럽게 더한다(카메라 속도가 점의 1배 → 2배 → 1배). e ≥ D부터는 점이
    c' ± hold에 머문다. p만의 연속 함수라 시각을 되돌려도 같은 화면이다. hold == edge면 넘은 만큼만(v2.0 첫 규칙)."""
    if p > c + edge:
        e, sign = p - (c + edge), 1.0
    elif p < c - edge:
        e, sign = (c - edge) - p, -1.0
    else:
        return c
    extra = edge - hold
    if extra <= 0.0:
        return p - sign * edge
    d = 2.0 * extra
    if e >= d:
        return p - sign * hold
    return c + sign * (e + 0.5 * (e - d / (2.0 * math.pi) * math.sin(2.0 * math.pi * e / d)))


def view_camera(box: ViewBox, follow: tuple[float, float] | None, width: float, height: float,
                zoom: float = 1.0) -> tuple[tuple[float, float], float]:
    """기본 보기 카메라 (중심, 배율 px/m): 상자가 화면에 들어오는 배율로 상자 가운데를 보다가, 따라갈 점이 보이는
    범위의 가장자리(VIEW_EDGE 안쪽)를 넘으면 따라간다. 가로는 따라가기 시작한 뒤 점을 화면 폭의 65%(VIEW_LEAD) 자리로
    옮겨 가는 쪽이 보이게 하고, 세로는 넘은 만큼만 옮긴다(_follow_axis). 따라갈 점만의 함수라 시각을 되돌려도 같다."""
    x0, y0, x1, y1 = box
    w, h = max(float(width), 100.0), max(float(height), 100.0)
    fit = min(w / (x1 - x0), h / (y1 - y0))
    scale = min(VIEW_SCALE[1], max(VIEW_SCALE[0], fit * zoom))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if follow is not None and all(math.isfinite(v) for v in follow):
        px, py = follow
        hw = (1.0 - VIEW_EDGE) * w / (2.0 * scale)
        hh = (1.0 - VIEW_EDGE) * h / (2.0 * scale)
        cx = _follow_axis(cx, px, hw, min(hw, VIEW_LEAD * w / (2.0 * scale)))
        cy = _follow_axis(cy, py, hh, hh)
    return (cx, cy), scale


def body_path(body: BodyGeom) -> tuple[QPainterPath, QPainterPath]:
    """물체 좌표의 (채울 모양: 다각형·원, 선분) 경로."""
    fill = QPainterPath()
    fill.setFillRule(Qt.FillRule.WindingFill)
    lines = QPainterPath()
    for s in body.shapes:
        if s.kind == "poly" and len(s.points) >= 3:
            fill.addPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in s.points]))
            fill.closeSubpath()
        elif s.kind == "circle" and s.points:
            fill.addEllipse(QPointF(float(s.points[0][0]), float(s.points[0][1])), float(s.radius), float(s.radius))
        elif s.kind == "segment" and len(s.points) >= 2:
            lines.moveTo(float(s.points[0][0]), float(s.points[0][1]))
            lines.lineTo(float(s.points[1][0]), float(s.points[1][1]))
    return fill, lines


@dataclass
class _Mover:
    """정해진 움직임 세계 물체 하나: 움직임과 물체 좌표 모양."""

    motion: MotionPlan
    fill: QPainterPath
    lines: QPainterPath


class PlayCanvas(QWidget):
    """세계를 그린다. 카메라는 에이전트와 목표 영역(패드 등)이 함께 보이게 맞춘다. 휠로 확대·축소."""

    def __init__(self) -> None:
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 480)
        self.geometry_: WorldGeometry | None = None
        self.frame: FrameState | None = None
        self.follow_index = 0
        self.scale = 22.0
        self.zoom = 1.0
        self.center = (0.0, 0.0)
        self.view_box: ViewBox | None = None
        """기본 보기(editor.view). 있으면 카메라가 이 상자에서 시작한다(view_camera)."""
        self._targets: list[tuple[float, float]] = []
        self.hud = ""
        self._terrain_path: QPainterPath | None = None
        self._terrain_fill: QPainterPath | None = None
        self._static_fill: QPainterPath | None = None
        """정적 물체의 다각형·원 모양(세계 좌표)."""
        self._pad_lines: list[tuple[QPointF, QPointF]] = []
        self._movers: list[_Mover] = []
        self.mover_poses: list[tuple[float, float, float]] = []
        """마지막으로 그린 정해진 움직임 물체의 자세(그린 순서). 테스트가 읽는다."""
        self.dust: list[tuple[float, float, float]] = []
        """점프 먼지: (뛴 시각, x, 물체 아랫면 y). 창이 넣고, DUST_SECONDS가 지나면 그리지 않는다."""
        self.on_key_press = None
        self.on_key_release = None
        self.on_focus_lost = None

    def set_world(self, geometry: WorldGeometry, frame: FrameState, follow: str | None,
                  view: ViewBox | None = None) -> None:
        self.geometry_ = geometry
        self.frame = frame
        self.view_box = view
        self.dust = []
        moving = [b for b in geometry.bodies if b.moving]
        ids = [b.id for b in moving]
        self.follow_index = ids.index(follow) if follow in ids else 0
        self._targets = [pt for r in geometry.regions if r.kind in ("pad", "goal") for pt in r.points]
        self._prepare_static()
        if frame.poses.shape[0] or view is not None:
            self.center, self.scale = self._camera_goal(frame)
        self.update()

    def _follow_point(self, frame: FrameState) -> tuple[float, float] | None:
        if not frame.poses.shape[0]:
            return None
        x, y, _ = frame.poses[min(self.follow_index, frame.poses.shape[0] - 1)]
        return float(x), float(y)

    def _camera_goal(self, frame: FrameState) -> tuple[tuple[float, float], float]:
        """따라갈 물체와 목표 영역이 함께 들어오는 중심과 배율(px/m). 기본 보기가 있으면 view_camera."""
        if self.view_box is not None:
            return view_camera(self.view_box, self._follow_point(frame), self.width(), self.height(), self.zoom)
        x, y, _ = frame.poses[self.follow_index]
        xs = [float(x)] + [p[0] for p in self._targets]
        ys = [float(y)] + [p[1] for p in self._targets]
        x0, x1, y0, y1 = min(xs) - 6.0, max(xs) + 6.0, min(ys) - 4.0, max(ys) + 4.0
        w, h = max(self.width(), 100), max(self.height(), 100)
        fit = min(w / (x1 - x0), h / (y1 - y0))
        scale = min(30.0, max(4.0, fit)) * self.zoom
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0), scale

    def set_frame(self, frame: FrameState, hud: str) -> None:
        self.frame = frame
        self.hud = hud
        if self.view_box is not None:
            # 기본 보기: 따라갈 점이 가장자리를 넘을 때만 넘은 만큼 옮기므로 부드럽게 따라가려고 늦추지 않는다
            self.center, self.scale = self._camera_goal(frame)
        elif frame.poses.shape[0]:
            (gx, gy), gs = self._camera_goal(frame)
            cx, cy = self.center
            self.center = (cx + (gx - cx) * 0.15, cy + (gy - cy) * 0.15)
            self.scale += (gs - self.scale) * 0.1
        self.update()

    def _prepare_static(self) -> None:
        g = self.geometry_
        assert g is not None
        path = QPainterPath()
        fill = QPainterPath()
        solid = QPainterPath()
        solid.setFillRule(Qt.FillRule.WindingFill)
        self._movers = []
        bottom = g.bounds[1] - 50.0
        for b in g.bodies:
            if b.moving:
                continue
            if b.motion is not None:  # 정해진 움직임: 프레임마다 그 시각의 자세로 그린다
                self._movers.append(_Mover(b.motion, *body_path(b)))
                continue
            shapes_fill, _ = body_path(b)
            if not shapes_fill.isEmpty():
                solid.addPath(QTransform().translate(b.pose[0], b.pose[1]).rotateRadians(b.pose[2]).map(shapes_fill))
            segs = [s for s in b.shapes if s.kind == "segment"]
            if not segs:
                continue
            x0, y0, a0 = b.pose
            pts = []
            for s in segs:
                for p in s.points:
                    c, si = math.cos(a0), math.sin(a0)
                    pts.append((x0 + p[0] * c - p[1] * si, y0 + p[0] * si + p[1] * c))
            path.moveTo(*pts[0])
            for p in pts[1:]:
                path.lineTo(*p)
            poly = QPolygonF([QPointF(*pts[0]), *[QPointF(*p) for p in pts[1:]]])
            poly.append(QPointF(pts[-1][0], bottom))
            poly.append(QPointF(pts[0][0], bottom))
            fill.addPolygon(poly)
        self._terrain_path = path
        self._terrain_fill = fill
        self._static_fill = solid
        self._pad_lines = []
        for r in g.regions:
            if r.kind == "pad" and len(r.points) >= 2:
                self._pad_lines.append((QPointF(*r.points[0]), QPointF(*r.points[1])))

    # ---- 그리기 ----
    def paintEvent(self, event: object) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), COLOR_BG)
        g, f = self.geometry_, self.frame
        if g is None or f is None:
            p.end()
            return
        w, h = self.width(), self.height()
        cx, cy = self.center
        s = self.scale
        p.setTransform(QTransform(s, 0.0, 0.0, -s, w / 2.0 - cx * s, h / 2.0 + cy * s))

        def pen(color: QColor, width: float = 1.5) -> QPen:
            q = QPen(color, width)
            q.setCosmetic(True)
            return q

        x0, y0, x1, y1 = g.bounds
        p.setPen(pen(COLOR_BOUNDS, 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QPolygonF([QPointF(x0, y0), QPointF(x1, y0), QPointF(x1, y1), QPointF(x0, y1)]).boundingRect())

        for r in g.regions:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(COLOR_REGION))
            p.drawPolygon(QPolygonF([QPointF(*pt) for pt in r.points]))
        if self._terrain_fill is not None and self._terrain_path is not None:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(COLOR_TERRAIN_FILL))
            p.drawPath(self._terrain_fill)
            p.setPen(pen(COLOR_TERRAIN_LINE, 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(self._terrain_path)
        if self._static_fill is not None and not self._static_fill.isEmpty():
            p.setPen(pen(COLOR_TERRAIN_LINE, 1.0))
            p.setBrush(QBrush(COLOR_WORLD_BODY))
            p.drawPath(self._static_fill)
        p.setPen(pen(COLOR_PAD, 4.0))
        for a, b in self._pad_lines:
            p.drawLine(a, b)
        self._draw_movers(p, pen, f.time)
        self._draw_dust(p, f.time)

        moving = [b for b in g.bodies if b.moving]
        poses = f.poses
        for i, b in enumerate(moving):
            x, y, ang = poses[i]
            p.save()
            p.translate(float(x), float(y))
            p.rotate(math.degrees(float(ang)))
            color = COLOR_LEG if "leg" in b.tags else (COLOR_AGENT if b.owner == "agent" else COLOR_WORLD_BODY)
            p.setPen(pen(COLOR_AGENT_EDGE, 1.0))
            p.setBrush(QBrush(color))
            for shape in b.shapes:
                if shape.kind == "poly":
                    p.drawPolygon(QPolygonF([QPointF(*pt) for pt in shape.points]))
                elif shape.kind == "circle":
                    c = shape.points[0]
                    p.drawEllipse(QPointF(*c), shape.radius, shape.radius)
                elif shape.kind == "segment":
                    p.drawLine(QPointF(*shape.points[0]), QPointF(*shape.points[1]))
            p.restore()

        index = {b.id: i for i, b in enumerate(moving)}
        names = list(f.outputs)
        for t in g.thrusters:
            k = self._output_index(t.channel)
            if k is None or k >= len(names) or t.body not in index:
                continue
            u = float(f.outputs[k])
            if u <= 0.0:
                continue
            x, y, ang = poses[index[t.body]]
            p.save()
            p.translate(float(x), float(y))
            p.rotate(math.degrees(float(ang)))
            p.translate(*t.at)
            p.rotate(math.degrees(t.dir + math.pi))
            length = 0.3 + 1.7 * u * min(1.0, t.max_force / 200.0 + 0.3)
            flame = QPolygonF([QPointF(0.0, -0.18), QPointF(length, 0.0), QPointF(0.0, 0.18)])
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(COLOR_FLAME))
            p.drawPolygon(flame)
            p.restore()

        p.resetTransform()
        p.setPen(COLOR_HUD)
        p.drawText(12, 22, self.hud)
        p.end()

    def _draw_movers(self, p: QPainter, pen: Callable[[QColor, float], QPen], t: float) -> None:
        """정해진 움직임 세계 물체를 시각 t의 자세(pose_at)로. 런타임이 그 스텝에 놓은 자리와 같다(설계 D7)."""
        poses: list[tuple[float, float, float]] = []
        for mv in self._movers:
            x, y, ang = pose_at(mv.motion, t)
            poses.append((x, y, ang))
            p.save()
            p.translate(x, y)
            p.rotate(math.degrees(ang))
            if not mv.fill.isEmpty():
                p.setPen(pen(COLOR_TERRAIN_LINE, 1.0))
                p.setBrush(QBrush(COLOR_WORLD_BODY))
                p.drawPath(mv.fill)
            if not mv.lines.isEmpty():
                p.setPen(pen(COLOR_WORLD_BODY, 2.0))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(mv.lines)
            p.restore()
        self.mover_poses = poses

    def _draw_dust(self, p: QPainter, t: float) -> None:
        """점프한 자리의 먼지: 물체 아랫면에서 양옆으로 퍼지며 옅어지는 작은 원들."""
        live = [d for d in self.dust if 0.0 <= t - d[0] < DUST_SECONDS]
        if len(live) != len(self.dust):
            self.dust = live
        p.setPen(Qt.PenStyle.NoPen)
        for t0, x, y in live:
            age = (t - t0) / DUST_SECONDS
            color = QColor(COLOR_TERRAIN_LINE)
            color.setAlpha(round(200 * (1.0 - age)))
            p.setBrush(QBrush(color))
            spread = 0.15 + 0.55 * age
            r = 0.05 + 0.07 * age
            for k in range(DUST_PUFFS):
                u = (k / (DUST_PUFFS - 1)) * 2.0 - 1.0
                p.drawEllipse(QPointF(x + u * spread, y + r * (1.0 - abs(u))), r, r)

    def _output_index(self, channel: str) -> int | None:
        names = getattr(self, "channel_names", None)
        if not names:
            return None
        try:
            return names.index(channel)
        except ValueError:
            return None

    # ---- 입력 ----
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = min(8.0, max(0.25, self.zoom * factor))
        self.scale = min(200.0, max(1.0, self.scale * factor))
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.isAutoRepeat():
            return
        if self.on_key_press is not None and self.on_key_press(event_key(event)):
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.isAutoRepeat():
            return
        if self.on_key_release is not None:
            self.on_key_release(event_key(event))

    def focusOutEvent(self, event: QFocusEvent) -> None:  # noqa: N802
        # 키를 누른 채 다른 창으로 가면 떼는 이벤트가 오지 않는다. 눌린 키를 모두 놓은 것으로 본다.
        if self.on_focus_lost is not None:
            self.on_focus_lost()
        super().focusOutEvent(event)


class PlayWindow(QWidget):
    """직접 조종 창. R 다시 시작, N 새 시드, Esc 일시정지(환경이 Space를 쓰지 않으면 Space도)."""

    def __init__(self, spec: EnvSpec, seed: int | None = None) -> None:
        super().__init__()
        self.spec = spec
        self.setWindowTitle(S.PLAY_TITLE.format(name=spec.file.name))
        self.resize(1280, 760)
        self.bindings = key_bindings(spec)
        self.space_pauses = not uses_space(self.bindings)
        """Space가 일시정지인가(점프 추진기가 Space를 쓰면 Space는 조종 키, 일시정지는 Esc만)."""
        self.view = view_box(spec.file)
        self.jumps: list[tuple[str, str, str]] = [
            (a.id, a.label or a.id, a.body) for a in spec.file.actuators
            if isinstance(a, m.Jump) and a.id in spec.actuator_channels
        ]
        """점프 추진기 (채널 = 구동기 id, 이름, 물체 id): HUD와 먼지 표시."""
        self.pressed: set[int] = set()
        self.dead_outputs: frozenset[int] = frozenset()
        """없는 부품의 구동기 출력 번호(spec.outputs 순서, v2.1). 그 키는 도움말에 넣지 않는다(눌러도 출력 0)."""
        self.absent_inputs: frozenset[str] = frozenset()
        """없는 부품에 단 센서의 입력 채널 이름(v2.1). 센서 표에서 「(없음)」."""
        self.paused = False
        self.replaying: PlayRecord | None = None
        self.episode: Episode | None = None
        self.record: PlayRecord | None = None
        self.last_result: EpisodeResult | None = None
        self._tick_count = 0

        self.canvas = PlayCanvas()
        self.canvas.on_key_press = self._key_press
        self.canvas.on_key_release = self._key_release
        self.canvas.on_focus_lost = self.pressed.clear
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.seed_label = QLabel()
        self.blocks = QTableWidget(0, 3)
        self.blocks.setHorizontalHeaderLabels([S.PLAY_COL_BLOCK, S.PLAY_COL_VALUE, S.PLAY_COL_SCORE])
        self.blocks.verticalHeader().setVisible(False)
        self.blocks.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.blocks.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.blocks.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sensors = QTableWidget(0, 2)
        self.sensors.setHorizontalHeaderLabels([S.PLAY_COL_CHANNEL, S.PLAY_COL_VALUE])
        self.sensors.verticalHeader().setVisible(False)
        self.sensors.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.sensors.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.sensors.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        def button(text: str, slot: object) -> QPushButton:
            b = QPushButton(text)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(slot)
            return b

        help_label = QLabel(self.help_text())
        help_label.setWordWrap(True)
        self.help_label = help_label
        self.structure_label = QLabel()
        """처음 몸에 없는 부품 안내(v2.1). 없는 부품이 없으면 숨긴다."""
        self.structure_label.setWordWrap(True)
        self.structure_label.hide()

        side = QVBoxLayout()
        side.addWidget(self.structure_label)
        side.addWidget(self.status)
        side.addWidget(self.seed_label)
        side.addWidget(QLabel(S.PLAY_BLOCKS_TITLE))
        side.addWidget(self.blocks, 3)
        side.addWidget(QLabel(S.PLAY_SENSORS_TITLE))
        side.addWidget(self.sensors, 2)
        row = QHBoxLayout()
        row.addWidget(button(S.PLAY_RESTART, self.restart))
        row.addWidget(button(S.PLAY_NEW_SEED, self.new_seed))
        side.addLayout(row)
        row2 = QHBoxLayout()
        self.save_button = button(S.PLAY_SAVE_RECORD, self.save_record)
        self.save_button.setEnabled(False)  # 끝난 기록만 저장한다(결과가 있어야 재생 검증이 된다)
        if not self.spec.outputs:  # 지켜보기만 하는 환경(글라이더): 사람의 입력이 없어 「내 기록」이 아니다
            self.save_button.setToolTip(S.PLAY_SAVE_WATCH_ONLY)
        row2.addWidget(self.save_button)
        row2.addWidget(button(S.PLAY_LOAD_RECORD, self.load_record))
        side.addLayout(row2)
        side.addWidget(help_label)
        panel = QWidget()
        panel.setLayout(side)
        panel.setFixedWidth(360)

        layout = QHBoxLayout(self)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(panel)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._tick)
        self.clock = QElapsedTimer()
        self._steps_since_clock = 0

        # 첫 시작이 실패하면(EpisodeBuildError) 그대로 올려 보내 창을 여는 쪽이 알린다.
        self._start_episode(seed if seed is not None else random.SystemRandom().randrange(1 << 31), None)
        self.timer.start()

    def help_text(self) -> str:
        """도움말 글: 조종 키(출력이 없는 환경은 「조종할 것이 없는 환경」)와 창 제어 키. 일시정지 키는 Esc, 환경이 Space를
        쓰지 않으면 Space·Esc."""
        if not self.spec.outputs:
            keys = S.PLAY_WATCH_ONLY
        else:
            keys = "\n".join(
                S.PLAY_KEY_LINE.format(keys=" / ".join(kb.key_names), label=kb.label) for kb in self.bindings
                if kb.output_index not in self.dead_outputs  # 없는 부품의 키(v2.1)는 듣지 않으므로 뺀다
            )
        pause = S.PLAY_PAUSE_KEYS["both" if self.space_pauses else "esc"]
        return S.PLAY_HELP.format(keys=keys, pause=pause)

    # ---- 진행 ----
    def start(self, seed: int, replaying: PlayRecord | None = None) -> bool:
        """새 에피소드. 이 시드로 조립할 수 없으면 알리고 지금 상태를 그대로 둔다."""
        try:
            self._start_episode(seed, replaying)
        except EpisodeBuildError as exc:
            QMessageBox.warning(self, S.PLAY_OPEN_FAILED, self.build_failed_text(seed, exc))
            return False
        return True

    def build_failed_text(self, seed: int, exc: EpisodeBuildError) -> str:
        """조립 실패 알림. 에이전트 물체의 모양·질량이 잘못되어 실패했으면 「이 몸은 만들 수 없습니다」를 앞에 붙인다."""
        text = S.PLAY_BUILD_FAILED.format(seed=seed, reason=exc)
        agents = [b.id for b in self.spec.file.bodies if b.owner == "agent"]
        body_errors = {f.format(body=b) for b in agents for f in (S.RT_DEGENERATE_SHAPE, S.RT_BAD_MASS)}
        return S.PLAY_BODY_BUILD_FAILED + "\n" + text if str(exc) in body_errors else text

    def _start_episode(self, seed: int, replaying: PlayRecord | None) -> None:
        episode = Episode(self.spec, None, Trial(seed))  # 실패하면 아무 상태도 바꾸지 않는다
        self.seed = seed
        self.replaying = replaying
        self.episode = episode
        self.record = new_record(self.spec, seed)
        self.last_result = None
        self.paused = False
        self.save_button.setEnabled(False)
        geom = self.episode.geometry()
        agent = next((b.id for b in self.spec.file.bodies if b.owner == "agent"), None)
        self.canvas.channel_names = self.episode.actuator_channels()
        self._jump_index = [(self.canvas.channel_names.index(ch), self.episode.bodies[body])
                            for ch, _label, body in self.jumps if body in self.episode.bodies]
        self.canvas.set_world(geom, self.episode.snapshot(), agent, self.view)
        self._show_structure(episode)
        self.seed_label.setText(S.PLAY_SEED.format(seed=seed))
        self._fill_tables(force=True)
        self.clock.start()
        self._steps_since_clock = 0
        self.canvas.setFocus()

    def _show_structure(self, episode: Episode) -> None:
        """처음 몸에 없는 부품(v2.1): 안내 글, 도움말에서 뺄 키, 센서 표의 「(없음)」. 몸 수치만의 함수라 시드와 무관하다."""
        dead = set(episode.dead_channels)
        self.dead_outputs = frozenset(i for i, ch in enumerate(self.spec.outputs) if ch in dead)
        absent = episode.absent
        sensors = {s.id for s in self.spec.file.sensors
                   if getattr(s, "body", None) in absent
                   or (isinstance(s, m.JointAngle) and s.joint in episode.absent_joints)}
        self.absent_inputs = frozenset(
            name for name in episode.inputs
            if (ch := self.spec.sensor_channels.get(name)) is not None and ch.owner in sensors)
        names = [b.label or b.id for b in self.spec.file.bodies if b.id in absent]
        self.structure_label.setText(S.PLAY_STRUCTURE_NOTE.format(names=", ".join(names)) if names else "")
        self.structure_label.setVisible(bool(names))
        self.help_label.setText(self.help_text())

    def restart(self) -> None:
        self.start(self.seed)

    def new_seed(self) -> None:
        self.start(random.SystemRandom().randrange(1 << 31))

    def current_action(self) -> list[float]:
        action = [0.0] * len(self.spec.outputs)
        for kb in self.bindings:
            if any(k in self.pressed for k in kb.keys):
                action[kb.output_index] += kb.value
        return [max(-1.0, min(1.0, x)) for x in action]

    def _tick(self) -> None:
        ep = self.episode
        if ep is None:
            return
        if not self.paused and not ep.done:
            target = int(self.clock.elapsed() / 1000.0 / ep.dt)
            due = target - self._steps_since_clock
            if due > MAX_STEPS_PER_TICK:  # 너무 밀리면(창 끌기 등) 밀린 몫은 버리고 시계는 그대로 둔다
                self._steps_since_clock = target - MAX_STEPS_PER_TICK
                due = MAX_STEPS_PER_TICK
            for _ in range(max(0, due)):
                if self.replaying is not None:
                    k = ep.steps
                    action = self.replaying.actions[k] if k < len(self.replaying.actions) else [0.0] * len(self.spec.outputs)
                else:
                    action = self.current_action()
                assert self.record is not None
                self.record.actions.append(list(action))
                ep.step(action)
                self._steps_since_clock += 1
                if self._jump_index:
                    self._note_jumps(ep)
                if ep.done:
                    self._finish()
                    break
        else:
            self.clock.restart()
            self._steps_since_clock = 0
        self._tick_count += 1
        self.canvas.set_frame(ep.snapshot(), self._hud())
        self._fill_tables(force=False)

    def _note_jumps(self, ep: Episode) -> None:
        """이 스텝에 뛴 점프 추진기(실제 출력 1)마다 물체 아랫면에 먼지 표시를 남긴다."""
        applied = ep.applied_outputs
        for k, body in self._jump_index:
            if applied[k] == 1.0:
                bottom = min((sh.bb.bottom for sh in body.shapes), default=float(body.position.y))
                self.canvas.dust.append((ep.time, float(body.position.x), float(bottom)))

    def _finish(self) -> None:
        assert self.episode is not None and self.record is not None
        res = self.episode.result()
        self.last_result = res
        self.record.score = res.score
        self.record.outcome = res.outcome
        self.record.steps = res.steps
        self.save_button.setEnabled(bool(self.spec.outputs))
        if self.replaying is not None and self.replaying.score is not None:
            if res.score == self.replaying.score and res.steps == self.replaying.steps:
                QMessageBox.information(self, S.PLAY_REPLAY_TITLE, S.PLAY_REPLAY_SAME.format(score=res.score))
            else:
                QMessageBox.warning(
                    self, S.PLAY_REPLAY_TITLE,
                    S.PLAY_REPLAY_DIFF.format(a=self.replaying.score, b=res.score),
                )

    def _hud(self) -> str:
        ep = self.episode
        assert ep is not None
        agent = next((b for b in self.spec.file.bodies if b.owner == "agent"), None)
        speed = 0.0
        if agent is not None:
            v = ep.bodies[agent.id].velocity
            speed = math.hypot(v.x, v.y)
        text = S.PLAY_HUD.format(t=ep.time, max=self.spec.file.world.max_time, speed=speed)
        if self.jumps:
            states = [(label, S.PLAY_JUMP_READY if ep.jump_ready(ch) else S.PLAY_JUMP_AIR) for ch, label, _ in self.jumps]
            text += "  " + (states[0][1] if len(states) == 1 else " · ".join(f"{a} {b}" for a, b in states))
        if self.paused:
            text += "  " + S.PLAY_PAUSED
        if self.replaying is not None:
            text += "  " + S.PLAY_REPLAYING
        return text

    def _fill_tables(self, force: bool) -> None:
        if not force and self._tick_count % TABLE_EVERY_TICKS:
            return
        ep = self.episode
        assert ep is not None
        # 진행 중에도 학습과 같은 계산(result)으로 보여 준다. 점수 수식이 있으면 블록 가중치는 쓰이지 않는다.
        res = self.last_result if (ep.done and self.last_result is not None) else ep.result()
        if ep.done and self.last_result is not None:
            self.status.setText(S.PLAY_DONE.format(label=res.end_label, score=res.score))
        else:
            self.status.setText(S.PLAY_RUNNING)
        formula = self.spec.formula is not None
        blocks = self.spec.file.scoring.blocks
        self.blocks.setRowCount(len(blocks) + 1)
        for r, b in enumerate(blocks):
            self._set(self.blocks, r, 0, b.label or b.id)
            self._set(self.blocks, r, 1, f"{res.blocks[b.id]:.3f}")
            self._set(self.blocks, r, 2, S.PLAY_NO_WEIGHT if formula else f"{res.weighted.get(b.id, 0.0):+.2f}")
        total = res.score
        self._set(self.blocks, len(blocks), 0, S.PLAY_TOTAL)
        self._set(self.blocks, len(blocks), 1, "")
        self._set(self.blocks, len(blocks), 2, f"{total:+.2f}")
        obs = ep.observe()
        names = ep.inputs
        self.sensors.setRowCount(len(names))
        for r, name in enumerate(names):
            self._set(self.sensors, r, 0, f"{name} {S.INSPECT_ABSENT}" if name in self.absent_inputs else name)
            self._set(self.sensors, r, 1, f"{float(obs[r]):+.3f}")

    @staticmethod
    def _set(table: QTableWidget, row: int, col: int, text: str) -> None:
        item = table.item(row, col)
        if item is None:
            item = QTableWidgetItem(text)
            if col > 0:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, col, item)
        elif item.text() != text:
            item.setText(text)

    # ---- 키 ----
    def _key_press(self, key: int) -> bool:
        if key == int(Qt.Key.Key_R.value):
            self.restart()
            return True
        if key == int(Qt.Key.Key_N.value):
            self.new_seed()
            return True
        if key == int(Qt.Key.Key_Escape.value) or (key == int(Qt.Key.Key_Space.value) and self.space_pauses):
            self.paused = not self.paused
            return True
        self.pressed.add(key)
        return True

    def _key_release(self, key: int) -> None:
        self.pressed.discard(key)

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        if event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow():
            self.pressed.clear()
        super().changeEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.timer.stop()  # 닫힌 창이 뒤에서 계속 시뮬레이션하지 않게
        super().closeEvent(event)

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        if not self.timer.isActive():
            self.clock.restart()
            self._steps_since_clock = 0
            self.timer.start()
        super().showEvent(event)

    # ---- 기록 ----
    def save_record(self) -> None:
        if self.record is None or self.record.score is None:
            return
        default = records_dir() / f"play_{self.seed}.evoplay"
        path, _ = QFileDialog.getSaveFileName(self, S.PLAY_SAVE_RECORD, str(default), S.PLAY_RECORD_FILTER)
        if not path:
            return
        try:
            self.record.save(path)
        except OSError as exc:
            QMessageBox.warning(self, S.PLAY_SAVE_RECORD, S.PLAY_SAVE_FAILED.format(reason=exc.strerror or exc))
            return
        QMessageBox.information(self, S.PLAY_SAVE_RECORD, S.PLAY_SAVED.format(path=path))

    def load_record(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, S.PLAY_LOAD_RECORD, str(records_dir()), S.PLAY_RECORD_FILTER)
        if not path:
            return
        self.play_record(Path(path))

    def play_record(self, path: Path) -> None:
        try:
            rec = align_record(self.spec, PlayRecord.load(path))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, S.PLAY_REPLAY_TITLE, S.PLAY_RECORD_BAD.format(reason=exc))
            return
        if rec.env_hash != env_hash(self.spec.file):
            QMessageBox.warning(self, S.PLAY_REPLAY_TITLE, S.PLAY_REPLAY_ENV_DIFF)
        self.start(rec.seed, replaying=rec)


def records_dir() -> Path:
    """입력 기록 기본 폴더: 데이터 폴더\\records (기본은 문서\\진화실험실\\records, 설정에서 데이터 폴더를 바꾸면 따라간다)."""
    path = paths.records_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return Path.home()
    return path


def record_dirs() -> list[Path]:
    """기록을 찾을 폴더: 지금 데이터 폴더의 records, 그리고 데이터 폴더를 바꾸기 전 기본 위치(있을 때만, 만들지 않는다)."""
    folders = [records_dir()]
    try:
        old = paths.default_records_dir()
        if old.is_dir() and all(old.resolve() != f.resolve() for f in folders):
            folders.append(old)
    except OSError:
        pass
    return folders

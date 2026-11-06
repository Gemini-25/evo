"""직접 조종 창: 학습 전에 키보드로 환경을 몰아 본다.

학습 워커와 같은 런타임(Episode)을 화면 프로세스 안에서 직접 쓴다(명세의 예외). 사람이 몰아 본 결과와
학습 중 시험 결과가 같은 규칙을 따른다.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QElapsedTimer, QEvent, QPointF, QStandardPaths, Qt, QTimer
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

from evolab.app import strings as S
from evolab.app.play.record import PlayRecord, align_record, new_record
from evolab.spec import model as m
from evolab.spec.compiled import EnvSpec
from evolab.spec.io import env_hash
from evolab.world.physics2d.episode import Episode, EpisodeBuildError, EpisodeResult, FrameState, Trial, WorldGeometry

TICK_MS = 16
MAX_STEPS_PER_TICK = 5
TABLE_EVERY_TICKS = 4

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


def key_bindings(spec: EnvSpec) -> list[KeyBinding]:
    """출력 채널마다 누르면 1이 되는 키. 지금은 추진기만(차량·모터 키는 M4·M5b)."""
    out: list[KeyBinding] = []
    by_id = {a.id: a for a in spec.file.actuators}
    for i, channel in enumerate(spec.outputs):
        a = by_id.get(spec.actuator_channels[channel].owner)
        if isinstance(a, m.Thruster) and a.keys:
            pairs = [(n, qt_key(n)) for n in a.keys]
            mapped = [(n, k) for n, k in pairs if k is not None]
            if mapped:
                out.append(KeyBinding(i, [k for _, k in mapped], a.label or a.id, [n for n, _ in mapped]))
    return out


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
        self._targets: list[tuple[float, float]] = []
        self.hud = ""
        self._terrain_path: QPainterPath | None = None
        self._terrain_fill: QPainterPath | None = None
        self._pad_lines: list[tuple[QPointF, QPointF]] = []
        self.on_key_press = None
        self.on_key_release = None
        self.on_focus_lost = None

    def set_world(self, geometry: WorldGeometry, frame: FrameState, follow: str | None) -> None:
        self.geometry_ = geometry
        self.frame = frame
        moving = [b for b in geometry.bodies if b.moving]
        ids = [b.id for b in moving]
        self.follow_index = ids.index(follow) if follow in ids else 0
        self._targets = [pt for r in geometry.regions if r.kind in ("pad", "goal") for pt in r.points]
        self._prepare_static()
        if frame.poses.shape[0]:
            self.center, self.scale = self._camera_goal(frame)
        self.update()

    def _camera_goal(self, frame: FrameState) -> tuple[tuple[float, float], float]:
        """따라갈 물체와 목표 영역이 함께 들어오는 중심과 배율(px/m)."""
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
        if frame.poses.shape[0]:
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
        bottom = g.bounds[1] - 50.0
        for b in g.bodies:
            if b.moving:
                continue
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
        p.setPen(pen(COLOR_PAD, 4.0))
        for a, b in self._pad_lines:
            p.drawLine(a, b)

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
    """직접 조종 창. R 다시 시작, N 새 시드, Space 일시정지."""

    def __init__(self, spec: EnvSpec, seed: int | None = None) -> None:
        super().__init__()
        self.spec = spec
        self.setWindowTitle(S.PLAY_TITLE.format(name=spec.file.name))
        self.resize(1280, 760)
        self.bindings = key_bindings(spec)
        self.pressed: set[int] = set()
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

        keys_help = "\n".join(
            S.PLAY_KEY_LINE.format(keys=" / ".join(kb.key_names), label=kb.label) for kb in self.bindings
        )
        help_label = QLabel(S.PLAY_HELP.format(keys=keys_help))
        help_label.setWordWrap(True)

        side = QVBoxLayout()
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

    # ---- 진행 ----
    def start(self, seed: int, replaying: PlayRecord | None = None) -> bool:
        """새 에피소드. 이 시드로 조립할 수 없으면 알리고 지금 상태를 그대로 둔다."""
        try:
            self._start_episode(seed, replaying)
        except EpisodeBuildError as exc:
            QMessageBox.warning(self, S.PLAY_OPEN_FAILED, S.PLAY_BUILD_FAILED.format(seed=seed, reason=exc))
            return False
        return True

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
        self.canvas.set_world(geom, self.episode.snapshot(), agent)
        self.seed_label.setText(S.PLAY_SEED.format(seed=seed))
        self._fill_tables(force=True)
        self.clock.start()
        self._steps_since_clock = 0
        self.canvas.setFocus()

    def restart(self) -> None:
        self.start(self.seed)

    def new_seed(self) -> None:
        self.start(random.SystemRandom().randrange(1 << 31))

    def current_action(self) -> list[float]:
        action = [0.0] * len(self.spec.outputs)
        for kb in self.bindings:
            if any(k in self.pressed for k in kb.keys):
                action[kb.output_index] = 1.0
        return action

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
                if ep.done:
                    self._finish()
                    break
        else:
            self.clock.restart()
            self._steps_since_clock = 0
        self._tick_count += 1
        self.canvas.set_frame(ep.snapshot(), self._hud())
        self._fill_tables(force=False)

    def _finish(self) -> None:
        assert self.episode is not None and self.record is not None
        res = self.episode.result()
        self.last_result = res
        self.record.score = res.score
        self.record.outcome = res.outcome
        self.record.steps = res.steps
        self.save_button.setEnabled(True)
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
            self._set(self.sensors, r, 0, name)
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
        if key == int(Qt.Key.Key_Space.value):
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
    """입력 기록 기본 폴더: 문서\\진화실험실\\records (OneDrive로 옮겨진 문서 폴더도 따라간다)."""
    docs = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    path = (Path(docs) if docs else Path.home()) / S.DATA_DIR_NAME / "records"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return Path.home()
    return path

"""에디터 왼쪽 부품 팔레트: 도구 목록(묶음별), 새 물체 소유, 격자 맞춤.

한 번에 도구 하나만 켜진다. 도구를 고르면 tool_changed(도구 id). Esc를 누르면 선택 도구로 돌아간다.
바로 실행하는 도구(지형 자동 생성, 트랙 자동 생성, 남은 시간)는 눌러도 켜진 상태로 남지 않는다.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QKeyEvent, QPainter, QPalette, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from evolab.app import strings as S
from evolab.app.editor.tools import INSTANT_TOOLS, TOOL_GROUPS, TOOL_HINTS, TOOL_IDS, TOOL_LABELS

__all__ = ["Palette", "TOOL_IDS", "tool_icon"]

GROUP_LABELS = {
    "select": S.EDC_GROUP_SELECT,
    "shapes": S.EDC_GROUP_SHAPES,
    "terrain": S.EDC_GROUP_TERRAIN,
    "joints": S.EDC_GROUP_JOINTS,
    "actuators": S.EDC_GROUP_ACTUATORS,
    "sensors": S.EDC_GROUP_SENSORS,
    "regions": S.EDC_GROUP_REGIONS,
}
_INSTANT_HINTS = {
    "terrain_auto": S.EDC_HINT_TERRAIN_AUTO,
    "hills_auto": S.EDC_HINT_HILLS_AUTO,
    "track_auto": S.EDC_HINT_TRACK_AUTO,
    "sensor_time": S.EDC_HINT_SENSOR_TIME,
}
ICON_SIZE = 20

_FG = QColor("#d8dee9")
_ACCENT = QColor("#88c0d0")
_FLAME = QColor("#ff9f43")
_GREEN = QColor("#3ddc84")
_RED = QColor("#ff6363")
_YELLOW = QColor("#ffd166")
_PURPLE = QColor("#b48ead")


def _draw_icon(p: QPainter, tool_id: str, fg: QColor) -> None:
    pen = QPen(fg, 1.6)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)

    def poly(*pts: tuple[float, float]) -> QPolygonF:
        return QPolygonF([QPointF(x, y) for x, y in pts])

    def use(color: QColor, width: float = 1.6, fill: QColor | None = None) -> None:
        q = QPen(color, width)
        q.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        q.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(q)
        p.setBrush(QBrush(fill) if fill is not None else Qt.BrushStyle.NoBrush)

    t = tool_id
    if t == "select":
        use(fg, 1.2, fg)
        p.drawPolygon(poly((5, 3), (5, 16), (8.5, 12.5), (11, 18), (13, 17), (10.5, 11.5), (15, 11.5)))
    elif t == "box":
        p.drawRect(QRectF(3, 5, 14, 10))
    elif t == "circle":
        p.drawEllipse(QRectF(3, 3, 14, 14))
    elif t == "poly":
        p.drawPolygon(poly((4, 15), (3, 7), (10, 3), (17, 8), (14, 16)))
    elif t == "terrain_line":
        p.drawPolyline(poly((2, 15), (6, 10), (9, 12), (13, 6), (18, 11)))
    elif t == "terrain_auto":
        use(QColor("#8a93a6"), 1.2, QColor("#2a2f3a"))
        p.drawPolygon(poly((2, 17), (2, 13), (6, 9), (9, 12), (13, 6), (18, 11), (18, 17)))
        use(_GREEN, 2.0)
        p.drawLine(QPointF(7, 10.5), QPointF(11, 10.5))
    elif t == "hills_auto":
        use(QColor("#8a93a6"), 1.2, QColor("#2a2f3a"))
        p.drawPolygon(poly((2, 17), (2, 15), (7, 14), (10, 11), (13, 9), (16, 5), (18, 4), (18, 17)))
    elif t == "motor":
        p.drawEllipse(QRectF(4, 4, 12, 12))
        p.drawLine(QPointF(10, 10), QPointF(15, 6))
        p.drawEllipse(QRectF(8.5, 8.5, 3, 3))
    elif t in ("track_draw", "track_auto"):
        p.drawEllipse(QRectF(2, 4, 16, 12))
        p.drawEllipse(QRectF(6, 7.5, 8, 5))
        if t == "track_auto":
            use(_YELLOW, 1.6)
            p.drawLine(QPointF(10, 4), QPointF(10, 7.5))
    elif t.startswith("joint_"):
        use(QColor("#4c566a"), 1.0, QColor("#d8dee9"))
        p.drawRect(QRectF(2, 8, 8, 4))
        p.drawRect(QRectF(10, 8, 8, 4))
        use(_ACCENT, 1.6)
        if t == "joint_weld":
            p.drawRect(QRectF(7, 7, 6, 6))
        elif t == "joint_spring":
            p.drawPolyline(poly((2, 10), (5, 6), (8, 14), (11, 6), (14, 14), (18, 10)))
        else:
            p.drawEllipse(QPointF(10, 10), 3, 3)
            if t == "joint_rot_spring":
                p.drawArc(QRectF(4, 4, 12, 12), 30 * 16, 240 * 16)
    elif t == "thruster":
        use(fg, 1.2, QColor("#4c566a"))
        p.drawRect(QRectF(6, 3, 8, 8))
        use(_FLAME, 1.2, _FLAME)
        p.drawPolygon(poly((7, 12), (13, 12), (10, 18)))
    elif t == "fuel_tank":
        use(fg, 1.4, QColor("#d08770"))
        p.drawRoundedRect(QRectF(5, 4, 10, 13), 3, 3)
        p.drawLine(QPointF(8, 2.5), QPointF(12, 2.5))
    elif t == "vehicle":
        use(fg, 1.2, QColor("#4c566a"))
        p.drawRoundedRect(QRectF(4, 6, 12, 8), 2, 2)
        use(fg, 1.0, fg)
        for x, y in ((5, 4), (13, 4), (5, 14), (13, 14)):
            p.drawRect(QRectF(x, y, 2.5, 2))
    elif t == "sensor_rays":
        use(_ACCENT, 1.2)
        for k in range(5):
            a = math.pi * (0.15 + 0.7 * k / 4)
            p.drawLine(QPointF(10, 17), QPointF(10 + 9 * math.cos(a), 17 - 9 * math.sin(a)))
    elif t == "sensor_body_state":
        p.drawEllipse(QPointF(8, 12), 4, 4)
        use(_ACCENT, 1.6)
        p.drawLine(QPointF(10, 10), QPointF(17, 3))
        p.drawPolyline(poly((13, 3), (17, 3), (17, 7)))
    elif t == "sensor_contact":
        p.drawEllipse(QPointF(10, 8), 4, 4)
        use(_ACCENT, 2.0)
        p.drawLine(QPointF(3, 14), QPointF(17, 14))
    elif t == "sensor_target":
        use(_PURPLE, 1.5)
        p.drawEllipse(QPointF(10, 10), 7, 7)
        p.drawEllipse(QPointF(10, 10), 3, 3)
    elif t == "sensor_joint_angle":
        p.drawLine(QPointF(4, 16), QPointF(17, 16))
        p.drawLine(QPointF(4, 16), QPointF(14, 5))
        use(_ACCENT, 1.4)
        p.drawArc(QRectF(-3, 9, 14, 14), 0, 48 * 16)
    elif t == "sensor_fuel":
        p.drawArc(QRectF(3, 5, 14, 14), 0, 180 * 16)
        use(_FLAME, 1.6)
        p.drawLine(QPointF(10, 12), QPointF(15, 7))
    elif t == "sensor_time":
        p.drawEllipse(QPointF(10, 10), 7, 7)
        use(_ACCENT, 1.6)
        p.drawLine(QPointF(10, 10), QPointF(10, 5))
        p.drawLine(QPointF(10, 10), QPointF(13.5, 12))
    elif t == "region_goal":
        use(_GREEN, 1.4, QColor(61, 220, 132, 70))
        p.drawRect(QRectF(3, 5, 14, 11))
    elif t == "region_hazard":
        use(_RED, 1.4, QColor(255, 99, 99, 70))
        p.drawRect(QRectF(3, 5, 14, 11))
        p.drawLine(QPointF(10, 7.5), QPointF(10, 11.5))
        p.drawPoint(QPointF(10, 14))
    elif t == "region_pad":
        use(_GREEN, 1.0, QColor(61, 220, 132, 50))
        p.drawRect(QRectF(3, 8, 14, 7))
        use(_GREEN, 3.0)
        p.drawLine(QPointF(3, 15), QPointF(17, 15))
    elif t == "checkpoints":
        use(_YELLOW, 1.6)
        for x in (4, 10, 16):
            p.drawLine(QPointF(x, 4), QPointF(x - 1, 16))
    elif t == "spawn":
        use(_PURPLE, 1.6)
        p.drawEllipse(QPointF(8, 12), 4, 4)
        p.drawLine(QPointF(8, 12), QPointF(17, 4))
        p.drawPolyline(poly((13, 4), (17, 4), (17, 8)))


def tool_icon(tool_id: str, fg: QColor | None = None) -> QIcon:
    """도구 아이콘(QPainter로 그린 작은 그림). fg는 선 색(기본: 밝은 회색, 팔레트는 글자색을 넘긴다)."""
    pm = QPixmap(ICON_SIZE * 2, ICON_SIZE * 2)
    pm.setDevicePixelRatio(2.0)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    _draw_icon(p, tool_id, fg if fg is not None else _FG)
    p.end()
    return QIcon(pm)


class Palette(QWidget):
    """도구 목록. set_tool(도구 id)로 바깥(캔버스)에서 켠 도구를 따라 표시한다(신호는 내지 않는다)."""

    tool_changed = Signal(str)
    owner_changed = Signal(str)
    """새 물체의 소유: agent 또는 world."""
    snap_changed = Signal(bool, float)
    """격자 맞춤 켬/끔, 간격(m)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._tool_id = "select"
        self.buttons: dict[str, QToolButton] = {}
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(2)
        for group, ids in TOOL_GROUPS:
            title = QLabel(GROUP_LABELS[group])
            title.setStyleSheet("font-weight: bold; color: #8a93a6; padding-top: 6px;")
            lay.addWidget(title)
            if group == "shapes":
                lay.addWidget(self._owner_row())
            for tool_id in ids:
                lay.addWidget(self._button(tool_id))
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        lay.addWidget(line)
        lay.addWidget(self._snap_row())
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.setMinimumWidth(170)
        self.buttons["select"].setChecked(True)

    def _button(self, tool_id: str) -> QToolButton:
        b = QToolButton()
        b.setText(TOOL_LABELS[tool_id])
        b.setIcon(tool_icon(tool_id, self.palette().color(QPalette.ColorRole.WindowText)))
        b.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        b.setAutoRaise(True)
        b.setToolTip(_INSTANT_HINTS.get(tool_id) or TOOL_HINTS.get(tool_id, ""))
        b.setObjectName(f"tool_{tool_id}")
        if tool_id in INSTANT_TOOLS:
            b.setCheckable(False)
            b.clicked.connect(lambda _=False, t=tool_id: self.tool_changed.emit(t))
        else:
            b.setCheckable(True)
            self.group.addButton(b)
            b.clicked.connect(lambda _=False, t=tool_id: self._choose(t))
        self.buttons[tool_id] = b
        return b

    def _owner_row(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(4, 0, 0, 2)
        row.addWidget(QLabel(S.EDC_OWNER_LABEL))
        self.owner_agent = QRadioButton(S.EDC_OWNER_AGENT)
        self.owner_world = QRadioButton(S.EDC_OWNER_WORLD)
        self.owner_agent.setChecked(True)
        self.owner_agent.toggled.connect(lambda on: on and self.owner_changed.emit("agent"))
        self.owner_world.toggled.connect(lambda on: on and self.owner_changed.emit("world"))
        row.addWidget(self.owner_agent)
        row.addWidget(self.owner_world)
        row.addStretch(1)
        return w

    def _snap_row(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(4, 0, 0, 0)
        self.snap_check = QCheckBox(S.EDC_SNAP)
        self.snap_check.setChecked(True)
        self.snap_step = QDoubleSpinBox()
        self.snap_step.setDecimals(2)
        self.snap_step.setRange(0.01, 100.0)
        self.snap_step.setSingleStep(0.1)
        self.snap_step.setValue(0.5)
        self.snap_step.setSuffix(S.EDC_SNAP_UNIT)
        self.snap_check.setToolTip(S.EDC_SNAP_TIP)
        self.snap_check.toggled.connect(self._emit_snap)
        self.snap_step.valueChanged.connect(self._emit_snap)
        row.addWidget(self.snap_check)
        row.addWidget(self.snap_step)
        return w

    def _emit_snap(self, *_args: object) -> None:
        self.snap_changed.emit(self.snap_check.isChecked(), float(self.snap_step.value()))

    def _choose(self, tool_id: str) -> None:
        if tool_id != self._tool_id:
            self._tool_id = tool_id
        self.tool_changed.emit(tool_id)

    @property
    def tool_id(self) -> str:
        return self._tool_id

    def set_tool(self, tool_id: str) -> None:
        """바깥에서 바뀐 도구를 표시만 한다(tool_changed를 내지 않는다)."""
        b = self.buttons.get(tool_id)
        if b is None or not b.isCheckable():
            return
        self._tool_id = tool_id
        b.setChecked(True)

    def select_tool(self, tool_id: str) -> None:
        """도구를 켜고 tool_changed를 낸다."""
        if tool_id in INSTANT_TOOLS:
            self.tool_changed.emit(tool_id)
            return
        self.set_tool(tool_id)
        self.tool_changed.emit(tool_id)

    def keyPressEvent(self, ev: QKeyEvent) -> None:  # noqa: N802
        if ev.key() == Qt.Key.Key_Escape:
            self.select_tool("select")
            return
        super().keyPressEvent(ev)

    def changeEvent(self, ev: QEvent) -> None:  # noqa: N802
        if ev.type() == QEvent.Type.PaletteChange:  # 밝은·어두운 테마가 바뀌면 아이콘 선 색도 바꾼다
            fg = self.palette().color(QPalette.ColorRole.WindowText)
            for tool_id, b in self.buttons.items():
                b.setIcon(tool_icon(tool_id, fg))
        super().changeEvent(ev)

    def connect_canvas(self, canvas: object) -> None:
        """캔버스와 양쪽으로 잇는다: 팔레트 -> canvas.set_tool / set_new_owner / set_snap, 캔버스 -> set_tool."""
        self.tool_changed.connect(canvas.set_tool)  # type: ignore[attr-defined]
        canvas.tool_changed.connect(self.set_tool)  # type: ignore[attr-defined]
        self.owner_changed.connect(canvas.set_new_owner)  # type: ignore[attr-defined]
        self.snap_changed.connect(canvas.set_snap)  # type: ignore[attr-defined]
        self.set_tool(canvas.tool_id)  # type: ignore[attr-defined]

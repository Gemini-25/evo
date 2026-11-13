"""틀 카드·슬라이더 위젯·보관함 창 (6.2.0, 설계 DU 4.2·7.3, 최종 설계 3.1 W).

- FlowLayout: 폭에 맞춰 줄을 바꾸는 배치(규칙 문장의 글과 빈칸, 카드 격자).
- TemplateCard: 그림(template.thumbnail) + 이름의 고르는 카드(QToolButton, 체크형).
- ParamSliders: 틀의 겉 슬라이더(셋까지). 범위가 0~1이고 간격이 1인 슬라이더는 체크 상자(「균형 보조」).
- TemplatePicker: 에디터 팔레트 「몸 틀…」「물건…」이 여는 보관함 창. choice()가 (틀, 값, 뒤집기, 진화할 슬라이더)를 돌려준다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PySide6.QtCore import QPoint, QPointF, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from evolab.app import strings as S
from evolab.app import style
from evolab.presets.library import Library, default_library
from evolab.presets.template import DrawOp, Param, Template, resolve_slots, thumbnail

THUMB = (96, 72)
"""카드 그림 크기(논리 화소)."""
CARD_WIDTH = 120
"""카드 하나의 폭: 그림 + 이름 두 줄."""
_ROLE_COLORS = {"agent": "#eb6834", "world": "#898781", "region": "#2a78d6"}
"""그림 색(그래프 범주 색·그림의 흐린 색에서: 새 색을 만들지 않음, 다듬기 계획 1.2)."""


# ---------------------------------------------------------------------------
# 줄을 바꾸는 배치
# ---------------------------------------------------------------------------


class FlowLayout(QLayout):
    """왼쪽에서 오른쪽으로 놓고 폭이 모자라면 다음 줄로(Qt 예제의 흐름 배치). 높이는 폭에 따라 정해진다."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 4) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._gap = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802
        self._items.append(item)

    def add(self, widget: QWidget) -> None:
        self.addChildWidget(widget)
        self.addItem(QWidgetItem(widget))

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._place(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._place(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _place(self, rect: QRect, *, apply: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line = area.x(), area.y(), 0
        for item in self._items:
            w = item.widget()
            if w is not None and w.isHidden():
                continue
            hint = item.sizeHint()
            if x + hint.width() > area.right() + 1 and line > 0:
                x = area.x()
                y += line + self._gap
                line = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._gap
            line = max(line, hint.height())
        return y + line - rect.y() + m.bottom()


def clear_layout(layout: QLayout) -> None:
    """배치 안의 위젯을 모두 지운다(다시 채우기 전)."""
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            break
        w = item.widget()
        if w is not None:
            w.hide()
            w.setParent(None)
            w.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


# ---------------------------------------------------------------------------
# 카드
# ---------------------------------------------------------------------------


def thumbnail_pixmap(ops: Sequence[DrawOp], size: tuple[int, int] = THUMB, ratio: float = 2.0) -> QPixmap:
    """template.thumbnail의 그리기 목록을 그림으로(바탕은 투명, 화소 비율 2)."""
    w, h = size
    pm = QPixmap(int(w * ratio), int(h * ratio))
    pm.setDevicePixelRatio(ratio)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for op in ops:
        color = QColor(_ROLE_COLORS.get(op.role, _ROLE_COLORS["world"]))
        fill = QColor(color)
        fill.setAlpha(110 if op.role != "region" else 50)
        p.setPen(QPen(color, 1.2))
        p.setBrush(fill)
        if op.kind == "circle" and op.points:
            c = op.points[0]
            r = max(op.radius, 1.0)
            p.drawEllipse(QPointF(c[0], c[1]), r, r)
        elif op.kind == "poly" and op.points:
            p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in op.points]))
        elif op.kind == "line" and len(op.points) >= 2:
            path = QPainterPath(QPointF(*op.points[0]))
            for x, y in op.points[1:]:
                path.lineTo(QPointF(x, y))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
    p.end()
    return pm


def template_icon(tpl: Template, values: Mapping[str, float] | None = None) -> QIcon:
    try:
        ops = thumbnail(tpl, THUMB, values)
    except Exception:  # noqa: BLE001 - 그림은 꾸밈이라 못 그리면 빈 그림
        ops = []
    return QIcon(thumbnail_pixmap(ops))


class Card(QToolButton):
    """고르는 카드: 위에 그림, 아래 이름(두 줄까지). 체크형이고 키보드로 고를 수 있다(Tab·Space)."""

    def __init__(self, key: str, title: str, tip: str = "", icon: QIcon | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self.setCheckable(True)
        self.setAutoRaise(False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon if icon is not None
                                else Qt.ToolButtonStyle.ToolButtonTextOnly)
        if icon is not None:
            self.setIcon(icon)
            self.setIconSize(QSize(*THUMB))
        self.title = title
        self.setText(self._wrapped(title))
        self.setToolTip(tip)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def _wrapped(self, text: str) -> str:
        """카드 폭을 넘는 이름은 낱말 사이에서 한 번 꺾는다."""
        fm = self.fontMetrics()
        limit = CARD_WIDTH - 12
        if fm.horizontalAdvance(text) <= limit or " " not in text:
            return text
        words = text.split(" ")
        for k in range(len(words) - 1, 0, -1):
            head = " ".join(words[:k])
            if fm.horizontalAdvance(head) <= limit:
                return head + "\n" + " ".join(words[k:])
        return text

    def sizeHint(self) -> QSize:  # noqa: N802
        hint = super().sizeHint()
        return QSize(max(CARD_WIDTH, hint.width()), hint.height())


class TemplateCard(Card):
    """틀 하나의 카드(DU 7.3). 이름·풍선은 틀의 name·description(내장 틀은 strings.py의 id로 찾은 것)."""

    def __init__(self, tpl: Template, parent: QWidget | None = None) -> None:
        tip = S.TPL_CARD_TIP.format(name=tpl.name, description=tpl.description) if tpl.description else tpl.name
        if tpl.source != "builtin":
            tip = f"{tip}\n{S.TPL_MINE}"
        super().__init__(tpl.id, tpl.name, tip, template_icon(tpl), parent)
        self.template = tpl


# ---------------------------------------------------------------------------
# 슬라이더
# ---------------------------------------------------------------------------


def is_switch(p: Param) -> bool:
    """0~1, 간격 1인 슬라이더는 켬·끔(체크 상자)."""
    return p.min == 0.0 and p.max == 1.0 and p.step >= 1.0


class ParamSliders(QWidget):
    """틀의 겉 슬라이더(셋까지). values()는 지금 값, changed는 사용자가 바꿀 때마다."""

    changed = Signal()

    def __init__(self, tpl: Template | None = None, values: Mapping[str, float] | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.template: Template | None = None
        self._rows: dict[str, tuple[Param, QWidget, QLabel | None]] = {}
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(style.SPACE[1])
        self._grid.setColumnStretch(1, 1)
        if tpl is not None:
            self.set_template(tpl, values)

    def set_template(self, tpl: Template, values: Mapping[str, float] | None = None) -> None:
        clear_layout(self._grid)
        self._rows = {}
        self.template = tpl
        values = dict(values or {})
        for row, p in enumerate(tpl.main_params()):
            v = p.clamp(values.get(p.id, p.value))
            tip = p.tip
            if is_switch(p):
                box = QCheckBox(p.label)
                box.setChecked(v >= 0.5)
                box.setToolTip(tip)
                box.toggled.connect(lambda _on: self.changed.emit())
                self._grid.addWidget(box, row, 0, 1, 3)
                self._rows[p.id] = (p, box, None)
                continue
            name = QLabel(p.label)
            name.setToolTip(tip)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, self._ticks(p))
            slider.setPageStep(max(1, self._ticks(p) // 10))
            slider.setValue(self._index(p, v))
            slider.setToolTip(tip)
            slider.setAccessibleName(p.label)
            slider.setMinimumWidth(120)
            value = QLabel()
            value.setToolTip(tip)
            value.setMinimumWidth(value.fontMetrics().horizontalAdvance(
                S.TPL_PARAM_VALUE.format(value=max(abs(p.min), abs(p.max)) * -1.0, unit=p.unit)) + 4)
            name.setBuddy(slider)
            slider.valueChanged.connect(lambda _i, pid=p.id: self._moved(pid))
            self._grid.addWidget(name, row, 0)
            self._grid.addWidget(slider, row, 1)
            self._grid.addWidget(value, row, 2)
            self._rows[p.id] = (p, slider, value)
            self._show_value(p.id)

    @staticmethod
    def _ticks(p: Param) -> int:
        return max(1, int(round((p.max - p.min) / p.step))) if p.step > 0 else 100

    def _index(self, p: Param, v: float) -> int:
        n = self._ticks(p)
        span = p.max - p.min
        return 0 if span <= 0 else int(round((v - p.min) / span * n))

    def _value_at(self, p: Param, i: int) -> float:
        n = self._ticks(p)
        return p.clamp(round(p.min + (p.max - p.min) * i / n, 6))

    def _show_value(self, pid: str) -> None:
        p, widget, label = self._rows[pid]
        if label is not None and isinstance(widget, QSlider):
            label.setText(S.TPL_PARAM_VALUE.format(value=self._value_at(p, widget.value()), unit=p.unit))

    def _moved(self, pid: str) -> None:
        self._show_value(pid)
        self.changed.emit()

    def values(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for pid, (p, widget, _label) in self._rows.items():
            if isinstance(widget, QCheckBox):
                out[pid] = 1.0 if widget.isChecked() else 0.0
            elif isinstance(widget, QSlider):
                out[pid] = self._value_at(p, widget.value())
        return out

    def set_values(self, values: Mapping[str, float]) -> None:
        """값만 바꾼다(신호 없이). 위젯은 그대로라 초점이 남는다."""
        for pid, (p, widget, _label) in self._rows.items():
            v = p.clamp(values.get(pid, p.value))
            widget.blockSignals(True)
            if isinstance(widget, QCheckBox):
                widget.setChecked(v >= 0.5)
            elif isinstance(widget, QSlider):
                widget.setValue(self._index(p, v))
            widget.blockSignals(False)
            self._show_value(pid)

    def widget_for(self, pid: str) -> QWidget:
        return self._rows[pid][1]

    def param_ids(self) -> list[str]:
        return list(self._rows)


# ---------------------------------------------------------------------------
# 보관함 창
# ---------------------------------------------------------------------------


class TemplatePicker(QDialog):
    """틀 보관함(DU 4.2·7.3): 왼쪽 카드 격자, 오른쪽 설명·슬라이더·「좌우 뒤집어 넣기」, 「넣기」.

    kind: "body" | "prop" | "arena", world: "side" | "top", data: 지금 환경 사전(빈자리를 채울 수 있나 보려고).
    「넣기」를 누르면 accept()하고 choice()가 (틀, 값, 뒤집기, 진화할 슬라이더)를 돌려준다(6.2.0은 진화할 슬라이더가 늘 빈 튜플).
    빈자리를 못 채우는 틀이면 그 까닭(TPL_NEEDS)과 「함께 넣기」 체크가 뜬다: 켜면 needs_add()가 먼저 넣을 틀을 돌려준다.
    """

    def __init__(self, kind: str, world: str, data: Mapping[str, Any], parent: QWidget | None = None, *,
                 library: Library | None = None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.world = world
        self.data = data
        self.library = library or default_library()
        self.setWindowTitle(S.TPL_PICKER_TITLE.get(kind, S.TPL_PICKER_TITLE["body"]))
        self.resize(720, 460)
        self._chosen: Template | None = None
        self._needs: list[Template] = []

        self.cards: list[TemplateCard] = []
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        grid_host = QWidget()
        self.grid = FlowLayout(grid_host, spacing=style.SPACE[1])
        templates = self.library.templates(kind, world)
        for tpl in templates:
            card = TemplateCard(tpl)
            self.grid.add(card)
            self.group.addButton(card)
            self.cards.append(card)
            card.toggled.connect(lambda on, c=card: on and self._pick(c.template))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(grid_host)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.title = QLabel()
        self.title.setFont(style.font(self.title, "title"))
        self.title.setWordWrap(True)
        self.blurb = style.note()
        self.sliders = ParamSliders()
        self.mirror = QCheckBox(S.TPL_MIRROR)
        self.mirror.setToolTip(S.TPL_MIRROR_TIP)
        self.needs_note = style.note()
        self.needs_box = QCheckBox()
        self.needs_box.setToolTip(S.TPL_NEEDS_ADD_TIP)
        self.needs_box.setChecked(True)
        problems = self.library.problems()
        self.problems_note = style.note(S.TPL_LIB_PROBLEMS.format(n=len(problems)) if problems else "")
        self.problems_note.setVisible(bool(problems))
        if problems:
            self.problems_note.setToolTip("\n".join(f"{p.name}: {why}" for p, why in problems))

        side = QVBoxLayout()
        side.addWidget(self.title)
        side.addWidget(self.blurb)
        side.addWidget(self.sliders)
        side.addWidget(self.mirror)
        side.addWidget(self.needs_note)
        side.addWidget(self.needs_box)
        side.addStretch(1)
        side.addWidget(self.problems_note)
        self.btn_insert = QPushButton(S.TPL_INSERT)
        self.btn_insert.setToolTip(S.TPL_INSERT_TIP)
        style.set_role(self.btn_insert, "primary")
        self.btn_insert.setDefault(True)
        self.btn_insert.clicked.connect(self.accept)
        self.btn_cancel = QPushButton(S.ED_REPLICATE_CANCEL)  # 에디터 대화 상자의 「취소」(같은 글의 상수는 하나)
        self.btn_cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.btn_insert)
        buttons.addWidget(self.btn_cancel)
        side.addLayout(buttons)
        side_host = QWidget()
        side_host.setLayout(side)
        side_host.setFixedWidth(300)

        root = QHBoxLayout(self)
        root.addWidget(scroll, 1)
        root.addWidget(side_host)
        self.mirror.setVisible(kind != "arena")
        if self.cards:
            self.cards[0].setChecked(True)
        else:
            style.set_enabled(self.btn_insert, False, S.TPL_WRONG_WORLD.format(world=S.TPL_WORLD_NAMES[world]))
            self._pick(None)

    def _pick(self, tpl: Template | None) -> None:
        self._chosen = tpl
        if tpl is None:
            self.title.setText("")
            self.blurb.setText("")
            self.needs_note.hide()
            self.needs_box.hide()
            return
        self.title.setText(tpl.name)
        self.blurb.setText(tpl.description)
        self.sliders.set_template(tpl)
        _filled, missing = resolve_slots(tpl, self.data)
        self._needs = []
        names: list[str] = []
        for name in missing:
            slot = tpl.slots.get(name)
            label = (slot.label if slot is not None and slot.label else name.lstrip("$"))
            names.append(label)
            if slot is None:
                continue
            for cand in self.library.templates("prop", self.world):
                tags = {t for b in cand.parts.get("bodies", []) if isinstance(b, Mapping)
                        for t in (b.get("tags") or []) if isinstance(t, str)}
                if tags & set(slot.tags) and cand not in self._needs:
                    self._needs.append(cand)
                    break
        if names:
            self.needs_note.setText(S.TPL_NEEDS.format(what=", ".join(names)))
            self.needs_note.show()
        else:
            self.needs_note.hide()
        if self._needs:
            self.needs_box.setText(S.TPL_NEEDS_ADD.format(what=", ".join(t.name for t in self._needs)))
            self.needs_box.show()
        else:
            self.needs_box.hide()

    def choice(self) -> tuple[Template, dict[str, float], bool, tuple[str, ...]] | None:
        """(틀, 슬라이더 값, 좌우 뒤집기, 진화할 슬라이더). 고른 것이 없거나 취소했으면 None."""
        if self._chosen is None or self.result() != QDialog.DialogCode.Accepted:
            return None
        return self._chosen, self.sliders.values(), self.mirror.isChecked() and self.kind != "arena", ()

    def needs_add(self) -> list[Template]:
        """「함께 넣기」를 켰으면 이 틀보다 먼저 넣을 틀(빈자리를 채울 물건). 아니면 빈 목록."""
        if self.needs_box.isHidden() or not self.needs_box.isChecked():
            return []
        return list(self._needs)


__all__ = ["Card", "FlowLayout", "ParamSliders", "TemplateCard", "TemplatePicker", "clear_layout", "is_switch",
           "template_icon", "thumbnail_pixmap"]

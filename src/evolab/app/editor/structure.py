"""에디터 아래쪽 「구조」 탭과 「배열로 늘리기」 대화 상자 (v2.1 몸 구조 진화, 설계 5.1·D43).

- 구조 탭: 진화로 없어질 수 있는 물체(spec.structure.optional_bodies, 검사·런타임과 같은 집합)마다 한 줄 [물체, 있음(값·수식),
  진화, 처음 상태, 함께 사라지는 부품 수]. 줄을 고르면 그 물체를 고르고(속성 패널), 그 물체가 없을 때 함께 없어지는 부품
  (attached_parts: 관절·모터·구동기·센서와 이어짐 규칙으로 끊기는 물체)을 캔버스에 강조한다(highlight_changed). 「진화」
  칸은 속성 패널의 있음 진화 체크와 같은 조작(forms.evolve_exists / fix_exists)이다. 비면 ED_STRUCT_EMPTY.
- 몸통(뿌리 물체)은 없앨 수 없다. 몸통을 고르면 ED_STRUCT_ROOT_LOCKED를 보인다(속성 패널의 있음 칸도 잠긴다).
- 「없는 부품도 보기」(show_absent_changed): 처음 몸에 없는 부품을 캔버스에 옅은 점선으로 그릴지. 끄면 처음 몸 그대로.
- 「배열로 늘리기…」(replicate_requested): 고른 물체를 늘리는 대화 상자(ReplicateDialog)를 연다. 실제 복사는 tools.replicate.

문서를 바꾸는 것은 모두 EnvDocument의 편집 함수를 거친다(되돌리기 한 단계). 탭은 보일 때만 다시 계산한다.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QHideEvent, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from evolab.app import strings as S
from evolab.app.editor import tools as T
from evolab.app.editor.document import EnvDocument, Ref
from evolab.app.editor.forms import evolve_exists, exists_param, fix_exists, fmt_value, plain_param, root_body_id
from evolab.spec import model as m
from evolab.spec import structure

COL_BODY, COL_EXISTS, COL_EVOLVE, COL_INITIAL, COL_PARTS = range(5)
_PARAM_NAME_RE = re.compile(r"(?<![\w.])p\.(\w+)")
"""수식 안의 p.<매개변수 id>."""
REFRESH_DEBOUNCE_MS = 30
"""문서가 바뀐 뒤 탭을 다시 계산하기까지(타이핑 한 번마다 다시 하지 않게)."""


# ---------------------------------------------------------------------------
# 계산 (화면 없이도 부른다: 캔버스와 테스트가 같이 쓴다)
# ---------------------------------------------------------------------------


def _items(data: Any, section: str) -> list[dict[str, Any]]:
    v = data.get(section) if isinstance(data, dict) else None
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


def attached_parts(data: dict[str, Any], bodies: Iterable[str]) -> list[Ref]:
    """bodies가 없으면 함께 없어지는 부품(문서 순서): 끝 하나가 그 물체인 관절, 그 관절의 모터, 그 물체의 추진기·차량·점프·
    공기역학, 그 물체·관절에 단 센서. 런타임 규칙(설계 D35, Episode)과 같다. 연료 탱크는 남는다."""
    gone = set(bodies)
    out: list[Ref] = []
    joints: set[str] = set()
    for j in _items(data, "joints"):
        if j.get("a") in gone or j.get("b") in gone:
            joints.add(str(j.get("id")))
            out.append(("joints", str(j.get("id"))))
    for a in _items(data, "actuators"):
        if a.get("kind") == "fuel_tank":
            continue
        if a.get("body") in gone or (a.get("kind") == "motor" and a.get("joint") in joints):
            out.append(("actuators", str(a.get("id"))))
    for s in _items(data, "sensors"):
        if s.get("body") in gone or (s.get("kind") == "joint_angle" and s.get("joint") in joints):
            out.append(("sensors", str(s.get("id"))))
    return out


def gone_with(env: m.EnvFile, data: dict[str, Any], body: str) -> list[Ref]:
    """이 물체가 없을 때 함께 없어지는 부품(자기 자신 제외): 이어짐 규칙으로 끊기는 물체(D34)와 attached_parts."""
    lost = structure.absent_from(env, {body})
    order = [str(b.get("id")) for b in _items(data, "bodies")]
    others = [("bodies", b) for b in order if b in lost and b != body]
    return others + attached_parts(data, lost)


def _eval_exists(value: Any, params: dict[str, float]) -> float | None:
    """있음 값(수식은 p.*만으로 계산). 계산할 수 없으면 None. 유한하지 않으면 0(structure.exists_value와 같은 규칙)."""
    if value is None:
        return 1.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        x = float(value)
    elif isinstance(value, str) and value.startswith("="):
        from evolab.scoring.expr import Sym, compile_expr, formula_body

        try:
            fn = compile_expr(formula_body(value), {k: Sym("num", "build") for k in params}, want="num",
                              max_phase="build")
            x = float(fn(params))
        except Exception:  # noqa: BLE001 (고치는 중인 수식)
            return None
    else:
        return None
    return x if math.isfinite(x) else 0.0


def initial_absent(doc: EnvDocument, env: m.EnvFile | None = None) -> frozenset[str] | None:
    """처음 몸(매개변수 값)에서 없는 에이전트 물체. 컴파일되면 런타임과 같은 함수(structure.present), 검사 오류가 있으면
    있음 칸만 p.*로 계산해 이어짐 규칙을 적용한다. 구조가 잘못돼 알 수 없으면 None."""
    spec = doc.spec()
    if spec is not None:
        agents = frozenset(b.id for b in spec.file.bodies if b.owner == "agent")
        return agents - structure.present(spec, structure.param_values(spec.file))
    if env is None:
        env = parse_or_none(doc)
    if env is None:
        return None
    params = structure.param_values(env)
    off: set[str] = set()
    for b in env.bodies:
        if b.owner != "agent" or b.exists is None:
            continue
        x = _eval_exists(b.exists, params)
        if x is None:
            return None
        if not x > structure.EXISTS_THRESHOLD:
            off.add(b.id)
    return structure.absent_from(env, off)


def parse_or_none(doc: EnvDocument) -> m.EnvFile | None:
    """문서의 모델(컴파일된 것이 있으면 그 파일, 아니면 구조 검사만). 구조가 잘못돼 읽을 수 없으면 None."""
    from evolab.spec.compiled import EnvFileError
    from evolab.spec.io import parse_env

    spec = doc.spec()  # 문서가 컴파일을 맡아 둔다(같은 내용이면 다시 하지 않음)
    if spec is not None:
        return spec.file
    try:
        return parse_env(doc.data)
    except (EnvFileError, ValueError, TypeError):
        return None


@dataclass(frozen=True)
class StructureRow:
    """구조 탭 한 줄."""

    body: str
    label: str
    exists: Any
    """있음 칸 값(None = 늘 있음, 숫자, 수식)."""
    param: str | None
    """있음 칸이 진화하는 매개변수 하나를 가리키면 그 id(진화 칸 체크)."""
    evolvable: bool
    """진화 칸을 켜고 끌 수 있나(비었거나 숫자거나 매개변수 하나)."""
    initial: bool | None
    """처음 몸에 있나(None = 알 수 없음)."""
    parts: tuple[Ref, ...]
    """함께 사라지는 부품(자기 자신 제외)."""
    driven_by: str | None = None
    """있음 칸이 매개변수 하나가 아닌 수식인데 진화하는 매개변수를 쓰면 그 첫 id(선언 순서, 예: 「개수를 몸 수치 하나로」의
    <이름>_count). 진화 칸은 켜진 채 잠기고(그 매개변수를 고쳐야 바뀜) 풍선 도움말이 그 이름을 알린다."""


def evolving_params_in(data: dict[str, Any], formula: Any) -> list[str]:
    """수식 formula가 쓰는 진화 매개변수 id(매개변수 선언 순서). 수식이 아니면 빈 목록."""
    if not isinstance(formula, str) or not formula.startswith("="):
        return []
    used = set(_PARAM_NAME_RE.findall(formula))
    return [str(p.get("id")) for p in _items(data, "params") if p.get("evolve") and str(p.get("id")) in used]


def structure_rows(doc: EnvDocument) -> list[StructureRow]:
    """구조 탭 줄: 없어질 수 있는 물체마다(선언 순서). 구조가 잘못된 문서면 빈 목록."""
    env = parse_or_none(doc)
    if env is None:
        return []
    optional = structure.optional_bodies(env)
    if not optional:
        return []
    absent = initial_absent(doc, env)
    data = doc.data
    params = {str(p.get("id")) for p in _items(data, "params")}
    rows: list[StructureRow] = []
    for b in _items(data, "bodies"):
        bid = str(b.get("id"))
        if bid not in optional:
            continue
        value = b.get("exists")
        linked = plain_param(value)
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        param = exists_param(doc, bid)
        driven = None if param is not None or linked is not None else next(iter(evolving_params_in(data, value)), None)
        rows.append(StructureRow(
            bid, str(b.get("label") or bid), value, param,
            value is None or number or (linked is not None and linked in params),
            None if absent is None else bid not in absent, tuple(gone_with(env, data, bid)), driven))
    return rows


def part_label(data: dict[str, Any], ref: Ref) -> str:
    for item in _items(data, ref[0]):
        if str(item.get("id")) == ref[1]:
            return str(item.get("label") or ref[1])
    return ref[1]


# ---------------------------------------------------------------------------
# 구조 탭
# ---------------------------------------------------------------------------


class StructureTab(QWidget):
    """아래쪽 「구조」 탭."""

    highlight_changed = Signal(list)
    """캔버스에 강조할 부품(list[Ref]: 고른 물체와 함께 사라지는 부품). 빈 목록이면 강조를 끈다."""
    show_absent_changed = Signal(bool)
    replicate_requested = Signal(str)
    """「배열로 늘리기…」: 늘릴 물체 id."""

    def __init__(self, doc: EnvDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.rows: list[StructureRow] = []
        self.refreshes = 0
        """다시 계산한 횟수(테스트·측정용)."""
        self._dirty = True
        self._syncing = False
        hint = QLabel(S.ED_STRUCT_HINT)
        hint.setWordWrap(True)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(list(S.ED_STRUCT_COLUMNS))
        self.table.horizontalHeaderItem(COL_PARTS).setToolTip(S.ED_STRUCT_PARTS_TIP)
        self.table.horizontalHeaderItem(COL_EXISTS).setToolTip(S.EDP_FIELD_TIPS.get("exists", ""))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_BODY, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_EXISTS, QHeaderView.ResizeMode.Stretch)
        for col in (COL_EVOLVE, COL_INITIAL, COL_PARTS):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        self.table.itemChanged.connect(self._on_item_changed)
        self.empty = QLabel(S.ED_STRUCT_EMPTY)
        self.empty.setWordWrap(True)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.replicate_button = QPushButton(S.ED_REPLICATE)
        self.replicate_button.clicked.connect(self._on_replicate)
        self.show_absent = QCheckBox(S.ED_STRUCT_SHOW_ABSENT)
        self.show_absent.setToolTip(S.ED_STRUCT_SHOW_ABSENT_TIP)
        self.show_absent.setChecked(True)
        self.show_absent.toggled.connect(self.show_absent_changed)
        self.root_note = QLabel(S.ED_STRUCT_ROOT_LOCKED)
        self.root_note.setStyleSheet("color: palette(placeholder-text);")
        self.root_note.hide()
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(self.replicate_button)
        bar.addWidget(self.root_note)
        bar.addStretch(1)
        bar.addWidget(self.show_absent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(hint)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.empty, 1)
        lay.addLayout(bar)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(REFRESH_DEBOUNCE_MS)
        self._timer.timeout.connect(self._refresh_if_visible)
        doc.changed.connect(self._on_doc_changed)
        doc.selection_changed.connect(self._on_selection)
        self.refresh()

    # ---- 갱신 ----
    def _on_doc_changed(self) -> None:
        self._dirty = True
        self._timer.start()

    def _refresh_if_visible(self) -> None:
        if self.isVisible():
            self.refresh()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        if self._dirty:
            self.refresh()
        else:
            self._emit_highlight()

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        super().hideEvent(event)
        self.highlight_changed.emit([])

    def refresh(self) -> None:
        """줄을 다시 계산해 표를 채운다(고른 줄은 지킨다)."""
        self._dirty = False
        self.refreshes += 1
        self.rows = structure_rows(self.doc)
        data = self.doc.data
        with QSignalBlocker(self.table):
            self.table.setRowCount(len(self.rows))
            for r, row in enumerate(self.rows):
                name = QTableWidgetItem(row.label)
                name.setToolTip(row.body)
                name.setData(Qt.ItemDataRole.UserRole, row.body)
                exists = QTableWidgetItem(fmt_value(row.exists) if row.exists is not None else S.ED_STRUCT_ON)
                evolve = QTableWidgetItem()
                flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable
                if row.evolvable:
                    flags |= Qt.ItemFlag.ItemIsEnabled
                evolve.setFlags(flags)
                evolving = bool(row.param or row.driven_by)
                evolve.setCheckState(Qt.CheckState.Checked if evolving else Qt.CheckState.Unchecked)
                if row.param:
                    tip = S.EDP_EVOLVE_ON_TIP.format(id=row.param)
                elif row.driven_by:  # 진화하는 몸 수치를 쓰는 수식(개수 하나로 묶은 부품): 켜진 채 잠금
                    tip = S.ED_STRUCT_EVOLVE_BY_PARAM_TIP.format(id=row.driven_by)
                else:
                    tip = S.EDP_EVOLVE_TIP if row.evolvable else S.EDP_EVOLVE_FORMULA_TIP
                evolve.setToolTip(tip)
                initial = QTableWidgetItem(S.ED_STRUCT_UNKNOWN if row.initial is None else
                                           (S.ED_STRUCT_ON if row.initial else S.ED_STRUCT_OFF))
                parts = QTableWidgetItem(str(len(row.parts)))
                parts.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                parts.setToolTip(", ".join(part_label(data, p) for p in row.parts) or S.ED_STRUCT_PARTS_TIP)
                for col, item in ((COL_BODY, name), (COL_EXISTS, exists), (COL_EVOLVE, evolve),
                                  (COL_INITIAL, initial), (COL_PARTS, parts)):
                    self.table.setItem(r, col, item)
        self.table.setVisible(bool(self.rows))
        self.empty.setVisible(not self.rows)
        self._on_selection(list(self.doc.selection))

    # ---- 고르기 ----
    def row_of(self, body: str) -> int:
        return next((i for i, row in enumerate(self.rows) if row.body == body), -1)

    def current_body(self) -> str | None:
        """구조 탭에서 고른 줄의 물체."""
        r = self.table.currentRow()
        sel = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self.rows[r].body if sel and 0 <= r < len(self.rows) else None

    def select_body(self, body: str | None) -> None:
        """그 물체의 줄을 고른다(없으면 고른 줄을 지운다). 문서 선택도 그 물체로."""
        r = self.row_of(body) if body is not None else -1
        if r < 0:
            self.table.clearSelection()
            return
        self.table.selectRow(r)

    def highlight(self) -> list[Ref]:
        """지금 강조할 부품: 고른 줄의 물체와 함께 사라지는 부품."""
        body = self.current_body()
        if body is None:
            return []
        row = self.rows[self.row_of(body)]
        return [("bodies", body), *row.parts]

    def _emit_highlight(self) -> None:
        if self.isVisible():
            self.highlight_changed.emit(self.highlight())

    def _on_row_selected(self) -> None:
        body = self.current_body()
        if not self._syncing and body is not None and self.doc.selection[:1] != [("bodies", body)]:
            self._syncing = True
            self.doc.select([("bodies", body)])
            self._syncing = False
        self._emit_highlight()
        self._update_bar()

    def _on_selection(self, refs: list[Ref]) -> None:
        if not self._syncing:
            first = refs[0] if refs else None
            body = first[1] if first is not None and first[0] == "bodies" else None
            self._syncing = True
            with QSignalBlocker(self.table):
                r = self.row_of(body) if body is not None else -1
                if r >= 0:
                    self.table.selectRow(r)
                else:
                    self.table.clearSelection()
            self._syncing = False
            self._emit_highlight()
        self._update_bar()

    def selected_body(self) -> str | None:
        """문서에서 고른 첫 물체(배열로 늘리기 대상)."""
        for section, cid in self.doc.selection:
            if section == "bodies":
                return cid
        return None

    def _update_bar(self) -> None:
        body = self.selected_body()
        root = root_body_id(self.doc.data)
        self.root_note.setVisible(body is not None and body == root)
        err = T.replicate_error(self.doc, body) if body is not None else S.ED_REPLICATE_NEED_AGENT
        self.replicate_button.setEnabled(err is None)
        self.replicate_button.setToolTip(err or S.ED_REPLICATE_HINT.format(name=body))

    # ---- 조작 ----
    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != COL_EVOLVE or not 0 <= item.row() < len(self.rows):
            return
        row = self.rows[item.row()]
        if not row.evolvable:  # 잠긴 칸(일반 수식): 켜고 끌 수 없다
            return
        on = item.checkState() == Qt.CheckState.Checked
        if on == (row.param is not None):
            return
        if on:
            evolve_exists(self.doc, row.body)
        else:
            fix_exists(self.doc, row.body)
        QTimer.singleShot(0, self.refresh)  # 이 칸의 신호가 끝난 뒤 표를 다시 채운다(칸을 지우는 중에 바꾸지 않게)

    def set_evolve(self, body: str, on: bool) -> None:
        """그 줄의 진화 칸을 켜고 끈다(마우스로 누른 것과 같다)."""
        r = self.row_of(body)
        if r >= 0:
            self.table.item(r, COL_EVOLVE).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

    def set_show_absent(self, on: bool) -> None:
        if self.show_absent.isChecked() != on:
            self.show_absent.setChecked(on)

    def _on_replicate(self) -> None:
        body = self.selected_body()
        if body is not None:
            self.replicate_requested.emit(body)


# ---------------------------------------------------------------------------
# 배열로 늘리기 대화 상자
# ---------------------------------------------------------------------------


def _spin(value: float, low: float = -1e4, high: float = 1e4, decimals: int = 3) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSingleStep(0.1)
    box.setValue(value)
    return box


class ReplicateDialog(QDialog):
    """「배열로 늘리기」: 개수, 놓는 방식(직선·원·좌우 거울)과 그 자리, 있음/없음도 진화, 개수를 몸 수치 하나로."""

    def __init__(self, doc: EnvDocument, body: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.body = body
        self.setWindowTitle(S.ED_REPLICATE_TITLE)
        item = doc.item(("bodies", body)) or {}
        name = str(item.get("label") or body)
        x, y, w, cx, cy = self._defaults()
        hint = QLabel(S.ED_REPLICATE_HINT.format(name=name))
        hint.setWordWrap(True)
        self.count = QSpinBox()
        self.count.setRange(*T.REPLICATE_COUNT)
        self.count.setValue(T.REPLICATE_COUNT[0])
        self.layout_box = QComboBox()
        for key in T.REPLICATE_LAYOUTS:
            self.layout_box.addItem(S.ED_REPLICATE_LAYOUT[key], key)
            self.layout_box.setItemData(self.layout_box.count() - 1, S.ED_REPLICATE_LAYOUT_TIPS[key],
                                        Qt.ItemDataRole.ToolTipRole)
        self.layout_tip = QLabel()
        self.layout_tip.setWordWrap(True)
        self.layout_tip.setStyleSheet("color: palette(placeholder-text);")
        end_x = -x if abs(x) > 0.05 else x + max(w, 0.5) + 0.25
        self.start = (_spin(x), _spin(y))
        for box in self.start:  # 직선은 원본 자리에서 시작한다(원본은 옮기지 않음)
            box.setEnabled(False)
            box.setToolTip(S.ED_REPLICATE_LAYOUT_TIPS["line"])
        self.end = (_spin(end_x), _spin(y))
        self.center = (_spin(cx), _spin(cy))
        self.radius = _spin(math.hypot(x - cx, y - cy) or 1.0, 0.0)
        self.axis = _spin(cx)
        self.evolve = QCheckBox(S.ED_REPLICATE_EVOLVE)
        self.evolve.setToolTip(S.ED_REPLICATE_EVOLVE_TIP)
        self.one_count = QCheckBox(S.ED_REPLICATE_ONE_COUNT)
        self.one_count.setToolTip(S.ED_REPLICATE_ONE_COUNT_TIP)
        self.one_count.setEnabled(False)
        self.evolve.toggled.connect(self.one_count.setEnabled)

        form = QFormLayout()
        form.addRow(S.ED_REPLICATE_COUNT, self.count)
        form.addRow(S.ED_REPLICATE_LAYOUT_LABEL, self.layout_box)
        form.addRow(self.layout_tip)
        self._rows: dict[str, list[QWidget]] = {}
        for key, label, widgets in (("line", S.ED_REPLICATE_START, self.start), ("line", S.ED_REPLICATE_END, self.end),
                                    ("circle", S.ED_REPLICATE_CENTER, self.center)):
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            for wdg in widgets:
                hl.addWidget(wdg)
            form.addRow(label, holder)
            self._rows.setdefault(key, []).append(holder)
        form.addRow(S.ED_REPLICATE_RADIUS, self.radius)
        self._rows["circle"].append(self.radius)
        form.addRow(S.ED_REPLICATE_AXIS, self.axis)
        self._rows["mirror"] = [self.axis]
        form.addRow(self.evolve)
        form.addRow(self.one_count)
        self.form = form
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(S.ED_REPLICATE_OK)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(S.ED_REPLICATE_CANCEL)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.buttons = buttons
        lay = QVBoxLayout(self)
        lay.addWidget(hint)
        lay.addLayout(form)
        lay.addWidget(buttons)
        self.layout_box.currentIndexChanged.connect(self._sync)
        self._sync()

    def _defaults(self) -> tuple[float, float, float, float, float]:
        """(원본 x, y, 원본 폭, 몸통 x, y) — 에이전트 좌표, 미리보기 자세."""
        from evolab.app.editor.canvas import build_scene

        scene = build_scene(self.doc)
        sp = scene.spawn
        shown = scene.body(self.body)
        x = y = w = 0.0
        if shown is not None:
            x, y = T.to_local(sp, (shown.pose[0], shown.pose[1]))
            w = shown.bbox[2] - shown.bbox[0]
        root = root_body_id(self.doc.data)
        rb = scene.body(root) if root is not None else None
        cx, cy = T.to_local(sp, (rb.pose[0], rb.pose[1])) if rb is not None else (0.0, 0.0)
        return T.clean(x), T.clean(y), w, T.clean(cx), T.clean(cy)

    def _sync(self) -> None:
        kind = str(self.layout_box.currentData())
        self.layout_tip.setText(S.ED_REPLICATE_LAYOUT_TIPS[kind])
        for key, widgets in self._rows.items():
            for wdg in widgets:
                self.form.setRowVisible(wdg, key == kind)
        mirror = kind == "mirror"
        self.count.setEnabled(not mirror)
        if mirror:
            self.count.setValue(2)

    def layout_spec(self) -> dict[str, Any]:
        """놓는 방식 사전(tools.replicate의 layout)."""
        kind = str(self.layout_box.currentData())
        if kind == "line":
            return {"kind": "line", "start": [self.start[0].value(), self.start[1].value()],
                    "end": [self.end[0].value(), self.end[1].value()]}
        if kind == "circle":
            return {"kind": "circle", "center": [self.center[0].value(), self.center[1].value()],
                    "radius": self.radius.value()}
        return {"kind": "mirror", "x": self.axis.value()}

    def values(self) -> tuple[int, dict[str, Any], bool, bool]:
        """(개수, 놓는 방식, 있음/없음도 진화, 개수를 몸 수치 하나로)."""
        evolve = self.evolve.isChecked()
        return self.count.value(), self.layout_spec(), evolve, evolve and self.one_count.isChecked()

    def apply(self) -> list[Ref]:
        """고른 값으로 늘린다(tools.replicate, 되돌리기 한 단계)."""
        count, layout, evolve, one_count = self.values()
        return T.replicate(self.doc, self.body, count, layout, evolve, one_count)

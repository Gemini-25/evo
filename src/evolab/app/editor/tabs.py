"""에디터 아래쪽 탭 (명세: 화면과 조작 > 에디터 > 배치).

세계 설정, 사건·종료 조건, 점수 블록과 수식, 랜덤 범위, 생성기, 평가 설정, 뇌 설정, 매개변수, 검사 결과.
칸 위젯은 속성 패널과 같은 폼 엔진(forms.py)을 쓰고, 모든 편집은 문서(EnvDocument)를 거친다.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from evolab.app import strings as S
from evolab.app.editor.document import EnvDocument, JsonPath, Ref, get_path, set_path, where_to_ref
from evolab.app.editor.forms import (
    ISSUE_COLOR,
    FieldWidget,
    FormContext,
    ModelForm,
    TypeInfo,
    UnionKindField,
    field_name,
    field_tip,
    fields_of,
    issues_at,
    kinds_of,
    kind_name,
    make_field,
    model_for,
    new_item,
    path_to_where,
    spec_for,
    type_info,
    watch_focus,
)
from evolab.spec import model as m
from evolab.spec import symbols as sy
from evolab.spec.compiled import EnvFileError, Issue
from evolab.spec.io import parse_env


def _scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(widget)
    return area


def _issue_text(issues: list[Issue]) -> str:
    return "\n".join(f"⚠ {i.message}" for i in issues)


class IssueLine(QLabel):
    """구역 전체를 가리키는 검사 결과 한 줄(예: 점수 블록 없음, 사건 순환)."""

    def __init__(self, doc: EnvDocument, wheres: tuple[str, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.wheres = wheres
        self.setWordWrap(True)
        self.setStyleSheet(f"color: {ISSUE_COLOR};")
        doc.checked.connect(self.update_issues)
        self.update_issues(list(doc.issues))

    def update_issues(self, issues: list[Issue]) -> None:
        mine = [i for i in issues if i.where in self.wheres]
        self.setText(_issue_text(mine))
        self.setVisible(bool(mine))


# ---------------------------------------------------------------------------
# 표 (목록 구역 하나)
# ---------------------------------------------------------------------------


class RecordTable(QWidget):
    """목록 구역(사건, 종료 조건, 점수 블록, 랜덤 값, 매개변수, 생성기)을 표로. 칸마다 폼 칸 위젯을 둔다.

    section이 있으면(부품 구역) 줄을 고르면 그 부품을 고르고(doc.select), 더하기·지우기는 doc.add / doc.remove를
    쓴다. 없으면(종료 조건) 목록을 직접 고친다. 줄 순서 바꾸기는 위로·아래로 단추.
    """

    current_changed = Signal(int)

    def __init__(self, doc: EnvDocument, path: JsonPath, item_type: Any, columns: list[str], *,
                 section: str | None = None, new_item_fn: Callable[[str | None], dict[str, Any]] | None = None,
                 add_kinds: list[str] | None = None, stretch: tuple[str, ...] = (), what: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.path = tuple(path)
        self.info: TypeInfo = type_info(item_type)
        self.columns = columns
        self.section = section
        self.new_item_fn = new_item_fn
        self.what = what
        self.cells: list[list[FieldWidget | None]] = []
        self._sig: Any = None
        self._syncing = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        self.add_button = QToolButton()
        self.add_button.setText(S.EDP_ADD)
        self.add_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder))
        self.add_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        if add_kinds:
            menu = QMenu(self.add_button)
            for kind in add_kinds:
                act = menu.addAction(kind_name(kind))
                act.triggered.connect(functools.partial(self.add_row, kind))
            self.add_button.setMenu(menu)
            self.add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        else:
            self.add_button.clicked.connect(lambda: self.add_row(None))
        self.remove_button = QToolButton()
        self.remove_button.setText(S.EDP_REMOVE)
        self.remove_button.clicked.connect(self.remove_row)
        self.up_button = QToolButton()
        self.up_button.setText(S.EDP_UP)
        self.up_button.setArrowType(Qt.ArrowType.UpArrow)
        self.up_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.up_button.clicked.connect(lambda: self.move_row(-1))
        self.down_button = QToolButton()
        self.down_button.setText(S.EDP_DOWN)
        self.down_button.setArrowType(Qt.ArrowType.DownArrow)
        self.down_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.down_button.clicked.connect(lambda: self.move_row(1))
        for b in (self.add_button, self.remove_button, self.up_button, self.down_button):
            bar.addWidget(b)
        bar.addStretch(1)
        lay.addLayout(bar)

        self.table = QTableWidget(0, len(columns))
        self.table.setHorizontalHeaderLabels([self._header(c) for c in columns])
        for c, key in enumerate(columns):
            item = self.table.horizontalHeaderItem(c)
            if item is not None:
                item.setToolTip(self._header_tip(key))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(not stretch)
        for c, key in enumerate(columns):
            if key in stretch:
                header.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
            else:
                header.resizeSection(c, self._width(key))
        self.table.verticalHeader().setDefaultSectionSize(self.fontMetrics().height() + 14)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.currentCellChanged.connect(self._on_current)
        lay.addWidget(self.table, 1)
        self.issue_line = IssueLine(doc, (path_to_where(self.path),))
        lay.addWidget(self.issue_line)

        doc.changed.connect(self.refresh)
        doc.checked.connect(self.apply_issues)
        doc.selection_changed.connect(self._on_selection)
        self._rebuild()

    # ---- 머리글 ----
    def _models(self) -> tuple[Any, ...]:
        if self.info.kind == "union":
            return self.info.models
        return (self.info.model,) if self.info.model is not None else ()

    def _header(self, key: str) -> str:
        for model in self._models():
            if spec_for(model, key) is not None:
                return field_name(model, key)
        return field_name(None, key)

    def _header_tip(self, key: str) -> str:
        for model in self._models():
            spec = spec_for(model, key)
            if spec is not None:
                return field_tip(model, spec.name, key)
        return ""

    def _width(self, key: str) -> int:
        fm = self.fontMetrics()
        wide = {"kind": 12, "id": 10, "label": 12, "event": 11, "outcome": 7, "dist": 9, "values": 10}
        return fm.averageCharWidth() * wide.get(key, 8) + 24

    # ---- 자료 ----
    def items(self) -> list[Any]:
        v = self.doc.get(self.path, [])
        return v if isinstance(v, list) else []

    def _signature(self) -> Any:
        items = self.items()
        kinds = tuple(it.get("kind") if isinstance(it, dict) else None for it in items) if self.info.kind == "union" else ()
        return (len(items), kinds)

    def _rebuild(self) -> None:
        current = self.table.currentRow()
        t = self.table
        for row in self.cells:
            for w in row:
                if w is not None:
                    for child in [w, *w.findChildren(QWidget)]:
                        child.blockSignals(True)
        t.blockSignals(True)
        t.clearContents()
        items = self.items()
        t.setRowCount(len(items))
        self.cells = []
        for r, item in enumerate(items):
            model = model_for(self.info, item)
            ctx = FormContext(self.doc, section=self.section, root=(*self.path, r), compact=True)
            row: list[FieldWidget | None] = []
            for c, key in enumerate(self.columns):
                w: FieldWidget | None = None
                if key == "kind" and self.info.kind == "union":
                    w = UnionKindField(ctx, (*self.path, r, "kind"), self.info)
                elif model is not None:
                    spec = spec_for(model, key)
                    if spec is not None:
                        w = make_field(ctx, (*self.path, r, key), spec)
                if w is not None:
                    t.setCellWidget(r, c, w)
                    watch_focus(w, functools.partial(self._focus_cell, r, c))
                row.append(w)
            self.cells.append(row)
            t.setVerticalHeaderItem(r, QTableWidgetItem(str(r + 1)))
        t.blockSignals(False)
        self._sig = self._signature()
        if items:
            self._syncing = True
            t.setCurrentCell(max(0, min(current, len(items) - 1)) if current >= 0 else 0, 0)
            self._syncing = False
        self._update_buttons()
        self.apply_issues()
        self.current_changed.emit(t.currentRow())

    def refresh(self) -> None:
        if self._signature() != self._sig:
            self._rebuild()
            return
        for row in self.cells:
            for w in row:
                if w is not None:
                    w.refresh()
        self._update_buttons()

    def apply_issues(self, issues: list[Issue] | None = None) -> None:
        issues = list(self.doc.issues if issues is None else issues)
        mine = issues_at(issues, self.path)
        for r, row in enumerate(self.cells):
            for w in row:
                if w is not None:
                    for leaf in w.leaves():
                        leaf.set_issues(mine)
            header = self.table.verticalHeaderItem(r)
            if header is not None:
                exact = issues_at(mine, (*self.path, r), exact=True)
                header.setForeground(QBrush(QColor(ISSUE_COLOR)) if exact else QBrush())
                header.setToolTip(_issue_text(exact))

    def cell(self, row: int, key: str) -> FieldWidget | None:
        try:
            return self.cells[row][self.columns.index(key)]
        except (IndexError, ValueError):
            return None

    # ---- 현재 줄 ----
    def current_row(self) -> int:
        return self.table.currentRow()

    def set_current_row(self, row: int) -> None:
        if 0 <= row < self.table.rowCount():
            self.table.setCurrentCell(row, max(0, self.table.currentColumn()))

    def _focus_cell(self, row: int, col: int) -> None:
        if self.table.currentRow() != row:
            self.table.setCurrentCell(row, col)

    def _ref(self, row: int) -> Ref | None:
        items = self.items()
        if self.section is None or not (0 <= row < len(items)) or not isinstance(items[row], dict):
            return None
        cid = items[row].get("id")
        return (self.section, str(cid)) if isinstance(cid, str) else None

    def _on_current(self, row: int, _col: int, prev: int, _pcol: int) -> None:
        self._update_buttons()
        if row == prev:
            return
        self.current_changed.emit(row)
        ref = self._ref(row)
        if ref is not None and not self._syncing and self.doc.selection != [ref]:
            self._syncing = True
            self.doc.select([ref])
            self._syncing = False

    def _on_selection(self, refs: list[Ref]) -> None:
        if self._syncing or self.section is None or not refs or refs[0][0] != self.section:
            return
        row = self.doc.find(refs[0])
        if row is not None and row != self.table.currentRow():
            self._syncing = True
            self.set_current_row(row)
            self._syncing = False

    def _update_buttons(self) -> None:
        n = self.table.rowCount()
        r = self.table.currentRow()
        self.remove_button.setEnabled(0 <= r < n)
        self.up_button.setEnabled(0 < r < n)
        self.down_button.setEnabled(0 <= r < n - 1)

    # ---- 편집 ----
    def add_row(self, kind: str | None = None) -> None:
        if self.new_item_fn is not None:
            item = self.new_item_fn(kind)
        else:
            model = kinds_of(self.info.models)[kind] if kind and self.info.kind == "union" else self.info.model
            assert model is not None
            item = new_item(model, self.doc.data, kind=kind)
        label = S.EDP_UNDO_ADD_ITEM.format(name=self.what)
        if self.section is not None:
            self.doc.add(self.section, item, label=label)
            row = len(self.items()) - 1
        else:
            path = self.path

            def mutate(d: dict[str, Any]) -> None:
                try:
                    lst = get_path(d, path)
                except (KeyError, IndexError, TypeError):
                    lst = None
                if not isinstance(lst, list):
                    set_path(d, path, [])
                    lst = get_path(d, path)
                lst.append(item)

            self.doc.edit(label, mutate)
            row = len(self.items()) - 1
        self.set_current_row(row)

    def remove_row(self) -> None:
        r = self.table.currentRow()
        items = self.items()
        if not (0 <= r < len(items)):
            return
        ref = self._ref(r)
        label = S.EDP_UNDO_REMOVE_ITEM.format(name=self.what)
        if ref is not None:
            self.doc.remove([ref], label=label)
        else:
            path = self.path
            self.doc.edit(label, lambda d: get_path(d, path).pop(r))
        self.set_current_row(min(r, len(self.items()) - 1))

    def move_row(self, delta: int) -> None:
        r = self.table.currentRow()
        n = len(self.items())
        j = r + delta
        if not (0 <= r < n and 0 <= j < n):
            return
        path = self.path

        def mutate(d: dict[str, Any]) -> None:
            lst = get_path(d, path)
            lst[r], lst[j] = lst[j], lst[r]

        self.doc.edit(S.EDP_UNDO_MOVE, mutate)
        self.set_current_row(j)


class _MasterDetail(QWidget):
    """표 + 고른 줄의 종류별 폼(점수 블록, 생성기)."""

    def __init__(self, doc: EnvDocument, table: RecordTable, detail_title: str, empty_text: str,
                 exclude: tuple[str, ...], item_type: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.table = table
        self.item_type = item_type
        self.exclude = exclude
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(table)
        box = QGroupBox(detail_title)
        box_lay = QVBoxLayout(box)
        box_lay.setContentsMargins(6, 4, 6, 4)
        self.empty_text = empty_text
        self.empty = QLabel(empty_text)
        self.empty.setWordWrap(True)
        box_lay.addWidget(self.empty)
        self.detail = ModelForm(doc)
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(0, 0, 4, 0)
        inner_lay.addWidget(self.detail)
        inner_lay.addStretch(1)
        box_lay.addWidget(_scroll(inner), 1)
        split.addWidget(box)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        self.splitter = split
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)
        table.current_changed.connect(self._on_row)
        self._on_row(table.current_row())

    def _on_row(self, row: int) -> None:
        if 0 <= row < len(self.table.items()):
            self.detail.set_target((*self.table.path, row), self.item_type, exclude=self.exclude,
                                   section=self.table.section)
            self.empty.setVisible(self.detail.rows is None or not self.detail.rows.fields)
            if self.detail.rows is not None and not self.detail.rows.fields:
                self.empty.setText(S.EDP_DETAIL_NO_FIELDS)
        else:
            self.detail.clear()
            self.empty.setText(self.empty_text)
            self.empty.show()


# ---------------------------------------------------------------------------
# 탭
# ---------------------------------------------------------------------------


class WorldTab(QWidget):
    """세계 설정: 환경 이름·설명, 세계(중력, 바람, 시간, 경계, 접촉 디바운스), 시작 위치."""

    def __init__(self, doc: EnvDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        body = QWidget()
        row = QHBoxLayout(body)
        row.setContentsMargins(4, 4, 4, 4)
        self.env_form = ModelForm(doc, (), m.EnvFile, only=("name", "description"))
        self.world_form = ModelForm(doc, ("world",), m.World)
        self.spawn_form = ModelForm(doc, ("agent", "spawn"), m.Spawn)
        for title, form, stretch in ((S.EDP_GROUP_ENV, self.env_form, 3), (S.EDP_GROUP_WORLD, self.world_form, 4),
                                     (S.EDP_GROUP_SPAWN, self.spawn_form, 3)):
            box = QGroupBox(title)
            bl = QVBoxLayout(box)
            bl.setContentsMargins(6, 4, 6, 4)
            bl.addWidget(form)
            bl.addStretch(1)
            row.addWidget(box, stretch)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_scroll(body))


def _event_item(doc: EnvDocument, _kind: str | None) -> dict[str, Any]:
    return {"id": "event", "label": "", "when": "time > 10", "hold": 0.0}


def _end_item(doc: EnvDocument, _kind: str | None) -> dict[str, Any]:
    events = [e.get("id") for e in doc.items("events") if isinstance(e.get("id"), str)]
    used = {t.get("event") for t in doc.items("termination")}
    event = next((e for e in events if e not in used), events[0] if events else "timeout")
    return {"event": event, "outcome": "failure", "label": ""}


class EventsTab(QWidget):
    """사건(이름 붙은 조건)과 종료 조건(사건 + 결과, 위가 먼저)."""

    def __init__(self, doc: EnvDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.events = RecordTable(doc, ("events",), m.Event, ["id", "label", "when", "hold"], section="events",
                                  new_item_fn=functools.partial(_event_item, doc), stretch=("when",),
                                  what=S.EDP_SECTION_NAMES["events"])
        self.termination = RecordTable(doc, ("termination",), m.End, ["event", "outcome", "label"],
                                       new_item_fn=functools.partial(_end_item, doc), stretch=("label",),
                                       what=S.EDP_SECTION_NAMES["termination"])
        split = QSplitter(Qt.Orientation.Horizontal)
        for title, table, tip in ((S.EDP_GROUP_EVENTS, self.events, S.EDP_EVENTS_TIP),
                                  (S.EDP_GROUP_TERMINATION, self.termination, S.EDP_TERMINATION_TIP)):
            box = QGroupBox(title)
            box.setToolTip(tip)
            bl = QVBoxLayout(box)
            bl.setContentsMargins(6, 4, 6, 4)
            bl.addWidget(table)
            split.addWidget(box)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)


BLOCK_KINDS = list(kinds_of(type_info(m.Block).models))
GENERATOR_KINDS = list(kinds_of(type_info(m.Generator).models))


def _block_item(doc: EnvDocument, kind: str | None) -> dict[str, Any]:
    kind = kind or "event"
    model = kinds_of(type_info(m.Block).models)[kind]
    return new_item(model, doc.data, kind=kind, id=f"s_{kind}", label=kind_name(kind), weight=1.0)


def _generator_item(doc: EnvDocument, kind: str | None) -> dict[str, Any]:
    kind = kind or "heightmap"
    model = kinds_of(type_info(m.Generator).models)[kind]
    return new_item(model, doc.data, kind=kind, id="terrain" if kind == "heightmap" else kind, label=kind_name(kind))


class ScoringTab(QWidget):
    """점수 블록(표 + 고른 블록의 종류별 칸)과 선택적 점수 수식."""

    def __init__(self, doc: EnvDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.blocks = RecordTable(doc, ("scoring", "blocks"), m.Block, ["kind", "id", "label", "weight"],
                                  section="scoring.blocks", new_item_fn=functools.partial(_block_item, doc),
                                  add_kinds=BLOCK_KINDS, stretch=("label",), what=S.EDP_SECTION_NAMES["scoring.blocks"])
        self.view = _MasterDetail(doc, self.blocks, S.EDP_GROUP_BLOCK, S.EDP_DETAIL_PICK,
                                  ("kind", "id", "label", "weight"), m.Block)
        self.formula_form = ModelForm(doc, ("scoring",), m.Scoring, only=("formula",))
        self.issue_line = IssueLine(doc, ("scoring",))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view, 1)
        bottom = QGroupBox(S.EDP_FORMULA)
        bottom.setToolTip(S.EDP_FORMULA_TIP)
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(6, 4, 6, 4)
        note = QLabel(S.EDP_FORMULA_TIP)
        note.setWordWrap(True)
        bl.addWidget(self.formula_form)
        bl.addWidget(note)
        bl.addWidget(self.issue_line)
        lay.addWidget(bottom)

    @property
    def detail(self) -> ModelForm:
        return self.view.detail


class RandomTab(QWidget):
    """랜덤 범위: 시험마다 시험 시드로 뽑는 값(r.<id>)."""

    def __init__(self, doc: EnvDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.table = RecordTable(
            doc, ("randomize",), m.RandVar, ["id", "label", "dist", "low", "high", "mean", "std", "values"],
            section="randomize", new_item_fn=lambda _k: {"id": "rand", "label": "", "dist": "uniform", "low": 0.0,
                                                         "high": 1.0},
            stretch=("label",), what=S.EDP_SECTION_NAMES["randomize"])
        note = QLabel(S.EDP_RANDOM_NOTE)
        note.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(note)
        lay.addWidget(self.table, 1)


class GeneratorsTab(QWidget):
    """생성기(지형 자동 생성, 트랙): 목록 + 고른 생성기의 설정."""

    def __init__(self, doc: EnvDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.table = RecordTable(doc, ("generators",), m.Generator, ["kind", "id", "label"], section="generators",
                                 new_item_fn=functools.partial(_generator_item, doc), add_kinds=GENERATOR_KINDS,
                                 stretch=("label",), what=S.EDP_SECTION_NAMES["generators"])
        self.view = _MasterDetail(doc, self.table, S.EDP_GROUP_GENERATOR, S.EDP_DETAIL_PICK, ("kind", "id", "label"),
                                  m.Generator)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)

    @property
    def detail(self) -> ModelForm:
        return self.view.detail


def track_set_size(data: dict[str, Any]) -> int | None:
    """학습 트랙 세트 크기(EnvSpec.track_set_size와 같은 규칙, 사전에서 바로)."""
    for g in data.get("generators", []) if isinstance(data.get("generators"), list) else []:
        if isinstance(g, dict) and g.get("kind") == "track" and g.get("source", "generated") == "generated":
            try:
                return int(g.get("set_size", 5))
            except (TypeError, ValueError):
                return None
    return None


class EvaluationTab(QWidget):
    """평가 설정: 시험 횟수, 집계 방식, 목표 점수. 학습 트랙 세트가 있으면 시험 횟수는 트랙 수."""

    def __init__(self, doc: EnvDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.form = ModelForm(doc, ("evaluation",), m.Evaluation)
        self.note = QLabel()
        self.note.setWordWrap(True)
        box = QGroupBox(S.EDP_TAB_EVAL)
        bl = QVBoxLayout(box)
        bl.setContentsMargins(6, 4, 6, 4)
        bl.addWidget(self.form)
        bl.addWidget(self.note)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(box, 1)
        lay.addStretch(1)
        doc.changed.connect(self._update)
        self._update()

    def _update(self) -> None:
        n = track_set_size(self.doc.data)
        trials = self.form.field("trials")
        if trials is not None and trials.editor is not None:
            trials.editor.setEnabled(n is None)
        self.note.setText(S.EDP_EVAL_TRACK_NOTE.format(n=n) if n is not None else "")
        self.note.setVisible(n is not None)


class ChannelList(QWidget):
    """뇌 입력 또는 출력 순서: 채널 목록(체크 = 씀) + 위로·아래로, 또는 '전부(선언 순서)'."""

    def __init__(self, doc: EnvDocument, key: str, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.key = key
        self.path: JsonPath = ("brain", key)
        box = QGroupBox(title)
        box.setToolTip(field_tip(m.Brain, key))
        bl = QVBoxLayout(box)
        bl.setContentsMargins(6, 4, 6, 4)
        self.all_box = QCheckBox(S.EDP_BRAIN_ALL)
        self.all_box.clicked.connect(self._on_all)
        bl.addWidget(self.all_box)
        self.list = QListWidget()
        self.list.itemChanged.connect(self._on_item)
        self.list.currentRowChanged.connect(lambda _r: self._update_buttons())
        bl.addWidget(self.list, 1)
        row = QHBoxLayout()
        self.up_button = QToolButton()
        self.up_button.setArrowType(Qt.ArrowType.UpArrow)
        self.up_button.setText(S.EDP_UP)
        self.up_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.up_button.clicked.connect(lambda: self.move(-1))
        self.down_button = QToolButton()
        self.down_button.setArrowType(Qt.ArrowType.DownArrow)
        self.down_button.setText(S.EDP_DOWN)
        self.down_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.down_button.clicked.connect(lambda: self.move(1))
        row.addWidget(self.up_button)
        row.addWidget(self.down_button)
        row.addStretch(1)
        bl.addLayout(row)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setStyleSheet(f"color: {ISSUE_COLOR};")
        bl.addWidget(self.message)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(box)
        self._entries: list[tuple[str, bool, bool, str]] | None = None
        self._issues: list[Issue] = []
        doc.changed.connect(self.refresh)
        doc.checked.connect(self.apply_issues)
        self.refresh()

    def channels(self) -> list[str] | None:
        """지금 파일의 채널 이름(선언 순서). 파일 구조가 틀려 읽을 수 없으면 None."""
        found = self._channels()
        return None if found is None else [name for name, _label in found]

    def _channels(self) -> list[tuple[str, str]] | None:
        """(채널 이름, 그 센서·구동기의 표시 이름)."""
        try:
            env = parse_env(self.doc.data)
        except EnvFileError:
            return None
        owners = env.sensors if self.key == "inputs" else env.actuators
        labels = {o.id: o.label for o in owners}
        chans = sy.sensor_channels(env) if self.key == "inputs" else sy.actuator_channels(env)
        return [(c.name, labels.get(c.owner, "")) for c in chans]

    def value(self) -> Any:
        return self.doc.get(self.path, "all")

    def checked_names(self) -> list[str]:
        out: list[str] = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return out

    def refresh(self) -> None:
        found = self._channels()
        chans = None if found is None else [name for name, _label in found]
        labels = dict(found or [])
        value = self.value()
        if chans is None:
            self.list.setEnabled(False)
            self.all_box.setEnabled(False)
            self.message.setText(S.EDP_BRAIN_NO_CHANNELS)
            self._update_buttons()
            return
        self.all_box.setEnabled(True)
        self.message.setText("")
        is_all = value == "all"
        chosen = list(chans) if is_all or not isinstance(value, list) else [str(v) for v in value]
        order = chosen + [c for c in chans if c not in chosen]
        entries = [(name, name in chosen, name in chans, labels.get(name, "")) for name in order]
        if entries != self._entries:
            current = self.list.currentItem()
            current_name = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
            self.list.blockSignals(True)
            self.list.clear()
            for name, on, known, label in entries:
                text = S.EDP_BRAIN_CHANNEL.format(name=name, label=label) if label else name
                item = QListWidgetItem(text if known else S.EDP_BRAIN_UNKNOWN.format(name=name))
                item.setData(Qt.ItemDataRole.UserRole, name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
                if not known:
                    item.setForeground(QBrush(QColor(ISSUE_COLOR)))
                self.list.addItem(item)
                if name == current_name:
                    self.list.setCurrentItem(item)
            self.list.blockSignals(False)
            self._entries = entries
        self.all_box.setChecked(is_all)
        self.list.setEnabled(not is_all)
        self._update_buttons()

    def apply_issues(self, issues: list[Issue]) -> None:
        mine = issues_at(issues, self.path)
        self.message.setText(_issue_text(mine) if mine else ("" if self.channels() is not None
                                                             else S.EDP_BRAIN_NO_CHANNELS))

    def _write(self, value: Any) -> None:
        self.doc.set_value(self.path, value, label=S.EDP_UNDO_FIELD.format(name=field_name(m.Brain, self.key)),
                           merge=False)

    def _on_all(self, checked: bool) -> None:
        chans = self.channels() or []
        self._write("all" if checked else list(chans))

    def _on_item(self, _item: QListWidgetItem) -> None:
        if self.value() == "all":
            return
        self._write(self.checked_names())

    def move(self, delta: int) -> None:
        """고른 채널을 위·아래로(켠 채널끼리)."""
        names = self.checked_names()
        cur = self.list.currentItem()
        if cur is None or self.value() == "all":
            return
        name = str(cur.data(Qt.ItemDataRole.UserRole))
        if name not in names:
            return
        i = names.index(name)
        j = i + delta
        if not 0 <= j < len(names):
            return
        names[i], names[j] = names[j], names[i]
        self._write(names)
        self.list.setCurrentRow(j)

    def _update_buttons(self) -> None:
        cur = self.list.currentItem()
        ok = (self.list.isEnabled() and cur is not None and cur.checkState() == Qt.CheckState.Checked)
        names = self.checked_names() if ok else []
        name = str(cur.data(Qt.ItemDataRole.UserRole)) if cur is not None else ""
        i = names.index(name) if name in names else -1
        self.up_button.setEnabled(ok and i > 0)
        self.down_button.setEnabled(ok and 0 <= i < len(names) - 1)


class BrainTab(QWidget):
    """뇌 설정: 입력 순서, 출력 순서, 은닉층, 활성화 함수, 파라미터 수."""

    def __init__(self, doc: EnvDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.inputs = ChannelList(doc, "inputs", S.EDP_BRAIN_INPUTS)
        self.outputs = ChannelList(doc, "outputs", S.EDP_BRAIN_OUTPUTS)
        self.form = ModelForm(doc, ("brain",), m.Brain, only=("hidden", "activation"))
        self.count_label = QLabel()
        self.count_label.setWordWrap(True)
        self.param_count: int | None = None
        """뇌 파라미터 수(가중치 + 편향). 검사 오류가 있으면 None."""
        self.genome_dim: int | None = None
        """유전자 차원(몸 유전자 + 뇌 파라미터)."""
        net = QGroupBox(S.EDP_BRAIN_NET)
        nl = QVBoxLayout(net)
        nl.setContentsMargins(6, 4, 6, 4)
        nl.addWidget(self.form)
        nl.addWidget(self.count_label)
        nl.addStretch(1)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.inputs, 3)
        lay.addWidget(self.outputs, 2)
        lay.addWidget(net, 3)
        doc.checked.connect(self._on_checked)
        self.update_count()

    def _on_checked(self, _issues: list[Issue]) -> None:
        self.update_count()

    def update_count(self) -> None:
        from evolab.policy.genome import layout_for

        spec = self.doc.spec() if not any(i.severity == "error" for i in self.doc.issues) else None
        layout = None
        if spec is not None:
            try:
                layout = layout_for(spec)
            except ValueError:
                layout = None
        if layout is None:
            self.param_count = self.genome_dim = None
            self.count_label.setText(S.EDP_BRAIN_PARAMS_NONE)
            return
        self.param_count = layout.brain.n_params
        self.genome_dim = layout.dim
        shape = " \u2192 ".join(str(n) for n in (layout.brain.n_in, *layout.brain.hidden, layout.brain.n_out))
        self.count_label.setText(S.EDP_BRAIN_PARAMS.format(shape=shape, brain=layout.brain.n_params,
                                                           body=layout.n_body, dim=layout.dim))


class ParamsTab(QWidget):
    """매개변수: 수식에서 p.<id>로 쓰는 값. evolve를 켜면 몸 유전자(최솟값~최댓값)."""

    def __init__(self, doc: EnvDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.table = RecordTable(
            doc, ("params",), m.Param, ["id", "label", "value", "evolve", "min", "max"], section="params",
            new_item_fn=lambda _k: {"id": "param", "label": "", "value": 1.0, "evolve": False, "min": 0.5, "max": 2.0},
            stretch=("label",), what=S.EDP_SECTION_NAMES["params"])
        note = QLabel(S.EDP_PARAMS_NOTE)
        note.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(note)
        lay.addWidget(self.table, 1)


class IssuesTab(QWidget):
    """검사 결과 전부. 두 번 누르면 그 부품을 고르고 issue_activated(ref, where)를 낸다."""

    issue_activated = Signal(object, str)
    """(Ref 또는 None, where). 캔버스가 그 부품으로 화면을 옮길 때 쓴다."""

    def __init__(self, doc: EnvDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(list(S.EDP_ISSUES_HEADERS))
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setWordWrap(True)
        self.tree.header().setStretchLastSection(True)
        self.tree.header().resizeSection(0, self.fontMetrics().averageCharWidth() * 8 + 30)
        self.tree.header().resizeSection(1, self.fontMetrics().averageCharWidth() * 28)
        self.tree.itemDoubleClicked.connect(self._on_double)
        self.empty = QLabel(S.EDP_ISSUES_NONE)
        hint = QLabel(S.EDP_ISSUES_HINT)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(hint)
        lay.addWidget(self.tree, 1)
        lay.addWidget(self.empty)
        doc.checked.connect(self.fill)
        self.fill(list(doc.issues))

    def where_text(self, where: str) -> str:
        if not where:
            return S.EDP_WHERE_FILE
        ref = where_to_ref(self.doc.data, where)
        if ref is not None and self.doc.find(ref) is not None:
            return S.EDP_WHERE_REF.format(section=S.EDP_SECTION_NAMES.get(ref[0], ref[0]), id=ref[1], where=where)
        top = where.split(".")[0].split("[")[0]
        name = S.EDP_SECTION_NAMES.get(top) or S.EDP_FIELD_NAMES.get(top)
        return S.EDP_WHERE_PATH.format(section=name, where=where) if name else where

    def fill(self, issues: list[Issue]) -> None:
        self.tree.clear()
        style = self.style()
        order = sorted(issues, key=lambda i: (i.severity != "error",))
        for issue in order:
            icon = style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxCritical if issue.severity == "error"
                                      else QStyle.StandardPixmap.SP_MessageBoxWarning)
            item = QTreeWidgetItem([S.EDP_ISSUES_SEVERITY.get(issue.severity, issue.severity),
                                    self.where_text(issue.where), issue.message])
            item.setIcon(0, icon)
            item.setData(0, Qt.ItemDataRole.UserRole, issue.where)
            item.setToolTip(2, issue.message)
            item.setToolTip(1, issue.where)
            self.tree.addTopLevelItem(item)
        self.empty.setVisible(not issues)
        self.tree.setVisible(bool(issues))

    def activate(self, where: str) -> None:
        ref = where_to_ref(self.doc.data, where)
        if ref is not None and self.doc.find(ref) is None:
            ref = None
        if ref is not None:
            self.doc.select([ref])
        self.issue_activated.emit(ref, where)

    def _on_double(self, item: QTreeWidgetItem, _col: int) -> None:
        self.activate(str(item.data(0, Qt.ItemDataRole.UserRole) or ""))


class BottomTabs(QTabWidget):
    """에디터 아래쪽 탭 묶음."""

    issue_activated = Signal(object, str)
    """검사 결과를 두 번 눌렀다: (Ref 또는 None, where). 부품이면 이미 doc.select 했다."""

    def __init__(self, doc: EnvDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.world_tab = WorldTab(doc)
        self.events_tab = EventsTab(doc)
        self.scoring_tab = ScoringTab(doc)
        self.random_tab = RandomTab(doc)
        self.generators_tab = GeneratorsTab(doc)
        self.eval_tab = EvaluationTab(doc)
        self.brain_tab = BrainTab(doc)
        self.params_tab = ParamsTab(doc)
        self.issues_tab = IssuesTab(doc)
        for w, title in ((self.world_tab, S.EDP_TAB_WORLD), (self.events_tab, S.EDP_TAB_EVENTS),
                         (self.scoring_tab, S.EDP_TAB_SCORING), (self.random_tab, S.EDP_TAB_RANDOM),
                         (self.generators_tab, S.EDP_TAB_GENERATORS), (self.eval_tab, S.EDP_TAB_EVAL),
                         (self.brain_tab, S.EDP_TAB_BRAIN), (self.params_tab, S.EDP_TAB_PARAMS),
                         (self.issues_tab, S.EDP_TAB_ISSUES)):
            self.addTab(w, title)
        self.issues_tab.issue_activated.connect(self._on_issue)
        doc.checked.connect(self._update_issue_title)
        self._update_issue_title(list(doc.issues))

    def _update_issue_title(self, issues: list[Issue]) -> None:
        i = self.indexOf(self.issues_tab)
        errors = sum(1 for x in issues if x.severity == "error")
        self.setTabText(i, S.EDP_TAB_ISSUES_COUNT.format(n=len(issues)) if issues else S.EDP_TAB_ISSUES)
        style = self.style()
        if errors:
            self.setTabIcon(i, style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxCritical))
        elif issues:
            self.setTabIcon(i, style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning))
        else:
            self.setTabIcon(i, style.standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))

    def tab_for(self, where: str) -> QWidget | None:
        """이 경로를 고치는 탭(부품 구역이면 None: 캔버스와 속성 패널에서 고친다)."""
        top = where.split(".")[0].split("[")[0]
        return {
            "world": self.world_tab, "agent": self.world_tab, "name": self.world_tab, "description": self.world_tab,
            "events": self.events_tab, "termination": self.events_tab, "scoring": self.scoring_tab,
            "randomize": self.random_tab, "generators": self.generators_tab, "evaluation": self.eval_tab,
            "brain": self.brain_tab, "params": self.params_tab,
        }.get(top)

    def show_where(self, where: str) -> None:
        """그 경로를 고치는 탭으로 옮긴다(있으면)."""
        tab = self.tab_for(where)
        if tab is not None:
            self.setCurrentWidget(tab)

    def _on_issue(self, ref: object, where: str) -> None:
        self.issue_activated.emit(ref, where)
        if self.tab_for(where) is not None:
            self.show_where(where)


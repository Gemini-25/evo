"""오른쪽 속성 패널 (명세: 화면과 조작 > 에디터 > 배치).

고른 부품(doc.selection의 첫 번째)의 이름(id), 표시 이름, 지우기 단추, 종류별 폼(forms.ModelForm), 그 부품의
검사 결과를 보인다. 아무것도 안 골랐으면 안내 문구.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from evolab.app import strings as S
from evolab.app.editor.document import EnvDocument, JsonPath, Ref
from evolab.app.editor.forms import (
    SECTION_TYPES,
    FieldWidget,
    FormContext,
    IdField,
    ModelForm,
    TextField,
    issues_at,
    kind_name,
    model_for,
    section_info,
    spec_for,
)
from evolab.app.editor.tabs import SEVERITY_ICONS
from evolab.spec.compiled import Issue


class PropertyPanel(QWidget):
    """고른 부품의 속성."""

    def __init__(self, doc: EnvDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self._ref: Ref | None = None
        self._path: JsonPath | None = None
        self._ids_before: set[str] = set()
        self.setMinimumWidth(300)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        self.title = QLabel()
        font = QFont(self.title.font())
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.1)
        self.title.setFont(font)
        self.title.setWordWrap(True)
        outer.addWidget(self.title)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.hide()
        outer.addWidget(self.note)
        self.hint = QLabel(S.EDP_PANEL_HINT)
        self.hint.setWordWrap(True)
        self.hint.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        outer.addWidget(self.hint)

        # 이름(id), 표시 이름, 지우기
        self.header = QWidget()
        self.header_layout = QFormLayout(self.header)
        self.header_layout.setContentsMargins(0, 0, 0, 0)
        self.header_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.id_field: IdField | None = None
        self.label_field: TextField | None = None
        self.delete_button = QPushButton(S.EDP_PANEL_DELETE)
        self.delete_button.setToolTip(S.EDP_PANEL_DELETE_TIP)
        self.delete_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
        self.delete_button.clicked.connect(self._on_delete)
        head_row = QHBoxLayout()
        head_row.setContentsMargins(0, 0, 0, 0)
        head_row.addWidget(self.header, 1)
        head_row.addWidget(self.delete_button, 0, Qt.AlignmentFlag.AlignTop)
        self.head_box = QWidget()
        self.head_box.setLayout(head_row)
        outer.addWidget(self.head_box)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        self.separator = sep
        outer.addWidget(sep)

        self.form = ModelForm(doc, auto=False)
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(0, 0, 4, 0)
        inner_lay.addWidget(self.form)
        inner_lay.addStretch(1)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidget(inner)
        outer.addWidget(self.scroll, 1)

        self.issue_box = QGroupBox(S.EDP_PANEL_ISSUES)
        box_lay = QVBoxLayout(self.issue_box)
        box_lay.setContentsMargins(4, 4, 4, 4)
        self.issue_list = QListWidget()
        self.issue_list.setWordWrap(True)
        self.issue_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.issue_list.setMaximumHeight(110)
        box_lay.addWidget(self.issue_list)
        outer.addWidget(self.issue_box)
        self.issue_box.hide()

        doc.selection_changed.connect(self._on_selection)
        doc.changed.connect(self._on_changed)
        doc.checked.connect(self._on_checked)
        self._build_header()
        self._update_title()
        self._on_selection(list(doc.selection))

    # ------------------------------------------------------------------
    @property
    def ref(self) -> Ref | None:
        """지금 보이는 부품."""
        return self._ref

    def _model(self) -> Any:
        assert self._ref is not None
        return SECTION_TYPES.get(self._ref[0])

    def _on_selection(self, refs: list[Ref]) -> None:
        refs = [r for r in refs if r[0] in SECTION_TYPES]
        if len(refs) > 1:
            self.note.setText(S.EDP_PANEL_MULTI.format(n=len(refs)))
            self.note.show()
        else:
            self.note.hide()
        self._set_ref(refs[0] if refs else None)

    def _set_ref(self, ref: Ref | None) -> None:
        path = self.doc.item_path(ref) if ref is not None else None
        if ref is not None and path is None:
            ref = None
        if ref == self._ref and path == self._path and (ref is None or self.form.path == path):
            self._update_title()
            return
        self._ref = ref
        self._path = path
        self._ids_before = self.doc.all_ids()
        self._build_header()
        if ref is None:
            self.form.clear()
        else:
            self.form.set_target(path, self._model(), exclude=("id", "label"), section=ref[0])
        self._update_title()
        self._on_checked(list(self.doc.issues))

    def _build_header(self) -> None:
        for w in (self.id_field, self.label_field):
            if w is not None:
                for child in [w, *w.findChildren(QWidget)]:
                    child.blockSignals(True)
                w.hide()
                w.deleteLater()
        while self.header_layout.rowCount():
            self.header_layout.removeRow(0)
        self.id_field = None
        self.label_field = None
        empty = self._ref is None
        self.hint.setVisible(empty)
        self.head_box.setVisible(not empty)
        self.separator.setVisible(not empty)
        self.scroll.setVisible(not empty)
        if empty or self._path is None:
            return
        section = self._ref[0]  # type: ignore[index]
        model = model_for(section_info(section), self.doc.get(self._path))
        if model is None:
            info = section_info(section)
            model = info.models[0] if info.models else info.model
        ctx = FormContext(self.doc, section=section, root=self._path)
        id_spec = spec_for(model, "id")
        label_spec = spec_for(model, "label")
        if id_spec is not None:
            self.id_field = IdField(ctx, (*self._path, "id"), id_spec)
            self._add_header_row(S.EDP_PANEL_ID, self.id_field)
        if label_spec is not None:
            self.label_field = TextField(ctx, (*self._path, "label"), label_spec)
            self._add_header_row(S.EDP_PANEL_LABEL, self.label_field)

    def _add_header_row(self, text: str, w: FieldWidget) -> None:
        label = QLabel(text)
        w.label = label
        label.setToolTip(w.tooltip())
        self.header_layout.addRow(label, w)

    def _update_title(self) -> None:
        if self._ref is None:
            self.title.setText(S.EDP_PANEL_TITLE_NONE)
            return
        section = self._ref[0]
        name = S.EDP_SECTION_NAMES.get(section, section)
        item = self.doc.get(self._path) if self._path is not None else None
        if isinstance(item, dict) and isinstance(item.get("kind"), str) and section_info(section).kind == "union":
            name = S.EDP_PANEL_TITLE_KIND.format(section=name, kind=kind_name(item["kind"]))
        self.title.setText(name)

    # ------------------------------------------------------------------
    def _on_changed(self) -> None:
        if self._ref is None:
            return
        path = self.doc.item_path(self._ref)
        if path is None and self._path is not None:
            # 이름이 바뀌었으면 같은 자리에 처음 보는 id가 있다(지운 거라면 원래 있던 id가 당겨져 온다).
            item = self.doc.get(self._path)
            if (isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"] not in self._ids_before
                    and self._ref[1] not in self.doc.all_ids()):
                self._ref = (self._ref[0], item["id"])
                path = self._path
        self._ids_before = self.doc.all_ids()
        if path is None:
            self._set_ref(None)
            return
        if path != self._path:
            self._ref, self._path = self._ref, None
            self._set_ref(self._ref)
            return
        for w in (self.id_field, self.label_field):
            if w is not None:
                w.refresh()
        self.form.refresh()
        self._update_title()

    def _on_checked(self, issues: list[Issue]) -> None:
        self.form.apply_issues(issues)
        mine: list[Issue] = []
        if self._ref is not None and self._path is not None:
            for w in (self.id_field, self.label_field):
                if w is not None:
                    w.set_issues(issues)
            mine = issues_at(issues, self._path)
        self.issue_list.clear()
        style = self.style()
        for issue in mine:
            icon = style.standardIcon(SEVERITY_ICONS.get(issue.severity, QStyle.StandardPixmap.SP_MessageBoxWarning))
            item = QListWidgetItem(icon, issue.message)
            item.setToolTip(issue.where)
            self.issue_list.addItem(item)
        self.issue_box.setVisible(bool(mine))

    def _on_delete(self) -> None:
        if self._ref is not None:
            self.doc.remove([self._ref])

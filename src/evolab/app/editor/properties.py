"""오른쪽 속성 패널 (명세: 화면과 조작 > 에디터 > 배치).

고른 부품(doc.selection의 첫 번째)의 이름(id), 표시 이름, 지우기 단추, 종류별 폼(forms.ModelForm), 그 부품의
검사 결과를 보인다. 아무것도 안 골랐으면 안내 문구.

6.2.0: 검사 결과 아래에 문제 목록 탭과 같은 「고치기」 단추(editor/tabs.fix_widget). 화면 다듬기 U38 ②: 부품을 고르면 패널의 최소 폭을
그 폼이 바라는 폭(+ 스크롤 막대·여백, 상한 PANEL_MAX_MIN)으로 올려 나눔 막대가 폼을 자르지 못하게 한다. U40: 빈 상태의 제목·안내가 맨 위,
검사 결과 목록은 줄 수만큼(셋까지) 높이.
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
from evolab.app.editor.tabs import SEVERITY_ICONS, fix_widget
from evolab.spec.compiled import Issue

PANEL_MIN = 300
"""패널의 처음 최소 폭(px)."""
PANEL_MAX_MIN = 440
"""폼이 바라는 폭으로 올리는 최소 폭의 상한(px, 화면 다듬기 U38 ②)."""
ISSUE_ROWS_SHOWN = 3
"""검사 결과 목록이 스크롤 없이 보이는 줄 수(U40)."""


class PropertyPanel(QWidget):
    """고른 부품의 속성."""

    def __init__(self, doc: EnvDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self._ref: Ref | None = None
        self._path: JsonPath | None = None
        self._ids_before: set[str] = set()
        self.setMinimumWidth(PANEL_MIN)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        self.title = QLabel()
        font = QFont(self.title.font())
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.1)
        self.title.setFont(font)
        self.title.setWordWrap(True)
        self.title.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        outer.addWidget(self.title)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.hide()
        outer.addWidget(self.note)
        self.hint = QLabel(S.EDP_PANEL_HINT)
        self.hint.setWordWrap(True)
        self.hint.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.hint.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
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
        self.fix_area = QWidget()
        """그 부품의 검사 결과마다 「고치기」 단추 줄(6.2.0)."""
        self.fix_layout = QVBoxLayout(self.fix_area)
        self.fix_layout.setContentsMargins(0, 0, 0, 0)
        self.fix_layout.setSpacing(2)
        box_lay.addWidget(self.fix_area)
        self.fix_widgets: list[QWidget] = []
        outer.addWidget(self.issue_box)
        self.issue_box.hide()
        self.filler = QWidget()
        """빈 상태에서만 보이는 아래 늘임(제목·안내가 맨 위에 오게, U40)."""
        self.filler.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        outer.addWidget(self.filler, 1)

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
        self._fit_width()
        self._update_title()
        self._on_checked(list(self.doc.issues))

    def wanted_width(self) -> int:
        """지금 폼을 자르지 않는 패널 폭: 폼이 바라는 최소 폭 + 세로 스크롤 막대 + 여백(U38 ②)."""
        bar = self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        m = self.layout().contentsMargins()
        form = max(self.form.minimumSizeHint().width(), self.form.sizeHint().width() if self._ref is not None else 0,
                   self.header.minimumSizeHint().width())
        return form + bar + m.left() + m.right() + 4 + 6

    def _fit_width(self) -> None:
        self.setMinimumWidth(PANEL_MIN if self._ref is None else max(PANEL_MIN, min(PANEL_MAX_MIN, self.wanted_width())))

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
        self.filler.setVisible(empty)
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
        self._fit_issue_list()
        for w in self.fix_widgets:
            w.hide()
            w.deleteLater()
        self.fix_widgets = []
        for issue in mine:
            box = fix_widget(self.doc, issue)
            if box is not None:
                self.fix_layout.addWidget(box)
                self.fix_widgets.append(box)
        self.fix_area.setVisible(bool(self.fix_widgets))
        self.issue_box.setVisible(bool(mine))

    def _fit_issue_list(self) -> None:
        """목록 높이 = 줄 수만큼(ISSUE_ROWS_SHOWN까지). 줄이 더 많으면 스크롤(U40)."""
        n = self.issue_list.count()
        if n == 0:
            return
        width = max(self.issue_list.viewport().width(), 200)
        rows = 0
        for k in range(min(n, ISSUE_ROWS_SHOWN)):
            rect = self.issue_list.visualItemRect(self.issue_list.item(k))
            h = rect.height() if rect.height() > 0 else self.issue_list.sizeHintForRow(k)
            rows += max(h, self.fontMetrics().boundingRect(0, 0, width, 1000, Qt.TextFlag.TextWordWrap,
                                                           self.issue_list.item(k).text()).height() + 4)
        frame = 2 * self.issue_list.frameWidth()
        self.issue_list.setMaximumHeight(rows + frame + 2)

    def _on_delete(self) -> None:
        if self._ref is not None:
            self.doc.remove([self._ref])

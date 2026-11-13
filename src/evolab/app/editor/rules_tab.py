"""에디터 아래 「규칙」 탭 (6.2.0, 설계 DU 4.3.5 + 최종 설계 D11·D39, 3.1 RulesTab).

규칙 하나를 쉬운 문장 한 줄로 보인다. 규칙과 사건·종료 조건·점수 블록 조각 사이의 바꾸기는 presets/rules.py가 한다.

- 「자기 규칙」(editor.rules에 적혀 있고 닫힌 검사를 넘은 것, RuleUse.editable)은 문장의 빈칸(고르기·숫자)을 바로 고치고 ✕로 뺀다.
  더하기·빼기·빈칸 바꾸기는 각각 doc.edit 한 번(되돌리기 한 단계).
- 모양은 맞지만 적혀 있지 않은 규칙(내장 예제)은 읽기 전용 문장 + 「고급에서 고치기」. 문장으로 못 읽는 조각은 「고급 탭에 있는 규칙 n개」.
- 제한 시간은 늘 있는 고정 줄: 숫자만 고치고 뺄 수 없다.
- 「＋ 규칙 더하기 ▾」는 지금 장면에서 쓸 수 없는 규칙을 흐리게 두고 그 까닭을 풍선에 보인다.
"""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from evolab.app import strings as S
from evolab.app import style
from evolab.app.editor.document import EnvDocument
from evolab.presets import rules as R
from evolab.presets.josa import fill

_FIELD_RE = re.compile(r"\{(\w+)\}")
NUMBER_MAX = 1e6
"""숫자 빈칸의 위 끝(모델에는 끝이 없다: 화면에서만)."""


def _decimals(slot: R.RuleSlot) -> int:
    return 3 if slot.id in ("weight", "per_m", "per_s") else 2


@dataclass
class RuleRow:
    """화면의 규칙 한 줄."""

    use: R.RuleUse
    text: str
    """조사까지 맞춘 문장(rules.sentence)."""
    widget: QWidget
    editors: dict[str, QWidget] = field(default_factory=dict)
    """빈칸 id → 고르기·숫자 위젯(자기 규칙과 제한 시간만)."""
    remove_button: QToolButton | None = None
    advanced_button: QPushButton | None = None
    labels: list[QLabel] = field(default_factory=list)
    """문장 조각(빈칸 사이의 글). 빈칸 바로 뒤의 조사는 빈칸의 값에 맞춰 고친다."""

    @property
    def editable(self) -> bool:
        return self.use.editable

    @property
    def kind(self) -> str:
        return self.use.kind


class _PartsButton(QToolButton):
    """여러 부분 고르기(넘어짐의 「땅에 닿으면 넘어진 것으로 볼 부분들」): 체크 메뉴."""

    changed = Signal(list)

    def __init__(self, choices: list[tuple[str, str]], value: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._menu = QMenu(self)
        self._names = dict(choices)
        self._acts = []
        for cid, name in choices:
            act = self._menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(cid in value)
            act.setData(cid)
            act.toggled.connect(self._toggled)
            self._acts.append(act)
        self.setMenu(self._menu)
        self._show(value)

    def value(self) -> list[str]:
        return [str(a.data()) for a in self._acts if a.isChecked()]

    def set_value(self, value: list[str]) -> None:
        for a in self._acts:
            a.blockSignals(True)
            a.setChecked(str(a.data()) in value)
            a.blockSignals(False)
        self._show(value)

    def _show(self, value: list[str]) -> None:
        self.setText(", ".join(self._names.get(v, v) for v in value) + " ▾")

    def _toggled(self, _on: bool) -> None:
        value = self.value()
        if not value:  # 하나는 남긴다(빈 목록은 규칙이 받지 않음)
            self.set_value(self._last)
            return
        self._last = value
        self._show(value)
        self.changed.emit(value)

    _last: list[str] = []


class RulesTab(QWidget):
    """「규칙」 탭. rows()가 지금 보이는 줄, add_button이 「＋ 규칙 더하기 ▾」."""

    advanced_requested = Signal(str)
    """「고급에서 고치기」·「고급 탭에 있는 규칙 n개」를 눌렀다: 옮겨 갈 경로(events·scoring)."""

    def __init__(self, doc: EnvDocument, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.setObjectName("rules_tab")
        self._rows: list[RuleRow] = []
        self._sig: Any = None
        self._rest = 0

        self.body = QWidget()
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(4, 4, 4, 4)
        self.body_lay.setSpacing(4)
        self.empty = style.note(S.RULES_EMPTY)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(self.body)
        self.scroll = area

        self.add_button = QToolButton()
        self.add_button.setObjectName("rules_add")
        self.add_button.setText(S.RULES_ADD)
        self.add_button.setToolTip(S.RULES_ADD_TIP)
        self.add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.add_menu = QMenu(self.add_button)
        self.add_menu.setToolTipsVisible(True)
        self.add_menu.aboutToShow.connect(self._update_add_menu)
        self.add_actions: dict[str, Any] = {}
        for kind in R.RULES:
            if kind == "time_limit":
                continue
            act = self.add_menu.addAction(S.RULE_TITLES.get(kind, kind))
            act.triggered.connect(functools.partial(self.add_rule, kind))
            self.add_actions[kind] = act
        self.add_button.setMenu(self.add_menu)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setToolTip(S.PRESET_RULE_SUMMARY_TIP)
        style.tone(self.summary, "ink2")
        self.advanced = QPushButton()
        self.advanced.setFlat(True)
        self.advanced.setToolTip(S.RULES_ADVANCED_TIP)
        self.advanced.clicked.connect(lambda: self.advanced_requested.emit("events"))
        bottom = QHBoxLayout()
        bottom.setContentsMargins(4, 0, 4, 0)
        bottom.addWidget(self.add_button)
        bottom.addWidget(self.summary, 1)
        bottom.addWidget(self.advanced)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(area, 1)
        lay.addLayout(bottom)
        doc.changed.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------
    def rows(self) -> list[RuleRow]:
        return list(self._rows)

    def rest(self) -> int:
        """고급 탭에 남은 줄 수(문장으로 못 읽은 사건·종료·블록)."""
        return self._rest

    def _signature(self, infos: list[R.RuleRowInfo]) -> Any:
        data = self.doc.data
        cands = tuple(tuple(R.candidates(data, s)) for k in R.RULES for s in R.RULES[k].slots
                      if s.kind not in ("number", "choice"))
        return (tuple((i.use.kind, i.use.editable, i.use.fixed, i.use.event, i.use.blocks) for i in infos), cands)

    def refresh(self) -> None:
        data = self.doc.data
        infos, rest = R.rows(data)
        self._rest = rest
        sig = self._signature(infos)
        if sig != self._sig:
            self._rebuild(infos)
            self._sig = sig
        else:
            for row, info in zip(self._rows, infos, strict=True):
                row.use, row.text = info.use, info.text
                self._show_values(row)
        formula = bool((data.get("scoring") or {}).get("formula")) if isinstance(data.get("scoring"), dict) else False
        text = R.summary(data)
        self.summary.setText(text if formula else S.PRESET_RULE_SUMMARY.format(formula=text))
        self.advanced.setText(S.RULES_ADVANCED.format(n=rest))
        self.advanced.setEnabled(rest > 0)
        self.empty.setVisible(all(r.use.fixed for r in self._rows))

    def _rebuild(self, infos: list[R.RuleRowInfo]) -> None:
        while self.body_lay.count():
            item = self.body_lay.takeAt(0)
            w = item.widget()
            if w is not None and w is not self.empty:
                for child in [w, *w.findChildren(QWidget)]:
                    child.blockSignals(True)
                w.hide()
                w.deleteLater()
        self._rows = []
        for info in infos:
            row = self._make_row(info)
            self._rows.append(row)
            self.body_lay.addWidget(row.widget)
        self.body_lay.addWidget(self.empty)
        self.body_lay.addStretch(1)

    # ---- 한 줄 ----
    def _make_row(self, info: R.RuleRowInfo) -> RuleRow:
        use = info.use
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        row = RuleRow(use, info.text, w)
        if not use.editable:
            label = QLabel(info.text)
            label.setWordWrap(True)
            label.setToolTip(info.tip or S.RULES_READONLY_TIP)
            lay.addWidget(label, 1)
            btn = QPushButton(S.RULES_TO_ADVANCED)
            btn.setToolTip(S.RULES_READONLY_TIP)
            target = "events" if use.event else "scoring"
            btn.clicked.connect(lambda _=False, t=target: self.advanced_requested.emit(t))
            lay.addWidget(btn)
            row.advanced_button = btn
            return row
        template = S.RULE_SENTENCES[use.kind]
        slots = {s.id: s for s in R.RULES[use.kind].slots}
        pos = 0
        prev: str | None = None
        for mt in _FIELD_RE.finditer(template):
            self._add_text(row, lay, template[pos:mt.start()], prev)
            sid = mt.group(1)
            slot = slots.get(sid)
            if slot is not None:
                ed = self._editor(row, slot)
                row.editors[sid] = ed
                lay.addWidget(ed)
            prev = sid
            pos = mt.end()
        self._add_text(row, lay, template[pos:], prev)
        lay.addStretch(1)
        if use.fixed:
            w.setToolTip(S.RULES_FIXED_TIP)
        else:
            rm = QToolButton()
            rm.setText("✕")
            rm.setToolTip(S.RULES_REMOVE_TIP)
            rm.setAutoRaise(True)
            rm.clicked.connect(lambda _=False, r=row: self.remove(r))
            lay.addWidget(rm)
            row.remove_button = rm
        self._show_values(row)
        return row

    def _add_text(self, row: RuleRow, lay: QHBoxLayout, text: str, prev: str | None) -> None:
        if not text:
            return
        label = QLabel(text)
        label.setProperty("evoRuleText", text)
        label.setProperty("evoRuleAfter", prev or "")
        lay.addWidget(label)
        row.labels.append(label)

    def _editor(self, row: RuleRow, slot: R.RuleSlot) -> QWidget:
        data = self.doc.data
        tip = S.RULE_SLOT_TIPS.get(slot.id, "")
        if slot.kind == "number":
            spin = QDoubleSpinBox()
            spin.setDecimals(_decimals(slot))
            low = slot.low if slot.low is not None else -NUMBER_MAX
            if slot.id == "seconds":
                low = 0.1
            spin.setRange(low, NUMBER_MAX)
            spin.setKeyboardTracking(False)
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setToolTip(tip)
            spin.valueChanged.connect(lambda v, r=row, s=slot.id: self.set_slot(r, s, float(v)))
            return spin
        if slot.kind == "parts":
            btn = _PartsButton(R.candidates(data, slot), list(row.use.args.get(slot.id) or []))
            btn.setToolTip(tip)
            btn.changed.connect(lambda v, r=row, s=slot.id: self.set_slot(r, s, v))
            return btn
        combo = QComboBox()
        combo.setToolTip(tip)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        for value, name in R.candidates(data, slot):
            combo.addItem(name, value)
        combo.activated.connect(lambda i, r=row, s=slot.id, c=combo: self.set_slot(r, s, c.itemData(i)))
        return combo

    def _show_values(self, row: RuleRow) -> None:
        if not row.use.editable:
            return
        data = self.doc.data
        names: dict[str, str] = {}
        for sid, ed in row.editors.items():
            v = row.use.args.get(sid)
            ed.blockSignals(True)
            if isinstance(ed, QDoubleSpinBox):
                ed.setValue(float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else 0.0)
            elif isinstance(ed, _PartsButton):
                ed.set_value(list(v or []))
                ed._last = list(v or [])
                names[sid] = ", ".join(R.display_name(data, x) for x in v or [])
            elif isinstance(ed, QComboBox):
                i = ed.findData(v)
                if i < 0 and v is not None:
                    ed.addItem(R.display_name(data, str(v)), v)
                    i = ed.count() - 1
                ed.setCurrentIndex(i)
                names[sid] = ed.itemText(i) if i >= 0 else ""
            ed.blockSignals(False)
        for label in row.labels:
            text = str(label.property("evoRuleText"))
            name = names.get(str(label.property("evoRuleAfter")), "")
            if name:  # 빈칸 바로 뒤의 조사(「이(가)」「(으)로」…)를 빈칸의 값에 맞춘다
                text = fill("{x}" + text, {"x": name})[len(name):]
            label.setText(text)

    # ---- 편집 ----
    def _current(self, row: RuleRow) -> R.RuleUse | None:
        for u in R.find_rules(self.doc.data)[0]:
            if u.kind == row.use.kind and u.event == row.use.event and u.blocks == row.use.blocks:
                return u
            if row.use.fixed and u.fixed:
                return u
        return None

    def set_slot(self, row: RuleRow, slot: str, value: Any) -> None:
        """빈칸 하나를 바꾼다(되돌리기 한 단계)."""
        use = self._current(row)
        if use is None or not use.editable:
            return

        def mutate(d: dict[str, Any]) -> None:
            R.set_rule(d, use, **{slot: value})

        try:
            self.doc.edit(S.RULES_UNDO_SET, mutate)
        except ValueError:
            self.refresh()

    def remove(self, row: RuleRow) -> None:
        use = self._current(row)
        if use is None or not use.editable or use.fixed:
            return
        self.doc.edit(S.RULES_UNDO_REMOVE, lambda d: R.remove_rule(d, use))

    def add_rule(self, kind: str, **args: Any) -> R.RuleUse | None:
        """규칙 하나를 더한다(되돌리기 한 단계). 지금 장면에서 쓸 수 없으면 None."""
        ok, _why = R.available(self.doc.data, kind)
        if not ok:
            return None
        return self.doc.edit(S.RULES_UNDO_ADD, lambda d: R.add_rule(d, kind, **args))  # type: ignore[no-any-return]

    def _update_add_menu(self) -> None:
        data = self.doc.data
        for kind, act in self.add_actions.items():
            ok, why = R.available(data, kind)
            act.setEnabled(ok)
            act.setToolTip(why)
            act.setProperty("evoWhyOff", why if not ok else "")

    def available(self) -> dict[str, tuple[bool, str]]:
        """규칙 종류마다 (지금 더할 수 있나, 못 하면 까닭)."""
        self._update_add_menu()
        return {k: R.available(self.doc.data, k) for k in self.add_actions}

    def open_add_menu(self) -> None:
        """「고치기」의 「규칙 더하기…」: 더하기 메뉴를 연다."""
        self._update_add_menu()
        self.add_button.showMenu()


__all__ = ["RuleRow", "RulesTab"]

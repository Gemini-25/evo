"""「새 프리셋」 창의 다섯 쪽 (6.2.0, 최종 설계 2.1·D42, DU 4.1·5.1).

① 무엇을 만들까요(첫 카드) → ② 몸(몸 틀 카드 + 슬라이더) → ③ 경기장과 물건(경기장 카드 + 슬라이더, 물건 체크 + 슬라이더) →
④ 규칙과 점수(규칙 문장 체크 + 문장 안의 빈칸, 「점수 = …」 한 줄) → ⑤ 해 보기(요약, 조종 키, 학습할 수 있나).

쪽은 창(PresetWizard)의 조리법을 읽어 그리고(sync), 사용자가 바꾸면 창의 change(바꾸는 함수, 행동 이름)를 부른다. 조리법이 바뀌어도
위젯은 될 수 있는 한 그대로 두고 값만 바꾼다(키보드 초점이 남게). 위젯을 새로 만드는 때는 세계 종류·틀이 바뀔 때뿐이다.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from evolab.app import strings as S
from evolab.app import style
from evolab.app.preset.cards import Card, FlowLayout, ParamSliders, TemplateCard, clear_layout, template_icon
from evolab.presets import rules as R
from evolab.presets.josa import fill
from evolab.presets.recipe import Recipe, Use
from evolab.presets.template import Template, expand

if TYPE_CHECKING:
    from evolab.app.preset.wizard import PresetWizard

_FIELD = re.compile(r"\{(\w+)\}")
NUMBER_MAX = 100000.0
SECONDS_MAX = 3600.0


def _heading(text: str, sub: str) -> tuple[QLabel, QLabel]:
    title = QLabel(text)
    title.setFont(style.font(title, "title"))
    title.setWordWrap(True)
    return title, style.note(sub)


class Page(QWidget):
    """쪽 하나. step: 0부터의 단계 번호."""

    step = 0

    def __init__(self, wizard: PresetWizard) -> None:
        super().__init__()
        self.w = wizard
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(style.SPACE[1], style.SPACE[1], style.SPACE[1], style.SPACE[1])
        self.box.setSpacing(style.SPACE[1])

    def sync(self) -> None:  # pragma: no cover - 쪽마다 다시 정의
        raise NotImplementedError

    def controls(self) -> list[QWidget]:
        """키보드 차례(Tab)에 넣을 조작 위젯(보이는 차례)."""
        return []


def _set_checked(button: Any, on: bool) -> None:
    if button.isChecked() != on:
        button.blockSignals(True)
        button.setChecked(on)
        button.blockSignals(False)


# ---------------------------------------------------------------------------
# ① 무엇을 만들까요
# ---------------------------------------------------------------------------


class StartPage(Page):
    step = 0

    def __init__(self, wizard: PresetWizard) -> None:
        super().__init__(wizard)
        title, sub = _heading(S.PRESET_PAGE_START, S.PRESET_PAGE_START_SUB)
        self.box.addWidget(title)
        self.box.addWidget(sub)
        host = QWidget()
        self.flow = FlowLayout(host, spacing=style.SPACE[1])
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.cards: dict[str, Card] = {}
        for st in wizard.starters:
            body = wizard.starter_body(st.id)
            icon = template_icon(body) if body is not None else None
            text = st.title
            tip = wizard.starter_blurb(st.id)
            if st.status == "experimental":
                text = f"{text}\n{S.PRESET_BADGE_EXPERIMENTAL}"
                tip = f"{tip}\n\n{S.PRESET_BADGE_EXPERIMENTAL_TIP}" if tip else S.PRESET_BADGE_EXPERIMENTAL_TIP
            card = Card(st.id, text, tip, icon)
            card.setObjectName(f"start_{st.id}")
            self.group.addButton(card)
            self.flow.add(card)
            self.cards[st.id] = card
            card.toggled.connect(lambda on, key=st.id: on and self.w.choose_start(key))
        self.box.addWidget(host)
        self.blurb = style.note()
        self.box.addWidget(self.blurb)
        self.box.addStretch(1)

    def sync(self) -> None:
        start = self.w.start
        for key, card in self.cards.items():
            _set_checked(card, key == start)
        self.blurb.setText(self.w.starter_blurb(start))

    def controls(self) -> list[QWidget]:
        return list(self.cards.values())


# ---------------------------------------------------------------------------
# ② 몸
# ---------------------------------------------------------------------------


class BodyPage(Page):
    step = 1

    def __init__(self, wizard: PresetWizard) -> None:
        super().__init__(wizard)
        title, sub = _heading(S.PRESET_PAGE_BODY, S.PRESET_PAGE_BODY_SUB)
        self.box.addWidget(title)
        self.box.addWidget(sub)
        host = QWidget()
        self.flow = FlowLayout(host, spacing=style.SPACE[1])
        self.box.addWidget(host)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.cards: dict[str, TemplateCard] = {}
        self.sliders = ParamSliders()
        self.sliders.changed.connect(self._values_changed)
        self.box.addWidget(self.sliders)
        self.box.addStretch(1)
        self._world: str | None = None

    def _rebuild_cards(self) -> None:
        for card in list(self.cards.values()):
            self.group.removeButton(card)
        clear_layout(self.flow)
        self.cards = {}
        for tpl in self.w.library.templates("body", self.w.recipe.world):
            card = TemplateCard(tpl)
            card.setObjectName(f"body_{tpl.id}")
            self.group.addButton(card)
            self.flow.add(card)
            self.cards[tpl.id] = card
            card.toggled.connect(lambda on, tid=tpl.id: on and self._pick(tid))

    def _pick(self, tid: str) -> None:
        def mutate(r: Recipe) -> None:
            if r.roles and r.roles[0].body.template != tid:
                r.roles[0].body = Use(tid)

        self.w.change(mutate, "body")

    def _values_changed(self) -> None:
        values = self.sliders.values()

        def mutate(r: Recipe) -> None:
            if r.roles:
                r.roles[0].body.values = dict(values)

        self.w.change(mutate, "slider")

    def sync(self) -> None:
        r = self.w.recipe
        if self._world != r.world:
            self._world = r.world
            self._rebuild_cards()
        body = r.roles[0].body if r.roles else None
        for tid, card in self.cards.items():
            _set_checked(card, body is not None and tid == body.template)
        if body is None or not self.w.library.has(body.template):
            return
        tpl = self.w.library.get(body.template)
        if self.sliders.template is None or self.sliders.template.id != tpl.id:
            self.sliders.set_template(tpl, body.values)
        else:
            self.sliders.set_values(body.values)

    def controls(self) -> list[QWidget]:
        return [*self.cards.values(), *(self.sliders.widget_for(p) for p in self.sliders.param_ids())]


# ---------------------------------------------------------------------------
# ③ 경기장과 물건
# ---------------------------------------------------------------------------


def _part_ids(tpl: Template) -> set[str]:
    try:
        parts = expand(tpl)
    except Exception:  # noqa: BLE001 - 잠금 표시용: 펼치지 못하면 틀 id만
        return {tpl.id}
    out = {tpl.id}
    for sec in ("bodies", "regions"):
        out.update(str(x["id"]) for x in parts.get(sec, []) if "id" in x)
    return out


def _rule_refs(rule: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for k, v in rule.items():
        if k in ("rule", "who"):
            continue
        for x in v if isinstance(v, list) else [v]:
            if isinstance(x, str):
                out.add(x)
    return out


class ArenaPage(Page):
    step = 2

    def __init__(self, wizard: PresetWizard) -> None:
        super().__init__(wizard)
        title, sub = _heading(S.PRESET_PAGE_ARENA, S.PRESET_PAGE_ARENA_SUB)
        self.box.addWidget(title)
        self.box.addWidget(sub)
        arena_title = QLabel(S.PRESET_ARENA_LABEL)
        arena_title.setFont(style.font(arena_title, "strong"))
        arena_title.setToolTip(S.PRESET_ARENA_TIP)
        self.box.addWidget(arena_title)
        host = QWidget()
        self.flow = FlowLayout(host, spacing=style.SPACE[1])
        self.box.addWidget(host)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.cards: dict[str, TemplateCard] = {}
        self.arena_sliders = ParamSliders()
        self.arena_sliders.changed.connect(self._arena_values)
        self.box.addWidget(self.arena_sliders)
        props_title = QLabel(S.PRESET_PROPS_LABEL)
        props_title.setFont(style.font(props_title, "strong"))
        props_title.setToolTip(S.PRESET_PROPS_TIP)
        self.box.addWidget(props_title)
        self.props_host = QWidget()
        self.props_box = QVBoxLayout(self.props_host)
        self.props_box.setContentsMargins(0, 0, 0, 0)
        self.box.addWidget(self.props_host)
        self.box.addStretch(1)
        self.checks: dict[str, QCheckBox] = {}
        self.prop_sliders: dict[str, ParamSliders] = {}
        self._world: str | None = None

    def _rebuild(self) -> None:
        for card in list(self.cards.values()):
            self.group.removeButton(card)
        clear_layout(self.flow)
        self.cards = {}
        world = self.w.recipe.world
        for tpl in self.w.library.templates("arena", world):
            card = TemplateCard(tpl)
            card.setObjectName(f"arena_{tpl.id}")
            self.group.addButton(card)
            self.flow.add(card)
            self.cards[tpl.id] = card
            card.toggled.connect(lambda on, tid=tpl.id: on and self._pick_arena(tid))
        clear_layout(self.props_box)
        self.checks = {}
        self.prop_sliders = {}
        for tpl in self.w.library.templates("prop", world):
            box = QCheckBox(tpl.name)
            box.setObjectName(f"prop_{tpl.id}")
            box.setToolTip(tpl.description or tpl.name)
            box.toggled.connect(lambda on, tid=tpl.id: self._toggle_prop(tid, on))
            self.props_box.addWidget(box)
            sliders = ParamSliders(tpl)
            sliders.setContentsMargins(style.SPACE[3], 0, 0, 0)
            sliders.changed.connect(lambda tid=tpl.id: self._prop_values(tid))
            self.props_box.addWidget(sliders)
            self.checks[tpl.id] = box
            self.prop_sliders[tpl.id] = sliders

    def _pick_arena(self, tid: str) -> None:
        def mutate(r: Recipe) -> None:
            if r.arena.template != tid:
                r.arena = Use(tid)

        self.w.change(mutate, "arena")

    def _arena_values(self) -> None:
        values = self.arena_sliders.values()

        def mutate(r: Recipe) -> None:
            r.arena.values = dict(values)

        self.w.change(mutate, "slider")

    def _spawn_for(self, tid: str) -> str | None:
        r = self.w.recipe
        if not self.w.library.has(r.arena.template):
            return None
        spawns = self.w.library.get(r.arena.template).spawns
        for name in (tid, "ahead", "center"):
            if name in spawns:
                return name
        return None

    def _toggle_prop(self, tid: str, on: bool) -> None:
        spawn = self._spawn_for(tid)

        def mutate(r: Recipe) -> None:
            have = [p for p in r.props if p.template == tid]
            if on and not have:
                r.props.append(Use(tid, {}, None, spawn))
            elif not on and have:
                r.props = [p for p in r.props if p.template != tid]

        self.w.change(mutate, "prop")

    def _prop_values(self, tid: str) -> None:
        values = self.prop_sliders[tid].values()

        def mutate(r: Recipe) -> None:
            for p in r.props:
                if p.template == tid:
                    p.values = dict(values)
                    break

        self.w.change(mutate, "slider")

    def locks(self) -> dict[str, str]:
        """끌 수 없는 물건(틀 id → 까닭: 그것을 쓰는 규칙 이름)."""
        out: dict[str, str] = {}
        r = self.w.recipe
        refs: list[tuple[set[str], str]] = [(_rule_refs(rule), S.RULE_TITLES.get(str(rule.get("rule")), ""))
                                            for rule in r.rules if rule.get("rule") != "time_limit"]
        for use in r.props:
            if not self.w.library.has(use.template):
                continue
            ids = _part_ids(self.w.library.get(use.template))
            for names, title in refs:
                if names & ids:
                    out[use.template] = title
                    break
        return out

    def sync(self) -> None:
        r = self.w.recipe
        if self._world != r.world:
            self._world = r.world
            self._rebuild()
        for tid, card in self.cards.items():
            _set_checked(card, tid == r.arena.template)
        if self.w.library.has(r.arena.template):
            tpl = self.w.library.get(r.arena.template)
            if self.arena_sliders.template is None or self.arena_sliders.template.id != tpl.id:
                self.arena_sliders.set_template(tpl, r.arena.values)
            else:
                self.arena_sliders.set_values(r.arena.values)
        used = {p.template: p for p in r.props}
        locks = self.locks()
        for tid, box in self.checks.items():
            on = tid in used
            _set_checked(box, on)
            why = S.PRESET_PROP_LOCKED_TIP.format(why=locks[tid]) if tid in locks else ""
            style.set_enabled(box, not why, why)
            sliders = self.prop_sliders[tid]
            sliders.setVisible(on and bool(sliders.param_ids()))
            if on:
                sliders.set_values(used[tid].values)

    def controls(self) -> list[QWidget]:
        out: list[QWidget] = [*self.cards.values()]
        out += [self.arena_sliders.widget_for(p) for p in self.arena_sliders.param_ids()]
        for tid, box in self.checks.items():
            out.append(box)
            sl = self.prop_sliders[tid]
            out += [sl.widget_for(p) for p in sl.param_ids()]
        return out


# ---------------------------------------------------------------------------
# ④ 규칙과 점수
# ---------------------------------------------------------------------------


class PartsButton(QToolButton):
    """여러 부분 고르기(「넘어지면 끝」의 땅에 닿으면 안 되는 부분들): 누르면 체크 메뉴."""

    def __init__(self, on_change: Callable[[list[str]], None]) -> None:
        super().__init__()
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.menu_ = QMenu(self)
        self.setMenu(self.menu_)
        self._on_change = on_change
        self._actions: list[QAction] = []

    def set_items(self, items: list[tuple[str, str]], chosen: list[str]) -> None:
        if [(a.data(), a.text()) for a in self._actions] != items:
            self.menu_.clear()
            self._actions = []
            for value, label in items:
                act = QAction(label, self.menu_)
                act.setCheckable(True)
                act.setData(value)
                act.toggled.connect(self._toggled)
                self.menu_.addAction(act)
                self._actions.append(act)
        for act in self._actions:
            act.blockSignals(True)
            act.setChecked(act.data() in chosen)
            act.blockSignals(False)
        self.setText(", ".join(a.text() for a in self._actions if a.isChecked()) or "-")

    def chosen(self) -> list[str]:
        return [str(a.data()) for a in self._actions if a.isChecked()]

    def _toggled(self, _on: bool) -> None:
        self._on_change(self.chosen())


class RuleRow(QWidget):
    """규칙 한 줄: 체크(제한 시간은 이름만) + 문장 안의 빈칸(글과 위젯이 폭에 맞춰 줄을 바꿈)."""

    def __init__(self, page: RulesPage, kind: str) -> None:
        super().__init__()
        self.page = page
        self.kind = kind
        self.fixed = kind == "time_limit"
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, style.SPACE[1])
        box.setSpacing(2)
        if self.fixed:
            self.check: QCheckBox | None = None
            head = QLabel(S.RULE_TITLES[kind])
            head.setFont(style.font(head, "strong"))
            head.setToolTip(S.RULES_FIXED_TIP)
            box.addWidget(head)
        else:
            self.check = QCheckBox(S.RULE_TITLES[kind])
            self.check.setObjectName(f"rule_{kind}")
            self.check.toggled.connect(self._toggled)
            box.addWidget(self.check)
        host = QWidget()
        self.flow = FlowLayout(host, spacing=3)
        host.setContentsMargins(style.SPACE[3], 0, 0, 0)
        box.addWidget(host)
        self.slots: dict[str, QWidget] = {}
        self.after: dict[str, tuple[QLabel, str]] = {}
        """빈칸 바로 뒤의 첫 낱말(조사를 값에 맞춰 바꿈): 빈칸 id → (이름표, 원래 글)."""
        self._build()

    def _build(self) -> None:
        sentence = S.RULE_SENTENCES[self.kind]
        kinds = {s.id: s for s in R.RULES[self.kind].slots}
        pos = 0
        prev: str | None = None
        for m in _FIELD.finditer(sentence):
            self._words(sentence[pos:m.start()], prev)
            sid = m.group(1)
            slot = kinds.get(sid)
            if slot is not None:
                widget = self._slot_widget(slot)
                self.slots[sid] = widget
                self.flow.add(widget)
                prev = sid
            pos = m.end()
        self._words(sentence[pos:], prev)

    def _words(self, text: str, prev: str | None) -> None:
        words = [w for w in text.split(" ")]
        first = True
        for word in words:
            if not word:
                first = False
                continue
            label = QLabel(word)
            self.flow.add(label)
            if first and prev is not None and not text.startswith(" "):
                self.after[prev] = (label, text.split(" ", 1)[0])
            first = False

    def _slot_widget(self, slot: R.RuleSlot) -> QWidget:
        tip = S.RULE_SLOT_TIPS.get(slot.id, "")
        widget: QWidget
        if slot.kind == "parts":
            widget = PartsButton(lambda values, sid=slot.id: self.page.set_slot(self.kind, sid, values))
        elif slot.kind == "number":
            spin = QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setKeyboardTracking(False)
            if slot.id == "seconds":
                spin.setRange(1.0, SECONDS_MAX)
                spin.setSingleStep(1.0)
            else:
                spin.setRange(slot.low if slot.low is not None else -NUMBER_MAX, NUMBER_MAX)
                spin.setSingleStep(0.1 if (slot.default or 0) < 5 else 1.0)
            spin.valueChanged.connect(lambda v, sid=slot.id: self.page.set_slot(self.kind, sid, round(float(v), 6)))
            widget = spin
        else:
            combo = QComboBox()
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            combo.activated.connect(lambda i, sid=slot.id, c=combo: self.page.set_slot(self.kind, sid, c.itemData(i)))
            widget = combo
        widget.setToolTip(tip)
        widget.setObjectName(f"rule_{self.kind}_{slot.id}")
        return widget

    def _toggled(self, on: bool) -> None:
        self.page.toggle_rule(self.kind, on)

    def sync(self, data: Mapping[str, Any], use: R.RuleUse | None, recipe_rule: Mapping[str, Any] | None) -> None:
        on = self.fixed or recipe_rule is not None
        if self.check is not None:
            _set_checked(self.check, on)
            why = ""
            if not on:
                ok, why = R.available(data, self.kind)
                why = "" if ok else why
            style.set_enabled(self.check, not why, why)
        args: Mapping[str, Any] = use.args if use is not None else {}
        for slot in R.RULES[self.kind].slots:
            widget = self.slots.get(slot.id)
            if widget is None:
                continue
            value = args.get(slot.id, recipe_rule.get(slot.id) if recipe_rule else None)
            if value is None:
                value = slot.default
            shown = ""
            if isinstance(widget, PartsButton):
                items = R.candidates(data, slot)
                chosen = [v for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)]
                widget.set_items(items, chosen)
                shown = widget.text()
            elif isinstance(widget, QDoubleSpinBox):
                v = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
                if abs(widget.value() - v) > 1e-9:
                    widget.blockSignals(True)
                    widget.setValue(v)
                    widget.blockSignals(False)
            elif isinstance(widget, QComboBox):
                items = list(R.candidates(data, slot))
                if isinstance(value, str) and value not in [v for v, _ in items]:
                    items.append((value, R.display_name(data, value)))
                current = [(widget.itemData(i), widget.itemText(i)) for i in range(widget.count())]
                if current != items:
                    widget.blockSignals(True)
                    widget.clear()
                    for v, label in items:
                        widget.addItem(label, v)
                    widget.blockSignals(False)
                idx = widget.findData(value)
                if idx >= 0 and widget.currentIndex() != idx:
                    widget.blockSignals(True)
                    widget.setCurrentIndex(idx)
                    widget.blockSignals(False)
                shown = widget.currentText()
            widget.setEnabled(on)
            if slot.id in self.after and shown:
                label, raw = self.after[slot.id]
                label.setText(fill("{v}" + raw, {"v": shown})[len(shown):])

    def controls(self) -> list[QWidget]:
        out: list[QWidget] = [self.check] if self.check is not None else []
        return out + list(self.slots.values())


class RulesPage(Page):
    step = 3

    def __init__(self, wizard: PresetWizard) -> None:
        super().__init__(wizard)
        title, sub = _heading(S.PRESET_PAGE_RULES, S.PRESET_PAGE_RULES_SUB)
        self.box.addWidget(title)
        self.box.addWidget(sub)
        self.rows: dict[str, RuleRow] = {}
        for kind in R.RULES:
            row = RuleRow(self, kind)
            self.rows[kind] = row
            self.box.addWidget(row)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setToolTip(S.PRESET_RULE_SUMMARY_TIP)
        self.summary.setFont(style.font(self.summary, "strong"))
        self.box.addWidget(self.summary)
        self.box.addStretch(1)

    def toggle_rule(self, kind: str, on: bool) -> None:
        def mutate(r: Recipe) -> None:
            have = [x for x in r.rules if x.get("rule") == kind]
            if on and not have:
                at = next((i for i, x in enumerate(r.rules) if x.get("rule") == "time_limit"), len(r.rules))
                r.rules.insert(at, {"rule": kind, "who": r.roles[0].id if r.roles else "player"})
            elif not on and have:
                r.rules = [x for x in r.rules if x.get("rule") != kind]

        self.w.change(mutate, "rule")

    def set_slot(self, kind: str, sid: str, value: Any) -> None:
        def mutate(r: Recipe) -> None:
            for x in r.rules:
                if x.get("rule") == kind:
                    x[sid] = copy.deepcopy(value)
                    return
            if kind == "time_limit":
                r.rules.append({"rule": kind, sid: value})

        self.w.change(mutate, "rule_slot")

    def sync(self) -> None:
        data = self.w.data or {}
        try:
            uses, _ = R.find_rules(data)
        except Exception:  # noqa: BLE001 - 알아보기는 보여 주기용
            uses = []
        by_kind: dict[str, R.RuleUse] = {}
        for u in uses:
            by_kind.setdefault(u.kind, u)
        recipe_rules: dict[str, Mapping[str, Any]] = {}
        for x in self.w.recipe.rules:
            recipe_rules.setdefault(str(x.get("rule")), x)
        for kind, row in self.rows.items():
            row.sync(data, by_kind.get(kind), recipe_rules.get(kind))
        try:
            formula = R.summary(data)
        except Exception:  # noqa: BLE001
            formula = ""
        self.summary.setText(S.PRESET_RULE_SUMMARY.format(formula=formula) if formula else "")

    def controls(self) -> list[QWidget]:
        out: list[QWidget] = []
        for row in self.rows.values():
            out += row.controls()
        return out


# ---------------------------------------------------------------------------
# ⑤ 해 보기
# ---------------------------------------------------------------------------


class TryPage(Page):
    step = 4

    def __init__(self, wizard: PresetWizard) -> None:
        super().__init__(wizard)
        title, sub = _heading(S.PRESET_PAGE_TRY, S.PRESET_PAGE_TRY_SUB)
        self.box.addWidget(title)
        self.box.addWidget(sub)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.box.addWidget(self.summary)
        keys_title = QLabel(S.PRESET_KEYS_TITLE)
        keys_title.setFont(style.font(keys_title, "strong"))
        self.box.addWidget(keys_title)
        self.keys = QLabel()
        self.keys.setWordWrap(True)
        self.keys.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.box.addWidget(self.keys)
        self.check = QLabel()
        self.check.setWordWrap(True)
        self.box.addWidget(self.check)
        self.box.addStretch(1)

    def sync(self) -> None:
        spec = self.w.spec
        if spec is None:
            self.summary.setText("")
            self.keys.setText("")
            return
        f = spec.file
        hidden = f.brain.hidden if f.brain is not None else []
        try:
            from html import escape

            name = escape(f.name)
        except Exception:  # noqa: BLE001
            name = f.name
        self.summary.setText(S.PRESET_SUMMARY.format(
            name=name, bodies=sum(1 for b in f.bodies if b.owner == "agent"), joints=len(f.joints),
            actuators=len(f.actuators), sensors=len(f.sensors), inputs=len(spec.inputs),
            hidden="·".join(str(h) for h in hidden) or "0", outputs=len(spec.outputs), seconds=float(f.world.max_time)))
        from evolab.app.play.window import key_bindings

        lines = [S.PLAY_KEY_LINE.format(keys=", ".join(b.key_names), label=b.label) for b in key_bindings(spec)]
        self.keys.setText("\n".join(lines) if lines else S.PRESET_KEYS_NONE)
        self.check.setText(self.w.check_text())


__all__ = ["ArenaPage", "BodyPage", "Page", "RuleRow", "RulesPage", "StartPage", "TryPage"]

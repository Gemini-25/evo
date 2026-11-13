"""6.2.0 에디터 「규칙」 탭 (설계 6.1 T30, D11·D39, DU 4.3.5·T26): editor/rules_tab.py.

자기 규칙 더하기·빼기·빈칸 바꾸기가 되돌리기 한 단계, 남의 규칙은 읽기 전용과 「고급에서 고치기」, 장면에 없는 것이 필요한 규칙은 흐림,
요약 줄, 제한 시간 고정 줄, 빈칸마다 풍선.
"""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox

from evolab.app import strings as S
from evolab.app.editor.document import EnvDocument
from evolab.app.editor.rules_tab import RulesTab
from evolab.app.editor.tabs import BottomTabs
from evolab.presets import recipe as RC
from evolab.presets import rules as R
from evolab.spec.editor_meta import RULES_KEY


def _scene() -> dict[str, Any]:
    return {
        "version": 1, "name": "규칙",
        "world": {"gravity": [0.0, -9.81], "max_time": 10.0},
        "bodies": [
            {"id": "ground", "label": "땅", "owner": "world", "type": "static", "shapes": [{"kind": "box", "w": 20, "h": 0.4}],
             "tags": ["ground"]},
            {"id": "torso", "label": "몸통", "pos": [0, 1.0], "shapes": [{"kind": "box", "w": 0.4, "h": 0.4}]},
        ],
        "actuators": [{"kind": "thruster", "id": "up", "body": "torso", "max_force": 10.0, "keys": ["Up"]}],
        "sensors": [{"kind": "body_state", "id": "st", "body": "torso", "fields": ["vx"]}],
        "termination": [{"event": "timeout", "outcome": "neutral"}],
    }


def _doc(data: dict[str, Any] | str) -> EnvDocument:
    doc = EnvDocument()
    doc.load_text(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False), None)
    return doc


def _tab(qtbot: Any, doc: EnvDocument) -> RulesTab:
    tab = RulesTab(doc)
    qtbot.addWidget(tab)
    tab.resize(1100, 320)
    tab.show()
    return tab


def _dump(doc: EnvDocument) -> str:
    return json.dumps(doc.data, sort_keys=True, ensure_ascii=False)


def test_add_and_remove_own_rule_are_one_undo_step_each(qtbot: Any) -> None:
    doc = _doc(_scene())
    tab = _tab(qtbot, doc)
    start = doc.text()
    assert [r.kind for r in tab.rows()] == ["time_limit"]
    assert tab.empty.isVisibleTo(tab)
    use = tab.add_rule("distance")
    assert use is not None and doc.undo_stack.index() == 1
    rows = tab.rows()
    assert [r.kind for r in rows] == ["distance", "time_limit"] and rows[0].editable
    assert doc.data["editor"][RULES_KEY][0]["kind"] == "distance"
    assert not tab.empty.isVisibleTo(tab)
    assert rows[0].text == "몸통이 오른쪽으로 간 거리만큼 점수 (1m에 1점)"
    rm = rows[0].remove_button
    assert rm is not None and rm.toolTip() == S.RULES_REMOVE_TIP
    rm.click()
    assert doc.undo_stack.index() == 2 and doc.text() == start  # 빼면 처음 환경과 같음
    doc.undo_stack.undo()
    assert [r.kind for r in tab.rows()] == ["distance", "time_limit"]
    doc.undo_stack.undo()
    assert doc.text() == start


def test_changing_a_blank_is_one_undo_step_and_fills_particles(qtbot: Any) -> None:
    doc = _doc(_scene())
    tab = _tab(qtbot, doc)
    tab.add_rule("distance")
    row = tab.rows()[0]
    direction = row.editors["direction"]
    assert isinstance(direction, QComboBox)
    k = direction.findData("up")
    direction.setCurrentIndex(k)
    direction.activated.emit(k)
    assert doc.undo_stack.index() == 2
    (use,) = [u for u in R.find_rules(doc.data)[0] if u.kind == "distance"]
    assert use.args["direction"] == "up"
    row = tab.rows()[0]
    texts = [lab.text() for lab in row.labels]
    assert texts[1].startswith("로 ")  # 「위로」: 빈칸의 값에 맞춘 조사
    spin = row.editors["per_m"]
    assert isinstance(spin, QDoubleSpinBox)
    spin.setValue(2.5)
    assert doc.undo_stack.index() == 3
    (use,) = [u for u in R.find_rules(doc.data)[0] if u.kind == "distance"]
    assert use.args["per_m"] == 2.5
    doc.undo_stack.undo()
    (use,) = [u for u in R.find_rules(doc.data)[0] if u.kind == "distance"]
    assert use.args["per_m"] == 1.0
    assert tab.rows()[0].editors["per_m"].value() == 1.0  # type: ignore[attr-defined]


def test_every_blank_has_a_tooltip(qtbot: Any) -> None:
    doc = _doc(json.dumps(RC.build(RC.load_builtin("running")), ensure_ascii=False))
    tab = _tab(qtbot, doc)
    seen = set()
    for row in tab.rows():
        for sid, ed in row.editors.items():
            assert ed.toolTip() == S.RULE_SLOT_TIPS[sid]
            seen.add(sid)
    assert {"part", "parts", "what", "root", "height", "penalty", "direction", "per_m", "weight", "outcome",
            "seconds"} <= seen


def test_other_rules_are_read_only_with_go_to_advanced(qtbot: Any) -> None:
    doc = EnvDocument()
    doc.from_builtin("hill_car.evoenv")
    tabs = BottomTabs(doc)
    qtbot.addWidget(tabs)
    tabs.show()
    tab = tabs.rules_tab
    rows = tab.rows()
    assert rows and all(not r.editable for r in rows if not r.use.fixed)
    ro = [r for r in rows if not r.editable]
    assert ro
    for r in ro:
        assert r.editors == {} and r.remove_button is None
        assert r.advanced_button is not None and r.advanced_button.text() == S.RULES_TO_ADVANCED
        assert r.advanced_button.toolTip() == S.RULES_READONLY_TIP
    before = json.dumps(doc.data, sort_keys=True)
    assert tab.advanced.text() == S.RULES_ADVANCED.format(n=tab.rest()) and tab.rest() > 0
    ro[0].advanced_button.click()  # type: ignore[union-attr]
    assert tabs.currentWidget() in (tabs.events_tab, tabs.scoring_tab)
    tabs.setCurrentWidget(tab)
    tab.advanced.click()
    assert tabs.currentWidget() is tabs.events_tab
    assert json.dumps(doc.data, sort_keys=True) == before and doc.undo_stack.count() == 0  # 읽기만으로 안 바뀜


def test_rules_tab_is_first_but_world_tab_opens(qtbot: Any) -> None:
    doc = EnvDocument()
    tabs = BottomTabs(doc)
    qtbot.addWidget(tabs)
    assert tabs.widget(0) is tabs.rules_tab and tabs.tabText(0) == S.EDP_TAB_RULES
    assert tabs.currentWidget() is tabs.world_tab
    assert tabs.tabText(tabs.indexOf(tabs.events_tab)) == S.EDP_TAB_EVENTS
    assert tabs.rules_tab.objectName() == "rules_tab"


def test_rules_needing_missing_things_are_greyed_with_a_reason(qtbot: Any) -> None:
    doc = _doc(_scene())
    tab = _tab(qtbot, doc)
    got = tab.available()
    assert got["reach"][0] is False and got["reach"][1] == "영역(골·결승선 같은 것)이 있어야 합니다."
    assert got["distance"] == (True, "")
    assert not tab.add_actions["reach"].isEnabled()
    assert tab.add_actions["reach"].toolTip() == got["reach"][1]
    assert tab.add_rule("reach") is None and doc.undo_stack.count() == 0
    assert tab.add_button.text() == S.RULES_ADD and tab.add_button.toolTip() == S.RULES_ADD_TIP
    assert "time_limit" not in tab.add_actions  # 고정 줄은 더하지 않음
    # 이미 있는 「밖으로 나가면 끝」은 다시 더하지 않음
    tab.add_rule("out_ends")
    assert tab.available()["out_ends"] == (False, S.RULES_ALREADY)
    # 옆에서 본 세계 규칙은 위에서 본 세계에서 흐림
    top = _scene()
    top["world"] = {"gravity": [0.0, 0.0], "view": "top", "damping": 0.5}
    tab2 = _tab(qtbot, _doc(top))
    ok, why = tab2.available()["fall"]
    assert not ok and why == S.TPL_WRONG_WORLD.format(world=S.TPL_WORLD_NAMES["side"])


def test_summary_line_and_formula_note(qtbot: Any) -> None:
    doc = _doc(_scene())
    tab = _tab(qtbot, doc)
    assert tab.summary.text() == S.PRESET_RULE_SUMMARY.format(formula=S.RULE_SUMMARY_NONE)
    tab.add_rule("distance")
    tab.add_rule("save_energy")
    assert tab.summary.text() == S.PRESET_RULE_SUMMARY.format(formula=R.summary(doc.data))
    assert "오른쪽으로 간 거리 × 1" in tab.summary.text() and "힘 × 0.05" in tab.summary.text()
    data = _scene()
    data["scoring"] = {"formula": "time"}
    tab2 = _tab(qtbot, _doc(data))
    assert tab2.summary.text() == S.RULES_FORMULA_NOTE
    assert tab2.available()["distance"] == (False, S.RULES_FORMULA_NOTE)


def test_time_limit_is_a_fixed_row(qtbot: Any) -> None:
    doc = _doc(_scene())
    tab = _tab(qtbot, doc)
    (row,) = [r for r in tab.rows() if r.kind == "time_limit"]
    assert row.use.fixed and row.remove_button is None and row.widget.toolTip() == S.RULES_FIXED_TIP
    spin = row.editors["seconds"]
    assert isinstance(spin, QDoubleSpinBox) and spin.value() == 10.0
    spin.setValue(12.0)
    assert doc.data["world"]["max_time"] == 12.0 and doc.undo_stack.index() == 1
    assert RULES_KEY not in doc.data.get("editor", {})  # 고정 줄은 editor.rules에 적지 않음


def test_fall_rule_parts_menu_changes_the_rule(qtbot: Any) -> None:
    doc = _doc(json.dumps(RC.build(RC.load_builtin("running")), ensure_ascii=False))
    tab = _tab(qtbot, doc)
    (row,) = [r for r in tab.rows() if r.kind == "fall"]
    assert row.editable
    parts = row.editors["parts"]
    before = list(row.use.args["parts"])
    acts = parts.menu().actions()  # type: ignore[attr-defined]
    extra = next(a for a in acts if not a.isChecked())
    extra.setChecked(True)
    (use,) = [u for u in R.find_rules(doc.data)[0] if u.kind == "fall"]
    assert list(use.args["parts"]) == [*before, str(extra.data())] or set(use.args["parts"]) == {*before, str(extra.data())}
    assert doc.undo_stack.index() == 1

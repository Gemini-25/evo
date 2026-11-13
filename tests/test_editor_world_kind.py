"""6.2.0 에디터 세계 종류와 형식 4 부품의 폼 (설계 6.1 T33, T34의 ED2 몫, D45, DU 4.3.8·T27).

- T33: 세계 종류 두 단추가 view·중력·바닥 마찰을 한 단계로 바꾸고, 손으로 적은 중력은 「직접 정함」으로 보인다. 옆에서 본 세계로 돌아가면
  가장 낮은 형식으로 남는다.
- T34(ED2): 밀기·균형 보조의 폼과 세계의 새 칸마다 이름과 풍선(DE T11).
"""

from __future__ import annotations

import json
from typing import Any

from evolab.app import strings as S
from evolab.app.editor.document import EnvDocument
from evolab.app.editor.forms import field_name, field_tip, fields_of
from evolab.app.editor.properties import PropertyPanel
from evolab.app.editor.tabs import WorldTab, set_world_kind, world_kind
from evolab.spec import model as m
from evolab.spec.io import dump_env, parse_env


def _tab(qtbot: Any, doc: EnvDocument) -> WorldTab:
    tab = WorldTab(doc)
    qtbot.addWidget(tab)
    tab.resize(1100, 300)
    tab.show()
    return tab


def _version(doc: EnvDocument) -> int:
    return int(json.loads(dump_env(parse_env(doc.data)))["version"])


def test_new_document_is_a_side_world(qtbot: Any) -> None:
    doc = EnvDocument()
    tab = _tab(qtbot, doc)
    row = tab.kind_row
    assert row.kind() == "side" and row.buttons["side"].isChecked() and not row.buttons["top"].isChecked()
    assert not row.custom.isVisibleTo(tab)
    assert row.objectName() == "world_kind" and row.toolTip() == S.EDP_WORLD_KIND_TIP
    for kind, b in row.buttons.items():
        assert b.text() == S.EDP_WORLD_KINDS[kind] and b.toolTip() == S.EDP_WORLD_KIND_TIPS[kind]
    assert row.label.text() == S.EDP_WORLD_KIND


def test_top_and_back_to_side_in_one_step_each(qtbot: Any) -> None:
    doc = EnvDocument()
    tab = _tab(qtbot, doc)
    original = doc.text()
    assert _version(doc) == 1
    tab.kind_row.buttons["top"].click()
    assert doc.undo_stack.index() == 1 and doc.undo_stack.undoText() == S.EDP_UNDO_WORLD_KIND
    world = doc.data["world"]
    assert world["view"] == "top" and world["gravity"] == [0.0, 0.0] and world["damping"] == 0.5
    assert tab.kind_row.kind() == "top" and tab.kind_row.buttons["top"].isChecked()
    assert _version(doc) == 4
    assert not [i for i in doc.run_check() if i.severity == "error"]  # 중력 0이라 top_gravity 없음
    tab.kind_row.buttons["side"].click()
    assert doc.undo_stack.index() == 2
    assert doc.text() == original and _version(doc) == 1  # 가장 낮은 형식으로 돌아감
    doc.undo_stack.undo()
    assert tab.kind_row.kind() == "top"
    doc.undo_stack.undo()
    assert doc.text() == original


def test_clicking_the_current_kind_does_nothing(qtbot: Any) -> None:
    doc = EnvDocument()
    tab = _tab(qtbot, doc)
    tab.kind_row.buttons["side"].click()
    assert doc.undo_stack.count() == 0 and tab.kind_row.buttons["side"].isChecked()


def test_hand_written_gravity_is_custom(qtbot: Any) -> None:
    doc = EnvDocument()
    doc.from_builtin("lander.evoenv")  # 달의 중력 −1.62
    tab = _tab(qtbot, doc)
    assert tab.kind_row.kind() == "custom"
    assert tab.kind_row.custom.isVisibleTo(tab) and tab.kind_row.custom.text() == S.EDP_WORLD_KINDS["custom"]
    assert not any(b.isChecked() for b in tab.kind_row.buttons.values())
    tab.kind_row.buttons["side"].click()
    assert doc.data["world"]["gravity"] == [0.0, -9.81] and tab.kind_row.kind() == "side"
    assert world_kind({"world": {"gravity": [0.0, -9.81]}}) == "side"
    assert world_kind({}) == "side"
    assert world_kind({"world": {"view": "top", "gravity": [0.0, -1.0]}}) == "top"
    data: dict[str, Any] = {}
    set_world_kind(data, "top")
    assert data == {"world": {"view": "top", "gravity": [0.0, 0.0], "damping": 0.5}}


def test_world_form_shows_floor_friction_not_view(qtbot: Any) -> None:
    doc = EnvDocument()
    tab = _tab(qtbot, doc)
    assert tab.world_form.field("view") is None
    damping = tab.world_form.field("damping")
    assert damping is not None and damping.tooltip()
    assert field_name(m.World, "damping") != "damping" and field_name(m.World, "view") == S.EDP_WORLD_KIND


def test_new_v4_fields_have_names_and_tooltips() -> None:
    for model in (m.Move, m.Balance):
        for f in fields_of(model):
            if f.key == "kind":
                continue
            assert field_name(model, f.key) != f.key, (model.__name__, f.key)
            assert field_tip(model, f.name, f.key), (model.__name__, f.key)
    for key in ("view", "damping"):
        spec = next(f for f in fields_of(m.World) if f.key == key)
        assert field_tip(m.World, spec.name, key)


def test_property_panel_shows_move_and_balance_forms(qtbot: Any) -> None:
    text = open("tests/data/presets/running_measured.evoenv", encoding="utf-8").read()
    data = json.loads(text)
    data["actuators"].append({"kind": "move", "id": "push", "body": "torso", "max_force": 50.0, "max_speed": 2.0,
                              "keys_left": ["Left"], "keys_right": ["Right"], "mode": "x"})
    doc = EnvDocument()
    doc.load_text(json.dumps(data, ensure_ascii=False), None)
    panel = PropertyPanel(doc)
    qtbot.addWidget(panel)
    panel.show()
    for aid, keys in (("push", ("mode", "max_force", "max_speed", "max_turn", "keys_left")),
                      ("bal", ("stiffness", "damping", "max_torque", "target"))):
        doc.select([("actuators", aid)])
        for key in keys:
            fw = panel.form.field(key)
            assert fw is not None, (aid, key)
            assert fw.tooltip(), (aid, key)
            assert fw.label is not None and fw.label.text()

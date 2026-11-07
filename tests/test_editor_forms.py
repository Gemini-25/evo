"""M5a: 속성 패널, 폼 엔진, 아래쪽 탭(편집은 모두 문서를 거치고 되돌리기 한 단계로 합쳐지는지)."""

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox, QLineEdit, QWidget

from evolab.app.editor.document import EnvDocument
from evolab.app.editor.forms import (
    IdField,
    LiteralField,
    NumEdit,
    NumField,
    RefField,
    UnionKindField,
    UnionListField,
    VecField,
    convert_item,
    field_name,
    issues_at,
    ref_candidates,
    rename,
    type_info,
)
from evolab.app.editor.properties import PropertyPanel
from evolab.app.editor.tabs import BottomTabs
from evolab.spec import model as m
from evolab.spec.io import compile_env, parse_env


@pytest.fixture
def lander(qtbot: Any) -> EnvDocument:
    doc = EnvDocument()
    doc.from_builtin("lander.evoenv")
    return doc


@pytest.fixture
def car(qtbot: Any) -> EnvDocument:
    doc = EnvDocument()
    doc.from_builtin("track_car.evoenv")
    return doc


def _panel(qtbot: Any, doc: EnvDocument) -> PropertyPanel:
    panel = PropertyPanel(doc)
    panel.resize(380, 700)
    qtbot.addWidget(panel)
    panel.show()
    return panel


def _tabs(qtbot: Any, doc: EnvDocument) -> BottomTabs:
    tabs = BottomTabs(doc)
    tabs.resize(1300, 420)
    qtbot.addWidget(tabs)
    tabs.show()
    return tabs


def _focus(qtbot: Any, w: QWidget) -> None:
    win = w.window()
    win.show()
    win.activateWindow()
    w.setFocus()
    qtbot.waitUntil(w.hasFocus, timeout=2000)


def _type(qtbot: Any, edit: QLineEdit, text: str, finish: bool = True) -> None:
    """칸을 비우고 사람처럼 타이핑한다(Enter로 끝)."""
    _focus(qtbot, edit)
    edit.selectAll()
    if text:
        QTest.keyClicks(edit, text)
    else:
        QTest.keyClick(edit, Qt.Key.Key_Backspace)
    if finish:
        QTest.keyClick(edit, Qt.Key.Key_Return)


def _choose(combo: QComboBox, data: Any) -> None:
    """사람이 고른 것처럼(activated 신호)."""
    idx = combo.findData(data)
    assert idx >= 0, data
    combo.setCurrentIndex(idx)
    combo.activated.emit(idx)


def _hull(doc: EnvDocument) -> int:
    i = doc.find(("bodies", "hull"))
    assert i is not None
    return i


# ---------------------------------------------------------------------------
# 속성 패널과 폼 엔진
# ---------------------------------------------------------------------------


def test_panel_hint_selection_and_header(qtbot: Any, lander: EnvDocument) -> None:
    panel = _panel(qtbot, lander)
    assert panel.ref is None and panel.hint.isVisible() and not panel.scroll.isVisible()
    lander.select([("bodies", "hull"), ("bodies", "leg_l")])
    assert panel.ref == ("bodies", "hull") and panel.note.isVisible()
    assert panel.id_field is not None and panel.id_field.editor.text() == "hull"  # type: ignore[union-attr]
    assert panel.label_field is not None and panel.label_field.text() == "본체"
    assert list(panel.form.rows.fields)[:2] == ["owner", "type"]  # type: ignore[union-attr]
    lander.select([("actuators", "main")])
    assert panel.title.text().endswith(S_kind("thruster"))
    panel.delete_button.click()
    assert lander.find(("actuators", "main")) is None and panel.ref is None and panel.hint.isVisible()


def S_kind(kind: str) -> str:  # noqa: N802
    from evolab.app.editor.forms import kind_name

    return kind_name(kind)


def test_number_typing_merges_into_one_undo_step(qtbot: Any, lander: EnvDocument) -> None:
    panel = _panel(qtbot, lander)
    lander.select([("bodies", "hull")])
    field = panel.form.field("material", "density")
    assert isinstance(field, NumField) and isinstance(field.editor, NumEdit)
    before = lander.undo_stack.count()
    rebuilds = panel.form.rebuilds
    _type(qtbot, field.editor, "75.5", finish=False)
    assert field.editor.hasFocus() and field.editor.text() == "75.5"  # 타이핑 중인 칸은 그대로
    QTest.keyClick(field.editor, Qt.Key.Key_Return)
    i = _hull(lander)
    assert lander.get(("bodies", i, "material", "density")) == 75.5
    assert lander.undo_stack.count() == before + 1  # 7, 75, 75., 75.5 -> 한 단계
    assert panel.form.rebuilds == rebuilds  # 값만 바뀌면 위젯을 다시 만들지 않는다
    lander.undo_stack.undo()
    assert lander.get(("bodies", i, "material", "density")) == 50.0
    assert field.editor.text() == "50.0"  # 되돌리기는 포커스가 있어도 보인다


def test_formula_and_invalid_number(qtbot: Any, lander: EnvDocument) -> None:
    panel = _panel(qtbot, lander)
    lander.select([("bodies", "hull")])
    i = _hull(lander)
    h = panel.form.field("shapes", 0, "h")
    assert isinstance(h, NumField)
    _type(qtbot, h.editor, "=p.leg_len * 2")  # type: ignore[arg-type]
    assert lander.get(("bodies", i, "shapes", 0, "h")) == "=p.leg_len * 2"
    assert compile_env(lander.data)
    _type(qtbot, h.editor, "abc")  # type: ignore[arg-type]
    assert h.editor.property("invalid") and h.invalid  # type: ignore[union-attr]
    assert lander.get(("bodies", i, "shapes", 0, "h")) == "=p.leg_len * 2"  # 틀린 값은 쓰지 않는다
    # 수식을 못 쓰는 칸(dt)
    tabs = _tabs(qtbot, lander)
    dt = tabs.world_tab.world_form.field("dt")
    assert isinstance(dt, NumField)
    _type(qtbot, dt.editor, "=1/60")  # type: ignore[arg-type]
    assert dt.invalid and lander.get(("world", "dt")) == pytest.approx(1 / 60)


def test_vec_literal_bool_and_optional_vec(qtbot: Any, lander: EnvDocument) -> None:
    panel = _panel(qtbot, lander)
    lander.select([("joints", "j_leg_l")])
    j = lander.find(("joints", "j_leg_l"))
    anchor = panel.form.field("anchor_a")
    assert isinstance(anchor, VecField)
    _type(qtbot, anchor.parts[0].editor, "-0.9")  # type: ignore[arg-type]
    assert lander.get(("joints", j, "anchor_a")) == [-0.9, -0.5]
    collide = panel.form.field("collide")
    assert collide is not None
    collide.editor.click()  # type: ignore[union-attr]
    assert lander.get(("joints", j, "collide")) is True
    kind = panel.form.field("kind")
    assert isinstance(kind, LiteralField)
    _choose(kind.editor, "pivot")  # type: ignore[arg-type]
    assert lander.get(("joints", j, "kind")) == "pivot"
    limit = panel.form.field("limit")
    assert isinstance(limit, VecField) and limit.use_box is not None and limit.use_box.isChecked()
    limit.use_box.click()
    assert lander.get(("joints", j, "limit")) is None
    limit.use_box.click()
    assert lander.get(("joints", j, "limit")) == [-0.5, 0.5]
    # 고르기 칸 이름은 한국어
    lander.select([("bodies", "hull")])
    owner = panel.form.field("owner")
    assert isinstance(owner, LiteralField) and owner.editor.itemText(0) == "에이전트"  # type: ignore[union-attr]
    assert field_name(m.Body, "owner") == "소속" and field_name(m.Body, "nonexistent") == "nonexistent"


def test_ref_combo_candidates_and_commit(qtbot: Any, lander: EnvDocument) -> None:
    panel = _panel(qtbot, lander)
    lander.select([("sensors", "pad_rel")])
    s = lander.find(("sensors", "pad_rel"))
    body = panel.form.field("body")
    target = panel.form.field("target")
    assert isinstance(body, RefField) and isinstance(target, RefField)
    combo = body.editor
    assert isinstance(combo, QComboBox)
    items = [combo.itemText(k) for k in range(combo.count())]
    assert items[:3] == ["hull", "leg_l", "leg_r"] and "terrain" in items
    assert "terrain.pad" in ref_candidates(lander.data, "target")
    _type(qtbot, combo.lineEdit(), "leg_l")
    assert lander.get(("sensors", s, "body")) == "leg_l"
    _type(qtbot, combo.lineEdit(), "bad ref!")
    assert body.invalid and lander.get(("sensors", s, "body")) == "leg_l"
    hits = ref_candidates(lander.data, "hit")
    assert {"tag.terrain", "tag.pad", "tag.hull", "tag.leg"} <= set(hits)
    assert ref_candidates(lander.data, "event")[-2:] == ["timeout", "out_of_bounds"]


def test_lists_dict_and_literal_list(qtbot: Any, lander: EnvDocument) -> None:
    panel = _panel(qtbot, lander)
    lander.select([("actuators", "main")])
    a = lander.find(("actuators", "main"))
    keys = panel.form.field("keys")
    assert keys is not None
    _type(qtbot, keys.editor, "Up, W, K")  # type: ignore[arg-type]
    assert lander.get(("actuators", a, "keys")) == ["Up", "W", "K"]
    lander.select([("bodies", "hull")])
    tags = panel.form.field("tags")
    assert tags is not None
    _type(qtbot, tags.editor, "hull, 2bad")  # type: ignore[arg-type]
    assert tags.invalid and lander.get(("bodies", _hull(lander), "tags")) == ["hull"]
    # 몸 상태 센서: 항목 체크박스와 정규화 크기 표
    lander.select([("sensors", "st")])
    s = lander.find(("sensors", "st"))
    fields = panel.form.field("fields")
    fields.boxes["speed"].click()  # type: ignore[union-attr]
    assert lander.get(("sensors", s, "fields")) == ["vx", "vy", "angle", "omega", "speed"]
    fields.boxes["vx"].click()  # type: ignore[union-attr]
    assert lander.get(("sensors", s, "fields")) == ["vy", "angle", "omega", "speed"]
    scale = panel.form.field("scale")
    assert scale is not None
    _choose(scale.add_combo, "speed")  # type: ignore[attr-defined]
    scale.add_button.click()  # type: ignore[attr-defined]
    assert lander.get(("sensors", s, "scale", "speed")) == 10.0
    _type(qtbot, scale.edits["speed"], "4")  # type: ignore[attr-defined]
    assert lander.get(("sensors", s, "scale", "speed")) == 4.0
    assert parse_env(lander.data)


def test_evolve_checkbox_creates_and_removes_param(qtbot: Any, lander: EnvDocument) -> None:
    panel = _panel(qtbot, lander)
    lander.select([("bodies", "hull")])
    i = _hull(lander)
    w = panel.form.field("shapes", 0, "w")
    assert isinstance(w, NumField) and w.evolve_box is not None and w.evolve_box.isEnabled()
    w.evolve_box.click()
    assert lander.get(("bodies", i, "shapes", 0, "w")) == "=p.hull_w"
    param = lander.item(("params", "hull_w"))
    assert param is not None and param["evolve"] and (param["min"], param["max"], param["value"]) == (1.0, 3.0, 2.0)
    assert param["label"] == "본체 너비"
    spec = compile_env(lander.data)
    assert spec.body_param_count == 1
    assert w.evolving == "hull_w" and w.editor.text() == "2.0" and w._range is not None and w._range.isVisible()
    _type(qtbot, w.max_edit, "4")  # type: ignore[arg-type]
    _type(qtbot, w.editor, "2.5")  # type: ignore[arg-type]  # 진화 중에는 초깃값을 고친다
    param = lander.item(("params", "hull_w"))
    assert param is not None and param["max"] == 4.0 and param["value"] == 2.5
    w.evolve_box.click()
    assert lander.get(("bodies", i, "shapes", 0, "w")) == 2.5
    assert lander.item(("params", "hull_w")) is None
    assert compile_env(lander.data).body_param_count == 0
    # 세계 물체·세계 설정에는 진화 체크박스가 없다
    tabs = _tabs(qtbot, lander)
    gx = tabs.world_tab.world_form.field("gravity")
    assert isinstance(gx, VecField) and gx.parts[0].evolve_box is None


def test_evolve_linked_param_and_formula(qtbot: Any, lander: EnvDocument) -> None:
    panel = _panel(qtbot, lander)
    lander.select([("bodies", "leg_l")])
    i = lander.find(("bodies", "leg_l"))
    h = panel.form.field("shapes", 0, "h")  # "=p.leg_len" (다리 둘과 관절이 함께 쓴다)
    assert isinstance(h, NumField) and h.evolve_box is not None and h.evolve_box.isEnabled()
    h.evolve_box.click()
    assert lander.item(("params", "leg_len"))["evolve"] is True  # type: ignore[index]
    assert lander.get(("bodies", i, "shapes", 0, "h")) == "=p.leg_len"
    assert compile_env(lander.data).body_param_count == 1
    h.evolve_box.click()  # 다른 칸도 쓰므로 매개변수는 두고 진화만 끈다
    assert lander.item(("params", "leg_len"))["evolve"] is False  # type: ignore[index]
    assert lander.get(("bodies", i, "shapes", 0, "h")) == "=p.leg_len"
    off = panel.form.field("shapes", 1, "offset")
    assert isinstance(off, VecField) and off.parts[1].evolve_box is not None
    assert not off.parts[1].evolve_box.isEnabled()  # 일반 수식 칸은 진화 대상으로 못 바꾼다


def test_kind_change_converts_item(qtbot: Any, lander: EnvDocument) -> None:
    panel = _panel(qtbot, lander)
    lander.select([("sensors", "st")])
    s = lander.find(("sensors", "st"))
    kind = panel.form.field("kind")
    assert isinstance(kind, UnionKindField)
    before = lander.undo_stack.count()
    rebuilds = panel.form.rebuilds
    _choose(kind.editor, "rays")  # type: ignore[arg-type]
    item = lander.get(("sensors", s))
    assert item["kind"] == "rays" and item["id"] == "st" and item["label"] == "몸 상태" and item["body"] == "hull"
    assert "fields" not in item and "scale" not in item and item["hits"] and item["count"] == 5
    assert lander.undo_stack.count() == before + 1 and panel.form.rebuilds == rebuilds + 1
    assert set(panel.form.rows.fields) >= {"origin", "count", "spread", "length", "hits"}  # type: ignore[union-attr]
    assert compile_env(lander.data)
    # 모양: 사각형 -> 원 (offset은 둔다)
    lander.select([("bodies", "hull")])
    i = _hull(lander)
    shape_kind = panel.form.field("shapes", 0, "kind")
    assert isinstance(shape_kind, UnionKindField)
    _choose(shape_kind.editor, "circle")  # type: ignore[arg-type]
    assert lander.get(("bodies", i, "shapes", 0)) == {"kind": "circle", "r": 0.5, "offset": [0.0, 0.0]}
    # 변환 함수: 공통 칸은 남고 맞지 않는 값은 기본값으로
    old = {"kind": "target", "id": "t", "label": "", "body": "hull", "target": "terrain.pad", "scale": 20.0}
    new = convert_item(old, type_info(m.Sensor), "body_state", lander.data)
    assert new["body"] == "hull" and new["fields"] == ["vx", "vy"] and "scale" not in new and "target" not in new


def test_shapes_add_remove(qtbot: Any, lander: EnvDocument) -> None:
    panel = _panel(qtbot, lander)
    lander.select([("bodies", "hull")])
    i = _hull(lander)
    shapes = panel.form.rows.fields["shapes"]  # type: ignore[union-attr]
    assert isinstance(shapes, UnionListField) and not shapes.remove_buttons[0].isEnabled()  # 최소 1개
    action = next(a for a in shapes.add_button.menu().actions() if a.text() == S_kind("poly"))
    action.trigger()
    assert [s["kind"] for s in lander.get(("bodies", i, "shapes"))] == ["box", "poly"]
    shapes = panel.form.rows.fields["shapes"]  # type: ignore[union-attr]  # 구조가 바뀌어 다시 만들었다
    points = panel.form.field("shapes", 1, "points")
    assert points is not None and points.table.rowCount() == 3  # type: ignore[attr-defined]
    points.table.item(0, 0).setText("-0.7")  # type: ignore[attr-defined]
    assert lander.get(("bodies", i, "shapes", 1, "points"))[0] == [-0.7, -0.5]
    assert compile_env(lander.data)
    shapes.remove_buttons[1].click()  # type: ignore[attr-defined]
    assert len(lander.get(("bodies", i, "shapes"))) == 1


def test_rename_validation_and_references(qtbot: Any, lander: EnvDocument) -> None:
    panel = _panel(qtbot, lander)
    lander.select([("bodies", "hull")])
    idf = panel.id_field
    assert isinstance(idf, IdField)
    before = lander.undo_stack.count()
    _type(qtbot, idf.editor, "leg_l", finish=False)  # type: ignore[arg-type]
    assert idf.invalid and idf.editor.property("invalid")  # type: ignore[union-attr]
    QTest.keyClick(idf.editor, Qt.Key.Key_Return)
    assert lander.undo_stack.count() == before and idf.last_error and idf.editor.text() == "hull"  # type: ignore[union-attr]
    _type(qtbot, idf.editor, "9x")  # type: ignore[arg-type]
    assert lander.find(("bodies", "hull")) is not None
    _type(qtbot, idf.editor, "body")  # type: ignore[arg-type]
    assert lander.undo_stack.count() == before + 1
    assert panel.ref == ("bodies", "body") and lander.selection == [("bodies", "body")]
    spec = compile_env(lander.data)
    assert any("contact(body, tag.terrain)" in e.when for e in spec.file.events)
    assert spec.file.sensors[0].body == "body"


def test_rename_params_generators_and_channels(qtbot: Any, lander: EnvDocument) -> None:
    assert rename(lander, ("params", "leg_len"), "leg")
    assert lander.get(("bodies", lander.find(("bodies", "leg_l")), "shapes", 0, "h")) == "=p.leg"
    assert rename(lander, ("randomize", "pad_x"), "px")
    assert rename(lander, ("generators", "terrain"), "ground")
    assert lander.get(("agent", "spawn", "x")) == "=gen.ground.pad.x + r.dx"
    assert lander.item(("sensors", "pad_rel"))["target"] == "ground.pad"  # type: ignore[index]
    lander.set_value(("brain", "inputs"), ["pad_rel.x", "st.vx", "fuel"])
    assert rename(lander, ("sensors", "st"), "state")
    assert lander.get(("brain", "inputs")) == ["pad_rel.x", "state.vx", "fuel"]
    assert lander.item(("bodies", "hull"))["tags"] == ["hull"]  # type: ignore[index]  # 태그는 그대로
    assert compile_env(lander.data)


# ---------------------------------------------------------------------------
# 아래쪽 탭
# ---------------------------------------------------------------------------


def test_world_tab(qtbot: Any, lander: EnvDocument) -> None:
    tabs = _tabs(qtbot, lander)
    gravity = tabs.world_tab.world_form.field("gravity")
    assert isinstance(gravity, VecField)
    _type(qtbot, gravity.parts[1].editor, "-3")  # type: ignore[arg-type]
    assert lander.get(("world", "gravity")) == [0.0, -3.0]
    sx = tabs.world_tab.spawn_form.field("x")
    assert sx is not None and sx.editor.text() == "=gen.terrain.pad.x + r.dx"  # type: ignore[union-attr]
    name = tabs.world_tab.env_form.field("name")
    assert name is not None and isinstance(name.editor, QLineEdit)
    _focus(qtbot, name.editor)
    name.editor.setText("새 착륙선")  # QTest.keyClicks는 한글을 못 보낸다: 사람이 친 것처럼 신호만
    name.editor.textEdited.emit("새 착륙선")
    assert lander.get(("name",)) == "새 착륙선"


def test_events_and_termination(qtbot: Any, lander: EnvDocument) -> None:
    tabs = _tabs(qtbot, lander)
    tabs.setCurrentWidget(tabs.events_tab)
    events = tabs.events_tab.events
    events.add_row()
    assert [e["id"] for e in lander.items("events")] == ["landed", "crash", "off_pad", "event"]
    assert lander.selection == [("events", "event")] and events.current_row() == 3
    when = events.cell(3, "when")
    assert when is not None
    _type(qtbot, when.editor, "time > 5 and hull.speed < 1")  # type: ignore[arg-type]
    assert lander.items("events")[3]["when"] == "time > 5 and hull.speed < 1"
    # 수식 오류는 실시간 검사 결과로 그 칸에 보인다
    _type(qtbot, when.editor, "hull.nope > 1")  # type: ignore[arg-type]
    lander.run_check()
    assert when.issue_messages and issues_at(lander.issues, ("events", 3, "when"))
    events.move_row(-1)
    assert [e["id"] for e in lander.items("events")] == ["landed", "crash", "event", "off_pad"]
    events.remove_row()
    assert [e["id"] for e in lander.items("events")] == ["landed", "crash", "off_pad"]
    # 종료 조건: 더하기, 순서(우선순위), 지우기
    term = tabs.events_tab.termination
    n = len(lander.items("termination"))
    term.add_row()
    assert len(lander.items("termination")) == n + 1
    term.set_current_row(n)
    term.move_row(-1)
    assert lander.items("termination")[n - 1]["outcome"] == "failure"
    assert lander.items("termination")[n]["event"] == "timeout"
    event_cell = term.cell(0, "event")
    assert isinstance(event_cell, RefField)
    combo = event_cell.editor
    assert isinstance(combo, QComboBox) and combo.findText("timeout") >= 0 and combo.findText("out_of_bounds") >= 0
    _type(qtbot, combo.lineEdit(), "crash")
    assert lander.items("termination")[0]["event"] == "crash"
    outcome = term.cell(0, "outcome")
    _choose(outcome.editor, "neutral")  # type: ignore[union-attr, arg-type]
    assert lander.items("termination")[0]["outcome"] == "neutral"
    term.set_current_row(0)
    term.remove_row()
    assert len(lander.items("termination")) == n
    assert compile_env(lander.data)


def test_score_blocks_and_formula(qtbot: Any, lander: EnvDocument) -> None:
    tabs = _tabs(qtbot, lander)
    tabs.setCurrentWidget(tabs.scoring_tab)
    blocks = tabs.scoring_tab.blocks
    blocks.add_row("survival_time")
    added = lander.items("scoring.blocks")[-1]
    assert added["kind"] == "survival_time" and added["id"] == "s_survival_time"
    assert lander.selection == [("scoring.blocks", "s_survival_time")]
    weight = blocks.cell(len(lander.items("scoring.blocks")) - 1, "weight")
    _type(qtbot, weight.editor, "0.5")  # type: ignore[union-attr, arg-type]
    assert lander.items("scoring.blocks")[-1]["weight"] == 0.5
    # 고른 블록의 종류별 칸(옆 폼)
    blocks.set_current_row(2)  # s_dist
    detail = tabs.scoring_tab.detail
    assert detail.path == ("scoring", "blocks", 2) and set(detail.rows.fields) == {"body", "target", "agg"}  # type: ignore[union-attr]
    agg = detail.field("agg")
    _choose(agg.editor, "min")  # type: ignore[union-attr, arg-type]
    assert lander.items("scoring.blocks")[2]["agg"] == "min"
    # 표의 종류 칸으로 바꾸기(물체는 남는다)
    kind = blocks.cell(3, "kind")
    _choose(kind.editor, "tilt")  # type: ignore[union-attr, arg-type]
    assert lander.items("scoring.blocks")[3] == {"id": "s_speed", "label": "종료 속도", "weight": -5.0, "kind": "tilt",
                                                 "body": "hull"}
    # 점수 수식(비우면 없음)
    formula = tabs.scoring_tab.formula_form.field("formula")
    _type(qtbot, formula.editor, "s_success + s_crash")  # type: ignore[union-attr, arg-type]
    assert lander.get(("scoring", "formula")) == "s_success + s_crash"
    _type(qtbot, formula.editor, "")  # type: ignore[union-attr, arg-type]
    assert lander.get(("scoring", "formula")) is None
    assert compile_env(lander.data)


def test_randomize_and_params_rows(qtbot: Any, lander: EnvDocument) -> None:
    tabs = _tabs(qtbot, lander)
    tabs.setCurrentWidget(tabs.random_tab)
    table = tabs.random_tab.table
    table.add_row()
    row = len(lander.items("randomize")) - 1
    assert lander.items("randomize")[row] == {"id": "rand", "label": "", "dist": "uniform", "low": 0.0, "high": 1.0}
    _type(qtbot, table.cell(row, "high").editor, "3")  # type: ignore[union-attr, arg-type]
    _choose(table.cell(row, "dist").editor, "choice")  # type: ignore[union-attr, arg-type]
    _type(qtbot, table.cell(row, "values").editor, "1, 2.5, 3")  # type: ignore[union-attr, arg-type]
    assert lander.items("randomize")[row] == {"id": "rand", "label": "", "dist": "choice", "low": 0.0, "high": 3.0,
                                              "values": [1.0, 2.5, 3.0]}
    assert compile_env(lander.data)
    # 매개변수 표: 진화 켜기
    tabs.setCurrentWidget(tabs.params_tab)
    params = tabs.params_tab.table
    params.cell(0, "evolve").editor.click()  # type: ignore[union-attr]
    assert lander.items("params")[0]["evolve"] is True
    assert compile_env(lander.data).body_param_count == 1


def test_brain_tab_order_and_parameter_count(qtbot: Any, lander: EnvDocument, car: EnvDocument) -> None:
    tabs = _tabs(qtbot, lander)
    brain = tabs.brain_tab
    tabs.setCurrentWidget(brain)
    assert brain.param_count == 211 and brain.genome_dim == 211
    inputs = brain.inputs
    assert inputs.all_box.isChecked() and inputs.list.count() == 9
    inputs.all_box.click()
    names = ["pad_rel.x", "pad_rel.y", "st.vx", "st.vy", "st.angle", "st.omega", "c_l", "c_r", "fuel"]
    assert lander.get(("brain", "inputs")) == names
    inputs.list.item(6).setCheckState(Qt.CheckState.Unchecked)  # c_l 빼기
    assert lander.get(("brain", "inputs")) == [n for n in names if n != "c_l"]
    inputs.list.setCurrentRow(2)  # st.vx를 위로
    inputs.move(-1)
    assert lander.get(("brain", "inputs"))[:3] == ["pad_rel.x", "st.vx", "pad_rel.y"]
    lander.run_check()
    assert brain.param_count == 8 * 16 + 16 + 16 * 3 + 3
    hidden = brain.form.field("hidden")
    _type(qtbot, hidden.editor, "8, 4")  # type: ignore[union-attr, arg-type]
    assert lander.get(("brain", "hidden")) == [8, 4]
    inputs.all_box.click()
    assert lander.get(("brain", "inputs")) == "all"
    lander.set_value(("brain", "hidden"), [16])
    lander.run_check()
    assert brain.param_count == 211
    car_tabs = _tabs(qtbot, car)
    assert car_tabs.brain_tab.param_count == 194
    assert car_tabs.brain_tab.outputs.channels() == ["drive.throttle", "drive.steer"]


def test_generators_and_evaluation(qtbot: Any, car: EnvDocument, lander: EnvDocument) -> None:
    tabs = _tabs(qtbot, car)
    tabs.setCurrentWidget(tabs.generators_tab)
    detail = tabs.generators_tab.detail
    assert detail.path == ("generators", 0) and "set_size" in detail.rows.fields  # type: ignore[union-attr]
    spin = detail.field("set_size").editor  # type: ignore[union-attr]
    spin.setValue(7)  # type: ignore[union-attr]
    assert car.get(("generators", 0, "set_size")) == 7
    tabs.setCurrentWidget(tabs.eval_tab)
    trials = tabs.eval_tab.form.field("trials")
    assert not trials.editor.isEnabled() and tabs.eval_tab.note.isVisible()  # type: ignore[union-attr]
    assert "7" in tabs.eval_tab.note.text()
    lt = _tabs(qtbot, lander)
    lt.setCurrentWidget(lt.eval_tab)
    assert lt.eval_tab.form.field("trials").editor.isEnabled() and not lt.eval_tab.note.isVisible()  # type: ignore[union-attr]
    lt.generators_tab.table.add_row("track")
    assert lander.items("generators")[-1] == {"kind": "track", "id": "track", "label": S_kind("track")}
    lt.generators_tab.table.remove_row()
    assert len(lander.items("generators")) == 1


def test_issue_list_double_click_selects_component(qtbot: Any, lander: EnvDocument) -> None:
    panel = _panel(qtbot, lander)
    tabs = _tabs(qtbot, lander)
    s = lander.find(("sensors", "pad_rel"))
    lander.set_value(("sensors", s, "body"), "nope")
    lander.run_check()
    tree = tabs.issues_tab.tree
    assert tree.topLevelItemCount() >= 1 and "(" in tabs.tabText(tabs.indexOf(tabs.issues_tab))
    item = next(tree.topLevelItem(k) for k in range(tree.topLevelItemCount())
                if tree.topLevelItem(k).data(0, Qt.ItemDataRole.UserRole) == f"sensors[{s}].body")
    got: list[tuple[Any, str]] = []
    tabs.issue_activated.connect(lambda ref, where: got.append((ref, where)))
    tree.itemDoubleClicked.emit(item, 0)
    assert lander.selection == [("sensors", "pad_rel")]
    assert got == [(("sensors", "pad_rel"), f"sensors[{s}].body")]
    body = panel.form.field("body")
    assert body is not None and body.issue_messages and body.label is not None
    assert body.label.styleSheet()  # 빨간 이름표
    assert panel.issue_box.isVisible() and panel.issue_list.count() >= 1
    # 부품이 아닌 곳(뇌 설정)은 그 탭으로 옮긴다
    lander.set_value(("sensors", s, "body"), "hull")
    lander.set_value(("brain", "inputs"), ["nope"])
    lander.run_check()
    tabs.issues_tab.activate("brain.inputs[0]")
    assert tabs.currentWidget() is tabs.brain_tab and got[-1] == (None, "brain.inputs[0]")


def test_undo_while_panel_open_updates_in_place(qtbot: Any, lander: EnvDocument) -> None:
    panel = _panel(qtbot, lander)
    lander.select([("actuators", "main")])
    a = lander.find(("actuators", "main"))
    force = panel.form.field("max_force")
    rebuilds = panel.form.rebuilds
    lander.set_value(("actuators", a, "max_force"), 123.0)
    assert force.editor.text() == "123.0" and panel.form.rebuilds == rebuilds  # type: ignore[union-attr]
    lander.undo_stack.undo()
    assert force.editor.text() == "350.0"  # type: ignore[union-attr]
    lander.remove([("actuators", "main")])
    assert panel.ref is None
    lander.undo_stack.undo()  # 되돌려도 선택은 비어 있다(문서 규칙)
    assert lander.find(("actuators", "main")) is not None

"""v2.1 에디터 (설계 5.1·7.3, 8.6 T70~T73): 속성 패널의 「있음」 칸과 진화 체크, 「배열로 늘리기」(tools.replicate),
아래 「구조」 탭(함께 사라지는 부품 수, 줄 고르기 강조), 캔버스의 점선·「?」 표지와 「없는 부품도 보기」, 물체 오른쪽 메뉴.

T63(언덕 넘는 자동차에서 바퀴 고르는 언덕차 만들기)은 tests/test_editor_rebuild.py에 있다.
"""

from __future__ import annotations

import copy
import json
import math
import time
from typing import Any

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtWidgets import QApplication, QDialog

from evolab.app import strings as S
from evolab.app.editor import tools as T
from evolab.app.editor.canvas import EditorCanvas, build_scene
from evolab.app.editor.document import EnvDocument
from evolab.app.editor.forms import EXISTS_OFF, EXISTS_ON, ExistsField, NumField
from evolab.app.editor.properties import PropertyPanel
from evolab.app.editor.structure import (
    COL_EVOLVE,
    COL_INITIAL,
    COL_PARTS,
    ReplicateDialog,
    StructureTab,
    attached_parts,
    initial_absent,
    structure_rows,
)
from evolab.app.editor.view import EditorView, builtin_menu_items
from evolab.spec import structure
from evolab.spec.io import compile_env, dump_env, parse_env

# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------


def leg_env() -> dict[str, Any]:
    """몸통 + 허벅지(있음 진화) + 정강이(무릎 관절로 놓임, place_by) + 세계 바닥. 늘리기·구조 탭 시험용."""
    return {
        "version": 3, "name": "leg",
        "world": {"kind": "physics2d", "gravity": [0.0, -9.81], "dt": 1 / 60, "max_time": 10.0,
                  "bounds": [-20.0, -20.0, 20.0, 20.0]},
        "params": [{"id": "thigh_on", "label": "", "value": 0.75, "evolve": True, "min": 0.0, "max": 1.0}],
        "agent": {"spawn": {"x": 2.0, "y": 3.0}},
        "bodies": [
            {"id": "torso", "label": "몸통", "owner": "agent", "shapes": [{"kind": "box", "w": 2.0, "h": 0.5}]},
            {"id": "thigh", "label": "허벅지", "owner": "agent", "pos": [0.8, -0.6], "angle": 0.2, "exists": "=p.thigh_on",
             "shapes": [{"kind": "poly", "points": [[0.0, 0.3], [0.1, -0.3], [-0.1, -0.3]]}]},
            {"id": "shin", "label": "정강이", "owner": "agent", "place_by": "knee",
             "shapes": [{"kind": "box", "w": 0.15, "h": 0.6, "offset": [0.05, 0.0]}]},
            {"id": "ground", "owner": "world", "type": "static", "pos": [0.0, -3.0],
             "shapes": [{"kind": "box", "w": 40.0, "h": 1.0}], "tags": ["ground"]},
        ],
        "joints": [
            {"id": "hip", "label": "허벅지 관절", "kind": "pivot", "a": "torso", "b": "thigh", "anchor_a": [0.8, -0.3],
             "anchor_b": [0.0, 0.3], "rest_angle": 0.2, "limit": [-0.5, 1.0]},
            {"id": "knee", "label": "무릎", "kind": "pivot", "a": "thigh", "b": "shin", "anchor_a": [0.0, -0.35],
             "anchor_b": [0.0, 0.35], "rest_angle": -0.3},
        ],
        "actuators": [
            {"kind": "motor", "id": "m_hip", "joint": "hip", "max_torque": 50.0, "max_speed": 5.0},
            {"kind": "motor", "id": "m_knee", "joint": "knee", "max_torque": 30.0, "max_speed": 5.0},
            {"kind": "thruster", "id": "jet", "body": "shin", "at": [0.1, -0.2], "dir": 1.2, "max_force": 10.0},
        ],
        "sensors": [
            {"kind": "body_state", "id": "st", "body": "torso", "fields": ["vx"]},
            {"kind": "contact", "id": "foot", "body": "shin", "with": ["tag.ground"]},
            {"kind": "joint_angle", "id": "knee_a", "joint": "knee"},
            {"kind": "rays", "id": "eye", "body": "thigh", "origin": [0.1, 0.0], "count": 3, "spread": 1.0,
             "center": 0.3, "length": 5.0, "hits": ["tag.ground"]},
        ],
        "termination": [{"event": "timeout", "outcome": "neutral", "label": "끝"}],
        "scoring": {"blocks": [{"kind": "survival_time", "id": "s", "weight": 1.0}]},
        "brain": {"inputs": ["st.vx", "foot", "knee_a", "eye.r0", "eye.r1", "eye.r2"], "hidden": [4],
                  "outputs": ["m_hip", "m_knee", "jet"]},
    }


def leg_doc() -> EnvDocument:
    doc = EnvDocument()
    doc.load_text(json.dumps(leg_env(), ensure_ascii=False))
    return doc


def one_wheel_doc() -> EnvDocument:
    """언덕 넘는 자동차에서 앞바퀴(와 거기 붙은 것)를 지운 문서."""
    doc = EnvDocument()
    doc.from_builtin("hill_car.evoenv")
    doc.remove([("bodies", "wheel_f"), ("joints", "j_f"), ("actuators", "m_f"), ("sensors", "c_f"),
                ("params", "front_r")])
    doc.set_value(("agent", "spawn", "y"), "=p.rear_r + 0.6")
    return doc


def wheels_doc() -> EnvDocument:
    doc = EnvDocument()
    doc.from_builtin("wheels.evoenv")
    return doc


def errors(doc: EnvDocument) -> list[Any]:
    return [i for i in doc.run_check() if i.severity == "error"]


def panel_for(qtbot: Any, doc: EnvDocument) -> PropertyPanel:
    panel = PropertyPanel(doc)
    qtbot.addWidget(panel)
    panel.resize(380, 900)
    panel.show()
    return panel


def make_view(qtbot: Any, doc_builtin: str | None = None, data: dict[str, Any] | None = None) -> EditorView:
    view = EditorView()
    qtbot.addWidget(view)
    view.resize(1400, 900)
    view.show()
    if doc_builtin is not None:
        view.doc.from_builtin(doc_builtin)
    elif data is not None:
        view.doc.load_text(json.dumps(data, ensure_ascii=False))
    view.tabs.setCurrentWidget(view.tabs.structure_tab)
    QApplication.processEvents()
    return view


def present_at(doc: EnvDocument, **values: float) -> frozenset[str]:
    spec = compile_env(copy.deepcopy(doc.data))
    params = structure.param_values(spec.file)
    params.update({f"p.{k}": float(v) for k, v in values.items()})
    return structure.present(spec, params)


def close(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b, strict=True))
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# T70: 있음 칸과 진화 체크
# ---------------------------------------------------------------------------


def test_t70_exists_evolve_check_creates_and_removes_the_on_param(qtbot: Any) -> None:
    doc = EnvDocument()
    doc.from_builtin("hill_car.evoenv")
    before = doc.text()
    genes = compile_env(doc.text()).body_param_count
    panel = panel_for(qtbot, doc)
    doc.select([("bodies", "wheel_f")])
    f = panel.form.field("exists")
    assert isinstance(f, ExistsField) and isinstance(f, NumField)
    assert f.evolve_box is not None and f.evolve_box.isEnabled() and not f.evolve_box.isChecked()
    assert f.editor is not None and f.editor.isEnabled() and f.editor.placeholderText() == S.ED_STRUCT_ON
    assert f.label is not None and f.label.text() == S.EDP_FIELD_NAMES["exists"]
    assert panel.form.rows is not None and panel.form.rows.layout_.isRowVisible(f)

    f.evolve_box.click()
    assert doc.item(("params", "wheel_f_on")) == {"id": "wheel_f_on", "label": "앞바퀴 있음", "value": EXISTS_ON,
                                                 "evolve": True, "min": 0.0, "max": 1.0}
    assert EXISTS_ON == 0.75 and EXISTS_OFF == 0.25
    assert doc.item(("bodies", "wheel_f"))["exists"] == "=p.wheel_f_on"  # type: ignore[index]
    assert f.evolve_box.isChecked() and f.evolving == "wheel_f_on"
    assert f._range is not None and not f._range.isVisible()  # 범위는 늘 0..1
    spec = compile_env(doc.text())
    assert spec.body_param_count == genes + 1
    assert structure.optional_bodies(spec.file) == {"wheel_f"}
    assert json.loads(doc.text())["version"] == 3  # 있음 칸을 쓰면 형식 3
    assert not errors(doc)

    f.evolve_box.click()  # 끄면 매개변수가 지워지고 칸이 빈다(처음 있음 0.75 > 0.5)
    assert doc.item(("params", "wheel_f_on")) is None
    assert "exists" not in doc.item(("bodies", "wheel_f"))  # type: ignore[operator]
    assert doc.text() == before  # 형식 1 그대로, 글자까지 같다
    doc.undo_stack.undo()
    assert doc.item(("params", "wheel_f_on")) is not None
    doc.undo_stack.undo()
    assert doc.text() == before


def test_t70_initially_absent_body_starts_at_025_and_keeps_its_state_when_turned_off(qtbot: Any) -> None:
    doc = one_wheel_doc()
    doc.set_value(("bodies", 1, "exists"), 0.2)  # 늘 없음(상수 0.5 이하)
    panel = panel_for(qtbot, doc)
    doc.select([("bodies", "wheel_r")])
    f = panel.form.field("exists")
    assert isinstance(f, ExistsField) and f.evolve_box is not None and f.evolve_box.isEnabled()
    f.evolve_box.click()
    assert doc.item(("params", "wheel_r_on"))["value"] == EXISTS_OFF  # type: ignore[index]
    f.evolve_box.click()
    assert doc.item(("params", "wheel_r_on")) is None
    assert doc.item(("bodies", "wheel_r"))["exists"] == EXISTS_OFF  # type: ignore[index]  # 처음 상태(없음)를 지킨다


def test_t70_param_used_elsewhere_is_kept_and_only_stops_evolving(qtbot: Any) -> None:
    doc = wheels_doc()
    panel = panel_for(qtbot, doc)
    doc.select([("bodies", "wheel2")])
    f = panel.form.field("exists")
    assert isinstance(f, ExistsField) and f.evolve_box is not None and f.evolve_box.isChecked()
    assert f.editor is not None and f.editor.text() == "0.25"  # 진화 중: 몸 수치의 처음 값
    f.evolve_box.click()  # 모터 힘 수식·시작 높이도 p.w2_on을 쓴다
    p = doc.item(("params", "w2_on"))
    assert p is not None and p["evolve"] is False
    assert doc.item(("bodies", "wheel2"))["exists"] == "=p.w2_on"  # type: ignore[index]
    f.evolve_box.click()  # 다시 켜면 같은 매개변수
    assert doc.item(("params", "w2_on"))["evolve"] is True  # type: ignore[index]
    assert [p["id"] for p in doc.items("params")].count("w2_on") == 1


def test_t70_root_is_locked_and_world_bodies_hide_the_row(qtbot: Any) -> None:
    doc = leg_doc()
    panel = panel_for(qtbot, doc)
    doc.select([("bodies", "torso")])
    f = panel.form.field("exists")
    assert isinstance(f, ExistsField) and f.locked
    assert f.editor is not None and not f.editor.isEnabled()
    assert f.editor.placeholderText() == S.ED_STRUCT_ROOT_LOCKED
    assert f.evolve_box is not None and not f.evolve_box.isEnabled()
    assert S.ED_STRUCT_ROOT_LOCKED in f.evolve_box.toolTip() and S.ED_STRUCT_ROOT_LOCKED in f.tooltip()
    # 잘못 적힌 뿌리 있음 값은 검사 오류이고, 칸은 지울 수 있게 열린다
    doc.set_value(("bodies", 0, "exists"), 0.3)
    assert any(i.message == S.SPEC_EXISTS_ROOT.format(id="torso") for i in errors(doc))
    assert f.editor.isEnabled() and not f.evolve_box.isEnabled()
    f.editor.setText("")
    f.editor.textEdited.emit("")
    assert "exists" not in doc.item(("bodies", "torso")) or doc.item(("bodies", "torso"))["exists"] is None  # type: ignore[index,operator]
    # 세계 물체: 줄을 숨긴다(값이 적혀 있으면 지울 수 있게 보인다)
    doc.select([("bodies", "ground")])
    g = panel.form.field("exists")
    assert isinstance(g, ExistsField) and panel.form.rows is not None
    assert not panel.form.rows.layout_.isRowVisible(g)
    doc.set_value(("bodies", 3, "exists"), 1.0)
    panel.form.refresh()
    assert panel.form.rows.layout_.isRowVisible(g)
    assert any(i.message == S.SPEC_EXISTS_WORLD.format(id="ground") for i in errors(doc))


# ---------------------------------------------------------------------------
# T71: 배열로 늘리기
# ---------------------------------------------------------------------------


def test_t71_replicate_one_wheel_to_four_with_one_count(qtbot: Any) -> None:
    doc = one_wheel_doc()
    before = copy.deepcopy(doc.data)
    steps = doc.undo_stack.count()
    new = T.replicate(doc, "wheel_r", 4, {"kind": "line", "end": [1.1, -0.25]}, False, True)
    assert doc.undo_stack.count() == steps + 1  # 되돌리기 한 단계
    ids = {s: [x["id"] for x in doc.items(s)] for s in ("bodies", "joints", "actuators", "sensors")}
    assert ids == {"bodies": ["chassis", "wheel_r", "wheel_r2", "wheel_r3", "wheel_r4"],
                   "joints": ["j_r", "j_r2", "j_r3", "j_r4"], "actuators": ["m_r", "m_r2", "m_r3", "m_r4"],
                   "sensors": ["st", "ray", "c_r", "c_r2", "c_r3", "c_r4"]}
    assert ("params", "wheel_r_count") in new and len([r for r in new if r[0] == "bodies"]) == 3
    for k, (w, j, mo, c) in enumerate(zip(ids["bodies"][1:], ids["joints"], ids["actuators"], ids["sensors"][2:],
                                          strict=True)):
        joint, motor, sensor, body = (doc.item(("joints", j)), doc.item(("actuators", mo)),
                                      doc.item(("sensors", c)), doc.item(("bodies", w)))
        assert joint is not None and joint["a"] == "chassis" and joint["b"] == w
        assert motor is not None and motor["joint"] == j and sensor is not None and sensor["body"] == w
        assert body is not None and body["exists"] == f"=p.wheel_r_count > {k}.5"  # 「개수 하나로」: 수식 4개
        assert body["shapes"][0]["r"] == "=p.rear_r"  # 몸 수치는 같이 쓴다
    assert [doc.item(("bodies", b))["label"] for b in ids["bodies"][1:]] == [  # type: ignore[index]
        "뒷바퀴", "뒷바퀴 2", "뒷바퀴 3", "뒷바퀴 4"]
    assert doc.item(("joints", "j_r3"))["label"] == "뒷바퀴 3 축"  # type: ignore[index]
    assert doc.item(("params", "wheel_r_count")) == {"id": "wheel_r_count", "label": "뒷바퀴 개수", "value": 1.0,
                                                     "evolve": True, "min": 1.0, "max": 4.0}
    assert not errors(doc)
    assert present_at(doc) == {"chassis", "wheel_r"}
    assert present_at(doc, wheel_r_count=3.0) == {"chassis", "wheel_r", "wheel_r2", "wheel_r3"}
    assert present_at(doc, wheel_r_count=4.0) == {"chassis", *ids["bodies"][1:]}
    # 직선: 사본은 원본 자리부터 끝점까지 같은 간격(원본 x 수식 -p.half는 수식으로 남는다)
    scene = build_scene(doc)
    xs = [scene.body(b).pose[0] - scene.spawn[0] for b in ids["bodies"][1:]]  # type: ignore[union-attr]
    assert close(xs, [-1.1, -1.1 + 2.2 / 3, -1.1 + 4.4 / 3, 1.1])
    doc.undo_stack.undo()
    assert doc.data == before
    doc.undo_stack.redo()
    assert len(doc.items("bodies")) == 5


def test_t71_replicate_with_evolve_makes_an_on_param_per_part(qtbot: Any) -> None:
    doc = one_wheel_doc()
    T.replicate(doc, "wheel_r", 3, {"kind": "line", "end": [1.1, -0.25]}, True, False)
    ons = [p for p in doc.items("params") if p["id"].endswith("_on")]
    assert [(p["id"], p["label"], p["value"]) for p in ons] == [
        ("wheel_r_on", "뒷바퀴 있음", 0.75), ("wheel_r2_on", "뒷바퀴 2 있음", 0.25), ("wheel_r3_on", "뒷바퀴 3 있음", 0.25)]
    assert all(p["evolve"] and p["min"] == 0.0 and p["max"] == 1.0 for p in ons)
    assert [b.get("exists") for b in doc.items("bodies")] == [None, "=p.wheel_r_on", "=p.wheel_r2_on", "=p.wheel_r3_on"]
    assert initial_absent(doc) == {"wheel_r2", "wheel_r3"}
    spec = compile_env(doc.text())
    assert structure.optional_bodies(spec.file) == {"wheel_r", "wheel_r2", "wheel_r3"}
    assert not errors(doc)


def test_t71_mirror_copies_the_whole_leg_with_reflected_geometry(qtbot: Any) -> None:
    doc = leg_doc()
    new = T.replicate(doc, "thigh", 2, {"kind": "mirror", "x": 0.0}, False, False)
    # 정강이는 허벅지가 없으면 함께 없어지므로(이어짐 규칙) 같이 복사된다
    assert [r for r in new if r[0] == "bodies"] == [("bodies", "thigh2"), ("bodies", "shin2")]
    thigh2, shin2 = doc.item(("bodies", "thigh2")), doc.item(("bodies", "shin2"))
    hip2, knee2 = doc.item(("joints", "hip2")), doc.item(("joints", "knee2"))
    assert thigh2 is not None and shin2 is not None and hip2 is not None and knee2 is not None
    assert thigh2["pos"] == [-0.8, -0.6] and thigh2["angle"] == -0.2 and thigh2["exists"] == "=p.thigh_on"
    assert thigh2["shapes"][0]["points"] == [[0.1, -0.3], [-0.1, -0.3], [0.0, 0.3]]  # 비추고 순서를 뒤집음
    assert shin2["place_by"] == "knee2" and shin2["shapes"][0]["offset"] == [-0.05, 0.0]
    assert (hip2["a"], hip2["b"], hip2["anchor_a"], hip2["anchor_b"]) == ("torso", "thigh2", [-0.8, -0.3], [0.0, 0.3])
    assert hip2["rest_angle"] == -0.2 and hip2["limit"] == [-1.0, 0.5]
    assert (knee2["a"], knee2["b"], knee2["anchor_a"], knee2["rest_angle"]) == ("thigh2", "shin2", [0.0, -0.35], 0.3)
    jet2 = doc.item(("actuators", "jet2"))
    assert jet2 is not None and jet2["body"] == "shin2" and jet2["at"] == [-0.1, -0.2]
    assert close(jet2["dir"], math.pi - 1.2)
    assert doc.item(("actuators", "m_hip2"))["joint"] == "hip2"  # type: ignore[index]
    assert doc.item(("actuators", "m_knee2"))["joint"] == "knee2"  # type: ignore[index]
    assert doc.item(("sensors", "foot2"))["body"] == "shin2"  # type: ignore[index]
    assert doc.item(("sensors", "knee_a2"))["joint"] == "knee2"  # type: ignore[index]
    eye2 = doc.item(("sensors", "eye2"))
    assert eye2 is not None and eye2["origin"] == [-0.1, 0.0] and close(eye2["center"], math.pi - 0.3)
    assert doc.item(("sensors", "st2")) is None  # 몸통 센서는 복사하지 않는다
    assert doc.get(("brain", "inputs"))[-5:] == ["foot2", "knee_a2", "eye2.r0", "eye2.r1", "eye2.r2"]
    assert doc.get(("brain", "outputs")) == ["m_hip", "m_knee", "jet", "m_hip2", "m_knee2", "jet2"]
    assert not errors(doc)
    # 런타임 자세: 사본은 시작 위치의 세로 축에 비친 자리(관절로 놓인 정강이 포함)
    scene = build_scene(doc)
    sx = scene.spawn[0]
    for a, b in (("thigh", "thigh2"), ("shin", "shin2")):
        pa, pb = scene.body(a).pose, scene.body(b).pose  # type: ignore[union-attr]
        assert close((2 * sx - pa[0], pa[1], -pa[2]), pb), (a, pa, pb)
        va = sorted((round(2 * sx - x, 6), round(y, 6)) for poly in scene.body(a).polys for x, y in poly)  # type: ignore[union-attr]
        vb = sorted((round(x, 6), round(y, 6)) for poly in scene.body(b).polys for x, y in poly)  # type: ignore[union-attr]
        assert close(va, vb, 1e-5)


def test_t71_circle_layout_rotates_copies_and_parent_anchors(qtbot: Any) -> None:
    data = leg_env()
    data["bodies"] = [data["bodies"][0], {"id": "arm", "label": "팔", "owner": "agent", "pos": [1.0, 0.0],
                                          "shapes": [{"kind": "box", "w": 1.0, "h": 0.2, "offset": [0.5, 0.0]}]}]
    data["joints"] = [{"id": "sh", "kind": "pivot", "a": "torso", "b": "arm", "anchor_a": [1.0, 0.0]}]
    data["actuators"] = [{"kind": "motor", "id": "m", "joint": "sh", "max_torque": 5.0, "max_speed": 2.0}]
    data["sensors"] = [data["sensors"][0]]
    data["brain"] = {"inputs": "all", "hidden": [], "outputs": "all"}
    data["params"] = []
    doc = EnvDocument()
    doc.load_text(json.dumps(data, ensure_ascii=False))
    T.replicate(doc, "arm", 4, {"kind": "circle", "center": [0.0, 0.0]}, False, False)
    for k, bid in enumerate(["arm2", "arm3", "arm4"], start=1):
        th = math.pi / 2 * k
        body, joint = doc.item(("bodies", bid)), doc.item(("joints", f"sh{k + 1}"))
        assert body is not None and joint is not None
        assert close(body["pos"], [math.cos(th), math.sin(th)]) and close(body["angle"], th)
        assert close(joint["anchor_a"], [math.cos(th), math.sin(th)]) and close(joint["rest_angle"], th)
        assert joint["b"] == bid
    assert not errors(doc)


def test_t71_line_layout_with_numbers_and_refusals(qtbot: Any) -> None:
    doc = leg_doc()
    with pytest.raises(T.ToolError, match=S.ED_REPLICATE_NEED_AGENT):
        T.replicate(doc, "ground", 2, {"kind": "line"}, False, False)
    with pytest.raises(T.ToolError, match=S.ED_REPLICATE_ROOT.split("(")[0]):
        T.replicate(doc, "torso", 2, {"kind": "line"}, False, False)
    with pytest.raises(T.ToolError):
        T.replicate(doc, "nope", 2, {"kind": "line"}, False, False)
    with pytest.raises(ValueError):
        T.replicate(doc, "thigh", 17, {"kind": "line"}, False, False)
    T.replicate(doc, "thigh", 3, {"kind": "line", "start": [0.8, -0.6], "end": [-0.8, -0.6]}, False, False)
    assert [b["id"] for b in doc.items("bodies")] == ["torso", "thigh", "shin", "thigh2", "shin2", "thigh3", "shin3",
                                                      "ground"]
    assert doc.item(("bodies", "thigh2"))["pos"] == [0.0, -0.6]  # type: ignore[index]
    assert doc.item(("bodies", "thigh3"))["pos"] == [-0.8, -0.6]  # type: ignore[index]
    assert doc.item(("joints", "hip3"))["anchor_a"] == [-0.8, -0.3]  # type: ignore[index]
    assert doc.item(("joints", "hip3"))["rest_angle"] == 0.2  # type: ignore[index]  # 옮기기만: 각도 그대로
    assert doc.item(("joints", "knee3"))["a"] == "thigh3"  # type: ignore[index]
    assert not errors(doc)


def test_t71_copy_names_and_labels() -> None:
    assert T.numbered("wheel1", 1) == "wheel2" and T.numbered("wheel1", 3) == "wheel4"
    assert T.numbered("wheel", 1) == "wheel2" and T.numbered("j_r", 2) == "j_r3"
    assert T.copy_label("바퀴1 축", "바퀴1", 2) == "바퀴3 축"
    assert T.copy_label("뒷바퀴 축", "뒷바퀴", 1) == "뒷바퀴 2 축"
    assert T.copy_label("모터", "뒷바퀴", 1) == "모터 2" and T.copy_label("", "x", 1) == ""
    assert T.lin_comb([(-1.0, "=p.half")]) == "=-p.half"
    assert T.lin_comb([(1.0, "=-p.half")], 2.2) == "=-p.half + 2.2"
    assert T.lin_comb([(-1.0, "=-p.half")]) == "=p.half"
    assert T.lin_comb([(2.0, "=p.a + p.b")], -1.0) == "=2 * (p.a + p.b) - 1"
    assert T.lin_comb([(-1.0, 0.8)], 0.0) == -0.8 and T.lin_comb([(1.0, 0.1), (1.0, 0.2)]) == 0.3
    assert T.map_num(math.pi / 2, -1.0, math.pi) == math.pi / 2  # π/2를 6자리로 자르지 않는다


# ---------------------------------------------------------------------------
# T72: 구조 탭
# ---------------------------------------------------------------------------


def test_t72_structure_tab_counts_parts_that_disappear_together(qtbot: Any) -> None:
    doc = wheels_doc()
    rows = structure_rows(doc)
    assert [(r.body, len(r.parts), r.initial, r.param) for r in rows] == [
        ("wheel1", 3, True, "w1_on"), ("wheel2", 3, False, "w2_on"), ("wheel3", 3, False, "w3_on"),
        ("wheel4", 3, False, "w4_on")]
    assert rows[1].parts == (("joints", "j2"), ("actuators", "m2"), ("sensors", "c2"))
    leg = leg_doc()
    rows = {r.body: r for r in structure_rows(leg)}
    assert set(rows) == {"thigh", "shin"}  # 정강이는 허벅지와 함께 없어질 수 있다
    assert rows["thigh"].parts == (("bodies", "shin"), ("joints", "hip"), ("joints", "knee"), ("actuators", "m_hip"),
                                   ("actuators", "m_knee"), ("actuators", "jet"), ("sensors", "foot"),
                                   ("sensors", "knee_a"), ("sensors", "eye"))
    assert rows["shin"].parts == (("joints", "knee"), ("actuators", "m_knee"), ("actuators", "jet"),
                                  ("sensors", "foot"), ("sensors", "knee_a"))
    assert not rows["shin"].param and rows["shin"].evolvable and rows["shin"].initial is True
    assert attached_parts(leg.data, ["shin"]) == list(rows["shin"].parts)


def test_t72_row_selection_highlights_parts_on_the_canvas(qtbot: Any) -> None:
    view = make_view(qtbot, data=leg_env())
    tab = view.tabs.structure_tab
    assert isinstance(tab, StructureTab)
    assert view.tabs.tabText(view.tabs.indexOf(tab)) == S.ED_TAB_STRUCTURE
    assert view.tabs.indexOf(tab) == view.tabs.indexOf(view.tabs.brain_tab) + 1  # 「뇌 설정」 다음
    assert tab.table.rowCount() == 2 and tab.table.isVisible() and not tab.empty.isVisible()
    assert [tab.table.item(r, COL_PARTS).text() for r in range(2)] == ["9", "5"]
    assert tab.table.item(0, COL_INITIAL).text() == S.ED_STRUCT_ON
    tab.table.selectRow(0)
    assert view.doc.selection == [("bodies", "thigh")]  # 줄을 고르면 그 물체를 고른다(속성 패널)
    want = {("bodies", "thigh"), ("bodies", "shin"), ("joints", "hip"), ("joints", "knee"), ("actuators", "m_hip"),
            ("actuators", "m_knee"), ("actuators", "jet"), ("sensors", "foot"), ("sensors", "knee_a"),
            ("sensors", "eye")}
    assert view.canvas.structure_highlight == want
    assert view.canvas.linked(("joints", "knee")) and not view.canvas.linked(("sensors", "st"))
    view.canvas.repaint()  # 강조가 있는 그림도 예외 없이
    # 캔버스에서 정강이를 고르면 표의 줄도 따라간다
    view.doc.select([("bodies", "shin")])
    assert tab.current_body() == "shin"
    assert view.canvas.structure_highlight == {("bodies", "shin"), *structure_rows(view.doc)[1].parts}
    # 몸통을 고르면 강조가 꺼지고 「몸통은 없앨 수 없습니다」
    view.doc.select([("bodies", "torso")])
    assert tab.current_body() is None and view.canvas.structure_highlight == frozenset()
    assert tab.root_note.isVisible() and tab.root_note.text() == S.ED_STRUCT_ROOT_LOCKED
    assert not tab.replicate_button.isEnabled()
    # 다른 탭으로 가면 강조를 끈다
    view.doc.select([("bodies", "thigh")])
    assert view.canvas.structure_highlight
    view.tabs.setCurrentWidget(view.tabs.params_tab)
    assert view.canvas.structure_highlight == frozenset()
    view.tabs.setCurrentWidget(tab)
    assert ("bodies", "shin") in view.canvas.structure_highlight


def test_t72_empty_state_and_evolve_column(qtbot: Any) -> None:
    view = make_view(qtbot, doc_builtin="hill_car.evoenv")
    tab = view.tabs.structure_tab
    assert not tab.table.isVisible() and tab.empty.isVisible() and tab.empty.text() == S.ED_STRUCT_EMPTY
    view.doc.select([("bodies", "wheel_f")])
    assert tab.replicate_button.isEnabled()
    # 속성 패널에서 있음 진화를 켜면 표에 줄이 생긴다(보이는 동안 다시 계산)
    view.props.form.field("exists").evolve_box.click()  # type: ignore[union-attr]
    qtbot.waitUntil(lambda: tab.table.rowCount() == 1, timeout=2000)
    assert tab.table.item(0, COL_EVOLVE).checkState() == Qt.CheckState.Checked
    tab.set_evolve("wheel_f", False)  # 표의 진화 칸 = 속성 패널의 체크
    assert view.doc.item(("params", "wheel_f_on")) is None
    qtbot.waitUntil(lambda: tab.table.rowCount() == 0, timeout=2000)
    assert tab.empty.isVisible()


def test_t72_initial_state_survives_validation_errors(qtbot: Any) -> None:
    doc = wheels_doc()
    doc.set_value(("events", 0, "when"), "wheel2.x > 3")  # 없어질 수 있는 물체의 위치: 검사 오류
    assert any(i.message.startswith("'wheel2'") for i in errors(doc)) and doc.spec() is None
    assert initial_absent(doc) == {"wheel2", "wheel3", "wheel4"}  # 있음 칸만 계산해 이어짐 규칙
    assert [r.initial for r in structure_rows(doc)] == [True, False, False, False]


def test_t72_structure_tab_refresh_is_fast_on_wheels(qtbot: Any) -> None:
    """측정: 바퀴 고르는 언덕차에서 구조 탭 다시 계산 한 번(문서 컴파일은 캐시된 상태: 검사 뒤 갱신과 같다)."""
    doc = wheels_doc()
    tab = StructureTab(doc)
    qtbot.addWidget(tab)
    tab.show()
    times = []
    for _ in range(15):
        t0 = time.perf_counter()
        tab.refresh()
        times.append(time.perf_counter() - t0)
    median = sorted(times)[len(times) // 2]
    print(f"\nstructure tab refresh on wheels: median {median * 1000:.2f} ms, min {min(times) * 1000:.2f} ms")
    assert median < 0.25  # 넉넉한 상한(윈도우 CI, 붐비는 기계)


# ---------------------------------------------------------------------------
# T73: 캔버스 점선·「?」 표지와 「없는 부품도 보기」
# ---------------------------------------------------------------------------


def make_canvas(qtbot: Any, doc: EnvDocument) -> EditorCanvas:
    c = EditorCanvas(doc)
    qtbot.addWidget(c)
    c.resize(900, 600)
    c.show()
    c.scene()
    c.center = (0.0, 0.8)
    c.scale = 110.0
    return c


def test_t73_show_absent_toggle_draws_dashed_outlines(qtbot: Any) -> None:
    doc = wheels_doc()
    c = make_canvas(qtbot, doc)
    scene = c.scene()
    assert scene.optional == {"wheel1", "wheel2", "wheel3", "wheel4"}
    assert scene.absent == {"wheel2", "wheel3", "wheel4"}
    assert ("sensors", "c4") in scene.absent_parts and ("joints", "j4") in scene.absent_parts
    assert c.show_absent  # 에디터 기본: 없는 부품도 보인다(골라서 고칠 수 있게)
    on = c.grab().toImage()
    assert c.drawn_optional == {"wheel1": "present", "wheel2": "absent", "wheel3": "absent", "wheel4": "absent"}
    w4 = scene.body("wheel4")
    assert w4 is not None and w4.absent and w4.optional
    center = (w4.pose[0] + 0.3, w4.pose[1] - 0.2)  # 바퀴 안쪽(가운데는 관절 표시가 먼저 잡힌다)
    assert c.pick(center) is not None and c.pick(center).ref == ("bodies", "wheel4")  # type: ignore[union-attr]
    with qtbot.waitSignal(c.show_absent_changed):
        c.set_show_absent(False)
    off = c.grab().toImage()
    assert c.drawn_optional == {"wheel1": "present"}
    picked = c.pick(center)
    assert picked is None or picked.ref != ("bodies", "wheel4")
    assert not c.visible_ref(("actuators", "m4")) and c.visible_ref(("actuators", "m1"))
    assert all(item.ref[1] not in {"c2", "c3", "c4", "m2", "m3", "m4"} for item, _r in c.badge_rects())
    # 그림이 다르다: 바퀴4 자리(옅은 점선 바퀴)가 꺼지면 사라진다
    s = c.world_to_screen(w4.pose[0], w4.pose[1] - 0.38)  # 바퀴 아래쪽 안(바퀴살·레이·관절 표시와 떨어진 곳)
    p = QPoint(round(s.x()), round(s.y()))
    assert on.pixelColor(p) != off.pixelColor(p)
    c.set_show_absent(True)
    assert c.grab().toImage().pixelColor(p) == on.pixelColor(p)


def test_t73_toggle_is_shared_by_tab_and_canvas(qtbot: Any) -> None:
    view = make_view(qtbot, doc_builtin="wheels.evoenv")
    tab = view.tabs.structure_tab
    view.doc.select([("bodies", "wheel3")])
    tab.show_absent.setChecked(False)
    assert not view.canvas.show_absent
    assert view.doc.selection == []  # 숨은 부품은 고른 상태로 두지 않는다
    view.canvas.set_show_absent(True)
    assert tab.show_absent.isChecked()


# ---------------------------------------------------------------------------
# 오른쪽 메뉴, 대화 상자, 화면 전체 흐름
# ---------------------------------------------------------------------------


def test_body_menu_and_context_event(qtbot: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    doc = wheels_doc()
    c = make_canvas(qtbot, doc)
    menu = c.body_menu("wheel1")
    assert [a.text() for a in menu.actions()] == [S.ED_REPLICATE]
    with qtbot.waitSignal(c.replicate_requested) as sig:
        menu.actions()[0].trigger()
    assert sig.args == ["wheel1"]
    assert not c.request_replicate("chassis")  # 몸통은 안 된다: 이유를 상태 줄에
    assert c.status_text == S.ED_REPLICATE_ROOT
    shown: list[Any] = []
    monkeypatch.setattr(EditorCanvas, "show_menu", lambda self, menu, _pos: shown.append([x.text() for x in menu.actions()]))
    w2 = c.scene().body("wheel2")
    assert w2 is not None
    pos = c.world_to_screen(w2.pose[0], w2.pose[1])
    ev = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(round(pos.x()), round(pos.y())),
                           c.mapToGlobal(QPoint(round(pos.x()), round(pos.y()))))
    QApplication.sendEvent(c, ev)
    assert shown == [[S.ED_REPLICATE]] and doc.selection == [("bodies", "wheel2")]
    # 빈 곳: 메뉴 없음
    empty = c.world_to_screen(3.5, 3.0)
    QApplication.sendEvent(c, QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(round(empty.x()),
                                                round(empty.y())), QPoint(0, 0)))
    assert len(shown) == 1


def test_replicate_dialog_defaults_and_values(qtbot: Any) -> None:
    doc = one_wheel_doc()
    dlg = ReplicateDialog(doc, "wheel_r")
    qtbot.addWidget(dlg)
    count, layout, evolve, one_count = dlg.values()
    assert count == 2 and not evolve and not one_count
    assert layout["kind"] == "line" and close(layout["start"], [-1.1, -0.25]) and close(layout["end"], [1.1, -0.25])
    assert not dlg.one_count.isEnabled()
    dlg.evolve.setChecked(True)
    assert dlg.one_count.isEnabled()
    dlg.one_count.setChecked(True)
    dlg.count.setValue(4)
    assert dlg.values()[0] == 4 and dlg.values()[2:] == (True, True)
    dlg.layout_box.setCurrentIndex(T.REPLICATE_LAYOUTS.index("circle"))
    lay = dlg.layout_spec()
    assert lay["kind"] == "circle" and close(lay["center"], [0.0, 0.0])
    assert close(lay["radius"], math.hypot(1.1, 0.25), 1e-3)
    assert dlg.layout_tip.text() == S.ED_REPLICATE_LAYOUT_TIPS["circle"]
    dlg.layout_box.setCurrentIndex(T.REPLICATE_LAYOUTS.index("mirror"))
    assert not dlg.count.isEnabled() and dlg.values()[0] == 2 and dlg.layout_spec() == {"kind": "mirror", "x": 0.0}
    refs = dlg.apply()
    assert ("bodies", "wheel_r2") in refs and doc.item(("bodies", "wheel_r2"))["pos"][0] == "=p.half"  # type: ignore[index]


def test_view_replicate_runs_the_dialog_as_one_undo_step(qtbot: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    view = make_view(qtbot, doc_builtin="hill_car.evoenv")
    view.doc.remove([("bodies", "wheel_f"), ("joints", "j_f"), ("actuators", "m_f"), ("sensors", "c_f")])
    steps = view.doc.undo_stack.count()

    def fake_exec(dlg: ReplicateDialog) -> int:
        dlg.count.setValue(3)
        dlg.evolve.setChecked(True)
        return int(QDialog.DialogCode.Accepted)

    monkeypatch.setattr(ReplicateDialog, "exec", fake_exec)
    view.doc.select([("bodies", "wheel_r")])
    view.tabs.structure_tab.replicate_button.click()
    assert view.doc.undo_stack.count() == steps + 1
    assert [b["id"] for b in view.doc.items("bodies")] == ["chassis", "wheel_r", "wheel_r2", "wheel_r3"]
    # 사본이 원본의 몸 수치를 그대로 쓰면 상태 줄이 그것도 알린다(검토 반영, ED_REPLICATE_SHARED)
    shared = T.shared_params(view.doc, [("bodies", "wheel_r2"), ("joints", "j_r2"), ("actuators", "m_r2"),
                                        ("sensors", "c_r2"), ("bodies", "wheel_r3"), ("joints", "j_r3"),
                                        ("actuators", "m_r3"), ("sensors", "c_r3"), ("params", "wheel_r2_on"),
                                        ("params", "wheel_r3_on")])
    assert shared and "wheel_r2_on" not in shared
    assert view.canvas.status_text == (S.ED_REPLICATE_DONE.format(id="wheel_r", n=3) + " "
                                       + S.ED_REPLICATE_SHARED.format(params=", ".join(shared)))
    qtbot.waitUntil(lambda: view.tabs.structure_tab.table.rowCount() == 3, timeout=2000)
    assert view.doc.selection == [("bodies", "wheel_r"), ("bodies", "wheel_r2"), ("bodies", "wheel_r3")]
    view.doc.undo_stack.undo()
    assert [b["id"] for b in view.doc.items("bodies")] == ["chassis", "wheel_r"]
    # 세계 물체는 대화 상자를 띄우지 않고 이유를 알린다
    view.doc.load_text(json.dumps(leg_env(), ensure_ascii=False))
    assert view.replicate("ground") is None and view.canvas.status_text == S.ED_REPLICATE_NEED_AGENT


def test_builtin_menu_names_wheels() -> None:
    assert ("wheels.evoenv", S.ED_BUILTIN_WHEELS) in builtin_menu_items()


def test_exists_free_files_keep_their_text_through_the_editor(qtbot: Any) -> None:
    """있음 칸을 안 쓰는 파일은 에디터로 열고 구조 탭·캔버스를 거쳐도 글자 그대로(형식 1·2)."""
    for name, version in (("hill_car.evoenv", 1), ("jump.evoenv", 2)):
        view = make_view(qtbot, doc_builtin=name)
        view.canvas.repaint()
        text = view.doc.text()
        assert json.loads(text)["version"] == version and '"exists"' not in text
        assert text == dump_env(parse_env(text))
        assert view.tabs.structure_tab.rows == [] and view.canvas.scene().optional == frozenset()
        view.close()


def test_preview_point_type_is_qpointf() -> None:
    # 캔버스의 화면 좌표 변환이 QPointF를 돌려준다(오른쪽 메뉴의 위치 계산이 기대하는 형)
    doc = EnvDocument()
    c = EditorCanvas(doc)
    assert isinstance(c.world_to_screen(0.0, 0.0), QPointF)

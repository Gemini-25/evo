"""6.2.0 묶음 ED1: 밀기(이동) 도구와 밀기·균형 보조 표지 (설계 T34의 ED1 몫, D45, DE T11).

- 밀기 도구는 에이전트의 움직이는 물체에만 붙고(세계 물체·빈 곳은 까닭을 상태 줄에), 비어 있는 방향키를 받는다.
  옆에서 본 세계는 좌우만(x), 위에서 본 세계는 앞뒤좌우(xy). 한 번 = 되돌리기 한 단계, 결과는 검사 오류 없이 컴파일된다.
- 캔버스는 밀기와 균형 보조를 문서에서 그린다(엔진은 표지용 기하를 내지 않는다): 표지(배지)와 물체 가운데의 그림.
  표지를 누르면 그 구동기가 골라진다.
- 새 칸마다 풍선(폼은 ED2, 칸 글은 바탕 단계): 도구 풍선과 EDP_FIELD_TIPS의 새 칸.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest

from evolab.app import strings as S
from evolab.app.editor import tools as T
from evolab.app.editor.canvas import BADGE_TEXT, EditorCanvas
from evolab.app.editor.document import EnvDocument
from evolab.app.editor.palette import Palette
from evolab.presets.library import default_library
from evolab.spec.io import canonical_dict, parse_env

LEFT = Qt.MouseButton.LeftButton
NO = Qt.KeyboardModifier.NoModifier


def make(qtbot: Any, doc: EnvDocument) -> EditorCanvas:
    c = EditorCanvas(doc)
    qtbot.addWidget(c)
    c.resize(900, 600)
    c.show()
    c.fit()  # 처음 맞춤을 먼저 끝낸다(뒤의 보기 설정을 덮어쓰지 않게)
    c.center, c.scale = (0.0, 0.0), 60.0
    c.set_snap(True, 0.5)
    return c


def click_at(c: EditorCanvas, x: float, y: float) -> None:
    p = c.world_to_screen(x, y)
    QTest.mouseClick(c, LEFT, NO, QPoint(round(p.x()), round(p.y())))


def top_doc() -> EnvDocument:
    doc = EnvDocument()
    doc.set_value(("world", "view"), "top")
    doc.set_value(("world", "gravity"), [0.0, 0.0])
    doc.set_value(("world", "damping"), 0.5)
    return doc


def test_move_tool_attaches_to_agent_body_with_free_arrow_keys(qtbot: Any) -> None:
    doc = top_doc()
    c = make(qtbot, doc)
    pal = Palette()
    qtbot.addWidget(pal)
    pal.connect_canvas(c)
    T.add_circle(doc, "agent", (0.0, 0.0), 0.4)
    T.add_thruster(doc, "ball", (0.0, 0.0), 0.0)  # 추진기가 먼저 ↑·W를 가져간다
    pal.buttons["move"].click()
    assert c.tool_id == "move" and c.status_text == S.EDC_HINT_MOVE
    n0 = doc.undo_stack.count()
    click_at(c, 0.0, 0.0)
    assert doc.undo_stack.count() == n0 + 1
    move = doc.items("actuators")[-1]
    assert move["kind"] == "move" and move["body"] == "ball" and move["mode"] == "xy"
    assert move["label"] == S.EDC_TOOL_MOVE and move["max_speed"] > 0 and move["max_force"] > 0
    assert move["keys_up"] == [] and move["keys_down"] == ["Down", "S"]
    assert move["keys_left"] == ["Left", "A"] and move["keys_right"] == ["Right", "D"]
    assert doc.selection == [("actuators", move["id"])]
    assert not [i for i in doc.run_check() if i.severity == "error"]
    spec = doc.spec()
    assert spec is not None and {f"{move['id']}.x", f"{move['id']}.y"} <= set(spec.outputs)
    # 형식 4 부품: 저장하면 형식 4(가장 낮은 형식으로 저장)
    assert canonical_dict(parse_env(doc.data))["version"] == 4
    doc.undo_stack.undo()
    assert [a["kind"] for a in doc.items("actuators")] == ["thruster"]


def test_move_tool_in_side_world_is_left_right_only(qtbot: Any) -> None:
    doc = EnvDocument()
    c = make(qtbot, doc)
    T.add_box(doc, "agent", (0.0, 0.0), 1.0, 1.0)
    c.set_tool("move")
    click_at(c, 0.0, 0.0)
    move = doc.items("actuators")[-1]
    assert move["mode"] == "x" and move["keys_left"] == ["Left", "A"] and move["keys_right"] == ["Right", "D"]
    assert "keys_up" not in move and "keys_down" not in move
    # 둘째 밀기: 방향키가 이미 쓰였으므로 빈 키 목록(키 없는 구동기는 ED2의 뜻 검사가 알린다)
    click_at(c, 0.0, 0.0)
    second = doc.items("actuators")[-1]
    assert second["id"] != move["id"] and second["keys_left"] == [] and second["keys_right"] == []
    spec = doc.spec()
    assert spec is not None and f"{move['id']}.x" in spec.outputs


def test_move_tool_refuses_world_bodies_and_empty_space(qtbot: Any) -> None:
    doc = EnvDocument()
    c = make(qtbot, doc)
    T.add_box(doc, "world", (0.0, -2.0), 6.0, 0.5)
    c.set_tool("move")
    n0 = doc.undo_stack.count()
    click_at(c, 0.0, -2.0)
    assert c.status_text == S.EDC_ERR_ACTUATOR_BODY.format(body="block")
    click_at(c, 3.0, 3.0)
    assert c.status_text == S.EDC_ERR_CLICK_AGENT_BODY
    assert doc.undo_stack.count() == n0 and not doc.items("actuators")


def test_move_force_follows_body_mass() -> None:
    doc = EnvDocument()
    T.add_box(doc, "agent", (0.0, 0.0), 1.0, 2.0)  # 넓이 2 × 밀도 50 = 100kg
    ref = T.add_move(doc, "block", mode="heading")
    move = doc.item(ref)
    assert move is not None and move["max_force"] == 100.0 * T.MOVE_ACCEL and move["mode"] == "heading"
    assert T.body_mass({"shapes": [{"kind": "circle", "r": "=p.r"}]}) is None


def _marks_doc() -> tuple[EnvDocument, str]:
    doc = EnvDocument()
    lib = default_library()
    T.insert_template(doc, lib.get("ground_flat"))
    T.insert_template(doc, lib.get("biped"))  # 균형 보조(bal)가 몸통에
    ref = T.add_move(doc, "torso")  # 틀의 랜덤 값 push와 겹치지 않는 id를 받는다
    return doc, ref[1]


def test_move_and_balance_marks_and_badges(qtbot: Any) -> None:
    doc, push = _marks_doc()
    c = make(qtbot, doc)
    c.center, c.scale = (0.0, 0.9), 120.0
    scene = c.scene()
    assert c.scale == 120.0
    moves = {m.ref[1]: m for m in scene.moves}
    balances = {b.ref[1]: b for b in scene.balances}
    assert moves[push].body == "torso" and moves[push].mode == "x"
    assert balances["bal"].body == "torso" and balances["bal"].target == 0.0
    kinds = {b.ref[1]: b.kind for b in scene.badges if b.ref[0] == "actuators"}
    assert kinds[push] == "move" and kinds["bal"] == "balance"
    assert BADGE_TEXT["move"] == S.EDC_TOOL_MOVE and BADGE_TEXT["balance"] == S.EDP_VALUE_NAMES["balance"]
    # 표지를 누르면 그 구동기가 골라진다
    rects = {item.ref: rect for item, rect in c.badge_rects()}
    for ref in (("actuators", push), ("actuators", "bal")):
        center = rects[ref].center()
        QTest.mouseClick(c, LEFT, NO, QPoint(round(center.x()), round(center.y())))
        assert doc.selection == [ref]
    # 물체 가운데의 그림이 실제로 그려진다: 밀기·보조를 뺀 문서와 그림이 다르다
    with_marks = c.grab().toImage()
    torso = c.scene().body("torso")
    assert torso is not None
    o = c.world_to_screen(torso.pose[0], torso.pose[1])
    doc.remove([("actuators", push), ("actuators", "bal")])
    c.scene()
    without = c.grab().toImage()
    diff = 0
    for dx in range(-24, 25, 2):
        for dy in range(-30, 31, 2):
            q = (o + QPointF(dx, dy)).toPoint()
            diff += with_marks.pixel(q) != without.pixel(q)
    assert diff > 10


def test_heading_and_mirrored_balance_marks_paint(qtbot: Any) -> None:
    doc = top_doc()
    T.insert_template(doc, default_library().get("disc"))
    go = doc.item(("actuators", "go"))
    assert go is not None
    doc.set_value(("actuators", 0, "mode"), "heading")
    doc.add("actuators", {"kind": "balance", "id": "keep", "body": "disc", "stiffness": 10.0, "max_torque": 5.0,
                          "target": 0.5})
    c = make(qtbot, doc)
    scene = c.scene()
    assert [m.mode for m in scene.moves] == ["heading"] and [b.target for b in scene.balances] == [0.5]
    c.grab()  # 그림 오류 없음


def test_new_fields_have_tips() -> None:
    assert T.TOOL_HINTS["move"] == S.EDC_HINT_MOVE and T.TOOL_LABELS["move"] == S.EDC_TOOL_MOVE
    for key in ("World.view", "World.damping", "Move.mode", "Move.max_force", "Move.max_speed", "Move.max_turn",
                "keys_up", "keys_down", "Balance.stiffness", "Balance.damping", "Balance.max_torque", "Balance.target"):
        assert S.EDP_FIELD_NAMES.get(key), key
    tips = S.EDP_FIELD_TIPS
    for key in ("World.view", "World.damping", "Move.mode", "Move.max_force", "Move.max_speed", "Move.max_turn",
                "Balance.stiffness", "Balance.damping", "Balance.max_torque", "Balance.target"):
        assert tips.get(key) or tips.get(key.split(".")[1]), key

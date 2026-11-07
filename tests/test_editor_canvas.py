"""M5a: 에디터 캔버스, 도구, 팔레트.

도구마다 캔버스에 마우스·키 입력을 넣어(화면 좌표는 world_to_screen으로 계산) 문서가 바뀌는지, 되돌리기로
없어지는지 본다. 마지막에 착륙선 에이전트와 트랙 자동차를 도구로 다시 만들어 컴파일되는지 확인한다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest

from evolab import paths
from evolab.app.editor import tools as T
from evolab.app.editor.canvas import EditorCanvas, build_scene
from evolab.app.editor.document import EnvDocument
from evolab.app.editor.palette import TOOL_IDS, Palette
from evolab.spec.io import compile_env, parse_env

SCRATCH = Path("/tmp/claude-0/-home-user-evo/6ff254e5-e86e-5068-a687-5ac6a7065af0/scratchpad")
NO = Qt.KeyboardModifier.NoModifier
LEFT = Qt.MouseButton.LeftButton


# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------


def make_canvas(qtbot: Any, doc: EnvDocument | None = None, center: tuple[float, float] = (0.0, 0.0),
                scale: float = 40.0, snap: float = 0.5) -> EditorCanvas:
    doc = doc or EnvDocument()
    c = EditorCanvas(doc)
    qtbot.addWidget(c)
    c.resize(1000, 700)
    c.show()
    c.fit()
    c.center = center
    c.scale = scale
    c.set_snap(True, snap)
    return c


def pt(c: EditorCanvas, x: float, y: float) -> QPoint:
    p = c.world_to_screen(x, y)
    return QPoint(round(p.x()), round(p.y()))


def click(c: EditorCanvas, x: float, y: float, mods: Qt.KeyboardModifier = NO) -> None:
    QTest.mouseClick(c, LEFT, mods, pt(c, x, y))


def drag(c: EditorCanvas, a: tuple[float, float], b: tuple[float, float], mods: Qt.KeyboardModifier = NO,
         steps: int = 4) -> None:
    QTest.mousePress(c, LEFT, mods, pt(c, *a))
    for k in range(1, steps + 1):
        t = k / steps
        QTest.mouseMove(c, pt(c, a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    QTest.mouseRelease(c, LEFT, mods, pt(c, *b))


def key(c: EditorCanvas, k: Qt.Key, mods: Qt.KeyboardModifier = NO) -> None:
    QTest.keyClick(c, k, mods)


def body(doc: EnvDocument, bid: str) -> dict[str, Any]:
    item = doc.item(("bodies", bid))
    assert item is not None, bid
    return item


def last(doc: EnvDocument, section: str) -> dict[str, Any]:
    return doc.items(section)[-1]


def approx(a: Any, b: Any, tol: float = 1e-6) -> bool:
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b, strict=True))


def agent_doc() -> EnvDocument:
    """시작 위치 (0, 0)의 새 문서에 에이전트 본체(2 x 1) 하나."""
    doc = EnvDocument()
    T.add_box(doc, "agent", (0.0, 0.0), 2.0, 1.0)
    doc.rename(("bodies", "block"), "hull")
    return doc


# ---------------------------------------------------------------------------
# 그리기
# ---------------------------------------------------------------------------


def test_render_builtin_examples(qtbot: Any) -> None:
    for name, png in (("lander", "m5a_canvas_lander.png"), ("track_car", "m5a_canvas_track.png")):
        doc = EnvDocument()
        doc.from_builtin(f"{name}.evoenv")
        c = make_canvas(qtbot, doc)
        scene = c.scene()
        assert scene.exact and not scene.message
        if name == "lander":
            assert {b.id for b in scene.bodies} == {"terrain", "hull", "leg_l", "leg_r"}
            assert len(scene.joints) == 2 and len(scene.thrusters) == 3 and scene.region("terrain.pad")
            assert len(scene.targets) == 1
            hull = scene.body("hull")
            assert hull is not None
            # 오류 강조가 보이게: 왼다리 접촉 센서가 없는 물체를 가리키게 해 둔다.
            i = doc.find(("sensors", "c_l"))
            doc.set_value(("sensors", i, "with"), ["tag.nope"])
            doc.run_check()
            assert c.issue_severity(("sensors", "c_l")) == "error"
            doc.select([("bodies", "hull"), ("actuators", "main")])
            c.center, c.scale = (hull.pose[0] - 1.5, hull.pose[1] - 2.0), 70.0
        else:
            assert len(scene.checkpoints) == 24 and len(scene.rays) == 1 and len(scene.rays[0].lines) == 7
            car = scene.body("car")
            assert car is not None
            doc.select([("bodies", "car")])
            c.fit()
        img = c.grab().toImage()
        assert img.width() == 1000 and img.height() == 700
        img.save(str(SCRATCH / png))
        # 전체 보기 그림도 남긴다
        c.fit()
        c.grab().toImage().save(str(SCRATCH / png.replace(".png", "_fit.png")))


def test_scene_preview_matches_runtime_poses(qtbot: Any) -> None:
    """미리보기 자세는 런타임 Episode의 시작 자세와 같다(place_by 다리, 수식 시작 위치 포함)."""
    from evolab.app.editor.canvas import _preview_episode

    doc = EnvDocument()
    doc.from_builtin("lander.evoenv")
    scene = build_scene(doc)
    ep = _preview_episode(doc.spec())
    for bid in ("hull", "leg_l", "leg_r"):
        b = scene.body(bid)
        assert b is not None
        assert approx(b.pose, (ep.bodies[bid].position.x, ep.bodies[bid].position.y, ep.bodies[bid].angle))
    j = next(x for x in scene.joints if x.ref == ("joints", "j_leg_l"))
    assert approx(j.pa, j.pb, 1e-9)  # 다리 앵커가 본체 앵커와 겹친다


def test_broken_document_still_draws(qtbot: Any) -> None:
    doc = EnvDocument()
    doc.from_builtin("lander.evoenv")
    i = doc.find(("bodies", "hull"))
    doc.set_value(("bodies", i, "shapes", 0, "w"), "=p.nope * 2")  # 수식 오류
    doc.set_value(("sensors", doc.find(("sensors", "pad_rel")), "body"), "ghost")  # 참조 오류
    doc.add("bodies", {"id": "half", "shapes": [{"kind": "box", "w": "=oops("}]})  # 구조 오류
    assert doc.spec() is None
    c = make_canvas(qtbot, doc)
    scene = c.scene()
    assert not scene.exact and scene.message
    ids = {b.id for b in scene.bodies}
    assert {"terrain", "hull", "leg_l", "leg_r", "half"} <= ids  # 오류 난 물체도 보인다
    assert scene.body("half").placeholder  # type: ignore[union-attr]
    c.grab()
    # 아주 망가진 내용도 죽지 않는다
    doc.edit("x", lambda d: d.update({"bodies": [{"id": "z", "shapes": "abc", "pos": "q"}, 5], "joints": [None],
                                      "generators": [{"kind": "heightmap", "id": "g", "x_range": "bad"}],
                                      "agent": {"spawn": {"x": "=", "y": None}}, "regions": [{"shape": 3}]}))
    c.grab()
    assert build_scene(doc) is not None


def test_camera_zoom_pan_fit_and_coords(qtbot: Any) -> None:
    c = make_canvas(qtbot, agent_doc(), center=(0.0, 0.0), scale=40.0)
    seen: list[tuple[float, float]] = []
    c.coords_changed.connect(lambda x, y: seen.append((x, y)))
    QTest.mouseMove(c, pt(c, 2.0, 1.0))
    assert seen and approx(seen[-1], (2.0, 1.0), 0.03)
    # 휠: 커서 아래 점이 그대로
    pos = QPointF(pt(c, 3.0, -2.0))
    w0 = c.screen_to_world(pos)
    ev = QWheelEvent(pos, c.mapToGlobal(pos), QPoint(0, 0), QPoint(0, 240), Qt.MouseButton.NoButton, NO,
                     Qt.ScrollPhase.NoScrollPhase, False)
    c.wheelEvent(ev)
    assert c.scale > 40.0 and approx(c.screen_to_world(pos), w0, 1e-9)
    # 가운데 단추로 이동
    c0 = c.center
    QTest.mousePress(c, Qt.MouseButton.MiddleButton, NO, QPoint(500, 350))
    QTest.mouseMove(c, QPoint(560, 350))
    QTest.mouseRelease(c, Qt.MouseButton.MiddleButton, NO, QPoint(560, 350))
    assert c.center[0] < c0[0]
    c.fit()
    hull = c.scene().body("hull")
    assert hull is not None
    s = c.world_to_screen(hull.pose[0], hull.pose[1])
    assert 0 < s.x() < c.width() and 0 < s.y() < c.height()


# ---------------------------------------------------------------------------
# 모양 도구
# ---------------------------------------------------------------------------


def test_box_tool_adds_body_and_undo(qtbot: Any) -> None:
    c = make_canvas(qtbot)
    doc = c.doc
    c.set_tool("box")
    drag(c, (-1.1, -0.4), (2.9, 1.6))  # 격자 0.5에 맞춰 (-1, -0.5)..(3, 1.5)
    b = last(doc, "bodies")
    assert b["owner"] == "agent" and b["type"] == "dynamic" and b["shapes"] == [{"kind": "box", "w": 4.0, "h": 2.0}]
    assert b["pos"] == [1.0, 0.5] and b["material"]["density"] == 50.0 and b["material"]["friction"] == 0.6
    assert b["label"] and doc.selection == [("bodies", b["id"])]
    doc.undo_stack.undo()
    assert not doc.items("bodies")
    # 세계 물체, 격자 끔(Alt)
    c.set_new_owner("world")
    drag(c, (0.0, 0.0), (1.3, 0.7), Qt.KeyboardModifier.AltModifier)
    b = last(doc, "bodies")
    assert b["owner"] == "world" and b["type"] == "static" and b["tags"] == ["obstacle"]
    assert approx((b["shapes"][0]["w"], b["shapes"][0]["h"]), (1.3, 0.7), 0.03)
    # 그냥 누르면 기본 크기
    click(c, 5.0, 5.0)
    assert last(doc, "bodies")["shapes"][0] == {"kind": "box", "w": 1.0, "h": 1.0}


def test_box_in_agent_frame_and_ctrl_adds_shape(qtbot: Any) -> None:
    doc = agent_doc()
    T.set_spawn(doc, 10.0, 5.0)
    c = make_canvas(qtbot, doc, center=(10.0, 5.0), scale=60.0)
    c.set_tool("box")
    drag(c, (12.0, 5.0), (13.0, 6.0))  # 세계 좌표 -> 에이전트 좌표(시작 위치 기준)
    assert last(doc, "bodies")["pos"] == [2.5, 0.5]
    doc.select([("bodies", "hull")])
    c.set_tool("circle")
    drag(c, (10.0, 4.5), (10.5, 4.5), Qt.KeyboardModifier.ControlModifier)
    hull = body(doc, "hull")
    assert hull["shapes"][1] == {"kind": "circle", "r": 0.5, "offset": [0.0, -0.5]}
    doc.undo_stack.undo()
    assert len(body(doc, "hull")["shapes"]) == 1


def test_circle_tool(qtbot: Any) -> None:
    c = make_canvas(qtbot)
    c.set_tool("circle")
    drag(c, (1.0, 2.0), (2.5, 2.0))
    b = last(c.doc, "bodies")
    assert b["pos"] == [1.0, 2.0] and b["shapes"] == [{"kind": "circle", "r": 1.5}]
    c.doc.undo_stack.undo()
    assert not c.doc.items("bodies")


def test_poly_tool_and_self_intersection(qtbot: Any) -> None:
    c = make_canvas(qtbot)
    doc = c.doc
    c.set_tool("poly")
    for x, y in ((0, 0), (4, 0), (4, 2)):
        click(c, x, y)
    QTest.mouseDClick(c, LEFT, NO, pt(c, 0, 2))
    b = last(doc, "bodies")
    assert b["pos"] == [2.0, 1.0]
    assert b["shapes"][0]["points"] == [[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]]
    assert parse_env(doc.data)
    # 꼬인 다각형(나비 모양)은 거부, 점은 남아서 고칠 수 있다
    n = len(doc.items("bodies"))
    messages: list[str] = []
    c.status_changed.connect(messages.append)
    for x, y in ((0, 5), (4, 7), (4, 5), (0, 7)):
        click(c, x, y)
    key(c, Qt.Key.Key_Return)
    assert len(doc.items("bodies")) == n and messages and "교차" in messages[-1]
    assert c.tool.busy()
    key(c, Qt.Key.Key_Escape)
    assert not c.tool.busy() and c.tool_id == "poly"
    key(c, Qt.Key.Key_Escape)  # 한 번 더 누르면 선택 도구로
    assert c.tool_id == "select"
    with pytest.raises(T.ToolError):
        T.add_poly(doc, "agent", [(0, 0), (1, 1), (2, 2)])


def test_terrain_line_tool(qtbot: Any) -> None:
    c = make_canvas(qtbot)
    c.set_tool("terrain_line")
    for x, y in ((-6, 0), (-2, 1), (2, -1), (6, 0)):
        click(c, x, y)
    key(c, Qt.Key.Key_Return)
    b = last(c.doc, "bodies")
    assert b["owner"] == "world" and b["type"] == "static" and b["tags"] == ["terrain"]
    assert b["shapes"][0]["kind"] == "chain" and len(b["shapes"][0]["points"]) == 4
    scene = c.scene()
    assert len(scene.body(b["id"]).segments) == 3  # type: ignore[union-attr]
    c.doc.undo_stack.undo()
    assert not c.doc.items("bodies")


def test_terrain_and_track_auto_tools(qtbot: Any) -> None:
    c = make_canvas(qtbot)
    doc = c.doc
    changed: list[str] = []
    c.tool_changed.connect(changed.append)
    c.set_tool("terrain_auto")
    g = last(doc, "generators")
    assert g["kind"] == "heightmap" and g["pad"]["width"] == 8.0 and g["x_range"] == [-50.0, 50.0]
    assert c.tool_id == "select" and changed[-1] == "select"
    assert c.scene().region(f"{g['id']}.pad") is not None and doc.spec() is not None
    c.set_tool("track_auto")
    t = last(doc, "generators")
    assert t["kind"] == "track" and t["source"] == "generated" and t["set_size"] == 5
    n = len(doc.items("generators"))
    messages: list[str] = []
    c.status_changed.connect(messages.append)
    c.set_tool("track_auto")  # 세트 트랙 생성기는 하나만
    assert len(doc.items("generators")) == n and messages
    doc.undo_stack.undo()
    doc.undo_stack.undo()
    assert not doc.items("generators")


def test_track_draw_tool(qtbot: Any) -> None:
    c = make_canvas(qtbot, center=(0.0, 0.0), scale=3.0, snap=1.0)
    c.set_tool("track_draw")
    loop = [(40 * math.cos(2 * math.pi * k / 8), 25 * math.sin(2 * math.pi * k / 8)) for k in range(8)]
    for x, y in loop[:-1]:
        click(c, x, y)
    QTest.mouseDClick(c, LEFT, NO, pt(c, *loop[-1]))
    g = last(c.doc, "generators")
    assert g["kind"] == "track" and g["source"] == "drawn" and len(g["centerline"]) == 8 and g["width"] == 8.0
    # 트랙만 있어도 에이전트 물체 없이는 검사 오류. 물체를 더하면 런타임 트랙이 그려진다
    T.add_box(c.doc, "agent", (0.0, 0.0), 2.0, 1.0)
    scene = c.scene()
    assert scene.body(f"{g['id']}.walls") is not None and len(scene.checkpoints) == 24
    # 너무 작은 중심선은 거부(트랙 생성기의 이유가 보인다)
    with pytest.raises(T.ToolError):
        T.add_drawn_track(c.doc, [(0, 0), (1, 0), (0, 1)])


# ---------------------------------------------------------------------------
# 관절
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool_id,kind", [("joint_pivot", "pivot"), ("joint_rot_spring", "rot_spring"),
                                          ("joint_weld", "weld"), ("joint_spring", "spring")])
def test_joint_tools(qtbot: Any, tool_id: str, kind: str) -> None:
    doc = agent_doc()
    T.add_box(doc, "agent", (4.0, -2.0), 0.5, 2.0)  # box
    c = make_canvas(qtbot, doc, center=(1.0, -1.0), scale=60.0)
    c.set_tool(tool_id)
    click(c, -1.0, -0.5)  # 본체 왼쪽 아래 모서리
    assert c.tool.busy()
    click(c, 4.0, -1.0)  # 막대 위 끝
    j = last(doc, "joints")
    assert j["kind"] == kind and j["a"] == "hull" and j["b"] == "block"
    assert j["anchor_a"] == [-1.0, -0.5] and j["anchor_b"] == [0.0, 1.0]
    placed = body(doc, "block").get("place_by")
    if kind == "spring":
        assert placed is None and j["stiffness"] == 100.0
    else:
        assert placed == j["id"]
        # 앵커가 겹치도록 막대가 옮겨진다
        jj = c.scene().joints[0]
        assert approx(jj.pa, jj.pb, 1e-9) and approx(jj.pa, (-1.0, -0.5), 1e-9)
    if kind == "rot_spring":
        assert j["stiffness"] == 600.0 and j["damping"] == 200.0 and j["rest_angle"] == 0.0
    assert doc.spec() is not None
    doc.undo_stack.undo()
    assert not doc.items("joints") and body(doc, "block").get("place_by") is None


def test_joint_needs_two_bodies(qtbot: Any) -> None:
    c = make_canvas(qtbot, agent_doc(), scale=60.0)
    c.set_tool("joint_pivot")
    click(c, 0.0, 0.0)
    click(c, 0.2, 0.1)  # 같은 물체: 둘째 물체로 치지 않는다
    assert not c.doc.items("joints") and c.tool.busy()
    click(c, 8.0, 8.0)  # 빈 곳
    assert not c.doc.items("joints")


# ---------------------------------------------------------------------------
# 구동기
# ---------------------------------------------------------------------------


def test_thruster_tool_click_and_drag(qtbot: Any) -> None:
    c = make_canvas(qtbot, agent_doc(), scale=80.0)
    doc = c.doc
    c.set_tool("thruster")
    click(c, 0.0, -0.5)  # 바닥 가운데: 위로 민다(분사는 바깥으로)
    t = last(doc, "actuators")
    assert t["kind"] == "thruster" and t["body"] == "hull" and t["at"] == [0.0, -0.5]
    assert t["dir"] == pytest.approx(math.pi / 2) and t["max_force"] == 100.0 and t["keys"] == ["Up", "W"]
    click(c, -1.0, 0.0)  # 왼쪽 옆: 오른쪽으로
    assert last(doc, "actuators")["dir"] == 0.0 and last(doc, "actuators")["keys"] == ["Right", "D"]
    drag(c, (1.0, 0.0), (1.0, -2.0))  # 분사를 아래로 끌면 힘은 위쪽
    t = last(doc, "actuators")
    assert t["at"] == [1.0, 0.0] and t["dir"] == pytest.approx(math.pi / 2)
    assert len(c.scene().thrusters) == 3
    doc.undo_stack.undo()
    assert len(doc.items("actuators")) == 2
    # 세계 물체에는 달 수 없다
    T.add_box(doc, "world", (0.0, -3.0), 4.0, 1.0)
    n = len(doc.items("actuators"))
    click(c, 0.0, -3.0)
    assert len(doc.items("actuators")) == n


def test_fuel_tank_vehicle_tools(qtbot: Any) -> None:
    doc = agent_doc()
    T.add_thruster(doc, "hull", (0.0, -0.5), math.pi / 2)
    T.add_thruster(doc, "hull", (-1.0, 0.0), 0.0)
    c = make_canvas(qtbot, doc, scale=60.0)
    c.set_tool("fuel_tank")
    click(c, 0.3, 0.2)
    tank = last(doc, "actuators")
    assert tank["kind"] == "fuel_tank" and tank["capacity"] == 100.0
    thrusters = [a for a in doc.items("actuators") if a["kind"] == "thruster"]
    assert all(t["tank"] == tank["id"] and t["fuel_per_s"] == 1.0 for t in thrusters)
    assert any(b.kind == "fuel_tank" and b.body == "hull" for b in c.scene().badges)
    doc.undo_stack.undo()  # 탱크와 추진기 연결을 한 번에 되돌린다
    assert all("tank" not in a for a in doc.items("actuators"))
    c.set_tool("vehicle")
    click(c, 0.3, 0.2)
    v = last(doc, "actuators")
    assert v["kind"] == "vehicle" and v["body"] == "hull" and v["wheelbase"] == 1.6
    # 추진기가 이미 쓰는 키(Up/W, Right/D)는 비워 둔다
    assert v["keys_forward"] == [] and v["keys_right"] == [] and v["keys_back"] == ["Down", "S"]
    assert doc.spec() is not None


# ---------------------------------------------------------------------------
# 센서
# ---------------------------------------------------------------------------


def test_sensor_tools(qtbot: Any) -> None:
    doc = agent_doc()
    T.add_heightmap(doc)
    T.set_spawn(doc, 0.0, 10.0)
    c = make_canvas(qtbot, doc, center=(0.0, 6.0), scale=40.0)
    # 레이: 누른 자리에서, 바깥쪽을 본다
    c.set_tool("sensor_rays")
    click(c, 1.0, 10.0)
    r = last(doc, "sensors")
    assert r["kind"] == "rays" and r["origin"] == [1.0, 0.0] and r["center"] == 0.0
    assert r["count"] == 7 and r["spread"] == pytest.approx(math.pi) and r["length"] == 40.0
    assert r["hits"] == ["tag.terrain", "tag.pad"]
    click(c, 0.0, 9.5)
    assert last(doc, "sensors")["center"] == pytest.approx(-math.pi / 2)
    # 몸 상태, 접촉
    c.set_tool("sensor_body_state")
    click(c, 0.0, 10.0)
    st = last(doc, "sensors")
    assert st["kind"] == "body_state" and st["fields"] == ["vx", "vy", "angle", "omega"] and st["frame"] == "world"
    c.set_tool("sensor_contact")
    click(c, 0.0, 10.0)
    ct = last(doc, "sensors")
    assert ct["kind"] == "contact" and ct["with"] == ["tag.terrain"] and ct["body"] == "hull"
    # 목표 위치: 물체 -> 패드 영역
    pad = c.scene().region("terrain.pad")
    assert pad is not None
    c.set_tool("sensor_target")
    click(c, 0.0, 10.0)
    px = sum(p[0] for p in pad.points) / 4
    py = sum(p[1] for p in pad.points) / 4
    click(c, px, py)
    tg = last(doc, "sensors")
    assert tg["kind"] == "target" and tg["body"] == "hull" and tg["target"] == "terrain.pad"
    assert len(c.scene().targets) == 1
    # 남은 시간(바로 실행)
    c.set_tool("sensor_time")
    assert last(doc, "sensors")["kind"] == "time_left" and c.tool_id == "select"
    # 남은 연료: 탱크가 없으면 안내만
    c.set_tool("sensor_fuel")
    n = len(doc.items("sensors"))
    click(c, 0.0, 10.0)
    assert len(doc.items("sensors")) == n
    T.add_thruster(doc, "hull", (0.0, -0.5), math.pi / 2)
    T.add_fuel_tank(doc, "hull")
    click(c, 0.0, 10.0)
    fs = last(doc, "sensors")
    assert fs["kind"] == "fuel" and fs["tank"] == "tank"
    assert doc.spec() is not None
    for _ in range(3):
        doc.undo_stack.undo()
    assert doc.items("sensors")[-1]["kind"] == "time_left"


def test_joint_angle_sensor_tool(qtbot: Any) -> None:
    doc = agent_doc()
    T.add_box(doc, "agent", (3.0, 0.0), 0.5, 1.0)
    T.add_joint(doc, "pivot", "hull", "block", (1.0, 0.0), (0.0, 0.5))
    c = make_canvas(qtbot, doc, scale=60.0)
    c.set_tool("sensor_joint_angle")
    click(c, 1.0, 0.0)
    s = last(doc, "sensors")
    assert s["kind"] == "joint_angle" and s["joint"] == "joint"
    assert any(b.kind == "joint_angle" for b in c.scene().badges)
    assert doc.spec() is not None


# ---------------------------------------------------------------------------
# 영역, 체크포인트, 시작 위치
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool_id,kind", [("region_goal", "goal"), ("region_hazard", "hazard"), ("region_pad", "pad")])
def test_region_tools(qtbot: Any, tool_id: str, kind: str) -> None:
    c = make_canvas(qtbot)
    c.set_tool(tool_id)
    drag(c, (2.0, 1.0), (6.0, 3.0))
    r = last(c.doc, "regions")
    assert r["kind"] == kind and r["pos"] == [4.0, 2.0] and r["shape"] == {"kind": "box", "w": 4.0, "h": 2.0}
    assert c.scene().region(r["id"]) is not None
    c.doc.undo_stack.undo()
    assert not c.doc.items("regions")


def test_checkpoint_tool(qtbot: Any) -> None:
    c = make_canvas(qtbot)
    c.set_tool("checkpoints")
    for a, b in (((0, -2), (0, 2)), ((5, -2), (5, 2)), ((10, -2), (10, 2))):
        click(c, *a)
        click(c, *b)
    key(c, Qt.Key.Key_Backspace)  # 마지막 선 지우기
    key(c, Qt.Key.Key_Return)
    cp = last(c.doc, "checkpoints")
    assert cp["loop"] is True and cp["lines"] == [[[0.0, -2.0], [0.0, 2.0]], [[5.0, -2.0], [5.0, 2.0]]]
    assert sum(1 for x in c.scene().checkpoints if x.ref == ("checkpoints", cp["id"])) == 2
    c.doc.undo_stack.undo()
    assert not c.doc.items("checkpoints")


def test_spawn_tool_and_formula_fields(qtbot: Any) -> None:
    c = make_canvas(qtbot, agent_doc())
    doc = c.doc
    c.set_tool("spawn")
    drag(c, (3.0, 4.0), (3.0, 8.0))
    sp = doc.get(("agent", "spawn"))
    assert sp["x"] == 3.0 and sp["y"] == 4.0 and sp["angle"] == pytest.approx(math.pi / 2)
    hull = c.scene().body("hull")
    assert hull is not None and approx(hull.pose, (3.0, 4.0, math.pi / 2))
    doc.undo_stack.undo()
    assert doc.get(("agent", "spawn", "x"), 0.0) == 0.0
    doc.set_value(("agent", "spawn", "x"), "=1 + 1")
    click(c, 5.0, 5.0)
    assert doc.get(("agent", "spawn", "x")) == "=1 + 1" and doc.get(("agent", "spawn", "y")) == 5.0
    assert "x" in c.status_text


# ---------------------------------------------------------------------------
# 선택·이동·회전
# ---------------------------------------------------------------------------


def three_bodies() -> EnvDocument:
    doc = agent_doc()
    T.add_box(doc, "agent", (4.0, 0.0), 1.0, 1.0)
    T.add_circle(doc, "world", (0.0, 4.0), 0.5)
    T.add_region(doc, "goal", (-5.0, 3.0), 2.0, 2.0)
    doc.select([])
    return doc


def test_select_click_shift_and_band(qtbot: Any) -> None:
    c = make_canvas(qtbot, three_bodies(), scale=40.0)
    doc = c.doc
    click(c, 0.2, 0.1)
    assert doc.selection == [("bodies", "hull")]
    click(c, 4.0, 0.0, Qt.KeyboardModifier.ShiftModifier)
    assert doc.selection == [("bodies", "hull"), ("bodies", "block")]
    click(c, 4.0, 0.0, Qt.KeyboardModifier.ShiftModifier)  # 다시 누르면 뺀다
    assert doc.selection == [("bodies", "hull")]
    click(c, 8.0, -6.0)  # 빈 곳
    assert doc.selection == []
    drag(c, (-7.0, 5.0), (1.5, -1.0))  # 사각형 선택: 본체, 원, 목표 영역
    assert set(doc.selection) == {("bodies", "hull"), ("bodies", "ball"), ("regions", "finish")}


def test_drag_move_snaps_and_is_one_undo_step(qtbot: Any) -> None:
    c = make_canvas(qtbot, three_bodies(), scale=40.0)
    doc = c.doc
    n = doc.undo_stack.count()
    drag(c, (4.0, 0.0), (5.6, 1.3), steps=6)  # 격자 0.5: (1.6, 1.3) -> (1.5, 1.5)
    assert body(doc, "block")["pos"] == [5.5, 1.5]
    assert doc.undo_stack.count() == n + 1
    doc.undo_stack.undo()
    assert body(doc, "block")["pos"] == [4.0, 0.0]
    # Alt: 격자 맞춤 없이
    drag(c, (4.0, 0.0), (4.3, 0.2), Qt.KeyboardModifier.AltModifier)
    assert approx(body(doc, "block")["pos"], (4.3, 0.2), 0.03)
    doc.undo_stack.undo()
    # 여러 개를 함께: 영역과 물체
    doc.select([("bodies", "block"), ("regions", "finish")])
    drag(c, (4.0, 0.0), (5.0, -1.0))
    assert body(doc, "block")["pos"] == [5.0, -1.0] and doc.item(("regions", "finish"))["pos"] == [-4.0, 2.0]
    # 작은 떨림(3px 미만)은 이동이 아니다
    before = doc.undo_stack.count()
    QTest.mousePress(c, LEFT, NO, pt(c, 5.0, -1.0))
    QTest.mouseMove(c, pt(c, 5.0, -1.0) + QPoint(1, 1))
    QTest.mouseRelease(c, LEFT, NO, pt(c, 5.0, -1.0) + QPoint(1, 1))
    assert doc.undo_stack.count() == before


def test_escape_cancels_drag_and_locked_hover(qtbot: Any) -> None:
    c = make_canvas(qtbot, three_bodies(), scale=40.0)
    doc = c.doc
    index = doc.undo_stack.index()
    QTest.mousePress(c, LEFT, NO, pt(c, 4.0, 0.0))
    QTest.mouseMove(c, pt(c, 6.0, 2.0))
    assert body(doc, "block")["pos"] == [6.0, 2.0]
    key(c, Qt.Key.Key_Escape)
    QTest.mouseRelease(c, LEFT, NO, pt(c, 6.0, 2.0))
    assert body(doc, "block")["pos"] == [4.0, 0.0] and doc.undo_stack.index() == index
    doc.set_value(("bodies", doc.find(("bodies", "block")), "pos"), ["=4", "=0"])
    QTest.mouseMove(c, pt(c, 4.0, 0.1))
    assert c.cursor().shape() == Qt.CursorShape.ForbiddenCursor and c.toolTip()
    QTest.mouseMove(c, pt(c, 8.0, -6.0))
    assert c.cursor().shape() == Qt.CursorShape.ArrowCursor and not c.toolTip()


def test_drag_agent_body_with_rotated_spawn(qtbot: Any) -> None:
    doc = three_bodies()
    T.set_spawn(doc, 0.0, 0.0, math.pi / 2)  # 에이전트 좌표 x가 세계 위쪽
    c = make_canvas(qtbot, doc, scale=40.0)
    box = c.scene().body("block")
    assert box is not None and approx(box.pose[:2], (0.0, 4.0), 1e-9)
    drag(c, (0.0, 4.0), (-1.0, 4.0))  # 세계 왼쪽 = 에이전트 +y
    assert body(doc, "block")["pos"] == [4.0, 1.0]


def test_rotate_handle(qtbot: Any) -> None:
    c = make_canvas(qtbot, three_bodies(), scale=40.0)
    doc = c.doc
    doc.select([("bodies", "hull")])
    hull = c.scene().body("hull")
    assert hull is not None
    h = c.rotate_handle(hull)
    # 손잡이를 왼쪽으로 돌려 90도
    r = math.hypot(*h)
    drag(c, h, (-r, 0.0), steps=8)
    assert body(doc, "hull")["angle"] == pytest.approx(math.pi / 2)
    doc.undo_stack.undo()
    assert body(doc, "hull")["angle"] == 0.0
    # 수식 각도는 돌리지 않는다
    doc.set_value(("bodies", doc.find(("bodies", "hull")), "angle"), "=0.1")
    doc.select([("bodies", "hull")])
    h = c.rotate_handle(c.scene().body("hull"))  # type: ignore[arg-type]
    drag(c, h, (-r, 0.0), steps=8)
    assert body(doc, "hull")["angle"] == "=0.1"


def test_delete_and_nudge(qtbot: Any) -> None:
    c = make_canvas(qtbot, three_bodies(), scale=40.0)
    doc = c.doc
    click(c, 4.0, 0.0)
    key(c, Qt.Key.Key_Right)
    key(c, Qt.Key.Key_Up)
    assert body(doc, "block")["pos"] == [4.5, 0.5]
    key(c, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
    assert body(doc, "block")["pos"] == [-0.5, 0.5]
    c.set_snap(False)
    key(c, Qt.Key.Key_Down)
    assert body(doc, "block")["pos"] == [-0.5, 0.4]
    n = doc.undo_stack.count()
    key(c, Qt.Key.Key_Delete)
    assert doc.item(("bodies", "block")) is None and doc.undo_stack.count() == n + 1
    doc.undo_stack.undo()
    assert doc.item(("bodies", "block")) is not None


def test_formula_locked_fields_are_not_dragged(qtbot: Any) -> None:
    doc = three_bodies()
    i = doc.find(("bodies", "block"))
    doc.set_value(("bodies", i, "pos"), ["=2 * 2", 0.0])  # x만 수식
    c = make_canvas(qtbot, doc, scale=40.0)
    drag(c, (4.0, 0.0), (6.0, 2.0))
    assert body(doc, "block")["pos"] == ["=2 * 2", 2.0]  # y만 옮긴다
    doc.set_value(("bodies", i, "pos"), ["=4", "=0"])
    n = doc.undo_stack.count()
    drag(c, (4.0, 0.0), (6.0, 2.0))
    assert body(doc, "block")["pos"] == ["=4", "=0"] and doc.undo_stack.count() == n
    assert c.status_text  # 끌 수 없다는 안내
    # 착륙선: 다리(관절 배치)와 수식 시작 위치는 끌리지 않는다
    lander = EnvDocument()
    lander.from_builtin("lander.evoenv")
    c2 = make_canvas(qtbot, lander, scale=60.0)
    leg = c2.scene().body("leg_l")
    assert leg is not None
    c2.center = (leg.pose[0], leg.pose[1])
    before = copy_data(lander)
    drag(c2, (leg.pose[0], leg.pose[1]), (leg.pose[0] + 2, leg.pose[1] + 1))
    assert lander.data == before
    sx, sy, _ = c2.scene().spawn
    c2.set_tool("spawn")
    click(c2, sx + 3, sy)
    assert lander.data["agent"] == before["agent"]


def copy_data(doc: EnvDocument) -> dict[str, Any]:
    import copy

    return copy.deepcopy(doc.data)


def test_drag_joint_anchor_thruster_and_checkpoint_end(qtbot: Any) -> None:
    doc = agent_doc()
    T.add_box(doc, "agent", (4.0, 0.0), 0.5, 2.0)
    T.add_joint(doc, "pivot", "hull", "block", (1.0, 0.0), (0.0, 1.0))  # 상자가 관절로 놓인다
    T.add_thruster(doc, "hull", (0.0, -0.5), math.pi / 2)
    T.add_checkpoints(doc, [((5.0, -3.0), (5.0, 3.0))])
    doc.select([])
    c = make_canvas(qtbot, doc, scale=40.0)
    drag(c, (1.0, 0.0), (0.5, 0.5))  # 관절 앵커
    j = doc.item(("joints", "joint"))
    assert j["anchor_a"] == [0.5, 0.5] and j["anchor_b"] == [0.0, 1.0]  # b는 관절로 놓였으니 b 앵커는 그대로
    drag(c, (0.0, -0.5), (-0.5, -0.5))  # 추진기 위치
    assert doc.item(("actuators", "engine"))["at"] == [-0.5, -0.5]
    t = next(x for x in c.scene().thrusters)
    _base, tail = c.thruster_points(t)  # type: ignore[misc]
    drag(c, tail, (-3.0, -0.5), steps=6)  # 화살표 꼬리(분사)를 왼쪽으로: 힘은 오른쪽
    assert doc.item(("actuators", "engine"))["dir"] == 0.0
    drag(c, (-0.5 - 0.8, -0.5), (2.0, -0.5), steps=6)
    assert doc.item(("actuators", "engine"))["dir"] == pytest.approx(math.pi)
    drag(c, (5.0, 3.0), (6.0, 3.0))  # 체크포인트 선 끝점
    assert doc.item(("checkpoints", "cp"))["lines"] == [[[5.0, -3.0], [6.0, 3.0]]]


# ---------------------------------------------------------------------------
# 팔레트
# ---------------------------------------------------------------------------


def test_palette(qtbot: Any) -> None:
    pal = Palette()
    qtbot.addWidget(pal)
    assert TOOL_IDS == T.TOOL_IDS and len(set(TOOL_IDS)) == len(TOOL_IDS)
    assert set(T.TOOL_LABELS) == set(TOOL_IDS)
    c = make_canvas(qtbot)
    pal.connect_canvas(c)
    got: list[str] = []
    pal.tool_changed.connect(got.append)
    pal.buttons["box"].click()
    assert got[-1] == "box" and c.tool_id == "box" and pal.buttons["box"].isChecked()
    pal.buttons["joint_rot_spring"].click()
    assert c.tool_id == "joint_rot_spring" and not pal.buttons["box"].isChecked()
    QTest.keyClick(pal, Qt.Key.Key_Escape)
    assert c.tool_id == "select" and pal.buttons["select"].isChecked()
    # 캔버스에서 바꾸면 팔레트가 따라간다
    c.set_tool("spawn")
    assert pal.buttons["spawn"].isChecked() and pal.tool_id == "spawn"
    # 바로 실행 도구: 부품을 더하고 선택 도구로
    pal.buttons["terrain_auto"].click()
    assert c.doc.items("generators") and pal.buttons["select"].isChecked() and c.tool_id == "select"
    pal.owner_world.setChecked(True)
    assert c.new_owner == "world"
    pal.snap_step.setValue(0.25)
    pal.snap_check.setChecked(False)
    assert c.snap_step == 0.25 and not c.snap_enabled
    for t in TOOL_IDS:  # 모든 도구가 만들어진다
        c.set_tool(t)
    pal.grab()


# ---------------------------------------------------------------------------
# 처음부터 다시 만들기
# ---------------------------------------------------------------------------


def _normalized(section: list[Any]) -> list[dict[str, Any]]:
    return [x.model_dump(mode="json", by_alias=True) for x in section]


def test_rebuild_lander_agent_with_tools(qtbot: Any) -> None:
    """착륙선 에이전트(본체, 다리 둘: 사각형 + 원 발, 회전 스프링 관절 둘과 place_by)를 도구로 만든다.
    정확한 수치와 수식은 사용자가 속성 패널에 적는 것이라 여기서는 set_value로 채운다."""
    doc = EnvDocument()
    c = make_canvas(qtbot, doc, center=(0.0, 14.0), scale=80.0, snap=0.025)
    # 시작 위치
    c.set_tool("spawn")
    click(c, 0.0, 15.0)
    assert doc.get(("agent", "spawn", "y")) == 15.0
    W = lambda x, y: (x, y + 15.0)  # noqa: E731 (에이전트 좌표 -> 세계 좌표)
    # 본체와 다리 두 개(사각형), 다리 끝에 원 발(Ctrl: 선택한 물체에 모양 더하기)
    c.set_tool("box")
    drag(c, W(-1.0, -0.5), W(1.0, 0.5))
    hull_id = last(doc, "bodies")["id"]
    legs = []
    for x in (-3.0, 3.0):
        c.set_tool("box")
        drag(c, W(x - 0.075, -1.8), W(x + 0.075, -0.6))
        legs.append(last(doc, "bodies")["id"])
        c.set_tool("circle")
        drag(c, W(x, -1.8), W(x + 0.125, -1.8), Qt.KeyboardModifier.ControlModifier)
    leg = body(doc, legs[0])
    assert leg["pos"] == [-3.0, -1.2] and leg["shapes"][0] == {"kind": "box", "w": 0.15, "h": 1.2}
    assert leg["shapes"][1] == {"kind": "circle", "r": 0.125, "offset": [0.0, -0.6]}
    # 회전 스프링 관절: 본체 아래 모서리 근처 -> 다리 위 끝. 다리는 place_by로 놓인다
    c.set_tool("joint_rot_spring")
    joints = []
    for x, leg_id in ((-0.8, legs[0]), (0.8, legs[1])):
        click(c, *W(x, -0.5))
        click(c, *W(3.0 if x > 0 else -3.0, -0.6))
        joints.append(last(doc, "joints")["id"])
        assert body(doc, leg_id)["place_by"] == joints[-1]
    assert doc.item(("joints", joints[0]))["anchor_a"] == [-0.8, -0.5]
    assert doc.item(("joints", joints[0]))["anchor_b"] == [0.0, 0.6]
    # 이름과 수치(속성 패널에서 할 일)
    for old, new in ((hull_id, "hull"), (legs[0], "leg_l"), (legs[1], "leg_r"), (joints[0], "j_leg_l"),
                     (joints[1], "j_leg_r")):
        assert doc.rename(("bodies" if old in (hull_id, *legs) else "joints", old), new)
    lander = parse_env((paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8"))
    for p in lander.params:
        doc.add("params", p.model_dump(mode="json"))

    def setv(ref: tuple[str, str], path: tuple[Any, ...], value: Any) -> None:
        doc.set_value((*doc.item_path(ref), *path), value)  # type: ignore[misc]

    setv(("bodies", "hull"), ("label",), "본체")
    setv(("bodies", "hull"), ("tags",), ["hull"])
    setv(("bodies", "hull"), ("drag",), 5.0)
    for side, label in (("l", "왼다리"), ("r", "오른다리")):
        ref = ("bodies", f"leg_{side}")
        setv(ref, ("label",), label)
        setv(ref, ("tags",), ["leg"])
        setv(ref, ("material",), {"density": 20.0, "friction": 1.0})
        setv(ref, ("shapes", 0, "h"), "=p.leg_len")
        setv(ref, ("shapes", 1, "r"), 0.12)
        setv(ref, ("shapes", 1, "offset"), [0.0, "=-p.leg_len / 2"])
        setv(ref, ("pos",), [0.0, 0.0])  # place_by 물체의 pos는 쓰이지 않는다(원본과 같게 정리)
        jref = ("joints", f"j_leg_{side}")
        sign = "-" if side == "l" else ""
        setv(jref, ("label",), f"{label} 관절")
        setv(jref, ("anchor_b",), [0.0, "=p.leg_len / 2"])
        setv(jref, ("rest_angle",), f"={sign}p.leg_spread")
        setv(jref, ("limit",), [f"={sign}p.leg_spread - 0.35", f"={sign}p.leg_spread + 0.35"])
    env = parse_env(doc.data)
    assert _normalized(env.bodies) == _normalized(lander.bodies)
    assert _normalized(env.joints) == _normalized(lander.joints)
    # 나머지 부품도 도구로: 지형, 추진기 셋, 연료 탱크, 센서
    c.set_tool("terrain_auto")
    c.set_tool("thruster")
    for at in ((0.0, -0.5), (-1.0, 0.2), (1.0, 0.2)):
        click(c, *W(*at))
    main, side_l, side_r = [a for a in doc.items("actuators") if a["kind"] == "thruster"]
    assert main["dir"] == pytest.approx(math.pi / 2) and side_l["dir"] == 0.0
    assert side_r["dir"] == pytest.approx(math.pi)
    assert [main["keys"], side_l["keys"], side_r["keys"]] == [["Up", "W"], ["Right", "D"], ["Left", "A"]]
    c.set_tool("fuel_tank")
    click(c, *W(0.3, 0.1))
    assert all(a.get("tank") == "tank" for a in doc.items("actuators") if a["kind"] == "thruster")
    c.set_tool("sensor_body_state")
    click(c, *W(0.3, 0.1))
    c.set_tool("sensor_contact")
    for leg_id in ("leg_l", "leg_r"):
        b = c.scene().body(leg_id)
        assert b is not None
        click(c, b.pose[0], b.pose[1])
        assert last(doc, "sensors")["body"] == leg_id and last(doc, "sensors")["with"] == ["tag.terrain"]
    c.set_tool("sensor_fuel")
    click(c, *W(0.3, 0.1))
    c.set_tool("sensor_target")
    click(c, *W(0.3, 0.1))
    pad = c.scene().region("terrain.pad")
    assert pad is not None
    c.center, c.scale = (0.0, 7.0), 20.0
    click(c, (pad.points[0][0] + pad.points[1][0]) / 2, pad.points[0][1] + 1.0)
    assert last(doc, "sensors")["target"] == "terrain.pad"
    for old, new in (("engine", "main"), ("engine_1", "side_l"), ("engine_2", "side_r")):
        assert doc.rename(("actuators", old), new)
    spec = doc.spec()
    assert spec is not None, [str(i) for i in doc.run_check() if i.severity == "error"]
    assert list(spec.actuator_channels) == list(compile_env(lander).actuator_channels)
    assert sorted(s.kind for s in spec.file.sensors) == sorted(s.kind for s in lander.sensors)
    # 미리보기의 다리 자세가 원본 착륙선과 같은 모양(본체 기준 상대 자세)
    scene = c.scene()
    ref_doc = EnvDocument()
    ref_doc.from_builtin("lander.evoenv")
    ref_scene = build_scene(ref_doc)

    def rel(sc: Any, bid: str) -> tuple[float, float, float]:
        h, b = sc.body("hull"), sc.body(bid)
        x, y = T.to_local(h.pose, (b.pose[0], b.pose[1]))
        return (x, y, b.pose[2] - h.pose[2])

    for leg_id in ("leg_l", "leg_r"):
        assert approx(rel(scene, leg_id), rel(ref_scene, leg_id), 1e-9)


def test_rebuild_track_car_with_tools(qtbot: Any) -> None:
    doc = EnvDocument()
    doc.set_value(("world", "gravity"), [0.0, 0.0])
    c = make_canvas(qtbot, doc, center=(0.0, 0.0), scale=60.0, snap=0.1)
    c.set_tool("track_auto")
    c.set_tool("box")
    drag(c, (-1.2, -0.6), (1.2, 0.6))
    doc.rename((("bodies", last(doc, "bodies")["id"])), "car")
    c.set_tool("vehicle")
    click(c, 0.0, 0.0)
    c.set_tool("sensor_rays")
    click(c, 1.2, 0.0)
    c.set_tool("sensor_body_state")
    click(c, -0.5, 0.0)
    ray = next(s for s in doc.items("sensors") if s["kind"] == "rays")
    st = next(s for s in doc.items("sensors") if s["kind"] == "body_state")
    drive = next(a for a in doc.items("actuators") if a["kind"] == "vehicle")
    assert ray["origin"] == [1.2, 0.0] and ray["center"] == 0.0 and ray["hits"] == ["tag.wall"]
    assert st["fields"] == ["v_fwd", "v_lat"] and st["frame"] == "body"
    assert drive["keys_forward"] == ["Up", "W"] and drive["wheelbase"] == pytest.approx(1.92)
    for k, v in (("x", "=gen.track.start.x"), ("y", "=gen.track.start.y"), ("angle", "=gen.track.start.angle")):
        doc.set_value(("agent", "spawn", k), v)
    spec = doc.spec()
    assert spec is not None
    assert spec.track_set_size == 5 and list(spec.actuator_channels) == ["drive.throttle", "drive.steer"]
    scene = c.scene()
    assert scene.exact and len(scene.rays[0].lines) == 7 and any(a < 1.0 for _o, _e, a in scene.rays[0].lines)

"""v2.0 에디터 (설계 5.1, 8.3 T37~T44): 움직임과 공기 도구, 길·날개·점프 그림, 미리보기 시각, 기본 보기,
속성 글, 겹침 검사, 검사 탭의 안내(info), 뇌 탭의 「은닉층 없음」.

도구는 캔버스에 마우스·키 입력을 넣어(화면 좌표는 world_to_screen) 문서가 바뀌는지 본다. 조작 함수(make_moving 등)는
설계 7.3의 서명 그대로 화면 없이도 부른다.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QPushButton, QStyle

from evolab import paths
from evolab.app import strings as S
from evolab.app.editor import tools as T
from evolab.app.editor.canvas import EditorCanvas, PreviewBar, build_scene, row_window
from evolab.app.editor.document import EnvDocument
from evolab.app.editor.forms import (
    UnionGroup,
    info_text,
    jump_height_text,
    motion_plan_of,
    path_info_text,
    wing_area_text,
)
from evolab.app.editor.palette import GROUP_LABELS, Palette
from evolab.app.editor.properties import PropertyPanel
from evolab.app.editor.tabs import BottomTabs
from evolab.app.editor.view import EditorView, builtin_menu_items
from evolab.spec import model as m
from evolab.spec.io import dump_env, env_hash, parse_env
from evolab.world.physics2d.motion import MotionPlan, pose_at

NO = Qt.KeyboardModifier.NoModifier
LEFT = Qt.MouseButton.LeftButton


# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------


def make_canvas(qtbot: Any, doc: EnvDocument, center: tuple[float, float] = (0.0, 0.0), scale: float = 40.0,
                snap: float = 0.5, size: tuple[int, int] = (1000, 700)) -> EditorCanvas:
    c = EditorCanvas(doc)
    qtbot.addWidget(c)
    c.resize(*size)
    c.show()
    c.scene()
    c.center = center
    c.scale = scale
    c.set_snap(True, snap)
    return c


def pt(c: EditorCanvas, x: float, y: float) -> QPoint:
    p = c.world_to_screen(x, y)
    return QPoint(round(p.x()), round(p.y()))


def click(c: EditorCanvas, x: float, y: float) -> None:
    QTest.mouseClick(c, LEFT, NO, pt(c, x, y))


def drag(c: EditorCanvas, a: tuple[float, float], b: tuple[float, float], steps: int = 4) -> None:
    QTest.mousePress(c, LEFT, NO, pt(c, *a))
    for k in range(1, steps + 1):
        f = k / steps
        QTest.mouseMove(c, pt(c, a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    QTest.mouseRelease(c, LEFT, NO, pt(c, *b))


def body(doc: EnvDocument, bid: str) -> dict[str, Any]:
    item = doc.item(("bodies", bid))
    assert item is not None, bid
    return item


def base_doc() -> EnvDocument:
    """시작 위치 (0, 0)의 새 문서: 에이전트 본체 hull(2 x 1, 가운데 (0, 0)), 세계 사각형 block(2 x 1, 가운데 (5, 0))."""
    doc = EnvDocument()
    T.add_box(doc, "agent", (0.0, 0.0), 2.0, 1.0)
    doc.rename(("bodies", "block"), "hull")
    T.add_box(doc, "world", (5.0, 0.0), 2.0, 1.0)
    return doc


def example(name: str) -> EnvDocument:
    """내장 예제(점프 게임, 글라이더)를 연 문서."""
    doc = EnvDocument()
    doc.from_builtin(name)
    return doc


def icon_image(icon: Any) -> QImage:
    return icon.pixmap(16, 16).toImage()


# ---------------------------------------------------------------------------
# 조작 함수 (설계 7.3 서명)
# ---------------------------------------------------------------------------


def test_make_moving_turns_a_world_body_into_a_path_and_undoes_in_one_step(qtbot: Any) -> None:
    doc = base_doc()
    before = json.dumps(doc.data, sort_keys=True)
    ref = T.make_moving(doc, "block", [(5.0, 2.0), (8.0, 2.0)])
    assert ref == ("bodies", "block") and doc.selection == [ref]
    b = body(doc, "block")
    assert b["type"] == "kinematic"
    assert b["motion"] == {"kind": "path", "speed": 1.0, "mode": "pingpong",
                           "points": [[0.0, 0.0], [0.0, 2.0], [3.0, 2.0]]}
    # 다시 그리면 점만 바뀌고 속력·방식은 남는다
    doc.set_value((*doc.item_path(ref), "motion", "speed"), 2.5)  # type: ignore[misc]
    T.make_moving(doc, "block", [(5.0, -3.0)])
    assert body(doc, "block")["motion"]["points"] == [[0.0, 0.0], [0.0, -3.0]]
    assert body(doc, "block")["motion"]["speed"] == 2.5
    for _ in range(3):
        doc.undo_stack.undo()
    assert json.dumps(doc.data, sort_keys=True) == before
    # 흔들기
    T.make_moving(doc, "block", [(5.0, 1.5)], kind="wave")
    assert body(doc, "block")["motion"] == {"kind": "wave", "period": 2.0, "amplitude": [0.0, 1.5]}
    assert not [i for i in doc.run_check() if i.severity == "error"]


def test_make_moving_refuses_agent_bodies_few_points_and_too_many(qtbot: Any) -> None:
    doc = base_doc()
    with pytest.raises(T.ToolError, match=S.EDC_PATH_WORLD_ONLY):
        T.make_moving(doc, "hull", [(1.0, 1.0)])
    with pytest.raises(T.ToolError, match=S.EDC_ERR_LINE_FEW):
        T.make_moving(doc, "block", [(5.0, 0.0)])  # 물체 자리와 같은 점뿐
    with pytest.raises(T.ToolError, match=S.EDC_PATH_TOO_MANY):
        T.make_moving(doc, "block", [(5.0 + k, 1.0 + (k % 2)) for k in range(T.MAX_PATH_POINTS)])
    T.make_moving(doc, "block", [(5.0 + k, 1.0 + (k % 2)) for k in range(T.MAX_PATH_POINTS - 1)])
    assert len(body(doc, "block")["motion"]["points"]) == T.MAX_PATH_POINTS
    i = doc.find(("bodies", "block"))
    doc.set_value(("bodies", i, "pos"), ["=1 + 4", 0.0])
    with pytest.raises(T.ToolError, match=S.EDC_LOCKED_FORMULA):
        T.make_moving(doc, "block", [(1.0, 1.0)])


def test_add_obstacles_jump_aero_and_wing(qtbot: Any) -> None:
    doc = base_doc()
    gref = T.add_obstacles(doc)
    gen = doc.item(gref)
    assert gen is not None and gen["kind"] == "obstacles" and gen["seed"] == "=r.oseed"
    assert gen["gap"] == [1.0, 2.4] and gen["accel"] == 0.25 and gen["air_chance"] == 0.35 and gen["count"] == 40
    assert [r["id"] for r in doc.items("randomize")] == ["oseed"]
    assert doc.items("randomize")[0]["dist"] == "int"
    doc.undo_stack.undo()  # 생성기와 시드 값이 한 단계
    assert not doc.items("generators") and not doc.items("randomize")
    T.add_obstacles(doc)
    T.add_obstacles(doc)  # 두 번째는 있는 oseed를 같이 쓴다
    assert [r["id"] for r in doc.items("randomize")] == ["oseed"]
    assert [g["id"] for g in doc.items("generators")] == ["obs", "obs_1"]

    # 점프: 기본 위쪽, 땅은 있는 태그, Space 키
    i = doc.find(("bodies", "block"))
    doc.set_value(("bodies", i, "tags"), ["ground"])
    jref = T.add_jump(doc, "hull")
    jump = doc.item(jref)
    assert jump is not None
    assert (jump["kind"], jump["speed"], jump["cooldown"], jump["dir"]) == ("jump", 6.0, 0.25, math.pi / 2)
    assert jump["ground"] == ["tag.ground"] and jump["keys"] == ["Up", "W", "Space"]
    assert T.add_jump(doc, "hull", math.pi / 4)
    assert doc.items("actuators")[-1]["keys"] == []  # 앞 점프가 키를 다 썼다
    with pytest.raises(T.ToolError, match=S.EDC_ACTUATOR_AGENT_ONLY):
        T.add_jump(doc, "block")
    # 공기역학
    aref = T.add_aero(doc, "hull")
    assert doc.item(aref) == {"kind": "aero", "id": "air", "label": S.EDC_TOOL_AERO, "body": "hull"}
    with pytest.raises(T.ToolError, match=S.EDC_ACTUATOR_AGENT_ONLY):
        T.add_aero(doc, "block")
    # 날개: 앞전 -> 뒷전(물체 좌표), 붙임각은 끈 방향의 반대
    n = len(body(doc, "hull")["shapes"])
    T.add_wing(doc, "hull", (0.5, 0.0), (-1.0, -0.5))
    w = body(doc, "hull")["shapes"][n]
    assert w["kind"] == "wing" and w["le"] == [0.5, 0.0] and w["thick"] == 0.04
    assert w["chord"] == pytest.approx(math.hypot(1.5, 0.5), abs=1e-6)
    assert w["angle"] == pytest.approx(math.atan2(0.5, 1.5), abs=1e-6)
    doc.undo_stack.undo()
    assert len(body(doc, "hull")["shapes"]) == n
    with pytest.raises(T.ToolError, match=S.EDC_ACTUATOR_AGENT_ONLY):
        T.add_wing(doc, "block", (0.0, 0.0), (-1.0, 0.0))
    errors = [i for i in doc.run_check() if i.severity == "error"]
    assert not errors, errors


def test_jump_ground_falls_back_to_terrain_and_obstacle_tags_are_world_tags(qtbot: Any) -> None:
    doc = base_doc()
    T.add_terrain_line(doc, [(-10.0, -1.0), (10.0, -1.0)])
    assert T.jump_ground(doc) == ["tag.terrain"]
    T.add_obstacles(doc)
    assert {"obstacle", "air"} <= set(T.world_tags(doc))
    rays = T.add_rays(doc, "hull", (0.0, 0.0))
    assert "tag.obstacle" in (doc.item(rays) or {})["hits"]


# ---------------------------------------------------------------------------
# T37: 길 도구
# ---------------------------------------------------------------------------


def test_t37_path_tool_world_body_three_points_enter_and_one_undo(qtbot: Any) -> None:
    doc = base_doc()
    c = make_canvas(qtbot, doc)
    before = json.dumps(doc.data, sort_keys=True)
    c.set_tool("path")
    assert c.status_text == S.EDC_HINT_PATH
    click(c, 5.0, 0.0)  # 세계 물체
    assert c.tool.busy()
    for p in ((5.0, 2.0), (8.0, 2.0), (8.0, 0.0)):
        click(c, *p)
    QTest.keyClick(c, Qt.Key.Key_Return)
    b = body(doc, "block")
    assert b["type"] == "kinematic"
    assert b["motion"]["points"] == [[0.0, 0.0], [0.0, 2.0], [3.0, 2.0], [3.0, 0.0]]
    assert b["motion"]["mode"] == "pingpong" and b["motion"]["speed"] == 1.0
    assert not c.tool.busy()
    doc.undo_stack.undo()
    assert json.dumps(doc.data, sort_keys=True) == before
    # 에이전트 물체를 누르면 안내만
    click(c, 0.0, 0.0)
    assert c.status_text == S.EDC_PATH_WORLD_ONLY and not c.tool.busy()
    # 두 번 누르기로 끝, Backspace로 마지막 점 지우기, Esc로 취소
    click(c, 5.0, 0.0)
    click(c, 5.0, 3.0)
    click(c, 6.0, 3.0)
    QTest.keyClick(c, Qt.Key.Key_Backspace)
    QTest.mouseDClick(c, LEFT, NO, pt(c, 7.0, 4.0))
    assert body(doc, "block")["motion"]["points"] == [[0.0, 0.0], [0.0, 3.0], [2.0, 4.0]]
    click(c, 5.0, 0.0)
    click(c, 9.0, 0.0)
    QTest.keyClick(c, Qt.Key.Key_Escape)
    assert not c.tool.busy() and body(doc, "block")["motion"]["points"] == [[0.0, 0.0], [0.0, 3.0], [2.0, 4.0]]
    c.grab()


def test_path_tool_double_click_on_a_body_only_picks_it(qtbot: Any) -> None:
    """물체를 두 번 눌러 고르면 고른 채로 둔다(점 없음, 문서 그대로). 전에는 고르자마자 한 점(0.5m)짜리 길로 끝났다."""
    doc = base_doc()
    c = make_canvas(qtbot, doc)
    before = json.dumps(doc.data, sort_keys=True)
    c.set_tool("path")
    QTest.mouseClick(c, LEFT, NO, pt(c, 5.6, 0.2))  # 실제 두 번 누르기: 누름·뗌·두 번 누름·뗌
    QTest.mouseDClick(c, LEFT, NO, pt(c, 5.6, 0.2))
    assert c.tool.busy() and body(doc, "block").get("type", "static") == "static"
    assert json.dumps(doc.data, sort_keys=True) == before
    p = pt(c, 5.6, 0.2)  # 두 번째 누름이 누름으로도 오는 순서: 같은 자리에 찍힌 점도 지운다
    c.tool.cancel()
    QTest.mousePress(c, LEFT, NO, p)
    QTest.mouseRelease(c, LEFT, NO, p)
    QTest.mousePress(c, LEFT, NO, p)
    QTest.mouseDClick(c, LEFT, NO, p)
    QTest.mouseRelease(c, LEFT, NO, p)
    assert c.tool.busy() and json.dumps(doc.data, sort_keys=True) == before
    # 고른 뒤 다른 자리를 누르고 두 번 누르면 지금처럼 끝(고른 자리 가까이 돌아와도)
    click(c, 5.0, 3.0)
    QTest.mouseDClick(c, LEFT, NO, pt(c, 5.6, 0.2))
    b = body(doc, "block")
    assert b["type"] == "kinematic" and b["motion"]["points"] == [[0.0, 0.0], [0.0, 3.0], [0.5, 0.0]]
    assert not c.tool.busy()


def test_path_points_are_drawn_and_draggable_with_the_select_tool(qtbot: Any) -> None:
    doc = base_doc()
    T.make_moving(doc, "block", [(5.0, 2.0), (8.0, 2.0)])
    c = make_canvas(qtbot, doc)
    scene = c.scene()
    b = scene.body("block")
    assert b is not None and b.kinematic and b.motion is not None and b.path_count == 3
    assert b.path_points() == [(5.0, 0.0), (5.0, 2.0), (8.0, 2.0)]
    c.set_tool("select")
    doc.select([("bodies", "block")])
    hit = c.pick((8.0, 2.0))
    assert hit is not None and hit.part == "path_pt" and hit.index == 2
    drag(c, (8.0, 2.0), (9.0, 3.0))
    assert body(doc, "block")["motion"]["points"] == [[0.0, 0.0], [0.0, 2.0], [4.0, 3.0]]
    doc.undo_stack.undo()
    assert body(doc, "block")["motion"]["points"][2] == [3.0, 2.0]
    c.grab()


# ---------------------------------------------------------------------------
# T38: 점프·날개·공기역학·장애물 줄 도구
# ---------------------------------------------------------------------------


def test_t38_jump_tool_direction_snaps_to_15_degrees(qtbot: Any) -> None:
    doc = base_doc()
    c = make_canvas(qtbot, doc)
    c.set_tool("jump")
    click(c, 0.0, 0.0)  # 누르기만: 위쪽
    assert doc.items("actuators")[-1]["dir"] == pytest.approx(math.pi / 2)
    c.set_tool("jump")
    drag(c, (0.0, 0.0), (2.0 * math.cos(0.87), 2.0 * math.sin(0.87)))  # 약 50도 -> 45도
    d = doc.items("actuators")[-1]["dir"]
    assert d == pytest.approx(math.pi / 4)
    assert abs(d / (math.pi / 12) - round(d / (math.pi / 12))) < 1e-9
    n = len(doc.items("actuators"))
    click(c, 5.0, 0.0)  # 세계 물체
    assert c.status_text == S.EDC_ACTUATOR_AGENT_ONLY and len(doc.items("actuators")) == n
    click(c, 0.0, 6.0)  # 빈 곳
    assert c.status_text == S.EDC_ERR_CLICK_AGENT_BODY


def test_t38_wing_tool_le_chord_angle_from_drag(qtbot: Any) -> None:
    doc = base_doc()
    c = make_canvas(qtbot, doc)
    c.set_tool("wing")
    drag(c, (0.5, 0.0), (-1.0, -0.5))
    w = body(doc, "hull")["shapes"][-1]
    assert w["kind"] == "wing" and w["le"] == [0.5, 0.0]
    assert w["chord"] == pytest.approx(math.hypot(1.5, 0.5), abs=1e-6)
    assert w["angle"] == pytest.approx(math.atan2(0.5, 1.5), abs=1e-6)
    # 누르기만: 시위 1m, 붙임각 0(뒷전이 왼쪽)
    click(c, 0.5, 0.0)
    w = body(doc, "hull")["shapes"][-1]
    assert (w["le"], w["chord"], w["angle"]) == ([0.5, 0.0], 1.0, 0.0)
    scene = c.scene()
    hull = scene.body("hull")
    assert hull is not None and len(hull.wing_les) == 2
    # 공기역학 도구, 장애물 줄 도구(바로 실행)
    c.set_tool("aero")
    click(c, 0.0, 0.0)
    assert doc.items("actuators")[-1]["kind"] == "aero"
    c.set_tool("obstacles")
    assert doc.items("generators")[-1]["kind"] == "obstacles" and c.tool_id == "select"
    c.grab()


def test_body_state_tool_on_the_obstacle_row(qtbot: Any) -> None:
    """점프 게임의 「장애물 속도」 센서처럼 몸 상태 센서는 장애물 줄(생성기 물체)에도 단다. 다른 센서는 그대로."""
    doc = example("jump.evoenv")
    c = make_canvas(qtbot, doc, center=(25.0, 1.0), scale=30.0)
    obs = c.scene().body("obs")
    assert obs is not None and obs.polys
    x0, y0, x1, y1 = min(p[0] for p in obs.polys[0]), min(p[1] for p in obs.polys[0]), max(
        p[0] for p in obs.polys[0]), max(p[1] for p in obs.polys[0])
    n = len(doc.items("sensors"))
    c.set_tool("sensor_contact")
    click(c, (x0 + x1) / 2, (y0 + y1) / 2)
    assert len(doc.items("sensors")) == n and c.status_text == S.EDC_ERR_CLICK_BODY
    c.set_tool("sensor_body_state")
    click(c, (x0 + x1) / 2, (y0 + y1) / 2)
    st = doc.items("sensors")[-1]
    assert (st["kind"], st["body"], st["fields"]) == ("body_state", "obs", ["vx", "vy"])
    assert not [i for i in doc.run_check() if i.severity == "error"]


def test_palette_group_and_icons(qtbot: Any) -> None:
    pal = Palette()
    qtbot.addWidget(pal)
    assert ("motion", ["path", "obstacles", "jump", "wing", "aero"]) in T.TOOL_GROUPS
    groups = [g for g, _ in T.TOOL_GROUPS]
    assert groups.index("motion") == groups.index("actuators") + 1
    assert GROUP_LABELS["motion"] == S.EDC_GROUP_MOTION
    for t in ("path", "obstacles", "jump", "wing", "aero"):
        b = pal.buttons[t]
        assert b.text() == T.TOOL_LABELS[t] and b.toolTip() and not b.icon().isNull()
        assert icon_image(b.icon()) != icon_image(pal.buttons["select"].icon())
    assert not pal.buttons["obstacles"].isCheckable() and pal.buttons["path"].isCheckable()
    pal.grab()


# ---------------------------------------------------------------------------
# 캔버스 그림
# ---------------------------------------------------------------------------


def test_examples_draw_motion_wings_jump_and_aero(qtbot: Any) -> None:
    jump = example("jump.evoenv")
    c = make_canvas(qtbot, jump)
    scene = c.scene()
    assert scene.exact and not scene.message
    obs = scene.body("obs")
    assert obs is not None and obs.generated and obs.kinematic and obs.motion is not None and obs.row is not None
    # 장애물 줄은 처음 15초 안에 닿는 것만, 나머지는 「→ 장애물 n개」
    assert 0 < len(obs.shapes) < len(obs.row) and obs.row_hidden == len(obs.row) - len(obs.shapes)
    assert scene.jumps and scene.jumps[0].body == "runner"
    assert ("actuators", "jump") in [bd.ref for bd in scene.badges]
    assert c.has_motion and scene.max_time == 40.0
    c.fit()
    c.grab()
    glider = example("glider.evoenv")
    g = make_canvas(qtbot, glider)
    gs = g.scene()
    assert gs.exact and gs.aero_bodies == {"glider"} and not g.has_motion
    gb = gs.body("glider")
    assert gb is not None and len(gb.wing_les) == 2
    assert ("actuators", "air") in [bd.ref for bd in gs.badges]
    g.grab()


def test_row_window_counts_later_obstacles() -> None:
    from evolab.world.physics2d.episode import ShapeGeom

    row = [ShapeGeom("poly", [(x, 0.0), (x + 1.0, 0.0), (x + 1.0, 1.0), (x, 1.0)]) for x in (5.0, 20.0, 40.0, 80.0)]
    plan = MotionPlan.path((0.0, 0.0), 0.0, [(0.0, 0.0), (-200.0, 0.0)], 2.0, 0.0, 0.0, "once", 0.0)
    keep, later = row_window(row, plan, 0.0)  # 15초 동안 30m
    assert [s.points[0][0] for s in keep] == [5.0, 20.0] and later == 2
    keep, later = row_window(row, plan, 20.0)  # 40m 지남: 지난 15초(10~40m)와 다음 15초(40~70m)
    assert [s.points[0][0] for s in keep] == [20.0, 40.0] and later == 1
    still = MotionPlan.path((0.0, 0.0), 0.0, [(0.0, 0.0), (-200.0, 0.0)], 0.0, 0.0, 0.0, "once", 0.0)
    keep, later = row_window(row, still, 0.0)  # 멈춘 줄도 가장 가까운 하나는 보인다
    assert [s.points[0][0] for s in keep] == [5.0] and later == 3


def test_broken_document_with_kinematic_body_draws_from_numbers(qtbot: Any) -> None:
    doc = base_doc()
    T.make_moving(doc, "block", [(5.0, 3.0)])
    i = doc.find(("bodies", "hull"))
    doc.set_value(("bodies", i, "shapes", 0, "w"), "=p.nope * 2")  # 수식 오류: 미리보기는 고친 사본으로
    j = doc.find(("bodies", "block"))
    doc.set_value(("bodies", j, "motion", "speed"), "=p.nope")  # 움직임 칸도 오류
    assert doc.spec() is None
    scene = build_scene(doc)
    b = scene.body("block")
    assert not scene.exact and b is not None and b.motion is not None
    assert pose_at(b.motion, 0.0)[:2] == (5.0, 0.0)
    c = make_canvas(qtbot, doc)
    c.set_preview_time(1.0)
    c.grab()
    assert c.drawn_poses["block"] == pose_at(b.motion, 1.0)
    # 장애물 줄 생성기에 오류가 있어도(런타임 미리보기에서 빠짐) 같은 생성기로 숫자만으로 그린다
    jump = example("jump.evoenv")
    g = jump.find(("generators", "obs"))
    jump.set_value(("generators", g, "x"), "=p.nope")
    assert jump.spec() is None
    js = build_scene(jump)
    obs = js.body("obs")
    assert obs is not None and obs.generated and obs.motion is not None and obs.row and obs.shapes
    assert min(p[0] for p in obs.shapes[0].points) == pytest.approx(25.0)  # 못 읽는 칸은 모델 기본값(25m)


# ---------------------------------------------------------------------------
# T39: 속성 글
# ---------------------------------------------------------------------------


def test_t39_property_texts(qtbot: Any) -> None:
    data: dict[str, Any] = {"world": {"gravity": [0.0, -9.81]}, "params": [{"id": "c", "value": 0.8, "evolve": True}],
                            "actuators": [{"kind": "aero", "id": "air", "body": "g", "span": 1.5}]}
    assert jump_height_text(data, {"kind": "jump", "speed": 6.0}) == S.EDP_JUMP_HEIGHT.format(h=36 / 19.62, g=9.81)
    assert "1.8m" in str(jump_height_text(data, {"kind": "jump", "speed": 6.0}))
    assert jump_height_text(data, {"kind": "jump"}) == jump_height_text(data, {"kind": "jump", "speed": 6.0})
    assert jump_height_text({"world": {"gravity": [0.0, "=r.g"]}}, {"kind": "jump"}) is None  # 중력이 수식
    assert jump_height_text({"world": {"gravity": [0.0, 0.0]}}, {"kind": "jump"}) is None
    assert wing_area_text(data, {"id": "g"}, {"kind": "wing", "chord": "=p.c"}) == S.EDP_WING_AREA.format(
        a=1.2, c=0.8, s=1.5)
    # 공기역학이 없으면 날개가 공기힘을 받지 않는다: 면적·폭 대신 시위만(공기역학을 붙이라는 안내)
    assert wing_area_text(data, {"id": "other"}, {"kind": "wing", "chord": 2.0}) == S.EDP_WING_CHORD.format(c=2.0)
    assert wing_area_text(data, {"id": "other"}, {"kind": "wing", "chord": "=p.nope"}) is None
    pp = {"kind": "path", "points": [[0, 0], [0, 3]], "speed": 1.0, "mode": "pingpong"}
    assert path_info_text(data, pp) == S.EDP_PATH_INFO.format(length=3.0, period=6.0)
    assert path_info_text(data, {**pp, "mode": "loop"}) == S.EDP_PATH_INFO.format(length=6.0, period=6.0)
    assert path_info_text(data, {**pp, "mode": "once"}) == S.EDP_PATH_LENGTH.format(length=3.0)
    assert path_info_text(data, {**pp, "accel": 0.5}) == S.EDP_PATH_LENGTH.format(length=3.0)
    assert path_info_text(data, {"kind": "wave", "amplitude": [0.0, 1.0], "period": 4.0}) == S.EDP_PATH_INFO.format(
        length=2.0, period=4.0)
    assert path_info_text(data, {**pp, "points": [[0, 0], ["=r.x", 3]]}) is None
    plan = motion_plan_of(data, pp)
    assert isinstance(plan, MotionPlan) and plan.mode == "pingpong"
    # 속성 패널에 보인다
    doc = example("jump.evoenv")
    panel = PropertyPanel(doc)
    qtbot.addWidget(panel)
    panel.show()
    doc.select([("actuators", "jump")])
    rows = panel.form.rows
    assert rows is not None and rows.info_label is not None
    assert rows.info_label.text() == jump_height_text(doc.data, doc.item(("actuators", "jump")))
    i = doc.find(("actuators", "jump"))
    doc.set_value(("actuators", i, "speed"), 5.0)
    assert "1.3m" in rows.info_label.text()
    glider = example("glider.evoenv")
    gpanel = PropertyPanel(glider)
    qtbot.addWidget(gpanel)
    gpanel.show()
    glider.select([("bodies", "glider")])
    path = glider.item_path(("bodies", "glider"))
    assert path is not None
    assert info_text(glider.data, m.Wing, (*path, "shapes", 1)) == S.EDP_WING_AREA.format(a=0.8, c=0.8, s=1.0)
    labels = [w for w in gpanel.form.findChildren(type(rows.info_label)) if w.text().startswith(
        S.EDP_WING_AREA.split("{")[0])]
    assert len(labels) == 2  # 주 날개와 꼬리날개


def test_motion_group_shows_for_kinematic_and_draw_button_starts_the_path_tool(qtbot: Any) -> None:
    doc = base_doc()
    canvas = make_canvas(qtbot, doc)
    panel = PropertyPanel(doc)
    qtbot.addWidget(panel)
    panel.show()
    doc.select([("bodies", "block")])
    rows = panel.form.rows
    assert rows is not None
    motion = rows.fields["motion"]
    assert isinstance(motion, UnionGroup) and motion.isHidden()  # 고정 물체: 「움직임」 묶음이 숨는다
    i = doc.find(("bodies", "block"))
    doc.set_value(("bodies", i, "type"), "kinematic")
    rows = panel.form.rows
    assert rows is not None
    motion = rows.fields["motion"]
    assert isinstance(motion, UnionGroup) and not motion.isHidden() and motion.draw_button is not None
    assert motion.draw_button.text() == S.EDP_PATH_DRAW
    # 단추 -> 캔버스의 길 그리기 도구가 이 물체로 켜진다
    motion.draw_button.click()
    assert canvas.tool_id == "path" and canvas.tool.busy()
    click(canvas, 5.0, 4.0)
    QTest.keyClick(canvas, Qt.Key.Key_Return)
    assert body(doc, "block")["motion"]["points"] == [[0.0, 0.0], [0.0, 4.0]]
    # 묶음 체크를 끄면 움직임이 지워지고, 켜면 기본 길(오른쪽 2m, 왕복)
    group = panel.form.rows.fields["motion"]  # type: ignore[union-attr]
    assert isinstance(group, UnionGroup) and group.rows is not None
    group.box.setChecked(False)  # 사용자가 상자 체크를 끔
    group._on_toggle(False)
    assert body(doc, "block").get("motion") is None
    group = panel.form.rows.fields["motion"]  # type: ignore[union-attr]
    assert isinstance(group, UnionGroup)
    group.box.setChecked(True)
    group._on_toggle(True)
    assert body(doc, "block")["motion"] == {"kind": "path", "points": [[0.0, 0.0], [2.0, 0.0]], "speed": 1.0,
                                            "mode": "pingpong"}
    rows = panel.form.rows
    assert rows is not None
    group = rows.fields["motion"]
    assert isinstance(group, UnionGroup) and group.rows is not None and group.rows.info_label is not None
    assert group.rows.info_label.text() == S.EDP_PATH_INFO.format(length=2.0, period=4.0)
    # 에이전트 물체에는 단추가 없다
    doc.select([("bodies", "hull")])
    hull_motion = panel.form.rows.fields["motion"]  # type: ignore[union-attr]
    assert isinstance(hull_motion, UnionGroup) and hull_motion.draw_button is None and hull_motion.isHidden()
    assert not [b for b in panel.findChildren(QPushButton) if b.text() == S.EDP_PATH_DRAW and b.isVisible()]


# ---------------------------------------------------------------------------
# T40: 미리보기 시각
# ---------------------------------------------------------------------------


def test_t40_preview_time_slider_play_and_reset(qtbot: Any) -> None:
    from evolab.app.editor.canvas import _preview_episode

    doc = example("jump.evoenv")
    c = make_canvas(qtbot, doc)
    bar = PreviewBar(c)
    qtbot.addWidget(bar)
    bar.show()
    assert bar.isVisible() and bar.slider.maximum() == 4000  # 0 ~ 40초, 0.01초 칸
    bar.slider.setValue(200)
    assert c.preview_time == pytest.approx(2.0)
    c.grab()
    ep = _preview_episode(doc.spec())
    plan = next(g.motion for g in ep.geometry().bodies if g.id == "obs")
    assert plan is not None
    assert c.drawn_poses["obs"] == pose_at(plan, 2.0)
    # 그린 장애물 사각형도 그 자리에 있다(줄 좌표 + 2초 동안 민 거리)
    scene = c.scene()
    obs = scene.body("obs")
    assert obs is not None and obs.row is not None
    x2, y2, _ = pose_at(plan, 2.0)
    shown, _later = row_window(obs.row, obs.motion, 2.0)  # type: ignore[arg-type]
    assert shown[0].points[0][0] + x2 == pytest.approx(obs.row[0].points[0][0] - (6.0 * 2 + 0.25 * 4 / 2))
    # ▶ 재생 중에 문서를 고치면 멈추고 0초
    c.play()
    assert c.playing and bar.play_button.text() == "⏸"
    qtbot.waitUntil(lambda: c.preview_time > 2.05, timeout=3000)
    i = doc.find(("actuators", "jump"))
    doc.set_value(("actuators", i, "speed"), 6.5)
    assert not c.playing and c.preview_time == 0.0 and bar.slider.value() == 0
    assert bar.play_button.text() == "▶"
    # 끝까지 가면 멈춘다
    c.set_preview_time(39.99)
    c.play()
    qtbot.waitUntil(lambda: not c.playing, timeout=3000)
    assert c.preview_time == 40.0
    # 정해진 움직임이 없는 환경에서는 줄이 숨는다
    plain = EnvDocument()
    plain.from_builtin("lander.evoenv")
    pc = make_canvas(qtbot, plain)
    pbar = PreviewBar(pc)
    qtbot.addWidget(pbar)
    assert pbar.isHidden()
    pc.play()
    assert not pc.playing


def test_preview_trial_switch_draws_another_obstacle_row_and_resets_on_open(qtbot: Any) -> None:
    """「미리보기 시험 바꾸기」(설계 5.1): 다른 검증 시험의 랜덤 값으로 장애물 줄을 다시 그린다. 문서·해시는 그대로."""
    from evolab.app.editor.canvas import _preview_episode, preview_trial
    from evolab.engine import seeds

    doc = example("jump.evoenv")
    text, digest = dump_env(parse_env(json.dumps(doc.data))), env_hash(doc.spec().file)
    c = make_canvas(qtbot, doc)
    bar = PreviewBar(c)
    qtbot.addWidget(bar)
    assert preview_trial(0) == seeds.display_trial() and c.preview_trial_index == 0
    assert bar.trial_button.text() == S.ED_PREVIEW_TRIAL.format(n=1)
    first = [s.points for s in c.scene().body("obs").row]  # type: ignore[union-attr]
    c.set_preview_time(3.0)
    bar.trial_button.click()
    assert c.preview_trial_index == 1 and bar.trial_button.text() == S.ED_PREVIEW_TRIAL.format(n=2)
    obs = c.scene().body("obs")
    assert obs is not None and obs.row is not None
    assert [s.points for s in obs.row] != first  # 다른 시드의 줄
    ep = _preview_episode(doc.spec(), 1)  # 런타임이 그 시험에서 만드는 줄과 같다
    row = next(g for g in ep.geometry().bodies if g.id == "obs")
    assert [list(s.points) for s in obs.row] == [list(s.points) for s in row.shapes]
    assert c.preview_time == 3.0  # 시각은 그대로
    c.grab()
    assert c.drawn_poses["obs"] == pose_at(row.motion, 3.0)  # type: ignore[arg-type]
    assert dump_env(parse_env(json.dumps(doc.data))) == text and env_hash(doc.spec().file) == digest
    assert not doc.undo_stack.count()
    # 새 파일을 열면 전시 시험으로 돌아간다
    doc.from_builtin("jump.evoenv")
    assert c.preview_trial_index == 0 and bar.trial_button.text() == S.ED_PREVIEW_TRIAL.format(n=1)
    assert [s.points for s in c.scene().body("obs").row] == first  # type: ignore[union-attr]


def test_t40_cached_overlay_frame_is_fast(qtbot: Any) -> None:
    """재생 중 한 프레임(정적 그림 이미지 + 움직이는 물체 덧그리기) < 5ms(오프스크린, 1000 x 700)."""
    doc = example("jump.evoenv")
    c = make_canvas(qtbot, doc, size=(1000, 700))
    c.fit()
    c.fit()  # 장면 전체(장애물이 많이 보이는 쪽)
    img = QImage(c.size(), QImage.Format.Format_ARGB32_Premultiplied)

    def frame_ms(n: int, moving: bool) -> list[float]:
        out = []
        for k in range(n):
            c.preview_time = 1.0 + k * 0.033 if moving else 0.0
            t0 = time.perf_counter()
            c.render(img)
            out.append((time.perf_counter() - t0) * 1000)
        return out

    full = statistics.median(frame_ms(10, moving=False))  # 0초: 캐시 없이 전부 그린다(v1과 같은 길)
    assert c._static_cache is None
    c.play()
    c._play_timer.stop()  # 시각은 직접 옮긴다(재생 상태만)
    c.render(img)  # 캐시를 만든다
    cache = c._static_cache
    cached = statistics.median(frame_ms(30, moving=True))
    assert c._static_cache is cache  # 같은 이미지를 계속 쓴다
    print(f"\neditor frame (jump, 1000x700): full {full:.2f} ms, cached overlay {cached:.2f} ms")
    assert cached < 5.0
    c.pause()


# ---------------------------------------------------------------------------
# T41은 tests/test_editor_rebuild.py(점프 게임·글라이더 갈래)
# T42: 겹침 검사
# ---------------------------------------------------------------------------


def _overlap_doc() -> EnvDocument:
    doc = EnvDocument()
    T.add_box(doc, "world", (0.0, -0.5), 20.0, 1.0)
    doc.rename(("bodies", "block"), "ground")
    for bid, x in (("k1", 5.0), ("k2", 5.5)):  # 서로 겹치고 땅과도 겹치는 정해진 움직임 물체 둘
        T.add_box(doc, "world", (x, 0.0), 2.0, 2.0)
        doc.rename(("bodies", "block"), bid)
        T.make_moving(doc, bid, [(x, 3.0)])
    T.add_box(doc, "world", (-5.0, 0.0), 1.0, 1.0)  # 땅과 겹친 움직이는 세계 물체
    doc.rename(("bodies", "block"), "crate")
    i = doc.find(("bodies", "crate"))
    doc.set_value(("bodies", i, "type"), "dynamic")
    T.add_box(doc, "world", (8.5, 6.0), 1.0, 1.0)  # 에이전트와 반쯤 겹친 정해진 움직임 물체
    doc.rename(("bodies", "block"), "k3")
    T.make_moving(doc, "k3", [(8.5, 9.0)])
    T.set_spawn(doc, 8.0, 6.0)
    T.add_box(doc, "agent", (0.0, 0.0), 1.0, 1.0)
    doc.rename(("bodies", "block"), "hero")
    return doc


def test_t42_overlap_warning_skips_kinematic_pairs(qtbot: Any) -> None:
    doc = _overlap_doc()
    issues = doc.run_check()
    assert not [i for i in issues if i.severity == "error"], issues
    ids = ("crate", "ground", "hero", "k1", "k2", "k3")
    every_pair = {S.ED_CHECK_OVERLAP.format(a=a, b=b) for a in ids for b in ids}
    overlap = sorted(i.message for i in issues if i.message in every_pair)
    assert overlap == sorted([S.ED_CHECK_OVERLAP.format(a="crate", b="ground"),
                              S.ED_CHECK_OVERLAP.format(a="hero", b="k3")]), overlap
    # 정해진 움직임을 끄면(정적끼리) v1처럼 경고가 그대로다 = 이 규칙은 kinematic이 있을 때만
    for bid in ("k1", "k2"):
        i = doc.find(("bodies", bid))
        doc.set_value(("bodies", i, "type"), "dynamic")
        doc.set_value(("bodies", i, "motion"), None)
    msgs = {i.message for i in doc.run_check()}
    assert S.ED_CHECK_OVERLAP.format(a="k1", b="k2") in msgs
    # 점프 게임: 장애물 줄이 땅 위에 있어도 경고 없음
    assert not [i for i in example("jump.evoenv").run_check() if i.severity != "info"]


# ---------------------------------------------------------------------------
# T43: 검사 탭의 안내(info)
# ---------------------------------------------------------------------------


def test_t43_info_issue_is_blue_and_not_counted(qtbot: Any) -> None:
    doc = example("glider.evoenv")
    issues = doc.run_check()
    assert [i.severity for i in issues] == ["info"] and issues[0].message == S.SPEC_BRAINLESS
    tabs = BottomTabs(doc)
    qtbot.addWidget(tabs)
    tabs.show()
    tree = tabs.issues_tab.tree
    assert tree.topLevelItemCount() == 1
    item = tree.topLevelItem(0)
    info_name = S.EDP_ISSUES_SEVERITY["info"]
    assert item.text(0) == info_name
    style = tabs.style()
    assert icon_image(item.icon(0)) == icon_image(style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation))
    i = tabs.indexOf(tabs.issues_tab)
    assert tabs.tabText(i) == S.EDP_TAB_ISSUES  # 개수 없음
    # 경고가 하나 생기면 개수는 1(안내는 빼고), 순서는 오류·경고·안내
    doc.add("bodies", {"id": "lonely", "owner": "agent", "shapes": [{"kind": "box", "w": 0.2, "h": 0.2}],
                       "pos": [0.0, 5.0]})  # 관절로 이어지지 않은 에이전트 물체: 경고
    issues = doc.run_check()
    sev = [x.severity for x in issues]
    assert "warning" in sev and "info" in sev
    assert tabs.tabText(i) == S.EDP_TAB_ISSUES_COUNT.format(n=sev.count("warning") + sev.count("error"))
    shown = [tree.topLevelItem(k).text(0) for k in range(tree.topLevelItemCount())]
    assert shown[-1] == info_name and shown[0] != info_name
    # 에디터 상태 줄: 안내만 있으면 「문제 없음」
    view = EditorView()
    qtbot.addWidget(view)
    view.start_from_builtin("glider.evoenv")
    view.doc.run_check()
    assert view.status.text() == S.ED_STATUS_OK and view.act_play.isEnabled()
    # 속성 패널의 문제 목록도 안내는 파란 i(검사 결과 탭과 같은 아이콘 표)
    from evolab.app.editor.properties import PropertyPanel
    from evolab.spec.compiled import Issue

    panel = PropertyPanel(view.doc)
    qtbot.addWidget(panel)
    view.doc.select([("actuators", "air")])
    where = "actuators[0].span"
    panel._on_checked([Issue("info", where, "i"), Issue("warning", where, "w"), Issue("error", where, "e")])
    got = [icon_image(panel.issue_list.item(k).icon()) for k in range(panel.issue_list.count())]
    want = [icon_image(style.standardIcon(px)) for px in (QStyle.StandardPixmap.SP_MessageBoxInformation,
                                                         QStyle.StandardPixmap.SP_MessageBoxWarning,
                                                         QStyle.StandardPixmap.SP_MessageBoxCritical)]
    assert got == want
    # 캔버스 강조: 안내는 색이 없고 경고를 덮지 않는다(같은 물체에 경고 다음 안내가 와도 경고)
    from evolab.spec.compiled import Issue

    c = view.canvas
    c._on_checked([Issue("warning", "bodies[1]", "w"), Issue("info", "bodies[1]", "i"),
                   Issue("info", "bodies[0]", "i")])
    assert c.issue_severity(("bodies", "glider")) == "warning"
    assert c.issue_severity(("bodies", "ground")) == "info" and c._highlight(("bodies", "ground"))[0] is None


def test_brain_tab_hidden_none_and_brainless_note(qtbot: Any) -> None:
    doc = example("glider.evoenv")
    tabs = BottomTabs(doc)
    qtbot.addWidget(tabs)
    tabs.show()
    brain = tabs.brain_tab
    hidden = brain.form.field("hidden")
    assert hidden is not None and hidden.editor.text() == ""  # type: ignore[attr-defined]
    assert hidden.editor.placeholderText() == S.EDP_HIDDEN_NONE  # type: ignore[attr-defined]
    assert not brain.none_label.isHidden() and brain.none_label.text() == S.EDP_BRAIN_NONE
    assert brain.param_count == 0 and brain.genome_dim == 8
    jump = example("jump.evoenv")
    jtabs = BottomTabs(jump)
    qtbot.addWidget(jtabs)
    jtabs.show()
    assert jtabs.brain_tab.none_label.isHidden() and jtabs.brain_tab.param_count == 209
    # 은닉층을 비우면(없음) 입력이 바로 출력으로: 11 x 1 + 1
    jhidden = jtabs.brain_tab.form.field("hidden")
    assert jhidden is not None
    jhidden.editor.setText("")  # type: ignore[attr-defined]
    jhidden.editor.textEdited.emit("")  # type: ignore[attr-defined]
    assert jump.get(("brain", "hidden")) == []
    jump.run_check()
    assert jtabs.brain_tab.param_count == 12


# ---------------------------------------------------------------------------
# T44: 기본 보기
# ---------------------------------------------------------------------------


def test_t44_view_box_save_clear_fit_cycle_and_hash(qtbot: Any, tmp_path: Path) -> None:
    doc = EnvDocument()
    doc.from_builtin("lander.evoenv")
    h0 = env_hash(parse_env(doc.data))
    text0 = dump_env(parse_env(doc.data))
    c = make_canvas(qtbot, doc)
    c.fit()
    assert c.view_box() is None and c._fit_mode == "full"
    full = (c.center, c.scale)
    c.fit()  # 기본 보기가 없으면 늘 장면 전체
    assert (c.center, c.scale) == full
    c.center, c.scale = (3.0, 12.0), 60.0
    box = c.save_view_box()
    assert c.view_box() == box and doc.data["editor"]["view"] == list(box)
    vx0, vy0, vx1, vy1 = c.visible_box()
    assert box == pytest.approx((vx0, vy0, vx1, vy1), abs=1e-3)
    assert c.status_text == S.ED_VIEW_SAVED
    assert env_hash(parse_env(doc.data)) == h0  # 해시 그대로
    assert '"version": 1' in dump_env(parse_env(doc.data))  # 형식 번호도 그대로(v1 파일)
    # 「전체 보기」 순환: 장면 전체 -> 기본 보기 -> 장면 전체 ...
    c.fit()
    assert c.center == pytest.approx(full[0]) and c.scale == pytest.approx(full[1])
    c.fit()
    assert c.center == pytest.approx((3.0, 12.0), abs=1e-3) and c.scale == pytest.approx(60.0, rel=1e-3)
    c.fit()
    assert c.center == pytest.approx(full[0]) and c.scale == pytest.approx(full[1])
    c.center, c.scale = (-20.0, 5.0), 10.0  # 사용자가 움직이면 다음은 다시 기본 보기
    c.fit()
    assert c.center == pytest.approx((3.0, 12.0), abs=1e-3)
    # 저장하고 다시 열면 기본 보기로 시작
    out = tmp_path / "lander_view.evoenv"
    doc.save(out)
    doc2 = EnvDocument()
    doc2.open(out)
    c2 = make_canvas(qtbot, doc2)
    c2._fit_pending = True
    c2._dirty = True
    c2.scene()
    assert c2.center == pytest.approx((3.0, 12.0), abs=1e-3) and c2.scale == pytest.approx(60.0, rel=1e-3)
    # 지우기(한 단계), 되돌리기
    c.clear_view_box()
    assert c.view_box() is None and "editor" not in doc.data
    assert dump_env(parse_env(doc.data)) == text0
    doc.undo_stack.undo()
    assert c.view_box() == box
    assert env_hash(parse_env(doc.data)) == h0


def test_view_menu_and_builtin_order(qtbot: Any) -> None:
    items = builtin_menu_items()
    names = [n for n, _ in items]
    assert names[:5] == ["lander.evoenv", "track_car.evoenv", "hill_car.evoenv", "jump.evoenv", "glider.evoenv"]
    assert [t for _, t in items][3:5] == [S.ED_BUILTIN_JUMP, S.ED_BUILTIN_GLIDER]
    assert names == [p.name for p in paths.builtin_envs()]
    view = EditorView()
    qtbot.addWidget(view)
    view.resize(1300, 850)
    view.show()
    assert [a.text() for a in view.builtin_menu.actions()] == [t for _, t in items]
    texts = [a.text() for a in view.view_menu.actions() if not a.isSeparator()]
    assert texts == [S.EDC_FIT, S.ED_VIEW_SAVE, S.ED_VIEW_CLEAR]
    view.start_from_builtin("jump.evoenv")
    assert view.canvas.view_box() is not None  # 내장 점프 게임은 기본 보기가 있다
    view.view_menu.aboutToShow.emit()
    assert view.act_view_clear.isEnabled()
    view.act_view_clear.trigger()
    view.view_menu.aboutToShow.emit()
    assert not view.act_view_clear.isEnabled()
    view.act_view_save.trigger()
    assert view.canvas.view_box() is not None
    assert view.preview_bar.isVisibleTo(view)
    view.grab()

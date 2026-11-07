"""에디터 도구: 캔버스에서 부품을 만들고 고치는 조작 (명세: 화면과 조작 > 에디터 > 도구).

두 층으로 나눈다.
- 조작 함수(add_box, add_joint ...): 문서(EnvDocument)만 받아 부품 하나를 더하거나 고친다. 화면 없이 테스트에서도
  부른다. 좌표는 파일에 적히는 좌표계 그대로다: 에이전트 물체는 에이전트 좌표(시작 위치 기준), 세계 물체·영역·
  체크포인트·시작 위치는 세계 좌표, 추진기 위치·관절 앵커·레이 시작점은 그 물체의 좌표.
- 도구 클래스(BoxTool, JointTool ...): 캔버스의 마우스·키 입력을 받아 화면 좌표를 위 좌표계로 바꾸고(미리보기
  자세 기준) 격자에 맞춘 뒤 조작 함수를 부른다.

모든 편집은 EnvDocument를 거친다(되돌리기·실시간 검사). 도구 하나의 동작은 되돌리기 한 단계다.
수식 칸('='로 시작)은 끌기·회전으로 덮어쓰지 않는다(그 칸만 건너뛴다).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QKeyEvent, QPainter, QPen, QPolygonF

from evolab.app import strings as S
from evolab.app.editor.document import EnvDocument, JsonPath, Ref, get_path, set_path
from evolab.world.physics2d import geometry as geo

if TYPE_CHECKING:
    from evolab.app.editor.canvas import CanvasEvent, EditorCanvas, Pick

Point = tuple[float, float]
Pose = tuple[float, float, float]

# ---------------------------------------------------------------------------
# 도구 목록
# ---------------------------------------------------------------------------

TOOL_GROUPS: list[tuple[str, list[str]]] = [
    ("select", ["select"]),
    ("shapes", ["box", "circle", "poly"]),
    ("terrain", ["terrain_line", "terrain_auto", "track_draw", "track_auto"]),
    ("joints", ["joint_pivot", "joint_rot_spring", "joint_weld", "joint_spring"]),
    ("actuators", ["thruster", "fuel_tank", "vehicle"]),
    ("sensors", ["sensor_rays", "sensor_body_state", "sensor_contact", "sensor_target", "sensor_joint_angle",
                 "sensor_fuel", "sensor_time"]),
    ("regions", ["region_goal", "region_hazard", "region_pad", "checkpoints", "spawn"]),
]
TOOL_IDS: list[str] = [t for _, ids in TOOL_GROUPS for t in ids]
INSTANT_TOOLS: frozenset[str] = frozenset({"terrain_auto", "track_auto", "sensor_time"})
"""고르면 바로 부품을 더하고 선택 도구로 돌아가는 도구."""

TOOL_LABELS: dict[str, str] = {
    "select": S.EDC_TOOL_SELECT,
    "box": S.EDC_TOOL_BOX,
    "circle": S.EDC_TOOL_CIRCLE,
    "poly": S.EDC_TOOL_POLY,
    "terrain_line": S.EDC_TOOL_TERRAIN_LINE,
    "terrain_auto": S.EDC_TOOL_TERRAIN_AUTO,
    "track_draw": S.EDC_TOOL_TRACK_DRAW,
    "track_auto": S.EDC_TOOL_TRACK_AUTO,
    "joint_pivot": S.EDC_TOOL_JOINT_PIVOT,
    "joint_rot_spring": S.EDC_TOOL_JOINT_ROT_SPRING,
    "joint_weld": S.EDC_TOOL_JOINT_WELD,
    "joint_spring": S.EDC_TOOL_JOINT_SPRING,
    "thruster": S.EDC_TOOL_THRUSTER,
    "fuel_tank": S.EDC_TOOL_FUEL_TANK,
    "vehicle": S.EDC_TOOL_VEHICLE,
    "sensor_rays": S.EDC_TOOL_SENSOR_RAYS,
    "sensor_body_state": S.EDC_TOOL_SENSOR_BODY_STATE,
    "sensor_contact": S.EDC_TOOL_SENSOR_CONTACT,
    "sensor_target": S.EDC_TOOL_SENSOR_TARGET,
    "sensor_joint_angle": S.EDC_TOOL_SENSOR_JOINT_ANGLE,
    "sensor_fuel": S.EDC_TOOL_SENSOR_FUEL,
    "sensor_time": S.EDC_TOOL_SENSOR_TIME,
    "region_goal": S.EDC_TOOL_REGION_GOAL,
    "region_hazard": S.EDC_TOOL_REGION_HAZARD,
    "region_pad": S.EDC_TOOL_REGION_PAD,
    "checkpoints": S.EDC_TOOL_CHECKPOINTS,
    "spawn": S.EDC_TOOL_SPAWN,
}

TOOL_HINTS: dict[str, str] = {
    "select": S.EDC_HINT_SELECT,
    "box": S.EDC_HINT_BOX,
    "circle": S.EDC_HINT_CIRCLE,
    "poly": S.EDC_HINT_POLY,
    "terrain_line": S.EDC_HINT_TERRAIN_LINE,
    "track_draw": S.EDC_HINT_TRACK_DRAW,
    "joint_pivot": S.EDC_HINT_JOINT,
    "joint_rot_spring": S.EDC_HINT_JOINT,
    "joint_weld": S.EDC_HINT_JOINT,
    "joint_spring": S.EDC_HINT_JOINT,
    "thruster": S.EDC_HINT_THRUSTER,
    "fuel_tank": S.EDC_HINT_FUEL_TANK,
    "vehicle": S.EDC_HINT_VEHICLE,
    "sensor_rays": S.EDC_HINT_SENSOR_BODY,
    "sensor_body_state": S.EDC_HINT_SENSOR_BODY,
    "sensor_contact": S.EDC_HINT_SENSOR_BODY,
    "sensor_target": S.EDC_HINT_SENSOR_TARGET,
    "sensor_joint_angle": S.EDC_HINT_SENSOR_JOINT_ANGLE,
    "sensor_fuel": S.EDC_HINT_SENSOR_FUEL,
    "region_goal": S.EDC_HINT_REGION,
    "region_hazard": S.EDC_HINT_REGION,
    "region_pad": S.EDC_HINT_REGION,
    "checkpoints": S.EDC_HINT_CHECKPOINTS,
    "spawn": S.EDC_HINT_SPAWN,
}

# ---------------------------------------------------------------------------
# 기본값
# ---------------------------------------------------------------------------

AGENT_MATERIAL = {"density": 50.0, "friction": 0.6}
WORLD_MATERIAL = {"density": 50.0, "friction": 0.6}
TERRAIN_FRICTION = 0.9
WORLD_TAG = "obstacle"
TERRAIN_TAG = "terrain"
DEFAULT_BOX = (1.0, 1.0)
DEFAULT_RADIUS = 0.5
DEFAULT_REGION = (4.0, 3.0)
THRUSTER_FORCE = 100.0
THRUSTER_FUEL_PER_S = 1.0
TANK_CAPACITY = 100.0
ROT_SPRING = {"stiffness": 600.0, "damping": 200.0}
DIST_SPRING = {"stiffness": 100.0, "damping": 10.0}
RAYS = {"count": 7, "spread": math.pi, "length": 40.0}
VEHICLE = {"max_accel": 8.0, "max_steer": 0.5, "lateral_grip": 0.9, "drag": 0.5, "max_lateral_accel": 8.0}
VEHICLE_WHEELBASE = 2.0
HEIGHTMAP = {"seed": 0.0, "segment": 1.0, "base_y": 0.0, "amplitude": 4.0, "roughness": 0.5,
             "friction": TERRAIN_FRICTION}
HEIGHTMAP_PAD = {"x": 0.0, "width": 8.0, "region_height": 3.0}
TRACK_GENERATED = {"source": "generated", "set_seed": 1, "set_size": 5, "width": 8.0, "segments": 12,
                   "curvature": 0.5, "length_range": [120.0, 200.0], "checkpoints": 24, "friction": 0.5,
                   "wall_radius": 0.1}
TRACK_DRAWN = {"source": "drawn", "width": 8.0, "checkpoints": 24, "friction": 0.5, "wall_radius": 0.1}
CHAIN_RADIUS = 0.05
THRUSTER_KEYS = [["Up", "W"], ["Right", "D"], ["Left", "A"], ["Down", "S"], ["Q"], ["E"], ["Z"], ["X"], ["C"]]
VEHICLE_KEYS = {"keys_forward": ["Up", "W"], "keys_back": ["Down", "S"], "keys_left": ["Left", "A"],
                "keys_right": ["Right", "D"]}
ANGLE_SNAP = math.pi / 12
"""격자 맞춤이 켜져 있으면 각도를 15도 단위로 맞춘다."""
MAX_POLY_POINTS = 64

JOINT_KINDS = {"joint_pivot": "pivot", "joint_rot_spring": "rot_spring", "joint_weld": "weld",
               "joint_spring": "spring"}
REGION_KINDS = {"region_goal": "goal", "region_hazard": "hazard", "region_pad": "pad"}
_JOINT_LABELS = {"pivot": S.EDC_LABEL_JOINT_PIVOT, "rot_spring": S.EDC_LABEL_JOINT_ROT_SPRING,
                 "weld": S.EDC_LABEL_JOINT_WELD, "spring": S.EDC_LABEL_JOINT_SPRING}
_REGION_IDS = {"goal": "finish", "hazard": "danger", "pad": "landing", "zone": "area"}
"""새 부품 id. 종류 이름(goal, box ...)과 같은 id는 피한다: 이름 바꾸기가 같은 글자의 종류 칸까지 바꿀 수 있다."""
_REGION_LABELS = {"goal": S.EDC_LABEL_REGION_GOAL, "hazard": S.EDC_LABEL_REGION_HAZARD,
                  "pad": S.EDC_LABEL_REGION_PAD, "zone": S.EDC_LABEL_REGION_ZONE}


class ToolError(ValueError):
    """도구 조작을 할 수 없다. 메시지는 strings.py의 한국어 문구."""


# ---------------------------------------------------------------------------
# 좌표·숫자 도우미
# ---------------------------------------------------------------------------


def is_formula(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("=")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def clean(x: float) -> float:
    """파일에 적을 좌표: 부동소수 잡음을 지운다(0.30000000000000004 -> 0.3, -0.0 -> 0.0)."""
    return round(float(x), 6) + 0.0


def snap_value(x: float, step: float) -> float:
    if step <= 0:
        return clean(x)
    return clean(round(x / step) * step)


def snap_angle(a: float) -> float:
    k = round(a / ANGLE_SNAP)
    return 0.0 if k == 0 else k * ANGLE_SNAP


def to_local(frame: Pose, p: Point) -> Point:
    """세계 좌표 p를 frame(x, y, 각도) 좌표로."""
    dx, dy = p[0] - frame[0], p[1] - frame[1]
    return geo.rotate((dx, dy), -frame[2])


def to_world(frame: Pose, p: Point) -> Point:
    return geo.transform(p, (frame[0], frame[1]), frame[2])


def rotate_delta(d: Point, angle: float) -> Point:
    return geo.rotate(d, angle)


def _vec_list(p: Point) -> list[float]:
    return [clean(p[0]), clean(p[1])]


def shifted_vec(value: Any, d: Point, snap: Callable[[float], float] | None = None) -> tuple[Any, bool]:
    """Vec2 값을 d만큼 옮긴 새 값. 수식 칸은 그대로 둔다. (새 값, 하나라도 옮겼나)."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return value, False
    out = list(value)
    moved = False
    for k in (0, 1):
        if is_number(value[k]):
            v = float(value[k]) + d[k]
            out[k] = snap(v) if snap is not None else clean(v)
            moved = True
    return out, moved


def vec_locked(value: Any) -> bool:
    """Vec2의 두 칸이 모두 숫자가 아니면 끌 수 없다."""
    return not (isinstance(value, (list, tuple)) and len(value) == 2 and any(is_number(v) for v in value))


def apply_changes(doc: EnvDocument, label: str, changes: Iterable[tuple[JsonPath, Any]],
                  merge_key: str | None = None) -> None:
    """여러 칸을 한 번에 바꾼다(되돌리기 한 단계)."""
    items = list(changes)

    def mutate(d: dict[str, Any]) -> None:
        for path, value in items:
            set_path(d, path, copy.deepcopy(value))

    doc.edit(label, mutate, merge_key)


def _list(d: dict[str, Any], section: str) -> list[Any]:
    lst = d.get(section)
    if not isinstance(lst, list):
        lst = []
        d[section] = lst
    return lst


def _insert(doc: EnvDocument, section: str, item: dict[str, Any], label: str,
            extra: Callable[[dict[str, Any], str], None] | None = None) -> Ref:
    """부품 하나를 더하고(id가 겹치면 바꾼다) extra로 다른 칸도 함께 고친다. 되돌리기 한 단계."""
    item = copy.deepcopy(item)
    item["id"] = doc.unique_id(str(item.get("id") or section.rstrip("s")))

    def mutate(d: dict[str, Any]) -> None:
        _list(d, section).append(copy.deepcopy(item))
        if extra is not None:
            extra(d, item["id"])

    doc.edit(label, mutate)
    ref = (section, item["id"])
    doc.select([ref])
    return ref


def body_data(doc: EnvDocument, body_id: str) -> dict[str, Any] | None:
    return doc.item(("bodies", body_id))


def _agent_dynamic(doc: EnvDocument, body_id: str) -> bool:
    b = body_data(doc, body_id)
    return b is not None and b.get("owner", "agent") == "agent" and b.get("type", "dynamic") == "dynamic"


def world_tags(doc: EnvDocument) -> list[str]:
    """세계 쪽 모양 태그(세계 물체와 생성기 물체). 레이 기본 대상과 접촉 기본 대상에 쓴다."""
    tags: list[str] = []

    def add(ts: Iterable[str]) -> None:
        for t in ts:
            if isinstance(t, str) and t not in tags:
                tags.append(t)

    for g in doc.items("generators"):
        kind = g.get("kind")
        if kind == "heightmap":
            add(["terrain", "pad"] if g.get("pad") else ["terrain"])
        elif kind == "track":
            add(["wall"])
        elif kind == "hills":
            add(["terrain"])
    for b in doc.items("bodies"):
        if b.get("owner", "agent") == "world":
            add(b.get("tags") or [])
    return tags


def _used_keys(doc: EnvDocument) -> set[str]:
    out: set[str] = set()
    for a in doc.items("actuators"):
        for name in ("keys", "keys_forward", "keys_back", "keys_left", "keys_right", "keys_pos", "keys_neg"):
            out |= {k for k in a.get(name) or [] if isinstance(k, str)}
    return out


def spawn_data(doc: EnvDocument) -> dict[str, Any]:
    """agent.spawn 사전(없거나 잘못된 모양이면 빈 사전)."""
    agent = doc.data.get("agent") if isinstance(doc.data, dict) else None
    sp = agent.get("spawn") if isinstance(agent, dict) else None
    return sp if isinstance(sp, dict) else {}


def tanks(doc: EnvDocument) -> list[str]:
    return [str(a.get("id")) for a in doc.items("actuators") if a.get("kind") == "fuel_tank"]


def tank_body(doc: EnvDocument, tank_id: str) -> str | None:
    """연료 탱크에는 물체 칸이 없다. 그 탱크를 쓰는 첫 추진기의 물체로 본다(그림·연료 센서 도구용)."""
    for a in doc.items("actuators"):
        if a.get("kind") == "thruster" and a.get("tank") == tank_id and isinstance(a.get("body"), str):
            return a["body"]
    return None


# ---------------------------------------------------------------------------
# 조작 함수: 모양과 물체
# ---------------------------------------------------------------------------


def box_shape(w: float, h: float, offset: Point = (0.0, 0.0), angle: float = 0.0) -> dict[str, Any]:
    s: dict[str, Any] = {"kind": "box", "w": clean(w), "h": clean(h)}
    if offset != (0.0, 0.0):
        s["offset"] = _vec_list(offset)
    if angle:
        s["angle"] = angle
    return s


def circle_shape(r: float, offset: Point = (0.0, 0.0)) -> dict[str, Any]:
    s: dict[str, Any] = {"kind": "circle", "r": clean(r)}
    if offset != (0.0, 0.0):
        s["offset"] = _vec_list(offset)
    return s


def check_polygon(points: list[Point]) -> list[Point]:
    """다각형 꼭짓점 검사: 이어서 같은 점은 하나로, 3개 이상, 넓이, 자기 교차. 잘못되면 ToolError."""
    pts: list[Point] = []
    for p in points:
        if not pts or math.dist(pts[-1], p) > 1e-9:
            pts.append((float(p[0]), float(p[1])))
    if len(pts) > 1 and math.dist(pts[0], pts[-1]) <= 1e-9:
        pts.pop()
    if len(pts) < 3:
        raise ToolError(S.EDC_ERR_POLY_FEW)
    if len(pts) > MAX_POLY_POINTS:
        raise ToolError(S.EDC_ERR_POLY_MANY.format(limit=MAX_POLY_POINTS))
    if not geo.is_simple(pts):
        raise ToolError(S.EDC_ERR_POLY_SELF_INTERSECT)
    if abs(geo.signed_area(pts)) < 1e-9:
        raise ToolError(S.EDC_ERR_POLY_AREA)
    return pts


def poly_shape(points: list[Point], offset: Point = (0.0, 0.0)) -> dict[str, Any]:
    pts = check_polygon(points)
    return {"kind": "poly", "points": [_vec_list((p[0] - offset[0], p[1] - offset[1])) for p in pts]}


def add_body(doc: EnvDocument, owner: str, pos: Point, shapes: list[dict[str, Any]], *, angle: float = 0.0,
             id_base: str = "body", label: str = "", tags: list[str] | None = None,
             body_type: str | None = None, undo_label: str | None = None) -> Ref:
    """물체 하나. owner가 agent면 pos는 에이전트 좌표, world면 세계 좌표. 세계 물체는 기본이 정적."""
    if owner not in ("agent", "world"):
        raise ValueError(owner)
    item: dict[str, Any] = {
        "id": id_base,
        "label": label,
        "owner": owner,
        "type": body_type or ("dynamic" if owner == "agent" else "static"),
        "pos": _vec_list(pos),
        "angle": angle,
        "shapes": shapes,
        "material": dict(AGENT_MATERIAL if owner == "agent" else WORLD_MATERIAL),
        "tags": list(tags if tags is not None else ([] if owner == "agent" else [WORLD_TAG])),
    }
    return doc.add("bodies", item, undo_label or S.EDC_UNDO_ADD_BODY)


def add_box(doc: EnvDocument, owner: str, center: Point, w: float, h: float, angle: float = 0.0) -> Ref:
    return add_body(doc, owner, center, [box_shape(w, h)], angle=angle, id_base="block", label=S.EDC_LABEL_BOX)


def add_circle(doc: EnvDocument, owner: str, center: Point, r: float) -> Ref:
    return add_body(doc, owner, center, [circle_shape(r)], id_base="ball", label=S.EDC_LABEL_CIRCLE)


def add_poly(doc: EnvDocument, owner: str, points: list[Point]) -> Ref:
    """꼭짓점(owner 좌표계)으로 다각형 물체. 물체 위치는 꼭짓점 상자의 가운데."""
    pts = check_polygon(points)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pos = (clean((min(xs) + max(xs)) / 2.0), clean((min(ys) + max(ys)) / 2.0))
    return add_body(doc, owner, pos, [poly_shape(pts, pos)], id_base="polygon", label=S.EDC_LABEL_POLY)


def add_shape(doc: EnvDocument, body: Ref, shape: dict[str, Any]) -> int:
    """있는 물체에 모양을 더한다(모양 좌표는 그 물체 좌표). 더한 모양의 번호."""
    path = doc.item_path(body)
    if path is None:
        raise ToolError(S.EDC_ERR_NO_BODY)
    index = len(doc.get((*path, "shapes"), []) or [])

    def mutate(d: dict[str, Any]) -> None:
        b = get_path(d, path)
        if not isinstance(b.get("shapes"), list):
            b["shapes"] = []
        b["shapes"].append(copy.deepcopy(shape))

    doc.edit(S.EDC_UNDO_ADD_SHAPE, mutate)
    doc.select([body])
    return index


def add_terrain_line(doc: EnvDocument, points: list[Point]) -> Ref:
    """지형 선: 세계 좌표 꼭짓점을 잇는 선 체인 하나로 된 정적 세계 물체(태그 terrain)."""
    pts: list[Point] = []
    for p in points:
        if not pts or math.dist(pts[-1], p) > 1e-9:
            pts.append(p)
    if len(pts) < 2:
        raise ToolError(S.EDC_ERR_LINE_FEW)
    shape = {"kind": "chain", "points": [_vec_list(p) for p in pts], "closed": False, "radius": CHAIN_RADIUS}
    item: dict[str, Any] = {
        "id": "ground", "label": S.EDC_LABEL_TERRAIN_LINE, "owner": "world", "type": "static",
        "pos": [0.0, 0.0], "angle": 0.0, "shapes": [shape],
        "material": {"density": 1.0, "friction": TERRAIN_FRICTION}, "tags": [TERRAIN_TAG],
    }
    return doc.add("bodies", item, S.EDC_UNDO_ADD_TERRAIN_LINE)


# ---------------------------------------------------------------------------
# 조작 함수: 생성기
# ---------------------------------------------------------------------------


def _world_x_range(doc: EnvDocument) -> list[float]:
    b = doc.get(("world", "bounds"))
    if isinstance(b, list) and len(b) == 4 and is_number(b[0]) and is_number(b[2]) and b[0] < b[2]:
        return [float(b[0]), float(b[2])]
    return [-50.0, 50.0]


def add_heightmap(doc: EnvDocument, pad: bool = True) -> Ref:
    """지형 자동 생성: 높이맵 생성기(세계 경계 폭, 가운데 착륙 패드). 수치는 생성기 탭에서 고친다."""
    x0, x1 = _world_x_range(doc)
    item: dict[str, Any] = {"kind": "heightmap", "id": "terrain", "label": S.EDC_LABEL_HEIGHTMAP,
                            "x_range": [x0, x1], **HEIGHTMAP}
    if pad:
        item["pad"] = {**HEIGHTMAP_PAD, "x": clean(min(max(0.0, x0 + 10.0), x1 - 10.0))}
    return doc.add("generators", item, S.EDC_UNDO_ADD_GENERATOR)


def add_generated_track(doc: EnvDocument) -> Ref:
    """트랙 자동 생성: 학습 트랙 세트를 만드는 트랙 생성기. 세트 트랙 생성기는 하나만 둘 수 있다."""
    if any(g.get("kind") == "track" and g.get("source", "generated") == "generated" for g in doc.items("generators")):
        raise ToolError(S.EDC_ERR_TRACK_ONE_SET)
    item = {"kind": "track", "id": "track", "label": S.EDC_LABEL_TRACK, **copy.deepcopy(TRACK_GENERATED)}
    return doc.add("generators", item, S.EDC_UNDO_ADD_GENERATOR)


def add_drawn_track(doc: EnvDocument, centerline: list[Point], width: float | None = None) -> Ref:
    """트랙 그리기: 중심선 꼭짓점(세계 좌표)과 폭으로 트랙 하나. 만들 수 없는 중심선이면 ToolError(이유)."""
    from evolab.generators import track as tg

    pts: list[Point] = []
    for p in centerline:
        if not pts or math.dist(pts[-1], p) > 1e-9:
            pts.append((float(p[0]), float(p[1])))
    w = float(width if width is not None else TRACK_DRAWN["width"])
    try:
        tg.track_from_centerline(pts, width=w, checkpoints=int(TRACK_DRAWN["checkpoints"]))
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    item = {"kind": "track", "id": "track", "label": S.EDC_LABEL_TRACK, **copy.deepcopy(TRACK_DRAWN),
            "width": w, "centerline": [_vec_list(p) for p in pts]}
    return doc.add("generators", item, S.EDC_UNDO_ADD_GENERATOR)


# ---------------------------------------------------------------------------
# 조작 함수: 관절
# ---------------------------------------------------------------------------


def _placed_parent_chain(doc: EnvDocument, body_id: str) -> set[str]:
    """body_id에서 place_by 부모를 따라 올라간 물체들(자기 포함)."""
    joints = {str(j.get("id")): j for j in doc.items("joints")}
    seen = {body_id}
    node = body_id
    while True:
        b = body_data(doc, node)
        if b is None or not b.get("place_by") or b["place_by"] not in joints:
            return seen
        node = str(joints[b["place_by"]].get("a"))
        if node in seen:
            return seen
        seen.add(node)


def can_place(doc: EnvDocument, a: str, b: str) -> bool:
    """관절로 b의 자세를 정할 수 있나(place_by): b가 문서 물체이고, 아직 다른 관절로 놓이지 않았고, 소유가 같고,
    순환이 생기지 않을 때."""
    bb, ba = body_data(doc, b), body_data(doc, a)
    if bb is None or ba is None or bb.get("place_by"):
        return False
    if bb.get("owner", "agent") != ba.get("owner", "agent"):
        return False
    return b not in _placed_parent_chain(doc, a)


def add_joint(doc: EnvDocument, kind: str, a: str, b: str, anchor_a: Point, anchor_b: Point, *,
              rest_angle: float = 0.0, place: bool | None = None) -> Ref:
    """관절. 앵커는 각 물체 좌표. place가 참(기본: 회전·고정 관절이고 가능하면)이면 b.place_by를 이 관절로 해서
    b가 앵커끼리 맞도록 놓이게 한다(착륙선 다리처럼). rest_angle은 b 각도 - a 각도."""
    if kind not in _JOINT_LABELS:
        raise ValueError(kind)
    if a == b:
        raise ToolError(S.EDC_ERR_JOINT_SAME)
    item: dict[str, Any] = {"id": "joint", "label": _JOINT_LABELS[kind], "kind": kind, "a": a, "b": b,
                            "anchor_a": _vec_list(anchor_a), "anchor_b": _vec_list(anchor_b)}
    if kind in ("pivot", "rot_spring", "weld"):
        item["rest_angle"] = rest_angle
    if kind == "rot_spring":
        item.update(ROT_SPRING)
    elif kind == "spring":
        item.update(DIST_SPRING)
    if place is None:
        place = kind != "spring"
    do_place = place and can_place(doc, a, b)

    def extra(d: dict[str, Any], jid: str) -> None:
        if do_place:
            for body in _list(d, "bodies"):
                if body.get("id") == b:
                    body["place_by"] = jid

    return _insert(doc, "joints", item, S.EDC_UNDO_ADD_JOINT, extra)


# ---------------------------------------------------------------------------
# 조작 함수: 구동기
# ---------------------------------------------------------------------------


def _need_agent_body(doc: EnvDocument, body: str) -> None:
    if not _agent_dynamic(doc, body):
        raise ToolError(S.EDC_ERR_ACTUATOR_BODY.format(body=body))


def add_thruster(doc: EnvDocument, body: str, at: Point, direction: float,
                 max_force: float = THRUSTER_FORCE) -> Ref:
    """추진기: at(물체 좌표)에서 direction(물체 좌표, 라디안) 쪽으로 미는 힘. 탱크가 하나뿐이면 그 탱크를 쓴다."""
    _need_agent_body(doc, body)
    item: dict[str, Any] = {"kind": "thruster", "id": "engine", "label": S.EDC_LABEL_THRUSTER, "body": body,
                            "at": _vec_list(at), "dir": direction, "max_force": float(max_force)}
    ts = tanks(doc)
    if len(ts) == 1:
        item["tank"] = ts[0]
        item["fuel_per_s"] = THRUSTER_FUEL_PER_S
    used = _used_keys(doc)
    for keys in THRUSTER_KEYS:
        if not used & set(keys):
            item["keys"] = list(keys)
            break
    return doc.add("actuators", item, S.EDC_UNDO_ADD_ACTUATOR)


def add_fuel_tank(doc: EnvDocument, body: str | None = None, capacity: float = TANK_CAPACITY) -> Ref:
    """연료 탱크. body를 주면 그 물체에서 탱크가 없는 추진기들이 이 탱크를 쓰게 한다(초당 소모량이 0이면 1)."""
    item = {"kind": "fuel_tank", "id": "tank", "label": S.EDC_LABEL_TANK, "capacity": float(capacity)}

    def extra(d: dict[str, Any], tid: str) -> None:
        if body is None:
            return
        for a in _list(d, "actuators"):
            if a.get("kind") == "thruster" and a.get("body") == body and not a.get("tank"):
                a["tank"] = tid
                if not a.get("fuel_per_s"):
                    a["fuel_per_s"] = THRUSTER_FUEL_PER_S

    return _insert(doc, "actuators", item, S.EDC_UNDO_ADD_ACTUATOR, extra)


def add_vehicle(doc: EnvDocument, body: str) -> Ref:
    """차량 구동기(위에서 본 차). 축거는 물체 첫 사각형 길이의 0.8배(없으면 2m)."""
    _need_agent_body(doc, body)
    wheelbase = VEHICLE_WHEELBASE
    b = body_data(doc, body) or {}
    shapes = b.get("shapes") or []
    if shapes and isinstance(shapes[0], dict) and shapes[0].get("kind") == "box" and is_number(shapes[0].get("w")):
        wheelbase = clean(0.8 * float(shapes[0]["w"]))
    item: dict[str, Any] = {"kind": "vehicle", "id": "drive", "label": S.EDC_LABEL_VEHICLE, "body": body,
                            **VEHICLE, "wheelbase": wheelbase}
    used = _used_keys(doc)
    for name, keys in VEHICLE_KEYS.items():
        item[name] = [] if used & set(keys) else list(keys)
    return doc.add("actuators", item, S.EDC_UNDO_ADD_ACTUATOR)


# ---------------------------------------------------------------------------
# 조작 함수: 센서
# ---------------------------------------------------------------------------


def _need_body(doc: EnvDocument, body: str) -> None:
    if body_data(doc, body) is None:
        raise ToolError(S.EDC_ERR_NO_BODY)


def add_rays(doc: EnvDocument, body: str, origin: Point, center: float = 0.0, *, count: int | None = None,
             spread: float | None = None, length: float | None = None, hits: list[str] | None = None) -> Ref:
    """부채꼴 레이. 기본: 7개, 180도, 40m, 세계 쪽 태그 전부에 맞음."""
    _need_body(doc, body)
    if hits is None:
        hits = [f"tag.{t}" for t in world_tags(doc)] or [f"tag.{TERRAIN_TAG}"]
    item = {"kind": "rays", "id": "ray", "label": S.EDC_LABEL_RAYS, "body": body, "origin": _vec_list(origin),
            "count": int(count or RAYS["count"]), "spread": float(spread if spread is not None else RAYS["spread"]),
            "center": center, "length": float(length if length is not None else RAYS["length"]), "hits": hits}
    return doc.add("sensors", item, S.EDC_UNDO_ADD_SENSOR)


def add_body_state(doc: EnvDocument, body: str, fields: list[str] | None = None, frame: str | None = None) -> Ref:
    """몸 상태. 기본: 차량 구동기가 달린 물체면 앞뒤·옆 속도(물체 좌표), 아니면 속도·각도·각속도(세계 좌표)."""
    _need_body(doc, body)
    car = any(a.get("kind") == "vehicle" and a.get("body") == body for a in doc.items("actuators"))
    if fields is None:
        fields = ["v_fwd", "v_lat"] if car else ["vx", "vy", "angle", "omega"]
    item = {"kind": "body_state", "id": "st", "label": S.EDC_LABEL_BODY_STATE, "body": body, "fields": fields,
            "frame": frame or ("body" if car else "world")}
    return doc.add("sensors", item, S.EDC_UNDO_ADD_SENSOR)


def add_contact(doc: EnvDocument, body: str, with_: list[str] | None = None) -> Ref:
    """접촉 센서. 기본 대상: tag.terrain, 없으면 첫 세계 쪽 태그."""
    _need_body(doc, body)
    if with_ is None:
        tags = world_tags(doc)
        with_ = [f"tag.{TERRAIN_TAG if TERRAIN_TAG in tags or not tags else tags[0]}"]
    item = {"kind": "contact", "id": f"{body}_touch", "label": S.EDC_LABEL_CONTACT, "body": body, "with": with_}
    return doc.add("sensors", item, S.EDC_UNDO_ADD_SENSOR)


def add_target(doc: EnvDocument, body: str, target: str) -> Ref:
    """목표 위치 센서: body에서 본 target(영역 또는 움직이는 물체)의 상대 위치."""
    _need_body(doc, body)
    if target == body:
        raise ToolError(S.EDC_ERR_TARGET_SELF)
    item = {"kind": "target", "id": "target_rel", "label": S.EDC_LABEL_TARGET, "body": body, "target": target,
            "frame": "world", "scale": 20.0}
    return doc.add("sensors", item, S.EDC_UNDO_ADD_SENSOR)


def add_joint_angle(doc: EnvDocument, joint: str) -> Ref:
    if doc.item(("joints", joint)) is None:
        raise ToolError(S.EDC_ERR_NO_JOINT)
    item = {"kind": "joint_angle", "id": f"{joint}_angle", "label": S.EDC_LABEL_JOINT_ANGLE, "joint": joint}
    return doc.add("sensors", item, S.EDC_UNDO_ADD_SENSOR)


def add_fuel_sensor(doc: EnvDocument, tank: str) -> Ref:
    if tank not in tanks(doc):
        raise ToolError(S.EDC_ERR_NO_TANK)
    item = {"kind": "fuel", "id": "fuel_left", "label": S.EDC_LABEL_FUEL, "tank": tank}
    return doc.add("sensors", item, S.EDC_UNDO_ADD_SENSOR)


def add_time_left(doc: EnvDocument) -> Ref:
    item = {"kind": "time_left", "id": "clock", "label": S.EDC_LABEL_TIME_LEFT}
    return doc.add("sensors", item, S.EDC_UNDO_ADD_SENSOR)


# ---------------------------------------------------------------------------
# 조작 함수: 영역, 체크포인트, 시작 위치
# ---------------------------------------------------------------------------


def add_region(doc: EnvDocument, kind: str, center: Point, w: float, h: float) -> Ref:
    """영역(세계 좌표의 사각형). kind: goal, hazard, pad, zone."""
    if kind not in _REGION_LABELS:
        raise ValueError(kind)
    item = {"id": _REGION_IDS[kind], "label": _REGION_LABELS[kind], "kind": kind,
            "shape": {"kind": "box", "w": clean(w), "h": clean(h)}, "pos": _vec_list(center), "tags": []}
    return doc.add("regions", item, S.EDC_UNDO_ADD_REGION)


def add_checkpoints(doc: EnvDocument, lines: list[tuple[Point, Point]], loop: bool = True) -> Ref:
    """순서 있는 체크포인트 선(세계 좌표). 0번이 출발선."""
    good = [(a, b) for a, b in lines if math.dist(a, b) > 1e-9]
    if not good:
        raise ToolError(S.EDC_ERR_CHECKPOINT_FEW)
    item = {"id": "cp", "label": S.EDC_LABEL_CHECKPOINTS, "lines": [[_vec_list(a), _vec_list(b)] for a, b in good],
            "loop": bool(loop)}
    return doc.add("checkpoints", item, S.EDC_UNDO_ADD_CHECKPOINTS)


def set_spawn(doc: EnvDocument, x: float | None = None, y: float | None = None, angle: float | None = None,
              merge_key: str | None = None) -> list[str]:
    """시작 위치(세계 좌표)를 바꾼다. 수식 칸은 건너뛰고 그 칸 이름을 돌려준다."""
    spawn = spawn_data(doc)
    changes: list[tuple[JsonPath, Any]] = []
    skipped: list[str] = []
    for name, value in (("x", x), ("y", y), ("angle", angle)):
        if value is None:
            continue
        if is_formula(spawn.get(name)):
            skipped.append(name)
            continue
        changes.append((("agent", "spawn", name), value if name == "angle" else clean(value)))
    if changes:

        def mutate(d: dict[str, Any]) -> None:
            if not isinstance(d.get("agent"), dict):
                d["agent"] = {}
            if not isinstance(d["agent"].get("spawn"), dict):
                d["agent"]["spawn"] = {}
            for path, value in changes:
                d["agent"]["spawn"][path[-1]] = value

        doc.edit(S.EDC_UNDO_SPAWN, mutate, merge_key)
    return skipped


# ---------------------------------------------------------------------------
# 조작 함수: 옮기기, 돌리기, 지우기
# ---------------------------------------------------------------------------


def move_changes(doc: EnvDocument, ref: Ref, d: Point, base: dict[str, Any] | None = None,
                 snap: Callable[[float], float] | None = None) -> list[tuple[JsonPath, Any]]:
    """부품을 d만큼(파일 좌표계) 옮기는 칸 변경 목록. 물체·영역은 pos, 체크포인트는 모든 선 끝점.
    수식 칸과 관절로 놓인 물체(place_by)는 건너뛴다. base를 주면 그 사본의 값에서 계산한다(끌기)."""
    data = doc.data if base is None else base
    path = doc.item_path(ref)
    if path is None:
        return []
    try:
        item = get_path(data, path)
    except (KeyError, IndexError, TypeError):
        return []
    out: list[tuple[JsonPath, Any]] = []
    section = ref[0]
    if section == "bodies":
        if item.get("place_by"):
            return []
        new, moved = shifted_vec(item.get("pos", [0.0, 0.0]), d, snap)
        if moved:
            out.append(((*path, "pos"), new))
    elif section == "regions":
        new, moved = shifted_vec(item.get("pos", [0.0, 0.0]), d, snap)
        if moved:
            out.append(((*path, "pos"), new))
    elif section == "checkpoints":
        lines = copy.deepcopy(item.get("lines") or [])
        moved_any = False
        for line in lines:
            if isinstance(line, list):
                for e in range(min(2, len(line))):
                    line[e], moved = shifted_vec(line[e], d, snap)
                    moved_any |= moved
        if moved_any:
            out.append(((*path, "lines"), lines))
    return out


def move_locked(doc: EnvDocument, ref: Ref) -> bool:
    """끌 수 없는 부품인가(수식 위치, 관절 배치)."""
    item = doc.item(ref)
    if item is None:
        return True
    if ref[0] == "bodies":
        return bool(item.get("place_by")) or vec_locked(item.get("pos", [0.0, 0.0]))
    if ref[0] == "regions":
        return vec_locked(item.get("pos", [0.0, 0.0]))
    if ref[0] == "checkpoints":
        return all(vec_locked(p) for line in item.get("lines") or [] if isinstance(line, list) for p in line)
    return True


def move_component(doc: EnvDocument, ref: Ref, dx: float, dy: float, merge_key: str | None = None) -> bool:
    """부품 하나를 파일 좌표계로 (dx, dy)만큼 옮긴다(화살표 키 이동과 같다). 옮겼나."""
    changes = move_changes(doc, ref, (dx, dy))
    if changes:
        apply_changes(doc, S.EDC_UNDO_MOVE, changes, merge_key)
    return bool(changes)


def rotate_body(doc: EnvDocument, ref: Ref, angle: float, merge_key: str | None = None) -> bool:
    """물체 각도(파일 좌표계)를 바꾼다. 수식이거나 관절로 놓인 물체면 바꾸지 않는다."""
    item = doc.item(ref)
    path = doc.item_path(ref)
    if item is None or path is None or item.get("place_by") or is_formula(item.get("angle", 0.0)):
        return False
    apply_changes(doc, S.EDC_UNDO_ROTATE, [((*path, "angle"), angle)], merge_key)
    return True


def delete(doc: EnvDocument, refs: list[Ref]) -> None:
    if refs:
        doc.remove(list(refs))


# ---------------------------------------------------------------------------
# 도구 클래스
# ---------------------------------------------------------------------------

COLOR_DRAFT = QColor("#ffd166")
COLOR_DRAFT_FILL = QColor(255, 209, 102, 40)


def _draft_pen(width: float = 1.5, dash: bool = False) -> QPen:
    pen = QPen(COLOR_DRAFT, width)
    if dash:
        pen.setStyle(Qt.PenStyle.DashLine)
    return pen


class Tool:
    """도구 하나. 캔버스가 마우스·키 입력을 넘긴다. 좌표는 CanvasEvent.world(세계 좌표)."""

    tool_id = ""
    cursor = Qt.CursorShape.CrossCursor

    def __init__(self, canvas: EditorCanvas) -> None:
        self.canvas = canvas

    @property
    def doc(self) -> EnvDocument:
        return self.canvas.doc

    def hint(self) -> str:
        return TOOL_HINTS.get(self.tool_id, "")

    def press(self, e: CanvasEvent) -> None:
        pass

    def move(self, e: CanvasEvent) -> None:
        pass

    def release(self, e: CanvasEvent) -> None:
        pass

    def double_click(self, e: CanvasEvent) -> None:
        self.press(e)

    def key(self, event: QKeyEvent) -> bool:
        return False

    def busy(self) -> bool:
        return False

    def cancel(self) -> None:
        pass

    def overlay(self, p: QPainter) -> None:
        pass

    def run(self, fn: Callable[[], Any]) -> Any:
        """조작을 하고, 할 수 없으면 이유를 상태 줄에 띄운다."""
        try:
            return fn()
        except ToolError as exc:
            self.canvas.message(str(exc))
            return None


# ---- 그리기 도우미 ----


def _poly_screen(canvas: EditorCanvas, frame: Pose, pts: list[Point]) -> QPolygonF:
    return QPolygonF([canvas.world_to_screen(*to_world(frame, q)) for q in pts])


class _RectDraft:
    """끌어서 만드는 사각형(좌표계 frame에서 축 정렬)."""

    def __init__(self) -> None:
        self.frame: Pose = (0.0, 0.0, 0.0)
        self.a: Point | None = None
        self.b: Point | None = None
        self.start_screen = QPointF()
        self.dragged = False

    def begin(self, canvas: EditorCanvas, e: CanvasEvent, frame: Pose) -> None:
        self.frame = frame
        self.a = canvas.snap(to_local(frame, e.world), e)
        self.b = self.a
        self.start_screen = e.screen
        self.dragged = False

    def update(self, canvas: EditorCanvas, e: CanvasEvent) -> None:
        if self.a is None:
            return
        if (e.screen - self.start_screen).manhattanLength() >= 4:
            self.dragged = True
        self.b = canvas.snap(to_local(self.frame, e.world), e)

    def rect(self) -> tuple[Point, float, float] | None:
        if self.a is None or self.b is None:
            return None
        (x0, y0), (x1, y1) = self.a, self.b
        return ((clean((x0 + x1) / 2), clean((y0 + y1) / 2)), clean(abs(x1 - x0)), clean(abs(y1 - y0)))

    def paint(self, canvas: EditorCanvas, p: QPainter) -> None:
        if self.a is None or self.b is None or not self.dragged:
            return
        (x0, y0), (x1, y1) = self.a, self.b
        p.setPen(_draft_pen())
        p.setBrush(QBrush(COLOR_DRAFT_FILL))
        p.drawPolygon(_poly_screen(canvas, self.frame, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

    def clear(self) -> None:
        self.a = self.b = None
        self.dragged = False


class _ShapeTool(Tool):
    """모양 도구 공통: 새 물체의 소유는 캔버스 설정(new_owner). Ctrl을 누른 채 그리면 선택한 물체에 모양을 더한다."""

    def _target(self, e: CanvasEvent) -> tuple[Pose, Ref | None]:
        if e.ctrl:
            body = self.canvas.selected_body()
            if body is not None:
                return body.pose, body.ref
        return self.canvas.owner_frame(self.canvas.new_owner), None


class BoxTool(_ShapeTool):
    tool_id = "box"

    def __init__(self, canvas: EditorCanvas) -> None:
        super().__init__(canvas)
        self.draft = _RectDraft()
        self.into: Ref | None = None

    def press(self, e: CanvasEvent) -> None:
        frame, self.into = self._target(e)
        self.draft.begin(self.canvas, e, frame)

    def move(self, e: CanvasEvent) -> None:
        self.draft.update(self.canvas, e)
        self.canvas.update()

    def release(self, e: CanvasEvent) -> None:
        self.draft.update(self.canvas, e)
        r = self.draft.rect()
        dragged = self.draft.dragged
        self.draft.clear()
        if r is None:
            return
        center, w, h = r
        if not dragged or w <= 1e-9 or h <= 1e-9:
            w, h = DEFAULT_BOX
            center = self.draft_center(e)
        if self.into is not None:
            self.run(lambda: add_shape(self.doc, self.into, box_shape(w, h, center)))  # type: ignore[arg-type]
        else:
            self.run(lambda: add_box(self.doc, self.canvas.new_owner, center, w, h))
        self.canvas.update()

    def draft_center(self, e: CanvasEvent) -> Point:
        frame, _ = self._target(e)
        return self.canvas.snap(to_local(frame, e.world), e)

    def busy(self) -> bool:
        return self.draft.a is not None

    def cancel(self) -> None:
        self.draft.clear()
        self.canvas.update()

    def overlay(self, p: QPainter) -> None:
        self.draft.paint(self.canvas, p)


class CircleTool(_ShapeTool):
    tool_id = "circle"

    def __init__(self, canvas: EditorCanvas) -> None:
        super().__init__(canvas)
        self.frame: Pose = (0.0, 0.0, 0.0)
        self.center: Point | None = None
        self.r = 0.0
        self.into: Ref | None = None
        self.start = QPointF()
        self.dragged = False

    def press(self, e: CanvasEvent) -> None:
        self.frame, self.into = self._target(e)
        self.center = self.canvas.snap(to_local(self.frame, e.world), e)
        self.r = 0.0
        self.start = e.screen
        self.dragged = False

    def move(self, e: CanvasEvent) -> None:
        if self.center is None:
            return
        if (e.screen - self.start).manhattanLength() >= 4:
            self.dragged = True
        q = to_local(self.frame, e.world)
        r = math.dist(q, self.center)
        self.r = self.canvas.snap_length(r, e)
        self.canvas.update()

    def release(self, e: CanvasEvent) -> None:
        if self.center is None:
            return
        self.move(e)
        center, r = self.center, self.r
        if not self.dragged or r <= 1e-9:
            r = DEFAULT_RADIUS
        self.center = None
        if self.into is not None:
            into = self.into
            self.run(lambda: add_shape(self.doc, into, circle_shape(r, center)))
        else:
            self.run(lambda: add_circle(self.doc, self.canvas.new_owner, center, r))
        self.canvas.update()

    def busy(self) -> bool:
        return self.center is not None

    def cancel(self) -> None:
        self.center = None
        self.canvas.update()

    def overlay(self, p: QPainter) -> None:
        if self.center is None or not self.dragged:
            return
        c = self.canvas.world_to_screen(*to_world(self.frame, self.center))
        rr = self.r * self.canvas.scale
        p.setPen(_draft_pen())
        p.setBrush(QBrush(COLOR_DRAFT_FILL))
        p.drawEllipse(c, rr, rr)


class _PointsTool(Tool):
    """점을 차례로 찍는 도구(다각형, 지형 선, 트랙 중심선). 두 번 누르기나 Enter로 끝, Backspace로 마지막 점 지우기."""

    closed = False
    min_points = 2

    def __init__(self, canvas: EditorCanvas) -> None:
        super().__init__(canvas)
        self.frame: Pose = (0.0, 0.0, 0.0)
        self.points: list[Point] = []
        self.hover: Point | None = None

    def frame_for(self, e: CanvasEvent) -> Pose:
        return (0.0, 0.0, 0.0)

    def press(self, e: CanvasEvent) -> None:
        if not self.points:
            self.frame = self.frame_for(e)
        q = self.canvas.snap(to_local(self.frame, e.world), e)
        if not self.points or math.dist(self.points[-1], q) > 1e-9:
            self.points.append(q)
        self.canvas.update()

    def move(self, e: CanvasEvent) -> None:
        if self.points:
            self.hover = self.canvas.snap(to_local(self.frame, e.world), e)
            self.canvas.update()

    def double_click(self, e: CanvasEvent) -> None:
        self.press(e)
        self.finish()

    def key(self, event: QKeyEvent) -> bool:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.points:
            self.finish()
            return True
        if event.key() == Qt.Key.Key_Backspace and self.points:
            self.points.pop()
            self.canvas.update()
            return True
        return False

    def finish(self) -> None:
        if len(self.points) < self.min_points:
            self.canvas.message(self.too_few_message())
            return
        result = self.run(lambda: self.commit(list(self.points)))
        if result is not None:
            self.points = []
            self.hover = None
        self.canvas.update()

    def too_few_message(self) -> str:
        return S.EDC_ERR_LINE_FEW

    def commit(self, points: list[Point]) -> Any:
        raise NotImplementedError

    def busy(self) -> bool:
        return bool(self.points)

    def cancel(self) -> None:
        self.points = []
        self.hover = None
        self.canvas.update()

    def overlay(self, p: QPainter) -> None:
        if not self.points:
            return
        pts = list(self.points) + ([self.hover] if self.hover is not None else [])
        poly = _poly_screen(self.canvas, self.frame, pts)
        p.setPen(_draft_pen())
        p.setBrush(Qt.BrushStyle.NoBrush)
        if self.closed and len(pts) >= 3:
            p.drawPolygon(poly)
        else:
            p.drawPolyline(poly)
        p.setBrush(QBrush(COLOR_DRAFT))
        for i in range(len(self.points)):
            p.drawEllipse(poly[i], 3.0, 3.0)


class PolyTool(_PointsTool):
    tool_id = "poly"
    closed = True
    min_points = 3

    def frame_for(self, e: CanvasEvent) -> Pose:
        return self.canvas.owner_frame(self.canvas.new_owner)

    def too_few_message(self) -> str:
        return S.EDC_ERR_POLY_FEW

    def commit(self, points: list[Point]) -> Any:
        return add_poly(self.doc, self.canvas.new_owner, points)


class TerrainLineTool(_PointsTool):
    tool_id = "terrain_line"

    def commit(self, points: list[Point]) -> Any:
        return add_terrain_line(self.doc, points)


class TrackDrawTool(_PointsTool):
    tool_id = "track_draw"
    closed = True
    min_points = 3

    def too_few_message(self) -> str:
        return S.EDC_ERR_TRACK_FEW

    def commit(self, points: list[Point]) -> Any:
        return add_drawn_track(self.doc, points)

    def overlay(self, p: QPainter) -> None:
        if self.points:
            pts = list(self.points) + ([self.hover] if self.hover is not None else [])
            if len(pts) >= 2:
                width = float(TRACK_DRAWN["width"]) * self.canvas.scale
                pen = QPen(QColor(255, 209, 102, 45), width)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolygon(_poly_screen(self.canvas, self.frame, pts))
        super().overlay(p)


class JointTool(Tool):
    """물체 A의 한 점, 물체 B의 한 점을 차례로 누른다. 앵커는 누른 점을 각 물체 좌표로 바꾼 것."""

    def __init__(self, canvas: EditorCanvas, kind: str) -> None:
        super().__init__(canvas)
        self.kind = kind
        self.tool_id = next(t for t, k in JOINT_KINDS.items() if k == kind)
        self.first: tuple[str, Pose, Point, Point] | None = None
        """(물체 id, 물체 자세, 앵커(물체 좌표), 누른 세계 점)"""
        self.hover: Point | None = None

    def hint(self) -> str:
        return S.EDC_HINT_JOINT_SECOND if self.first else S.EDC_HINT_JOINT

    def press(self, e: CanvasEvent) -> None:
        exclude = (self.first[0],) if self.first else ()
        body = self.canvas.pick_body(e.world, exclude=exclude)
        if body is None:
            self.canvas.message(S.EDC_ERR_CLICK_BODY)
            return
        anchor = self.canvas.snap(to_local(body.pose, e.world), e)
        if self.first is None:
            self.first = (body.id, body.pose, anchor, to_world(body.pose, anchor))
            self.canvas.message(self.hint())
            self.canvas.update()
            return
        a_id, a_pose, a_anchor, _ = self.first
        self.first = None
        rest = geo.wrap_angle(body.pose[2] - a_pose[2])
        rest = 0.0 if abs(rest) < 1e-9 else rest
        self.run(lambda: add_joint(self.doc, self.kind, a_id, body.id, a_anchor, anchor, rest_angle=rest))
        self.canvas.update()

    def move(self, e: CanvasEvent) -> None:
        if self.first:
            self.hover = e.world
            self.canvas.update()

    def double_click(self, e: CanvasEvent) -> None:
        pass

    def busy(self) -> bool:
        return self.first is not None

    def cancel(self) -> None:
        self.first = None
        self.canvas.update()

    def overlay(self, p: QPainter) -> None:
        if self.first is None:
            return
        a = self.canvas.world_to_screen(*self.first[3])
        p.setPen(_draft_pen())
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(a, 5.0, 5.0)
        if self.hover is not None:
            p.setPen(_draft_pen(dash=True))
            p.drawLine(a, self.canvas.world_to_screen(*self.hover))


class ThrusterTool(Tool):
    """에이전트 물체를 눌러 위치, 분사가 나갈 쪽으로 끌어 방향을 정한다(힘은 그 반대로 민다). 그냥 누르면 분사는
    물체 중심에서 먼 쪽, 힘은 중심 쪽."""

    tool_id = "thruster"

    def __init__(self, canvas: EditorCanvas) -> None:
        super().__init__(canvas)
        self.body: tuple[str, Pose] | None = None
        self.at: Point = (0.0, 0.0)
        self.dir = math.pi / 2
        self.start = QPointF()
        self.dragged = False

    def _default_dir(self, at: Point) -> float:
        if math.hypot(*at) < 1e-9:
            return math.pi / 2
        return math.atan2(-at[1], -at[0])

    def press(self, e: CanvasEvent) -> None:
        body = self.canvas.pick_body(e.world)
        if body is None or body.ref[0] != "bodies":
            self.canvas.message(S.EDC_ERR_CLICK_AGENT_BODY)
            return
        if not _agent_dynamic(self.doc, body.id):
            self.canvas.message(S.EDC_ERR_ACTUATOR_BODY.format(body=body.id))
            return
        self.body = (body.id, body.pose)
        self.at = self.canvas.snap(to_local(body.pose, e.world), e)
        self.dir = self.canvas.snap_dir(self._default_dir(self.at), e, step=math.pi / 4)
        self.start = e.screen
        self.dragged = False

    def move(self, e: CanvasEvent) -> None:
        if self.body is None:
            return
        if (e.screen - self.start).manhattanLength() >= 6:
            self.dragged = True
        if self.dragged:
            q = to_local(self.body[1], e.world)
            self.dir = self.canvas.snap_dir(math.atan2(self.at[1] - q[1], self.at[0] - q[0]), e)
        self.canvas.update()

    def release(self, e: CanvasEvent) -> None:
        if self.body is None:
            return
        self.move(e)
        bid, at, d = self.body[0], self.at, self.dir
        self.body = None
        self.run(lambda: add_thruster(self.doc, bid, at, d))
        self.canvas.update()

    def double_click(self, e: CanvasEvent) -> None:
        pass

    def busy(self) -> bool:
        return self.body is not None

    def cancel(self) -> None:
        self.body = None
        self.canvas.update()

    def overlay(self, p: QPainter) -> None:
        if self.body is None:
            return
        pose = self.body[1]
        base = to_world(pose, self.at)
        length = self.canvas.thrust_length(THRUSTER_FORCE)
        tail = (base[0] - length * math.cos(pose[2] + self.dir), base[1] - length * math.sin(pose[2] + self.dir))
        p.setPen(_draft_pen(2.0))
        p.drawLine(self.canvas.world_to_screen(*tail), self.canvas.world_to_screen(*base))
        p.setBrush(QBrush(COLOR_DRAFT))
        p.drawEllipse(self.canvas.world_to_screen(*base), 3.0, 3.0)


class _BodyClickTool(Tool):
    """물체 하나를 누르면 부품을 더하는 도구."""

    agent_only = False

    def press(self, e: CanvasEvent) -> None:
        body = self.canvas.pick_body(e.world)
        if body is None or body.ref[0] != "bodies":
            self.canvas.message(S.EDC_ERR_CLICK_AGENT_BODY if self.agent_only else S.EDC_ERR_CLICK_BODY)
            return
        local = self.canvas.snap(to_local(body.pose, e.world), e)
        self.run(lambda: self.commit(body.id, local, e))
        self.canvas.update()

    def double_click(self, e: CanvasEvent) -> None:
        pass

    def commit(self, body: str, local: Point, e: CanvasEvent) -> Any:
        raise NotImplementedError


class FuelTankTool(Tool):
    tool_id = "fuel_tank"

    def press(self, e: CanvasEvent) -> None:
        body = self.canvas.pick_body(e.world)
        bid = body.id if body is not None and body.ref[0] == "bodies" else None
        self.run(lambda: add_fuel_tank(self.doc, bid))
        self.canvas.update()

    def double_click(self, e: CanvasEvent) -> None:
        pass


class VehicleTool(_BodyClickTool):
    tool_id = "vehicle"
    agent_only = True

    def commit(self, body: str, local: Point, e: CanvasEvent) -> Any:
        return add_vehicle(self.doc, body)


class RaysTool(_BodyClickTool):
    tool_id = "sensor_rays"

    def commit(self, body: str, local: Point, e: CanvasEvent) -> Any:
        center = 0.0 if math.hypot(*local) < 1e-9 else self.canvas.snap_dir(math.atan2(local[1], local[0]), e)
        return add_rays(self.doc, body, local, center)


class BodyStateTool(_BodyClickTool):
    tool_id = "sensor_body_state"

    def commit(self, body: str, local: Point, e: CanvasEvent) -> Any:
        return add_body_state(self.doc, body)


class ContactTool(_BodyClickTool):
    tool_id = "sensor_contact"

    def commit(self, body: str, local: Point, e: CanvasEvent) -> Any:
        return add_contact(self.doc, body)


class TargetTool(Tool):
    """물체를 누른 뒤 목표(영역 또는 움직이는 물체)를 누른다."""

    tool_id = "sensor_target"

    def __init__(self, canvas: EditorCanvas) -> None:
        super().__init__(canvas)
        self.body: tuple[str, Point] | None = None
        self.hover: Point | None = None

    def hint(self) -> str:
        return S.EDC_HINT_SENSOR_TARGET_SECOND if self.body else S.EDC_HINT_SENSOR_TARGET

    def press(self, e: CanvasEvent) -> None:
        if self.body is None:
            body = self.canvas.pick_body(e.world)
            if body is None or body.ref[0] != "bodies":
                self.canvas.message(S.EDC_ERR_CLICK_BODY)
                return
            self.body = (body.id, (body.pose[0], body.pose[1]))
            self.canvas.message(self.hint())
            self.canvas.update()
            return
        target = self.canvas.pick_target(e.world, exclude=(self.body[0],))
        if target is None:
            self.canvas.message(S.EDC_ERR_CLICK_TARGET)
            return
        bid = self.body[0]
        self.body = None
        self.run(lambda: add_target(self.doc, bid, target))
        self.canvas.update()

    def move(self, e: CanvasEvent) -> None:
        if self.body:
            self.hover = e.world
            self.canvas.update()

    def double_click(self, e: CanvasEvent) -> None:
        pass

    def busy(self) -> bool:
        return self.body is not None

    def cancel(self) -> None:
        self.body = None
        self.canvas.update()

    def overlay(self, p: QPainter) -> None:
        if self.body is not None and self.hover is not None:
            p.setPen(_draft_pen(dash=True))
            p.drawLine(self.canvas.world_to_screen(*self.body[1]), self.canvas.world_to_screen(*self.hover))


class JointAngleTool(Tool):
    tool_id = "sensor_joint_angle"

    def press(self, e: CanvasEvent) -> None:
        joint = self.canvas.pick_joint(e.world)
        if joint is None:
            self.canvas.message(S.EDC_ERR_CLICK_JOINT)
            return
        self.run(lambda: add_joint_angle(self.doc, joint))
        self.canvas.update()

    def double_click(self, e: CanvasEvent) -> None:
        pass


class FuelSensorTool(Tool):
    """탱크 표시나 물체를 누른다: 그 물체의 추진기가 쓰는 탱크, 없으면 첫 탱크."""

    tool_id = "sensor_fuel"

    def press(self, e: CanvasEvent) -> None:
        all_tanks = tanks(self.doc)
        if not all_tanks:
            self.canvas.message(S.EDC_ERR_NO_TANK)
            return
        tank = self.canvas.pick_tank(e.world)
        if tank is None:
            body = self.canvas.pick_body(e.world)
            if body is not None:
                tank = next((t for t in all_tanks if tank_body(self.doc, t) == body.id), None)
        tank = tank or all_tanks[0]
        self.run(lambda: add_fuel_sensor(self.doc, tank))
        self.canvas.update()

    def double_click(self, e: CanvasEvent) -> None:
        pass


class RegionTool(Tool):
    def __init__(self, canvas: EditorCanvas, kind: str) -> None:
        super().__init__(canvas)
        self.kind = kind
        self.tool_id = next(t for t, k in REGION_KINDS.items() if k == kind)
        self.draft = _RectDraft()

    def press(self, e: CanvasEvent) -> None:
        self.draft.begin(self.canvas, e, (0.0, 0.0, 0.0))

    def move(self, e: CanvasEvent) -> None:
        self.draft.update(self.canvas, e)
        self.canvas.update()

    def release(self, e: CanvasEvent) -> None:
        self.draft.update(self.canvas, e)
        r = self.draft.rect()
        dragged = self.draft.dragged
        self.draft.clear()
        if r is None:
            return
        center, w, h = r
        if not dragged or w <= 1e-9 or h <= 1e-9:
            w, h = DEFAULT_REGION
            center = self.canvas.snap(e.world, e)
        self.run(lambda: add_region(self.doc, self.kind, center, w, h))
        self.canvas.update()

    def double_click(self, e: CanvasEvent) -> None:
        pass

    def busy(self) -> bool:
        return self.draft.a is not None

    def cancel(self) -> None:
        self.draft.clear()
        self.canvas.update()

    def overlay(self, p: QPainter) -> None:
        self.draft.paint(self.canvas, p)


class CheckpointTool(Tool):
    """선마다 두 점을 누른다. Enter로 묶음을 끝낸다(첫 선이 출발선, 한 바퀴 돌기 켬)."""

    tool_id = "checkpoints"

    def __init__(self, canvas: EditorCanvas) -> None:
        super().__init__(canvas)
        self.lines: list[tuple[Point, Point]] = []
        self.first: Point | None = None
        self.hover: Point | None = None

    def press(self, e: CanvasEvent) -> None:
        q = self.canvas.snap(e.world, e)
        if self.first is None:
            self.first = q
        else:
            if math.dist(self.first, q) > 1e-9:
                self.lines.append((self.first, q))
            self.first = None
        self.canvas.update()

    def double_click(self, e: CanvasEvent) -> None:
        self.press(e)

    def move(self, e: CanvasEvent) -> None:
        self.hover = self.canvas.snap(e.world, e)
        if self.first is not None or self.lines:
            self.canvas.update()

    def key(self, event: QKeyEvent) -> bool:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.finish()
            return True
        if event.key() == Qt.Key.Key_Backspace and (self.first is not None or self.lines):
            if self.first is not None:
                self.first = None
            else:
                self.lines.pop()
            self.canvas.update()
            return True
        return False

    def finish(self) -> None:
        if not self.lines:
            self.canvas.message(S.EDC_ERR_CHECKPOINT_FEW)
            return
        lines = list(self.lines)
        if self.run(lambda: add_checkpoints(self.doc, lines, loop=True)) is not None:
            self.lines = []
            self.first = None
        self.canvas.update()

    def busy(self) -> bool:
        return self.first is not None or bool(self.lines)

    def cancel(self) -> None:
        self.lines = []
        self.first = None
        self.canvas.update()

    def overlay(self, p: QPainter) -> None:
        c = self.canvas
        p.setPen(_draft_pen(2.0))
        for i, (a, b) in enumerate(self.lines):
            p.drawLine(c.world_to_screen(*a), c.world_to_screen(*b))
            p.drawText(c.world_to_screen(*a) + QPointF(4, -4), str(i))
        if self.first is not None:
            p.setBrush(QBrush(COLOR_DRAFT))
            p.drawEllipse(c.world_to_screen(*self.first), 3.0, 3.0)
            if self.hover is not None:
                p.setPen(_draft_pen(dash=True))
                p.drawLine(c.world_to_screen(*self.first), c.world_to_screen(*self.hover))


class SpawnTool(Tool):
    """눌러서 시작 위치, 끌어서 시작 각도(수식 칸은 그대로 둔다)."""

    tool_id = "spawn"

    def __init__(self, canvas: EditorCanvas) -> None:
        super().__init__(canvas)
        self.pos: Point | None = None
        self.angle: float | None = None
        self.start = QPointF()

    def press(self, e: CanvasEvent) -> None:
        self.pos = self.canvas.snap(e.world, e)
        self.angle = None
        self.start = e.screen

    def move(self, e: CanvasEvent) -> None:
        if self.pos is None:
            return
        if (e.screen - self.start).manhattanLength() >= 6:
            self.angle = self.canvas.snap_dir(math.atan2(e.world[1] - self.pos[1], e.world[0] - self.pos[0]), e)
        self.canvas.update()

    def release(self, e: CanvasEvent) -> None:
        if self.pos is None:
            return
        self.move(e)
        pos, angle = self.pos, self.angle
        self.pos = None
        skipped = set_spawn(self.doc, pos[0], pos[1], angle)
        if skipped:
            self.canvas.message(S.EDC_LOCKED_SPAWN.format(fields=", ".join(skipped)))
        self.canvas.update()

    def double_click(self, e: CanvasEvent) -> None:
        pass

    def busy(self) -> bool:
        return self.pos is not None

    def cancel(self) -> None:
        self.pos = None
        self.canvas.update()

    def overlay(self, p: QPainter) -> None:
        if self.pos is None:
            return
        c = self.canvas.world_to_screen(*self.pos)
        p.setPen(_draft_pen(2.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, 7.0, 7.0)
        if self.angle is not None:
            p.drawLine(c, c + QPointF(30 * math.cos(self.angle), -30 * math.sin(self.angle)))


# ---- 선택·이동·회전 ----


@dataclass
class _Drag:
    kind: str
    """move, band, rotate, spawn, spawn_dir, cp_end, joint, thruster, thruster_tip, ray"""
    pick: Pick | None
    start_world: Point
    start_screen: QPointF
    base: dict[str, Any]
    key: str
    additive: bool = False
    current_screen: QPointF | None = None


class SelectTool(Tool):
    """누르면 선택(Shift: 더하기·빼기), 빈 곳을 끌면 사각형 선택, 부품을 끌면 이동, 회전 손잡이로 회전.
    Delete로 지우기, 화살표로 격자 한 칸(Shift: 열 칸) 옮기기."""

    tool_id = "select"
    cursor = Qt.CursorShape.ArrowCursor
    _counter = 0

    def __init__(self, canvas: EditorCanvas) -> None:
        super().__init__(canvas)
        self.drag: _Drag | None = None
        self.moved = False
        self.press_alt = False

    def _new_key(self, kind: str) -> str:
        SelectTool._counter += 1
        return f"edc-{kind}-{SelectTool._counter}"

    # ---- 마우스 ----
    def press(self, e: CanvasEvent) -> None:
        c, doc = self.canvas, self.doc
        self.moved = False
        self.press_alt = e.alt
        hit = c.pick(e.world)
        base = copy.deepcopy(doc.data)
        if hit is None:
            if not e.shift:
                doc.select([])
            self.drag = _Drag("band", None, e.world, e.screen, base, "", additive=e.shift)
            return
        if hit.ref is not None:
            sel = list(doc.selection)
            if e.shift:
                if hit.ref in sel and hit.part in ("body", "region", "cp_line", "badge", "gen"):
                    sel.remove(hit.ref)
                    doc.select(sel)
                    self.drag = None
                    return
                if hit.ref not in sel:
                    doc.select([*sel, hit.ref])
            elif hit.ref not in sel:
                doc.select([hit.ref])
        kinds = {"body": "move", "region": "move", "cp_line": "move", "rotate": "rotate", "spawn": "spawn",
                 "spawn_dir": "spawn_dir", "cp_end": "cp_end", "joint": "joint", "thruster": "thruster",
                 "thruster_tip": "thruster_tip", "ray": "ray"}
        kind = kinds.get(hit.part)
        if kind is None:
            self.drag = None
            return
        if self._locked(kind, hit):
            c.message(S.EDC_LOCKED_FORMULA if not self._placed(hit) else S.EDC_LOCKED_PLACED)
            self.drag = None
            return
        self.drag = _Drag(kind, hit, e.world, e.screen, base, self._new_key(kind))

    def _placed(self, hit: Pick) -> bool:
        item = self.doc.item(hit.ref) if hit.ref else None
        return bool(hit.part == "body" and item and item.get("place_by"))

    def _locked(self, kind: str, hit: Pick) -> bool:
        """누른 부품을 끌 수 없나(수식, 관절 배치). 사각형 선택 중이면 다른 선택 부품은 따로 건너뛴다."""
        doc = self.doc
        item = doc.item(hit.ref) if hit.ref else None
        if kind == "move":
            return hit.ref is None or hit.ref[0] not in ("bodies", "regions", "checkpoints") or move_locked(doc, hit.ref)
        if kind == "rotate":
            return item is None or is_formula(item.get("angle", 0.0)) or bool(item.get("place_by"))
        if kind == "spawn":
            sp = spawn_data(doc)
            return is_formula(sp.get("x", 0.0)) and is_formula(sp.get("y", 0.0))
        if kind == "spawn_dir":
            return is_formula(spawn_data(doc).get("angle", 0.0))
        if item is None:
            return True
        if kind == "cp_end":
            k, end = divmod(hit.index, 2)
            try:
                return vec_locked(item["lines"][k][end])
            except (KeyError, IndexError, TypeError):
                return True
        if kind == "joint":
            names = ["anchor_a", "anchor_b"] if item.get("kind") != "spring" else [["anchor_a", "anchor_b"][hit.index]]
            return all(vec_locked(item.get(n, [0.0, 0.0])) for n in names)
        if kind == "thruster":
            return vec_locked(item.get("at", [0.0, 0.0]))
        if kind == "thruster_tip":
            return is_formula(item.get("dir", 0.0))
        if kind == "ray":
            return vec_locked(item.get("origin", [0.0, 0.0]))
        return True

    def move(self, e: CanvasEvent) -> None:
        d = self.drag
        if d is None:
            self._hover(e)
            return
        if not self.moved and (e.screen - d.start_screen).manhattanLength() < 3:
            return
        self.moved = True
        if d.kind == "band":
            d.current_screen = e.screen
            self.canvas.update()
            return
        self.canvas.begin_live_drag()
        handler = getattr(self, f"_drag_{d.kind}")
        handler(d, e)

    def release(self, e: CanvasEvent) -> None:
        d = self.drag
        self.drag = None
        self.canvas.end_live_drag()
        if d is None:
            return
        if d.kind == "band" and self.moved:
            refs = self.canvas.refs_in_rect(QRectF(d.start_screen, e.screen).normalized())
            sel = list(self.doc.selection) if d.additive else []
            self.doc.select(sel + [r for r in refs if r not in sel])
        self.canvas.update()

    def double_click(self, e: CanvasEvent) -> None:
        self.press(e)

    def busy(self) -> bool:
        return self.drag is not None

    def cancel(self) -> None:
        """끌던 것을 취소(Esc): 이 끌기로 쌓인 되돌리기 단계를 되돌린다."""
        d = self.drag
        if d is not None and self.moved and d.kind != "band":
            stack = self.doc.undo_stack
            top = stack.command(stack.index() - 1) if stack.index() > 0 else None
            if top is not None and getattr(top, "merge_key", None) == d.key:
                stack.undo()
        self.drag = None
        self.canvas.end_live_drag()
        self.canvas.update()

    def _hover(self, e: CanvasEvent) -> None:
        hit = self.canvas.pick(e.world)
        cursor = Qt.CursorShape.ArrowCursor
        if hit is not None:
            kind = {"body": "move", "region": "move", "cp_line": "move", "rotate": "rotate", "spawn": "spawn",
                    "spawn_dir": "spawn_dir", "cp_end": "cp_end", "joint": "joint", "thruster": "thruster",
                    "thruster_tip": "thruster_tip", "ray": "ray"}.get(hit.part)
            if kind is None:
                cursor = Qt.CursorShape.PointingHandCursor
            elif self._locked(kind, hit):
                cursor = Qt.CursorShape.ForbiddenCursor
            else:
                cursor = Qt.CursorShape.SizeAllCursor
        self.canvas.set_hover_cursor(cursor)

    # ---- 끌기 처리 ----
    def _snapper(self, e: CanvasEvent) -> Callable[[float], float] | None:
        if self.canvas.snap_on(e) and not self.press_alt:
            step = self.canvas.snap_step
            return lambda v: snap_value(v, step)
        return None

    def _alt_event(self, e: CanvasEvent) -> CanvasEvent:
        return e.with_alt(self.press_alt)

    def _world_delta(self, d: _Drag, e: CanvasEvent) -> Point:
        return (e.world[0] - d.start_world[0], e.world[1] - d.start_world[1])

    def _frame_angle(self, ref: Ref) -> float:
        """부품 좌표값이 적힌 좌표계의 각도(에이전트 물체: 시작 각도)."""
        if ref[0] == "bodies":
            item = self.doc.item(ref) or {}
            return self.canvas.owner_frame(item.get("owner", "agent"))[2]
        return 0.0

    def _drag_move(self, d: _Drag, e: CanvasEvent) -> None:
        assert d.pick is not None and d.pick.ref is not None
        doc = self.doc
        dw = self._world_delta(d, e)
        refs = [d.pick.ref] + [r for r in doc.selection if r != d.pick.ref and r[0] in ("bodies", "regions", "checkpoints")]
        snap = self._snapper(e)
        # 누른 부품을 격자에 맞춘 이동량을 세계 좌표로 되돌려 나머지에 같은 이동을 준다.
        lead = d.pick.ref
        la = self._frame_angle(lead)
        ld = rotate_delta(dw, -la)
        lead_changes = move_changes(doc, lead, ld, d.base, snap)
        snapped = self._snapped_delta(doc, lead, d.base, lead_changes) or ld
        dw = rotate_delta(snapped, la)
        changes = list(lead_changes)
        for r in refs[1:]:
            changes += move_changes(doc, r, rotate_delta(dw, -self._frame_angle(r)), d.base)
        apply_changes(doc, S.EDC_UNDO_MOVE, changes, d.key)

    def _snapped_delta(self, doc: EnvDocument, ref: Ref, base: dict[str, Any],
                       changes: list[tuple[JsonPath, Any]]) -> Point | None:
        """격자에 맞춘 뒤 실제로 옮겨진 양(파일 좌표). 계산할 수 없으면 None."""
        path = doc.item_path(ref)
        if path is None or not changes:
            return None
        item = get_path(base, path)
        new = changes[0][1]
        if ref[0] in ("bodies", "regions"):
            old = item.get("pos", [0.0, 0.0])
        else:
            old = (item.get("lines") or [[[0.0, 0.0]]])[0][0]
            new = new[0][0]
        dx = float(new[0]) - float(old[0]) if is_number(old[0]) and is_number(new[0]) else None
        dy = float(new[1]) - float(old[1]) if is_number(old[1]) and is_number(new[1]) else None
        if dx is None and dy is None:
            return None
        return (dx if dx is not None else 0.0, dy if dy is not None else 0.0)

    def _drag_rotate(self, d: _Drag, e: CanvasEvent) -> None:
        assert d.pick is not None and d.pick.ref is not None
        body = self.canvas.scene().body(d.pick.ref[1])
        path = self.doc.item_path(d.pick.ref)
        if body is None or path is None:
            return
        old = get_path(d.base, path).get("angle", 0.0)
        if not is_number(old):
            return
        center = d.pick.center or (body.pose[0], body.pose[1])
        a0 = math.atan2(d.start_world[1] - center[1], d.start_world[0] - center[0])
        a1 = math.atan2(e.world[1] - center[1], e.world[0] - center[0])
        new = float(old) + geo.wrap_angle(a1 - a0)
        new = snap_angle(new) if self._snapper(e) is not None else new
        rotate_body(self.doc, d.pick.ref, new, d.key)

    def _drag_spawn(self, d: _Drag, e: CanvasEvent) -> None:
        agent = d.base.get("agent")
        sp = agent.get("spawn") if isinstance(agent, dict) else None
        sp = sp if isinstance(sp, dict) else {}
        new, _ = shifted_vec([sp.get("x", 0.0), sp.get("y", 0.0)], self._world_delta(d, e), self._snapper(e))
        set_spawn(self.doc, new[0] if is_number(new[0]) else None, new[1] if is_number(new[1]) else None,
                  merge_key=d.key)

    def _drag_spawn_dir(self, d: _Drag, e: CanvasEvent) -> None:
        x, y, _ = self.canvas.scene().spawn
        a = math.atan2(e.world[1] - y, e.world[0] - x)
        set_spawn(self.doc, angle=self.canvas.snap_dir(a, self._alt_event(e)), merge_key=d.key)

    def _local_delta(self, body_id: str | None, dw: Point) -> Point:
        if body_id is None:
            return dw
        body = self.canvas.scene().body(body_id)
        return rotate_delta(dw, -(body.pose[2] if body else 0.0))

    def _drag_cp_end(self, d: _Drag, e: CanvasEvent) -> None:
        assert d.pick is not None and d.pick.ref is not None
        path = self.doc.item_path(d.pick.ref)
        if path is None:
            return
        k, end = divmod(d.pick.index, 2)
        lines = copy.deepcopy(get_path(d.base, path).get("lines") or [])
        try:
            lines[k][end], moved = shifted_vec(lines[k][end], self._world_delta(d, e), self._snapper(e))
        except (IndexError, TypeError):
            return
        if moved:
            apply_changes(self.doc, S.EDC_UNDO_MOVE, [((*path, "lines"), lines)], d.key)

    def _drag_joint(self, d: _Drag, e: CanvasEvent) -> None:
        assert d.pick is not None and d.pick.ref is not None
        doc = self.doc
        path = doc.item_path(d.pick.ref)
        if path is None:
            return
        j = get_path(d.base, path)
        dw = self._world_delta(d, e)
        snap = self._snapper(e)
        changes: list[tuple[JsonPath, Any]] = []
        a, b = str(j.get("a")), str(j.get("b"))
        if j.get("kind") == "spring":
            name, body = (("anchor_a", a), ("anchor_b", b))[d.pick.index]
            new, moved = shifted_vec(j.get(name, [0.0, 0.0]), self._local_delta(body, dw), snap)
            if moved:
                changes.append(((*path, name), new))
        else:
            old_a = j.get("anchor_a", [0.0, 0.0])
            new_a, moved_a = shifted_vec(old_a, self._local_delta(a, dw), snap)
            if moved_a:
                changes.append(((*path, "anchor_a"), new_a))
                # 실제로 옮겨진 양(격자 맞춤 뒤)으로 b 앵커도 옮겨 두 앵커가 겹친 채로 둔다.
                body_a = self.canvas.scene().body(a)
                da = (float(new_a[0]) - float(old_a[0]) if is_number(old_a[0]) else 0.0,
                      float(new_a[1]) - float(old_a[1]) if is_number(old_a[1]) else 0.0)
                dw = rotate_delta(da, body_a.pose[2]) if body_a else dw
            placed = (doc.item(("bodies", b)) or {}).get("place_by") == j.get("id")
            if not placed:
                new_b, moved_b = shifted_vec(j.get("anchor_b", [0.0, 0.0]), self._local_delta(b, dw))
                if moved_b:
                    changes.append(((*path, "anchor_b"), new_b))
        if changes:
            apply_changes(doc, S.EDC_UNDO_MOVE, changes, d.key)

    def _drag_thruster(self, d: _Drag, e: CanvasEvent) -> None:
        self._drag_local_vec(d, e, "at")

    def _drag_ray(self, d: _Drag, e: CanvasEvent) -> None:
        self._drag_local_vec(d, e, "origin")

    def _drag_local_vec(self, d: _Drag, e: CanvasEvent, name: str) -> None:
        assert d.pick is not None and d.pick.ref is not None
        path = self.doc.item_path(d.pick.ref)
        if path is None:
            return
        item = get_path(d.base, path)
        new, moved = shifted_vec(item.get(name, [0.0, 0.0]),
                                 self._local_delta(item.get("body"), self._world_delta(d, e)), self._snapper(e))
        if moved:
            apply_changes(self.doc, S.EDC_UNDO_MOVE, [((*path, name), new)], d.key)

    def _drag_thruster_tip(self, d: _Drag, e: CanvasEvent) -> None:
        assert d.pick is not None and d.pick.ref is not None
        path = self.doc.item_path(d.pick.ref)
        item = self.doc.item(d.pick.ref)
        if path is None or item is None:
            return
        body = self.canvas.scene().body(str(item.get("body")))
        if body is None or d.pick.center is None:
            return
        base = d.pick.center  # 추진기 위치(세계). 꼬리를 끄는 쪽이 분사, 힘은 반대쪽
        a = math.atan2(base[1] - e.world[1], base[0] - e.world[0]) - body.pose[2]
        a = self.canvas.snap_dir(geo.wrap_angle(a), self._alt_event(e))
        apply_changes(self.doc, S.EDC_UNDO_ROTATE, [((*path, "dir"), a)], d.key)

    # ---- 키 ----
    def key(self, event: QKeyEvent) -> bool:
        doc = self.doc
        k = event.key()
        if k in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and doc.selection:
            delete(doc, list(doc.selection))
            return True
        arrows = {Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0), Qt.Key.Key_Up: (0, 1), Qt.Key.Key_Down: (0, -1)}
        if k in arrows and doc.selection:
            step = self.canvas.snap_step if self.canvas.snap_enabled else 0.1
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                step *= 10
            dx, dy = arrows[k]
            changes: list[tuple[JsonPath, Any]] = []
            for r in doc.selection:
                changes += move_changes(doc, r, (dx * step, dy * step))
            if changes:
                key = "edc-nudge:" + ",".join(f"{s}/{i}" for s, i in doc.selection)
                apply_changes(doc, S.EDC_UNDO_MOVE, changes, key)
            elif any(move_locked(doc, r) for r in doc.selection):
                self.canvas.message(S.EDC_LOCKED_FORMULA)
            return True
        return False

    def overlay(self, p: QPainter) -> None:
        d = self.drag
        if d is not None and d.kind == "band" and d.current_screen is not None:
            p.setPen(_draft_pen(1.0, dash=True))
            p.setBrush(QBrush(QColor(255, 209, 102, 25)))
            p.drawRect(QRectF(d.start_screen, d.current_screen).normalized())


# ---- 바로 실행하는 도구 ----


def run_instant(tool_id: str, canvas: EditorCanvas) -> Ref | None:
    doc = canvas.doc
    try:
        if tool_id == "terrain_auto":
            return add_heightmap(doc)
        if tool_id == "track_auto":
            return add_generated_track(doc)
        if tool_id == "sensor_time":
            return add_time_left(doc)
    except ToolError as exc:
        canvas.message(str(exc))
    return None


def make_tool(tool_id: str, canvas: EditorCanvas) -> Tool:
    if tool_id in JOINT_KINDS:
        return JointTool(canvas, JOINT_KINDS[tool_id])
    if tool_id in REGION_KINDS:
        return RegionTool(canvas, REGION_KINDS[tool_id])
    classes: dict[str, type[Tool]] = {
        "select": SelectTool, "box": BoxTool, "circle": CircleTool, "poly": PolyTool,
        "terrain_line": TerrainLineTool, "track_draw": TrackDrawTool, "thruster": ThrusterTool,
        "fuel_tank": FuelTankTool, "vehicle": VehicleTool, "sensor_rays": RaysTool,
        "sensor_body_state": BodyStateTool, "sensor_contact": ContactTool, "sensor_target": TargetTool,
        "sensor_joint_angle": JointAngleTool, "sensor_fuel": FuelSensorTool, "checkpoints": CheckpointTool,
        "spawn": SpawnTool,
    }
    cls = classes.get(tool_id)
    if cls is None:
        raise KeyError(tool_id)
    return cls(canvas)

"""검사 결과의 「고치기」 (6.2.0, 설계 DU 4.3.6 + 최종 설계 D40, 3.1 fixes_for).

검사 결과(Issue)의 코드(Issue.code)로 고칠 길을 찾는다. 고치기 하나 = doc.edit 한 번 = 되돌리기 한 단계.

- 에디터 뜻 검사 여섯(document.EDITOR_LINTS)과 「아무 데도 안 붙은 물체」(unattached), 엔진 검사의 no_sensors, no_actuators,
  no_scoring, unknown_tag, unknown_ref, top_no_damping에 고치기가 있다. 그 밖의 코드는 빈 목록.
- 문서를 바로 고치지 않고 화면을 여는 고치기(「몸 틀 넣기…」, 「규칙 더하기…」)는 doc.fix_request(이름)을 낸다. 에디터 화면이 받아
  팔레트의 몸 틀 단추·규칙 탭의 더하기 메뉴를 연다(Fix.opens).
- 고를 것이 여럿인 고치기(「있는 것에서 고르기 ▾」, 「고르기 ▾」)는 고를 것마다 Fix 하나이고 Fix.menu가 같다(화면이 ▾ 메뉴로 묶음).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from evolab.app import strings as S
from evolab.app.editor.document import (
    IGNORABLE,
    IGNORED_KEY,
    KEYED_KINDS,
    EnvDocument,
    JsonPath,
    get_path,
    ignore_key,
    set_path,
)
from evolab.spec import symbols as sy
from evolab.spec.compiled import Issue

FLOOR_FRICTION = 0.5
"""「바닥 마찰 넣기」가 적는 세계의 바닥 마찰(감쇠). 빈 세계(위에서 본) 조리법과 같은 값."""
STATE_FIELDS = ["vx", "vy", "angle", "angvel"]
"""「몸 상태 센서 달기」가 다는 센서의 칸."""
FIX_CODES: tuple[str, ...] = ("unattached", "ground_is_body", "score_far_is_good", "no_keys", "motor_twice", "joint_apart",
                              "never_stops", "no_sensors", "no_actuators", "no_scoring", "unknown_tag", "unknown_ref",
                              "top_no_damping")
"""고치기가 있는 코드(설계 3.1의 열둘 + unattached)."""
OPEN_TEMPLATE_BODY = "template_body"
OPEN_RULES_ADD = "rules_add"

_WHERE_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
_KEY_POOL: tuple[tuple[str, str], ...] = (("Q", "A"), ("W", "S"), ("E", "D"), ("U", "J"), ("I", "K"), ("O", "L"),
                                         ("T", "G"), ("Y", "H"), ("Z", "X"), ("C", "V"), ("B", "M"), ("1", "2"),
                                         ("3", "4"), ("5", "6"), ("7", "8"), ("9", "0"))
"""키 자동 배정: 위·아래 짝(직접 조종의 R·N·Space·Esc는 쓰지 않는다). 방향키는 밀기가 먼저 쓴다."""


@dataclass(frozen=True)
class Fix:
    """고칠 길 하나. apply(doc)는 doc.edit 한 번(되돌리기 한 단계)이거나 화면을 여는 요청(opens) 하나다."""

    label: str
    tip: str
    apply: Callable[[EnvDocument], None]
    key: str = ""
    """FIX_LABELS의 키(시험·도움말용)."""
    menu: str = ""
    """비어 있지 않으면 같은 menu의 Fix들을 ▾ 메뉴 하나로 묶는다(메뉴 이름)."""
    opens: str = ""
    """화면을 여는 고치기면 그 이름(OPEN_*): 문서는 바뀌지 않는다."""


def where_to_path(where: str) -> JsonPath:
    """"actuators[0].keys[1]" -> ("actuators", 0, "keys", 1). 검사 결과의 where를 사전 경로로."""
    out: list[str | int] = []
    for name, index in _WHERE_TOKEN.findall(where):
        out.append(int(index) if index else name)
    return tuple(out)


def _fix(key: str, apply: Callable[[EnvDocument], None], *, label: str | None = None, menu: str = "",
         opens: str = "", tip: str | None = None) -> Fix:
    return Fix(label if label is not None else S.FIX_LABELS[key], tip if tip is not None else S.FIX_TIPS[key], apply,
               key, menu, opens)


def _edit(label_key: str, mutate: Callable[[dict[str, Any]], None], name: str = "") -> Callable[[EnvDocument], None]:
    """문서를 한 번 고치는 apply. 되돌리기 이름은 고치기 단추의 글."""
    def apply(doc: EnvDocument) -> None:
        doc.edit(S.FIX_LABELS[label_key].format(name=name), mutate)

    return apply


def _open(name: str) -> Callable[[EnvDocument], None]:
    def apply(doc: EnvDocument) -> None:
        doc.fix_request.emit(name)

    return apply


def _item_at(data: dict[str, Any], where: str, section: str) -> tuple[int, dict[str, Any]] | None:
    path = where_to_path(where)
    if len(path) < 2 or path[0] != section or not isinstance(path[1], int):
        return None
    try:
        item = data[section][path[1]]
    except (KeyError, IndexError, TypeError):
        return None
    return (path[1], item) if isinstance(item, dict) else None


def _body_id(data: dict[str, Any], where: str) -> str | None:
    got = _item_at(data, where, "bodies")
    return str(got[1].get("id")) if got is not None and isinstance(got[1].get("id"), str) else None


def _find(d: dict[str, Any], section: str, cid: str) -> dict[str, Any] | None:
    for item in d.get(section, []) if isinstance(d.get(section), list) else []:
        if isinstance(item, dict) and item.get("id") == cid:
            return item
    return None


# ---------------------------------------------------------------------------
# 코드마다 고치기
# ---------------------------------------------------------------------------


def _to_world(bid: str, kind: str, ground: bool = False) -> Callable[[dict[str, Any]], None]:
    def mutate(d: dict[str, Any]) -> None:
        b = _find(d, "bodies", bid)
        if b is None:
            return
        b["owner"] = "world"
        b["type"] = kind
        b.pop("exists", None)
        if ground:
            tags = [t for t in b.get("tags", []) or [] if isinstance(t, str)]
            if "ground" not in tags:
                tags.append("ground")
            b["tags"] = tags

    return mutate


def _near_is_good(where: str) -> Callable[[dict[str, Any]], None]:
    path = where_to_path(where)

    def mutate(d: dict[str, Any]) -> None:
        try:
            block = get_path(d, path)
        except (KeyError, IndexError, TypeError):
            return
        w = block.get("weight", 1.0)
        if isinstance(w, (int, float)) and not isinstance(w, bool):
            block["weight"] = -abs(float(w))

    return mutate


def _used_keys(d: dict[str, Any]) -> set[str]:
    used: set[str] = set()
    for a in d.get("actuators", []) or []:
        if isinstance(a, dict):
            for k, v in a.items():
                if k.startswith("keys") and isinstance(v, list):
                    used |= {x for x in v if isinstance(x, str)}
    return used


def assign_keys(d: dict[str, Any]) -> None:
    """키가 없는 구동기에 비어 있는 키를 준다(모터: 위·아래 짝, 추진기·점프: 하나, 밀기: 방향키 또는 짝 둘)."""
    used = _used_keys(d)

    def take_pair() -> tuple[str, str] | None:
        for a, b in _KEY_POOL:
            if a not in used and b not in used and a in sy.KEY_NAMES and b in sy.KEY_NAMES:
                used.update((a, b))
                return a, b
        return None

    for a in d.get("actuators", []) or []:
        if not isinstance(a, dict) or a.get("kind") not in KEYED_KINDS:
            continue
        if any(isinstance(v, list) and v for k, v in a.items() if k.startswith("keys")):
            continue
        kind = a["kind"]
        if kind == "motor":
            pair = take_pair()
            if pair:
                a["keys_pos"], a["keys_neg"] = [pair[0]], [pair[1]]
        elif kind in ("thruster", "jump"):
            pair = take_pair()
            if pair:
                a["keys"] = [pair[0]]
                used.discard(pair[1])
        elif kind == "move":
            arrows = ("Up", "Down", "Left", "Right")
            if not used & set(arrows):
                used.update(arrows)
                a["keys_up"], a["keys_down"], a["keys_left"], a["keys_right"] = (["Up"], ["Down"], ["Left"], ["Right"])
            else:
                p1, p2 = take_pair(), take_pair()
                if p1 and p2:
                    a["keys_up"], a["keys_down"], a["keys_left"], a["keys_right"] = ([p1[0]], [p1[1]], [p2[0]], [p2[1]])


def _drop_motor(where: str) -> Callable[[dict[str, Any]], None]:
    path = where_to_path(where)

    def mutate(d: dict[str, Any]) -> None:
        try:
            joint = get_path(d, path).get("joint")
        except (KeyError, IndexError, TypeError, AttributeError):
            return
        acts = d.get("actuators", [])
        first = next((k for k, a in enumerate(acts) if isinstance(a, dict) and a.get("kind") == "motor"
                      and a.get("joint") == joint), None)
        d["actuators"] = [a for k, a in enumerate(acts) if k == first or not (
            isinstance(a, dict) and a.get("kind") == "motor" and a.get("joint") == joint)]

    return mutate


def _align_joint(doc: EnvDocument, where: str) -> Callable[[dict[str, Any]], None] | None:
    """앵커 b를 지금 자리에서 앵커 a와 같은 점으로(물체는 움직이지 않는다). 계산할 수 없으면 None."""
    from evolab.app.editor.document import _anchor_gap, _start_episode

    got = _item_at(doc.data, where, "joints")
    spec = doc.spec()
    if got is None or spec is None:
        return None
    k, item = got
    ep = _start_episode(spec)
    if ep is None or k >= len(spec.file.joints):
        return None
    j = spec.file.joints[k]
    if j.a not in ep.bodies or j.b not in ep.bodies or _anchor_gap(ep, j) is None:
        return None
    st = sy.joint_stage(ep.env, j)
    world = ep.bodies[j.a].local_to_world(ep._vec(j.anchor_a, st))
    local = ep.bodies[j.b].world_to_local(world)
    value = [round(float(local.x), 6) + 0.0, round(float(local.y), 6) + 0.0]
    jid = item.get("id")

    def mutate(d: dict[str, Any]) -> None:
        joint = _find(d, "joints", str(jid))
        if joint is not None:
            joint["anchor_b"] = value

    return mutate


def _floor_friction(d: dict[str, Any]) -> None:
    d.setdefault("world", {})["damping"] = FLOOR_FRICTION


def _root_body(data: dict[str, Any]) -> str | None:
    from evolab.presets.rules import root_body

    return root_body(data)


def _state_sensor(root: str) -> Callable[[EnvDocument], None]:
    def apply(doc: EnvDocument) -> None:
        sid = doc.unique_id(f"{root}_state")
        item = {"kind": "body_state", "id": sid, "label": "", "body": root, "fields": list(STATE_FIELDS)}

        def mutate(d: dict[str, Any]) -> None:
            d.setdefault("sensors", []).append(item)

        doc.edit(S.FIX_LABELS["add_state_sensor"], mutate)

    return apply


def _set_at(where: str, value: Any, key: str) -> Callable[[EnvDocument], None]:
    path = where_to_path(where)

    def apply(doc: EnvDocument) -> None:
        doc.edit(S.FIX_LABELS[key], lambda d: set_path(d, path, value))

    return apply


def _ref_model_key(data: dict[str, Any], where: str) -> tuple[Any, str]:
    """이 경로의 참조 칸이 든 모델과 칸 이름(목록 칸이면 목록의 이름)."""
    from evolab.app.editor.forms import model_for, section_info

    path = where_to_path(where)
    key = next((p for p in reversed(path) if isinstance(p, str)), "")
    model = None
    if path and isinstance(path[0], str) and len(path) > 1 and isinstance(path[1], int):
        try:
            section = path[0]
            model = model_for(section_info(section), data[section][path[1]])
        except (KeyError, IndexError, TypeError, ValueError):
            model = None
    return model, key


def _pick_refs(doc: EnvDocument, issue: Issue) -> list[Fix]:
    from evolab.app.editor.forms import ref_candidates, ref_kind

    model, key = _ref_model_key(doc.data, issue.where)
    try:
        current = get_path(doc.data, where_to_path(issue.where))
    except (KeyError, IndexError, TypeError):
        return []
    kind = ref_kind(model, key)
    names = [c for c in ref_candidates(doc.data, kind) if c != current and not c.startswith("tag.")]
    return [_fix("pick_ref", _set_at(issue.where, c, "pick_ref"), label=c, menu=S.FIX_LABELS["pick_ref"],
                 tip=S.FIX_TIPS["pick_ref"]) for c in names]


def _world_tags(data: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for b in data.get("bodies", []) or []:
        if isinstance(b, dict) and b.get("owner", "agent") == "world":
            tags += [t for t in b.get("tags", []) or [] if isinstance(t, str) and t not in tags]
    return tags


def _ground_body(data: dict[str, Any]) -> dict[str, Any] | None:
    """태그를 붙일 세계 물체: 고정 세계 물체 가운데 첫째(땅), 없으면 세계 물체 가운데 첫째."""
    world = [b for b in data.get("bodies", []) or [] if isinstance(b, dict) and b.get("owner", "agent") == "world"]
    fixed = [b for b in world if b.get("type", "dynamic") == "static"]
    return (fixed or world or [None])[0]


def _tag_fixes(doc: EnvDocument, issue: Issue) -> list[Fix]:
    try:
        current = get_path(doc.data, where_to_path(issue.where))
    except (KeyError, IndexError, TypeError):
        return []
    if not isinstance(current, str) or not current.startswith("tag."):
        return []
    out = [_fix("pick_tag", _set_at(issue.where, "tag." + t, "pick_tag"), label=t, menu=S.FIX_LABELS["pick_tag"],
                tip=S.FIX_TIPS["pick_tag"]) for t in _world_tags(doc.data)]
    target = _ground_body(doc.data)
    if target is not None and isinstance(target.get("id"), str):
        tid, tag = str(target["id"]), current[4:]

        def mutate(d: dict[str, Any]) -> None:
            b = _find(d, "bodies", tid)
            if b is not None:
                tags = [t for t in b.get("tags", []) or [] if isinstance(t, str)]
                if tag not in tags:
                    b["tags"] = [*tags, tag]

        name = str(target.get("label") or tid)
        out.append(_fix("tag_body", _edit("tag_body", mutate, name), label=S.FIX_LABELS["tag_body"].format(name=name)))
    return out


def fixes_for(doc: EnvDocument, issue: Issue) -> list[Fix]:
    """이 검사 결과를 고칠 길(없으면 []). 코드(Issue.code)로 고른다."""
    code = issue.code
    data = doc.data
    if code == "unattached":
        bid = _body_id(data, issue.where)
        if bid is None:
            return []
        return [_fix("to_world_static", _edit("to_world_static", _to_world(bid, "static"))),
                _fix("to_world_dynamic", _edit("to_world_dynamic", _to_world(bid, "dynamic")))]
    if code == "ground_is_body":
        bid = _body_id(data, issue.where)
        return [] if bid is None else [_fix("to_ground", _edit("to_ground", _to_world(bid, "static", ground=True)))]
    if code == "score_far_is_good":
        return [_fix("near_is_good", _edit("near_is_good", _near_is_good(issue.where)))]
    if code == "no_keys":
        return [_fix("assign_keys", _edit("assign_keys", assign_keys))]
    if code == "motor_twice":
        return [_fix("drop_motor", _edit("drop_motor", _drop_motor(issue.where)))]
    if code == "joint_apart":
        mutate = _align_joint(doc, issue.where)
        return [] if mutate is None else [_fix("align_joint", _edit("align_joint", mutate))]
    if code in ("never_stops", "top_no_damping"):
        return [_fix("add_floor_friction", _edit("add_floor_friction", _floor_friction))]
    if code == "no_sensors":
        out = [_fix("add_body_template", _open(OPEN_TEMPLATE_BODY), opens=OPEN_TEMPLATE_BODY)]
        root = _root_body(data)
        if root is not None:
            out.append(_fix("add_state_sensor", _state_sensor(root)))
        return out
    if code == "no_actuators":
        return [_fix("add_body_template", _open(OPEN_TEMPLATE_BODY), opens=OPEN_TEMPLATE_BODY)]
    if code == "no_scoring":
        return [_fix("add_rule", _open(OPEN_RULES_ADD), opens=OPEN_RULES_ADD)]
    if code == "unknown_tag":
        return _tag_fixes(doc, issue)
    if code == "unknown_ref":
        return _pick_refs(doc, issue)
    return []


def ignore(doc: EnvDocument, issue: Issue) -> bool:
    """「괜찮습니다」: 이 살펴볼 것을 이 문서에서 숨긴다(editor.ignored, 되돌리기 한 단계). 숨길 수 없는 코드면 False."""
    if issue.code not in IGNORABLE or issue.severity != "info":
        return False
    key = ignore_key(issue)

    def mutate(d: dict[str, Any]) -> None:
        editor = d.setdefault("editor", {})
        rows = editor.get(IGNORED_KEY)
        rows = [r for r in rows if isinstance(r, str)] if isinstance(rows, list) else []
        if key not in rows:
            rows.append(key)
        editor[IGNORED_KEY] = rows

    doc.edit(S.FIX_IGNORE, mutate)
    return True


def can_ignore(issue: Issue) -> bool:
    return issue.code in IGNORABLE and issue.severity == "info"


__all__ = ["FIX_CODES", "FLOOR_FRICTION", "Fix", "OPEN_RULES_ADD", "OPEN_TEMPLATE_BODY", "assign_keys", "can_ignore",
           "fixes_for", "ignore", "where_to_path"]

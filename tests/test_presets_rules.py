"""6.2.0 묶음 P: 규칙 ↔ 조각 (스포츠 설계 6.1 T16, D11·D39).

더하기 → 같은 값으로 알아봄 → 빼면 사전이 처음과 같음. 값을 안 바꾼 set_rule은 사전 바이트 그대로. 내장 예제 6개는 알아본 수가 고정값이고
읽기만으로 바뀌지 않으며 모두 editable == False. 비슷하지만 다른 조각 200개는 알아보지 않는다.
"""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import pytest

from evolab.app import strings as S
from evolab.presets import rules as R
from evolab.spec.io import compile_env

ROOT = Path(__file__).resolve().parents[1]
OLD_SIX = ("lander", "track_car", "hill_car", "jump", "glider", "wheels")


def scene(world: str = "side") -> dict[str, Any]:
    """몸(몸통 + 다리 둘) + 땅 + 공 + 골 영역이 있는 작은 장면."""
    data: dict[str, Any] = {
        "version": 1, "name": "rules",
        "world": {"gravity": [0.0, -9.81] if world == "side" else [0.0, 0.0], "max_time": 12.0},
        "bodies": [
            {"id": "ground", "label": "땅", "owner": "world", "type": "static", "tags": ["ground"],
             "shapes": [{"kind": "box", "w": 40.0, "h": 1.0}], "pos": [0.0, -0.5]},
            {"id": "ball", "label": "공", "owner": "world", "type": "dynamic", "tags": ["ball"],
             "shapes": [{"kind": "circle", "r": 0.25}], "pos": [3.0, 0.3]},
            {"id": "torso", "label": "몸통", "pos": [0.0, 1.0], "shapes": [{"kind": "box", "w": 0.3, "h": 0.6}]},
            {"id": "leg", "label": "다리", "place_by": "hip", "shapes": [{"kind": "box", "w": 0.1, "h": 0.5}]},
        ],
        "joints": [{"id": "hip", "kind": "pivot", "a": "torso", "b": "leg", "anchor_a": [0.0, -0.3],
                    "anchor_b": [0.0, 0.25]}],
        "actuators": [{"kind": "motor", "id": "m_hip", "joint": "hip", "max_torque": 50.0, "max_speed": 5.0}],
        "sensors": [{"kind": "joint_angle", "id": "a_hip", "joint": "hip"}],
        "regions": [{"id": "goal", "label": "골", "kind": "goal", "pos": [8.0, 0.5],
                     "shape": {"kind": "box", "w": 1.0, "h": 2.0}}],
        "termination": [{"event": "timeout", "outcome": "neutral", "label": "시간 끝"}],
    }
    if world == "top":
        data["world"]["view"] = "top"
        data["version"] = 4
    return data


CASES: list[tuple[str, dict[str, Any]]] = [
    ("distance", {"part": "torso", "direction": "right", "per_m": 1.0}),
    ("distance", {"part": "leg", "direction": "up", "per_m": 2.5}),
    ("fall", {"parts": ["torso", "leg"], "what": "tag.ground", "root": "torso", "height": 0.8, "penalty": 1.0}),
    ("fall", {"parts": ["torso"], "what": "tag.ground", "root": "torso", "height": 0.0, "penalty": 0.0}),
    ("touch_ends", {"part": "torso", "what": "tag.ground", "outcome": "failure"}),
    ("touch_ends", {"part": "leg", "what": "ball", "outcome": "success"}),
    ("touch_penalty", {"part": "torso", "what": "tag.ground", "per_s": 0.5}),
    ("reach", {"part": "torso", "where": "goal", "outcome": "success", "bonus": 100.0}),
    ("reach", {"part": "torso", "where": "goal", "outcome": "neutral", "bonus": 0.0}),
    ("near", {"thing": "ball", "where": "goal", "per_m": 1.0, "when": "min"}),
    ("near", {"thing": "torso", "where": "ball", "per_m": 2.0, "when": "final"}),
    ("survive", {"per_s": 1.0}),
    ("upright", {"part": "torso", "weight": 0.5}),
    ("save_energy", {"weight": 0.05}),
    ("out_ends", {"outcome": "failure"}),
]


def _editable(data: dict[str, Any]) -> list[R.RuleUse]:
    return [u for u in R.find_rules(data)[0] if u.editable and not u.fixed]


@pytest.mark.parametrize(("kind", "args"), CASES, ids=[f"{k}-{i}" for i, (k, _) in enumerate(CASES)])
def test_add_find_remove_round_trip(kind: str, args: dict[str, Any]) -> None:
    data = scene()
    before = copy.deepcopy(data)
    use = R.add_rule(data, kind, **args)
    assert use.editable and not use.fixed and use.kind == kind
    assert compile_env(copy.deepcopy(data)) is not None  # 만든 조각은 컴파일된다
    (found,) = _editable(data)
    assert (found.kind, dict(found.args)) == (kind, args)
    assert found.event == use.event and found.blocks == use.blocks and set(found.refs) == set(use.refs)
    assert data["editor"]["rules"] == [{"kind": kind, "event": use.event, "blocks": list(use.blocks)}]
    text = json.dumps(data, sort_keys=True)
    again = R.set_rule(data, found, **args)  # 값을 안 바꾼 고치기: 바이트 그대로
    assert json.dumps(data, sort_keys=True) == text and again.args == found.args
    R.remove_rule(data, found)
    assert data == before


def test_rule_ids_follow_the_table_and_number_on_collision() -> None:
    data = scene()
    a = R.add_rule(data, "fall", parts=["torso"], what="tag.ground", root="torso", height=0.5, penalty=2.0)
    b = R.add_rule(data, "fall", parts=["leg"], what="tag.ground", root="torso", height=0.0, penalty=1.0)
    d = R.add_rule(data, "distance", part="torso", direction="right", per_m=1.0)
    assert (a.event, a.blocks, b.event, b.blocks, d.blocks) == ("fall", ("s_fall",), "fall_2", ("s_fall_2",), ("s_dist",))
    ev = {e["id"]: e for e in data["events"]}
    assert ev["fall"]["when"] == "contact(torso, tag.ground) or torso.y < 0.5"
    assert ev["fall_2"]["when"] == "contact(leg, tag.ground)"
    ends = [t["event"] for t in data["termination"]]
    assert ends == ["fall", "fall_2", "timeout"]  # 새 종료 줄은 시간 끝 줄 앞에
    assert [u.kind for u in R.find_rules(data)[0]] == ["fall", "fall", "distance", "time_limit"]


def test_set_rule_changes_only_the_blank() -> None:
    data = scene()
    use = R.add_rule(data, "near", thing="ball", where="goal", per_m=1.0, when="min")
    block = data["scoring"]["blocks"][0]
    block["label"] = "손으로 바꾼 이름"
    new = R.set_rule(data, use, per_m=3.0)
    assert block == {"kind": "distance_to", "id": "s_near", "label": "손으로 바꾼 이름", "body": "ball", "target": "goal",
                     "agg": "min", "weight": -3.0}
    assert new.args["per_m"] == 3.0 and new.editable
    R.set_rule(data, new, when="final", thing="torso")
    assert (block["agg"], block["body"]) == ("final", "torso")


def test_penalty_and_bonus_add_or_drop_their_block() -> None:
    data = scene()
    use = R.add_rule(data, "fall", parts=["torso"], what="tag.ground", root="torso", height=0.0, penalty=0.0)
    assert use.blocks == () and "scoring" not in data
    use = R.set_rule(data, use, penalty=2.0)
    assert use.blocks == ("s_fall",) and data["scoring"]["blocks"][0]["weight"] == -2.0
    assert data["editor"]["rules"][0]["blocks"] == ["s_fall"]
    use = R.set_rule(data, use, penalty=0.0)
    assert use.blocks == () and "scoring" not in data and data["editor"]["rules"][0]["blocks"] == []
    reach = R.add_rule(data, "reach", part="torso", where="goal", outcome="success", bonus=0.0)
    reach = R.set_rule(data, reach, bonus=50.0)
    assert [u.kind for u in _editable(data)] == ["fall", "reach"] and reach.blocks == ("s_reach",)


def test_time_limit_is_a_fixed_row() -> None:
    data = scene()
    uses, _ = R.find_rules(data)
    limit = uses[-1]
    assert (limit.kind, limit.fixed, limit.editable, limit.args["seconds"]) == ("time_limit", True, True, 12.0)
    with pytest.raises(ValueError):
        R.remove_rule(data, limit)
    R.set_rule(data, limit, seconds=30.0)
    assert data["world"]["max_time"] == 30.0 and "editor" not in data
    text = json.dumps(data)
    R.set_rule(data, R.find_rules(data)[0][-1], seconds=30.0)
    assert json.dumps(data) == text
    empty: dict[str, Any] = {"name": "e"}
    R.add_rule(empty, "time_limit", seconds=15.0)
    assert empty["world"]["max_time"] == 15.0 and empty["termination"][0]["event"] == "timeout"
    assert R.available(data, "time_limit")[0] is False


def test_read_only_rules_refuse_changes() -> None:
    data = scene()
    use = R.add_rule(data, "touch_ends", part="torso", what="tag.ground", outcome="failure")
    del data["editor"]  # 적혀 있지 않은 것은 읽기 전용
    (ro,) = [u for u in R.find_rules(data)[0] if not u.fixed]
    assert not ro.editable and ro.args == use.args
    with pytest.raises(ValueError):
        R.set_rule(data, ro, outcome="success")
    with pytest.raises(ValueError):
        R.remove_rule(data, ro)


def test_recorded_rule_that_is_not_closed_is_not_editable() -> None:
    """D39: 적혀 있어도 다른 것이 그 사건을 가리키면(언덕차의 crash처럼 −20 블록) 고칠 수 없다."""
    data = scene()
    R.add_rule(data, "touch_ends", part="torso", what="tag.ground", outcome="failure")
    data.setdefault("scoring", {}).setdefault("blocks", []).append(
        {"kind": "event", "id": "s_crash", "event": "touch", "weight": -20.0})
    assert _editable(data) == []
    uses, rest = R.find_rules(data)
    assert [u.kind for u in uses] == ["time_limit"] and rest == 3


# 내장 예제 6개: 알아본 규칙(종류 순서)과 고급 탭에 남는 줄 수. 모두 읽기 전용(editor.rules가 없음)
BUILTIN_RULES = {
    "glider": (["touch_ends", "distance", "out_ends", "time_limit"], 0),
    "hill_car": (["distance", "save_energy", "out_ends", "time_limit"], 8),
    "jump": (["touch_ends", "survive", "out_ends", "time_limit"], 1),
    "lander": (["near", "out_ends", "time_limit"], 11),
    "track_car": (["out_ends", "time_limit"], 13),
    "wheels": (["distance", "save_energy", "out_ends", "time_limit"], 8),
}


@pytest.mark.parametrize("name", OLD_SIX)
def test_builtin_examples_are_read_only_and_fixed(name: str) -> None:
    raw = (ROOT / "environments" / f"{name}.evoenv").read_bytes()
    data = json.loads(raw)
    uses, rest = R.find_rules(data)
    assert ([u.kind for u in uses], rest) == BUILTIN_RULES[name]
    assert all(not u.editable for u in uses if not u.fixed)
    assert json.dumps(data, ensure_ascii=False) == json.dumps(json.loads(raw), ensure_ascii=False)  # 읽기만으로 안 바뀜
    for u in uses:
        assert R.sentence(data, u) and "{" not in R.sentence(data, u)
    if name == "hill_car":  # crash는 −20 블록이 함께 가리켜 문장이 아니다(D39)
        assert all(u.event != "crash" for u in uses)


def _perturbations() -> list[tuple[str, str, Any]]:
    """(규칙 종류, 흩뜨리는 방법 이름, 흩뜨리기 함수(data, use) -> None)."""

    def ev(data: dict[str, Any], use: R.RuleUse) -> dict[str, Any]:
        return next(e for e in data["events"] if e["id"] == use.event)

    def blk(data: dict[str, Any], use: R.RuleUse) -> dict[str, Any]:
        return next(b for b in data["scoring"]["blocks"] if b["id"] == use.blocks[0])

    def end(data: dict[str, Any], use: R.RuleUse) -> dict[str, Any]:
        return next(t for t in data["termination"] if t["event"] == use.event)

    def mention_event(data: dict[str, Any], use: R.RuleUse) -> None:
        data.setdefault("events", []).append({"id": "after", "when": f"{use.event} and time > 1"})

    def mention_block(data: dict[str, Any], use: R.RuleUse) -> None:
        data.setdefault("scoring", {})["formula"] = f"{use.blocks[0]} * 2"

    def second_end(data: dict[str, Any], use: R.RuleUse) -> None:
        data["termination"].insert(0, {"event": use.event, "outcome": "neutral"})

    def stray_block(data: dict[str, Any], use: R.RuleUse) -> None:
        data.setdefault("scoring", {}).setdefault("blocks", []).append(
            {"kind": "event", "id": "stray", "event": use.event, "weight": 3.0, "value": "count"})

    event_kinds = ("fall", "touch_ends", "reach")
    out: list[tuple[str, str, Any]] = []
    for kind in event_kinds:
        out += [
            (kind, "hold", lambda d, u: ev(d, u).update(hold=0.5)),
            (kind, "and", lambda d, u: ev(d, u).update(when=ev(d, u)["when"] + " and time > 2")),
            (kind, "not", lambda d, u: ev(d, u).update(when="not " + ev(d, u)["when"])),
            (kind, "extra_key", lambda d, u: end(d, u).update(note="x")),
            (kind, "mention", mention_event),
            (kind, "second_end", second_end),
            (kind, "stray_block", stray_block),
        ]
    out += [
        ("fall", "outcome", lambda d, u: end(d, u).update(outcome="success")),
        ("fall", "height_first", lambda d, u: ev(d, u).update(when="torso.y < 0.8 or contact(torso, tag.ground)")),
        ("fall", "mixed_what", lambda d, u: ev(d, u).update(when="contact(torso, tag.ground) or contact(leg, ball)")),
        ("fall", "bonus_sign", lambda d, u: blk(d, u).update(weight=2.0)),
        ("reach", "bonus_sign", lambda d, u: blk(d, u).update(weight=-2.0)),
        ("reach", "value_count", lambda d, u: blk(d, u).update(value="count")),
    ]
    for kind in ("distance", "touch_penalty", "near", "survive", "upright", "save_energy"):
        out += [
            (kind, "sign", lambda d, u: blk(d, u).update(weight=-float(blk(d, u)["weight"]) or -1.0)),
            (kind, "mention", mention_block),
            (kind, "extra_key", lambda d, u: blk(d, u).update(note="x")),
        ]
    out += [
        ("distance", "odd_direction", lambda d, u: blk(d, u).update(direction=0.3)),
        ("touch_penalty", "two_targets", lambda d, u: blk(d, u).update({"with": ["tag.ground", "ball"]})),
        ("upright", "final", lambda d, u: blk(d, u).update(agg="final")),
        ("save_energy", "some", lambda d, u: blk(d, u).update(actuators=["m_hip"])),
        ("near", "agg_max", lambda d, u: blk(d, u).update(agg="max")),
        ("out_ends", "extra_key", lambda d, u: next(t for t in d["termination"]
                                                    if t["event"] == "out_of_bounds").update(note="x")),
        ("out_ends", "second", lambda d, u: d["termination"].insert(0, {"event": "out_of_bounds",
                                                                        "outcome": "neutral"})),
    ]
    return out


def test_two_hundred_near_miss_fragments_are_not_recognised() -> None:
    perturb = _perturbations()
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for kind, args in CASES:
        by_kind.setdefault(kind, []).append(args)
    rng = random.Random(20261113)
    checked = 0
    for k in range(200):
        kind, how, fn = perturb[k % len(perturb)] if k < len(perturb) else rng.choice(perturb)
        options = [a for a in by_kind[kind] if not (kind in ("fall", "reach") and how in ("bonus_sign", "value_count")
                                                     and not R.add_rule(scene(), kind, **a).blocks)]
        if kind == "fall":  # 높이도 감점도 없는 fall은 글자 그대로 「닿으면 끝(실패)」과 같은 조각이라 뺀다
            options = [a for a in options if a["height"] > 0 or a["penalty"] > 0]
        args = rng.choice(options)
        data = scene()
        if rng.random() < 0.5:  # 다른 규칙 하나가 옆에 있어도 같다
            R.add_rule(data, "survive" if kind != "survive" else "save_energy", **(
                {"per_s": 1.0} if kind != "survive" else {"weight": 0.05}))
        use = R.add_rule(data, kind, **args)
        fn(data, use)
        uses, rest = R.find_rules(data)
        mine = [u for u in uses if set(u.refs) & set(use.refs)]
        assert mine == [], (kind, how, args, [(u.kind, u.args, u.editable) for u in mine])
        assert rest >= 1, (kind, how)
        checked += 1
    assert checked == 200


def test_candidates_available_and_sentence() -> None:
    data = scene()
    slots = {s.id: s for s in R.RULES["touch_ends"].slots}
    assert R.candidates(data, slots["part"]) == [("torso", "몸통"), ("leg", "다리")]
    what = R.candidates(data, slots["what"])
    assert what[:2] == [("tag.ground", "땅"), ("tag.ball", "공")] and ("ball", "공") in what
    assert R.candidates(data, slots["outcome"]) == [(k, S.RULE_OUTCOMES[k]) for k in ("failure", "success", "neutral")]
    assert R.candidates(data, {s.id: s for s in R.RULES["reach"].slots}["where"])[0] == ("goal", "골")
    assert R.available(data, "fall") == (True, "")
    ok, why = R.available(scene("top"), "fall")
    assert not ok and why == S.TPL_WRONG_WORLD.format(world=S.TPL_WORLD_NAMES["side"])
    no_region = scene()
    no_region["regions"] = []
    ok, why = R.available(no_region, "reach")
    assert not ok and S.RULE_NEEDS_WHAT["region"] in why and "이(가)" not in why
    use = R.add_rule(data, "distance", part="torso", direction="right", per_m=1.0)
    assert R.sentence(data, use) == "몸통이 오른쪽으로 간 거리만큼 점수 (1m에 1점)"
    R.add_rule(data, "out_ends", outcome="failure")
    assert R.available(data, "out_ends") == (False, S.RULES_ALREADY)
    data["scoring"]["formula"] = "s_dist"
    assert R.available(data, "survive") == (False, S.RULES_FORMULA_NOTE)


def test_summary_line() -> None:
    data = scene()
    assert R.summary(data) == S.RULE_SUMMARY_NONE
    R.add_rule(data, "distance", part="torso", direction="right", per_m=1.0)
    R.add_rule(data, "fall", parts=["torso"], what="tag.ground", root="torso", height=0.0, penalty=1.0)
    R.add_rule(data, "save_energy", weight=0.05)
    R.add_rule(data, "near", thing="ball", where="goal", per_m=2.0, when="min")
    assert R.summary(data) == "앞으로 간 거리 × 1 − 넘어짐 × 1 − 힘 × 0.05 − 공에서 골까지 거리 × 2"
    data["scoring"]["blocks"].append({"kind": "final_speed", "id": "fast", "body": "torso"})
    assert R.summary(data).endswith(" + " + S.RULE_SUMMARY_TERMS["other"].format(n=1))
    data["scoring"]["formula"] = "fast"
    assert R.summary(data) == S.RULES_FORMULA_NOTE


def test_rule_table_matches_the_strings() -> None:
    assert tuple(R.RULES) == tuple(S.RULE_TITLES)
    for kind, rk in R.RULES.items():
        fields = {f[1] for f in __import__("string").Formatter().parse(S.RULE_SENTENCES[kind]) if f[1]}
        assert fields == {s.id for s in rk.slots}, kind
    assert math.isclose(R.DIRECTIONS["left"], math.pi)

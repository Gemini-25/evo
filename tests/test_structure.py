"""v2.1 0'단계: 몸 구조(물체의 있음 칸)와 형식 3 (v2 설계 2.1·2.2·2.3·4.11·D33~D37, 테스트 T57~T60).

- spec/structure.py의 순수 함수: 뿌리, 관절 이어짐, 이어짐 규칙(없는 물체 = OFF ∪ (FULL − KEPT)), 없어질 수 있는 물체,
  몸 수치에서 있는 물체(present), 「몸 구조」 타일 열쇠(structure_key), Episode.params와 같은 매개변수 사전.
- 형식 3: exists 하나만 써도 형식 3 파일, 지우면 처음 글·해시로 돌아옴. v2.0 프로그램(형식 2까지)은 「파일 버전 3」으로 거부.
- 검사: 있음 칸의 수식은 만들기 단계 structure(p.*만).

런타임(Episode.absent)과 검사 규칙(SPEC_EXISTS_*)은 A' 묶음이 넣었다(그 묶음의 테스트: tests/test_structure_runtime.py).
"""

from __future__ import annotations

import copy
import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab.app import strings as S
from evolab.engine import seeds
from evolab.spec import model as m
from evolab.spec import structure as st
from evolab.spec import symbols as sy
from evolab.spec.compiled import EnvFileError, check_env
from evolab.spec.io import compile_env, dump_env, env_hash, feature_version, parse_env
from evolab.world.physics2d.episode import Episode, Trial

ROOT = Path(__file__).resolve().parents[1]
ENVS = ROOT / "environments"

def _box(w: float, h: float) -> dict[str, Any]:
    return {"kind": "box", "w": w, "h": h}


# 바퀴 둘(하나는 있음 진화), 팔 사슬(팔 → 손(place_by) → 손가락(관절만)), 개수 매개변수 하나로 켜는 부품 둘,
# 뿌리와 이어지지 않은 물체 둘(늘 꺼진 loose와 거기에만 붙은 loose_child), 세계 기둥을 거쳐서만 차체와 닿는 pod.
BASE: dict[str, Any] = {
    "version": 3,
    "name": "structure test",
    "world": {"max_time": 2.0, "bounds": [-50.0, -10.0, 50.0, 50.0]},
    "params": [
        {"id": "w2_on", "label": "바퀴2 있음", "value": 0.75, "evolve": True, "min": 0.0, "max": 1.0},
        {"id": "arm_on", "label": "팔 있음", "value": 0.25, "evolve": True, "min": 0.0, "max": 1.0},
        {"id": "extra_count", "value": 1.0, "evolve": True, "min": 0.0, "max": 2.0},
        {"id": "wheel_r", "value": 0.3, "evolve": True, "min": 0.2, "max": 0.5},
    ],
    "randomize": [{"id": "rr", "dist": "uniform", "low": 0.0, "high": 1.0}],
    "bodies": [
        {"id": "ground", "owner": "world", "type": "static", "tags": ["ground"],
         "shapes": [{"kind": "box", "w": 80.0, "h": 1.0, "offset": [0.0, -0.5]}]},
        {"id": "post", "owner": "world", "type": "static", "pos": [6.0, 3.0], "shapes": [_box(0.2, 0.2)]},
        {"id": "chassis", "pos": [0.0, 1.5], "shapes": [_box(2.0, 0.4)]},
        {"id": "wheel1", "pos": [-0.8, 1.0], "shapes": [{"kind": "circle", "r": "=p.wheel_r"}], "tags": ["wheel"]},
        {"id": "wheel2", "pos": [0.8, 1.0], "exists": "=p.w2_on", "shapes": [{"kind": "circle", "r": "=p.wheel_r"}],
         "tags": ["wheel"]},
        {"id": "arm", "pos": [0.0, 2.2], "exists": "=p.arm_on", "shapes": [_box(0.2, 1.0)]},
        {"id": "hand", "place_by": "j_hand", "shapes": [_box(0.3, 0.3)]},
        {"id": "finger", "pos": [0.0, 3.1], "shapes": [_box(0.1, 0.3)]},
        {"id": "extra1", "pos": [-0.9, 1.9], "exists": "=p.extra_count > 0.5", "shapes": [_box(0.2, 0.2)]},
        {"id": "extra2", "pos": [0.9, 1.9], "exists": "=p.extra_count > 1.5", "shapes": [_box(0.2, 0.2)]},
        {"id": "loose", "pos": [4.0, 1.0], "exists": 0.25, "shapes": [_box(0.3, 0.3)]},
        {"id": "loose_child", "pos": [4.0, 1.5], "shapes": [_box(0.2, 0.2)]},
        {"id": "pod", "pos": [6.0, 2.0], "shapes": [_box(0.2, 0.2)]},
    ],
    "joints": [
        {"id": "j1", "kind": "pivot", "a": "chassis", "b": "wheel1", "anchor_a": [-0.8, -0.5]},
        {"id": "j2", "kind": "pivot", "a": "chassis", "b": "wheel2", "anchor_a": [0.8, -0.5]},
        {"id": "j_arm", "kind": "pivot", "a": "chassis", "b": "arm", "anchor_a": [0.0, 0.2], "anchor_b": [0.0, -0.5]},
        {"id": "j_hand", "kind": "weld", "a": "arm", "b": "hand", "anchor_a": [0.0, 0.5]},
        {"id": "j_finger", "kind": "pivot", "a": "hand", "b": "finger", "anchor_a": [0.0, 0.15], "anchor_b": [0.0, -0.15]},
        {"id": "j_e1", "kind": "weld", "a": "chassis", "b": "extra1", "anchor_a": [-0.9, 0.4]},
        {"id": "j_e2", "kind": "weld", "a": "chassis", "b": "extra2", "anchor_a": [0.9, 0.4]},
        {"id": "j_loose", "kind": "weld", "a": "loose", "b": "loose_child", "anchor_a": [0.0, 0.5]},
        {"id": "j_post_pod", "kind": "spring", "a": "post", "b": "pod", "stiffness": 10.0},
        {"id": "j_post_chassis", "kind": "spring", "a": "post", "b": "chassis", "stiffness": 10.0},
    ],
    "actuators": [
        {"kind": "motor", "id": "m1", "joint": "j1", "max_torque": 50.0, "max_speed": 10.0},
        {"kind": "motor", "id": "m2", "joint": "j2", "max_torque": 50.0, "max_speed": 10.0},
        {"kind": "motor", "id": "m_arm", "joint": "j_arm", "max_torque": 20.0, "max_speed": 3.0},
        {"kind": "thruster", "id": "push", "body": "hand", "max_force": 10.0},
    ],
    "sensors": [
        {"kind": "body_state", "id": "cs", "body": "chassis", "fields": ["vx", "angle"]},
        {"kind": "contact", "id": "w2_down", "body": "wheel2", "with": ["tag.ground"]},
        {"kind": "body_state", "id": "w2s", "body": "wheel2", "fields": ["vx"]},
        {"kind": "rays", "id": "eye", "body": "hand", "count": 3, "spread": 1.0, "length": 5.0, "hits": ["tag.ground"]},
        {"kind": "joint_angle", "id": "elbow", "joint": "j_arm"},
    ],
    "regions": [{"id": "goal", "kind": "goal", "pos": [20.0, 1.0], "shape": _box(2.0, 2.0)}],
    "events": [{"id": "touch", "when": "contact(wheel2, tag.ground)"}],
    "termination": [],
    "scoring": {"blocks": [{"kind": "progress", "id": "travel", "body": "chassis"}]},
    "brain": {"hidden": [4]},
}
AGENTS = frozenset({"chassis", "wheel1", "wheel2", "arm", "hand", "finger", "extra1", "extra2", "loose", "loose_child",
                    "pod"})
OPTIONAL = frozenset({"wheel2", "arm", "hand", "finger", "extra1", "extra2", "loose"})


def _data(**changes: Any) -> dict[str, Any]:
    d = copy.deepcopy(BASE)
    d.update(copy.deepcopy(changes))
    return d


def _env(data: dict[str, Any] | None = None) -> m.EnvFile:
    return parse_env(copy.deepcopy(data if data is not None else BASE))


def _spec(data: dict[str, Any] | None = None) -> Any:
    return compile_env(copy.deepcopy(data if data is not None else BASE))


def _expected_absent(w2: float, arm: float, count: float) -> frozenset[str]:
    """손으로 적은 이어짐 규칙: 꺼진 팔이면 손·손가락도 없고, loose는 늘 꺼짐(loose_child는 뿌리와 안 이어져 그대로)."""
    off = {"loose"}
    if not w2 > 0.5:
        off.add("wheel2")
    if not arm > 0.5:
        off |= {"arm", "hand", "finger"}
    if not count > 0.5:
        off.add("extra1")
    if not count > 1.5:
        off.add("extra2")
    return frozenset(off)


def _params(env: m.EnvFile, **values: float) -> dict[str, float]:
    out = st.param_values(env)
    out.update({f"p.{k}": float(v) for k, v in values.items()})
    return out


# ---------------------------------------------------------------------------
# T57: 뿌리, 이어짐, present
# ---------------------------------------------------------------------------


def test_base_env_compiles_without_errors() -> None:
    spec = _spec()
    assert not [i for i in spec.issues if i.severity == "error"]
    assert feature_version(spec.file) == 3
    assert {k for k in spec.formulas if k[1] == "structure"} == {
        ("=p.w2_on", "structure"), ("=p.arm_on", "structure"), ("=p.extra_count > 0.5", "structure"),
        ("=p.extra_count > 1.5", "structure")}


def test_root_body_is_the_first_agent_body() -> None:
    env = _env()
    assert st.root_body(env) == "chassis"  # 세계 물체(ground, post)가 먼저 와도 첫 에이전트 물체
    assert st.root_body(m.EnvFile(name="빈 환경")) is None
    assert st.connected(m.EnvFile(name="빈 환경"), frozenset()) == frozenset()
    assert st.optional_bodies(m.EnvFile(name="빈 환경")) == frozenset()
    world_only = _data(bodies=BASE["bodies"][:2], joints=[], actuators=[], sensors=[], events=[],
                       scoring={"blocks": []})
    assert st.root_body(_env(world_only)) is None


def test_connected_follows_agent_joints_only() -> None:
    env = _env()
    full = st.connected(env, frozenset())
    # pod는 세계 기둥을 거쳐서만 차체와 닿으므로(세계 물체는 그래프에 없음) 뿌리와 이어지지 않는다. loose 쌍도 따로 논다.
    assert full == {"chassis", "wheel1", "wheel2", "arm", "hand", "finger", "extra1", "extra2"}
    assert st.connected(env, {"arm"}) == full - {"arm", "hand", "finger"}
    assert st.connected(env, {"hand"}) == full - {"hand", "finger"}
    assert st.connected(env, {"chassis"}) == frozenset()  # 뿌리를 빼면 아무것도 이어지지 않음
    assert st.connected(env, {"ground", "nonexistent"}) == full  # 에이전트 물체가 아닌 이름은 상관없음


def test_absent_from_chain_end_and_joint_only_parts() -> None:
    """사슬 끝: 팔을 끄면 place_by로 붙은 손과 관절로만 붙은 손가락까지 없다(env 원형의 place_by만 따라가기와 다름)."""
    env = _env()
    assert st.absent_from(env, frozenset()) == frozenset()
    assert st.absent_from(env, {"arm"}) == {"arm", "hand", "finger"}
    assert st.absent_from(env, {"hand"}) == {"hand", "finger"}
    assert st.absent_from(env, {"finger"}) == {"finger"}
    assert st.absent_from(env, {"wheel2", "extra1"}) == {"wheel2", "extra1"}


def test_absent_from_keeps_parts_with_another_path_to_the_root() -> None:
    data = _data()
    data["joints"].append({"id": "j_brace", "kind": "spring", "a": "chassis", "b": "hand", "stiffness": 5.0})
    env = _env(data)
    assert st.absent_from(env, {"arm"}) == {"arm"}  # 손은 버팀 관절로 차체와 이어져 남고, 손가락도 남는다
    assert st.absent_from(env, {"arm", "hand"}) == {"arm", "hand", "finger"}


def test_absent_from_leaves_bodies_never_connected_to_the_root() -> None:
    """뿌리와 처음부터 안 이어진 물체는 자기 있음 칸만 따른다: loose가 꺼져도 loose_child는 남는다(v1 파일 호환)."""
    env = _env()
    assert st.absent_from(env, {"loose"}) == {"loose"}
    assert st.absent_from(env, {"loose", "arm"}) == {"loose", "arm", "hand", "finger"}
    assert "pod" not in st.absent_from(env, OPTIONAL)


def test_present_matches_the_rule_for_every_combination() -> None:
    spec = _spec()
    env = spec.file
    for w2, arm, count in itertools.product((0.0, 0.5, 0.500001, 1.0), (0.25, 0.75), (0.0, 0.5, 1.0, 1.5, 2.0)):
        prm = _params(env, w2_on=w2, arm_on=arm, extra_count=count)
        want = AGENTS - _expected_absent(w2, arm, count)
        assert st.present(spec, prm) == want, (w2, arm, count)
        assert st.structure_key(spec, prm) == tuple(b.id for b in env.bodies if b.id in want and b.id in OPTIONAL)
    # 처음 몸(매개변수 값): 바퀴2 있음(0.75), 팔 없음(0.25), 부품 하나(개수 1)
    first = st.param_values(env)
    assert st.present(spec, first) == AGENTS - {"arm", "hand", "finger", "extra2", "loose"}
    assert st.structure_key(spec, first) == ("wheel2", "extra1")


def test_present_threshold_constants_and_safe_formula_values() -> None:
    data = _data()
    bodies = {b["id"]: b for b in data["bodies"]}
    bodies["wheel2"]["exists"] = 0.5  # 같으면 없음
    bodies["extra1"]["exists"] = 0.51
    bodies["extra2"]["exists"] = "=p.extra_count / 0"  # 수식 언어에서 0으로 나누면 0 → 없음
    bodies["arm"]["exists"] = "=p.arm_on * 1e308 * 10"  # 넘침은 수식 언어가 유한한 큰 값으로 자름 → 있음
    spec = _spec(data)
    prm = st.param_values(spec.file)
    here = st.present(spec, prm)
    assert "wheel2" not in here and "extra1" in here and "extra2" not in here
    assert {"arm", "hand", "finger"} <= here
    # 상수 0.51은 늘 있으므로 없어질 수 있는 물체가 아니다. 상수 0.5와 수식은 없어질 수 있다.
    assert st.optional_bodies(spec.file) == (OPTIONAL - {"extra1"})
    assert st.exists_value(spec, bodies_model(spec, "chassis"), prm) == 1.0
    assert st.exists_value(spec, bodies_model(spec, "extra2"), prm) == 0.0
    assert st.exists_value(spec, bodies_model(spec, "wheel2"), prm) == 0.5


def bodies_model(spec: Any, body_id: str) -> m.Body:
    return next(b for b in spec.file.bodies if b.id == body_id)


def test_files_without_exists_have_every_agent_body() -> None:
    for name in ("lander", "track_car", "hill_car", "jump", "glider"):
        spec = compile_env((ENVS / f"{name}.evoenv").read_bytes())
        agents = frozenset(sy.agent_body_ids(spec.file))
        assert st.optional_bodies(spec.file) == frozenset()
        prm = st.param_values(spec.file)
        assert st.present(spec, prm) == agents and st.structure_key(spec, prm) == ()
        assert not [k for k in spec.formulas if k[1] == "structure"]


def test_param_values_is_the_same_dict_as_episode_params() -> None:
    spec = _spec()
    env = spec.file
    rng = np.random.default_rng(57)
    trial = Trial(seeds.derive(9, 0))
    assert st.param_values(env) == Episode(spec, None, trial).params
    evolved = sy.evolved_params(env)
    for _ in range(20):
        body = rng.uniform(-1.0, 3.0, len(evolved))  # 범위 밖 값은 자름
        body[rng.integers(len(evolved))] = np.nan if rng.random() < 0.3 else body[0]
        assert st.param_values(env, body) == Episode(spec, body, trial).params
    with pytest.raises(ValueError):
        st.param_values(env, [0.5])


def _random_bodies(spec: Any, n: int, seed: int) -> np.ndarray:
    evolved = sy.evolved_params(spec.file)
    lo = np.array([p.min for p in evolved])
    hi = np.array([p.max for p in evolved])
    return np.random.default_rng(seed).uniform(lo, hi, (n, len(evolved)))


def test_t57_present_equals_runtime_absent_on_2000_random_bodies() -> None:
    spec = _spec()
    trial = Trial(seeds.derive(9, 1))
    for body in _random_bodies(spec, 2000, 2057):
        ep = Episode(spec, body, trial)
        assert ep.absent == AGENTS - st.present(spec, ep.params)


# ---------------------------------------------------------------------------
# T58: 없어질 수 있는 물체 = 검사 = 런타임
# ---------------------------------------------------------------------------


def test_t58_optional_bodies_is_the_union_of_every_possible_absence() -> None:
    spec = _spec()
    env = spec.file
    assert st.optional_bodies(env) == OPTIONAL
    seen: set[str] = set()
    for w2, arm, count in itertools.product((0.0, 1.0), (0.0, 1.0), (0.0, 1.0, 2.0)):
        absent = AGENTS - st.present(spec, _params(env, w2_on=w2, arm_on=arm, extra_count=count))
        assert absent <= OPTIONAL
        seen |= absent
    assert seen == OPTIONAL
    # 단조: 더 끄면 더 많이 없어진다
    for a, b in itertools.combinations(sorted(OPTIONAL), 2):
        assert st.absent_from(env, {a}) <= st.absent_from(env, {a, b})


def test_t58_constant_exists_above_the_threshold_is_not_optional() -> None:
    data = _data()
    for b in data["bodies"]:
        if b["id"] == "arm":
            b["exists"] = 0.75
    env = _env(data)
    assert st.optional_bodies(env) == OPTIONAL - {"arm", "hand", "finger"}


def test_t58_runtime_absent_stays_inside_optional_bodies() -> None:
    spec = _spec()
    trial = Trial(seeds.derive(9, 2))
    seen: set[str] = set()
    for body in _random_bodies(spec, 200, 2058):
        absent = Episode(spec, body, trial).absent
        assert absent <= OPTIONAL
        seen |= absent
    assert seen == OPTIONAL


def _exists_ref_messages(issues: list[Any], body: str) -> list[str]:
    head, tail = S.SPEC_EXISTS_REF.split("{where}")
    pattern = re.compile(re.escape(head.format(body=body)) + ".+" + re.escape(tail))
    return [i.message for i in issues if i.severity == "error" and pattern.fullmatch(i.message)]


def test_t58_validation_flags_exactly_the_optional_bodies() -> None:
    for body in sorted(AGENTS):
        data = _data()
        data["scoring"]["blocks"].append({"kind": "final_speed", "id": "spd", "body": body})
        found = _exists_ref_messages(check_env(_env(data)), body)
        assert bool(found) == (body in OPTIONAL), body


# ---------------------------------------------------------------------------
# T59: 형식 3
# ---------------------------------------------------------------------------


def _with_pod(data: dict[str, Any], exists: Any) -> dict[str, Any]:
    """첫 에이전트 물체에 관절로 붙은 작은 물체 하나를 더한다(exists가 None이면 있음 칸 없이)."""
    out = copy.deepcopy(data)
    root = next(b for b in out["bodies"] if b.get("owner", "agent") == "agent")
    pod: dict[str, Any] = {"id": "pod_x", "pos": [0.0, 0.0], "shapes": [{"kind": "circle", "r": 0.1}]}
    if exists is not None:
        pod["exists"] = exists
    out["bodies"].append(pod)
    out.setdefault("joints", []).append({"id": "pod_x_joint", "kind": "weld", "a": root["id"], "b": "pod_x"})
    return out


@pytest.mark.parametrize("name", ["lander", "jump"])
def test_t59_exists_alone_makes_a_version_3_file_and_removing_it_restores_the_text(name: str) -> None:
    src = json.loads((ENVS / f"{name}.evoenv").read_bytes())
    base_env = parse_env(src)
    since = feature_version(base_env)
    assert since == {"lander": 1, "jump": 2}[name]
    plain = _with_pod(src, None)  # 물체·관절을 더해도 있음 칸이 없으면 형식 그대로
    t0, h0 = dump_env(parse_env(plain)), env_hash(parse_env(plain))
    assert json.loads(t0)["version"] == since and '"exists"' not in t0
    for value in ("=0.75", 0.25, 1.0):  # 상수 1.0(늘 있음)이어도 칸을 쓰면 형식 3
        data = json.loads(t0)
        data["bodies"][-1]["exists"] = value
        env = parse_env(data)
        assert feature_version(env) == 3
        text = dump_env(env)
        saved = json.loads(text)
        assert saved["version"] == 3
        # 형식 3 글에는 모든 물체에 있음 칸이 있다(기본값이면 null). 형식 2 칸도 모두 있다(형식 2 > 1).
        assert [b["exists"] for b in saved["bodies"]] == [None] * (len(saved["bodies"]) - 1) + [value]
        assert all("motion" in b for b in saved["bodies"])
        assert parse_env(text) == env and dump_env(parse_env(text)) == text
        assert env_hash(env) != h0
        compile_env(env)  # 검사도 통과(뿌리가 아닌 에이전트 물체, p.*·상수만)
        data["bodies"][-1]["exists"] = None  # 비우면 처음 글·해시
        assert dump_env(parse_env(data)) == t0 and env_hash(parse_env(data)) == h0
        del data["bodies"][-1]["exists"]
        assert dump_env(parse_env(data)) == t0


def test_t59_a_v20_program_refuses_a_version_3_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """v2.0 상수(형식 2까지)로 읽으면 「… (파일 버전 3) …」(v2.0 코드 사본으로도 확인: 0'단계 보고)."""
    text = dump_env(_env())
    assert json.loads(text)["version"] == 3
    monkeypatch.setattr(m, "CURRENT_VERSION", 2)
    with pytest.raises(EnvFileError) as exc:
        parse_env(text)
    assert [i.message for i in exc.value.issues] == [S.SPEC_NEWER_VERSION.format(version=3)]
    assert "(파일 버전 3)" in exc.value.issues[0].message


def test_t59_version_2_files_read_and_save_as_before() -> None:
    """형식 2 파일은 형식 3 프로그램에서 2 → 3으로 읽히지만(사전 그대로) 저장은 다시 2."""
    text = (ENVS / "glider.evoenv").read_text(encoding="utf-8")
    env = parse_env(text)
    assert env.version == m.CURRENT_VERSION == 3
    assert json.loads(dump_env(env))["version"] == 2
    assert all(b.exists is None for b in env.bodies)


def test_t59_exists_validation_error_path() -> None:
    data = _data()
    for b in data["bodies"]:
        if b["id"] == "wheel2":
            b["exists"] = "nope"
    with pytest.raises(EnvFileError) as exc:
        parse_env(data)
    assert [(i.where, i.message) for i in exc.value.issues] == [("bodies[4].exists", S.SPEC_ERR_NUM_OR_FORMULA)]


# ---------------------------------------------------------------------------
# T60: 검사
# ---------------------------------------------------------------------------


def test_t60_structure_stage_has_only_body_values() -> None:
    env = _env()
    assert sy.build_symbols(env, "structure") == {f"p.{p.id}": sy.NUM_BUILD for p in env.params}
    assert "r.rr" in sy.build_symbols(env, "agent")


@pytest.mark.parametrize("formula", ["=p.w2_on + r.rr", "=gen.hills.x_max", "=time"])
def test_t60_exists_may_not_use_random_or_generator_values(formula: str) -> None:
    data = _data()
    data["generators"] = [{"kind": "hills", "id": "hills", "x_range": [-20.0, 60.0]}]
    data["bodies"][4]["exists"] = formula
    assert data["bodies"][4]["id"] == "wheel2"
    with pytest.raises(EnvFileError) as exc:
        compile_env(data)
    bad = formula[1:].split(" + ")[-1]
    errors = [i for i in exc.value.issues if i.severity == "error"]
    assert [i.where for i in errors] == ["bodies[4].exists"]
    assert S.EXPR_ERR_UNKNOWN_NAME.format(name=bad) in errors[0].message
    # 같은 이름을 다른 에이전트 칸(바퀴 반지름)에 쓰는 것은 된다(r.*, gen.*)
    ok = copy.deepcopy(data)
    ok["bodies"][4]["exists"] = "=p.w2_on"
    if bad != "time":
        ok["bodies"][4]["shapes"][0]["r"] = f"=0.3 + 0 * {bad}"
    compile_env(ok)


def test_t60_contact_and_sensors_on_optional_bodies_are_allowed() -> None:
    """contact()·접촉 센서·몸 상태 센서·레이 센서, 없어질 수 있는 관절의 모터·관절 각도, 없어질 수 있는 물체의 추진기는 된다."""
    spec = _spec()  # BASE에 이미 모두 있다
    assert not [i for i in spec.issues if i.severity == "error"]
    assert _exists_ref_messages(spec.issues, "wheel2") == [] and _exists_ref_messages(spec.issues, "hand") == []
    assert spec.inputs == ["cs.vx", "cs.angle", "w2_down", "w2s.vx", "eye.r0", "eye.r1", "eye.r2", "elbow"]
    assert spec.outputs == ["m1", "m2", "m_arm", "push"]


def test_t60_exists_on_a_world_body_is_an_error() -> None:
    data = _data()
    data["bodies"][1]["exists"] = 0.75  # post(세계 물체)
    issues = check_env(_env(data))
    assert ("error", "bodies[1].exists", S.SPEC_EXISTS_WORLD.format(id="post")) in [
        (i.severity, i.where, i.message) for i in issues]


def test_t60_exists_on_the_root_body_is_an_error() -> None:
    data = _data()
    data["bodies"][2]["exists"] = "=p.arm_on"  # chassis(뿌리)
    issues = check_env(_env(data))
    assert ("error", "bodies[2].exists", S.SPEC_EXISTS_ROOT.format(id="chassis")) in [
        (i.severity, i.where, i.message) for i in issues]


def _add_event(when: str) -> dict[str, Any]:
    data = _data()
    data["events"].append({"id": "probe", "when": when})
    return data


def _add_block(block: dict[str, Any]) -> dict[str, Any]:
    data = _data()
    data["scoring"]["blocks"].append({"id": "probe", **block})
    return data


def _add_checkpoints() -> dict[str, Any]:
    data = _data()
    data["checkpoints"] = [{"id": "cp", "lines": [[[5.0, 3.0], [5.0, 0.0]]], "loop": False, "body": "wheel2"}]
    return data


def _add_target() -> dict[str, Any]:
    data = _data()
    data["sensors"].append({"kind": "target", "id": "aim", "body": "chassis", "target": "wheel2"})
    return data


REF_CASES: dict[str, Any] = {
    "event body state": lambda: _add_event("wheel2.x > 3"),
    "event dist": lambda: _add_event("dist(wheel2, goal) < 1"),
    "event dist target": lambda: _add_event("dist(chassis, wheel2) < 1"),
    "event inside": lambda: _add_event("inside(wheel2, goal)"),
    "expr block body state": lambda: _add_block({"kind": "expr", "expr": "wheel2.vx"}),
    "progress block": lambda: _add_block({"kind": "progress", "body": "wheel2"}),
    "distance block": lambda: _add_block({"kind": "distance_to", "body": "wheel2", "target": "goal"}),
    "distance block target": lambda: _add_block({"kind": "distance_to", "body": "chassis", "target": "wheel2"}),
    "speed block": lambda: _add_block({"kind": "final_speed", "body": "wheel2"}),
    "tilt block": lambda: _add_block({"kind": "tilt", "body": "wheel2"}),
    "checkpoints": _add_checkpoints,
    "target sensor": _add_target,
}


@pytest.mark.parametrize("case", list(REF_CASES))
def test_t60_references_compile_when_the_body_cannot_disappear(case: str) -> None:
    """같은 참조를 늘 있는 물체(wheel2의 있음 칸을 비움)에 쓰면 오류가 아니다(아래 오류 경우의 대조군)."""
    data = REF_CASES[case]()
    data["bodies"][4]["exists"] = None
    assert data["bodies"][4]["id"] == "wheel2"
    compile_env(data)


@pytest.mark.parametrize("case", list(REF_CASES))
def test_t60_references_to_an_optional_body_are_errors(case: str) -> None:
    issues = check_env(_env(REF_CASES[case]()))
    assert _exists_ref_messages(issues, "wheel2"), [str(i) for i in issues]

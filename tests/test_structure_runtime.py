"""v2.1 A' 묶음: 몸 구조 런타임·검사·예제 (v2 설계 4.11·4.14·2.3·D34~D39, 테스트 T61·T62·T64·T65).

- T61 런타임: 없는 물체·관절·모터·추진기·점프·차량·공기역학은 만들지 않고, 그 구동기 채널은 실제 출력 0(에너지·행동 분포·관찰
  기록도 0), 그 물체·관절에 단 센서는 0(레이도 0), contact()는 거짓, 개체마다 moving_bodies 길이가 다르고 뿌리가 첫 에이전트
  물체. 없는 부품이 있는 몸은 그 부품을 파일에서 지운 환경과 물리가 비트 단위로 같다. 레이·정해진 움직임의 C 함수 길과 공개
  API 길이 같다.
- 검사(2.3 v2.1 줄): T60(tests/test_structure.py)에 더해 뿌리 오류가 다른 오류를 쏟지 않음, 체크포인트 기본 물체, 접촉 시간
  블록·목표 센서 물체·관절 각도는 통과.
- T62: 있음 칸을 쓰지 않는 파일은 새 길을 지나지 않고, v2.0 예제 결과가 v2.0 코드(커밋 4d3e3e6)로 만든 값과 같다(같은
  플랫폼에서만 숫자 비교, tests/data/v2_structure/v20_regress.npz). v1 예제의 같은 비교는 tests/test_runtime_v2.py의 T29.
- 내장 예제 「바퀴 고르는 언덕차」(4.14): 13 입력·4 출력·307차원, 처음 몸 바퀴 1개, 모터 힘 나누기, 내장 예제 순서·첫 실행
  복사·자체 점검.
- T64 결정성: 워커 0·2, 이어하기, 재현 묶음. T65(느림): 9.3 v2.1 완료 기준 절차.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab import cli, paths, selftest
from evolab.app import strings as S
from evolab.engine import seeds
from evolab.engine import trainer as trainer_mod
from evolab.engine.evaluate import OBSERVE_ACTIONS, evaluate
from evolab.engine.pool import WorkerPool, default_workers
from evolab.engine.runsetup import create_run, platform_info
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.policy.genome import layout_for
from evolab.runs.bundle import compare_stats, export_bundle, reproduce
from evolab.runs.store import RunStore
from evolab.spec import structure as st
from evolab.spec import symbols as sy
from evolab.spec.compiled import check_env
from evolab.spec.io import compile_env, dump_env, env_hash, feature_version, load_env, parse_env
from evolab.world.physics2d import episode as episode_mod
from evolab.world.physics2d.episode import Episode, Trial

ENVS = paths.builtin_env_dir()
WHEELS = ENVS / "wheels.evoenv"
WHEELS_TEXT = WHEELS.read_text(encoding="utf-8")
DATA = Path(__file__).parent / "data" / "v2_structure"
FIXED1 = DATA / "wheels_fixed1.evoenv"
FIXED2 = DATA / "wheels_fixed2.evoenv"
REGRESS = DATA / "v20_regress.npz"
TIME_COLUMNS = set(trainer_mod.TIME_COLUMNS)
TRIAL = Trial(seeds.derive(9, 61))


def _box(w: float, h: float) -> dict[str, Any]:
    return {"kind": "box", "w": w, "h": h}


# 모든 종류의 부품을 없어질 수 있는 물체에 단 환경: 바퀴(관절·모터·관절 각도), 짐칸 pod(추진기+연료, 점프, 공기역학, 몸
# 상태·목표·접촉·레이 센서, 바람 항력), 수레 cart(차량). 뿌리 chassis에는 늘 있는 몸 상태·레이 센서.
PARTS: dict[str, Any] = {
    "version": 3,
    "name": "parts",
    "world": {"max_time": 3.0, "bounds": [-50.0, -10.0, 50.0, 50.0], "wind": [2.0, 0.0]},
    "params": [
        {"id": "w_on", "value": 0.75, "evolve": True, "min": 0.0, "max": 1.0},
        {"id": "pod_on", "value": 0.75, "evolve": True, "min": 0.0, "max": 1.0},
        {"id": "cart_on", "value": 0.75, "evolve": True, "min": 0.0, "max": 1.0},
    ],
    "bodies": [
        {"id": "ground", "owner": "world", "type": "static", "tags": ["ground"],
         "shapes": [{"kind": "box", "w": 80.0, "h": 1.0, "offset": [0.0, -0.5]}]},
        {"id": "chassis", "pos": [0.0, 1.0], "shapes": [_box(2.0, 0.4)]},
        {"id": "wheel", "pos": [0.8, 0.5], "exists": "=p.w_on", "shapes": [{"kind": "circle", "r": 0.4}],
         "tags": ["wheel"]},
        {"id": "pod", "pos": [-0.8, 1.5], "exists": "=p.pod_on", "drag": 1.0, "shapes": [_box(0.4, 0.4)]},
        {"id": "cart", "pos": [0.0, 1.8], "exists": "=p.cart_on", "shapes": [_box(0.3, 0.2)]},
    ],
    "joints": [
        {"id": "jw", "kind": "pivot", "a": "chassis", "b": "wheel", "anchor_a": [0.8, -0.5]},
        {"id": "jp", "kind": "weld", "a": "chassis", "b": "pod", "anchor_a": [-0.8, 0.2]},
        {"id": "jc", "kind": "spring", "a": "chassis", "b": "cart", "stiffness": 50.0, "damping": 2.0},
    ],
    "actuators": [
        {"kind": "motor", "id": "mw", "joint": "jw", "max_torque": 40.0, "max_speed": 8.0},
        {"kind": "fuel_tank", "id": "tank", "capacity": 5.0},
        {"kind": "thruster", "id": "push", "body": "pod", "max_force": 30.0, "tank": "tank", "fuel_per_s": 1.0},
        {"kind": "jump", "id": "hop", "body": "pod", "speed": 2.0, "ground": []},
        {"kind": "vehicle", "id": "drive", "body": "cart", "max_accel": 3.0, "max_steer": 0.5},
        {"kind": "aero", "id": "air", "body": "pod"},
    ],
    "sensors": [
        {"kind": "body_state", "id": "cs", "body": "chassis", "fields": ["vx", "angle"]},
        {"kind": "rays", "id": "cr", "body": "chassis", "count": 3, "spread": 1.0, "center": -1.2, "length": 4.0,
         "hits": ["tag.ground"]},
        {"kind": "contact", "id": "wc", "body": "wheel", "with": ["tag.ground"]},
        {"kind": "joint_angle", "id": "ja", "joint": "jw"},
        {"kind": "body_state", "id": "ps", "body": "pod", "fields": ["vx", "vy", "omega"]},
        {"kind": "target", "id": "pt", "body": "pod", "target": "goal"},
        {"kind": "contact", "id": "pc", "body": "pod", "with": ["tag.ground"]},
        {"kind": "rays", "id": "pr", "body": "pod", "count": 2, "spread": 0.5, "center": -1.57, "length": 6.0,
         "hits": ["tag.ground"]},
        {"kind": "fuel", "id": "fuel", "tank": "tank"},
    ],
    "regions": [{"id": "goal", "kind": "goal", "pos": [20.0, 1.0], "shape": _box(2.0, 2.0)}],
    "events": [{"id": "touch", "when": "contact(wheel, tag.ground)"}],
    "termination": [],
    "scoring": {"blocks": [
        {"kind": "progress", "id": "go", "body": "chassis"},
        {"kind": "energy", "id": "en"},
        {"kind": "smoothness", "id": "sm", "actuators": ["mw", "push"]},
        {"kind": "contact_time", "id": "ct", "body": "wheel", "with": ["tag.ground"]},
        {"kind": "event", "id": "tc", "event": "touch", "value": "count"},
    ]},
    "brain": {"hidden": [4]},
}
POD_PARTS_CHANNELS = ("push", "hop", "drive.throttle", "drive.steer")


def parts_spec(data: dict[str, Any] | None = None) -> Any:
    return compile_env(json.dumps(data if data is not None else PARTS, ensure_ascii=False))


def body_of(spec: Any, **on: float) -> np.ndarray:
    """매개변수 값 = 처음 몸, on으로 고친 것(몸 유전자 순서)."""
    return np.array([on.get(p.id, p.value) for p in sy.evolved_params(spec.file)], dtype=np.float64)


def run_steps(ep: Episode, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """출력 범위 안의 무작위 행동으로 n스텝. 스텝마다 관찰을 돌려준다."""
    out = []
    lo = np.array([ep.spec.actuator_channels[c].low for c in ep.outputs])
    hi = np.array([ep.spec.actuator_channels[c].high for c in ep.outputs])
    for _ in range(n):
        if ep.done:
            break
        ep.step(rng.uniform(lo, hi))
        out.append(ep.observe())
    return out


# ---------------------------------------------------------------------------
# T61: 런타임
# ---------------------------------------------------------------------------


def test_t61_parts_env_compiles_and_everything_optional_is_allowed() -> None:
    spec = parts_spec()
    assert [i for i in spec.issues if i.severity == "error"] == []
    assert st.optional_bodies(spec.file) == {"wheel", "pod", "cart"}
    assert spec.outputs == ["mw", "push", "hop", "drive.throttle", "drive.steer"]


def test_t61_all_present_builds_every_part() -> None:
    spec = parts_spec()
    ep = Episode(spec, None, TRIAL)
    assert ep.absent == frozenset() and ep._dead_channels == []
    assert set(ep.bodies) == {"ground", "chassis", "wheel", "pod", "cart"}
    assert ep.moving_bodies == ["chassis", "wheel", "pod", "cart"] and ep.anchor_body == "chassis"
    assert len(ep._thrusters) == len(ep._jumps) == len(ep._vehicles) == len(ep._motors) == 1
    assert ep.aero_edge_count == 4 and [d[0] for d in ep._drag] == ["pod"]


def test_t61_absent_parts_are_not_built() -> None:
    spec = parts_spec()
    ep = Episode(spec, body_of(spec, w_on=0.2, pod_on=0.5, cart_on=0.0), TRIAL)
    assert ep.absent == {"wheel", "pod", "cart"} and isinstance(ep.absent, frozenset)
    with pytest.raises(AttributeError):
        ep.absent = frozenset()  # type: ignore[misc]
    assert set(ep.bodies) == {"ground", "chassis"} and set(ep.space.bodies) == set(ep.bodies.values())
    assert len(ep.space.constraints) == 0  # 관절 셋(각각 끝 하나가 없음)과 모터가 없다
    assert ep.moving_bodies == ["chassis"] and ep.anchor_body == "chassis"
    assert [g.id for g in ep.geometry().bodies] == ["ground", "chassis"] and ep.geometry().thrusters == []
    assert ep._thrusters == [] and ep._jumps == [] and ep._vehicles == [] and ep._motors == [] and ep._aero == []
    assert ep.aero_edge_count == 0 and ep._drag == [] and ep._agent_ids == ["chassis"]
    assert ep.absent_joints == {"jw", "jp", "jc"} and isinstance(ep.absent_joints, frozenset)
    index = {c: i for i, c in enumerate(ep.actuator_channels())}
    assert ep._dead_channels == sorted(index[c] for c in ("mw", *POD_PARTS_CHANNELS))
    assert not ep.jump_ready("hop")
    assert "tank" in ep._tanks  # 연료 탱크는 남는다(D35)


def test_t61_dead_channels_output_zero_and_cost_no_energy() -> None:
    spec = parts_spec()
    layout = layout_for(spec)
    rng = np.random.default_rng(61)
    ep = Episode(spec, body_of(spec, w_on=0.2, pod_on=0.2, cart_on=0.2), TRIAL)
    for _ in range(120):
        ep.step(np.ones(len(spec.outputs)))
        assert ep.applied_outputs == [0.0] * len(spec.outputs)
    res = ep.result()
    assert res.action_mean_abs == {c: 0.0 for c in ep.actuator_channels()}
    assert res.blocks["en"] == 0.0 and res.blocks["sm"] == 0.0  # 명령도 0이라 부드러움 블록도 세지 않는다
    # 같은 개체를 평가기로: 관찰 기록(v1.2)의 실제 출력 합도 0
    genome = layout.initial().copy()
    genome[: layout.n_body] = 2.0 * body_of(spec, w_on=0.2, pod_on=0.2, cart_on=0.2) - 1.0  # 범위 0..1 → 유전자 -1..1
    assert layout.body_values(genome).tolist() == pytest.approx([0.2, 0.2, 0.2])
    genome[layout.n_body:] = rng.normal(0.0, 1.0, layout.dim - layout.n_body)
    r = evaluate(spec, layout, genome[None, :], [(0, TRIAL)], observe=[OBSERVE_ACTIONS])
    assert r.observations is not None and r.observations[0] is not None
    assert r.observations[0].out_sum.tolist() == [0.0] * len(spec.outputs)
    assert r.action_abs.tolist() == [[0.0] * len(spec.outputs)]


def test_t61_live_channels_still_work_next_to_dead_ones() -> None:
    """바퀴만 있고 짐칸·수레가 없으면 모터 채널은 그대로 쓰이고 짐칸·수레 채널만 0이다."""
    spec = parts_spec()
    ep = Episode(spec, body_of(spec, w_on=0.9, pod_on=0.1, cart_on=0.1), TRIAL)
    assert ep.absent == {"pod", "cart"}
    for _ in range(30):
        ep.step(np.full(len(spec.outputs), 0.7))
    applied = dict(zip(ep.actuator_channels(), ep.applied_outputs, strict=True))
    assert applied["mw"] == 0.7 and all(applied[c] == 0.0 for c in POD_PARTS_CHANNELS)
    assert ep.result().blocks["sm"] == 0.0 and ep.result().blocks["en"] == pytest.approx(0.7 * 30 * ep.dt)


def test_t61_sensors_on_absent_bodies_and_joints_read_zero_rays_included() -> None:
    spec = parts_spec()
    rng = np.random.default_rng(62)
    ep = Episode(spec, body_of(spec, w_on=0.0, pod_on=0.0), TRIAL)
    names = ep.inputs
    zero = [k for k, n in enumerate(names) if n.split(".")[0] in ("wc", "ja", "ps", "pt", "pc", "pr")]
    assert len(zero) == 1 + 1 + 3 + 2 + 1 + 2
    live_rays = [k for k, n in enumerate(names) if n.startswith("cr.")]
    obs = [ep.observe(), *run_steps(ep, 200, rng)]
    for o in obs:
        assert o[zero].tolist() == [0.0] * len(zero)
    assert max(float(o[live_rays].min()) for o in obs) > 0.0  # 늘 있는 차체의 레이는 진짜 값
    values = ep.channel_values()
    assert all(values[names[k]] == 0.0 for k in zero)
    # 레이 그림: 없는 물체의 레이는 그리지 않지만 레이 센서 순서의 자리는 남는다(에디터가 순서로 읽음)
    assert len(ep.ray_view()) == 3 and len(ep._ray_views) == 2 and ep._ray_views[1][4]() == (0.0, 0.0)


def test_t61_contact_with_an_absent_body_is_false() -> None:
    spec = parts_spec()
    here = Episode(spec, body_of(spec, w_on=0.9), TRIAL)
    gone = Episode(spec, body_of(spec, w_on=0.1), TRIAL)
    for ep in (here, gone):
        for _ in range(150):
            ep.step(np.zeros(len(spec.outputs)))
    # 바퀴가 있으면 땅에 닿아 사건·접촉 시간·접촉 센서가 참, 없으면 늘 거짓
    assert here.result().blocks["tc"] >= 1 and here.result().blocks["ct"] > 0
    assert gone.result().blocks["tc"] == 0 and gone.result().blocks["ct"] == 0.0
    assert gone._values["contact:wheel:tag.ground"] is False
    assert here.observe()[here.inputs.index("wc")] == 1.0 and gone.observe()[gone.inputs.index("wc")] == 0.0


def test_t61_moving_bodies_differ_per_individual_and_the_root_is_first() -> None:
    spec = compile_env(WHEELS_TEXT)
    layout = layout_for(spec)
    genomes = layout.initial()[None, :] + 0.5 * np.random.default_rng(64).standard_normal((40, layout.dim))
    counts = set()
    for g in genomes:
        ep = Episode(spec, layout.body_values(g), TRIAL)
        counts.add(len(ep.moving_bodies))
        assert ep.moving_bodies[0] == "chassis" == ep.anchor_body == st.root_body(spec.file)
        assert ep.moving_bodies == [b.id for b in spec.file.bodies if b.id not in ep.absent]
        snap = ep.snapshot()
        assert snap.poses.shape == (len(ep.moving_bodies), 3)
    assert len(counts) >= 3  # 차체 + 바퀴 0~4개 중 여러 가지


def strip_absent(data: dict[str, Any], absent: set[str]) -> dict[str, Any]:
    """없는 물체와 거기 딸린 관절·구동기·센서·접촉 시간 블록을 파일에서 지우고, 남은 물체의 있음 칸도 지운다."""
    out = copy.deepcopy(data)
    out["bodies"] = [b for b in out["bodies"] if b["id"] not in absent]
    for b in out["bodies"]:
        b.pop("exists", None)
    gone_joints = {j["id"] for j in out.get("joints", []) if j["a"] in absent or j["b"] in absent}
    out["joints"] = [j for j in out.get("joints", []) if j["id"] not in gone_joints]
    out["actuators"] = [a for a in out["actuators"] if a.get("body") not in absent and a.get("joint") not in gone_joints]
    out["sensors"] = [s for s in out["sensors"] if s.get("body") not in absent and s.get("joint") not in gone_joints]
    out["scoring"]["blocks"] = [b for b in out["scoring"]["blocks"] if b.get("body") not in absent]
    return out


@pytest.mark.parametrize("on", [(0.75, 0.25, 0.25, 0.25), (0.25, 0.75, 0.75, 0.25), (0.75, 0.25, 0.25, 0.75),
                                (0.75, 0.75, 0.75, 0.75)])
def test_t61_absent_parts_behave_exactly_like_parts_deleted_from_the_file(on: tuple[float, ...]) -> None:
    """바퀴 고르는 언덕차의 한 몸 = 그 몸에서 없는 바퀴(물체·관절·모터·접지 센서)를 지운 파일(있음 칸 없음, v1 길):
    같은 행동(죽은 채널에는 아무 값)에서 매 스텝 자세·관찰(죽은 센서 0을 뺀 것)·끝 점수·블록이 비트 단위로 같다."""
    data = json.loads(WHEELS_TEXT)
    spec = compile_env(WHEELS_TEXT)
    values = {f"w{k + 1}_on": v for k, v in enumerate(on)}
    body = body_of(spec, **values)
    ep = Episode(spec, body, TRIAL)
    absent = {f"wheel{k + 1}" for k, v in enumerate(on) if v < 0.5}
    assert ep.absent == absent
    plain_data = strip_absent(data, absent)
    plain = compile_env(json.dumps(plain_data, ensure_ascii=False))
    assert feature_version(plain.file) == 1  # 있음 칸이 없는 v1 파일
    ref = Episode(plain, body, TRIAL)  # 몸 수치(매개변수)는 그대로 남아 있다
    assert ref.absent == frozenset() and ref.moving_bodies == ep.moving_bodies
    live_out = [k for k, c in enumerate(spec.outputs) if c in plain.outputs]
    live_in = [k for k, c in enumerate(spec.inputs) if c in plain.inputs]
    rng = np.random.default_rng(sum(int(4 * v) << k for k, v in enumerate(on)))
    assert ep.observe()[live_in].tobytes() == ref.observe().tobytes()
    while not ep.done:
        action = rng.uniform(-1.0, 1.0, len(spec.outputs))
        ep.step(action)
        ref.step(action[live_out])
        assert ep.snapshot().poses.tobytes() == ref.snapshot().poses.tobytes(), ep.steps
        assert ep.observe()[live_in].tobytes() == ref.observe().tobytes(), ep.steps
        assert ep.done == ref.done
    a, b = ep.result(), ref.result()
    assert (a.score, a.end_event, a.steps) == (b.score, b.end_event, b.steps)
    assert a.blocks == b.blocks and ep.steps > 10


def test_t61_motor_torque_is_shared_by_the_wheels_that_exist() -> None:
    spec = compile_env(WHEELS_TEXT)
    for on, want in (((0.75, 0.25, 0.25, 0.25), [600.0]), ((0.75, 0.25, 0.25, 0.75), [300.0, 300.0]),
                     ((0.75, 0.75, 0.75, 0.75), [150.0] * 4), ((0.25, 0.25, 0.25, 0.25), [])):
        ep = Episode(spec, body_of(spec, **{f"w{k + 1}_on": v for k, v in enumerate(on)}), TRIAL)
        assert [mo.constraint.max_force for mo in ep._motors] == want, on


@pytest.mark.parametrize("which", ["wheels", "parts"])
def test_t61_fast_c_paths_match_the_public_api_with_absent_parts(which: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """레이 첫 맞음 질의·정해진 움직임의 C 함수 길과 공개 API 길이 없는 부품이 있는 몸에서도 비트 단위로 같다."""
    if which == "wheels":
        spec = compile_env(WHEELS_TEXT)
    else:
        data = copy.deepcopy(PARTS)  # 정해진 움직임 발판도 하나(_drive_kinematic의 C 길)
        data["bodies"].append({"id": "lift", "owner": "world", "type": "kinematic", "pos": [3.0, 0.3],
                               "shapes": [_box(1.0, 0.2)], "tags": ["ground"],
                               "motion": {"kind": "path", "points": [[0, 0], [0, 1]], "speed": 0.5,
                                          "mode": "pingpong", "spin": 0.3}})
        spec = parts_spec(data)
    layout = layout_for(spec)
    genomes = layout.initial()[None, :] + 0.5 * np.random.default_rng(65).standard_normal((6, layout.dim))
    pairs = [(g, t) for g in range(6) for t in seeds.validation_trials(2)]
    assert len({frozenset(Episode(spec, layout.body_values(g), TRIAL).absent) for g in genomes}) >= 2

    def run() -> Any:
        return evaluate(spec, layout, genomes, pairs, record=[True] * len(pairs), every=1)

    assert episode_mod._CP_FAST
    fast = run()
    monkeypatch.setattr(episode_mod, "_CP_FAST", False)
    public = run()
    assert fast.scores.tobytes() == public.scores.tobytes() and fast.steps.tolist() == public.steps.tolist()
    assert fast.end_events == public.end_events and fast.blocks.tobytes() == public.blocks.tobytes()
    assert fast.trajectories is not None and public.trajectories is not None
    for x, y in zip(fast.trajectories, public.trajectories, strict=True):
        assert x is not None and y is not None
        assert x.poses.tobytes() == y.poses.tobytes() and x.outputs.tobytes() == y.outputs.tobytes()


def test_t61_absent_depends_only_on_the_body_values() -> None:
    """구조는 몸 수치만의 함수: 시험이 달라도 같고, 짧은 체인 끝까지 이어짐 규칙(물체를 끄면 관절로만 매달린 것도)."""
    data = copy.deepcopy(PARTS)
    data["bodies"].append({"id": "tip", "pos": [-0.8, 2.0], "shapes": [_box(0.1, 0.3)]})
    data["joints"].append({"id": "jt", "kind": "pivot", "a": "pod", "b": "tip", "anchor_a": [0.0, 0.2]})
    data["sensors"].append({"kind": "body_state", "id": "ts", "body": "tip", "fields": ["x"]})
    spec = parts_spec(data)
    for trial in seeds.validation_trials(3):
        ep = Episode(spec, body_of(spec, pod_on=0.3), trial)
        assert ep.absent == {"pod", "tip"} and "tip" not in ep.bodies
        assert ep.observe()[ep.inputs.index("ts.x")] == 0.0
        assert ep.absent == frozenset(sy.agent_body_ids(spec.file)) - st.present(spec, ep.params)


# ---------------------------------------------------------------------------
# 검사 (2.3 v2.1 줄, T60에 더해)
# ---------------------------------------------------------------------------


def _errors(data: dict[str, Any]) -> list[tuple[str, str]]:
    return [(i.where, i.message) for i in check_env(parse_env(json.dumps(data, ensure_ascii=False)))
            if i.severity == "error"]


def test_exists_ref_message_names_the_place() -> None:
    data = copy.deepcopy(PARTS)
    data["events"].append({"id": "far", "when": "pod.x > 3 or dist(chassis, pod) > 2"})
    data["scoring"]["blocks"].append({"kind": "final_speed", "id": "spd", "body": "cart"})
    data["scoring"]["formula"] = "go + wheel.vx"
    assert _errors(data) == [
        ("events[1].when", S.SPEC_EXISTS_REF.format(body="pod", where=S.SPEC_EXISTS_WHERE_EVENT.format(id="far"))),
        ("scoring.blocks[5].body", S.SPEC_EXISTS_REF.format(body="cart",
                                                            where=S.SPEC_EXISTS_WHERE_BLOCK.format(id="spd"))),
        ("scoring.formula", S.SPEC_EXISTS_REF.format(body="wheel", where=S.SPEC_EXISTS_WHERE_FORMULA)),
    ]


def test_exists_on_the_root_is_one_error_not_a_flood() -> None:
    """뿌리에 있음 칸을 쓰면 그 오류 하나만(모든 부품이 없어질 수 있다고 보고 참조 오류를 쏟지 않는다)."""
    data = copy.deepcopy(PARTS)
    data["bodies"][1]["exists"] = "=p.w_on"
    assert _errors(data) == [("bodies[1].exists", S.SPEC_EXISTS_ROOT.format(id="chassis"))]


def test_checkpoints_default_body_follows_the_vehicle() -> None:
    """체크포인트 물체를 비우면 차량 구동기의 물체를 쓴다(런타임 규칙): 그 물체가 없어질 수 있으면 오류, 뿌리면 통과."""
    data = copy.deepcopy(PARTS)
    data["checkpoints"] = [{"id": "cp", "lines": [[[5.0, 3.0], [5.0, 0.0]], [[9.0, 3.0], [9.0, 0.0]]], "loop": False}]
    assert _errors(data) == [("checkpoints[0].body", S.SPEC_EXISTS_REF.format(
        body="cart", where=S.SPEC_EXISTS_WHERE_CHECKPOINTS.format(id="cp")))]
    data["actuators"] = [a for a in data["actuators"] if a["kind"] != "vehicle"]
    assert _errors(data) == []  # 차량이 없으면 뿌리(늘 있음)
    data["checkpoints"][0]["body"] = "pod"
    assert [w for w, _ in _errors(data)] == ["checkpoints[0].body"]


def test_contact_time_block_and_sensors_on_optional_parts_are_allowed() -> None:
    """접촉 시간 블록(접촉), 없어질 수 있는 물체에 단 목표·레이·몸 상태 센서, 없어질 수 있는 관절의 각도 센서는 된다.
    목표 센서가 없어질 수 있는 물체를 가리키는(target) 것만 오류다."""
    assert _errors(PARTS) == []
    data = copy.deepcopy(PARTS)
    data["sensors"].append({"kind": "target", "id": "aim", "body": "chassis", "target": "pod"})
    assert _errors(data) == [("sensors[9].target", S.SPEC_EXISTS_REF.format(
        body="pod", where=S.SPEC_EXISTS_WHERE_TARGET.format(id="aim")))]


def test_constant_exists_above_the_threshold_is_never_absent() -> None:
    data = copy.deepcopy(PARTS)
    data["bodies"][2]["exists"] = 1.0
    data["events"].append({"id": "spin", "when": "wheel.omega > 1"})  # 늘 있는 바퀴라 몸 상태를 써도 된다
    spec = parts_spec(data)
    assert st.optional_bodies(spec.file) == {"pod", "cart"}
    assert Episode(spec, None, TRIAL).absent == frozenset()


# ---------------------------------------------------------------------------
# T62: 있음 칸을 쓰지 않는 파일은 v2.0과 같다
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["lander", "track_car", "hill_car", "jump", "glider"])
def test_t62_files_without_exists_take_the_old_path(name: str) -> None:
    spec = compile_env((ENVS / f"{name}.evoenv").read_text(encoding="utf-8"))
    layout = layout_for(spec)
    ep = Episode(spec, layout.body_values(layout.initial()) if layout.n_body else None, TRIAL)
    assert ep.absent == frozenset() and ep.absent_joints == frozenset() and ep._dead_channels == []
    assert set(ep.bodies) >= set(sy.agent_body_ids(spec.file))
    assert not [k for k in spec.formulas if k[1] == "structure"]


def test_t62_always_present_exists_gives_the_same_results_as_no_exists() -> None:
    """언덕차의 바퀴에 늘 있는 있음 칸(상수 1, 수식 > 0.5)을 달면 새 길(present 계산)을 지나지만 결과는 비트 단위로 같다
    (모든 플랫폼에서 도는 T62의 몫)."""
    text = (ENVS / "hill_car.evoenv").read_text(encoding="utf-8")
    data = json.loads(text)
    data["bodies"][1]["exists"] = 1.0
    data["bodies"][2]["exists"] = "=p.rear_r + 0.26"  # 범위 0.25..1.0이라 늘 0.5보다 크다
    with_exists = json.dumps(data, ensure_ascii=False)
    assert feature_version(parse_env(with_exists)) == 3 and feature_version(parse_env(text)) == 1
    spec_a, spec_b = compile_env(text), compile_env(with_exists)
    layout = layout_for(spec_a)
    assert layout_for(spec_b).signature() == layout.signature()
    genomes = layout.initial()[None, :] + 0.5 * np.random.default_rng(62).standard_normal((4, layout.dim))
    pairs = [(g, t) for g in range(4) for t in seeds.validation_trials(2)]
    a = evaluate(spec_a, layout, genomes, pairs, record=[True] * len(pairs), every=1)
    b = evaluate(spec_b, layout, genomes, pairs, record=[True] * len(pairs), every=1)
    assert a.scores.tobytes() == b.scores.tobytes() and a.steps.tolist() == b.steps.tolist()
    assert a.blocks.tobytes() == b.blocks.tobytes() and a.action_abs.tobytes() == b.action_abs.tobytes()
    assert a.trajectories is not None and b.trajectories is not None
    for x, y in zip(a.trajectories, b.trajectories, strict=True):
        assert x is not None and y is not None
        assert x.poses.tobytes() == y.poses.tobytes() and x.outputs.tobytes() == y.outputs.tobytes()


def _same_platform(data: Any) -> bool:
    """자료를 만든 플랫폼과 같은가(tests/test_runtime_v2.py의 T29와 같은 규칙: python, numpy, pymunk, cma, cpu)."""
    if "platform_info" not in data.files:
        return False
    made = json.loads(str(data["platform_info"]))
    now = platform_info(0)
    return all(made.get(k) == now[k] for k in ("python", "numpy", "pymunk", "cma", "cpu"))


def test_t62_v20_reference_inputs_are_the_current_examples() -> None:
    """v20_regress.npz(v2.0 코드로 만든 자료)의 환경 글이 지금 예제 파일과 같다(다른 플랫폼에서도 도는 부분)."""
    data = np.load(REGRESS, allow_pickle=False)
    for name in ("jump", "glider"):
        assert str(data[f"{name}.env_text"]) == (ENVS / f"{name}.evoenv").read_text(encoding="utf-8")


def test_t62_v20_examples_match_the_v20_reference() -> None:
    """T62: 점프 게임·글라이더의 무작위 유전자 6 x 시험 2의 점수·스텝·종료·블록·출력 평균·매 스텝 궤적이 v2.0 코드(커밋
    4d3e3e6)로 만든 값과 같다. 숫자 비교는 자료를 만든 플랫폼에서만 한다(명세 「정답 해시를 박지 않음」)."""
    import hashlib

    data = np.load(REGRESS, allow_pickle=False)
    if not _same_platform(data):
        pytest.skip(f"reference made on {data['platform']} {data['platform_info'] if 'platform_info' in data else ''}")
    trials = [Trial(int(s)) for s in data["trial_seeds"]]
    for name in ("jump", "glider"):
        spec = compile_env(str(data[f"{name}.env_text"]))
        layout = layout_for(spec)
        genomes = data[f"{name}.genomes"]
        pairs = [(g, t) for g in range(len(genomes)) for t in trials]
        r = evaluate(spec, layout, genomes, pairs, record=[True] * len(pairs), every=1)
        assert r.scores.tobytes() == data[f"{name}.scores"].tobytes(), name
        assert np.array_equal(r.steps, data[f"{name}.steps"]), name
        assert list(r.end_events) == list(data[f"{name}.end_events"]), name
        assert list(r.outcomes) == list(data[f"{name}.outcomes"]), name
        assert r.blocks.tobytes() == data[f"{name}.blocks"].tobytes(), name
        assert r.action_abs.tobytes() == data[f"{name}.action_abs"].tobytes(), name
        assert r.trajectories is not None
        assert [tr.frames for tr in r.trajectories] == data[f"{name}.frames"].tolist(), name
        for k, tr in enumerate(r.trajectories):
            assert hashlib.sha256(tr.poses.tobytes()).digest() == bytes(data[f"{name}.poses_sha256"][k]), (name, k)
            assert hashlib.sha256(tr.outputs.tobytes()).digest() == bytes(data[f"{name}.outputs_sha256"][k]), (name, k)
        final = np.concatenate([tr.poses[-1] for tr in r.trajectories])
        assert final.tobytes() == data[f"{name}.final_poses"].tobytes(), name


# ---------------------------------------------------------------------------
# 내장 예제 「바퀴 고르는 언덕차」 (4.14), 내장 예제 순서·첫 실행 복사·자체 점검 (5.6)
# ---------------------------------------------------------------------------


def test_wheels_example_matches_the_design() -> None:
    spec = load_env(WHEELS)
    layout = layout_for(spec)
    assert spec.issues == [] and spec.file.name == S.ED_BUILTIN_WHEELS
    assert json.loads(WHEELS.read_bytes())["version"] == 3 == feature_version(spec.file)
    assert dump_env(parse_env(WHEELS_TEXT)) != WHEELS_TEXT  # 손으로 다듬은 글(언덕차와 같은 모양), 내용은 아래로 고정
    assert len(spec.inputs) == 13 and spec.outputs == ["m1", "m2", "m3", "m4"]
    assert layout.n_body == 15 and layout.dim == 307 and spec.file.brain.hidden == [16]
    assert st.root_body(spec.file) == "chassis"
    assert st.optional_bodies(spec.file) == {"wheel1", "wheel2", "wheel3", "wheel4"}
    assert {p.id: p.value for p in spec.file.params if p.id.endswith("_on")} == {
        "w1_on": 0.75, "w2_on": 0.25, "w3_on": 0.25, "w4_on": 0.25}
    for b in spec.file.bodies[1:]:
        assert b.exists == f"=p.w{b.id[-1]}_on"
    ep = Episode(spec, None, TRIAL)
    assert ep.absent == {"wheel2", "wheel3", "wheel4"} and ep.moving_bodies == ["chassis", "wheel1"]
    assert st.structure_key(spec, st.param_values(spec.file)) == ("wheel1",)
    assert spec.file.evaluation.trials == 3 and spec.file.world.max_time == 30.0
    # 언덕·세계·점수·종료는 언덕차와 같다
    hill = json.loads((ENVS / "hill_car.evoenv").read_text(encoding="utf-8"))
    wheels = json.loads(WHEELS_TEXT)
    for key in ("world", "randomize", "generators", "events", "termination", "scoring", "brain", "evaluation"):
        assert wheels[key] == hill[key], key


def test_wheels_baselines_differ_from_the_example_only_in_the_wheel_switches() -> None:
    """완료 기준의 고정 구조 기준선(tests/data/v2_structure): 예제에서 있음 매개변수의 진화를 끄고 값만 정한 파일."""
    base = json.loads(WHEELS_TEXT)
    for path, on in ((FIXED1, {"w1"}), (FIXED2, {"w1", "w4"})):
        data = json.loads(path.read_text(encoding="utf-8"))
        for p in data["params"]:
            if p["id"].endswith("_on"):
                assert p["evolve"] is False and p["value"] == (0.75 if p["id"][:2] in on else 0.25)
                p.update(evolve=True, value=0.75 if p["id"] == "w1_on" else 0.25)
        assert data == base
        spec = load_env(path)
        assert spec.issues == [] and layout_for(spec).n_body == 11
        assert Episode(spec, None, TRIAL).absent == {f"wheel{k}" for k in range(1, 5) if f"w{k}" not in on}


def test_builtin_order_ends_with_wheels_and_first_run_copies_it(tmp_path: Path) -> None:
    """T35(v2.1): 내장 예제 순서 …, 글라이더, 바퀴 고르는 언덕차. v2.1을 처음 켜면 새 예제 하나만 복사한다.

    (6.2.0, 설계 4.5) 옛 여섯은 이름으로 못 박고, 그 뒤는 종목 프리셋(BUILTIN_ORDER[6:]: 저장소에 있는 것만 목록에 든다)."""
    old = ("lander", "track_car", "hill_car", "jump", "glider", "wheels")
    assert paths.BUILTIN_ORDER[:6] == old and paths.BUILTIN_ORDER[4:6] == ("glider", "wheels")
    names = [p.name for p in paths.builtin_envs()]
    assert names[:6] == [f"{n}.evoenv" for n in old]
    presets = names[6:]
    assert presets == [f"{n}.evoenv" for n in paths.BUILTIN_ORDER[6:] if f"{n}.evoenv" in presets]
    for name in ("lander", "track_car", "hill_car", "jump", "glider"):
        (tmp_path / f"{name}.evoenv").write_text(f"고친 {name}", encoding="utf-8")
    assert [p.name for p in paths.install_examples(tmp_path)] == ["wheels.evoenv", *presets]
    assert (tmp_path / "wheels.evoenv").read_bytes() == WHEELS.read_bytes()
    assert (tmp_path / "jump.evoenv").read_text(encoding="utf-8") == "고친 jump"


def test_selftest_runs_wheels_twice() -> None:
    """T36(v2.1): 자체 점검의 필수 예제에 바퀴 고르는 언덕차, 바퀴 둘(1·4)인 몸을 두 번 돌린 결과가 같다."""
    assert "wheels.evoenv" in selftest.REQUIRED_EXAMPLES
    spec = load_env(WHEELS)
    assert Episode(spec, selftest._two_wheel_body(spec), TRIAL).absent == {"wheel2", "wheel3"}
    first = selftest._check_v2_examples()
    assert "wheels.evoenv: steps=" in first and selftest._check_v2_examples() == first
    assert "wheels.evoenv" in selftest._check_bundle_files()


# ---------------------------------------------------------------------------
# T64: 결정성 (워커 수, 이어하기, 재현 묶음)
# ---------------------------------------------------------------------------


def short_wheels() -> str:
    """빠른 테스트용: 바퀴 고르는 언덕차를 6초로(구조·부품은 그대로)."""
    data = json.loads(WHEELS_TEXT)
    data["world"]["max_time"] = 6.0
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def small_config(**over: Any) -> TrainConfig:
    c = TrainConfig(popsize=8, trials=2, validate_every=40, validation_seeds=3, checkpoint_every=2, run_seed=11)
    for k, v in over.items():
        setattr(c, k, v)
    return c


def strip(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in r.items() if k not in TIME_COLUMNS} for r in rows]


def run_trainer(text: str, workers: int, gens: int, store: RunStore) -> tuple[Trainer, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    with WorkerPool(workers) as pool:
        t = Trainer(text, small_config(max_generations=gens), pool, store)
        t.run(rows.append)
    return t, rows


def test_t64_wheels_training_is_independent_of_worker_count_and_resume(tmp_path: Path) -> None:
    """T64: 워커 0·2개 학습 3세대의 통계(시간 열 제외)·챔피언이 같고, 3세대 + 이어서 3세대 = 6세대 한 번에. 집단에 구조가
    다른 개체가 섞여 있다(없는 바퀴가 개체마다 다름)."""
    text = short_wheels()
    spec = compile_env(text)

    def store(label: str) -> RunStore:
        s = RunStore.create(tmp_path, label, {"env_name": spec.file.name})
        s.add_env_version(text, 0)
        return s

    straight, _ = run_trainer(text, 0, 6, store("한 번에"))
    halves = store("끊고 이어서")
    first, first_rows = run_trainer(text, 2, 3, halves)
    rows = strip(straight.store.read_stats())  # type: ignore[union-attr]
    assert strip(first_rows) == rows[:3]
    with WorkerPool(0) as pool:
        resumed = Trainer.resume(halves, pool, None, max_generations=6)
        resumed.run()
    assert strip(halves.read_stats()) == rows
    assert resumed.learner.best().tobytes() == straight.learner.best().tobytes()
    assert resumed.state.champion.score == straight.state.champion.score
    assert halves.champion_generations("champion") == straight.store.champion_generations("champion")  # type: ignore[union-attr]
    # 마지막 세대 집단의 구조가 여럿이다(구조가 다른 개체끼리의 결정성을 본 것)
    layout = layout_for(spec)
    pop = straight.learner.ask()
    keys = {st.structure_key(spec, st.param_values(spec.file, layout.body_values(x))) for x in pop}
    assert len(keys) >= 2


def test_t64_wheels_bundle_reproduces_identical_stats(tmp_path: Path) -> None:
    """T64: 작은 실행의 재현 묶음을 다시 돌리면 compare_stats가 None."""
    text = short_wheels()
    config = small_config(max_generations=3)
    store = create_run(tmp_path / "원래", text, compile_env(text), config, 0)
    with WorkerPool(0) as pool:
        t = Trainer(text, config, pool, store)
        t.run()
        t.checkpoint()
    new = reproduce(export_bundle(store.root, tmp_path / "묶음.zip"), tmp_path / "다시", workers=2)
    assert compare_stats(store.read_stats(), RunStore.open(new).read_stats()) is None
    assert len(store.read_stats()) == 3
    first = store.read_run_info()["env_versions"][0]["id"]
    assert env_hash(parse_env(store.env_text(first))) == env_hash(parse_env(text))


# ---------------------------------------------------------------------------
# T65 (느림): 9.3 v2.1 완료 기준 1의 절차 (측정 단계에서 돌린다)
# ---------------------------------------------------------------------------


def train_and_pick_champion(env: Path, seed: int, budget: int, out: Path) -> tuple[Any, np.ndarray, int]:
    """명령줄 학습(CMA-ES 기본값, 자동 재시작 끔, 고정 실행 시드)을 학습 시험 budget회까지 돌리고, 그 예산까지의 마지막
    챔피언(stats.csv에서 evaluations ≤ budget인 마지막 줄의 champion_generation)을 고른다(설계 9.1의 판정 절차,
    tests/test_examples_v2.py의 T32·T33과 같은 방법)."""
    code = cli.main(["train", str(env), "--seed", str(seed), "--workers", str(default_workers()),
                     "--max-evaluations", str(budget), "--no-restart", "--out", str(out)])
    assert code == 0
    run = next(out.iterdir())
    rows = list(csv.DictReader((run / "stats.csv").open(encoding="utf-8-sig")))
    ok = [r for r in rows if int(r["evaluations"]) <= budget and r["champion_generation"] not in ("", None)
          and int(float(r["champion_generation"])) >= 1]
    gen = int(float(ok[-1]["champion_generation"]))
    genome = np.load(run / "champions" / f"gen_{gen:06d}_champion.npz")["genome"]
    spec = load_env(sorted((run / "envs").glob("*.evoenv"))[-1])
    return spec, genome, gen


def judged_progress(spec: Any, genome: np.ndarray, count: int = 100) -> np.ndarray:
    """판정 세트 count개에서 전진 거리(점수 블록 s_prog, m)."""
    layout = layout_for(spec)
    res = evaluate(spec, layout, genome[None, :], [(0, t) for t in seeds.judge_trials(count)])
    return res.blocks[:, [b.id for b in spec.file.scoring.blocks].index("s_prog")]


@pytest.mark.slow
def test_t65_structure_evolution_completion_criterion(tmp_path: Path) -> None:
    """T65 (9.3 v2.1 기준 1): 고정 실행 시드 1·2·3, 학습 시험 6,000회 챔피언. 같은 절차로 바퀴 1개 고정·2개 고정을 학습해
    구조 진화의 판정 전진 거리 중앙값 ≥ 1개 고정 중앙값 x 1.1 그리고 ≥ 2개 고정 중앙값 x 0.95, 세 시드 중 둘 이상에서
    마지막 챔피언의 구조(있는 바퀴 자리 모음)가 처음(뒤 바퀴 하나)과 다름. 원형: 255.1 / 221.2 / 257.9m, 세 시드 모두 구조 변화."""
    medians: dict[str, list[float]] = {"wheels": [], "fixed1": [], "fixed2": []}
    changed = 0
    initial_key = st.structure_key(load_env(WHEELS), st.param_values(load_env(WHEELS).file))
    for label, env in (("wheels", WHEELS), ("fixed1", FIXED1), ("fixed2", FIXED2)):
        for seed in (1, 2, 3):
            spec, genome, gen = train_and_pick_champion(env, seed, 6_000, tmp_path / f"{label}_s{seed}")
            medians[label].append(float(np.median(judged_progress(spec, genome))))
            key = st.structure_key(spec, st.param_values(spec.file, layout_for(spec).body_values(genome)))
            if label == "wheels":
                changed += key != initial_key
            print(json.dumps({"env": label, "seed": seed, "champion_generation": gen, "median_m": medians[label][-1],
                              "structure": list(key)}, ensure_ascii=False))
    evo, one, two = (float(np.median(medians[k])) for k in ("wheels", "fixed1", "fixed2"))
    print(json.dumps({"median_m": {"wheels": evo, "fixed1": one, "fixed2": two}, "changed_seeds": changed}))
    assert evo >= 1.1 * one and evo >= 0.95 * two and changed >= 2, (medians, changed)

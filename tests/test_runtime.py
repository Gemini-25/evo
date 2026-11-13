"""M1: 물리 런타임 부품(추진기, 연료, 접촉, 사건, 종료, 점수)과 결정성."""

import math
from typing import Any

import numpy as np
import pytest

from evolab import paths
from evolab.spec.compiled import EnvSpec
from evolab.spec.io import compile_env, load_env
from evolab.world.physics2d.episode import Episode, EpisodeBuildError, Trial
from lander_pilot import pilot

LANDER = paths.builtin_env_dir() / "lander.evoenv"


@pytest.fixture(scope="module")
def lander() -> EnvSpec:
    return load_env(LANDER)


def run(spec: EnvSpec, seed: int, policy: Any = None, body: Any = None, record: bool = False) -> tuple[Any, list[bytes]]:
    ep = Episode(spec, body, Trial(seed))
    frames: list[bytes] = []
    zero = [0.0] * len(spec.outputs)
    while not ep.done:
        ep.step(policy(ep.observe()) if policy is not None else zero)
        if record:
            frames.append(ep.snapshot().poses.tobytes())
    return ep.result(), frames


def box_env(**over: Any) -> dict[str, Any]:
    """중력 없는 공간의 상자 하나 + 오른쪽으로 미는 추진기. 부품 단위 시험용."""
    env: dict[str, Any] = {
        "name": "box",
        "world": {"gravity": [0, 0], "max_time": 2.0, "bounds": [-100, -100, 100, 100]},
        "bodies": [{"id": "box", "shapes": [{"kind": "box", "w": 1.0, "h": 1.0}], "material": {"density": 2.0}}],
        "actuators": [
            {"kind": "fuel_tank", "id": "tank", "capacity": 1.0},
            {"kind": "thruster", "id": "push", "body": "box", "dir": 0.0, "max_force": 4.0, "tank": "tank", "fuel_per_s": 1.0},
        ],
        "sensors": [{"kind": "body_state", "id": "st", "body": "box", "fields": ["vx", "vy"]}],
        "scoring": {"blocks": [
            {"kind": "survival_time", "id": "t"},
            {"kind": "fuel_used", "id": "fuel", "tank": "tank"},
        ]},
    }
    env.update(over)
    return env


# ---------------------------------------------------------------------------
# 결정성
# ---------------------------------------------------------------------------


def test_same_seed_same_result_and_trajectory(lander: EnvSpec) -> None:
    a, fa = run(lander, 11, pilot, record=True)
    b, fb = run(lander, 11, pilot, record=True)
    assert a == b
    assert fa == fb


def test_result_does_not_depend_on_previous_episodes(lander: EnvSpec) -> None:
    first, _ = run(lander, 5, pilot)
    for s in (1, 2, 3):
        run(lander, s, pilot)
    again, _ = run(lander, 5, pilot)
    assert again == first


def test_different_seeds_give_different_worlds(lander: EnvSpec) -> None:
    a = Episode(lander, None, Trial(1))
    b = Episode(lander, None, Trial(2))
    assert a.randoms != b.randoms
    assert a.gen_values != b.gen_values


def test_adding_a_random_value_keeps_the_others(lander: EnvSpec) -> None:
    data = lander.file.model_dump(mode="json", by_alias=True)
    data["randomize"].insert(0, {"id": "extra", "dist": "uniform", "low": 0, "high": 1})
    other = compile_env(data)
    a = Episode(lander, None, Trial(3)).randoms
    b = Episode(other, None, Trial(3)).randoms
    assert all(b[k] == v for k, v in a.items())


# ---------------------------------------------------------------------------
# 착륙선 전체
# ---------------------------------------------------------------------------


def test_lander_observation_shape_and_range(lander: EnvSpec) -> None:
    ep = Episode(lander, None, Trial(0))
    obs = ep.observe()
    assert obs.dtype == np.float32 and obs.shape == (9,)
    assert np.all(np.abs(obs) <= 5.0)


def test_lander_falls_without_thrust(lander: EnvSpec) -> None:
    res, _ = run(lander, 7)
    assert res.outcome == "failure"
    assert res.end_label in ("충돌", "패드 밖 착지")
    assert res.score < 0


def test_lander_is_landable_by_a_simple_pilot(lander: EnvSpec) -> None:
    results = [run(lander, s, pilot)[0] for s in range(30)]
    success = sum(r.outcome == "success" for r in results)
    assert success >= 18, [(r.end_label, round(r.score, 1)) for r in results]
    best = max(results, key=lambda r: r.score)
    assert best.blocks["s_success"] == 1.0 and best.blocks["s_dist"] == 0.0


def test_wrong_action_length_is_an_error(lander: EnvSpec) -> None:
    ep = Episode(lander, None, Trial(0))
    with pytest.raises(ValueError):
        ep.step([1.0])


def test_nan_action_is_treated_as_zero(lander: EnvSpec) -> None:
    a = Episode(lander, None, Trial(0))
    b = Episode(lander, None, Trial(0))
    a.step([math.nan, 0.0, 0.0])
    b.step([0.0, 0.0, 0.0])
    assert a.snapshot().poses.tobytes() == b.snapshot().poses.tobytes()


def test_body_params_change_the_body_and_are_clipped() -> None:
    data = load_env(LANDER).file.model_dump(mode="json", by_alias=True)
    data["params"][0]["evolve"] = True  # leg_len 0.6..2.0
    spec = compile_env(data)
    long = Episode(spec, [5.0], Trial(0))
    assert long.params["p.leg_len"] == 2.0
    default = Episode(spec, None, Trial(0))
    assert default.params["p.leg_len"] == 1.2
    with pytest.raises(EpisodeBuildError):
        Episode(spec, [1.0, 2.0], Trial(0))


# ---------------------------------------------------------------------------
# 부품
# ---------------------------------------------------------------------------


def test_thruster_pushes_in_body_direction_and_uses_fuel() -> None:
    spec = compile_env(box_env())
    ep = Episode(spec, None, Trial(0))
    for _ in range(30):  # 0.5초
        ep.step([1.0])
    v = ep.bodies["box"].velocity
    mass = ep.bodies["box"].mass  # 1 x 1 x 2 = 2 kg
    assert mass == pytest.approx(2.0)
    assert v.x == pytest.approx(4.0 / mass * 0.5, rel=1e-6)
    assert v.y == pytest.approx(0.0, abs=1e-12)
    assert ep.block_values()["fuel"] == pytest.approx(0.5)


def test_thruster_stops_when_tank_is_empty() -> None:
    spec = compile_env(box_env())
    ep = Episode(spec, None, Trial(0))
    for _ in range(90):  # 1.5초, 연료는 1초 분량
        ep.step([1.0])
    assert ep.bodies["box"].velocity.x == pytest.approx(4.0 / 2.0 * 1.0, rel=1e-6)
    assert ep.block_values()["fuel"] == pytest.approx(1.0)
    assert float(ep.snapshot().outputs[0]) == 0.0


def test_rotated_body_thrust_direction() -> None:
    env = box_env()
    env["bodies"][0]["angle"] = math.pi / 2
    ep = Episode(compile_env(env), None, Trial(0))
    for _ in range(6):
        ep.step([1.0])
    v = ep.bodies["box"].velocity
    assert v.y > 0 and abs(v.x) < 1e-9


def ground_env(**over: Any) -> dict[str, Any]:
    env: dict[str, Any] = {
        "name": "ground",
        "world": {"gravity": [0, -10], "max_time": 3.0, "bounds": [-50, -50, 50, 50]},
        "bodies": [
            {"id": "floor", "owner": "world", "type": "static",
             "shapes": [{"kind": "chain", "points": [[-20, 0], [20, 0]], "radius": 0.05}], "tags": ["ground"]},
            {"id": "box", "pos": [0, 2], "shapes": [{"kind": "box", "w": 1, "h": 1}]},
        ],
        "actuators": [{"kind": "thruster", "id": "up", "body": "box", "max_force": 1000.0}],
        "sensors": [{"kind": "contact", "id": "touch", "body": "box", "with": ["tag.ground"]}],
        "events": [{"id": "resting", "when": "contact(box, floor) and box.speed < 0.05", "hold": 0.5}],
        "termination": [{"event": "resting", "outcome": "success", "label": "멈춤"}],
        "scoring": {"blocks": [
            {"kind": "contact_time", "id": "ct", "body": "box", "with": ["tag.ground"]},
            {"kind": "survival_time", "id": "t"},
            {"kind": "event", "id": "rest", "event": "resting", "weight": 10},
        ]},
    }
    env.update(over)
    return env


def test_contact_sensor_event_hold_and_termination() -> None:
    spec = compile_env(ground_env())
    ep = Episode(spec, None, Trial(0))
    touched_at = None
    while not ep.done:
        ep.step([0.0])
        if touched_at is None and ep.observe()[0] == 1.0:
            touched_at = ep.time
    res = ep.result()
    assert touched_at is not None and touched_at < 1.0  # 1.5m 떨어지는 데 약 0.55초
    assert res.outcome == "success" and res.end_label == "멈춤"
    assert res.time >= touched_at + 0.5 - 1e-9  # 0.5초 유지 뒤에 끝난다
    assert res.blocks["ct"] == pytest.approx(res.time - touched_at, abs=0.05)
    assert res.score == pytest.approx(10.0 + res.blocks["ct"] + res.blocks["t"])  # 가중치 기본 1
    assert res.weighted["rest"] == 10.0


def test_contact_debounce_holds_through_short_bounces() -> None:
    spec = compile_env(ground_env())
    ep = Episode(spec, None, Trial(0))
    while ep.observe()[0] == 0.0:
        ep.step([0.0])
    ep.step([1.0])  # 강하게 밀어 올려 떨어지게 한다
    ep.step([1.0])
    assert ep.observe()[0] == 1.0  # 떨어진 지 0.1초 안이라 아직 닿은 것으로 본다
    for _ in range(12):
        ep.step([0.0])
    assert ep.observe()[0] == 0.0


def test_termination_order_is_priority() -> None:
    env = ground_env()
    env["events"] = [{"id": "a", "when": "time > 0.5"}, {"id": "b", "when": "time > 0.5"}]
    env["termination"] = [{"event": "b", "outcome": "failure", "label": "B"}, {"event": "a", "outcome": "success", "label": "A"}]
    env["scoring"] = {"blocks": [{"kind": "survival_time", "id": "t"}]}
    ep = Episode(compile_env(env), None, Trial(0))
    while not ep.done:
        ep.step([0.0])
    assert ep.result().end_label == "B"


def test_out_of_bounds_and_timeout_builtin_events() -> None:
    env = box_env()
    env["world"]["bounds"] = [-1, -1, 1, 1]
    env["actuators"][1]["max_force"] = 400.0
    env["termination"] = [{"event": "out_of_bounds", "outcome": "failure", "label": "이탈"}]
    ep = Episode(compile_env(env), None, Trial(0))
    while not ep.done:
        ep.step([1.0])
    assert ep.result().end_event == "out_of_bounds"
    ep2 = Episode(compile_env(box_env()), None, Trial(0))
    while not ep2.done:
        ep2.step([0.0])
    res = ep2.result()
    assert res.end_event == "timeout" and res.steps == 120 and res.time == pytest.approx(2.0)


def test_scoring_formula_overrides_weights() -> None:
    env = ground_env()
    env["scoring"]["formula"] = "rest * 50 + (5 if success else 0) - ct"
    res = None
    ep = Episode(compile_env(env), None, Trial(0))
    while not ep.done:
        ep.step([0.0])
    res = ep.result()
    assert res.weighted == {}
    assert res.score == pytest.approx(55.0 - res.blocks["ct"])


def test_expr_block_aggregations() -> None:
    env = box_env()
    env["scoring"]["blocks"] = [
        {"kind": "expr", "id": "integ", "expr": "1", "agg": "integral"},
        {"kind": "expr", "id": "mean_t", "expr": "time", "agg": "mean"},
        {"kind": "expr", "id": "max_v", "expr": "box.vx", "agg": "max"},
        {"kind": "expr", "id": "last", "expr": "time * 2", "agg": "final"},
    ]
    ep = Episode(compile_env(env), None, Trial(0))
    while not ep.done:
        ep.step([0.5])
    b = ep.result().blocks
    assert b["integ"] == pytest.approx(2.0)
    assert b["mean_t"] == pytest.approx((1 / 60 + 2.0) / 2, rel=1e-3)
    assert b["max_v"] > 0
    assert b["last"] == pytest.approx(4.0)


def test_place_by_puts_child_at_joint_anchor(lander: EnvSpec) -> None:
    ep = Episode(lander, None, Trial(0))
    hull, leg = ep.bodies["hull"], ep.bodies["leg_l"]
    anchor_hull = hull.local_to_world((-0.8, -0.5))
    anchor_leg = leg.local_to_world((0.0, 0.6))
    assert (anchor_hull - anchor_leg).length < 1e-9
    assert leg.angle - hull.angle == pytest.approx(-0.45)


def test_rot_spring_holds_rest_angle(lander: EnvSpec) -> None:
    # 공중에서 주 추진기만 켜면 다리는 쉬는 각도(좌 -0.45, 우 +0.45) 근처에 머물러야 한다.
    ep = Episode(lander, None, Trial(1))
    for _ in range(150):
        ep.step([0.5, 0.0, 0.0])
    hull = ep.bodies["hull"]
    assert ep.bodies["leg_l"].angle - hull.angle == pytest.approx(-0.45, abs=0.03)
    assert ep.bodies["leg_r"].angle - hull.angle == pytest.approx(0.45, abs=0.03)


def test_episode_build_cost_is_small(lander: EnvSpec) -> None:
    """에피소드 하나를 만드는 비용이 작다(시험마다 세계를 새로 만드는 설계의 전제).

    처음 한 번은 재지 않는다(모듈 불러오기·C 함수 찾기 같은 한 번뿐인 일이 섞인다). 20번의 평균이 아니라 중앙값을 본다:
    윈도우 CI 러너가 잠깐 멈추면 평균이 한두 번의 긴 값에 끌려간다(v6.1 CI에서 평균 52.9ms로 한 번 넘음. 리눅스의 실제 값은 2~3ms)."""
    import statistics
    import time

    Episode(lander, None, Trial(0))
    times = []
    for s in range(20):
        t0 = time.perf_counter()
        Episode(lander, None, Trial(s))
        times.append(time.perf_counter() - t0)
    per = statistics.median(times)
    print(f"lander episode build: median {per * 1000:.2f} ms, max {max(times) * 1000:.2f} ms")
    assert per < 0.05

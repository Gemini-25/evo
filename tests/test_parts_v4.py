"""형식 4 부품의 런타임 (스포츠 설계 6.1 T5, DE 4.2, 묶음 R).

- 밀기(Move): 최고 속도가 max_speed에 수렴(1%), 방식 셋(xy·x·heading)의 힘 방향, heading의 도는 속도 = 출력 x max_turn,
  없는 물체(v2.1)에 단 밀기의 채널은 죽은 채널.
- 균형 보조(Balance): 기울인 물체가 목표 각도로 돌아옴, 토크가 max_torque에서 잘림, 원형(eng/wtb 패치)의 식 그대로.
  측정에 쓴 달리기 원형 파일에서 보조를 빼면 행동 0으로 더 빨리 넘어진다.
- 바닥 마찰(World.damping): 0.5면 1초 뒤 속도가 절반(2%).
- 기본값이면 v6.1과 같은 연산: 바닥 마찰 1을 적은 파일, 힘 0인 밀기, 세기 0인 보조는 그 부품이 없는 파일과 궤적이 바이트까지
  같다(옛 파일 전부의 얼린 v6.1 비교는 tests/test_frozen_identity.py).
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab.engine import seeds
from evolab.spec.io import compile_env
from evolab.world.physics2d.episode import Episode, Trial

ROOT = Path(__file__).resolve().parents[1]
MEASURED = ROOT / "tests" / "data" / "presets" / "running_measured.evoenv"
DT = 1.0 / 60.0


def world(view: str = "top", damping: float = 1.0, gravity: tuple[float, float] = (0.0, 0.0),
          angle: float = 0.0, actuators: list[dict[str, Any]] | None = None, shape: dict[str, Any] | None = None,
          max_time: float = 10.0) -> dict[str, Any]:
    """물체 하나(마찰 없는 원)와 구동기만 있는 세계. 땅도 벽도 없어 접촉이 없다."""
    return {
        "version": 4, "name": "부품",
        "world": {"view": view, "gravity": list(gravity), "damping": damping, "bounds": [-500.0, -500.0, 500.0, 500.0],
                  "max_time": max_time, "dt": DT},
        "agent": {"spawn": {"x": 0.0, "y": 0.0, "angle": angle}},
        "bodies": [{"id": "disc", "shapes": [shape or {"kind": "circle", "r": 0.4}], "material": {"density": 2.0}}],
        "actuators": actuators if actuators is not None else [],
        "sensors": [{"kind": "body_state", "id": "me", "body": "disc", "fields": ["x", "y", "vx", "vy"]}],
        "scoring": {"blocks": [{"kind": "progress", "id": "go", "body": "disc", "weight": 1.0}]},
    }


def move(mode: str = "xy", **kw: Any) -> dict[str, Any]:
    return {"kind": "move", "id": "push", "body": "disc", "mode": mode, "max_force": 30.0, "max_speed": 2.5, **kw}


def run(data: dict[str, Any], action: list[float], steps: int, trial: int = 0) -> Episode:
    spec = compile_env(data)
    assert [i for i in spec.issues if i.severity == "error"] == []
    ep = Episode(spec, None, Trial(trial))
    for _ in range(steps):
        ep.step(list(action))
    return ep


def velocity(ep: Episode) -> tuple[float, float, float]:
    b = ep.bodies["disc"]
    return b.velocity.x, b.velocity.y, b.angular_velocity


# ---------------------------------------------------------------------------
# 밀기
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("action, want", [([1.0, 0.0], (2.5, 0.0)), ([0.0, -1.0], (0.0, -2.5)),
                                          ([-1.0, 1.0], (-2.5, 2.5)), ([0.4, 0.0], (1.0, 0.0))])
def test_move_xy_reaches_max_speed_times_the_output_in_world_axes(action: list[float], want: tuple[float, float]) -> None:
    for angle in (0.0, 1.0):  # 몸 각도와 무관하게 세계 좌표의 두 방향
        ep = run(world(actuators=[move()], angle=angle), action, 180)
        vx, vy, w = velocity(ep)
        assert vx == pytest.approx(want[0], rel=0.01, abs=1e-6) and vy == pytest.approx(want[1], rel=0.01, abs=1e-6)
        assert w == pytest.approx(0.0, abs=1e-9)
        assert ep.applied_outputs == pytest.approx(action)


def test_move_speed_never_overshoots_and_rises_like_the_exponential() -> None:
    data = world(actuators=[move()])
    spec = compile_env(data)
    ep = Episode(spec, None, Trial(0))
    mass = ep.bodies["disc"].mass
    k = 30.0 / 2.5 / mass
    speeds = []
    for _ in range(120):
        ep.step([1.0, 0.0])
        speeds.append(ep.bodies["disc"].velocity.x)
    assert max(speeds) <= 2.5 * (1 + 1e-9)
    assert all(b >= a for a, b in itertools.pairwise(speeds))
    for n in (10, 30, 60):  # 지수 적분: v(n) = vmax (1 − e^(−k n dt))
        assert speeds[n - 1] == pytest.approx(2.5 * (1 - math.exp(-k * n * DT)), rel=0.01)


def test_move_x_pushes_only_along_the_world_x_axis() -> None:
    data = world(view="side", actuators=[move("x")], gravity=(0.0, -9.81))
    spec = compile_env(data)
    assert spec.outputs == ["push.x"]
    ep = run(data, [-1.0], 120)
    vx, vy, _w = velocity(ep)
    assert vx == pytest.approx(-2.5, rel=0.01)
    assert vy == pytest.approx(-9.81 * 2.0, rel=1e-6)  # 세로는 중력만(밀기가 막지 않음)


def test_move_heading_pushes_along_the_body_and_turns_at_max_turn() -> None:
    data = world(actuators=[move("heading", max_turn=2.0)], angle=math.pi / 2)
    spec = compile_env(data)
    assert spec.outputs == ["push.forward", "push.turn"]
    ep = run(data, [1.0, 0.0], 180)
    vx, vy, w = velocity(ep)
    assert vy == pytest.approx(2.5, rel=0.01) and abs(vx) < 0.01 and w == 0.0
    ep = run(data, [0.0, -0.5], 3)
    assert velocity(ep)[2] == pytest.approx(-1.0)
    assert ep.bodies["disc"].angle == pytest.approx(math.pi / 2 - 3 * DT * 1.0)
    # 돌면서 밀면 몸이 향한 쪽으로 간다: 반 바퀴 돈 뒤에는 처음과 반대쪽
    ep = run(world(actuators=[move("heading", max_turn=math.pi)]), [1.0, 1.0], 60)
    assert ep.bodies["disc"].angle == pytest.approx(math.pi, rel=1e-9)


def test_move_on_an_absent_body_is_a_dead_channel() -> None:
    data = world(actuators=[move()])
    data["params"] = [{"id": "on", "value": 0.0, "min": 0.0, "max": 1.0, "evolve": True}]
    data["bodies"].append({"id": "pod", "pos": [1.0, 0.0], "exists": "=p.on", "place_by": "hinge",
                           "shapes": [{"kind": "circle", "r": 0.2}]})
    data["joints"] = [{"kind": "pivot", "id": "hinge", "a": "disc", "b": "pod", "anchor_a": [0.5, 0.0],
                       "anchor_b": [-0.5, 0.0]}]
    data["actuators"].append({**move(), "id": "pod_push", "body": "pod"})
    spec = compile_env(data)
    assert [i for i in spec.issues if i.severity == "error"] == []
    ep = Episode(spec, np.array([0.0]), Trial(0))
    assert ep.absent == frozenset({"pod"})
    assert ep.dead_channels == ("pod_push.x", "pod_push.y")
    for _ in range(30):
        ep.step([1.0, 0.0, 1.0, 1.0])
    assert ep.applied_outputs == [1.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# 균형 보조
# ---------------------------------------------------------------------------

def balance(**kw: Any) -> dict[str, Any]:
    return {"kind": "balance", "id": "bal", "body": "disc", "stiffness": 40.0, "damping": 8.0, "max_torque": 1e6, **kw}


BOX = {"kind": "box", "w": 0.4, "h": 1.2}


@pytest.mark.parametrize("target", [0.0, 0.3])
def test_balance_brings_a_tilted_body_back_to_the_target(target: float) -> None:
    data = world(view="side", actuators=[balance(target=target)], angle=0.8, shape=BOX)
    spec = compile_env(data)
    assert spec.outputs == []
    ep = run(data, [], 300)
    assert ep.bodies["disc"].angle == pytest.approx(target, abs=1e-3)
    assert abs(ep.bodies["disc"].angular_velocity) < 1e-2


def test_balance_torque_is_the_prototype_formula_and_is_cut_at_max_torque() -> None:
    for max_torque, angle in ((1e6, 0.5), (0.7, 0.5), (0.7, -0.5)):
        data = world(view="side", actuators=[balance(stiffness=10.0, damping=3.0, max_torque=max_torque)], angle=angle,
                     shape=BOX)
        ep = Episode(compile_env(data), None, Trial(0))
        moment = ep.bodies["disc"].moment
        w0 = ep.bodies["disc"].angular_velocity
        ep.step([])
        dw = ep.bodies["disc"].angular_velocity - w0
        tq = max(-max_torque, min(max_torque, -10.0 * angle - 3.0 * w0))
        assert dw == pytest.approx(tq / moment * DT, rel=1e-9)


def test_balance_keeps_the_measured_running_body_up_longer_with_no_action() -> None:
    """원형 파일(보조 100): 행동 0으로 서 있는 시간의 합이 보조를 뺀 몸보다 길다(DE 9.1-1: 보조 100도 2.7초 무렵 넘어짐)."""
    data = json.loads(MEASURED.read_text(encoding="utf-8"))
    plain = {**data, "actuators": [a for a in data["actuators"] if a["kind"] != "balance"]}
    times = {}
    for label, d in (("bal", data), ("plain", plain)):
        spec = compile_env(d)
        total = 0.0
        for t in seeds.validation_trials(4):
            ep = Episode(spec, None, t)
            while not ep.done:
                ep.step([0.0] * len(spec.outputs))
            total += ep.result().time
        times[label] = total
    assert times["bal"] > times["plain"] + 1.0


# ---------------------------------------------------------------------------
# 바닥 마찰
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("damping", [0.5, 0.2, 0.9])
def test_damping_leaves_that_share_of_the_speed_after_one_second(damping: float) -> None:
    ep = Episode(compile_env(world(damping=damping)), None, Trial(0))
    ep.bodies["disc"].velocity = (2.0, -1.0)
    ep.bodies["disc"].angular_velocity = 3.0
    for _ in range(60):
        ep.step([])
    vx, vy, w = velocity(ep)
    assert vx == pytest.approx(2.0 * damping, rel=0.02)
    assert vy == pytest.approx(-1.0 * damping, rel=0.02)
    assert w == pytest.approx(3.0 * damping, rel=0.02)
    assert ep.space.damping == damping


def test_move_still_reaches_its_speed_on_a_damped_floor() -> None:
    """바닥 마찰이 있으면 같은 힘으로 덜 빨라진다(최고 속도는 넘지 않는다)."""
    free = velocity(run(world(actuators=[move()]), [1.0, 0.0], 180))[0]
    damped = velocity(run(world(actuators=[move()], damping=0.5), [1.0, 0.0], 180))[0]
    assert damped < free <= 2.5 * (1 + 1e-9) and damped > 1.0


# ---------------------------------------------------------------------------
# 기본값이면 같은 연산
# ---------------------------------------------------------------------------

def trace(data: dict[str, Any], n_out: int, steps: int = 240) -> bytes:
    spec = compile_env(data)
    rng = np.random.default_rng(5)
    out = []
    for t in seeds.validation_trials(2):
        ep = Episode(spec, None, t)
        for _ in range(steps):
            if ep.done:
                break
            act = list(rng.uniform(-1, 1, n_out))
            ep.step(act + [0.0] * (len(spec.outputs) - n_out))
            out.append(ep.snapshot().poses.tobytes())
        r = ep.result()  # 행동 크기 사전에는 더한 채널(늘 0)이 끼므로 빼고 견준다
        out.append(repr((r.score, r.blocks, r.weighted, r.outcome, r.end_event, r.time, r.steps)).encode())
    return b"".join(out)


def test_default_valued_parts_change_nothing_in_the_measured_running_file() -> None:
    data = json.loads(MEASURED.read_text(encoding="utf-8"))
    plain = {**data, "actuators": [a for a in data["actuators"] if a["kind"] != "balance"]}
    ref = trace(plain, 6)
    torso = "torso"
    zero_balance = {**plain, "actuators": [*plain["actuators"], {"kind": "balance", "id": "bal", "body": torso,
                                                                   "stiffness": 0.0, "max_torque": 100.0}]}
    zero_move = {**plain, "actuators": [*plain["actuators"], {"kind": "move", "id": "push", "body": torso, "mode": "x",
                                                                "max_force": 0.0, "max_speed": 1.0}]}
    explicit = {**plain, "world": {**plain["world"], "damping": 1.0, "view": "side"}}
    assert trace(zero_balance, 6) == ref
    assert trace(zero_move, 6) == ref
    assert trace(explicit, 6) == ref
    assert trace(data, 6) != ref  # 보조를 켜면 다르다(헛도는 비교가 아님)

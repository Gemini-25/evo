"""v2.0 점프 추진기(설계 4.3, D10·D11, T12~T14): 뛰는 규칙, 코요테 시간, 정수 스텝 냉각, 높이, 세계 좌표 방향."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from evolab import paths
from evolab.spec.io import compile_env
from evolab.world.physics2d.episode import Episode, Trial

JUMP_TEXT = (paths.builtin_env_dir() / "jump.evoenv").read_text(encoding="utf-8")
G = 9.81
DT = 1.0 / 60.0


def without_obstacles(data: dict[str, Any]) -> dict[str, Any]:
    """점프 게임에서 장애물 줄과 그것을 보는 센서·사건을 뺀다(땅과 선수, 점프 추진기만)."""
    data["generators"] = []
    data["randomize"] = []
    data["sensors"] = [s for s in data["sensors"] if s["kind"] not in ("rays",) and s["body"] != "obs"]
    data["events"] = []
    data["termination"] = [t for t in data["termination"] if t["event"] != "hit"]
    return data


def jump_env(*, obstacles: bool = False, spawn: dict[str, Any] | None = None, **jump: Any) -> Any:
    """점프 게임에서 장애물 줄을 뺀(기본) 작은 환경. jump 칸은 점프 추진기에 덮어쓴다."""
    data = json.loads(JUMP_TEXT)
    if not obstacles:
        without_obstacles(data)
    data["actuators"][0].update(jump)
    if spawn:
        data["agent"]["spawn"].update(spawn)
    return compile_env(json.dumps(data))


def settle(ep: Episode, steps: int = 30) -> None:
    for _ in range(steps):
        ep.step([0.0])


def jumped(ep: Episode) -> bool:
    return ep.applied_outputs[ep.output_channel_index[0]] == 1.0


def test_jump_from_ground_sets_the_jump_speed() -> None:
    ep = Episode(jump_env(), None, Trial(1))
    runner = ep.bodies["runner"]
    settle(ep)
    assert abs(runner.velocity.y) < 1e-6 and ep.jump_ready("jump")
    ep.step([0.49])  # 0.5 미만은 뛰지 않는다
    assert not jumped(ep) and abs(runner.velocity.y) < 1e-6
    ep.step([0.5])
    assert jumped(ep)
    # 뛴 스텝: 속도를 6m/s로 맞춘 뒤 그 스텝의 중력만 더해진다(땅이 붙잡지 않음).
    assert runner.velocity.y == pytest.approx(6.0 - G * DT, abs=1e-9)
    assert not ep.jump_ready("jump")


def test_no_jump_in_the_air_and_output_is_one_only_on_jump_steps() -> None:
    ep = Episode(jump_env(), None, Trial(1))
    runner = ep.bodies["runner"]
    settle(ep)
    applied = []
    for _ in range(60):
        ep.step([1.0])  # 계속 누름: 착지하면 다시 뛴다
        applied.append(ep.applied_outputs[0])
    assert applied[0] == 1.0 and set(applied) == {0.0, 1.0}
    first_air = applied.index(0.0)
    # 공중(디바운스·냉각이 지난 뒤)에서는 누르고 있어도 속도가 바뀌지 않는다: 중력만큼만 줄어든다.
    vy_before = runner.velocity.y
    ep.step([1.0])
    assert ep.applied_outputs[0] == 0.0
    assert runner.velocity.y == pytest.approx(vy_before - G * DT, abs=1e-9)
    assert runner.position.y > 0.6  # 아직 공중(착지는 약 73스텝째)
    assert sum(applied) == 1.0  # 60스텝(1초) 동안 공중이라 한 번만 뛰었다
    assert first_air == 1


def test_coyote_time_within_contact_debounce() -> None:
    """땅을 떠난 뒤에도 접촉 디바운스(0.1초) 안이면 뛸 수 있다. 냉각 0으로 연달아 뛰어 본다."""
    ep = Episode(jump_env(cooldown=0.0), None, Trial(1))
    settle(ep)
    flags = []
    for _ in range(20):
        ep.step([1.0])
        flags.append(jumped(ep))
    # 처음 뛴 뒤 디바운스 0.1초(6스텝) 동안은 공중에서도 다시 뛴다(이미 위로 빠르면 속도는 그대로).
    run = flags.index(False)
    assert 6 <= run <= 8
    assert not any(flags[run:])  # 디바운스가 지나면 공중이라 못 뛴다


def test_cooldown_is_an_integer_number_of_steps() -> None:
    """냉각 0.25초 = ⌈0.25 x 60 - 1e-9⌉ = 15스텝: 마지막으로 뛴 스텝에서 15스텝째에 처음 다시 뛸 수 있다."""
    ep = Episode(jump_env(ground=[]), None, Trial(1))  # 땅이 비면 공중에서도 뛴다
    steps = []
    for k in range(46):
        ep.step([1.0])
        if jumped(ep):
            steps.append(k)
    assert steps == [0, 15, 30, 45]
    ep2 = Episode(jump_env(ground=[], cooldown=0.26), None, Trial(1))  # 15.6 → 16스텝
    steps2 = []
    for k in range(40):
        ep2.step([1.0])
        if jumped(ep2):
            steps2.append(k)
    assert steps2 == [0, 16, 32]


def test_empty_ground_jumps_in_the_air_and_faster_bodies_keep_their_speed() -> None:
    ep = Episode(jump_env(ground=[], cooldown=0.0, speed=6.0), None, Trial(1))
    runner = ep.bodies["runner"]
    ep.step([1.0])  # 시작부터 공중 점프(땅이 비었으므로)
    assert jumped(ep) and runner.velocity.y == pytest.approx(6.0 - G * DT, abs=1e-9)
    fast = Episode(jump_env(ground=[], spawn={"vy": 10.0}), None, Trial(1))
    body = fast.bodies["runner"]
    fast.step([1.0])
    assert jumped(fast)  # 조건이 맞으면 뛴 스텝(냉각 시작)이지만
    assert body.velocity.y == pytest.approx(10.0 - G * DT, abs=1e-9)  # 이미 더 빠르면 속도는 그대로


def test_jump_height_and_ready_again_steps_match_the_measurement() -> None:
    """T13: 점프 게임 몸(30kg)에서 속도 6m/s → 최고 높이 1.885 ± 0.01m, 다시 뛸 수 있을 때까지 76스텝(설계 측정)."""
    ep = Episode(jump_env(), None, Trial(1))
    runner = ep.bodies["runner"]
    settle(ep)
    y0 = runner.position.y
    ep.step([1.0])
    assert jumped(ep)
    apex = runner.position.y - y0
    steps = 1
    while True:
        ep.step([1.0])
        if jumped(ep):
            break
        apex = max(apex, runner.position.y - y0)
        steps += 1
    assert apex == pytest.approx(1.885, abs=0.01)
    # 설계의 76스텝은 원형 계획기의 점프 궤적 길이(뛴 호출 직전 자세 + 다시 뛰는 호출 직전까지의 자세)다.
    # 뛴 호출 사이 간격으로는 75스텝이다(tests/jump_planner.py의 JumpProfile.ready_after와 같은 값).
    assert steps + 1 == 76
    assert 6.0 ** 2 / (2 * G) < apex  # 어림 speed²/(2g) = 1.83m보다 조금 높다(반암시적 적분, 마지막 스텝)


def test_jump_direction_is_in_world_coordinates() -> None:
    """T14: dir은 세계 좌표라 물체가 기울어도 같은 방향으로 뛴다."""
    out = []
    for angle in (0.0, 0.7, -1.2):
        spec = jump_env(ground=[], dir=math.pi / 3, spawn={"angle": angle, "y": 5.0})
        ep = Episode(spec, None, Trial(1))
        ep.step([1.0])
        v = ep.bodies["runner"].velocity
        out.append((v.x, v.y + G * DT))
    for vx, vy in out:
        assert vx == pytest.approx(6.0 * math.cos(math.pi / 3), abs=1e-9)
        assert vy == pytest.approx(6.0 * math.sin(math.pi / 3), abs=1e-9)


def test_jump_ready_is_false_for_other_channels_and_in_the_air() -> None:
    ep = Episode(jump_env(), None, Trial(1))
    assert not ep.jump_ready("jump")  # 첫 스텝 전에는 아직 땅 접촉을 모른다(앞 스텝 값)
    settle(ep, 5)
    assert ep.jump_ready("jump") and not ep.jump_ready("nope")
    ep.step([1.0])
    for _ in range(20):
        ep.step([0.0])
    assert not ep.jump_ready("jump")  # 공중


def test_jump_energy_block_counts_jumps() -> None:
    data = without_obstacles(json.loads(JUMP_TEXT))
    data["scoring"]["blocks"].append({"id": "e", "kind": "energy"})
    data["world"]["max_time"] = 3.0
    ep = Episode(compile_env(json.dumps(data)), None, Trial(1))
    jumps = 0
    while not ep.done:
        ep.step([1.0])
        jumps += jumped(ep)
    res = ep.result()
    assert jumps == 3  # 0초, 약 1.27초, 약 2.53초
    assert res.blocks["e"] == pytest.approx(jumps * DT)
    assert res.action_mean_abs["jump"] == pytest.approx(jumps / res.steps)


def test_jump_game_keeps_obstacle_free_start() -> None:
    spec = jump_env(obstacles=True)
    ep = Episode(spec, None, Trial(7))
    settle(ep, 10)
    assert not ep.done and ep.jump_ready("jump")


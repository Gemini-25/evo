"""v2.0 정해진 움직임 물체의 런타임(설계 4.1, D5~D8, T9·T10).

순수 함수 pose_at 자체(T7)는 tests/test_format_v2.py가 본다. 여기서는 Episode가 스텝마다 그 자세에 정확히 닿는 속도를
주는지, 위에 올라탄 물체가 실려 가는지, 레이가 그 스텝의 자세를 보는지, 개체 행동과 무관한지를 본다.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from evolab import paths
from evolab.spec.io import compile_env
from evolab.world.physics2d.episode import Episode, Trial
from evolab.world.physics2d.motion import pose_at

JUMP_TEXT = (paths.builtin_env_dir() / "jump.evoenv").read_text(encoding="utf-8")


def mover(bid: str, pos: list[float], motion: dict[str, Any], w: float = 1.0, h: float = 0.4,
          friction: float = 0.7, angle: float = 0.0) -> dict[str, Any]:
    return {"id": bid, "owner": "world", "type": "kinematic", "pos": pos, "angle": angle,
            "shapes": [{"kind": "box", "w": w, "h": h}], "material": {"friction": friction}, "motion": motion}


def env(bodies: list[dict[str, Any]], *, gravity: float = -9.81, sensors: list[dict[str, Any]] | None = None,
        max_time: float = 20.0) -> Any:
    data = {
        "name": "움직임 시험",
        "world": {"gravity": [0.0, gravity], "max_time": max_time, "bounds": [-100, -100, 100, 100]},
        "bodies": bodies,
        "sensors": sensors or [],
        "scoring": {"blocks": [{"id": "t", "kind": "survival_time"}]},
    }
    return compile_env(json.dumps(data))


MOTIONS = {
    "once": {"kind": "path", "points": [[0, 0], [3, 0], [3, 2]], "speed": 1.5, "accel": 0.2, "start": 0.5},
    "loop": {"kind": "path", "points": [[0, 0], [2, 0], [2, 2]], "speed": 2.0, "mode": "loop"},
    "pingpong": {"kind": "path", "points": [[0, 0], [0, 3]], "speed": 1.0, "mode": "pingpong"},
    "spin": {"kind": "path", "points": [[0, 0], [1, 1]], "speed": 0.5, "spin": -1.2},
    "wave": {"kind": "wave", "amplitude": [0.5, 1.0], "period": 1.7, "phase": 0.25, "spin": 0.7},
    "slowing": {"kind": "path", "points": [[0, 0], [20, 0]], "speed": 3.0, "accel": -1.0},
}


def test_runtime_pose_follows_pose_at_for_every_mode() -> None:
    """T9: 600스텝 동안 런타임 위치·각도와 pose_at(스텝 x dt)의 차 ≤ 1e-9."""
    bodies = [mover(name, [6.0 * k, 3.0], mo, angle=0.3 * k) for k, (name, mo) in enumerate(MOTIONS.items())]
    spec = env(bodies)
    ep = Episode(spec, None, Trial(1))
    plans = {g.id: g.motion for g in ep.geometry().bodies}
    assert all(plans[name] is not None for name in MOTIONS)
    assert ep.moving_bodies == []  # 정해진 움직임 물체는 궤적(움직이는 물체)에 넣지 않는다
    worst = 0.0
    for _ in range(600):
        ep.step([])
        t = ep.steps * ep.dt
        for name in MOTIONS:
            b = ep.bodies[name]
            x, y, a = pose_at(plans[name], t)
            worst = max(worst, abs(b.position.x - x), abs(b.position.y - y), abs(b.angle - a))
    assert worst <= 1e-9


def test_start_pose_is_pose_at_zero_and_geometry_carries_the_plan() -> None:
    spec = env([mover("m", [1.0, 2.0], MOTIONS["wave"], angle=0.4)])
    ep = Episode(spec, None, Trial(1))
    geo = next(g for g in ep.geometry().bodies if g.id == "m")
    assert not geo.moving and geo.motion is not None
    assert geo.pose == pytest.approx(pose_at(geo.motion, 0.0), abs=1e-12)
    b = ep.bodies["m"]
    assert (b.position.x, b.position.y, b.angle) == pytest.approx(pose_at(geo.motion, 0.0), abs=1e-12)


def test_box_rides_a_pingpong_platform() -> None:
    """T9: 왕복 발판(마찰 1) 위 상자는 일정 구간에서 발판과 같이 간다(상대 위치 변화 < 1mm)."""
    platform = mover("plat", [5.0, 1.0], {"kind": "path", "points": [[0, 0], [4, 0]], "speed": 2.0, "mode": "pingpong"},
                     w=3.0, h=0.2, friction=1.0)
    box = {"id": "box", "owner": "agent", "pos": [5.0, 1.3], "shapes": [{"kind": "box", "w": 0.4, "h": 0.4}],
           "material": {"density": 2.0, "friction": 1.0}}
    spec = env([platform, box])
    ep = Episode(spec, None, Trial(1))
    rel = []
    for _ in range(int(1.9 / ep.dt)):
        ep.step([])
        if ep.time >= 0.6:  # 처음 미끄러짐이 끝난 뒤, 첫 되돌아옴(2초) 전
            rel.append(ep.bodies["box"].position.x - ep.bodies["plat"].position.x)
    assert max(rel) - min(rel) < 1e-3
    assert ep.bodies["box"].position.x > 7.0  # 실제로 실려 갔다


def test_teleport_free_velocity_matches_motion() -> None:
    """정해진 움직임 물체의 속도는 그 스텝에 준 속도(다음 자세까지의 평균 속도)라 몸 상태 이름으로 읽을 수 있다."""
    spec = env([mover("m", [0.0, 0.0], {"kind": "path", "points": [[0, 0], [10, 0]], "speed": 2.0})])
    ep = Episode(spec, None, Trial(1))
    ep.step([])
    assert ep.bodies["m"].velocity.x == pytest.approx(2.0, abs=1e-9)


def test_obstacle_pose_does_not_depend_on_the_agent() -> None:
    """T9: 장애물 자세는 개체 행동과 무관하다(행동이 다른 두 개체에서 같은 값)."""
    spec = compile_env(JUMP_TEXT)
    a = Episode(spec, None, Trial(5))
    b = Episode(spec, None, Trial(5))
    same = 0
    for k in range(400):
        if a.done or b.done:
            break
        a.step([0.0])
        b.step([1.0 if k % 30 < 2 else 0.0])
        pa, pb = a.bodies["obs"].position, b.bodies["obs"].position
        assert (pa.x, pa.y) == (pb.x, pb.y)
        same += 1
    assert same > 100


def test_rays_see_the_pose_of_this_step() -> None:
    """T10: 다가오는 kinematic 물체의 레이 거리 = 그 스텝 pose_at으로 계산한 거리(1e-6 안, 모양 캐시 늦음 없음)."""
    agent = {"id": "me", "owner": "agent", "pos": [0.0, 0.0], "shapes": [{"kind": "box", "w": 1.0, "h": 1.0}]}
    target = mover("wall", [12.0, 0.0], {"kind": "path", "points": [[0, 0], [-8, 0]], "speed": 3.0}, w=1.0, h=2.0)
    rays = {"kind": "rays", "id": "eye", "body": "me", "origin": [0.5, 0.0], "count": 1, "spread": 0.0,
            "length": 20.0, "hits": ["wall"]}
    spec = env([agent, target], gravity=0.0, sensors=[rays])
    ep = Episode(spec, None, Trial(1))
    plan = next(g.motion for g in ep.geometry().bodies if g.id == "wall")
    checked = 0
    for _ in range(200):
        x_left = pose_at(plan, ep.steps * ep.dt)[0] - 0.5
        expected = (x_left - 0.5) / 20.0
        got = float(ep.channel_values()["eye.r0"])
        assert got == pytest.approx(expected, abs=1e-6)
        checked += 1
        ep.step([])
    assert checked == 200 and ep.channel_values()["eye.r0"] < 0.3  # 실제로 다가왔다


def test_v1_files_have_no_kinematic_bodies() -> None:
    for name in ("lander", "track_car", "hill_car"):
        spec = compile_env((paths.builtin_env_dir() / f"{name}.evoenv").read_text(encoding="utf-8"))
        ep = Episode(spec, None, Trial(1))
        assert ep._kinematic == [] and all(g.motion is None for g in ep.geometry().bodies)


def test_spinning_wave_angle_is_linear_in_time() -> None:
    spec = env([mover("m", [0.0, 0.0], MOTIONS["wave"], angle=0.2)])
    ep = Episode(spec, None, Trial(1))
    for _ in range(90):
        ep.step([])
    assert ep.bodies["m"].angle == pytest.approx(0.2 + 0.7 * 90 * ep.dt, abs=1e-9)
    assert math.isfinite(np.float64(ep.bodies["m"].position.y))

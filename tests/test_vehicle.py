"""M4: 차량 구동기, 레이 센서, 체크포인트 선(손으로 놓은 것). 트랙 생성기와 무관한 부분."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pytest

from evolab.spec.compiled import EnvFileError
from evolab.spec.io import compile_env
from evolab.world.physics2d.episode import Episode, Trial

BASE: dict[str, Any] = {
    "version": 1,
    "name": "차량 시험장",
    "world": {"kind": "physics2d", "gravity": [0.0, 0.0], "dt": 1.0 / 60.0, "max_time": 20.0,
              "bounds": [-100.0, -100.0, 100.0, 100.0]},
    "bodies": [
        {"id": "car", "owner": "agent", "shapes": [{"kind": "box", "w": 2.0, "h": 1.0}],
         "material": {"density": 50.0, "friction": 0.5}, "tags": ["car"]},
        {"id": "wall", "owner": "world", "type": "static", "pos": [30.0, 0.0],
         "shapes": [{"kind": "box", "w": 1.0, "h": 60.0}], "tags": ["wall"]},
    ],
    "actuators": [
        {"kind": "vehicle", "id": "drive", "body": "car", "max_accel": 6.0, "max_steer": 0.5, "lateral_grip": 1.0,
         "wheelbase": 2.0, "drag": 0.5, "keys_forward": ["Up"], "keys_back": ["Down"], "keys_left": ["Left"],
         "keys_right": ["Right"]},
    ],
    "sensors": [
        {"kind": "rays", "id": "ray", "body": "car", "origin": [1.0, 0.0], "count": 3, "spread": math.pi / 2,
         "length": 40.0, "hits": ["tag.wall"]},
        {"kind": "body_state", "id": "st", "body": "car", "fields": ["v_fwd", "v_lat"], "frame": "body"},
    ],
    "checkpoints": [
        {"id": "cp", "lines": [[[0.0, -5.0], [0.0, 5.0]], [[10.0, -5.0], [10.0, 5.0]], [[20.0, -5.0], [20.0, 5.0]]],
         "loop": False},
    ],
    "events": [
        {"id": "crash", "when": "contact(car, tag.wall)"},
        {"id": "stall", "when": "cp.since > 3.0"},
        {"id": "done", "when": "cp.count >= 2"},
    ],
    "termination": [
        {"event": "done", "outcome": "success"},
        {"event": "crash", "outcome": "failure"},
        {"event": "stall", "outcome": "failure"},
        {"event": "timeout", "outcome": "neutral"},
    ],
    "scoring": {"blocks": [
        {"kind": "checkpoints", "id": "s_cp", "checkpoints": "cp", "weight": 10.0},
        {"kind": "expr", "id": "s_prog", "expr": "cp.progress", "agg": "final", "weight": 1.0},
    ]},
}


def env(**changes: Any) -> Any:
    data = copy.deepcopy(BASE)
    for k, v in changes.items():
        data[k] = v
    return compile_env(data)


def run(ep: Episode, action: list[float], steps: int) -> None:
    for _ in range(steps):
        if ep.done:
            return
        ep.step(action)


def test_vehicle_accelerates_to_top_speed_and_turns_left() -> None:
    spec = env(events=[], termination=[{"event": "timeout", "outcome": "neutral"}],
               bodies=[BASE["bodies"][0]], sensors=[BASE["sensors"][1]], checkpoints=[],
               scoring={"blocks": []})
    ep = Episode(spec, None, Trial(1))
    run(ep, [1.0, 0.0], 600)
    v = ep.channel_values()
    assert v["st.v_fwd"] == pytest.approx(6.0 / 0.5, rel=0.02)  # 최고 속도 = max_accel / drag
    assert abs(v["st.v_lat"]) < 1e-9
    ep2 = Episode(spec, None, Trial(1))
    run(ep2, [1.0, 0.0], 60)
    a0 = ep2.snapshot().poses[0, 2]
    run(ep2, [1.0, 1.0], 30)
    assert ep2.snapshot().poses[0, 2] > a0 + 0.1  # 조향 +는 왼쪽(반시계)
    # 뒤로(가속 -1)
    ep3 = Episode(spec, None, Trial(1))
    run(ep3, [-1.0, 0.0], 60)
    assert ep3.channel_values()["st.v_fwd"] < -1.0


def test_lateral_grip_lets_the_car_slide() -> None:
    bodies = [dict(BASE["bodies"][0])]
    acts = [dict(BASE["actuators"][0], lateral_grip=0.0)]
    spec = env(bodies=bodies, actuators=acts, sensors=[BASE["sensors"][1]], checkpoints=[], events=[],
               termination=[{"event": "timeout", "outcome": "neutral"}], scoring={"blocks": []},
               agent={"spawn": {"vy": 5.0}})
    ep = Episode(spec, None, Trial(1))
    run(ep, [0.0, 0.0], 30)
    assert ep.channel_values()["st.v_lat"] == pytest.approx(5.0, rel=1e-6)  # 미끄럼 마찰 0: 옆 속도 그대로
    grip = env(bodies=bodies, actuators=[dict(BASE["actuators"][0], lateral_grip=1.0)], sensors=[BASE["sensors"][1]],
               checkpoints=[], events=[], termination=[{"event": "timeout", "outcome": "neutral"}],
               scoring={"blocks": []}, agent={"spawn": {"vy": 5.0}})
    ep2 = Episode(grip, None, Trial(1))
    run(ep2, [0.0, 0.0], 2)
    assert abs(ep2.channel_values()["st.v_lat"]) < 1e-9


def test_rays_measure_distance_to_matching_shapes_only() -> None:
    ep = Episode(env(), None, Trial(1))
    v = ep.channel_values()
    # 가운데 레이: 레이 시작점 x=1, 벽 왼쪽 면 x=29.5 → 28.5m / 40m
    assert v["ray.r1"] == pytest.approx(28.5 / 40.0, abs=1e-6)
    # 옆 레이(±45°)는 벽에 사선으로: 28.5 * sqrt(2)가 40보다 멀어 안 맞는다
    assert v["ray.r0"] == 1.0 and v["ray.r2"] == 1.0
    obs = ep.observe()
    assert obs[:3].tolist() == pytest.approx([1.0, 28.5 / 40.0, 1.0], abs=1e-6)
    view = ep.ray_view()
    assert len(view) == 3 and view[1][1][0] == pytest.approx(29.5, abs=1e-6)
    # 감지 대상이 아니면 맞지 않고, 자기 몸은 대상이어도 맞지 않는다
    ep2 = Episode(env(sensors=[dict(BASE["sensors"][0], hits=["tag.car"]), BASE["sensors"][1]]), None, Trial(1))
    assert ep2.channel_values()["ray.r1"] == 1.0


def test_checkpoints_in_order_score_and_success() -> None:
    ep = Episode(env(), None, Trial(1))
    run(ep, [1.0, 0.0], 2000)
    r = ep.result()
    assert r.end_event == "done" and r.outcome == "success"
    assert r.blocks["s_cp"] == 2.0
    assert r.blocks["s_prog"] == 2.0  # 손으로 놓은 체크포인트의 진행 = 지난 개수


def test_wall_crash_and_stall() -> None:
    events = [dict(e) for e in BASE["events"]]
    events[1]["when"] = "cp.since > 10.0"
    spec = env(checkpoints=[{"id": "cp", "lines": [[[0.0, -5.0], [0.0, 5.0]], [[40.0, -5.0], [40.0, 5.0]]],
                             "loop": False}], events=events)
    ep = Episode(spec, None, Trial(1))
    run(ep, [1.0, 0.0], 2000)
    assert ep.result().end_event == "crash"
    ep2 = Episode(spec, None, Trial(1))
    run(ep2, [0.0, 0.0], 2000)
    r2 = ep2.result()
    assert r2.end_event == "stall" and r2.time == pytest.approx(10.0 + 1.0 / 60.0, abs=1e-9)


def test_loop_checkpoints_count_laps() -> None:
    lines = [[[0.0, -3.0], [0.0, 3.0]], [[0.0, 7.0], [0.0, 13.0]]]  # 반지름 5 원을 도는 두 점
    spec = env(bodies=[BASE["bodies"][0]], sensors=[BASE["sensors"][1]],
               checkpoints=[{"id": "cp", "lines": lines, "loop": True}],
               events=[{"id": "laps", "when": "cp.laps >= 2"}],
               termination=[{"event": "laps", "outcome": "success"}, {"event": "timeout", "outcome": "neutral"}],
               scoring={"blocks": [{"kind": "lap_time", "id": "lap", "checkpoints": "cp", "weight": -1.0},
                                   {"kind": "checkpoints", "id": "n", "checkpoints": "cp"}]},
               world={**BASE["world"], "max_time": 60.0})
    ep = Episode(spec, None, Trial(1))
    # 원을 따라 돌게 조향(정속 주행)
    steer = math.atan(2.0 / 5.0) / 0.5
    run(ep, [0.3, steer], 60 * 60)
    r = ep.result()
    assert r.end_event == "laps"
    assert r.blocks["n"] == 4.0
    assert r.blocks["lap"] > 0


def test_vehicle_keys_and_bad_refs() -> None:
    from evolab.app.play.window import key_bindings

    spec = env()
    kb = key_bindings(spec)
    assert sorted((b.output_index, b.value) for b in kb) == [(0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0)]
    bad = copy.deepcopy(BASE)
    bad["scoring"]["blocks"][0]["checkpoints"] = "nope"
    with pytest.raises(EnvFileError):
        compile_env(bad)


def test_same_trial_twice_is_identical() -> None:
    spec = env()
    out = []
    for _ in range(2):
        ep = Episode(spec, None, Trial(5))
        rng = np.random.default_rng(1)
        poses = []
        while not ep.done:
            ep.step(rng.uniform(-1, 1, 2).tolist())
            poses.append(ep.snapshot().poses.copy())
        out.append((ep.result().score, np.array(poses).tobytes()))
    assert out[0] == out[1]


def test_lateral_accel_limit_makes_fast_cars_turn_wider() -> None:
    def turn_radius(lat_max: float, throttle: float) -> float:
        acts = [dict(BASE["actuators"][0], max_lateral_accel=lat_max)]
        spec = env(bodies=[BASE["bodies"][0]], actuators=acts, sensors=[BASE["sensors"][1]], checkpoints=[],
                   events=[], termination=[{"event": "timeout", "outcome": "neutral"}], scoring={"blocks": []})
        ep = Episode(spec, None, Trial(1))
        run(ep, [throttle, 0.0], 600)  # 최고 속도까지
        v = ep.channel_values()["st.v_fwd"]
        a0 = ep.snapshot().poses[0, 2]
        run(ep, [throttle, 1.0], 30)
        omega = (ep.snapshot().poses[0, 2] - a0) / 0.5
        return v / omega

    free = turn_radius(0.0, 1.0)
    limited = turn_radius(4.0, 1.0)
    assert free == pytest.approx(2.0 / math.tan(0.5), rel=0.05)  # 제한 없으면 자전거 모형 반지름
    assert limited == pytest.approx(12.0**2 / 4.0, rel=0.05)  # v²/a: 최고 속도 12m/s, 4m/s²
    assert turn_radius(4.0, 0.2) < limited  # 느리면 더 꺾인다

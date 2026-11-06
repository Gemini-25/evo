"""M5b: 모터 구동기와 언덕 생성기, hill_car.evoenv."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from evolab import paths
from evolab.generators.terrain import generate_hills
from evolab.policy.genome import layout_for
from evolab.spec.compiled import EnvFileError
from evolab.spec.io import compile_env, load_env
from evolab.world.physics2d.episode import Episode, Trial

MOTOR_ENV: dict[str, Any] = {
    "version": 1,
    "name": "모터 시험",
    "world": {"kind": "physics2d", "gravity": [0.0, 0.0], "dt": 1.0 / 60.0, "max_time": 5.0,
              "bounds": [-50.0, -50.0, 50.0, 50.0]},
    "bodies": [
        {"id": "base", "owner": "agent", "shapes": [{"kind": "box", "w": 1.0, "h": 1.0}],
         "material": {"density": 1000.0}},
        {"id": "wheel", "owner": "agent", "pos": [2.0, 0.0], "shapes": [{"kind": "circle", "r": 0.5}],
         "material": {"density": 1.0}},
    ],
    "joints": [{"id": "axle", "kind": "pivot", "a": "base", "b": "wheel", "anchor_a": [2.0, 0.0]}],
    "actuators": [{"kind": "motor", "id": "mot", "joint": "axle", "max_torque": 1000.0, "max_speed": 3.0,
                   "keys_pos": ["Left"], "keys_neg": ["Right"]}],
    "sensors": [{"kind": "body_state", "id": "w", "body": "wheel", "fields": ["omega"]}],
    "termination": [{"event": "timeout", "outcome": "neutral"}],
}


@pytest.mark.parametrize("u", [1.0, -0.5])
def test_motor_spins_child_relative_to_parent(u: float) -> None:
    ep = Episode(compile_env(MOTOR_ENV), None, Trial(1))
    for _ in range(60):
        ep.step([u])
    base_w = ep._bstate("base")[5]
    wheel_w = ep.channel_values()["w.omega"]
    assert wheel_w - base_w == pytest.approx(u * 3.0, rel=0.02)  # +출력 = b가 a에 대해 반시계


def test_motor_needs_a_rotating_joint_and_keys_map() -> None:
    from evolab.app.play.window import key_bindings

    bad = copy.deepcopy(MOTOR_ENV)
    bad["joints"][0]["kind"] = "weld"
    with pytest.raises(EnvFileError):
        compile_env(bad)
    kb = key_bindings(compile_env(MOTOR_ENV))
    assert sorted(b.value for b in kb) == [-1.0, 1.0]


def test_hills_are_deterministic_flat_at_start_and_rise() -> None:
    a = generate_hills(5, x_min=-20.0, x_max=400.0, segment=1.0, slope=0.06, growth=0.03, flat_start=25.0)
    b = generate_hills(5, x_min=-20.0, x_max=400.0, segment=1.0, slope=0.06, growth=0.03, flat_start=25.0)
    assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()
    xs, ys = a
    assert np.all(ys[xs <= 5.0] == 0.0)
    assert ys[xs > 300].mean() > 10.0  # 오르막
    c = generate_hills(6, x_min=-20.0, x_max=400.0, segment=1.0)
    assert c[1].tobytes() != a[1].tobytes()


def test_hill_car_example() -> None:
    spec = load_env(paths.builtin_env_dir() / "hill_car.evoenv")
    layout = layout_for(spec)
    assert layout.n_body == 5 and layout.dim == 5 + 11 * 16 + 16 + 16 * 2 + 2
    body = layout.body_values(layout.initial()[None, :])[0]
    ep = Episode(spec, body, Trial(3))
    for _ in range(600):
        ep.step([-1.0, -1.0])
    assert ep.snapshot().poses[0, 0] > 20.0  # 바퀴를 시계 방향으로 돌리면 오른쪽으로 간다

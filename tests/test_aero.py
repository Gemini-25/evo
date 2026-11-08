"""v2.0 공기역학(설계 4.5, D14·D15, T15·T16·T22·T23): 평판 계수, 받는 변, 바람 0 수동성, 한 스텝 상한."""

from __future__ import annotations

import itertools
import json
import math
from typing import Any

import numpy as np
import pytest

from evolab import paths
from evolab.engine import seeds
from evolab.policy.genome import layout_for
from evolab.spec.io import compile_env
from evolab.world.physics2d.episode import Episode, Trial

GLIDER_TEXT = (paths.builtin_env_dir() / "glider.evoenv").read_text(encoding="utf-8")
CHAMPIONS = [
    # 설계 원형의 120초 판 글라이더 학습(시드 1·2·3, 6천 회) 마지막 챔피언 유전자(final/runs/gliderT_s*/gen 63)
    [-0.8936761625966062, 0.7220583545774123, -0.3016982634671899, -0.543682995759927, -0.9807046974885023,
     0.9512479285094804, 0.20107579570797318, 0.02737360515322257],
    [0.15384888540814717, 0.9973306521418275, -0.3103235564308366, -0.5058435579634832, -0.9998783649127481,
     0.7356034940982833, 0.15299052251326978, 0.44088215707733497],
    [0.9557101440572817, 0.9958328113605847, -0.25854995879760995, -0.5922356213569221, -0.9736434284533184,
     0.9070238358611461, 0.17328455474166365, 0.6497271028545876],
]


def one_body(shapes: list[dict[str, Any]], *, angle: float = 0.0, vx: float = 0.0, wind: float = 0.0,
             aero: dict[str, Any] | None = None, density: float = 500.0) -> Any:
    data = {
        "name": "공기역학 시험",
        "world": {"gravity": [0.0, 0.0], "wind": [wind, 0.0], "max_time": 5.0, "bounds": [-1000, -1000, 1000, 1000]},
        "agent": {"spawn": {"angle": angle, "vx": vx}},
        "bodies": [{"id": "plate", "shapes": shapes, "material": {"density": density}}],
        "actuators": [{"kind": "aero", "id": "air", "body": "plate", **(aero or {})}],
        "scoring": {"blocks": [{"id": "t", "kind": "survival_time"}]},
    }
    return compile_env(json.dumps(data))


DIAMOND = [{"kind": "poly", "points": [[0.5, 0.0], [0.0, 0.02], [-0.5, 0.0], [0.0, -0.02]]}]
"""마름모 판: 시위 1m, 두께 4%."""


def coefficients(deg: float, speed: float = 10.0) -> tuple[float, float]:
    """받음각 deg로 speed m/s 나는 마름모 판의 (CL, CD): 한 스텝 속도 변화 x 질량 / dt를 ½ρU² x 시위 x 폭으로 나눈 값."""
    ep = Episode(one_body(DIAMOND, angle=math.radians(deg), vx=speed), None, Trial(1))
    body = ep.bodies["plate"]
    v0 = body.velocity
    ep.step([])
    v1 = body.velocity
    q = 0.5 * 1.2 * speed * speed * 1.0 * 1.0
    fx, fy = (v1.x - v0.x) * body.mass / ep.dt, (v1.y - v0.y) * body.mass / ep.dt
    assert ep.aero_clamped == 0
    return fy / q, -fx / q


def test_flat_plate_coefficients() -> None:
    """T15: 마름모 판(4%, AR 6) 6°에서 CL 0.44~0.52·CD 0.03~0.045, 12° CL 0.70~0.82, 90° CD 1.15~1.30, 0° 양력 0."""
    cl, cd = coefficients(6.0)
    assert 0.44 <= cl <= 0.52 and 0.03 <= cd <= 0.045
    assert cl / cd > 12.0  # 원형 측정 L/D 13.2
    cl12, _ = coefficients(12.0)
    assert 0.70 <= cl12 <= 0.82
    cl90, cd90 = coefficients(90.0)
    assert 1.15 <= cd90 <= 1.30 and abs(cl90) < 1e-6
    cl0, cd0 = coefficients(0.0)
    assert abs(cl0) < 1e-12 and cd0 > 0  # 대칭이라 양력 0, 표면 마찰만
    clm, cdm = coefficients(-6.0)
    assert clm == pytest.approx(-cl, abs=1e-9) and cdm == pytest.approx(cd, abs=1e-9)


def test_no_relative_wind_means_no_force() -> None:
    """상대 바람 0: 바람과 같은 속도로 움직이거나 둘 다 멈추면 힘이 없다."""
    for vx, wind in ((0.0, 0.0), (4.0, 4.0)):
        ep = Episode(one_body(DIAMOND, angle=0.3, vx=vx, wind=wind), None, Trial(1))
        v0 = ep.bodies["plate"].velocity
        ep.step([])
        v1 = ep.bodies["plate"].velocity
        assert (v1.x, v1.y) == (v0.x, v0.y) and ep.bodies["plate"].angular_velocity == 0.0


def test_wind_pushes_a_plate_at_rest() -> None:
    ep = Episode(one_body(DIAMOND, angle=math.pi / 2, wind=5.0), None, Trial(1))
    ep.step([])
    assert ep.bodies["plate"].velocity.x > 0  # 바람 쪽으로 밀린다


def test_edges_follow_surfaces_and_skip_inner_edges() -> None:
    """T16: all이면 원·선을 뺀 모든 모양 테두리, wings면 날개 모양만, 볼록 분해 안쪽 변은 세지 않는다."""
    shapes = [
        {"kind": "box", "w": 1.0, "h": 0.1},  # 4
        {"kind": "poly", "points": [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], "density": 1.0},  # 오목: 6
        {"kind": "radial", "radii": [0.3] * 8},  # 8(삼각형 조각 8개의 안쪽 변은 빼고)
        {"kind": "wing", "le": [0.5, 0.2], "chord": 0.8, "thick": 0.04},  # 4
        {"kind": "circle", "r": 0.2},  # 0
        {"kind": "chain", "points": [[0, 0], [1, 1], [2, 0]]},  # 0
    ]
    all_ep = Episode(one_body(shapes), None, Trial(1))
    assert all_ep.aero_edge_count == 4 + 6 + 8 + 4
    wings_ep = Episode(one_body(shapes, aero={"surfaces": "wings"}), None, Trial(1))
    assert wings_ep.aero_edge_count == 4
    glider = Episode(compile_env(GLIDER_TEXT), None, Trial(1))
    assert glider.aero_edge_count == 4 + 4 + 4  # 동체 막대, 주 날개, 꼬리날개(추는 원)


def energy(body: Any) -> float:
    v = body.velocity
    cog_y = body.local_to_world(body.center_of_gravity).y
    return 0.5 * body.mass * (v.x * v.x + v.y * v.y) + 0.5 * body.moment * body.angular_velocity ** 2 + body.mass * 9.81 * cog_y


def test_aero_is_passive_without_wind() -> None:
    """T22: 바람을 0으로 고정하면 글라이더 몸 수치 범위 전체(무작위 160 + 범위 끝 조합 40)에서 NaN 0, 폭주 0,
    에너지가 1% 넘게 늘어나는 몸 0(힘 거는 점과 바람 재는 점이 같아 공기힘이 일을 빼앗기만 한다)."""
    spec = compile_env(GLIDER_TEXT)
    layout = layout_for(spec)
    rng = np.random.default_rng(3)
    corners = [np.array(c, dtype=float) for c in itertools.product((-1.0, 1.0), repeat=layout.n_body)]
    bodies = [rng.uniform(-1, 1, layout.n_body) for _ in range(160)]
    bodies += [corners[int(i)] for i in rng.choice(len(corners), 40, replace=False)]
    trial = seeds.judge_trials(1)[0]
    bad: list[tuple[int, str]] = []
    for k, genes in enumerate(bodies):
        ep = Episode(spec, layout.body_values(genes), trial)
        ep.wind = (0.0, 0.0)
        body = ep.bodies["glider"]
        e0 = emax = energy(body)
        while not ep.done:
            ep.step([])
            v, w = body.velocity, body.angular_velocity
            if not (math.isfinite(v.x) and math.isfinite(v.y) and math.isfinite(w)):
                bad.append((k, "nan"))
                break
            if abs(v.x) + abs(v.y) > 60 or abs(w) > 60:
                bad.append((k, "blow-up"))
                break
            emax = max(emax, energy(body))
        if emax > e0 * 1.01:
            bad.append((k, f"energy +{100 * (emax / e0 - 1):.2f}%"))
    assert bad == []


def fly(spec: Any, body_values: Any, trial: Trial, cap: int | None = None) -> tuple[int, int, float, float]:
    """(상한 발동 스텝, 스텝, 최대 |v|, 최대 |ω|)."""
    ep = Episode(spec, body_values, trial)
    body = ep.bodies["glider"]
    vmax = wmax = 0.0
    while not ep.done and (cap is None or ep.steps < cap):
        ep.step([])
        v, w = body.velocity, body.angular_velocity
        assert math.isfinite(v.x) and math.isfinite(v.y) and math.isfinite(w)
        vmax, wmax = max(vmax, math.hypot(v.x, v.y)), max(wmax, abs(w))
    return ep.aero_clamped, ep.steps, vmax, wmax


def normal_bodies(layout: Any, randoms: int) -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    return [layout.initial(), *(np.array(c) for c in CHAMPIONS)] + [rng.uniform(-1, 1, layout.dim) for _ in range(randoms)]


def test_clamp_never_acts_on_the_builtin_glider() -> None:
    """T23 (빠른 판): 처음 몸·챔피언 셋·무작위 몸 20개 x 판정 시드 5개에서 상한 발동 0. 시드 20개 판은 느린 테스트."""
    spec = compile_env(GLIDER_TEXT)
    layout = layout_for(spec)
    total = 0
    for genes in normal_bodies(layout, 20):
        for trial in seeds.judge_trials(5):
            total += fly(spec, layout.body_values(genes), trial)[0]
    assert total == 0


@pytest.mark.slow
def test_clamp_never_acts_on_the_builtin_glider_judge_20() -> None:
    """T23: 처음 몸·저장된 챔피언·무작위 몸 20개의 판정 시드 20개에서 aero_clamped == 0."""
    spec = compile_env(GLIDER_TEXT)
    layout = layout_for(spec)
    total = steps = 0
    for genes in normal_bodies(layout, 20):
        for trial in seeds.judge_trials(20):
            c, n, _, _ = fly(spec, layout.body_values(genes), trial)
            total += c
            steps += n
    assert total == 0 and steps > 100_000


def test_clamp_keeps_extreme_values_finite() -> None:
    """T23: 폭 30·공기 밀도 10·재질 밀도 0.5(상한이 없으면 몇 스텝 만에 10⁸m/s 넘게 폭주)에서 3,600스텝 동안 유한하고
    속력·각속도 < 10³, 상한이 걸린다."""
    data = json.loads(GLIDER_TEXT)
    data["actuators"][0].update({"span": 30.0, "density": 10.0})
    data["bodies"][1]["material"]["density"] = 0.5
    data["world"]["max_time"] = 60.0
    spec = compile_env(json.dumps(data))
    layout = layout_for(spec)
    clamped, steps, vmax, wmax = fly(spec, layout.body_values(layout.initial()), seeds.judge_trials(1)[0], cap=3600)
    assert clamped > 0 and steps > 10
    assert vmax < 1e3 and wmax < 1e3

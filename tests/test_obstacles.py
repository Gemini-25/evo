"""v2.0 장애물 줄 생성기(설계 4.2, D9, T11): 결정성, 초 단위 간격, 공중 장애물 태그, 줄 길이, 멈추는 줄."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from typing import Any

import pytest

from evolab import paths
from evolab.generators.obstacles import LENGTH_TAIL, MIN_GAP, generate_obstacles, place_boxes, scrolled_distance
from evolab.spec.compiled import EnvFileError
from evolab.spec.io import compile_env
from evolab.world.physics2d.episode import Episode, Trial
from evolab.world.physics2d.motion import pose_at

JUMP_TEXT = (paths.builtin_env_dir() / "jump.evoenv").read_text(encoding="utf-8")
ROW = dict(count=40, x=20.0, y=0.0, gap=(1.0, 2.4), width=(0.5, 1.5), height=(0.4, 1.3), air_chance=0.35,
           air_y=(1.25, 1.5), air_height=(0.4, 0.8), speed=6.0, accel=0.25)


def arrival(x: float, speed: float, accel: float) -> float:
    """scrolled_distance(t) = x인 t(닫힌 식, accel > 0)."""
    return (-speed + math.sqrt(speed * speed + 2.0 * accel * x)) / accel


def test_same_seed_same_row_in_another_process() -> None:
    row = generate_obstacles(123456789, **ROW)  # type: ignore[arg-type]
    code = ("from evolab.generators.obstacles import generate_obstacles; "
            f"r = generate_obstacles(123456789, **{ROW!r}); print(repr((r.boxes, r.air, r.length)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == repr((row.boxes, row.air, row.length))
    assert generate_obstacles(123456789, **ROW) == row  # type: ignore[arg-type]
    assert generate_obstacles(987654321, **ROW) != row  # type: ignore[arg-type]


def test_gaps_are_seconds_between_arrivals() -> None:
    """이웃 장애물이 x = 0에 닿는 시각 차 ≥ gap 최솟값(빨라져도 시간 간격이 남는다)."""
    for seed in range(20):
        row = generate_obstacles(seed, **ROW)  # type: ignore[arg-type]
        assert len(row.boxes) == 40
        assert row.boxes[0][0] == 20.0  # 첫 장애물은 x에서 출발
        times = [arrival(b[0], 6.0, 0.25) for b in row.boxes]
        gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
        assert min(gaps) >= 1.0 - 1e-6 and max(gaps) <= 2.4 + 1e-6
        dist = [b[0] - a[0] for a, b in zip(row.boxes, row.boxes[1:], strict=False)]
        assert dist[-1] > dist[0]  # 같은 시간 간격이라도 줄이 빨라지면 거리 간격이 커진다


def test_air_obstacles_and_sizes() -> None:
    row = generate_obstacles(7, **ROW)  # type: ignore[arg-type]
    assert 0 < sum(row.air) < len(row.air)
    for (x0, y0, x1, y1), air in zip(row.boxes, row.air, strict=True):
        assert 0.5 <= x1 - x0 <= 1.5
        if air:
            assert 1.25 <= y0 <= 1.5 and 0.4 <= y1 - y0 <= 0.8
        else:
            assert y0 == 0.0 and 0.4 <= y1 - y0 <= 1.3
    assert row.length == pytest.approx(row.boxes[-1][2] + LENGTH_TAIL)
    always = generate_obstacles(7, **{**ROW, "air_chance": 1.0})  # type: ignore[arg-type]
    never = generate_obstacles(7, **{**ROW, "air_chance": 0.0})  # type: ignore[arg-type]
    assert all(always.air) and not any(never.air)


def test_ranges_in_any_order_and_minimum_gap() -> None:
    a = generate_obstacles(3, **{**ROW, "gap": (2.4, 1.0)})  # type: ignore[arg-type]
    b = generate_obstacles(3, **ROW)  # type: ignore[arg-type]
    assert a == b
    tiny = generate_obstacles(3, **{**ROW, "gap": (0.0, 0.0), "accel": 0.0})  # type: ignore[arg-type]
    xs = [box[0] for box in tiny.boxes]
    assert all(q - p == pytest.approx(MIN_GAP * 6.0) for p, q in zip(xs, xs[1:], strict=False))


def test_slowing_row_stops_and_keeps_unreachable_obstacles_out() -> None:
    """가속이 음수면 줄이 멈춘다. 멈추기 전에 선수 자리에 닿을 수 없는 장애물은 만들지 않는다."""
    row = generate_obstacles(5, **{**ROW, "accel": -0.5})  # type: ignore[arg-type]
    t_stop = 6.0 / 0.5
    assert 1 < len(row.boxes) < 40
    for x0, *_ in row.boxes[1:]:
        assert x0 < scrolled_distance(t_stop, 6.0, -0.5)
    assert scrolled_distance(t_stop + 5.0, 6.0, -0.5) == scrolled_distance(t_stop, 6.0, -0.5)
    still = generate_obstacles(5, **{**ROW, "speed": 0.0, "accel": 0.0})  # type: ignore[arg-type]
    assert len(still.boxes) == 1 and still.boxes[0][0] == 20.0  # 움직이지 않는 줄: 첫 장애물만


def jump_data() -> dict[str, Any]:
    return json.loads(JUMP_TEXT)


def test_episode_builds_one_moving_body_with_tags_and_values() -> None:
    spec = compile_env(JUMP_TEXT)
    ep = Episode(spec, None, Trial(11))
    geo = next(g for g in ep.geometry().bodies if g.id == "obs")
    assert geo.owner == "world" and not geo.moving and geo.tags == ["obstacle"] and geo.motion is not None
    assert len(geo.shapes) == 40
    shapes = list(ep.bodies["obs"].shapes)
    tags = [ep._shape_info[s][1] for s in shapes]
    assert all("obstacle" in t for t in tags) and any("air" in t for t in tags)
    assert ep.gen_values["gen.obs.speed"] == 6.0 and ep.gen_values["gen.obs.accel"] == 0.25
    assert ep.gen_values["gen.obs.length"] == pytest.approx(geo.motion.cum[-1])
    # 줄 전체가 같은 물체라 시험 시드(r.oseed)만 같으면 같은 줄
    again = next(g for g in Episode(spec, None, Trial(11)).geometry().bodies if g.id == "obs")
    other = next(g for g in Episode(spec, None, Trial(12)).geometry().bodies if g.id == "obs")
    assert [s.points for s in again.shapes] == [s.points for s in geo.shapes]
    assert [s.points for s in other.shapes] != [s.points for s in geo.shapes]


def test_row_moves_left_with_acceleration_and_stops_at_the_end() -> None:
    data = jump_data()
    data["generators"][0]["count"] = 2
    spec = compile_env(json.dumps(data))
    ep = Episode(spec, None, Trial(1))
    plan = next(g.motion for g in ep.geometry().bodies if g.id == "obs")
    length = plan.cum[-1]
    t_end = arrival(length, 6.0, 0.25)
    assert pose_at(plan, t_end + 1.0)[0] == pytest.approx(-length)
    assert pose_at(plan, 1.0)[0] == pytest.approx(-(6.0 + 0.125))


def test_obstacle_seed_must_be_a_world_value() -> None:
    data = jump_data()
    data["params"] = [{"id": "k", "value": 1.0}]
    data["generators"][0]["seed"] = "=p.k"  # 생성기 단계는 r.*만: 개체마다 다른 줄이 되면 안 된다
    with pytest.raises(EnvFileError):
        compile_env(json.dumps(data))


def test_place_boxes_keeps_direction_pi_exact_and_mirrors_direction_zero() -> None:
    row = generate_obstacles(31, **ROW)  # type: ignore[arg-type]
    assert place_boxes(row, math.pi) == row.boxes  # 내장 점프 게임: 줄 좌표 그대로(sin π만큼도 옮기지 않음)
    assert place_boxes(row, 0.0) == [(-x1, y0, -x0, y1) for x0, y0, x1, y1 in row.boxes]  # 왼쪽에서 옴: 좌우 뒤집기


@pytest.mark.parametrize("direction", [math.pi, 0.0, 3.14, math.pi - 0.3, 0.2, -math.pi])
def test_every_obstacle_passes_the_runner_spot_at_floor_height_in_any_direction(direction: float) -> None:
    """미는 방향이 π가 아니어도 장애물마다 앞 모서리가 줄 좌표 거리 s만큼 밀린 시각에 바닥 높이 그대로 선수 자리(x = 0)를
    지난다. 전에는 줄 좌표를 방향 π로만 놓아 다른 방향이면 첫 장애물 말고는 선수 자리에 오지 않았다(검토 측정 0~1/6)."""
    data = jump_data()
    data["generators"][0].update({"direction": direction, "count": 6})
    spec = compile_env(json.dumps(data))
    assert [i for i in spec.issues if i.severity != "info"] == []  # x = 20: 어느 방향이든 선수에서 20m
    ep = Episode(spec, None, Trial(7))
    geo = next(g for g in ep.geometry().bodies if g.id == "obs")
    assert geo.motion is not None and len(geo.shapes) == 6
    ux = -math.cos(direction)  # 장애물이 오는 쪽의 x 성분(양수면 오른쪽에서 옴)
    for shape in geo.shapes:
        xs, ys = [p[0] for p in shape.points], [p[1] for p in shape.points]
        lead = min(xs) if ux >= 0.0 else max(xs)  # 앞 모서리
        s = lead / ux  # 선수 자리까지 줄을 따라 남은 거리
        assert s >= 20.0 - 1e-9
        px, py, _ = pose_at(geo.motion, arrival(s, 6.0, 0.25))
        assert lead + px == pytest.approx(0.0, abs=1e-6)
        floor = min(ys) + py
        assert floor == pytest.approx(0.0, abs=1e-6) or 1.25 - 1e-6 <= floor <= 1.5 + 1e-6  # 땅 또는 공중 장애물

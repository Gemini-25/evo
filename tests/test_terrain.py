"""지형 자동 생성기 (generators/terrain.py)."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import pickle
import subprocess
import sys
import time
import tracemalloc
import warnings

import numpy as np
import pytest

from evolab.app import strings as S
from evolab.generators import terrain as T
from evolab.generators.terrain import Heightmap, generate_heightmap

WORLD = {"x_min": 0.0, "x_max": 140.0}


def _digest(h: Heightmap) -> str:
    return hashlib.sha256(h.xs.tobytes() + h.ys.tobytes() + repr(h.pad).encode()).hexdigest()


def _slopes(h: Heightmap) -> np.ndarray:
    return np.diff(h.ys) / np.diff(h.xs)


def _random_pad_cases(n: int, base_seed: int) -> list[tuple[int, float, float, float]]:
    """(지형 시드, 거칠기, 패드 위치, 패드 폭). 지형 밖 위치도 섞어서 옮기기도 함께 시험한다."""
    rng = np.random.default_rng(base_seed)
    return [
        (s, float(rng.uniform(0.0, 1.0)), float(rng.uniform(-20.0, 160.0)), float(rng.uniform(2.0, 14.0)))
        for s in range(n)
    ]


# ---- 결정성 ----


def test_same_seed_gives_bit_identical_terrain() -> None:
    for kwargs in ({}, {"pad_x": 61.3, "pad_width": 7.7}):
        a = generate_heightmap(11, **WORLD, **kwargs)
        for s in range(5):  # 다른 시드를 사이에 돌려도 상태가 남지 않는다
            generate_heightmap(s, **WORLD, **kwargs)
        b = generate_heightmap(11, **WORLD, **kwargs)
        assert a.xs.tobytes() == b.xs.tobytes()
        assert a.ys.tobytes() == b.ys.tobytes()
        assert a.pad == b.pad
        assert a == b and hash(a) == hash(b)


def test_different_seeds_differ() -> None:
    maps = [generate_heightmap(s, **WORLD) for s in range(20)]
    digests = {_digest(h) for h in maps}
    assert len(digests) == len(maps)
    assert not np.array_equal(maps[0].ys, maps[1].ys)


def test_same_terrain_in_another_process() -> None:
    # 해시 시드가 다른 새 프로세스에서도 비트까지 같아야 한다.
    code = (
        "import hashlib\n"
        "from evolab.generators.terrain import generate_heightmap\n"
        "for kw in ({}, {'pad_x': 61.3, 'pad_width': 7.7}):\n"
        "    h = generate_heightmap(11, x_min=0.0, x_max=140.0, roughness=0.8, **kw)\n"
        "    print(hashlib.sha256(h.xs.tobytes() + h.ys.tobytes() + repr(h.pad).encode()).hexdigest())\n"
    )
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    here = [
        _digest(generate_heightmap(11, **WORLD, roughness=0.8)),
        _digest(generate_heightmap(11, **WORLD, roughness=0.8, pad_x=61.3, pad_width=7.7)),
    ]
    assert out.stdout.split() == here


def test_pad_does_not_change_terrain_far_away() -> None:
    # 패드는 난수를 뽑지 않는다. 포락선이 진폭 띠 전체를 덮는 거리(2·진폭/0.5 + 여유/2)부터는 원래 지형이다.
    plain = generate_heightmap(5, **WORLD)
    padded = generate_heightmap(5, **WORLD, pad_x=30.0, pad_width=8.0)
    left, right, _ = padded.pad or (0.0, 0.0, 0.0)
    far = right + 2.0 * 4.0 / 0.5 + 1.0 + 1e-9
    keep_plain = plain.xs >= far
    keep_padded = padded.xs >= far
    assert keep_plain.sum() > 50
    assert np.array_equal(plain.xs[keep_plain], padded.xs[keep_padded])
    assert np.array_equal(plain.ys[keep_plain], padded.ys[keep_padded])


# ---- 모양 ----


@pytest.mark.parametrize("segment", [1.0, 0.5, 0.7, 3.0, 200.0])
def test_grid_covers_world(segment: float) -> None:
    h = generate_heightmap(3, x_min=-12.5, x_max=127.5, segment=segment)
    assert h.xs.dtype == np.float64 and h.ys.dtype == np.float64
    assert h.xs.shape == h.ys.shape
    assert h.xs[0] == -12.5 and h.xs[-1] == 127.5
    assert np.all(np.diff(h.xs) > 0.0)
    assert np.all(np.diff(h.xs) <= segment * (1.0 + 1e-12))
    assert h.pad is None


def test_default_world_has_one_metre_segments() -> None:
    h = generate_heightmap(0, **WORLD)
    assert h.xs.size == 141
    assert np.allclose(np.diff(h.xs), 1.0)


def test_arrays_are_read_only() -> None:
    h = generate_heightmap(0, **WORLD)
    with pytest.raises(ValueError):
        h.ys[0] = 1.0
    with pytest.raises(ValueError):
        h.xs[0] = 1.0


@pytest.mark.parametrize("roughness", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(("base_y", "amplitude"), [(0.0, 4.0), (-3.25, 1.5), (100.0, 10.0)])
def test_heights_stay_in_amplitude_band(roughness: float, base_y: float, amplitude: float) -> None:
    low, high = base_y - amplitude, base_y + amplitude
    for seed in range(40):
        h = generate_heightmap(seed, **WORLD, base_y=base_y, amplitude=amplitude, roughness=roughness)
        assert h.ys.min() >= low and h.ys.max() <= high
        # 패드가 없으면 기준 높이에서 가장 먼 점이 정확히 진폭만큼 떨어져 있다.
        assert np.max(np.abs(h.ys - base_y)) == pytest.approx(amplitude, rel=1e-12)
        p = generate_heightmap(seed, **WORLD, base_y=base_y, amplitude=amplitude, roughness=roughness,
                               pad_x=20.0 + 3.0 * seed, pad_width=8.0)
        assert p.ys.min() >= low and p.ys.max() <= high
        assert p.pad is not None and low <= p.pad[2] <= high


def test_zero_amplitude_is_flat() -> None:
    h = generate_heightmap(9, **WORLD, base_y=2.5, amplitude=0.0, pad_x=70.0, pad_width=10.0)
    assert np.all(h.ys == 2.5)
    assert h.pad == (65.0, 75.0, 2.5)


def test_roughness_changes_jaggedness() -> None:
    def mean_abs_second_difference(roughness: float) -> float:
        return float(np.mean([np.mean(np.abs(np.diff(generate_heightmap(s, **WORLD, roughness=roughness).ys, 2)))
                              for s in range(30)]))

    smooth = mean_abs_second_difference(0.0)
    middle = mean_abs_second_difference(0.5)
    rough = mean_abs_second_difference(1.0)
    assert smooth < middle < rough
    assert rough > 3.0 * smooth


def test_smaller_segment_adds_detail_but_keeps_big_features() -> None:
    # 뽑는 순서가 큰 특징부터라서, 구간을 줄이면 짧은 옥타브만 더해진다.
    coarse = generate_heightmap(4, **WORLD, segment=1.0)
    fine = generate_heightmap(4, **WORLD, segment=0.5)
    assert fine.xs.size == 281
    assert np.corrcoef(coarse.ys, fine.ys[::2])[0, 1] > 0.9


# ---- 착륙 패드 ----


def test_pad_is_exactly_flat_and_edges_are_vertices() -> None:
    for seed, roughness, pad_x, pad_width in _random_pad_cases(200, base_seed=1):
        h = generate_heightmap(seed, **WORLD, roughness=roughness, pad_x=pad_x, pad_width=pad_width)
        assert h.pad is not None
        left, right, y = h.pad
        assert right - left == pytest.approx(pad_width, rel=1e-12)
        i = int(np.searchsorted(h.xs, left))
        j = int(np.searchsorted(h.xs, right))
        assert h.xs[i] == left and h.xs[j] == right
        assert np.all(h.ys[i : j + 1] == y)
        assert j == i + 1  # 패드 안쪽 꼭짓점은 지운다: 패드는 선분 하나
        assert np.all(np.diff(h.xs) > 0.0)
        assert h.xs[0] == 0.0 and h.xs[-1] == 140.0
        for x in np.linspace(left, right, 17):
            assert h.height_at(float(x)) == y


def test_pad_height_is_terrain_height_at_pad_centre() -> None:
    for seed, roughness, pad_x, pad_width in _random_pad_cases(50, base_seed=2):
        plain = generate_heightmap(seed, **WORLD, roughness=roughness)
        h = generate_heightmap(seed, **WORLD, roughness=roughness, pad_x=pad_x, pad_width=pad_width)
        assert h.pad is not None
        left, right, y = h.pad
        assert y == plain.height_at(0.5 * (left + right))


def test_pad_position_is_kept_when_it_fits() -> None:
    h = generate_heightmap(1, **WORLD, pad_x=42.5, pad_width=9.0)
    assert h.pad is not None
    assert h.pad[:2] == (38.0, 47.0)


@pytest.mark.parametrize("pad_margin", [0.0, 2.0, 5.0])
def test_pad_is_clamped_inside_world_with_margin(pad_margin: float) -> None:
    for pad_x, expected_left in [(-50.0, pad_margin), (0.0, pad_margin), (3.0, pad_margin),
                                 (139.0, 140.0 - pad_margin - 8.0), (1e6, 140.0 - pad_margin - 8.0)]:
        h = generate_heightmap(2, **WORLD, pad_x=pad_x, pad_width=8.0, pad_margin=pad_margin)
        assert h.pad is not None
        left, right, _ = h.pad
        assert left == pytest.approx(expected_left, abs=1e-12)
        assert right - left == pytest.approx(8.0, abs=1e-12)
        assert left - pad_margin >= -1e-12 and right + pad_margin <= 140.0 + 1e-12


def test_pad_exactly_as_wide_as_allowed() -> None:
    h = generate_heightmap(2, **WORLD, pad_x=10.0, pad_width=136.0, pad_margin=2.0)
    assert h.pad is not None
    assert h.pad[:2] == pytest.approx((2.0, 138.0))
    whole = generate_heightmap(2, **WORLD, pad_x=10.0, pad_width=140.0, pad_margin=0.0)
    assert whole.pad == (0.0, 140.0, whole.ys[0])
    assert whole.xs.tolist() == [0.0, 140.0]


def test_terrain_meets_pad_without_cliffs() -> None:
    # 기본값(여유 2m, 구간 1m)에서 보장하는 값:
    # - 패드에 붙은 구간: 기울기 <= 0.5 · 1 / (2 · 2) = 0.125 (패드에서 거의 수평으로 떠남)
    # - 패드 양옆 2m 안의 구간: 기울기 <= (0.5 / (2 · 2)) · ((2 - 1)² + 2²) / 1 = 0.625 (약 32도)
    worst_adjacent = 0.0
    worst_margin = 0.0
    for seed, roughness, pad_x, pad_width in _random_pad_cases(300, base_seed=3):
        h = generate_heightmap(seed, **WORLD, roughness=roughness, pad_x=pad_x, pad_width=pad_width)
        assert h.pad is not None
        left, right, _ = h.pad
        slopes = np.abs(_slopes(h))
        a, b = h.xs[:-1], h.xs[1:]
        i = int(np.searchsorted(h.xs, left))
        j = int(np.searchsorted(h.xs, right))
        adjacent = ([slopes[i - 1]] if i > 0 else []) + ([slopes[j]] if j < slopes.size else [])
        worst_adjacent = max([worst_adjacent, *adjacent])
        in_margin = ((a >= left - 2.0) & (b <= left)) | ((a >= right) & (b <= right + 2.0))
        worst_margin = max(worst_margin, float(slopes[in_margin].max()))
    assert worst_adjacent <= 0.125 + 1e-9
    assert worst_margin <= 0.625 + 1e-9


def test_pad_flattening_tames_rough_terrain_near_pad() -> None:
    # 거친 지형은 원래 그 자리에서 0.625보다 훨씬 가파르기도 하지만, 패드를 놓으면 양옆 2m가 잔잔해진다.
    plain_worst, padded_worst = 0.0, 0.0
    for seed in range(40):
        plain = generate_heightmap(seed, **WORLD, roughness=1.0)
        h = generate_heightmap(seed, **WORLD, roughness=1.0, pad_x=70.0, pad_width=8.0)
        for m in (plain, h):
            a, b = m.xs[:-1], m.xs[1:]
            near = ((a >= 64.0) & (b <= 66.0)) | ((a >= 74.0) & (b <= 76.0))
            worst = float(np.abs(_slopes(m))[near].max())
            if m is plain:
                plain_worst = max(plain_worst, worst)
            else:
                padded_worst = max(padded_worst, worst)
    assert plain_worst > 1.0
    assert padded_worst <= 0.625 + 1e-9


def test_no_pad_requested_means_no_pad() -> None:
    h = generate_heightmap(3, **WORLD, pad_width=8.0)
    assert h.pad is None
    assert h == generate_heightmap(3, **WORLD)


# ---- height_at ----


def test_height_at_interpolates_and_clamps() -> None:
    h = generate_heightmap(6, **WORLD, pad_x=100.0, pad_width=6.0)
    for x, y in zip(h.xs.tolist(), h.ys.tolist(), strict=True):
        assert h.height_at(x) == y
    for k in range(h.xs.size - 1):
        mid = 0.5 * (h.xs[k] + h.xs[k + 1])
        assert h.height_at(float(mid)) == pytest.approx(0.5 * (h.ys[k] + h.ys[k + 1]), abs=1e-12)
        quarter = h.xs[k] + 0.25 * (h.xs[k + 1] - h.xs[k])
        assert h.height_at(float(quarter)) == pytest.approx(0.75 * h.ys[k] + 0.25 * h.ys[k + 1], abs=1e-12)
    assert h.height_at(-5.0) == h.ys[0]
    assert h.height_at(1e9) == h.ys[-1]
    assert isinstance(h.height_at(12.3), float)


def test_heightmap_from_hand_drawn_points() -> None:
    h = Heightmap(xs=[0, 1, 3, 4], ys=[0, 2, 2, 1], pad=(1, 3, 2))
    assert h.pad == (1.0, 3.0, 2.0)
    assert h.xs.dtype == np.float64
    assert h.height_at(0.5) == 1.0
    assert h.height_at(2.0) == 2.0
    assert h.height_at(3.5) == 1.5
    assert h == Heightmap(np.array([0.0, 1.0, 3.0, 4.0]), np.array([0.0, 2.0, 2.0, 1.0]), (1.0, 3.0, 2.0))
    assert h != Heightmap(xs=[0, 1, 3, 4], ys=[0, 2, 2, 1], pad=None)
    # 넘긴 배열을 나중에 바꿔도 높이맵은 그대로다.
    ys = np.array([0.0, 1.0])
    g = Heightmap(np.array([0.0, 1.0]), ys, None)
    ys[0] = 9.0
    assert g.ys[0] == 0.0


@pytest.mark.parametrize(
    ("xs", "ys", "pad", "message"),
    [
        ([0.0], [0.0], None, S.TERRAIN_ERR_HEIGHTMAP_SHAPE.format(nx=(1,), ny=(1,))),
        ([0.0, 1.0], [0.0, 1.0, 2.0], None, S.TERRAIN_ERR_HEIGHTMAP_SHAPE.format(nx=(2,), ny=(3,))),
        ([0.0, float("nan")], [0.0, 1.0], None, S.TERRAIN_ERR_HEIGHTMAP_NOT_FINITE),
        ([0.0, 1.0], [0.0, float("inf")], None, S.TERRAIN_ERR_HEIGHTMAP_NOT_FINITE),
        ([0.0, 2.0, 2.0], [0.0, 1.0, 2.0], None, S.TERRAIN_ERR_HEIGHTMAP_ORDER.format(index=2)),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 1.0], (0.5, 2.0, 1.0), S.TERRAIN_ERR_HEIGHTMAP_PAD.format(pad=(0.5, 2.0, 1.0))),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 1.5], (1.0, 2.0, 1.0), S.TERRAIN_ERR_HEIGHTMAP_PAD.format(pad=(1.0, 2.0, 1.0))),
        ([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], (2.0, 1.0, 1.0), S.TERRAIN_ERR_HEIGHTMAP_PAD.format(pad=(2.0, 1.0, 1.0))),
        ([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], (1.0, 3.0, 1.0), S.TERRAIN_ERR_HEIGHTMAP_PAD.format(pad=(1.0, 3.0, 1.0))),
        ([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], (1.0, 2.0), S.TERRAIN_ERR_HEIGHTMAP_PAD.format(pad=(1.0, 2.0))),
    ],
)
def test_heightmap_rejects_bad_points(
    xs: list[float], ys: list[float], pad: tuple[float, ...] | None, message: str
) -> None:
    with pytest.raises(ValueError) as info:
        Heightmap(np.array(xs), np.array(ys), pad)
    assert str(info.value) == message


# ---- 인자 검사 ----


@pytest.mark.parametrize(
    ("seed", "kwargs", "message"),
    [
        (-1, {}, S.TERRAIN_ERR_SEED.format(value=-1)),
        (1.5, {}, S.TERRAIN_ERR_SEED.format(value=1.5)),
        (0, {"x_min": 5.0, "x_max": 5.0}, S.TERRAIN_ERR_RANGE.format(x_min=5.0, x_max=5.0)),
        (0, {"x_min": 5.0, "x_max": -5.0}, S.TERRAIN_ERR_RANGE.format(x_min=5.0, x_max=-5.0)),
        (0, {"x_min": float("-inf")}, S.TERRAIN_ERR_RANGE.format(x_min=float("-inf"), x_max=140.0)),
        (0, {"x_max": float("nan")}, S.TERRAIN_ERR_RANGE.format(x_min=0.0, x_max=float("nan"))),
        (0, {"segment": 0.0}, S.TERRAIN_ERR_SEGMENT.format(value=0.0)),
        (0, {"segment": -1.0}, S.TERRAIN_ERR_SEGMENT.format(value=-1.0)),
        (0, {"segment": float("nan")}, S.TERRAIN_ERR_SEGMENT.format(value=float("nan"))),
        (0, {"segment": 1e-5}, S.TERRAIN_ERR_TOO_MANY_VERTICES.format(count=14_000_001, limit=1_000_000)),
        (0, {"base_y": float("inf")}, S.TERRAIN_ERR_BASE_Y.format(value=float("inf"))),
        (0, {"amplitude": -0.1}, S.TERRAIN_ERR_AMPLITUDE.format(value=-0.1)),
        (0, {"amplitude": float("inf")}, S.TERRAIN_ERR_AMPLITUDE.format(value=float("inf"))),
        (0, {"roughness": -0.01}, S.TERRAIN_ERR_ROUGHNESS.format(value=-0.01)),
        (0, {"roughness": 1.01}, S.TERRAIN_ERR_ROUGHNESS.format(value=1.01)),
        (0, {"roughness": float("nan")}, S.TERRAIN_ERR_ROUGHNESS.format(value=float("nan"))),
        (0, {"pad_margin": -1.0}, S.TERRAIN_ERR_PAD_MARGIN.format(value=-1.0)),
        (0, {"pad_x": float("nan"), "pad_width": 8.0}, S.TERRAIN_ERR_PAD_X.format(value=float("nan"))),
        (0, {"pad_x": 70.0}, S.TERRAIN_ERR_PAD_WIDTH_MISSING),
        (0, {"pad_x": 70.0, "pad_width": 0.0}, S.TERRAIN_ERR_PAD_WIDTH.format(value=0.0)),
        (0, {"pad_x": 70.0, "pad_width": -3.0}, S.TERRAIN_ERR_PAD_WIDTH.format(value=-3.0)),
        (0, {"pad_x": 70.0, "pad_width": 1e-300}, S.TERRAIN_ERR_PAD_WIDTH.format(value=1e-300)),
        (0, {"pad_x": 70.0, "pad_width": 141.0, "pad_margin": 0.0},
         S.TERRAIN_ERR_PAD_TOO_WIDE.format(width=141.0, margin=0.0, world=140.0)),
        (0, {"pad_x": 70.0, "pad_width": 137.0},
         S.TERRAIN_ERR_PAD_TOO_WIDE.format(width=137.0, margin=2.0, world=140.0)),
        # 검토 보완: 폭 상한(구간이 길어도 분화구·격자점 난수가 폭에 비례), 높이 띠 넘침
        (0, {"x_max": 1e9, "segment": 1e7}, S.TERRAIN_ERR_TOO_WIDE.format(width=1e9, limit=1_000_000)),
        (0, {"x_min": -500_000.0, "x_max": 500_001.0, "segment": 100.0},
         S.TERRAIN_ERR_TOO_WIDE.format(width=1_000_001.0, limit=1_000_000)),
        (0, {"base_y": 1e308, "amplitude": 1e308}, S.TERRAIN_ERR_HEIGHT_RANGE.format(base_y=1e308, amplitude=1e308)),
        (0, {"amplitude": 1e308}, S.TERRAIN_ERR_HEIGHT_RANGE.format(base_y=0.0, amplitude=1e308)),
        (0, {"base_y": -1.7e308, "amplitude": 1e307},
         S.TERRAIN_ERR_HEIGHT_RANGE.format(base_y=-1.7e308, amplitude=1e307)),
    ],
)
def test_bad_arguments_raise_korean_messages(seed: float, kwargs: dict[str, float], message: str) -> None:
    args: dict[str, float] = {**WORLD, **kwargs}
    with pytest.raises(ValueError) as info:
        generate_heightmap(seed, **args)  # type: ignore[arg-type]  # 정수가 아닌 시드도 시험한다
    assert str(info.value) == message


@pytest.mark.parametrize("segment", [5e-324, 1e-300, 1e-200])
def test_absurdly_small_segment_is_a_short_value_error(segment: float) -> None:
    # 폭 / 구간이 inf가 되면 math.ceil이 OverflowError를 냈고, 아주 크면 수백 자리 정수를 문구에 넣었다.
    with pytest.raises(ValueError) as info:
        generate_heightmap(0, **WORLD, segment=segment)
    message = str(info.value)
    assert message.startswith(S.TERRAIN_ERR_TOO_MANY_VERTICES.split("{")[0])
    assert len(message) < 120


def test_valid_extreme_arguments_raise_no_numpy_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cases: list[dict[str, float]] = [
            {"amplitude": 4e307, "base_y": 0.0, "pad_x": 70.0, "pad_width": 8.0},
            {"amplitude": 1e300, "base_y": -5e307, "pad_x": 3.0, "pad_width": 8.0, "pad_margin": 0.0},
            {"pad_x": 70.0, "pad_width": 8.0, "pad_margin": 1e-300},
            {"x_min": 1e9, "x_max": 1e9 + 140.0, "pad_x": 1e9 + 70.0, "pad_width": 8.0},
        ]
        for kwargs in cases:
            args: dict[str, float] = {**WORLD, **kwargs}
            for seed in range(5):
                h = generate_heightmap(seed, **args)
                assert np.all(np.isfinite(h.ys))


def _dense_craters(seed: int, xs: np.ndarray, x_min: float, x_max: float, roughness: float) -> tuple[np.ndarray, int]:
    """비교 기준: 분화구마다 모든 꼭짓점에서 계산해 번호 순서대로 더하는 원래 방식. (합, 겹친 점 수)."""
    rng = np.random.default_rng(seed)
    most = math.floor((x_max - x_min) * T._CRATERS_PER_METRE + 0.5)
    count = int(rng.integers(most // 2, most, endpoint=True))
    centres = rng.uniform(x_min, x_max, count)
    u = rng.random(count)
    radii = T._CRATER_RADIUS_MIN + (T._CRATER_RADIUS_MAX - T._CRATER_RADIUS_MIN) * (u * u)
    depths = rng.uniform(T._CRATER_DEPTH_MIN, T._CRATER_DEPTH_MAX, count) * (radii / T._CRATER_RADIUS_MAX)
    depths *= 0.5 + 0.5 * roughness
    out = np.zeros_like(xs)
    touching = np.zeros(xs.size, dtype=np.intp)
    for c, r, d in zip(centres, radii, depths, strict=True):
        q = np.abs(xs - c) / r
        bowl = np.maximum(1.0 - q * q, 0.0)
        rr = (q - 1.0) * (1.0 / T._CRATER_RIM_HALF_WIDTH)
        rim = np.maximum(1.0 - rr * rr, 0.0)
        shape = d * (T._CRATER_RIM_HEIGHT * (rim * rim) - bowl * bowl)
        touching += shape != 0.0
        out += shape
    return out, int(np.sum(touching >= 2))


def test_local_craters_match_dense_sum_bit_for_bit() -> None:
    # 분화구는 자기 둘레의 꼭짓점만 계산한다(예전에는 분화구 수 × 꼭짓점 수). 결과는 비트까지 같아야 한다.
    overlapping = 0
    for x_min, x_max, n in [(0.0, 140.0, 141), (-70.0, 70.0, 381), (0.0, 600.0, 601), (1e7, 1e7 + 600.0, 1601),
                            (0.0, 90.0, 4), (-3.0, 300.0, 9001)]:
        xs = np.linspace(x_min, x_max, n)
        for seed in range(25):
            for roughness in (0.0, 0.8):
                expected, overlaps = _dense_craters(seed, xs, x_min, x_max, roughness)
                got = T._craters(np.random.default_rng(seed), xs, x_min, x_max, roughness)
                assert got.tobytes() == expected.tobytes()
                overlapping += overlaps
    assert overlapping > 100  # 겹친 분화구에서도 더하는 순서가 같다는 것을 실제로 시험했다


def test_hill_chunking_does_not_change_terrain(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs: dict[str, float] = {"x_min": -3.0, "x_max": 5000.0, "segment": 0.9, "pad_x": 2500.0, "pad_width": 8.0}
    whole = generate_heightmap(8, **kwargs)
    assert whole.xs.size > T._HILL_CHUNK  # 기본 크기로도 여러 덩어리
    monkeypatch.setattr(T, "_HILL_CHUNK", 7)  # 꼬리가 남는 작은 덩어리
    assert generate_heightmap(8, **kwargs) == whole
    monkeypatch.setattr(T, "_HILL_CHUNK", 10**9)  # 한 덩어리
    assert generate_heightmap(8, **kwargs) == whole


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x_min": 0.0, "x_max": 100_000.0},  # 넓은 지형: 예전에는 분화구 임시 배열만 약 2.7GB
        {"x_min": 0.0, "x_max": 140.0, "segment": 0.001},  # 잘게 나눈 지형: 예전에는 옥타브 15개 × 14만 점
        {"x_min": 0.0, "x_max": 1_000_000.0, "segment": 10.0, "pad_x": 5e5, "pad_width": 10.0},  # 폭 상한
    ],
)
def test_work_and_memory_stay_proportional_to_vertices(kwargs: dict[str, float]) -> None:
    tracemalloc.start()
    try:
        h = generate_heightmap(1, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    output_bytes = h.xs.nbytes + h.ys.nbytes
    assert peak < 40 * output_bytes + 8_000_000, f"peak {peak / 1e6:.1f} MB for {h.xs.size} vertices"


def test_copies_and_pickles_stay_read_only() -> None:
    h = generate_heightmap(4, **WORLD, pad_x=50.0, pad_width=8.0)
    for other in (pickle.loads(pickle.dumps(h)), copy.copy(h), copy.deepcopy(h)):
        assert type(other) is Heightmap
        assert other == h and hash(other) == hash(h)
        assert other.pad == h.pad
        for array in (other.xs, other.ys):
            with pytest.raises(ValueError):
                array[0] = 1.0


# ---- 속도 ----


def test_generation_is_fast(capsys: pytest.CaptureFixture[str]) -> None:
    for s in range(20):  # 데우기
        generate_heightmap(s, **WORLD, pad_x=70.0, pad_width=8.0)
    timings: dict[str, float] = {}
    for name, kwargs in (("pad none", {}), ("pad", {"pad_x": 70.0, "pad_width": 8.0})):
        samples = []
        for s in range(300):
            t0 = time.perf_counter()
            generate_heightmap(s, **WORLD, **kwargs)
            samples.append(time.perf_counter() - t0)
        timings[name] = float(np.median(samples))
    with capsys.disabled():
        print(
            "\nterrain 140m @ 1m: "
            + ", ".join(f"{name} median {seconds * 1e6:.0f} us" for name, seconds in timings.items())
        )
    # 목표는 1ms보다 훨씬 아래(측정값은 위에 출력). 느린 CI에서 흔들리지 않게 여유를 둔다.
    assert max(timings.values()) < 2e-3

"""트랙 생성기 (generators/track.py)."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from evolab.app import strings as S
from evolab.generators import track as T
from evolab.generators.track import Track, generate_track, progress_along, track_from_centerline, track_seed, track_set


def _digest(t: Track) -> str:
    h = hashlib.sha256()
    for a in (t.centerline, t.inner, t.outer, t.cumulative):
        h.update(a.tobytes())
    h.update(repr((t.checkpoints, t.start, t.length, t.width)).encode())
    return h.hexdigest()


def _prefix(template: str) -> str:
    """문구 틀에서 첫 자리표시자 앞부분(오류 문구 비교용)."""
    return re.escape(template.split("{")[0])


# ---- 테스트 쪽에서 따로 짠 기하 검사 (모듈 코드를 쓰지 않는다) ----


def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _crossing_matrix(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """닫힌 꺾은선 p의 선분 i와 q의 선분 j가 만나면(닿기 포함) True인 (len p, len q) 행렬."""
    a, b = p, np.roll(p, -1, axis=0)
    c, d = q, np.roll(q, -1, axis=0)

    def side(o: np.ndarray, e: np.ndarray, r: np.ndarray) -> np.ndarray:
        # o→e에서 본 r의 방향 부호 (배열 모양은 브로드캐스트로 맞춘다)
        ex, ey = e[..., 0] - o[..., 0], e[..., 1] - o[..., 1]
        return np.sign(ex * (r[..., 1] - o[..., 1]) - ey * (r[..., 0] - o[..., 0]))

    A, B = a[:, None, :], b[:, None, :]
    C, D = c[None, :, :], d[None, :, :]
    s1 = side(A, B, C) * side(A, B, D)
    s2 = side(C, D, A) * side(C, D, B)
    box = (
        (np.minimum(A[..., 0], B[..., 0]) <= np.maximum(C[..., 0], D[..., 0]))
        & (np.minimum(C[..., 0], D[..., 0]) <= np.maximum(A[..., 0], B[..., 0]))
        & (np.minimum(A[..., 1], B[..., 1]) <= np.maximum(C[..., 1], D[..., 1]))
        & (np.minimum(C[..., 1], D[..., 1]) <= np.maximum(A[..., 1], B[..., 1]))
    )
    return (s1 <= 0) & (s2 <= 0) & box


def _is_simple(p: np.ndarray) -> bool:
    n = len(p)
    m = _crossing_matrix(p, p)
    i, j = np.indices((n, n))
    gap = (j - i) % n
    return not bool(np.any(m & (gap > 1) & (gap < n - 1)))


def _inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """짝홀 규칙으로 점이 다각형 안에 있나."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x0, y0 = poly[:, 0][None, :], poly[:, 1][None, :]
    x1, y1 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    straddle = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return (np.sum(straddle & (x < xc), axis=1) % 2) == 1


def _min_radius(p: np.ndarray) -> float:
    """세 점을 지나는 원의 반지름 R = abc / (4·넓이)의 최솟값."""
    a = np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1)
    b = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    c = np.linalg.norm(np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0), axis=1)
    u, v = p - np.roll(p, 1, axis=0), np.roll(p, -1, axis=0) - p
    twice_area = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    with np.errstate(divide="ignore"):
        r = a * b * c / (2.0 * twice_area)
    return float(np.min(r))


def _check_track(t: Track, *, checkpoints: int, spacing: float) -> None:
    """모듈 설명의 형태와 유효성 조건을 모두 확인한다."""
    c, inner, outer = t.centerline, t.inner, t.outer
    n = len(c)
    w = t.width
    assert c.shape == inner.shape == outer.shape == (n, 2) and c.dtype == np.float64
    assert n % checkpoints == 0 and n >= 3
    assert not np.array_equal(c[0], c[-1])  # 첫 점을 되풀이하지 않는다
    # 반시계, 고른 간격, 호 길이
    assert _area(c) > 0.0
    seg = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)
    assert seg.max() <= spacing * (1.0 + 1e-6)
    assert seg.max() / seg.min() < 1.02
    assert t.cumulative[0] == 0.0
    assert np.all(np.diff(t.cumulative) > 0.0)
    assert t.cumulative == pytest.approx(np.concatenate(([0.0], np.cumsum(seg)[:-1])), rel=1e-12, abs=1e-9)
    assert t.length == pytest.approx(float(seg.sum()), rel=1e-12)
    # 단순 닫힌 꺾은선, 벽끼리 안 만남, 안쪽 벽이 바깥쪽 벽 안
    assert _is_simple(c) and _is_simple(inner) and _is_simple(outer)
    assert not np.any(_crossing_matrix(inner, outer))
    assert np.all(_inside(outer, inner)) and np.all(_inside(outer, c))
    assert not np.any(_inside(inner, c)) and not np.any(_inside(inner, outer))
    assert 0.0 < _area(inner) < _area(c) < _area(outer)
    # 곡률 반지름
    assert _min_radius(c) > 0.5 * w * 1.2
    # 벽은 왼쪽(안쪽)·오른쪽으로 폭의 절반
    tangent = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
    left = tangent[:, 0] * (inner - c)[:, 1] - tangent[:, 1] * (inner - c)[:, 0]
    assert np.all(left > 0.0)
    assert np.linalg.norm(inner - c, axis=1) == pytest.approx(0.5 * w, rel=1e-9)
    assert np.linalg.norm(outer - c, axis=1) == pytest.approx(0.5 * w, rel=1e-9)
    # 체크포인트: 개수, 순서(호 길이 k·L/K), 길이, 벽 너머까지
    assert len(t.checkpoints) == checkpoints
    step = n // checkpoints
    for k, (a, b) in enumerate(t.checkpoints):
        a, b = np.array(a), np.array(b)
        i = k * step
        assert 0.5 * (a + b) == pytest.approx(c[i], abs=1e-9)
        assert np.linalg.norm(b - a) == pytest.approx(w + 0.2, rel=1e-9)
        assert np.linalg.norm(a - inner[i]) == pytest.approx(0.1, abs=1e-9)
        assert np.linalg.norm(b - outer[i]) == pytest.approx(0.1, abs=1e-9)
        assert t.cumulative[i] == pytest.approx(k * t.length / checkpoints, abs=0.1 * spacing)
    # 출발: 0번 체크포인트 한가운데, 호 길이가 늘어나는 쪽
    x, y, heading = t.start
    assert (x, y) == (c[0, 0], c[0, 1])
    d = c[1] - c[-1]
    assert heading == pytest.approx(math.atan2(d[1], d[0]), abs=1e-12)


# ---- 결정성 ----


def test_same_seed_gives_bit_identical_track() -> None:
    a = generate_track(17)
    for s in range(5):  # 다른 시드를 사이에 돌려도 상태가 남지 않는다
        generate_track(s, curvature=0.9)
    b = generate_track(17)
    for name in ("centerline", "inner", "outer", "cumulative"):
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes()
    assert a.checkpoints == b.checkpoints and a.start == b.start and a.length == b.length
    assert a == b and hash(a) == hash(b) and _digest(a) == _digest(b)


def test_different_seeds_differ() -> None:
    digests = {_digest(generate_track(s)) for s in range(30)}
    assert len(digests) == 30


def test_same_tracks_in_another_process(tmp_path: Path) -> None:
    # 해시 시드가 다른 새 프로세스에서 스크립트 파일로 만들어도 비트까지 같아야 한다.
    script = tmp_path / "make_tracks.py"
    script.write_text(
        "import hashlib\n"
        "from evolab.generators.track import generate_track, track_from_centerline, track_set\n"
        "def digest(t):\n"
        "    h = hashlib.sha256()\n"
        "    for a in (t.centerline, t.inner, t.outer, t.cumulative):\n"
        "        h.update(a.tobytes())\n"
        "    h.update(repr((t.checkpoints, t.start, t.length, t.width)).encode())\n"
        "    return h.hexdigest()\n"
        "tracks = [generate_track(s, curvature=c) for s, c in ((0, 0.5), (7, 1.0), (123456789, 0.2))]\n"
        "tracks += track_set(42, 3)\n"
        "tracks.append(track_from_centerline([(0, 0), (60, 0), (70, 30), (10, 40)], width=8, checkpoints=10))\n"
        "for t in tracks:\n"
        "    print(digest(t))\n",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONHASHSEED="4321")
    out = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, check=True)
    here = [generate_track(s, curvature=c) for s, c in ((0, 0.5), (7, 1.0), (123456789, 0.2))]
    here += track_set(42, 3)
    here.append(track_from_centerline([(0, 0), (60, 0), (70, 30), (10, 40)], width=8, checkpoints=10))
    assert out.stdout.split() == [_digest(t) for t in here]


# ---- 유효성 ----


def test_default_tracks_are_valid_over_many_seeds() -> None:
    lengths = []
    for seed in range(200):
        t = generate_track(seed)
        _check_track(t, checkpoints=24, spacing=1.0)
        assert 150.0 <= t.length <= 250.0
        lengths.append(t.length)
    # 목표 길이는 범위에서 고르게 뽑는다
    assert min(lengths) < 160.0 and max(lengths) > 240.0


@pytest.mark.parametrize(
    "params",
    [
        {"curvature": 0.0},
        {"curvature": 1.0},
        {"segments": 3},
        {"segments": 6, "curvature": 0.8},
        {"segments": 24},
        {"width": 5.0, "checkpoints": 10},
        {"width": 12.0, "curvature": 0.9},
        {"spacing": 0.5, "length_range": (100.0, 120.0)},
        {"spacing": 2.5, "checkpoints": 7},
        {"length_range": (400.0, 400.0), "checkpoints": 1},
    ],
)
def test_tracks_are_valid_for_other_parameters(params: dict) -> None:
    lo, hi = params.get("length_range", (150.0, 250.0))
    for seed in range(15):
        t = generate_track(seed, **params)
        _check_track(t, checkpoints=params.get("checkpoints", 24), spacing=params.get("spacing", 1.0))
        assert lo * (1 - 1e-9) <= t.length <= hi * (1 + 1e-9)
        assert t.width == params.get("width", 8.0)


def test_curvature_makes_tracks_less_round() -> None:
    def roundness(t: Track) -> float:
        # 넓이 / (같은 둘레 원의 넓이): 원이면 1
        return _area(t.centerline) / (t.length**2 / (4.0 * math.pi))

    smooth = np.mean([roundness(generate_track(s, curvature=0.0)) for s in range(20)])
    wild = np.mean([roundness(generate_track(s, curvature=1.0)) for s in range(20)])
    assert smooth > 0.95 > 0.8 > wild


def test_tracks_have_straights_and_varied_corners() -> None:
    t = generate_track(3)
    r = 1.0 / np.maximum(np.abs(T._curvature(t.centerline)), 1e-12)
    assert np.mean(r > 1000.0) > 0.15  # 곧은 길
    assert r.min() < 10.0  # 급한 굽이
    assert np.any((r > 15.0) & (r < 100.0))  # 완만한 굽이


def test_generation_is_fast() -> None:
    generate_track(0)
    times = []
    for seed in range(40):
        t0 = time.perf_counter()
        generate_track(seed)
        times.append(time.perf_counter() - t0)
    mean_ms = 1000.0 * sum(times) / len(times)
    print(f"track generation: mean {mean_ms:.2f} ms, max {1000.0 * max(times):.2f} ms")
    assert mean_ms < 200.0


def test_track_object_is_read_only_and_survives_pickle() -> None:
    t = generate_track(5)
    with pytest.raises(ValueError):
        t.centerline[0, 0] = 1.0
    for other in (pickle.loads(pickle.dumps(t)), copy.deepcopy(t), copy.copy(t)):
        assert other == t and hash(other) == hash(t)
        assert not other.inner.flags.writeable
        assert progress_along(other, *t.centerline[10]) == progress_along(t, *t.centerline[10])
    assert t != generate_track(6)
    x0, y0, x1, y1 = t.bounds
    assert x0 <= t.outer[:, 0].min() and x1 >= t.outer[:, 0].max() and y0 < 0.0 < y1


def test_bad_track_data_is_rejected() -> None:
    t = generate_track(5)
    good = (t.centerline, t.inner, t.outer, t.checkpoints, t.start, t.length, t.width, t.cumulative)
    for k, bad in ((1, t.inner[:-1]), (7, t.cumulative[:-1]), (0, np.full((10, 2), np.nan)), (4, (1.0, 2.0)), (5, "x")):
        args = list(good)
        args[k] = bad
        with pytest.raises(ValueError, match=_prefix(S.TRACK_ERR_DATA)):
            Track(*args)


@pytest.mark.parametrize(
    ("kwargs", "template"),
    [
        ({"seed": -1}, S.TRACK_ERR_SEED),
        ({"seed": 1.5}, S.TRACK_ERR_SEED),
        ({"seed": True}, S.TRACK_ERR_SEED),
        ({"segments": 2}, S.TRACK_ERR_SEGMENTS),
        ({"segments": 12.0}, S.TRACK_ERR_SEGMENTS),
        ({"curvature": 1.5}, S.TRACK_ERR_CURVATURE),
        ({"curvature": float("nan")}, S.TRACK_ERR_CURVATURE),
        ({"curvature": "x"}, S.TRACK_ERR_CURVATURE),
        ({"width": 0.0}, S.TRACK_ERR_WIDTH),
        ({"width": float("inf")}, S.TRACK_ERR_WIDTH),
        ({"length_range": (0.0, 100.0)}, S.TRACK_ERR_LENGTH_RANGE),
        ({"length_range": (200.0, 100.0)}, S.TRACK_ERR_LENGTH_RANGE),
        ({"length_range": (100.0,)}, S.TRACK_ERR_LENGTH_RANGE),
        ({"checkpoints": 0}, S.TRACK_ERR_CHECKPOINTS),
        ({"checkpoints": 5000}, S.TRACK_ERR_CHECKPOINTS),
        ({"spacing": 0.0}, S.TRACK_ERR_SPACING),
        ({"spacing": 0.001}, S.TRACK_ERR_TOO_MANY_POINTS),
        ({"length_range": (100.0, 1e300)}, S.TRACK_ERR_TOO_MANY_POINTS),
    ],
)
def test_bad_arguments_raise_korean_errors(kwargs: dict, template: str) -> None:
    seed = kwargs.pop("seed", 0)
    with pytest.raises(ValueError, match=_prefix(template)):
        generate_track(seed, **kwargs)


def test_impossible_parameters_fail_after_bounded_attempts() -> None:
    # 길이 60m에 폭 40m 트랙은 들어가지 않는다: 50번 시도 뒤 한국어 오류
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match=_prefix(S.TRACK_ERR_GENERATE_FAILED)):
        generate_track(1, width=40.0, length_range=(60.0, 60.0), checkpoints=4)
    assert time.perf_counter() - t0 < 5.0


# ---- 학습 트랙 세트 ----


def test_track_set_uses_independent_derived_seeds() -> None:
    tracks = track_set(7, 5)
    assert len(tracks) == 5
    for k, t in enumerate(tracks):
        seed = int(np.random.SeedSequence([7, k]).generate_state(1, dtype=np.uint64)[0])
        assert track_seed(7, k) == seed
        assert t == generate_track(seed)
    assert len({_digest(t) for t in tracks}) == 5
    seeds = {track_seed(s, k) for s in range(20) for k in range(20)}
    assert len(seeds) == 400  # (세트 시드, 번호)마다 다른 시드
    assert track_seed(1, 2) != track_seed(2, 1)


def test_track_set_is_stable() -> None:
    a = track_set(3, 4, curvature=0.7, width=7.0)
    assert a == track_set(3, 4, curvature=0.7, width=7.0)
    assert a[:2] == track_set(3, 2, curvature=0.7, width=7.0)  # 세트 크기를 바꿔도 앞 트랙은 그대로
    assert track_set(4, 1, curvature=0.7, width=7.0)[0] != a[0]
    assert track_set(3, 0) == []
    with pytest.raises(ValueError, match=_prefix(S.TRACK_ERR_SET_COUNT)):
        track_set(3, -1)
    with pytest.raises(ValueError, match=_prefix(S.TRACK_ERR_SEED)):
        track_set(-3, 2)
    with pytest.raises(ValueError, match=_prefix(S.TRACK_ERR_SET_INDEX)):
        track_seed(3, -1)


# ---- 진행 거리 ----


def test_progress_of_centerline_points_is_their_arc_length() -> None:
    t = generate_track(11)
    n = len(t.centerline)
    for i in range(n):
        s, seg = progress_along(t, *t.centerline[i])
        assert s == pytest.approx(t.cumulative[i], abs=1e-9)
        assert seg in (i, (i - 1) % n)
        mid = 0.5 * (t.centerline[i] + t.centerline[(i + 1) % n])
        s, seg = progress_along(t, *mid)
        seg_len = np.linalg.norm(t.centerline[(i + 1) % n] - t.centerline[i])
        assert seg == i and s == pytest.approx(t.cumulative[i] + 0.5 * seg_len, abs=1e-9)
    assert progress_along(t, *t.centerline[0]) == (0.0, 0)


def test_progress_of_points_across_the_track() -> None:
    # 법선을 따라 벽 쪽으로 옮긴 점은 그 중심선 점의 호 길이 둘레다
    t = generate_track(12)
    for i in range(0, len(t.centerline), 3):
        for f in (-0.95, -0.5, 0.5, 0.95):
            p = t.centerline[i] + f * (t.inner[i] - t.centerline[i])
            s, _ = progress_along(t, *p)
            diff = (s - t.cumulative[i] + 0.5 * t.length) % t.length - 0.5 * t.length
            assert abs(diff) < 0.6


def _assert_on_segment(t: Track, s: float, k: int) -> None:
    """호 길이 s가 선분 k 위에 있다(끝 선분은 0으로 감길 수 있다)."""
    n = len(t.centerline)
    seg = float(np.linalg.norm(t.centerline[(k + 1) % n] - t.centerline[k]))
    if k == n - 1 and s < seg:
        s += t.length
    assert t.cumulative[k] - 1e-9 <= s <= t.cumulative[k] + seg + 1e-9


def test_hint_search_matches_full_search() -> None:
    rng = np.random.default_rng(0)
    for seed in range(10):
        t = generate_track(seed, curvature=1.0)
        n = len(t.centerline)
        for _ in range(300):
            i = int(rng.integers(n))
            f = float(rng.random())
            base = t.centerline[i] + f * (t.centerline[(i + 1) % n] - t.centerline[i])
            normal = t.inner[i] - t.centerline[i]
            p = base + float(rng.uniform(-0.98, 0.98)) * normal
            full = progress_along(t, *p)
            _assert_on_segment(t, *full)
            for hint in (i, i + int(rng.integers(-8, 9)), i + n // 2, -5, 10 * n + i):
                hinted = progress_along(t, *p, hint=hint)
                assert hinted[0] == pytest.approx(full[0], abs=1e-9)
                _assert_on_segment(t, *hinted)
    # 트랙 밖의 점도 힌트와 상관없이 전체 탐색과 같다
    t = generate_track(0)
    for p in ((500.0, 0.0), (0.0, 0.0), (-300.0, 200.0)):
        assert progress_along(t, *p, hint=3) == progress_along(t, *p)


def test_progress_wraps_continuously_across_the_start_line() -> None:
    t = generate_track(21)
    n = len(t.centerline)
    x, y, heading = t.start
    direction = np.array([math.cos(heading), math.sin(heading)])
    hint = n - 3
    last = None
    seen = []
    for d in np.linspace(-3.0, 3.0, 61):  # 출발선 3m 앞에서 3m 뒤까지 중심선 근처를 따라간다
        p = np.array([x, y]) + d * direction
        s, hint = progress_along(t, *p, hint=hint)
        assert 0.0 <= s < t.length
        seen.append(hint)
        if last is not None:
            step = (s - last + 0.5 * t.length) % t.length - 0.5 * t.length
            assert step == pytest.approx(0.1, abs=0.01)
        last = s
    assert seen[0] >= n - 4 and seen[-1] <= 3
    assert progress_along(t, *(np.array([x, y]) - 0.01 * direction))[0] == pytest.approx(t.length - 0.01, abs=1e-3)
    assert progress_along(t, *(np.array([x, y]) + 0.01 * direction))[0] == pytest.approx(0.01, abs=1e-3)


def test_hint_search_is_fast() -> None:
    t = generate_track(4)
    pts = [t.centerline[i] + 0.3 * (t.inner[i] - t.centerline[i]) for i in range(len(t.centerline))]
    hint = 0
    t0 = time.perf_counter()
    for _ in range(20):
        for p in pts:
            _, hint = progress_along(t, p[0], p[1], hint)
    per_call = (time.perf_counter() - t0) / (20 * len(pts))
    print(f"progress_along with hint: {per_call * 1e6:.1f} us/call")
    assert per_call < 200e-6


# ---- 직접 그린 트랙 ----


def _rounded_rectangle(w: float, h: float, r: float, per_corner: int = 6) -> list[tuple[float, float]]:
    pts = []
    centres = ((w / 2 - r, h / 2 - r), (-w / 2 + r, h / 2 - r), (-w / 2 + r, -h / 2 + r), (w / 2 - r, -h / 2 + r))
    for q, (cx, cy) in enumerate(centres):
        for k in range(per_corner + 1):
            a = (q + k / per_corner) * math.pi / 2
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def test_drawn_rounded_rectangle() -> None:
    pts = _rounded_rectangle(60.0, 30.0, 10.0)
    t = track_from_centerline(pts, width=8.0, checkpoints=12)
    _check_track(t, checkpoints=12, spacing=1.0)
    assert t.length == pytest.approx(2 * (40 + 10) + 2 * math.pi * 10, rel=2e-3)
    assert t.start[:2] == pytest.approx(pts[0])  # 첫 꼭짓점이 출발선
    assert t.start[2] == pytest.approx(math.pi / 2, abs=0.1)  # 오른쪽 변을 위로(반시계)
    # 닫는 점을 붙여도 같다
    assert track_from_centerline([*pts, pts[0]], width=8.0, checkpoints=12) == t


def test_drawn_clockwise_track_is_reversed_but_keeps_its_start() -> None:
    pts = _rounded_rectangle(60.0, 30.0, 10.0)[::-1]
    t = track_from_centerline(pts, width=8.0, checkpoints=6, spacing=0.8)
    _check_track(t, checkpoints=6, spacing=0.8)
    assert t.start[:2] == pytest.approx(pts[0])


def test_drawn_polygon_passes_through_its_vertices() -> None:
    pts = [(0.0, 0.0), (60.0, 0.0), (70.0, 30.0), (10.0, 40.0)]
    t = track_from_centerline(pts, width=8.0, checkpoints=10)
    _check_track(t, checkpoints=10, spacing=1.0)
    for p in pts:
        assert np.min(np.linalg.norm(t.centerline - np.array(p), axis=1)) < 0.6


def test_drawn_figure_eight_is_rejected() -> None:
    pts = [(40.0 * math.sin(2 * math.pi * k / 24), 20.0 * math.sin(4 * math.pi * k / 24)) for k in range(24)]
    with pytest.raises(ValueError, match=_prefix(S.TRACK_ERR_SELF_INTERSECT)):
        track_from_centerline(pts, width=4.0, checkpoints=8)


def test_drawn_track_too_tight_for_its_width() -> None:
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    with pytest.raises(ValueError, match=_prefix(S.TRACK_ERR_TOO_TIGHT)):
        track_from_centerline(square, width=8.0, checkpoints=4)
    track_from_centerline(square, width=2.0, checkpoints=4)  # 좁은 길이면 된다


def _horseshoe(gap_angle: float) -> list[tuple[float, float]]:
    """말굽 모양 닫힌 길: 바깥 원호, 끝 반원, 안쪽 원호(거꾸로), 끝 반원. gap_angle이 작으면 두 끝이 가깝다."""
    big, small = 30.0, 18.0
    cap, mid = 0.5 * (big - small), 0.5 * (big + small)

    def arc(cx: float, cy: float, r: float, a0: float, a1: float) -> list[tuple[float, float]]:
        n = max(2, int(abs(a1 - a0) * r / 2.0))
        angles = [a0 + (a1 - a0) * k / n for k in range(n)]
        return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]

    g = gap_angle
    pts = arc(0.0, 0.0, big, g, 2 * math.pi - g)
    pts += arc(mid * math.cos(g), -mid * math.sin(g), cap, -g, -g + math.pi)
    pts += arc(0.0, 0.0, small, 2 * math.pi - g, g)
    pts += arc(mid * math.cos(g), mid * math.sin(g), cap, g + math.pi, g + 2 * math.pi)
    return pts


def test_drawn_track_whose_parts_touch_is_rejected() -> None:
    _check_track(track_from_centerline(_horseshoe(0.6), width=8.0, checkpoints=10), checkpoints=10, spacing=1.0)
    with pytest.raises(ValueError, match=_prefix(S.TRACK_ERR_WALLS_CROSS)):
        track_from_centerline(_horseshoe(0.3), width=8.0, checkpoints=10)


@pytest.mark.parametrize(
    ("points", "template"),
    [
        ([], S.TRACK_ERR_TOO_FEW_POINTS),
        ([(0.0, 0.0), (10.0, 0.0)], S.TRACK_ERR_TOO_FEW_POINTS),
        ([(0.0, 0.0), (10.0, 0.0), (10.0, 0.0), (0.0, 0.0)], S.TRACK_ERR_TOO_FEW_POINTS),
        ([(0.0, 0.0), (10.0, 0.0), (float("nan"), 5.0)], S.TRACK_ERR_POINTS_NOT_FINITE),
        ([(0.0, 0.0, 1.0), (10.0, 0.0, 1.0), (5.0, 5.0, 1.0)], S.TRACK_ERR_POINTS_SHAPE),
        ([(0.0, 0.0), (1.0,)], S.TRACK_ERR_POINTS_SHAPE),
        ([(float(i), 0.0) for i in range(6000)], S.TRACK_ERR_TOO_MANY_DRAWN),
    ],
)
def test_bad_drawn_points(points: list, template: str) -> None:
    with pytest.raises(ValueError, match=_prefix(template)):
        track_from_centerline(points, width=8.0, checkpoints=4)


def test_drawn_track_argument_errors() -> None:
    pts = _rounded_rectangle(60.0, 30.0, 10.0)
    with pytest.raises(ValueError, match=_prefix(S.TRACK_ERR_WIDTH)):
        track_from_centerline(pts, width=-1.0, checkpoints=4)
    with pytest.raises(ValueError, match=_prefix(S.TRACK_ERR_CHECKPOINTS)):
        track_from_centerline(pts, width=8.0, checkpoints=0)
    with pytest.raises(ValueError, match=_prefix(S.TRACK_ERR_TOO_MANY_POINTS)):
        track_from_centerline([(x * 1e6, y * 1e6) for x, y in pts], width=8.0, checkpoints=4)


# ---- 사칙연산 삼각함수 ----


def test_polynomial_trig_matches_libm() -> None:
    theta = np.linspace(-20.0, 20.0, 200_001)
    sin, cos = T._sin_cos(theta)
    assert np.max(np.abs(sin - np.sin(theta))) < 4e-15
    assert np.max(np.abs(cos - np.cos(theta))) < 4e-15
    rng = np.random.default_rng(1)
    for y, x in [*rng.normal(size=(2000, 2)) * 10.0, (0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0), (1e-300, 1.0)]:
        assert T._atan2(float(y), float(x)) == pytest.approx(math.atan2(y, x), abs=2e-15)


# ---- 그림 ----


def test_draw_generated_tracks(qapp, tmp_path: Path) -> None:  # noqa: ANN001
    """자동 생성 트랙 6개를 그린다. EVOLAB_TRACK_PNG에 경로를 주면 거기에 저장해 눈으로 확인할 수 있다."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF

    tracks = [generate_track(s) for s in range(6)]
    cell, metres = 420, 110.0
    img = QImage(3 * cell, 2 * cell, QImage.Format.Format_RGB32)
    img.fill(QColor(28, 31, 36))
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    scale = cell / metres
    for k, t in enumerate(tracks):
        ox, oy = (k % 3 + 0.5) * cell, (k // 3 + 0.5) * cell

        def pt(p, ox=ox, oy=oy) -> QPointF:  # noqa: ANN001
            return QPointF(ox + float(p[0]) * scale, oy - float(p[1]) * scale)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(72, 76, 84))
        painter.drawPolygon(QPolygonF([pt(p) for p in t.outer]))
        painter.setBrush(QColor(38, 62, 40))
        painter.drawPolygon(QPolygonF([pt(p) for p in t.inner]))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(235, 235, 235), 1.5))
        painter.drawPolygon(QPolygonF([pt(p) for p in t.outer]))
        painter.setPen(QPen(QColor(240, 180, 60), 1.5))
        painter.drawPolygon(QPolygonF([pt(p) for p in t.inner]))
        for j, (a, b) in enumerate(t.checkpoints):
            painter.setPen(QPen(QColor(255, 70, 70) if j == 0 else QColor(110, 150, 255), 2.5 if j == 0 else 1.0))
            painter.drawLine(pt(a), pt(b))
        x, y, h = t.start
        painter.setPen(QPen(QColor(255, 255, 255), 2.0))
        painter.drawLine(pt((x, y)), pt((x + 7.0 * math.cos(h), y + 7.0 * math.sin(h))))
        painter.drawText(QPointF(k % 3 * cell + 8, k // 3 * cell + 18), f"seed {k}  {t.length:.0f} m")
    painter.end()
    target = Path(os.environ.get("EVOLAB_TRACK_PNG", str(tmp_path / "tracks.png")))
    assert img.save(str(target))
    assert target.stat().st_size > 10_000
    # 길 색이 칸마다 그려졌다
    road = QColor(72, 76, 84).rgb()
    for k, t in enumerate(tracks):
        p = t.centerline[len(t.centerline) // 3 + 2]  # 체크포인트 선 사이
        assert img.pixel(int((k % 3 + 0.5) * cell + p[0] * scale), int((k // 3 + 0.5) * cell - p[1] * scale)) == road

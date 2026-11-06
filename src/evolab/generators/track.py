"""트랙: 시드로 자동 생성하거나 직접 그린 닫힌 길, 양쪽 벽, 체크포인트 선.

명세 "트랙 자동차 - 트랙": 닫힌 길의 안쪽·바깥쪽 벽(정적 선 체인)과 순서 있는 체크포인트 선.
직접 그리기(중심선 + 폭) 또는 시드로 자동 생성한다. 학습 트랙 세트는 환경 파일의 트랙 세트 시드로
만든다(`track_set`).

모든 트랙은 같은 형태로 나온다.

- 중심선은 반시계 방향(부호 있는 넓이가 양수)으로 도는 닫힌 꺾은선이고, 첫 점을 끝에 되풀이하지 않는다.
  점 간격은 호 길이로 고르게 `length / N`(요청한 `spacing` 이하)이고, 점 수 N은 체크포인트 수 K의
  배수다. 그래서 체크포인트 k는 정확히 중심선 점 k·N/K에 있고, 그 호 길이는 cumulative[k·N/K]
  (≈ k·length/K)다.
- 안쪽 벽(inner)은 달리는 방향의 왼쪽, 바깥쪽 벽(outer)은 오른쪽이다. 반시계 방향 트랙에서 왼쪽이
  곧 안쪽이다. 두 벽은 중심선 점마다 그 점의 법선(앞뒤 이웃을 잇는 방향의 왼쪽 수직)으로 폭의 절반씩
  떨어진 점을 이은 닫힌 꺾은선이고, 점 i가 중심선 점 i와 짝이다.
- 체크포인트 선은 안쪽 벽 쪽 끝에서 바깥쪽 벽 쪽 끝까지 중심선 법선을 따라 긋고, 차가 수치 오차로
  빠져나가지 않게 양끝을 벽 너머로 `_CHECKPOINT_OVERHANG`m씩 늘린다. 0번이 출발·결승선이다.
- 출발 위치는 0번 체크포인트의 한가운데(중심선 0번 점)이고, 방향은 호 길이가 늘어나는 쪽이다.

유효성: 중심선과 두 벽이 각각 스스로 만나지 않는 단순 닫힌 꺾은선이고, 두 벽이 서로 만나지 않으며
(안쪽 벽이 바깥쪽 벽 안에 있다), 중심선의 가장 작은 곡률 반지름이 폭/2 × `_RADIUS_MARGIN`보다 크다
(그보다 작으면 굽이 안쪽 벽이 접힌다). 자동 생성은 이 조건을 못 맞추면 정해진 다음 하위 시드로 다시
뽑고(최대 `_MAX_ATTEMPTS`번), 직접 그린 트랙은 그 자리에서 ValueError를 낸다.

결정성: 같은 인자는 호출과 프로세스에 상관없이 비트까지 같은 배열을 낸다. 난수는
`SeedSequence([시드, 시도 번호])`로 만든 생성기 하나에서 정해진 순서로만 뽑고, 문자열 해시나
dict·set 순서에 기대지 않는다. 계산은 사칙연산, 제곱근, floor/rint, 최솟값·최댓값과 numpy 누적합
(앞에서부터 차례로 더한다)만 쓴다. sin·cos·atan2는 라이브러리마다 마지막 비트가 다를 수 있어서,
사칙연산만 쓰는 다항식으로 직접 계산한다(`_sin_cos`, `_atan2`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from evolab.app import strings as S

# 유효성
_RADIUS_MARGIN: Final = 1.2  # 중심선 최소 곡률 반지름 > 폭/2 × 이 값
_CHECKPOINT_OVERHANG: Final = 0.1  # m. 체크포인트 선을 양쪽 벽 너머로 늘리는 길이
_MAX_ATTEMPTS: Final = 50  # 자동 생성에서 하위 시드를 바꿔 다시 뽑는 최대 횟수

# 자동 생성 모양: 제어 다각형 (반지름은 단위 원 기준, 크기는 목표 길이로 맞춘다)
_RADIUS_SPREAD: Final = 0.8  # 곡률 1에서 꼭짓점 반지름이 1 ± 이 값 안에서 고르게
_ANGLE_JITTER: Final = 0.7  # 곡률 1에서 꼭짓점 각도 흔들림, 꼭짓점 간격 대비(±절반). 1 미만이라 순서가 유지됨
_ASPECT_MAX: Final = 0.6  # 곡률 1에서 가로로 늘이는 비율의 최대 추가분 (1 ~ 1.6배, 화면이 가로로 길다)
# 굽이: 반지름은 (폭/2 × _RADIUS_MARGIN) × _CORNER_MIN_FACTOR 이상, 양옆 변의 절반에 들어가는 만큼 이하
_CORNER_MIN_FACTOR: Final = 1.25
_RELAX_PASSES: Final = 30  # 굽이가 들어가지 않는 꼭짓점을 펴는 최대 횟수
_START_INTO_STRAIGHT: Final = 0.25  # 출발선: 가장 긴 직선의 시작에서 그 길이의 이 비율만큼 들어간 곳

# 직접 그린 트랙의 호 길이 표: 점 간격 하나에 이만큼, 꼭짓점 사이마다 적어도 _DENSE_MIN개 찍는다
_DENSE_PER_SPACING: Final = 8
_DENSE_MIN: Final = 8

# 안전 한도
_MAX_POINTS: Final = 5000  # 중심선 점 수 (겹침 검사 일이 점 수의 제곱에 가깝다)
_MAX_SEGMENTS: Final = 100
_MAX_CHECKPOINTS: Final = 1000
_MAX_DRAWN_POINTS: Final = 5000
_CROSS_CHUNK: Final = 512  # 겹침 검사에서 한 번에 다루는 선분 행 수 (임시 배열 크기를 묶는다)

# progress_along 힌트 탐색: 힌트에서 한쪽으로 이 거리(m)를 넘게 걸어가야 하면 전체 탐색으로 바꾼다
_HINT_REACH: Final = 30.0


@dataclass(frozen=True)
class Track:
    """닫힌 트랙 하나. 형태는 모듈 설명과 같다.

    만들 때 배열을 float64로 복사하고 읽기 전용으로 바꾼다. checkpoints는 (안쪽 끝, 바깥쪽 끝) 쌍의
    목록이고 0번이 출발·결승선이다. start는 (x, y, 방향 라디안), cumulative[i]는 중심선 i번 점까지의
    호 길이(cumulative[0] = 0)이고 length는 닫힌 중심선 전체 길이(마지막 점에서 0번 점까지 포함)다.
    """

    centerline: np.ndarray
    inner: np.ndarray
    outer: np.ndarray
    checkpoints: list[tuple[tuple[float, float], tuple[float, float]]]
    start: tuple[float, float, float]
    length: float
    width: float
    cumulative: np.ndarray
    _segments: _Segments = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        center = _frozen_points(self.centerline)
        inner = _frozen_points(self.inner)
        outer = _frozen_points(self.outer)
        cumulative = np.array(self.cumulative, dtype=np.float64, copy=True)
        n = center.shape[0]
        if n < 3 or inner.shape != center.shape or outer.shape != center.shape or cumulative.shape != (n,):
            raise ValueError(S.TRACK_ERR_DATA)
        try:
            length = float(self.length)
            width = float(self.width)
            checkpoints = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))) for a, b in self.checkpoints]
            x, y, heading = (float(v) for v in self.start)
        except (TypeError, ValueError, IndexError):
            raise ValueError(S.TRACK_ERR_DATA) from None
        if not (np.all(np.isfinite(cumulative)) and math.isfinite(length) and length > 0.0 and math.isfinite(width)):
            raise ValueError(S.TRACK_ERR_DATA)
        cumulative.flags.writeable = False
        object.__setattr__(self, "centerline", center)
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "outer", outer)
        object.__setattr__(self, "cumulative", cumulative)
        object.__setattr__(self, "checkpoints", checkpoints)
        object.__setattr__(self, "start", (x, y, heading))
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "_segments", _Segments(center, cumulative, length, width))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """두 벽을 모두 담는 상자 (x_min, y_min, x_max, y_max)."""
        both = np.concatenate((self.inner, self.outer))
        lo = both.min(axis=0)
        hi = both.max(axis=0)
        return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Track):
            return NotImplemented
        return (
            self.start == other.start
            and self.length == other.length
            and self.width == other.width
            and self.checkpoints == other.checkpoints
            and np.array_equal(self.centerline, other.centerline)
            and np.array_equal(self.inner, other.inner)
            and np.array_equal(self.outer, other.outer)
            and np.array_equal(self.cumulative, other.cumulative)
        )

    def __hash__(self) -> int:
        # +0.0으로 -0.0을 0.0으로 바꿔서, 같다고 판정되는 두 트랙의 해시가 같게 한다.
        arrays = tuple((a + 0.0).tobytes() for a in (self.centerline, self.inner, self.outer, self.cumulative))
        return hash((arrays, tuple(self.checkpoints), self.start, self.length, self.width))

    def __reduce__(self) -> tuple[type[Track], tuple[object, ...]]:
        # pickle(워커 사이 전달)과 copy가 __post_init__을 다시 거치게 한다. 기본 방식은 __dict__를 그대로
        # 되살려서 복사본의 배열이 쓰기 가능해진다.
        fields = (self.centerline, self.inner, self.outer, self.checkpoints, self.start, self.length, self.width)
        return (Track, (*fields, self.cumulative))


class _Segments:
    """progress_along이 매 스텝 쓰는 중심선 선분 표. 힌트 탐색은 파이썬 수로, 전체 탐색은 배열로 한다.

    선분 i는 중심선 점 i에서 점 (i+1) mod N까지다.
    """

    __slots__ = ("n", "length", "reach", "inside2", "ax", "ay", "dx", "dy", "inv", "lists")

    def __init__(self, center: np.ndarray, cumulative: np.ndarray, length: float, width: float) -> None:
        d = np.roll(center, -1, axis=0) - center
        len2 = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]
        self.n = center.shape[0]
        self.length = length
        self.reach = max(4, math.ceil(_HINT_REACH * self.n / length))
        self.inside2 = 0.25 * width * width  # 힌트 탐색 결과를 믿는 거리(폭/2)의 제곱
        self.ax = center[:, 0].copy()
        self.ay = center[:, 1].copy()
        self.dx = d[:, 0].copy()
        self.dy = d[:, 1].copy()
        self.inv = np.where(len2 > 0.0, 1.0 / np.where(len2 > 0.0, len2, 1.0), 0.0)
        seg = np.sqrt(len2)
        # (ax, ay, dx, dy, 1/길이², 길이, 시작 호 길이)를 선분마다 묶은 파이썬 튜플
        self.lists = list(zip(*(a.tolist() for a in (self.ax, self.ay, self.dx, self.dy, self.inv, seg, cumulative))))


def _frozen_points(points: np.ndarray) -> np.ndarray:
    try:
        a = np.array(points, dtype=np.float64, copy=True)
    except (TypeError, ValueError):
        raise ValueError(S.TRACK_ERR_DATA) from None
    if a.ndim != 2 or a.shape[1] != 2 or not np.all(np.isfinite(a)):
        raise ValueError(S.TRACK_ERR_DATA)
    a.flags.writeable = False
    return a


# ---------------------------------------------------------------------------
# 공개 함수
# ---------------------------------------------------------------------------


def generate_track(
    seed: int,
    *,
    segments: int = 12,
    curvature: float = 0.5,
    width: float = 8.0,
    length_range: tuple[float, float] = (150.0, 250.0),
    checkpoints: int = 24,
    spacing: float = 1.0,
) -> Track:
    """시드로 정해지는 트랙을 만든다.

    1. 꼭짓점 `segments`개를 고른 간격의 각도에 두고 반지름과 각도를 `curvature`(0..1)만큼 흔든
       다각형을 만든다(곡률이 클수록 가로로도 더 늘인다). 꼭짓점이 원점 둘레를 차례로 돌므로 반시계
       방향이다. 목표 길이를 `length_range`에서 고르게 뽑아(같은 생성기) 둘레를 그 길이로 맞춘다.
    2. 꼭짓점마다 모서리를 원호로 깎는다(`_Path`). 원호 사이는 정확한 직선이라 곧은 길과 크기가
       제각각인 굽이가 번갈아 나온다. 반지름은 폭에 맞춘 최솟값과 양옆 변에 들어가는 최댓값 사이에서
       고른다. 곡률 0이면 모두 최댓값이라 원에 가까운 둥근 다각형이고, 곡률이 클수록 급한 굽이와 안으로
       파인 굽이가 많다. 최솟값도 들어가지 않는 꼭짓점은 먼저 이웃 쪽으로 편다(`_relax`).
    3. 길을 호 길이로 고르게 다시 찍고 전체 길이를 목표 길이로 맞춘다. 출발선은 가장 긴 직선에 두고,
       트랙은 중심선 상자의 가운데가 원점에 오게 놓는다.

    유효하지 않으면 다음 하위 시드로 다시 뽑는다. `_MAX_ATTEMPTS`번 모두 실패하면 ValueError.
    """
    _validate_seed(seed)
    if isinstance(segments, bool) or not isinstance(segments, (int, np.integer)) or not 3 <= segments <= _MAX_SEGMENTS:
        raise ValueError(S.TRACK_ERR_SEGMENTS.format(value=segments, limit=_MAX_SEGMENTS))
    curvature = _number(curvature, S.TRACK_ERR_CURVATURE)
    if not 0.0 <= curvature <= 1.0:  # NaN도 여기서 걸린다
        raise ValueError(S.TRACK_ERR_CURVATURE.format(value=curvature))
    try:
        lo, hi = (float(v) for v in length_range)
    except (TypeError, ValueError):
        raise ValueError(S.TRACK_ERR_LENGTH_RANGE.format(value=length_range)) from None
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo <= hi):
        raise ValueError(S.TRACK_ERR_LENGTH_RANGE.format(value=length_range))
    width, checkpoints, spacing = _validate_common(width, checkpoints, spacing)
    _point_count(hi, checkpoints, spacing)  # 가장 긴 경우에 한도를 넘으면 미리 알린다

    n = int(segments)
    corner_min = 0.5 * width * _RADIUS_MARGIN * _CORNER_MIN_FACTOR
    reason = ""
    for attempt in range(_MAX_ATTEMPTS):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), attempt]))
        # 뽑는 순서와 개수는 인자로만 정해진다: 목표 길이, 가로 비율, 시작 각도, 반지름, 각도 흔들림, 굽이.
        target = lo + (hi - lo) * float(rng.random())
        aspect = 1.0 + _ASPECT_MAX * curvature * float(rng.random())
        phase = float(rng.random())
        radii = 1.0 + _RADIUS_SPREAD * curvature * rng.uniform(-1.0, 1.0, n)
        jitter = _ANGLE_JITTER * curvature * rng.uniform(-0.5, 0.5, n)
        tightness = 1.0 - min(1.0, 2.0 * curvature) * rng.random(n)

        sin, cos = _sin_cos((np.arange(n) + phase + jitter) * (2.0 * math.pi / n))
        poly = np.stack((aspect * radii * cos, radii * sin), axis=1)
        poly = _relax(poly * (target / _perimeter(poly)), corner_min, target)
        center = _sample_to_length(_Path(poly, tightness, corner_min), target, checkpoints, spacing)
        center = center - 0.5 * (center.min(axis=0) + center.max(axis=0))

        track = _assemble(center, width, checkpoints)
        problem = _problem(track)
        if problem is None:
            if lo * (1.0 - 1e-9) <= track.length <= hi * (1.0 + 1e-9):
                return track
            problem = S.TRACK_ERR_LENGTH_RANGE.format(value=length_range)
        reason = problem
    raise ValueError(S.TRACK_ERR_GENERATE_FAILED.format(attempts=_MAX_ATTEMPTS, seed=seed, reason=reason))


def track_from_centerline(
    points: Sequence[tuple[float, float]],
    *,
    width: float,
    checkpoints: int,
    spacing: float = 1.0,
) -> Track:
    """직접 그린 중심선 꼭짓점으로 트랙을 만든다(에디터의 트랙 그리기).

    꼭짓점을 차례로 모두 지나는 닫힌 구심 캣멀-롬 스플라인으로 매끄럽게 잇고, 호 길이로 고르게 다시
    찍는다. 첫 꼭짓점이 출발선이다. 시계 방향으로 그렸으면 반시계 방향이 되게 순서를 뒤집는다(출발점은
    그대로, 달리는 방향만 바뀐다). 이어서 같은 꼭짓점(마지막 점이 첫 점과 같은 경우 포함)은 하나로 친다.
    꼭짓점이 3개보다 적거나, 중심선이 스스로 만나거나, 굽이가 폭에 비해 너무 급하거나, 트랙의 다른
    부분끼리 벽이 겹치면 ValueError. 손으로 그린 떨리는 선은 급한 굽이로 걸릴 수 있으니 꼭짓점을
    줄여서 넘기는 편이 좋다.
    """
    width, checkpoints, spacing = _validate_common(width, checkpoints, spacing)
    try:
        raw = np.array(points, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(S.TRACK_ERR_POINTS_SHAPE) from None
    if raw.ndim != 2 or raw.shape[1] != 2:
        if raw.size == 0:
            raise ValueError(S.TRACK_ERR_TOO_FEW_POINTS.format(count=0))
        raise ValueError(S.TRACK_ERR_POINTS_SHAPE)
    if raw.shape[0] > _MAX_DRAWN_POINTS:
        raise ValueError(S.TRACK_ERR_TOO_MANY_DRAWN.format(count=raw.shape[0], limit=_MAX_DRAWN_POINTS))
    if not np.all(np.isfinite(raw)):
        raise ValueError(S.TRACK_ERR_POINTS_NOT_FINITE)
    ctrl = _drop_repeats(raw)
    if ctrl.shape[0] < 3:
        raise ValueError(S.TRACK_ERR_TOO_FEW_POINTS.format(count=ctrl.shape[0]))
    if _signed_area(ctrl) < 0.0:
        ctrl = np.concatenate((ctrl[:1], ctrl[:0:-1]))  # 첫 점은 그대로, 나머지 순서를 뒤집는다

    # 꼭짓점을 차례로 지나는 곡선은 꼭짓점을 이은 다각형보다 길다: 호 길이 표를 만들기 전에 한도를 본다
    _point_count(_perimeter(ctrl), checkpoints, spacing)
    table = _arc_table(_catmull_rom_closed, ctrl, spacing)
    count = _point_count(float(table[1][-1]), checkpoints, spacing)
    center = _resample(_catmull_rom_closed, ctrl, table, count)
    if _signed_area(center) <= 0.0 or _polygon_self_crosses(center):
        raise ValueError(S.TRACK_ERR_SELF_INTERSECT)
    track = _assemble(center, width, checkpoints)
    problem = _problem(track)
    if problem is not None:
        raise ValueError(problem)
    return track


def track_seed(set_seed: int, index: int) -> int:
    """학습 트랙 세트의 index번 트랙 시드: `SeedSequence([set_seed, index])`가 내는 64비트 정수 하나."""
    _validate_seed(set_seed)
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0:
        raise ValueError(S.TRACK_ERR_SET_INDEX.format(value=index))
    state = np.random.SeedSequence([int(set_seed), int(index)]).generate_state(1, dtype=np.uint64)
    return int(state[0])


def track_set(set_seed: int, count: int, **params: object) -> list[Track]:
    """환경 파일의 학습 트랙 세트: k번 트랙 = generate_track(track_seed(set_seed, k), **params)."""
    _validate_seed(set_seed)
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError(S.TRACK_ERR_SET_COUNT.format(value=count))
    return [generate_track(track_seed(set_seed, k), **params) for k in range(int(count))]  # type: ignore[arg-type]


def progress_along(track: Track, x: float, y: float, hint: int | None = None) -> tuple[float, int]:
    """(x, y)에서 가장 가까운 중심선 위 점의 호 길이 [0, length)와 그 점이 있는 선분 번호.

    선분 i는 중심선 점 i에서 점 (i+1) mod N까지다. 같은 거리의 선분이 여럿이면 번호가 작은 쪽이다.
    `hint`(지난번 선분 번호)를 주면 거기서 거리가 줄어드는 쪽으로 이웃 선분을 따라 걸어가 찾는다(매
    스텝 부르는 런타임용, 보통 선분 두세 개만 본다). 최소 곡률 반지름이 폭의 절반보다 크고 벽이 서로
    만나지 않으니, 트랙 안의 점에서 중심선까지의 거리는 가장 가까운 점 둘레에서 한 번만 줄었다 는다.
    그래서 트랙 안의 점이면 전체 탐색과 같은 점을 찾는다. 찾은 점이 폭/2보다 멀거나(트랙 밖의 점,
    또는 엉뚱한 힌트로 다른 곳의 길에 닿음. 벽이 만나지 않으니 트랙 안의 점은 다른 곳의 길에서 폭/2
    넘게 떨어져 있다) 한쪽으로 `_HINT_REACH`m를 넘게 걸어야 하면 힌트를 믿지 않고 전체 탐색을 한다.
    """
    g = track._segments
    x = float(x)
    y = float(y)
    if hint is not None:
        found = _walk(g, x, y, int(hint) % g.n)
        if found is not None:
            return found
    px = x - g.ax
    py = y - g.ay
    t = np.clip((px * g.dx + py * g.dy) * g.inv, 0.0, 1.0)
    ex = px - t * g.dx
    ey = py - t * g.dy
    i = int(np.argmin(ex * ex + ey * ey))
    return _arc(g, i, float(t[i]))


# ---------------------------------------------------------------------------
# progress_along 보조
# ---------------------------------------------------------------------------


def _dist2(seg: tuple[float, ...], x: float, y: float) -> tuple[float, float]:
    """선분까지 거리의 제곱과 가장 가까운 점의 선분 위 비율 t (0..1)."""
    ax, ay, dx, dy, inv = seg[0], seg[1], seg[2], seg[3], seg[4]
    px = x - ax
    py = y - ay
    t = (px * dx + py * dy) * inv
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    ex = px - t * dx
    ey = py - t * dy
    return ex * ex + ey * ey, t


def _walk(g: _Segments, x: float, y: float, i: int) -> tuple[float, int] | None:
    """힌트 i에서 거리가 줄어드는 쪽으로 걸어간다. 앞으로 먼저, 앞이 줄지 않으면 뒤로."""
    segs = g.lists
    best, t = _dist2(segs[i], x, y)
    for step in (1, -1):
        moved = False
        for _ in range(g.reach):
            j = (i + step) % g.n
            d, tj = _dist2(segs[j], x, y)
            if not d < best:
                break
            i, best, t, moved = j, d, tj, True
        else:
            return None  # 한쪽으로 너무 멀리 걸어감: 힌트가 맞지 않는다
        if moved:
            break
    if best > g.inside2:
        return None  # 트랙 밖의 점이거나 다른 곳의 힌트: 이 둘레의 가장 가까운 점이 답이라는 보장이 없다
    return _arc(g, i, t)


def _arc(g: _Segments, i: int, t: float) -> tuple[float, int]:
    seg = g.lists[i]
    s = seg[6] + t * seg[5]
    if s >= g.length:
        s -= g.length
    return (s if s > 0.0 else 0.0), i


# ---------------------------------------------------------------------------
# 인자와 유효성 검사
# ---------------------------------------------------------------------------


def _number(value: object, message: str) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(message.format(value=value)) from None


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(S.TRACK_ERR_SEED.format(value=seed))


def _validate_common(width: float, checkpoints: int, spacing: float) -> tuple[float, int, float]:
    width_f = _number(width, S.TRACK_ERR_WIDTH)
    if not (math.isfinite(width_f) and width_f > 0.0):
        raise ValueError(S.TRACK_ERR_WIDTH.format(value=width))
    spacing_f = _number(spacing, S.TRACK_ERR_SPACING)
    if not (math.isfinite(spacing_f) and spacing_f > 0.0):
        raise ValueError(S.TRACK_ERR_SPACING.format(value=spacing))
    if (
        isinstance(checkpoints, bool)
        or not isinstance(checkpoints, (int, np.integer))
        or not 1 <= checkpoints <= _MAX_CHECKPOINTS
    ):
        raise ValueError(S.TRACK_ERR_CHECKPOINTS.format(value=checkpoints, limit=_MAX_CHECKPOINTS))
    return width_f, int(checkpoints), spacing_f


def _point_count(length: float, checkpoints: int, spacing: float) -> int:
    """간격이 spacing 이하이고 체크포인트 수의 배수인 가장 작은 점 수(3 이상)."""
    ratio = length / (spacing * checkpoints)  # 체크포인트 한 칸에 드는 점 수
    if not ratio * checkpoints <= _MAX_POINTS:  # inf, NaN도 여기서 걸린다
        raise ValueError(S.TRACK_ERR_TOO_MANY_POINTS.format(length=length, spacing=spacing, limit=_MAX_POINTS))
    count = max(1, math.ceil(ratio - 1e-9)) * checkpoints
    while count < 3:
        count += checkpoints
    if count > _MAX_POINTS:
        raise ValueError(S.TRACK_ERR_TOO_MANY_POINTS.format(length=length, spacing=spacing, limit=_MAX_POINTS))
    return count


def _problem(track: Track) -> str | None:
    """트랙이 유효하지 않으면 그 이유(화면 문구), 유효하면 None. 싼 검사부터 한다."""
    center = track.centerline
    limit = 0.5 * track.width * _RADIUS_MARGIN
    kappa = np.abs(_curvature(center))
    worst = int(np.argmax(kappa))
    if kappa[worst] * limit >= 1.0:
        x, y = center[worst]
        return S.TRACK_ERR_TOO_TIGHT.format(
            radius=1.0 / kappa[worst], limit=limit, width=track.width, x=float(x), y=float(y)
        )
    if _signed_area(center) <= 0.0 or _polygon_self_crosses(center):
        return S.TRACK_ERR_SELF_INTERSECT
    if not 0.0 < _signed_area(track.inner) < _signed_area(track.outer):
        return S.TRACK_ERR_WALLS_CROSS
    if _polygon_self_crosses(track.inner) or _polygon_self_crosses(track.outer):
        return S.TRACK_ERR_WALLS_CROSS
    if _polygons_cross(track.inner, track.outer):
        return S.TRACK_ERR_WALLS_CROSS
    return None


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """a→b에서 본 c의 방향 부호 (+1 왼쪽, -1 오른쪽, 0 한 직선). 모두 (k, 2) 배열."""
    return np.sign((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def _crossing_rows(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray, rows: slice, same: bool) -> bool:
    """선분 p0[rows]→p1[rows]와 선분 q0→q1 가운데 만나는(닿기 포함) 쌍이 있나.

    same이면 같은 닫힌 꺾은선의 선분끼리라서, 같은 선분과 꼭짓점을 나누는 이웃 선분은 뺀다.
    상자 겹침으로 후보를 먼저 줄이고 후보 쌍에만 방향 검사를 한다.
    """
    a0 = p0[rows]
    a1 = p1[rows]
    amin = np.minimum(a0, a1)
    amax = np.maximum(a0, a1)
    bmin = np.minimum(q0, q1)
    bmax = np.maximum(q0, q1)
    near = (
        (amin[:, None, 0] <= bmax[None, :, 0])
        & (bmin[None, :, 0] <= amax[:, None, 0])
        & (amin[:, None, 1] <= bmax[None, :, 1])
        & (bmin[None, :, 1] <= amax[:, None, 1])
    )
    if same:
        n = q0.shape[0]
        gap = (np.arange(n)[None, :] - np.arange(rows.start, rows.start + a0.shape[0])[:, None]) % n
        near &= (gap > 1) & (gap < n - 1)
    ii, jj = np.nonzero(near)
    if ii.size == 0:
        return False
    a0, a1, b0, b1 = a0[ii], a1[ii], q0[jj], q1[jj]
    return bool(
        np.any((_orient(a0, a1, b0) * _orient(a0, a1, b1) <= 0.0) & (_orient(b0, b1, a0) * _orient(b0, b1, a1) <= 0.0))
    )


def _polygon_self_crosses(poly: np.ndarray) -> bool:
    """닫힌 꺾은선이 스스로 만나나(이웃하지 않는 두 선분이 닿거나 교차)."""
    nxt = np.roll(poly, -1, axis=0)
    n = poly.shape[0]
    return any(
        _crossing_rows(poly, nxt, poly, nxt, slice(s, min(s + _CROSS_CHUNK, n)), True)
        for s in range(0, n, _CROSS_CHUNK)
    )


def _polygons_cross(a: np.ndarray, b: np.ndarray) -> bool:
    """두 닫힌 꺾은선이 만나나."""
    an = np.roll(a, -1, axis=0)
    bn = np.roll(b, -1, axis=0)
    n = a.shape[0]
    return any(
        _crossing_rows(a, an, b, bn, slice(s, min(s + _CROSS_CHUNK, n)), False) for s in range(0, n, _CROSS_CHUNK)
    )


# ---------------------------------------------------------------------------
# 기하
# ---------------------------------------------------------------------------


def _assemble(center: np.ndarray, width: float, checkpoints: int) -> Track:
    """중심선(반시계, 점 수는 체크포인트 수의 배수)에서 벽, 체크포인트, 출발 위치, 호 길이를 만든다."""
    n = center.shape[0]
    nxt = np.roll(center, -1, axis=0)
    tangent = nxt - np.roll(center, 1, axis=0)
    norm = np.sqrt(tangent[:, 0] * tangent[:, 0] + tangent[:, 1] * tangent[:, 1])
    tangent = tangent / norm[:, None]
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)  # 왼쪽 = 안쪽
    half = 0.5 * width
    inner = center + half * normal
    outer = center - half * normal

    seg = nxt - center
    run = np.cumsum(np.sqrt(seg[:, 0] * seg[:, 0] + seg[:, 1] * seg[:, 1]))  # 앞에서부터 차례로 더한다
    cumulative = np.concatenate(((0.0,), run[:-1]))

    reach = half + _CHECKPOINT_OVERHANG
    lines = []
    for k in range(checkpoints):
        i = k * (n // checkpoints)
        cx, cy = float(center[i, 0]), float(center[i, 1])
        nx, ny = float(normal[i, 0]), float(normal[i, 1])
        lines.append(((cx + reach * nx, cy + reach * ny), (cx - reach * nx, cy - reach * ny)))
    start = (float(center[0, 0]), float(center[0, 1]), _atan2(float(tangent[0, 1]), float(tangent[0, 0])))
    return Track(center, inner, outer, lines, start, float(run[-1]), width, cumulative)


def _drop_repeats(points: np.ndarray) -> np.ndarray:
    """이어서 같은 꼭짓점(끝에서 첫 점으로 돌아온 점 포함)을 하나로 줄인다."""
    tol = 1e-9 * max(1.0, float(np.max(np.abs(points))))
    keep = [0]
    for i in range(1, points.shape[0]):
        d = points[i] - points[keep[-1]]
        if abs(float(d[0])) > tol or abs(float(d[1])) > tol:
            keep.append(i)
    while len(keep) > 1:
        d = points[keep[-1]] - points[keep[0]]
        if abs(float(d[0])) > tol or abs(float(d[1])) > tol:
            break
        keep.pop()
    return points[keep]


def _signed_area(poly: np.ndarray) -> float:
    """닫힌 꺾은선의 부호 있는 넓이(반시계 양수). 신발끈 공식, 차례로 더한다."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.cumsum(x * np.roll(y, -1) - np.roll(x, -1) * y)[-1])


def _perimeter(poly: np.ndarray) -> float:
    d = np.roll(poly, -1, axis=0) - poly
    return float(np.cumsum(np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]))[-1])


def _curvature(poly: np.ndarray) -> np.ndarray:
    """닫힌 꺾은선 꼭짓점마다 앞뒤 이웃과 세 점을 지나는 원의 부호 있는 곡률(멩거 곡률, 왼쪽으로 굽으면 양수)."""
    prev = np.roll(poly, 1, axis=0)
    nxt = np.roll(poly, -1, axis=0)
    a = poly - prev
    b = nxt - poly
    c = nxt - prev
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    la = np.sqrt(a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1])
    lb = np.sqrt(b[:, 0] * b[:, 0] + b[:, 1] * b[:, 1])
    lc = np.sqrt(c[:, 0] * c[:, 0] + c[:, 1] * c[:, 1])
    denom = la * lb * lc
    return np.where(denom > 0.0, 2.0 * cross / np.where(denom > 0.0, denom, 1.0), np.inf)


def _corner_room(poly: np.ndarray) -> tuple[np.ndarray, ...]:
    """꼭짓점 i마다 (나가는 변 길이, 나가는 방향, 꺾임의 cross, dot, tan(꺾인 각/2), 들어가는 가장 큰 반지름).

    i번 꼭짓점의 나가는 변은 i → i+1이다. 가장 큰 반지름은 원호가 양옆 변의 절반 안에 들어가는
    반지름이다(곧은 꼭짓점은 inf). tan(각/2) = |cross| / (1 + dot)이고, 거의 되돌아가는 꼭짓점은
    반지름이 0에 가깝다.
    """
    edge = np.roll(poly, -1, axis=0) - poly
    edge_len = np.sqrt(edge[:, 0] * edge[:, 0] + edge[:, 1] * edge[:, 1])
    out_dir = edge / edge_len[:, None]
    in_dir = np.roll(out_dir, 1, axis=0)
    cross = in_dir[:, 0] * out_dir[:, 1] - in_dir[:, 1] * out_dir[:, 0]
    dot = in_dir[:, 0] * out_dir[:, 0] + in_dir[:, 1] * out_dir[:, 1]
    tan_half = np.abs(cross) / np.maximum(1.0 + dot, 1e-12)
    room = 0.5 * np.minimum(np.roll(edge_len, 1), edge_len)
    bends = tan_half > 1e-12
    fit = np.where(bends, room / np.where(bends, tan_half, 1.0), np.inf)
    return edge_len, out_dir, cross, dot, tan_half, fit


def _relax(poly: np.ndarray, corner_min: float, perimeter: float) -> np.ndarray:
    """반지름 corner_min인 원호가 들어가지 않는 꼭짓점을 이웃 둘의 가운데 쪽으로 반씩 당긴다.

    꺾인 각이 줄고 원호가 들어갈 자리가 생긴다. 당길 때마다 둘레를 perimeter로 다시 맞추고,
    `_RELAX_PASSES`번까지 되풀이한다(그래도 안 되면 유효성 검사가 걸러 낸다).
    """
    for _ in range(_RELAX_PASSES):
        bad = _corner_room(poly)[5] < corner_min
        if not np.any(bad):
            break
        mid = 0.5 * (np.roll(poly, 1, axis=0) + np.roll(poly, -1, axis=0))
        poly = np.where(bad[:, None], 0.5 * (poly + mid), poly)
        poly = poly * (perimeter / _perimeter(poly))
    return poly


class _Path:
    """다각형 꼭짓점마다 모서리를 원호로 깎은 닫힌 길. 직선과 원호가 접선이 이어지게 번갈아 놓인다.

    꼭짓점 i의 굽이 반지름은 corner_min + (fit_i - corner_min) × tightness_i다(fit_i는 `_corner_room`의
    가장 큰 반지름). 원호가 양옆 변의 절반 안에 들어가니 이웃 굽이끼리 겹치지 않는다. fit_i가 corner_min보다
    작으면 fit_i를 쓰고, 너무 급하면 유효성 검사가 걸러 낸다. tightness가 작으면 급한 굽이 옆에 긴
    직선이 남고, 크면 변을 다 쓰는 완만한 곡선이 된다.

    조각 순서는 꼭짓점 0의 원호, 변 0의 직선, 꼭짓점 1의 원호, ...이다. 출발점(호 길이 0)은 가장 긴
    직선의 시작에서 그 길이의 `_START_INTO_STRAIGHT`만큼 들어간 곳이다(가장 긴 직선이 여럿이면 앞 번호).
    """

    def __init__(self, poly: np.ndarray, tightness: np.ndarray, corner_min: float) -> None:
        edge_len, out_dir, cross, dot, tan_half, fit = _corner_room(poly)
        in_dir = np.roll(out_dir, 1, axis=0)
        bends = tan_half > 1e-12
        radius = np.where(fit > corner_min, corner_min + (np.minimum(fit, 1e6) - corner_min) * tightness, fit)
        radius = np.where(bends, radius, 0.0)
        cut = radius * tan_half  # 꼭짓점에서 원호 양끝(접점)까지 거리
        sign = np.sign(cross)
        turn = np.array([_atan2(abs(c), d) for c, d in zip(cross.tolist(), dot.tolist())])  # 꺾인 각 크기

        self.arc_start = poly - cut[:, None] * in_dir
        self.arc_centre = self.arc_start + (sign * radius)[:, None] * np.stack((-in_dir[:, 1], in_dir[:, 0]), axis=1)
        self.arc_radius = np.where(bends, radius, 1.0)
        self.arc_sign = sign
        self.line_start = poly + cut[:, None] * out_dir
        self.line_dir = out_dir
        line_len = np.maximum(edge_len - cut - np.roll(cut, -1), 0.0)

        pieces = np.stack((radius * turn, line_len), axis=1).reshape(-1)  # 원호0, 직선0, 원호1, ...
        run = np.cumsum(pieces)
        self.piece_start = np.concatenate(((0.0,), run[:-1]))
        self.total = float(run[-1])
        longest = int(np.argmax(line_len))
        self.origin = float(self.piece_start[2 * longest + 1]) + _START_INTO_STRAIGHT * float(line_len[longest])

    def sample(self, count: int) -> np.ndarray:
        """출발점에서 호 길이로 고르게 count개."""
        s = self.origin + np.arange(count) * (self.total / count)
        s = np.where(s >= self.total, s - self.total, s)
        piece = np.clip(np.searchsorted(self.piece_start, s, side="right") - 1, 0, self.piece_start.size - 1)
        local = s - self.piece_start[piece]
        k = piece // 2
        # 원호 위의 점: 접점을 중심 둘레로 sign × local / 반지름만큼 돌린다
        sin, cos = _sin_cos(self.arc_sign[k] * local / self.arc_radius[k])
        rel = self.arc_start[k] - self.arc_centre[k]
        turned = np.stack((cos * rel[:, 0] - sin * rel[:, 1], sin * rel[:, 0] + cos * rel[:, 1]), axis=1)
        arc = self.arc_centre[k] + turned
        line = self.line_start[k] + local[:, None] * self.line_dir[k]
        return np.where((piece % 2 == 0)[:, None], arc, line)


def _sample_to_length(path: _Path, target: float, checkpoints: int, spacing: float) -> np.ndarray:
    """길을 고르게 찍고 꺾은선 길이가 정확히 target이 되게 늘인다.

    원호 위의 현은 호보다 조금 짧아서 늘이면 직선 위 간격이 spacing을 살짝 넘을 수 있다. 그러면 점을
    체크포인트 수만큼 늘려 다시 찍는다(보통 한 번도 안 한다).
    """
    count = _point_count(target, checkpoints, spacing)
    while True:
        center = path.sample(count)
        center = center * (target / _perimeter(center))
        d = np.roll(center, -1, axis=0) - center
        longest2 = float(np.max(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]))
        if longest2 <= spacing * spacing or count + checkpoints > _MAX_POINTS:
            return center
        count += checkpoints


# ---------------------------------------------------------------------------
# 직접 그린 트랙: 구심 캣멀-롬 스플라인
# ---------------------------------------------------------------------------

_Curve = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _catmull_rom_closed(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    """닫힌 구심 캣멀-롬 스플라인(배리-골드먼 식). u는 [0, 꼭짓점 수) 전역 매개변수이고 정수 u에서
    그 꼭짓점을 지난다. 구심 매개화라 한 구간 안에 고리나 뾰족점이 생기지 않는다."""
    n = ctrl.shape[0]
    base = np.floor(u)
    f = (u - base)[:, None]
    i = base.astype(np.intp) % n
    p0 = ctrl[(i - 1) % n]
    p1 = ctrl[i]
    p2 = ctrl[(i + 1) % n]
    p3 = ctrl[(i + 2) % n]

    def knot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = b - a
        return np.sqrt(np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]))[:, None]  # 거리의 제곱근

    t1 = knot(p0, p1)
    t2 = t1 + knot(p1, p2)
    t3 = t2 + knot(p2, p3)
    t = t1 + f * (t2 - t1)
    a1 = ((t1 - t) * p0 + t * p1) / t1
    a2 = ((t2 - t) * p1 + (t - t1) * p2) / (t2 - t1)
    a3 = ((t3 - t) * p2 + (t - t2) * p3) / (t3 - t2)
    b1 = ((t2 - t) * a1 + t * a2) / t2
    b2 = ((t3 - t) * a2 + (t - t1) * a3) / (t3 - t1)
    return ((t2 - t) * b1 + (t - t1) * b2) / (t2 - t1)


def _arc_table(curve: _Curve, ctrl: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """호 길이 표: 촘촘한 표본의 매개변수 u(끝에 꼭짓점 수)와 그 점까지의 누적 호 길이(끝에 전체 길이).

    꼭짓점 사이마다 그 거리를 spacing / `_DENSE_PER_SPACING`으로 나눈 만큼(최소 `_DENSE_MIN`개) 찍는다.
    긴 변일수록 많이 찍어서, 표 칸 하나가 언제나 점 간격보다 훨씬 짧다.
    """
    n = ctrl.shape[0]
    chord = np.roll(ctrl, -1, axis=0) - ctrl
    chord_len = np.sqrt(chord[:, 0] * chord[:, 0] + chord[:, 1] * chord[:, 1])
    per = np.maximum(_DENSE_MIN, np.ceil(chord_len * (_DENSE_PER_SPACING / spacing))).astype(np.intp)
    first = np.concatenate(((0,), np.cumsum(per)[:-1]))
    owner = np.repeat(np.arange(n), per)
    k = np.arange(owner.size) - first[owner]
    u = owner + k / per[owner]
    dense = curve(ctrl, u)
    d = np.roll(dense, -1, axis=0) - dense
    cum = np.concatenate(((0.0,), np.cumsum(np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]))))
    return np.concatenate((u, (float(n),))), cum


def _resample(curve: _Curve, ctrl: np.ndarray, table: tuple[np.ndarray, np.ndarray], count: int) -> np.ndarray:
    """곡선을 u=0(첫 꼭짓점)에서 시작해 호 길이로 고르게 count개 다시 찍는다.

    호 길이 표에서 목표 길이마다 매개변수를 직선 보간으로 찾고 그 매개변수에서 곡선을 다시 계산한다.
    그래서 점이 꺾은선이 아니라 곡선 위에 있어 곡률이 매끄럽다.
    """
    u, cum = table
    seg = np.diff(cum)
    targets = np.arange(count) * (float(cum[-1]) / count)
    j = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, seg.size - 1)
    frac = np.clip((targets - cum[j]) / np.where(seg[j] > 0.0, seg[j], 1.0), 0.0, 1.0)
    return curve(ctrl, u[j] + frac * (u[j + 1] - u[j]))


# ---------------------------------------------------------------------------
# 사칙연산만 쓰는 삼각함수 (어느 플랫폼에서나 같은 비트)
# ---------------------------------------------------------------------------

_HALF_PI_HI: Final = 1.5707963267948966  # π/2에 가장 가까운 double
_HALF_PI_LO: Final = 6.123233995736766e-17  # π/2 - _HALF_PI_HI
# |r| ≤ π/4에서 sin은 r^19, cos는 r^20 항까지 쓰면 double 정밀도보다 정확하다. atan은 |t| ≤ tan(π/16)에서 t^27까지.
_SIN_COEF: Final = tuple((1.0 if k % 2 == 0 else -1.0) / math.factorial(2 * k + 1) for k in range(10))
_COS_COEF: Final = tuple((1.0 if k % 2 == 0 else -1.0) / math.factorial(2 * k) for k in range(11))
_ATAN_COEF: Final = tuple((1.0 if k % 2 == 0 else -1.0) / (2 * k + 1) for k in range(14))


def _sin_cos(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """배열 theta의 (sin, cos). π/2 단위로 줄인 나머지(|r| ≤ π/4)에 테일러 다항식을 쓴다."""
    k = np.rint(theta / _HALF_PI_HI)
    r = (theta - k * _HALF_PI_HI) - k * _HALF_PI_LO
    r2 = r * r
    sin_r = np.full_like(r, _SIN_COEF[-1])
    for c in _SIN_COEF[-2::-1]:
        sin_r = sin_r * r2 + c
    sin_r = sin_r * r
    cos_r = np.full_like(r, _COS_COEF[-1])
    for c in _COS_COEF[-2::-1]:
        cos_r = cos_r * r2 + c
    q = k.astype(np.int64) % 4
    sin = np.choose(q, (sin_r, cos_r, -sin_r, -cos_r))
    cos = np.choose(q, (cos_r, -sin_r, -cos_r, sin_r))
    return sin, cos


def _atan2(y: float, x: float) -> float:
    """math.atan2와 같은 뜻의 각도 [-π, π]. 반각 공식을 두 번 써서 |t| ≤ tan(π/16)으로 줄이고 급수를 쓴다."""
    if x == 0.0 and y == 0.0:
        return 0.0
    ax = abs(x)
    ay = abs(y)
    swap = ay > ax
    t = ax / ay if swap else ay / ax  # 0 ≤ t ≤ 1
    for _ in range(2):  # atan t = 2·atan(t / (1 + √(1 + t²)))
        t = t / (1.0 + math.sqrt(1.0 + t * t))
    t2 = t * t
    acc = _ATAN_COEF[-1]
    for c in _ATAN_COEF[-2::-1]:
        acc = acc * t2 + c
    a = 4.0 * (acc * t)
    if swap:
        a = (_HALF_PI_HI - a) + _HALF_PI_LO
    if x < 0.0:
        a = math.pi - a
    return -a if y < 0.0 else a

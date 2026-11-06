"""지형 자동 생성: 시드로 정해지는 높이맵과 평평한 착륙 패드.

명세 "달 착륙선 - 지형": 자동 생성 높이맵 + 평평한 착륙 패드. 패드 위치와 폭은 시드로 정한다
(위치와 폭을 뽑는 일은 `randomize`가 하고, 여기서는 받은 값으로 지형을 만든다).
`pad_x=None`이면 패드 없는 지형(언덕 넘는 자동차용)을 만든다.

지형은 세 가지를 더해 만든다.

- 언덕: 1차원 그래디언트 노이즈(펄린) 여러 옥타브의 합. 가장 긴 파장은 48m이고, 파장을 반씩
  줄여 가며 구간 길이의 두 배까지 쌓는다. 거칠기가 옥타브마다 줄어드는 비율(지속도)을 정한다.
  0이면 긴 파장만 남아 완만한 구릉, 1이면 짧은 파장이 살아 있어 들쭉날쭉하다.
- 분화구: 사발 모양 웅덩이와 둘레의 낮은 테두리. 개수는 지형 폭 30m당 최대 한 개이고,
  거칠기가 낮을수록 얕다(오래되어 깎인 분화구).
- 착륙 패드: 패드 구간을 한 높이로 평평하게 만들고, 양옆 `pad_margin` 안에서 지형을 패드 높이로
  부드럽게 이어 붙인다. 패드 가장자리에서 경사가 0에서 시작해 최대 `_PAD_RAMP_SLOPE`까지만
  커지는 포락선 안으로 지형을 눌러서, 패드 옆이 절벽이 되지 않는다.

결정성: 같은 인자는 호출과 프로세스에 상관없이 비트까지 같은 배열을 낸다. 난수는
`np.random.default_rng(seed)` 하나에서 정해진 순서로만 뽑고, 문자열 해시나 dict·set 순서에 기대지
않는다. 계산은 사칙연산, floor, abs, 최솟값·최댓값과 `math.fsum`(정확한 합)만 쓴다. exp, sin, pow처럼
라이브러리마다 마지막 비트가 다를 수 있는 함수와, 더하는 순서가 배열 모양에 따라 바뀔 수 있는 numpy
합계는 쓰지 않는다(여러 줄을 더할 때는 파이썬 반복문이나 `np.add.at`으로 정해진 순서대로 차례로 더한다).

안전 한도: 꼭짓점 `_MAX_VERTICES`개, 폭 `_MAX_WIDTH`m. 일과 메모리는 꼭짓점 수와 폭에 비례하게
묶여 있다(언덕은 점을 나눠 계산하고, 분화구는 자기 둘레의 꼭짓점만 계산한다).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Final

import numpy as np

from evolab.app import strings as S

# 언덕 옥타브
_BASE_WAVELENGTH: Final = 48.0  # m. 140m 세계에 큰 언덕이 서너 개
_MIN_WAVELENGTH_IN_SEGMENTS: Final = 2.0  # 가장 짧은 파장 = 구간 길이의 두 배 (그보다 짧으면 표본에 안 잡힘)
_MAX_OCTAVES: Final = 16
_PERSISTENCE_SMOOTH: Final = 0.28  # 거칠기 0의 옥타브 감쇠 비율
_PERSISTENCE_ROUGH: Final = 0.72  # 거칠기 1의 옥타브 감쇠 비율

# 분화구 (깊이는 진폭 단위, 반지름은 m)
_CRATERS_PER_METRE: Final = 1.0 / 30.0
_CRATER_RADIUS_MIN: Final = 3.0
_CRATER_RADIUS_MAX: Final = 10.0
_CRATER_DEPTH_MIN: Final = 0.3  # 가장 큰 분화구의 깊이, 진폭 대비
_CRATER_DEPTH_MAX: Final = 0.6
_CRATER_RIM_HEIGHT: Final = 0.35  # 테두리 높이, 깊이 대비
_CRATER_RIM_HALF_WIDTH: Final = 0.5  # 테두리 반폭, 반지름 대비

# 착륙 패드 옆 경사 포락선의 최대 기울기 (약 27도)
_PAD_RAMP_SLOPE: Final = 0.5

_MAX_VERTICES: Final = 1_000_000
# 지형 폭 상한(m). 분화구 수와 긴 파장 옥타브의 격자점 수는 꼭짓점 수가 아니라 폭에 비례하므로,
# 폭을 막지 않으면 구간을 길게 잡은 아주 넓은 지형에서 난수 배열만으로 메모리가 바닥난다.
_MAX_WIDTH: Final = 1_000_000.0
# 언덕을 계산할 때 한 번에 다루는 점 수. (옥타브, 점) 임시 배열 크기를 이 정도로 묶어 둔다.
_HILL_CHUNK: Final = 4096


@dataclass(frozen=True)
class Heightmap:
    """x 순서대로 이은 지형 꺾은선.

    xs는 계속 커지고 xs[0]과 xs[-1]이 지형의 양끝이다. pad가 있으면 (왼쪽 x, 오른쪽 x, 높이)이고,
    양끝은 xs의 꼭짓점이며 그 사이 모든 꼭짓점의 높이가 정확히 pad 높이와 같다.
    만들 때 배열을 float64로 복사하고 읽기 전용으로 바꾼다.
    """

    xs: np.ndarray
    ys: np.ndarray
    pad: tuple[float, float, float] | None

    def __post_init__(self) -> None:
        xs = np.array(self.xs, dtype=np.float64, copy=True)
        ys = np.array(self.ys, dtype=np.float64, copy=True)
        if xs.ndim != 1 or ys.ndim != 1 or xs.shape != ys.shape or xs.size < 2:
            raise ValueError(S.TERRAIN_ERR_HEIGHTMAP_SHAPE.format(nx=xs.shape, ny=ys.shape))
        if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
            raise ValueError(S.TERRAIN_ERR_HEIGHTMAP_NOT_FINITE)
        not_increasing = np.flatnonzero(np.diff(xs) <= 0.0)
        if not_increasing.size:
            raise ValueError(S.TERRAIN_ERR_HEIGHTMAP_ORDER.format(index=int(not_increasing[0]) + 1))
        pad = self.pad
        if pad is not None:
            try:
                left, right, y = (float(v) for v in pad)
            except (TypeError, ValueError):
                raise ValueError(S.TERRAIN_ERR_HEIGHTMAP_PAD.format(pad=pad)) from None
            pad = (left, right, y)
            if not _pad_matches(xs, ys, pad):
                raise ValueError(S.TERRAIN_ERR_HEIGHTMAP_PAD.format(pad=pad))
        xs.flags.writeable = False
        ys.flags.writeable = False
        object.__setattr__(self, "xs", xs)
        object.__setattr__(self, "ys", ys)
        object.__setattr__(self, "pad", pad)

    def height_at(self, x: float) -> float:
        """x에서의 지형 높이. 꼭짓점 사이는 직선 보간, 지형 밖은 가까운 끝의 높이."""
        return float(np.interp(x, self.xs, self.ys))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Heightmap):
            return NotImplemented
        return self.pad == other.pad and np.array_equal(self.xs, other.xs) and np.array_equal(self.ys, other.ys)

    def __hash__(self) -> int:
        # +0.0으로 -0.0을 0.0으로 바꿔서, 같다고 판정되는 두 높이맵의 해시가 같게 한다.
        return hash(((self.xs + 0.0).tobytes(), (self.ys + 0.0).tobytes(), self.pad))

    def __reduce__(self) -> tuple[type[Heightmap], tuple[np.ndarray, np.ndarray, tuple[float, float, float] | None]]:
        # pickle(워커 사이 전달)과 copy가 __post_init__을 다시 거치게 한다. 기본 방식은 __dict__를 그대로
        # 되살려서 복사본의 배열이 쓰기 가능해지고, 고치면 해시가 바뀐다.
        return (Heightmap, (self.xs, self.ys, self.pad))


def _pad_matches(xs: np.ndarray, ys: np.ndarray, pad: tuple[float, float, float]) -> bool:
    left, right, y = pad
    if not (math.isfinite(left) and math.isfinite(right) and math.isfinite(y) and left < right):
        return False
    i = int(np.searchsorted(xs, left))
    j = int(np.searchsorted(xs, right))
    if j >= xs.size or xs[i] != left or xs[j] != right:
        return False
    return bool(np.all(ys[i : j + 1] == y))


def generate_heightmap(
    seed: int,
    *,
    x_min: float,
    x_max: float,
    segment: float = 1.0,
    base_y: float = 0.0,
    amplitude: float = 4.0,
    roughness: float = 0.5,
    pad_x: float | None = None,
    pad_width: float | None = None,
    pad_margin: float = 2.0,
) -> Heightmap:
    """시드로 정해지는 지형을 만든다.

    - 구간 길이는 `segment` 이하로, 지형 폭을 같은 길이 구간으로 나눈다(xs[0]=x_min, xs[-1]=x_max).
    - 모든 높이는 [base_y - amplitude, base_y + amplitude] 안이다. 패드가 없으면 평균 높이가 base_y
      근처이고 base_y에서 가장 먼 점이 정확히 amplitude만큼 떨어져 있다.
    - `pad_x`를 주면 [pad_x - pad_width/2, pad_x + pad_width/2]가 한 높이로 평평한 착륙 패드가 된다.
      패드 높이는 평평하게 만들기 전 지형의 pad_x 높이(`height_at`과 같은 직선 보간)다. 패드와 양옆
      `pad_margin`이 지형 안에 들어가도록 pad_x를 옮긴다. 패드 안쪽 꼭짓점은 지우고 양끝을 꼭짓점으로
      넣으므로 패드는 선분 하나다.
    - 패드는 난수를 뽑지 않는다. 같은 시드라면 패드가 있든 없든 패드에서 먼 지형은 같다.
    """
    _validate(seed, x_min, x_max, segment, base_y, amplitude, roughness, pad_x, pad_width, pad_margin)

    width = x_max - x_min
    ratio = width / segment
    if not ratio < _MAX_VERTICES:  # inf도 여기서 걸린다(math.ceil(inf)는 OverflowError)
        raise ValueError(S.TERRAIN_ERR_TOO_MANY_VERTICES.format(count=_vertex_count_text(ratio), limit=_MAX_VERTICES))
    n_segments = max(1, math.ceil(ratio - 1e-9))
    if n_segments + 1 > _MAX_VERTICES:
        raise ValueError(S.TERRAIN_ERR_TOO_MANY_VERTICES.format(count=n_segments + 1, limit=_MAX_VERTICES))
    xs = np.linspace(x_min, x_max, n_segments + 1)
    step = width / n_segments

    rng = np.random.default_rng(seed)
    # 뽑는 순서: 분화구 먼저, 그다음 긴 파장부터 짧은 파장 순. 구간 길이만 바꾸면 짧은 옥타브의
    # 난수가 뒤에 붙을 뿐이라 분화구와 큰 언덕의 위치와 모양은 그대로다.
    craters = _craters(rng, xs, x_min, x_max, roughness)
    hills = _hills(rng, xs, x_min, x_max, step, roughness)
    # 언덕을 먼저 진폭 1로 맞춘 뒤 분화구(진폭 단위)를 더하고, 합을 다시 진폭 1로 맞춘다. 합만 한 번
    # 맞추면 언덕이 우연히 낮은 시드에서 분화구까지 몇 배로 키워져 가파른 구덩이가 된다.
    shape = _unit_peak(_unit_peak(hills) + craters)

    low = base_y - amplitude
    high = base_y + amplitude
    # shape는 정확히 [-1, 1] 안이고 반올림은 순서를 지키므로 높이가 띠를 넘지 않는다. clip은 안전장치다.
    ys = np.clip(base_y + amplitude * shape, low, high)

    if pad_x is None:
        return Heightmap(xs, ys, None)

    assert pad_width is not None  # _validate가 확인함
    half = 0.5 * pad_width
    lo_centre = x_min + half + pad_margin
    hi_centre = x_max - half - pad_margin
    centre = min(max(float(pad_x), lo_centre), hi_centre) if lo_centre <= hi_centre else 0.5 * (x_min + x_max)
    left = max(centre - half, x_min)
    right = min(centre + half, x_max)
    if not left < right:
        raise ValueError(S.TERRAIN_ERR_PAD_WIDTH.format(value=pad_width))
    pad_y = min(max(float(np.interp(centre, xs, ys)), low), high)
    pad_xs, pad_ys = _flatten_for_pad(xs, ys, left, right, pad_y, pad_margin, low, high)
    return Heightmap(pad_xs, pad_ys, (left, right, pad_y))


def _validate(
    seed: int,
    x_min: float,
    x_max: float,
    segment: float,
    base_y: float,
    amplitude: float,
    roughness: float,
    pad_x: float | None,
    pad_width: float | None,
    pad_margin: float,
) -> None:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(S.TERRAIN_ERR_SEED.format(value=seed))
    if not (math.isfinite(x_min) and math.isfinite(x_max) and x_max > x_min and math.isfinite(x_max - x_min)):
        raise ValueError(S.TERRAIN_ERR_RANGE.format(x_min=x_min, x_max=x_max))
    if x_max - x_min > _MAX_WIDTH:
        raise ValueError(S.TERRAIN_ERR_TOO_WIDE.format(width=x_max - x_min, limit=int(_MAX_WIDTH)))
    if not (math.isfinite(segment) and segment > 0.0):
        raise ValueError(S.TERRAIN_ERR_SEGMENT.format(value=segment))
    if not math.isfinite(base_y):
        raise ValueError(S.TERRAIN_ERR_BASE_Y.format(value=base_y))
    if not (math.isfinite(amplitude) and amplitude >= 0.0):
        raise ValueError(S.TERRAIN_ERR_AMPLITUDE.format(value=amplitude))
    # 높이 띠 [base_y - amplitude, base_y + amplitude]와 그 폭(패드 높이와의 차)이 유한해야 계산 중 넘침이 없다.
    if not math.isfinite(abs(base_y) + 2.0 * amplitude):
        raise ValueError(S.TERRAIN_ERR_HEIGHT_RANGE.format(base_y=base_y, amplitude=amplitude))
    if not 0.0 <= roughness <= 1.0:  # NaN도 여기서 걸린다
        raise ValueError(S.TERRAIN_ERR_ROUGHNESS.format(value=roughness))
    if not (math.isfinite(pad_margin) and pad_margin >= 0.0):
        raise ValueError(S.TERRAIN_ERR_PAD_MARGIN.format(value=pad_margin))
    if pad_x is None:
        return
    if not math.isfinite(pad_x):
        raise ValueError(S.TERRAIN_ERR_PAD_X.format(value=pad_x))
    if pad_width is None:
        raise ValueError(S.TERRAIN_ERR_PAD_WIDTH_MISSING)
    if not (math.isfinite(pad_width) and pad_width > 0.0):
        raise ValueError(S.TERRAIN_ERR_PAD_WIDTH.format(value=pad_width))
    if pad_width + 2.0 * pad_margin > x_max - x_min:
        raise ValueError(S.TERRAIN_ERR_PAD_TOO_WIDE.format(width=pad_width, margin=pad_margin, world=x_max - x_min))


def _vertex_count_text(ratio: float) -> str:
    """오류 문구에 넣을 꼭짓점 수. 아주 크면 자릿수가 수백 개인 정수 대신 지수 표기로 줄인다."""
    if ratio < 1e15:
        return str(max(1, math.ceil(ratio - 1e-9)) + 1)
    if math.isfinite(ratio):
        return f"{ratio:.3g}"
    return f">{sys.float_info.max:.3g}"


def _unit_peak(values: np.ndarray) -> np.ndarray:
    """평균을 빼고 가장 큰 절댓값이 정확히 1이 되게 나눈다. 모두 같으면 0."""
    centred = values - math.fsum(values.tolist()) / values.size
    peak = float(np.max(np.abs(centred)))
    if peak == 0.0:
        return np.zeros_like(values)
    return centred / peak


def _hills(
    rng: np.random.Generator, xs: np.ndarray, x_min: float, x_max: float, step: float, roughness: float
) -> np.ndarray:
    """옥타브를 쌓은 1차원 그래디언트 노이즈(펄린). 크기는 부르는 쪽에서 맞춘다.

    옥타브 k의 파장은 48m / 2^k이고, 격자점마다 [-1, 1] 기울기를 두고 5차 이음 곡선으로 섞는다.
    옥타브마다 격자 시작점을 무작위로 밀어서 격자점(노이즈가 0인 곳)이 겹치지 않게 한다.
    """
    wavelengths = [_BASE_WAVELENGTH]
    while len(wavelengths) < _MAX_OCTAVES and wavelengths[-1] * 0.5 >= _MIN_WAVELENGTH_IN_SEGMENTS * step:
        wavelengths.append(wavelengths[-1] * 0.5)
    persistence = _PERSISTENCE_SMOOTH + (_PERSISTENCE_ROUGH - _PERSISTENCE_SMOOTH) * roughness
    weights = [1.0]
    while len(weights) < len(wavelengths):
        weights.append(weights[-1] * persistence)  # pow 대신 곱셈: 어느 CPU에서나 같은 비트

    wl = np.array(wavelengths)
    # 시작점은 언제나 _MAX_OCTAVES개를 뽑고, 기울기는 긴 파장부터 이어서 뽑는다. 그래서 구간 길이를
    # 줄여 옥타브가 늘어도 앞 옥타브들이 쓰는 난수는 그대로다.
    origins = x_min - rng.random(_MAX_OCTAVES)[: wl.size] * wl
    counts = np.floor((x_max - origins) / wl).astype(np.intp) + 2  # 각 옥타브가 쓰는 격자점 수
    starts = np.concatenate(((0,), np.cumsum(counts)[:-1]))
    gradients = rng.uniform(-1.0, 1.0, int(counts.sum()))

    # 점을 _HILL_CHUNK개씩 나눠 계산해서 (옥타브, 점) 임시 배열이 꼭짓점 수에 비례해 커지지 않게 한다.
    # 계산은 모두 점마다 따로라서 나누는 방식과 상관없이 결과가 비트까지 같다.
    total = np.empty_like(xs)
    last = (counts - 2)[:, np.newaxis]
    for start in range(0, xs.size, _HILL_CHUNK):
        part = xs[start : start + _HILL_CHUNK]
        pos = (part[np.newaxis, :] - origins[:, np.newaxis]) / wl[:, np.newaxis]  # (옥타브, 점)
        cell = np.floor(pos)
        u = pos - cell
        i = np.minimum(cell.astype(np.intp), last) + starts[:, np.newaxis]
        a = gradients[i] * u
        b = gradients[i + 1] * (u - 1.0)
        fade = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
        octaves = a + fade * (b - a)

        # 옥타브를 차례로 더한다(더하는 순서를 고정).
        block = weights[0] * octaves[0]
        for k in range(1, wl.size):
            block += weights[k] * octaves[k]
        total[start : start + part.size] = block
    return total


def _craters(rng: np.random.Generator, xs: np.ndarray, x_min: float, x_max: float, roughness: float) -> np.ndarray:
    """분화구 높이 합(진폭 단위). 사발 (1 - q²)²과 테두리 고리, 둘 다 경계에서 값과 기울기가 0이다."""
    most = math.floor((x_max - x_min) * _CRATERS_PER_METRE + 0.5)
    if most <= 0:
        return np.zeros_like(xs)
    count = int(rng.integers(most // 2, most, endpoint=True))
    if count == 0:
        return np.zeros_like(xs)
    centres = rng.uniform(x_min, x_max, count)
    u = rng.random(count)
    radii = _CRATER_RADIUS_MIN + (_CRATER_RADIUS_MAX - _CRATER_RADIUS_MIN) * (u * u)  # 작은 분화구가 더 많게
    depths = rng.uniform(_CRATER_DEPTH_MIN, _CRATER_DEPTH_MAX, count) * (radii / _CRATER_RADIUS_MAX)
    depths *= 0.5 + 0.5 * roughness

    # 분화구는 중심에서 (1 + 테두리 반폭)·반지름 = 1.5·반지름 안에만 있다(밖에서는 정확히 0). 그래서
    # 중심에서 2·반지름(+ 좌표 반올림 여유) 안의 꼭짓점만 계산한다. 모든 꼭짓점을 계산하면 일이
    # 분화구 수 × 꼭짓점 수라서, 넓은 지형(분화구 수도 폭에 비례)에서 시간과 메모리가 폭의 제곱으로 는다.
    reach = 2.0 * radii + 4.0 * np.spacing(np.abs(centres) + 2.0 * radii)
    lo = np.searchsorted(xs, centres - reach, side="left")
    hi = np.searchsorted(xs, centres + reach, side="right")
    lengths = hi - lo
    owner = np.repeat(np.arange(count), lengths)  # (분화구, 점) 쌍마다 분화구 번호. 분화구 순서대로 놓인다
    point = np.arange(int(lengths.sum())) + np.repeat(lo - (np.cumsum(lengths) - lengths), lengths)
    q = np.abs(xs[point] - centres[owner]) / radii[owner]
    bowl = np.maximum(1.0 - q * q, 0.0)
    r = (q - 1.0) * (1.0 / _CRATER_RIM_HALF_WIDTH)
    rim = np.maximum(1.0 - r * r, 0.0)
    shapes = depths[owner] * (_CRATER_RIM_HEIGHT * (rim * rim) - bowl * bowl)
    # 쌍 순서대로 하나씩 더한다(np.add.at은 버퍼 없이 차례로 더한다). 한 점에는 분화구 번호 순서로
    # 더해지므로, 분화구를 하나씩 통째로 더하는 것과 비트까지 같다(범위 밖에서 더하던 값은 +0.0이다).
    out = np.zeros_like(xs)
    np.add.at(out, point, shapes)
    return out


def _flatten_for_pad(
    xs: np.ndarray,
    ys: np.ndarray,
    left: float,
    right: float,
    pad_y: float,
    margin: float,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """패드 안 꼭짓점을 지우고 양끝을 넣은 뒤, 패드 옆 지형을 패드 높이로 이어 붙인다.

    패드 가장자리에서 거리 d인 점의 높이 차(지형 - 패드)를 두 단계로 줄인다.
    1. d < margin에서 smoothstep(d / margin)을 곱해 지형의 잔물결을 패드 쪽으로 잦아들게 한다.
    2. 포락선 g(d) 안으로 자른다. g는 d < margin에서 k·d²/(2·margin)(기울기 0에서 k까지),
       그 뒤로 k·(d - margin/2)인 직선이다(k = _PAD_RAMP_SLOPE). 잘린 곳의 기울기는 k 이하라서
       깊은 골짜기에 패드가 있어도 옆이 절벽이 되지 않는다.
    포락선이 지형 높이 차보다 넓어지는 곳부터는 지형이 그대로다.
    """
    outside = (xs < left) | (xs > right)
    ox = xs[outside]
    oy = ys[outside]
    d = np.where(ox < left, left - ox, ox - right)
    above = oy - pad_y
    if margin > 0.0:
        t = np.minimum(d / margin, 1.0)
        fade = t * t * (3.0 - 2.0 * t)
        curve = (0.5 * _PAD_RAMP_SLOPE / margin) * (d * d)
        envelope = np.where(d < margin, curve, _PAD_RAMP_SLOPE * (d - 0.5 * margin))
        rise = np.clip(fade * above, -envelope, envelope)
        touched = (d < margin) | (np.abs(above) > envelope)
    else:
        envelope = _PAD_RAMP_SLOPE * d
        rise = np.clip(above, -envelope, envelope)
        touched = np.abs(above) > envelope
    # 손대지 않는 점은 원래 값을 그대로 둔다(pad_y + (y - pad_y)는 y와 마지막 비트가 다를 수 있다).
    oy = np.where(touched, np.clip(pad_y + rise, low, high), oy)
    split = int(np.searchsorted(ox, left))
    out_xs = np.concatenate((ox[:split], (left, right), ox[split:]))
    out_ys = np.concatenate((oy[:split], (pad_y, pad_y), oy[split:]))
    return out_xs, out_ys

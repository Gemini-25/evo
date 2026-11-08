"""장애물 줄 생성기(v2 설계 4.2, D9): 점프 게임의 다가오는 장애물.

움직이는 세계 물체 하나에 사각형 장애물 count개를 줄지어 놓는다(줄 좌표: 첫 장애물이 x에서 출발, 바닥 높이 y). 줄 전체가
같은 속도로 direction 쪽으로 밀리므로 장애물끼리 앞지르거나 흘러 붙지 않는다. 런타임은 이 줄을 kinematic 물체 하나로 만들고
길 [(0, 0), (length·cos d, length·sin d)]을 speed에서 시작해 초마다 accel씩 빨라지는 한 번(once) 움직임으로 민다.

간격(gap)은 초다: 줄이 민 거리 scrolled(t) = speed·t + accel·t²/2(속력은 0 아래로 내려가지 않음)일 때, 장애물 k가 줄
좌표 x_k에 있으면 세계 x = 0(선수 자리)에 닿는 시각은 scrolled(t_k) = x_k인 t_k다. 첫 장애물은 x에서 출발하고(t_0 =
scrolled⁻¹(x)), 다음 장애물은 t_k = t_{k-1} + U(gap)(최소 0.05초)에 닿도록 x_k = scrolled(t_k)에 둔다. 그래서 줄이 빨라져도
뛸 틈(시간)이 남는다(미터 간격이면 손 조종기 0/30, 초 간격 84/100: 설계 측정).

결정성: 난수는 np.random.default_rng(seed) 하나에서 정해진 순서로만 뽑는다: 장애물마다 (간격 →) 폭 → 공중인가 → (공중이면
두께 → 높이, 아니면 높이). scrolled⁻¹는 이분법 60번이다. 같은 인자는 프로세스와 상관없이 같은 줄을 낸다.

줄이 멈추는 경우(accel < 0으로 속력이 0이 되거나, speed와 accel이 모두 0 이하): 멈추기 전에 x = 0에 닿을 수 없는
장애물(t_k가 멈춤 시각 이상)은 만들지 않는다(그런 장애물은 영영 오지 않는다. 모두 멈춘 자리에 쌓아 두면 선수 자리에서
멈춘 장애물 더미가 된다). 첫 장애물은 늘 x에 만든다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MIN_GAP = 0.05
"""간격(초)의 하한."""
MIN_SIZE = 0.05
"""장애물 폭·높이·두께(m)의 하한."""
LENGTH_TAIL = 1.0
"""줄 길이 = 마지막 장애물 오른쪽 끝 + 이만큼(m): 마지막 장애물이 선수 자리를 1m 지난 뒤 줄이 멈춘다."""
_BISECT_STEPS = 60
_BISECT_MAX_TIME = 1e6


@dataclass(frozen=True)
class ObstacleRow:
    """장애물 줄 하나(줄 좌표, 움직이기 전)."""

    boxes: list[tuple[float, float, float, float]]
    """장애물마다 (x0, y0, x1, y1): 왼쪽 아래와 오른쪽 위 모서리."""
    air: list[bool]
    """장애물마다 공중 장애물인가(모양 태그 obstacle과 air)."""
    length: float
    """줄이 밀려 가는 길의 길이(m)."""


def scrolled_distance(t: float, speed: float, accel: float) -> float:
    """t초 동안 줄이 민 거리. 속력 = speed + accel·t이고 0 아래로 내려가면 그 자리에 멈춘다(motion.pose_at과 같은 식)."""
    if accel < 0.0 and speed + accel * t <= 0.0:
        t = speed / -accel
    return speed * t + 0.5 * accel * t * t


def _stop_time(speed: float, accel: float) -> float:
    """줄이 멈추는 시각(초). 멈추지 않으면 무한대, 처음부터 움직이지 않으면 0."""
    if accel < 0.0:
        return speed / -accel
    if accel == 0.0 and speed <= 0.0:
        return 0.0
    return math.inf


def _time_at(dist: float, speed: float, accel: float) -> float:
    """scrolled_distance(t) = dist가 되는 t(이분법, 결정적). 닿지 못하면 이분법 상한 근처 값."""
    lo, hi = 0.0, 1.0
    while scrolled_distance(hi, speed, accel) < dist and hi < _BISECT_MAX_TIME:
        hi *= 2.0
    for _ in range(_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if scrolled_distance(mid, speed, accel) < dist:
            lo = mid
        else:
            hi = mid
    return hi


def generate_obstacles(seed: int, *, count: int, x: float, y: float, gap: tuple[float, float],
                       width: tuple[float, float], height: tuple[float, float], air_chance: float,
                       air_y: tuple[float, float], air_height: tuple[float, float], speed: float,
                       accel: float) -> ObstacleRow:
    """장애물 줄을 만든다. 범위 인자는 (a, b) 순서와 상관없이 [min, max]에서 고르게 뽑는다.

    seed는 63비트로 자른다. speed는 0 아래를 0으로, air_chance는 [0, 1]로 자른다. 공중 장애물의 아랫면은 바닥 y +
    U(air_y)(0 이상), 땅 장애물은 바닥 y에 선다.
    """
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFFFFFFFFFF)
    speed = max(0.0, float(speed))
    accel = float(accel)
    air_p = min(max(float(air_chance), 0.0), 1.0) if air_chance == air_chance else 0.0

    def pick(r: tuple[float, float], lo: float = 0.0) -> float:
        a, b = float(r[0]), float(r[1])
        return max(lo, float(rng.uniform(min(a, b), max(a, b))))

    t_stop = _stop_time(speed, accel)
    boxes: list[tuple[float, float, float, float]] = []
    air: list[bool] = []
    t_k = _time_at(max(0.0, float(x)), speed, accel)
    xk = float(x)
    w = 0.0
    for k in range(int(count)):
        if k:
            t_k += pick(gap, MIN_GAP)
            if t_k >= t_stop:
                break  # 줄이 멈추기 전에 선수 자리에 닿지 못한다
            xk = scrolled_distance(t_k, speed, accel)
        w = pick(width, MIN_SIZE)
        is_air = float(rng.uniform()) < air_p
        if is_air:
            h = pick(air_height, MIN_SIZE)
            y0 = y + pick(air_y)
        else:
            h = pick(height, MIN_SIZE)
            y0 = y
        boxes.append((xk, y0, xk + w, y0 + h))
        air.append(is_air)
    return ObstacleRow(boxes, air, xk + w + LENGTH_TAIL)


__all__ = ["LENGTH_TAIL", "MIN_GAP", "MIN_SIZE", "ObstacleRow", "generate_obstacles", "scrolled_distance"]

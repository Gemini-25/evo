"""착륙선용 손 조종기. 환경이 실제로 착륙 가능한지 확인하고 처리량을 재는 데 쓴다(학습과 무관).

관측값만 보고 움직인다: 기울여서 수평 이동, 측면 추진기로 자세, 주 추진기로 내려오는 속도.
패드에서 멀면 일정 높이를 지키며 다가가고, 다리가 닿으면 추진을 끈다. 시드 500개에서 약 60% 착륙하고,
실패는 거의 모두 20초 안에 패드까지 못 가는 시간 초과다.
"""

from __future__ import annotations

import math

import numpy as np

MASS = 107.2
G = 1.62
MAIN_FORCE = 350.0


def clip(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def pilot(obs: np.ndarray) -> list[float]:
    dx, dy = float(obs[0]) * 20.0, float(obs[1]) * 20.0
    vx, vy = float(obs[2]) * 10.0, float(obs[3]) * 10.0
    ang, om = float(obs[4]) * math.pi, float(obs[5]) * 2.0
    height = -dy

    vx_des = clip(0.3 * dx, -4.0, 4.0)
    ang_des = clip(-0.24 * (vx_des - vx), -0.25, 0.25)
    if height < 2.0:
        ang_des = 0.0
    torque = 3.0 * (ang_des - ang) - 2.5 * om
    side_l = clip(-torque, 0.0, 1.0)
    side_r = clip(torque, 0.0, 1.0)

    if abs(dx) > 1.5:
        vy_des = -clip(0.4 * (height - 6.0), -1.0, 3.0)
    else:
        vy_des = -clip(0.35 * height, 0.35, 3.5)
    a_cmd = G + 1.2 * (vy_des - vy)
    main = clip(MASS * a_cmd / (MAIN_FORCE * max(0.5, math.cos(ang))), 0.0, 1.0)
    if float(obs[6]) > 0.5 or float(obs[7]) > 0.5:
        main = side_l = side_r = 0.0
    return [main, side_l, side_r]

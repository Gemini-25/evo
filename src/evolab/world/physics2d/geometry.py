"""런타임과 화면이 함께 쓰는 2D 기하 함수. 파이썬 float만 쓴다(결정성, 작은 입력에서 빠름)."""

from __future__ import annotations

import math
from collections.abc import Sequence

Point = tuple[float, float]

TAU = 2.0 * math.pi


def wrap_angle(a: float) -> float:
    """-pi..pi로 감싼다."""
    return (a + math.pi) % TAU - math.pi


def rotate(p: Point, angle: float) -> Point:
    c, s = math.cos(angle), math.sin(angle)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def transform(p: Point, origin: Point, angle: float) -> Point:
    """물체 좌표 p를 세계 좌표로."""
    x, y = rotate(p, angle)
    return (x + origin[0], y + origin[1])


def box_points(w: float, h: float, offset: Point = (0.0, 0.0), angle: float = 0.0) -> list[Point]:
    hw, hh = w / 2.0, h / 2.0
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [transform(c, offset, angle) for c in corners]


def circle_points(r: float, offset: Point = (0.0, 0.0), n: int = 24) -> list[Point]:
    return [(offset[0] + r * math.cos(TAU * i / n), offset[1] + r * math.sin(TAU * i / n)) for i in range(n)]


def radial_points(radii: Sequence[float], angle: float = 0.0) -> list[Point]:
    n = len(radii)
    return [(r * math.cos(angle + TAU * i / n), r * math.sin(angle + TAU * i / n)) for i, r in enumerate(radii)]


def signed_area(points: Sequence[Point]) -> float:
    a = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return a / 2.0


def is_convex(points: Sequence[Point]) -> bool:
    n = len(points)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        ax, ay = points[i]
        bx, by = points[(i + 1) % n]
        cx, cy = points[(i + 2) % n]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cross) < 1e-12:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return sign != 0


def _segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    """선분 ab와 cd가 끝점이 아닌 곳에서 만나는가(끝점끼리 닿는 것은 제외)."""

    def orient(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1, d2 = orient(c, d, a), orient(c, d, b)
    d3, d4 = orient(a, b, c), orient(a, b, d)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and 0.0 not in (d1, d2, d3, d4)


def is_simple(points: Sequence[Point]) -> bool:
    """이웃하지 않은 변끼리 교차하지 않는 다각형인가(별 모양처럼 꼬인 다각형은 False)."""
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # 첫 변과 마지막 변은 이웃
            if _segments_cross(a, b, points[j], points[(j + 1) % n]):
                return False
    return True


def point_in_polygon(p: Point, poly: Sequence[Point]) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def dist_point_segment(p: Point, a: Point, b: Point) -> float:
    px, py = p
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0.0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy)


def dist_point_polygon(p: Point, poly: Sequence[Point]) -> float:
    """다각형 안이면 0, 밖이면 가장 가까운 변까지 거리."""
    if point_in_polygon(p, poly):
        return 0.0
    n = len(poly)
    return min(dist_point_segment(p, poly[i], poly[(i + 1) % n]) for i in range(n))


def polygon_center(poly: Sequence[Point]) -> Point:
    """넓이 중심. 넓이가 0이면 꼭짓점 평균."""
    a = signed_area(poly)
    if abs(a) < 1e-12:
        n = len(poly)
        return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)
    cx = cy = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    return (cx / (6.0 * a), cy / (6.0 * a))

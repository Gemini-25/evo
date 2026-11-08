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


WING_MIN_CHORD = 0.01
"""날개 모양 시위의 하한(m). 수식이 0 이하를 내도 모양이 꼬이지 않게 자른다(검사는 숫자 칸의 0 이하를 오류로 낸다)."""
WING_MAX_CHORD = 1e4
WING_MIN_THICK = 0.005
"""날개 두께(시위 비율)의 하한. 0보다 크면 위 꼭짓점이 늘 아래 꼭짓점보다 법선 쪽 멀리 있어 단순 다각형이다."""
WING_MAX_THICK = 1.0
WING_MAX_CAMBER = 1.0


def wing_points(le: Point, chord: float, angle: float, thick: float, camber: float) -> list[Point]:
    """날개 모양(v2 설계 4.6)의 꼭짓점 넷: 앞전, 위, 뒷전, 아래(반시계).

    뒷전 = le - chord x (cos a, sin a). 시위 가운데 mid에서 법선 n = (-sin a, cos a) 쪽으로 위 = mid + chord x (camber +
    thick / 2) x n, 아래 = mid + chord x (camber - thick / 2) x n. chord는 [0.01, 1e4], thick은 [0.005, 1], camber는 [-1, 1]로
    자른다. thick > 0이라 어떤 값에서도 꼬이지 않는다(휨이 커서 위·아래가 같은 쪽이면 오목한 화살촉 꼴이지만 여전히 단순).
    런타임(Episode)과 에디터 그림이 같은 함수를 쓴다.
    """
    lx, ly = float(le[0]), float(le[1])
    c = min(max(float(chord), WING_MIN_CHORD), WING_MAX_CHORD) if chord == chord else WING_MIN_CHORD
    th = min(max(float(thick), WING_MIN_THICK), WING_MAX_THICK) if thick == thick else WING_MIN_THICK
    k = min(max(float(camber), -WING_MAX_CAMBER), WING_MAX_CAMBER) if camber == camber else 0.0
    ca, sa = math.cos(angle), math.sin(angle)
    mx, my = lx - 0.5 * c * ca, ly - 0.5 * c * sa
    nx, ny = -sa, ca
    up, lo = c * (k + 0.5 * th), c * (k - 0.5 * th)
    return [(lx, ly), (mx + up * nx, my + up * ny), (lx - c * ca, ly - c * sa), (mx + lo * nx, my + lo * ny)]


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

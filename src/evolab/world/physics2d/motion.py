"""정해진 움직임(kinematic 세계 물체)의 자세 = 시간만의 함수 (v2 설계 4.1).

런타임(Episode)은 스텝마다 다음 스텝 시각의 자세를 pose_at으로 구해 그 자세에 정확히 닿는 속도를 물체에 준다(위치를
덮어쓰지 않으므로 위에 올라탄 물체가 함께 실려 가고 레이·접촉이 그 스텝의 자세와 맞다). 화면(관전, 재생, 경주, 영상,
직접 조종, 에디터 미리보기)은 같은 함수로 그 시각의 자세를 그린다. 그래서 궤적 파일에 움직이는 세계 물체를 담지 않는다.

모든 함수는 순수 함수다(물리 엔진·난수·전역 상태 없음). 수식은 만들 때 계산해 MotionPlan에 숫자로 굳힌다.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass

TAU = 2.0 * math.pi
MIN_PERIOD = 1e-3
"""흔들기 주기의 하한(초). 수식이 0 이하를 내도 자세가 유한하게(검사는 숫자 칸의 0 이하를 오류로 낸다)."""
PATH_MODES = ("once", "loop", "pingpong")

Point = tuple[float, float]


@dataclass(frozen=True)
class MotionPlan:
    """수식을 계산해 숫자로 굳힌 움직임. 자세 = pose_at(plan, t). 만들 때는 MotionPlan.path / MotionPlan.wave를 쓴다."""

    kind: str
    """종류: path(길 따라) 또는 wave(제자리 흔들기)."""
    origin: tuple[float, float]
    """물체 pos(세계 좌표). 길의 점과 흔들림은 이 점 기준이다."""
    angle0: float
    """처음 각도(라디안)."""
    points: tuple[tuple[float, float], ...] = ()
    """path: origin 기준 점들. loop면 끝점 -> 첫 점 구간을 더한 닫힌 길(마지막 점 = 첫 점)."""
    speed: float = 0.0
    """path: 처음 속력(m/s, 0 이상)."""
    accel: float = 0.0
    """path: 초마다 늘어나는 속력(m/s²). 음수면 속력이 0이 될 때 멈춘다."""
    start: float = 0.0
    """path: 이 시각(초)까지 첫 점에 서 있다(0 이상)."""
    mode: str = "once"
    """path: once(끝에서 멈춤), loop(닫힌 길을 돎), pingpong(끝에서 되돌아옴)."""
    amplitude: tuple[float, float] = (0.0, 0.0)
    """wave: 흔들림 폭 (x, y)."""
    period: float = 1.0
    """wave: 주기(초, MIN_PERIOD 이상)."""
    phase: float = 0.0
    """wave: 시작 위상(주기 비율)."""
    spin: float = 0.0
    """제자리 회전 속도(rad/s). 각도 = angle0 + spin x t(path도 출발 시각과 무관하게 0초부터)."""
    cum: tuple[float, ...] = ()
    """path: 점마다 길 시작부터의 거리(마지막 값 = 길 전체 길이)."""

    @staticmethod
    def path(origin: Sequence[float], angle0: float, points: Sequence[Sequence[float]], speed: float, accel: float,
             start: float, mode: str, spin: float) -> MotionPlan:
        """길 따라 움직임. points는 origin 기준(보통 첫 점 (0, 0)). 속력과 출발 시각은 0 아래를 0으로 자른다."""
        if mode not in PATH_MODES:
            raise ValueError(f"unknown path mode: {mode!r}")
        pts = [(float(x), float(y)) for x, y in points]
        if not pts:
            raise ValueError("a path needs at least one point")
        if mode == "loop" and pts[-1] != pts[0]:
            pts.append(pts[0])
        cum = [0.0]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
            cum.append(cum[-1] + math.hypot(x1 - x0, y1 - y0))
        return MotionPlan("path", (float(origin[0]), float(origin[1])), float(angle0), tuple(pts),
                          speed=max(0.0, float(speed)), accel=float(accel), start=max(0.0, float(start)), mode=mode,
                          spin=float(spin), cum=tuple(cum))

    @staticmethod
    def wave(origin: Sequence[float], angle0: float, amplitude: Sequence[float], period: float, phase: float,
             spin: float) -> MotionPlan:
        """제자리 흔들기: 위치 = origin + amplitude x sin(2π(t / period + phase))."""
        return MotionPlan("wave", (float(origin[0]), float(origin[1])), float(angle0),
                          amplitude=(float(amplitude[0]), float(amplitude[1])), period=max(MIN_PERIOD, float(period)),
                          phase=float(phase), spin=float(spin))


def _distance(plan: MotionPlan, t: float) -> float:
    """출발 뒤 길을 따라 간 거리. 속력 = speed + accel x τ이고 0 아래로 내려가지 않는다(내려가면 그 자리에 멈춤)."""
    tau = t - plan.start
    if tau <= 0.0:
        return 0.0
    v0, a = plan.speed, plan.accel
    if a < 0.0:
        t_stop = v0 / -a
        if tau >= t_stop:
            tau = t_stop
    return v0 * tau + 0.5 * a * tau * tau


def _point_at(plan: MotionPlan, s: float) -> Point:
    """길을 따라 s만큼 간 점(origin 기준)."""
    cum, pts = plan.cum, plan.points
    total = cum[-1]
    if total <= 0.0 or s <= 0.0:
        return pts[0]
    if plan.mode == "once":
        s = min(s, total)
    elif plan.mode == "loop":
        s = math.fmod(s, total)
    else:  # pingpong: 갈 때 0 -> total, 올 때 total -> 0
        s = math.fmod(s, 2.0 * total)
        if s > total:
            s = 2.0 * total - s
    k = min(max(bisect.bisect_right(cum, s) - 1, 0), len(pts) - 2)
    seg = cum[k + 1] - cum[k]
    f = 0.0 if seg <= 0.0 else min(1.0, (s - cum[k]) / seg)
    (x0, y0), (x1, y1) = pts[k], pts[k + 1]
    return (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)


def pose_at(plan: MotionPlan, t: float) -> tuple[float, float, float]:
    """시각 t(초)의 자세 (x, y, 각도). t < 0은 0으로 본다. 같은 (plan, t)면 늘 같은 값(순수 함수)."""
    t = max(0.0, float(t))
    ox, oy = plan.origin
    if plan.kind == "path":
        dx, dy = _point_at(plan, _distance(plan, t))
    else:
        k = math.sin(TAU * (t / plan.period + plan.phase))
        dx, dy = plan.amplitude[0] * k, plan.amplitude[1] * k
    return (ox + dx, oy + dy, plan.angle0 + plan.spin * t)


def path_length(plan: MotionPlan) -> float:
    """에디터 「길 길이」(m). path: 점을 잇는 길이(loop는 끝점 -> 첫 점 구간 포함). wave: 양 끝 사이 거리(2 x 폭)."""
    if plan.kind == "path":
        return plan.cum[-1] if plan.cum else 0.0
    return 2.0 * math.hypot(plan.amplitude[0], plan.amplitude[1])


def period_of(plan: MotionPlan) -> float | None:
    """한 바퀴 시간(초). loop: 길이 / 속력, pingpong: 2 x 길이 / 속력, wave: 주기.

    once, 가속이 있는 길, 움직이지 않는 길(길이나 속력이 0)은 되풀이되지 않으므로 None.
    """
    if plan.kind == "wave":
        return plan.period
    length = path_length(plan)
    if plan.mode == "once" or plan.accel != 0.0 or plan.speed <= 0.0 or length <= 0.0:
        return None
    return (length if plan.mode == "loop" else 2.0 * length) / plan.speed


__all__ = ["MIN_PERIOD", "PATH_MODES", "MotionPlan", "path_length", "period_of", "pose_at"]

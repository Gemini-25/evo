"""PX5 원형 세계: 숨바꼭질(scratch 세계, 손으로 짠 pymunk 세계. 제품 런타임이 아니다).

출처: scratchpad/sports/eng/proto/arena.py의 HideWorld(DE 4.9-(5), 9.1-14와 같은 물리·입력·점수). PX5에서 바꾼 것:
  - 모든 무작위 값을 시드를 받은 numpy Generator에서 뽑는다(원형은 random.Random). 무작위 행동 기준 상대도 numpy(PX4와 같은 꼴).
  - 「찾기」(되돌릴 지점 B17) FindWorld: 술래 혼자. 숨는 쪽은 술래의 시작 자리에서 보이지 않는(벽 뒤) 무작위 자리에 가만히 있다.

세계(DE 4.9-(5)): 위에서 봄, 중력 0, 감쇠 0.3, 1/60초, 최대 15초. 경기장 16 x 12m, 벽 6개(길이 2.4~4m, 자리가 시험마다 ±1m).
시야: 두 선수 사이 선분이 벽에 막히지 않으면 「보임」(벽만 불투명, match.opaque = [tag.wall]). 보이면 상대의 지금 자리, 안 보이면 마지막으로 본 자리
(sight·remember). 입력 15: 내 x·y·vx·vy, 보임, 상대(마지막으로 본) dx·dy, 벽 레이 8개(6m). 출력 2(xy 힘 70N, 끌림 14로 최고 5m/s).
잡힘: 두 중심 거리 < 1.0m = 술래 승, 15초 끝 = 숨는 쪽 승.
점수: 술래 = 승패 + 0.3 x 본 시간 비율 + 0.2 x (1 - 평균 거리/16) + 0.3 x 빨리 잡음, 숨는 쪽 = 승패 + 0.3 x 안 보인 비율 + 0.2 x 멀리 + 0.3 x 오래 버팀.

공통 규격(selfplay.py가 쓰는 것): cls.ROLES, cls.N_IN[role], cls.N_OUT[role], cls.seat_roles(), w.agents [(자리, 역할)],
w.observe(), w.step(actions), w.done, w.result() -> (자리마다 점수, 자리마다 승점, info), w.script(agent, kind).
"""
from __future__ import annotations

import math

import numpy as np
import pymunk
from pymunk._chipmunk_cffi import ffi as _ffi
from pymunk._chipmunk_cffi import lib as cp


def st(body: pymunk.Body) -> tuple[float, float, float, float, float, float]:
    raw = body._body
    p = cp.cpBodyGetPosition(raw)
    v = cp.cpBodyGetVelocity(raw)
    return p.x, p.y, cp.cpBodyGetAngle(raw), v.x, v.y, cp.cpBodyGetAngularVelocity(raw)


def clip(x: float, lo: float = -5.0, hi: float = 5.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


def seg_dist(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


class HideWorld:
    """술래(자리 0, 역할 0) 대 숨는 쪽(자리 1, 역할 1)."""

    ROLES = ["seeker", "hider"]
    N_IN = {0: 15, 1: 15}
    N_OUT = {0: 2, 1: 2}
    MAX_T = 15.0
    DTW = 1.0 / 60.0
    HX, HY = 8.0, 6.0
    F, DRAG = 70.0, 14.0
    RAYS, RAY_LEN = 8, 6.0
    CATCH = 1.0
    WALLS = ((-4.0, 2.5, True), (-4.0, -2.5, False), (0.0, 0.0, False), (4.0, 2.5, False), (4.0, -2.5, True), (0.0, 4.0, True))

    @classmethod
    def seat_roles(cls) -> list[int]:
        return [0, 1]

    def __init__(self, seed: int, **_: object) -> None:
        self.rng = np.random.default_rng(seed)
        self._rseed = int(seed)
        self.space = pymunk.Space()
        self.space.collision_slop = 0.01
        self.steps = 0
        self.done = False
        self.max_steps = int(round(self.MAX_T / self.DTW))
        self._first = True
        self._rand_key: dict = {}
        self._rand_act: dict = {}
        sp = self.space
        sp.gravity = (0.0, 0.0)
        sp.damping = 0.3
        HX, HY = self.HX, self.HY
        segs = [((-HX, -HY), (HX, -HY)), ((HX, -HY), (HX, HY)), ((HX, HY), (-HX, HY)), ((-HX, HY), (-HX, -HY))]
        self.inner = []
        for cx, cy, horiz in self.WALLS:
            cx += self.u(-1.0, 1.0)
            cy += self.u(-1.0, 1.0)
            L = self.u(1.2, 2.0)
            s = ((cx - L, cy), (cx + L, cy)) if horiz else ((cx, cy - L), (cx, cy + L))
            segs.append(s)
            self.inner.append(s)
        for a, b in segs:
            w = pymunk.Segment(sp.static_body, a, b, 0.15)
            w.friction = 0.3
            w.filter = pymunk.ShapeFilter(categories=0b10)
            sp.add(w)
        self._filt = _ffi.new("cpShapeFilter *", (0, 0xFFFFFFFF, 0b10))
        self._info = _ffi.new("cpSegmentQueryInfo *")
        self._dirs = [(math.cos(2 * math.pi * k / self.RAYS), math.sin(2 * math.pi * k / self.RAYS)) for k in range(self.RAYS)]
        self.queries = 0
        self.agents = self._agents()
        self.b = []
        for x, y in self._spawns():
            b = pymunk.Body()
            b.position = (x, y)
            c = pymunk.Circle(b, 0.4)
            c.mass = 6.0
            c.friction = 0.3
            c.filter = pymunk.ShapeFilter(categories=0b01)
            sp.add(b, c)
            self.b.append(b)
        self.last = [None, None]  # 선수마다 상대를 마지막으로 본 자리
        self.seen_steps = 0
        self.dist_sum = 0.0
        self.caught = False
        self._vis = False

    def u(self, lo: float, hi: float) -> float:
        return float(self.rng.uniform(lo, hi))

    def _agents(self) -> list[tuple[int, int]]:
        return [(0, 0), (1, 1)]

    def _spawns(self) -> list[tuple[float, float]]:
        out = []
        for x in (-6.5, 6.5):
            px = x + self.u(-0.8, 0.8)
            out.append((px, self.u(-4.5, 4.5)))
        return out

    def _space_step(self) -> None:
        if self._first:
            self.space.step(self.DTW)
            self._first = False
        else:
            cp.cpSpaceStep(self.space._space, self.DTW)
        self.steps += 1

    def _los(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        cp.cpSpaceSegmentQueryFirst(self.space._space, a, b, 0.0, self._filt[0], self._info)
        self.queries += 1
        return self._info.alpha >= 1.0

    def _obs_rows(self, idx: tuple[int, ...]) -> list[list[float]]:
        ps = [st(b) for b in self.b]
        a, h = (ps[0][0], ps[0][1]), (ps[1][0], ps[1][1])
        vis = self._los(a, h)
        self._vis = vis
        out = []
        raw = self.space._space
        for i in (0, 1):
            x, y, _, vx, vy, _ = ps[i]
            o = (ps[1 - i][0], ps[1 - i][1])
            if vis:
                self.last[i] = o
            if i not in idx:
                continue
            lx, ly = self.last[i] if self.last[i] is not None else (x, y)
            row = [x / self.HX, y / self.HY, vx / 5.0, vy / 5.0, 1.0 if vis else 0.0, clip((lx - x) / 8.0), clip((ly - y) / 8.0)]
            for dx, dy in self._dirs:
                cp.cpSpaceSegmentQueryFirst(raw, (x, y), (x + dx * self.RAY_LEN, y + dy * self.RAY_LEN), 0.0, self._filt[0], self._info)
                row.append(self._info.alpha)
            self.queries += self.RAYS
            out.append(row)
        return out

    def observe(self) -> list[list[float]]:
        return self._obs_rows((0, 1))

    def _apply(self, actions: list[list[float]]) -> None:
        for b, a in zip(self.b, actions):
            raw = b._body
            v = cp.cpBodyGetVelocity(raw)
            cp.cpBodySetForce(raw, (a[0] * self.F - self.DRAG * v.x, a[1] * self.F - self.DRAG * v.y))

    def step(self, actions: list[list[float]]) -> None:
        if self.done:
            return
        self._apply(actions)
        self._space_step()
        p, q = cp.cpBodyGetPosition(self.b[0]._body), cp.cpBodyGetPosition(self.b[1]._body)
        d = math.hypot(p.x - q.x, p.y - q.y)
        self.dist_sum += d
        if self._vis:
            self.seen_steps += 1
        if d < self.CATCH:
            self.caught = True
            self.done = True
        elif self.steps >= self.max_steps:
            self.done = True

    def _scores(self) -> tuple[list[float], list[float], dict]:
        n = max(self.steps, 1)
        seen = self.seen_steps / n
        far = min(1.0, (self.dist_sum / n) / 16.0)
        pts = [1.0, 0.0] if self.caught else [0.0, 1.0]
        s0 = pts[0] + 0.3 * seen + 0.2 * (1.0 - far) + (0.3 * (1.0 - self.steps / self.max_steps) if self.caught else 0.0)
        s1 = pts[1] + 0.3 * (1.0 - seen) + 0.2 * far + (0.3 * self.steps / self.max_steps if self.caught else 0.0)
        return [s0, s1], pts, {"steps": self.steps, "end": "caught" if self.caught else "time", "seen": seen}

    def result(self) -> tuple[list[float], list[float], dict]:
        return self._scores()

    def script(self, agent: int, kind: str) -> list[float]:
        n = self.N_OUT[self.agents[agent][1]]
        if kind == "still":
            return [0.0] * n
        if kind == "chase" and agent == 0:  # 마지막으로 본 자리로 곧장(벽은 피하지 않음), 본 적 없으면 가운데로
            x, y = st(self.b[0])[:2]
            lx, ly = self.last[0] if self.last[0] is not None else (0.0, 0.0)
            dx, dy = lx - x, ly - y
            m = math.hypot(dx, dy) or 1.0
            return [dx / m, dy / m]
        if kind == "random":  # 10스텝마다 바뀌는 무작위 행동(세계 시드·선수·구간에서 뽑음)
            key = self.steps // 10
            if self._rand_key.get(agent) != key:
                self._rand_key[agent] = key
                r = np.random.default_rng([self._rseed & 0xFFFFFFFF, agent, key])
                self._rand_act[agent] = r.uniform(-1.0, 1.0, n).tolist()
            return self._rand_act[agent]
        raise ValueError(kind)


class FindWorld(HideWorld):
    """B17 「찾기」: 술래 혼자(자리 0, 역할 0, 입력 15·출력 2). 숨는 쪽은 술래의 시작 자리에서 보이지 않는(벽 뒤) 무작위 자리에 가만히 있다.

    숨는 자리: 경기장 안(가장자리에서 0.6m 안쪽) 고른 분포에서 뽑아 (a) 모든 안쪽 벽 선분에서 0.7m 넘게 떨어지고 (b) 술래 시작 자리에서
    4m 넘게 떨어지고 (c) 술래 시작 자리와의 선분이 벽에 막힌 첫 자리. 2,000번 안에 못 찾으면 (a)(b)만 맞는 마지막 자리(info.hidden = False).
    점수 = 술래 점수 그대로, 성공(승점 1) = 잡음.
    """

    ROLES = ["seeker"]
    N_IN = {0: 15}
    N_OUT = {0: 2}

    @classmethod
    def seat_roles(cls) -> list[int]:
        return [0]

    def _agents(self) -> list[tuple[int, int]]:
        return [(0, 0)]

    def _spawns(self) -> list[tuple[float, float]]:
        sx, sy = -6.5 + self.u(-0.8, 0.8), self.u(-4.5, 4.5)
        self.hidden_spawn = False
        cand = None
        for _ in range(2000):
            hx, hy = self.u(-self.HX + 0.6, self.HX - 0.6), self.u(-self.HY + 0.6, self.HY - 0.6)
            if min(seg_dist((hx, hy), a, b) for a, b in self.inner) <= 0.7 or math.hypot(hx - sx, hy - sy) <= 4.0:
                continue
            cand = (hx, hy)
            if not self._los((sx, sy), (hx, hy)):
                self.hidden_spawn = True
                break
        assert cand is not None
        return [(sx, sy), cand]

    def observe(self) -> list[list[float]]:
        return self._obs_rows((0,))

    def _apply(self, actions: list[list[float]]) -> None:
        super()._apply([actions[0], [0.0, 0.0]])

    def result(self) -> tuple[list[float], list[float], dict]:
        sc, pts, info = self._scores()
        info["hidden_spawn"] = self.hidden_spawn
        return [sc[0]], [pts[0]], info


GAMES = {"hide": HideWorld, "find": FindWorld}

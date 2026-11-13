"""PX2b(축구 1대1 되살리기): PX2의 arena.py + SoccerWorld의 설정 옵션(ball_rx·ball_ry·timeout_ball·draw_pen·w_adv·w_close·w_near).
옵션을 주지 않으면 PX2와 같은 연산·같은 난수 소비다. 아래는 원래 머리글.

P2 대전 원형의 세계 셋(저장소 밖 scratch). pymunk 7.3.1, 상태 읽기·한 스텝은 evolab 런타임처럼 C 함수를 바로 부른다.

공통 규격(물리와 무관한 대전 층 league.py·selfplay.py가 쓰는 것):
    cls.ROLES: 역할 이름 목록, cls.N_IN[role], cls.N_OUT[role]: 뇌 입출력 수
    cls.SEATS: 자리 수(뇌 하나가 맡는 선수 묶음), cls.seat_roles(): 자리마다 역할 번호
    w = cls(seed, **opts); w.agents: [(자리 번호, 역할 번호)] 선수마다
    w.observe() -> 선수마다 관측 목록, w.step(actions), w.done
    w.result() -> (자리마다 점수 목록, 자리마다 승점(1 / 0.5 / 0) 목록, info 사전)
    w.script(agent, kind) -> 행동 목록(기준 상대: still, random 등)
"""
from __future__ import annotations

import math
import random

import pymunk
from pymunk._chipmunk_cffi import lib as cp

DT = 1.0 / 60.0


def st(body: pymunk.Body) -> tuple[float, float, float, float, float, float]:
    raw = body._body
    p = cp.cpBodyGetPosition(raw)
    v = cp.cpBodyGetVelocity(raw)
    return p.x, p.y, cp.cpBodyGetAngle(raw), v.x, v.y, cp.cpBodyGetAngularVelocity(raw)


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clip(x: float, lo: float = -5.0, hi: float = 5.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


def box(body: pymunk.Body, w: float, h: float, mass: float, group: int, friction: float = 0.6,
        offset: tuple[float, float] = (0.0, 0.0)) -> pymunk.Poly:
    hw, hh = w / 2, h / 2
    ox, oy = offset
    s = pymunk.Poly(body, [(ox - hw, oy - hh), (ox + hw, oy - hh), (ox + hw, oy + hh), (ox - hw, oy + hh)])
    s.mass = mass
    s.friction = friction
    s.filter = pymunk.ShapeFilter(group=group)
    return s


class _Base:
    ROLES = ["player"]
    SEATS = 2
    MAX_T = 10.0
    DTW = DT

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.space = pymunk.Space()
        self.space.collision_slop = 0.01
        self.steps = 0
        self.done = False
        self.max_steps = int(round(self.MAX_T / self.DTW))
        self._first = True

    @classmethod
    def seat_roles(cls) -> list[int]:
        return [0] * cls.SEATS

    def _space_step(self) -> None:
        if self._first:
            self.space.step(self.DTW)
            self._first = False
        else:
            cp.cpSpaceStep(self.space._space, self.DTW)
        self.steps += 1

    def script(self, agent: int, kind: str) -> list[float]:
        n = self.N_OUT[self.agents[agent][1]]
        if kind == "still":
            return [0.0] * n
        if kind == "random":  # 10스텝마다 바뀌는 무작위 행동(세계 시드에서)
            key = (agent, self.steps // 10)
            if getattr(self, "_rand_key", {}).get(agent) != key:
                self.__dict__.setdefault("_rand_key", {})[agent] = key
                r = random.Random(hash((self._rseed, agent, self.steps // 10)) & 0xFFFFFFFF)
                self.__dict__.setdefault("_rand_act", {})[agent] = [r.uniform(-1, 1) for _ in range(n)]
            return self._rand_act[agent]
        raise ValueError(kind)


# ---------------------------------------------------------------------------
# (a) 옆에서 본 1대1 밀어내기·권투: 미끄러지는 받침 + 균형 스프링 몸통 + 팔(어깨, 팔꿈치) + 주먹
# ---------------------------------------------------------------------------


class PushWorld(_Base):
    """링(폭 8m) 위 두 선수. 상대를 링 밖으로 밀어내거나(받침이 떨어짐) 쓰러뜨리면(몸통 기울기 1.0rad 넘게 0.3초) 이긴다.

    오른쪽 선수는 왼쪽 선수의 거울상으로 짓고(위치·각도·앵커·한계·모터 부호), 관측도 거울로 바꿔 한 뇌가 양쪽을 맡는다.
    """

    N_IN = {0: 16}
    N_OUT = {0: 3}
    MAX_T = 12.0
    RING = 4.0
    WALK_F = 450.0
    WALK_DRAG = 90.0

    def __init__(self, seed: int, sym: bool = False, decide: float = 0.0, **_: object) -> None:
        super().__init__(seed)
        self._rseed = seed
        self.decide = decide  # 0보다 크면: 시간 끝에 밀어낸 정도(adv) 차가 이보다 크면 승패
        sp = self.space
        sp.gravity = (0.0, -9.81)
        ground = pymunk.Segment(sp.static_body, (-self.RING, 0.0), (self.RING, 0.0), 0.05)
        ground.friction = 0.3
        sp.add(ground)
        self.agents = [(0, 0), (1, 0)]
        self.f: list[dict] = []
        off0 = self.rng.uniform(-0.3, 0.3)
        for i, s in enumerate((1.0, -1.0)):
            off = off0 if (sym or i == 0) else self.rng.uniform(-0.3, 0.3)
            x0 = -s * (1.4 + off)
            self.f.append(self._fighter(i, s, x0))
        self.hits = [0, 0]
        self.ko_steps = [0, 0]
        self.lost = [False, False]
        for i in (0, 1):
            j = 1 - i

            def begin(arb: pymunk.Arbiter, space: pymunk.Space, data: object, i: int = i, j: int = j) -> None:
                fi, tj = st(self.f[i]["fore"]), st(self.f[j]["torso"])
                if math.hypot(fi[3] - tj[3], fi[4] - tj[4]) > 1.5:
                    self.hits[i] += 1

            sp.on_collision(10 + i, 20 + j, begin=begin)

    def _fighter(self, i: int, s: float, x0: float) -> dict:
        sp = self.space
        g = i + 1
        base = pymunk.Body()
        base.position = (x0, 0.05 + 0.15)
        bs = box(base, 0.7, 0.3, 30.0, g, friction=0.3)
        torso = pymunk.Body()
        torso.position = (x0, 0.35 + 0.4)
        ts = box(torso, 0.3, 0.8, 30.0, g)
        ts.collision_type = 20 + i
        a_up = s * -0.5
        upper = pymunk.Body()
        sh = (x0, 0.35 + 0.65)
        upper.position = (sh[0] + s * 0.2 * math.cos(a_up) * s, sh[1] + 0.2 * math.sin(a_up) * s)
        # 팔은 상자(자기 축에 대칭)라 거울상 = 각도 부호와 앵커 x 부호만 바꾼 것
        upper.angle = a_up
        upper.position = (sh[0] + s * 0.2 * math.cos(a_up), sh[1] + s * 0.2 * math.sin(a_up))
        us = box(upper, 0.4, 0.1, 4.0, g)
        a_fo = s * 0.7
        fore = pymunk.Body()
        el = (sh[0] + s * 0.4 * math.cos(a_up), sh[1] + s * 0.4 * math.sin(a_up))
        fore.angle = a_fo
        fore.position = (el[0] + s * 0.2 * math.cos(a_fo), el[1] + s * 0.2 * math.sin(a_fo))
        fs = box(fore, 0.4, 0.1, 3.0, g)
        fist = pymunk.Circle(fore, 0.11, (s * 0.2, 0.0))
        fist.mass = 2.0
        fist.friction = 0.6
        fist.filter = pymunk.ShapeFilter(group=g)
        fist.collision_type = 10 + i
        sp.add(base, bs, torso, ts, upper, us, fore, fs, fist)
        j1 = pymunk.PivotJoint(base, torso, (0.0, 0.15), (0.0, -0.4))
        spring = pymunk.DampedRotarySpring(base, torso, 0.0, 1800.0, 220.0)
        lim = pymunk.RotaryLimitJoint(base, torso, -1.3, 1.3)
        j2 = pymunk.PivotJoint(torso, upper, (0.0, 0.25), (-s * 0.2, 0.0))
        lo, hi = (-1.2, 2.2) if s > 0 else (-2.2, 1.2)
        l2 = pymunk.RotaryLimitJoint(torso, upper, lo, hi)
        m2 = pymunk.SimpleMotor(torso, upper, 0.0)
        m2.max_force = 350.0
        j3 = pymunk.PivotJoint(upper, fore, (s * 0.2, 0.0), (-s * 0.2, 0.0))
        lo, hi = (-0.2, 2.4) if s > 0 else (-2.4, 0.2)
        l3 = pymunk.RotaryLimitJoint(upper, fore, lo, hi)
        m3 = pymunk.SimpleMotor(upper, fore, 0.0)
        m3.max_force = 250.0
        for c in (j1, spring, lim, j2, l2, m2, j3, l3, m3):
            c.collide_bodies = False
        sp.add(j1, spring, lim, j2, l2, m2, j3, l3, m3)
        return {"s": s, "base": base, "torso": torso, "upper": upper, "fore": fore, "m_sh": m2, "m_el": m3, "x0": x0}

    def _fist(self, f: dict) -> tuple[float, float]:
        x, y, a = st(f["fore"])[:3]
        s = f["s"]
        return x + s * 0.2 * math.cos(a), y + s * 0.2 * math.sin(a)

    def observe(self) -> list[list[float]]:
        T = self.max_steps
        states = [(st(f["base"]), st(f["torso"]), st(f["upper"]), st(f["fore"]), self._fist(f)) for f in self.f]
        out = []
        for i in (0, 1):
            f = self.f[i]
            s = f["s"]
            b, t, u, fo, fist = states[i]
            ob, ot, _ou, _ofo, ofist = states[1 - i]
            out.append([
                clip(s * b[3] / 3.0), clip(s * t[2]), clip(s * t[5] / 5.0),
                clip(s * (u[2] - t[2]) / 2.0), clip(s * (fo[2] - u[2]) / 2.0),
                clip((self.RING + s * b[0]) / 4.0), clip((self.RING - s * b[0]) / 4.0),
                clip(s * (ob[0] - b[0]) / 3.0), clip(s * (ob[3] - b[3]) / 3.0), clip(s * ot[2]), clip(s * ot[5] / 5.0),
                clip(s * (ofist[0] - t[0]) / 2.0), clip((ofist[1] - t[1]) / 2.0),
                clip(s * (fist[0] - ot[0]) / 2.0), clip((fist[1] - ot[1]) / 2.0),
                1.0 - self.steps / T,
            ])
        return out

    def step(self, actions: list[list[float]]) -> None:
        if self.done:
            return
        for f, a in zip(self.f, actions):
            s = f["s"]
            base = f["base"]
            raw = base._body
            v = cp.cpBodyGetVelocity(raw)
            cp.cpBodySetForce(raw, (s * a[0] * self.WALK_F - self.WALK_DRAG * v.x, 0.0))
            cp.cpSimpleMotorSetRate(f["m_sh"]._constraint, -s * a[1] * 10.0)
            cp.cpSimpleMotorSetRate(f["m_el"]._constraint, -s * a[2] * 12.0)
        self._space_step()
        for i, f in enumerate(self.f):
            t = st(f["torso"])
            b = st(f["base"])
            self.ko_steps[i] = self.ko_steps[i] + 1 if abs(t[2]) > 1.0 else 0
            if self.ko_steps[i] >= 18 or b[1] < -0.6:
                self.lost[i] = True
        if any(self.lost) or self.steps >= self.max_steps:
            self.done = True

    def result(self) -> tuple[list[float], list[float], dict]:
        l0, l1 = self.lost
        pts = [0.5, 0.5] if l0 == l1 else ([0.0, 1.0] if l0 else [1.0, 0.0])
        x = [st(f["base"])[0] for f in self.f]
        scores = []
        if l0 == l1 and not l0 and self.decide > 0.0:
            a0 = ((x[1] - self.f[1]["x0"]) + (x[0] - self.f[0]["x0"])) / 8.0
            if abs(a0) > self.decide:
                pts = [1.0, 0.0] if a0 > 0 else [0.0, 1.0]
        for i in (0, 1):
            j = 1 - i
            s = self.f[i]["s"]
            adv = s * ((x[j] - self.f[j]["x0"]) + (x[i] - self.f[i]["x0"])) / 8.0
            hd = clip(float(self.hits[i] - self.hits[j]), -3.0, 3.0) / 3.0
            scores.append(pts[i] + 0.2 * clip(adv, -1.0, 1.0) + 0.1 * hd)
        return scores, pts, {"steps": self.steps, "hits": list(self.hits), "end": "ko/out" if any(self.lost) else "time"}

    def script(self, agent: int, kind: str) -> list[float]:
        if kind == "pusher":  # 팔을 앞으로 뻗고 끝까지 민다
            f = self.f[agent]
            s = f["s"]
            t, u, fo = st(f["torso"]), st(f["upper"]), st(f["fore"])
            sh = s * (u[2] - t[2])
            el = s * (fo[2] - u[2])
            return [1.0, clip((0.2 - sh) * 3.0, -1, 1), clip((0.0 - el) * 3.0, -1, 1)]
        return super().script(agent, kind)


# ---------------------------------------------------------------------------
# (b) 위에서 본 축구: 원형 선수(움직임 힘 2축) + 공 + 골대, 중력 없음, 감쇠
# ---------------------------------------------------------------------------


class SoccerWorld(_Base):
    """경기장 20x12m, 골문 폭 4m. 먼저 넣으면 끝(20초). 팀 B는 경기장 가운데로 180도 돌린 좌표로 보고 움직인다(한 뇌가 양쪽)."""

    MAX_T = 20.0
    N_TEAM = 1
    N_IN = {0: 12}
    N_OUT = {0: 2}
    F = 70.0
    DRAG = 14.0  # 선수 선형 감쇠 N/(m/s): 최고 속도 F/DRAG = 5 m/s
    HX, HY, GOAL = 10.0, 6.0, 2.0
    W_ADV, W_CLOSE = 0.25, 0.1
    W_NEAR = 0.0   # 공이 상대 골문 가운데에 가장 가까웠던 거리로 주는 길잡이 점수
    SPAWN_X, SPAWN_DX = 4.0, 3.0
    BRX, BRY, PRX, PRY = 0.2, 0.66, 0.1, 0.5   # 공·선수 시작 자리의 흩어짐(경기장 반폭·반높이에 대한 비율)

    def __init__(self, seed: int, n: int | None = None, sym: int = 0, flip: bool = False, ball_rx: float | None = None,
                 ball_ry: float | None = None, timeout_ball: float | None = None, draw_pen: float = 0.0, w_adv: float | None = None,
                 w_close: float | None = None, w_near: float | None = None, **_: object) -> None:
        """PX2: sym = 0 원형 그대로(선수마다 시작 자리를 따로 뽑음) · 1 = 대칭 시작(팀 A의 자리를 뽑고 팀 B는 경기장 가운데로 180도 돌린
        자리, 공은 뽑은 자리 그대로) · 2 = 1 + 공이 정확히 가운데. flip = 공의 시작 자리를 180도 돌림(회전 확인용: sym 1의 경기를 통째로
        돌린 경기를 만든다). sym 0·flip 거짓이면 난수 소비와 자리가 원형과 같다."""
        super().__init__(seed)
        self._rseed = seed
        # PX2b 옵션. ball_rx/ball_ry: 공 시작 자리의 흩어짐(m, 가운데 기준 ±). 난수는 원래와 같은 두 번만 쓴다.
        # timeout_ball: 시간 끝에 공의 x(m)가 이 값보다 크면 그쪽을 공격하던 팀의 승(Match.timeout_block = 공 위치 블록, timeout_margin).
        # draw_pen: 무승부(시간 끝 0.5)일 때 두 팀의 학습 점수에서 뺌(승점·엘로에는 영향 없음). w_*: 길잡이 가중치 덮어쓰기.
        self.ball_rx, self.ball_ry = ball_rx, ball_ry
        self.timeout_ball, self.draw_pen = timeout_ball, float(draw_pen)
        if w_adv is not None:
            self.W_ADV = float(w_adv)
        if w_close is not None:
            self.W_CLOSE = float(w_close)
        if w_near is not None:
            self.W_NEAR = float(w_near)
        self.n = n or self.N_TEAM
        self.sym, self.flip = int(sym), bool(flip)
        sp = self.space
        sp.gravity = (0.0, 0.0)
        sp.damping = 0.55  # 공: 1초에 속도 55% 남음
        HX, HY, G = self.HX, self.HY, self.GOAL
        walls = [((-HX, HY), (HX, HY)), ((-HX, -HY), (HX, -HY))]
        for sx in (-1.0, 1.0):
            walls += [((sx * HX, G), (sx * HX, HY)), ((sx * HX, -G), (sx * HX, -HY)),
                      ((sx * HX, G), (sx * (HX + 1.5), G)), ((sx * HX, -G), (sx * (HX + 1.5), -G)),
                      ((sx * (HX + 1.5), -G), (sx * (HX + 1.5), G))]
        for a, b in walls:
            w = pymunk.Segment(sp.static_body, a, b, 0.1)
            w.elasticity = 0.8
            w.friction = 0.3
            sp.add(w)
        r = self.rng
        self.ball = pymunk.Body()
        if ball_rx is not None or ball_ry is not None:
            rx = self.BRX * HX if ball_rx is None else float(ball_rx)
            ry = self.BRY * HY if ball_ry is None else float(ball_ry)
            bx0, by0 = r.uniform(-rx, rx), r.uniform(-ry, ry)
        else:
            bx0, by0 = r.uniform(-self.BRX, self.BRX) * HX, r.uniform(-self.BRY, self.BRY) * HY
        if self.sym == 2:
            bx0, by0 = 0.0, 0.0
        if self.flip:
            bx0, by0 = -bx0, -by0
        self.ball.position = (bx0, by0)
        bs = pymunk.Circle(self.ball, 0.25)
        bs.mass = 0.45
        bs.elasticity = 0.7
        bs.friction = 0.3
        sp.add(self.ball, bs)
        self.players: list[tuple[pymunk.Body, float, int]] = []  # (물체, 팀 부호, 자리)
        self.agents = []
        first: list[tuple[float, float]] = []
        for team, s in enumerate((1.0, -1.0)):
            for k in range(self.n):
                b = pymunk.Body()
                if self.sym and team == 1:  # 대칭 시작: 팀 A의 같은 번호 선수를 가운데로 180도 돌린 자리(난수를 더 쓰지 않음)
                    pos = (-first[k][0], -first[k][1])
                else:
                    pos = (-s * (self.SPAWN_X + self.SPAWN_DX * k + r.uniform(-self.PRX, self.PRX) * HX), r.uniform(-self.PRY, self.PRY) * HY)
                    first.append(pos)
                b.position = pos
                c = pymunk.Circle(b, 0.45)
                c.mass = 6.0
                c.elasticity = 0.2
                c.friction = 0.3
                sp.add(b, c)
                self.players.append((b, s, team))
                self.agents.append((team, 0))
        self.goal = 0  # +1: 팀 A 득점, -1: 팀 B 득점
        self.adv = 0.0
        self.close = [0.0, 0.0]
        self.touch = [0, 0]
        self.near = [1e9, 1e9]

    def observe(self) -> list[list[float]]:
        bx, by, _, bvx, bvy, _ = st(self.ball)
        ps = [st(b) for b, _, _ in self.players]
        out = []
        for i, (_, s, team) in enumerate(self.players):
            x, y, _, vx, vy, _ = ps[i]
            o = [s * x / 10.0, s * y / 6.0, s * vx / 6.0, s * vy / 6.0,
                 s * (bx - x) / 10.0, s * (by - y) / 10.0, s * bvx / 8.0, s * bvy / 8.0]
            mates = [j for j, p in enumerate(self.players) if p[2] == team and j != i]
            opps = [j for j, p in enumerate(self.players) if p[2] != team]
            for j in opps + mates:
                q = ps[j]
                o += [s * (q[0] - x) / 10.0, s * (q[1] - y) / 10.0, s * q[3] / 6.0, s * q[4] / 6.0]
            out.append([clip(v) for v in o])
        return out

    def step(self, actions: list[list[float]]) -> None:
        if self.done:
            return
        for (b, s, _), a in zip(self.players, actions):
            raw = b._body
            v = cp.cpBodyGetVelocity(raw)
            cp.cpBodySetForce(raw, (s * a[0] * self.F - self.DRAG * v.x, s * a[1] * self.F - self.DRAG * v.y))
        self._space_step()
        p = cp.cpBodyGetPosition(self.ball._body)
        self.adv += p.x
        self.near[0] = min(self.near[0], math.hypot(p.x - self.HX, p.y))
        self.near[1] = min(self.near[1], math.hypot(p.x + self.HX, p.y))
        for team in (0, 1):
            d = 1e9
            for b, _, t in self.players:
                if t == team:
                    q = cp.cpBodyGetPosition(b._body)
                    d = min(d, math.hypot(q.x - p.x, q.y - p.y))
            self.close[team] += d
            if d < 0.75:
                self.touch[team] += 1
        if abs(p.x) > self.HX + 0.25 and abs(p.y) < self.GOAL:
            self.goal = 1 if p.x > 0 else -1
            self.done = True
        elif self.steps >= self.max_steps:
            self.done = True

    def result(self) -> tuple[list[float], list[float], dict]:
        pts = [0.5, 0.5] if self.goal == 0 else ([1.0, 0.0] if self.goal > 0 else [0.0, 1.0])
        end = "goal" if self.goal else "time"
        if self.goal == 0 and self.timeout_ball is not None:  # PX2b: 시간 끝을 공 위치로 판정
            bx = st(self.ball)[0]
            if bx > self.timeout_ball:
                pts, end = [1.0, 0.0], "time_ball"
            elif bx < -self.timeout_ball:
                pts, end = [0.0, 1.0], "time_ball"
        n = max(self.steps, 1)
        scores = []
        for team, s in enumerate((1.0, -1.0)):
            adv = clip(s * (self.adv / n) / self.HX, -1.0, 1.0)
            close = 1.0 - min(1.0, (self.close[team] / n) / (1.2 * self.HX))
            near = 1.0 - min(1.0, self.near[team] / self.HX)
            sc = pts[team] + self.W_ADV * adv + self.W_CLOSE * close + self.W_NEAR * near
            if pts[team] == 0.5 and self.draw_pen:
                sc -= self.draw_pen
            scores.append(sc)
        return scores, pts, {"steps": self.steps, "end": end, "touch": list(self.touch)}

    def script(self, agent: int, kind: str) -> list[float]:
        if kind == "chaser":  # 공 뒤로 돌아가 상대 골문 쪽으로 민다
            b, s, _ = self.players[agent]
            x, y = st(b)[:2]
            bx, by = st(self.ball)[:2]
            gx, gy = s * self.HX, 0.0
            dx, dy = gx - bx, gy - by
            dn = math.hypot(dx, dy) or 1.0
            dx, dy = dx / dn, dy / dn
            tx, ty = bx - 0.6 * dx, by - 0.6 * dy
            ex, ey = tx - x, ty - y
            en = math.hypot(ex, ey)
            if en > 0.35:
                ux, uy = ex / en, ey / en
            else:
                ux, uy = dx, dy
            return [s * ux, s * uy]
        return super().script(agent, kind)


class Soccer2(SoccerWorld):
    N_TEAM = 2
    N_IN = {0: 20}


class SoccerEasy(SoccerWorld):
    """쉬운 판: 작은 경기장(14x9m), 12초, 길잡이 점수를 키움(공 전진 0.5, 공에 가까움 0.3)."""

    MAX_T = 12.0
    HX, HY, GOAL = 7.0, 4.5, 2.0
    W_ADV, W_CLOSE = 0.5, 0.3
    SPAWN_X, SPAWN_DX = 3.0, 2.5


class Soccer2Easy(SoccerEasy):
    N_TEAM = 2
    N_IN = {0: 20}


class SoccerKick(SoccerEasy):
    """킥오프 판: 쉬운 판 + 공이 가운데 근처(±0.3m, ±0.9m), 선수가 공 뒤쪽(y ±0.9m)에서 시작. 「앞으로 밀기」만으로도 공이 움직인다."""

    BRX, BRY, PRX, PRY = 0.04, 0.2, 0.07, 0.2


class Soccer2Kick(SoccerKick):
    N_TEAM = 2
    N_IN = {0: 20}


class SoccerNear(SoccerKick):
    """킥오프 판 + 길잡이 점수를 「공이 상대 골문에 가장 가까웠던 거리」(0.5)와 공에 가까움(0.2)으로(공 전진 평균은 뺌)."""

    W_ADV, W_CLOSE, W_NEAR = 0.0, 0.2, 0.5


class Soccer2Near(SoccerNear):
    N_TEAM = 2
    N_IN = {0: 20}


class Soccer2NearR(Soccer2Near):
    """PX3: DE 4.9-(3)의 2대2 센서 그대로 — 상대 둘을 `other` 순위 0·1(그 선수에게서 가까운 순서, 같으면 번호 순)로 넣는다.
    Soccer2Near(원형)는 상대를 번호 순으로 넣는다. 그 밖의 연산·난수 소비는 Soccer2Near와 같다."""

    def observe(self) -> list[list[float]]:
        bx, by, _, bvx, bvy, _ = st(self.ball)
        ps = [st(b) for b, _, _ in self.players]
        out = []
        for i, (_, s, team) in enumerate(self.players):
            x, y, _, vx, vy, _ = ps[i]
            o = [s * x / 10.0, s * y / 6.0, s * vx / 6.0, s * vy / 6.0,
                 s * (bx - x) / 10.0, s * (by - y) / 10.0, s * bvx / 8.0, s * bvy / 8.0]
            mates = [j for j, p in enumerate(self.players) if p[2] == team and j != i]
            opps = [j for j, p in enumerate(self.players) if p[2] != team]
            opps.sort(key=lambda j: ((ps[j][0] - x) ** 2 + (ps[j][1] - y) ** 2, j))
            for j in opps + mates:
                q = ps[j]
                o += [s * (q[0] - x) / 10.0, s * (q[1] - y) / 10.0, s * q[3] / 6.0, s * q[4] / 6.0]
            out.append([clip(v) for v in o])
        return out


# ---------------------------------------------------------------------------
# (c) 옆에서 본 투수 대 타자: 역할이 다른 두 뇌
# ---------------------------------------------------------------------------


class PitchWorld(_Base):
    """투수(어깨·팔꿈치 모터 + 놓기)가 10m 앞 타자(어깨·손목 모터, 배트)에게 던진다.

    투수 점수: 스트라이크(스트라이크 존 통과, 안 맞음) 1.0, 맞으면 0.5 - 0.5·min(1, 날아간 거리 / 12), 볼이면 0.3·(1 - 존까지
    가장 가까웠던 거리 / 8). 타자 점수 = 1 - 투수 점수. 승점: 스트라이크면 투수 1, 맞혀서 3m 넘게 보내면 타자 1, 그 밖은 0.5.
    """

    ROLES = ["pitcher", "batter"]
    N_IN = {0: 7, 1: 9}
    N_OUT = {0: 3, 1: 2}
    MAX_T = 6.0
    PLATE_X = 9.7
    ZONE = (0.7, 1.5)
    BX = 10.0       # 타자 어깨 x
    PASS_X = 11.5   # 여기를 지나면 타자를 지나친 공
    W1, W2 = 14.0, 18.0   # 투수 어깨·팔꿈치 최고 각속도

    @classmethod
    def seat_roles(cls) -> list[int]:
        return [0, 1]

    def __init__(self, seed: int, noise: bool = True, ball_w: float = 0.3, hit_floor: float = 0.0, ball_loss: bool = False,
                 **_: object) -> None:
        super().__init__(seed)
        self._rseed = seed
        self.ball_w, self.hit_floor = ball_w, hit_floor  # 볼의 길잡이 점수 상한, 맞았을 때 투수 점수의 바닥
        self.ball_loss = ball_loss  # 참이면 볼은 타자의 승(승점 0·1), 거짓이면 무승부
        r = self.rng
        sp = self.space
        sp.gravity = (0.0, -9.81)
        ground = pymunk.Segment(sp.static_body, (-30.0, 0.0), (16.0, 0.0), 0.05)
        ground.friction = 0.6
        ground.elasticity = 0.4
        ground.collision_type = 3
        sp.add(ground)
        self.agents = [(0, 0), (1, 1)]
        self.noise = r.uniform(-1.0, 1.0) if noise else 0.0
        # 투수
        S = (0.0, 1.7)
        a1 = -2.2 + r.uniform(-0.2, 0.2)
        a2 = a1 + 0.3
        self.p_up = pymunk.Body()
        self.p_up.angle = a1
        self.p_up.position = (S[0] + 0.25 * math.cos(a1), S[1] + 0.25 * math.sin(a1))
        el = (S[0] + 0.5 * math.cos(a1), S[1] + 0.5 * math.sin(a1))
        self.p_fo = pymunk.Body()
        self.p_fo.angle = a2
        self.p_fo.position = (el[0] + 0.25 * math.cos(a2), el[1] + 0.25 * math.sin(a2))
        hand = (el[0] + 0.5 * math.cos(a2), el[1] + 0.5 * math.sin(a2))
        self.ball = pymunk.Body()
        self.ball.position = hand
        bs = pymunk.Circle(self.ball, 0.08)
        bs.mass = 0.25
        bs.elasticity = 0.5
        bs.friction = 0.5
        bs.filter = pymunk.ShapeFilter(group=1)
        bs.collision_type = 1
        sp.add(self.p_up, box(self.p_up, 0.5, 0.08, 3.0, 1), self.p_fo, box(self.p_fo, 0.5, 0.08, 2.0, 1), self.ball, bs)
        j1 = pymunk.PivotJoint(sp.static_body, self.p_up, S, (-0.25, 0.0))
        self.pm1 = pymunk.SimpleMotor(sp.static_body, self.p_up, 0.0)
        self.pm1.max_force = 400.0
        j2 = pymunk.PivotJoint(self.p_up, self.p_fo, (0.25, 0.0), (-0.25, 0.0))
        l2 = pymunk.RotaryLimitJoint(self.p_up, self.p_fo, -0.2, 2.4)
        self.pm2 = pymunk.SimpleMotor(self.p_up, self.p_fo, 0.0)
        self.pm2.max_force = 250.0
        self.grip = pymunk.PivotJoint(self.p_fo, self.ball, (0.25, 0.0), (0.0, 0.0))
        for c in (j1, self.pm1, j2, l2, self.pm2, self.grip):
            c.collide_bodies = False
        sp.add(j1, self.pm1, j2, l2, self.pm2, self.grip)
        # 타자
        B = (self.BX, 1.75)
        b1 = -0.4 + r.uniform(-0.1, 0.1)
        b2 = b1 + 0.3
        self.b_up = pymunk.Body()
        self.b_up.angle = b1
        self.b_up.position = (B[0] + 0.175 * math.cos(b1), B[1] + 0.175 * math.sin(b1))
        wr = (B[0] + 0.35 * math.cos(b1), B[1] + 0.35 * math.sin(b1))
        self.bat = pymunk.Body()
        self.bat.angle = b2
        self.bat.position = (wr[0] + 0.5 * math.cos(b2), wr[1] + 0.5 * math.sin(b2))
        bat_s = box(self.bat, 1.0, 0.07, 2.0, 2)
        bat_s.elasticity = 0.6
        bat_s.collision_type = 2
        sp.add(self.b_up, box(self.b_up, 0.35, 0.08, 3.0, 2), self.bat, bat_s)
        k1 = pymunk.PivotJoint(sp.static_body, self.b_up, B, (-0.175, 0.0))
        self.bm1 = pymunk.SimpleMotor(sp.static_body, self.b_up, 0.0)
        self.bm1.max_force = 500.0
        k2 = pymunk.PivotJoint(self.b_up, self.bat, (0.175, 0.0), (-0.5, 0.0))
        kl = pymunk.RotaryLimitJoint(self.b_up, self.bat, -1.2, 1.2)
        self.bm2 = pymunk.SimpleMotor(self.b_up, self.bat, 0.0)
        self.bm2.max_force = 250.0
        for c in (k1, self.bm1, k2, kl, self.bm2):
            c.collide_bodies = False
        sp.add(k1, self.bm1, k2, kl, self.bm2)
        self.B = B
        self.held = True
        self.hit = False
        self.hit_step = 0
        self.crossed: float | None = None  # 존 평면을 지날 때의 높이
        self.closest = 99.0
        self.land_x: float | None = None
        self.grounded = False
        self.passed = False

        def on_bat(arb: pymunk.Arbiter, space: pymunk.Space, data: object) -> None:
            if not self.held and not self.passed and not self.hit:
                self.hit = True
                self.hit_step = self.steps

        def on_ground(arb: pymunk.Arbiter, space: pymunk.Space, data: object) -> None:
            if not self.held:
                self.grounded = True

        sp.on_collision(1, 2, begin=on_bat)
        sp.on_collision(1, 3, begin=on_ground)

    def observe(self) -> list[list[float]]:
        u, f = st(self.p_up), st(self.p_fo)
        t = self.steps * self.DTW
        held = 1.0 if self.held else 0.0
        po = [wrap(u[2]) / math.pi, clip((f[2] - u[2]) / 2.4), clip(u[5] / self.W1), clip((f[5] - u[5]) / self.W2), held,
              clip(t / 2.0), self.noise]
        a, b, ball = st(self.b_up), st(self.bat), st(self.ball)
        bo = [wrap(a[2]) / math.pi, clip((b[2] - a[2]) / 1.2), clip(a[5] / 14.0),
              clip((ball[0] - self.B[0]) / 10.0), clip((ball[1] - self.B[1]) / 3.0), clip(ball[3] / 20.0),
              clip(ball[4] / 10.0), held, clip(t / 6.0)]
        return [po, bo]

    def step(self, actions: list[list[float]]) -> None:
        if self.done:
            return
        pa, ba = actions
        cp.cpSimpleMotorSetRate(self.pm1._constraint, -pa[0] * self.W1)
        cp.cpSimpleMotorSetRate(self.pm2._constraint, -pa[1] * self.W2)
        t = self.steps * self.DTW
        if self.held and ((pa[2] < 0.0 and t >= 0.2) or t >= 2.0):
            self.held = False
            self.grip.max_force = 0.0  # 관절을 빼지 않고 힘만 0으로(공간 구성을 바꾸지 않음)
        cp.cpSimpleMotorSetRate(self.bm1._constraint, -ba[0] * 14.0)
        cp.cpSimpleMotorSetRate(self.bm2._constraint, -ba[1] * 12.0)
        x0 = cp.cpBodyGetPosition(self.ball._body).x
        self._space_step()
        p = cp.cpBodyGetPosition(self.ball._body)
        if not self.held:
            if not self.hit and not self.grounded:
                zc = (self.ZONE[0] + self.ZONE[1]) / 2
                self.closest = min(self.closest, math.hypot(p.x - self.PLATE_X, p.y - zc))
                if self.crossed is None and x0 < self.PLATE_X <= p.x:
                    self.crossed = p.y
            if not self.hit and p.x > self.PASS_X:
                self.passed = True
                self.done = True
            if self.hit and (self.grounded or (self.steps - self.hit_step) * self.DTW > 4.0 or p.x < -25.0):
                self.land_x = p.x
                self.done = True
            if not self.hit and self.grounded:
                self.done = True  # 맞지 않고 땅에 떨어진 공(존을 지났으면 스트라이크·볼은 이미 정해짐)
        if self.steps >= self.max_steps:
            self.done = True

    def result(self) -> tuple[list[float], list[float], dict]:
        strike = (self.crossed is not None and self.ZONE[0] <= self.crossed <= self.ZONE[1] and not self.hit)
        if self.hit:
            lx = self.land_x if self.land_x is not None else cp.cpBodyGetPosition(self.ball._body).x
            dist = max(0.0, self.PLATE_X - lx)
            ps = 0.5 - (0.5 - self.hit_floor) * min(1.0, dist / 12.0)
            pts = [0.0, 1.0] if dist > 3.0 else [0.5, 0.5]
            end = "hit"
        elif strike:
            ps, pts, end, dist = 1.0, [1.0, 0.0], "strike", 0.0
        else:
            ps = self.ball_w * (1.0 - min(1.0, self.closest / 8.0))
            pts, end, dist = ([0.0, 1.0] if self.ball_loss else [0.5, 0.5]), "ball", 0.0
        return [ps, 1.0 - ps], pts, {"steps": self.steps, "end": end, "dist": dist, "closest": self.closest}

    def script(self, agent: int, kind: str) -> list[float]:
        if kind == "swing" and agent == 1:  # 공이 가까워지면 아래로 휘두르는 타자
            ball = st(self.ball)
            if not self.held and ball[0] > self.PLATE_X - 3.7:
                return [-1.0, -0.3]
            return [0.0, 0.0]
        return super().script(agent, kind)


class PitchEasy(PitchWorld):
    """쉬운 판: 조종 120Hz(놓는 순간의 각도 눈금을 절반으로), 느린 팔(8·10 rad/s), 가까운 타자(7m), 큰 존(1.2m)."""

    DTW = 1.0 / 120.0
    PLATE_X = 7.0
    ZONE = (0.6, 1.8)
    BX = 7.3
    PASS_X = 8.8
    W1, W2 = 8.0, 10.0



# ---------------------------------------------------------------------------
# (d) 위에서 본 숨바꼭질: 벽 + 시야(선분 질의), 술래 대 숨는 사람(역할이 다른 두 뇌)
# ---------------------------------------------------------------------------

from pymunk._chipmunk_cffi import ffi as _ffi  # noqa: E402


class HideWorld(_Base):
    """경기장 16x12m에 벽 6개(시험마다 자리가 조금 다름). 술래가 15초 안에 1m 안으로 다가가면(잡음) 술래 승, 아니면 숨는 쪽 승.

    둘 다 벽 레이 8개(6m)와 상대 자리(보이면 지금 자리, 안 보이면 마지막으로 본 자리)와 보임 여부를 본다.
    """

    ROLES = ["seeker", "hider"]
    N_IN = {0: 15, 1: 15}
    N_OUT = {0: 2, 1: 2}
    MAX_T = 15.0
    HX, HY = 8.0, 6.0
    F, DRAG = 70.0, 14.0
    RAYS, RAY_LEN = 8, 6.0

    @classmethod
    def seat_roles(cls) -> list[int]:
        return [0, 1]

    def __init__(self, seed: int, **_: object) -> None:
        super().__init__(seed)
        self._rseed = seed
        r = self.rng
        sp = self.space
        sp.gravity = (0.0, 0.0)
        sp.damping = 0.3
        HX, HY = self.HX, self.HY
        segs = [((-HX, -HY), (HX, -HY)), ((HX, -HY), (HX, HY)), ((HX, HY), (-HX, HY)), ((-HX, HY), (-HX, -HY))]
        for cx, cy, horiz in ((-4.0, 2.5, True), (-4.0, -2.5, False), (0.0, 0.0, False), (4.0, 2.5, False),
                              (4.0, -2.5, True), (0.0, 4.0, True)):
            cx += r.uniform(-1.0, 1.0)
            cy += r.uniform(-1.0, 1.0)
            L = r.uniform(1.2, 2.0)
            segs.append(((cx - L, cy), (cx + L, cy)) if horiz else ((cx, cy - L), (cx, cy + L)))
        for a, b in segs:
            w = pymunk.Segment(sp.static_body, a, b, 0.15)
            w.friction = 0.3
            w.filter = pymunk.ShapeFilter(categories=0b10)
            sp.add(w)
        self.agents = [(0, 0), (1, 1)]
        self.b = []
        for i, x in enumerate((-6.5, 6.5)):
            b = pymunk.Body()
            b.position = (x + r.uniform(-0.8, 0.8), r.uniform(-4.5, 4.5))
            c = pymunk.Circle(b, 0.4)
            c.mass = 6.0
            c.friction = 0.3
            c.filter = pymunk.ShapeFilter(categories=0b01)
            sp.add(b, c)
            self.b.append(b)
        self._filt = _ffi.new("cpShapeFilter *", (0, 0xFFFFFFFF, 0b10))
        self._info = _ffi.new("cpSegmentQueryInfo *")
        self._dirs = [(math.cos(2 * math.pi * k / self.RAYS), math.sin(2 * math.pi * k / self.RAYS)) for k in range(self.RAYS)]
        self.last = [None, None]  # 선수마다 상대를 마지막으로 본 자리
        self.seen_steps = 0
        self.dist_sum = 0.0
        self.caught = False
        self.queries = 0

    def _los(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        cp.cpSpaceSegmentQueryFirst(self.space._space, a, b, 0.0, self._filt[0], self._info)
        self.queries += 1
        return self._info.alpha >= 1.0

    def observe(self) -> list[list[float]]:
        ps = [st(b) for b in self.b]
        a, h = (ps[0][0], ps[0][1]), (ps[1][0], ps[1][1])
        vis = self._los(a, h)
        self._vis = vis
        out = []
        for i in (0, 1):
            x, y, _, vx, vy, _ = ps[i]
            o = (ps[1 - i][0], ps[1 - i][1])
            if vis:
                self.last[i] = o
            lx, ly = self.last[i] if self.last[i] is not None else (x, y)
            row = [x / self.HX, y / self.HY, vx / 5.0, vy / 5.0, 1.0 if vis else 0.0, clip((lx - x) / 8.0), clip((ly - y) / 8.0)]
            raw = self.space._space
            for dx, dy in self._dirs:
                cp.cpSpaceSegmentQueryFirst(raw, (x, y), (x + dx * self.RAY_LEN, y + dy * self.RAY_LEN), 0.0, self._filt[0], self._info)
                row.append(self._info.alpha)
            self.queries += self.RAYS
            out.append(row)
        return out

    def step(self, actions: list[list[float]]) -> None:
        if self.done:
            return
        for b, a in zip(self.b, actions):
            raw = b._body
            v = cp.cpBodyGetVelocity(raw)
            cp.cpBodySetForce(raw, (a[0] * self.F - self.DRAG * v.x, a[1] * self.F - self.DRAG * v.y))
        self._space_step()
        p, q = cp.cpBodyGetPosition(self.b[0]._body), cp.cpBodyGetPosition(self.b[1]._body)
        d = math.hypot(p.x - q.x, p.y - q.y)
        self.dist_sum += d
        if getattr(self, "_vis", False):
            self.seen_steps += 1
        if d < 1.0:
            self.caught = True
            self.done = True
        elif self.steps >= self.max_steps:
            self.done = True

    def result(self) -> tuple[list[float], list[float], dict]:
        n = max(self.steps, 1)
        seen = self.seen_steps / n
        far = min(1.0, (self.dist_sum / n) / 16.0)
        pts = [1.0, 0.0] if self.caught else [0.0, 1.0]
        s0 = pts[0] + 0.3 * seen + 0.2 * (1.0 - far) + (0.3 * (1.0 - self.steps / self.max_steps) if self.caught else 0.0)
        s1 = pts[1] + 0.3 * (1.0 - seen) + 0.2 * far + (0.3 * self.steps / self.max_steps if self.caught else 0.0)
        return [s0, s1], pts, {"steps": self.steps, "end": "caught" if self.caught else "time", "seen": seen}

    def script(self, agent: int, kind: str) -> list[float]:
        if kind == "chase" and agent == 0:  # 마지막으로 본 자리로 곧장(벽은 피하지 않음)
            x, y = st(self.b[0])[:2]
            lx, ly = self.last[0] if self.last[0] is not None else (0.0, 0.0)
            dx, dy = lx - x, ly - y
            n = math.hypot(dx, dy) or 1.0
            return [dx / n, dy / n]
        return super().script(agent, kind)


GAMES = {"hide": HideWorld, "push": PushWorld, "soccer1": SoccerWorld, "soccer2": Soccer2, "soccer1e": SoccerEasy, "soccer2e": Soccer2Easy, "soccer1k": SoccerKick, "soccer2k": Soccer2Kick, "soccer1n": SoccerNear, "soccer2n": Soccer2Near, "soccer2r": Soccer2NearR, "pitch": PitchWorld, "pitch_easy": PitchEasy}

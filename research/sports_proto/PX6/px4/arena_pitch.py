"""PX4 원형 세계: 투수 대 타자(scratch 세계, 손으로 짠 pymunk 세계. 제품 런타임이 아니다).

출처: scratchpad/sports/eng/proto/arena.py의 PitchWorld·PitchEasy(PX1·PX2 사본과 같은 물리). PX4에서 바꾼 것:
  - 모든 무작위 값을 시드를 받은 numpy Generator에서 뽑는다(원형은 random.Random).
  - 혼자 하는 대신 프리셋 둘(되돌릴 지점 B16): PitchPractice(투구 연습: 타자 없음, 존에 던지기), BatPractice(타격 연습: 정해진 범위에서
    무작위로 날아오는 공 치기).
점수(DE 4.9-(4)): 스트라이크 1.0, 맞으면 0.5 - 0.4 x min(1, 날아간 거리 / 12m)(바닥 0.1), 볼이면 0.1 x (1 - 존까지 가장 가까웠던 거리 / 8m).
타자 점수 = 1 - 투수 점수. 승점: 스트라이크 = 투수 승, 3m 넘게 날아간 안타 = 타자 승, 볼 = 타자 승(ball_loss), 빗맞음 = 무승부.

공통 규격(selfplay.py가 쓰는 것): cls.ROLES, cls.N_IN[role], cls.N_OUT[role], cls.seat_roles(), w.agents [(자리, 역할)],
w.observe(), w.step(actions), w.done, w.result() -> (자리마다 점수, 자리마다 승점, info), w.script(agent, kind).
"""
from __future__ import annotations

import math

import numpy as np
import pymunk
from pymunk._chipmunk_cffi import lib as cp


def st(body: pymunk.Body) -> tuple[float, float, float, float, float, float]:
    raw = body._body
    p = cp.cpBodyGetPosition(raw)
    v = cp.cpBodyGetVelocity(raw)
    return p.x, p.y, cp.cpBodyGetAngle(raw), v.x, v.y, cp.cpBodyGetAngularVelocity(raw)


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clip(x: float, lo: float = -5.0, hi: float = 5.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


def box(body: pymunk.Body, w: float, h: float, mass: float, group: int, friction: float = 0.6) -> pymunk.Poly:
    hw, hh = w / 2, h / 2
    s = pymunk.Poly(body, [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
    s.mass = mass
    s.friction = friction
    s.filter = pymunk.ShapeFilter(group=group)
    return s


class _Base:
    ROLES = ["player"]
    MAX_T = 10.0
    DTW = 1.0 / 60.0

    def __init__(self, seed: int) -> None:
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

    def u(self, lo: float, hi: float) -> float:
        return float(self.rng.uniform(lo, hi))

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
        if kind == "random":  # 10스텝마다 바뀌는 무작위 행동(세계 시드·선수·구간에서 뽑음)
            key = self.steps // 10
            if self._rand_key.get(agent) != key:
                self._rand_key[agent] = key
                r = np.random.default_rng([self._rseed & 0xFFFFFFFF, agent, key])
                self._rand_act[agent] = r.uniform(-1.0, 1.0, n).tolist()
            return self._rand_act[agent]
        raise ValueError(kind)


class PitchWorld(_Base):
    """투수(어깨·팔꿈치 모터 + 놓기)가 앞의 타자(어깨·손목 모터, 배트)에게 던진다. 원형 PitchWorld와 같은 물리."""

    ROLES = ["pitcher", "batter"]
    N_IN = {0: 7, 1: 9}
    N_OUT = {0: 3, 1: 2}
    MAX_T = 6.0
    PLATE_X = 9.7
    ZONE = (0.7, 1.5)
    BX = 10.0
    PASS_X = 11.5
    W1, W2 = 14.0, 18.0
    BAT_LEN = 1.0
    HAS_PITCHER = True
    HAS_BATTER = True

    @classmethod
    def seat_roles(cls) -> list[int]:
        return [0, 1]

    def __init__(self, seed: int, noise: bool = True, ball_w: float = 0.1, hit_floor: float = 0.1, ball_loss: bool = True,
                 bat_len: float | None = None, zone: tuple[float, float] | None = None, **_: object) -> None:
        super().__init__(seed)
        self.ball_w, self.hit_floor, self.ball_loss = ball_w, hit_floor, ball_loss
        if bat_len is not None:
            self.BAT_LEN = float(bat_len)
        if zone is not None:
            self.ZONE = (float(zone[0]), float(zone[1]))
        sp = self.space
        sp.gravity = (0.0, -9.81)
        ground = pymunk.Segment(sp.static_body, (-30.0, 0.0), (16.0, 0.0), 0.05)
        ground.friction = 0.6
        ground.elasticity = 0.4
        ground.collision_type = 3
        sp.add(ground)
        self.agents = self._agents()
        self.noise = self.u(-1.0, 1.0) if noise else 0.0
        self.ball = pymunk.Body()
        bs = pymunk.Circle(self.ball, 0.08)
        bs.mass = 0.25
        bs.elasticity = 0.5
        bs.friction = 0.5
        bs.filter = pymunk.ShapeFilter(group=1)
        bs.collision_type = 1
        self._ball_shape = bs
        if self.HAS_PITCHER:
            self._make_pitcher()
        else:
            self._make_launcher()
        if self.HAS_BATTER:
            self._make_batter()
        self.held = True
        self.hit = False
        self.hit_step = 0
        self.crossed: float | None = None
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

    def _agents(self) -> list[tuple[int, int]]:
        return [(0, 0), (1, 1)]

    def _make_pitcher(self) -> None:
        sp = self.space
        S = (0.0, 1.7)
        a1 = -2.2 + self.u(-0.2, 0.2)
        a2 = a1 + 0.3
        self.p_up = pymunk.Body()
        self.p_up.angle = a1
        self.p_up.position = (S[0] + 0.25 * math.cos(a1), S[1] + 0.25 * math.sin(a1))
        el = (S[0] + 0.5 * math.cos(a1), S[1] + 0.5 * math.sin(a1))
        self.p_fo = pymunk.Body()
        self.p_fo.angle = a2
        self.p_fo.position = (el[0] + 0.25 * math.cos(a2), el[1] + 0.25 * math.sin(a2))
        hand = (el[0] + 0.5 * math.cos(a2), el[1] + 0.5 * math.sin(a2))
        self.ball.position = hand
        sp.add(self.p_up, box(self.p_up, 0.5, 0.08, 3.0, 1), self.p_fo, box(self.p_fo, 0.5, 0.08, 2.0, 1), self.ball, self._ball_shape)
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

    def _make_launcher(self) -> None:  # BatPractice가 덮어씀
        raise NotImplementedError

    def _make_batter(self) -> None:
        sp = self.space
        B = (self.BX, 1.75)
        b1 = -0.4 + self.u(-0.1, 0.1)
        b2 = b1 + 0.3
        self.b_up = pymunk.Body()
        self.b_up.angle = b1
        self.b_up.position = (B[0] + 0.175 * math.cos(b1), B[1] + 0.175 * math.sin(b1))
        wr = (B[0] + 0.35 * math.cos(b1), B[1] + 0.35 * math.sin(b1))
        L = self.BAT_LEN
        self.bat = pymunk.Body()
        self.bat.angle = b2
        self.bat.position = (wr[0] + 0.5 * L * math.cos(b2), wr[1] + 0.5 * L * math.sin(b2))
        bat_s = box(self.bat, L, 0.07, 2.0, 2)
        bat_s.elasticity = 0.6
        bat_s.collision_type = 2
        sp.add(self.b_up, box(self.b_up, 0.35, 0.08, 3.0, 2), self.bat, bat_s)
        k1 = pymunk.PivotJoint(sp.static_body, self.b_up, B, (-0.175, 0.0))
        self.bm1 = pymunk.SimpleMotor(sp.static_body, self.b_up, 0.0)
        self.bm1.max_force = 500.0
        k2 = pymunk.PivotJoint(self.b_up, self.bat, (0.175, 0.0), (-0.5 * L, 0.0))
        kl = pymunk.RotaryLimitJoint(self.b_up, self.bat, -1.2, 1.2)
        self.bm2 = pymunk.SimpleMotor(self.b_up, self.bat, 0.0)
        self.bm2.max_force = 250.0
        for c in (k1, self.bm1, k2, kl, self.bm2):
            c.collide_bodies = False
        sp.add(k1, self.bm1, k2, kl, self.bm2)
        self.B = B

    def _pitcher_obs(self) -> list[float]:
        u, f = st(self.p_up), st(self.p_fo)
        t = self.steps * self.DTW
        return [wrap(u[2]) / math.pi, clip((f[2] - u[2]) / 2.4), clip(u[5] / self.W1), clip((f[5] - u[5]) / self.W2),
                1.0 if self.held else 0.0, clip(t / 2.0), self.noise]

    def _ball_state(self) -> tuple[float, ...]:
        return st(self.ball)

    def _batter_obs(self) -> list[float]:
        a, b, ball = st(self.b_up), st(self.bat), self._ball_state()
        t = self.steps * self.DTW
        return [wrap(a[2]) / math.pi, clip((b[2] - a[2]) / 1.2), clip(a[5] / 14.0),
                clip((ball[0] - self.B[0]) / 10.0), clip((ball[1] - self.B[1]) / 3.0), clip(ball[3] / 20.0),
                clip(ball[4] / 10.0), 1.0 if self.held else 0.0, clip(t / 6.0)]

    def observe(self) -> list[list[float]]:
        return [self._pitcher_obs(), self._batter_obs()]

    def _pitcher_act(self, pa: list[float]) -> None:
        cp.cpSimpleMotorSetRate(self.pm1._constraint, -pa[0] * self.W1)
        cp.cpSimpleMotorSetRate(self.pm2._constraint, -pa[1] * self.W2)
        t = self.steps * self.DTW
        if self.held and ((pa[2] < 0.0 and t >= 0.2) or t >= 2.0):
            self.held = False
            self.grip.max_force = 0.0  # 관절을 빼지 않고 힘만 0으로(공간 구성을 바꾸지 않음)

    def _batter_act(self, ba: list[float]) -> None:
        cp.cpSimpleMotorSetRate(self.bm1._constraint, -ba[0] * 14.0)
        cp.cpSimpleMotorSetRate(self.bm2._constraint, -ba[1] * 12.0)

    def _apply(self, actions: list[list[float]]) -> None:
        pa, ba = actions
        self._pitcher_act(pa)
        self._batter_act(ba)

    def step(self, actions: list[list[float]]) -> None:
        if self.done:
            return
        self._apply(actions)
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
                self.done = True
        if self.steps >= self.max_steps:
            self.done = True

    def outcome(self) -> tuple[float, str, float]:
        """(투수 점수, 결말, 날아간 거리)."""
        strike = (self.crossed is not None and self.ZONE[0] <= self.crossed <= self.ZONE[1] and not self.hit)
        if self.hit:
            lx = self.land_x if self.land_x is not None else cp.cpBodyGetPosition(self.ball._body).x
            dist = max(0.0, self.PLATE_X - lx)
            ps = 0.5 - (0.5 - self.hit_floor) * min(1.0, dist / 12.0)
            return ps, ("hit" if dist > 3.0 else "foul"), dist
        if strike:
            return 1.0, "strike", 0.0
        return self.ball_w * (1.0 - min(1.0, self.closest / 8.0)), "ball", 0.0

    def result(self) -> tuple[list[float], list[float], dict]:
        ps, end, dist = self.outcome()
        if end == "hit":
            pts = [0.0, 1.0]
        elif end == "foul":
            pts = [0.5, 0.5]
        elif end == "strike":
            pts = [1.0, 0.0]
        else:
            pts = [0.0, 1.0] if self.ball_loss else [0.5, 0.5]
        return [ps, 1.0 - ps], pts, {"steps": self.steps, "end": end, "dist": dist, "closest": self.closest}

    def script(self, agent: int, kind: str) -> list[float]:
        if kind == "swing" and self.agents[agent][1] == 1:  # 공이 가까워지면 아래로 휘두르는 타자(원형의 손으로 짠 기준)
            ball = self._ball_state()
            if not self.held and ball[0] > self.PLATE_X - 3.7:
                return [-1.0, -0.3]
            return [0.0, 0.0]
        return super().script(agent, kind)


class PitchEasy(PitchWorld):
    """DE 4.9-(4)의 판(원형 「쉬운 판」): 조종 120Hz, 느린 팔(8·10 rad/s), 타자 7.3m, 존 x = 7.0, y 0.6..1.8m."""

    DTW = 1.0 / 120.0
    PLATE_X = 7.0
    ZONE = (0.6, 1.8)
    BX = 7.3
    PASS_X = 8.8
    W1, W2 = 8.0, 10.0


class PitchPractice(PitchEasy):
    """B16 「투구 연습」: 타자 없이 존에 던지기(Grip만 씀). 혼자: 자리 하나, 역할 0. 점수·결말은 PitchEasy의 투수 것 그대로."""

    ROLES = ["pitcher"]
    N_IN = {0: 7}
    N_OUT = {0: 3}
    HAS_BATTER = False

    @classmethod
    def seat_roles(cls) -> list[int]:
        return [0]

    def _agents(self) -> list[tuple[int, int]]:
        return [(0, 0)]

    def observe(self) -> list[list[float]]:
        return [self._pitcher_obs()]

    def _apply(self, actions: list[list[float]]) -> None:
        self._pitcher_act(actions[0])

    def result(self) -> tuple[list[float], list[float], dict]:
        ps, end, dist = self.outcome()
        return [ps], [1.0 if end == "strike" else 0.0], {"steps": self.steps, "end": end, "dist": dist, "closest": self.closest}


class BatPractice(PitchEasy):
    """B16 「타격 연습」: 정해진 범위에서 무작위로 날아오는 공 치기. 혼자: 자리 하나, 역할 0(타자의 입력 9·출력 2).

    공은 놓는 자리 x = 0.55, y ~ U(1.6, 2.0)에 떠 있다가 t ~ U(0.25, 0.75)초에 날아온다: 가로 속도 vx ~ U(7, 11)m/s, 세로 속도는
    존 평면(x = 7.0)을 지나는 높이 h ~ U(0.7, 1.7)m(존 0.6..1.8 안)가 되게 계산. 점수 = 타자 점수(1 - 투수 점수): 안 맞으면 0,
    맞으면 0.5 + 0.4 x min(1, 날아간 거리 / 12m). 성공(승점 1) = 3m 넘게 날아간 안타.
    """

    ROLES = ["batter"]
    N_IN = {0: 9}
    N_OUT = {0: 2}
    HAS_PITCHER = False
    VX = (7.0, 11.0)
    H = (0.7, 1.7)
    T_LAUNCH = (0.25, 0.75)

    @classmethod
    def seat_roles(cls) -> list[int]:
        return [0]

    def _agents(self) -> list[tuple[int, int]]:
        return [(0, 0)]  # 혼자: 역할 번호 0(입력·출력은 타자의 것)

    def _make_launcher(self) -> None:
        x0, y0 = 0.55, self.u(1.6, 2.0)
        self.t_launch = self.u(*self.T_LAUNCH)
        vx = self.u(*self.VX)
        h = self.u(*self.H)
        t = (self.PLATE_X - x0) / vx
        vy = (h - y0 + 0.5 * 9.81 * t * t) / t
        self.launch = (x0, y0, vx, vy)
        self.ball.position = (x0, y0)
        self.in_space = False

    def _ball_state(self) -> tuple[float, ...]:
        if not self.in_space:
            return (self.launch[0], self.launch[1], 0.0, 0.0, 0.0, 0.0)
        return st(self.ball)

    def observe(self) -> list[list[float]]:
        return [self._batter_obs()]

    def _apply(self, actions: list[list[float]]) -> None:
        if self.held and self.steps * self.DTW >= self.t_launch:
            x0, y0, vx, vy = self.launch
            self.ball.position = (x0, y0)
            self.ball.velocity = (vx, vy)
            self.space.add(self.ball, self._ball_shape)
            self.in_space = True
            self.held = False
        self._batter_act(actions[0])

    def step(self, actions: list[list[float]]) -> None:
        if self.done:
            return
        if self.held:  # 공이 아직 공간에 없음: 타자만 움직임
            self._apply(actions)
            if self.held:
                self._space_step()
                if self.steps >= self.max_steps:
                    self.done = True
                return
            # 이 스텝에 공을 놓았음: 아래 공통 경로(x0 = 놓는 자리)
            x0 = self.launch[0]
            self._space_step()
            self._after(x0)
            return
        self._apply(actions)
        x0 = cp.cpBodyGetPosition(self.ball._body).x
        self._space_step()
        self._after(x0)

    def _after(self, x0: float) -> None:
        p = cp.cpBodyGetPosition(self.ball._body)
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
            self.done = True
        if self.steps >= self.max_steps:
            self.done = True

    def result(self) -> tuple[list[float], list[float], dict]:
        ps, end, dist = self.outcome()
        return [1.0 - ps], [1.0 if end == "hit" else 0.0], {"steps": self.steps, "end": end, "dist": dist, "closest": self.closest,
                                                          "crossed": self.crossed}


GAMES = {"pitch": PitchWorld, "pitch_easy": PitchEasy, "pitch_practice": PitchPractice, "bat_practice": BatPractice}

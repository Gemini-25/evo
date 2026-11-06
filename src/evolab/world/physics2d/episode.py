"""환경 런타임: 시험 한 번(에피소드).

결과는 (환경 버전, 몸 수치, Trial)만의 함수다. 물리 세계는 시험마다 새로 만들고(재설정 재사용은 pymunk
관절의 누적 충격량과 접촉 캐시 때문에 결정성을 깬다), 난수는 시험 시드에서만 뽑는다.

학습 워커, 직접 조종, 재생이 모두 이 클래스를 쓴다.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pymunk
from pymunk.autogeometry import convex_decomposition

from evolab.app import strings as S
from evolab.generators.terrain import generate_heightmap
from evolab.spec import model as m
from evolab.spec import symbols as sy
from evolab.spec.compiled import EnvSpec, finite
from evolab.world.physics2d import geometry as geo

AGENT_GROUP = 1
OBS_CLIP = 5.0
COLLISION_SLOP = 0.01
NUM_LIMIT = 1e15
"""만들기 단계 수식 결과를 이 범위로 자른다(무한대가 물리로 들어가지 않게)."""
FORCE_LIMIT = 1e8
FIELD_LIMIT = 1e4
"""중력, 바람의 크기 한계."""


@dataclass(frozen=True)
class Trial:
    """시험 하나의 조건. 같은 Trial이면 모든 개체가 같은 세계에서 시험받는다."""

    seed: int
    track: int | None = None


@dataclass
class EpisodeResult:
    score: float
    blocks: dict[str, float]
    """점수 블록 id -> 가중치 적용 전 값."""
    weighted: dict[str, float]
    """점수 블록 id -> 가중치 적용 값(수식 점수면 비어 있음)."""
    outcome: str
    """success, failure, neutral"""
    end_event: str
    end_label: str
    time: float
    steps: int
    action_mean_abs: dict[str, float]


@dataclass
class FrameState:
    time: float
    poses: np.ndarray
    """(움직이는 물체 수, 3): x, y, 각도. 순서는 Episode.moving_bodies."""
    outputs: np.ndarray
    """구동기 채널별 실제 출력(연료가 떨어지면 0)."""


@dataclass
class ShapeGeom:
    kind: str
    """poly, circle, segment"""
    points: list[geo.Point]
    """poly: 꼭짓점, circle: [중심], segment: [a, b] (모두 물체 좌표)"""
    radius: float = 0.0


@dataclass
class BodyGeom:
    id: str
    owner: str
    moving: bool
    tags: list[str]
    shapes: list[ShapeGeom]
    pose: tuple[float, float, float]
    """시작 자세(정적 물체는 늘 이 자세)."""


@dataclass
class RegionGeom:
    id: str
    kind: str
    points: list[geo.Point]


@dataclass
class ThrusterGeom:
    channel: str
    body: str
    at: geo.Point
    dir: float
    max_force: float


@dataclass
class WorldGeometry:
    """화면 그리기용. 시험이 시작할 때 한 번 만든다."""

    bounds: tuple[float, float, float, float]
    bodies: list[BodyGeom]
    regions: list[RegionGeom]
    thrusters: list[ThrusterGeom]


class EpisodeBuildError(ValueError):
    """환경을 조립할 수 없다(수식 결과가 잘못된 모양 등). 결정적인 오류다."""


def _clamp(x: float, lo: float, hi: float) -> float:
    """범위로 자른다. NaN이면 lo(min/max 조합은 NaN을 그대로 통과시킨다)."""
    if not x >= lo:
        return lo
    return hi if x > hi else x


# ---------------------------------------------------------------------------
# 만들기 단계 도우미
# ---------------------------------------------------------------------------


def _rand_values(env: m.EnvFile, seed: int) -> dict[str, float]:
    """랜덤 값마다 (시험 시드, id의 CRC) 독립 흐름에서 뽑는다. 랜덤 값을 더하거나 빼도 다른 값은 그대로다."""
    out: dict[str, float] = {}
    for v in env.randomize:
        ss = np.random.SeedSequence(entropy=seed & ((1 << 128) - 1), spawn_key=(zlib.crc32(v.id.encode("utf-8")),))
        rng = np.random.default_rng(ss)
        if v.dist == "uniform":
            x = float(rng.uniform(v.low, v.high)) if v.high != v.low else float(v.low or 0.0)
        elif v.dist == "int":
            x = float(rng.integers(int(v.low or 0), int(v.high or 0), endpoint=True))
        elif v.dist == "normal":
            x = float(rng.normal(v.mean, v.std))
        else:
            vals = v.values or [0.0]
            x = float(vals[int(rng.integers(len(vals)))])
        out[f"r.{v.id}"] = min(max(finite(x), -NUM_LIMIT), NUM_LIMIT)
    return out


def _param_values(env: m.EnvFile, body_params: Sequence[float] | np.ndarray | None) -> dict[str, float]:
    out = {f"p.{p.id}": float(p.value) for p in env.params}
    evolved = sy.evolved_params(env)
    if body_params is not None:
        values = [float(x) for x in np.asarray(body_params, dtype=np.float64).ravel()]
        if len(values) != len(evolved):
            raise EpisodeBuildError(S.RT_BODY_PARAM_COUNT.format(got=len(values), want=len(evolved)))
        for p, x in zip(evolved, values, strict=True):
            lo = p.min if p.min is not None else x
            hi = p.max if p.max is not None else x
            out[f"p.{p.id}"] = min(max(x, lo), hi) if math.isfinite(x) else float(p.value)
    return out


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


@dataclass
class _ContactChannel:
    body: str
    selectors: tuple[str, ...]
    raw: bool = False
    last_true: float = -1e9
    value: bool = False


@dataclass
class _EventState:
    id: str
    hold_steps: int
    consec: int = 0
    occurred: bool = False
    count: int = 0
    last_time: float | None = None


@dataclass
class _Thruster:
    channel_index: int
    body: pymunk.Body
    at: tuple[float, float]
    force: tuple[float, float]
    """최대 출력에서 물체 좌표계 힘."""
    tank: str | None
    fuel_per_s: float
    slot: int = 0
    """힘을 모으는 물체 칸 번호(_force_bodies)."""
    torque: float = 0.0
    """최대 출력에서 무게중심에 대한 토크."""


@dataclass
class _Tank:
    capacity: float
    remaining: float


@dataclass
class _Region:
    points: list[geo.Point]
    center: geo.Point


@dataclass
class _BlockState:
    block: m.Block
    acc: float = 0.0
    count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class Episode:
    """시험 한 번. observe() -> step(행동) 반복, done이 되면 result()."""

    def __init__(self, spec: EnvSpec, body_params: Sequence[float] | np.ndarray | None, trial: Trial) -> None:
        self.spec = spec
        env = spec.file
        self.env = env
        self.trial = trial
        self.dt = float(env.world.dt)
        self.max_steps = max(1, math.ceil(env.world.max_time / self.dt - 1e-9))
        self.steps = 0
        self.time = 0.0
        self._done = False
        self._end: m.End | None = None

        # ---- 만들기 단계 값 ----
        self.randoms = _rand_values(env, trial.seed)
        self.params = _param_values(env, body_params)
        self.gen_values: dict[str, float] = {}
        self._build_env: dict[sy.BuildStage, dict[str, float]] = {"generator": dict(self.randoms)}

        self.space = pymunk.Space()
        self.space.collision_slop = COLLISION_SLOP
        self.bodies: dict[str, pymunk.Body] = {}
        self._shape_info: dict[pymunk.Shape, tuple[str, frozenset[str]]] = {}
        self._geoms: list[BodyGeom] = []
        self._regions: dict[str, _Region] = {}
        self._region_geoms: list[RegionGeom] = []
        self._agent_ids: list[str] = []
        self._drag: list[tuple[str, float]] = []
        self._start_pos: dict[str, geo.Point] = {}

        self._build_generators()
        self._build_env["world"] = {**self.randoms, **self.gen_values}
        self._build_env["agent"] = {**self.randoms, **self.gen_values, **self.params}

        w = env.world
        lim = lambda x: _clamp(x, -FIELD_LIMIT, FIELD_LIMIT)  # noqa: E731
        self.gravity = (lim(self._num(w.gravity[0], "world")), lim(self._num(w.gravity[1], "world")))
        self.wind = (lim(self._num(w.wind[0], "world")), lim(self._num(w.wind[1], "world")))
        b = w.bounds
        self.bounds = (self._num(b[0], "world"), self._num(b[1], "world"), self._num(b[2], "world"), self._num(b[3], "world"))
        self.space.gravity = self.gravity

        self._build_regions()
        self._build_bodies()
        self._build_joints()
        self._build_actuators()
        self._setup_contacts()
        self._values: dict[str, Any] = {}
        self._build_sensors()
        self._build_scoring()

        self.moving_bodies = [g.id for g in self._geoms if g.moving]
        self._moving = [self.bodies[i] for i in self.moving_bodies]
        # hold초 유지: 조건이 처음 참이 된 스텝부터 hold초가 지난 스텝에 일어난다(hold=0이면 바로).
        self._events = [_EventState(e, 1 + max(0, math.ceil(ev.hold / self.dt - 1e-9))) for e, ev in self._event_defs()]
        self._builtin = {"timeout": _EventState("timeout", 1), "out_of_bounds": _EventState("out_of_bounds", 1)}
        states = {**self._builtin, **{st.id: st for st in self._events}}
        self._end_states = [(end, states[end.event]) for end in spec.terminations if end.event in states]
        self._refresh_values()

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self._done

    @property
    def inputs(self) -> list[str]:
        return self.spec.inputs

    @property
    def outputs(self) -> list[str]:
        return self.spec.outputs

    def observe(self) -> np.ndarray:
        """센서 값을 brain.inputs 순서로. 정규화하고 -5..5로 자른 float32."""
        raws = [f() for f in self._sensor_fns]
        out = []
        for si, ci, k in self._obs_index:
            x = raws[si][ci] * k
            if x != x:  # NaN (물리가 터진 경우)은 0으로
                x = 0.0
            out.append(OBS_CLIP if x > OBS_CLIP else (-OBS_CLIP if x < -OBS_CLIP else x))
        return np.array(out, dtype=np.float32)

    def step(self, action: Sequence[float] | np.ndarray) -> None:
        """구동기 출력(brain.outputs 순서, 각 구동기 고유 범위)으로 한 스텝 진행한다."""
        if self._done:
            return
        values = action.tolist() if isinstance(action, np.ndarray) else list(action)
        if len(values) != len(self._out_index):
            raise ValueError(S.RT_ACTION_SIZE.format(got=len(values), want=len(self._out_index)))
        self._u_prev = self._u
        u = [0.0] * len(self._u_names)
        for x, idx in zip(values, self._out_index, strict=True):
            x = float(x)
            if x != x or x in (math.inf, -math.inf):
                x = 0.0
            lo, hi = self._u_range[idx]
            u[idx] = lo if x < lo else (hi if x > hi else x)
        self._u = u

        self._apply_actuators()
        self._apply_forces()
        self.space.step(self.dt)
        self._cache.clear()
        self._pos_cache.clear()
        self.steps += 1
        self.time = self.steps * self.dt

        self._update_contacts()
        self._refresh_values()
        self._update_events()
        self._accumulate_blocks()
        self._check_termination()

    def result(self) -> EpisodeResult:
        blocks = self.block_values()
        weighted: dict[str, float] = {}
        if self.spec.formula is not None:
            values = dict(self._values)
            values.update(blocks)
            outcome = self._end.outcome if self._end is not None else "neutral"
            for f in sy.END_FLAGS:
                values[f] = outcome == f
            score = finite(float(self.spec.formula(values)))
        else:
            for b in self.env.scoring.blocks:
                weighted[b.id] = b.weight * blocks[b.id]
            score = finite(sum(weighted.values()))
        end = self._end
        steps = max(self.steps, 1)
        return EpisodeResult(
            score=score,
            blocks=blocks,
            weighted=weighted,
            outcome=end.outcome if end is not None else "neutral",
            end_event=end.event if end is not None else "",
            end_label=(end.label or end.event) if end is not None else "",
            time=self.time,
            steps=self.steps,
            action_mean_abs={c: self._abs_sum[i] / steps for i, c in enumerate(self._u_names)},
        )

    def snapshot(self) -> FrameState:
        poses = np.empty((len(self._moving), 3), dtype=np.float64)
        for i, b in enumerate(self._moving):
            poses[i, 0], poses[i, 1] = b.position
            poses[i, 2] = b.angle
        return FrameState(self.time, poses, np.array(self._u_applied, dtype=np.float32))

    def geometry(self) -> WorldGeometry:
        thrusters = [
            ThrusterGeom(self._u_names[t.channel_index], self._body_id[t.body], t.at,
                         math.atan2(t.force[1], t.force[0]), math.hypot(*t.force))
            for t in self._thrusters
        ]
        return WorldGeometry(self.bounds, list(self._geoms), list(self._region_geoms), thrusters)

    def block_values(self) -> dict[str, float]:
        """점수 블록별 지금까지의 값(가중치 전). 끝나기 전이면 지금 끝났다고 친 값."""
        return {st.block.id: finite(self._block_value(st)) for st in self._blocks}

    def channel_values(self) -> dict[str, float]:
        """센서 채널의 원래 값(정규화 전). 화면 표시용."""
        return {name: float(g()) for name, g in self._raw_getters.items()}

    def actuator_channels(self) -> list[str]:
        return list(self._u_names)

    # ------------------------------------------------------------------
    # 만들기
    # ------------------------------------------------------------------

    def _num(self, value: float | str, stage: sy.BuildStage) -> float:
        if isinstance(value, (int, float)):
            x = float(value)
        else:
            x = float(self.spec.formulas[(value, stage)](self._build_env[stage]))
        return _clamp(finite(x), -NUM_LIMIT, NUM_LIMIT)

    def _vec(self, v: Sequence[float | str], stage: sy.BuildStage) -> geo.Point:
        return (self._num(v[0], stage), self._num(v[1], stage))

    def _add_body(
        self,
        body_id: str,
        body: pymunk.Body,
        shapes: list[tuple[pymunk.Shape, ShapeGeom]],
        tags: list[str],
        owner: str,
        per_shape_tags: list[list[str]] | None = None,
    ) -> None:
        self.bodies[body_id] = body
        pyshapes = [s for s, _ in shapes]
        for k, s in enumerate(pyshapes):
            t = per_shape_tags[k] if per_shape_tags is not None else tags
            self._shape_info[s] = (body_id, frozenset(t))
        try:
            self.space.add(body, *pyshapes)
        except AssertionError:  # pymunk: 질량이나 관성이 0, 무한대, NaN
            raise EpisodeBuildError(S.RT_BAD_MASS.format(body=body_id)) from None
        if body.body_type == pymunk.Body.DYNAMIC and not (
            math.isfinite(body.mass) and body.mass > 0 and math.isfinite(body.moment) and body.moment > 0
        ):
            raise EpisodeBuildError(S.RT_BAD_MASS.format(body=body_id))
        moving = body.body_type == pymunk.Body.DYNAMIC
        self._geoms.append(
            BodyGeom(body_id, owner, moving, sorted(set(tags)), [g for _, g in shapes],
                     (body.position.x, body.position.y, body.angle))
        )

    def _build_generators(self) -> None:
        for g in self.env.generators:
            if isinstance(g, m.HeightmapGen):
                self._build_heightmap(g)
        self._body_id: dict[pymunk.Body, str] = {}

    def _build_heightmap(self, g: m.HeightmapGen) -> None:
        num = lambda v: self._num(v, "generator")  # noqa: E731
        x_min, x_max = num(g.x_range[0]), num(g.x_range[1])
        pad_x = pad_w = None
        if g.pad is not None:
            pad_x, pad_w = num(g.pad.x), num(g.pad.width)
        try:
            hm = generate_heightmap(
                int(num(g.seed)) & 0x7FFFFFFFFFFFFFFF,
                x_min=x_min,
                x_max=x_max,
                segment=g.segment,
                base_y=num(g.base_y),
                amplitude=max(0.0, num(g.amplitude)),
                roughness=min(1.0, max(0.0, num(g.roughness))),
                pad_x=pad_x,
                pad_width=pad_w,
            )
        except ValueError as exc:
            raise EpisodeBuildError(f"generators.{g.id}: {exc}") from None
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        friction = _clamp(num(g.friction), 0.0, 1e3)
        shapes: list[tuple[pymunk.Shape, ShapeGeom]] = []
        tags: list[list[str]] = []
        xs, ys = hm.xs, hm.ys
        for i in range(len(xs) - 1):
            a = (float(xs[i]), float(ys[i]))
            b = (float(xs[i + 1]), float(ys[i + 1]))
            seg = pymunk.Segment(body, a, b, g.radius)
            seg.friction = friction
            on_pad = hm.pad is not None and hm.pad[0] - 1e-9 <= a[0] and b[0] <= hm.pad[1] + 1e-9
            shapes.append((seg, ShapeGeom("segment", [a, b], g.radius)))
            tags.append(["terrain", "pad"] if on_pad else ["terrain"])
        self._add_body(g.id, body, shapes, ["terrain", "pad"] if hm.pad else ["terrain"], "world", tags)
        self._geoms[-1].tags = ["terrain"]
        self.gen_values[f"gen.{g.id}.x_min"] = float(xs[0])
        self.gen_values[f"gen.{g.id}.x_max"] = float(xs[-1])
        if hm.pad is not None and g.pad is not None:
            left, right, y = hm.pad
            self.gen_values.update({
                f"gen.{g.id}.pad.x": (left + right) / 2.0,
                f"gen.{g.id}.pad.y": y,
                f"gen.{g.id}.pad.w": right - left,
                f"gen.{g.id}.pad.left": left,
                f"gen.{g.id}.pad.right": right,
            })
            h = g.pad.region_height
            pts = [(left, y), (right, y), (right, y + h), (left, y + h)]
            self._regions[f"{g.id}.pad"] = _Region(pts, geo.polygon_center(pts))
            self._region_geoms.append(RegionGeom(f"{g.id}.pad", "pad", pts))

    def _shape_points(self, shape: m.Box | m.Circle | m.Poly, stage: sy.BuildStage) -> list[geo.Point]:
        if isinstance(shape, m.Box):
            return geo.box_points(self._num(shape.w, stage), self._num(shape.h, stage),
                                  self._vec(shape.offset, stage), self._num(shape.angle, stage))
        if isinstance(shape, m.Circle):
            return geo.circle_points(self._num(shape.r, stage), self._vec(shape.offset, stage))
        return [self._vec(p, stage) for p in shape.points]

    def _build_regions(self) -> None:
        for r in self.env.regions:
            pos = self._vec(r.pos, "world")
            pts = [(x + pos[0], y + pos[1]) for x, y in self._shape_points(r.shape, "world")]
            self._regions[r.id] = _Region(pts, geo.polygon_center(pts))
            self._region_geoms.append(RegionGeom(r.id, r.kind, pts))

    def _make_shapes(self, body: pymunk.Body, b: m.Body, stage: sy.BuildStage) -> list[tuple[pymunk.Shape, ShapeGeom]]:
        mat = b.material
        # Chipmunk는 음수 마찰·탄성에서 프로세스를 강제 종료하므로 여기서 범위를 잡는다.
        density = _clamp(self._num(mat.density, stage), 1e-6, 1e9)
        friction = _clamp(self._num(mat.friction, stage), 0.0, 1e3)
        elasticity = _clamp(self._num(mat.elasticity, stage), 0.0, 1.0)
        out: list[tuple[pymunk.Shape, ShapeGeom]] = []

        def poly(points: list[geo.Point]) -> None:
            if abs(geo.signed_area(points)) < 1e-9 or not geo.is_simple(points):
                raise EpisodeBuildError(S.RT_DEGENERATE_SHAPE.format(body=b.id))
            if geo.signed_area(points) < 0:  # pymunk의 볼록 분해는 반시계 방향만 받는다
                points = points[::-1]
            if geo.is_convex(points):
                pieces = [points]
            else:
                try:
                    pieces = [[(p.x, p.y) for p in piece] for piece in convex_decomposition(points + [points[0]], 0.01)]
                except Exception:  # pymunk가 자기 교차 다각형에서 실패할 수 있다
                    raise EpisodeBuildError(S.RT_DEGENERATE_SHAPE.format(body=b.id)) from None
            if len(pieces) > 64:
                raise EpisodeBuildError(S.RT_DEGENERATE_SHAPE.format(body=b.id))
            for piece in pieces:
                out.append((pymunk.Poly(body, piece), ShapeGeom("poly", piece)))

        for s in b.shapes:
            if isinstance(s, m.Box):
                w, h = self._num(s.w, stage), self._num(s.h, stage)
                if not (0 < w <= 1e5 and 0 < h <= 1e5):
                    raise EpisodeBuildError(S.RT_DEGENERATE_SHAPE.format(body=b.id))
                poly(geo.box_points(w, h, self._vec(s.offset, stage), self._num(s.angle, stage)))
            elif isinstance(s, m.Circle):
                r = self._num(s.r, stage)
                if not 0 < r <= 1e5:
                    raise EpisodeBuildError(S.RT_DEGENERATE_SHAPE.format(body=b.id))
                off = self._vec(s.offset, stage)
                out.append((pymunk.Circle(body, r, off), ShapeGeom("circle", [off], r)))
            elif isinstance(s, m.Poly):
                poly([self._vec(p, stage) for p in s.points])
            elif isinstance(s, m.Radial):
                radii = [_clamp(self._num(r, stage), 0.01, 1e5) for r in s.radii]
                pts = geo.radial_points(radii, self._num(s.angle, stage))
                for i in range(len(pts)):
                    tri = [(0.0, 0.0), pts[i], pts[(i + 1) % len(pts)]]
                    out.append((pymunk.Poly(body, tri), ShapeGeom("poly", tri)))
            elif isinstance(s, m.Chain):
                pts = [self._vec(p, stage) for p in s.points]
                pairs = list(zip(pts, pts[1:], strict=False))
                if s.closed and len(pts) > 2:
                    pairs.append((pts[-1], pts[0]))
                for a, c in pairs:
                    out.append((pymunk.Segment(body, a, c, s.radius), ShapeGeom("segment", [a, c], s.radius)))
        for shape, _ in out:
            shape.friction = friction
            shape.elasticity = elasticity
            if body.body_type == pymunk.Body.DYNAMIC:
                shape.density = max(density, 1e-6)
        return out

    def _agent_poses(self) -> dict[str, tuple[geo.Point, float]]:
        """에이전트 좌표계의 자세. place_by가 있으면 부모 관절 앵커와 rest_angle에서 계산한다."""
        joints = {j.id: j for j in self.env.joints}
        bodies = {b.id: b for b in self.env.bodies}
        poses: dict[str, tuple[geo.Point, float]] = {}

        def pose(bid: str) -> tuple[geo.Point, float]:
            if bid in poses:
                return poses[bid]
            b = bodies[bid]
            stage: sy.BuildStage = "agent" if b.owner == "agent" else "world"
            if b.place_by is not None:
                j = joints[b.place_by]
                if j.a in bodies:
                    (pp, pa) = pose(j.a)
                else:
                    pp, pa = (0.0, 0.0), 0.0
                js = sy.joint_stage(self.env, j)
                angle = pa + self._num(j.rest_angle, js)
                anchor_world = geo.transform(self._vec(j.anchor_a, js), pp, pa)
                ab = geo.rotate(self._vec(j.anchor_b, js), angle)
                result = ((anchor_world[0] - ab[0], anchor_world[1] - ab[1]), angle)
            else:
                result = (self._vec(b.pos, stage), self._num(b.angle, stage))
            poses[bid] = result
            return result

        for b in self.env.bodies:
            pose(b.id)
        return poses

    def _build_bodies(self) -> None:
        env = self.env
        sp = env.agent.spawn
        spawn = self._vec((sp.x, sp.y), "agent")
        spawn_angle = self._num(sp.angle, "agent")
        spawn_v = self._vec((sp.vx, sp.vy), "agent")
        spawn_w = self._num(sp.omega, "agent")
        poses = self._agent_poses()
        types = {"dynamic": pymunk.Body.DYNAMIC, "static": pymunk.Body.STATIC, "kinematic": pymunk.Body.KINEMATIC}
        for b in env.bodies:
            body = pymunk.Body(body_type=types[b.type])
            (lp, la) = poses[b.id]
            stage: sy.BuildStage = "agent" if b.owner == "agent" else "world"
            if b.owner == "agent":
                wp = geo.transform(lp, spawn, spawn_angle)
                body.position = wp
                body.angle = la + spawn_angle
                self._agent_ids.append(b.id)
            else:
                body.position = lp
                body.angle = la
            shapes = self._make_shapes(body, b, stage)
            if b.owner == "agent":
                for s, _ in shapes:
                    s.filter = pymunk.ShapeFilter(group=AGENT_GROUP)
            self._add_body(b.id, body, shapes, list(b.tags), b.owner)
            if b.owner == "agent" and b.type == "dynamic":
                # 강체 전체가 함께 도는 시작: 속도는 무게중심 속도이므로 무게중심 위치로 계산한다.
                c = body.local_to_world(body.center_of_gravity)
                rx, ry = c.x - spawn[0], c.y - spawn[1]
                body.velocity = (spawn_v[0] - spawn_w * ry, spawn_v[1] + spawn_w * rx)
                body.angular_velocity = spawn_w
            drag = _clamp(self._num(b.drag, stage), 0.0, 1e6)
            if drag != 0.0 and b.type == "dynamic":
                # 선형 항력을 한 스텝 동안 정확히 적분한 계수: v가 바람 속도를 넘어서 진동·발산하지 않는다.
                k = body.mass * (1.0 - math.exp(-drag * self.dt / body.mass)) / self.dt
                self._drag.append((b.id, k))
            self._start_pos[b.id] = (body.position.x, body.position.y)
        self._body_id = {v: k for k, v in self.bodies.items()}

    def _build_joints(self) -> None:
        for j in self.env.joints:
            try:
                self._build_joint(j)
            except AssertionError:  # pymunk: 양쪽 모두 정적 물체 등
                raise EpisodeBuildError(S.SPEC_JOINT_NO_DYNAMIC.format(id=j.id)) from None
        self._joints = {j.id: j for j in self.env.joints}

    def _build_joint(self, j: m.Joint) -> None:
        st = sy.joint_stage(self.env, j)  # 세계 물체끼리의 관절은 세계 단계(p.* 없음)
        num = lambda v: self._num(v, st)  # noqa: E731
        a, b = self.bodies[j.a], self.bodies[j.b]
        aa, ab = self._vec(j.anchor_a, st), self._vec(j.anchor_b, st)
        cons: list[pymunk.Constraint] = []
        if j.kind in ("pivot", "rot_spring", "weld"):
            cons.append(pymunk.PivotJoint(a, b, aa, ab))
        if j.kind == "weld":
            cons.append(pymunk.GearJoint(a, b, b.angle - a.angle, 1.0))
        stiffness = _clamp(num(j.stiffness), 0.0, 1e9)
        damping = _clamp(num(j.damping), 0.0, 1e9)
        if j.kind == "rot_spring":
            # Chipmunk의 회전 스프링은 쉬는 각도를 (a 각도 - b 각도)로 잰다. 각도 제한(RotaryLimitJoint)은
            # (b - a)로 잰다. 환경 파일은 둘 다 (b - a) 기준이므로 스프링에는 부호를 바꿔 넘긴다.
            cons.append(pymunk.DampedRotarySpring(a, b, -num(j.rest_angle), stiffness, damping))
        if j.kind == "spring":
            wa, wb = a.local_to_world(aa), b.local_to_world(ab)
            rest = num(j.rest_length) if j.rest_length is not None else (wa - wb).length
            cons.append(pymunk.DampedSpring(a, b, aa, ab, _clamp(rest, 0.0, 1e5), stiffness, damping))
        if j.limit is not None and j.kind in ("pivot", "rot_spring"):
            lo, hi = num(j.limit[0]), num(j.limit[1])
            cons.append(pymunk.RotaryLimitJoint(a, b, min(lo, hi), max(lo, hi)))
        for c in cons:
            c.collide_bodies = j.collide
        self.space.add(*cons)

    def _build_actuators(self) -> None:
        channels = list(self.spec.actuator_channels.values())
        self._u_names = [c.name for c in channels]
        self._u_range = [(c.low, c.high) for c in channels]
        n = len(channels)
        self._u = [0.0] * n
        self._u_prev = [0.0] * n
        self._u_applied = [0.0] * n
        self._abs_sum = [0.0] * n
        index = {c.name: i for i, c in enumerate(channels)}
        self._out_index = [index[name] for name in self.spec.outputs]
        self._tanks: dict[str, _Tank] = {}
        self._thrusters: list[_Thruster] = []
        for a in self.env.actuators:
            if isinstance(a, m.FuelTank):
                cap = max(self._num(a.capacity, "agent"), 1e-9)
                self._tanks[a.id] = _Tank(cap, cap)
        for a in self.env.actuators:
            if isinstance(a, m.Thruster):
                d = self._num(a.dir, "agent")
                f = min(max(self._num(a.max_force, "agent"), 0.0), FORCE_LIMIT)
                self._thrusters.append(_Thruster(index[a.id], self.bodies[a.body], self._vec(a.at, "agent"),
                                                 (f * math.cos(d), f * math.sin(d)), a.tank,
                                                 max(0.0, self._num(a.fuel_per_s, "agent"))))
        self._thruster_channels = {t.channel_index for t in self._thrusters}
        # 물체마다 힘·토크를 모아 스텝마다 한 번에 넣는다(pymunk 호출을 줄인다).
        slot_of: dict[str, int] = {}
        self._force_bodies: list[tuple[str, pymunk.Body]] = []
        for t in self._thrusters:
            bid = self._body_id[t.body]
            if bid not in slot_of:
                slot_of[bid] = len(self._force_bodies)
                self._force_bodies.append((bid, t.body))
            t.slot = slot_of[bid]
            cog = t.body.center_of_gravity
            rx, ry = t.at[0] - cog.x, t.at[1] - cog.y
            t.torque = rx * t.force[1] - ry * t.force[0]
        self._drag_slots: list[tuple[int, float]] = []
        for bid, k in self._drag:
            if bid not in slot_of:
                slot_of[bid] = len(self._force_bodies)
                self._force_bodies.append((bid, self.bodies[bid]))
            self._drag_slots.append((slot_of[bid], k))
        self._acc = [[0.0, 0.0, 0.0] for _ in self._force_bodies]

    def _apply_forces(self) -> None:
        """모은 추진기 힘(물체 좌표)과 바람 항력(세계 좌표, 무게중심)을 물체마다 한 번에 넣는다."""
        acc = self._acc
        wx, wy = self.wind
        drag = [(0.0, 0.0)] * len(acc)
        for slot, k in self._drag_slots:
            st = self._bstate(self._force_bodies[slot][0])
            drag[slot] = (k * (wx - st[3]), k * (wy - st[4]))
        for slot, (bid, body) in enumerate(self._force_bodies):
            a = acc[slot]
            fx, fy, tq = a
            dx, dy = drag[slot]
            if fx == 0.0 and fy == 0.0 and tq == 0.0 and dx == 0.0 and dy == 0.0:
                continue
            ang = self._bstate(bid)[2]
            c, s_ = math.cos(ang), math.sin(ang)
            body.force = (fx * c - fy * s_ + dx, fx * s_ + fy * c + dy)
            body.torque = tq
            a[0] = a[1] = a[2] = 0.0

    def _apply_actuators(self) -> None:
        u = self._u
        applied = list(u)
        for i in self._thruster_channels:
            applied[i] = 0.0
        dt = self.dt
        for t in self._thrusters:
            x = u[t.channel_index]
            if x <= 0.0:
                continue
            if t.tank is not None and t.fuel_per_s > 0.0:
                tank = self._tanks[t.tank]
                need = x * t.fuel_per_s * dt
                if tank.remaining <= 0.0:
                    continue
                if need > tank.remaining:
                    x *= tank.remaining / need
                    need = tank.remaining
                tank.remaining = max(0.0, tank.remaining - need)
            applied[t.channel_index] = x
            acc = self._acc[t.slot]
            acc[0] += t.force[0] * x
            acc[1] += t.force[1] * x
            acc[2] += t.torque * x
        self._u_applied = applied
        sums = self._abs_sum
        for i, x in enumerate(applied):
            sums[i] += x if x >= 0.0 else -x

    # ------------------------------------------------------------------
    # 접촉
    # ------------------------------------------------------------------

    def _setup_contacts(self) -> None:
        self._touching: dict[str, dict[pymunk.Shape, int]] = {b: {} for b in self.bodies}
        self.space.on_collision(None, None, begin=self._on_begin, separate=self._on_separate)
        self._contacts: dict[tuple[str, tuple[str, ...]], _ContactChannel] = {}
        self._debounce = float(self.env.world.contact_debounce)
        self._contacts_dirty = True
        self._cache: dict[str, tuple[float, float, float, float, float, float]] = {}
        self._pos_cache: dict[str, tuple[float, float]] = {}

    def _contact_channel(self, body: str, selectors: Sequence[str]) -> _ContactChannel:
        key = (body, tuple(selectors))
        ch = self._contacts.get(key)
        if ch is None:
            ch = _ContactChannel(body, tuple(selectors))
            self._contacts[key] = ch
        return ch

    def _on_begin(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: Any) -> None:
        s1, s2 = arbiter.shapes
        for x, y in ((s1, s2), (s2, s1)):
            info = self._shape_info.get(x)
            if info is not None:
                t = self._touching[info[0]]
                t[y] = t.get(y, 0) + 1
        self._contacts_dirty = True

    def _on_separate(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: Any) -> None:
        s1, s2 = arbiter.shapes
        for x, y in ((s1, s2), (s2, s1)):
            info = self._shape_info.get(x)
            if info is not None:
                t = self._touching[info[0]]
                n = t.get(y, 0) - 1
                if n > 0:
                    t[y] = n
                else:
                    t.pop(y, None)
        self._contacts_dirty = True

    def _matches(self, shape: pymunk.Shape, selectors: tuple[str, ...]) -> bool:
        info = self._shape_info.get(shape)
        if info is None:
            return False
        body_id, tags = info
        for sel in selectors:
            if sel.startswith("tag."):
                if sel[4:] in tags:
                    return True
            elif sel == body_id:
                return True
        return False

    def _update_contacts(self) -> None:
        t = self.time
        dirty = self._contacts_dirty
        self._contacts_dirty = False
        limit = self._debounce + 1e-9
        for ch in self._contacts.values():
            if dirty:
                touching = self._touching[ch.body]
                ch.raw = any(self._matches(s, ch.selectors) for s in touching) if touching else False
            if ch.raw:
                ch.last_true = t
                ch.value = True
            else:
                ch.value = (t - ch.last_true) <= limit
        v = self._values
        for ch, name in self._contact_values:
            v[name] = ch.value

    # ------------------------------------------------------------------
    # 센서와 값
    # ------------------------------------------------------------------

    def _bpos(self, body_id: str) -> tuple[float, float]:
        """위치만. 경계 검사·거리·영역 판정은 속도가 필요 없으므로 pymunk 속성을 하나만 읽는다."""
        c = self._cache.get(body_id)
        if c is not None:
            return (c[0], c[1])
        p = self._pos_cache.get(body_id)
        if p is None:
            v = self.bodies[body_id].position
            p = (v.x, v.y)
            self._pos_cache[body_id] = p
        return p

    def _bstate(self, body_id: str) -> tuple[float, float, float, float, float, float]:
        """(x, y, 각도, vx, vy, 각속도). 스텝마다 물체당 한 번만 pymunk에서 읽는다."""
        c = self._cache.get(body_id)
        if c is None:
            b = self.bodies[body_id]
            p = b.position
            v = b.velocity
            c = (p.x, p.y, b.angle, v.x, v.y, b.angular_velocity)
            self._cache[body_id] = c
        return c

    def _state(self, body_id: str, f: str, frame: str = "world") -> float:
        x, y, a, vx, vy, w = self._bstate(body_id)
        if f == "x":
            return x
        if f == "y":
            return y
        if f == "angle":
            return geo.wrap_angle(a)
        if f == "omega":
            return w
        if f == "speed":
            return math.hypot(vx, vy)
        if f in ("v_fwd", "v_lat") or frame == "body":
            c, s = math.cos(a), math.sin(a)
            fwd, lat = vx * c + vy * s, -vx * s + vy * c
            if f in ("v_fwd", "vx"):
                return fwd
            return lat
        return vx if f == "vx" else vy

    def _target_point(self, ref: str) -> geo.Point:
        if ref in self._regions:
            return self._regions[ref].center
        return self._bpos(ref)

    def _distance(self, body_id: str, ref: str) -> float:
        c = self._bpos(body_id)
        if ref in self._regions:
            return geo.dist_point_polygon(c, self._regions[ref].points)
        q = self._bpos(ref)
        return math.hypot(c[0] - q[0], c[1] - q[1])

    def _inside(self, body_id: str, ref: str) -> bool:
        return geo.point_in_polygon(self._bpos(body_id), self._regions[ref].points)

    def _sensor_fn(self, s: m.Sensor) -> tuple[list[str], list[float], Callable[[], tuple[float, ...]]]:
        """센서 하나: (채널 이름들, scale들, 원래 값을 한 번에 돌려주는 함수)."""
        if isinstance(s, m.BodyState):
            names = [f"{s.id}.{f}" for f in s.fields]
            scales = [s.scale.get(f, sy.DEFAULT_STATE_SCALE[f]) for f in s.fields]
            fields = tuple(s.fields)
            body, frame = s.body, s.frame

            def body_state() -> tuple[float, ...]:
                x, y, a, vx, vy, w = self._bstate(body)
                need_body = frame == "body" or "v_fwd" in fields or "v_lat" in fields
                if need_body:
                    c, si = math.cos(a), math.sin(a)
                    fwd, lat = vx * c + vy * si, -vx * si + vy * c
                out = []
                for f in fields:
                    if f == "x":
                        out.append(x)
                    elif f == "y":
                        out.append(y)
                    elif f == "angle":
                        out.append((a + math.pi) % geo.TAU - math.pi)
                    elif f == "omega":
                        out.append(w)
                    elif f == "speed":
                        out.append(math.hypot(vx, vy))
                    elif f == "v_fwd" or (f == "vx" and frame == "body"):
                        out.append(fwd)
                    elif f == "v_lat" or (f == "vy" and frame == "body"):
                        out.append(lat)
                    else:
                        out.append(vx if f == "vx" else vy)
                return tuple(out)

            return names, scales, body_state
        if isinstance(s, m.Target):
            tgt, body, frame = s.target, s.body, s.frame

            def target() -> tuple[float, ...]:
                x, y, a, *_ = self._bstate(body)
                tx, ty = self._target_point(tgt)
                dx, dy = tx - x, ty - y
                if frame == "body":
                    dx, dy = geo.rotate((dx, dy), -a)
                return (dx, dy)

            return [f"{s.id}.x", f"{s.id}.y"], [s.scale, s.scale], target
        if isinstance(s, m.ContactSensor):
            ch = self._contact_channel(s.body, s.with_)
            return [s.id], [1.0], (lambda ch=ch: (1.0 if ch.value else 0.0,))
        if isinstance(s, m.JointAngle):
            j = self._joints[s.joint]
            return [s.id], [s.scale], (
                lambda a=j.a, b=j.b: (geo.wrap_angle(self._bstate(b)[2] - self._bstate(a)[2]),)
            )
        if isinstance(s, m.FuelLeft):
            tank = self._tanks[s.tank]
            return [s.id], [1.0], (lambda t=tank: (t.remaining / t.capacity,))
        if isinstance(s, m.TimeLeft):
            max_time = self.env.world.max_time
            return [s.id], [1.0], (lambda: (max(0.0, 1.0 - self.time / max_time),))
        raise EpisodeBuildError(S.SPEC_NOT_YET.format(milestone="M4"))

    def _build_sensors(self) -> None:
        raw: dict[str, Callable[[], float]] = {}
        where: dict[str, tuple[int, int, float]] = {}
        self._sensor_fns: list[Callable[[], tuple[float, ...]]] = []
        for s in self.env.sensors:
            names, scales, fn = self._sensor_fn(s)
            si = len(self._sensor_fns)
            self._sensor_fns.append(fn)
            for ci, (name, sc) in enumerate(zip(names, scales, strict=True)):
                raw[name] = (lambda fn=fn, ci=ci: fn()[ci])
                where[name] = (si, ci, 1.0 / sc)
        self._raw_getters = raw
        self._obs_index = [where[name] for name in self.spec.inputs]

        # 수식이 읽는 값
        getters: dict[str, Callable[[], Any]] = {}
        needed: set[str] = set()
        for c in self.spec.events.values():
            needed |= c.names
        for c in self.spec.blocks.values():
            needed |= c.names
        if self.spec.formula is not None:
            needed |= self.spec.formula.names
        for q in self.spec.queries.values():
            if q.func == "contact":
                ch = self._contact_channel(q.a, (q.b,))
                getters[q.channel] = (lambda ch=ch: ch.value)
            elif q.func == "inside":
                getters[q.channel] = (lambda q=q: self._inside(q.a, q.b))
            else:
                getters[q.channel] = (lambda q=q: self._distance(q.a, q.b))
        for bid in self.bodies:
            for f in sy.BODY_STATE:
                getters[f"{bid}.{f}"] = (lambda b=bid, f=f: self._state(b, f))
        for name, g in raw.items():
            ch = self.spec.sensor_channels[name]
            getters[name] = (lambda g=g: g() > 0.5) if ch.type == "bool" else g
        for i, name in enumerate(self._u_names):
            getters[name] = (lambda i=i: self._u_applied[i])
        for tid in self._tanks:
            getters[tid] = (lambda t=tid: self._tanks[t].remaining / self._tanks[t].capacity)
        getters["time"] = lambda: self.time
        # 사건 값은 _update_events가 채운다.
        event_names = {e for e in [*sy.BUILTIN_EVENTS, *(ev.id for ev in self.env.events)]}
        event_keys = {f"{e}{suffix}" for e in event_names for suffix in ("", ".count", ".since")}
        # 접촉 질의 값은 _update_contacts가 바로 써 넣는다.
        self._contact_values: list[tuple[_ContactChannel, str]] = []
        for q in self.spec.queries.values():
            if q.func == "contact" and q.channel in needed:
                self._contact_values.append((self._contact_channel(q.a, (q.b,)), q.channel))
        direct = {name for _, name in self._contact_values}
        self._value_getters = [
            (n, getters[n]) for n in sorted(needed) if n in getters and n not in event_keys and n not in direct
        ]
        block_events = [b.event for b in self.env.scoring.blocks if isinstance(b, m.EventBlock)]
        self._count_needed = {e for e in event_names if f"{e}.count" in needed or e in block_events}
        self._since_needed = {e for e in event_names if f"{e}.since" in needed}
        self._values_static = {**self._build_env["agent"]}

    def _refresh_values(self) -> None:
        v = self._values
        if not v:
            v.update(self._values_static)
            for e in [*sy.BUILTIN_EVENTS, *(ev.id for ev in self.env.events)]:
                v[e] = False
                v[f"{e}.count"] = 0.0
                v[f"{e}.since"] = 0.0
            for ch, name in self._contact_values:
                v[name] = ch.value
        for name, g in self._value_getters:
            v[name] = g()

    # ------------------------------------------------------------------
    # 사건, 종료
    # ------------------------------------------------------------------

    def _event_defs(self) -> list[tuple[str, m.Event]]:
        defs = {e.id: e for e in self.env.events}
        return [(e, defs[e]) for e in self.spec.event_order]

    def _fire(self, st: _EventState, cond: bool) -> None:
        if cond:
            st.consec += 1
            if st.consec == st.hold_steps:
                st.occurred = True
                st.count += 1
                st.last_time = self.time
        else:
            st.consec = 0
        v = self._values
        v[st.id] = st.occurred
        if st.id in self._count_needed:
            v[f"{st.id}.count"] = float(st.count)
        if st.id in self._since_needed:
            v[f"{st.id}.since"] = self.time - st.last_time if st.last_time is not None else self.time

    def _update_events(self) -> None:
        self._fire(self._builtin["timeout"], self.steps >= self.max_steps)
        x0, y0, x1, y1 = self.bounds
        oob = False
        for bid in self._agent_ids:
            px, py = self._bpos(bid)
            if not (x0 <= px <= x1 and y0 <= py <= y1):  # NaN도 여기서 걸린다
                oob = True
                break
        self._fire(self._builtin["out_of_bounds"], oob)
        for st in self._events:
            self._fire(st, bool(self.spec.events[st.id](self._values)))

    def _check_termination(self) -> None:
        for end, st in self._end_states:
            if st.occurred:
                self._done = True
                self._end = end
                return
        if self.steps >= self.max_steps:  # 안전장치(timeout이 늘 목록에 있으므로 보통 오지 않는다)
            self._done = True

    # ------------------------------------------------------------------
    # 점수
    # ------------------------------------------------------------------

    def _build_scoring(self) -> None:
        """블록마다 매 스텝 할 일을 만들 때 정해 둔다(스텝마다 종류를 가리지 않게)."""
        self._blocks = [_BlockState(b) for b in self.env.scoring.blocks]
        index = {name: i for i, name in enumerate(self._u_names)}
        actuator_of = {c.name: c.owner for c in self.spec.actuator_channels.values()}
        dt = self.dt
        steps: list[Callable[[], None]] = []
        for st in self._blocks:
            b = st.block
            if isinstance(b, (m.EnergyBlock, m.SmoothnessBlock)):
                chosen = b.actuators
                idx = [i for name, i in index.items() if chosen == "all" or actuator_of[name] in chosen]
                if isinstance(b, m.EnergyBlock):
                    def energy(st: _BlockState = st, idx: list[int] = idx) -> None:
                        u = self._u_applied
                        st.acc += sum(abs(u[i]) for i in idx) * dt
                    steps.append(energy)
                else:
                    def smooth(st: _BlockState = st, idx: list[int] = idx) -> None:
                        if self.steps > 1:
                            u, p = self._u, self._u_prev
                            st.acc += sum(abs(u[i] - p[i]) for i in idx)
                    steps.append(smooth)
            elif isinstance(b, m.ContactTimeBlock):
                ch = self._contact_channel(b.body, b.with_)
                def contact_time(st: _BlockState = st, ch: _ContactChannel = ch) -> None:
                    if ch.value:
                        st.acc += dt
                steps.append(contact_time)
            elif isinstance(b, m.TiltBlock) and b.agg == "mean":
                def tilt(st: _BlockState = st, body: str = b.body) -> None:
                    st.acc += abs(geo.wrap_angle(self._bstate(body)[2]))
                    st.count += 1
                steps.append(tilt)
            elif isinstance(b, m.DistanceBlock) and b.agg == "min":
                st.acc = math.inf
                def dmin(st: _BlockState = st, body: str = b.body, target: str = b.target) -> None:
                    d = self._distance(body, target)
                    if d < st.acc:
                        st.acc = d
                steps.append(dmin)
            elif isinstance(b, m.ExprBlock):
                fn = self.spec.blocks[b.id]
                values = self._values
                if b.agg == "integral":
                    def integ(st: _BlockState = st, fn: Any = fn) -> None:
                        st.acc += float(fn(values)) * dt
                    steps.append(integ)
                elif b.agg == "mean":
                    def mean(st: _BlockState = st, fn: Any = fn) -> None:
                        st.acc += float(fn(values))
                        st.count += 1
                    steps.append(mean)
                elif b.agg in ("min", "max"):
                    st.acc = math.inf if b.agg == "min" else -math.inf
                    pick = min if b.agg == "min" else max
                    def extreme(st: _BlockState = st, fn: Any = fn, pick: Any = pick) -> None:
                        st.acc = pick(st.acc, float(fn(values)))
                    steps.append(extreme)
                else:
                    def final(st: _BlockState = st, fn: Any = fn) -> None:
                        st.acc = float(fn(values))
                    steps.append(final)
        self._block_steps = steps

    def _accumulate_blocks(self) -> None:
        for f in self._block_steps:
            f()

    def _block_value(self, st: _BlockState) -> float:
        b = st.block
        if isinstance(b, m.EventBlock):
            ev = self._values.get(b.event if b.value == "occurred" else f"{b.event}.count", 0.0)
            return float(ev)
        if isinstance(b, m.DistanceBlock):
            return self._distance(b.body, b.target) if b.agg == "final" or st.acc == math.inf else st.acc
        if isinstance(b, m.ProgressBlock):
            d = self._num(b.direction, "agent")
            p = self.bodies[b.body].position
            s = self._start_pos[b.body]
            return (p.x - s[0]) * math.cos(d) + (p.y - s[1]) * math.sin(d)
        if isinstance(b, m.SurvivalBlock):
            return self.time
        if isinstance(b, m.SpeedBlock):
            v = self.bodies[b.body].velocity
            return math.hypot(v.x, v.y)
        if isinstance(b, m.TiltBlock):
            if b.agg == "mean":
                return st.acc / st.count if st.count else abs(geo.wrap_angle(self.bodies[b.body].angle))
            return abs(geo.wrap_angle(self.bodies[b.body].angle))
        if isinstance(b, m.FuelBlock):
            t = self._tanks[b.tank]
            return (t.capacity - t.remaining) / t.capacity
        if isinstance(b, m.ExprBlock):
            if b.agg == "mean":
                return st.acc / st.count if st.count else 0.0
            if b.agg in ("min", "max") and not math.isfinite(st.acc):
                return 0.0
            return st.acc
        return st.acc

"""환경 파일 검사와 컴파일.

EnvSpec은 검사를 통과한 환경 파일 하나와, 거기서 미리 만들어 둔 것들(수식, 사건 순서, 채널 목록)이다.
워커는 환경 버전마다 한 번 만들고 시험마다 재사용한다.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from pydantic import BaseModel

from evolab.app import strings as S
from evolab.scoring.expr import Compiled, ExprError, Sym, compile_expr, formula_body, is_formula
from evolab.spec import model as m
from evolab.spec import structure as st
from evolab.spec import symbols as sy

Severity = Literal["error", "warning", "info"]
"""error: 쓸 수 없음. warning: 쓸 수 있지만 아마 실수. info: 일부러 만든 환경을 알려 주는 안내(v2: 뇌 없는 환경)."""
RAND_LIMIT = 1e15
OBSTACLES_MIN_LEAD = 2.0
"""장애물 줄의 첫 장애물이 에이전트 시작 위치에서 이만큼(m) 안쪽에서 출발하면 경고한다(v2 설계 R11)."""
AERO_STALL_MAX = 1.5
"""공기역학 실속 각도(rad)의 상한. 범위는 (0, 1.5]."""


@dataclass(frozen=True)
class Issue:
    severity: Severity
    where: str
    """문제가 있는 칸의 경로. 예: bodies[1].shapes[0].h"""
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message}" if self.where else self.message


class EnvFileError(ValueError):
    """환경 파일을 쓸 수 없다. issues에 한국어 설명이 있다."""

    def __init__(self, issues: list[Issue]) -> None:
        self.issues = issues
        lines = [str(i) for i in issues if i.severity == "error"]
        super().__init__(S.SPEC_INVALID_FILE.format(count=len(lines)) + "\n" + "\n".join(lines))


# 아직 런타임이 없는 기능과 그 마일스톤.
NOT_YET: dict[type[BaseModel], str] = {}


@dataclass
class EnvSpec:
    """검사하고 컴파일한 환경."""

    file: m.EnvFile
    issues: list[Issue]
    formulas: dict[tuple[str, sy.BuildStage], Compiled]
    """(수식 원문, 단계) -> 컴파일 결과. 만들기 단계 숫자 칸용."""
    events: dict[str, Compiled]
    event_order: list[str]
    blocks: dict[str, Compiled]
    """expr 점수 블록."""
    formula: Compiled | None
    queries: dict[str, sy.Query]
    inputs: list[str]
    outputs: list[str]
    sensor_channels: dict[str, sy.Channel]
    actuator_channels: dict[str, sy.Channel]
    terminations: list[m.End] = field(default_factory=list)

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def infos(self) -> list[Issue]:
        """안내(v2 심각도 info). 오류·경고 개수에 넣지 않는다."""
        return [i for i in self.issues if i.severity == "info"]

    @property
    def track_set_size(self) -> int | None:
        """학습 트랙 세트가 있으면 그 크기(시험 = 트랙마다 한 번, 시험 번호 = 트랙 번호), 없으면 None."""
        for g in self.file.generators:
            if isinstance(g, m.TrackGen) and g.source == "generated":
                return int(g.set_size)
        return None

    @property
    def body_param_count(self) -> int:
        return len(sy.evolved_params(self.file))

    @classmethod
    def build(cls, env: m.EnvFile) -> EnvSpec:
        """검사하고 컴파일한다. 오류가 하나라도 있으면 EnvFileError."""
        spec = _Compiler(env).run()
        if any(i.severity == "error" for i in spec.issues):
            raise EnvFileError(spec.issues)
        return spec


def check_env(env: m.EnvFile) -> list[Issue]:
    """오류와 경고 목록. 에디터의 검사 결과 탭이 쓴다."""
    return _Compiler(env).run().issues


# ---------------------------------------------------------------------------
# 모델 안의 수식 칸 찾기
# ---------------------------------------------------------------------------


def _admits_formula(annotation: Any) -> bool:
    """칸의 타입이 수식(Formula)을 받는가. Num, Vec2, list[Vec2] 등 안쪽까지 본다."""
    if annotation is m.Formula:
        return True
    return any(_admits_formula(a) for a in get_args(annotation))


@functools.cache
def _formula_fields(cls: type[BaseModel]) -> frozenset[str]:
    return frozenset(name for name, f in cls.model_fields.items() if _admits_formula(f.annotation))


def iter_formulas(value: Any, path: str, formula_ok: bool = False) -> Iterator[tuple[str, str]]:
    """모델 안에서 수식을 받는 숫자 칸에 들어 있는 '=' 수식을 (경로, 원문)으로 돌려준다.

    이름(label), 설명, 사건 조건처럼 수식 칸이 아닌 글자 칸은 '='로 시작해도 수식으로 보지 않는다.
    """
    if isinstance(value, BaseModel):
        allowed = _formula_fields(type(value))
        for name in type(value).model_fields:
            yield from iter_formulas(getattr(value, name), f"{path}.{name}" if path else name, name in allowed)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from iter_formulas(v, f"{path}[{i}]", formula_ok)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from iter_formulas(v, f"{path}.{k}", formula_ok)
    elif formula_ok and isinstance(value, str) and is_formula(value):
        yield path, value


def _index(path: str, prefix: str) -> int:
    return int(path[len(prefix) : path.index("]")])


def _stage_of(env: m.EnvFile, path: str) -> sy.BuildStage:
    """칸이 계산되는 만들기 단계. 런타임(Episode)도 같은 규칙으로 계산한다."""
    if path.startswith("generators"):
        return "generator"
    if path.startswith("bodies["):
        if path.split(".", 1)[1:] == ["exists"]:
            return "structure"  # 있음 칸(v2.1): p.*만(spec/structure.present가 같은 단계로 계산한다)
        return "agent" if env.bodies[_index(path, "bodies[")].owner == "agent" else "world"
    if path.startswith("joints["):
        return sy.joint_stage(env, env.joints[_index(path, "joints[")])
    if path.startswith(("actuators", "sensors", "agent", "scoring")):
        return "agent"
    return "world"


class _Compiler:
    def __init__(self, env: m.EnvFile) -> None:
        self.env = env
        self.issues: list[Issue] = []
        self.resolver = sy.QueryResolver(env)

    def error(self, where: str, message: str) -> None:
        self.issues.append(Issue("error", where, message))

    def warn(self, where: str, message: str) -> None:
        self.issues.append(Issue("warning", where, message))

    def info(self, where: str, message: str) -> None:
        self.issues.append(Issue("info", where, message))

    def compile(
        self, where: str, source: str, symbols: dict[str, Sym], *, want: Literal["num", "bool"], max_phase: str
    ) -> Compiled | None:
        try:
            return compile_expr(source, symbols, want=want, max_phase=max_phase, query=self.resolver)  # type: ignore[arg-type]
        except ExprError as exc:
            self.error(where, str(exc))
            return None

    # ------------------------------------------------------------------
    def run(self) -> EnvSpec:
        env = self.env
        self._check_world()
        self._check_ids()
        self._check_params()
        self._check_materials()
        self._check_refs()
        self._check_not_yet()
        self._check_v2()
        formulas = self._compile_formulas()
        events, order = self._compile_events()
        blocks, formula = self._compile_scoring()
        self._check_structure(events, blocks, formula)
        sensors = {c.name: c for c in sy.sensor_channels(env)}
        actuators = {c.name: c for c in sy.actuator_channels(env)}
        inputs, outputs = self._check_brain(sensors, actuators)
        terminations = self._check_termination()
        return EnvSpec(
            file=env,
            issues=self.issues,
            formulas=formulas,
            events=events,
            event_order=order,
            blocks=blocks,
            formula=formula,
            queries=dict(self.resolver.queries),
            inputs=inputs,
            outputs=outputs,
            sensor_channels=sensors,
            actuator_channels=actuators,
            terminations=terminations,
        )

    # ------------------------------------------------------------------
    def _check_world(self) -> None:
        w = self.env.world
        if w.kind != "physics2d":
            self.error("world.kind", S.SPEC_GRID_NOT_YET)
        if isinstance(w.bounds[0], float) and isinstance(w.bounds[2], float) and w.bounds[0] >= w.bounds[2]:
            self.error("world.bounds", S.SPEC_BOUNDS_ORDER)
        if isinstance(w.bounds[1], float) and isinstance(w.bounds[3], float) and w.bounds[1] >= w.bounds[3]:
            self.error("world.bounds", S.SPEC_BOUNDS_ORDER)

    def _check_ids(self) -> None:
        seen: dict[str, str] = {}
        for group, i in sy.component_ids(self.env):
            if i in sy.RESERVED:
                self.error(group, S.SPEC_ID_RESERVED.format(id=i))
            if i in seen:
                self.error(group, S.SPEC_ID_DUPLICATE.format(id=i, other=seen[i]))
            seen.setdefault(i, group)
        for group, ids in (("params", [p.id for p in self.env.params]), ("randomize", [v.id for v in self.env.randomize])):
            dup = {i for i in ids if ids.count(i) > 1}
            for i in sorted(dup):
                self.error(group, S.SPEC_ID_DUPLICATE.format(id=i, other=group))
            for i in ids:
                if i in sy.RESERVED:
                    self.error(group, S.SPEC_ID_RESERVED.format(id=i))
        all_ids = [i for _, i in sy.component_ids(self.env)] + [p.id for p in self.env.params] + [
            v.id for v in self.env.randomize
        ]
        for i in sorted({i for i in all_ids if "__" in i}):
            self.error("", S.SPEC_ID_DUNDER.format(id=i))

    def _check_params(self) -> None:
        for i, p in enumerate(self.env.params):
            where = f"params[{i}]"
            if p.evolve:
                if p.min is None or p.max is None:
                    self.error(where, S.SPEC_PARAM_RANGE_MISSING.format(id=p.id))
                elif not p.min < p.max:
                    self.error(where, S.SPEC_PARAM_RANGE_ORDER.format(id=p.id))
                elif not p.min <= p.value <= p.max:
                    self.error(where, S.SPEC_PARAM_VALUE_OUT.format(id=p.id))
        for i, v in enumerate(self.env.randomize):
            where = f"randomize[{i}]"
            numbers = [x for x in (v.low, v.high, v.mean, v.std, *(v.values or [])) if x is not None]
            if any(not math.isfinite(x) or abs(x) > RAND_LIMIT for x in numbers):
                self.error(where, S.SPEC_RAND_HUGE.format(id=v.id))
                continue
            if v.dist in ("uniform", "int"):
                if v.low is None or v.high is None or not v.low <= v.high:
                    self.error(where, S.SPEC_RAND_RANGE.format(id=v.id))
            elif v.dist == "normal":
                if v.mean is None or v.std is None or v.std < 0:
                    self.error(where, S.SPEC_RAND_NORMAL.format(id=v.id))
            elif v.dist == "choice" and not v.values:
                self.error(where, S.SPEC_RAND_CHOICE.format(id=v.id))

    def _check_materials(self) -> None:
        """숫자로 적은 재료 값이 음수면 알려 준다(수식 결과는 런타임이 범위를 잡는다)."""
        for i, b in enumerate(self.env.bodies):
            for name in ("density", "friction", "elasticity"):
                value = getattr(b.material, name)
                if isinstance(value, float) and value < 0:
                    self.error(f"bodies[{i}].material.{name}", S.SPEC_NEGATIVE.format(name=name))
        for i, g in enumerate(self.env.generators):
            if isinstance(g, m.HeightmapGen) and isinstance(g.friction, float) and g.friction < 0:
                self.error(f"generators[{i}].friction", S.SPEC_NEGATIVE.format(name="friction"))

    def _check_refs(self) -> None:
        env = self.env
        bodies = set(sy.body_ids(env))
        agent_bodies = set(sy.agent_body_ids(env))
        regions = set(sy.region_ids(env))
        tags = set(sy.all_tags(env))
        joints = {j.id: j for j in env.joints}
        tanks = {t.id for t in sy.tanks(env)}

        def body_ref(where: str, ref: str) -> None:
            if ref not in bodies:
                self.error(where, S.SPEC_REF_BODY.format(ref=ref))

        dynamic = {b.id for b in env.bodies if b.type == "dynamic"}
        agent_dynamic = {b.id for b in env.bodies if b.type == "dynamic" and b.owner == "agent"}

        def target_ref(where: str, ref: str, allow_body: bool) -> None:
            # 정적 물체(지형 등)는 위치가 원점이라 거리 대상으로 쓰면 뜻이 없다. 영역을 쓰게 한다.
            if ref in regions or (allow_body and ref in dynamic):
                return
            self.error(where, S.SPEC_REF_REGION.format(ref=ref))

        def hit_ref(where: str, ref: str, who: str | None = None) -> None:
            if ref in bodies or (ref.startswith("tag.") and ref[4:] in tags):
                if who is not None and sy.owner_of(env, who) == "agent" and sy.agent_only_selector(env, ref):
                    self.error(where, S.SPEC_AGENT_SELF_CONTACT.format(a=who, b=ref))
                return
            self.error(where, S.SPEC_REF_BODY_OR_TAG.format(ref=ref))

        if env.actuators and not agent_bodies:
            self.error("bodies", S.SPEC_NO_AGENT_BODY)
        own = {b.id: b for b in env.bodies}
        for i, b in enumerate(env.bodies):
            where = f"bodies[{i}]"
            if b.owner == "agent" and b.type == "static":
                self.warn(where, S.SPEC_AGENT_NOT_DYNAMIC.format(id=b.id))
            if b.place_by is not None:
                j = joints.get(b.place_by)
                if j is None:
                    self.error(f"{where}.place_by", S.SPEC_REF_JOINT.format(ref=b.place_by))
                elif j.b != b.id:
                    self.error(f"{where}.place_by", S.SPEC_PLACE_BY_CHILD.format(joint=j.id, body=b.id))
                elif j.a not in own or own[j.a].owner != b.owner:
                    # 부모와 자식의 좌표계(에이전트 좌표 / 세계 좌표)가 같아야 앵커가 맞는다.
                    self.error(f"{where}.place_by", S.SPEC_PLACE_BY_OWNER.format(joint=j.id, body=b.id))
        if self._place_cycle():
            self.error("bodies", S.SPEC_PLACE_CYCLE)
        for i, j in enumerate(env.joints):
            body_ref(f"joints[{i}].a", j.a)
            body_ref(f"joints[{i}].b", j.b)
            if j.a == j.b:
                self.error(f"joints[{i}]", S.SPEC_JOINT_SAME_BODY.format(id=j.id))
            elif j.a in bodies and j.b in bodies and not (sy.is_dynamic(env, j.a) or sy.is_dynamic(env, j.b)):
                self.error(f"joints[{i}]", S.SPEC_JOINT_NO_DYNAMIC.format(id=j.id))
            if j.collide and sy.owner_of(env, j.a) == "agent" and sy.owner_of(env, j.b) == "agent":
                self.warn(f"joints[{i}].collide", S.SPEC_AGENT_COLLIDE.format(id=j.id))
        for i, a in enumerate(env.actuators):
            where = f"actuators[{i}]"
            if isinstance(a, (m.Thruster, m.Vehicle, m.Jump, m.Aero)):
                body_ref(f"{where}.body", a.body)
                if a.body in bodies and a.body not in agent_dynamic:
                    self.error(f"{where}.body", S.SPEC_ACTUATOR_BODY.format(ref=a.body))
            if isinstance(a, m.Jump):
                for k, ref in enumerate(a.ground):
                    hit_ref(f"{where}.ground[{k}]", ref, a.body)
            for k, key in enumerate(_keys_of(a)):
                if sy.key_allowed(a, key):
                    continue
                if key in sy.JUMP_ONLY_KEYS:
                    self.error(f"{where}.keys[{k}]", S.SPEC_KEY_SPACE_JUMP_ONLY.format(id=a.id))
                elif key in sy.RESERVED_KEYS:
                    self.error(f"{where}.keys[{k}]", S.SPEC_KEY_RESERVED.format(key=key))
                else:
                    self.error(f"{where}.keys[{k}]", S.SPEC_KEY_UNKNOWN.format(key=key))
            if isinstance(a, m.Thruster) and a.tank is not None and a.tank not in tanks:
                self.error(f"{where}.tank", S.SPEC_REF_TANK.format(ref=a.tank))
            if isinstance(a, m.Motor) and a.joint not in joints:
                self.error(f"{where}.joint", S.SPEC_REF_JOINT.format(ref=a.joint))
            elif isinstance(a, m.Motor):
                jm = joints[a.joint]
                if jm.kind not in ("pivot", "rot_spring"):
                    self.error(f"{where}.joint", S.SPEC_MOTOR_JOINT_KIND.format(ref=a.joint))
                elif not ({jm.a, jm.b} & agent_dynamic):
                    self.error(f"{where}.joint", S.SPEC_MOTOR_NOT_AGENT.format(ref=a.joint))
        for i, s in enumerate(env.sensors):
            where = f"sensors[{i}]"
            if isinstance(s, (m.BodyState, m.Target, m.ContactSensor, m.Rays)):
                body_ref(f"{where}.body", s.body)
            if isinstance(s, m.Target):
                target_ref(f"{where}.target", s.target, allow_body=True)
            if isinstance(s, m.ContactSensor):
                for k, ref in enumerate(s.with_):
                    hit_ref(f"{where}.with[{k}]", ref, s.body)
            if isinstance(s, m.Rays):
                for k, ref in enumerate(s.hits):
                    hit_ref(f"{where}.hits[{k}]", ref)
            if isinstance(s, m.JointAngle) and s.joint not in joints:
                self.error(f"{where}.joint", S.SPEC_REF_JOINT.format(ref=s.joint))
            if isinstance(s, m.FuelLeft) and s.tank not in tanks:
                self.error(f"{where}.tank", S.SPEC_REF_TANK.format(ref=s.tank))
        events = {e.id for e in env.events} | set(sy.BUILTIN_EVENTS)
        actuator_ids = {a.id for a in env.actuators if not isinstance(a, m.FuelTank)}
        for i, b in enumerate(env.scoring.blocks):
            where = f"scoring.blocks[{i}]"
            if isinstance(b, m.EventBlock) and b.event not in events:
                self.error(f"{where}.event", S.SPEC_REF_EVENT.format(ref=b.event))
            if isinstance(b, (m.DistanceBlock, m.ProgressBlock, m.SpeedBlock, m.TiltBlock, m.ContactTimeBlock)):
                body_ref(f"{where}.body", b.body)
            if isinstance(b, m.DistanceBlock):
                target_ref(f"{where}.target", b.target, allow_body=True)
            if isinstance(b, m.ContactTimeBlock):
                for k, ref in enumerate(b.with_):
                    hit_ref(f"{where}.with[{k}]", ref, b.body)
            if isinstance(b, m.FuelBlock) and b.tank not in tanks:
                self.error(f"{where}.tank", S.SPEC_REF_TANK.format(ref=b.tank))
            if isinstance(b, m.CheckpointBlock) and b.checkpoints not in sy.checkpoint_ids(env):
                self.error(f"{where}.checkpoints", S.SPEC_REF_CHECKPOINTS.format(ref=b.checkpoints))
            if isinstance(b, (m.EnergyBlock, m.SmoothnessBlock)) and b.actuators != "all":
                for k, ref in enumerate(b.actuators):
                    if ref not in actuator_ids:
                        self.error(f"{where}.actuators[{k}]", S.SPEC_REF_ACTUATOR.format(ref=ref))
        for i, c in enumerate(env.checkpoints):
            if c.body is not None:
                body_ref(f"checkpoints[{i}].body", c.body)
            if c.loop and len(c.lines) < 3:
                self.error(f"checkpoints[{i}].lines", S.SPEC_CHECKPOINTS_LOOP_MIN.format(id=c.id))
        tracks = [g for g in env.generators if isinstance(g, m.TrackGen)]
        for i, g in enumerate(env.generators):
            if isinstance(g, m.TrackGen) and g.source == "drawn" and (g.centerline is None or len(g.centerline) < 3):
                self.error(f"generators[{i}].centerline", S.SPEC_TRACK_CENTERLINE)
        if len([g for g in tracks if g.source == "generated"]) > 1:
            self.error("generators", S.SPEC_TRACK_ONE_SET)
        if env.checkpoints or tracks:
            if not [a for a in env.actuators if isinstance(a, m.Vehicle)] and not agent_bodies:
                self.error("checkpoints", S.SPEC_NO_AGENT_BODY)
        if not sy.actuator_channels(env) and sy.evolved_params(env):
            # 뇌 없는 환경(v2 설계 D20): 출력이 없고 몸 수치가 진화하면 일부러 만든 환경이라 경고 대신 안내 하나.
            self.info("brain", S.SPEC_BRAINLESS)
            if env.brain.hidden:
                # 은닉층이 남아 있으면 출력이 없어도 그 가중치가 유전자에 들어가 헛되이 진화한다(D20: 유전자 = 몸 수치).
                # 배치를 바꾸면 이미 있는 실행의 유전자 길이가 달라지므로 고치지 않고 알리기만 한다.
                self.warn("brain.hidden", S.SPEC_BRAINLESS_HIDDEN)
        else:
            if not env.sensors:
                self.warn("sensors", S.SPEC_NO_SENSORS)
            if not [a for a in env.actuators if not isinstance(a, (m.FuelTank, m.Aero))]:
                self.warn("actuators", S.SPEC_NO_ACTUATORS)
        if not env.scoring.blocks and env.scoring.formula is None:
            self.warn("scoring", S.SPEC_NO_SCORE)

    def _place_cycle(self) -> bool:
        joints = {j.id: j for j in self.env.joints}
        parent: dict[str, str] = {}
        for b in self.env.bodies:
            if b.place_by is not None and b.place_by in joints:
                parent[b.id] = joints[b.place_by].a
        for start in parent:
            seen = {start}
            node = start
            while node in parent:
                node = parent[node]
                if node in seen:
                    return True
                seen.add(node)
        return False

    def _check_not_yet(self) -> None:
        def walk(value: Any, path: str) -> None:
            milestone = NOT_YET.get(type(value))
            if milestone is not None:
                self.error(path, S.SPEC_NOT_YET.format(milestone=milestone))
                return
            if isinstance(value, BaseModel):
                for name in type(value).model_fields:
                    walk(getattr(value, name), f"{path}.{name}" if path else name)
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    walk(v, f"{path}[{i}]")

        walk(self.env, "")

    # ------------------------------------------------------------------
    # v2.0 부품 (설계 2.3). 숫자로 적은 칸만 본다(수식 결과는 런타임이 범위를 잡는다).
    # ------------------------------------------------------------------
    def _check_v2(self) -> None:
        env = self.env
        for i, b in enumerate(env.bodies):
            where = f"bodies[{i}]"
            if b.type == "kinematic" and b.owner == "agent":
                self.error(f"{where}.type", S.SPEC_KINEMATIC_AGENT.format(id=b.id))
            if b.motion is not None and b.type != "kinematic":
                self.error(f"{where}.motion", S.SPEC_MOTION_NOT_KINEMATIC.format(id=b.id))
            elif b.motion is None and b.type == "kinematic" and b.owner == "world":
                self.warn(f"{where}.type", S.SPEC_KINEMATIC_NO_MOTION.format(id=b.id))
            if b.motion is not None:
                self._check_motion(f"{where}.motion", b)
            for k, s in enumerate(b.shapes):
                swhere = f"{where}.shapes[{k}]"
                if isinstance(s, m.Wing) and (_below(s.chord, 0.0, strict=False) or _below(s.thick, 0.0, strict=False)):
                    self.error(swhere, S.SPEC_WING_SIZE.format(id=b.id))
                density = getattr(s, "density", None)
                if density is None:
                    continue
                if b.type != "dynamic":
                    self.warn(f"{swhere}.density", S.SPEC_SHAPE_DENSITY_STATIC.format(id=b.id))
                elif _below(density, 0.0):
                    self.error(f"{swhere}.density", S.SPEC_NEGATIVE.format(name="density"))
        for i, r in enumerate(env.regions):
            if getattr(r.shape, "density", None) is not None:
                self.warn(f"regions[{i}].shape.density", S.SPEC_REGION_DENSITY.format(id=r.id))
        own = {b.id: b for b in env.bodies}
        aero_bodies: set[str] = set()
        for i, a in enumerate(env.actuators):
            where = f"actuators[{i}]"
            if isinstance(a, m.Jump):
                if _below(a.speed, 0.0):
                    self.error(f"{where}.speed", S.SPEC_NEGATIVE.format(name="speed"))
            elif isinstance(a, m.Aero):
                if a.body in aero_bodies:  # 같은 테두리를 두 번 센다(상한은 물체마다라 터지지는 않지만 힘이 겹침)
                    self.warn(f"{where}.body", S.SPEC_AERO_TWICE.format(body=a.body))
                aero_bodies.add(a.body)
                for name in ("density", "span", "aspect_ratio", "friction"):
                    if _below(getattr(a, name), 0.0):
                        self.error(f"{where}.{name}", S.SPEC_AERO_RANGE.format(id=a.id, name=name))
                stall = _number(a.stall)
                if stall is not None and not 0.0 < stall <= AERO_STALL_MAX:
                    self.error(f"{where}.stall", S.SPEC_AERO_RANGE.format(id=a.id, name="stall"))
                body = own.get(a.body)
                if body is not None and not any(_takes_air(s, a.surfaces) for s in body.shapes):
                    self.warn(f"{where}.surfaces", S.SPEC_AERO_NO_EDGES.format(id=a.id, body=a.body))
        for i, g in enumerate(env.generators):
            if not isinstance(g, m.ObstacleGen):
                continue
            where = f"generators[{i}]"
            if _below(g.gap[0], 0.0, strict=False) or _below(g.gap[1], 0.0, strict=False):
                self.error(f"{where}.gap", S.SPEC_OBSTACLES_GAP.format(id=g.id))
            values = [_number(v) for v in (g.x, g.y, g.direction)]
            starts = self._agent_starts()
            if starts and all(v is not None for v in values):
                x, y, d = (float(v) for v in values)  # type: ignore[arg-type]
                # 첫 장애물의 앞 모서리는 (0, y) + x·u에서 출발한다(u = (−cos d, −sin d): 줄이 오는 쪽, generators/
                # obstacles.place_boxes). 에이전트 물체의 시작점마다 u 쪽으로 잰 거리 중 가장 짧은 것. 방향 π면 x − 시작 x.
                ux, uy = -math.cos(d), -math.sin(d)
                lead = min(x + (0.0 - px) * ux + (y - py) * uy for px, py in starts)
                if lead < OBSTACLES_MIN_LEAD:
                    self.warn(f"{where}.x", S.SPEC_OBSTACLES_START.format(id=g.id))

    def _agent_starts(self) -> list[tuple[float, float]]:
        """에이전트 물체마다 숫자로 알 수 있는 세계 시작점(런타임 배치: 시작 위치 + 시작 각도로 돌린 pos). 관절로 놓는
        물체(place_by)나 pos·시작 각도가 수식이면 시작 위치. 시작 위치가 수식이거나 에이전트 물체가 없으면 빈 목록."""
        env = self.env
        sp = env.agent.spawn
        sx, sy_, sa = _number(sp.x), _number(sp.y), _number(sp.angle)
        if sx is None or sy_ is None:
            return []
        out: list[tuple[float, float]] = []
        for b in env.bodies:
            if b.owner != "agent":
                continue
            bx, by = _number(b.pos[0]), _number(b.pos[1])
            if b.place_by is not None or bx is None or by is None or sa is None:
                out.append((sx, sy_))
                continue
            c, s_ = math.cos(sa), math.sin(sa)
            out.append((sx + bx * c - by * s_, sy_ + bx * s_ + by * c))
        return out

    def _check_motion(self, where: str, b: m.Body) -> None:
        mo = b.motion
        if isinstance(mo, m.WaveMotion):
            period = _number(mo.period)
            if period is not None and period <= 0:
                self.error(f"{where}.period", S.SPEC_MOTION_PERIOD.format(id=b.id))
            return
        assert isinstance(mo, m.PathMotion)
        if _number(mo.spin) != 0.0:
            return  # 제자리에서 돈다(또는 수식이라 알 수 없다)
        pts = {(_number(p[0]), _number(p[1])) for p in mo.points}
        still_path = len(pts) == 1 and None not in next(iter(pts))  # 숫자로 적은 점이 모두 같다: 길이 0
        still_speed = _below(mo.speed, 0.0, strict=False) and _below(mo.accel, 0.0, strict=False)
        if still_path or still_speed:
            self.warn(where, S.SPEC_MOTION_STILL.format(id=b.id))

    # ------------------------------------------------------------------
    # v2.1 몸 구조 (설계 2.3, D33·D36·D37). 있음 칸 수식이 p.*만 쓰는지는 만들기 단계 structure가 본다(_compile_formulas).
    # ------------------------------------------------------------------
    def _check_structure(
        self, events: dict[str, Compiled], blocks: dict[str, Compiled], formula: Compiled | None
    ) -> None:
        """있음 칸은 뿌리가 아닌 에이전트 물체에만. 없어질 수 있는 물체(structure.optional_bodies, 런타임과 같은 규칙)를
        사건·점수 블록·점수 수식·체크포인트·목표 센서가 위치로 가리키면 오류(몸 상태 이름, dist, inside, 블록·체크포인트의 물체,
        거리·목표 대상). 접촉(contact(), 접촉 센서, 접촉 시간 블록)과 그 물체에 단 센서·구동기는 된다: 없으면 0·거짓."""
        env = self.env
        root = st.root_body(env)
        root_has_exists = False
        for i, b in enumerate(env.bodies):
            if b.exists is None:
                continue
            if b.owner != "agent":
                self.error(f"bodies[{i}].exists", S.SPEC_EXISTS_WORLD.format(id=b.id))
            elif b.id == root:
                self.error(f"bodies[{i}].exists", S.SPEC_EXISTS_ROOT.format(id=b.id))
                root_has_exists = True
        if root_has_exists:
            # 뿌리의 있음 칸은 위에서 오류로 알렸다. 그것까지 넣으면 모든 부품이 없어질 수 있다고 나와 아래 오류가 쏟아진다.
            bodies = [b.model_copy(update={"exists": None}) if b.id == root else b for b in env.bodies]
            optional = st.optional_bodies(env.model_copy(update={"bodies": bodies}))
        else:
            optional = st.optional_bodies(env)
        if not optional:
            return
        order = {b.id: k for k, b in enumerate(env.bodies)}

        def refuse(where: str, refs: set[str], place: str) -> None:
            for body in sorted(refs, key=order.__getitem__):
                self.error(where, S.SPEC_EXISTS_REF.format(body=body, where=place))

        for i, e in enumerate(env.events):
            if e.id in events:
                refuse(f"events[{i}].when", _optional_names(events[e.id].names, optional),
                       S.SPEC_EXISTS_WHERE_EVENT.format(id=e.id))
        for i, b in enumerate(env.scoring.blocks):
            where, place = f"scoring.blocks[{i}]", S.SPEC_EXISTS_WHERE_BLOCK.format(id=b.id)
            if isinstance(b, m.ExprBlock) and b.id in blocks:
                refuse(f"{where}.expr", _optional_names(blocks[b.id].names, optional), place)
            if isinstance(b, (m.DistanceBlock, m.ProgressBlock, m.SpeedBlock, m.TiltBlock)):
                refuse(f"{where}.body", {b.body} & optional, place)
            if isinstance(b, m.DistanceBlock):
                refuse(f"{where}.target", {b.target} & optional, place)
        if formula is not None:
            refuse("scoring.formula", _optional_names(formula.names, optional), S.SPEC_EXISTS_WHERE_FORMULA)
        # 체크포인트의 물체를 비우면 차량 구동기의 물체, 그것도 없으면 뿌리(늘 있음)를 쓴다(런타임 _build_checkpoints).
        vehicles = [a.body for a in env.actuators if isinstance(a, m.Vehicle)]
        default = vehicles[0] if vehicles else root
        for i, c in enumerate(env.checkpoints):
            refuse(f"checkpoints[{i}].body", {c.body if c.body is not None else default} & optional,
                   S.SPEC_EXISTS_WHERE_CHECKPOINTS.format(id=c.id))
        for i, g in enumerate(env.generators):
            for cid in sy.generator_checkpoints(g):
                refuse(f"generators[{i}]", {default} & optional, S.SPEC_EXISTS_WHERE_CHECKPOINTS.format(id=cid))
        for i, s in enumerate(env.sensors):
            if isinstance(s, m.Target):
                refuse(f"sensors[{i}].target", {s.target} & optional, S.SPEC_EXISTS_WHERE_TARGET.format(id=s.id))

    def _compile_formulas(self) -> dict[tuple[str, sy.BuildStage], Compiled]:
        out: dict[tuple[str, sy.BuildStage], Compiled] = {}
        tables = {stage: sy.build_symbols(self.env, stage) for stage in ("generator", "world", "agent", "structure")}
        for path, source in iter_formulas(self.env, ""):
            stage = _stage_of(self.env, path)
            key = (source, stage)
            if key in out:
                continue
            body = formula_body(source)
            compiled = self.compile(path, body, tables[stage], want="num", max_phase="build")  # type: ignore[arg-type]
            if compiled is not None:
                out[key] = compiled
        return out

    def _compile_events(self) -> tuple[dict[str, Compiled], list[str]]:
        symbols = sy.step_symbols(self.env)
        out: dict[str, Compiled] = {}
        for i, e in enumerate(self.env.events):
            c = self.compile(f"events[{i}].when", e.when, symbols, want="bool", max_phase="step")
            if c is not None:
                out[e.id] = c
        order = sy.event_order(self.env, {k: v.names for k, v in out.items()})
        if order is None:
            self.error("events", S.SPEC_EVENT_CYCLE)
            order = [e.id for e in self.env.events]
        return out, [e for e in order if e in out]

    def _compile_scoring(self) -> tuple[dict[str, Compiled], Compiled | None]:
        step = sy.step_symbols(self.env)
        blocks: dict[str, Compiled] = {}
        for i, b in enumerate(self.env.scoring.blocks):
            if isinstance(b, m.ExprBlock):
                c = self.compile(f"scoring.blocks[{i}].expr", b.expr, step, want="num", max_phase="step")
                if c is not None:
                    blocks[b.id] = c
        formula = None
        if self.env.scoring.formula is not None:
            formula = self.compile(
                "scoring.formula", self.env.scoring.formula, sy.end_symbols(self.env), want="num", max_phase="end"
            )
        return blocks, formula

    def _check_brain(
        self, sensors: dict[str, sy.Channel], actuators: dict[str, sy.Channel]
    ) -> tuple[list[str], list[str]]:
        inputs = sy.brain_inputs(self.env)
        outputs = sy.brain_outputs(self.env)
        for k, name in enumerate(inputs):
            if name not in sensors:
                self.error(f"brain.inputs[{k}]", S.SPEC_REF_CHANNEL.format(ref=name))
        for k, name in enumerate(outputs):
            if name not in actuators:
                self.error(f"brain.outputs[{k}]", S.SPEC_REF_CHANNEL.format(ref=name))
        for group, names in (("brain.inputs", inputs), ("brain.outputs", outputs)):
            dup = sorted({n for n in names if names.count(n) > 1})
            for n in dup:
                self.error(group, S.SPEC_ID_DUPLICATE.format(id=n, other=group))
        return inputs, outputs

    def _check_termination(self) -> list[m.End]:
        events = {e.id for e in self.env.events} | set(sy.BUILTIN_EVENTS)
        ends = list(self.env.termination)
        for i, t in enumerate(ends):
            if t.event not in events:
                self.error(f"termination[{i}].event", S.SPEC_REF_EVENT.format(ref=t.event))
        if not any(t.event == "timeout" for t in ends):
            ends.append(m.End(event="timeout", outcome="neutral", label=S.END_TIMEOUT))
        return ends


def _optional_names(names: frozenset[str], optional: frozenset[str]) -> set[str]:
    """수식이 읽는 이름 중 없어질 수 있는 물체를 위치로 가리키는 것의 물체 id: 몸 상태(<물체>.x 등), dist(a, b)의 a·b,
    inside(a, 영역)의 a. 접촉 질의(contact)와 센서·구동기 채널 이름은 넣지 않는다(없으면 0·거짓)."""
    out: set[str] = set()
    for n in names:
        if n.startswith(("dist:", "inside:")):  # 질의 채널 이름 '<함수>:<a>:<b>'(symbols.Query.channel)
            out.update(x for x in n.split(":", 2)[1:] if x in optional)
            continue
        if n.startswith("contact:"):
            continue
        head, dot, tail = n.rpartition(".")
        if dot and tail in sy.BODY_STATE and head in optional:
            out.add(head)
    return out


def _keys_of(a: m.Actuator) -> list[str]:
    if isinstance(a, m.Thruster):
        return list(a.keys)
    if isinstance(a, m.Vehicle):
        return [*a.keys_forward, *a.keys_back, *a.keys_left, *a.keys_right]
    if isinstance(a, m.Motor):
        return [*a.keys_pos, *a.keys_neg]
    if isinstance(a, m.Jump):
        return list(a.keys)
    return []


def _number(value: Any) -> float | None:
    """숫자로 적은 칸이면 그 값, 수식이면 None."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _below(value: Any, limit: float, *, strict: bool = True) -> bool:
    """숫자로 적은 칸이 limit보다 작은가(strict=False면 limit 이하도). 수식이면 False(런타임이 범위를 잡는다)."""
    x = _number(value)
    return x is not None and (x < limit if strict else x <= limit)


def _takes_air(shape: Any, surfaces: str) -> bool:
    """공기역학 부품이 이 모양의 테두리에 공기힘을 거는가(원·선은 받지 않음, wings면 날개 모양만)."""
    if surfaces == "wings":
        return isinstance(shape, m.Wing)
    return isinstance(shape, (m.Box, m.Poly, m.Radial, m.Wing))


def finite(x: float) -> float:
    """수식 결과가 유한하지 않으면 0. 물리 값에 NaN이 들어가는 것을 막는다."""
    return x if math.isfinite(x) else 0.0

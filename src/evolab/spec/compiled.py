"""환경 파일 검사와 컴파일.

EnvSpec은 검사를 통과한 환경 파일 하나와, 거기서 미리 만들어 둔 것들(수식, 사건 순서, 채널 목록)이다.
워커는 환경 버전마다 한 번 만들고 시험마다 재사용한다.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from evolab.app import strings as S
from evolab.scoring.expr import Compiled, ExprError, Sym, compile_expr, formula_body, is_formula
from evolab.spec import model as m
from evolab.spec import symbols as sy

Severity = Literal["error", "warning"]
RAND_LIMIT = 1e15


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
NOT_YET: dict[type[BaseModel], str] = {
    m.Vehicle: "M4",
    m.Motor: "M5b",
    m.Rays: "M4",
    m.TrackGen: "M4",
    m.CheckpointBlock: "M4",
}


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


def iter_formulas(value: Any, path: str) -> Iterator[tuple[str, str]]:
    """모델 안에서 '='로 시작하는 문자열 칸을 모두 (경로, 원문)으로 돌려준다."""
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from iter_formulas(getattr(value, name), f"{path}.{name}" if path else name)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from iter_formulas(v, f"{path}[{i}]")
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from iter_formulas(v, f"{path}.{k}")
    elif isinstance(value, str) and is_formula(value):
        yield path, value


def _stage_of(env: m.EnvFile, path: str) -> sy.BuildStage:
    if path.startswith("generators"):
        return "generator"
    if path.startswith("bodies["):
        index = int(path[len("bodies[") : path.index("]")])
        return "agent" if env.bodies[index].owner == "agent" else "world"
    if path.startswith(("joints", "actuators", "sensors", "agent", "scoring")):
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
        formulas = self._compile_formulas()
        events, order = self._compile_events()
        blocks, formula = self._compile_scoring()
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

        def target_ref(where: str, ref: str, allow_body: bool) -> None:
            if ref in regions or (allow_body and ref in bodies):
                return
            self.error(where, S.SPEC_REF_REGION.format(ref=ref))

        def hit_ref(where: str, ref: str) -> None:
            if ref in bodies or (ref.startswith("tag.") and ref[4:] in tags):
                return
            self.error(where, S.SPEC_REF_BODY_OR_TAG.format(ref=ref))

        if env.actuators and not agent_bodies:
            self.error("bodies", S.SPEC_NO_AGENT_BODY)
        for i, b in enumerate(env.bodies):
            where = f"bodies[{i}]"
            if b.owner == "agent" and b.type != "dynamic":
                self.warn(where, S.SPEC_AGENT_NOT_DYNAMIC.format(id=b.id))
            if b.place_by is not None:
                j = joints.get(b.place_by)
                if j is None:
                    self.error(f"{where}.place_by", S.SPEC_REF_JOINT.format(ref=b.place_by))
                elif j.b != b.id:
                    self.error(f"{where}.place_by", S.SPEC_PLACE_BY_CHILD.format(joint=j.id, body=b.id))
        if self._place_cycle():
            self.error("bodies", S.SPEC_PLACE_CYCLE)
        for i, j in enumerate(env.joints):
            body_ref(f"joints[{i}].a", j.a)
            body_ref(f"joints[{i}].b", j.b)
            if j.a == j.b:
                self.error(f"joints[{i}]", S.SPEC_JOINT_SAME_BODY.format(id=j.id))
        for i, a in enumerate(env.actuators):
            where = f"actuators[{i}]"
            if isinstance(a, (m.Thruster, m.Vehicle)):
                body_ref(f"{where}.body", a.body)
            if isinstance(a, m.Thruster) and a.tank is not None and a.tank not in tanks:
                self.error(f"{where}.tank", S.SPEC_REF_TANK.format(ref=a.tank))
            if isinstance(a, m.Motor) and a.joint not in joints:
                self.error(f"{where}.joint", S.SPEC_REF_JOINT.format(ref=a.joint))
        for i, s in enumerate(env.sensors):
            where = f"sensors[{i}]"
            if isinstance(s, (m.BodyState, m.Target, m.ContactSensor, m.Rays)):
                body_ref(f"{where}.body", s.body)
            if isinstance(s, m.Target):
                target_ref(f"{where}.target", s.target, allow_body=True)
            if isinstance(s, m.ContactSensor):
                for k, ref in enumerate(s.with_):
                    hit_ref(f"{where}.with[{k}]", ref)
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
                    hit_ref(f"{where}.with[{k}]", ref)
            if isinstance(b, m.FuelBlock) and b.tank not in tanks:
                self.error(f"{where}.tank", S.SPEC_REF_TANK.format(ref=b.tank))
            if isinstance(b, (m.EnergyBlock, m.SmoothnessBlock)) and b.actuators != "all":
                for k, ref in enumerate(b.actuators):
                    if ref not in actuator_ids:
                        self.error(f"{where}.actuators[{k}]", S.SPEC_REF_ACTUATOR.format(ref=ref))
        if not env.sensors:
            self.warn("sensors", S.SPEC_NO_SENSORS)
        if not [a for a in env.actuators if not isinstance(a, m.FuelTank)]:
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
        for i, b in enumerate(self.env.bodies):
            if b.type == "kinematic":
                self.error(f"bodies[{i}].type", S.SPEC_NOT_YET.format(milestone="2"))
        if self.env.checkpoints:
            self.error("checkpoints", S.SPEC_NOT_YET.format(milestone="M4"))

    def _compile_formulas(self) -> dict[tuple[str, sy.BuildStage], Compiled]:
        out: dict[tuple[str, sy.BuildStage], Compiled] = {}
        tables = {st: sy.build_symbols(self.env, st) for st in ("generator", "world", "agent")}
        for path, source in iter_formulas(self.env, ""):
            if path.startswith(("editor", "events", "scoring")):
                continue
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


def finite(x: float) -> float:
    """수식 결과가 유한하지 않으면 0. 물리 값에 NaN이 들어가는 것을 막는다."""
    return x if math.isfinite(x) else 0.0

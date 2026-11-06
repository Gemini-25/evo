"""환경 파일 안의 이름들: 수식에서 쓸 수 있는 기호, 센서 채널, 구동기 채널, 참조 대상.

검사기(spec/check.py)와 런타임(world/physics2d)이 같은 표를 쓴다. 그래야 검사를 통과한 수식이 런타임에서도
같은 뜻이 된다.

단계:
- build: 만들기 단계. p.<매개변수>, r.<랜덤 값>, gen.<생성기 값>, pi. 칸마다 쓸 수 있는 묶음이 다르다.
- step: 매 스텝. 물체 상태(<물체>.x 등), 센서 채널, 구동기 출력, 연료, 사건, time, 질의 함수.
- end: 종료 시. 위의 모든 것 + 점수 블록 id, success / failure / neutral.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from evolab.scoring.expr import Sym
from evolab.spec import model as m

BODY_STATE: tuple[str, ...] = ("x", "y", "vx", "vy", "speed", "angle", "omega")
BUILTIN_EVENTS: tuple[str, ...] = ("timeout", "out_of_bounds")
END_FLAGS: tuple[str, ...] = ("success", "failure", "neutral")
RESERVED: frozenset[str] = frozenset(
    {"time", "pi", "p", "r", "gen", "tag", "True", "False", "and", "or", "not", "if", "else", "contact", "inside", "dist"}
    | set(BUILTIN_EVENTS)
    | set(END_FLAGS)
)

BuildStage = Literal["generator", "world", "agent"]

NUM_BUILD = Sym("num", "build")
NUM_STEP = Sym("num", "step")
BOOL_STEP = Sym("bool", "step")
NUM_END = Sym("num", "end")
BOOL_END = Sym("bool", "end")

DEFAULT_STATE_SCALE: dict[str, float] = {
    "x": 50.0,
    "y": 50.0,
    "vx": 10.0,
    "vy": 10.0,
    "speed": 10.0,
    "v_fwd": 10.0,
    "v_lat": 10.0,
    "angle": 3.141592653589793,
    "omega": 4.0,
}


@dataclass(frozen=True)
class Channel:
    """신경망 입력(센서) 또는 출력(구동기) 하나."""

    name: str
    owner: str
    """센서나 구동기 id."""
    low: float
    high: float
    type: Literal["num", "bool"] = "num"


# ---------------------------------------------------------------------------
# 생성기 출력
# ---------------------------------------------------------------------------


def generator_values(gen: m.Generator) -> list[str]:
    """gen.<id>.* 로 쓸 수 있는 값 이름(앞의 'gen.'은 뺀 것)."""
    if isinstance(gen, m.HeightmapGen):
        names = [f"{gen.id}.x_min", f"{gen.id}.x_max"]
        if gen.pad is not None:
            names += [f"{gen.id}.pad.{k}" for k in ("x", "y", "w", "left", "right")]
        return names
    if isinstance(gen, m.TrackGen):
        return [f"{gen.id}.start.x", f"{gen.id}.start.y", f"{gen.id}.start.angle", f"{gen.id}.length"]
    return []


def generator_bodies(gen: m.Generator) -> dict[str, list[str]]:
    """생성기가 만드는 물체 id와 그 모양 태그."""
    if isinstance(gen, m.HeightmapGen):
        tags = ["terrain", "pad"] if gen.pad is not None else ["terrain"]
        return {gen.id: tags}
    if isinstance(gen, m.TrackGen):
        return {f"{gen.id}.walls": ["wall"]}
    return {}


def generator_regions(gen: m.Generator) -> list[str]:
    if isinstance(gen, m.HeightmapGen) and gen.pad is not None:
        return [f"{gen.id}.pad"]
    return []


def generator_checkpoints(gen: m.Generator) -> list[str]:
    if isinstance(gen, m.TrackGen):
        return [f"{gen.id}.cp"]
    return []


# ---------------------------------------------------------------------------
# 이름 모음
# ---------------------------------------------------------------------------


def body_ids(env: m.EnvFile) -> list[str]:
    out = [b.id for b in env.bodies]
    for g in env.generators:
        out += list(generator_bodies(g))
    return out


def agent_body_ids(env: m.EnvFile) -> list[str]:
    return [b.id for b in env.bodies if b.owner == "agent"]


def region_ids(env: m.EnvFile) -> list[str]:
    out = [r.id for r in env.regions]
    for g in env.generators:
        out += generator_regions(g)
    return out


def checkpoint_ids(env: m.EnvFile) -> list[str]:
    out = [c.id for c in env.checkpoints]
    for g in env.generators:
        out += generator_checkpoints(g)
    return out


def all_tags(env: m.EnvFile) -> list[str]:
    tags: list[str] = []
    for b in env.bodies:
        tags += b.tags
    for r in env.regions:
        tags += r.tags
    for g in env.generators:
        for t in generator_bodies(g).values():
            tags += t
    return sorted(set(tags))


def tanks(env: m.EnvFile) -> list[m.FuelTank]:
    return [a for a in env.actuators if isinstance(a, m.FuelTank)]


def sensor_channels(env: m.EnvFile) -> list[Channel]:
    """센서 채널, 선언 순서. 관측값 범위는 정규화 뒤 기준."""
    out: list[Channel] = []
    for s in env.sensors:
        if isinstance(s, m.BodyState):
            out += [Channel(f"{s.id}.{f}", s.id, -1.0, 1.0) for f in s.fields]
        elif isinstance(s, m.Target):
            out += [Channel(f"{s.id}.x", s.id, -1.0, 1.0), Channel(f"{s.id}.y", s.id, -1.0, 1.0)]
        elif isinstance(s, m.ContactSensor):
            out.append(Channel(s.id, s.id, 0.0, 1.0, "bool"))
        elif isinstance(s, (m.JointAngle,)):
            out.append(Channel(s.id, s.id, -1.0, 1.0))
        elif isinstance(s, (m.FuelLeft, m.TimeLeft)):
            out.append(Channel(s.id, s.id, 0.0, 1.0))
        elif isinstance(s, m.Rays):
            out += [Channel(f"{s.id}.{i}", s.id, 0.0, 1.0) for i in range(s.count)]
    return out


def actuator_channels(env: m.EnvFile) -> list[Channel]:
    """구동기 출력 채널, 선언 순서. 범위는 구동기 고유 범위(0..1 또는 -1..1)."""
    out: list[Channel] = []
    for a in env.actuators:
        if isinstance(a, m.Thruster):
            out.append(Channel(a.id, a.id, 0.0, 1.0))
        elif isinstance(a, m.Vehicle):
            out += [Channel(f"{a.id}.throttle", a.id, -1.0, 1.0), Channel(f"{a.id}.steer", a.id, -1.0, 1.0)]
        elif isinstance(a, m.Motor):
            out.append(Channel(a.id, a.id, -1.0, 1.0))
    return out


def brain_inputs(env: m.EnvFile) -> list[str]:
    names = [c.name for c in sensor_channels(env)]
    return names if env.brain.inputs == "all" else list(env.brain.inputs)


def brain_outputs(env: m.EnvFile) -> list[str]:
    names = [c.name for c in actuator_channels(env)]
    return names if env.brain.outputs == "all" else list(env.brain.outputs)


def evolved_params(env: m.EnvFile) -> list[m.Param]:
    """몸 유전자 순서."""
    return [p for p in env.params if p.evolve]


def component_ids(env: m.EnvFile) -> Iterator[tuple[str, str]]:
    """(구역 이름, id). 수식 이름이 겹치지 않게 모든 부품 id는 서로 달라야 한다."""
    groups: list[tuple[str, list[str]]] = [
        ("generators", [g.id for g in env.generators]),
        ("bodies", [b.id for b in env.bodies]),
        ("joints", [j.id for j in env.joints]),
        ("actuators", [a.id for a in env.actuators]),
        ("sensors", [s.id for s in env.sensors]),
        ("regions", [r.id for r in env.regions]),
        ("checkpoints", [c.id for c in env.checkpoints]),
        ("events", [e.id for e in env.events]),
        ("scoring.blocks", [b.id for b in env.scoring.blocks]),
    ]
    for group, ids in groups:
        for i in ids:
            yield group, i


# ---------------------------------------------------------------------------
# 기호 표
# ---------------------------------------------------------------------------


def build_symbols(env: m.EnvFile, stage: BuildStage) -> dict[str, Sym]:
    """만들기 단계 칸에서 쓸 수 있는 이름.

    generator: r.* (생성기는 모든 개체가 같은 세계를 쓰도록 p.*를 못 쓴다)
    world: r.*, gen.* (세계 설정, 세계 물체, 영역)
    agent: p.*, r.*, gen.* (에이전트 물체, 관절, 구동기, 센서, 시작 위치)
    """
    out: dict[str, Sym] = {f"r.{v.id}": NUM_BUILD for v in env.randomize}
    if stage in ("world", "agent"):
        for g in env.generators:
            out.update({f"gen.{n}": NUM_BUILD for n in generator_values(g)})
    if stage == "agent":
        out.update({f"p.{p.id}": NUM_BUILD for p in env.params})
    return out


def step_symbols(env: m.EnvFile) -> dict[str, Sym]:
    out = build_symbols(env, "agent")
    out["time"] = NUM_STEP
    for b in body_ids(env):
        out.update({f"{b}.{f}": NUM_STEP for f in BODY_STATE})
    for c in sensor_channels(env):
        out[c.name] = BOOL_STEP if c.type == "bool" else NUM_STEP
    for c in actuator_channels(env):
        out[c.name] = NUM_STEP
    for t in tanks(env):
        out[t.id] = NUM_STEP
    for e in [*BUILTIN_EVENTS, *(ev.id for ev in env.events)]:
        out[e] = BOOL_STEP
        out[f"{e}.count"] = NUM_STEP
        out[f"{e}.since"] = NUM_STEP
    return out


def end_symbols(env: m.EnvFile) -> dict[str, Sym]:
    out = step_symbols(env)
    for b in env.scoring.blocks:
        out[b.id] = NUM_END
    for f in END_FLAGS:
        out[f] = BOOL_END
    return out


@dataclass(frozen=True)
class Query:
    """수식 안의 질의 함수 하나. 런타임은 이 채널 값을 매 스텝 계산한다."""

    func: Literal["contact", "inside", "dist"]
    a: str
    b: str

    @property
    def channel(self) -> str:
        return f"{self.func}:{self.a}:{self.b}"


class QueryResolver:
    """수식 컴파일 때 contact / inside / dist 호출을 채널 이름으로 바꾼다. 쓰인 질의를 모아 둔다."""

    def __init__(self, env: m.EnvFile) -> None:
        self._bodies = set(body_ids(env))
        self._regions = set(region_ids(env))
        self._tags = set(all_tags(env))
        self.queries: dict[str, Query] = {}

    def __call__(self, func: str, args: tuple[str, ...]) -> tuple[str, Sym]:
        from evolab.app import strings as S
        from evolab.scoring.expr import ExprError

        if len(args) != 2:
            raise ExprError(S.SPEC_QUERY_ARGS.format(func=func))
        a, b = args
        if a not in self._bodies:
            raise ExprError(S.SPEC_QUERY_FIRST_BODY.format(func=func, name=a))
        if func == "contact":
            ok = b in self._bodies or (b.startswith("tag.") and b[4:] in self._tags)
            want = S.SPEC_QUERY_WANT_BODY_OR_TAG
        elif func == "inside":
            ok = b in self._regions
            want = S.SPEC_QUERY_WANT_REGION
        else:
            ok = b in self._regions or b in self._bodies
            want = S.SPEC_QUERY_WANT_REGION_OR_BODY
        if not ok:
            raise ExprError(S.SPEC_QUERY_BAD_SECOND.format(func=func, name=b, want=want))
        q = Query(func, a, b)  # type: ignore[arg-type]
        self.queries[q.channel] = q
        return q.channel, (NUM_STEP if func == "dist" else BOOL_STEP)


def event_order(env: m.EnvFile, uses: dict[str, frozenset[str]]) -> list[str] | None:
    """사건을 서로 참조하는 순서대로 정렬한다. uses: 사건 id -> 그 when 수식이 읽는 이름들.

    순환이 있으면 None.
    """
    ids = [e.id for e in env.events]
    deps: dict[str, set[str]] = {}
    for e in ids:
        names = uses.get(e, frozenset())
        deps[e] = {d for d in ids if d != e and any(n == d or n.startswith(d + ".") for n in names)}
        if any(n == e or n.startswith(e + ".") for n in names):
            deps[e].add(e)
    order: list[str] = []
    state: dict[str, int] = {}

    def visit(n: str) -> bool:
        if state.get(n) == 2:
            return True
        if state.get(n) == 1:
            return False
        state[n] = 1
        for d in sorted(deps[n]):
            if d == n or not visit(d):
                return False
        state[n] = 2
        order.append(n)
        return True

    for e in ids:
        if not visit(e):
            return None
    return order

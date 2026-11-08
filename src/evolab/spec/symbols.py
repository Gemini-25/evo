"""환경 파일 안의 이름들: 수식에서 쓸 수 있는 기호, 센서 채널, 구동기 채널, 참조 대상.

검사기(spec/check.py)와 런타임(world/physics2d)이 같은 표를 쓴다. 그래야 검사를 통과한 수식이 런타임에서도
같은 뜻이 된다.

단계:
- build: 만들기 단계. p.<매개변수>, r.<랜덤 값>, gen.<생성기 값>, pi. 칸마다 쓸 수 있는 묶음이 다르다.
- step: 매 스텝. 물체 상태(<물체>.x 등), 센서 채널, 구동기 출력, 연료, 사건, time, 질의 함수.
- end: 종료 시. 위의 모든 것 + 점수 블록 id, success / failure / neutral.
"""

from __future__ import annotations

import keyword
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from evolab.scoring.expr import Sym
from evolab.spec import model as m

BODY_STATE: tuple[str, ...] = ("x", "y", "vx", "vy", "speed", "angle", "omega")
BUILTIN_EVENTS: tuple[str, ...] = ("timeout", "out_of_bounds")
END_FLAGS: tuple[str, ...] = ("success", "failure", "neutral")
RESERVED: frozenset[str] = frozenset(
    {"time", "pi", "p", "r", "gen", "tag", "None", "contact", "inside", "dist"}
    | set(keyword.kwlist)
    | set(BUILTIN_EVENTS)
    | set(END_FLAGS)
)

# 직접 조종 키 이름(Qt 키 이름에서 Key_를 뺀 것). R, N, Space는 직접 조종 창 제어용이라 구동기에 쓸 수 없다.
# v2.0부터 Space는 점프 추진기에만 쓸 수 있다(직접 조종 창의 Esc는 늘 일시정지, Space는 환경이 쓰지 않을 때만 일시정지).
CONTROL_KEYS: frozenset[str] = frozenset({"R", "N", "Space"})
JUMP_ONLY_KEYS: frozenset[str] = frozenset({"Space"})
"""점프 추진기(jump)의 keys에만 쓸 수 있는 키(v2 설계 D12)."""
RESERVED_KEYS: frozenset[str] = CONTROL_KEYS - JUMP_ONLY_KEYS
"""어떤 구동기에도 쓸 수 없는 직접 조종 창 제어 키(R 다시 시작, N 새 시드)."""
KEY_NAMES: frozenset[str] = frozenset(
    {chr(c) for c in range(ord("A"), ord("Z") + 1)}
    | {str(d) for d in range(10)}
    | {f"F{i}" for i in range(1, 13)}
    | {"Up", "Down", "Left", "Right", "Shift", "Control", "Alt", "Return", "Enter", "Tab", "Backspace",
       "Insert", "Delete", "Home", "End", "PageUp", "PageDown", "Comma", "Period", "Slash", "Semicolon",
       "Minus", "Equal", "BracketLeft", "BracketRight", "Backslash", "Apostrophe", "QuoteLeft"}
) - CONTROL_KEYS

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
    if isinstance(gen, m.HillGen):
        return [f"{gen.id}.x_min", f"{gen.id}.x_max"]
    if isinstance(gen, m.ObstacleGen):
        return [f"{gen.id}.speed", f"{gen.id}.accel", f"{gen.id}.length"]
    return []


def generator_bodies(gen: m.Generator) -> dict[str, list[str]]:
    """생성기가 만드는 물체 id와 그 모양 태그."""
    if isinstance(gen, m.HeightmapGen):
        tags = ["terrain", "pad"] if gen.pad is not None else ["terrain"]
        return {gen.id: tags}
    if isinstance(gen, m.TrackGen):
        return {f"{gen.id}.walls": ["wall"]}
    if isinstance(gen, m.HillGen):
        return {gen.id: ["terrain"]}
    if isinstance(gen, m.ObstacleGen):
        return {gen.id: ["obstacle", "air"]}
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


CHECKPOINT_FIELDS = ("count", "laps", "since", "progress")


def checkpoint_ids(env: m.EnvFile) -> list[str]:
    out = [c.id for c in env.checkpoints]
    for g in env.generators:
        out += generator_checkpoints(g)
    return out


def body_tags(env: m.EnvFile) -> dict[str, list[str]]:
    """물체 id -> 모양 태그. 접촉은 모양 사이에서만 생기므로 contact()와 접촉 센서는 이 태그만 쓴다."""
    out = {b.id: list(b.tags) for b in env.bodies}
    for g in env.generators:
        out.update(generator_bodies(g))
    return out


def all_tags(env: m.EnvFile) -> list[str]:
    """물체에 붙은 태그 전부(영역 태그는 빼고)."""
    tags: set[str] = set()
    for t in body_tags(env).values():
        tags.update(t)
    return sorted(tags)


def owner_of(env: m.EnvFile, body_id: str) -> str:
    """'agent' 또는 'world'. 생성기 물체는 world."""
    for b in env.bodies:
        if b.id == body_id:
            return b.owner
    return "world"


def is_dynamic(env: m.EnvFile, body_id: str) -> bool:
    return any(b.id == body_id and b.type == "dynamic" for b in env.bodies)


def joint_stage(env: m.EnvFile, joint: m.Joint) -> BuildStage:
    """양쪽이 모두 세계 물체(생성기 물체 포함)인 관절은 세계 단계(p.*를 못 씀). 그래야 모든 개체가 같은 세계를 쓴다."""
    if owner_of(env, joint.a) == "world" and owner_of(env, joint.b) == "world":
        return "world"
    return "agent"


def agent_only_selector(env: m.EnvFile, selector: str) -> bool:
    """접촉 대상이 에이전트 물체뿐인가. 에이전트 물체끼리는 서로 닿지 않으므로 그런 접촉 조건은 늘 거짓이다."""
    tags = body_tags(env)
    if selector.startswith("tag."):
        owners = {owner_of(env, b) for b, t in tags.items() if selector[4:] in t}
        return owners == {"agent"}
    return owner_of(env, selector) == "agent"


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
            # '<id>.0'은 수식 문법에서 쓸 수 없으므로 '<id>.r0'처럼 이름 붙인다.
            out += [Channel(f"{s.id}.r{i}", s.id, 0.0, 1.0) for i in range(s.count)]
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
        elif isinstance(a, m.Jump):
            out.append(Channel(a.id, a.id, 0.0, 1.0))
    return out


def key_allowed(actuator: m.Actuator, key: str) -> bool:
    """이 구동기의 직접 조종 키로 쓸 수 있는가. Space는 점프 추진기에만, R·N은 어디에도 못 쓴다. 모르는 키 이름은 False."""
    if key in JUMP_ONLY_KEYS:
        return isinstance(actuator, m.Jump)
    return key in KEY_NAMES


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
    for c in checkpoint_ids(env):
        for f in CHECKPOINT_FIELDS:
            out[f"{c}.{f}"] = NUM_STEP
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
        self._env = env
        self._bodies = set(body_ids(env))
        self._dynamic = {b.id for b in env.bodies if b.type == "dynamic"}
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
            ok = b in self._regions or b in self._dynamic
            want = S.SPEC_QUERY_WANT_REGION_OR_BODY
        if not ok:
            raise ExprError(S.SPEC_QUERY_BAD_SECOND.format(func=func, name=b, want=want))
        if func == "contact" and owner_of(self._env, a) == "agent" and agent_only_selector(self._env, b):
            raise ExprError(S.SPEC_AGENT_SELF_CONTACT.format(a=a, b=b))
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

"""환경 파일(.evoenv) 모델.

에디터가 쓰고 런타임이 읽는 JSON 구조를 pydantic으로 정의한다. 단위는 SI(미터, 초, 킬로그램, 라디안).

숫자 칸 대부분은 숫자이거나 '='로 시작하는 수식이다(예: "=p.leg_len / 2"). 수식은 만들기 단계에서
한 번 계산하며 쓸 수 있는 이름은 칸마다 다르다(spec/symbols.py). 사건 조건(when), 점수 수식처럼 늘 수식인
칸은 '=' 없이 쓴다.

참조는 부품 id 또는 'tag.<이름>'(그 태그가 붙은 모양 전부)으로 한다. 생성기 결과는 '<생성기 id>.<출력>'
(예: terrain.pad) 이름으로 참조한다.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

CURRENT_VERSION = 1

Id = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_\uac00-\ud7a3][A-Za-z0-9_\uac00-\ud7a3]*$", max_length=40)]
Ref = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_\uac00-\ud7a3][A-Za-z0-9_.\uac00-\ud7a3]*$", max_length=80)]
Formula = Annotated[str, StringConstraints(pattern=r"^=", max_length=1000)]
Expr = Annotated[str, StringConstraints(min_length=1, max_length=1000)]
Num = Union[float, Formula]
Vec2 = tuple[Num, Num]
Label = Annotated[str, StringConstraints(max_length=80)]


class _Model(BaseModel):
    # NaN과 무한대는 받지 않는다(Chipmunk가 음수·NaN 마찰 같은 값에서 프로세스를 강제 종료한다).
    model_config = ConfigDict(extra="forbid", populate_by_name=True, allow_inf_nan=False)


# ---------------------------------------------------------------------------
# 세계, 매개변수, 랜덤 값
# ---------------------------------------------------------------------------


class World(_Model):
    kind: Literal["physics2d", "grid"] = "physics2d"
    gravity: Vec2 = (0.0, -9.81)
    wind: Vec2 = (0.0, 0.0)
    """공기 속도(m/s). 물체의 drag와 함께 바람 힘이 된다."""
    dt: float = Field(default=1.0 / 60.0, gt=0.0, le=0.1)
    max_time: float = Field(default=20.0, gt=0.0, le=3600.0)
    bounds: tuple[Num, Num, Num, Num] = (-50.0, -10.0, 50.0, 50.0)
    """(x_min, y_min, x_max, y_max). 에이전트 물체 중심이 밖으로 나가면 out_of_bounds 사건."""
    contact_debounce: float = Field(default=0.1, ge=0.0, le=1.0)
    """접촉 센서와 contact() 조건은 이 시간 안에 닿았으면 닿은 것으로 본다(튀는 다리의 깜빡임 방지)."""


class Param(_Model):
    """이름 붙은 매개변수. 몸 유전자는 evolve=True인 매개변수만 가리킨다(선언 순서)."""

    id: Id
    label: Label = ""
    value: float
    evolve: bool = False
    min: float | None = None
    max: float | None = None


class RandVar(_Model):
    """시험마다 시험 시드로 뽑는 값. 다른 칸에서 r.<id>로 쓴다."""

    id: Id
    label: Label = ""
    dist: Literal["uniform", "int", "normal", "choice"] = "uniform"
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    std: float | None = None
    values: list[float] | None = None


# ---------------------------------------------------------------------------
# 모양, 물체, 관절
# ---------------------------------------------------------------------------


class Box(_Model):
    kind: Literal["box"]
    w: Num
    h: Num
    offset: Vec2 = (0.0, 0.0)
    angle: Num = 0.0


class Circle(_Model):
    kind: Literal["circle"]
    r: Num
    offset: Vec2 = (0.0, 0.0)


class Poly(_Model):
    kind: Literal["poly"]
    points: list[Vec2] = Field(min_length=3, max_length=64)
    """오목해도 된다. 만들 때 볼록 조각으로 나눈다."""


class Radial(_Model):
    kind: Literal["radial"]
    radii: list[Num] = Field(min_length=3, max_length=32)
    """중심에서 같은 각도 간격으로 뻗은 반지름들. 늘 올바른 모양이라 진화하는 차체에 쓴다."""
    angle: Num = 0.0


class Chain(_Model):
    kind: Literal["chain"]
    points: list[Vec2] = Field(min_length=2, max_length=4096)
    closed: bool = False
    radius: float = Field(default=0.05, ge=0.0)


Shape = Annotated[Union[Box, Circle, Poly, Radial, Chain], Field(discriminator="kind")]


class Material(_Model):
    density: Num = 1.0
    friction: Num = 0.7
    elasticity: Num = 0.0


class Body(_Model):
    id: Id
    label: Label = ""
    owner: Literal["agent", "world"] = "agent"
    type: Literal["dynamic", "static", "kinematic"] = "dynamic"
    pos: Vec2 = (0.0, 0.0)
    """에이전트 물체는 에이전트 좌표계(시작 위치 기준), 세계 물체는 세계 좌표."""
    angle: Num = 0.0
    place_by: Ref | None = None
    """관절 id. 주면 그 관절의 부모 앵커와 rest_angle로 자세를 정한다(pos, angle 무시)."""
    shapes: list[Shape] = Field(min_length=1, max_length=64)
    material: Material = Material()
    tags: list[Id] = []
    drag: Num = 0.0
    """바람에 대한 선형 항력 계수 N/(m/s)."""


class Joint(_Model):
    id: Id
    label: Label = ""
    kind: Literal["pivot", "weld", "rot_spring", "spring"]
    """pivot: 회전축(각도 제한 선택). weld: 고정. rot_spring: 회전축 + 회전 스프링(+ 각도 제한).
    spring: 두 앵커 사이 거리 스프링."""
    a: Ref
    b: Ref
    anchor_a: Vec2 = (0.0, 0.0)
    anchor_b: Vec2 = (0.0, 0.0)
    rest_angle: Num = 0.0
    """b의 각도 - a의 각도. rot_spring의 쉬는 각도이자 place_by 배치 각도."""
    stiffness: Num = 0.0
    damping: Num = 0.0
    rest_length: Num | None = None
    limit: tuple[Num, Num] | None = None
    """상대 각도(b - a) 범위."""
    collide: bool = False


# ---------------------------------------------------------------------------
# 구동기 (신경망 출력)
# ---------------------------------------------------------------------------


class FuelTank(_Model):
    kind: Literal["fuel_tank"]
    id: Id
    label: Label = ""
    capacity: Num = 100.0


class Thruster(_Model):
    """출력 하나, 범위 0..1."""

    kind: Literal["thruster"]
    id: Id
    label: Label = ""
    body: Ref
    at: Vec2 = (0.0, 0.0)
    dir: Num = 1.5707963267948966
    """물체 좌표계에서 힘이 물체를 미는 방향(라디안). 기본은 위쪽."""
    max_force: Num
    tank: Ref | None = None
    fuel_per_s: Num = 0.0
    """최대 출력에서 초당 연료 소모량."""
    keys: list[str] = []
    """직접 조종 키 이름(Qt 키 이름, 예: Up, W). 누르는 동안 출력 1."""


class Vehicle(_Model):
    """출력 두 개(가속, 조향), 범위 -1..1. M4에서 런타임 구현."""

    kind: Literal["vehicle"]
    id: Id
    label: Label = ""
    body: Ref
    max_accel: Num
    max_steer: Num
    lateral_grip: Num = 1.0
    keys_forward: list[str] = []
    keys_back: list[str] = []
    keys_left: list[str] = []
    keys_right: list[str] = []


class Motor(_Model):
    """출력 하나, 범위 -1..1. M5b에서 런타임 구현."""

    kind: Literal["motor"]
    id: Id
    label: Label = ""
    joint: Ref
    max_torque: Num
    max_speed: Num
    keys_pos: list[str] = []
    keys_neg: list[str] = []


Actuator = Annotated[Union[FuelTank, Thruster, Vehicle, Motor], Field(discriminator="kind")]


# ---------------------------------------------------------------------------
# 센서 (신경망 입력). 관측값 = clip(값 / scale, -5, 5)
# ---------------------------------------------------------------------------

StateField = Literal["x", "y", "vx", "vy", "speed", "angle", "omega", "v_fwd", "v_lat"]


class BodyState(_Model):
    kind: Literal["body_state"]
    id: Id
    label: Label = ""
    body: Ref
    fields: list[StateField] = Field(min_length=1)
    frame: Literal["world", "body"] = "world"
    """속도를 세계 좌표로 볼지 물체 좌표로 볼지. 각도는 늘 -pi..pi로 감싼다."""
    scale: dict[StateField, Annotated[float, Field(gt=0.0)]] = {}


class Target(_Model):
    kind: Literal["target"]
    id: Id
    label: Label = ""
    body: Ref
    target: Ref
    frame: Literal["world", "body"] = "world"
    scale: float = Field(default=20.0, gt=0.0)


class ContactSensor(_Model):
    kind: Literal["contact"]
    id: Id
    label: Label = ""
    body: Ref
    with_: list[Ref] = Field(alias="with", min_length=1)


class JointAngle(_Model):
    kind: Literal["joint_angle"]
    id: Id
    label: Label = ""
    joint: Ref
    scale: float = Field(default=3.141592653589793, gt=0.0)


class FuelLeft(_Model):
    kind: Literal["fuel"]
    id: Id
    label: Label = ""
    tank: Ref


class TimeLeft(_Model):
    kind: Literal["time_left"]
    id: Id
    label: Label = ""


class Rays(_Model):
    """부채꼴 레이. 채널 <id>.r0 .. <id>.r(count-1), 값 = 맞은 거리 / length. M4에서 런타임 구현."""

    kind: Literal["rays"]
    id: Id
    label: Label = ""
    body: Ref
    origin: Vec2 = (0.0, 0.0)
    count: int = Field(ge=1, le=64)
    spread: Num
    center: Num = 0.0
    length: Num
    hits: list[Ref] = Field(min_length=1)


Sensor = Annotated[
    Union[BodyState, Target, ContactSensor, JointAngle, FuelLeft, TimeLeft, Rays], Field(discriminator="kind")
]


# ---------------------------------------------------------------------------
# 영역, 체크포인트, 생성기
# ---------------------------------------------------------------------------


class Region(_Model):
    id: Id
    label: Label = ""
    kind: Literal["goal", "hazard", "pad", "zone"] = "zone"
    shape: Annotated[Union[Box, Circle, Poly], Field(discriminator="kind")]
    pos: Vec2 = (0.0, 0.0)
    tags: list[Id] = []


class Checkpoints(_Model):
    """순서 있는 체크포인트 선. M4에서 런타임 구현."""

    id: Id
    label: Label = ""
    lines: list[tuple[Vec2, Vec2]] = Field(min_length=1)
    loop: bool = True


class PadSpec(_Model):
    x: Num
    width: Num
    region_height: float = Field(default=3.0, gt=0.0)
    """패드 위로 이 높이까지를 영역 <생성기>.pad로 내놓는다(거리 점수와 inside 판정용)."""


class HeightmapGen(_Model):
    """출력: 지형 물체 <id> (모양 태그 terrain, 패드 구간은 terrain과 pad), 영역 <id>.pad,
    값 gen.<id>.pad.x / .y / .w / .left / .right."""

    kind: Literal["heightmap"]
    id: Id
    label: Label = ""
    seed: Num = 0.0
    x_range: tuple[Num, Num] = (-50.0, 50.0)
    segment: float = Field(default=1.0, gt=0.0)
    base_y: Num = 0.0
    amplitude: Num = 4.0
    roughness: Num = 0.5
    pad: PadSpec | None = None
    friction: Num = 0.9
    radius: float = Field(default=0.1, ge=0.0)


class TrackGen(_Model):
    """M4에서 런타임 구현."""

    kind: Literal["track"]
    id: Id
    label: Label = ""
    source: Literal["generated", "drawn"] = "generated"
    set_seed: int = 0
    set_size: int = Field(default=5, ge=1, le=50)
    centerline: list[Vec2] | None = None
    width: Num = 8.0
    segments: int = 12
    curvature: Num = 0.5
    length_range: tuple[float, float] = (150.0, 250.0)
    checkpoints: int = 24


Generator = Annotated[Union[HeightmapGen, TrackGen], Field(discriminator="kind")]


# ---------------------------------------------------------------------------
# 에이전트, 사건, 종료
# ---------------------------------------------------------------------------


class Spawn(_Model):
    x: Num = 0.0
    y: Num = 0.0
    angle: Num = 0.0
    vx: Num = 0.0
    vy: Num = 0.0
    omega: Num = 0.0


class Agent(_Model):
    spawn: Spawn = Spawn()


class Event(_Model):
    """이름 붙은 조건. when이 hold초 동안 이어서 참이면 일어난다.

    다른 수식에서 <id>(한 번이라도 일어났나), <id>.count(일어난 횟수), <id>.since(마지막으로 일어난 뒤
    지난 초, 한 번도 없으면 시작부터 지난 초)로 쓴다. 내장 사건: timeout, out_of_bounds.
    """

    id: Id
    label: Label = ""
    when: Expr
    hold: float = Field(default=0.0, ge=0.0)


class End(_Model):
    event: Ref
    outcome: Literal["success", "failure", "neutral"]
    label: Label = ""


# ---------------------------------------------------------------------------
# 점수 블록
# ---------------------------------------------------------------------------


class _Block(_Model):
    id: Id
    label: Label = ""
    weight: float = 1.0


class EventBlock(_Block):
    kind: Literal["event"]
    event: Ref
    value: Literal["occurred", "count"] = "occurred"


class DistanceBlock(_Block):
    kind: Literal["distance_to"]
    body: Ref
    target: Ref
    agg: Literal["final", "min"] = "final"


class ProgressBlock(_Block):
    kind: Literal["progress"]
    body: Ref
    direction: Num = 0.0
    """시작 위치에서 이 방향(라디안)으로 간 거리. 트랙 중심선 진행은 M4."""


class SurvivalBlock(_Block):
    kind: Literal["survival_time"]


class SpeedBlock(_Block):
    kind: Literal["final_speed"]
    body: Ref


class TiltBlock(_Block):
    kind: Literal["tilt"]
    body: Ref
    agg: Literal["final", "mean"] = "final"


class EnergyBlock(_Block):
    kind: Literal["energy"]
    actuators: list[Ref] | Literal["all"] = "all"


class FuelBlock(_Block):
    kind: Literal["fuel_used"]
    tank: Ref


class ContactTimeBlock(_Block):
    kind: Literal["contact_time"]
    body: Ref
    with_: list[Ref] = Field(alias="with", min_length=1)


class SmoothnessBlock(_Block):
    kind: Literal["smoothness"]
    actuators: list[Ref] | Literal["all"] = "all"


class CheckpointBlock(_Block):
    """M4에서 런타임 구현."""

    kind: Literal["checkpoints", "lap_time"]
    checkpoints: Ref


class ExprBlock(_Block):
    kind: Literal["expr"]
    expr: Expr
    agg: Literal["final", "mean", "min", "max", "integral"] = "integral"


Block = Annotated[
    Union[
        EventBlock,
        DistanceBlock,
        ProgressBlock,
        SurvivalBlock,
        SpeedBlock,
        TiltBlock,
        EnergyBlock,
        FuelBlock,
        ContactTimeBlock,
        SmoothnessBlock,
        CheckpointBlock,
        ExprBlock,
    ],
    Field(discriminator="kind"),
]


class Scoring(_Model):
    blocks: list[Block] = []
    formula: Expr | None = None
    """있으면 가중치 합 대신 쓴다. 블록 id(가중치 적용 전 값), time, success, failure, 사건을 쓸 수 있다."""


# ---------------------------------------------------------------------------
# 뇌, 평가, 파일
# ---------------------------------------------------------------------------


class Brain(_Model):
    inputs: list[str] | Literal["all"] = "all"
    hidden: list[Annotated[int, Field(ge=1, le=1024)]] = [16]
    activation: Literal["tanh", "relu"] = "tanh"
    """은닉층에만 적용한다. 출력층은 늘 tanh."""
    outputs: list[str] | Literal["all"] = "all"


class Evaluation(_Model):
    trials: int = Field(default=3, ge=1, le=100)
    aggregate: Literal["mean", "min", "q25"] = "mean"
    target_score: float | None = None


class EnvFile(_Model):
    version: int = CURRENT_VERSION
    name: Label
    description: str = ""
    world: World = World()
    params: list[Param] = []
    randomize: list[RandVar] = []
    generators: list[Generator] = []
    agent: Agent = Agent()
    bodies: list[Body] = []
    joints: list[Joint] = []
    actuators: list[Actuator] = []
    sensors: list[Sensor] = []
    regions: list[Region] = []
    checkpoints: list[Checkpoints] = []
    events: list[Event] = []
    termination: list[End] = []
    scoring: Scoring = Scoring()
    brain: Brain = Brain()
    evaluation: Evaluation = Evaluation()
    editor: dict[str, Any] = {}
    """에디터 전용(보기, 색, 미리보기 시드). 런타임과 유전자 서명에서 뺀다."""

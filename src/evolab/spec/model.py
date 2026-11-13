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

CURRENT_VERSION = 4
"""환경 파일 형식 번호. 1: v1.x. 2: v2.0(정해진 움직임 물체, 장애물 줄, 점프 추진기, 공기역학, 날개 모양, 모양별 밀도).
3: v2.1(몸 구조 진화: 물체의 있음 칸 exists). 4: 6.2.0(세계 종류 view, 바닥 마찰 damping, 밀기 구동기 Move, 균형 보조 Balance).

칸·종류·값마다 생긴 형식을 spec/io.py의 FIELD_SINCE·KIND_SINCE·VALUE_SINCE에 적고, 저장과 해시는 그 내용을 담을 수
있는 가장 낮은 형식으로 한다(v1 기능만 쓴 파일은 v1.x가 쓴 글과 글자까지 같다). 새 칸은 반드시 다음 형식 번호로
등록한다(tests/test_format_v2.py가 지킨다).
"""

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
    view: Literal["side", "top"] = "side"
    """세계 종류(형식 4). side: 옆에서 본 세계(중력이 있음). top: 위에서 본 세계(중력 없이 바닥 위를 미끄러짐).
    물리는 바꾸지 않는다(그리기·에디터·검사만): top이면 중력이 (0, 0)이어야 한다."""
    damping: float = Field(default=1.0, gt=0.0, le=1.0)
    """바닥 마찰(감쇠, 형식 4): 1초 뒤에 남는 속도의 비율(0 < d ≤ 1, 선속도·각속도 모두). 1이면 줄지 않는다(옛 뜻)."""


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
    density: Num | None = None
    """이 모양만 따로 쓰는 밀도. 비우면 물체 재질의 밀도. 작고 무거운 추로 무게 중심을 옮길 때 쓴다(형식 2)."""


class Circle(_Model):
    kind: Literal["circle"]
    r: Num
    offset: Vec2 = (0.0, 0.0)
    density: Num | None = None
    """이 모양만 따로 쓰는 밀도. 비우면 물체 재질의 밀도. 작고 무거운 추로 무게 중심을 옮길 때 쓴다(형식 2)."""


class Poly(_Model):
    kind: Literal["poly"]
    points: list[Vec2] = Field(min_length=3, max_length=64)
    """오목해도 된다. 만들 때 볼록 조각으로 나눈다."""
    density: Num | None = None
    """이 모양만 따로 쓰는 밀도. 비우면 물체 재질의 밀도. 작고 무거운 추로 무게 중심을 옮길 때 쓴다(형식 2)."""


class Radial(_Model):
    kind: Literal["radial"]
    radii: list[Num] = Field(min_length=3, max_length=32)
    """중심에서 같은 각도 간격으로 뻗은 반지름들. 늘 올바른 모양이라 진화하는 차체에 쓴다."""
    angle: Num = 0.0
    density: Num | None = None
    """이 모양만 따로 쓰는 밀도. 비우면 물체 재질의 밀도. 작고 무거운 추로 무게 중심을 옮길 때 쓴다(형식 2)."""


class Chain(_Model):
    kind: Literal["chain"]
    points: list[Vec2] = Field(min_length=2, max_length=4096)
    closed: bool = False
    radius: float = Field(default=0.05, ge=0.0)


class Wing(_Model):
    """날개 모양(형식 2): 꼭짓점 넷(앞전, 위, 뒷전, 아래, 반시계)인 다각형.

    뒷전 = le - chord x (cos angle, sin angle). 시위 가운데에서 법선 (-sin angle, cos angle) 쪽으로 위 꼭짓점은
    chord x (camber + thick / 2), 아래 꼭짓점은 chord x (camber - thick / 2)만큼 떨어진다. thick > 0이면 어떤 값에서도
    꼬이지 않는 단순 다각형이라 몸 수치로 진화하는 날개에 쓴다.
    """

    kind: Literal["wing"]
    le: Vec2 = (0.0, 0.0)
    """앞전(공기를 먼저 만나는 앞쪽 끝), 물체 좌표."""
    chord: Num = 1.0
    """시위 길이: 앞전에서 뒷전까지(m)."""
    angle: Num = 0.0
    """붙임각(라디안). 양수면 앞전이 들린다."""
    thick: Num = 0.04
    """두께(시위 길이에 대한 비율)."""
    camber: Num = 0.0
    """휨(시위 길이에 대한 비율). 양수면 위로 휜다."""
    density: Num | None = None
    """이 모양만 따로 쓰는 밀도. 비우면 물체 재질의 밀도."""


Shape = Annotated[Union[Box, Circle, Poly, Radial, Chain, Wing], Field(discriminator="kind")]


class PathMotion(_Model):
    """정해진 길을 따라 움직임(형식 2). 시간만의 함수라 같은 시험의 모든 개체가 같은 움직임을 본다.

    출발(start초) 뒤 t초 동안 길을 따라 간 거리 = speed x t + accel x t² / 2(속력은 0 아래로 내려가지 않는다).
    once: 끝점에서 멈춤, loop: 끝점에서 첫 점으로 이어 돎(닫힌 길), pingpong: 끝점에서 되돌아옴.
    """

    kind: Literal["path"]
    points: list[Vec2] = Field(min_length=2, max_length=256)
    """지날 점. 물체 pos를 기준으로 한 위치이고 첫 점은 보통 (0, 0)이다."""
    speed: Num = 1.0
    """처음 속력(m/s)."""
    accel: Num = 0.0
    """속력이 초마다 늘어나는 양(m/s²). 음수면 느려지다 멈춘다."""
    start: Num = 0.0
    """이 시각(초)까지 첫 점에 서 있다."""
    mode: Literal["once", "loop", "pingpong"] = "once"
    spin: Num = 0.0
    """제자리 회전 속도(rad/s, +는 반시계). 출발 시각(start)과 상관없이 0초부터 돈다."""


class WaveMotion(_Model):
    """제자리에서 흔들림(형식 2). 위치 = pos + amplitude x sin(2π(t / period + phase))."""

    kind: Literal["wave"]
    amplitude: Vec2 = (0.0, 1.0)
    """흔들림 폭 (x, y)(m)."""
    period: Num = 2.0
    """한 번 오갔다 오는 시간(초)."""
    phase: Num = 0.0
    """시작 위상(주기에 대한 비율, 0..1)."""
    spin: Num = 0.0
    """제자리 회전 속도(rad/s, +는 반시계)."""


Motion = Annotated[Union[PathMotion, WaveMotion], Field(discriminator="kind")]


class Material(_Model):
    density: Num = 1.0
    friction: Num = 0.7
    elasticity: Num = 0.0


class Body(_Model):
    id: Id
    label: Label = ""
    owner: Literal["agent", "world"] = "agent"
    type: Literal["dynamic", "static", "kinematic"] = "dynamic"
    """dynamic: 힘을 받아 움직임. static: 고정. kinematic: motion을 따라 스스로 움직임(세계 물체만, 형식 2)."""
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
    motion: Motion | None = None
    """정해진 움직임(type이 kinematic인 세계 물체만, 형식 2). 수식은 r.*과 gen.*만 쓸 수 있다."""
    exists: Num | None = None
    """있음(에이전트 물체만, 형식 3). 값이 0.5보다 크면 이 물체를 만들고, 비우면 늘 있다. 수식은 p.*(몸 수치)만 쓸 수 있어
    몸 구조는 몸 유전자만의 함수다. 뿌리 물체(선언 순서 첫 에이전트 물체)에는 쓸 수 없다. 없는 물체와, 그 때문에 관절로
    뿌리와 이어지지 않게 된 물체는 만들지 않고, 거기 붙은 관절·모터·구동기·센서도 없다(spec/structure.py)."""


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
    """위에서 본 차량 구동기. 출력 두 개: <id>.throttle(앞뒤 가속), <id>.steer(조향, +는 왼쪽), 범위 -1..1.

    물체 좌표 +x가 앞이다. 스텝마다 앞쪽 힘 = throttle x max_accel x 질량, 구름 저항 = -drag x 앞뒤 속도 x 질량
    (최고 속도 = max_accel / drag), 옆미끄럼 속도는 lateral_grip(0..1) 비율만큼 없앤다(1이면 옆으로 미끄러지지 않음),
    회전 속도 = 앞뒤 속도 x tan(steer x max_steer) / wheelbase(자전거 모형). max_lateral_accel(m/s², 0이면 제한 없음)을
    넘는 횡가속도가 필요하면 그만큼만 돈다(빠르면 덜 꺾인다: 코너 앞에서 속도를 줄여야 한다).
    """

    kind: Literal["vehicle"]
    id: Id
    label: Label = ""
    body: Ref
    max_accel: Num
    max_steer: Num
    lateral_grip: Num = 1.0
    wheelbase: Num = 2.5
    drag: Num = 0.3
    max_lateral_accel: Num = 0.0
    keys_forward: list[str] = []
    keys_back: list[str] = []
    keys_left: list[str] = []
    keys_right: list[str] = []


class Motor(_Model):
    """회전 관절(pivot, rot_spring)을 돌리는 모터. 출력 하나, 범위 -1..1.

    출력 u면 b 물체가 a 물체에 대해 u x max_speed(rad/s, +는 반시계)로 돌도록 최대 max_torque(N·m)까지 쓴다.
    """

    kind: Literal["motor"]
    id: Id
    label: Label = ""
    joint: Ref
    max_torque: Num
    max_speed: Num
    keys_pos: list[str] = []
    keys_neg: list[str] = []


class Jump(_Model):
    """점프 추진기(형식 2). 출력 하나, 범위 0..1.

    출력이 0.5 이상이고, ground에 닿아 있고(세계 contact_debounce 적용, 앞 스텝 기준), 마지막으로 뛴 스텝에서
    cooldown이 지났으면(스텝 수로 올림) 뛰는 방향(dir)의 속도를 speed로 맞춘다(이미 더 빠르면 그대로). 질량과 무관하게
    점프 높이는 대략 speed² / (2 x 중력)이다. 실제 출력(에너지 블록, 행동 분포)은 뛴 스텝만 1이고 나머지는 0이다.
    """

    kind: Literal["jump"]
    id: Id
    label: Label = ""
    body: Ref
    speed: Num = 6.0
    """뛰는 방향 속도(m/s)."""
    dir: Num = 1.5707963267948966
    """뛰는 방향(라디안, 세계 좌표). 기본은 위쪽. 물체가 기울어도 바뀌지 않는다."""
    ground: list[Ref] = ["tag.ground"]
    """여기(물체 id나 tag.<태그>)에 닿아 있을 때만 뛴다. 비우면 공중에서도 뛴다."""
    cooldown: float = Field(default=0.25, ge=0.0, le=60.0)
    """한 번 뛴 뒤 다시 뛸 수 있을 때까지의 시간(초)."""
    keys: list[str] = []
    """직접 조종 키 이름. 누르는 동안 출력 1. 점프에는 Space도 쓸 수 있다."""


class Aero(_Model):
    """공기역학(형식 2, 출력 없음). 물체 모양의 테두리 변마다 작은 평판으로 보고 양력과 항력을 건다.

    변마다 그 점에서 본 상대 바람으로 받음각을 재고, 실속 전에는 양력 기울기 2π / (1 + 2 / aspect_ratio)인 붙은 흐름,
    stall 근처부터 평판 수직 힘으로 부드럽게 잇는다. 유도 항력과 표면 마찰을 더한다. 바람은 세계의 wind를 쓴다.
    원과 선 모양은 공기를 받지 않는다.
    """

    kind: Literal["aero"]
    id: Id
    label: Label = ""
    body: Ref
    density: Num = 1.2
    """공기 밀도(kg/m³)."""
    span: Num = 1.0
    """화면 안쪽 방향 날개 폭(m). 공기힘이 이 폭에 비례한다."""
    aspect_ratio: Num = 6.0
    """가로세로비(날개 폭 / 시위). 0이면 끝없이 긴 날개(2D)."""
    stall: Num = 0.25
    """실속 받음각(라디안)."""
    friction: Num = 0.01
    """표면 마찰 계수."""
    surfaces: Literal["all", "wings"] = "all"
    """all: 사각형·다각형·방사형·날개 모양 테두리가 모두 공기를 받음. wings: 날개 모양만."""


class Move(_Model):
    """밀기(이동) 구동기(형식 4). 물체를 바닥 위에서 민다(원형 선수, 미끄러지는 받침). 출력 범위 -1..1.

    mode가 xy면 출력 둘 <id>.x, <id>.y(두 방향으로 밂), x면 <id>.x 하나(옆에서 본 세계의 좌우), heading이면
    <id>.forward(몸 방향으로 밂), <id>.turn(각속도 = 출력 x max_turn). 힘 = 출력 x max_force − (max_force / max_speed) x 속도라서
    최고 속도가 max_speed다(한 스텝을 지수로 적분).
    """

    kind: Literal["move"]
    id: Id
    label: Label = ""
    body: Ref
    mode: Literal["xy", "x", "heading"] = "xy"
    max_force: Num
    """출력이 1일 때의 힘(N)."""
    max_speed: Num
    """이 속도에서 더 빨라지지 않는다(m/s). 0보다 커야 한다."""
    max_turn: Num = 6.0
    """heading에서 출력이 1일 때 도는 속도(rad/s)."""
    keys_up: list[str] = []
    keys_down: list[str] = []
    keys_left: list[str] = []
    keys_right: list[str] = []


class Balance(_Model):
    """균형 보조(형식 4, 출력 채널 없음). 물체에 바깥 토크를 건다(반작용 없음):
    토크 = clamp(−stiffness x (각도 − target) − damping x 각속도, ±max_torque). 진화가 걷기를 더 빨리 찾게 돕는다."""

    kind: Literal["balance"]
    id: Id
    label: Label = ""
    body: Ref
    stiffness: Num
    """세우는 힘(N·m/rad)."""
    damping: Num = 0.0
    """도는 속도를 줄이는 정도(N·m·s/rad)."""
    max_torque: Num
    """보조가 낼 수 있는 가장 큰 토크(N·m)."""
    target: Num = 0.0
    """보조가 맞추려는 몸 각도(라디안). 0은 똑바로 선 자세."""


Actuator = Annotated[
    Union[FuelTank, Thruster, Vehicle, Motor, Jump, Aero, Move, Balance], Field(discriminator="kind")
]


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
    """부채꼴 레이. 채널 <id>.r0 .. <id>.r(count-1), 값 = 맞은 거리 / length(안 맞으면 1).

    레이 i의 방향(물체 좌표) = center - spread/2 + spread x i/(count-1) (count가 1이면 center). hits에 맞는
    모양(물체 id 또는 tag.<태그>)만 맞는다. 자기 몸은 맞지 않는다.
    """

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
    """순서 있는 체크포인트 선. 0번은 출발선이다(출발할 때 지난 것으로 본다).

    선마다 진행 방향으로 볼 때 왼쪽 끝 → 오른쪽 끝 순서로 적는다. 그 방향으로(뒤에서 앞으로) 지날 때만 센다.
    loop면 선이 3개 이상이어야 한다.

    수식에서 <id>.count(지난 체크포인트 수), <id>.laps(바퀴 수, loop일 때), <id>.since(마지막으로 지난 뒤 초),
    <id>.progress(진행: 트랙 생성기의 체크포인트면 중심선을 따라 간 거리 m, 아니면 지난 체크포인트 수)를 쓴다.
    """

    id: Id
    label: Label = ""
    lines: list[tuple[Vec2, Vec2]] = Field(min_length=1)
    loop: bool = True
    body: Ref | None = None
    """선을 지나는지 볼 물체. 없으면 차량 구동기의 물체, 그것도 없으면 첫 에이전트 물체."""


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
    """닫힌 트랙. 물체 <id>.walls(안쪽·바깥쪽 벽, 태그 wall)와 체크포인트 <id>.cp를 내놓는다.

    gen.<id>.start.x / start.y / start.angle(출발선 가운데, 진행 방향), gen.<id>.length(중심선 길이)를 쓸 수 있다.
    source가 generated면: 시험에 트랙 번호가 있으면 학습 트랙 세트(set_seed로 만든 set_size개)의 그 번호 트랙,
    없으면(검증·비교·판정) 시험 시드로 새 트랙을 만든다. drawn이면 centerline(중심선 꼭짓점)으로 만든 트랙 하나.
    """

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
    checkpoints: int = Field(default=24, ge=3, le=2000)
    """한 바퀴의 체크포인트 선 수(3개 이상: 앞뒤로 흔들어 바퀴를 세지 못하게)."""
    friction: Num = 0.5
    wall_radius: float = Field(default=0.1, ge=0.0)


class HillGen(_Model):
    """언덕길: 오른쪽(+x)으로 갈수록 오르막과 굴곡이 커지는 지형(언덕 넘는 자동차). 물체 <id>(태그 terrain)를 내놓는다.

    높이 = base_y + slope x (x - 평지 끝) + 굴곡. 굴곡 크기는 amplitude에서 시작해 1m마다 growth씩 커진다.
    처음 flat_start m는 평지다. gen.<id>.x_min / x_max를 쓸 수 있다.
    """

    kind: Literal["hills"]
    id: Id
    label: Label = ""
    seed: Num = 0
    x_range: tuple[Num, Num] = (-20.0, 400.0)
    segment: float = Field(default=2.0, gt=0.0)
    base_y: Num = 0.0
    amplitude: Num = 0.5
    growth: Num = 0.01
    roughness: Num = 0.5
    slope: Num = 0.03
    flat_start: Num = 15.0
    friction: Num = 0.9
    radius: float = Field(default=0.1, ge=0.0)


class ObstacleGen(_Model):
    """장애물 줄(형식 2). 정해진 움직임 세계 물체 <id> 하나(모양 태그 obstacle)에 사각형 장애물 count개를 줄지어 놓고
    direction 쪽으로 speed에서 시작해 초마다 accel씩 빨라지며 민다(줄 전체가 같은 속도).

    간격 gap은 초다: 앞 장애물과 이 장애물이 x = 0(선수 자리)에 닿는 시각의 차이라 빨라져도 뛸 틈이 남는다. 첫 장애물은
    x에서 출발한다. 장애물마다 폭 width, 높이 height를 시드로 뽑고, air_chance 확률로 바닥 y + air_y에 두께 air_height인
    공중 장애물(태그 obstacle과 air)이 된다. gen.<id>.speed, gen.<id>.accel, gen.<id>.length를 쓸 수 있다.
    """

    kind: Literal["obstacles"]
    id: Id
    label: Label = ""
    seed: Num = 0.0
    """장애물 줄 시드. 보통 랜덤 값(=r.<id>)이라 시험마다 다른 줄이 된다."""
    count: int = Field(default=30, ge=1, le=500)
    """장애물 수."""
    x: Num = 25.0
    """첫 장애물이 출발하는 x(m)."""
    y: Num = 0.0
    """바닥 높이(m)."""
    gap: tuple[Num, Num] = (1.4, 3.0)
    """간격 범위(초)."""
    width: tuple[Num, Num] = (0.6, 1.4)
    """장애물 폭 범위(m)."""
    height: tuple[Num, Num] = (0.5, 1.2)
    """땅 장애물 높이 범위(m)."""
    air_chance: Num = 0.25
    """공중 장애물 비율(0..1)."""
    air_y: tuple[Num, Num] = (1.3, 1.6)
    """공중 장애물 아랫면 높이 범위(바닥 기준, m)."""
    air_height: tuple[Num, Num] = (0.5, 0.8)
    """공중 장애물 두께 범위(m)."""
    speed: Num = 6.0
    """처음 속도(m/s)."""
    accel: Num = 0.15
    """초마다 빨라지는 양(m/s²). 음수면 느려지다 멈춘다."""
    direction: Num = 3.141592653589793
    """미는 방향(라디안). 기본은 왼쪽(선수 쪽)."""
    friction: Num = 0.5
    """장애물 마찰."""


Generator = Annotated[Union[HeightmapGen, TrackGen, HillGen, ObstacleGen], Field(discriminator="kind")]


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
    """checkpoints: 지난 체크포인트 수. lap_time: 끝낸 바퀴의 평균 시간(초, 끝낸 바퀴가 없으면 0)."""

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

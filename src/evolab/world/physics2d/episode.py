"""환경 런타임: 시험 한 번(에피소드).

결과는 (환경 버전, 몸 수치, Trial)만의 함수다. 물리 세계는 시험마다 새로 만들고(재설정 재사용은 pymunk
관절의 누적 충격량과 접촉 캐시 때문에 결정성을 깬다), 난수는 시험 시드에서만 뽑는다.

학습 워커, 직접 조종, 재생이 모두 이 클래스를 쓴다.

v2.0 부품(설계 4.0~4.6)은 모두 "그 부품이 파일에 있을 때만" 지나가는 길이다: 정해진 움직임 물체(_kinematic), 점프
추진기(_jumps), 공기역학(_aero)이 비어 있으면 v1과 같은 계산을 같은 순서로 한다(모양별 밀도가 없으면 모든 조각에
물체 재질 밀도). 스텝 순서: 구동기(점프 → 추진기·모터·차량) → 힘(추진기·바람 항력) → 공기역학(한 스텝 상한) →
정해진 움직임 물체에 다음 자세로 가는 속도 → space.step.

v2.1 몸 구조(설계 4.11, D34·D35·D39): 물체의 있음 칸(exists)으로 없는 에이전트 물체(Episode.absent = 에이전트 물체 −
spec/structure.present)는 만들지 않는다. 그 물체에 끝이 닿은 관절·모터, 그 물체의 추진기·점프·차량·공기역학도 없고, 그
구동기 채널은 죽은 채널(실제 출력·명령 모두 0)이다. 그 물체·관절에 단 센서는 늘 0(레이도 0), 접촉 질의는 거짓. 있음 칸을
쓰는 물체가 없으면 없는 물체가 없어(빈 집합) v1·v2.0 파일은 예전과 같은 계산을 같은 순서로 한다.

v6.0 런타임 가속(v6 설계 4.1~4.6·4.8, D6~D14): 결과는 v3.1과 비트 단위로 같다(tests/oracle의 얼린 v3.1 사본과 같은 기계에서
바이트 비교: tests/test_speed_identity.py, tests/test_speed_runtime.py). 바꾼 것은 셋 중 하나다: 공개 API가 부르는 같은 C 함수를
같은 인수로 바로 부름(물체 상태 읽기, 힘·토크·속도·모터 쓰기 = _setup_c_paths, 둘째 스텝부터 cpSpaceStep = _first_space_step),
같은 식을 같은 순서로 계산하되 스텝마다 하던 문자열 비교·사전 찾기·객체 만들기를 만들 때 한 번으로(값 읽기·몸 상태 센서
특수화, 사건 한 루프 _fire_plan, 접촉 맞음 기억, 공기역학 _aero_fast, 레이 필터 한 번 쓰기, 꼬리표 frozenset 공유), 행동을
목록으로 받는 step_values. 초월 함수는 math만 쓴다(numpy는 끝 비트가 다르다). 규칙: Episode는 만든 뒤 공간에 물체·모양·관절을
더하거나 빼지 않고 접촉 콜백은 공간을 바꾸지 않는다(cpSpaceStep을 바로 부르는 전제). 그런 코드가 생기면(예: v5 대전의 물체
추가) 그 Episode는 _raw_space를 None으로 되돌려 공개 Space.step을 쓴다. _CP_FAST가 거짓이면(pymunk 내부가 바뀐 판, 만들 때
정함) 모든 곳이 공개 API 길을 지난다(같은 결과).
"""

from __future__ import annotations

import math
import operator
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
from evolab.spec import structure
from evolab.spec import symbols as sy
from evolab.spec.compiled import EnvSpec, finite
from evolab.world.physics2d import geometry as geo
from evolab.world.physics2d.motion import MotionPlan, pose_at

AGENT_GROUP = 1
OBS_CLIP = 5.0
COLLISION_SLOP = 0.01
NUM_LIMIT = 1e15
"""만들기 단계 수식 결과를 이 범위로 자른다(무한대가 물리로 들어가지 않게)."""
FORCE_LIMIT = 1e8
FIELD_LIMIT = 1e4
"""중력, 바람의 크기 한계."""
MAX_STEPS = 1e9
"""스텝 수 상한(hold가 아주 크거나 dt가 아주 작아도 정수로 바꿀 수 있게)."""
SHUFFLE_ABOVE = 1000
"""정적 물체의 모양이 이보다 많으면 섞어서 넣는다(기본 예제는 이보다 적어 결과가 바뀌지 않는다)."""
SHUFFLE_SEED = 0x5EED

# 공기역학 모델 상수(설계 4.5). 형식에 넣지 않은 런타임 상수라 바꾸면 결과가 바뀐다: 바꿀 때는 형식 번호를 올리고
# 옛 형식 파일은 옛 상수로 계산해야 한다(설계 11.1 R4). tests/test_aero.py가 계수를 지킨다.
AERO_STALL_WIDTH = 0.035
"""실속 시그모이드의 폭(rad): σ = 1 / (1 + exp(-(받음각 - stall) / 폭))."""
AERO_CP_ATTACHED = 0.25
"""붙은 흐름의 압력 중심(앞쪽 끝에서 변 길이 비율). 완전 실속이면 여기에 AERO_CP_STALL_SHIFT를 더한다."""
AERO_CP_STALL_SHIFT = 0.25
AERO_CD90_BASE = 1.11
AERO_CD90_PER_AR = 0.018
AERO_CD90_AR_CAP = 50.0
AERO_CD90_2D = 1.98
"""가로세로비 0(끝없이 긴 날개)의 평판 수직 힘 계수."""
AERO_INDUCED_EFF = 0.9
"""유도 항력 CL² / (e·π·AR)의 효율 e."""
AERO_STALL_RANGE = (0.01, 1.5)
AERO_MAX_DENSITY = 1e3
AERO_MAX_SPAN = 1e4
AERO_MAX_FRICTION = 10.0
AERO_MIN_SPEED2 = 1e-12
"""상대 바람 속력²이 이보다 작으면 그 변에는 힘이 없다."""
JUMP_NEVER = -(10**9)
"""아직 뛴 적 없는 점프 추진기의 '마지막 점프 스텝'(첫 스텝부터 뛸 수 있게)."""
_INFS = (math.inf, -math.inf)
"""step_values가 0으로 바꾸는 무한대 둘(x in _INFS = x가 ±무한대)."""
_NO_DRAG: list[tuple[float, float]] = []
"""바람 항력을 받는 물체가 없을 때 _apply_forces가 쓰는 표시(물체마다 항력 (0.0, 0.0))."""


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
    motion: MotionPlan | None = None
    """정해진 움직임 세계 물체(v2)면 그 움직임: 시각 t의 자세 = world.physics2d.motion.pose_at(motion, t).
    궤적(moving_bodies)에는 들어가지 않으므로 화면은 이 함수로 그린다."""


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
# pymunk C 함수 바로 부르기 (레이 질의, 정해진 움직임 물체)
# ---------------------------------------------------------------------------

try:
    # pymunk의 cffi 바인딩(공개 API가 아님). pymunk의 공개 API는 같은 C 함수를 부른 뒤 결과를 파이썬 객체로 감싼다:
    # Space.segment_query_first는 객체 넷(모양 찾기, Vec2d 둘, SegmentQueryInfo), Body.position은 Vec2d, 속성 쓰기는
    # mypy용 __setattr__ 한 겹을 더 지난다. 레이 하나에 그 감싸기가 C 질의의 2~5배라 레이 7개와 정해진 움직임 물체가 있는
    # 점프 게임 스텝 비용의 큰 몫이었다(v2.0 M2). 같은 C 함수를 같은 인수로 바로 부르므로 결과는 비트 단위로 같다
    # (레이: C 함수가 결과를 {모양 없음, alpha 1.0}으로 시작해 더 가까운 맞음만 덮어쓰므로, 맞지 않으면 alpha가 공개 API
    # 길이 쓰는 1.0 그대로다). 이 판의 pymunk에 없으면 공개 API로 돌아간다(tests/test_runtime_v2.py가 두 길의 결과가
    # 같은지와 고정한 pymunk에서 빠른 길이 쓰이는지를 지킨다).
    from pymunk._chipmunk_cffi import ffi as _CP_FFI
    from pymunk._chipmunk_cffi import lib as _CP_LIB

    _CP_SEGMENT_FIRST: Any = _CP_LIB.cpSpaceSegmentQueryFirst
    _CP_GET_POSITION: Any = _CP_LIB.cpBodyGetPosition
    _CP_GET_ANGLE: Any = _CP_LIB.cpBodyGetAngle
    _CP_GET_VELOCITY: Any = _CP_LIB.cpBodyGetVelocity
    _CP_GET_ANGULAR_VELOCITY: Any = _CP_LIB.cpBodyGetAngularVelocity
    _CP_GET_FORCE: Any = _CP_LIB.cpBodyGetForce
    _CP_GET_TORQUE: Any = _CP_LIB.cpBodyGetTorque
    _CP_SET_VELOCITY: Any = _CP_LIB.cpBodySetVelocity
    _CP_SET_ANGULAR_VELOCITY: Any = _CP_LIB.cpBodySetAngularVelocity
    _CP_SET_FORCE: Any = _CP_LIB.cpBodySetForce
    _CP_SET_TORQUE: Any = _CP_LIB.cpBodySetTorque
    _CP_MOTOR_SET_RATE: Any = _CP_LIB.cpSimpleMotorSetRate
    _CP_SPACE_STEP: Any = _CP_LIB.cpSpaceStep
    _CP_FAST = True
except (ImportError, AttributeError):  # pragma: no cover - pymunk 내부가 바뀐 판
    _CP_FAST = False
"""C 함수를 바로 부르는 길을 쓰는가. Episode를 만들 때 읽는다(테스트는 False로 바꿔 공개 API 길과 비교한다)."""


def _segment_first_c(space: pymunk.Space, filt: pymunk.ShapeFilter) -> tuple[Any, Any, Any, Any, Any] | None:
    """space.segment_query_first(시작, 끝, 0.0, filt)와 같은 C 질의를 감싸기 없이 부를 재료:
    (C 함수, cpSpace, 필터 구조체, 필터 구조체의 주인, 결과 구조체). 쓸 수 없으면 None(공개 API를 쓴다).

    부르는 법: query(raw, 시작, 끝, 0.0, cfilt, info) 뒤 info.alpha가 맞은 비율(맞지 않으면 1.0). 결과 구조체는 이 레이
    센서만 쓴다(질의마다 C 함수가 처음 값으로 다시 채운다). 주인을 함께 붙잡아 필터 구조체 메모리가 살아 있게 한다."""
    raw = _space_c(space)
    if raw is None:
        return None
    owner = _CP_FFI.new("cpShapeFilter *", (filt.group, filt.categories, filt.mask))
    return _CP_SEGMENT_FIRST, raw, owner[0], owner, _CP_FFI.new("cpSegmentQueryInfo *")


def _body_c(body: pymunk.Body) -> Any:
    """body의 cpBody 포인터(C 함수에 바로 넘길 것). 쓸 수 없으면 None(공개 API를 쓴다). 포인터 객체가 C 메모리의 주인이라
    붙잡고 있는 동안 물체가 풀리지 않는다."""
    return getattr(body, "_body", None) if _CP_FAST else None


def _space_c(space: pymunk.Space) -> Any:
    """space의 cpSpace 포인터(cpSpaceStep·레이 질의에 바로 넘길 것). 쓸 수 없으면 None(공개 API를 쓴다)."""
    return getattr(space, "_space", None) if _CP_FAST else None


def _constraint_c(constraint: pymunk.Constraint) -> Any:
    """관절(모터)의 cpConstraint 포인터. 쓸 수 없으면 None(공개 API를 쓴다)."""
    return getattr(constraint, "_constraint", None) if _CP_FAST else None


def fastpath_in_use(spec: EnvSpec, trial: Trial | None = None) -> bool:
    """이 환경의 Episode가 정말 C 함수 길을 모두 쓰는가(진단: bench 「pymunk C 함수 바로 부르기」 줄, 자체 점검 「빠른 계산 길」).

    _CP_FAST는 pymunk의 cp* 함수를 불러왔다는 것뿐이다. 포인터는 Episode마다 pymunk 내부 속성(_body·_space·_constraint)에서
    따로 얻으므로, 그 이름이 바뀐 판이면 _CP_FAST가 참이어도 Episode가 (일부) 공개 API 길로 돌아간다(결과는 같고 느림). 처음 몸
    Episode 하나를 만들어 두 스텝 돌린 뒤 모든 물체의 cpBody 포인터, cpSpace 포인터(레이 질의도 같은 속성을 읽는다), 모터가
    있으면 모터 포인터를 본다. 학습·평가와 무관한 Episode라 결과에 영향이 없다."""
    if not _CP_FAST:
        return False
    ep = Episode(spec, None, trial if trial is not None else Trial(0))
    for _ in range(2):
        ep.step([0.0] * len(spec.outputs))
    return (bool(ep._raw) and all(r is not None for r in ep._raw.values()) and ep._raw_space is not None
            and (not ep._motors or ep._motor_raw is not None))


# ---------------------------------------------------------------------------
# 만들기 단계 도우미
# ---------------------------------------------------------------------------


_TRACK_CACHE: dict[tuple[Any, ...], Any] = {}
"""만든 트랙(순수 함수 결과). 워커는 같은 학습 트랙 세트를 계속 쓰므로 한 번만 만든다."""


def _cached_track(key: tuple[Any, ...], make: Callable[[], Any]) -> Any:
    t = _TRACK_CACHE.get(key)
    if t is None:
        if len(_TRACK_CACHE) >= 256:
            _TRACK_CACHE.clear()
        t = make()
        _TRACK_CACHE[key] = t
    return t


def _sub_seed(seed: int, name: str) -> int:
    """시험 시드와 이름에서 63비트 시드를 만든다(파이썬 hash와 무관)."""
    ss = np.random.SeedSequence(entropy=seed & ((1 << 128) - 1), spawn_key=(zlib.crc32(name.encode("utf-8")),))
    a, b = (int(x) for x in ss.generate_state(2, np.uint32))
    return (a | (b << 32)) & 0x7FFFFFFFFFFFFFFF


def _cross(p: geo.Point, q: geo.Point, a: geo.Point, b: geo.Point) -> bool:
    """p에서 q로 움직이며 체크포인트 선 ab를 앞쪽으로 지났는가.

    선의 방향은 진행 방향으로 볼 때 왼쪽 끝 a → 오른쪽 끝 b다(트랙 생성기는 안쪽 벽 → 바깥쪽 벽). 그러면 앞쪽은
    orient(a, b, ·) > 0인 쪽이다. 뒤쪽(또는 선 위)에서 앞쪽으로 넘어갈 때만 센다: 뒤로 넘거나 앞뒤로 흔들어도 세지 않는다.
    """
    def orient(u: geo.Point, v: geo.Point, w: geo.Point) -> float:
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])

    d_from, d_to = orient(a, b, p), orient(a, b, q)
    if not (d_from <= 0.0 < d_to):
        return False
    d3, d4 = orient(p, q, a), orient(p, q, b)
    return (d3 > 0) != (d4 > 0) or d3 == 0 or d4 == 0


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
# 몸 상태 값·센서 특수화 (v6 설계 4.3, D9)
# ---------------------------------------------------------------------------
# 스텝마다 항목 이름을 문자열로 비교하던 일을 만들 때 한 번으로 옮긴다. 식(각도 (a + π) % τ − π, 속력 hypot, 몸 좌표 회전)과
# 값은 Episode._state·옛 body_state와 같다(tests/test_speed_runtime.py T8이 모든 항목·좌표계를 무작위 상태로 비교한다).

_STATE_INDEX = {"x": 0, "y": 1, "vx": 3, "vy": 4, "omega": 5}
"""_bstate 튜플 (x, y, 각도, vx, vy, 각속도)에서 그대로 읽는 세계 좌표 항목."""
_ANGLE, _SPEED, _FWD, _LAT = 6, 7, 8, 9
"""몸 상태 센서가 _bstate 튜플 뒤에 덧붙여 읽는 계산 값의 자리: 감싼 각도, 속력, 앞뒤 속도, 옆 속도."""


def _state_getter(bstate: Callable[[str], tuple[float, ...]], bid: str, f: str) -> Callable[[], float]:
    """수식 값 "<물체>.<항목>"(sy.BODY_STATE, 세계 좌표) 하나를 읽는 함수 = v3.1 Episode._state(bid, f)와 같은 값."""
    if f == "angle":
        pi, tau = math.pi, geo.TAU

        def angle() -> float:
            return (bstate(bid)[2] + pi) % tau - pi  # geo.wrap_angle과 같은 식
        return angle
    if f == "speed":
        hypot = math.hypot

        def speed() -> float:
            st = bstate(bid)
            return hypot(st[3], st[4])
        return speed
    k = _STATE_INDEX.get(f, 3 if f == "vx" else 4)

    def item() -> float:
        return bstate(bid)[k]
    return item


def _body_state_fn(bstate: Callable[[str], tuple[float, ...]], body: str, fields: tuple[str, ...],
                   frame: str) -> Callable[[], tuple[float, ...]]:
    """몸 상태 센서 하나의 원래 값 함수. 항목마다 읽을 자리를 만들 때 한 번 정한다: 세계 좌표 항목은 _bstate 튜플에서
    그대로(operator.itemgetter), 감싼 각도·속력·몸 좌표 속도가 필요하면 그 값만 계산해 튜플 뒤에 붙인 뒤 읽는다.
    몸 좌표 속도는 한 번의 cos·sin 뒤 fwd = vx·c + vy·s, lat = −vx·s + vy·c(옛 body_state와 같은 식)."""
    need_body = frame == "body" or "v_fwd" in fields or "v_lat" in fields
    slots: list[int] = []
    for f in fields:
        if f == "angle":
            slots.append(_ANGLE)
        elif f == "speed":
            slots.append(_SPEED)
        elif f == "v_fwd" or (f == "vx" and frame == "body"):
            slots.append(_FWD)
        elif f == "v_lat" or (f == "vy" and frame == "body"):
            slots.append(_LAT)
        else:
            slots.append(_STATE_INDEX.get(f, 3 if f == "vx" else 4))
    need_angle, need_speed = _ANGLE in slots, _SPEED in slots
    extra = need_body or need_angle or need_speed
    pick: Callable[[tuple[float, ...]], tuple[float, ...]]
    if len(slots) == 1:
        k = slots[0]
        pick = lambda st: st[k:k + 1]  # noqa: E731 - 항목 하나: 튜플 한 칸
    else:
        pick = operator.itemgetter(*slots)
    if not extra:
        return lambda: pick(bstate(body))
    cos, sin, hypot, pi, tau = math.cos, math.sin, math.hypot, math.pi, geo.TAU

    def body_state() -> tuple[float, ...]:
        st = bstate(body)
        a, vx, vy = st[2], st[3], st[4]
        if need_body:
            c, si = cos(a), sin(a)
            fwd, lat = vx * c + vy * si, -vx * si + vy * c
        else:
            fwd = lat = 0.0
        return pick(st + ((a + pi) % tau - pi if need_angle else 0.0, hypot(vx, vy) if need_speed else 0.0, fwd, lat))

    return body_state


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
    memo: dict[Any, bool] = field(default_factory=dict)
    """모양 → 선택자에 맞음(v6 D11). 모양의 물체·꼬리표는 만든 뒤 바뀌지 않으므로 처음 볼 때 한 번 _matches로 정한다."""


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
class _Vehicle:
    body: pymunk.Body
    bid: str
    ch_throttle: int
    ch_steer: int
    accel: float
    steer: float
    grip: float
    wheelbase: float
    drag: float
    mass: float
    lat_max: float = 0.0
    slot: int = 0
    decay: float = 1.0
    """한 스텝 동안 구름 저항으로 남는 속도 비율 exp(-drag x dt)."""
    gain: float = 0.0
    """한 스텝 동안 가속 1(m/s²)이 더하는 속도 (1 - decay) / drag (drag가 0이면 dt)."""
    raw: Any = None
    """cpBody 포인터(_setup_c_paths, v6 D7). None이면 공개 API를 쓴다."""


@dataclass
class _Motor:
    constraint: pymunk.SimpleMotor
    channel_index: int
    max_speed: float


@dataclass
class _Jump:
    """점프 추진기(v2 설계 4.3)."""

    channel_index: int
    body: pymunk.Body
    bid: str
    speed: float
    """뛰는 방향 속도(m/s, 0 이상)."""
    dx: float
    dy: float
    """뛰는 방향 단위벡터(세계 좌표)."""
    selectors: tuple[str, ...]
    """땅으로 볼 것(물체 id, tag.<태그>). 비면 공중에서도 뛴다."""
    cooldown_steps: int
    """마지막으로 뛴 스텝에서 이만큼 지나야 다시 뛴다: ⌈cooldown / dt - 1e-9⌉(사건 hold와 같은 정수 규칙)."""
    ground: _ContactChannel | None = None
    """땅 접촉 채널(앞 스텝 값, 디바운스 적용). _setup_contacts 뒤에 _connect_jumps가 잇는다."""
    last: int = JUMP_NEVER
    raw: Any = None
    """cpBody 포인터(_setup_c_paths, v6 D7). None이면 공개 API를 쓴다."""


@dataclass
class _AeroPart:
    """공기역학 부품 하나(v2 설계 4.5): 물체 좌표의 테두리 변 목록과 계수."""

    body: pymunk.Body
    bid: str
    edges: list[tuple[float, float, float, float, float, float, float]]
    """변마다 (바깥 법선 x, y, 변 벡터 x1 - x0, y1 - y0, 변 가운데 - 무게중심 x, y, 길이), 물체 좌표(반시계).
    무게중심은 만든 뒤 바뀌지 않으므로 스텝마다 다시 빼지 않는다(같은 값)."""
    qk: float
    """½ x 공기 밀도 x 날개 폭."""
    slope: float
    """붙은 흐름 양력 기울기 2π / (1 + 2 / AR)(AR = 0이면 2π)."""
    cd90: float
    stall: float
    friction: float
    induced: float
    """유도 항력 계수 1 / (e·π·AR)(AR = 0이면 0)."""


@dataclass
class _Checkpoints:
    id: str
    lines: list[tuple[geo.Point, geo.Point]]
    loop: bool
    body: str
    track: Any = None
    next: int = 0
    count: int = 0
    laps: int = 0
    last_time: float = 0.0
    lap_start: float = 0.0
    lap_times: list[float] = field(default_factory=list)
    prev: geo.Point = (0.0, 0.0)
    progress: float = 0.0
    s_prev: float = 0.0
    hint: int | None = None
    done: bool = False
    need_progress: bool = False


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
        self.max_steps = max(1, math.ceil(min(env.world.max_time / self.dt, MAX_STEPS) - 1e-9))
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
        self._tagsets: dict[tuple[str, ...], frozenset[str]] = {}
        """꼬리표 목록 → frozenset(같은 내용이면 하나를 나눠 씀, v6 D13b)."""
        self._geoms: list[BodyGeom] = []
        self._regions: dict[str, _Region] = {}
        self._region_geoms: list[RegionGeom] = []
        self._agent_ids: list[str] = []
        self._drag: list[tuple[str, float]] = []
        self._start_pos: dict[str, geo.Point] = {}
        self._kinematic: list[tuple[pymunk.Body, MotionPlan]] = []
        """정해진 움직임 물체(v2): 장애물 줄(생성기 순서) 다음 kinematic 세계 물체(선언 순서)."""
        self._kinematic_drive: list[tuple[pymunk.Body, Any, MotionPlan, bool]] = []
        """_kinematic과 같은 순서로 (물체, cpBody 포인터 또는 None, 움직임, 도는가). _drive_kinematic이 쓴다."""
        self._aero_clamped = 0
        self._absent = self._structure_absent()
        """없는 에이전트 물체(v2.1). 있음 칸을 쓰는 물체가 없으면 빈 집합(그때는 아래 만들기가 v2.0과 같은 길만 지난다)."""
        self._absent_joints = frozenset(j.id for j in env.joints if j.a in self._absent or j.b in self._absent)
        """끝 하나가 없는 물체라 만들지 않는 관절."""
        self._dead_channels: list[int] = []
        """없는 부품의 구동기 채널 번호(오름차순). 명령도 실제 출력도 늘 0이다."""

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
        self._setup_c_paths()
        self._connect_jumps()
        self._build_checkpoints()
        self._values: dict[str, Any] = {}
        self._build_sensors()
        self._build_scoring()

        self.moving_bodies = [g.id for g in self._geoms if g.moving]
        self._moving = [self.bodies[i] for i in self.moving_bodies]
        self._moving_raw = [self._raw[i] for i in self.moving_bodies]
        # hold초 유지: 조건이 처음 참이 된 스텝부터 hold초가 지난 스텝에 일어난다(hold=0이면 바로).
        self._events = [_EventState(e, 1 + max(0, math.ceil(min(ev.hold / self.dt, MAX_STEPS) - 1e-9)))
                        for e, ev in self._event_defs()]
        self._builtin = {"timeout": _EventState("timeout", 1), "out_of_bounds": _EventState("out_of_bounds", 1)}
        states = {**self._builtin, **{st.id: st for st in self._events}}
        self._end_states = [(end, states[end.event]) for end in spec.terminations if end.event in states]
        # 사건 갱신 계획(v6 D10): (상태, 조건 함수(내장 사건은 None), "<id>.count" 또는 None, "<id>.since" 또는 None).
        # 순서는 옛 _update_events와 같다: timeout, out_of_bounds, 사건 순서(spec.event_order).
        self._fire_plan: list[tuple[_EventState, Callable[[dict[str, Any]], Any] | None, str | None, str | None]] = [
            (st, cond, f"{st.id}.count" if st.id in self._count_needed else None,
             f"{st.id}.since" if st.id in self._since_needed else None)
            for st, cond in [(self._builtin["timeout"], None), (self._builtin["out_of_bounds"], None),
                             *((st, spec.events[st.id]) for st in self._events)]
        ]
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
        self.step_values(action.tolist() if isinstance(action, np.ndarray) else list(action))

    def step_values(self, values: list[float]) -> None:
        """step()의 몸통(v6 설계 D14·7.3): 행동을 파이썬 float 목록으로 받는다. 평가기는 묶음마다 ndarray.tolist()를 한 번
        하고 행마다 이것을 부른다(쌍마다 ndarray 행을 만들지 않는다). 규칙은 step()과 같다: 끝났으면 아무 일도 안 하고, 길이가
        다르면 RT_ACTION_SIZE ValueError, 값마다 float() 뒤 NaN·무한대는 0, 구동기 범위로 자르고, 없는 부품(v2.1)의 채널은 0.
        받은 목록은 고치지 않는다."""
        if self._done:
            return
        if len(values) != len(self._out_index):
            raise ValueError(S.RT_ACTION_SIZE.format(got=len(values), want=len(self._out_index)))
        self._u_prev = self._u
        u = [0.0] * len(self._u_names)
        for x, (idx, lo, hi) in zip(values, self._out_plan, strict=True):
            x = float(x)
            if x != x or x in _INFS:
                x = 0.0
            u[idx] = lo if x < lo else (hi if x > hi else x)
        for idx in self._dead_channels:  # 없는 부품의 구동기(v2.1): 명령도 0(부드러움 블록이 세지 않게)
            u[idx] = 0.0
        self._u = u

        self._apply_actuators()
        if self._force_raw:  # 힘을 모으는 물체가 없으면 _apply_forces는 할 일이 없다
            self._apply_forces()
        if self._aero:
            self._apply_aero()
        if self._kinematic:
            self._drive_kinematic()
        raw_space = self._raw_space
        if raw_space is not None:  # 둘째 스텝부터: Space.step과 같은 C 함수(v6 D8, _first_space_step)
            _CP_SPACE_STEP(raw_space, self.dt)
        else:
            self._first_space_step()
        self._cache.clear()
        self._pos_cache.clear()
        self.steps += 1
        self.time = self.steps * self.dt

        if self._cps:
            self._update_checkpoints()
        self._update_contacts()
        if self._value_getters:  # 첫 _refresh_values(만들 때) 뒤로는 읽을 값이 없으면 할 일이 없다
            self._refresh_values()
        self._update_events()
        for f in self._block_steps:  # 점수 블록 누적(v3.1 _accumulate_blocks)
            f()
        for end, st in self._end_states:  # 종료(v3.1 _check_termination): 종료 목록 순서로 처음 일어난 사건
            if st.occurred:
                self._done = True
                self._end = end
                return
        if self.steps >= self.max_steps:  # 안전장치(timeout이 늘 목록에 있으므로 보통 오지 않는다)
            self._done = True

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
        for i, (b, raw) in enumerate(zip(self._moving, self._moving_raw, strict=True)):
            if raw is not None:  # 공개 API와 같은 C 함수(v6 D6)
                p = _CP_GET_POSITION(raw)
                poses[i, 0], poses[i, 1] = p.x, p.y
                poses[i, 2] = _CP_GET_ANGLE(raw)
            else:
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

    @property
    def applied_outputs(self) -> list[float]:
        """구동기 채널별 마지막 스텝의 실제 출력(연료가 떨어지면 0). 관찰 기록이 읽기만 한다(고치지 말 것).

        _apply_actuators가 스텝마다 새 리스트로 바꾸므로, 받은 리스트는 그 스텝의 값으로 남는다."""
        return self._u_applied

    @property
    def output_channel_index(self) -> list[int]:
        """신경망 출력 k번 → 채널 번호(applied_outputs의 색인)."""
        return list(self._out_index)

    @property
    def aero_clamped(self) -> int:
        """공기힘 한 스텝 상한이 걸린 (물체, 스텝) 수(v2 설계 D15·7.3). 보통 값에서는 0이다(테스트, bench용)."""
        return self._aero_clamped

    @property
    def aero_edge_count(self) -> int:
        """공기를 받는 테두리 변의 수(모든 공기역학 부품의 합, bench용)."""
        return sum(len(a.edges) for a in self._aero)

    def jump_ready(self, channel: str) -> bool:
        """점프 추진기 채널(구동기 id)에 다음 step()에서 출력 1을 주면 뛰는가: 땅 접촉(앞 스텝 값, 디바운스 적용)과
        냉각 조건만 본다(직접 조종 HUD 「점프 준비됨 / 공중」). 점프 추진기 채널이 아니면 False. 읽기 전용."""
        for jp in self._jumps:
            if self._u_names[jp.channel_index] == channel:
                return self._jump_allowed(jp)
        return False

    @property
    def anchor_body(self) -> str | None:
        """궤적 밀도의 기준 물체: 움직이는 에이전트 물체 중 첫째(캔버스 꼬리와 같은 moving_bodies[0] 규칙). 없으면 None."""
        return next((g.id for g in self._geoms if g.moving and g.owner == "agent"), None)

    @property
    def absent(self) -> frozenset[str]:
        """없는 에이전트 물체 id(v2.1 몸 구조, 읽기 전용): 에이전트 물체 − structure.present(spec, params). 몸 수치만의 함수라
        시험·세대·워커와 무관하다. 있음 칸을 쓰는 물체가 없는 환경이면 빈 집합. 클릭 검사 「(없음)」 표시가 쓴다."""
        return self._absent

    @property
    def absent_joints(self) -> frozenset[str]:
        """끝 하나가 없는 물체라 만들지 않은 관절 id(v2.1, 읽기 전용). 그 관절의 모터 채널은 죽은 채널이고 관절 각도 센서는 0."""
        return self._absent_joints

    @property
    def dead_channels(self) -> tuple[str, ...]:
        """없는 부품의 구동기 채널 이름(v2.1, 읽기 전용, actuator_channels 순서). 명령도 실제 출력도 늘 0이다. 직접 조종
        창이 듣지 않는 키를 도움말에서 뺄 때 쓴다."""
        return tuple(self._u_names[i] for i in self._dead_channels)

    def _structure_absent(self) -> frozenset[str]:
        """이어짐 규칙(spec/structure.py)으로 없는 에이전트 물체를 구한다. 있음 칸이 하나도 없으면 계산하지 않는다."""
        env = self.env
        if all(b.exists is None for b in env.bodies):
            return frozenset()
        agents = frozenset(b.id for b in env.bodies if b.owner == "agent")
        return agents - structure.present(self.spec, self.params)

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
        motion: MotionPlan | None = None,
    ) -> None:
        self.bodies[body_id] = body
        pyshapes = [s for s, _ in shapes]
        # 같은 (물체, 꼬리표)의 모양은 (id, frozenset) 하나를 함께 쓴다(v6 D13b: 읽기만 함, Episode 안에서만 나눔).
        infos: dict[tuple[str, ...], tuple[str, frozenset[str]]] = {}
        tagsets = self._tagsets
        for k, s in enumerate(pyshapes):
            t = tuple(per_shape_tags[k] if per_shape_tags is not None else tags)
            info = infos.get(t)
            if info is None:
                fs = tagsets.get(t)
                if fs is None:
                    fs = tagsets[t] = frozenset(t)
                info = infos[t] = (body_id, fs)
            self._shape_info[s] = info
        if body.body_type == pymunk.Body.STATIC and len(pyshapes) > SHUFFLE_ABOVE:
            # Chipmunk의 경계 상자 트리는 한 줄로 늘어선 선분을 차례로 넣으면 한쪽으로 쏠려 넣기가 제곱 시간이 된다
            # (2만 개: 차례로 4.4초, 섞으면 0.04초). 늘 같은 순서로 섞어 결정성은 그대로다.
            order = np.random.default_rng(SHUFFLE_SEED).permutation(len(pyshapes))
            pyshapes = [pyshapes[int(k)] for k in order]
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
                     (body.position.x, body.position.y, body.angle), motion)
        )
        if motion is not None:
            self._kinematic.append((body, motion))
            # 돌지 않는 움직임(spin 0, 물체 각도 = 움직임 각도)이면 각도가 늘 같아 _drive_kinematic의 각속도가 늘 0.0이다.
            # 처음 각속도가 0.0이고 그 물체의 각속도를 바꾸는 것이 없으므로 그때는 각속도 쓰기를 건너뛰어도 같다.
            turns = not (motion.spin == 0.0 and body.angle == pose_at(motion, 0.0)[2] and body.angular_velocity == 0.0)
            self._kinematic_drive.append((body, _body_c(body), motion, turns))

    def _build_generators(self) -> None:
        self._cp_defs: list[tuple[str, list[tuple[geo.Point, geo.Point]], bool, str | None, Any]] = []
        for g in self.env.generators:
            if isinstance(g, m.HeightmapGen):
                self._build_heightmap(g)
            elif isinstance(g, m.TrackGen):
                self._build_track(g)
            elif isinstance(g, m.HillGen):
                self._build_hills(g)
            elif isinstance(g, m.ObstacleGen):
                self._build_obstacles(g)
        self._body_id: dict[pymunk.Body, str] = {}

    def _build_obstacles(self, g: m.ObstacleGen) -> None:
        """장애물 줄(v2 설계 4.2): 정해진 움직임 세계 물체 하나에 장애물 사각형을 줄지어 놓는다.

        시드와 모든 칸은 생성기 단계(r.*)라 같은 시험의 모든 개체가 같은 줄을 본다. 물체는 세계 원점에 서고, 장애물은
        place_boxes가 미는 방향 d에 맞춰 놓은 물체 좌표 사각형이다(방향 π면 줄 좌표 그대로). 움직임은 길
        [(0, 0), length x (cos d, sin d)]을 speed·accel로 한 번 가는 MotionPlan이다."""
        from evolab.generators.obstacles import generate_obstacles, place_boxes

        num = lambda v: self._num(v, "generator")  # noqa: E731
        pair = lambda r: (num(r[0]), num(r[1]))  # noqa: E731
        speed, accel = max(0.0, num(g.speed)), num(g.accel)
        row = generate_obstacles(
            int(num(g.seed)), count=int(g.count), x=num(g.x), y=num(g.y), gap=pair(g.gap), width=pair(g.width),
            height=pair(g.height), air_chance=_clamp(num(g.air_chance), 0.0, 1.0), air_y=pair(g.air_y),
            air_height=pair(g.air_height), speed=speed, accel=accel,
        )
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        friction = _clamp(num(g.friction), 0.0, 1e3)
        shapes: list[tuple[pymunk.Shape, ShapeGeom]] = []
        d = num(g.direction)
        for x0, y0, x1, y1 in place_boxes(row, d):
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            poly = pymunk.Poly(body, pts)
            poly.friction = friction
            shapes.append((poly, ShapeGeom("poly", pts)))
        tags = [["obstacle", "air"] if a else ["obstacle"] for a in row.air]
        plan = MotionPlan.path((0.0, 0.0), 0.0, [(0.0, 0.0), (row.length * math.cos(d), row.length * math.sin(d))],
                               speed, accel, 0.0, "once", 0.0)
        self._add_body(g.id, body, shapes, ["obstacle"], "world", tags, motion=plan)
        self.gen_values.update({f"gen.{g.id}.speed": speed, f"gen.{g.id}.accel": accel,
                                f"gen.{g.id}.length": row.length})

    def _build_hills(self, g: m.HillGen) -> None:
        from evolab.generators.terrain import generate_hills

        num = lambda v: self._num(v, "generator")  # noqa: E731
        try:
            xs, ys = generate_hills(
                int(num(g.seed)) & 0x7FFFFFFFFFFFFFFF, x_min=num(g.x_range[0]), x_max=num(g.x_range[1]),
                segment=g.segment, base_y=num(g.base_y), amplitude=max(0.0, num(g.amplitude)),
                growth=max(0.0, num(g.growth)), roughness=min(1.0, max(0.0, num(g.roughness))),
                slope=num(g.slope), flat_start=max(0.0, num(g.flat_start)),
            )
        except ValueError as exc:
            raise EpisodeBuildError(f"generators.{g.id}: {exc}") from None
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        friction = _clamp(num(g.friction), 0.0, 1e3)
        shapes: list[tuple[pymunk.Shape, ShapeGeom]] = []
        for i in range(len(xs) - 1):
            a = (float(xs[i]), float(ys[i]))
            b = (float(xs[i + 1]), float(ys[i + 1]))
            seg = pymunk.Segment(body, a, b, g.radius)
            seg.friction = friction
            shapes.append((seg, ShapeGeom("segment", [a, b], g.radius)))
        self._add_body(g.id, body, shapes, ["terrain"], "world")
        self.gen_values[f"gen.{g.id}.x_min"] = float(xs[0])
        self.gen_values[f"gen.{g.id}.x_max"] = float(xs[-1])

    def _choose_track(self, g: m.TrackGen) -> Any:
        from evolab.generators import track as tg

        num = lambda v: self._num(v, "generator")  # noqa: E731
        width = num(g.width)
        params = {
            "segments": int(g.segments), "curvature": num(g.curvature), "width": width,
            "length_range": (float(g.length_range[0]), float(g.length_range[1])), "checkpoints": int(g.checkpoints),
        }
        frozen = tuple(sorted(params.items()))
        if g.source == "drawn":
            pts = tuple(self._vec(q, "generator") for q in (g.centerline or []))
            return _cached_track(("drawn", pts, width, int(g.checkpoints)),
                                 lambda: tg.track_from_centerline(pts, width=width, checkpoints=int(g.checkpoints)))
        if self.trial.track is not None:
            tracks = _cached_track(("set", int(g.set_seed), int(g.set_size), frozen),
                                   lambda: tg.track_set(int(g.set_seed), int(g.set_size), **params))
            return tracks[int(self.trial.track) % len(tracks)]
        seed = _sub_seed(self.trial.seed, f"track:{g.id}")
        return _cached_track(("trial", seed, frozen), lambda: tg.generate_track(seed, **params))

    def _build_track(self, g: m.TrackGen) -> None:
        try:
            track = self._choose_track(g)
        except ValueError as exc:
            raise EpisodeBuildError(f"generators.{g.id}: {exc}") from None
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        friction = _clamp(self._num(g.friction, "generator"), 0.0, 1e3)
        shapes: list[tuple[pymunk.Shape, ShapeGeom]] = []
        for poly in (track.inner, track.outer):
            pts = [(float(x), float(y)) for x, y in poly]
            n = len(pts)
            for i in range(n):
                a, b = pts[i], pts[(i + 1) % n]
                seg = pymunk.Segment(body, a, b, g.wall_radius)
                seg.friction = friction
                shapes.append((seg, ShapeGeom("segment", [a, b], g.wall_radius)))
        self._add_body(f"{g.id}.walls", body, shapes, ["wall"], "world")
        x, y, heading = track.start
        self.gen_values.update({
            f"gen.{g.id}.start.x": float(x), f"gen.{g.id}.start.y": float(y),
            f"gen.{g.id}.start.angle": float(heading), f"gen.{g.id}.length": float(track.length),
        })
        lines = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))) for a, b in track.checkpoints]
        self._cp_defs.append((f"{g.id}.cp", lines, True, None, track))

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
        for c in self.env.checkpoints:
            lines = [(self._vec(a, "world"), self._vec(b, "world")) for a, b in c.lines]
            self._cp_defs.append((c.id, lines, c.loop, c.body, None))
        for cid, lines, _loop, _body, _track in self._cp_defs:
            for k, (a, b) in enumerate(lines):
                self._region_geoms.append(RegionGeom(f"{cid}.{k}", "checkpoint", [a, b]))
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

        own_density: list[tuple[int, int, float]] = []
        """모양별 밀도(v2): (첫 조각, 끝 조각, 밀도). 모양 하나가 볼록 조각 여럿이 될 수 있다."""
        for s in b.shapes:
            first = len(out)
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
            elif isinstance(s, m.Wing):
                poly(self._wing_points(s, stage))
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
            own = getattr(s, "density", None)
            if own is not None:
                own_density.append((first, len(out), _clamp(self._num(own, stage), 1e-6, 1e9)))
        densities = [density] * len(out)
        for first, last, d in own_density:
            densities[first:last] = [d] * (last - first)
        for shape_density, (shape, _) in zip(densities, out, strict=True):
            shape.friction = friction
            shape.elasticity = elasticity
            if body.body_type == pymunk.Body.DYNAMIC:
                shape.density = max(shape_density, 1e-6)
        return out

    def _wing_points(self, s: m.Wing, stage: sy.BuildStage) -> list[geo.Point]:
        """날개 모양의 꼭짓점 넷(물체 좌표). 공기역학의 테두리 변도 이 점들이다."""
        return geo.wing_points(self._vec(s.le, stage), self._num(s.chord, stage), self._num(s.angle, stage),
                               self._num(s.thick, stage), self._num(s.camber, stage))

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
        absent = self._absent
        for b in env.bodies:
            if b.id in absent:  # 없는 물체(v2.1): 공간·self.bodies·그리기·궤적·경계 검사·바람 항력에 들어가지 않는다
                continue
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
            plan = None
            if b.type == "kinematic" and b.motion is not None and b.owner == "world":
                plan = self._motion_plan(b.motion, lp, la)
                px, py, pa = pose_at(plan, 0.0)
                body.position = (px, py)
                body.angle = pa
            shapes = self._make_shapes(body, b, stage)
            if b.owner == "agent":
                for s, _ in shapes:
                    s.filter = pymunk.ShapeFilter(group=AGENT_GROUP)
            self._add_body(b.id, body, shapes, list(b.tags), b.owner, motion=plan)
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

    def _motion_plan(self, mo: m.PathMotion | m.WaveMotion, pos: geo.Point, angle: float) -> MotionPlan:
        """정해진 움직임의 수식을 세계 단계(r.*, gen.*)로 계산해 숫자로 굳힌다(설계 4.1)."""
        num = lambda v: self._num(v, "world")  # noqa: E731
        if isinstance(mo, m.PathMotion):
            pts = [self._vec(q, "world") for q in mo.points]
            return MotionPlan.path(pos, angle, pts, num(mo.speed), num(mo.accel), num(mo.start), mo.mode, num(mo.spin))
        return MotionPlan.wave(pos, angle, self._vec(mo.amplitude, "world"), num(mo.period), num(mo.phase),
                               num(mo.spin))

    def _drive_kinematic(self) -> None:
        """정해진 움직임 물체마다 다음 스텝 시각의 자세 pose_at((스텝 + 1) x dt)에 정확히 닿는 속도·각속도를 준다.

        위치는 덮어쓰지 않는다: Chipmunk가 그 속도로 위치를 적분해 다음 자세에 닿고(오차가 쌓이지 않게 지금 위치에서
        다시 계산), 접촉하는 물체는 실제 속도를 받아 발판에 실려 가며, 모양 경계 상자가 스텝 안에서 갱신되어 레이와 접촉이
        그 스텝의 자세와 맞다(위치 순간 이동은 위에 올라탄 물체를 남기고 레이가 한 스텝 늦다: 설계 D8)."""
        t1 = (self.steps + 1) * self.dt
        inv = 1.0 / self.dt
        for body, raw, plan, turns in self._kinematic_drive:
            x, y, a = pose_at(plan, t1)
            if raw is not None:  # 공개 API와 같은 C 함수(_body_c)
                p = _CP_GET_POSITION(raw)
                _CP_SET_VELOCITY(raw, ((x - p.x) * inv, (y - p.y) * inv))
                if turns:
                    _CP_SET_ANGULAR_VELOCITY(raw, (a - _CP_GET_ANGLE(raw)) * inv)
            else:
                p = body.position
                body.velocity = ((x - p.x) * inv, (y - p.y) * inv)
                if turns:
                    body.angular_velocity = (a - body.angle) * inv

    def _build_joints(self) -> None:
        for j in self.env.joints:
            if j.id in self._absent_joints:  # 끝 하나가 없는 물체(v2.1)
                continue
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
        self._out_plan = [(i, *self._u_range[i]) for i in self._out_index]
        """신경망 출력 k번 → (채널 번호, 범위 아래, 위). step_values가 스텝마다 찾던 것을 한 번에(v6 D14)."""
        self._tanks: dict[str, _Tank] = {}
        self._thrusters: list[_Thruster] = []
        for a in self.env.actuators:
            if isinstance(a, m.FuelTank):
                cap = max(self._num(a.capacity, "agent"), 1e-9)
                self._tanks[a.id] = _Tank(cap, cap)
        absent = self._absent
        dead = self._dead_channels
        for a in self.env.actuators:
            if isinstance(a, m.Thruster):
                if a.body in absent:
                    dead.append(index[a.id])
                    continue
                d = self._num(a.dir, "agent")
                f = min(max(self._num(a.max_force, "agent"), 0.0), FORCE_LIMIT)
                self._thrusters.append(_Thruster(index[a.id], self.bodies[a.body], self._vec(a.at, "agent"),
                                                 (f * math.cos(d), f * math.sin(d)), a.tank,
                                                 max(0.0, self._num(a.fuel_per_s, "agent"))))
        self._thruster_channels = {t.channel_index for t in self._thrusters}
        self._jumps: list[_Jump] = []
        for a in self.env.actuators:
            if isinstance(a, m.Jump):
                if a.body in absent:
                    dead.append(index[a.id])
                    continue
                d = self._num(a.dir, "agent")
                self._jumps.append(_Jump(
                    index[a.id], self.bodies[a.body], a.body, _clamp(self._num(a.speed, "agent"), 0.0, FIELD_LIMIT),
                    math.cos(d), math.sin(d), tuple(a.ground),
                    max(0, math.ceil(min(float(a.cooldown) / self.dt, MAX_STEPS) - 1e-9)),
                ))
        self._vehicles: list[_Vehicle] = []
        for a in self.env.actuators:
            if isinstance(a, m.Vehicle):
                if a.body in absent:
                    dead += [index[f"{a.id}.throttle"], index[f"{a.id}.steer"]]
                    continue
                body = self.bodies[a.body]
                self._vehicles.append(_Vehicle(
                    body, a.body, index[f"{a.id}.throttle"], index[f"{a.id}.steer"],
                    _clamp(self._num(a.max_accel, "agent"), 0.0, FIELD_LIMIT),
                    _clamp(self._num(a.max_steer, "agent"), 0.0, 1.5),
                    _clamp(self._num(a.lateral_grip, "agent"), 0.0, 1.0),
                    max(1e-3, _clamp(self._num(a.wheelbase, "agent"), 1e-3, FIELD_LIMIT)),
                    _clamp(self._num(a.drag, "agent"), 0.0, 1e3),
                    float(body.mass),
                    _clamp(self._num(a.max_lateral_accel, "agent"), 0.0, FIELD_LIMIT),
                ))
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
        self._motors: list[_Motor] = []
        joints = {j.id: j for j in self.env.joints}
        for a in self.env.actuators:
            if isinstance(a, m.Motor):
                if a.joint in self._absent_joints:
                    dead.append(index[a.id])
                    continue
                j = joints[a.joint]
                motor = pymunk.SimpleMotor(self.bodies[j.a], self.bodies[j.b], 0.0)
                motor.max_force = _clamp(self._num(a.max_torque, "agent"), 0.0, FORCE_LIMIT)
                motor.collide_bodies = j.collide
                self.space.add(motor)
                self._motors.append(_Motor(motor, index[a.id], _clamp(self._num(a.max_speed, "agent"), 0.0, 1e4)))
        for v in self._vehicles:
            # 구름 저항은 지수 감쇠로 한 스텝을 정확히 푼다: 저항이 커도(drag x dt > 1) 발산하지 않고,
            # 최고 속도가 정확히 max_accel / drag다(바람 항력과 같은 방식).
            v.decay = math.exp(-v.drag * self.dt)
            v.gain = (1.0 - v.decay) / v.drag if v.drag > 0.0 else self.dt
        for v in self._vehicles:
            if v.bid not in slot_of:
                slot_of[v.bid] = len(self._force_bodies)
                self._force_bodies.append((v.bid, v.body))
            v.slot = slot_of[v.bid]
        self._drag_slots: list[tuple[int, float]] = []
        for bid, k in self._drag:
            if bid not in slot_of:
                slot_of[bid] = len(self._force_bodies)
                self._force_bodies.append((bid, self.bodies[bid]))
            self._drag_slots.append((slot_of[bid], k))
        self._acc = [[0.0, 0.0, 0.0] for _ in self._force_bodies]
        dead.sort()
        self._build_aero()  # 없는 물체의 공기역학은 그 물체가 self.bodies에 없어 건너뛴다(출력 채널 없음)

    def _connect_jumps(self) -> None:
        """점프 추진기의 땅 접촉 채널을 만든다. 접촉 표(_setup_contacts)가 있어야 하므로 그 뒤에 부른다."""
        for jp in self._jumps:
            if jp.selectors:
                jp.ground = self._contact_channel(jp.bid, jp.selectors)

    def _jump_allowed(self, jp: _Jump) -> bool:
        """땅 접촉(앞 스텝 값, 디바운스 안이면 땅을 떠난 뒤에도: 코요테 시간)과 정수 스텝 냉각."""
        return (jp.ground is None or jp.ground.value) and self.steps - jp.last >= jp.cooldown_steps

    # ------------------------------------------------------------------
    # 공기역학 (v2 설계 4.5)
    # ------------------------------------------------------------------

    def _build_aero(self) -> None:
        """공기역학 부품마다 물체 좌표의 테두리 변 목록을 한 번 만든다.

        surfaces가 all이면 사각형·다각형·방사형·날개 모양, wings면 날개 모양만. 모양의 원래 꼭짓점을 쓰므로 볼록 분해
        조각의 안쪽 변(방사형의 부채꼴 변 포함)은 세지 않는다. 반시계로 맞추고 바깥 법선을 둔다. 원과 선은 받지 않는다."""
        self._aero: list[_AeroPart] = []
        specs = {b.id: b for b in self.env.bodies}
        for a in self.env.actuators:
            if not isinstance(a, m.Aero) or a.body not in specs or a.body not in self.bodies:
                continue
            sb = specs[a.body]
            stage: sy.BuildStage = "agent" if sb.owner == "agent" else "world"
            edges: list[tuple[float, float, float, float, float, float, float]] = []
            cog = self.bodies[a.body].center_of_gravity
            gx, gy = cog.x, cog.y
            for sh in sb.shapes:
                if a.surfaces == "wings" and not isinstance(sh, m.Wing):
                    continue
                if isinstance(sh, m.Box):
                    pts = geo.box_points(self._num(sh.w, stage), self._num(sh.h, stage), self._vec(sh.offset, stage),
                                         self._num(sh.angle, stage))
                elif isinstance(sh, m.Poly):
                    pts = [self._vec(q, stage) for q in sh.points]
                elif isinstance(sh, m.Wing):
                    pts = self._wing_points(sh, stage)
                elif isinstance(sh, m.Radial):
                    pts = geo.radial_points([_clamp(self._num(r, stage), 0.01, 1e5) for r in sh.radii],
                                            self._num(sh.angle, stage))
                else:
                    continue
                if geo.signed_area(pts) < 0:
                    pts = pts[::-1]
                n = len(pts)
                for i in range(n):
                    (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
                    ln = math.hypot(x1 - x0, y1 - y0)
                    if ln <= 1e-9:
                        continue
                    # 반시계 다각형의 바깥 법선 = 변 방향을 오른쪽으로 90도
                    edges.append(((y1 - y0) / ln, -(x1 - x0) / ln, x1 - x0, y1 - y0,
                                  (x0 + x1) * 0.5 - gx, (y0 + y1) * 0.5 - gy, ln))
            ar = max(0.0, self._num(a.aspect_ratio, "agent"))
            self._aero.append(_AeroPart(
                body=self.bodies[a.body], bid=a.body, edges=edges,
                qk=0.5 * _clamp(self._num(a.density, "agent"), 0.0, AERO_MAX_DENSITY)
                * _clamp(self._num(a.span, "agent"), 0.0, AERO_MAX_SPAN),
                slope=geo.TAU / (1.0 + 2.0 / ar) if ar > 0 else geo.TAU,
                cd90=AERO_CD90_BASE + AERO_CD90_PER_AR * min(ar, AERO_CD90_AR_CAP) if ar > 0 else AERO_CD90_2D,
                stall=_clamp(self._num(a.stall, "agent"), *AERO_STALL_RANGE),
                friction=_clamp(self._num(a.friction, "agent"), 0.0, AERO_MAX_FRICTION),
                induced=1.0 / (AERO_INDUCED_EFF * math.pi * ar) if ar > 0 else 0.0,
            ))
        # 한 스텝 상한은 물체마다(설계 D15): 한 물체에 공기역학 부품이 여럿이어도 그 물체의 모든 부품의 힘을 한데 모아
        # 한 번 줄인다. 물체는 처음 나온 부품 순서, 부품은 선언 순서(부품이 하나뿐이면 부품마다 계산하던 것과 같은 계산).
        groups: dict[str, tuple[pymunk.Body, str, list[_AeroPart]]] = {}
        for part in self._aero:
            groups.setdefault(part.bid, (part.body, part.bid, []))[2].append(part)
        self._aero_bodies: list[tuple[pymunk.Body, str, list[_AeroPart]]] = list(groups.values())

    def _apply_aero(self) -> None:
        """변마다 평판 힘을 모아 물체 무게중심의 힘·토크로 더한다(세계 좌표, 추진기·바람 항력에 더함).

        변 가운데에서 잰 상대 바람으로 표면 마찰과 받음각·실속 정도를 정하고, 압력 중심(흐름이 먼저 닿는 끝에서 변 길이
        0.25 + 0.25σ)을 고른 뒤, 그 점에서 다시 잰 바람으로 붙은 흐름 양력·실속 평판 수직 힘·유도 항력을 계산해 그 점에
        건다. 힘 거는 점과 바람 재는 점이 같아야 바람이 없을 때 모든 힘이 그 점의 속도에 수직(양력)이거나 반대(항력)라
        일을 빼앗기만 한다(가운데에서 잰 바람으로 1/4 점에 걸면 도는 물체가 스스로 빨라졌다: 설계 측정).

        한 스텝 상한(설계 D15): D = Σ_힘 (|F| / |u|) x (1/m + ((r x F) / |F|)² / I)이고 D x dt > 1이면 그 물체의 공기힘·토크를
        1 / (D x dt)배로 줄인다(한 스텝의 공기힘 충격이 힘 거는 점의 상대 속도보다 크게 속도를 바꾸지 않게). 합과 상한은 물체마다:
        한 물체에 단 공기역학 부품들의 변을 모두 더한 뒤 한 번 줄이고, 발동 수도 (물체, 스텝)마다 한 번 센다."""
        wx, wy = self.wind
        exp, asin, sqrt, hypot, cos, sin = math.exp, math.asin, math.sqrt, math.hypot, math.cos, math.sin
        width = AERO_STALL_WIDTH
        cp_attached, cp_shift, min_speed2 = AERO_CP_ATTACHED, AERO_CP_STALL_SHIFT, AERO_MIN_SPEED2
        dt = self.dt
        # v6 설계 4.5(D12): 1/질량·1/관성과 부품 값은 만들 때 한 번(_aero_fast), 힘 읽기·쓰기는 C. 식·순서는 그대로이고
        # min(1.0, x)만 같은 값의 조건식 x if x < 1.0 else 1.0(파이썬 min은 같거나 NaN이면 첫 인수 1.0).
        for body, bid, raw, inv_m, inv_i, parts in self._aero_fast:
            _x, _y, a, vx, vy, w = self._bstate(bid)
            c, s_ = cos(a), sin(a)
            fx = fy = tq = 0.0
            dsum = 0.0
            for qk, slope, cd90, stall, fric, induced, edges in parts:
                for nx_, ny_, dx, dy, mx, my, ln in edges:
                    nx, ny = nx_ * c - ny_ * s_, nx_ * s_ + ny_ * c
                    ex, ey = dx * c - dy * s_, dx * s_ + dy * c
                    rx, ry = mx * c - my * s_, mx * s_ + my * c
                    # 1) 변 가운데에서 잰 바람: 표면 마찰, 받음각·실속 정도, 압력 중심
                    ux, uy = wx - (vx - w * ry), wy - (vy + w * rx)
                    u2 = ux * ux + uy * uy
                    if u2 < min_speed2:
                        continue
                    speed = sqrt(u2)
                    un = ux * nx + uy * ny
                    k = qk * speed * ln * fric  # 표면 마찰(모든 변): 변을 따라 흐르는 성분 쪽으로
                    ffx, ffy = k * (ux - un * nx), k * (uy - un * ny)
                    fx += ffx
                    fy += ffy
                    tq += rx * ffy - ry * ffx
                    fm = hypot(ffx, ffy)
                    if fm > 0.0:
                        cr = (rx * ffy - ry * ffx) / fm
                        dsum += fm / speed * (inv_m + cr * cr * inv_i)
                    if un >= 0.0:
                        continue  # 바람을 등진 변
                    sa = -un / speed
                    sa = sa if sa < 1.0 else 1.0  # = min(1.0, sa)
                    sig = 1.0 / (1.0 + exp(-(asin(sa) - stall) / width))
                    xcp = cp_attached + cp_shift * sig
                    f = xcp - 0.5 if ux * ex + uy * ey >= 0.0 else 0.5 - xcp  # 흐름이 먼저 닿는 끝에서 xcp
                    px, py = rx + ex * f, ry + ey * f
                    # 2) 압력 중심에서 다시 잰 바람으로 힘을 계산해 그 점에 건다
                    ux, uy = wx - (vx - w * py), wy - (vy + w * px)
                    u2 = ux * ux + uy * uy
                    if u2 < min_speed2:
                        continue
                    speed = sqrt(u2)
                    un = ux * nx + uy * ny
                    if un >= 0.0:
                        continue
                    sa = -un / speed
                    sa = sa if sa < 1.0 else 1.0  # = min(1.0, sa)
                    ca = sqrt(1.0 - sa * sa)
                    sig = 1.0 / (1.0 + exp(-(asin(sa) - stall) / width))
                    q = qk * u2 * ln
                    cl = slope * sa * ca
                    att = (1.0 - sig) * q
                    if ca > 1e-9:
                        lx, ly = (-nx + un * ux / u2) / ca, (-ny + un * uy / u2) / ca  # -n의 흐름에 수직인 성분 방향
                    else:
                        lx = ly = 0.0
                    fi = att * cl * cl * induced
                    fst = sig * q * cd90 * sa / (0.56 + 0.44 * sa)
                    force_x = att * cl * lx - fst * nx + fi * ux / speed
                    force_y = att * cl * ly - fst * ny + fi * uy / speed
                    fx += force_x
                    fy += force_y
                    tq += px * force_y - py * force_x
                    fm = hypot(force_x, force_y)
                    if fm > 0.0:
                        cr = (px * force_y - py * force_x) / fm
                        dsum += fm / speed * (inv_m + cr * cr * inv_i)
            if dsum * dt > 1.0:
                scale = 1.0 / (dsum * dt)
                fx *= scale
                fy *= scale
                tq *= scale
                self._aero_clamped += 1
            if fx or fy or tq:
                if raw is not None:  # 공개 API와 같은 C 함수(v6 D7)
                    f0 = _CP_GET_FORCE(raw)
                    _CP_SET_FORCE(raw, (f0.x + fx, f0.y + fy))
                    _CP_SET_TORQUE(raw, _CP_GET_TORQUE(raw) + tq)
                else:
                    f0 = body.force
                    body.force = (f0.x + fx, f0.y + fy)
                    body.torque = body.torque + tq

    def _build_checkpoints(self) -> None:
        default = self._vehicles[0].bid if self._vehicles else (self._agent_ids[0] if self._agent_ids else None)
        self._cps: list[_Checkpoints] = []
        for cid, lines, loop, body, track in self._cp_defs:
            bid = body or default
            if bid is None or bid not in self.bodies:
                raise EpisodeBuildError(S.SPEC_NO_AGENT_BODY)
            cp = _Checkpoints(cid, lines, loop, bid, track)
            cp.next = 1 % len(lines) if loop else 1
            cp.done = not loop and len(lines) <= 1
            cp.prev = self._bpos(bid)
            if track is not None:
                from evolab.generators.track import progress_along

                cp.s_prev, cp.hint = progress_along(track, cp.prev[0], cp.prev[1], None)
            self._cps.append(cp)
        self._cp_by_id = {cp.id: cp for cp in self._cps}

    def _update_checkpoints(self) -> None:
        t = self.time
        for cp in self._cps:
            p = self._bpos(cp.body)
            q = cp.prev
            cp.prev = p
            for _ in range(len(cp.lines)):  # 빠른 차가 한 스텝에 선을 여러 개 지나도 모두 센다
                if cp.done:
                    break
                a, b = cp.lines[cp.next]
                if not _cross(q, p, a, b):
                    break
                cp.count += 1
                cp.last_time = t
                if cp.loop:
                    if cp.next == 0:
                        cp.laps += 1
                        cp.lap_times.append(t - cp.lap_start)
                        cp.lap_start = t
                    cp.next = (cp.next + 1) % len(cp.lines)
                else:
                    cp.next += 1
                    if cp.next >= len(cp.lines):
                        cp.done = True
                        cp.laps = 1
                        cp.lap_times.append(t)
            if cp.need_progress:
                if cp.track is not None:
                    from evolab.generators.track import progress_along

                    s_now, cp.hint = progress_along(cp.track, p[0], p[1], cp.hint)
                    d = s_now - cp.s_prev
                    half = cp.track.length / 2.0
                    if d > half:
                        d -= cp.track.length
                    elif d < -half:
                        d += cp.track.length
                    cp.progress += d
                    cp.s_prev = s_now
                else:
                    cp.progress = float(cp.count)

    def _apply_forces(self) -> None:
        """모은 추진기 힘(물체 좌표)과 바람 항력(세계 좌표, 무게중심)을 물체마다 한 번에 넣는다."""
        acc = self._acc
        drag = _NO_DRAG
        if self._drag_slots:
            wx, wy = self.wind
            drag = [(0.0, 0.0)] * len(acc)
            for slot, k in self._drag_slots:
                st = self._bstate(self._force_bodies[slot][0])
                drag[slot] = (k * (wx - st[3]), k * (wy - st[4]))
        for slot, (bid, body, raw) in enumerate(self._force_raw):
            a = acc[slot]
            fx, fy, tq = a
            dx, dy = drag[slot] if drag is not _NO_DRAG else (0.0, 0.0)
            if fx == 0.0 and fy == 0.0 and tq == 0.0 and dx == 0.0 and dy == 0.0:
                continue
            ang = self._bstate(bid)[2]
            c, s_ = math.cos(ang), math.sin(ang)
            if raw is not None:  # 공개 API와 같은 C 함수(v6 D7)
                _CP_SET_FORCE(raw, (fx * c - fy * s_ + dx, fx * s_ + fy * c + dy))
                _CP_SET_TORQUE(raw, tq)
            else:
                body.force = (fx * c - fy * s_ + dx, fx * s_ + fy * c + dy)
                body.torque = tq
            a[0] = a[1] = a[2] = 0.0

    def _apply_actuators(self) -> None:
        u = self._u
        applied = list(u)
        for i in self._thruster_channels:
            applied[i] = 0.0
        for jp in self._jumps:
            # 점프(설계 4.3): 출력 0.5 이상 + 땅 접촉 + 냉각이면 뛰는 방향 속도를 speed로 맞춘다(더 빠르면 그대로).
            # 실제 출력은 뛴 스텝만 1(에너지 블록·행동 분포가 뛴 횟수를 센다).
            i = jp.channel_index
            applied[i] = 0.0
            if u[i] >= 0.5 and self._jump_allowed(jp):
                raw = jp.raw
                v = _CP_GET_VELOCITY(raw) if raw is not None else jp.body.velocity
                vd = v.x * jp.dx + v.y * jp.dy
                if vd < jp.speed:
                    if raw is not None:  # 공개 API와 같은 C 함수(v6 D7)
                        _CP_SET_VELOCITY(raw, (v.x + (jp.speed - vd) * jp.dx, v.y + (jp.speed - vd) * jp.dy))
                    else:
                        jp.body.velocity = (v.x + (jp.speed - vd) * jp.dx, v.y + (jp.speed - vd) * jp.dy)
                    self._cache.pop(jp.bid, None)
                jp.last = self.steps
                applied[i] = 1.0
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
        # Chipmunk SimpleMotor는 (b 각속도 - a 각속도) = -rate로 맞춘다. +출력이 b를 a에 대해 반시계로 돌게 부호를 바꾼다.
        if self._motor_raw is not None:  # 공개 API(SimpleMotor.rate)와 같은 C 함수(v6 D7)
            for raw, ci, max_speed in self._motor_raw:
                _CP_MOTOR_SET_RATE(raw, -u[ci] * max_speed)
        else:
            for mo in self._motors:
                mo.constraint.rate = -u[mo.channel_index] * mo.max_speed
        for v in self._vehicles:
            # 자전거 모형: 옆미끄럼을 grip만큼 없애고, 회전 속도는 앞뒤 속도와 조향각으로 정한다.
            _x, _y, ang, vx, vy, _w = self._bstate(v.bid)
            c, s_ = math.cos(ang), math.sin(ang)
            fwd = vx * c + vy * s_
            lat = (-vx * s_ + vy * c) * (1.0 - v.grip)
            raw = v.raw
            if raw is not None:  # 공개 API와 같은 C 함수(v6 D7)
                _CP_SET_VELOCITY(raw, (fwd * c - lat * s_, fwd * s_ + lat * c))
            else:
                v.body.velocity = (fwd * c - lat * s_, fwd * s_ + lat * c)
            omega = fwd * math.tan(u[v.ch_steer] * v.steer) / v.wheelbase
            if v.lat_max > 0.0 and abs(omega * fwd) > v.lat_max:
                omega = math.copysign(v.lat_max / abs(fwd), omega)  # 횡가속도 한계: 빠르면 덜 꺾인다
            if raw is not None:
                _CP_SET_ANGULAR_VELOCITY(raw, omega)
            else:
                v.body.angular_velocity = omega
            self._cache.pop(v.bid, None)
            acc = self._acc[v.slot]
            target = fwd * v.decay + u[v.ch_throttle] * v.accel * v.gain
            acc[0] += (target - fwd) * v.mass / self.dt
        self._u_applied = applied
        sums = self._abs_sum
        for i, x in enumerate(applied):
            sums[i] += x if x >= 0.0 else -x

    # ------------------------------------------------------------------
    # 접촉
    # ------------------------------------------------------------------

    def _setup_contacts(self) -> None:
        self._touching: dict[str, dict[pymunk.Shape, int]] = {b: {} for b in self.bodies}
        for b in self._absent:  # 없는 물체(v2.1)는 빈 칸: 그 물체의 접촉 채널(contact(), 접촉 센서·블록)은 늘 거짓
            self._touching[b] = {}
        self.space.on_collision(None, None, begin=self._on_begin, separate=self._on_separate)
        self._contacts: dict[tuple[str, tuple[str, ...]], _ContactChannel] = {}
        self._debounce = float(self.env.world.contact_debounce)
        self._contacts_dirty = True
        self._cache: dict[str, tuple[float, float, float, float, float, float]] = {}
        self._pos_cache: dict[str, tuple[float, float]] = {}

    def _setup_c_paths(self) -> None:
        """C 함수를 바로 부를 포인터를 한 번 모은다(v6 설계 4.1, D6·D7). self.bodies는 없는 물체(v2.1)를 뺀 것이다.

        _CP_FAST가 아니면(pymunk 내부가 바뀐 판) 포인터가 모두 None이고 모든 곳이 공개 API 길을 지난다(결과는 같다:
        공개 API가 같은 C 함수를 같은 인수로 부른 뒤 Vec2d로 감쌀 뿐이다). 물체·관절은 Episode가 붙잡고 있으므로 포인터가
        가리키는 C 메모리는 Episode가 사는 동안 살아 있다."""
        self._fast = _CP_FAST
        self._raw_space: Any = None
        """cpSpace 포인터. 첫 스텝(공개 Space.step) 뒤에 채운다(_first_space_step)."""
        self._raw: dict[str, Any] = {bid: _body_c(b) for bid, b in self.bodies.items()}
        self._force_raw: list[tuple[str, pymunk.Body, Any]] = [(bid, b, self._raw[bid]) for bid, b in self._force_bodies]
        # 포인터가 하나라도 없으면(pymunk 내부 이름이 바뀐 판) 경계 검사는 _bpos 길(같은 값)로(v6 통합: 자체 점검 T18이 찾음)
        agent_raw = [self._raw[b] for b in self._agent_ids]
        self._agent_raw: list[Any] | None = agent_raw if self._fast and all(r is not None for r in agent_raw) else None
        motor_raw = [_constraint_c(mo.constraint) if self._fast else None for mo in self._motors]
        self._motor_raw: list[tuple[Any, int, float]] | None = (
            [(raw, mo.channel_index, mo.max_speed) for raw, mo in zip(motor_raw, self._motors, strict=True)]
            if all(raw is not None for raw in motor_raw) else None)
        for v in self._vehicles:
            v.raw = self._raw[v.bid]
        for jp in self._jumps:
            jp.raw = self._raw[jp.bid]
        # 공기역학(v6 D12): 물체마다 (물체, id, 포인터, 1/질량, 1/관성, 부품마다 (qk, slope, cd90, stall, friction, induced,
        # 변 목록)). 물체 질량·관성은 모양을 모두 단 뒤 바뀌지 않는다(_apply_aero가 스텝마다 나누던 값과 같다).
        self._aero_fast: list[tuple[pymunk.Body, str, Any, float, float, list[tuple[Any, ...]]]] = [
            (body, bid, self._raw[bid], 1.0 / body.mass, 1.0 / body.moment,
             [(p.qk, p.slope, p.cd90, p.stall, p.friction, p.induced, p.edges) for p in parts])
            for body, bid, parts in self._aero_bodies
        ]

    def _first_space_step(self) -> None:
        """공개 Space.step 한 번(v6 설계 4.2, D8). 첫 스텝에 pymunk가 동적 물체의 질량·관성을 검사하고 검사 목록을 비운다.

        그 뒤로 Space.step의 파이썬 부분(검사 목록, 잠금, 스텝 중 더하기·빼기 목록, 빈 모양 목록, 스텝 뒤 콜백)은 이 런타임에서
        늘 빈 일이라(tests/test_speed_runtime.py T7) 둘째 스텝부터는 그 안의 cpSpaceStep(같은 C 함수, 같은 인수)을 바로 부른다.
        전제: Episode는 만든 뒤 공간에 물체·모양·관절을 더하거나 빼지 않고, 접촉 콜백(_on_begin·_on_separate)은 공간을 바꾸지
        않는다. 그런 코드가 생기면(예: v5 대전의 물체 추가) 그 Episode는 _raw_space를 None으로 되돌려 공개 Space.step을 쓴다.
        _CP_FAST가 아니었으면(만들 때 정함) 늘 공개 Space.step이다."""
        space = self.space
        space.step(self.dt)
        # pymunk 내부 이름이 바뀐 판이면(속성이 없으면) 공개 Space.step을 계속 쓴다.
        if self._fast and getattr(space, "threaded", True) is False and getattr(space, "_bodies_to_check", None) == set():
            self._raw_space = _space_c(space)

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
                hit = False
                if touching:  # = any(self._matches(s, ch.selectors) for s in touching), 모양마다 한 번만 판정(v6 D11)
                    memo = ch.memo
                    for s in touching:
                        m_ = memo.get(s)
                        if m_ is None:
                            m_ = memo[s] = self._matches(s, ch.selectors)
                        if m_:
                            hit = True
                            break
                ch.raw = hit
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
            raw = self._raw[body_id]
            v = _CP_GET_POSITION(raw) if raw is not None else self.bodies[body_id].position  # 같은 C 함수(v6 D6)
            p = (v.x, v.y)
            self._pos_cache[body_id] = p
        return p

    def _bstate(self, body_id: str) -> tuple[float, float, float, float, float, float]:
        """(x, y, 각도, vx, vy, 각속도). 스텝마다 물체당 한 번만 pymunk에서 읽는다."""
        c = self._cache.get(body_id)
        if c is None:
            raw = self._raw[body_id]
            if raw is not None:  # 공개 API(Body.position 등)와 같은 C 함수, Vec2d 없이(v6 D6)
                p = _CP_GET_POSITION(raw)
                v = _CP_GET_VELOCITY(raw)
                c = (p.x, p.y, _CP_GET_ANGLE(raw), v.x, v.y, _CP_GET_ANGULAR_VELOCITY(raw))
            else:
                b = self.bodies[body_id]
                p = b.position
                v = b.velocity
                c = (p.x, p.y, b.angle, v.x, v.y, b.angular_velocity)
            self._cache[body_id] = c
        return c

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
        if self._absent and (getattr(s, "body", None) in self._absent
                             or (isinstance(s, m.JointAngle) and s.joint in self._absent_joints)):
            return self._absent_sensor_fn(s)
        if isinstance(s, m.BodyState):
            names = [f"{s.id}.{f}" for f in s.fields]
            scales = [s.scale.get(f, sy.DEFAULT_STATE_SCALE[f]) for f in s.fields]
            return names, scales, _body_state_fn(self._bstate, s.body, tuple(s.fields), s.frame)
        if isinstance(s, m.Target):
            tgt, body, frame = s.target, s.body, s.frame

            def target() -> tuple[float, ...]:
                st = self._bstate(body)
                x, y, a = st[0], st[1], st[2]
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
        if isinstance(s, m.Rays):
            return self._rays_fn(s)
        raise EpisodeBuildError(S.SPEC_NOT_YET.format(milestone="M4"))

    def _absent_sensor_fn(self, s: m.Sensor) -> tuple[list[str], list[float], Callable[[], tuple[float, ...]]]:
        """없는 물체·관절에 단 센서(v2.1, 설계 D35): 모든 채널이 늘 0(레이도 0: 「멀리 아무것도 없음」 1과 가른다)."""
        names = [c.name for c in self.spec.sensor_channels.values() if c.owner == s.id]
        zeros = (0.0,) * len(names)
        fn: Callable[[], tuple[float, ...]] = lambda: zeros  # noqa: E731
        if isinstance(s, m.Rays):
            # 레이 그림 목록의 자리는 남긴다(에디터는 레이 센서 순서로 이 목록을 읽는다). ray_view는 없는 물체를 건너뛴다.
            self._ray_views.append((s.body, (0.0, 0.0), [0.0] * len(names), 0.0, fn))
        return names, [1.0] * len(names), fn

    def _ray_filter(self, selectors: Sequence[str]) -> pymunk.ShapeFilter:
        """selectors에 맞는 모양에만 범주 비트를 하나 붙이고, 그 비트만 보는 질의 필터를 돌려준다.

        다른 모양의 범주는 기본 비트 하나로 바꾼다. 충돌 마스크는 그대로(모두)라 부딪힘은 바뀌지 않는다.
        """
        key = tuple(selectors)
        if not hasattr(self, "_ray_bits"):
            self._ray_bits: dict[tuple[str, ...], int] = {}
            self._first_ray_filters()
        bit = self._ray_bits.get(key)
        if bit is None:
            bit = 1 << (1 + len(self._ray_bits))
            if bit >= 1 << 31:
                raise EpisodeBuildError(S.RT_TOO_MANY_RAY_TARGETS)
            self._ray_bits[key] = bit
            for shape in self._shape_info:
                if self._matches(shape, key):
                    f = shape.filter
                    shape.filter = pymunk.ShapeFilter(group=f.group, categories=f.categories | bit, mask=f.mask)
        return pymunk.ShapeFilter(group=AGENT_GROUP, mask=bit)  # 에이전트 모양(자기 몸)은 맞지 않는다

    def _first_ray_filters(self) -> None:
        """첫 _ray_filter 호출에서 모든 모양의 범주를 한 번만 쓴다(v6 설계 4.6, D13a).

        옛 길은 모든 모양을 범주 1로 쓴 뒤 레이 꼬리표 묶음마다 맞는 모양을 다시 썼다(맞는 묶음 수 + 1번). 여기서는 레이
        센서가 _ray_filter를 부를 순서(센서 선언 순서, 없는 물체에 단 레이 센서는 부르지 않음)로 꼬리표 묶음에 같은 비트
        1 << (1 + k)를 미리 정하고, 모양마다 범주 = 1 | (맞는 비트들)을 한 번 쓴다. 마지막 필터 값(그룹·범주·마스크)이 옛
        길과 같다(cpShapeSetFilter의 다른 일은 물체 깨우기뿐이고 이 공간은 잠들지 않는다). 31비트를 넘는 묶음은 여기서 비트를
        정하지 않아, 그 묶음을 부르는 레이 센서에서 옛 길처럼 RT_TOO_MANY_RAY_TARGETS가 난다. 미리 정하지 않은 묶음(센서가 아닌
        호출)은 _ray_filter가 옛 길대로 다음 비트를 붙인다."""
        keys: list[tuple[str, ...]] = []
        for sensor in self.env.sensors:
            if isinstance(sensor, m.Rays) and not (self._absent and sensor.body in self._absent):
                k = tuple(sensor.hits)
                if k not in keys:
                    keys.append(k)
        bits = self._ray_bits
        for k in keys:
            bit = 1 << (1 + len(bits))
            if bit >= 1 << 31:
                break
            bits[k] = bit
        plan = list(bits.items())
        matches = self._matches
        for shape in self._shape_info:
            cats = 1
            for k, bit in plan:
                if matches(shape, k):
                    cats |= bit
            f = shape.filter
            shape.filter = pymunk.ShapeFilter(group=f.group, categories=cats, mask=f.mask)

    def _rays_fn(self, s: m.Rays) -> tuple[list[str], list[float], Callable[[], tuple[float, ...]]]:
        n = s.count
        spread = self._num(s.spread, "agent")
        center = self._num(s.center, "agent")
        length = max(1e-6, _clamp(self._num(s.length, "agent"), 1e-6, FIELD_LIMIT))
        ox, oy = self._vec(s.origin, "agent")
        angles = [center] if n == 1 else [center - spread / 2.0 + spread * i / (n - 1) for i in range(n)]
        filt = self._ray_filter(s.hits)
        bid = s.body
        space = self.space
        memo: list[Any] = [-1, ()]
        # 에이전트 물체는 질의 필터(에이전트 그룹)로 이미 빠진다. 세계 물체에 단 레이는 자기 모양을 직접 건너뛴다.
        own = None if bid in self._agent_ids else set(self.bodies[bid].shapes)
        # 첫 맞음 질의는 C 함수를 바로 부른다(_segment_first_c, 결과는 공개 API와 비트 단위로 같다). 레이마다 같은 식
        # (각도 a + ang의 cos·sin)으로 끝점을 구하는 것은 그대로다.
        fast = _segment_first_c(space, filt) if own is None else None
        cos, sin = math.cos, math.sin
        bstate = self._bstate
        if fast is not None:
            query, raw, cfilt, _owner, info = fast

        def rays() -> tuple[float, ...]:
            if memo[0] == self.steps:
                return memo[1]
            st = bstate(bid)
            x, y, a = st[0], st[1], st[2]
            c, si = cos(a), sin(a)
            px, py = x + ox * c - oy * si, y + ox * si + oy * c
            start = (px, py)
            out: list[float] = []
            if fast is not None:
                for ang in angles:
                    th = a + ang
                    query(raw, start, (px + length * cos(th), py + length * sin(th)), 0.0, cfilt, info)
                    out.append(info.alpha)
            else:
                for ang in angles:
                    th = a + ang
                    end = (px + length * cos(th), py + length * sin(th))
                    if own is None:
                        hit = space.segment_query_first(start, end, 0.0, filt)
                        out.append(hit.alpha if hit is not None else 1.0)
                    else:
                        alphas = [h.alpha for h in space.segment_query(start, end, 0.0, filt) if h.shape not in own]
                        out.append(min(alphas) if alphas else 1.0)
            memo[0], memo[1] = self.steps, tuple(out)
            return memo[1]

        self._ray_views.append((bid, (ox, oy), angles, length, rays))
        return [f"{s.id}.r{i}" for i in range(n)], [1.0] * n, rays

    def ray_view(self) -> list[tuple[geo.Point, geo.Point, float]]:
        """화면 표시용: 레이마다 (시작점, 끝점(맞은 곳 또는 최대 길이), 맞은 비율 0..1)."""
        out: list[tuple[geo.Point, geo.Point, float]] = []
        for bid, (ox, oy), angles, length, fn in self._ray_views:
            if bid in self._absent:
                continue
            x, y, a, *_ = self._bstate(bid)
            c, si = math.cos(a), math.sin(a)
            px, py = x + ox * c - oy * si, y + ox * si + oy * c
            for ang, alpha in zip(angles, fn(), strict=True):
                th = a + ang
                d = alpha * length
                out.append(((px, py), (px + d * math.cos(th), py + d * math.sin(th)), alpha))
        return out

    def _build_sensors(self) -> None:
        self._ray_views: list[tuple[str, geo.Point, list[float], float, Callable[[], tuple[float, ...]]]] = []
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
                getters[f"{bid}.{f}"] = _state_getter(self._bstate, bid, f)
        if self._absent:
            # 검사가 없어질 수 있는 물체의 몸 상태·거리·영역을 수식에서 막으므로(SPEC_EXISTS_REF) 보통 쓰이지 않는다. 쓰여도
            # 멈추지 않게 0을 둔다(접촉 질의는 위에서 빈 접촉 칸이라 늘 거짓).
            for bid in self._absent:
                for f in sy.BODY_STATE:
                    getters[f"{bid}.{f}"] = (lambda: 0.0)
            for q in self.spec.queries.values():
                if q.func != "contact" and (q.a in self._absent or q.b in self._absent):
                    getters[q.channel] = (lambda: False) if q.func == "inside" else (lambda: 0.0)
        for name, g in raw.items():
            ch = self.spec.sensor_channels[name]
            getters[name] = (lambda g=g: g() > 0.5) if ch.type == "bool" else g
        for i, name in enumerate(self._u_names):
            getters[name] = (lambda i=i: self._u_applied[i])
        for tid in self._tanks:
            getters[tid] = (lambda t=tid: self._tanks[t].remaining / self._tanks[t].capacity)
        getters["time"] = lambda: self.time
        for cp in self._cps:
            getters[f"{cp.id}.count"] = (lambda cp=cp: float(cp.count))
            getters[f"{cp.id}.laps"] = (lambda cp=cp: float(cp.laps))
            getters[f"{cp.id}.since"] = (lambda cp=cp: self.time - cp.last_time)
            getters[f"{cp.id}.progress"] = (lambda cp=cp: cp.progress)
            cp.need_progress = f"{cp.id}.progress" in needed
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

    def _update_events(self) -> None:
        """사건마다 v3.1 _fire와 같은 일을 한 루프로(v6 설계 4.4, D10): 조건 → 연속 수·발생·횟수·마지막 시각 → 값 사전.
        사건 조건은 앞 사건이 이 스텝에 써 넣은 값을 본다(순서 그대로). 내장 사건 둘의 조건은 값 사전을 읽지 않으므로 먼저
        계산한다: timeout = 스텝 수 ≥ 상한, out_of_bounds = 에이전트 물체 하나라도 경계 밖(NaN 포함, 첫 이탈에서 멈춤)."""
        x0, y0, x1, y1 = self.bounds
        oob = False
        agent_raw = self._agent_raw
        if agent_raw is not None:  # 위치는 공개 API와 같은 C 함수로(v6 D6)
            for raw in agent_raw:
                p = _CP_GET_POSITION(raw)
                px, py = p.x, p.y
                if not (x0 <= px <= x1 and y0 <= py <= y1):  # NaN도 여기서 걸린다
                    oob = True
                    break
        else:
            for bid in self._agent_ids:
                px, py = self._bpos(bid)
                if not (x0 <= px <= x1 and y0 <= py <= y1):
                    oob = True
                    break
        builtin = (self.steps >= self.max_steps, oob)
        v = self._values
        t = self.time
        k = 0
        for st, cond, count_key, since_key in self._fire_plan:
            if cond is None:
                c = builtin[k]
                k += 1
            else:
                c = cond(v)  # if c: = bool(c)
            if c:
                n = st.consec + 1
                st.consec = n
                if n == st.hold_steps:
                    st.occurred = True
                    st.count += 1
                    st.last_time = t
            else:
                st.consec = 0
            v[st.id] = st.occurred
            if count_key is not None:
                v[count_key] = float(st.count)
            if since_key is not None:
                last = st.last_time
                v[since_key] = t - last if last is not None else t

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

    def _block_value(self, st: _BlockState) -> float:
        b = st.block
        if isinstance(b, m.EventBlock):
            ev = self._values.get(b.event if b.value == "occurred" else f"{b.event}.count", 0.0)
            return float(ev)
        if isinstance(b, m.DistanceBlock):
            return self._distance(b.body, b.target) if b.agg == "final" or st.acc == math.inf else st.acc
        if isinstance(b, m.ProgressBlock):
            d = self._num(b.direction, "agent")
            raw = self._raw[b.body]  # 끝값도 공개 API와 같은 C 함수로(v6 D6)
            p = _CP_GET_POSITION(raw) if raw is not None else self.bodies[b.body].position
            s = self._start_pos[b.body]
            return (p.x - s[0]) * math.cos(d) + (p.y - s[1]) * math.sin(d)
        if isinstance(b, m.SurvivalBlock):
            return self.time
        if isinstance(b, m.SpeedBlock):
            raw = self._raw[b.body]
            v = _CP_GET_VELOCITY(raw) if raw is not None else self.bodies[b.body].velocity
            return math.hypot(v.x, v.y)
        if isinstance(b, m.TiltBlock):
            if b.agg == "mean" and st.count:
                return st.acc / st.count
            raw = self._raw[b.body]
            return abs(geo.wrap_angle(_CP_GET_ANGLE(raw) if raw is not None else self.bodies[b.body].angle))
        if isinstance(b, m.FuelBlock):
            t = self._tanks[b.tank]
            return (t.capacity - t.remaining) / t.capacity
        if isinstance(b, m.CheckpointBlock):
            cp = self._cp_by_id[b.checkpoints]
            if b.kind == "checkpoints":
                return float(cp.count)
            return sum(cp.lap_times) / len(cp.lap_times) if cp.lap_times else 0.0
        if isinstance(b, m.ExprBlock):
            if b.agg == "mean":
                return st.acc / st.count if st.count else 0.0
            if b.agg in ("min", "max") and not math.isfinite(st.acc):
                return 0.0
            return st.acc
        return st.acc

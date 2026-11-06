"""수식 언어 하나로 세 가지를 쓴다.

- 조립 시점 수치: 환경 파일에서 '='로 시작하는 문자열 (호출하는 쪽이 '='를 떼고 넘긴다).
- 매 스텝 사건·종료 조건.
- 종료 시점 점수 수식과 'expr' 점수 블록.

파이썬 eval은 쓰지 않는다. ``ast.parse(mode='eval')``로 읽고 허용 목록에 있는 구문만
받아서, 한 번 컴파일하면 파이썬 클로저를 겹친 함수가 된다. 평가는 에피소드마다 스칼라
(파이썬 float/bool, math 모듈)로 한다. 에피소드가 매 물리 스텝마다 자기 조건을 평가하므로
numpy보다 빠르다.

문법
    숫자(정수도 float로), True/False, 상수 ``pi``, 점 이름(``hull.speed``, ``p.leg_len``,
    ``gen.terrain.pad.x``; 기호표와 값 사전에서 통째로 한 키다). ``__``가 든 이름은 거부.
    ``pi``는 기호표에 같은 이름이 있어도 언제나 상수다.
    ``+ - * / **``, 단항 ``- +``, 비교 ``< <= > >= == !=``(``0 < x < 1`` 같은 연쇄 포함),
    ``and or not``, ``a if 조건 else b``.
    함수 ``min max``(인자 2..8개) ``abs sqrt exp log clamp(x, lo, hi) sin cos atan2(y, x)``.
    질의 함수 ``contact(a, b) inside(a, b) dist(a, b)``: 인자는 점 이름(부품 id)이어야 하고,
    컴파일 때 ``query`` 해석기가 (채널 이름, Sym)으로 바꾼다. 평가 때는 그 채널을 값 사전에서 읽는다.

형
    ``and or not``과 if 조건에는 참/거짓이 필요하다. 비교의 양쪽은 숫자여야 한다.
    산술과 함수 인자에 쓴 참/거짓은 1.0/0.0으로 바뀐다(``success * 100``).
    ``want='bool'``이면 결과가 참/거짓이어야 한다. ``want='num'``이면 참/거짓 결과도 받고,
    그때 ``Compiled.type``은 'num'이며 호출하면 1.0/0.0을 돌려준다.

단계
    기호마다 단계(build < step < end)가 있다. ``max_phase``보다 늦은 기호를 쓰면 오류다.
    ``Compiled.phase``는 쓴 기호 가운데 가장 늦은 단계이고, 상수식이면 'build'.

안전
    길이 1000자, 식 요소 300개, 중첩 100단계까지. 깊은 중첩도 RecursionError가 아니라 ExprError다.
    평가는 유한한 입력에 대해 예외를 내지 않는다.
    0으로 나누기 → 0.0, sqrt(음수) → 0.0, log(1e-12 이하) → log(1e-12), exp 인자는 50까지,
    ``**`` 결과는 ±1e12로 자르고 정의되지 않으면(음수의 분수 거듭제곱, 0의 음수 거듭제곱) 0.0,
    sin/cos(±inf) → nan. 최종 결과(숫자)는 nan → 0.0, ±1e12 밖은 ±1e12로 자른다.
    and/or는 단락 평가를 하지만 언제나 bool을 돌려준다.

값 사전
    ``Compiled.names``에 든 키가 모두 있어야 한다(없으면 KeyError; 단락 평가 때문에 안 읽힐 수도 있다).
    숫자 기호의 값은 float, bool, float 범위 안의 int(np.float64 포함). int끼리 더하고 빼고 곱한
    결과는 float로 계산하므로 큰 정수가 생겨 예외가 나는 일은 없다. 참/거짓 기호의 값은
    참·거짓으로 해석할 수 있는 무엇이든 된다. ``names``는 frozenset이라 순회 순서가 해시에
    따라 달라지므로, 순서가 필요하면 정렬해서 쓴다.

열 번호
    ``ExprError.col``은 넘긴 ``source`` 문자열 안의 0부터 세는 글자 위치다('='를 뗀 뒤 기준).

성능
    ``Compiled.__call__``이 공개 규격이고, 더 빠른 길로 ``Compiled.fn``(같은 함수, 메서드 호출
    한 단계가 없음)을 쓸 수 있다. 클로저라서 Compiled는 pickle되지 않는다. 프로세스마다 컴파일한다.
    compile_expr는 여러 스레드에서 동시에 불러도 된다(파싱 경고를 끄는 부분만 잠금으로 한 번에 하나).
"""

from __future__ import annotations

import ast
import difflib
import math
import operator
import re
import threading
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from evolab.app import strings as S

Phase = Literal["build", "step", "end"]
ValueType = Literal["num", "bool"]
Values = Mapping[str, float | bool]
_Fn = Callable[[Values], Any]

PHASES: Final[tuple[Phase, ...]] = ("build", "step", "end")
PHASE_RANK: Final[dict[str, int]] = {"build": 0, "step": 1, "end": 2}

MAX_LENGTH: Final = 1000
MAX_NODES: Final = 300
MAX_DEPTH: Final = 100
LIMIT: Final = 1e12
EXP_CAP: Final = 50.0
LOG_FLOOR: Final = math.log(1e-12)

# 이름: (최소 인자 수, 최대 인자 수). 사전 순서가 오류 문구에 쓰는 순서다.
FUNCTIONS: Final[dict[str, tuple[int, int]]] = {
    "min": (2, 8),
    "max": (2, 8),
    "abs": (1, 1),
    "sqrt": (1, 1),
    "exp": (1, 1),
    "log": (1, 1),
    "clamp": (3, 3),
    "sin": (1, 1),
    "cos": (1, 1),
    "atan2": (2, 2),
}
QUERY_FUNCTIONS: Final[tuple[str, ...]] = ("contact", "inside", "dist")
QUERY_ARITY: Final = 2


@dataclass(frozen=True)
class Sym:
    """기호표 항목: 값의 형과 그 값이 정해지는 단계."""

    type: ValueType = "num"
    phase: Phase = "step"


QueryResolver = Callable[[str, tuple[str, ...]], tuple[str, Sym]]


class ExprError(ValueError):
    """수식 오류. ``message``는 strings.py의 한국어 문구, ``col``은 0부터 세는 글자 위치."""

    def __init__(self, message: str, col: int | None = None) -> None:
        super().__init__(message, col)
        self.message = message
        self.col = col

    def __str__(self) -> str:
        if self.col is None:
            return self.message
        return S.EXPR_ERR_AT.format(message=self.message, column=self.col + 1)


@dataclass(frozen=True, slots=True)
class Compiled:
    """컴파일된 수식. 값 사전을 받아 float 또는 bool을 돌려준다."""

    source: str
    type: ValueType
    phase: Phase
    names: frozenset[str]
    fn: Callable[[Values], float | bool] = field(repr=False, compare=False)

    def __call__(self, values: Values) -> float | bool:
        return self.fn(values)


def is_formula(value: object) -> bool:
    """환경 파일의 값이 수식('='로 시작하는 문자열)인가."""
    return isinstance(value, str) and value.startswith("=")


def formula_body(value: str) -> str:
    """앞의 '='와 앞뒤 공백을 뗀 수식 본문."""
    text = value.strip()
    if text.startswith("="):
        text = text[1:]
    return text.strip()


# ---------------------------------------------------------------------------
# 평가 도우미 (예외 없이 값을 돌려준다)


def _pow(a: Any, b: Any) -> float:
    try:
        x = float(a)
        y = float(b)
    except OverflowError:  # float 범위 밖의 int (규격 밖 입력)
        return LIMIT
    try:
        r: float | complex = x**y
    except ZeroDivisionError:  # 0.0 ** 음수
        return 0.0
    except OverflowError:
        if x < 0.0 and not y.is_integer():  # 음수의 분수 거듭제곱: 크기와 상관없이 정의되지 않음
            return 0.0
        return -LIMIT if (x < 0.0 and y % 2.0 == 1.0) else LIMIT
    if isinstance(r, complex):  # 음수의 분수 거듭제곱
        return 0.0
    if r > LIMIT:
        return LIMIT
    if r < -LIMIT:
        return -LIMIT
    return r


def _finite(r: Any) -> float:
    """최종 숫자 결과: nan → 0.0, ±1e12 밖은 자르기, 파이썬 float로."""
    if r != r:
        return 0.0
    if r > LIMIT:
        return LIMIT
    if r < -LIMIT:
        return -LIMIT
    return float(r)


_EMPTY: Final[dict[str, float | bool]] = {}


# ---------------------------------------------------------------------------
# 컴파일 중간 표현


@dataclass(slots=True)
class _Node:
    fn: _Fn
    type: ValueType
    exact: bool = False  # 참/거짓 형이고 fn이 언제나 파이썬 bool을 돌려준다
    const: bool = False
    value: Any = None
    key: str | None = None  # 값 사전에서 그대로 읽는 노드


def _const_num(value: float) -> _Node:
    def f(v: Values) -> float:
        return value

    return _Node(f, "num", const=True, value=value)


def _const_bool(value: bool) -> _Node:
    def f(v: Values) -> bool:
        return value

    return _Node(f, "bool", exact=True, const=True, value=value)


def _read(key: str, typ: ValueType) -> _Node:
    return _Node(operator.itemgetter(key), typ, key=key)


def _fold(node: _Node) -> _Node:
    """자식이 모두 상수인 노드를 컴파일 때 한 번 계산한다 (실행 때와 같은 함수라 결과도 같다)."""
    r = node.fn(_EMPTY)
    if node.type == "bool":
        return _const_bool(True if r else False)
    return _const_num(r)


def _as_num(node: _Node) -> _Node:
    """참/거짓을 산술에 쓸 때 1.0/0.0으로."""
    if node.type == "num":
        return node
    if node.const:
        return _const_num(1.0 if node.value else 0.0)
    if node.key is not None:
        k = node.key

        def fk(v: Values) -> float:
            return 1.0 if v[k] else 0.0

        return _Node(fk, "num")
    g = node.fn

    def f(v: Values) -> float:
        return 1.0 if g(v) else 0.0

    return _Node(f, "num")


# ---- 비교 -------------------------------------------------------------------

_FLIP: Final[dict[type, type]] = {
    ast.Lt: ast.Gt,
    ast.LtE: ast.GtE,
    ast.Gt: ast.Lt,
    ast.GtE: ast.LtE,
    ast.Eq: ast.Eq,
    ast.NotEq: ast.NotEq,
}
_CMP_FUNC: Final[dict[type, Callable[[Any, Any], Any]]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_CMP_SYMBOL: Final[dict[type, str]] = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Eq: "==",
    ast.NotEq: "!=",
}


def _cmp_key_const(op: type, k: str, c: Any) -> _Fn:
    if op is ast.Lt:

        def f(v: Values) -> bool:
            return True if v[k] < c else False

    elif op is ast.LtE:

        def f(v: Values) -> bool:
            return True if v[k] <= c else False

    elif op is ast.Gt:

        def f(v: Values) -> bool:
            return True if v[k] > c else False

    elif op is ast.GtE:

        def f(v: Values) -> bool:
            return True if v[k] >= c else False

    elif op is ast.Eq:

        def f(v: Values) -> bool:
            return True if v[k] == c else False

    else:

        def f(v: Values) -> bool:
            return True if v[k] != c else False

    return f


def _cmp_fn_const(op: type, g: _Fn, c: Any) -> _Fn:
    if op is ast.Lt:

        def f(v: Values) -> bool:
            return True if g(v) < c else False

    elif op is ast.LtE:

        def f(v: Values) -> bool:
            return True if g(v) <= c else False

    elif op is ast.Gt:

        def f(v: Values) -> bool:
            return True if g(v) > c else False

    elif op is ast.GtE:

        def f(v: Values) -> bool:
            return True if g(v) >= c else False

    elif op is ast.Eq:

        def f(v: Values) -> bool:
            return True if g(v) == c else False

    else:

        def f(v: Values) -> bool:
            return True if g(v) != c else False

    return f


def _cmp_fn_fn(op: type, g: _Fn, h: _Fn) -> _Fn:
    if op is ast.Lt:

        def f(v: Values) -> bool:
            return True if g(v) < h(v) else False

    elif op is ast.LtE:

        def f(v: Values) -> bool:
            return True if g(v) <= h(v) else False

    elif op is ast.Gt:

        def f(v: Values) -> bool:
            return True if g(v) > h(v) else False

    elif op is ast.GtE:

        def f(v: Values) -> bool:
            return True if g(v) >= h(v) else False

    elif op is ast.Eq:

        def f(v: Values) -> bool:
            return True if g(v) == h(v) else False

    else:

        def f(v: Values) -> bool:
            return True if g(v) != h(v) else False

    return f


def _range(lo: Any, op1: type, g: _Fn, op2: type, hi: Any) -> _Fn:
    """``lo < x < hi`` 꼴 (op1, op2는 < 또는 <=)."""
    if op1 is ast.Lt and op2 is ast.Lt:

        def f(v: Values) -> bool:
            return True if lo < g(v) < hi else False

    elif op1 is ast.Lt:

        def f(v: Values) -> bool:
            return True if lo < g(v) <= hi else False

    elif op2 is ast.Lt:

        def f(v: Values) -> bool:
            return True if lo <= g(v) < hi else False

    else:

        def f(v: Values) -> bool:
            return True if lo <= g(v) <= hi else False

    return f


def _chain(fns: list[_Fn], ops: list[type]) -> _Fn:
    first = fns[0]
    pairs = tuple(zip([_CMP_FUNC[o] for o in ops], fns[1:], strict=True))

    def f(v: Values) -> bool:
        a = first(v)
        for op, h in pairs:
            b = h(v)
            if not op(a, b):
                return False
            a = b
        return True

    return f


# ---- 산술 -------------------------------------------------------------------


def _arith(op: type, a: _Node, b: _Node) -> _Fn:
    # 두 쪽 모두 값에서 오면 왼쪽에 1.0을 곱해 float로 만든다. float에는 결과가 비트까지 같고,
    # int 값(float 범위 안)끼리 곱하거나 더해 float 범위를 넘는 큰 정수가 생겨
    # 뒤에서 OverflowError가 나는 일을 막는다. 상수 쪽은 이미 float라 필요 없다.
    g = a.fn
    h = b.fn
    if op is ast.Add:
        if b.const:
            c = b.value

            def f(v: Values) -> Any:
                return g(v) + c

        elif a.const:
            c = a.value

            def f(v: Values) -> Any:
                return c + h(v)

        else:

            def f(v: Values) -> Any:
                return g(v) * 1.0 + h(v)

    elif op is ast.Sub:
        if b.const:
            c = b.value

            def f(v: Values) -> Any:
                return g(v) - c

        elif a.const:
            c = a.value

            def f(v: Values) -> Any:
                return c - h(v)

        else:

            def f(v: Values) -> Any:
                return g(v) * 1.0 - h(v)

    elif op is ast.Mult:
        if b.const:
            c = b.value

            def f(v: Values) -> Any:
                return g(v) * c

        elif a.const:
            c = a.value

            def f(v: Values) -> Any:
                return c * h(v)

        else:

            def f(v: Values) -> Any:
                return g(v) * 1.0 * h(v)

    elif op is ast.Div:
        if b.const and b.value:
            c = b.value

            def f(v: Values) -> Any:
                return g(v) / c

        elif b.const:

            def f(v: Values) -> Any:
                return 0.0

        else:

            def f(v: Values) -> Any:
                d = h(v)
                return g(v) / d if d else 0.0

    else:  # ast.Pow

        def f(v: Values) -> Any:
            return _pow(g(v), h(v))

    return f


# ---- 함수 -------------------------------------------------------------------


def _call(name: str, args: list[_Node]) -> _Fn:
    fns = [a.fn for a in args]
    if name == "abs":
        (g,) = fns
        k = args[0].key
        if k is not None:

            def f(v: Values) -> Any:
                return abs(v[k])

        else:

            def f(v: Values) -> Any:
                return abs(g(v))

    elif name in ("min", "max"):
        return _minmax(name == "min", fns)
    elif name == "sqrt":
        (g,) = fns
        sqrt = math.sqrt

        def f(v: Values) -> Any:
            x = g(v)
            return sqrt(x) if x > 0.0 else 0.0

    elif name == "exp":
        (g,) = fns
        exp = math.exp

        def f(v: Values) -> Any:
            x = g(v)
            return exp(EXP_CAP if x > EXP_CAP else x)

    elif name == "log":
        (g,) = fns
        log = math.log

        def f(v: Values) -> Any:
            x = g(v)
            return log(x) if x > 1e-12 else LOG_FLOOR

    elif name in ("sin", "cos"):
        (g,) = fns
        trig = math.sin if name == "sin" else math.cos

        def f(v: Values) -> Any:
            try:
                return trig(g(v))
            except (ValueError, OverflowError):  # ±inf
                return math.nan

    elif name == "atan2":
        gy, gx = fns
        atan2 = math.atan2

        def f(v: Values) -> Any:
            return atan2(gy(v), gx(v))

    else:  # clamp: min(max(x, lo), hi)와 같은 규칙 (lo > hi면 hi)
        g, glo, ghi = fns
        if args[1].const and args[2].const:
            lo = args[1].value
            hi = args[2].value

            def f(v: Values) -> Any:
                x = g(v)
                x = lo if lo > x else x
                return hi if hi < x else x

        else:

            def f(v: Values) -> Any:
                x = g(v)
                lo = glo(v)
                x = lo if lo > x else x
                hi = ghi(v)
                return hi if hi < x else x

    return f


def _minmax(is_min: bool, fns: list[_Fn]) -> _Fn:
    # 내장 min/max와 같은 규칙: 앞에서부터, 더 작으면(크면) 바꾼다.
    if len(fns) == 2:
        g, h = fns
        if is_min:

            def f(v: Values) -> Any:
                a = g(v)
                b = h(v)
                return b if b < a else a

        else:

            def f(v: Values) -> Any:
                a = g(v)
                b = h(v)
                return b if b > a else a

        return f
    tup = tuple(fns)
    pick = min if is_min else max

    def fn(v: Values) -> Any:
        return pick([h(v) for h in tup])

    return fn


# ---- 논리 -------------------------------------------------------------------


def _and(fns: list[_Fn]) -> _Fn:
    if len(fns) == 2:
        a, b = fns

        def f2(v: Values) -> bool:
            return True if (a(v) and b(v)) else False

        return f2
    if len(fns) == 3:
        a, b, c = fns

        def f3(v: Values) -> bool:
            return True if (a(v) and b(v) and c(v)) else False

        return f3
    tup = tuple(fns)

    def fn(v: Values) -> bool:
        for h in tup:
            if not h(v):
                return False
        return True

    return fn


def _or(fns: list[_Fn]) -> _Fn:
    if len(fns) == 2:
        a, b = fns

        def f2(v: Values) -> bool:
            return True if (a(v) or b(v)) else False

        return f2
    if len(fns) == 3:
        a, b, c = fns

        def f3(v: Values) -> bool:
            return True if (a(v) or b(v) or c(v)) else False

        return f3
    tup = tuple(fns)

    def fn(v: Values) -> bool:
        for h in tup:
            if h(v):
                return True
        return False

    return fn


def _logic(is_and: bool, parts: list[_Node]) -> _Fn:
    """and/or. 값을 그대로 읽는 피연산자(최대 3개)는 함수 호출 없이 먼저 읽는다.

    피연산자는 부작용이 없고 결과는 언제나 bool이라, 읽는 순서를 바꿔도 값이 같다.
    """
    keys = [p.key for p in parts if p.key is not None][:3]
    rest = [p.fn for p in parts if p.key is None]
    rest += [p.fn for p in parts if p.key is not None][3:]
    if not keys:
        return _and(rest) if is_and else _or(rest)
    g: _Fn | None
    if not rest:
        g = None
    elif len(rest) == 1:
        g = rest[0]
    else:
        g = _and(rest) if is_and else _or(rest)
    if is_and:
        return _and_keys(keys, g)
    return _or_keys(keys, g)


def _and_keys(keys: list[str], g: _Fn | None) -> _Fn:
    if len(keys) == 1:
        (k1,) = keys
        assert g is not None  # and에는 피연산자가 둘 이상

        def a1(v: Values) -> bool:
            return True if (v[k1] and g(v)) else False

        return a1
    if len(keys) == 2:
        k1, k2 = keys
        if g is None:

            def a2(v: Values) -> bool:
                return True if (v[k1] and v[k2]) else False

            return a2

        def a2g(v: Values) -> bool:
            return True if (v[k1] and v[k2] and g(v)) else False

        return a2g
    k1, k2, k3 = keys
    if g is None:

        def a3(v: Values) -> bool:
            return True if (v[k1] and v[k2] and v[k3]) else False

        return a3

    def a3g(v: Values) -> bool:
        return True if (v[k1] and v[k2] and v[k3] and g(v)) else False

    return a3g


def _or_keys(keys: list[str], g: _Fn | None) -> _Fn:
    if len(keys) == 1:
        (k1,) = keys
        assert g is not None  # or에는 피연산자가 둘 이상

        def o1(v: Values) -> bool:
            return True if (v[k1] or g(v)) else False

        return o1
    if len(keys) == 2:
        k1, k2 = keys
        if g is None:

            def o2(v: Values) -> bool:
                return True if (v[k1] or v[k2]) else False

            return o2

        def o2g(v: Values) -> bool:
            return True if (v[k1] or v[k2] or g(v)) else False

        return o2g
    k1, k2, k3 = keys
    if g is None:

        def o3(v: Values) -> bool:
            return True if (v[k1] or v[k2] or v[k3]) else False

        return o3

    def o3g(v: Values) -> bool:
        return True if (v[k1] or v[k2] or v[k3] or g(v)) else False

    return o3g


# ---------------------------------------------------------------------------
# 구문 검사

_WS_TABLE: Final = str.maketrans({"\t": " ", "\n": " ", "\r": " ", "\x0b": " ", "\x0c": " "})
_PAIR: Final[dict[str, str]] = {"(": ")", "[": "]", "{": "}"}
_LONE_EQ: Final = re.compile(r"(?<![=<>!:])=(?!=)")
_NOT_COUNTED: Final = (ast.expr_context, ast.boolop, ast.operator, ast.unaryop, ast.cmpop)
_PARSE_LOCK: Final = threading.RLock()
_REJECTED_OPS: Final[dict[type, str]] = {
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.MatMult: "@",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
    ast.Invert: "~",
    ast.In: "in",
    ast.NotIn: "not in",
    ast.Is: "is",
    ast.IsNot: "is not",
}


_PHASE_VALUE_LABEL: Final[dict[str, str]] = {
    "build": S.EXPR_PHASE_VALUE_BUILD,
    "step": S.EXPR_PHASE_VALUE_STEP,
    "end": S.EXPR_PHASE_VALUE_END,
}
_PHASE_CONTEXT_LABEL: Final[dict[str, str]] = {
    "build": S.EXPR_PHASE_CONTEXT_BUILD,
    "step": S.EXPR_PHASE_CONTEXT_STEP,
    "end": S.EXPR_PHASE_CONTEXT_END,
}


def _what(node: ast.AST) -> str:
    """거부한 구문의 이름 (오류 문구용)."""
    if isinstance(node, ast.Subscript):
        return S.EXPR_WHAT_SUBSCRIPT
    if isinstance(node, ast.Slice):
        return S.EXPR_WHAT_SLICE
    if isinstance(node, ast.Lambda):
        return S.EXPR_WHAT_LAMBDA
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return S.EXPR_WHAT_COMPREHENSION
    if isinstance(node, ast.NamedExpr):
        return S.EXPR_WHAT_WALRUS
    if isinstance(node, ast.Starred):
        return S.EXPR_WHAT_STARRED
    if isinstance(node, (ast.JoinedStr, ast.FormattedValue)):
        return S.EXPR_WHAT_FSTRING
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        return S.EXPR_WHAT_CONTAINER
    if isinstance(node, (ast.Await, ast.Yield, ast.YieldFrom)):
        return S.EXPR_WHAT_AWAIT
    return S.EXPR_WHAT_OTHER


def _constant_what(value: object) -> str:
    if isinstance(value, str):
        return S.EXPR_WHAT_STRING
    if isinstance(value, bytes):
        return S.EXPR_WHAT_BYTES
    if value is None:
        return S.EXPR_WHAT_NONE
    if isinstance(value, complex):
        return S.EXPR_WHAT_COMPLEX
    if value is Ellipsis:
        return S.EXPR_WHAT_ELLIPSIS
    return S.EXPR_WHAT_OTHER


def _scan(text: str) -> tuple[str, int] | None:
    """괄호 짝, 괄호 깊이, 주석('#')을 파싱 전에 검사한다. 문자열 리터럴 안은 건너뛴다."""
    stack: list[tuple[str, int]] = []
    quote: str | None = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#":
            return S.EXPR_ERR_COMMENT, i
        elif ch in "([{":
            stack.append((ch, i))
            if len(stack) > MAX_DEPTH:
                return S.EXPR_ERR_TOO_DEEP.format(limit=MAX_DEPTH), i
        elif ch in ")]}":
            if not stack or _PAIR[stack[-1][0]] != ch:
                return S.EXPR_ERR_UNMATCHED.format(char=ch), i
            stack.pop()
        i += 1
    if stack:
        ch, i = stack[-1]
        return S.EXPR_ERR_UNCLOSED.format(char=ch), i
    return None


def _dotted(node: ast.AST) -> str | None:
    """``a.b.c`` 꼴이면 그 문자열, 아니면 None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _hint(name: str, candidates: list[str]) -> str:
    match = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    return " " + S.EXPR_DID_YOU_MEAN.format(hint=match[0]) if match else ""


class _Compiler:
    def __init__(
        self,
        text: str,
        lead: int,
        symbols: Mapping[str, Sym],
        max_phase: Phase,
        query: QueryResolver | None,
    ) -> None:
        self.text = text
        self.raw = text.encode("utf-8")
        self.ascii = text.isascii()
        self.lead = lead
        self.symbols = symbols
        self.max_phase = max_phase
        self.max_rank = PHASE_RANK[max_phase]
        self.query = query
        self.names: set[str] = set()
        self.rank = 0

    # ---- 위치와 오류 ----
    def col_at(self, byte_offset: int) -> int:
        if self.ascii:
            return byte_offset + self.lead
        return len(self.raw[:byte_offset].decode("utf-8", "replace")) + self.lead

    def col(self, node: ast.AST) -> int | None:
        off = getattr(node, "col_offset", None)
        return None if off is None else self.col_at(off)

    def op_col(self, left: ast.AST, right: ast.AST) -> int | None:
        """두 피연산자 사이 연산자의 위치 (닫는 괄호와 공백을 건너뛴다)."""
        start = getattr(left, "end_col_offset", None)
        stop = getattr(right, "col_offset", None)
        if start is None or stop is None:
            return self.col(left)
        i = start
        while i < stop and self.raw[i : i + 1] in (b" ", b")"):
            i += 1
        return self.col_at(i)

    def fail(self, message: str, node: ast.AST | None) -> ExprError:
        return ExprError(message, None if node is None else self.col(node))

    def use(self, name: str, sym: Sym, node: ast.AST, label: str | None = None) -> None:
        rank = PHASE_RANK[sym.phase]
        if rank > self.max_rank:
            raise self.fail(
                S.EXPR_ERR_PHASE.format(
                    name=label or name,
                    phase=_PHASE_VALUE_LABEL[sym.phase],
                    context=_PHASE_CONTEXT_LABEL[self.max_phase],
                ),
                node,
            )
        self.names.add(name)
        if rank > self.rank:
            self.rank = rank

    def need_bool(self, node: _Node, src: ast.AST, op: str) -> None:
        if node.type != "bool":
            raise self.fail(S.EXPR_ERR_NEED_BOOL.format(op=op), src)

    # ---- 노드 ----
    def compile(self, n: ast.AST) -> _Node:
        if isinstance(n, ast.Constant):
            return self.constant(n)
        if isinstance(n, (ast.Name, ast.Attribute)):
            return self.name(n)
        if isinstance(n, ast.UnaryOp):
            return self.unary(n)
        if isinstance(n, ast.BinOp):
            return self.binop(n)
        if isinstance(n, ast.BoolOp):
            return self.boolop(n)
        if isinstance(n, ast.Compare):
            return self.compare(n)
        if isinstance(n, ast.IfExp):
            return self.ifexp(n)
        if isinstance(n, ast.Call):
            return self.call(n)
        raise self.fail(S.EXPR_ERR_SYNTAX_KIND.format(what=_what(n)), n)

    def constant(self, n: ast.Constant) -> _Node:
        value = n.value
        if isinstance(value, bool):
            return _const_bool(value)
        if isinstance(value, int):
            try:
                return _const_num(float(value))
            except OverflowError:
                return _const_num(math.inf)
        if isinstance(value, float):
            return _const_num(value)
        raise self.fail(S.EXPR_ERR_CONSTANT.format(what=_constant_what(value)), n)

    def name(self, n: ast.Name | ast.Attribute) -> _Node:
        name = _dotted(n)
        if name is None:
            cur: ast.AST = n
            while isinstance(cur, ast.Attribute):
                if "__" in cur.attr:
                    raise self.fail(S.EXPR_ERR_DUNDER.format(name=cur.attr), n)
                cur = cur.value
            raise self.fail(S.EXPR_ERR_ATTRIBUTE, n)
        if "__" in name:
            raise self.fail(S.EXPR_ERR_DUNDER.format(name=name), n)
        if name == "pi":
            return _const_num(math.pi)
        sym = self.symbols.get(name)
        if sym is None:
            candidates = sorted({*self.symbols, "pi", "True", "False"})
            raise self.fail(S.EXPR_ERR_UNKNOWN_NAME.format(name=name) + _hint(name, candidates), n)
        self.use(name, sym, n)
        return _read(name, sym.type)

    def unary(self, n: ast.UnaryOp) -> _Node:
        if isinstance(n.op, ast.Invert):
            raise ExprError(S.EXPR_ERR_OPERATOR.format(op="~"), self.col(n))
        operand = self.compile(n.operand)
        if isinstance(n.op, ast.Not):
            self.need_bool(operand, n.operand, "not")
            if operand.const:
                return _const_bool(not operand.value)
            if operand.key is not None:
                k = operand.key

                def fk(v: Values) -> bool:
                    return False if v[k] else True

                return _Node(fk, "bool", exact=True)
            g = operand.fn

            def f(v: Values) -> bool:
                return False if g(v) else True

            return _Node(f, "bool", exact=True)
        operand = _as_num(operand)
        if isinstance(n.op, ast.UAdd):
            return operand
        # ast.USub
        if operand.const:
            return _const_num(-operand.value)
        if operand.key is not None:
            key = operand.key

            def nk(v: Values) -> Any:
                return -v[key]

            return _Node(nk, "num")
        h = operand.fn

        def neg(v: Values) -> Any:
            return -h(v)

        return _Node(neg, "num")

    def binop(self, n: ast.BinOp) -> _Node:
        op = type(n.op)
        if op not in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow):
            message = S.EXPR_ERR_OPERATOR.format(op=_REJECTED_OPS.get(op, "?"))
            raise ExprError(message, self.op_col(n.left, n.right))
        a = _as_num(self.compile(n.left))
        b = _as_num(self.compile(n.right))
        node = _Node(_arith(op, a, b), "num")
        return _fold(node) if a.const and b.const else node

    def boolop(self, n: ast.BoolOp) -> _Node:
        word = "and" if isinstance(n.op, ast.And) else "or"
        parts = [self.compile(x) for x in n.values]
        for part, src in zip(parts, n.values, strict=True):
            self.need_bool(part, src, word)
        node = _Node(_logic(word == "and", parts), "bool", exact=True)
        return _fold(node) if all(p.const for p in parts) else node

    def compare(self, n: ast.Compare) -> _Node:
        srcs = [n.left, *n.comparators]
        ops = [type(o) for o in n.ops]
        for i, op in enumerate(ops):
            if op not in _CMP_SYMBOL:
                message = S.EXPR_ERR_OPERATOR.format(op=_REJECTED_OPS.get(op, "?"))
                raise ExprError(message, self.op_col(srcs[i], srcs[i + 1]))
        parts = [self.compile(x) for x in srcs]
        for i, (part, src) in enumerate(zip(parts, srcs, strict=True)):
            if part.type != "num":
                symbol = _CMP_SYMBOL[ops[i - 1] if i > 0 else ops[0]]
                raise self.fail(S.EXPR_ERR_NEED_NUM_COMPARE.format(op=symbol), src)
        if all(p.const for p in parts):
            return _fold(_Node(_chain([p.fn for p in parts], ops), "bool", exact=True))
        if len(ops) == 1:
            a, b = parts
            op = ops[0]
            if b.const and a.key is not None:
                fn = _cmp_key_const(op, a.key, b.value)
            elif a.const and b.key is not None:
                fn = _cmp_key_const(_FLIP[op], b.key, a.value)
            elif b.const:
                fn = _cmp_fn_const(op, a.fn, b.value)
            elif a.const:
                fn = _cmp_fn_const(_FLIP[op], b.fn, a.value)
            else:
                fn = _cmp_fn_fn(op, a.fn, b.fn)
            return _Node(fn, "bool", exact=True)
        if len(ops) == 2 and parts[0].const and parts[2].const and not parts[1].const:
            lo, mid, hi = parts
            if ops[0] in (ast.Lt, ast.LtE) and ops[1] in (ast.Lt, ast.LtE):
                return _Node(_range(lo.value, ops[0], mid.fn, ops[1], hi.value), "bool", exact=True)
            if ops[0] in (ast.Gt, ast.GtE) and ops[1] in (ast.Gt, ast.GtE):
                # hi > x > lo  ==  lo < x < hi
                return _Node(
                    _range(hi.value, _FLIP[ops[1]], mid.fn, _FLIP[ops[0]], lo.value), "bool", exact=True
                )
        return _Node(_chain([p.fn for p in parts], ops), "bool", exact=True)

    def ifexp(self, n: ast.IfExp) -> _Node:
        test = self.compile(n.test)
        if test.type != "bool":
            raise self.fail(S.EXPR_ERR_NEED_BOOL_IF, n.test)
        body = self.compile(n.body)
        orelse = self.compile(n.orelse)
        typ: ValueType = "bool" if body.type == orelse.type == "bool" else "num"
        if typ == "num":
            body = _as_num(body)
            orelse = _as_num(orelse)
        c, a, b = test.fn, body.fn, orelse.fn

        def f(v: Values) -> Any:
            return a(v) if c(v) else b(v)

        exact = typ == "bool" and body.exact and orelse.exact
        node = _Node(f, typ, exact=exact)
        return _fold(node) if test.const and body.const and orelse.const else node

    def call(self, n: ast.Call) -> _Node:
        if not isinstance(n.func, ast.Name):
            dotted = _dotted(n.func)
            if dotted is not None and "__" in dotted:
                raise self.fail(S.EXPR_ERR_DUNDER.format(name=dotted), n.func)
            raise self.fail(S.EXPR_ERR_CALL_TARGET, n.func)
        fname = n.func.id
        if "__" in fname:
            raise self.fail(S.EXPR_ERR_DUNDER.format(name=fname), n.func)
        if n.keywords:
            kw = n.keywords[0]
            if kw.arg is None:  # **kwargs
                raise self.fail(S.EXPR_ERR_SYNTAX_KIND.format(what=S.EXPR_WHAT_STARRED), kw)
            raise self.fail(S.EXPR_ERR_KEYWORD.format(name=kw.arg), kw)
        for a in n.args:
            if isinstance(a, ast.Starred):
                raise self.fail(S.EXPR_ERR_SYNTAX_KIND.format(what=S.EXPR_WHAT_STARRED), a)
        if fname in FUNCTIONS:
            lo, hi = FUNCTIONS[fname]
            if not lo <= len(n.args) <= hi:
                expected = str(lo) if lo == hi else f"{lo}~{hi}"
                raise self.fail(S.EXPR_ERR_ARG_COUNT.format(func=fname, expected=expected, got=len(n.args)), n)
            args = [_as_num(self.compile(a)) for a in n.args]
            node = _Node(_call(fname, args), "num")
            return _fold(node) if all(a.const for a in args) else node
        if fname in QUERY_FUNCTIONS:
            return self.query_call(n, fname)
        available = list(FUNCTIONS) + (list(QUERY_FUNCTIONS) if self.query is not None else [])
        message = S.EXPR_ERR_UNKNOWN_FUNC.format(name=fname, funcs=", ".join(available))
        raise self.fail(message + _hint(fname, available), n.func)

    def query_call(self, n: ast.Call, fname: str) -> _Node:
        if self.query is None:
            raise self.fail(S.EXPR_ERR_QUERY_UNAVAILABLE.format(func=fname), n)
        if len(n.args) != QUERY_ARITY:
            raise self.fail(
                S.EXPR_ERR_ARG_COUNT.format(func=fname, expected=str(QUERY_ARITY), got=len(n.args)), n
            )
        refs: list[str] = []
        for a in n.args:
            ref = _dotted(a)
            if ref is None:
                raise self.fail(S.EXPR_ERR_QUERY_ARG.format(func=fname), a)
            if "__" in ref:
                raise self.fail(S.EXPR_ERR_DUNDER.format(name=ref), a)
            refs.append(ref)
        try:
            channel, sym = self.query(fname, tuple(refs))
        except ExprError as exc:
            raise ExprError(exc.message, self.col(n)) from exc
        self.use(channel, sym, n, label=f"{fname}({', '.join(refs)})")
        return _read(channel, sym.type)


def _check_size(tree: ast.Expression, comp: _Compiler) -> None:
    count = 0
    deepest = 0
    deepest_node: ast.AST = tree.body
    stack: list[tuple[ast.AST, int]] = [(tree.body, 1)]
    while stack:
        node, depth = stack.pop()
        count += 1
        if depth > deepest:
            deepest = depth
            deepest_node = node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, _NOT_COUNTED):
                stack.append((child, depth + 1))
    if count > MAX_NODES:
        raise ExprError(S.EXPR_ERR_TOO_COMPLEX.format(count=count, limit=MAX_NODES))
    if deepest > MAX_DEPTH:
        raise comp.fail(S.EXPR_ERR_TOO_DEEP.format(limit=MAX_DEPTH), deepest_node)


def _syntax_error(text: str, lead: int, exc: BaseException) -> ExprError:
    if isinstance(exc, (RecursionError, MemoryError)):
        return ExprError(S.EXPR_ERR_TOO_DEEP.format(limit=MAX_DEPTH))
    m = _LONE_EQ.search(text)
    if m is not None:
        return ExprError(S.EXPR_ERR_ASSIGN, m.start() + lead)
    if not isinstance(exc, SyntaxError) or not isinstance(exc.offset, int):
        return ExprError(S.EXPR_ERR_SYNTAX)
    offset = exc.offset
    if "'else'" in (exc.msg or ""):
        return ExprError(S.EXPR_ERR_IF_ELSE, max(offset - 1, 0) + lead)
    if offset < 1 or offset > len(text):  # 입력 끝에서 막힘
        return ExprError(S.EXPR_ERR_INCOMPLETE, len(text) + lead)
    return ExprError(S.EXPR_ERR_SYNTAX, offset - 1 + lead)


def _finish(node: _Node, want: ValueType | None) -> tuple[ValueType, Callable[[Values], float | bool]]:
    """최종 함수: 숫자면 nan/범위 처리, 참/거짓이면 파이썬 bool."""
    if node.type == "bool" and want != "num":
        if node.exact:
            return "bool", node.fn
        g = node.fn

        def to_bool(v: Values) -> bool:
            return True if g(v) else False

        return "bool", to_bool
    num = _as_num(node)
    if num.const:
        value = _finite(num.value)

        def const(v: Values) -> float:
            return value

        return "num", const
    h = num.fn
    lim = LIMIT

    def to_num(v: Values) -> float:
        r = h(v)
        if r != r:
            return 0.0
        if r > lim:
            return lim
        if r < -lim:
            return -lim
        return float(r)

    return "num", to_num


def compile_expr(
    source: str,
    symbols: Mapping[str, Sym],
    *,
    want: ValueType | None = None,
    max_phase: Phase = "end",
    query: QueryResolver | None = None,
) -> Compiled:
    """수식을 컴파일한다. 잘못된 수식이면 한국어 문구의 ExprError."""
    if want not in (None, "num", "bool"):
        raise ValueError(f"want must be None, 'num' or 'bool', not {want!r}")
    if max_phase not in PHASE_RANK:
        raise ValueError(f"max_phase must be one of {PHASES}, not {max_phase!r}")
    if not isinstance(source, str):
        raise TypeError(f"source must be str, not {type(source).__name__}")
    if len(source) > MAX_LENGTH:
        raise ExprError(S.EXPR_ERR_TOO_LONG.format(length=len(source), limit=MAX_LENGTH))
    spaced = source.translate(_WS_TABLE)
    text = spaced.lstrip(" ")
    lead = len(spaced) - len(text)
    text = text.rstrip(" ")
    if not text:
        raise ExprError(S.EXPR_ERR_EMPTY)
    problem = _scan(text)
    if problem is not None:
        message, at = problem
        raise ExprError(message, at + lead)
    try:
        # catch_warnings는 프로세스 전체의 warnings.filters를 바꿨다가 되돌린다. 두 스레드가 동시에
        # 들어가면 되돌리는 순서가 엇갈려 'ignore' 필터가 영영 남으므로 잠금으로 한 번에 하나만 들어간다.
        with _PARSE_LOCK, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError) as exc:
        raise _syntax_error(text, lead, exc) from None
    comp = _Compiler(text, lead, symbols, max_phase, query)
    _check_size(tree, comp)
    try:
        node = comp.compile(tree.body)
    except RecursionError:
        raise ExprError(S.EXPR_ERR_TOO_DEEP.format(limit=MAX_DEPTH)) from None
    if want == "bool" and node.type != "bool":
        raise comp.fail(S.EXPR_ERR_WANT_BOOL, tree.body)
    typ, fn = _finish(node, want)
    return Compiled(
        source=source,
        type=typ,
        phase=PHASES[comp.rank],
        names=frozenset(comp.names),
        fn=fn,
    )

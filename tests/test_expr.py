"""수식 언어 (evolab.scoring.expr) 테스트."""

from __future__ import annotations

import ast
import math
import pickle
import random
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab.app import strings as S
from evolab.scoring import expr as E
from evolab.scoring.expr import Compiled, ExprError, Sym, compile_expr, formula_body, is_formula

HANGUL = re.compile(r"[가-힣]")
EXPR_SOURCE = Path(E.__file__)

SYMS: dict[str, Sym] = {
    "x": Sym("num", "step"),
    "y": Sym("num", "step"),
    "z": Sym("num", "step"),
    "a": Sym("bool", "step"),
    "b": Sym("bool", "step"),
    "c": Sym("bool", "step"),
    "contact_l": Sym("bool", "step"),
    "contact_r": Sym("bool", "step"),
    "hull.speed": Sym("num", "step"),
    "hull.angle": Sym("num", "step"),
    "landed.since": Sym("num", "step"),
    "p.leg_len": Sym("num", "build"),
    "r.wind": Sym("num", "build"),
    "gen.terrain.pad.x": Sym("num", "build"),
    "time": Sym("num", "end"),
    "success": Sym("bool", "end"),
    "fuel_used": Sym("num", "end"),
    "속도": Sym("num", "step"),
}

VALUES: dict[str, float | bool] = {
    "x": 0.25,
    "y": -1.5,
    "z": 3.0,
    "a": True,
    "b": False,
    "c": True,
    "contact_l": True,
    "contact_r": True,
    "hull.speed": 0.3,
    "hull.angle": -0.1,
    "landed.since": 1.2,
    "p.leg_len": 0.8,
    "r.wind": 2.0,
    "gen.terrain.pad.x": 10.0,
    "time": 12.5,
    "success": True,
    "fuel_used": 0.4,
    "속도": 2.0,
}


def ev(source: str, values: dict[str, Any] | None = None, **kw: Any) -> float | bool:
    return compile_expr(source, SYMS, **kw)({**VALUES, **(values or {})})


def err(source: str, symbols: dict[str, Sym] | None = None, **kw: Any) -> ExprError:
    with pytest.raises(ExprError) as info:
        compile_expr(source, SYMS if symbols is None else symbols, **kw)
    e = info.value
    assert HANGUL.search(e.message), f"message is not Korean: {e.message!r}"
    return e


# ---------------------------------------------------------------------------
# '=' 수식 표시


def test_is_formula() -> None:
    assert is_formula("=1+2")
    assert is_formula("= p.leg_len * 2")
    assert not is_formula("1+2")
    assert not is_formula(" =1")
    assert not is_formula(1.0)
    assert not is_formula(None)
    assert not is_formula(["=1"])


def test_formula_body() -> None:
    assert formula_body("=1+2") == "1+2"
    assert formula_body("=  p.leg_len * 2  ") == "p.leg_len * 2"
    assert formula_body("  = x ") == "x"
    assert formula_body("x") == "x"


# ---------------------------------------------------------------------------
# 기본 문법과 값


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1", 1.0),
        ("1 + 2 * 3", 7.0),
        ("(1 + 2) * 3", 9.0),
        ("7 / 2", 3.5),
        ("2 ** 3 ** 2", 512.0),  # 오른쪽 결합
        ("-2 ** 2", -4.0),
        ("+3", 3.0),
        ("- -3", 3.0),
        ("0x10 + 1_000", 1016.0),
        ("1e3", 1000.0),
        ("pi", math.pi),
        ("2 * pi", 2 * math.pi),
        ("x * 4", 1.0),
        ("x - y", 1.75),
        ("y - x", -1.75),
        ("1 - x", 0.75),
        ("x / 0.5", 0.5),
        ("1 / x", 4.0),
        ("x / y", 0.25 / -1.5),
        ("x ** 2", 0.0625),
        ("-x", -0.25),
        ("+x", 0.25),
        ("-(x + 1)", -1.25),
        ("hull.speed * 10", 3.0),
        ("gen.terrain.pad.x - p.leg_len", 9.2),
        ("p . leg_len", 0.8),
        ("속도 * 2", 4.0),
    ],
)
def test_arithmetic(source: str, expected: float) -> None:
    result = ev(source)
    assert type(result) is float
    assert result == pytest.approx(expected, rel=1e-15, abs=0.0)


def test_int_literals_become_float() -> None:
    c = compile_expr("3", SYMS)
    assert c({}) == 3.0 and type(c({})) is float
    assert type(ev("7 // 1" if False else "7 * 1")) is float


def test_dotted_name_is_one_key() -> None:
    c = compile_expr("hull.speed + gen.terrain.pad.x", SYMS)
    assert c.names == frozenset({"hull.speed", "gen.terrain.pad.x"})
    assert c({"hull.speed": 1.0, "gen.terrain.pad.x": 2.0}) == 3.0


def test_constants_and_bools() -> None:
    assert ev("True") is True
    assert ev("False") is False
    assert ev("not False") is True
    c = compile_expr("2 * pi + 1", SYMS)
    assert c.phase == "build" and c.names == frozenset() and c.type == "num"


def test_pi_is_always_the_constant() -> None:
    c = compile_expr("pi", {"pi": Sym("num", "step")})
    assert c({"pi": 3.0}) == math.pi
    assert c.names == frozenset()


# ---------------------------------------------------------------------------
# 비교


@pytest.mark.parametrize(
    ("source", "xs"),
    [
        ("0 < x < 1", {-1.0: False, 0.0: False, 0.5: True, 1.0: False, 2.0: False}),
        ("0 <= x < 1", {-1.0: False, 0.0: True, 0.5: True, 1.0: False}),
        ("0 < x <= 1", {0.0: False, 0.5: True, 1.0: True, 1.5: False}),
        ("0 <= x <= 1", {0.0: True, 1.0: True, -0.1: False, 1.1: False}),
        ("1 > x > 0", {0.0: False, 0.5: True, 1.0: False}),
        ("1 >= x >= 0", {0.0: True, 1.0: True, 1.5: False}),
        ("1 >= x > 0", {0.0: False, 1.0: True}),
        ("x < 1 < 2", {0.0: True, 1.0: False}),
        ("0 < x < 1 < 2", {0.5: True, 1.5: False}),
        ("y < x < z", {0.0: True, -2.0: False, 3.0: False}),
        ("0 < x > -1", {0.5: True, -0.5: False}),
        ("x == 1", {1.0: True, 0.0: False}),
        ("x != 1", {1.0: False, 0.0: True}),
        ("1 == x", {1.0: True, 0.0: False}),
        ("1 != x", {1.0: False, 0.0: True}),
        ("0.5 > x", {0.4: True, 0.5: False, 0.6: False}),
        ("0.5 >= x", {0.5: True, 0.6: False}),
        ("0.5 < x", {0.4: False, 0.6: True}),
        ("0.5 <= x", {0.5: True, 0.4: False}),
        ("x < 0.5", {0.4: True, 0.5: False}),
        ("x <= 0.5", {0.5: True, 0.6: False}),
        ("x > 0.5", {0.6: True, 0.5: False}),
        ("x >= 0.5", {0.5: True, 0.4: False}),
        ("x * 2 < 1", {0.4: True, 0.5: False}),
        ("1 < x * 2", {0.4: False, 0.6: True}),
        ("x < y + 1", {-0.6: True, -0.5: False}),
        ("abs(x) < 0.3", {-0.2: True, 0.31: False}),
    ],
)
def test_comparisons(source: str, xs: dict[float, bool]) -> None:
    c = compile_expr(source, SYMS, want="bool")
    for x, expected in xs.items():
        result = c({**VALUES, "x": x})
        assert result is expected, (source, x)


def test_chain_evaluates_middle_once_semantics() -> None:
    # 0 < x < 1 은 (0 < x) and (x < 1) 과 같다
    c = compile_expr("0 < x < 1", SYMS)
    d = compile_expr("0 < x and x < 1", SYMS)
    for x in (-1.0, 0.0, 0.3, 1.0, 7.0):
        assert c({"x": x}) is d({"x": x})


# ---------------------------------------------------------------------------
# 논리와 형


def test_and_or_not_return_bool() -> None:
    for source in ("a and b", "a or b", "not a", "a and b and c", "a or b or c", "a and x < 1", "x < 1 or b"):
        for va in (True, False):
            for vb in (True, False):
                r = ev(source, {"a": va, "b": vb})
                assert type(r) is bool, source


def test_bool_symbols_with_truthy_values_still_give_bool() -> None:
    values = {"a": 1.0, "b": 2.0, "c": 0.0, "x": 0.0}
    assert compile_expr("a and b", SYMS)(values) is True
    assert compile_expr("a and c", SYMS)(values) is False
    assert compile_expr("a", SYMS)(values) is True
    assert compile_expr("c", SYMS, want="bool")(values) is False
    assert compile_expr("a if x < 1 else c", SYMS)(values) is True


def test_numpy_inputs_give_python_types() -> None:
    values = {"a": np.bool_(True), "b": np.bool_(False), "x": np.float64(0.25), "y": np.float64(2.0)}
    r = compile_expr("x < y", SYMS)(values)
    assert type(r) is bool and r is True
    r = compile_expr("a and not b", SYMS)(values)
    assert type(r) is bool and r is True
    r = compile_expr("b", SYMS)(values)
    assert r is False
    r = compile_expr("x * y + 1", SYMS)(values)
    assert type(r) is float and r == 1.5
    r = compile_expr("a * 2 - b", SYMS)(values)  # numpy bool 뺄셈도 0/1로 바꿔서 한다
    assert type(r) is float and r == 2.0


def test_exhaustive_logic_truth_tables() -> None:
    """and/or 최적화(그대로 읽는 피연산자를 먼저 읽음)가 결과를 바꾸지 않는다."""
    shapes = [
        "a and b",
        "a or b",
        "a and b and c",
        "a or b or c",
        "a and x < 0",
        "x < 0 and a",
        "x < 0 or a or b",
        "a and x < 0 and b",
        "a or x < 0 or b or c",
        "a and b and c and x < 0",
        "a and b and c and contact_l and x < 0",
        "a or b or c or contact_l",
        "a and b and c and contact_l",
        "not a or b and c",
        "(a or b) and (c or x < 0)",
        "a and (b or x < 0) and not c",
    ]
    for shape in shapes:
        c = compile_expr(shape, SYMS)
        for bits in range(32):
            va, vb, vc, vl = (bool(bits >> i & 1) for i in range(4))
            vx = -1.0 if bits & 16 else 1.0
            env = {"a": va, "b": vb, "c": vc, "contact_l": vl, "x": vx}
            ref = eval(shape, {"__builtins__": {}}, env)  # 테스트 기준값으로만 쓴다
            assert c(env) is bool(ref), (shape, env)


def test_bool_in_arithmetic_is_zero_or_one() -> None:
    assert ev("success * 100") == 100.0
    assert ev("success * 100", {"success": False}) == 0.0
    assert ev("-success") == -1.0
    assert ev("success + success") == 2.0
    assert ev("(x < 1) + (x < 2) + (x < 3)") == 3.0
    assert ev("max(success, 0.5)") == 1.0
    assert ev("100 * success - 20 * fuel_used") == pytest.approx(92.0)
    assert ev("True + 1") == 2.0


def test_ifexp() -> None:
    assert ev("1 if a else 2") == 1.0
    assert ev("1 if b else 2") == 2.0
    assert ev("a if b else c") is True
    assert ev("x if success else 0") == 0.25
    # 섞인 형은 숫자: 참/거짓 쪽이 1.0/0.0
    c = compile_expr("a if b else 5", SYMS)
    assert c.type == "num"
    assert c({**VALUES, "b": True}) == 1.0
    assert c({**VALUES, "b": False}) == 5.0
    c = compile_expr("x if y < 0 else z", SYMS)
    assert c.names == frozenset({"x", "y", "z"})


def test_result_types_and_want() -> None:
    assert compile_expr("x + 1", SYMS).type == "num"
    assert compile_expr("x < 1", SYMS).type == "bool"
    assert compile_expr("x < 1", SYMS, want="bool").type == "bool"
    c = compile_expr("x < 1", SYMS, want="num")  # want='num'은 참/거짓을 받고 1.0/0.0으로 돌려준다
    assert c.type == "num"
    assert c({"x": 0.0}) == 1.0 and type(c({"x": 0.0})) is float
    assert c({"x": 2.0}) == 0.0
    assert compile_expr("success", SYMS, want="num")({"success": True}) == 1.0
    assert compile_expr("True", SYMS, want="num")({}) == 1.0
    e = err("x + 1", want="bool")
    assert e.message == S.EXPR_ERR_WANT_BOOL and e.col == 0


@pytest.mark.parametrize(
    ("source", "message", "col"),
    [
        ("x and a", S.EXPR_ERR_NEED_BOOL.format(op="and"), 0),
        ("a and x", S.EXPR_ERR_NEED_BOOL.format(op="and"), 6),
        ("a or 1", S.EXPR_ERR_NEED_BOOL.format(op="or"), 5),
        ("not x", S.EXPR_ERR_NEED_BOOL.format(op="not"), 4),
        ("1 if x else 2", S.EXPR_ERR_NEED_BOOL_IF, 5),
        ("success < 1", S.EXPR_ERR_NEED_NUM_COMPARE.format(op="<"), 0),
        ("1 == success", S.EXPR_ERR_NEED_NUM_COMPARE.format(op="=="), 5),
        ("0 < a < 1", S.EXPR_ERR_NEED_NUM_COMPARE.format(op="<"), 4),
        ("a == b", S.EXPR_ERR_NEED_NUM_COMPARE.format(op="=="), 0),
    ],
)
def test_type_errors(source: str, message: str, col: int) -> None:
    e = err(source)
    assert e.message == message
    assert e.col == col


# ---------------------------------------------------------------------------
# 함수


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("min(3, 1)", 1.0),
        ("max(3, 1)", 3.0),
        ("min(x, y)", -1.5),
        ("max(x, y)", 0.25),
        ("min(5, 4, 3, 2, 1, 0, -1, -2)", -2.0),
        ("max(5, 4, 3, 2, 1, 0, -1, x)", 5.0),
        ("min(x, y, z)", -1.5),
        ("max(x, y, z)", 3.0),
        ("abs(-2)", 2.0),
        ("abs(y)", 1.5),
        ("abs(y * 2)", 3.0),
        ("sqrt(16)", 4.0),
        ("sqrt(z + 1)", 2.0),
        ("exp(0)", 1.0),
        ("exp(x - 0.25)", 1.0),
        ("log(1)", 0.0),
        ("log(exp(z))", 3.0),
        ("clamp(5, 0, 1)", 1.0),
        ("clamp(-5, 0, 1)", 0.0),
        ("clamp(x, 0, 1)", 0.25),
        ("clamp(z, 0, 1)", 1.0),
        ("clamp(y, 0, 1)", 0.0),
        ("clamp(z, y, x)", 0.25),
        ("clamp(x, 2, 1)", 1.0),  # lo > hi 이면 min(max(x, lo), hi) 처럼 hi
        ("sin(0)", 0.0),
        ("cos(0)", 1.0),
        ("sin(pi / 2)", 1.0),
        ("cos(x)", math.cos(0.25)),
        ("sin(y)", math.sin(-1.5)),
        ("atan2(1, 1)", math.pi / 4),
        ("atan2(y, z)", math.atan2(-1.5, 3.0)),
    ],
)
def test_functions(source: str, expected: float) -> None:
    assert ev(source) == pytest.approx(expected, rel=1e-15, abs=1e-15)


@pytest.mark.parametrize(
    ("source", "func", "expected", "got"),
    [
        ("min(1)", "min", "2~8", 1),
        ("max(1, 2, 3, 4, 5, 6, 7, 8, 9)", "max", "2~8", 9),
        ("abs()", "abs", "1", 0),
        ("abs(1, 2)", "abs", "1", 2),
        ("clamp(1, 2)", "clamp", "3", 2),
        ("atan2(1)", "atan2", "2", 1),
        ("sqrt(1, 2)", "sqrt", "1", 2),
    ],
)
def test_wrong_arg_count(source: str, func: str, expected: str, got: int) -> None:
    e = err(source)
    assert e.message == S.EXPR_ERR_ARG_COUNT.format(func=func, expected=expected, got=got)
    assert e.col == 0


def test_unknown_function_lists_functions_and_suggests() -> None:
    e = err("sqr(4)")
    assert e.message.startswith(S.EXPR_ERR_UNKNOWN_FUNC.format(name="sqr", funcs=", ".join(E.FUNCTIONS)))
    assert S.EXPR_DID_YOU_MEAN.format(hint="sqrt") in e.message
    e = err("foo(1)")
    assert S.EXPR_DID_YOU_MEAN.format(hint="sqrt") not in e.message
    for name in ("eval", "exec", "compile", "open", "getattr", "globals", "print", "len", "round", "pow"):
        e = err(f"{name}(1)")
        assert e.message.startswith(S.EXPR_ERR_UNKNOWN_FUNC.format(name=name, funcs=", ".join(E.FUNCTIONS)))


# ---------------------------------------------------------------------------
# 수치 안전


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1 / 0", 0.0),
        ("x / 0", 0.0),
        ("x / (y - y)", 0.0),
        ("x / -0.0", 0.0),
        ("sqrt(-1)", 0.0),
        ("sqrt(y)", 0.0),
        ("log(0)", math.log(1e-12)),
        ("log(-5)", math.log(1e-12)),
        ("log(y)", math.log(1e-12)),
        ("log(exp(1000))", 50.0),  # exp 인자는 50까지
        ("exp(z * 1000) / exp(50)", 1.0),
        ("exp(1000)", 1e12),  # exp(50)은 1e12보다 커서 최종 결과에서 잘린다
        ("exp(-1e308)", 0.0),
        ("9**9**9", 1e12),
        ("2 ** 1000000", 1e12),
        ("10 ** 10 ** 10 ** 10", 1e12),
        ("(-10) ** 401", -1e12),
        ("(-10) ** 400", 1e12),
        ("z ** 1e9", 1e12),
        ("y ** 1e9", 1e12),
        ("y ** 1000001", -1e12),
        ("10 ** -400", 0.0),
        ("(-8) ** (1/3)", 0.0),
        ("y ** 0.5", 0.0),
        ("0 ** -1", 0.0),
        ("(x - x) ** -2", 0.0),
        ("1e308 * 1e308", 1e12),
        ("-1e308 * 1e308", -1e12),
        ("1e308 * 1e308 - 1e308 * 1e308", 0.0),  # inf - inf = nan -> 0
        ("1e999", 1e12),
        ("-1e999", -1e12),
        ("1" + "0" * 400, 1e12),
        ("2e12", 1e12),
        ("-5e15", -1e12),
        ("sin(1e308 * 1e308)", 0.0),  # sin(inf) = nan -> 0
        ("cos(-1e308 * 1e308)", 0.0),
        ("x * 1e308 * 1e308", 1e12),
        ("min(1e308 * 1e308, 5)", 5.0),
        ("sqrt(1e308 * 1e308)", 1e12),
    ],
)
def test_numeric_safety(source: str, expected: float) -> None:
    result = ev(source)
    assert type(result) is float
    assert result == expected


def test_huge_powers_are_fast() -> None:
    t0 = time.perf_counter()
    for source in ("9**9**9", "2**1000000", "9**9**9**9**9", "x ** 9 ** 9 ** 9"):
        c = compile_expr(source, SYMS)
        assert c({"x": 3.0}) == 1e12
        assert c({"x": 3.0}) == 1e12
    assert time.perf_counter() - t0 < 0.5


EXTREME = [0.0, -0.0, 1.0, -1.0, 0.5, 1e-300, -1e-300, 5e-324, 1e308, -1e308, 1.7976931348623157e308, 7.0, -7.0]
NON_FINITE = [math.inf, -math.inf, math.nan]

FUZZ_EXPRESSIONS = [
    "x + y * z - x / y",
    "x / y / z",
    "x ** y",
    "y ** x ** z",
    "(x * y) ** (z / x)",
    "-x ** -y",
    "sqrt(x) + sqrt(y * z)",
    "exp(x) - exp(-y) * exp(z)",
    "log(x) + log(y) * log(z)",
    "sin(x * y) + cos(y * z)",
    "atan2(x * z, y * z)",
    "min(x, y, z) * max(x * y, z, -x)",
    "clamp(x * y, y, z)",
    "abs(x * y) / abs(z)",
    "x * 1e308 * y",
    "(x * 1e308 * 1e308) - (y * 1e308 * 1e308)",
    "0 < x < y <= z",
    "x == y or y != z and not (x > z)",
    "x if y > z else y ** z",
    "a * x + b * y",
    "clamp(x, 0, 1) * 100 - fuel_used * 20 + success * 50",
]


def _fuzz_values(rng: random.Random, pool: list[float]) -> dict[str, float | bool]:
    vals: dict[str, float | bool] = {k: rng.choice(pool) for k in ("x", "y", "z", "fuel_used")}
    vals.update({k: rng.random() < 0.5 for k in ("a", "b", "success")})
    return vals


@pytest.mark.parametrize("source", FUZZ_EXPRESSIONS)
def test_evaluation_never_raises_for_finite_inputs(source: str) -> None:
    c = compile_expr(source, SYMS)
    rng = random.Random(1234)
    for _ in range(400):
        r = c(_fuzz_values(rng, EXTREME))
        if c.type == "num":
            assert type(r) is float and math.isfinite(r) and -1e12 <= r <= 1e12
        else:
            assert type(r) is bool


@pytest.mark.parametrize("source", FUZZ_EXPRESSIONS)
def test_evaluation_does_not_raise_for_non_finite_inputs(source: str) -> None:
    c = compile_expr(source, SYMS)
    rng = random.Random(99)
    for _ in range(200):
        r = c(_fuzz_values(rng, EXTREME + NON_FINITE))
        if c.type == "num":
            assert type(r) is float and math.isfinite(r)


# ---------------------------------------------------------------------------
# 무작위 식을 파이썬 기준값과 비교 (특수화한 클로저가 모두 맞는지)


def _gen_num(rng: random.Random, depth: int) -> str:
    k = rng.randrange(9 if depth > 0 else 2)
    if k == 0:
        return repr(rng.choice([0.5, -2.0, 3.0, 1.0, 0.0, -0.25, 2.0]))
    if k == 1:
        return rng.choice(["x", "y", "z"])
    d = depth - 1
    if k == 2:
        return f"({_gen_num(rng, d)} {rng.choice(['+', '-', '*'])} {_gen_num(rng, d)})"
    if k == 3:
        return f"(-{_gen_num(rng, d)})"
    if k == 4:
        n = rng.randrange(2, 5)
        return f"{rng.choice(['min', 'max'])}({', '.join(_gen_num(rng, d) for _ in range(n))})"
    if k == 5:
        return f"abs({_gen_num(rng, d)})"
    if k == 6:
        return f"clamp({_gen_num(rng, d)}, {_gen_num(rng, d)}, {_gen_num(rng, d)})"
    if k == 7:
        return f"({_gen_num(rng, d)} if {_gen_bool(rng, d)} else {_gen_num(rng, d)})"
    return f"({_gen_bool(rng, d)} * {_gen_num(rng, d)})"


def _gen_bool(rng: random.Random, depth: int) -> str:
    k = rng.randrange(7 if depth > 0 else 2)
    if k == 0:
        return rng.choice(["a", "b", "c", "contact_l"])
    if k == 1:
        return rng.choice(["True", "False", "a", "b"])
    d = depth - 1
    ops = ["<", "<=", ">", ">=", "==", "!="]
    if k == 2:
        return f"({_gen_num(rng, d)} {rng.choice(ops)} {_gen_num(rng, d)})"
    if k == 3:
        return f"({_gen_num(rng, d)} {rng.choice(ops)} {_gen_num(rng, d)} {rng.choice(ops)} {_gen_num(rng, d)})"
    if k == 4:
        n = rng.randrange(2, 6)
        return "(" + f" {rng.choice(['and', 'or'])} ".join(_gen_bool(rng, d) for _ in range(n)) + ")"
    if k == 5:
        return f"(not {_gen_bool(rng, d)})"
    return f"({_gen_bool(rng, d)} if {_gen_bool(rng, d)} else {_gen_bool(rng, d)})"


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def test_random_expressions_match_python_reference() -> None:
    rng = random.Random(20261106)
    ref_globals: dict[str, Any] = {"__builtins__": {}, "min": min, "max": max, "abs": abs, "clamp": _clamp}
    grid = [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    checked = 0
    for i in range(600):
        source = _gen_bool(rng, 3) if i % 2 else _gen_num(rng, 3)
        try:
            c = compile_expr(source, SYMS)
        except ExprError as exc:  # 너무 커진 식은 건너뛴다
            assert exc.message.startswith(S.EXPR_ERR_TOO_COMPLEX.split("(")[0])
            continue
        for _ in range(8):
            env: dict[str, Any] = {k: rng.choice(grid) for k in ("x", "y", "z")}
            env.update({k: rng.random() < 0.5 for k in ("a", "b", "c", "contact_l")})
            ref = eval(source, ref_globals, dict(env))  # 테스트 기준값으로만 쓴다
            got = c(env)
            if c.type == "bool":
                assert got is bool(ref), source
            else:
                assert type(got) is float
                assert got == float(ref), (source, env)
            checked += 1
    assert checked > 3000


# ---------------------------------------------------------------------------
# 단계


def test_phase_of_result() -> None:
    assert compile_expr("1 + 2", SYMS).phase == "build"
    assert compile_expr("p.leg_len * 2", SYMS).phase == "build"
    assert compile_expr("p.leg_len * hull.speed", SYMS).phase == "step"
    assert compile_expr("time + hull.speed + p.leg_len", SYMS).phase == "end"
    assert compile_expr("1 if success else hull.speed", SYMS).phase == "end"


def test_phase_errors() -> None:
    e = err("p.leg_len * 2 + hull.speed", max_phase="build")
    assert e.message == S.EXPR_ERR_PHASE.format(
        name="hull.speed", phase=S.EXPR_PHASE_VALUE_STEP, context=S.EXPR_PHASE_CONTEXT_BUILD
    )
    assert e.col == 16
    e = err("contact_l and time > 3", max_phase="step")
    assert e.message == S.EXPR_ERR_PHASE.format(
        name="time", phase=S.EXPR_PHASE_VALUE_END, context=S.EXPR_PHASE_CONTEXT_STEP
    )
    assert e.col == 14
    # 쓰지 않는 가지라도 단계 검사는 한다
    err("1 if True else success * 2", max_phase="step")
    # 허용되는 경우
    assert compile_expr("p.leg_len * 2 + r.wind", SYMS, max_phase="build")({**VALUES}) == 3.6
    assert compile_expr("hull.speed < p.leg_len", SYMS, max_phase="step").phase == "step"
    assert compile_expr("success and time < 10", SYMS, max_phase="end").phase == "end"


def test_invalid_api_arguments() -> None:
    with pytest.raises(ValueError):
        compile_expr("1", SYMS, max_phase="later")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compile_expr("1", SYMS, want="int")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compile_expr(1, SYMS)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 질의 함수


class Resolver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def __call__(self, func: str, args: tuple[str, ...]) -> tuple[str, Sym]:
        self.calls.append((func, args))
        if args[0] == "nope":
            raise ExprError("부품 'nope'이 없습니다.")
        return f"{func}:{':'.join(args)}", Sym("num" if func == "dist" else "bool", "step")


def test_query_resolver() -> None:
    r = Resolver()
    c = compile_expr(
        "contact(leg_l, tag.pad) and inside(leg_r, pad) and dist(hull, pad) < 2", SYMS, want="bool", query=r
    )
    assert r.calls == [("contact", ("leg_l", "tag.pad")), ("inside", ("leg_r", "pad")), ("dist", ("hull", "pad"))]
    assert c.names == frozenset({"contact:leg_l:tag.pad", "inside:leg_r:pad", "dist:hull:pad"})
    assert c.phase == "step"
    vals = {"contact:leg_l:tag.pad": True, "inside:leg_r:pad": True, "dist:hull:pad": 1.0}
    assert c(vals) is True
    assert c({**vals, "dist:hull:pad": 3.0}) is False
    assert c({**vals, "inside:leg_r:pad": False}) is False
    assert len(r.calls) == 3  # 컴파일 때만 부른다


def test_query_types_are_checked() -> None:
    r = Resolver()
    e = err("dist(hull, pad) and a", query=r)
    assert e.message == S.EXPR_ERR_NEED_BOOL.format(op="and")
    assert compile_expr("dist(hull, pad) * 2", SYMS, query=r)({"dist:hull:pad": 1.5}) == 3.0
    assert compile_expr("contact(a, b) * 10", SYMS, query=r)({"contact:a:b": True}) == 10.0


def test_query_errors() -> None:
    e = err("x + contact(leg_l, ground)")
    assert e.message == S.EXPR_ERR_QUERY_UNAVAILABLE.format(func="contact") and e.col == 4
    r = Resolver()
    e = err("a and contact(nope, ground)", query=r)
    assert e.message == "부품 'nope'이 없습니다." and e.col == 6
    e = err("contact(leg_l, 1)", query=r)
    assert e.message == S.EXPR_ERR_QUERY_ARG.format(func="contact") and e.col == 15
    e = err("inside(leg_l.x + 1, pad)", query=r)
    assert e.message == S.EXPR_ERR_QUERY_ARG.format(func="inside") and e.col == 7
    e = err("dist(hull)", query=r)
    assert e.message == S.EXPR_ERR_ARG_COUNT.format(func="dist", expected="2", got=1)
    e = err("dist(hull, pad, x)", query=r)
    assert e.message == S.EXPR_ERR_ARG_COUNT.format(func="dist", expected="2", got=3)
    e = err("contact(leg_l, __class__)", query=r)
    assert e.message == S.EXPR_ERR_DUNDER.format(name="__class__")
    e = err("contact(a=leg_l, b=pad)", query=r)
    assert e.message == S.EXPR_ERR_KEYWORD.format(name="a")
    e = err("dist(hull, pad) < 1", query=r, max_phase="build")
    assert e.message == S.EXPR_ERR_PHASE.format(
        name="dist(hull, pad)", phase=S.EXPR_PHASE_VALUE_STEP, context=S.EXPR_PHASE_CONTEXT_BUILD
    )
    assert r.calls[-1] == ("dist", ("hull", "pad"))


def test_query_function_names_listed_only_with_resolver() -> None:
    e = err("contakt(a, b)", query=Resolver())
    assert "contact" in e.message
    e = err("contakt(a, b)")
    assert "contact" not in e.message


# ---------------------------------------------------------------------------
# 거부하는 입력 (악의적인 입력 포함)


MALICIOUS = [
    "__import__('os')",
    "__import__('os').system('ls')",
    "().__class__",
    "().__class__.__bases__[0].__subclasses__()",
    "x.__class__",
    "a__b",
    "p.__dict__",
    "'a' * 10000",
    "'a'",
    "b'a'",
    "f'{x}'",
    "None",
    "1j",
    "...",
    "[1, 2]",
    "(1, 2)",
    "{1, 2}",
    "{1: 2}",
    "x[0]",
    "x[1:2]",
    "lambda: 1",
    "lambda x=1: x",
    "[i for i in x]",
    "{i for i in x}",
    "(i for i in x)",
    "{i: i for i in x}",
    "(y := 1)",
    "min(*x)",
    "min(**x)",
    "min(x, key=y)",
    "x.real",
    "(1).real",
    "hull.speed.__class__",
    "math.sqrt(4)",
    "x.y(1)",
    "sqrt(4)(1)",
    "eval('1')",
    "exec('1')",
    "compile('1', '', 'eval')",
    "getattr(x, 'y')",
    "globals()",
    "open('/etc/passwd')",
    "x // 2",
    "x % 2",
    "x @ y",
    "x | y",
    "x & y",
    "x ^ y",
    "x << 1",
    "x >> 1",
    "~x",
    "x in y",
    "x not in y",
    "x is y",
    "x is not y",
    "await x",
    "x = 1",
    "x == 1 = 2",
    "1; 2",
    "import os",
    "x # comment",
    "x\x00",
    "x +",
    "",
    "   ",
    "\n\t",
    "'(' * 300",
    "(" * 300 + "1" + ")" * 300,
    "(" * 101 + "x" + ")" * 101,
    "[" * 300,
    ")" * 10,
    "-" * 999 + "x",
    "not " * 240 + "a",
    "-" * 150 + "x",
    "min(" * 120 + "x" + ", 1)" * 120,
    "x" + " + x" * 200,
    "a" * 10000,
    "1" * 1001,
    "x if a",
    "\ud800",
    "x y",
    "x ! y",
    "$x",
]


@pytest.mark.parametrize("source", MALICIOUS)
def test_rejected_inputs_raise_korean_expr_error(source: str) -> None:
    e = err(source, query=Resolver())
    assert e.col is None or 0 <= e.col <= len(source)
    assert isinstance(e, ValueError)


def test_specific_rejection_messages() -> None:
    assert err("__import__('os')").message == S.EXPR_ERR_DUNDER.format(name="__import__")
    assert err("().__class__").message == S.EXPR_ERR_DUNDER.format(name="__class__")
    assert err("x.__class__").message == S.EXPR_ERR_DUNDER.format(name="x.__class__")
    assert err("(1).real").message == S.EXPR_ERR_ATTRIBUTE
    assert err("x[0]").message == S.EXPR_ERR_SYNTAX_KIND.format(what=S.EXPR_WHAT_SUBSCRIPT)
    assert err("lambda: 1").message == S.EXPR_ERR_SYNTAX_KIND.format(what=S.EXPR_WHAT_LAMBDA)
    assert err("[i for i in x]").message == S.EXPR_ERR_SYNTAX_KIND.format(what=S.EXPR_WHAT_COMPREHENSION)
    assert err("(i for i in x)").message == S.EXPR_ERR_SYNTAX_KIND.format(what=S.EXPR_WHAT_COMPREHENSION)
    assert err("(y := 1)").message == S.EXPR_ERR_SYNTAX_KIND.format(what=S.EXPR_WHAT_WALRUS)
    assert err("min(*x)").message == S.EXPR_ERR_SYNTAX_KIND.format(what=S.EXPR_WHAT_STARRED)
    assert err("min(**x)").message == S.EXPR_ERR_SYNTAX_KIND.format(what=S.EXPR_WHAT_STARRED)
    assert err("f'{x}'").message == S.EXPR_ERR_SYNTAX_KIND.format(what=S.EXPR_WHAT_FSTRING)
    assert err("(1, 2)").message == S.EXPR_ERR_SYNTAX_KIND.format(what=S.EXPR_WHAT_CONTAINER)
    assert err("'a' * 10000").message == S.EXPR_ERR_CONSTANT.format(what=S.EXPR_WHAT_STRING)
    assert err("b'a'").message == S.EXPR_ERR_CONSTANT.format(what=S.EXPR_WHAT_BYTES)
    assert err("None").message == S.EXPR_ERR_CONSTANT.format(what=S.EXPR_WHAT_NONE)
    assert err("1j").message == S.EXPR_ERR_CONSTANT.format(what=S.EXPR_WHAT_COMPLEX)
    assert err("...").message == S.EXPR_ERR_CONSTANT.format(what=S.EXPR_WHAT_ELLIPSIS)
    assert err("min(x, key=y)").message == S.EXPR_ERR_KEYWORD.format(name="key")
    assert err("math.sqrt(4)").message == S.EXPR_ERR_CALL_TARGET
    assert err("x // 2").message == S.EXPR_ERR_OPERATOR.format(op="//")
    assert err("~x").message == S.EXPR_ERR_OPERATOR.format(op="~")
    assert err("x not in y").message == S.EXPR_ERR_OPERATOR.format(op="not in")
    assert err("x is y").message == S.EXPR_ERR_OPERATOR.format(op="is")
    assert err("a__b").message == S.EXPR_ERR_DUNDER.format(name="a__b")
    assert err("x = 1").message == S.EXPR_ERR_ASSIGN
    assert err("x # comment").message == S.EXPR_ERR_COMMENT
    assert err("").message == S.EXPR_ERR_EMPTY
    assert err(" \n\t ").message == S.EXPR_ERR_EMPTY
    assert err("x +").message == S.EXPR_ERR_INCOMPLETE
    assert err("a if b").message == S.EXPR_ERR_IF_ELSE
    assert err("a b").message == S.EXPR_ERR_SYNTAX


def test_size_limits() -> None:
    e = err("a" * 10000)
    assert e.message == S.EXPR_ERR_TOO_LONG.format(length=10000, limit=E.MAX_LENGTH) and e.col is None
    assert compile_expr("x" + " " * 999, SYMS)({"x": 1.0}) == 1.0  # 정확히 1000자는 된다
    err("x" + " " * 1000)
    # 요소 수
    e = err("x" + " + x" * 200)
    assert e.message == S.EXPR_ERR_TOO_COMPLEX.format(count=401, limit=E.MAX_NODES)
    ok = "x" + " + x" * 99  # 199 요소, 깊이 100
    assert compile_expr(ok, SYMS)({"x": 1.0}) == 100.0
    # 깊이
    e = err("-" * 150 + "x")
    assert e.message == S.EXPR_ERR_TOO_DEEP.format(limit=E.MAX_DEPTH)
    assert compile_expr("-" * 99 + "x", SYMS)({"x": 2.0}) == -2.0
    e = err("(" * 300 + "1" + ")" * 300)
    assert e.message == S.EXPR_ERR_TOO_DEEP.format(limit=E.MAX_DEPTH) and e.col == E.MAX_DEPTH
    assert compile_expr("(" * 100 + "x" + ")" * 100, SYMS)({"x": 2.0}) == 2.0
    e = err("min(" * 120 + "x" + ", 1)" * 120)
    assert e.message == S.EXPR_ERR_TOO_DEEP.format(limit=E.MAX_DEPTH)


@pytest.mark.parametrize("exc", [RecursionError, MemoryError])
def test_parser_exhaustion_becomes_expr_error(monkeypatch: pytest.MonkeyPatch, exc: type[BaseException]) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise exc()

    monkeypatch.setattr(E.ast, "parse", boom)
    e = err("x + 1")
    assert e.message == S.EXPR_ERR_TOO_DEEP.format(limit=E.MAX_DEPTH)


def test_compile_recursion_becomes_expr_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self: Any, n: ast.AST) -> Any:
        raise RecursionError()

    monkeypatch.setattr(E._Compiler, "compile", boom)
    e = err("x + 1")
    assert e.message == S.EXPR_ERR_TOO_DEEP.format(limit=E.MAX_DEPTH)


def test_source_never_uses_eval_exec_compile() -> None:
    tree = ast.parse(EXPR_SOURCE.read_text(encoding="utf-8"))
    banned = {"eval", "exec", "compile", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in banned, f"line {node.lineno}: {node.func.id}()"
        if isinstance(node, ast.Name):
            assert node.id not in ("builtins", "__builtins__"), f"line {node.lineno}"


# ---------------------------------------------------------------------------
# 이름과 제안


def test_unknown_name_suggestion() -> None:
    e = err("hull.sped < 1")
    assert e.message == S.EXPR_ERR_UNKNOWN_NAME.format(name="hull.sped") + " " + S.EXPR_DID_YOU_MEAN.format(
        hint="hull.speed"
    )
    assert e.col == 0
    e = err("x + qqqqqq")
    assert e.message == S.EXPR_ERR_UNKNOWN_NAME.format(name="qqqqqq") and e.col == 4
    e = err("true")
    assert S.EXPR_DID_YOU_MEAN.format(hint="True") in e.message
    e = err("p.leg_lne * 2")
    assert S.EXPR_DID_YOU_MEAN.format(hint="p.leg_len") in e.message
    e = err("x", symbols={})
    assert e.message == S.EXPR_ERR_UNKNOWN_NAME.format(name="x")


# ---------------------------------------------------------------------------
# 열 번호


@pytest.mark.parametrize(
    ("source", "col"),
    [
        ("hull.speed + foo", 13),
        ("  foo", 2),
        ("\tfoo", 1),
        ("x +\n  foo", 6),
        ("속도 + foo", 5),
        ("속도 + (1", 5),
        ("x + 1)", 5),
        ("min(x, (y]", 9),
        ("a b", 2),
        ("x +", 3),
        ("  x +  ", 5),
        ("x // 2", 2),
        ("(x)  %  2", 5),
        ("x in y", 2),
        ("0 < x is y", 6),
        ("min(x, key=1)", 7),
        ("x == 1 or y = 2", 12),
        ("x # c", 2),
        ("1 + 'a'", 4),
        ("속도 + x[0]", 5),
        ("속도 and a", 0),
        ("a and 속도", 6),
        ("abs(속도) + nope", 10),
    ],
)
def test_columns(source: str, col: int) -> None:
    e = err(source)
    assert e.col == col, e
    assert str(e) == S.EXPR_ERR_AT.format(message=e.message, column=col + 1)


def test_str_without_column_is_message() -> None:
    e = ExprError("메시지")
    assert e.col is None and str(e) == "메시지" and e.message == "메시지"
    e = ExprError("메시지", 4)
    assert str(e) == S.EXPR_ERR_AT.format(message="메시지", column=5)


def test_expr_error_pickles() -> None:
    e = ExprError("메시지", 3)
    back = pickle.loads(pickle.dumps(e))
    assert type(back) is ExprError and back.message == "메시지" and back.col == 3 and str(back) == str(e)


# ---------------------------------------------------------------------------
# Compiled, 결정성


def test_compiled_dataclass() -> None:
    c1 = compile_expr("x + 1", SYMS)
    c2 = compile_expr("x + 1", SYMS)
    assert c1 == c2 and hash(c1) == hash(c2)
    assert "fn=" not in repr(c1)
    assert c1.source == "x + 1"
    with pytest.raises(AttributeError):
        c1.source = "y"  # type: ignore[misc]
    assert c1.fn({"x": 1.0}) == c1({"x": 1.0}) == 2.0
    assert isinstance(c1.names, frozenset)


def test_source_is_kept_verbatim() -> None:
    assert compile_expr("  x +\n1 ", SYMS).source == "  x +\n1 "


def test_same_inputs_same_bits() -> None:
    rng = random.Random(7)
    sources = FUZZ_EXPRESSIONS + ["contact_l and contact_r and hull.speed < 0.5"]
    for source in sources:
        c1 = compile_expr(source, SYMS)
        c2 = compile_expr(source, SYMS)
        for _ in range(100):
            vals = {**VALUES, **_fuzz_values(rng, EXTREME + [rng.uniform(-10, 10) for _ in range(5)])}
            r1, r2 = c1(vals), c2(vals)
            assert type(r1) is type(r2)
            if isinstance(r1, float):
                assert np.float64(r1).tobytes() == np.float64(r2).tobytes()
            else:
                assert r1 is r2


DETERMINISM_SCRIPT = r"""
import math, struct
from evolab.scoring.expr import compile_expr, Sym, ExprError
syms = {f"s{i}.v{j}": Sym("num" if (i + j) % 2 else "bool", ("build", "step", "end")[(i * j) % 3])
        for i in range(12) for j in range(12)}
syms.update({"aa": Sym(), "ab": Sym(), "ba": Sym(), "abc": Sym(), "acb": Sym()})
out = []
for src in ["s1.v2 + s3.v4 * 2", "s1.v1 and s2.v2 or not s3.v3", "s1.v3 if s2.v2 else s5.v6",
            "s1.v", "s1.v1x", "abd", "ac", "sqr(1)", "s0.v1 + s9.v2 ** 9 ** 9", "s1.v1 < 1"]:
    try:
        c = compile_expr(src, syms, max_phase="end")
        vals = {k: (1.5 if s.type == "num" else True) for k, s in syms.items()}
        r = c(vals)
        out.append(f"{src}|{c.type}|{c.phase}|{sorted(c.names)}|{struct.pack('<d', float(r)).hex()}")
    except ExprError as e:
        out.append(f"{src}|ERR|{e}")
print("\n".join(out))
"""


def test_deterministic_across_hash_seeds() -> None:
    outputs = []
    for seed in ("0", "1", "4242"):
        env = {"PYTHONHASHSEED": seed, "PATH": "", "SYSTEMROOT": ""}
        proc = subprocess.run(
            [sys.executable, "-c", DETERMINISM_SCRIPT],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**_base_env(), **env},
            check=True,
        )
        outputs.append(proc.stdout)
    assert outputs[0] == outputs[1] == outputs[2]
    assert "ERR" in outputs[0] and "|num|" in outputs[0]


def _base_env() -> dict[str, str]:
    import os

    keep = {k: v for k, v in os.environ.items() if k not in ("PYTHONHASHSEED",)}
    keep["PYTHONIOENCODING"] = "utf-8"
    return keep


# ---------------------------------------------------------------------------
# 성능 (출력만, 느슨한 상한)


def _per_call(fn: Callable[[Any], Any], arg: Any, number: int) -> float:
    best = math.inf
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(number):
            fn(arg)
        best = min(best, (time.perf_counter() - t0) / number)
    return best


def test_benchmark_condition() -> None:
    c = compile_expr("contact_l and contact_r and hull.speed < 0.5", SYMS, want="bool", max_phase="step")
    vals = {"contact_l": True, "contact_r": True, "hull.speed": 0.3}
    assert c(vals) is True
    n = 100_000
    t_call = _per_call(c, vals, n)
    t_fn = _per_call(c.fn, vals, n)
    t_ref = _per_call(lambda v: v["contact_l"] and v["contact_r"] and v["hull.speed"] < 0.5, vals, n)
    score = compile_expr("success * 100 - fuel_used * 20 + clamp(10 - time, 0, 10)", SYMS, want="num")
    t_score = _per_call(score, VALUES, n)
    t_compile = _per_call(
        lambda s: compile_expr(s, SYMS, want="bool"), "contact_l and contact_r and hull.speed < 0.5", 2_000
    )
    print(
        f"\n[expr bench] condition __call__ {t_call * 1e9:.0f} ns, .fn {t_fn * 1e9:.0f} ns, "
        f"hand-written lambda {t_ref * 1e9:.0f} ns; score formula {t_score * 1e9:.0f} ns; "
        f"compile {t_compile * 1e6:.1f} us"
    )
    assert t_call < 20e-6
    assert t_fn < 20e-6

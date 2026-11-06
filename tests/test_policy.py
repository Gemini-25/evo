"""M2: 신경망(BatchBrain)과 유전자 구조(GenomeLayout).

- 파라미터 수, 펼치기·되돌리기, 파라미터 순서
- 순전파가 행마다 독립(N=1 계산, 다른 묶음 크기, 섞은 순서, 다른 행의 값과 무관하게 비트 단위로 같음)
- 출력 범위, relu 은닉층, 구동기 범위 변환
- 몸 수치 변환과 잘라내기, 초기 유전자
- 서명이 프로세스와 상관없이 같고, 구조가 바뀔 때만 바뀜
"""

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pytest

from evolab import paths
from evolab.policy.brain import BatchBrain, BrainLayout, BrainSizeError, pack, unpack
from evolab.policy.genome import BodyGene, GenomeError, GenomeLayout, layout_for
from evolab.spec.io import compile_env

LANDER_TEXT = (paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8")
LANDER_DATA: dict[str, Any] = json.loads(LANDER_TEXT)


def lander(edit: Any = None) -> GenomeLayout:
    """내장 착륙선의 유전자 구조. edit(data)로 환경 파일 내용을 고친 뒤 컴파일한다."""
    data = copy.deepcopy(LANDER_DATA)
    if edit is not None:
        edit(data)
    return layout_for(compile_env(data))


def random_brain(layout: BrainLayout, n: int, seed: int, scale: float = 1.0) -> BatchBrain:
    rng = np.random.default_rng(seed)
    return unpack(layout, scale * rng.standard_normal((n, layout.n_params)))


def reference_forward(layout: BrainLayout, flat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """느리지만 뻔한 계산(행마다 np.dot). 값 비교용(비트 비교 아님)."""
    out = []
    for row, xi in zip(flat, x, strict=True):
        h = np.asarray(xi, dtype=np.float64)
        off = 0
        for k, (fi, fo) in enumerate(layout.shapes):
            w = row[off : off + fi * fo].reshape(fi, fo)
            off += fi * fo
            b = row[off : off + fo]
            off += fo
            h = h @ w + b
            last = k == len(layout.shapes) - 1
            h = np.tanh(h) if last or layout.activation == "tanh" else np.maximum(h, 0.0)
        out.append(h)
    return np.array(out).reshape(len(x), layout.n_out)


# ---------------------------------------------------------------------------
# 크기와 펼치기
# ---------------------------------------------------------------------------


def test_param_counts() -> None:
    assert BrainLayout(9, (16,), 3).n_params == 211  # 착륙선 9 -> 16 -> 3
    assert BrainLayout(9, (16,), 2).n_params == 194  # 트랙 자동차 9 -> 16 -> 2
    assert BrainLayout(4, (), 2).n_params == 4 * 2 + 2
    assert BrainLayout(3, (5, 7), 2).shapes == [(3, 5), (5, 7), (7, 2)]
    assert BrainLayout(3, (5, 7), 2).n_params == 3 * 5 + 5 + 5 * 7 + 7 + 7 * 2 + 2

    layout = lander()
    assert layout.n_body == 0 and layout.brain.n_params == 211 and layout.dim == 211
    assert layout.brain == BrainLayout(9, (16,), 3, "tanh")
    assert layout.output_ranges == ((0.0, 1.0),) * 3
    assert layout.inputs == ("pad_rel.x", "pad_rel.y", "st.vx", "st.vy", "st.angle", "st.omega", "c_l", "c_r", "fuel")
    assert layout.outputs == ("main", "side_l", "side_r")


def test_layout_validation() -> None:
    with pytest.raises(ValueError):
        BrainLayout(9, (0,), 3)
    with pytest.raises(ValueError):
        BrainLayout(9, (16,), 3, "sigmoid")  # type: ignore[arg-type]
    assert BrainLayout(9, [16, 8], 3).hidden == (16, 8)  # type: ignore[arg-type]  # JSON의 list도 받는다
    with pytest.raises(ValueError):
        GenomeLayout((), BrainLayout(2, (), 2), ((0.0, 1.0),))  # 출력 범위 수가 다름
    with pytest.raises(ValueError):
        BodyGene("a", 1.0, 1.0, 1.0)  # low < high 아님
    with pytest.raises(ValueError):
        BodyGene("a", 0.0, 1.0, 2.0)  # 초깃값이 범위 밖


def test_pack_unpack_round_trip_and_order() -> None:
    layout = BrainLayout(3, (4, 5), 2)
    rng = np.random.default_rng(3)
    flat = rng.standard_normal((7, layout.n_params))
    brain = unpack(layout, flat)
    assert brain.n == 7
    for (fi, fo), w, b in zip(layout.shapes, brain.weights, brain.biases, strict=True):
        assert w.shape == (7, fi, fo) and b.shape == (7, fo)
        assert w.dtype == np.float64 and w.flags.c_contiguous and b.flags.c_contiguous
    back = pack(brain)
    assert back.shape == flat.shape and back.flags.c_contiguous
    assert back.tobytes() == flat.tobytes()
    again = unpack(layout, back)
    assert all(np.array_equal(a, b) for a, b in zip(again.weights, brain.weights, strict=True))

    # 순서: 층마다 W(행 우선, fan_in x fan_out) 다음 b
    seq = np.arange(layout.n_params, dtype=np.float64)[None, :]
    b2 = unpack(layout, seq)
    assert b2.weights[0][0, 1, 2] == 1 * 4 + 2
    assert np.array_equal(b2.biases[0][0], np.arange(12, 16))
    assert b2.weights[1][0, 0, 0] == 16
    assert b2.weights[1][0, 3, 4] == 16 + 3 * 5 + 4
    assert np.array_equal(b2.biases[2][0], [layout.n_params - 2, layout.n_params - 1])

    # 펼친 가중치는 원본과 메모리를 같이 쓰지 않는다
    flat[0, 0] = 1e9
    assert brain.weights[0][0, 0, 0] != 1e9

    # 1차원은 N=1
    assert unpack(layout, seq[0]).n == 1
    with pytest.raises(BrainSizeError, match="필요합니다"):
        unpack(layout, np.zeros((2, layout.n_params + 1)))


# ---------------------------------------------------------------------------
# 순전파
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_forward_is_row_independent(activation: str) -> None:
    layout = BrainLayout(9, (16,), 3, activation)  # type: ignore[arg-type]
    deep = BrainLayout(5, (12, 7), 4, activation)  # type: ignore[arg-type]
    wide = BrainLayout(9, (64,), 2, activation)  # type: ignore[arg-type]
    for lay in (layout, deep, wide):
        rng = np.random.default_rng(11)
        n = 37
        flat = rng.standard_normal((n, lay.n_params))
        x = rng.uniform(-5, 5, (n, lay.n_in)).astype(np.float32)
        full = unpack(lay, flat).forward(x)
        assert full.shape == (n, lay.n_out) and full.dtype == np.float64

        # N=1로 한 행씩
        for i in range(n):
            one = unpack(lay, flat[i : i + 1]).forward(x[i : i + 1])
            assert one.tobytes() == full[i : i + 1].tobytes(), i

        # 여러 묶음 크기와 섞은 순서
        for size in (2, 3, 5, 8, 16, 31, 32, 33, 64, 100, 160):
            rows = rng.integers(0, n, size)
            sub = unpack(lay, flat[rows]).forward(x[rows])
            assert sub.tobytes() == full[rows].tobytes(), size
        perm = rng.permutation(n)
        assert unpack(lay, flat[perm]).forward(x[perm]).tobytes() == full[perm].tobytes()

        # 다른 행이 무엇이든(아주 큰 값, NaN) 같은 행의 결과는 같다
        wild = rng.standard_normal((n, lay.n_params)) * 1e6
        wild[::3] = np.nan
        wx = rng.uniform(-5, 5, (n, lay.n_in))
        wild[5], wx[5] = flat[5], x[5]
        assert unpack(lay, wild).forward(wx)[5].tobytes() == full[5].tobytes()

        # select로 뽑은 묶음도 같은 값(같은 번호를 여러 번 써도 됨)
        brain = unpack(lay, flat)
        rows = np.array([4, 4, 0, 36, 17])
        sel = brain.select(rows)
        assert sel.n == 5 and all(w.flags.c_contiguous for w in sel.weights)
        assert sel.forward(x[rows]).tobytes() == full[rows].tobytes()
        mask = np.zeros(n, dtype=bool)
        mask[[1, 2, 30]] = True
        assert brain.select(mask).forward(x[mask]).tobytes() == full[mask].tobytes()


def test_forward_matches_reference_and_input_dtype() -> None:
    for lay in (BrainLayout(9, (16,), 3), BrainLayout(4, (6, 5), 2, "relu"), BrainLayout(3, (), 2)):
        rng = np.random.default_rng(5)
        flat = rng.standard_normal((10, lay.n_params))
        x32 = rng.uniform(-5, 5, (10, lay.n_in)).astype(np.float32)
        brain = unpack(lay, flat)
        y = brain.forward(x32)
        assert np.allclose(y, reference_forward(lay, flat, x32), rtol=0, atol=1e-12)
        # float32 입력은 float64로 바꿔 계산한다(같은 값이면 같은 결과)
        assert brain.forward(x32.astype(np.float64)).tobytes() == y.tobytes()
        assert brain.forward(x32.tolist()).tobytes() == y.tobytes()  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        unpack(BrainLayout(9, (16,), 3), np.zeros((2, 211))).forward(np.zeros((3, 9)))
    with pytest.raises(ValueError):
        unpack(BrainLayout(9, (16,), 3), np.zeros((2, 211))).forward(np.zeros((2, 8)))
    # 빈 묶음
    assert unpack(BrainLayout(9, (16,), 3), np.zeros((0, 211))).forward(np.zeros((0, 9))).shape == (0, 3)


def test_forward_same_result_twice_and_zero_brain() -> None:
    lay = BrainLayout(9, (16,), 3)
    brain = random_brain(lay, 32, seed=2)
    x = np.random.default_rng(9).uniform(-5, 5, (32, 9))
    assert brain.forward(x).tobytes() == brain.forward(x).tobytes()
    assert not np.shares_memory(brain.forward(x), x)
    zero = unpack(lay, np.zeros((4, lay.n_params))).forward(np.ones((4, 9)))
    assert np.array_equal(zero, np.zeros((4, 3)))  # 뇌 가중치 0 -> 출력 0


@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_output_range(activation: str) -> None:
    lay = BrainLayout(9, (16, 16), 3, activation)  # type: ignore[arg-type]
    brain = random_brain(lay, 64, seed=4, scale=30.0)  # 크게 포화시켜도
    x = np.random.default_rng(1).uniform(-5, 5, (64, 9))
    y = brain.forward(x)
    assert np.all(np.isfinite(y)) and np.all(y >= -1.0) and np.all(y <= 1.0)
    assert (np.abs(y) == 1.0).any()  # 실제로 포화된 값이 있다


def test_relu_hidden_layer() -> None:
    # 은닉 뉴런 하나: 입력 x -> relu(w*x) -> 출력 tanh(v*h)
    lay = BrainLayout(1, (1,), 1, "relu")
    flat = np.array([[2.0, 0.0, 1.0, 0.0]])  # W1=2, b1=0, W2=1, b2=0
    brain = unpack(lay, flat)
    y = brain.forward(np.array([[-3.0]]))
    assert y[0, 0] == 0.0  # relu가 음수를 0으로
    y = brain.forward(np.array([[0.25]]))
    assert y[0, 0] == np.tanh(0.5)
    tanh_brain = unpack(BrainLayout(1, (1,), 1, "tanh"), flat)
    assert tanh_brain.forward(np.array([[-3.0]]))[0, 0] == np.tanh(np.tanh(-6.0))
    # 출력층은 relu여도 tanh: 음수 출력이 가능
    neg = unpack(lay, np.array([[1.0, 0.0, -5.0, 0.0]])).forward(np.array([[1.0]]))
    assert neg[0, 0] == np.tanh(-5.0) and neg[0, 0] < 0


def test_forward_speed(capsys: pytest.CaptureFixture[str]) -> None:
    lay = BrainLayout(9, (16,), 3)
    brain = random_brain(lay, 32, seed=0)
    layout = GenomeLayout((), lay, ((0.0, 1.0),) * 3)
    x = np.random.default_rng(0).uniform(-5, 5, (32, 9)).astype(np.float32)
    for _ in range(200):
        brain.forward(x)
    reps = 5000
    t = time.perf_counter()
    for _ in range(reps):
        layout.to_actions(brain.forward(x))
    us = (time.perf_counter() - t) / reps * 1e6
    with capsys.disabled():
        print(f"\n[policy] forward + to_actions, 9->16->3, N=32: {us:.1f} us per call")
    assert us < 2000  # 아주 느려졌을 때만 실패(보통 10~20us)


# ---------------------------------------------------------------------------
# 구동기 범위
# ---------------------------------------------------------------------------


def test_to_actions_mapping() -> None:
    lay = GenomeLayout((), BrainLayout(2, (), 3), ((0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)))
    raw = np.array([[-1.0, -1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1e-20, -0.5], [0.25, -0.75, 0.999]])
    a = lay.to_actions(raw)
    expect = np.array([[0.0, -1.0, 1.0], [0.5, 0.0, 0.5], [1.0, 1e-20, 0.25], [0.625, -0.75, (0.999 + 1) / 2]])
    assert a.tobytes() == expect.tobytes()
    assert a[2, 1] == 1e-20  # (-1, 1) 채널은 값 그대로(비트까지)
    assert raw[0, 0] == -1.0  # 입력을 바꾸지 않는다

    unit = GenomeLayout((), BrainLayout(2, (), 2), ((0.0, 1.0), (0.0, 1.0)))
    assert unit.to_actions(np.array([[-1.0, 1.0]])).tolist() == [[0.0, 1.0]]
    sign = GenomeLayout((), BrainLayout(2, (), 2), ((-1.0, 1.0), (-1.0, 1.0)))
    r = np.array([[-0.3, 0.7]])
    s = sign.to_actions(r)
    assert s.tobytes() == r.tobytes() and not np.shares_memory(s, r)
    other = GenomeLayout((), BrainLayout(1, (), 2), ((0.0, 1.0), (2.0, 6.0)))
    assert other.to_actions(np.array([[0.0, 0.0], [1.0, -1.0]])).tolist() == [[0.5, 4.0], [1.0, 2.0]]
    # 1차원(개체 하나)도 된다
    assert lay.to_actions(np.array([0.0, 0.5, -1.0])).tolist() == [0.5, 0.5, 0.0]


def test_lander_actions_in_actuator_range() -> None:
    layout = lander()
    rng = np.random.default_rng(8)
    genomes = rng.standard_normal((16, layout.dim)) * 3
    brain = unpack(layout.brain, layout.brain_params(genomes))
    a = layout.to_actions(brain.forward(rng.uniform(-5, 5, (16, 9))))
    assert a.shape == (16, 3) and np.all(a >= 0.0) and np.all(a <= 1.0)
    # 초기 유전자(뇌 0)는 0..1 구동기에 0.5
    a0 = layout.to_actions(unpack(layout.brain, layout.brain_params(layout.initial())).forward(np.zeros((1, 9))))
    assert a0.tolist() == [[0.5, 0.5, 0.5]]


# ---------------------------------------------------------------------------
# 몸 수치
# ---------------------------------------------------------------------------


def test_body_mapping_and_clipping() -> None:
    body = (BodyGene("leg", 0.6, 2.0, 1.2), BodyGene("side", -0.4, 0.4, 0.2), BodyGene("w", 1.0, 3.0, 1.0))
    layout = GenomeLayout(body, BrainLayout(2, (3,), 1), ((-1.0, 1.0),))
    assert layout.n_body == 3 and layout.dim == 3 + layout.brain.n_params
    g = np.zeros((5, layout.dim))
    g[0, :3] = [-1.0, -1.0, -1.0]
    g[1, :3] = [1.0, 1.0, 1.0]
    g[2, :3] = [0.0, 0.0, 0.0]
    g[3, :3] = [7.0, -9.0, 1e300]  # 범위 밖은 잘라낸다
    g[4, :3] = [0.5, -0.5, -np.inf]
    v = layout.body_values(g)
    assert v.shape == (5, 3) and v.flags.c_contiguous
    assert v[0].tolist() == [0.6, -0.4, 1.0]
    assert v[1].tolist() == [2.0, 0.4, 3.0]
    assert np.allclose(v[2], [1.3, 0.0, 2.0], rtol=0, atol=1e-15)
    assert v[3].tolist() == [2.0, -0.4, 3.0]
    assert np.allclose(v[4], [1.65, -0.2, 1.0], rtol=0, atol=1e-15)
    lows, highs = np.array([0.6, -0.4, 1.0]), np.array([2.0, 0.4, 3.0])
    rnd = np.random.default_rng(0).uniform(-3, 3, (200, layout.dim))
    vr = layout.body_values(rnd)
    assert np.all(vr >= lows) and np.all(vr <= highs)
    # 공식 그대로: low + (g + 1) / 2 * (high - low)
    inside = np.clip(rnd[:, :3], -1, 1)
    assert np.allclose(vr, lows + (inside + 1) / 2 * (highs - lows), rtol=0, atol=1e-15)
    # 1차원 유전자
    assert layout.body_values(g[1]).tolist() == [2.0, 0.4, 3.0]

    # 초기 유전자: 몸은 초깃값을 -1..1로, 뇌는 0
    init = layout.initial()
    assert init.shape == (layout.dim,) and init.dtype == np.float64
    assert np.allclose(init[:3], [2 * (1.2 - 0.6) / 1.4 - 1, 0.5, -1.0], rtol=0, atol=1e-15)
    assert np.all(init[3:] == 0.0)
    assert np.allclose(layout.body_values(init), [1.2, 0.2, 1.0], rtol=0, atol=1e-12)

    # 뇌 파라미터는 뒤쪽, 복사본
    bp = layout.brain_params(rnd)
    assert bp.shape == (200, layout.brain.n_params) and bp.flags.c_contiguous
    assert np.array_equal(bp, rnd[:, 3:])
    bp[0, 0] = 123.0
    assert rnd[0, 3] != 123.0
    with pytest.raises(GenomeError, match="유전자 길이"):
        layout.body_values(np.zeros((2, layout.dim - 1)))
    with pytest.raises(GenomeError):
        layout.brain_params(np.zeros(layout.dim + 1))


def test_lander_with_evolved_body() -> None:
    def evolve_all(d: dict[str, Any]) -> None:
        for p in d["params"]:
            p["evolve"] = True

    layout = lander(evolve_all)
    assert [g.id for g in layout.body] == ["leg_len", "leg_spread", "side_h"]  # 선언 순서
    assert layout.n_body == 3 and layout.dim == 214
    params = {p["id"]: p for p in LANDER_DATA["params"]}
    for g in layout.body:
        assert (g.low, g.high, g.initial) == (params[g.id]["min"], params[g.id]["max"], params[g.id]["value"])
    init = layout.initial()
    assert np.allclose(layout.body_values(init), [1.2, 0.45, 0.2], rtol=0, atol=1e-12)
    assert np.all(init[3:] == 0.0)


# ---------------------------------------------------------------------------
# 서명과 저장
# ---------------------------------------------------------------------------


def _digest_forward() -> str:
    lay = BrainLayout(9, (16,), 3)
    flat = np.random.default_rng(123).standard_normal((32, lay.n_params))
    x = np.random.default_rng(456).uniform(-5, 5, (32, 9)).astype(np.float32)
    return hashlib.sha256(unpack(lay, flat).forward(x).tobytes()).hexdigest()


SUBPROCESS_CODE = """
import hashlib, sys
import numpy as np
from evolab import paths
from evolab.policy.brain import BrainLayout, unpack
from evolab.policy.genome import layout_for
from evolab.spec.io import load_env
print(layout_for(load_env(paths.builtin_env_dir() / "lander.evoenv")).signature())
lay = BrainLayout(9, (16,), 3)
flat = np.random.default_rng(123).standard_normal((32, lay.n_params))
x = np.random.default_rng(456).uniform(-5, 5, (32, 9)).astype(np.float32)
print(hashlib.sha256(unpack(lay, flat).forward(x).tobytes()).hexdigest())
"""


def test_signature_and_forward_same_in_other_processes() -> None:
    here = [lander().signature(), _digest_forward()]
    assert len(here[0]) == 64 and int(here[0], 16) >= 0
    # 해시 시드와 BLAS 스레드 수가 달라도 같다
    variants = [
        {"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1"},
        {"PYTHONHASHSEED": "1"},
        {"PYTHONHASHSEED": "4242", "OPENBLAS_NUM_THREADS": "4"},
    ]
    for extra in variants:
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHASHSEED", "OPENBLAS_NUM_THREADS")}
        env.update(extra)
        out = subprocess.run([sys.executable, "-c", SUBPROCESS_CODE], env=env, capture_output=True, text=True, check=True)
        assert out.stdout.split() == here, out.stderr


def test_signature_sensitivity() -> None:
    base = lander().signature()
    assert lander().signature() == base

    def set_brain(**kw: Any) -> Any:
        def edit(d: dict[str, Any]) -> None:
            d["brain"].update(kw)

        return edit

    changed = {
        "hidden": lander(set_brain(hidden=[12])),
        "two_layers": lander(set_brain(hidden=[16, 16])),
        "relu": lander(set_brain(activation="relu")),
        "inputs_order": lander(
            set_brain(inputs=["pad_rel.y", "pad_rel.x", "st.vx", "st.vy", "st.angle", "st.omega", "c_l", "c_r", "fuel"])
        ),
        "fewer_inputs": lander(set_brain(inputs=["pad_rel.x", "pad_rel.y", "st.vx", "st.vy", "st.angle", "st.omega"])),
        "outputs_order": lander(set_brain(outputs=["side_l", "main", "side_r"])),
    }

    def evolve(pid: str, **kw: Any) -> Any:
        def edit(d: dict[str, Any]) -> None:
            for p in d["params"]:
                if p["id"] == pid:
                    p["evolve"] = True
                    p.update(kw)

        return edit

    changed["evolve_leg"] = lander(evolve("leg_len"))
    changed["evolve_leg_range"] = lander(evolve("leg_len", max=2.5))
    changed["evolve_spread"] = lander(evolve("leg_spread"))
    sigs = {k: v.signature() for k, v in changed.items()}
    assert base not in sigs.values()
    assert len(set(sigs.values())) == len(sigs)  # 서로도 다르다

    # 같은 이름이라도 출력 범위가 다르면 다르다
    a = GenomeLayout((), BrainLayout(1, (), 1), ((0.0, 1.0),), ("x",), ("u",))
    b = GenomeLayout((), BrainLayout(1, (), 1), ((-1.0, 1.0),), ("x",), ("u",))
    c = GenomeLayout((), BrainLayout(1, (), 1), ((0.0, 1.0),), ("x",), ("v",))
    assert len({a.signature(), b.signature(), c.signature()}) == 3
    assert GenomeLayout((), BrainLayout(1, (), 1), ((-0.0, 1.0),), ("x",), ("u",)).signature() == a.signature()

    # 유전자의 뜻을 바꾸지 않는 수정은 서명을 바꾸지 않는다
    def cosmetic(d: dict[str, Any]) -> None:
        d["name"] = d["name"] + " (2)"
        d["description"] = "changed"
        d["params"][0]["label"] = "leg length"
        d["params"][0]["value"] = 1.3  # 진화하지 않는 매개변수 값
        d["evaluation"]["trials"] = 9
        d["editor"] = {"zoom": 2}

    assert lander(cosmetic).signature() == base

    def initial_only(d: dict[str, Any]) -> None:
        d["params"][0]["evolve"] = True
        d["params"][0]["value"] = 1.5  # 몸 유전자의 초깃값만 다름

    assert lander(initial_only).signature() == changed["evolve_leg"].signature()


def test_to_dict_from_dict_round_trip() -> None:
    def evolve_all(d: dict[str, Any]) -> None:
        for p in d["params"]:
            p["evolve"] = True

    for layout in (
        lander(),
        lander(evolve_all),
        GenomeLayout(
            (BodyGene("다리", -1.5, 2.5, 0.0),),
            BrainLayout(2, (4, 3), 2, "relu"),
            ((0.0, 1.0), (-1.0, 1.0)),
            ("a", "b"),
            ("u", "v"),
        ),
    ):
        d = layout.to_dict()
        text = json.dumps(d, ensure_ascii=False, allow_nan=False)  # JSON으로 쓸 수 있다
        back = GenomeLayout.from_dict(json.loads(text))
        assert back == layout
        assert back.signature() == layout.signature()
        assert back.initial().tobytes() == layout.initial().tobytes()

    good = lander().to_dict()
    broken: list[Any] = [
        {},
        [],
        {**good, "format": 99},
        {**good, "brain": {**good["brain"], "hidden": "16"}},
        {**good, "brain": {**good["brain"], "activation": "sigmoid"}},
        {**good, "brain": {**good["brain"], "n_in": 9.5}},
        {**good, "output_ranges": [[0.0, 1.0]]},  # 출력 수와 다름
        {**good, "output_ranges": [[0.0, 1.0, 2.0]] * 3},
        {**good, "body": [{"id": "x", "low": 1.0, "high": 0.0, "initial": 0.5}]},
        {**good, "body": [{"id": "x", "low": 0.0}]},
        {**good, "inputs": ["only_one"]},
        {**good, "inputs": "pad_rel.x"},
    ]
    for d in broken:
        with pytest.raises(GenomeError, match="유전자 구조 정보"):
            GenomeLayout.from_dict(d)


def test_layout_pickles() -> None:
    import pickle

    layout = lander()
    layout.to_actions(np.zeros((1, 3)))  # 미리 계산한 배열이 생긴 뒤에도
    back = pickle.loads(pickle.dumps(layout))
    assert back == layout and back.signature() == layout.signature() and hash(back) == hash(layout)
    brain = random_brain(layout.brain, 3, seed=1)
    b2 = pickle.loads(pickle.dumps(brain))
    x = np.ones((3, 9))
    assert b2.forward(x).tobytes() == brain.forward(x).tobytes()


# ---------------------------------------------------------------------------
# 적대적 검토에서 찾은 결함의 회귀 테스트
# ---------------------------------------------------------------------------


def test_unpack_never_shares_memory_even_for_one_row() -> None:
    # N=1(2차원 한 줄, 1차원)일 때도 펼친 가중치가 원본의 보기(view)가 아니어야 한다.
    # 보기였을 때는 나중에 유전자 배열을 고치면 이미 펼친 뇌가 몰래 바뀌었다.
    lay = BrainLayout(3, (4,), 2)
    x = np.ones((1, 3))
    for shape in [(1, lay.n_params), (lay.n_params,)]:
        flat = np.random.default_rng(0).standard_normal(shape)
        brain = unpack(lay, flat)
        assert not any(np.shares_memory(a, flat) for a in brain.weights + brain.biases)
        before = brain.forward(x).tobytes()
        flat[...] = 1e6
        assert brain.forward(x).tobytes() == before
    # 읽기 전용 원본(np.load 등)이어도 펼친 가중치는 쓸 수 있는 새 배열
    ro = np.zeros((1, lay.n_params))
    ro.flags.writeable = False
    assert all(w.flags.writeable and w.flags.c_contiguous for w in unpack(lay, ro).weights)


def test_pack_unpack_empty_batch() -> None:
    for lay in (BrainLayout(9, (16,), 3), BrainLayout(3, (4, 5), 2), BrainLayout(0, (2,), 1)):
        empty = unpack(lay, np.zeros((0, lay.n_params)))
        assert empty.n == 0
        assert pack(empty).shape == (0, lay.n_params)  # 예전에는 reshape(0, -1)에서 ValueError
        assert empty.select(np.array([], dtype=np.int64)).n == 0
        assert pack(empty.select([])).shape == (0, lay.n_params)  # type: ignore[arg-type]


def test_select_index_forms() -> None:
    lay = BrainLayout(2, (3,), 1)
    flat = np.random.default_rng(1).standard_normal((4, lay.n_params))
    brain = unpack(lay, flat)
    one = brain.select(np.int64(2))  # type: ignore[arg-type]  # 번호 하나 -> N=1
    assert one.n == 1 and pack(one).tobytes() == flat[2:3].tobytes()
    assert pack(brain.select([3, 0])).tobytes() == flat[[3, 0]].tobytes()  # type: ignore[arg-type]
    assert pack(brain.select(np.array([3, 0], dtype=np.int32))).tobytes() == flat[[3, 0]].tobytes()
    with pytest.raises(TypeError):
        brain.select(np.array([1.7]))  # 실수 번호를 몰래 1로 자르지 않는다
    with pytest.raises(IndexError):
        brain.select(np.array([True, False]))  # 길이가 다른 마스크


def test_shape_errors_are_not_reported_as_wrong_length() -> None:
    # 차원 수가 틀린 배열(프로그램 오류)에 "211개인데 211개가 필요" 같은 엉뚱한 안내를 하지 않는다.
    lay = BrainLayout(9, (16,), 3)
    for bad in (np.zeros((2, 2, 211)), np.float64(1.0)):
        with pytest.raises(ValueError) as info:
            unpack(lay, bad)
        assert not isinstance(info.value, BrainSizeError)
    layout = GenomeLayout((BodyGene("a", 0.0, 1.0, 0.5),), BrainLayout(2, (), 1), ((0.0, 1.0),))
    for bad in (np.zeros((2, 2, layout.dim)), np.float64(1.0)):
        with pytest.raises(ValueError) as ginfo:
            layout.body_values(bad)
        assert not isinstance(ginfo.value, GenomeError)
    # 진짜 길이 오류는 한국어 안내(조사가 숫자 발음에 따라 틀리지 않게 '개'로 센다)
    with pytest.raises(GenomeError) as size:
        lander().brain_params(np.zeros(214))
    assert str(size.value).startswith("유전자 길이가 214개인데 이 환경에는 211개(")


def test_to_actions_rejects_wrong_width() -> None:
    # 'unit'(모두 0..1) 빠른 경로가 모양을 보지 않아 열 수가 달라도 조용히 통과했었다.
    unit = GenomeLayout((), BrainLayout(2, (), 3), ((0.0, 1.0),) * 3)
    sign = GenomeLayout((), BrainLayout(2, (), 2), ((-1.0, 1.0),) * 2)
    for layout, width in ((unit, 2), (unit, 4), (sign, 3)):
        with pytest.raises(ValueError):
            layout.to_actions(np.zeros((5, width)))
    with pytest.raises(ValueError):
        unit.to_actions(np.zeros((2, 2, 3)))
    assert unit.to_actions(np.zeros((0, 3))).shape == (0, 3)


def test_body_values_reject_nan_and_huge_ranges() -> None:
    layout = GenomeLayout(
        (BodyGene("leg", 0.6, 2.0, 1.2), BodyGene("side", -0.4, 0.4, 0.2)), BrainLayout(1, (), 1), ((0.0, 1.0),)
    )
    g = np.zeros((3, layout.dim))
    g[1, 1] = np.nan
    with pytest.raises(GenomeError, match="'side'"):  # NaN은 잘라낼 곳이 없다: 범위 밖 몸 수치를 내지 않는다
        layout.body_values(g)
    g[1, 1] = 0.0
    g[:, 2:] = np.nan  # 뇌 쪽 NaN은 몸 수치와 상관없다
    assert np.all(np.isfinite(layout.body_values(g)))
    # 폭(high - low)이 float 범위를 넘으면 NaN이 나오므로 만들 때 거부(한국어 안내)
    with pytest.raises(GenomeError, match="너무 넓어"):
        BodyGene("w", -1e308, 1e308, 0.0)
    wide = BodyGene("w", -1e307, 1e307, 0.0)
    wl = GenomeLayout((wide,), BrainLayout(1, (), 1), ((0.0, 1.0),))
    assert np.isfinite(wl.initial()).all()
    assert wl.body_values(np.array([[-1.0, 0, 0], [1.0, 0, 0]]))[:, 0].tolist() == [-1e307, 1e307]


def test_layout_rejects_non_integer_sizes() -> None:
    with pytest.raises(TypeError):
        BrainLayout(9.7, (16,), 3)  # type: ignore[arg-type]  # 예전에는 몰래 9로 잘랐다
    with pytest.raises(TypeError):
        BrainLayout(9, (16.0,), 3)  # type: ignore[arg-type]
    assert BrainLayout(np.int64(9), (np.int32(16),), np.int64(3)) == BrainLayout(9, (16,), 3)  # type: ignore[arg-type]


def test_batch_brain_compares_by_identity() -> None:
    lay = BrainLayout(2, (3,), 1)
    a = unpack(lay, np.zeros((2, lay.n_params)))
    b = unpack(lay, np.zeros((2, lay.n_params)))
    assert a == a and a != b  # 예전에는 배열 비교라 ValueError(진릿값이 모호함)
    assert len({a, b}) == 2


def test_forward_ignores_input_layout_and_alignment() -> None:
    # 입력·가중치가 어떤 메모리 배치(전치, 띄엄띄엄, 64바이트 경계에서 어긋남)여도 같은 비트.
    lay = BrainLayout(9, (16,), 3)
    rng = np.random.default_rng(21)
    flat = rng.standard_normal((6, lay.n_params))
    x = rng.uniform(-5, 5, (6, 9))
    brain = unpack(lay, flat)
    ref = brain.forward(x).tobytes()
    assert brain.forward(np.asfortranarray(x)).tobytes() == ref
    wide = np.zeros((6, 27))
    wide[:, ::3] = x
    assert brain.forward(wide[:, ::3]).tobytes() == ref

    def shifted(a: np.ndarray, off: int) -> np.ndarray:
        raw = np.zeros(a.nbytes + 128, dtype=np.uint8)
        start = (-raw.ctypes.data) % 64 + off
        view = raw[start : start + a.nbytes].view(np.float64).reshape(a.shape)
        view[...] = a
        return view

    for off in (8, 24, 40, 56):
        moved = BatchBrain(lay, [shifted(w, off) for w in brain.weights], [shifted(b, 64 - off) for b in brain.biases])
        assert moved.forward(shifted(x, off)).tobytes() == ref, off

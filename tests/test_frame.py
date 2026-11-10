"""v6.1 0'단계: 자라는 뇌의 틀 policy/frame.py (v6 설계 4.14, 7.3, 8.2 T30).

- active_coords(counts)가 unpack한 가중치·편향의 그 칸들과 정확히 같다(무작위 값을 켜진 좌표에만 넣고 펼쳐 비교). 은닉층 1·2·3개,
  몸 수치 있음·없음, 입력 없음, relu. 몸 수치·출력 편향은 늘 켜짐.
- neuron_in·neuron_out·bias가 unpack의 그 뉴런 열·행·편향 자리이고, 켜진 뉴런의 것을 모으면 active_coords와 같다.
- 켜진 좌표만 채운 틀 뇌 = counts 크기의 작은 뇌(꺼진 뉴런의 활성은 정확히 0).
- active_neurons·counts_of·links가 만든 유전자와 맞고, 유전자만 본다(틀 전체의 들어오는·나가는 가중치, 편향, -0.0은 0).
- 은닉층 없는 뇌·뇌 없는 환경, 잘못된 counts·뉴런 자리·유전자 모양에서 FrameError(한국어).
- 내장 예제: 착륙선 뉴런 2·4·8·16개면 켜진 좌표 29·55·107·211, 다 자란 틀 = 유전자 전체.
"""

from __future__ import annotations

import copy
import json
import pickle
import re
from typing import Any

import numpy as np
import pytest

from evolab import paths
from evolab.app import strings as S
from evolab.policy.brain import BatchBrain, BrainLayout, unpack
from evolab.policy.frame import BrainFrame, FrameError
from evolab.policy.genome import BodyGene, GenomeLayout, layout_for
from evolab.spec.io import compile_env

HANGUL = re.compile(r"[가-힣]")


def _layout(n_in: int, hidden: tuple[int, ...], n_out: int, n_body: int = 0, activation: str = "tanh") -> GenomeLayout:
    body = tuple(BodyGene(id=f"p{k}", low=-1.0 - k, high=2.0 + k, initial=0.5) for k in range(n_body))
    return GenomeLayout(body=body, brain=BrainLayout(n_in=n_in, hidden=hidden, n_out=n_out, activation=activation),  # type: ignore[arg-type]
                        output_ranges=((-1.0, 1.0),) * n_out)


FRAMES: dict[str, GenomeLayout] = {
    "one": _layout(9, (16,), 3),
    "one_body": _layout(11, (7,), 2, n_body=5),
    "two": _layout(4, (5, 7), 3, n_body=2),
    "three": _layout(3, (4, 6, 5), 2, n_body=1, activation="relu"),
    "no_inputs": _layout(0, (3, 4), 2),
}


def _counts_cases(sizes: tuple[int, ...], seed: int) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    cases = [tuple(1 for _ in sizes), tuple(sizes), tuple(min(2, s) for s in sizes)]
    cases += [tuple(int(rng.integers(1, s + 1)) for s in sizes) for _ in range(6)]
    return list(dict.fromkeys(cases))


def _expected_blocks(layout: GenomeLayout, counts: tuple[int, ...]) -> list[tuple[np.ndarray, np.ndarray]]:
    """층마다 (W 켜짐 (fan_in, fan_out), b 켜짐 (fan_out,)): 설계 4.14의 규칙을 그대로 옮김(frame.py와 따로 계산)."""
    shapes = layout.brain.shapes
    last = len(shapes) - 1
    out = []
    for k, (fi, fo) in enumerate(shapes):
        rows = np.arange(fi) < (fi if k == 0 else counts[k - 1])
        cols = np.arange(fo) < (fo if k == last else counts[k])
        out.append((rows[:, None] & cols[None, :], cols))
    return out


def _fill(layout: GenomeLayout, coords: np.ndarray, seed: int) -> np.ndarray:
    """coords 자리만 0이 아닌 무작위 값(크기 0.5~1.5, 부호 섞음), 나머지는 정확히 0.0."""
    rng = np.random.default_rng(seed)
    g = np.zeros(layout.dim, dtype=np.float64)
    g[coords] = rng.uniform(0.5, 1.5, coords.size) * rng.choice([-1.0, 1.0], coords.size)
    return g


def _ids() -> list[str]:
    return list(FRAMES)


# ---------------------------------------------------------------------------
# active_coords
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", _ids())
def test_active_coords_are_exactly_the_unpacked_cells(name: str) -> None:
    layout = FRAMES[name]
    frame = BrainFrame(layout)
    assert frame.sizes == layout.brain.hidden
    for i, counts in enumerate(_counts_cases(frame.sizes, seed=len(name))):
        coords = frame.active_coords(counts)
        assert coords.dtype == np.intp and coords.ndim == 1
        assert np.all(np.diff(coords) > 0)  # 오름차순, 겹침 없음
        assert coords[0] >= 0 and coords[-1] < layout.dim
        n_bias = sum(counts) + layout.brain.n_out
        assert coords.size == layout.n_body + frame.links(counts) + n_bias
        g = _fill(layout, coords, seed=i)
        assert np.all(g[: layout.n_body] != 0)  # 몸 수치는 늘 켜짐
        brain = unpack(layout.brain, layout.brain_params(g))
        for k, (w_on, b_on) in enumerate(_expected_blocks(layout, counts)):
            np.testing.assert_array_equal(brain.weights[k][0] != 0, w_on, err_msg=f"{name} {counts} W{k}")
            np.testing.assert_array_equal(brain.biases[k][0] != 0, b_on, err_msg=f"{name} {counts} b{k}")
        assert np.all(brain.biases[-1][0] != 0)  # 출력 편향은 늘 켜짐


@pytest.mark.parametrize("name", _ids())
def test_full_frame_is_the_whole_genome(name: str) -> None:
    layout = FRAMES[name]
    frame = BrainFrame(layout)
    np.testing.assert_array_equal(frame.active_coords(frame.sizes), np.arange(layout.dim))
    assert frame.full_links() == frame.links(frame.sizes) == sum(fi * fo for fi, fo in layout.brain.shapes)


def test_counts_accept_numpy_integers_and_lists() -> None:
    frame = BrainFrame(FRAMES["two"])
    a = frame.active_coords((2, 3))
    np.testing.assert_array_equal(frame.active_coords([2, 3]), a)
    np.testing.assert_array_equal(frame.active_coords(np.array([2, 3], dtype=np.int64)), a)
    np.testing.assert_array_equal(frame.active_coords((np.int32(2), np.int64(3))), a)


# ---------------------------------------------------------------------------
# 뉴런 하나의 좌표
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", _ids())
def test_neuron_coords_point_at_the_unpacked_column_row_and_bias(name: str) -> None:
    """유전자에 좌표 번호를 그대로 넣고 펼치면, 그 뉴런의 들어오는 열·나가는 행·편향 칸에 그 번호들이 있다."""
    layout = FRAMES[name]
    frame = BrainFrame(layout)
    g = np.arange(layout.dim, dtype=np.float64)
    brain = unpack(layout.brain, layout.brain_params(g))
    sizes = frame.sizes
    last = len(sizes) - 1
    for counts in _counts_cases(sizes, seed=7):
        for li, size in enumerate(sizes):
            for j in range(size):  # 꺼진 뉴런도: 자랄 때 새 뉴런의 들어오는 가중치를 새 counts로 구한다
                rows = layout.brain.n_in if li == 0 else counts[li - 1]
                cols = layout.brain.n_out if li == last else counts[li + 1]
                inc = frame.neuron_in(li, j, counts)
                out = frame.neuron_out(li, j, counts)
                assert inc.dtype == np.intp and out.dtype == np.intp
                np.testing.assert_array_equal(inc, brain.weights[li][0][:rows, j].astype(np.intp))
                np.testing.assert_array_equal(out, brain.weights[li + 1][0][j, :cols].astype(np.intp))
                assert frame.bias(li, j) == int(brain.biases[li][0][j])
                assert isinstance(frame.bias(li, j), int)


@pytest.mark.parametrize("name", _ids())
def test_active_neurons_coords_union_is_active_coords(name: str) -> None:
    layout = FRAMES[name]
    frame = BrainFrame(layout)
    last = len(frame.sizes) - 1
    for counts in _counts_cases(frame.sizes, seed=11):
        parts = [np.arange(layout.n_body, dtype=np.intp)]
        for li, c in enumerate(counts):
            for j in range(c):
                parts += [frame.neuron_in(li, j, counts), np.array([frame.bias(li, j)], dtype=np.intp)]
                if li == last:
                    parts.append(frame.neuron_out(li, j, counts))
        b_out = layout.dim - layout.brain.n_out
        parts.append(np.arange(b_out, layout.dim, dtype=np.intp))
        union = np.unique(np.concatenate(parts))
        np.testing.assert_array_equal(union, frame.active_coords(counts))
        # 꺼진 뉴런의 편향·들어오는 가중치는 켜진 좌표 밖
        active = set(frame.active_coords(counts).tolist())
        for li, c in enumerate(counts):
            for j in range(c, frame.sizes[li]):
                assert frame.bias(li, j) not in active
                assert not active & set(frame.neuron_in(li, j, counts).tolist())


# ---------------------------------------------------------------------------
# 켜진 좌표만 채운 틀 = 작은 뇌
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", _ids())
def test_frame_brain_equals_the_smaller_brain(name: str) -> None:
    layout = FRAMES[name]
    frame = BrainFrame(layout)
    rng = np.random.default_rng(3)
    x = rng.uniform(-2.0, 2.0, (6, layout.brain.n_in))
    for i, counts in enumerate(_counts_cases(frame.sizes, seed=5)):
        g = _fill(layout, frame.active_coords(counts), seed=100 + i)
        big = unpack(layout.brain, layout.brain_params(g))
        small_layout = BrainLayout(n_in=layout.brain.n_in, hidden=counts, n_out=layout.brain.n_out,
                                   activation=layout.brain.activation)
        sizes = small_layout.sizes
        small = BatchBrain(small_layout, [w[:, : sizes[k], : sizes[k + 1]] for k, w in enumerate(big.weights)],
                           [b[:, : sizes[k + 1]] for k, b in enumerate(big.biases)])
        trace: list[np.ndarray] = []
        y_big = big.select(np.zeros(6, dtype=np.intp)).forward(x, trace=trace)
        y_small = small.select(np.zeros(6, dtype=np.intp)).forward(x)
        np.testing.assert_allclose(y_big, y_small, rtol=1e-12, atol=1e-15)
        for li, c in enumerate(counts):  # 꺼진 뉴런의 활성은 정확히 0.0(들어오는 가중치·편향이 0)
            assert np.all(trace[li][:, c:] == 0.0) and not np.any(np.signbit(trace[li][:, c:]))


# ---------------------------------------------------------------------------
# 유전자만 보고 켜짐 판단
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", _ids())
def test_active_neurons_counts_of_and_links_match_the_built_genome(name: str) -> None:
    layout = FRAMES[name]
    frame = BrainFrame(layout)
    for i, counts in enumerate(_counts_cases(frame.sizes, seed=13)):
        g = _fill(layout, frame.active_coords(counts), seed=200 + i)
        act = frame.active_neurons(g)
        assert len(act) == len(frame.sizes)
        for li, (a, c) in enumerate(zip(act, counts, strict=True)):
            assert a.dtype == np.bool_ and a.shape == (frame.sizes[li],)
            np.testing.assert_array_equal(a, np.arange(frame.sizes[li]) < c)
        assert frame.counts_of(g) == counts
        brain = unpack(layout.brain, layout.brain_params(g))
        assert frame.links(counts) == sum(int(np.count_nonzero(w)) for w in brain.weights)
        # float16(관찰 기록 obs_genomes)과 읽기 전용 배열에서도 같다. 유전자를 고치지 않는다.
        ro = g.copy()
        ro.setflags(write=False)
        assert frame.counts_of(ro) == counts and frame.counts_of(g.astype(np.float16)) == counts
        np.testing.assert_array_equal(ro, g)


def test_fixed_size_genome_has_every_neuron_on() -> None:
    """크기 고정 실행의 학습된 유전자(모든 좌표가 0이 아님): 모든 뉴런이 켜져 그림이 지금과 같다(설계 4.14)."""
    for layout in FRAMES.values():
        frame = BrainFrame(layout)
        g = np.random.default_rng(1).standard_normal(layout.dim)
        assert all(a.all() for a in frame.active_neurons(g))
        assert frame.counts_of(g) == frame.sizes


def test_active_neurons_look_at_the_whole_frame_not_counts() -> None:
    """들어오는 가중치 하나·편향 하나·나가는 가중치 하나만 0이 아니어도 켜짐. 켜진 좌표 밖(꺼진 앞 뉴런에서 오는 가중치)도 본다."""
    layout = FRAMES["two"]  # 입력 4 → 5 → 7 → 출력 3, 몸 2
    frame = BrainFrame(layout)
    ones = (1, 1)
    zero = np.zeros(layout.dim)
    assert [a.tolist() for a in frame.active_neurons(zero)] == [[False] * 5, [False] * 7]
    assert frame.counts_of(zero) == ones  # 하나도 없으면 1

    g = zero.copy()
    g[frame.neuron_in(0, 3, ones)[2]] = 0.25  # 첫 은닉층 뉴런 3: 입력 2에서 오는 가중치 하나
    g[frame.bias(1, 5)] = -0.5  # 둘째 은닉층 뉴런 5: 편향 하나
    g[frame.neuron_out(1, 2, ones)[1]] = 1.0  # 둘째 은닉층 뉴런 2: 출력 1로 나가는 가중치 하나
    act = frame.active_neurons(g)
    assert np.flatnonzero(act[0]).tolist() == [3] and np.flatnonzero(act[1]).tolist() == [2, 5]
    assert frame.counts_of(g) == (4, 6)  # 마지막 켜진 번호 + 1(사이가 비어도)

    h = zero.copy()
    # 둘째 은닉층 뉴런 6으로 들어오는 가중치 중 첫 은닉층 뉴런 4(counts (1, 1)에서는 꺼짐)에서 오는 것: 켜진 좌표 밖이지만 본다
    h[frame.neuron_in(1, 6, (5, 1))[4]] = 2.0
    act = frame.active_neurons(h)
    assert np.flatnonzero(act[0]).tolist() == [4] and np.flatnonzero(act[1]).tolist() == [6]  # 보내는 쪽(나가는 가중치)도 켜짐
    assert frame.links((1, 1)) == 4 * 1 + 1 * 1 + 1 * 3

    m = zero.copy()
    m[frame.active_coords((2, 3))] = -0.0  # -0.0은 0(꺼진 좌표는 +0.0이어야 하지만 판단은 값으로)
    assert frame.counts_of(m) == ones and not any(a.any() for a in frame.active_neurons(m))
    m[layout.n_body - 1] = 1.0  # 몸 수치는 뉴런 켜짐과 무관
    m[layout.dim - 1] = 1.0  # 출력 편향도
    assert not any(a.any() for a in frame.active_neurons(m))


# ---------------------------------------------------------------------------
# 오류
# ---------------------------------------------------------------------------
def _korean(exc: pytest.ExceptionInfo[FrameError]) -> str:
    text = str(exc.value)
    assert HANGUL.search(text), text
    assert isinstance(exc.value, ValueError)
    return text


def test_no_hidden_layer_and_no_brain_are_refused_in_korean() -> None:
    with pytest.raises(FrameError) as exc:
        BrainFrame(_layout(9, (), 3))
    assert _korean(exc) == S.FRAME_NO_HIDDEN
    with pytest.raises(FrameError) as exc:
        BrainFrame(_layout(0, (), 0, n_body=3))
    assert _korean(exc) == S.FRAME_NO_BRAIN
    with pytest.raises(FrameError) as exc:  # 은닉층이 있어도 출력이 없으면 뇌 없는 환경
        BrainFrame(_layout(2, (4,), 0, n_body=1))
    assert _korean(exc) == S.FRAME_NO_BRAIN


def _builtin(name: str, edit: Any = None) -> GenomeLayout:
    data = json.loads((paths.builtin_env_dir() / name).read_text(encoding="utf-8-sig"))
    if edit is not None:
        data = copy.deepcopy(data)
        edit(data)
    return layout_for(compile_env(data))


def test_real_environments_without_hidden_layer_or_brain() -> None:
    with pytest.raises(FrameError) as exc:
        BrainFrame(_builtin("glider.evoenv"))  # 뇌 없이 몸만 진화(v2.0)
    assert _korean(exc) == S.FRAME_NO_BRAIN

    def no_hidden(d: dict[str, Any]) -> None:
        d["brain"]["hidden"] = []

    with pytest.raises(FrameError) as exc:
        BrainFrame(_builtin("lander.evoenv", no_hidden))  # 에디터 「뇌 설정」에서 은닉층을 모두 지운 착륙선
    assert _korean(exc) == S.FRAME_NO_HIDDEN


@pytest.mark.parametrize("counts", [(0,), (17,), (-1,), (1, 1), (), (2.0,), (True,), ("2",), 3, "2", None])
def test_bad_counts_are_refused_in_korean(counts: Any) -> None:
    frame = BrainFrame(FRAMES["one"])
    for call in (frame.active_coords, frame.links, lambda c: frame.neuron_in(0, 0, c), lambda c: frame.neuron_out(0, 0, c)):
        with pytest.raises(FrameError) as exc:
            call(counts)
        assert "[16]" in _korean(exc)


@pytest.mark.parametrize(("layer", "j"), [(0, 16), (0, -1), (1, 0), (-1, 0), (0, 1.0), (True, 0), (0, None)])
def test_bad_neurons_are_refused_in_korean(layer: Any, j: Any) -> None:
    frame = BrainFrame(FRAMES["one"])
    for call in (lambda: frame.bias(layer, j), lambda: frame.neuron_in(layer, j, (4,)), lambda: frame.neuron_out(layer, j, (4,))):
        with pytest.raises(FrameError) as exc:
            call()
        _korean(exc)


@pytest.mark.parametrize("shape", [(210,), (212,), (2, 211), (), (211, 1)])
def test_bad_genome_shapes_are_refused_in_korean(shape: tuple[int, ...]) -> None:
    frame = BrainFrame(FRAMES["one"])
    with pytest.raises(FrameError) as exc:
        frame.active_neurons(np.zeros(shape))
    assert "211" in _korean(exc)
    with pytest.raises(FrameError):
        frame.counts_of(np.zeros(shape))


# ---------------------------------------------------------------------------
# 내장 예제, 피클
# ---------------------------------------------------------------------------
def test_lander_growth_path_matches_the_design() -> None:
    """착륙선 9 → 16 → 3: 뉴런 2·4·8·16개일 때 켜진 좌표 29·55·107·211(= 지금 유전자 길이), 연결 24·48·96·192(설계 D30, 4.14)."""
    layout = _builtin("lander.evoenv")
    frame = BrainFrame(layout)
    assert frame.sizes == (16,)
    assert [frame.active_coords((c,)).size for c in (2, 4, 8, 16)] == [29, 55, 107, 211] and layout.dim == 211
    assert [frame.links((c,)) for c in (2, 4, 8, 16)] == [24, 48, 96, 192]
    assert frame.full_links() == 192


def test_every_builtin_brain_env_has_a_frame() -> None:
    seen = 0
    for path in paths.builtin_envs():
        layout = _builtin(path.name)
        if layout.brain.n_out == 0:
            with pytest.raises(FrameError):
                BrainFrame(layout)
            continue
        frame = BrainFrame(layout)
        seen += 1
        np.testing.assert_array_equal(frame.active_coords(frame.sizes), np.arange(layout.dim))
        start = frame.active_coords(tuple(min(2, s) for s in frame.sizes))
        assert np.all(start[: layout.n_body] == np.arange(layout.n_body))  # 몸 유전자(있음 유전자 포함)는 늘 켜짐
    assert seen >= 5  # 착륙선, 트랙, 언덕차, 점프, 바퀴 고르는 언덕차


def test_frame_pickles_compares_and_leaves_numpy_random_alone() -> None:
    before = np.random.get_state()
    frame = BrainFrame(_builtin("hill_car.evoenv"))  # 몸 수치 5개
    coords = frame.active_coords((4,))
    again = pickle.loads(pickle.dumps(frame))
    assert again == frame and hash(again) == hash(frame)
    np.testing.assert_array_equal(again.active_coords((4,)), coords)
    g = np.random.default_rng(9).standard_normal(frame.layout.dim)
    assert again.counts_of(g) == frame.counts_of(g)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]

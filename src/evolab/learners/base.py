"""학습 방법 규격 (명세: 공통 인터페이스, 학습 방법 규격 ask/tell).

모든 학습 방법(CMA-ES, 유전 알고리즘, 담금질)이 따르는 규격과 공통 도우미.

- 점수는 높을수록 좋다. 최소화하는 라이브러리(pycma)는 감싸는 쪽에서 부호를 뒤집는다.
- ask()는 (개체 수, 차원) float64 후보를 돌려주고, tell()은 그 후보와 같은 순서의 점수를 받는다.
- state_dict()는 그대로 피클해서 다른 프로세스에서 load_state_dict()로 불러와도 비트 단위로 같게 이어간다.
- 난수는 학습 방법마다 전용 생성기를 쓰고 상태에 함께 담는다(numpy 전역 난수 금지).
"""

from __future__ import annotations

from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

from evolab.app import strings as S

# 유효한 점수가 하나도 없을 때 모든 개체에 주는 점수(아주 나쁨, 모두 같음).
ALL_INVALID_SCORE: Final = -1e300


@runtime_checkable
class Learner(Protocol):
    name: str

    @property
    def popsize(self) -> int:
        """지금 개체 수(다음 ask()가 돌려줄 행 수). IPOP 재시작이 있으면 바뀔 수 있다."""
        ...

    def ask(self) -> np.ndarray:
        """(popsize, 차원) float64 후보."""
        ...

    def tell(self, candidates: np.ndarray, scores: np.ndarray) -> None:
        """마지막 ask()의 후보와 그 점수(높을수록 좋음)."""
        ...

    def best(self) -> np.ndarray:
        """학습 방법이 추천하는 해 (차원,). CMA-ES는 경계 안으로 옮긴 평균."""
        ...

    def stats(self) -> dict[str, float]:
        """그래프용 수치. 키 집합은 실행 내내 같다."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """이어 가기에 필요한 모든 것. 피클할 수 있고, 지금 상태의 사본이다.

        ask()와 tell() 사이에 불러도 된다(워커 오류로 세대 도중에 멈춘 Trainer가 저장한다). 그때는 그 ask() 직전
        상태를 돌려주고, 불러온 뒤 ask()는 같은 후보를 다시 낸다.
        """
        ...

    def load_state_dict(self, d: dict[str, Any]) -> None: ...


def sanitize_scores(scores: np.ndarray, count: int) -> np.ndarray:
    """점수를 (count,) float64 사본으로 바꾼다. 유한하지 않은 값(NaN, ±무한대)은 가장 나쁜 점수로 둔다.

    가장 나쁜 점수 = 유한한 점수의 최솟값 − 1 (값이 너무 커서 1을 빼도 같으면 최솟값 − |최솟값|).
    유한한 점수가 하나도 없으면 모두 ALL_INVALID_SCORE.
    """
    arr = np.asarray(scores)
    if arr.shape != (count,):
        raise ValueError(S.LEARNER_ERR_SCORES_SHAPE.format(got=tuple(arr.shape), count=count))
    out = np.array(arr, dtype=np.float64, copy=True)
    bad = ~np.isfinite(out)
    if bad.any():
        good = out[~bad]
        if good.size == 0:
            out[:] = ALL_INVALID_SCORE
        else:
            lo = float(good.min())
            worst = lo - 1.0
            if not worst < lo:  # |lo|가 2**53보다 커서 1을 빼도 그대로
                worst = max(lo - abs(lo), -float(np.finfo(np.float64).max))
            out[bad] = worst
    return out


# ---------------------------------------------------------------------------- 직접 구현한 방법(GA, 담금질)의 공통 도우미


def is_int(v: object) -> bool:
    """정수(bool 제외)인가."""
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def is_real(v: object) -> bool:
    """실수로 쓸 수 있는 숫자(bool 제외)인가. 유한한지는 따로 본다."""
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def initial_genome(x0: np.ndarray, bounded_dims: int) -> np.ndarray:
    """초기 유전자를 검사해 float64 사본으로 돌려준다. 앞쪽 bounded_dims차원(몸 유전자)은 [-1, 1]로 잘라낸다."""
    x = np.array(x0, dtype=np.float64, copy=True)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError(S.LEARNER_ERR_X0)
    if bounded_dims > x.size:
        raise ValueError(S.LEARNER_ERR_BOUNDED_DIMS_TOO_MANY.format(value=bounded_dims, dim=x.size))
    np.clip(x[:bounded_dims], -1.0, 1.0, out=x[:bounded_dims])
    return x


def clip_bounded(X: np.ndarray, bounded_dims: int) -> None:
    """(N, 차원) 배열의 앞쪽 bounded_dims차원을 그 자리에서 [-1, 1]로 잘라낸다."""
    if bounded_dims:
        np.clip(X[:, :bounded_dims], -1.0, 1.0, out=X[:, :bounded_dims])


def new_generator(seed: int) -> np.random.Generator:
    """학습 방법 전용 난수 생성기(PCG64). 상태는 generator_state()로 꺼내 피클한다."""
    return np.random.Generator(np.random.PCG64(seed))


def generator_state(rng: np.random.Generator) -> dict[str, Any]:
    """생성기 상태의 사본(정수와 문자열만 든 dict라 프로세스가 달라도 그대로 이어진다)."""
    state = rng.bit_generator.state
    return {**state, "state": dict(state["state"])}


def generator_from_state(state: Any) -> np.random.Generator:
    """generator_state()로 꺼낸 상태에서 생성기를 만든다. 잘못된 상태면 ValueError."""
    rng = new_generator(0)
    try:
        rng.bit_generator.state = state
    except (TypeError, ValueError, KeyError) as e:
        raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=e)) from e
    return rng


def check_tell(pending: np.ndarray | None, candidates: np.ndarray) -> None:
    """tell()에 넘긴 후보가 마지막 ask()의 후보와 같은지 본다. ask() 전이면 RuntimeError, 다르면 ValueError."""
    if pending is None:
        raise RuntimeError(S.LEARNER_ERR_TELL_WITHOUT_ASK)
    cand = np.asarray(candidates)
    if cand.shape != pending.shape or not np.array_equal(cand, pending):
        raise ValueError(S.LEARNER_ERR_TELL_CANDIDATES)


def state_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    """상태에서 읽은 배열을 검사해 float64 사본으로 돌려준다(모양이 다르거나 유한하지 않으면 ValueError)."""
    try:
        arr = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as e:
        raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=e)) from e
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=f"shape {arr.shape} != {shape}"))
    return arr

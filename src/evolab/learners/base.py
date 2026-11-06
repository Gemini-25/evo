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

"""자라는 뇌의 틀 (v6.1, v6 설계 4.14·7.3, D26).

틀 = 환경에서 만든 GenomeLayout 그대로다. 유전자는 늘 틀 길이(layout.dim)이고, 자라는 뇌의 학습 방법(learners/grow.py)은 켜진
좌표만 탐색하며 나머지는 정확히 0.0으로 둔다. 그래서 재생·기록·경주·판정·재현은 지금 코드 그대로 틀 전체를 계산한다.

유전자 배치(brain.unpack 순서): 몸 수치 n_body개, 그다음 층마다 W(fan_in × fan_out, 행 우선) 뒤 b(fan_out). 은닉층 l(0부터)은
신경망 층 l의 출력이다: 들어오는 가중치는 층 l의 W 열 j, 편향은 층 l의 b[j], 나가는 가중치는 층 l + 1의 W 행 j다.

- 켜진 뉴런은 층마다 앞쪽부터 채운다(뉴런 j가 켜졌으면 0..j−1도 켜짐). 상태는 은닉층마다 숫자 하나(counts).
- 켜진 좌표 = 몸 수치 전부 + 은닉층마다 켜진 뉴런의 (앞층의 켜진 뉴런 또는 모든 입력에서 들어오는 가중치, 편향)
  + 마지막 은닉층의 켜진 뉴런에서 모든 출력으로 나가는 가중치 + 출력 편향. 이웃한 은닉층 사이는 켜진 뉴런끼리만.
- active_neurons(유전자)는 학습 방법 상태 없이 유전자만 본다: 그 뉴런의 들어오는·나가는 가중치(틀 전체)와 편향 중 0이 아닌 것이
  있으면 켜짐. 기록·재생·경주·옛 프로그램이 쓴 실행에서도 같은 답이다.
- 은닉층이 없는 뇌(입력 → 출력)와 뇌 없는 환경(출력 없음, 글라이더)은 틀을 만들 수 없다(FrameError, 한국어).

이 모듈은 난수를 쓰지 않고 유전자를 고치지 않는다(읽기와 좌표 계산만).
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evolab.app import strings as S
from evolab.policy.genome import GenomeLayout


class FrameError(ValueError):
    """뇌 틀을 만들 수 없거나 켜진 뉴런 수·뉴런 자리·유전자 모양이 틀과 맞지 않는다. 메시지는 한국어(화면에 보일 수 있음)."""


@dataclass(frozen=True)
class BrainFrame:
    """환경 유전자 배치 위의 뇌 틀. 만들 때 은닉층이 있고 출력이 있는 뇌인지 확인한다."""

    layout: GenomeLayout

    def __post_init__(self) -> None:
        brain = self.layout.brain
        if brain.n_out == 0:  # 뇌 없는 환경(글라이더): 몸 수치만 진화한다(v2.0 설계 D20)
            raise FrameError(S.FRAME_NO_BRAIN)
        if not brain.hidden:
            raise FrameError(S.FRAME_NO_HIDDEN)

    # ---- 크기 ----

    @property
    def sizes(self) -> tuple[int, ...]:
        """틀 = 은닉층마다 다 자란 크기(layout.brain.hidden)."""
        return self.layout.brain.hidden

    @functools.cached_property
    def _offsets(self) -> tuple[tuple[int, int, int, int], ...]:
        """신경망 층마다 (W 시작 좌표, b 시작 좌표, fan_in, fan_out). 층 수 = 은닉층 수 + 1(출력층)."""
        out: list[tuple[int, int, int, int]] = []
        off = self.layout.n_body
        for fi, fo in self.layout.brain.shapes:
            out.append((off, off + fi * fo, fi, fo))
            off += fi * fo + fo
        if off != self.layout.dim:  # 배치 규칙(brain.unpack)과 어긋나면 틀이 틀린 좌표를 낸다
            raise AssertionError(f"frame offsets end at {off}, layout dim is {self.layout.dim}")
        return tuple(out)

    # ---- 검사 ----

    def _counts(self, counts: Sequence[int]) -> tuple[int, ...]:
        """counts를 층마다 정수 튜플로. 길이가 은닉층 수와 다르거나 1..그 층 크기 밖이면 FrameError."""
        sizes = self.sizes
        try:
            if isinstance(counts, (str, bytes)):
                raise TypeError("counts must be a sequence of integers")
            items = tuple(counts)
            if any(isinstance(c, (bool, np.bool_)) for c in items):
                raise TypeError("counts must be integers, not booleans")
            values = tuple(operator.index(c) for c in items)
        except TypeError:
            raise FrameError(S.FRAME_BAD_COUNTS.format(counts=_show(counts), layers=len(sizes), sizes=list(sizes))) from None
        if len(values) != len(sizes) or any(not 1 <= c <= s for c, s in zip(values, sizes, strict=False)):
            raise FrameError(S.FRAME_BAD_COUNTS.format(counts=list(values), layers=len(sizes), sizes=list(sizes)))
        return values

    def _neuron(self, layer: int, j: int) -> tuple[int, int]:
        sizes = self.sizes
        try:
            if isinstance(layer, (bool, np.bool_)) or isinstance(j, (bool, np.bool_)):
                raise TypeError("layer and neuron must be integers")
            li, ji = operator.index(layer), operator.index(j)
        except TypeError:
            raise FrameError(S.FRAME_BAD_NEURON.format(layer=_show(layer), j=_show(j), sizes=list(sizes))) from None
        if not 0 <= li < len(sizes) or not 0 <= ji < sizes[li]:
            # 사용자에게 보이는 번호는 1부터(은닉층 1의 1번째 뉴런)
            raise FrameError(S.FRAME_BAD_NEURON.format(layer=li + 1, j=ji + 1, sizes=list(sizes)))
        return li, ji

    def _genome(self, genome: np.ndarray) -> np.ndarray:
        g = np.asarray(genome)
        if g.shape != (self.layout.dim,):
            raise FrameError(S.FRAME_GENOME_SHAPE.format(shape=tuple(g.shape), dim=self.layout.dim))
        return g

    # ---- 좌표 ----

    def active_coords(self, counts: Sequence[int]) -> np.ndarray:
        """켜진 좌표(유전자 번호, 오름차순 np.intp). 1 ≤ counts[l] ≤ sizes[l]."""
        c = self._counts(counts)
        mask = np.zeros(self.layout.dim, dtype=bool)
        mask[: self.layout.n_body] = True
        last = len(c)  # 출력층 번호
        for k, (w_off, b_off, fi, fo) in enumerate(self._offsets):
            block = np.zeros((fi, fo), dtype=bool)
            rows = fi if k == 0 else c[k - 1]  # 들어오는 쪽: 입력 전부, 또는 앞 은닉층의 켜진 뉴런
            cols = fo if k == last else c[k]  # 받는 쪽: 출력 전부, 또는 이 은닉층의 켜진 뉴런
            block[:rows, :cols] = True
            mask[w_off:b_off] = block.ravel()
            mask[b_off : b_off + cols] = True
        return np.flatnonzero(mask).astype(np.intp, copy=False)

    def neuron_in(self, layer: int, j: int, counts: Sequence[int]) -> np.ndarray:
        """은닉층 layer의 뉴런 j로 들어오는 가중치 좌표(오름차순 np.intp): 첫 은닉층은 입력 전부, 그 밖에는 앞 은닉층의 켜진 뉴런.

        j가 counts 밖이어도 된다(자랄 때 새 뉴런의 들어오는 가중치를 새 counts로 구한다). 들어오는 쪽만 counts를 본다.
        """
        c = self._counts(counts)
        li, ji = self._neuron(layer, j)
        w_off, _b_off, fi, fo = self._offsets[li]
        rows = fi if li == 0 else c[li - 1]
        return (w_off + np.arange(rows, dtype=np.intp) * fo + ji).astype(np.intp, copy=False)

    def neuron_out(self, layer: int, j: int, counts: Sequence[int]) -> np.ndarray:
        """은닉층 layer의 뉴런 j에서 나가는 가중치 좌표(오름차순 np.intp): 다음 은닉층의 켜진 뉴런, 마지막 은닉층은 출력 전부."""
        c = self._counts(counts)
        li, ji = self._neuron(layer, j)
        w_off, _b_off, _fi, fo = self._offsets[li + 1]
        cols = fo if li == len(c) - 1 else c[li + 1]
        return (w_off + ji * fo + np.arange(cols, dtype=np.intp)).astype(np.intp, copy=False)

    def bias(self, layer: int, j: int) -> int:
        """은닉층 layer의 뉴런 j의 편향 좌표."""
        li, ji = self._neuron(layer, j)
        return self._offsets[li][1] + ji

    # ---- 유전자에서 읽기 ----

    def active_neurons(self, genome: np.ndarray) -> list[np.ndarray]:
        """층마다 (sizes[l],) bool: 그 뉴런의 들어오는·나가는 가중치(틀 전체)와 편향 중 0이 아닌 것이 있나. -0.0은 0으로 본다."""
        g = self._genome(genome)
        out: list[np.ndarray] = []
        for li in range(len(self.sizes)):
            w_off, b_off, fi, fo = self._offsets[li]
            n_off, n_b, nfi, nfo = self._offsets[li + 1]
            incoming = (g[w_off:b_off].reshape(fi, fo) != 0).any(axis=0)
            biased = g[b_off : b_off + fo] != 0
            outgoing = (g[n_off:n_b].reshape(nfi, nfo) != 0).any(axis=1)
            out.append(np.asarray(incoming | biased | outgoing, dtype=bool))
        return out

    def counts_of(self, genome: np.ndarray) -> tuple[int, ...]:
        """층마다 「마지막 켜진 뉴런 번호 + 1」(하나도 없으면 1). 켜진 뉴런이 앞쪽부터 채워진 유전자면 그 층의 counts와 같다."""
        out: list[int] = []
        for act in self.active_neurons(genome):
            on = np.flatnonzero(act)
            out.append(int(on[-1]) + 1 if on.size else 1)
        return tuple(out)

    def links(self, counts: Sequence[int]) -> int:
        """켜진 가중치 수(편향 제외): 입력 × 첫 층 + 이웃 은닉층의 켜진 뉴런끼리 + 마지막 층 × 출력."""
        c = self._counts(counts)
        brain = self.layout.brain
        return brain.n_in * c[0] + sum(a * b for a, b in zip(c[:-1], c[1:], strict=True)) + c[-1] * brain.n_out

    def full_links(self) -> int:
        """다 자란 틀의 가중치 수(편향 제외)."""
        return self.links(self.sizes)


def _show(value: object) -> str:
    """오류 문구용 짧은 표현."""
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."

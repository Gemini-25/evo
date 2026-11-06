"""고정 구조 신경망(MLP)을 개체 여러 개 묶어서 계산한다.

학습(워커), 재생(화면), 개체 클릭 검사가 모두 BatchBrain.forward 하나를 쓴다. 계산 경로가 다르면
재생 결과가 달라지기 때문이다.

결정성: forward의 결과 i행은 i행의 가중치와 입력만의 함수다. 묶음 크기, 행 순서, 다른 행의 값과
상관없이 비트 단위로 같다. 층마다 (N, 1, 입력) x (N, 입력, 출력) 묶음 행렬 곱 한 번(개체마다 따로
계산되는 곱)과 원소별 연산(편향 더하기, 활성화)만 쓰므로 행끼리 섞이는 계산이 없다.
tests/test_policy.py가 N=1 계산, 다른 묶음 크기, 섞은 순서와 비교해 확인한다.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np

from evolab.app import strings as S

Activation = Literal["tanh", "relu"]
ACTIVATIONS: tuple[str, ...] = get_args(Activation)


class BrainSizeError(ValueError):
    """뇌 파라미터 수가 신경망 구조와 맞지 않는다. 메시지는 한국어(화면에 보일 수 있음)."""


@dataclass(frozen=True)
class BrainLayout:
    """신경망 구조. 은닉층 활성화는 activation, 출력층은 늘 tanh(-1..1)."""

    n_in: int
    hidden: tuple[int, ...]
    n_out: int
    activation: Activation = "tanh"

    def __post_init__(self) -> None:
        # JSON에서 온 list도 받는다(얼린 dataclass라 object.__setattr__로 고친다).
        # operator.index: 정수(파이썬·numpy)만 받고 9.7 같은 실수를 몰래 9로 자르지 않는다.
        object.__setattr__(self, "hidden", tuple(operator.index(h) for h in self.hidden))
        object.__setattr__(self, "n_in", operator.index(self.n_in))
        object.__setattr__(self, "n_out", operator.index(self.n_out))
        if self.n_in < 0 or self.n_out < 0:
            raise ValueError(f"n_in and n_out must be >= 0, got {self.n_in}, {self.n_out}")
        if any(h < 1 for h in self.hidden):
            raise ValueError(f"hidden layer sizes must be >= 1, got {self.hidden}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {self.activation!r}")

    @property
    def sizes(self) -> tuple[int, ...]:
        """층별 뉴런 수: (입력, 은닉..., 출력)."""
        return (self.n_in, *self.hidden, self.n_out)

    @property
    def shapes(self) -> list[tuple[int, int]]:
        """층마다 (fan_in, fan_out)."""
        s = self.sizes
        return [(s[k], s[k + 1]) for k in range(len(s) - 1)]

    @property
    def n_params(self) -> int:
        return sum(fi * fo + fo for fi, fo in self.shapes)

    def describe(self) -> str:
        """'9 → 16 → 3, tanh' 형태. 보고와 오류 문구용."""
        return " → ".join(str(n) for n in self.sizes) + f", {self.activation}"


@dataclass(eq=False)  # 배열 필드라 자동 ==는 오류를 낸다. 같은 객체인지로만 비교한다.
class BatchBrain:
    """신경망 N개의 가중치. 한 번 펼쳐 두고, 층마다 연속(C-contiguous) float64 배열로 둔다."""

    layout: BrainLayout
    weights: list[np.ndarray]
    """층 k: (N, fan_in, fan_out)."""
    biases: list[np.ndarray]
    """층 k: (N, fan_out)."""

    def __post_init__(self) -> None:
        shapes = self.layout.shapes
        if len(self.weights) != len(shapes) or len(self.biases) != len(shapes):
            raise ValueError(f"expected {len(shapes)} layers, got {len(self.weights)} weights / {len(self.biases)} biases")
        self.weights = [np.ascontiguousarray(w, dtype=np.float64) for w in self.weights]
        self.biases = [np.ascontiguousarray(b, dtype=np.float64) for b in self.biases]
        n = self.weights[0].shape[0] if self.weights[0].ndim == 3 else -1
        for (fi, fo), w, b in zip(shapes, self.weights, self.biases, strict=True):
            if w.shape != (n, fi, fo) or b.shape != (n, fo):
                raise ValueError(f"layer shape mismatch: weights {w.shape}, biases {b.shape}, want ({n}, {fi}, {fo})")

    @property
    def n(self) -> int:
        """신경망 수."""
        return int(self.weights[0].shape[0])

    def forward(self, x: np.ndarray, trace: list[np.ndarray] | None = None) -> np.ndarray:
        """x: (N, n_in) 실수 배열 -> (N, n_out) float64, 범위 -1..1.

        층마다 h = matmul(h[:, None, :], W)[:, 0, :] + b, 은닉층은 activation, 마지막 층은 tanh.
        trace를 주면 층마다 활성화 값의 사본을 덧붙인다(화면의 뉴런 표시용, 계산은 같다).
        """
        h = np.ascontiguousarray(x, dtype=np.float64)
        if h.shape != (self.n, self.layout.n_in):
            raise ValueError(f"input shape {h.shape} does not match ({self.n}, {self.layout.n_in})")
        last = len(self.weights) - 1
        relu = self.layout.activation == "relu"
        for k, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            h = np.matmul(h[:, None, :], w)[:, 0, :]  # (N, fan_out), 새로 만든 연속 배열의 보기
            h += b
            if relu and k < last:
                np.maximum(h, 0.0, out=h)
            else:
                np.tanh(h, out=h)
            if trace is not None:
                trace.append(h.copy())
        return h

    def select(self, rows: np.ndarray) -> BatchBrain:
        """rows번째 신경망들만 뽑은 새 묶음(복사, 연속 배열). 같은 번호를 여러 번 써도 된다.

        rows: 정수 번호 배열(번호 하나면 N=1) 또는 길이 N의 bool 마스크.
        """
        idx = np.asarray(rows)
        if idx.dtype != np.bool_:
            if idx.size and idx.dtype.kind not in "iu":
                raise TypeError(f"rows must be integer indices or a bool mask, got dtype {idx.dtype}")
            idx = np.atleast_1d(idx.astype(np.intp, copy=False))
        return BatchBrain(
            self.layout,
            [np.ascontiguousarray(w[idx]) for w in self.weights],
            [np.ascontiguousarray(b[idx]) for b in self.biases],
        )


def unpack(layout: BrainLayout, flat: np.ndarray) -> BatchBrain:
    """뇌 파라미터 (N, n_params)를 층별 가중치로 펼친다. 1차원 (n_params,)는 N=1로 본다.

    층마다 W를 행 우선(fan_in x fan_out)으로, 이어서 b(fan_out)를 담는다.
    가중치는 언제나 새로 복사한다: N=1이어도 flat과 메모리를 같이 쓰지 않으므로, 나중에 flat을
    고쳐도 이미 펼친 뇌는 바뀌지 않는다.
    """
    arr = np.asarray(flat, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"brain parameters must be (N, {layout.n_params}) or ({layout.n_params},), got {arr.shape}")
    if arr.shape[1] != layout.n_params:
        raise BrainSizeError(S.POLICY_BRAIN_SIZE.format(got=arr.shape[1], want=layout.n_params, shape=layout.describe()))
    n = arr.shape[0]
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    off = 0
    for fi, fo in layout.shapes:
        weights.append(np.array(arr[:, off : off + fi * fo], order="C", copy=True).reshape(n, fi, fo))
        off += fi * fo
        biases.append(np.array(arr[:, off : off + fo], order="C", copy=True))
        off += fo
    return BatchBrain(layout, weights, biases)


def pack(brain: BatchBrain) -> np.ndarray:
    """unpack의 역. (N, n_params) float64 연속 배열."""
    n = brain.n
    parts: list[np.ndarray] = []
    for w, b in zip(brain.weights, brain.biases, strict=True):
        parts.append(w.reshape(n, w.shape[1] * w.shape[2]))  # -1은 N=0일 때 모양을 못 정한다
        parts.append(b)
    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float64)

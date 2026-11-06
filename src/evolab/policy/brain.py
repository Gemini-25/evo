"""고정 구조 신경망(MLP)을 개체 여러 개 묶어서 계산한다.

학습(워커), 재생(화면), 개체 클릭 검사가 모두 BatchBrain.forward 하나를 쓴다. 계산 경로가 다르면
재생 결과가 달라지기 때문이다.

결정성: forward의 결과 i행은 i행의 가중치와 입력만의 함수다. 묶음 크기, 행 순서, 다른 행의 값과
상관없이 비트 단위로 같다. 층마다 (N, 1, 입력) x (N, 입력, 출력) 묶음 행렬 곱 한 번(개체마다 따로
계산되는 곱)과 원소별 연산(편향 더하기, 활성화)만 쓰므로 행끼리 섞이는 계산이 없다.
tests/test_policy.py가 N=1 계산, 다른 묶음 크기, 섞은 순서와 비교해 확인한다.
"""

from __future__ import annotations

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
        object.__setattr__(self, "hidden", tuple(int(h) for h in self.hidden))
        object.__setattr__(self, "n_in", int(self.n_in))
        object.__setattr__(self, "n_out", int(self.n_out))
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


@dataclass
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

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, n_in) 실수 배열 -> (N, n_out) float64, 범위 -1..1.

        층마다 h = matmul(h[:, None, :], W)[:, 0, :] + b, 은닉층은 activation, 마지막 층은 tanh.
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
        return h

    def select(self, rows: np.ndarray) -> BatchBrain:
        """rows번째 신경망들만 뽑은 새 묶음(복사, 연속 배열). 같은 번호를 여러 번 써도 된다."""
        idx = np.asarray(rows)
        if idx.dtype != np.bool_:
            idx = idx.astype(np.intp, copy=False)
        return BatchBrain(
            self.layout,
            [np.ascontiguousarray(w[idx]) for w in self.weights],
            [np.ascontiguousarray(b[idx]) for b in self.biases],
        )


def unpack(layout: BrainLayout, flat: np.ndarray) -> BatchBrain:
    """뇌 파라미터 (N, n_params)를 층별 가중치로 펼친다. 1차원 (n_params,)는 N=1로 본다.

    층마다 W를 행 우선(fan_in x fan_out)으로, 이어서 b(fan_out)를 담는다.
    """
    arr = np.asarray(flat, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != layout.n_params:
        got = arr.shape[-1] if arr.ndim else 0
        raise BrainSizeError(S.POLICY_BRAIN_SIZE.format(got=got, want=layout.n_params, shape=layout.describe()))
    n = arr.shape[0]
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    off = 0
    for fi, fo in layout.shapes:
        weights.append(np.ascontiguousarray(arr[:, off : off + fi * fo]).reshape(n, fi, fo))
        off += fi * fo
        biases.append(np.ascontiguousarray(arr[:, off : off + fo]))
        off += fo
    return BatchBrain(layout, weights, biases)


def pack(brain: BatchBrain) -> np.ndarray:
    """unpack의 역. (N, n_params) float64 연속 배열."""
    n = brain.n
    parts: list[np.ndarray] = []
    for w, b in zip(brain.weights, brain.biases, strict=True):
        parts.append(w.reshape(n, -1))
        parts.append(b)
    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float64)

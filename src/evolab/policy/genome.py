"""유전자 벡터 <-> 몸 수치와 뇌 가중치.

유전자는 float64 벡터 하나다. 앞쪽 n_body개는 몸 수치(유전자 공간 -1..1), 뒤쪽은 뇌 파라미터
(brain.unpack의 순서). 차원 = 몸 수치 개수 + 뇌 파라미터 수.

signature()는 유전자의 "뜻"을 정하는 것(몸 유전자 id와 범위, 신경망 구조, 입력·출력 채널 이름과
출력 범위)만으로 만든 해시다. 환경을 고친 뒤 이어하기가 되는지(같은 유전자가 같은 뜻인지) 판단할 때 쓴다.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from evolab.app import strings as S
from evolab.policy.brain import ACTIVATIONS, BrainLayout
from evolab.spec import symbols as sy
from evolab.spec.compiled import EnvSpec

LAYOUT_FORMAT = 1
"""to_dict와 signature의 형식 번호. 파라미터 순서나 사상 규칙을 바꾸면 올린다."""


class GenomeError(ValueError):
    """유전자 길이나 구조 정보가 맞지 않는다. 메시지는 한국어(화면에 보일 수 있음)."""


def _num(x: float) -> float:
    """해시·저장용 실수: -0.0을 0.0으로, 유한하지 않으면 거부."""
    v = float(x) + 0.0
    if not math.isfinite(v):
        raise ValueError(f"value must be finite, got {x!r}")
    return v


@dataclass(frozen=True)
class BodyGene:
    """진화하는 몸 수치 하나. 물리 단위의 범위와 초깃값."""

    id: str
    low: float
    high: float
    initial: float

    def __post_init__(self) -> None:
        for name in ("low", "high", "initial"):
            object.__setattr__(self, name, _num(getattr(self, name)))
        if not self.low < self.high:
            raise ValueError(f"body gene {self.id!r}: low must be < high, got {self.low}, {self.high}")
        if not self.low <= self.initial <= self.high:
            raise ValueError(f"body gene {self.id!r}: initial {self.initial} outside [{self.low}, {self.high}]")
        if not math.isfinite(self.high - self.low):
            # 환경 검사는 유한한 최솟값·최댓값만 보므로 -1e308..1e308 같은 범위가 들어올 수 있다.
            # 폭이 무한대가 되면 body_values와 initial이 NaN을 낸다.
            raise GenomeError(S.POLICY_BODY_RANGE_HUGE.format(id=self.id, low=self.low, high=self.high))


@dataclass(frozen=True)
class GenomeLayout:
    """유전자 벡터의 구조. 몸 유전자가 앞, 뇌 파라미터가 뒤."""

    body: tuple[BodyGene, ...]
    """evolve=True인 매개변수, 선언 순서."""
    brain: BrainLayout
    output_ranges: tuple[tuple[float, float], ...]
    """뇌 출력 채널마다 구동기 고유 범위 (low, high): (0, 1) 또는 (-1, 1)."""
    inputs: tuple[str, ...] = ()
    """뇌 입력 채널 이름(순서대로). 비우면 이름 없음."""
    outputs: tuple[str, ...] = ()
    """뇌 출력 채널 이름(순서대로). 비우면 이름 없음."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "body", tuple(self.body))
        object.__setattr__(self, "output_ranges", tuple((_num(lo), _num(hi)) for lo, hi in self.output_ranges))
        object.__setattr__(self, "inputs", tuple(str(n) for n in self.inputs))
        object.__setattr__(self, "outputs", tuple(str(n) for n in self.outputs))
        ids = [g.id for g in self.body]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate body gene ids: {ids}")
        if len(self.output_ranges) != self.brain.n_out:
            raise ValueError(f"{len(self.output_ranges)} output ranges for {self.brain.n_out} brain outputs")
        if any(not lo < hi for lo, hi in self.output_ranges):
            raise ValueError(f"output ranges must have low < high: {self.output_ranges}")
        if self.inputs and len(self.inputs) != self.brain.n_in:
            raise ValueError(f"{len(self.inputs)} input names for {self.brain.n_in} brain inputs")
        if self.outputs and len(self.outputs) != self.brain.n_out:
            raise ValueError(f"{len(self.outputs)} output names for {self.brain.n_out} brain outputs")

    # ---- 크기 ----

    @property
    def n_body(self) -> int:
        return len(self.body)

    @property
    def dim(self) -> int:
        return self.n_body + self.brain.n_params

    # ---- 미리 계산한 배열 (얼린 dataclass라도 cached_property는 인스턴스 __dict__에 바로 쓴다) ----

    @functools.cached_property
    def _body_low(self) -> np.ndarray:
        return np.array([g.low for g in self.body], dtype=np.float64)

    @functools.cached_property
    def _body_high(self) -> np.ndarray:
        return np.array([g.high for g in self.body], dtype=np.float64)

    @functools.cached_property
    def _action_plan(self) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(종류, (0, 1) 열, 그 밖의 범위 열, 그 열들의 low, high).

        종류: 모든 열이 (0, 1)이면 'unit', 모두 (-1, 1)이면 'sign', 섞였으면 'mixed'.
        (-1, 1) 열은 값을 그대로 둔다(low + (y + 1) / 2 * 2로 계산하면 작은 값의 비트가 사라진다).
        """
        n = len(self.output_ranges)
        unit = [k for k, r in enumerate(self.output_ranges) if r == (0.0, 1.0)]
        sign = [k for k, r in enumerate(self.output_ranges) if r == (-1.0, 1.0)]
        other = [k for k in range(n) if k not in unit and k not in sign]
        kind = "unit" if len(unit) == n else ("sign" if len(sign) == n else "mixed")
        lo = np.array([self.output_ranges[k][0] for k in other], dtype=np.float64)
        hi = np.array([self.output_ranges[k][1] for k in other], dtype=np.float64)
        return kind, np.array(unit, dtype=np.intp), np.array(other, dtype=np.intp), lo, hi

    # ---- 유전자 <-> 몸·뇌 ----

    def _check(self, genomes: np.ndarray) -> np.ndarray:
        g = np.asarray(genomes, dtype=np.float64)
        if g.ndim not in (1, 2):
            raise ValueError(f"genomes must be (N, {self.dim}) or ({self.dim},), got {g.shape}")
        if g.shape[-1] != self.dim:
            raise GenomeError(
                S.POLICY_GENOME_SIZE.format(got=g.shape[-1], want=self.dim, body=self.n_body, brain=self.brain.n_params)
            )
        return g

    def initial(self) -> np.ndarray:
        """초기 유전자 (dim,): 몸 수치는 초깃값을 -1..1로 옮긴 값, 뇌 가중치는 0."""
        out = np.zeros(self.dim, dtype=np.float64)
        if self.body:
            init = np.array([g.initial for g in self.body], dtype=np.float64)
            low, high = self._body_low, self._body_high
            out[: self.n_body] = np.clip(2.0 * (init - low) / (high - low) - 1.0, -1.0, 1.0)
        return out

    def body_values(self, genomes: np.ndarray) -> np.ndarray:
        """(N, dim) -> (N, n_body) 물리 단위 몸 수치. 1차원 (dim,)이면 (n_body,).

        유전자를 -1..1로 자르고 low + (g + 1) / 2 * (high - low). 반올림으로 범위를 넘지 않게 한 번 더 자른다.
        NaN은 자를 곳이 없으므로 GenomeError(결과는 언제나 [low, high] 안).
        """
        g = self._check(genomes)
        low, high = self._body_low, self._body_high
        raw = g[..., : self.n_body]
        bad = np.isnan(raw)
        if bad.any():
            k = int(np.argwhere(bad)[0][-1])
            raise GenomeError(S.POLICY_GENOME_NAN.format(gene=self.body[k].id))
        x = np.clip(raw, -1.0, 1.0)
        v = low + (x + 1.0) / 2.0 * (high - low)
        return np.ascontiguousarray(np.clip(v, low, high))

    def brain_params(self, genomes: np.ndarray) -> np.ndarray:
        """(N, dim) -> (N, n_params) 뇌 파라미터(복사, 연속 배열). 1차원 (dim,)이면 (n_params,)."""
        g = self._check(genomes)
        return np.array(g[..., self.n_body :], dtype=np.float64, order="C", copy=True)

    def to_actions(self, raw: np.ndarray) -> np.ndarray:
        """뇌 출력(tanh, -1..1) (N, n_out) -> 구동기 고유 범위. (0, 1) 채널은 (y + 1) / 2, (-1, 1)은 그대로."""
        y = np.asarray(raw, dtype=np.float64)
        if y.ndim not in (1, 2) or y.shape[-1] != self.brain.n_out:
            raise ValueError(f"brain outputs must be (N, {self.brain.n_out}) or ({self.brain.n_out},), got {y.shape}")
        kind, unit, other, lo, hi = self._action_plan
        if kind == "unit":
            return (y + 1.0) / 2.0
        out = np.array(y, dtype=np.float64, order="C", copy=True)
        if kind == "sign":
            return out
        if unit.size:
            out[..., unit] = (y[..., unit] + 1.0) / 2.0
        if other.size:
            out[..., other] = lo + (y[..., other] + 1.0) / 2.0 * (hi - lo)
        return out

    # ---- 서명과 저장 ----

    def _canonical(self) -> dict[str, Any]:
        """유전자의 뜻을 정하는 것만. 몸 수치 초깃값은 뜻을 바꾸지 않으므로 뺀다."""
        return {
            "format": LAYOUT_FORMAT,
            "body": [[g.id, _num(g.low), _num(g.high)] for g in self.body],
            "brain": {
                "n_in": self.brain.n_in,
                "hidden": list(self.brain.hidden),
                "n_out": self.brain.n_out,
                "activation": self.brain.activation,
                "output": "tanh",
            },
            "output_ranges": [[_num(lo), _num(hi)] for lo, hi in self.output_ranges],
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
        }

    def signature(self) -> str:
        """구조의 sha256 16진 문자열. 프로세스, 실행, PYTHONHASHSEED와 상관없이 같다."""
        text = json.dumps(self._canonical(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        return hashlib.sha256(text.encode("ascii")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """JSON으로 쓸 수 있는 dict (run.json용)."""
        return {
            "format": LAYOUT_FORMAT,
            "body": [{"id": g.id, "low": g.low, "high": g.high, "initial": g.initial} for g in self.body],
            "brain": {
                "n_in": self.brain.n_in,
                "hidden": list(self.brain.hidden),
                "n_out": self.brain.n_out,
                "activation": self.brain.activation,
            },
            "output_ranges": [[lo, hi] for lo, hi in self.output_ranges],
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GenomeLayout:
        """to_dict의 역. 형식이 잘못되면 GenomeError(한국어)."""
        try:
            if not isinstance(d, Mapping):
                raise TypeError(f"expected a mapping, got {type(d).__name__}")
            fmt = d.get("format", LAYOUT_FORMAT)
            if fmt != LAYOUT_FORMAT:
                raise ValueError(f"unknown format {fmt!r}")
            b = d["brain"]
            activation = b.get("activation", "tanh")
            if activation not in ACTIVATIONS:
                raise ValueError(f"unknown activation {activation!r}")
            brain = BrainLayout(
                n_in=_int(b["n_in"]),
                hidden=tuple(_int(h) for h in b["hidden"]),
                n_out=_int(b["n_out"]),
                activation=activation,
            )
            body = tuple(
                BodyGene(id=str(g["id"]), low=g["low"], high=g["high"], initial=g["initial"]) for g in d["body"]
            )
            ranges: list[tuple[float, float]] = []
            for r in d["output_ranges"]:
                if isinstance(r, str) or len(r) != 2:
                    raise ValueError(f"output range must be [low, high], got {r!r}")
                ranges.append((r[0], r[1]))
            return cls(
                body=body,
                brain=brain,
                output_ranges=tuple(ranges),
                inputs=_names(d.get("inputs", [])),
                outputs=_names(d.get("outputs", [])),
            )
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            raise GenomeError(S.POLICY_LAYOUT_BROKEN.format(detail=f"{type(exc).__name__}: {exc}")) from None


def _int(x: Any) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise TypeError(f"expected an integer, got {x!r}")
    return x


def _names(x: Any) -> tuple[str, ...]:
    if isinstance(x, str) or not isinstance(x, Sequence) or not all(isinstance(n, str) for n in x):
        raise TypeError(f"expected a list of names, got {x!r}")
    return tuple(x)


def layout_for(spec: EnvSpec) -> GenomeLayout:
    """환경의 유전자 구조. 몸 유전자는 evolve=True 매개변수(선언 순서), 뇌는 spec.inputs -> hidden -> spec.outputs."""
    body: list[BodyGene] = []
    for p in sy.evolved_params(spec.file):
        if p.min is None or p.max is None:  # 검사(EnvSpec.build)에서 이미 걸러진다
            raise ValueError(f"evolved param {p.id!r} has no range")
        body.append(BodyGene(id=p.id, low=p.min, high=p.max, initial=p.value))
    brain = BrainLayout(
        n_in=len(spec.inputs),
        hidden=tuple(spec.file.brain.hidden),
        n_out=len(spec.outputs),
        activation=spec.file.brain.activation,
    )
    ranges = tuple((spec.actuator_channels[name].low, spec.actuator_channels[name].high) for name in spec.outputs)
    return GenomeLayout(
        body=tuple(body),
        brain=brain,
        output_ranges=ranges,
        inputs=tuple(spec.inputs),
        outputs=tuple(spec.outputs),
    )

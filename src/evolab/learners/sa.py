"""담금질 학습 방법: 직접 구현 (명세: 학습 방법, 유전자 벡터, 검토 반영 결정 1·2).

코어를 다 쓰려고 한 단계에 이웃 여러 개를 동시에 평가하고, 그중 최고 이웃에 수락 판정을 한다.

한 단계:
1. ask()는 [현재해, 이웃 1..K]를 돌려준다(K = neighbors, 행 수 K + 1). 이웃 = 현재해 + step × N(0, 1)(모든 유전자),
   몸 유전자(앞쪽 bounded_dims차원)는 [-1, 1]을 넘으면 잘라낸다. 현재해는 매 단계 다시 평가한다: 공통 시드에서는
   단계마다 시험 조건이 바뀌므로 지난 단계 점수를 들고 가지 않고 이번 단계 점수끼리 비교한다.
2. tell()은 이웃 중 점수가 가장 높은 것(동점이면 앞쪽)을 고른다. 이번 단계 현재해 점수 이상이면 수락하고,
   더 나쁘면 확률 exp((이웃 점수 − 현재해 점수) / T)로 수락한다. 그 뒤 T에 cooling을 곱한다.
3. 초기 온도가 비어 있으면(None) 첫 tell()에서 첫 이웃들 점수의 표준편차로 정한다(현재해 점수는 빼고).
   온도가 0이면 더 나쁜 이웃은 받지 않는다(같은 점수는 받는다).

- 수락 판정의 균등 난수는 결과와 상관없이 단계마다 한 번, 다음 이웃보다 먼저 같은 전용 Generator(PCG64)에서 뽑는다.
- 다음 후보(현재해와 이웃)는 tell()에서 바로 만들어 두므로 ask()는 난수를 쓰지 않는다. 그래서 ask()와 tell() 사이의
  state_dict()도 그 단계 경계 상태이고, 불러온 뒤 ask()는 같은 후보를 낸다.
- best()는 현재해.
- 구조 뒤집기(v2.1, structure_flip > 0): 이웃을 만든 뒤(현재해 제외) 이웃 순서로, 있음 유전자(structure_genes)마다 확률
  structure_flip으로 부호를 뒤집어 ∓0.5에 둔다(learners/structure.flip_structure). 확률이 0이면 난수를 쓰지 않아 v2.0과
  비트 단위로 같고, 상태(state_dict)의 설정에 structure_* 키도 쓰지 않는다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Final

import numpy as np

from evolab.app import strings as S
from evolab.learners.base import (
    check_tell,
    clip_bounded,
    generator_from_state,
    generator_state,
    initial_genome,
    is_int,
    is_real,
    new_generator,
    sanitize_scores,
    state_array,
)
from evolab.learners.structure import check_structure_config, config_for_state, flip_structure

STATE_FORMAT: Final = 1
ACCEPTANCE_WINDOW: Final = 50
"""stats()의 수락 비율은 최근 이만큼 단계에서 센다."""
_SEED_LIMIT: Final = 2**32


@dataclass(frozen=True)
class SAConfig:
    neighbors: int = 8
    """한 단계에 평가하는 이웃 수(워커 수와 무관한 고정값). 한 단계 후보 수는 이웃 수 + 1(현재해)."""
    step: float = 0.1
    """이웃 폭: 현재해에 더하는 가우시안 잡음의 표준편차."""
    initial_temperature: float | None = None
    """None이면 첫 이웃들 점수의 표준편차."""
    cooling: float = 0.995
    """단계마다 온도에 곱하는 값(0 초과 1 이하)."""
    seed: int = 1
    """학습 방법 전용 난수 생성기 시드(0 ≤ seed < 2**32). Trainer가 실행 시드에서 만든다."""
    bounded_dims: int = 0
    """앞쪽 k차원(몸 유전자)은 [-1, 1]을 넘으면 잘라낸다. 나머지(뇌 가중치)는 경계 없음."""
    structure_flip: float = 0.0
    """구조 뒤집기 확률(v2.1): 이웃마다, 있음 유전자마다 이 확률로 부호를 뒤집는다. 0이면 끔(난수를 쓰지 않음)."""
    structure_genes: tuple[int, ...] = ()
    """있음 유전자 번호(GenomeLayout.on_genes, 몸 유전자 안). 상태에서 읽은 목록은 tuple로 바꾼다."""

    def __post_init__(self) -> None:
        if not is_int(self.neighbors) or self.neighbors < 1:
            raise ValueError(S.LEARNER_SA_ERR_NEIGHBORS.format(value=self.neighbors))
        if not is_real(self.step) or not math.isfinite(self.step) or self.step <= 0:
            raise ValueError(S.LEARNER_SA_ERR_STEP.format(value=self.step))
        t = self.initial_temperature
        if t is not None and (not is_real(t) or not math.isfinite(t) or t < 0):
            raise ValueError(S.LEARNER_SA_ERR_TEMPERATURE.format(value=t))
        if not is_real(self.cooling) or not 0.0 < float(self.cooling) <= 1.0:
            raise ValueError(S.LEARNER_SA_ERR_COOLING.format(value=self.cooling))
        if not is_int(self.seed) or not 0 <= self.seed < _SEED_LIMIT:
            raise ValueError(S.LEARNER_ERR_SEED.format(value=self.seed))
        if not is_int(self.bounded_dims) or self.bounded_dims < 0:
            raise ValueError(S.LEARNER_ERR_BOUNDED_DIMS.format(value=self.bounded_dims))
        genes = check_structure_config(self.structure_flip, self.structure_genes, self.bounded_dims)
        object.__setattr__(self, "structure_genes", genes)


def initial_temperature(neighbor_scores: np.ndarray) -> float:
    """첫 이웃들 점수의 표준편차. 값이 너무 커서 넘치면 0(더 나쁜 이웃은 받지 않음)."""
    with np.errstate(over="ignore", invalid="ignore"):
        t = float(np.std(neighbor_scores))
    return t if math.isfinite(t) else 0.0


class SimulatedAnnealing:
    """Learner 구현. 점수는 높을수록 좋다."""

    name = "sa"

    def __init__(self, x0: np.ndarray, config: SAConfig) -> None:
        x = initial_genome(x0, config.bounded_dims)
        self._config = config
        self._dim = int(x.size)
        self._rng = new_generator(config.seed)
        t = config.initial_temperature
        self._temperature: float | None = None if t is None else float(t)
        self._generation = 0  # tell 횟수
        self._evaluations = 0
        self._current = x
        self._current_score = math.nan
        self._accepted = 0
        self._recent: list[bool] = []  # 최근 ACCEPTANCE_WINDOW 단계의 수락 여부
        self._candidates = self._propose()
        self._asked = False

    # ------------------------------------------------------------------ 읽기 전용
    @property
    def popsize(self) -> int:
        """한 단계 후보 수 = 이웃 수 + 1(현재해)."""
        return self._config.neighbors + 1

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def config(self) -> SAConfig:
        return self._config

    def set_structure(self, flip: float, genes: Sequence[int]) -> None:
        """구조 뒤집기 칸(structure_flip, structure_genes)을 바꾼다. 다음 tell()이 만드는 이웃부터 쓴다.

        Trainer.replace_env가 쓴다: 학습 중 있음 칸 수식을 고쳐 있음 유전자(GenomeLayout.on_genes)가 바뀌면 지금 학습
        방법도 새 목록을 뒤집는다(설계 4.12·3.5). 이미 만든 다음 후보와 난수 상태는 그대로다. 값이 잘못되면 ValueError
        (한국어)이고 이 학습 방법은 그대로다. 바꾼 값은 state_dict()의 설정에 들어가므로 이어하기·재현이 같다.
        """
        self._config = replace(self._config, structure_flip=flip, structure_genes=tuple(genes))

    @property
    def generation(self) -> int:
        """지금까지 tell()한 단계 수."""
        return self._generation

    @property
    def temperature(self) -> float | None:
        """지금 온도. 초기 온도를 자동으로 정하게 했으면 첫 tell() 전에는 None."""
        return self._temperature

    # ------------------------------------------------------------------ ask/tell
    def ask(self) -> np.ndarray:
        """(이웃 수 + 1, 차원): 첫 행은 현재해, 나머지는 이웃. 난수를 쓰지 않는다."""
        self._asked = True
        return self._candidates.copy()

    def tell(self, candidates: np.ndarray, scores: np.ndarray) -> None:
        check_tell(self._candidates if self._asked else None, candidates)
        c = self._config
        s = sanitize_scores(scores, self.popsize)
        t = initial_temperature(s[1:]) if self._temperature is None else self._temperature
        current_score = float(s[0])
        j = 1 + int(np.argmax(s[1:]))  # 동점이면 앞쪽 이웃
        neighbor_score = float(s[j])
        u = float(self._rng.random())  # 결과와 상관없이 단계마다 한 번(난수 순서 고정)
        accept = neighbor_score >= current_score or (t > 0 and u < math.exp((neighbor_score - current_score) / t))
        if accept:
            self._current = self._candidates[j].copy()
            self._current_score = neighbor_score
            self._accepted += 1
        else:
            self._current_score = current_score
        self._recent = [*self._recent, accept][-ACCEPTANCE_WINDOW:]
        self._temperature = t * float(c.cooling)
        self._generation += 1
        self._evaluations += self.popsize
        self._candidates = self._propose()
        self._asked = False

    def best(self) -> np.ndarray:
        """현재해."""
        return self._current.copy()

    def stats(self) -> dict[str, float]:
        """sigma: 이웃 폭, temperature: 지금 온도(자동이면 첫 tell 전 NaN), acceptance: 최근 ACCEPTANCE_WINDOW 단계의
        수락 비율(전에는 NaN), current_score: 현재해의 마지막 점수(전에는 NaN), accepted: 누적 수락 수,
        popsize(이웃 수 + 1), generation, evaluations.
        """
        recent = self._recent
        return {
            "sigma": float(self._config.step),
            "temperature": math.nan if self._temperature is None else self._temperature,
            "acceptance": sum(recent) / len(recent) if recent else math.nan,
            "current_score": self._current_score,
            "accepted": float(self._accepted),
            "popsize": float(self.popsize),
            "generation": float(self._generation),
            "evaluations": float(self._evaluations),
        }

    # ------------------------------------------------------------------ 상태
    def state_dict(self) -> dict[str, Any]:
        """단계 경계 상태의 사본. 다음 후보는 tell()에서 이미 만들었으므로 ask()와 tell() 사이에 불러도 같다."""
        return {
            "format": STATE_FORMAT,
            "name": self.name,
            "config": config_for_state(asdict(self._config)),  # 구조 뒤집기를 끄면 v2.0과 같은 키
            "dim": self._dim,
            "generation": self._generation,
            "evaluations": self._evaluations,
            "temperature": self._temperature,
            "current": self._current.copy(),
            "current_score": self._current_score,
            "accepted": self._accepted,
            "recent": list(self._recent),
            "candidates": self._candidates.copy(),
            "rng": generator_state(self._rng),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """저장된 상태로 바꾼다. 설정도 저장된 값이 생성자 값보다 우선한다. 유전자 차원이 다르면 ValueError.
        상태가 잘못되었으면 ValueError이고, 그때 이 학습 방법은 그대로다.
        """
        if not isinstance(d, Mapping):
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=type(d).__name__))
        if d.get("name") != self.name:
            raise ValueError(S.LEARNER_ERR_STATE_KIND.format(got=d.get("name"), want=self.name))
        if d.get("format") != STATE_FORMAT:
            raise ValueError(S.LEARNER_ERR_STATE_FORMAT.format(got=d.get("format")))
        # 모두 읽고 검사한 뒤에야 바꾼다(중간에 실패해도 반쯤 불러온 상태가 남지 않게).
        try:
            dim = int(d["dim"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=e)) from e
        if dim != self._dim:
            raise ValueError(S.LEARNER_ERR_STATE_DIM.format(got=dim, want=self._dim))
        try:
            config = SAConfig(**d["config"])
            generation = int(d["generation"])
            evaluations = int(d["evaluations"])
            raw_t = d["temperature"]
            temperature = None if raw_t is None else float(raw_t)
            current = state_array(d["current"], (dim,))
            current_score = float(d["current_score"])
            accepted = int(d["accepted"])
            recent = [bool(v) for v in d["recent"]][-ACCEPTANCE_WINDOW:]
            candidates = state_array(d["candidates"], (config.neighbors + 1, dim))
            rng = generator_from_state(d["rng"])
        except (KeyError, TypeError, ValueError) as e:  # 빠진 항목, 모르는 설정 이름, 잘못된 값
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=e)) from e
        # 설정의 structure_* 키가 없으면(v2.0 상태, 뒤집기를 끈 상태) 데이터클래스 기본값 0이다.
        if temperature is not None and not (math.isfinite(temperature) and temperature >= 0):
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=f"temperature={temperature}"))
        if generation < 0 or evaluations < 0 or not 0 <= accepted <= generation:
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=f"generation={generation}, accepted={accepted}"))
        if not np.array_equal(candidates[0], current):
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason="candidates[0] != current"))
        self._config = config
        self._generation = generation
        self._evaluations = evaluations
        self._temperature = temperature
        self._current = current
        self._current_score = current_score
        self._accepted = accepted
        self._recent = recent
        self._candidates = candidates
        self._rng = rng
        self._asked = False

    # ------------------------------------------------------------------ 내부
    def _propose(self) -> np.ndarray:
        """[현재해, 이웃 1..K]. 이웃 잡음은 전용 생성기에서 (K, 차원) 한 번에 뽑는다. 구조 뒤집기는 그 뒤 이웃 순서로."""
        c = self._config
        out = np.empty((c.neighbors + 1, self._dim), dtype=np.float64)
        out[0] = self._current
        out[1:] = self._current + float(c.step) * self._rng.standard_normal((c.neighbors, self._dim))
        clip_bounded(out, c.bounded_dims)
        flip_structure(out, range(1, c.neighbors + 1), c.structure_genes, c.structure_flip, self._rng)
        return out

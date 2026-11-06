"""유전 알고리즘 학습 방법: 직접 구현 (명세: 학습 방법, 유전자 벡터, 검토 반영 결정 1).

한 세대:
1. ask()는 지금 개체군 전체를 돌려준다. 앞쪽 elites개는 지난 세대 엘리트를 바꾸지 않고 그대로 담는다.
   공통 시드에서는 세대마다 시험 조건이 바뀌므로 엘리트도 이번 세대 시험으로 다시 평가받는다(지난 점수를 들고 가지 않음).
2. tell()은 점수로 순위를 매긴다(안정 정렬: 동점이면 앞쪽 개체가 앞). 상위 elites개를 그대로 남기고, 나머지는
   토너먼트 선택 → 균등 교차(확률 crossover) → 유전자별 가우시안 돌연변이(확률 mutation_rate, 폭 sigma)로 채운다.
   몸 유전자(앞쪽 bounded_dims차원)는 [-1, 1]을 넘으면 잘라낸다.
3. 돌연변이 폭은 세대마다 sigma_decay를 곱한다. 세대 g의 개체군은 폭 mutation_sigma × sigma_decay**g로 만든다.

- 초기 개체군: 첫 개체는 x0 그대로, 나머지는 x0 + mutation_sigma × N(0, 1)(모든 유전자).
- 다음 개체군은 tell()에서 바로 만들어 두므로 ask()는 난수를 쓰지 않는다. 그래서 ask()와 tell() 사이의
  state_dict()도 그 세대 경계 상태이고, 불러온 뒤 ask()는 같은 후보를 낸다.
- 난수는 전용 Generator(PCG64) 하나를 정해진 순서로 쓰고, 그 상태(정수만 든 dict)를 상태에 담는다.
- best()는 마지막 tell()에서 점수가 가장 높은 개체(다음 개체군의 첫 엘리트). tell() 전에는 x0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
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

STATE_FORMAT: Final = 1
_SEED_LIMIT: Final = 2**32


def _probability(v: object) -> bool:
    return is_real(v) and 0.0 <= float(v) <= 1.0  # type: ignore[arg-type]


@dataclass(frozen=True)
class GAConfig:
    popsize: int = 64
    elites: int = 2
    """바꾸지 않고 다음 세대로 넘기는 상위 개체 수(다음 세대에 다시 평가한다). 0 이상, 개체 수 미만."""
    tournament: int = 3
    """토너먼트 선택 크기: 개체군에서 복원 추출한 이만큼 중 순위가 가장 높은 개체가 부모."""
    crossover: float = 0.7
    """균등 교차 확률. 교차하면 유전자마다 반반으로 두 부모 중 하나에서 가져온다."""
    mutation_rate: float = 0.1
    """유전자별 돌연변이 확률."""
    mutation_sigma: float = 0.1
    """가우시안 돌연변이 폭(초기 개체군을 흩뜨리는 폭이기도 하다)."""
    sigma_decay: float = 1.0
    """세대마다 돌연변이 폭에 곱하는 값(0 초과 1 이하)."""
    seed: int = 1
    """학습 방법 전용 난수 생성기 시드(0 ≤ seed < 2**32). Trainer가 실행 시드에서 만든다."""
    bounded_dims: int = 0
    """앞쪽 k차원(몸 유전자)은 [-1, 1]을 넘으면 잘라낸다. 나머지(뇌 가중치)는 경계 없음."""

    def __post_init__(self) -> None:
        if not is_int(self.popsize) or self.popsize < 2:
            raise ValueError(S.LEARNER_ERR_POPSIZE.format(value=self.popsize))
        if not is_int(self.elites) or not 0 <= self.elites < self.popsize:
            raise ValueError(S.LEARNER_GA_ERR_ELITES.format(value=self.elites, popsize=self.popsize))
        if not is_int(self.tournament) or not 1 <= self.tournament <= self.popsize:
            raise ValueError(S.LEARNER_GA_ERR_TOURNAMENT.format(value=self.tournament, popsize=self.popsize))
        if not _probability(self.crossover):
            raise ValueError(S.LEARNER_GA_ERR_CROSSOVER.format(value=self.crossover))
        if not _probability(self.mutation_rate):
            raise ValueError(S.LEARNER_GA_ERR_MUTATION_RATE.format(value=self.mutation_rate))
        if not is_real(self.mutation_sigma) or not math.isfinite(self.mutation_sigma) or self.mutation_sigma <= 0:
            raise ValueError(S.LEARNER_GA_ERR_MUTATION_SIGMA.format(value=self.mutation_sigma))
        if not is_real(self.sigma_decay) or not 0.0 < float(self.sigma_decay) <= 1.0:
            raise ValueError(S.LEARNER_GA_ERR_SIGMA_DECAY.format(value=self.sigma_decay))
        if not is_int(self.seed) or not 0 <= self.seed < _SEED_LIMIT:
            raise ValueError(S.LEARNER_ERR_SEED.format(value=self.seed))
        if not is_int(self.bounded_dims) or self.bounded_dims < 0:
            raise ValueError(S.LEARNER_ERR_BOUNDED_DIMS.format(value=self.bounded_dims))


class GeneticAlgorithm:
    """Learner 구현. 점수는 높을수록 좋다."""

    name = "ga"

    def __init__(self, x0: np.ndarray, config: GAConfig) -> None:
        x = initial_genome(x0, config.bounded_dims)
        self._config = config
        self._dim = int(x.size)
        self._rng = new_generator(config.seed)
        self._sigma = float(config.mutation_sigma)
        self._generation = 0  # tell 횟수
        self._evaluations = 0
        self._best = x.copy()
        self._best_score = math.nan
        population = np.empty((config.popsize, self._dim), dtype=np.float64)
        population[0] = x
        population[1:] = x + self._sigma * self._rng.standard_normal((config.popsize - 1, self._dim))
        clip_bounded(population, config.bounded_dims)
        self._population = population
        self._asked = False

    # ------------------------------------------------------------------ 읽기 전용
    @property
    def popsize(self) -> int:
        return self._config.popsize

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def config(self) -> GAConfig:
        return self._config

    @property
    def generation(self) -> int:
        """지금까지 tell()한 세대 수."""
        return self._generation

    @property
    def elites(self) -> np.ndarray:
        """다음 ask()의 앞쪽 elites개 행(지난 세대 상위 개체 그대로). 사본."""
        return self._population[: self._config.elites].copy()

    # ------------------------------------------------------------------ ask/tell
    def ask(self) -> np.ndarray:
        """(popsize, 차원) 개체군 전체. 앞쪽 elites개는 지난 세대 엘리트 그대로. 난수를 쓰지 않는다."""
        self._asked = True
        return self._population.copy()

    def tell(self, candidates: np.ndarray, scores: np.ndarray) -> None:
        check_tell(self._population if self._asked else None, candidates)
        c = self._config
        fitness = sanitize_scores(scores, c.popsize)
        order = np.argsort(-fitness, kind="stable")  # 높은 점수부터, 동점이면 앞쪽 개체가 앞
        ranked = self._population[order]
        self._best = ranked[0].copy()
        self._best_score = float(fitness[order[0]])
        self._sigma *= float(c.sigma_decay)
        children = self._offspring(ranked, c.popsize - c.elites)
        self._population = np.concatenate([ranked[: c.elites], children], axis=0)
        self._asked = False
        self._generation += 1
        self._evaluations += c.popsize

    def best(self) -> np.ndarray:
        """마지막 tell()에서 점수가 가장 높은 개체(엘리트). tell() 전에는 x0."""
        return self._best.copy()

    def stats(self) -> dict[str, float]:
        """sigma: 지금 개체군(다음 ask)을 만든 돌연변이 폭, best_score: 마지막 tell의 최고 점수(전에는 NaN),
        popsize, generation, evaluations.
        """
        return {
            "sigma": self._sigma,
            "best_score": self._best_score,
            "popsize": float(self._config.popsize),
            "generation": float(self._generation),
            "evaluations": float(self._evaluations),
        }

    # ------------------------------------------------------------------ 상태
    def state_dict(self) -> dict[str, Any]:
        """세대 경계 상태의 사본. 다음 개체군은 tell()에서 이미 만들었으므로 ask()와 tell() 사이에 불러도 같다."""
        return {
            "format": STATE_FORMAT,
            "name": self.name,
            "config": asdict(self._config),
            "dim": self._dim,
            "generation": self._generation,
            "evaluations": self._evaluations,
            "sigma": self._sigma,
            "best": self._best.copy(),
            "best_score": self._best_score,
            "population": self._population.copy(),
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
            config = GAConfig(**d["config"])
            generation = int(d["generation"])
            evaluations = int(d["evaluations"])
            sigma = float(d["sigma"])
            best_score = float(d["best_score"])
            best = state_array(d["best"], (dim,))
            population = state_array(d["population"], (config.popsize, dim))
            rng = generator_from_state(d["rng"])
        except (KeyError, TypeError, ValueError) as e:  # 빠진 항목, 모르는 설정 이름, 잘못된 값
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=e)) from e
        # 감쇠가 오래 이어지면 폭이 0으로 언더플로할 수 있어 0도 받는다.
        if not (math.isfinite(sigma) and sigma >= 0) or generation < 0 or evaluations < 0:
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=f"sigma={sigma}, generation={generation}"))
        self._config = config
        self._generation = generation
        self._evaluations = evaluations
        self._sigma = sigma
        self._best = best
        self._best_score = best_score
        self._population = population
        self._rng = rng
        self._asked = False

    # ------------------------------------------------------------------ 내부
    def _offspring(self, ranked: np.ndarray, n: int) -> np.ndarray:
        """순위순 개체군에서 자식 n개. 난수는 늘 같은 양을 같은 순서로 뽑는다(교차하지 않는 자식도)."""
        c, rng, dim = self._config, self._rng, self._dim
        popsize = ranked.shape[0]
        # 토너먼트: 순위 번호를 복원 추출해서 가장 작은 번호(점수가 가장 높고, 동점이면 앞쪽)가 이긴다.
        first = rng.integers(0, popsize, size=(n, c.tournament)).min(axis=1)
        second = rng.integers(0, popsize, size=(n, c.tournament)).min(axis=1)
        cross = rng.random(n) < c.crossover
        take_second = rng.random((n, dim)) < 0.5
        mutate = rng.random((n, dim)) < c.mutation_rate
        noise = rng.standard_normal((n, dim))
        children = np.where(cross[:, None] & take_second, ranked[second], ranked[first])
        children += np.where(mutate, self._sigma * noise, 0.0)
        clip_bounded(children, c.bounded_dims)
        return children

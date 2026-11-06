"""Trainer: 세대 루프 (명세: 학습 방법, 결정성, 저장).

한 세대:
1. 학습 방법에서 후보를 받는다(ask).
2. 같은 세대 공통 시험 조건으로 모든 후보를 시험한다(시드 = 실행 시드, 세대, 시험 번호).
3. 시험 점수를 집계해 학습 방법에 돌려준다(tell).
4. 학습 시험이 일정 횟수 쌓일 때마다 후보(학습 점수 상위 몇 개 + 학습 방법의 추천해)를 검증 세트로 다시
   평가해 챔피언을 고른다. 정체·목표 점수 판정은 검증 점수 기준이다.
5. 통계, 세대 대표, 챔피언, 체크포인트를 남긴다.

결과는 워커 수와 무관하고, 체크포인트에서 이어가면 끊지 않은 실행과 같다(경과 시간 열만 다르다).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from evolab.engine import seeds
from evolab.engine.evaluate import EvalResult
from evolab.learners.base import Learner
from evolab.policy.genome import GenomeLayout, layout_for
from evolab.spec.compiled import EnvSpec
from evolab.spec.io import compile_env, env_hash
from evolab.world.physics2d.episode import Trial

CHECKPOINT_FORMAT = 1


class Evaluator(Protocol):
    def evaluate(
        self, env_version: str, env_text: str, layout: GenomeLayout, genomes: np.ndarray, pairs: list[tuple[int, Trial]]
    ) -> EvalResult: ...


@dataclass
class TrainConfig:
    method: str = "cmaes"
    run_seed: int = 1
    popsize: int = 32
    sigma0: float = 0.5
    trials: int | None = None
    """None이면 환경 파일의 evaluation.trials."""
    aggregate: str | None = None
    """None이면 환경 파일의 evaluation.aggregate."""
    diagonal_threshold: int = 1000
    ipop: bool = False
    validate_every: int = 1000
    """학습 시험이 이만큼 쌓일 때마다 검증."""
    validation_seeds: int = seeds.VALIDATION_SET
    validation_candidates: int = 3
    checkpoint_every: int = 10
    max_generations: int | None = None
    max_evaluations: int | None = None
    max_time: float | None = None
    target_score: float | None = None
    stagnation_generations: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Champion:
    genome: np.ndarray | None = None
    score: float = -math.inf
    generation: int = -1


@dataclass
class TrainState:
    generation: int = 0
    """끝낸 세대 수(다음 세대 번호)."""
    evaluations: int = 0
    validation_evaluations: int = 0
    elapsed: float = 0.0
    validations_done: int = 0
    champion: Champion = field(default_factory=Champion)
    best_validation: float = -math.inf
    improved_generation: int = 0


STOP_GENERATIONS = "max_generations"
STOP_EVALUATIONS = "max_evaluations"
STOP_TIME = "max_time"
STOP_TARGET = "target_score"
STOP_STAGNATION = "stagnation"
STOP_REQUESTED = "requested"


def aggregate(scores: np.ndarray, how: str) -> np.ndarray:
    """(개체, 시험) -> (개체,). 하위 25%는 아래쪽 ceil(시험 수 x 0.25)개의 평균."""
    if how == "mean":
        return scores.mean(axis=1)
    if how == "min":
        return scores.min(axis=1)
    k = max(1, math.ceil(scores.shape[1] * 0.25))
    return np.sort(scores, axis=1, kind="stable")[:, :k].mean(axis=1)


def make_learner(config: TrainConfig, layout: GenomeLayout) -> Learner:
    if config.method == "cmaes":
        from evolab.learners.cmaes import CMAES, CMAESConfig

        return CMAES(
            layout.initial(),
            CMAESConfig(
                popsize=config.popsize,
                sigma0=config.sigma0,
                diagonal_threshold=config.diagonal_threshold,
                ipop=config.ipop,
                seed=seeds.learner_seed(config.run_seed),
                bounded_dims=layout.n_body,
            ),
        )
    raise ValueError(config.method)  # 유전 알고리즘, 담금질은 M6


class Trainer:
    def __init__(
        self,
        env_text: str,
        config: TrainConfig,
        evaluator: Evaluator,
        store: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.env_text = env_text
        self.spec: EnvSpec = compile_env(env_text)
        self.env_version = env_hash(self.spec.file)[:16]
        self.config = config
        self.evaluator = evaluator
        self.store = store
        self.clock = clock
        self.layout = layout_for(self.spec)
        self.trials = config.trials or self.spec.file.evaluation.trials
        self.how = config.aggregate or self.spec.file.evaluation.aggregate
        self.learner = make_learner(config, self.layout)
        self.state = TrainState()
        self.block_ids = [b.id for b in self.spec.file.scoring.blocks]
        self.end_ids = [t.event for t in self.spec.terminations]
        self.last_stats: dict[str, Any] = {}
        self.stop_reason: str | None = None

    # ------------------------------------------------------------------
    def run(
        self,
        on_generation: Callable[[dict[str, Any]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        """정지 조건이 될 때까지 세대를 돈다. 정지 원인을 돌려준다. 끝나면 마지막 검증과 체크포인트."""
        reason = self.check_stop()
        while reason is None:
            row = self.step()
            if on_generation is not None:
                on_generation(row)
            reason = self.check_stop()
            if reason is None and should_stop is not None and should_stop():
                reason = STOP_REQUESTED
        self.stop_reason = reason
        self.finish()
        return reason

    def finish(self) -> None:
        """멈출 때 체크포인트를 남긴다.

        멈출 때 따로 검증하지 않는다. 검증은 학습 시험 수로 정한 주기에만 해야, 끊었다 이어 간 실행이 끊지 않은
        실행과 같다(챔피언과 정체 판정이 멈춘 시점에 따라 달라지지 않게).
        """
        self.checkpoint()

    def check_stop(self) -> str | None:
        c, s = self.config, self.state
        if c.max_generations is not None and s.generation >= c.max_generations:
            return STOP_GENERATIONS
        if c.max_evaluations is not None and s.evaluations >= c.max_evaluations:
            return STOP_EVALUATIONS
        if c.max_time is not None and s.elapsed >= c.max_time:
            return STOP_TIME
        if c.target_score is not None and s.champion.score >= c.target_score:
            return STOP_TARGET
        if c.stagnation_generations is not None and s.generation - s.improved_generation >= c.stagnation_generations:
            return STOP_STAGNATION
        return None

    def step(self) -> dict[str, Any]:
        """한 세대. 통계 한 줄을 돌려준다(저장소가 있으면 stats.csv에도 쓴다)."""
        t0 = self.clock()
        s = self.state
        gen = s.generation
        X = self.learner.ask()
        trials = seeds.train_trials(self.config.run_seed, gen, self.trials)
        pairs = [(i, t) for i in range(X.shape[0]) for t in trials]
        res = self.evaluator.evaluate(self.env_version, self.env_text, self.layout, X, pairs)
        pop, T = X.shape[0], len(trials)
        scores = res.scores.reshape(pop, T)
        agg = aggregate(scores, self.how)
        self.learner.tell(X, agg)
        s.evaluations += pop * T
        rep = int(np.argmax(agg))  # 동점이면 앞쪽(결정적)

        validated: float | None = None
        bucket = s.evaluations // self.config.validate_every
        if bucket > s.validations_done:
            s.validations_done = bucket
            validated = self._validate(X, agg)
        s.generation = gen + 1
        gen_time = self.clock() - t0
        s.elapsed += gen_time

        row = self._stats_row(gen, X, agg, res, rep, validated, gen_time)
        self.last_stats = row
        if self.store is not None:
            self.store.append_stats(row)
            self.store.save_champion(gen, "representative", X[rep], score=float(agg[rep]))
            if self.config.checkpoint_every and s.generation % self.config.checkpoint_every == 0:
                self.checkpoint()
        return row

    def _validate(self, X: np.ndarray, agg: np.ndarray) -> float:
        """후보를 검증 세트로 평가해 챔피언을 고른다. 이번 검증의 최고 점수를 돌려준다."""
        s = self.state
        k = min(self.config.validation_candidates, X.shape[0])
        top = np.argsort(-agg, kind="stable")[:k]
        cands = np.vstack([X[top], self.learner.best()[None, :]])
        trials = seeds.validation_trials(self.config.validation_seeds)
        pairs = [(i, t) for i in range(cands.shape[0]) for t in trials]
        res = self.evaluator.evaluate(self.env_version, self.env_text, self.layout, cands, pairs)
        s.validation_evaluations += len(pairs)
        val = res.scores.reshape(cands.shape[0], len(trials)).mean(axis=1)
        best = int(np.argmax(val))
        if val[best] > s.champion.score:
            s.champion = Champion(cands[best].copy(), float(val[best]), s.generation)
            if self.store is not None:
                self.store.save_champion(s.generation, "champion", cands[best], score=float(val[best]))
        if val[best] > s.best_validation:
            s.best_validation = float(val[best])
            s.improved_generation = s.generation
        return float(val[best])

    def _stats_row(
        self, gen: int, X: np.ndarray, agg: np.ndarray, res: EvalResult, rep: int, validated: float | None, gen_time: float
    ) -> dict[str, Any]:
        s = self.state
        q = np.quantile(agg, [0.25, 0.5, 0.75])
        learner_stats = self.learner.stats()
        steps = int(res.steps.sum())
        row: dict[str, Any] = {
            "generation": gen,
            "evaluations": s.evaluations,
            "validation_evaluations": s.validation_evaluations,
            "score_best": float(agg.max()),
            "score_mean": float(agg.mean()),
            "score_q25": float(q[0]),
            "score_median": float(q[1]),
            "score_q75": float(q[2]),
            "score_min": float(agg.min()),
            "validation_best": validated if validated is not None else "",
            "champion_score": s.champion.score if math.isfinite(s.champion.score) else "",
            "champion_generation": s.champion.generation,
            "sigma": float(learner_stats.get("sigma", math.nan)),
            "diversity": float(np.mean(np.std(X, axis=0))),
            "steps": steps,
        }
        n = len(res.end_events)
        for e in self.end_ids:
            row[f"end_{e}"] = sum(1 for x in res.end_events if x == e) / n
        for j, b in enumerate(self.block_ids):
            row[f"block_{b}"] = float(res.blocks[:, j].mean())
        # 아래는 시간에 따라 달라지는 열(결정성 비교에서 뺀다)
        row["gen_seconds"] = gen_time
        row["elapsed"] = s.elapsed
        row["steps_per_second"] = steps / gen_time if gen_time > 0 else 0.0
        return row

    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        s = self.state
        return {
            "format": CHECKPOINT_FORMAT,
            "config": self.config.to_dict(),
            "env_version": self.env_version,
            "layout_signature": self.layout.signature(),
            "learner": self.learner.state_dict(),
            "state": {
                "generation": s.generation,
                "evaluations": s.evaluations,
                "validation_evaluations": s.validation_evaluations,
                "elapsed": s.elapsed,
                "validations_done": s.validations_done,
                "champion_genome": None if s.champion.genome is None else s.champion.genome.copy(),
                "champion_score": s.champion.score,
                "champion_generation": s.champion.generation,
                "best_validation": s.best_validation,
                "improved_generation": s.improved_generation,
            },
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        if d.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("checkpoint format")
        if d["layout_signature"] != self.layout.signature():
            raise ValueError("layout")
        self.learner.load_state_dict(d["learner"])
        st = d["state"]
        self.state = TrainState(
            generation=st["generation"],
            evaluations=st["evaluations"],
            validation_evaluations=st["validation_evaluations"],
            elapsed=st["elapsed"],
            validations_done=st["validations_done"],
            champion=Champion(st["champion_genome"], st["champion_score"], st["champion_generation"]),
            best_validation=st["best_validation"],
            improved_generation=st["improved_generation"],
        )

    def checkpoint(self) -> None:
        if self.store is not None and self.state.generation:
            self.store.save_checkpoint(self.state.generation, self.state_dict())

    @classmethod
    def resume(
        cls, store: Any, evaluator: Evaluator, generation: int | None = None, clock: Callable[[], float] = time.perf_counter,
        **overrides: Any,
    ) -> Trainer:
        """저장소의 체크포인트에서 이어간다. overrides로 정지 조건 같은 설정을 바꿀 수 있다."""
        gen, d = store.load_checkpoint(generation)
        info = store.read_run_info()
        config = TrainConfig.from_dict({**d["config"], **overrides})
        version = info["env_versions"][-1]["id"]
        trainer = cls(store.env_text(version), config, evaluator, store, clock)
        trainer.load_state_dict(d)
        store.truncate_after(gen)
        return trainer

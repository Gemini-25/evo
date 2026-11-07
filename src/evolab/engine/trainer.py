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
        self, env_version: str, env_text: str, layout: GenomeLayout, genomes: np.ndarray, pairs: list[tuple[int, Trial]],
        record: list[bool] | None = None,
    ) -> EvalResult: ...


TIME_COLUMNS = ("gen_seconds", "elapsed", "steps_per_second", "worker_utilization")
"""시간에 따라 달라지는 통계 열. 결정성 비교에서 뺀다."""

LIVE_CONFIG_KEYS = (
    "max_generations", "max_evaluations", "max_time", "target_score", "stagnation_generations",
    "checkpoint_every", "trials", "aggregate", "validate_every",
)
"""학습 중(멈춘 상태)에 바꿀 수 있는 설정. 학습 방법·시드·개체 수·시그마를 바꾸면 다른 실험이 되므로 새 실행으로 한다."""


class TrainerChangeError(ValueError):
    """멈춘 학습에 적용할 수 없는 변경. message는 한국어."""


@dataclass
class Generation:
    """마지막 세대의 자세한 결과(화면에 보낼 것). 통계 줄과 같은 세대."""

    generation: int
    population: np.ndarray
    scores: np.ndarray
    """개체별 집계 점수."""
    representative: int
    trial0: Trial
    trajectories: list[Any] | None
    """관전 모드면 개체별 시험 0번 궤적(Trajectory), 아니면 None."""
    env_version: str
    champion_changed: bool


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
    # 유전 알고리즘(method="ga"). 개체 수는 popsize를 쓴다(명세 기본 64는 GAConfig().popsize).
    ga_elites: int = 2
    ga_tournament: int = 3
    ga_crossover: float = 0.7
    ga_mutation_rate: float = 0.1
    ga_mutation_sigma: float = 0.1
    ga_sigma_decay: float = 1.0
    # 담금질(method="sa"). popsize는 쓰지 않고, 한 세대 후보 수 = sa_neighbors + 1(현재해).
    sa_neighbors: int = 8
    sa_step: float = 0.1
    sa_initial_temperature: float | None = None
    """None이면 첫 이웃들 점수의 표준편차."""
    sa_cooling: float = 0.995

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        """값 범위를 확인한다. 잘못되면 TrainerChangeError(한국어)."""
        from evolab.app import strings as S

        def bad(name: str) -> TrainerChangeError:
            return TrainerChangeError(S.TRAIN_CONFIG_INVALID.format(name=name, value=getattr(self, name)))

        def is_int(v: Any) -> bool:
            return isinstance(v, int) and not isinstance(v, bool)

        def is_num(v: Any) -> bool:
            return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

        for name in ("popsize", "validate_every", "validation_seeds", "validation_candidates"):
            if not is_int(getattr(self, name)) or getattr(self, name) < 1:
                raise bad(name)
        if self.popsize < 2:
            raise bad("popsize")
        if not is_int(self.run_seed) or self.run_seed < 0:
            raise bad("run_seed")
        if not is_num(self.sigma0) or self.sigma0 <= 0:
            raise bad("sigma0")
        if not is_int(self.diagonal_threshold) or self.diagonal_threshold < 0:
            raise bad("diagonal_threshold")
        if not is_int(self.checkpoint_every) or self.checkpoint_every < 0:
            raise bad("checkpoint_every")
        for name in ("trials", "max_generations", "max_evaluations", "stagnation_generations"):
            v = getattr(self, name)
            if v is not None and (not is_int(v) or v < 1):
                raise bad(name)
        if self.max_time is not None and (not is_num(self.max_time) or self.max_time <= 0):
            raise bad("max_time")
        if self.target_score is not None and not is_num(self.target_score):
            raise bad("target_score")
        if self.aggregate not in (None, "mean", "min", "q25"):
            raise bad("aggregate")
        if self.method not in ("cmaes", "ga", "sa"):
            raise bad("method")
        if self.method == "ga" and self.popsize < 2:
            raise bad("popsize")
        if not is_int(self.ga_elites) or self.ga_elites < 0 or (self.method == "ga" and self.ga_elites >= self.popsize):
            raise bad("ga_elites")
        if not is_int(self.ga_tournament) or self.ga_tournament < 1 or (
            self.method == "ga" and self.ga_tournament > self.popsize
        ):
            raise bad("ga_tournament")
        for name in ("ga_crossover", "ga_mutation_rate"):
            if not is_num(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise bad(name)
        for name in ("ga_mutation_sigma", "sa_step"):
            if not is_num(getattr(self, name)) or getattr(self, name) <= 0:
                raise bad(name)
        for name in ("ga_sigma_decay", "sa_cooling"):
            if not is_num(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise bad(name)
        if not is_int(self.sa_neighbors) or self.sa_neighbors < 1:
            raise bad("sa_neighbors")
        t = self.sa_initial_temperature
        if t is not None and (not is_num(t) or t < 0):
            raise bad("sa_initial_temperature")

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
    compare_score: float = math.nan
    """챔피언을 비교 평가 세트(선정에 쓰지 않은 시드 32개)로 잰 점수. 비교 그래프·표의 기본 값."""
    compare_success: float = math.nan
    """같은 평가에서 성공(outcome success) 비율."""
    compare_evaluations: int = 0


STOP_GENERATIONS = "max_generations"
STOP_EVALUATIONS = "max_evaluations"
STOP_TIME = "max_time"
STOP_TARGET = "target_score"
STOP_STAGNATION = "stagnation"
STOP_REQUESTED = "requested"


def training_trials(spec: EnvSpec, run_seed: int, generation: int, count: int) -> list[Trial]:
    """학습 시험 목록. 학습 트랙 세트가 있으면 시험 = 트랙마다 한 번(시험 번호 = 트랙 번호, 시험 수 = 세트 크기)."""
    size = spec.track_set_size
    if size:
        return [Trial(t.seed, k) for k, t in enumerate(seeds.train_trials(run_seed, generation, size))]
    return seeds.train_trials(run_seed, generation, count)


def trials_for(spec: EnvSpec, config: TrainConfig) -> int:
    return spec.track_set_size or config.trials or spec.file.evaluation.trials


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
    if config.method == "ga":
        from evolab.learners.ga import GAConfig, GeneticAlgorithm

        return GeneticAlgorithm(
            layout.initial(),
            GAConfig(
                popsize=config.popsize,
                elites=config.ga_elites,
                tournament=config.ga_tournament,
                crossover=config.ga_crossover,
                mutation_rate=config.ga_mutation_rate,
                mutation_sigma=config.ga_mutation_sigma,
                sigma_decay=config.ga_sigma_decay,
                seed=seeds.learner_seed(config.run_seed),
                bounded_dims=layout.n_body,
            ),
        )
    if config.method == "sa":
        from evolab.learners.sa import SAConfig, SimulatedAnnealing

        return SimulatedAnnealing(
            layout.initial(),
            SAConfig(
                neighbors=config.sa_neighbors,
                step=config.sa_step,
                initial_temperature=config.sa_initial_temperature,
                cooling=config.sa_cooling,
                seed=seeds.learner_seed(config.run_seed),
                bounded_dims=layout.n_body,
            ),
        )
    raise ValueError(config.method)


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
        self.trials = trials_for(self.spec, config)
        self.how = config.aggregate or self.spec.file.evaluation.aggregate
        self.learner = make_learner(config, self.layout)
        self.state = TrainState()
        self.block_ids = [b.id for b in self.spec.file.scoring.blocks]
        self.end_ids = [t.event for t in self.spec.terminations]
        self.last_stats: dict[str, Any] = {}
        self.last_generation: Generation | None = None
        self.stop_reason: str | None = None
        self.weighted_blocks = self.spec.file.scoring.formula is None
        self._header: list[str] | None = None
        if store is not None:
            self._header = store.stats_columns() if hasattr(store, "stats_columns") else None

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

    def step(self, record: bool = False) -> dict[str, Any]:
        """한 세대. 통계 한 줄을 돌려준다(저장소가 있으면 stats.csv에도 쓴다).

        record면 모든 개체의 시험 0번 궤적을 남긴다(관전 모드). 기록은 결과를 바꾸지 않는다.
        """
        t0 = self.clock()
        s = self.state
        gen = s.generation
        # 세대가 끝까지(통계·챔피언 저장까지) 가지 않으면 이 트레이너는 반쯤 바뀐 상태다(learner.tell, 세대 번호).
        # 그 상태로는 체크포인트를 남기지 않는다(리뷰: 끝나지 않은 세대가 체크포인트되어 통계 줄이 빠짐).
        self.in_step = True
        X = self.learner.ask()
        trials = training_trials(self.spec, self.config.run_seed, gen, self.trials)
        pairs = [(i, t) for i in range(X.shape[0]) for t in trials]
        t_eval = self.clock()
        if record:
            flags = [k == 0 for _ in range(X.shape[0]) for k in range(len(trials))]
            res = self.evaluator.evaluate(self.env_version, self.env_text, self.layout, X, pairs, record=flags)
        else:
            res = self.evaluator.evaluate(self.env_version, self.env_text, self.layout, X, pairs)
        eval_wall = self.clock() - t_eval
        busy = float(getattr(self.evaluator, "last_busy", 0.0) or 0.0)  # 검증 평가가 덮어쓰기 전에 읽는다
        pop, T = X.shape[0], len(trials)
        scores = res.scores.reshape(pop, T)
        agg = aggregate(scores, self.how)
        self.learner.tell(X, agg)
        s.evaluations += pop * T
        rep = int(np.argmax(agg))  # 동점이면 앞쪽(결정적)
        # 기록용 세대 번호는 '끝낸 세대 수'(1부터): 통계 줄, 챔피언 파일, 체크포인트가 같은 번호를 쓴다.
        s.generation = gen + 1
        label = s.generation

        validated: float | None = None
        champion_before = s.champion.generation
        bucket = s.evaluations // self.config.validate_every
        if bucket > s.validations_done:
            s.validations_done = bucket
            validated = self._validate(X, agg)
        gen_time = self.clock() - t0
        s.elapsed += gen_time

        row = self._stats_row(label, X, agg, res, rep, validated, gen_time)
        workers = max(1, int(getattr(self.evaluator, "workers", 1) or 1))
        row["worker_utilization"] = min(1.0, busy / (workers * eval_wall)) if eval_wall > 0 else 0.0
        row = self._fit_columns(row)
        self.last_stats = row
        traj = None
        if record and res.trajectories is not None:
            traj = [res.trajectories[i * T] for i in range(pop)]
        self.last_generation = Generation(label, X, agg, rep, trials[0], traj, self.env_version,
                                          s.champion.generation != champion_before)
        if self.store is not None:
            self.store.append_stats(row)
            self.store.save_champion(label, "representative", X[rep], score=float(agg[rep]),
                                     env_version=self.env_version)
        self.in_step = False
        if self.store is not None and self.config.checkpoint_every and s.generation % self.config.checkpoint_every == 0:
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
            self._compare(cands[best])
            if self.store is not None:
                self.store.save_champion(s.generation, "champion", cands[best], score=float(val[best]),
                                         compare_score=s.compare_score, env_version=self.env_version)
        if val[best] > s.best_validation:
            s.best_validation = float(val[best])
            s.improved_generation = s.generation
        return float(val[best])

    def _compare(self, genome: np.ndarray) -> None:
        """새 챔피언을 비교 평가 세트로 잰다(명세: 공정한 비교). 이 시험은 누적 시험 횟수와 따로 센다."""
        s = self.state
        trials = seeds.compare_trials()
        pairs = [(0, t) for t in trials]
        res = self.evaluator.evaluate(self.env_version, self.env_text, self.layout, genome[None, :], pairs)
        s.compare_evaluations += len(pairs)
        s.compare_score = float(res.scores.mean())
        s.compare_success = sum(1 for o in res.outcomes if o == "success") / len(pairs)

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
            "compare_score": s.compare_score if math.isfinite(s.compare_score) else "",
            "compare_success": s.compare_success if math.isfinite(s.compare_success) else "",
            "compare_evaluations": s.compare_evaluations,
            "sigma": float(learner_stats.get("sigma", math.nan)),
            "diversity": float(np.mean(np.std(X, axis=0))),
            "distance": mean_pairwise_distance(X),
            "steps": steps,
        }
        n = len(res.end_events)
        for e in self.end_ids:
            row[f"end_{e}"] = sum(1 for x in res.end_events if x == e) / n
        for j, b in enumerate(self.block_ids):
            row[f"block_{b}"] = float(res.blocks[:, j].mean())
        if self.weighted_blocks and res.weighted is not None and self.block_ids:
            per = res.weighted.reshape(X.shape[0], -1, len(self.block_ids)).mean(axis=1)
            for j, b in enumerate(self.block_ids):
                row[f"contrib_mean_{b}"] = float(per[:, j].mean())
                row[f"contrib_rep_{b}"] = float(per[rep, j])
        row["scores"] = " ".join(repr(round(float(v), 4)) for v in agg)
        # 아래는 시간에 따라 달라지는 열(결정성 비교에서 뺀다)
        row["gen_seconds"] = gen_time
        row["elapsed"] = s.elapsed
        row["steps_per_second"] = steps / gen_time if gen_time > 0 else 0.0
        return row

    def _fit_columns(self, row: dict[str, Any]) -> dict[str, Any]:
        """예전 형식으로 시작한 실행이면 그 열에 맞춘다(없는 열은 빈칸, 새 열은 뺀다)."""
        if self._header is None or list(row) == self._header:
            return row
        return {k: row.get(k, "") for k in self._header}

    # ------------------------------------------------------------------
    # 멈춘 상태에서의 변경 (세대 경계)
    # ------------------------------------------------------------------
    def apply_config(self, changes: dict[str, Any]) -> dict[str, Any]:
        """정지 조건 같은 설정을 바꾼다. 바꾼 값만 돌려준다. 체크포인트에 새 설정이 들어간다."""
        from evolab.app import strings as S

        bad = [k for k in changes if k not in LIVE_CONFIG_KEYS]
        if bad:
            raise TrainerChangeError(S.TRAIN_CHANGE_NOT_ALLOWED.format(keys=", ".join(bad)))
        trial_config = TrainConfig.from_dict({**self.config.to_dict(), **changes})
        trial_config.validate()
        done = {k: v for k, v in changes.items() if getattr(self.config, k) != v}
        self.config = trial_config
        self.trials = trials_for(self.spec, self.config)
        self.how = self.config.aggregate or self.spec.file.evaluation.aggregate
        if "validate_every" in done:
            # 검증은 학습 시험 수가 새 주기의 배수를 넘을 때마다: 지금까지 한 수를 새 단위로 바꿔야 바로 이어진다
            self.state.validations_done = self.state.evaluations // self.config.validate_every
        return done

    def changeable_state(self) -> dict[str, Any]:
        """설정 변경·환경 교체를 되돌릴 때 쓰는 사본(저장에 실패하면 트레이너를 그대로 돌려놓는다)."""
        return {"config": TrainConfig.from_dict(self.config.to_dict()), "env_text": self.env_text, "spec": self.spec,
                "env_version": self.env_version, "trials": self.trials, "how": self.how,
                "validations_done": self.state.validations_done}

    def restore_changeable(self, snap: dict[str, Any]) -> None:
        self.config = snap["config"]
        self.env_text, self.spec, self.env_version = snap["env_text"], snap["spec"], snap["env_version"]
        self.trials, self.how = snap["trials"], snap["how"]
        self.state.validations_done = snap["validations_done"]

    def replace_env(self, env_text: str) -> str:
        """환경을 바꾼다(유전자 배치, 점수 블록, 종료 조건 목록이 같아야 한다). 새 환경 버전 id(해시)를 돌려준다."""
        from evolab.app import strings as S

        spec = compile_env(env_text)
        layout = layout_for(spec)
        if layout.signature() != self.layout.signature():
            raise TrainerChangeError(S.TRAIN_ENV_LAYOUT_CHANGED)
        if [b.id for b in spec.file.scoring.blocks] != self.block_ids or [t.event for t in spec.terminations] != self.end_ids:
            raise TrainerChangeError(S.TRAIN_ENV_COLUMNS_CHANGED)
        if (spec.file.scoring.formula is None) != self.weighted_blocks:
            raise TrainerChangeError(S.TRAIN_ENV_COLUMNS_CHANGED)
        self.env_text = env_text
        self.spec = spec
        self.env_version = env_hash(spec.file)[:16]
        self.trials = trials_for(spec, self.config)
        self.how = self.config.aggregate or spec.file.evaluation.aggregate
        return self.env_version

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
                "compare_score": s.compare_score,
                "compare_success": s.compare_success,
                "compare_evaluations": s.compare_evaluations,
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
            compare_score=st.get("compare_score", math.nan),
            compare_success=st.get("compare_success", math.nan),
            compare_evaluations=st.get("compare_evaluations", 0),
        )

    def checkpoint(self) -> None:
        if getattr(self, "in_step", False):
            return  # 끝나지 않은 세대(오류로 중간에 멈춤): 마지막 정상 체크포인트를 그대로 둔다
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
        versions = list(info["env_versions"])
        # 체크포인트 시점에 쓰던 환경: 적용 세대가 체크포인트 세대 이하인 마지막 기록
        usable = [v for v in versions if int(v.get("from_generation", 0)) <= gen] or versions[:1]
        version = usable[-1]["id"]
        text = store.env_text(version)
        if versions[-1]["id"] != version:
            store.add_env_version(text, gen)  # 마지막 기록이 지금 쓰는 환경이 되게
        trainer = cls(text, config, evaluator, store, clock)
        trainer.load_state_dict(d)
        store.truncate_after(gen)
        return trainer


def mean_pairwise_distance(X: np.ndarray) -> float:
    """개체 사이 유클리드 거리의 평균(서로 다른 쌍)."""
    n = X.shape[0]
    if n < 2:
        return 0.0
    sq = np.einsum("ij,ij->i", X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    iu = np.triu_indices(n, 1)
    return float(np.sqrt(np.maximum(d2[iu], 0.0)).mean())

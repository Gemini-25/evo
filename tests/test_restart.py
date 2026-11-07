"""v1.1 학습 안정성: 정체하면 새로 시작(자동 재시작)과 예선 (명세: 학습 방법, 부록 A25~A32).

설계 15절 A~F. 정확한 시점은 가짜 평가기(착륙선을 돌리지 않고 정한 점수를 줌)로 보고, 결정성은 실제 착륙선과 작은
설정으로 본다. 결정성 테스트는 예선 경계(다음 출전, 예선 끝)와 자동 재시작이 시험 창 안에서 실제로 생기는지 assert한다.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import json
import math
import pickle
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab import cli, paths
from evolab.app import strings as S
from evolab.engine import seeds
from evolab.engine import trainer as trainer_mod
from evolab.engine.evaluate import EvalResult
from evolab.engine.pool import PoolError, WorkerPool
from evolab.engine.runsetup import create_run, platform_info
from evolab.engine.trainer import (
    LEARNER_COLUMNS,
    MAX_QUALIFY_ENTRANTS,
    NEW_RUN_DEFAULTS,
    RESTART_FROM,
    SWITCH_NONE,
    SWITCH_QUALIFY_NEXT,
    SWITCH_QUALIFY_PICK,
    SWITCH_RESTART_CHAMPION,
    SWITCH_RESTART_FRESH,
    Champion,
    LearnerRecord,
    TrainConfig,
    Trainer,
    TrainerChangeError,
    generation_trials,
    make_learner,
    qualify_budget_problem,
)
from evolab.policy.genome import layout_for
from evolab.runs import store as store_mod
from evolab.runs.bundle import compare_stats, export_bundle, reproduce
from evolab.runs.catalog import branch_run
from evolab.runs.store import CheckpointPolicy, RunStore
from evolab.spec.io import compile_env

LANDER = paths.builtin_env_dir() / "lander.evoenv"
LANDER_TEXT = LANDER.read_text(encoding="utf-8")
LANDER_SPEC = compile_env(LANDER_TEXT)
LAYOUT = layout_for(LANDER_SPEC)
TIME = set(trainer_mod.TIME_COLUMNS)
HANGUL = re.compile(r"[가-힣]")
V10_RUN = Path(__file__).parent / "data" / "v10_run_ga" / "run"
METHODS = ("cmaes", "ga", "sa")
KEEP_ALL = CheckpointPolicy(keep_last=10**6)
FAST = {"validation_seeds": 4, "validation_candidates": 1}
"""실제 착륙선 테스트용: 검증 세트를 줄여 빠르게(검증 주기·판단 규칙은 그대로)."""


def per_method(method: str) -> dict[str, Any]:
    """세 방법 모두 세대당 학습 시험 12회(시험 2회 × 후보 6)."""
    return {"cmaes": {"popsize": 6}, "ga": {"popsize": 6}, "sa": {"sa_neighbors": 5}}[method]


def small_config(**over: Any) -> TrainConfig:
    """test_engine.small_config와 같은 바탕: 개체 6, 시험 2(세대당 12회), 검증 주기 24."""
    return TrainConfig(**{"popsize": 6, "trials": 2, "validate_every": 24, "checkpoint_every": 2, "run_seed": 7,
                          **over})


def strip(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in r.items() if k not in TIME} for r in rows]


def lander_text(gravity: float) -> str:
    data = json.loads(LANDER_TEXT)
    data["world"]["gravity"] = [0.0, gravity]
    return json.dumps(data, ensure_ascii=False, indent=2)


def make_store(base: Path, name: str, policy: CheckpointPolicy | None = None) -> RunStore:
    store = RunStore.create(base, name, {"env_name": "착륙선"}, policy=policy)
    store.add_env_version(LANDER_TEXT, 0)
    return store


def same_state(a: Any, b: Any) -> bool:
    """학습 방법 상태 비교(배열은 바이트, NaN은 NaN끼리 같음)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return bool(a == b)


# ---------------------------------------------------------------------------
# 가짜 평가기
# ---------------------------------------------------------------------------

VALIDATION_SEEDS = {seeds.derive(seeds.TAG_VALIDATION, k) for k in range(64)}
COMPARE_SEEDS = {seeds.derive(seeds.TAG_COMPARE, k) for k in range(64)}
Script = Callable[[int, int, int], float]


class FakeEvaluator:
    """착륙선을 돌리지 않는 평가기.

    학습 시험은 모두 train 점수, 검증 시험([1, k])은 validation 점수다. validation이 함수면
    (학습 방법 번호, 그 학습 방법의 검증 수(이번 포함), 후보 번호)로 정한다(ScriptedEvaluator). 숫자면 모든 쌍에 같은
    점수라 정체가 확실히 생긴다(ConstantEvaluator).
    """

    def __init__(self, validation: float | Script = 5.0, train: float = 0.0) -> None:
        self.validation = validation
        self.train = train
        self.trainer: Trainer | None = None
        self.train_seeds: list[list[int]] = []
        """세대마다 학습 시험 시드."""
        self._blocks: dict[str, int] = {}

    def evaluate(self, env_version: str, env_text: str, layout: Any, genomes: np.ndarray,
                 pairs: list[tuple[int, Any]], record: list[bool] | None = None) -> EvalResult:
        if env_version not in self._blocks:
            self._blocks[env_version] = len(compile_env(env_text).file.scoring.blocks)
        nb, n = self._blocks[env_version], len(pairs)
        scores = np.zeros(n)
        first = pairs[0][1].seed
        if first in VALIDATION_SEEDS:
            assert self.trainer is not None
            cur = self.trainer.state.current
            v = self.validation
            for j, (i, _) in enumerate(pairs):
                scores[j] = v(cur.index, cur.validations, i) if callable(v) else v
        elif first not in COMPARE_SEEDS:
            self.train_seeds.append(sorted({t.seed for _, t in pairs}))
            scores[:] = self.train
        return EvalResult(scores=scores, steps=np.ones(n, dtype=np.int64), end_events=["timeout"] * n,
                          outcomes=["failure"] * n, blocks=np.zeros((n, nb)), action_abs=np.zeros((n, 3)),
                          weighted=np.zeros((n, nb)))


def fake_trainer(validation: float | Script = 5.0, store: Any = None, **over: Any) -> tuple[Trainer, FakeEvaluator]:
    ev = FakeEvaluator(validation)
    t = Trainer(LANDER_TEXT, small_config(**over), ev, store)
    ev.trainer = t
    return t, ev


def steps(t: Trainer, n: int) -> list[dict[str, Any]]:
    return [t.step() for _ in range(n)]


def switches(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [(int(r["generation"]), int(r["switch"])) for r in rows if int(r["switch"])]


# ---------------------------------------------------------------------------
# A. 설정·시드
# ---------------------------------------------------------------------------


def test_learner_seed_words_are_explicit_and_spaces_stay_disjoint() -> None:
    mask = 0xFFFFFFFF
    assert seeds.SEED_SCHEME_VERSION == 2
    for s in range(51):
        assert seeds.learner_seed(s) == seeds.learner_seed(s, 0) == seeds.derive(4, s) & mask  # v1.0과 같은 값
        for n in range(1, 21):
            assert seeds.learner_seed(s, n) == seeds.derive(4, s, n) & mask  # 끝 0 처리에 기대지 않음
    learner = {seeds.learner_seed(s, n) for s in range(51) for n in range(21)}
    assert len(learner) == 51 * 21
    words = {seeds.derive(4, s) for s in range(51)} | {seeds.derive(4, s, n) for s in range(51) for n in range(1, 21)}
    others = ({t.seed for s in range(4) for g in range(5) for t in seeds.train_trials(s, g, 5)}
              | {t.seed for t in seeds.validation_trials() + seeds.compare_trials() + seeds.judge_trials()})
    assert not words & others
    assert not learner & {x & mask for x in others}


def test_dataclass_defaults_are_off_and_new_runs_get_recommended_values() -> None:
    c = TrainConfig()
    assert (c.restart_patience, c.restart_from, c.qualify_entrants, c.qualify_trials) == (None, "fresh", 1, 12000)
    n = TrainConfig.for_new_run()
    assert {k: getattr(n, k) for k in NEW_RUN_DEFAULTS} == NEW_RUN_DEFAULTS
    assert dataclasses.replace(n, **{k: getattr(c, k) for k in NEW_RUN_DEFAULTS}) == c  # 나머지 칸은 그대로
    n.validate()
    assert TrainConfig.for_new_run(method="ga", restart_patience=None).restart_patience is None
    assert TrainConfig.from_dict({}) == TrainConfig()
    old = json.loads((V10_RUN / "run.json").read_text(encoding="utf-8"))["config"]
    assert not set(NEW_RUN_DEFAULTS) & set(old)
    v10 = TrainConfig.from_dict(old)
    assert (v10.restart_patience, v10.restart_from, v10.qualify_entrants) == (None, "fresh", 1)
    full = TrainConfig.for_new_run(qualify_entrants=3, qualify_trials=4000, restart_from="alternate")
    assert TrainConfig.from_dict(full.to_dict()) == full
    assert TrainConfig.from_dict(json.loads(json.dumps(full.to_dict()))) == full
    assert {"restart_patience", "restart_from"} <= set(trainer_mod.LIVE_CONFIG_KEYS)
    assert not {"qualify_entrants", "qualify_trials"} & set(trainer_mod.LIVE_CONFIG_KEYS)
    assert RESTART_FROM == ("fresh", "champion", "alternate") and MAX_QUALIFY_ENTRANTS == 8
    assert trainer_mod.CHECKPOINT_FORMAT == 2 and trainer_mod.READ_FORMATS == (1, 2)
    assert LEARNER_COLUMNS == ("learner", "learner_trials", "learner_best", "learner_stall", "restarts",
                               "qualifying", "switch")
    assert not set(LEARNER_COLUMNS) & set(trainer_mod.TIME_COLUMNS)


def _invalid(name: str, value: Any) -> str:
    return S.TRAIN_CONFIG_INVALID.format(name=name, value=value)


@pytest.mark.parametrize(("over", "message"), [
    ({"restart_patience": 0}, _invalid("restart_patience", 0)),
    ({"restart_patience": True}, _invalid("restart_patience", True)),
    ({"restart_patience": 1.5}, _invalid("restart_patience", 1.5)),
    ({"restart_patience": 999}, S.TRAIN_RESTART_TOO_SHORT.format(value=999, every=1000)),
    ({"restart_from": "x"}, _invalid("restart_from", "x")),
    ({"restart_from": None}, _invalid("restart_from", None)),
    ({"qualify_entrants": 0}, S.TRAIN_QUALIFY_ENTRANTS.format(max=8, value=0)),
    ({"qualify_entrants": 9}, S.TRAIN_QUALIFY_ENTRANTS.format(max=8, value=9)),
    ({"qualify_entrants": True}, S.TRAIN_QUALIFY_ENTRANTS.format(max=8, value=True)),
    ({"qualify_entrants": 2.0}, S.TRAIN_QUALIFY_ENTRANTS.format(max=8, value=2.0)),
    ({"qualify_trials": 0}, _invalid("qualify_trials", 0)),
    ({"qualify_trials": None}, _invalid("qualify_trials", None)),
    ({"qualify_entrants": 2, "qualify_trials": 999}, S.TRAIN_QUALIFY_TOO_SHORT.format(value=999, every=1000)),
])
def test_validate_rejects_bad_stability_settings_in_korean(over: dict[str, Any], message: str) -> None:
    with pytest.raises(TrainerChangeError) as exc:
        TrainConfig(**over).validate()
    assert str(exc.value) == message and HANGUL.search(message)


def test_validate_accepts_edge_values() -> None:
    TrainConfig(restart_patience=1000).validate()  # 검증 주기와 같으면 된다
    TrainConfig(qualify_entrants=1, qualify_trials=1).validate()  # 예선이 꺼져 있으면 시험 수 하한을 보지 않는다
    TrainConfig(qualify_entrants=8, qualify_trials=1000).validate()
    for how in RESTART_FROM:
        TrainConfig(restart_from=how).validate()


def test_apply_config_rechecks_patience_against_validate_every() -> None:
    t, _ = fake_trainer(restart_patience=48)
    steps(t, 3)
    config, current = t.config.to_dict(), dataclasses.replace(t.state.current)
    with pytest.raises(TrainerChangeError) as exc:
        t.apply_config({"validate_every": 49})
    assert str(exc.value) == S.TRAIN_RESTART_TOO_SHORT.format(value=48, every=49)
    assert t.config.to_dict() == config and t.state.current == current
    assert t.apply_config({"validate_every": 48}) == {"validate_every": 48}
    assert t.state.current.validations == t.state.current.trials // 48


def test_generation_trials_and_qualify_budget() -> None:
    assert generation_trials(TrainConfig(popsize=6, trials=2), LANDER_SPEC) == 12
    assert generation_trials(TrainConfig(method="sa", sa_neighbors=5, popsize=99, trials=2), LANDER_SPEC) == 12
    assert generation_trials(TrainConfig(), LANDER_SPEC) == 32 * LANDER_SPEC.file.evaluation.trials
    assert qualify_budget_problem(TrainConfig(qualify_entrants=3, qualify_trials=100, max_evaluations=300)) == \
        S.TRAIN_QUALIFY_OVER_BUDGET.format(k=3, b=100, total=300, max=300)
    assert qualify_budget_problem(TrainConfig(qualify_entrants=3, qualify_trials=100, max_evaluations=301)) is None
    assert qualify_budget_problem(TrainConfig(qualify_entrants=3, qualify_trials=100)) is None  # 예산 없음
    assert qualify_budget_problem(TrainConfig(qualify_trials=10**6, max_evaluations=10)) is None  # 예선 끔


@pytest.mark.parametrize("method", METHODS)
def test_make_learner_index_and_start(method: str) -> None:
    config = TrainConfig(method=method, run_seed=5, **per_method(method))
    default, zero = make_learner(config, LAYOUT), make_learner(config, LAYOUT, index=0)
    assert default.config == zero.config and default.config.seed == seeds.learner_seed(5)
    a, b = default.state_dict(), zero.state_dict()
    if method == "cmaes":  # pycma 객체는 피클 바이트로 담겨 따로 만든 둘을 바이트로 비교하지 않는다
        a, b = {k: v for k, v in a.items() if k != "es"}, {k: v for k, v in b.items() if k != "es"}
    assert same_state(a, b)
    assert default.ask().tobytes() == zero.ask().tobytes()
    for n in (1, 2, 7):
        other = make_learner(config, LAYOUT, index=n)
        assert other.config.seed == seeds.learner_seed(5, n)
        assert other.ask().tobytes() != default.ask().tobytes()
    g = LAYOUT.initial() + 0.3 * np.random.default_rng(1).standard_normal(LAYOUT.dim)
    started = make_learner(config, LAYOUT, index=3, x0=g)
    assert np.array_equal(started.best(), g)  # CMA-ES는 평균, GA는 tell 전 x0, 담금질은 현재해
    assert np.array_equal(started.ask()[0], g) or method == "cmaes"  # GA 0행(엘리트 자리), 담금질 0행(현재해)


# ---------------------------------------------------------------------------
# B. 재시작 동작 (가짜 평가기로 정확한 시점)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_restart_happens_exactly_when_stall_reaches_patience(method: str) -> None:
    P = 48
    t, _ = fake_trainer(5.0, method=method, restart_patience=P, **per_method(method))
    rows = steps(t, 2)
    champion = (t.state.champion.genome.copy(), t.state.champion.score, t.state.champion.generation)
    rows += steps(t, 12)
    # 첫 검증(2세대, 학습 시험 24)에 기준점이 잡히고, 그 뒤 48회(6세대)에 바꾼다. 새 학습 방법도 같은 규칙.
    assert switches(rows) == [(6, SWITCH_RESTART_FRESH), (12, SWITCH_RESTART_FRESH)]
    for r in rows:
        assert (r["switch"] == SWITCH_RESTART_FRESH) == (r["learner_stall"] >= P)
    r6, r7 = rows[5], rows[6]
    assert (r6["learner"], r6["learner_stall"], r6["restarts"], r6["learner_trials"]) == (0, 48, 0, 72)
    assert (r7["learner"], r7["restarts"], r7["learner_trials"], r7["learner_best"]) == (1, 1, 12, "")
    assert [r["learner"] for r in rows] == [0] * 6 + [1] * 6 + [2] * 2
    # 챔피언·기록은 그대로 이어진다
    assert [r["champion_generation"] for r in rows[1:]] == [2] * 13
    assert all(r["champion_score"] == 5.0 for r in rows[1:])
    assert t.state.champion.genome.tobytes() == champion[0].tobytes()
    assert (t.state.champion.score, t.state.champion.generation) == champion[1:]
    assert [r["evaluations"] for r in rows] == [12 * g for g in range(1, 15)]
    assert (t.state.restarts, t.state.learners, t.state.current.index, t.state.current.trials) == (2, 3, 2, 24)


def test_restart_judges_the_current_learner_not_the_champion() -> None:
    def script(index: int, validations: int, cand: int) -> float:
        # 학습 방법 0: 100에서 멈춤 → 재시작. 새 학습 방법: 챔피언보다 낮지만 검증마다 오름 → 재시작하지 않음.
        return 100.0 if index == 0 else -100.0 + validations

    t, _ = fake_trainer(script, restart_patience=48)
    rows = steps(t, 30)
    assert switches(rows) == [(6, SWITCH_RESTART_FRESH)]
    assert max(r["learner_stall"] for r in rows[6:]) == 12
    assert (t.state.restarts, t.state.champion.score, t.state.champion.generation) == (1, 100.0, 2)
    assert rows[-1]["learner_best"] == -100.0 + 12 and rows[-1]["champion_score"] == 100.0


@pytest.mark.parametrize("method", METHODS)
def test_restart_from_fresh_champion_and_alternate(method: str) -> None:
    t, _ = fake_trainer(5.0, method=method, restart_patience=48, restart_from="alternate", **per_method(method))
    codes = []
    for _ in range(19):
        r = t.step()
        if not r["switch"]:
            continue
        codes.append(r["switch"])
        first = t.learner.ask()
        if r["switch"] == SWITCH_RESTART_FRESH:  # 처음 유전자 + 새 시드 = 그 번호로 새로 만든 학습 방법
            assert first.tobytes() == make_learner(t.config, t.layout, index=t.state.current.index).ask().tobytes()
        else:  # 챔피언 유전자가 출발점(CMA-ES 평균, GA 0행, 담금질 0행)
            start = t.learner.best() if method == "cmaes" else first[0]
            assert np.array_equal(start, t.state.champion.genome)
    assert codes == [SWITCH_RESTART_FRESH, SWITCH_RESTART_CHAMPION, SWITCH_RESTART_FRESH]


def test_champion_restart_without_champion_starts_fresh() -> None:
    t, _ = fake_trainer(-math.inf, restart_patience=24, restart_from="champion")  # 검증 점수가 챔피언이 되지 못함
    rows = steps(t, 5)
    assert switches(rows) == [(2, SWITCH_RESTART_FRESH), (4, SWITCH_RESTART_FRESH)]
    assert t.state.champion.genome is None and rows[-1]["champion_score"] == ""
    t.state.champion = Champion(LAYOUT.initial(), 1.0, 1)
    t.state.current.trials = 24
    assert t._decide() == SWITCH_RESTART_CHAMPION
    t.state.champion = Champion()
    assert t._decide() == SWITCH_RESTART_FRESH
    t.apply_config({"restart_patience": None})
    assert t._decide() == SWITCH_NONE


def test_ipop_population_returns_to_config_popsize_after_restart() -> None:
    t, _ = fake_trainer(5.0, ipop=True, restart_patience=48)
    steps(t, 2)
    if t.learner.popsize == t.config.popsize:
        t.learner._restart()  # type: ignore[attr-defined]  # IPOP 재시작(개체 수 2배)
    assert t.learner.popsize > t.config.popsize
    for _ in range(30):
        if t.step()["switch"]:
            break
    assert t.state.restarts == 1
    assert t.learner.popsize == t.config.popsize and t.learner.restarts == 0  # type: ignore[attr-defined]


def test_restart_clock_is_recorded_but_changes_nothing_until_it_fires() -> None:
    runs = []
    for p in (None, 10**9):
        with WorkerPool(0) as pool:
            t = Trainer(LANDER_TEXT, small_config(max_generations=4, restart_patience=p, **FAST), pool)
            rows: list[dict[str, Any]] = []
            t.run(rows.append)
        runs.append(rows)
    assert strip(runs[0]) == strip(runs[1])
    rows = runs[0]
    assert all(set(LEARNER_COLUMNS) <= set(r) for r in rows)
    assert list(rows[0])[list(rows[0]).index("compare_evaluations") + 1:][:8] == [*LEARNER_COLUMNS, "sigma"]
    assert [(r["learner"], r["restarts"], r["qualifying"], r["switch"]) for r in rows] == [(0, 0, 0, 0)] * 4
    assert [r["learner_trials"] for r in rows] == [r["evaluations"] for r in rows]


# ---------------------------------------------------------------------------
# C. 결정성 (실제 착륙선): 예선 3 × 36회, 재시작 24회, 13세대 안에 모든 경계가 생긴다
# ---------------------------------------------------------------------------

DET_N = 13
DET = {"cmaes": (4, "alternate"), "ga": (1, "champion"), "sa": (4, "alternate")}
"""방법별 (실행 시드, 새로 시작할 곳). 테스트를 쓸 때 돌려 보고 고른 시드: 통과자가 출전 2, 1, 3으로 다르다."""


def det_config(method: str, **over: Any) -> TrainConfig:
    seed, how = DET[method]
    return TrainConfig(**{"method": method, "run_seed": seed, "trials": 2, "validate_every": 24, **FAST,
                          "checkpoint_every": 1, "restart_patience": 24, "restart_from": how, "qualify_entrants": 3,
                          "qualify_trials": 36, "max_generations": DET_N, **per_method(method), **over})


@dataclasses.dataclass
class Straight:
    store: RunStore
    rows: list[dict[str, Any]]
    stats: list[dict[str, str]]
    trainer: Trainer
    closing: dict[int, dict[str, Any]]
    """출전 번호 → 그 출전이 끝날 때(교체 직전) 학습 방법 상태."""
    restart_gen: int


_STRAIGHT: dict[str, Straight] = {}


def straight(method: str, factory: pytest.TempPathFactory) -> Straight:
    """끊지 않은 기준 실행(방법마다 한 번). 체크포인트를 모두 남긴다."""
    if method not in _STRAIGHT:
        store = make_store(factory.mktemp(f"straight-{method}"), "한 번에", KEEP_ALL)
        closing: dict[int, dict[str, Any]] = {}
        with WorkerPool(0) as pool:
            t = Trainer(LANDER_TEXT, det_config(method), pool, store)
            real = t._apply_switch

            def hook(code: int) -> None:
                if code in (SWITCH_QUALIFY_NEXT, SWITCH_QUALIFY_PICK):
                    closing[t.state.current.index] = pickle.loads(pickle.dumps(t.learner.state_dict()))
                real(code)

            t._apply_switch = hook  # type: ignore[method-assign]
            rows: list[dict[str, Any]] = []
            t.run(rows.append)
        found = switches(rows)
        # 경계가 시험 창 안에서 실제로 생긴다: 예선 다음 출전 2번(3, 6세대), 예선 끝(9세대), 자동 재시작
        assert found[:3] == [(3, SWITCH_QUALIFY_NEXT), (6, SWITCH_QUALIFY_NEXT), (9, SWITCH_QUALIFY_PICK)]
        restarts = [g for g, c in found if c in (SWITCH_RESTART_FRESH, SWITCH_RESTART_CHAMPION)]
        assert restarts and restarts[0] < DET_N
        _STRAIGHT[method] = Straight(store, rows, strip(store.read_stats()), t, closing, restarts[0])
    return _STRAIGHT[method]


def next_ask(t: Trainer) -> np.ndarray:
    """다음 세대 후보. 트레이너의 학습 방법은 건드리지 않는다(CMA-ES ask()는 stats()의 diversity를 바꾼다)."""
    learner = make_learner(t.config, t.layout, index=t.state.current.index)
    learner.load_state_dict(t.learner.state_dict())
    return learner.ask()


def same_end(t: Trainer, base: Straight) -> None:
    b = base.trainer
    assert t.learner.best().tobytes() == b.learner.best().tobytes()
    assert str(t.learner.stats()) == str(b.learner.stats())
    assert dataclasses.asdict(t.state.current) == dataclasses.asdict(b.state.current)
    assert (t.state.restarts, t.state.learners, t.state.qualifying) == (b.state.restarts, b.state.learners, False)
    assert t.state.champion.genome.tobytes() == b.state.champion.genome.tobytes()  # type: ignore[union-attr]
    assert next_ask(t).tobytes() == next_ask(b).tobytes()


@pytest.mark.parametrize("method", METHODS)
def test_switches_do_not_depend_on_workers_or_recording(method: str, tmp_path: Path,
                                                         tmp_path_factory: pytest.TempPathFactory) -> None:
    base = straight(method, tmp_path_factory)
    store = make_store(tmp_path, "워커 2, 관전", KEEP_ALL)
    with WorkerPool(2) as pool:
        t = Trainer(LANDER_TEXT, det_config(method), pool, store)
        while t.check_stop() is None:
            t.step(record=True)
        t.finish()
    assert strip(store.read_stats()) == base.stats
    for kind in ("champion", "representative"):
        gens = store.champion_generations(kind)
        assert gens == base.store.champion_generations(kind)
        for g in gens:
            a, b = store.load_champion(g, kind), base.store.load_champion(g, kind)
            assert sorted(a) == sorted(b) and all(a[k].tobytes() == b[k].tobytes() for k in a)
    champions = store.champion_generations("champion")
    assert {int(store.load_champion(g, "champion")["learner"]) for g in champions} >= {0}
    same_end(t, base)


@pytest.mark.parametrize("method", METHODS)
def test_resume_at_every_generation_matches_uninterrupted(method: str, tmp_path: Path,
                                                          tmp_path_factory: pytest.TempPathFactory) -> None:
    """모든 세대 경계(예선 다음, 예선 끝, 재시작 포함)에서 끊고 체크포인트로 이어 가도 같다."""
    base = straight(method, tmp_path_factory)
    store = make_store(tmp_path, "세대마다 끊고 이어서")
    with WorkerPool(0) as pool:
        t = Trainer(LANDER_TEXT, det_config(method), pool, store)
        t.step()
        while t.state.generation < DET_N:
            t = Trainer.resume(store, pool)  # 바로 앞 세대 체크포인트에서
            t.step()
    assert strip(store.read_stats()) == base.stats
    same_end(t, base)


@pytest.mark.parametrize("method", METHODS)
def test_branch_at_switch_boundaries_rebuilds_the_parent(method: str, tmp_path: Path,
                                                         tmp_path_factory: pytest.TempPathFactory) -> None:
    """예선 중(보관 출전자가 있는 3세대), 예선 끝, 재시작 경계에서 분기해 끊지 않고 이어 가면 부모와 같다."""
    base = straight(method, tmp_path_factory)
    for g in (3, 9, base.restart_gen):
        child = RunStore.open(branch_run(base.store.root, g, tmp_path / f"분기{g}"))
        _, ckpt = child.load_checkpoint()
        st = ckpt["state"]
        if g == 3:
            assert st["qualifying"] and st["reserve"]["record"]["index"] == 0 and st["current"]["index"] == 1
        else:
            assert not st["qualifying"] and st["reserve"] is None
        with WorkerPool(0) as pool:
            t = Trainer.resume(child, pool)
            t.run()
        assert strip(child.read_stats()) == base.stats, g
        same_end(t, base)


class FlakyEvaluator:
    """fail이 켜지면 다음 평가에서 PoolError(워커가 죽은 것처럼)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.fail = False

    def evaluate(self, *args: Any, **kwargs: Any) -> EvalResult:
        if self.fail:
            self.fail = False
            raise PoolError("worker died")
        return self.inner.evaluate(*args, **kwargs)


@pytest.mark.parametrize("where", ["evaluate", "save_champion"])
def test_failure_in_the_restart_generation_is_never_checkpointed(where: str, tmp_path: Path,
                                                                 tmp_path_factory: pytest.TempPathFactory) -> None:
    base = straight("cmaes", tmp_path_factory)
    R = base.restart_gen
    store = make_store(tmp_path, "재시작 세대 오류")
    with WorkerPool(0) as pool:
        ev = FlakyEvaluator(pool)
        t = Trainer(LANDER_TEXT, det_config("cmaes"), ev, store)
        while t.state.generation < R - 1:
            t.step()
        if where == "evaluate":
            ev.fail = True
            with pytest.raises(PoolError):
                t.step()
        else:  # 통계 줄은 썼고 학습 방법을 바꾸기 전에 멈춤
            real = store.save_champion

            def broken(generation: int, kind: str, *a: Any, **k: Any) -> None:
                if kind == "representative":
                    raise OSError("locked")
                real(generation, kind, *a, **k)

            store.save_champion = broken  # type: ignore[method-assign]
            with pytest.raises(OSError):
                t.step()
            store.save_champion = real  # type: ignore[method-assign]
            assert t.state.generation == R and t.state.restarts == 0
        t.checkpoint()  # 끝나지 않은 세대: 남기지 않는다
        assert store.checkpoints()[-1] == R - 1
        resumed = Trainer.resume(store, pool)
        resumed.run()
    assert strip(store.read_stats()) == base.stats
    same_end(resumed, base)


def test_training_trials_never_repeat_across_learners() -> None:
    t, ev = fake_trainer(5.0, restart_patience=24, qualify_entrants=3, qualify_trials=24)
    rows = steps(t, 30)
    assert len(switches(rows)) >= 8
    assert len(ev.train_seeds) == 30 and all(len(g) == 2 for g in ev.train_seeds)
    flat = [s for g in ev.train_seeds for s in g]
    assert len(set(flat)) == len(flat)  # 학습 시험 시드는 전역 세대 번호를 쓴다


# ---------------------------------------------------------------------------
# D. 예선
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_qualifying_winner_is_the_best_entrant_and_continues_from_its_saved_state(
        method: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    base = straight(method, tmp_path_factory)
    rows = base.rows
    ends = [r for r in rows if r["switch"] in (SWITCH_QUALIFY_NEXT, SWITCH_QUALIFY_PICK)]
    assert [r["learner"] for r in ends] == [0, 1, 2] and all(r["qualifying"] == 1 for r in rows[:9])
    bests = [r["learner_best"] for r in ends]
    winner = bests.index(max(bests))  # 첫 최댓값
    after = rows[9]  # 10세대: 통과자가 이어 간다
    assert (after["learner"], after["qualifying"], after["restarts"]) == (winner, 0, 0)
    assert after["learner_trials"] == ends[winner]["learner_trials"] + 12 == 48
    # 통과자 = 챔피언을 낸 출전자(같은 검증 점수 최댓값을 같은 부등호로 따라감)
    champ_gen = rows[8]["champion_generation"]
    assert rows[champ_gen - 1]["learner"] == winner
    assert int(base.store.load_champion(champ_gen, "champion")["learner"]) == winner
    assert rows[8]["champion_score"] == max(bests)
    # 통과자의 다음 ask() = 그 출전자가 끝날 때 보관한 상태의 ask()
    config = det_config(method)
    saved = make_learner(config, LAYOUT, index=winner)
    saved.load_state_dict(base.closing[winner])
    _, ckpt = base.store.load_checkpoint(9)
    assert ckpt["state"]["current"]["index"] == winner and ckpt["state"]["learners"] == 3
    resumed = make_learner(config, LAYOUT, index=winner)
    resumed.load_state_dict(ckpt["learner"])
    assert resumed.ask().tobytes() == saved.ask().tobytes()
    # 예선 뒤 첫 재시작 번호는 K, 만든 수 = K + 재시작 수
    first = rows[base.restart_gen]
    assert (first["learner"], first["restarts"], first["learner_trials"]) == (3, 1, 12)
    assert base.trainer.state.learners == 3 + base.trainer.state.restarts


@pytest.mark.parametrize(("scores", "winner"), [
    ((7.0, 7.0, 7.0), 0), ((1.0, 7.0, 7.0), 1), ((1.0, 2.0, 7.0), 2), ((9.0, 2.0, 7.0), 0),
])
def test_qualifying_tie_goes_to_the_earlier_entrant(scores: tuple[float, float, float], winner: int) -> None:
    t, _ = fake_trainer(lambda index, v, c: scores[index] if index < 3 else 0.0, qualify_entrants=3,
                        qualify_trials=24)
    rows = steps(t, 7)
    assert [r["switch"] for r in rows[:6]] == [0, SWITCH_QUALIFY_NEXT, 0, SWITCH_QUALIFY_NEXT, 0, SWITCH_QUALIFY_PICK]
    assert rows[6]["learner"] == winner and t.state.current.index == winner
    assert (t.state.learners, t.state.qualifying, t.state.reserve) == (3, False, None)
    assert t.state.champion.score == max(scores)
    assert rows[t.state.champion.generation - 1]["learner"] == winner


def test_no_restart_during_qualifying_and_winner_keeps_its_clock() -> None:
    # 출전마다 60회(5세대). 출전 안에서 정체가 P(24)를 넘어도 예선 중에는 바꾸지 않고, 통과자(동점이라 출전 1)는
    # 자기 시험 수·기준점·정체 시계를 이어 써서 예선이 끝난 바로 다음 세대에 재시작한다. 재시작 번호는 K부터.
    t, _ = fake_trainer(5.0, restart_patience=24, qualify_entrants=3, qualify_trials=60)
    rows = steps(t, 20)
    assert switches(rows)[:4] == [(5, SWITCH_QUALIFY_NEXT), (10, SWITCH_QUALIFY_NEXT), (15, SWITCH_QUALIFY_PICK),
                                  (16, SWITCH_RESTART_FRESH)]
    assert max(r["learner_stall"] for r in rows[:15]) == 36
    assert (rows[15]["learner"], rows[15]["learner_trials"], rows[15]["learner_stall"]) == (0, 72, 48)
    assert (rows[16]["learner"], rows[16]["restarts"]) == (3, 1)
    assert t.state.learners == 3 + t.state.restarts


def test_compare_set_is_not_used_for_selection(monkeypatch: pytest.MonkeyPatch,
                                               tmp_path_factory: pytest.TempPathFactory) -> None:
    base = straight("cmaes", tmp_path_factory)
    monkeypatch.setattr(seeds, "compare_trials", lambda count=seeds.COMPARE_SET: seeds.fixed_trials(seeds.TAG_JUDGE, 8))
    with WorkerPool(0) as pool:
        t = Trainer(LANDER_TEXT, det_config("cmaes"), pool)
        rows: list[dict[str, Any]] = []
        t.run(rows.append)

    def keep(rs: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [{k: v for k, v in r.items() if not k.startswith("compare_")} for r in strip(rs)]

    assert keep(rows) == keep(base.rows)
    assert strip(rows) != strip(base.rows)  # 비교 값은 실제로 바뀌었다
    assert t.state.champion.genome.tobytes() == base.trainer.state.champion.genome.tobytes()  # type: ignore[union-attr]


@pytest.mark.parametrize("trials", [30, 26])
def test_each_entrant_is_validated_at_its_last_generation(trials: int) -> None:
    """출전 시험 수가 세대 크기(12)나 검증 주기(24)의 배수가 아니어도 출전마다 마지막 세대에 검증한다."""
    t, _ = fake_trainer(lambda index, v, c: float(index), qualify_entrants=3, qualify_trials=trials)
    rows = steps(t, 9)
    ends = [r for r in rows if r["switch"] in (SWITCH_QUALIFY_NEXT, SWITCH_QUALIFY_PICK)]
    assert [r["generation"] for r in ends] == [3, 6, 9]
    assert all(r["validation_best"] != "" and r["learner_trials"] == 36 for r in ends)
    assert [r["validation_best"] != "" for r in rows] == [False, True, True] * 3
    assert (t.state.current.index, t.state.current.validations) == (2, 1)  # 통과자: 36 // 24
    assert t.step()["validation_best"] != ""  # 통과자는 자기 시험 48회에서 다시 검증


@pytest.mark.parametrize("n", [1, 2])
def test_stagnation_stop_rests_during_qualifying(n: int) -> None:
    t, _ = fake_trainer(5.0, qualify_entrants=3, qualify_trials=24, stagnation_generations=n, max_generations=40)
    assert t.run() == "stagnation"
    assert t.state.generation == 6 + n  # 예선 끝(6세대)부터 N세대
    alone, _ = fake_trainer(5.0, stagnation_generations=n, max_generations=40)
    assert alone.run() == "stagnation" and alone.state.generation < 6


def test_environment_change_is_refused_during_qualifying() -> None:
    t, _ = fake_trainer(5.0, qualify_entrants=2, qualify_trials=24)
    hard = lander_text(-1.4)
    assert t.replace_env(hard) == t.env_version  # 0세대(평가 전)에는 바꿀 수 있다
    for gen, n in ((1, 1), (3, 2)):
        while t.state.generation < gen:
            t.step()
        before = (t.env_text, t.env_version, dataclasses.asdict(t.state.current), t.config.to_dict())
        with pytest.raises(TrainerChangeError) as exc:
            t.replace_env(LANDER_TEXT)
        assert str(exc.value) == S.TRAIN_ENV_DURING_QUALIFY.format(n=n, k=2)
        assert (t.env_text, t.env_version, dataclasses.asdict(t.state.current), t.config.to_dict()) == before
    t.step()  # 4세대 끝: 예선 끝
    assert not t.state.qualifying and math.isfinite(t.state.current.ref)
    old_version = t.env_version
    assert t.replace_env(LANDER_TEXT) != old_version  # 예선 뒤에는 바꿀 수 있고 기준점을 다시 잡는다
    assert (t.state.current.ref, t.state.current.ref_at) == (-math.inf, t.state.current.trials)
    t.state.current.ref, ref_at = 5.0, t.state.current.ref_at
    t.replace_env(LANDER_TEXT)  # 같은 버전: 그대로
    assert (t.state.current.ref, t.state.current.ref_at) == (5.0, ref_at)


# ---------------------------------------------------------------------------
# E. 이전 실행 호환
# ---------------------------------------------------------------------------


def downgrade_to_v10(run_dir: Path) -> None:
    """v1.1(끔)로 돈 실행을 v1.0이 남겼을 모양으로: 체크포인트 형식 1, run.json·stats.csv에 새 키·열 없음."""
    store = RunStore.open(run_dir, policy=KEEP_ALL)
    for g in store.checkpoints():
        _, d = store.load_checkpoint(g)
        st = dict(d["state"])
        assert (st["learners"], st["restarts"], st["qualifying"], st["reserve"]) == (1, 0, False, None)
        current = st["current"]
        for key in ("current", "learners", "restarts", "qualifying", "reserve"):
            del st[key]
        st["validations_done"] = current["validations"]
        config = {k: v for k, v in d["config"].items() if k not in NEW_RUN_DEFAULTS}
        store.save_checkpoint(g, {**d, "format": 1, "config": config, "state": st})
    info = store.read_run_info()
    store.update_run_info(config={k: v for k, v in info["config"].items() if k not in NEW_RUN_DEFAULTS},
                          seed_scheme=1)
    path = run_dir / "stats.csv"
    with open(path, encoding="utf-8-sig", newline="") as f:
        table = list(csv.reader(f))
    keep = [j for j, name in enumerate(table[0]) if name not in LEARNER_COLUMNS]
    text = "".join(store_mod._csv_line([row[j] for j in keep]) for row in table)
    path.write_bytes(store_mod._BOM + text.encode("utf-8"))


def v11_run(base: Path, gens: int, **over: Any) -> Path:
    """v1.1(끔)로 gens세대 돈 실행 폴더."""
    config = small_config(max_generations=gens, **FAST, **over)
    store = create_run(base, LANDER_TEXT, LANDER_SPEC, config, 0)
    with WorkerPool(0) as pool:
        Trainer(LANDER_TEXT, config, pool, store).run()
    return store.root


def test_v10_shaped_run_resumes_exactly_like_v10(tmp_path: Path) -> None:
    whole = RunStore.open(v11_run(tmp_path / "한 번에", 6)).read_stats()
    run = v11_run(tmp_path / "옛 모양", 3)
    downgrade_to_v10(run)
    store = RunStore.open(run)
    header = store.stats_columns()
    assert header is not None and not set(LEARNER_COLUMNS) & set(header)
    _, ckpt = store.load_checkpoint()
    assert ckpt["format"] == 1 and "validations_done" in ckpt["state"]
    with WorkerPool(0) as pool:
        t = Trainer.resume(store, pool, None, max_generations=6)
        assert (t.config.restart_patience, t.config.qualify_entrants) == (None, 1)
        assert t.state.current == LearnerRecord(0, 36, 1, t.state.best_validation, t.state.best_validation, 36)
        t.run()
    assert store.stats_columns() == header
    assert strip(store.read_stats()) == [{k: r[k] for k in header if k not in TIME} for r in strip(whole)]
    assert store.load_checkpoint()[1]["format"] == 2  # 다음 체크포인트부터 형식 2


def copy_v10(tmp_path: Path) -> Path:
    dst = tmp_path / "v10" / V10_RUN.name
    shutil.copytree(V10_RUN, dst)
    return dst


def same_platform(info: dict[str, Any]) -> bool:
    now = platform_info(0)
    return all(info["platform"].get(k) == now[k] for k in ("python", "numpy", "pymunk", "cma", "cpu"))


def test_real_v10_run_opens_and_resumes_with_restart_off(tmp_path: Path) -> None:
    store = RunStore.open(copy_v10(tmp_path))
    header = store.stats_columns()
    assert header is not None and not set(LEARNER_COLUMNS) & set(header)
    _, ckpt = store.load_checkpoint()
    assert ckpt["format"] == 1
    with WorkerPool(0) as pool:
        t = Trainer.resume(store, pool, None, max_generations=6)
        assert (t.config.restart_patience, t.config.qualify_entrants, t.state.learners) == (None, 1, 1)
        assert (t.state.current.trials, t.state.current.validations) == (48, ckpt["state"]["validations_done"])
        assert t.state.current.best == ckpt["state"]["best_validation"]
        rows: list[dict[str, Any]] = []
        t.run(rows.append)
    assert [list(r) for r in rows] == [header, header]
    assert store.stats_columns() == header and len(store.read_stats()) == 6
    assert store.load_checkpoint(6)[1]["format"] == 2


def test_real_v10_run_recomputes_its_rows_and_bundle_on_the_same_platform(tmp_path: Path) -> None:
    run = copy_v10(tmp_path)
    store = RunStore.open(run)
    if not same_platform(store.read_run_info()):
        pytest.skip("v1.0 기록과 다른 플랫폼: 숫자 비교는 같은 플랫폼에서만")
    recorded = store.read_stats()
    bundle = export_bundle(run, tmp_path / "v10.zip")
    with WorkerPool(0) as pool:
        Trainer.resume(store, pool, 2).run()  # 2세대에서 이어 4세대까지 다시 계산
    assert strip(store.read_stats()) == strip(recorded)
    new = reproduce(bundle, tmp_path / "다시", workers=0)
    again = RunStore.open(new).read_stats()
    assert list(again[0]) == list(recorded[0]) and compare_stats(recorded, again) is None


def _ckpt_variants(good: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []

    def variant(name: str, edit: Callable[[dict[str, Any]], None]) -> None:
        d = copy.deepcopy(good)
        edit(d)
        out.append((name, d))

    def st(d: dict[str, Any]) -> dict[str, Any]:
        return d["state"]  # type: ignore[no-any-return]

    variant("format 3", lambda d: d.update(format=3))
    variant("format 1 with qualifying", lambda d: (d.update(format=1), st(d).update(validations_done=1)))
    variant("broken reserve state", lambda d: st(d)["reserve"]["learner"].update(es=b"broken"))
    variant("reserve learner not a dict", lambda d: st(d)["reserve"].update(learner=[1, 2]))
    variant("reserve record missing key", lambda d: st(d)["reserve"]["record"].pop("ref"))
    variant("restarts during qualifying", lambda d: st(d).update(restarts=1))
    variant("reserve index not before current", lambda d: st(d)["reserve"]["record"].update(index=1))
    variant("no reserve for entrant 2", lambda d: st(d).update(reserve=None))
    variant("reserve for entrant 1", lambda d: (st(d)["current"].update(index=0), st(d).update(learners=1)))
    variant("learners too many", lambda d: st(d).update(learners=3))
    variant("index past learners", lambda d: st(d)["current"].update(index=2))
    variant("negative index", lambda d: st(d)["current"].update(index=-1))
    variant("reserve outside qualifying", lambda d: st(d).update(qualifying=False))
    variant("too few learners after qualifying",
            lambda d: st(d).update(qualifying=False, reserve=None))
    variant("qualifying not bool", lambda d: st(d).update(qualifying=1))
    variant("current not a record", lambda d: st(d).update(current=[0, 1]))
    variant("current trials float", lambda d: st(d)["current"].update(trials=36.0))
    variant("missing learners", lambda d: st(d).pop("learners"))
    return out


def test_bad_checkpoints_are_refused_and_the_trainer_is_unchanged() -> None:
    t, _ = fake_trainer(5.0, qualify_entrants=3, qualify_trials=24)
    steps(t, 3)  # 출전 2 진행 중, 보관 출전자 = 출전 1
    good = t.state_dict()
    assert good["format"] == 2 and good["state"]["reserve"]["record"]["index"] == 0
    for name, bad in _ckpt_variants(good):
        fresh, _ = fake_trainer(5.0, qualify_entrants=3, qualify_trials=24)
        learner, state = fresh.learner, fresh.state
        with pytest.raises(ValueError):
            fresh.load_state_dict(bad)
        assert fresh.learner is learner and fresh.state is state, name
    fresh, _ = fake_trainer(5.0, qualify_entrants=3, qualify_trials=24)
    fresh.load_state_dict(good)
    assert dataclasses.asdict(fresh.state.current) == dataclasses.asdict(t.state.current)
    assert fresh.state.reserve is not None and fresh.state.reserve.record == t.state.reserve.record  # type: ignore[union-attr]
    assert fresh.learner.config.seed == seeds.learner_seed(7, 1)  # type: ignore[attr-defined]
    # 예선을 켠 설정으로 형식 1(v1.0) 체크포인트는 열 수 없다
    plain, _ = fake_trainer(5.0)
    steps(plain, 2)
    v10 = plain.state_dict()
    v10["format"] = 1
    v10["state"]["validations_done"] = 1
    other, _ = fake_trainer(5.0, qualify_entrants=2, qualify_trials=24)
    with pytest.raises(ValueError):
        other.load_state_dict(v10)
    for key in ("current", "learners", "restarts", "qualifying", "reserve"):
        del v10["state"][key]
    off, _ = fake_trainer(5.0)
    off.load_state_dict(v10)
    assert off.state.current == LearnerRecord(0, 24, 1, 5.0, 5.0, 24)


def test_v10_shaped_bundle_reproduces_with_the_old_columns(tmp_path: Path) -> None:
    run = v11_run(tmp_path / "옛 모양", 4)
    downgrade_to_v10(run)
    bundle = export_bundle(run, tmp_path / "묶음.zip")
    new = reproduce(bundle, tmp_path / "다시", workers=0)
    a, b = RunStore.open(run).read_stats(), RunStore.open(new).read_stats()
    assert list(b[0]) == list(a[0]) and not set(LEARNER_COLUMNS) & set(b[0])
    assert compare_stats(a, b) is None
    assert TrainConfig.from_dict(RunStore.open(new).read_run_info()["config"]).restart_patience is None


def test_enabling_restart_on_a_v10_run_is_reproduced(tmp_path: Path) -> None:
    """v1.0 실행을 이어 가며 재시작을 켜도(D17: 켜면 기준점을 다시 잡음) 묶음 재현이 같다."""
    run = v11_run(tmp_path / "옛 모양", 3)
    downgrade_to_v10(run)
    store = RunStore.open(run)
    with WorkerPool(0) as pool:
        t = Trainer.resume(store, pool)
        g = t.state.generation
        changes = t.apply_config({"restart_patience": 48})
        assert changes == {"restart_patience": 48}
        assert (t.state.current.ref, t.state.current.ref_at) == (-math.inf, 36)
        t.checkpoint()
        store.protect(g)
        store.append_event("config_change", g, changes=changes)
        while t.state.restarts == 0 and t.state.generation < 24:
            t.step()
        assert t.state.restarts == 1  # 재시작이 실제로 일어났다(옛 열 머리라 통계 열에는 남지 않음)
        steps(t, 2)
        t.checkpoint()
    assert "switch" not in store.stats_columns()  # type: ignore[operator]
    bundle = export_bundle(run, tmp_path / "묶음.zip")
    new = reproduce(bundle, tmp_path / "다시", workers=0)
    assert compare_stats(store.read_stats(), RunStore.open(new).read_stats()) is None


# ---------------------------------------------------------------------------
# F. 학습 중 변경, 명령줄
# ---------------------------------------------------------------------------


def test_apply_config_restart_keys_and_clock_rules() -> None:
    t, _ = fake_trainer(5.0)  # 재시작 끔
    for key, value in (("qualify_entrants", 2), ("qualify_trials", 48)):
        with pytest.raises(TrainerChangeError) as exc:
            t.apply_config({key: value})
        assert str(exc.value) == S.TRAIN_CHANGE_NOT_ALLOWED.format(keys=key)
    rows = steps(t, 10)  # 2세대 첫 검증 뒤로 오르지 않음: 정체 96회
    assert rows[-1]["learner_stall"] == 96 and not switches(rows)
    snap = t.changeable_state()
    assert t.apply_config({"restart_patience": 48, "restart_from": "champion"}) == {"restart_patience": 48,
                                                                                   "restart_from": "champion"}
    assert (t.state.current.ref, t.state.current.ref_at) == (-math.inf, 120)
    t.restore_changeable(snap)  # 저장 실패 등으로 되돌리면 기록도 되돌아간다
    assert t.config.restart_patience is None and (t.state.current.ref, t.state.current.ref_at) == (5.0, 24)
    # 켜도 바로 재시작하지 않고, 켠 뒤 첫 검증(12세대, 학습 시험 144)부터 48회 뒤(16세대)에 바꾼다
    t.apply_config({"restart_patience": 48})
    rows = steps(t, 10)
    assert switches(rows) == [(16, SWITCH_RESTART_FRESH)]
    assert [r["learner_stall"] for r in rows[:6]] == [12, 0, 12, 24, 36, 48]
    # 값만 바꾸면 시계는 그대로: 지금 정체(24) 아래로 내리면 다음 세대 끝에 재시작한다(손 재시작 대용)
    assert rows[-1]["generation"] == 20 and rows[-1]["learner_stall"] == 24 and t.state.current.index == 1
    ref = (t.state.current.ref, t.state.current.ref_at)
    t.apply_config({"restart_patience": 24})
    assert (t.state.current.ref, t.state.current.ref_at) == ref
    assert switches([t.step()]) == [(21, SWITCH_RESTART_FRESH)]
    # 끄면 판단하지 않지만 시계는 계속 기록한다
    t.apply_config({"restart_patience": None})
    rows = steps(t, 8)
    assert not switches(rows) and rows[-1]["learner_stall"] == 72


def _train(tmp_path: Path, *extra: str, gens: str = "2") -> tuple[int, list[Path]]:
    out = tmp_path / "runs"
    code = cli.main(["train", str(LANDER), "--seed", "1", "--popsize", "4", "--trials", "1", "--max-generations", gens,
                     "--validate-every", "8", "--workers", "0", "--out", str(out), *extra])
    return code, sorted(p for p in out.iterdir() if p.is_dir()) if out.exists() else []


def _stability_keys(run: Path) -> tuple[Any, ...]:
    c = RunStore.open(run).read_run_info()["config"]
    return tuple(c[k] for k in ("restart_patience", "restart_from", "qualify_entrants", "qualify_trials"))


def test_cli_new_runs_get_restart_by_default_and_take_stability_flags(tmp_path: Path,
                                                                      capsys: pytest.CaptureFixture[str]) -> None:
    code, runs = _train(tmp_path / "a", "--restart", "50", "--restart-from", "champion", "--qualify", "3",
                        "--qualify-trials", "100")
    assert code == 0 and _stability_keys(runs[0]) == (50, "champion", 3, 100)
    out = capsys.readouterr().out
    assert S.TRAIN_STABILITY.format(restart=S.TRAIN_STABILITY_RESTART_ON.format(p=50, where=S.RESTART_FROM_NAMES["champion"]),
                                    qualify=S.TRAIN_STABILITY_QUALIFY_ON.format(k=3, b=100)) in out
    assert RunStore.open(runs[0]).read_run_info()["seed_scheme"] == 2
    code, runs = _train(tmp_path / "b", gens="1")
    assert code == 0 and _stability_keys(runs[0]) == (20000, "fresh", 1, 12000)
    assert S.TRAIN_STABILITY.format(restart=S.TRAIN_STABILITY_RESTART_ON.format(p=20000, where=S.RESTART_FROM_NAMES["fresh"]),
                                    qualify=S.TRAIN_STABILITY_OFF) in capsys.readouterr().out
    code, runs = _train(tmp_path / "c", "--no-restart", gens="1")
    assert code == 0 and _stability_keys(runs[0])[0] is None
    with pytest.raises(SystemExit) as exc:
        _train(tmp_path / "d", "--restart", "50", "--no-restart")
    assert exc.value.code == 2


@pytest.mark.parametrize(("extra", "message"), [
    (["--qualify", "9"], S.TRAIN_QUALIFY_ENTRANTS.format(max=8, value=9)),
    (["--restart", "5"], S.TRAIN_RESTART_TOO_SHORT.format(value=5, every=8)),
    (["--qualify", "3", "--qualify-trials", "100", "--max-evaluations", "300"],
     S.TRAIN_QUALIFY_OVER_BUDGET.format(k=3, b=100, total=300, max=300)),
])
def test_cli_rejects_bad_stability_settings_before_creating_a_run(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                                  extra: list[str], message: str) -> None:
    code, runs = _train(tmp_path, *extra)
    assert code == 2 and not runs
    err = capsys.readouterr().err
    assert message in err and "Traceback" not in err


def test_cli_switch_lines_are_printed_at_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, runs = _train(tmp_path, "--qualify", "2", "--qualify-trials", "8", "--restart", "8", gens="5")
    assert code == 0
    out = capsys.readouterr().out
    rows = RunStore.open(runs[0]).read_stats()
    assert [r["switch"] for r in rows[:4]] == ["0", "3", "0", "4"]
    assert S.TRAIN_SWITCH_QUALIFY_NEXT.format(gen=2, n=1, score=f"{float(rows[1]['learner_best']):.1f}", next=2) in out
    winner = int(rows[4]["learner"])
    assert S.TRAIN_SWITCH_QUALIFY_PICK.split("{winner}")[0].format(gen=4) + str(winner + 1) in out


def test_cli_resume_changes_restart_like_the_lab(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, runs = _train(tmp_path)
    run = runs[0]
    store = RunStore.open(run)
    events, ckpts = store.read_events(), store.checkpoints()
    # 예선 설정은 이어 할 때 바꿀 수 없다(아무것도 건드리지 않고 종료 코드 2)
    for flags, extra in (("--qualify", ["--qualify", "3"]), ("--qualify-trials", ["--qualify-trials", "50"]),
                         ("--qualify, --qualify-trials", ["--qualify", "3", "--qualify-trials", "50"])):
        assert cli.main(["train", "--resume", str(run), "--workers", "0", *extra]) == 2
        assert S.TRAIN_RESUME_QUALIFY_FIXED.format(flags=flags) in capsys.readouterr().err
    # 검증 주기보다 짧은 P도 워커를 띄우고 뒤를 잘라 내기 전에 거른다
    assert cli.main(["train", "--resume", str(run), "--workers", "0", "--restart", "5"]) == 2
    assert S.TRAIN_RESTART_TOO_SHORT.format(value=5, every=8) in capsys.readouterr().err
    assert store.read_events() == events and store.checkpoints() == ckpts
    capsys.readouterr()
    assert cli.main(["train", "--resume", str(run), "--workers", "0", "--restart", "15000", "--restart-from",
                     "alternate", "--max-generations", "3"]) == 0
    out = capsys.readouterr().out
    change = [e for e in store.read_events() if e["kind"] == "config_change"]
    assert [(e["generation"], e["changes"]) for e in change] == [(2, {"restart_patience": 15000,
                                                                       "restart_from": "alternate"})]
    kinds = [e["kind"] for e in store.read_events()]
    assert kinds.index("resume") < kinds.index("config_change")
    assert 2 in store.read_run_info()["protected_checkpoints"]
    assert store.load_checkpoint(2)[1]["config"]["restart_patience"] == 15000
    assert S.TRAIN_CONFIG_CHANGED.format(gen=2, changes="restart_patience=15000, restart_from=alternate") in out
    # 같은 값이면 바뀐 것이 없어 사건을 남기지 않는다. --no-restart는 끈다.
    assert cli.main(["train", "--resume", str(run), "--workers", "0", "--restart", "15000"]) == 0
    assert len([e for e in store.read_events() if e["kind"] == "config_change"]) == 1
    assert cli.main(["train", "--resume", str(run), "--workers", "0", "--no-restart", "--max-generations", "4"]) == 0
    assert store.load_checkpoint()[1]["config"]["restart_patience"] is None

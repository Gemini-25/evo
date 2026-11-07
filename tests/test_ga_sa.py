"""유전 알고리즘과 담금질 (명세: 학습 방법, 유전자 벡터, 테스트 계획 "학습 방법", 검토 반영 결정 1·2).

- 학습 방법 규격(ask/tell, best, stats, state_dict)을 따른다
- 엘리트(GA)와 현재해(담금질)는 다음 ask에 그대로 들어가 이번 세대 시험으로 다시 평가받는다
- 같은 시드면 같은 결과, 상태 저장 → 피클 → 새 프로세스에서 이어 가기가 비트 단위로 같다(ask와 tell 사이 저장 포함)
- 몸 유전자 경계([-1, 1]) 잘라내기, NaN 점수 처리, 10차원 구면에서 같은 시험 횟수의 무작위 탐색보다 낫다
- Trainer에서 착륙선으로 몇 세대 돌고, 결과가 워커 수와 무관하다
"""

from __future__ import annotations

import math
import os
import pickle
import subprocess
import sys
import textwrap
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab import paths
from evolab.engine import seeds
from evolab.engine import trainer as trainer_mod
from evolab.engine.pool import WorkerPool
from evolab.engine.trainer import TrainConfig, Trainer, TrainerChangeError, make_learner
from evolab.learners.base import Learner
from evolab.learners.ga import GAConfig, GeneticAlgorithm
from evolab.learners.sa import ACCEPTANCE_WINDOW, SAConfig, SimulatedAnnealing
from evolab.policy.genome import layout_for
from evolab.runs.store import RunStore
from evolab.spec.io import compile_env

Objective = Callable[[np.ndarray], np.ndarray]
AnyLearner = GeneticAlgorithm | SimulatedAnnealing

LANDER_TEXT = (paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8")
TIME_COLUMNS = set(trainer_mod.TIME_COLUMNS)


def sphere(X: np.ndarray) -> np.ndarray:
    return -np.sum(X**2, axis=1)


def stepped(X: np.ndarray) -> np.ndarray:
    """계단 점수(동점이 많아 안정 정렬과 동점 처리도 지난다)."""
    return -np.round(np.sum((X - 0.3) ** 2, axis=1), 1)


def make(method: str, x0: np.ndarray, **kw: Any) -> AnyLearner:
    if method == "ga":
        return GeneticAlgorithm(x0, GAConfig(**kw))
    return SimulatedAnnealing(x0, SAConfig(**kw))


def run(learner: AnyLearner, f: Objective, gens: int) -> list[np.ndarray]:
    asked = []
    for _ in range(gens):
        X = learner.ask()
        asked.append(X.copy())
        learner.tell(X, f(X))
    return asked


def same(a: list[np.ndarray], b: list[np.ndarray]) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


METHODS = ["ga", "sa"]
SMALL = {"ga": {"popsize": 10, "elites": 2, "seed": 3}, "sa": {"neighbors": 6, "seed": 3}}


# ---------------------------------------------------------------------------- 규격


@pytest.mark.parametrize("method", METHODS)
def test_implements_learner_protocol(method: str) -> None:
    learner = make(method, np.zeros(7), **SMALL[method])
    assert isinstance(learner, Learner)
    assert learner.name == method
    rows = 10 if method == "ga" else 7  # 담금질: 이웃 6 + 현재해
    assert learner.popsize == rows
    keys = set(learner.stats())
    assert "sigma" in keys
    for _ in range(5):
        X = learner.ask()
        assert X.shape == (rows, 7) and X.dtype == np.float64 and X.flags.c_contiguous
        assert set(learner.stats()) == keys
        learner.tell(X, sphere(X))
        st = learner.stats()
        assert set(st) == keys
        assert all(isinstance(v, float) and math.isfinite(v) for v in st.values())
    b = learner.best()
    assert b.shape == (7,) and b.dtype == np.float64
    b[:] = 99.0  # 사본이라 학습 방법에 영향 없음
    assert not np.any(learner.best() == 99.0)
    X = learner.ask()
    X[:] = 99.0
    assert not np.any(learner.ask() == 99.0)


def test_default_configs_match_spec() -> None:
    g = GAConfig()
    assert (g.popsize, g.elites, g.tournament, g.crossover, g.mutation_rate, g.mutation_sigma, g.sigma_decay) == (
        64, 2, 3, 0.7, 0.1, 0.1, 1.0
    )
    assert g.bounded_dims == 0
    s = SAConfig()
    assert (s.neighbors, s.step, s.initial_temperature, s.cooling, s.bounded_dims) == (8, 0.1, None, 0.995, 0)
    assert SimulatedAnnealing(np.zeros(3), s).popsize == 9


@pytest.mark.parametrize("method", METHODS)
def test_ask_is_idempotent_until_tell(method: str) -> None:
    a = make(method, np.zeros(4), **SMALL[method])
    b = make(method, np.zeros(4), **SMALL[method])
    X1, X2 = a.ask(), a.ask()
    assert np.array_equal(X1, X2)
    a.tell(X2, sphere(X2))
    Y = b.ask()
    b.tell(Y, sphere(Y))
    assert np.array_equal(a.ask(), b.ask())  # 두 번 물어도 난수를 더 쓰지 않음


@pytest.mark.parametrize("method", METHODS)
def test_misuse_is_rejected(method: str) -> None:
    learner = make(method, np.zeros(4), **SMALL[method])
    n = learner.popsize
    with pytest.raises(RuntimeError):
        learner.tell(np.zeros((n, 4)), np.zeros(n))
    X = learner.ask()
    with pytest.raises(ValueError):
        learner.tell(X[::-1], sphere(X))  # 순서가 바뀐 후보
    Y = X.copy()
    Y[1, 0] += 1e-12
    with pytest.raises(ValueError):
        learner.tell(Y, sphere(X))
    with pytest.raises(ValueError):
        learner.tell(X, np.zeros(n + 1))
    learner.tell(X, sphere(X))  # 거절된 뒤에도 정상적으로 이어진다
    with pytest.raises(RuntimeError):
        learner.tell(X, sphere(X))  # 같은 세대를 두 번 tell
    learner.ask()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"popsize": 1},
        {"popsize": True},
        {"popsize": 2.0},
        {"elites": -1},
        {"elites": 64},
        {"popsize": 4, "elites": 4},
        {"tournament": 0},
        {"popsize": 4, "tournament": 5},
        {"crossover": -0.1},
        {"crossover": 1.5},
        {"crossover": math.nan},
        {"mutation_rate": 2},
        {"mutation_rate": None},
        {"mutation_sigma": 0.0},
        {"mutation_sigma": math.inf},
        {"sigma_decay": 0.0},
        {"sigma_decay": 1.01},
        {"sigma_decay": math.nan},
        {"seed": -1},
        {"seed": 2**32},
        {"bounded_dims": -1},
    ],
)
def test_ga_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        GAConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"neighbors": 0},
        {"neighbors": 2.0},
        {"neighbors": True},
        {"step": 0.0},
        {"step": -0.1},
        {"step": math.nan},
        {"initial_temperature": -1.0},
        {"initial_temperature": math.inf},
        {"initial_temperature": "auto"},
        {"cooling": 0.0},
        {"cooling": 1.5},
        {"seed": 2**32},
        {"bounded_dims": -2},
    ],
)
def test_sa_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SAConfig(**kwargs)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("x0", [np.zeros(0), np.zeros((2, 3)), np.array([0.0, np.nan]), np.array([np.inf])])
def test_x0_validation(method: str, x0: np.ndarray) -> None:
    with pytest.raises(ValueError):
        make(method, x0)


@pytest.mark.parametrize("method", METHODS)
def test_bounded_dims_larger_than_dim(method: str) -> None:
    with pytest.raises(ValueError):
        make(method, np.zeros(3), bounded_dims=4)


# ---------------------------------------------------------------------------- 유전 알고리즘


def test_ga_initial_population_is_x0_plus_noise() -> None:
    x0 = np.linspace(-0.5, 2.0, 50)
    ga = GeneticAlgorithm(x0, GAConfig(popsize=40, mutation_sigma=0.2, seed=8))
    X = ga.ask()
    assert np.array_equal(X[0], x0)
    assert np.array_equal(ga.best(), x0)  # tell 전 추천해는 x0
    d = X[1:] - x0
    assert np.all(d != 0)  # 모든 유전자를 흩뜨린다
    assert np.std(d) == pytest.approx(0.2, rel=0.1)
    assert abs(np.mean(d)) < 0.03
    assert ga.stats()["sigma"] == 0.2


def test_ga_elites_are_asked_again_unchanged_and_rescored() -> None:
    ga = GeneticAlgorithm(np.zeros(6), GAConfig(popsize=12, elites=3, seed=5))
    X = ga.ask()
    s = sphere(X)
    ga.tell(X, s)
    order = np.argsort(-s, kind="stable")
    Y = ga.ask()
    assert np.array_equal(Y[:3], X[order[:3]])  # 지난 세대 상위 3개가 그대로, 순위순으로
    assert np.array_equal(ga.elites, Y[:3])
    assert np.array_equal(ga.best(), X[order[0]])
    assert ga.stats()["best_score"] == s[order[0]]
    # 다시 평가한 점수만 쓴다: 지난 세대 1등이 이번에 가장 나쁜 점수를 받으면 엘리트에서 빠진다.
    t = np.arange(12, dtype=np.float64)
    t[0] = -100.0
    ga.tell(Y, t)
    Z = ga.ask()
    assert np.array_equal(Z[:3], Y[[11, 10, 9]])
    assert not any(np.array_equal(row, Y[0]) for row in Z)  # 교차·돌연변이 없이 복사될 확률은 사실상 0


def test_ga_ties_keep_order() -> None:
    ga = GeneticAlgorithm(np.zeros(4), GAConfig(popsize=8, elites=3, seed=2))
    X = ga.ask()
    s = np.array([1.0, 5.0, 5.0, 2.0, 5.0, 0.0, 5.0, -1.0])
    ga.tell(X, s)
    assert np.array_equal(ga.ask()[:3], X[[1, 2, 4]])  # 동점 5점 넷 중 앞쪽 셋
    assert np.array_equal(ga.best(), X[1])


def test_ga_without_elites_still_reports_best() -> None:
    ga = GeneticAlgorithm(np.zeros(4), GAConfig(popsize=8, elites=0, seed=2))
    X = ga.ask()
    s = sphere(X)
    ga.tell(X, s)
    assert np.array_equal(ga.best(), X[int(np.argmax(s))])
    Y = ga.ask()
    assert Y.shape == (8, 4)


def test_ga_selection_and_crossover_without_mutation_only_reuse_parent_genes() -> None:
    # 돌연변이 0: 교차가 없으면 자식은 부모 사본, 균등 교차면 유전자마다 같은 자리의 부모 유전자.
    for crossover in (0.0, 1.0):
        ga = GeneticAlgorithm(np.zeros(5), GAConfig(popsize=16, elites=1, crossover=crossover, mutation_rate=0.0))
        X = ga.ask()
        ga.tell(X, sphere(X))
        Y = ga.ask()
        if crossover == 0.0:
            assert all(any(np.array_equal(row, p) for p in X) for row in Y)
        else:
            assert all(np.isin(Y[:, j], X[:, j]).all() for j in range(5))
            assert not all(any(np.array_equal(row, p) for p in X) for row in Y[1:])  # 섞인 자식이 있다


def test_ga_mutation_rate_and_sigma_decay() -> None:
    ga = GeneticAlgorithm(np.zeros(200), GAConfig(popsize=30, elites=0, crossover=0.0, mutation_rate=0.1, seed=9))
    X = ga.ask()
    ga.tell(X, np.zeros(30))  # 모두 동점
    Y = ga.ask()
    # 교차 없음: 자식마다 부모(어떤 개체 그대로)와 다른 유전자의 비율 ≈ 0.1
    changed = [min(np.mean(row != p) for p in X) for row in Y]
    assert np.mean(changed) == pytest.approx(0.1, abs=0.02)

    decay = GeneticAlgorithm(np.zeros(3), GAConfig(popsize=6, mutation_sigma=0.5, sigma_decay=0.9))
    for g in range(1, 6):
        X = decay.ask()
        decay.tell(X, sphere(X))
        assert decay.stats()["sigma"] == pytest.approx(0.5 * 0.9**g, rel=1e-12)


def test_ga_nan_scores_are_worst() -> None:
    ga = GeneticAlgorithm(np.zeros(4), GAConfig(popsize=8, elites=2, seed=1))
    X = ga.ask()
    s = sphere(X)
    top = np.argsort(-s, kind="stable")
    s[top[0]] = np.nan
    s[top[1]] = np.inf
    ga.tell(X, s)
    Y = ga.ask()
    assert np.array_equal(Y[:2], X[top[2:4]])
    assert np.array_equal(ga.best(), X[top[2]])
    # 모두 무효여도 멈추지 않는다
    ga.tell(Y, np.full(8, np.nan))
    assert np.array_equal(ga.ask()[:2], Y[:2])


# ---------------------------------------------------------------------------- 담금질


def test_sa_asks_current_first_and_neighbours_around_it() -> None:
    x0 = np.linspace(-1.0, 1.0, 100)
    sa = SimulatedAnnealing(x0, SAConfig(neighbors=8, step=0.3, seed=4))
    X = sa.ask()
    assert X.shape == (9, 100)
    assert np.array_equal(X[0], x0) and np.array_equal(sa.best(), x0)
    d = X[1:] - x0
    assert np.all(d != 0)
    assert np.std(d) == pytest.approx(0.3, rel=0.1)
    for _ in range(20):
        X = sa.ask()
        assert np.array_equal(X[0], sa.best())  # 현재해는 매 단계 다시 평가
        sa.tell(X, sphere(X))
        assert sa.stats()["sigma"] == 0.3


def test_sa_moves_to_best_neighbour_when_it_is_not_worse() -> None:
    sa = SimulatedAnnealing(np.zeros(3), SAConfig(neighbors=4, initial_temperature=0.0, seed=1))
    X = sa.ask()
    sa.tell(X, np.array([0.0, 1.0, 3.0, 3.0, 2.0]))  # 동점 최고 이웃이면 앞쪽
    assert np.array_equal(sa.best(), X[2])
    Y = sa.ask()
    assert np.array_equal(Y[0], X[2])
    # 온도 0: 더 나쁜 이웃은 받지 않는다(지난 점수가 아니라 이번 현재해 점수와 비교)
    sa.tell(Y, np.array([-5.0, -6.0, -7.0, -6.5, -9.0]))
    assert np.array_equal(sa.best(), X[2])
    assert sa.stats()["current_score"] == -5.0
    # 같은 점수는 받는다
    Z = sa.ask()
    sa.tell(Z, np.array([1.0, 0.0, 1.0, 0.5, 1.0]))
    assert np.array_equal(sa.best(), Z[2])
    st = sa.stats()
    assert (st["accepted"], st["acceptance"], st["generation"]) == (2.0, pytest.approx(2 / 3), 3.0)


def test_sa_acceptance_probability_of_worse_neighbour() -> None:
    # 현재해 0, 이웃 모두 −1, 온도 1(감쇠 없음): 수락 확률 exp(−1) ≈ 0.368
    sa = SimulatedAnnealing(np.zeros(2), SAConfig(neighbors=3, initial_temperature=1.0, cooling=1.0, seed=6))
    steps = 4000
    for _ in range(steps):
        X = sa.ask()
        sa.tell(X, np.array([0.0, -1.0, -1.0, -1.0]))
    st = sa.stats()
    assert st["accepted"] / steps == pytest.approx(math.exp(-1.0), abs=0.03)
    assert st["temperature"] == 1.0
    assert 0.0 <= st["acceptance"] <= 1.0


def test_sa_initial_temperature_from_first_neighbours_and_cooling() -> None:
    sa = SimulatedAnnealing(np.zeros(3), SAConfig(neighbors=4, cooling=0.9, seed=2))
    assert sa.temperature is None and math.isnan(sa.stats()["temperature"])
    assert math.isnan(sa.stats()["acceptance"])
    X = sa.ask()
    scores = np.array([100.0, 1.0, 2.0, 4.0, 8.0])  # 현재해 점수는 빼고 이웃 점수의 표준편차
    sa.tell(X, scores)
    t0 = float(np.std(scores[1:]))
    assert sa.temperature == pytest.approx(t0 * 0.9, rel=1e-12)
    for g in range(2, 6):
        X = sa.ask()
        sa.tell(X, sphere(X))
        assert sa.stats()["temperature"] == pytest.approx(t0 * 0.9**g, rel=1e-12)
    fixed = SimulatedAnnealing(np.zeros(3), SAConfig(initial_temperature=2.5))
    assert fixed.temperature == 2.5


def test_sa_acceptance_rate_is_over_recent_window() -> None:
    sa = SimulatedAnnealing(np.zeros(2), SAConfig(neighbors=2, initial_temperature=0.0, seed=3))
    for _ in range(ACCEPTANCE_WINDOW):
        X = sa.ask()
        sa.tell(X, np.array([0.0, 1.0, 0.5]))  # 늘 수락
    assert sa.stats()["acceptance"] == 1.0
    for _ in range(ACCEPTANCE_WINDOW // 2):
        X = sa.ask()
        sa.tell(X, np.array([0.0, -1.0, -2.0]))  # 늘 거절(온도 0)
    assert sa.stats()["acceptance"] == pytest.approx(0.5)
    assert sa.stats()["accepted"] == float(ACCEPTANCE_WINDOW)


def test_sa_nan_scores() -> None:
    sa = SimulatedAnnealing(np.zeros(3), SAConfig(neighbors=4, seed=5))
    X = sa.ask()
    # 이웃이 모두 NaN이면 가장 나쁜 점수(현재해 − 1)로 바뀌어 표준편차 0 → 온도 0 → 거절
    sa.tell(X, np.array([-1.0, np.nan, np.nan, np.nan, np.nan]))
    assert np.array_equal(sa.best(), X[0])
    assert sa.temperature == 0.0
    Y = sa.ask()
    sa.tell(Y, np.array([-1.0, np.nan, -0.5, -np.inf, -2.0]))  # 유한한 이웃 중 최고(−0.5)는 더 좋다
    assert np.array_equal(sa.best(), Y[2])


# ---------------------------------------------------------------------------- 경계


@pytest.mark.parametrize("method", METHODS)
def test_bounded_dims_stay_in_bounds_and_others_are_free(method: str) -> None:
    # 몸 유전자 3개: 최적점이 경계 밖(+3, −3)과 안(0.4). 뇌 3개: 최적점 2.5 (경계 없음이라 1을 넘어가야 함).
    target = np.array([3.0, -3.0, 0.4, 2.5, -2.5, 2.5])

    def f(X: np.ndarray) -> np.ndarray:
        return -np.sum((X - target) ** 2, axis=1)

    if method == "ga":
        learner: AnyLearner = GeneticAlgorithm(np.zeros(6), GAConfig(popsize=24, bounded_dims=3, mutation_sigma=0.3))
        gens = 300
    else:
        learner = SimulatedAnnealing(np.zeros(6), SAConfig(bounded_dims=3, step=0.1))
        gens = 1500
    asked = np.concatenate(run(learner, f, gens))
    assert asked[:, :3].min() >= -1.0 and asked[:, :3].max() <= 1.0
    assert np.any(asked[:, 0] == 1.0) and np.any(asked[:, 1] == -1.0)  # 넘친 값은 잘라낸다
    assert asked[:, 3:].max() > 2.0
    b = learner.best()
    assert np.all(np.abs(b[:3]) <= 1.0)
    assert np.allclose(b[:3], [1.0, -1.0, 0.4], atol=0.1)
    assert np.allclose(b[3:], target[3:], atol=0.2)


@pytest.mark.parametrize("method", METHODS)
def test_x0_body_genes_are_clipped_into_bounds(method: str) -> None:
    learner = make(method, np.array([1.7, -3.0, 0.2, 5.0]), bounded_dims=2)
    assert np.array_equal(learner.best(), [1.0, -1.0, 0.2, 5.0])
    X = learner.ask()
    assert np.array_equal(X[0], [1.0, -1.0, 0.2, 5.0])
    assert X[:, :2].min() >= -1.0 and X[:, :2].max() <= 1.0


# ---------------------------------------------------------------------------- 수렴


def random_search(x0: np.ndarray, evaluations: int, seed: int) -> float:
    """x0 + N(0, 1)에서 evaluations개를 뽑은 최고값의 구면 함수 값(작을수록 좋음)."""
    X = x0 + np.random.default_rng(seed).standard_normal((evaluations, x0.size))
    return float(-sphere(X).max())


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ga_sphere_10d_within_budget(seed: int) -> None:
    # 시드 30개, 200세대(개체 64 = 12,800회)에서 최댓값 4e-5, 중앙값 1.7e-5. 같은 횟수의 무작위 탐색은 약 1.2~2.3.
    x0 = np.full(10, 1.0)
    ga = GeneticAlgorithm(x0, GAConfig(seed=seed))
    run(ga, sphere, 200)
    value = float(-sphere(ga.best()[None])[0])
    assert value < 1e-3
    assert value < random_search(x0, 200 * ga.popsize, seed) / 100


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_sa_sphere_10d_within_budget(seed: int) -> None:
    # 시드 30개, 1,000단계(9 × 1,000 = 9,000회)에서 최댓값 0.043, 중앙값 0.024 (이웃 폭 0.1에서 더 내려가기 어렵다).
    # 같은 횟수의 무작위 탐색은 약 1.4~2.5.
    x0 = np.full(10, 1.0)
    sa = SimulatedAnnealing(x0, SAConfig(seed=seed))
    run(sa, sphere, 1000)
    value = float(-sphere(sa.best()[None])[0])
    assert value < 0.2
    assert value < random_search(x0, 1000 * sa.popsize, seed) / 5


def test_ga_converges_with_nan_region() -> None:
    # x0 < 0 영역은 점수가 NaN(환경 오류 흉내). 최적점 (1, ..., 1)에 다가가야 한다.
    def f(X: np.ndarray) -> np.ndarray:
        out = -np.sum((X - 1.0) ** 2, axis=1)
        out[X[:, 0] < 0] = np.nan
        return out

    ga = GeneticAlgorithm(np.full(5, 0.2), GAConfig(popsize=32, seed=4))
    run(ga, f, 150)
    assert np.allclose(ga.best(), 1.0, atol=0.05)


# ---------------------------------------------------------------------------- 결정성과 이어 가기


@pytest.mark.parametrize("method", METHODS)
def test_same_seed_same_result_and_seed_matters(method: str) -> None:
    kw = {**SMALL[method], "bounded_dims": 2}
    a = run(make(method, np.zeros(6), **{**kw, "seed": 11}), stepped, 30)
    b = run(make(method, np.zeros(6), **{**kw, "seed": 11}), stepped, 30)
    c = run(make(method, np.zeros(6), **{**kw, "seed": 12}), stepped, 30)
    assert same(a, b)
    assert not np.array_equal(a[0], c[0])
    assert not np.array_equal(a[-1], c[-1])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mid_generation", [False, True], ids=["after-tell", "between-ask-and-tell"])
def test_resume_from_state_dict(method: str, mid_generation: bool) -> None:
    kw = {**SMALL[method], "bounded_dims": 2}
    straight = make(method, np.zeros(6), **kw)
    asked_straight = run(straight, stepped, 40)

    learner = make(method, np.zeros(6), **kw)
    run(learner, stepped, 20)
    before = pickle.dumps(learner.state_dict())
    stats_before, best_before = learner.stats(), learner.best()
    if mid_generation:
        X = learner.ask()
        assert np.array_equal(X, asked_straight[20])
        assert pickle.dumps(learner.state_dict()) == before  # ask와 tell 사이 상태 = 그 세대 경계 상태
        assert learner.stats() == stats_before
    state = pickle.loads(pickle.dumps(learner.state_dict()))
    if mid_generation:
        learner.tell(X, stepped(X))  # 저장해도 기다리는 후보는 그대로 tell할 수 있다

    other = {"ga": {"popsize": 4, "elites": 1, "seed": 99}, "sa": {"neighbors": 2, "seed": 99}}[method]
    resumed = make(method, np.zeros(6), **other)  # 저장된 설정이 생성자 설정보다 우선
    resumed.load_state_dict(state)
    assert resumed.generation == 20 and np.array_equal(resumed.best(), best_before)
    assert resumed.stats() == stats_before and resumed.popsize == straight.popsize
    again = run(resumed, stepped, 20)
    assert same(again, asked_straight[20:])
    assert np.array_equal(resumed.best(), straight.best())
    assert resumed.stats() == straight.stats()


@pytest.mark.parametrize("method", METHODS)
def test_state_dict_is_a_snapshot(method: str) -> None:
    learner = make(method, np.zeros(6), **SMALL[method])
    run(learner, stepped, 5)
    state = learner.state_dict()
    blob = pickle.dumps(state)
    straight = run(learner, stepped, 5)  # state를 만든 뒤 계속 진행해도
    resumed = make(method, np.zeros(6), **SMALL[method])
    resumed.load_state_dict(state)  # 그때의 상태 그대로
    assert same(run(resumed, stepped, 5), straight)
    assert pickle.dumps(state) == blob


_CONTINUE_SCRIPT = textwrap.dedent(
    """
    import pickle
    import sys

    import numpy as np

    from evolab.learners.ga import GAConfig, GeneticAlgorithm
    from evolab.learners.sa import SAConfig, SimulatedAnnealing


    def stepped(X):
        return -np.round(np.sum((X - 0.3) ** 2, axis=1), 1)


    with open(sys.argv[1], "rb") as fh:
        job = pickle.load(fh)
    if job["method"] == "ga":
        learner = GeneticAlgorithm(job["x0"], GAConfig(**job["config"]))
    else:
        learner = SimulatedAnnealing(job["x0"], SAConfig(**job["config"]))
    learner.load_state_dict(job["state"])
    asked = []
    for _ in range(job["gens"]):
        X = learner.ask()
        asked.append(X)
        learner.tell(X, stepped(X))
    with open(sys.argv[2], "wb") as fh:
        pickle.dump({"asked": asked, "best": learner.best(), "stats": learner.stats(),
                     "state": learner.state_dict()}, fh)
    """
)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mid_generation", [False, True], ids=["after-tell", "between-ask-and-tell"])
def test_resume_in_new_process_is_bit_identical(method: str, mid_generation: bool, tmp_path: Path) -> None:
    config: GAConfig | SAConfig
    if method == "ga":
        config = GAConfig(popsize=20, elites=3, seed=21, bounded_dims=4, sigma_decay=0.99)
    else:
        config = SAConfig(neighbors=8, seed=22, bounded_dims=4)
    x0 = np.linspace(-0.5, 0.5, 30)
    straight = make(method, x0, **asdict(config))
    asked_straight = run(straight, stepped, 40)

    first = make(method, x0, **asdict(config))
    assert same(run(first, stepped, 20), asked_straight[:20])
    if mid_generation:
        assert np.array_equal(first.ask(), asked_straight[20])  # 이 세대는 평가하지 못하고 저장
    job = {"method": method, "x0": x0, "config": asdict(config), "state": first.state_dict(), "gens": 20}
    script = tmp_path / "continue_learner.py"
    script.write_text(_CONTINUE_SCRIPT, encoding="utf-8")
    job_file, out_file = tmp_path / "job.pkl", tmp_path / "out.pkl"
    job_file.write_bytes(pickle.dumps(job, protocol=5))
    subprocess.run(
        [sys.executable, str(script), str(job_file), str(out_file)],
        check=True,
        env={**os.environ, "PYTHONHASHSEED": "4242"},
        timeout=120,
    )
    out = pickle.loads(out_file.read_bytes())
    for g, (x, y) in enumerate(zip(out["asked"], asked_straight[20:], strict=True)):
        assert np.array_equal(x, y), f"generation {20 + g} differs"
    assert np.array_equal(out["best"], straight.best())
    assert out["stats"] == straight.stats()
    assert pickle.dumps(out["state"]) == pickle.dumps(straight.state_dict())


@pytest.mark.parametrize("method", METHODS)
def test_load_state_dict_validation(method: str) -> None:
    source = make(method, np.zeros(4), **SMALL[method])
    run(source, stepped, 3)
    good = source.state_dict()
    with pytest.raises(ValueError):
        make(method, np.zeros(5), **SMALL[method]).load_state_dict(good)  # 차원이 다름
    other_method = "sa" if method == "ga" else "ga"
    with pytest.raises(ValueError):
        make(other_method, np.zeros(4)).load_state_dict(good)  # 다른 학습 방법의 상태


def _broken_states(method: str, good: dict[str, Any]) -> list[Any]:
    rows = good["population"] if method == "ga" else good["candidates"]
    out: list[Any] = [
        None,
        [],
        method,
        {**good, "format": 999},
        {**good, "name": "cmaes"},
        {**good, "dim": "four"},
        {**good, "config": "nope"},
        {**good, "config": {**good["config"], "bogus": 1}},
        {**good, "rng": {"bit_generator": "MT19937"}},
        {**good, "rng": None},
        {**good, "generation": "x"},
        {**good, "evaluations": -1},
        {**good, "best" if method == "ga" else "current": np.zeros(3)},
        {**good, "population" if method == "ga" else "candidates": rows[:-1]},
        {**good, "population" if method == "ga" else "candidates": np.full_like(rows, np.nan)},
    ]
    out += [{k: v for k, v in good.items() if k != key} for key in good if key not in ("name", "format")]
    if method == "ga":
        out += [{**good, "sigma": -1.0}, {**good, "config": {**good["config"], "popsize": 99}}]
    else:
        out += [
            {**good, "temperature": -1.0},
            {**good, "temperature": math.nan},
            {**good, "accepted": 10**6},
            {**good, "current": good["current"] + 1.0},  # 후보 첫 행(현재해)과 안 맞음
            {**good, "config": {**good["config"], "neighbors": 99}},
        ]
    return out


@pytest.mark.parametrize("method", METHODS)
def test_failed_load_raises_value_error_and_changes_nothing(method: str) -> None:
    source = make(method, np.zeros(4), **{**SMALL[method], "seed": 2})
    run(source, stepped, 3)
    good = source.state_dict()
    for broken in _broken_states(method, good):
        target = make(method, np.zeros(4), **{**SMALL[method], "seed": 3})
        untouched = make(method, np.zeros(4), **{**SMALL[method], "seed": 3})
        run(target, stepped, 2)
        run(untouched, stepped, 2)
        with pytest.raises(ValueError):
            target.load_state_dict(broken)
        assert target.generation == 2 and target.config == untouched.config
        assert np.array_equal(target.best(), untouched.best()) and target.stats() == untouched.stats()
        assert same(run(target, stepped, 3), run(untouched, stepped, 3))


@pytest.mark.parametrize("method", METHODS)
def test_global_numpy_rng_has_no_influence_and_is_not_touched(method: str) -> None:
    def noisy_run(disturb: bool) -> list[np.ndarray]:
        if disturb:
            np.random.seed(12345)
            np.random.rand(1000)
        learner = make(method, np.zeros(8), **{**SMALL[method], "bounded_dims": 2})
        asked = []
        for _ in range(30):
            if disturb:
                np.random.rand(int(np.random.randint(1, 50)))
            X = learner.ask()
            asked.append(X)
            learner.tell(X, stepped(X))
        return asked

    assert same(noisy_run(False), noisy_run(True))

    np.random.seed(2024)
    before = np.random.get_state()
    learner = make(method, np.zeros(6), **SMALL[method])
    run(learner, stepped, 20)
    learner.stats()
    make(method, np.zeros(6)).load_state_dict(pickle.loads(pickle.dumps(learner.state_dict())))
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


# ---------------------------------------------------------------------------- Trainer


def test_make_learner_maps_train_config() -> None:
    layout = layout_for(compile_env(LANDER_TEXT))
    ga = make_learner(
        TrainConfig(method="ga", popsize=12, run_seed=5, ga_elites=3, ga_tournament=4, ga_crossover=0.5,
                    ga_mutation_rate=0.2, ga_mutation_sigma=0.3, ga_sigma_decay=0.98),
        layout,
    )
    assert isinstance(ga, GeneticAlgorithm)
    assert ga.config == GAConfig(popsize=12, elites=3, tournament=4, crossover=0.5, mutation_rate=0.2,
                                 mutation_sigma=0.3, sigma_decay=0.98, seed=seeds.learner_seed(5),
                                 bounded_dims=layout.n_body)
    assert np.array_equal(ga.best(), layout.initial())
    sa = make_learner(
        TrainConfig(method="sa", popsize=99, run_seed=5, sa_neighbors=5, sa_step=0.2, sa_initial_temperature=3.0,
                    sa_cooling=0.99),
        layout,
    )
    assert isinstance(sa, SimulatedAnnealing)
    assert sa.config == SAConfig(neighbors=5, step=0.2, initial_temperature=3.0, cooling=0.99,
                                 seed=seeds.learner_seed(5), bounded_dims=layout.n_body)
    assert sa.popsize == 6  # popsize 설정은 쓰지 않는다
    defaults = make_learner(TrainConfig(method="sa"), layout)
    assert defaults.popsize == 9
    # v1.1: index=0(처음 학습 방법)은 번호를 넣기 전 호출과 같은 시드·같은 상태(v1.0 실행이 그대로 이어진다)
    for config, cls, cfg in ((TrainConfig(method="ga", popsize=12, run_seed=5), GeneticAlgorithm, ga.config),
                             (TrainConfig(method="sa", run_seed=5, sa_neighbors=5, sa_step=0.2,
                                          sa_initial_temperature=3.0, sa_cooling=0.99), SimulatedAnnealing, sa.config)):
        before = cls(layout.initial(), type(cfg)(**{**asdict(make_learner(config, layout).config),
                                                    "seed": seeds.derive(4, 5) & 0xFFFFFFFF}))
        zero = make_learner(config, layout, index=0)
        assert pickle.dumps(zero.state_dict()) == pickle.dumps(before.state_dict())
        third = make_learner(config, layout, index=3)
        assert third.config.seed == seeds.learner_seed(5, 3) != zero.config.seed


def test_train_config_defaults_and_validation() -> None:
    c = TrainConfig()
    assert (c.ga_elites, c.ga_tournament, c.ga_crossover, c.ga_mutation_rate, c.ga_mutation_sigma, c.ga_sigma_decay) == (
        2, 3, 0.7, 0.1, 0.1, 1.0
    )
    assert (c.sa_neighbors, c.sa_step, c.sa_initial_temperature, c.sa_cooling) == (8, 0.1, None, 0.995)
    for method in ("cmaes", "ga", "sa"):
        TrainConfig(method=method).validate()
    TrainConfig(method="sa", sa_initial_temperature=0).validate()
    TrainConfig(method="cmaes", popsize=2, ga_elites=5).validate()  # 다른 방법의 개체 수와는 비교하지 않는다
    assert TrainConfig.from_dict(TrainConfig(method="ga").to_dict()) == TrainConfig(method="ga")
    assert TrainConfig.from_dict({"method": "sa"}).sa_neighbors == 8  # 예전 run.json에는 새 칸이 없다
    bad = [
        {"method": "pso"},
        {"method": "ga", "popsize": 1},
        {"method": "ga", "popsize": 4, "ga_elites": 4},
        {"ga_elites": -1},
        {"ga_elites": 1.0},
        {"ga_tournament": 0},
        {"method": "ga", "popsize": 4, "ga_tournament": 5},
        {"ga_crossover": 1.1},
        {"ga_mutation_rate": -0.1},
        {"ga_mutation_sigma": 0.0},
        {"ga_mutation_sigma": math.nan},
        {"ga_sigma_decay": 0.0},
        {"ga_sigma_decay": 2.0},
        {"sa_neighbors": 0},
        {"sa_neighbors": True},
        {"sa_step": -1.0},
        {"sa_initial_temperature": -0.5},
        {"sa_initial_temperature": math.inf},
        {"sa_cooling": 1.5},
    ]
    for kw in bad:
        with pytest.raises(TrainerChangeError):
            TrainConfig(**kw).validate()


def _config(method: str, **over: Any) -> TrainConfig:
    per = {"ga": {"popsize": 8}, "sa": {"sa_neighbors": 5}}[method]
    return TrainConfig(method=method, trials=2, validate_every=32, checkpoint_every=2, run_seed=7, **{**per, **over})


def _strip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k not in TIME_COLUMNS} for r in rows]


def _run_trainer(method: str, workers: int, gens: int, store: RunStore | None = None) -> tuple[Trainer, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    with WorkerPool(workers) as pool:
        t = Trainer(LANDER_TEXT, _config(method, max_generations=gens), pool, store)
        t.run(rows.append)
    return t, rows


@pytest.mark.parametrize("method", METHODS)
def test_trainer_runs_and_is_independent_of_worker_count(method: str) -> None:
    a, rows_a = _run_trainer(method, 0, 3)
    b, rows_b = _run_trainer(method, 2, 3)
    assert a.stop_reason == "max_generations" and len(rows_a) == 3
    pop = 8 if method == "ga" else 6  # 담금질: 이웃 5 + 현재해
    assert a.learner.popsize == pop
    assert [len(r["scores"].split()) for r in rows_a] == [pop] * 3
    assert [r["evaluations"] for r in rows_a] == [pop * 2 * g for g in (1, 2, 3)]
    assert all(math.isfinite(r["score_best"]) and math.isfinite(r["sigma"]) for r in rows_a)
    assert a.state.validations_done >= 1 and a.state.champion.genome is not None
    assert _strip(rows_a) == _strip(rows_b)
    assert a.learner.best().tobytes() == b.learner.best().tobytes()
    assert a.learner.stats() == b.learner.stats()
    assert a.state.champion.score == b.state.champion.score


@pytest.mark.parametrize("method", METHODS)
def test_trainer_resume_matches_uninterrupted(method: str, tmp_path: Path) -> None:
    def store(name: str) -> RunStore:
        s = RunStore.create(tmp_path, name, {"env_name": "착륙선"})
        s.add_env_version(LANDER_TEXT, 0)
        return s

    straight, _ = _run_trainer(method, 0, 4, store("한 번에"))
    interrupted = store("끊고 이어서")
    _run_trainer(method, 0, 2, interrupted)  # 2세대에서 멈춤(끝날 때 체크포인트)
    with WorkerPool(0) as pool:
        resumed = Trainer.resume(interrupted, pool, None, max_generations=4)
        resumed.run()
    assert isinstance(resumed.learner, GeneticAlgorithm if method == "ga" else SimulatedAnnealing)
    rows_a = _strip([{k: str(v) for k, v in r.items()} for r in straight.store.read_stats()])  # type: ignore[union-attr]
    rows_b = _strip([{k: str(v) for k, v in r.items()} for r in interrupted.read_stats()])
    assert rows_a == rows_b
    assert resumed.learner.best().tobytes() == straight.learner.best().tobytes()
    assert resumed.learner.stats() == straight.learner.stats()

"""학습 방법 규격과 CMA-ES (명세: 학습 방법, 결정성, 테스트 계획 "학습 방법").

- 구면 함수(10차원)와 로젠브록 함수(5차원)가 정해진 시험 횟수 안에 수렴
- 몸 유전자 경계([-1, 1]) 지키기, 추천해(best)가 경계 안
- NaN·무한대 점수 처리, IPOP 재시작
- 상태 저장 → 피클 → 새 프로세스에서 불러와 이어 가기가 끝까지 비트 단위로 같음 (1000차원 대각 포함)
- numpy 전역 난수와 무관 (테스트 코드는 전역 난수를 일부러 쓴다. 규칙은 src/에만 적용)
"""

from __future__ import annotations

import math
import os
import pickle
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from evolab.learners.base import ALL_INVALID_SCORE, Learner, sanitize_scores
from evolab.learners.cmaes import IPOP_MAX_DOUBLINGS, CMAES, CMAESConfig

Objective = Callable[[np.ndarray], np.ndarray]  # (N, dim) -> (N,) 점수, 높을수록 좋음


def sphere(X: np.ndarray) -> np.ndarray:
    return -np.sum(X**2, axis=1)


def rosenbrock(X: np.ndarray) -> np.ndarray:
    return -np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1.0 - X[:, :-1]) ** 2, axis=1)


def ellipsoid(X: np.ndarray) -> np.ndarray:
    scale = 10.0 ** np.linspace(0, 3, X.shape[1])
    return -np.sum((scale * X) ** 2, axis=1)


def run(learner: CMAES, f: Objective, gens: int) -> list[np.ndarray]:
    asked = []
    for _ in range(gens):
        X = learner.ask()
        asked.append(X.copy())
        learner.tell(X, f(X))
    return asked


# ---------------------------------------------------------------------------- 규격


def test_implements_learner_protocol() -> None:
    learner = CMAES(np.zeros(7), CMAESConfig(popsize=12))
    assert isinstance(learner, Learner)
    assert learner.name == "cmaes"
    assert learner.popsize == 12
    X = learner.ask()
    assert X.shape == (12, 7) and X.dtype == np.float64 and X.flags.c_contiguous
    learner.tell(X, sphere(X))
    b = learner.best()
    assert b.shape == (7,) and b.dtype == np.float64
    b[:] = 99.0  # 사본이라 학습 방법에 영향 없음
    assert not np.any(learner.best() == 99.0)


def test_default_config_matches_spec() -> None:
    c = CMAESConfig()
    assert (c.popsize, c.sigma0, c.diagonal_threshold, c.ipop, c.bounded_dims) == (32, 0.5, 1000, False, 0)


def test_starts_at_x0() -> None:
    x0 = np.linspace(-0.5, 2.0, 9)
    learner = CMAES(x0, CMAESConfig(bounded_dims=2))
    assert np.allclose(learner.best(), x0, rtol=0, atol=1e-12)
    X = learner.ask()
    assert np.allclose(X.mean(axis=0), x0, atol=0.5)


# ---------------------------------------------------------------------------- 수렴


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_sphere_10d_converges_within_budget(seed: int) -> None:
    # 시드 20개에서 1e-8까지 97~116세대(개체 32). 예산 200세대 = 6,400회.
    learner = CMAES(np.full(10, 3.0), CMAESConfig(seed=seed))
    run(learner, sphere, 200)
    assert -sphere(learner.best()[None])[0] < 1e-8


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rosenbrock_5d_converges_within_budget(seed: int) -> None:
    # 시드 20개에서 1e-8까지 92~110세대. 예산 200세대 = 6,400회.
    learner = CMAES(np.zeros(5), CMAESConfig(seed=seed))
    run(learner, rosenbrock, 200)
    assert -rosenbrock(learner.best()[None])[0] < 1e-8
    assert np.allclose(learner.best(), 1.0, atol=1e-3)


# ---------------------------------------------------------------------------- 경계


def test_bounded_dims_stay_in_bounds_and_others_are_free() -> None:
    # 몸 유전자 3개: 최적점이 경계 밖(+3, -3)과 안(0.4). 뇌 3개: 최적점 2.5 (경계 없음이라 1을 넘어가야 함).
    target = np.array([3.0, -3.0, 0.4, 2.5, -2.5, 2.5])

    def f(X: np.ndarray) -> np.ndarray:
        return -np.sum((X - target) ** 2, axis=1)

    learner = CMAES(np.zeros(6), CMAESConfig(popsize=16, bounded_dims=3, seed=5))
    asked = np.concatenate(run(learner, f, 150))
    assert asked[:, :3].min() >= -1.0 and asked[:, :3].max() <= 1.0
    assert asked[:, 3:].max() > 2.0  # 경계 없는 차원은 1을 넘는다
    b = learner.best()
    assert np.all(np.abs(b[:3]) <= 1.0)
    assert np.allclose(b[:3], [1.0, -1.0, 0.4], atol=1e-3)
    assert np.allclose(b[3:], target[3:], atol=1e-3)
    # 내부 평균은 경계 밖으로 나가 있고, best()는 그것을 경계 안으로 옮긴 pycma xfavorite와 같다.
    es = learner._es
    assert np.any(np.abs(es.mean[:2]) > 1.0)
    assert np.array_equal(b, es.result.xfavorite)


def test_x0_body_genes_are_clipped_into_bounds() -> None:
    learner = CMAES(np.array([1.7, -3.0, 0.2, 5.0]), CMAESConfig(bounded_dims=2))
    b = learner.best()
    assert np.allclose(b, [1.0, -1.0, 0.2, 5.0], atol=1e-12)


def test_all_dims_bounded() -> None:
    learner = CMAES(np.zeros(4), CMAESConfig(popsize=8, bounded_dims=4))
    asked = np.concatenate(run(learner, lambda X: X.sum(axis=1), 60))
    assert asked.min() >= -1.0 and asked.max() <= 1.0
    assert np.allclose(learner.best(), 1.0, atol=1e-2)


# ---------------------------------------------------------------------------- 점수 처리


def test_sanitize_scores() -> None:
    s = sanitize_scores(np.array([1.0, np.nan, 3.0, -np.inf, np.inf]), 5)
    assert s.tolist() == [1.0, 0.0, 3.0, 0.0, 0.0]
    assert sanitize_scores(np.array([np.nan, np.nan]), 2).tolist() == [ALL_INVALID_SCORE] * 2
    big = sanitize_scores(np.array([-1e300, np.nan]), 2)
    assert big[1] < big[0] and np.isfinite(big[1])
    huge = sanitize_scores(np.array([-1.7e308, np.nan]), 2)
    assert huge[1] <= huge[0] and np.isfinite(huge[1])
    src = np.array([2.0, 5.0])
    assert sanitize_scores(src, 2) is not src
    assert sanitize_scores([1, 2, 3], 3).dtype == np.float64
    with pytest.raises(ValueError):
        sanitize_scores(np.zeros(3), 4)
    with pytest.raises(ValueError):
        sanitize_scores(np.zeros((4, 1)), 4)


def test_non_finite_scores_never_reach_pycma() -> None:
    learner = CMAES(np.zeros(4), CMAESConfig(popsize=8))
    told: list[list[float]] = []
    real_tell = learner._es.tell

    def spy(solutions: Any, values: list[float]) -> None:
        told.append(list(values))
        real_tell(solutions, values)

    learner._es.tell = spy
    X = learner.ask()
    scores = sphere(X)
    scores[[1, 4, 6]] = [np.nan, np.inf, -np.inf]
    learner.tell(X, scores)
    values = np.array(told[0])
    assert np.all(np.isfinite(values))
    worst = values[[1, 4, 6]]
    assert np.all(worst > np.delete(values, [1, 4, 6]).max())  # 최소화 쪽에서 가장 나쁨


def test_converges_with_nan_region() -> None:
    # x0 < 0 영역은 점수가 NaN(환경 오류 흉내). 최적점 (1, ..., 1)에는 도달해야 한다.
    def f(X: np.ndarray) -> np.ndarray:
        out = -np.sum((X - 1.0) ** 2, axis=1)
        out[X[:, 0] < 0] = np.nan
        return out

    learner = CMAES(np.full(5, 0.2), CMAESConfig(popsize=16, seed=4))
    run(learner, f, 200)
    assert np.allclose(learner.best(), 1.0, atol=1e-4)


# ---------------------------------------------------------------------------- 종료 조건과 IPOP


def test_pycma_stop_does_not_end_the_run_without_ipop() -> None:
    learner = CMAES(np.zeros(5), CMAESConfig(popsize=8))
    for _ in range(60):
        X = learner.ask()
        learner.tell(X, np.zeros(8))  # 평탄한 점수 → pycma 'tolflatfitness'
    assert "tolflatfitness" in learner.stop_reasons
    assert learner.popsize == 8 and learner.restarts == 0
    X = learner.ask()
    assert X.shape == (8, 5) and np.all(np.isfinite(X))


def test_ipop_restarts_with_double_popsize_from_current_best() -> None:
    learner = CMAES(np.full(3, 2.0), CMAESConfig(popsize=8, ipop=True, seed=3))
    for _ in range(400):
        X = learner.ask()
        learner.tell(X, sphere(X))
        if learner.restarts:
            break
    assert learner.restarts == 1
    assert learner.stop_reasons  # 재시작의 원인
    assert learner.popsize == 16
    assert learner.ask().shape == (16, 3)
    # 새 출발점은 재시작 직전 추천해(그 세대 tell 뒤의 평균): 이미 거의 0
    assert np.linalg.norm(learner.best()) < 1e-6
    assert learner.stats()["restarts"] == 1.0
    assert learner.stats()["sigma"] == pytest.approx(0.5)


def test_ipop_popsize_cap() -> None:
    learner = CMAES(np.zeros(2), CMAESConfig(popsize=4, ipop=True))
    sizes = []
    for _ in range(80):
        X = learner.ask()
        learner.tell(X, np.zeros(X.shape[0]))  # 평탄 → 2세대마다 재시작
        sizes.append(learner.popsize)
    assert learner.restarts > IPOP_MAX_DOUBLINGS + 2
    assert max(sizes) == 4 * 2**IPOP_MAX_DOUBLINGS
    assert sizes[-1] == 4 * 2**IPOP_MAX_DOUBLINGS


# ---------------------------------------------------------------------------- 대각 공분산


def test_diagonal_switch_threshold() -> None:
    assert not CMAES(np.zeros(1000), CMAESConfig()).diagonal  # "1000 초과"에서 전환
    big = CMAES(np.zeros(1001), CMAESConfig())
    assert big.diagonal and big._es.opts["CMA_diagonal"] is True
    assert type(big._es.sm).__name__ == "GaussStandardConstant"
    small = CMAES(np.zeros(20), CMAESConfig(diagonal_threshold=10))
    assert small.diagonal
    full = CMAES(np.zeros(20), CMAESConfig())
    assert type(full._es.sm).__name__ == "GaussFullSampler"


@pytest.mark.parametrize("threshold", [1000, 5])
def test_axis_ratio_tracks_ill_conditioning(threshold: int) -> None:
    learner = CMAES(np.ones(8), CMAESConfig(popsize=16, diagonal_threshold=threshold))
    assert learner.stats()["axis_ratio"] == pytest.approx(1.0, rel=1e-3)  # pycma가 초기 C 대각을 아주 조금 흩뜨린다
    run(learner, ellipsoid, 250)
    # 축 길이 비 1000의 타원: 학습한 분포도 그만큼 길쭉해야 한다.
    assert 100 < learner.stats()["axis_ratio"] < 1e4


# ---------------------------------------------------------------------------- 통계


def test_stats_keys_are_stable() -> None:
    learner = CMAES(np.zeros(4), CMAESConfig(popsize=6, ipop=True))
    keys = set(learner.stats())
    assert {"sigma", "axis_ratio", "mean_step", "diversity", "popsize", "restarts"} <= keys
    for _ in range(30):
        X = learner.ask()
        assert set(learner.stats()) == keys
        learner.tell(X, np.zeros(X.shape[0]))
        st = learner.stats()
        assert set(st) == keys
        assert all(isinstance(v, float) and math.isfinite(v) for v in st.values())
    assert learner.restarts > 0
    X = learner.ask()
    assert learner.stats()["diversity"] == pytest.approx(float(np.mean(np.std(X, axis=0))))


def test_mean_step_is_distance_moved_by_best() -> None:
    learner = CMAES(np.full(5, 1.0), CMAESConfig(popsize=10))
    X = learner.ask()
    b0 = learner.best()
    learner.tell(X, sphere(X))
    assert learner.stats()["mean_step"] == pytest.approx(float(np.linalg.norm(learner.best() - b0)))


# ---------------------------------------------------------------------------- 사용 오류


def test_ask_is_idempotent_until_tell() -> None:
    a = CMAES(np.zeros(4), CMAESConfig(popsize=6))
    b = CMAES(np.zeros(4), CMAESConfig(popsize=6))
    X1 = a.ask()
    X2 = a.ask()
    assert np.array_equal(X1, X2)
    a.tell(X2, sphere(X2))
    Y = b.ask()
    b.tell(Y, sphere(Y))
    assert np.array_equal(a.ask(), b.ask())  # 두 번 물어도 난수를 더 쓰지 않음


def test_misuse_is_rejected() -> None:
    learner = CMAES(np.zeros(4), CMAESConfig(popsize=6))
    with pytest.raises(RuntimeError):
        learner.tell(np.zeros((6, 4)), np.zeros(6))
    X = learner.ask()
    with pytest.raises(RuntimeError):
        learner.state_dict()  # 세대 도중에는 저장 불가
    with pytest.raises(ValueError):
        learner.tell(X[::-1], sphere(X))  # 순서가 바뀐 후보
    Y = X.copy()
    Y[0, 0] += 1e-12
    with pytest.raises(ValueError):
        learner.tell(Y, sphere(X))
    with pytest.raises(ValueError):
        learner.tell(X, np.zeros(5))
    learner.tell(X, sphere(X))  # 거절된 뒤에도 정상적으로 이어진다
    learner.state_dict()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"popsize": 1},
        {"popsize": 0},
        {"popsize": True},
        {"popsize": 2.5},
        {"sigma0": 0.0},
        {"sigma0": -1.0},
        {"sigma0": math.nan},
        {"sigma0": math.inf},
        {"diagonal_threshold": -1},
        {"seed": -1},
        {"seed": 2**32},
        {"bounded_dims": -1},
        {"ipop": 1},
    ],
)
def test_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CMAESConfig(**kwargs)


@pytest.mark.parametrize("x0", [np.zeros(0), np.zeros((2, 3)), np.array([0.0, np.nan]), np.array([np.inf])])
def test_x0_validation(x0: np.ndarray) -> None:
    with pytest.raises(ValueError):
        CMAES(x0, CMAESConfig())


def test_bounded_dims_larger_than_dim() -> None:
    with pytest.raises(ValueError):
        CMAES(np.zeros(3), CMAESConfig(bounded_dims=4))


def test_load_state_dict_validation() -> None:
    learner = CMAES(np.zeros(4), CMAESConfig(popsize=6))
    run(learner, sphere, 2)
    good = learner.state_dict()
    other = CMAES(np.zeros(5), CMAESConfig(popsize=6))
    with pytest.raises(ValueError):
        other.load_state_dict(good)  # 차원이 다름
    target = CMAES(np.zeros(4), CMAESConfig(popsize=6))
    with pytest.raises(ValueError):
        target.load_state_dict({**good, "name": "ga"})
    with pytest.raises(ValueError):
        target.load_state_dict({**good, "format": 999})
    with pytest.raises(ValueError):
        target.load_state_dict({**good, "es": b"not a pickle"})
    target.load_state_dict(good)
    assert np.array_equal(target.best(), learner.best())


# ---------------------------------------------------------------------------- 결정성


def test_same_seed_same_result_and_seed_matters() -> None:
    a = run(CMAES(np.zeros(6), CMAESConfig(popsize=8, seed=11)), sphere, 15)
    b = run(CMAES(np.zeros(6), CMAESConfig(popsize=8, seed=11)), sphere, 15)
    c = run(CMAES(np.zeros(6), CMAESConfig(popsize=8, seed=12)), sphere, 15)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not np.array_equal(a[0], c[0])


def test_global_numpy_rng_has_no_influence() -> None:
    def noisy_run(disturb: bool) -> tuple[list[np.ndarray], np.ndarray]:
        if disturb:
            np.random.seed(12345)
            np.random.rand(1000)
        learner = CMAES(np.zeros(8), CMAESConfig(popsize=10, bounded_dims=2, ipop=True, seed=9))
        asked = []
        for _ in range(40):
            if disturb:
                np.random.rand(int(np.random.randint(1, 50)))
            X = learner.ask()
            asked.append(X)
            if disturb:
                np.random.randn(7)
            learner.tell(X, np.round(sphere(X), 1))  # 계단 점수 → 평탄 구간과 재시작도 섞인다
        return asked, learner.best()

    a, best_a = noisy_run(False)
    b, best_b = noisy_run(True)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert np.array_equal(best_a, best_b)


def test_does_not_touch_global_numpy_rng() -> None:
    np.random.seed(2024)
    before = np.random.get_state()
    learner = CMAES(np.zeros(6), CMAESConfig(popsize=8, bounded_dims=1, ipop=True))
    for _ in range(30):
        X = learner.ask()
        learner.tell(X, np.zeros(X.shape[0]))
    learner.stats()
    state = pickle.loads(pickle.dumps(learner.state_dict()))
    CMAES(np.zeros(6), CMAESConfig()).load_state_dict(state)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_state_dict_is_a_snapshot() -> None:
    learner = CMAES(np.zeros(6), CMAESConfig(popsize=8, bounded_dims=2))
    run(learner, sphere, 5)
    state = learner.state_dict()
    blob = pickle.dumps(state)
    straight = run(learner, sphere, 5)  # state를 만든 뒤 계속 진행해도
    resumed = CMAES(np.zeros(6), CMAESConfig(popsize=8, bounded_dims=2))
    resumed.load_state_dict(state)  # 그때의 상태 그대로
    again = run(resumed, sphere, 5)
    assert all(np.array_equal(x, y) for x, y in zip(straight, again, strict=True))
    assert pickle.dumps(state) == blob


def test_reading_does_not_change_trajectory() -> None:
    # pycma stop()에는 부작용이 있어서, best()/stats()/state_dict()를 언제 부르든 진행이 같아야 한다.
    def go(read: bool) -> list[np.ndarray]:
        learner = CMAES(np.zeros(6), CMAESConfig(popsize=8, bounded_dims=2, ipop=True, seed=31))
        asked = []
        for _ in range(60):
            if read:
                learner.best()
                learner.stats()
            X = learner.ask()
            asked.append(X)
            if read:
                learner.best()
                learner.stats()
                learner._es.result  # pycma 쪽 읽기도 (안에서 stop()을 부름)
            learner.tell(X, np.round(sphere(X), 2))
            if read:
                learner.state_dict()
                learner.stats()
        return asked

    a, b = go(False), go(True)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_files_and_signals_file_in_cwd_are_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # pycma 기본 설정은 실행 폴더의 cma_signals.in을 매 세대 읽어 설정을 바꾼다. 꺼 두었으므로 결과가 같아야 한다.
    plain = tmp_path / "plain"
    signals = tmp_path / "signals"
    plain.mkdir()
    signals.mkdir()
    (signals / "cma_signals.in").write_text("{'maxstd': 1e-6, 'minstd': 1e-6}", encoding="utf-8")
    results = []
    for folder in (plain, signals):
        monkeypatch.chdir(folder)
        results.append(run(CMAES(np.zeros(5), CMAESConfig(popsize=8)), sphere, 10))
    assert all(np.array_equal(x, y) for x, y in zip(*results, strict=True))
    assert list(plain.iterdir()) == []  # 로그 파일(outcmaes...)을 쓰지 않는다
    assert [p.name for p in signals.iterdir()] == ["cma_signals.in"]


_CONTINUE_SCRIPT = textwrap.dedent(
    """
    import pickle, sys
    import numpy as np
    from evolab.learners.cmaes import CMAES, CMAESConfig

    sys.path.insert(0, sys.argv[3])
    from test_learners import OBJECTIVES

    with open(sys.argv[1], "rb") as fh:
        job = pickle.load(fh)
    learner = CMAES(job["x0"], CMAESConfig(**job["config"]))
    learner.load_state_dict(job["state"])
    f = OBJECTIVES[job["objective"]]
    asked = []
    for _ in range(job["gens"]):
        X = learner.ask()
        asked.append(X)
        learner.tell(X, f(X))
    with open(sys.argv[2], "wb") as fh:
        pickle.dump({"asked": asked, "best": learner.best(), "stats": learner.stats(),
                     "popsize": learner.popsize, "restarts": learner.restarts}, fh)
    """
)


def flat(X: np.ndarray) -> np.ndarray:
    return np.zeros(X.shape[0])


def shifted_ellipsoid(X: np.ndarray) -> np.ndarray:
    return ellipsoid(X - 0.3)


OBJECTIVES: dict[str, Objective] = {"shifted_ellipsoid": shifted_ellipsoid, "flat": flat, "sphere": sphere}


@pytest.mark.parametrize(
    ("dim", "config", "objective"),
    [
        (10, CMAESConfig(popsize=32, seed=21), "shifted_ellipsoid"),
        (12, CMAESConfig(popsize=8, seed=22, bounded_dims=4, ipop=True), "flat"),  # 재시작이 저장 앞뒤로 생김
        (1000, CMAESConfig(popsize=32, seed=23, diagonal_threshold=999, bounded_dims=3), "shifted_ellipsoid"),
    ],
    ids=["full-10d", "ipop-bounded-12d", "diagonal-1000d"],
)
def test_resume_in_new_process_is_bit_identical(dim: int, config: CMAESConfig, objective: str, tmp_path: Path) -> None:
    f = OBJECTIVES[objective]
    x0 = np.linspace(-0.5, 0.5, dim)
    straight = CMAES(x0, config)
    asked_straight = run(straight, f, 40)

    first = CMAES(x0, config)
    asked_first = run(first, f, 20)
    assert all(np.array_equal(x, y) for x, y in zip(asked_first, asked_straight[:20], strict=True))
    job = {"x0": x0, "config": asdict(config), "state": first.state_dict(), "objective": objective, "gens": 20}
    job_file = tmp_path / "job.pkl"
    out_file = tmp_path / "out.pkl"
    job_file.write_bytes(pickle.dumps(job, protocol=5))
    env = {**os.environ, "PYTHONHASHSEED": "4242"}  # bytes hash가 이 프로세스와 다르게
    subprocess.run(
        [sys.executable, "-c", _CONTINUE_SCRIPT, str(job_file), str(out_file), str(Path(__file__).parent)],
        check=True,
        env=env,
        timeout=300,
    )
    out = pickle.loads(out_file.read_bytes())
    assert len(out["asked"]) == 20
    for g, (x, y) in enumerate(zip(out["asked"], asked_straight[20:], strict=True)):
        assert np.array_equal(x, y), f"generation {20 + g} differs"
    assert np.array_equal(out["best"], straight.best())
    assert out["stats"] == straight.stats()
    assert (out["popsize"], out["restarts"]) == (straight.popsize, straight.restarts)
    if config.ipop:
        assert first.restarts > 0 and straight.restarts > first.restarts
    if dim == 1000:
        assert straight.diagonal


# ---------------------------------------------------------------------------- 속도


def test_speed_ask_tell() -> None:
    lines = []
    for dim, threshold in [(211, 1000), (211, 0), (1000, 999)]:
        learner = CMAES(np.zeros(dim), CMAESConfig(diagonal_threshold=threshold))
        run(learner, ellipsoid, 5)
        n = 40
        t0 = time.perf_counter()
        for _ in range(n):
            X = learner.ask()
            learner.tell(X, ellipsoid(X))
        dt = (time.perf_counter() - t0) / n
        t0 = time.perf_counter()
        blob = pickle.dumps(learner.state_dict(), protocol=5)
        t_state = time.perf_counter() - t0
        mode = "diagonal" if learner.diagonal else "full"
        lines.append(f"dim {dim} {mode}: ask+tell {dt * 1e3:.2f} ms, state {len(blob) / 1e6:.2f} MB in {t_state * 1e3:.1f} ms")
        assert dt < 0.25  # 병적인 느려짐만 잡는다
    print("\nCMA-ES popsize 32: " + "; ".join(lines))

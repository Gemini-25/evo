"""v6.0 오라클 바이트 비교, 평가기 부분 (v6 설계 8.1 T1·T2, 3.2, E 묶음).

지금 평가기(engine/evaluate.py, 평가 창)를 v3.1 출시판의 얼린 사본(tests/oracle/evaluate_v31.py)과 같은 프로세스에서 돌려
EvalResult의 모든 칸(점수·스텝·블록·가중 블록·행동 평균·종료 사건·결과·궤적·관찰)을 바이트로 비교한다. 정답 해시를 박지
않는다(명세 「같은 머신에서 두 번 돌려 비교」).

- T1: 내장 환경 6개 × (무작위 유전자 6 × 학습 시험, 픽스처 챔피언 × 검증 시험 6) × (궤적 기록 3쌍마다 1, 관찰 코드 0·1·2
  섞음 + 그 세대 격자 틀). 두 가지로 돈다:
  (a) 지금 평가기 + 지금 런타임 대 오라클(v6.0 전체: 평가 창 + R 묶음의 런타임),
  (b) 지금 평가기 + 오라클 Episode 대 오라클(평가 창만 따로: 런타임 변경과 떼어 보고, step_values가 없는 Episode의
      공개 step 길도 지난다).
- T2: 몸 수치 때문에 조립되지 않는 쌍(꼬인 착륙선, 다각형 날개 글라이더, 이어지지 않는 바퀴 고르는 언덕차)이 섞인 묶음과
  처음 몸 실패(오류)가 오라클과 같다.

오라클의 EvalResult·Trajectory·PairObservation·EpisodeBuildError는 지금 코드와 다른 클래스라 칸마다 tobytes()/repr로만
비교한다(tests/oracle/__init__.py). Episode 단위 비교(스텝마다 상태)는 R 묶음의 tests/test_speed_identity.py에 있다.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tests.oracle.episode_v31 as old_runtime
from evolab import paths
from evolab.engine import evaluate as evaluate_mod
from evolab.engine import observe as observing
from evolab.engine import seeds
from evolab.engine.evaluate import EVAL_WINDOW, evaluate
from evolab.engine.trainer import training_trials
from evolab.policy.genome import GenomeLayout, layout_for
from evolab.spec.compiled import EnvSpec
from evolab.spec.io import compile_env, env_hash
from evolab.world.physics2d.episode import EpisodeBuildError, Trial
from tests.oracle.evaluate_v31 import evaluate as evaluate_v31

ENVS = paths.builtin_env_dir()
NAMES = ("lander", "track_car", "hill_car", "jump", "glider", "wheels")
CHAMPIONS = Path(__file__).resolve().parent / "data" / "v6_champions"


@functools.cache
def env(name: str) -> tuple[str, EnvSpec, GenomeLayout]:
    text = (ENVS / f"{name}.evoenv").read_text(encoding="utf-8")
    spec = compile_env(text)
    return text, spec, layout_for(spec)


def same_eval(a: Any, b: Any) -> None:
    """두 EvalResult(클래스가 달라도 됨)가 바이트까지 같다: 배열은 dtype·모양·tobytes, 나머지는 repr."""
    for name in ("scores", "steps", "blocks", "action_abs", "weighted"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), name
    assert repr(a.end_events) == repr(b.end_events) and repr(a.outcomes) == repr(b.outcomes)
    assert (a.trajectories is None) == (b.trajectories is None)
    for u, v in zip(a.trajectories or [], b.trajectories or [], strict=True):
        assert (u is None) == (v is None)
        if u is not None:
            for name in ("poses", "outputs"):
                x, y = getattr(u, name), getattr(v, name)
                assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), name
            assert repr((u.every, u.dt, u.end_step)) == repr((v.every, v.dt, v.end_step))
    assert (a.observations is None) == (b.observations is None)
    for u, v in zip(a.observations or [], b.observations or [], strict=True):
        assert (u is None) == (v is None)
        if u is not None:
            for name in ("hist", "out_sum", "cells", "counts"):
                x, y = getattr(u, name), getattr(v, name)
                assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), name
            assert (u.steps, u.outside) == (v.steps, v.outside)


def with_oracle_runtime(mp: pytest.MonkeyPatch) -> None:
    """지금 평가기가 오라클 Episode로 돌게(평가 창만 따로 시험). 오라클 Episode에는 step_values가 없다."""
    mp.setattr(evaluate_mod, "Episode", old_runtime.Episode)
    mp.setattr(evaluate_mod, "EpisodeBuildError", old_runtime.EpisodeBuildError)


def t1_batch(name: str) -> tuple[EnvSpec, GenomeLayout, np.ndarray, list[tuple[int, Trial]], dict[str, Any]]:
    _text, spec, layout = env(name)
    rng = np.random.default_rng(61)
    champion = np.load(CHAMPIONS / f"{name}.npz")["genome"]
    genomes = np.vstack([layout.initial()[None, :] + 0.6 * rng.standard_normal((6, layout.dim)), champion[None, :]])
    trials = training_trials(spec, 3, 1, 2)
    pairs = [(g, t) for g in range(6) for t in trials] + [(6, t) for t in seeds.validation_trials(6)]
    frame, _ = observing.world_frame(spec, env_hash(spec.file)[:16], trials[0])
    n = len(pairs)
    kw = {"record": [i % 3 == 0 for i in range(n)], "every": 2, "observe": [i % 3 for i in range(n)],
          "frame": None if frame is None else tuple(frame.as_array())}
    return spec, layout, genomes, pairs, kw


@functools.cache
def oracle_t1(name: str) -> Any:
    spec, layout, genomes, pairs, kw = t1_batch(name)
    return evaluate_v31(spec, layout, genomes, pairs, **kw)


@pytest.mark.parametrize("name", NAMES)
def test_t1_evaluate_matches_the_oracle_evaluator(name: str) -> None:
    """T1(a): 지금 평가기(창 EVAL_WINDOW) + 지금 런타임이 v3.1 평가기와 결과·궤적·관찰 바이트까지 같다."""
    spec, layout, genomes, pairs, kw = t1_batch(name)
    ref = oracle_t1(name)
    same_eval(evaluate(spec, layout, genomes, pairs, **kw), ref)
    assert len(pairs) > EVAL_WINDOW  # 창이 다시 채워지는 묶음
    assert ref.trajectories is not None and ref.observations is not None
    assert int(ref.steps.sum()) > 600 and len(set(ref.steps.tolist())) > 1  # 끝나는 때가 쌍마다 다름


@pytest.mark.parametrize("name", NAMES)
def test_t1_window_alone_matches_the_oracle_evaluator(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """T1(b): 평가 창만(오라클 Episode, 공개 step 길)으로 v3.1 평가기와 바이트까지 같다. 창 5는 묶음 안에서 여러 번
    다시 채운다."""
    spec, layout, genomes, pairs, kw = t1_batch(name)
    ref = oracle_t1(name)
    with monkeypatch.context() as mp:
        with_oracle_runtime(mp)
        assert not hasattr(evaluate_mod.Episode, "step_values")
        now = evaluate(spec, layout, genomes, pairs, window=5, **kw)
    same_eval(now, ref)


# ---------------------------------------------------------------------------
# T2 조립 실패
# ---------------------------------------------------------------------------


def twisted_wheels() -> str:
    """바퀴 고르는 언덕차(v2.1 있음 칸이 있는 몸)에 몸 수치로 조립이 실패하는 모양을 더한 글.

    v2.1 몸 구조는 없는 부품을 빼고 조립하므로 있음 칸만으로는 조립 실패가 나지 않는다(설계 T2의 「이어지지 않는 몸」은
    지금 런타임에 없는 실패라). 그래서 v2.0 T54의 꼬인 착륙선처럼 차체에 진화 수치 twist로 꼬이는 사각형을 더한다
    (twist < −1이면 조립 실패)."""
    text, _spec, _layout = env("wheels")
    d = json.loads(text)
    d["params"].append({"id": "twist", "label": "twist", "value": 0.5, "evolve": True, "min": -3.0, "max": 1.0})
    chassis = next(b for b in d["bodies"] if b["id"] == "chassis")
    chassis["shapes"].append({"kind": "poly", "points": [[-0.3, -0.1], [0.3, -0.1], ["=p.twist * 0.3", 0.1],
                                                         [-0.3, 0.1]]})
    return json.dumps(d, ensure_ascii=False)


def failure_cases() -> list[tuple[str, EnvSpec, GenomeLayout, np.ndarray]]:
    from tests.test_build_failure import gene, poly_wing_glider, twist_lander

    out = []
    spec = compile_env(twist_lander())
    layout = layout_for(spec)
    rng = np.random.default_rng(54)
    X = np.vstack([gene(layout, {"twist": t}) for t in (0.0, -2.0, 0.5, -1.5, 0.8, -2.5)])
    X[:, layout.n_body:] = 0.3 * rng.standard_normal((len(X), layout.dim - layout.n_body))
    out.append(("twist_lander", spec, layout, X))
    spec = compile_env(poly_wing_glider())
    layout = layout_for(spec)
    X = np.vstack([gene(layout, {"flap": f}) for f in (0.02, -0.03, 0.05, -0.01)])
    out.append(("poly_wing_glider", spec, layout, X))
    spec = compile_env(twisted_wheels())
    layout = layout_for(spec)
    X = np.vstack([gene(layout, {"twist": t}) for t in (0.5, -2.0, 0.9, -1.2)])
    X[:, layout.n_body:] = 0.3 * rng.standard_normal((len(X), layout.dim - layout.n_body))
    out.append(("twisted_wheels", spec, layout, X))
    return out


@pytest.mark.parametrize("window", [None, 1, 3])
def test_t2_batches_with_build_failures_match_the_oracle(window: int | None, monkeypatch: pytest.MonkeyPatch) -> None:
    """T2: 조립 실패가 섞인 묶음(실패 쌍 NaN·failure·궤적 0프레임·관찰 0, 나머지는 그대로)이 오라클과 바이트까지 같다.
    창 1·3은 실패 쌍을 건너뛰며 다시 채우는 길, 기본 창은 실패가 창 안에 섞이는 길."""
    trials = [Trial(seeds.derive(9, k)) for k in range(3)]
    for label, spec, layout, X in failure_cases():
        pairs = [(i, t) for i in range(len(X)) for t in trials]
        n = len(pairs)
        frame, _ = observing.world_frame(spec, "t2", trials[0])
        kw: dict[str, Any] = {"record": [i % 2 == 0 for i in range(n)], "observe": [(i // 2) % 3 for i in range(n)],
                              "frame": None if frame is None else tuple(frame.as_array())}
        ref = evaluate_v31(spec, layout, X, pairs, **kw)
        failed = int(np.isnan(ref.scores).sum())
        assert 0 < failed < n, label
        same_eval(evaluate(spec, layout, X, pairs, window=window, **kw), ref)
        with monkeypatch.context() as mp:
            with_oracle_runtime(mp)
            same_eval(evaluate(spec, layout, X, pairs, window=window, **kw), ref)
        plain = evaluate(spec, layout, X, pairs, window=window)  # 기록·관찰 없이
        same_eval(plain, evaluate_v31(spec, layout, X, pairs))


def test_t2_all_pairs_failing_matches_the_oracle() -> None:
    from tests.test_build_failure import gene, twist_lander

    spec = compile_env(twist_lander())
    layout = layout_for(spec)
    X = np.vstack([gene(layout, {"twist": t}) for t in (-2.0, -1.5)])
    pairs = [(i, Trial(seeds.derive(9, k))) for i in range(2) for k in range(3)]
    kw: dict[str, Any] = {"record": [True] * 6, "observe": [1] * 6}
    ref = evaluate_v31(spec, layout, X, pairs, **kw)
    assert np.isnan(ref.scores).all()
    for window in (None, 1, 4):
        same_eval(evaluate(spec, layout, X, pairs, window=window, **kw), ref)


@pytest.mark.parametrize("window", [None, 1, 2])
def test_t2_first_body_failure_raises_the_same_error_as_the_oracle(window: int | None) -> None:
    """T2: 처음 몸으로도 조립되지 않는 시험이면 오류(몸이 아니라 환경의 문제). 실패가 처음 쌍이 아니어도, 앞에 몸 탓
    실패가 있어도 오라클과 같은 쌍의 같은 글을 올린다(창은 쌍 순서대로 만든다)."""
    from tests.test_build_failure import gene, twist_lander

    d = json.loads(twist_lander())
    d["randomize"].append({"id": "k", "label": "k", "dist": "uniform", "low": -1.0, "high": 1.0})
    shape = d["bodies"][0]["shapes"][0]
    shape["points"][2][0] = "=p.twist + 3 * r.k"  # 시험(r.k)에 따라 처음 몸까지 꼬임
    text = json.dumps(d, ensure_ascii=False)
    spec = compile_env(text)
    layout = layout_for(spec)
    trials = [Trial(seeds.derive(9, k)) for k in range(12)]
    first_ok = [_builds(spec, t) for t in trials]
    assert any(first_ok) and not all(first_ok)
    X = np.vstack([gene(layout, {"twist": t}) for t in (0.5, -2.0)])
    pairs = [(i, t) for t in trials for i in range(2)]
    with pytest.raises(old_runtime.EpisodeBuildError) as ref:
        evaluate_v31(spec, layout, X, pairs)
    with pytest.raises(EpisodeBuildError) as now:
        evaluate(spec, layout, X, pairs, window=window)
    assert str(now.value) == str(ref.value)
    assert first_ok.index(False) > 0  # 앞 시험들은 처음 몸으로 조립된다(몸 탓 실패가 먼저 지나감)


def _builds(spec: EnvSpec, trial: Trial) -> bool:
    try:
        old_runtime.Episode(spec, None, trial)
    except old_runtime.EpisodeBuildError:
        return False
    return True


def test_t1_formula_scoring_env_matches_the_oracle() -> None:
    """수식 점수 환경(가중 블록 NaN, 조립 실패 쌍의 빈 가중치)도 같다."""
    from tests.test_build_failure import gene, twist_lander

    d = json.loads(twist_lander(max_time=2.0))
    d["scoring"]["formula"] = "s_dist + s_fuel"
    spec = compile_env(json.dumps(d, ensure_ascii=False))
    layout = layout_for(spec)
    X = np.vstack([gene(layout, {"twist": t}) for t in (0.0, -2.0, 0.6)])
    pairs = [(i, Trial(seeds.derive(9, k))) for i in range(3) for k in range(4)]
    ref = evaluate_v31(spec, layout, X, pairs)
    assert ref.weighted is not None and np.isnan(ref.weighted).all() and math.isnan(float(ref.scores[4]))
    same_eval(evaluate(spec, layout, X, pairs, window=5), ref)

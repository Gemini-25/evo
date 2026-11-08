"""v2.0 개체 조립 실패 (설계 4.9·D22, 테스트 T54~T56).

몸 수치 때문에 조립되지 않는 (개체, 시험) 쌍은 그 쌍만 실패다: 점수 NaN, 결과 failure, 종료 사건 BUILD_FAILED_EVENT,
스텝·블록·행동 0, 궤적 프레임 0개, 관찰 0. 학습은 끝까지 가고, 실패 개체는 통계·세대 대표에서 그 세대 꼴찌(유한한
최솟값 − 1)이며 챔피언이 될 수 없다. 처음 몸부터 조립되지 않으면 학습을 시작하지 않는다.

T55(실패가 없는 실행은 v1 코드와 비트 동일)는 v1 코드가 만든 실제 실행 자료(tests/data의 v1.0 유전 알고리즘 실행,
v1.1 CMA-ES 실행, 둘 다 작은 착륙선)를 지금 코드로 처음부터 다시 돌려 통계·챔피언·세대 대표·체크포인트를 비교한다
(워커 0·2, v1.1 실행은 느린 테스트).
숫자 비교는 자료를 만든 플랫폼에서만 한다(명세 "정답 해시를 박지 않음", 기존 v1.0·v1.1 자료와 같은 규칙).

꼬일 수 있는 모양 두 가지(모두 시험 환경, 저장소 예제는 바꾸지 않음):
- 꼬인 착륙선: 본체를 꼭짓점 하나가 몸 수치 twist로 움직이는 사각형으로. twist < −1이면 꼬인 다각형. 뇌가 있어 신경망
  묶음에서 실패 쌍을 빼는 길을 본다.
- 다각형 날개 글라이더(설계 T54): 내장 글라이더의 주 날개를 꼭짓점 넷 다각형으로, 뒷전 위 꼭짓점 높이 = 몸 수치 flap.
  flap < 0이면 꼬인 다각형. 뇌가 없다.
"""

from __future__ import annotations

import json
import math
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab import cli, paths
from evolab.app import strings as S
from evolab.app.lab.history import decode_observation
from evolab.engine import observe as observing
from evolab.engine import seeds
from evolab.engine import trainer as trainer_mod
from evolab.engine.evaluate import (
    BUILD_FAILED_EVENT,
    OBSERVE_ACTIONS,
    OBSERVE_NONE,
    OBSERVE_POSITIONS,
    EvalResult,
    build_failed_result,
    evaluate,
)
from evolab.engine.pool import WorkerPool
from evolab.engine.replay import agent_geoms, replay
from evolab.engine.runsetup import create_run, platform_info
from evolab.engine.trainer import TrainConfig, Trainer, TrainerChangeError, aggregate
from evolab.learners.base import sanitize_scores
from evolab.policy.genome import GenomeLayout, layout_for
from evolab.runs import observe as obs_fmt
from evolab.runs.bundle import compare_stats, export_bundle, reproduce
from evolab.runs.store import RunStore
from evolab.spec.compiled import EnvSpec
from evolab.spec.io import compile_env, env_hash
from evolab.world.physics2d.episode import Episode, EpisodeBuildError, Trial

LANDER_TEXT = (paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8")
GLIDER_TEXT = (paths.builtin_env_dir() / "glider.evoenv").read_text(encoding="utf-8")
DATA = Path(__file__).resolve().parent / "data"
TIME = set(trainer_mod.TIME_COLUMNS)

TWIST = (-3.0, 1.0)
"""꼬인 착륙선의 twist 범위. 유전자 g → twist = −3 + 2(g + 1)이라 g < 0(twist < −1)이면 조립 실패."""
FLAP = (-0.04, 0.06)
"""다각형 날개의 flap 범위. 유전자 g → flap = −0.04 + 0.05(g + 1)이라 g < −0.2(flap < 0)이면 조립 실패."""


def twist_lander(initial: float = 0.5, *, max_time: float = 1.0, shift: float = 0.0) -> str:
    """꼬인 착륙선 글. shift는 꼭짓점 x에 더하는 상수(처음 몸까지 꼬이게 할 때, 유전자 배치는 그대로)."""
    d = json.loads(LANDER_TEXT)
    d["world"]["max_time"] = max_time
    d["params"].append({"id": "twist", "label": "twist", "value": initial, "evolve": True,
                        "min": TWIST[0], "max": TWIST[1]})
    x = "=p.twist" if shift == 0.0 else f"=p.twist + {shift!r}"
    d["bodies"][0]["shapes"] = [{"kind": "poly", "points": [[-1.0, -0.5], [1.0, -0.5], [x, 0.5], [-1.0, 0.5]]}]
    return json.dumps(d, ensure_ascii=False)


def poly_wing_glider(initial: float = 0.02, *, max_time: float = 2.0) -> str:
    """다각형 날개 글라이더 글(설계 T54): 주 날개 = 앞전 위·뒷전 위(높이 0.06 + flap)·뒷전 아래·앞전 아래."""
    d = json.loads(GLIDER_TEXT)
    d["world"]["max_time"] = max_time
    d["params"].append({"id": "flap", "label": "flap", "value": initial, "evolve": True,
                        "min": FLAP[0], "max": FLAP[1]})
    body = next(b for b in d["bodies"] if b["id"] == "glider")
    k = next(i for i, s in enumerate(body["shapes"]) if s["kind"] == "wing")
    body["shapes"][k] = {"kind": "poly", "points": [["=p.wing_x", 0.10], ["=p.wing_x - p.chord", "=0.06 + p.flap"],
                                                    ["=p.wing_x - p.chord", 0.06], ["=p.wing_x", 0.02]]}
    return json.dumps(d, ensure_ascii=False)


def gene(layout: GenomeLayout, values: dict[str, float]) -> np.ndarray:
    """처음 유전자에서 몸 수치 몇 개를 물리 값으로 바꾼 유전자."""
    x = layout.initial().copy()
    for k, g in enumerate(layout.body):
        if g.id in values:
            x[k] = 2.0 * (values[g.id] - g.low) / (g.high - g.low) - 1.0
    return x


def builds(spec: EnvSpec, layout: GenomeLayout, genome: np.ndarray, trial: Trial) -> bool:
    body = layout.body_values(genome[None, :])[0] if layout.n_body else None
    try:
        Episode(spec, body, trial)
    except EpisodeBuildError:
        return False
    return True


def same_result(a: EvalResult, b: EvalResult) -> None:
    """두 평가 결과가 비트 단위로 같다(NaN은 같은 비트, 궤적·관찰 포함)."""
    for name in ("scores", "steps", "blocks", "action_abs", "weighted"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), name
    assert a.end_events == b.end_events and a.outcomes == b.outcomes
    assert (a.trajectories is None) == (b.trajectories is None)
    for u, v in zip(a.trajectories or [], b.trajectories or [], strict=True):
        assert (u is None) == (v is None)
        if u is not None and v is not None:
            assert u.poses.shape == v.poses.shape and u.poses.tobytes() == v.poses.tobytes()
            assert u.outputs.shape == v.outputs.shape and u.outputs.tobytes() == v.outputs.tobytes()
            assert (u.every, u.dt, u.end_step) == (v.every, v.dt, v.end_step)
    assert (a.observations is None) == (b.observations is None)
    for u, v in zip(a.observations or [], b.observations or [], strict=True):
        assert (u is None) == (v is None)
        if u is not None and v is not None:
            for name in ("hist", "out_sum", "cells", "counts"):
                assert getattr(u, name).tobytes() == getattr(v, name).tobytes(), name
            assert (u.steps, u.outside) == (v.steps, v.outside)


def pick(res: EvalResult, rows: list[int]) -> EvalResult:
    """결과에서 몇 줄만(원래 순서) 뽑은 EvalResult."""
    return EvalResult.concat([res], [(0, r) for r in rows])


# ---------------------------------------------------------------------------
# T54 평가: 실패 쌍
# ---------------------------------------------------------------------------


def lander_batch() -> tuple[str, EnvSpec, GenomeLayout, np.ndarray, list[tuple[int, Trial]], list[int]]:
    """꼬인 착륙선 개체 넷(1·3번은 꼬임) × 시험 둘. 돌려주는 끝 값은 실패 쌍 번호."""
    text = twist_lander()
    spec = compile_env(text)
    layout = layout_for(spec)
    rng = np.random.default_rng(54)
    X = np.vstack([gene(layout, {"twist": t}) for t in (0.0, -2.0, 0.5, -1.5)])
    X[:, layout.n_body:] = 0.3 * rng.standard_normal((4, layout.dim - layout.n_body))
    trials = [Trial(seeds.derive(9, k)) for k in range(2)]
    pairs = [(i, t) for i in range(4) for t in trials]
    return text, spec, layout, X, pairs, [2, 3, 6, 7]


def test_t54_failed_pairs_are_nan_failure_with_zero_steps() -> None:
    _text, spec, layout, X, pairs, failed = lander_batch()
    for i, (g, t) in enumerate(pairs):
        assert builds(spec, layout, X[g], t) == (i not in failed)
    frame, _ = observing.world_frame(spec, "v", pairs[0][1])
    assert frame is not None
    codes = [OBSERVE_POSITIONS if k % 2 == 0 else OBSERVE_ACTIONS for k in range(len(pairs))]
    res = evaluate(spec, layout, X, pairs, record=[True] * len(pairs), observe=codes, frame=tuple(frame))
    n_blocks, n_out = len(spec.file.scoring.blocks), len(spec.outputs)
    assert BUILD_FAILED_EVENT == ""
    for i in range(len(pairs)):
        tr, ob = res.trajectories[i], res.observations[i]  # type: ignore[index]
        assert tr is not None and ob is not None
        if i in failed:
            assert math.isnan(res.scores[i]) and res.outcomes[i] == "failure"
            assert res.end_events[i] == BUILD_FAILED_EVENT
            assert res.steps[i] == 0 and not res.blocks[i].any() and not res.action_abs[i].any()
            assert res.weighted is not None and not res.weighted[i].any()  # 가중치 점수 환경: 기여 0(NaN 아님)
            assert tr.frames == 0 and tr.poses.shape == (0, 0, 3) and tr.outputs.shape == (0, n_out)
            assert tr.end_step == 0 and tr.duration == 0.0 and len(tr.times()) == 0
            assert ob.steps == 0 and not ob.hist.any() and ob.hist.shape == (n_out * 20,) and not ob.out_sum.any()
            assert ob.cells.size == ob.counts.size == ob.outside == 0
        else:
            assert math.isfinite(res.scores[i]) and res.steps[i] > 0 and res.end_events[i] != BUILD_FAILED_EVENT
            assert tr.frames > 1 and ob.steps == res.steps[i]
    assert res.blocks.shape == (len(pairs), n_blocks)
    # 실패 쌍을 빼고 평가해도(신경망 묶음 구성이 달라도) 나머지 쌍은 비트 단위로 같다
    ok = [i for i in range(len(pairs)) if i not in failed]
    alone = evaluate(spec, layout, X, [pairs[i] for i in ok], record=[True] * len(ok),
                     observe=[codes[i] for i in ok], frame=tuple(frame))
    same_result(pick(res, ok), alone)
    # 궤적·관찰을 요청하지 않은 쌍은 None, 하나도 요청하지 않으면 None(v1.2 규칙 그대로)
    some = evaluate(spec, layout, X, pairs, record=[i == 2 for i in range(len(pairs))],
                    observe=[OBSERVE_NONE if i == 2 else c for i, c in enumerate(codes)], frame=tuple(frame))
    assert some.trajectories is not None and [t is not None for t in some.trajectories] == [i == 2 for i in range(8)]
    assert some.observations is not None and some.observations[2] is None and some.observations[3] is not None
    plain = evaluate(spec, layout, X, pairs)
    assert plain.trajectories is None and plain.observations is None
    assert plain.scores.tobytes() == res.scores.tobytes()


def test_t54_all_pairs_failing_and_formula_scores() -> None:
    text, spec, layout, X, pairs, failed = lander_batch()
    res = evaluate(spec, layout, X, [pairs[i] for i in failed], record=[True] * 4)
    assert np.isnan(res.scores).all() and res.outcomes == ["failure"] * 4 and not res.steps.any()
    assert res.trajectories is not None and all(t is not None and t.frames == 0 for t in res.trajectories)
    # 수식 점수 환경: 가중치 블록 값은 Episode.result()처럼 NaN, 결과의 끝 이름은 「조립 실패」
    d = json.loads(text)
    d["scoring"]["formula"] = "s_dist + s_fuel"
    fspec = compile_env(json.dumps(d, ensure_ascii=False))
    fres = evaluate(fspec, layout, X, pairs)
    assert fres.weighted is not None and np.isnan(fres.weighted[failed]).all()
    r = build_failed_result(fspec)
    assert (r.outcome, r.end_event, r.end_label, r.steps, r.time) == ("failure", "", S.RT_BUILD_FAILED_END, 0, 0.0)
    assert math.isnan(r.score) and r.weighted == {} and set(r.action_mean_abs) == set(spec.outputs)


def test_environment_errors_still_stop_the_evaluation() -> None:
    """처음 몸으로도 그 시험을 조립할 수 없으면 몸이 아니라 환경의 문제다: v1.2처럼 오류(워커 풀에서는 PoolError)."""
    from evolab.engine.pool import PoolError

    _text, _spec, layout, X, pairs, _failed = lander_batch()
    broken = twist_lander(shift=-2.0)  # 처음 몸(twist 0.5)도 꼬임. 배치는 같다
    bspec = compile_env(broken)
    with pytest.raises(EpisodeBuildError) as exc:
        evaluate(bspec, layout, X, pairs)
    assert str(exc.value) == S.RT_DEGENERATE_SHAPE.format(body="hull")
    with WorkerPool(1) as pool, pytest.raises(PoolError):
        pool.evaluate("broken", broken, layout, X, pairs)
    # 몸 유전자가 없는 환경의 조립 오류도 그대로 오류(v1.2 test_engine의 망가진 착륙선과 같은 경우)
    d = json.loads(LANDER_TEXT)
    d["randomize"].append({"id": "hw", "dist": "uniform", "low": -3, "high": -1})
    d["bodies"][0]["shapes"][0]["w"] = "=r.hw"
    lspec = compile_env(json.dumps(d, ensure_ascii=False))
    llayout = layout_for(lspec)
    with pytest.raises(EpisodeBuildError):
        evaluate(lspec, llayout, np.zeros((2, llayout.dim)), [(0, pairs[0][1]), (1, pairs[1][1])])
    # 몸 유전자가 없는 환경은 학습기가 처음 몸을 미리 보지 않는다: v1.2처럼 시작하고 첫 세대 평가가 오류를 올린다
    # (워커 0은 그 프로세스 안이라 EpisodeBuildError 그대로)
    with WorkerPool(0) as pool:
        t = Trainer(json.dumps(d, ensure_ascii=False), TrainConfig(popsize=4, trials=2), pool)
        with pytest.raises(EpisodeBuildError):
            t.step()


def test_t54_failures_are_the_same_in_any_worker() -> None:
    text, spec, layout, X, pairs, failed = lander_batch()
    local = evaluate(spec, layout, X, pairs, record=[k % 2 == 0 for k in range(len(pairs))])
    with WorkerPool(0) as p0, WorkerPool(2) as p2:
        a = p0.evaluate("t54", text, layout, X, pairs, record=[k % 2 == 0 for k in range(len(pairs))])
        b = p2.evaluate("t54", text, layout, X, pairs, record=[k % 2 == 0 for k in range(len(pairs))])
    same_result(local, a)
    same_result(a, b)
    assert np.isnan(b.scores[failed]).all()


# ---------------------------------------------------------------------------
# T54 학습: 끝까지 가고 실패 개체는 꼴찌, 챔피언이 되지 않음
# ---------------------------------------------------------------------------


def test_aggregate_spreads_nan_and_keeps_v1_values() -> None:
    rng = np.random.default_rng(7)
    scores = rng.standard_normal((5, 4)) * 100
    k = math.ceil(4 * 0.25)
    ref = {"mean": scores.mean(axis=1), "min": scores.min(axis=1),
           "q25": np.sort(scores, axis=1, kind="stable")[:, :k].mean(axis=1)}  # v1.2의 식
    for how, want in ref.items():
        assert aggregate(scores, how).tobytes() == want.tobytes()
        bad = scores.copy()
        bad[1, 3] = math.nan  # 시험 하나가 조립 실패
        got = aggregate(bad, how)
        assert math.isnan(got[1]) and np.delete(got, 1).tobytes() == np.delete(want, 1).tobytes()
        s = sanitize_scores(got, 5)
        assert s[1] == np.delete(got, 1).min() - 1.0
    clean = ref["mean"]
    s = sanitize_scores(clean, 5)
    assert s is not clean and s.tobytes() == clean.tobytes()  # 실패가 없으면 비트 단위로 같은 사본


def small_config(**over: Any) -> TrainConfig:
    values: dict[str, Any] = {"method": "cmaes", "run_seed": 5, "popsize": 8, "trials": 2, "sigma0": 0.6,
                              "validate_every": 16, "validation_seeds": 3, "validation_candidates": 2,
                              "checkpoint_every": 2, "max_generations": 6}
    values.update(over)
    return TrainConfig(**values)


@pytest.mark.parametrize(("env", "method"), [("glider", "cmaes"), ("glider", "ga"), ("glider", "sa"),
                                             ("lander", "cmaes")])
def test_t54_training_runs_to_the_end_and_failed_genomes_never_win(env: str, method: str, tmp_path: Path) -> None:
    text = poly_wing_glider() if env == "glider" else twist_lander(initial=0.0)  # 처음 유전자 0.2·0.5: 실패가 잦다
    spec = compile_env(text)
    over: dict[str, Any] = {"method": method}
    if method == "ga":
        over.update(ga_mutation_rate=1.0, ga_mutation_sigma=0.5)
    if method == "sa":
        over.update(sa_step=0.5)
    config = small_config(**over)
    store = create_run(tmp_path / "runs", text, spec, config, 0)
    failures = 0
    with WorkerPool(0) as pool:
        t = Trainer(text, config, pool, store)
        while t.check_stop() is None:
            row = t.step()
            g = t.last_generation
            assert g is not None
            pop = g.population.shape[0]
            failed_pairs = t.last_build_failures
            failures += failed_pairs
            trials = trainer_mod.training_trials(t.spec, config.run_seed, g.generation - 1, t.trials)
            bad = [i for i in range(pop) if not all(builds(spec, t.layout, g.population[i], tr) for tr in trials)]
            assert failed_pairs == sum(1 for i in bad for tr in trials
                                       if not builds(spec, t.layout, g.population[i], tr))
            # 통계·세대 대표·화면 점수: 모두 유한, 실패 개체는 유한한 최솟값 − 1, 세대 대표는 실패 개체가 아님
            assert np.isfinite(g.scores).all() and g.representative not in bad
            if bad and len(bad) < pop:
                good = np.delete(g.scores, bad)
                assert (g.scores[bad] == good.min() - 1.0).all()
            for key in ("score_best", "score_mean", "score_q25", "score_median", "score_q75", "score_min"):
                assert math.isfinite(row[key]), key
            assert all(math.isfinite(float(v)) for v in row["scores"].split())
            ends = sum(row[f"end_{e}"] for e in t.end_ids)
            assert ends == pytest.approx(1.0 - failed_pairs / (pop * len(trials)))  # 모자라는 몫 = 조립 실패
            # 관찰 기록: 요약 실패 없이, 점수는 agg_s, 실패 쌍은 결과 2·종료 없음
            assert t.observe_problem is None and g.observation is not None
            assert g.observation[obs_fmt.SCORES].tobytes() == g.scores.tobytes()
            outcome, end = g.observation[obs_fmt.OUTCOME], g.observation[obs_fmt.END]
            for i in bad:
                for k, tr in enumerate(trials):
                    if not builds(spec, t.layout, g.population[i], tr):
                        assert (outcome[i, k], end[i, k]) == (obs_fmt.OUTCOME_CODES["failure"], obs_fmt.END_NONE)
            rep = store.load_champion(g.generation, "representative")
            assert rep["genome"].tobytes() == g.population[g.representative].tobytes()
            assert float(rep["score"]) == float(g.scores[g.representative])
            # 디스크의 관찰 기록이 같은 세대 통계 줄과 맞는다(v1.2 D10의 검사: 최고점, 점수 글, 세대 대표 점수)
            arrays = store.load_observation(g.generation)
            assert arrays is not None
            disk, line = arrays[obs_fmt.SCORES], store.read_stats()[-1]
            assert float(line["score_best"]) == float(disk.max()) and line["scores"] == obs_fmt.scores_text(disk)
            assert float(disk[g.representative]) == float(rep["score"])
            # 화면이 읽는 길 그대로(뇌 없는 환경은 행동 묶음을 빼고 읽음: history.decode_observation)
            dec, reason = decode_observation(arrays, g.generation, t.layout)
            assert reason == "" and dec is not None and obs_fmt.consistent(dec, line, rep["score"])
        t.finish()
    assert failures > 0  # 이 설정에서 실제로 조립 실패가 생긴다(테스트가 헛돌지 않게)
    champ = t.state.champion
    assert champ.genome is not None and math.isfinite(champ.score)
    for tr in seeds.validation_trials(config.validation_seeds):
        assert builds(spec, t.layout, champ.genome, tr)
    assert t.state.generation == config.max_generations and store.checkpoints()[-1] == config.max_generations


def test_t54_run_with_failures_is_deterministic_across_workers_and_resume(tmp_path: Path) -> None:
    text = poly_wing_glider()
    spec = compile_env(text)
    config = small_config(max_generations=4)

    def run(name: str, workers: int, stop: int | None = None) -> RunStore:
        first = config if stop is None else TrainConfig.from_dict({**config.to_dict(), "max_generations": stop})
        store = create_run(tmp_path / name, text, spec, first, workers)
        with WorkerPool(workers) as pool:
            Trainer(text, first, pool, store).run()
            if stop is not None:  # 멈춘 곳(체크포인트)에서 이어 간다
                Trainer.resume(store, pool, None, max_generations=config.max_generations).run()
        return store

    a, b, c = run("a", 0), run("b", 2), run("c", 0, stop=2)
    rows = [[{k: v for k, v in r.items() if k not in TIME} for r in s.read_stats()] for s in (a, b, c)]
    assert rows[0] == rows[1] == rows[2]
    assert any(sum(float(r[f"end_{e}"]) for e in ("landed", "out_of_bounds", "timeout")) < 1 for r in rows[0])
    for s in (b, c):
        assert s.champion_generations("champion") == a.champion_generations("champion")
        for g in a.champion_generations("champion"):
            mine, first = s.load_champion(g, "champion"), a.load_champion(g, "champion")
            assert mine["genome"].tobytes() == first["genome"].tobytes()
    # 재현 묶음도 같은 통계(조립 실패가 있는 세대 포함)
    again = RunStore.open(reproduce(export_bundle(a.root, tmp_path / "a.zip"), tmp_path / "다시", workers=0))
    assert compare_stats(a.read_stats(), again.read_stats()) is None and len(again.read_stats()) == len(rows[0])


class _ScriptedValidation:
    """학습 시험은 진짜 평가, 검증 시험은 후보마다 정한 점수(NaN = 그 후보가 조립 실패)."""

    def __init__(self, pool: WorkerPool, values: list[float]) -> None:
        self.pool, self.values = pool, values
        self.validation = {seeds.derive(seeds.TAG_VALIDATION, k) for k in range(64)}

    def evaluate(self, env_version: str, env_text: str, layout: Any, genomes: np.ndarray,
                 pairs: list[tuple[int, Trial]], record: list[bool] | None = None) -> EvalResult:
        res = self.pool.evaluate(env_version, env_text, layout, genomes, pairs, record=record)
        if pairs[0][1].seed in self.validation:
            res.scores = np.array([self.values[i] for i, _ in pairs], dtype=np.float64)
        return res


def test_t54_nan_validation_candidate_cannot_become_champion() -> None:
    text = twist_lander()
    config = TrainConfig(method="cmaes", run_seed=2, popsize=4, trials=2, validate_every=8, validation_seeds=2,
                         validation_candidates=2, checkpoint_every=0)
    with WorkerPool(0) as pool:
        ev = _ScriptedValidation(pool, [math.nan, 3.0, 1.0])  # 학습 1등 후보가 검증에서 조립 실패
        t = Trainer(text, config, ev)
        row = t.step()
        assert row["validation_best"] == 3.0 and t.state.champion.score == 3.0
        X = t.last_generation.population  # type: ignore[union-attr]
        top = np.argsort(-t.last_generation.scores, kind="stable")[:2]  # type: ignore[union-attr]
        assert t.state.champion.genome is not None and t.state.champion.genome.tobytes() == X[top[1]].tobytes()
        ev.values = [math.nan] * 3  # 모든 후보가 실패: 챔피언·검증 최고점 그대로, 이번 최고점은 −∞
        best_before = t.state.best_validation
        row = t.step()
        assert row["validation_best"] == -math.inf and t.state.champion.score == 3.0
        assert t.state.best_validation == best_before and t.state.current.best == 3.0
        ev.values = [math.nan, math.nan, 7.0]  # 학습 방법 추천해(마지막 후보)만 조립됨
        t.step()
        assert t.state.champion.score == 7.0
        assert t.state.champion.genome is not None
        # 학습 방법 기록에 NaN이 남지 않는다(체크포인트 검사가 NaN을 거절함)
        t.load_state_dict(pickle.loads(pickle.dumps(t.state_dict())))


def trial_width_lander() -> str:
    """선체 폭이 몸 수치와 시험 난수의 합(=p.w + r.k)인 착륙선 글. w가 작으면 시험에 따라 폭이 0 이하가 되어 조립되지 않는다
    (실패가 몸 수치와 시험 둘 다의 함수: 설계 D22). 처음 몸 w = 1.5는 늘 조립된다."""
    d = json.loads(LANDER_TEXT)
    d["world"]["max_time"] = 1.0
    d["params"].append({"id": "w", "label": "w", "value": 1.5, "evolve": True, "min": 0.05, "max": 2.0})
    d["randomize"].append({"id": "k", "dist": "uniform", "low": -1.0, "high": 1.0})
    d["bodies"][0]["shapes"][0]["w"] = "=p.w + r.k"
    return json.dumps(d, ensure_ascii=False)


def test_champion_that_fails_some_compare_courses_gets_the_mean_of_the_built_ones(tmp_path: Path) -> None:
    """검증 시험에서는 모두 조립되어 챔피언이 된 몸이 비교 코스 몇 개에서 조립되지 않아도 비교 점수는 조립된 코스들의
    평균이다(판정 평균과 같은 규칙). 성공률은 실패 코스를 실패로 세어 32로 나눈다."""
    text = trial_width_lander()
    spec = compile_env(text)
    config = TrainConfig(method="cmaes", run_seed=4, popsize=4, trials=2, validate_every=10**6, validation_seeds=3,
                         validation_candidates=1, checkpoint_every=0)
    store = create_run(tmp_path / "runs", text, spec, config, 0)
    with WorkerPool(0) as pool:
        t = Trainer(text, config, pool, store)
        g = gene(t.layout, {"w": 0.85})
        assert all(builds(spec, t.layout, g, tr) for tr in seeds.validation_trials(3))
        courses = seeds.compare_trials()
        res = pool.evaluate(t.env_version, t.env_text, t.layout, g[None, :], [(0, tr) for tr in courses])
        failed = np.isnan(res.scores)
        assert 0 < int(failed.sum()) < len(courses)  # 비교 코스 일부만 조립 실패(테스트가 헛돌지 않게)
        t.learner.best = lambda: g.copy()  # type: ignore[method-assign]
        v = t._validate(g[None, :], np.array([0.0]))
        assert math.isfinite(v) and t.state.champion.genome is not None
        want = float(res.scores[~failed].mean())
        assert t.state.compare_score == want and math.isfinite(want)
        assert t.state.compare_success == sum(1 for o in res.outcomes if o == "success") / len(courses)
        row = t.step()
    assert row["compare_score"] == want
    assert float(store.read_stats()[-1]["compare_score"]) == want
    champ = store.load_champion(0, "champion")
    assert float(champ["compare_score"]) == want


class _InfiniteFirstPair:
    """학습 시험의 첫 쌍 점수만 +무한대로 바꾸는 평가기(NaN 없음)."""

    def __init__(self, pool: WorkerPool) -> None:
        self.pool = pool

    def evaluate(self, env_version: str, env_text: str, layout: Any, genomes: np.ndarray,
                 pairs: list[tuple[int, Trial]], record: list[bool] | None = None) -> EvalResult:
        res = self.pool.evaluate(env_version, env_text, layout, genomes, pairs, record=record)
        res.scores[0] = math.inf
        return res


def test_scores_without_nan_are_used_as_they_are() -> None:
    """NaN(조립 실패)이 없으면 통계·세대 대표·화면 점수는 집계 점수 그대로다(v1.2와 같음, ±무한대도 바꾸지 않는다)."""
    config = TrainConfig(method="cmaes", run_seed=3, popsize=4, trials=2, validate_every=10**6, checkpoint_every=0,
                         aggregate="mean")
    with WorkerPool(0) as pool:
        t = Trainer(LANDER_TEXT, config, _InfiniteFirstPair(pool))  # 조립 실패가 없는 환경
        row = t.step()
    g = t.last_generation
    assert g is not None and g.scores[0] == math.inf and g.representative == 0
    assert row["score_best"] == math.inf and t.last_build_failures == 0
    assert np.isfinite(g.scores[1:]).all()


def test_t54_initial_body_that_fails_refuses_to_start(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    reason = S.RT_DEGENERATE_SHAPE.format(body="hull")
    want = S.TRAIN_INITIAL_BODY_FAILED.format(reason=reason)
    bad = twist_lander(initial=-2.0)
    with WorkerPool(0) as pool, pytest.raises(EpisodeBuildError) as exc:
        Trainer(bad, TrainConfig(popsize=4, trials=2), pool)
    assert str(exc.value) == want
    path = tmp_path / "꼬인 착륙선.evoenv"
    path.write_text(bad, encoding="utf-8")
    code = cli.main(["train", str(path), "--out", str(tmp_path / "runs"), "--max-generations", "1", "--workers", "0",
                     "--popsize", "4", "--trials", "2"])
    assert code == 2 and S.TRAIN_CANNOT_START.format(reason=want) in capsys.readouterr().err
    # 바꿀 환경의 처음 몸이 꼬이면 거절하고 트레이너는 그대로(배치는 같고 꼭짓점만 옮긴 글)
    good = twist_lander()
    with WorkerPool(0) as pool:
        t = Trainer(good, TrainConfig(popsize=4, trials=2), pool)
        version = t.env_version
        with pytest.raises(TrainerChangeError) as change:
            t.replace_env(twist_lander(shift=-2.0))
        assert str(change.value) == want and (t.env_text, t.env_version) == (good, version)
        assert t.replace_env(twist_lander(shift=0.25)) != version  # 처음 몸이 조립되는 변경은 그대로 된다


# ---------------------------------------------------------------------------
# T55 실패가 없는 실행 = v1 코드가 만든 실제 실행(통계·챔피언·세대 대표·체크포인트)
# ---------------------------------------------------------------------------


def same_platform(info: dict[str, Any]) -> bool:
    now = platform_info(0)
    return all(info["platform"].get(k) == now[k] for k in ("python", "numpy", "pymunk", "cma", "cpu"))


def strip(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in r.items() if k not in TIME} for r in rows]


def _exact(v: Any) -> Any:
    """비교용 값: 숫자는 float의 16진 표기(NaN·부호까지 비트로), 배열은 (모양, 바이트), 나머지는 그대로."""
    if isinstance(v, np.ndarray):
        return (v.dtype.str, v.shape, v.tobytes())
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v).hex()
    return v


def checkpoint_view(d: dict[str, Any], layout: GenomeLayout) -> tuple[Any, ...]:
    """체크포인트에서 이후 학습을 정하는 것(형식 1·2 공통): 학습기 상태(경과 시간 뺌)와, 불러온 학습 방법의 상태
    (pycma 객체 뺌: 벽시계 타이머를 품는다)·다음 ask()·best()의 바이트."""
    config = TrainConfig.from_dict(d["config"])
    st = d["state"]
    index = int(st["current"]["index"]) if "current" in st else 0
    learner = trainer_mod.make_learner(config, layout, index=index)
    learner.load_state_dict(d["learner"])
    loaded = {k: v for k, v in learner.state_dict().items() if k != "es"}
    keys = ("generation", "evaluations", "validation_evaluations", "champion_score", "champion_generation",
            "best_validation", "compare_score", "compare_success", "compare_evaluations", "improved_generation")
    validations = st["current"]["validations"] if "current" in st else st["validations_done"]
    genome = st["champion_genome"]
    champion = None if genome is None else _exact(np.asarray(genome))
    return (tuple(_exact(st[k]) for k in keys), int(validations), champion, pickle.dumps(loaded, protocol=5),
            learner.ask().tobytes(), learner.best().tobytes())


def same_as_fixture(old: RunStore, new: RunStore, layout: GenomeLayout, checkpoints: list[int]) -> None:
    assert strip(new.read_stats()) == strip(old.read_stats())
    for kind in ("champion", "representative"):
        gens = old.champion_generations(kind)
        assert new.champion_generations(kind) == gens and gens
        for g in gens:
            a, b = old.load_champion(g, kind), new.load_champion(g, kind)
            assert set(a) <= set(b), (kind, g)  # 새 파일에만 있는 키(learner, 관찰)는 비교하지 않는다
            for k in a:
                assert np.asarray(a[k]).tobytes() == np.asarray(b[k]).tobytes(), (kind, g, k)
    for g in checkpoints:
        assert checkpoint_view(old.load_checkpoint(g)[1], layout) == checkpoint_view(new.load_checkpoint(g)[1], layout)


def fixture_store(tmp_path: Path, name: str) -> RunStore:
    dst = tmp_path / name
    shutil.copytree(DATA / name / "run", dst)
    store = RunStore.open(dst)
    if not same_platform(store.read_run_info()):
        pytest.skip("v1 기록과 다른 플랫폼: 숫자 비교는 같은 플랫폼에서만")
    return store


@pytest.mark.parametrize("workers", [0, 2])
def test_t55_v10_ga_run_recomputed_from_scratch_is_bit_identical(workers: int, tmp_path: Path) -> None:
    old = fixture_store(tmp_path, "v10_run_ga")
    info = old.read_run_info()
    text = old.env_text(info["env_versions"][0]["id"])
    config = TrainConfig.from_dict(info["config"])
    spec = compile_env(text)
    new = create_run(tmp_path / "새로", text, spec, config, workers)
    with WorkerPool(workers) as pool:
        t = Trainer(text, config, pool, new, stats_columns=old.stats_columns(), observe=False)
        t.run()
    assert t.last_build_failures == 0
    assert new.stats_columns() == old.stats_columns()
    assert old.checkpoints() == new.checkpoints() == [2, 4]
    same_as_fixture(old, new, t.layout, [2, 4])


@pytest.mark.slow
@pytest.mark.parametrize("workers", [0, 2])
def test_t55_v11_cmaes_run_recomputed_from_scratch_is_bit_identical(workers: int, tmp_path: Path) -> None:
    """v1.1 실행(재시작 한 번)을 같은 순서(6세대에서 멈춤 → 12세대까지 이어 감)로 처음부터 다시 돌린다.

    느린 테스트: 빠른 묶음에서는 위 v1.0 실행(워커 0·2, 통계·챔피언·세대 대표·체크포인트)과 tests/test_v11_fixture.py
    (워커 0으로 이 실행의 7~12세대 통계와 재현 묶음을 다시 계산)가 같은 것을 본다."""
    old = fixture_store(tmp_path, "v11_run_restart")
    info = old.read_run_info()
    text = old.env_text(info["env_versions"][0]["id"])
    config = TrainConfig.from_dict(info["config"])
    assert config.max_generations == 6
    spec = compile_env(text)
    new = create_run(tmp_path / "새로", text, spec, config, workers)
    with WorkerPool(workers) as pool:
        Trainer(text, config, pool, new, observe=False).run()
        t = Trainer.resume(new, pool, None, observe=False, max_generations=12)
        t.run()
    assert t.state.restarts == 1
    assert new.stats_columns() == old.stats_columns()
    same_as_fixture(old, new, t.layout, [6, 12])


# ---------------------------------------------------------------------------
# T56 화면: 관전·재생·경주가 실패 개체를 빈 모양·안내 글로(예외 없음)
# ---------------------------------------------------------------------------


def test_t56_spectator_canvas_skips_a_failed_individual(qtbot: Any) -> None:
    from evolab.app.lab.canvas import SpectatorCanvas

    _text, spec, layout, X, pairs, _failed = lander_batch()
    trial = pairs[0][1]
    first = [(i, trial) for i in range(4)]
    res = evaluate(spec, layout, X, first, record=[True] * 4)
    assert res.trajectories is not None and [t.frames == 0 for t in res.trajectories] == [False, True, False, True]
    world = None
    agents: list[list[Any]] = []
    for i in range(4):
        try:
            ep = Episode(spec, layout.body_values(X[i][None, :])[0], trial)
        except EpisodeBuildError:
            agents.append([])  # 빈 모양
            continue
        world = world or ep.geometry()
        agents.append(agent_geoms(ep.geometry(), ep.moving_bodies))
    canvas = SpectatorCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(400, 300)
    canvas.set_scene(world, agents)
    scores = sanitize_scores(res.scores, 4)
    ranks = np.argsort(np.argsort(-scores, kind="stable"), kind="stable")
    canvas.set_population(res.trajectories, ranks=ranks, champion=int(np.argmax(scores)))
    for t in (0.0, 0.5, 5.0):
        canvas.set_time(t)
        canvas.grab()
    assert sorted(a.index for a in canvas._population) == [0, 2]


def test_t56_build_scene_gives_a_failed_individual_an_empty_shape() -> None:
    from evolab.app.lab.view import build_scene

    _text, spec, layout, X, pairs, _failed = lander_batch()
    trial = pairs[0][1]
    for order in ([0, 1, 2, 3], [1, 0, 3, 2], [1, 3]):  # 첫 개체가 실패해도, 모두 실패해도
        world, agents = build_scene(spec, layout, X[order], trial)
        assert world is not None and agents is not None and len(agents) == len(order)
        for k, i in enumerate(order):
            if i in (1, 3):
                assert not agents[k]
            else:
                ep = Episode(spec, layout.body_values(X[i][None, :])[0], trial)
                assert [b.id for b in agents[k]] == ep.moving_bodies


def test_t56_replay_and_race_report_a_failed_body_without_crashing(qtbot: Any) -> None:
    from evolab.app.race.model import RaceEntry
    from evolab.app.race.view import RaceView

    text, spec, layout, X, pairs, _failed = lander_batch()
    trial = pairs[0][1]
    with pytest.raises(EpisodeBuildError):  # 재생은 지금 길(EpisodeBuildError)로 알린다
        replay(spec, layout, X[1], trial)
    rep = replay(spec, layout, X[0], trial)
    assert rep.trajectory.frames > 1

    version = env_hash(spec.file)[:16]
    entries = [RaceEntry("ok", "", "#3987e5", text, version, X[0], 1),
               RaceEntry("bad", "", "#d95926", text, version, X[1], 1)]
    view = RaceView()
    qtbot.addWidget(view)
    view.set_race(entries)
    view.set_course(1)
    qtbot.waitUntil(lambda: view.ready or bool(view.failure), timeout=30000)
    # 경주는 조립할 수 없는 칸이 있으면 준비를 그만두고 「이 몸은 만들 수 없습니다」와 까닭을 보인다(묶음 C, race/view.py)
    assert not view.ready
    assert view.failure == S.RACE_FAILED.format(
        reason=S.PLAY_BODY_BUILD_FAILED + " " + S.RT_DEGENERATE_SHAPE.format(body="hull"))
    view.grab()


def test_dashboard_end_tile_draws_the_missing_share_as_build_failed(qtbot: Any, tmp_path: Path) -> None:
    """설계 4.9: 조립 실패 쌍은 어느 종료 사건에도 들지 않아 end_* 합이 1보다 작다. 종료 원인 타일은 모자라는 몫을
    「조립 실패」로 쌓는다(새 통계 열 없음). 실패가 없는 실행은 그 계열이 없다(v1.2 그림 그대로)."""
    from test_dashboard import END_EVENTS, curve_xy, make_dashboard, synthetic_rows

    dash = make_dashboard(qtbot, tmp_path)
    rows = synthetic_rows(10)
    dash.set_rows(rows)
    ends = dash.tile("ends")
    assert [name for _, name in ends._series] == [label for _, label in END_EVENTS]
    assert S.DASH_END_BUILD_FAILED not in ends.legend.text()
    for r in rows[5:]:  # 5세대부터 시험의 10%가 조립 실패
        for eid, _ in END_EVENTS:
            r[f"end_{eid}"] *= 0.9
    dash.set_rows(rows)
    assert ends._series[-1][1] == S.DASH_END_BUILD_FAILED and len(ends._series) == len(END_EVENTS) + 1
    assert S.DASH_END_BUILD_FAILED in ends.legend.text()
    share = ends._values[-1]
    assert np.allclose(share[:5], 0.0) and np.allclose(share[5:], 10.0)
    assert np.allclose(curve_xy(ends._pos[-1])[1], 100.0)  # 쌓은 맨 위는 늘 100%

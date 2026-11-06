"""M2: 시드, 묶음 평가, 워커 풀, Trainer의 결정성과 이어하기."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab import cli, paths
from evolab.engine import seeds
from evolab.engine.pool import PoolError, WorkerPool
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.policy.genome import layout_for
from evolab.runs.store import RunStore
from evolab.spec.io import compile_env, env_hash

LANDER = paths.builtin_env_dir() / "lander.evoenv"
LANDER_TEXT = LANDER.read_text(encoding="utf-8")
TIME_COLUMNS = {"gen_seconds", "elapsed", "steps_per_second"}


def small_config(**over: Any) -> TrainConfig:
    c = TrainConfig(popsize=6, trials=2, validate_every=24, checkpoint_every=2, run_seed=7)
    for k, v in over.items():
        setattr(c, k, v)
    return c


def strip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k not in TIME_COLUMNS} for r in rows]


# ---------------------------------------------------------------------------
# 시드
# ---------------------------------------------------------------------------


def test_seed_spaces_are_disjoint_and_stable() -> None:
    train = {t.seed for g in range(5) for t in seeds.train_trials(1, g, 5)}
    other = {t.seed for t in seeds.validation_trials() + seeds.compare_trials() + seeds.judge_trials()}
    assert len(train) == 25 and not train & other
    assert len({t.seed for t in seeds.judge_trials()}) == seeds.JUDGE_SET
    assert seeds.train_trials(1, 0, 1) != seeds.train_trials(2, 0, 1)
    code = "from evolab.engine import seeds; print(seeds.derive(0, 1, 2, 3), seeds.learner_seed(9))"
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONHASHSEED": str(h), **_env()}).stdout
            for h in (0, 1, 77)}
    assert len(outs) == 1 and outs.pop().split()[0] == str(seeds.derive(0, 1, 2, 3))


def _env() -> dict[str, str]:
    import os

    return {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}


# ---------------------------------------------------------------------------
# 묶음 평가와 워커 풀
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def batch() -> tuple[Any, np.ndarray, list[Any]]:
    spec = compile_env(LANDER_TEXT)
    layout = layout_for(spec)
    rng = np.random.default_rng(3)
    genomes = 0.5 * rng.standard_normal((5, layout.dim))
    pairs = [(i, t) for i in range(5) for t in seeds.train_trials(1, 0, 3)]
    return layout, genomes, pairs


def test_results_do_not_depend_on_worker_count(batch: Any) -> None:
    layout, genomes, pairs = batch
    version = env_hash(compile_env(LANDER_TEXT).file)[:16]
    results = []
    for w in (0, 1, 3):
        with WorkerPool(w) as pool:
            results.append(pool.evaluate(version, LANDER_TEXT, layout, genomes, pairs))
    a = results[0]
    for r in results[1:]:
        assert r.scores.tobytes() == a.scores.tobytes()
        assert r.steps.tolist() == a.steps.tolist() and r.end_events == a.end_events
        assert r.blocks.tobytes() == a.blocks.tobytes()


def test_single_pair_matches_batched(batch: Any) -> None:
    layout, genomes, pairs = batch
    version = "v"
    pool = WorkerPool(0)
    full = pool.evaluate(version, LANDER_TEXT, layout, genomes, pairs)
    for k in (0, 7, len(pairs) - 1):
        one = pool.evaluate(version, LANDER_TEXT, layout, genomes, [pairs[k]])
        assert one.scores[0] == full.scores[k]


def test_dead_worker_is_replaced_and_result_is_the_same(batch: Any) -> None:
    layout, genomes, pairs = batch
    ref = WorkerPool(0).evaluate("v", LANDER_TEXT, layout, genomes, pairs)
    with WorkerPool(2) as pool:
        pool._procs[0].kill()
        pool._procs[0].join()
        res = pool.evaluate("v", LANDER_TEXT, layout, genomes, pairs)
        assert pool._procs[0].is_alive()
    assert res.scores.tobytes() == ref.scores.tobytes()


def test_environment_error_becomes_pool_error(batch: Any) -> None:
    layout, genomes, pairs = batch
    data = json.loads(LANDER_TEXT)
    data["randomize"].append({"id": "hw", "dist": "uniform", "low": -3, "high": -1})
    data["bodies"][0]["shapes"][0]["w"] = "=r.hw"
    bad = json.dumps(data, ensure_ascii=False)
    with WorkerPool(1) as pool, pytest.raises(PoolError):
        pool.evaluate("bad", bad, layout, genomes, pairs[:2])


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def run_trainer(workers: int, gens: int, store: RunStore | None = None, **over: Any) -> Trainer:
    with WorkerPool(workers) as pool:
        t = Trainer(LANDER_TEXT, small_config(max_generations=gens, **over), pool, store)
        rows: list[dict[str, Any]] = []
        t.run(rows.append)
        t.rows = rows  # type: ignore[attr-defined]
    return t


def test_trainer_is_independent_of_worker_count() -> None:
    a = run_trainer(0, 5)
    b = run_trainer(3, 5)
    assert strip(a.rows) == strip(b.rows)  # type: ignore[attr-defined]
    assert a.learner.best().tobytes() == b.learner.best().tobytes()
    assert a.state.champion.score == b.state.champion.score
    assert a.state.validations_done >= 1


def make_store(tmp_path: Path, name: str) -> RunStore:
    store = RunStore.create(tmp_path, name, {"env_name": "착륙선"})
    store.add_env_version(LANDER_TEXT, 0)
    return store


def test_resume_matches_uninterrupted(tmp_path: Path) -> None:
    straight = run_trainer(0, 6, make_store(tmp_path, "한 번에"))
    store = make_store(tmp_path, "끊고 이어서")
    run_trainer(0, 3, store)  # 3세대에서 멈춤(끝날 때 체크포인트)
    with WorkerPool(2) as pool:
        resumed = Trainer.resume(store, pool, None, max_generations=6)
        resumed.run()
    rows_a = strip([{k: str(v) for k, v in r.items()} for r in straight.store.read_stats()])  # type: ignore[union-attr]
    rows_b = strip([{k: str(v) for k, v in r.items()} for r in store.read_stats()])
    assert rows_a == rows_b
    assert resumed.learner.best().tobytes() == straight.learner.best().tobytes()
    assert resumed.state.champion.score == straight.state.champion.score
    assert store.champion_generations("champion") == straight.store.champion_generations("champion")  # type: ignore[union-attr]


def test_stop_conditions() -> None:
    t = run_trainer(0, 100, max_evaluations=40)
    assert t.stop_reason == "max_evaluations" and t.state.evaluations >= 40
    t2 = run_trainer(0, 100, stagnation_generations=3)
    assert t2.stop_reason in ("stagnation", "max_generations")


# ---------------------------------------------------------------------------
# 명령줄
# ---------------------------------------------------------------------------


def test_cli_train_judge_bench(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "실행들"
    code = cli.main(["train", str(LANDER), "--seed", "3", "--popsize", "6", "--trials", "2", "--max-generations", "4",
                     "--workers", "1", "--out", str(out), "--validate-every", "12"])
    assert code == 0
    runs = list(out.iterdir())
    assert len(runs) == 1
    store = RunStore.open(runs[0])
    assert len(store.read_stats()) == 4 and store.checkpoints()
    assert cli.main(["judge", str(runs[0]), "--seeds", "4", "--workers", "1"]) == 0
    assert "판정" in capsys.readouterr().out
    assert cli.main(["train", "--resume", str(runs[0]), "--max-generations", "6", "--workers", "1"]) == 0
    assert len(store.read_stats()) == 6
    assert cli.main(["bench", str(LANDER), "--workers", "1", "--generations", "1", "--popsize", "4"]) == 0

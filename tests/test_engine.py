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
from evolab.engine import trainer as trainer_mod
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.policy.genome import layout_for
from evolab.runs.store import RunStore
from evolab.spec.io import compile_env, env_hash

LANDER = paths.builtin_env_dir() / "lander.evoenv"
LANDER_TEXT = LANDER.read_text(encoding="utf-8")
TIME_COLUMNS = set(trainer_mod.TIME_COLUMNS)


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


def test_resume_after_unclean_stop_does_not_duplicate_rows(tmp_path: Path) -> None:
    straight = run_trainer(0, 7, make_store(tmp_path, "기준"))
    store = make_store(tmp_path, "비정상 종료")
    run_trainer(0, 5, store)  # 체크포인트: 2, 4, 그리고 끝날 때 5
    (store.root / "checkpoints" / "gen_000005.pkl").unlink()  # 5세대 줄은 썼지만 체크포인트 전에 꺼진 것처럼
    with WorkerPool(0) as pool:
        resumed = Trainer.resume(store, pool, None, max_generations=7)
        resumed.run()
    gens = [int(r["generation"]) for r in store.read_stats()]
    assert gens == list(range(1, 8))
    rows_a = strip([{k: str(v) for k, v in r.items()} for r in straight.store.read_stats()])  # type: ignore[union-attr]
    rows_b = strip([{k: str(v) for k, v in r.items()} for r in store.read_stats()])
    assert rows_a == rows_b


DEAD_WORKER_SCRIPT = '''
import sys
import numpy as np
from evolab import paths
from evolab.engine import seeds
from evolab.engine.pool import WorkerPool
from evolab.policy.genome import layout_for
from evolab.spec.io import compile_env

if __name__ == "__main__":
    text = (paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8")
    layout = layout_for(compile_env(text))
    genomes = 0.5 * np.random.default_rng(0).standard_normal((200, layout.dim))
    pairs = [(i, t) for i in range(200) for t in seeds.train_trials(1, 0, 1)]
    with WorkerPool(2, chunks_per_worker=1) as pool:
        pool._procs[0].kill()
        pool._procs[0].join()
        res = pool.evaluate("v", text, layout, genomes, pairs)
    print("evaluated", len(res.scores))
'''


def test_process_exits_after_a_dead_worker_was_sent_a_large_task(tmp_path: Path) -> None:
    """죽은 워커에게 보낸 큰 작업(파이프 버퍼보다 큼)이 프로세스 종료를 막지 않는다.

    예전에는 공유 Queue의 보내기 스레드가 아무도 읽지 않는 파이프에 쓰다 멈춰, 테스트는 다 통과했는데
    pytest 프로세스가 끝나지 않았다(윈도우 CI가 45분 제한에 걸림).
    """
    script = tmp_path / "dead_worker.py"
    script.write_text(DEAD_WORKER_SCRIPT, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "evaluated 200" in proc.stdout


def test_pool_error_does_not_leave_workers_behind(batch: Any) -> None:
    layout, genomes, pairs = batch
    data = json.loads(LANDER_TEXT)
    data["randomize"].append({"id": "hw", "dist": "uniform", "low": -3, "high": -1})
    data["bodies"][0]["shapes"][0]["w"] = "=r.hw"
    bad = json.dumps(data, ensure_ascii=False)
    pool = WorkerPool(3)
    procs = list(pool._procs)
    with pytest.raises(PoolError):
        pool.evaluate("bad", bad, layout, genomes, pairs)
    pool.close()
    assert not any(p.is_alive() for p in procs)


def test_recording_trajectories_does_not_change_results(batch: Any) -> None:
    layout, genomes, pairs = batch
    pool = WorkerPool(0)
    plain = pool.evaluate("v", LANDER_TEXT, layout, genomes, pairs)
    record = [k % 3 == 0 for k in range(len(pairs))]
    with WorkerPool(2) as workers:
        rec = workers.evaluate("v", LANDER_TEXT, layout, genomes, pairs, record=record, every=2)
    assert rec.scores.tobytes() == plain.scores.tobytes()
    assert plain.trajectories is None and rec.trajectories is not None
    spec = compile_env(LANDER_TEXT)
    for k, (g, trial) in enumerate(pairs):
        tr = rec.trajectories[k]
        if not record[k]:
            assert tr is None
            continue
        steps = int(rec.steps[k])
        assert tr.frames == steps // 2 + 1 + (1 if steps % 2 else 0)
        assert tr.end_step == steps and abs(tr.duration - steps * spec.file.world.dt) < 1e-9
        # 마지막 프레임 = 같은 시험을 직접 돌린 끝 상태
        from evolab.world.physics2d.episode import Episode

        ep = Episode(spec, None, trial)
        brain = __import__("evolab.policy.brain", fromlist=["unpack"]).unpack(layout.brain, layout.brain_params(genomes[g:g + 1]))
        while not ep.done:
            ep.step(layout.to_actions(brain.forward(ep.observe()[None, :].astype(np.float64)))[0])
        assert np.allclose(tr.poses[-1], ep.snapshot().poses.astype(np.float32))
        assert tr.poses.shape[1] == len(ep.moving_bodies)
    assert rec.weighted is not None and rec.weighted.shape == rec.blocks.shape
    assert np.allclose(rec.weighted.sum(axis=1), rec.scores)

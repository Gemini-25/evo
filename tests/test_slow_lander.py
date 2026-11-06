"""느린 학습 테스트 (명세 완료 기준): uv run pytest -m slow tests/test_slow_lander.py

착륙선: 고정 실행 시드 3개로 15분씩 학습하고, 15분 시점의 챔피언을 판정 세트(시드 100개)로 한 번 평가한다.
패드 위 착지 성공률의 중앙값이 80% 이상이어야 한다. 이 PC에서 약 45분 걸린다.
"""

import statistics
from pathlib import Path

import pytest

from evolab import paths
from evolab.engine import seeds
from evolab.engine.pool import WorkerPool
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.policy.genome import layout_for
from evolab.runs.store import RunStore
from evolab.spec.io import compile_env, env_hash

LANDER_TEXT = (paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8")


@pytest.mark.slow
def test_lander_reaches_80_percent_in_15_minutes(tmp_path: Path) -> None:
    rates = []
    spec = compile_env(LANDER_TEXT)
    layout = layout_for(spec)
    with WorkerPool() as pool:
        for run_seed in (1, 2, 3):
            store = RunStore.create(tmp_path, f"seed{run_seed}", {"env_name": spec.file.name})
            store.add_env_version(LANDER_TEXT, 0)
            trainer = Trainer(LANDER_TEXT, TrainConfig(run_seed=run_seed, max_time=900.0), pool, store)
            trainer.run()
            champ = trainer.state.champion.genome
            assert champ is not None
            trials = seeds.judge_trials()
            res = pool.evaluate(env_hash(spec.file)[:16], LANDER_TEXT, layout, champ[None, :], [(0, t) for t in trials])
            rate = sum(o == "success" for o in res.outcomes) / len(trials)
            print(f"run seed {run_seed}: generations {trainer.state.generation}, success {rate:.0%}")
            rates.append(rate)
    assert statistics.median(rates) >= 0.8, rates

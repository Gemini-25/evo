"""느린 학습 테스트 (명세 완료 기준): uv run pytest -m slow tests/test_slow_track.py

트랙 자동차: 고정 실행 시드 3개로 30분씩 학습하고, 30분 시점의 챔피언을 판정용 자동 생성 트랙 10개(판정 시드 공간,
학습·선정에 쓰지 않음)로 한 번 평가한다. 완주 수의 중앙값이 7 이상이어야 한다. 이 PC에서 약 90분 걸린다.
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

TRACK_TEXT = (paths.builtin_env_dir() / "track_car.evoenv").read_text(encoding="utf-8")


@pytest.mark.slow
def test_track_car_finishes_7_of_10_new_tracks_in_30_minutes(tmp_path: Path) -> None:
    finished = []
    spec = compile_env(TRACK_TEXT)
    layout = layout_for(spec)
    with WorkerPool() as pool:
        for run_seed in (1, 2, 3):
            store = RunStore.create(tmp_path, f"seed{run_seed}", {"env_name": spec.file.name})
            store.add_env_version(TRACK_TEXT, 0)
            trainer = Trainer(TRACK_TEXT, TrainConfig(run_seed=run_seed, max_time=1800.0), pool, store)
            trainer.run()
            champ = trainer.state.champion.genome
            assert champ is not None
            trials = seeds.judge_trials(10)  # 트랙 번호가 없는 시험 = 판정 시드로 만든 새 트랙
            res = pool.evaluate(env_hash(spec.file)[:16], TRACK_TEXT, layout, champ[None, :], [(0, t) for t in trials])
            n = sum(o == "success" for o in res.outcomes)
            print(f"run seed {run_seed}: generations {trainer.state.generation}, finished {n}/10")
            finished.append(n)
    assert statistics.median(finished) >= 7, finished

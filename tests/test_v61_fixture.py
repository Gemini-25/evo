"""v6.1 자라는 뇌 실제 실행 자료(tests/data/v61_grow_run, v6 설계 7.1, 테스트 T55).

v6.1 프로그램이 만든 작은 착륙선 자라는 실행(CMA-ES · 자라는 뇌, 개체 8, 시험 2, 3세대마다 두 배: 3·6·9세대 끝에 자람, 체크포인트
6·12)과 그 재현 묶음을 지금 프로그램이 그대로 열고, 이어 가고, 분기해 다시 돌리고(세 번째 자람을 지나감), 재현하는지 본다.
숫자 비교(다시 계산한 통계 줄, compare_stats)는 자료를 만든 플랫폼에서만 한다(v1.0·v1.1 자료와 같은 규칙). 자료 폴더는 임시
폴더로 복사해서 쓰고 고치지 않는다. 자세한 내용은 자료의 README.txt.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab.engine import trainer as trainer_mod
from evolab.engine.pool import WorkerPool
from evolab.engine.runsetup import platform_info
from evolab.engine.trainer import GROW_COLUMNS, LEARNER_COLUMNS, Trainer
from evolab.learners.grow import GrowingCMAES
from evolab.policy.frame import BrainFrame
from evolab.policy.genome import layout_for
from evolab.runs.bundle import compare_stats, read_bundle, reproduce
from evolab.runs.catalog import branch_run
from evolab.runs.observe import consistent, decode
from evolab.runs.store import RunStore
from evolab.spec.io import compile_env

FIXTURE = Path(__file__).parent / "data" / "v61_grow_run"
TIME = set(trainer_mod.TIME_COLUMNS)
GENS = 12
GROWN = [(3, 2, 24), (6, 4, 48), (9, 8, 96)]
"""(세대, 그 줄의 brain_neurons = 자라기 전, brain_links): grow = 1인 줄."""


def strip(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in r.items() if k not in TIME} for r in rows]


def copy_run(tmp_path: Path) -> Path:
    dst = tmp_path / "v61" / "run"
    shutil.copytree(FIXTURE / "run", dst)
    return dst


def same_platform(info: dict[str, Any]) -> bool:
    now = platform_info(0)
    return all(info["platform"].get(k) == now[k] for k in ("python", "numpy", "pymunk", "cma", "cpu"))


def skip_unless_same_platform(store: RunStore) -> None:
    if not same_platform(store.read_run_info()):
        pytest.skip("v6.1 기록과 다른 플랫폼: 숫자 비교는 같은 플랫폼에서만")


def grown(rows: list[dict[str, str]]) -> list[tuple[int, int, int]]:
    return [(int(r["generation"]), int(r["brain_neurons"]), int(r["brain_links"])) for r in rows if r["grow"] == "1"]


def check_observations(store: RunStore, rows: list[dict[str, str]], gens: range, layout: Any) -> None:
    for g in gens:
        arrays = store.load_observation(g)
        assert arrays is not None, g
        obs, reason = decode(arrays, dim=layout.dim, n_body=layout.n_body, n_out=layout.brain.n_out, generation=g)
        assert reason == "" and obs is not None, (g, reason)
        assert consistent(obs, rows[g - 1], store.load_champion(g, "representative")["score"]), g
        assert obs.population == 8 and obs.trials == 2


def test_real_v61_run_opens_and_resumes_growing(tmp_path: Path) -> None:
    """열기: 학습 방법 grow, 통계 열(switch 바로 뒤 GROW_COLUMNS), 자란 줄, 체크포인트 6·12의 학습 방법 상태, 틀 길이 유전자(꺼진 자리 0).
    2세대 이어 가면 열 머리와 앞 12줄은 그대로이고 다 자란 크기(16)로 이어진다."""
    store = RunStore.open(copy_run(tmp_path))
    info = store.read_run_info()
    assert info["method"] == "grow" and (info["config"]["grow_start"], info["config"]["grow_every"]) == (2, 3)
    header = store.stats_columns()
    assert header is not None and set(LEARNER_COLUMNS) <= set(header)
    k = header.index("switch")
    assert tuple(header[k + 1:k + 4]) == GROW_COLUMNS
    recorded = store.read_stats()
    assert [int(r["generation"]) for r in recorded] == list(range(1, GENS + 1))
    assert grown(recorded) == GROWN
    assert [int(r["brain_neurons"]) for r in recorded] == [2] * 3 + [4] * 3 + [8] * 3 + [16] * 3
    assert store.checkpoints() == [6, 12]
    for g, (counts, growths) in ((6, ([8], 2)), (12, ([16], 3))):
        _, ckpt = store.load_checkpoint(g)
        assert ckpt["format"] == 2 and ckpt["config"]["method"] == "grow"
        learner = ckpt["learner"]
        assert (learner["name"], learner["format"], learner["counts"], learner["growths"]) == ("grow", 1, counts, growths)
    spec = compile_env(store.env_text(info["env_versions"][0]["id"]))
    layout = layout_for(spec)
    frame = BrainFrame(layout)
    for g in range(1, GENS + 1):
        genome = store.load_champion(g, "representative")["genome"]
        assert genome.shape == (layout.dim,)
        size = int(recorded[g - 1]["brain_neurons"])
        assert frame.counts_of(genome) == (size,)
        off = np.ones(layout.dim, dtype=bool)
        off[frame.active_coords((size,))] = False
        assert np.all(genome[off] == 0.0)
    check_observations(store, recorded, range(1, GENS + 1), layout)
    with WorkerPool(0) as pool:
        t = Trainer.resume(store, pool, None, max_generations=GENS + 2)
        assert isinstance(t.learner, GrowingCMAES) and t.learner.counts == (16,) and t.learner.growths == 3
        t.run()
    assert store.stats_columns() == header
    rows = store.read_stats()
    assert rows[:GENS] == recorded and len(rows) == GENS + 2
    assert [r["brain_neurons"] for r in rows[GENS:]] == ["16", "16"] and [r["grow"] for r in rows[GENS:]] == ["0", "0"]
    check_observations(store, rows, range(GENS + 1, GENS + 3), t.layout)


def test_real_v61_run_branch_recomputes_rows_across_the_third_growth(tmp_path: Path) -> None:
    """6세대 체크포인트(두 번째 자람 직후)에서 분기해 12세대까지 다시 돌리면 v6.1이 쓴 통계 줄·세대 대표·챔피언과 같다."""
    run = copy_run(tmp_path)
    parent = RunStore.open(run)
    skip_unless_same_platform(parent)
    recorded = parent.read_stats()
    child = RunStore.open(branch_run(run, 6, tmp_path / "분기"))
    assert child.checkpoints() == [6] and len(child.read_stats()) == 6
    with WorkerPool(0) as pool:
        Trainer.resume(child, pool, None, max_generations=GENS).run()
    assert strip(child.read_stats()) == strip(recorded)
    for g in range(7, GENS + 1):
        a = child.load_champion(g, "representative")["genome"]
        b = parent.load_champion(g, "representative")["genome"]
        assert a.tobytes() == b.tobytes(), g
    assert child.champion_generations("champion")[-1] == parent.champion_generations("champion")[-1]
    g = parent.champion_generations("champion")[-1]
    assert (child.load_champion(g, "champion")["genome"].tobytes()
            == parent.load_champion(g, "champion")["genome"].tobytes())


def test_real_v61_bundle_reproduces(tmp_path: Path) -> None:
    """자료의 재현 묶음(v6.1 bundle 명령)을 다시 돌리면 compare_stats가 None이고 열 머리가 같다."""
    bundle = tmp_path / "bundle.zip"
    shutil.copyfile(FIXTURE / "bundle.zip", bundle)
    data = read_bundle(bundle)
    original = data["stats"]
    assert data["info"]["config"]["method"] == "grow"
    skip_unless_same_platform(RunStore.open(copy_run(tmp_path)))
    new = RunStore.open(reproduce(bundle, tmp_path / "다시", workers=0))
    rows = new.read_stats()
    assert list(rows[0]) == list(original[0]) and len(rows) == GENS
    assert compare_stats(original, rows) is None
    assert grown(rows) == GROWN

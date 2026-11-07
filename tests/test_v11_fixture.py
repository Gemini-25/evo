"""v1.1 실제 실행 자료(tests/data/v11_run_restart, 설계 D33, 테스트 T12).

v1.1 프로그램(커밋 420c662)이 만든 작은 착륙선 실행(CMA-ES, 개체 8, 시험 2, 자동 재시작 한 번, 체크포인트 형식 2)과
그 재현 묶음을 v1.2가 그대로 열고, 이어 가고(관찰 기록은 이어 간 세대부터), 분기해 다시 돌리고, 재현하는지 본다.
숫자 비교(다시 계산한 통계 줄, compare_stats)는 자료를 만든 플랫폼에서만 한다(윈도우 등은 건너뜀, v1.0 자료와 같은
규칙). 자료 폴더는 임시 폴더로 복사해서 쓰고 고치지 않는다. 자세한 내용은 자료의 README.txt.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from evolab.engine import trainer as trainer_mod
from evolab.engine.pool import WorkerPool
from evolab.engine.runsetup import platform_info
from evolab.engine.trainer import LEARNER_COLUMNS, SWITCH_RESTART_FRESH, Trainer
from evolab.policy.genome import layout_for
from evolab.runs.bundle import compare_stats, read_bundle, reproduce
from evolab.runs.catalog import branch_run
from evolab.runs.observe import consistent, decode
from evolab.runs.store import RunStore
from evolab.spec.io import compile_env

FIXTURE = Path(__file__).parent / "data" / "v11_run_restart"
TIME = set(trainer_mod.TIME_COLUMNS)
BASE_KEYS = ["env_version", "genome", "score"]
GENS = 12
RESTART_ROW = 8
"""이 세대 줄의 switch = 1(처음 상태에서 새로 시작), 다음 세대부터 learner 1."""


def strip(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in r.items() if k not in TIME} for r in rows]


def copy_run(tmp_path: Path) -> Path:
    dst = tmp_path / "v11" / "run"
    shutil.copytree(FIXTURE / "run", dst)
    return dst


def same_platform(info: dict[str, Any]) -> bool:
    now = platform_info(0)
    return all(info["platform"].get(k) == now[k] for k in ("python", "numpy", "pymunk", "cma", "cpu"))


def skip_unless_same_platform(store: RunStore) -> None:
    if not same_platform(store.read_run_info()):
        pytest.skip("v1.1 기록과 다른 플랫폼: 숫자 비교는 같은 플랫폼에서만")


def check_observations(store: RunStore, rows: list[dict[str, str]], gens: range, layout: Any) -> None:
    """이 세대들의 세대 대표에 형식 1 관찰이 있고 같은 세대 통계 줄과 맞는다(D10)."""
    for g in gens:
        arrays = store.load_observation(g)
        assert arrays is not None, g
        obs, reason = decode(arrays, dim=layout.dim, n_body=layout.n_body, n_out=layout.brain.n_out, generation=g)
        assert reason == "" and obs is not None, (g, reason)
        row = rows[g - 1]
        assert consistent(obs, row, store.load_champion(g, "representative")["score"]), g
        assert (obs.learner, int(obs.qualifying)) == (int(row["learner"]), int(row["qualifying"])), g
        assert obs.has_actions and obs.has_density and obs.population == 8 and obs.trials == 2


def test_real_v11_run_opens_and_resumes_with_restart(tmp_path: Path) -> None:
    """열기: v1.1 학습 방법 열, 재시작 기록 줄, 형식 2 체크포인트, 관찰 키 없는 세대 대표. 재시작 켠 채 2세대 이어 가면
    열 머리와 앞 12줄은 그대로이고 관찰 기록은 13·14세대에만 생긴다."""
    store = RunStore.open(copy_run(tmp_path))
    header = store.stats_columns()
    assert header is not None and set(LEARNER_COLUMNS) <= set(header)
    recorded = store.read_stats()
    assert [int(r["generation"]) for r in recorded] == list(range(1, GENS + 1))
    switches = [(int(r["generation"]), int(r["switch"])) for r in recorded if int(r["switch"])]
    assert switches == [(RESTART_ROW, SWITCH_RESTART_FRESH)]
    assert [int(r["learner"]) for r in recorded] == [0] * RESTART_ROW + [1] * (GENS - RESTART_ROW)
    assert [int(r["restarts"]) for r in recorded] == [0] * RESTART_ROW + [1] * (GENS - RESTART_ROW)
    assert store.checkpoints() == [6, 12]
    for g, (restarts, learners) in ((6, (0, 1)), (12, (1, 2))):
        _, ckpt = store.load_checkpoint(g)
        assert ckpt["format"] == 2
        assert (ckpt["state"]["restarts"], ckpt["state"]["learners"]) == (restarts, learners)
    for g in range(1, GENS + 1):
        assert store.load_observation(g) is None  # v1.1 세대: 관찰 기록 없음(화면은 세대 대표로 대체, D12)
        assert sorted(store.load_champion(g, "representative")) == BASE_KEYS
    with WorkerPool(0) as pool:
        t = Trainer.resume(store, pool, None, max_generations=GENS + 2)
        assert (t.config.restart_patience, t.config.restart_from, t.state.restarts) == (48, "fresh", 1)
        assert t.observe
        t.run()
    assert store.stats_columns() == header
    rows = store.read_stats()
    assert rows[:GENS] == recorded and len(rows) == GENS + 2
    assert store.checkpoints()[-1] == GENS + 2 and store.load_checkpoint()[1]["format"] == 2
    for g in range(1, GENS + 1):
        assert store.load_observation(g) is None
    check_observations(store, rows, range(GENS + 1, GENS + 3), t.layout)


def test_real_v11_run_branch_recomputes_rows_across_the_restart(tmp_path: Path) -> None:
    """앞 체크포인트(6세대, 재시작 전)에서 분기해 12세대까지 다시 돌리면 v1.1이 쓴 통계 줄과 같다(재시작을 지나감).
    다시 돈 7~12세대에는 관찰 기록이 생긴다."""
    run = copy_run(tmp_path)
    parent = RunStore.open(run)
    skip_unless_same_platform(parent)
    recorded = parent.read_stats()
    child = RunStore.open(branch_run(run, 6, tmp_path / "분기"))
    assert child.checkpoints() == [6] and len(child.read_stats()) == 6
    with WorkerPool(0) as pool:
        t = Trainer.resume(child, pool, None, max_generations=GENS)
        t.run()
    rows = child.read_stats()
    assert strip(rows) == strip(recorded)
    assert t.state.restarts == 1
    for g in range(1, 7):
        assert child.load_observation(g) is None
    check_observations(child, rows, range(7, GENS + 1), t.layout)


def test_real_v11_bundle_reproduces(tmp_path: Path) -> None:
    """자료의 재현 묶음(v1.1 bundle 명령으로 만듦)을 다시 돌리면 compare_stats가 None이고 열 머리가 같다. 다시 만든
    실행에는 모든 세대에 관찰 기록이 있다."""
    bundle = tmp_path / "bundle.zip"
    shutil.copyfile(FIXTURE / "bundle.zip", bundle)
    original = read_bundle(bundle)["stats"]
    skip_unless_same_platform(RunStore.open(copy_run(tmp_path)))
    new = RunStore.open(reproduce(bundle, tmp_path / "다시", workers=0))
    rows = new.read_stats()
    assert list(rows[0]) == list(original[0]) and len(rows) == GENS
    assert compare_stats(original, rows) is None
    layout = layout_for(compile_env(new.env_text(new.read_run_info()["env_versions"][0]["id"])))
    check_observations(new, rows, range(1, GENS + 1), layout)

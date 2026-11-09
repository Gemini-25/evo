"""v6.0 느린 테스트 S1 (v6 설계 8.3, 3.2-6): 실제 학습이 v3.1 평가기와 같다.

같은 명령줄 학습(`evolab train`, 기본 설정, 실행 시드 1)을 두 번 돌린다.
- v6: 워커 3개 풀(spawn 워커 안에서 지금 evaluate = 평가 창 + v6 런타임 + 워커 시작 GC 얼림).
- 오라클: WorkerPool(0)이 부르는 평가 함수를 v3.1 출시판의 얼린 사본(tests/oracle/evaluate_v31.py, 그 안의 Episode도 v3.1
  사본)으로 바꾼 한 프로세스 학습.
두 실행 폴더의 stats.csv(시간 열 빼고)와 모든 챔피언·세대 대표 npz(키마다 dtype·모양·바이트)가 같아야 한다.

- 점프 게임: 학습 시험 2,400회(15세대)
- 트랙 자동차: 10세대(학습 트랙 세트, 검증 포함)
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab import cli, paths
from evolab.engine import pool as pool_mod
from evolab.engine import trainer as trainer_mod
from evolab.runs.store import CHAMPIONS_DIR
from tests.oracle.evaluate_v31 import evaluate as evaluate_v31

TIME = set(trainer_mod.TIME_COLUMNS)
CASES = {
    "jump": ["--max-evaluations", "2400"],
    "track_car": ["--max-generations", "10"],
}


def oracle_evaluate(*args: Any, window: int | None = None, **kwargs: Any) -> Any:
    """WorkerPool(0)이 부르는 evaluate 자리에 v3.1 평가기(창 인수가 없음)."""
    return evaluate_v31(*args, **kwargs)


def train(env: str, workers: int, out: Path) -> tuple[Path, float]:
    t0 = time.perf_counter()
    code = cli.main(["train", str(paths.builtin_env_dir() / f"{env}.evoenv"), "--seed", "1", "--workers", str(workers),
                     *CASES[env], "--out", str(out)])
    assert code == 0
    runs = list(out.iterdir())
    assert len(runs) == 1
    return runs[0], time.perf_counter() - t0


def stats(run: Path) -> tuple[list[str], list[dict[str, str]]]:
    with (run / "stats.csv").open(encoding="utf-8-sig", newline="") as f:  # 머리에 BOM
        reader = csv.DictReader(f)
        rows = [{k: v for k, v in r.items() if k not in TIME} for r in reader]
        return list(reader.fieldnames or []), rows


def npz(path: Path) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    with np.load(path, allow_pickle=False) as z:
        return {k: (str(z[k].dtype), tuple(z[k].shape), z[k].tobytes()) for k in z.files}


@pytest.mark.slow
@pytest.mark.parametrize("env", list(CASES))
def test_s1_training_with_the_v6_pool_matches_the_v31_evaluator(env: str, tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))  # 사용자 폴더(기록·설정)를 건드리지 않게
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    v6_run, v6_s = train(env, 3, tmp_path / "v6")
    monkeypatch.setattr(pool_mod, "evaluate", oracle_evaluate)
    old_run, old_s = train(env, 0, tmp_path / "oracle")

    head_v6, rows_v6 = stats(v6_run)
    head_old, rows_old = stats(old_run)
    assert head_v6 == head_old
    assert rows_v6 == rows_old and rows_v6
    evals = int(rows_v6[-1]["evaluations"])
    if env == "jump":
        assert evals == 2400 and len(rows_v6) == 15
    else:
        assert int(rows_v6[-1]["generation"]) == 10

    files = sorted(p.name for p in (v6_run / CHAMPIONS_DIR).iterdir())
    assert files == sorted(p.name for p in (old_run / CHAMPIONS_DIR).iterdir()) and files
    for name in files:
        assert npz(v6_run / CHAMPIONS_DIR / name) == npz(old_run / CHAMPIONS_DIR / name), name
    print(f"S1 {env}: generations={len(rows_v6)} evaluations={evals} champion files={len(files)} "
          f"v6 3 workers {v6_s:.1f}s, oracle 1 process {old_s:.1f}s, champion={rows_v6[-1].get('champion_score')}")

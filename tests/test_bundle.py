"""M6: 재현 묶음(같은 PC에서 다시 돌리면 통계가 같음)과 통계 CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from evolab import cli, paths
from evolab.engine.pool import WorkerPool
from evolab.engine.runsetup import create_run
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.runs.bundle import compare_stats, export_bundle, export_csv, read_bundle, reproduce
from evolab.runs.store import RunStore
from evolab.spec.io import compile_env

LANDER_TEXT = (paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8")


def make_run_with_changes(tmp: Path) -> Path:
    """3세대 → 정지 조건·시험 횟수 변경 + 중력 바꾼 환경 → 6세대. 학습 화면에서 멈추고 고친 것과 같은 기록."""
    config = TrainConfig(popsize=6, trials=2, validate_every=24, checkpoint_every=2, run_seed=9)
    store = create_run(tmp, LANDER_TEXT, compile_env(LANDER_TEXT), config, 1)
    with WorkerPool(1) as pool:
        t = Trainer(LANDER_TEXT, config, pool, store)
        for _ in range(3):
            t.step()
        changes = t.apply_config({"trials": 3, "max_generations": 6})
        store.append_event("config_change", 3, changes=changes)
        data = json.loads(LANDER_TEXT)
        data["world"]["gravity"] = [0.0, -1.4]
        text = json.dumps(data, ensure_ascii=False, indent=2)
        t.replace_env(text)
        vid = store.add_env_version(text, 3)
        store.append_event("env_change", 3, version=vid)
        while t.state.generation < 6:
            t.step()
        t.checkpoint()
    return store.root


def test_bundle_reproduces_identical_stats(tmp_path: Path) -> None:
    run = make_run_with_changes(tmp_path / "원래")
    bundle = export_bundle(run, tmp_path / "묶음" / "재현.zip")
    data = read_bundle(bundle)
    assert len(data["envs"]) == 2 and len(data["stats"]) == 6
    new = reproduce(bundle, tmp_path / "다시", workers=2)
    a = RunStore.open(run).read_stats()
    b = RunStore.open(new).read_stats()
    assert compare_stats(a, b) is None
    assert compare_stats(data["stats"], b) is None
    info = RunStore.open(new).read_run_info()
    assert info["reproduced_from"]["run"] == run.name
    assert [v["from_generation"] for v in info["env_versions"]] == [0, 3]
    # 변경을 빼면 다르다(비교가 실제로 무언가를 본다)
    b[4] = {**b[4], "score_best": "0"}
    assert compare_stats(a, b) == 5


def test_cli_bundle_reproduce_csv(tmp_path: Path, capsys: Any) -> None:
    out = tmp_path / "실행들"
    assert cli.main(["train", str(paths.builtin_env_dir() / "lander.evoenv"), "--seed", "4", "--popsize", "6",
                     "--trials", "2", "--max-generations", "3", "--workers", "0", "--out", str(out)]) == 0
    run = next(out.iterdir())
    assert cli.main(["bundle", str(run)]) == 0
    zips = list((run / "exports").glob("*.zip"))
    assert len(zips) == 1
    assert cli.main(["reproduce", str(zips[0]), "--out", str(tmp_path / "다시"), "--workers", "1"]) == 0
    assert "같습니다" in capsys.readouterr().out
    assert cli.main(["csv", str(run), "--out", str(tmp_path / "통계.csv")]) == 0
    raw = (tmp_path / "통계.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    rows = list(csv.DictReader((tmp_path / "통계.csv").open(encoding="utf-8-sig")))
    assert [r["generation"] for r in rows] == ["1", "2", "3"]
    assert export_csv(run, tmp_path / "b.csv").exists()


def test_bundle_with_restarts_qualifying_and_changes_reproduces(tmp_path: Path) -> None:
    """v1.1: 예선(2 × 24회)과 자동 재시작이 있는 실행에 시험 수·P 변경, 재시작 끄기와 다시 켜기, 환경 교체를 넣어도
    재현 통계가 같다(재시작·예선은 사건 기록에 없고 같은 세대 경계의 변경 재적용으로 다시 생긴다)."""
    config = TrainConfig(popsize=6, trials=2, validate_every=24, validation_seeds=4, validation_candidates=1,
                         checkpoint_every=2, run_seed=9, restart_patience=24, qualify_entrants=2, qualify_trials=24)
    store = create_run(tmp_path / "원래", LANDER_TEXT, compile_env(LANDER_TEXT), config, 0)
    with WorkerPool(0) as pool:
        t = Trainer(LANDER_TEXT, config, pool, store)

        def until(gen: int) -> None:
            while t.state.generation < gen:
                t.step()

        def change(**values: Any) -> None:
            changes = t.apply_config(values)
            assert changes == values
            store.append_event("config_change", t.state.generation, changes=changes)

        until(4)  # 예선 끝
        change(trials=3, restart_patience=36)
        until(7)
        change(restart_patience=None)
        until(9)
        change(restart_patience=24)  # 다시 켬: 기준점을 다시 잡는다
        data = json.loads(LANDER_TEXT)
        data["world"]["gravity"] = [0.0, -1.4]
        text = json.dumps(data, ensure_ascii=False, indent=2)
        t.replace_env(text)
        store.append_event("env_change", 9, version=store.add_env_version(text, 9))
        until(16)
        t.checkpoint()
    a = store.read_stats()
    switches = [(int(r["generation"]), int(r["switch"])) for r in a if r["switch"] != "0"]
    assert switches[:2] == [(2, 3), (4, 4)]  # 예선 다음 출전, 예선 끝
    assert [g for g, c in switches if c in (1, 2)] == [6, 12]  # P를 바꾼 뒤, 다시 켜고 환경을 바꾼 뒤의 재시작
    assert {e["kind"] for e in store.read_events()} == {"start", "config_change", "env_change"}
    bundle = export_bundle(store.root, tmp_path / "재현.zip")
    new = reproduce(bundle, tmp_path / "다시", workers=0)
    b = RunStore.open(new).read_stats()
    assert list(b[0]) == list(a[0])
    assert compare_stats(a, b) is None

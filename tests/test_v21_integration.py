"""v2.1 통합: 묶음 사이의 이음매(구조 뒤집기 학습의 이어하기·재현 묶음, 명령줄 이어하기 안내, 반복 실행 대화 상자의 새 실행
권장값, 대시보드 닫기)를 한 번에 본다. 각 묶음의 테스트(test_structure*.py, test_editor_v21.py)가 덮지 않는 곳만."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab import paths
from evolab.app import strings as S
from evolab.engine.pool import WorkerPool
from evolab.engine.runsetup import create_run
from evolab.engine.trainer import STRUCTURE_FLIP_NEW_RUN, TIME_COLUMNS, TrainConfig, Trainer
from evolab.policy.genome import layout_for
from evolab.runs.bundle import compare_stats, export_bundle, reproduce
from evolab.runs.store import RunStore
from evolab.spec import structure as st
from evolab.spec.io import compile_env

WHEELS = paths.builtin_env_dir() / "wheels.evoenv"
LANDER = paths.builtin_env_dir() / "lander.evoenv"


def short_wheels() -> str:
    """빠른 테스트용: 바퀴 고르는 언덕차를 4초로(구조·부품은 그대로)."""
    data = json.loads(WHEELS.read_text(encoding="utf-8"))
    data["world"]["max_time"] = 4.0
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def strip(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in r.items() if k not in TIME_COLUMNS} for r in rows]


def ga_config(spec: Any, gens: int) -> TrainConfig:
    """실험실·명령줄·반복 실행과 같은 새 실행 권장값(for_new_run(spec)): 바퀴 환경이면 GA 구조 뒤집기 0.05."""
    return TrainConfig.for_new_run(spec, method="ga", popsize=8, trials=2, validate_every=40, validation_seeds=3,
                                   checkpoint_every=1, run_seed=7, max_generations=gens)


def test_ga_structure_flip_run_resumes_and_reproduces_bit_identically(tmp_path: Path) -> None:
    """GA + 구조 뒤집기(0.05) 실행: 4세대 한 번에(워커 0) = 2세대 + 이어서 2세대(워커 2), 재현 묶음도 같은 통계.
    학습 방법 상태에 structure_* 키가 있고(0이 아니므로), 집단의 구조가 처음 몸(바퀴1만)에서 실제로 바뀐다."""
    text = short_wheels()
    spec = compile_env(text)
    layout = layout_for(spec)
    assert layout.on_genes and ga_config(spec, 4).ga_structure_flip == STRUCTURE_FLIP_NEW_RUN

    straight = create_run(tmp_path / "한 번에", text, spec, ga_config(spec, 4), 0)
    with WorkerPool(0) as pool:
        t = Trainer(text, ga_config(spec, 4), pool, straight)
        t.run()
        t.checkpoint()
    state = t.learner.state_dict()["config"]
    assert state["structure_flip"] == STRUCTURE_FLIP_NEW_RUN and tuple(state["structure_genes"]) == layout.on_genes
    first = st.structure_key(spec, st.param_values(spec.file, layout.body_values(layout.initial())))
    keys = {st.structure_key(spec, st.param_values(spec.file, layout.body_values(x))) for x in t.learner.ask()}
    assert first == ("wheel1",) and keys - {first}

    halves = create_run(tmp_path / "끊고 이어서", text, spec, ga_config(spec, 2), 2)
    with WorkerPool(2) as pool:
        h = Trainer(text, ga_config(spec, 2), pool, halves)
        h.run()
        h.checkpoint()
    with WorkerPool(2) as pool:
        r = Trainer.resume(RunStore.open(halves.root), pool, None, max_generations=4)
        r.run()
        r.checkpoint()
    assert strip(halves.read_stats()) == strip(straight.read_stats())
    assert r.learner.best().tobytes() == t.learner.best().tobytes()
    assert r.state.champion.score == t.state.champion.score

    new = reproduce(export_bundle(straight.root, tmp_path / "묶음.zip"), tmp_path / "다시", workers=2)
    assert compare_stats(straight.read_stats(), RunStore.open(new).read_stats()) is None


def test_cli_resume_of_a_flip_off_ga_run_on_wheels_prints_the_notice_once(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """명령줄: --structure-flip 0으로 시작한 GA 실행을 이어 가면 LAB_GA_NO_STRUCTURE를 한 번 알리고, 0.05 실행은 알리지 않는다."""
    from evolab.cli import main

    env = tmp_path / "wheels.evoenv"
    env.write_text(short_wheels(), encoding="utf-8")
    base = ["train", str(env), "--method", "ga", "--popsize", "4", "--trials", "1", "--max-generations", "1",
            "--workers", "0", "--no-observe"]

    def start(extra: list[str], name: str) -> Path:
        out = tmp_path / name
        assert main([*base, *extra, "--out", str(out)]) == 0
        capsys.readouterr()
        (run_dir,) = [p for p in out.iterdir() if p.is_dir()]
        return run_dir

    for extra, name, noted in ((["--structure-flip", "0"], "off", True), ([], "default", False)):
        run_dir = start(extra, name)
        assert main(["train", "--resume", str(run_dir), "--max-generations", "2", "--workers", "0"]) == 0
        out = capsys.readouterr()
        text = out.out + out.err
        assert text.count(S.LAB_GA_NO_STRUCTURE) == (1 if noted else 0), text


def test_repeat_dialog_new_runs_get_the_structure_flip_only_on_structure_envs(qtbot: Any, tmp_path: Path) -> None:
    """반복 실행 대화 상자(D41): 있음 유전자가 있는 환경의 GA·담금질 새 실행은 0.05, 구조 없는 환경은 0."""
    from evolab.app.compare.repeat import RepeatDialog

    dlg = RepeatDialog(runs_dir=tmp_path, workers=1)
    qtbot.addWidget(dlg)
    for env, want in ((WHEELS, STRUCTURE_FLIP_NEW_RUN), (LANDER, 0.0)):
        assert dlg.set_env_file(env) and dlg.spec_obj is not None
        for method in ("ga", "sa"):
            assert dlg.set_method(method)
            c = dlg.config()
            assert (c.ga_structure_flip, c.sa_structure_flip) == (want, want), (env.name, method)
            assert dlg.plan().config["ga_structure_flip"] == want


def test_dashboard_shutdown_waits_for_the_structure_tile_jobs(qtbot: Any, monkeypatch: pytest.MonkeyPatch,
                                                             tmp_path: Path) -> None:
    """대시보드를 닫으면(Dashboard.shutdown) 「몸 구조」 타일의 배경 몸 그림 스레드도 기다린다(실험실 화면이 아닌 곳에서도)."""
    from PySide6.QtCore import QSettings

    from evolab.app.lab import structure_tile
    from evolab.app.lab.dashboard import Dashboard

    calls: list[str] = []
    monkeypatch.setattr(structure_tile, "wait_jobs", lambda: calls.append("structure"))
    dash = Dashboard(settings=QSettings(str(tmp_path / "d.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(dash)
    dash.shutdown()
    assert calls == ["structure"]


def test_present_and_runtime_agree_on_the_built_in_example_bodies() -> None:
    """내장 예제 처음 몸과 완료 기준의 두 기준선: 런타임 Episode.absent = 구조 함수, moving_bodies는 설계 4.11의 확인 출력
    (처음 몸 [chassis, wheel1], 바퀴 2개 고정 [chassis, wheel1, wheel4])."""
    from evolab.engine import seeds
    from evolab.world.physics2d.episode import Episode

    data = Path(__file__).resolve().parent / "data" / "v2_structure"
    for path, moving in ((WHEELS, ["chassis", "wheel1"]), (data / "wheels_fixed1.evoenv", ["chassis", "wheel1"]),
                         (data / "wheels_fixed2.evoenv", ["chassis", "wheel1", "wheel4"])):
        spec = compile_env(path.read_text(encoding="utf-8"))
        layout = layout_for(spec)
        body = layout.body_values(layout.initial()) if layout.n_body else None
        ep = Episode(spec, body, seeds.display_trial())
        agents = frozenset(b.id for b in spec.file.bodies if b.owner == "agent")
        present = st.present(spec, st.param_values(spec.file, body))
        assert ep.absent == agents - present == agents - set(moving), path.name
        assert list(ep.moving_bodies) == moving, path.name
        assert np.all(np.isfinite(ep.observe()))

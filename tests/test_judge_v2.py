"""v2.0 판정·비교·경주·bench (설계 4.10, D21·D48, 8.4절 T53): 성공 판정이 없는 환경(글라이더)의 판정 출력과 「—」,
조립 실패 종료 원인 이름, bench의 v2 부품 줄."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QSettings

from evolab import cli, paths
from evolab.app import strings as S
from evolab.engine import commands, seeds
from evolab.engine.evaluate import BUILD_FAILED_EVENT, evaluate
from evolab.engine.pool import WorkerPool
from evolab.engine.runsetup import create_run
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.policy.genome import layout_for
from evolab.runs.store import RunStore
from evolab.spec.io import compile_env, load_env
from evolab.world.physics2d.episode import Episode

ENV = paths.builtin_env_dir()
JUMP = ENV / "jump.evoenv"
GLIDER = ENV / "glider.evoenv"
LANDER = ENV / "lander.evoenv"


def train_small(runs: Path, path: Path, gens: int = 2, seed: int = 7) -> RunStore:
    """작은 실행(개체 4 × 시험 1, 세대마다 검증해 챔피언이 있음)을 이 프로세스에서(워커 0)."""
    text = path.read_text(encoding="utf-8")
    spec = compile_env(text)
    config = TrainConfig(popsize=4, trials=1, run_seed=seed, checkpoint_every=2, max_generations=gens, validate_every=4)
    store = create_run(runs, text, spec, config, 0)
    with WorkerPool(0) as pool:
        t = Trainer(text, config, pool, store)
        reason = t.run()
    store.append_event("stop", t.state.generation, reason=reason)
    return store


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, RunStore]:
    base = tmp_path_factory.mktemp("runs")
    return {"glider": train_small(base, GLIDER), "lander": train_small(base, LANDER)}


def judged(store: RunStore, n: int) -> tuple[Any, Any, int]:
    """judge 명령과 같은 계산을 직접: (환경, 판정 결과, 챔피언 세대)."""
    gen = store.champion_generations("champion")[-1]
    champ = store.load_champion(gen, "champion")
    info = store.read_run_info()
    spec = compile_env(store.env_text(info["env_versions"][-1]["id"]))
    layout = layout_for(spec)
    trials = seeds.judge_trials(n)
    res = evaluate(spec, layout, np.asarray(champ["genome"])[None, :], [(0, t) for t in trials])
    return spec, res, gen


# ---------------------------------------------------------------------------
# T53: 판정 출력
def test_t53_judge_without_success_prints_mean_and_progress(runs: dict[str, RunStore],
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    store = runs["glider"]
    assert commands.judge_command(argparse.Namespace(run=str(store.root), seeds=6, generation=None, workers=0)) == 0
    out = capsys.readouterr().out.strip().split("\n")
    spec, res, gen = judged(store, 6)
    assert not commands.has_success(spec)
    dist = res.blocks[:, 0]
    want = S.JUDGE_RESULT_NO_SUCCESS.format(gen=gen, n=6, mean=float(res.scores.mean()),
                                            ends=commands.ends_text(spec, res.end_events)).split("\n")
    want.append(S.CLI_JUDGE_PROGRESS.format(block="비행 거리", median=float(np.median(dist)),
                                            p10=float(np.percentile(dist, 10))))
    assert out == want
    assert "%" not in out[0] and "성공 0/" not in out[0]  # 성공률(「성공 0/6 (0%)」)이 실패처럼 보이지 않게
    event = [e for e in store.read_events() if e.get("kind") == "judge"][-1]
    assert event["success"] == 0 and event["seeds"] == 6


def test_t53_judge_with_success_is_unchanged(runs: dict[str, RunStore], capsys: pytest.CaptureFixture[str]) -> None:
    store = runs["lander"]
    assert commands.judge_command(argparse.Namespace(run=str(store.root), seeds=4, generation=None, workers=0)) == 0
    out = capsys.readouterr().out.strip()
    spec, res, gen = judged(store, 4)
    success = sum(1 for o in res.outcomes if o == "success")
    assert out == S.JUDGE_RESULT.format(gen=gen, n=4, success=success, rate=100.0 * success / 4,
                                        mean=float(res.scores.mean()), ends=commands.ends_text(spec, res.end_events))
    assert commands.has_success(spec) and S.CLI_JUDGE_PROGRESS.split("(")[0] not in out


def test_ends_text_names_build_failures_and_progress_lines() -> None:
    spec = load_env(GLIDER)
    assert BUILD_FAILED_EVENT == ""
    assert commands.ends_text(spec, ["landed", BUILD_FAILED_EVENT, "landed", "timeout"]) == \
        f"착지 2, {S.RT_BUILD_FAILED_END} 1, 시간 초과 1"
    lines = commands.progress_lines(spec, np.array([[10.0], [20.0], [30.0], [math.nan]]))
    assert lines == [S.CLI_JUDGE_PROGRESS.format(block="비행 거리", median=20.0, p10=12.0)]
    jump = load_env(JUMP)
    assert commands.progress_lines(jump, np.zeros((3, 2))) == []  # 진행 거리 블록이 없으면 줄도 없음
    assert commands.has_success(jump) and commands.has_success(load_env(LANDER))


# ---------------------------------------------------------------------------
# T53: 비교 표·경주·기록
def test_t53_compare_table_shows_a_dash_for_runs_without_success(qtbot: Any, runs: dict[str, RunStore],
                                                                 tmp_path: Path) -> None:
    from evolab.app.compare import view as cmp
    from evolab.app.compare.view import TABLE_COL_SUCCESS, CompareView

    base = runs["glider"].root.parent
    view = CompareView(lambda: base, settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.show()
    view.set_selected_runs([runs["glider"].root, runs["lander"].root])
    qtbot.waitUntil(lambda: len(view.rows) == 2, timeout=10000)
    texts = {row.paths[0]: t for row, t in zip(view.rows, view.table_texts(), strict=True)}
    assert texts[runs["glider"].root][TABLE_COL_SUCCESS] == S.NO_SUCCESS_CELL
    assert texts[runs["lander"].root][TABLE_COL_SUCCESS].endswith("%")
    assert cmp.env_has_success(runs["lander"].root) and not cmp.env_has_success(runs["glider"].root)
    assert cmp.env_has_success(tmp_path / "없는 실행")  # 읽을 수 없는 실행은 지금처럼 숫자
    out = tmp_path / "표.csv"
    view.save_table_csv(out)
    rows = out.read_text(encoding="utf-8-sig").splitlines()
    glider_row = next(r for r in rows[1:] if r.startswith("글라이더"))
    lander_row = next(r for r in rows[1:] if r.startswith("달 착륙선"))
    assert glider_row.split(",")[-2:] == ["", ""]  # 성공률 칸은 비움(0%가 아님)
    assert lander_row.split(",")[-2] != ""


def test_t53_race_text_and_records_have_no_success_rate_for_the_glider(qtbot: Any, runs: dict[str, RunStore],
                                                                       tmp_path: Path) -> None:
    """경주 칸 글·결과 글과 기록 화면은 성공률을 보이지 않는다(끝 이름·점수·순위만): 「0%」가 나올 자리가 없다."""
    from evolab.app.race.model import entries_from_runs
    from evolab.app.race.view import RaceView
    from evolab.app.records.view import RecordsView

    entries, _warn, _skipped = entries_from_runs([(runs["glider"].root, "#e76f51"), (runs["glider"].root, "#2a9d8f")],
                                                 "last")
    view = RaceView()
    qtbot.addWidget(view)
    view.set_race(entries)
    qtbot.waitUntil(lambda: view.ready, timeout=30000)
    view.pause()
    view.advance(view.race_duration() + 5.0)
    view.advance(2.0)
    assert view.finished
    lanes = [lane.status.text() for lane in view.lanes]
    assert all("%" not in t and any(e in t for e in ("착지", "이탈", "시간 초과")) for t in lanes)
    assert view.result_label.text() and "%" not in view.result_label.text()
    rec = RecordsView(lambda: runs["glider"].root.parent)
    qtbot.addWidget(rec)
    rec.refresh()
    cells = [rec.table.item(r, c).text() for r in range(rec.table.rowCount()) for c in range(rec.table.columnCount())
             if rec.table.item(r, c) is not None]
    assert cells and not any(t.endswith("%") for t in cells)
    view.shutdown()


# ---------------------------------------------------------------------------
# bench의 v2 부품 줄(D48)
def test_bench_v2_parts_counts_movers_edges_and_clamps() -> None:
    for path, movers in ((JUMP, 1), (GLIDER, 0)):
        spec = load_env(path)
        layout = layout_for(spec)
        X = layout.initial()[None, :] + 0.5 * np.random.default_rng(0).standard_normal((2, layout.dim))
        pairs = [(i, t) for i in range(2) for t in seeds.train_trials(1, 0, 2)]
        parts = commands.v2_parts(spec, layout, X, pairs)
        ep = Episode(spec, None, pairs[0][1])
        assert parts == (movers, ep.aero_edge_count, 0)
    assert commands.v2_parts(load_env(GLIDER), layout_for(load_env(GLIDER)), np.zeros((1, 8)), []) is None
    lander = load_env(LANDER)
    ll = layout_for(lander)
    assert commands.v2_parts(lander, ll, ll.initial()[None, :], [(0, seeds.display_trial())]) is None


def test_bench_prints_the_v2_parts_line(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["bench", str(GLIDER), "--workers", "1", "--generations", "1", "--popsize", "2"]) == 0
    out = capsys.readouterr().out.strip().split("\n")
    edges = Episode(load_env(GLIDER), None, seeds.display_trial()).aero_edge_count
    assert out[-1] == S.BENCH_V2_PARTS.format(movers=0, edges=edges, clamped=0)
    assert out[-2].startswith(S.BENCH_WORST.split("{")[0])

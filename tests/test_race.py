"""나란히 경주 (설계 4.6, 테스트 T42~T46): 칸 고르기, 코스, 정확성 고리, 경주 창, 경주 GIF, 비교 화면 입구."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QWidget

from evolab import paths
from evolab.app import strings as S
from evolab.app.race import model as M
from evolab.app.race.model import RaceEntry
from evolab.app.race.view import FAST_MS, PAINT_WINDOW, SLOW_MS, RaceDialog, RaceView, show_race
from evolab.engine import seeds
from evolab.engine.evaluate import evaluate
from evolab.engine.pool import WorkerPool
from evolab.engine.replay import replay
from evolab.engine.runsetup import create_run, platform_info
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.policy.genome import layout_for
from evolab.runs.store import RunStore
from evolab.spec.io import compile_env, env_hash

LANDER = paths.builtin_env_dir() / "lander.evoenv"
LANDER_TEXT = LANDER.read_text(encoding="utf-8")
GRAVITY = '"gravity": [0.0, -1.62]'
OTHER_TEXT = LANDER_TEXT.replace(GRAVITY, '"gravity": [0.0, -1.3]')
DATA = Path(__file__).resolve().parent / "data"
COLORS = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9")


def lander_hash(text: str) -> str:
    return env_hash(compile_env(text).file)[:16]


def make_run(base: Path, method: str, seed: int, pop: int, gens: int, validate_every: int,
             changes: dict[int, str] | None = None) -> Path:
    """작은 착륙선 실행(워커 없이, 검증은 후보 1개 × 시드 2개로 짧게). changes[g]: g세대를 끝낸 뒤 바꿀 환경 글."""
    config = TrainConfig(method=method, run_seed=seed, popsize=pop, trials=2, validate_every=validate_every,
                         validation_seeds=2, validation_candidates=1, checkpoint_every=0, max_generations=gens)
    store = create_run(base, LANDER_TEXT, compile_env(LANDER_TEXT), config, 0)
    with WorkerPool(0) as pool:
        t = Trainer(LANDER_TEXT, config, pool, store, observe=False)
        while t.state.generation < gens:
            if changes and t.state.generation in changes:
                text = changes[t.state.generation]
                t.replace_env(text)
                store.add_env_version(text, t.state.generation)
            t.step()
        t.finish()
    return store.root


@dataclass
class Runs:
    base: Path
    a: Path
    """CMA-ES 개체 4, 4세대(누적 시험 8·16·24·32), 검증 16회마다(2·4세대)."""
    b: Path
    """GA 개체 6, 2세대(12·24), 검증 2세대."""
    c: Path
    """CMA-ES 개체 4, 1세대, 검증 없음(챔피언 없음)."""
    d: Path
    """CMA-ES 개체 4, 4세대, 매 세대 검증, 2세대 끝에 중력을 바꿈(3·4세대는 다른 환경)."""


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Runs:
    assert OTHER_TEXT != LANDER_TEXT
    base = tmp_path_factory.mktemp("race") / "runs"
    a = make_run(base, "cmaes", 1, 4, 4, 16)
    b = make_run(base, "ga", 2, 6, 2, 24)
    c = make_run(base, "cmaes", 3, 4, 1, 1000)
    d = make_run(base, "cmaes", 4, 4, 4, 8, changes={2: OTHER_TEXT})
    return Runs(base, a, b, c, d)


def rows(run: Path) -> list[dict[str, str]]:
    return RunStore.open(run).read_stats()


def champion_at(row: dict[str, str]) -> int:
    return int(float(row["champion_generation"]))


def same_platform(run: Path) -> bool:
    now = platform_info(0)
    info = RunStore.open(run).read_run_info()
    return all(info["platform"].get(k) == now[k] for k in ("python", "numpy", "pymunk", "cma", "cpu"))


def copy_fixture(name: str, tmp_path: Path) -> Path:
    dst = tmp_path / name
    shutil.copytree(DATA / name / "run", dst)
    return dst


def replay_entry(entry: RaceEntry, course: int) -> Any:
    spec = M.compile_cached(entry.env_text)
    return replay(spec, layout_for(spec), entry.genome, M.course_trial(course))


# ---------------------------------------------------------------------------
# T42 모델
def test_courses_are_the_compare_set_and_wrap() -> None:
    trials = seeds.compare_trials()
    assert M.COURSES == seeds.COMPARE_SET == 32
    assert [M.course_trial(n) for n in range(1, 33)] == trials
    assert M.next_course(1) == 2 and M.next_course(31) == 32 and M.next_course(32) == 1
    for bad in (0, 33, -1, 1.0, True):
        with pytest.raises(ValueError):
            M.course_trial(bad)  # type: ignore[arg-type]
    validation = {t.seed for t in seeds.validation_trials()}
    assert not validation & {t.seed for t in trials}  # 선정에 쓴 시드와 겹치지 않음
    assert [M.lane_grid(n) for n in range(1, 7)] == [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (2, 3)]


def test_last_champion_entries(runs: Runs) -> None:
    entries, warnings, n = M.entries_from_runs([(runs.a, COLORS[0]), (runs.b, COLORS[1])], "last")
    assert n is None and warnings == []
    assert [e.color for e in entries] == [COLORS[0], COLORS[1]]
    assert [e.label for e in entries] == [S.RACE_LANE_RUN.format(method=S.METHOD_NAMES["cmaes"], seed=1),
                                          S.RACE_LANE_RUN.format(method=S.METHOD_NAMES["ga"], seed=2)]
    for e, run in zip(entries, (runs.a, runs.b), strict=True):
        last = rows(run)[-1]
        champ = champion_at(last)
        assert champ >= 1 and e.generation == champ
        data = RunStore.open(run).load_champion(champ, "champion")
        assert e.genome.tobytes() == np.asarray(data["genome"], dtype=np.float64).tobytes()
        assert not e.genome.flags.writeable
        assert e.env_hash == str(data["env_version"]) == lander_hash(LANDER_TEXT)
        evals = next(int(float(r["evaluations"])) for r in rows(run) if int(r["generation"]) == champ)
        assert e.sub == S.RACE_LANE_CHAMPION.format(gen=champ, evals=evals)
        assert e.run == run
    assert M.same_env(entries)


def test_same_budget_rule_and_missing_champion(runs: Runs) -> None:
    # N = min(마지막 누적 시험) = 24 (B). A는 누적 24 이하 마지막 줄(3세대)의 챔피언
    entries, warnings, n = M.entries_from_runs([(runs.a, COLORS[0]), (runs.b, COLORS[1])], "budget")
    assert n == 24 and warnings == []
    a_row = [r for r in rows(runs.a) if float(r["evaluations"]) <= 24][-1]
    assert int(a_row["generation"]) == 3
    assert entries[0].generation == champion_at(a_row)
    assert entries[1].generation == champion_at(rows(runs.b)[-1])
    assert M.common_budget([(runs.a, ""), (runs.b, "")]) == 24

    # C(1세대, 챔피언 없음)가 끼면 N = 8: A·B 모두 그때 챔피언이 없어 세대 대표로 달림
    picks = [(runs.a, COLORS[0]), (runs.b, COLORS[1]), (runs.c, COLORS[2])]
    entries, warnings, n = M.entries_from_runs(picks, "budget")
    assert n == 8
    assert [e.generation for e in entries] == [1, 1, 1]
    for e, run in zip(entries, (runs.a, runs.b, runs.c), strict=True):
        rep = RunStore.open(run).load_champion(1, "representative")
        assert e.genome.tobytes() == np.asarray(rep["genome"]).tobytes()
        assert e.sub == S.RACE_LANE_REP_INSTEAD.format(gen=1)
        assert S.RACE_NO_CHAMPION.format(name=e.label, gen=1) in warnings
    # 마지막 챔피언이라도 챔피언이 한 번도 없던 실행(C)은 마지막 세대 대표
    entries, warnings, _ = M.entries_from_runs(picks, "last")
    assert entries[2].generation == 1 and entries[2].sub == S.RACE_LANE_REP_INSTEAD.format(gen=1)
    assert warnings == [S.RACE_NO_CHAMPION.format(name=entries[2].label, gen=1)]


def test_each_lane_runs_in_the_env_it_was_trained_in(runs: Runs, tmp_path: Path) -> None:
    store = RunStore.open(runs.d)
    assert [v["from_generation"] for v in store.read_run_info()["env_versions"]] == [0, 2]
    assert M.env_text_at(store, 1) == M.env_text_at(store, 2) == LANDER_TEXT
    assert M.env_text_at(store, 3) == M.env_text_at(store, 4) == OTHER_TEXT

    entries, warnings = M.entries_from_generations(runs.d, 4, None, COLORS)
    assert warnings == [] and [e.generation for e in entries] == [1, 2, 3, 4]
    assert [e.env_text for e in entries] == [LANDER_TEXT, LANDER_TEXT, OTHER_TEXT, OTHER_TEXT]
    for e in entries:
        assert e.env_hash == str(store.load_champion(e.generation, "representative")["env_version"])
    assert not M.same_env(entries)
    assert M.ranks(entries, [1.0, 2.0, 3.0, 4.0]) is None

    (last,), _, _ = M.entries_from_runs([(runs.d, COLORS[0])], "last")
    champ_file = store.load_champion(last.generation, "champion")
    assert last.env_hash == str(champ_file["env_version"])
    assert last.env_text == (OTHER_TEXT if last.generation >= 3 else LANDER_TEXT)

    # 드문 경우: run.json의 적용 세대가 파일과 어긋나면 파일의 env_version과 해시가 같은 버전을 쓴다
    moved = tmp_path / "moved"
    shutil.copytree(runs.d, moved)
    ms = RunStore.open(moved)
    versions = ms.read_run_info()["env_versions"]
    versions[1]["from_generation"] = 3
    ms.update_run_info(env_versions=versions)
    assert M.env_text_at(ms, 3) == LANDER_TEXT
    entries, _ = M.entries_from_generations(moved, 4, None, COLORS)
    assert [e.env_text for e in entries] == [LANDER_TEXT, LANDER_TEXT, OTHER_TEXT, OTHER_TEXT]


def test_generation_race_entries(runs: Runs) -> None:
    assert M.race_generations(4) == [1, 2, 3, 4]
    assert M.race_generations(10) == [1, 3, 5, 8, 10]
    assert M.race_generations(2) == [1, 2]
    assert M.entries_from_generations(runs.a, 1, None, COLORS) == ([], [S.RACE_NEED_GENS])
    champ = (2, RunStore.open(runs.a).load_champion(2, "champion")["genome"])
    entries, warnings = M.entries_from_generations(runs.a, 4, champ, COLORS)
    assert warnings == []
    assert len(entries) == 5 <= M.MAX_LANES
    assert [e.label for e in entries] == [S.RACE_LANE_GEN.format(gen=g) for g in (1, 2, 3, 4)] + [
        S.RACE_LANE_NOW.format(gen=2)]
    assert [e.color for e in entries] == list(COLORS[:5])
    assert entries[-1].genome.tobytes() == np.asarray(champ[1]).tobytes()
    assert M.same_env(entries)
    # 세대 대표 파일이 없으면 빼고 알린다
    gone = RunStore.open(runs.a).root / "champions" / "gen_000003_representative.npz"
    saved = gone.read_bytes()
    gone.unlink()
    try:
        entries, warnings = M.entries_from_generations(runs.a, 4, None, COLORS)
        assert [e.generation for e in entries] == [1, 2, 4]
        assert len(warnings) == 1 and warnings[0].startswith(S.RACE_LANE_GEN.format(gen=3))
    finally:
        gone.write_bytes(saved)


def test_old_runs_too_many_and_unreadable(runs: Runs, tmp_path: Path) -> None:
    v10 = copy_fixture("v10_run_ga", tmp_path)
    v11 = copy_fixture("v11_run_restart", tmp_path)
    entries, warnings, _ = M.entries_from_runs([(v10, COLORS[0]), (v11, COLORS[1]), (runs.a, COLORS[2])], "last")
    assert warnings == []
    assert [e.label for e in entries][:2] == [S.RACE_LANE_RUN.format(method=S.METHOD_NAMES["ga"], seed=3),
                                              S.RACE_LANE_RUN.format(method=S.METHOD_NAMES["cmaes"], seed=11)]
    assert entries[0].generation == champion_at(rows(v10)[-1])
    assert entries[1].generation == champion_at(rows(v11)[-1])
    assert M.same_env(entries)

    seven = [(runs.a, c) for c in COLORS]
    entries, warnings, _ = M.entries_from_runs(seven, "last")
    assert len(entries) == M.MAX_LANES == 6
    assert warnings == [S.RACE_TOO_MANY.format(max=6)]
    assert [e.color for e in entries] == list(COLORS[:6])

    broken = tmp_path / "broken"
    broken.mkdir()
    entries, warnings, _ = M.entries_from_runs([(broken, COLORS[0]), (runs.a, COLORS[1])], "last")
    assert len(entries) == 1 and len(warnings) == 1 and warnings[0].startswith("broken: ")
    empty = RunStore.create(tmp_path, "empty", {"env_name": "x", "method": "cmaes", "config": {"run_seed": 1}})
    empty.add_env_version(LANDER_TEXT, 0)
    entries, warnings, _ = M.entries_from_runs([(empty.root, COLORS[0]), (runs.a, COLORS[1])], "last")
    assert len(entries) == 1
    assert warnings == [S.RACE_SKIPPED.format(name=S.RACE_LANE_RUN_ENV.format(
        env="x", method=S.METHOD_NAMES["cmaes"], seed=1), reason=S.RACE_NO_GENERATIONS)]
    with pytest.raises(ValueError):
        M.entries_from_runs([], "first")


def test_ranks_with_ties() -> None:
    def entry(h: str) -> RaceEntry:
        return RaceEntry("x", "", "#fff", "", h, np.zeros(3), 1)

    same = [entry("e")] * 4
    assert M.ranks(same, [5.0, 7.0, 7.0, 1.0]) == [(3, False), (1, True), (1, True), (4, False)]
    assert M.ranks(same, [2.0, 2.0, 2.0, 2.0]) == [(1, True)] * 4
    assert M.ranks(same, [1.0, math.nan, 3.0, 2.0]) == [(3, False), (4, False), (1, False), (2, False)]
    assert M.ranks([entry("e"), entry("f")], [1.0, 2.0]) is None
    with pytest.raises(ValueError):
        M.ranks(same, [1.0])
    genome = np.arange(3.0)
    e = RaceEntry("x", "", "#fff", "", "e", genome, 1)
    genome[0] = 9.0
    assert e.genome[0] == 0.0  # 사본


# ---------------------------------------------------------------------------
# T43 정확성 고리, T44 재생 점수 = 워커 점수
def ring_mean(entry: RaceEntry) -> float:
    return float(np.array([replay_entry(entry, n).result.score for n in range(1, M.COURSES + 1)]).mean())


def test_accuracy_ring_course_mean_equals_compare_score(runs: Runs) -> None:
    for run in (runs.a, runs.d):
        (entry,), _, _ = M.entries_from_runs([(run, COLORS[0])], "last")
        data = RunStore.open(run).load_champion(entry.generation, "champion")
        assert ring_mean(entry) == float(data["compare_score"])


def test_accuracy_ring_on_real_v10_and_v11_runs(tmp_path: Path) -> None:
    for name in ("v10_run_ga", "v11_run_restart"):
        run = copy_fixture(name, tmp_path)
        if not same_platform(run):
            pytest.skip("fixture numbers are platform specific")
        (entry,), _, _ = M.entries_from_runs([(run, COLORS[0])], "last")
        data = RunStore.open(run).load_champion(entry.generation, "champion")
        assert ring_mean(entry) == float(data["compare_score"])


def test_lane_replay_scores_equal_worker_scores(qtbot: Any, runs: Runs) -> None:
    picks = [(runs.a, COLORS[0]), (runs.b, COLORS[1]), (runs.c, COLORS[2]), (runs.d, COLORS[3])]
    entries, _, _ = M.entries_from_runs(picks, "last")
    view = RaceView()
    qtbot.addWidget(view)
    view.set_race(entries)
    view.set_course(7)
    qtbot.waitUntil(lambda: view.ready, timeout=30000)
    trial = M.course_trial(7)
    with WorkerPool(1) as pool:
        for k, e in enumerate(entries):
            spec = compile_env(e.env_text)
            layout = layout_for(spec)
            res = pool.evaluate(e.env_hash, e.env_text, layout, e.genome[None, :], [(0, trial)])
            local = evaluate(spec, layout, e.genome[None, :], [(0, trial)])
            assert view.scores[k] == float(res.scores[0]) == float(local.scores[0])
            assert view.replays[k].result.steps == int(res.steps[0])


# ---------------------------------------------------------------------------
# T45 경주 창
@pytest.fixture()
def owner(qtbot: Any) -> QWidget:
    w = QWidget()
    qtbot.addWidget(w)
    w.show()
    return w


def wait_ready(qtbot: Any, view: RaceView) -> None:
    qtbot.waitUntil(lambda: view.ready, timeout=30000)
    view.pause()


def test_race_dialog_runs_to_ranks(qtbot: Any, runs: Runs, owner: QWidget) -> None:
    (a, b), _, _ = M.entries_from_runs([(runs.a, COLORS[0]), (runs.b, COLORS[1])], "last")
    twin = RaceEntry("twin", "", COLORS[2], a.env_text, a.env_hash, a.genome, a.generation)
    dlg = show_race(owner, [a, b, twin], title=S.RACE_TITLE)
    assert isinstance(dlg, RaceDialog) and dlg.parent() is owner.window() and dlg.isVisible()
    assert dlg.windowTitle() == S.RACE_TITLE
    view = dlg.view
    assert len(view.lanes) == 3
    assert {view.grid.getItemPosition(view.grid.indexOf(lane))[:2] for lane in view.lanes} == {(0, 0), (0, 1), (0, 2)}
    assert all(lane.status.text() == S.RACE_PREPARING.format(done=0, total=3) for lane in view.lanes)
    # 준비 중인 칸 캔버스에는 실험실 빈 화면 글(SPEC_NO_SCENE)을 쓰지 않는다(칸 아래 글줄이 알린다)
    assert all(lane.canvas.world is None and lane.canvas.empty_text == "" for lane in view.lanes)
    qtbot.waitUntil(lambda: view.ready and view.playing, timeout=30000)  # 준비되면 저절로 출발
    view.pause()
    assert view.start_button.text() == S.RACE_START
    assert not view.who_box.isVisible()  # 세대·묶음 없이 연 경주: 누구를 없음
    duration = view.race_duration()
    assert duration == max(r.trajectory.duration for r in view.replays) > 0

    view.seek(0.0)
    view.set_speed(4.0)
    view.advance(0.25)
    assert view.t == pytest.approx(min(1.0, duration))
    assert view.hud_label.text() == S.RACE_HUD.format(n=1, t=view.t, speed=4.0)
    assert all(lane.status.text() == S.RACE_RUNNING.format(t=view.t) for lane in view.lanes
               if lane.duration > view.t)
    view.advance(1e6)
    assert view.t == duration and not view.finished
    for lane in view.lanes:
        r = lane.replay.result
        assert lane.status.text() == S.RACE_LANE_DONE.format(end=r.end_label, t=r.time, score=r.score)
    view.advance(0.5)
    assert not view.finished  # 다 끝나고 1초(실제 시간)를 기다린다
    view.advance(0.5)
    assert view.finished and not view.playing
    rk = view.rank_list
    assert rk is not None and rk[0] == rk[2] and rk[0][1]  # 같은 유전자 = 공동 순위
    for lane, (rank, tie) in zip(view.lanes, rk, strict=True):
        r = lane.replay.result
        fmt = S.RACE_LANE_RANK_TIE if tie else S.RACE_LANE_RANK
        assert lane.status.text() == fmt.format(rank=rank, end=r.end_label, t=r.time, score=r.score)
    assert view.result_label.text().startswith(S.RACE_RESULT.format(order="")[:3])
    assert view.scores[0] == view.scores[2]

    # 출발을 다시 누르면 처음부터
    view.start()
    assert view.playing and view.t == 0.0 and view.rank_list is None
    view.pause()

    # 한 화면에 겹쳐서: 캔버스 하나에 모든 칸
    view.set_mode("overlay")
    assert view.stack.currentWidget() is view.overlay_canvas
    assert all(view.overlay_canvas.overlay_frame(k) is not None for k in range(3))
    view.set_mode("split")
    assert view.stack.currentWidget() is view.grid_widget

    # 다른 코스: 다음 번호, 32 다음 1, 바꾸면 다시 준비
    view.set_course(32)
    assert not view.ready
    view.next_course()
    assert view.course == 1 and view.course_box.currentData() == 1
    wait_ready(qtbot, view)

    # 다시 열면 같은 창이 앞으로(내용만 바뀜)
    dlg.close()
    again = show_race(owner, [a, b], title=S.RACE_TITLE)
    assert again is dlg and again.isVisible() and len(again.view.lanes) == 2
    wait_ready(qtbot, again.view)
    again.close()


def test_mixed_envs_have_no_ranks_and_no_overlay(qtbot: Any, runs: Runs, owner: QWidget) -> None:
    entries, _ = M.entries_from_generations(runs.d, 4, None, COLORS)
    dlg = show_race(owner, entries, title=S.RACE_TITLE_GENERATIONS.format(run="d"))
    view = dlg.view
    assert dlg.windowTitle() == S.RACE_TITLE_GENERATIONS.format(run="d")
    assert {view.grid.getItemPosition(view.grid.indexOf(lane))[:2] for lane in view.lanes} == {
        (0, 0), (0, 1), (1, 0), (1, 1)}
    assert S.RACE_DIFFERENT_ENV in view.band_lines()
    assert not view.overlay_allowed()
    item = view.view_box.model().item(1)
    assert not item.isEnabled()
    view.set_mode("overlay")
    assert view.mode == "split"
    wait_ready(qtbot, view)
    view.advance(1e6)
    view.advance(2.0)
    assert view.finished and view.rank_list is None
    assert view.result_label.text() == ""
    for lane in view.lanes:
        r = lane.replay.result
        assert lane.status.text() == S.RACE_LANE_DONE.format(end=r.end_label, t=r.time, score=r.score)
    dlg.close()


def test_who_switch_rebuilds_lanes(qtbot: Any, runs: Runs, owner: QWidget) -> None:
    picks = [(runs.a, COLORS[0]), (runs.b, COLORS[1]), (runs.c, COLORS[2])]
    entries, warnings, _ = M.entries_from_runs(picks, "last")
    dlg = show_race(owner, entries, title=S.RACE_TITLE, warnings=warnings, runs=picks)
    view = dlg.view
    assert view.who_box.isVisible() and view.who == "last"
    assert view.who_box.toolTip() == S.RACE_WHO_TIP.format(trials=8)
    assert view.band_lines() == warnings
    wait_ready(qtbot, view)
    view.set_who("budget")
    assert [e.generation for e in view.entries] == [1, 1, 1]
    assert len(view.band_lines()) == 3  # 세 실행 모두 그때 챔피언이 없음
    wait_ready(qtbot, view)
    dlg.close()


def test_frame_rate_drops_to_30_when_painting_is_slow(qtbot: Any, runs: Runs) -> None:
    (a, b), _, _ = M.entries_from_runs([(runs.a, COLORS[0]), (runs.b, COLORS[1])], "last")
    view = RaceView()
    qtbot.addWidget(view)
    view.set_race([a, b])
    qtbot.waitUntil(lambda: view.ready, timeout=30000)
    assert view.interval_ms == FAST_MS
    for _ in range(PAINT_WINDOW):
        view._on_paint(5.0)
        view._record_paint()
    assert view.interval_ms == FAST_MS
    for _ in range(PAINT_WINDOW):
        view._on_paint(13.0)
        view._record_paint()
    assert view.interval_ms == SLOW_MS
    view.prepare()  # 새 경주는 다시 60fps부터
    assert view.interval_ms == FAST_MS
    qtbot.waitUntil(lambda: view.ready, timeout=30000)
    view.shutdown()


def test_real_clock_paints_lanes(qtbot: Any, runs: Runs, owner: QWidget) -> None:
    (a, b), _, _ = M.entries_from_runs([(runs.a, COLORS[0]), (runs.b, COLORS[1])], "last")
    dlg = show_race(owner, [a, b], title=S.RACE_TITLE)
    view = dlg.view
    qtbot.waitUntil(lambda: view.playing, timeout=30000)
    qtbot.waitUntil(lambda: len(view.paint_log) >= 5 and view.t > 0.05, timeout=10000)
    assert all(ms > 0 for ms in view.paint_log)
    dlg.close()
    assert not view.playing


# ---------------------------------------------------------------------------
# T46 경주 GIF
def test_race_gif_frames_and_grid(qtbot: Any, runs: Runs, tmp_path: Path) -> None:
    from PIL import Image
    from PySide6.QtGui import QColor

    from evolab.app.export.video import export_race_gif, race_frame_count, render_race_frames

    entries, _, _ = M.entries_from_runs([(runs.a, COLORS[0]), (runs.b, COLORS[1]), (runs.c, COLORS[2])], "last")
    reps = [replay_entry(e, 3) for e in entries]
    lanes = [(r, QColor(e.color), e.label) for r, e in zip(reps, entries, strict=True)]
    durations = [r.trajectory.duration for r in reps]
    longest = max(durations)
    expected = math.ceil((longest + 1.0) * 20) + 1
    assert race_frame_count(longest) == expected
    frames = render_race_frames(lanes, 3, title=S.VIDEO_RACE_TITLE.format(n=3))
    assert len(frames) == expected
    assert {(f.width(), f.height()) for f in frames} == {(960, 200)}
    frames = render_race_frames(lanes, 2, progress=lambda d, t: d < 1)  # 첫 프레임에서 멈춤
    assert [(f.width(), f.height()) for f in frames] == [(640, 400)]  # 3칸을 2열로: 2줄

    path = tmp_path / "race.gif"
    calls: list[int] = []
    assert export_race_gif(lanes[:2], path, 2, progress=lambda d, t: calls.append(d) is None and d < 3) is None
    assert not path.exists()  # 취소하면 쓰지 않음
    out = export_race_gif(lanes[:2], path, 2)
    assert out == path
    with Image.open(path) as im:
        assert im.size == (640, 200)
        assert im.n_frames == race_frame_count(max(durations[:2]))


def test_race_view_saves_gif(qtbot: Any, runs: Runs, owner: QWidget, tmp_path: Path) -> None:
    from PIL import Image

    (a, b), _, _ = M.entries_from_runs([(runs.a, COLORS[0]), (runs.b, COLORS[1])], "last")
    dlg = show_race(owner, [a, b], title=S.RACE_TITLE)
    view = dlg.view
    wait_ready(qtbot, view)
    default = view.default_gif_path()
    assert default.parent == runs.a / "exports" and default.suffix == ".gif"
    view.ask_save_path = lambda d: tmp_path / "split"  # type: ignore[method-assign]
    out = view.save_gif()
    assert out == tmp_path / "split.gif" and out.exists()
    assert view.message.text() == S.VIDEO_DONE.format(path=out)
    with Image.open(out) as im:
        assert im.size == (640, 200)
    view.set_mode("overlay")
    out2 = view.save_gif(tmp_path / "overlay.gif")
    with Image.open(out2) as im:
        assert im.size == (640, 400)  # 기존 겹친 영상 크기
    dlg.close()


# ---------------------------------------------------------------------------
# 비교 화면 입구
def test_compare_view_race_button(qtbot: Any, runs: Runs, tmp_path: Path) -> None:
    from evolab.app.compare.view import CompareView

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    view = CompareView(lambda: runs.base, settings=settings)
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.show()
    assert not view.race_button.isEnabled()
    assert view.race_button.text() == S.CMP_RACE_BUTTON
    view.set_selected_runs([runs.a])
    assert not view.race_button.isEnabled()
    assert S.RACE_TOO_FEW in view.race_button.toolTip()
    view.set_selected_runs([runs.a, runs.b])
    assert view.race_button.isEnabled()
    colors = {str(e.path): s.color for s in view.shown for e in s.entries}
    dlg = view.open_race()
    assert isinstance(dlg, RaceDialog) and dlg.isVisible() and view.race_dialog is dlg
    assert dlg.windowTitle() == S.RACE_TITLE
    assert [e.run for e in dlg.view.entries] == view.selected_runs()
    assert [e.color for e in dlg.view.entries] == [colors[str(e.run)] for e in dlg.view.entries]
    assert dlg.view.who_box.isVisible()
    wait_ready(qtbot, dlg.view)

    # 묶음: 구성 실행이 묶음 색으로 들어감, 다시 누르면 같은 창
    saved = {run: (run / "run.json").read_bytes() for run in (runs.a, runs.c)}
    RunStore.open(runs.a).update_run_info(group={"id": "g-race", "name": "묶음", "index": 0, "count": 2})
    RunStore.open(runs.c).update_run_info(group={"id": "g-race", "name": "묶음", "index": 1, "count": 2})
    try:
        view.refresh()
        view.set_selected_runs([])
        assert view.set_group_checked("g-race")
        group = next(s for s in view.shown if s.kind == "group")
        again = view.open_race()
        assert again is dlg
        assert [e.color for e in again.view.entries] == [group.color, group.color]
        wait_ready(qtbot, again.view)
    finally:
        for run, data in saved.items():
            (run / "run.json").write_bytes(data)
    dlg.close()

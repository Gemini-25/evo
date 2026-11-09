"""v1.2 실험실 연결 (설계 4.3, 5.1, D16·D17·D19·D24, 8.3절 T38~T41): 보는 재미 기록·보는 세대·세대 넘겨 보기·
뇌 원천·지도 점 재생·궤적 밀도 세계·세대끼리 경주.

학습 프로세스를 띄우는 시험은 둘(관전, 고속: 워커 1, 개체 6 × 시험 2, 3세대)이다. 나머지는 이 프로세스에서 학습한
작은 실행(워커 0)이나 가짜 기록으로 만든 실행을 연다.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QElapsedTimer, QSettings, QTimer
from PySide6.QtWidgets import QMessageBox, QSlider
from test_history import fake_generation, stats_row

from evolab import paths
from evolab.app import errors
from evolab.app import strings as S
from evolab.app.lab import view as lab_view
from evolab.app.lab.brain_view import BrainTile
from evolab.app.lab.view import LabView
from evolab.app.race.view import RaceDialog
from evolab.engine.pool import WorkerPool
from evolab.engine.runsetup import create_run
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.policy.genome import layout_for
from evolab.runs import observe as ob
from evolab.runs.store import CHAMPIONS_DIR, STATS_FILE, RunStore
from evolab.spec.io import compile_env

LANDER = paths.builtin_env_dir() / "lander.evoenv"
LANDER_TEXT = LANDER.read_text(encoding="utf-8")
FRAME = ob.GridFrame(-10.0, -5.0, 0.8, 32, 24)


@pytest.fixture()
def lab(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LabView:
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path / "runs")
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    view = LabView(settings=settings)
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.show()
    qtbot.waitExposed(view)
    assert view.load_env(LANDER)
    view.popsize.setValue(6)
    view.trials.setValue(2)
    view.workers.setValue(1)
    view.dashboard.set_page("fun")
    try:
        yield view
    finally:
        view.client.abandon(timeout=60)  # 실패한 테스트도 학습 프로세스를 남기지 않게
        view.shutdown()


@pytest.fixture()
def message_boxes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    boxes: list[tuple[str, str]] = []

    def record(_parent: Any, title: str, text: str, *_a: Any, **_k: Any) -> Any:
        boxes.append((title, text))
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "information", staticmethod(record))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(record))
    return boxes


def spy_replays(monkeypatch: pytest.MonkeyPatch, view: LabView) -> list[tuple[Any, list[Any]]]:
    calls: list[tuple[Any, list[Any]]] = []
    real = view._run_replays

    def spy(tag: Any, jobs: list[Any]) -> None:
        calls.append((tag, jobs))
        real(tag, jobs)

    monkeypatch.setattr(view, "_run_replays", spy)
    return calls


def wait_state(qtbot: Any, view: LabView, state: str, timeout: int = 240000) -> None:
    qtbot.waitUntil(lambda: view.state == state, timeout=timeout)


def train_run(runs: Path, gens: int, **kw: Any) -> RunStore:
    """train 명령과 같은 실행 폴더를 이 프로세스에서(워커 0) 학습한다. 관찰 기록은 켜져 있다(저장소가 있음)."""
    config = TrainConfig(popsize=6, trials=2, run_seed=7, checkpoint_every=2, max_generations=gens, **kw)
    spec = compile_env(LANDER_TEXT)
    store = create_run(runs, LANDER_TEXT, spec, config, 0)
    with WorkerPool(0) as pool:
        t = Trainer(LANDER_TEXT, config, pool, store)
        reason = t.run()
    store.append_event("stop", t.state.generation, reason=reason)
    return store


def live_message(gen: int, population: np.ndarray, representative: int, row: dict[str, Any]) -> dict[str, Any]:
    """학습 프로세스의 generation 알림(관찰 기록·관전 궤적 없음)."""
    return {"row": row, "generation": gen, "population": population, "representative": representative,
            "scores": np.zeros(len(population)), "trial": (1, None), "trajectories": None, "champion": None,
            "env_version": None, "observation": None}


# ---------------------------------------------------------------------------
# T38: 실제 학습(관전) — 세대마다 관찰이 들어오고, 뇌 타일이 보이면 관전 중 대표 추적이 붙고, 그동안 사용자가 시작한
# 타임라인 재생은 지워지지 않는다(_replay_tag가 그대로)
# ---------------------------------------------------------------------------


def test_spectate_feeds_history_and_live_brain_without_cancelling_user_replay(
        qtbot: Any, lab: LabView, monkeypatch: pytest.MonkeyPatch) -> None:
    dash = lab.dashboard
    dash.set_tile_visible("genemap", False)  # 이 시험은 지도를 보지 않는다(테스트 프로세스의 축 맞추기를 건너뛴다)
    gate = threading.Event()
    brain_calls: list[int] = []
    real_replay = lab_view.replay

    def gated(spec: Any, layout: Any, genome: Any, trial: Any, every: int = 2, trace: Any = False) -> Any:
        if trace == "brain":
            brain_calls.append(len(brain_calls) + 1)
            if len(brain_calls) == 2:  # 둘째 뇌 추적(둘째 관전 세대)을 붙잡아 '도는 중'을 만든다
                gate.wait(60)
        return real_replay(spec, layout, genome, trial, every, trace)

    monkeypatch.setattr(lab_view, "replay", gated)
    assert dash.brain_visible()
    lab.rb_spectate.setChecked(True)
    lab.speed_combo.setCurrentIndex(2)  # 4배
    lab.max_generations.setValue(3)
    assert lab.start_new()
    # 첫 관전 세대: 대표를 시험 0번으로 따로 다시 돌린 추적이 붙는다(원천 ②)
    qtbot.waitUntil(lambda: dash.brain_source is not None and dash.brain_source.kind == "live"
                    and dash.brain_source.traced, timeout=120000)
    src = dash.brain_source
    assert src is not None and lab.playing is not None and src.generation == int(lab.playing["generation"])
    rep_i = int(lab.playing["representative"])
    assert np.array_equal(src.genome, np.asarray(lab.playing["population"][rep_i]))
    assert src.replay is not None and src.replay.raw is None  # 가벼운 추적
    # 관찰 기록이 세대 메시지로 들어온다
    entry = dash.history.entry(1)
    assert entry is not None and entry.obs is not None and dash.history.has_exact(1)
    # 대표 추적 = 워커와 같은 궤적: 캔버스가 재생 중인 대표 개체의 시험 0번 궤적과 비트 단위로 같다
    worker = lab.playing["trajectories"][rep_i]
    assert src.replay.trajectory.every == worker.every
    assert src.replay.trajectory.poses.tobytes() == worker.poses.tobytes()
    # 뇌 타일이 실제로 그린다(관전 제목)
    tile = dash.tile("brain")
    assert isinstance(tile, BrainTile)
    qtbot.waitUntil(lambda: tile.source_kind == "live", timeout=30000)

    # 둘째 관전 세대의 뇌 추적이 도는 중일 때 사용자가 타임라인 재생을 시작한다
    qtbot.waitUntil(lambda: len(brain_calls) >= 2 and lab.brain_busy, timeout=120000)
    tag_before = lab._replay_tag
    assert lab.play_generation(1)
    assert lab._replay_tag == tag_before + 1
    qtbot.waitUntil(lambda: lab._last_video is not None and lab._last_video[1] == S.LAB_LABEL_REP.format(gen=1),
                    timeout=60000)
    gate.set()
    qtbot.waitUntil(lambda: not lab.brain_busy, timeout=60000)
    qtbot.wait(50)
    assert lab._replay_tag == tag_before + 1  # 뇌 작업은 _replay_tag를 올리지 않는다
    assert lab.view_mode == "replay" and lab._last_video[1] == S.LAB_LABEL_REP.format(gen=1)
    src = dash.brain_source  # 뇌는 캔버스가 재생 중인 개체(①)
    assert src is not None and src.kind == "playing" and src.label == S.LAB_LABEL_REP.format(gen=1)
    assert src.traced and src.generation == 1
    wait_state(qtbot, lab, "finished")
    qtbot.waitUntil(lambda: len(dash.history) == 3, timeout=60000)
    assert all(dash.history.entry(g).obs is not None for g in (1, 2, 3))
    # 궤적 밀도 타일이 청한 세계 그림에 답했다
    dash.flush()
    obs3 = dash.observation(3)
    assert obs3 is not None and obs3.world_key is not None
    qtbot.waitUntil(lambda: dash.world_geometry(obs3.world_key) is not None, timeout=30000)


# ---------------------------------------------------------------------------
# 관전 중 보는 재미 그래프는 캔버스가 재생 중인 세대를 따른다(학습은 한 세대 앞서 계산한다). 클릭 검사한 개체는 다음
# 세대 재생이 시작되면 뇌 원천에서 빠진다
# ---------------------------------------------------------------------------


def test_spectating_fun_tiles_follow_the_played_generation_and_inspect_ends_with_it(
        qtbot: Any, lab: LabView, message_boxes: list[tuple[str, str]]) -> None:
    dash = lab.dashboard
    lab.rb_spectate.setChecked(True)
    lab.speed_combo.setCurrentIndex(0)  # 1배: 1세대를 보는 동안 2세대가 계산되어 온다
    lab.max_generations.setValue(3)
    assert lab.start_new()

    def ahead() -> bool:
        p = lab.playing
        return p is not None and dash.history.latest is not None and dash.history.latest > int(p["generation"])

    qtbot.waitUntil(ahead, timeout=240000)
    qtbot.waitUntil(lambda: dash.brain_source is not None and dash.brain_source.kind == "live", timeout=60000)
    dash.flush()
    p = int(lab.playing["generation"])
    assert lab.timeline.value() == lab.timeline.maximum() and dash.focus is None
    assert dash.follow == p and dash.view_generation() == p
    gm, dens = dash.tile("genemap"), dash.tile("density")
    assert gm.shown_generation == p
    assert dens.density_view is not None and dens.density_view.generation == p
    assert dens.density_view.world_key == dash.observation(p).world_key  # 관전 화면과 같은 세계
    assert dash.brain_source.generation == p
    assert lab.timeline_label.text() == S.LAB_TIMELINE_OF.format(gen=p, total=lab.timeline.maximum())

    # 재생 중인 세대의 개체를 클릭 검사하면 뇌는 그 개체(①), 다음 세대 재생이 시작되면 그 세대 대표(②)
    lab._inspect(2)
    qtbot.waitUntil(lambda: dash.brain_source is not None and dash.brain_source.kind == "playing", timeout=60000)
    assert dash.brain_source.label == S.GMAP_LABEL_INDIVIDUAL.format(gen=p, index=2)
    lab.speed_combo.setCurrentIndex(lab.speed_combo.count() - 1)
    qtbot.waitUntil(lambda: lab.playing is not None and int(lab.playing["generation"]) > p, timeout=240000)
    src = dash.brain_source
    assert src is not None and src.kind == "live" and src.generation == int(lab.playing["generation"])
    assert dash.follow == int(lab.playing["generation"])
    # 타임라인으로 다른 세대를 고르면 그쪽이 먼저(사용자가 고른 세대)
    lab.timeline.setValue(1)
    qtbot.waitUntil(lambda: dash.focus == 1, timeout=5000)
    assert dash.view_generation() == 1
    lab.client.send({"type": "pause"})
    assert message_boxes == []


def test_brain_tile_does_not_stay_preparing_after_spectating_on_the_learning_page(qtbot: Any,
                                                                                 lab: LabView) -> None:
    """뇌 타일이 안 보이는 동안(학습 그래프 쪽) 마지막 관전 세대 재생이 끝났으면, 나중에 「보는 재미」를 열어도
    「계산하는 중」에 머물지 않고 챔피언 멈춘 그림(③)을 보인다."""
    dash = lab.dashboard
    dash.set_tile_visible("genemap", False)
    dash.set_page("learning")
    assert not dash.brain_visible()
    lab.rb_spectate.setChecked(True)
    lab.speed_combo.setCurrentIndex(lab.speed_combo.count() - 1)
    lab.max_generations.setValue(2)
    assert lab.start_new()
    wait_state(qtbot, lab, "finished")
    qtbot.waitUntil(lambda: lab.playing is None and not lab.queue and not lab._scene_jobs and lab._last_played
                    is not None, timeout=120000)
    dash.set_page("fun")
    qtbot.wait(300)
    dash.flush()
    tile = dash.tile("brain")
    assert isinstance(tile, BrainTile)
    assert S.BRAIN_PREPARING not in tile._notes
    assert dash.brain_source is None and tile.source_kind in ("champion", "rep")


def test_failed_brain_trace_of_a_playing_replay_falls_back_instead_of_preparing(qtbot: Any, lab: LabView) -> None:
    """재생 중인 개체(①)의 뇌 추적이 실패하면 그 재생이 되풀이되는 동안에도 「계산하는 중」에 머물지 않는다."""
    dash = lab.dashboard
    dash.set_tile_visible("genemap", False)
    lab.rb_fast.setChecked(True)
    lab.fast_interval.setValue(1)
    lab.max_generations.setValue(2)
    assert lab.start_new()
    wait_state(qtbot, lab, "finished")
    qtbot.waitUntil(lambda: lab._brain_play is not None and not lab.brain_busy, timeout=60000)
    job = lab._brain_play
    job["rep"], job["tried"] = None, True
    lab._brain_tag += 1
    lab._brain_job = job
    lab._brain_done(("brain", lab._brain_tag, "play", job), RuntimeError("boom"))
    assert lab._brain_play is None and lab._brain_job is None
    src = dash.brain_source
    assert src is None or src.kind != "playing"
    dash.flush()
    tile = dash.tile("brain")
    assert isinstance(tile, BrainTile) and S.BRAIN_PREPARING not in tile._notes


# ---------------------------------------------------------------------------
# T38: 고속 모드 — 챔피언 재생 뒤 뇌가 「재생 중」이고 play_t에 따라 뉴런이 바뀐다. 타임라인·세대 넘겨 보기
# ---------------------------------------------------------------------------


def test_fast_mode_brain_timeline_focus_and_flip(qtbot: Any, lab: LabView) -> None:
    dash = lab.dashboard
    dash.set_tile_visible("genemap", False)
    lab.rb_fast.setChecked(True)
    lab.fast_interval.setValue(1)  # 세대마다 챔피언(또는 세대 대표)을 재생
    lab.max_generations.setValue(3)
    assert lab.start_new()
    wait_state(qtbot, lab, "finished")
    qtbot.waitUntil(lambda: dash.brain_source is not None and dash.brain_source.kind == "playing"
                    and dash.brain_source.traced and dash.brain_source.label == lab._replay_label, timeout=60000)
    src = dash.brain_source
    assert src is not None and src.label == lab._replay_label and lab.view_mode == "fast"
    tile = dash.tile("brain")
    assert isinstance(tile, BrainTile)
    dash.flush()
    assert tile.source_kind == "playing" and tile.brain.traced
    dt = tile.brain.dt
    qtbot.waitUntil(lambda: tile.brain.step_index() is not None and tile.brain.step_index() >= 10, timeout=30000)
    k1 = tile.brain.step_index()
    v1 = tile.brain.neuron_values()
    assert k1 == min(tile.brain.steps - 1, int(np.floor(dash.play_time / dt + 1e-9)))
    qtbot.waitUntil(lambda: tile.brain.step_index() != k1, timeout=30000)  # 재생 시각을 따라 뉴런이 바뀐다
    v2 = tile.brain.neuron_values()
    assert v1 is not None and v2 is not None and any(not np.array_equal(a, b) for a, b in zip(v1, v2, strict=True))
    assert tile.title_text.startswith(src.label)

    # 타임라인이 보는 세대를 정한다(맨 끝 = 최신 따라가기)
    assert lab.timeline.maximum() == 3 and dash.focus is None
    lab.timeline.setValue(2)
    assert lab.timeline_label.text() == S.LAB_TIMELINE_OF.format(gen=2, total=3)
    qtbot.waitUntil(lambda: dash.focus == 2, timeout=5000)
    assert dash.view_generation() == 2
    lab.timeline.setValue(3)
    qtbot.waitUntil(lambda: dash.focus is None, timeout=5000)

    # 세대 넘겨 보기: 맨 끝에서 누르면 1세대부터, 끝까지 가면 멈춘다(최신 따라가기)
    assert lab.btn_flip.isEnabled()
    lab.btn_flip.click()
    assert lab.flipping and dash.flipping and lab.timeline.value() == 1 and dash.focus == 1
    assert lab.flip_rate == 2.0 and lab.btn_flip.text() == S.LAB_FLIP_STOP_V2  # clamp(3 ÷ 20, 2, 15)
    seen: list[int | None] = []
    dash_set_focus = dash.set_focus

    def record(gen: int | None) -> None:
        seen.append(gen)
        dash_set_focus(gen)

    dash.set_focus = record  # type: ignore[method-assign]
    qtbot.waitUntil(lambda: not lab.flipping, timeout=10000)
    assert lab.timeline.value() == 3 and dash.focus is None and not dash.flipping
    assert 2 in seen and lab.btn_flip.text() == S.LAB_FLIP and not lab.btn_flip.isChecked()
    # 손으로 옮기면(키·휠·끌기) 멈춘다
    assert lab.start_flip()
    lab.timeline.triggerAction(QSlider.SliderAction.SliderSingleStepAdd)
    assert not lab.flipping and lab.timeline.value() == 2
    qtbot.waitUntil(lambda: dash.focus == 2, timeout=5000)
    assert lab.start_flip() and lab.flipping
    lab.timeline.sliderPressed.emit()
    assert not lab.flipping
    # 앞 세대 점을 누르면(지도의 focus_requested) 타임라인이 그 세대로
    dash.focus_requested.emit(1)
    assert lab.timeline.value() == 1 and dash.focus == 1


# ---------------------------------------------------------------------------
# T39: 관찰 기록 750세대짜리 실행을 열 때 이벤트 루프 틈
# ---------------------------------------------------------------------------


def fake_run(base: Path, gens: int) -> Path:
    """관찰 기록이 있는 세대 대표 파일 gens개와 그 통계 줄을 가진 실행 폴더(무작위 기록, D10 검사를 통과)."""
    spec = compile_env(LANDER_TEXT)
    lay = layout_for(spec)
    store = create_run(base, LANDER_TEXT, spec, TrainConfig(popsize=6, trials=2), 0)
    folder = store.root / CHAMPIONS_DIR
    folder.mkdir(exist_ok=True)
    rows: list[dict[str, str]] = []
    for g in range(1, gens + 1):
        obs, X = fake_generation(g, lay, pop=6, frame=FRAME, world_key=f"world{g:011d}")
        rep = obs.representative
        arrays = {"genome": X[rep], "score": np.asarray(float(obs.scores[rep])),
                  "env_version": np.asarray("0123456789abcdef"), **obs.to_arrays()}
        with open(folder / f"gen_{g:06d}_representative.npz", "wb") as f:  # fsync 없이(시험 준비를 빠르게)
            np.savez(f, allow_pickle=False, **arrays)
        rows.append({**stats_row(obs), "evaluations": str(g * 12), "elapsed": str(g * 1.5),
                     "champion_score": "", "champion_generation": "-1"})
    with open(store.root / STATS_FILE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    return store.root


def measure_open(run: Path, settings_dir: Path) -> dict[str, Any]:
    """실험실을 띄워 「보는 재미」 쪽에서 run을 열고, 기록을 다 읽고 펌프가 타일을 그릴 때까지의 이벤트 루프 틈을 잰다."""
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    view = LabView(settings=QSettings(str(settings_dir / "settings.ini"), QSettings.Format.IniFormat))
    try:
        view.resize(1280, 800)
        view.show()
        for _ in range(20):
            app.processEvents()
        assert view.load_env(LANDER)
        dash = view.dashboard
        dash.set_page("fun")
        gaps: list[float] = []
        clock = QElapsedTimer()
        probe = QTimer()
        probe.setInterval(10)
        probe.timeout.connect(lambda: gaps.append(clock.restart()))
        # 탐침은 열기 전에 켠다: 화면 스레드에서 도는 open_run(통계·사건 읽기, 화면 다시 만들기)도 틈에 들어간다
        clock.start()
        probe.start()
        warm = QElapsedTimer()
        warm.start()
        while warm.elapsed() < 100:
            QCoreApplication.processEvents()
        before = len(gaps)
        t0 = QElapsedTimer()
        t0.start()
        assert view.open_run(run)
        open_ms = t0.elapsed()
        assert dash.wait_history(60000)
        load_ms = t0.elapsed()
        settle = QElapsedTimer()
        settle.start()
        while settle.elapsed() < 800:  # 펌프가 보이는 타일을 하나씩 그린다
            QCoreApplication.processEvents()
        probe.stop()
        gaps = gaps[before:]  # 열기 직전 틈부터(열기를 덮는 틈 포함)
        return {"open_ms": open_ms, "load_ms": load_ms, "worst": max(gaps), "p99": float(np.percentile(gaps, 99)),
                "worst_after_open": max(gaps[1:]) if len(gaps) > 1 else 0.0,
                "n": len(gaps), "gens": len(dash.history), "observed": dash.history.load_summary.get("observed"),
                "maximum": view.timeline.maximum(), "focus": dash.focus, "pump": dash.pump_steps}
    finally:
        view.shutdown()
        view.deleteLater()


def test_opening_a_750_generation_run_keeps_the_event_loop_responsive(tmp_path: Path) -> None:
    """T39. 앱처럼 BLAS 스레드 1개인 새 프로세스에서 잰다(startup.limit_blas_threads). 이 시험 프로세스는 BLAS 스레드를
    묶지 않아, 부하가 큰 기계에서는 유전자 지도 축 맞추기(eigh)가 여러 스레드로 코어를 다투며 수백 ms씩 걸린다."""
    import json
    import os
    import subprocess
    import sys

    run = fake_run(tmp_path / "fake", 750)
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    out = subprocess.run([sys.executable, __file__, "measure-open", str(run), str(tmp_path)], env=env,
                         capture_output=True, text=True, timeout=300, check=False)
    assert out.returncode == 0, out.stderr[-4000:]
    m = json.loads(out.stdout.strip().splitlines()[-1])
    print(f"\n[lab] open 750-generation run: open_run {m['open_ms']} ms (synchronous), history loaded at "
          f"{m['load_ms']} ms, event-loop gap including the open: worst {m['worst']:.0f} ms, p99 {m['p99']:.0f} ms, "
          f"worst after the open {m['worst_after_open']:.0f} ms, n={m['n']}, pump steps {m['pump']}")
    assert m["gens"] == 750 and m["observed"] == 750 and m["maximum"] == 750 and m["focus"] is None
    # 기록은 배경에서 읽는다: 여는 동안에도 이벤트 루프가 돈다(세대 대표 파일 750개를 화면 스레드에서 읽으면 한 번에
    # 0.5~2초가 막힌다). 최대 틈에는 화면 스레드에서 도는 open_run도 들어간다. 다른 학습이 도는 이 공유 기계에서는 틈이
    # 크게 흔들려서 여기서는 큰 퇴행만 잡는다. 설계 T39·M8 목표(최대 틈 100ms 미만, p99 50ms 미만)는 한가한 PC에서
    # EVOLAB_STRICT_LATENCY=1로 잰다(마무리 단계).
    assert m["open_ms"] < 1000 and m["n"] >= 20
    assert m["worst"] < 1000
    if os.environ.get("EVOLAB_STRICT_LATENCY") == "1":
        assert m["worst"] < 100 and m["p99"] < 50, m


# ---------------------------------------------------------------------------
# T40: 되감기 뒤 기록을 다시 읽어 잘린 세대가 사라진다
# ---------------------------------------------------------------------------


def snapshot(store: RunStore) -> dict[str, Any]:
    """Session._snapshot과 같은 모양(되감기 알림 rewind의 내용)."""
    info = store.read_run_info()
    stats = store.read_stats()
    return {"run_dir": str(store.root), "run_info": info, "env_text": LANDER_TEXT, "env_version": None,
            "generation": len(stats), "config": info["config"], "workers": 1, "layout": info["layout"],
            "stats": stats, "events": store.read_events(), "champion": None, "spectate": False}


def test_rewind_reloads_history_without_the_cut_generations(qtbot: Any, lab: LabView, tmp_path: Path) -> None:
    store = train_run(tmp_path / "runs", 4)
    dash = lab.dashboard
    assert lab.open_run(store.root)
    assert dash.wait_history(30000)
    assert dash.history.gens == [1, 2, 3, 4] and all(dash.history.entry(g).obs is not None for g in (1, 2, 3, 4))
    # 학습 중처럼 5세대가 왔다(정확한 집단이 화면 메모리에 남는다)
    X5 = np.random.default_rng(5).normal(0.0, 0.3, (6, lab.layout.dim))
    lab._on_generation(live_message(5, X5, 2, {"generation": 5}))
    assert dash.history.has_exact(5) and dash.exact_genome(5, 2) is not None
    # 평가 오류로 2세대 체크포인트까지 되감겼다(Trainer.resume이 자르는 것과 같다)
    store.truncate_after(2)
    lab._on_rewound(snapshot(store))
    assert dash.wait_history(30000)
    assert dash.history.gens == [1, 2]
    assert dash.exact_genome(5, 2) is None and dash.observation(3) is None and dash.observation(2) is not None
    assert lab.timeline.maximum() == 2 and dash.focus is None and len(lab.rows) == 2


# ---------------------------------------------------------------------------
# T41: 지도 점 두 번 누르기(최근 세대는 정확한 유전자, 오래된 세대는 대표 + GMAP_INEXACT), 세대끼리 경주, 세계 그림
# ---------------------------------------------------------------------------


def test_gene_map_point_replay_race_and_world_geometry(qtbot: Any, lab: LabView, tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch,
                                                        message_boxes: list[tuple[str, str]]) -> None:
    store = train_run(tmp_path / "runs", 3)
    dash = lab.dashboard
    assert lab.open_run(store.root)
    assert dash.wait_history(30000)
    calls = spy_replays(monkeypatch, lab)
    rep2 = dash.rep_index(2)
    assert rep2 is not None
    rep2_genome = np.asarray(store.load_champion(2, "representative")["genome"], dtype=np.float64)

    # 다시 연 실행: 화면 메모리에 정확한 유전자가 없다 → 그 세대 대표를 대신(알림)
    other = (rep2 + 1) % 6
    dash.individual_requested.emit(2, other)
    tag, jobs = calls[-1]
    assert tag[0] == "individual" and np.array_equal(jobs[0][2], rep2_genome)
    assert lab.last_notice == S.GMAP_INEXACT.format(gen=2) and S.GMAP_INEXACT.format(gen=2) in lab.status.text()
    assert lab._replay_label == S.LAB_LABEL_REP.format(gen=2) and lab.view_mode == "replay"
    dash.individual_requested.emit(2, other)  # 알림은 한 자리: 다시 눌러도 쌓이지 않는다
    assert lab.status.text().count(S.GMAP_INEXACT.format(gen=2)) == 1
    qtbot.waitUntil(lambda: lab._last_video is not None and lab._last_video[1] == S.LAB_LABEL_REP.format(gen=2),
                    timeout=60000)
    # 대표 점 자신(정확함): 알림 없음
    dash.individual_requested.emit(2, rep2)
    tag, jobs = calls[-1]
    assert np.array_equal(jobs[0][2], rep2_genome) and lab.last_notice == ""
    assert S.GMAP_INEXACT.format(gen=2) not in lab.status.text()  # 정확한 재생이면 앞 알림을 지운다
    assert lab._replay_label == S.GMAP_LABEL_INDIVIDUAL.format(gen=2, index=rep2)
    dash.individual_requested.emit(3, (dash.rep_index(3) + 1) % 6)
    assert lab.status.text().endswith("\n" + S.GMAP_INEXACT.format(gen=3))
    lab.back_to_live()  # 실시간 보기로 돌아가면 재생 알림도 지운다
    assert S.GMAP_INEXACT.format(gen=3) not in lab.status.text() and lab.last_notice == ""
    # 집단 기록이 없는 세대의 대표 점(-1)
    dash.individual_requested.emit(1, -1)
    tag, jobs = calls[-1]
    rep1_genome = np.asarray(store.load_champion(1, "representative")["genome"], dtype=np.float64)
    assert np.array_equal(jobs[0][2], rep1_genome) and lab.last_notice == ""

    # 세대끼리 경주(세대 대표 2세대 이상): 칸 6개 이하의 경주 창
    assert lab.btn_race.isEnabled()
    dlg = lab.race_generations()
    assert isinstance(dlg, RaceDialog) and dlg.isVisible()
    assert dlg.windowTitle() == S.RACE_TITLE_GENERATIONS.format(run=store.root.name)
    assert 2 <= len(dlg.view.lanes) <= 6
    assert [e.generation for e in dlg.view.entries][:3] == [1, 2, 3]
    assert lab.race_generations() is dlg  # 다시 누르면 같은 창
    dlg.close()

    # 궤적 밀도 타일이 청한 세계 그림(시험 0번 시드·트랙, 그 세대 환경)에 답한다
    obs2 = dash.observation(2)
    assert obs2 is not None and obs2.world_key is not None
    obs1 = dash.observation(1)
    assert obs1 is not None and obs1.world_key is not None and obs1.world_key != obs2.world_key  # 착륙선: 세대마다 지형
    lab._geometry_needed(2, obs2.world_key)
    lab._geometry_needed(1, obs1.world_key)  # 만드는 동안 온 청도 답한다
    assert lab.world_busy
    qtbot.waitUntil(lambda: not lab.world_busy, timeout=30000)
    geometry = dash.world_geometry(obs2.world_key)
    assert geometry is not None and geometry.bodies and dash.world_geometry(obs1.world_key) is not None

    # 학습 중 받은 세대(최근 64세대): 정확한 유전자로 재생, 알림 없음
    X4 = np.random.default_rng(4).normal(0.0, 0.3, (6, lab.layout.dim))
    lab._on_generation(live_message(4, X4, 0, {"generation": 4}))
    dash.individual_requested.emit(4, 3)
    tag, jobs = calls[-1]
    assert np.array_equal(jobs[0][2], X4[3]) and lab.last_notice == ""
    assert lab._replay_label == S.GMAP_LABEL_INDIVIDUAL.format(gen=4, index=3)
    qtbot.waitUntil(lambda: lab._last_video is not None
                    and lab._last_video[1] == S.GMAP_LABEL_INDIVIDUAL.format(gen=4, index=3), timeout=60000)
    # 뇌 그림이 보이면 지도 점 재생이 뇌 원천 ①
    src = dash.brain_source
    assert src is not None and src.kind == "playing" and src.label == S.GMAP_LABEL_INDIVIDUAL.format(gen=4, index=3)
    assert np.array_equal(src.genome, X4[3]) and src.traced
    assert message_boxes == []


def test_generation_race_needs_two_generations(qtbot: Any, lab: LabView,
                                               message_boxes: list[tuple[str, str]]) -> None:
    assert not lab.btn_race.isEnabled() and not lab.btn_flip.isEnabled()
    assert lab.race_generations() is None
    assert message_boxes == [(S.LAB_RACE_GENERATIONS, S.RACE_NEED_GENS)]
    assert lab.timeline_label.text() == S.LAB_TIMELINE_OF.format(gen=0, total=0)
    assert not lab.start_flip() and not lab.flipping


def test_brain_source_priority_and_hidden_brain(qtbot: Any, lab: LabView, tmp_path: Path) -> None:
    """D24 우선순위: 캔버스 재생(①)이 관전 대표(②)보다 앞, 뇌가 안 보이면 추적하지 않는다, 늦게 보이면 따로 추적."""
    store = train_run(tmp_path / "runs", 2)
    dash = lab.dashboard
    assert lab.open_run(store.root)
    assert dash.wait_history(30000)
    dash.set_page("learning")  # 뇌가 안 보인다: 재생은 추적 없이
    assert not lab._brain_wanted()
    assert lab.play_generation(2)
    qtbot.waitUntil(lambda: dash.brain_source is not None and dash.brain_source.kind == "playing", timeout=60000)
    assert not dash.brain_source.traced and lab._brain_play is not None and not lab._brain_play["tried"]
    tag = lab._replay_tag
    dash.set_page("fun")  # 늦게 보이면(_tick) 같은 개체를 따로 가볍게 추적한다(_replay_tag는 그대로)
    qtbot.waitUntil(lambda: dash.brain_source is not None and dash.brain_source.traced, timeout=60000)
    assert lab._replay_tag == tag and dash.brain_source.label == S.LAB_LABEL_REP.format(gen=2)
    # '실시간 보기'로 돌아가면 ①이 없어지고, 관전 중이 아니므로 원천이 없다(챔피언, 그때 챔피언이 없으면 보는 세대
    # 대표의 멈춘 그림)
    lab.back_to_live()
    assert dash.brain_source is None
    dash.flush()
    tile = dash.tile("brain")
    assert isinstance(tile, BrainTile) and tile.source_kind in ("champion", "rep")


def test_opening_another_run_clears_the_canvas_hud(qtbot: Any, lab: LabView, tmp_path: Path) -> None:
    """다른 실행을 열면 앞 실행의 HUD 글(「n세대 · 시험 0번」)이 남지 않는다."""
    store = train_run(tmp_path / "runs", 2)
    lab.canvas.set_hud(S.LAB_HUD_LIVE.format(gen=6, t=1.0, speed=1))
    assert lab.open_run(store.root)
    assert lab.canvas.hud == ""


def test_opened_run_draws_the_still_scene_on_a_later_turn_only_for_the_current_run(
        qtbot: Any, lab: LabView, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T39 고침: open_run은 전시 시드 세계(Episode·정적 그림)를 다음 이벤트 루프 차례에 그린다. 그사이 다시 열거나
    (다른 실행·되감기) 다른 장면이 캔버스를 차지하면 앞 요청은 그리지 않는다."""
    from evolab.world.physics2d.episode import Episode

    store = train_run(tmp_path / "runs", 2)
    drawn: list[Any] = []
    real = lab._show_still_scene
    monkeypatch.setattr(lab, "_show_still_scene", lambda: (drawn.append(lab.run_dir), real())[1])
    lab.canvas.world = None
    assert lab.open_run(store.root)
    assert drawn == [] and lab.canvas.world is None  # open_run 안에서는 그리지 않는다
    assert lab.open_run(store.root)  # 다시 열기: 앞 요청은 버린다
    qtbot.waitUntil(lambda: bool(drawn))
    qtbot.wait(30)
    assert drawn == [store.root] and lab.canvas.world is not None
    assert lab.open_run(store.root)
    other = Episode(lab.spec, None, lab_view.seeds.display_trial()).geometry()
    lab.canvas.set_scene(other)  # 다음 차례 전에 다른 장면(끝난 재생 결과 등)이 캔버스를 차지했다
    qtbot.wait(50)
    assert drawn == [store.root] and lab.canvas.world is other
    assert lab.open_run(store.root)  # 다시 열면 그 실행의 전시 세계로
    qtbot.waitUntil(lambda: len(drawn) == 2)
    assert lab.canvas.world is not other


def test_lab_freezes_long_lived_objects_until_shutdown(qtbot: Any, lab: LabView) -> None:
    """T39·M4b 고침: 실험실을 만들면 그때 살아 있는 객체를 순환 GC 밖으로 옮기고(전체 수집이 짧게), 닫을 때 되돌린다.
    지금 플랫폼 정보는 처음 한 번만 계산한다(실행을 열 때마다 uname·WMI를 부르지 않게)."""
    import gc

    assert gc.get_freeze_count() > 0
    assert lab_view.platform_now() is lab_view.platform_now()
    lab.shutdown()
    assert gc.get_freeze_count() == 0


def test_observation_summary_failure_is_a_status_notice_not_a_box(
        qtbot: Any, lab: LabView, tmp_path: Path, message_boxes: list[tuple[str, str]]) -> None:
    """D9: 관찰 기록 요약 실패(notice가 붙은 error)는 학습을 멈추지 않으므로 막는 상자 대신 상태 글 끝줄로, 세대가 와서
    상태 글을 다시 만들어도, 되감아도 남고, 다른 실행을 열면 지운다. 다른 오류는 지금처럼 상자."""
    store = train_run(tmp_path / "runs", 2)
    assert lab.open_run(store.root)
    note = S.TRAIN_OBSERVE_FAILED_ONWARD.format(reason="boom")
    lab.client._dispatch({"type": "error", "fatal": False, "notice": True, "message": note})
    assert message_boxes == [] and lab.status.text().endswith("\n" + note)
    lab._update_status()
    assert lab.status.text().endswith("\n" + note)
    lab.client._dispatch({"type": "error", "fatal": False, "message": "other"})
    # v3.0(설계 D26·3.2): 다른 오류는 지금처럼 상자지만 show_error 상자(다음 할 일 + 오류 보고 묶음 단추)다
    boxes = [b for b in errors.open_error_boxes() if b.parent() is lab]
    assert message_boxes == [] and [(b.windowTitle(), b.friendly.message) for b in boxes] == [(S.LAB_ERROR_TITLE, "other")]
    lab._on_rewound(snapshot(store))  # 같은 학습 프로세스 안의 되감기: 알림은 한 번만 오므로 남긴다
    assert lab.status.text().endswith("\n" + note)
    assert lab.open_run(store.root)
    assert note not in lab.status.text()


# ---------------------------------------------------------------------------
# 프로그램 끝내기: 기록을 읽는 중이거나 경주를 준비하는 중에 닫아도 프로세스가 abort하지 않는다
# ---------------------------------------------------------------------------


def quit_while_busy(run: Path, settings_dir: Path) -> dict[str, Any]:
    """실험실에서 run을 열어 배경 읽기가 도는 동안 세대끼리 경주 창을 열고(준비 중), 주 창을 닫을 때처럼 shutdown한 뒤
    이벤트 루프를 끝낸다. 읽기·준비를 느리게 해 닫는 순간 둘 다 돌고 있게 한다. 이 프로세스가 정상 종료(0)해야 한다:
    돌던 QThread가 프로그램이 끝날 때 지워지면 Qt가 abort한다(종료 코드 134)."""
    import time

    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    from evolab.app.lab import history as hs
    from evolab.app.race import view as race_view

    app = QApplication.instance() or QApplication([])
    real_load = hs.load_generation
    real_replay = race_view.replay

    def slow_load(*a: Any, **k: Any) -> Any:
        time.sleep(0.4)
        return real_load(*a, **k)

    def slow_replay(*a: Any, **k: Any) -> Any:
        time.sleep(0.3)
        return real_replay(*a, **k)

    hs.load_generation = slow_load
    race_view.replay = slow_replay
    view = LabView(settings=QSettings(str(settings_dir / "settings.ini"), QSettings.Format.IniFormat))
    view.resize(1280, 800)
    view.show()
    for _ in range(10):
        app.processEvents()
    assert view.load_env(LANDER)
    view.dashboard.set_page("fun")
    assert view.open_run(run)
    dlg = view.race_generations()
    assert dlg is not None
    clock = QElapsedTimer()
    clock.start()
    while clock.elapsed() < 150:
        QCoreApplication.processEvents()
    busy = {"loading": view.dashboard.history_loading is not None, "racing": bool(race_view._LIVE_THREADS)}
    view.shutdown()  # 주 창 closeEvent와 같다(경주 창은 닫히지 않고 남는다)
    view.hide()
    QTimer.singleShot(0, app.quit)
    app.exec()
    return busy


def test_quitting_while_loading_history_or_preparing_a_race_exits_cleanly(tmp_path: Path) -> None:
    import json
    import os
    import subprocess
    import sys

    store = train_run(tmp_path / "runs", 3)
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    out = subprocess.run([sys.executable, __file__, "quit-while-busy", str(store.root), str(tmp_path)], env=env,
                         capture_output=True, text=True, timeout=300, check=False)
    assert out.returncode == 0, (out.returncode, out.stderr[-4000:])
    busy = json.loads(out.stdout.strip().splitlines()[-1])
    assert busy == {"loading": True, "racing": True}  # 닫는 순간 둘 다 돌고 있었다
    assert "QThread: Destroyed" not in out.stderr


if __name__ == "__main__":  # 측정·종료 시험의 자식 프로세스
    import json
    import sys

    if len(sys.argv) == 4 and sys.argv[1] == "measure-open":
        print(json.dumps(measure_open(Path(sys.argv[2]), Path(sys.argv[3]))))
    elif len(sys.argv) == 4 and sys.argv[1] == "quit-while-busy":
        print(json.dumps(quit_while_busy(Path(sys.argv[2]), Path(sys.argv[3]))), flush=True)

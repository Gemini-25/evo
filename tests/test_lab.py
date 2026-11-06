"""M3: 실험실 화면 전체(학습 프로세스 + 관전 재생 + 대시보드 + 타임라인 + 클릭 검사 + 입력 지연)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QElapsedTimer, QSettings, QTimer

from evolab import paths
from evolab.app.lab.view import LabView
from evolab.runs.store import RunStore

LANDER = paths.builtin_env_dir() / "lander.evoenv"


@pytest.fixture()
def lab(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LabView:
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path / "runs")
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    view = LabView(settings=settings)
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.show()
    assert view.load_env(LANDER)
    view.popsize.setValue(6)
    view.trials.setValue(2)
    view.workers.setValue(1)
    try:
        yield view
    finally:
        view.client.abandon(timeout=60)  # 실패한 테스트도 학습 프로세스를 남기지 않게
        view.shutdown()


def wait_state(qtbot: Any, view: LabView, state: str, timeout: int = 120000) -> None:
    qtbot.waitUntil(lambda: view.state == state, timeout=timeout)


def test_fast_mode_run_timeline_and_versus(qtbot: Any, lab: LabView) -> None:
    lab.rb_fast.setChecked(True)
    lab.max_generations.setValue(4)
    assert lab.start_new()
    wait_state(qtbot, lab, "finished")
    assert len(lab.rows) == 4 and lab.timeline.maximum() == 4
    run_dir = lab.run_dir
    assert run_dir is not None and len(RunStore.open(run_dir).read_stats()) == 4
    assert lab.play_generation(2)
    qtbot.waitUntil(lambda: lab.canvas.duration() > 0 and lab.view_mode == "replay", timeout=60000)
    assert lab.play_versus()
    qtbot.waitUntil(lambda: lab._timer.isActive(), timeout=60000)
    lab.back_to_live()
    lab.end_session()
    assert not lab.client.active
    # 기록 화면에서 연 것처럼 다시 열기
    assert lab.open_run(run_dir)
    assert len(lab.rows) == 4 and lab.btn_continue.isEnabled()


def test_spectate_plays_every_generation_and_stays_responsive(qtbot: Any, lab: LabView) -> None:
    lab.rb_spectate.setChecked(True)
    lab.speed_combo.setCurrentIndex(3)  # 8배
    lab.max_generations.setValue(3)
    gaps: list[float] = []
    clock = QElapsedTimer()
    clock.start()
    probe = QTimer()
    probe.setInterval(10)

    def tick() -> None:
        gaps.append(clock.restart())

    probe.timeout.connect(tick)
    probe.start()
    assert lab.start_new()
    qtbot.waitUntil(lambda: lab.playing is not None, timeout=120000)
    # 재생 중인 개체를 클릭해 들여다보기
    lab._inspect(0)
    qtbot.waitUntil(lambda: lab.inspect.isVisible(), timeout=60000)
    assert lab.inspect.replay is not None and lab.inspect.replay.observations is not None
    lab._close_inspect()
    wait_state(qtbot, lab, "finished", timeout=240000)
    qtbot.waitUntil(lambda: lab.playing is None and not lab.queue, timeout=120000)
    probe.stop()
    assert [int(float(r["generation"])) for r in lab.rows] == [1, 2, 3]
    worst = max(gaps[5:]) if len(gaps) > 5 else 0.0
    print(f"\n[lab] event-loop gap while training+spectating: worst {worst:.0f} ms, n={len(gaps)}")
    assert worst < 250  # 명세 목표 100ms는 실제 PC에서 따로 잰다(CI 러너는 느리고 들쭉날쭉하다)


def test_pause_apply_and_continue(qtbot: Any, lab: LabView) -> None:
    lab.rb_fast.setChecked(True)
    lab.max_generations.setValue(50)
    assert lab.start_new()
    qtbot.waitUntil(lambda: len(lab.rows) >= 1, timeout=120000)
    lab.btn_pause.click()
    wait_state(qtbot, lab, "paused")
    assert lab.btn_apply.isEnabled() and lab.btn_step.isEnabled()
    n = len(lab.rows)
    lab.max_generations.setValue(n + 1)
    lab.apply_settings()
    qtbot.waitUntil(lambda: any(e.get("kind") == "config_change" for e in lab.events), timeout=30000)
    lab.btn_continue.click()
    wait_state(qtbot, lab, "finished")
    assert len(lab.rows) == n + 1
    t0 = time.monotonic()
    lab.end_session()
    assert time.monotonic() - t0 < 30

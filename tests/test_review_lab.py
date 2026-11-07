"""검토 수정(실험실 화면, 영상 내보내기, 주 창 연결)의 회귀 테스트."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox, QProgressDialog

from evolab import paths
from evolab.app import strings as S
from evolab.app.lab import view as lab_view
from evolab.app.lab.view import COLOR_NOW, SPEEDS, LabView
from evolab.engine import seeds
from evolab.engine.pool import WorkerPool, default_workers
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.runs.catalog import clone_settings
from evolab.runs.store import RunStore
from evolab.spec.io import compile_env, env_hash

LANDER = paths.builtin_env_dir() / "lander.evoenv"
LANDER_TEXT = LANDER.read_text(encoding="utf-8")
TRACK = paths.builtin_env_dir() / "track_car.evoenv"
HILL = paths.builtin_env_dir() / "hill_car.evoenv"


def lander_variant() -> str:
    """같은 유전자 배치에 중력만 다른 환경(에디터에서 고친 환경 흉내)."""
    data = json.loads(LANDER_TEXT)
    data["world"]["gravity"] = [0.0, -6.0]
    return json.dumps(data, ensure_ascii=False, indent=2)


def new_run(runs: Path, name: str, config: TrainConfig) -> RunStore:
    """train 명령과 같은 run.json으로 실행을 만들고 학습 프로세스 없이(워커 0) 학습한다."""
    store = RunStore.create(runs, name, {"program": "evolab", "created": "2026-11-07T15:30:00", "env_name": "달 착륙선",
                                         "method": config.method, "config": config.to_dict()})
    store.add_env_version(LANDER_TEXT, 0)
    store.append_event("start", 0, workers=0)
    with WorkerPool(0) as pool:
        t = Trainer(LANDER_TEXT, config, pool, store)
        reason = t.run()
    store.append_event("stop", t.state.generation, reason=reason)
    return store


def small_config(gens: int, **kw: Any) -> TrainConfig:
    return TrainConfig(popsize=6, trials=2, run_seed=7, checkpoint_every=2, max_generations=gens, **kw)


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


@pytest.fixture()
def message_boxes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """QMessageBox.information/warning을 띄우지 않고 (제목, 내용)으로 모은다."""
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


# ---------------------------------------------------------------------------
# 0. 관전 중 '실시간 보기'를 눌러도 재생과 학습이 멈추지 않는다
# ---------------------------------------------------------------------------


def test_back_to_live_while_spectating_keeps_playing(qtbot: Any, lab: LabView) -> None:
    lab.rb_spectate.setChecked(True)
    lab.speed_combo.setCurrentIndex(3)  # 8배
    lab.max_generations.setValue(3)
    assert lab.start_new()
    qtbot.waitUntil(lambda: lab.playing is not None, timeout=120000)
    lab.btn_live.click()
    assert lab._timer.isActive()  # 보던 세대를 끝까지 재생해 끝 알림(played)을 보낸다
    wait_state(qtbot, lab, "finished")
    qtbot.waitUntil(lambda: lab.playing is None and not lab.queue, timeout=120000)
    assert [int(float(r["generation"])) for r in lab.rows] == [1, 2, 3]


# ---------------------------------------------------------------------------
# 1, 17. 다른 곳(반복 실행)에서 학습 중인 실행은 실험실에서 이어 가지 않는다
# ---------------------------------------------------------------------------


def test_lab_refuses_to_resume_run_busy_elsewhere(qtbot: Any, lab: LabView, tmp_path: Path,
                                                   message_boxes: list[tuple[str, str]]) -> None:
    run = new_run(tmp_path / "runs", "busy", small_config(2)).root
    assert lab.open_run(run)
    assert lab.btn_continue.isEnabled()
    lab.run_busy = lambda p: p.resolve() == run.resolve()
    lab.busy_runs_changed()
    assert not lab.btn_continue.isEnabled() and lab.btn_continue.toolTip() == S.RVL_CONTINUE_BUSY_TIP
    lab.continue_run()  # 단추가 꺼지기 전에 눌렸어도
    assert not lab.client.active and lab.state == "closed"
    assert message_boxes == [(S.APP_NAME, S.RVL_RESUME_BUSY)]
    assert not lab.resume_run(run, 2)
    assert not lab.client.active and len(message_boxes) == 2
    assert lab.rows and lab.run_dir == run  # 보던 실행은 그대로
    lab.run_busy = lambda _p: False
    lab.busy_runs_changed()
    assert lab.btn_continue.isEnabled() and lab.btn_continue.toolTip() == ""


def test_main_window_blocks_lab_continue_of_repeat_run(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       message_boxes: list[tuple[str, str]]) -> None:
    from evolab.app.main import MainWindow

    monkeypatch.setattr(paths, "user_state_dir", lambda: tmp_path / "state")
    paths.set_custom_data_dir(tmp_path / "data")
    run = new_run(paths.runs_dir(), "repeat", small_config(2)).root
    window = MainWindow()
    qtbot.addWidget(window)
    window.compare.repeat_run_started.emit(run)  # 비교 화면의 반복 실행이 이 실행을 학습하기 시작했다
    assert window.run_is_busy(run) and window.records.is_busy(run)
    window.records.open_run(run)  # 기록 > 열기(보기만은 된다)
    assert window.tabs.currentWidget() is window.lab and window.lab.run_dir == run
    assert not window.lab.btn_continue.isEnabled()
    window.lab.continue_run()
    assert not window.lab.client.active
    assert message_boxes == [(S.APP_NAME, S.RVL_RESUME_BUSY)]
    window._resume_run(run, 2)  # 기록 > 이어하기에서 온 요청도 실험실이 막는다
    assert not window.lab.client.active and len(message_boxes) == 2
    window.compare.repeat_finished.emit(True)  # 반복 실행이 끝나면 다시 이어 갈 수 있다
    assert not window.run_is_busy(run) and window.lab.btn_continue.isEnabled()


# ---------------------------------------------------------------------------
# 2. 앞 실행의 재생 상태(고속 모드 후보, 저장할 영상)가 새 실행으로 넘어가지 않는다
# ---------------------------------------------------------------------------


def test_previous_run_state_does_not_leak(qtbot: Any, lab: LabView, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          message_boxes: list[tuple[str, str]]) -> None:
    calls = spy_replays(monkeypatch, lab)
    stale_genome = np.full(lab.layout.dim, 0.25)
    stale = {"generation": 9, "representative": 0, "population": [stale_genome]}

    # 다른 환경을 불러온 뒤 고속 모드로 시작하면 앞 실행 유전자를 재생하지 않는다
    lab.last_fast_candidate = stale
    lab._last_video = ([], "stale")
    assert lab.load_env(TRACK)
    assert lab.last_fast_candidate is None and lab._last_video is None and lab._fast_last == 0
    lab.rb_fast.setChecked(True)
    lab._maybe_fast_replay(force=True)
    assert calls == []
    assert lab.save_video(tmp_path / "x.gif") is None
    assert message_boxes == [(S.VIDEO_SAVE_TITLE, S.VIDEO_NOTHING)]

    # 기록에서 연 실행(챔피언 없음)의 '1세대 vs 지금'은 그 실행의 대표를 쓴다
    store = new_run(tmp_path / "runs", "opened", small_config(3))
    assert lab.load_env(LANDER)
    lab.last_fast_candidate = stale
    assert lab.open_run(store.root)
    assert lab.champion is None and lab.last_fast_candidate is None
    assert lab.play_versus()
    (_tag, jobs) = calls[-1]
    own = np.asarray(store.load_champion(3, "representative")["genome"], dtype=np.float64)
    assert np.array_equal(jobs[1][2], own) and not np.array_equal(jobs[1][2], stale_genome)


# ---------------------------------------------------------------------------
# 3. 영상 내보내기를 취소하면 파일을 쓰지 않는다
# ---------------------------------------------------------------------------


def _lander_replay() -> Any:
    from evolab.engine.replay import replay
    from evolab.policy.genome import layout_for

    spec = compile_env(LANDER_TEXT)
    layout = layout_for(spec)
    return replay(spec, layout, layout.initial(), seeds.display_trial())


def test_cancelled_video_export_writes_nothing(qtbot: Any, tmp_path: Path) -> None:
    from evolab.app.export.video import export_gif

    rep = _lander_replay()
    items = [(rep, COLOR_NOW, "x")]
    path = tmp_path / "a.gif"
    path.write_bytes(b"ORIGINAL")
    seen: list[int] = []

    def cancel_at_15(done: int, _total: int) -> bool:
        seen.append(done)
        return done < 15

    assert export_gif(items, path, progress=cancel_at_15) is None
    assert seen[-1] == 15 and path.read_bytes() == b"ORIGINAL"
    assert not path.with_name(path.name + ".tmp").exists()
    total: list[int] = []
    out = export_gif(items, path, progress=lambda done, n: total.append(n) or True)
    assert out == path and path.read_bytes()[:3] == b"GIF"


def test_save_video_cancel_keeps_existing_file(qtbot: Any, lab: LabView, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    rep = _lander_replay()
    lab._last_video = ([(rep, COLOR_NOW, "x")], "label")
    path = tmp_path / "b.gif"
    path.write_bytes(b"ORIGINAL")
    asked = {"n": 0}

    def canceled_after_40(_self: Any) -> bool:
        asked["n"] += 1
        return asked["n"] >= 40  # 사용자가 40번째 프레임쯤 '취소'를 눌렀다

    monkeypatch.setattr(QProgressDialog, "wasCanceled", canceled_after_40)
    before = lab.status.text()
    assert lab.save_video(path) is None
    assert path.read_bytes() == b"ORIGINAL" and lab.status.text() == before


# ---------------------------------------------------------------------------
# 4. 학습 중 타임라인·겹쳐 보기, 재생이 끝난 세대 클릭 검사, 같은 환경 다시 불러오기 표시
# ---------------------------------------------------------------------------


def test_spectate_buttons_inspect_after_playback_and_reload_marks(qtbot: Any, lab: LabView) -> None:
    lab.rb_spectate.setChecked(True)
    lab.speed_combo.setCurrentIndex(3)  # 8배
    lab.max_generations.setValue(2)
    assert lab.start_new()
    qtbot.waitUntil(lambda: len(lab.rows) >= 1, timeout=120000)
    assert lab.state == "running"  # 관전 모드: 2세대는 1세대 재생이 끝나야 끝난다
    assert lab.btn_timeline.isEnabled() and lab.btn_versus.isEnabled()
    wait_state(qtbot, lab, "finished")
    qtbot.waitUntil(lambda: lab.playing is None and not lab.queue and not lab._timer.isActive(), timeout=120000)
    # 재생이 끝나 화면에 남은 개체를 클릭해도 들여다볼 수 있다
    lab.canvas.individual_clicked.emit(1)
    qtbot.waitUntil(lambda: lab.inspect.isVisible(), timeout=60000)
    assert lab._inspect_index == 1 and lab.inspect.replay is not None and lab.inspect.replay.observations is not None
    assert lab._timer.isActive()
    lab._close_inspect()
    # 같은 파일을 다시 불러오면 학습 프로세스는 기록하지 않으므로 그래프 표시도 없다
    with qtbot.waitSignal(lab.client.applied, timeout=60000):
        lab.reload_env()
    assert not any(e.get("kind") == "env_change" for e in lab.events)
    with qtbot.waitSignal(lab.client.applied, timeout=60000):
        assert lab.send_env_text(lander_variant())
    assert [e for e in lab.events if e.get("kind") == "env_change"] == [{"kind": "env_change", "generation": 2}]
    assert [e["kind"] for e in RunStore.open(lab.run_dir).read_events()].count("env_change") == 1


# ---------------------------------------------------------------------------
# 5. 몸 유전자 환경(언덕 자동차) 관전 장면은 화면 스레드 밖에서 만든다
# ---------------------------------------------------------------------------


def test_body_gene_spectate_scene_is_built_off_the_gui_thread(qtbot: Any, lab: LabView,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    on_main: list[bool] = []
    real = lab_view.build_scene

    def spy(*args: Any) -> Any:
        on_main.append(threading.current_thread() is threading.main_thread())
        return real(*args)

    monkeypatch.setattr(lab_view, "build_scene", spy)
    played: list[int] = []
    send = lab.client.send

    def spy_send(msg: dict[str, Any]) -> bool:
        if msg.get("type") == "played":
            played.append(int(msg["generation"]))
        return send(msg)

    monkeypatch.setattr(lab.client, "send", spy_send)
    assert lab.load_env(HILL)
    assert lab.layout is not None and lab.layout.n_body > 0
    lab.popsize.setValue(6)
    lab.trials.setValue(1)
    lab.rb_spectate.setChecked(True)
    lab.speed_combo.setCurrentIndex(3)
    lab.max_generations.setValue(2)
    assert lab.start_new()
    qtbot.waitUntil(lambda: lab.playing is not None, timeout=240000)
    assert lab.canvas._scene_agents is not None and len(lab.canvas._scene_agents) == 6  # 개체마다 몸 모양
    wait_state(qtbot, lab, "finished", timeout=480000)
    qtbot.waitUntil(lambda: lab.playing is None and not lab.queue and not lab._scene_jobs, timeout=240000)
    assert played == [1, 2]  # 세대 순서대로 다 재생했다
    assert on_main and not any(on_main)


# ---------------------------------------------------------------------------
# 18. 실험실 보기 설정(모드, 배속, 챔피언 재생 간격, 워커 수)이 다시 켜도 남는다
# ---------------------------------------------------------------------------


def test_lab_settings_survive_restart(qtbot: Any, tmp_path: Path) -> None:
    ini = tmp_path / "settings.ini"
    workers = min(256, default_workers() + 1)
    s1 = QSettings(str(ini), QSettings.Format.IniFormat)
    v1 = LabView(settings=s1)
    qtbot.addWidget(v1)
    v1.workers.setValue(workers)
    v1.rb_fast.setChecked(True)
    v1.speed_combo.setCurrentIndex(SPEEDS.index(4))
    v1.fast_interval.setValue(25)  # 간격만 바꿔도 저장된다
    s1.sync()
    s2 = QSettings(str(ini), QSettings.Format.IniFormat)
    v2 = LabView(settings=s2)
    qtbot.addWidget(v2)
    assert v2.workers.value() == workers and v2.speed == 4 and v2.speed_combo.currentData() == 4
    assert v2.rb_fast.isChecked() and v2.view_mode == "fast" and v2.fast_interval.value() == 25
    s2.sync()
    s3 = QSettings(str(ini), QSettings.Format.IniFormat)
    assert (int(s3.value("lab/speed")), str(s3.value("lab/mode")), int(s3.value("lab/fast_interval")),
            int(s3.value("lab/workers"))) == (4, "fast", 25, workers)


# ---------------------------------------------------------------------------
# 23. 고속 모드 챔피언 재생은 챔피언을 뽑은 세대의 환경 버전으로 한다
# ---------------------------------------------------------------------------


def test_fast_replay_uses_champion_env_version(qtbot: Any, lab: LabView, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    store = new_run(tmp_path / "runs", "edited", small_config(4, validate_every=24))
    champ_gen = store.champion_generations("champion")[-1]
    assert champ_gen <= 4
    store.add_env_version(lander_variant(), 4)  # 4세대에서 멈추고 환경을 고쳤다
    store.append_event("env_change", 4)
    calls = spy_replays(monkeypatch, lab)
    lab.rb_fast.setChecked(True)
    assert lab.open_run(store.root)
    assert lab.champion is not None and lab.champion["generation"] == champ_gen
    old, new = env_hash(compile_env(LANDER_TEXT).file), env_hash(compile_env(lander_variant()).file)
    assert env_hash(lab.spec.file) == new  # 지금 실험실 환경은 고친 것
    lab.back_to_live()
    (tag, jobs) = calls[-1]
    assert tag[0] == "fast" and env_hash(jobs[0][0].file) == old


# ---------------------------------------------------------------------------
# 24. 복제해서 새 실행: 화면에 칸이 없는 학습 설정도 그대로 가져간다
# ---------------------------------------------------------------------------


def test_clone_keeps_settings_without_form_fields(qtbot: Any, lab: LabView, tmp_path: Path) -> None:
    original = small_config(3, validate_every=24, max_evaluations=36, diagonal_threshold=7)
    store = new_run(tmp_path / "runs", "cloned", original)
    info = clone_settings(store.root)
    assert lab.use_env_text(info["env_text"], info.get("env_name", ""), info.get("config") or {})
    assert lab.config_from_form().to_dict() == original.to_dict()
    lab.seed.setValue(11)  # 칸으로 바꾼 값만 바뀐다
    assert lab.config_from_form().to_dict() == {**original.to_dict(), "run_seed": 11}
    # 다른 환경 파일을 새로 불러오면 칸이 없는 값은 기본으로 돌아간다
    assert lab.load_env(LANDER)
    c = lab.config_from_form()
    assert (c.validate_every, c.max_evaluations, c.diagonal_threshold) == (1000, None, 1000)


def test_main_window_records_resume_waits_for_lab_without_branch_folder(
        qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, message_boxes: list[tuple[str, str]]) -> None:
    """실험실이 학습 중이면 기록 > 이어하기(옛 체크포인트 분기 포함)는 폴더를 만들기 전에 거절한다(검토 13, 19)."""
    from evolab.app.main import MainWindow
    from evolab.app.records.view import RecordsView

    monkeypatch.setattr(paths, "user_state_dir", lambda: tmp_path / "state")
    paths.set_custom_data_dir(tmp_path / "data")
    runs = paths.runs_dir()
    run = new_run(runs, "parent", small_config(2))
    run.save_checkpoint(2, {"x": 2})
    run.save_checkpoint(4, {"x": 4})
    window = MainWindow()
    qtbot.addWidget(window)
    errors: list[str] = []
    asked: list[Any] = []
    monkeypatch.setattr(window.records, "show_error", errors.append)
    monkeypatch.setattr(window.records, "ask_checkpoint", lambda s, gens: asked.append(gens) or gens[0])
    monkeypatch.setattr(window.records, "confirm_branch", lambda s, g: True)
    before = sorted(p.name for p in runs.iterdir())
    window.lab.client._proc = object()  # 실험실에서 다른 실행을 학습 중(가짜)
    try:
        window.records.refresh()
        assert window.records.select_run(run.root)
        with qtbot.assertNotEmitted(window.records.resume_requested):
            window.records.resume_button.click()
        assert asked == [] and errors == [S.REC_RESUME_FAILED.format(reason=S.MAIN_LAB_BUSY)]
        assert sorted(p.name for p in runs.iterdir()) == before
        assert isinstance(window.records, RecordsView)
    finally:
        window.lab.client._proc = None  # 창을 닫기 전에 가짜 세션을 치운다

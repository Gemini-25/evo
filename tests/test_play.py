"""M1: 직접 조종 창, 입력 기록 저장·재생, 재생 결정성."""

from pathlib import Path

from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from evolab import cli, paths
from evolab.app.play.record import PlayRecord, new_record, replay, verify_record
from evolab.app.play.window import PlayWindow, key_bindings, qt_key
from evolab.spec.io import load_env
from evolab.world.physics2d.episode import Episode, Trial
from lander_pilot import pilot

LANDER = paths.builtin_env_dir() / "lander.evoenv"


def recorded_run(seed: int) -> PlayRecord:
    spec = load_env(LANDER)
    rec = new_record(spec, seed)
    ep = Episode(spec, None, Trial(seed))
    while not ep.done:
        action = pilot(ep.observe())
        rec.actions.append(action)
        ep.step(action)
    res = ep.result()
    rec.score, rec.outcome, rec.steps = res.score, res.outcome, res.steps
    return rec


def test_record_round_trip_and_replay_is_bit_identical(tmp_path: Path) -> None:
    rec = recorded_run(21)
    path = tmp_path / "기록.evoplay"
    rec.save(path)
    loaded = PlayRecord.load(path)
    assert loaded.actions == rec.actions
    spec = load_env(LANDER)
    res = replay(spec, loaded)
    assert res.score == rec.score and res.steps == rec.steps
    same, _, _ = verify_record(spec, path)
    assert same


def test_cli_verify_reports_same(tmp_path: Path) -> None:
    rec = recorded_run(4)
    path = tmp_path / "r.evoplay"
    rec.save(path)
    assert cli.main(["play", str(LANDER), "--verify", str(path)]) == 0
    rec.score = (rec.score or 0.0) + 1.0
    rec.save(path)
    assert cli.main(["play", str(LANDER), "--verify", str(path)]) == 1


def test_lander_key_bindings() -> None:
    spec = load_env(LANDER)
    kb = key_bindings(spec)
    assert [b.output_index for b in kb] == [0, 1, 2]
    assert qt_key("Up") in kb[0].keys and qt_key("W") in kb[0].keys


def test_play_window_steps_with_keyboard(qtbot: QtBot) -> None:
    window = PlayWindow(load_env(LANDER), seed=3)
    qtbot.addWidget(window)
    window.show()
    qtbot.keyPress(window.canvas, Qt.Key.Key_Up)
    qtbot.waitUntil(lambda: window.episode is not None and window.episode.steps > 20, timeout=5000)
    qtbot.keyRelease(window.canvas, Qt.Key.Key_Up)
    assert window.record is not None
    assert any(a[0] == 1.0 for a in window.record.actions)  # 위쪽 키 = 주 추진기
    assert window.blocks.rowCount() == len(window.spec.file.scoring.blocks) + 1
    assert window.sensors.rowCount() == 9


def test_play_window_restart_and_pause(qtbot: QtBot) -> None:
    window = PlayWindow(load_env(LANDER), seed=8)
    qtbot.addWidget(window)
    window.show()
    qtbot.keyClick(window.canvas, Qt.Key.Key_Space)
    assert window.paused
    steps = window.episode.steps if window.episode else 0
    qtbot.wait(200)
    assert window.episode is not None and window.episode.steps == steps
    qtbot.keyClick(window.canvas, Qt.Key.Key_R)
    assert not window.paused and window.seed == 8 and window.episode.steps == 0


def test_main_window_opens_builtin_lander(qtbot: QtBot) -> None:
    from evolab.app.main import MainWindow

    main = MainWindow()
    qtbot.addWidget(main)
    w = main.open_play(LANDER)
    assert w is not None
    qtbot.addWidget(w)

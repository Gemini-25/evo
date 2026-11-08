"""v2.0 직접 조종 (설계 5.2, D12·D49, 8.4절 T45~T48): 점프 게임의 Space·Esc, HUD, 움직이는 장애물·먼지 그림, 기본 보기
카메라, 글라이더 지켜보기, 새 칸 이름·도움말, 입력 기록 재생과 「내 최고 기록」."""

from __future__ import annotations

import math
import typing
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QMessageBox
from pytestqt.qtbot import QtBot

from evolab import paths
from evolab.app import strings as S
from evolab.app.play import window as play_window
from evolab.app.play.record import PlayRecord, best_record_score, new_record, verify_record
from evolab.app.play.window import (
    COLOR_WORLD_BODY,
    DUST_SECONDS,
    PlayWindow,
    key_bindings,
    qt_key,
    uses_space,
    view_camera,
)
from evolab.engine.pool import WorkerPool
from evolab.engine.runsetup import create_run
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.spec import io
from evolab.spec.editor_meta import view_box
from evolab.spec.io import compile_env, env_hash, load_env
from evolab.world.physics2d.episode import Episode, EpisodeBuildError, Trial
from evolab.world.physics2d.motion import pose_at

ENV = paths.builtin_env_dir()
JUMP = ENV / "jump.evoenv"
GLIDER = ENV / "glider.evoenv"
LANDER = ENV / "lander.evoenv"
SPACE = int(Qt.Key.Key_Space.value)
ESC = int(Qt.Key.Key_Escape.value)


class FakeClock:
    """창의 실제 시계 대신: elapsed()를 부를 때마다 MAX_STEPS_PER_TICK 스텝만큼 흐른다(_tick이 매번 그만큼 돈다)."""

    def __init__(self, dt: float) -> None:
        self.step_ms = play_window.MAX_STEPS_PER_TICK * dt * 1000.0
        self.ms = 0.0

    def elapsed(self) -> int:
        self.ms += self.step_ms
        return int(math.ceil(self.ms))

    def restart(self) -> int:
        self.ms = 0.0
        return 0

    def start(self) -> None:
        self.ms = 0.0


def make_window(qtbot: QtBot, path: Path, seed: int = 5, *, show: bool = False) -> PlayWindow:
    w = PlayWindow(load_env(path), seed=seed)
    qtbot.addWidget(w)
    if show:
        w.show()
        qtbot.waitExposed(w)
    else:
        w.timer.stop()
    return w


def step(w: PlayWindow, action: list[float], n: int = 1) -> None:
    """창의 _tick과 같은 순서(기록 → step → 먼지)로 n스텝."""
    ep = w.episode
    assert ep is not None and w.record is not None
    for _ in range(n):
        if ep.done:
            return
        w.record.actions.append(list(action))
        ep.step(action)
        if w._jump_index:
            w._note_jumps(ep)


def finish(w: PlayWindow, limit: int = 100000) -> None:
    """가짜 시계로 _tick을 에피소드가 끝날 때까지 부른다(실제 시간을 기다리지 않음)."""
    assert w.episode is not None
    w.clock = FakeClock(w.episode.dt)  # type: ignore[assignment]
    w._steps_since_clock = 0
    for _ in range(limit):
        if w.episode.done:
            return
        w._tick()
    raise AssertionError("episode did not finish")


# ---------------------------------------------------------------------------
# T45: Space·Esc와 도움말
def test_t45_jump_game_space_is_a_jump_key_and_esc_pauses(qtbot: QtBot) -> None:
    w = make_window(qtbot, JUMP)
    assert [kb.key_names for kb in w.bindings] == [["Up", "W", "Space"]]
    assert uses_space(w.bindings) and not w.space_pauses
    keys = S.PLAY_KEY_LINE.format(keys="Up / W / Space", label="점프")
    assert w.help_label.text() == S.PLAY_HELP.format(keys=keys, pause=S.PLAY_PAUSE_KEYS["esc"])
    assert "Space" not in w.help_label.text().split("\n")[-1]  # 창 제어 줄에는 Esc만

    qtbot.keyPress(w.canvas, Qt.Key.Key_Space)
    assert not w.paused and SPACE in w.pressed and w.current_action() == [1.0]
    qtbot.keyRelease(w.canvas, Qt.Key.Key_Space)
    assert SPACE not in w.pressed and w.current_action() == [0.0]
    qtbot.keyClick(w.canvas, Qt.Key.Key_Escape)
    assert w.paused and ESC not in w.pressed
    qtbot.keyClick(w.canvas, Qt.Key.Key_Escape)
    assert not w.paused


def test_t45_lander_space_and_esc_both_pause(qtbot: QtBot) -> None:
    w = make_window(qtbot, LANDER)
    assert not uses_space(w.bindings) and w.space_pauses
    assert w.help_label.text().endswith(S.PLAY_HELP.split("\n")[-1].format(pause=S.PLAY_PAUSE_KEYS["both"]))
    qtbot.keyClick(w.canvas, Qt.Key.Key_Space)
    assert w.paused and SPACE not in w.pressed
    qtbot.keyClick(w.canvas, Qt.Key.Key_Escape)
    assert not w.paused
    qtbot.keyClick(w.canvas, Qt.Key.Key_Escape)
    assert w.paused


def test_t45_holding_space_jumps_in_the_running_window(qtbot: QtBot) -> None:
    """실제 시계로 도는 창: Space를 누르고 있으면 뛰고(실제 출력 1인 스텝의 먼지), 일시정지하지 않는다."""
    w = make_window(qtbot, JUMP, show=True)
    qtbot.keyPress(w.canvas, Qt.Key.Key_Space)
    qtbot.waitUntil(lambda: w.episode is not None and w.episode.steps > 30 and bool(w.canvas.dust), timeout=10000)
    qtbot.keyRelease(w.canvas, Qt.Key.Key_Space)
    assert not w.paused and w.record is not None
    assert any(a == [1.0] for a in w.record.actions)
    qtbot.keyClick(w.canvas, Qt.Key.Key_Escape)
    assert w.paused
    steps = w.episode.steps
    qtbot.wait(150)
    assert w.episode.steps == steps  # 일시정지 중에는 진행하지 않는다


# ---------------------------------------------------------------------------
# T46: HUD, 글라이더 지켜보기
def test_t46_hud_shows_jump_ready_and_air(qtbot: QtBot) -> None:
    w = make_window(qtbot, JUMP)
    step(w, [0.0], 10)
    assert w._hud().endswith(S.PLAY_JUMP_READY) and w.episode.jump_ready("jump")
    assert not w.canvas.dust
    step(w, [1.0])
    assert w.episode.applied_outputs[0] == 1.0
    (t0, x, y), = w.canvas.dust
    assert t0 == w.episode.time and x == w.episode.bodies["runner"].position.x and abs(x) < 0.01
    assert y == pytest.approx(min(s.bb.bottom for s in w.episode.bodies["runner"].shapes))
    step(w, [1.0], 5)  # 누르고 있어도 공중에서는 다시 뛰지 않는다
    assert w._hud().endswith(S.PLAY_JUMP_AIR) and len(w.canvas.dust) == 1
    # 먼지는 DUST_SECONDS 뒤에 사라진다
    w.canvas.set_frame(w.episode.snapshot(), w._hud())
    w.canvas.grab()
    assert len(w.canvas.dust) == 1
    step(w, [0.0], int(DUST_SECONDS / w.episode.dt) + 2)
    w.canvas.set_frame(w.episode.snapshot(), w._hud())
    w.canvas.grab()
    assert w.canvas.dust == []


def test_t46_glider_is_watch_only(qtbot: QtBot) -> None:
    w = make_window(qtbot, GLIDER)
    assert w.bindings == [] and w.jumps == [] and w.space_pauses
    assert w.help_label.text() == S.PLAY_HELP.format(keys=S.PLAY_WATCH_ONLY, pause=S.PLAY_PAUSE_KEYS["both"])
    step(w, [], 30)
    hud = w._hud()
    assert S.PLAY_JUMP_READY not in hud and S.PLAY_JUMP_AIR not in hud
    assert w.current_action() == []
    qtbot.keyClick(w.canvas, Qt.Key.Key_Space)
    assert w.paused


# ---------------------------------------------------------------------------
# 그리기: 움직이는 장애물, 땅, 기본 보기 카메라
def _color_near(img: QImage, x: float, y: float, color: QColor, tol: int = 12) -> bool:
    c = img.pixelColor(round(x), round(y))
    return all(abs(a - b) <= tol for a, b in zip((c.red(), c.green(), c.blue()),
                                                (color.red(), color.green(), color.blue()), strict=True))


def test_moving_obstacles_and_ground_are_drawn_where_the_runtime_has_them(qtbot: QtBot) -> None:
    w = make_window(qtbot, JUMP)
    w.resize(1280, 760)
    ep = w.episode
    geom = ep.geometry()
    obs = next(b for b in geom.bodies if b.motion is not None)
    assert obs.id == "obs" and not obs.moving
    for n in (0, 90, 150):
        step(w, [0.0], n - ep.steps)
        frame = ep.snapshot()
        w.canvas.set_frame(frame, w._hud())
        img = w.canvas.grab().toImage()
        (x, y, a), = w.canvas.mover_poses
        assert (x, y, a) == pose_at(obs.motion, frame.time)
        rx, ry = ep.bodies["obs"].position
        assert abs(rx - x) < 1e-9 and abs(ry - y) < 1e-9  # 런타임이 그 스텝에 놓은 자리
        # 첫 장애물 가운데 화소가 세계 물체 색(가장자리 선이 아닌 안쪽)
        x0, y0, x1, y1 = (min(p[0] for p in obs.shapes[0].points), min(p[1] for p in obs.shapes[0].points),
                          max(p[0] for p in obs.shapes[0].points), max(p[1] for p in obs.shapes[0].points))
        cx, cy = x + (x0 + x1) / 2.0, y + (y0 + y1) / 2.0
        s = w.canvas.scale
        (ox, oy) = w.canvas.center
        sx, sy = w.canvas.width() / 2.0 + (cx - ox) * s, w.canvas.height() / 2.0 - (cy - oy) * s
        if 0 <= sx < img.width() and 0 <= sy < img.height():
            assert _color_near(img, sx, sy, COLOR_WORLD_BODY)
        # 땅(정적 사각형, 윗면 y = 0)도 그린다(v1 직접 조종 화면은 선분 지형만 그렸다)
        gx, gy = w.canvas.width() / 2.0 + (8.0 - ox) * s, w.canvas.height() / 2.0 - (-0.5 - oy) * s
        assert _color_near(img, gx, gy, COLOR_WORLD_BODY)


def test_view_camera_is_a_pure_function_that_moves_only_past_the_edge() -> None:
    box = (-4.0, -1.0, 16.0, 6.0)
    (cx, cy), s = view_camera(box, (0.0, 0.5), 1000, 700)
    assert (cx, cy) == (6.0, 2.5) and s == pytest.approx(min(1000 / 20, 700 / 7))
    assert view_camera(box, None, 1000, 700) == ((6.0, 2.5), s)
    # 보이는 범위의 가장자리(VIEW_EDGE 안쪽) 안에서는 그대로
    hw = (1.0 - play_window.VIEW_EDGE) * 1000 / (2 * s)
    assert view_camera(box, (6.0 + hw - 0.01, 0.5), 1000, 700)[0] == (6.0, 2.5)
    # 넘으면 넘은 만큼만(배율 그대로)
    (cx2, cy2), s2 = view_camera(box, (6.0 + hw + 3.0, 0.5), 1000, 700)
    assert cx2 == pytest.approx(9.0) and cy2 == 2.5 and s2 == s
    (cx3, _), _ = view_camera(box, (6.0 - hw - 2.0, 0.5), 1000, 700)
    assert cx3 == pytest.approx(4.0)
    # 확대는 배율만
    assert view_camera(box, (0.0, 0.5), 1000, 700, zoom=2.0)[1] == pytest.approx(2 * s)


def test_play_camera_starts_at_the_default_view_and_follows_the_glider(qtbot: QtBot) -> None:
    w = make_window(qtbot, JUMP)
    w.resize(1280, 760)
    w.canvas.set_world(w.episode.geometry(), w.episode.snapshot(), "runner", w.view)
    assert w.view == view_box(w.spec.file) == (-3.0, -1.0, 22.0, 6.0)
    box_center = (9.5, 2.5)
    assert w.canvas.center == box_center
    step(w, [1.0], 20)  # 뛰어도(최고 1.9m) 상자 안이라 카메라가 그대로
    w.canvas.set_frame(w.episode.snapshot(), w._hud())
    assert w.canvas.center == box_center

    g = make_window(qtbot, GLIDER)
    g.resize(1280, 760)
    g.canvas.set_world(g.episode.geometry(), g.episode.snapshot(), "glider", g.view)
    assert g.canvas.center == (1.0, 11.0)
    xs = []
    for _ in range(6):
        step(g, [], 60)
        g.canvas.set_frame(g.episode.snapshot(), g._hud())
        xs.append(g.canvas.center[0])
    gx = g.episode.bodies["glider"].position.x
    assert xs[-1] > 1.0 and xs == sorted(xs)  # 멀리 날아가면 따라간다
    hw = (1.0 - play_window.VIEW_EDGE) * g.canvas.width() / (2 * g.canvas.scale)
    assert gx <= g.canvas.center[0] + hw + 1e-9

    lander = make_window(qtbot, LANDER)
    assert lander.view is None and lander.canvas.view_box is None  # 기본 보기가 없으면 지금 규칙


def test_body_build_failure_message(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    w = make_window(qtbot, JUMP)
    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda _p, _t, text, *a, **k: shown.append(text)))
    body_error = EpisodeBuildError(S.RT_DEGENERATE_SHAPE.format(body="runner"))
    world_error = EpisodeBuildError("generators.obs: nope")
    assert w.build_failed_text(3, body_error) == (
        S.PLAY_BODY_BUILD_FAILED + "\n" + S.PLAY_BUILD_FAILED.format(seed=3, reason=body_error))
    assert w.build_failed_text(3, world_error) == S.PLAY_BUILD_FAILED.format(seed=3, reason=world_error)

    def broken(*_a: Any, **_k: Any) -> Any:
        raise body_error

    monkeypatch.setattr(play_window, "Episode", broken)
    old = w.episode
    assert not w.start(9)
    assert shown == [w.build_failed_text(9, body_error)] and w.episode is old and w.seed == 5


# ---------------------------------------------------------------------------
# T47: 새 칸·고르기 값의 한국어 이름과 도움말(묶은 exe에는 model.py 설명이 없다)
def _literals(t: Any) -> list[Any]:
    if typing.get_origin(t) is typing.Literal:
        return list(typing.get_args(t))
    return [v for a in typing.get_args(t) for v in _literals(a)]


def test_t47_every_new_field_and_value_has_a_korean_name_and_tip(monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.editor import forms

    monkeypatch.setattr(forms, "_docstrings", lambda: {})  # 묶은 exe처럼 model.py 설명 없이
    fields = [(model, key) for model, key in io.FIELD_SINCE]
    fields += [(model, f.alias or name) for model in io.KIND_SINCE for name, f in model.model_fields.items()
               if name != "kind"]  # kind는 칸이 아니라 종류 고르기(이름은 값 이름)
    assert len(fields) > 40
    missing = [f"{m.__name__}.{k}" for m, k in fields
               if forms.field_name(m, k) == k or not forms.field_tip(m, k, k)]
    assert missing == []
    values = [(name, v) for model in io.KIND_SINCE for name, f in model.model_fields.items()
              for v in _literals(f.annotation)]
    values += [(key, v) for _model, key, v in io.VALUE_SINCE]
    assert ("mode", "pingpong") in values and ("surfaces", "wings") in values and ("type", "kinematic") in values
    assert [(k, v) for k, v in values if forms.value_name(k, v) == str(v)] == []
    assert all(any("가" <= ch <= "힣" for ch in forms.value_name(k, v)) for k, v in values)


# ---------------------------------------------------------------------------
# T48: 입력 기록 재생, 「내 최고 기록」(env_hash로 묶임) → 실험실 점수 그래프 선
def jump_record(seed: int) -> PlayRecord:
    """레이로 가까운 장애물이 보이면 뛰는 손 규칙으로 한 판(기록과 결과)."""
    spec = load_env(JUMP)
    rec = new_record(spec, seed)
    ep = Episode(spec, None, Trial(seed))
    while not ep.done:
        action = [1.0 if float(ep.observe()[:7].min()) < 0.3 else 0.0]
        rec.actions.append(action)
        ep.step(action)
    res = ep.result()
    rec.score, rec.outcome, rec.steps = res.score, res.outcome, res.steps
    return rec


def test_t48_jump_record_replays_identically_in_the_window(qtbot: QtBot, tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    rec = jump_record(17)
    assert any(a == [1.0] for a in rec.actions) and rec.steps is not None and rec.steps > 60
    path = tmp_path / "점프.evoplay"
    rec.save(path)
    same, res, _ = verify_record(load_env(JUMP), path)
    assert same and res.score == rec.score
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda _p, t, text, *a, **k: shown.append((t, text))))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda _p, t, text, *a, **k: shown.append((t, text))))
    w = make_window(qtbot, JUMP, seed=99)
    w.play_record(path)
    assert w.replaying is not None and w.seed == 17
    finish(w)
    assert w.last_result is not None and w.last_result.score == rec.score and w.last_result.steps == rec.steps
    assert shown == [(S.PLAY_REPLAY_TITLE, S.PLAY_REPLAY_SAME.format(score=rec.score))]
    assert w.record is not None and w.record.actions == rec.actions  # 먼지를 남기며 재생해도 같은 입력·결과


def test_t48_best_record_is_bound_to_env_hash_and_drawn_in_the_lab(qtbot: QtBot, tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.lab import view as lab_view
    from evolab.app.lab.view import LabView

    records = tmp_path / "records"
    records.mkdir()
    monkeypatch.setattr(paths, "records_dir", lambda: records)
    monkeypatch.setattr(paths, "default_records_dir", lambda: tmp_path / "없음")
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path / "runs")
    a, b = jump_record(17), jump_record(23)
    a.save(records / "a.evoplay")
    b.save(records / "b.evoplay")
    jump, lander = load_env(JUMP), load_env(LANDER)
    best = max(a.score, b.score)
    assert best_record_score(jump, [records]) == best == lab_view.my_best_score(jump)
    assert lab_view.my_best_score(lander) is None  # 다른 환경(해시)의 기록은 쓰지 않는다
    assert a.env_hash == env_hash(jump.file)

    # 작은 점프 게임 실행을 열면 점수 그래프에 「내 최고 기록」 선
    text = JUMP.read_text(encoding="utf-8")
    spec = compile_env(text)
    config = TrainConfig(popsize=4, trials=1, run_seed=3, checkpoint_every=2, max_generations=2)
    store = create_run(tmp_path / "runs", text, spec, config, 0)
    with WorkerPool(0) as pool:
        Trainer(text, config, pool, store).run()
    view = LabView(settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(view)
    try:
        assert view.open_run(store.root)
        assert view.dashboard.my_best == best
        tile = view.dashboard.tile("trend")
        assert tile.my_best.value() == pytest.approx(best) and tile.my_best.isVisible()
    finally:
        view.shutdown()


def test_key_bindings_cover_jump_and_keep_v1_bindings() -> None:
    kb = key_bindings(load_env(JUMP))
    assert len(kb) == 1 and kb[0].output_index == 0 and kb[0].value == 1.0
    assert kb[0].keys == [qt_key("Up"), qt_key("W"), qt_key("Space")]
    assert [b.output_index for b in key_bindings(load_env(LANDER))] == [0, 1, 2]

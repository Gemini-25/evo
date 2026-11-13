"""v2.0 실험실 화면 (설계 5.3·5.4, D7·D25·D27, 4.9 화면 쪽, 8.4절 T49~T52): 관전 캔버스의 움직이는 물체 층, 기본 보기 처음
카메라, 관전 256개체 속도, 경주·GIF의 장애물, 뇌 없는 환경의 대시보드 빈 상태, 조립 실패 개체의 관전 장면, 예제 순서."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QSettings
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QMessageBox
from pytestqt.qtbot import QtBot

from evolab import paths
from evolab.app import errors
from evolab.app import strings as S
from evolab.app.lab import canvas as canvas_mod
from evolab.app.play import window as play_window
from evolab.app.lab.canvas import COLOR_WORLD_BODY, SpectatorCanvas
from evolab.app.lab.view import LabView, build_failures, build_scene
from evolab.engine import seeds
from evolab.engine.evaluate import Trajectory, evaluate
from evolab.engine.pool import WorkerPool
from evolab.engine.replay import replay
from evolab.engine.runsetup import create_run
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.policy.genome import layout_for
from evolab.spec.editor_meta import view_box
from evolab.spec.io import compile_env, env_hash, load_env
from evolab.world.physics2d.episode import BodyGeom, Episode, EpisodeBuildError, WorldGeometry
from evolab.world.physics2d.motion import pose_at

ENV = paths.builtin_env_dir()
JUMP = ENV / "jump.evoenv"
GLIDER = ENV / "glider.evoenv"
LANDER = ENV / "lander.evoenv"
W, H = 1000, 600
STEP = 2.0 / 60.0
"""관전 궤적 프레임 간격(every 2, dt 1/60)."""


def population(path: Path, n: int, seed: int = 3, sigma: float = 0.5) -> tuple[Any, Any, np.ndarray, Any]:
    spec = load_env(path)
    layout = layout_for(spec)
    X = layout.initial()[None, :] + sigma * np.random.default_rng(seed).standard_normal((n, layout.dim))
    trial = seeds.display_trial()
    res = evaluate(spec, layout, X, [(i, trial) for i in range(n)], record=[True] * n, every=2)
    return spec, layout, X, res


@pytest.fixture(scope="module")
def jump32() -> tuple[Any, Any, np.ndarray, Any]:
    return population(JUMP, 32)


def ranks_of(scores: np.ndarray) -> tuple[np.ndarray, int]:
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(len(scores))
    return ranks, int(order[0])


def make_canvas(qtbot: QtBot, spec: Any, view: Any = "env", size: tuple[int, int] = (W, H)) -> SpectatorCanvas:
    c = SpectatorCanvas()
    qtbot.addWidget(c)
    c.resize(*size)
    ep = Episode(spec, None, seeds.display_trial())
    c.set_scene(ep.geometry(), channels=ep.actuator_channels(), view=view_box(spec.file) if view == "env" else view)
    return c


def render(c: SpectatorCanvas) -> QImage:
    img = QImage(c.width(), c.height(), QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    c.render(img)
    return img


def near(img: QImage, p: QPointF, color: QColor, tol: int = 14) -> bool:
    x, y = round(p.x()), round(p.y())
    if not (0 <= x < img.width() and 0 <= y < img.height()):
        return False
    c = img.pixelColor(x, y)
    return abs(c.red() - color.red()) <= tol and abs(c.green() - color.green()) <= tol and \
        abs(c.blue() - color.blue()) <= tol


def first_box(world: WorldGeometry) -> tuple[BodyGeom, tuple[float, float]]:
    """장애물 줄 물체와 그 첫 장애물 사각형의 가운데(물체 좌표)."""
    obs = next(b for b in world.bodies if b.motion is not None)
    pts = obs.shapes[0].points
    return obs, ((min(p[0] for p in pts) + max(p[0] for p in pts)) / 2.0,
                 (min(p[1] for p in pts) + max(p[1] for p in pts)) / 2.0)


def box_center_at(obs: BodyGeom, local: tuple[float, float], t: float) -> tuple[float, float]:
    x, y, a = pose_at(obs.motion, t)
    c, s = np.cos(a), np.sin(a)
    return x + local[0] * c - local[1] * s, y + local[0] * s + local[1] * c


# ---------------------------------------------------------------------------
# T49: 움직이는 물체 층
def test_t49_moving_bodies_are_not_in_the_static_cache_and_are_drawn_at_pose_at(qtbot: QtBot, jump32: Any) -> None:
    spec, _layout, _X, res = jump32
    c = make_canvas(qtbot, spec)
    ranks, champ = ranks_of(res.scores)
    c.set_population(list(res.trajectories), ranks=ranks, champion=champ)
    obs, local = first_box(c.world)
    assert [mv.id for mv in c._movers] == ["obs"]
    # 정적 그림(캐시)에는 장애물 줄이 없다: 첫 장애물의 시작 자리는 정적 채우기 경로 밖
    start = QPointF(*box_center_at(obs, local, 0.0))
    assert not c._static_fill.contains(start)
    assert c.world is not None and obs in c.world.bodies and not obs.moving
    for t in (0.0, 1.0, 2.0, 2.5, 1.0, 0.0):  # 타임라인처럼 시각을 앞뒤로 옮겨도 시각만의 함수
        for _ in range(3):
            c.set_time(t)
        img = render(c)
        assert c.mover_time == pytest.approx(t, abs=1e-12)
        assert c.mover_poses == [pose_at(obs.motion, c.mover_time)]
        p = c.world_to_screen(*box_center_at(obs, local, t))
        if 0 <= p.x() < W:
            assert near(img, p, COLOR_WORLD_BODY), (t, p)
    # 프레임 사이 시각은 보여 주는 프레임의 시각으로 내린다(개체와 장애물이 시뮬레이션처럼 맞물리게)
    c.set_time(1.01)
    render(c)
    assert c.mover_time == pytest.approx(30 * STEP) and c.frame_time() == c.mover_time
    # 정적 그림을 그려 둔 상태에서도 장애물은 프레임마다 옮겨 그린다
    c.set_time(2.0)
    render(c)
    render(c)
    assert c._static_pixmap is not None
    c.set_time(2.5)
    img = render(c)
    p = c.world_to_screen(*box_center_at(obs, local, 2.5))
    old = c.world_to_screen(*box_center_at(obs, local, 2.0))
    assert near(img, p, COLOR_WORLD_BODY) and abs(p.x() - old.x()) > 5


def test_t49_runtime_obstacle_pose_matches_the_drawn_pose(jump32: Any) -> None:
    """화면이 그리는 자세 = 런타임이 그 스텝에 둔 자세(궤적에 담지 않아도 같은 시간 함수)."""
    spec = jump32[0]
    ep = Episode(spec, None, seeds.display_trial())
    obs = next(b for b in ep.geometry().bodies if b.motion is not None)
    for k in range(1, 121):
        ep.step([0.0])
        if k % 30 == 0:
            x, y, a = pose_at(obs.motion, ep.time)
            body = ep.bodies["obs"]
            assert abs(body.position.x - x) < 1e-9 and abs(body.position.y - y) < 1e-9 and abs(body.angle - a) < 1e-9


def test_t49_versus_and_timeline_replays_draw_the_obstacles(qtbot: QtBot, jump32: Any) -> None:
    """1세대 vs 지금(겹쳐 보기 재생 둘)과 타임라인 재생(하나)도 같은 캔버스·같은 장애물 층."""
    spec, layout, X, _res = jump32
    reps = [replay(spec, layout, X[k], seeds.display_trial()) for k in (0, 1)]
    c = SpectatorCanvas()
    qtbot.addWidget(c)
    c.resize(W, H)
    c.set_scene(reps[0].geometry, None, channels=reps[0].channels, view=view_box(spec.file))
    c.set_overlay([(reps[0].trajectory, reps[0].agent, QColor(80, 140, 255), "1"),
                   (reps[1].trajectory, reps[1].agent, QColor(255, 120, 40), "2")])
    obs, local = first_box(c.world)
    for t in (2.0, 0.5, 3.0):
        c.set_time(t)
        img = render(c)
        assert c.mover_poses == [pose_at(obs.motion, c.mover_time)]
        p = c.world_to_screen(*box_center_at(obs, local, c.mover_time))
        if 0 <= p.x() < W:
            assert near(img, p, COLOR_WORLD_BODY)


# ---------------------------------------------------------------------------
# T49: 기본 보기 = 처음 카메라
def test_t49_default_view_is_the_first_camera_in_fit_and_follow(qtbot: QtBot, jump32: Any) -> None:
    spec, _layout, _X, res = jump32
    box = view_box(spec.file)
    assert box == (-3.0, -1.0, 22.0, 6.0)
    c = make_canvas(qtbot, spec)
    ranks, champ = ranks_of(res.scores)
    c.set_population(list(res.trajectories), ranks=ranks, champion=champ)
    c.set_time(0.0)
    want_scale = min(W / 25.0, H / 7.0)
    assert c.camera() == ((9.5, 2.5), pytest.approx(want_scale))
    c.set_follow("champion")
    assert c.camera() == ((9.5, 2.5), pytest.approx(want_scale))
    # 기본 보기가 없으면 지금 규칙(배율 30 이하, 상자 가운데가 아님)
    plain = make_canvas(qtbot, spec, view=None)
    plain.set_population(list(res.trajectories), ranks=ranks, champion=champ)
    plain.set_time(0.0)
    assert plain.camera()[1] <= 30.0 and plain.camera()[0] != (9.5, 2.5)


def test_t49_default_view_widens_only_when_agents_leave_the_box(qtbot: QtBot) -> None:
    spec, _layout, _X, res = population(GLIDER, 12)
    box = view_box(spec.file)
    c = make_canvas(qtbot, spec)
    ranks, champ = ranks_of(res.scores)
    c.set_population(list(res.trajectories), ranks=ranks, champion=champ)
    c.set_time(0.0)
    assert c.camera()[0] == ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    for _ in range(40):
        c.set_time(6.0)
    r = c._fit_rect
    assert r is not None and r != box
    assert r[0] <= box[0] and r[1] <= box[1] and r[2] >= box[2] and r[3] >= box[3]  # 상자를 품고 넓어짐
    xs = c._poses[:, 0]
    assert r[2] >= xs.max()
    # 챔피언 따라가기: 상자 배율 그대로, 챔피언이 가장자리를 넘으면 따라감(가는 쪽이 보이게 화면 폭의 65% 자리로)
    c.set_follow("champion")
    (cx, _cy), s = c.camera()
    assert s == pytest.approx(min(W / (box[2] - box[0]), H / (box[3] - box[1])))
    slot = c._follow_slot()
    assert slot is not None
    px = float(c._poses[c._offsets[slot], 0])
    hw = (1.0 - play_window.VIEW_EDGE) * W / (2.0 * s)
    assert px <= cx + hw + 1e-9
    py = float(c._poses[c._offsets[slot], 1])
    assert c.camera()[0] == pytest.approx(play_window.view_camera(box, (px, py), W, H)[0])  # 직접 조종과 같은 카메라


# ---------------------------------------------------------------------------
# T50: 관전 256개체
@pytest.mark.parametrize("path", [JUMP, GLIDER], ids=["jump", "glider"])
def test_t50_render_speed_256_agents(qtbot: QtBot, path: Path) -> None:
    """명세: 관전 개체 256개 60fps. CI 여유를 두고 50ms 아래인지만 본다(기존 착륙선 시험과 같은 방법). 측정 출력."""
    spec, layout, X, res = population(path, 32, seed=11)
    trs = (list(res.trajectories) * 8)[:256]
    agents = None
    if layout.n_body:
        _world, one = build_scene(spec, layout, X, seeds.display_trial())
        agents = (one * 8)[:256]
    c = SpectatorCanvas()
    qtbot.addWidget(c)
    c.resize(1280, 720)
    ep = Episode(spec, None, seeds.display_trial())
    c.set_scene(ep.geometry(), agents, channels=ep.actuator_channels(), view=view_box(spec.file))
    scores = np.array([t.duration for t in trs])
    ranks, champ = ranks_of(scores)
    c.set_population(trs, ranks=ranks, champion=champ, labels={champ: "#1"})
    c.set_hud("gen 1")
    img = QImage(1280, 720, QImage.Format.Format_ARGB32_Premultiplied)
    c.set_time(0.0)
    c.render(img)
    frames = 60
    times = []
    for f in range(frames):
        t0 = time.perf_counter()
        c.set_time(1.0 + f / 60.0)
        c.render(img)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = float(np.median(times))
    print(f"\nspectator canvas: 256 {path.stem}, 1280x720: median {ms:.2f} ms/frame, max {max(times):.2f}")
    assert ms < 50.0


# ---------------------------------------------------------------------------
# T51: 경주·GIF에 장애물, 개체마다 물체 수가 다른 장면
def _jump_entries(n: int = 2) -> list[Any]:
    from evolab.app.race.model import RaceEntry

    text = JUMP.read_text(encoding="utf-8")
    spec = compile_env(text)
    layout = layout_for(spec)
    X = layout.initial()[None, :] + 0.5 * np.random.default_rng(5).standard_normal((n, layout.dim))
    h = env_hash(spec.file)[:16]
    colors = ["#e76f51", "#2a9d8f", "#e9c46a"]
    return [RaceEntry(f"칸{k}", "", colors[k], text, h, X[k], 1) for k in range(n)]


def test_t51_race_lanes_show_the_obstacles_from_the_default_view(qtbot: QtBot) -> None:
    from evolab.app.race.view import RaceView, entry_view

    entries = _jump_entries(2)
    view = RaceView()
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.set_race(entries)
    qtbot.waitUntil(lambda: view.ready, timeout=30000)
    assert entry_view(entries[0]) == view_box(compile_env(entries[0].env_text).file)
    lane = view.lanes[0]
    c = lane.canvas
    assert c._view == (-3.0, -1.0, 22.0, 6.0) and [mv.id for mv in c._movers] == ["obs"]
    obs, local = first_box(c.world)
    t = 2.0  # 첫 장애물이 x = 20 - 6·2 - … ≈ 7.5m: 기본 보기 안
    c.set_time(t)
    img = render(c)
    p = c.world_to_screen(*box_center_at(obs, local, c.mover_time))
    assert 0 <= p.x() < c.width() and near(img, p, COLOR_WORLD_BODY)
    view.shutdown()


def test_t51_gif_frames_show_the_obstacles(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    """GIF(영상 모듈은 고치지 않음)도 같은 캔버스라 장애물을 그린다: 그린 프레임의 장애물 자리 화소."""
    from evolab.app.export import video

    entries = _jump_entries(1)
    spec = compile_env(entries[0].env_text)
    rep = replay(spec, layout_for(spec), entries[0].genome, seeds.display_trial())
    seen: list[tuple[QImage, QPointF | None, float]] = []

    class Recording(SpectatorCanvas):
        def render(self, target: Any, *args: Any) -> None:  # type: ignore[override]
            super().render(target, *args)
            obs, local = first_box(self.world)
            seen.append((target, self.world_to_screen(*box_center_at(obs, local, self.mover_time))
                         if self.mover_poses else None, self.mover_time))

    monkeypatch.setattr(canvas_mod, "SpectatorCanvas", Recording)
    frames = video.render_frames([(rep, QColor(255, 120, 40), "x")])
    assert len(frames) == len(seen) > 10
    # 선수가 부딪힌 뒤에는 끝난 선수가 장애물 위에 그려지므로 그 전 프레임만 본다
    hit = rep.trajectory.duration - 0.1
    shown = [near(img, p, COLOR_WORLD_BODY) for img, p, t in seen if p is not None and t < hit
             and 0 <= p.x() < img.width() and 0 <= p.y() < img.height()]
    assert len(shown) >= 5 and all(shown)  # 화면 안에 들어온 프레임마다 장애물이 보인다
    race = video.render_race_frames([(rep, QColor(255, 120, 40), "x")], 1)
    assert race and len(seen) > len(frames)


def test_t51_scenes_with_different_body_counts_per_individual(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    """(v2.1 자료를 미리) 개체마다 움직이는 물체 수가 다른 장면이 관전·경주·GIF 캔버스에서 예외 없이 그려진다."""
    from evolab.app.export import video

    spec = load_env(LANDER)
    ep = Episode(spec, None, seeds.display_trial())
    world = ep.geometry()
    moving = [b for b in world.bodies if b.moving]
    assert len(moving) == 3
    base = np.array([b.pose for b in moving], dtype=np.float32)
    full = Trajectory(np.stack([base + [0.0, -0.1 * k, 0.0] for k in range(10)]), np.zeros((10, 3), np.float32),
                      2, 1 / 60, 18)
    hull_only = Trajectory(full.poses[:, :1].copy(), np.zeros((10, 3), np.float32), 2, 1 / 60, 18)
    c = SpectatorCanvas()
    qtbot.addWidget(c)
    c.resize(640, 400)
    c.set_scene(world, [moving, moving[:1]], channels=ep.actuator_channels())
    c.set_population([full, hull_only], ranks=np.array([0, 1]), champion=0)
    for t in (0.0, 0.1, 1.0):
        c.set_time(t)
        render(c)
    assert c._offsets.tolist() == [0, 1, 4] or c._offsets.tolist() == [0, 3, 4]
    rep = replay(spec, layout_for(spec), layout_for(spec).initial(), seeds.display_trial())
    jump = _jump_entries(1)[0]
    jspec = compile_env(jump.env_text)
    rep2 = replay(jspec, layout_for(jspec), jump.genome, seeds.display_trial())
    assert len(rep.agent) == 3 and len(rep2.agent) == 1
    assert video.render_race_frames([(rep, QColor(255, 0, 0), "a"), (rep2, QColor(0, 255, 0), "b")], 2,
                                    progress=lambda d, _t: d < 5)


# ---------------------------------------------------------------------------
# T52: 뇌 없는 환경(글라이더)의 대시보드 빈 상태
def test_t52_brainless_glider_brain_and_action_tiles_say_why_they_are_empty(qtbot: QtBot, tmp_path: Path) -> None:
    from test_dashboard_v12 import feed, fun_dashboard

    dash = fun_dashboard(qtbot, tmp_path, "glider", genemap=False)
    brain, actions = dash.tile("brain"), dash.tile("actions")
    assert dash.watch_meta is not None and dash.watch_meta.n_out == 0 and dash.watch_meta.n_body == 8
    brain.update_data()
    actions.update_data()
    assert brain.message.text() == S.DASH_EMPTY_NO_BRAIN and brain.stack.currentWidget() is brain.message
    assert brain.source_kind is None and brain.title_text == "" and brain._notes == []
    assert actions.message.text() == S.DASH_EMPTY_NO_OUTPUTS and actions.stack.currentWidget() is actions.message
    assert actions.entries() == [] and actions.channel_box.count() == 0
    dash.grab()
    # 뇌가 있는 환경은 그대로(빈 상태 문구가 아님)
    lander = fun_dashboard(qtbot, tmp_path, "lander", name="lander.ini", genemap=False)
    feed(lander, "lander", range(1, 4), champion_every=1)
    lander.tile("brain").update_data()
    lander.tile("actions").update_data()
    assert lander.tile("brain").message.text() != S.DASH_EMPTY_NO_BRAIN
    assert lander.tile("actions").stack.currentWidget() is not lander.tile("actions").message


def test_t52_glider_generations_reach_the_fun_page(qtbot: QtBot, tmp_path: Path) -> None:
    from test_dashboard_v12 import feed, fun_dashboard

    dash = fun_dashboard(qtbot, tmp_path, "glider", genemap=False)
    feed(dash, "glider", range(1, 5), champion_every=2)  # 학습 중 세대 알림(관찰 기록 포함)
    for tid in ("brain", "actions", "body"):
        dash.tile(tid).update_data()
    assert dash.tile("brain").message.text() == S.DASH_EMPTY_NO_BRAIN
    assert dash.tile("actions").message.text() == S.DASH_EMPTY_NO_OUTPUTS
    body = dash.tile("body")
    assert body.stack.currentWidget() is not body.message  # 몸 수치 추이가 주인공(8개 수치)


def train_small(runs: Path, path: Path, gens: int = 2, **kw: Any) -> Any:
    text = path.read_text(encoding="utf-8")
    spec = compile_env(text)
    config = TrainConfig(popsize=4, trials=1, run_seed=7, checkpoint_every=2, max_generations=gens,
                         validate_every=4, **kw)
    store = create_run(runs, text, spec, config, 0)
    with WorkerPool(0) as pool:
        t = Trainer(text, config, pool, store)
        reason = t.run()
    store.append_event("stop", t.state.generation, reason=reason)
    return store


@pytest.fixture()
def lab(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path / "runs")
    view = LabView(settings=QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.show()
    qtbot.waitExposed(view)
    try:
        yield view
    finally:
        view.client.abandon(timeout=60)
        view.shutdown()


def test_t52_real_glider_run_opens_with_history_empty_tiles_and_default_view(qtbot: QtBot, lab: Any,
                                                                              tmp_path: Path) -> None:
    store = train_small(tmp_path / "runs", GLIDER)
    assert lab.open_run(store.root)
    dash = lab.dashboard
    assert dash.wait_history(8000)
    assert len(dash.history) == 2
    dash.set_page("fun")
    for tid in ("brain", "actions", "body"):
        dash.tile(tid).update_data()
    assert dash.tile("brain").message.text() == S.DASH_EMPTY_NO_BRAIN
    assert dash.tile("actions").message.text() == S.DASH_EMPTY_NO_OUTPUTS
    assert dash.tile("body").stack.currentWidget() is not dash.tile("body").message
    # 전시 장면(다음 차례): 기본 보기가 처음 카메라
    qtbot.waitUntil(lambda: lab.canvas.world is not None, timeout=10000)
    assert lab.canvas._view == view_box(lab.spec.file)
    lab.grab()


# ---------------------------------------------------------------------------
# 조립 실패(설계 4.9 화면 쪽): 관전 장면, 상태 줄, 재생 알림
def failing_env(tmp_path: Path) -> Path:
    """선수 폭이 몸 수치(w)인 점프 게임: w ≤ 0인 몸은 조립할 수 없다(처음 몸 0.6은 된다)."""
    data = json.loads(JUMP.read_text(encoding="utf-8"))
    data["params"] = [{"id": "w", "label": "폭", "value": 0.6, "evolve": True, "min": -0.5, "max": 1.0}]
    data["bodies"][1]["shapes"][0]["w"] = "=p.w"
    data["world"]["max_time"] = 3.0
    path = tmp_path / "fragile_jump.evoenv"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def fragile_population(spec: Any, layout: Any) -> tuple[np.ndarray, list[int]]:
    X = np.tile(layout.initial(), (6, 1))
    X[:, 0] = np.linspace(-1.0, 1.0, 6)  # 몸 유전자가 앞(w: -0.5..1.0)
    w = layout.body_values(X)[:, 0]
    bad = [i for i in range(6) if not w[i] > 0]
    assert 0 < len(bad) < 6, w
    return X, bad


def test_build_scene_leaves_unbuildable_individuals_empty(tmp_path: Path) -> None:
    spec = load_env(failing_env(tmp_path))
    layout = layout_for(spec)
    X, bad = fragile_population(spec, layout)
    world, agents = build_scene(spec, layout, X, seeds.display_trial())
    assert agents is not None and [i for i, a in enumerate(agents) if not a] == bad
    assert build_failures(agents) == len(bad) and build_failures(None) == 0
    good = next(i for i in range(6) if i not in bad)
    assert [b.id for b in world.bodies] == [b.id for b in Episode(spec, layout.body_values(X)[good],
                                                                   seeds.display_trial()).geometry().bodies]
    # 모두 실패하면 세계는 처음 몸으로 만든다
    world2, agents2 = build_scene(spec, layout, X[bad], seeds.display_trial())
    assert build_failures(agents2) == len(bad) and world2.bodies
    with pytest.raises(EpisodeBuildError):
        Episode(spec, layout.body_values(X)[bad[0]], seeds.display_trial())


def test_lab_plays_a_generation_with_unbuildable_individuals(qtbot: QtBot, lab: Any, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    path = failing_env(tmp_path)
    assert lab.load_env(path)
    spec, layout = lab.spec, lab.layout
    X, bad = fragile_population(spec, layout)
    trial = seeds.display_trial()
    res = evaluate(spec, layout, X, [(i, trial) for i in range(6)], record=[True] * 6, every=2)
    assert all(res.trajectories[i].frames == 0 for i in bad)
    scores = np.where(np.isfinite(res.scores), res.scores, np.nanmin(res.scores) - 1.0)
    rep = int(np.argmax(scores))
    msg = {"row": {"generation": 1}, "generation": 1, "population": X, "representative": rep, "scores": scores,
           "trial": (trial.seed, trial.track), "trajectories": list(res.trajectories), "champion": None,
           "env_version": env_hash(spec.file)[:16], "observation": None,
           "_scene": (spec, build_scene(spec, layout, X, trial))}
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(lab.client, "send", lambda m: sent.append(m) or True)
    lab.queue.append(msg)
    lab._play_next()
    line = S.LAB_BUILD_FAILED_N.format(n=len(bad))
    assert line in lab.status.text() and lab.status.text().count(line) == 1
    assert len(lab.canvas._population) == 6 - len(bad)
    assert lab.canvas._view == view_box(spec.file)
    lab.canvas.set_time(1.0)
    lab.canvas.grab()
    # 실패가 없는 다음 세대는 알림을 지운다
    good = np.tile(layout.initial(), (6, 1))
    res2 = evaluate(spec, layout, good, [(i, trial) for i in range(6)], record=[True] * 6, every=2)
    msg2 = dict(msg, generation=2, population=good, scores=res2.scores, trajectories=list(res2.trajectories),
                representative=0, _scene=(spec, build_scene(spec, layout, good, trial)))
    lab.playing = None
    lab.queue.append(msg2)
    lab._play_next()
    assert line not in lab.status.text()


def test_replay_of_an_unbuildable_body_says_so(qtbot: QtBot, lab: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    boxes: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda _p, _t, text, *a, **k: boxes.append(text)))
    err = EpisodeBuildError(S.RT_DEGENERATE_SHAPE.format(body="runner"))
    lab._replay_done((lab._replay_tag, ("timeline", ["x"])), err)
    lab._replay_done((lab._replay_tag, ("timeline", ["x"])), ValueError("other"))
    # v3.0(설계 D26·3.2): show_error 상자로 보인다. 모르는 예외(ValueError("other"))의 원문은 본문이 아니라 「자세히」로 간다
    shown = [b.friendly for b in errors.open_error_boxes() if b.parent() is lab]
    assert boxes == [] and [f.message for f in shown] == [S.PLAY_BODY_BUILD_FAILED + "\n" + str(err),
                                                          S.ERR_UNEXPECTED.format(kind="ValueError")]
    assert shown[0].next_step == S.ERR_BUILD_NEXT and "ValueError: other" in shown[1].detail


def test_race_build_failure_message(qtbot: QtBot) -> None:
    from evolab.app.race.view import RaceView

    view = RaceView()
    qtbot.addWidget(view)
    view._tag = 5
    err = EpisodeBuildError(S.RT_DEGENERATE_SHAPE.format(body="runner"))
    view._on_ready(5, err)
    assert view.failure == S.RACE_FAILED.format(reason=S.PLAY_BODY_BUILD_FAILED + " " + str(err))


# ---------------------------------------------------------------------------
# 내장 예제 순서(D27): 실험실 환경 목록, 반복 실행 대화 상자
def test_lab_and_repeat_env_lists_use_the_builtin_order(qtbot: QtBot, lab: Any, tmp_path: Path) -> None:
    from evolab.app.compare.repeat import RepeatDialog

    want = [str(p) for p in paths.builtin_envs()]
    assert [Path(p).stem for p in want][:5] == ["lander", "track_car", "hill_car", "jump", "glider"]
    combo = lab.env_combo
    assert combo.itemData(0) is None
    # 6.2.0(설계 4.5): 내장 묶음의 항목 == 내장 목록. 그 뒤에는 「내 프리셋」 묶음(머리 줄 + 내 환경 파일)이 올 수 있다
    assert [combo.itemData(i) for i in range(1, 1 + len(want))] == want
    assert combo.count() == 1 + len(want) or combo.itemData(1 + len(want)) is None
    dlg = RepeatDialog(runs_dir=tmp_path, workers=1)
    qtbot.addWidget(dlg)
    assert [dlg.env_combo.itemData(i) for i in range(dlg.env_combo.count())] == want
    # 점프 게임을 고르면 그 환경으로 학습할 수 있다
    assert lab.load_env(JUMP) and lab.spec is not None and lab.spec.file.name == load_env(JUMP).file.name
    qtbot.waitUntil(lambda: lab.canvas.world is not None, timeout=10000)


def test_live_scene_uses_the_default_view(qtbot: QtBot, lab: Any, jump32: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    spec, _layout, X, res = jump32
    assert lab.load_env(JUMP)
    trial = seeds.display_trial()
    monkeypatch.setattr(lab.client, "send", lambda m: True)
    msg = {"row": {"generation": 1}, "generation": 1, "population": X, "representative": int(np.argmax(res.scores)),
           "scores": res.scores, "trial": (trial.seed, trial.track), "trajectories": list(res.trajectories),
           "champion": None, "env_version": env_hash(lab.spec.file)[:16], "observation": None}
    lab.queue.append(msg)
    lab._play_next()
    assert lab.canvas._view == view_box(spec.file) and [mv.id for mv in lab.canvas._movers] == ["obs"]
    assert lab._build_note == "" and S.LAB_BUILD_FAILED_N.split("{")[0] not in lab.status.text()
    lab.canvas.set_time(0.0)
    x0, y0, x1, y1 = view_box(spec.file)  # type: ignore[misc]
    assert lab.canvas.camera()[0] == ((x0 + x1) / 2, (y0 + y1) / 2)


# ---------------------------------------------------------------------------
# 궤적 밀도 타일(v1.2 캔버스)의 카메라: 글라이더처럼 넓은 세계
def test_density_camera_shows_the_visited_cells_of_a_wide_world(qtbot: QtBot) -> None:
    """격자가 가장 작은 맞춤 배율(SCALE_FIT[0])로도 다 보이지 않는 넓은 세계(글라이더 1,230m)는 격자 가운데만 보이면
    지나간 칸이 화면 밖에 남는다: 지나간 칸 범위를 보인다. 격자가 다 보이는 세계(착륙선)는 v1.2처럼 격자 전체."""
    from evolab.app.lab.watch_tiles import density_image

    def camera(path: Path, counts: np.ndarray, frame: tuple[float, float, float, float]) -> tuple[Any, float, int, int]:
        c = SpectatorCanvas()
        qtbot.addWidget(c)
        c.resize(600, 180)  # 대시보드 타일 안의 캔버스 크기(캔버스 최소 크기가 더 크면 그 크기)
        spec = load_env(path)
        c.set_scene(Episode(spec, None, seeds.display_trial()).geometry(), None, None)
        c.set_density(density_image(counts), frame, fit=True)
        goal = c._camera_goal()
        assert goal is not None
        c.grab()
        return goal[0], goal[1], c.width(), c.height()

    # 글라이더: 128×14칸(9.6m), 지나간 칸은 x 37~130m, y -5~24m 근처
    counts = np.zeros((14, 128))
    counts[0, 7:17] = 5
    counts[2, 4:9] = 40
    (cx, cy), scale, w, h = camera(GLIDER, counts, (-30.0, -5.0, 128 * 9.6, 14 * 9.6))
    assert min(w / (128 * 9.6), h / (14 * 9.6)) < canvas_mod.SCALE_FIT[0]  # 격자 전체는 가장 작은 배율로도 안 보임
    used = (-30.0 + 3 * 9.6, -5.0, -30.0 + 18 * 9.6, -5.0 + 4 * 9.6)  # 지나간 칸 + 둘레 한 칸(격자 안)
    assert scale == pytest.approx(min(w / (used[2] - used[0]), h / (used[3] - used[1])))
    assert scale > canvas_mod.SCALE_FIT[0]
    assert (cx, cy) == pytest.approx(((used[0] + used[2]) / 2, (used[1] + used[3]) / 2))
    # 착륙선 크기 격자(140×90m)는 지금처럼 격자 전체를 보인다(지나간 칸과 무관)
    small = np.zeros((18, 28))
    small[3, 4] = 9
    (lx, ly), lscale, w, h = camera(LANDER, small, (-70.0, -20.0, 140.0, 90.0))
    assert (lx, ly) == (0.0, 25.0) and lscale == pytest.approx(min(w / 140.0, h / 90.0))


def test_density_tile_does_not_draw_the_obstacle_row_frozen_at_zero_seconds(qtbot: QtBot, tmp_path: Path) -> None:
    """궤적 밀도 타일은 여러 시각을 합친 그림이라 장애물 줄(정해진 움직임 물체)을 그리지 않는다(전에는 0초 자리에 멈춘
    장애물이 그려졌다). 관전 캔버스는 그대로 그린다."""
    from test_dashboard import make_dashboard

    from evolab.app.lab.watch_tiles import density_image

    spec = load_env(JUMP)
    geometry = Episode(spec, None, seeds.display_trial()).geometry()
    counts = np.zeros((8, 16))
    counts[2, 3] = 4
    dash = make_dashboard(qtbot, tmp_path)
    tile = dash.tile("density")
    spectator = SpectatorCanvas()
    qtbot.addWidget(spectator)
    for c, drawn in ((tile.canvas, False), (spectator, True)):  # type: ignore[attr-defined]
        c.resize(400, 200)
        c.set_scene(geometry, None, None)
        c.set_density(density_image(counts), (-10.0, -2.0, 64.0, 32.0), fit=True)
        c.mover_poses = []
        c.grab()
        assert [mv.id for mv in c._movers] == ["obs"]
        assert bool(c.mover_poses) == drawn


def test_inspect_panel_of_a_brainless_glider_says_there_is_no_brain(qtbot: QtBot) -> None:
    """뇌 없는 글라이더 개체를 눌러 들여다보면 빈 제목(센서·신경망 활성화·출력) 대신 「뇌가 없습니다」 안내."""
    from evolab.app.lab.view import InspectPanel

    panel = InspectPanel()
    qtbot.addWidget(panel)
    spec = load_env(GLIDER)
    layout = layout_for(spec)
    rep = replay(spec, layout, layout.initial(), seeds.display_trial(), trace=True)
    assert rep.inputs == [] and rep.channels == []
    panel.set_replay(3, rep)
    panel.show_time(2.0)
    assert panel.body.text() == S.DASH_EMPTY_NO_BRAIN
    lander = load_env(LANDER)
    ll = layout_for(lander)
    panel.set_replay(1, replay(lander, ll, ll.initial(), seeds.display_trial(), trace=True))
    panel.show_time(0.5)
    text = panel.body.text()
    assert S.LAB_INSPECT_SENSORS in text and S.LAB_INSPECT_OUTPUTS in text and S.DASH_EMPTY_NO_BRAIN not in text

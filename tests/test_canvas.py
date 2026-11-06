"""관전 화면(SpectatorCanvas): 장면, 집단 층, 시각별 프레임, 클릭, 겹쳐 보기, 카메라, 그리기 속도."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QWheelEvent
from pytestqt.qtbot import QtBot

from evolab import paths
from evolab.app.lab.canvas import SpectatorCanvas
from evolab.engine import seeds
from evolab.engine.evaluate import Trajectory, evaluate
from evolab.policy.genome import layout_for
from evolab.spec.io import load_env
from evolab.world.physics2d.episode import BodyGeom, Episode, ShapeGeom, WorldGeometry

W, H = 1280, 720
DT = 1.0 / 60.0


@pytest.fixture(scope="module")
def lander() -> tuple[object, Episode]:
    spec = load_env(paths.builtin_env_dir() / "lander.evoenv")
    return spec, Episode(spec, None, seeds.train_trials(1, 0, 1)[0])


@pytest.fixture(scope="module")
def recorded(lander: tuple[object, Episode]) -> list[Trajectory]:
    """실제 착륙선 궤적 32개(무작위 유전자)."""
    spec, ep = lander
    layout = layout_for(spec)
    rng = np.random.default_rng(11)
    genomes = 0.3 * rng.standard_normal((32, layout.dim))
    trial = seeds.train_trials(1, 0, 1)[0]
    res = evaluate(spec, layout, genomes, [(i, trial) for i in range(32)], record=[True] * 32, every=2)
    assert res.trajectories is not None
    return [t for t in res.trajectories if t is not None]


def start_poses(world: WorldGeometry) -> np.ndarray:
    return np.array([b.pose for b in world.bodies if b.moving], dtype=np.float32)


def synthetic(world: WorldGeometry, offsets: list[tuple[float, float]], *, every: int = 2, dt: float = DT,
              end_step: int | None = None, outputs: np.ndarray | None = None) -> Trajectory:
    """시작 자세를 프레임마다 (dx, dy)만큼 옮긴 궤적."""
    base = start_poses(world)
    poses = np.stack([base + np.array([dx, dy, 0.0], dtype=np.float32) for dx, dy in offsets])
    u = outputs if outputs is not None else np.zeros((len(offsets), 3), dtype=np.float32)
    steps = end_step if end_step is not None else (len(offsets) - 1) * every
    return Trajectory(poses, u.astype(np.float32), every, dt, steps)


def make_canvas(qtbot: QtBot, ep: Episode, show: bool = False) -> SpectatorCanvas:
    c = SpectatorCanvas()
    qtbot.addWidget(c)
    c.resize(W, H)
    if show:
        c.show()
        qtbot.waitExposed(c)
    c.set_scene(ep.geometry(), channels=ep.actuator_channels())
    return c


def render(c: SpectatorCanvas) -> QImage:
    img = QImage(c.width(), c.height(), QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    c.render(img)
    return img


def hull_xy(tr: Trajectory, k: int) -> tuple[float, float]:
    return float(tr.poses[k, 0, 0]), float(tr.poses[k, 0, 1])


def settle(c: SpectatorCanvas, t: float, n: int = 120) -> None:
    for _ in range(n):
        c.set_time(t)


# ---------------------------------------------------------------------------


def test_scene_draws_terrain_and_pad_without_population(qtbot: QtBot, lander) -> None:  # noqa: ANN001
    _, ep = lander
    c = make_canvas(qtbot, ep)
    assert c.duration() == 0.0 and c.agent_frame(0) is None
    img = render(c)
    g = ep.geometry()
    pad = next(r for r in g.regions if r.kind == "pad")
    (x0, y0), (x1, _) = pad.points[0], pad.points[1]
    pt = c.world_to_screen((x0 + x1) / 2.0, y0)
    assert 0 <= pt.x() < W and 0 <= pt.y() < H
    color = img.pixelColor(round(pt.x()), round(pt.y()))
    assert color.green() > 150 and color.red() < 120  # 패드 선(초록)
    below = img.pixelColor(round(pt.x()), min(H - 1, round(pt.y()) + 40))
    assert below != QColor("#0b0e14")  # 지형 채우기
    # 장면이 없으면 안내 문구만 그린다
    empty = SpectatorCanvas()
    qtbot.addWidget(empty)
    empty.resize(400, 300)
    render(empty)


def test_population_layers_ranks_and_champion(qtbot: QtBot, lander, recorded) -> None:  # noqa: ANN001
    _, ep = lander
    c = make_canvas(qtbot, ep)
    trs = recorded[:20]
    ranks = np.arange(20)[::-1].copy()  # 마지막 개체가 1등
    c.set_population(trs, ranks=ranks, champion=19, labels={19: "#1"})
    layers = dict(c._layers)
    names = {c._agents[i].index: name for name, slots in c._layers for i in slots}
    assert names[19] == "champion"
    assert [i for i, n in names.items() if n == "top"] == [18]  # 20개의 10% = 2개(챔피언 포함)
    assert len(layers["rest"]) == 18
    # 나머지는 나쁜 순위부터 그린다(좋은 개체가 위에)
    rest_ranks = [c._agents[i].rank for i in layers["rest"]]
    assert rest_ranks == sorted(rest_ranks, reverse=True)
    assert c.duration() == pytest.approx(max(t.duration for t in trs))
    c.set_time(1.0)
    img = render(c)
    x, y = hull_xy(trs[19], c.agent_frame(19)[0])
    pt = c.world_to_screen(x, y)
    col = img.pixelColor(round(pt.x()), round(pt.y()))
    assert col.red() > 200 and col.green() > 150 and col.blue() < 160  # 챔피언 강조색(노랑)
    with pytest.raises(ValueError):
        c.set_population(trs, ranks=np.arange(3))
    with pytest.raises(ValueError):
        c.set_population(trs, champion=20)


def test_set_time_picks_last_frame_not_after_t(qtbot: QtBot, lander) -> None:  # noqa: ANN001
    _, ep = lander
    c = make_canvas(qtbot, ep)
    g = ep.geometry()
    # every=2, dt=0.1: 프레임 시각 0, .2, .4, .6, .8 그리고 끝(9스텝) .9
    short = synthetic(g, [(0.0, -i * 0.1) for i in range(6)], dt=0.1, end_step=9)
    assert np.allclose(short.times(), [0, 0.2, 0.4, 0.6, 0.8, 0.9])
    long = synthetic(g, [(1.0, 0.0)] * 11, dt=0.1)  # 끝 2.0초
    c.set_population([short, long])
    assert c.duration() == pytest.approx(2.0)
    expect = {0.0: (0, False), 0.19: (0, False), 0.2: (1, False), 0.5: (2, False), 0.85: (4, False),
              0.9: (5, True), 1.5: (5, True), 100.0: (5, True)}
    for t, want in expect.items():
        c.set_time(t)
        assert c.agent_frame(0) == want, t
    c.set_time(1.5)
    assert c.agent_frame(1) == (7, False)
    c.set_time(2.0)
    assert c.agent_frame(1) == (10, True)
    # 그린 자세 = 그 프레임 자세(끝난 개체는 마지막 자세에 머문다)
    c.set_time(50.0)
    assert np.allclose(c._poses[:3], short.poses[-1])


def test_set_time_matches_trajectory_times_on_real_runs(qtbot: QtBot, lander, recorded) -> None:  # noqa: ANN001
    _, ep = lander
    c = make_canvas(qtbot, ep)
    trs = recorded[:4]
    c.set_population(trs)
    for i, tr in enumerate(trs):
        times = tr.times()
        for k in list(range(0, tr.frames, 17)) + [tr.frames - 2, tr.frames - 1]:
            c.set_time(float(times[k]))
            assert c.agent_frame(i) == (k, k == tr.frames - 1)
            if k + 1 < tr.frames:
                c.set_time(float((times[k] + times[k + 1]) / 2.0))
                assert c.agent_frame(i) == (k, False)


def test_none_trajectories_are_skipped(qtbot: QtBot, lander, recorded) -> None:  # noqa: ANN001
    _, ep = lander
    c = make_canvas(qtbot, ep)
    trs: list[Trajectory | None] = [None, recorded[0], None, recorded[1]]
    c.set_population(trs, ranks=np.array([0, 1, 2, 3]), champion=0)
    assert c.agent_frame(0) is None and c.agent_frame(2) is None
    assert c.agent_frame(1) == (0, False) and c.agent_frame(3) == (0, False)
    assert c._champion_slot is None  # 궤적 없는 챔피언은 그리지 않는다
    c.set_time(0.5)
    render(c)


def test_empty_population(qtbot: QtBot, lander) -> None:  # noqa: ANN001
    _, ep = lander
    c = make_canvas(qtbot, ep, show=True)
    c.set_population([])
    c.set_time(3.0)
    assert c.duration() == 0.0
    render(c)
    with qtbot.assertNotEmitted(c.individual_clicked):
        qtbot.mouseClick(c, Qt.MouseButton.LeftButton, pos=QPoint(W // 2, H // 2))
    assert c.hit_test(QPointF(W / 2, H / 2)) is None


def test_click_picks_topmost_individual(qtbot: QtBot, lander) -> None:  # noqa: ANN001
    _, ep = lander
    c = make_canvas(qtbot, ep, show=True)
    g = ep.geometry()
    a = synthetic(g, [(0.0, 0.0)] * 4)
    b = synthetic(g, [(8.0, 0.0)] * 4)
    same = synthetic(g, [(0.0, 0.0)] * 4)  # a와 같은 자리: 순위가 좋은 쪽이 위
    c.set_population([a, b, same], ranks=np.array([2, 0, 1]))
    c.set_time(0.0)

    def click(x: float, y: float) -> list[int]:
        pt = c.world_to_screen(x, y)
        with qtbot.waitSignal(c.individual_clicked, timeout=1000) as blocker:
            qtbot.mouseClick(c, Qt.MouseButton.LeftButton, pos=QPoint(round(pt.x()), round(pt.y())))
        return blocker.args

    assert click(*hull_xy(b, 0)) == [1]
    assert click(*hull_xy(a, 0)) == [2]
    # 몇 픽셀 밖도 잡힌다: 몸통 오른쪽 끝에서 2픽셀 바깥
    hull = next(bd for bd in g.bodies if bd.moving)
    right = max(p[0] for p in hull.shapes[0].points)
    x, y = hull_xy(b, 0)
    ang = float(b.poses[0, 0, 2])
    edge = c.world_to_screen(x + right * np.cos(ang), y + right * np.sin(ang))
    pt = QPointF(edge.x() + 2.0 * np.cos(ang), edge.y() - 2.0 * np.sin(ang))
    assert c.hit_test(pt) == ("population", 1)
    # 아무것도 없는 곳
    far = c.world_to_screen(x + 4.0, y + 6.0)
    assert c.hit_test(far) is None


def test_individual_shapes_are_used_for_picking(qtbot: QtBot, lander) -> None:  # noqa: ANN001
    _, ep = lander
    g = ep.geometry()
    moving = [bd for bd in g.bodies if bd.moving]
    big_hull = replace(moving[0], shapes=[ShapeGeom("poly", [(-4.0, -4.0), (4.0, -4.0), (4.0, 4.0), (-4.0, 4.0)])])
    agents = [moving, [big_hull] + moving[1:]]
    c = SpectatorCanvas()
    qtbot.addWidget(c)
    c.resize(W, H)
    c.set_scene(g, agents, channels=ep.actuator_channels())
    t0 = synthetic(g, [(0.0, 0.0)] * 2)
    c.set_population([t0, t0], ranks=np.array([0, 1]))
    x, y = hull_xy(t0, 0)
    # 몸통 중심에서 3m 떨어진 점: 보통 몸통 밖, 큰 몸통 안
    assert c.hit_test(c.world_to_screen(x, y + 3.0)) == ("population", 1)
    assert c.hit_test(c.world_to_screen(x, y)) == ("population", 0)  # 0번이 순위가 좋아 위에 있다
    with pytest.raises(ValueError):
        c.set_population([Trajectory(np.zeros((2, 2, 3), np.float32), np.zeros((2, 3), np.float32), 2, DT, 2)])


def test_overlay_items(qtbot: QtBot, lander, recorded) -> None:  # noqa: ANN001
    _, ep = lander
    c = make_canvas(qtbot, ep, show=True)
    g = ep.geometry()
    bodies = [bd for bd in g.bodies if bd.moving]
    first = synthetic(g, [(-6.0, 0.0)] * 30)
    now = recorded[0]
    c.set_population(recorded[:5], ranks=np.arange(5))
    c.set_overlay([(first, bodies, QColor("#5e81ac"), "first"), (now, bodies, QColor("#bf616a"), "now")])
    assert c.duration() == pytest.approx(max(first.duration, *(t.duration for t in recorded[:5])))
    assert [t for _, t in c._legend_entries()][-2:] == ["first", "now"]
    c.set_time(0.2)
    assert c.overlay_frame(0) == (6, False)
    render(c)
    pt = c.world_to_screen(*hull_xy(first, 0))
    with qtbot.waitSignal(c.overlay_clicked, timeout=1000) as blocker:
        qtbot.mouseClick(c, Qt.MouseButton.LeftButton, pos=QPoint(round(pt.x()), round(pt.y())))
    assert blocker.args == [0]
    c.set_overlay([])
    assert c.overlay_frame(0) is None
    assert c.duration() == pytest.approx(max(t.duration for t in recorded[:5]))


def test_champion_flames_and_trail_change_the_picture(qtbot: QtBot, lander) -> None:  # noqa: ANN001
    _, ep = lander
    c = make_canvas(qtbot, ep)
    g = ep.geometry()
    path = [(0.0, -0.05 * i) for i in range(40)]
    off = synthetic(g, path)
    on = synthetic(g, path, outputs=np.tile(np.array([1.0, 0.0, 0.0], np.float32), (40, 1)))
    c.set_population([off], champion=0)
    settle(c, 1.0)
    a = render(c)
    c.set_population([on], champion=0)
    settle(c, 1.0)
    b = render(c)
    assert a != b  # 주 추진기 불꽃
    # 지나온 궤적: 시작점 바로 아래(개체는 이미 한참 내려가 있다)가 강조색 선
    start = c.world_to_screen(*hull_xy(on, 0))
    now = c.world_to_screen(*hull_xy(on, c.agent_frame(0)[0]))
    assert now.y() - start.y() > 20
    col = b.pixelColor(round(start.x()), round(start.y() + 3))
    assert col.red() > 120 and col.blue() < col.red()


def test_follow_modes(qtbot: QtBot, lander) -> None:  # noqa: ANN001
    _, ep = lander
    c = make_canvas(qtbot, ep)
    g = ep.geometry()
    spread = [synthetic(g, [(dx * i / 59, 0.0) for i in range(60)]) for dx in (-40.0, -5.0, 0.0, 10.0, 25.0)]
    c.set_population(spread, ranks=np.arange(5), champion=4)
    scales = {}
    for mode in ("fit", "champion"):
        c.set_follow(mode)
        for t in np.linspace(0.0, 2.0, 120):
            c.set_time(float(t))
        settle(c, 2.0)
        pts = [c.world_to_screen(*hull_xy(tr, tr.frames - 1)) for tr in spread]
        scales[mode] = c.camera()[1]
        if mode == "fit":
            assert all(0 <= p.x() <= W and 0 <= p.y() <= H for p in pts)
            pad = next(r for r in g.regions if r.kind == "pad")
            assert all(0 <= q.x() <= W and 0 <= q.y() <= H for q in (c.world_to_screen(*pt) for pt in pad.points))
        else:
            champ = pts[4]
            assert abs(champ.x() - W / 2) < W / 2 - 40 and 0 < champ.y() < H
            assert c.camera()[1] >= 4.0
            assert not (0 <= pts[0].x() <= W)  # 맨 왼쪽 개체는 화면 밖
    assert scales["champion"] > 1.3 * scales["fit"]
    with pytest.raises(ValueError):
        c.set_follow("nobody")
    # 휠 확대
    s0 = c.camera()[1]
    c.wheelEvent(_wheel(c, 120))
    assert c.camera()[1] > s0


def _wheel(c: SpectatorCanvas, delta: int) -> QWheelEvent:
    pos = QPointF(c.width() / 2, c.height() / 2)
    return QWheelEvent(pos, c.mapToGlobal(pos), QPoint(0, 0), QPoint(0, delta), Qt.MouseButton.NoButton,
                       Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)


def test_hud_and_new_scene_clears(qtbot: QtBot, lander, recorded) -> None:  # noqa: ANN001
    _, ep = lander
    c = make_canvas(qtbot, ep)
    c.set_population(recorded[:3])
    c.set_hud("gen 3\nx2")
    render(c)
    c.set_scene(ep.geometry(), channels=ep.actuator_channels())
    assert c.duration() == 0.0 and c.agent_frame(0) is None


def test_render_speed_256_landers(qtbot: QtBot, lander, recorded) -> None:  # noqa: ANN001
    """명세: 관전 모드에서 개체 256개까지 60fps. CI 여유를 두고 50ms 아래인지만 본다."""
    _, ep = lander
    c = make_canvas(qtbot, ep)
    trs = (recorded * 8)[:256]
    scores = np.array([-t.duration for t in trs])
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(256, dtype=np.int64)
    ranks[order] = np.arange(256)
    c.set_population(trs, ranks=ranks, champion=int(order[0]), labels={int(order[0]): "#1"})
    c.set_hud("gen 1")
    img = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)
    c.set_time(0.0)
    c.render(img)
    frames = 60
    start = time.perf_counter()
    for f in range(frames):
        c.set_time(f / 60.0)
        c.render(img)
    ms = (time.perf_counter() - start) / frames * 1000.0
    print(f"\nspectator canvas: 256 landers, {W}x{H}: {ms:.2f} ms/frame")
    assert ms < 50.0


def test_broken_physics_frames_do_not_break_the_camera(qtbot: QtBot, lander) -> None:  # noqa: ANN001
    _, ep = lander
    c = make_canvas(qtbot, ep)
    g = ep.geometry()
    bad = synthetic(g, [(0.0, 0.0)] * 5)
    bad.poses[3:] = np.nan
    good = synthetic(g, [(2.0, 0.0)] * 5)
    c.set_population([bad, good], ranks=np.array([0, 1]), champion=0)
    for t in (0.0, 0.1, 1.0):
        settle(c, t, 5)
        (cx, cy), s = c.camera()
        assert np.isfinite([cx, cy, s]).all()
        render(c)

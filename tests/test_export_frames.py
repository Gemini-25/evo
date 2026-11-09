"""내보내기 프레임 그리기 (v3 설계 4.15, 8.1 T39: export/frames.py).

- 고정 카메라 상자(fixed_view)가 모든 칸 궤적의 모든 물체(모양 반지름까지)와 세계 영역 꼭짓점을 담고, 8% 여유·짧은 변 8m 이상,
  첫 칸에 기본 보기가 있으면 그 상자.
- T39: 언덕차가 30초 동안 169m를 달리는 재생에서 고정 카메라는 이웃 프레임 사이에 바뀐 16×16 블록 수의 중앙값이 전체의 5% 이하
  (따라가기는 그보다 크다: 내장 MP4 크기를 카메라가 정한다, 설계 D50).
- 「따라가기」 GIF 작업의 프레임과 지금 GIF 길(export.video.render_frames·render_race_frames)의 프레임이 v3.0(e5b65f7)
  video._iter_frames·_iter_race_frames 몸통을 얼려 둔 사본(아래 _v30_*)의 그림과 화소까지 같다. 지금 두 길은 모두 frames.py
  위에 있어 서로 비교만 하면 frames.py가 바뀌어도 잡지 못한다(v3.1 검토).
- 프레임 수 = frame_count, 크기 = job.size, 움직이는 세계 물체(점프 게임 장애물 줄)를 그린다, v2.1 없는 물체 재생도 그린다.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QColor, QImage, QPainter

from evolab import paths
from evolab.app import strings as S
from evolab.app.export import frames as F
from evolab.app.export import job as J
from evolab.app.export import video
from evolab.app.export.h264_pcm import frame_pixels
from evolab.engine import seeds
from evolab.engine.evaluate import Trajectory
from evolab.engine.replay import Replay, replay
from evolab.policy.genome import layout_for
from evolab.spec.io import load_env

_CACHE: dict[tuple[str, int], Replay] = {}


def _replay(env: str, seed: int, scale: float = 1.0) -> Replay:
    key = (env, seed)
    if key not in _CACHE:
        spec = load_env(paths.builtin_env_dir() / f"{env}.evoenv")
        layout = layout_for(spec)
        genome = np.random.default_rng(seed).normal(0.0, scale, layout.dim)
        _CACHE[key] = replay(spec, layout, genome, seeds.display_trial())
    return _CACHE[key]


def _job(items: list[tuple[Replay, str, str]], fmt: str = "mp4", *, camera: str = "fixed", size: str = "small",
         view: J.ViewBox | None = None) -> J.ExportJob:
    return J.job_from_replays(items, fmt, size=size, camera=camera, title="t", meta={}, view=view)


def _radius(body: Any) -> float:
    r = 0.0
    for s in body.shapes:
        for x, y in s.points:
            r = max(r, math.hypot(x, y) + (s.radius if s.kind != "poly" else 0.0))
    return r


def _changed_blocks(a: np.ndarray, b: np.ndarray) -> int:
    d = a != b
    h, w = d.shape
    ph, pw = -(-h // 16) * 16, -(-w // 16) * 16
    pad = np.zeros((ph, pw), dtype=bool)
    pad[:h, :w] = d
    return int(pad.reshape(ph // 16, 16, pw // 16, 16).any(axis=(1, 3)).sum())


# ---------------------------------------------------------------------------
# 고정 카메라 상자
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env,seeds_", [("lander", (1, 2)), ("track_car", (3,)), ("hill_car", (2, 3))])
def test_fixed_view_holds_every_trajectory_point_and_region(env: str, seeds_: tuple[int, ...]) -> None:
    reps = [_replay(env, s) for s in seeds_]
    items = [J.ExportItem(r, "#ff7828", str(k)) for k, r in enumerate(reps)]
    x0, y0, x1, y1 = F.fixed_view(items)
    inner = []  # 여유 없이 담아야 할 범위
    for r in reps:
        radius = max(_radius(b) for b in r.agent)
        xy = r.trajectory.poses[..., :2].reshape(-1, 2).astype(np.float64)
        inner.append(np.r_[xy.min(axis=0) - radius, xy.max(axis=0) + radius])
        for region in r.geometry.regions:
            p = np.asarray(region.points, dtype=np.float64)
            inner.append(np.r_[p.min(axis=0), p.max(axis=0)])
        # 관전 캔버스가 상자 안이라고 보는 조건(물체마다 자세 ± 모양 반지름)과 같은 검사
        for k, body in enumerate(r.agent):
            p = r.trajectory.poses[:, k, :2].astype(np.float64)
            rad = _radius(body)
            assert (p[:, 0] - rad >= x0).all() and (p[:, 0] + rad <= x1).all()
            assert (p[:, 1] - rad >= y0).all() and (p[:, 1] + rad <= y1).all()
    lo = np.min([b[:2] for b in inner], axis=0)
    hi = np.max([b[2:] for b in inner], axis=0)
    w, h = hi - lo
    # 여유 8%(짧으면 8m로 넓힌 쪽은 가운데)
    if w * (1 + 2 * F.FIXED_MARGIN) >= F.FIXED_MIN_SIZE:
        assert x0 == pytest.approx(lo[0] - w * F.FIXED_MARGIN) and x1 == pytest.approx(hi[0] + w * F.FIXED_MARGIN)
    else:
        assert x1 - x0 == pytest.approx(F.FIXED_MIN_SIZE)
    assert y1 - y0 >= F.FIXED_MIN_SIZE - 1e-9 and x1 - x0 >= F.FIXED_MIN_SIZE - 1e-9
    assert y0 <= lo[1] and y1 >= hi[1]


def test_fixed_view_prefers_the_first_items_default_view() -> None:
    a, b = _replay("jump", 1), _replay("lander", 1)
    box = (-3.0, -1.0, 22.0, 6.0)
    assert F.fixed_view([J.ExportItem(a, "#ff7828", "a", box), J.ExportItem(b, "#508cff", "b")]) == box
    # 첫 칸에 보기가 없으면 뒤 칸의 보기는 쓰지 않고 궤적 상자
    got = F.fixed_view([J.ExportItem(b, "#508cff", "b"), J.ExportItem(a, "#ff7828", "a", box)])
    assert got != box


def test_fixed_view_min_size_and_exploded_frames() -> None:
    rep = _replay("lander", 1)
    still = np.repeat(rep.trajectory.poses[:1], 5, axis=0)
    tiny = dataclasses.replace(rep, trajectory=Trajectory(still, rep.trajectory.outputs[:5], rep.trajectory.every,
                                                         rep.trajectory.dt))
    x0, y0, x1, y1 = F.fixed_view([J.ExportItem(tiny, "#ff7828", "a")], margin=0.0)
    # 착륙선 패드 영역까지 담으므로 가로는 넓고, 세로는 8m 이상
    assert y1 - y0 >= F.FIXED_MIN_SIZE - 1e-9 and x1 - x0 >= F.FIXED_MIN_SIZE - 1e-9
    poses = still.copy()
    poses[2, 0, 0] = np.nan  # 물리가 터진 프레임: 관전 캔버스는 원점에 그린다
    poses[3, 1, 1] = np.inf
    bad = dataclasses.replace(rep, trajectory=Trajectory(poses, rep.trajectory.outputs[:5], rep.trajectory.every,
                                                        rep.trajectory.dt))
    bx0, by0, bx1, by1 = F.fixed_view([J.ExportItem(bad, "#ff7828", "a")])
    assert all(math.isfinite(v) for v in (bx0, by0, bx1, by1))
    assert bx0 <= 0.0 <= bx1 and by0 <= 0.0 <= by1
    with pytest.raises(ValueError):
        F.fixed_view([])


# ---------------------------------------------------------------------------
# T39: 고정 카메라는 바뀌는 블록이 적다
# ---------------------------------------------------------------------------
def _block_medians(job: J.ExportJob) -> tuple[float, int]:
    prev: np.ndarray | None = None
    counts = []
    n = 0
    for img in F.iter_frames(job):
        px, _keep = frame_pixels(img)
        px = px.copy()
        if prev is not None:
            counts.append(_changed_blocks(prev, px))
        prev = px
        n += 1
    total = (-(-job.size[0] // 16)) * (-(-job.size[1] // 16))
    return float(np.median(counts)) / total, n


def test_t39_fixed_camera_changes_few_blocks_on_a_driving_hill_car(qtbot: Any) -> None:
    rep = _replay("hill_car", 2)  # 시드 2의 무작위 유전자: 30초 동안 169m를 달린다(측정)
    xs = rep.trajectory.poses[:, 0, 0]
    assert rep.trajectory.duration > 25 and float(xs.max() - xs.min()) > 100
    fixed = _job([(rep, J.NOW_COLOR, "c")], camera="fixed")
    follow = _job([(rep, J.NOW_COLOR, "c")], camera="follow")
    fixed_share, n = _block_medians(fixed)
    follow_share, _ = _block_medians(follow)
    assert n == F.frame_count(fixed)
    assert fixed_share <= 0.05, fixed_share  # 측정: 0.43%
    assert follow_share > fixed_share  # 측정: 따라가기 11.2%


# ---------------------------------------------------------------------------
# 프레임 수·크기·v3.0 GIF와 같은 그림
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fmt,size,want", [("mp4", "small", (640, 360)), ("mp4", "medium", (1280, 720)),
                                           ("gif", "small", (640, 400)), ("html", "medium", (640, 400))])
def test_overlay_frames_count_and_size(qtbot: Any, fmt: str, size: str, want: tuple[int, int]) -> None:
    rep = _replay("hill_car", 1)  # 3초
    job = _job([(rep, J.OLD_COLOR, "a"), (rep, J.NOW_COLOR, "b")], fmt, size=size)
    imgs = list(F.iter_frames(job))
    assert len(imgs) == F.frame_count(job) == math.ceil((rep.trajectory.duration + 0.5) * job.fps) + 1
    assert all((im.width(), im.height()) == want and im.format() == QImage.Format.Format_RGB32 for im in imgs)


def test_lane_frames_count_size_and_fixed_boxes(qtbot: Any) -> None:
    lanes = [(_replay("lander", 1), "#3987e5", "a"), (_replay("track_car", 3), "#d95926", "b"),
             (_replay("jump", 1), "#199e70", "c")]
    from evolab.app.race.model import lane_grid

    for fmt, size in (("mp4", "small"), ("gif", "medium")):
        job = J.job_from_race(lanes, lane_grid(len(lanes))[1], fmt, size=size, camera="fixed", title="경주")
        rows, cols, pw, ph = J.lane_layout(job)
        imgs = list(F.iter_frames(job))
        longest = max(r.trajectory.duration for r, _c, _l in lanes)
        assert len(imgs) == F.frame_count(job) == math.ceil((longest + 1.0) * job.fps) + 1
        assert all((im.width(), im.height()) == job.size == (cols * pw, rows * ph) for im in imgs)


# ---- v3.0(e5b65f7) export/video.py의 _iter_frames·_iter_race_frames 몸통을 그대로 얼려 둔 사본(바꾸지 말 것) ----
_V30_SIZE = (640, 400)
_V30_FPS = 20
_V30_PANE = (320, 200)
_V30_RACE_TAIL = 1.0
_V30_RACE_BAR = 3
_V30_RACE_SEPARATOR = QColor(255, 255, 255, 40)


def _v30_iter_frames(items: list[tuple[Replay, QColor, str]], *, size: tuple[int, int] = _V30_SIZE,
                     fps: int = _V30_FPS, speed: float = 1.0, title: str = "") -> Any:
    from evolab.app.lab.canvas import SpectatorCanvas

    if not items:
        return
    canvas = SpectatorCanvas()
    try:
        canvas.resize(*size)
        first = items[0][0]
        canvas.set_scene(first.geometry, None, channels=first.channels)
        canvas.set_overlay([(r.trajectory, r.agent, color, label) for r, color, label in items])
        canvas.set_follow("fit")
        duration = canvas.duration() + 0.5
        count = max(1, math.ceil(duration * fps / max(speed, 1e-6)) + 1)
        for k in range(count):
            t = min(duration, k * speed / fps)
            canvas.set_time(t)
            canvas.set_hud(S.VIDEO_HUD.format(title=title, t=t) if title else S.VIDEO_HUD_TIME.format(t=t))
            img = QImage(size[0], size[1], QImage.Format.Format_RGB32)
            canvas.render(img)
            yield k, count, img
    finally:
        canvas.deleteLater()


def _v30_iter_race_frames(lanes: list[tuple[Replay, QColor, str]], cols: int, *,
                          pane: tuple[int, int] = _V30_PANE, fps: int = _V30_FPS, title: str = "") -> Any:
    from evolab.app.lab.canvas import COLOR_BG, SpectatorCanvas

    if not lanes:
        return
    cols = max(1, min(int(cols), len(lanes)))
    rows = math.ceil(len(lanes) / cols)
    pw, ph = int(pane[0]), int(pane[1])
    canvases: list[SpectatorCanvas] = []
    try:
        for rep, color, _label in lanes:
            c = SpectatorCanvas()
            c.setMinimumSize(1, 1)
            c.resize(pw, ph)
            c.set_scene(rep.geometry, None, channels=rep.channels)
            c.set_overlay([(rep.trajectory, rep.agent, QColor(color), "")])
            c.set_follow("champion")
            canvases.append(c)
        longest = max(rep.trajectory.duration for rep, _c, _l in lanes)
        end = longest + _V30_RACE_TAIL
        count = max(1, math.ceil((float(longest) + _V30_RACE_TAIL) * fps) + 1)
        for k in range(count):
            t = min(end, k / fps)
            img = QImage(cols * pw, rows * ph, QImage.Format.Format_RGB32)
            img.fill(COLOR_BG)
            for i, c in enumerate(canvases):
                label = lanes[i][2]
                hud = S.VIDEO_HUD.format(title=label, t=t) if label else S.VIDEO_HUD_TIME.format(t=t)
                c.set_time(t)
                c.set_hud(f"{title}\n{hud}" if i == 0 and title else hud)
                c.render(img, QPoint((i % cols) * pw, (i // cols) * ph))
            p = QPainter(img)
            for i, (_rep, color, _label) in enumerate(lanes):
                p.fillRect(QRect((i % cols) * pw, (i // cols) * ph, pw, _V30_RACE_BAR), QColor(color))
            p.setPen(_V30_RACE_SEPARATOR)
            for col in range(1, cols):
                p.drawLine(col * pw, 0, col * pw, rows * ph)
            for row in range(1, rows):
                p.drawLine(0, row * ph, cols * pw, row * ph)
            p.end()
            yield k, count, img
    finally:
        for c in canvases:
            c.deleteLater()


def _same_frames(got: list[QImage], want: list[QImage]) -> None:
    assert len(got) == len(want)
    for k, (x, y) in enumerate(zip(got, want, strict=True)):
        assert (x.width(), x.height()) == (y.width(), y.height()), k
        assert x == y, f"frame {k} differs from the v3.0 frame"


def test_follow_gif_jobs_draw_the_v30_gif_frames(qtbot: Any) -> None:
    """「따라가기」 GIF 작업과 지금 GIF 길(video.render_frames·render_race_frames) = 얼려 둔 v3.0 몸통의 그림(화소까지)."""
    a, b = _replay("lander", 1), _replay("jump", 1)
    items = [(a, QColor(J.OLD_COLOR), "1세대"), (a, QColor(J.NOW_COLOR), "지금")]
    want = [img for _k, _n, img in _v30_iter_frames(items, title="t")]
    job = _job([(r, c.name(), label) for r, c, label in items], "gif", camera="follow")
    _same_frames(list(F.iter_frames(job)), want)
    _same_frames(video.render_frames(items, title="t"), want)
    lanes = [(a, QColor("#3987e5"), "a"), (b, QColor("#d95926"), "b")]
    want = [img for _k, _n, img in _v30_iter_race_frames(lanes, 2, title="경주")]
    job = J.job_from_race([(r, c.name(), label) for r, c, label in lanes], 2, "gif", size="small", camera="follow",
                          title="경주")
    _same_frames(list(F.iter_frames(job)), want)
    _same_frames(video.render_race_frames(lanes, 2, title="경주"), want)


def _rgba(img: QImage) -> np.ndarray:
    px, _keep = frame_pixels(img)
    return px.copy().view(np.uint8).reshape(img.height(), img.width(), 4).astype(np.int16)


def test_last_frame_is_the_last_fixed_frame(qtbot: Any) -> None:
    """끝 장면 그림(폰 재생 파일) = 고정 카메라로 끝까지 그린 마지막 프레임. 정적 세계를 그려 둔 그림을 다시 쓰는지에 따라
    가장자리 몇 화소의 안티앨리어싱이 1 다를 수 있다(측정: 256,000화소 중 11개, 차이 1)."""
    rep = _replay("lander", 2)
    job = _job([(rep, J.NOW_COLOR, "a")], "html")
    *_rest, last = F.iter_frames(dataclasses.replace(job, fmt="gif"))
    diff = np.abs(_rgba(F.last_frame(job)) - _rgba(last)).max(axis=2)
    assert diff.max() <= 2 and (diff > 0).mean() < 1e-3
    follow = dataclasses.replace(job, camera="follow")
    img = F.last_frame(follow)
    assert (img.width(), img.height()) == job.size


def test_moving_world_objects_and_absent_bodies_are_drawn(qtbot: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.lab.canvas import SpectatorCanvas

    seen: list[list[tuple[float, float, float]]] = []
    real = SpectatorCanvas._draw_movers

    def spy(self: SpectatorCanvas, p: Any, base: Any) -> None:
        real(self, p, base)
        seen.append(list(self.mover_poses))

    monkeypatch.setattr(SpectatorCanvas, "_draw_movers", spy)
    jump = _replay("jump", 1)
    assert any(b.motion is not None for b in jump.geometry.bodies)
    job = _job([(jump, J.NOW_COLOR, "j")], "gif", view=(-3.0, -1.0, 22.0, 6.0))
    n = sum(1 for _ in F.iter_frames(job))
    assert len(seen) == n and len({tuple(p) for poses in seen for p in poses}) > 1  # 장애물 줄이 시각마다 움직인다
    # v2.1 없는 물체(바퀴 고르는 언덕차): 재생의 물체 목록에서 이미 빠져 있다
    wheels = _replay("wheels", 1)  # 몸 유전자가 바퀴 자리 넷 중 셋을 끈 개체(차체와 바퀴 3만 있음)
    assert [b.id for b in wheels.agent] == ["chassis", "wheel3"]
    imgs = list(F.iter_frames(_job([(wheels, J.NOW_COLOR, "w")], "mp4")))
    assert len(imgs) == F.frame_count(_job([(wheels, J.NOW_COLOR, "w")], "mp4"))
    assert len(wheels.agent) == wheels.trajectory.poses.shape[1]

"""내보내기 프레임 그리기 (v3 설계 4.14·4.15, 7.3 「app/export/frames.py (F)」).

내보내기 작업(ExportJob)의 재생을 관전 캔버스(lab.canvas.SpectatorCanvas)로 한 프레임씩 그린다. MP4·GIF 작성기와 폰 재생
파일의 끝 장면 그림이 모두 이 그림을 쓴다. 화면 프로세스의 옛 GIF 길(export.video의 export_gif·export_race_gif·render_frames)도
이 모듈의 같은 생성기 위에 있는 얇은 껍데기라, 「따라가기」 카메라 GIF는 v3.0과 같은 그림이다.

- 겹쳐 보기(overlay): 한 세계에 모든 재생을 겹쳐(칸 색, 범례 이름, 지나온 궤적) 그린다. 시간은 0초부터 가장 긴 재생이 끝난 뒤
  0.5초까지, 1/fps 간격(v3.0 GIF와 같음).
- 경주 칸(lanes): 칸마다 관전 캔버스 하나, 그림 하나의 격자 자리(lane_layout)에 칸 색 띠와 칸 사이 선. 끝난 뒤 1초 더.
- 카메라(설계 D50·4.15): 「따라가기」는 v3.0 GIF와 같다(겹쳐 보기는 모든 개체가 보이게 맞추는 캔버스 fit, 경주 칸은 칸 개체를
  따라가는 champion). 「고정」은 fixed_view 상자를 캔버스의 기본 보기(set_scene(view=상자))로 주고 fit으로 둔다: 개체가 상자
  안에 있는 동안 카메라가 멈추므로 프레임 사이에 바뀌는 블록이 적어 내장 MP4가 작다(캔버스는 고치지 않는다). 경주 칸은 칸마다
  그 칸의 상자.

QApplication이 있는 스레드에서만 부른다(관전 캔버스가 QWidget). Qt는 맨 위에서 불러오지만 이 모듈을 불러오는 곳(worker.py,
video.py)은 그림을 그릴 때만 불러온다.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QColor, QImage, QPainter

from evolab.app import strings as S
from evolab.app.export.job import ExportItem, ExportJob, ViewBox, lane_layout

if TYPE_CHECKING:
    from evolab.engine.replay import Replay
    from evolab.world.physics2d.episode import BodyGeom

OVERLAY_TAIL = 0.5
"""겹쳐 보기에서 가장 긴 재생이 끝난 뒤 더 보이는 시간(초, v3.0 GIF와 같음)."""
RACE_TAIL = 1.0
"""경주 칸에서 가장 긴 칸이 끝난 뒤 더 보이는 시간(초, v3.0 경주 GIF와 같음)."""
RACE_BAR = 3
"""경주 칸 위쪽의 칸 색 띠 두께(화소)."""
RACE_SEPARATOR = QColor(255, 255, 255, 40)
FIXED_MARGIN = 0.08
"""고정 카메라 상자의 양쪽 여유(가로·세로 길이의 비율, 설계 4.15)."""
FIXED_MIN_SIZE = 8.0
"""고정 카메라 상자의 가로·세로 최소 길이(m, 설계 4.15)."""

FOLLOW_OVERLAY = "fit"
"""「따라가기」 겹쳐 보기의 캔버스 카메라(v3.0 GIF: 모든 개체가 보이게 맞춤)."""
FOLLOW_LANES = "champion"
"""「따라가기」 경주 칸의 캔버스 카메라(v3.0 경주 GIF: 칸 개체를 따라감)."""


# ---------------------------------------------------------------------------
# 프레임 수와 고정 카메라
# ---------------------------------------------------------------------------
def overlay_count(duration: float, fps: int, speed: float = 1.0) -> int:
    """겹쳐 보기 프레임 수: ⌈(가장 긴 재생 + 0.5초) × fps / 배속⌉ + 1(0초 프레임 포함, v3.0 GIF와 같은 식)."""
    return max(1, math.ceil((float(duration) + OVERLAY_TAIL) * fps / max(speed, 1e-6)) + 1)


def race_count(duration: float, fps: int) -> int:
    """경주 칸 프레임 수: ⌈(가장 긴 재생 + 1초) × fps⌉ + 1(v3.0 video.race_frame_count와 같은 식)."""
    return max(1, math.ceil((float(duration) + RACE_TAIL) * fps) + 1)


def frame_count(job: ExportJob) -> int:
    """iter_frames(job)가 만드는 프레임 수(그리지 않고 셈: 대화 상자의 「길이·프레임 수」, 진행 메시지의 n)."""
    if job.layout == "lanes":
        return race_count(job.duration, job.fps)
    return overlay_count(job.duration, job.fps)


def _body_radius(body: BodyGeom) -> float:
    """물체 원점에서 가장 먼 모양 점까지(관전 캔버스 _build_shape의 radius와 같은 규칙)."""
    r = 0.0
    for s in body.shapes:
        if s.kind == "poly":
            r = max([r, *(math.hypot(float(x), float(y)) for x, y in s.points)])
        else:
            r = max([r, *(math.hypot(float(x), float(y)) + float(s.radius) for x, y in s.points)])
    return r


def fixed_view(items: Sequence[ExportItem], margin: float = FIXED_MARGIN,
               min_size: float = FIXED_MIN_SIZE) -> ViewBox:
    """고정 카메라 상자 (x0, y0, x1, y1)(설계 4.15).

    첫 칸에 기본 보기(환경 editor.view)가 있으면 그 상자. 없으면 모든 칸 궤적의 모든 물체 위치(모양 반지름 최댓값만큼 넓힘)와
    세계 영역 꼭짓점을 담는 상자에 양쪽 margin 비율 여유를 두고, 가로·세로가 min_size보다 짧으면 가운데를 두고 넓힌다. 물리가
    터진 프레임(NaN·무한)은 관전 캔버스처럼 원점으로 본다(캔버스가 그 자리에 그리므로 상자 안에 둔다).
    """
    rows = list(items)
    if not rows:
        raise ValueError(S.VIDEO_NO_FRAMES)
    if rows[0].view is not None:
        return rows[0].view
    lo = np.array([np.inf, np.inf])
    hi = -lo
    for it in rows:
        rep = it.replay
        poses = np.asarray(rep.trajectory.poses, dtype=np.float64)
        if poses.size:
            xy = poses[..., :2].reshape(-1, 2)
            good = np.isfinite(xy).all(axis=1)
            pts = xy[good]
            if not good.all():
                pts = np.vstack([pts, [[0.0, 0.0]]])
            radius = max((_body_radius(b) for b in rep.agent), default=0.0)
            if len(pts):
                lo = np.minimum(lo, pts.min(axis=0) - radius)
                hi = np.maximum(hi, pts.max(axis=0) + radius)
        for region in rep.geometry.regions:
            if region.points:
                p = np.asarray(region.points, dtype=np.float64).reshape(-1, 2)
                lo = np.minimum(lo, p.min(axis=0))
                hi = np.maximum(hi, p.max(axis=0))
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        bx0, by0, bx1, by1 = rows[0].replay.geometry.bounds
        lo, hi = np.array([bx0, by0], dtype=np.float64), np.array([bx1, by1], dtype=np.float64)
    x0, y0 = (float(v) for v in lo)
    x1, y1 = (float(v) for v in hi)
    w, h = x1 - x0, y1 - y0
    x0, x1, y0, y1 = x0 - w * margin, x1 + w * margin, y0 - h * margin, y1 + h * margin
    if x1 - x0 < min_size:
        c = (x0 + x1) / 2.0
        x0, x1 = c - min_size / 2.0, c + min_size / 2.0
    if y1 - y0 < min_size:
        c = (y0 + y1) / 2.0
        y0, y1 = c - min_size / 2.0, c + min_size / 2.0
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# 그리기
# ---------------------------------------------------------------------------
def iter_frames(job: ExportJob) -> Iterator[QImage]:
    """작업의 프레임(QImage RGB32, 크기 = job.size)을 하나씩 만든다(쌓아 두지 않음). 수 = frame_count(job)."""
    for _k, _n, img in numbered_frames(job):
        yield img


def numbered_frames(job: ExportJob, *, last_only: bool = False) -> Iterator[tuple[int, int, QImage]]:
    """(번호, 전체 수, 프레임). last_only면 마지막 프레임 하나만(폰 재생 파일의 끝 장면 그림)."""
    fixed = job.camera == "fixed"
    if job.layout == "lanes":
        rows, cols, pw, ph = lane_layout(job)
        views = [fixed_view([it]) for it in job.items] if fixed else [None] * len(job.items)
        yield from lane_frames([(it.replay, QColor(it.color), it.label) for it in job.items], cols, rows=rows,
                               pane=(pw, ph), size=job.size, fps=job.fps, title=job.title, views=views,
                               follow="fit" if fixed else FOLLOW_LANES, last_only=last_only)
        return
    yield from overlay_frames([(it.replay, QColor(it.color), it.label) for it in job.items], size=job.size,
                              fps=job.fps, title=job.title, view=fixed_view(job.items) if fixed else None,
                              follow=FOLLOW_OVERLAY, last_only=last_only)


def last_frame(job: ExportJob) -> QImage:
    """마지막 프레임 하나(끝 장면 그림). 카메라는 그 시각에 맞춘다(따라가기도 처음부터 따라온 것과 같은 자리로 바로)."""
    for _k, _n, img in numbered_frames(job, last_only=True):
        return img
    raise ValueError(S.VIDEO_NO_FRAMES)


def overlay_frames(items: Sequence[tuple[Replay, QColor, str]], *, size: tuple[int, int], fps: int, title: str,
                   speed: float = 1.0, view: ViewBox | None = None, follow: str = FOLLOW_OVERLAY,
                   last_only: bool = False) -> Iterator[tuple[int, int, QImage]]:
    """겹쳐 보기 프레임 (번호, 전체 수, 그림). view = 고정 카메라 상자(None이면 v3.0 GIF와 같은 맞춤 카메라)."""
    from evolab.app.lab.canvas import SpectatorCanvas

    if not items:
        return
    canvas = SpectatorCanvas()
    try:
        canvas.setMinimumSize(1, 1)  # 작은 MP4(640×360)·끝 장면 그림도 화면용 최소 크기와 상관없이 그 크기로
        canvas.resize(*size)
        first = items[0][0]
        canvas.set_scene(first.geometry, None, channels=first.channels, view=view)
        canvas.set_overlay([(r.trajectory, r.agent, color, label) for r, color, label in items])
        canvas.set_follow(follow)
        duration = canvas.duration() + OVERLAY_TAIL
        count = max(1, math.ceil(duration * fps / max(speed, 1e-6)) + 1)
        for k in range(count - 1 if last_only else 0, count):
            t = min(duration, k * speed / fps)
            canvas.set_time(t)
            if last_only:
                canvas.set_follow(follow)  # 처음부터 따라온 카메라 자리로 바로(한 번만 그리므로 따라오는 중간이 없다)
            canvas.set_hud(S.VIDEO_HUD.format(title=title, t=t) if title else S.VIDEO_HUD_TIME.format(t=t))
            img = QImage(size[0], size[1], QImage.Format.Format_RGB32)
            canvas.render(img)
            yield k, count, img
    finally:
        canvas.deleteLater()


def lane_frames(lanes: Sequence[tuple[Replay, QColor, str]], cols: int, *, pane: tuple[int, int], fps: int,
                title: str, rows: int | None = None, size: tuple[int, int] | None = None,
                views: Sequence[ViewBox | None] | None = None, follow: str = FOLLOW_LANES,
                last_only: bool = False) -> Iterator[tuple[int, int, QImage]]:
    """경주 칸 프레임 (번호, 전체 수, 그림). 칸 k는 격자 (k // cols, k % cols) 자리에 pane 크기로. size를 주면 그림이 그
    크기(격자보다 크면 남는 곳은 배경색), 아니면 열 × 줄 칸. views[k] = 칸 k의 고정 카메라 상자(None이면 v3.0처럼 따라감)."""
    from evolab.app.lab.canvas import COLOR_BG, SpectatorCanvas

    if not lanes:
        return
    cols = max(1, min(int(cols), len(lanes)))
    rows = rows if rows is not None else math.ceil(len(lanes) / cols)
    pw, ph = int(pane[0]), int(pane[1])
    width, height = size if size is not None else (cols * pw, rows * ph)
    views = list(views) if views is not None else [None] * len(lanes)
    canvases: list[SpectatorCanvas] = []
    try:
        for (rep, color, _label), view in zip(lanes, views, strict=True):
            c = SpectatorCanvas()
            c.setMinimumSize(1, 1)  # 관전 캔버스의 화면용 최소 크기보다 작은 칸(320×200)도 그린다
            c.resize(pw, ph)
            c.set_scene(rep.geometry, None, channels=rep.channels, view=view)
            c.set_overlay([(rep.trajectory, rep.agent, QColor(color), "")])
            c.set_follow(follow)
            canvases.append(c)
        longest = max(rep.trajectory.duration for rep, _c, _l in lanes)
        end = longest + RACE_TAIL
        count = race_count(longest, fps)
        for k in range(count - 1 if last_only else 0, count):
            t = min(end, k / fps)
            img = QImage(width, height, QImage.Format.Format_RGB32)
            img.fill(COLOR_BG)
            for i, c in enumerate(canvases):
                label = lanes[i][2]
                hud = S.VIDEO_HUD.format(title=label, t=t) if label else S.VIDEO_HUD_TIME.format(t=t)
                c.set_time(t)
                if last_only:
                    c.set_follow(follow)
                c.set_hud(f"{title}\n{hud}" if i == 0 and title else hud)
                c.render(img, QPoint((i % cols) * pw, (i // cols) * ph))
            p = QPainter(img)
            for i, (_rep, color, _label) in enumerate(lanes):
                p.fillRect(QRect((i % cols) * pw, (i // cols) * ph, pw, RACE_BAR), QColor(color))
            p.setPen(RACE_SEPARATOR)
            for col in range(1, cols):
                p.drawLine(col * pw, 0, col * pw, rows * ph)
            for row in range(1, rows):
                p.drawLine(0, row * ph, cols * pw, row * ph)
            p.end()
            yield k, count, img
    finally:
        for c in canvases:
            c.deleteLater()


__all__ = [
    "FIXED_MARGIN", "FIXED_MIN_SIZE", "OVERLAY_TAIL", "RACE_BAR", "RACE_SEPARATOR", "RACE_TAIL", "fixed_view",
    "frame_count", "iter_frames", "lane_frames", "last_frame", "numbered_frames", "overlay_count", "overlay_frames",
    "race_count",
]

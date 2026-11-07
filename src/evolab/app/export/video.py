"""챔피언 영상 내보내기 (명세: 내보내기 > 챔피언 영상). 다시 시뮬레이션한 재생을 관전 캔버스로 그려 GIF로 묶는다.

"1세대 vs 지금" 겹친 영상도 같은 함수로 만든다(재생 여러 개를 겹쳐 그린다). 그리기는 화면 스레드에서 하고,
프레임 사이마다 progress 콜백을 불러 창이 멈추지 않게 한다(취소하려면 콜백이 False를 돌려준다).

나란히 경주 영상(v1.2, 설계 D31): render_race_frames가 칸마다 관전 캔버스를 하나씩 두고 프레임마다 한 그림의 격자
자리(칸 320×200)에 그린다. 끝 장면을 1초 더 보여 주고, export_race_gif가 같은 save_gif로 GIF 하나에 담는다.
MP4는 챔피언 영상과 같은 이유(A19)로 만들지 않는다.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QColor, QImage, QPainter

from evolab.app import strings as S
from evolab.engine.replay import Replay

DEFAULT_SIZE = (640, 400)
DEFAULT_FPS = 20
RACE_PANE = (320, 200)
"""경주 영상의 칸 하나 크기."""
RACE_TAIL = 1.0
"""경주 영상에서 가장 긴 칸이 끝난 뒤 더 보여 주는 시간(초)."""
RACE_BAR = 3
"""경주 영상 칸 위쪽의 칸 색 띠 두께(화소): 칸 색으로 어느 칸인지 알아보게."""
RACE_SEPARATOR = QColor(255, 255, 255, 40)


def render_frames(items: Sequence[tuple[Replay, QColor, str]], *, size: tuple[int, int] = DEFAULT_SIZE,
                  fps: int = DEFAULT_FPS, speed: float = 1.0, title: str = "",
                  progress: Callable[[int, int], bool] | None = None) -> list[QImage]:
    """재생들을 겹쳐 그린 프레임 목록. 시간은 0부터 가장 긴 재생이 끝날 때까지(끝난 뒤 0.5초 더)."""
    from evolab.app.lab.canvas import SpectatorCanvas

    if not items:
        return []
    canvas = SpectatorCanvas()
    canvas.resize(*size)
    first = items[0][0]
    canvas.set_scene(first.geometry, None, channels=first.channels)
    canvas.set_overlay([(r.trajectory, r.agent, color, label) for r, color, label in items])
    canvas.set_follow("fit")
    duration = canvas.duration() + 0.5
    count = max(1, math.ceil(duration * fps / max(speed, 1e-6)) + 1)
    frames: list[QImage] = []
    for k in range(count):
        t = min(duration, k * speed / fps)
        canvas.set_time(t)
        canvas.set_hud(S.VIDEO_HUD.format(title=title, t=t) if title else S.VIDEO_HUD_TIME.format(t=t))
        img = QImage(size[0], size[1], QImage.Format.Format_RGB32)
        canvas.render(img)
        frames.append(img)
        if progress is not None and progress(k + 1, count) is False:
            break
    canvas.deleteLater()
    return frames


def save_gif(frames: Sequence[QImage], path: Path, fps: int = DEFAULT_FPS) -> Path:
    """프레임을 GIF로. 첫 프레임에서 팔레트를 만들고 모든 프레임에 같은 팔레트를 써서 빠르고 깜빡이지 않게 한다."""
    from PIL import Image

    if not frames:
        raise ValueError(S.VIDEO_NO_FRAMES)
    pil = []
    palette_src = None
    for img in frames:
        rgb = img.convertToFormat(QImage.Format.Format_RGB888)
        w, h = rgb.width(), rgb.height()
        stride = rgb.bytesPerLine()
        buf = bytes(rgb.constBits())[: stride * h]
        im = Image.frombuffer("RGB", (w, h), buf, "raw", "RGB", stride, 1)
        if palette_src is None:
            palette_src = im.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
            pil.append(palette_src)
        else:
            pil.append(im.quantize(palette=palette_src, dither=Image.Dither.NONE))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pil[0].save(tmp, format="GIF", save_all=True, append_images=pil[1:], duration=max(10, round(1000 / fps)),
                loop=0, optimize=False, disposal=1)
    tmp.replace(path)
    return path


def export_gif(items: Sequence[tuple[Replay, QColor, str]], path: Path, *, size: tuple[int, int] = DEFAULT_SIZE,
               fps: int = DEFAULT_FPS, speed: float = 1.0, title: str = "",
               progress: Callable[[int, int], bool] | None = None) -> Path | None:
    """재생들을 GIF 파일로. 취소하면 None(파일을 쓰지 않는다: 같은 이름의 파일이 있으면 그대로 둔다)."""
    cancelled = False

    def on_progress(done: int, total: int) -> bool:
        nonlocal cancelled
        if progress is not None and progress(done, total) is False:
            cancelled = True
            return False
        return True

    frames = render_frames(items, size=size, fps=fps, speed=speed, title=title,
                           progress=on_progress if progress is not None else None)
    if not frames or cancelled:
        return None
    return save_gif(frames, path, fps)


def race_frame_count(duration: float, fps: int = DEFAULT_FPS) -> int:
    """경주 영상 프레임 수: ⌈(가장 긴 재생 + 1초) × fps⌉ + 1(0초 프레임 포함)."""
    return max(1, math.ceil((float(duration) + RACE_TAIL) * fps) + 1)


def render_race_frames(lanes: Sequence[tuple[Replay, QColor, str]], cols: int, *,
                       pane: tuple[int, int] = RACE_PANE, fps: int = DEFAULT_FPS, title: str = "",
                       progress: Callable[[int, int], bool] | None = None) -> list[QImage]:
    """나란히 경주의 프레임 목록. 칸 k는 격자 (k // cols, k % cols) 자리에 pane 크기로 그린다(그림 = 열 × 줄 칸).

    칸마다 관전 캔버스가 그 칸 개체를 따라간다(경주 창과 같은 그림). 칸 위쪽에 칸 색 띠, 글상자에 칸 이름과 시각(첫
    칸은 첫 줄에 title도). 작은 칸이 붐비지 않게 개체 옆 이름표와 범례는 그리지 않는다. 시간은 0부터 가장 긴 재생이
    끝난 뒤 1초까지(1/fps 간격, 끝 시각에서 멈춤).
    """
    from evolab.app.lab.canvas import COLOR_BG, SpectatorCanvas

    if not lanes:
        return []
    cols = max(1, min(int(cols), len(lanes)))
    rows = math.ceil(len(lanes) / cols)
    pw, ph = int(pane[0]), int(pane[1])
    canvases: list[SpectatorCanvas] = []
    try:
        for rep, color, _label in lanes:
            c = SpectatorCanvas()
            c.setMinimumSize(1, 1)  # 관전 캔버스의 화면용 최소 크기보다 작은 칸(320×200)도 그린다
            c.resize(pw, ph)
            c.set_scene(rep.geometry, None, channels=rep.channels)
            c.set_overlay([(rep.trajectory, rep.agent, QColor(color), "")])
            c.set_follow("champion")
            canvases.append(c)
        longest = max(rep.trajectory.duration for rep, _c, _l in lanes)
        end = longest + RACE_TAIL
        count = race_frame_count(longest, fps)
        frames: list[QImage] = []
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
                p.fillRect(QRect((i % cols) * pw, (i // cols) * ph, pw, RACE_BAR), QColor(color))
            p.setPen(RACE_SEPARATOR)
            for col in range(1, cols):
                p.drawLine(col * pw, 0, col * pw, rows * ph)
            for row in range(1, rows):
                p.drawLine(0, row * ph, cols * pw, row * ph)
            p.end()
            frames.append(img)
            if progress is not None and progress(k + 1, count) is False:
                break
        return frames
    finally:
        for c in canvases:
            c.deleteLater()


def export_race_gif(lanes: Sequence[tuple[Replay, QColor, str]], path: Path, cols: int, *,
                    pane: tuple[int, int] = RACE_PANE, fps: int = DEFAULT_FPS, title: str = "",
                    progress: Callable[[int, int], bool] | None = None) -> Path | None:
    """나란히 경주를 GIF 하나로. 취소하면 None(파일을 쓰지 않는다: 같은 이름의 파일이 있으면 그대로 둔다)."""
    cancelled = False

    def on_progress(done: int, total: int) -> bool:
        nonlocal cancelled
        if progress is not None and progress(done, total) is False:
            cancelled = True
            return False
        return True

    frames = render_race_frames(lanes, cols, pane=pane, fps=fps, title=title,
                                progress=on_progress if progress is not None else None)
    if not frames or cancelled:
        return None
    return save_gif(frames, path, fps)

"""챔피언 영상 내보내기 (명세: 내보내기 > 챔피언 영상). 다시 시뮬레이션한 재생을 관전 캔버스로 그려 GIF로 묶는다.

"1세대 vs 지금" 겹친 영상도 같은 함수로 만든다(재생 여러 개를 겹쳐 그린다). 그리기는 화면 스레드에서 하고,
프레임 사이마다 progress 콜백을 불러 창이 멈추지 않게 한다(취소하려면 콜백이 False를 돌려준다).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

from PySide6.QtGui import QColor, QImage

from evolab.app import strings as S
from evolab.engine.replay import Replay

DEFAULT_SIZE = (640, 400)
DEFAULT_FPS = 20


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

"""챔피언 영상 내보내기 (명세: 내보내기 > 챔피언 영상). 다시 시뮬레이션한 재생을 관전 캔버스로 그려 GIF로 묶는다.

"1세대 vs 지금" 겹친 영상도 같은 함수로 만든다(재생 여러 개를 겹쳐 그린다). 그리기는 화면 스레드에서 하고,
프레임 사이마다 progress 콜백을 불러 창이 멈추지 않게 한다(취소하려면 콜백이 False를 돌려준다). 파일로 쓸 때는 프레임을
쌓아 두지 않는다: 한 프레임을 그리면 바로 팔레트 색으로 바꿔 GIF에 쓴다(_GifWriter). 그래서 긴 영상도 메모리가 프레임
두세 장 몫이고, 그리기와 쓰기 내내 진행 창이 움직이며 취소할 수 있다.

나란히 경주 영상(v1.2, 설계 D31): render_race_frames가 칸마다 관전 캔버스를 하나씩 두고 프레임마다 한 그림의 격자
자리(칸 320×200)에 그린다. 끝 장면을 1초 더 보여 주고, export_race_gif가 같은 방식으로 GIF 하나에 담는다.

v3.1(설계 D48~D50, 7.1): 그리기는 export.frames로 옮겼다. 이 모듈의 _iter_frames·_iter_race_frames는 frames 위의 얇은 껍데기라
export_gif·export_race_gif·render_frames의 서명과 결과(「따라가기」 카메라 그림)는 v3.0 그대로다. 내보내기 대화 상자(MP4·GIF·폰
재생 파일, 고정 카메라)는 내보내기 자식 프로세스(export.worker)가 같은 frames와 이 모듈의 _GifWriter로 만든다. 경로를 주는 옛
호출(lab.save_video(path)·race.save_gif(path))은 지금처럼 이 모듈로 화면 프로세스에서 GIF를 만든다.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any, BinaryIO

from PySide6.QtGui import QColor, QImage

from evolab.app import strings as S
from evolab.engine.replay import Replay

DEFAULT_SIZE = (640, 400)
DEFAULT_FPS = 20
RACE_PANE = (320, 200)
"""경주 영상의 칸 하나 크기."""
RACE_TAIL = 1.0
"""경주 영상에서 가장 긴 칸이 끝난 뒤 더 보여 주는 시간(초, = frames.RACE_TAIL)."""


def _iter_frames(items: Sequence[tuple[Replay, QColor, str]], *, size: tuple[int, int], fps: int, speed: float,
                 title: str) -> Iterator[tuple[int, int, QImage]]:
    """render_frames의 몸통: (번호, 전체 수, 프레임)을 하나씩 만든다(프레임을 쌓아 두지 않는다). frames.overlay_frames의
    「따라가기」(맞춤) 카메라 그대로."""
    from evolab.app.export import frames

    return frames.overlay_frames(items, size=size, fps=fps, speed=speed, title=title)


def render_frames(items: Sequence[tuple[Replay, QColor, str]], *, size: tuple[int, int] = DEFAULT_SIZE,
                  fps: int = DEFAULT_FPS, speed: float = 1.0, title: str = "",
                  progress: Callable[[int, int], bool] | None = None) -> list[QImage]:
    """재생들을 겹쳐 그린 프레임 목록. 시간은 0부터 가장 긴 재생이 끝날 때까지(끝난 뒤 0.5초 더). 파일로 쓸 때는
    export_gif가 프레임을 쌓지 않고 하나씩 GIF에 쓴다."""
    return [img for img in _with_progress(_iter_frames(items, size=size, fps=fps, speed=speed, title=title),
                                          progress, None)]


def _with_progress(frames: Iterator[tuple[int, int, QImage]], progress: Callable[[int, int], bool] | None,
                   cancelled: list[bool] | None) -> Iterator[QImage]:
    """프레임을 하나씩 넘기고, 받는 쪽이 그 프레임을 다 쓴 뒤(GIF에 쓰기까지) progress(끝낸 수, 전체 수)를 부른다.
    progress가 False를 돌려주면 멈추고 cancelled[0]을 True로."""
    with closing(frames):
        for k, count, img in frames:
            yield img
            if progress is not None and progress(k + 1, count) is False:
                if cancelled is not None:
                    cancelled[0] = True
                return


class _GifWriter:
    """GIF를 한 프레임씩 파일에 쓴다(프레임을 메모리에 쌓지 않는다: 긴 경주도 메모리가 프레임 두세 장 몫).

    첫 프레임에서 팔레트(255색)를 만들어 모든 프레임에 같은 팔레트를 쓴다(빠르고 깜빡이지 않게, 전역 색표 하나).
    앞 프레임과 달라진 사각형만 쓰고(disposal 1), 똑같은 프레임은 앞 프레임의 표시 시간에 더한다(Pillow의 save_all과
    같은 결과). 그래서 한 프레임을 늦게 쓴다(close에서 마지막 프레임).
    """

    def __init__(self, fp: BinaryIO, fps: int) -> None:
        self.fp = fp
        self.duration = max(10, round(1000 / fps))
        self.count = 0
        self._palette: Any = None
        self._prev: Any = None
        self._pending: list[Any] | None = None  # [P 그림(잘라 낸), (x, y), 표시 시간 ms]

    def add(self, img: QImage) -> None:
        from PIL import GifImagePlugin, Image, ImageChops

        rgb = img.convertToFormat(QImage.Format.Format_RGB888)
        w, h = rgb.width(), rgb.height()
        stride = rgb.bytesPerLine()
        im = Image.frombuffer("RGB", (w, h), bytes(rgb.constBits())[: stride * h], "raw", "RGB", stride, 1)
        if self._palette is None:
            frame = im.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
            self._palette = frame
            header, _used = GifImagePlugin.getheader(frame, None, {"loop": 0, "duration": self.duration,
                                                                   "optimize": False})
            for chunk in header:
                self.fp.write(chunk)
            self._pending = [frame, (0, 0), self.duration]
        else:
            frame = im.quantize(palette=self._palette, dither=Image.Dither.NONE)
            bbox = ImageChops.subtract_modulo(frame, self._prev).getbbox()
            if bbox is None:  # 앞 프레임과 같다: 앞 프레임을 더 오래 보인다
                assert self._pending is not None
                self._pending[2] += self.duration
            else:
                self._flush()
                self._pending = [frame.crop(bbox), bbox[:2], self.duration]
        self._prev = frame
        self.count += 1

    def _flush(self) -> None:
        from PIL import GifImagePlugin

        if self._pending is None:
            return
        frame, offset, duration = self._pending
        self._pending = None
        for chunk in GifImagePlugin.getdata(frame, offset, duration=duration, disposal=1):
            self.fp.write(chunk)

    def close(self) -> int:
        """마지막 프레임과 파일 끝을 쓴다. 쓴 프레임 수(같은 프레임을 합치기 전)를 돌려준다."""
        self._flush()
        if self.count:
            self.fp.write(b";")
        return self.count


def _write_gif(frames: Iterable[QImage], path: Path, fps: int, cancelled: list[bool] | None = None) -> Path | None:
    """프레임을 하나씩 임시 파일에 쓰고 끝나면 path로 바꾼다. 프레임이 없거나 취소했으면(cancelled[0]) 임시 파일을
    지우고 None(같은 이름의 파일이 있으면 그대로 둔다). 쓰다가 오류(디스크·메모리)가 나도 임시 파일을 남기지 않는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fp:
            count = _stream(frames, fp, fps)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if not count or (cancelled is not None and cancelled[0]):
        tmp.unlink(missing_ok=True)
        return None
    tmp.replace(path)
    return path


def _stream(frames: Iterable[QImage], fp: BinaryIO, fps: int) -> int:
    writer = _GifWriter(fp, fps)
    for img in frames:
        writer.add(img)
    return writer.close()


def save_gif(frames: Iterable[QImage], path: Path, fps: int = DEFAULT_FPS) -> Path:
    """프레임을 GIF로(목록이든 하나씩 만드는 반복자든). 첫 프레임에서 팔레트를 만들고 모든 프레임에 같은 팔레트를 써서
    빠르고 깜빡이지 않게 한다. 프레임을 쌓아 두지 않고 하나씩 쓴다."""
    out = _write_gif(frames, path, fps)
    if out is None:
        raise ValueError(S.VIDEO_NO_FRAMES)
    return out


def export_gif(items: Sequence[tuple[Replay, QColor, str]], path: Path, *, size: tuple[int, int] = DEFAULT_SIZE,
               fps: int = DEFAULT_FPS, speed: float = 1.0, title: str = "",
               progress: Callable[[int, int], bool] | None = None) -> Path | None:
    """재생들을 GIF 파일로. 프레임마다 그리고 바로 GIF에 쓴 뒤 progress를 부른다(그리기와 쓰기 모두 창이 멈추지 않고
    취소할 수 있다). 취소하면 None(파일을 쓰지 않는다: 같은 이름의 파일이 있으면 그대로 둔다)."""
    cancelled = [False]
    frames = _with_progress(_iter_frames(items, size=size, fps=fps, speed=speed, title=title), progress, cancelled)
    return _write_gif(frames, path, fps, cancelled)


def race_frame_count(duration: float, fps: int = DEFAULT_FPS) -> int:
    """경주 영상 프레임 수: ⌈(가장 긴 재생 + 1초) × fps⌉ + 1(0초 프레임 포함, = frames.race_count)."""
    return max(1, math.ceil((float(duration) + RACE_TAIL) * fps) + 1)


def _iter_race_frames(lanes: Sequence[tuple[Replay, QColor, str]], cols: int, *, pane: tuple[int, int], fps: int,
                      title: str) -> Iterator[tuple[int, int, QImage]]:
    """render_race_frames의 몸통: (번호, 전체 수, 프레임)을 하나씩 만든다(프레임을 쌓아 두지 않는다). frames.lane_frames의
    「따라가기」(칸 개체) 카메라 그대로."""
    from evolab.app.export import frames

    return frames.lane_frames(lanes, cols, pane=pane, fps=fps, title=title)


def render_race_frames(lanes: Sequence[tuple[Replay, QColor, str]], cols: int, *,
                       pane: tuple[int, int] = RACE_PANE, fps: int = DEFAULT_FPS, title: str = "",
                       progress: Callable[[int, int], bool] | None = None) -> list[QImage]:
    """나란히 경주의 프레임 목록. 칸 k는 격자 (k // cols, k % cols) 자리에 pane 크기로 그린다(그림 = 열 × 줄 칸).

    칸마다 관전 캔버스가 그 칸 개체를 따라간다(경주 창과 같은 그림). 칸 위쪽에 칸 색 띠, 글상자에 칸 이름과 시각(첫
    칸은 첫 줄에 title도). 작은 칸이 붐비지 않게 개체 옆 이름표와 범례는 그리지 않는다. 시간은 0부터 가장 긴 재생이
    끝난 뒤 1초까지(1/fps 간격, 끝 시각에서 멈춤). 파일로 쓸 때는 export_race_gif가 프레임을 쌓지 않고 하나씩 쓴다
    (6칸 60초 경주의 RGB32 프레임을 다 쌓으면 2GB가 넘는다).
    """
    return list(_with_progress(_iter_race_frames(lanes, cols, pane=pane, fps=fps, title=title), progress, None))


def export_race_gif(lanes: Sequence[tuple[Replay, QColor, str]], path: Path, cols: int, *,
                    pane: tuple[int, int] = RACE_PANE, fps: int = DEFAULT_FPS, title: str = "",
                    progress: Callable[[int, int], bool] | None = None) -> Path | None:
    """나란히 경주를 GIF 하나로. 프레임마다 그리고 바로 GIF에 쓴 뒤 progress를 부른다. 취소하면 None(파일을 쓰지
    않는다: 같은 이름의 파일이 있으면 그대로 둔다)."""
    cancelled = [False]
    frames = _with_progress(_iter_race_frames(lanes, cols, pane=pane, fps=fps, title=title), progress, cancelled)
    return _write_gif(frames, path, fps, cancelled)

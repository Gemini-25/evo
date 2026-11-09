"""MP4 쓰기 (v3 설계 D51~D54, 2.7, 4.16·4.17, 7.3 「app/export/mp4.py (M)」).

두 인코더 가운데 하나로 MP4 파일을 만든다(새 의존성 없음).

- **Media Foundation**(mf_win.py, 윈도우만): 윈도우에 들어 있는 H.264 인코더. 작은 표준 파일(평균 비트레이트 640×360
  1Mbps, 1280×720 2Mbps). MF_ENABLED이고 윈도우이고 시험(64×64 두 장)이 통과할 때만 쓴다.
- **내장 쓰기**(h264_pcm.py + 이 모듈의 Mp4Muxer): 손실 없는 I_PCM + P_Skip. 늘 된다(리눅스·N판 윈도우·CI). 바뀐 블록만
  쓰므로 크기는 카메라가 정한다(고정 카메라면 작다).

open_writer(prefer="auto")는 encoder_status()가 "mf"면 MF, 아니면 내장 쓰기를 연다. 시험 결과는 프로세스마다 한 번
기억한다(encoder_status(refresh=True)로 다시 시험). write_mp4는 내보내기 자식(묶음 F의 run_job)이 쓰는 한 번에 만들기:
MF가 쓰는 중에 실패하면 같은 프레임을 내장 쓰기로 처음부터 다시 만들고 EXP_MF_FALLBACK 한 줄을 note로 알린다(D51).

파일: 쓰는 동안은 임시 파일(temp_path: `<이름>.tmp.mp4`), close가 끝나면 최종 이름으로 바꾼다. abort·취소·오류면 임시
파일을 지우고 같은 이름의 원래 파일은 그대로 둔다(지금 GIF 규칙, 설계 2.6). 임시 이름이 `.mp4`로 끝나는 것은 Media
Foundation 싱크 라이터가 확장자로 컨테이너를 고르기 때문이다(설계 2.6의 `<이름>.tmp`에서 바꿈).

내장 쓰기의 MP4 상자: ftyp + mdat(64비트 크기, 흘려 씀) + moov(끝). 비디오 트랙 하나, 1초(fps 프레임)마다 청크 하나,
`stco`(청크 위치가 4GB를 넘으면 `co64`), 동기 표본은 첫 프레임(IDR) 하나. avc1 안에 avcC(SPS·PPS, 닫을 때 고른 레벨),
colr(nclx BT.709 제한 범위), pasp(1:1). 소리 없음.
"""

from __future__ import annotations

import os
import struct
import sys
from collections.abc import Callable, Iterable
from contextlib import closing, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final, Literal, Protocol

import numpy as np

from evolab.app import strings as S
from evolab.app.export import mf_win
from evolab.app.export.h264_pcm import (
    CONSTRAINT_FLAGS,
    PROFILE_IDC,
    PcmEncoder,
    frame_pixels,
)

MF_ENABLED: Final = True
"""False면 Media Foundation을 시험하지도 쓰지도 않는다(설계 7.4 되돌릴 지점 8)."""
BUILTIN_ENABLED: Final = True
"""Q4(특허) 결정이 「내장 H.264 쓰기를 내보내지 않음」이면 False: MF가 없는 PC는 MP4를 쓸 수 없다(encoder_status "none")."""

MF_BITRATE_SMALL: Final = 1_000_000
MF_BITRATE_MEDIUM: Final = 2_000_000
"""MF 평균 비트레이트(설계 D53): 640×400 이하(작게) 1Mbps, 그보다 크면(보통 1280×720) 2Mbps."""

ENCODER_KINDS: Final = ("mf", "builtin", "none")
PREFERS: Final = ("auto", "mf", "builtin")


@dataclass(frozen=True)
class EncoderStatus:
    """이 프로세스가 쓸 MP4 인코더. kind="mf"면 detail = 인코더 이름(MFT_FRIENDLY_NAME), 아니면 MF를 못 쓰는 까닭
    (EXP_MF_OFF, EXP_MF_NOT_WINDOWS, 실패한 호출과 HRESULT 16진수)."""

    kind: Literal["mf", "builtin", "none"]
    detail: str


class Mp4Writer(Protocol):
    """MP4 작성기(내장 쓰기·MF 공통)."""

    kind: str
    """"builtin" 또는 "mf"."""

    def add(self, frame: Any) -> None:
        """프레임 하나: QImage(크기 = 너비×높이) 또는 (H, W) uint32·(H, W, 4) uint8 BGRA 배열."""

    def bytes_written(self) -> int:
        """지금까지 쓴 바이트(진행 창의 「지금까지 nMB」)."""

    def close(self) -> int:
        """마무리하고 임시 파일을 최종 이름으로 바꾼다. 쓴 프레임 수."""

    def abort(self) -> None:
        """그만둔다: 임시 파일을 지운다(최종 이름의 파일은 그대로). close 뒤에 부르면 아무것도 하지 않는다."""


class EncoderUnavailable(RuntimeError):
    """요청한 인코더를 이 PC에서 쓸 수 없다(open_writer(prefer="mf")인데 MF 시험 실패, 또는 BUILTIN_ENABLED가 False).
    메시지 = 까닭(EncoderStatus.detail과 같은 글: EXP_INFO_BUILTIN_TIP의 {reason})."""


def temp_path(path: Path | str) -> Path:
    """쓰는 동안의 임시 파일 `<이름 줄기>.tmp<확장자>`(챔피언.mp4 → 챔피언.tmp.mp4, 두 작성기 공통). 확장자를 남기는 것은
    Media Foundation 싱크 라이터가 경로의 확장자로 컨테이너를 고르기 때문이다(MF_TRANSCODE_CONTAINERTYPE도 함께 준다).
    내보내기 자식을 강제로 끝냈으면 부르는 쪽이 이 파일을 지운다."""
    p = Path(path)
    return p.with_name(f"{p.stem}.tmp{p.suffix or '.mp4'}")


def mf_bitrate(width: int, height: int) -> int:
    """MF 평균 비트레이트(비트/초). 대화 상자의 예상 크기 = mf_bitrate × 길이 / 8."""
    return MF_BITRATE_SMALL if int(width) * int(height) <= 640 * 400 else MF_BITRATE_MEDIUM


# ---------------------------------------------------------------------------
# MP4 상자
# ---------------------------------------------------------------------------
def _box(kind: bytes, *payload: bytes) -> bytes:
    body = b"".join(payload)
    return struct.pack(">I", 8 + len(body)) + kind + body


def _full(kind: bytes, version: int, flags: int, *payload: bytes) -> bytes:
    return _box(kind, struct.pack(">I", (version << 24) | (flags & 0xFFFFFF)), *payload)


_MATRIX: Final = struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)


class Mp4Muxer:
    """H.264 표본(NAL 단위 하나씩)을 MP4 파일에 흘려 쓴다. finish가 mdat 크기를 채우고 moov를 끝에 쓴다."""

    TICKS_PER_FRAME: Final = 512
    """미디어 시간 단위: 프레임 하나 = 512틱(timescale = 512 × fps, 어떤 정수 fps에서도 나누어 떨어짐)."""

    def __init__(self, fp: BinaryIO, width: int, height: int, fps: int) -> None:
        self.fp = fp
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.timescale = self.TICKS_PER_FRAME * self.fps
        self.sizes: list[int] = []
        self.sync: list[int] = []
        self._base = fp.tell()
        fp.write(_box(b"ftyp", b"isom", struct.pack(">I", 0x200), b"isomiso2avc1mp41"))
        self.mdat_pos = fp.tell()
        fp.write(struct.pack(">I4sQ", 1, b"mdat", 0))  # 64비트 크기(largesize), finish에서 채움
        self.data_start = fp.tell()
        self.data_bytes = 0

    @property
    def bytes_written(self) -> int:
        return self.data_start - self._base + self.data_bytes

    def add_sample(self, nal: bytes, *, sync: bool) -> None:
        self.fp.write(struct.pack(">I", len(nal)))
        self.fp.write(nal)
        self.sizes.append(len(nal) + 4)
        self.data_bytes += len(nal) + 4
        if sync:
            self.sync.append(len(self.sizes))

    def finish(self, sps: bytes, pps: bytes, *, profile: int, compat: int, level: int) -> int:
        """mdat 크기를 채우고 moov를 쓴다. 파일 전체 바이트를 돌려준다."""
        fp = self.fp
        end = self.data_start + self.data_bytes
        fp.seek(self.mdat_pos)
        fp.write(struct.pack(">I4sQ", 1, b"mdat", end - self.mdat_pos))
        fp.seek(end)
        fp.write(self._moov(sps, pps, profile, compat, level))
        fp.flush()
        return fp.tell() - self._base

    def _moov(self, sps: bytes, pps: bytes, profile: int, compat: int, level: int) -> bytes:
        n = len(self.sizes)
        w, h = self.width, self.height
        media_dur = n * self.TICKS_PER_FRAME
        movie_dur = round(n * 1000 / self.fps)
        mvhd = _full(b"mvhd", 0, 0, struct.pack(">IIII", 0, 0, 1000, movie_dur), struct.pack(">IH", 0x10000, 0x100),
                     b"\0" * 10, _MATRIX, b"\0" * 24, struct.pack(">I", 2))
        tkhd = _full(b"tkhd", 0, 3, struct.pack(">IIIII", 0, 0, 1, 0, movie_dur), b"\0" * 8,
                     struct.pack(">hhhH", 0, 0, 0, 0), _MATRIX, struct.pack(">II", w << 16, h << 16))
        mdhd = _full(b"mdhd", 0, 0, struct.pack(">IIIIHH", 0, 0, self.timescale, media_dur, 0x55C4, 0))  # und
        hdlr = _full(b"hdlr", 0, 0, struct.pack(">I", 0), b"vide", b"\0" * 12, b"VideoHandler\0")
        vmhd = _full(b"vmhd", 0, 1, b"\0" * 8)
        dinf = _box(b"dinf", _full(b"dref", 0, 0, struct.pack(">I", 1), _full(b"url ", 0, 1)))
        avcc = _box(b"avcC", bytes([1, profile & 0xFF, compat & 0xFF, level & 0xFF, 0xFF, 0xE1]),
                    struct.pack(">H", len(sps)), sps, bytes([1]), struct.pack(">H", len(pps)), pps)
        colr = _box(b"colr", b"nclx", struct.pack(">HHHB", 1, 1, 1, 0))  # BT.709, 제한 범위
        pasp = _box(b"pasp", struct.pack(">II", 1, 1))
        avc1 = _box(b"avc1", b"\0" * 6, struct.pack(">H", 1), b"\0" * 16, struct.pack(">HH", w, h),
                    struct.pack(">II", 0x480000, 0x480000), struct.pack(">IH", 0, 1), b"\0" * 32,
                    struct.pack(">Hh", 0x18, -1), avcc, colr, pasp)
        stsd = _full(b"stsd", 0, 0, struct.pack(">I", 1), avc1)
        stts = _full(b"stts", 0, 0, struct.pack(">III", 1, n, self.TICKS_PER_FRAME))
        stss = _full(b"stss", 0, 0, struct.pack(">I", len(self.sync)), struct.pack(f">{len(self.sync)}I", *self.sync))
        per = self.fps
        n_chunks = -(-n // per)
        rem = n - (n_chunks - 1) * per
        stsc_rows = [(1, min(per, n), 1)]
        if n_chunks > 1 and rem != per:
            stsc_rows.append((n_chunks, rem, 1))
        stsc = _full(b"stsc", 0, 0, struct.pack(">I", len(stsc_rows)),
                     b"".join(struct.pack(">III", *row) for row in stsc_rows))
        sizes = np.asarray(self.sizes, dtype=np.int64)
        stsz = _full(b"stsz", 0, 0, struct.pack(">II", 0, n), sizes.astype(">u4").tobytes())
        starts = self.data_start + np.concatenate([[0], np.cumsum(sizes)])[0:n:per]
        if starts.size and int(starts[-1]) > 0xFFFFFFFF:
            stco = _full(b"co64", 0, 0, struct.pack(">I", starts.size), starts.astype(">u8").tobytes())
        else:
            stco = _full(b"stco", 0, 0, struct.pack(">I", starts.size), starts.astype(">u4").tobytes())
        stbl = _box(b"stbl", stsd, stts, stss, stsc, stsz, stco)
        minf = _box(b"minf", vmhd, dinf, stbl)
        mdia = _box(b"mdia", mdhd, hdlr, minf)
        trak = _box(b"trak", tkhd, mdia)
        return _box(b"moov", mvhd, trak)


# ---------------------------------------------------------------------------
# 내장 쓰기
# ---------------------------------------------------------------------------
class BuiltinWriter:
    """내장 H.264 쓰기 → MP4(Mp4Writer). 프레임을 쌓지 않는다: 프레임마다 바로 파일에 쓴다."""

    kind: str = "builtin"

    def __init__(self, path: Path | str, width: int, height: int, fps: int) -> None:
        self.path = Path(path)
        self.tmp = temp_path(self.path)
        self.encoder = PcmEncoder(width, height, fps)
        self.level = 0
        self._state = "open"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp: BinaryIO = open(self.tmp, "wb")  # noqa: SIM115 - close·abort가 닫는다
        try:
            self._mux = Mp4Muxer(self._fp, width, height, fps)
        except BaseException:
            self._fp.close()
            self.tmp.unlink(missing_ok=True)
            raise

    @property
    def frames(self) -> int:
        return self.encoder.frames

    def add(self, frame: Any) -> None:
        if self._state != "open":
            raise ValueError("the MP4 writer is closed")
        px, _keep = frame_pixels(frame)
        data = self.encoder.encode(px)
        self._mux.add_sample(data, sync=self.encoder.frames == 1)

    def bytes_written(self) -> int:
        return self._mux.bytes_written

    def close(self) -> int:
        if self._state != "open":
            raise ValueError("the MP4 writer is closed")
        if not self.encoder.frames:
            self.abort()
            raise ValueError(S.VIDEO_NO_FRAMES)
        try:
            self.level = self.encoder.finish(extra_bytes_per_frame=4)
            self._mux.finish(self.encoder.sps_nal(), self.encoder.pps_nal(), profile=PROFILE_IDC,
                             compat=CONSTRAINT_FLAGS, level=self.level)
            self._fp.close()
            os.replace(self.tmp, self.path)
        except BaseException:
            self.abort()
            raise
        self._state = "done"
        return self.encoder.frames

    def abort(self) -> None:
        if self._state != "open":
            return
        self._state = "aborted"
        try:
            self._fp.close()
        except OSError:
            pass
        self.tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 인코더 고르기
# ---------------------------------------------------------------------------
_STATUS: EncoderStatus | None = None


def _is_windows() -> bool:
    return sys.platform == "win32"


def _mf_probe() -> tuple[bool, str]:
    """(MF를 쓸 수 있나, 인코더 이름 또는 까닭)."""
    if not MF_ENABLED:
        return False, S.EXP_MF_OFF
    if not _is_windows():
        return False, S.EXP_MF_NOT_WINDOWS
    try:
        return mf_win.probe()
    except Exception as exc:  # noqa: BLE001 - 시험 실패는 까닭 글로 남기고 내장 쓰기로
        return False, f"{type(exc).__name__}: {exc}"


def encoder_status(refresh: bool = False) -> EncoderStatus:
    """이 프로세스가 쓸 MP4 인코더(시험은 처음 한 번, refresh면 다시). 윈도우에서는 MF 시험이 64×64 두 장을 임시
    폴더에 써 본다(1초 안팎: 대화 상자는 작은 자식 프로세스에서 부른다, 설계 D51)."""
    global _STATUS
    if _STATUS is not None and not refresh:
        return _STATUS
    ok, detail = _mf_probe()
    if ok:
        status = EncoderStatus("mf", detail)
    elif BUILTIN_ENABLED:
        status = EncoderStatus("builtin", detail)
    else:
        status = EncoderStatus("none", detail)
    _STATUS = status
    return status


def _open_mf(path: Path, width: int, height: int, fps: int) -> Mp4Writer:
    return mf_win.MfWriter(path, width, height, fps, bitrate=mf_bitrate(width, height), tmp=temp_path(path))


def open_writer(path: Path | str, width: int, height: int, fps: int, *,
                prefer: Literal["auto", "mf", "builtin"] = "auto") -> Mp4Writer:
    """MP4 작성기를 연다. auto: MF를 쓸 수 있으면 MF, 아니면 내장 쓰기. mf: MF만(못 쓰면 EncoderUnavailable).
    builtin: 내장 쓰기만(BUILTIN_ENABLED가 False면 EncoderUnavailable). MF 작성기를 만들다 실패하면 mf_win.MfError."""
    if prefer not in PREFERS:
        raise ValueError(f"prefer must be one of {PREFERS}, got {prefer!r}")
    path = Path(path)
    if prefer in ("auto", "mf"):
        status = encoder_status()
        if status.kind == "mf":
            return _open_mf(path, width, height, fps)
        if prefer == "mf":
            raise EncoderUnavailable(status.detail)
    if not BUILTIN_ENABLED:
        raise EncoderUnavailable(encoder_status().detail)
    return BuiltinWriter(path, width, height, fps)


def write_mp4(frames: Callable[[], Iterable[Any]], path: Path | str, width: int, height: int, fps: int, *,
              prefer: Literal["auto", "mf", "builtin"] = "auto",
              progress: Callable[[int, int], bool | None] | None = None,
              note: Callable[[str], None] | None = None) -> dict[str, Any] | None:
    """프레임을 MP4 파일 하나로(내보내기 자식의 MP4 길).

    frames() = 프레임 반복자를 새로 만드는 함수(MF가 쓰는 중 실패하면 한 번 더 불러 내장 쓰기로 처음부터).
    progress(쓴 프레임 수, 지금까지 바이트)가 False를 돌려주면 그만두고 None(임시 파일 없음, 원래 파일 그대로).
    note(글): MF가 실패해 내장 쓰기로 다시 만들 때 EXP_MF_FALLBACK 한 줄.
    돌려주는 것: {"path": 최종 경로(str), "frames": 프레임 수, "bytes": 파일 크기, "encoder": "mf"|"builtin"}
    (2.6 「done」 메시지의 칸들). 프레임이 없으면 ValueError(VIDEO_NO_FRAMES).
    """
    path = Path(path)
    use_builtin = prefer == "builtin"
    while True:
        writer: Mp4Writer | None = None
        try:
            if use_builtin:
                if not BUILTIN_ENABLED:
                    raise EncoderUnavailable(encoder_status().detail)
                writer = BuiltinWriter(path, width, height, fps)
            else:
                writer = open_writer(path, width, height, fps, prefer=prefer)
            count = 0
            source = frames()
            with closing(source) if hasattr(source, "close") else nullcontext():
                for frame in source:
                    writer.add(frame)
                    count += 1
                    if progress is not None and progress(count, writer.bytes_written()) is False:
                        writer.abort()
                        return None
            written = writer.close()
            return {"path": str(path), "frames": int(written), "bytes": path.stat().st_size, "encoder": writer.kind}
        except mf_win.MfError as exc:
            if writer is not None:
                writer.abort()
            if use_builtin or not BUILTIN_ENABLED:
                raise
            if note is not None:
                note(S.EXP_MF_FALLBACK.format(reason=str(exc)))
            use_builtin = True
        except BaseException:
            if writer is not None:
                writer.abort()
            raise


def read_info(path: Path | str) -> dict[str, Any]:
    """MP4 파일의 상자 구조 요약(자체 점검·명령줄 확인용, 이 모듈이 쓰는 꼴을 읽는다): {"boxes": 맨 위 상자 이름들,
    "codec": "avc1", "width", "height", "fps", "frames", "sync": 동기 표본 번호들, "profile", "level"}. 상자가 깨졌으면
    ValueError. 화소는 풀지 않는다(복호 확인은 tests/h264_subset_decoder.py)."""
    data = Path(path).read_bytes()

    def boxes(start: int, end: int) -> list[tuple[bytes, int, int]]:
        out, pos = [], start
        while pos < end:
            if end - pos < 8:
                raise ValueError(f"truncated MP4 box at {pos}")
            size, kind = struct.unpack_from(">I4s", data, pos)
            head = 8
            if size == 1:
                size, head = struct.unpack_from(">Q", data, pos + 8)[0], 16
            elif size == 0:
                size = end - pos
            if size < head or pos + size > end:
                raise ValueError(f"bad MP4 box {kind!r} at {pos}")
            out.append((kind, pos + head, pos + size))
            pos += size
        return out

    top = boxes(0, len(data))
    found: dict[bytes, tuple[int, int]] = {}

    def walk(start: int, end: int) -> None:
        for kind, s, e in boxes(start, end):
            if kind in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
                walk(s, e)
            else:
                found.setdefault(kind, (s, e))

    moov = [(s, e) for k, s, e in top if k == b"moov"]
    if not top or top[0][0] != b"ftyp" or len(moov) != 1 or not any(k == b"mdat" for k, _s, _e in top):
        raise ValueError("an MP4 file needs ftyp, mdat and one moov")
    walk(*moov[0])
    try:
        timescale = struct.unpack_from(">I", data, found[b"mdhd"][0] + 12)[0]
        delta = struct.unpack_from(">I", data, found[b"stts"][0] + 12)[0]
        frames = struct.unpack_from(">I", data, found[b"stsz"][0] + 8)[0]
        s, _e = found[b"stss"]
        sync = list(struct.unpack_from(f">{struct.unpack_from('>I', data, s + 4)[0]}I", data, s + 8))
        s, e = found[b"stsd"]
        entry = boxes(s + 8, e)[0]
        width, height = struct.unpack_from(">HH", data, entry[1] + 24)
        avcc = next(bs for k, bs, _be in boxes(entry[1] + 78, entry[2]) if k == b"avcC")
    except (KeyError, IndexError, StopIteration, struct.error) as exc:
        raise ValueError(f"incomplete MP4 sample table: {exc!r}") from exc
    return {"boxes": [k.decode("latin-1") for k, _s, _e in top], "codec": entry[0].decode("latin-1"),
            "width": int(width), "height": int(height), "fps": timescale / delta if delta else 0.0,
            "frames": int(frames), "sync": sync, "profile": data[avcc + 1], "level": data[avcc + 3]}


__all__ = [
    "BUILTIN_ENABLED", "ENCODER_KINDS", "MF_BITRATE_MEDIUM", "MF_BITRATE_SMALL", "MF_ENABLED", "PREFERS",
    "BuiltinWriter", "EncoderStatus", "EncoderUnavailable", "Mp4Muxer", "Mp4Writer", "encoder_status", "frame_pixels",
    "mf_bitrate", "open_writer", "read_info", "temp_path", "write_mp4",
]

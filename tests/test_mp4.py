"""MP4 상자와 인코더 고르기 (v3 설계 D51·D54, 2.7, T37).

- 상자: ftyp + mdat(64비트) + moov(끝), 1초마다 청크, stco(4GB를 넘으면 co64), 동기 표본은 첫 프레임, avcC·colr·pasp.
- T37: 윈도우가 아니면 내장 쓰기, MF_ENABLED = False면 시험 없이 내장 쓰기, MF 시험이 통과하면 MF 작성기, 가짜 MF 작성기가
  중간에 실패하면 write_mp4(내보내기 자식의 MP4 길)가 내장 쓰기로 처음부터 다시 만들고 note로 EXP_MF_FALLBACK을 알린다.
"""

from __future__ import annotations

import io
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from evolab.app import strings as S
from evolab.app.export import mf_win, mp4

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h264_subset_decoder as D  # noqa: E402


def frames_of(n: int, width: int = 64, height: int = 48) -> list[np.ndarray]:
    out = []
    for k in range(n):
        a = np.full((height, width), 0xFF0B0E14, dtype=np.uint32)
        a[10:20, (3 * k) % (width - 10):(3 * k) % (width - 10) + 10] = 0xFFFF7828
        out.append(a)
    return out


# ---------------------------------------------------------------------------
# 상자
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n, fps, rows, chunks", [
    (75, 30, [(1, 30, 1), (3, 15, 1)], 3),
    (60, 30, [(1, 30, 1)], 2),
    (7, 30, [(1, 7, 1)], 1),
    (41, 20, [(1, 20, 1), (3, 1, 1)], 3),
])
def test_container_layout(tmp_path: Path, n: int, fps: int, rows: list[tuple[int, int, int]], chunks: int) -> None:
    path = tmp_path / "v.mp4"
    writer = mp4.BuiltinWriter(path, 64, 48, fps)
    nals = []
    for f in frames_of(n):
        writer.add(f)
    assert writer.close() == n
    data = path.read_bytes()
    video = D.read_mp4(data)
    assert video.top_level == ["ftyp", "mdat", "moov"]
    assert video.mdat_largesize
    assert (video.major_brand, video.brands) == ("isom", ["isom", "iso2", "avc1", "mp41"])
    assert (video.width, video.height) == (64, 48)
    assert video.timescale == 512 * fps and video.sample_deltas == [512] * n
    assert video.media_duration == 512 * n
    assert (video.movie_timescale, video.movie_duration) == (1000, round(n * 1000 / fps))
    assert video.stsc == rows and len(video.chunk_offsets) == chunks and video.chunk_box == "stco"
    assert video.sync == [1]
    assert video.colr == ("nclx", 1, 1, 1, 0)
    assert video.pasp == (1, 1)
    assert video.avcc["length_size"] == 4
    assert (video.avcc["profile"], video.avcc["compat"], video.avcc["level"]) == (66, 0xC0, writer.level)
    # 표본 = 쓴 NAL(길이 4바이트 + NAL 하나), mdat 안에 차례로
    sizes = [len(s) for s in video.samples]
    assert sizes == [s + 4 for s in writer.encoder.frame_sizes]
    for s in video.samples:
        nals.append(D.split_nals(s))
        assert len(nals[-1]) == 1
    assert nals[0][0][0] & 31 == 5 and all(x[0][0] & 31 == 1 for x in nals[1:])
    # mdat 크기 = 머리 16 + 표본 합, moov는 파일 끝까지
    mdat_pos = data.index(b"mdat") - 4
    assert struct.unpack_from(">I4sQ", data, mdat_pos) == (1, b"mdat", 16 + sum(sizes))
    assert len(data) == mdat_pos + 16 + sum(sizes) + struct.unpack_from(">I", data, mdat_pos + 16 + sum(sizes))[0]


def test_read_info_summarises_the_file(tmp_path: Path) -> None:
    path = tmp_path / "v.mp4"
    writer = mp4.BuiltinWriter(path, 640, 360, 30)
    for f in frames_of(31, 640, 360):
        writer.add(f)
    writer.close()
    assert mp4.read_info(path) == {"boxes": ["ftyp", "mdat", "moov"], "codec": "avc1", "width": 640, "height": 360,
                                   "fps": 30.0, "frames": 31, "sync": [1], "profile": 66, "level": writer.level}
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(path.read_bytes()[:-100])
    with pytest.raises(ValueError):
        mp4.read_info(broken)
    gif = tmp_path / "x.mp4"
    gif.write_bytes(b"GIF89a" + b"\0" * 20)
    with pytest.raises(ValueError):
        mp4.read_info(gif)


def test_chunk_offsets_switch_to_co64_past_4gb() -> None:
    mux = mp4.Mp4Muxer(io.BytesIO(), 64, 48, 30)
    for _ in range(40):
        mux.add_sample(b"\x65" + b"\x88" * 10, sync=False)
    mux.data_start = (1 << 32) + 100  # 4GB를 넘은 자리라고 치고 moov만 만든다(실제로 4GB를 쓰지 않음)
    moov = mux._moov(b"\x67\x42", b"\x68\xce", 66, 0xC0, 30)
    found: dict[bytes, bytes] = {}

    def walk(buf: bytes, start: int, end: int) -> None:
        for kind, s, e in D.iter_boxes(buf, start, end):
            if kind in D.CONTAINERS:
                walk(buf, s, e)
            else:
                found[kind] = buf[s:e]

    walk(moov, 0, len(moov))
    assert b"co64" in found and b"stco" not in found
    count = struct.unpack_from(">I", found[b"co64"], 4)[0]
    offsets = struct.unpack_from(f">{count}Q", found[b"co64"], 8)
    assert offsets == ((1 << 32) + 100, (1 << 32) + 100 + 30 * 15)


def test_bytes_written_grows_with_frames(tmp_path: Path) -> None:
    writer = mp4.BuiltinWriter(tmp_path / "g.mp4", 64, 48, 30)
    seen = []
    for f in frames_of(5):
        writer.add(f)
        seen.append(writer.bytes_written())
    assert seen == sorted(seen) and seen[0] > 64 * 48 * 3 // 2
    writer._fp.flush()  # 파이썬 버퍼에 남은 것까지(bytes_written은 버퍼와 무관하게 쓴 양)
    assert mp4.temp_path(tmp_path / "g.mp4").stat().st_size == seen[-1]
    writer.close()
    assert (tmp_path / "g.mp4").stat().st_size > seen[-1]  # + moov


def test_mf_bitrate_and_temp_path() -> None:
    assert mp4.mf_bitrate(640, 360) == mp4.mf_bitrate(640, 400) == 1_000_000
    assert mp4.mf_bitrate(1280, 720) == mp4.mf_bitrate(1278, 720) == 2_000_000
    assert mp4.temp_path(Path("a/챔피언.mp4")) == Path("a/챔피언.tmp.mp4")
    assert mp4.temp_path("b/clip") == Path("b/clip.tmp.mp4")


# ---------------------------------------------------------------------------
# 인코더 고르기 (T37)
# ---------------------------------------------------------------------------
@pytest.fixture
def fresh_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mp4, "_STATUS", None)


class FakeMf:
    """MF 작성기 흉내: 프레임을 세다가 fail_at번째에서 MfError. 파일은 임시 이름에만 쓴다."""

    kind = "mf"
    made: ClassVar[list[FakeMf]] = []

    def __init__(self, path: Path, width: int, height: int, fps: int, *, bitrate: int, tmp: Path,
                 fail_at: int | None = None) -> None:
        self.path, self.tmp, self.size, self.fps, self.bitrate = Path(path), Path(tmp), (width, height), fps, bitrate
        self.fail_at = fail_at
        self.frames = 0
        self.aborted = False
        self.tmp.write_bytes(b"")
        FakeMf.made.append(self)

    def add(self, frame: Any) -> None:
        if self.fail_at is not None and self.frames + 1 == self.fail_at:
            raise mf_win.MfError("IMFSinkWriter::WriteSample", 0xC00D36B4)
        self.frames += 1
        with open(self.tmp, "ab") as fp:
            fp.write(b"x" * 100)

    def bytes_written(self) -> int:
        return self.tmp.stat().st_size

    def close(self) -> int:
        self.tmp.replace(self.path)
        return self.frames

    def abort(self) -> None:
        self.aborted = True
        self.tmp.unlink(missing_ok=True)


def use_fake_mf(monkeypatch: pytest.MonkeyPatch, *, fail_at: int | None = None, name: str = "Fake H264 MFT") -> list:
    calls: list[str] = []
    FakeMf.made = []

    def probe() -> tuple[bool, str]:
        calls.append("probe")
        return True, name

    def make(*args: Any, **kwargs: Any) -> FakeMf:
        return FakeMf(*args, fail_at=fail_at, **kwargs)

    monkeypatch.setattr(mp4, "_is_windows", lambda: True)
    monkeypatch.setattr(mf_win, "probe", probe)
    monkeypatch.setattr(mf_win, "MfWriter", make)
    return calls


@pytest.mark.skipif(sys.platform == "win32", reason="not-Windows branch")
def test_not_windows_means_builtin(tmp_path: Path, fresh_status: None) -> None:
    status = mp4.encoder_status()
    assert status == mp4.EncoderStatus("builtin", S.EXP_MF_NOT_WINDOWS)
    writer = mp4.open_writer(tmp_path / "a.mp4", 64, 48, 30)
    assert isinstance(writer, mp4.BuiltinWriter) and writer.kind == "builtin"
    writer.abort()
    with pytest.raises(mp4.EncoderUnavailable, match=S.EXP_MF_NOT_WINDOWS):
        mp4.open_writer(tmp_path / "b.mp4", 64, 48, 30, prefer="mf")
    assert not list(tmp_path.iterdir())
    assert mf_win.probe() == (False, S.EXP_MF_NOT_WINDOWS)
    with pytest.raises(mf_win.MfError):
        mf_win.load_api()


def test_mf_disabled_means_builtin_without_probing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   fresh_status: None) -> None:
    calls = use_fake_mf(monkeypatch)
    monkeypatch.setattr(mp4, "MF_ENABLED", False)
    assert mp4.encoder_status() == mp4.EncoderStatus("builtin", S.EXP_MF_OFF)
    assert calls == []
    writer = mp4.open_writer(tmp_path / "a.mp4", 64, 48, 30)
    assert writer.kind == "builtin"
    writer.abort()


def test_mf_used_when_the_probe_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_status: None) -> None:
    calls = use_fake_mf(monkeypatch)
    assert mp4.encoder_status() == mp4.EncoderStatus("mf", "Fake H264 MFT")
    assert mp4.encoder_status().kind == "mf" and calls == ["probe"]  # 프로세스마다 한 번
    mp4.encoder_status(refresh=True)
    assert calls == ["probe", "probe"]
    writer = mp4.open_writer(tmp_path / "a.mp4", 1280, 720, 30)
    assert writer.kind == "mf" and writer.bitrate == 2_000_000 and writer.tmp == tmp_path / "a.tmp.mp4"
    writer.abort()
    small = mp4.open_writer(tmp_path / "b.mp4", 640, 360, 30, prefer="mf")
    assert small.bitrate == 1_000_000
    small.abort()
    builtin = mp4.open_writer(tmp_path / "c.mp4", 64, 48, 30, prefer="builtin")
    assert isinstance(builtin, mp4.BuiltinWriter)
    builtin.abort()


def test_probe_failure_reason_is_kept(monkeypatch: pytest.MonkeyPatch, fresh_status: None) -> None:
    monkeypatch.setattr(mp4, "_is_windows", lambda: True)
    monkeypatch.setattr(mf_win, "probe", lambda: (False, "MFStartup 0xC00D36E3"))
    assert mp4.encoder_status() == mp4.EncoderStatus("builtin", "MFStartup 0xC00D36E3")
    with pytest.raises(mp4.EncoderUnavailable, match="0xC00D36E3"):
        mp4.open_writer("x.mp4", 64, 48, 30, prefer="mf")

    def boom() -> tuple[bool, str]:
        raise OSError("access violation reading 0x0")

    monkeypatch.setattr(mf_win, "probe", boom)
    status = mp4.encoder_status(refresh=True)
    assert status.kind == "builtin" and "OSError" in status.detail


def test_builtin_disabled_leaves_no_mp4_without_mf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   fresh_status: None) -> None:
    monkeypatch.setattr(mp4, "BUILTIN_ENABLED", False)
    monkeypatch.setattr(mp4, "_is_windows", lambda: False)
    assert mp4.encoder_status() == mp4.EncoderStatus("none", S.EXP_MF_NOT_WINDOWS)
    for prefer in ("auto", "builtin", "mf"):
        with pytest.raises(mp4.EncoderUnavailable):
            mp4.open_writer(tmp_path / "a.mp4", 64, 48, 30, prefer=prefer)
    with pytest.raises(mp4.EncoderUnavailable):
        mp4.write_mp4(lambda: iter(frames_of(3)), tmp_path / "a.mp4", 64, 48, 30)
    assert not list(tmp_path.iterdir())


def test_bad_prefer_is_refused() -> None:
    with pytest.raises(ValueError):
        mp4.open_writer("x.mp4", 64, 48, 30, prefer="ffmpeg")  # type: ignore[arg-type]


def test_write_mp4_builtin(tmp_path: Path, fresh_status: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mp4, "_is_windows", lambda: False)
    seen: list[tuple[int, int]] = []
    out = mp4.write_mp4(lambda: iter(frames_of(12)), tmp_path / "v.mp4", 64, 48, 30,
                        progress=lambda k, b: seen.append((k, b)) or True)
    assert out == {"path": str(tmp_path / "v.mp4"), "frames": 12, "bytes": (tmp_path / "v.mp4").stat().st_size,
                   "encoder": "builtin"}
    assert [k for k, _b in seen] == list(range(1, 13))
    assert [b for _k, b in seen] == sorted(b for _k, b in seen)
    assert len(D.decode_file(tmp_path / "v.mp4")[2]) == 12


def test_write_mp4_falls_back_when_mf_fails_midway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   fresh_status: None) -> None:
    use_fake_mf(monkeypatch, fail_at=5)
    made_iters: list[str] = []

    def frames() -> Any:
        made_iters.append("start")
        try:
            yield from frames_of(9)
        finally:
            made_iters.append("closed")

    notes: list[str] = []
    target = tmp_path / "v.mp4"
    out = mp4.write_mp4(frames, target, 64, 48, 30, note=notes.append)
    assert out is not None and out["encoder"] == "builtin" and out["frames"] == 9
    assert notes == [S.EXP_MF_FALLBACK.format(reason="IMFSinkWriter::WriteSample 0xC00D36B4")]
    assert made_iters == ["start", "closed", "start", "closed"]  # 처음부터 다시
    assert FakeMf.made[0].aborted and FakeMf.made[0].frames == 4
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v.mp4"]
    _video, _sps, pictures, _info = D.decode_file(target)
    assert len(pictures) == 9


def test_write_mp4_uses_mf_when_it_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_status: None) -> None:
    use_fake_mf(monkeypatch)
    out = mp4.write_mp4(lambda: iter(frames_of(4)), tmp_path / "v.mp4", 64, 48, 30)
    assert out is not None and out["encoder"] == "mf" and out["frames"] == 4 and out["bytes"] == 400
    out = mp4.write_mp4(lambda: iter(frames_of(4)), tmp_path / "w.mp4", 64, 48, 30, prefer="builtin")
    assert out is not None and out["encoder"] == "builtin"


def test_write_mp4_cancel_keeps_the_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             fresh_status: None) -> None:
    monkeypatch.setattr(mp4, "_is_windows", lambda: False)
    target = tmp_path / "v.mp4"
    target.write_bytes(b"before")
    out = mp4.write_mp4(lambda: iter(frames_of(10)), target, 64, 48, 30, progress=lambda k, b: k < 3)
    assert out is None
    assert target.read_bytes() == b"before"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v.mp4"]


def test_write_mp4_without_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_status: None) -> None:
    monkeypatch.setattr(mp4, "_is_windows", lambda: False)
    with pytest.raises(ValueError, match=S.VIDEO_NO_FRAMES):
        mp4.write_mp4(lambda: iter([]), tmp_path / "v.mp4", 64, 48, 30)
    assert not list(tmp_path.iterdir())


def test_write_mp4_other_errors_clean_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_status: None) -> None:
    monkeypatch.setattr(mp4, "_is_windows", lambda: False)

    def frames() -> Any:
        yield from frames_of(2)
        raise RuntimeError("drawing failed")

    with pytest.raises(RuntimeError, match="drawing failed"):
        mp4.write_mp4(frames, tmp_path / "v.mp4", 64, 48, 30)
    assert not list(tmp_path.iterdir())


def test_mp4_modules_do_not_import_qt_and_write_without_it(tmp_path: Path) -> None:
    """작성기는 Qt 없이 불러오고 배열 프레임으로 쓴다(자식 프로세스가 QApplication 전에 작성기를 열 수 있고, Qt는 QImage를
    받았을 때만 불러온다)."""
    code = ("import sys, json, numpy as np; from evolab.app.export import mp4, mf_win, h264_pcm; "
            "w = mp4.BuiltinWriter(sys.argv[1], 32, 32, 30); w.add(np.zeros((32, 32), np.uint32)); w.close(); "
            "print(json.dumps(['PySide6' in sys.modules, mp4.encoder_status.__module__]))")
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "a.mp4")], capture_output=True, text=True,
                         timeout=120, check=True)
    assert json.loads(out.stdout) == [False, "evolab.app.export.mp4"]
    assert mp4.read_info(tmp_path / "a.mp4")["frames"] == 1

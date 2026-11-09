"""내장 H.264 쓰기 (v3 설계 D52, 4.16, T35·T36).

- T35: 고정 배경 + 움직이는 사각형 40프레임을 640×360(아래 8줄 채우고 자르기)·1280×720으로 쓰고, 테스트 복호기
  (tests/h264_subset_decoder.py, 쓰는 쪽 코드를 쓰지 않음)로 풀면 프레임마다 쓴 YUV와 같다. 프레임 수, 레벨이 표 A-1
  규칙대로, 같은 입력이면 같은 바이트, 취소하면 파일·임시 파일이 없다.
- T36: ffmpeg가 있으면 ffmpeg 복호도 쓴 YUV와 같고, ffprobe가 Constrained Baseline·bt709·tv·고른 레벨을 읽는다(없으면 건너뜀).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evolab.app import strings as S
from evolab.app.export import h264_pcm as H
from evolab.app.export import mp4

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h264_subset_decoder as D  # noqa: E402

BG = 0xFF101828
SQUARE = 0xFFFF7828


def moving_square(width: int, height: int, n: int = 40, size: int = 40) -> list[np.ndarray]:
    """고정 배경 + 오른쪽으로 움직이는 사각형(16의 배수가 아닌 자리를 지나 블록 경계를 걸침) + 위쪽 시각 띠."""
    out = []
    for k in range(n):
        a = np.full((height, width), BG, dtype=np.uint32)
        x = (13 + 9 * k) % (width - size)
        y = height // 3 + (k % 5)
        a[y:y + size, x:x + size] = SQUARE
        a[2:6, 2:2 + (k % 30) * 3] = 0xFFE0E0E0  # 시각 글상자 흉내(왼쪽 위 작은 변화)
        out.append(a)
    return out


def write_frames(path: Path, frames: list, fps: int = 30) -> tuple[mp4.BuiltinWriter, list[bytes]]:
    first = frames[0]
    h, w = first.shape[:2] if isinstance(first, np.ndarray) else (first.height(), first.width())
    writer = mp4.BuiltinWriter(path, w, h, fps)
    expected: list[bytes] = []
    for f in frames:
        writer.add(f)
        expected.append(writer.encoder.visible_yuv())
    assert writer.close() == len(frames)
    return writer, expected


def reference_yuv(px: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BT.709 제한 범위 float 기준식(쓰는 쪽의 정수식과 따로)."""
    r = ((px >> 16) & 0xFF).astype(np.float64)
    g = ((px >> 8) & 0xFF).astype(np.float64)
    b = (px & 0xFF).astype(np.float64)
    kr, kb = 0.2126, 0.0722
    kg = 1 - kr - kb
    y = 16 + 219 / 255 * (kr * r + kg * g + kb * b)
    cb_full = 224 / 255 * (b - (kr * r + kg * g + kb * b)) / (2 * (1 - kb))
    cr_full = 224 / 255 * (r - (kr * r + kg * g + kb * b)) / (2 * (1 - kr))
    h, w = px.shape

    def avg(c: np.ndarray) -> np.ndarray:
        return c.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))

    return y, 128 + avg(cb_full), 128 + avg(cr_full)


# ---------------------------------------------------------------------------
# 비트열 부품
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value, bits", [(0, "1"), (1, "010"), (2, "011"), (3, "00100"), (25, "000011010"),
                                         (30, "000011111"), (254, "000000011111111")])
def test_ue_codes(value: int, bits: str) -> None:
    w = H.BitWriter()
    w.ue(value)
    w.u(1, 1)
    w.align_zero()
    got = "".join(f"{b:08b}" for b in w.out)
    assert got.startswith(bits + "1")
    r = D.BitReader(bytes(w.out))
    assert r.ue() == value


@pytest.mark.parametrize("value", [0, 1, -1, 2, -2, 26, -26, 1000, -1000])
def test_se_round_trip(value: int) -> None:
    w = H.BitWriter()
    w.se(value)
    w.trailing()
    assert D.BitReader(bytes(w.out)).se() == value


@pytest.mark.parametrize("raw, escaped", [
    (b"\x00\x00\x00", b"\x00\x00\x03\x00"),
    (b"\x00\x00\x01\x00\x00\x02", b"\x00\x00\x03\x01\x00\x00\x03\x02"),
    (b"\x00\x00\x03", b"\x00\x00\x03\x03"),
    (b"\x00\x00\x04\x00\x00", b"\x00\x00\x04\x00\x00"),
    (b"\x00\x00\x00\x00\x00", b"\x00\x00\x03\x00\x00\x03\x00"),
    (b"\x12\x34", b"\x12\x34"),
])
def test_emulation_prevention(raw: bytes, escaped: bytes) -> None:
    assert H.ebsp(raw) == escaped
    assert D.unescape(escaped)[0] == raw


def test_aligned_mb_headers() -> None:
    assert H._IDR_MB_HEADER == b"\x0d\x00"  # ue(25) + 0 7개
    w = H.BitWriter()
    w.ue(0)
    w.ue(30)
    w.align_zero()
    assert H._aligned_header(0, 30) == bytes(w.out) == b"\x87\xc0"


# ---------------------------------------------------------------------------
# 레벨 (표 A-1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("w_mbs, h_mbs, fps, bits_1s, frame_bits, level, ok", [
    (4, 4, 30, 10_000, 10_000, 10, True),            # 64×64: 16 MB, 480 MB/s
    (4, 4, 30, 100_000, 10_000, 11, True),           # 64 kbit/s(1)를 넘으면 1.1
    (40, 23, 30, 2_000_000, 2_826_240, 30, True),    # 640×368: 920 MB × 30 = 27,600 MB/s > 2.2의 20,250
    (40, 23, 20, 2_000_000, 2_826_240, 22, True),    # 20fps면 18,400 MB/s → 2.2
    (80, 45, 30, 12_000_000, 11_059_200, 31, True),  # 720p 30fps: 108,000 MB/s = 3.1 상한
    (80, 45, 30, 16_000_000, 11_059_200, 32, True),  # 14 Mbit/s를 넘으면 3.2
    (80, 45, 30, 21_000_000, 11_059_200, 41, True),  # 20 Mbit/s(3.2·4)를 넘으면 4.1
    (80, 45, 60, 21_000_000, 11_059_200, 41, True),  # 60fps: 216,000 MB/s ≤ 4.1의 245,760
    (80, 45, 30, 1_000_000, 15_000_000, 32, True),   # 프레임 하나가 14 Mbit(3.1 CPB)를 넘으면 3.2
    (120, 68, 30, 21_000_000, 1, 41, True),          # 1920×1088: 8,160 MB ≤ 8,192, 244,800 MB/s ≤ 245,760
    (500, 4, 30, 1, 1, 51, True),                    # 폭 500 MB: √(8·MaxFS) ≥ 500인 첫 레벨 5.1
    (600, 4, 1, 1, 1, 52, False),                    # 폭 600 MB > √(8·36864) = 543: 어느 레벨에도 안 듦
])
def test_pick_level(w_mbs: int, h_mbs: int, fps: int, bits_1s: float, frame_bits: float, level: int, ok: bool) -> None:
    assert H.pick_level(w_mbs, h_mbs, fps, bits_1s, frame_bits) == (level, ok)


def test_level_rule_is_the_lowest_level_that_fits() -> None:
    """표의 모든 행에서: 고른 레벨은 조건을 만족하고, 그보다 낮은 레벨은 모두 하나 이상 어긴다(임의 입력 200개)."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        w, h = int(rng.integers(1, 130)), int(rng.integers(1, 80))
        fps = int(rng.choice([1, 10, 20, 25, 30, 60]))
        bits = float(rng.uniform(0, 300e6))
        frame = float(rng.uniform(0, 300e6))
        level, ok = H.pick_level(w, h, fps, bits, frame)

        def fits(row: tuple[int, int, int, int, int], w: int = w, h: int = h, fps: int = fps, bits: float = bits,
                 frame: float = frame) -> bool:
            _lv, mbps, fs, br, cpb = row
            side = (8 * fs) ** 0.5
            return (w * h <= fs and w <= side and h <= side and w * h * fps <= mbps and bits <= br * 1000
                    and frame <= cpb * 1000)

        idx = [row[0] for row in H.LEVELS].index(level)
        if ok:
            assert fits(H.LEVELS[idx])
            assert not any(fits(row) for row in H.LEVELS[:idx])
        else:
            assert level == H.MAX_LEVEL and not any(fits(row) for row in H.LEVELS)


# ---------------------------------------------------------------------------
# 색
# ---------------------------------------------------------------------------
def test_colour_conversion_matches_bt709_float_reference() -> None:
    rng = np.random.default_rng(3)
    px = rng.integers(0, 1 << 24, size=(64, 96), dtype=np.uint32) | np.uint32(0xFF000000)
    y, u, v = H.rgb32_to_yuv420(px)
    ry, ru, rv = reference_yuv(px)
    assert np.abs(y.astype(np.float64) - ry).max() <= 0.51 + 1e-9
    assert np.abs(u.astype(np.float64) - ru).max() <= 0.51 + 1e-9
    assert np.abs(v.astype(np.float64) - rv).max() <= 0.51 + 1e-9
    # 블록 변환(I_PCM 바이트)도 같은 식
    blocks = px.reshape(4, 16, 6, 16).transpose(0, 2, 1, 3).reshape(24, 16, 16)
    mbs = H.yuv_blocks(blocks)
    assert np.array_equal(mbs[0, :256].reshape(16, 16), y[:16, :16])
    assert np.array_equal(mbs[0, 256:320].reshape(8, 8), u[:8, :8])
    assert np.array_equal(mbs[7, 320:].reshape(8, 8), v[8:16, 8:16])


@pytest.mark.parametrize("rgb, yuv", [((0, 0, 0), (16, 128, 128)), ((255, 255, 255), (235, 128, 128)),
                                      ((128, 128, 128), (126, 128, 128)), ((255, 0, 0), (63, 102, 240)),
                                      ((0, 255, 0), (173, 42, 26)), ((0, 0, 255), (32, 240, 118))])
def test_colour_conversion_known_values(rgb: tuple[int, int, int], yuv: tuple[int, int, int]) -> None:
    px = np.full((2, 2), 0xFF000000 | (rgb[0] << 16) | (rgb[1] << 8) | rgb[2], dtype=np.uint32)
    y, u, v = H.rgb32_to_yuv420(px)
    assert (int(y[0, 0]), int(u[0, 0]), int(v[0, 0])) == yuv


# ---------------------------------------------------------------------------
# T35 왕복
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width, height", [(640, 360), (1280, 720)])
def test_round_trip_decodes_to_the_written_yuv(tmp_path: Path, width: int, height: int) -> None:
    frames = moving_square(width, height)
    path = tmp_path / "square.mp4"
    writer, expected = write_frames(path, frames)
    video, sps, pictures, info = D.decode_file(path)
    assert len(pictures) == len(frames) == 40
    assert [D.i420(p) for p in pictures] == expected
    assert (video.width, video.height) == (width, height)
    assert (sps["width"], sps["height"]) == (width, height)
    coded_h = -(-height // 16) * 16
    assert sps["height_mbs"] * 16 == coded_h
    assert sps["crop"] == (0, 0, 0, (coded_h - height) // 2)
    assert (sps["profile_idc"], sps["constraint_flags"]) == (66, 0xC0)
    assert sps["level_idc"] == writer.level
    vui = sps["vui"]
    assert (vui["colour_primaries"], vui["transfer"], vui["matrix"], vui["full_range"]) == (1, 1, 1, 0)
    assert (vui["time_scale"], vui["num_units_in_tick"], vui["fixed_frame_rate"]) == (60, 1, 1)
    assert vui["max_dec_frame_buffering"] == 1 and vui["max_num_reorder_frames"] == 0
    # 첫 프레임은 IDR(모든 블록), 그 뒤는 바뀐 블록만
    assert info[0].idr and info[0].pcm_blocks == sps["width_mbs"] * sps["height_mbs"]
    assert not any(f.idr for f in info[1:])
    assert [f.pcm_blocks for f in info] == writer.encoder.changed
    assert max(f.pcm_blocks for f in info[1:]) <= 32  # 사각형의 앞뒤 자리 + 시각 띠만
    assert video.sync == [1]
    # 레벨은 표 A-1 규칙(닫을 때 크기로)
    sizes = np.asarray([len(s) for s in video.samples], dtype=np.int64)
    win = min(len(sizes), 30)
    csum = np.concatenate([[0], np.cumsum(sizes)])
    bits = float((csum[win:] - csum[:-win]).max()) * 8 * 30 / win
    assert H.pick_level(sps["width_mbs"], sps["height_mbs"], 30, bits, float(sizes.max()) * 8) == (writer.level, True)


def test_decoded_pictures_match_the_source_colours(tmp_path: Path) -> None:
    """복호한 그림이 원본 RGB의 BT.709 기준식과 ±0.5(반올림) 안: 「복원 = 입력」을 YUV로."""
    rng = np.random.default_rng(11)
    frames = []
    base = rng.integers(0, 1 << 24, size=(48, 80), dtype=np.uint32) | np.uint32(0xFF000000)
    for k in range(6):
        a = base.copy()
        a[8 * k:8 * k + 8, :] = rng.integers(0, 1 << 24, size=(8, 80), dtype=np.uint32) | np.uint32(0xFF000000)
        frames.append(a)
    path = tmp_path / "noise.mp4"
    write_frames(path, frames)
    _video, _sps, pictures, _info = D.decode_file(path)
    for src, (y, u, v) in zip(frames, pictures, strict=True):
        ry, ru, rv = reference_yuv(src)
        assert np.abs(y - ry).max() <= 0.51 + 1e-9
        assert np.abs(u - ru).max() <= 0.51 + 1e-9
        assert np.abs(v - rv).max() <= 0.51 + 1e-9


def test_every_block_changing_and_no_block_changing(tmp_path: Path) -> None:
    """따라가기 카메라처럼 모든 블록이 바뀌는 프레임과, 아무것도 바뀌지 않는 프레임(건너뛰기 하나)."""
    rng = np.random.default_rng(5)
    frames = [rng.integers(0, 1 << 24, size=(64, 96), dtype=np.uint32) | np.uint32(0xFF000000) for _ in range(3)]
    frames.insert(2, frames[1].copy())
    path = tmp_path / "all.mp4"
    writer, expected = write_frames(path, frames)
    _video, _sps, pictures, info = D.decode_file(path)
    assert [D.i420(p) for p in pictures] == expected
    assert writer.encoder.changed == [24, 24, 0, 24]
    assert [f.skipped for f in info] == [0, 0, 24, 0]


@pytest.mark.parametrize("width, height", [(2, 2), (18, 34), (100, 30), (16, 16)])
def test_small_and_odd_multiple_sizes(tmp_path: Path, width: int, height: int) -> None:
    frames = [np.full((height, width), 0xFF000000 | (40 * k << 8), dtype=np.uint32) for k in range(3)]
    frames[1][0, 0] = 0xFFFFFFFF
    path = tmp_path / "small.mp4"
    _writer, expected = write_frames(path, frames)
    video, _sps, pictures, _info = D.decode_file(path)
    assert (video.width, video.height) == (width, height)
    assert [D.i420(p) for p in pictures] == expected


@pytest.mark.parametrize("width, height, fps", [(3, 4, 30), (4, 5, 30), (0, 4, 30), (4, 4, 0), (4, 4, 241)])
def test_bad_sizes_and_fps_are_refused(width: int, height: int, fps: int) -> None:
    with pytest.raises(ValueError):
        H.PcmEncoder(width, height, fps)


def test_wrong_frame_size_is_refused(tmp_path: Path) -> None:
    writer = mp4.BuiltinWriter(tmp_path / "x.mp4", 32, 32, 30)
    with pytest.raises(ValueError):
        writer.add(np.zeros((16, 32), dtype=np.uint32))
    writer.abort()
    assert not list(tmp_path.iterdir())


def test_same_input_gives_the_same_bytes(tmp_path: Path) -> None:
    frames = moving_square(320, 180, n=12)
    write_frames(tmp_path / "a.mp4", frames)
    write_frames(tmp_path / "b.mp4", [f.copy() for f in frames])
    assert (tmp_path / "a.mp4").read_bytes() == (tmp_path / "b.mp4").read_bytes()


def test_bgra_bytes_and_uint32_views_give_the_same_file(tmp_path: Path) -> None:
    frames = moving_square(64, 48, n=5)
    write_frames(tmp_path / "u32.mp4", frames)
    bgra = [np.ascontiguousarray(f).view(np.uint8).reshape(48, 64, 4) for f in frames]
    write_frames(tmp_path / "bgra.mp4", bgra)
    assert (tmp_path / "u32.mp4").read_bytes() == (tmp_path / "bgra.mp4").read_bytes()


def test_qimage_frames_match_array_frames(tmp_path: Path, qapp) -> None:
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QColor, QImage, QPainter

    images, arrays = [], []
    for k in range(4):
        img = QImage(96, 54, QImage.Format.Format_RGB32)
        img.fill(QColor(16, 24, 40))
        p = QPainter(img)
        p.fillRect(QRect(5 + 10 * k, 10, 20, 20), QColor(255, 120, 40))
        p.end()
        images.append(img)
        arrays.append(np.array([[img.pixel(x, y) for x in range(96)] for y in range(54)], dtype=np.uint32))
    # 다른 형식(RGB888)의 QImage도 RGB32로 바꿔 받는다
    images[3] = images[3].convertToFormat(QImage.Format.Format_RGB888)
    write_frames(tmp_path / "q.mp4", images)
    write_frames(tmp_path / "a.mp4", arrays)
    assert (tmp_path / "q.mp4").read_bytes() == (tmp_path / "a.mp4").read_bytes()


def test_abort_removes_the_temp_file_and_keeps_the_old_file(tmp_path: Path) -> None:
    target = tmp_path / "out.mp4"
    target.write_bytes(b"old video")
    writer = mp4.BuiltinWriter(target, 64, 64, 30)
    writer.add(np.zeros((64, 64), dtype=np.uint32))
    assert mp4.temp_path(target).exists()
    assert writer.bytes_written() > 64 * 64
    writer.abort()
    writer.abort()  # 두 번 불러도 됨
    assert target.read_bytes() == b"old video"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.mp4"]
    with pytest.raises(ValueError):
        writer.add(np.zeros((64, 64), dtype=np.uint32))


def test_close_without_frames_leaves_nothing(tmp_path: Path) -> None:
    writer = mp4.BuiltinWriter(tmp_path / "empty.mp4", 64, 64, 30)
    with pytest.raises(ValueError, match=S.VIDEO_NO_FRAMES):
        writer.close()
    assert not list(tmp_path.iterdir())


def test_abort_after_close_keeps_the_file(tmp_path: Path) -> None:
    path = tmp_path / "done.mp4"
    writer, _expected = write_frames(path, moving_square(64, 64, n=3))
    writer.abort()
    assert path.exists() and not mp4.temp_path(path).exists()


def test_korean_folder_and_name(tmp_path: Path) -> None:
    folder = tmp_path / "진화 실험실 내보내기"
    path = folder / "챔피언 영상.mp4"
    write_frames(path, moving_square(64, 48, n=4))
    assert D.decode_file(path)[2][0][0].shape == (48, 64)


def test_command_line_decoder_reports_ok(tmp_path: Path) -> None:
    path = tmp_path / "cli 확인.mp4"
    write_frames(path, moving_square(160, 90, n=8))
    run = subprocess.run([sys.executable, str(Path(__file__).with_name("h264_subset_decoder.py")), str(path)],
                         capture_output=True, text=True, timeout=120, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip().endswith("OK")
    assert "decoded: 8 frames" in run.stdout
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(path.read_bytes()[:-40])
    run = subprocess.run([sys.executable, str(Path(__file__).with_name("h264_subset_decoder.py")), str(bad)],
                         capture_output=True, text=True, timeout=120, check=False)
    assert run.returncode == 1 and "FAIL" in run.stdout


def test_decoder_refuses_what_it_does_not_understand() -> None:
    """복호기가 아무 비트열이나 통과시키지 않는다: 디블로킹 켬, 다른 mb_type, frame_num 건너뜀."""
    enc = H.PcmEncoder(32, 32, 30)
    sps = D.parse_sps(D.unescape(enc.sps_nal()[1:])[0])
    pps = D.parse_pps(D.unescape(enc.pps_nal()[1:])[0])
    frame = np.full((32, 32), BG, dtype=np.uint32)
    idr = enc.encode(frame)
    frame2 = frame.copy()
    frame2[0, 0] = SQUARE
    p1 = enc.encode(frame2)
    p2 = enc.encode(frame)
    dec = D.Decoder(sps, pps)
    dec.decode_nal(idr)
    with pytest.raises(D.BadStream, match="frame_num"):
        dec.decode_nal(p2)  # p1을 건너뜀
    # mb_type 0(I_16x16 …)이 든 IDR을 손으로: 머리는 같고 첫 mb_type만 바꿈
    w = H.BitWriter()
    for v in (0, 7, 0):
        w.ue(v)
    w.u(0, 16)
    w.ue(0)
    w.u(0, 1)
    w.u(0, 1)
    w.se(0)
    w.ue(1)
    w.ue(0)  # mb_type 0
    w.trailing()
    with pytest.raises(D.Unsupported, match="mb_type"):
        D.Decoder(sps, pps).decode_nal(H.nal(3, 5, bytes(w.out)))
    assert len(p1) > 0


# ---------------------------------------------------------------------------
# T36 ffmpeg (있을 때만)
# ---------------------------------------------------------------------------
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not installed (test-only tool)")
@pytest.mark.parametrize("width, height", [(640, 360), (1280, 720)])
def test_ffmpeg_decodes_to_the_written_yuv(tmp_path: Path, width: int, height: int) -> None:
    path = tmp_path / "square.mp4"
    writer, expected = write_frames(path, moving_square(width, height))
    env = dict(os.environ, AV_LOG_FORCE_NOCOLOR="1")
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                         capture_output=True, timeout=120, env=env, check=True).stdout
    size = width * height * 3 // 2
    assert len(raw) == size * len(expected)
    assert [raw[k * size:(k + 1) * size] for k in range(len(expected))] == expected
    probe = subprocess.run([FFPROBE, "-v", "error", "-show_entries",
                            ("stream=codec_name,profile,level,width,height,pix_fmt,color_range,color_space,"
                             "color_primaries,color_transfer,nb_frames,r_frame_rate"), "-of", "json", str(path)],
                           capture_output=True, text=True, timeout=60, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["profile"] == "Constrained Baseline"
    assert (stream["width"], stream["height"]) == (width, height)
    assert stream["pix_fmt"] == "yuv420p"
    assert (stream["color_range"], stream["color_space"], stream["color_primaries"], stream["color_transfer"]) == \
        ("tv", "bt709", "bt709", "bt709")
    assert int(stream["level"]) == writer.level
    assert stream["nb_frames"] == "40" and stream["r_frame_rate"] == "30/1"

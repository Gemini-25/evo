"""내장 H.264 쓰기 (v3 설계 D52, 2.7, 4.16). numpy와 표준 라이브러리만 쓴다(새 의존성 없음, Qt도 부르지 않음).

H.264 Constrained Baseline, CAVLC, 조각 하나. 첫 프레임(IDR)은 모든 16×16 매크로블록을 I_PCM(압축하지 않은 YUV 4:2:0,
384바이트)으로, 그 뒤 프레임(P)은 앞 프레임과 화소가 다른 블록만 I_PCM으로 쓰고 나머지는 P_Skip(앞 프레임 그대로)으로
건너뛴다. 디블로킹을 끄고(조각마다 disable_deblocking_filter_idc 1) 모든 움직임 벡터가 0이 되므로(인트라 블록의 벡터는 0,
P_Skip의 예측은 이웃의 중앙값이라 귀납으로 0) **디코더가 내는 그림 = 이 모듈이 쓴 YUV**(손실 없음, 어긋남이 쌓이지 않음).

- 색: QImage(Format_RGB32)의 화소를 uint32(0xffRRGGBB)로 보고 BT.709 제한 범위(Y 16..235, Cb·Cr 16..240)로 바꾼다.
  고정소수점 정수 식이라 같은 입력이면 어느 PC에서나 같은 바이트다. 크로마는 2×2 화소의 평균(합을 한 번에 나눔).
  VUI에 BT.709·제한 범위를 적는다. 제한 범위라 PCM 바이트에 0이 없다(시작 코드 흉내 방지는 그래도 한다).
- 크기: 너비·높이는 짝수. 16의 배수가 아니면 아래·오른쪽을 가장자리 화소로 채워 16의 배수로 만들고 SPS의
  frame_cropping으로 잘라 보이게 한다(4:2:0이라 자르기 단위는 2화소).
- 레벨: 닫을 때(finish) 고른다. 프레임 MB 수·폭·높이, 초당 MB 수, 1초 창(프레임 fps개)의 최대 비트수, 가장 큰 프레임의
  비트수로 표 A-1(Baseline)에서 가장 낮은 레벨. 넘으면 5.2. SPS·PPS는 비트열 안에 넣지 않고 MP4 avcC에만 둔다
  (mp4.Mp4Muxer). 그래서 레벨을 닫을 때 정할 수 있다.
- 결정적: 같은 프레임들이면 같은 바이트. 메모리는 앞 프레임 화소 한 장 + 지금 YUV 블록 표 한 장.

설계 원형(share proto/h264pcm.py, proto2/pcm_fast.py)을 옮기며 바꾼 것: 색 변환을 float32에서 고정소수점 정수로(어느 PC에서나
같은 바이트, 차이는 ±1 안), 블록 머리 비트를 미리 만든 바이트로(따라가기 카메라에서 블록마다 비트 쓰기를 부르지 않음),
낮은 레벨(1~2.2)과 프레임 비트수 ≤ MaxCPB 조건을 표에 더함, VUI에 timing_info(fps)를 더함.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np

PROFILE_IDC: Final = 66
"""Baseline. constraint_set0·1을 켜서 Constrained Baseline으로 알린다(CONSTRAINT_FLAGS)."""
CONSTRAINT_FLAGS: Final = 0xC0
"""constraint_set0_flag = constraint_set1_flag = 1, 나머지 0."""
LOG2_MAX_FRAME_NUM: Final = 16
MB_BYTES: Final = 384
"""I_PCM 매크로블록 하나: Y 16×16 + Cb 8×8 + Cr 8×8."""

LEVELS: Final[tuple[tuple[int, int, int, int, int], ...]] = (
    # (level_idc, MaxMBPS, MaxFS, MaxBR, MaxCPB) — H.264 표 A-1. MaxBR·MaxCPB는 1000비트 단위(Baseline VCL 배수 1000).
    # 레벨 1b는 constraint_set3로 알려야 해서 뺐다.
    (10, 1485, 99, 64, 175),
    (11, 3000, 396, 192, 500),
    (12, 6000, 396, 384, 1000),
    (13, 11880, 396, 768, 2000),
    (20, 11880, 396, 2000, 2000),
    (21, 19800, 792, 4000, 4000),
    (22, 20250, 1620, 4000, 4000),
    (30, 40500, 1620, 10000, 10000),
    (31, 108000, 3600, 14000, 14000),
    (32, 216000, 5120, 20000, 20000),
    (40, 245760, 8192, 20000, 25000),
    (41, 245760, 8192, 50000, 62500),
    (42, 522240, 8704, 50000, 62500),
    (50, 589824, 22080, 135000, 135000),
    (51, 983040, 36864, 240000, 240000),
    (52, 2073600, 36864, 240000, 240000),
)
MAX_LEVEL: Final = 52

# BT.709 제한 범위, 2^16 배 고정소수점(행마다 합 = 219/255 · 2^16, 크로마 행은 합 0).
_KY = (11966, 40254, 4064)  # R, G, B
_KCB = (-6596, -22189, 28785)
_KCR = (28785, -26145, -2640)


def pick_level(width_mbs: int, height_mbs: int, fps: int, max_bits_1s: float, max_frame_bits: float) -> tuple[int, bool]:
    """(level_idc, 표 안에 드는가). 표 A-1에서 모든 조건을 만족하는 가장 낮은 레벨, 없으면 (52, False).

    조건: 프레임 MB 수 ≤ MaxFS, 폭·높이(MB) ≤ √(8·MaxFS), 초당 MB 수 ≤ MaxMBPS, 1초 창 최대 비트수 ≤ MaxBR·1000,
    가장 큰 프레임 비트수 ≤ MaxCPB·1000(프레임 하나가 코드 그림 버퍼에 들어가야 함).
    """
    fs = int(width_mbs) * int(height_mbs)
    for level, max_mbps, max_fs, max_br, max_cpb in LEVELS:
        side = math.sqrt(8 * max_fs)
        if (fs <= max_fs and width_mbs <= side and height_mbs <= side and fs * fps <= max_mbps
                and max_bits_1s <= max_br * 1000 and max_frame_bits <= max_cpb * 1000):
            return level, True
    return MAX_LEVEL, False


# ---------------------------------------------------------------------------
# 비트 쓰기
# ---------------------------------------------------------------------------
class BitWriter:
    """MSB 먼저 비트를 모은다(H.264 RBSP). u(n), ue(v), se(v), 0비트 정렬, rbsp 끝 비트."""

    __slots__ = ("out", "acc", "n")

    def __init__(self) -> None:
        self.out = bytearray()
        self.acc = 0
        self.n = 0

    def u(self, value: int, bits: int) -> None:
        if bits <= 0:
            return
        self.acc = (self.acc << bits) | (int(value) & ((1 << bits) - 1))
        self.n += bits
        while self.n >= 8:
            self.n -= 8
            self.out.append((self.acc >> self.n) & 0xFF)
        self.acc &= (1 << self.n) - 1

    def ue(self, value: int) -> None:
        if value < 0:
            raise ValueError("ue(v) needs a non-negative value")
        v = int(value) + 1
        self.u(v, 2 * v.bit_length() - 1)  # (길이−1)개의 0 + v

    def se(self, value: int) -> None:
        value = int(value)
        self.ue(2 * value - 1 if value > 0 else -2 * value)

    @property
    def aligned(self) -> bool:
        return self.n == 0

    def align_zero(self) -> None:
        if self.n:
            self.u(0, 8 - self.n)

    def trailing(self) -> None:
        """rbsp_trailing_bits: 1 하나와 바이트 끝까지 0."""
        self.u(1, 1)
        self.align_zero()

    def raw(self, data: bytes | bytearray | memoryview) -> None:
        if self.n:
            raise ValueError("raw bytes need a byte-aligned writer")
        self.out += data


def ebsp(rbsp: bytes | bytearray) -> bytes:
    """시작 코드 흉내 방지: 00 00 뒤에 00·01·02·03이 오면 그 앞에 03을 끼운다(7.4.1)."""
    data = bytes(rbsp)
    if b"\x00\x00" not in data:
        return data
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        j = data.find(b"\x00\x00", i)
        if j < 0 or j + 2 >= n:
            out += data[i:]
            break
        out += data[i:j + 2]
        if data[j + 2] <= 3:
            out.append(3)
        i = j + 2
    return bytes(out)


def nal(ref_idc: int, kind: int, rbsp: bytes | bytearray) -> bytes:
    """NAL 단위(머리 1바이트 + 흉내 방지한 RBSP). 시작 코드·길이는 붙이지 않는다."""
    return bytes([((ref_idc & 3) << 5) | (kind & 31)]) + ebsp(rbsp)


def _ue_bits(value: int) -> tuple[int, int]:
    v = value + 1
    return v, 2 * v.bit_length() - 1


def _aligned_header(*codes: int) -> bytes:
    """바이트 경계에서 시작해 ue 값들을 쓰고 0으로 바이트를 채운 것(I_PCM 블록 앞 머리)."""
    w = BitWriter()
    for c in codes:
        w.ue(c)
    w.align_zero()
    return bytes(w.out)


_IDR_MB_HEADER: Final = _aligned_header(25)
"""I 조각의 I_PCM(mb_type 25) + pcm_alignment_zero_bits: 바이트 경계에서 시작하면 늘 0x0D 0x00."""
MB_TYPE_I_PCM_IN_I: Final = 25
MB_TYPE_I_PCM_IN_P: Final = 30
"""P 조각의 I_PCM(5 + 25)."""


# ---------------------------------------------------------------------------
# 색 변환
# ---------------------------------------------------------------------------
def _channels(px: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uint32(0x??RRGGBB) 배열 → R, G, B int32(모양 그대로). 바이트 순서와 무관(값으로 자름)."""
    v = px.astype(np.int32, copy=False) if px.dtype == np.uint32 else px.astype(np.uint32).astype(np.int32)
    return (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF


def yuv_blocks(blocks: np.ndarray) -> np.ndarray:
    """(n, 16, 16) uint32 RGB32 블록 → (n, 384) uint8 I_PCM 바이트(Y 256 래스터, Cb 64, Cr 64)."""
    n = blocks.shape[0]
    r, g, b = _channels(blocks)
    y = (_KY[0] * r + _KY[1] * g + _KY[2] * b + (16 << 16) + (1 << 15)) >> 16
    # 2×2 합(0..1020)에 계수를 곱하고 4·2^16으로 나눔(반올림)
    def quad(c: np.ndarray) -> np.ndarray:
        return c.reshape(n, 8, 2, 8, 2).sum(axis=(2, 4))

    sr, sg, sb = quad(r), quad(g), quad(b)
    cb = (_KCB[0] * sr + _KCB[1] * sg + _KCB[2] * sb + (128 << 18) + (1 << 17)) >> 18
    cr = (_KCR[0] * sr + _KCR[1] * sg + _KCR[2] * sb + (128 << 18) + (1 << 17)) >> 18
    out = np.empty((n, MB_BYTES), dtype=np.uint8)
    out[:, :256] = np.clip(y, 16, 235).reshape(n, 256)
    out[:, 256:320] = np.clip(cb, 16, 240).reshape(n, 64)
    out[:, 320:] = np.clip(cr, 16, 240).reshape(n, 64)
    return out


def rgb32_to_yuv420(px: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(H, W) uint32 RGB32(H·W 짝수) → Y(H, W), Cb(H/2, W/2), Cr(H/2, W/2) uint8, yuv_blocks와 같은 식.
    Media Foundation 쓰기의 NV12 입력과 테스트가 쓴다."""
    h, w = px.shape
    if h % 2 or w % 2:
        raise ValueError(f"even size needed, got {w}x{h}")
    r, g, b = _channels(px)
    y = (_KY[0] * r + _KY[1] * g + _KY[2] * b + (16 << 16) + (1 << 15)) >> 16

    def quad(c: np.ndarray) -> np.ndarray:
        return c.reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))

    sr, sg, sb = quad(r), quad(g), quad(b)
    cb = (_KCB[0] * sr + _KCB[1] * sg + _KCB[2] * sb + (128 << 18) + (1 << 17)) >> 18
    cr = (_KCR[0] * sr + _KCR[1] * sg + _KCR[2] * sb + (128 << 18) + (1 << 17)) >> 18
    return (np.clip(y, 16, 235).astype(np.uint8), np.clip(cb, 16, 240).astype(np.uint8),
            np.clip(cr, 16, 240).astype(np.uint8))


def as_rgb32(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """프레임 배열 → (높이, 너비) uint32 보기. (H, W) uint32 또는 (H, W, 4) uint8 BGRA(QImage Format_RGB32의 메모리,
    리틀 엔디언)를 받는다. 크기가 다르면 ValueError."""
    a = np.asarray(frame)
    if a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 4:
        a = np.ascontiguousarray(a).view(np.uint32).reshape(a.shape[0], a.shape[1])
        if np.little_endian is False:  # pragma: no cover - 빅 엔디언 기계 없음
            a = a.byteswap()
    if a.dtype != np.uint32 or a.ndim != 2:
        raise ValueError(f"frame must be (H, W) uint32 or (H, W, 4) uint8, got {a.dtype} {a.shape}")
    if a.shape != (height, width):
        raise ValueError(f"frame size {a.shape[1]}x{a.shape[0]} != {width}x{height}")
    return a


# ---------------------------------------------------------------------------
# 부호기
# ---------------------------------------------------------------------------
class PcmEncoder:
    """RGB32 프레임을 받아 H.264 NAL 단위(조각 하나, 길이·시작 코드 없음)를 만든다. 파일은 쓰지 않는다(mp4.py가 담음).

    encode(프레임) → NAL 바이트. finish() → 레벨(그 뒤 sps_nal()·pps_nal()이 그 레벨). frame_sizes는 프레임마다 NAL 바이트 수.
    """

    def __init__(self, width: int, height: int, fps: int) -> None:
        for name, v in (("width", width), ("height", height)):
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < 2 or int(v) % 2:
                raise ValueError(f"{name} must be a positive even integer, got {v!r}")
        if isinstance(fps, bool) or not isinstance(fps, (int, np.integer)) or not 1 <= int(fps) <= 240:
            raise ValueError(f"fps must be an integer 1..240, got {fps!r}")
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.mbw, self.mbh = -(-self.width // 16), -(-self.height // 16)
        self.pw, self.ph = self.mbw * 16, self.mbh * 16
        self.n_mb = self.mbw * self.mbh
        self.crop = ((self.pw - self.width) // 2, (self.ph - self.height) // 2)
        """(오른쪽, 아래) 자르기 값(2화소 단위, SPS frame_crop_right/bottom_offset)."""
        self.level = 0
        self.level_ok = True
        self.max_bits_1s = 0.0
        self.frame_sizes: list[int] = []
        self.changed: list[int] = []
        """프레임마다 I_PCM으로 쓴 블록 수(첫 프레임은 전체)."""
        self.mbs = np.zeros((self.n_mb, MB_BYTES), dtype=np.uint8)
        """디코더가 지금 가진 그림(매크로블록 래스터 순서, 블록마다 I_PCM 바이트)."""
        self._cur = np.zeros((self.ph, self.pw), dtype=np.uint32)
        self._prev = np.zeros((self.ph, self.pw), dtype=np.uint32)
        self._diff = np.zeros((self.ph, self.pw), dtype=bool)
        self._frame_num = 0
        self._run_headers: dict[int, bytes] = {}

    # ---- 매개변수 집합
    def sps_rbsp(self, level: int | None = None) -> bytes:
        b = BitWriter()
        b.u(PROFILE_IDC, 8)
        b.u(CONSTRAINT_FLAGS, 8)
        b.u(self.level if level is None else level, 8)
        b.ue(0)  # seq_parameter_set_id
        b.ue(LOG2_MAX_FRAME_NUM - 4)
        b.ue(2)  # pic_order_cnt_type 2: 출력 순서 = 복호 순서
        b.ue(1)  # max_num_ref_frames
        b.u(0, 1)  # gaps_in_frame_num_value_allowed_flag
        b.ue(self.mbw - 1)
        b.ue(self.mbh - 1)
        b.u(1, 1)  # frame_mbs_only_flag
        b.u(1, 1)  # direct_8x8_inference_flag
        cx, cy = self.crop
        if cx or cy:
            b.u(1, 1)
            b.ue(0)
            b.ue(cx)
            b.ue(0)
            b.ue(cy)
        else:
            b.u(0, 1)
        b.u(1, 1)  # vui_parameters_present_flag
        b.u(1, 1)  # aspect_ratio_info_present_flag
        b.u(1, 8)  # aspect_ratio_idc 1 = 1:1
        b.u(0, 1)  # overscan_info_present_flag
        b.u(1, 1)  # video_signal_type_present_flag
        b.u(5, 3)  # video_format 5 = 정하지 않음
        b.u(0, 1)  # video_full_range_flag 0 = 제한 범위
        b.u(1, 1)  # colour_description_present_flag
        b.u(1, 8)  # colour_primaries BT.709
        b.u(1, 8)  # transfer_characteristics BT.709
        b.u(1, 8)  # matrix_coefficients BT.709
        b.u(0, 1)  # chroma_loc_info_present_flag
        b.u(1, 1)  # timing_info_present_flag
        b.u(1, 32)  # num_units_in_tick
        b.u(2 * self.fps, 32)  # time_scale(프레임 하나 = 틱 둘)
        b.u(1, 1)  # fixed_frame_rate_flag
        b.u(0, 1)  # nal_hrd_parameters_present_flag
        b.u(0, 1)  # vcl_hrd_parameters_present_flag
        b.u(0, 1)  # pic_struct_present_flag
        b.u(1, 1)  # bitstream_restriction_flag
        b.u(1, 1)  # motion_vectors_over_pic_boundaries_flag
        b.ue(0)  # max_bytes_per_pic_denom
        b.ue(0)  # max_bits_per_mb_denom
        b.ue(16)  # log2_max_mv_length_horizontal
        b.ue(16)  # log2_max_mv_length_vertical
        b.ue(0)  # max_num_reorder_frames
        b.ue(1)  # max_dec_frame_buffering
        b.trailing()
        return bytes(b.out)

    @staticmethod
    def pps_rbsp() -> bytes:
        b = BitWriter()
        b.ue(0)  # pic_parameter_set_id
        b.ue(0)  # seq_parameter_set_id
        b.u(0, 1)  # entropy_coding_mode_flag 0 = CAVLC
        b.u(0, 1)  # bottom_field_pic_order_in_frame_present_flag
        b.ue(0)  # num_slice_groups_minus1
        b.ue(0)  # num_ref_idx_l0_default_active_minus1
        b.ue(0)  # num_ref_idx_l1_default_active_minus1
        b.u(0, 1)  # weighted_pred_flag
        b.u(0, 2)  # weighted_bipred_idc
        b.se(-26)  # pic_init_qp_minus26 → QP 0
        b.se(0)  # pic_init_qs_minus26
        b.se(0)  # chroma_qp_index_offset
        b.u(1, 1)  # deblocking_filter_control_present_flag
        b.u(0, 1)  # constrained_intra_pred_flag
        b.u(0, 1)  # redundant_pic_cnt_present_flag
        b.trailing()
        return bytes(b.out)

    def sps_nal(self) -> bytes:
        return nal(3, 7, self.sps_rbsp())

    def pps_nal(self) -> bytes:
        return nal(3, 8, self.pps_rbsp())

    # ---- 프레임
    @property
    def frames(self) -> int:
        return len(self.frame_sizes)

    def encode(self, frame: np.ndarray) -> bytes:
        """프레임 하나((H, W) uint32 또는 (H, W, 4) uint8 BGRA) → NAL 단위 바이트. 첫 프레임은 IDR."""
        src = as_rgb32(frame, self.width, self.height)
        cur = self._cur
        cur[: self.height, : self.width] = src
        if self.pw != self.width:
            cur[: self.height, self.width:] = src[:, -1:]
        if self.ph != self.height:
            cur[self.height:, :] = cur[self.height - 1: self.height, :]
        first = not self.frame_sizes
        if first:
            changed = np.arange(self.n_mb)
        else:
            np.not_equal(cur, self._prev, out=self._diff)
            changed = np.flatnonzero(self._diff.reshape(self.mbh, 16, self.mbw, 16).any(axis=(1, 3)))
        if changed.size:
            rows, cols = np.divmod(changed, self.mbw)
            blocks = cur.reshape(self.mbh, 16, self.mbw, 16).transpose(0, 2, 1, 3)[rows, cols]
            self.mbs[changed] = yuv_blocks(blocks)
        data = self._slice(changed, idr=first)
        self._cur, self._prev = self._prev, self._cur
        self.frame_sizes.append(len(data))
        self.changed.append(int(changed.size))
        return data

    def _slice(self, changed: np.ndarray, *, idr: bool) -> bytes:
        b = BitWriter()
        b.ue(0)  # first_mb_in_slice
        b.ue(7 if idr else 5)  # slice_type: I(7) 또는 P(5), 그림의 모든 조각이 같은 형식
        b.ue(0)  # pic_parameter_set_id
        if idr:
            self._frame_num = 0
        b.u(self._frame_num, LOG2_MAX_FRAME_NUM)
        if idr:
            b.ue(0)  # idr_pic_id
        else:
            b.u(0, 1)  # num_ref_idx_active_override_flag
            b.u(0, 1)  # ref_pic_list_modification_flag_l0
        if idr:
            b.u(0, 1)  # no_output_of_prior_pics_flag
            b.u(0, 1)  # long_term_reference_flag
        else:
            b.u(0, 1)  # adaptive_ref_pic_marking_mode_flag(미끄럼 창)
        b.se(0)  # slice_qp_delta
        b.ue(1)  # disable_deblocking_filter_idc 1 = 끔
        mbs = self.mbs
        parts: list[bytes] = []
        if idr:
            b.ue(MB_TYPE_I_PCM_IN_I)
            b.align_zero()
            parts.append(bytes(b.out))
            body = np.empty((self.n_mb, 2 + MB_BYTES), dtype=np.uint8)
            body[:, 0], body[:, 1] = _IDR_MB_HEADER[0], _IDR_MB_HEADER[1]
            body[:, 2:] = mbs
            parts.append(body.tobytes()[2:])  # 첫 블록 머리는 위 비트 쓰기에 이미 있다
            parts.append(b"\x80")  # rbsp_trailing_bits(바이트 경계)
            ref_idc, kind = 3, 5
        else:
            idx = changed.tolist()
            last = -1
            if idx:
                b.ue(idx[0])  # mb_skip_run
                b.ue(MB_TYPE_I_PCM_IN_P)
                b.align_zero()
                parts.append(bytes(b.out))
                parts.append(mbs[idx[0]].tobytes())
                last = idx[0]
                heads = self._run_headers
                for i in idx[1:]:
                    run = i - last - 1
                    h = heads.get(run)
                    if h is None:
                        h = heads[run] = _aligned_header(run, MB_TYPE_I_PCM_IN_P)
                    parts.append(h)
                    parts.append(mbs[i].tobytes())
                    last = i
                tail = BitWriter()
            else:
                tail = b
            if last != self.n_mb - 1:
                tail.ue(self.n_mb - 1 - last)  # 끝에 남은 건너뛰기
            tail.trailing()
            parts.append(bytes(tail.out))
            ref_idc, kind = 2, 1
        self._frame_num = (self._frame_num + 1) % (1 << LOG2_MAX_FRAME_NUM)
        return nal(ref_idc, kind, b"".join(parts))

    def finish(self, extra_bytes_per_frame: int = 0) -> int:
        """레벨을 정한다(프레임마다 extra_bytes_per_frame을 더한 크기로: MP4의 길이 4바이트). 그 뒤 sps_nal()이 그 레벨."""
        sizes = np.asarray(self.frame_sizes, dtype=np.int64) + int(extra_bytes_per_frame)
        if sizes.size:
            win = min(sizes.size, self.fps)
            csum = np.concatenate([[0], np.cumsum(sizes)])
            max_bits = float((csum[win:] - csum[:-win]).max()) * 8.0 * self.fps / win
            max_frame = float(sizes.max()) * 8.0
        else:
            max_bits = max_frame = 0.0
        self.max_bits_1s = max_bits
        self.level, self.level_ok = pick_level(self.mbw, self.mbh, self.fps, max_bits, max_frame)
        return self.level

    # ---- 검사용
    def visible_yuv(self) -> bytes:
        """디코더가 지금 프레임으로 내야 할 그림(잘라 낸 크기, yuv420p 평면 Y·Cb·Cr 순서)."""
        mb = self.mbs
        y = mb[:, :256].reshape(self.mbh, self.mbw, 16, 16).transpose(0, 2, 1, 3).reshape(self.ph, self.pw)
        u = mb[:, 256:320].reshape(self.mbh, self.mbw, 8, 8).transpose(0, 2, 1, 3).reshape(self.ph // 2, self.pw // 2)
        v = mb[:, 320:].reshape(self.mbh, self.mbw, 8, 8).transpose(0, 2, 1, 3).reshape(self.ph // 2, self.pw // 2)
        w, h = self.width, self.height
        return y[:h, :w].tobytes() + u[: h // 2, : w // 2].tobytes() + v[: h // 2, : w // 2].tobytes()


def level_name(level_idc: int) -> str:
    """30 → "3", 31 → "3.1"(ffprobe가 보이는 숫자는 level_idc 그대로)."""
    major, minor = divmod(int(level_idc), 10)
    return f"{major}" if minor == 0 else f"{major}.{minor}"


__all__: Sequence[str] = [
    "CONSTRAINT_FLAGS", "LEVELS", "LOG2_MAX_FRAME_NUM", "MAX_LEVEL", "MB_BYTES", "PROFILE_IDC", "BitWriter", "PcmEncoder",
    "as_rgb32", "ebsp", "level_name", "nal", "pick_level", "rgb32_to_yuv420", "yuv_blocks",
]

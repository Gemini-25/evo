"""테스트 도우미: 내장 H.264 쓰기(src/evolab/app/export/h264_pcm.py + mp4.py)가 만드는 부분집합만 푸는 MP4 읽기·복호기.

쓰는 쪽 코드를 하나도 쓰지 않고 표준만 보고 따로 짰다(같은 실수를 두 번 하지 않게): MP4 상자(ftyp, mdat, moov/trak/…/stbl,
avcC, colr), 시작 코드 흉내 방지 풀기, SPS(VUI 포함)·PPS, 조각 머리, CAVLC 조각 자료 가운데
- I 조각: 모든 매크로블록이 I_PCM(mb_type 25)
- P 조각: mb_skip_run + I_PCM(mb_type 30), P_Skip은 움직임 벡터 예측(8.4.1.1·8.4.1.3)을 실제로 계산해 0인지 확인한 뒤
  참조 그림(바로 앞 그림)의 같은 자리를 그대로 가져옴
- 디블로킹 꺼짐(disable_deblocking_filter_idc 1), 조각 하나, frame_num 연속(gaps 없음), pic_order_cnt_type 2
만 푼다. 그 밖의 것(다른 mb_type, 디블로킹, 여러 조각, 0이 아닌 움직임 벡터 …)을 만나면 Unsupported를 던진다.

복호 결과는 프레임마다 (Y, Cb, Cr) uint8 평면(자르기 뒤 크기)이다. 명령줄:

    python tests/h264_subset_decoder.py FILE.mp4      # 상자·형식 요약, 모든 프레임 복호, 종료 코드 0(성공)·1(실패)

윈도우 CI(W7)가 exe로 만든 MP4를 이것으로 확인한다. 출력은 ASCII만(콘솔 코드 페이지와 무관하게).
"""

from __future__ import annotations

import struct
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class Unsupported(ValueError):
    """이 부분집합 복호기가 다루지 않는 비트열."""


class BadStream(ValueError):
    """표준에 맞지 않는 상자·비트열."""


# ---------------------------------------------------------------------------
# MP4
# ---------------------------------------------------------------------------
CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"dinf", b"edts"}


def iter_boxes(data: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """(종류, 내용 시작, 내용 끝). size 1(64비트 largesize)과 0(끝까지)을 다룬다."""
    pos = start
    while pos < end:
        if end - pos < 8:
            raise BadStream(f"truncated box header at {pos}")
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            if end - pos < 16:
                raise BadStream("truncated largesize")
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            raise BadStream(f"box {kind!r} at {pos} has bad size {size}")
        yield kind, pos + header, pos + size
        pos += size


@dataclass
class Mp4Video:
    brands: list[str]
    major_brand: str
    width: int
    height: int
    timescale: int
    media_duration: int
    movie_timescale: int
    movie_duration: int
    sample_deltas: list[int]
    samples: list[bytes]
    sync: list[int]
    chunk_offsets: list[int]
    chunk_box: str
    stsc: list[tuple[int, int, int]]
    avcc: dict[str, Any]
    colr: tuple[str, int, int, int, int] | None
    pasp: tuple[int, int] | None
    top_level: list[str]
    mdat_largesize: bool

    @property
    def fps(self) -> float:
        return self.timescale / self.sample_deltas[0] if self.sample_deltas else 0.0


def _full_box(data: bytes, start: int) -> tuple[int, int, int]:
    vf = struct.unpack_from(">I", data, start)[0]
    return vf >> 24, vf & 0xFFFFFF, start + 4


def read_mp4(source: str | Path | bytes) -> Mp4Video:
    """MP4 파일(또는 바이트)에서 첫 비디오 트랙의 표본·avcC·색 정보를 읽는다."""
    data = bytes(source) if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    top = list(iter_boxes(data, 0, len(data)))
    kinds = [k.decode("latin-1") for k, _s, _e in top]
    if not top or top[0][0] != b"ftyp":
        raise BadStream("first box is not ftyp")
    fs, fe = top[0][1], top[0][2]
    major = data[fs:fs + 4].decode("latin-1")
    brands = [data[i:i + 4].decode("latin-1") for i in range(fs + 8, fe, 4)]
    mdat = [(s, e) for k, s, e in top if k == b"mdat"]
    if len(mdat) != 1:
        raise BadStream(f"expected one mdat, found {len(mdat)}")
    mdat_start, mdat_end = mdat[0]
    largesize = mdat_start >= 16 and data[mdat_start - 12:mdat_start - 8] == b"mdat" \
        and struct.unpack_from(">I", data, mdat_start - 16)[0] == 1
    moov = [(s, e) for k, s, e in top if k == b"moov"]
    if len(moov) != 1:
        raise BadStream(f"expected one moov, found {len(moov)}")
    boxes: dict[bytes, tuple[int, int]] = {}
    movie_ts = movie_dur = 0
    traks = []
    for kind, s, e in iter_boxes(data, *moov[0]):
        if kind == b"mvhd":
            version, _f, p = _full_box(data, s)
            if version == 1:
                movie_ts, movie_dur = struct.unpack_from(">IQ", data, p + 16)
            else:
                movie_ts, movie_dur = struct.unpack_from(">II", data, p + 8)
        elif kind == b"trak":
            traks.append((s, e))
    if len(traks) != 1:
        raise Unsupported(f"{len(traks)} tracks")

    def walk(start: int, end: int) -> None:
        for kind, s, e in iter_boxes(data, start, end):
            if kind in CONTAINERS:
                walk(s, e)
            else:
                boxes.setdefault(kind, (s, e))

    walk(*traks[0])
    for need in (b"tkhd", b"mdhd", b"hdlr", b"stsd", b"stts", b"stsc", b"stsz"):
        if need not in boxes:
            raise BadStream(f"missing {need.decode()} box")
    s, _e = boxes[b"hdlr"]
    if data[s + 8:s + 12] != b"vide":
        raise BadStream("first track is not video")
    version, _f, p = _full_box(data, boxes[b"mdhd"][0])
    if version == 1:
        timescale, media_dur = struct.unpack_from(">IQ", data, p + 16)
    else:
        timescale, media_dur = struct.unpack_from(">II", data, p + 8)
    # stsd → avc1 → avcC, colr, pasp
    s, e = boxes[b"stsd"]
    _v, _f, p = _full_box(data, s)
    count = struct.unpack_from(">I", data, p)[0]
    if count != 1:
        raise Unsupported(f"{count} sample descriptions")
    entries = list(iter_boxes(data, p + 4, e))
    kind, es, ee = entries[0]
    if kind != b"avc1":
        raise Unsupported(f"sample entry {kind!r}")
    width, height = struct.unpack_from(">HH", data, es + 24)
    avcc: dict[str, Any] = {}
    colr = None
    pasp = None
    for sub, ss, se in iter_boxes(data, es + 78, ee):
        if sub == b"avcC":
            avcc = parse_avcc(data[ss:se])
        elif sub == b"colr":
            ctype = data[ss:ss + 4].decode("latin-1")
            if ctype == "nclx":
                prim, trans, mat = struct.unpack_from(">HHH", data, ss + 4)
                colr = (ctype, prim, trans, mat, data[ss + 10] >> 7)
        elif sub == b"pasp":
            pasp = struct.unpack_from(">II", data, ss)
    if not avcc:
        raise BadStream("avc1 without avcC")
    # 표본 표
    _v, _f, p = _full_box(data, boxes[b"stts"][0])
    n = struct.unpack_from(">I", data, p)[0]
    deltas: list[int] = []
    for i in range(n):
        c, d = struct.unpack_from(">II", data, p + 4 + 8 * i)
        deltas += [d] * c
    _v, _f, p = _full_box(data, boxes[b"stsz"][0])
    uniform, n_samples = struct.unpack_from(">II", data, p)
    sizes = [uniform] * n_samples if uniform else list(struct.unpack_from(f">{n_samples}I", data, p + 8))
    _v, _f, p = _full_box(data, boxes[b"stsc"][0])
    n = struct.unpack_from(">I", data, p)[0]
    stsc = [struct.unpack_from(">III", data, p + 4 + 12 * i) for i in range(n)]
    if b"stco" in boxes:
        chunk_box = "stco"
        _v, _f, p = _full_box(data, boxes[b"stco"][0])
        n = struct.unpack_from(">I", data, p)[0]
        offsets = list(struct.unpack_from(f">{n}I", data, p + 4))
    elif b"co64" in boxes:
        chunk_box = "co64"
        _v, _f, p = _full_box(data, boxes[b"co64"][0])
        n = struct.unpack_from(">I", data, p)[0]
        offsets = list(struct.unpack_from(f">{n}Q", data, p + 4))
    else:
        raise BadStream("missing stco/co64")
    sync: list[int] = []
    if b"stss" in boxes:
        _v, _f, p = _full_box(data, boxes[b"stss"][0])
        n = struct.unpack_from(">I", data, p)[0]
        sync = list(struct.unpack_from(f">{n}I", data, p + 4))
    if len(deltas) != n_samples:
        raise BadStream(f"stts covers {len(deltas)} samples, stsz has {n_samples}")
    # 청크 → 표본 위치
    samples: list[bytes] = []
    k = 0
    for ci, off in enumerate(offsets, start=1):
        per = next(spc for first, spc, _d in reversed(stsc) if first <= ci)
        pos = off
        for _ in range(per):
            if k >= n_samples:
                raise BadStream("stsc describes more samples than stsz")
            if pos < mdat_start or pos + sizes[k] > mdat_end:
                raise BadStream(f"sample {k + 1} lies outside mdat")
            samples.append(data[pos:pos + sizes[k]])
            pos += sizes[k]
            k += 1
    if k != n_samples:
        raise BadStream(f"chunks hold {k} samples, stsz has {n_samples}")
    return Mp4Video(brands=brands, major_brand=major, width=width, height=height, timescale=timescale,
                    media_duration=media_dur, movie_timescale=movie_ts, movie_duration=movie_dur,
                    sample_deltas=deltas, samples=samples, sync=sync, chunk_offsets=offsets, chunk_box=chunk_box,
                    stsc=[tuple(r) for r in stsc], avcc=avcc, colr=colr, pasp=pasp, top_level=kinds,
                    mdat_largesize=largesize)


def parse_avcc(b: bytes) -> dict[str, Any]:
    if len(b) < 7 or b[0] != 1:
        raise BadStream("bad avcC")
    out: dict[str, Any] = {"profile": b[1], "compat": b[2], "level": b[3], "length_size": (b[4] & 3) + 1}
    n_sps = b[5] & 31
    pos = 6
    sps: list[bytes] = []
    for _ in range(n_sps):
        ln = struct.unpack_from(">H", b, pos)[0]
        sps.append(b[pos + 2:pos + 2 + ln])
        pos += 2 + ln
    n_pps = b[pos]
    pos += 1
    pps: list[bytes] = []
    for _ in range(n_pps):
        ln = struct.unpack_from(">H", b, pos)[0]
        pps.append(b[pos + 2:pos + 2 + ln])
        pos += 2 + ln
    out["sps"], out["pps"] = sps, pps
    return out


def split_nals(sample: bytes, length_size: int = 4) -> list[bytes]:
    out: list[bytes] = []
    pos = 0
    while pos < len(sample):
        if pos + length_size > len(sample):
            raise BadStream("truncated NAL length")
        ln = int.from_bytes(sample[pos:pos + length_size], "big")
        pos += length_size
        if ln == 0 or pos + ln > len(sample):
            raise BadStream("bad NAL length")
        out.append(sample[pos:pos + ln])
        pos += ln
    return out


# ---------------------------------------------------------------------------
# 비트 읽기
# ---------------------------------------------------------------------------
def unescape(payload: bytes) -> tuple[bytes, int]:
    """emulation_prevention_three_byte를 뺀 RBSP와 뺀 개수. 흉내 방지가 빠진 00 00 0x(x ≤ 2)를 만나면 BadStream."""
    out = bytearray()
    zeros = 0
    removed = 0
    for byte in payload:
        if zeros >= 2 and byte == 3:
            zeros = 0
            removed += 1
            continue
        if zeros >= 2 and byte <= 2:
            raise BadStream("start code emulation inside a NAL unit")
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out), removed


class BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.total = 8 * len(data)
        self.pos = 0
        last = len(data) - 1
        while last >= 0 and data[last] == 0:
            last -= 1
        if last < 0:
            raise BadStream("RBSP without a stop bit")
        tz = (data[last] & -data[last]).bit_length() - 1
        self.stop = 8 * last + (7 - tz)
        """rbsp_stop_one_bit의 비트 위치."""

    def u(self, n: int) -> int:
        if n == 0:
            return 0
        pos, end = self.pos, self.pos + n
        if end > self.total:
            raise BadStream("read past the end of the RBSP")
        first, last = pos >> 3, (end - 1) >> 3
        chunk = int.from_bytes(self.data[first:last + 1], "big")
        self.pos = end
        return (chunk >> ((last + 1) * 8 - end)) & ((1 << n) - 1)

    def ue(self) -> int:
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
            if zeros > 31:
                raise BadStream("ue(v) too long")
        return (1 << zeros) - 1 + self.u(zeros)

    def se(self) -> int:
        k = self.ue()
        return (k + 1) // 2 if k % 2 else -(k // 2)

    @property
    def byte_aligned(self) -> bool:
        return self.pos % 8 == 0

    def more_rbsp_data(self) -> bool:
        return self.pos < self.stop

    def trailing(self) -> None:
        """rbsp_trailing_bits가 정확히 남았는지 본다."""
        if self.pos != self.stop:
            raise BadStream(f"{self.stop - self.pos} unread bits before the stop bit")


# ---------------------------------------------------------------------------
# 매개변수 집합
# ---------------------------------------------------------------------------
def parse_sps(rbsp: bytes) -> dict[str, Any]:
    r = BitReader(rbsp)
    s: dict[str, Any] = {"profile_idc": r.u(8), "constraint_flags": r.u(8), "level_idc": r.u(8), "id": r.ue()}
    if s["profile_idc"] in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135):
        raise Unsupported(f"profile {s['profile_idc']}")
    s["log2_max_frame_num"] = r.ue() + 4
    s["poc_type"] = r.ue()
    if s["poc_type"] == 0:
        s["log2_max_poc_lsb"] = r.ue() + 4
    elif s["poc_type"] == 1:
        raise Unsupported("pic_order_cnt_type 1")
    s["max_num_ref_frames"] = r.ue()
    s["gaps_allowed"] = r.u(1)
    s["width_mbs"] = r.ue() + 1
    s["height_map_units"] = r.ue() + 1
    s["frame_mbs_only"] = r.u(1)
    if not s["frame_mbs_only"]:
        raise Unsupported("interlaced")
    s["height_mbs"] = s["height_map_units"]
    s["direct_8x8_inference"] = r.u(1)
    s["crop"] = (0, 0, 0, 0)
    if r.u(1):
        s["crop"] = (r.ue(), r.ue(), r.ue(), r.ue())  # left, right, top, bottom(4:2:0: 2화소 단위)
    s["vui"] = parse_vui(r) if r.u(1) else None
    r.trailing()
    left, right, top, bottom = s["crop"]
    s["width"] = s["width_mbs"] * 16 - 2 * (left + right)
    s["height"] = s["height_mbs"] * 16 - 2 * (top + bottom)
    return s


def parse_vui(r: BitReader) -> dict[str, Any]:
    v: dict[str, Any] = {}
    if r.u(1):
        v["aspect_ratio_idc"] = r.u(8)
        if v["aspect_ratio_idc"] == 255:
            v["sar"] = (r.u(16), r.u(16))
    if r.u(1):
        v["overscan_appropriate"] = r.u(1)
    if r.u(1):
        v["video_format"] = r.u(3)
        v["full_range"] = r.u(1)
        if r.u(1):
            v["colour_primaries"], v["transfer"], v["matrix"] = r.u(8), r.u(8), r.u(8)
    if r.u(1):
        v["chroma_loc"] = (r.ue(), r.ue())
    if r.u(1):
        v["num_units_in_tick"], v["time_scale"], v["fixed_frame_rate"] = r.u(32), r.u(32), r.u(1)
    nal_hrd, vcl_hrd = r.u(1), 0
    if nal_hrd:
        raise Unsupported("HRD parameters")
    vcl_hrd = r.u(1)
    if vcl_hrd:
        raise Unsupported("HRD parameters")
    v["pic_struct_present"] = r.u(1)
    if r.u(1):
        v["mv_over_boundaries"] = r.u(1)
        v["max_bytes_per_pic_denom"] = r.ue()
        v["max_bits_per_mb_denom"] = r.ue()
        v["log2_max_mv_length"] = (r.ue(), r.ue())
        v["max_num_reorder_frames"] = r.ue()
        v["max_dec_frame_buffering"] = r.ue()
    return v


def parse_pps(rbsp: bytes) -> dict[str, Any]:
    r = BitReader(rbsp)
    p: dict[str, Any] = {"id": r.ue(), "sps_id": r.ue(), "entropy_coding_mode": r.u(1),
                         "bottom_field_pic_order": r.u(1), "num_slice_groups": r.ue() + 1}
    if p["entropy_coding_mode"]:
        raise Unsupported("CABAC")
    if p["num_slice_groups"] != 1:
        raise Unsupported("slice groups (FMO)")
    p["num_ref_idx_l0_default"] = r.ue() + 1
    p["num_ref_idx_l1_default"] = r.ue() + 1
    p["weighted_pred"] = r.u(1)
    p["weighted_bipred_idc"] = r.u(2)
    p["pic_init_qp"] = 26 + r.se()
    p["pic_init_qs"] = 26 + r.se()
    p["chroma_qp_index_offset"] = r.se()
    p["deblocking_filter_control_present"] = r.u(1)
    p["constrained_intra_pred"] = r.u(1)
    p["redundant_pic_cnt_present"] = r.u(1)
    if r.more_rbsp_data():
        raise Unsupported("PPS extension (High profile)")
    r.trailing()
    return p


# ---------------------------------------------------------------------------
# 복호
# ---------------------------------------------------------------------------
@dataclass
class FrameInfo:
    idr: bool
    frame_num: int
    pcm_blocks: int
    skipped: int
    removed_emulation_bytes: int


@dataclass
class Decoder:
    sps: dict[str, Any]
    pps: dict[str, Any]
    frames: list[FrameInfo] = field(default_factory=list)
    _ref: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    _prev_frame_num: int = -1

    def __post_init__(self) -> None:
        if self.pps["sps_id"] != self.sps["id"]:
            raise BadStream("PPS refers to another SPS")
        self.mbw, self.mbh = self.sps["width_mbs"], self.sps["height_mbs"]
        self.n_mb = self.mbw * self.mbh

    def decode_nal(self, nal: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """NAL 단위 하나. 그림 조각이면 그 그림(자르기 뒤 Y, Cb, Cr)을, 아니면 None."""
        if not nal:
            raise BadStream("empty NAL")
        head = nal[0]
        if head & 0x80:
            raise BadStream("forbidden_zero_bit set")
        ref_idc, kind = (head >> 5) & 3, head & 31
        if kind in (7, 8):
            return None  # 비트열 안 SPS·PPS(이 부분집합 쓰기는 넣지 않지만 받아는 줌)
        if kind not in (1, 5):
            raise Unsupported(f"NAL type {kind}")
        rbsp, removed = unescape(nal[1:])
        return self._slice(BitReader(rbsp), idr=kind == 5, ref_idc=ref_idc, removed=removed)

    def _slice(self, r: BitReader, *, idr: bool, ref_idc: int, removed: int) -> tuple[np.ndarray, ...]:
        sps, pps = self.sps, self.pps
        first_mb = r.ue()
        slice_type = r.ue() % 5
        if first_mb != 0:
            raise Unsupported("more than one slice per picture")
        if r.ue() != pps["id"]:
            raise BadStream("unknown PPS id")
        frame_num = r.u(sps["log2_max_frame_num"])
        max_frame_num = 1 << sps["log2_max_frame_num"]
        if idr:
            if slice_type != 2:
                raise BadStream("IDR picture with a non-I slice")
            if frame_num != 0:
                raise BadStream("IDR frame_num must be 0")
            r.ue()  # idr_pic_id
        else:
            if self._ref is None:
                raise BadStream("P picture before the first IDR")
            if not sps["gaps_allowed"] and frame_num != (self._prev_frame_num + 1) % max_frame_num:
                raise BadStream(f"frame_num gap: {self._prev_frame_num} -> {frame_num}")
        if sps["poc_type"] == 0:
            r.u(sps["log2_max_poc_lsb"])
            if pps["bottom_field_pic_order"]:
                r.se()
        if pps["redundant_pic_cnt_present"]:
            r.ue()
        if slice_type == 0:  # P
            num_ref = pps["num_ref_idx_l0_default"]
            if r.u(1):
                num_ref = r.ue() + 1
            if r.u(1):
                raise Unsupported("ref_pic_list_modification")
            if num_ref != 1:
                raise Unsupported("more than one active reference")
            if pps["weighted_pred"]:
                raise Unsupported("weighted prediction")
        elif slice_type != 2:
            raise Unsupported(f"slice type {slice_type}")
        if ref_idc:
            if idr:
                r.u(1)  # no_output_of_prior_pics_flag
                if r.u(1):
                    raise Unsupported("long-term reference")
            elif r.u(1):
                raise Unsupported("adaptive reference picture marking")
        qp = pps["pic_init_qp"] + r.se()
        if not 0 <= qp <= 51:
            raise BadStream(f"QP {qp} out of range")
        if not pps["deblocking_filter_control_present"]:
            raise Unsupported("deblocking filter is on (no control in the PPS)")
        disable = r.ue()
        if disable != 1:
            raise Unsupported(f"deblocking filter idc {disable}")
        # 조각 자료
        pw, ph = self.mbw * 16, self.mbh * 16
        if idr or self._ref is None:
            y = np.zeros((ph, pw), np.uint8)
            cb = np.zeros((ph // 2, pw // 2), np.uint8)
            cr = np.zeros((ph // 2, pw // 2), np.uint8)
        else:
            y, cb, cr = (a.copy() for a in self._ref)
        intra = np.zeros(self.n_mb, dtype=bool)
        decoded = np.zeros(self.n_mb, dtype=bool)
        pcm_addr: list[int] = []
        pcm_pos: list[int] = []
        skipped = 0
        addr = 0
        more = True
        while more:
            if slice_type == 0:
                run = r.ue()
                for _ in range(run):
                    if addr >= self.n_mb:
                        raise BadStream("mb_skip_run past the end of the picture")
                    self._check_skip_mv(addr, intra, decoded)
                    decoded[addr] = True
                    addr += 1
                    skipped += 1
                if run:
                    more = r.more_rbsp_data()
            if more:
                if addr >= self.n_mb:
                    raise BadStream("macroblock past the end of the picture")
                mb_type = r.ue()
                want = 25 if slice_type == 2 else 30
                if mb_type != want:
                    raise Unsupported(f"mb_type {mb_type} (only I_PCM)")
                while not r.byte_aligned:
                    if r.u(1):
                        raise BadStream("pcm_alignment_zero_bit is 1")
                pcm_addr.append(addr)
                pcm_pos.append(r.pos // 8)
                r.pos += 384 * 8
                if r.pos > r.total:
                    raise BadStream("truncated I_PCM macroblock")
                intra[addr] = True
                decoded[addr] = True
                addr += 1
                more = r.more_rbsp_data()
        r.trailing()
        if addr != self.n_mb:
            raise BadStream(f"picture covers {addr} of {self.n_mb} macroblocks")
        if pcm_addr:
            idx = np.asarray(pcm_pos, dtype=np.int64)[:, None] + np.arange(384)
            blocks = np.frombuffer(r.data, dtype=np.uint8)[idx]
            rows, cols = np.divmod(np.asarray(pcm_addr), self.mbw)
            n = len(pcm_addr)
            y.reshape(self.mbh, 16, self.mbw, 16).transpose(0, 2, 1, 3)[rows, cols] = blocks[:, :256].reshape(n, 16, 16)
            cb.reshape(self.mbh, 8, self.mbw, 8).transpose(0, 2, 1, 3)[rows, cols] = blocks[:, 256:320].reshape(n, 8, 8)
            cr.reshape(self.mbh, 8, self.mbw, 8).transpose(0, 2, 1, 3)[rows, cols] = blocks[:, 320:].reshape(n, 8, 8)
        self._ref = (y, cb, cr)
        self._prev_frame_num = frame_num if ref_idc else self._prev_frame_num
        self.frames.append(FrameInfo(idr, frame_num, len(pcm_addr), skipped, removed))
        left, right, top, bottom = sps["crop"]
        y0, y1, x0, x1 = 2 * top, ph - 2 * bottom, 2 * left, pw - 2 * right
        return (y[y0:y1, x0:x1].copy(), cb[y0 // 2:y1 // 2, x0 // 2:x1 // 2].copy(),
                cr[y0 // 2:y1 // 2, x0 // 2:x1 // 2].copy())

    def _check_skip_mv(self, addr: int, intra: np.ndarray, decoded: np.ndarray) -> None:
        """P_Skip의 움직임 벡터(8.4.1.1): 이 부분집합에서는 모든 인터 블록이 P_Skip이고 그 벡터가 0이어야 한다.
        이웃 A(왼쪽)·B(위)·C(오른쪽 위, 없으면 D 왼쪽 위)의 refIdx·mv로 표준 규칙을 계산해 0이 아니면 Unsupported."""
        row, col = divmod(addr, self.mbw)

        def neighbour(r_: int, c_: int) -> tuple[int, tuple[int, int]] | None:
            if r_ < 0 or c_ < 0 or c_ >= self.mbw:
                return None
            a = r_ * self.mbw + c_
            if a >= addr or not decoded[a]:
                return None
            return (-1, (0, 0)) if intra[a] else (0, (0, 0))  # 인트라: refIdx −1·mv 0, 인터(P_Skip): refIdx 0·mv 0

        na, nb = neighbour(row, col - 1), neighbour(row - 1, col)
        if na is None or nb is None:
            return  # mv = 0
        if (na[0] == 0 and na[1] == (0, 0)) or (nb[0] == 0 and nb[1] == (0, 0)):
            return  # mv = 0
        nc = neighbour(row - 1, col + 1) or neighbour(row - 1, col - 1)
        cand = [na, nb, nc if nc is not None else (-1, (0, 0))]
        same = [n for n in cand if n[0] == 0]
        if len(same) == 1:
            mv = same[0][1]
        else:
            mv = (int(np.median([n[1][0] for n in cand])), int(np.median([n[1][1] for n in cand])))
        if mv != (0, 0):
            raise Unsupported("P_Skip with a non-zero motion vector")


def open_decoder(video: Mp4Video) -> Decoder:
    """avcC의 SPS·PPS로 복호기를 만든다(avcC와 SPS의 프로파일·레벨이 같은지도 본다)."""
    av = video.avcc
    if len(av["sps"]) != 1 or len(av["pps"]) != 1:
        raise Unsupported("expected one SPS and one PPS in avcC")
    if av["sps"][0][0] & 31 != 7 or av["pps"][0][0] & 31 != 8:
        raise BadStream("avcC holds non-SPS/PPS NAL units")
    sps = parse_sps(unescape(av["sps"][0][1:])[0])
    pps = parse_pps(unescape(av["pps"][0][1:])[0])
    if (av["profile"], av["compat"], av["level"]) != (sps["profile_idc"], sps["constraint_flags"], sps["level_idc"]):
        raise BadStream("avcC profile/level differs from the SPS")
    return Decoder(sps, pps)


def iter_pictures(video: Mp4Video, dec: Decoder) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """표본마다 그림 하나(자르기 뒤 Y, Cb, Cr). 프레임 정보는 dec.frames에 쌓인다."""
    for k, sample in enumerate(video.samples):
        pictures = [p for p in (dec.decode_nal(n) for n in split_nals(sample, video.avcc["length_size"]))
                    if p is not None]
        if len(pictures) != 1:
            raise BadStream(f"sample {k + 1} holds {len(pictures)} pictures")
        if (k + 1 in video.sync) != dec.frames[-1].idr:
            raise BadStream(f"sample {k + 1}: stss and IDR disagree")
        yield pictures[0]


def decode_file(path: str | Path | bytes) -> tuple[Mp4Video, dict[str, Any],
                                                   list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[FrameInfo]]:
    """(MP4 정보, SPS, 프레임마다 (Y, Cb, Cr), 프레임마다 FrameInfo)."""
    video = read_mp4(path)
    dec = open_decoder(video)
    frames = list(iter_pictures(video, dec))
    return video, dec.sps, frames, dec.frames


def i420(planes: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bytes:
    return b"".join(p.tobytes() for p in planes)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    try:
        sys.stdout.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
    if len(args) != 1:
        print("usage: python tests/h264_subset_decoder.py FILE.mp4")
        return 2
    path = Path(args[0])
    try:
        video = read_mp4(path)
        dec = open_decoder(video)
        sps = dec.sps
        count = 0
        for y, _cb, _cr in iter_pictures(video, dec):  # 프레임을 쌓지 않는다(720p 프레임 하나 1.4MB)
            if (y.shape[1], y.shape[0]) != (video.width, video.height):
                raise BadStream("decoded size differs from the sample entry")
            count += 1
        if count == 0:
            raise BadStream("no frames")
        vui = sps.get("vui") or {}
        print(f"file: {path.stat().st_size} bytes, top-level boxes {' '.join(video.top_level)}, "
              f"mdat largesize {video.mdat_largesize}")
        print(f"ftyp: {video.major_brand} [{', '.join(video.brands)}]")
        print(f"video: {video.width}x{video.height} (coded {sps['width_mbs'] * 16}x{sps['height_mbs'] * 16}, "
              f"crop {sps['crop']}), {video.fps:g} fps, {len(video.samples)} samples, sync {video.sync[:5]}, "
              f"{len(video.chunk_offsets)} chunks ({video.chunk_box})")
        print(f"profile: {sps['profile_idc']} constraints 0x{sps['constraint_flags']:02x} level {sps['level_idc']}")
        print(f"colour: primaries {vui.get('colour_primaries')} transfer {vui.get('transfer')} "
              f"matrix {vui.get('matrix')} full_range {vui.get('full_range')}; colr {video.colr}")
        pcm = sum(f.pcm_blocks for f in dec.frames)
        skip = sum(f.skipped for f in dec.frames)
        print(f"decoded: {count} frames, {pcm} I_PCM macroblocks, {skip} skipped")
    except (BadStream, Unsupported, OSError, struct.error, IndexError, KeyError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

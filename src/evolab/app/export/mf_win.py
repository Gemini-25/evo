"""Media Foundation H.264 쓰기 (윈도우만, v3 설계 D53, 4.17). ctypes로 COM을 부른다(새 의존성 없음).

호출 순서(share 4.3, 최종 설계 4.17): CoInitializeEx(MULTITHREADED) → MFStartup(MF_VERSION, FULL) → 속성
(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS = 1, MF_SINK_WRITER_DISABLE_THROTTLING = 1, MF_TRANSCODE_CONTAINERTYPE = MPEG4:
임시 파일 이름(mp4.temp_path, `.mp4`로 끝남)과 함께 컨테이너를 두 번 정함) → MFCreateSinkWriterFromURL(UTF-16 경로) → 출력 형식(H.264 Main, 평균
비트레이트, 진행 방식, 크기·비율·화소 비율) → AddStream → 입력 형식 RGB32(위가 먼저인 줄 간격 +4w, 싱크 라이터가 색
변환을 넣음), 거절되면 NV12(numpy, h264_pcm과 같은 BT.709 제한 범위 식) → SetInputMediaType(인코더 속성: GOP = 2초,
거절되면 속성 없이) → BeginWriting → 프레임마다 MFCreateMemoryBuffer·Lock·복사·Unlock·SetCurrentLength·MFCreateSample·
AddBuffer·SetSampleTime·SetSampleDuration(100ns 단위)·WriteSample → Finalize → Release·MFShutdown·CoUninitialize.
품질 매개변수(VBR 품질)는 쓰지 않는다: 평균 비트레이트만.

- vtable 번호(VTABLE)와 GUID·상수(GUIDS, CONSTANTS)는 SDK 헤더(mfobjects.h, mfidl.h, mfreadwrite.h, mfapi.h, codecapi.h)
  선언 순서·값에서 옮긴 표 하나에 둔다. 윈도우 CI(W6)가 SDK 헤더로 컴파일한 packaging/ci/mf_abi.c의 출력과 비교한다
  (tests/test_mf_abi.py). 함수 원형도 표(_FLAT, _METHODS) 하나에 두고, 테스트가 그 표로 만든 가짜 COM 객체로 호출
  순서·인자 마샬링·Release 짝을 리눅스에서도 검사한다.
- 이 모듈은 어디서나 불러올 수 있다(리눅스 테스트): 윈도우 DLL은 load_api()를 부를 때만 연다.
- 내보내기 자식 프로세스 안에서만 쓴다(설계 D49): ctypes COM 실수가 화면·학습을 죽이지 않게.
- 실패한 호출은 MfError(호출 이름 + HRESULT 16진수). mp4.write_mp4가 받아 내장 쓰기로 처음부터 다시 만든다.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
import tempfile
import uuid
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_int64,
    c_long,
    c_ubyte,
    c_uint32,
    c_uint64,
    c_ulong,
    c_ushort,
    c_void_p,
    c_wchar_p,
)
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import numpy as np

from evolab.app import strings as S
from evolab.app.export.h264_pcm import as_rgb32, frame_pixels, rgb32_to_yuv420


class MfError(RuntimeError):
    """Media Foundation 호출 실패. str = 「호출 이름 0xHRESULT」(EncoderStatus.detail, EXP_MF_FALLBACK의 {reason})."""

    def __init__(self, what: str, hresult: int = 0) -> None:
        self.what = what
        self.hresult = int(hresult) & 0xFFFFFFFF
        super().__init__(f"{what} 0x{self.hresult:08X}" if hresult else what)


class GUID(Structure):
    _fields_ = [("Data1", c_uint32), ("Data2", c_ushort), ("Data3", c_ushort), ("Data4", c_ubyte * 8)]

    @classmethod
    def of(cls, name_or_text: str) -> GUID:
        text = GUIDS.get(name_or_text, name_or_text)
        return cls.from_buffer_copy(uuid.UUID(text).bytes_le)

    def text(self) -> str:
        return str(uuid.UUID(bytes_le=bytes(self)))


class MFT_REGISTER_TYPE_INFO(Structure):
    _fields_ = [("guidMajorType", GUID), ("guidSubtype", GUID)]


# ---------------------------------------------------------------------------
# SDK 헤더에서 옮긴 표 (W6가 packaging/ci/mf_abi.c 출력과 비교)
# ---------------------------------------------------------------------------
VTABLE: Final[dict[str, dict[str, int]]] = {
    # IUnknown(unknwn.h)
    "IUnknown": {"QueryInterface": 0, "AddRef": 1, "Release": 2},
    # IMFAttributes(mfobjects.h): GetItem 3 … GetAllocatedString 13 … SetItem 18, DeleteItem 19, DeleteAllItems 20,
    # SetUINT32 21, SetUINT64 22, SetDouble 23, SetGUID 24 … CopyAllItems 32
    "IMFAttributes": {"GetAllocatedString": 13, "SetUINT32": 21, "SetUINT64": 22, "SetGUID": 24},
    # IMFMediaType : IMFAttributes(mfobjects.h)
    "IMFMediaType": {"SetUINT32": 21, "SetUINT64": 22, "SetGUID": 24},
    # IMFActivate : IMFAttributes(mfobjects.h)
    "IMFActivate": {"GetAllocatedString": 13},
    # IMFMediaBuffer : IUnknown(mfobjects.h): Lock 3, Unlock 4, GetCurrentLength 5, SetCurrentLength 6, GetMaxLength 7
    "IMFMediaBuffer": {"Lock": 3, "Unlock": 4, "SetCurrentLength": 6},
    # IMFSample : IMFAttributes(mfobjects.h): GetSampleFlags 33, SetSampleFlags 34, GetSampleTime 35, SetSampleTime 36,
    # GetSampleDuration 37, SetSampleDuration 38, GetBufferCount 39, GetBufferByIndex 40, ConvertToContiguousBuffer 41,
    # AddBuffer 42 …
    "IMFSample": {"SetSampleTime": 36, "SetSampleDuration": 38, "AddBuffer": 42},
    # IMFSinkWriter : IUnknown(mfreadwrite.h): AddStream 3, SetInputMediaType 4, BeginWriting 5, WriteSample 6,
    # SendStreamTick 7, PlaceMarker 8, NotifyEndOfSegment 9, Flush 10, Finalize 11, GetServiceForStream 12,
    # GetStatistics 13
    "IMFSinkWriter": {"AddStream": 3, "SetInputMediaType": 4, "BeginWriting": 5, "WriteSample": 6, "Finalize": 11},
}

GUIDS: Final[dict[str, str]] = {
    "MF_MT_MAJOR_TYPE": "48eba18e-f8c9-4687-bf11-0a74c9f96a8f",
    "MF_MT_SUBTYPE": "f7e34c9a-42e8-4714-b74b-cb29d72c35e5",
    "MF_MT_AVG_BITRATE": "20332624-fb0d-4d9e-bd0d-cbf6786c102e",
    "MF_MT_INTERLACE_MODE": "e2724bb8-e676-4806-b4b2-a8d6efb44ccd",
    "MF_MT_FRAME_SIZE": "1652c33d-d6b2-4012-b834-72030849a37d",
    "MF_MT_FRAME_RATE": "c459a2e8-3d2c-4e44-b132-fee5156c7bb0",
    "MF_MT_PIXEL_ASPECT_RATIO": "c6376a1e-8d0a-4027-be45-6d9a0ad39bb6",
    "MF_MT_MPEG2_PROFILE": "ad76a80b-2d5c-4e0b-b375-64e520137036",
    "MF_MT_DEFAULT_STRIDE": "644b4e48-1e02-4516-b0eb-c01ca9d49ac6",
    "MFMediaType_Video": "73646976-0000-0010-8000-00aa00389b71",
    "MFVideoFormat_H264": "34363248-0000-0010-8000-00aa00389b71",
    "MFVideoFormat_RGB32": "00000016-0000-0010-8000-00aa00389b71",
    "MFVideoFormat_NV12": "3231564e-0000-0010-8000-00aa00389b71",
    "MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS": "a634a91c-822b-41b9-a494-4de4643612b0",
    "MF_SINK_WRITER_DISABLE_THROTTLING": "08b845d8-2b74-4afe-9d53-be16d2d5ae4f",
    "MF_TRANSCODE_CONTAINERTYPE": "150ff23f-4abc-478b-ac4f-e1916fba1cca",
    "MFTranscodeContainerType_MPEG4": "dc6cd05d-b9d0-40ef-bd35-fa622c1ab28a",
    "CODECAPI_AVEncMPVGOPSize": "95f31b26-95a4-41aa-9303-246a7fc6eef1",
    "MFT_CATEGORY_VIDEO_ENCODER": "f79eac7d-e545-4387-bdee-d647d7bde42a",
    "MFT_FRIENDLY_NAME_Attribute": "314ffbae-5b41-4c95-9c19-4e7d586face3",
}

CONSTANTS: Final[dict[str, int]] = {
    "MF_VERSION": 0x00020070,  # (MF_SDK_VERSION 0x0002 << 16) | MF_API_VERSION 0x0070
    "MFSTARTUP_FULL": 0,
    "COINIT_MULTITHREADED": 0,
    "MFVideoInterlace_Progressive": 2,
    "eAVEncH264VProfile_Main": 77,
    "MFT_ENUM_FLAG_SYNCMFT": 0x1,
    "MFT_ENUM_FLAG_ASYNCMFT": 0x2,
    "MFT_ENUM_FLAG_HARDWARE": 0x4,
    "MFT_ENUM_FLAG_SORTANDFILTER": 0x40,
    "S_FALSE": 1,
    "RPC_E_CHANGED_MODE": 0x80010106,
    "SIZEOF_GUID": 16,
    "SIZEOF_MFT_REGISTER_TYPE_INFO": 32,
}

PROBE_SIZE: Final = (64, 64)
PROBE_BITRATE: Final = 250_000
GOP_SECONDS: Final = 2
_HNS: Final = 10_000_000
"""100ns 단위(초당)."""

# 함수 원형: 이름 → (DLL, 돌려주는 형식, 인자 형식). 가짜 API(테스트)도 이 표로 만든다.
_FLAT: Final[dict[str, tuple[str, Any, tuple[Any, ...]]]] = {
    "CoInitializeEx": ("ole32", c_long, (c_void_p, c_uint32)),
    "CoUninitialize": ("ole32", None, ()),
    "CoTaskMemFree": ("ole32", None, (c_void_p,)),
    "MFStartup": ("mfplat", c_long, (c_ulong, c_uint32)),
    "MFShutdown": ("mfplat", c_long, ()),
    "MFCreateAttributes": ("mfplat", c_long, (POINTER(c_void_p), c_uint32)),
    "MFCreateMediaType": ("mfplat", c_long, (POINTER(c_void_p),)),
    "MFCreateMemoryBuffer": ("mfplat", c_long, (c_uint32, POINTER(c_void_p))),
    "MFCreateSample": ("mfplat", c_long, (POINTER(c_void_p),)),
    "MFTEnumEx": ("mfplat", c_long, (GUID, c_uint32, POINTER(MFT_REGISTER_TYPE_INFO), POINTER(MFT_REGISTER_TYPE_INFO),
                                     POINTER(POINTER(c_void_p)), POINTER(c_uint32))),
    "MFCreateSinkWriterFromURL": ("mfreadwrite", c_long, (c_wchar_p, c_void_p, c_void_p, POINTER(c_void_p))),
}

# COM 메서드 원형(this 다음 인자): 이름 → (돌려주는 형식, 인자 형식). VTABLE의 메서드를 모두 덮는다(테스트).
_METHODS: Final[dict[str, tuple[Any, tuple[Any, ...]]]] = {
    "QueryInterface": (c_long, (POINTER(GUID), POINTER(c_void_p))),
    "AddRef": (c_ulong, ()),
    "Release": (c_ulong, ()),
    "GetAllocatedString": (c_long, (POINTER(GUID), POINTER(c_void_p), POINTER(c_uint32))),
    "SetUINT32": (c_long, (POINTER(GUID), c_uint32)),
    "SetUINT64": (c_long, (POINTER(GUID), c_uint64)),
    "SetGUID": (c_long, (POINTER(GUID), POINTER(GUID))),
    "Lock": (c_long, (POINTER(c_void_p), POINTER(c_uint32), POINTER(c_uint32))),
    "Unlock": (c_long, ()),
    "SetCurrentLength": (c_long, (c_uint32,)),
    "SetSampleTime": (c_long, (c_int64,)),
    "SetSampleDuration": (c_long, (c_int64,)),
    "AddBuffer": (c_long, (c_void_p,)),
    "AddStream": (c_long, (c_void_p, POINTER(c_uint32))),
    "SetInputMediaType": (c_long, (c_uint32, c_void_p, c_void_p)),
    "BeginWriting": (c_long, ()),
    "WriteSample": (c_long, (c_uint32, c_void_p)),
    "Finalize": (c_long, ()),
}

FUNCTYPE: Final = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
"""COM 메서드 호출 규약(윈도우 x64에서는 WINFUNCTYPE = CFUNCTYPE). 가짜 COM 객체(테스트)도 이것으로 만든다."""


def load_api() -> SimpleNamespace:
    """윈도우 DLL(ole32, mfplat, mfreadwrite)을 열고 _FLAT의 함수를 원형과 함께 돌려준다. 없으면 MfError."""
    if sys.platform != "win32":
        raise MfError(S.EXP_MF_NOT_WINDOWS)
    libs: dict[str, Any] = {}
    for name in sorted({lib for lib, _r, _a in _FLAT.values()}):
        try:
            libs[name] = ctypes.WinDLL(name)  # type: ignore[attr-defined]
        except OSError as exc:
            raise MfError(f"{name}.dll", getattr(exc, "winerror", 0) or 0) from exc
    funcs: dict[str, Any] = {}
    for fname, (lib, restype, argtypes) in _FLAT.items():
        try:
            fn = getattr(libs[lib], fname)
        except AttributeError as exc:
            raise MfError(f"{lib}.dll!{fname}") from exc
        fn.restype = restype
        fn.argtypes = argtypes
        funcs[fname] = fn
    return SimpleNamespace(**funcs)


class _Com:
    """vtable 번호로 COM 메서드를 부른다(원형은 메서드마다 한 번 만들어 둠)."""

    def __init__(self) -> None:
        self._protos: dict[str, Any] = {}

    def call(self, obj: int | None, iface: str, method: str, *args: Any) -> int:
        if not obj:
            raise MfError(f"{iface}::{method} on NULL")
        proto = self._protos.get(method)
        if proto is None:
            restype, argtypes = _METHODS[method]
            proto = self._protos[method] = FUNCTYPE(restype, c_void_p, *argtypes)
        vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
        fn = proto(vtbl[VTABLE[iface][method]])
        return int(fn(obj, *args) or 0)

    def check(self, obj: int | None, iface: str, method: str, *args: Any) -> None:
        hr = self.call(obj, iface, method, *args)
        if hr < 0:
            raise MfError(f"{iface}::{method}", hr)

    def release(self, obj: int | None) -> None:
        if obj:
            self.call(obj, "IUnknown", "Release")


def _check(hr: int | None, what: str) -> None:
    if hr is not None and int(hr) < 0:
        raise MfError(what, int(hr))


class _Session:
    """CoInitializeEx + MFStartup 한 쌍(닫을 때 되돌림)."""

    def __init__(self, api: SimpleNamespace) -> None:
        self.api = api
        self.com = False
        self.mf = False
        hr = int(api.CoInitializeEx(None, CONSTANTS["COINIT_MULTITHREADED"]) or 0)
        if hr >= 0:  # S_OK, S_FALSE(이미 같은 방식): 짝으로 CoUninitialize
            self.com = True
        elif hr & 0xFFFFFFFF != CONSTANTS["RPC_E_CHANGED_MODE"]:  # 다른 방식으로 이미 초기화돼 있으면 그대로 쓴다
            raise MfError("CoInitializeEx", hr)
        try:
            _check(api.MFStartup(CONSTANTS["MF_VERSION"], CONSTANTS["MFSTARTUP_FULL"]), "MFStartup")
        except BaseException:
            self.close()
            raise
        self.mf = True

    def close(self) -> None:
        if self.mf:  # 정리 중 실패는 무시한다(이미 다른 오류를 알리는 중일 수 있음)
            self.mf = False
            with contextlib.suppress(Exception):
                self.api.MFShutdown()
        if self.com:
            self.com = False
            with contextlib.suppress(Exception):
                self.api.CoUninitialize()


class MfWriter:
    """Media Foundation 싱크 라이터로 H.264 MP4를 쓴다(mp4.Mp4Writer). 만든 스레드에서만 쓴다."""

    kind: str = "mf"

    def __init__(self, path: Path | str, width: int, height: int, fps: int, *, bitrate: int,
                 tmp: Path | None = None, api: SimpleNamespace | None = None) -> None:
        for name, v in (("width", width), ("height", height)):
            if int(v) < 2 or int(v) % 2:
                raise ValueError(f"{name} must be a positive even integer, got {v!r}")
        self.path = Path(path)
        self.tmp = Path(tmp) if tmp is not None else self.path.with_name(f"{self.path.stem}.tmp{self.path.suffix}")
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.bitrate = int(bitrate)
        self.input_subtype = ""
        """실제로 받아들여진 입력 형식("MFVideoFormat_RGB32" 또는 "MFVideoFormat_NV12")."""
        self.frames = 0
        self._api = api if api is not None else load_api()
        self._com = _Com()
        self._session: _Session | None = None
        self._writer: int | None = None
        self._stream = 0
        self._state = "open"
        try:
            self._start()
        except BaseException:
            self._teardown()
            self._unlink_tmp()
            self._state = "aborted"
            raise

    # ---- 시작
    def _new(self, factory: str, *args: Any) -> int:
        out = c_void_p()
        _check(getattr(self._api, factory)(*args, byref(out)), factory)
        if not out.value:
            raise MfError(f"{factory} NULL")
        return int(out.value)

    def _attrs(self, values: dict[str, tuple[str, Any]], size: int) -> int:
        obj = self._create_attributes(size)
        try:
            self._set_all(obj, "IMFAttributes", values)
        except BaseException:
            self._com.release(obj)
            raise
        return obj

    def _create_attributes(self, size: int) -> int:
        out = c_void_p()
        _check(self._api.MFCreateAttributes(byref(out), size), "MFCreateAttributes")
        if not out.value:
            raise MfError("MFCreateAttributes NULL")
        return int(out.value)

    def _media_type(self, values: dict[str, tuple[str, Any]]) -> int:
        obj = self._new("MFCreateMediaType")
        try:
            self._set_all(obj, "IMFMediaType", values)
        except BaseException:
            self._com.release(obj)
            raise
        return obj

    def _set_all(self, obj: int, iface: str, values: dict[str, tuple[str, Any]]) -> None:
        for key, (kind, value) in values.items():
            k = GUID.of(key)
            if kind == "u32":
                self._com.check(obj, iface, "SetUINT32", byref(k), int(value) & 0xFFFFFFFF)
            elif kind == "u64":
                self._com.check(obj, iface, "SetUINT64", byref(k), int(value))
            else:
                g = GUID.of(value)
                self._com.check(obj, iface, "SetGUID", byref(k), byref(g))

    def _start(self) -> None:
        api, com = self._api, self._com
        self._session = _Session(api)
        w, h, fps = self.width, self.height, self.fps
        size64, rate64, par64 = (w << 32) | h, (fps << 32) | 1, (1 << 32) | 1
        progressive = CONSTANTS["MFVideoInterlace_Progressive"]
        attrs = self._attrs({"MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS": ("u32", 1),
                             "MF_SINK_WRITER_DISABLE_THROTTLING": ("u32", 1),
                             "MF_TRANSCODE_CONTAINERTYPE": ("guid", "MFTranscodeContainerType_MPEG4")}, 3)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._unlink_tmp()
            out = c_void_p()
            hr = api.MFCreateSinkWriterFromURL(os.path.abspath(self.tmp), None, attrs, byref(out))
            _check(hr, "MFCreateSinkWriterFromURL")
            if not out.value:
                raise MfError("MFCreateSinkWriterFromURL NULL")
            self._writer = int(out.value)
        finally:
            com.release(attrs)
        target = self._media_type({"MF_MT_MAJOR_TYPE": ("guid", "MFMediaType_Video"),
                                   "MF_MT_SUBTYPE": ("guid", "MFVideoFormat_H264"),
                                   "MF_MT_AVG_BITRATE": ("u32", self.bitrate),
                                   "MF_MT_INTERLACE_MODE": ("u32", progressive),
                                   "MF_MT_FRAME_SIZE": ("u64", size64),
                                   "MF_MT_FRAME_RATE": ("u64", rate64),
                                   "MF_MT_PIXEL_ASPECT_RATIO": ("u64", par64),
                                   "MF_MT_MPEG2_PROFILE": ("u32", CONSTANTS["eAVEncH264VProfile_Main"])})
        try:
            stream = c_uint32()
            com.check(self._writer, "IMFSinkWriter", "AddStream", target, byref(stream))
            self._stream = int(stream.value)
        finally:
            com.release(target)
        last = 0
        for subtype, stride in (("MFVideoFormat_RGB32", 4 * w), ("MFVideoFormat_NV12", w)):
            source = self._media_type({"MF_MT_MAJOR_TYPE": ("guid", "MFMediaType_Video"),
                                       "MF_MT_SUBTYPE": ("guid", subtype),
                                       "MF_MT_INTERLACE_MODE": ("u32", progressive),
                                       "MF_MT_FRAME_SIZE": ("u64", size64),
                                       "MF_MT_FRAME_RATE": ("u64", rate64),
                                       "MF_MT_PIXEL_ASPECT_RATIO": ("u64", par64),
                                       "MF_MT_DEFAULT_STRIDE": ("u32", stride)})
            try:
                enc = self._attrs({"CODECAPI_AVEncMPVGOPSize": ("u32", GOP_SECONDS * fps)}, 1)
                try:
                    hr = com.call(self._writer, "IMFSinkWriter", "SetInputMediaType", self._stream, source, enc)
                finally:
                    com.release(enc)
                if hr < 0:  # 인코더 속성을 받지 않는 인코더: 속성 없이
                    hr = com.call(self._writer, "IMFSinkWriter", "SetInputMediaType", self._stream, source, None)
            finally:
                com.release(source)
            if hr >= 0:
                self.input_subtype = subtype
                break
            last = hr
        else:
            raise MfError("IMFSinkWriter::SetInputMediaType", last)
        com.check(self._writer, "IMFSinkWriter", "BeginWriting")

    # ---- 프레임
    def add(self, frame: Any) -> None:
        if self._state != "open":
            raise ValueError("the MP4 writer is closed")
        px, _keep = frame_pixels(frame)
        rgb = np.ascontiguousarray(as_rgb32(px, self.width, self.height))
        if self.input_subtype == "MFVideoFormat_NV12":
            y, u, v = rgb32_to_yuv420(rgb)
            data = np.concatenate([y.ravel(), np.stack([u, v], axis=-1).ravel()])
        else:
            data = rgb.view(np.uint8).ravel()
        n = int(data.size)
        com = self._com
        buf = self._new("MFCreateMemoryBuffer", n)
        sample = None
        try:
            ptr, cap, cur = c_void_p(), c_uint32(), c_uint32()
            com.check(buf, "IMFMediaBuffer", "Lock", byref(ptr), byref(cap), byref(cur))
            try:
                if not ptr.value or cap.value < n:
                    raise MfError("IMFMediaBuffer::Lock size")
                ctypes.memmove(ptr.value, data.ctypes.data, n)
            finally:
                com.call(buf, "IMFMediaBuffer", "Unlock")
            com.check(buf, "IMFMediaBuffer", "SetCurrentLength", n)
            sample = self._new("MFCreateSample")
            com.check(sample, "IMFSample", "AddBuffer", buf)
            k = self.frames
            t0, t1 = k * _HNS // self.fps, (k + 1) * _HNS // self.fps
            com.check(sample, "IMFSample", "SetSampleTime", t0)
            com.check(sample, "IMFSample", "SetSampleDuration", t1 - t0)
            com.check(self._writer, "IMFSinkWriter", "WriteSample", self._stream, sample)
        finally:
            com.release(sample)
            com.release(buf)
        self.frames += 1

    def bytes_written(self) -> int:
        try:
            return os.path.getsize(self.tmp)
        except OSError:
            return 0

    def close(self) -> int:
        if self._state != "open":
            raise ValueError("the MP4 writer is closed")
        if not self.frames:
            self.abort()
            raise ValueError(S.VIDEO_NO_FRAMES)
        try:
            self._com.check(self._writer, "IMFSinkWriter", "Finalize")
            self._teardown()
            os.replace(self.tmp, self.path)
        except BaseException:
            self.abort()
            raise
        self._state = "done"
        return self.frames

    def abort(self) -> None:
        if self._state != "open":
            return
        self._state = "aborted"
        self._teardown()
        self._unlink_tmp()

    # ---- 정리
    def _teardown(self) -> None:
        writer, self._writer = self._writer, None
        if writer:
            with contextlib.suppress(Exception):  # 정리 중 실패는 무시
                self._com.release(writer)
        if self._session is not None:
            self._session.close()
            self._session = None

    def _unlink_tmp(self) -> None:
        try:
            self.tmp.unlink(missing_ok=True)
        except OSError:
            pass


def encoder_name(api: SimpleNamespace) -> str:
    """H.264 인코더 MFT의 이름(MFTEnumEx가 고른 첫 번째, 하드웨어가 먼저). 모르면 빈 글."""
    com = _Com()
    session = _Session(api)
    try:
        info = MFT_REGISTER_TYPE_INFO(GUID.of("MFMediaType_Video"), GUID.of("MFVideoFormat_H264"))
        arr = POINTER(c_void_p)()
        count = c_uint32()
        flags = (CONSTANTS["MFT_ENUM_FLAG_SYNCMFT"] | CONSTANTS["MFT_ENUM_FLAG_ASYNCMFT"]
                 | CONSTANTS["MFT_ENUM_FLAG_HARDWARE"] | CONSTANTS["MFT_ENUM_FLAG_SORTANDFILTER"])
        hr = int(api.MFTEnumEx(GUID.of("MFT_CATEGORY_VIDEO_ENCODER"), flags, None, byref(info), byref(arr),
                               byref(count)) or 0)
        if hr < 0 or not arr:
            return ""
        try:
            if count.value < 1:
                return ""
            text, length = c_void_p(), c_uint32()
            key = GUID.of("MFT_FRIENDLY_NAME_Attribute")
            hr = com.call(arr[0], "IMFActivate", "GetAllocatedString", byref(key), byref(text), byref(length))
            name = ctypes.wstring_at(text.value, length.value) if hr >= 0 and text.value else ""
            if text.value:
                api.CoTaskMemFree(text)
            return name
        finally:
            for i in range(count.value):
                com.release(arr[i])
            api.CoTaskMemFree(ctypes.cast(arr, c_void_p))
    finally:
        session.close()


def probe(api: SimpleNamespace | None = None) -> tuple[bool, str]:
    """MF로 64×64 두 장을 임시 폴더에 써 본다. (쓸 수 있나, 인코더 이름 또는 실패 까닭)."""
    try:
        api = api if api is not None else load_api()
    except MfError as exc:
        return False, str(exc)
    w, h = PROBE_SIZE
    with tempfile.TemporaryDirectory(prefix="evolab-mf-", ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "probe.mp4"
        try:
            writer = MfWriter(path, w, h, 30, bitrate=PROBE_BITRATE, api=api)
            try:
                writer.add(np.full((h, w), 0xFF2050A0, dtype=np.uint32))
                second = np.full((h, w), 0xFF2050A0, dtype=np.uint32)
                second[16:48, 16:48] = 0xFFF0C020
                writer.add(second)
                writer.close()
            except BaseException:
                writer.abort()
                raise
            size = path.stat().st_size
        except (MfError, OSError) as exc:
            return False, str(exc)
    if size <= 0:
        return False, "MFCreateSinkWriterFromURL: 0 bytes"
    try:
        name = encoder_name(api)
    except (MfError, OSError):
        name = ""
    return True, name or "H.264"


__all__ = [
    "CONSTANTS", "FUNCTYPE", "GUID", "GUIDS", "MFT_REGISTER_TYPE_INFO", "PROBE_BITRATE", "PROBE_SIZE", "VTABLE",
    "MfError", "MfWriter", "encoder_name", "load_api", "probe",
]

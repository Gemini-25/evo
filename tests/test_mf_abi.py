"""Media Foundation 표와 호출 (v3 설계 D53, 4.17, T38, W6).

- 표 자체 검사: 인터페이스마다 vtable 번호가 겹치지 않고 IUnknown 셋 다음부터, 상속한 메서드는 부모와 같은 번호,
  GUID 모양, 함수·메서드 원형이 표의 모든 메서드를 덮음.
- SDK 헤더와 비교(W6): 윈도우 CI가 packaging/ci/mf_abi.c를 SDK 헤더로 컴파일해 낸 JSON을 환경 변수 EVOLAB_MF_ABI(또는
  conftest가 등록했다면 --mf-abi 인자)로 주면 VTABLE·GUIDS·CONSTANTS가 그것과 같아야 한다. 주지 않으면 건너뛴다.
- mingw-w64 헤더와 비교(테스트 전용 도구가 있을 때만, 없으면 건너뜀): x86_64-w64-mingw32-gcc가 있으면 vtable 번호·상수를
  _Static_assert로 컴파일만 해서 확인(빠름), wine64까지 있으면 느린 테스트가 mf_abi.c를 돌린 출력(GUID 포함)과 비교.
- 가짜 COM: mf_win의 원형 표(_FLAT, _METHODS)와 호출 규약(FUNCTYPE)으로 만든 ctypes 콜백 vtable 객체로 MfWriter·probe를
  돌려, 호출 순서·인자 마샬링(GUID 포인터, UINT64, 100ns 시각, 버퍼 내용)·Release 짝·실패 처리를 어느 OS에서나 검사한다.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import subprocess
from ctypes import POINTER, c_ubyte, c_void_p
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from evolab.app.export import h264_pcm
from evolab.app.export import mf_win as M

ROOT = Path(__file__).resolve().parents[1]
ABI_C = ROOT / "packaging" / "ci" / "mf_abi.c"
E_FAIL = -0x7FFFBFFB  # 0x80004005
E_NOINTERFACE = -0x7FFFBFFE  # 0x80004002
MF_E_INVALIDMEDIATYPE = 0xC00D36B4 - (1 << 32)
MF_E_NOT_INITIALIZED = 0xC00D36B6 - (1 << 32)
ATTRIBUTE_DESCENDANTS = ("IMFMediaType", "IMFActivate", "IMFSample")


# ---------------------------------------------------------------------------
# 표 자체 검사 (T38)
# ---------------------------------------------------------------------------
def test_vtable_slots_are_unique_and_follow_iunknown() -> None:
    assert M.VTABLE["IUnknown"] == {"QueryInterface": 0, "AddRef": 1, "Release": 2}
    for iface, slots in M.VTABLE.items():
        assert len(set(slots.values())) == len(slots), iface
        if iface != "IUnknown":
            assert min(slots.values()) >= 3, iface
    attrs = M.VTABLE["IMFAttributes"]
    assert max(attrs.values()) <= 32  # IMFAttributes 메서드는 3..32(30개)
    for iface in ATTRIBUTE_DESCENDANTS:
        for name, slot in M.VTABLE[iface].items():
            if name in attrs:
                assert slot == attrs[name], (iface, name)
            else:
                assert slot >= 33, (iface, name)  # IMFAttributes 뒤에 붙은 메서드


def test_prototypes_cover_every_method_and_function() -> None:
    names = {m for slots in M.VTABLE.values() for m in slots}
    assert names <= set(M._METHODS)
    assert set(M._FLAT) == {"CoInitializeEx", "CoUninitialize", "CoTaskMemFree", "MFStartup", "MFShutdown",
                            "MFCreateAttributes", "MFCreateMediaType", "MFCreateMemoryBuffer", "MFCreateSample",
                            "MFTEnumEx", "MFCreateSinkWriterFromURL"}
    assert {lib for lib, _r, _a in M._FLAT.values()} == {"ole32", "mfplat", "mfreadwrite"}
    # 코드(표 뒤)에 글로 적힌 GUID·상수 이름은 모두 표에 있다(표에 없는 이름을 쓰면 KeyError가 실행 중에야 남)
    source = Path(M.__file__).read_text(encoding="utf-8").split("PROBE_SIZE: Final")[1].split("__all__")[0]
    used = set(re.findall(r'"((?:MF|CODECAPI_|COINIT_|eAVEnc|RPC_E_)[A-Za-z0-9_]+)"', source))
    assert used and used <= set(M.GUIDS) | set(M.CONSTANTS) | set(M._FLAT)


def test_guid_table_is_well_formed() -> None:
    pattern = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    assert all(pattern.fullmatch(v) for v in M.GUIDS.values())
    assert len(set(M.GUIDS.values())) == len(M.GUIDS)
    for name, text in M.GUIDS.items():
        g = M.GUID.of(name)
        assert g.text() == text
    g = M.GUID.of("MFVideoFormat_H264")
    assert g.Data1 == 0x34363248 and bytes(g.Data4) == bytes.fromhex("800000aa00389b71")  # FOURCC 'H264'


def test_ctypes_layouts_match_the_constants() -> None:
    assert ctypes.sizeof(M.GUID) == M.CONSTANTS["SIZEOF_GUID"]
    assert ctypes.sizeof(M.MFT_REGISTER_TYPE_INFO) == M.CONSTANTS["SIZEOF_MFT_REGISTER_TYPE_INFO"]
    assert M.CONSTANTS["MF_VERSION"] == (0x0002 << 16) | 0x0070


def test_abi_program_is_plain_ascii_and_lists_every_entry() -> None:
    text = ABI_C.read_text(encoding="ascii")
    for iface, slots in M.VTABLE.items():
        for name in slots:
            assert f"SLOT({iface}, {name})" in text, (iface, name)
    for name in M.GUIDS:
        assert f'guid("{name}"' in text, name
    for name in M.CONSTANTS:
        assert f'constant("{name}"' in text, name


def compare_with(abi: dict[str, Any]) -> None:
    assert abi["vtable"] == M.VTABLE
    assert abi["guids"] == M.GUIDS
    assert abi["constants"] == M.CONSTANTS


def test_tables_match_the_windows_sdk_headers(request: pytest.FixtureRequest) -> None:
    """W6: 윈도우 CI가 SDK 헤더로 컴파일한 mf_abi.exe의 출력(EVOLAB_MF_ABI 또는 --mf-abi)과 같다."""
    given = os.environ.get("EVOLAB_MF_ABI") or request.config.getoption("--mf-abi", default=None)
    if not given:
        pytest.skip("EVOLAB_MF_ABI not set (Windows CI step W6 sets it)")
    path = Path(given)
    assert path.is_file(), f"EVOLAB_MF_ABI points to a missing file: {path}"
    raw = path.read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    compare_with(json.loads(text))


MINGW = shutil.which("x86_64-w64-mingw32-gcc")
WINE = shutil.which("wine64") or (str(Path("/usr/lib/wine/wine64")) if Path("/usr/lib/wine/wine64").exists() else None)


@pytest.mark.skipif(not MINGW, reason="mingw-w64 not installed (test-only tool)")
def test_vtable_and_constants_match_the_mingw_headers_at_compile_time(tmp_path: Path) -> None:
    """mf_abi.c의 헤더 묶음 + 표마다 _Static_assert를 mingw-w64 헤더로 컴파일만 한다(실행하지 않아 빠름)."""
    head = ABI_C.read_text(encoding="ascii").split("#define SLOT(")[0]
    lines = [head, "#define SLOT(iface, method) ((int)(offsetof(iface##Vtbl, method) / sizeof(void *)))"]
    for iface, slots in M.VTABLE.items():
        for name, slot in slots.items():
            lines.append(f'_Static_assert(SLOT({iface}, {name}) == {slot}, "{iface}::{name} != {slot}");')
    for name, value in M.CONSTANTS.items():
        expr = {"SIZEOF_GUID": "sizeof(GUID)", "SIZEOF_MFT_REGISTER_TYPE_INFO": "sizeof(MFT_REGISTER_TYPE_INFO)"}
        lines.append(f'_Static_assert((unsigned long)({expr.get(name, name)}) == {value}UL, "{name} != {value}");')
    source = tmp_path / "abi_check.c"
    source.write_text("\n".join(lines) + "\n", encoding="ascii")
    run = subprocess.run([MINGW, "-fsyntax-only", "-Werror", str(source)], capture_output=True, text=True,
                         timeout=120, check=False)
    if run.returncode and "No such file or directory" in run.stderr:
        pytest.skip("this mingw-w64 has no Media Foundation headers")
    assert run.returncode == 0, run.stderr[-3000:]


def test_abi_program_has_no_argument_forwarding_macros() -> None:
    """W6는 MSVC로 컴파일한다. MSVC의 기본(전통) 전처리기는 넘겨받은 인자(STATIC_ GUID 값 목록처럼 쉼표가 든 것)를 다른
    함수 모양 매크로의 여러 인자로 나누지 않아, 목록을 매크로로 넘겨 GUID를 만들면 MSVC에서만 깨진 초기화가 된다(gcc·clang은
    나눔: v3.1 검토 결함). 함수 모양 매크로는 SLOT과 DEFINE_GUIDEX만, GOP GUID는 평평한 배열에서 코드로 만든다."""
    text = ABI_C.read_text(encoding="ascii")
    macros = re.findall(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_][A-Za-z0-9_]*)\(", text, re.MULTILINE)
    assert sorted(macros) == ["DEFINE_GUIDEX", "SLOT"]
    assert "static const unsigned long k_gop_vals[11] = { STATIC_CODECAPI_AVEncMPVGOPSize };" in text
    assert re.search(r"cl /nologo /W3 /Zc:preprocessor ", text)


CC = shutil.which("cc") or shutil.which("gcc")


@pytest.mark.skipif(not CC, reason="no host C compiler (test-only tool)")
def test_gop_size_guid_is_assembled_from_its_value_list(tmp_path: Path) -> None:
    """mf_abi.c의 k_gop_vals·gop_size_guid를 그대로 떼어 호스트 컴파일러로 돌리면(STATIC_ 목록 = mf_win.GUIDS 값) 같은 GUID 글.
    wine이 없어도 GUID를 만드는 코드 자체를 확인한다."""
    text = ABI_C.read_text(encoding="ascii")
    start = text.index("static const unsigned long k_gop_vals[11]")
    end = text.index("\n}\n", text.index("static GUID gop_size_guid(void)")) + 3
    want = M.GUIDS["CODECAPI_AVEncMPVGOPSize"]
    h = want.replace("-", "")
    values = [f"0x{h[0:8]}", f"0x{h[8:12]}", f"0x{h[12:16]}"] + [f"0x{h[16 + 2 * i:18 + 2 * i]}" for i in range(8)]
    source = tmp_path / "gop.c"
    source.write_text("\n".join([
        "#include <stdio.h>",
        "typedef struct { unsigned long Data1; unsigned short Data2; unsigned short Data3; unsigned char Data4[8]; } GUID;",
        f"#define STATIC_CODECAPI_AVEncMPVGOPSize {', '.join(values)}",
        text[start:end],
        "int main(void) { GUID g = gop_size_guid();",
        ' printf("%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", g.Data1, (unsigned)g.Data2, (unsigned)g.Data3,',
        "        g.Data4[0], g.Data4[1], g.Data4[2], g.Data4[3], g.Data4[4], g.Data4[5], g.Data4[6], g.Data4[7]);",
        " return 0; }",
    ]) + "\n", encoding="ascii")
    exe = tmp_path / "gop"
    subprocess.run([CC, "-Wall", "-Werror", str(source), "-o", str(exe)], check=True, capture_output=True, timeout=120)
    assert subprocess.run([str(exe)], capture_output=True, text=True, timeout=60, check=True).stdout == want


@pytest.mark.skipif(not MINGW, reason="mingw-w64 not installed (test-only tool)")
def test_abi_program_builds_with_mingw_warnings_as_errors(tmp_path: Path) -> None:
    """mf_abi.c 전체를 mingw-w64로 -Wall -Werror 빌드(실행은 wine이 있는 느린 시험만). 중괄호 생략 같은 경고도 막는다."""
    exe = tmp_path / "mf_abi.exe"
    run = subprocess.run([MINGW, "-Wall", "-Werror", str(ABI_C), "-o", str(exe), "-lmfuuid", "-lole32"],
                         capture_output=True, text=True, timeout=120, check=False)
    if run.returncode and "No such file or directory" in run.stderr:
        pytest.skip("this mingw-w64 has no Media Foundation headers")
    assert run.returncode == 0, run.stderr[-3000:]
    assert exe.stat().st_size > 0


@pytest.mark.slow
@pytest.mark.skipif(not (MINGW and WINE), reason="mingw-w64 and wine64 not installed (test-only tools)")
def test_tables_match_the_mingw_headers(tmp_path: Path) -> None:
    """GUID까지: mf_abi.c를 mingw-w64로 컴파일해 wine으로 돌린 출력과 비교(wine 접두사가 수백 MB라 느린 테스트)."""
    exe = tmp_path / "mf_abi.exe"
    subprocess.run([MINGW, "-Wall", "-Werror", str(ABI_C), "-o", str(exe), "-lmfuuid", "-lole32"], check=True,
                   capture_output=True, timeout=120)
    prefix = tmp_path / "wine"
    env = dict(os.environ, WINEPREFIX=str(prefix), WINEDEBUG="-all")
    try:
        run = subprocess.run([WINE, str(exe)], capture_output=True, timeout=240, env=env, check=False)
    finally:
        server = Path(WINE).with_name("wineserver")
        if server.exists():
            subprocess.run([str(server), "-k"], env=env, capture_output=True, timeout=60, check=False)
        shutil.rmtree(prefix, ignore_errors=True)
    if run.returncode != 0 or not run.stdout.strip():
        pytest.skip(f"wine could not run the program (exit {run.returncode})")
    compare_with(json.loads(run.stdout.decode("ascii")))


# ---------------------------------------------------------------------------
# 가짜 COM
# ---------------------------------------------------------------------------
class FakeCom:
    """vtable(ctypes 콜백 배열)을 가진 가짜 COM 객체. 주소 = 첫 칸이 vtable 포인터인 메모리."""

    INTERFACES: ClassVar[dict[str, tuple[str, ...]]] = {
        "attrs": ("IUnknown", "IMFAttributes"),
        "type": ("IUnknown", "IMFAttributes", "IMFMediaType"),
        "activate": ("IUnknown", "IMFAttributes", "IMFActivate"),
        "sample": ("IUnknown", "IMFAttributes", "IMFSample"),
        "buffer": ("IUnknown", "IMFMediaBuffer"),
        "writer": ("IUnknown", "IMFSinkWriter"),
    }

    def __init__(self, world: FakeMF, kind: str, **extra: Any) -> None:
        self.world, self.kind, self.refs = world, kind, 1
        self.values: dict[str, tuple[str, Any]] = {}
        self.__dict__.update(extra)
        slots: dict[int, str] = {}
        for iface in self.INTERFACES[kind]:
            for name, slot in M.VTABLE[iface].items():
                slots[slot] = name
        self.vtbl = (c_void_p * (max(slots) + 1))()
        self._callbacks = []
        for slot, name in slots.items():
            restype, argtypes = M._METHODS[name]
            cb = M.FUNCTYPE(restype, c_void_p, *argtypes)(self._wrap(name))
            self._callbacks.append(cb)
            self.vtbl[slot] = ctypes.cast(cb, c_void_p).value
        self.mem = (c_void_p * 1)(ctypes.addressof(self.vtbl))
        self.address = ctypes.addressof(self.mem)
        world.objects[self.address] = self

    def _wrap(self, name: str) -> Any:
        def call(this: int, *args: Any) -> int:
            try:
                assert this == self.address, f"{name}: wrong this"
                self.world.calls.append(f"{self.kind}.{name}")
                return getattr(self, "m_" + name)(*args)
            except Exception as exc:  # noqa: BLE001 - 테스트가 world.errors로 확인
                self.world.errors.append(f"{self.kind}.{name}: {exc!r}")
                return E_FAIL

        return call

    # IUnknown
    def m_QueryInterface(self, riid: Any, out: Any) -> int:
        return E_NOINTERFACE

    def m_AddRef(self) -> int:
        self.refs += 1
        return self.refs

    def m_Release(self) -> int:
        self.refs -= 1
        assert self.refs >= 0, f"{self.kind} released too often"
        if self.refs == 0:
            self.freed()
        return self.refs

    def freed(self) -> None:
        if self.kind == "sample":
            for b in self.buffers:
                b.m_Release()
        if self.kind == "writer" and self.fp is not None:
            self.fp.close()
            self.fp = None

    # IMFAttributes
    def m_SetUINT32(self, key: Any, value: int) -> int:
        self.values[key.contents.text()] = ("u32", value)
        return 0

    def m_SetUINT64(self, key: Any, value: int) -> int:
        self.values[key.contents.text()] = ("u64", value)
        return 0

    def m_SetGUID(self, key: Any, value: Any) -> int:
        self.values[key.contents.text()] = ("guid", value.contents.text())
        return 0

    def m_GetAllocatedString(self, key: Any, out: Any, length: Any) -> int:
        if key.contents.text() != M.GUIDS["MFT_FRIENDLY_NAME_Attribute"]:
            return E_FAIL
        buf = ctypes.create_unicode_buffer(self.name)
        self.world.strings.append(buf)
        out[0] = ctypes.addressof(buf)
        length[0] = len(self.name)
        return 0

    # IMFMediaBuffer
    def m_Lock(self, out: Any, cap: Any, cur: Any) -> int:
        out[0] = ctypes.addressof(self.data)
        cap[0] = len(self.data)
        cur[0] = self.length
        self.locked = True
        return 0

    def m_Unlock(self) -> int:
        self.locked = False
        return 0

    def m_SetCurrentLength(self, n: int) -> int:
        assert n <= len(self.data)
        self.length = n
        return 0

    # IMFSample
    def m_AddBuffer(self, addr: int) -> int:
        buf = self.world.objects[addr]
        buf.m_AddRef()
        self.buffers.append(buf)
        return 0

    def m_SetSampleTime(self, t: int) -> int:
        self.time = t
        return 0

    def m_SetSampleDuration(self, d: int) -> int:
        self.duration = d
        return 0

    # IMFSinkWriter
    def m_AddStream(self, target: int, out: Any) -> int:
        self.world.output_type = dict(self.world.objects[target].values)
        out[0] = 0
        return 0

    def m_SetInputMediaType(self, stream: int, source: int, params: int | None) -> int:
        values = dict(self.world.objects[source].values)
        enc = dict(self.world.objects[params].values) if params else None
        self.world.input_tries.append((values[M.GUIDS["MF_MT_SUBTYPE"]][1], enc))
        if self.world.reject_rgb32 and values[M.GUIDS["MF_MT_SUBTYPE"]][1] == M.GUIDS["MFVideoFormat_RGB32"]:
            return MF_E_INVALIDMEDIATYPE
        if self.world.reject_params and params:
            return MF_E_INVALIDMEDIATYPE
        self.world.input_type, self.world.encoder_params = values, enc
        return 0

    def m_BeginWriting(self) -> int:
        self.world.began = True
        return 0

    def m_WriteSample(self, stream: int, addr: int) -> int:
        sample = self.world.objects[addr]
        if self.world.fail_write_at is not None and len(self.world.samples) + 1 == self.world.fail_write_at:
            return MF_E_NOT_INITIALIZED
        (buf,) = sample.buffers
        data = bytes(buf.data[:buf.length])
        self.world.samples.append((sample.time, sample.duration, data))
        self.fp.write(data)
        self.fp.flush()
        return 0

    def m_Finalize(self) -> int:
        self.fp.write(b"moov")
        self.fp.close()
        self.fp = None
        self.world.finalized = True
        return 0


class FakeMF:
    """가짜 ole32·mfplat·mfreadwrite. api는 mf_win._FLAT 원형으로 만든 ctypes 함수들."""

    def __init__(self) -> None:
        self.objects: dict[int, FakeCom] = {}
        self.calls: list[str] = []
        self.errors: list[str] = []
        self.strings: list[Any] = []
        self.counts: dict[str, int] = {}
        self.coinit_result = 0
        self.startup_result = 0
        self.reject_rgb32 = False
        self.reject_params = False
        self.fail_write_at: int | None = None
        self.output_type: dict[str, Any] = {}
        self.input_type: dict[str, Any] = {}
        self.encoder_params: dict[str, Any] | None = None
        self.input_tries: list[tuple[str, Any]] = []
        self.samples: list[tuple[int, int, bytes]] = []
        self.url = ""
        self.writer_attrs: dict[str, Any] = {}
        self.began = self.finalized = False
        self._keep: list[Any] = []
        funcs = {}
        for name, (_lib, restype, argtypes) in M._FLAT.items():
            cb = M.FUNCTYPE(restype, *argtypes)(self._wrap(name, restype))
            self._keep.append(cb)
            funcs[name] = cb
        self.api = SimpleNamespace(**funcs)

    def _wrap(self, name: str, restype: Any) -> Any:
        def call(*args: Any) -> Any:
            self.counts[name] = self.counts.get(name, 0) + 1
            self.calls.append(name)
            try:
                return getattr(self, "f_" + name)(*args)
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"{name}: {exc!r}")
                return None if restype is None else E_FAIL

        return call

    def f_CoInitializeEx(self, reserved: Any, mode: int) -> int:
        assert reserved is None and mode == M.CONSTANTS["COINIT_MULTITHREADED"]
        return self.coinit_result

    def f_CoUninitialize(self) -> None:
        return None

    def f_CoTaskMemFree(self, p: Any) -> None:
        return None

    def f_MFStartup(self, version: int, flags: int) -> int:
        assert (version, flags) == (M.CONSTANTS["MF_VERSION"], M.CONSTANTS["MFSTARTUP_FULL"])
        return self.startup_result

    def f_MFShutdown(self) -> int:
        return 0

    def f_MFCreateAttributes(self, out: Any, size: int) -> int:
        out[0] = FakeCom(self, "attrs").address
        return 0

    def f_MFCreateMediaType(self, out: Any) -> int:
        out[0] = FakeCom(self, "type").address
        return 0

    def f_MFCreateMemoryBuffer(self, size: int, out: Any) -> int:
        out[0] = FakeCom(self, "buffer", data=(c_ubyte * size)(), length=0, locked=False).address
        return 0

    def f_MFCreateSample(self, out: Any) -> int:
        out[0] = FakeCom(self, "sample", buffers=[], time=None, duration=None).address
        return 0

    def f_MFTEnumEx(self, category: Any, flags: int, inp: Any, outp: Any, arr: Any, count: Any) -> int:
        assert category.text() == M.GUIDS["MFT_CATEGORY_VIDEO_ENCODER"]
        assert not inp
        assert outp.contents.guidSubtype.text() == M.GUIDS["MFVideoFormat_H264"]
        acts = [FakeCom(self, "activate", name="Fake Hardware H264 Encoder MFT"),
                FakeCom(self, "activate", name="H264 Encoder MFT")]
        table = (c_void_p * len(acts))(*[a.address for a in acts])
        self._keep.append(table)
        arr[0] = ctypes.cast(table, POINTER(c_void_p))
        count[0] = len(acts)
        return 0

    def f_MFCreateSinkWriterFromURL(self, url: str, stream: Any, attrs: int, out: Any) -> int:
        assert stream is None
        self.url = url
        self.writer_attrs = dict(self.objects[attrs].values)
        w = FakeCom(self, "writer", fp=open(url, "wb"))  # noqa: SIM115
        w.fp.write(b"ftyp")
        out[0] = w.address
        return 0

    def live(self) -> list[str]:
        return [f"{o.kind}:{o.refs}" for o in self.objects.values() if o.refs]


def frames(width: int, height: int, n: int) -> list[np.ndarray]:
    out = []
    for k in range(n):
        a = np.full((height, width), 0xFF0B0E14, dtype=np.uint32)
        a[k:k + 4, 2:6] = 0xFFFF7828
        out.append(a)
    return out


def test_fake_com_writer_call_sequence_and_arguments(tmp_path: Path) -> None:
    fake = FakeMF()
    target = tmp_path / "한글 폴더" / "영상.mp4"
    writer = M.MfWriter(target, 64, 48, 30, bitrate=2_000_000, tmp=target.with_name("영상.tmp.mp4"), api=fake.api)
    src = frames(64, 48, 3)
    for f in src:
        writer.add(f)
    assert writer.bytes_written() == 4 + 3 * 64 * 48 * 4
    assert writer.close() == 3
    assert fake.errors == []
    assert fake.live() == []  # 모든 COM 객체가 Release됨
    assert fake.counts["CoInitializeEx"] == fake.counts["CoUninitialize"] == 1
    assert fake.counts["MFStartup"] == fake.counts["MFShutdown"] == 1
    assert fake.url == os.path.abspath(target.with_name("영상.tmp.mp4"))
    assert target.read_bytes().startswith(b"ftyp") and target.read_bytes().endswith(b"moov")
    assert not target.with_name("영상.tmp.mp4").exists()
    g = M.GUIDS
    assert fake.writer_attrs == {g["MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS"]: ("u32", 1),
                                 g["MF_SINK_WRITER_DISABLE_THROTTLING"]: ("u32", 1),
                                 g["MF_TRANSCODE_CONTAINERTYPE"]: ("guid", g["MFTranscodeContainerType_MPEG4"])}
    assert fake.output_type == {g["MF_MT_MAJOR_TYPE"]: ("guid", g["MFMediaType_Video"]),
                                g["MF_MT_SUBTYPE"]: ("guid", g["MFVideoFormat_H264"]),
                                g["MF_MT_AVG_BITRATE"]: ("u32", 2_000_000),
                                g["MF_MT_INTERLACE_MODE"]: ("u32", 2),
                                g["MF_MT_FRAME_SIZE"]: ("u64", (64 << 32) | 48),
                                g["MF_MT_FRAME_RATE"]: ("u64", (30 << 32) | 1),
                                g["MF_MT_PIXEL_ASPECT_RATIO"]: ("u64", (1 << 32) | 1),
                                g["MF_MT_MPEG2_PROFILE"]: ("u32", 77)}
    assert fake.input_type[g["MF_MT_SUBTYPE"]] == ("guid", g["MFVideoFormat_RGB32"])
    assert fake.input_type[g["MF_MT_DEFAULT_STRIDE"]] == ("u32", 64 * 4)
    assert fake.input_type[g["MF_MT_FRAME_SIZE"]] == ("u64", (64 << 32) | 48)
    assert fake.encoder_params == {g["CODECAPI_AVEncMPVGOPSize"]: ("u32", 60)}
    assert writer.input_subtype == "MFVideoFormat_RGB32"
    assert [(t, d) for t, d, _b in fake.samples] == [(0, 333333), (333333, 333333), (666666, 333334)]
    assert [b for _t, _d, b in fake.samples] == [f.tobytes() for f in src]
    order = [c for c in fake.calls if c in ("CoInitializeEx", "MFStartup", "MFCreateSinkWriterFromURL",
                                            "writer.AddStream", "writer.SetInputMediaType", "writer.BeginWriting",
                                            "writer.WriteSample", "writer.Finalize", "MFShutdown", "CoUninitialize")]
    assert order == ["CoInitializeEx", "MFStartup", "MFCreateSinkWriterFromURL", "writer.AddStream",
                     "writer.SetInputMediaType", "writer.BeginWriting", "writer.WriteSample", "writer.WriteSample",
                     "writer.WriteSample", "writer.Finalize", "MFShutdown", "CoUninitialize"]


def test_fake_com_nv12_when_rgb32_is_refused(tmp_path: Path) -> None:
    fake = FakeMF()
    fake.reject_rgb32 = True
    writer = M.MfWriter(tmp_path / "v.mp4", 32, 16, 25, bitrate=1_000_000, api=fake.api)
    src = frames(32, 16, 2)
    for f in src:
        writer.add(f)
    writer.close()
    g = M.GUIDS
    assert [t for t, _e in fake.input_tries] == [g["MFVideoFormat_RGB32"], g["MFVideoFormat_RGB32"],
                                                 g["MFVideoFormat_NV12"]]
    assert writer.input_subtype == "MFVideoFormat_NV12"
    assert fake.input_type[g["MF_MT_DEFAULT_STRIDE"]] == ("u32", 32)
    y, u, v = h264_pcm.rgb32_to_yuv420(src[1])
    assert fake.samples[1][2] == y.tobytes() + np.stack([u, v], axis=-1).tobytes()
    assert len(fake.samples[0][2]) == 32 * 16 * 3 // 2
    assert fake.errors == [] and fake.live() == []


def test_fake_com_encoder_parameters_refused_retries_without(tmp_path: Path) -> None:
    fake = FakeMF()
    fake.reject_params = True
    writer = M.MfWriter(tmp_path / "v.mp4", 32, 16, 30, bitrate=1_000_000, api=fake.api)
    writer.add(frames(32, 16, 1)[0])
    writer.close()
    assert [e is None for _t, e in fake.input_tries] == [False, True]
    assert fake.encoder_params is None and writer.input_subtype == "MFVideoFormat_RGB32"
    assert fake.errors == [] and fake.live() == []


def test_fake_com_failure_while_writing_raises_and_cleans_up(tmp_path: Path) -> None:
    fake = FakeMF()
    fake.fail_write_at = 2
    target = tmp_path / "v.mp4"
    target.write_bytes(b"old")
    writer = M.MfWriter(target, 32, 16, 30, bitrate=1_000_000, tmp=tmp_path / "v.tmp.mp4", api=fake.api)
    writer.add(frames(32, 16, 1)[0])
    with pytest.raises(M.MfError) as info:
        writer.add(frames(32, 16, 1)[0])
    assert str(info.value) == "IMFSinkWriter::WriteSample 0xC00D36B6"
    assert info.value.hresult == 0xC00D36B6
    writer.abort()
    assert target.read_bytes() == b"old" and not (tmp_path / "v.tmp.mp4").exists()
    assert fake.live() == []
    assert fake.counts["MFShutdown"] == 1 and fake.counts["CoUninitialize"] == 1
    assert fake.errors == []


def test_ctypes_faults_become_mf_errors_named_after_the_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ctypes가 외부 호출에서 낸 잘못(윈도우 SEH 접근 위반 → OSError, 마샬링 → ctypes.ArgumentError)은 그 호출 이름을 단
    MfError: write_mp4가 받아 내장 쓰기로 다시 만든다(시험 probe가 이미 (MfError, OSError)로 보듯). v3.1 검토 결함: 쓰는 중의
    OSError는 대체 없이 실패로 끝났다."""
    fake = FakeMF()
    target = tmp_path / "v.mp4"
    writer = M.MfWriter(target, 32, 16, 30, bitrate=1_000_000, tmp=tmp_path / "v.tmp.mp4", api=fake.api)
    writer.add(frames(32, 16, 1)[0])
    # DLL 함수(팩토리)의 접근 위반
    real_sample = fake.api.MFCreateSample

    def access_violation(*_args: Any) -> int:
        raise OSError("exception: access violation reading 0x0000000000000010")

    fake.api.MFCreateSample = access_violation
    with pytest.raises(M.MfError, match=r"^MFCreateSample: OSError: exception: access violation") as info:
        writer.add(frames(32, 16, 1)[0])
    assert isinstance(info.value.__cause__, OSError)
    fake.api.MFCreateSample = real_sample
    # 버퍼 복사(memmove)의 접근 위반
    real_memmove = ctypes.memmove

    def bad_memmove(dst: Any, src: Any, count: int) -> Any:
        raise OSError("exception: access violation writing 0x0")

    monkeypatch.setattr(ctypes, "memmove", bad_memmove)
    with pytest.raises(M.MfError, match=r"^IMFMediaBuffer memmove: OSError"):
        writer.add(frames(32, 16, 1)[0])
    monkeypatch.setattr(ctypes, "memmove", real_memmove)
    # COM 메서드 인자 마샬링(ctypes.ArgumentError)
    com = M._Com()
    sample = FakeCom(fake, "sample", buffers=[], time=None, duration=None)
    with pytest.raises(M.MfError, match=r"^IMFSample::SetSampleTime: ArgumentError"):
        com.call(sample.address, "IMFSample", "SetSampleTime", "not a time")
    com.release(sample.address)
    writer.add(frames(32, 16, 1)[0])  # 실패한 프레임 뒤에도 작성기는 그대로 쓸 수 있다
    writer.abort()
    assert not target.exists() and not (tmp_path / "v.tmp.mp4").exists()
    assert fake.live() == [] and fake.errors == []


def test_write_mp4_rebuilds_with_the_builtin_writer_after_an_access_violation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MF 작성기(가짜 COM)가 셋째 프레임에서 접근 위반 → MfError → write_mp4가 내장 쓰기로 처음부터 + EXP_MF_FALLBACK 한 줄."""
    from evolab.app import strings as S
    from evolab.app.export import mp4

    fake = FakeMF()
    calls = {"n": 0}
    real_sample = fake.api.MFCreateSample

    def sample_then_fault(out: Any) -> int:
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("exception: access violation reading 0x0")
        return int(real_sample(out))

    fake.api.MFCreateSample = sample_then_fault
    monkeypatch.setattr(mp4, "_STATUS", mp4.EncoderStatus("mf", "Fake H264 MFT"))
    monkeypatch.setattr(mp4, "_open_mf", lambda path, w, h, fps: M.MfWriter(
        path, w, h, fps, bitrate=mp4.mf_bitrate(w, h), tmp=mp4.temp_path(path), api=fake.api))
    notes: list[str] = []
    target = tmp_path / "v.mp4"
    out = mp4.write_mp4(lambda: iter(frames(32, 16, 6)), target, 32, 16, 30, note=notes.append)
    assert out is not None and out["encoder"] == "builtin" and out["frames"] == 6
    assert notes == [S.EXP_MF_FALLBACK.format(reason="MFCreateSample: OSError: exception: access violation reading 0x0")]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v.mp4"]
    assert mp4.read_info(target)["frames"] == 6
    assert fake.live() == []


def test_fake_com_start_failure_and_changed_mode(tmp_path: Path) -> None:
    fake = FakeMF()
    fake.startup_result = 0xC00D36E3 - (1 << 32)
    with pytest.raises(M.MfError, match="MFStartup 0xC00D36E3"):
        M.MfWriter(tmp_path / "v.mp4", 32, 16, 30, bitrate=1_000_000, api=fake.api)
    assert fake.counts["CoUninitialize"] == 1 and "MFShutdown" not in fake.counts
    assert not list(tmp_path.iterdir())
    # COM이 이미 다른 방식(STA)으로 초기화돼 있으면 그대로 쓰고 CoUninitialize를 부르지 않는다
    fake = FakeMF()
    fake.coinit_result = M.CONSTANTS["RPC_E_CHANGED_MODE"] - (1 << 32)
    writer = M.MfWriter(tmp_path / "v.mp4", 32, 16, 30, bitrate=1_000_000, api=fake.api)
    writer.add(frames(32, 16, 1)[0])
    writer.close()
    assert "CoUninitialize" not in fake.counts and fake.live() == [] and fake.errors == []


def test_fake_probe_reports_the_encoder_name() -> None:
    fake = FakeMF()
    assert M.probe(fake.api) == (True, "Fake Hardware H264 Encoder MFT")
    assert fake.errors == [] and fake.live() == []
    assert fake.counts["CoTaskMemFree"] == 2  # 이름 글 + 활성화 객체 배열
    assert fake.counts["MFStartup"] == fake.counts["MFShutdown"] == 2  # 시험 쓰기 + 이름 찾기
    assert len(fake.samples) == 2 and all(len(b) == 64 * 64 * 4 for _t, _d, b in fake.samples)


def test_fake_probe_failure_gives_the_hresult() -> None:
    fake = FakeMF()
    fake.reject_rgb32 = True
    fake.reject_params = True
    fake_ok = FakeMF()
    fake_ok.fail_write_at = 1
    assert M.probe(fake_ok.api) == (False, "IMFSinkWriter::WriteSample 0xC00D36B6")
    assert fake_ok.live() == []
    original = fake.f_MFCreateSinkWriterFromURL

    def refuse(url: str, stream: Any, attrs: int, out: Any) -> int:
        return MF_E_INVALIDMEDIATYPE

    fake.f_MFCreateSinkWriterFromURL = refuse  # type: ignore[method-assign]
    assert M.probe(fake.api) == (False, "MFCreateSinkWriterFromURL 0xC00D36B4")
    assert fake.live() == [] and original is not None


def test_mf_writer_refuses_odd_sizes(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        M.MfWriter(tmp_path / "v.mp4", 33, 16, 30, bitrate=1, api=FakeMF().api)

"""윈도우 전용 작은 도우미(Qt 없이 ctypes로). 윈도우가 아니면 아무것도 하지 않는다(v3 설계 D32·D40, 4.9).

- `create_app_mutex()`: 화면 프로세스가 이름 있는 뮤텍스 `evolab-app-running`을 만들고 끝까지 쥔다. v3.1 설치 프로그램
  (Inno Setup `AppMutex`)이 프로그램이 떠 있는지 알아보는 데 쓴다. 학습 프로세스·워커도 같은 exe라 화면 프로세스만 만든다.
- `message_box(title, text)`: Qt 없이 띄우는 오류 상자(창을 못 띄우는 시작 실패). 다른 OS는 표준 오류에 쓴다.
- `known_folder(name)`: 바탕 화면·문서 폴더(OneDrive로 옮긴 폴더 포함). Qt 없는 명령줄 `evolab report`가 쓴다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final

MUTEX_NAME: Final = "evolab-app-running"

_MB_OK: Final = 0x0
_MB_ICONERROR: Final = 0x10
_MB_SETFOREGROUND: Final = 0x10000
_MB_TOPMOST: Final = 0x40000

_FOLDER_IDS: Final = {
    "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
}

_held: list[Any] = []
"""만든 뮤텍스 핸들. 프로세스가 끝날 때까지 닫지 않는다(닫으면 설치 프로그램이 실행 중인 것을 모른다)."""


def create_app_mutex() -> object | None:
    """kernel32 CreateMutexW(None, False, "evolab-app-running"). 윈도우가 아니거나 실패하면 None."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel32.CreateMutexW
        create.restype = wintypes.HANDLE
        create.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        handle = create(None, False, MUTEX_NAME)
        if not handle:
            return None
        _held.append(handle)
        return handle
    except Exception:
        return None


def message_box(title: str, text: str) -> None:
    """Qt 없는 오류 상자(user32 MessageBoxW). 윈도우가 아니면(또는 실패하면) 표준 오류에 쓴다."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)
            box = user32.MessageBoxW
            box.restype = ctypes.c_int
            box.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT]
            box(None, str(text), str(title), _MB_OK | _MB_ICONERROR | _MB_SETFOREGROUND | _MB_TOPMOST)
            return
        except Exception:
            pass
    try:
        if sys.stderr is not None:
            print(f"{title}\n{text}", file=sys.stderr)
    except Exception:
        pass


def known_folder(name: str) -> Path | None:
    """윈도우 알려진 폴더(SHGetKnownFolderPath): "desktop" 또는 "documents". 윈도우가 아니거나 실패하면 None."""
    if sys.platform != "win32" or name not in _FOLDER_IDS:
        return None
    try:
        import ctypes
        import uuid
        from ctypes import wintypes

        class _GUID(ctypes.Structure):
            _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16),
                        ("Data4", ctypes.c_ubyte * 8)]

        u = uuid.UUID(_FOLDER_IDS[name])
        guid = _GUID(u.time_low, u.time_mid, u.time_hi_version, (ctypes.c_ubyte * 8)(*u.bytes[8:]))
        shell32 = ctypes.WinDLL("shell32")
        ole32 = ctypes.WinDLL("ole32")
        get = shell32.SHGetKnownFolderPath
        get.restype = ctypes.c_long
        get.argtypes = [ctypes.POINTER(_GUID), wintypes.DWORD, wintypes.HANDLE, ctypes.POINTER(ctypes.c_void_p)]
        ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
        ole32.CoTaskMemFree.restype = None
        out = ctypes.c_void_p()
        hr = get(ctypes.byref(guid), 0, None, ctypes.byref(out))
        try:
            if hr != 0 or not out.value:
                return None
            text = ctypes.wstring_at(out.value)
        finally:
            if out.value:
                ole32.CoTaskMemFree(out)
        return Path(text) if text else None
    except Exception:
        return None

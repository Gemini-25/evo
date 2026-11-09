"""진단: 오류 기록(errors.jsonl), 치명적 오류 스택(faults.log), 갑자기 닫힘 표시, 종료 코드 풀이 (v3 설계 D27~D30, 4.9).

이 모듈은 맨 위에서 Qt도 numpy도 불러오지 않는다. exe 진입점(packaging/entry.py)과 `python -m evolab`가
`ensure_std_streams()` 바로 뒤, `freeze_support()` 전에 `enable_fault_log()`를 부르므로, spawn으로 만든 자식(학습
프로세스, 워커)까지 모든 exe 프로세스가 같은 faults.log에 치명적 오류 스택을 남긴다. 화면 쪽 것(`install_gui`)만 그 안에서
Qt를 불러온다.

- `install(where)`: sys.excepthook·threading.excepthook·sys.unraisablehook. "gui"면 시작 기록 한 줄도 남긴다. "cli"는 훅만.
- `install_gui(on_crash)`: Qt 메시지 처리기(경고 이상, 같은 글은 60초에 한 번), `running-<pid>.flag`, 지난 실행의 표시 확인,
  faults.log 자르기. 잡히지 않은 예외는 화면 스레드의 이벤트 루프에서 `on_crash(action, line)`으로 알린다.
  action = "box"(새 오류: 비모달 상자, 한 실행에 3번까지), "again"(같은 서명의 오류가 또: 상태 줄), "more"(상자를 이미 3번
  보인 뒤의 다른 오류: 상태 줄). line = 「예외 종류: 글」 한 줄.
- `record(kind, message, detail, where)`: errors.jsonl에 한 줄. 이 프로세스가 `install`을 불렀거나(화면·명령줄 진입)
  EVOLAB_STATE_DIR가 있을 때(자체 점검·테스트)만 쓴다. 그래서 테스트가 학습 오류를 일으켜도 사용자의 진짜 로그 폴더에
  쓰지 않는다(설계 R16). 쓰기 실패는 조용히 넘어간다.

errors.jsonl은 사용자 PC 안에만 있으므로 경로를 그대로 적는다. 사용자 이름이 든 경로는 오류 보고 묶음을 만들 때 가린다.
"""

from __future__ import annotations

import faulthandler
import json
import os
import platform
import sys
import threading
import time
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from evolab import __version__, paths
from evolab.app import strings as S

ERRORS_FILE: Final = "errors.jsonl"
FAULTS_FILE: Final = "faults.log"
FLAG_PREFIX: Final = "running-"
FLAG_SUFFIX: Final = ".flag"

ERRORS_MAX_LINES: Final = 500
ERRORS_MAX_BYTES: Final = 1024 * 1024
"""이 줄 수나 크기를 넘으면 앞쪽을 잘라 다시 쓴다(설계 2.3)."""
ERRORS_KEEP_LINES: Final = 400
ERRORS_KEEP_BYTES: Final = 768 * 1024
"""자를 때 남기는 최근 줄. 상한보다 조금 적게 남겨, 상한에 닿은 뒤 한 줄 쓸 때마다 파일 전체를 다시 쓰지 않게 한다."""
MESSAGE_MAX: Final = 2000
DETAIL_MAX: Final = 64 * 1024
"""한 줄에 넣는 message·detail 최대 글자 수(긴 traceback 하나가 기록을 다 차지하지 않게)."""

FAULTS_MAX_BYTES: Final = 1024 * 1024
FAULTS_KEEP_BYTES: Final = 256 * 1024
"""faults.log가 화면 시작 때 이보다 크면 끝 256KB만 남긴다(열린 핸들이 없을 때만: 윈도우에서 다른 프로세스가 쥐고 있으면 그대로)."""

MAX_BOXES: Final = 3
"""한 실행에서 잡히지 않은 예외 상자를 보이는 최대 횟수. 그 뒤는 상태 줄."""
QT_REPEAT_S: Final = 60.0
"""같은 Qt 경고는 이 초 안에 한 번만 기록."""
AGAIN_REPEAT_S: Final = 2.0
"""같은 서명의 예외를 상태 줄로 다시 알리는 최소 간격(그리기 안에서 매번 나는 예외가 상태 줄을 쏟아내지 않게)."""

SELFTEST_CRASH_ENV: Final = "EVOLAB_SELFTEST_CRASH"
CI_ENV: Final = "CI"
"""이 환경 변수가 비어 있지 않으면(GitHub Actions는 CI=true) 명령줄 훅이 한 줄 앞에 traceback 전체도 stderr에 쓴다."""
"""이 환경 변수가 "1"이면 자체 점검이 일부러 세그폴트하는 자식으로 faults.log를 확인한다(윈도우 CI W2). 사용자 PC에서는 돌지 않는다."""
CRASH_MARKERS: Final = ("Fatal Python error", "Windows fatal exception")
"""faulthandler가 치명적 오류 때 쓰는 머리글(리눅스·윈도우의 raise(SIGSEGV)는 앞쪽, 윈도우 SEH 예외는 뒤쪽)."""

_now: Callable[[], float] = time.monotonic
"""시계(테스트가 바꿔 끼운다)."""


class _State:
    def __init__(self) -> None:
        self.where: str | None = None
        self.prev_excepthook: Any = None
        self.prev_threadhook: Any = None
        self.prev_unraisable: Any = None
        self.on_crash: Callable[[str, str], None] | None = None
        self.bridge: Any = None
        self.prev_qt_handler: Any = None
        self.qt_installed = False
        self.qt_seen: dict[tuple[str, str], float] = {}
        self.seen: dict[str, int] = {}
        self.again_at: dict[str, float] = {}
        self.boxes = 0
        self.flag: Path | None = None


_state = _State()
_lock = threading.RLock()
_fault_file: Any = None


# ---------------------------------------------------------------------------
# faulthandler
# ---------------------------------------------------------------------------
def enable_fault_log() -> None:
    """치명적 오류(세그폴트, 접근 위반, abort) 때 모든 스레드의 파이썬 스택을 logs/faults.log에 덧붙이게 한다.

    여러 번 불러도 한 번. 폴더나 파일을 못 열면 조용히 넘어간다(진단 때문에 프로그램이 못 뜨면 안 된다).
    """
    global _fault_file
    if _fault_file is not None:
        return
    try:
        folder = paths.log_dir()
        folder.mkdir(parents=True, exist_ok=True)
        f = open(folder / FAULTS_FILE, "a", encoding="utf-8")  # noqa: SIM115  faulthandler가 끝까지 쥔다
        try:
            faulthandler.enable(file=f, all_threads=True)
        except Exception:
            f.close()
            raise
        _fault_file = f
    except Exception:
        _fault_file = None


def _trim_faults() -> None:
    """faults.log가 1MB를 넘으면 끝 256KB만 남긴다. 화면 시작 때 한 번(그때는 학습 프로세스·워커가 아직 없다)."""
    global _fault_file
    path = paths.log_dir() / FAULTS_FILE
    try:
        if path.stat().st_size <= FAULTS_MAX_BYTES:
            return
    except OSError:
        return
    reopen = _fault_file is not None
    if reopen:  # 이 프로세스의 핸들부터 닫는다(윈도우에서 열린 파일은 바꿔 쓸 수 없다)
        try:
            faulthandler.disable()
            _fault_file.close()
        except Exception:
            pass
        _fault_file = None
    tmp = path.with_name(f".{FAULTS_FILE}.{os.getpid()}.tmp")
    try:
        with open(path, "rb") as f:
            f.seek(-FAULTS_KEEP_BYTES, os.SEEK_END)
            tail = f.read()
        cut = tail.find(b"\n")
        if 0 <= cut < len(tail) - 1:
            tail = tail[cut + 1:]
        tmp.write_bytes(tail)
        os.replace(tmp, path)  # 다른 프로세스가 쥐고 있으면(윈도우) 실패: 그대로 둔다(설계 되돌릴 지점 5)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
    finally:
        if reopen:
            enable_fault_log()


def _crash_child() -> None:
    """일부러 세그폴트하는 자식(spawn 대상). exe에서는 진입점이 이미 faulthandler를 켰으므로 그것을 확인하려고 다시 켜지 않는다.

    개발 실행(`uv run`)의 spawn 자식은 진입점을 다시 거치지 않으므로 직접 켠다(설계 D28).
    """
    if not getattr(sys, "frozen", False):
        enable_fault_log()
    faulthandler._sigsegv()  # type: ignore[attr-defined]


def crash_check(timeout: float = 60.0) -> tuple[bool, str]:
    """같은 exe(개발 실행이면 같은 파이썬)로 spawn 자식을 띄워 세그폴트시키고, 종료 코드 ≠ 0과 faults.log에 남은 치명적 오류
    머리글·함수 이름을 확인한다(설계 4.9, W2). 돌려주는 값: (통과, 자세히 한 줄: ASCII).

    자체 점검이 `EVOLAB_SELFTEST_CRASH=1`일 때만 부른다. faults.log 위치는 지금 환경 변수(EVOLAB_STATE_DIR)를 따른다.
    """
    import multiprocessing as mp

    path = paths.log_dir() / FAULTS_FILE
    try:
        start = path.stat().st_size
    except OSError:
        start = 0
    proc = mp.get_context("spawn").Process(target=_crash_child, name="evolab-crash-check", daemon=True)
    t0 = time.monotonic()
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return False, f"timeout {timeout:.0f}s"
    code = proc.exitcode
    try:
        with open(path, "rb") as f:
            f.seek(start)
            added = f.read().decode("utf-8", "replace")
    except OSError:
        added = ""
    marker = any(m in added for m in CRASH_MARKERS)
    named = "_crash_child" in added
    ok = code not in (0, None) and marker and named
    detail = (f"exit={code} faults.log+={len(added.encode('utf-8'))}B marker={'yes' if marker else 'no'} "
              f"function={'yes' if named else 'no'} {time.monotonic() - t0:.2f}s")
    return ok, detail


# ---------------------------------------------------------------------------
# 기록
# ---------------------------------------------------------------------------
def _enabled() -> bool:
    return _state.where is not None or bool(os.environ.get(paths.STATE_DIR_ENV, "").strip())


def errors_path() -> Path:
    return paths.log_dir() / ERRORS_FILE


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def record(kind: str, message: str, detail: str = "", where: str = "gui") -> None:
    """errors.jsonl에 한 줄을 덧붙인다(설계 2.3). 500줄 또는 1MB를 넘으면 앞쪽을 잘라 다시 쓴다. 실패는 조용히 넘어간다."""
    if not _enabled():
        return
    entry = {
        "t": datetime.now().isoformat(timespec="seconds"),
        "kind": str(kind),
        "where": str(where),
        "pid": os.getpid(),
        "version": __version__,
        "message": _clip(str(message), MESSAGE_MAX),
        "detail": _clip(str(detail or ""), DETAIL_MAX),
    }
    line = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8", "replace")
    with _lock:
        try:
            path = errors_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "ab") as f:
                f.write(line)
            _trim_errors(path)
        except Exception:
            pass


def _trim_errors(path: Path) -> None:
    size = path.stat().st_size
    # 한 줄은 110바이트보다 짧을 수 없어(필드 이름만으로) 이보다 작으면 500줄을 넘지 않는다: 그때는 읽지 않는다.
    if size <= ERRORS_MAX_LINES * 100 and size <= ERRORS_MAX_BYTES:
        return
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    if len(lines) <= ERRORS_MAX_LINES and len(data) <= ERRORS_MAX_BYTES:
        return
    keep: list[bytes] = []
    total = 0
    for raw in reversed(lines[-ERRORS_KEEP_LINES:]):
        if total + len(raw) > ERRORS_KEEP_BYTES:
            break
        keep.append(raw)
        total += len(raw)
    keep.reverse()
    tmp = path.with_name(f".{ERRORS_FILE}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(b"".join(keep))
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass


def recent(n: int = 200) -> list[dict[str, Any]]:
    """errors.jsonl의 최근 n줄(읽을 수 없는 줄은 건너뜀). 없으면 빈 목록."""
    try:
        lines = errors_path().read_bytes().splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for raw in lines[-n:] if n > 0 else []:
        try:
            item = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# 종료 코드
# ---------------------------------------------------------------------------
def exit_code_text(code: int | None) -> str:
    """학습 프로세스 종료 코드의 뜻(설계 D29). None·0은 빈 글, 음수(-1..-255)는 신호 이름, 윈도우 NTSTATUS는 표, 그 밖은 「알 수 없는 원인」.

    윈도우 종료 코드가 부호 있는 32비트로 오면(-1073741819) 부호 없는 값(3221225477)으로 바꿔 찾는다.
    """
    if code is None or code == 0:
        return ""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return S.EXIT_CODE_UNKNOWN
    if -256 < code < 0:
        import signal

        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = str(-code)
        return S.EXIT_CODE_SIGNAL.format(name=name)
    if code < 0:
        code &= 0xFFFFFFFF
    return S.EXIT_CODE_TEXT.get(code, S.EXIT_CODE_UNKNOWN)


# ---------------------------------------------------------------------------
# 이 PC 정보 (경로·사용자 이름·컴퓨터 이름 없음)
# ---------------------------------------------------------------------------
def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            try:
                value, _kind = winreg.QueryValueEx(key, "ProcessorNameString")
            finally:
                winreg.CloseKey(key)
            if str(value).strip():
                return " ".join(str(value).split())
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for row in f:
                    if row.lower().startswith("model name"):
                        return " ".join(row.split(":", 1)[1].split())
        except OSError:
            pass
    return platform.processor() or platform.machine() or "-"


def _qt_text() -> str:
    """Qt 버전. 이미 Qt를 불러온 프로세스(화면)에서만 Qt에 묻고, 아니면 설치 정보만 읽는다(명령줄 `report`는 Qt를 불러오지 않는다)."""
    core = sys.modules.get("PySide6.QtCore")
    if core is not None:
        try:
            import PySide6

            return f"PySide6 {PySide6.__version__}, Qt {core.qVersion()}"
        except Exception:
            pass
    try:
        from importlib import metadata

        return f"PySide6 {metadata.version('pyside6-essentials')}"
    except Exception:
        return "—"


def _screen_text() -> str:
    """화면 크기와 배율(화면 프로세스에서만, 명령줄은 「—」)."""
    gui = sys.modules.get("PySide6.QtGui")
    if gui is None:
        return "—"
    try:
        app = gui.QGuiApplication.instance()
        if app is None:
            return "—"
        parts = []
        for screen in gui.QGuiApplication.screens():
            size = screen.size()
            scale = round(screen.devicePixelRatio() * 100)
            parts.append(f"{size.width()}×{size.height()} ({scale}%)")
        return ", ".join(parts) or "—"
    except Exception:
        return "—"


def platform_info() -> dict[str, str]:
    """오류 보고에 적는 이 PC 정보. 경로, 사용자 이름, 컴퓨터 이름은 넣지 않는다."""
    info: dict[str, str] = {
        "version": __version__,
        "frozen": "1" if getattr(sys, "frozen", False) else "0",
        "os": platform.platform(),
        "cpu": _cpu_name(),
        "python": f"{platform.python_version()} ({platform.architecture()[0]})",
        "qt": _qt_text(),
        "screen": _screen_text(),
    }
    try:
        import psutil

        info["cores"] = f"{psutil.cpu_count(logical=False) or '?'}/{psutil.cpu_count(logical=True) or '?'}"
        info["memory"] = f"{psutil.virtual_memory().total / 2**30:.1f} GB"
    except Exception:
        info["cores"] = f"?/{os.cpu_count() or '?'}"
        info["memory"] = "—"
    return info


# ---------------------------------------------------------------------------
# 훅
# ---------------------------------------------------------------------------
def install(where: str) -> None:
    """잡히지 않은 예외 훅 셋을 건다(여러 번 불러도 훅은 한 번). where = "gui"(화면) 또는 "cli"(명령줄).

    "gui"는 시작 기록(kind="start") 한 줄도 남긴다. 명령줄에서 화면으로 넘어가면(인수 없는 `evolab`) 나중 값이 이긴다.
    """
    with _lock:
        first = _state.where is None
        changed = _state.where != where
        _state.where = where
        if first:
            _state.prev_excepthook = sys.excepthook
            _state.prev_threadhook = threading.excepthook
            _state.prev_unraisable = sys.unraisablehook
            sys.excepthook = _excepthook
            threading.excepthook = _threadhook
            sys.unraisablehook = _unraisablehook
    if where == "gui" and changed:
        detail = json.dumps({"frozen": bool(getattr(sys, "frozen", False)), "os": _os_short(),
                             "python": platform.python_version()}, ensure_ascii=False)
        record("start", f"evolab {__version__}", detail, where="gui")


def _os_short() -> str:
    """시작 기록용 OS 이름(빠른 것만: 파이썬 3.12의 platform.platform()은 윈도우에서 WMI에 물어 느릴 수 있다)."""
    if sys.platform == "win32":
        try:
            v = sys.getwindowsversion()  # type: ignore[attr-defined]
            return f"Windows {v.major}.{v.minor}.{v.build}"
        except Exception:
            return "Windows"
    try:
        u = os.uname()
        return f"{u.sysname} {u.release} {u.machine}"
    except Exception:
        return sys.platform


def uninstall() -> None:
    """훅과 Qt 메시지 처리기를 걸기 전으로 되돌리고 상태를 지운다(테스트용). 표시 파일은 지우지 않는다."""
    with _lock:
        if _state.where is not None:
            if sys.excepthook is _excepthook:
                sys.excepthook = _state.prev_excepthook or sys.__excepthook__
            if threading.excepthook is _threadhook:
                threading.excepthook = _state.prev_threadhook or threading.__excepthook__
            if sys.unraisablehook is _unraisablehook:
                sys.unraisablehook = _state.prev_unraisable or sys.__unraisablehook__
        if _state.qt_installed:
            try:
                from PySide6.QtCore import qInstallMessageHandler

                # 우리 처리기가 아직 걸려 있을 때만 앞의 것으로 되돌린다. 그사이 다른 쪽(pytest-qt의 로그 잡기 등)이 처리기를
                # 바꿨으면 그대로 둔다: 그 쪽이 이미 끝나 버린 처리기를 다시 걸면 이후 Qt 메시지가 죽은 곳으로 간다.
                current = qInstallMessageHandler(_state.prev_qt_handler)
                if current is not _qt_handler:
                    qInstallMessageHandler(current)
            except Exception:
                pass
        bridge = _state.bridge
        _state.__init__()  # type: ignore[misc]
    if bridge is not None:
        try:
            bridge.deleteLater()
        except Exception:
            pass


def _one_line(exc_type: type[BaseException], exc: BaseException | None) -> str:
    text = " ".join(str(exc).split()) if exc is not None else ""
    return f"{exc_type.__name__}: {text}" if text else exc_type.__name__


def _signature(exc_type: type[BaseException], tb: Any) -> str:
    """같은 오류인지 가르는 서명: 예외 종류 + 맨 아래 프레임의 파일:줄."""
    try:
        frames = traceback.extract_tb(tb)
    except Exception:
        frames = []
    last = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "?"
    return f"{exc_type.__module__}.{exc_type.__qualname__}@{last}"


def _handle(kind: str, exc_type: type[BaseException], exc: BaseException | None, tb: Any, header: str = "") -> str:
    """기록하고(같은 서명은 한 실행에 한 번), 화면이면 상자·상태 줄을 정한다. 돌려주는 값: 한 줄 요약."""
    line = _one_line(exc_type, exc)
    sig = _signature(exc_type, tb)
    with _lock:
        count = _state.seen.get(sig, 0)
        _state.seen[sig] = count + 1
        where = _state.where or "gui"
        action: str | None = None
        if _state.on_crash is not None and kind != "unraisable":
            if count == 0:
                if _state.boxes < MAX_BOXES:
                    _state.boxes += 1
                    action = "box"
                else:
                    action = "more"
            else:
                now = _now()
                if now - _state.again_at.get(sig, -1e9) >= AGAIN_REPEAT_S:
                    _state.again_at[sig] = now
                    action = "again"
    if count == 0:
        detail = header + "".join(traceback.format_exception(exc_type, exc, tb))
        record(kind, line, detail, where=where)
    if action is not None:
        _deliver(action, line)
    return line


def _deliver(action: str, line: str) -> None:
    bridge = _state.bridge
    if bridge is None:
        return
    try:
        bridge.crashed.emit(action, line)  # 화면 스레드의 이벤트 루프에서 on_crash를 부른다(큐 연결)
    except Exception:
        pass


def _excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
    if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
        (_state.prev_excepthook or sys.__excepthook__)(exc_type, exc, tb)
        return
    try:
        kind = "uncaught" if threading.current_thread() is threading.main_thread() else "thread"
        line = _handle(kind, exc_type, exc, tb)
    except Exception:
        line = exc_type.__name__
    if _state.where == "cli":
        # 명령줄: 기록 + 한 줄(설계 4.9). 종료 코드 1은 파이썬이 정한다. CI(환경 변수 CI)에서는 기록 파일을 볼 수 없을 수
        # 있으므로 traceback 전체도 stderr에 남긴다.
        if os.environ.get(CI_ENV, "").strip():
            try:
                (_state.prev_excepthook or sys.__excepthook__)(exc_type, exc, tb)
            except Exception:
                pass
        try:
            if sys.stderr is not None:
                print(S.CLI_UNCAUGHT.format(reason=line, path=errors_path()), file=sys.stderr)
        except Exception:
            pass
        return
    try:
        (_state.prev_excepthook or sys.__excepthook__)(exc_type, exc, tb)
    except Exception:
        pass


def _threadhook(args: Any) -> None:
    if args.exc_type is SystemExit:
        (_state.prev_threadhook or threading.__excepthook__)(args)
        return
    try:
        name = getattr(args.thread, "name", "?")
        _handle("thread", args.exc_type, args.exc_value, args.exc_traceback, header=f"thread: {name}\n")
    except Exception:
        pass
    try:
        (_state.prev_threadhook or threading.__excepthook__)(args)
    except Exception:
        pass


def _unraisablehook(args: Any) -> None:
    try:
        exc_type = args.exc_type or Exception
        header = f"{args.err_msg or 'Exception ignored in'}: {args.object!r}\n"
        _handle("unraisable", exc_type, args.exc_value, args.exc_traceback, header=header)
    except Exception:
        pass
    try:
        (_state.prev_unraisable or sys.__unraisablehook__)(args)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 화면 프로세스
# ---------------------------------------------------------------------------
def _flag_path(pid: int) -> Path:
    return paths.log_dir() / f"{FLAG_PREFIX}{pid}{FLAG_SUFFIX}"


def _create_time(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _flag_alive(pid: int, flag: Path) -> bool:
    """표시를 남긴 프로세스가 아직 살아 있는가. 표시 안의 시작 시각이 지금 그 pid의 시작 시각과 다르면(pid 재사용) 죽은 것."""
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
    except Exception:
        return True  # 모르면 지우지 않는다
    try:
        written = float(flag.read_text(encoding="ascii").strip() or "nan")
    except (OSError, ValueError):
        return True
    now = _create_time(pid)
    if now is None or written != written:  # 읽을 수 없거나 빈 표시(옛 형식): pid만 본다
        return True
    return abs(now - written) < 2.0


def _check_flags() -> bool:
    """살아 있지 않은 프로세스가 남긴 running-<pid>.flag가 있으면 지우고 참."""
    found = False
    try:
        flags = list(paths.log_dir().glob(f"{FLAG_PREFIX}*{FLAG_SUFFIX}"))
    except OSError:
        return False
    for flag in flags:
        try:
            pid = int(flag.name[len(FLAG_PREFIX): -len(FLAG_SUFFIX)])
        except ValueError:
            continue
        if pid == os.getpid() or not _flag_alive(pid, flag):
            found = True
            try:
                flag.unlink()
            except OSError:
                pass
    return found


def install_gui(on_crash: Callable[[str, str], None]) -> bool:
    """화면 프로세스 진단(설계 4.9): Qt 메시지 처리기, 실행 중 표시, 잡히지 않은 예외 알림, faults.log 자르기.

    QApplication을 만든 뒤, 창을 만들기 전에 화면 스레드에서 부른다. `install("gui")`를 먼저 부르지 않았으면 부른다.
    돌려주는 값: 지난 실행이 갑자기 끝났는가(살아 있지 않은 프로세스의 표시가 있었나). 두 번째부터는 콜백만 바꾸고 거짓.
    """
    if _state.where != "gui":
        install("gui")
    from PySide6.QtCore import QObject, Qt, Signal, qInstallMessageHandler

    with _lock:
        _state.on_crash = on_crash
        if _state.bridge is not None:
            return False

    class _Bridge(QObject):
        crashed = Signal(str, str)

        def deliver(self, action: str, line: str) -> None:
            callback = _state.on_crash
            if callback is None:
                return
            try:
                callback(action, line)
            except Exception:  # 알리기가 실패해도 다시 훅으로 들어가지 않게
                pass

    bridge = _Bridge()
    bridge.crashed.connect(bridge.deliver, Qt.ConnectionType.QueuedConnection)
    _state.bridge = bridge
    _state.prev_qt_handler = qInstallMessageHandler(_qt_handler)
    _state.qt_installed = True
    _trim_faults()
    previous = _check_flags()
    try:
        flag = _flag_path(os.getpid())
        flag.parent.mkdir(parents=True, exist_ok=True)
        created = _create_time(os.getpid())
        flag.write_text("" if created is None else repr(created), encoding="ascii")
        _state.flag = flag
    except OSError:
        _state.flag = None
    return previous


def _qt_handler(mode: Any, context: Any, message: str) -> None:
    """Qt 메시지: 경고 이상은 기록(같은 글은 60초에 한 번), 모두 원래처럼 표준 오류(또는 앞 처리기)로."""
    try:
        name = getattr(mode, "name", str(mode))
        if name in ("QtWarningMsg", "QtCriticalMsg", "QtFatalMsg"):
            key = (name, str(message))
            now = _now()
            with _lock:
                last = _state.qt_seen.get(key)
                fresh = last is None or now - last >= QT_REPEAT_S
                if fresh:
                    _state.qt_seen[key] = now
            if fresh:
                where = f"{getattr(context, 'file', None) or ''}:{getattr(context, 'line', 0)}"
                func = getattr(context, "function", None) or ""
                detail = f"{name} {where} {func}".strip()
                record("qt", str(message), detail, where=_state.where or "gui")
    except Exception:
        pass
    try:
        prev = _state.prev_qt_handler
        if prev is not None:
            prev(mode, context, message)
        elif sys.stderr is not None:
            try:
                from PySide6.QtCore import qFormatLogMessage

                text = qFormatLogMessage(mode, context, message)
            except Exception:
                text = str(message)
            sys.stderr.write(text + "\n")
    except Exception:
        pass


def mark_clean_exit() -> None:
    """정상 종료: 이 프로세스의 running-<pid>.flag를 지운다(주 창 closeEvent 끝, 시작 실패 처리). 여러 번 불러도 된다."""
    flag = _state.flag or _flag_path(os.getpid())
    try:
        flag.unlink()
    except OSError:
        pass
    _state.flag = None

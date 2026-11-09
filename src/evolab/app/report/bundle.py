"""오류 보고 묶음 만들기(v3 설계 D31~D33, 2.4, 4.10). Qt를 불러오지 않는다(명령줄 `evolab report`, 시작 실패 처리).

묶음은 문제를 알릴 때 사람이 직접 보내는 zip 하나다. 인터넷으로 보내지 않는다. 재현에 필요한 최소만 넣는다:

- `report.txt`(사람이 읽는 한국어: 버전, exe 여부, OS, CPU, 코어, 메모리, 화면, 파이썬, Qt, 데이터 폴더, 만든 까닭, 메모),
  `manifest.json`(형식 번호, 넣은 파일과 바이트)
- 로그 폴더: `errors.jsonl` 최근 200줄, `faults.log`·`stderr.log`·`selftest.log` 끝 256KB
- 설정: `settings.ini`, `paths.ini`
- (고르면) 지금 실행: `run.json`, `events.jsonl` 끝 200줄, `stats.csv` 머리 + 끝 50줄, 마지막 환경 사본, 파일 이름·크기 목록
- (고르면) 지금 열린 환경 글

넣지 않는 것: 체크포인트(.pkl: 크고, 열면 코드가 실행될 수 있는 피클), 챔피언·세대 대표(.npz), 직접 조종 기록(.evoplay),
내보낸 영상, 다른 실행, 화면 그림, 환경 변수. 모든 글 파일에서 홈 폴더 경로·사용자 이름·컴퓨터 이름을 가린다. 합계 5MB 상한
(넘으면 오래된 로그부터 뺀다). 대화 상자의 미리 보기(`collect`)와 쓰기(`write_report`)가 같은 목록 함수를 쓴다.

줄 끝은 바이트 그대로다(윈도우에서도): 원본은 모두 바이너리로 읽고, 줄은 `\\n`으로 세어 자르며, zip에는 바이트를 넣는다.
글 모드(`open(..., "w")`, `Path.write_text`·`read_text`)는 쓰지 않는다: 윈도우의 글 모드 쓰기는 `\\n`을 `\\r\\n`으로 바꿔
`stats.csv`의 `\\r\\n`이 `\\r\\r\\n`이 되고, 읽기는 어디서나 `\\r\\n`을 `\\n`으로 바꾼다. 우리가 만드는 report.txt는 `\\r\\n`
(메모 줄도), files.txt·manifest.json은 `\\n`이다.
"""

from __future__ import annotations

import errno
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from evolab import __version__, paths
from evolab.app import strings as S

REPORT_FORMAT: Final = "evolab-report"
REPORT_VERSION: Final = 1
MANIFEST_NAME: Final = "manifest.json"

MAX_TOTAL_BYTES: Final = 5 * 1024 * 1024
LOG_TAIL_BYTES: Final = 256 * 1024
ERRORS_LINES: Final = 200
EVENTS_LINES: Final = 200
STATS_LINES: Final = 50
FILES_MAX_LINES: Final = 5000
TEXT_MAX_BYTES: Final = 2 * 1024 * 1024
"""run.json·설정 파일·환경 사본 한 개의 상한(이보다 크면 넣지 않는다: 잘린 JSON은 쓸모가 없다)."""

LOG_KEYS: Final = ("errors", "faults", "stderr", "selftest")
GROUP_KEYS: Final = ("report", "errors", "faults", "stderr", "selftest", "settings", "run", "env")
"""대화 상자 체크 칸 순서 = strings.REP_ITEM_NAMES의 키."""
EXCLUDED_SUFFIXES: Final = (".pkl", ".npz", ".evoplay")
"""묶음에 절대 넣지 않는 확장자(테스트와 자체 점검이 확인)."""


@dataclass(frozen=True)
class BundleOptions:
    reason: str = "menu"
    """만든 까닭: strings.REP_REASONS의 키(menu, error, crash, previous, startup, cli, tour)."""
    note: str = ""
    run_dir: Path | None = None
    env_text: str | None = None
    include_run: bool = True
    include_env: bool = True


@dataclass(frozen=True)
class ReportItem:
    key: str
    """묶음(체크 칸) 키: GROUP_KEYS."""
    arcname: str
    size: int
    default_on: bool


@dataclass
class _Entry:
    key: str
    arcname: str
    data: bytes
    mtime: float = 0.0
    default_on: bool = True


# ---------------------------------------------------------------------------
# 가림
# ---------------------------------------------------------------------------
def identity() -> tuple[list[str], list[str], list[str]]:
    """가릴 값: (홈 폴더 경로들, 사용자 이름들, 컴퓨터 이름들). 환경 변수와 OS에서 읽는다."""
    homes: list[str] = []
    try:
        homes.append(str(Path.home()))
    except (RuntimeError, OSError, KeyError):
        pass
    for name in ("USERPROFILE", "HOME"):
        value = os.environ.get(name, "")
        if value:
            homes.append(value)
    users = [os.environ.get(name, "") for name in ("USERNAME", "USER", "LOGNAME")]
    try:
        users.append(os.getlogin())
    except OSError:
        pass
    hosts = [os.environ.get("COMPUTERNAME", ""), os.environ.get("HOSTNAME", "")]
    try:
        hosts.append(platform.node())
    except OSError:
        pass
    return homes, users, hosts


def _variants(text: str) -> set[str]:
    """한 값의 여러 적힌 모양: 그대로, 구분자를 바꾼 것, JSON 안의 이스케이프 표기(역슬래시 두 개, \\uXXXX).

    역슬래시가 든 모양은 JSON 이스케이프를 한 번 더 한 꼴도 넣는다: str(OSError)는 경로를 repr로 써서 역슬래시가 이미 두 배이고,
    그 글을 errors.jsonl에 JSON으로 쓰면 네 배가 된다."""
    out = {text}
    if "\\" in text or "/" in text:
        out.add(text.replace("\\", "/"))
        out.add(text.replace("/", "\\"))
    for value in list(out):
        once = {json.dumps(value, ensure_ascii=True)[1:-1], json.dumps(value, ensure_ascii=False)[1:-1]}
        out |= once
        for escaped in once:
            if "\\" in escaped:
                out.add(json.dumps(escaped, ensure_ascii=True)[1:-1])
                out.add(json.dumps(escaped, ensure_ascii=False)[1:-1])
    return {v for v in out if v}


_ASCII_WORD: Final = "A-Za-z0-9"
"""이름을 찾는 낱말 경계: 영문·숫자 사이에서는 가리지 않는다(「hong」은 「hongkong」 안에서 그대로). 밑줄은 경계로 본다:
「hong_자료」, 실행 폴더 「2026-11-09_1200_홍길동의 언덕_CMA-ES」 같은 이름의 사용자 이름도 가린다."""


def anonymize(text: str, *, homes: Sequence[str] | None = None, users: Sequence[str] | None = None,
              hosts: Sequence[str] | None = None) -> str:
    """홈 폴더 경로 → 「~」, 사용자 이름 → REP_USER_MASK, 컴퓨터 이름 → REP_HOST_MASK (설계 2.4).

    홈 경로는 역슬래시·빗금 두 모양과 JSON 이스케이프 표기(두 번 한 것까지)를 대소문자 무시로 찾는다. 이름은 2자 이상만, 영문·숫자
    낱말 경계에서 찾는다(「hong」은 「hongkong」 안에서 가리지 않고, 「hong_x」·「_홍길동의」에서는 가림). 한글 이름은 조사가 붙어도
    가린다. 값을 주지 않으면 이 PC의 값(`identity()`)을 쓴다.
    """
    if homes is None or users is None or hosts is None:
        my_homes, my_users, my_hosts = identity()
        homes = my_homes if homes is None else homes
        users = my_users if users is None else users
        hosts = my_hosts if hosts is None else hosts
    home_forms: set[str] = set()
    for home in homes:
        home = str(home).strip().rstrip("\\/")
        if len(home) < 3 or re.fullmatch(r"[A-Za-z]:", home):
            continue
        home_forms |= _variants(home)
    if home_forms:
        alts = "|".join(re.escape(v) for v in sorted(home_forms, key=len, reverse=True))
        text = re.sub(f"(?:{alts})(?!\\w)", "~", text, flags=re.IGNORECASE)
    masks: dict[str, str] = {}
    for values, mask in ((hosts, S.REP_HOST_MASK), (users, S.REP_USER_MASK)):  # 사용자가 이김(나중에 덮어씀)
        for value in values:
            value = str(value).strip()
            if len(value) < 2:
                continue
            for form in _variants(value):
                masks[form.casefold()] = mask
    if masks:
        alts = "|".join(re.escape(v) for v in sorted(masks, key=len, reverse=True))
        pattern = re.compile(f"(?<![{_ASCII_WORD}])(?:{alts})(?![{_ASCII_WORD}])", flags=re.IGNORECASE)
        text = pattern.sub(lambda m: masks.get(m.group(0).casefold(), S.REP_USER_MASK), text)
    return text


# ---------------------------------------------------------------------------
# 파일 조각 읽기
# ---------------------------------------------------------------------------
def _tail_bytes(path: Path, limit: int) -> bytes:
    """파일 끝 limit 바이트. 중간에서 잘랐으면 첫 줄 조각은 버린다(여러 바이트 글자가 깨지지 않게)."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size <= limit:
            f.seek(0)
            return f.read()
        f.seek(size - limit)
        data = f.read()
    cut = data.find(b"\n")
    return data[cut + 1:] if 0 <= cut < len(data) - 1 else data


def _split_lines(data: bytes) -> list[bytes]:
    """`\\n`마다 나눈 줄(줄 끝 바이트 그대로: `\\r\\n`은 `\\r\\n`, 줄 안의 홀로 선 `\\r`은 줄 안에). 마지막 조각은 `\\n`이 없어도
    한 줄이다. `bytes.splitlines`는 홀로 선 `\\r`도 줄 끝으로 봐서, 진행 표시(`\\r`)나 윈도우 글 모드로 한 번 더 바뀐 `\\r\\r\\n`이
    든 파일의 줄을 둘로 세고 꼬리가 `\\r\\n` 조각으로 시작할 수 있다."""
    parts = data.split(b"\n")
    lines = [part + b"\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _tail_lines(path: Path, n: int, max_bytes: int = 8 * 1024 * 1024) -> tuple[list[bytes], bool]:
    """마지막 n줄(`\\n`으로 센 줄, 바이트 그대로)과, 파일 처음까지 읽었는지. 뒤에서부터 덩어리로 읽어 큰 파일도 빠르다."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        pos = size
        data = b""
        while pos > 0 and data.count(b"\n") <= n and size - pos < max_bytes:
            step = min(64 * 1024, pos)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
    lines = _split_lines(data)
    start = pos == 0
    if not start and lines:
        lines = lines[1:]  # 잘린 첫 줄
    if len(lines) > n:
        lines = lines[-n:]
        start = False
    return lines, start


def _text(data: bytes) -> bytes:
    """가린 바이트. UTF-8로 풀고 다시 묶을 뿐 줄 끝은 건드리지 않는다(BOM도 그대로)."""
    return anonymize(data.decode("utf-8", "replace")).encode("utf-8")


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# ---------------------------------------------------------------------------
# 데이터 폴더(Qt 없이)
# ---------------------------------------------------------------------------
def data_dir() -> Path:
    """데이터 폴더. 화면처럼 Qt가 이미 올라온 프로세스는 `paths.data_dir()`, 명령줄은 Qt 없이 같은 규칙으로 찾는다."""
    if os.environ.get(paths.DATA_DIR_ENV, "").strip() or "PySide6.QtCore" in sys.modules:
        return paths.data_dir()
    custom = paths.custom_data_dir()
    if custom is not None:
        return custom
    from evolab import winapp

    docs = winapp.known_folder("documents") or Path.home() / "Documents"
    return docs / S.DATA_DIR_NAME


def _data_dir_changed() -> bool:
    return bool(os.environ.get(paths.DATA_DIR_ENV, "").strip()) or paths.custom_data_dir() is not None


def desktop_dir() -> Path | None:
    """바탕 화면 폴더(OneDrive로 옮긴 것 포함). 없으면 None."""
    from evolab import winapp

    folder = winapp.known_folder("desktop")
    if folder is None:
        core = sys.modules.get("PySide6.QtCore")
        if core is not None:
            try:
                loc = core.QStandardPaths.writableLocation(core.QStandardPaths.StandardLocation.DesktopLocation)
                folder = Path(loc) if loc else None
            except Exception:
                folder = None
    if folder is None:
        folder = Path.home() / "Desktop"
    return folder if folder.is_dir() else None


def default_bundle_path(now: datetime | None = None) -> Path:
    """기본 저장 위치: 바탕 화면(없으면 데이터 폴더)/REP_FILE_NAME. 같은 이름이 있으면 뒤에 _2, _3…"""
    now = now or datetime.now()
    folder = desktop_dir() or data_dir()
    name = S.REP_FILE_NAME.format(stamp=now.strftime("%Y%m%d_%H%M"))
    path = folder / name
    i = 2
    while path.exists():
        path = folder / f"{Path(name).stem}_{i}{Path(name).suffix}"
        i += 1
    return path


# ---------------------------------------------------------------------------
# 내용 만들기
# ---------------------------------------------------------------------------
def _report_text(opts: BundleOptions, now: datetime) -> bytes:
    from evolab import diagnostics

    info = diagnostics.platform_info()
    values = {
        "created": now.strftime("%Y-%m-%d %H:%M"),
        "version": info["version"],
        "frozen": S.REP_YES if info["frozen"] == "1" else S.REP_NO,
        "os": info["os"],
        "cpu": info["cpu"],
        "cores": info["cores"],
        "memory": info["memory"],
        "screen": info["screen"],
        "python": info["python"],
        "qt": info["qt"],
        "data_dir": str(data_dir()),
        "custom_data_dir": S.REP_YES if _data_dir_changed() else S.REP_NO,
        "reason": S.REP_REASONS.get(opts.reason, opts.reason),
    }
    lines = [S.REP_TXT_TITLE, "=" * 40]
    for key, label in S.REP_TXT_FIELDS.items():
        if key == "note":  # 대화 상자 메모는 \n 줄바꿈: report.txt의 다른 줄처럼 \r\n으로
            note = "\r\n".join(opts.note.strip().splitlines())
            lines.append(f"{label}:" + (f"\r\n{note}" if note else " —"))
        else:
            lines.append(f"{label}: {values.get(key, '') or chr(0x2014)}")
    return anonymize("\r\n".join(lines) + "\r\n").encode("utf-8")


def _log_entry(key: str, name: str, *, lines: int | None = None) -> _Entry | None:
    path = paths.log_dir() / name
    if not path.is_file():
        return None
    try:
        data = b"".join(_tail_lines(path, lines)[0]) if lines is not None else _tail_bytes(path, LOG_TAIL_BYTES)
    except OSError:
        return None
    if not data.strip():  # 빈 기록(예: 오류가 한 번도 없었던 faults.log)은 넣지 않는다
        return None
    return _Entry(key, f"logs/{name}", _text(data), mtime=_mtime(path))


def _small_file(key: str, path: Path, arcname: str, default_on: bool = True) -> _Entry | None:
    try:
        if not path.is_file() or path.stat().st_size > TEXT_MAX_BYTES:
            return None
        return _Entry(key, arcname, _text(path.read_bytes()), mtime=_mtime(path), default_on=default_on)
    except OSError:
        return None


def _run_entries(run_dir: Path, default_on: bool) -> list[_Entry]:
    from evolab.runs.store import ENVS_DIR, EVENTS_FILE, RUN_INFO_FILE, STATS_FILE

    out: list[_Entry] = []
    entry = _small_file("run", run_dir / RUN_INFO_FILE, "run/run.json", default_on)
    if entry is not None:
        out.append(entry)
    events = run_dir / EVENTS_FILE
    if events.is_file():
        try:
            out.append(_Entry("run", "run/events_tail.jsonl", _text(b"".join(_tail_lines(events, EVENTS_LINES)[0])),
                              default_on=default_on))
        except OSError:
            pass
    stats = run_dir / STATS_FILE
    if stats.is_file():
        try:
            tail, whole = _tail_lines(stats, STATS_LINES + 1)
            if whole:
                data = b"".join(tail)
            else:
                with open(stats, "rb") as f:
                    header = f.readline(256 * 1024)
                data = header + b"".join(tail[-STATS_LINES:])
            out.append(_Entry("run", "run/stats_tail.csv", _text(data), default_on=default_on))
        except OSError:
            pass
    envs = sorted((p for p in (run_dir / ENVS_DIR).glob("v*.evoenv") if p.is_file()),
                  key=lambda p: (len(p.stem), p.stem))
    if envs:
        entry = _small_file("run", envs[-1], f"run/envs/{envs[-1].name}", default_on)
        if entry is not None:
            out.append(entry)
    rows: list[str] = []
    for folder, dirs, files in os.walk(run_dir):
        dirs.sort()
        for name in sorted(files):
            full = Path(folder) / name
            try:
                size = full.stat().st_size
            except OSError:
                continue
            rows.append(f"{full.relative_to(run_dir).as_posix()}\t{size}")
    extra = len(rows) - FILES_MAX_LINES
    if extra > 0:
        rows = rows[:FILES_MAX_LINES] + [f"... (+{extra})"]
    out.append(_Entry("run", "run/files.txt", _text(("\n".join(rows) + "\n").encode("utf-8")), default_on=default_on))
    return out


def _entries(opts: BundleOptions, now: datetime | None = None) -> list[_Entry]:
    now = now or datetime.now()
    out: list[_Entry] = [_Entry("report", "report.txt", _report_text(opts, now))]
    from evolab.diagnostics import ERRORS_FILE, FAULTS_FILE

    for key, name, lines in (("errors", ERRORS_FILE, ERRORS_LINES), ("faults", FAULTS_FILE, None),
                             ("stderr", "stderr.log", None), ("selftest", "selftest.log", None)):
        entry = _log_entry(key, name, lines=lines)
        if entry is not None:
            out.append(entry)
    state = paths.user_state_dir()
    for name in ("settings.ini", paths.PATHS_FILE):
        entry = _small_file("settings", state / name, f"settings/{name}")
        if entry is not None:
            out.append(entry)
    if opts.run_dir is not None and Path(opts.run_dir).is_dir():
        out.extend(_run_entries(Path(opts.run_dir), opts.include_run))
    if opts.env_text is not None and len(opts.env_text.encode("utf-8")) <= TEXT_MAX_BYTES:
        out.append(_Entry("env", "env/current.evoenv", _text(opts.env_text.encode("utf-8")),
                          default_on=opts.include_env))
    return out


def _cap(entries: list[_Entry], limit: int = MAX_TOTAL_BYTES) -> list[_Entry]:
    """합계가 limit를 넘으면 오래된 로그부터 빼고, 그래도 넘으면 report.txt 말고 큰 것부터 뺀다."""
    total = sum(len(e.data) for e in entries)
    if total <= limit:
        return entries
    dropped: set[int] = set()
    for e in sorted((e for e in entries if e.key in LOG_KEYS), key=lambda e: e.mtime):
        if total <= limit:
            break
        dropped.add(id(e))
        total -= len(e.data)
    for e in sorted((e for e in entries if e.key != "report" and id(e) not in dropped), key=lambda e: -len(e.data)):
        if total <= limit:
            break
        dropped.add(id(e))
        total -= len(e.data)
    return [e for e in entries if id(e) not in dropped]


def collect(opts: BundleOptions) -> list[ReportItem]:
    """넣을 수 있는 파일 목록(대화 상자 미리 보기와 `write_report`가 같은 것을 씀). 크기는 가림·자르기를 한 뒤의 바이트."""
    return [ReportItem(e.key, e.arcname, len(e.data), e.default_on) for e in _cap(_entries(opts))]


def write_report(out: Path, opts: BundleOptions, items: Sequence[ReportItem] | None = None) -> list[tuple[str, int]]:
    """묶음 zip을 쓴다. items가 없으면 `collect(opts)`에서 기본으로 켠 것. 돌려주는 값: 넣은 (묶음 안 이름, 바이트)
    (manifest.json 제외). 임시 이름으로 쓰고 끝나면 바꾼다. 실패하면 OSError.
    """
    now = datetime.now()
    entries = _entries(opts, now)
    if items is None:
        chosen = [e for e in _cap(entries) if e.default_on]
    else:
        wanted = {i.arcname for i in items}
        chosen = _cap([e for e in entries if e.arcname in wanted])
    written = [(e.arcname, len(e.data)) for e in chosen]
    manifest = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "reason": opts.reason,
        "created": now.isoformat(timespec="seconds"),
        "app_version": __version__,
        "items": [{"name": name, "bytes": size} for name, size in written],
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    stamp = now.timetuple()[:6]
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for e in chosen:
                z.writestr(zipfile.ZipInfo(e.arcname, stamp), e.data, compress_type=zipfile.ZIP_DEFLATED)
            z.writestr(zipfile.ZipInfo(MANIFEST_NAME, stamp),
                       json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"),
                       compress_type=zipfile.ZIP_DEFLATED)
        os.replace(tmp, out)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return written


# ---------------------------------------------------------------------------
# 도우미: 크기 글, 오류 까닭, 폴더 열기
# ---------------------------------------------------------------------------
def human_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{max(1, round(n / 1024))}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def reason_text(exc: BaseException) -> str:
    """사용자에게 보일 짧은 까닭(Qt 없이). 날것 예외 글 대신 strings.py 문구, 모르는 것은 예외 종류 이름만."""
    if isinstance(exc, PermissionError):
        return S.ERR_PERMISSION
    if isinstance(exc, FileNotFoundError):
        return S.ERR_NOT_FOUND.format(path=exc.filename or "")
    if isinstance(exc, OSError) and (exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) == 112):
        return S.ERR_DISK_FULL
    if isinstance(exc, MemoryError):
        return S.ERR_MEMORY
    if isinstance(exc, OSError) and exc.strerror:
        return str(exc.strerror)
    return type(exc).__name__


def reveal(path: Path) -> None:
    """탐색기(파인더, 파일 관리자)로 그 파일이 있는 폴더를 연다. 실패는 조용히 넘어간다."""
    path = Path(path)
    try:
        if sys.platform == "win32":
            subprocess.Popen(f'explorer /select,"{path}"')  # noqa: S602  explorer는 목록 인수를 따옴표째 못 읽는다
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path.parent)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def _err(message: str) -> None:
    if sys.stderr is not None:
        print(message, file=sys.stderr)


# ---------------------------------------------------------------------------
# 명령줄, 시작 실패, 자체 점검
# ---------------------------------------------------------------------------
def report_command(out: Path | None, run: Path | None, open_folder: bool = False) -> int:
    """`evolab report [--out zip] [--run 폴더] [--open]` (Qt 없이). 종료 코드: 0 만듦, 1 쓰기 실패, 2 실행 폴더 없음."""
    if run is not None and not Path(run).is_dir():
        _err(S.REP_FAILED.format(reason=S.ERR_NOT_FOUND.format(path=run)))
        return 2
    try:
        target = Path(out) if out is not None else default_bundle_path()
        written = write_report(target, BundleOptions(reason="cli", run_dir=run))
        size = target.stat().st_size
    except OSError as exc:
        _err(S.REP_FAILED.format(reason=reason_text(exc)))
        return 1
    if sys.stdout is not None:
        print(S.CLI_REPORT_DONE.format(path=target, count=len(written), size=human_size(size)))
    if open_folder:
        reveal(target)
    return 0


def report_startup_failure(exc: BaseException) -> int:
    """창을 못 띄운 시작 실패(설계 D32, 4.9): 기록, 바탕 화면에 묶음(reason="startup"), Qt 없는 상자(윈도우가 아니면 표준 오류).
    돌려주는 값은 종료 코드 1. `run_gui()`가 `MainWindow()`·`show()`의 예외를 잡아 부른다.
    """
    from evolab import diagnostics, winapp

    try:
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        diagnostics.record("uncaught", f"{type(exc).__name__}: {' '.join(str(exc).split())}", detail, where="gui")
    except Exception:
        pass
    reason = reason_text(exc)
    try:
        path = default_bundle_path()
        write_report(path, BundleOptions(reason="startup"))
        text = S.DIAG_STARTUP_TEXT.format(reason=reason, path=path)
    except Exception as err:  # 묶음도 못 만들면 그 까닭만
        text = S.DIAG_STARTUP_NO_BUNDLE.format(reason=reason, error=reason_text(err))
    diagnostics.mark_clean_exit()
    winapp.message_box(S.DIAG_STARTUP_TITLE, text)
    return 1


def self_check(work_dir: Path) -> str:
    """자체 점검 「기록 폴더 쓰기와 오류 보고 묶음」의 몸통(Qt 없음): 로그 폴더에 쓰고 지워 보고, work_dir에 묶음을 만들어
    report.txt·manifest.json이 있고 체크포인트·유전자·입력 기록이 없고 목록이 맞는지 본다. 돌려주는 값: 자세히 한 줄(ASCII).
    문제가 있으면 RuntimeError(문제 목록, ASCII). 자체 점검은 EVOLAB_STATE_DIR·EVOLAB_DATA_DIR를 임시 폴더로 두고 부른다.
    """
    t0 = time.perf_counter()
    folder = paths.log_dir()
    folder.mkdir(parents=True, exist_ok=True)
    probe = folder / f".write-check-{os.getpid()}.tmp"
    probe.write_bytes(b"ok")
    probe.unlink()
    out = Path(work_dir) / "report-check.zip"
    written = write_report(out, BundleOptions(reason="cli"))
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        manifest = json.loads(z.read(MANIFEST_NAME).decode("utf-8")) if MANIFEST_NAME in names else {}
    problems = []
    if "report.txt" not in names:
        problems.append("no report.txt")
    if manifest.get("format") != REPORT_FORMAT:
        problems.append("bad manifest")
    if any(n.lower().endswith(EXCLUDED_SUFFIXES) for n in names):
        problems.append("excluded file in bundle")
    if set(names) != {n for n, _ in written} | {MANIFEST_NAME}:
        problems.append("file list differs")
    if problems:
        raise RuntimeError(", ".join(problems))
    return f"files={len(names)} bytes={out.stat().st_size} {time.perf_counter() - t0:.2f}s"

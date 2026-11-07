"""실행 폴더 저장소 (명세: 저장, 기록, 내보내기).

실행 하나가 폴더 하나다. 그 폴더만 있으면 이어하기, 다시 보기, 재현이 된다.

```text
<실행 폴더>\\
  run.json          학습 설정, 시드, 버전, 환경 버전 목록, 보호한 체크포인트 (UTF-8, 들여쓰기 2)
  envs\\            환경 버전별 사본 v000.evoenv, v001.evoenv … (내용 해시는 run.json에)
  stats.csv         세대별 통계 (BOM 붙은 UTF-8: 한국어 윈도우의 엑셀에서 바로 열린다)
  events.jsonl      멈춤, 이어하기, 환경 수정 같은 사건 (한 줄에 JSON 객체 하나)
  checkpoints\\     gen_000010.pkl … 학습 상태 (피클 프로토콜 5)
  champions\\       gen_000010_representative.npz, gen_000010_champion.npz …
                    (세대 대표에는 v1.2부터 그래프용 관찰 기록 obs_* 키도 함께: 형식은 runs/observe.py)
  exports\\         내보낸 영상, CSV
```

윈도우에서 생기는 일에 맞춘 점:

- 반쯤 쓴 파일이 남으면 안 되는 파일(run.json, 체크포인트, 챔피언, 환경 사본, stats.csv 고쳐 쓰기)은 같은
  폴더의 임시 파일에 다 쓰고 디스크에 내린 뒤 os.replace로 바꾼다. 윈도우에서는 다른 프로그램(백신, OneDrive
  동기화, 화면 프로세스)이 그 순간 파일을 열고 있으면 바꾸기가 PermissionError로 실패하므로 잠깐씩 다시 시도한다.
- 사용자가 학습 중에 stats.csv를 엑셀로 열면 엑셀이 파일을 잠가 덧붙이기가 실패한다. 이때 학습을 멈추지 않고
  통계 줄을 stats.csv.pending에 따로 쌓아 두었다가, 잠금이 풀린 뒤 첫 덧붙이기 때 stats.csv에 옮긴다.
  read_stats()는 두 파일을 합쳐 읽는다. events.jsonl도 같은 방식이다. 옮긴 직후 끊기거나 옆 파일을 지우지
  못해도 같은 줄이 두 번 들어가지 않는다(RunStore._append_line 참고).
- 전원이 나가면 덧붙이던 마지막 줄이 잘리거나 NUL 바이트로 채워질 수 있다. 읽을 때는 줄바꿈으로 끝나지 않은
  마지막 줄을 버리고, 처음 덧붙이기 전에(그리고 덧붙이기가 실패한 뒤에) 그 꼬리를 잘라 낸다. 그래서 통계 칸과
  열 이름에는 줄바꿈을 받지 않는다(한 줄 = 한 세대). 덧붙이기는 매번 디스크에 내리지 않고 체크포인트를 쓰기 직전에
  내린다: 체크포인트가 그 세대까지의 통계 줄보다 앞서 남지 않는다.
- 폴더 이름에서 윈도우가 금지하는 글자(<>:"/\\|?*, 제어 문자), 끝의 점·공백, 예약 이름(CON, NUL, COM1 …)을
  고친다. 한글과 공백은 그대로 쓴다. 경로 길이(MAX_PATH 260)를 넘지 않게 폴더 이름은 80글자로 자른다.

세대 번호: 이 모듈은 받은 번호를 그대로 쓴다. truncate_after(g)는 번호가 g보다 큰 통계 줄(generation 열),
챔피언 파일, 체크포인트를 지운다.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import itertools
import json
import operator
import os
import pickle
import re
import shutil
import sys
import time
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Final, Literal

import numpy as np

from evolab.app import strings as S
from evolab.runs.observe import OBS_PREFIX, read_representative

STORE_FORMAT: Final = 1
"""run.json의 store_format. 이보다 큰 값은 새 버전 프로그램이 만든 실행 기록이다."""

RUN_INFO_FILE: Final = "run.json"
ENVS_DIR: Final = "envs"
STATS_FILE: Final = "stats.csv"
EVENTS_FILE: Final = "events.jsonl"
CHECKPOINTS_DIR: Final = "checkpoints"
CHAMPIONS_DIR: Final = "champions"
EXPORTS_DIR: Final = "exports"
PENDING_SUFFIX: Final = ".pending"
"""잠긴 파일 대신 줄을 쌓아 두는 옆 파일의 꼬리 (stats.csv.pending). 엑셀과 연결되지 않는 확장자다."""

PICKLE_PROTOCOL: Final = 5
MAX_FOLDER_NAME: Final = 80
CHAMPION_KINDS: Final = ("representative", "champion")
ChampionKind = Literal["representative", "champion"]

_BOM: Final = b"\xef\xbb\xbf"
_CSV_LINE_END: Final = "\r\n"  # 엑셀 기본 형식

# 윈도우에서 다른 프로그램이 잠깐 파일을 잡고 있을 때(백신, OneDrive, 다른 프로세스의 읽기) 다시 시도하는 간격(초).
_REPLACE_RETRY_DELAYS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
# 덧붙이기는 짧게만 기다린다. 엑셀처럼 오래 잡고 있으면 옆 파일에 쌓는다.
_APPEND_RETRY_DELAYS: tuple[float, ...] = (0.01, 0.03)
# 이보다 오래된 체크포인트 임시 파일은 죽은 프로세스가 남긴 것으로 보고 지운다(초).
_STALE_TMP_SECONDS: Final = 600.0

_ILLEGAL_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_RESERVED_NAMES = re.compile(r"(con|prn|aux|nul|conin\$|conout\$|com[0-9¹²³]|lpt[0-9¹²³])",
                             re.IGNORECASE)
_VERSION_ID = re.compile(r"v(\d{3,})")
_CHECKPOINT_NAME = re.compile(r"gen_(\d{6,})\.pkl")
_CHECKPOINT_TMP = re.compile(r"\.gen_\d+\.pkl\.\d+\.\d+\.tmp")
_LINE_BREAK = re.compile(r"[\r\n]")
_STORE_TMP = re.compile(r"\..+\.\d+\.\d+\.tmp")
_SAVEZ_RESERVED: Final = frozenset({"file", "allow_pickle"})
_EVENT_RESERVED: Final = frozenset({"kind", "generation", "time"})

_tmp_counter = itertools.count()


class RunStoreError(ValueError):
    """실행 폴더를 읽거나 고칠 수 없다. 메시지는 한국어로 사용자에게 그대로 보여 줄 수 있다."""


@dataclass(frozen=True)
class CheckpointPolicy:
    """체크포인트 정리 규칙: 최근 keep_last개, keep_every의 배수 세대, protect()로 표시한 세대를 남긴다.

    keep_every가 0이면 배수 규칙을 쓰지 않는다. 방금 저장한 체크포인트는 언제나 남는다.
    """

    keep_last: int = 5
    keep_every: int = 100

    def __post_init__(self) -> None:
        ok = all(type(v) is int for v in (self.keep_last, self.keep_every))
        if not ok or self.keep_last < 1 or self.keep_every < 0:
            raise ValueError(S.RUNS_BAD_POLICY.format(keep_last=self.keep_last, keep_every=self.keep_every))


def sanitize_folder_name(name: str) -> str:
    """윈도우에서 쓸 수 있는 폴더 이름으로 고친다. 한글과 공백은 그대로 둔다.

    - 금지 글자 <>:"/\\|?* 와 제어 문자는 '_'로 바꾼다.
    - 앞뒤 공백, 끝의 점과 공백을 지운다(윈도우는 끝의 점·공백을 말없이 떼어 다른 이름이 된다).
    - 예약 이름(CON, PRN, AUX, NUL, COM0~9, LPT0~9 …, 확장자가 붙어도)이면 앞에 '_'를 붙인다.
    - 유니코드는 NFC로 맞추고 MAX_FOLDER_NAME 글자로 자른다. 비면 기본 이름을 쓴다.
    """
    s = unicodedata.normalize("NFC", str(name))
    s = _ILLEGAL_NAME_CHARS.sub("_", s).strip()
    s = s[:MAX_FOLDER_NAME].rstrip(" .")
    if not s:
        s = S.RUNS_DEFAULT_FOLDER_NAME
    if _RESERVED_NAMES.fullmatch(s.split(".", 1)[0].rstrip(" ")):
        s = ("_" + s)[:MAX_FOLDER_NAME].rstrip(" .")  # '_'를 붙여도 MAX_FOLDER_NAME을 넘지 않게
    return s


# ---------------------------------------------------------------------------
# 파일 도우미
# ---------------------------------------------------------------------------


def _check_generation(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(S.RUNS_BAD_GENERATION.format(value=value))
    try:
        g = operator.index(value)
    except TypeError:
        raise ValueError(S.RUNS_BAD_GENERATION.format(value=value)) from None
    if g < 0:
        raise ValueError(S.RUNS_BAD_GENERATION.format(value=value))
    return g


def _check_kind(kind: str) -> str:
    if kind not in CHAMPION_KINDS:
        raise ValueError(f"champion kind must be one of {CHAMPION_KINDS}, got {kind!r}")
    return kind


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        try:
            return sorted(o)  # 집합 순서는 PYTHONHASHSEED마다 달라진다: 같은 내용이면 같은 파일이 되게
        except TypeError:
            return list(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default) + "\n").encode("utf-8")


def _sleep(delay: float) -> None:
    time.sleep(delay)


def _replace(src: Path, dst: Path) -> None:
    """os.replace. 윈도우에서 다른 프로그램이 잠깐 잡고 있어 PermissionError가 나면 조금씩 기다렸다 다시 한다."""
    for delay in (*_REPLACE_RETRY_DELAYS, None):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if delay is None:
                raise
            _sleep(delay)


def _remove(path: Path) -> None:
    """파일을 지운다. 없으면 그만. 잠겨 있으면 잠깐씩 다시 시도하고, 끝내 안 되면 PermissionError."""
    for delay in (*_REPLACE_RETRY_DELAYS, None):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if delay is None:
                raise
            _sleep(delay)


def _fsync_dir(path: Path) -> None:
    """이름 바꾸기를 디스크에 내린다(POSIX만; 윈도우는 폴더를 열 수 없고 NTFS 메타데이터 기록이 대신한다)."""
    if sys.platform == "win32":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(path: Path, write: Callable[[IO[bytes]], object]) -> None:
    """같은 폴더의 임시 파일에 다 쓰고, 디스크에 내린 뒤 바꾼다. 실패하면 임시 파일을 지우고 원래 파일은 그대로."""
    while True:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{next(_tmp_counter)}.tmp")
        try:
            out = open(tmp, "xb")
            break
        except FileExistsError:
            continue  # 같은 pid였던 죽은 프로세스가 남긴 임시 파일(윈도우는 pid를 금방 다시 쓴다): 다음 번호로
    try:
        with out as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        try:
            _replace(tmp, path)
        except PermissionError as exc:
            raise RunStoreError(S.RUNS_FILE_BUSY.format(path=path)) from exc
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


def _sync_file(path: Path) -> None:
    """덧붙여 온 파일을 디스크에 내린다(있을 때만, 할 수 없으면 그만). 윈도우의 fsync는 쓰기 핸들이 필요하다."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0))
    except OSError:
        return  # 없음, 또는 엑셀이 잠금: 그 줄들은 옆 파일에 있고 옆 파일도 따로 내린다
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _tail_bytes(path: Path, n: int) -> bytes:
    """파일의 마지막 n바이트(짧으면 전부, 없으면 빈 바이트)."""
    try:
        with open(path, "rb") as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - n))
            return f.read()
    except FileNotFoundError:
        return b""


def _overlap(main_tail: bytes, side: bytes) -> int:
    """옆 파일 내용(side, 완전한 줄들)의 앞부분 가운데 본 파일 끝과 같은 가장 긴 것의 길이(줄 경계). 없으면 0.

    통계 줄은 세대 번호가, 사건 줄은 세대와 시각이 들어 있어 서로 다르므로 겹침은 이미 옮긴 줄뿐이다.
    """
    view = memoryview(side)
    j = len(side)
    while j > 0:
        if main_tail.endswith(view[:j]):
            return j
        j = side.rfind(b"\n", 0, j - 1) + 1
    return 0


def _remove_stale_tmp(folder: Path, pattern: re.Pattern[str]) -> None:
    """죽은 프로세스가 남긴 임시 파일을 지운다. 지금 쓰는 중일 수 있는 새 파일은 두려고 오래된 것만."""
    now = time.time()
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return
    for entry in entries:
        if pattern.fullmatch(entry.name):
            try:
                if now - entry.stat().st_mtime > _STALE_TMP_SECONDS:
                    os.unlink(entry.path)
            except OSError:
                pass


def _complete_lines(data: bytes) -> bytes:
    """줄바꿈으로 끝나는 부분까지만 (끝의 NUL과 잘린 마지막 줄은 버린다)."""
    stripped = data.rstrip(b"\x00")
    if stripped.endswith(b"\n"):
        return stripped
    return stripped[: stripped.rfind(b"\n") + 1]


def _log_bytes(path: Path) -> tuple[bytes, bytes, bool]:
    """(본 파일의 완전한 줄들, 옆 파일(…pending)에서 아직 옮기지 않은 줄들, 옆 파일에 이미 옮긴 줄이 남아 있는가).

    옮긴 뒤 옆 파일을 비우지 못한 경우(그 사이 끊김, 윈도우에서 다른 프로세스가 읽는 중)에 같은 줄이 두 번 보이지 않게
    한다(RunStore._append_line의 옮기는 순서 참고).
    """
    main = _complete_lines(_read_bytes(path))
    side = _complete_lines(_read_bytes(path.with_name(path.name + PENDING_SUFFIX)))
    j = _overlap(main, side) if side else 0
    return main, side[j:], j > 0


def _read_or_none(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""


def _csv_line(cells: Iterable[str]) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator=_CSV_LINE_END).writerow(cells)
    return buf.getvalue()


def _cell(key: str, value: Any) -> str:
    """통계 칸 하나를 글자로. 실수는 repr(가장 짧게 원래 값으로 되돌아오는 자릿수)."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, str):
        if _LINE_BREAK.search(value):
            # 한 줄 = 한 통계 줄이어야 전원이 끊겨 잘린 줄을 줄바꿈 기준으로 잘라 낼 수 있다(따옴표 속 줄바꿈이
            # 잘리면 열린 따옴표가 뒤의 모든 줄을 삼킨다).
            raise ValueError(f"stats value {key!r} must not contain line breaks")
        return value
    raise TypeError(f"stats value {key!r} must be a scalar (number, str, bool or None), got {type(value).__name__}")


def _parse_csv(data: bytes) -> list[list[str]]:
    text = data.decode("utf-8")
    return list(csv.reader(io.StringIO(text, newline="")))


def _generation_of(cells: list[str], index: int | None) -> int | None:
    if index is None or index >= len(cells):
        return None
    try:
        return int(cells[index])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 저장소
# ---------------------------------------------------------------------------


class RunStore:
    """실행 폴더 하나. 학습 프로세스가 쓰고, 화면 프로세스는 주로 읽는다.

    run.json은 읽을 때마다 디스크에서 새로 읽는다(다른 프로세스가 고친 내용이 보이게). 쓰기는 한 프로세스(학습)에서
    하는 것을 전제로 한다. 화면이 메모 같은 것을 update_run_info로 쓸 수는 있지만, 동시에 쓰면 나중 쓰기가 이긴다.
    """

    root: Path
    policy: CheckpointPolicy

    def __init__(self, root: Path, policy: CheckpointPolicy | None = None) -> None:
        self.root = Path(root)
        self.policy = policy or CheckpointPolicy()
        self._stats_header: list[str] | None = None
        self._tail_checked: set[Path] = set()

    def __repr__(self) -> str:
        return f"RunStore({str(self.root)!r})"

    # -- 만들기, 열기 ---------------------------------------------------------
    @classmethod
    def create(
        cls, base_dir: Path, folder_name: str, run_info: dict[str, Any], *, policy: CheckpointPolicy | None = None
    ) -> RunStore:
        """base_dir 아래에 새 실행 폴더를 만든다. 이름이 겹치면 _2, _3 …을 붙인다."""
        info = dict(run_info)
        info.setdefault("store_format", STORE_FORMAT)
        info.setdefault("env_versions", [])
        info.setdefault("protected_checkpoints", [])
        payload = _json_bytes(info)  # 폴더를 만들기 전에 직렬화 오류를 낸다
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        root = cls._make_unique_dir(base, sanitize_folder_name(folder_name))
        try:
            for sub in (ENVS_DIR, CHECKPOINTS_DIR, CHAMPIONS_DIR, EXPORTS_DIR):
                (root / sub).mkdir()
            _atomic_write(root / RUN_INFO_FILE, lambda f: f.write(payload))
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise
        return cls(root, policy)

    @staticmethod
    def _make_unique_dir(base: Path, name: str) -> Path:
        """이름, 이름_2, 이름_3 … 가운데 아직 없는 첫 폴더를 만든다(mkdir이 원자적이라 동시에 만들어도 겹치지 않는다)."""
        n = 1
        while True:
            root = base / (name if n == 1 else f"{name}_{n}")
            try:
                root.mkdir()
                return root
            except FileExistsError:
                n += 1

    @classmethod
    def open(cls, path: Path, *, policy: CheckpointPolicy | None = None) -> RunStore:
        """있는 실행 폴더를 연다. 폴더 안의 run.json 경로를 줘도 된다."""
        p = Path(path)
        if p.name == RUN_INFO_FILE and p.is_file():
            p = p.parent
        if not (p / RUN_INFO_FILE).is_file():
            raise RunStoreError(S.RUNS_NOT_A_RUN.format(path=p))
        store = cls(p, policy)
        fmt = store.read_run_info().get("store_format", STORE_FORMAT)
        if not isinstance(fmt, int) or fmt > STORE_FORMAT:
            raise RunStoreError(S.RUNS_NEWER_FORMAT.format(got=fmt, want=STORE_FORMAT))
        return store

    # -- run.json -------------------------------------------------------------
    def read_run_info(self) -> dict[str, Any]:
        path = self.root / RUN_INFO_FILE
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise RunStoreError(S.RUNS_NOT_A_RUN.format(path=self.root)) from None
        try:
            info = json.loads(data.decode("utf-8-sig"))
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise RunStoreError(S.RUNS_INFO_BROKEN.format(path=path, detail=exc)) from None
        if not isinstance(info, dict):
            raise RunStoreError(S.RUNS_INFO_BROKEN.format(path=path, detail=type(info).__name__))
        return info

    def update_run_info(self, **changes: Any) -> None:
        """맨 위 키 단위로 합쳐서(얕은 병합) run.json을 통째로 다시 쓴다."""
        info = self.read_run_info()
        info.update(changes)
        self._write_run_info(info)

    def _write_run_info(self, info: dict[str, Any]) -> None:
        payload = _json_bytes(info)
        _atomic_write(self.root / RUN_INFO_FILE, lambda f: f.write(payload))

    # -- 환경 버전 ------------------------------------------------------------
    def add_env_version(self, env_text: str, from_generation: int) -> str:
        """환경 사본을 남기고 버전 id("v000" …)를 돌려준다.

        내용(UTF-8 바이트의 sha256)이 이미 있는 버전과 같으면 새 파일을 만들지 않고 그 id를 돌려준다.
        run.json의 env_versions는 적용 순서 기록이다: 마지막 기록과 같은 내용이면 아무것도 바꾸지 않고, 예전 버전으로
        되돌린 것이면 그 id로 기록을 하나 더한다(마지막 기록이 언제나 지금 쓰는 환경이 되게).
        """
        gen = _check_generation(from_generation)
        data = env_text.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        info = self.read_run_info()
        versions: list[dict[str, Any]] = list(info.get("env_versions", []))
        if versions and versions[-1].get("sha256") == digest:
            return str(versions[-1]["id"])
        same = [v for v in versions if v.get("sha256") == digest]
        if same:
            vid = str(same[0]["id"])
        else:
            numbers = [int(m.group(1)) for v in versions if (m := _VERSION_ID.fullmatch(str(v.get("id", ""))))]
            vid = f"v{max(numbers, default=-1) + 1:03d}"
            envs = self.root / ENVS_DIR
            envs.mkdir(exist_ok=True)
            _atomic_write(envs / f"{vid}.evoenv", lambda f: f.write(data))
        versions.append({"id": vid, "sha256": digest, "from_generation": gen})
        info["env_versions"] = versions
        self._write_run_info(info)
        return vid

    def env_text(self, version: str) -> str:
        """환경 사본의 글자 그대로(줄바꿈, BOM 글자 포함). 기록된 해시와 다르면 RunStoreError."""
        if not isinstance(version, str) or not _VERSION_ID.fullmatch(version):
            raise RunStoreError(S.RUNS_ENV_VERSION_MISSING.format(version=version))
        records = [v for v in self.read_run_info().get("env_versions", []) if v.get("id") == version]
        path = self.root / ENVS_DIR / f"{version}.evoenv"
        if not records or not path.is_file():
            raise RunStoreError(S.RUNS_ENV_VERSION_MISSING.format(version=version))
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != records[0].get("sha256"):
            raise RunStoreError(S.RUNS_ENV_VERSION_CHANGED.format(path=path))
        return data.decode("utf-8")

    # -- 줄 덧붙이기 (stats.csv, events.jsonl) ---------------------------------
    def _open_append(self, path: Path) -> IO[bytes]:
        """덧붙이기용으로 연다. 테스트에서 엑셀 잠금을 흉내 내려고 따로 둔 함수."""
        return open(path, "ab")

    def _repair_tail(self, path: Path) -> None:
        """이 저장소 객체로 처음 덧붙이기 전에, 잘린 마지막 줄(전원 끊김)을 잘라 낸다."""
        if path in self._tail_checked:
            return
        try:
            with open(path, "r+b") as f:
                size = f.seek(0, os.SEEK_END)
                start = max(0, size - 65536)
                f.seek(start)
                tail = f.read()
                good = _complete_lines(tail)
                if len(good) != len(tail):
                    if not good and start > 0:  # 64KB 넘는 한 줄: 전체를 본다
                        f.seek(0)
                        start, good = 0, _complete_lines(f.read())
                    f.truncate(start + len(good))
        except FileNotFoundError:
            pass
        self._tail_checked.add(path)

    def _append_line(self, path: Path, line: bytes) -> None:
        """한 줄(또는 여러 줄)을 덧붙인다. 파일이 잠겨 있으면 옆 파일(…pending)에 쌓아 두고, 풀리면 옮긴다.

        옮기는 순서: 옆 파일에서 아직 옮기지 않은 부분을 본 파일에 덧붙이고, 옆 파일을 비우고, 새 줄은 비우기에
        성공했을 때만 본 파일에 쓴다(실패하면 옆 파일에). 그래서 본 파일은 언제나 옆 파일의 앞부분(줄 단위)으로
        끝나거나 옆 파일과 겹치지 않고, 그 겹친 길이(_overlap)로 이미 옮긴 줄을 알아본다. 옮긴 직후 끊기거나 윈도우에서
        화면 프로세스가 옆 파일을 읽고 있어 비우지 못해도 같은 줄이 두 번 들어가지 않는다.
        """
        sidecar = path.with_name(path.name + PENDING_SUFFIX)
        raw = _read_or_none(sidecar)
        side = _complete_lines(raw or b"")
        delays: tuple[float, ...] = () if side else _APPEND_RETRY_DELAYS
        for delay in (*delays, None):
            try:
                self._repair_tail(path)
                if side:
                    rest = side[_overlap(_tail_bytes(path, len(side)), side):]
                    if rest:
                        with self._open_append(path) as f:
                            f.write(rest)
                    side = raw = b""
                    if not self._clear_sidecar(sidecar):
                        self._write_sidecar(sidecar, line)
                        return
                with self._open_append(path) as f:
                    f.write(line)
                if raw is not None and not side:
                    self._clear_sidecar(sidecar)  # 줄 없이 남은 옆 파일(비우기만 됐거나 잘린 줄뿐)
                return
            except PermissionError:
                self._tail_checked.discard(path)
                if delay is None:
                    self._write_sidecar(sidecar, line)
                    return
                _sleep(delay)
            except BaseException:
                # 디스크 가득 참 등으로 줄의 앞부분만 써졌을 수 있다: 다음 덧붙이기 전에 그 꼬리를 다시 잘라 낸다
                # (안 그러면 잘린 "1"과 다음 줄 "2,…"가 붙어 12세대 줄이 된다).
                self._tail_checked.discard(path)
                raise

    def _write_sidecar(self, sidecar: Path, line: bytes) -> None:
        """옆 파일에 덧붙인다. 옆 파일마저 잠겨 있으면(윈도우: OneDrive가 올리는 중 등) 잠깐씩 다시 시도한다."""
        for delay in (*_REPLACE_RETRY_DELAYS, None):
            try:
                self._repair_tail(sidecar)
                with open(sidecar, "ab") as f:
                    f.write(line)
                return
            except PermissionError as exc:
                self._tail_checked.discard(sidecar)
                if delay is None:
                    raise RunStoreError(S.RUNS_FILE_BUSY.format(path=sidecar)) from exc
                _sleep(delay)
            except BaseException:
                self._tail_checked.discard(sidecar)
                raise

    def _clear_sidecar(self, sidecar: Path) -> bool:
        """옮긴 옆 파일을 지운다. 윈도우에서 다른 프로세스가 읽고 있으면 지우기는 막히지만 비우기는 되므로 비운다."""
        self._tail_checked.discard(sidecar)
        try:
            sidecar.unlink(missing_ok=True)
            return True
        except PermissionError:
            pass
        try:
            with open(sidecar, "r+b") as f:
                f.truncate(0)
            return True
        except OSError:
            return False

    # -- stats.csv ------------------------------------------------------------
    def _stats_path(self) -> Path:
        return self.root / STATS_FILE

    def _read_stats_header(self) -> list[str] | None:
        if self._stats_header is None:
            try:
                with open(self._stats_path(), "rb") as f:
                    first = f.readline()
            except FileNotFoundError:
                return None
            if not first.endswith(b"\n"):
                return None
            try:
                rows = _parse_csv(first.removeprefix(_BOM))
            except UnicodeDecodeError:  # 엑셀이 ANSI(CP949) CSV로 다시 저장한 파일
                raise RunStoreError(S.RUNS_STATS_NOT_UTF8.format(path=self._stats_path())) from None
            if not rows or not rows[0]:
                return None
            self._stats_header = rows[0]
        return self._stats_header

    def stats_columns(self) -> list[str] | None:
        """stats.csv의 열 이름(아직 없으면 None)."""
        header = self._read_stats_header()
        return list(header) if header is not None else None

    def append_stats(self, row: dict[str, Any]) -> None:
        """통계 한 줄. 열 이름은 첫 줄의 키, 이후 줄은 같은 키여야 한다(순서는 상관없고 첫 줄 순서로 쓴다)."""
        keys = list(row)
        if not keys or not all(isinstance(k, str) for k in keys):
            raise TypeError("stats row keys must be non-empty str")
        if any(_LINE_BREAK.search(k) for k in keys):
            raise ValueError("stats column names must not contain line breaks")
        if self._stats_header is not None and not self._stats_path().exists():
            self._stats_header = None  # 누가 지웠다: 열 이름부터 다시 쓴다
            self._tail_checked.discard(self._stats_path())
        header = self._read_stats_header()
        if header is None:
            # 열 이름 줄은 원자적으로 쓴다. 옆 파일에 쌓인 (더 오래된) 줄이 있으면 그 뒤에, 새 줄은 맨 뒤에.
            sidecar = self._stats_path().with_name(STATS_FILE + PENDING_SUFFIX)
            line = _csv_line(_cell(k, row[k]) for k in keys).encode("utf-8")
            text = _csv_line(keys).encode("utf-8")
            pending = _complete_lines(_read_bytes(sidecar))
            _atomic_write(self._stats_path(), lambda f: f.write(_BOM + text + pending))
            self._stats_header = keys
            self._tail_checked.add(self._stats_path())
            self._append_line(self._stats_path(), line)
            return
        if set(keys) != set(header):
            missing = [k for k in header if k not in row]
            extra = [k for k in keys if k not in header]
            raise ValueError(S.RUNS_STATS_KEYS.format(missing=", ".join(missing) or "-", extra=", ".join(extra) or "-"))
        line = _csv_line(_cell(k, row[k]) for k in header).encode("utf-8")
        self._append_line(self._stats_path(), line)

    def _stats_table(self) -> tuple[list[str], list[list[str]], list[list[str]], bool]:
        """(열 이름, stats.csv의 줄들, 옆 파일에서 아직 옮기지 않은 줄들, 옆 파일에 이미 옮긴 줄이 남아 있는가). 열 수가 맞지 않는 줄은 버린다."""
        path = self._stats_path()
        main, side, merged = _log_bytes(path)
        try:
            rows = _parse_csv(main.removeprefix(_BOM))
            pending = _parse_csv(side)
        except UnicodeDecodeError:
            raise RunStoreError(S.RUNS_STATS_NOT_UTF8.format(path=path)) from None
        if not rows:
            return [], [], [], merged
        header = rows[0]
        n = len(header)
        return header, [r for r in rows[1:] if len(r) == n], [r for r in pending if len(r) == n], merged

    def read_stats(self) -> list[dict[str, str]]:
        """모든 통계 줄(값은 글자). 엑셀 잠금 때문에 옆 파일에 쌓인 줄도 뒤에 붙인다."""
        header, rows, pending, _ = self._stats_table()
        return [dict(zip(header, r, strict=True)) for r in (*rows, *pending)]

    @property
    def pending_stats(self) -> int:
        """stats.csv가 잠겨 있어(엑셀 등) 옆 파일에 쌓아 둔 통계 줄 수. 0이 아니면 S.RUNS_STATS_PENDING으로 알린다."""
        path = self._stats_path()
        if not _complete_lines(_read_bytes(path.with_name(STATS_FILE + PENDING_SUFFIX))):
            return 0
        return _log_bytes(path)[1].count(b"\n")

    # -- events.jsonl ---------------------------------------------------------
    def append_event(self, kind: str, generation: int, **data: Any) -> None:
        """사건 한 줄: {"kind", "generation", "time"(현지 시각 ISO 8601), **data}."""
        clash = _EVENT_RESERVED & data.keys()
        if clash:
            raise TypeError(f"event data may not use reserved keys {sorted(clash)}")
        record = {
            "kind": str(kind),
            "generation": _check_generation(generation),
            "time": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            **data,
        }
        line = json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
        self._append_line(self.root / EVENTS_FILE, line.encode("utf-8"))

    def read_events(self) -> list[dict[str, Any]]:
        main, side, _ = _log_bytes(self.root / EVENTS_FILE)
        data = main + side
        out: list[dict[str, Any]] = []
        for raw in data.split(b"\n"):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue  # 손상된 줄은 건너뛴다(기록 화면이 멈추지 않게)
            if isinstance(obj, dict):
                out.append(obj)
        return out

    # -- 체크포인트 -----------------------------------------------------------
    def _checkpoint_path(self, generation: int) -> Path:
        return self.root / CHECKPOINTS_DIR / f"gen_{generation:06d}.pkl"

    def save_checkpoint(self, generation: int, state: dict[str, Any]) -> Path:
        """상태를 피클(프로토콜 5)로 원자적으로 저장하고 정리 규칙을 적용한다."""
        g = _check_generation(generation)
        folder = self.root / CHECKPOINTS_DIR
        folder.mkdir(exist_ok=True)
        path = self._checkpoint_path(g)
        # 덧붙이기는 매번 디스크에 내리지 않는다(느림). 체크포인트 앞에서 한 번 내려, 전원이 끊겨도 체크포인트가
        # 그 세대까지의 통계·사건 줄보다 앞서 남지 않게 한다(이어 가면 빈 세대가 생긴다).
        for log in (STATS_FILE, EVENTS_FILE):
            _sync_file(self.root / log)
            _sync_file(self.root / (log + PENDING_SUFFIX))
        _atomic_write(path, lambda f: pickle.dump(state, f, protocol=PICKLE_PROTOCOL))
        self._apply_retention(g)
        return path

    def _protected(self) -> list[int]:
        return [int(g) for g in self.read_run_info().get("protected_checkpoints", [])]

    def _apply_retention(self, just_saved: int) -> None:
        gens = self.checkpoints()
        p = self.policy
        keep = set(gens[-p.keep_last:]) | {just_saved} | set(self._protected())
        if p.keep_every > 0:
            keep |= {g for g in gens if g % p.keep_every == 0}
        for g in gens:
            if g not in keep:
                try:
                    self._checkpoint_path(g).unlink(missing_ok=True)
                except OSError:
                    pass  # 화면 프로세스가 읽는 중이면 다음 저장 때 다시 지운다
        _remove_stale_tmp(self.root / CHECKPOINTS_DIR, _CHECKPOINT_TMP)

    def protect(self, generation: int) -> None:
        """이 세대 체크포인트를 정리에서 뺀다(분기 지점, 환경 수정 지점). run.json의 protected_checkpoints."""
        g = _check_generation(generation)
        info = self.read_run_info()
        marks = sorted({int(x) for x in info.get("protected_checkpoints", [])} | {g})
        if marks != info.get("protected_checkpoints"):
            info["protected_checkpoints"] = marks
            self._write_run_info(info)

    def checkpoints(self) -> list[int]:
        try:
            names = os.listdir(self.root / CHECKPOINTS_DIR)
        except FileNotFoundError:
            return []
        return sorted(int(m.group(1)) for n in names if (m := _CHECKPOINT_NAME.fullmatch(n)))

    def load_checkpoint(self, generation: int | None = None) -> tuple[int, dict[str, Any]]:
        """(세대, 상태). generation이 None이면 가장 최근 체크포인트."""
        gens = self.checkpoints()
        if not gens:
            raise RunStoreError(S.RUNS_NO_CHECKPOINT.format(path=self.root))
        g = gens[-1] if generation is None else _check_generation(generation)
        if g not in gens:
            raise RunStoreError(S.RUNS_CHECKPOINT_MISSING.format(generation=g, available=", ".join(map(str, gens))))
        path = self._checkpoint_path(g)
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
        except OSError:
            raise
        except Exception as exc:  # 손상, 다른 버전의 클래스 등 피클이 낼 수 있는 모든 오류
            raise RunStoreError(S.RUNS_CHECKPOINT_BROKEN.format(path=path, detail=exc)) from exc
        if not isinstance(state, dict):
            raise RunStoreError(S.RUNS_CHECKPOINT_BROKEN.format(path=path, detail=type(state).__name__))
        return g, state

    # -- 세대 대표와 챔피언 ---------------------------------------------------
    def _champion_path(self, generation: int, kind: str) -> Path:
        return self.root / CHAMPIONS_DIR / f"gen_{generation:06d}_{kind}.npz"

    def save_champion(self, generation: int, kind: ChampionKind, genome: np.ndarray, **data: Any) -> None:
        """유전자(float64 1차원)와 점수·시드·환경 버전 같은 값을 npz로(피클 없이) 원자적으로 저장한다."""
        g = _check_generation(generation)
        _check_kind(kind)
        arrays: dict[str, np.ndarray] = {"genome": np.ascontiguousarray(genome, dtype=np.float64)}
        if arrays["genome"].ndim != 1:
            raise ValueError(f"genome must be 1-D, got shape {arrays['genome'].shape}")
        for key, value in data.items():
            if key in _SAVEZ_RESERVED:
                raise TypeError(f"champion data may not use the key {key!r}")
            a = np.asarray(value)
            if a.dtype.hasobject:
                raise TypeError(f"champion data {key!r} must be numeric or str (no pickled objects)")
            arrays[key] = a
        folder = self.root / CHAMPIONS_DIR
        folder.mkdir(exist_ok=True)
        # 관찰 기록(obs_*)이 든 세대 대표만 압축: 착륙선 초반 세대 30.4KB → 18.7KB, 쓰기 +1ms·읽기 +0.2ms(V12 측정).
        # 다른 파일은 v1.1과 같은 무압축. np.load는 둘 다 읽는다.
        save = np.savez_compressed if any(k.startswith(OBS_PREFIX) for k in arrays) else np.savez
        _atomic_write(self._champion_path(g, kind), lambda f: save(f, allow_pickle=False, **arrays))

    def load_champion(self, generation: int, kind: str) -> dict[str, np.ndarray]:
        """챔피언·세대 대표 파일의 값(genome, score, env_version …). 관찰 기록 키(obs_*)는 빼고 돌려준다(v1.1과 같은
        키): 학습·재생·판정은 관찰 기록을 읽지 않는다. 관찰 기록은 load_observation."""
        g = _check_generation(generation)
        _check_kind(kind)
        path = self._champion_path(g, kind)
        label = S.RUNS_CHAMPION_KIND_LABELS.get(kind, kind)
        if not path.is_file():
            raise RunStoreError(S.RUNS_CHAMPION_MISSING.format(generation=g, kind=label))
        try:
            with np.load(path, allow_pickle=False) as z:  # with: 윈도우에서 파일을 잡고 있지 않게 바로 닫는다
                return {k: z[k] for k in z.files if not k.startswith(OBS_PREFIX)}
        except OSError:
            raise
        except Exception as exc:
            raise RunStoreError(S.RUNS_CHAMPION_BROKEN.format(path=path, detail=exc)) from exc

    def load_observation(self, generation: int) -> dict[str, np.ndarray] | None:
        """세대 대표 파일의 관찰 기록 키(obs_*)만. 파일이 없거나, 관찰 키가 없거나(v1.2 전 세대, --no-observe),
        읽을 수 없으면 None(예외를 내지 않는다). 검사는 runs.observe.decode가 한다."""
        got = read_representative(self._champion_path(_check_generation(generation), "representative"))
        if got is None or not got[1]:
            return None
        return got[1]

    def champion_generations(self, kind: str) -> list[int]:
        _check_kind(kind)
        pattern = re.compile(rf"gen_(\d{{6,}})_{kind}\.npz")
        try:
            names = os.listdir(self.root / CHAMPIONS_DIR)
        except FileNotFoundError:
            return []
        return sorted(int(m.group(1)) for n in names if (m := pattern.fullmatch(n)))

    # -- 끊긴 실행 정리 -------------------------------------------------------
    def truncate_after(self, generation: int) -> None:
        """세대 번호가 generation보다 큰 체크포인트, 챔피언 파일, 통계 줄을 지운다(보호 표시도). 사건 기록은 그대로.

        최신 체크포인트에서 이어가기 전에 부른다. 도중에 끊겨도 같은 값으로 다시 부르면 마저 정리되도록, 이어갈
        기준이 되는 체크포인트부터 지우고 통계는 맨 나중에 고친다.
        """
        g = _check_generation(generation)
        try:
            for c in self.checkpoints():
                if c > g:
                    _remove(self._checkpoint_path(c))
            for kind in CHAMPION_KINDS:
                for c in self.champion_generations(kind):
                    if c > g:
                        _remove(self._champion_path(c, kind))
        except PermissionError as exc:
            raise RunStoreError(S.RUNS_FILE_BUSY.format(path=exc.filename or self.root)) from exc
        # 끊긴 실행이 남긴 임시 파일(챔피언은 세대마다 쓰므로 여기에 쌓인다). 체크포인트 폴더는 저장 때마다 본다.
        for folder in (self.root, self.root / ENVS_DIR, self.root / CHAMPIONS_DIR):
            _remove_stale_tmp(folder, _STORE_TMP)
        info = self.read_run_info()
        marks = [int(x) for x in info.get("protected_checkpoints", [])]
        if any(x > g for x in marks):
            info["protected_checkpoints"] = [x for x in marks if x <= g]
            self._write_run_info(info)
        self._truncate_stats(g)

    def _truncate_stats(self, g: int) -> None:
        """generation 열 값이 g보다 큰 줄을 지운다. generation 열이 없으면 판단할 수 없으니 그대로 둔다.

        옆 파일을 먼저 고친다. 이미 옮긴 줄이 옆 파일에 남아 있는데 stats.csv를 먼저 줄이면, 그 사이에 끊겼을 때 다음
        덧붙이기에서 지운 줄이 되살아난다.
        """
        path = self._stats_path()
        sidecar = path.with_name(STATS_FILE + PENDING_SUFFIX)
        header, rows, pending, merged = self._stats_table()
        if "generation" not in header:
            return
        col = header.index("generation")

        def keep(r: list[str]) -> bool:
            gen = _generation_of(r, col)
            return gen is None or gen <= g

        kept, kept_pending = [r for r in rows if keep(r)], [r for r in pending if keep(r)]
        if merged or len(kept_pending) != len(pending):
            if kept_pending:
                ptext = "".join(_csv_line(r) for r in kept_pending).encode("utf-8")
                _atomic_write(sidecar, lambda f: f.write(ptext))
                self._tail_checked.discard(sidecar)
            elif not self._clear_sidecar(sidecar):
                raise RunStoreError(S.RUNS_FILE_BUSY.format(path=sidecar))
        raw = _read_bytes(path)
        if len(kept) != len(rows) or len(_complete_lines(raw)) != len(raw):
            text = "".join(_csv_line(r) for r in (header, *kept)).encode("utf-8")
            try:
                _atomic_write(path, lambda f: f.write(_BOM + text))
            except RunStoreError:
                raise RunStoreError(S.RUNS_STATS_LOCKED.format(path=path)) from None
            self._tail_checked.add(path)

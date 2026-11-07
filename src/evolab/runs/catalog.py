"""실행 목록, 메모, 삭제, 분기, 복제 (명세: 저장, 기록, 내보내기 > 이어하기, 기록 화면).

기록 화면이 쓰는 함수들. 화면 없이도 쓸 수 있고, 실행 폴더 형식은 store.py를 따른다.

- list_runs: 실행 폴더 목록(최근 것부터). 폴더마다 run.json과 stats.csv의 첫 줄(열 이름)·마지막 몇 줄만 읽는다.
  통계 파일이 수십 MB여도 빠르다. 읽을 수 없는 폴더는 예외 대신 ok=False로 목록에 넣는다.
- delete_run: runs 폴더 바로 아래의, run.json이 있는 폴더만 지운다. 먼저 숨긴 이름(.deleting-…)으로 바꾼 뒤
  지운다: 윈도우에서 다른 프로그램이 안의 파일을 열고 있으면 이름 바꾸기가 실패해 아무것도 지우지 않고, 지우다가
  끊겨도 반쯤 지워진 실행이 목록에 남지 않는다(남은 숨긴 폴더는 다음 삭제 때 치운다).
- branch_run: 최신이 아닌 체크포인트에서 이어 가려면 새 실행 폴더로 분기한다. 그 세대까지의 기록(환경 버전, 통계,
  사건, 챔피언)과 그 체크포인트를 복사하고 run.json에 부모 실행과 세대를 적는다. 숨긴 임시 폴더(.branching-…)에
  다 만든 뒤 이름을 바꾸므로, 도중에 끊긴 분기는 목록에 나오지 않는다.
- clone_settings: 복제해서 새 실행할 때 필요한 환경 글과 학습 설정.

환경 버전의 적용 세대(from_generation)는 '끝낸 세대 수'다. 체크포인트 g에서 이어 갈 때 쓰는 환경은 적용 세대가
g 이하인 마지막 기록이다(Trainer.resume과 같은 규칙). 분기는 그 기록까지만 가져간다.
"""

from __future__ import annotations

import csv
import datetime as dt
import errno
import hashlib
import itertools
import json
import math
import os
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from evolab.app import strings as S
from evolab.runs import store as _st
from evolab.runs.store import (
    CHAMPION_KINDS,
    CHAMPIONS_DIR,
    CHECKPOINTS_DIR,
    ENVS_DIR,
    EVENTS_FILE,
    EXPORTS_DIR,
    PENDING_SUFFIX,
    RUN_INFO_FILE,
    STATS_FILE,
    STORE_FORMAT,
    RunStore,
    RunStoreError,
    sanitize_folder_name,
)

_TAIL_CHUNK: Final = 65536
"""stats.csv 끝에서 처음 읽는 바이트 수. 마지막 줄이 이보다 길면 늘려 가며 읽는다."""

DELETING_PREFIX: Final = ".deleting-"
"""지우는 중인 실행 폴더의 숨긴 이름 앞부분. 목록에는 나오지 않는다."""
BRANCHING_PREFIX: Final = ".branching-"
"""분기로 만드는 중인 실행 폴더의 숨긴 이름 앞부분. 다 만든 뒤 이름을 바꾼다."""
_STALE_BRANCH_SECONDS: Final = 3600.0
"""이보다 오래된 분기 임시 폴더는 끊긴 분기가 남긴 것으로 보고 지운다(초). 다른 창이 지금 만드는 중일 수 있어 넉넉히."""

_tmp_counter = itertools.count()


@dataclass
class RunSummary:
    """기록 화면의 한 줄."""

    path: Path
    name: str
    """폴더 이름."""
    env_name: str = ""
    method: str = ""
    """학습 방법 id(cmaes …). 화면 이름은 S.METHOD_NAMES."""
    started: str = ""
    """시작 시각(ISO 8601). run.json의 created, 없으면 첫 사건의 시각, 그것도 없으면 run.json의 수정 시각."""
    generations: int = 0
    """통계 마지막 줄의 세대 번호(끝낸 세대 수). 통계가 없으면 0."""
    best_score: float | None = None
    """챔피언 검증 점수(마지막 줄의 champion_score). 챔피언이 아직 없으면 읽은 끝부분 줄들의 score_best 최댓값."""
    memo: str = ""
    parent: dict[str, Any] | None = None
    """분기한 실행이면 {"run": 부모 폴더 이름, "path": 부모 경로, "generation": 분기 세대}."""
    ok: bool = True
    """False면 run.json이 없거나 읽을 수 없는 폴더(열기·이어하기 불가)."""
    error: str | None = None
    """ok가 False인 까닭, 또는 ok인데 통계를 읽지 못한 경고(한국어)."""


# ---------------------------------------------------------------------------
# 목록
# ---------------------------------------------------------------------------


def list_runs(runs_dir: Path) -> list[RunSummary]:
    """runs 폴더 아래의 실행들(최근 시작한 것부터). 숨긴 폴더(이름이 '.'으로 시작)는 뺀다."""
    base = Path(runs_dir)
    try:
        with os.scandir(base) as it:
            entries = [e for e in it if not e.name.startswith(".") and e.is_dir()]
    except (FileNotFoundError, NotADirectoryError):
        return []
    keyed: list[tuple[float, str, RunSummary]] = []
    for entry in entries:
        summary = _summarize(Path(entry.path))
        keyed.append((_sort_time(summary), summary.name, summary))
    keyed.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [s for _, _, s in keyed]


def read_summary(run: Path) -> RunSummary:
    """실행 폴더 하나의 요약(목록의 한 줄과 같다)."""
    return _summarize(Path(run))


def _summarize(root: Path) -> RunSummary:
    summary = RunSummary(path=root, name=root.name)
    info_path = root / RUN_INFO_FILE
    if not info_path.is_file():
        summary.ok = False
        summary.error = S.RUNS_NOT_A_RUN.format(path=root)
        summary.started = _mtime_iso(root)
        return summary
    try:
        info = RunStore(root).read_run_info()
    except (RunStoreError, OSError) as exc:
        summary.ok = False
        summary.error = str(exc)
        summary.started = _mtime_iso(info_path)
        return summary
    summary.env_name = _text(info.get("env_name"))
    config = info.get("config") if isinstance(info.get("config"), dict) else {}
    summary.method = _text(info.get("method") or config.get("method"))
    summary.memo = _text(info.get("memo"))
    parent = info.get("parent")
    summary.parent = dict(parent) if isinstance(parent, dict) else None
    summary.started = _text(info.get("created")) or _first_event_time(root) or _mtime_iso(info_path)
    fmt = info.get("store_format", STORE_FORMAT)
    if not isinstance(fmt, int) or fmt > STORE_FORMAT:
        summary.ok = False
        summary.error = S.RUNS_NEWER_FORMAT.format(got=fmt, want=STORE_FORMAT)
        return summary
    try:
        header, rows = _last_stats_rows(root)
    except UnicodeDecodeError:
        summary.error = S.RUNS_STATS_NOT_UTF8.format(path=root / STATS_FILE)
        return summary
    except csv.Error as exc:
        summary.error = S.REC_STATS_BROKEN.format(path=root / STATS_FILE, detail=exc)
        return summary
    except OSError as exc:
        summary.error = str(exc)
        return summary
    summary.generations, summary.best_score = _progress(header, rows)
    return summary


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _mtime_iso(path: Path) -> str:
    try:
        return dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""


def _first_event_time(root: Path) -> str:
    """events.jsonl 첫 줄의 시각(첫 줄만 읽는다)."""
    try:
        with open(root / EVENTS_FILE, "rb") as f:
            first = f.readline(65536)
    except OSError:
        return ""
    try:
        obj = json.loads(first.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return ""
    return _text(obj.get("time")) if isinstance(obj, dict) else ""


def _sort_time(summary: RunSummary) -> float:
    try:
        return dt.datetime.fromisoformat(summary.started).timestamp()  # 시간대 없는 값은 현지 시각으로
    except (ValueError, OverflowError, OSError):
        pass
    try:
        return summary.path.stat().st_mtime
    except OSError:
        return 0.0


def _stats_header(path: Path) -> list[str]:
    try:
        with open(path, "rb") as f:
            first = f.readline()
    except FileNotFoundError:
        return []
    if not first.endswith(b"\n"):
        return []
    rows = _st._parse_csv(first.removeprefix(_st._BOM))
    return rows[0] if rows else []


def _tail_rows(path: Path, has_header: bool) -> list[list[str]]:
    """파일 끝의 완전한 줄들을 CSV로 읽는다. 마지막 줄을 하나라도 얻을 만큼만 읽는다(처음 64KB, 모자라면 늘림).

    has_header면 파일 첫 줄(열 이름)은 뺀다. 줄바꿈으로 끝나지 않은 마지막 줄(쓰는 중, 전원 끊김)은 버린다.
    """
    try:
        f = open(path, "rb")
    except FileNotFoundError:
        return []
    with f:
        size = f.seek(0, os.SEEK_END)
        n = _TAIL_CHUNK
        while True:
            start = max(0, size - n)
            f.seek(start)
            data = _st._complete_lines(f.read(size - start))
            if start > 0 or has_header:
                # 앞쪽은 잘린 줄이거나 열 이름이다(통계 칸에는 줄바꿈이 없으므로 줄 경계 = 통계 줄 경계)
                cut = data.find(b"\n")
                data = data[cut + 1:] if cut >= 0 else b""
            if data or start == 0:
                return _st._parse_csv(data.removeprefix(_st._BOM))
            n *= 4


def _last_stats_rows(root: Path) -> tuple[list[str], list[list[str]]]:
    """(열 이름, 끝부분 통계 줄들 오래된 것부터). 엑셀 잠금 때문에 옆 파일(…pending)에 쌓인 줄이 더 새 줄이다.

    옆 파일의 앞부분이 이미 stats.csv로 옮겨졌을 수 있지만(store._log_bytes 참고) 같은 줄이 두 번 보여도 마지막 줄과
    최댓값은 그대로다.
    """
    path = root / STATS_FILE
    header = _stats_header(path)
    if not header:
        return [], []
    rows = _tail_rows(path, True) + _tail_rows(path.with_name(STATS_FILE + PENDING_SUFFIX), False)
    return header, [r for r in rows if len(r) == len(header)]


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def _progress(header: list[str], rows: list[list[str]]) -> tuple[int, float | None]:
    """(세대 수, 최고 점수). 챔피언 점수는 좋아질 때만 바뀌므로 마지막 줄 값이 가장 높다."""
    col = {name: i for i, name in enumerate(header)}
    generations = 0
    gi = col.get("generation")
    if gi is not None:
        for r in reversed(rows):
            g = _st._generation_of(r, gi)
            if g is not None:
                generations = g
                break
    best: float | None = None
    ci = col.get("champion_score")
    if ci is not None and rows:
        best = _float_or_none(rows[-1][ci])
    si = col.get("score_best")
    if best is None and si is not None:
        values = [x for r in rows if (x := _float_or_none(r[si])) is not None]
        best = max(values) if values else None
    return generations, best


# ---------------------------------------------------------------------------
# 메모, 삭제
# ---------------------------------------------------------------------------


def set_memo(run: Path, memo: str) -> None:
    """run.json의 memo를 바꾼다. 빈 글이면 메모를 지운 것으로 둔다."""
    RunStore.open(run).update_run_info(memo=str(memo))


def _is_link(path: Path) -> bool:
    """심볼릭 링크나 윈도우 연결 지점(junction): 따라가서 지우면 runs 폴더 밖의 파일이 지워진다."""
    try:
        return path.is_symlink() or path.is_junction()
    except OSError:
        return False


def _check_deletable(run: Path, runs_dir: Path) -> Path:
    """지워도 되는 실행 폴더인지 확인하고 실제 경로를 돌려준다. 아니면 RunStoreError."""
    if _is_link(run):
        raise RunStoreError(S.REC_DELETE_LINK.format(path=run))
    try:
        real = run.resolve(strict=True)
        base = runs_dir.resolve(strict=True)
    except OSError:
        raise RunStoreError(S.REC_DELETE_OUTSIDE.format(path=run, runs_dir=runs_dir)) from None
    if real == base or real.parent != base or not real.is_dir() or real.name.startswith("."):
        raise RunStoreError(S.REC_DELETE_OUTSIDE.format(path=run, runs_dir=runs_dir))
    info = real / RUN_INFO_FILE
    if _is_link(info) or not info.is_file():
        raise RunStoreError(S.REC_DELETE_NOT_RUN.format(path=run))
    return real


def _rmtree_once(path: Path) -> None:
    def retry_writable(func: Any, p: str, exc: BaseException) -> None:
        # 윈도우: 읽기 전용 표시가 있는 파일은 지울 수 없다. 표시를 풀고 한 번 더.
        if not isinstance(exc, PermissionError):
            raise exc
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        func(p)

    shutil.rmtree(path, onexc=retry_writable)


def _rmtree(path: Path) -> bool:
    """폴더를 통째로 지운다. 잠긴 파일이 있으면 잠깐씩 다시 시도한다. 끝내 못 지우면 False."""
    for delay in (*_st._REPLACE_RETRY_DELAYS, None):
        try:
            _rmtree_once(path)
            return True
        except FileNotFoundError:
            if not path.exists():
                return True
        except OSError:
            if delay is None:
                return False
            _st._sleep(delay)
    return not path.exists()


def _clean_leftovers(base: Path, prefix: str, older_than: float = 0.0) -> None:
    """끊긴 삭제·분기가 남긴 숨긴 폴더(prefix로 시작)를 치운다(older_than초보다 오래된 것만, 할 수 없으면 그만)."""
    now = time.time()
    try:
        with os.scandir(base) as it:
            names = [e.name for e in it if e.name.startswith(prefix) and e.is_dir(follow_symlinks=False)
                     and now - e.stat(follow_symlinks=False).st_mtime >= older_than]
    except OSError:
        return
    for name in names:
        shutil.rmtree(base / name, ignore_errors=True)


def delete_run(run: Path, runs_dir: Path) -> None:
    """실행 폴더를 통째로 지운다(되돌릴 수 없다). runs_dir 바로 아래의, run.json이 있는 폴더만 지운다.

    링크 폴더, runs_dir 밖의 폴더, run.json이 없는 폴더는 RunStoreError(한국어)로 거절한다.
    """
    real = _check_deletable(Path(run), Path(runs_dir))
    base = real.parent
    _clean_leftovers(base, DELETING_PREFIX)
    trash = base / f"{DELETING_PREFIX}{os.getpid()}-{next(_tmp_counter)}-{real.name}"
    for delay in (*_st._REPLACE_RETRY_DELAYS, None):
        try:
            os.rename(real, trash)
            break
        except PermissionError as exc:
            # 윈도우: 안의 파일을 다른 프로그램(학습 프로세스, 엑셀, 백신)이 열고 있으면 폴더 이름을 바꿀 수 없다
            if delay is None:
                raise RunStoreError(S.REC_DELETE_FILES_BUSY.format(path=real)) from exc
            _st._sleep(delay)
    _rmtree(trash)  # 남으면 숨긴 폴더라 목록에 안 나오고 다음 삭제 때 다시 치운다


# ---------------------------------------------------------------------------
# 분기
# ---------------------------------------------------------------------------


def _from_generation(record: dict[str, Any]) -> int:
    try:
        return int(record.get("from_generation", 0))
    except (TypeError, ValueError):
        return 0


def env_versions_until(info: dict[str, Any], generation: int) -> list[dict[str, Any]]:
    """체크포인트 generation까지 적용된 환경 버전 기록(적용 순서대로). 마지막 기록이 그 시점의 환경이다."""
    versions = [v for v in info.get("env_versions", []) if isinstance(v, dict)]
    return [v for v in versions if _from_generation(v) <= generation] or versions[:1]


def branch_folder_name(info: dict[str, Any], generation: int, now: dt.datetime) -> str:
    """분기 실행의 폴더 이름: 새 실행과 같은 '날짜_시각_환경_방법' 뒤에 분기 세대를 붙인다."""
    method = _text(info.get("method"))
    return S.REC_BRANCH_FOLDER.format(
        stamp=f"{now:%Y-%m-%d_%H%M}", env=_text(info.get("env_name")) or S.RUNS_DEFAULT_FOLDER_NAME,
        method=S.METHOD_NAMES.get(method, method), generation=generation,
    )


def branch_run(run: Path, generation: int, runs_dir: Path) -> Path:
    """체크포인트 generation에서 새 실행 폴더로 분기한다. 새 폴더 경로를 돌려준다.

    새 폴더에는 그 세대까지의 환경 버전, 통계 줄, 사건(과 새 "branch" 사건), 세대 대표·챔피언 파일과 체크포인트
    generation 하나가 들어간다. run.json은 부모의 것을 복사해 parent, created, env_versions,
    protected_checkpoints(분기 지점 = generation)를 고치고 메모는 비운다. 반복 실행 묶음 소속(group)도 빼서
    분기가 부모의 묶음에 끼어 평균·띠를 바꾸지 않게 한다. 원래 실행은 바꾸지 않는다.
    RunStore.open(새 폴더).load_checkpoint()는 generation을 돌려주고, 거기서 Trainer.resume으로 이어 가면
    부모 실행을 그대로 다시 만든다(환경과 설정을 바꾸지 않았다면).
    """
    src = RunStore.open(Path(run))
    g = _st._check_generation(generation)
    gens = src.checkpoints()
    if g not in gens:
        raise RunStoreError(S.RUNS_CHECKPOINT_MISSING.format(generation=g, available=", ".join(map(str, gens)) or "-"))
    src.load_checkpoint(g)  # 손상된 체크포인트면 폴더를 만들기 전에 RunStoreError
    info = src.read_run_info()
    versions = env_versions_until(info, g)
    env_files = _read_env_files(src, versions)
    header, rows, pending, _ = src._stats_table()
    stats_rows = _rows_until(header, [*rows, *pending], g)
    events = [e for e in src.read_events() if (eg := _event_generation(e)) is not None and eg <= g]

    now = dt.datetime.now()
    # 메모와 반복 실행 묶음 소속은 물려받지 않는다(분기는 같은 시드의 실행을 하나 더 만드는 것이라 묶음에 넣으면
    # 그 시드가 두 번 세어진다)
    new_info = {k: v for k, v in info.items() if k not in ("memo", "group")}
    new_info.update(
        created=now.isoformat(timespec="seconds"),
        parent={"run": src.root.name, "path": str(src.root.resolve()), "generation": g},
        env_versions=versions,
        protected_checkpoints=[g],  # 분기 지점은 정리하지 않는다(store.protect와 같은 기록)
    )
    payload = _st._json_bytes(new_info)  # 폴더를 만들기 전에 직렬화 오류를 낸다
    events.append({
        "kind": "branch", "generation": g, "time": now.astimezone().isoformat(timespec="seconds"),
        "parent": src.root.name,
    })
    event_bytes = "".join(json.dumps(e, ensure_ascii=False, default=_st._json_default) + "\n" for e in events).encode("utf-8")

    base = Path(runs_dir)
    base.mkdir(parents=True, exist_ok=True)
    _clean_leftovers(base, BRANCHING_PREFIX, _STALE_BRANCH_SECONDS)
    # 숨긴 임시 폴더에 다 만든 뒤 이름을 바꾼다: 도중에 끊겨도 반쯤 만든 실행이 목록에 나오지 않는다.
    root = RunStore._make_unique_dir(base, f"{BRANCHING_PREFIX}{os.getpid()}-{next(_tmp_counter)}")
    try:
        for sub in (ENVS_DIR, CHECKPOINTS_DIR, CHAMPIONS_DIR, EXPORTS_DIR):
            (root / sub).mkdir()
        for vid, data in env_files.items():
            _st._atomic_write(root / ENVS_DIR / f"{vid}.evoenv", lambda f, d=data: f.write(d))
        if header:
            text = "".join(_st._csv_line(r) for r in (header, *stats_rows)).encode("utf-8")
            _st._atomic_write(root / STATS_FILE, lambda f: f.write(_st._BOM + text))
        _st._atomic_write(root / EVENTS_FILE, lambda f: f.write(event_bytes))
        for kind in CHAMPION_KINDS:
            for c in src.champion_generations(kind):
                if c <= g:
                    shutil.copyfile(src._champion_path(c, kind), root / CHAMPIONS_DIR / src._champion_path(c, kind).name)
        ckpt = src._checkpoint_path(g)
        with open(ckpt, "rb") as fin:
            _st._atomic_write(root / CHECKPOINTS_DIR / ckpt.name, lambda f: shutil.copyfileobj(fin, f))
        _st._atomic_write(root / RUN_INFO_FILE, lambda f: f.write(payload))
        return _move_into_place(root, base, sanitize_folder_name(branch_folder_name(info, g, now)))
    except BaseException as exc:
        shutil.rmtree(root, ignore_errors=True)
        if isinstance(exc, PermissionError):
            raise RunStoreError(S.RUNS_FILE_BUSY.format(path=exc.filename or root)) from exc
        raise


def _move_into_place(tmp: Path, base: Path, name: str) -> Path:
    """다 만든 임시 폴더를 이름, 이름_2, 이름_3 … 가운데 아직 없는 첫 이름으로 바꾼다(RunStore.create와 같은 규칙).

    윈도우에서 백신이 막 쓴 파일을 검사하느라 잡고 있으면 폴더 이름 바꾸기가 PermissionError로 실패하므로 잠깐씩
    다시 시도한다.
    """
    n = 1
    while True:
        target = base / (name if n == 1 else f"{name}_{n}")
        n += 1
        if os.path.lexists(target):
            continue
        for delay in (*_st._REPLACE_RETRY_DELAYS, None):
            try:
                os.rename(tmp, target)
                return target
            except PermissionError:
                if delay is None:
                    raise
                _st._sleep(delay)
            except OSError as exc:
                if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                    break  # 그 사이에 같은 이름이 생겼다: 다음 이름
                raise


def _read_env_files(src: RunStore, versions: list[dict[str, Any]]) -> dict[str, bytes]:
    """환경 버전 id -> 사본 바이트. 기록된 해시와 다르거나 없으면 RunStoreError."""
    out: dict[str, bytes] = {}
    for v in versions:
        vid = _text(v.get("id"))
        if vid in out:
            continue
        if not _st._VERSION_ID.fullmatch(vid):
            raise RunStoreError(S.RUNS_ENV_VERSION_MISSING.format(version=vid))
        path = src.root / ENVS_DIR / f"{vid}.evoenv"
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise RunStoreError(S.RUNS_ENV_VERSION_MISSING.format(version=vid)) from None
        if hashlib.sha256(data).hexdigest() != v.get("sha256"):
            raise RunStoreError(S.RUNS_ENV_VERSION_CHANGED.format(path=path))
        out[vid] = data
    return out


def _rows_until(header: list[str], rows: list[list[str]], g: int) -> list[list[str]]:
    if "generation" not in header:
        return []
    col = header.index("generation")
    kept = []
    for r in rows:
        gen = _st._generation_of(r, col)
        if gen is not None and gen <= g:
            kept.append(r)
    return kept


def _event_generation(event: dict[str, Any]) -> int | None:
    gen = event.get("generation")
    return gen if isinstance(gen, int) and not isinstance(gen, bool) else None


# ---------------------------------------------------------------------------
# 복제
# ---------------------------------------------------------------------------


def clone_settings(run: Path) -> dict[str, Any]:
    """복제해서 새 실행할 때 쓸 것: {"env_text": 마지막 환경 버전의 글, "env_name", "config": 학습 설정 dict}.

    config는 run.json의 config(실행을 시작할 때의 설정)다. run.json에 없으면 최신 체크포인트의 설정.
    """
    store = RunStore.open(Path(run))
    info = store.read_run_info()
    versions = [v for v in info.get("env_versions", []) if isinstance(v, dict)]
    if not versions:
        raise RunStoreError(S.RUNS_ENV_VERSION_MISSING.format(version="-"))
    env_text = store.env_text(_text(versions[-1].get("id")))
    config = info.get("config")
    if not isinstance(config, dict):
        try:
            config = store.load_checkpoint()[1].get("config", {})
        except RunStoreError:
            config = {}
    return {"env_text": env_text, "env_name": _text(info.get("env_name")), "config": dict(config)}

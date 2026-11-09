"""v3.0 D 묶음: 오류 보고 묶음(app/report/bundle.py, dialog.py, `evolab report`) — 설계 T24~T27.

EVOLAB_STATE_DIR·EVOLAB_DATA_DIR를 임시 폴더로, 바탕 화면도 임시 폴더로 바꿔 끼운다(사용자의 진짜 폴더를 건드리지 않음).
가림 시험은 가짜 사용자(홈 `C:\\Users\\홍길동`·`/home/hong`, 이름 홍길동·hong, 컴퓨터 HONG-PC·hong-laptop)로 한다.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from evolab import diagnostics, paths
from evolab.app import strings as S
from evolab.app.report import bundle
from evolab.app.report.bundle import BundleOptions

ROOT = Path(__file__).resolve().parents[1]
V11_RUN = ROOT / "tests" / "data" / "v11_run_restart" / "run"
WIN_HOME = "C:\\Users\\홍길동"
NIX_HOME = "/home/hong"
SECRETS = ("홍길동", "hong", "HONG-PC", "hong-laptop", "\\ud64d\\uae38\\ub3d9", "users\\\\")
"""묶음의 어느 글 파일에도 남으면 안 되는 것(소문자로 비교)."""


@pytest.fixture
def iso(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """임시 상태·데이터·바탕 화면 폴더, 폴더 열기는 기록만. 돌려주는 값: 이름 공간(state, logs, desk, revealed)."""
    state = tmp_path / "state"
    logs = state / "logs"
    desk = tmp_path / "desk"
    desk.mkdir()
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(state))
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setattr(bundle, "desktop_dir", lambda: desk)
    revealed: list[Path] = []
    monkeypatch.setattr(bundle, "reveal", lambda p: revealed.append(Path(p)))
    diagnostics.uninstall()

    class NS:
        pass

    ns = NS()
    ns.state, ns.logs, ns.desk, ns.revealed, ns.tmp = state, logs, desk, revealed, tmp_path  # type: ignore[attr-defined]
    yield ns
    diagnostics.uninstall()
    diagnostics.mark_clean_exit()


@pytest.fixture
def fake_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", NIX_HOME)
    monkeypatch.setenv("USERPROFILE", WIN_HOME)
    monkeypatch.setenv("USERNAME", "홍길동")
    monkeypatch.setenv("USER", "hong")
    monkeypatch.setenv("LOGNAME", "hong")
    monkeypatch.setenv("COMPUTERNAME", "HONG-PC")
    monkeypatch.delenv("HOSTNAME", raising=False)
    monkeypatch.setattr(platform, "node", lambda: "hong-laptop")
    monkeypatch.setattr(os, "getlogin", lambda: "hong")


def _write_logs(logs: Path, *, big_stderr: bool = False) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(300):
        rows.append(json.dumps({"t": "2026-12-01T14:03:22", "kind": "uncaught", "where": "gui", "pid": 1,
                                "version": "3.0.0", "message": f"OSError {i}",
                                "detail": f"File \"{WIN_HOME}\\Documents\\진화실험실\\runs\\a{i}\" on HONG-PC"},
                               ensure_ascii=i % 2 == 0))
    rows.append(json.dumps({"kind": "qt", "message": f"{NIX_HOME}/x", "detail": "hong-laptop user hong"}))
    (logs / diagnostics.ERRORS_FILE).write_text("\n".join(rows) + "\n", encoding="utf-8")
    (logs / diagnostics.FAULTS_FILE).write_text(
        f'Fatal Python error: Segmentation fault\n  File "{WIN_HOME}\\AppData\\Local\\x.py", line 3 in f\n'
        f'  File "c:\\users\\홍길동\\y.py", line 4 in g\n', encoding="utf-8")
    stderr = logs / "stderr.log"
    with open(stderr, "wb") as f:
        if big_stderr:
            chunk = (b"noise line from an old session " * 3 + b"\n") * 10_000
            while f.tell() < 50 * 1024 * 1024:
                f.write(chunk)
        f.write(f"Traceback in {NIX_HOME}/evo/x.py by hong on hong-laptop\nLAST LINE\n".encode())
    (logs / "selftest.log").write_text(f"executable: {WIN_HOME}\\evolab\\evolab-cli.exe\n", encoding="utf-8")
    state = logs.parent
    (state / "settings.ini").write_text("[lab]\nlast_env=C:/Users/홍길동/Documents/a.evoenv\n", encoding="utf-8")
    (state / paths.PATHS_FILE).write_text(f"[paths]\ndata_dir = {WIN_HOME}\\OneDrive\\문서\n", encoding="utf-8")


def _zip_texts(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as z:
        return {n: z.read(n).decode("utf-8") for n in z.namelist()}


# ---------------------------------------------------------------------------
# 가림
# ---------------------------------------------------------------------------
def test_anonymize_homes_names_and_hosts() -> None:
    homes, users, hosts = [WIN_HOME, NIX_HOME, "/"], ["홍길동", "hong", "x"], ["HONG-PC", "hong-laptop"]

    def a(text: str) -> str:
        return bundle.anonymize(text, homes=homes, users=users, hosts=hosts)

    u, h = S.REP_USER_MASK, S.REP_HOST_MASK
    assert a(f"{WIN_HOME}\\Documents") == "~\\Documents"
    assert a("c:/users/홍길동/Documents") == "~/Documents"  # 빗금 모양, 대소문자 무시
    assert a(json.dumps(WIN_HOME + "\\a")) == '"~\\\\a"'  # JSON 이스케이프(\\uXXXX)
    assert a(json.dumps(WIN_HOME + "\\a", ensure_ascii=False)) == '"~\\\\a"'
    assert a(f"{NIX_HOME}/runs and {NIX_HOME}") == "~/runs and ~"
    assert a("/home/hongkong/x") == "/home/hongkong/x"  # 다른 사용자의 홈은 그대로
    assert a("hong ran it; Hong. hongkong hong_x") == f"{u} ran it; {u}. hongkong {u}_x"  # 밑줄은 낱말 경계
    # 사용자가 지은 폴더·실행 폴더 이름(<시각>_<환경 이름>_<방법>)에 붙은 이름
    assert a("D:\\hong_자료\\진화실험실") == f"D:\\{u}_자료\\진화실험실"
    assert a("D:\\홍길동_자료\\runs") == f"D:\\{u}_자료\\runs"
    assert a("D:\\자료\\2026-11-09_1200_홍길동의 언덕_CMA-ES") == f"D:\\자료\\2026-11-09_1200_{u}의 언덕_CMA-ES"
    assert a("홍길동입니다 / 홍길동") == f"{u}입니다 / {u}"  # 한글 이름은 조사가 붙어도
    assert a(json.dumps("홍길동")) == f'"{u}"'
    assert a("on HONG-PC and hong-pc, hong-laptop") == f"on {h} and {h}, {h}"
    assert a("x marks / is root") == "x marks / is root"  # 한 글자 이름, 루트 홈은 가리지 않음


def test_anonymize_home_inside_json_of_an_os_error_text() -> None:
    """str(OSError)는 경로를 repr로 써서 역슬래시가 두 배, 그것을 errors.jsonl에 JSON으로 쓰면 네 배. 홈 폴더 이름이 사용자 이름과
    달라도(이름을 바꾼 계정) 가려야 한다."""
    home = "C:\\Users\\oldname"
    exc = FileNotFoundError(2, "No such file or directory", home + "\\Documents\\x.evoenv")
    record = json.dumps({"message": f"FileNotFoundError: {exc}", "detail": f"{exc}"}, ensure_ascii=False)
    assert "oldname" in record and "\\\\\\\\" in record  # 네 배 역슬래시
    out = bundle.anonymize(record, homes=[home], users=["newname"], hosts=[])
    assert "oldname" not in out
    assert json.loads(out)["message"] == "FileNotFoundError: [Errno 2] No such file or directory: '~\\\\Documents\\\\x.evoenv'"
    assert bundle.anonymize(str(exc), homes=[home], users=["newname"], hosts=[]) == \
        "[Errno 2] No such file or directory: '~\\\\Documents\\\\x.evoenv'"


def test_anonymize_defaults_to_this_pc(fake_user: None) -> None:
    text = bundle.anonymize(f"{WIN_HOME}\\a {NIX_HOME}/b hong 홍길동 HONG-PC hong-laptop")
    assert text == f"~\\a ~/b {S.REP_USER_MASK} {S.REP_USER_MASK} {S.REP_HOST_MASK} {S.REP_HOST_MASK}"


# ---------------------------------------------------------------------------
# T24 묶음
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["no-run", "v11-run", "editor-text"])
def test_bundle_contents_match_collect_and_hide_identity(iso: Any, fake_user: None, case: str) -> None:
    _write_logs(iso.logs, big_stderr=case == "v11-run")
    env_text = json.dumps({"version": 3, "name": "홍길동의 언덕", "note": f"{WIN_HOME}\\a"}) if case == "editor-text" \
        else None
    opts = BundleOptions(reason="menu", note="hong이 학습을 눌렀더니 HONG-PC가 멈춤",
                         run_dir=V11_RUN if case == "v11-run" else None, env_text=env_text)
    items = bundle.collect(opts)
    out = iso.tmp / "묶음.zip"
    t0 = time.perf_counter()
    written = bundle.write_report(out, opts)
    elapsed = time.perf_counter() - t0
    texts = _zip_texts(out)

    names = [i.arcname for i in items]
    assert names == [n for n, _ in written] == [n for n in texts if n != bundle.MANIFEST_NAME]
    assert list(texts)[-1] == bundle.MANIFEST_NAME
    manifest = json.loads(texts[bundle.MANIFEST_NAME])
    assert manifest["format"] == "evolab-report" and manifest["version"] == 1 and manifest["reason"] == "menu"
    assert [(x["name"], x["bytes"]) for x in manifest["items"]] == written
    assert {i.key for i in items} <= set(bundle.GROUP_KEYS)
    assert not any(n.lower().endswith(bundle.EXCLUDED_SUFFIXES) for n in texts)
    for name in ("report.txt", "logs/errors.jsonl", "logs/faults.log", "logs/stderr.log", "logs/selftest.log",
                 "settings/settings.ini", "settings/paths.ini"):
        assert name in texts, name
    for name, text in texts.items():
        low = text.lower()
        for secret in SECRETS:
            assert secret.lower() not in low, (name, secret)
    assert texts["logs/errors.jsonl"].count("\n") == bundle.ERRORS_LINES
    assert "~\\\\Documents" in texts["logs/errors.jsonl"] and S.REP_HOST_MASK in texts["logs/errors.jsonl"]
    assert texts["logs/stderr.log"].endswith("LAST LINE\n")
    assert len(texts["logs/stderr.log"].encode()) <= bundle.LOG_TAIL_BYTES
    assert sum(size for _, size in written) <= bundle.MAX_TOTAL_BYTES
    assert elapsed <= 2.0, elapsed
    report = texts["report.txt"]
    assert report.startswith(S.REP_TXT_TITLE)
    for label in S.REP_TXT_FIELDS.values():
        assert f"\r\n{label}:" in report, label
    assert f"{S.REP_TXT_FIELDS['reason']}: {S.REP_REASONS['menu']}" in report
    assert f"{S.REP_USER_MASK}이 학습을 눌렀더니 {S.REP_HOST_MASK}가 멈춤" in report
    if case == "v11-run":
        run_names = {n for n in texts if n.startswith("run/")}
        assert run_names == {"run/run.json", "run/events_tail.jsonl", "run/stats_tail.csv", "run/envs/v000.evoenv",
                             "run/files.txt"}
        assert json.loads(texts["run/run.json"]) == json.loads((V11_RUN / "run.json").read_text(encoding="utf-8"))
        stats = (V11_RUN / "stats.csv").read_bytes().decode("utf-8").splitlines(keepends=True)
        assert texts["run/stats_tail.csv"] == "".join(stats if len(stats) <= 51 else stats[:1] + stats[-50:])
        files = texts["run/files.txt"]
        assert "checkpoints/gen_000012.pkl\t" in files and "champions/gen_000001_champion.npz\t" in files
    else:
        assert not any(n.startswith("run/") for n in texts)
    if case == "editor-text":
        env = json.loads(texts["env/current.evoenv"])
        assert env["name"] == f"{S.REP_USER_MASK}의 언덕" and env["note"] == "~\\a"
    else:
        assert "env/current.evoenv" not in texts


def test_long_run_tails(iso: Any, tmp_path: Path) -> None:
    """stats.csv·events.jsonl은 RunStore처럼 바이트로 쓴다. 예전에는 `write_text`로 써서 윈도우 CI에서만 실패했다(v3.0 e5b65f7):
    윈도우의 글 모드 쓰기가 `\\n`을 `\\r\\n`으로 바꿔 디스크의 stats.csv 줄 끝이 `\\r\\r\\n`이 되었고, 묶음은 그 바이트를
    그대로 옮겼다(머리가 '\\ufeffgeneration,score\\r'로 쪼개짐)."""
    run = tmp_path / "run"
    (run / "envs").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    for v in (0, 2, 10):
        (run / "envs" / f"v{v:03d}.evoenv").write_text(json.dumps({"v": v}), encoding="utf-8")
    (run / "run.json").write_text("{}", encoding="utf-8")
    header = "\ufeffgeneration,score\r\n"
    rows = [f"{g},{g * 2}\r\n".encode() for g in range(1, 5001)]
    (run / "stats.csv").write_bytes(header.encode("utf-8") + b"".join(rows))
    event_lines = [json.dumps({"g": g}).encode() + b"\n" for g in range(1000)]
    (run / "events.jsonl").write_bytes(b"".join(event_lines))
    for g in range(5050):  # 파일 목록 상한(5000줄)을 넘게
        (run / "checkpoints" / f"gen_{g:06d}.pkl").write_bytes(b"")
    out = tmp_path / "x.zip"
    bundle.write_report(out, BundleOptions(run_dir=run))
    with zipfile.ZipFile(out) as z:  # 꼬리는 원본 바이트 그대로(줄 끝 포함)
        assert z.read("run/stats_tail.csv") == header.encode("utf-8") + b"".join(rows[-bundle.STATS_LINES:])
        assert z.read("run/events_tail.jsonl") == b"".join(event_lines[-bundle.EVENTS_LINES:])
    texts = _zip_texts(out)
    stats = texts["run/stats_tail.csv"].splitlines(keepends=True)
    assert stats[0] == header and len(stats) == 51 and stats[1] == "4951,9902\r\n" and stats[-1] == "5000,10000\r\n"
    events = texts["run/events_tail.jsonl"].splitlines()
    assert len(events) == bundle.EVENTS_LINES and events[-1] == '{"g": 999}' and events[0] == '{"g": 800}'
    assert "run/envs/v010.evoenv" in texts and "run/envs/v002.evoenv" not in texts
    files = texts["run/files.txt"].splitlines()
    assert len(files) == bundle.FILES_MAX_LINES + 1 and files[-1].startswith("... (+")


@contextlib.contextmanager
def _windows_text_mode() -> Iterator[None]:
    """윈도우의 글 모드 줄 끝 바꿈을 리눅스에서도 흉내 낸다(윈도우에서는 원래 이렇다): newline을 주지 않고 글 모드로 쓰면
    `\\n`이 os.linesep(`\\r\\n`)이 된다. 글 모드 읽기(`\\r\\n`·`\\r` → `\\n`)는 리눅스도 같다. 내장 `open`과 pathlib가 쓰는
    `io.open`을 바꾼다(바이너리 모드와 zipfile은 그대로)."""
    import builtins
    import io

    real_open = io.open

    def win_open(file: Any, mode: str = "r", buffering: int = -1, encoding: str | None = None,
                 errors: str | None = None, newline: str | None = None, closefd: bool = True,
                 opener: Any = None) -> Any:
        if "b" not in mode and newline is None and set(mode) & set("wax+"):
            newline = "\r\n"
        return real_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(builtins, "open", win_open)
        m.setattr(io, "open", win_open)
        m.setattr(os, "linesep", "\r\n")
        yield


def test_bundle_keeps_line_endings_byte_for_byte(iso: Any, fake_user: None, tmp_path: Path) -> None:
    """test_long_run_tails의 윈도우 CI 실패(v3.0 e5b65f7) 뒤에 둔 회귀 시험: 윈도우 글 모드를 흉내 낸 채로 묶음을 만든다. 묶음이
    어디서든 글 모드로 읽거나 쓰면(줄 끝을 바꾸면) 리눅스에서도 실패한다. zip 안의 바이트가 원본 바이트와 같아야 한다(꼬리는
    `\\n`으로 센 줄, 가린 곳만 다름). 줄 안의 홀로 선 `\\r`(진행 표시, 글 모드로 한 번 더 바뀐 `\\r\\r\\n`)은 줄을 나누지
    않는다. 우리가 만드는 report.txt는 메모 줄까지 모두 `\\r\\n`, files.txt·manifest.json은 `\\n`이다."""
    assert bundle._split_lines(b"a\r\r\nb\rc\nd") == [b"a\r\r\n", b"b\rc\n", b"d"]
    assert bundle._split_lines(b"") == [] and bundle._split_lines(b"\n\n") == [b"\n", b"\n"]
    iso.logs.mkdir(parents=True, exist_ok=True)
    run = tmp_path / "run"
    (run / "envs").mkdir(parents=True)
    bom_header = b"\xef\xbb\xbfgeneration,score\r\n"
    rows = [f"{g},{g * 2}\r\n".encode() for g in range(1, 121)]
    for i in (-50, -3):  # 꼬리 첫 줄과 끝 가까운 줄: 글 모드로 한 번 더 바뀐 줄 끝
        rows[i] = rows[i].replace(b"\r\n", b"\r\r\n")
    events = [json.dumps({"g": g}).encode() + (b"\r\n" if g % 7 == 0 else b"\n") for g in range(300)]
    errors = [json.dumps({"kind": "shown", "message": f"오류 {i}"}, ensure_ascii=i % 2 == 0).encode() + b"\n"
              for i in range(260)]
    stderr = (f"학습 시작\r\n진행 10%\r진행 50%\r진행 100%\r\nTraceback in {WIN_HOME}\\evo\\x.py\r\n"
              "마지막 줄(줄 끝 없음)").encode()
    sources = {
        "faults": (iso.logs / diagnostics.FAULTS_FILE,
                   b"Fatal Python error: Segmentation fault\n\nCurrent thread 0x1:\n  File \"x.py\", line 3 in f\n"),
        "selftest": (iso.logs / "selftest.log", "점검 1: 통과\r\n점검 2: 통과\r\n".encode()),
        "settings": (iso.state / "settings.ini", b"[lab]\r\nspeed=4\r\n"),
        "paths": (iso.state / paths.PATHS_FILE, b"[paths]\ndata_dir = D:/evo data\n"),
        "run": (run / "run.json", b'{\r\n  "seed": 1\r\n}\r\n'),
        "envs": (run / "envs" / "v001.evoenv", b'{\r\n  "version": 3\r\n}'),
    }
    for path, data in sources.values():
        path.write_bytes(data)
    (iso.logs / diagnostics.ERRORS_FILE).write_bytes(b"".join(errors))
    (iso.logs / "stderr.log").write_bytes(stderr)
    (run / "stats.csv").write_bytes(bom_header + b"".join(rows))
    (run / "events.jsonl").write_bytes(b"".join(events))
    env_text = '{\n  "name": "언덕"\r\n}\n'
    out = tmp_path / "bytes.zip"
    with _windows_text_mode():
        probe = tmp_path / "probe.txt"
        probe.write_text("a\r\nb\n", encoding="utf-8")
        assert probe.read_bytes() == b"a\r\r\nb\r\n"  # 흉내가 켜졌다: 글 모드로 쓰면 이렇게 바뀐다
        written = bundle.write_report(out, BundleOptions(note="첫 줄\n둘째 줄\r\n셋째 줄", run_dir=run,
                                                         env_text=env_text))
    with zipfile.ZipFile(out) as z:
        got = {n: z.read(n) for n in z.namelist()}

    assert got["logs/errors.jsonl"] == b"".join(errors[-bundle.ERRORS_LINES:])
    assert got["logs/stderr.log"] == stderr.replace(WIN_HOME.encode(), b"~")
    assert got["logs/faults.log"] == sources["faults"][1] and got["logs/selftest.log"] == sources["selftest"][1]
    assert got["settings/settings.ini"] == sources["settings"][1] and got["settings/paths.ini"] == sources["paths"][1]
    assert got["run/run.json"] == sources["run"][1] and got["run/envs/v001.evoenv"] == sources["envs"][1]
    assert got["run/stats_tail.csv"] == bom_header + b"".join(rows[-bundle.STATS_LINES:])
    assert got["run/events_tail.jsonl"] == b"".join(events[-bundle.EVENTS_LINES:])
    assert got["env/current.evoenv"] == env_text.encode()
    report = got["report.txt"]
    assert report.count(b"\n") == report.count(b"\r\n") and b"\r\r" not in report
    assert f"{S.REP_TXT_FIELDS['note']}:\r\n첫 줄\r\n둘째 줄\r\n셋째 줄\r\n".encode() in report
    for name in ("run/files.txt", bundle.MANIFEST_NAME):
        assert b"\r" not in got[name], name
    manifest = json.loads(got[bundle.MANIFEST_NAME])
    assert [(x["name"], x["bytes"]) for x in manifest["items"]] == written
    assert written == [(n, len(data)) for n, data in got.items() if n != bundle.MANIFEST_NAME]


def test_total_size_cap_drops_oldest_logs_first(iso: Any, tmp_path: Path) -> None:
    _write_logs(iso.logs)
    blob = "x" * (1536 * 1024) + "\n"  # 셋이면 4.5MB: 로그를 더하면 5MB를 넘는다
    (iso.state / "settings.ini").write_text(blob, encoding="utf-8")
    (iso.state / paths.PATHS_FILE).write_text(blob, encoding="utf-8")
    (iso.logs / "stderr.log").write_text(("e" * 99 + "\n") * 3000, encoding="utf-8")
    (iso.logs / "selftest.log").write_text(("s" * 99 + "\n") * 3000, encoding="utf-8")
    now = time.time()
    for i, name in enumerate(("stderr.log", "selftest.log", diagnostics.FAULTS_FILE, diagnostics.ERRORS_FILE)):
        os.utime(iso.logs / name, (now - 1000 + i, now - 1000 + i))  # stderr가 가장 오래됨
    opts = BundleOptions(env_text=blob)
    items = bundle.collect(opts)
    names = [i.arcname for i in items]
    assert sum(i.size for i in items) <= bundle.MAX_TOTAL_BYTES
    assert names == ["report.txt", "logs/errors.jsonl", "logs/faults.log", "logs/selftest.log", "settings/settings.ini",
                     "settings/paths.ini", "env/current.evoenv"]  # 가장 오래된 stderr.log 하나만 빠짐
    written = bundle.write_report(tmp_path / "cap.zip", opts)
    assert [n for n, _ in written] == names


def test_write_report_uses_given_items_only(iso: Any, tmp_path: Path) -> None:
    _write_logs(iso.logs)
    opts = BundleOptions(run_dir=V11_RUN, env_text="{}")
    items = [i for i in bundle.collect(opts) if i.key in ("report", "env")]
    written = bundle.write_report(tmp_path / "some.zip", opts, items)
    assert [n for n, _ in written] == ["report.txt", "env/current.evoenv"]
    off = bundle.collect(replace(opts, include_run=False, include_env=False))
    assert {i.key: i.default_on for i in off}["run"] is False and {i.key: i.default_on for i in off}["env"] is False
    written = bundle.write_report(tmp_path / "default.zip", replace(opts, include_run=False, include_env=False))
    assert not any(n.startswith(("run/", "env/")) for n, _ in written)


def test_write_failure_leaves_no_files(iso: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*a: Any, **k: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(zipfile.ZipFile, "writestr", broken)
    out = tmp_path / "fail" / "x.zip"
    with pytest.raises(OSError):
        bundle.write_report(out, BundleOptions())
    assert list(out.parent.iterdir()) == []
    assert bundle.reason_text(OSError(28, "No space left on device")) == S.ERR_DISK_FULL


def test_default_bundle_path(iso: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    when = datetime(2026, 12, 1, 14, 3)
    first = bundle.default_bundle_path(when)
    assert first == iso.desk / "진화실험실_오류보고_20261201_1403.zip"
    first.write_bytes(b"")
    assert bundle.default_bundle_path(when).name == "진화실험실_오류보고_20261201_1403_2.zip"
    monkeypatch.setattr(bundle, "desktop_dir", lambda: None)
    assert bundle.default_bundle_path(when).parent == paths.data_dir() == iso.tmp / "data"


def test_human_size_and_reason_text() -> None:
    assert [bundle.human_size(n) for n in (0, 1023, 1024, 41_000, 5 * 1024 * 1024)] == ["0B", "1023B", "1KB", "40KB",
                                                                                        "5.0MB"]
    assert bundle.reason_text(PermissionError(13, "denied")) == S.ERR_PERMISSION
    assert bundle.reason_text(FileNotFoundError(2, "x", "/a/b")) == S.ERR_NOT_FOUND.format(path="/a/b")
    assert bundle.reason_text(MemoryError()) == S.ERR_MEMORY
    assert bundle.reason_text(RuntimeError("raw english")) == "RuntimeError"


def test_self_check_for_selftest(iso: Any, tmp_path: Path) -> None:
    _write_logs(iso.logs)
    detail = bundle.self_check(tmp_path)
    assert detail.startswith("files=") and (tmp_path / "report-check.zip").is_file()
    assert not list(iso.logs.glob(".write-check-*"))


# ---------------------------------------------------------------------------
# T25 명령줄
# ---------------------------------------------------------------------------
def test_cli_report_runs_without_qt(tmp_path: Path) -> None:
    out = tmp_path / "보고" / "묶음.zip"
    code = ("import json, sys\nsys.argv = ['evolab', 'report', '--out', sys.argv[1], '--run', sys.argv[2]]\n"
            "from evolab.cli import main\nrc = main()\n"
            "print(json.dumps({'rc': rc, 'qt': sorted(m for m in sys.modules if m.split('.')[0] in "
            "('PySide6', 'shiboken6'))}))\n")
    env = {k: v for k, v in os.environ.items() if k != paths.DATA_DIR_ENV}  # 데이터 폴더도 Qt 없이 찾는다
    env.update({paths.STATE_DIR_ENV: str(tmp_path / "state"), "PYTHONIOENCODING": "utf-8"})
    proc = subprocess.run([sys.executable, "-c", code, str(out), str(V11_RUN)], capture_output=True, text=True,
                          encoding="utf-8", env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    first, last = proc.stdout.strip().splitlines()[0], proc.stdout.strip().splitlines()[-1]
    assert json.loads(last) == {"rc": 0, "qt": []}
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    assert first == S.CLI_REPORT_DONE.format(path=out, count=len(names) - 1,
                                             size=bundle.human_size(out.stat().st_size))
    assert "report.txt" in names and "run/run.json" in names
    assert not any(n.endswith(bundle.EXCLUDED_SUFFIXES) for n in names)
    assert json.loads(zipfile.ZipFile(out).read(bundle.MANIFEST_NAME))["reason"] == "cli"


def test_python_m_evolab_report_and_bad_run(tmp_path: Path) -> None:
    env = {**os.environ, paths.STATE_DIR_ENV: str(tmp_path / "state"), paths.DATA_DIR_ENV: str(tmp_path / "data"),
           "PYTHONIOENCODING": "utf-8"}
    out = tmp_path / "x.zip"
    proc = subprocess.run([sys.executable, "-m", "evolab", "report", "--out", str(out)], capture_output=True,
                          text=True, encoding="utf-8", env=env, timeout=120, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert out.is_file() and (tmp_path / "state" / "logs" / diagnostics.FAULTS_FILE).exists()  # 진입점이 켰다
    missing = tmp_path / "없는 실행"
    proc = subprocess.run([sys.executable, "-m", "evolab", "report", "--out", str(tmp_path / "y.zip"), "--run",
                           str(missing)], capture_output=True, text=True, encoding="utf-8", env=env, timeout=120,
                          cwd=tmp_path)
    assert proc.returncode == 2
    assert proc.stderr.strip() == S.REP_FAILED.format(reason=S.ERR_NOT_FOUND.format(path=missing))
    assert not (tmp_path / "y.zip").exists()


def test_report_command_in_process(iso: Any, capsys: pytest.CaptureFixture[str]) -> None:
    from evolab import cli

    assert cli.main(["report", "--open"]) == 0  # --out 없음: (가짜) 바탕 화면
    (made,) = list(iso.desk.glob("*.zip"))
    assert made.name.startswith("진화실험실_오류보고_") and iso.revealed == [made]
    assert capsys.readouterr().out.startswith(S.CLI_REPORT_DONE.split("{")[0])
    blocker = iso.tmp / "file"
    blocker.write_text("", encoding="utf-8")
    assert bundle.report_command(blocker / "x.zip", None) == 1  # 폴더 자리에 파일: 쓰기 실패
    assert capsys.readouterr().err.startswith(S.REP_FAILED.split("{")[0])
    assert sys.excepthook is not diagnostics._excepthook  # 인수를 주고 부르면 훅을 걸지 않는다


# ---------------------------------------------------------------------------
# T26 대화 상자
# ---------------------------------------------------------------------------
def test_report_dialog_preview_and_save(qtbot: Any, iso: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QFileDialog

    from evolab.app.report import dialog

    _write_logs(iso.logs)
    dlg = dialog.ReportDialog(None, BundleOptions(reason="error", run_dir=V11_RUN, env_text='{"a": 1}'))
    qtbot.addWidget(dlg)
    assert all(box.isEnabled() and box.isChecked() for box in dlg.checks.values())
    every = [i.arcname for i in dlg.items]
    assert every == [i.arcname for i in bundle.collect(BundleOptions(run_dir=V11_RUN, env_text='{"a": 1}'))]
    assert dlg.files_label.text() == S.REP_FILES.format(count=len(every),
                                                        size=bundle.human_size(sum(i.size for i in dlg.items)))
    assert dlg.file_list.count() == len(every)
    dlg.checks["run"].setChecked(False)
    assert not any(i.arcname.startswith("run/") for i in dlg.items) and dlg.file_list.count() < len(every)
    dlg.checks["stderr"].setChecked(False)
    dlg.checks["env"].setChecked(False)
    kept = [i.arcname for i in dlg.items]
    assert "logs/stderr.log" not in kept and "env/current.evoenv" not in kept
    assert S.REP_FILES.format(count=len(kept), size="").split(",")[0] in dlg.files_label.text()
    dlg.note.setPlainText("학습 시작 직후 멈춤")

    asked: list[str] = []

    def fake_save(parent: Any, title: str, start: str, flt: str) -> tuple[str, str]:
        asked.append(start)
        return str(iso.tmp / "보낼 묶음"), flt

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fake_save)
    path = dlg.save()
    assert path == iso.tmp / "보낼 묶음.zip" and dlg.saved_path == path
    assert Path(asked[0]).parent == iso.desk
    texts = _zip_texts(path)
    assert [n for n in texts if n != bundle.MANIFEST_NAME] == kept
    assert "학습 시작 직후 멈춤" in texts["report.txt"]
    assert S.REP_REASONS["error"] in texts["report.txt"]
    assert dlg.status.text() == S.REP_DONE.format(path=path) and dlg.btn_open_folder.isVisibleTo(dlg)
    dlg.btn_open_folder.click()
    assert iso.revealed == [path]


def test_report_dialog_grey_when_missing_and_failure(qtbot: Any, iso: Any) -> None:
    from evolab.app.report import dialog

    dlg = dialog.ReportDialog(None, BundleOptions())
    qtbot.addWidget(dlg)
    assert dlg.checks["report"].isEnabled() and dlg.checks["report"].isChecked()
    for key in ("run", "env", "faults", "stderr"):
        assert not dlg.checks[key].isEnabled() and not dlg.checks[key].isChecked(), key
    blocker = iso.tmp / "file"
    blocker.write_text("", encoding="utf-8")
    assert dlg.save_to(blocker / "x.zip") is None and dlg.saved_path is None
    assert dlg.status.text().startswith(S.REP_FAILED.split("{")[0]) and not dlg.btn_open_folder.isVisibleTo(dlg)
    assert [r["kind"] for r in diagnostics.recent()] == ["shown"]


def test_open_report_dialog_returns_saved_path(qtbot: Any, iso: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.report import dialog

    target = iso.tmp / "y.zip"
    monkeypatch.setattr(dialog.ReportDialog, "exec", lambda self: (self.save_to(target), 1)[1])
    assert dialog.open_report_dialog(None, BundleOptions(reason="tour")) == target
    assert json.loads(zipfile.ZipFile(target).read(bundle.MANIFEST_NAME))["reason"] == "tour"
    monkeypatch.setattr(dialog.ReportDialog, "exec", lambda self: 0)
    assert dialog.open_report_dialog(None, BundleOptions()) is None


def test_crash_box_status_line_and_previous_question(qtbot: Any, iso: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QWidget

    from evolab.app.report import dialog

    opened: list[tuple[Any, BundleOptions]] = []
    monkeypatch.setattr(dialog, "open_report_dialog", lambda parent, opts: opened.append((parent, opts)) or
                        iso.tmp / "p.zip")

    class Lab(QWidget):
        run_dir = V11_RUN
        env_text = "{}"

    class Main(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.lab = Lab()
            self.setCentralWidget(self.lab)

    main = Main()
    qtbot.addWidget(main)
    main.show()
    dialog.on_crash("again", "ZeroDivisionError: A")
    assert main.statusBar().currentMessage() == S.DIAG_CRASH_AGAIN.format(line="ZeroDivisionError: A")
    dialog.on_crash("more", "KeyError: 'B'")
    assert main.statusBar().currentMessage() == S.DIAG_CRASH_MORE.format(line="KeyError: 'B'")

    dialog.on_crash("box", "ValueError: C")
    boxes = [w for w in QApplication.topLevelWidgets() if isinstance(w, QMessageBox) and w.isVisible()]
    (box,) = boxes
    assert box.windowTitle() == S.DIAG_CRASH_TITLE and box.text() == S.DIAG_CRASH_TEXT
    assert box.detailedText() == "ValueError: C" and not box.isModal()
    (report,) = [b for b in box.buttons() if b.text() == S.ERR_REPORT_BUTTON]
    report.click()
    qtbot.waitUntil(lambda: bool(opened), timeout=3000)
    parent, opts = opened[0]
    assert parent is main and opts.reason == "crash" and opts.run_dir == V11_RUN and opts.env_text == "{}"

    def answer(text: str) -> None:
        modal = QApplication.activeModalWidget()
        if isinstance(modal, QMessageBox):
            (button,) = [b for b in modal.buttons() if b.text() == text]
            button.click()
        else:
            QTimer.singleShot(20, lambda: answer(text))

    QTimer.singleShot(0, lambda: answer(S.DIAG_PREVIOUS_YES))
    assert dialog.ask_previous(main) == iso.tmp / "p.zip"
    assert opened[-1][1].reason == "previous"
    QTimer.singleShot(0, lambda: answer(S.DIAG_PREVIOUS_NO))
    assert dialog.ask_previous(main) is None and len(opened) == 2


# ---------------------------------------------------------------------------
# T27 시작 실패
# ---------------------------------------------------------------------------
def test_startup_failure_makes_bundle_and_native_box(iso: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab import winapp

    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(winapp, "message_box", lambda title, text: shown.append((title, text)))
    try:
        raise RuntimeError("창 만들기 실패")
    except RuntimeError as exc:
        assert bundle.report_startup_failure(exc) == 1
    (made,) = list(iso.desk.glob("*.zip"))
    assert json.loads(zipfile.ZipFile(made).read(bundle.MANIFEST_NAME))["reason"] == "startup"
    assert shown == [(S.DIAG_STARTUP_TITLE, S.DIAG_STARTUP_TEXT.format(reason="RuntimeError", path=made))]
    (rec,) = diagnostics.recent()
    assert rec["kind"] == "uncaught" and "창 만들기 실패" in rec["detail"]
    texts = _zip_texts(made)
    assert "창 만들기 실패" in texts["logs/errors.jsonl"] and S.REP_REASONS["startup"] in texts["report.txt"]

    def no_room(*a: Any, **k: Any) -> Any:
        raise PermissionError(13, "denied")

    monkeypatch.setattr(bundle, "write_report", no_room)
    shown.clear()
    assert bundle.report_startup_failure(ValueError("x")) == 1
    assert shown == [(S.DIAG_STARTUP_TITLE, S.DIAG_STARTUP_NO_BUNDLE.format(reason="ValueError",
                                                                           error=S.ERR_PERMISSION))]


@pytest.mark.qt_no_exception_capture
@pytest.mark.no_qt_log
def test_run_gui_startup_failure(qtbot: Any, iso: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """T27 그대로: MainWindow를 예외로 바꾼 run_gui() → 가짜 바탕 화면에 묶음, 가짜 message_box, 종료 코드 1."""
    from evolab import winapp
    from evolab.app import main as main_mod

    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(winapp, "message_box", lambda title, text: shown.append((title, text)))
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)

    def broken() -> Any:
        raise RuntimeError("no window")

    monkeypatch.setattr(main_mod, "MainWindow", broken)
    assert main_mod.run_gui() == 1
    (made,) = list(iso.desk.glob("*.zip"))
    assert json.loads(zipfile.ZipFile(made).read(bundle.MANIFEST_NAME))["reason"] == "startup"
    assert [t for t, _ in shown] == [S.DIAG_STARTUP_TITLE]
    assert not list(iso.logs.glob("running-*.flag"))


def test_open_for_uses_main_window_lab(qtbot: Any, iso: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """도움말 메뉴·오류 상자의 「오류 보고 묶음 만들기…」(C·I 묶음이 부르는 open_for): 주 창 실험실의 실행·환경을 넣는다."""
    from PySide6.QtWidgets import QMainWindow, QWidget

    from evolab.app.report import dialog

    seen: list[tuple[Any, BundleOptions]] = []
    monkeypatch.setattr(dialog, "open_report_dialog", lambda parent, opts: seen.append((parent, opts)) or None)

    class Lab(QWidget):
        run_dir = V11_RUN
        env_text = '{"x": 1}'

    class Main(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.lab = Lab()

    main = Main()
    qtbot.addWidget(main)
    main.show()
    assert dialog.main_window() is main
    assert dialog.open_for(main, "error") is None
    (parent, opts), = seen
    assert parent is main and opts == BundleOptions(reason="error", run_dir=V11_RUN, env_text='{"x": 1}')
    assert dialog.options_for(QWidget(), "menu") == BundleOptions(reason="menu")


def test_bundle_hides_names_planted_like_windows_ci(iso: Any, fake_user: None) -> None:
    """W2(윈도우 CI)가 심는 모양을 여기서도: 홈 아래 「<사용자>_자료」 실행 폴더, errors.jsonl의 str(OSError)(역슬래시 네 개),
    stderr.log의 그대로 쓴 경로. 묶음의 어느 글에도 이름·홈 경로가 남지 않는다."""
    import shutil

    run = iso.tmp / "홍길동_자료" / "run"
    shutil.copytree(V11_RUN, run)
    iso.logs.mkdir(parents=True, exist_ok=True)
    path = WIN_HOME + "\\홍길동_자료\\x.evoenv"
    exc = FileNotFoundError(2, "No such file or directory", path)
    rec = {"t": "2026-11-09T10:00:00", "kind": "shown", "where": "cli", "pid": 0, "version": "ci",
           "message": f"ci plant: {path} / FileNotFoundError: {exc}", "detail": f"user 홍길동 at {path}"}
    (iso.logs / diagnostics.ERRORS_FILE).write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    (iso.logs / "stderr.log").write_text(f"ci plant: {path} by 홍길동\n", encoding="utf-8")
    out = iso.tmp / "w2.zip"
    bundle.write_report(out, BundleOptions(reason="cli", run_dir=run))
    texts = _zip_texts(out)
    assert any(n.startswith("run/") for n in texts)
    joined = "\n".join(texts.values())
    assert joined.count("ci plant") >= 2
    for secret in SECRETS:
        assert secret.casefold() not in joined.casefold(), secret

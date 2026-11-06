"""M2: 실행 폴더 저장소 (run.json, 환경 버전, stats.csv, events.jsonl, 체크포인트, 챔피언, 이어하기 정리)."""

import csv
import io
import json
import os
import pickle
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab.app import strings as S
from evolab.runs import store as store_mod
from evolab.runs.store import CheckpointPolicy, RunStore, RunStoreError, sanitize_folder_name

BOM = b"\xef\xbb\xbf"
NAME = "2026-11-07_1530_달 착륙선_CMA-ES"


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """한글과 공백이 들어간 기본 위치 (문서\\진화실험실\\runs 흉내)."""
    return tmp_path / "홍길동 OneDrive" / "문서" / "진화실험실" / "runs"


@pytest.fixture
def store(base: Path) -> RunStore:
    return RunStore.create(base, NAME, {"env_name": "달 착륙선", "method": "cmaes", "config": {"popsize": 32}})


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_mod, "_REPLACE_RETRY_DELAYS", (0.0, 0.0))
    monkeypatch.setattr(store_mod, "_APPEND_RETRY_DELAYS", (0.0,))


def leftovers(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.rglob("*.tmp"))


def stats_row(gen: int, **extra: Any) -> dict[str, Any]:
    return {"generation": gen, "evaluations": 160 * (gen + 1), "score_best": 0.1 * gen + 0.2, "sigma": np.float64(0.5 / (gen + 1)),
            "validation_best": "" if gen % 2 else -12.5, "block_착지": np.float32(0.1), **extra}


# ---------------------------------------------------------------------------
# 만들기, 열기, 이름
# ---------------------------------------------------------------------------


def test_create_open_round_trip(base: Path, store: RunStore) -> None:
    assert store.root == base / NAME
    for sub in ("envs", "checkpoints", "champions", "exports"):
        assert (store.root / sub).is_dir()
    raw = (store.root / "run.json").read_bytes()
    assert not raw.startswith(BOM)
    text = raw.decode("utf-8")
    assert "달 착륙선" in text and "\\u" not in text  # ensure_ascii=False
    assert text.startswith('{\n  "env_name"')  # 들여쓰기 2
    info = json.loads(text)
    assert info["env_name"] == "달 착륙선" and info["config"] == {"popsize": 32}
    assert info["env_versions"] == [] and info["protected_checkpoints"] == [] and info["store_format"] == 1
    again = RunStore.open(store.root)
    assert again.read_run_info() == info
    assert RunStore.open(store.root / "run.json").root == store.root  # run.json 경로를 줘도 된다
    assert RunStore.open(str(store.root)).root == store.root  # type: ignore[arg-type]
    assert leftovers(store.root) == []


def test_update_run_info_merges(store: RunStore) -> None:
    store.update_run_info(memo="메모 ✓", generations=np.int64(12), config={"popsize": 64})
    info = store.read_run_info()
    assert info["memo"] == "메모 ✓" and info["generations"] == 12 and info["config"] == {"popsize": 64}
    assert info["env_name"] == "달 착륙선"
    assert leftovers(store.root) == []


def test_create_with_unserializable_info_leaves_nothing(base: Path) -> None:
    with pytest.raises(TypeError):
        RunStore.create(base, "망가진 정보", {"bad": object()})
    assert not (base / "망가진 정보").exists()


def test_open_rejects_non_runs(tmp_path: Path) -> None:
    empty = tmp_path / "빈 폴더"
    empty.mkdir()
    with pytest.raises(RunStoreError, match="run.json"):
        RunStore.open(empty)
    (empty / "run.json").write_text("{깨진", encoding="utf-8")
    with pytest.raises(RunStoreError):
        RunStore.open(empty)
    (empty / "run.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(RunStoreError):
        RunStore.open(empty)
    (empty / "run.json").write_text('{"store_format": 99}', encoding="utf-8")
    with pytest.raises(RunStoreError, match="새 버전"):
        RunStore.open(empty)
    # 메모장이 BOM을 붙여 저장해도 읽는다
    (empty / "run.json").write_bytes(BOM + '{"env_name": "착륙선"}'.encode())
    assert RunStore.open(empty).read_run_info() == {"env_name": "착륙선"}


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        (NAME, NAME),
        ('a<b>c:d"e/f\\g|h?i*j', "a_b_c_d_e_f_g_h_i_j"),
        ("착륙선. . ", "착륙선"),
        ("  앞뒤 공백  ", "앞뒤 공백"),
        ("줄\n바꿈\t탭", "줄_바꿈_탭"),
        ("CON", "_CON"),
        ("con.txt", "_con.txt"),
        ("Nul", "_Nul"),
        ("COM1", "_COM1"),
        ("lpt9.log", "_lpt9.log"),
        ("COM¹", "_COM¹"),
        ("CONSOLE", "CONSOLE"),
        ("COM10", "COM10"),
        ("", S.RUNS_DEFAULT_FOLDER_NAME),
        ("...", S.RUNS_DEFAULT_FOLDER_NAME),
        (unicodedata.normalize("NFD", "한글 이름"), "한글 이름"),
    ],
)
def test_sanitize_folder_name(raw: str, want: str) -> None:
    assert sanitize_folder_name(raw) == want


def test_sanitize_long_name_is_cut_and_still_valid() -> None:
    name = sanitize_folder_name("가" * 79 + ". 나머지는 잘림")
    assert name == "가" * 79
    assert len(sanitize_folder_name("x" * 500)) == store_mod.MAX_FOLDER_NAME


def test_create_appends_suffix_when_taken(base: Path) -> None:
    roots = [RunStore.create(base, NAME, {"i": i}).root for i in range(3)]
    assert [r.name for r in roots] == [NAME, f"{NAME}_2", f"{NAME}_3"]
    # 고친 이름이 같아지는 경우도 겹치지 않는다
    a = RunStore.create(base, "가:나", {}).root
    b = RunStore.create(base, "가?나", {}).root
    assert (a.name, b.name) == ("가_나", "가_나_2")
    (base / "파일").write_text("x", encoding="utf-8")  # 같은 이름의 파일이 있어도
    assert RunStore.create(base, "파일", {}).root.name == "파일_2"
    assert [RunStore.open(r).read_run_info()["i"] for r in roots] == [0, 1, 2]


# ---------------------------------------------------------------------------
# 환경 버전
# ---------------------------------------------------------------------------


def test_env_versions_dedupe_and_round_trip(store: RunStore) -> None:
    text_a = '﻿{"name": "달 착륙선",\r\n "version": 1}\n'
    text_b = '{"name": "달 착륙선 (중력 수정)"}'
    assert store.add_env_version(text_a, 0) == "v000"
    assert store.add_env_version(text_a, 5) == "v000"  # 같은 내용: 새 파일도 새 기록도 없음
    assert store.add_env_version(text_b, 20) == "v001"
    assert store.add_env_version(text_a, 40) == "v000"  # 되돌림: 파일은 그대로, 적용 기록만 추가
    assert sorted(p.name for p in (store.root / "envs").iterdir()) == ["v000.evoenv", "v001.evoenv"]
    versions = store.read_run_info()["env_versions"]
    assert [(v["id"], v["from_generation"]) for v in versions] == [("v000", 0), ("v001", 20), ("v000", 40)]
    import hashlib

    assert versions[0]["sha256"] == hashlib.sha256((store.root / "envs" / "v000.evoenv").read_bytes()).hexdigest()
    assert store.env_text("v000") == text_a  # BOM 글자와 \r\n까지 그대로
    assert RunStore.open(store.root).env_text("v001") == text_b
    assert store.add_env_version("세 번째", 41) == "v002"
    with pytest.raises(RunStoreError):
        store.env_text("v009")
    with pytest.raises(RunStoreError):
        store.env_text("../run")
    (store.root / "envs" / "v001.evoenv").write_text("손으로 고침", encoding="utf-8")
    with pytest.raises(RunStoreError, match="해시"):
        store.env_text("v001")
    assert leftovers(store.root) == []


# ---------------------------------------------------------------------------
# stats.csv
# ---------------------------------------------------------------------------


def test_stats_header_values_and_excel_encoding(store: RunStore) -> None:
    for g in range(4):
        store.append_stats(stats_row(g))
    raw = (store.root / "stats.csv").read_bytes()
    assert raw.startswith(BOM) and raw.count(BOM) == 1  # 엑셀(한국어 윈도우)이 UTF-8로 알아보는 BOM, 한 번만
    assert raw.endswith(b"\r\n") and raw.count(b"\r\n") == 5
    text = raw.decode("utf-8-sig")
    lines = text.splitlines()
    assert lines[0] == "generation,evaluations,score_best,sigma,validation_best,block_착지"
    assert lines[2] == "1,320,0.30000000000000004,0.25,,0.10000000149011612"  # repr 정밀도, 빈 칸
    rows = store.read_stats()
    assert len(rows) == 4 and rows[3]["generation"] == "3"
    assert float(rows[3]["score_best"]) == 0.1 * 3 + 0.2  # 글자에서 원래 실수로 정확히 되돌아온다
    assert float(rows[2]["sigma"]) == 0.5 / 3
    assert rows[0]["validation_best"] == "-12.5" and rows[1]["validation_best"] == ""
    # 엑셀처럼 BOM을 인코딩 표시로 읽으면 열이 그대로 나뉜다
    parsed = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    assert parsed[0][-1] == "block_착지" and len({len(r) for r in parsed}) == 1


def test_stats_keys_must_match(store: RunStore) -> None:
    store.append_stats({"generation": 0, "a": 1.5, "b": "x,y"})
    store.append_stats({"b": 'say "hi"', "generation": 1, "a": 2})  # 순서가 달라도 같은 키면 첫 줄 순서로
    with pytest.raises(ValueError, match="빠진 열: a"):
        store.append_stats({"generation": 2, "b": 1})
    with pytest.raises(ValueError, match="새 열: c"):
        store.append_stats({"generation": 2, "a": 1, "b": 1, "c": 1})
    with pytest.raises(TypeError):
        store.append_stats({"generation": 2, "a": [1, 2], "b": 1})
    other = RunStore.open(store.root)  # 열 이름은 파일에서 다시 읽는다
    with pytest.raises(ValueError):
        other.append_stats({"generation": 2})
    other.append_stats({"generation": 2, "a": np.int32(7), "b": None})
    assert store.read_stats() == [
        {"generation": "0", "a": "1.5", "b": "x,y"},
        {"generation": "1", "a": "2", "b": 'say "hi"'},
        {"generation": "2", "a": "7", "b": ""},
    ]


def test_stats_torn_last_line_is_dropped_and_repaired(store: RunStore) -> None:
    for g in range(3):
        store.append_stats(stats_row(g))
    path = store.root / "stats.csv"
    good = path.read_bytes()
    with open(path, "ab") as f:  # 전원이 나가 마지막 줄이 잘리고 NUL로 채워진 파일
        f.write(b"3,640,0.5" + b"\x00" * 50)
    assert [r["generation"] for r in store.read_stats()] == ["0", "1", "2"]
    reopened = RunStore.open(store.root)
    reopened.append_stats(stats_row(3))
    assert path.read_bytes().startswith(good)
    assert [r["generation"] for r in reopened.read_stats()] == ["0", "1", "2", "3"]
    assert b"\x00" not in path.read_bytes()


def test_stats_header_torn_is_rewritten(store: RunStore) -> None:
    (store.root / "stats.csv").write_bytes(BOM + b"generation,sco")
    assert store.read_stats() == []
    store.append_stats({"generation": 0, "score": 1.0})
    assert store.read_stats() == [{"generation": "0", "score": "1.0"}]
    (store.root / "stats.csv").unlink()  # 학습 중에 사용자가 지워도 열 이름부터 다시 쓴다
    store.append_stats({"generation": 1, "score": 2.0})
    assert (store.root / "stats.csv").read_bytes().startswith(BOM + b"generation,score\r\n")
    assert store.read_stats() == [{"generation": "1", "score": "2.0"}]


def test_stats_locked_by_excel_goes_to_pending_and_is_merged(store: RunStore, monkeypatch: pytest.MonkeyPatch) -> None:
    store.append_stats(stats_row(0))
    stats = store.root / "stats.csv"
    real_open = RunStore._open_append
    locked = {"on": True}

    def fake_open(self: RunStore, path: Path) -> Any:
        if locked["on"] and path == stats:
            raise PermissionError(13, "The process cannot access the file because it is being used by another process", str(path))
        return real_open(self, path)

    monkeypatch.setattr(RunStore, "_open_append", fake_open)
    before = stats.read_bytes()
    for g in (1, 2):
        store.append_stats(stats_row(g))  # 학습은 멈추지 않는다
    assert stats.read_bytes() == before
    assert store.pending_stats == 2 and (store.root / "stats.csv.pending").is_file()
    assert [r["generation"] for r in store.read_stats()] == ["0", "1", "2"]  # 화면에는 다 보인다
    assert S.RUNS_STATS_PENDING.format(count=store.pending_stats)
    locked["on"] = False  # 엑셀을 닫음
    store.append_stats(stats_row(3))
    assert store.pending_stats == 0 and not (store.root / "stats.csv.pending").exists()
    lines = stats.read_bytes().decode("utf-8-sig").splitlines()
    assert [ln.split(",")[0] for ln in lines] == ["generation", "0", "1", "2", "3"]


# ---------------------------------------------------------------------------
# events.jsonl
# ---------------------------------------------------------------------------


def test_events_round_trip(store: RunStore) -> None:
    store.append_event("start", 0, workers=3)
    store.append_event("env_edit", 12, version="v001", note="중력 1.62 → 3.0\n두 줄", score=np.float64(-3.5), seeds=np.arange(3))
    store.append_event("stop", 20, reason="요청", path=store.root)
    with pytest.raises(TypeError):
        store.append_event("x", 1, time=3)
    path = store.root / "events.jsonl"
    raw = path.read_bytes()
    assert raw.count(b"\n") == 3 and "중력".encode() in raw
    events = store.read_events()
    assert [(e["kind"], e["generation"]) for e in events] == [("start", 0), ("env_edit", 12), ("stop", 20)]
    assert events[1]["note"] == "중력 1.62 → 3.0\n두 줄" and events[1]["score"] == -3.5 and events[1]["seeds"] == [0, 1, 2]
    assert events[2]["path"] == str(store.root)
    assert all("time" in e for e in events)
    with open(path, "ab") as f:
        f.write(b'{"kind": "half')  # 잘린 줄
    assert len(store.read_events()) == 3
    RunStore.open(store.root).append_event("resume", 20)
    assert [e["kind"] for e in store.read_events()] == ["start", "env_edit", "stop", "resume"]


# ---------------------------------------------------------------------------
# 체크포인트
# ---------------------------------------------------------------------------


def make_state(gen: int) -> dict[str, Any]:
    rng = np.random.default_rng(gen)
    return {"generation": gen, "mean": rng.normal(size=211), "rng": np.random.RandomState(gen).get_state(), "이름": "상태"}


@pytest.mark.parametrize("protect_first", [False, True])
def test_checkpoint_retention(store: RunStore, protect_first: bool) -> None:
    if protect_first:
        store.protect(37)
    for g in range(1, 251):
        store.save_checkpoint(g, {"g": g})
        if g == 37 and not protect_first:
            store.protect(37)
        assert len(store.checkpoints()) <= 5 + g // 100 + 1
    assert store.checkpoints() == [37, 100, 200, 246, 247, 248, 249, 250]
    assert store.read_run_info()["protected_checkpoints"] == [37]
    assert store.load_checkpoint(37) == (37, {"g": 37})
    assert leftovers(store.root) == []


def test_checkpoint_policy_variants(base: Path) -> None:
    s = RunStore.create(base, "정책", {}, policy=CheckpointPolicy(keep_last=2, keep_every=0))
    for g in range(10, 60, 10):
        s.save_checkpoint(g, {})
    assert s.checkpoints() == [40, 50]
    s.save_checkpoint(5, {})  # 예전 세대를 저장해도 방금 저장한 것은 남는다
    assert s.checkpoints() == [5, 40, 50]
    for bad in ((0, 100), (5, -1), (1.5, 100)):
        with pytest.raises(ValueError):
            CheckpointPolicy(*bad)  # type: ignore[arg-type]


def test_load_checkpoint_latest_and_specific(store: RunStore) -> None:
    with pytest.raises(RunStoreError, match="체크포인트가 없습니다"):
        store.load_checkpoint()
    for g in (10, 20, 30):
        path = store.save_checkpoint(g, make_state(g))
        assert path.name == f"gen_{g:06d}.pkl"
    gen, state = store.load_checkpoint()
    assert gen == 30
    want = make_state(30)
    assert state["mean"].tobytes() == want["mean"].tobytes()
    assert pickle.dumps(state["rng"], protocol=5) == pickle.dumps(want["rng"], protocol=5)
    assert state["이름"] == "상태"
    assert store.load_checkpoint(np.int64(10))[0] == 10
    with pytest.raises(RunStoreError, match="10, 20, 30"):
        store.load_checkpoint(15)
    with pytest.raises(ValueError):
        store.load_checkpoint(-1)
    with pytest.raises(ValueError):
        store.save_checkpoint(True, {})  # type: ignore[arg-type]
    (store.root / "checkpoints" / "gen_000040.pkl").write_bytes(b"\x80\x05garbage")
    with pytest.raises(RunStoreError, match="gen_000040"):
        store.load_checkpoint()


class ExplodesWhilePickling:
    def __reduce__(self) -> Any:
        raise RuntimeError("boom")


def test_checkpoint_crash_safety(store: RunStore) -> None:
    store.save_checkpoint(10, make_state(10))
    good = (store.root / "checkpoints" / "gen_000010.pkl").read_bytes()
    big = {"mean": np.zeros(100_000), "later": ExplodesWhilePickling()}  # 일부를 쓴 뒤 실패
    with pytest.raises(RuntimeError, match="boom"):
        store.save_checkpoint(20, big)
    with pytest.raises(RuntimeError, match="boom"):
        store.save_checkpoint(10, big)  # 같은 세대 덮어쓰기 실패: 원래 파일 그대로
    assert store.checkpoints() == [10]
    assert sorted(p.name for p in (store.root / "checkpoints").iterdir()) == ["gen_000010.pkl"]
    assert (store.root / "checkpoints" / "gen_000010.pkl").read_bytes() == good
    gen, state = store.load_checkpoint()
    assert gen == 10 and state["mean"].tobytes() == make_state(10)["mean"].tobytes()


def test_replace_retries_while_file_is_busy(store: RunStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """윈도우: 백신이나 OneDrive가 잠깐 파일을 잡고 있으면 os.replace가 PermissionError를 낸다."""
    real = os.replace
    fails = {"left": 2}

    def flaky(src: Any, dst: Any) -> None:
        if fails["left"] > 0:
            fails["left"] -= 1
            raise PermissionError(13, "Access is denied", str(dst))
        real(src, dst)

    monkeypatch.setattr(store_mod.os, "replace", flaky)
    store.save_checkpoint(10, {"x": 1})
    assert store.load_checkpoint() == (10, {"x": 1})
    fails["left"] = 100  # 끝내 안 풀리면 알기 쉬운 오류, 원래 파일과 임시 파일은 남지 않음
    with pytest.raises(RunStoreError, match="다른 프로그램"):
        store.save_checkpoint(10, {"x": 2})
    assert store.load_checkpoint() == (10, {"x": 1})
    assert leftovers(store.root) == []


def test_stale_checkpoint_tmp_files_are_cleaned(store: RunStore) -> None:
    stale = store.root / "checkpoints" / ".gen_000030.pkl.4242.7.tmp"
    stale.write_bytes(b"half")
    old = time.time() - 3600
    os.utime(stale, (old, old))
    fresh = store.root / "checkpoints" / ".gen_000040.pkl.4243.1.tmp"  # 다른 프로세스가 지금 쓰는 중일 수 있음
    fresh.write_bytes(b"writing")
    store.save_checkpoint(10, {})
    assert not stale.exists() and fresh.exists()


def test_checkpoint_loads_in_fresh_process(store: RunStore) -> None:
    store.save_checkpoint(7, make_state(7))
    store.append_stats(stats_row(0))
    code = (
        "import sys; from pathlib import Path; from evolab.runs.store import RunStore\n"
        "s = RunStore.open(Path(sys.argv[1])); g, st = s.load_checkpoint()\n"
        "print(g, st['mean'].tobytes().hex()[:32], s.read_stats()[0]['block_착지'], s.read_run_info()['env_name'])\n"
    )
    out = subprocess.run([sys.executable, "-c", code, str(store.root)], capture_output=True, check=True,
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"}).stdout.decode("utf-8").split()
    assert out == ["7", make_state(7)["mean"].tobytes().hex()[:32], "0.10000000149011612", "달", "착륙선"]


# ---------------------------------------------------------------------------
# 챔피언
# ---------------------------------------------------------------------------


def test_champion_save_load(store: RunStore) -> None:
    rng = np.random.default_rng(3)
    genomes = {g: rng.normal(size=211) for g in (0, 1, 2, 10)}
    for g, x in genomes.items():
        store.save_champion(g, "representative", x, score=np.float64(-g * 1.5), trial_scores=np.arange(5) * 0.5,
                            seeds=np.array([2**63 + 5, 7], dtype=np.uint64), env_version="v000")
    store.save_champion(2, "champion", genomes[2].astype(np.float32), score=1.25, env_version="v001")
    assert store.champion_generations("representative") == [0, 1, 2, 10]
    assert store.champion_generations("champion") == [2]
    rep = store.load_champion(10, "representative")
    assert rep["genome"].dtype == np.float64 and rep["genome"].tobytes() == genomes[10].tobytes()
    assert float(rep["score"]) == -15.0 and rep["trial_scores"].tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert int(rep["seeds"][0]) == 2**63 + 5 and str(rep["env_version"]) == "v000"
    champ = store.load_champion(2, "champion")
    assert champ["genome"].dtype == np.float64 and float(champ["score"]) == 1.25
    assert (store.root / "champions" / "gen_000010_representative.npz").is_file()
    with pytest.raises(TypeError):
        store.save_champion(3, "champion", genomes[0], meta={"a": 1})  # 피클이 필요한 값은 안 됨
    with pytest.raises(ValueError):
        store.save_champion(3, "best", genomes[0])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        store.save_champion(3, "champion", np.zeros((2, 3)))
    with pytest.raises(RunStoreError, match="챔피언"):
        store.load_champion(3, "champion")
    assert store.champion_generations("champion") == [2]
    assert leftovers(store.root) == []
    # 파일을 닫았으므로 바로 지울 수 있다(윈도우에서 열린 파일은 지울 수 없음)
    (store.root / "champions" / "gen_000010_representative.npz").unlink()


# ---------------------------------------------------------------------------
# 끊긴 실행 정리
# ---------------------------------------------------------------------------


def test_truncate_after(store: RunStore) -> None:
    for g in range(15):
        store.append_stats(stats_row(g))
        store.save_champion(g, "representative", np.full(3, g))
        if g % 4 == 0:
            store.save_champion(g, "champion", np.full(3, g))
        if g in (5, 10, 13):
            store.save_checkpoint(g, {"g": g})
        store.append_event("tick", g)
    store.protect(10)
    store.protect(13)
    lines_before = (store.root / "stats.csv").read_bytes().split(b"\r\n")
    store.truncate_after(10)
    assert store.checkpoints() == [5, 10]
    assert store.champion_generations("representative") == list(range(11))
    assert store.champion_generations("champion") == [0, 4, 8]
    assert [int(r["generation"]) for r in store.read_stats()] == list(range(11))
    assert (store.root / "stats.csv").read_bytes().split(b"\r\n")[:12] == lines_before[:12]  # 남은 줄은 바이트 그대로
    assert store.read_run_info()["protected_checkpoints"] == [10]
    assert len(store.read_events()) == 15  # 사건 기록은 남긴다
    # 이어서 쓰면 중복 없이 이어진다
    store.append_stats(stats_row(11))
    assert [int(r["generation"]) for r in store.read_stats()] == list(range(12))
    mtime = (store.root / "stats.csv").stat().st_mtime_ns
    store.truncate_after(11)  # 지울 것이 없으면 고쳐 쓰지 않는다
    assert (store.root / "stats.csv").stat().st_mtime_ns == mtime
    assert leftovers(store.root) == []


def test_truncate_after_with_pending_rows(store: RunStore, monkeypatch: pytest.MonkeyPatch) -> None:
    store.append_stats(stats_row(0))
    store.append_stats(stats_row(1))
    stats = store.root / "stats.csv"
    real_open = RunStore._open_append

    def locked(self: RunStore, path: Path) -> Any:
        if path == stats:
            raise PermissionError(13, "locked", str(path))
        return real_open(self, path)

    monkeypatch.setattr(RunStore, "_open_append", locked)
    for g in (2, 3, 4):
        store.append_stats(stats_row(g))
    store.truncate_after(2)
    assert [r["generation"] for r in store.read_stats()] == ["0", "1", "2"] and store.pending_stats == 1
    store.truncate_after(0)  # stats.csv 자체를 고쳐 써야 함: 이 테스트의 잠금은 덧붙이기만 막는다
    assert [r["generation"] for r in store.read_stats()] == ["0"] and store.pending_stats == 0


def test_truncate_reports_locked_stats(store: RunStore, monkeypatch: pytest.MonkeyPatch) -> None:
    for g in range(3):
        store.append_stats(stats_row(g))
    real = os.replace

    def deny_stats(src: Any, dst: Any) -> None:
        if Path(dst).name == "stats.csv":
            raise PermissionError(13, "Access is denied", str(dst))
        real(src, dst)

    monkeypatch.setattr(store_mod.os, "replace", deny_stats)
    with pytest.raises(RunStoreError, match="엑셀"):
        store.truncate_after(0)
    assert len(store.read_stats()) == 3 and leftovers(store.root) == []


# ---------------------------------------------------------------------------
# 속도 (보고용 출력)
# ---------------------------------------------------------------------------


def test_store_speed_report(store: RunStore, capsys: pytest.CaptureFixture[str]) -> None:
    row = stats_row(0) | {f"block_{i}": 0.25 * i for i in range(12)}
    genome = np.random.default_rng(0).normal(size=211)
    state = {"mean": genome, "C": np.eye(211), "rng": np.random.RandomState(1).get_state()}
    n = 200
    t0 = time.perf_counter()
    for g in range(n):
        store.append_stats(row | {"generation": g})
    t1 = time.perf_counter()
    for g in range(n):
        store.save_champion(g, "representative", genome, score=1.0, env_version="v000")
    t2 = time.perf_counter()
    for g in range(1, 21):
        store.save_checkpoint(g * 10, state)
    t3 = time.perf_counter()
    store.read_stats()
    t4 = time.perf_counter()
    with capsys.disabled():
        print(f"\n[runs] append_stats {1e6 * (t1 - t0) / n:.0f}us, save_champion {1e6 * (t2 - t1) / n:.0f}us, "
              f"save_checkpoint(211 full C) {1e3 * (t3 - t2) / 20:.2f}ms, read_stats({n}) {1e3 * (t4 - t3):.2f}ms")
    assert (t3 - t0) < 30

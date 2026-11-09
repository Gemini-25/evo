"""M3: 실행 목록, 메모, 삭제 안전장치, 분기(최신이 아닌 체크포인트에서 이어 가기), 기록 화면."""

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from evolab import cli, paths
from evolab.app import strings as S
from evolab.engine.pool import WorkerPool
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.runs import catalog
from evolab.runs import store as store_mod
from evolab.runs.catalog import RunSummary, branch_run, clone_settings, delete_run, list_runs, set_memo
from evolab.runs.store import RunStore, RunStoreError

LANDER = paths.builtin_env_dir() / "lander.evoenv"
LANDER_TEXT = LANDER.read_text(encoding="utf-8")
TIME_COLUMNS = {"gen_seconds", "elapsed", "steps_per_second", "worker_utilization"}


def lander_variant() -> str:
    """같은 유전자 배치·점수 블록에 중력만 다른 환경(환경 수정 흉내)."""
    data = json.loads(LANDER_TEXT)
    data["world"]["gravity"] = [0.0, -1.9]
    return json.dumps(data, ensure_ascii=False, indent=2)


def strip(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{k: v for k, v in r.items() if k not in TIME_COLUMNS} for r in rows]


def config(gens: int) -> TrainConfig:
    return TrainConfig(popsize=6, trials=2, validate_every=24, checkpoint_every=2, run_seed=7, max_generations=gens)


def new_run(runs: Path, name: str, gens: int, created: str) -> RunStore:
    """train 명령과 같은 run.json으로 실행을 만들고 gens세대까지 학습한다(체크포인트 2세대마다)."""
    c = config(gens)
    store = RunStore.create(runs, name, {"program": "evolab", "created": created, "env_name": "달 착륙선",
                                         "method": c.method, "config": c.to_dict()})
    store.add_env_version(LANDER_TEXT, 0)
    store.append_event("start", 0, workers=0)
    with WorkerPool(0) as pool:
        t = Trainer(LANDER_TEXT, c, pool, store)
        reason = t.run()
    store.append_event("stop", t.state.generation, reason=reason)
    return store


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_mod, "_REPLACE_RETRY_DELAYS", (0.0, 0.0))
    monkeypatch.setattr(store_mod, "_APPEND_RETRY_DELAYS", (0.0,))


@pytest.fixture(scope="module")
def parent_run(tmp_path_factory: pytest.TempPathFactory) -> RunStore:
    """4세대 학습 -> 4세대에서 환경 수정 -> 8세대까지 이어 간 실행. 체크포인트 2, 4(보호), 6, 8."""
    runs = tmp_path_factory.mktemp("부모") / "runs"
    store = new_run(runs, "2026-11-07_1530_달 착륙선_CMA-ES", 4, "2026-11-07T15:30:00")
    store.add_env_version(lander_variant(), 4)
    store.append_event("env_change", 4)
    store.protect(4)
    with WorkerPool(0) as pool:
        t = Trainer.resume(store, pool, None, max_generations=8)
        store.append_event("resume", 4, workers=0)
        reason = t.run()
    store.append_event("stop", 8, reason=reason)
    assert store.checkpoints() == [2, 4, 6, 8]
    return store


# ---------------------------------------------------------------------------
# 목록
# ---------------------------------------------------------------------------


def test_list_runs_real_and_broken(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    code = cli.main(["train", str(LANDER), "--seed", "3", "--popsize", "6", "--trials", "2", "--max-generations", "3",
                     "--workers", "0", "--out", str(runs), "--validate-every", "12"])
    assert code == 0
    (cli_run,) = list(runs.iterdir())
    old = new_run(runs, "예전 실행", 2, "2020-01-01T09:00:00")
    broken = runs / "망가진 실행"
    broken.mkdir()
    (broken / "run.json").write_text("{ 반쯤", encoding="utf-8")
    (runs / "빈 폴더").mkdir()
    (runs / ".deleting-1-0-숨김").mkdir()
    (runs / "파일.txt").write_text("x", encoding="utf-8")

    found = list_runs(runs)
    by_name = {s.name: s for s in found}
    assert set(by_name) == {cli_run.name, "예전 실행", "망가진 실행", "빈 폴더"}
    c = by_name[cli_run.name]
    assert c.ok and c.error is None and c.env_name == "달 착륙선" and c.method == "cmaes"
    assert c.generations == 3 and c.memo == "" and c.parent is None
    stats = RunStore.open(cli_run).read_stats()
    assert c.best_score == float(stats[-1]["champion_score"])  # 검증이 있었으므로 챔피언 점수
    assert c.started == RunStore.open(cli_run).read_run_info()["created"]
    o = by_name["예전 실행"]
    assert o.ok and o.generations == 2 and o.started == "2020-01-01T09:00:00"
    assert found.index(c) < found.index(o)  # 최근 것이 앞
    for name in ("망가진 실행", "빈 폴더"):
        assert not by_name[name].ok and by_name[name].error
    assert "run.json" in by_name["빈 폴더"].error  # type: ignore[operator]
    assert list_runs(tmp_path / "없는 폴더") == []
    assert old.root.name == "예전 실행"


def test_list_reads_only_the_tail_and_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    store = RunStore.create(runs, "큰 통계", {"env_name": "e", "method": "cmaes", "created": "2026-01-01T00:00:00"})
    pad = "x" * 300
    for g in range(1, 1001):  # 약 330KB: 끝의 64KB만 읽는다
        store.append_stats({"generation": g, "score_best": float(g), "champion_score": "", "pad": pad})
    real_open = open
    read_sizes: list[int] = []

    class Spy:
        def __init__(self, f: Any) -> None:
            self.f = f

        def __enter__(self) -> "Spy":
            return self

        def __exit__(self, *a: object) -> None:
            self.f.close()

        def read(self, n: int = -1) -> bytes:
            data = self.f.read(n)
            read_sizes.append(len(data))
            return data

        def __getattr__(self, name: str) -> Any:
            return getattr(self.f, name)

    def spy_open(path: Any, mode: str = "r", *a: Any, **k: Any) -> Any:
        f = real_open(path, mode, *a, **k)
        return Spy(f) if str(path).endswith("stats.csv") and mode == "rb" else f

    with monkeypatch.context() as m:
        m.setattr("builtins.open", spy_open)
        (s,) = list_runs(runs)
    assert s.generations == 1000
    assert s.best_score == 1000.0  # 챔피언이 없으면 끝부분 줄들의 score_best 최댓값
    assert read_sizes and max(read_sizes) <= 65536
    # 엑셀 잠금으로 옆 파일에 쌓인 줄이 더 새 줄이다. 잘린 마지막 줄은 버린다.
    side = store.root / "stats.csv.pending"
    side.write_bytes(b"1001,5.0,12.5," + pad.encode() + b"\r\n1002,6.0,")
    (s,) = list_runs(runs)
    assert s.generations == 1001 and s.best_score == 12.5
    # 한 줄이 64KB보다 길어도 마지막 줄을 찾는다
    store2 = RunStore.create(runs, "긴 줄", {"env_name": "e", "created": "2025-01-01T00:00:00"})
    wide = {f"pad{i}": "y" * 60_000 for i in range(4)}  # 칸 하나는 csv 모듈 한도(128KB)보다 작게
    store2.append_stats({"generation": 1, "score_best": 1.0, **wide})
    store2.append_stats({"generation": 2, "score_best": 3.0, **wide})
    s2 = next(x for x in list_runs(runs) if x.name == "긴 줄")
    assert s2.generations == 2 and s2.best_score == 3.0


def test_list_runs_stats_not_utf8_is_a_warning(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    store = RunStore.create(runs, "엑셀", {"env_name": "e"})
    (store.root / "stats.csv").write_bytes("세대,점수\r\n1,2\r\n".encode("cp949"))
    (s,) = list_runs(runs)
    assert s.ok and s.error and "UTF-8" in s.error and s.generations == 0


def test_newer_format_is_not_ok(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    RunStore.create(runs, "새 형식", {"env_name": "e", "store_format": 99})
    (s,) = list_runs(runs)
    assert not s.ok and s.env_name == "e" and "99" in (s.error or "")


# ---------------------------------------------------------------------------
# 메모, 삭제
# ---------------------------------------------------------------------------


def test_memo_round_trip(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    store = RunStore.create(runs, "메모", {"env_name": "e", "config": {"popsize": 3}})
    set_memo(store.root, "다리 길이 1.4\n바람 셈")
    info = store.read_run_info()
    assert info["memo"] == "다리 길이 1.4\n바람 셈" and info["config"] == {"popsize": 3}
    assert list_runs(runs)[0].memo == "다리 길이 1.4\n바람 셈"
    with pytest.raises(RunStoreError):
        set_memo(tmp_path, "실행 아님")


def test_delete_refuses_unsafe_targets(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    store = RunStore.create(runs, "지울 실행", {"env_name": "e"})
    outside = RunStore.create(tmp_path / "다른 곳", "밖의 실행", {"env_name": "e"})
    nested = RunStore.create(store.root / "exports", "안쪽", {"env_name": "e"})
    plain = runs / "그냥 폴더"
    plain.mkdir()
    (plain / "중요.txt").write_text("x", encoding="utf-8")
    for target in (outside.root, nested.root, plain, runs, tmp_path, runs / "없음", runs / ".." / "runs"):
        with pytest.raises(RunStoreError):
            delete_run(target, runs)
    assert outside.root.is_dir() and nested.root.is_dir() and (plain / "중요.txt").is_file() and runs.is_dir()
    if hasattr(os, "symlink"):
        link = runs / "링크"
        try:
            os.symlink(outside.root, link, target_is_directory=True)
        except OSError:
            pass  # 윈도우에서 개발자 모드가 아니면 링크를 못 만든다
        else:
            with pytest.raises(RunStoreError):
                delete_run(link, runs)
            assert (outside.root / "run.json").is_file()
            link.unlink()
    delete_run(store.root, runs)
    assert not store.root.exists()
    assert sorted(p.name for p in runs.iterdir()) == ["그냥 폴더"]  # 숨긴 임시 폴더도 남지 않는다


def test_delete_when_folder_is_locked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    store = RunStore.create(runs, "잠긴 실행", {"env_name": "e"})
    calls = []

    def locked(src: Any, dst: Any) -> None:
        calls.append(src)
        raise PermissionError(13, "in use", str(src))

    with monkeypatch.context() as m:
        m.setattr(catalog.os, "rename", locked)
        with pytest.raises(RunStoreError, match="삭제할 수 없습니다"):
            delete_run(store.root, runs)
    assert len(calls) == len(store_mod._REPLACE_RETRY_DELAYS) + 1  # 잠깐씩 다시 시도했다
    assert (store.root / "run.json").is_file()  # 아무것도 지우지 않았다
    # 지우다 남은 숨긴 폴더는 다음 삭제 때 치운다
    leftover = runs / ".deleting-1-0-남은 것"
    leftover.mkdir()
    (leftover / "a.npz").write_bytes(b"x")
    delete_run(store.root, runs)
    assert list(runs.iterdir()) == []


def test_delete_cleans_leftovers_made_this_clock_tick(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """윈도우 CI(e5b65f7)에서 바로 위 테스트가 남은 폴더를 못 치운 까닭을 어느 OS에서나 그대로 만든다.

    윈도우에서는 파일 시각과 time.time()(파이썬 3.12: GetSystemTimeAsFileTime)이 같은 시스템 시계 눈금에서 나온다.
    방금 만든 폴더는 지금과 똑같은 100ns 값을 갖기 쉽고, 두 값을 초(float)로 바꾸는 식이 달라(time.time은 ns / 1e9,
    st_mtime은 초 + 나노초 * 1e-9) 같은 순간의 약 12%에서 now - st_mtime이 -2.4e-7이 된다. 옛 코드는 나이 조건
    `now - st_mtime >= 0.0`으로 그 폴더를 건너뛰었다. 리눅스에서는 파일 시각이 거친 시계라 늘 time.time()보다 조금
    앞선 값이어서 생기지 않았다. 네트워크 드라이브의 시계가 빠르거나 FAT가 2초 단위로 올려 적은 '미래' 시각도 같다.
    """
    runs = tmp_path / "runs"
    store = RunStore.create(runs, "지울 실행", {"env_name": "e"})
    same, future = runs / ".deleting-1-0-같은 순간", runs / ".deleting-1-1-미래 시각"
    for d in (same, future):
        d.mkdir()
        (d / "a.npz").write_bytes(b"x")
    # 같은 순간의 두 float가 어긋나는 100ns 값을 찾는다(약 8번에 한 번)
    base_ns = int(store.root.stat().st_mtime) * 1_000_000_000
    for k in range(10_000):
        tick_ns = base_ns + k * 100
        os.utime(same, ns=(tick_ns, tick_ns))
        if tick_ns / 1e9 - same.stat().st_mtime < 0:
            break
    else:
        pytest.fail("float가 어긋나는 순간을 찾지 못했다")
    os.utime(future, ns=(tick_ns + 2_000_000_000,) * 2)
    with monkeypatch.context() as m:
        m.setattr(catalog.time, "time", lambda: tick_ns / 1e9)  # time.time()이 돌려주는 값과 같은 식
        delete_run(store.root, runs)
    assert list(runs.iterdir()) == []


def test_delete_cleans_leftover_with_read_only_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """남은 폴더 안의 읽기 전용 파일도 치운다(삭제 본 경로 _rmtree와 같게). 리눅스에서는 윈도우처럼 지우기를 막는다."""
    runs = tmp_path / "runs"
    store = RunStore.create(runs, "지울 실행", {"env_name": "e"})
    leftover = runs / ".deleting-1-0-읽기 전용"
    leftover.mkdir()
    (leftover / "a.npz").write_bytes(b"x")
    os.chmod(leftover / "a.npz", stat.S_IREAD)
    real_unlink = os.unlink

    def windows_unlink(path: Any, *, dir_fd: Any = None) -> None:
        # 윈도우: 읽기 전용 표시가 있는 파일은 os.unlink가 PermissionError(파이썬 3.12의 shutil.rmtree는 풀지 않는다)
        if not os.stat(path, dir_fd=dir_fd, follow_symlinks=False).st_mode & stat.S_IWRITE:
            raise PermissionError(13, "Access is denied", str(path))
        real_unlink(path, dir_fd=dir_fd)

    with monkeypatch.context() as m:
        m.setattr(catalog.os, "unlink", windows_unlink)
        delete_run(store.root, runs)
    assert list(runs.iterdir()) == []


# ---------------------------------------------------------------------------
# 분기
# ---------------------------------------------------------------------------


def test_branch_cuts_records_and_reproduces_parent(tmp_path: Path, parent_run: RunStore) -> None:
    runs = tmp_path / "runs"
    parent_stats = parent_run.read_stats()
    assert [int(r["generation"]) for r in parent_stats] == list(range(1, 9))
    pinfo = parent_run.read_run_info()
    assert [v["from_generation"] for v in pinfo["env_versions"]] == [0, 4]

    for g, versions in ((2, ["v000"]), (4, ["v000", "v001"]), (6, ["v000", "v001"])):
        new = branch_run(parent_run.root, g, runs)
        child = RunStore.open(new)
        info = child.read_run_info()
        assert info["parent"] == {"run": parent_run.root.name, "path": str(parent_run.root.resolve()), "generation": g}
        assert "memo" not in info and info["created"] != pinfo["created"]
        assert info["config"] == pinfo["config"] and info["env_name"] == pinfo["env_name"]
        assert [v["id"] for v in info["env_versions"]] == versions
        assert child.env_text(versions[-1]) == parent_run.env_text(versions[-1])
        assert sorted(p.name for p in (new / "envs").iterdir()) == [f"{v}.evoenv" for v in versions]
        assert child.checkpoints() == [g] and info["protected_checkpoints"] == [g]
        assert child.load_checkpoint()[0] == g
        assert (new / "checkpoints" / f"gen_{g:06d}.pkl").read_bytes() == \
            (parent_run.root / "checkpoints" / f"gen_{g:06d}.pkl").read_bytes()
        assert child.read_stats() == parent_stats[:g]  # 시간 열까지 그대로 복사
        for kind in ("representative", "champion"):
            assert child.champion_generations(kind) == [c for c in parent_run.champion_generations(kind) if c <= g]
        events = child.read_events()
        assert all(e["generation"] <= g for e in events)
        assert events[-1]["kind"] == "branch" and events[-1]["generation"] == g
        assert events[-1]["parent"] == parent_run.root.name
        assert [e["kind"] for e in events[:-1]] == [e["kind"] for e in parent_run.read_events() if e["generation"] <= g]
        assert not list((new / "exports").iterdir())
        # 분기 세대부터 부모의 마지막 세대까지 이어 가면 부모와 같다(시간 열 빼고)
        if g in (2, 4):
            with WorkerPool(0) as pool:
                t = Trainer.resume(child, pool, None, max_generations=4 if g == 2 else 8)
                t.run()
            n = 4 if g == 2 else 8
            assert strip(child.read_stats()) == strip(parent_stats[:n])
            assert child.champion_generations("champion") == [c for c in parent_run.champion_generations("champion")
                                                               if c <= n]
    # 원래 실행은 그대로
    assert parent_run.checkpoints() == [2, 4, 6, 8] and len(parent_run.read_stats()) == 8
    names = [s.name for s in list_runs(runs)]
    assert len(names) == 3 and len(set(names)) == 3
    assert all(S.REC_BRANCH_FOLDER.split("_")[-1].format(generation=g) in " ".join(names) for g in (2, 4, 6))
    summaries = {s.parent["generation"]: s for s in list_runs(runs) if s.parent}  # type: ignore[index]
    assert summaries[6].generations == 6 and summaries[4].generations == 8


def test_branch_errors_leave_nothing(tmp_path: Path, parent_run: RunStore) -> None:
    runs = tmp_path / "runs"
    with pytest.raises(RunStoreError, match="3세대"):
        branch_run(parent_run.root, 3, runs)
    with pytest.raises(ValueError):
        branch_run(parent_run.root, -1, runs)
    with pytest.raises(RunStoreError):
        branch_run(tmp_path, 2, runs)
    assert not runs.exists() or list(runs.iterdir()) == []


def test_branch_failure_midway_removes_folder(tmp_path: Path, parent_run: RunStore, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"

    def boom(*a: Any, **k: Any) -> None:
        raise OSError(28, "disk full")

    with monkeypatch.context() as m:
        m.setattr(catalog.shutil, "copyfile", boom)
        with pytest.raises(OSError):
            branch_run(parent_run.root, 4, runs)
    assert list(runs.iterdir()) == []
    # 전원이 끊겨 남은 분기 임시 폴더는 목록에 안 나오고, 오래되면 다음 분기 때 치운다
    stale, fresh = runs / ".branching-1-0", runs / ".branching-1-1"
    for d in (stale, fresh):
        (d / "champions").mkdir(parents=True)
    os.utime(stale, (1.0, 1.0))
    assert list_runs(runs) == []
    new = branch_run(parent_run.root, 4, runs)
    assert sorted(p.name for p in runs.iterdir()) == sorted([new.name, fresh.name])


def test_branch_includes_pending_stats(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    store = RunStore.create(runs, "잠긴 통계", {"env_name": "e", "method": "cmaes"})
    store.add_env_version("{}", 0)
    for g in (1, 2):
        store.append_stats({"generation": g, "score_best": float(g)})
    store.save_checkpoint(2, {"x": 1})
    sidecar = store.root / "stats.csv.pending"
    sidecar.write_bytes(b"3,3.0\r\n4,4.0\r\n")  # 엑셀이 잡고 있어 옆 파일에 쌓인 줄
    store.save_checkpoint(3, {"x": 2})
    assert len(store.read_stats()) == 4
    new = branch_run(store.root, 3, runs)
    assert [r["generation"] for r in RunStore.open(new).read_stats()] == ["1", "2", "3"]
    assert not (new / "stats.csv.pending").exists()


def test_clone_settings(parent_run: RunStore) -> None:
    got = clone_settings(parent_run.root)
    assert got["env_text"] == lander_variant()  # 마지막 환경 버전
    assert got["env_name"] == "달 착륙선"
    assert got["config"] == parent_run.read_run_info()["config"] and got["config"]["popsize"] == 6


# ---------------------------------------------------------------------------
# 기록 화면
# ---------------------------------------------------------------------------


def row_of(view: Any, path: Path) -> int:
    from evolab.app.records.view import COL_ENV

    for r in range(view.table.rowCount()):
        if Path(view.table.item(r, COL_ENV).data(256)) == path:  # Qt.ItemDataRole.UserRole
            return r
    return -1


@pytest.fixture
def view_runs(tmp_path: Path, parent_run: RunStore) -> Path:
    import shutil

    runs = tmp_path / "runs"
    runs.mkdir()
    shutil.copytree(parent_run.root, runs / parent_run.root.name)
    one = RunStore.create(runs, "체크포인트 하나", {"env_name": "e", "method": "cmaes", "created": "2026-11-08T10:00:00",
                                                  "config": {"popsize": 4}})
    one.add_env_version("{}", 0)
    one.append_stats({"generation": 1, "score_best": 2.5, "champion_score": ""})
    one.save_checkpoint(1, {"x": 1})
    (runs / "망가짐").mkdir()
    return runs


def test_view_lists_runs(qtbot: Any, view_runs: Path) -> None:
    from PySide6.QtCore import Qt

    from evolab.app.records.view import COL_BEST, COL_ENV, COL_GENERATIONS, COL_MEMO, COL_METHOD, RecordsView

    view = RecordsView(lambda: view_runs)
    qtbot.addWidget(view)
    t = view.table
    assert [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())] == \
        ["환경", "방법", "시작 시각", "세대 수", "최고 점수", "메모"]
    assert t.rowCount() == 3
    texts = {t.item(r, COL_ENV).text(): r for r in range(t.rowCount())}
    r = texts["e"]
    assert t.item(r, COL_METHOD).text() == "CMA-ES" and t.item(r, COL_GENERATIONS).text() == "1"
    assert t.item(r, COL_BEST).text() == "2.50"
    assert t.item(texts["망가짐"], COL_METHOD).text() == S.REC_UNREADABLE
    assert t.item(texts["달 착륙선"], COL_GENERATIONS).text() == "8"
    assert t.item(r, COL_MEMO).flags() & Qt.ItemFlag.ItemIsEditable
    assert not t.item(r, COL_ENV).flags() & Qt.ItemFlag.ItemIsEditable
    # 고르지 않으면 단추가 꺼져 있다. 망가진 실행은 열 수 없다.
    assert not view.open_button.isEnabled()
    t.selectRow(texts["망가짐"])
    assert not view.open_button.isEnabled() and view.delete_button.isEnabled()
    t.selectRow(r)
    assert view.open_button.isEnabled() and view.resume_button.isEnabled()
    assert str(view_runs) in view.folder_label.text() and "3" in view.folder_label.text()
    empty = RecordsView(lambda: view_runs / "없음")
    qtbot.addWidget(empty)
    assert empty.table.rowCount() == 0 and not empty.empty_label.isHidden()


def test_view_open_clone_and_memo(qtbot: Any, view_runs: Path) -> None:
    from evolab.app.records.view import COL_ENV, COL_MEMO, RecordsView

    view = RecordsView(lambda: view_runs)
    qtbot.addWidget(view)
    run = view_runs / "체크포인트 하나"
    assert view.select_run(run)
    with qtbot.waitSignal(view.open_requested, check_params_cb=lambda p: p == run):
        view.open_button.click()
    with qtbot.waitSignal(view.clone_requested, check_params_cb=lambda p: p == run):
        view.clone_button.click()
    row = row_of(view, run)
    with qtbot.waitSignal(view.open_requested):
        view._on_double_click(row, COL_ENV)
    with qtbot.assertNotEmitted(view.open_requested):
        view._on_double_click(row, COL_MEMO)  # 메모 칸은 고치기
    # 메모 칸을 고치면 run.json에 저장된다
    view.table.item(row, COL_MEMO).setText("바람 강하게")
    assert RunStore.open(run).read_run_info()["memo"] == "바람 강하게"
    assert view.selected_run().memo == "바람 강하게"  # type: ignore[union-attr]
    view.refresh()
    assert view.table.item(row_of(view, run), COL_MEMO).text() == "바람 강하게"
    assert view.save_memo(run, "") is None and RunStore.open(run).read_run_info()["memo"] == ""
    errors: list[str] = []
    view.show_error = errors.append  # type: ignore[method-assign]
    (run / "run.json").unlink()
    view.table.item(row_of(view, run), COL_MEMO).setText("못 저장")
    assert errors and errors[0].startswith(S.REC_MEMO_FAILED.split("{")[0])
    # 실패하면 신호 처리가 끝난 뒤 목록을 다시 읽는다
    qtbot.waitUntil(lambda: not next(s for s in view.runs() if s.path == run).ok)


def test_view_resume_latest_and_branch(qtbot: Any, view_runs: Path, parent_run: RunStore) -> None:
    from evolab.app.records.view import RecordsView

    view = RecordsView(lambda: view_runs)
    qtbot.addWidget(view)
    errors: list[str] = []
    view.show_error = errors.append  # type: ignore[method-assign]
    one = view_runs / "체크포인트 하나"
    view.select_run(one)
    asked: list[Any] = []
    view.ask_checkpoint = lambda s, gens: asked.append(gens)  # type: ignore[method-assign,assignment,return-value]
    with qtbot.waitSignal(view.resume_requested, check_params_cb=lambda p, g: p == one and g == 1):
        view.resume_button.click()
    assert asked == []  # 체크포인트가 하나면 묻지 않는다

    parent = view_runs / parent_run.root.name
    view.select_run(parent)
    view.ask_checkpoint = lambda s, gens: gens[-1]  # type: ignore[method-assign]
    with qtbot.waitSignal(view.resume_requested, check_params_cb=lambda p, g: p == parent and g == 8):
        view.resume_button.click()

    # 최신이 아닌 체크포인트: 알리고, 취소하면 아무것도 하지 않는다
    view.select_run(parent)
    view.ask_checkpoint = lambda s, gens: 4  # type: ignore[method-assign]
    view.confirm_branch = lambda s, g: False  # type: ignore[method-assign]
    with qtbot.assertNotEmitted(view.resume_requested):
        view.resume_button.click()
    assert len(list(view_runs.iterdir())) == 3
    view.confirm_branch = lambda s, g: True  # type: ignore[method-assign]
    got: list[tuple[Path, int]] = []
    view.resume_requested.connect(lambda p, g: got.append((p, g)))
    view.resume_button.click()
    assert len(got) == 1 and got[0][1] == 4 and got[0][0] != parent
    new = got[0][0]
    assert RunStore.open(new).read_run_info()["parent"]["generation"] == 4
    assert view.table.rowCount() == 4 and view.selected_run().path == new  # type: ignore[union-attr]
    assert errors == []
    # 학습 중인 실행은 최신 체크포인트에서 이어 가거나 지울 수 없다
    view.set_busy_runs([one])
    view.select_run(one)
    assert not view.delete_button.isEnabled()
    view.resume_button.click()
    assert errors and errors[-1] == S.REC_RESUME_FAILED.format(reason=S.REC_RESUME_BUSY)


def test_view_delete_confirms(qtbot: Any, view_runs: Path) -> None:
    from evolab.app.records.view import RecordsView

    view = RecordsView(lambda: view_runs)
    qtbot.addWidget(view)
    errors: list[str] = []
    view.show_error = errors.append  # type: ignore[method-assign]
    one = view_runs / "체크포인트 하나"
    asked: list[RunSummary] = []
    view.select_run(one)
    view.confirm_delete = lambda s: asked.append(s) or False  # type: ignore[method-assign]
    view.delete_button.click()
    assert one.is_dir() and asked and asked[0].path == one
    view.confirm_delete = lambda s: True  # type: ignore[method-assign]
    view.delete_button.click()
    assert not one.exists() and view.table.rowCount() == 2 and errors == []
    # run.json이 없는 폴더는 지우지 않고 까닭을 알린다
    view.select_run(view_runs / "망가짐")
    view.delete_button.click()
    assert (view_runs / "망가짐").is_dir() and errors and "run.json" in errors[0]


def test_confirm_dialogs_use_message_box(qtbot: Any, view_runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QInputDialog, QMessageBox

    from evolab.app.records.view import RecordsView

    view = RecordsView(lambda: view_runs)
    qtbot.addWidget(view)
    s = view.runs()[0]
    seen: list[str] = []

    def warning(parent: Any, title: str, text: str, *a: Any) -> Any:
        seen.append(text)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "warning", warning)
    assert view.confirm_delete(s) and "되돌릴 수 없습니다" in seen[-1]
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Cancel)
    assert not view.confirm_branch(s, 4)
    monkeypatch.setattr(QInputDialog, "getItem", lambda parent, title, label, items, cur, ed: (items[2], True))
    assert view.ask_checkpoint(s, [2, 4, 6, 8]) == 4
    monkeypatch.setattr(QInputDialog, "getItem", lambda parent, title, label, items, cur, ed: (items[0], False))
    assert view.ask_checkpoint(s, [2, 4, 6, 8]) is None

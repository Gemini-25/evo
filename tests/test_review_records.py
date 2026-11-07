"""검토 수정(기록 화면, 실행 목록, 비교·반복 실행, 재현 묶음)의 회귀 테스트."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractItemDelegate, QAbstractItemView, QApplication, QLineEdit

from evolab import cli, paths
from evolab.app import strings as S
from evolab.app.compare.repeat import RepeatPlan, RepeatRunner
from evolab.app.compare.view import CompareView, aggregate, list_entries
from evolab.app.records.view import COL_MEMO, RecordsView
from evolab.runs import bundle as bundle_mod
from evolab.runs import store as store_mod
from evolab.runs.catalog import branch_run
from evolab.runs.store import RunStore, RunStoreError

LANDER = paths.builtin_env_dir() / "lander.evoenv"
LANDER_TEXT = LANDER.read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_mod, "_REPLACE_RETRY_DELAYS", (0.0, 0.0))
    monkeypatch.setattr(store_mod, "_APPEND_RETRY_DELAYS", (0.0,))


def cheap_run(runs: Path, name: str, gens: int = 2, checkpoints: tuple[int, ...] = (1, 2), seed: int = 1,
              group: dict[str, Any] | None = None, created: str = "2026-11-07T10:00:00",
              compare: bool = True) -> RunStore:
    """학습 없이 만든 실행: 환경 버전 하나, gens세대 통계, 체크포인트 몇 개(내용은 가짜)."""
    info: dict[str, Any] = {"program": "evolab", "created": created, "env_name": "달 착륙선", "method": "cmaes",
                            "config": {"run_seed": seed, "popsize": 4}, "memo": "부모 메모"}
    if group is not None:
        info["group"] = group
    store = RunStore.create(runs, name, info)
    store.add_env_version(LANDER_TEXT, 0)
    store.append_event("start", 0, workers=0)
    for g in range(1, gens + 1):
        store.append_stats({"generation": g, "evaluations": 10 * g, "elapsed": float(g), "score_best": float(seed + g),
                            "compare_score": float(seed + g) if compare else "", "champion_score": ""})
    for g in checkpoints:
        store.save_checkpoint(g, {"x": g})
    return store


def folders(runs: Path) -> list[str]:
    return sorted(p.name for p in runs.iterdir() if not p.name.startswith("."))


# ---------------------------------------------------------------------------
# 12: 분기는 반복 실행 묶음 소속을 물려받지 않는다
# ---------------------------------------------------------------------------


def test_branch_does_not_join_parent_group(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    group = {"id": "grp-1", "name": "묶음", "index": 0, "count": 2}
    a = cheap_run(runs, "시드1", seed=1, group=group, created="2026-11-07T10:00:00")
    cheap_run(runs, "시드2", seed=2, group={**group, "index": 1}, created="2026-11-07T10:05:00")
    new = branch_run(a.root, 1, runs)
    info = RunStore.open(new).read_run_info()
    assert "group" not in info and "memo" not in info
    assert info["parent"]["generation"] == 1 and info["config"]["run_seed"] == 1
    assert RunStore.open(a.root).read_run_info()["group"] == group  # 부모는 그대로
    by_name = {e.summary.name: e for e in list_entries(runs)}
    assert by_name[new.name].group is None
    members = [e for e in by_name.values() if e.group is not None]
    assert sorted(e.seed for e in members) == [1, 2]  # 묶음에는 원래 두 실행만


# ---------------------------------------------------------------------------
# 11(기록 화면 쪽), 13, 19: 학습 중인 실행·받을 수 없는 이어하기는 분기하기 전에 거절한다
# ---------------------------------------------------------------------------


@pytest.fixture
def records(qtbot: Any, tmp_path: Path) -> tuple[RecordsView, Path, RunStore, list[str], list[Any]]:
    runs = tmp_path / "runs"
    run = cheap_run(runs, "부모", gens=4, checkpoints=(2, 4))
    view = RecordsView(lambda: runs)
    qtbot.addWidget(view)
    errors: list[str] = []
    asked: list[Any] = []
    view.show_error = errors.append  # type: ignore[method-assign]
    view.ask_checkpoint = lambda s, gens: asked.append(gens) or gens[0]  # type: ignore[method-assign]
    view.confirm_branch = lambda s, g: True  # type: ignore[method-assign]
    return view, runs, run, errors, asked


def test_busy_run_cannot_be_resumed_or_branched(qtbot: Any, records: Any) -> None:
    view, runs, run, errors, asked = records
    before = folders(runs)
    view.set_busy_runs([run.root])
    assert view.select_run(run.root)
    with qtbot.assertNotEmitted(view.resume_requested):
        view.resume_button.click()
        with pytest.raises(RunStoreError, match=S.REC_RESUME_BUSY):
            view.resume_run(run.root, 2)  # 최신이 아닌 체크포인트(분기)도
        with pytest.raises(RunStoreError, match=S.REC_RESUME_BUSY):
            view.resume_run(run.root)
    assert errors == [S.REC_RESUME_FAILED.format(reason=S.REC_RESUME_BUSY)]
    assert asked == []  # 체크포인트를 묻기 전에 거절
    assert folders(runs) == before
    # 학습이 끝나면 분기할 수 있다
    view.set_busy_runs([])
    view.select_run(run.root)
    got: list[tuple[Path, int]] = []
    view.resume_requested.connect(lambda p, g: got.append((p, g)))
    view.resume_button.click()
    assert asked == [[2, 4]] and len(got) == 1 and got[0][1] == 2 and got[0][0] != run.root
    assert len(folders(runs)) == len(before) + 1


def test_refused_resume_leaves_no_branch_folder(qtbot: Any, records: Any) -> None:
    view, runs, run, errors, asked = records
    before = folders(runs)
    blocked: list[str | None] = [S.MAIN_LAB_BUSY]
    view.resume_blocked = lambda: blocked[0]
    view.select_run(run.root)
    with qtbot.assertNotEmitted(view.resume_requested):
        view.resume_button.click()
        view.resume_button.click()  # 다시 해도 폴더가 늘지 않는다
        with pytest.raises(RunStoreError) as exc:
            view.resume_run(run.root, 2)
        assert str(exc.value) == S.MAIN_LAB_BUSY
        with pytest.raises(RunStoreError):
            view.resume_run(run.root)
    assert errors == [S.REC_RESUME_FAILED.format(reason=S.MAIN_LAB_BUSY)] * 2
    assert asked == []  # 체크포인트 고르기·분기 확인을 띄우지 않는다
    assert folders(runs) == before and view.table.rowCount() == 1
    # 받을 수 있게 되면 분기하고 보낸다
    blocked[0] = None
    with qtbot.waitSignal(view.resume_requested, check_params_cb=lambda p, g: g == 2 and p != run.root):
        view.resume_button.click()
    assert len(folders(runs)) == len(before) + 1 and errors == [S.REC_RESUME_FAILED.format(reason=S.MAIN_LAB_BUSY)] * 2


def test_records_view_alone_allows_resume(qtbot: Any, records: Any) -> None:
    """주 창 없이 만든 기록 화면은 따로 막는 것이 없다(resume_blocked 기본값)."""
    view, _runs, run, errors, _asked = records
    assert view.resume_blocked() is None
    with qtbot.waitSignal(view.resume_requested, check_params_cb=lambda p, g: p == run.root and g == 4):
        assert view.resume_run(run.root) == run.root
    assert errors == []


# ---------------------------------------------------------------------------
# 15: 메모를 고치는 중에 목록을 다시 읽어도 쓰던 메모가 사라지지 않는다
# ---------------------------------------------------------------------------


def _open_memo_editor(qtbot: Any, view: RecordsView, run: Path) -> QLineEdit:
    view.resize(900, 400)
    view.show()
    qtbot.waitExposed(view)
    assert view.select_run(run)
    view.memo_button.click()
    editor = view.table.findChild(QLineEdit)
    assert editor is not None and editor.isVisible()
    return editor


def _close_editor(qtbot: Any, view: RecordsView, editor: QLineEdit, commit: bool) -> None:
    """사용자가 Enter(저장) 또는 Esc(취소)를 누른다: 칸 위임자가 commitData·closeEditor를 보낸다(Enter는 다음 차례에)."""
    QTest.keyClick(editor, Qt.Key.Key_Return if commit else Qt.Key.Key_Escape)
    qtbot.waitUntil(lambda: view.table.state() != QAbstractItemView.State.EditingState)


def test_refresh_while_editing_memo_keeps_text(qtbot: Any, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    a = cheap_run(runs, "실행A", created="2026-11-07T09:00:00")
    b = cheap_run(runs, "실행B", created="2026-11-07T09:30:00")
    view = RecordsView(lambda: runs)
    qtbot.addWidget(view)
    editor = _open_memo_editor(qtbot, view, a.root)
    editor.setText("중요한 메모")
    # 반복 실행이 다음 실행을 시작했다(주 창의 _set_busy가 하는 일) + 새 실행 폴더가 생겼다
    cheap_run(runs, "실행C", created="2026-11-07T11:00:00")
    assert view.delete_button.isEnabled()
    view.set_busy_runs([a.root, b.root])
    view.refresh()
    QApplication.processEvents()
    assert view.table.findChild(QLineEdit) is editor and editor.isVisible() and editor.text() == "중요한 메모"
    assert view.table.rowCount() == 2  # 고치는 동안은 다시 읽지 않는다
    assert not view.delete_button.isEnabled()  # 학습 중 표시(지우기 막기)는 바로 바뀐다
    _close_editor(qtbot, view, editor, commit=True)
    assert RunStore.open(a.root).read_run_info()["memo"] == "중요한 메모"
    qtbot.waitUntil(lambda: view.table.rowCount() == 3)  # 미뤄 둔 다시 읽기
    assert view.selected_run() is not None and view.selected_run().path == a.root  # type: ignore[union-attr]
    assert "중요한 메모" in [view.table.item(r, COL_MEMO).text() for r in range(view.table.rowCount())]


def test_refresh_waits_for_cancelled_memo_edit(qtbot: Any, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    a = cheap_run(runs, "실행A")
    view = RecordsView(lambda: runs)
    qtbot.addWidget(view)
    editor = _open_memo_editor(qtbot, view, a.root)
    editor.setText("버릴 메모")
    cheap_run(runs, "실행B", created="2026-11-07T11:00:00")
    view.refresh()
    assert view.table.rowCount() == 1
    _close_editor(qtbot, view, editor, commit=False)  # Esc: 저장하지 않지만 미뤄 둔 다시 읽기는 한다
    qtbot.waitUntil(lambda: view.table.rowCount() == 2)
    assert RunStore.open(a.root).read_run_info()["memo"] == "부모 메모"
    view.refresh()  # 편집기가 없으면 바로 다시 읽는다
    assert view.table.rowCount() == 2
    # 편집기를 코드로 닫아도(위임자 신호 없이) 미뤄 둔 다시 읽기를 한다
    editor = _open_memo_editor(qtbot, view, a.root)
    cheap_run(runs, "실행C", created="2026-11-07T12:00:00")
    view.refresh()
    assert view.table.rowCount() == 2
    view.table.closeEditor(editor, QAbstractItemDelegate.EndEditHint.NoHint)
    qtbot.waitUntil(lambda: view.table.rowCount() == 3)


# ---------------------------------------------------------------------------
# 14: 값이 없는 묶음 실행이 평균 선과 띠를 지우지 않는다, 취소로 생긴 빈 실행은 지운다
# ---------------------------------------------------------------------------


def test_group_curve_ignores_member_without_values() -> None:
    a = (np.array([0.0, 10.0, 20.0]), np.array([1.0, 2.0, 3.0]))
    b = (np.array([0.0, 10.0, 20.0]), np.array([3.0, 4.0, 5.0]))
    no_values = (np.array([0.0, 10.0]), np.array([math.nan, math.nan]))  # 첫 비교 평가 전에 멈춘 실행
    empty = (np.empty(0), np.empty(0))  # 0세대로 끝난 실행
    g = aggregate([a, b, no_values, empty], step=True)
    assert g.n == 2
    assert g.full.tolist() == [True, True, True] and not g.partial.any()
    assert np.allclose(g.mean, [2, 3, 4]) and g.lo.tolist() == [1, 2, 3] and g.hi.tolist() == [3, 4, 5]
    # 일찍 끝난 실행은 그대로 '일부 실행만'
    short = (np.array([0.0]), np.array([7.0]))
    g = aggregate([a, short], step=False)
    assert g.n == 2 and g.full.tolist() == [True, False, False] and g.partial.tolist() == [False, True, True]
    # 어느 실행에도 값이 없으면 모든 실행에 값이 있는 점도 없다
    g = aggregate([no_values, empty], step=True)
    assert g.n == 0 and not g.full.any() and not g.partial.any()


def test_group_with_empty_member_draws_mean_and_band(qtbot: Any, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    group = {"id": "grp-e", "name": "빈 실행 묶음", "count": 3}
    cheap_run(runs, "r0", gens=3, seed=1, group={**group, "index": 0}, created="2026-11-07T10:00:00")
    cheap_run(runs, "r1", gens=3, seed=2, group={**group, "index": 1}, created="2026-11-07T10:01:00")
    cheap_run(runs, "r2", gens=0, checkpoints=(), seed=3, group={**group, "index": 2}, created="2026-11-07T10:02:00")
    view = CompareView(lambda: runs, settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.show()
    assert view.set_group_checked("grp-e")
    curve = view.shown[0].curve
    assert curve is not None and len(view.shown[0].entries) == 3
    assert curve.n == 2 and curve.full.all() and not curve.partial.any()
    legend = view.legend.text()
    assert S.CMP_LEGEND_GROUP.format(name="빈 실행 묶음", n=2) in legend and S.CMP_LEGEND_BAND in legend
    assert S.CMP_LEGEND_PARTIAL not in legend


def test_group_legend_has_no_band_without_full_points(qtbot: Any, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    group = {"id": "grp-p", "name": "어긋난 묶음", "count": 2}
    a = cheap_run(runs, "p0", gens=0, checkpoints=(), seed=1, group={**group, "index": 0}, created="2026-11-07T10:00:00")
    b = cheap_run(runs, "p1", gens=0, checkpoints=(), seed=2, group={**group, "index": 1}, created="2026-11-07T10:01:00")
    for g in (1, 2):  # 두 실행의 시험 횟수가 겹치지 않는다
        a.append_stats({"generation": g, "evaluations": g, "compare_score": 1.0})
        b.append_stats({"generation": g, "evaluations": 100 + g, "compare_score": 2.0})
    view = CompareView(lambda: runs, settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(view)
    view.show()
    view.set_y_metric("compare_score")
    view.set_x_axis("evaluations")
    assert view.set_group_checked("grp-p")
    curve = view.shown[0].curve
    assert curve is not None and curve.n == 2 and not curve.full.any() and curve.partial.any()
    legend = view.legend.text()
    assert S.CMP_LEGEND_BAND not in legend and S.CMP_LEGEND_PARTIAL in legend


class FakeClient(QObject):
    """학습 프로세스 대신: start에서 실행 폴더를 만들고(세션의 _open처럼), 'started'는 shutdown 도중에야 온다."""

    started = Signal(dict)
    generation = Signal(dict)
    state_changed = Signal(dict)
    failed = Signal(str, bool)
    closed = Signal()

    def __init__(self, parent: QObject | None, rows: int) -> None:
        super().__init__(parent)
        self.rows = rows
        self.run_dir: Path | None = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self, req: Any) -> None:
        store = RunStore.create(Path(req.runs_dir), "반복", {"program": "evolab", "config": req.config, **req.info})
        self.run_dir = store.root
        self._active = True

    def send(self, msg: dict[str, Any]) -> None:
        pass

    def shutdown(self) -> None:
        assert self.run_dir is not None
        store = RunStore.open(self.run_dir)
        for g in range(1, self.rows + 1):  # quit을 읽기 전에 마친 세대
            store.append_stats({"generation": g, "evaluations": g})
        if self.rows:
            store.save_checkpoint(self.rows, {"x": 1})
        store.append_event("stop", self.rows, reason="requested")
        self.started.emit({"type": "started", "run_dir": str(self.run_dir)})
        self._active = False
        self.closed.emit()


@pytest.mark.parametrize("rows", [0, 1])
def test_repeat_cancel_while_starting_removes_empty_run(qtbot: Any, tmp_path: Path, rows: int) -> None:
    runs = tmp_path / "runs"
    plan = RepeatPlan(env_text=LANDER_TEXT, config={"method": "cmaes", "max_evaluations": 60}, count=3, base_seed=5,
                      workers=0, runs_dir=runs, group_name="묶음", env_name="달 착륙선")
    clients: list[FakeClient] = []

    def factory(parent: QObject) -> FakeClient:
        clients.append(FakeClient(parent, rows))
        return clients[-1]

    runner = RepeatRunner(plan, client_factory=factory)  # type: ignore[arg-type]
    finished: list[bool] = []
    runner.finished.connect(finished.append)
    runner.start()
    made = clients[0].run_dir
    assert made is not None and made.is_dir() and runner.run_dirs == []
    runner.cancel()
    assert finished == [False] and runner.state == "cancelled" and runner.client is None
    if rows == 0:
        # 한 세대도 마치지 못한 실행은 묶음에 남기지 않는다
        assert not made.exists() and runner.run_dirs == [] and folders(runs) == []
    else:
        # 세대를 마쳤으면(체크포인트도 남았다) 그대로 둔다
        assert made.is_dir() and runner.run_dirs == [made]
        assert RunStore.open(made).read_run_info()["group"]["index"] == 0
    assert len(clients) == 1  # 다음 실행은 띄우지 않는다


# ---------------------------------------------------------------------------
# 16: 재현 묶음 내보내기가 실패하면 임시 파일을 남기지 않고, 망가진 기록은 한국어 오류로
# ---------------------------------------------------------------------------


def test_bundle_failures_leave_no_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    run = cheap_run(runs, "실행")
    out_dir = tmp_path / "내보내기"
    good = bundle_mod.export_bundle(run.root, out_dir / "좋음.zip")
    assert good.is_file()
    old = good.read_bytes()

    # 1) 환경 사본이 바뀌었다: RunStoreError, 임시 파일 없음, 원래 zip은 그대로
    env = run.root / "envs" / "v000.evoenv"
    original = env.read_bytes()
    env.write_bytes(original + b" ")
    with pytest.raises(RunStoreError):
        bundle_mod.export_bundle(run.root, good)
    assert sorted(p.name for p in out_dir.iterdir()) == ["좋음.zip"] and good.read_bytes() == old
    env.write_bytes(original)

    # 2) 바꿀 zip을 다른 프로그램이 잡고 있다: RUNS_FILE_BUSY, 임시 파일 없음
    def locked(src: Path, dst: Path) -> None:
        raise PermissionError(13, "locked", str(dst))

    monkeypatch.setattr(bundle_mod._st, "_replace", locked)
    with pytest.raises(RunStoreError) as exc:
        bundle_mod.export_bundle(run.root, good)
    assert str(exc.value) == S.RUNS_FILE_BUSY.format(path=good)
    assert sorted(p.name for p in out_dir.iterdir()) == ["좋음.zip"] and good.read_bytes() == old
    monkeypatch.undo()

    # 3) env_versions 기록에 id가 없다: KeyError가 아니라 RunStoreError(한국어)
    info = run.read_run_info()
    run.update_run_info(env_versions=[{k: v for k, v in info["env_versions"][0].items() if k != "id"}])
    with pytest.raises(RunStoreError, match="환경 버전"):
        bundle_mod.export_bundle(run.root, out_dir / "망가짐.zip")
    run.update_run_info(env_versions=["v000"])
    with pytest.raises(RunStoreError):
        bundle_mod.export_bundle(run.root, out_dir / "망가짐.zip")
    run.update_run_info(env_versions={"id": "v000"})
    with pytest.raises(RunStoreError):
        bundle_mod.export_bundle(run.root, out_dir / "망가짐.zip")
    assert sorted(p.name for p in out_dir.iterdir()) == ["좋음.zip"]
    # 명령줄도 오류 글과 종료 코드 2로 끝난다(추적 정보 없이)
    assert cli.main(["bundle", str(run.root), "--out", str(out_dir / "명령.zip")]) == 2
    assert sorted(p.name for p in out_dir.iterdir()) == ["좋음.zip"]


def test_records_bundle_with_broken_env_record_shows_error(qtbot: Any, tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    runs = tmp_path / "runs"
    run = cheap_run(runs, "실행")
    info = run.read_run_info()
    run.update_run_info(env_versions=[{k: v for k, v in info["env_versions"][0].items() if k != "id"}])
    view = RecordsView(lambda: runs)
    qtbot.addWidget(view)
    errors: list[str] = []
    view.show_error = errors.append  # type: ignore[method-assign]
    out = tmp_path / "내보내기" / "묶음.zip"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: pytest.fail("성공으로 알리면 안 된다"))
    assert view.select_run(run.root)
    view.bundle_button.click()
    assert len(errors) == 1 and errors[0].startswith(S.REC_EXPORT_FAILED)
    assert not out.exists() and not out.with_name(out.name + ".tmp").exists()


def test_bundle_still_round_trips(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = cheap_run(runs, "실행")
    run.add_env_version(LANDER_TEXT.replace("\n", "\r\n"), 2)
    out = bundle_mod.export_bundle(run.root, tmp_path / "b.zip")
    data = bundle_mod.read_bundle(out)
    assert sorted(data["envs"]) == ["v000", "v001"] and len(data["stats"]) == 2
    assert json.loads(json.dumps(data["info"]))["env_versions"][1]["from_generation"] == 2
    assert not (tmp_path / "b.zip.tmp").exists()

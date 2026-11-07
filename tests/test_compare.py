"""M6: 비교 화면(실행 겹쳐 그리기, 반복 실행 묶음의 평균·범위, 비교 표, 내보내기)과 반복 실행."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt

from evolab import cli, paths
from evolab.app import strings as S
from evolab.app.compare import repeat as repeat_mod
from evolab.app.compare.repeat import RepeatDialog, RepeatPlan, RepeatRunner, available_methods
from evolab.app.compare.view import (
    LIST_COL_GROUP,
    TABLE_COL_FINAL,
    TABLE_COL_TARGET,
    X_AXES,
    CompareView,
    aggregate,
    ffill,
    load_stats,
    run_metrics,
    sample,
    series,
    step_xy,
)
from evolab.engine.trainer import TrainConfig, make_learner
from evolab.policy.genome import layout_for
from evolab.runs.store import RunStore
from evolab.spec.io import load_env

LANDER = paths.builtin_env_dir() / "lander.evoenv"
GROUP_ID = "grp-test"
GROUP_NAME = "착륙선 반복 묶음"


@dataclass
class Runs:
    dir: Path
    g1: Path
    g2: Path
    single: Path
    synthetic: Path
    old: Path
    cp949: Path
    broken: Path
    empty: Path


def _train(out: Path, seed: int) -> Path:
    before = set(out.iterdir()) if out.exists() else set()
    code = cli.main(["train", str(LANDER), "--popsize", "6", "--trials", "2", "--max-generations", "4", "--workers", "0",
                     "--out", str(out), "--seed", str(seed), "--validate-every", "24"])
    assert code == 0
    (new,) = set(out.iterdir()) - before
    return new


SYN_COMPARE = ["", "", "10", "10", "50", "90", "90"]
SYN_SUCCESS = ["", "", "0.1", "0.1", "0.4", "0.75", "0.75"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Runs:
    base = tmp_path_factory.mktemp("compare") / "runs"
    g1 = _train(base, 1)
    g2 = _train(base, 2)
    single = _train(base, 7)
    for i, run in enumerate((g1, g2)):
        RunStore.open(run).update_run_info(group={"id": GROUP_ID, "name": GROUP_NAME, "index": i, "count": 3})

    # 값을 정해 둔 합성 실행 (목표 점수 도달, 최종 점수를 정확히 확인)
    syn = RunStore.create(base, "합성", {"env_name": "합성 환경", "method": "cmaes", "created": "2020-01-01T00:00:00",
                                       "config": {"method": "cmaes", "run_seed": 5}})
    for g in range(7):
        syn.append_stats({
            "generation": g + 1, "evaluations": 100 * (g + 1), "elapsed": 1.5 * (g + 1),
            "compare_score": SYN_COMPARE[g], "compare_success": SYN_SUCCESS[g], "champion_score": "",
            "score_best": float(g),
        })
    # 예전 형식(비교 열 없음) + 쓰다 끊긴 마지막 줄
    old = RunStore.create(base, "예전", {"env_name": "예전 환경", "method": "cmaes", "created": "2020-01-02T00:00:00",
                                       "config": {"method": "cmaes", "run_seed": 9}})
    (old.root / "stats.csv").write_bytes(b"\xef\xbb\xbfgeneration,evaluations,score_best\r\n1,10,-5\r\n2,20,-3\r\n3,3")
    # 엑셀이 CP949로 다시 저장한 통계
    bad = RunStore.create(base, "엑셀", {"env_name": "엑셀 환경", "method": "cmaes", "created": "2020-01-03T00:00:00",
                                       "config": {"method": "cmaes", "run_seed": 4}})
    (bad.root / "stats.csv").write_bytes("세대,평가\r\n1,2\r\n".encode("cp949"))
    broken = base / "망가진 실행"
    broken.mkdir()
    (broken / "run.json").write_text("{망가짐", encoding="utf-8")
    empty = base / "빈 폴더"
    empty.mkdir()
    return Runs(base, g1, g2, single, syn.root, old.root, bad.root, broken, empty)


@pytest.fixture()
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def make_view(qtbot: Any, runs_dir: Path, settings: QSettings) -> CompareView:
    view = CompareView(lambda: runs_dir, settings=settings)
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.show()
    return view


def stats(run: Path) -> list[dict[str, str]]:
    return RunStore.open(run).read_stats()


def col(run: Path, name: str) -> np.ndarray:
    return np.array([float(r[name]) if r.get(name, "") != "" else math.nan for r in stats(run)])


def expected_to_target(run: Path, target: float) -> float:
    for r in stats(run):
        v = r.get("compare_score", "")
        if v != "" and float(v) >= target:
            return float(r["evaluations"])
    return math.nan


def last_compare(run: Path) -> float:
    vals = [float(r["compare_score"]) for r in stats(run) if r.get("compare_score", "") != ""]
    return vals[-1] if vals else math.nan


def shown_of(view: CompareView, kind: str, contains: Path | None = None) -> Any:
    for s in view.shown:
        if s.kind == kind and (contains is None or any(e.path == contains for e in s.entries)):
            return s
    raise AssertionError(f"no shown {kind} {contains}")


# ---------------------------------------------------------------------------
# 계산
def test_series_helpers() -> None:
    y = np.array([math.nan, 1.0, math.nan, math.nan, 3.0, math.nan])
    assert np.array_equal(ffill(y), np.array([math.nan, 1, 1, 1, 3, 3]), equal_nan=True)
    sx, sy = step_xy(np.array([0.0, 10.0, 20.0]), np.array([1.0, 2.0, 3.0]))
    assert sx.tolist() == [0, 10, 10, 20, 20] and sy.tolist() == [1, 1, 2, 2, 3]

    grid = np.array([-5.0, 0.0, 5.0, 10.0, 25.0])
    x, yy = np.array([0.0, 10.0, 20.0]), np.array([1.0, 2.0, 3.0])
    assert np.array_equal(sample(x, yy, grid, step=True), [math.nan, 1, 1, 2, math.nan], equal_nan=True)
    assert np.array_equal(sample(x, yy, grid, step=False), [math.nan, 1, 1.5, 2, math.nan], equal_nan=True)

    a = (np.array([0.0, 10.0, 20.0]), np.array([1.0, 2.0, 3.0]))
    b = (np.array([0.0, 10.0, 20.0, 30.0]), np.array([math.nan, 5.0, 6.0, 7.0]))
    g = aggregate([a, b], step=True)
    assert g.x.tolist() == [0, 10, 20, 30]
    assert g.count.tolist() == [1, 2, 2, 1]
    assert np.allclose(g.mean, [1, 3.5, 4.5, 7])
    assert g.lo.tolist() == [1, 2, 3, 7] and g.hi.tolist() == [1, 5, 6, 7]
    assert g.full.tolist() == [False, True, True, False]
    assert g.partial.tolist() == [True, False, False, True]
    # 격자 점이 많으면 줄인다
    many = (np.arange(10000.0), np.arange(10000.0))
    assert len(aggregate([many], step=False, max_grid=500).x) == 500


def test_metrics_from_synthetic_run(runs: Runs) -> None:
    data = load_stats(runs.synthetic)
    m = run_metrics(data, 80.0)
    assert m.final == 90 and m.to_target == 600 and m.elapsed == pytest.approx(10.5)
    assert m.success == pytest.approx(75.0) and m.generations == 7
    assert run_metrics(data, 10.0).to_target == 300
    assert math.isnan(run_metrics(data, 1000.0).to_target)
    x, y = series(data, "evaluations", "compare_score")
    assert x.tolist() == [100, 200, 300, 400, 500, 600, 700]
    assert np.array_equal(y, [math.nan, math.nan, 10, 10, 50, 90, 90], equal_nan=True)
    _, pct = series(data, "generation", "compare_success")
    assert pct[-1] == pytest.approx(75.0)
    # 예전 형식: 없는 열은 NaN, 끊긴 마지막 줄은 버린다
    old = load_stats(runs.old)
    assert old.n == 2 and np.isnan(old.col("compare_score")).all()
    assert math.isnan(run_metrics(old, 0.0).final)
    bad = load_stats(runs.cp949)
    assert bad.n == 0 and bad.error


# ---------------------------------------------------------------------------
# 목록
def test_list_groups_runs_and_group_checkbox(qtbot: Any, runs: Runs, settings: QSettings) -> None:
    view = make_view(qtbot, runs.dir, settings)
    top = view.group_item(GROUP_ID)
    assert top is not None and top.childCount() == 2
    assert top.text(0) == GROUP_NAME
    assert top.text(LIST_COL_GROUP) == S.CMP_GROUP_MEMBERS.format(n=2, count=3)
    assert [top.child(i).data(0, Qt.ItemDataRole.UserRole) for i in range(2)] == [str(runs.g1), str(runs.g2)]
    single = view.run_item(runs.single)
    assert single is not None and single.parent() is None and single.text(2) == "7"
    for broken in (runs.broken, runs.empty):
        item = view.run_item(broken)
        assert item is not None and not item.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert view.shown == [] and view.stack.currentWidget() is view.message

    # 묶음 체크박스 하나로 묶음 전체
    top.setCheckState(0, Qt.CheckState.Checked)
    qtbot.waitUntil(lambda: len(view.shown) == 1)
    assert view.shown[0].kind == "group" and set(view.selected_runs()) == {runs.g1, runs.g2}
    assert top.child(0).checkState(0) == Qt.CheckState.Checked
    top.child(1).setCheckState(0, Qt.CheckState.Unchecked)
    qtbot.waitUntil(lambda: len(view.shown) == 1 and len(view.shown[0].entries) == 1)
    assert top.checkState(0) == Qt.CheckState.PartiallyChecked
    # 고른 것은 새로 고침 뒤에도 그대로
    view.refresh()
    assert view.selected_runs() == [runs.g1]
    # 읽을 수 없는 폴더는 고를 수 없다
    view.set_selected_runs([runs.broken, runs.single])
    assert view.selected_runs() == [runs.single]
    assert view.set_group_checked(GROUP_ID)
    assert set(view.selected_runs()) == {runs.single, runs.g1, runs.g2}
    assert not view.set_group_checked("없는 묶음")


# ---------------------------------------------------------------------------
# 그래프와 표
def test_curves_and_table_values(qtbot: Any, runs: Runs, settings: QSettings) -> None:
    view = make_view(qtbot, runs.dir, settings)
    view.set_selected_runs([runs.g1, runs.g2, runs.single])
    assert sorted(s.kind for s in view.shown) == ["group", "run"]
    run = shown_of(view, "run", runs.single)
    # 기본: 가로축 누적 시험 횟수, 세로축 비교 점수(이어 가는 계단, 첫 비교 전에는 NaN)
    assert view.x_axis == "evaluations" and view.y_metric == "compare_score"
    assert np.array_equal(run.x, col(runs.single, "evaluations"))
    assert np.array_equal(run.y, ffill(col(runs.single, "compare_score")), equal_nan=True)
    assert math.isnan(run.y[0]) and np.isfinite(run.y[1:]).all()  # 검증(비교)은 시험 24회째부터

    # 묶음: 공통 격자 위 평균·최소·최대
    group = shown_of(view, "group")
    assert group.name == GROUP_NAME and len(group.entries) == 2
    members = [(col(r, "evaluations"), ffill(col(r, "compare_score"))) for r in (runs.g1, runs.g2)]
    grid = np.unique(np.concatenate([m[0] for m in members]))
    assert np.array_equal(group.x, grid)
    vals = np.vstack([sample(x, y, grid, True) for x, y in members])
    assert np.allclose(group.curve.mean[1:], vals[:, 1:].mean(axis=0))
    assert np.allclose(group.curve.lo[1:], vals[:, 1:].min(axis=0))
    assert np.allclose(group.curve.hi[1:], vals[:, 1:].max(axis=0))
    assert group.curve.full[1:].all()

    # 표: 최종 점수, 목표 도달 시험 횟수, 소요 시간, 성공률
    finals = [last_compare(r) for r in (runs.g1, runs.g2)]
    first_value = float(next(r["compare_score"] for r in stats(runs.single) if r["compare_score"] != ""))
    view.set_target(first_value)
    rows = {r.is_group: r for r in view.rows}
    single_row, group_row = rows[False], rows[True]
    assert single_row.final == pytest.approx(last_compare(runs.single))
    assert single_row.to_target == expected_to_target(runs.single, first_value) == 24
    assert single_row.elapsed == pytest.approx(col(runs.single, "elapsed")[-1])
    assert single_row.success == pytest.approx(100 * col(runs.single, "compare_success")[-1])
    assert group_row.final == pytest.approx(np.mean(finals))
    assert group_row.final_std == pytest.approx(np.std(finals, ddof=1))
    reached = [t for t in (expected_to_target(r, first_value) for r in (runs.g1, runs.g2)) if math.isfinite(t)]
    assert group_row.reached == len(reached)
    if reached:
        assert group_row.to_target == pytest.approx(np.mean(reached))
    assert group_row.elapsed == pytest.approx(np.mean([col(r, "elapsed")[-1] for r in (runs.g1, runs.g2)]))
    texts = view.table_texts()
    assert len(texts) == 2
    single_texts = next(t for t in texts if t[0] == run.name)
    group_texts = next(t for t in texts if t[0] == GROUP_NAME)
    assert single_texts[TABLE_COL_TARGET] == "24"
    assert " ± " in group_texts[TABLE_COL_FINAL]
    assert group_texts[2] == S.CMP_SEED_RANGE.format(first=1, last=2, n=2)
    # 아무도 못 닿는 목표
    view.set_target(10**6)
    assert all(math.isnan(r.to_target) for r in view.rows)
    assert all(t[TABLE_COL_TARGET] == S.CMP_NONE for t in view.table_texts())

    # 묶음 평균을 끄면 실행마다 선 하나
    view.set_group_mean(False)
    assert [s.kind for s in view.shown] == ["run", "run", "run"]
    view.set_group_mean(True)
    assert len(view.shown) == 2


def test_axis_and_metric_switching(qtbot: Any, runs: Runs, settings: QSettings) -> None:
    view = make_view(qtbot, runs.dir, settings)
    view.set_selected_runs([runs.single, runs.g1, runs.g2])
    for axis in X_AXES:
        view.x_box.setCurrentIndex(view.x_box.findData(axis))  # 화면에서 고른 것처럼
        assert view.x_axis == axis
        run = shown_of(view, "run", runs.single)
        assert np.array_equal(run.x, col(runs.single, axis))
        assert view.plot.getAxis("bottom").labelText == S.CMP_X_NAMES[axis]
    view.set_y_metric("compare_success")
    run = shown_of(view, "run", runs.single)
    assert np.array_equal(run.y, ffill(100 * col(runs.single, "compare_success")), equal_nan=True)
    assert view.plot.getAxis("left").labelText == S.CMP_Y_AXIS_NAMES["compare_success"]
    view.set_y_metric("champion_score")
    assert np.array_equal(shown_of(view, "run", runs.single).y, ffill(col(runs.single, "champion_score")),
                          equal_nan=True)
    view.set_y_metric("score_best")  # 세대마다 바뀌는 값: 이어 가지 않고 선형 보간
    assert np.array_equal(shown_of(view, "run", runs.single).y, col(runs.single, "score_best"))
    group = shown_of(view, "group")
    xs = [col(r, "generation") for r in (runs.g1, runs.g2)]
    ys = [col(r, "score_best") for r in (runs.g1, runs.g2)]
    assert np.allclose(group.curve.mean, (ys[0] + ys[1]) / 2) and np.array_equal(group.x, xs[0])
    # 설정은 저장되고 다음에 열 때 그대로
    view2 = make_view(qtbot, runs.dir, settings)
    assert view2.x_axis == "generation" and view2.y_metric == "score_best"
    with pytest.raises(ValueError):
        view.set_x_axis("nope")


def test_export_png_and_csv(qtbot: Any, runs: Runs, settings: QSettings, tmp_path: Path) -> None:
    view = make_view(qtbot, runs.dir, settings)
    assert not view.png_button.isEnabled()
    view.set_selected_runs([runs.g1, runs.g2, runs.single, runs.synthetic])
    assert view.png_button.isEnabled() and view.csv_button.isEnabled()
    png = tmp_path / "out" / "비교.png"
    assert view.save_plot_png(png)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # 단추: 확장자가 없으면 붙인다
    view.ask_save_path = lambda *_a: tmp_path / "그래프"  # type: ignore[method-assign]
    view.png_button.click()
    assert (tmp_path / "그래프.png").is_file()
    view.ask_save_path = lambda *_a: tmp_path / "표"  # type: ignore[method-assign]
    view.set_target(80)
    view.csv_button.click()
    raw = (tmp_path / "표.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # 엑셀에서 한글이 깨지지 않게 BOM
    with open(tmp_path / "표.csv", encoding="utf-8-sig", newline="") as f:
        table = list(csv.reader(f))
    assert tuple(table[0]) == S.CMP_CSV_HEADERS and len(table) == 1 + len(view.rows) == 4
    syn = next(r for r in table[1:] if r[0] == S.CMP_RUN_NAME.format(env="합성 환경", method="CMA-ES", seed=5))
    h = list(S.CMP_CSV_HEADERS)
    assert float(syn[h.index("최종 점수(평균)")]) == 90
    assert float(syn[h.index("목표 점수 도달까지 시험 횟수(평균)")]) == 600
    assert float(syn[h.index("성공률(%, 평균)")]) == 75
    group = next(r for r in table[1:] if r[0] == GROUP_NAME)
    assert group[h.index("실행 수")] == "2" and group[h.index("시드")] == "1 2"


def test_robust_to_broken_and_old_runs(qtbot: Any, runs: Runs, settings: QSettings) -> None:
    view = make_view(qtbot, runs.dir, settings)
    view.set_selected_runs([runs.old, runs.cp949, runs.synthetic])
    assert len(view.shown) == 3
    texts = {t[0]: t for t in view.table_texts()}
    old = texts[S.CMP_RUN_NAME.format(env="예전 환경", method="CMA-ES", seed=9)]
    assert old[TABLE_COL_FINAL] == S.CMP_NONE and old[TABLE_COL_TARGET] == S.CMP_NONE
    bad = next(r for r in view.rows if r.paths == [runs.cp949])
    assert math.isnan(bad.final) and bad.errors
    # 비교 점수가 아무 데도 없으면 안내
    view.set_selected_runs([runs.old, runs.cp949])
    assert view.note_label.isVisibleTo(view)
    view.set_y_metric("score_best")
    assert not view.note_label.isVisibleTo(view)


def test_rereads_growing_run(qtbot: Any, tmp_path: Path, settings: QSettings) -> None:
    """쓰는 중인 실행: 줄이 늘면 다시 읽고, 그대로면 캐시를 쓴다."""
    base = tmp_path / "runs"
    store = RunStore.create(base, "도는 중", {"env_name": "환경", "method": "cmaes", "config": {"run_seed": 3}})
    row = {"generation": 1, "evaluations": 10, "elapsed": 0.5, "compare_score": 7.0, "compare_success": 0.0,
           "champion_score": 7.0, "score_best": 9.0}
    store.append_stats(row)
    view = make_view(qtbot, base, settings)
    view.set_selected_runs([store.root])
    first = view.shown[0].datas[0]
    view.redraw(keep_view=True)
    assert view.shown[0].datas[0] is first
    store.append_stats({**row, "generation": 2, "evaluations": 20, "compare_score": 95.0})
    view.redraw(keep_view=True)
    assert len(view.shown[0].x) == 2 and view.rows[0].final == 95


def test_default_target_from_env(qtbot: Any, settings: QSettings, tmp_path: Path) -> None:
    import json

    env = json.loads(LANDER.read_text(encoding="utf-8"))
    env["evaluation"]["target_score"] = 42.5
    base = tmp_path / "runs"
    store = RunStore.create(base, "목표", {"env_name": "목표 환경", "method": "cmaes", "config": {"run_seed": 1}})
    store.add_env_version(json.dumps(env, ensure_ascii=False), 0)
    view = make_view(qtbot, base, settings)
    assert view.target == 80.0
    view.set_selected_runs([store.root])
    assert view.target == 42.5
    view.set_target(55.0)  # 사용자가 정하면 그대로
    view.set_selected_runs([])
    view.set_selected_runs([store.root])
    assert view.target == 55.0


# ---------------------------------------------------------------------------
# 반복 실행
def test_repeat_dialog_methods_and_plan(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dlg = RepeatDialog(runs_dir=tmp_path, workers=1)
    qtbot.addWidget(dlg)
    layout = layout_for(load_env(LANDER))
    expected = []
    for m in ("cmaes", "ga", "sa"):
        try:
            make_learner(TrainConfig(method=m, popsize=64 if m == "ga" else 32), layout)
            expected.append(m)
        except Exception:  # 아직 만들지 않은(만드는 중인) 방법
            pass
    assert dlg.methods() == expected and "cmaes" in expected
    assert dlg.env_name == "달 착륙선" and dlg.count_spin.value() == 5
    plan = dlg.plan()
    assert plan.count == 5 and plan.seeds() == [1, 2, 3, 4, 5] and plan.workers == 1 and plan.runs_dir == tmp_path
    assert plan.config["max_evaluations"] == 30000 and plan.config["max_time"] is None
    assert plan.config["popsize"] == 32
    assert [plan.config_for(i)["run_seed"] for i in range(5)] == [1, 2, 3, 4, 5]
    dlg.count_spin.setValue(3)
    dlg.base_seed.setValue(10)
    dlg.stop_time.setChecked(True)
    dlg.max_minutes.setValue(2.5)
    dlg.trials.setValue(2)
    dlg.group_name.setText("내 묶음")
    plan = dlg.plan()
    assert plan.seeds() == [10, 11, 12] and plan.group_name == "내 묶음"
    assert plan.config["max_time"] == 150.0 and plan.config["max_evaluations"] is None and plan.config["trials"] == 2
    if "sa" in expected:
        assert dlg.set_method("sa")
        assert dlg.size_label.text() == S.CMP_REPEAT_NEIGHBORS and dlg.plan().config["sa_neighbors"] == 8
    # 아직 만들지 않은 방법은 목록에 없다
    real = repeat_mod.make_learner

    def fake(config: TrainConfig, lay: Any) -> Any:
        if config.method == "ga":
            raise NotImplementedError
        if config.method == "sa":
            raise ValueError(config.method)
        return real(config, lay)

    monkeypatch.setattr(repeat_mod, "make_learner", fake)
    assert available_methods(layout) == ["cmaes"]
    # 읽을 수 없는 환경 파일
    broken = tmp_path / "망가짐.evoenv"
    broken.write_text("{", encoding="utf-8")
    assert not dlg.set_env_file(broken)
    errors: list[str] = []
    accepted: list[bool] = []
    dlg.show_error = errors.append  # type: ignore[method-assign]
    dlg.accepted.connect(lambda: accepted.append(True))
    dlg.accept()
    assert errors and not accepted


def _plan(runs_dir: Path, count: int, max_evaluations: int, validate_every: int = 1000) -> RepeatPlan:
    config = TrainConfig(popsize=6, trials=2, max_evaluations=max_evaluations, validate_every=validate_every)
    return RepeatPlan(env_text=LANDER.read_text(encoding="utf-8"), config=config.to_dict(), count=count, base_seed=11,
                      workers=1, runs_dir=runs_dir, group_name="테스트 묶음", env_name="달 착륙선")


def test_repeat_runs_through_compare_view(qtbot: Any, tmp_path: Path, settings: QSettings) -> None:
    base = tmp_path / "runs"
    view = make_view(qtbot, base, settings)
    seen: list[tuple[int, bool, int]] = []
    progress: list[tuple[int, int]] = []
    finished: list[bool] = []
    try:
        runner = view.start_repeat(_plan(base, 2, 24, validate_every=12))
        assert runner is not None and view.repeat_running and not view.repeat_button.isEnabled()
        gid = runner.group["id"]

        def on_started(index: int, run_dir: Path) -> None:
            # 새 실행 폴더가 생기자마자 목록의 묶음 아래에 보이고, 그래프에도 들어간다
            top = view.group_item(gid)
            seen.append((index, view.run_item(run_dir) is not None and view.run_item(run_dir).parent() is top,
                         len(view.shown[0].entries) if view.shown else 0))

        runner.run_started.connect(on_started)
        runner.progress.connect(lambda i, g, _e: progress.append((i, g)))
        runner.finished.connect(finished.append)
        assert view.start_repeat(_plan(base, 2, 24)) is None  # 하나씩만
        qtbot.waitUntil(lambda: bool(finished), timeout=240000)
        assert finished == [True] and runner.state == "done" and runner.fraction() == 1.0
        assert [s[0] for s in seen] == [0, 1] and all(s[1] for s in seen)
        assert [s[2] for s in seen] == [1, 2]
        assert {i for i, _ in progress} == {0, 1} and max(g for _, g in progress) == 2
        assert len(runner.run_dirs) == 2 and runner.client is None
        for i, run in enumerate(runner.run_dirs):
            info = RunStore.open(run).read_run_info()
            assert info["group"] == {"id": gid, "name": "테스트 묶음", "index": i, "count": 2}
            assert info["config"]["run_seed"] == 11 + i and info["config"]["max_evaluations"] == 24
            assert stats(run)[-1]["evaluations"] == "24"
            assert any(e["kind"] == "stop" and e.get("reason") == "max_evaluations" for e in RunStore.open(run).read_events())
        assert not view.repeat_running and view.repeat_button.isEnabled() and not view.repeat_box.isVisibleTo(view)
        assert view.repeat_status.text() == S.CMP_REPEAT_DONE.format(name="테스트 묶음", n=2)
        top = view.group_item(gid)
        assert top is not None and top.childCount() == 2 and top.checkState(0) == Qt.CheckState.Checked
        assert len(view.shown) == 1 and view.shown[0].kind == "group" and len(view.shown[0].entries) == 2
        assert np.isfinite(view.shown[0].y).any()
        assert view.rows[0].is_group and view.rows[0].runs == 2
    finally:
        view.shutdown()


def test_repeat_cancel_stops_cleanly(qtbot: Any, tmp_path: Path) -> None:
    base = tmp_path / "runs"
    runner = RepeatRunner(_plan(base, 3, 10**7))
    finished: list[bool] = []
    progress: list[int] = []
    runner.finished.connect(finished.append)
    runner.progress.connect(lambda _i, g, _e: progress.append(g))
    try:
        runner.start()
        qtbot.waitUntil(lambda: len(progress) >= 2, timeout=120000)
        assert runner.running and runner.client is not None and runner.client.active
        runner.cancel()
        assert finished == [False] and runner.state == "cancelled" and runner.client is None
        qtbot.wait(1500)  # 다음 실행을 띄우지 않는다
        assert finished == [False] and len(runner.run_dirs) == 1
        assert [p for p in base.iterdir() if not p.name.startswith(".")] == runner.run_dirs
        store = RunStore.open(runner.run_dirs[0])
        assert store.checkpoints(), "취소해도 체크포인트를 남긴다"
        assert store.read_run_info()["group"]["index"] == 0
        assert any(e["kind"] == "stop" and e.get("reason") == "requested" for e in store.read_events())
        runner.cancel()  # 두 번 불러도 괜찮다
        assert finished == [False]
    finally:
        runner.shutdown()

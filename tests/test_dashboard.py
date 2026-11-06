"""M3: 실험실 통계 대시보드 (타일, 가로축 전환, 세로선, 정체 음영, 분포 지도, 배치 저장, 속도)."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from evolab.app import strings as S
from evolab.app.lab.dashboard import (
    HEATMAP_BINS,
    TILE_IDS,
    Dashboard,
    ScoreHistogram,
    fmt_value,
    parse_scores,
    stagnation_ranges,
    to_float,
)

END_EVENTS = [
    ("success", "성공"),
    ("off_pad", "패드 밖 착지"),
    ("crash", "충돌"),
    ("out_of_bounds", "이탈"),
    ("timeout", "시간 초과"),
]
BLOCKS = [("landing", "착지"), ("distance", "패드까지 거리"), ("fuel", "연료 사용"), ("speed", "속력")]
POP = 32


def synthetic_rows(n: int, *, seed: int = 0, contrib: bool = True, strings: bool = False) -> list[dict[str, Any]]:
    """착륙선 학습처럼 보이는 가짜 통계 줄. strings=True면 RunStore.read_stats()처럼 값이 글자."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    champion = math.nan
    evals = 0
    elapsed = 0.0
    for g in range(1, n + 1):
        plateau = min(g, 180)  # 180세대 뒤로는 거의 안 오름 -> 정체 구간
        mu = -120 + 260 * (1 - math.exp(-plateau / 70))
        spread = 70 * math.exp(-g / 120) + 6
        scores = rng.normal(mu, spread, POP)
        scores[rng.integers(POP)] -= 150  # 가끔 크게 실패하는 개체
        evals += POP * 5
        gen_seconds = 2.6 + 0.3 * rng.random()
        elapsed += gen_seconds
        validation: float | str = ""
        if g % 5 == 0:
            validation = float(np.max(scores) - 10 + rng.normal(0, 3))
            if not (validation <= champion):
                champion = validation
        q = np.quantile(scores, [0.25, 0.5, 0.75])
        sigma = 0.5 * math.exp(-g / 200)
        steps = int(POP * 5 * 600 * (1 + 0.3 * rng.random()))
        row: dict[str, Any] = {
            "generation": g,
            "evaluations": evals,
            "validation_evaluations": (g // 5) * 20,
            "score_best": float(scores.max()),
            "score_mean": float(scores.mean()),
            "score_q25": float(q[0]),
            "score_median": float(q[1]),
            "score_q75": float(q[2]),
            "score_min": float(scores.min()),
            "validation_best": validation,
            "champion_score": champion if math.isfinite(champion) else "",
            "champion_generation": g,
            "sigma": sigma,
            "diversity": sigma * 0.9,
            "distance": sigma * 0.9 * math.sqrt(2 * 211),
            "steps": steps,
        }
        success = min(0.85, g / 250)
        rest = rng.dirichlet([3, 2, 1, 2]) * (1 - success)
        for (eid, _), frac in zip(END_EVENTS, [success, *rest], strict=True):
            row[f"end_{eid}"] = float(frac)
        if contrib:
            for kind, k in (("rep", 1.0), ("mean", 0.7)):
                row[f"contrib_{kind}_landing"] = k * 100 * success
                row[f"contrib_{kind}_distance"] = -k * 40 * (1 - success)
                row[f"contrib_{kind}_fuel"] = -k * (5 + 3 * rng.random())
                row[f"contrib_{kind}_speed"] = -k * 10 * (1 - success) + 2 * rng.random()
        row["scores"] = " ".join(f"{s:.3f}" for s in scores)
        row["gen_seconds"] = gen_seconds
        row["elapsed"] = elapsed
        row["steps_per_second"] = steps / gen_seconds
        row["worker_utilization"] = 0.85 + 0.1 * rng.random()
        if strings:
            row = {k: (v if isinstance(v, str) else repr(float(v)) if isinstance(v, float) else str(v)) for k, v in row.items()}
        rows.append(row)
    return rows


def make_dashboard(qtbot: QtBot, tmp_path: Path, name: str = "settings.ini") -> Dashboard:
    settings = QSettings(str(tmp_path / name), QSettings.Format.IniFormat)
    dash = Dashboard(settings=settings)
    qtbot.addWidget(dash)
    dash.set_meta(end_events=END_EVENTS, blocks=BLOCKS, stagnation_generations=40)
    return dash


def curve_xy(curve: Any) -> tuple[np.ndarray, np.ndarray]:
    x, y = curve.getData()
    return np.asarray(x), np.asarray(y)


def test_parse_helpers() -> None:
    assert math.isnan(to_float("")) and math.isnan(to_float(None)) and math.isnan(to_float("abc"))
    assert to_float("1.5") == 1.5 and to_float(3) == 3.0
    assert parse_scores("12.5 -3.25  7").tolist() == [12.5, -3.25, 7.0]
    assert parse_scores("").size == 0 and parse_scores(None).size == 0
    assert parse_scores("1 nan x 2").tolist() == [1.0, 2.0]
    assert parse_scores([1.0, 2.0]).tolist() == [1.0, 2.0]


def test_tile_ids_and_titles(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    assert dash.tile_ids == ["trend", "heatmap", "diversity", "blocks", "ends", "perf"]
    assert list(TILE_IDS) == dash.tile_ids
    for tid in dash.tile_ids:
        assert dash.tile(tid).title.text() == S.DASH_TILE_TITLES[tid]


def test_set_rows_with_csv_strings_then_live_rows(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows = synthetic_rows(120, strings=True)
    assert all(isinstance(v, str) for v in rows[0].values())
    dash.set_rows(rows)
    assert dash.table.n == 120
    trend = dash.tile("trend")
    x, y = curve_xy(trend.lines["score_best"])
    assert len(x) == 120 and x[0] == 1 and x[-1] == 120
    assert y[0] == pytest.approx(float(rows[0]["score_best"]))
    # 첫 검증 전 챔피언 점수는 '' -> NaN
    _, champ = curve_xy(trend.champion)
    assert np.isnan(champ[:4]).all() and np.isfinite(champ[4:]).all()

    live = synthetic_rows(121)[-1]
    dash.add_row(live)
    assert dash.table.n == 121
    x, y = curve_xy(trend.lines["score_best"])
    assert len(x) == 121 and y[-1] == pytest.approx(live["score_best"])
    for tid in dash.tile_ids:
        assert not dash.tile(tid).dirty


def test_empty_and_missing_values_do_not_crash(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows = [
        {"generation": "1", "score_best": "", "score_mean": "", "scores": ""},
        {"generation": "2", "score_best": "3.5", "sigma": "", "elapsed": "", "evaluations": ""},
        {"score_best": "4"},  # 세대 열도 없음 -> 줄 순서로
        {"generation": 4, "score_best": 5.0, "scores": "1 2 3", "end_success": 1.0, "worker_utilization": ""},
    ]
    dash.set_rows(rows)
    assert dash.table.col("generation").tolist() == [1.0, 2.0, 3.0, 4.0]
    _, y = curve_xy(dash.tile("trend").lines["score_best"])
    assert np.isnan(y[0]) and y[1:].tolist() == [3.5, 4.0, 5.0]
    assert np.isnan(dash.col("distance")).all()
    for axis in ("evaluations", "elapsed", "generation"):
        dash.set_axis(axis)
    dash.set_marks([(2, "resume"), (3, "something_else")])
    dash.add_row({"generation": 5})
    dash.add_row({})
    assert dash.table.n == 6
    m = dash.hist.matrix()
    assert m.shape == (6, HEATMAP_BINS)
    assert m[3].sum() == pytest.approx(1.0) and m[0].sum() == 0
    dash.clear()
    assert dash.table.n == 0 and dash.hist.n == 0
    dash.add_row({"generation": 1, "score_best": 1.0})


def test_formula_env_without_contrib_columns_shows_message(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    blocks = dash.tile("blocks")
    dash.set_rows(synthetic_rows(30, contrib=False, strings=True))
    assert blocks.stack.currentWidget() is blocks.message
    assert blocks.message.text() == S.DASH_BLOCKS_NONE
    dash.set_rows(synthetic_rows(30))
    assert blocks.stack.currentWidget() is blocks.view


def test_block_contributions_stack_positive_and_negative(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows = synthetic_rows(50)
    dash.set_rows(rows)
    blocks = dash.tile("blocks")
    assert [name for _, name in blocks._series] == [name for _, name in BLOCKS]
    r = rows[-1]
    values = {b: r[f"contrib_rep_{b}"] for b, _ in BLOCKS}
    top_pos = curve_xy(blocks._pos[-1])[1][-1]
    top_neg = curve_xy(blocks._neg[-1])[1][-1]
    assert top_pos == pytest.approx(sum(v for v in values.values() if v > 0))
    assert top_neg == pytest.approx(sum(v for v in values.values() if v < 0))
    x, _ = curve_xy(blocks._pos[0])
    assert len(x) == 51  # 세대마다 막대: 경계 n+1개
    assert curve_xy(blocks.total)[1][-1] == pytest.approx(sum(values.values()))
    blocks.set_mode("mean")
    top_pos_mean = curve_xy(blocks._pos[-1])[1][-1]
    assert top_pos_mean == pytest.approx(sum(r[f"contrib_mean_{b}"] for b, _ in BLOCKS if r[f"contrib_mean_{b}"] > 0))
    assert S.DASH_BLOCKS_TOTAL in blocks.legend.text()


def test_end_causes_use_meta_labels_and_sum_to_100(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    dash.set_rows(synthetic_rows(40))
    ends = dash.tile("ends")
    text = ends.legend.text()
    for _, label in END_EVENTS:
        assert label in text
    top = curve_xy(ends._pos[-1])[1]
    assert np.allclose(top, 100.0)
    # 환경 정보에 없는 종료 열도 이름 그대로 보인다
    dash.add_row({"generation": 41, "end_mystery": 1.0})
    assert "mystery" in ends.legend.text()


def test_axis_switching(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows = synthetic_rows(60)
    dash.set_rows(rows)
    trend = dash.tile("trend")
    evals = np.array([r["evaluations"] for r in rows], dtype=float)
    elapsed = np.array([r["elapsed"] for r in rows], dtype=float)

    dash.set_axis("evaluations")
    assert dash.axis_box.currentData() == "evaluations"
    assert np.array_equal(curve_xy(trend.lines["score_mean"])[0], evals)
    assert np.array_equal(curve_xy(dash.tile("perf").curves["gen_seconds"])[0], evals)
    assert np.array_equal(curve_xy(dash.tile("diversity").curves["sigma"])[0], evals)
    assert np.array_equal(curve_xy(dash.tile("ends")._pos[0])[0], evals)
    assert trend.plots[-1].getAxis("bottom").labelText == S.DASH_AXIS_NAMES["evaluations"]
    # 분포 지도와 블록 막대는 언제나 세대 기준
    assert curve_xy(dash.tile("blocks").total)[0][-1] == 60
    assert dash.tile("heatmap").plots[0].getAxis("bottom").labelText == S.DASH_AXIS_NAMES["generation"]

    dash.axis_box.setCurrentIndex(2)  # 콤보 상자로 바꿔도 같다
    assert dash.axis == "elapsed"
    assert np.allclose(curve_xy(trend.champion)[0], elapsed)
    dash.add_row(synthetic_rows(61)[-1])
    assert len(curve_xy(trend.champion)[0]) == 61
    dash.set_axis("generation")
    assert curve_xy(trend.champion)[0][-1] == 61
    with pytest.raises(ValueError):
        dash.set_axis("weeks")


def visible_marks(tile: Any, plot_index: int = 0) -> list[float]:
    return sorted(line.value() for line in tile._mark_pools[plot_index] if line.isVisible())


def test_marks_on_every_tile_and_follow_axis(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows = synthetic_rows(150)
    dash.set_rows(rows)
    dash.set_marks([(50, "resume"), (120, "env_change"), (9999, "branch")])
    for tid in dash.tile_ids:
        tile = dash.tile(tid)
        for k in range(len(tile.plots)):
            assert visible_marks(tile, k) == [50.0, 120.0], tid
    trend = dash.tile("trend")
    labels = [line.label.format for line in trend._mark_pools[0] if line.isVisible()]
    assert labels == [S.DASH_MARK_LABELS["resume"], S.DASH_MARK_LABELS["env_change"]]
    dash.set_axis("evaluations")
    assert visible_marks(trend) == [float(rows[49]["evaluations"]), float(rows[119]["evaluations"])]
    assert visible_marks(dash.tile("heatmap")) == [50.0, 120.0]
    dash.set_marks([])
    assert visible_marks(trend) == []


def test_my_best_line(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    dash.set_rows(synthetic_rows(30))
    trend = dash.tile("trend")
    assert not trend.my_best.isVisible()
    dash.set_my_best(321.5)
    assert trend.my_best.isVisible() and trend.my_best.value() == 321.5
    assert S.DASH_MY_BEST in trend.legend.text()
    # 가로선이 자료보다 높아도 자동 범위에 들어간다
    bounds = trend.plot.vb.childrenBounds()
    assert bounds[1][1] >= 321.5
    dash.set_my_best(None)
    assert not trend.my_best.isVisible()
    assert S.DASH_MY_BEST not in trend.legend.text()


def test_stagnation_ranges_and_shading(qtbot: QtBot, tmp_path: Path) -> None:
    gens = np.arange(1, 11, dtype=float)
    champ = np.array([np.nan, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    assert stagnation_ranges(gens, champ, 3) == [(2.0, 5.0), (5.0, 10.0)]
    assert stagnation_ranges(gens, champ, 4) == [(5.0, 10.0)]
    assert stagnation_ranges(gens, np.full(10, np.nan), 1) == []
    assert stagnation_ranges(gens[:0], champ[:0], 1) == []

    dash = make_dashboard(qtbot, tmp_path)
    dash.set_rows(synthetic_rows(300))
    trend = dash.tile("trend")
    assert trend.ranges, "plateau after generation 180 should be shaded"
    assert all(b - a >= 40 for a, b in trend.ranges)
    shown = [r.getRegion() for r in trend._regions if r.isVisible()]
    assert len(shown) == len(trend.ranges)
    assert S.DASH_STAGNATION_LEGEND.format(n=40) in trend.legend.text()
    dash.set_meta(end_events=END_EVENTS, blocks=BLOCKS, stagnation_generations=None)
    assert dash.stagnation_generations == 50


def test_heatmap_shape_and_range_growth(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows = synthetic_rows(200)
    dash.set_rows(rows[:100])
    heat = dash.tile("heatmap")
    assert heat.image_array().shape == (100, HEATMAP_BINS)
    assert np.allclose(dash.hist.matrix().sum(axis=1), 1.0)
    for r in rows[100:]:
        dash.add_row(r)
    m = dash.hist.matrix()
    assert m.shape == (200, HEATMAP_BINS) and heat.image_array().shape == (200, HEATMAP_BINS)
    assert np.allclose(m.sum(axis=1), 1.0)
    # 바로 넣은 것과 한 번에 다시 센 것이 같다
    again = ScoreHistogram()
    for r in rows:
        again.append(parse_scores(r["scores"]), update=False)
    again.finish()
    again_lo, again_hi = again.lo, again.hi
    assert m.max() <= 1.0
    if (again_lo, again_hi) == (dash.hist.lo, dash.hist.hi):
        assert np.allclose(again.matrix(), m)
    # 평균이 위로 올라가면 진한 칸도 위로 이동한다
    centers = np.linspace(dash.hist.lo, dash.hist.hi, HEATMAP_BINS)
    assert (m[-1] * centers).sum() > (m[0] * centers).sum()


def test_score_histogram_rebuilds_rarely_and_stays_exact() -> None:
    rng = np.random.default_rng(3)
    h = ScoreHistogram(bins=20)
    for g in range(2000):
        h.append(rng.normal(g * 0.5, 5.0, 16))
    assert h.n == 2000
    assert h.rebuilds < 25  # 범위를 넉넉히 넓혀 다시 세는 일은 드물다
    # 같은 구간으로 처음부터 센 값과 같다
    full = ScoreHistogram(bins=20)
    rng = np.random.default_rng(3)
    for g in range(2000):
        full.append(rng.normal(g * 0.5, 5.0, 16), update=False)
    full.lo, full.hi = h.lo, h.hi
    full._rebuild()
    assert np.allclose(full.matrix(), h.matrix())


def test_hover_shows_values(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows = synthetic_rows(20)
    dash.set_rows(rows)
    trend = dash.tile("trend")
    trend.hover(10.2, 0.0, 0)
    text = trend.legend.text()
    assert S.DASH_HOVER_X["generation"].format(gen=10, x="10") in text
    assert f"{S.DASH_TREND_BEST} {fmt_value(rows[9]['score_best'])}" in text
    trend.unhover()
    assert S.DASH_HOVER_X["generation"].format(gen=10, x="10") not in trend.legend.text()
    heat = dash.tile("heatmap")
    heat.hover(5.0, float(rows[4]["score_median"]), 0)
    assert "%" in heat.legend.text()

    # 실제 마우스 이동 신호 경로(장면 좌표)도 같은 값을 보인다
    dash.resize(1200, 800)
    dash.show()
    qtbot.wait(20)
    scene_pos = trend.plot.vb.mapViewToScene(QPointF(10.0, float(rows[9]["score_median"])))
    trend._on_mouse((scene_pos,))
    assert S.DASH_HOVER_X["generation"].format(gen=10, x="10") in trend.legend.text()
    assert any(line.isVisible() for line in trend._cross)
    trend._on_mouse((QPointF(-1000.0, -1000.0),))
    assert not any(line.isVisible() for line in trend._cross)


def test_plain_wheel_is_left_to_scroll_area_and_ctrl_wheel_zooms(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    dash.resize(1000, 700)
    dash.show()
    dash.set_rows(synthetic_rows(100))
    qtbot.wait(20)
    trend = dash.tile("trend")
    viewport = trend.view.viewport()
    before = trend.plot.vb.viewRange()

    def wheel(mods: Qt.KeyboardModifier) -> QWheelEvent:
        local = QPointF(viewport.width() / 2, viewport.height() / 2)
        glob = QPointF(viewport.mapToGlobal(local.toPoint()))
        return QWheelEvent(
            local, glob, QPoint(0, 0), QPoint(0, 120), Qt.MouseButton.NoButton, mods, Qt.ScrollPhase.NoScrollPhase, False
        )

    plain = wheel(Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(viewport, plain)
    assert not plain.isAccepted()
    assert trend.plot.vb.viewRange() == before
    QApplication.sendEvent(viewport, wheel(Qt.KeyboardModifier.ControlModifier))
    assert trend.plot.vb.viewRange() != before


def test_tiles_menu(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    dash._fill_tiles_menu()
    actions = [a for a in dash.tiles_menu.actions() if a.isCheckable()]
    assert [a.text() for a in actions] == [S.DASH_TILE_TITLES[t] for t in TILE_IDS]
    assert all(a.isChecked() for a in actions)
    actions[1].setChecked(False)
    assert not dash.is_tile_visible("heatmap")
    actions[1].setChecked(True)
    assert dash.is_tile_visible("heatmap")


def test_layout_visibility_order_and_settings_round_trip(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    dash.set_rows(synthetic_rows(20))
    dash.set_tile_visible("perf", False)
    assert not dash.is_tile_visible("perf")
    dash.move_tile("ends", 0)
    dash.move_tile_by("trend", 1)  # 보이는 타일 사이에서 한 칸 뒤로
    dash.set_axis("elapsed")
    dash.set_columns(3)
    dash.tile("blocks").set_mode("mean")
    order = dash.tile_order()
    assert order[0] == "ends" and order.index("trend") == order.index("heatmap") + 1
    with pytest.raises(KeyError):
        dash.set_tile_visible("nope", True)

    # 숨긴 타일은 갱신하지 않고, 다시 켤 때 채운다
    dash.add_row(synthetic_rows(21)[-1])
    perf = dash.tile("perf")
    assert perf.dirty
    dash.set_tile_visible("perf", True)
    assert not perf.dirty and len(curve_xy(perf.curves["gen_seconds"])[0]) == 21
    dash.set_tile_visible("perf", False)

    again = make_dashboard(qtbot, tmp_path)
    assert again.tile_order() == order
    assert not again.is_tile_visible("perf") and again.is_tile_visible("trend")
    assert again.axis == "elapsed" and again.axis_box.currentData() == "elapsed"
    assert again.columns == 3
    assert again.tile("blocks").mode == "mean"

    again.reset_layout()
    third = make_dashboard(qtbot, tmp_path)
    assert third.tile_order() == list(TILE_IDS) and third.columns == 2
    assert all(third.is_tile_visible(t) for t in TILE_IDS)


def test_bad_settings_are_ignored(qtbot: QtBot, tmp_path: Path) -> None:
    path = tmp_path / "bad.ini"
    s = QSettings(str(path), QSettings.Format.IniFormat)
    s.beginGroup("dashboard")
    s.setValue("order", "nope,perf,perf,trend")
    s.setValue("hidden", ["ghost", "heatmap"])
    s.setValue("columns", "many")
    s.setValue("axis", "weeks")
    s.setValue("blocks_mode", 7)
    s.endGroup()
    s.sync()
    dash = make_dashboard(qtbot, tmp_path, "bad.ini")
    assert dash.tile_order() == ["perf", "trend", "heatmap", "diversity", "blocks", "ends"]
    assert not dash.is_tile_visible("heatmap")
    assert dash.columns == 2 and dash.axis == "generation" and dash.tile("blocks").mode == "rep"


def test_add_row_speed_with_1000_rows(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    dash.resize(1400, 900)
    dash.show()
    rows = synthetic_rows(1040)
    t0 = time.perf_counter()
    dash.set_rows(rows[:1000])
    set_rows_ms = (time.perf_counter() - t0) * 1000
    dash.set_marks([(200, "resume"), (600, "env_change")])
    qtbot.wait(10)
    times = []
    for r in rows[1000:]:
        t0 = time.perf_counter()
        dash.add_row(r)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    median, worst = times[len(times) // 2], times[-1]
    print(f"\nset_rows(1000): {set_rows_ms:.1f} ms, add_row with 1000+ rows: median {median:.2f} ms, max {worst:.2f} ms")
    assert worst < 100.0


def test_render_screenshot(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    dash.resize(1400, 1000)
    dash.show()
    dash.set_rows(synthetic_rows(300, strings=True))
    dash.set_marks([(90, "resume"), (210, "env_change")])
    dash.set_my_best(150.0)
    qtbot.wait(50)
    image = dash.grab().toImage()
    assert image.width() > 0 and image.height() > 0
    out = os.environ.get("EVOLAB_DASHBOARD_PNG")
    target = Path(out) if out else tmp_path / "dashboard.png"
    assert image.save(str(target))

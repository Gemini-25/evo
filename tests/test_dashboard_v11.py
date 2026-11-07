"""v1.1 대시보드: 학습 방법 교체 자동 세로선(switch 열), 지금 학습 방법 최고점 선, 예선 띠 (설계 11.2, 테스트 30)."""

from __future__ import annotations

import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot
from test_dashboard import POP, curve_xy, make_dashboard, synthetic_rows

from evolab.app import strings as S
from evolab.app.lab.dashboard import Dashboard, fmt_value
from evolab.runs.store import RunStore

V10_RUN = Path(__file__).parent / "data" / "v10_run_ga" / "run"
NEW_COLUMNS = ("learner", "learner_trials", "learner_best", "learner_stall", "restarts", "qualifying", "switch")


def switch_rows() -> list[dict[str, Any]]:
    """switch가 0,0,1,0,3,4인 여섯 줄과 통과자를 알려 주는 일곱째 줄(설계 15절 테스트 30)."""
    plan = [
        # learner, learner_trials, learner_best, learner_stall, restarts, qualifying, switch
        (0, 12, "", 12, 0, 0, 0),
        (0, 24, 5.0, 0, 0, 0, 0),
        (0, 36, 5.0, 24000, 0, 0, 1),
        (1, 12, "", 12, 1, 1, 0),
        (1, 24, 7.25, 0, 1, 1, 3),
        (2, 24, 9.5, 0, 1, 1, 4),
        (1, 36, 7.25, 12, 1, 0, 0),  # 예선 끝 다음 줄: 통과자는 learner 1(출전 2)
    ]
    rows = []
    for g, values in enumerate(plan, start=1):
        row: dict[str, Any] = {
            "generation": g,
            "evaluations": 12 * g + 100,
            "elapsed": 1.5 * g,
            "score_best": float(g),
            "champion_score": 5.0 if g >= 2 else "",
            "scores": "1 2 3",
        }
        row.update(zip(NEW_COLUMNS, values, strict=True))
        rows.append(row)
    return rows


def stability_rows(
    n: int = 200,
    *,
    entrants: int = 3,
    per_entrant: int = 10,
    entrant_best: tuple[float, ...] = (50.0, 80.0, 60.0),
    restart_at: tuple[int, ...] = (80, 140),
    strings: bool = False,
) -> list[dict[str, Any]]:
    """v1.1 열이 든 착륙선 같은 통계 줄: 예선 출전 entrants개(출전마다 per_entrant세대, entrants=1이면 예선 없음),
    restart_at 세대 끝에 자동 재시작(번갈아: 처음 상태, 챔피언). 검증은 5세대마다와 출전 마지막 세대."""
    base = synthetic_rows(n)
    gen_trials = POP * 5
    qual_end = entrants * per_entrant if entrants > 1 else 0
    bests = [math.nan] * (entrants + len(restart_at) + 1)
    trials = [0] * len(bests)
    ref = [0] * len(bests)
    learner, restarts = 0, 0
    rows = []
    for g, row in enumerate(base, start=1):
        qualifying = g <= qual_end
        if qualifying:
            learner = (g - 1) // per_entrant
        trials[learner] += gen_trials
        last_of_entrant = qualifying and g % per_entrant == 0
        if g % 5 == 0 or last_of_entrant:
            score = (entrant_best[learner] if learner < entrants else 20.0 * restarts) - 30.0 / (1 + trials[learner] / 1000)
            if not score <= bests[learner]:
                bests[learner] = score
                ref[learner] = trials[learner]
        switch = 0
        if last_of_entrant:
            switch = 4 if g == qual_end else 3
        elif g in restart_at:
            switch = 1 if restarts % 2 == 0 else 2
        best = bests[learner]
        row.update(
            learner=learner,
            learner_trials=trials[learner],
            learner_best=best if math.isfinite(best) else "",
            learner_stall=trials[learner] - ref[learner],
            restarts=restarts,
            qualifying=int(qualifying),
            switch=switch,
        )
        if switch == 4:  # 통과자: 첫 최댓값(동점은 앞 번호)
            learner = int(np.argmax(np.array(bests[:entrants])))
        elif switch in (1, 2):
            restarts += 1
            learner = entrants + restarts - 1 if entrants > 1 else restarts
        if strings:
            row = {k: (v if isinstance(v, str) else repr(float(v)) if isinstance(v, float) else str(v)) for k, v in row.items()}
        rows.append(row)
    return rows


def shown(tile: Any, plot_index: int = 0) -> list[tuple[float, str]]:
    """보이는 세로선 (가로축 값, 풍선 도움말), 가로축 순."""
    return sorted((line.value(), line.toolTip()) for line in tile._mark_pools[plot_index] if line.isVisible())


def shown_x(tile: Any, plot_index: int = 0) -> list[float]:
    return [x for x, _ in shown(tile, plot_index)]


def drawn_labels(tile: Any) -> list[tuple[float, str]]:
    """보이는 세로선 (가로축 값, 실제로 그려진 이름표 글자), 가로축 순. label.format(틀)이 아니라 글자 칸을 본다."""
    return sorted((line.value(), line.label.textItem.toPlainText()) for line in tile._mark_pools[0] if line.isVisible())


def line_at(tile: Any, x: float, plot_index: int = 0) -> Any:
    found = [line for line in tile._mark_pools[plot_index] if line.isVisible() and line.value() == x]
    assert len(found) == 1, (x, found)
    return found[0]


def pen_of(line: Any) -> tuple[str, Qt.PenStyle, float]:
    pen = line.pen
    return pen.color().name(), pen.style(), pen.widthF()


def expected_pens(dash: Dashboard) -> dict[str, tuple[str, Qt.PenStyle, float]]:
    t = dash.theme
    return {
        "restart": (t.series[7], Qt.PenStyle.SolidLine, 1.5),
        "qualify": (t.series[6], Qt.PenStyle.DotLine, 1.5),
        "event": (t.ink2, Qt.PenStyle.DashLine, 1.0),
    }


def assert_no_v11_drawing(dash: Dashboard) -> None:
    trend = dash.tile("trend")
    assert dash.auto_marks == [] and dash.qualify_rows is None and not dash.learner_switched
    assert not trend.learner_line.isVisible() and len(curve_xy(trend.learner_line)[0]) == 0
    assert not trend.qualify_band.isVisible()
    text = trend.legend.text()
    assert S.DASH_TREND_LEARNER not in text and S.DASH_QUALIFY_LEGEND not in text


# ---------------------------------------------------------------------------
def test_switch_marks_generations_labels_pens_and_tips(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows = switch_rows()
    dash.set_rows(rows[:6])
    assert [(g, k) for g, k, _ in dash.auto_marks] == [(3, "restart"), (5, "qualify_next"), (6, "qualify_pick")]
    info = dash.auto_marks[0][2]
    assert (info["switch"], info["learner"], info["learner_stall"], info["learner_best"]) == (1.0, 0.0, 24000.0, 5.0)

    tip_restart = S.DASH_MARK_TIP_RESTART.format(gen=3, where=S.RESTART_FROM_NAMES_SHORT["fresh"], stall=24000, best="5.00")
    tip_next = S.DASH_MARK_TIP_QUALIFY_NEXT.format(gen=5, n=2, best="7.25")
    # 다음 줄이 아직 없으면 예선 표시의 learner_best 첫 최댓값(출전 3)으로 채운다
    assert shown(dash.tile("trend")) == [
        (3.0, tip_restart),
        (5.0, tip_next),
        (6.0, S.DASH_MARK_TIP_QUALIFY_PICK.format(gen=6, winner=3)),
    ]
    # 다음 줄이 오면 그 줄의 learner가 통과자다(여기서는 일부러 다른 값: 출전 2)
    dash.add_row(rows[6])
    tip_pick = S.DASH_MARK_TIP_QUALIFY_PICK.format(gen=6, winner=2)
    assert dash.auto_marks[-1][2]["winner"] == 1.0
    pens = expected_pens(dash)
    for tid in dash.tile_ids:
        tile = dash.tile(tid)
        for k in range(len(tile.plots)):
            assert shown(tile, k) == [(3.0, tip_restart), (5.0, tip_next), (6.0, tip_pick)], (tid, k)
            assert pen_of(line_at(tile, 3.0, k)) == pens["restart"], (tid, k)
            assert pen_of(line_at(tile, 5.0, k)) == pens["qualify"], (tid, k)
            assert pen_of(line_at(tile, 6.0, k)) == pens["qualify"], (tid, k)
    trend = dash.tile("trend")
    labels = [line_at(trend, x).label.format for x in (3.0, 5.0, 6.0)]
    assert labels == [S.DASH_MARK_LABELS[k] for k in ("restart", "qualify_next", "qualify_pick")]
    assert [x for x, _ in drawn_labels(trend)] == [3.0, 5.0, 6.0] and [t for _, t in drawn_labels(trend)] == labels
    # 처음부터 일곱 줄을 한 번에 열어도 같다
    again = make_dashboard(qtbot, tmp_path, "again.ini")
    again.set_rows(rows)
    assert shown(again.tile("trend")) == shown(trend)


def test_mark_labels_are_drawn_on_lines_shown_later_and_reused(qtbot: QtBot, tmp_path: Path) -> None:
    """pyqtgraph 이름표는 숨은 선에 쓴 글자를 그리지 않고, 선을 다시 보일 때도 고치지 않는다. 학습 중에 더한 사건 선,
    자동 재시작으로 늘어난 선, 다른 실행을 열어 다시 쓰는 선에도 제 이름표가 그려진다."""
    names = S.DASH_MARK_LABELS
    dash = make_dashboard(qtbot, tmp_path)
    trend = dash.tile("trend")
    rows = switch_rows()
    dash.set_rows(rows[:2])
    dash.set_marks([(1, "resume")])  # 실험실의 '적용' 뒤(_update_marks)처럼 학습 중에 더한 사건
    assert drawn_labels(trend) == [(1.0, names["resume"])]
    dash.add_row(rows[2])  # 3세대 끝 자동 재시작: 선이 하나 늘어난다
    assert drawn_labels(trend) == [(1.0, names["resume"]), (3.0, names["restart"])]
    dash.set_marks([(1, "resume"), (2, "config_change")])
    assert drawn_labels(trend) == [(1.0, names["resume"]), (2.0, names["config_change"]), (3.0, names["restart"])]
    # 다른 실행을 열 때(실험실 순서: 비우기 → 사건 없음 → 통계 줄): 숨겼던 선을 다시 써도 앞 표시의 글자가 남지 않는다
    dash.clear()
    dash.set_marks([])
    assert drawn_labels(trend) == []
    dash.set_rows(stability_rows(200, entrants=1))
    assert [(g, k) for g, k, _ in dash.auto_marks] == [(80, "restart"), (140, "restart")]
    assert drawn_labels(trend) == [(80.0, names["restart"]), (140.0, names["restart"])]


def test_champion_restart_tip_and_csv_strings(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows = stability_rows(200, strings=True)
    assert all(isinstance(v, str) for v in rows[0].values())
    dash.set_rows(rows)
    kinds = [(g, k) for g, k, _ in dash.auto_marks]
    assert kinds == [(10, "qualify_next"), (20, "qualify_next"), (30, "qualify_pick"), (80, "restart"), (140, "restart")]
    by_gen = {g: rows[g - 1] for g, _ in kinds}
    tips = dict(shown(dash.tile("trend")))
    for g, where in ((80, "fresh"), (140, "champion")):
        r = by_gen[g]
        assert tips[float(g)] == S.DASH_MARK_TIP_RESTART.format(
            gen=g, where=S.RESTART_FROM_NAMES_SHORT[where], stall=int(r["learner_stall"]), best=fmt_value(float(r["learner_best"]))
        )
    assert tips[10.0] == S.DASH_MARK_TIP_QUALIFY_NEXT.format(gen=10, n=1, best=fmt_value(float(by_gen[10]["learner_best"])))
    assert tips[30.0] == S.DASH_MARK_TIP_QUALIFY_PICK.format(gen=30, winner=int(rows[30]["learner"]) + 1)
    assert rows[30]["learner"] == "1"  # 출전 2가 가장 높았다
    # 숫자 줄로 연 것과 같다
    numeric = make_dashboard(qtbot, tmp_path, "numeric.ini")
    numeric.set_rows(stability_rows(200))
    assert shown(numeric.tile("trend")) == shown(dash.tile("trend"))
    assert numeric.qualify_rows == dash.qualify_rows == (0, 29)


def test_event_and_auto_marks_on_every_axis(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows = stability_rows(200)
    dash.set_rows(rows)
    dash.set_marks([(50, "resume"), (120, "config_change"), (9999, "branch")])
    gens = [10.0, 20.0, 30.0, 50.0, 80.0, 120.0, 140.0]
    pens = expected_pens(dash)
    style_at = {10.0: "qualify", 20.0: "qualify", 30.0: "qualify", 50.0: "event", 80.0: "restart", 120.0: "event", 140.0: "restart"}
    for axis in ("generation", "evaluations", "elapsed"):
        dash.set_axis(axis)
        xs = [float(rows[int(g) - 1][axis]) for g in gens]
        for tid in dash.tile_ids:
            tile = dash.tile(tid)
            want = xs if tile.follows_axis else gens
            for k in range(len(tile.plots)):
                assert shown_x(tile, k) == sorted(want), (axis, tid, k)
                for g, x in zip(gens, want, strict=True):
                    assert pen_of(line_at(tile, x, k)) == pens[style_at[g]], (axis, tid, k, g)
    trend = dash.tile("trend")
    assert line_at(trend, float(rows[49]["elapsed"])).toolTip() == S.DASH_MARK_TOOLTIP.format(
        gen=50, label=S.DASH_MARK_LABELS["resume"]
    )
    # 사건 표시를 바꾸거나 지워도 자동 표시는 남고, 펜은 선 자리에 맞게 바뀐다
    dash.set_axis("generation")
    dash.set_marks([])
    assert shown_x(trend) == [10.0, 20.0, 30.0, 80.0, 140.0]
    assert pen_of(line_at(trend, 80.0)) == pens["restart"] and pen_of(line_at(trend, 10.0)) == pens["qualify"]
    dash.set_marks([(60, "resume")])
    assert shown_x(trend) == [10.0, 20.0, 30.0, 60.0, 80.0, 140.0]
    assert pen_of(line_at(trend, 60.0)) == pens["event"] and pen_of(line_at(trend, 140.0)) == pens["restart"]


def test_old_rows_draw_nothing_new(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    dash.set_rows(synthetic_rows(120, strings=True))
    dash.set_marks([(40, "resume")])
    assert_no_v11_drawing(dash)
    pens = expected_pens(dash)
    for tid in dash.tile_ids:
        tile = dash.tile(tid)
        for k in range(len(tile.plots)):
            assert shown_x(tile, k) == [40.0]
            assert pen_of(line_at(tile, 40.0, k)) == pens["event"]
    dash.add_row(synthetic_rows(121)[-1])
    assert_no_v11_drawing(dash)

    # 실제 v1.0 실행의 stats.csv (복사본을 연다)
    run = tmp_path / "v10"
    shutil.copytree(V10_RUN, run)
    rows = RunStore.open(run).read_stats()
    assert rows and not any(c in rows[0] for c in NEW_COLUMNS)
    dash.set_rows(rows)
    assert dash.table.n == len(rows) and not dash.table.has("switch")
    assert_no_v11_drawing(dash)
    for axis in ("evaluations", "elapsed", "generation"):
        dash.set_axis(axis)
        assert_no_v11_drawing(dash)

    # v1.1 실행이라도 재시작·예선이 없으면(learner 늘 0, switch 0) 새로 그릴 것이 없다
    off = stability_rows(120, entrants=1, restart_at=())
    assert {r["learner"] for r in off} == {0} and {r["switch"] for r in off} == {0}
    dash.set_rows(off)
    assert_no_v11_drawing(dash)


def test_learner_line_only_after_a_switch(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows = stability_rows(200, entrants=1, restart_at=(80,))
    dash.set_rows(rows[:80])  # 80세대 끝에 재시작했지만 아직 learner 0의 줄뿐
    trend = dash.tile("trend")
    assert [k for _, k, _ in dash.auto_marks] == ["restart"]
    assert not dash.learner_switched and not trend.learner_line.isVisible()
    dash.add_row(rows[80])
    assert dash.learner_switched and trend.learner_line.isVisible()
    for r in rows[81:]:
        dash.add_row(r)
    x, y = curve_xy(trend.learner_line)
    want = np.array([float(r["learner_best"]) if r["learner_best"] != "" else np.nan for r in rows])
    assert np.array_equal(x, np.arange(1, 201, dtype=float))
    assert np.array_equal(y, want, equal_nan=True)
    # 재시작 뒤 첫 검증 전은 비어 있고, 그 뒤 챔피언보다 아래에서 다시 오른다
    assert np.isnan(y[80:84]).all() and np.isfinite(y[84])
    champ = curve_xy(trend.champion)[1]
    assert y[84] < champ[84]
    assert S.DASH_TREND_LEARNER in trend.legend.text()
    trend.hover(100.0, 0.0, 0)
    assert f"{S.DASH_TREND_LEARNER} {fmt_value(want[99])}" in trend.legend.text()
    trend.unhover()
    dash.set_axis("evaluations")
    assert np.array_equal(curve_xy(trend.learner_line)[0], np.array([float(r["evaluations"]) for r in rows]))


def test_qualify_band_range(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows = stability_rows(200, entrants=4, per_entrant=12, entrant_best=(10.0, 30.0, 30.0, 20.0))
    dash.set_rows(rows[:20])  # 예선 도중
    trend = dash.tile("trend")
    assert trend.qualify_band.isVisible() and trend.qualify_band.getRegion() == (1.0, 20.0)
    assert S.DASH_QUALIFY_LEGEND in trend.legend.text()
    for r in rows[20:]:
        dash.add_row(r)
    assert dash.qualify_rows == (0, 47)
    assert trend.qualify_band.getRegion() == (1.0, 48.0)
    # 동점(출전 2와 3)은 앞 번호
    assert int(rows[48]["learner"]) == 1
    assert dict(shown(trend))[48.0] == S.DASH_MARK_TIP_QUALIFY_PICK.format(gen=48, winner=2)
    for axis in ("evaluations", "elapsed"):
        dash.set_axis(axis)
        assert trend.qualify_band.getRegion() == (float(rows[0][axis]), float(rows[47][axis]))
    assert trend.qualify_band.zValue() == -35
    # 예선이 없으면 띠도 범례도 없다
    dash.set_rows(stability_rows(100, entrants=1))
    assert not trend.qualify_band.isVisible() and S.DASH_QUALIFY_LEGEND not in trend.legend.text()


def test_clear_and_set_rows_reset_auto_marks(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows = stability_rows(200)
    dash.set_rows(rows)
    trend = dash.tile("trend")
    assert len(dash.auto_marks) == 5 and dash.learner_switched and trend.qualify_band.isVisible()
    dash.clear()
    assert_no_v11_drawing(dash)
    for tid in dash.tile_ids:
        tile = dash.tile(tid)
        for k in range(len(tile.plots)):
            assert shown_x(tile, k) == []
    dash.set_rows(rows)
    assert shown_x(trend) == [10.0, 20.0, 30.0, 80.0, 140.0]
    dash.set_rows(rows[:50])  # 다시 열면 앞 실행의 표시를 버린다
    assert shown_x(trend) == [10.0, 20.0, 30.0]
    dash.set_rows(synthetic_rows(60))
    assert_no_v11_drawing(dash)
    assert shown_x(trend) == []
    dash.add_row(rows[60])  # 다른 실행의 줄을 넣어도 앞 표시는 되살아나지 않는다
    assert dash.auto_marks == []


def test_hidden_tile_gets_marks_when_shown(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows = stability_rows(200)
    dash.set_rows(rows[:70])
    dash.set_tile_visible("perf", False)
    for r in rows[70:]:
        dash.add_row(r)
    perf = dash.tile("perf")
    assert perf.dirty
    dash.set_tile_visible("perf", True)
    for k in range(len(perf.plots)):
        assert shown_x(perf, k) == [10.0, 20.0, 30.0, 80.0, 140.0]


def test_bad_switch_values_are_ignored_or_get_a_short_tip(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows: list[dict[str, Any]] = [
        {"generation": 1, "switch": ""},
        {"generation": 2, "switch": "abc"},
        {"generation": 3, "switch": 7},
        {"generation": 4, "switch": 1.5},
        {"generation": 5, "switch": -1},
        {"generation": 6, "switch": 0, "learner": "", "qualifying": ""},
        {"generation": 7, "switch": "1"},  # 정체 시험 수가 빠진 줄
        {"generation": 8, "switch": 3},  # learner가 빠진 줄
        {"generation": 9, "switch": 4, "learner": 0},  # 예선 기록이 없어 통과자를 모름
    ]
    dash.set_rows(rows)
    assert [(g, k) for g, k, _ in dash.auto_marks] == [(7, "restart"), (8, "qualify_next"), (9, "qualify_pick")]
    tips = dict(shown(dash.tile("trend")))
    for g, kind in ((7, "restart"), (8, "qualify_next"), (9, "qualify_pick")):
        assert tips[float(g)] == S.DASH_MARK_TOOLTIP.format(gen=g, label=S.DASH_MARK_LABELS[kind])
    assert not dash.learner_switched and dash.qualify_rows is None


def test_add_row_speed_with_restarts_and_long_run(qtbot: QtBot, tmp_path: Path) -> None:
    """수천 세대·잦은 재시작에서도 세대마다 갱신이 빠르다(줄마다 자동 표시 O(1), 바뀐 세로선만 고침)."""
    dash = make_dashboard(qtbot, tmp_path)
    dash.resize(1400, 900)
    dash.show()
    n = 2400
    restart_at = tuple(range(60, n + 41, 20))  # 재시작 약 120번 → 세로선 약 120개 × 그래프 9칸
    rows = stability_rows(n + 40, restart_at=restart_at)
    t0 = time.perf_counter()
    dash.set_rows(rows[:n])
    set_rows_ms = (time.perf_counter() - t0) * 1000
    dash.set_marks([(200, "resume"), (900, "env_change")])
    qtbot.wait(10)
    marks_before = len(dash.auto_marks)
    times = []
    for r in rows[n:]:
        t0 = time.perf_counter()
        dash.add_row(r)
        times.append((time.perf_counter() - t0) * 1000)
    assert len(dash.auto_marks) == marks_before + 2
    times.sort()
    median, worst = times[len(times) // 2], times[-1]
    print(f"\nset_rows({n}, {marks_before} auto marks): {set_rows_ms:.1f} ms, add_row: median {median:.2f} ms, max {worst:.2f} ms")
    assert worst < 100.0


def test_render_screenshot_v11(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    dash.resize(1400, 1000)
    dash.show()
    dash.set_rows(stability_rows(300, restart_at=(110, 220), strings=True))
    dash.set_marks([(160, "resume")])
    qtbot.wait(50)
    image = dash.grab().toImage()
    assert image.width() > 0 and image.height() > 0
    out = os.environ.get("EVOLAB_DASHBOARD_V11_PNG")
    target = Path(out) if out else tmp_path / "dashboard_v11.png"
    assert image.save(str(target))


@pytest.mark.parametrize("strings", [False, True])
def test_stability_rows_fixture_is_consistent(strings: bool) -> None:
    """가짜 줄 자체 점검: 설계 8절 예처럼 예선 끝 다음 줄은 통과자의 시험 수를 잇고, 재시작 다음 줄은 새 학습 방법이다."""
    rows = stability_rows(200, strings=strings)
    f = [{k: float(v) if k in NEW_COLUMNS and v != "" else v for k, v in r.items()} for r in rows]
    assert [r["switch"] for r in f if r["switch"]] == [3, 3, 4, 1, 2]
    assert f[29]["qualifying"] == 1 and f[30]["qualifying"] == 0
    assert f[30]["learner"] == 1 and f[30]["learner_trials"] == f[19]["learner_trials"] + POP * 5
    assert f[80]["learner"] == 3 and f[80]["restarts"] == 1 and f[80]["learner_trials"] == POP * 5
    assert f[80]["learner_best"] == "" and f[140]["learner"] == 4

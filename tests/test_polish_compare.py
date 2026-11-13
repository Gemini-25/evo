"""6.2.0 다듬기 U24·U29 (polish/plan.md 2.4, 완료 기준 C11): 비교 화면의 색 칸과 숫자 형식.

- 하나를 더 체크하면 그 실행이 다음 칸 k를 받고 다른 실행의 색은 그대로.
- 체크를 풀면 가장 높은 칸의 실행 하나만 빈 칸으로 내려온다(색이 바뀌는 실행은 많아야 하나).
- 어느 때나 보이는 실행의 칸 집합 == {0..k−1}. 한꺼번에 고르는 길은 목록 차례(지금과 같은 결과).
- 체크한 줄의 0열에 그 색의 견본.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QSettings, Qt

from evolab.app import strings as S
from evolab.app import style
from evolab.app.compare.view import SWATCH_ROLE, TABLE_COL_FINAL, CompareView
from evolab.runs.store import RunStore

N_RUNS = 6


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("polish_compare") / "runs"
    for k in range(N_RUNS):
        store = RunStore.create(root, f"실행{k}", {
            "env_name": f"환경{k}", "method": "cmaes", "created": f"2020-01-0{k + 1}T00:00:00",
            "config": {"method": "cmaes", "run_seed": k + 1}})
        for g in range(3):
            store.append_stats({"generation": g + 1, "evaluations": 10 * (g + 1), "elapsed": 1.0 + g,
                                "compare_score": 1234.5 * (k + 1) + g, "compare_success": "", "champion_score": "",
                                "score_best": float(g)})
    return root


@pytest.fixture()
def view(qtbot: Any, base: Path, tmp_path: Path) -> CompareView:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    v = CompareView(lambda: base, settings=settings)
    qtbot.addWidget(v)
    v.resize(1280, 800)
    v.show()
    return v


def items(view: CompareView) -> list[Any]:
    return [view.run_item(e.path) for e in view.entries]


def check(qtbot: Any, view: CompareView, item: Any, on: bool) -> None:
    n = len(view.shown) + (1 if on else -1)
    item.setCheckState(0, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
    qtbot.waitUntil(lambda: len(view.shown) == n)


def colors(view: CompareView) -> dict[str, str]:
    return {s.entries[0].key: s.color for s in view.shown}


def slots_ok(view: CompareView) -> None:
    series = list(view.theme.series)
    assert sorted(series.index(s.color) for s in view.shown) == list(range(len(view.shown)))
    assert sorted(view.slots.values()) == list(range(len(view.shown)))


def test_adding_keeps_colors_and_removing_moves_at_most_one(qtbot: Any, view: CompareView) -> None:
    it = items(view)
    series = view.theme.series
    keys = [str(i.data(0, Qt.ItemDataRole.UserRole)) for i in it]
    # 셋을 차례로: 칸 0·1·2(목록 차례가 아니라 체크한 차례)
    for k in (2, 0, 4):
        check(qtbot, view, it[k], True)
        slots_ok(view)
    assert colors(view) == {keys[2]: series[0], keys[0]: series[1], keys[4]: series[2]}
    # 넷째를 더해도 앞 셋의 색 그대로
    before = colors(view)
    check(qtbot, view, it[1], True)
    after = colors(view)
    assert {k: after[k] for k in before} == before and after[keys[1]] == series[3]
    slots_ok(view)
    # 가운데(칸 1 = keys[0])를 풀면 칸 3의 실행(keys[1])만 칸 1로, 나머지 둘은 그대로
    check(qtbot, view, it[0], False)
    now = colors(view)
    assert now == {keys[2]: series[0], keys[1]: series[1], keys[4]: series[2]}
    slots_ok(view)
    # 가장 높은 칸을 풀면 아무것도 안 바뀜
    check(qtbot, view, it[4], False)
    assert colors(view) == {keys[2]: series[0], keys[1]: series[1]}
    slots_ok(view)
    # 다시 체크하면 다음 칸
    check(qtbot, view, it[0], True)
    assert colors(view)[keys[0]] == series[2]
    slots_ok(view)


def test_swatches_follow_the_colors(qtbot: Any, view: CompareView) -> None:
    it = items(view)
    check(qtbot, view, it[3], True)
    check(qtbot, view, it[1], True)
    by_key = colors(view)
    for item in it:
        key = str(item.data(0, Qt.ItemDataRole.UserRole))
        assert item.data(0, SWATCH_ROLE) == by_key.get(key)
        assert item.icon(0).isNull() == (key not in by_key)
    # 아이콘을 바꿔도 체크 바뀜으로 읽지 않는다(다시 그리기 고리가 없음)
    shown = list(view.shown)
    qtbot.wait(50)
    assert [s.entries[0].key for s in view.shown] == [s.entries[0].key for s in shown]
    check(qtbot, view, it[3], False)
    assert it[3].icon(0).isNull() and it[1].data(0, SWATCH_ROLE) == view.theme.series[0]


def test_bulk_selection_uses_list_order(view: CompareView) -> None:
    runs = [e.path for e in view.entries]
    view.set_selected_runs([runs[4], runs[1], runs[2]])
    series = view.theme.series
    order = [s.entries[0].path for s in view.shown]
    assert order == [runs[1], runs[2], runs[4]]
    assert [s.color for s in view.shown] == [series[0], series[1], series[2]]
    view.set_selected_runs([runs[4]])
    assert [s.color for s in view.shown] == [series[0]]
    view.set_selected_runs([])
    assert view.shown == [] and view.slots == {}


def test_table_scores_use_fmt_score(view: CompareView) -> None:
    runs = [e.path for e in view.entries]
    view.set_selected_runs(runs[:2])
    texts = view.table_texts()
    finals = sorted(r.final for r in view.rows)
    assert finals[-1] >= 1000
    cells = {t[TABLE_COL_FINAL] for t in texts}
    assert {style.fmt_score(f) for f in finals} == cells
    assert all("," in c for c in cells)


def test_axes_use_thousands_and_integer_generations(view: CompareView) -> None:
    from evolab.app.lab.dashboard import _Axis

    left, bottom = view.plot.getAxis("left"), view.plot.getAxis("bottom")
    assert isinstance(left, _Axis) and isinstance(bottom, _Axis)
    assert left.tickStrings([40000.0], 1.0, 10000.0) == ["40,000"]
    assert not bottom.integer
    view.set_x_axis("generation")
    assert bottom.integer and bottom.tickStrings([1.0, 2.0], 1.0, 1.0) == ["1", "2"]
    view.set_x_axis("evaluations")
    assert not bottom.integer


def test_race_lanes_of_a_group_get_distinct_colors(qtbot: Any, tmp_path: Path) -> None:
    """U25: 묶음이 실행 둘 이상을 내면 첫 칸은 묶음 색, 나머지는 보이는 항목의 칸 0..k−1 다음 칸부터."""
    root = tmp_path / "runs"
    paths: list[Path] = []
    for k in range(4):
        info: dict[str, Any] = {"env_name": "환경", "method": "cmaes", "created": f"2020-02-0{k + 1}T00:00:00",
                                "config": {"method": "cmaes", "run_seed": k + 1}}
        if k < 3:
            info["group"] = {"id": "g", "name": "묶음", "index": k, "count": 3}
        store = RunStore.create(root, f"r{k}", info)
        store.append_stats({"generation": 1, "evaluations": 10, "elapsed": 1.0, "compare_score": 1.0 + k,
                            "compare_success": "", "champion_score": "", "score_best": 1.0})
        paths.append(store.root)
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    v = CompareView(lambda: root, settings=settings)
    qtbot.addWidget(v)
    v.set_selected_runs(paths)
    assert len(v.shown) == 2
    series = v.theme.series
    group = next(s for s in v.shown if s.kind == "group")
    single = next(s for s in v.shown if s.kind == "run")
    lanes = dict(v.race_runs())
    members = [e.path for e in group.entries]
    assert lanes[members[0]] == group.color and lanes[single.entries[0].path] == single.color
    assert [lanes[p] for p in members[1:]] == [series[2], series[3]]
    assert len(set(lanes.values())) == 4


def test_layout_table_rows_folder_line_and_mixed_env_note(qtbot: Any, view: CompareView, base: Path) -> None:
    """U28: 표 최소 높이 = 머리 + min(줄 수, 5)줄, 왼쪽 처음 폭 360px, 폴더 경로 한 줄(전체는 풍선), 환경이 섞이면 보조 글."""
    from evolab.app.compare.view import LEFT_WIDTH

    assert view.split.sizes()[0] == LEFT_WIDTH
    runs = [e.path for e in view.entries]
    view.set_selected_runs(runs[:3])
    head = view.table.horizontalHeader().sizeHint().height()
    row = view.table.verticalHeader().defaultSectionSize()
    assert view.table.minimumHeight() >= head + 3 * row
    view.set_selected_runs(runs)
    assert view.table.minimumHeight() < head + 6 * row  # 5줄까지만
    # 서로 다른 환경(환경0…환경5): 보조 글, 하나면 없음
    assert view.mix_label.isVisibleTo(view) and view.mix_label.text() == S.CMP_MIXED_ENVS
    view.set_selected_runs(runs[:1])
    assert not view.mix_label.isVisibleTo(view)
    # 폴더 줄: 한 줄로 줄여 그림, 풍선에 전체
    assert view.folder_label.toolTip() == view.folder_label.text() and str(base) in view.folder_label.text()
    view.folder_label.resize(80, view.folder_label.height())
    assert "…" in view.folder_label.shown_text()


def test_empty_list_shows_text_and_repeat_button(qtbot: Any, tmp_path: Path) -> None:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    v = CompareView(lambda: tmp_path / "없음", settings=settings)
    qtbot.addWidget(v)
    v.resize(1280, 720)
    v.show()
    assert v.empty_label.isVisible() and v.empty_repeat_button.isVisible() and not v.tree.isVisible()
    assert v.empty_repeat_button.text() == S.CMP_REPEAT_BUTTON


def test_splitter_positions_are_remembered(qtbot: Any, base: Path, tmp_path: Path) -> None:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    a = CompareView(lambda: base, settings=settings)
    qtbot.addWidget(a)
    a.resize(1280, 800)
    a.show()
    a.split.moveSplitter(500, 1)
    a.right_split.moveSplitter(300, 1)
    sizes = (a.split.sizes(), a.right_split.sizes())
    assert sizes[0][0] == 500
    b = CompareView(lambda: base, settings=settings)
    qtbot.addWidget(b)
    b.resize(1280, 800)
    b.show()
    assert (b.split.sizes(), b.right_split.sizes()) == sizes

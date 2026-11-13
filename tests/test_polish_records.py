"""6.2.0 다듬기 U26·U29 (polish/plan.md 2.4): 기록 목록.

- 「열기」가 으뜸(고른 것이 없으면 꺼진 으뜸 + 까닭), 「삭제」는 위험 단추.
- 고른 것이 없으면 단추 줄 왼쪽에 안내 글. 꺼진 단추의 까닭은 style.set_enabled(toolTip()은 그대로).
- 줄의 오른쪽 클릭 메뉴(단추와 같은 행동·켜짐). 시작 시각은 fmt_when(풍선에 ISO), 점수는 fmt_score.
- 열 「시드」·「묶음」(RunSummary.seed·group), 「환경」 열 최소 180px.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import Qt

from evolab.app import strings as S
from evolab.app import style
from evolab.app.records.view import (
    COL_BEST,
    COL_ENV,
    COL_GROUP,
    COL_SEED,
    COL_STARTED,
    ENV_MIN_WIDTH,
    RecordsView,
)
from evolab.runs.catalog import list_runs
from evolab.runs.store import RunStore


@pytest.fixture()
def base(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    for k, (seed, group) in enumerate(((3, {"id": "g1", "name": "착륙 반복", "index": 0, "count": 2}),
                                       (4, {"id": "g1", "name": "착륙 반복", "index": 1, "count": 2}),
                                       (9, None))):
        info: dict[str, Any] = {"env_name": "달 착륙선", "method": "cmaes", "created": f"2024-03-0{k + 1}T12:0{k}:00",
                                "config": {"method": "cmaes", "run_seed": seed}}
        if group is not None:
            info["group"] = group
        store = RunStore.create(runs, f"r{k}", info)
        store.append_stats({"generation": 1, "score_best": 1824.25 + k, "champion_score": ""})
    (runs / "망가짐").mkdir()
    return runs


@pytest.fixture()
def view(qtbot: Any, base: Path) -> RecordsView:
    v = RecordsView(lambda: base)
    qtbot.addWidget(v)
    v.resize(1280, 720)
    v.show()
    return v


def row_of(view: RecordsView, name: str) -> int:
    for r in range(view.table.rowCount()):
        s = view._summary_at(r)
        if s is not None and s.name == name:
            return r
    raise AssertionError(name)


def test_summary_reads_seed_and_group(base: Path) -> None:
    by_name = {s.name: s for s in list_runs(base)}
    assert by_name["r0"].seed == 3 and by_name["r0"].group == "착륙 반복"
    assert by_name["r2"].seed == 9 and by_name["r2"].group == ""
    assert by_name["망가짐"].seed is None and by_name["망가짐"].group == ""


def test_columns_and_formats(view: RecordsView) -> None:
    t = view.table
    r = row_of(view, "r0")
    assert t.item(r, COL_SEED).text() == "3" and t.item(r, COL_GROUP).text() == "착륙 반복"
    assert t.item(row_of(view, "r2"), COL_GROUP).text() == S.REC_NONE
    assert t.item(r, COL_BEST).text() == style.fmt_score(1824.25) == "1,824"
    started = t.item(r, COL_STARTED)
    assert started.text() == style.fmt_when("2024-03-01T12:00:00") and "2024" in started.text()
    assert "2024-03-01 12:00:00" in started.toolTip()
    assert t.columnWidth(COL_ENV) >= ENV_MIN_WIDTH
    # 시드로 정렬하면 값으로(글자 아님)
    t.sortItems(COL_SEED, Qt.SortOrder.AscendingOrder)
    seeds = [t.item(i, COL_SEED).text() for i in range(t.rowCount())]
    assert seeds[1:] == ["3", "4", "9"]


def test_button_roles_reasons_and_note(view: RecordsView) -> None:
    assert view.open_button.property(style.ROLE_PROPERTY) == "primary"
    assert view.delete_button.property(style.ROLE_PROPERTY) == "danger"
    # 고른 것이 없음: 꺼진 으뜸 + 까닭, 안내 글
    assert not view.open_button.isEnabled() and style.why_off(view.open_button) == S.REC_WHY_PICK
    assert view.open_button.toolTip() == S.REC_OPEN_TIP  # 풍선 글은 그대로(까닭은 속성에만)
    assert style.tip_text(view.open_button) == f"{S.REC_OPEN_TIP}\n\n{S.REC_WHY_PICK}"
    for b in (view.resume_button, view.clone_button, view.memo_button, view.bundle_button, view.csv_button,
              view.delete_button, view.compare_button):
        assert not b.isEnabled() and style.why_off(b) == S.REC_WHY_PICK
    assert view.pick_note.isVisible() and view.pick_note.text() == S.REC_PICK_NOTE
    x = view.open_button.mapTo(view, view.open_button.rect().topLeft()).x()
    # 하나 고름: 모두 켜짐, 안내 글은 숨지만 단추 자리는 그대로
    view.table.selectRow(row_of(view, "r1"))
    assert view.open_button.isEnabled() and style.why_off(view.open_button) == ""
    assert [b for b in style.primaries(view) if b.isEnabled()] == [view.open_button]
    assert not view.pick_note.isVisible()
    assert view.open_button.mapTo(view, view.open_button.rect().topLeft()).x() == x
    # 둘 고름: 하나짜리 행동은 꺼짐(까닭), 비교는 켜짐
    view.table.selectAll()
    assert not view.open_button.isEnabled() and style.why_off(view.open_button) == S.REC_WHY_ONE
    assert view.compare_button.isEnabled()
    # 읽을 수 없는 실행
    view.table.clearSelection()
    view.table.selectRow(row_of(view, "망가짐"))
    assert style.why_off(view.open_button) == S.REC_WHY_UNREADABLE and view.delete_button.isEnabled()
    # 학습 중인 실행은 지울 수 없다(까닭)
    view.select_run(view.runs()[0].path)
    view.set_busy_runs([view.runs()[0].path])
    assert not view.delete_button.isEnabled() and style.why_off(view.delete_button) == S.REC_DELETE_BUSY


def test_context_menu_mirrors_buttons(qtbot: Any, view: RecordsView) -> None:
    menu = view.context_menu()
    acts = [a for a in menu.actions() if not a.isSeparator()]
    assert [a.text() for a in acts] == [S.REC_OPEN, S.REC_RESUME, S.REC_CLONE, S.REC_EDIT_MEMO, S.CMP_RECORDS_BUTTON,
                                        S.REC_BUNDLE, S.REC_CSV, S.REC_DELETE]
    assert not any(a.isEnabled() for a in acts)
    assert acts[0].toolTip().endswith(S.REC_WHY_PICK)
    menu.deleteLater()
    run = next(s.path for s in view.runs() if s.name == "r2")
    view.select_run(run)
    menu = view.context_menu()
    acts = [a for a in menu.actions() if not a.isSeparator()]
    assert all(a.isEnabled() for a in acts)
    with qtbot.waitSignal(view.open_requested) as sig:
        acts[0].trigger()
    assert sig.args == [run]
    menu.deleteLater()

"""6.2.0 다듬기 U23 (polish/plan.md 2.4, 완료 기준 C12): 나란히 경주 창의 키.

- 기본 단추는 「출발」(으뜸 단추). 창이 뜬 그대로 Enter·Space = 출발/일시 정지, 코스 번호 그대로.
- 다른 단추의 autoDefault는 그대로: 「내보내기…」에 초점을 두고 Space·Enter → 내보내기(출발 아님).
- 고르기 칸에서 R → 글자 찾기(처음부터 아님). 칸 밖에서 R → 처음부터.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from evolab import paths
from evolab.app import strings as S
from evolab.app import style
from evolab.app.race import model as M
from evolab.app.race.model import RaceEntry
from evolab.app.race.view import RaceDialog, RaceView
from evolab.policy.genome import layout_for
from evolab.spec.io import compile_env, env_hash

LANDER_TEXT = (paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8")


def entries(n: int = 2) -> list[RaceEntry]:
    spec = M.compile_cached(LANDER_TEXT)
    size = layout_for(spec).dim
    h = env_hash(compile_env(LANDER_TEXT).file)[:16]
    rng = np.random.default_rng(7)
    colors = ("#3987e5", "#d95926", "#199e70")
    return [RaceEntry(f"칸 {k}", "", colors[k], LANDER_TEXT, h, rng.normal(0, 0.5, size), 1) for k in range(n)]


@pytest.fixture()
def race(qtbot: Any) -> RaceDialog:
    dlg = RaceDialog()
    qtbot.addWidget(dlg)
    dlg.set_race(entries(), title=S.RACE_TITLE)
    dlg.show()
    dlg.activateWindow()
    QTest.qWaitForWindowExposed(dlg)
    qtbot.waitUntil(lambda: dlg.view.ready, timeout=30000)
    QApplication.processEvents()
    return dlg


def test_start_is_the_primary_default_button(race: RaceDialog) -> None:
    view = race.view
    assert view.start_button.text() == S.RACE_START or view.start_button.text() == S.RACE_PAUSE
    assert view.start_button.isDefault()
    assert view.start_button.property(style.ROLE_PROPERTY) == "primary"
    # 다른 단추의 autoDefault는 끄지 않는다(초점을 둔 단추를 Enter로 누를 수 있게: V5)
    assert view.next_button.autoDefault() and view.gif_button.autoDefault() and view.restart_button.autoDefault()
    assert [b for b in style.primaries(view) if b.isEnabled()] == [view.start_button]
    # 준비가 끝나면 초점이 「출발」에
    assert QApplication.focusWidget() is view.start_button
    assert "Space" in view.start_button.toolTip() and "(R)" in view.restart_button.toolTip()


def test_enter_and_space_start_and_pause_keep_the_course(race: RaceDialog) -> None:
    view = race.view
    view.pause()
    course = view.course
    QTest.keyClick(view.start_button, Qt.Key.Key_Return)
    assert view.playing and view.course == course == view.course_box.currentData()
    # 칸 밖(경주 칸)에서 Space → 일시 정지, 다시 Space → 출발
    lane = view.lanes[0].canvas
    lane.setFocus()
    QTest.keyClick(lane, Qt.Key.Key_Space)
    assert not view.playing
    QTest.keyClick(lane, Qt.Key.Key_Space)
    assert view.playing
    # 칸 밖에서 Enter → 기본 단추(출발/일시 정지), 코스 그대로
    QTest.keyClick(lane, Qt.Key.Key_Return)
    assert not view.playing and view.course == course
    # 고르기 칸에서 Enter도 기본 단추
    view.speed_box.setFocus()
    QTest.keyClick(view.speed_box, Qt.Key.Key_Return)
    assert view.playing and view.course == course
    view.pause()


def test_space_on_export_opens_export_not_start(race: RaceDialog, monkeypatch: pytest.MonkeyPatch) -> None:
    import evolab.app.export.dialog as D

    calls: list[str] = []

    def fake(*_a: Any, **kw: Any) -> None:
        calls.append(kw["kind"])
        return None

    monkeypatch.setattr(D, "choose_and_export", fake)
    view = race.view
    view.pause()
    view.gif_button.setFocus()
    QTest.keyClick(view.gif_button, Qt.Key.Key_Space)
    assert calls == ["race"] and not view.playing
    QTest.keyClick(view.gif_button, Qt.Key.Key_Return)
    assert calls == ["race", "race"] and not view.playing
    # 「다른 코스」에 초점 + Space → 다른 코스(그 단추)
    course = view.course
    view.next_button.setFocus()
    QTest.keyClick(view.next_button, Qt.Key.Key_Space)
    assert view.course == M.next_course(course)


def test_r_restarts_outside_fields_but_searches_in_a_combo(race: RaceDialog) -> None:
    view = race.view
    view.pause()
    view.seek(0.5)
    assert view.t > 0
    # 고르기 칸(코스)에서 R: 칸의 글자 찾기가 먹는다(처음부터 아님)
    view.course_box.setFocus()
    QTest.keyClick(view.course_box, Qt.Key.Key_R)
    assert view.t > 0 and not view.playing
    # 체크 상자·단추가 아닌 칸 밖(경주 칸)에서 R → 처음부터(출발함)
    lane = view.lanes[1].canvas
    lane.setFocus()
    QTest.keyClick(lane, Qt.Key.Key_R)
    assert view.playing and view.t < 0.5
    view.pause()
    # 「출발」에 초점이 있어도 R은 처음부터(단추는 R을 쓰지 않는다)
    view.seek(0.5)
    view.start_button.setFocus()
    QTest.keyClick(view.start_button, Qt.Key.Key_R)
    assert view.playing and view.t < 0.5
    view.pause()


def test_disabled_buttons_while_preparing_have_a_reason(qtbot: Any) -> None:
    view = RaceView()
    qtbot.addWidget(view)
    view.set_race(entries())
    assert not view.ready
    for b in (view.start_button, view.restart_button, view.gif_button):
        assert not b.isEnabled() and style.why_off(b) == S.RACE_WHY_PREPARING
    assert style.tip_text(view.start_button).endswith(S.RACE_WHY_PREPARING)
    qtbot.waitUntil(lambda: view.ready, timeout=30000)
    assert style.why_off(view.start_button) == ""
    view.shutdown()


def test_focus_stays_on_a_field_the_user_chose(race: RaceDialog) -> None:
    view = race.view
    view.speed_box.setFocus()
    view.focus_start()
    assert QApplication.focusWidget() is view.speed_box
    view.next_button.setFocus()
    view.focus_start()
    assert QApplication.focusWidget() is view.start_button

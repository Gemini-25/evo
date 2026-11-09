"""T16: 실험실 오른쪽 패널 순서 (v3 설계 D17, 4.3, 5.2).

「진행」·「보기」 묶음을 패널 맨 위로 옮겨, 흔한 다섯 창 크기에서 「▶ 학습 시작」·「멈춤」·「재생 배속」이 스크롤하지 않아도 보인다
(옮기기 전에는 다섯 크기 모두에서 스크롤 아래: 설계 9.1 측정). 위젯은 속성으로 찾는다(묶음 순서에 기대지 않음).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QPoint, QSettings
from PySide6.QtWidgets import QGroupBox, QScrollArea, QWidget

from evolab import paths
from evolab.app import strings as S

SIZES = [(1280, 720), (1280, 800), (1366, 728), (1366, 768), (1536, 824)]
"""설계 T16의 다섯 창 크기(노트북 화면에서 작업 표시줄을 뺀 크기)."""


@pytest.fixture()
def window(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from evolab.app.main import MainWindow

    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv("EVOLAB_NO_TOUR", "1")
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    w.tabs.setCurrentWidget(w.lab)
    return w


def _scroll_viewport(widget: QWidget) -> QWidget:
    w = widget.parentWidget()
    while w is not None and not isinstance(w, QScrollArea):
        w = w.parentWidget()
    assert w is not None
    return w.viewport()


def test_panel_group_order_puts_progress_and_view_first(window: Any) -> None:
    lab = window.lab
    panel = lab.btn_start.parentWidget().parentWidget()
    lay = panel.layout()
    groups = [lay.itemAt(i).widget() for i in range(lay.count()) if isinstance(lay.itemAt(i).widget(), QGroupBox)]
    assert [g.title() for g in groups] == [S.LAB_CONTROL_GROUP, S.LAB_VIEW_GROUP, S.LAB_ENV_GROUP, S.LAB_SETTINGS_GROUP,
                                           S.LAB_STOP_GROUP, S.LAB_STABILITY_GROUP]
    assert lab.btn_start.parentWidget() is groups[0] and lab.speed_combo.parentWidget() is groups[1]
    assert lab.env_combo.parentWidget() is groups[2] and lab.seed.parentWidget() is groups[3]


def test_tab_order_follows_the_panel_order(window: Any) -> None:
    """검토 수정: 묶음을 화면 차례로 옮긴 뒤에도 Tab 차례(초점 사슬)가 화면 차례와 같다(진행·보기가 맨 뒤로 가지 않음)."""
    lab = window.lab
    panel = lab.btn_start.parentWidget().parentWidget()
    lay = panel.layout()
    layout_titles = [lay.itemAt(i).widget().title() for i in range(lay.count())
                     if isinstance(lay.itemAt(i).widget(), QGroupBox)]
    seen: list[str] = []
    w = panel
    for _ in range(5000):  # 패널에서 사슬을 한 바퀴
        g = w
        while g is not None and not (isinstance(g, QGroupBox) and g.parentWidget() is panel):
            g = g.parentWidget()
        if g is not None and g.title() not in seen:  # 묶음마다 처음 나온 차례(묶음 상자 자신은 그 첫 단추보다 앞)
            seen.append(g.title())
        w = w.nextInFocusChain()
        if w is panel:
            break
    assert seen == layout_titles, seen
    chain: list[Any] = []
    w = lab.btn_start
    while True:
        chain.append(w)
        w = w.nextInFocusChain()
        if w is lab.btn_start:
            break
    pos = {id(x): i for i, x in enumerate(chain)}
    for widget in (lab.rb_spectate, lab.speed_combo, lab.fast_interval):
        assert pos[id(widget)] < pos[id(lab.env_combo)]


def test_start_pause_and_speed_are_visible_without_scrolling(window: Any, qtbot: Any) -> None:
    lab = window.lab
    slack: dict[tuple[int, int], int] = {}
    for size in SIZES:
        window.resize(*size)
        qtbot.waitUntil(lambda s=size: (window.width(), window.height()) == s, timeout=3000)
        qtbot.wait(50)  # 배치가 끝나게
        for name in ("btn_start", "btn_pause", "speed_combo"):
            w = getattr(lab, name)
            region = w.visibleRegion()
            assert not region.isEmpty(), (size, name)
            assert region.boundingRect() == w.rect(), (size, name)  # 잘리지 않고 다 보인다
        vp = _scroll_viewport(lab.speed_combo)
        slack[size] = vp.height() - lab.speed_combo.mapTo(vp, QPoint(0, lab.speed_combo.height())).y()
    assert min(slack.values()) >= 0, slack


def test_start_button_wording(window: Any) -> None:
    assert S.LAB_START == "▶ 학습 시작"
    assert window.lab.btn_start.text() == S.LAB_START != S.LAB_RESTART  # 「정체하면 새로 시작」 칸과 헷갈리지 않게


def test_continue_tooltip_is_restored_after_busy(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """다른 곳에서 학습 중이라 꺼진 「이어서」의 풍선은 바쁨이 풀리면 원래 글(도움말 풍선 HELP_TIPS)로 돌아간다."""
    from evolab.app.lab.view import LabView

    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    lab = LabView(settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(lab)
    lab.btn_continue.setToolTip(S.HELP_TIPS["lab.continue"])  # 풍선 도움말(B 묶음)이 단 글
    run = tmp_path / "run"
    lab.run_dir = run
    lab.run_busy = lambda p: Path(p) == run
    lab.busy_runs_changed()
    assert lab.btn_continue.toolTip() == S.RVL_CONTINUE_BUSY_TIP
    lab.busy_runs_changed()  # 바쁜 동안 다시 불러도 원래 글을 잃지 않는다
    lab.run_busy = lambda _p: False
    lab.busy_runs_changed()
    assert lab.btn_continue.toolTip() == S.HELP_TIPS["lab.continue"]
    lab._update_buttons()
    assert lab.btn_continue.toolTip() == S.HELP_TIPS["lab.continue"]  # 상태가 바뀔 때마다 지우지 않는다

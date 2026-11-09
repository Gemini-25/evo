"""T16: 실험실 오른쪽 패널 순서 (v3 설계 D17, 4.3, 5.2).

「진행」·「보기」 묶음을 패널 맨 위로 옮겨, 흔한 다섯 창 크기에서 「▶ 학습 시작」·「멈춤」·「재생 배속」이 스크롤하지 않아도 보인다
(옮기기 전에는 다섯 크기 모두에서 스크롤 아래: 설계 9.1 측정). 위젯은 속성으로 찾는다(묶음 순서에 기대지 않음).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtWidgets import QGroupBox, QScrollArea, QSplitter, QVBoxLayout, QWidget

from evolab import paths
from evolab.app import strings as S

SIZES = [(1280, 720), (1280, 800), (1366, 728), (1366, 768), (1536, 824)]
"""설계 T16의 다섯 창 크기(노트북 화면에서 작업 표시줄을 뺀 크기)."""

LONG_FOLDER = "진화실험실데이터" * 7
"""띄어쓰기 없는 한글 폴더 이름 56자(UTF-8 168바이트: 리눅스 이름 한도 255바이트 안). 두 겹 + 윈도우 CI 임시 폴더(약 100자)도
윈도우 경로 한도 260자 안. 폴더는 만들지 않는다(화면에 나오는 경로 글만 본다)."""


def _tab_minimums(window: Any) -> dict[str, tuple[int, int]]:
    """탭마다 최소 크기. 주 창의 최소 크기는 가장 큰 탭이 정하므로, 창이 설계 크기까지 줄지 않으면 어느 화면 탓인지 보인다."""
    tabs = window.tabs
    hints = {tabs.tabText(i): tabs.widget(i).minimumSizeHint() for i in range(tabs.count())}
    return {name: (s.width(), s.height()) for name, s in hints.items()}


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
    """다섯 크기 모두에서 세 위젯이 잘리지 않고 다 보인다(가로 스크롤 막대에 잘려도 실패).

    v3.0 윈도우 CI(e5b65f7)에서 첫 크기 1280×720의 크기 기다림이 3초를 넘겼다: 주 창의 최소 너비가 1280보다 컸다. 윈도우의
    offscreen 플랫폼은 글꼴 폴더가 없어(PySide6 휠에 lib/fonts가 없음) 모든 글자가 12px 너비 네모이고, 기록 탭의 「실행 n개 ·
    <실행 폴더>」 글(줄바꿈 없음)에 CI 임시 폴더 경로(약 120자)가 들어가 1,440px이 된다(리눅스에서 글꼴 없는 fontconfig로 재현).
    크기를 바꾸기 전에 최소 크기를 먼저 확인해, 창이 그 크기까지 줄지 않으면 시간 초과 대신 탭별 최소 크기를 보인다.
    """
    lab = window.lab
    slack: dict[tuple[int, int], int] = {}
    for size in SIZES:
        lo = window.minimumSize()
        assert lo.width() <= size[0] and lo.height() <= size[1], (size, (lo.width(), lo.height()), _tab_minimums(window))
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


def test_panel_area_starts_as_wide_as_its_content(qtbot: Any) -> None:
    """패널 스크롤 영역(_PanelScroll)의 처음 너비는 내용의 최소 너비 + 세로 막대 이상이다(가로 스크롤 막대 없음).

    Qt의 QScrollArea.sizeHint는 너비를 글자 높이의 36배에서 자르고 세로 막대 자리를 빼먹는다. 그래서 윈도우 CI의 네모 글자(높이
    12px, 패널 내용 672px)에서 패널이 432px로 잡혀 「멈춤」·「재생 배속」이 가로로 잘렸다. 글꼴과 상관없이 그 상황을 만든다:
    내용이 글자 높이의 40배 너비이고 영역보다 키가 크다.
    """
    from evolab.app.lab.view import _PanelScroll

    content = QWidget()
    inner = QWidget()
    em = content.fontMetrics().height()
    inner.setFixedSize(40 * em, 3000)
    QVBoxLayout(content).addWidget(inner)
    area = _PanelScroll()
    area.setWidgetResizable(True)
    area.setWidget(content)
    split = QSplitter(Qt.Orientation.Horizontal)
    split.addWidget(QWidget())
    split.addWidget(area)
    split.setStretchFactor(0, 1)
    split.setStretchFactor(1, 0)
    qtbot.addWidget(split)
    split.resize(40 * em + 600, 400)
    split.show()
    qtbot.waitExposed(split)
    qtbot.waitUntil(lambda: area.verticalScrollBar().isVisible(), timeout=2000)
    assert area.sizeHint().width() > 36 * area.fontMetrics().height()  # Qt의 상한을 넘는다
    assert area.viewport().width() >= content.minimumSizeHint().width()
    assert not area.horizontalScrollBar().isVisible()
    assert inner.visibleRegion().boundingRect().width() == inner.width()


def test_long_data_folder_does_not_widen_the_window(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """데이터 폴더 경로가 길어도 주 창을 설계의 가장 작은 크기(1280×720)까지 줄일 수 있다.

    v3.0 윈도우 CI에서 T16이 실패한 까닭을 리눅스 글꼴로 재현한다: 그곳에서는 경로가 약 120자였지만 글자가 모두 12px 네모라
    1,440px, 여기서는 띄어쓰기 없는 한글 폴더 두 겹(112자, 한 글자 12px)으로 1,300px을 넘긴다. 사용자도 데이터 폴더 경로가
    길면(설정 → 데이터 폴더) 창을 줄일 수 없고 화면보다 넓어진다. 윈도우에서는 줄을 바꾸는 비교 탭 글도 경로가 \\로 이어져
    한 덩어리라 같은 문제가 난다.
    """
    from evolab.app.main import MainWindow

    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / LONG_FOLDER / LONG_FOLDER / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv("EVOLAB_NO_TOUR", "1")
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    width, height = SIZES[0]
    lo = w.minimumSize()
    assert lo.width() <= width and lo.height() <= height, ((lo.width(), lo.height()), _tab_minimums(w))


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

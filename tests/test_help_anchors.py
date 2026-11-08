"""v3.0 0단계 연결표 (v3 설계 4.5, 7.3, 8.1 T1): ANCHORS의 모든 키가 진짜 창에서 위젯으로 풀리고 screen_of가 화면마다 맞다.

화면 파일이 속성 이름이나 단추 글을 바꾸면 이 테스트가 깨진다. 그때는 app/help/anchors.py의 표만 고친다.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QAbstractButton, QMessageBox, QWidget

from evolab import paths
from evolab.app import strings as S
from evolab.app.help import anchors as A
from evolab.app.help.anchors import ANCHORS, Anchor, editor_is_empty, resolve, screen_of, show_screen

DATA = Path(__file__).resolve().parent / "data"
LANDER = paths.builtin_env_dir() / "lander.evoenv"

EXTRA_KEYS = {"lab.canvas", "lab.dashboard", "main.help_corner", "lab.timeline", "lab.timeline_play", "lab.flip",
              "lab.race", "lab.reload", "lab.edit", "lab.apply"}
"""설계 7.3: HELP_TIPS 키 말고 연결표에 꼭 있어야 하는 키(그 밖에 lab.tile.<타일 id> 전부)."""

LATER_GROUPS = {"main.help_corner": "evolab.app.help.setup"}
"""다른 묶음이 만드는 위젯: 그 모듈이 생기기 전에는 풀리지 않아도 된다(I 묶음이 주 창 탭 줄 모서리에 단다)."""


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """실제 데이터·설정 폴더를 건드리지 않는다(설계 D5). 처음 실행 안내도 끈다(설계 D12)."""
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv("EVOLAB_NO_TOUR", "1")
    return tmp_path


@pytest.fixture()
def main_window(qtbot: Any, isolated: Path) -> Any:
    from evolab.app.main import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    return window


def _pending(key: str) -> bool:
    module = LATER_GROUPS.get(key)
    return module is not None and importlib.util.find_spec(module) is None


# ---------------------------------------------------------------------------
# 표
# ---------------------------------------------------------------------------
def test_anchor_table_shape() -> None:
    from evolab.app.lab.dashboard import ALL_TILE_IDS

    for key, anchor in ANCHORS.items():
        assert isinstance(anchor, Anchor)
        assert anchor.screen in A.SCREENS, key
        assert key.split(".")[0] == anchor.screen, key
        assert sum(x is not None for x in (anchor.path, anchor.text, anchor.tile)) == 1, key
        if anchor.text is not None:
            assert isinstance(getattr(S, anchor.text, None), str), key
        if anchor.tile is not None:
            assert anchor.tile in ALL_TILE_IDS, key
    assert set(S.HELP_TIPS) <= set(ANCHORS)
    assert EXTRA_KEYS <= set(ANCHORS)
    assert {f"lab.tile.{t}" for t in ALL_TILE_IDS} <= set(ANCHORS)
    assert A.keys_for("play") == ["play.restart", "play.new_seed", "play.save", "play.load"]


# ---------------------------------------------------------------------------
# 주 창 (탭 화면과 "main")
# ---------------------------------------------------------------------------
def test_every_main_window_anchor_resolves(main_window: Any) -> None:
    unresolved, wrong_screen = [], []
    for key, anchor in ANCHORS.items():
        if anchor.screen not in ("main", *A.TAB_SCREENS):
            continue
        widget = resolve(main_window, key)
        if widget is None:
            if not _pending(key):
                unresolved.append(key)
            continue
        if screen_of(widget) != anchor.screen:
            wrong_screen.append(f"{key}: {screen_of(widget)}")
    assert not unresolved, unresolved
    assert not wrong_screen, wrong_screen


def test_resolve_finds_the_intended_widgets(main_window: Any) -> None:
    w = main_window
    assert resolve(w, "lab.start") is w.lab.btn_start
    assert resolve(w, "lab.speed") is w.lab.speed_combo
    assert resolve(w, "lab.versus") is w.lab.btn_versus
    assert resolve(w, "lab.canvas") is w.lab.canvas
    assert resolve(w, "lab.env") is w.lab.env_combo
    assert resolve(w, "compare.repeat") is w.compare.repeat_button
    assert resolve(w, "records.memo") is w.records.memo_button
    for tile_id in S.DASH_TILE_TITLES:
        tile = resolve(w, f"lab.tile.{tile_id}")
        assert tile is not None and tile.tile_id == tile_id  # type: ignore[attr-defined]
    builtin = resolve(w, "editor.builtin")
    assert isinstance(builtin, QAbstractButton) and builtin.text() == S.ED_FROM_BUILTIN
    # 주 창 안의 아무 위젯에서 시작해도 같은 위젯(주 창으로 올라감)
    assert resolve(w.records, "lab.start") is w.lab.btn_start
    assert resolve(w, "no.such.key") is None


def test_help_corner_resolves_once_setup_exists(main_window: Any) -> None:
    if _pending("main.help_corner"):
        pytest.skip("주 창 탭 줄 모서리 단추는 I 묶음(app/help/setup.py)이 단다")
    corner = resolve(main_window, "main.help_corner")
    assert corner is not None and screen_of(corner) == "main"


def test_screen_of_main_window_parts(main_window: Any, qtbot: Any) -> None:
    w = main_window
    for tab in A.TAB_SCREENS:
        page = getattr(w, tab)
        assert screen_of(page) == tab
        child = page.findChildren(QWidget)[0]
        assert screen_of(child) == tab
    assert screen_of(w) == "main"
    assert screen_of(w.menuBar()) == "main"
    assert screen_of(w.tabs.tabBar()) == "main"
    # 탭 안에서 띄운 대화 상자는 그 탭 화면(F1이 그 화면 도움말을 엶)
    box = QMessageBox(w.lab)
    qtbot.addWidget(box)
    assert screen_of(box) == "lab"
    loose = QWidget()
    qtbot.addWidget(loose)
    assert screen_of(loose) is None
    assert screen_of(None) is None


def test_show_screen(main_window: Any) -> None:
    w = main_window
    for tab in A.TAB_SCREENS:
        assert show_screen(w, tab) is True
        assert w.tabs.currentWidget() is getattr(w, tab)
    assert show_screen(w, "main") is True
    for screen in ("play", "race", "repeat", "report", "export", "nope"):
        assert show_screen(w, screen) is False


def test_editor_is_empty(main_window: Any) -> None:
    doc = main_window.editor.doc
    assert editor_is_empty(main_window)  # 새 문서: 세계 설정과 시간 끝 조건뿐
    doc.from_builtin("hill_car.evoenv")
    assert not editor_is_empty(main_window)
    doc.new()
    assert editor_is_empty(main_window)


def test_window_anchors_need_an_open_window(main_window: Any) -> None:
    assert resolve(main_window, "play.restart") is None  # 직접 조종 창이 열려 있지 않음


# ---------------------------------------------------------------------------
# 따로 뜨는 창
# ---------------------------------------------------------------------------
def test_play_window_anchors(qtbot: Any, isolated: Path, main_window: Any) -> None:
    from evolab.app.play.window import PlayWindow
    from evolab.spec.io import load_env

    play = PlayWindow(load_env(LANDER))
    qtbot.addWidget(play)
    play.show()
    for key in A.keys_for("play"):
        widget = resolve(play, key)
        assert widget is not None, key
        assert screen_of(widget) == "play", key
        assert resolve(main_window, key) is widget, key  # 주 창에서 찾아도 열린 창의 위젯
    assert resolve(play, "play.save") is play.save_button
    assert resolve(play, "play.restart").text() == S.PLAY_RESTART  # type: ignore[union-attr]
    assert resolve(play, "lab.start") is main_window.lab.btn_start  # 창에서 주 창 위젯도 찾는다
    assert screen_of(play) == "play" and show_screen(main_window, "play") is False


def test_race_dialog_screen(qtbot: Any, isolated: Path) -> None:
    from evolab.app.race import model as M
    from evolab.app.race.view import RaceDialog, show_race, shutdown_race

    runs = []
    for name in ("v10_run_ga", "v11_run_restart"):
        dst = isolated / name
        shutil.copytree(DATA / name / "run", dst)
        runs.append(dst)
    entries, _warnings, _n = M.entries_from_runs([(runs[0], "#3987e5"), (runs[1], "#d95926")], "last")
    assert len(entries) == 2
    owner = QWidget()
    qtbot.addWidget(owner)
    owner.show()
    dlg = show_race(owner, entries, title=S.RACE_TITLE)
    try:
        assert isinstance(dlg, RaceDialog)
        assert screen_of(dlg) == "race"
        assert screen_of(dlg.view) == "race"
        assert screen_of(dlg.findChildren(QAbstractButton)[0]) == "race"
        assert screen_of(owner) is None
    finally:
        shutdown_race(dlg)
        dlg.close()


def test_repeat_dialog_screen(qtbot: Any, isolated: Path) -> None:
    from evolab.app.compare.repeat import RepeatDialog

    dlg = RepeatDialog(runs_dir=isolated / "runs", workers=1)
    qtbot.addWidget(dlg)
    assert screen_of(dlg) == "repeat"
    assert screen_of(dlg.buttons) == "repeat"
    assert screen_of(dlg.env_open) == "repeat"

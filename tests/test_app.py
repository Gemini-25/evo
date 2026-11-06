"""M0: 창, 탭, 명령줄."""

import re

import pytest
from pytestqt.qtbot import QtBot

from evolab import __version__, cli
from evolab.app import strings as S
from evolab.app.main import MainWindow

HANGUL = re.compile(r"[가-힣]")


def test_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_window_has_korean_title(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    assert window.windowTitle() == S.APP_TITLE
    assert HANGUL.search(window.windowTitle())


def test_four_tabs_in_order_and_switchable(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    names = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert names == [S.TAB_EDITOR, S.TAB_LAB, S.TAB_COMPARE, S.TAB_RECORDS]
    for i in range(window.tabs.count()):
        window.tabs.setCurrentIndex(i)
        assert window.tabs.currentIndex() == i


def test_cli_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "selftest" in capsys.readouterr().out

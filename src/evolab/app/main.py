"""주 창: 에디터, 실험실, 비교, 기록 네 화면을 탭으로 둔다."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QTabWidget, QWidget

from evolab import __version__
from evolab.app import strings as S


def _placeholder(text: str) -> QWidget:
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


class MainWindow(QMainWindow):
    """프로그램 창 하나. 화면마다 탭 하나."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(S.APP_TITLE)
        self.resize(1280, 800)

        self.tabs = QTabWidget()
        self.tabs.addTab(_placeholder(S.PLACEHOLDER_EDITOR), S.TAB_EDITOR)
        self.tabs.addTab(_placeholder(S.PLACEHOLDER_LAB), S.TAB_LAB)
        self.tabs.addTab(_placeholder(S.PLACEHOLDER_COMPARE), S.TAB_COMPARE)
        self.tabs.addTab(_placeholder(S.PLACEHOLDER_RECORDS), S.TAB_RECORDS)
        self.setCentralWidget(self.tabs)


def create_application(argv: list[str] | None = None) -> QApplication:
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    app = QApplication(argv if argv is not None else sys.argv[:1])
    app.setApplicationName("evolab")
    app.setApplicationDisplayName(S.APP_NAME)
    app.setApplicationVersion(__version__)
    return app


def run_gui() -> int:
    app = create_application()
    window = MainWindow()
    window.show()
    return app.exec()

"""주 창: 에디터, 실험실, 비교, 기록 네 화면을 탭으로 둔다."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox, QTabWidget, QWidget

from evolab import __version__, paths
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
        self._play_windows: list[QWidget] = []

        menu = self.menuBar().addMenu(S.MENU_PLAY)
        act = menu.addAction(S.MENU_PLAY_LANDER)
        act.triggered.connect(lambda: self.open_play(paths.builtin_env_dir() / "lander.evoenv"))
        act_open = menu.addAction(S.MENU_PLAY_OPEN)
        act_open.triggered.connect(self._choose_and_play)

    def _choose_and_play(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, S.MENU_PLAY_OPEN, str(paths.builtin_env_dir()), S.ENV_FILE_FILTER)
        if path:
            self.open_play(Path(path))

    def open_play(self, path: Path) -> QWidget | None:
        """환경 파일을 읽어 직접 조종 창을 연다. 파일에 오류가 있으면 알려 주고 None."""
        from evolab.app.play.window import PlayWindow
        from evolab.spec.compiled import EnvFileError
        from evolab.spec.io import load_env

        try:
            spec = load_env(path)
        except EnvFileError as exc:
            QMessageBox.warning(self, S.PLAY_OPEN_FAILED, str(exc))
            return None
        window = PlayWindow(spec)
        window.show()
        self._play_windows = [w for w in self._play_windows if w.isVisible()] + [window]
        return window


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

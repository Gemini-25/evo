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

        from evolab.app.lab.view import LabView
        from evolab.app.records.view import RecordsView

        self.tabs = QTabWidget()
        self.lab = LabView()
        self.records = RecordsView()
        self.tabs.addTab(_placeholder(S.PLACEHOLDER_EDITOR), S.TAB_EDITOR)
        self.tabs.addTab(self.lab, S.TAB_LAB)
        self.tabs.addTab(_placeholder(S.PLACEHOLDER_COMPARE), S.TAB_COMPARE)
        self.tabs.addTab(self.records, S.TAB_RECORDS)
        self.tabs.setCurrentWidget(self.lab)
        self.setCentralWidget(self.tabs)
        self.records.open_requested.connect(self._open_run)
        self.records.resume_requested.connect(self._resume_run)
        self.records.clone_requested.connect(self._clone_run)
        # 학습 중인 실행은 기록 화면에서 지우거나 최신 체크포인트에서 이어 할 수 없다.
        self.lab.run_started.connect(lambda p: (self.records.set_busy_runs([p]), self.records.refresh()))
        self.lab.client.closed.connect(lambda: (self.records.set_busy_runs([]), self.records.refresh()))
        builtin = paths.builtin_env_dir() / "lander.evoenv"
        if builtin.exists():
            self.lab.load_env(builtin)
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
        from evolab.world.physics2d.episode import EpisodeBuildError

        try:
            spec = load_env(path)
            window = PlayWindow(spec)
        except (EnvFileError, EpisodeBuildError) as exc:
            QMessageBox.warning(self, S.PLAY_OPEN_FAILED, str(exc))
            return None
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._play_windows.append(window)
        window.destroyed.connect(lambda _=None, w=window: self._forget(w))
        window.show()
        return window

    def _forget(self, window: QWidget) -> None:
        self._play_windows = [w for w in self._play_windows if w is not window]

    def _busy(self) -> bool:
        if self.lab.client.active:
            QMessageBox.information(self, S.APP_NAME, S.MAIN_LAB_BUSY)
            return True
        return False

    def _open_run(self, run: Path) -> None:
        if not self._busy() and self.lab.open_run(run):
            self.tabs.setCurrentWidget(self.lab)

    def _resume_run(self, run: Path, generation: int) -> None:
        if not self._busy() and self.lab.resume_run(run, generation):
            self.tabs.setCurrentWidget(self.lab)

    def _clone_run(self, run: Path) -> None:
        from evolab.runs.catalog import clone_settings

        if self._busy():
            return
        try:
            info = clone_settings(run)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, S.APP_NAME, str(exc))
            return
        if self.lab.use_env_text(info["env_text"], info.get("env_name", ""), info.get("config") or {}):
            self.tabs.setCurrentWidget(self.lab)

    def closeEvent(self, event: object) -> None:  # noqa: N802
        self.lab.shutdown()
        super().closeEvent(event)  # type: ignore[arg-type]


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

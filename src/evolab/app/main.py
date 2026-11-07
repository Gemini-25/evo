"""주 창: 에디터, 실험실, 비교, 기록 네 화면을 탭으로 둔다."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QTabWidget, QWidget

from evolab import __version__, paths
from evolab.app import strings as S


class MainWindow(QMainWindow):
    """프로그램 창 하나. 화면마다 탭 하나."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(S.APP_TITLE)
        self.resize(1280, 800)
        if paths.icon_path().exists():
            from PySide6.QtGui import QIcon

            self.setWindowIcon(QIcon(str(paths.icon_path())))

        from evolab.app.compare.view import CompareView
        from evolab.app.editor.view import EditorView
        from evolab.app.lab.view import LabView
        from evolab.app.records.view import RecordsView

        self.tabs = QTabWidget()
        self.editor = EditorView()
        self.lab = LabView()
        self.compare = CompareView()
        self.records = RecordsView()
        self.tabs.addTab(self.editor, S.TAB_EDITOR)
        self.tabs.addTab(self.lab, S.TAB_LAB)
        self.tabs.addTab(self.compare, S.TAB_COMPARE)
        self.tabs.addTab(self.records, S.TAB_RECORDS)
        self.editor.send_to_lab.connect(self._editor_to_lab)
        self.editor.play_requested.connect(self.open_play_spec)
        self.editor.title_changed.connect(
            lambda t: self.tabs.setTabText(self.tabs.indexOf(self.editor), f"{S.TAB_EDITOR} · {t}"))
        self.lab.edit_env_requested.connect(self._lab_to_editor)
        self.records.compare_requested.connect(self._compare_runs)
        self.tabs.setCurrentWidget(self.lab)
        self.setCentralWidget(self.tabs)
        self.records.open_requested.connect(self._open_run)
        self.records.resume_requested.connect(self._resume_run)
        self.records.clone_requested.connect(self._clone_run)
        # 학습 중인 실행(실험실, 반복 실행)은 기록 화면에서 지우거나 최신 체크포인트에서 이어 할 수 없다.
        self._busy_lab: Path | None = None
        self._busy_repeat: Path | None = None
        self.lab.run_started.connect(lambda p: self._set_busy(lab=p))
        self.lab.client.closed.connect(lambda: self._set_busy(lab=None))
        self.compare.repeat_run_started.connect(lambda p: self._set_busy(repeat=p))
        self.compare.repeat_finished.connect(lambda _ok: self._set_busy(repeat=None))
        builtin = paths.builtin_env_dir() / "lander.evoenv"
        if builtin.exists():
            self.lab.load_env(builtin)
        self._play_windows: list[QWidget] = []

        menu = self.menuBar().addMenu(S.MENU_PLAY)
        act = menu.addAction(S.MENU_PLAY_LANDER)
        act.triggered.connect(lambda: self.open_play(paths.builtin_env_dir() / "lander.evoenv"))
        act_open = menu.addAction(S.MENU_PLAY_OPEN)
        act_open.triggered.connect(self._choose_and_play)
        help_menu = self.menuBar().addMenu(S.MENU_HELP)
        help_menu.addAction(S.MENU_ABOUT).triggered.connect(self.show_about)
        help_menu.addAction(S.MENU_OPEN_DATA).triggered.connect(self._open_data_dir)

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

    def about_text(self) -> str:
        notices = paths.bundle_dir("licenses") / "THIRD_PARTY_NOTICES.txt"
        try:
            body = notices.read_text(encoding="utf-8")
        except OSError:
            body = S.ABOUT_NO_NOTICES
        return S.ABOUT_TEXT.format(name=S.APP_NAME, version=__version__, notices=body)

    def show_about(self) -> None:
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPlainTextEdit, QVBoxLayout

        dlg = QDialog(self)
        dlg.setWindowTitle(S.MENU_ABOUT)
        text = QPlainTextEdit(self.about_text())
        text.setReadOnly(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.reject)
        lic = paths.bundle_dir("licenses")
        if lic.is_dir():
            btn = buttons.addButton(S.ABOUT_OPEN_LICENSES, QDialogButtonBox.ButtonRole.ActionRole)
            btn.clicked.connect(lambda: _open_folder(lic))
        lay = QVBoxLayout(dlg)
        lay.addWidget(text)
        lay.addWidget(buttons)
        dlg.resize(640, 480)
        dlg.exec()

    def _open_data_dir(self) -> None:
        folder = paths.data_dir()
        folder.mkdir(parents=True, exist_ok=True)
        _open_folder(folder)

    def _forget(self, window: QWidget) -> None:
        self._play_windows = [w for w in self._play_windows if w is not window]

    def _set_busy(self, **changes: Path | None) -> None:
        if "lab" in changes:
            self._busy_lab = changes["lab"]
        if "repeat" in changes:
            self._busy_repeat = changes["repeat"]
        self.records.set_busy_runs([p for p in (self._busy_lab, self._busy_repeat) if p is not None])
        self.records.refresh()

    def _editor_to_lab(self, text: str) -> None:
        if self.lab.can_take_env():
            if self.lab.send_env_text(text):
                self.tabs.setCurrentWidget(self.lab)
                self.statusBar().showMessage(S.MAIN_ENV_SENT, 8000)
            return
        if self.lab.client.active:
            QMessageBox.information(self, S.APP_NAME, S.MAIN_LAB_BUSY)
            return
        if self.lab.send_env_text(text):
            self.tabs.setCurrentWidget(self.lab)

    def _lab_to_editor(self, text: str, _name: str) -> None:
        if self.editor.load_text(text, from_lab=self.lab.can_take_env()):
            self.tabs.setCurrentWidget(self.editor)

    def _compare_runs(self, runs: list) -> None:
        self.compare.set_selected_runs(runs)
        self.tabs.setCurrentWidget(self.compare)

    def open_play_spec(self, spec: object) -> QWidget | None:
        """컴파일한 환경으로 직접 조종 창을 연다(에디터에서)."""
        from evolab.app.play.window import PlayWindow
        from evolab.world.physics2d.episode import EpisodeBuildError

        try:
            window = PlayWindow(spec)  # type: ignore[arg-type]
        except EpisodeBuildError as exc:
            QMessageBox.warning(self, S.PLAY_OPEN_FAILED, str(exc))
            return None
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._play_windows.append(window)
        window.destroyed.connect(lambda _=None, w=window: self._forget(w))
        window.show()
        return window

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
        if not self.editor.maybe_save():
            event.ignore()  # type: ignore[attr-defined]
            return
        self.lab.shutdown()
        self.compare.shutdown()
        super().closeEvent(event)  # type: ignore[arg-type]


def _open_folder(folder: Path) -> None:
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


def create_application(argv: list[str] | None = None) -> QApplication:
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    app = QApplication(argv if argv is not None else sys.argv[:1])
    app.setApplicationName("evolab")
    app.setApplicationDisplayName(S.APP_NAME)
    app.setApplicationVersion(__version__)
    if paths.icon_path().exists():
        from PySide6.QtGui import QIcon

        app.setWindowIcon(QIcon(str(paths.icon_path())))
    return app


def run_gui() -> int:
    app = create_application()
    try:
        paths.install_examples()  # 첫 실행: 내장 예제를 문서\진화실험실\environments로(있으면 그대로)
    except OSError:
        pass
    window = MainWindow()
    window.show()
    return app.exec()

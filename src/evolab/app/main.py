"""주 창: 에디터, 실험실, 비교, 기록 네 화면을 탭으로 둔다.

v3.0(설계 7.1): 주 창이 만들어질 때 도움말 메뉴·모서리 단추·F1·풍선 도움말·빈 상태를 `app/help/setup.install`로 붙이고,
`create_application`이 Qt 기본 문구를 한국어로(qtbase_ko), `run_gui`가 진단(오류 기록, 갑자기 닫힘 표시, 뮤텍스), 창을 못 띄운 시작
실패의 오류 보고 묶음, 처음 실행 안내 제안을 맡는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QLibraryInfo, QLocale, Qt, QTimer, QTranslator
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QTabWidget, QWidget

from evolab import __version__, paths
from evolab.app import strings as S

STATE_TICK_MS = 1000
"""학습 중(시작하는 중·학습 중)에만 도는 주 창 타이머: 상태 막대 꼬리표와 창 제목의 세대를 이 간격으로만 고친다(6.2.0 U32, 세대마다 하지 않음)."""

TICKING_STATES = ("starting", "running")
"""타이머가 도는 실험실 상태."""


class MainWindow(QMainWindow):
    """프로그램 창 하나. 화면마다 탭 하나."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(S.APP_TITLE)
        self.resize(1280, 800)
        # 6.2.0 U32: 상태 막대를 처음부터 만든다(첫 알림 때 생기며 가운데가 줄지 않게). 오른쪽 꼬리표는 실험실 상태를
        # 다른 탭에서도 보이게 한다. 최소 폭 1: 주 창 최소 폭을 넓히지 않는다.
        from evolab.app import style

        self.state_chip = style.StateChip()
        self.state_chip.setObjectName("state_chip")
        self.state_chip.setMinimumWidth(1)
        self.statusBar().addPermanentWidget(self.state_chip)
        self._state_timer = QTimer(self)
        self._state_timer.setInterval(STATE_TICK_MS)
        self._state_timer.timeout.connect(self.refresh_state)
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
        # 6.2.0: 「새 프리셋」 창(설계 D42). 창은 에디터가 띄우고(저장 물음·끝난 뒤 열기), 「실험실로 보내기」 전에 실험실이 받을 수
        # 있는지 본다(학습 중이면 창이 남고 파일을 쓰지 않음). 실험실의 「새 프리셋…」 단추도 같은 창을 연다.
        self.editor.lab_ready = self.lab_can_take_preset
        new_preset = getattr(self.lab, "new_preset_requested", None)
        if new_preset is not None:
            new_preset.connect(self.open_preset_wizard)
        self.records.compare_requested.connect(self._compare_runs)
        self.tabs.setCurrentWidget(self.lab)
        self.setCentralWidget(self.tabs)
        self.records.open_requested.connect(self._open_run)
        self.records.resume_requested.connect(self._resume_run)
        self.records.clone_requested.connect(self._clone_run)
        # 학습 중인 실행(실험실, 반복 실행)은 기록 화면에서 지우거나 최신 체크포인트에서 이어 할 수 없다.
        self._busy_lab: Path | None = None
        self._busy_repeat: Path | None = None
        # 실험실의 '이어서'와 기록의 이어하기도 같은 규칙: 다른 곳에서 학습 중인 폴더는 이어 가지 않는다.
        self.lab.run_busy = self.run_is_busy
        self.records.resume_blocked = lambda: S.MAIN_LAB_BUSY if self.lab.client.active else None
        self.lab.run_started.connect(lambda p: self._set_busy(lab=p))
        self.lab.client.closed.connect(lambda: self._set_busy(lab=None))
        self.compare.repeat_run_started.connect(lambda p: self._set_busy(repeat=p))
        self.compare.repeat_finished.connect(lambda _ok: self._set_busy(repeat=None))
        builtin = paths.builtin_env_dir() / "lander.evoenv"
        if builtin.exists():
            self.lab.load_env(builtin)
        self._play_windows: list[QWidget] = []
        # 꼬리표·제목은 상태가 바뀔 때(state_changed: Hb의 계약 3.3)와 학습 중 1초 타이머로만 고친다. run_started·closed는
        # 그 신호가 없는 실험실에서도 시작·끝을 놓치지 않게 함께 받는다(같은 글이면 아무것도 하지 않는다).
        state_changed = getattr(self.lab, "state_changed", None)
        if state_changed is not None:
            state_changed.connect(lambda _state: self.refresh_state())
        self.lab.run_started.connect(lambda _p: self.refresh_state())
        self.lab.client.closed.connect(self.refresh_state)
        self.refresh_state()

        menu = self.menuBar().addMenu(S.MENU_PLAY)
        act = menu.addAction(S.MENU_PLAY_LANDER)
        act.triggered.connect(lambda: self.open_play(paths.builtin_env_dir() / "lander.evoenv"))
        act_open = menu.addAction(S.MENU_PLAY_OPEN)
        act_open.triggered.connect(self._choose_and_play)
        settings_menu = self.menuBar().addMenu(S.MENU_SETTINGS)
        settings_menu.addAction(S.MENU_DATA_DIR).triggered.connect(self.choose_data_dir)
        settings_menu.addAction(S.MENU_DATA_DIR_RESET).triggered.connect(lambda: self.set_data_dir(None))
        help_menu = self.menuBar().addMenu(S.MENU_HELP)
        help_menu.addAction(S.MENU_OPEN_DATA).triggered.connect(self._open_data_dir)
        help_menu.addAction(S.MENU_ABOUT).triggered.connect(self.show_about)

        # v3.0: 도움말 메뉴 위쪽 항목, 탭 줄 모서리 「? 도움말」, F1·Shift+F1, 풍선 도움말, 빈 상태(설계 7.1, 5.1).
        # 처음 실행 안내는 여기서 띄우지 않는다(run_gui만: 설계 D12).
        from evolab.app.help import setup as help_setup

        help_setup.install(self)

    # -- 6.2.0 U32: 상태 막대 꼬리표와 창 제목 ---------------------------------------
    def state_texts(self) -> tuple[str, str, str]:
        """(상태 종류, 꼬리표 글, 창 제목). 실행이 없으면 꼬리표는 상태 이름만, 제목은 APP_TITLE."""
        lab = self.lab
        kind = str(getattr(lab, "state", "idle") or "idle")
        name = S.LAB_STATE_NAMES.get(kind, kind)
        gen = int(getattr(lab, "generation", 0) or 0)
        if getattr(lab, "run_dir", None) is None:
            return kind, name, S.APP_TITLE
        spec = getattr(lab, "spec", None)
        env = spec.file.name if spec is not None else ""
        chip = S.MAIN_CHIP.format(state=name, gen=gen)
        title = S.MAIN_TITLE_RUN.format(env=env, state=name, gen=gen, app=S.APP_TITLE) if env else S.APP_TITLE
        return kind, chip, title

    def refresh_state(self) -> None:
        """꼬리표와 제목을 실험실 상태에 맞춘다(글이 같으면 다시 그리지 않는다). 학습 중에만 1초 타이머가 돈다."""
        try:
            kind, chip, title = self.state_texts()
        except RuntimeError:  # 닫히는 중
            return
        self.state_chip.set_state(kind, chip)
        if self.windowTitle() != title:
            self.setWindowTitle(title)
        ticking = kind in TICKING_STATES
        if ticking and not self._state_timer.isActive():
            self._state_timer.start()
        elif not ticking and self._state_timer.isActive():
            self._state_timer.stop()

    def lab_can_take_preset(self) -> bool:
        """「실험실로 보내기」를 받을 수 있는가: 학습 프로세스가 없거나, 멈춘 학습이 환경을 바꿔 이을 수 있을 때(_editor_to_lab과 같은 규칙)."""
        return not self.lab.client.active or self.lab.can_take_env()

    def open_preset_wizard(self) -> QWidget:
        """「새 프리셋」 창을 연다(실험실 단추·에디터 빈 캔버스 카드). 「에디터에서 다듬기」로 끝나면 에디터 탭으로 간다."""
        wizard = self.editor.new_preset()
        wizard.finished_with.connect(self._preset_finished)
        return wizard  # type: ignore[no-any-return]

    def _preset_finished(self, _text: str, then: str, _path: object) -> None:
        if then == "edit":
            self.tabs.setCurrentWidget(self.editor)

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
            from evolab.app.errors import show_error

            show_error(self, S.PLAY_OPEN_FAILED, exc)
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
        return S.ABOUT_TEXT_V3.format(name=S.APP_NAME, version=__version__, data_dir=paths.data_dir(),
                                      state_dir=paths.user_state_dir(), notices=body)

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

    def choose_data_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, S.MENU_DATA_DIR, str(paths.data_dir()))
        if folder:
            self.set_data_dir(Path(folder))

    def set_data_dir(self, folder: Path | None) -> None:
        """데이터 폴더(환경 파일, 실행 기록)를 바꾼다. 새 실행부터 그 폴더에 저장한다(예전 실행은 옮기지 않는다)."""
        paths.set_custom_data_dir(folder)
        try:
            paths.install_examples()
        except OSError:
            pass
        self.records.refresh()
        self.compare.refresh()
        self.statusBar().showMessage(S.MAIN_DATA_DIR_SET.format(path=paths.data_dir()), 8000)

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
        self.records.set_busy_runs(self.busy_runs())
        self.records.refresh()
        self.lab.busy_runs_changed()

    def busy_runs(self) -> list[Path]:
        """지금 학습 중인 실행 폴더(실험실, 비교 화면의 반복 실행)."""
        return [p for p in (self._busy_lab, self._busy_repeat) if p is not None]

    def run_is_busy(self, run: Path) -> bool:
        """그 실행 폴더를 지금 어느 세션이든 학습하고 있는가."""
        return any(_same_folder(Path(run), p) for p in self.busy_runs())

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
            from evolab.app.errors import show_error

            show_error(self, S.PLAY_OPEN_FAILED, exc)
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
            from evolab.app.errors import os_cause, show_error

            show_error(self, S.APP_NAME, exc,
                       next_step=None if os_cause(exc) else S.ERR_RUN_STORE_NEXT.format(records=S.TAB_RECORDS))
            return
        if self.lab.use_env_text(info["env_text"], info.get("env_name", ""), info.get("config") or {}):
            self.tabs.setCurrentWidget(self.lab)

    def closeEvent(self, event: object) -> None:  # noqa: N802
        if not self.editor.maybe_save():
            event.ignore()  # type: ignore[attr-defined]
            return
        self._state_timer.stop()
        self.lab.shutdown()
        self.compare.shutdown()
        super().closeEvent(event)  # type: ignore[arg-type]
        from evolab import diagnostics

        diagnostics.mark_clean_exit()  # 정상 종료: running-<pid>.flag를 지운다(설계 D30)


def _same_folder(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a == b


def _open_folder(folder: Path) -> None:
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


QT_TRANSLATOR_NAME = "evolab_qtbase_ko"
"""Qt 기본 문구 번역기의 객체 이름(앱에 한 번만 넣으려고)."""


def qt_translation_dirs() -> list[Path]:
    """qtbase_ko.qm을 찾을 폴더: Qt가 아는 번역 폴더, 그다음 PySide6 꾸러미 안(묶인 exe에서 Qt 경로가 다를 때)."""
    dirs = [Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath))]
    try:
        import PySide6

        base = Path(PySide6.__file__).resolve().parent
        dirs += [base / "Qt" / "translations", base / "translations"]
    except (ImportError, OSError, TypeError):
        pass
    out: list[Path] = []
    for d in dirs:
        if str(d) and d not in out:
            out.append(d)
    return out


def install_qt_translation(app: QApplication) -> bool:
    """Qt 기본 문구(표준 단추 저장·닫기·취소·예·아니요, 파일 대화 상자, 「자세한 정보 표시...」)를 한국어로(설계 D25, 4.7).

    앱에 한 번만 넣는다. 번역 파일이 없으면 오류 기록에 한 줄 남기고 영어 그대로 계속한다. 넣었으면(또는 이미 있으면) 참.
    """
    if app.findChild(QTranslator, QT_TRANSLATOR_NAME) is not None:
        return True
    tr = QTranslator(app)
    tr.setObjectName(QT_TRANSLATOR_NAME)
    locale = QLocale(QLocale.Language.Korean, QLocale.Country.SouthKorea)
    for folder in qt_translation_dirs():
        if tr.load(locale, "qtbase", "_", str(folder)):
            app.installTranslator(tr)
            return True
    tr.deleteLater()
    from evolab import diagnostics

    diagnostics.record("qt", "qtbase_ko not loaded", "; ".join(str(d) for d in qt_translation_dirs()))
    return False


def create_application(argv: list[str] | None = None) -> QApplication:
    app = QApplication.instance()
    if isinstance(app, QApplication):
        install_qt_translation(app)  # 테스트(pytest-qt)처럼 앱을 먼저 만든 경우에도 한국어 기본 문구
        return app
    app = QApplication(argv if argv is not None else sys.argv[:1])
    app.setApplicationName("evolab")
    app.setApplicationDisplayName(S.APP_NAME)
    app.setApplicationVersion(__version__)
    if paths.icon_path().exists():
        from PySide6.QtGui import QIcon

        app.setWindowIcon(QIcon(str(paths.icon_path())))
    install_qt_translation(app)
    return app


def on_crash(action: str, line: str) -> None:
    """잡히지 않은 예외 알림(`diagnostics.install_gui`에 넘김): 오류 보고 모듈(report.dialog·bundle)은 처음 알릴 때 불러온다
    (설계 D62: 시작할 때 불러오지 않음, M3 창 뜨는 시간)."""
    from evolab.app.report import dialog as report_dialog

    report_dialog.on_crash(action, line)


def run_gui() -> int:
    """화면 프로그램(`evolab`, `evolab.exe`). 설계 4.9 순서: 진단 훅 → 앱(한국어 Qt) → Qt 메시지·실행 중 표시 → 뮤텍스 → 창.

    창을 만들거나 띄우다 실패하면 바탕 화면에 오류 보고 묶음을 만들고 Qt 없는 상자로 알린 뒤 종료 코드 1(설계 D32).
    창이 뜨면 이벤트 루프에서 지난번 갑자기 닫힘 질문(있으면)과 처음 실행 안내 제안(setup.after_show)을 한다.
    """
    from evolab import diagnostics, winapp

    diagnostics.install("gui")
    app = create_application()
    previous = diagnostics.install_gui(on_crash)
    winapp.create_app_mutex()  # 설치·제거 프로그램이 실행 중인지 안다(v3.1, 화면 프로세스만)
    try:
        paths.install_examples()  # 첫 실행: 내장 예제를 문서\진화실험실\environments로(있으면 그대로)
    except OSError:
        pass
    try:
        window = MainWindow()
        window.show()
    except Exception as exc:
        from evolab.app.report.bundle import report_startup_failure

        return report_startup_failure(exc)
    from evolab.app.help import setup as help_setup

    help_setup.schedule_after_show(window, previous)
    code = app.exec()
    diagnostics.mark_clean_exit()
    return code

"""오류 보고 묶음 대화 상자와 잡히지 않은 예외 상자(v3 설계 D27·D30·D32·D33, 5.4·5.5).

- `open_report_dialog(parent, opts)`: 메모 칸, 넣을 것 체크 칸(없는 것은 회색), 넣을 파일 목록과 크기, 개인 정보 안내,
  「저장…」 → 파일 대화 상자(기본 바탕 화면) → 배경 스레드에서 쓰기 → 「만들었습니다」 + 「폴더 열기」. 만든 zip 경로를
  돌려준다(만들지 않고 닫으면 None). 미리 보기와 쓰기는 `bundle.collect`의 같은 목록을 쓴다.
- `show_crash_box(parent, line, opts)`: 비모달 「문제가 생겼습니다」 상자(+ 「오류 보고 묶음 만들기…」).
- `on_crash(action, line)`: `diagnostics.install_gui`에 넘기는 알림 함수. 주 창을 찾아 상자("box") 또는 상태 줄("again",
  "more")로 알린다.
- `ask_previous(parent)`: 지난 실행이 갑자기 닫혔을 때 한 번 묻고, 「묶음 만들기…」면 대화 상자를 연다.
- `open_for(parent, reason)`: 도움말 메뉴·오류 상자의 단추. 주 창(`main_window()`)의 실험실 실행·환경을 넣은
  설정(`options_for`)으로 대화 상자를 연다.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from evolab.app import strings as S
from evolab.app.report import bundle
from evolab.app.report.bundle import BundleOptions, ReportItem

STATUS_MS = 10000
"""상태 줄 알림을 보이는 시간."""

_open_boxes: set[QMessageBox] = set()
"""부모 없이 띄운 비모달 상자가 사라지지 않게 쥐어 둔다(닫히면 뺀다)."""


class ReportDialog(QDialog):
    """오류 보고 묶음 대화 상자(설계 5.4)."""

    def __init__(self, parent: QWidget | None, opts: BundleOptions) -> None:
        super().__init__(parent)
        self.setWindowTitle(S.REP_TITLE)
        self.opts = opts
        self.saved_path: Path | None = None
        self.items: list[ReportItem] = []
        self._all: list[ReportItem] = []

        lay = QVBoxLayout(self)
        intro = QLabel(S.REP_INTRO)
        intro.setWordWrap(True)
        lay.addWidget(intro)
        lay.addWidget(QLabel(S.REP_NOTE_LABEL))
        self.note = QPlainTextEdit()
        self.note.setPlaceholderText(S.REP_NOTE_PLACEHOLDER)
        self.note.setPlainText(opts.note)
        self.note.setFixedHeight(80)
        lay.addWidget(self.note)

        lay.addWidget(QLabel(S.REP_ITEMS_LABEL))
        grid = QGridLayout()
        self.checks: dict[str, QCheckBox] = {}
        for i, key in enumerate(bundle.GROUP_KEYS):
            box = QCheckBox(S.REP_ITEM_NAMES[key])
            box.toggled.connect(self._refresh)
            self.checks[key] = box
            grid.addWidget(box, i // 2, i % 2)
        lay.addLayout(grid)
        self.files_label = QLabel()
        lay.addWidget(self.files_label)
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(120)
        lay.addWidget(self.file_list)
        privacy = QLabel(S.REP_PRIVACY.format(mask=S.REP_USER_MASK))
        privacy.setWordWrap(True)
        lay.addWidget(privacy)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.hide()
        lay.addWidget(self.status)

        row = QHBoxLayout()
        self.btn_open_folder = QPushButton(S.REP_OPEN_FOLDER)
        self.btn_open_folder.clicked.connect(self.open_folder)
        self.btn_open_folder.hide()
        row.addWidget(self.btn_open_folder)
        row.addStretch(1)
        self.btn_save = QPushButton(S.REP_SAVE)
        self.btn_save.setDefault(True)
        self.btn_save.clicked.connect(self.save)
        self.btn_close = QPushButton(S.REP_CANCEL)
        self.btn_close.clicked.connect(self.reject)
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_close)
        lay.addLayout(row)

        self._reload()
        self.resize(560, 560)

    # ------------------------------------------------------------------
    def _reload(self) -> None:
        """넣을 수 있는 것을 다시 모은다(실행·환경은 넣는 쪽으로 모아 두고 체크 칸으로 거른다)."""
        self._all = bundle.collect(replace(self.opts, note=self.note.toPlainText(), include_run=True,
                                           include_env=True))
        present = {i.key for i in self._all}
        defaults = {"run": self.opts.include_run, "env": self.opts.include_env}
        for key, box in self.checks.items():
            box.blockSignals(True)
            box.setEnabled(key in present)
            box.setChecked(key in present and defaults.get(key, True))
            box.blockSignals(False)
        self._refresh()

    def _refresh(self) -> None:
        self.items = [i for i in self._all if self.checks[i.key].isChecked()]
        total = sum(i.size for i in self.items)
        self.files_label.setText(S.REP_FILES.format(count=len(self.items), size=bundle.human_size(total)))
        self.file_list.clear()
        for item in self.items:
            self.file_list.addItem(f"{item.arcname}  ({bundle.human_size(item.size)})")
        self.btn_save.setEnabled(bool(self.items))

    def current_options(self) -> BundleOptions:
        return replace(self.opts, note=self.note.toPlainText(), include_run=self.checks["run"].isChecked(),
                       include_env=self.checks["env"].isChecked())

    # ------------------------------------------------------------------
    def save(self) -> Path | None:
        """파일 대화 상자로 고른 곳에 묶음을 쓴다. 쓰는 동안은 배경 스레드, 화면은 이벤트를 계속 처리한다."""
        start = str(bundle.default_bundle_path())
        path, _ = QFileDialog.getSaveFileName(self, S.REP_SAVE_TITLE, start, S.REP_FILTER)
        if not path:
            return None
        target = Path(path)
        if target.suffix.lower() != ".zip":
            target = target.with_name(target.name + ".zip")
        return self.save_to(target)

    def save_to(self, target: Path) -> Path | None:
        opts, items = self.current_options(), list(self.items)
        result: dict[str, Any] = {}

        def work() -> None:
            try:
                result["written"] = bundle.write_report(target, opts, items)
            except Exception as exc:  # 화면에는 까닭만, 원문은 기록에
                result["error"] = exc

        self.btn_save.setEnabled(False)
        thread = threading.Thread(target=work, name="report-bundle", daemon=True)
        thread.start()
        loop = QEventLoop()
        timer = QTimer()
        timer.setInterval(20)
        timer.timeout.connect(lambda: None if thread.is_alive() else loop.quit())
        timer.start()
        if thread.is_alive():
            loop.exec()
        timer.stop()
        thread.join()
        self.btn_save.setEnabled(bool(self.items))
        self.status.show()
        error = result.get("error")
        if error is not None:
            from evolab import diagnostics

            diagnostics.record("shown", S.REP_FAILED.format(reason=bundle.reason_text(error)), repr(error))
            self.status.setText(S.REP_FAILED.format(reason=bundle.reason_text(error)))
            self.btn_open_folder.hide()
            return None
        self.saved_path = target
        self.status.setText(S.REP_DONE.format(path=target))
        self.btn_open_folder.show()
        return target

    def open_folder(self) -> None:
        if self.saved_path is not None:
            bundle.reveal(self.saved_path)


def open_report_dialog(parent: QWidget | None, opts: BundleOptions) -> Path | None:
    """오류 보고 묶음 대화 상자(모달). 만든 zip 경로, 만들지 않았으면 None."""
    dlg = ReportDialog(parent, opts)
    try:
        dlg.exec()
        return dlg.saved_path
    finally:
        dlg.deleteLater()


def show_crash_box(parent: QWidget | None, line: str, opts: BundleOptions) -> None:
    """잡히지 않은 예외의 비모달 상자(설계 5.5): DIAG_CRASH_TEXT, 자세히 = 한 줄 요약, [확인] [오류 보고 묶음 만들기…]."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(S.DIAG_CRASH_TITLE)
    box.setText(S.DIAG_CRASH_TEXT)
    box.setDetailedText(line)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    report = box.addButton(S.ERR_REPORT_BUTTON, QMessageBox.ButtonRole.ActionRole)
    box.setModal(False)
    box.setWindowModality(Qt.WindowModality.NonModal)
    box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    _open_boxes.add(box)

    def done(_result: int) -> None:
        _open_boxes.discard(box)
        if box.clickedButton() is report:
            open_report_dialog(parent, opts)

    box.finished.connect(done)
    box.show()


def main_window() -> QMainWindow | None:
    """주 창(실험실을 가진 QMainWindow). 없으면 활성 창이 주 창이면 그것."""
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return None
    mains = [w for w in app.topLevelWidgets() if isinstance(w, QMainWindow) and hasattr(w, "lab")]
    mains.sort(key=lambda w: not w.isVisible())  # 보이는 창 먼저(닫혀 지워지기를 기다리는 창은 뒤로)
    if mains:
        return mains[0]
    active = app.activeWindow()
    return active if isinstance(active, QMainWindow) else None


def options_for(main: QWidget | None, reason: str) -> BundleOptions:
    """까닭 reason으로, 주 창 실험실의 지금 실행 폴더와 환경 글을 넣는 묶음 설정(주 창이 아니면 둘 다 없음)."""
    lab = getattr(main, "lab", None)
    run_dir = getattr(lab, "run_dir", None)
    env_text = getattr(lab, "env_text", None)
    return BundleOptions(reason=reason, run_dir=run_dir if isinstance(run_dir, Path) else None,
                         env_text=env_text if isinstance(env_text, str) else None)


def open_for(parent: QWidget | None, reason: str) -> Path | None:
    """도움말 메뉴·오류 상자의 「오류 보고 묶음 만들기…」: 주 창을 찾아 그 실험실의 실행·환경을 넣은 대화 상자를 연다."""
    return open_report_dialog(parent, options_for(main_window(), reason))


def on_crash(action: str, line: str) -> None:
    """`diagnostics.install_gui`의 알림 함수(화면 스레드에서 불린다). "box" = 비모달 상자, "again"·"more" = 주 창 상태 줄."""
    main = main_window()
    if action == "box":
        show_crash_box(main, line, options_for(main, "crash"))
        return
    text = (S.DIAG_CRASH_AGAIN if action == "again" else S.DIAG_CRASH_MORE).format(line=line)
    if main is not None:
        main.statusBar().showMessage(text, STATUS_MS)


def ask_previous(parent: QWidget | None) -> Path | None:
    """지난 실행이 갑자기 닫혔을 때(설계 D30) 한 번 묻는다. 「묶음 만들기…」면 대화 상자(까닭 "previous")."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(S.DIAG_PREVIOUS_TITLE)
    box.setText(S.DIAG_PREVIOUS_TEXT)
    yes = box.addButton(S.DIAG_PREVIOUS_YES, QMessageBox.ButtonRole.AcceptRole)
    no = box.addButton(S.DIAG_PREVIOUS_NO, QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(no)
    box.exec()
    clicked = box.clickedButton()
    box.deleteLater()
    if clicked is not yes:
        return None
    return open_report_dialog(parent, options_for(parent, "previous"))

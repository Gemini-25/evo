"""에디터 화면 (명세: 화면과 조작 > 에디터).

왼쪽 부품 팔레트, 가운데 캔버스, 오른쪽 속성 패널, 아래 탭(세계·사건·점수·랜덤·생성기·평가·뇌·매개변수·검사 결과).
기본 기능: 새로 만들기, 열기, 저장, 다른 이름으로 저장, 내장 예제에서 시작, 되돌리기·다시하기, 복사·붙여넣기,
직접 조종, 실험실로 보내기(멈춘 실험이면 고친 환경으로 이어서 학습, 아니면 새 실행에 쓸 환경).
v2: 캔버스 아래 미리보기 시각 줄(정해진 움직임 물체가 있을 때), 보기 메뉴(전체 보기, 기본 보기 저장·지우기), 내장 예제
목록은 paths.builtin_envs() 순서.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QSplitter,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from evolab import paths
from evolab.app import strings as S
from evolab.app.editor.canvas import EditorCanvas, PreviewBar
from evolab.app.editor.document import EnvDocument
from evolab.app.editor.forms import FieldWidget
from evolab.app.editor.palette import Palette
from evolab.app.editor.properties import PropertyPanel
from evolab.app.editor.tabs import BottomTabs
from evolab.spec.compiled import EnvFileError
from evolab.spec.io import parse_env

BUILTIN_TITLES = {"lander": S.ED_BUILTIN_LANDER, "track_car": S.ED_BUILTIN_TRACK, "hill_car": S.ED_BUILTIN_HILL,
                  "jump": S.ED_BUILTIN_JUMP, "glider": S.ED_BUILTIN_GLIDER}
"""내장 예제 메뉴 이름(파일 이름 줄기 -> 이름). 여기 없는 파일은 그 파일의 name."""


def _builtin_title(path: Path) -> str:
    title = BUILTIN_TITLES.get(path.stem)
    if title is not None:
        return title
    try:
        name = json.loads(path.read_text(encoding="utf-8-sig")).get("name")
    except (OSError, ValueError, AttributeError):
        name = None
    return name if isinstance(name, str) and name else path.stem


def builtin_menu_items() -> list[tuple[str, str]]:
    """「내장 예제에서 시작하기」 메뉴: (파일 이름, 메뉴 이름), 정해진 순서."""
    return [(f.name, _builtin_title(f)) for f in paths.builtin_envs()]


class EditorView(QWidget):
    send_to_lab = Signal(str)
    """실험실로 보내기(환경 내용)."""
    play_requested = Signal(object)
    """직접 조종(컴파일한 EnvSpec)."""
    title_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = EnvDocument(parent=self)
        self.from_lab = False
        """실험실의 멈춘 실험에서 연 환경인가(보내면 이어서 학습)."""

        self.canvas = EditorCanvas(self.doc)
        self.preview_bar = PreviewBar(self.canvas)
        self.palette = Palette()
        self.palette.connect_canvas(self.canvas)
        self.props = PropertyPanel(self.doc)
        self.tabs = BottomTabs(self.doc)
        self.tabs.issue_activated.connect(lambda _ref, _where: self.canvas.update())
        self.status = QLabel()
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.coords = QLabel()
        self.canvas.coords_changed.connect(lambda x, y: self.coords.setText(S.ED_COORDS.format(x=x, y=y)))
        self.canvas.status_changed.connect(self.status.setText)
        self.doc.checked.connect(self._on_checked)
        self.doc.file_changed.connect(lambda: self.title_changed.emit(self.doc.title()))

        self.toolbar = self._build_toolbar()
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(2)
        cl.addWidget(self.canvas, 1)
        cl.addWidget(self.preview_bar)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.palette)
        top.addWidget(center)
        top.addWidget(self.props)
        top.setStretchFactor(0, 0)
        top.setStretchFactor(1, 1)
        top.setStretchFactor(2, 0)
        top.setSizes([170, 800, 330])
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.toolbar)
        lay.addWidget(split, 1)
        status_row = QWidget()
        sl = QHBoxLayout(status_row)
        sl.setContentsMargins(4, 0, 4, 0)
        sl.addWidget(self.status, 1)
        sl.addWidget(self.coords)
        lay.addWidget(status_row)
        self._on_checked(self.doc.issues)

    # ------------------------------------------------------------------
    def _build_toolbar(self) -> QToolBar:
        bar = QToolBar()

        def act(text: str, slot: object, shortcut: QKeySequence | QKeySequence.StandardKey | None = None) -> QAction:
            a = QAction(text, self)
            a.triggered.connect(slot)  # type: ignore[arg-type]
            if shortcut is not None:
                a.setShortcut(shortcut)
                a.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            bar.addAction(a)
            self.addAction(a)
            return a

        self.act_new = act(S.ED_NEW, self.new_file, QKeySequence.StandardKey.New)
        self.act_open = act(S.ED_OPEN, self.open_file, QKeySequence.StandardKey.Open)
        self.act_save = act(S.ED_SAVE, self.save, QKeySequence.StandardKey.Save)
        self.act_save_as = act(S.ED_SAVE_AS, self.save_as, QKeySequence.StandardKey.SaveAs)
        menu = QMenu(self)
        for name, title in builtin_menu_items():
            menu.addAction(title).triggered.connect(lambda _c=False, n=name: self.start_from_builtin(n))
        self.builtin_menu = menu
        btn = QToolButton()
        btn.setText(S.ED_FROM_BUILTIN)
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        btn.setMenu(menu)
        bar.addWidget(btn)
        # 보기: 전체 보기(F, 기본 보기 <-> 장면 전체), 지금 보기를 기본 보기로, 기본 보기 지우기
        view_menu = QMenu(self)
        view_menu.addAction(self.canvas.fit_action)
        view_menu.addSeparator()
        self.act_view_save = view_menu.addAction(S.ED_VIEW_SAVE)
        self.act_view_save.triggered.connect(self.canvas.save_view_box)
        self.act_view_clear = view_menu.addAction(S.ED_VIEW_CLEAR)
        self.act_view_clear.triggered.connect(self.canvas.clear_view_box)
        view_menu.aboutToShow.connect(lambda: self.act_view_clear.setEnabled(self.canvas.view_box() is not None))
        self.view_menu = view_menu
        vbtn = QToolButton()
        vbtn.setText(S.ED_VIEW_MENU)
        vbtn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        vbtn.setMenu(view_menu)
        bar.addWidget(vbtn)
        bar.addSeparator()
        undo = self.doc.undo_stack.createUndoAction(self, S.ED_UNDO)
        undo.setShortcut(QKeySequence.StandardKey.Undo)
        redo = self.doc.undo_stack.createRedoAction(self, S.ED_REDO)
        redo.setShortcut(QKeySequence.StandardKey.Redo)
        for a in (undo, redo):
            a.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            bar.addAction(a)
            self.addAction(a)
        self.act_copy = act(S.ED_COPY, self.copy, QKeySequence.StandardKey.Copy)
        self.act_paste = act(S.ED_PASTE, self.paste, QKeySequence.StandardKey.Paste)
        bar.addSeparator()
        self.act_play = act(S.ED_PLAY, self.play)
        self.act_lab = act(S.ED_TO_LAB, self.to_lab)
        return bar

    def _on_checked(self, issues: list) -> None:
        errors = sum(1 for i in issues if i.severity == "error")
        warnings = sum(1 for i in issues if i.severity == "warning")  # 안내(info)는 문제가 아니라 세지 않는다
        self.status.setText(S.ED_STATUS.format(errors=errors, warnings=warnings) if errors or warnings
                            else S.ED_STATUS_OK)
        self.act_play.setEnabled(errors == 0)
        self.act_lab.setEnabled(errors == 0)

    # ------------------------------------------------------------------
    # 파일
    # ------------------------------------------------------------------
    def commit_pending(self) -> None:
        """포커스가 있는 칸에 아직 문서에 안 쓴 글자(참조 칸, 이름 칸)가 있으면 지금 쓴다.

        참조·이름 칸은 Enter나 칸을 떠날 때 쓴다. 저장 단축키(Ctrl+S)와 도구 모음 단추는 포커스를 옮기지 않으므로,
        저장·닫기·보내기 전에 먼저 쓰지 않으면 보이는 값과 저장한 값이 다르고 고친 것을 잃는다.
        """
        w = QApplication.focusWidget()
        while w is not None and w is not self:
            if isinstance(w, FieldWidget):
                w.flush()
                return
            w = w.parentWidget()

    def _check_saveable(self) -> bool:
        """구조 오류가 있으면 저장하지 않는다고 알린다(그런 파일은 다시 열 수 없다). 저장해도 되면 True."""
        try:
            parse_env(self.doc.data)
        except EnvFileError as exc:
            errors = "\n".join(str(i) for i in exc.issues if i.severity == "error")
            QMessageBox.warning(self, S.ED_SAVE_FAILED, S.RVE_SAVE_INVALID.format(errors=errors))
            return False
        return True

    def maybe_save(self) -> bool:
        """저장하지 않은 고침이 있으면 물어본다. 계속해도 되면 True."""
        self.commit_pending()
        if not self.doc.dirty:
            return True
        answer = QMessageBox.question(
            self, S.ED_UNSAVED_TITLE, S.ED_UNSAVED.format(name=self.doc.title()),
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def new_file(self) -> None:
        if self.maybe_save():
            self.doc.new()
            self.from_lab = False
            self.canvas.fit()

    def open_file(self, path: Path | None = None) -> bool:
        if not self.maybe_save():
            return False
        if path is None:
            start = str(self.doc.path.parent if self.doc.path else paths.user_env_dir())
            chosen, _ = QFileDialog.getOpenFileName(self, S.ED_OPEN, start, S.ENV_FILE_FILTER)
            if not chosen:
                return False
            path = Path(chosen)
        try:
            self.doc.open(path)
        except EnvFileError as exc:
            QMessageBox.warning(self, S.ED_OPEN_FAILED, str(exc))
            return False
        self.from_lab = False
        self.canvas.fit()
        return True

    def start_from_builtin(self, name: str) -> None:
        if self.maybe_save():
            self.doc.from_builtin(name)
            self.from_lab = False
            self.canvas.fit()

    def load_text(self, text: str, from_lab: bool = False) -> bool:
        """실험실 등에서 받은 환경 내용을 연다(저장 위치 없음)."""
        if not self.maybe_save():
            return False
        try:
            self.doc.load_text(text, None)
        except EnvFileError as exc:
            QMessageBox.warning(self, S.ED_OPEN_FAILED, str(exc))
            return False
        self.from_lab = from_lab
        self.canvas.fit()
        return True

    def save(self) -> bool:
        """저장. 구조 오류(다시 열 수 없는 내용)가 있으면 저장하지 않고 알린다(save_as도 같다)."""
        self.commit_pending()
        if self.doc.path is None:
            return self.save_as()
        if not self._check_saveable():
            return False
        try:
            self.doc.save()
        except (OSError, EnvFileError) as exc:
            QMessageBox.warning(self, S.ED_SAVE_FAILED, str(exc))
            return False
        return True

    def save_as(self, path: Path | None = None) -> bool:
        self.commit_pending()
        if not self._check_saveable():
            return False
        if path is None:
            folder = paths.user_env_dir()
            folder.mkdir(parents=True, exist_ok=True)
            name = str(self.doc.data.get("name") or S.ED_UNTITLED)
            chosen, _ = QFileDialog.getSaveFileName(self, S.ED_SAVE_AS, str(folder / f"{name}.evoenv"),
                                                    S.ENV_FILE_FILTER)
            if not chosen:
                return False
            path = Path(chosen)
            if path.suffix != ".evoenv":
                path = path.with_suffix(".evoenv")
        try:
            self.doc.save(path)
        except (OSError, EnvFileError) as exc:
            QMessageBox.warning(self, S.ED_SAVE_FAILED, str(exc))
            return False
        return True

    # ------------------------------------------------------------------
    # 편집
    # ------------------------------------------------------------------
    def copy(self) -> None:
        self.commit_pending()
        if self.doc.selection:
            QApplication.clipboard().setText(self.doc.copy_text())

    def paste(self) -> None:
        refs = self.doc.paste_text(QApplication.clipboard().text())
        if not refs:
            self.status.setText(S.ED_PASTE_NOTHING)

    def play(self) -> None:
        self.commit_pending()
        spec = self.doc.spec()
        if spec is None:
            QMessageBox.warning(self, S.ED_PLAY, S.ED_HAS_ERRORS)
            return
        self.play_requested.emit(spec)

    def to_lab(self) -> None:
        self.commit_pending()
        if self.doc.spec() is None:
            QMessageBox.warning(self, S.ED_TO_LAB, S.ED_HAS_ERRORS)
            return
        self.send_to_lab.emit(json.dumps(json.loads(self.doc.text()), ensure_ascii=False, indent=2) + "\n")

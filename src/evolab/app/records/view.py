"""기록 화면: 실행 목록과 열기, 이어하기, 복제해서 새 실행, 삭제, 메모 (명세: 저장, 기록, 내보내기 > 기록 화면).

목록은 runs 폴더를 읽어 만든다(evolab.runs.catalog.list_runs). 이 화면은 실행 폴더를 직접 바꾸는 일(메모, 삭제,
분기)만 하고, 실제 열기·학습은 신호로 주 창에 넘긴다.

- 이어하기: 체크포인트가 여럿이면 고르게 한다(기본은 최신). 최신이 아닌 것을 고르면 새 실행 폴더로 분기한다는
  것을 알리고, 분기한 뒤 resume_requested(새 폴더, 세대)를 보낸다.
- 학습 중인 실행은 set_busy_runs로 알려 준다: 그 실행은 지울 수 없고 이어 가거나(최신 체크포인트) 분기할 수 없다.
- resume_blocked: 이어하기를 받는 쪽(주 창의 실험실)이 지금 받을 수 없으면 까닭(한국어)을 돌려주는 함수. 주 창이
  넣는다. 체크포인트를 묻거나 분기하기 전에 물어서, 거절될 이어하기 때문에 분기 폴더가 생기지 않게 한다.
- 메모를 고치는 중에는 목록을 다시 읽지 않고(편집기가 닫히며 메모가 저장된 뒤에) 미룬다.
- 대화상자는 ask_checkpoint, confirm_branch, confirm_delete, show_error 메서드로만 띄운다. 테스트는 이 메서드를
  바꿔 끼우거나, 대화상자 없이 open_run / resume_run / clone_run / delete / save_memo를 바로 부른다.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemDelegate,
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from evolab import paths
from evolab.app import errors
from evolab.app import strings as S
from evolab.runs.catalog import RunSummary, branch_run, delete_run, list_runs, read_summary, set_memo
from evolab.runs.store import RunStore, RunStoreError

COL_ENV, COL_METHOD, COL_STARTED, COL_GENERATIONS, COL_BEST, COL_MEMO = range(6)
COLUMN_TITLES = (S.REC_COL_ENV, S.REC_COL_METHOD, S.REC_COL_STARTED, S.REC_COL_GENERATIONS, S.REC_COL_BEST,
                 S.REC_COL_MEMO)
PATH_ROLE = Qt.ItemDataRole.UserRole
COLOR_BROKEN = QColor("#9aa0aa")


def format_started(iso: str) -> str:
    """ISO 8601 시각을 '2026-11-07 15:30'으로. 읽을 수 없으면 그대로."""
    try:
        return f"{dt.datetime.fromisoformat(iso):%Y-%m-%d %H:%M}"
    except ValueError:
        return iso


def format_score(score: float | None) -> str:
    return S.REC_NONE if score is None else f"{score:.2f}"


def _started_key(iso: str) -> float:
    try:
        return dt.datetime.fromisoformat(iso).timestamp()
    except (ValueError, OverflowError, OSError):
        return float("-inf")


def _one_line(text: str) -> str:
    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def _same(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a == b


class _Item(QTableWidgetItem):
    """정렬 키를 따로 가진 칸(세대 수, 점수, 시각을 글자가 아니라 값으로 정렬)."""

    def __init__(self, text: str, key: Any) -> None:
        super().__init__(text)
        self.sort_key = key

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, _Item):
            try:
                return bool(self.sort_key < other.sort_key)
            except TypeError:
                pass
        return super().__lt__(other)


class _Table(QTableWidget):
    """칸 편집기가 닫힐 때(저장했든 Esc로 취소했든) editor_closed를 보내는 표."""

    editor_closed = Signal()

    def closeEditor(self, editor: QWidget, hint: QAbstractItemDelegate.EndEditHint) -> None:  # noqa: N802
        super().closeEditor(editor, hint)
        self.editor_closed.emit()


class RecordsView(QWidget):
    """실행 목록 화면."""

    open_requested = Signal(Path)
    """실행 폴더를 연다(다시 보기, 통계)."""
    resume_requested = Signal(Path, int)
    """(실행 폴더, 체크포인트 세대)에서 이어 간다. 최신이 아닌 체크포인트였다면 이미 분기한 새 폴더다."""
    clone_requested = Signal(Path)
    compare_requested = Signal(list)
    """고른 실행 폴더들(list[Path])을 비교 화면에 겹쳐 그린다."""
    """이 실행의 환경과 설정으로 새 실행을 만든다(catalog.clone_settings)."""

    def __init__(self, runs_dir_provider: Callable[[], Path] = paths.runs_dir, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._runs_dir_provider = runs_dir_provider
        self._runs: list[RunSummary] = []
        self._by_path: dict[str, RunSummary] = {}
        self._busy: list[Path] = []
        self._filling = False
        self._refresh_pending = False
        self.resume_blocked: Callable[[], str | None] = lambda: None
        """이어하기를 지금 시작할 수 없으면 까닭(한국어), 할 수 있으면 None. 주 창이 바꿔 넣는다(실험실이 학습 중 등)."""

        self.folder_label = QLabel()
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # 경로 글의 너비가 주 창의 최소 너비가 되지 않게(긴 데이터 폴더, 윈도우 CI의 네모 글자): 넘치면 잘리고 풍선에 전체 경로
        self.folder_label.setMinimumWidth(1)
        self.empty_label = QLabel(S.REC_EMPTY)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setWordWrap(True)

        self.table = _Table(0, len(COLUMN_TITLES))
        self.table.setHorizontalHeaderLabels(list(COLUMN_TITLES))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # 여러 줄을 골라 '비교'할 수 있다. 열기·이어하기 같은 단일 동작은 한 줄만 골랐을 때 된다.
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # 메모 칸만 고칠 수 있다(두 번 누르기 또는 F2). 다른 칸을 두 번 누르면 연다.
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(False)
        header = self.table.horizontalHeader()
        for col in range(len(COLUMN_TITLES)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_MEMO, QHeaderView.ResizeMode.Stretch)
        self.table.setSortingEnabled(True)
        self.table.sortItems(COL_STARTED, Qt.SortOrder.DescendingOrder)

        def button(text: str, slot: Callable[[], None], tip: str = "") -> QPushButton:
            b = QPushButton(text)
            b.clicked.connect(slot)
            if tip:
                b.setToolTip(tip)
            return b

        self.refresh_button = button(S.REC_REFRESH, self.refresh)
        self.open_button = button(S.REC_OPEN, self._on_open, S.REC_OPEN_TIP)
        self.resume_button = button(S.REC_RESUME, self._on_resume, S.REC_RESUME_TIP)
        self.clone_button = button(S.REC_CLONE, self._on_clone, S.REC_CLONE_TIP)
        self.memo_button = button(S.REC_EDIT_MEMO, self._on_edit_memo)
        self.delete_button = button(S.REC_DELETE, self._on_delete, S.REC_DELETE_TIP)
        self.compare_button = button(S.CMP_RECORDS_BUTTON, self._on_compare, S.CMP_RECORDS_BUTTON_TIP)
        self.bundle_button = button(S.REC_BUNDLE, self._on_bundle, S.REC_BUNDLE_TIP)
        self.csv_button = button(S.REC_CSV, self._on_csv, S.REC_CSV_TIP)

        buttons = QHBoxLayout()
        for b in (self.open_button, self.resume_button, self.clone_button, self.memo_button, self.compare_button,
                  self.bundle_button, self.csv_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.delete_button)
        buttons.addWidget(self.refresh_button)

        top = QHBoxLayout()
        top.addWidget(self.folder_label, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.empty_label)
        layout.addLayout(buttons)

        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.editor_closed.connect(self._after_edit)
        self.refresh()

    # ------------------------------------------------------------------
    # 목록
    # ------------------------------------------------------------------
    def runs_dir(self) -> Path:
        return Path(self._runs_dir_provider())

    def runs(self) -> list[RunSummary]:
        """지금 보이는 실행들(최근 것부터, 목록을 읽은 순서)."""
        return list(self._runs)

    def refresh(self) -> None:
        """runs 폴더를 다시 읽는다. 고른 실행은 그대로 고른 채로 둔다.

        메모를 고치는 중이면 미룬다: 표를 다시 채우면 열린 편집기가 저장되지 않고 닫혀 쓰던 메모가 사라진다.
        편집기가 닫히면(메모를 저장한 뒤) _after_edit가 다시 부른다.
        """
        if self.table.state() == QAbstractItemView.State.EditingState:
            self._refresh_pending = True
            return
        self._refresh_pending = False
        selected = self.selected_run()
        keep = selected.path if selected is not None else None
        base = self.runs_dir()
        self._runs = list_runs(base)
        self._by_path = {str(s.path): s for s in self._runs}
        self._fill()
        self.folder_label.setText(S.REC_FOLDER.format(count=len(self._runs), path=base))
        self.folder_label.setToolTip(str(base))
        self.empty_label.setVisible(not self._runs)
        if keep is not None:
            self.select_run(keep)
        self._update_buttons()

    def _after_edit(self) -> None:
        """메모 편집기가 닫혔다(저장한 뒤). 미뤄 둔 다시 읽기가 있으면 신호 처리가 끝난 뒤 한다."""
        if self._refresh_pending:
            QTimer.singleShot(0, self.refresh)

    def _fill(self) -> None:
        self._filling = True
        self.table.setSortingEnabled(False)
        try:
            self.table.clearContents()
            self.table.setRowCount(len(self._runs))
            for row, s in enumerate(self._runs):
                self._set_row(row, s)
        finally:
            self.table.setSortingEnabled(True)  # 머리글에서 고른 정렬 기준으로 다시 정렬
            self._filling = False

    @staticmethod
    def _cells(s: RunSummary) -> list[tuple[str, Any]]:
        """칸마다 (보이는 글, 정렬 키)."""
        method = S.METHOD_NAMES.get(s.method, s.method)
        env = s.env_name or s.name
        if s.parent is not None:
            env = S.REC_ENV_BRANCHED.format(env=env, generation=s.parent.get("generation", "?"))
        return [
            (env, env),
            (method if s.ok else S.REC_UNREADABLE, method),
            (format_started(s.started), _started_key(s.started)),
            (str(s.generations) if s.ok else S.REC_NONE, s.generations),
            (format_score(s.best_score), float("-inf") if s.best_score is None else s.best_score),
            (_one_line(s.memo), s.memo),
        ]

    def _set_row(self, row: int, s: RunSummary) -> None:
        tip = self._tooltip(s)
        for col, (text, key) in enumerate(self._cells(s)):
            item = _Item(text, key)
            flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
            if col == COL_MEMO and s.ok:
                flags |= Qt.ItemFlag.ItemIsEditable
            item.setFlags(flags)
            item.setData(PATH_ROLE, str(s.path))
            item.setToolTip(tip)
            if not s.ok:
                item.setForeground(QBrush(COLOR_BROKEN))
            if col in (COL_GENERATIONS, COL_BEST):
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, col, item)

    def _update_row(self, row: int, s: RunSummary) -> None:
        """있는 칸의 글과 정렬 키만 바꾼다(메모를 고친 itemChanged 안에서 칸을 바꿔 끼우면 안 된다)."""
        tip = self._tooltip(s)
        for col, (text, key) in enumerate(self._cells(s)):
            item = self.table.item(row, col)
            if isinstance(item, _Item):
                item.sort_key = key
                if item.text() != text:
                    item.setText(text)
                item.setToolTip(tip)

    @staticmethod
    def _tooltip(s: RunSummary) -> str:
        lines = [S.REC_TIP_FOLDER.format(path=s.path)]
        if s.parent is not None:
            lines.append(S.REC_TIP_PARENT.format(run=s.parent.get("run", "?"), generation=s.parent.get("generation", "?")))
        if s.error:
            lines.append(s.error)
        return "\n".join(lines)

    def _summary_at(self, row: int) -> RunSummary | None:
        item = self.table.item(row, COL_ENV)
        if item is None:
            return None
        return self._by_path.get(str(item.data(PATH_ROLE)))

    def selected_run(self) -> RunSummary | None:
        rows = {i.row() for i in self.table.selectedItems()}
        if len(rows) != 1:
            return None
        return self._summary_at(rows.pop())

    def selected_runs(self) -> list[RunSummary]:
        rows = sorted({i.row() for i in self.table.selectedItems()})
        return [s for s in (self._summary_at(r) for r in rows) if s is not None]

    def export_bundle(self, run: Path, out: Path) -> Path:
        from evolab.runs.bundle import export_bundle

        return export_bundle(run, out)

    def export_csv(self, run: Path, out: Path) -> Path:
        from evolab.runs.bundle import export_csv

        return export_csv(run, out)

    def _export(self, what: str) -> None:
        from PySide6.QtWidgets import QFileDialog

        s = self.selected_run()
        if s is None or not s.ok:
            return
        if what == "bundle":
            default, flt, fn = s.path / "exports" / S.BUNDLE_FILE_NAME.format(run=s.path.name), S.REC_ZIP_FILTER, \
                self.export_bundle
        else:
            default, flt, fn = s.path / "exports" / "stats.csv", S.REC_CSV_FILTER, self.export_csv
        chosen, _ = QFileDialog.getSaveFileName(self, S.REC_BUNDLE if what == "bundle" else S.REC_CSV, str(default), flt)
        if not chosen:
            return
        try:
            out = fn(s.path, Path(chosen))
        except (OSError, ValueError) as exc:
            why = errors.reason(exc)  # 날것 예외 글 대신 한국어 까닭(v3.1, v3.0 고침 13)
            self.show_error(f"{S.REC_EXPORT_FAILED}: {why}")
            return
        QMessageBox.information(self, S.REC_TITLE, S.REC_EXPORT_DONE.format(path=out))

    def _on_bundle(self) -> None:
        self._export("bundle")

    def _on_csv(self) -> None:
        self._export("csv")

    def _on_compare(self) -> None:
        runs = [s.path for s in self.selected_runs() if s.ok]
        if runs:
            self.compare_requested.emit(runs)

    def select_run(self, path: Path) -> bool:
        """이 폴더의 줄을 고른다. 목록에 없으면 False."""
        for row in range(self.table.rowCount()):
            s = self._summary_at(row)
            if s is not None and _same(s.path, Path(path)):
                self.table.selectRow(row)
                self.table.scrollToItem(self.table.item(row, COL_ENV))
                return True
        return False

    def set_busy_runs(self, runs: Iterable[Path]) -> None:
        """지금 학습 중인 실행 폴더들. 이 실행은 지울 수 없고 이어 가거나 분기할 수 없다."""
        self._busy = [Path(p) for p in runs]
        self._update_buttons()

    def is_busy(self, path: Path) -> bool:
        return any(_same(Path(path), b) for b in self._busy)

    def _update_buttons(self) -> None:
        s = self.selected_run()
        ok = s is not None and s.ok
        for b in (self.open_button, self.resume_button, self.clone_button, self.memo_button, self.bundle_button,
                  self.csv_button):
            b.setEnabled(ok)
        self.delete_button.setEnabled(s is not None and not self.is_busy(s.path))
        self.compare_button.setEnabled(any(r.ok for r in self.selected_runs()))

    # ------------------------------------------------------------------
    # 동작 (대화상자 없이 부를 수 있는 것)
    # ------------------------------------------------------------------
    def open_run(self, path: Path) -> None:
        self.open_requested.emit(Path(path))

    def clone_run(self, path: Path) -> None:
        self.clone_requested.emit(Path(path))

    def resume_run(self, path: Path, generation: int | None = None) -> Path:
        """체크포인트 generation(None이면 최신)에서 이어 가도록 resume_requested를 보낸다. 이어 갈 폴더를 돌려준다.

        최신이 아니면 새 실행 폴더로 분기하고(catalog.branch_run) 그 폴더를 보낸다. 실패하면 RunStoreError.
        학습 중인 실행(set_busy_runs)이거나 이어하기를 지금 받을 수 없으면(resume_blocked) 분기하기 전에 거절한다.
        """
        path = Path(path)
        self._check_can_resume(path)
        gens = RunStore.open(path).checkpoints()
        if not gens:
            raise RunStoreError(S.RUNS_NO_CHECKPOINT.format(path=path))
        g = gens[-1] if generation is None else int(generation)
        if g == gens[-1]:
            self.resume_requested.emit(path, g)
            return path
        new = branch_run(path, g, self.runs_dir())
        self.refresh()
        self.select_run(new)
        self.resume_requested.emit(new, g)
        return new

    def _check_can_resume(self, path: Path) -> None:
        """이 실행에서 이어 가거나 분기할 수 없으면 RunStoreError(한국어 까닭)."""
        if self.is_busy(path):
            # 다른 세션이 쓰고 있는 폴더: 이어 가면 두 프로세스가 같은 폴더에 쓰고, 분기는 쓰는 중인 파일을 복사한다
            raise RunStoreError(S.REC_RESUME_BUSY)
        why = self.resume_blocked()
        if why:
            raise RunStoreError(why)

    def delete(self, path: Path) -> None:
        """실행 폴더를 지우고 목록을 다시 읽는다(되돌릴 수 없다). 실패하면 RunStoreError."""
        path = Path(path)
        if self.is_busy(path):
            raise RunStoreError(S.REC_DELETE_BUSY)
        try:
            delete_run(path, self.runs_dir())
        finally:
            self.refresh()

    def save_memo(self, path: Path, memo: str) -> None:
        """메모를 run.json에 저장하고 그 줄을 고친다. 실패하면 RunStoreError(목록은 곧 다시 읽어 저장된 값으로 되돌린다).

        메모 칸의 itemChanged 안에서도 불리므로 칸을 바로 바꿔 끼우지 않는다(다시 읽기는 신호 처리가 끝난 뒤에).
        """
        path = Path(path)
        try:
            set_memo(path, memo)
        except (RunStoreError, OSError):
            QTimer.singleShot(0, self.refresh)
            raise
        self._refresh_one(path)

    def _refresh_one(self, path: Path) -> None:
        summary = read_summary(path)
        key = str(path)
        if key not in self._by_path:
            QTimer.singleShot(0, self.refresh)
            return
        self._by_path[key] = summary
        self._runs = [summary if str(s.path) == key else s for s in self._runs]
        for row in range(self.table.rowCount()):
            item = self.table.item(row, COL_ENV)
            if item is not None and str(item.data(PATH_ROLE)) == key:
                self._filling = True
                sorting = self.table.isSortingEnabled()
                self.table.setSortingEnabled(False)  # 칸을 바꾸는 동안 줄이 움직이지 않게
                try:
                    self._update_row(row, summary)
                finally:
                    self.table.setSortingEnabled(sorting)
                    self._filling = False
                break

    # ------------------------------------------------------------------
    # 대화상자 (테스트에서 바꿔 끼울 수 있다)
    # ------------------------------------------------------------------
    def ask_checkpoint(self, summary: RunSummary, generations: list[int]) -> int | None:
        """이어 갈 체크포인트를 고르게 한다(최신이 기본). 취소하면 None."""
        latest = generations[-1]
        choices = [(S.REC_CHECKPOINT_LATEST if g == latest else S.REC_CHECKPOINT_BRANCH).format(generation=g)
                   for g in reversed(generations)]
        text, ok = QInputDialog.getItem(self, S.REC_RESUME_TITLE, S.REC_RESUME_PICK, choices, 0, False)
        if not ok or text not in choices:
            return None
        return list(reversed(generations))[choices.index(text)]

    def confirm_branch(self, summary: RunSummary, generation: int) -> bool:
        answer = QMessageBox.question(
            self, S.REC_RESUME_TITLE, S.REC_BRANCH_CONFIRM.format(generation=generation),
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Ok,
        )
        return answer == QMessageBox.StandardButton.Ok

    def confirm_delete(self, summary: RunSummary) -> bool:
        answer = QMessageBox.warning(
            self, S.REC_DELETE_TITLE, S.REC_DELETE_CONFIRM.format(name=summary.name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def show_error(self, text: str) -> None:
        """오류 상자(다음 할 일 + 오류 보고 묶음 단추, errors.show_error). 테스트는 이 메서드를 바꿔 끼운다."""
        errors.show_error(self, S.REC_TITLE, text)

    # ------------------------------------------------------------------
    # 단추와 표
    # ------------------------------------------------------------------
    def _selected_ok(self) -> RunSummary | None:
        s = self.selected_run()
        return s if s is not None and s.ok else None

    def _on_open(self) -> None:
        s = self._selected_ok()
        if s is not None:
            self.open_run(s.path)

    def _on_clone(self) -> None:
        s = self._selected_ok()
        if s is not None:
            self.clone_run(s.path)

    def _on_resume(self) -> None:
        s = self._selected_ok()
        if s is None:
            return
        try:
            self._check_can_resume(s.path)  # 체크포인트를 묻고 분기를 확인받기 전에
            gens = RunStore.open(s.path).checkpoints()
        except (RunStoreError, OSError) as exc:
            why = errors.reason(exc)
            self.show_error(S.REC_RESUME_FAILED.format(reason=why))
            return
        if not gens:
            self.show_error(S.RUNS_NO_CHECKPOINT.format(path=s.path))
            return
        g: int | None = gens[-1]
        if len(gens) > 1:
            g = self.ask_checkpoint(s, gens)
            if g is None:
                return
            if g != gens[-1] and not self.confirm_branch(s, g):
                return
        try:
            self.resume_run(s.path, g)
        except (RunStoreError, ValueError, OSError) as exc:
            why = errors.reason(exc)
            self.show_error(S.REC_RESUME_FAILED.format(reason=why))

    def _on_delete(self) -> None:
        s = self.selected_run()
        if s is None or not self.confirm_delete(s):
            return
        try:
            self.delete(s.path)
        except (RunStoreError, OSError) as exc:
            why = errors.reason(exc)
            self.show_error(S.REC_DELETE_FAILED.format(reason=why))

    def _on_edit_memo(self) -> None:
        s = self._selected_ok()
        if s is None:
            return
        for row in range(self.table.rowCount()):
            if self._summary_at(row) is s:
                item = self.table.item(row, COL_MEMO)
                self.table.setCurrentItem(item)
                self.table.editItem(item)
                return

    def _on_double_click(self, row: int, col: int) -> None:
        if col == COL_MEMO:
            return  # 메모 칸은 고치기
        s = self._summary_at(row)
        if s is not None and s.ok:
            self.open_run(s.path)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling or item.column() != COL_MEMO:
            return
        s = self._by_path.get(str(item.data(PATH_ROLE)))
        if s is None or item.text() == _one_line(s.memo):
            return
        try:
            self.save_memo(s.path, item.text())
        except (RunStoreError, OSError) as exc:
            why = errors.reason(exc)
            self.show_error(S.REC_MEMO_FAILED.format(reason=why))

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh()  # 화면에 돌아올 때마다(학습이 끝났거나 새 실행이 생겼을 수 있다)

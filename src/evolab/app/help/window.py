"""도움말 창 (v3 설계 4.4, D18, D23).

비모달 창 하나(다시 열면 같은 창): 왼쪽 찾기 칸과 쪽 목록(찾으면 결과 목록과 앞뒤 글), 오른쪽 `QTextBrowser.setMarkdown`,
위쪽 ◀ ▶ · 처음 · 글자 크게/작게 · 「처음 실행 안내 다시 보기」, 아래쪽 알림 줄. 창 크기·글자 크기·마지막 쪽은 settings.ini
`[help]`에 둔다(설계 2.2).

링크(설계 2.5, D23):
- `evolab://help/<쪽>[#<제목>]`: 그 쪽을 열고 그 제목 블록으로 스크롤한다(Qt Markdown은 HTML 앵커를 버린다).
- `evolab://show/<연결표 키>`: 주 창을 그 탭으로 돌리고 그 위젯을 짚는다(A 묶음 `coach.point`, 없으면 풍선 글).
- `evolab://tour`(실험실이 쉴 때만), `evolab://report`, `evolab://data`, `evolab://logs`.
- 그 밖의 주소는 열지 않고 알림 줄에 HELPW_LINK_BLOCKED.

`hooks`: 통합(I 묶음 `help/setup.py`)이 `tour`·`report`·`data`·`logs` 동작을 바꿔 끼울 수 있는 자리(인자는 도움말 창의 주인 창).
비어 있으면 이 모듈의 기본 동작(A 묶음 `tour.start_tour`, D 묶음 `report.dialog.open_report_dialog`, 폴더 열기)을 쓴다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QCloseEvent, QColor, QHideEvent, QPalette, QTextBlockFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from evolab import paths
from evolab.app import strings as S
from evolab.app.help import anchors, content

SETTINGS_FILE: Final = "settings.ini"
"""실험실·대시보드·비교가 쓰는 화면 설정 파일(상태 폴더)."""

KEY_FONT: Final = "help/font_delta"
KEY_PAGE: Final = "help/last_page"
KEY_GEOMETRY: Final = "help/geometry"

FONT_DELTA_MIN: Final = -2
FONT_DELTA_MAX: Final = 6
"""글자 크기 조절 범위(앱 글꼴에 더하는 pt, 설계 2.2)."""

DEFAULT_SIZE: Final = (900, 640)
TEXT_MARGIN: Final = 16
"""글 둘레 여백(px, 다듬기 U34: 간격 표 16)."""
TEXT_MAX_WIDTH: Final = 720
"""창이 넓어도 글 줄은 이 폭(px)까지(한 줄이 한글 60자 안팎)."""
HEADING_SPACE: Final = {1: (0, 8), 2: (24, 8), 3: (16, 8)}
"""제목 수준 → (위, 아래) 틈(px). 쪽 제목(첫 줄)은 위 틈 없음."""
LINE_HEIGHT_PCT: Final = 150
"""본문 줄 높이(%)."""
LIST_INDENT: Final = 20
"""목록 들여쓰기 한 단(px, Qt 기본 40에서 줄임)."""
STATUS_MS: Final = 10000
"""알림 줄 글이 남아 있는 시간."""

SCHEME: Final = "evolab"
SIMPLE_ACTIONS: Final = ("tour", "report", "data", "logs")

hooks: dict[str, Callable[[QWidget | None], object]] = {}
"""링크 동작 바꿔 끼우기: "tour" | "report" | "data" | "logs" → f(주인 창). 없으면 기본 동작."""

_instance: HelpWindow | None = None


def _settings() -> QSettings:
    return QSettings(str(paths.user_state_dir() / SETTINGS_FILE), QSettings.Format.IniFormat)


def _alive(obj: object) -> bool:
    if obj is None:
        return False
    try:
        from shiboken6 import isValid

        return bool(isValid(obj))
    except (ImportError, RuntimeError):  # pragma: no cover - shiboken6는 PySide6와 함께 온다
        return True


# ---------------------------------------------------------------------------
# 주인 창 찾기
# ---------------------------------------------------------------------------
def main_window(widget: QWidget | None = None) -> QMainWindow | None:
    """widget이 속한 주 창, 없으면 보이는 주 창(닫혔지만 아직 지워지지 않은 창은 고르지 않는다)."""
    candidates: list[QWidget] = []
    if widget is not None and _alive(widget):
        candidates += [widget, QWidget.window(widget)]  # 속성 window를 가린 위젯도 있다(궤적 밀도 타일)
    app = QApplication.instance()
    if isinstance(app, QApplication):
        candidates += [w for w in app.topLevelWidgets() if w.isVisible()]
    for w in candidates:
        if anchors._is_main_window(w):  # noqa: SLF001 - 연결표와 같은 규칙(클래스 이름)
            return w  # type: ignore[return-value]
    return None


def _owner_for(parent: QWidget | None) -> QWidget | None:
    """도움말 창의 주인(Qt 부모): 모달 대화 상자가 떠 있으면 그 상자(아니면 도움말 창이 막힌다), 아니면 주 창, 없으면 parent의 창.

    주인이 있으면 주 창을 닫을 때 도움말 창도 함께 닫히고(마지막 창 판단에 들지 않음), 주인 위에 뜬다.
    """
    modal = QApplication.activeModalWidget()
    if modal is not None:
        return modal
    main = main_window(parent)
    if main is not None:
        return main
    if parent is not None and _alive(parent):
        return QWidget.window(parent)
    return None


# ---------------------------------------------------------------------------
# 짚기 (coach.point)
# ---------------------------------------------------------------------------
def point_at(widget: QWidget, text: str, more: tuple[str, str | None] | None = None) -> object | None:
    """widget을 짚는 말풍선(어둡게 하지 않음, 4초 또는 클릭으로 닫힘: coach.point)."""
    from evolab.app.help.coach import point

    return point(QWidget.window(widget), widget, text, more)  # type: ignore[arg-type]


def _ensure_visible(widget: QWidget) -> None:
    """widget을 감싼 스크롤 영역들이 widget을 보여 주게 스크롤한다."""
    w = widget.parentWidget()
    while w is not None:
        if isinstance(w, QScrollArea) and w.widget() is not None:
            w.ensureWidgetVisible(widget, 16, 16)
        w = w.parentWidget()


def _anchor_name(key: str) -> str:
    """연결표 키를 사람이 읽는 이름으로(위젯이 없을 때 알림 글): 단추 글, 타일 제목, 그 화면 도움말 쪽 제목."""
    anchor = anchors.ANCHORS.get(key)
    if anchor is None:
        return key
    if anchor.text is not None and isinstance(getattr(S, anchor.text, None), str):
        return str(getattr(S, anchor.text))
    if anchor.tile is not None:
        return S.DASH_TILE_TITLES.get(anchor.tile, key)
    from evolab.app.help.context import SCREEN_PAGES

    page = SCREEN_PAGES.get(anchor.screen)
    return content.title(page) if page is not None and content.has_page(page) else key


def show_anchor(key: str, parent: QWidget | None = None) -> QWidget | None:
    """`evolab://show/<key>`: 그 화면(탭)으로 가서 위젯이 보이게 하고 짚는다. 위젯을 찾지 못하면 None."""
    anchor = anchors.ANCHORS.get(key)
    main = main_window(parent)
    root = main if main is not None else parent
    if anchor is None or root is None:
        return None
    if main is not None:
        anchors.show_screen(main, anchor.screen)
        if anchor.tile is not None:
            dash = getattr(getattr(main, "lab", None), "dashboard", None)
            from evolab.app.lab.dashboard import PAGE_OF

            page = PAGE_OF.get(anchor.tile)
            if dash is not None and page is not None and getattr(dash, "page", page) != page:
                dash.set_page(page)
    widget = anchors.resolve(root, key)
    if widget is None or not widget.isVisibleTo(QWidget.window(widget)) or not QWidget.window(widget).isVisible():
        return None
    _ensure_visible(widget)
    win = QWidget.window(widget)
    win.raise_()
    win.activateWindow()
    point_at(widget, _anchor_tip(key, widget))
    return widget


def _anchor_tip(key: str, widget: QWidget) -> str:
    """짚기 말풍선 글: HELP_TIPS, 위젯 풍선, (타일이면) 제목 풍선, 이름 순."""
    title = getattr(widget, "title", None)
    title_tip = title.toolTip() if isinstance(title, QLabel) else ""
    return S.HELP_TIPS.get(key) or widget.toolTip() or title_tip or _anchor_name(key)


# ---------------------------------------------------------------------------
# 링크 기본 동작
# ---------------------------------------------------------------------------
def _open_folder(folder: Any) -> None:
    from PySide6.QtGui import QDesktopServices

    folder.mkdir(parents=True, exist_ok=True)
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


def restart_tour(owner: QWidget | None) -> str | None:
    """처음 실행 안내 다시 보기(도움말 링크·도움말 창 단추, 도움말 메뉴도 쓸 수 있음). 학습 중이면 HELPW_TOUR_BUSY(알림 글)를
    돌려주고 시작하지 않는다. 시작하면 None."""
    main = main_window(owner)
    if main is None:
        return S.HELPW_TOUR_BUSY
    client = getattr(getattr(main, "lab", None), "client", None)
    if client is not None and getattr(client, "active", False):
        return S.HELPW_TOUR_BUSY
    from evolab.app.help import tour

    lab_settings = getattr(getattr(main, "lab", None), "settings", None)  # 실험실과 같은 settings.ini 객체
    tour.start_tour(main, lab_settings if isinstance(lab_settings, QSettings) else _settings())
    return None


def _default_report(owner: QWidget | None) -> None:
    """오류 보고 묶음 대화 상자(report.dialog, 설계 7.3): 까닭 "menu", 지금 실험실의 실행 폴더와 환경 글."""
    from evolab.app.report.bundle import BundleOptions
    from evolab.app.report.dialog import open_report_dialog

    main = main_window(owner)
    lab = getattr(main, "lab", None)
    run_dir = getattr(lab, "run_dir", None)
    env_text = getattr(lab, "env_text", None)
    open_report_dialog(main if main is not None else owner,
                       BundleOptions(reason="menu", run_dir=run_dir, env_text=env_text if isinstance(env_text, str) else None))


def _default_data(_owner: QWidget | None) -> None:
    _open_folder(paths.data_dir())


def _default_logs(_owner: QWidget | None) -> None:
    _open_folder(paths.log_dir())


_DEFAULT_ACTIONS: Final[dict[str, Callable[[QWidget | None], object]]] = {
    "tour": restart_tour, "report": _default_report, "data": _default_data, "logs": _default_logs,
}


# ---------------------------------------------------------------------------
# 창
# ---------------------------------------------------------------------------
class HelpWindow(QWidget):
    """도움말 창. open_help()로 연다(하나를 다시 쓴다)."""

    def __init__(self, parent: QWidget | None = None, settings: QSettings | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(S.HELPW_TITLE)
        self.setObjectName("evolab_help_window")
        self.settings = settings if settings is not None else _settings()
        self.page: str = content.HOME
        """지금 쪽 id."""
        self.heading: str | None = None
        self._back: list[tuple[str, str | None]] = []
        self._forward: list[tuple[str, str | None]] = []
        self._query = ""
        """찾기 칸의 지금 글(앞뒤 빈칸 뺌)."""
        self.results: list[tuple[str, int, str]] = []
        """지금 찾기 결과 (쪽, 맞은 수, 앞뒤 글)."""
        self.status_texts: list[str] = []
        """알림 줄에 보인 글(차례대로, 테스트·자체 점검용)."""

        # 위쪽 단추 줄
        self.btn_back = QPushButton(S.HELPW_BACK)
        self.btn_forward = QPushButton(S.HELPW_FORWARD)
        self.btn_home = QPushButton(S.HELPW_HOME)
        self.btn_smaller = QPushButton(S.HELPW_FONT_SMALLER)
        self.btn_bigger = QPushButton(S.HELPW_FONT_BIGGER)
        self.btn_tour = QPushButton(S.MENU_HELP_TOUR)
        self.btn_back.clicked.connect(self.go_back)
        self.btn_forward.clicked.connect(self.go_forward)
        self.btn_home.clicked.connect(lambda: self.show_page(content.HOME))
        self.btn_smaller.clicked.connect(lambda: self.set_font_delta(self.font_delta - 1))
        self.btn_bigger.clicked.connect(lambda: self.set_font_delta(self.font_delta + 1))
        self.btn_tour.clicked.connect(lambda: self.run_action("tour"))
        bar = QHBoxLayout()
        for b in (self.btn_back, self.btn_forward, self.btn_home):
            bar.addWidget(b)
        bar.addStretch(1)
        for b in (self.btn_smaller, self.btn_bigger, self.btn_tour):
            bar.addWidget(b)

        # 왼쪽: 찾기 칸 + (쪽 목록 | 찾기 결과)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText(S.HELPW_SEARCH_PLACEHOLDER)
        self.search_box.setClearButtonEnabled(True)
        self.search_box.textChanged.connect(self.set_query)
        self.search_box.returnPressed.connect(self.find_next)
        self.page_list = QListWidget()
        for page in content.PAGES:
            item = QListWidgetItem(content.title(page))
            item.setData(Qt.ItemDataRole.UserRole, page)
            self.page_list.addItem(item)
        self.page_list.currentRowChanged.connect(self._page_row_changed)
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        self.result_list = QListWidget()
        self.result_list.setWordWrap(True)
        self.result_list.setSpacing(2)  # 결과 사이 4px(U34: 위아래 2px씩)
        self.result_list.itemClicked.connect(self._result_chosen)
        self.result_list.itemActivated.connect(self._result_chosen)
        results = QWidget()
        rl = QVBoxLayout(results)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.result_label)
        rl.addWidget(self.result_list, 1)
        self.left_stack = QStackedWidget()
        self.left_stack.addWidget(self.page_list)
        self.left_stack.addWidget(results)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.search_box)
        ll.addWidget(self.left_stack, 1)

        # 오른쪽: 글
        self.browser = QTextBrowser()
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.anchorClicked.connect(self.follow_link)
        self.browser.viewport().installEventFilter(self)  # 창 폭이 바뀌면 글 폭을 TEXT_MAX_WIDTH까지로(U34)
        self._base_pt = self.browser.font().pointSizeF()
        if self._base_pt <= 0:
            self._base_pt = 10.0

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.browser)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([240, 660])

        self.status = QLabel()
        self.status.setWordWrap(True)
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.setInterval(STATUS_MS)
        self._status_timer.timeout.connect(self.status.clear)
        self._heading_pos = -1
        self._heading_timer = QTimer(self)
        self._heading_timer.setSingleShot(True)
        self._heading_timer.setInterval(0)
        self._heading_timer.timeout.connect(lambda: self._scroll_block_top(self._heading_pos))

        lay = QVBoxLayout(self)
        lay.addLayout(bar)
        lay.addWidget(split, 1)
        lay.addWidget(self.status)

        self.font_delta = 0
        self.set_font_delta(self._int_setting(KEY_FONT, 0), save=False)
        self.resize(*DEFAULT_SIZE)
        geometry = self.settings.value(KEY_GEOMETRY)
        if geometry:
            try:
                self.restoreGeometry(geometry)  # type: ignore[arg-type]
            except TypeError:
                pass
        self._update_nav()

    # -- 설정 -----------------------------------------------------------------
    def _int_setting(self, key: str, default: int) -> int:
        try:
            return int(self.settings.value(key, default) or default)
        except (TypeError, ValueError):
            return default

    def save_geometry(self) -> None:
        self.settings.setValue(KEY_GEOMETRY, self.saveGeometry())
        self.settings.sync()

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        self.save_geometry()
        super().hideEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.save_geometry()
        super().closeEvent(event)

    def set_font_delta(self, delta: int, *, save: bool = True) -> None:
        """글자 크기: 앱 글꼴 + delta pt(−2..+6)."""
        delta = max(FONT_DELTA_MIN, min(FONT_DELTA_MAX, int(delta)))
        self.font_delta = delta
        font = self.browser.font()
        font.setPointSizeF(self._base_pt + delta)
        self.browser.setFont(font)
        self.browser.document().setDefaultFont(font)
        self.btn_smaller.setEnabled(delta > FONT_DELTA_MIN)
        self.btn_bigger.setEnabled(delta < FONT_DELTA_MAX)
        if save:
            self.settings.setValue(KEY_FONT, delta)

    # -- 알림 줄 --------------------------------------------------------------
    def say(self, text: str) -> None:
        self.status.setText(text)
        self.status_texts.append(text)
        self._status_timer.start()

    # -- 쪽 -------------------------------------------------------------------
    def show_page(self, page: str, heading: str | None = None, *, record: bool = True) -> bool:
        """쪽을 연다(제목이 있으면 그 제목으로 스크롤). 모르는 쪽이면 처음 쪽과 HELPW_PAGE_MISSING, 거짓."""
        ok = content.has_page(page)
        if not ok:
            self.say(S.HELPW_PAGE_MISSING.format(page=page))
            page, heading = content.HOME, None
        if record and (page, heading) != (self.page, self.heading) and self.browser.document().characterCount() > 1:
            self._back.append((self.page, self.heading))
            self._forward.clear()
        changed = page != self.page or self.browser.document().characterCount() <= 1
        self.page, self.heading = page, heading
        if changed:
            self.browser.setMarkdown(content.load_page(page))
            self.format_document()
            self.settings.setValue(KEY_PAGE, page)
        self._select_row(page)
        if heading:
            if not self.scroll_to_heading(heading):
                self.heading = None
        elif changed:
            self.browser.moveCursor(QTextCursor.MoveOperation.Start)
            self.browser.verticalScrollBar().setValue(0)
        self._update_nav()
        return ok

    # -- 글 모양 (6.2.0 다듬기 U34) -----------------------------------------
    def format_document(self) -> None:
        """setMarkdown 뒤 한 번: 글 여백 16, 제목 위아래 틈, 본문 줄 높이 150%, 목록 들여쓰기 줄임, 링크 색(ui.link).

        Qt의 setDefaultStyleSheet는 setMarkdown에 듣지 않으므로 글 덩이 형식을 직접 고친다. 글과 제목 자리는 그대로다(찾기·제목 스크롤)."""
        from evolab.app import style

        doc = self.browser.document()
        doc.setDocumentMargin(TEXT_MARGIN)
        doc.setIndentWidth(LIST_INDENT)
        pal = self.browser.palette()
        link = QColor(style.ui_for(self.browser).link)
        if pal.color(QPalette.ColorRole.Link) != link:
            pal.setColor(QPalette.ColorRole.Link, link)
            self.browser.setPalette(pal)
        cursor = QTextCursor(doc)
        cursor.beginEditBlock()
        block = doc.begin()
        first = True
        while block.isValid():
            fmt = QTextBlockFormat(block.blockFormat())
            level = fmt.headingLevel()
            if level > 0:
                top, bottom = HEADING_SPACE.get(level, (16, 8))
                fmt.setTopMargin(0 if first else top)
                fmt.setBottomMargin(bottom)
            else:
                fmt.setLineHeight(LINE_HEIGHT_PCT, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
            cursor.setPosition(block.position())
            cursor.setBlockFormat(fmt)
            first = False
            block = block.next()
        cursor.endEditBlock()
        self._fit_text_width()

    def _fit_text_width(self) -> None:
        """글 줄 폭을 창 폭과 TEXT_MAX_WIDTH(+ 여백) 가운데 작은 쪽으로."""
        from PySide6.QtWidgets import QTextEdit

        limit = TEXT_MAX_WIDTH + 2 * TEXT_MARGIN
        if self.browser.viewport().width() > limit:
            if (self.browser.lineWrapMode() != QTextEdit.LineWrapMode.FixedPixelWidth
                    or self.browser.lineWrapColumnOrWidth() != limit):
                self.browser.setLineWrapMode(QTextEdit.LineWrapMode.FixedPixelWidth)
                self.browser.setLineWrapColumnOrWidth(limit)
        elif self.browser.lineWrapMode() != QTextEdit.LineWrapMode.WidgetWidth:
            self.browser.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self.browser.viewport() and event.type() == QEvent.Type.Resize:
            self._fit_text_width()
        return super().eventFilter(obj, event)

    def _select_row(self, page: str) -> None:
        row = content.PAGES.index(page)
        if self.page_list.currentRow() != row:
            self.page_list.blockSignals(True)
            self.page_list.setCurrentRow(row)
            self.page_list.blockSignals(False)

    def _page_row_changed(self, row: int) -> None:
        if 0 <= row < len(content.PAGES):
            self.show_page(content.PAGES[row])

    def scroll_to_heading(self, heading: str) -> bool:
        """그 글의 제목 블록으로 커서를 옮기고 맨 위에 오게 스크롤한다. 없으면 거짓."""
        doc = self.browser.document()
        block = doc.begin()
        while block.isValid():
            if block.blockFormat().headingLevel() > 0 and block.text().strip() == heading:
                cursor = QTextCursor(block)
                self.browser.setTextCursor(cursor)
                self._heading_pos = block.position()
                self._scroll_block_top(self._heading_pos)
                self._heading_timer.start()  # 창이 처음 보일 때는 배치가 끝난 뒤 한 번 더
                return True
            block = block.next()
        return False

    def _scroll_block_top(self, position: int) -> None:
        if position < 0:
            return
        block = self.browser.document().findBlock(position)
        if not block.isValid():
            return
        top = self.browser.document().documentLayout().blockBoundingRect(block).top()
        self.browser.verticalScrollBar().setValue(int(top))

    def current_heading_block(self) -> str | None:
        """커서가 있는 블록이 제목이면 그 글(테스트용)."""
        block = self.browser.textCursor().block()
        return block.text() if block.blockFormat().headingLevel() > 0 else None

    # -- 앞뒤 -----------------------------------------------------------------
    def go_back(self) -> None:
        if not self._back:
            return
        self._forward.append((self.page, self.heading))
        page, heading = self._back.pop()
        self.show_page(page, heading, record=False)

    def go_forward(self) -> None:
        if not self._forward:
            return
        self._back.append((self.page, self.heading))
        page, heading = self._forward.pop()
        self.show_page(page, heading, record=False)

    def _update_nav(self) -> None:
        self.btn_back.setEnabled(bool(self._back))
        self.btn_forward.setEnabled(bool(self._forward))

    # -- 찾기 -----------------------------------------------------------------
    def set_query(self, text: str) -> None:
        """찾기 칸 글이 바뀜: 비면 쪽 목록, 아니면 결과 목록(찾은 곳 수, 쪽마다 맞은 수와 앞뒤 글)."""
        self._query = text.strip()
        if not self._query:
            self.results = []
            self.left_stack.setCurrentIndex(0)
            return
        self.results = content.search(self._query)
        self.result_list.clear()
        for page, count, snippet in self.results:
            item = QListWidgetItem(S.HELPW_RESULT_ITEM.format(page=content.title(page), count=count, snippet=snippet))
            item.setData(Qt.ItemDataRole.UserRole, page)
            self.result_list.addItem(item)
        if self.results:
            self.result_label.setText(S.HELPW_RESULTS.format(n=content.total_matches(self.results)))
        else:
            self.result_label.setText(self._no_results())
        self.left_stack.setCurrentIndex(1)

    def _no_results(self) -> str:
        return S.HELPW_NO_RESULTS.format(q=self._query, glossary=content.title(content.GLOSSARY))

    def _result_chosen(self, item: QListWidgetItem) -> None:
        page = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(page, str):
            self.open_result(page)

    def open_result(self, page: str) -> bool:
        """찾기 결과의 쪽을 열고 첫 맞은 곳을 고른다(강조)."""
        self.show_page(page)
        return self.highlight(first=True)

    def highlight(self, *, first: bool = False) -> bool:
        """지금 쪽에서 찾는 글을 고른다: first면 처음부터, 아니면 지금 고른 곳 다음부터(끝이면 처음으로 돌아감)."""
        if not self._query:
            return False
        if first:
            self.browser.moveCursor(QTextCursor.MoveOperation.Start)
        if self.browser.find(self._query):
            return True
        self.browser.moveCursor(QTextCursor.MoveOperation.Start)
        return self.browser.find(self._query)

    def find_next(self) -> bool:
        """찾기 칸에서 Enter: 지금 쪽에 맞는 곳이 있으면 다음 곳, 없으면 첫 결과 쪽을 열어 강조. 결과가 없으면 알림."""
        if not self._query:
            return False
        if not self.results:
            self.say(self._no_results())
            return False
        if any(page == self.page for page, _c, _s in self.results):
            return self.highlight()
        return self.open_result(self.results[0][0])

    # -- 링크 -----------------------------------------------------------------
    def follow_link(self, url: QUrl | str) -> bool:
        """도움말 링크를 따른다(설계 2.5). 우리 링크가 아니면 열지 않고 HELPW_LINK_BLOCKED. 따랐으면 참."""
        url = QUrl(url) if isinstance(url, str) else url
        text = url.toString()
        if url.scheme() != SCHEME:
            self.say(S.HELPW_LINK_BLOCKED.format(url=text))
            return False
        kind = url.host()
        if kind == "help":
            page = url.path().strip("/")
            heading = url.fragment(QUrl.ComponentFormattingOption.FullyDecoded) or None
            return self.show_page(page, heading)
        if kind == "show":
            key = url.path().strip("/")
            if show_anchor(key, self.parentWidget() or self) is None:
                self.say(S.HELPW_SHOW_MISSING.format(what=_anchor_name(key)))
                return False
            return True
        if kind in SIMPLE_ACTIONS and not url.path().strip("/"):
            return self.run_action(kind)
        self.say(S.HELPW_LINK_BLOCKED.format(url=text))
        return False

    def run_action(self, kind: str) -> bool:
        """tour·report·data·logs 동작. 바꿔 끼운 hooks가 있으면 그것. 안내를 시작하면 도움말 창은 닫는다(코치 말풍선을 가리지 않게)."""
        action = hooks.get(kind) or _DEFAULT_ACTIONS[kind]
        owner = self.parentWidget()
        result = action(owner)
        if isinstance(result, str):  # 알림 글(예: 학습 중이라 안내를 시작하지 못함)
            self.say(result)
            return False
        if kind == "tour":
            self.close()
        return True


# ---------------------------------------------------------------------------
# 공개 함수 (서명 고정: 설계 7.3)
# ---------------------------------------------------------------------------
def help_window() -> HelpWindow | None:
    """열려 있는(만들어 둔) 도움말 창. 없으면 None."""
    global _instance
    if _instance is not None and not _alive(_instance):
        _instance = None
    return _instance


class _ModalOwnerWatch(QObject):
    """도움말 창을 모달 상자(오류 상자, 보고 대화 상자)에 옮겨 단 동안 그 상자를 지켜본다: 상자가 숨거나 닫히면 지워지기 전에
    도움말 창을 원래 주인(주 창, 그 위에 다른 모달이 아직 떠 있으면 그 모달)으로 되돌린다. 상자는 닫히면 지워지므로(WA_DeleteOnClose,
    deleteLater) 그대로 두면 자식인 도움말 창도 함께 지워진다. destroyed는 늦다(~QWidget이 자식부터 지운다)."""

    def __init__(self, win: HelpWindow, modal: QWidget, home: QWidget | None) -> None:
        super().__init__(win)
        self.win = win
        self.modal = modal
        self.home = home
        modal.installEventFilter(self)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self.modal and event.type() in (QEvent.Type.Hide, QEvent.Type.Close):
            self.release()
        return False

    def release(self) -> None:
        """도움말 창을 모달에서 떼어 다음 주인에게 단다(보이던 창이면 다시 보인다). 두 번째부터는 아무것도 하지 않는다."""
        modal, self.modal = self.modal, None
        if modal is None:
            return
        if _alive(modal):
            modal.removeEventFilter(self)
        win = self.win
        if not _alive(win) or win.parentWidget() is not modal:
            self.deleteLater()
            return
        other = QApplication.activeModalWidget()
        if other is not None and other is not modal and other.isVisible() and not modal.isAncestorOf(other):
            owner: QWidget | None = other
        else:
            owner = main_window(self.home) if self.home is None or _alive(self.home) else main_window(None)
        _attach(win, owner, self.home)
        self.deleteLater()


def _attach(win: HelpWindow, owner: QWidget | None, home: QWidget | None) -> None:
    """도움말 창을 owner에 단다(창 자리·보임은 그대로). owner가 모달 상자면 상자가 닫힐 때 home 쪽으로 되돌아오게 지켜본다."""
    if win.parentWidget() is not owner:
        visible = win.isVisible()
        geometry = win.saveGeometry()
        win.setParent(owner, Qt.WindowType.Window)  # 부모를 바꾸면 숨는다
        win.restoreGeometry(geometry)
        if visible:
            win.show()
    if owner is not None and owner.isModal() and not any(w.modal is owner for w in win.findChildren(_ModalOwnerWatch)):
        _ModalOwnerWatch(win, owner, home)


def open_help(page: str = "start", heading: str | None = None, parent: QWidget | None = None) -> QWidget:
    """도움말 창을 열어 그 쪽(과 제목)을 보인다. 이미 있으면 같은 창을 다시 쓴다."""
    global _instance
    owner = _owner_for(parent)
    home = main_window(parent)
    win = help_window()
    if win is None:
        win = HelpWindow(owner)
        _instance = win
        if owner is not None and owner.isModal():
            _ModalOwnerWatch(win, owner, home)
    else:
        _attach(win, owner, home)
    win.show_page(page, heading)
    if win.isMinimized():
        win.setWindowState(win.windowState() & ~Qt.WindowState.WindowMinimized)
    win.show()
    win.raise_()
    win.activateWindow()
    return win


def browse_help(parent: QWidget | None = None) -> QWidget:
    """「도움말 찾아보기…」: 마지막에 본 쪽(없으면 처음 쪽)을 열고 찾기 칸에 초점을 둔다."""
    win = help_window()
    if win is not None and win.isVisible():
        page = win.page
    else:
        last = _settings().value(KEY_PAGE, content.HOME)
        page = last if isinstance(last, str) and content.has_page(last) else content.HOME
    shown = open_help(page, None, parent)
    assert isinstance(shown, HelpWindow)
    shown.search_box.setFocus(Qt.FocusReason.ShortcutFocusReason)
    shown.search_box.selectAll()
    return shown


def is_help_window(widget: QWidget | None) -> bool:
    """widget이 도움말 창(또는 그 안)인가."""
    while widget is not None:
        if isinstance(widget, HelpWindow):
            return True
        widget = widget.parentWidget()
    return False


__all__ = ["HelpWindow", "browse_help", "help_window", "hooks", "is_help_window", "main_window", "open_help",
           "point_at", "restart_tour", "show_anchor"]


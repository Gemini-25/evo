"""주 창에 v3.0 「처음 만나기」를 붙인다: 도움말 메뉴, 탭 줄 모서리 단추, F1·Shift+F1, 풍선 도움말, 빈 상태 (v3 설계 7.1, 5.1).

`MainWindow.__init__`가 `install(self)`를 부른다. 화면 파일(lab/view.py 등)은 고치지 않고 연결표(anchors.py)로 바깥에서 붙인다.

- 도움말 메뉴(설계 5.1 순서): 이 화면 도움말(F1) · 도움말 찾아보기… · 이건 뭐예요?(Shift+F1) · 처음 실행 안내 다시 보기 ─
  오류 보고 묶음 만들기… · 오류 기록 폴더 열기 ─ (주 창이 이미 단) 데이터 폴더 열기 · 정보….
  메뉴 항목의 F1·Shift+F1은 표시용이다. 키는 앱 이벤트 필터(context.HelpShortcuts)가 먼저 받아 마우스 아래 위젯의 도움말을 연다.
- 탭 줄 오른쪽 모서리 「? 도움말」: 지금 탭의 도움말(연결표 키 main.help_corner = `tabs.cornerWidget()`).
- 처음 실행 안내 제안은 여기서 하지 않는다. `run_gui()`가 창을 띄운 뒤 `after_show()`로 한다(설계 D12: `MainWindow()`만 만드는 자체
  점검·테스트·`evolab play`에는 안내가 없다).

도움말 창·안내·오류 보고 모듈은 메뉴를 처음 누를 때 불러온다(설계 D62).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QMainWindow, QMenu, QPushButton

from evolab import paths
from evolab.app import strings as S

STATUS_MS: Final = 10_000
"""메뉴 동작이 상태 줄에 보이는 알림 시간."""
CORNER_NAME: Final = "evolab_help_corner"
INSTALLED_PROPERTY: Final = "evolab_help_installed"


def help_menu(window: QMainWindow) -> QMenu:
    """주 창 메뉴 막대의 「도움말」 메뉴(없으면 만든다)."""
    for action in window.menuBar().actions():
        menu = action.menu()
        if isinstance(menu, QMenu) and action.text() == S.MENU_HELP:
            return menu
    return window.menuBar().addMenu(S.MENU_HELP)


def _action(menu: QMenu, text: str, slot: Callable[[], Any], shortcut: str | None = None) -> QAction:
    act = QAction(text, menu)
    if shortcut is not None:
        act.setShortcut(QKeySequence(shortcut))
    act.triggered.connect(lambda _checked=False: slot())
    return act


# ---------------------------------------------------------------------------
# 메뉴 동작
# ---------------------------------------------------------------------------
def open_screen_help(window: QMainWindow) -> None:
    from evolab.app.help import context

    context.open_screen_help(window)


def browse_help(window: QMainWindow) -> None:
    from evolab.app.help.window import browse_help as browse

    browse(window)


def whats_this(window: QMainWindow) -> None:
    from evolab.app.help import context

    context.start_whats_this(window)


def restart_tour(window: QMainWindow) -> Any:
    """「처음 실행 안내 다시 보기」: 처음부터(학습 중이면 상태 줄에 ONB_BUSY만, 설계 D13·D14)."""
    from evolab.app.help import tour

    return tour.start_tour(window, tour.tour_settings(window))


def make_report(window: QMainWindow) -> Any:
    """「오류 보고 묶음 만들기…」: 까닭 menu, 지금 실험실의 실행 폴더와 환경 글을 넣는 대화 상자(설계 D32)."""
    from evolab.app.report.dialog import open_for

    return open_for(window, "menu")


def open_logs(_window: QMainWindow) -> None:
    """「오류 기록 폴더 열기」: 상태 폴더의 logs(errors.jsonl, faults.log, stderr.log, selftest.log)."""
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    folder = paths.log_dir()
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


def add_help_menu(window: QMainWindow) -> list[QAction]:
    """도움말 메뉴 맨 위에 도움말·안내·오류 보고 항목을 넣는다(주 창이 단 항목 앞, 설계 5.1)."""
    menu = help_menu(window)
    existing = menu.actions()
    before = existing[0] if existing else None
    actions = [
        _action(menu, S.MENU_HELP_SCREEN, lambda: open_screen_help(window), "F1"),
        _action(menu, S.MENU_HELP_BROWSE, lambda: browse_help(window)),
        _action(menu, S.MENU_HELP_WHATS_THIS, lambda: whats_this(window), "Shift+F1"),
        _action(menu, S.MENU_HELP_TOUR, lambda: restart_tour(window)),
        None,
        _action(menu, S.MENU_REPORT, lambda: make_report(window)),
        _action(menu, S.MENU_OPEN_LOGS, lambda: open_logs(window)),
    ]
    made: list[QAction] = []
    for act in actions:
        if act is None:
            sep = QAction(menu)
            sep.setSeparator(True)
            act = sep
        if before is not None:
            menu.insertAction(before, act)
        else:
            menu.addAction(act)
        made.append(act)
    if before is not None:
        menu.insertSeparator(before)
    return made


def add_corner_button(window: QMainWindow) -> QPushButton:
    """탭 줄 오른쪽 모서리 「? 도움말」(설계 D21, 5.1): 누르면 지금 탭의 도움말."""
    btn = QPushButton(S.HELP_CORNER_BUTTON)
    btn.setObjectName(CORNER_NAME)
    btn.setToolTip(S.HELP_CORNER_TIP)
    btn.setFlat(True)
    btn.clicked.connect(lambda _checked=False: open_screen_help(window))
    window.tabs.setCornerWidget(btn)  # type: ignore[attr-defined]
    return btn


# ---------------------------------------------------------------------------
# 붙이기
# ---------------------------------------------------------------------------
def install(window: QMainWindow) -> None:
    """도움말 메뉴·모서리 단추·F1 필터·풍선 도움말·빈 상태를 주 창에 붙인다. 같은 창에 두 번 불러도 한 번만."""
    if window.property(INSTALLED_PROPERTY):
        return
    window.setProperty(INSTALLED_PROPERTY, True)
    from evolab.app.help import hints, tooltips

    add_help_menu(window)
    add_corner_button(window)
    hints.install_hints(window)
    for screen in tooltips.MAIN_SCREENS:
        tooltips.apply_tooltips(window, screen)
    # 앱 전체 이벤트 필터 둘(F1·Shift+F1, 따로 뜨는 창의 풍선)은 창이 처음 그려진 다음 이벤트 루프 차례에 단다: 첫 보이기·첫
    # 그리기의 이벤트 수만 개가 파이썬 필터를 거치지 않게(M3: 창 뜨는 시간). 보이지 않는 창(만들기만 한 창)은 달지 않는다.
    window.installEventFilter(_FirstShow(window))


FILTERS_FALLBACK_MS: Final = 1000
"""주 창이 보인 뒤 이 시간 안에 그려지지 않으면(최소화한 채 시작 등) 그때 앱 전체 필터를 단다."""


class _FirstShow(QObject):
    """주 창이 처음 보인 뒤 처음 그려지면 다음 이벤트 루프 차례로 install_app_filters를 예약하고 스스로 떨어진다(주 창 하나의
    이벤트만 본다).

    보일 때(Show) 곧바로 0ms 타이머를 걸면, 그 타이머가 같은 이벤트 처리 차례의 첫 그리기(Expose → Paint)보다 먼저 돌아 첫 그리기의
    이벤트가 모두 파이썬 필터를 거쳤다(필터 하나에 첫 그리기 5~10ms, M3 재측정). 그래서 Paint를 받은 뒤에 예약한다. 그리기 중에
    건 0ms 타이머는 그 처리 차례가 끝난 뒤에 돈다. 그려지지 않으면 FILTERS_FALLBACK_MS 뒤에 단다."""

    def __init__(self, window: QMainWindow) -> None:
        super().__init__(window)
        self._window = window
        self._shown = False

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        kind = event.type()
        if kind == QEvent.Type.Show and obj is self._window and not self._shown:
            self._shown = True
            QTimer.singleShot(FILTERS_FALLBACK_MS, self, self._install)
        elif kind == QEvent.Type.Paint and obj is self._window and self._shown:
            obj.removeEventFilter(self)
            QTimer.singleShot(0, self, self._install)
        return False

    def _install(self) -> None:
        window = self._window
        try:
            window.removeEventFilter(self)
        except RuntimeError:  # 창이 이미 지워졌다
            return
        self.deleteLater()
        install_app_filters(window)


def install_app_filters(window: QMainWindow) -> None:
    """F1·Shift+F1 필터(context)와 따로 뜨는 창 풍선 필터(tooltips)를 앱에 단다. 이미 이 창의 F1 필터가 있으면 다시 달지 않는다."""
    from evolab.app.help import context, tooltips

    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return
    current = context.shortcuts()
    if current is None or current.window is not window:
        context.install_shortcuts(app, window)
    tooltips.install(app, window)


def after_show(window: QMainWindow, previous_crash: bool = False) -> None:
    """`run_gui()`가 창을 띄운 뒤 이벤트 루프에서 한 번: 지난번 갑자기 닫혔으면 묻고(설계 D30), 그다음 처음 실행 안내를
    제안한다(설계 D12: 설정 키가 없고 실행 기록이 없으면 환영 카드, 실행 기록이 있으면 상태 줄 알림 한 번, EVOLAB_NO_TOUR면 없음)."""
    if previous_crash:
        from evolab.app.report import dialog

        dialog.ask_previous(window)  # 모달: 끝난 뒤에 안내를 띄워 상자와 겹치지 않게
    from evolab.app.help import tour

    tour.offer(window)


def schedule_after_show(window: QMainWindow, previous_crash: bool = False) -> None:
    QTimer.singleShot(0, window, lambda: after_show(window, previous_crash))


__all__ = ["add_corner_button", "add_help_menu", "after_show", "browse_help", "help_menu", "install", "install_app_filters",
           "make_report", "open_logs", "open_screen_help", "restart_tour", "schedule_after_show", "whats_this"]

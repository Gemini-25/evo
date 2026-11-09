"""v3.0 통합 (v3 설계 7.1 I 묶음, 8.1 T19·T28·T29): 주 창에 붙인 도움말 메뉴·모서리 단추, Qt 한국어, run_gui의 진단·안내 제안,
자체 점검의 새 점검 넷.

모든 테스트가 데이터·상태 폴더를 임시 폴더로 둔다(EVOLAB_DATA_DIR·EVOLAB_STATE_DIR, 설계 D5).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QTranslator, QUrl
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

from evolab import diagnostics, paths, selftest
from evolab.app import strings as S
from evolab.app.help import anchors, coach, tour
from evolab.app.help import setup as help_setup

DATA = Path(__file__).resolve().parent / "data"
OLD_RUN = DATA / "v11_run_restart" / "run"
NEW_CHECKS = (S.SELFTEST_CHECK_HELP, S.SELFTEST_CHECK_LOGS, S.SELFTEST_CHECK_QT_KO, S.SELFTEST_CHECK_TOUR)


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv(tour.NO_TOUR_ENV, "1")
    return tmp_path


@pytest.fixture()
def window(qtbot: Any, isolated: Path) -> Any:
    from evolab.app.main import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1280, 800)
    w.show()
    qtbot.waitExposed(w)
    yield w
    w.lab.client.abandon(timeout=60)


def _items(menu: Any) -> list[str]:
    return ["-" if a.isSeparator() else a.text() for a in menu.actions()]


def _overlays(w: QMainWindow) -> list[coach.CoachOverlay]:
    return [o for o in w.findChildren(coach.CoachOverlay) if o.running]


# ---------------------------------------------------------------------------
# T19 Qt 한국어
# ---------------------------------------------------------------------------
def _qm_available() -> bool:
    from evolab.app.main import qt_translation_dirs

    return any((d / "qtbase_ko.qm").is_file() for d in qt_translation_dirs())


def test_create_application_translates_qt_standard_buttons(qtbot: Any) -> None:
    """T19: create_application 뒤 표준 단추가 저장·닫기·취소·예(&Y)·아니요(&N). 번역기는 앱에 하나만."""
    if not _qm_available():
        pytest.skip("qtbase_ko.qm이 없는 Qt 설치")
    from evolab.app.main import QT_TRANSLATOR_NAME, create_application

    app = create_application()
    assert selftest.qt_button_texts() == ["저장", "닫기", "취소", "예(&Y)", "아니요(&N)"]
    create_application()
    assert len(app.findChildren(QTranslator, QT_TRANSLATOR_NAME)) == 1


def test_missing_qt_translation_is_recorded_and_english_kept(qtbot: Any, isolated: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """번역 파일을 못 찾으면 실패해도 계속하고 오류 기록에 qt 한 줄(설계 4.7). 다시 찾을 수 있으면 다시 넣는다."""
    from evolab.app import main as main_mod

    app = QApplication.instance()
    assert isinstance(app, QApplication)
    for tr in app.findChildren(QTranslator, main_mod.QT_TRANSLATOR_NAME):
        app.removeTranslator(tr)
        tr.setObjectName("")
        tr.deleteLater()
    monkeypatch.setattr(main_mod, "qt_translation_dirs", lambda: [isolated / "없는 폴더"])
    try:
        assert main_mod.install_qt_translation(app) is False
        assert "Save" in selftest.qt_button_texts()[0]
        (rec,) = diagnostics.recent()
        assert rec["kind"] == "qt" and rec["message"] == "qtbase_ko not loaded"
    finally:
        monkeypatch.undo()
        main_mod.install_qt_translation(app)


# ---------------------------------------------------------------------------
# T29 도움말 메뉴·모서리 단추·안내 제안
# ---------------------------------------------------------------------------
def test_help_menu_order_and_shortcuts(window: Any) -> None:
    """T29: 도움말 메뉴 항목 순서(설계 5.1), F1·Shift+F1 표시, 같은 창에 두 번 붙여도 한 번."""
    menu = help_setup.help_menu(window)
    want = [S.MENU_HELP_SCREEN, S.MENU_HELP_BROWSE, S.MENU_HELP_WHATS_THIS, S.MENU_HELP_TOUR, "-", S.MENU_REPORT,
            S.MENU_OPEN_LOGS, "-", S.MENU_OPEN_DATA, S.MENU_ABOUT]
    assert _items(menu) == want
    by_text = {a.text(): a for a in menu.actions()}
    assert by_text[S.MENU_HELP_SCREEN].shortcut() == QKeySequence("F1")
    assert by_text[S.MENU_HELP_WHATS_THIS].shortcut() == QKeySequence("Shift+F1")
    help_setup.install(window)
    assert _items(menu) == want
    assert [a.text() for a in window.menuBar().actions()] == [S.MENU_PLAY, S.MENU_SETTINGS, S.MENU_HELP]


def test_corner_button_is_the_help_corner_anchor(window: Any) -> None:
    corner = window.tabs.cornerWidget()
    assert isinstance(corner, QPushButton) and corner.text() == S.HELP_CORNER_BUTTON
    assert corner.toolTip() == S.HELP_CORNER_TIP
    assert anchors.resolve(window, "main.help_corner") is corner and corner.isVisible()
    assert anchors.screen_of(corner) == "main"


def test_help_menu_actions_do_their_jobs(qtbot: Any, window: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """메뉴 동작: 이 화면 도움말(지금 탭 쪽), 찾아보기, 이건 뭐예요?, 안내 다시 보기, 오류 보고 묶음(까닭 menu), 오류 기록 폴더."""
    from PySide6.QtGui import QDesktopServices

    from evolab.app.help import context
    from evolab.app.help import window as HW
    from evolab.app.report import dialog

    by_text = {a.text(): a for a in help_setup.help_menu(window).actions()}
    try:
        window.tabs.setCurrentWidget(window.records)
        by_text[S.MENU_HELP_SCREEN].trigger()
        win = HW.help_window()
        assert win is not None and win.page == "records"
        by_text[S.MENU_HELP_BROWSE].trigger()
        assert HW.help_window() is win

        by_text[S.MENU_HELP_WHATS_THIS].trigger()
        flt = context.shortcuts()
        assert flt is not None and flt.whats_this_active and flt.window is window
        flt.stop_whats_this()

        seen: list[Any] = []
        monkeypatch.setattr(dialog, "open_report_dialog", lambda parent, opts: seen.append((parent, opts)) or None)
        by_text[S.MENU_REPORT].trigger()
        (parent, opts), = seen
        assert parent is window and opts.reason == "menu" and opts.run_dir is None
        assert opts.env_text == window.lab.env_text

        opened: list[QUrl] = []
        monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url) or True))
        by_text[S.MENU_OPEN_LOGS].trigger()
        assert [Path(u.toLocalFile()) for u in opened] == [paths.log_dir()] and paths.log_dir().is_dir()

        by_text[S.MENU_HELP_TOUR].trigger()  # 다시 보기는 끄개(EVOLAB_NO_TOUR)와 상관없이 처음부터
        ov = tour.running_tour(window)
        assert ov is not None and ov.current is not None and ov.current.key == "welcome"
        assert ov.press(S.ONB_WELCOME_LATER)
    finally:
        win = HW.help_window()
        if win is not None:
            win.close()
            win.deleteLater()
        HW._instance = None


def test_escape_in_whats_this_mode_does_not_skip_a_running_tour(qtbot: Any, window: Any) -> None:
    """F1 필터는 주 창을 만들 때, 안내의 코치 필터는 안내를 띄울 때 달린다(나중 것이 먼저 받음). 「이건 뭐예요?」 중의 Esc는
    그 모드만 끝내고 안내는 건너뛰지 않는다."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from evolab.app.help import context

    ov = tour.start_tour(window, window.lab.settings)
    assert ov is not None and ov.current is not None and ov.current.key == "welcome"
    context.start_whats_this(window)
    flt = context.shortcuts()
    assert flt is not None and flt.whats_this_active
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert not flt.whats_this_active and ov.running and ov.result is None
    QTest.keyClick(window, Qt.Key.Key_Escape)  # 모드가 끝난 뒤의 Esc는 안내 건너뛰기(설계 D13)
    assert ov.result == "skipped"


def test_about_text_names_the_real_data_and_state_folders(window: Any) -> None:
    text = window.about_text()
    assert str(paths.data_dir()) in text and str(paths.user_state_dir()) in text
    assert text.startswith(f"{S.APP_NAME} ")


def test_play_open_failure_uses_friendly_error_box(window: Any, isolated: Path) -> None:
    """주 창의 날것 예외 상자 셋을 show_error로(설계 D26): 망가진 환경 파일 → 다음 할 일과 묶음 단추가 있는 상자."""
    from evolab.app import errors

    bad = isolated / "망가진.evoenv"
    bad.write_text("{", encoding="utf-8")
    assert window.open_play(bad) is None
    boxes = [b for b in errors.open_error_boxes() if b.parent() is window]
    assert len(boxes) == 1 and boxes[0].windowTitle() == S.PLAY_OPEN_FAILED
    assert boxes[0].friendly.next_step and boxes[0].detailedText()
    boxes[0].close()


def test_closing_the_window_clears_the_running_flag(window: Any, isolated: Path) -> None:
    """closeEvent 끝에서 mark_clean_exit(설계 D30): 정상으로 닫으면 running-<pid>.flag가 없다."""
    flag = paths.log_dir() / f"{diagnostics.FLAG_PREFIX}{os.getpid()}{diagnostics.FLAG_SUFFIX}"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("", encoding="ascii")
    window.close()
    assert not flag.exists()


class _FakeApp:
    """run_gui의 app.exec 대신: 이벤트를 돌려 창이 뜬 뒤의 일을 보고 창을 닫는다."""

    def __init__(self, qtbot: Any, until: Any) -> None:
        self.qtbot = qtbot
        self.until = until
        self.seen: dict[str, Any] = {}

    def exec(self) -> int:
        main = next(w for w in QApplication.topLevelWidgets() if isinstance(w, QMainWindow) and hasattr(w, "lab")
                    and w.isVisible())
        self.seen["flag"] = sorted(p.name for p in paths.log_dir().glob("running-*.flag"))
        self.qtbot.waitUntil(lambda: self.until(main), timeout=10_000)
        self.qtbot.wait(100)
        self.seen["window"] = main
        self.seen["overlays"] = [o.current.key for o in _overlays(main) if o.current is not None]
        self.seen["status"] = main.statusBar().currentMessage()
        self.seen["state"] = tour.tour_state(main.lab.settings)
        for ov in _overlays(main):
            ov.finish("later")
        main.close()
        main.deleteLater()
        return 0


@pytest.fixture()
def gui_run(qtbot: Any, isolated: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """run_gui()를 그대로 부르되 앱 루프만 가짜로. 진단 훅은 끝나면 되돌린다."""
    from evolab.app import main as main_mod

    monkeypatch.delenv(tour.NO_TOUR_ENV, raising=False)
    diagnostics.uninstall()

    def run(until: Any) -> dict[str, Any]:
        fake = _FakeApp(qtbot, until)
        monkeypatch.setattr(main_mod, "create_application", lambda: fake)
        assert main_mod.run_gui() == 0
        fake.seen["errors"] = diagnostics.recent()
        fake.seen["flag_after"] = sorted(p.name for p in paths.log_dir().glob("running-*.flag"))
        return fake.seen

    yield run
    diagnostics.uninstall()
    diagnostics.mark_clean_exit()


def test_run_gui_first_launch_shows_welcome_card(gui_run: Any) -> None:
    """D12: 설정 키도 실행 기록도 없으면 run_gui가 창을 띄운 뒤 환영 카드. 실행 중 표시가 생기고 정상 종료에 지워진다(D30),
    시작 기록(start) 한 줄(D29), 내장 예제가 데이터 폴더에(첫 실행)."""
    seen = gui_run(lambda w: bool(_overlays(w)))
    assert seen["overlays"] == ["welcome"]
    assert seen["flag"] == [f"running-{os.getpid()}.flag"] and seen["flag_after"] == []
    kinds = [r["kind"] for r in seen["errors"]]
    assert kinds[0] == "start" and set(kinds[1:]) <= {"qt"}  # 오프스크린 Qt 경고(propagateSizeHints)는 qt 기록
    assert json.loads(seen["errors"][0]["detail"])["frozen"] is False
    assert (paths.user_env_dir() / "hill_car.evoenv").is_file()


def test_run_gui_existing_user_gets_hint_once(gui_run: Any) -> None:
    """T29: 실행 기록이 있으면 환영 카드 대신 상태 줄 알림 한 번(offered), 다음 실행에는 없음."""
    shutil.copytree(OLD_RUN, paths.runs_dir() / "old")
    hint = S.ONB_EXISTING_USER_HINT.format(menu=S.MENU_HELP)
    first = gui_run(lambda w: w.statusBar().currentMessage() == hint)
    assert first["overlays"] == [] and first["status"] == hint and first["state"] == "offered"
    second = gui_run(lambda w: True)
    assert second["overlays"] == [] and second["status"] == "" and second["state"] == "offered"


def test_run_gui_asks_about_previous_crash_before_the_tour(gui_run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """지난 실행이 갑자기 끝났으면(살아 있지 않은 pid의 표시) 먼저 묻고, 그다음 안내를 제안한다(상자와 겹치지 않게)."""
    from evolab.app.report import dialog

    order: list[str] = []
    monkeypatch.setattr(dialog, "ask_previous", lambda parent: order.append("ask") or None)
    real_offer = tour.offer
    monkeypatch.setattr(tour, "offer", lambda w, s=None: order.append("offer") or real_offer(w, s))
    dead = paths.log_dir() / "running-999999999.flag"
    dead.parent.mkdir(parents=True, exist_ok=True)
    dead.write_text("", encoding="ascii")
    seen = gui_run(lambda w: bool(_overlays(w)))
    assert order == ["ask", "offer"] and seen["overlays"] == ["welcome"]
    assert not dead.exists()


def test_main_window_alone_installs_help_but_offers_nothing(qtbot: Any, window: Any) -> None:
    """MainWindow()만(자체 점검·테스트·evolab play): 도움말·풍선·빈 상태는 붙고, 안내 제안은 없다(D12). 앱 전체 필터는 창이
    처음 그려진 다음 차례에 달린다(M3 고침: 아래 test_app_wide_filters_wait_for_the_first_paint)."""
    from evolab.app.help import context, hints, tooltips

    assert getattr(window, "evolab_hints", None) and hints.hint_of(window.lab.canvas) is not None
    qtbot.waitUntil(lambda: context.shortcuts() is not None and tooltips.window_tips() is not None, timeout=3000)
    assert context.shortcuts() is not None and context.shortcuts().window is window
    assert tooltips.window_tips() is not None
    assert window.lab.btn_start.toolTip()
    assert not _overlays(window) and tour.tour_state(window.lab.settings) == ""


def test_app_wide_filters_wait_for_the_first_show(qtbot: Any, isolated: Path) -> None:
    """앱 전체 필터(F1, 따로 뜨는 창 풍선)는 주 창이 처음 보인 뒤에 단다(M3: 첫 보이기의 이벤트가 파이썬 필터를 거치지 않게).
    만들기만 한 창은 달지 않는다. 주 창 안 풍선 도움말은 만들 때 바로 붙는다."""
    from evolab.app.help import context, tooltips
    from evolab.app.main import MainWindow

    assert context.shortcuts() is None and tooltips.window_tips() is None
    w = MainWindow()
    qtbot.addWidget(w)
    qtbot.wait(50)
    assert context.shortcuts() is None and tooltips.window_tips() is None
    assert w.lab.btn_start.toolTip() == S.HELP_TIPS["lab.start"]
    w.show()
    qtbot.waitUntil(lambda: context.shortcuts() is not None and tooltips.window_tips() is not None, timeout=3000)
    assert context.shortcuts().window is w


def test_app_wide_filters_wait_for_the_first_paint(qtbot: Any, isolated: Path) -> None:
    """M3 고침: 앱 전체 필터는 보인 차례가 아니라 창이 처음 그려진 다음 차례에 단다. 보일 때 건 0ms 타이머는 같은 이벤트
    처리 차례의 첫 그리기보다 먼저 돌아 첫 그리기 이벤트가 모두 파이썬 필터를 거쳤다."""
    from PySide6.QtCore import QEvent, QObject

    from evolab.app.help import context, tooltips
    from evolab.app.main import MainWindow

    seen: list[tuple[bool, bool]] = []

    class FirstPaint(QObject):
        def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802
            if event.type() == QEvent.Type.Paint and not seen:
                seen.append((context.shortcuts() is None, tooltips.window_tips() is None))
            return False

    w = MainWindow()
    qtbot.addWidget(w)
    probe = FirstPaint(w)
    w.installEventFilter(probe)
    w.show()
    qtbot.waitUntil(lambda: bool(seen), timeout=3000)
    assert seen[0] == (True, True)  # 첫 그리기 때는 아직 없다
    qtbot.waitUntil(lambda: context.shortcuts() is not None and tooltips.window_tips() is not None, timeout=3000)
    assert context.shortcuts().window is w


def test_app_wide_filters_fall_back_when_the_window_is_not_painted(qtbot: Any, isolated: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """보였다는 알림 뒤 그려지지 않아도(최소화한 채 시작 등) FILTERS_FALLBACK_MS 뒤에는 단다."""
    from PySide6.QtGui import QShowEvent

    from evolab.app.help import context, tooltips
    from evolab.app.main import MainWindow

    monkeypatch.setattr(help_setup, "FILTERS_FALLBACK_MS", 50)
    w = MainWindow()
    qtbot.addWidget(w)
    QApplication.sendEvent(w, QShowEvent())  # 창은 숨긴 채(그리기 없음) 보였다는 알림만
    assert context.shortcuts() is None
    qtbot.waitUntil(lambda: context.shortcuts() is not None and tooltips.window_tips() is not None, timeout=3000)
    assert context.shortcuts().window is w and not w.isVisible()


def test_welcome_card_started_before_the_first_paint_takes_keys_after_it(qtbot: Any, isolated: Path) -> None:
    """run_gui처럼 창이 그려지기 전에 안내를 띄우면 코치의 앱 전체 필터(Esc·Enter)는 오버레이가 처음 그려진 다음 차례에
    단다. 그 뒤 Esc는 그대로 건너뛰기다(설계 D13). 이미 그려진 창에서 띄우면 바로 단다(기존 테스트들)."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from evolab.app.main import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1280, 800)
    w.show()
    ov = tour.start_tour(w, w.lab.settings)  # 같은 차례: 아직 Expose 전
    assert ov is not None and ov.current is not None and ov.current.key == "welcome"
    assert ov._app_filter_pending  # noqa: SLF001
    qtbot.waitUntil(lambda: not ov._app_filter_pending, timeout=3000)  # noqa: SLF001
    assert ov.paint_ms > 0.0  # 오버레이가 그려진 뒤
    QTest.keyClick(w, Qt.Key.Key_Escape)
    assert ov.result == "skipped"
    ov2 = tour.start_tour(w, w.lab.settings)
    assert ov2 is not None and not ov2._app_filter_pending  # noqa: SLF001
    ov2.finish("later")


def test_run_gui_loads_the_report_modules_only_when_a_crash_is_reported(tmp_path: Path) -> None:
    """M3 고침(설계 D62): run_gui는 오류 보고 모듈(report.dialog·bundle)을 시작할 때 불러오지 않는다. 잡히지 않은 예외 알림은
    main.on_crash가 받아 그때 불러온다."""
    code = (
        "import sys\n"
        "import PySide6.QtWidgets as W\n"
        "class App(W.QApplication):\n"
        "    def exec(self):\n"
        "        for _ in range(5):\n"
        "            self.processEvents()\n"
        "        from evolab import diagnostics\n"
        "        from evolab.app import main\n"
        "        print('ON_CRASH', diagnostics._state.on_crash is main.on_crash)\n"
        "        print('REPORT', sorted(m for m in sys.modules if m.startswith('evolab.app.report')))\n"
        "        for w in self.topLevelWidgets():\n"
        "            w.close()\n"
        "        return 0\n"
        "W.QApplication = App\n"
        "from evolab.app.main import run_gui\n"
        "sys.exit(run_gui())\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env.update(QT_QPA_PLATFORM="offscreen", EVOLAB_DATA_DIR=str(tmp_path / "data"),
               EVOLAB_STATE_DIR=str(tmp_path / "state"), EVOLAB_NO_TOUR="1",
               PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    import subprocess

    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=240)
    assert done.returncode == 0, done.stderr[-3000:]
    lines = dict(line.split(" ", 1) for line in done.stdout.splitlines() if line.startswith(("ON_CRASH", "REPORT")))
    assert lines == {"ON_CRASH": "True", "REPORT": "[]"}


def test_main_on_crash_forwards_to_the_report_dialog(monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app import main as main_mod
    from evolab.app.report import dialog

    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(dialog, "on_crash", lambda action, line: seen.append((action, line)))
    main_mod.on_crash("box", "ValueError: x")
    assert seen == [("box", "ValueError: x")]


# ---------------------------------------------------------------------------
# T28 자체 점검의 새 점검 넷
# ---------------------------------------------------------------------------
def test_selftest_check_lists() -> None:
    names = [n for n, _ in selftest.selftest_checks(2, check_gui=True)]
    assert names[-5:] == [S.SELFTEST_CHECK_HELP, S.SELFTEST_CHECK_LOGS, S.SELFTEST_CHECK_GUI, S.SELFTEST_CHECK_QT_KO,
                          S.SELFTEST_CHECK_TOUR]
    assert set(NEW_CHECKS) <= set(names) and len(names) == 11
    no_gui = [n for n, _ in selftest.selftest_checks(2, check_gui=False)]
    assert no_gui == names[:8] and S.SELFTEST_CHECK_LOGS in no_gui and S.SELFTEST_CHECK_HELP in no_gui
    full = [n for n, _ in selftest.selftest_checks(2, check_gui=True, tour="full")]
    assert full[-1] == S.SELFTEST_CHECK_TOUR_FULL.format(goal=tour.TOUR_GOAL)
    with pytest.raises(ValueError):
        selftest.selftest_checks(2, tour="gen10")


def test_cli_selftest_tour_option() -> None:
    from evolab.cli import _build_parser

    parser = _build_parser()
    assert parser.parse_args(["selftest"]).tour == "gen1"
    assert parser.parse_args(["selftest", "--tour", "full", "--no-gui"]).tour == "full"
    with pytest.raises(SystemExit):
        parser.parse_args(["selftest", "--tour", "gen5"])


def _files(folder: Path) -> list[str]:
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*")) if folder.exists() else []


def test_new_selftest_checks_pass_and_touch_only_temporary_folders(qtbot: Any, tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """T28: 새 점검 넷이 오프스크린에서 통과하고 초를 적는다. 데이터·상태 폴더는 임시 폴더로 바꿔 쓰고 끝나면 지운다: 사용자의
    (여기서는 가짜) 데이터·상태 폴더에 남는 파일이 없다. EVOLAB_SELFTEST_CRASH=1이면 묶음 점검이 자식 세그폴트 기록도 본다."""
    real_data, real_state, tmp_root = tmp_path / "문서 데이터", tmp_path / "설정", tmp_path / "임시"
    tmp_root.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(real_data))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(real_state))
    monkeypatch.setenv(diagnostics.SELFTEST_CRASH_ENV, "1")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_root))
    checks = dict(selftest.selftest_checks(2, check_gui=True))
    log = tmp_path / "점검 기록" / "selftest.log"
    monkeypatch.setattr(selftest, "selftest_checks", lambda *a, **k: [(n, checks[n]) for n in NEW_CHECKS])
    code = selftest.run_selftest(log_path=log, workers=2, check_gui=True)
    report = log.read_text(encoding="utf-8")
    assert code == 0, report
    for name in NEW_CHECKS:
        assert f"[{S.SELFTEST_PASS}] {name} (" in report
    assert S.SELFTEST_HELP_DETAIL.format(pages=len(S.HELP_PAGES)) in report
    assert "crash child: exit=" in report and "marker=yes function=yes" in report
    tour_line = next(line.strip() for line in report.splitlines() if line.strip().startswith("창 → 1세대 재생"))
    assert tour_line.endswith("누름 2번")
    seconds = float(tour_line.split("재생 ")[1].split("초")[0])
    assert 0.0 < seconds <= selftest.TOUR_LIMIT_S["gen1"]
    assert _files(real_data) == [] and _files(real_state) == []
    if sys.platform != "win32":  # 윈도우는 자식 프로세스의 파일 핸들이 늦게 풀려 임시 폴더가 남을 수 있다(사용자 폴더는 위에서 봄)
        assert _files(tmp_root) == []
    assert os.environ[paths.DATA_DIR_ENV] == str(real_data)


def test_tour_check_reports_slow_and_failed_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """기준을 넘으면 실패(초와 기준을 적음), 안내가 끝까지 못 가면 실패, 안 끝나면 시간 초과."""
    base = {"result": "done", "clicks": 2, "first": 43.9, "goal": 228.8, "workers": 2, "gen1": 3.5, "goal_s": 70.0,
            "log": ["welcome"]}
    monkeypatch.setattr(selftest, "run_tour_check", lambda until: {**base, "seconds": 75.0 if until == "gen1" else 70.0})
    with pytest.raises(selftest.SelfTestError) as slow:
        selftest._check_tour("gen1")
    assert S.SELFTEST_ERR_TOUR_SLOW.format(what=S.SELFTEST_TOUR_WHAT_GEN1, seconds=75.0, limit=60) in str(slow.value)
    detail = selftest._check_tour("full")
    assert S.SELFTEST_TOUR_FULL_DETAIL.format(goal=tour.TOUR_GOAL, seconds=70.0, first=43.9, now=228.8) in detail
    monkeypatch.setattr(selftest, "run_tour_check", lambda until: {**base, "result": "failed", "seconds": None})
    with pytest.raises(selftest.SelfTestError, match="result=failed"):
        selftest._check_tour("gen1")
    monkeypatch.setattr(selftest, "run_tour_check", lambda until: {**base, "result": None, "seconds": None})
    with pytest.raises(selftest.SelfTestError) as timeout:
        selftest._check_tour("gen1")
    assert str(timeout.value) == S.SELFTEST_ERR_TIMEOUT.format(seconds=int(selftest.TOUR_WAIT_S["gen1"]))


# ---------------------------------------------------------------------------
# 느린: T30 안내 설정이 「배우는 모습」을 보여 주는지
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_tour_settings_learn_visibly_in_ten_generations(tmp_path: Path) -> None:
    """T30(설계 8.1): 안내 설정(내장 언덕차, 새 실행 기본값: CMA-ES·개체 32·시드 1·시험 수 = 환경 파일 값)으로 화면 없이 10세대:
    10세대 평균 ≥ 1세대 평균 × 2.5(측정 43.9 → 228.8 = 5.2배), 10세대 멈춤 비율 ≤ 0.2(측정 0.031). 학습 결과는 워커 수와 무관하다."""
    import csv

    from evolab.engine.pool import WorkerPool
    from evolab.engine.runsetup import create_run
    from evolab.engine.trainer import TrainConfig, Trainer
    from evolab.spec.io import compile_env

    path = tour.tour_env_path()
    assert path is not None and path.stem == tour.TOUR_ENV
    text = path.read_text(encoding="utf-8-sig")
    spec = compile_env(text)
    config = TrainConfig.for_new_run(spec, max_generations=tour.TOUR_GOAL)
    assert (config.method, config.run_seed, config.popsize, config.trials) == ("cmaes", 1, 32, None)
    store = create_run(tmp_path / "runs", text, spec, config, 2)
    with WorkerPool(2) as pool:
        Trainer(text, config, pool, store).run()
    with (store.root / "stats.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == tour.TOUR_GOAL
    first, last = float(rows[0]["score_mean"]), float(rows[-1]["score_mean"])
    stall = float(rows[-1]["end_stall"])
    print(f"tour settings: gen1 mean {first:.1f} -> gen{tour.TOUR_GOAL} mean {last:.1f} ({last / first:.2f}x), "
          f"stall {float(rows[0]['end_stall']):.3f} -> {stall:.3f}")
    assert first > 0 and last >= first * 2.5
    assert stall <= 0.2

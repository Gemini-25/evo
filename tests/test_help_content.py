"""v3.0 B 묶음 도움말 쪽과 도움말 창 (v3 설계 4.4, D18~D20, D23, 8.1 T11·T12·T14).

- T11 쪽: 12쪽 모두 열림, 채운 글에 자리가 남지 않음, 「」 규칙, 링크는 정해진 형식만이고 도움말 창에서 모두 따라갈 수 있음, 쪽 그리기 ≤ 50ms.
- T12 빠짐: 화면마다 쪽, 진짜 대시보드 타일 제목·진짜 에디터 아래 탭·내장 예제·낱말이 모두 쪽에 있음(strings.py만 보는 검사는
  test_v3_strings.py).
- T14 도움말 창: 앞뒤·처음, 찾기 결과 수와 앞뒤 글, Enter 강조, `evolab://show/…` → 그 탭 + 짚기, `evolab://tour`는 쉴 때만,
  인터넷 주소는 열지 않음, 글자 크기·마지막 쪽·창 크기 저장.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QSettings, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QScrollArea, QTextBrowser

from evolab import paths
from evolab.app import strings as S
from evolab.app.help import anchors, content, context
from evolab.app.help import window as HW

PAGE_IDS = ("start", "lab", "dashboard", "editor", "preset", "play", "compare", "race", "records", "examples", "trouble",
            "words", "install")
"""설계 D19의 쪽(v3.0판 「설치와 제거」 포함 12쪽) + 6.2.0의 「새 프리셋 만들기」(스포츠 설계 4.5: 13쪽)."""

RENDER_LIMIT_MS = 50.0
"""쪽 하나 그리기 상한(설계 T11: 최종 글 중앙값 1.81ms를 잰 값의 여유 있는 위)."""


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """실제 데이터·설정 폴더를 건드리지 않는다(설계 D5). 처음 실행 안내도 끈다."""
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv("EVOLAB_NO_TOUR", "1")
    monkeypatch.setattr(HW, "hooks", {})
    yield tmp_path
    win = HW.help_window()
    if win is not None:
        win.close()
        win.deleteLater()
    HW._instance = None


@pytest.fixture()
def opened_urls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """바깥 프로그램(탐색기·브라우저)을 실제로 열지 않고 연 주소를 모은다."""
    urls: list[str] = []
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda url: urls.append(QUrl(url).toString()) or True))
    return urls


@pytest.fixture()
def points(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, str, Any]]:
    """짚기 말풍선(A 묶음 coach.point 또는 풍선 글) 대신 부른 것을 모은다."""
    calls: list[tuple[Any, str, Any]] = []
    monkeypatch.setattr(HW, "point_at", lambda widget, text, more=None: calls.append((widget, text, more)))
    return calls


@pytest.fixture()
def main_window(qtbot: Any, isolated: Path) -> Any:
    from evolab.app.main import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1280, 720)
    window.show()
    return window


def _help(page: str = "start", heading: str | None = None, parent: Any = None) -> HW.HelpWindow:
    win = HW.open_help(page, heading, parent)
    assert isinstance(win, HW.HelpWindow)
    return win


# ---------------------------------------------------------------------------
# T11 쪽
# ---------------------------------------------------------------------------
def test_pages_follow_strings_order() -> None:
    assert content.PAGES == tuple(S.HELP_PAGES) == PAGE_IDS
    assert content.HOME in content.PAGES and content.GLOSSARY in content.PAGES
    for page in content.PAGES:
        assert content.title(page) == S.HELP_PAGES[page][0]
        assert content.headings(page)[0] == content.title(page)  # 첫 제목 = 쪽 제목


def test_every_page_fills_all_placeholders() -> None:
    for page in content.PAGES:
        text = content.load_page(page)
        assert "{{" not in text and "}}" not in text, page
        assert content.unknown_placeholders(page) == [], page
        assert content.load_page(page) is text  # 처음 한 번 채워 기억
    # 자리를 채우는 규칙: 상수, 사전 키, 모르는 자리는 그대로
    assert content.fill("{{LAB_START}}") == S.LAB_START
    assert content.fill("{{DASH_TILE_TITLES.trend}}") == S.DASH_TILE_TITLES["trend"]
    for unknown in ("{{NO_SUCH_NAME}}", "{{DASH_TILE_TITLES.no_such_tile}}", "{{LAB_START.key}}", "{{HELP_PAGES}}"):
        assert content.fill(unknown) == unknown


def test_corner_quotes_hold_only_placeholders() -> None:
    """「…」 안은 우리 화면 이름 자리만(설계 D19): 원문에서 「 다음은 언제나 {{, 」 앞은 언제나 }}."""
    for page, (_title, md) in S.HELP_PAGES.items():
        for inner in re.findall(r"「([^」]*)」", md):
            assert content.PLACEHOLDER.fullmatch(inner), f"{page}: 「{inner}」"
        assert md.count("「") == md.count("」"), page


def test_links_use_known_forms_and_targets() -> None:
    simple = {f"evolab://{kind}" for kind in HW.SIMPLE_ACTIONS}
    seen = set()
    for page in content.PAGES:
        for url in content.links(page):
            seen.add(url.split("/")[2] if url.startswith("evolab://") else url)
            if url in simple:
                continue
            if url.startswith("evolab://help/"):
                target, _, heading = url.removeprefix("evolab://help/").partition("#")
                assert content.has_page(target), f"{page}: {url}"
                assert not heading or heading in content.headings(target), f"{page}: {url}"
            elif url.startswith("evolab://show/"):
                assert url.removeprefix("evolab://show/") in anchors.ANCHORS, f"{page}: {url}"
            else:
                pytest.fail(f"{page}: 정해지지 않은 링크 {url}")
    assert {"help", "show", "tour", "report", "data", "logs"} <= seen  # 여섯 가지 링크를 모두 씀


def test_every_link_in_every_page_can_be_followed(main_window: Any, points: list[Any], opened_urls: list[str]) -> None:
    """쪽에 있는 링크를 도움말 창에서 차례로 눌러도 막힘(HELPW_LINK_BLOCKED)이나 못 찾음이 없다."""
    calls: list[str] = []
    for kind in ("tour", "report"):
        HW.hooks[kind] = lambda _owner, k=kind: calls.append(k)
    urls = [(page, url) for page in content.PAGES for url in content.links(page)]
    assert len(urls) >= 15  # 지금 글: 16개
    for page, url in urls:
        win = _help(page, parent=main_window)  # 「처음 실행 안내」 링크는 도움말 창을 닫으므로 다시 연다
        before = len(win.status_texts)
        assert win.follow_link(QUrl(url)), f"{page}: {url} {win.status_texts[before:]}"
        assert win.status_texts[before:] == [], f"{page}: {url}"
    assert "tour" in calls and "report" in calls
    assert points  # show 링크가 위젯을 짚음
    assert len(opened_urls) >= 2  # 데이터 폴더, 오류 기록 폴더


def test_page_render_time(qtbot: Any) -> None:
    browser = QTextBrowser()
    qtbot.addWidget(browser)
    browser.resize(800, 700)
    worst = 0.0
    for page in content.PAGES:
        times = []
        for _ in range(3):
            t = time.perf_counter()
            browser.setMarkdown(content.load_page(page))
            browser.document().size()
            browser.viewport().grab()
            times.append((time.perf_counter() - t) * 1000)
        worst = max(worst, statistics.median(times))
    assert worst <= RENDER_LIMIT_MS, worst


def test_plain_text_and_search() -> None:
    for page in content.PAGES:
        plain = content.plain_text(page)
        assert "**" not in plain and "evolab://" not in plain and "\n" not in plain, page
        assert not plain.startswith("#"), page
    results = content.search("시드")
    assert results, results
    counts = [c for _p, c, _s in results]
    assert counts == sorted(counts, reverse=True)
    for page, count, snippet in results:
        assert content.plain_text(page).casefold().count("시드") == count
        assert "시드" in snippet
        assert len(snippet) <= len("시드") + 2 * content.SNIPPET
    assert content.total_matches(results) == sum(counts)
    # 대소문자 무시, 앞뒤 빈칸·여러 빈칸 무시, 링크 글은 찾지만 링크 주소는 찾지 않음
    assert content.search("cma-es") == content.search("CMA-ES") != []
    assert content.search("  재현   묶음 ") == content.search("재현 묶음") != []
    assert content.search("evolab://") == []
    assert content.search("") == [] and content.search("   ") == []
    assert content.search("없는낱말xyz") == []


# ---------------------------------------------------------------------------
# T12 빠짐 (진짜 화면에서)
# ---------------------------------------------------------------------------
def test_every_screen_has_a_page() -> None:
    assert set(anchors.SCREENS) == set(context.SCREEN_PAGES)
    assert all(content.has_page(p) for p in context.SCREEN_PAGES.values())
    heading = getattr(S, context.SCREEN_HEADINGS["export"])
    assert heading in content.headings(context.SCREEN_PAGES["export"])  # 내보내기 → 「기록과 내보내기」 쪽 「영상」 절


def test_dashboard_page_covers_real_tiles(qtbot: Any, isolated: Path) -> None:
    from evolab.app.lab.dashboard import ALL_TILE_IDS, Dashboard

    dash = Dashboard(settings=QSettings(str(isolated / "dash.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(dash)
    heads = content.headings("dashboard")
    titles = [dash.tile(tid).title.text() for tid in ALL_TILE_IDS]
    assert [t for t in titles if t not in heads] == []
    assert len(titles) >= 12  # v2.1 「몸 구조」까지


def test_editor_page_covers_real_bottom_tabs(qtbot: Any, isolated: Path) -> None:
    from evolab.app.editor.view import EditorView

    editor = EditorView()
    qtbot.addWidget(editor)
    tabs = editor.tabs
    heads = content.headings("editor")
    found = [context.editor_tab_heading(tabs, tabs.widget(i)) for i in range(tabs.count())]
    assert None not in found
    assert [h for h in found if h not in heads] == []
    assert len(set(found)) == tabs.count() >= 10  # v2.1 「구조」 탭까지
    assert S.ED_TAB_STRUCTURE in found


def test_examples_and_words_pages() -> None:
    names = [json.loads(p.read_text(encoding="utf-8-sig"))["name"] for p in paths.builtin_envs()]
    assert [n for n in names if n not in content.headings("examples")] == []
    bold = set(re.findall(r"\*\*([^*]+)\*\*", content.load_page("words")))
    assert [t for t in S.HELP_GLOSSARY_TERMS if t not in bold] == []


# ---------------------------------------------------------------------------
# T14 도움말 창
# ---------------------------------------------------------------------------
def test_open_help_reuses_one_window(main_window: Any) -> None:
    win = _help("lab", parent=main_window)
    assert win.isVisible() and win.windowTitle() == S.HELPW_TITLE
    assert win.parentWidget() is main_window  # 주 창을 닫으면 함께 닫히고 마지막 창 판단에 들지 않음
    assert win.isWindow()
    assert _help("records", parent=main_window.lab.btn_start) is win
    assert win.page == "records"
    assert win.page_list.currentItem().text() == content.title("records")
    assert HW.help_window() is win


def test_unknown_page_opens_home_with_notice(isolated: Path, qtbot: Any) -> None:
    win = _help("no_such_page")
    assert win.page == content.HOME
    assert win.status_texts[-1] == S.HELPW_PAGE_MISSING.format(page="no_such_page")
    assert not win.follow_link("evolab://help/nope")
    assert win.status_texts[-1] == S.HELPW_PAGE_MISSING.format(page="nope")


def test_heading_links_scroll_to_the_heading(isolated: Path) -> None:
    win = _help("start")
    win.resize(800, 500)
    for page in content.PAGES:
        for heading in content.headings(page)[1:]:
            assert win.follow_link(QUrl(f"evolab://help/{page}#{heading}")), (page, heading)
            assert win.page == page and win.heading == heading
            assert win.current_heading_block() == heading, (page, heading)
    # 긴 쪽 끝 제목으로 가면 스크롤이 내려가 있다
    win.show_page("dashboard", S.DASH_TILE_TITLES["structure"])
    assert win.browser.verticalScrollBar().value() > 0
    assert win.show_page("dashboard", "없는 제목") and win.heading is None


def test_back_forward_and_home(isolated: Path) -> None:
    win = _help("start")
    assert not win.btn_back.isEnabled() and not win.btn_forward.isEnabled()
    win.show_page("lab")
    win.page_list.setCurrentRow(content.PAGES.index("dashboard"))  # 쪽 목록에서 고름
    assert win.page == "dashboard"
    win.btn_back.click()
    assert win.page == "lab"
    win.btn_back.click()
    assert win.page == "start" and not win.btn_back.isEnabled()
    win.btn_forward.click()
    assert win.page == "lab" and win.btn_forward.isEnabled()
    win.show_page("words")  # 새 쪽을 열면 앞으로 기록은 버림
    assert not win.btn_forward.isEnabled()
    win.btn_home.click()
    assert win.page == content.HOME
    win.go_back()
    assert win.page == "words"


def test_search_results_and_enter_highlight(isolated: Path, qtbot: Any) -> None:
    win = _help("start")
    win.search_box.setText("시드")  # 한글은 QTest.keyClicks로 칠 수 없다(글자 → 키 변환이 없음)
    results = content.search("시드")
    assert win.left_stack.currentIndex() == 1
    assert win.result_label.text() == S.HELPW_RESULTS.format(n=content.total_matches(results))
    assert win.result_list.count() == len(results)
    page, count, snippet = results[0]
    assert win.result_list.item(0).text() == S.HELPW_RESULT_ITEM.format(page=content.title(page), count=count,
                                                                       snippet=snippet)
    QTest.keyClick(win.search_box, Qt.Key.Key_Return)  # Enter: 첫 결과 쪽을 열고 첫 맞은 곳을 고름
    assert win.page == page
    cursor = win.browser.textCursor()
    assert cursor.selectedText() == "시드"
    first = cursor.selectionStart()
    QTest.keyClick(win.search_box, Qt.Key.Key_Return)  # 다시 Enter: 같은 쪽의 다음 곳
    assert win.page == page and win.browser.textCursor().selectionStart() > first
    # 결과 목록에서 다른 쪽을 고르면 그 쪽의 첫 맞은 곳
    other = results[1][0]
    win.result_list.itemClicked.emit(win.result_list.item(1))
    assert win.page == other and win.browser.textCursor().selectedText() == "시드"
    # 맞는 곳이 없음
    win.search_box.setText("없는낱말xyz")
    assert win.result_list.count() == 0
    no = S.HELPW_NO_RESULTS.format(q="없는낱말xyz", glossary=content.title("words"))
    assert win.result_label.text() == no
    QTest.keyClick(win.search_box, Qt.Key.Key_Return)
    assert win.status_texts[-1] == no
    win.search_box.clear()
    assert win.left_stack.currentIndex() == 0


def test_show_link_switches_tab_and_points(main_window: Any, points: list[Any]) -> None:
    from evolab.app.lab.dashboard import PAGE_OF

    w = main_window
    w.tabs.setCurrentWidget(w.editor)
    win = _help("lab", parent=w)
    assert win.follow_link(QUrl("evolab://show/lab.start"))
    assert w.tabs.currentWidget() is w.lab
    widget, text, _more = points[-1]
    assert widget is w.lab.btn_start and text == S.HELP_TIPS["lab.start"]
    # 스크롤 영역 아래에 있던 칸도 보이게 스크롤한 뒤 짚는다
    w.tabs.setCurrentWidget(w.records)
    assert win.follow_link("evolab://show/lab.speed")
    assert points[-1][0] is w.lab.speed_combo and not w.lab.speed_combo.visibleRegion().isEmpty()
    # 검토 수정: 「보기」 묶음이 패널 맨 위로 와서 배속 칸은 늘 보인다. 스크롤 0에서 가려지는 패널 아래쪽 칸으로 스크롤도 본다.
    w.tabs.setCurrentWidget(w.lab)
    target = w.lab.target
    area = target.parentWidget()
    while not isinstance(area, QScrollArea):
        area = area.parentWidget()
    area.verticalScrollBar().setValue(0)
    QApplication.processEvents()
    assert target.isVisible() and target.visibleRegion().isEmpty()  # 스크롤 0: 가려짐
    w.tabs.setCurrentWidget(w.records)
    assert win.follow_link("evolab://show/lab.target")
    assert w.tabs.currentWidget() is w.lab
    assert points[-1][0] is target and not target.visibleRegion().isEmpty()
    # 다른 쪽 대시보드 타일: 그 쪽으로 바꾼 뒤 짚는다
    fun_tile = next(t for t, p in PAGE_OF.items() if p != w.lab.dashboard.page)
    assert win.follow_link(f"evolab://show/lab.tile.{fun_tile}")
    assert w.lab.dashboard.page == PAGE_OF[fun_tile]
    tile = w.lab.dashboard.tile(fun_tile)
    assert points[-1][0] is tile
    assert points[-1][1] == (tile.toolTip() or tile.title.toolTip() or S.DASH_TILE_TITLES[fun_tile])
    # 열려 있지 않은 창의 단추: 짚지 않고 알림
    n = len(points)
    assert not win.follow_link("evolab://show/play.restart")
    assert len(points) == n
    assert win.status_texts[-1] == S.HELPW_SHOW_MISSING.format(what=S.PLAY_RESTART)
    assert not win.follow_link("evolab://show/no.such.key")


def test_tour_link_only_when_lab_is_idle(main_window: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app import help as help_pkg

    client_class = type(main_window.lab.client)

    started: list[tuple[Any, Any]] = []
    fake = types.ModuleType("evolab.app.help.tour")
    fake.start_tour = lambda window, settings, **kw: started.append((window, settings))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "evolab.app.help.tour", fake)
    monkeypatch.setattr(help_pkg, "tour", fake, raising=False)

    monkeypatch.setattr(client_class, "active", property(lambda self: True))
    win = _help("start", parent=main_window)
    assert not win.follow_link("evolab://tour")
    assert win.status_texts[-1] == S.HELPW_TOUR_BUSY and started == [] and win.isVisible()
    win.btn_tour.click()
    assert started == [] and win.status_texts[-1] == S.HELPW_TOUR_BUSY

    monkeypatch.setattr(client_class, "active", property(lambda self: False))
    assert win.follow_link("evolab://tour")
    assert len(started) == 1 and started[0][0] is main_window and isinstance(started[0][1], QSettings)
    assert Path(started[0][1].fileName()).parent == paths.user_state_dir()
    assert not win.isVisible()  # 코치 말풍선을 가리지 않게 닫음
    win = _help("start", parent=main_window)
    win.btn_tour.click()
    assert len(started) == 2


def test_outside_links_are_never_opened(isolated: Path, opened_urls: list[str]) -> None:
    win = _help("start")
    for url in ("http://example.com", "https://example.com/x", "file:///etc/passwd", "mailto:a@b.c", "evolab://nope",
                "evolab://tour/extra", "evolab:help/start"):
        assert not win.follow_link(QUrl(url)), url
        assert win.status_texts[-1] == S.HELPW_LINK_BLOCKED.format(url=QUrl(url).toString()), url
    win.browser.anchorClicked.emit(QUrl("https://example.com"))  # 글 속 링크를 누른 것과 같음
    assert win.status_texts[-1].endswith("https://example.com")
    assert opened_urls == []
    assert win.browser.openLinks() is False and win.browser.openExternalLinks() is False


def test_report_data_logs_links(main_window: Any, opened_urls: list[str], isolated: Path) -> None:
    owners: list[Any] = []
    HW.hooks["report"] = owners.append
    win = _help("trouble", parent=main_window)
    assert win.follow_link("evolab://report")
    assert owners == [main_window]
    assert win.follow_link("evolab://data") and win.follow_link("evolab://logs")
    assert opened_urls == [QUrl.fromLocalFile(str(paths.data_dir())).toString(),
                           QUrl.fromLocalFile(str(paths.log_dir())).toString()]
    assert paths.data_dir().is_relative_to(isolated) and paths.log_dir().is_relative_to(isolated)


def test_font_size_last_page_and_geometry_are_saved(isolated: Path) -> None:
    win = _help("compare")
    base = win.browser.font().pointSizeF() - win.font_delta
    win.btn_bigger.click()
    win.btn_bigger.click()
    assert win.font_delta == 2 and win.browser.font().pointSizeF() == pytest.approx(base + 2)
    for _ in range(20):
        win.btn_smaller.click()  # 끝에 닿으면 단추가 꺼진다
    assert win.font_delta == HW.FONT_DELTA_MIN and not win.btn_smaller.isEnabled()
    win.set_font_delta(100)
    assert win.font_delta == HW.FONT_DELTA_MAX and not win.btn_bigger.isEnabled()
    win.set_font_delta(3)
    win.resize(700, 520)
    win.close()
    settings = QSettings(str(paths.user_state_dir() / HW.SETTINGS_FILE), QSettings.Format.IniFormat)
    assert int(settings.value(HW.KEY_FONT)) == 3
    assert settings.value(HW.KEY_PAGE) == "compare"
    assert settings.value(HW.KEY_GEOMETRY)
    win.deleteLater()
    HW._instance = None
    again = HW.browse_help()  # 「도움말 찾아보기…」: 마지막 쪽, 찾기 칸에 초점
    assert isinstance(again, HW.HelpWindow) and again is not win
    assert again.page == "compare" and again.font_delta == 3
    assert (again.width(), again.height()) == (700, 520)
    assert paths.user_state_dir().is_relative_to(isolated)  # 실제 설정 폴더가 아님


def test_help_window_follows_modal_dialog(main_window: Any, qtbot: Any) -> None:
    """모달 대화 상자(반복 실행 등)가 떠 있으면 도움말 창이 막히지 않게 그 상자를 주인으로 삼는다."""
    win = _help("compare", parent=main_window)
    dlg = QDialog(main_window)
    qtbot.addWidget(dlg)
    dlg.setModal(True)
    dlg.show()
    assert HW._owner_for(dlg) is dlg
    again = _help("compare", parent=dlg)
    assert again is win and win.parentWidget() is dlg and win.isVisible() and win.isWindow()
    dlg.close()
    again = _help("lab", parent=main_window)
    assert again is win and win.parentWidget() is main_window


def test_every_page_opens_in_the_window(isolated: Path) -> None:
    win = _help("start")
    for page in content.PAGES:
        assert win.show_page(page) and win.page == page
        assert win.browser.document().characterCount() > 100
        assert win.browser.document().firstBlock().text() == content.title(page)
        assert win.page_list.currentItem().text() == content.title(page)


# ---------------------------------------------------------------------------
# 안내(A 묶음)와 이어서
# ---------------------------------------------------------------------------
def test_show_link_uses_coach_point(main_window: Any) -> None:
    win = _help("lab", parent=main_window)
    widget = HW.show_anchor("lab.start", win)
    assert widget is main_window.lab.btn_start
    overlay = HW.point_at(widget, S.HELP_TIPS["lab.start"], ("lab", None))  # 짚기 말풍선 = 진짜 창 위 오버레이
    assert overlay is not None and overlay.parentWidget() is main_window  # type: ignore[attr-defined]


def test_tour_link_starts_the_real_tour_when_idle(main_window: Any) -> None:
    from evolab.app.help import tour

    win = _help("start", parent=main_window)
    assert win.follow_link("evolab://tour")
    assert tour.running_tour(main_window) is not None
    assert not win.isVisible()

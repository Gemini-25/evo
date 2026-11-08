"""v3.0 0단계 문자열 (v3 설계 8.1 T2, 6절, 2.5): 새 상수, format 자리, 도움말 쪽의 자리·「」 규칙·링크·빠짐.

도움말 쪽 그리기·찾기(T11), 진짜 에디터 아래 탭과의 빠짐(T12)은 B 묶음 테스트가 다시 본다. 여기서는 strings.py만으로 볼 수 있는
것(자리가 모두 풀림, 「」 안은 자리만, 링크 형식, 타일·탭·내장 예제·낱말 빠짐)을 본다.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import string
from pathlib import Path
from typing import Any

import pytest

from evolab import paths
from evolab.app import strings as S
from evolab.app.help.anchors import ANCHORS

STRINGS_FILE = Path(S.__file__)

V30_NAMES: tuple[str, ...] = (
    "ABOUT_TEXT_V3", "ONB_WELCOME_TITLE", "ONB_WELCOME_BODY", "ONB_WELCOME_START", "ONB_WELCOME_LATER",
    "ONB_LATER_HINT", "ONB_RESUME_TITLE", "ONB_RESUME_BODY", "ONB_RESUME_CONTINUE", "ONB_RESUME_RESTART",
    "ONB_RESUME_STOP", "ONB_STEP_PRESS_START", "ONB_STEP_WAIT_PROCESS", "ONB_STEP_WAIT_GEN1", "ONB_STEP_WAIT_SLOW",
    "ONB_STEP_WAIT_STUCK", "ONB_STUCK_WAIT", "ONB_STUCK_HELP", "ONB_STEP_WATCH", "ONB_SPEED_SET",
    "ONB_STEP_PROGRESS", "ONB_PROGRESS_LINE", "ONB_PROGRESS_FLAT", "ONB_STEP_VERSUS", "ONB_STEP_DONE", "ONB_NEXT",
    "ONB_SKIP", "ONB_SKIP_STEP", "ONB_FINISH", "ONB_CLOSE", "ONB_PROGRESS", "ONB_BACK_TO_LAB", "ONB_FAILED",
    "ONB_BUSY", "ONB_NO_EXAMPLE", "ONB_EXISTING_USER_HINT", "MENU_HELP_SCREEN", "MENU_HELP_BROWSE",
    "MENU_HELP_WHATS_THIS", "MENU_HELP_TOUR", "MENU_REPORT", "MENU_OPEN_LOGS", "HELP_CORNER_BUTTON",
    "HELP_CORNER_TIP", "HELPW_TITLE", "HELPW_SEARCH_PLACEHOLDER", "HELPW_BACK", "HELPW_FORWARD", "HELPW_HOME",
    "HELPW_FONT_BIGGER", "HELPW_FONT_SMALLER", "HELPW_RESULTS", "HELPW_RESULT_ITEM", "HELPW_NO_RESULTS",
    "HELPW_PAGE_MISSING", "HELPW_SHOW_MISSING", "HELPW_TOUR_BUSY", "HELPW_LINK_BLOCKED", "WHATS_THIS_HINT",
    "WHATS_THIS_NONE", "WHATS_THIS_MORE", "HELP_GLOSSARY_TERMS", "HELP_PAGES", "HELP_TIPS", "HINT_LAB_IDLE_TITLE",
    "HINT_LAB_IDLE", "HINT_LAB_TOUR", "HINT_DASH_EMPTY", "HINT_DASH_ONE", "HINT_EDITOR_EMPTY", "ERR_NEXT_LABEL",
    "ERR_DETAILS", "ERR_REPORT_BUTTON", "ERR_PERMISSION", "ERR_PERMISSION_NEXT", "ERR_NOT_FOUND",
    "ERR_NOT_FOUND_NEXT", "ERR_DISK_FULL", "ERR_DISK_FULL_NEXT", "ERR_MEMORY", "ERR_MEMORY_NEXT",
    "ERR_ENV_FILE_NEXT", "ERR_BUILD_NEXT", "ERR_RUN_STORE_NEXT", "ERR_RECORD_NEXT", "ERR_UNEXPECTED",
    "ERR_UNEXPECTED_NEXT", "DIAG_CRASH_TITLE", "DIAG_CRASH_TEXT", "DIAG_CRASH_AGAIN", "DIAG_PREVIOUS_TITLE",
    "DIAG_PREVIOUS_TEXT", "DIAG_PREVIOUS_YES", "DIAG_PREVIOUS_NO", "DIAG_STARTUP_TITLE", "DIAG_STARTUP_TEXT",
    "SESSION_LOST_MEANING", "EXIT_CODE_TEXT", "EXIT_CODE_SIGNAL", "EXIT_CODE_UNKNOWN", "REP_TITLE", "REP_INTRO",
    "REP_NOTE_LABEL", "REP_NOTE_PLACEHOLDER", "REP_ITEMS_LABEL", "REP_ITEM_NAMES", "REP_PRIVACY", "REP_FILES",
    "REP_SAVE", "REP_CANCEL", "REP_SAVE_TITLE", "REP_FILTER", "REP_DONE", "REP_OPEN_FOLDER", "REP_FAILED",
    "REP_FILE_NAME", "REP_USER_MASK", "REP_HOST_MASK", "REP_TXT_TITLE", "REP_TXT_FIELDS", "REP_REASONS", "REP_YES",
    "REP_NO", "CLI_HELP_REPORT", "CLI_HELP_REPORT_OUT", "CLI_HELP_REPORT_RUN", "CLI_HELP_REPORT_OPEN",
    "CLI_REPORT_DONE", "CLI_UNCAUGHT", "SELFTEST_CHECK_TOUR", "SELFTEST_TOUR_DETAIL", "SELFTEST_TOUR_FULL_DETAIL",
    "SELFTEST_ERR_TOUR_SLOW", "SELFTEST_CHECK_HELP", "SELFTEST_HELP_DETAIL", "SELFTEST_CHECK_QT_KO",
    "SELFTEST_ERR_QT_KO", "SELFTEST_CHECK_LOGS", "CLI_HELP_SELFTEST_TOUR",
)
"""v3 설계 6절의 v3.0 새 상수(ABOUT_TEXT는 ABOUT_TEXT_V3로: strings.py v3.0 머리말 참고)."""

V31_NAMES = ("INST_PUBLISHER", "PORTABLE_README", "EXP_FORMAT", "EXP_INFO_MF", "PHONE_TEXT", "PHONE_TOO_BIG",
             "CLI_HELP_EXPORT", "CLI_HELP_VIDEO_PROBE", "SELFTEST_CHECK_EXPORT")
"""v3.1 0단계가 더할 상수(설치, 내보내기, 폰 재생 파일, 명령줄 내보내기): v3.0에는 아직 없다."""

HELP_PAGE_IDS = ("start", "lab", "dashboard", "editor", "play", "compare", "race", "records", "examples", "trouble",
                 "words", "install")
"""설계 D19의 쪽 순서(v3.0판 「설치와 제거」 포함 12쪽)."""

SCREEN_PAGES = {"editor": "editor", "lab": "lab", "compare": "compare", "records": "records", "play": "play",
                "race": "race", "repeat": "compare", "report": "trouble", "export": "records", "main": "start"}
"""설계 4.5: 화면 → 도움말 쪽."""

PLACE = re.compile(r"\{\{([A-Z0-9_]+)(?:\.([a-z0-9_]+))?\}\}")
LINK = re.compile(r"\]\(([^)\s]+)\)")
SIMPLE_LINKS = {"evolab://tour", "evolab://report", "evolab://data", "evolab://logs"}


def _fill(value: str) -> str:
    """{{NAME}}·{{NAME.key}} 자리를 strings 값으로 채운다(content.load_page와 같은 규칙: 설계 2.5). 모르는 자리는 그대로 둔다."""

    def sub(m: re.Match[str]) -> str:
        name, key = m.group(1), m.group(2)
        value = getattr(S, name, None)
        if key is not None:
            return str(value[key]) if isinstance(value, dict) and key in value else m.group(0)
        return str(value) if isinstance(value, str) else m.group(0)

    return PLACE.sub(sub, value)


def _headings(markdown: str) -> list[str]:
    return [ln.lstrip("#").strip() for ln in markdown.splitlines() if ln.startswith("#")]


def _resolved() -> dict[str, str]:
    return {pid: _fill(md) for pid, (_title, md) in S.HELP_PAGES.items()}


# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
def test_v30_constants_exist_and_v31_ones_do_not_yet() -> None:
    missing = [n for n in V30_NAMES if not hasattr(S, n)]
    assert not missing, missing
    early = [n for n in V31_NAMES if hasattr(S, n)]
    assert not early, early


def test_no_name_is_assigned_twice_in_strings_py() -> None:
    """같은 이름을 strings.py 끝에 다시 쓰면 앞의 값이 조용히 바뀐다. 값을 바꿀 때는 제자리에서 고친다(0단계)."""
    tree = ast.parse(STRINGS_FILE.read_text(encoding="utf-8"))
    seen: dict[str, int] = {}
    twice: list[str] = []
    for node in tree.body:
        targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets if isinstance(node, ast.Assign) else []
        for t in targets:
            if isinstance(t, ast.Name):
                if t.id in seen:
                    twice.append(f"{t.id}: {seen[t.id]}, {node.lineno}")
                seen[t.id] = node.lineno
    assert not twice, twice


def test_changed_constants() -> None:
    assert S.LAB_START == "▶ 학습 시작"  # 설계 D17: 「정체하면 새로 시작」과 헷갈리던 「새로 시작」
    fields = {f for _, f, _, _ in string.Formatter().parse(S.ABOUT_TEXT_V3) if f}
    assert fields == {"name", "version", "data_dir", "state_dir", "notices"}
    assert "문서\\진화실험실" not in S.ABOUT_TEXT_V3  # 데이터 폴더를 글자로 박지 않음(설정에서 바꿀 수 있음)


def _strings_of(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [x for v in value.values() for x in _strings_of(v)]
    if isinstance(value, (tuple, list)):
        return [x for v in value for x in _strings_of(v)]
    return []


def test_every_format_placeholder_fills_with_dummy_values() -> None:
    errors: list[str] = []
    for name in V30_NAMES:
        if name == "HELP_PAGES":  # 도움말 쪽은 format이 아니라 {{자리}}로 채운다
            continue
        for text in _strings_of(getattr(S, name)):
            fields = [f for _, f, _, _ in string.Formatter().parse(text) if f]
            try:
                text.format(**{f.split(".")[0].split("[")[0]: 1.0 for f in fields})
            except (KeyError, ValueError, IndexError) as exc:
                errors.append(f"{name}: {exc!r}")
    assert not errors, errors


def test_exit_code_meanings_are_windows_ntstatus_codes() -> None:
    assert all(isinstance(k, int) and k > 0xC0000000 for k in S.EXIT_CODE_TEXT)
    assert S.EXIT_CODE_TEXT[0xC0000005] == "메모리 접근 오류"


def test_help_tips_keys_are_anchors() -> None:
    assert set(S.HELP_TIPS) <= set(ANCHORS), sorted(set(S.HELP_TIPS) - set(ANCHORS))
    assert all(isinstance(v, str) and v.strip() for v in S.HELP_TIPS.values())


def test_help_pages_keys_match_content_pages() -> None:
    """HELP_PAGES 키 = content.PAGES(B 묶음). content.py가 생기기 전에는 건너뛴다."""
    if importlib.util.find_spec("evolab.app.help.content") is None:
        pytest.skip("app/help/content.py는 B 묶음이 만든다")
    from evolab.app.help import content

    assert tuple(S.HELP_PAGES) == tuple(content.PAGES)


# ---------------------------------------------------------------------------
# 도움말 쪽 (설계 2.5, D19, D20)
# ---------------------------------------------------------------------------
def test_help_pages_order_and_shape() -> None:
    assert tuple(S.HELP_PAGES) == HELP_PAGE_IDS
    assert set(SCREEN_PAGES.values()) <= set(S.HELP_PAGES)
    for pid, value in S.HELP_PAGES.items():
        assert isinstance(value, tuple) and len(value) == 2, pid
        title, md = value
        assert md.startswith(f"# {title}\n"), pid  # 쪽 제목이 첫 줄 제목
        assert "|" not in md and "![" not in md, pid  # 표·그림을 쓰지 않음


def test_help_placeholders_all_resolve() -> None:
    unresolved = {pid: PLACE.findall(text) for pid, text in _resolved().items() if PLACE.search(text)}
    assert not unresolved, unresolved
    assert not any("{{" in t or "}}" in t for t in _resolved().values())


def test_corner_quotes_hold_only_placeholders() -> None:
    """「…」 안에는 우리 화면 이름 자리만(바깥 이름은 “…”), 자리 뒤에 조사를 붙이지 않는다(설계 D19)."""
    bad = {pid: [q for q in re.findall(r"「([^」]*)」", md) if not PLACE.fullmatch(q)]
           for pid, (_t, md) in S.HELP_PAGES.items()}
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}


def test_help_links_use_only_known_forms() -> None:
    resolved = _resolved()
    heads = {pid: _headings(text) for pid, text in resolved.items()}
    wrong: list[str] = []
    for pid, text in resolved.items():
        for url in LINK.findall(text):
            if url in SIMPLE_LINKS:
                continue
            if url.startswith("evolab://help/"):
                page, _, heading = url.removeprefix("evolab://help/").partition("#")
                if page not in S.HELP_PAGES or (heading and heading not in heads[page]):
                    wrong.append(f"{pid}: {url}")
            elif url.startswith("evolab://show/"):
                if url.removeprefix("evolab://show/") not in ANCHORS:
                    wrong.append(f"{pid}: {url}")
            else:
                wrong.append(f"{pid}: {url}")
    assert not wrong, wrong


def test_dashboard_page_has_every_tile() -> None:
    from evolab.app.lab.dashboard import ALL_TILE_IDS

    heads = set(_headings(_resolved()["dashboard"]))
    assert set(S.DASH_TILE_TITLES) == set(ALL_TILE_IDS)
    assert [tid for tid in ALL_TILE_IDS if S.DASH_TILE_TITLES[tid] not in heads] == []
    assert len(ALL_TILE_IDS) >= 12  # v2.1 「몸 구조」까지


def test_editor_page_has_every_bottom_tab(qtbot: Any) -> None:
    from evolab.app.editor.document import EnvDocument
    from evolab.app.editor.tabs import BottomTabs

    tabs = BottomTabs(EnvDocument())
    qtbot.addWidget(tabs)
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    issues = tabs.indexOf(tabs.issues_tab)
    titles[issues] = S.EDP_TAB_ISSUES  # 「검사 결과 (3)」처럼 개수가 붙은 글은 탭 이름으로 본다
    heads = set(_headings(_resolved()["editor"]))
    assert [t for t in titles if t not in heads] == []
    assert S.ED_TAB_STRUCTURE in titles  # v2.1 「구조」 탭


def test_examples_page_has_every_builtin_env() -> None:
    names = [json.loads(p.read_text(encoding="utf-8-sig"))["name"] for p in paths.builtin_envs()]
    heads = set(_headings(_resolved()["examples"]))
    assert [n for n in names if n not in heads] == []
    assert len(names) >= 6  # 착륙선, 트랙, 언덕차, 점프 게임, 글라이더, 바퀴 고르는 언덕차


def test_words_page_defines_every_glossary_term() -> None:
    bold = set(re.findall(r"\*\*([^*]+)\*\*", _resolved()["words"]))
    assert [t for t in S.HELP_GLOSSARY_TERMS if t not in bold] == []
    assert len(set(S.HELP_GLOSSARY_TERMS)) == len(S.HELP_GLOSSARY_TERMS)


def test_v30_pages_describe_v30_features_only() -> None:
    """v3.0판: 영상은 GIF만, 설치는 zip(설치 프로그램·MP4·폰 재생 파일은 v3.1에서 쪽 글을 바꾼다)."""
    resolved = _resolved()
    assert "MP4" not in resolved["records"] and ".html" not in resolved["records"]
    assert "zip" in resolved["install"] and "설치 파일" not in resolved["install"]
    assert "MP4" not in S.HELP_TIPS["lab.video"]

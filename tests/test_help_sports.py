"""6.2.0 도움말·자체 점검 (묶음 Ha): 최종 설계 6.1의 T37(연결표와 마지막 조각 빼고), T38, T40(자체 점검 쪽).

- T37 도움말 13쪽(「새 프리셋 만들기」는 「에디터」 바로 뒤), 새 화면(「새 프리셋」 창·틀 보관함)과 새 탭(「규칙」)의 F1 쪽·절,
  바뀐 절(에디터의 「규칙」「문제 목록」「세계 종류」), 낱말 풀이 다섯, 찾기에 새 쪽이 걸림.
  마지막 조각(W·ED1·ED2가 합쳐진 뒤): 새 연결표 키(`editor.*`·`preset.*`·`lab.new_preset`)가 그 위젯으로 풀리고 `HELP_TIPS` ⊆ 연결표,
  「이건 뭐예요?」(Shift+F1)와 `evolab://show/` 짚기가 새 위젯에서 된다(새 화면의 풍선 감사는 test_tooltips_sports.py).
- T38 `app/preset/`·`presets/*.py`·새 에디터 파일(`fixes.py`·`rules_tab.py`)에 한글 글자 상수 0(문서 문자열 빼고). 내장 틀·조리법의
  한글은 환경 내용이 되는 칸에만(틀: 부품의 `label`, 조리법: 환경 `name`·`description`. D55).
- T40 자체 점검 「프리셋」: 점검 목록에 있고(창 없는 점검, 「도움말 쪽」 다음), 내장 조리법 전부와 달리기 한 세대로 통과하고,
  내장 틀 하나를 깨뜨리면 그 파일 이름으로 실패한다.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from evolab import paths, selftest
from evolab.app import strings as S
from evolab.app.help import anchors, content, context

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "evolab"
HANGUL = re.compile(r"[가-힣ㄱ-ㆎ]")


# ---------------------------------------------------------------------------
# T37: 쪽과 절
# ---------------------------------------------------------------------------
def test_thirteen_pages_with_the_preset_page_after_the_editor() -> None:
    assert len(content.PAGES) == 13
    assert content.PAGES.index("preset") == content.PAGES.index("editor") + 1
    assert content.title("preset") == S.PRESET_TITLE
    heads = content.headings("preset")
    assert heads[0] == S.PRESET_TITLE
    assert len(heads) >= 6  # 다섯 단계, 아래 줄의 단추, 몸 틀과 슬라이더, 만든 뒤에 고치기, 저장되는 곳, 잘 안 배울 때
    for page in content.PAGES:
        assert not content.unknown_placeholders(page), page
    text = content.load_page("preset")
    for name in ("ED_NEW_PRESET", "ED_TO_LAB", "PRESET_BTN_PLAY", "PRESET_BTN_EDIT"):
        assert getattr(S, name) in text, name  # 단추 이름은 자리로 채워져 화면 글과 같다
    assert S.LAB_ENV_GROUPS["mine"] in text and S.TPL_PARAM_LABELS["balance"] in text


def test_changed_sections_of_the_editor_page_and_the_new_words() -> None:
    heads = set(content.headings("editor"))
    assert {S.EDP_TAB_RULES, S.EDP_TAB_ISSUES, S.EDP_WORLD_KIND} <= heads
    assert "evolab://help/preset" in content.links("editor")
    words = content.load_page(content.GLOSSARY)
    for term in ("프리셋", "몸 틀", "규칙", "위에서 본 세계", "균형 보조"):
        assert term in S.HELP_GLOSSARY_TERMS and f"**{term}**" in words, term
    # 내장 목록의 이름마다 「내장 예제」 쪽에 머리가 있다(달리기 포함, D47)
    from evolab.spec.io import load_env

    example_heads = set(content.headings("examples"))
    for name in paths.BUILTIN_ORDER:
        path = paths.builtin_env_dir() / f"{name}.evoenv"
        if path.exists():
            assert load_env(path).file.name in example_heads, name


def test_search_finds_the_new_page() -> None:
    pages = [page for page, _n, _snip in content.search("슬라이더")]
    assert "preset" in pages
    pages = [page for page, _n, _snip in content.search(S.PRESET_TITLE)]
    assert pages and "preset" in pages


def test_screens_and_pages_stay_in_step() -> None:
    assert set(anchors.SCREENS) == set(context.SCREEN_PAGES)
    assert len(anchors.SCREENS) == len(set(anchors.SCREENS))
    assert anchors.WINDOW_SCREENS["PresetWizard"] == anchors.WINDOW_SCREENS["TemplatePicker"] == "preset"
    assert context.SCREEN_PAGES["preset"] == "preset"
    assert ("rules_tab", "EDP_TAB_RULES") in context.EDITOR_TABS


# ---------------------------------------------------------------------------
# T37: F1과 화면 (창)
# ---------------------------------------------------------------------------
@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from evolab.app.help import tooltips
    from evolab.app.help import window as HW

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
    app = QApplication.instance()
    if isinstance(app, QApplication):
        context.uninstall_shortcuts(app)
        tooltips.uninstall(app)


def _wizard(qtbot: Any, monkeypatch: pytest.MonkeyPatch, parent: QWidget | None = None) -> Any:
    from evolab.app.preset.wizard import PresetWizard

    monkeypatch.setattr(PresetWizard, "ask_discard", lambda self: True)  # 닫을 때 「버릴까요?」 상자를 띄우지 않게
    wizard = PresetWizard(parent)
    qtbot.addWidget(wizard)
    return wizard


def test_f1_in_the_preset_wizard_and_the_template_picker(qtbot: Any, isolated: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.help import window as HW
    from evolab.app.preset.cards import TemplatePicker

    wizard = _wizard(qtbot, monkeypatch)
    assert anchors.screen_of(wizard) == "preset"
    assert context.context_page(wizard) == ("preset", None)
    for name in ("btn_edit", "btn_play", "btn_lab", "btn_reset_step"):  # 3.1의 위젯 이름(안쪽 위젯에서 올라가도)
        assert context.context_page(getattr(wizard, name)) == ("preset", None), name
    picker = TemplatePicker("body", "side", {}, wizard)
    qtbot.addWidget(picker)
    assert context.context_page(picker) == ("preset", None)
    # F1 필터가 그 쪽을 연다
    flt = context.HelpShortcuts(None)
    opened = flt.help_for(wizard.btn_lab)
    assert flt.opened == [("preset", None)]
    assert isinstance(opened, HW.HelpWindow) and opened.page == "preset"


def test_f1_on_the_rules_tab_and_the_wizard_from_the_main_window(qtbot: Any, isolated: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtTest import QTest

    from evolab.app.main import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    QTest.qWaitForWindowExposed(window)
    tabs = window.editor.tabs
    assert context.context_page(tabs.rules_tab) == ("editor", S.EDP_TAB_RULES)
    assert context.context_page(tabs.rules_tab.add_button) == ("editor", S.EDP_TAB_RULES)
    # 주 창에 묶인 모달 창도 제 화면(탭 화면이 아니라)
    wizard = _wizard(qtbot, monkeypatch, window)
    assert anchors.screen_of(wizard.btn_play) == "preset"
    assert context.context_page(wizard.btn_play) == ("preset", None)


# ---------------------------------------------------------------------------
# T38: 한글 글자 상수와 자산
# ---------------------------------------------------------------------------
def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for stmt in body:  # 모듈·클래스·함수 첫 줄과 필드 아래 설명: 값으로 쓰이지 않는 문자열 문장
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                ids.add(id(stmt.value))
    return ids


def _hangul_literals(path: Path) -> list[str]:
    """글자 그대로 적힌 한글 문자열 상수(조사 고르기의 「\uc73c」 같은 이스케이프는 코드용이라 셈하지 않는다: test_rules와 같은 규칙)."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    skip = _docstring_ids(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            text = ast.get_source_segment(source, node) or ""
            if HANGUL.search(text):
                out.append(f"{path.relative_to(ROOT)}:{node.lineno}: {text[:40]!r}")
    return out


def t38_files() -> list[Path]:
    files = sorted((SRC / "app" / "preset").glob("*.py")) + sorted((SRC / "presets").glob("*.py"))
    files += [p for p in (SRC / "app" / "editor" / "fixes.py", SRC / "app" / "editor" / "rules_tab.py") if p.exists()]
    return files


def test_no_hangul_literals_in_the_new_code() -> None:
    files = t38_files()
    assert any(p.parent.name == "preset" for p in files) and any(p.parent.name == "presets" for p in files)
    offenders = [line for path in files for line in _hangul_literals(path)]
    assert not offenders, "\n".join(offenders)


def _hangul_paths(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _hangul_paths(item, (*path, str(key)))
    elif isinstance(value, list):
        for item in value:
            yield from _hangul_paths(item, path)
    elif isinstance(value, str) and HANGUL.search(value):
        yield path


def test_asset_hangul_only_in_fields_that_become_environment_content() -> None:
    """D55: 내장 틀의 화면 글(이름·슬라이더·단위)은 strings.py의 id로. 한글은 펼친 뒤 환경 파일에 들어가는 부품 이름표뿐.
    내장 조리법의 한글은 만든 환경의 이름과 설명뿐."""
    templates = sorted((SRC / "assets" / "templates").glob("*.evotpl"))
    recipes = sorted((SRC / "assets" / "recipes").glob("*.json"))
    assert templates and recipes
    bad: list[str] = []
    for path in templates:
        for where in _hangul_paths(json.loads(path.read_text(encoding="utf-8"))):
            if not (len(where) == 3 and where[0] == "parts" and where[2] == "label"):
                bad.append(f"{path.name}: {'.'.join(where)}")
    for path in recipes:
        for where in _hangul_paths(json.loads(path.read_text(encoding="utf-8"))):
            if where not in (("name",), ("description",)):
                bad.append(f"{path.name}: {'.'.join(where)}")
    assert not bad, bad


# ---------------------------------------------------------------------------
# T40: 자체 점검 「프리셋」
# ---------------------------------------------------------------------------
def test_presets_check_is_registered_without_a_window() -> None:
    for gui in (False, True):
        names = [n for n, _fn in selftest.selftest_checks(2, check_gui=gui)]
        k = names.index(S.SELFTEST_CHECK_PRESETS)
        assert names[k - 1] == S.SELFTEST_CHECK_HELP
    assert dict(selftest.selftest_checks(2, check_gui=False))[S.SELFTEST_CHECK_PRESETS] is selftest._check_presets


def test_presets_check_passes_on_the_builtin_files() -> None:
    from evolab.presets.recipe import builtin_names

    r = selftest._run_check(S.SELFTEST_CHECK_PRESETS, selftest._check_presets)
    assert r.ok, r.detail
    assert r.detail.startswith(S.SELFTEST_PRESETS_DETAIL.split("{", 1)[0])
    assert f"{len(builtin_names())}" in r.detail


def test_presets_check_names_a_broken_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "templates"
    shutil.copytree(paths.template_dirs()[0], folder)
    broken = folder / "wall.evotpl"
    broken.write_text(broken.read_text(encoding="utf-8").replace('"parts"', '"pieces"', 1), encoding="utf-8")
    monkeypatch.setattr(paths, "template_dirs", lambda: [folder])
    r = selftest._run_check(S.SELFTEST_CHECK_PRESETS, selftest._check_presets)
    assert not r.ok
    assert "wall.evotpl" in r.detail and r.detail.startswith(S.SELFTEST_ERR_PRESET.split("{", 1)[0])


# ---------------------------------------------------------------------------
# 주 창의 「새 프리셋」 길(app/main.py): 실험실 단추 → 에디터가 띄우는 창, 「에디터에서 다듬기」 → 에디터 탭, 학습 중이면 보내지 않음
# ---------------------------------------------------------------------------
def test_main_window_opens_the_wizard_from_the_lab_button(qtbot: Any, isolated: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtTest import QTest

    from evolab.app.main import MainWindow
    from evolab.app.preset.wizard import PresetWizard

    monkeypatch.setattr(PresetWizard, "ask_discard", lambda self: True)
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    QTest.qWaitForWindowExposed(window)
    assert window.editor.lab_ready is not None and window.editor.lab_ready()  # 학습 프로세스 없음
    button = window.lab.btn_new_preset
    window.tabs.setCurrentWidget(window.lab)
    button.click()
    wizard = window.editor.wizard
    assert isinstance(wizard, PresetWizard) and wizard.isVisible() and wizard.isModal()
    assert QWidget.window(wizard.parentWidget()) is window
    text = (paths.builtin_env_dir() / "jump.evoenv").read_text(encoding="utf-8")
    wizard.finished_with.emit(text, "edit", None)
    assert window.tabs.currentWidget() is window.editor
    assert window.editor.doc.data["name"] == json.loads(text)["name"]
    wizard.close()


# ---------------------------------------------------------------------------
# T37 마지막 조각: 연결표와 HELP_TIPS, Shift+F1, evolab://show/
# ---------------------------------------------------------------------------
NEW_TIPS = {
    "editor.new_preset": "ED_NEW_PRESET_TIP", "editor.rules": "RULES_ADD_TIP", "editor.issues": "EDP_ISSUES_HINT",
    "editor.owner_agent": "EDC_OWNER_AGENT_TIP", "editor.owner_world": "EDC_OWNER_WORLD_TIP",
    "editor.world_kind": "EDP_WORLD_KIND_TIP", "preset.steps": "PRESET_STEPS_TIP", "preset.preview": "PRESET_PREVIEW_TIP",
    "preset.name": "PRESET_NAME_TIP", "preset.edit": "PRESET_BTN_EDIT_TIP", "preset.play": "PRESET_BTN_PLAY_TIP",
    "preset.lab": "PRESET_BTN_LAB_TIP", "lab.new_preset": "ED_NEW_PRESET_TIP",
}
"""장부의 「HELP_TIPS.update()」(6.2.0) 열세 키와 값."""


def test_new_help_tips_are_in_the_link_table() -> None:
    for key, name in NEW_TIPS.items():
        assert S.HELP_TIPS[key] == getattr(S, name), key
        assert key in anchors.ANCHORS, key
    assert set(S.HELP_TIPS) <= set(anchors.ANCHORS)
    assert {anchors.ANCHORS[k].screen for k in NEW_TIPS if k.startswith("preset.")} == {"preset"}


def _main(qtbot: Any) -> Any:
    from PySide6.QtTest import QTest

    from evolab.app.main import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1280, 720)
    window.show()
    QTest.qWaitForWindowExposed(window)
    return window


def test_editor_and_lab_keys_resolve_to_the_named_widgets(qtbot: Any, isolated: Path) -> None:
    window = _main(qtbot)
    ed = window.editor
    expected = {
        "editor.new_preset": ed.toolbar.widgetForAction(ed.act_new_preset),
        "editor.rules": ed.tabs.rules_tab.add_button, "editor.issues": ed.tabs.issues_tab,
        "editor.world_kind": ed.tabs.world_tab.kind_row, "editor.owner_agent": ed.palette.owner_agent,
        "editor.owner_world": ed.palette.owner_world, "editor.world_motion": ed.palette.world_motion,
        "editor.go_bar": ed.go_bar, "lab.new_preset": window.lab.btn_new_preset,
    }
    for key, widget in expected.items():
        assert widget is not None and anchors.resolve(window, key) is widget, key
        assert anchors.screen_of(widget) == anchors.ANCHORS[key].screen, key
    # 빈 캔버스 안내 카드에도 같은 글의 단추가 있지만 연결표는 도구 줄 단추를 고른다
    assert anchors.resolve(window, "editor.new_preset").parentWidget() is ed.toolbar
    # Shift+F1(「이건 뭐예요?」)의 글과 F1의 쪽·절
    for key in ("editor.owner_agent", "editor.owner_world", "editor.rules", "lab.new_preset"):
        widget = expected[key]
        target, text = context.whats_this_text(widget)
        assert target is widget and text == S.HELP_TIPS[key], key
    assert context.context_page(ed.palette.owner_world) == ("editor", None)
    assert context.context_page(ed.palette.buttons["tpl_body"]) == ("editor", None)
    assert context.context_page(ed.tabs.world_tab.kind_row) == ("editor", S.EDP_TAB_WORLD)
    assert context.context_page(ed.tabs.issues_tab) == ("editor", S.EDP_TAB_ISSUES)
    assert context.context_page(window.lab.btn_new_preset) == ("lab", None)


def test_show_links_reach_widgets_inside_editor_tabs(qtbot: Any, isolated: Path) -> None:
    from evolab.app.help import window as HW

    window = _main(qtbot)
    tabs = window.editor.tabs
    window.tabs.setCurrentWidget(window.lab)
    assert tabs.currentWidget() is not tabs.rules_tab  # 처음 보이는 탭은 「세계」
    shown = HW.show_anchor("editor.rules", window)
    assert shown is tabs.rules_tab.add_button
    assert window.tabs.currentWidget() is window.editor and tabs.currentWidget() is tabs.rules_tab
    assert HW.show_anchor("editor.issues", window) is tabs.issues_tab and tabs.currentWidget() is tabs.issues_tab
    assert HW.show_anchor("editor.world_kind", window) is tabs.world_tab.kind_row
    assert HW.show_anchor("lab.new_preset", window) is window.lab.btn_new_preset
    assert window.tabs.currentWidget() is window.lab


def test_preset_keys_resolve_in_the_wizard_only(qtbot: Any, isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app import style
    from evolab.app.help import tooltips
    from evolab.app.preset.cards import TemplatePicker

    window = _main(qtbot)
    wizard = _wizard(qtbot, monkeypatch, window)
    wizard.open()
    qtbot.waitExposed(wizard)
    attrs = {"preset.steps": "rail", "preset.preview": "preview", "preset.name": "name_edit", "preset.edit": "btn_edit",
             "preset.play": "btn_play", "preset.lab": "btn_lab", "preset.reset_step": "btn_reset_step"}
    assert set(attrs) == set(anchors.keys_for("preset"))
    for key, attr in attrs.items():
        widget = getattr(wizard, attr)
        assert widget.objectName() == attr  # 3.1: objectName = 속성 이름
        assert anchors.resolve(wizard, key) is widget, key
        assert anchors.resolve(window, key) is widget, key  # 주 창에서 찾아도 열린 창의 것
        assert anchors.resolve(wizard.status, key) is widget, key  # 창 안 아무 위젯에서도
    # 풍선이 빈 위젯만 HELP_TIPS로 채운다
    tooltips.apply_tooltips(wizard, "preset")  # (앱 필터가 없는 테스트: 첫 보임에 하던 일)
    assert all(anchors.resolve(wizard, key).toolTip() for key in NEW_TIPS if key.startswith("preset."))
    wizard.btn_play.setToolTip("")
    assert tooltips.apply_tooltips(wizard, "preset") == 1
    assert wizard.btn_play.toolTip() == S.HELP_TIPS["preset.play"]
    # Shift+F1: 꺼진 「실험실로 보내기」면 까닭도(U9)
    target, text = context.whats_this_text(wizard.btn_lab)
    assert target is wizard.btn_lab and text == style.tip_text(wizard.btn_lab)
    assert text.startswith(S.HELP_TIPS["preset.lab"])
    assert context.whats_this(wizard.rail) is not None
    # 틀 보관함 창도 화면 "preset"이지만 이 속성이 없다
    picker = TemplatePicker("body", "side", {}, wizard)
    qtbot.addWidget(picker)
    picker.show()
    qtbot.waitExposed(picker)
    assert anchors.resolve(picker, "preset.lab") is None
    picker.close()
    wizard.close()

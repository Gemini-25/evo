"""6.2.0 새 화면의 풍선 (묶음 Ha, 최종 설계 T37의 풍선 쪽, DU T37).

- 「새 프리셋」 창(다섯 단계 모두), 틀 보관함 창(몸·물건), 에디터 「규칙」 탭에 **이름표도 풍선도 없는 조작 위젯 0**
  (기준은 v3.0 풍선 감사 `tests/test_tooltips_audit.py`의 것을 그대로 씀).
- 앱 필터(`WindowTips`)가 「새 프리셋」 창을 「preset」 화면으로 알아보고 처음 보일 때 한 번 붙인다(연결표 키는 W·ED1·ED2가
  합쳐진 뒤의 단계가 더한다: 지금은 0개여도 된다). 그 창의 긴 풍선은 420px로 꺾인다(U3의 필터 그대로).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QEvent, QPoint
from PySide6.QtGui import QHelpEvent
from PySide6.QtWidgets import QApplication, QToolTip, QWidget

from evolab import paths
from evolab.app import style
from evolab.app.help import anchors
from evolab.app.help import tooltips as T
from tests.test_tooltips_audit import audit  # 같은 감사 규칙(조작 위젯 · 풍선 · 이름표)


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from evolab.app.preset.wizard import PresetWizard

    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv("EVOLAB_NO_TOUR", "1")
    monkeypatch.setattr(PresetWizard, "ask_discard", lambda self: True)
    yield tmp_path
    app = QApplication.instance()
    if isinstance(app, QApplication):
        T.uninstall(app)


def _wizard(qtbot: Any, **kw: Any) -> Any:
    from evolab.app.preset.wizard import PresetWizard

    w = PresetWizard(**kw)
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    return w


def _steps(wizard: Any) -> int:
    from evolab.app import strings as S

    return len(S.PRESET_STEPS)


def test_no_bare_controls_in_the_preset_wizard(qtbot: Any, isolated: Path) -> None:
    wizard = _wizard(qtbot)
    bare: list[str] = []
    for step in range(_steps(wizard)):
        wizard.go(step)
        qtbot.wait(10)
        r = audit(wizard, f"preset step {step}")
        assert r.controls > 0
        bare += [f"step {step}: {b}" for b in r.bare]
    assert bare == []


def test_no_bare_controls_in_the_template_picker(qtbot: Any, isolated: Path) -> None:
    from evolab.app.preset.cards import TemplatePicker

    for kind, world in (("body", "side"), ("body", "top"), ("prop", "side"), ("prop", "top")):
        picker = TemplatePicker(kind, world, {})
        qtbot.addWidget(picker)
        picker.show()
        qtbot.waitExposed(picker)
        if picker.cards:
            picker.cards[0].click()  # 고른 틀의 슬라이더가 오른쪽에 뜬다
            qtbot.wait(10)
        r = audit(picker, f"picker {kind}/{world}")
        assert r.bare == [], (kind, world, r.bare)
        picker.close()


def test_no_bare_controls_in_the_rules_tab(qtbot: Any, isolated: Path) -> None:
    from evolab.app.editor.document import EnvDocument
    from evolab.app.editor.rules_tab import RulesTab

    for name in ("running.evoenv", "lander.evoenv", "hill_car.evoenv"):
        doc = EnvDocument()
        doc.load_text((paths.builtin_env_dir() / name).read_text(encoding="utf-8"))
        tab = RulesTab(doc)
        qtbot.addWidget(tab)
        tab.show()
        qtbot.waitExposed(tab)
        r = audit(tab, f"rules {name}")
        assert r.controls > 0 and r.bare == [], (name, r.bare)


def test_window_tips_knows_the_preset_wizard_and_wraps_long_tips(qtbot: Any, isolated: Path) -> None:
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    flt = T.WindowTips(app)
    app.installEventFilter(flt)
    try:
        wizard = _wizard(qtbot)
        assert anchors.screen_of(wizard) == "preset"
        assert [s for s, _n in flt.applied] == ["preset"]  # 처음 보일 때 한 번
        wizard.hide()
        wizard.show()
        assert [s for s, _n in flt.applied] == ["preset"]
        target: QWidget = wizard.btn_lab
        long_tip = "가" * 200
        target.setToolTip(long_tip)
        event = QHelpEvent(QEvent.Type.ToolTip, QPoint(2, 2), target.mapToGlobal(QPoint(2, 2)))
        assert flt.show_tip(target, event)
        qtbot.waitUntil(QToolTip.isVisible, timeout=2000)
        assert QToolTip.text() == style.wrap_tip(long_tip)
        QToolTip.hideText()
    finally:
        app.removeEventFilter(flt)


# ---------------------------------------------------------------------------
# 연결표가 합쳐진 뒤(마지막 조각): 에디터의 새 부분(틀 단추, 소유 두 칸, 세계 물체 움직임, 세계 종류, 고치기·괜찮습니다 단추,
# 오른쪽 go_bar)과 실험실 「새 프리셋…」 단추. 글이 있는 단추도 **제 풍선**이 있어야 한다(새 칸마다 풍선: 하드 규칙 2)
# ---------------------------------------------------------------------------
AUDIT_DIR = Path(__file__).resolve().parent / "data" / "audit"


def _own_tip(widget: QWidget) -> bool:
    return bool(widget.toolTip())


def test_new_editor_parts_have_their_own_tips(qtbot: Any, isolated: Path) -> None:
    from PySide6.QtWidgets import QAbstractButton

    from evolab.app.editor.view import EditorView

    view = EditorView()
    qtbot.addWidget(view)
    view.resize(1280, 720)
    view.show()
    qtbot.waitExposed(view)
    pal = view.palette
    pal.set_owner_mode("world")  # 「움직임」 고르기가 보이게
    qtbot.wait(10)
    named = {"tpl_body": pal.buttons["tpl_body"], "tpl_prop": pal.buttons["tpl_prop"], "owner_agent": pal.owner_agent,
             "owner_world": pal.owner_world, "world_motion": pal.world_motion}
    assert pal.world_motion.isVisibleTo(view)
    kind_buttons = view.tabs.world_tab.kind_row.findChildren(QAbstractButton)
    assert kind_buttons
    named.update({f"world_kind[{k}]": b for k, b in enumerate(kind_buttons)})
    go = view.go_bar.findChildren(QAbstractButton)
    named.update({f"go_bar[{k}]": b for k, b in enumerate(go) if b.isVisibleTo(view) and b.text()})
    missing = [name for name, w in named.items() if not _own_tip(w)]
    assert missing == []
    r = audit(pal, "palette")
    assert r.controls > 0 and r.bare == [], r.bare
    view.tabs.setCurrentWidget(view.tabs.world_tab)
    assert audit(view.tabs.world_tab, "world tab").bare == []


@pytest.mark.parametrize("name", sorted(p.name for p in AUDIT_DIR.glob("*.evoenv")))
def test_fix_and_ignore_buttons_have_tips(qtbot: Any, isolated: Path, name: str) -> None:
    from evolab.app.editor.view import EditorView

    view = EditorView()
    qtbot.addWidget(view)
    view.resize(1280, 720)
    view.show()
    qtbot.waitExposed(view)
    assert view.load_text((AUDIT_DIR / name).read_text(encoding="utf-8"))
    code = name.removesuffix(".evoenv")
    issues = view.tabs.issues_tab
    qtbot.waitUntil(lambda: issues.row_of(code) >= 0, timeout=5000)
    view.tabs.setCurrentWidget(issues)
    qtbot.wait(10)
    seen = 0
    for row in range(len(issues.issues)):
        for button in (issues.fix_button(row), issues.ignore_button(row)):
            if button is not None:
                seen += 1
                assert _own_tip(button), (name, row, button.text())
    assert issues.fix_button(issues.row_of(code)) is not None and seen > 0
    r = audit(issues, f"issues {name}")
    assert r.controls > 0 and r.bare == [], (name, r.bare)


def test_lab_new_preset_button_has_its_tip(qtbot: Any, isolated: Path) -> None:
    from evolab.app import strings as S
    from evolab.app.lab.view import LabView

    lab = LabView()
    qtbot.addWidget(lab)
    assert lab.btn_new_preset.toolTip() == S.HELP_TIPS["lab.new_preset"]

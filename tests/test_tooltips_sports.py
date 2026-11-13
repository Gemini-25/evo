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

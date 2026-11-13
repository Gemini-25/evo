"""「새 프리셋」 창의 키보드 (6.2.0, 묶음 W, 최종 설계 T21·D42).

모든 단추 autoDefault 거짓(Enter가 단추를 누르지 않음), 이름 칸의 Enter는 이름만 확정, Alt+←/→ 이전·다음, Ctrl+Enter 「실험실로
보내기」, Tab이 켜진 모든 조작 위젯에 닿음, 키보드만으로 달리기 프리셋을 실험실로.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSlider,
    QWidget,
)

from evolab import paths
from evolab.app.preset import wizard as WZ
from evolab.app.preset.wizard import PresetWizard

CONTROLS = (QAbstractButton, QComboBox, QLineEdit, QAbstractSpinBox, QSlider, QListWidget)


@pytest.fixture()
def iso(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("EVOLAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EVOLAB_STATE_DIR", str(tmp_path / "state"))
    # 테스트 끝에 창을 닫을 때 「버릴까요?」 상자가 뜨지 않게(물음 자체는 T20이 따로 본다)
    monkeypatch.setattr(PresetWizard, "ask_discard", lambda self: True)
    return tmp_path


def make(qtbot: Any, **kw: Any) -> PresetWizard:
    w = PresetWizard(**kw)
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    w.activateWindow()
    return w


def env_files() -> list[Path]:
    folder = paths.user_env_dir()
    return sorted(folder.glob("*.evoenv")) if folder.is_dir() else []


def focus() -> QWidget | None:
    return QApplication.focusWidget()


def test_t21_no_button_is_auto_default(qtbot: Any, iso: Path) -> None:
    w = make(qtbot)
    for start in ("running", "drill"):
        w.choose_start(start)
        for step in range(WZ.STEP_COUNT):
            w.go(step)
            for b in w.findChildren(QPushButton):
                assert not b.autoDefault() and not b.isDefault(), b.text()
    # Enter는 초점이 있는 단추도 누르지 않는다(D42: Enter가 첫 단추를 눌러 저장·닫히던 사고)
    got: list[Any] = []
    w.finished_with.connect(lambda *a: got.append(a))
    w.btn_lab.setFocus()
    qtbot.keyClick(w.btn_lab, Qt.Key.Key_Return)
    qtbot.keyClick(w.btn_lab, Qt.Key.Key_Enter)
    assert w.isVisible() and not got and env_files() == []


def test_t21_enter_in_the_name_field_only_sets_the_name(qtbot: Any, iso: Path) -> None:
    w = make(qtbot)
    got: list[Any] = []
    w.finished_with.connect(lambda *a: got.append(a))
    w.name_edit.setFocus()
    w.name_edit.selectAll()
    qtbot.keyClicks(w.name_edit, "My run")
    qtbot.keyClick(w.name_edit, Qt.Key.Key_Return)
    assert w.isVisible() and not got and env_files() == []
    assert w.data["name"] == "My run" and w.recipe.name == "My run"
    assert focus() is w.name_edit


def test_t21_alt_arrows_and_ctrl_enter(qtbot: Any, iso: Path) -> None:
    w = make(qtbot)
    w.rail.setFocus()
    qtbot.keyClick(w.rail, Qt.Key.Key_Right, Qt.KeyboardModifier.AltModifier)
    assert w.step == 1
    qtbot.keyClick(focus() or w, Qt.Key.Key_Right, Qt.KeyboardModifier.AltModifier)
    qtbot.keyClick(focus() or w, Qt.Key.Key_Right, Qt.KeyboardModifier.AltModifier)
    assert w.step == 3
    qtbot.keyClick(focus() or w, Qt.Key.Key_Left, Qt.KeyboardModifier.AltModifier)
    assert w.step == 2
    for _ in range(6):
        qtbot.keyClick(focus() or w, Qt.Key.Key_Right, Qt.KeyboardModifier.AltModifier)
    assert w.step == WZ.STEP_COUNT - 1
    got: list[tuple[str, str, Any]] = []
    w.finished_with.connect(lambda text, what, path: got.append((text, what, path)))
    qtbot.keyClick(focus() or w, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert got and got[0][1] == "lab" and not w.isVisible()


def _tab_walk(qtbot: Any, w: PresetWizard, limit: int = 200) -> list[QWidget]:
    w.rail.setFocus()
    seen: list[QWidget] = []
    for _ in range(limit):
        f = focus()
        if f is None:
            break
        if f in seen:
            if f is w.rail:
                break
        else:
            seen.append(f)
        qtbot.keyClick(f, Qt.Key.Key_Tab)
    return seen


def _expected(w: PresetWizard) -> set[QWidget]:
    out: set[QWidget] = set()
    for c in w.findChildren(QWidget):
        if not isinstance(c, CONTROLS) or not c.isVisible() or not c.isEnabled() or c.window() is not w:
            continue
        if not c.focusPolicy() & Qt.FocusPolicy.TabFocus:
            continue
        if isinstance(c, QLineEdit) and isinstance(c.parent(), QAbstractSpinBox | QComboBox):
            continue  # 칸 안의 글 칸은 그 칸이 받는다
        out.add(c)
    return out


@pytest.mark.parametrize("start", ["running", "drill"])
def test_t21_tab_reaches_every_enabled_control(qtbot: Any, iso: Path, start: str) -> None:
    w = make(qtbot)
    w.choose_start(start)
    for step in range(WZ.STEP_COUNT):
        w.go(step)
        qtbot.wait(1)
        reached = set(_tab_walk(qtbot, w))
        missing = _expected(w) - reached
        assert not missing, [(type(m).__name__, m.objectName(), getattr(m, "text", lambda: "")()) for m in missing]
        assert w.step == step  # Tab은 단계를 바꾸지 않는다
        assert w.isVisible()


def test_t21_keyboard_only_running_to_lab(qtbot: Any, iso: Path) -> None:
    """키보드만으로: (달리기 카드가 골라진 채) 단계를 끝까지 넘기고 Tab으로 「실험실로 보내기」에 가서 Space."""
    w = make(qtbot)
    got: list[tuple[str, str, Any]] = []
    w.finished_with.connect(lambda text, what, path: got.append((text, what, path)))
    w.rail.setFocus()
    # ① 카드에 Tab으로 가서 Space로 고른다(이미 골라져 있어도 그대로)
    for _ in range(40):
        if focus() is w.pages[0].cards["running"]:
            break
        qtbot.keyClick(focus(), Qt.Key.Key_Tab)
    assert focus() is w.pages[0].cards["running"]
    qtbot.keyClick(focus(), Qt.Key.Key_Space)
    assert w.start == "running"
    for _ in range(WZ.STEP_COUNT - 1):
        qtbot.keyClick(focus(), Qt.Key.Key_Right, Qt.KeyboardModifier.AltModifier)
    assert w.step == WZ.STEP_COUNT - 1
    for _ in range(80):
        if focus() is w.btn_lab:
            break
        qtbot.keyClick(focus(), Qt.Key.Key_Tab)
    assert focus() is w.btn_lab
    qtbot.keyClick(w.btn_lab, Qt.Key.Key_Space)
    assert len(got) == 1 and got[0][1] == "lab"
    assert Path(got[0][2]).is_file()


def test_t21_keyboard_changes_a_rule_slot_and_a_slider(qtbot: Any, iso: Path) -> None:
    w = make(qtbot)
    w.go(1)
    slider = w.pages[1].sliders.widget_for("power")
    slider.setFocus()
    before = w.recipe.roles[0].body.values.get("power", 1.0)
    qtbot.keyClick(slider, Qt.Key.Key_Right)
    assert w.recipe.roles[0].body.values["power"] > before
    assert focus() is slider  # 값이 바뀌어도 위젯은 그대로(초점이 남음)
    w.go(3)
    combo = w.pages[3].rows["distance"].slots["direction"]
    assert isinstance(combo, QComboBox)
    combo.setFocus()
    qtbot.keyClick(combo, Qt.Key.Key_Down)
    rule = next(r for r in w.recipe.rules if r["rule"] == "distance")
    assert rule["direction"] != "right"
    assert focus() is combo

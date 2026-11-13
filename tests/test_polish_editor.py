"""6.2.0 화면 다듬기: 에디터 (polish/plan.md U38 ①②·U39·U40, 완료 기준 C2·C5·C9·C17) + 「새 프리셋…」 연결(설계 3.1 EditorView).

- U38 ①: 팔레트·속성 패널 폭을 실행 중에 잰다(팔레트 = 바라는 폭 + 스크롤 막대, 속성 패널 390). ②: 부품을 고르면 패널 최소 폭이 그
  폼이 바라는 폭까지 올라간다(상한 440). 내장 예제의 모든 부품 폼이 잘리지 않는다(1280×720, 글꼴 ×1.0·×1.10).
- U39: 「직접 조종」·「실험실로 보내기」는 따로 선 오른쪽 줄(go_bar). 1280 창 · 글꼴 ×1.25 · 「새 프리셋…」 단추가 있어도 도구 줄의 넘침
  단추가 안 보이고 으뜸 단추가 온전히 보인다(C17). 오류가 있으면 두 단추가 꺼지고 풍선에 까닭(C9), toolTip()은 그대로.
- U40: 빈 상태의 안내가 맨 위, 검사 결과 목록은 줄 수만큼.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication, QToolButton

from evolab import paths
from evolab.app import strings as S
from evolab.app import style
from evolab.app.editor.properties import PANEL_MAX_MIN, PANEL_MIN
from evolab.app.editor.view import PROPS_WIDTH, EditorView

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("bodies", "joints", "actuators", "sensors", "regions")


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """안내 창(쓰임 기록·QSettings·틀 폴더)과 에디터가 실제 사용자 폴더를 읽고 쓰지 않게(검토 고침)."""
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "evolab_data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "evolab_state"))


@pytest.fixture
def font_scale() -> Iterator[Any]:
    """앱 글꼴을 배율로 바꾸고 테스트가 끝나면 되돌린다(맑은 고딕이 리눅스 글꼴보다 넓은 몫: 계획 1.4·C2)."""
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    original = app.font()

    def scale(k: float) -> None:
        f = original.__class__(original)
        f.setPointSizeF(original.pointSizeF() * k)
        app.setFont(f)

    yield scale
    app.setFont(original)


def _view(qtbot: Any, size: tuple[int, int] = (1272, 660)) -> EditorView:
    view = EditorView()
    qtbot.addWidget(view)
    view.resize(*size)
    view.show()
    qtbot.waitExposed(view)
    QApplication.processEvents()
    return view


def _fully_visible(w: Any) -> bool:
    return w.isVisible() and w.visibleRegion().boundingRect().size() == w.size()


# ---------------------------------------------------------------------------
# U39: 도구 줄
# ---------------------------------------------------------------------------


def test_go_bar_holds_play_and_lab_and_toolbar_order(qtbot: Any) -> None:
    view = _view(qtbot)
    assert view.go_bar.actions() == [view.act_play, view.act_lab]
    assert view.act_play not in view.toolbar.actions() and view.act_lab not in view.toolbar.actions()
    texts = [a.text() for a in view.toolbar.actions() if a.text()]
    assert texts[:5] == [S.ED_NEW, S.ED_NEW_PRESET, S.ED_OPEN, S.ED_SAVE, S.ED_SAVE_AS]
    assert texts[-4:] == [S.ED_UNDO, S.ED_REDO, S.ED_COPY, S.ED_PASTE]
    assert view.go_bar.objectName() == "go_bar" and view.toolbar.objectName() == "toolbar"
    # 「내장 예제에서 시작」 단추는 그대로 도구 줄에(도움말 연결표가 글로 찾는다)
    assert any(isinstance(w, QToolButton) and w.text() == S.ED_FROM_BUILTIN for w in view.toolbar.findChildren(QToolButton))


def test_lab_button_is_the_one_primary_with_native_height(qtbot: Any) -> None:
    view = _view(qtbot)
    prim = style.primaries(view)
    assert prim == [view.lab_button]
    assert [b for b in prim if b.isEnabled()] == [view.lab_button]
    play = view.go_bar.widgetForAction(view.act_play)
    assert play is not None and view.lab_button.height() == play.height()


@pytest.mark.parametrize("scale", [1.0, 1.10, 1.25])
def test_toolbar_never_hides_the_primary_at_1280(qtbot: Any, font_scale: Any, scale: float) -> None:
    font_scale(scale)
    view = _view(qtbot, (1272, 660))  # 1280 창의 안쪽(테 4px씩)
    ext = view.toolbar.findChild(QToolButton, "qt_toolbar_ext_button")
    assert ext is None or not ext.isVisible()
    assert _fully_visible(view.lab_button)
    assert view.go_bar.width() >= view.go_bar.sizeHint().width()
    assert view.toolbar.sizeHint().width() <= view.toolbar.width()


def test_errors_turn_both_off_with_a_reason_and_keep_the_tooltip(qtbot: Any) -> None:
    view = _view(qtbot)
    lab = view.lab_button
    play = view.go_bar.widgetForAction(view.act_play)
    assert lab.isEnabled() and style.tip_text(lab) == lab.toolTip() == S.ED_TO_LAB
    data = json.loads((ROOT / "environments" / "lander.evoenv").read_text(encoding="utf-8-sig"))
    data["sensors"][0]["body"] = "nope"
    data["actuators"][1]["body"] = "nope"
    view.doc.load_text(json.dumps(data, ensure_ascii=False), None)
    view.doc.run_check()
    errors = sum(1 for i in view.doc.issues if i.severity == "error")
    assert errors >= 2
    why = S.ED_WHY_ERRORS.format(n=errors)
    for w in (lab, play):
        assert not w.isEnabled()
        assert style.why_off(w) == why
        assert style.tip_text(w).endswith(why) and w.toolTip() in style.tip_text(w)
    assert lab.toolTip() == S.ED_TO_LAB  # 까닭은 풍선을 띄울 때만 붙는다(toolTip은 그대로)
    assert style.primaries(view) == [lab] and not [b for b in style.primaries(view) if b.isEnabled()]
    view.doc.undo_stack.clear()
    view.doc.load_text((ROOT / "environments" / "lander.evoenv").read_text(encoding="utf-8-sig"), None)
    view.doc.run_check()
    assert lab.isEnabled() and style.tip_text(lab) == S.ED_TO_LAB


# ---------------------------------------------------------------------------
# U38: 나눔 폭과 폼 폭
# ---------------------------------------------------------------------------


def test_splitter_sizes_are_measured(qtbot: Any) -> None:
    view = _view(qtbot)
    pal, center, props = view.top_split.sizes()
    assert pal == view.palette_width() >= view.palette.sizeHint().width()
    assert props >= PROPS_WIDTH
    assert center >= 560
    ow = view.palette.owner_world
    assert _fully_visible(ow)  # 「세계(땅·벽·공)」가 다 보인다(계획 V10)


@pytest.mark.parametrize("scale", [1.0, 1.10])
def test_no_builtin_part_form_is_cut(qtbot: Any, font_scale: Any, scale: float) -> None:
    font_scale(scale)
    view = _view(qtbot)
    seen = 0
    for f in paths.builtin_envs():
        view.doc.from_builtin(f.name)
        QApplication.processEvents()
        for sec in SECTIONS:
            for item in view.doc.items(sec):
                view.doc.select([(sec, str(item["id"]))])
                QApplication.processEvents()
                panel = view.props
                need = panel.form.minimumSizeHint().width()
                have = panel.scroll.viewport().width()
                assert need <= have, (f.name, sec, item["id"], need, have)
                assert PANEL_MIN <= panel.minimumWidth() <= PANEL_MAX_MIN
                seen += 1
    assert seen >= 80


def test_panel_minimum_width_follows_the_form(qtbot: Any) -> None:
    view = _view(qtbot)
    view.doc.from_builtin("glider.evoenv")
    view.doc.select([("bodies", "glider")])
    QApplication.processEvents()
    panel = view.props
    assert panel.minimumWidth() == max(PANEL_MIN, min(PANEL_MAX_MIN, panel.wanted_width()))
    assert panel.minimumWidth() > panel.form.minimumSizeHint().width()
    view.doc.select([])
    QApplication.processEvents()
    assert panel.minimumWidth() == PANEL_MIN


# ---------------------------------------------------------------------------
# U40: 빈 상태와 문제 상자
# ---------------------------------------------------------------------------


def test_empty_panel_puts_the_hint_on_top(qtbot: Any) -> None:
    view = _view(qtbot)
    panel = view.props
    view.doc.select([])
    QApplication.processEvents()
    top = panel.hint.mapTo(panel, QPoint(0, 0)).y()
    assert panel.title.isVisible() and panel.hint.isVisible()
    assert top < 80  # 가운데(약 185px)가 아니라 맨 위
    assert panel.filler.isVisible()


def test_issue_list_is_as_tall_as_its_rows(qtbot: Any) -> None:
    view = _view(qtbot)
    view.doc.load_text((ROOT / "tests" / "data" / "audit" / "joint_apart.evoenv").read_text(encoding="utf-8"), None)
    view.doc.select([("joints", "hip")])
    view.doc.run_check()
    QApplication.processEvents()
    panel = view.props
    assert panel.issue_box.isVisible() and panel.issue_list.count() == 1
    assert panel.issue_list.maximumHeight() < 110  # 한 줄 문제에 110px 상자를 쓰지 않는다
    assert not panel.filler.isVisible()


# ---------------------------------------------------------------------------
# 「새 프리셋…」 (설계 3.1 EditorView, D42)
# ---------------------------------------------------------------------------


def test_new_preset_action_and_shortcuts(qtbot: Any) -> None:
    view = _view(qtbot)
    assert view.act_new_preset.text() == S.ED_NEW_PRESET and view.act_new_preset.toolTip() == S.ED_NEW_PRESET_TIP
    assert view.act_new_preset.shortcut() == QKeySequence("Ctrl+Shift+N")
    assert view.act_new.shortcut() == QKeySequence(QKeySequence.StandardKey.New)  # Ctrl+N은 그대로 빈 환경
    assert view.act_new_preset.objectName() == "act_new_preset"


def test_new_preset_opens_a_window_modal_wizard(qtbot: Any) -> None:
    from evolab.app.preset.wizard import PresetWizard

    view = _view(qtbot)
    wizard = view.new_preset()
    assert isinstance(wizard, PresetWizard) and view.wizard is wizard
    assert wizard.isVisible() and wizard.windowModality() == Qt.WindowModality.WindowModal
    assert wizard.parent() is view
    wizard.reject()


def test_load_preset_opens_the_saved_file_and_can_send_to_lab(qtbot: Any, tmp_path: Path) -> None:
    view = _view(qtbot)
    text = (ROOT / "environments" / "jump.evoenv").read_text(encoding="utf-8")
    path = tmp_path / "점프.evoenv"
    path.write_text(text, encoding="utf-8")
    sent: list[str] = []
    view.send_to_lab.connect(sent.append)
    assert view.load_preset(text, path, "edit")
    assert view.doc.path == path and not view.doc.dirty and sent == []
    assert view.load_preset(text, str(path), "lab")
    assert len(sent) == 1 and json.loads(sent[0])["name"] == json.loads(text)["name"]


def test_wizard_can_leave_asks_the_editor_and_the_lab(qtbot: Any, monkeypatch: Any) -> None:
    view = _view(qtbot)
    answers = {"save": False}
    monkeypatch.setattr(view, "maybe_save", lambda: answers["save"])
    assert not view._wizard_can_leave("edit")
    answers["save"] = True
    assert view._wizard_can_leave("edit") and view._wizard_can_leave("lab")
    shown: list[str] = []
    from evolab.app.editor import view as V

    monkeypatch.setattr(V.QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    view.lab_ready = lambda: False
    assert view._wizard_can_leave("edit") and not view._wizard_can_leave("lab")
    assert shown == [S.MAIN_LAB_BUSY]

"""6.2.0 화면 다듬기 바탕(U1~U3): app/style.py와 풍선 필터.

계획(polish/plan.md 2.1)의 ① 대비 ② 숫자 형식 ③ 테마 ④ 으뜸 단추 ⑤ 끈 까닭, U2의 색 글자 자리, U3의 풍선 폭을 본다.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, Qt
from PySide6.QtGui import QAction, QColor, QHelpEvent, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolButton,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from evolab.app import strings as S
from evolab.app import style

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "evolab" / "app"


# ---------------------------------------------------------------------------
# ① 대비 (색 계산만)
# ---------------------------------------------------------------------------
def _lum(color: str) -> float:
    c = QColor(color)

    def ch(v: int) -> float:
        x = v / 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

    return 0.2126 * ch(c.red()) + 0.7152 * ch(c.green()) + 0.0722 * ch(c.blue())


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.mark.parametrize(("ui", "surfaces"), [(style.UI_LIGHT, style.LIGHT_SURFACES), (style.UI_DARK, style.DARK_SURFACES)])
def test_text_tokens_reach_4_5_on_every_surface(ui, surfaces):
    for token in ("ink2", "muted", "link", "danger"):
        for surface in surfaces:
            assert contrast(getattr(ui, token), surface) >= 4.5, (token, surface)


@pytest.mark.parametrize(("ui", "surfaces"), [(style.UI_LIGHT, style.LIGHT_SURFACES), (style.UI_DARK, style.DARK_SURFACES)])
def test_picture_tokens_reach_3_on_every_surface(ui, surfaces):
    for token in ("ok", "warn", "info", "idle", "bad", "accent"):
        for surface in surfaces:
            assert contrast(getattr(ui, token), surface) >= 3.0, (token, surface)


@pytest.mark.parametrize("ui", [style.UI_LIGHT, style.UI_DARK])
def test_primary_button_text_contrast(ui):
    for fill in (ui.accent, ui.accent_hover, ui.accent_down):
        assert contrast(ui.on_accent, fill) >= 4.5, fill


def test_tick_is_the_text_muted_and_picture_muted_is_unchanged():
    assert style.LIGHT.tick == style.UI_LIGHT.muted == "#6b6963"
    assert style.DARK.tick == style.UI_DARK.muted == "#a3a199"
    # 그림의 흐린 색은 v6.0 뇌 그림과 바이트가 같아야 하므로 바꾸지 않는다(V4)
    assert style.LIGHT.muted == style.DARK.muted == "#898781"
    for surface in style.LIGHT_SURFACES:
        assert contrast(style.LIGHT.tick, surface) >= 4.5
    for surface in style.DARK_SURFACES:
        assert contrast(style.DARK.tick, surface) >= 4.5


def test_dashboard_reexports_the_same_objects():
    from evolab.app.lab import dashboard as D

    assert D.Theme is style.Theme and D.LIGHT is style.LIGHT and D.DARK is style.DARK
    assert D.theme_for is style.theme_for
    assert D.fmt_value is style.fmt_score
    assert [f for f in style.Theme.__dataclass_fields__] == [
        "surface", "ink", "ink2", "muted", "grid", "axis", "series", "ramp", "tick"]


# ---------------------------------------------------------------------------
# ② 숫자 형식
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("value", "text"), [
    (1824.4, "1,824"), (-162.734, "-162.7"), (2.5, "2.50"), (0.125, "0.125"), (0.0, "0"), (-0.0, "0"),
    (-0.0004, "-0.0004"), (float("nan"), S.CMP_NONE), (None, S.CMP_NONE), (float("inf"), S.CMP_NONE), (100.0, "100.0"),
])
def test_fmt_score(value, text):
    assert style.fmt_score(value) == text


def test_fmt_score_matches_v61_fmt_value_on_finite_numbers():
    def old(v: float) -> str:  # v6.1 dashboard.fmt_value 그대로
        a = abs(v)
        if a >= 1000:
            return f"{v:,.0f}"
        if a >= 100:
            return f"{v:.1f}"
        if a >= 1:
            return f"{v:.2f}"
        if a == 0:
            return "0"
        return f"{v:.3g}"

    for v in (12345.678, -999.95, 150.04, -1.005, 0.000123, -0.5, 7, 1e9):
        assert style.fmt_score(v) == old(v), v


def test_other_formats():
    assert style.fmt_count(40000) == "40,000"
    assert style.fmt_count(float("nan")) == S.CMP_NONE
    assert style.fmt_seconds(2.44) == "2.4초"
    assert style.fmt_seconds(-0.04) == "0.0초"
    assert style.fmt_duration(12.0) == "12.0초"
    assert style.fmt_duration(192) == "3분 12초"
    assert style.fmt_duration(3 * 3600 + 5 * 60) == "3시간 05분"
    assert style.fmt_percent(85.2) == "85%"
    import datetime as dt

    now = dt.datetime(2026, 11, 13, 9, 0)
    assert style.fmt_when("2026-11-13T12:07:30", now) == "11월 13일 12:07"
    assert style.fmt_when("2025-01-02T03:04:00", now) == "2025년 1월 2일 03:04"
    assert style.fmt_when("어제", now) == "어제"


# ---------------------------------------------------------------------------
# ③ 테마
# ---------------------------------------------------------------------------
def _dark_palette() -> QPalette:
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#323232"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#f0efec"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#191919"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#f0efec"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#323232"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#f0efec"))
    return pal


@pytest.fixture
def light_app(qapp):
    old = QApplication.palette()
    if style.is_dark():
        QApplication.setPalette(QApplication.style().standardPalette())
    style.refresh_theme()
    yield qapp
    QApplication.setPalette(old)
    style.refresh_theme()


def test_theme_follows_the_app_palette_not_the_styled_button(light_app, qtbot):
    btn = QPushButton("시작")
    qtbot.addWidget(btn)
    style.set_role(btn, "primary")
    btn.show()
    assert not style.is_dark()
    assert not style.is_dark(btn)
    assert style.theme_for(btn) is style.LIGHT and style.ui_for(btn) is style.UI_LIGHT


def test_set_role_twice_keeps_the_style_sheet_and_refresh_switches_theme(light_app, qtbot):
    root = QWidget()
    qtbot.addWidget(root)
    lay = QHBoxLayout(root)
    btn = QPushButton("이어서 학습")
    lab = QLabel("안내")
    lay.addWidget(btn)
    lay.addWidget(lab)
    style.set_role(btn, "primary")
    style.tone(lab, "ink2")
    sheet = btn.styleSheet()
    assert style.UI_LIGHT.accent in sheet
    assert btn.property(style.ROLE_PROPERTY) == "primary"
    style.set_role(btn, "primary")
    assert btn.styleSheet() == sheet
    assert style.refresh_theme() == 0  # 테마 그대로

    QApplication.setPalette(_dark_palette())
    assert style.is_dark()
    assert style.refresh_theme() >= 2
    assert style.UI_DARK.accent in btn.styleSheet() and style.UI_LIGHT.accent not in btn.styleSheet()
    assert style.UI_DARK.ink2 in lab.styleSheet()
    assert style.refresh_theme() == 0
    QApplication.setPalette(QApplication.style().standardPalette())
    assert style.refresh_theme() >= 2
    assert btn.styleSheet() == sheet
    assert style.UI_LIGHT.ink2 in lab.styleSheet()


def test_on_dark_card_uses_dark_values_in_the_light_theme(light_app, qtbot):
    btn = QPushButton("학습 시작")
    qtbot.addWidget(btn)
    style.set_role(btn, "primary", on_dark_card=True)
    assert style.UI_DARK.accent in btn.styleSheet()


def test_registry_forgets_deleted_widgets(light_app):
    lab = QLabel("x")
    style.tone(lab, "muted")
    key = id(lab)
    assert key in style._styled  # noqa: SLF001
    lab.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    QTest.qWait(10)
    assert key not in style._styled  # noqa: SLF001


def test_unknown_roles_are_refused(qtbot):
    btn = QPushButton()
    qtbot.addWidget(btn)
    with pytest.raises(ValueError):
        style.set_role(btn, "loud")
    with pytest.raises(ValueError):
        style.tone(btn, "pink")
    with pytest.raises(ValueError):
        style.font(btn, "huge")


def test_font_roles(qtbot):
    lab = QLabel("x")
    qtbot.addWidget(lab)
    base = lab.font().pointSizeF()
    assert style.font(lab, "body").pointSizeF() == base
    assert style.font(lab, "strong").bold()
    assert style.font(lab, "title").pointSizeF() == pytest.approx(base + 2)
    assert style.font(lab, "display").pointSizeF() == pytest.approx(base + 4)
    assert style.font(lab, "small").pointSizeF() <= base
    assert style.font(lab, "mono").families()[0] == style.MONO_FAMILIES[0]


def test_note_is_a_wrapped_secondary_line(light_app):
    lab = style.note("설명")
    assert lab.wordWrap() and lab.text() == "설명"
    assert style.UI_LIGHT.ink2 in lab.styleSheet()
    lab.deleteLater()


# ---------------------------------------------------------------------------
# ④ 으뜸 단추
# ---------------------------------------------------------------------------
def test_primary_keeps_native_height_for_push_and_tool_buttons(light_app, qtbot):
    win = QMainWindow()
    qtbot.addWidget(win)
    bar = QToolBar()
    win.addToolBar(bar)
    act = QAction("실험실로 보내기", win)
    bar.addAction(act)
    tool = bar.widgetForAction(act)
    assert isinstance(tool, QToolButton)
    central = QWidget()
    lay = QHBoxLayout(central)
    push = QPushButton("잠시 멈춤")
    side = QPushButton("한 세대만 더")
    lay.addWidget(push)
    lay.addWidget(side)
    win.setCentralWidget(central)
    win.show()
    qtbot.waitExposed(win)
    push_h, tool_h = push.sizeHint().height(), tool.sizeHint().height()
    style.set_role(push, "primary")
    style.set_role(tool, "primary")
    assert push.sizeHint().height() == push_h
    assert tool.sizeHint().height() == tool_h
    push.setEnabled(False)
    assert push.sizeHint().height() == push_h
    style.set_role(push, "danger")
    assert push.sizeHint().height() == push_h
    style.set_role(push, "normal")
    assert push.styleSheet() == ""


def test_disabled_primary_is_not_filled(light_app, qtbot):
    root = QWidget()  # 투명한 바탕은 부모가 칠한다: 부모째 찍는다
    root.setAutoFillBackground(True)
    qtbot.addWidget(root)
    lay = QHBoxLayout(root)
    btn = QPushButton("이어서 학습")
    lay.addWidget(btn)
    style.set_role(btn, "primary")
    root.show()
    qtbot.waitExposed(root)
    accent = QColor(style.UI_LIGHT.accent)

    def left_mid() -> QColor:
        img = root.grab(btn.geometry()).toImage()
        return img.pixelColor(6, img.height() // 2)

    assert left_mid() == accent
    btn.setEnabled(False)
    assert left_mid() != accent


def test_quiet_role_and_primaries(light_app, qtbot):
    root = QWidget()
    qtbot.addWidget(root)
    lay = QHBoxLayout(root)
    tb = QToolButton()
    a, b = QPushButton("a"), QPushButton("b")
    for w in (tb, a, b):
        lay.addWidget(w)
    style.set_role(tb, "quiet")
    assert tb.autoRaise()
    style.set_role(tb, "normal")
    assert not tb.autoRaise()
    style.set_role(a, "primary")
    style.set_role(b, "primary")
    b.setEnabled(False)
    root.show()
    found = style.primaries(root)
    assert set(found) == {a, b}
    assert [x for x in found if x.isEnabled()] == [a]
    b.hide()
    assert style.primaries(root) == [a]


def test_state_dot_and_chip(light_app, qtbot):
    chip = style.StateChip()
    qtbot.addWidget(chip)
    chip.set_state("running", "학습 중 · 3세대")
    assert chip.dot.kind == "running" and chip.dot.color() == style.UI_LIGHT.ok
    assert chip.label.text() == "학습 중 · 3세대"
    for kind, token in (("paused", "warn"), ("finished", "info"), ("closed", "idle"), ("failed", "bad"), ("??", "idle")):
        chip.dot.set_kind(kind)
        assert chip.dot.color() == getattr(style.UI_LIGHT, token)
    assert chip.dot.focusPolicy() == Qt.FocusPolicy.NoFocus
    chip.show()
    chip.grab()  # 그리기가 터지지 않음


# ---------------------------------------------------------------------------
# ⑤ 끈 까닭
# ---------------------------------------------------------------------------
class _PropCount(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def eventFilter(self, obj, event):  # noqa: N802
        if event.type() == QEvent.Type.DynamicPropertyChange:
            self.count += 1
        return False


def test_set_enabled_keeps_tooltip_and_adds_reason(qtbot):
    btn = QPushButton("비교")
    qtbot.addWidget(btn)
    btn.setToolTip("두 세대를 나란히 봅니다.")
    style.set_enabled(btn, False, S.LAB_WHY_NEED_TWO)
    assert not btn.isEnabled()
    assert btn.toolTip() == "두 세대를 나란히 봅니다."
    assert style.why_off(btn) == S.LAB_WHY_NEED_TWO
    assert style.tip_text(btn) == f"두 세대를 나란히 봅니다.\n\n{S.LAB_WHY_NEED_TWO}"
    style.set_enabled(btn, True, S.LAB_WHY_NEED_TWO)
    assert style.tip_text(btn) == btn.toolTip()
    assert style.why_off(btn) == ""
    # 누가 setEnabled(True)를 직접 불러도 옛 까닭은 보이지 않는다
    style.set_enabled(btn, False, S.LAB_WHY_NEED_TWO)
    btn.setEnabled(True)
    assert style.tip_text(btn) == btn.toolTip()
    # 풍선이 없는 꺼진 위젯은 까닭만
    bare = QPushButton()
    qtbot.addWidget(bare)
    style.set_enabled(bare, False, "까닭")
    assert style.tip_text(bare) == "까닭"


def test_same_value_does_not_touch_properties(qtbot):
    btn = QPushButton()
    qtbot.addWidget(btn)
    counter = _PropCount()
    btn.installEventFilter(counter)
    style.set_enabled(btn, False, "까닭")
    first = counter.count
    assert first >= 1
    style.set_enabled(btn, False, "까닭")
    assert counter.count == first
    style.set_role(btn, "quiet")
    after_role = counter.count
    style.set_role(btn, "quiet")
    assert counter.count == after_role


def test_toolbar_button_reads_the_reason_from_its_action(qtbot):
    win = QMainWindow()
    qtbot.addWidget(win)
    bar = QToolBar()
    win.addToolBar(bar)
    act = QAction("실험실로 보내기", win)
    act.setToolTip("이 환경을 실험실로 보냅니다.")
    bar.addAction(act)
    tool = bar.widgetForAction(act)
    style.set_enabled(act, False, S.ED_WHY_ERRORS)
    assert not tool.isEnabled()
    assert style.why_off(tool) == S.ED_WHY_ERRORS
    assert S.ED_WHY_ERRORS in style.tip_text(tool)


# ---------------------------------------------------------------------------
# U3 풍선 필터
# ---------------------------------------------------------------------------
LONG_TIP = max(S.HELP_TIPS.values(), key=len)


def test_wrap_tip_rules():
    assert style.wrap_tip("짧은 풍선") is None
    assert style.wrap_tip("") is None
    assert style.wrap_tip("<b>서식</b> " + "가" * 400) is None
    wrapped = style.wrap_tip(LONG_TIP + "\n<둘째 줄> & 끝")
    assert wrapped is not None and wrapped.startswith('<table width="420">')
    assert "&lt;둘째 줄&gt; &amp; 끝" in wrapped and "<br>" in wrapped


@pytest.fixture
def tips_filter(qapp):
    from evolab.app.help import tooltips

    flt = tooltips.WindowTips(qapp)
    qapp.installEventFilter(flt)
    yield flt
    qapp.removeEventFilter(flt)
    QToolTip.hideText()


def _shown_tip(target: QWidget, pos: QPoint) -> tuple[int, str] | None:
    QToolTip.hideText()
    QTest.qWait(30)
    QApplication.sendEvent(target, QHelpEvent(QEvent.Type.ToolTip, pos, target.mapToGlobal(pos)))
    QTest.qWait(80)
    labels = [w for w in QApplication.topLevelWidgets() if w.inherits("QTipLabel") and w.isVisible()]
    if not labels:
        return None
    return labels[0].width(), labels[0].text()


def test_filter_wraps_long_tips_and_adds_reasons(tips_filter, qtbot):
    root = QWidget()
    qtbot.addWidget(root)
    lay = QVBoxLayout(root)
    on, off, short_off, short = QPushButton("켜짐"), QPushButton("꺼짐"), QPushButton("짧음"), QPushButton("짧은 켜짐")
    for w in (on, off, short_off, short):
        lay.addWidget(w)
    on.setToolTip(LONG_TIP)
    off.setToolTip(LONG_TIP)
    style.set_enabled(off, False, S.LAB_WHY_NEED_GEN)
    short_off.setToolTip("짧은 풍선")
    style.set_enabled(short_off, False, S.LAB_WHY_NEED_GEN)
    short.setToolTip("짧은 풍선")
    root.show()
    qtbot.waitExposed(root)
    centre = QPoint(5, 5)

    shown = _shown_tip(on, centre)
    assert shown is not None and shown[0] <= 440
    shown = _shown_tip(off, centre)
    assert shown is not None and shown[0] <= 440
    assert S.LAB_WHY_NEED_GEN in shown[1]
    shown = _shown_tip(short_off, centre)
    assert shown is not None and S.LAB_WHY_NEED_GEN in shown[1] and "짧은 풍선" in shown[1]
    # 짧고 까닭 없는 풍선은 Qt가 그대로 띄운다(필터가 받지 않음)
    event = QHelpEvent(QEvent.Type.ToolTip, centre, short.mapToGlobal(centre))
    assert tips_filter.show_tip(short, event) is False
    assert on.toolTip() == LONG_TIP and off.toolTip() == LONG_TIP


def test_filter_wraps_long_item_tips_without_changing_them(tips_filter, qtbot):
    root = QWidget()
    qtbot.addWidget(root)
    lay = QVBoxLayout(root)
    tbl = QTableWidget(1, 2)
    cell = QTableWidgetItem("칸")
    cell.setToolTip(LONG_TIP)
    tbl.setItem(0, 0, cell)
    short = QTableWidgetItem("짧은 칸")
    short.setToolTip("짧은 풍선")
    tbl.setItem(0, 1, short)
    tree = QTreeWidget()
    node = QTreeWidgetItem(["마디"])
    node.setToolTip(0, LONG_TIP)
    tree.addTopLevelItem(node)
    lst = QListWidget()
    lst.addItem("줄")
    lst.item(0).setToolTip(LONG_TIP)
    for w in (tbl, tree, lst):
        lay.addWidget(w)
    root.resize(500, 420)
    root.show()
    qtbot.waitExposed(root)
    for view, rect in ((tbl, tbl.visualItemRect(cell)), (tree, tree.visualItemRect(node)), (lst, lst.visualItemRect(lst.item(0)))):
        shown = _shown_tip(view.viewport(), rect.center())
        assert shown is not None and shown[0] <= 440, view
    event = QHelpEvent(QEvent.Type.ToolTip, tbl.visualItemRect(short).center(),
                       tbl.viewport().mapToGlobal(tbl.visualItemRect(short).center()))
    assert tips_filter.show_tip(tbl.viewport(), event) is False
    assert cell.toolTip() == LONG_TIP and node.toolTip(0) == LONG_TIP


def test_filter_refreshes_theme_on_application_palette_change(tips_filter, light_app, qtbot):
    lab = QLabel("안내")
    qtbot.addWidget(lab)
    style.tone(lab, "ink2")
    QApplication.setPalette(_dark_palette())
    QTest.qWait(10)
    assert style.UI_DARK.ink2 in lab.styleSheet()
    QApplication.setPalette(QApplication.style().standardPalette())
    QTest.qWait(10)
    assert style.UI_LIGHT.ink2 in lab.styleSheet()


# ---------------------------------------------------------------------------
# U2 색 글자 자리와 모듈 경계
# ---------------------------------------------------------------------------
ALLOWED_8A93A6 = {"editor/palette.py": 2, "export/phone.py": 1, "play/window.py": 1}
"""늘 어두운 캔버스 위의 그림 색(글자가 아님): 이름으로 허용한 네 곳."""


def test_old_text_colours_are_gone_outside_style():
    counts: dict[str, int] = {}
    for path in APP.rglob("*.py"):
        rel = path.relative_to(APP).as_posix()
        if rel == "style.py":
            continue
        text = path.read_text(encoding="utf-8").lower()
        assert "#9aa0aa" not in text, rel
        assert "#d9534f" not in text, rel
        n = text.count("#8a93a6")
        if n:
            counts[rel] = n
    assert counts == ALLOWED_8A93A6


def test_text_uses_of_muted_moved_to_tick():
    pattern = re.compile(r"setTextPen\(_pen\([a-z_.]*muted")
    for rel in ("lab/dashboard.py", "lab/watch_tiles.py", "lab/grow_tile.py", "compare/view.py"):
        assert not pattern.search((APP / rel).read_text(encoding="utf-8")), rel
    assert "note_label.setStyleSheet(f\"color: {t.muted}" not in (APP / "compare/view.py").read_text(encoding="utf-8")


def test_style_imports_no_numpy_pyqtgraph_or_screens():
    code = ("import sys, evolab.app.style; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('numpy', 'pyqtgraph') "
            "or (m.startswith('evolab') and m not in ('evolab', 'evolab.app', 'evolab.app.strings', 'evolab.app.style'))]; "
            "print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         env={**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"

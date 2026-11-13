"""6.2.0 묶음 ED1: 「새로 그리는 것」 두 단추와 고정/움직임, 캔버스의 소유 글자 (설계 T29, DU T17·4.3.2, 다듬기 U38 ④).

- 팔레트를 최소 폭으로 줄여도(글꼴 ×1.25에서도) 두 단추가 가로로 다 보인다(가려진 폭 0).
- 「세계 · 움직임」으로 그린 원은 세계·동적, 「세계 · 고정」은 세계·정적, 에이전트는 늘 동적.
- 모양 도구가 켜져 있으면 커서 옆에 소유 글자(에이전트/세계)가 소유 색으로 그려진다.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QFont
from PySide6.QtTest import QTest

from evolab.app import strings as S
from evolab.app.editor import tools as T
from evolab.app.editor.canvas import EditorCanvas
from evolab.app.editor.document import EnvDocument
from evolab.app.editor.palette import Palette

LEFT = Qt.MouseButton.LeftButton
NO = Qt.KeyboardModifier.NoModifier


def make(qtbot: Any) -> tuple[EditorCanvas, Palette]:
    c = EditorCanvas(EnvDocument())
    qtbot.addWidget(c)
    c.resize(900, 600)
    c.show()
    c.fit()  # 처음 맞춤을 먼저 끝낸다(뒤의 보기 설정을 덮어쓰지 않게)
    c.center, c.scale = (0.0, 0.0), 40.0
    c.set_snap(True, 0.5)
    pal = Palette()
    qtbot.addWidget(pal)
    pal.connect_canvas(c)
    return c, pal


def click_at(c: EditorCanvas, x: float, y: float) -> None:
    p = c.world_to_screen(x, y)
    QTest.mouseClick(c, LEFT, NO, QPoint(round(p.x()), round(p.y())))


def _fully_visible(w: Any) -> bool:
    return w.isVisible() and w.visibleRegion().boundingRect() == w.rect() and w.width() >= w.sizeHint().width()


def test_owner_buttons_fit_at_the_palette_minimum_width(qtbot: Any) -> None:
    for factor in (1.0, 1.25):
        pal = Palette()
        qtbot.addWidget(pal)
        if factor != 1.0:
            f = QFont(pal.font())
            f.setPointSizeF(f.pointSizeF() * factor)
            pal.setFont(f)
            qtbot.wait(10)
        pal.resize(pal.minimumWidth(), 700)
        pal.show()
        qtbot.waitExposed(pal)
        assert pal.width() == pal.minimumWidth() and pal.sizeHint().width() == pal.minimumWidth()
        for b in (pal.owner_agent, pal.owner_world, pal.buttons["tpl_body"], pal.buttons["tpl_prop"], pal.owner_label):
            assert _fully_visible(b), (factor, b.objectName() or b.text())
        pal.owner_world.click()
        qtbot.wait(1)
        assert _fully_visible(pal.world_motion), factor
        pal.close()


def test_owner_widgets_names_and_tips(qtbot: Any) -> None:
    _c, pal = make(qtbot)
    assert pal.owner_agent.objectName() == "owner_agent" and pal.owner_world.objectName() == "owner_world"
    assert pal.owner_agent.text() == S.EDC_OWNER_AGENT and pal.owner_world.text() == S.EDC_OWNER_WORLD
    assert pal.owner_label.text() == S.EDC_OWNER_LABEL
    assert pal.owner_agent.toolTip() == S.EDC_OWNER_AGENT_TIP and pal.owner_world.toolTip() == S.EDC_OWNER_WORLD_TIP
    assert pal.world_motion.toolTip() == S.EDC_WORLD_MOTION_TIP
    assert [pal.world_motion.itemText(i) for i in range(pal.world_motion.count())] == [
        S.EDC_WORLD_MOTION["static"], S.EDC_WORLD_MOTION["dynamic"]]
    assert pal.owner_agent.isChecked() and pal.world_motion.isHidden()  # 처음: 에이전트, 고르기는 세계일 때만
    assert not pal.owner_agent.icon().isNull() and not pal.owner_world.icon().isNull()  # 소유 색 띠


def test_world_dynamic_circle_and_static_box(qtbot: Any) -> None:
    c, pal = make(qtbot)
    doc = c.doc
    pal.owner_world.click()
    assert c.new_owner == "world" and not pal.world_motion.isHidden() and c.new_motion == "static"
    pal.world_motion.setCurrentIndex(T.WORLD_MOTIONS.index("dynamic"))
    assert c.new_motion == "dynamic"
    pal.buttons["circle"].click()
    click_at(c, 1.0, 1.0)
    ball = doc.items("bodies")[-1]
    assert (ball["owner"], ball["type"]) == ("world", "dynamic") and ball["shapes"][0]["kind"] == "circle"
    pal.set_owner_mode("world", "static")
    assert c.new_motion == "static"
    pal.buttons["box"].click()
    click_at(c, 4.0, 1.0)
    wall = doc.items("bodies")[-1]
    assert (wall["owner"], wall["type"]) == ("world", "static")
    # 에이전트는 고정/움직임과 무관하게 움직이는 물체
    pal.world_motion.setCurrentIndex(T.WORLD_MOTIONS.index("dynamic"))
    pal.owner_agent.click()
    assert pal.world_motion.isHidden() and c.new_owner == "agent"
    click_at(c, -3.0, 1.0)
    agent = doc.items("bodies")[-1]
    assert (agent["owner"], agent["type"]) == ("agent", "dynamic")
    # 고정/움직임 고르기는 화면 상태: 되돌리기에 들어가지 않는다
    n = doc.undo_stack.count()
    pal.set_owner_mode("world", "dynamic")
    assert doc.undo_stack.count() == n
    # 조작 함수도 같은 규칙(움직임 인자가 없거나 에이전트면 옛 기본)
    assert doc.item(T.add_poly(doc, "world", [(0, 5), (1, 5), (0, 6)], motion="dynamic"))["type"] == "dynamic"
    assert doc.item(T.add_box(doc, "world", (8, 0), 1, 1))["type"] == "static"
    assert doc.item(T.add_circle(doc, "agent", (8, 3), 0.5, motion="static"))["type"] == "dynamic"


def test_owner_text_beside_the_cursor(qtbot: Any) -> None:
    c, pal = make(qtbot)
    assert c.owner_cursor_text() == ""  # 선택 도구에서는 없다
    pal.buttons["box"].click()
    assert c.owner_cursor_text() == S.EDC_CURSOR_OWNER["agent"] and c.owner_color() == T.OWNER_COLORS["agent"]
    pal.owner_world.click()
    assert c.owner_cursor_text() == S.EDC_CURSOR_OWNER["world"] and c.owner_color() == T.OWNER_COLORS["world"]
    for tool in ("circle", "poly"):
        pal.buttons[tool].click()
        assert c.owner_cursor_text() == S.EDC_CURSOR_OWNER["world"]
    pal.buttons["joint_pivot"].click()
    assert c.owner_cursor_text() == ""
    pal.buttons["box"].click()
    c.cursor_world = (0.0, 0.0)
    with_text = c.grab().toImage()
    pal.buttons["select"].click()
    c.cursor_world = (0.0, 0.0)
    assert c.grab().toImage() != with_text  # 글자가 실제로 그려진다

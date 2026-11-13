"""6.2.0 묶음 ED1: 팔레트 「틀」 묶음과 틀 넣기 (설계 T28, D48, DU 4.2, M7의 넣기 몫).

- 「몸 틀…」「물건…」은 도구가 아닌 보통 단추다(TOOL_IDS 밖, 도구 무리 밖). TOOL_IDS는 35개이고 새 것은 "move" 하나다.
- 단추 -> 보관함 창(여기서는 바꿔 끼운 가짜 창) -> 캔버스를 누른 자리에 넣기. 넣기 한 번은 되돌리기 한 단계이고, 되돌리면 사전이
  넣기 전과 같다. 같은 틀을 두 번 넣으면 둘째 묶음이 모두 같은 번호(_2)를 받는다.
"""

from __future__ import annotations

import copy
import statistics
import time
from typing import Any

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog

from evolab.app import strings as S
from evolab.app.editor import tools as T
from evolab.app.editor.canvas import EditorCanvas
from evolab.app.editor.document import EnvDocument
from evolab.app.editor.palette import TOOL_IDS, Palette
from evolab.presets.library import default_library

LEFT = Qt.MouseButton.LeftButton
NO = Qt.KeyboardModifier.NoModifier


class FakePicker:
    """app.preset.cards.TemplatePicker(kind, world, data, parent)의 자리. exec()·choice()만 쓴다."""

    calls: list[tuple[str, str]] = []
    answer: tuple[Any, ...] | None = None

    def __init__(self, kind: str, world: str, data: Any, parent: Any = None) -> None:
        FakePicker.calls.append((kind, world))
        self.data = data

    def exec(self) -> QDialog.DialogCode:
        return QDialog.DialogCode.Accepted if FakePicker.answer is not None else QDialog.DialogCode.Rejected

    def choice(self) -> tuple[Any, ...] | None:
        return FakePicker.answer


def make(qtbot: Any, doc: EnvDocument | None = None) -> tuple[EditorCanvas, Palette]:
    doc = doc or EnvDocument()
    c = EditorCanvas(doc)
    qtbot.addWidget(c)
    c.resize(900, 600)
    c.show()
    c.fit()  # 처음 맞춤을 먼저 끝낸다(뒤의 보기 설정을 덮어쓰지 않게)
    c.center, c.scale = (0.0, 1.0), 40.0
    c.set_snap(True, 0.5)
    c.picker_factory = FakePicker
    pal = Palette()
    qtbot.addWidget(pal)
    pal.connect_canvas(c)
    FakePicker.calls = []
    FakePicker.answer = None
    return c, pal


def click_at(c: EditorCanvas, x: float, y: float, button: Qt.MouseButton = LEFT) -> None:
    p = c.world_to_screen(x, y)
    QTest.mouseClick(c, button, NO, QPoint(round(p.x()), round(p.y())))


def test_template_buttons_are_plain_buttons_outside_tool_ids(qtbot: Any) -> None:
    pal = Palette()
    qtbot.addWidget(pal)
    assert len(TOOL_IDS) == 35 and TOOL_IDS == T.TOOL_IDS
    assert "move" in TOOL_IDS and TOOL_IDS[-1] != "move"  # 새 도구는 구동기 묶음 안
    assert ("actuators", ["thruster", "fuel_tank", "vehicle", "motor", "move"]) in T.TOOL_GROUPS
    for bid, kind in (("tpl_body", "body"), ("tpl_prop", "prop")):
        assert bid not in TOOL_IDS and T.TEMPLATE_BUTTONS[bid] == kind
        b = pal.buttons[bid]
        assert not b.isCheckable() and b not in pal.group.buttons()
        assert b.text() == T.TEMPLATE_LABELS[bid] and b.toolTip() == T.TEMPLATE_HINTS[bid]
        assert not b.icon().isNull()
    assert pal.buttons["tpl_body"].text() == S.EDC_TOOL_TPL_BODY
    assert pal.buttons["tpl_prop"].text() == S.EDC_TOOL_TPL_PROP
    # 「틀」 묶음이 맨 앞(선택 도구 위)에 있다
    lay = pal.inner.layout()
    widgets = [lay.itemAt(i).widget() for i in range(lay.count()) if lay.itemAt(i).widget() is not None]
    assert widgets.index(pal.buttons["tpl_body"]) < widgets.index(pal.buttons["select"])
    got: list[str] = []
    pal.template_requested.connect(got.append)
    pal.buttons["tpl_prop"].click()
    pal.buttons["tpl_body"].click()
    assert got == ["prop", "body"] and pal.tool_id == "select"
    pal.set_tool("tpl_body")  # 도구가 아니므로 켜지지 않는다
    assert pal.buttons["select"].isChecked()


def test_picked_template_goes_in_as_one_undo_step(qtbot: Any) -> None:
    c, pal = make(qtbot)
    doc = c.doc
    lib = default_library()
    before = copy.deepcopy(doc.data)
    n0 = doc.undo_stack.count()
    FakePicker.answer = (lib.get("ground_flat"), {}, False, ())  # 4번째(진화할 슬라이더)가 와도 받는다
    pal.buttons["tpl_prop"].click()
    assert FakePicker.calls == [("prop", "side")]
    assert c.template_drop is not None and c.status_text == S.TPL_INSERT_HINT.format(name=lib.get("ground_flat").name)
    assert doc.undo_stack.count() == n0  # 고르기만으로는 문서가 바뀌지 않는다
    click_at(c, 2.2, 0.1)
    assert c.template_drop is None
    assert doc.undo_stack.count() == n0 + 1
    ground = doc.item(("bodies", "ground"))
    assert ground is not None and ground["pos"][0] == 2.0  # 누른 자리(격자 0.5에 맞춤)가 틀의 (0, 0)
    assert doc.undo_stack.undoText() == S.TPL_UNDO_INSERT.format(name=lib.get("ground_flat").name)
    assert c.status_text.startswith(S.TPL_INSERTED.format(name=lib.get("ground_flat").name, bodies=1, joints=0,
                                                          actuators=0, sensors=0))
    after = copy.deepcopy(doc.data)
    doc.undo_stack.undo()
    assert doc.data == before
    doc.undo_stack.redo()
    assert doc.data == after
    # 몸 틀: 땅이 있으면 빈자리가 그 태그로 풀리고, 검사 오류 없이 컴파일된다. 뿌리 물체가 골라진다
    FakePicker.answer = (lib.get("biped"), lib.get("biped").defaults(), False)
    pal.buttons["tpl_body"].click()
    click_at(c, 0.0, 0.0)
    assert doc.undo_stack.count() == n0 + 2
    assert doc.selection == [("bodies", "torso")]
    contacts = [s for s in doc.items("sensors") if s.get("kind") == "contact"]
    assert contacts and all(s["with"] == ["tag.ground"] for s in contacts)
    assert not [i for i in doc.run_check() if i.severity == "error"]
    assert doc.spec() is not None


def test_same_template_twice_gets_one_number_and_root_is_selected(qtbot: Any) -> None:
    c, _pal = make(qtbot)
    lib = default_library()
    first = T.insert_template(c.doc, lib.get("ball"), at=(1.0, 2.0))
    second = T.insert_template(c.doc, lib.get("ball"), at=(3.0, 2.0))
    assert [r[1] + "_2" for r in first] == [r[1] for r in second]
    assert c.doc.selection == [second[0]]
    biped = T.insert_template(c.doc, lib.get("biped"))
    again = T.insert_template(c.doc, lib.get("biped"), mirror=True)
    assert ("bodies", "torso") in biped and c.doc.selection == [("bodies", "torso_2")] and len(again) == len(biped)


def test_cancel_paths_leave_the_document_alone(qtbot: Any) -> None:
    c, pal = make(qtbot)
    doc = c.doc
    lib = default_library()
    n0 = doc.undo_stack.count()
    FakePicker.answer = None  # 창에서 취소
    pal.buttons["tpl_prop"].click()
    assert c.template_drop is None
    FakePicker.answer = (lib.get("wall"), {}, False)
    pal.buttons["tpl_prop"].click()
    QTest.keyClick(c, Qt.Key.Key_Escape)  # Esc: 놓기를 그만둔다(선택 도구는 그대로)
    assert c.template_drop is None and c.tool_id == "select"
    pal.buttons["tpl_prop"].click()
    click_at(c, 0.0, 0.0, Qt.MouseButton.RightButton)  # 오른쪽 단추도 그만둔다
    assert c.template_drop is None
    pal.buttons["tpl_prop"].click()
    pal.buttons["box"].click()  # 다른 도구를 고르면 그만둔다
    assert c.template_drop is None and c.tool_id == "box"
    assert doc.undo_stack.count() == n0 and not doc.items("bodies")


def test_world_kind_reaches_the_picker_and_missing_slot_is_named(qtbot: Any) -> None:
    doc = EnvDocument()
    doc.set_value(("world", "view"), "top")
    doc.set_value(("world", "gravity"), [0.0, 0.0])
    c, pal = make(qtbot, doc)
    lib = default_library()
    FakePicker.answer = (lib.get("disc"), {}, False)
    pal.buttons["tpl_body"].click()
    assert FakePicker.calls == [("body", "top")]
    click_at(c, 0.0, 0.0)
    assert any(a.get("kind") == "move" for a in doc.items("actuators"))
    # 땅이 없는 장면에 두 발 막대 사람: 넣기는 되고, 무엇이 더 있어야 하는지 상태 줄에
    c2, _ = make(qtbot)
    FakePicker.answer = (lib.get("biped"), {}, False)
    c2.pick_template("body")
    click_at(c2, 0.0, 0.0)
    slot = lib.get("biped").slots["$ground"]
    assert S.TPL_NEEDS.format(what=slot.label) in c2.status_text


def test_ghost_outline_paints_while_waiting(qtbot: Any) -> None:
    c, _pal = make(qtbot)
    c.begin_template_drop(default_library().get("biped"))
    c.cursor_world = (1.0, 0.0)
    before = c.grab().toImage()
    c.cursor_world = None
    without = c.grab().toImage()
    assert before != without  # 커서 자리에 윤곽이 그려진다


def test_every_builtin_template_inserts_and_undoes_exactly(qtbot: Any) -> None:
    """내장 틀 전부: 넣기 = 되돌리기 한 단계, 되돌리면 사전이 같음, 넣기 + 검사 시간(M7의 넣기 몫, 보고는 재는 스크립트)."""
    lib = default_library()
    times: list[float] = []
    for tpl in lib.all():
        doc = EnvDocument()
        if tpl.world == "top":
            doc.set_value(("world", "view"), "top")
            doc.set_value(("world", "gravity"), [0.0, 0.0])
        if tpl.kind == "body":
            T.insert_template(doc, lib.get("yard_top" if tpl.world == "top" else "ground_flat"))
        before = copy.deepcopy(doc.data)
        n0 = doc.undo_stack.count()
        t0 = time.perf_counter()
        refs = T.insert_template(doc, tpl, tpl.defaults(), at=(0.5, 0.0))
        issues = doc.run_check()
        times.append(time.perf_counter() - t0)
        assert refs and doc.undo_stack.count() == n0 + 1, tpl.id
        assert not [i for i in issues if i.severity == "error"], (tpl.id, issues)
        doc.undo_stack.undo()
        assert doc.data == before, tpl.id
    assert statistics.median(times) < 0.5  # 목표 20ms(M7). CI 여유로 크게: 값은 보고서

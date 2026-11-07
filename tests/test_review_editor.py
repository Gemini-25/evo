"""리뷰 수정(환경 에디터): 다시 열 수 없는 파일 저장, 붙여넣기, 저장 전 덜 쓴 칸, 진화 끄기, 이름 바꾸기."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLineEdit, QMessageBox, QWidget

from evolab.app import strings as S
from evolab.app.editor.canvas import build_scene
from evolab.app.editor.document import EnvDocument
from evolab.app.editor.forms import (
    IdField,
    NumField,
    RefField,
    TypeInfo,
    id_renamer,
    param_uses,
    parse_list,
    parse_number,
    rename,
    rename_error,
)
from evolab.app.editor.properties import PropertyPanel
from evolab.app.editor.tabs import _generator_item
from evolab.app.editor.view import EditorView
from evolab.spec.compiled import EnvFileError
from evolab.spec.io import compile_env, load_env, parse_env


@pytest.fixture
def lander(qtbot: Any) -> EnvDocument:
    doc = EnvDocument()
    doc.from_builtin("lander.evoenv")
    return doc


@pytest.fixture
def boxes(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """대화 상자를 띄우지 않고 기록한다. question은 answer[0]으로 답한다(기본: 취소)."""
    calls: dict[str, list[Any]] = {"warning": [], "question": [], "answer": [QMessageBox.StandardButton.Cancel]}

    def warning(_parent: Any, title: str, text: str, *_a: Any, **_k: Any) -> Any:
        calls["warning"].append((title, text))
        return QMessageBox.StandardButton.Ok

    def question(_parent: Any, title: str, text: str, *_a: Any, **_k: Any) -> Any:
        calls["question"].append((title, text))
        return calls["answer"][0]

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(warning))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return calls


def _view(qtbot: Any) -> EditorView:
    view = EditorView()
    view.resize(1400, 900)
    qtbot.addWidget(view)
    view.show()
    return view


def _focus(qtbot: Any, w: QWidget) -> None:
    win = w.window()
    win.show()
    win.activateWindow()
    w.setFocus()
    qtbot.waitUntil(w.hasFocus, timeout=2000)


def _type(qtbot: Any, edit: QLineEdit, text: str, finish: bool = True) -> None:
    _focus(qtbot, edit)
    edit.selectAll()
    QTest.keyClicks(edit, text)
    if finish:
        QTest.keyClick(edit, Qt.Key.Key_Return)


def _errors(doc: EnvDocument) -> list[str]:
    return [f"{i.where}: {i.message}" for i in doc.run_check() if i.severity == "error"]


def _saved_lander(view: EditorView, tmp_path: Path) -> Path:
    view.start_from_builtin("lander.evoenv")
    out = tmp_path / "착륙선.evoenv"
    assert view.save_as(out)
    return out


# ---------------------------------------------------------------------------
# 6. 구조 오류가 있는 문서를 저장하면 다시 열 수 없다
# ---------------------------------------------------------------------------


def test_plain_number_out_of_range_is_not_written(qtbot: Any, boxes: dict[str, list[Any]]) -> None:
    view = _view(qtbot)
    view.start_from_builtin("lander.evoenv")
    view.tabs.setCurrentWidget(view.tabs.world_tab)
    dt = view.tabs.world_tab.world_form.field("dt")
    assert isinstance(dt, NumField)
    before = view.doc.get(("world", "dt"))
    _type(qtbot, dt.editor, "0.2")  # type: ignore[arg-type]  # 모델: 0 < dt <= 0.1
    assert view.doc.get(("world", "dt")) == before and not view.doc.dirty
    assert dt.invalid is not None and "0.1" in dt.invalid and S.RVE_RANGE_VALUE in dt.invalid
    _type(qtbot, dt.editor, "0.05")  # type: ignore[arg-type]
    assert view.doc.get(("world", "dt")) == 0.05 and dt.invalid is None
    view.tabs.setCurrentWidget(view.tabs.events_tab)
    view.doc.select([("events", "landed")])
    hold = view.tabs.events_tab.events.cell(0, "hold")  # 사건 hold(>= 0)도 같다
    assert isinstance(hold, NumField)
    held = view.doc.get(("events", 0, "hold"))
    _type(qtbot, hold.editor, "-3")  # type: ignore[arg-type]
    assert view.doc.get(("events", 0, "hold")) == held and hold.invalid is not None
    assert not _errors(view.doc)


def test_parse_number_and_list_check_model_range() -> None:
    assert parse_number("0", low=0.0, low_open=True)[0] is False
    assert parse_number("0", low=0.0)[:2] == (True, 0.0)
    assert parse_number("0.1", high=0.1)[0] and not parse_number("0.1", high=0.1, high_open=True)[0]
    ok, value, err = parse_number("=p.x * 2", low=0.0, low_open=True)  # 수식은 모델처럼 범위를 보지 않는다
    assert ok and value == "=p.x * 2" and err is None
    ok, _v, err = parse_number("-3", formula=False, low=0.0)
    assert not ok and err == S.RVE_OUT_OF_RANGE.format(range=f"0.0 ≤ {S.RVE_RANGE_VALUE}")
    floats = TypeInfo("list", item=TypeInfo("float", low=0.0, low_open=True))
    assert parse_list(floats, "1, 2")[:2] == (True, [1.0, 2.0])
    assert parse_list(floats, "1, 0")[0] is False


def test_save_refuses_document_it_could_not_reopen(qtbot: Any, tmp_path: Path, boxes: dict[str, list[Any]]) -> None:
    view = _view(qtbot)
    out = _saved_lander(view, tmp_path)
    good = out.read_text(encoding="utf-8")
    view.doc.set_value(("world", "dt"), 0.5)  # 붙여넣기 등 다른 길로 생긴 구조 오류
    assert view.doc.dirty
    with pytest.raises(EnvFileError):
        view.doc.save(out)  # 문서 쪽에서도 쓰지 않는다
    assert not view.save() and not view.save_as(tmp_path / "other.evoenv")
    assert not (tmp_path / "other.evoenv").exists()
    assert boxes["warning"] and all(title == S.ED_SAVE_FAILED and "world.dt" in text for title, text in boxes["warning"])
    assert out.read_text(encoding="utf-8") == good and view.doc.dirty
    boxes["answer"][0] = QMessageBox.StandardButton.Save  # 창을 닫을 때 '저장'을 골라도 덮어쓰지 않고 닫지 않는다
    assert not view.maybe_save()
    assert out.read_text(encoding="utf-8") == good
    assert view.doc.dirty and view.doc.get(("world", "dt")) == 0.5
    view.doc.undo_stack.undo()
    assert view.save() and load_env(out)


def test_reference_errors_still_save_and_reopen(qtbot: Any, tmp_path: Path, boxes: dict[str, list[Any]]) -> None:
    view = _view(qtbot)
    out = _saved_lander(view, tmp_path)
    view.doc.remove([("bodies", "leg_l")])  # 관절·센서가 없는 물체를 가리킨다(구조는 맞다)
    assert _errors(view.doc)
    assert view.save() and not boxes["warning"]
    assert view.open_file(out) and _errors(view.doc)


# ---------------------------------------------------------------------------
# 7. 붙여넣기: 종류·태그는 그대로, 관절 없이 붙인 물체는 place_by를 지운다
# ---------------------------------------------------------------------------


def test_paste_keeps_kind_and_tags_equal_to_id(qtbot: Any, lander: EnvDocument, tmp_path: Path) -> None:
    refs = lander.paste_text(lander.copy_text([("sensors", "fuel")]))
    assert refs == [("sensors", "fuel_1")]
    assert lander.item(refs[0])["kind"] == "fuel"  # type: ignore[index]
    refs = lander.paste_text(lander.copy_text([("bodies", "hull")]))
    assert lander.item(refs[0])["tags"] == ["hull"]  # type: ignore[index]
    assert not _errors(lander)
    out = tmp_path / "붙임.evoenv"
    lander.save(out)
    EnvDocument().open(out)  # 저장한 파일을 다시 열 수 있다
    doc = EnvDocument()
    doc.add("generators", _generator_item(doc, "hills"))  # 생성기 탭이 만드는 id == kind
    refs = doc.paste_text(doc.copy_text([("generators", "hills")]))
    assert doc.item(refs[0])["kind"] == "hills" and parse_env(doc.data)  # type: ignore[index]


def test_paste_placed_body_without_its_joint(qtbot: Any, lander: EnvDocument) -> None:
    before = build_scene(lander).body("leg_l")
    assert before is not None
    refs = lander.paste_text(lander.copy_text([("bodies", "leg_l")]))
    copy = lander.item(refs[0])
    assert copy is not None and "place_by" not in copy
    assert not _errors(lander)
    scene = build_scene(lander)
    after = scene.body(refs[0][1])
    assert after is not None
    a = scene.spawn[2]
    dx, dy = math.cos(a) * 1.0 + math.sin(a) * 1.0, math.sin(a) * 1.0 - math.cos(a) * 1.0  # (1, -1)을 시작 자세로 돌림
    assert after.pose[0] == pytest.approx(before.pose[0] + dx, abs=1e-4)
    assert after.pose[1] == pytest.approx(before.pose[1] + dy, abs=1e-4)
    assert after.pose[2] == pytest.approx(before.pose[2], abs=1e-4)
    # 관절과 함께 붙이면 새 관절로 놓인다
    refs = lander.paste_text(lander.copy_text([("bodies", "leg_r"), ("joints", "j_leg_r")]))
    assert lander.item(refs[0])["place_by"] == refs[1][1]  # type: ignore[index]
    assert not _errors(lander)


def test_paste_relinks_formulas_between_pasted_parts(qtbot: Any, lander: EnvDocument) -> None:
    refs = lander.paste_text(lander.copy_text([("bodies", "hull"), ("events", "crash")]))
    when = lander.item(refs[1])["when"]  # type: ignore[index]
    assert "contact(hull_1, tag.terrain)" in when and "hull_1.speed" in when and "contact(leg_l, tag.terrain)" in when


def test_id_renamer_is_simultaneous_and_key_aware() -> None:
    fix = id_renamer({("bodies", "leg"): "leg_1", ("bodies", "leg_1"): "leg_2", ("params", "w"): "w_1"})
    item = {"id": "leg", "kind": "leg", "tags": ["leg"], "label": "leg", "body": "leg_1", "a": "leg.x",
            "when": "contact(leg, leg_1) and p.w > 1 and min(leg.x, 2) > 0", "h": "=p.w * 2"}
    out = fix(item, None)
    assert out["id"] == "leg" and out["kind"] == "leg" and out["tags"] == ["leg"] and out["label"] == "leg"
    assert out["body"] == "leg_2" and out["a"] == "leg_1.x" and out["h"] == "=p.w_1 * 2"
    assert out["when"] == "contact(leg_1, leg_2) and p.w_1 > 1 and min(leg_1.x, 2) > 0"


# ---------------------------------------------------------------------------
# 8. 저장·닫기 전에 덜 쓴 참조·이름 칸을 문서에 쓴다
# ---------------------------------------------------------------------------


def test_save_commits_text_still_typed_in_ref_and_id_fields(qtbot: Any, tmp_path: Path,
                                                            boxes: dict[str, list[Any]]) -> None:
    view = _view(qtbot)
    out = _saved_lander(view, tmp_path)
    view.doc.select([("sensors", "st")])
    body = view.props.form.field("body")  # type: ignore[union-attr]
    assert isinstance(body, RefField)
    _type(qtbot, body.editor.lineEdit(), "leg_l", finish=False)  # type: ignore[union-attr]
    assert not view.doc.dirty  # Enter도 안 눌렀고 칸도 안 떠났다
    view.act_save.trigger()  # 도구 모음 단추·Ctrl+S는 포커스를 옮기지 않는다
    i = view.doc.find(("sensors", "st"))
    assert json.loads(out.read_text(encoding="utf-8"))["sensors"][i]["body"] == "leg_l"
    assert not view.doc.dirty and "*" not in view.doc.title()
    idf = view.props.id_field
    assert isinstance(idf, IdField)
    _type(qtbot, idf.editor, "state", finish=False)  # type: ignore[arg-type]
    view.act_save.trigger()
    assert [s["id"] for s in json.loads(out.read_text(encoding="utf-8"))["sensors"]][i] == "state"
    assert not view.doc.dirty


def test_close_asks_when_only_typed_text_is_unsaved(qtbot: Any, tmp_path: Path, boxes: dict[str, list[Any]]) -> None:
    view = _view(qtbot)
    _saved_lander(view, tmp_path)
    view.doc.select([("sensors", "st")])
    body = view.props.form.field("body")  # type: ignore[union-attr]
    assert isinstance(body, RefField)
    _type(qtbot, body.editor.lineEdit(), "leg_l", finish=False)  # type: ignore[union-attr]
    assert not view.maybe_save()  # 물어보고(취소) 닫지 않는다
    assert len(boxes["question"]) == 1 and view.doc.dirty


# ---------------------------------------------------------------------------
# 9. 진화 끄기: 이름 단위로 '아직 쓰나'를 본다
# ---------------------------------------------------------------------------


def _two_shapes(doc: EnvDocument) -> int:
    i = doc.find(("bodies", "hull"))
    assert i is not None
    doc.edit("", lambda d: d["bodies"][i]["shapes"].append({"kind": "box", "w": 1.0, "h": 1.0}))
    return i


def test_evolve_off_removes_param_whose_name_is_a_prefix_of_another(qtbot: Any, lander: EnvDocument) -> None:
    i = _two_shapes(lander)
    w0, w1 = ("bodies", i, "shapes", 0, "w"), ("bodies", i, "shapes", 1, "w")
    assert lander.make_evolvable(w0, 1.0, 3.0, name="hull_w") == "hull_w"
    assert lander.make_evolvable(w1, 0.5, 1.5, name="hull_w") == "hull_w_1"
    assert param_uses(lander.data, "hull_w") == 1  # p.hull_w_1은 p.hull_w를 쓰는 것이 아니다
    lander.make_fixed(w0)
    assert lander.get(w0) == 2.0 and lander.item(("params", "hull_w")) is None
    assert compile_env(lander.data).body_param_count == 1
    # 다른 수식이 정말로 쓰면 매개변수는 남는다
    lander.make_evolvable(w0, 1.0, 3.0, name="hull_w")
    lander.set_value(("bodies", i, "shapes", 1, "h"), "=p.hull_w / 2")
    lander.make_fixed(w0)
    assert lander.item(("params", "hull_w")) is not None


def test_evolve_checkbox_off_and_on_keeps_gene_count(qtbot: Any, lander: EnvDocument) -> None:
    i = _two_shapes(lander)
    panel = PropertyPanel(lander)
    panel.resize(380, 700)
    qtbot.addWidget(panel)
    panel.show()
    lander.select([("bodies", "hull")])
    boxes = [panel.form.field("shapes", k, "w") for k in (0, 1)]
    for f in boxes:
        assert isinstance(f, NumField) and f.evolve_box is not None
        f.evolve_box.click()
    assert compile_env(lander.data).body_param_count == 2
    boxes[0].evolve_box.click()  # type: ignore[union-attr]
    assert compile_env(lander.data).body_param_count == 1
    assert [p["id"] for p in lander.items("params") if p.get("evolve")] == ["hull_w_1"]
    boxes[0].evolve_box.click()  # type: ignore[union-attr]
    assert compile_env(lander.data).body_param_count == 2
    assert lander.get(("bodies", i, "shapes", 0, "w")) == "=p.hull_w"


# ---------------------------------------------------------------------------
# 10. 이름 바꾸기: 모델과 같은 id 규칙, 이름이 아닌 글자는 건드리지 않는다
# ---------------------------------------------------------------------------


def test_rename_rejects_reserved_and_accepts_korean_ids(qtbot: Any, lander: EnvDocument) -> None:
    ref = ("bodies", "hull")
    for bad in ("time", "p", "tag", "dist", "success", "timeout", "a__b", "9x", "x" * 41):
        assert rename_error(lander, ref, bad) is not None, bad
        assert not rename(lander, ref, bad)
    assert rename_error(lander, ref, "몸통") is None
    assert rename(lander, ref, "몸통") and not _errors(lander)
    assert lander.unique_id("time") == "time_1"  # 예약된 이름은 피한다
    assert lander.unique_id("다리") == "다리"
    assert lander.unique_id("a" * 45) == "a" * 40
    refs = lander.paste_text(lander.copy_text([("bodies", "몸통")]))
    assert refs == [("bodies", "몸통_1")]
    lander.add("bodies", {"id": "b" * 40, "owner": "world", "type": "static",
                          "shapes": [{"kind": "box", "w": 1.0, "h": 1.0}]})
    refs = lander.paste_text(lander.copy_text([("bodies", "b" * 40)]))
    assert len(refs[0][1]) <= 40 and refs[0][1] != "b" * 40 and parse_env(lander.data)


def test_rename_matches_whole_names_including_korean(qtbot: Any, lander: EnvDocument) -> None:
    assert rename(lander, ("bodies", "leg_l"), "앞다리") and rename(lander, ("bodies", "leg_r"), "다리")
    assert rename(lander, ("bodies", "다리"), "leg")
    when = lander.item(("events", "landed"))["when"]  # type: ignore[index]
    assert "contact(앞다리, tag.pad)" in when and "contact(leg, tag.pad)" in when
    assert not _errors(lander)
    # 함수 이름과 같은 부품 id: 함수 호출은 그대로
    assert rename(lander, ("bodies", "hull"), "abs")
    lander.edit("", lambda d: d["events"].append({"id": "tilt", "when": "abs(abs.angle) > 1"}))
    assert rename(lander, ("bodies", "abs"), "hull")
    assert lander.item(("events", "tilt"))["when"] == "abs(hull.angle) > 1"  # type: ignore[index]
    assert not _errors(lander)


def test_renaming_an_existing_reserved_id_leaves_builtin_symbols(qtbot: Any, lander: EnvDocument) -> None:
    # 파일에서 연 문서에 이미 예약된 id가 있다: 물체 'time'(참조 칸은 이 물체를 가리킨다)
    def mutate(d: dict[str, Any]) -> None:
        text = json.dumps(d).replace('"hull"', '"time"')
        d.clear()
        d.update(json.loads(text))
        d["events"].append({"id": "late", "when": "time > 10"})

    lander.edit("", mutate)
    assert any("time" in e for e in _errors(lander))
    assert rename(lander, ("bodies", "time"), "body")
    assert lander.item(("events", "late"))["when"] == "time > 10"  # type: ignore[index]  # 내장 time은 그대로
    assert [j["a"] for j in lander.items("joints")] == ["body", "body"]
    assert lander.item(("sensors", "st"))["body"] == "body"  # type: ignore[index]
    # 내장 사건과 같은 이름의 사건: 종료 조건의 내장 'timeout'은 그 사건이 아니다
    lander.edit("", lambda d: d["events"][0].__setitem__("id", "timeout"))
    assert rename(lander, ("events", "timeout"), "touchdown")
    assert [t["event"] for t in lander.items("termination")][-2:] == ["out_of_bounds", "timeout"]

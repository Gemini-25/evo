"""6.2.0 에디터 뜻 검사와 고치기 (설계 6.1 T31·T32, D40, DU 4.3.6): document.editor_checks, editor/fixes.py, 문제 목록 탭.

- T31: 감사의 여섯 상태(tests/data/audit/, 만든 법은 tests/data/audit/make_audit.py)를 저마다 잡고, 고치기 한 번에 사라지며 되돌리면 다시
  뜬다. 내장 예제·내장 조리법·tests/data 환경(audit/와 키 없는 presets/running_measured.evoenv는 빼고)에서 어떤 심각도의 항목도 0
  (예외는 이름으로: 글라이더의 엔진 안내 brainless). 엔진 검사 코드의 고치기.
- T32: 문제 목록의 줄·고치기 단추·키보드로 단추에 닿음, 상태 줄이 오류·경고·살펴볼 것을 제 이름으로 셈.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QToolButton

from evolab import paths
from evolab.app import strings as S
from evolab.app.editor.document import EDITOR_CODES, EDITOR_LINTS, IGNORED_KEY, EnvDocument, editor_checks
from evolab.app.editor.fixes import FIX_CODES, OPEN_RULES_ADD, OPEN_TEMPLATE_BODY, Fix, fixes_for, ignore, where_to_path
from evolab.app.editor.tabs import BottomTabs, IssuesTab
from evolab.app.editor.view import EditorView
from evolab.presets import recipe as RC
from evolab.spec.compiled import EnvFileError, check_env
from evolab.spec.io import compile_env, env_hash, parse_env

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "tests" / "data" / "audit"
SIX = ("ground_is_body", "score_far_is_good", "no_keys", "motor_twice", "joint_apart", "never_stops")
LINT_EXCEPTIONS = {("glider.evoenv", "brainless")}
"""이름으로 둔 예외: 글라이더는 뇌 없는 환경이라 엔진이 안내(brainless) 하나를 낸다(뜻 검사가 아님)."""
SKIP_DATA = ("audit", "presets/running_measured.evoenv")
"""T31의 「tests/data 환경」에서 빼는 것: 일부러 문제를 넣은 audit/, 키가 없는 측정 원형 파일."""


def _doc(qtbot: Any, text: str) -> EnvDocument:
    doc = EnvDocument()
    doc.load_text(text, None)
    return doc


def _audit(qtbot: Any, code: str) -> EnvDocument:
    return _doc(qtbot, (AUDIT / f"{code}.evoenv").read_text(encoding="utf-8"))


def _codes(doc: EnvDocument) -> list[str]:
    return [i.code for i in doc.run_check()]


def _issue(doc: EnvDocument, code: str) -> Any:
    found = [i for i in doc.run_check() if i.code == code]
    assert found, (code, [(i.code, i.message) for i in doc.issues])
    return found[0]


def _apply(doc: EnvDocument, fix: Fix) -> int:
    """고치기 하나를 하고 되돌리기 단계가 몇 개 늘었는지."""
    before = doc.undo_stack.index()
    fix.apply(doc)
    return doc.undo_stack.index() - before


# ---------------------------------------------------------------------------
# T31: 감사의 여섯 상태
# ---------------------------------------------------------------------------


def test_audit_files_exist_and_lints_are_the_six() -> None:
    assert EDITOR_LINTS == SIX
    assert set(EDITOR_CODES) == {"unattached", "overlap", *SIX}
    assert sorted(p.stem for p in AUDIT.glob("*.evoenv")) == sorted(SIX)
    for code in SIX:  # 견본은 구조·참조 오류가 없는 형식 1 파일(엔진 검사 글 0)
        text = (AUDIT / f"{code}.evoenv").read_text(encoding="utf-8")
        assert check_env(parse_env(text)) == [] and json.loads(text)["version"] == 1


@pytest.mark.parametrize("code", SIX)
def test_audit_state_is_caught_fixed_in_one_step_and_undo_brings_it_back(qtbot: Any, code: str) -> None:
    doc = _audit(qtbot, code)
    issue = _issue(doc, code)
    assert issue.message and issue.severity in ("warning", "info")
    fixes = fixes_for(doc, issue)
    assert fixes and all(f.label and f.tip for f in fixes)
    original = json.dumps(doc.data, sort_keys=True)
    assert _apply(doc, fixes[0]) == 1  # 되돌리기 한 단계
    assert code not in _codes(doc)
    assert not [i for i in doc.issues if i.severity == "error"]
    doc.undo_stack.undo()
    assert json.dumps(doc.data, sort_keys=True) == original
    assert code in _codes(doc)


def test_audit_messages_use_display_names_and_particles(qtbot: Any) -> None:
    doc = _audit(qtbot, "score_far_is_good")
    assert _issue(doc, "score_far_is_good").message == "지금은 「몸통」이 「골」에서 멀수록 점수가 오릅니다."
    doc = _audit(qtbot, "never_stops")
    assert _issue(doc, "never_stops").message == "「공」은 한번 밀리면 멈추지 않습니다(바닥 마찰이 없습니다)."
    doc = _audit(qtbot, "joint_apart")
    assert _issue(doc, "joint_apart").message == S.ED_CHECK_JOINT_APART.format(joint="엉덩이", gap=0.25)
    doc = _audit(qtbot, "no_keys")
    assert _issue(doc, "no_keys").message == S.ED_CHECK_NO_KEYS.format(n=3)


def test_fix_results_are_what_the_label_says(qtbot: Any) -> None:
    doc = _audit(qtbot, "ground_is_body")
    fixes_for(doc, _issue(doc, "ground_is_body"))[0].apply(doc)
    ground = doc.item(("bodies", "ground"))
    assert ground is not None and ground["owner"] == "world" and ground["type"] == "static" and "ground" in ground["tags"]
    doc = _audit(qtbot, "score_far_is_good")
    fixes_for(doc, _issue(doc, "score_far_is_good"))[0].apply(doc)
    assert doc.item(("scoring.blocks", "s_near"))["weight"] == -1.0  # type: ignore[index]
    doc = _audit(qtbot, "no_keys")
    fixes_for(doc, _issue(doc, "no_keys"))[0].apply(doc)
    keys = [k for a in doc.items("actuators") for k in a.get("keys_pos", []) + a.get("keys_neg", [])]
    assert len(keys) == 6 and len(set(keys)) == 6 and not set(keys) & {"R", "N", "Space", "Escape"}
    doc = _audit(qtbot, "motor_twice")
    fixes_for(doc, _issue(doc, "motor_twice"))[0].apply(doc)
    assert [a["id"] for a in doc.items("actuators")] == ["m_hip"]  # 나중에 단 모터를 지움
    doc = _audit(qtbot, "joint_apart")
    leg_before = json.dumps(doc.item(("bodies", "leg")), sort_keys=True)
    fixes_for(doc, _issue(doc, "joint_apart"))[0].apply(doc)
    assert json.dumps(doc.item(("bodies", "leg")), sort_keys=True) == leg_before  # 물체는 움직이지 않음
    assert doc.item(("joints", "hip"))["anchor_b"] == [-0.25, 0.25]  # type: ignore[index]
    doc = _audit(qtbot, "never_stops")
    fixes_for(doc, _issue(doc, "never_stops"))[0].apply(doc)
    assert doc.data["world"]["damping"] == 0.5


def test_ignore_hides_an_info_lint_outside_the_hash(qtbot: Any) -> None:
    doc = _audit(qtbot, "score_far_is_good")
    issue = _issue(doc, "score_far_is_good")
    before = env_hash(parse_env(doc.data))
    assert ignore(doc, issue)
    assert "score_far_is_good" not in _codes(doc)
    assert doc.data["editor"][IGNORED_KEY] == ["score_far_is_good@scoring.blocks[1]"]
    assert env_hash(parse_env(doc.data)) == before
    doc.undo_stack.undo()
    assert "score_far_is_good" in _codes(doc)
    # 경고는 숨길 수 없다
    doc = _audit(qtbot, "motor_twice")
    assert not ignore(doc, _issue(doc, "motor_twice"))


# ---------------------------------------------------------------------------
# T31: 내장 것에서 거짓 경보 0
# ---------------------------------------------------------------------------


def _data_envs() -> list[Path]:
    base = ROOT / "tests" / "data"
    out = []
    for p in sorted(base.rglob("*.evoenv")):
        rel = p.relative_to(base).as_posix()
        if any(rel == s or rel.startswith(s + "/") for s in SKIP_DATA):
            continue
        out.append(p)
    return out


def _all_issues(text: str) -> list[Any]:
    env = parse_env(text)
    try:
        spec = compile_env(json.loads(text))
    except EnvFileError:
        spec = None
    return check_env(env) + editor_checks(env, spec)


def test_builtins_recipes_and_data_envs_have_no_items_of_any_severity() -> None:
    builtins = paths.builtin_envs()
    assert len(builtins) >= 7  # 옛 여섯 + 달리기
    seen: list[str] = []
    for p in builtins:
        got = [(p.name, i.code) for i in _all_issues(p.read_text(encoding="utf-8-sig"))]
        assert set(got) <= LINT_EXCEPTIONS, got
        seen.append(p.name)
    assert "glider.evoenv" in seen
    for name in RC.builtin_names():
        data = RC.build(RC.load_builtin(name))
        got = [(name, i.code, i.message) for i in _all_issues(json.dumps(data, ensure_ascii=False))]
        assert got == []
    data_envs = _data_envs()
    assert data_envs
    for p in data_envs:
        got = [(p.name, i.code, i.message) for i in _all_issues(p.read_text(encoding="utf-8-sig"))]
        assert got == []


def test_measured_running_file_is_excluded_only_for_its_missing_keys() -> None:
    text = (ROOT / "tests" / "data" / "presets" / "running_measured.evoenv").read_text(encoding="utf-8")
    assert [i.code for i in _all_issues(text)] == ["no_keys"]


# ---------------------------------------------------------------------------
# 엔진 검사 코드의 고치기
# ---------------------------------------------------------------------------


def _scene(**extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "version": 1, "name": "고치기",
        "world": {"gravity": [0.0, -9.81], "max_time": 5.0},
        "bodies": [
            {"id": "ground", "label": "땅", "owner": "world", "type": "static", "shapes": [{"kind": "box", "w": 20, "h": 0.4}],
             "tags": ["ground"]},
            {"id": "torso", "label": "몸통", "pos": [0, 1.0], "shapes": [{"kind": "box", "w": 0.4, "h": 0.4}]},
        ],
        "actuators": [{"kind": "thruster", "id": "up", "body": "torso", "max_force": 10.0, "keys": ["Up"]}],
        "sensors": [{"kind": "body_state", "id": "st", "body": "torso", "fields": ["vx"]}],
        "termination": [{"event": "timeout", "outcome": "neutral"}],
        "scoring": {"blocks": [{"kind": "survival_time", "id": "s"}]},
    }
    data.update(extra)
    return data


def test_no_sensors_offers_template_and_state_sensor(qtbot: Any) -> None:
    doc = _doc(qtbot, json.dumps(_scene(sensors=[]), ensure_ascii=False))
    fixes = fixes_for(doc, _issue(doc, "no_sensors"))
    assert [f.key for f in fixes] == ["add_body_template", "add_state_sensor"]
    asked: list[str] = []
    doc.fix_request.connect(asked.append)
    assert _apply(doc, fixes[0]) == 0 and asked == [OPEN_TEMPLATE_BODY] and fixes[0].opens == OPEN_TEMPLATE_BODY
    assert _apply(doc, fixes[1]) == 1
    assert "no_sensors" not in _codes(doc) and doc.items("sensors")[0]["body"] == "torso"


def test_no_actuators_and_no_scoring_open_screens(qtbot: Any) -> None:
    doc = _doc(qtbot, json.dumps(_scene(actuators=[], scoring={}), ensure_ascii=False))
    asked: list[str] = []
    doc.fix_request.connect(asked.append)
    (act,) = fixes_for(doc, _issue(doc, "no_actuators"))
    (score,) = fixes_for(doc, _issue(doc, "no_scoring"))
    assert act.key == "add_body_template" and score.key == "add_rule"
    act.apply(doc)
    score.apply(doc)
    assert asked == [OPEN_TEMPLATE_BODY, OPEN_RULES_ADD] and doc.undo_stack.index() == 0


def test_unknown_tag_pick_and_tag_body(qtbot: Any) -> None:
    sensor = {"kind": "contact", "id": "touch", "body": "torso", "with": ["tag.floor"]}
    doc = _doc(qtbot, json.dumps(_scene(sensors=[sensor]), ensure_ascii=False))
    issue = _issue(doc, "unknown_tag")
    assert where_to_path(issue.where) == ("sensors", 0, "with", 0)
    fixes = fixes_for(doc, issue)
    picks = [f for f in fixes if f.key == "pick_tag"]
    assert [f.label for f in picks] == ["ground"] and picks[0].menu == S.FIX_LABELS["pick_tag"]
    tag_body = [f for f in fixes if f.key == "tag_body"]
    assert tag_body and tag_body[0].label == S.FIX_LABELS["tag_body"].format(name="땅")
    assert _apply(doc, picks[0]) == 1 and "unknown_tag" not in _codes(doc)
    assert doc.items("sensors")[0]["with"] == ["tag.ground"]
    doc.undo_stack.undo()
    assert _apply(doc, tag_body[0]) == 1 and "unknown_tag" not in _codes(doc)
    assert doc.item(("bodies", "ground"))["tags"] == ["ground", "floor"]  # type: ignore[index]


def test_unknown_ref_picks_a_part_of_the_right_kind(qtbot: Any) -> None:
    sensor = {"kind": "body_state", "id": "st", "body": "nope", "fields": ["vx"]}
    doc = _doc(qtbot, json.dumps(_scene(sensors=[sensor]), ensure_ascii=False))
    fixes = fixes_for(doc, _issue(doc, "unknown_ref"))
    assert [f.label for f in fixes] == ["torso", "ground"] and {f.menu for f in fixes} == {S.FIX_LABELS["pick_ref"]}
    assert _apply(doc, fixes[0]) == 1 and "unknown_ref" not in _codes(doc)


def test_top_no_damping_gets_floor_friction(qtbot: Any) -> None:
    ball = {"id": "ball", "label": "공", "owner": "world", "pos": [2, 1], "shapes": [{"kind": "circle", "r": 0.3}]}
    data = _scene(world={"gravity": [0.0, 0.0], "view": "top", "max_time": 5.0})
    data["bodies"] = [data["bodies"][1], ball]
    doc = _doc(qtbot, json.dumps(data, ensure_ascii=False))
    assert "never_stops" not in _codes(doc)  # 위에서 본 세계는 엔진 검사 하나로(같은 고치기)
    (fix,) = fixes_for(doc, _issue(doc, "top_no_damping"))
    assert fix.key == "add_floor_friction" and _apply(doc, fix) == 1
    assert "top_no_damping" not in _codes(doc)


def test_unattached_offers_world_static_and_dynamic(qtbot: Any) -> None:
    data = _scene()
    data["bodies"].append({"id": "lonely", "label": "외톨이", "pos": [3, 1], "shapes": [{"kind": "box", "w": 0.2, "h": 0.2}]})
    doc = _doc(qtbot, json.dumps(data, ensure_ascii=False))
    fixes = fixes_for(doc, _issue(doc, "unattached"))
    assert [f.key for f in fixes] == ["to_world_static", "to_world_dynamic"]
    assert _apply(doc, fixes[1]) == 1 and "unattached" not in _codes(doc)
    assert doc.item(("bodies", "lonely"))["type"] == "dynamic"  # type: ignore[index]


def test_every_fix_label_is_reachable_and_codes_without_fixes_are_empty(qtbot: Any) -> None:
    assert set(FIX_CODES) >= set(SIX)
    keys: set[str] = set()
    for code in SIX:
        doc = _audit(qtbot, code)
        keys |= {f.key for f in fixes_for(doc, _issue(doc, code))}
    for data, code in ((_scene(sensors=[]), "no_sensors"), (_scene(actuators=[]), "no_actuators"),
                       (_scene(scoring={}), "no_scoring"),
                       (_scene(sensors=[{"kind": "contact", "id": "t", "body": "torso", "with": ["tag.x"]}]), "unknown_tag"),
                       (_scene(sensors=[{"kind": "body_state", "id": "t", "body": "x", "fields": ["vx"]}]), "unknown_ref")):
        doc = _doc(qtbot, json.dumps(data, ensure_ascii=False))
        keys |= {f.key for f in fixes_for(doc, _issue(doc, code))}
    data = _scene()
    data["bodies"].append({"id": "lonely", "pos": [3, 1], "shapes": [{"kind": "box", "w": 0.2, "h": 0.2}]})
    doc = _doc(qtbot, json.dumps(data, ensure_ascii=False))
    keys |= {f.key for f in fixes_for(doc, _issue(doc, "unattached"))}
    assert keys == set(S.FIX_LABELS) == set(S.FIX_TIPS)
    # 고치기가 없는 코드
    doc = _doc(qtbot, json.dumps(_scene(), ensure_ascii=False))
    from evolab.spec.compiled import Issue

    assert fixes_for(doc, Issue("error", "world.dt", "x", "step")) == []
    assert fixes_for(doc, Issue("error", "", "x", "")) == []


# ---------------------------------------------------------------------------
# T32: 문제 목록 탭과 상태 줄
# ---------------------------------------------------------------------------


def test_issues_tab_rows_have_fix_buttons_that_fix(qtbot: Any) -> None:
    doc = _audit(qtbot, "motor_twice")
    tab = IssuesTab(doc)
    qtbot.addWidget(tab)
    tab.show()
    doc.run_check()
    assert tab.tree.columnCount() == 4
    assert [tab.tree.headerItem().text(c) for c in range(4)] == list(S.EDP_ISSUES_HEADERS)
    row = tab.row_of("motor_twice")
    assert row >= 0
    button = tab.fix_button(row)
    assert isinstance(button, QToolButton) and button.text() == S.FIX_LABELS["drop_motor"]
    assert button.toolTip() == S.FIX_TIPS["drop_motor"]
    done: list[str] = []
    tab.fixed.connect(done.append)
    button.click()
    assert done == [S.FIX_DONE.format(what=S.FIX_LABELS["drop_motor"])]
    doc.run_check()
    assert tab.row_of("motor_twice") == -1


def test_menu_fixes_and_ignore_button(qtbot: Any) -> None:
    sensor = {"kind": "body_state", "id": "st", "body": "nope", "fields": ["vx"]}
    doc = _doc(qtbot, json.dumps(_scene(sensors=[sensor]), ensure_ascii=False))
    tab = IssuesTab(doc)
    qtbot.addWidget(tab)
    doc.run_check()
    button = tab.fix_button(tab.row_of("unknown_ref"))
    assert button is not None and button.menu() is not None
    assert button.popupMode() == QToolButton.ToolButtonPopupMode.InstantPopup
    sub = button.menu().actions()[0].menu()
    assert sub is not None and [a.text() for a in sub.actions()] == ["torso", "ground"]
    sub.actions()[0].trigger()
    assert "unknown_ref" not in _codes(doc)
    doc2 = _audit(qtbot, "score_far_is_good")
    tab2 = IssuesTab(doc2)
    qtbot.addWidget(tab2)
    doc2.run_check()
    row = tab2.row_of("score_far_is_good")
    ignore_button = tab2.ignore_button(row)
    assert ignore_button is not None and ignore_button.text() == S.FIX_IGNORE
    assert ignore_button.toolTip() == S.FIX_IGNORE_TIP
    ignore_button.click()
    doc2.run_check()
    assert tab2.row_of("score_far_is_good") == -1


def test_fix_button_is_reachable_with_the_keyboard(qtbot: Any) -> None:
    doc = _audit(qtbot, "joint_apart")
    tabs = BottomTabs(doc)
    qtbot.addWidget(tabs)
    tabs.resize(1000, 400)
    tabs.show()
    tabs.setCurrentWidget(tabs.issues_tab)
    doc.run_check()
    tab = tabs.issues_tab
    button = tab.fix_button(tab.row_of("joint_apart"))
    assert button is not None and button.focusPolicy() & Qt.FocusPolicy.TabFocus
    tab.tree.setFocus()
    QApplication.processEvents()
    reached = False
    for _ in range(40):
        w = QApplication.focusWidget()
        if w is button:
            reached = True
            break
        assert w is not None
        w.focusNextChild()
        QApplication.processEvents()
    assert reached
    qtbot.keyClick(button, Qt.Key.Key_Space)
    assert "joint_apart" not in _codes(doc)


def test_status_line_counts_errors_warnings_and_infos_by_name(qtbot: Any) -> None:
    view = EditorView()
    qtbot.addWidget(view)
    view.doc.load_text((AUDIT / "motor_twice.evoenv").read_text(encoding="utf-8"), None)
    view.doc.run_check()
    assert view.status.text() == S.ED_STATUS.format(errors=0, warnings=1, infos=0)
    for word in ("오류", "경고", "살펴볼 것"):
        assert word in view.status.text()
    data = json.loads((AUDIT / "motor_twice.evoenv").read_text(encoding="utf-8"))
    data["scoring"]["blocks"].append({"kind": "distance_to", "id": "far", "body": "torso", "target": "leg", "weight": 1})
    view.doc.load_text(json.dumps(data, ensure_ascii=False), None)
    view.doc.run_check()
    assert view.status.text() == S.ED_STATUS.format(errors=0, warnings=1, infos=1)
    # 살펴볼 것만 있으면 「문제 없음」(D38: 상태 줄은 L1만 말함)
    view.doc.load_text((AUDIT / "no_keys.evoenv").read_text(encoding="utf-8"), None)
    view.doc.run_check()
    assert view.status.text() == S.ED_STATUS_OK


def test_editor_view_routes_fix_requests(qtbot: Any) -> None:
    view = EditorView()
    qtbot.addWidget(view)
    view.popup_menus = False
    view.show()
    view.doc.load_text(json.dumps(_scene(scoring={}), ensure_ascii=False), None)
    (fix,) = fixes_for(view.doc, _issue(view.doc, "no_scoring"))
    fix.apply(view.doc)
    assert view.tabs.currentWidget() is view.tabs.rules_tab
    # 몸 틀 보관함: 팔레트의 「몸 틀…」 단추를 누른다(캔버스가 보관함 창을 띄움)
    asked: list[str] = []
    view.canvas.pick_template = lambda kind: asked.append(kind) or False  # type: ignore[method-assign]
    view.palette.template_requested.disconnect()
    view.palette.template_requested.connect(view.canvas.pick_template)
    view.doc.load_text(json.dumps(_scene(actuators=[]), ensure_ascii=False), None)
    (fix,) = fixes_for(view.doc, _issue(view.doc, "no_actuators"))
    fix.apply(view.doc)
    qtbot.waitUntil(lambda: asked == ["body"], timeout=2000)


def test_property_panel_shows_the_same_fix_buttons(qtbot: Any) -> None:
    from evolab.app.editor.properties import PropertyPanel

    doc = _audit(qtbot, "joint_apart")
    panel = PropertyPanel(doc)
    qtbot.addWidget(panel)
    panel.show()
    doc.select([("joints", "hip")])
    doc.run_check()
    assert panel.issue_box.isVisible() and panel.issue_list.count() == 1
    assert len(panel.fix_widgets) == 1
    button = panel.fix_widgets[0].fix_button  # type: ignore[attr-defined]
    assert button.text() == S.FIX_LABELS["align_joint"]
    button.click()
    doc.run_check()
    assert not panel.issue_box.isVisible()

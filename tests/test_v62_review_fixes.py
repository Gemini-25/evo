"""6.2.0 검토 고침 회귀 시험: 검토·적대 검증에서 확인된 결함마다 하나 이상(고친 뒤 통과, 고치기 전 실패를 확인함)."""

from __future__ import annotations

import os
import json
from pathlib import Path

import pytest

from evolab import cli, paths
from evolab.app import strings as S

ROOT = Path(__file__).resolve().parents[1]
RUNNING = ROOT / "environments" / "running.evoenv"


@pytest.fixture
def iso(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """데이터·상태 폴더를 tmp_path 아래로(실제 사용자 폴더를 건드리지 않게)."""
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    return tmp_path


# ---------------------------------------------------------------------------
# CLI


def test_cli_preset_output_is_byte_identical_even_with_windows_text_mode(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """`evolab preset running -o`는 내장 파일과 바이트까지 같아야 한다(윈도우 CI가 SHA-256으로 비교). 윈도우의 텍스트 모드
    (줄바꿈 → CRLF)를 흉내 내도 같아야 한다."""
    real = Path.write_text

    def windows_write_text(self: Path, data: str, encoding: str | None = None, errors: str | None = None,
                           newline: str | None = None) -> int:
        return real(self, data, encoding=encoding, errors=errors, newline="\r\n" if newline is None else newline)

    monkeypatch.setattr(Path, "write_text", windows_write_text)
    out = tmp_path / "made.evoenv"
    assert cli.main(["preset", "running", "-o", str(out)]) == 0
    assert out.read_bytes() == RUNNING.read_bytes()


def test_cli_train_running_with_a_large_validate_every_drops_the_preset_qualifying(
        iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """프리셋 권장 예선(출전마다 9,600회)이 준 검증 주기(10,000회)와 맞지 않으면 학습이 거절되지 않고 예선만 꺼진다."""
    capsys.readouterr()
    code = cli.main(["train", str(RUNNING), "--validate-every", "10000", "--max-evaluations", "96", "--workers", "0",
                     "--out", str(iso / "runs")])
    printed = capsys.readouterr()
    assert code == 0, printed.err
    line = next(ln for ln in printed.out.splitlines() if ln.startswith(S.TRAIN_STABILITY.split("{")[0]))
    assert line.endswith(S.TRAIN_STABILITY_OFF)  # 예선: 끔


def test_cli_train_running_without_flags_keeps_the_preset_qualifying(iso: Path) -> None:
    from evolab.engine.commands import _config_from_args
    from evolab.spec.io import compile_env

    args = cli._build_parser().parse_args(["train", str(RUNNING)])
    c = _config_from_args(args, None, compile_env(RUNNING.read_text(encoding="utf-8")))
    assert c.qualify_entrants == 3 and c.qualify_trials == 9600
    args = cli._build_parser().parse_args(["train", str(RUNNING), "--validate-every", "10000"])
    c = _config_from_args(args, None, compile_env(RUNNING.read_text(encoding="utf-8")))
    assert c.qualify_entrants == 1 and c.validate_every == 10000
    c.validate()


# ---------------------------------------------------------------------------
# 반복 실행 창


def test_repeat_dialog_shows_why_the_preset_qualifying_is_off(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from evolab.app.compare.repeat import RepeatDialog

    d = RepeatDialog(runs_dir=tmp_path / "runs", env_path=RUNNING)
    qtbot.addWidget(d)
    d.stop_evals.setChecked(True)
    d.max_evaluations.setValue(30_000)
    assert d.config().qualify_entrants == 1
    assert d.qualify_note.isVisibleTo(d) and d.qualify_note.text()
    d.max_evaluations.setValue(100_000)
    assert d.config().qualify_entrants == 3
    assert not d.qualify_note.isVisibleTo(d)


# ---------------------------------------------------------------------------
# 다듬기(U18·U32·1.5·C1)


def test_tiles_show_no_start_sentence_while_training_is_starting(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from tests.test_polish_dashboard import make_dashboard, refresh_all

    from evolab.app.lab.dashboard import TILE_IDS

    dash = make_dashboard(qtbot, tmp_path)
    dash.set_live(True)  # 시작하는 중: 통계 줄 0
    refresh_all(dash)
    for tid in TILE_IDS:
        tile = dash.tile(tid)
        assert tile.stack.currentWidget() is not tile.message or tile.message.text() != S.DASH_TILE_EMPTY[tid], tid
    dash.set_live(False)  # 끝나고 줄이 없으면 다시 한 문장
    refresh_all(dash)
    assert all(dash.tile(t).message.text() == S.DASH_TILE_EMPTY[t] for t in TILE_IDS)


def test_lab_tells_the_dashboard_when_training_is_live(qtbot, iso: Path) -> None:  # type: ignore[no-untyped-def]
    from evolab.app.lab.view import LabView

    lab = LabView()
    qtbot.addWidget(lab)
    assert lab.dashboard.live is False
    lab._update_roles(True)
    assert lab.dashboard.live is True
    lab._update_roles(False)
    assert lab.dashboard.live is False


def test_window_title_follows_a_second_opened_run(qtbot, iso: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    import shutil

    from evolab.app.main import MainWindow

    monkeypatch.setenv("EVOLAB_NO_TOUR", "1")
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    gens = {}
    for name in ("v10_run_ga", "v61_grow_run"):
        dst = iso / name
        shutil.copytree(ROOT / "tests" / "data" / name / "run", dst)
        assert w.lab.open_run(dst)
        qtbot.wait(50)
        gens[name] = w.lab.generation
        assert f"{w.lab.generation}" in w.windowTitle(), (name, w.windowTitle())
        assert w.state_texts()[2] == w.windowTitle()
    assert gens["v10_run_ga"] != gens["v61_grow_run"]


def test_colour_bar_captions_use_the_theme_ink(qtbot) -> None:  # type: ignore[no-untyped-def]
    import pyqtgraph as pg

    from evolab.app import style
    from evolab.app.lab import watch_tiles as WT

    bar = pg.ColorBarItem(values=(0, 1), label="x", interactive=False, colorMapMenu=False)
    theme = style.LIGHT
    WT._style_bar(bar, theme)
    assert bar.getAxis("left").labelStyle["color"].lower() == theme.ink2.lower()


# ---------------------------------------------------------------------------
# 프리셋: 조리법·규칙 칸(rules.candidates, recipe.adjust·build)


def _starter(name: str):  # type: ignore[no-untyped-def]
    from evolab.presets import recipe as RC

    return RC.load_builtin(name)


def test_rule_slot_choices_exclude_targets_the_compiler_refuses() -> None:
    from evolab.presets import recipe as RC
    from evolab.presets import rules as R

    for name in RC.builtin_names():
        data = RC.build(_starter(name))
        static = {b["id"] for b in data["bodies"] if b.get("owner", "agent") == "world" and b.get("type") == "static"}
        agent = {b["id"] for b in data["bodies"] if b.get("owner", "agent") == "agent"}
        regions = {r["id"] for r in data.get("regions", [])}
        reach_where = {v for v, _ in R.candidates(data, {s.id: s for s in R.RULES["reach"].slots}["where"])}
        assert reach_where == regions, name  # 「들어가면 성공」의 자리는 영역만
        for sid in ("where", "thing"):
            near = {v for v, _ in R.candidates(data, {s.id: s for s in R.RULES["near"].slots}[sid])}
            assert not (near & static), (name, sid)  # 땅·벽 같은 고정 물체는 거리 대상이 아니다
        assert agent <= {v for v, _ in R.candidates(data, {s.id: s for s in R.RULES["near"].slots}["thing"])}


def test_near_needs_a_movable_target_and_adds_a_goal_in_the_side_world() -> None:
    from evolab.presets import recipe as RC
    from evolab.presets import rules as R
    from evolab.presets.josa import fill

    data = RC.build(_starter("blank_side"))  # 땅(고정)뿐
    ok, why = R.available(data, "near")
    assert not ok and why == fill(S.RULES_NEEDS, {"what": S.RULE_NEEDS_WHAT["target"]})
    r = _starter("blank_side")
    r.rules.insert(0, {"rule": "near"})
    fixed, notes = RC.adjust(r)
    assert any(x.get("rule") == "near" for x in fixed.rules), notes  # 꺼지지 않고 골대를 함께 넣는다
    assert [p.template for p in fixed.props] == ["goal"]


@pytest.mark.parametrize("name", ["running", "blank_side", "blank_top", "drill"])
def test_auto_added_goal_is_not_on_the_player(name: str) -> None:
    from evolab.presets import recipe as RC
    from evolab.presets.trainable import alive, quick_check

    r = _starter(name)
    r.props = []
    r.rules = [x for x in r.rules if x.get("rule") != "reach"] + [{"rule": "reach"}]
    fixed, _notes = RC.adjust(r)
    assert [p.template for p in fixed.props] == ["goal"]
    quick = quick_check(RC.build(fixed))
    assert quick.ok and quick.spec is not None
    assert alive(quick.spec).ok, name


def _boxes(data: dict, ids: set[str]) -> list[tuple[str, float, float, float, float]]:
    """(id, x0, y0, x1, y1): 원·상자 모양만(땅 같은 사슬은 뺀다)."""
    out = []
    for sec in ("bodies", "regions"):
        for item in data.get(sec, []):
            if item.get("id") not in ids:
                continue
            px, py = (item.get("pos") or [0.0, 0.0])[:2]
            if sec == "bodies" and item.get("owner", "agent") == "agent":  # 에이전트 물체는 agent.spawn에 대한 자리
                spawn = (data.get("agent") or {}).get("spawn") or {}
                px, py = px + float(spawn.get("x", 0.0)), py + float(spawn.get("y", 0.0))
            shapes = item.get("shapes") or ([item["shape"]] if "shape" in item else [])
            for sh in shapes:
                ox, oy = (sh.get("offset") or [0.0, 0.0])[:2]
                if sh.get("kind") == "circle":
                    hw = hh = float(sh["r"])
                elif sh.get("kind") == "box":
                    hw, hh = float(sh["w"]) / 2, float(sh["h"]) / 2
                else:
                    continue
                cx, cy = px + ox, py + oy
                out.append((item["id"], cx - hw, cy - hh, cx + hw, cy + hh))
    return out


@pytest.mark.parametrize("world", ["side", "top"])
def test_props_toggled_together_never_overlap_each_other_or_the_player(world: str) -> None:
    import itertools

    from evolab.presets import recipe as RC
    from evolab.presets.library import default_library

    lib = default_library()
    start = "blank_side" if world == "side" else "blank_top"
    base = _starter(start)
    arena = lib.get(base.arena.template)
    kinds = [t.id for t in lib.all() if t.kind == "prop" and t.world in (world, "any") and t.id != "ground_flat"]
    assert {"ball", "goal", "wall"} <= set(kinds)
    for k in range(2, len(kinds) + 1):
        for subset in itertools.combinations(kinds, k):
            r = _starter(start)
            r.props = [RC.Use(t, {}, None, RC.prop_spawn(arena, t)) for t in subset]  # 화면(③)이 넣는 그대로
            fixed, _ = RC.adjust(r)
            data = RC.build(fixed)
            prop_ids = {i["id"] for sec in ("bodies", "regions") for i in data.get(sec, [])
                        if i.get("owner", "world") == "world" and i["id"] not in ("ground",) and not i["id"].startswith("wall_")}
            agent_ids = {b["id"] for b in data["bodies"] if b.get("owner", "agent") == "agent"}
            boxes = _boxes(data, prop_ids | agent_ids)
            for (a, *ra), (b, *rb) in itertools.combinations(boxes, 2):
                if a in agent_ids and b in agent_ids:
                    continue
                apart = ra[2] <= rb[0] or rb[2] <= ra[0] or ra[3] <= rb[1] or rb[3] <= ra[1]
                assert apart, (world, subset, a, b, ra, rb)


def test_prop_spawn_prefers_a_place_away_from_the_seat() -> None:
    from evolab.presets import recipe as RC
    from evolab.presets.library import default_library

    lib = default_library()
    assert RC.prop_spawn(lib.get("track_side"), "goal") == "finish"
    assert RC.prop_spawn(lib.get("yard_top"), "goal") == "goal"
    assert RC.prop_spawn(lib.get("track_side"), "ball") == "ahead"


def test_rule_summary_uses_the_chosen_direction() -> None:
    from evolab.presets import recipe as RC
    from evolab.presets import rules as R

    r = _starter("blank_side")
    for rule in r.rules:
        if rule.get("rule") == "distance":
            rule["direction"] = "left"
    data = RC.build(RC.adjust(r)[0])
    text = R.summary(data)
    assert text.startswith(S.RULE_DIRECTIONS["left"] + "으로 간 거리"), text
    assert "앞으로" not in text


def test_side_only_rules_become_read_only_after_switching_to_the_top_world() -> None:
    import copy

    from evolab.presets import rules as R

    data = json.loads(RUNNING.read_text(encoding="utf-8"))
    rows, _ = R.rows(data)
    assert next(i for i in rows if i.use.kind == "fall").use.editable
    top = copy.deepcopy(data)
    top["world"]["view"] = "top"
    top["world"]["gravity"] = [0.0, 0.0]
    rows, _ = R.rows(top)
    fall = next(i for i in rows if i.use.kind == "fall")
    assert not fall.use.editable
    assert fall.tip == S.RULES_WRONG_WORLD_TIP.format(world=S.TPL_WORLD_NAMES["side"])
    assert next(i for i in rows if i.use.kind == "distance").use.editable  # 어느 세계나 쓰는 규칙은 그대로


# ---------------------------------------------------------------------------
# 프리셋 만들기 창


@pytest.fixture
def wiso(iso: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from evolab.app.preset.wizard import PresetWizard

    monkeypatch.setattr(PresetWizard, "ask_discard", lambda self: True)
    return iso


def _wizard(qtbot, **kw):  # type: ignore[no-untyped-def]
    from evolab.app.preset.wizard import PresetWizard

    w = PresetWizard(**kw)
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    w.activateWindow()
    if w.windowHandle() is not None:
        w.windowHandle().requestActivate()
    qtbot.waitUntil(lambda: w.isActiveWindow(), timeout=5000)
    return w


def test_saving_a_long_taken_name_twice_gives_two_files(qtbot, wiso: Path) -> None:  # type: ignore[no-untyped-def]
    import faulthandler

    name = "아주 긴 이름 " * 12
    assert len(name.strip()) > paths.SAFE_STEM_MAX
    faulthandler.dump_traceback_later(20, exit=True)  # 예전에는 여기서 영원히 돌았다
    try:
        made = []
        for _ in range(3):
            w = _wizard(qtbot, start="running")
            w.name_edit.setText(name)
            made.append(w.save())
            w.close()
    finally:
        faulthandler.cancel_dump_traceback_later()
    assert len(set(made)) == 3 and all(p.exists() for p in made)
    assert all(len(p.stem) <= paths.SAFE_STEM_MAX for p in made)


def test_rejected_choice_snaps_back_to_the_accepted_value(qtbot, wiso: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from evolab.app.preset import wizard as WZ
    from evolab.presets import rules as R
    from evolab.presets import trainable as TR

    w = _wizard(qtbot, start="drill")
    w.go(3)
    combo = w.pages[3].rows["near"].slots["where"]
    before = next(u for u in R.find_rules(w.data)[0] if u.kind == "near").args["where"]
    other = next(i for i in range(combo.count()) if combo.itemData(i) != before)
    real = TR.quick_check
    monkeypatch.setattr(WZ.TR, "quick_check", lambda data: TR.Quick(False, [], 0, 0, 0, None))  # 만들 수 없는 바꿈
    combo.setCurrentIndex(other)
    combo.activated.emit(other)
    monkeypatch.setattr(WZ.TR, "quick_check", real)
    assert combo.currentData() == before  # 화면도 받아들인 값으로 돌아간다
    assert w.status.text() == S.PRESET_CHOICE_UNDONE
    assert next(u for u in R.find_rules(w.data)[0] if u.kind == "near").args["where"] == before


def test_instant_end_fix_button_goes_to_the_rule_that_ends_it(qtbot, wiso: Path) -> None:  # type: ignore[no-untyped-def]
    w = _wizard(qtbot, start="running")
    w.go(3)
    row = w.pages[3].rows["fall"]
    row.slots["height"].setValue(50.0)  # 몸통이 50m보다 낮으면 넘어짐: 시작하자마자 끝
    assert w.alive is not None and not w.alive.ok and w.alive.fix == "move_part"
    assert w.instant_rule() == "fall"
    assert w.btn_fix.isVisible() and w.btn_fix.text() == S.PRESET_ALIVE_FIXES["move_part"]
    w.go(1)
    w.btn_fix.click()
    assert w.step == 3  # 그 규칙이 있는 ④
    focus = QApplicationFocus()
    assert focus is not None and focus in row.controls(), focus


def QApplicationFocus():  # type: ignore[no-untyped-def]
    from PySide6.QtWidgets import QApplication

    return QApplication.focusWidget()


def test_ctrl_enter_takes_the_number_just_typed(qtbot, wiso: Path) -> None:  # type: ignore[no-untyped-def]
    import json as _json

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    w = _wizard(qtbot, start="running")
    got = []
    w.finished_with.connect(lambda text, what, path: got.append((_json.loads(text), what)))
    w.go(3)
    spin = w.pages[3].rows["time_limit"].slots["seconds"]
    spin.setFocus()
    qtbot.waitUntil(lambda: spin.hasFocus(), timeout=3000)
    spin.selectAll()
    QTest.keyClicks(spin, "30")
    QTest.keyClick(spin, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert got and got[0][1] == "lab"
    assert got[0][0]["world"]["max_time"] == 30.0


def test_reach_switched_on_in_the_side_world_leaves_the_lab_button_on(qtbot, wiso: Path) -> None:  # type: ignore[no-untyped-def]
    w = _wizard(qtbot, start="running")
    w.go(3)
    w.pages[3].toggle_rule("reach", True)
    assert any(r.get("rule") == "reach" for r in w.recipe.rules)
    assert w.alive is not None and w.alive.ok and w.btn_lab.isEnabled()
    goal = next(r for r in w.data["regions"] if r["id"] == "goal")
    assert goal["pos"][0] > 10.0  # 결승점 쪽(선수 자리 (0, 0)이 아니라)


# ---------------------------------------------------------------------------
# 에디터: 보관함 창의 「…도 함께 넣기」


def test_template_picker_also_inserts_the_needed_template(qtbot, iso: Path) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtCore import QTimer

    from evolab.app.editor.view import EditorView
    from evolab.app.preset.cards import TemplatePicker

    v = EditorView()
    qtbot.addWidget(v)
    v.show()
    c = v.canvas

    def factory(kind, world, data, parent):  # type: ignore[no-untyped-def]
        p = TemplatePicker(kind, world, data, parent)
        next(cd for cd in p.cards if cd.template.id == "biped").setChecked(True)
        assert not p.needs_box.isHidden() and p.needs_box.isChecked()
        QTimer.singleShot(0, p.btn_insert.click)
        return p

    c.picker_factory = factory
    assert c.pick_template("body")
    c.drop_template_at((0.0, 1.0))
    ids = [b["id"] for b in v.doc.data.get("bodies", [])]
    assert "ground" in ids and "torso" in ids
    ground = next(b for b in v.doc.data["bodies"] if b["id"] == "ground")
    assert ground.get("owner") == "world" and "ground" in (ground.get("tags") or [])
    v.doc.undo_stack.undo()  # 한 단계로 둘 다 빠진다
    assert not v.doc.data.get("bodies")

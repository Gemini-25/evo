"""M1 독립 검토에서 나온 결함들의 회귀 테스트."""

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFocusEvent
from pytestqt.qtbot import QtBot

from evolab import cli, paths
from evolab.app import strings as S
from evolab.app.play.record import PlayRecord, align_record, replay
from evolab.app.play.window import PlayWindow
from evolab.spec.compiled import EnvFileError, check_env
from evolab.spec.io import compile_env, load_env, parse_env
from evolab.world.physics2d.episode import Episode, EpisodeBuildError, Trial
from test_runtime import box_env

LANDER_PATH = paths.builtin_env_dir() / "lander.evoenv"
LANDER = json.loads(LANDER_PATH.read_text(encoding="utf-8"))


def lander(**over: Any) -> dict[str, Any]:
    d = copy.deepcopy(LANDER)
    d.update(over)
    return d


def errors_of(data: Any) -> list[str]:
    with pytest.raises(EnvFileError) as exc:
        compile_env(data)
    return [str(i) for i in exc.value.issues]


# ---------------------------------------------------------------------------
# 수식 칸
# ---------------------------------------------------------------------------


def test_progress_direction_formula_is_compiled() -> None:
    env = box_env()
    env["scoring"]["blocks"] = [{"kind": "progress", "id": "prog", "body": "box", "direction": "=pi / 2"}]
    ep = Episode(compile_env(env), None, Trial(0))
    ep.step([1.0])
    assert "prog" in ep.block_values() and math.isfinite(ep.result().score)
    env["scoring"]["blocks"][0]["direction"] = "=bogus + 1"
    assert any(e.startswith("scoring.blocks[0].direction") for e in errors_of(env))


def test_formula_with_leading_space_is_rejected() -> None:
    env = box_env()
    env["bodies"][0]["shapes"][0]["w"] = " =1.5"
    with pytest.raises(EnvFileError):
        parse_env(env)


def test_label_starting_with_equals_is_just_text() -> None:
    d = lander()
    d["bodies"][0]["label"] = "=본체"
    d["description"] = "= 착륙 연습"
    compile_env(d)


# ---------------------------------------------------------------------------
# 잘못된 숫자
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["nan", "inf", float("nan"), float("inf")])
def test_nan_and_infinity_are_rejected(value: Any) -> None:
    d = lander()
    d["bodies"][0]["material"]["friction"] = value
    with pytest.raises(EnvFileError):
        parse_env(d)


@pytest.mark.parametrize("text", ['{"name": "x", "world": {"dt": NaN}}', '{"name": "x", "world": {"dt": 1e400}}'])
def test_json_nan_and_overflow_are_rejected(text: str) -> None:
    with pytest.raises(EnvFileError):
        parse_env(text)


def test_body_state_scale_must_be_positive() -> None:
    d = lander()
    d["sensors"][1]["scale"]["vx"] = 0.0
    with pytest.raises(EnvFileError):
        parse_env(d)


def test_not_utf8_and_too_deep_files(tmp_path: Path) -> None:
    path = tmp_path / "ansi.evoenv"
    path.write_bytes(LANDER_PATH.read_text(encoding="utf-8").encode("cp949"))
    with pytest.raises(EnvFileError) as exc:
        load_env(path)
    assert S.SPEC_NOT_UTF8 in str(exc.value)
    with pytest.raises(EnvFileError) as exc2:
        parse_env("[" * 100000)
    assert str(exc2.value)


def test_not_object_message_has_single_braces() -> None:
    with pytest.raises(EnvFileError) as exc:
        parse_env("[]")
    assert "{{" not in str(exc.value)


# ---------------------------------------------------------------------------
# 참조 규칙
# ---------------------------------------------------------------------------


def test_joint_needs_a_dynamic_body() -> None:
    d = lander()
    d["bodies"].append({"id": "rock", "owner": "world", "type": "static", "shapes": [{"kind": "box", "w": 1, "h": 1}]})
    d["joints"].append({"id": "j_rock", "kind": "pivot", "a": "rock", "b": "terrain"})
    assert any(S.SPEC_JOINT_NO_DYNAMIC.format(id="j_rock") in e for e in errors_of(d))


def test_place_by_parent_must_share_owner() -> None:
    d = lander()
    d["joints"][0]["a"] = "terrain"
    assert any(S.SPEC_PLACE_BY_OWNER.format(joint="j_leg_l", body="leg_l") in e for e in errors_of(d))


def test_world_joint_cannot_use_body_params() -> None:
    d = lander()
    d["bodies"] += [
        {"id": "post", "owner": "world", "type": "static", "pos": [30, 10], "shapes": [{"kind": "box", "w": 0.2, "h": 1}]},
        {"id": "door", "owner": "world", "pos": [31, 10], "shapes": [{"kind": "box", "w": 2, "h": 0.2}]},
    ]
    d["joints"].append({"id": "hinge", "kind": "rot_spring", "a": "post", "b": "door", "stiffness": "=p.leg_len * 10"})
    assert any(e.startswith("joints[2].stiffness") for e in errors_of(d))


def test_region_only_tag_is_not_a_contact_target() -> None:
    d = lander()
    d["regions"] = [{"id": "lava", "tags": ["hot"], "shape": {"kind": "box", "w": 5, "h": 5}}]
    d["events"][1]["when"] = "contact(hull, tag.hot)"
    assert any(e.startswith("events[1].when") for e in errors_of(d))


def test_agent_self_contact_is_rejected() -> None:
    d = lander()
    d["events"][1]["when"] = "contact(hull, leg_l)"
    assert any(e.startswith("events[1].when") for e in errors_of(d))
    d2 = lander()
    d2["sensors"][2]["with"] = ["tag.leg"]
    assert any(e.startswith("sensors[2].with[0]") for e in errors_of(d2))
    d3 = lander()
    d3["joints"][0]["collide"] = True
    assert [i.where for i in check_env(parse_env(d3)) if i.severity == "warning"] == ["joints[0].collide"]


def test_distance_targets_must_be_regions_or_moving_bodies() -> None:
    d = lander()
    d["scoring"]["blocks"][2]["target"] = "terrain"
    assert any(e.startswith("scoring.blocks[2].target") for e in errors_of(d))


def test_thruster_must_be_on_agent_dynamic_body() -> None:
    d = lander()
    d["actuators"][1]["body"] = "terrain"
    assert any(e.startswith("actuators[1].body") for e in errors_of(d))


@pytest.mark.parametrize(("key", "expected"), [("R", S.SPEC_KEY_RESERVED), ("w", S.SPEC_KEY_UNKNOWN)])
def test_key_names_are_checked(key: str, expected: str) -> None:
    d = lander()
    d["actuators"][1]["keys"] = [key]
    assert any(expected.format(key=key) in e for e in errors_of(d))


@pytest.mark.parametrize("bad", ["lambda", "None", "leg__x"])
def test_ids_that_cannot_be_used_in_formulas(bad: str) -> None:
    d = lander()
    d["regions"] = [{"id": bad, "shape": {"kind": "box", "w": 1, "h": 1}}]
    assert errors_of(d)


# ---------------------------------------------------------------------------
# 오류 문구
# ---------------------------------------------------------------------------


def test_unknown_kind_lists_allowed_kinds_and_literal_has_no_english() -> None:
    d = lander()
    d["sensors"][0]["kind"] = "sonar"
    with pytest.raises(EnvFileError) as exc:
        parse_env(d)
    assert "body_state" in str(exc.value)
    d2 = lander()
    d2["termination"][0]["outcome"] = "win"
    with pytest.raises(EnvFileError) as exc2:
        parse_env(d2)
    assert " or " not in str(exc2.value)


def test_names_or_all_fields_get_matching_message() -> None:
    d = lander()
    d["brain"]["inputs"] = "st.vx"
    with pytest.raises(EnvFileError) as exc:
        parse_env(d)
    assert S.SPEC_ERR_NAMES_OR_ALL in str(exc.value)


def test_termination_event_path_is_kept() -> None:
    d = lander()
    del d["termination"][0]["event"]
    with pytest.raises(EnvFileError) as exc:
        parse_env(d)
    assert "termination[0].event" in str(exc.value)


# ---------------------------------------------------------------------------
# 런타임 물리
# ---------------------------------------------------------------------------


def wind_env(offset: float, drag: float, mass_density: float = 1.0) -> dict[str, Any]:
    env = box_env()
    env["world"]["wind"] = [0.0, 5.0]
    env["bodies"][0]["shapes"][0]["offset"] = [offset, 0.0]
    env["bodies"][0]["material"]["density"] = mass_density
    env["bodies"][0]["drag"] = drag
    return env


def test_drag_acts_at_center_of_gravity() -> None:
    ep = Episode(compile_env(wind_env(2.0, 1.0)), None, Trial(0))
    for _ in range(60):
        ep.step([0.0])
    assert abs(ep.bodies["box"].angular_velocity) < 1e-9


def test_strong_drag_does_not_explode() -> None:
    ep = Episode(compile_env(wind_env(0.0, 1e5)), None, Trial(0))
    for _ in range(60):
        ep.step([0.0])
        assert abs(ep.bodies["box"].velocity.y) <= 5.0 + 1e-9
    assert ep.bodies["box"].velocity.y == pytest.approx(5.0, rel=1e-6)


L_SHAPE = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]


def test_clockwise_concave_polygon_works() -> None:
    masses = []
    for pts in (L_SHAPE, L_SHAPE[::-1]):
        env = box_env()
        env["bodies"][0]["shapes"] = [{"kind": "poly", "points": pts}]
        masses.append(Episode(compile_env(env), None, Trial(0)).bodies["box"].mass)
    assert masses[0] == pytest.approx(masses[1]) == pytest.approx(6.0)


def test_self_intersecting_polygon_is_rejected() -> None:
    star = [[math.cos(a) * 2, math.sin(a) * 2] for a in (math.pi / 2 + k * 4 * math.pi / 5 for k in range(5))]
    env = box_env()
    env["bodies"][0]["shapes"] = [{"kind": "poly", "points": star}]
    with pytest.raises(EpisodeBuildError):
        Episode(compile_env(env), None, Trial(0))


def test_spawn_spin_is_rigid() -> None:
    d = lander()
    d["agent"]["spawn"]["omega"] = 2.0
    ep = Episode(compile_env(d), None, Trial(0))
    hull, leg = ep.bodies["hull"], ep.bodies["leg_l"]
    va = hull.velocity_at_local_point((-0.8, -0.5))
    vb = leg.velocity_at_local_point((0.0, 0.6))
    assert (va - vb).length < 1e-9


def test_gentle_landing_on_pad_succeeds() -> None:
    d = lander()
    d["world"]["wind"] = [0.0, 0.0]
    d["agent"]["spawn"] = {"x": "=gen.terrain.pad.x", "y": "=gen.terrain.pad.y + 1.9", "angle": 0.0, "vx": 0.0, "vy": 0.0}
    spec = compile_env(d)
    for seed in range(15):
        ep = Episode(spec, None, Trial(seed))
        while not ep.done:
            ep.step([0.0, 0.0, 0.0])
        assert ep.result().end_label == "성공", seed


# ---------------------------------------------------------------------------
# 직접 조종 창과 기록
# ---------------------------------------------------------------------------


def test_focus_loss_releases_keys(qtbot: QtBot) -> None:
    w = PlayWindow(load_env(LANDER_PATH), seed=1)
    qtbot.addWidget(w)
    w.show()
    qtbot.keyPress(w.canvas, Qt.Key.Key_Up)
    assert w.pressed
    w.canvas.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut, Qt.FocusReason.ActiveWindowFocusReason))
    assert not w.pressed and w.current_action() == [0.0, 0.0, 0.0]


def test_closing_stops_the_timer(qtbot: QtBot) -> None:
    w = PlayWindow(load_env(LANDER_PATH), seed=1)
    qtbot.addWidget(w)
    w.show()
    w.close()
    assert not w.timer.isActive()


def test_save_button_only_after_the_end(qtbot: QtBot) -> None:
    w = PlayWindow(load_env(LANDER_PATH), seed=2)
    qtbot.addWidget(w)
    assert not w.save_button.isEnabled()
    ep = w.episode
    assert ep is not None
    while not ep.done:
        ep.step([0.0, 0.0, 0.0])
    w._finish()
    assert w.save_button.isEnabled()


def test_formula_scoring_total_matches_result(qtbot: QtBot) -> None:
    d = lander()
    d["scoring"]["formula"] = "100 * s_success - s_dist"
    spec = compile_env(d)
    w = PlayWindow(spec, seed=3)
    qtbot.addWidget(w)
    ep = w.episode
    assert ep is not None
    for _ in range(30):
        ep.step([0.5, 0.0, 0.0])
    w._fill_tables(force=True)
    total = w.blocks.item(len(spec.file.scoring.blocks), 2).text()
    assert total == f"{ep.result().score:+.2f}"
    assert w.blocks.item(0, 2).text() == S.PLAY_NO_WEIGHT


def test_failed_start_keeps_the_running_episode(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    w = PlayWindow(load_env(LANDER_PATH), seed=5)
    qtbot.addWidget(w)
    old_episode = w.episode
    monkeypatch.setattr("evolab.app.play.window.QMessageBox.warning", lambda *a, **k: None)

    def boom(*a: Any, **k: Any) -> None:
        raise EpisodeBuildError("x")

    monkeypatch.setattr("evolab.app.play.window.Episode", boom)
    assert w.start(99) is False
    assert w.seed == 5 and w.episode is old_episode


def test_record_outputs_are_matched_by_name() -> None:
    spec = load_env(LANDER_PATH)
    rec = PlayRecord("x", "h", 4, ["side_r", "main", "side_l"], [[0.0, 1.0, 0.0]] * 50)
    aligned = align_record(spec, rec)
    assert aligned.outputs == spec.outputs and aligned.actions[0] == [1.0, 0.0, 0.0]
    bad = PlayRecord("x", "h", 4, ["main", "side_l"], [[1.0, 0.0]])
    with pytest.raises(ValueError):
        replay(spec, bad)


@pytest.mark.parametrize("content", ["not json", "[]", '{"format": "evolab-play-record", "actions": null}'])
def test_cli_verify_broken_record_exits_2(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.evoplay"
    path.write_text(content, encoding="utf-8")
    assert cli.main(["play", str(LANDER_PATH), "--verify", str(path)]) == 2


def test_cli_verify_record_without_result_exits_2(tmp_path: Path) -> None:
    spec = load_env(LANDER_PATH)
    rec = PlayRecord(spec.file.name, "h", 1, list(spec.outputs), [[0.0, 0.0, 0.0]] * 10)
    path = tmp_path / "half.evoplay"
    rec.save(path)
    assert cli.main(["play", str(LANDER_PATH), "--verify", str(path)]) == 2

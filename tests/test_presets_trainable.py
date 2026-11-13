"""6.2.0 묶음 P: 「늘 올바른 파일」의 L1·L2·L3 (스포츠 설계 6.1 T17, T40의 P 쪽, D38).

L2의 까닭(0스텝 종료, 점수 항 없음, 입출력 없음)과 고치기, L3 글, 같은 입력 = 같은 결과. 자체 점검 「프리셋」: 내장 조리법 전부와
달리기 한 세대, 틀 하나를 깨뜨리면 그 이름으로 실패.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from evolab import paths
from evolab.app import strings as S
from evolab.presets import rules as R
from evolab.presets import trainable as TR
from evolab.spec.io import compile_env, parse_env

ROOT = Path(__file__).resolve().parents[1]
BUILTIN = ROOT / "src" / "evolab" / "assets" / "templates"
RECIPES = ROOT / "src" / "evolab" / "assets" / "recipes"
FIXTURES = ROOT / "tests" / "fixtures" / "recipes"
MEASURED = ROOT / "tests" / "data" / "presets" / "running_measured.evoenv"


def walker_scene(**world: Any) -> dict[str, Any]:
    """서 있는 몸(몸통 + 다리 둘, 모터 둘)과 땅. 점수 규칙은 아직 없다."""
    data: dict[str, Any] = {
        "version": 1, "name": "walker",
        "world": {"max_time": 10.0, **world},
        "bodies": [
            {"id": "ground", "label": "땅", "owner": "world", "type": "static", "tags": ["ground"],
             "shapes": [{"kind": "box", "w": 40.0, "h": 1.0}], "pos": [0.0, -0.5]},
            {"id": "torso", "label": "몸통", "pos": [0.0, 0.9], "shapes": [{"kind": "box", "w": 0.4, "h": 0.6}],
             "material": {"density": 50.0}},
            {"id": "leg_l", "label": "왼 다리", "place_by": "hip_l", "shapes": [{"kind": "box", "w": 0.4, "h": 0.6}]},
            {"id": "leg_r", "label": "오른 다리", "place_by": "hip_r", "shapes": [{"kind": "box", "w": 0.4, "h": 0.6}]},
        ],
        "joints": [{"id": "hip_l", "kind": "weld", "a": "torso", "b": "leg_l", "anchor_a": [0.0, -0.3],
                    "anchor_b": [0.0, 0.3]},
                   {"id": "hip_r", "kind": "pivot", "a": "torso", "b": "leg_r", "anchor_a": [0.0, -0.3],
                    "anchor_b": [0.0, 0.3], "limit": [-0.5, 0.5]}],
        "actuators": [{"kind": "motor", "id": "m_hip", "joint": "hip_r", "max_torque": 80.0, "max_speed": 5.0}],
        "sensors": [{"kind": "joint_angle", "id": "a_hip", "joint": "hip_r"},
                    {"kind": "body_state", "id": "st", "body": "torso", "fields": ["x", "vx"]}],
        "termination": [{"event": "timeout", "outcome": "neutral"}],
    }
    return data


def spec_of(data: dict[str, Any]) -> Any:
    quick = TR.quick_check(data)
    assert quick.ok, [str(i) for i in quick.errors]
    return quick.spec


def test_quick_check_reports_errors_and_sizes() -> None:
    data = walker_scene()
    quick = TR.quick_check(data)
    assert (quick.ok, quick.inputs, quick.outputs, quick.body_genes) == (True, 3, 1, 0)
    data["joints"][0]["a"] = "nobody"
    bad = TR.quick_check(data)
    assert not bad.ok and bad.spec is None and bad.errors and all(i.severity == "error" for i in bad.errors)
    assert not TR.quick_check({"version": 99, "name": "x"}).ok
    assert not TR.quick_check("{ not json").ok


def test_alive_without_score_offers_add_rule() -> None:
    verdict = TR.alive(spec_of(walker_scene()))
    assert (verdict.ok, verdict.reason, verdict.fix) == (False, S.PRESET_ALIVE_NO_SCORE, "add_rule")
    data = walker_scene()
    R.add_rule(data, "survive", per_s=1.0)  # 시간 끝만 있으면 버틴 시간은 모두 같다
    assert TR.alive(spec_of(data)).reason == S.PRESET_ALIVE_NO_SCORE
    R.add_rule(data, "touch_ends", part="torso", what="tag.ground", outcome="failure")
    assert TR.alive(spec_of(data)).ok
    data2 = walker_scene()
    R.add_rule(data2, "distance", part="torso", direction="right", per_m=0.0)  # 가중치 0은 세지 않는다
    assert TR.alive(spec_of(data2)).reason == S.PRESET_ALIVE_NO_SCORE


def test_alive_without_inputs_or_outputs() -> None:
    data = walker_scene()
    R.add_rule(data, "distance", part="torso", direction="right", per_m=1.0)
    data["sensors"] = []
    verdict = TR.alive(spec_of(data))
    assert (verdict.ok, verdict.reason, verdict.fix) == (False, S.PRESET_ALIVE_NO_IO, None)
    data = walker_scene()
    R.add_rule(data, "distance", part="torso", direction="right", per_m=1.0)
    data["actuators"] = []
    assert TR.alive(spec_of(data)).reason == S.PRESET_ALIVE_NO_IO
    # 뇌 없이 몸 수치만 진화하는 몸은 출력이 없어도 된다
    data["params"] = [{"id": "w", "value": 0.4, "evolve": True, "min": 0.2, "max": 0.8}]
    data["bodies"][1]["shapes"][0]["w"] = "=p.w"
    assert TR.alive(spec_of(data)).ok


def test_alive_catches_an_end_at_step_zero_and_names_the_rule() -> None:
    data = walker_scene()
    R.add_rule(data, "distance", part="torso", direction="right", per_m=1.0)
    R.add_rule(data, "touch_ends", part="leg_l", what="tag.ground", outcome="failure")  # 다리는 처음부터 땅에 닿음
    data["bodies"][1]["pos"] = [0.0, 0.6]
    verdict = TR.alive(spec_of(data))
    assert not verdict.ok and verdict.fix == "move_part"
    assert verdict.reason == S.PRESET_ALIVE_INSTANT.format(rule=S.RULE_TITLES["touch_ends"])
    assert verdict.still_end is not None and verdict.still_end[2] == "touch"


def test_alive_reports_a_still_end_as_a_notice_only() -> None:
    data = walker_scene()
    R.add_rule(data, "distance", part="torso", direction="right", per_m=1.0)
    R.add_rule(data, "fall", parts=["torso"], what="tag.ground", root="torso", height=0.5, penalty=1.0)
    data["agent"] = {"spawn": {"angle": 0.6}}  # 기울어 시작해 1초 안에 넘어진다
    verdict = TR.alive(spec_of(data))
    assert verdict.ok and verdict.still_end is not None
    t, label, event = verdict.still_end
    assert 0.0 < t < TR.ALIVE_SECONDS and event == "fall" and label
    assert verdict.notice() == S.PRESET_ALIVE_STILL_END.format(t=t, end=label)
    assert TR.alive(spec_of(data)) == verdict  # 같은 입력 = 같은 결과


def test_alive_on_the_measured_running_file() -> None:
    """보조 100의 몸은 행동 0이면 2.7초쯤 넘어진다(DE 9.1-1): 2초 관문 안에서는 알림도 없다."""
    from evolab.spec.io import load_env

    verdict = TR.alive(load_env(MEASURED))
    assert verdict.ok and verdict.still_end is None


def test_probe_texts_and_determinism() -> None:
    data = walker_scene()
    R.add_rule(data, "distance", part="torso", direction="right", per_m=1.0)
    spec = spec_of(data)
    first = TR.probe(spec, n=8, seed=3)
    assert first == TR.probe(spec, n=8, seed=3)
    assert first.ok and first.reason == S.PRESET_PROBE_OK.format(n=8) and first.score_spread > 0
    assert first.build_failures == 0 and 0 < first.first_end_s <= 10.0
    flat = walker_scene()
    flat["scoring"] = {"blocks": [{"kind": "progress", "id": "s_ground", "body": "ground", "weight": 1.0}]}
    probe = TR.probe(spec_of(flat), n=4)
    assert not probe.ok and probe.reason == S.PRESET_PROBE_FLAT.format(n=4) and probe.score_spread == 0.0


def test_probe_counts_bodies_that_do_not_build() -> None:
    data = walker_scene()
    R.add_rule(data, "distance", part="torso", direction="right", per_m=1.0)
    data["params"] = [{"id": "w", "value": 0.4, "evolve": True, "min": -1.0, "max": 1.0}]
    data["bodies"][2]["shapes"][0]["w"] = "=p.w"  # 음수 폭이면 조립되지 않는다
    probe = TR.probe(spec_of(data), n=16, seed=0)
    assert probe.build_failures > 0
    data["bodies"][2]["shapes"][0]["w"] = -0.5
    data["params"] = []
    probe = TR.probe(compile_env(copy.deepcopy(data)), n=4)
    assert (probe.ok, probe.reason, probe.build_failures) == (False, S.PRESET_PROBE_BUILD, 4)
    assert TR.alive(compile_env(copy.deepcopy(data))).reason == S.PRESET_PROBE_BUILD


def test_has_state_score_table() -> None:
    env = parse_env(walker_scene())
    assert not TR.has_state_score(env)
    for block, expected in (
            ({"kind": "progress", "id": "b", "body": "torso"}, True),
            ({"kind": "energy", "id": "b", "weight": -0.1}, True),
            ({"kind": "survival_time", "id": "b"}, False),
            ({"kind": "event", "id": "b", "event": "timeout"}, False),
            ({"kind": "event", "id": "b", "event": "out_of_bounds", "weight": -1.0}, True)):
        data = walker_scene()
        data["scoring"] = {"blocks": [block]}
        assert TR.has_state_score(parse_env(data)) is expected, block
    data = walker_scene()
    data["scoring"] = {"formula": "time"}
    assert TR.has_state_score(parse_env(data))


# ---------------------------------------------------------------------------
# 자체 점검 「프리셋」(T40의 P 쪽: selftest.py에 등록하는 것은 묶음 H)
# ---------------------------------------------------------------------------


def _stand_in_assets(tmp: Path) -> tuple[Path, Path]:
    """내장 틀·조리법의 사본 + (PM의 것이 아직 없으면) biped·disc 대역과 측정 원형으로 만든 달리기 조리법."""
    tdir, rdir = tmp / "templates", tmp / "recipes"
    shutil.copytree(BUILTIN, tdir)
    shutil.copytree(RECIPES, rdir)
    measured = json.loads(MEASURED.read_text(encoding="utf-8"))
    if not (tdir / "biped.evotpl").exists():
        body = copy.deepcopy(measured)
        sensors = body["sensors"]
        for s in sensors:
            if s["kind"] == "contact":
                s["with"] = ["$ground"]
        tpl = {"format": "evolab-template", "version": 1, "id": "biped", "kind": "body", "world": "side",
               "name_id": "biped", "params": [], "root": "torso",
               "parts": {"bodies": [b for b in body["bodies"] if b["id"] != "ground"], "joints": body["joints"],
                         "actuators": body["actuators"], "sensors": sensors},
               "slots": {"$ground": {"tags": ["ground"], "label_id": "ground_flat"}},
               "randoms": body["randomize"], "spawn": body["agent"]["spawn"]}
        (tdir / "biped.evotpl").write_text(json.dumps(tpl, ensure_ascii=False), encoding="utf-8")
    if not (tdir / "disc.evotpl").exists():
        raw = json.loads((FIXTURES / "fixture_puck.evotpl").read_text(encoding="utf-8"))
        raw.update(id="disc")
        (tdir / "disc.evotpl").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    if not (rdir / "running.json").exists():
        rec = {"recipe": 1, "start": "running", "name": "달리기", "world": "side",
               "arena": {"template": "track_side", "values": {}},
               "roles": [{"id": "runner", "label": "", "body": {"template": "biped", "values": {}}}], "props": [],
               "rules": [{"rule": "distance", "part": "torso", "direction": "right", "per_m": 1.0},
                         {"rule": "fall", "parts": ["torso", "thigh_l", "thigh_r"], "what": "$ground", "root": "torso",
                          "height": 0.8, "penalty": 1.0},
                         {"rule": "save_energy", "weight": 0.02}, {"rule": "out_ends", "outcome": "failure"},
                         {"rule": "time_limit", "seconds": 15.0}],
               "players": {"mode": "solo", "seats": [{"role": "runner", "count": 1, "spawn": "start"}]},
               "evaluation": {"trials": 3}, "train": {"qualify_entrants": 3, "qualify_trials": 9600}}
        (rdir / "running.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return tdir, rdir


@pytest.fixture()
def assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    tdir, rdir = _stand_in_assets(tmp_path)
    monkeypatch.setattr(paths, "template_dirs", lambda: [tdir, tmp_path / "user"])
    monkeypatch.setattr(paths, "recipe_dir", lambda: rdir)
    return tdir, rdir


def test_selftest_check_builds_every_builtin_recipe(assets: tuple[Path, Path]) -> None:
    ok, text = TR.selftest_check()
    assert ok, text
    n = len(list(assets[1].glob("*.json")))
    assert text.startswith(S.SELFTEST_PRESETS_DETAIL.split("{")[0]) and f"{n}" in text


def test_selftest_check_names_a_broken_template(assets: tuple[Path, Path]) -> None:
    tdir, _ = assets
    raw = json.loads((tdir / "wall.evotpl").read_text(encoding="utf-8"))
    raw["params"][0]["min"] = 99.0
    (tdir / "wall.evotpl").write_text(json.dumps(raw), encoding="utf-8")
    ok, text = TR.selftest_check()
    assert not ok and "wall.evotpl" in text and text.startswith(S.SELFTEST_ERR_PRESET.split("「")[0])


def test_selftest_check_names_a_broken_recipe(assets: tuple[Path, Path]) -> None:
    _, rdir = assets
    raw = json.loads((rdir / "blank_top.json").read_text(encoding="utf-8"))
    raw["rules"] = [{"rule": "time_limit", "seconds": 15.0}]  # 점수 규칙이 없으면 L2를 못 넘는다
    (rdir / "blank_top.json").write_text(json.dumps(raw), encoding="utf-8")
    ok, text = TR.selftest_check()
    assert not ok and "blank_top" in text and S.PRESET_ALIVE_NO_SCORE in text

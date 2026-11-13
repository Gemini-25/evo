"""6.2.0 묶음 PM: 내장 프리셋 「달리기」와 몸 틀 biped·disc, 조리법 running·drill (스포츠 설계 6.1 T8·T11·T14, D27·D29·D32).

- T8: environments/running.evoenv는 형식 4, 입력 13·출력 6·유전자 326, editor.train의 예선 값, 측정에 쓴 원형 파일
  (tests/data/presets/running_measured.evoenv)과 물리 동일(physics_view), 한 세대(개체 8)가 워커 0·2에서 같다.
- T14(PM 쪽): 내장 파일의 환경 해시 = build(load_builtin("running")), make_presets의 견주기가 문제 없음.
- T11(PM 쪽): biped·disc의 겉 슬라이더 귀퉁이(최소·최대 조합과 기본값)에서 L1·L2, 60스텝 자세가 유한.
- D27: 균형 보조 체크를 끄면 부품이 빠져 형식 1이고, 나머지 물리는 원형 파일에서 균형 보조만 뺀 것과 같다.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from evolab import paths
from evolab.engine.pool import WorkerPool
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.policy.genome import layout_for
from evolab.presets import recipe as RC
from evolab.presets import template as T
from evolab.presets.library import Library
from evolab.presets.trainable import alive, quick_check
from evolab.spec import symbols as sy
from evolab.spec.io import compile_env, dump_env, env_hash, feature_version, parse_env
from evolab.world.physics2d.episode import Episode, Trial

ROOT = Path(__file__).resolve().parents[1]
BUILTIN = ROOT / "src" / "evolab" / "assets" / "templates"
RUNNING = ROOT / "environments" / "running.evoenv"
MEASURED = ROOT / "tests" / "data" / "presets" / "running_measured.evoenv"
TRAIN = {"qualify_entrants": 3, "qualify_trials": 9600}
"""달리기의 권장 새 실행 설정(D32, PX0 뒤에도 그대로: 예선 3 × 9,600회, 정체하면 새로 시작은 끔)."""


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _lib() -> Library:
    return Library([BUILTIN])


def _spec(data: dict[str, Any] | str | bytes) -> Any:
    return compile_env(parse_env(data if not isinstance(data, dict) else json.dumps(data, ensure_ascii=False)))


# ---------------------------------------------------------------------------
# T8 내장 파일
# ---------------------------------------------------------------------------


def test_running_file_shape() -> None:
    data = _json(RUNNING)
    assert data["version"] == 4
    spec = _spec(RUNNING.read_bytes())
    assert feature_version(spec.file) == 4
    assert spec.issues == [], [str(i) for i in spec.issues]
    layout = layout_for(spec)
    assert (len(spec.inputs), len(spec.outputs), layout.dim) == (13, 6, 326)
    assert layout.brain.hidden == (16,)
    assert spec.outputs == ["m_hip_l", "m_knee_l", "m_ankle_l", "m_hip_r", "m_knee_r", "m_ankle_r"]
    assert data["editor"]["train"] == TRAIN
    assert [r["kind"] for r in data["editor"]["rules"]] == ["distance", "fall", "save_energy", "out_ends"]
    assert data["editor"]["recipe"]["built"] == env_hash(spec.file)
    assert data["name"] == "달리기"
    assert data["evaluation"]["trials"] == 3


def test_running_is_physics_equal_to_the_measured_prototype() -> None:
    """G1(D29의 (1)): 측정에 쓴 원형 파일과 이름·설명·표시 이름·키·형식 번호만 다르다."""
    assert RC.physics_view(_json(RUNNING)) == RC.physics_view(_json(MEASURED))
    built = RC.build(RC.load_builtin("running"), _lib())
    assert RC.physics_view(built) == RC.physics_view(_json(MEASURED))


def test_running_file_is_the_recipe_output() -> None:
    """G1(D29의 (2)): 내장 파일 = build(조리법), make_presets가 쓴 글 그대로."""
    data = _json(RUNNING)
    built = RC.build(RC.load_builtin("running"))
    assert env_hash(parse_env(data)) == env_hash(parse_env(built))
    assert RUNNING.read_text(encoding="utf-8") == dump_env(parse_env(built))
    sys.path.insert(0, str(ROOT / "packaging"))
    try:
        import make_presets
    finally:
        sys.path.remove(str(ROOT / "packaging"))
    assert "running" in make_presets.SPORTS
    assert make_presets.check("running") == []


def test_build_is_deterministic() -> None:
    for name in ("running", "drill"):
        a = dump_env(parse_env(RC.build(RC.load_builtin(name))))
        b = dump_env(parse_env(RC.build(RC.load_builtin(name))))
        assert a == b


def test_new_run_settings_come_from_the_preset() -> None:
    cfg = TrainConfig.for_new_run(_spec(RUNNING.read_bytes()))
    assert (cfg.qualify_entrants, cfg.qualify_trials) == (3, 9600)
    assert cfg.restart_patience is None  # 정체하면 새로 시작은 켜지 않는다(D32)
    assert TrainConfig.for_new_run(_spec(RUNNING.read_bytes()), qualify_entrants=1).qualify_entrants == 1


def test_running_is_the_seventh_builtin() -> None:
    assert paths.BUILTIN_ORDER[-1] == "running"
    assert [p.stem for p in paths.builtin_envs()][:7] == list(paths.BUILTIN_ORDER)


def test_one_generation_is_the_same_with_0_and_2_workers() -> None:
    text = RUNNING.read_text(encoding="utf-8")
    rows = []
    for workers in (0, 2):
        cfg = TrainConfig(run_seed=5, popsize=8, validate_every=10**9, max_generations=1)
        with WorkerPool(workers) as pool:
            row = Trainer(text, cfg, pool, None, observe=False).step()
        rows.append({k: v for k, v in row.items() if k not in ("elapsed", "worker_utilization", "gen_seconds", "steps_per_second")})
    assert rows[0] == rows[1]
    assert rows[0]["evaluations"] == 24 and "block_s_fall" in rows[0]


# ---------------------------------------------------------------------------
# 몸 틀 biped (D27)
# ---------------------------------------------------------------------------


def _side_scene(values: dict[str, float]) -> dict[str, Any]:
    lib = _lib()
    data: dict[str, Any] = {"version": 4, "name": "t", "world": {"gravity": [0.0, -9.81], "max_time": 15.0}}
    T.insert(data, T.expand(lib.get("track_side"), at=(0.0, 0.0)))
    biped = lib.get("biped")
    filled, missing = T.resolve_slots(biped, data)
    assert not missing
    T.insert(data, T.expand(biped, values, slots=filled))
    data["scoring"] = {"blocks": [{"kind": "progress", "id": "s_go", "body": "torso", "weight": 1.0}]}
    return data


def test_biped_template_matches_the_measured_body_table() -> None:
    tpl = _lib().get("biped")
    assert (tpl.kind, tpl.world, tpl.root) == ("body", "side", "torso")
    assert [p.id for p in tpl.params if p.main] == ["size", "power", "balance"]  # 「다리 길이」는 없다(D27)
    assert tpl.param("balance").min == 0.0 and tpl.param("balance").max == 1.0 and tpl.param("balance").step == 1.0
    parts = T.expand(tpl, tpl.defaults())
    assert len(parts["bodies"]) == 7 and len(parts["joints"]) == 6
    kinds = [a["kind"] for a in parts["actuators"]]
    assert kinds == ["motor"] * 6 + ["balance"]
    measured = _json(MEASURED)
    mine = {a["id"]: a for a in parts["actuators"]}
    for a in measured["actuators"]:
        for k, v in a.items():
            if k not in ("label",):
                assert mine[a["id"]][k] == v, (a["id"], k)
    assert parts["spawn"] == measured["agent"]["spawn"]
    assert [r["id"] for r in parts["randomize"]] == ["tilt", "push"]


def test_biped_keys_one_pair_per_joint() -> None:
    parts = T.expand(_lib().get("biped"), _lib().get("biped").defaults())
    keys = []
    for a in parts["actuators"]:
        if a["kind"] != "motor":
            continue
        assert len(a["keys_pos"]) == len(a["keys_neg"]) == 1
        keys += a["keys_pos"] + a["keys_neg"]
    assert keys == ["Q", "A", "W", "S", "E", "D", "U", "J", "I", "K", "O", "L"]
    assert all(k in sy.KEY_NAMES for k in keys)


def test_balance_off_drops_the_part_and_falls_to_format_1() -> None:
    """체크 「균형 보조」를 끄면 부품을 넣지 않는다: 형식 1, 입력·출력은 그대로, 나머지 물리는 원형에서 보조만 뺀 것."""
    recipe = RC.load_builtin("running")
    recipe.roles[0].body.values["balance"] = 0.0
    data = RC.build(recipe)
    assert data["version"] == 1 and feature_version(parse_env(data)) == 1
    spec = _spec(data)
    assert (len(spec.inputs), len(spec.outputs), layout_for(spec).dim) == (13, 6, 326)
    measured = _json(MEASURED)
    measured["actuators"] = [a for a in measured["actuators"] if a["kind"] != "balance"]
    assert RC.physics_view(data) == RC.physics_view(measured)


def _corners(tpl: T.Template) -> list[dict[str, float]]:
    base = tpl.defaults()
    out = [dict(base)]
    sliders = [p for p in tpl.params if p.main and p.id != "balance"]
    for combo in itertools.product(*[(p.min, p.max) for p in sliders]):
        out.append(dict(base, **{p.id: v for p, v in zip(sliders, combo, strict=True)}))
    return out


@pytest.mark.parametrize("balance", [1.0, 0.0])
def test_biped_corners_build_stand_and_stay_finite(balance: float) -> None:
    tpl = _lib().get("biped")
    for values in _corners(tpl):
        values["balance"] = balance
        data = _side_scene(values)
        quick = quick_check(data)
        assert quick.ok and quick.spec is not None, (values, [str(i) for i in quick.errors])
        assert [i for i in quick.spec.issues if i.severity == "error"] == []
        verdict = alive(quick.spec)
        assert verdict.ok, (values, verdict.reason)
        ep = Episode(quick.spec, None, Trial(3))
        for _ in range(60):
            ep.step([0.0] * len(quick.spec.outputs))
        for body in ep.bodies.values():
            assert all(math.isfinite(float(x)) for x in (*body.position, body.angle, *body.velocity)), values


def test_biped_corner_sizes_scale_the_body_and_keep_the_feet_above_the_ground() -> None:
    tpl = _lib().get("biped")
    for size in (tpl.param("size").min, 1.0, tpl.param("size").max):
        parts = T.expand(tpl, dict(tpl.defaults(), size=size))
        torso = next(b for b in parts["bodies"] if b["id"] == "torso")
        assert torso["shapes"][0]["h"] == pytest.approx(0.7 * size, abs=1e-6)
        assert parts["spawn"]["y"] == pytest.approx(1.05 * size, abs=1e-6)
        motor = next(a for a in parts["actuators"] if a["id"] == "m_hip_l")
        assert motor["max_torque"] == pytest.approx(160.0 * size**3, abs=1e-3)


# ---------------------------------------------------------------------------
# 몸 틀 disc와 공 넣기 연습(drill)
# ---------------------------------------------------------------------------


def test_disc_template_uses_move_and_is_format_4() -> None:
    tpl = _lib().get("disc")
    assert (tpl.kind, tpl.world, tpl.root) == ("body", "top", "disc")
    parts = T.expand(tpl, tpl.defaults())
    (move,) = parts["actuators"]
    assert (move["kind"], move["mode"], move["body"]) == ("move", "xy", "disc")
    assert move["keys_up"] and move["keys_down"] and move["keys_left"] and move["keys_right"]
    for values in _corners(tpl):
        p = T.expand(tpl, values)
        assert p["actuators"][0]["max_force"] > 0 and p["actuators"][0]["max_speed"] > 0


def test_drill_recipe_builds_the_ball_drill() -> None:
    recipe = RC.load_builtin("drill")
    data = RC.build(recipe)
    assert data["version"] == 4 and data["world"]["view"] == "top" and data["world"]["damping"] == 0.5
    quick = quick_check(data)
    assert quick.ok and quick.spec is not None and quick.spec.issues == [], [str(i) for i in quick.spec.issues]
    assert alive(quick.spec).ok
    bodies = {b["id"]: b for b in data["bodies"]}
    assert bodies["ball"]["pos"] == [0.0, 0.0]
    assert {r["id"]: r["pos"] for r in data["regions"]} == {"goal": [6.5, 0.0]}
    assert data["agent"]["spawn"]["x"] == -3.5
    spec = quick.spec
    assert spec.outputs == ["go.x", "go.y"]
    assert [r["kind"] for r in data["editor"]["rules"]] == ["near", "reach"]
    assert "train" not in data["editor"]
    # 가만히 있으면 공은 골에 들어가지 않는다(점수 = −(공과 골의 가장 가까운 거리))
    ep = Episode(spec, None, Trial(1))
    while not ep.done:
        ep.step([0.0, 0.0])
    assert ep.result().score < 0


def test_starters_show_running_and_drill_first() -> None:
    cards = RC.starters()
    assert [c.id for c in cards][:4] == ["running", "drill", "blank_side", "blank_top"]
    assert {c.id: c.status for c in cards}["running"] in ("ok", "experimental")


def test_recipe_files_round_trip_and_adjust_changes_nothing() -> None:
    for name in ("running", "drill"):
        recipe = RC.load_builtin(name)
        fixed, notes = RC.adjust(copy.deepcopy(recipe))
        assert notes == []
        assert env_hash(parse_env(RC.build(fixed))) == env_hash(parse_env(RC.build(recipe)))

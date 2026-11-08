"""v2.0 런타임 결정성·호환성·검사 규칙(설계 3절, 2.3, 4.6, 4.8; T17·T18·T20·T24~T30).

- 점프 게임·글라이더 학습이 워커 수·이어하기·재현 묶음과 무관하게 같다(T17, T18), 같은 (유전자, 시험)은 새 프로세스
  첫 평가와 다른 시험 뒤에서 같다(T20).
- 날개 모양은 늘 단순 다각형(T24), 모양별 밀도(T25), 뇌 없는 환경(T26).
- 검사 규칙 줄마다 심각도와 문구(T27), Space 키(T28).
- v1 파일은 새 코드 길을 지나지 않고 결과가 v1.2와 같다(T29, 같은 플랫폼에서만 숫자 비교), 관찰 격자 틀(T30).
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab import paths
from evolab.app import strings as S
from evolab.engine import observe as observing
from evolab.engine import seeds
from evolab.engine import trainer as trainer_mod
from evolab.engine.evaluate import OBSERVE_POSITIONS, evaluate
from evolab.engine.pool import WorkerPool
from evolab.engine.runsetup import create_run
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.policy.genome import layout_for
from evolab.runs.bundle import compare_stats, export_bundle, reproduce
from evolab.runs.store import RunStore
from evolab.spec import symbols as sy
from evolab.spec.compiled import check_env
from evolab.spec.io import compile_env, env_hash, parse_env
from evolab.world.physics2d import geometry as geo
from evolab.world.physics2d.episode import Episode, Trial

ENVS = paths.builtin_env_dir()
JUMP_TEXT = (ENVS / "jump.evoenv").read_text(encoding="utf-8")
GLIDER_TEXT = (ENVS / "glider.evoenv").read_text(encoding="utf-8")
V2_TEXTS = {"jump": JUMP_TEXT, "glider": GLIDER_TEXT}
TIME_COLUMNS = set(trainer_mod.TIME_COLUMNS)
REGRESS = Path(__file__).parent / "data" / "v1_regress.npz"


def small_config(**over: Any) -> TrainConfig:
    c = TrainConfig(popsize=6, trials=2, validate_every=36, validation_seeds=4, checkpoint_every=2, run_seed=7)
    for k, v in over.items():
        setattr(c, k, v)
    return c


def strip(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in r.items() if k not in TIME_COLUMNS} for r in rows]


# ---------------------------------------------------------------------------
# T17, T18, T20: 결정성
# ---------------------------------------------------------------------------


def run_trainer(text: str, workers: int, gens: int, store: RunStore | None = None) -> Trainer:
    with WorkerPool(workers) as pool:
        t = Trainer(text, small_config(max_generations=gens), pool, store)
        rows: list[dict[str, Any]] = []
        t.run(rows.append)
        t.rows = rows  # type: ignore[attr-defined]
    return t


@pytest.mark.parametrize("name", ["jump", "glider"])
def test_training_is_independent_of_worker_count_and_resume(name: str, tmp_path: Path) -> None:
    """T17: 워커 0·2개 학습 3세대의 통계(시간 열 제외)·챔피언이 같고, 3세대 + 이어서 3세대 = 6세대 한 번에.

    워커 2개로 3세대를 돌려 멈춘 실행을 워커 0개로 6세대까지 이어 가고, 워커 0개로 한 번에 6세대 돈 실행과 비교한다."""
    text = V2_TEXTS[name]
    spec = compile_env(text)

    def store(label: str) -> RunStore:
        s = RunStore.create(tmp_path, label, {"env_name": spec.file.name})
        s.add_env_version(text, 0)
        return s

    straight = run_trainer(text, 0, 6, store("한 번에"))
    halves = store("끊고 이어서")
    first = run_trainer(text, 2, 3, halves)
    rows = strip(straight.store.read_stats())  # type: ignore[union-attr]
    assert strip(first.rows) == rows[:3]  # type: ignore[attr-defined]
    assert first.state.champion.generation <= 3
    with WorkerPool(0) as pool:
        resumed = Trainer.resume(halves, pool, None, max_generations=6)
        resumed.run()
    assert strip(halves.read_stats()) == rows
    assert resumed.learner.best().tobytes() == straight.learner.best().tobytes()
    assert resumed.state.champion.score == straight.state.champion.score
    assert halves.champion_generations("champion") == straight.store.champion_generations("champion")  # type: ignore[union-attr]


@pytest.mark.parametrize("name", ["jump", "glider"])
def test_bundle_reproduces_identical_stats(name: str, tmp_path: Path) -> None:
    """T18: 작은 실행의 재현 묶음을 다시 돌리면 compare_stats가 None."""
    text = V2_TEXTS[name]
    config = small_config(max_generations=3)
    store = create_run(tmp_path / "원래", text, compile_env(text), config, 0)
    with WorkerPool(0) as pool:
        t = Trainer(text, config, pool, store)
        t.run()
        t.checkpoint()
    bundle = export_bundle(store.root, tmp_path / "묶음.zip")
    new = reproduce(bundle, tmp_path / "다시", workers=0)
    assert compare_stats(store.read_stats(), RunStore.open(new).read_stats()) is None
    assert len(store.read_stats()) == 3


def _digest(text: str, genome_rows: list[list[float]], trial_seed: int) -> str:
    spec = compile_env(text)
    layout = layout_for(spec)
    g = np.array(genome_rows, dtype=np.float64)
    r = evaluate(spec, layout, g, [(0, Trial(trial_seed))], record=[True], every=1)
    h = hashlib.sha256(r.scores.tobytes() + r.steps.tobytes() + r.blocks.tobytes())
    for tr in r.trajectories or []:
        h.update(tr.poses.tobytes() + tr.outputs.tobytes())
    h.update(repr((r.end_events, r.outcomes)).encode())
    return h.hexdigest()


@pytest.mark.parametrize("name", ["jump", "glider"])
def test_same_pair_same_result_first_in_new_process_and_after_other_trials(name: str) -> None:
    """T20: 같은 (유전자, 시험)이 새 프로세스의 첫 평가와 다른 시험 20개 뒤에서 결과 전체가 같다."""
    text = V2_TEXTS[name]
    spec = compile_env(text)
    layout = layout_for(spec)
    genome = (layout.initial() + 0.3 * np.random.default_rng(4).standard_normal(layout.dim)).tolist()
    seed = seeds.judge_trials(3)[2].seed
    others = [(0, t) for t in seeds.validation_trials(20)]
    evaluate(spec, layout, np.array([genome]), others)  # 다른 시험 20개를 먼저(같은 프로세스)
    here = _digest(text, [genome], seed)
    code = (
        "import json, sys; sys.path.insert(0, %r); from test_runtime_v2 import _digest; "
        "print(_digest(open(%r, encoding='utf-8').read(), json.loads(%r), %d))"
    ) % (str(Path(__file__).parent), str(ENVS / f"{name}.evoenv"), json.dumps([genome]), seed)
    fresh = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert fresh == here


# ---------------------------------------------------------------------------
# T24, T25: 날개 모양, 모양별 밀도
# ---------------------------------------------------------------------------


def body_env(shapes: list[dict[str, Any]], *, body_type: str = "dynamic", owner: str = "agent",
             density: float = 1.0) -> Any:
    data = {"name": "모양 시험", "world": {"gravity": [0.0, 0.0]},
            "bodies": [{"id": "b", "owner": owner, "type": body_type, "shapes": shapes,
                        "material": {"density": density}}],
            "scoring": {"blocks": [{"id": "t", "kind": "survival_time"}]}}
    return compile_env(json.dumps(data))


def test_wing_shape_vertices_are_wing_points() -> None:
    wing = {"kind": "wing", "le": [0.4, 0.1], "chord": 1.2, "angle": 0.1, "thick": 0.05, "camber": 0.02}
    ep = Episode(body_env([wing]), None, Trial(1))
    shape = ep.geometry().bodies[0].shapes
    assert len(shape) == 1 and shape[0].points == geo.wing_points((0.4, 0.1), 1.2, 0.1, 0.05, 0.02)
    le, up, te, lo = shape[0].points
    assert le == (0.4, 0.1)
    assert te == pytest.approx((0.4 - 1.2 * math.cos(0.1), 0.1 - 1.2 * math.sin(0.1)))
    assert geo.signed_area(shape[0].points) > 0  # 반시계


def test_wing_points_are_always_simple() -> None:
    """T24: 범위 전체 무작위 20만 개(극단 휨·얇은 두께·잘린 값 포함)가 모두 단순 다각형이고 넓이가 있다."""
    rng = np.random.default_rng(24)
    n = 200_000
    cols = [rng.uniform(-3, 3, n), rng.uniform(-3, 3, n), rng.uniform(-0.5, 3.0, n), rng.uniform(-math.pi, math.pi, n),
            rng.uniform(-0.1, 1.2, n), rng.uniform(-1.5, 1.5, n)]
    for lx, ly, chord, angle, thick, camber in zip(*cols, strict=True):
        pts = geo.wing_points((lx, ly), chord, angle, thick, camber)
        assert geo.is_simple(pts) and geo.signed_area(pts) > 1e-9


def test_wing_with_big_camber_builds() -> None:
    wing = {"kind": "wing", "chord": 1.0, "thick": 0.04, "camber": 0.3}
    ep = Episode(body_env([wing]), None, Trial(1))
    assert ep.bodies["b"].mass > 0 and len(ep.geometry().bodies[0].shapes) >= 2  # 오목해서 볼록 조각으로 나뉨


def test_shape_density_moves_the_centre_of_gravity() -> None:
    """T25: 작은 무거운 추를 옮기면 무게 중심이 계산값만큼 옮겨진다. 질량은 모양별 질량의 합."""
    box_m = 2.0 * 0.1 * 1.0
    r = 0.07
    ball_m = math.pi * r * r * 120.0
    for x in (-0.5, 0.0, 0.8):
        shapes = [{"kind": "box", "w": 2.0, "h": 0.1}, {"kind": "circle", "r": r, "offset": [x, 0.0], "density": 120.0}]
        ep = Episode(body_env(shapes), None, Trial(1))
        b = ep.bodies["b"]
        assert b.mass == pytest.approx(box_m + ball_m, rel=1e-9)
        assert b.center_of_gravity.x == pytest.approx(ball_m * x / (box_m + ball_m), abs=1e-9)
    # 밀도가 없는 모양은 물체 재질 밀도: v1과 같은 질량
    plain = Episode(body_env([{"kind": "box", "w": 2.0, "h": 0.1}], density=3.0), None, Trial(1))
    assert plain.bodies["b"].mass == pytest.approx(2.0 * 0.1 * 3.0, rel=1e-12)


def test_shape_density_applies_to_every_convex_piece_of_a_shape() -> None:
    concave = {"kind": "poly", "points": [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], "density": 5.0}
    ep = Episode(body_env([concave, {"kind": "box", "w": 0.2, "h": 0.2, "offset": [-1, -1]}]), None, Trial(1))
    assert ep.bodies["b"].mass == pytest.approx(3.0 * 5.0 + 0.04 * 1.0, rel=1e-9)


def test_static_body_ignores_shape_density() -> None:
    shapes = [{"kind": "box", "w": 1.0, "h": 1.0, "density": 50.0}]
    data = {"name": "정적", "bodies": [{"id": "wall", "owner": "world", "type": "static", "shapes": shapes}],
            "scoring": {"blocks": [{"id": "t", "kind": "survival_time"}]}}
    issues = check_env(parse_env(json.dumps(data)))
    assert ("warning", S.SPEC_SHAPE_DENSITY_STATIC.format(id="wall")) in [(i.severity, i.message) for i in issues]
    ep = Episode(compile_env(json.dumps(data)), None, Trial(1))
    assert math.isinf(ep.bodies["wall"].mass)


# ---------------------------------------------------------------------------
# T26: 뇌 없는 환경
# ---------------------------------------------------------------------------


def test_brainless_glider() -> None:
    spec = compile_env(GLIDER_TEXT)
    assert [(i.severity, i.message) for i in spec.issues] == [("info", S.SPEC_BRAINLESS)]
    assert spec.inputs == [] and spec.outputs == [] and spec.warnings == [] and spec.infos == spec.issues
    layout = layout_for(spec)
    assert layout.dim == 8 and layout.n_body == 8 and layout.brain.n_params == 0
    ep = Episode(spec, layout.body_values(layout.initial()), Trial(1))
    assert ep.observe().shape == (0,)
    ep.step([])
    ep.step(np.zeros(0))
    assert ep.steps == 2 and ep.applied_outputs == []


def test_no_outputs_without_body_params_still_warns() -> None:
    data = json.loads(GLIDER_TEXT)
    for p in data["params"]:
        p["evolve"] = False
    msgs = [(i.severity, i.message) for i in check_env(parse_env(json.dumps(data)))]
    assert ("warning", S.SPEC_NO_SENSORS) in msgs and ("warning", S.SPEC_NO_ACTUATORS) in msgs
    assert ("info", S.SPEC_BRAINLESS) not in msgs


# ---------------------------------------------------------------------------
# T27, T28: 검사 규칙
# ---------------------------------------------------------------------------


def issues_of(data: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [(i.severity, i.where, i.message) for i in check_env(parse_env(json.dumps(data, ensure_ascii=False)))]


def jump_data() -> dict[str, Any]:
    return json.loads(JUMP_TEXT)


def glider_data() -> dict[str, Any]:
    return json.loads(GLIDER_TEXT)


def lift(motion: dict[str, Any], body_type: str = "kinematic", owner: str = "world") -> dict[str, Any]:
    body: dict[str, Any] = {"id": "lift", "owner": owner, "type": body_type, "pos": [5.0, 1.0],
                            "shapes": [{"kind": "box", "w": 2.0, "h": 0.2}]}
    if motion:
        body["motion"] = motion
    return body


PATH = {"kind": "path", "points": [[0, 0], [0, 3]], "speed": 1.0, "mode": "pingpong"}


def with_body(body: dict[str, Any]) -> dict[str, Any]:
    data = jump_data()
    data["bodies"].append(body)
    return data


@pytest.mark.parametrize("case", [
    "agent_kinematic", "motion_not_kinematic", "kinematic_no_motion", "still_path", "still_speed", "wave_period",
    "motion_param", "jump_body", "aero_body", "jump_ground", "jump_self", "jump_speed", "aero_no_edges",
    "aero_wings_only", "aero_range", "aero_stall", "wing_size", "shape_density_negative", "shape_density_static",
    "region_density", "obstacles_gap", "obstacles_start",
])
def test_v2_rules(case: str) -> None:
    """T27: 설계 2.3의 v2.0 줄마다 심각도·위치·문구."""
    data = jump_data()
    if case == "agent_kinematic":
        data["bodies"][1]["type"] = "kinematic"
        want = ("error", "bodies[1].type", S.SPEC_KINEMATIC_AGENT.format(id="runner"))
    elif case == "motion_not_kinematic":
        data = with_body(lift(PATH, body_type="static"))
        want = ("error", "bodies[2].motion", S.SPEC_MOTION_NOT_KINEMATIC.format(id="lift"))
    elif case == "kinematic_no_motion":
        data = with_body(lift({}))
        want = ("warning", "bodies[2].type", S.SPEC_KINEMATIC_NO_MOTION.format(id="lift"))
    elif case == "still_path":
        data = with_body(lift({"kind": "path", "points": [[1, 1], [1, 1]]}))
        want = ("warning", "bodies[2].motion", S.SPEC_MOTION_STILL.format(id="lift"))
    elif case == "still_speed":
        data = with_body(lift({**PATH, "speed": 0.0}))
        want = ("warning", "bodies[2].motion", S.SPEC_MOTION_STILL.format(id="lift"))
    elif case == "wave_period":
        data = with_body(lift({"kind": "wave", "period": 0.0}))
        want = ("error", "bodies[2].motion.period", S.SPEC_MOTION_PERIOD.format(id="lift"))
    elif case == "motion_param":
        data = with_body(lift({**PATH, "speed": "=p.k"}))
        data["params"] = [{"id": "k", "value": 1.0}]
        found = issues_of(data)
        assert any(s == "error" and w == "bodies[2].motion.speed" for s, w, _ in found), found
        return
    elif case == "jump_body":
        data["actuators"][0]["body"] = "ground"
        want = ("error", "actuators[0].body", S.SPEC_ACTUATOR_BODY.format(ref="ground"))
    elif case == "aero_body":
        data["actuators"].append({"kind": "aero", "id": "air", "body": "obs"})
        want = ("error", "actuators[1].body", S.SPEC_ACTUATOR_BODY.format(ref="obs"))
    elif case == "jump_ground":
        data["actuators"][0]["ground"] = ["tag.nothing"]
        want = ("error", "actuators[0].ground[0]", S.SPEC_REF_BODY_OR_TAG.format(ref="tag.nothing"))
    elif case == "jump_self":
        data["actuators"][0]["ground"] = ["tag.runner"]
        want = ("error", "actuators[0].ground[0]", S.SPEC_AGENT_SELF_CONTACT.format(a="runner", b="tag.runner"))
    elif case == "jump_speed":
        data["actuators"][0]["speed"] = -1.0
        want = ("error", "actuators[0].speed", S.SPEC_NEGATIVE.format(name="speed"))
    elif case == "aero_no_edges":
        data["bodies"][1]["shapes"] = [{"kind": "circle", "r": 0.4}]
        data["actuators"].append({"kind": "aero", "id": "air", "body": "runner"})
        want = ("warning", "actuators[1].surfaces", S.SPEC_AERO_NO_EDGES.format(id="air", body="runner"))
    elif case == "aero_wings_only":
        data["actuators"].append({"kind": "aero", "id": "air", "body": "runner", "surfaces": "wings"})
        want = ("warning", "actuators[1].surfaces", S.SPEC_AERO_NO_EDGES.format(id="air", body="runner"))
    elif case == "aero_range":
        data["actuators"].append({"kind": "aero", "id": "air", "body": "runner", "span": -1.0})
        want = ("error", "actuators[1].span", S.SPEC_AERO_RANGE.format(id="air", name="span"))
    elif case == "aero_stall":
        data["actuators"].append({"kind": "aero", "id": "air", "body": "runner", "stall": 1.6})
        want = ("error", "actuators[1].stall", S.SPEC_AERO_RANGE.format(id="air", name="stall"))
    elif case == "wing_size":
        data["bodies"][1]["shapes"].append({"kind": "wing", "chord": 0.0})
        want = ("error", "bodies[1].shapes[1]", S.SPEC_WING_SIZE.format(id="runner"))
    elif case == "shape_density_negative":
        data["bodies"][1]["shapes"][0]["density"] = -2.0
        want = ("error", "bodies[1].shapes[0].density", S.SPEC_NEGATIVE.format(name="density"))
    elif case == "shape_density_static":
        data["bodies"][0]["shapes"][0]["density"] = 2.0
        want = ("warning", "bodies[0].shapes[0].density", S.SPEC_SHAPE_DENSITY_STATIC.format(id="ground"))
    elif case == "region_density":
        data["regions"] = [{"id": "zone", "shape": {"kind": "box", "w": 1, "h": 1, "density": 2.0}}]
        want = ("warning", "regions[0].shape.density", S.SPEC_REGION_DENSITY.format(id="zone"))
    elif case == "obstacles_gap":
        data["generators"][0]["gap"] = [0.0, 2.0]
        want = ("error", "generators[0].gap", S.SPEC_OBSTACLES_GAP.format(id="obs"))
    else:
        data["generators"][0]["x"] = 1.5
        want = ("warning", "generators[0].x", S.SPEC_OBSTACLES_START.format(id="obs"))
    found = issues_of(data)
    assert want in found, found


def test_obstacles_start_uses_the_direction() -> None:
    data = jump_data()
    data["generators"][0].update({"x": 3.0})
    assert issues_of(data) == []  # 3m 앞: 괜찮음
    data["agent"]["spawn"]["x"] = 1.5
    assert ("warning", "generators[0].x", S.SPEC_OBSTACLES_START.format(id="obs")) in issues_of(data)
    data["agent"]["spawn"]["x"] = "=0"  # 수식이면 검사하지 않는다
    assert issues_of(data) == []


def test_moving_bodies_with_spin_or_formulas_are_not_still() -> None:
    for motion in ({"kind": "path", "points": [[1, 1], [1, 1]], "spin": 1.0},
                   {**PATH, "speed": "=r.v"}, {**PATH, "speed": 0.0, "accel": 0.5}):
        data = with_body(lift(motion))
        data["randomize"].append({"id": "v", "low": 1.0, "high": 2.0})
        assert not [i for i in issues_of(data) if i[2] == S.SPEC_MOTION_STILL.format(id="lift")]


def test_kinematic_world_body_with_motion_compiles_and_runs() -> None:
    data = with_body(lift(PATH))
    spec = compile_env(json.dumps(data, ensure_ascii=False))
    assert spec.issues == []
    ep = Episode(spec, None, Trial(1))
    ep.step([0.0])
    assert ep.bodies["lift"].velocity.y == pytest.approx(1.0)


def test_space_key_is_for_jump_only() -> None:
    """T28: 점프 키 Space는 통과, 추진기·모터·차량의 Space는 SPEC_KEY_SPACE_JUMP_ONLY, R·N은 어디서나 SPEC_KEY_RESERVED."""
    data = jump_data()
    assert data["actuators"][0]["keys"] == ["Up", "W", "Space"] and issues_of(data) == []
    data["actuators"].append({"kind": "thruster", "id": "push", "body": "runner", "max_force": 10.0, "keys": ["Space"]})
    assert ("error", "actuators[1].keys[0]", S.SPEC_KEY_SPACE_JUMP_ONLY.format(id="push")) in issues_of(data)
    for key in ("R", "N"):
        d = jump_data()
        d["actuators"][0]["keys"] = [key]
        assert ("error", "actuators[0].keys[0]", S.SPEC_KEY_RESERVED.format(key=key)) in issues_of(d)
    lander = json.loads((ENVS / "lander.evoenv").read_text(encoding="utf-8"))
    lander["actuators"][1]["keys"] = ["Space"]
    assert ("error", "actuators[1].keys[0]", S.SPEC_KEY_SPACE_JUMP_ONLY.format(id=lander["actuators"][1]["id"])) in \
        issues_of(lander)
    hill = json.loads((ENVS / "hill_car.evoenv").read_text(encoding="utf-8"))
    motor = next(k for k, a in enumerate(hill["actuators"]) if a["kind"] == "motor")
    hill["actuators"][motor]["keys_pos"] = ["Space"]
    assert any(s == "error" and m == S.SPEC_KEY_SPACE_JUMP_ONLY.format(id=hill["actuators"][motor]["id"])
               for s, _, m in issues_of(hill))
    track = json.loads((ENVS / "track_car.evoenv").read_text(encoding="utf-8"))
    vehicle = next(k for k, a in enumerate(track["actuators"]) if a["kind"] == "vehicle")
    track["actuators"][vehicle]["keys_left"] = ["Space"]
    assert any(s == "error" and m == S.SPEC_KEY_SPACE_JUMP_ONLY.format(id=track["actuators"][vehicle]["id"])
               for s, _, m in issues_of(track))


def test_key_allowed() -> None:
    env = parse_env(JUMP_TEXT)
    jump = env.actuators[0]
    lander = parse_env((ENVS / "lander.evoenv").read_text(encoding="utf-8"))
    thruster = next(a for a in lander.actuators if a.kind == "thruster")
    assert sy.key_allowed(jump, "Space") and not sy.key_allowed(thruster, "Space")
    assert sy.key_allowed(jump, "Up") and sy.key_allowed(thruster, "Up")
    for key in ("R", "N", "Esc", "Nope"):
        assert not sy.key_allowed(jump, key) and not sy.key_allowed(thruster, key)


# ---------------------------------------------------------------------------
# T29, T30: v1 파일 경로 그대로, 관찰 격자 틀
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["lander", "track_car", "hill_car"])
def test_v1_files_skip_the_new_code_paths(name: str) -> None:
    spec = compile_env((ENVS / f"{name}.evoenv").read_text(encoding="utf-8"))
    layout = layout_for(spec)
    ep = Episode(spec, layout.body_values(layout.initial()) if layout.n_body else None, Trial(3))
    assert ep._kinematic == [] and ep._jumps == [] and ep._aero == []
    assert ep.aero_clamped == 0 and ep.aero_edge_count == 0


def _same_platform(data: Any) -> bool:
    return str(data["platform"]) == f"{sys.platform}-{platform.machine()}"


def test_v1_results_match_the_v12_reference() -> None:
    """T29: 착륙선·트랙·언덕차 무작위 유전자 8 x 시험 3의 점수·스텝·블록·궤적이 v1.2 코드로 만든 값과 같다.

    명세 「정답 해시를 박지 않음」: 숫자 비교는 자료를 만든 플랫폼(npz의 platform)에서만 한다."""
    data = np.load(REGRESS, allow_pickle=False)
    if not _same_platform(data):
        pytest.skip(f"reference made on {data['platform']}")
    trials = [Trial(int(s)) for s in data["trial_seeds"]]
    for name in ("lander", "track_car", "hill_car"):
        text = str(data[f"{name}.env_text"])
        assert text == (ENVS / f"{name}.evoenv").read_text(encoding="utf-8")
        spec = compile_env(text)
        layout = layout_for(spec)
        genomes = data[f"{name}.genomes"]
        pairs = [(g, t) for g in range(len(genomes)) for t in trials]
        r = evaluate(spec, layout, genomes, pairs, record=[True] * len(pairs), every=1)
        assert r.scores.tobytes() == data[f"{name}.scores"].tobytes(), name
        assert np.array_equal(r.steps, data[f"{name}.steps"]), name
        assert list(r.end_events) == list(data[f"{name}.end_events"]), name
        assert list(r.outcomes) == list(data[f"{name}.outcomes"]), name
        assert r.blocks.tobytes() == data[f"{name}.blocks"].tobytes(), name
        assert r.action_abs.tobytes() == data[f"{name}.action_abs"].tobytes(), name
        assert r.trajectories is not None
        assert [tr.frames for tr in r.trajectories] == data[f"{name}.frames"].tolist(), name
        for k, tr in enumerate(r.trajectories):
            assert hashlib.sha256(tr.poses.tobytes()).digest() == bytes(data[f"{name}.poses_sha256"][k]), (name, k)
            assert hashlib.sha256(tr.outputs.tobytes()).digest() == bytes(data[f"{name}.outputs_sha256"][k]), (name, k)
        final = np.concatenate([tr.poses[-1] for tr in r.trajectories])
        assert final.tobytes() == data[f"{name}.final_poses"].tobytes(), name


def test_v1_reference_inputs_are_the_current_examples() -> None:
    """T29 자료의 환경 글이 지금 예제 파일과 같고 해시도 같다(다른 플랫폼에서도 도는 부분)."""
    data = np.load(REGRESS, allow_pickle=False)
    for name in ("lander", "track_car", "hill_car"):
        text = str(data[f"{name}.env_text"])
        assert env_hash(parse_env(text)) == env_hash(parse_env((ENVS / f"{name}.evoenv").read_text(encoding="utf-8")))


def test_obstacle_row_observation_frame_is_clipped_to_the_world() -> None:
    """T30: 장애물 줄(수백 m)이 있어도 시험 0번 관찰 격자 틀은 세계 경계로 잘리고, 관찰을 켜도 점수가 같다."""
    spec = compile_env(JUMP_TEXT)
    version = env_hash(spec.file)[:16]
    trial0 = seeds.train_trials(1, 0, 1)[0]
    frame, key = observing.world_frame(spec, version, trial0)
    assert frame is not None and key is not None
    x0, y0, x1, y1 = spec.file.world.bounds
    fx, fy, fw, fh = frame.rect
    assert fx >= x0 - 1e-9 and fy >= y0 - 1e-9
    assert fx + fw <= x1 + frame.cell + 1e-9 and fy + fh <= y1 + frame.cell + 1e-9
    plain = json.loads(JUMP_TEXT)
    plain["generators"] = []
    plain["randomize"] = []
    plain["sensors"] = [s for s in plain["sensors"] if s["kind"] != "rays" and s["body"] != "obs"]
    plain["events"] = []
    plain["termination"] = [t for t in plain["termination"] if t["event"] != "hit"]
    frame_plain, _ = observing.world_frame(compile_env(json.dumps(plain)), version, trial0)
    assert frame_plain is not None and frame.nx == frame_plain.nx and frame.x0 == frame_plain.x0
    layout = layout_for(spec)
    genomes = 0.5 * np.random.default_rng(30).standard_normal((3, layout.dim))
    pairs = [(g, t) for g in range(3) for t in seeds.train_trials(1, 0, 2)]
    off = evaluate(spec, layout, genomes, pairs)
    on = evaluate(spec, layout, genomes, pairs, observe=[OBSERVE_POSITIONS] * len(pairs), frame=frame.as_array())
    assert on.scores.tobytes() == off.scores.tobytes() and on.steps.tolist() == off.steps.tolist()

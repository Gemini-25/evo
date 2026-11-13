"""v6.0 런타임 가속(결과 비트 동일)의 단위 테스트 (v6 설계 8.1 T5~T8, T10~T13, 4.1~4.6·4.8, R 묶음).

- T5: C 함수 길(_CP_FAST)과 공개 API 길을 같은 Episode로 나란히 돌려 스텝마다 모든 물체 상태·관측·실제 출력·레이·센서
  원래 값·블록 값이 바이트 같다(내장 환경 6개 + 모든 부품이 든 시험 세계 + 정해진 움직임 발판).
- T6: 고정한 pymunk(uv.lock)에서 빠른 길이 켜진다: _CP_FAST, 모든 물체의 cpBody 포인터, 모터 포인터, 둘째 스텝부터 _raw_space.
- T7: 긴 에피소드 내내 pymunk Space의 파이썬 쪽 목록(_add_later, _remove_later, _post_step_callbacks, _bodies_to_check,
  _removed_shapes)이 비어 있고 잠기지 않았다(공개 Space.step을 건너뛰고 cpSpaceStep을 바로 부르는 전제).
- T8: 특수화한 몸 상태 값·센서가 옛 _state·body_state 식과 모든 항목·좌표계에서 같은 값(무작위 상태, NaN·±무한대·−0.0).
- T10: 사건 한 루프가 옛 _fire 순서와 같은 값(무작위 조건 열, hold 0·1·여러 스텝, count·since 필요 여부 조합).
- T11: 레이 필터 한 번: 모든 모양의 (그룹, 범주, 마스크)와 모양 수가 오라클 Episode와 같고, 레이 꼬리표 31개 넘침 오류 그대로.
- T12: 공기역학: 내장 글라이더·다각형 날개 글라이더 1,000스텝의 물체 힘·토크·aero_clamped가 오라클과 같다(물리 스텝 직전에 읽음).
- T13: 접촉 맞음 기억이 모든 내장 환경의 (모양, 선택자)에서 옛 _matches와 같다.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pytest

from evolab import paths
from evolab.engine import seeds
from evolab.policy.brain import unpack
from evolab.policy.genome import GenomeLayout, layout_for
from evolab.spec.compiled import EnvSpec
from evolab.spec.io import compile_env, load_env
from evolab.world.physics2d import episode as episode_mod
from evolab.world.physics2d.episode import Episode, Trial
from tests.test_speed_identity import NAMES, body_of, champion, random_genomes, same_now, same_static

ENVS = paths.builtin_env_dir()


def builtin(name: str) -> tuple[EnvSpec, GenomeLayout]:
    spec = load_env(ENVS / f"{name}.evoenv")
    return spec, layout_for(spec)


def parts_world(lift: bool = True) -> EnvSpec:
    """모든 부품을 없어질 수 있는 물체에 단 시험 세계(tests/test_structure_runtime.PARTS) + 도는 정해진 움직임 발판."""
    from tests.test_structure_runtime import PARTS

    data = copy.deepcopy(PARTS)
    if lift:
        data["bodies"].append({"id": "lift", "owner": "world", "type": "kinematic", "pos": [3.0, 0.3],
                               "shapes": [{"kind": "box", "w": 1.0, "h": 0.2}], "tags": ["ground"],
                               "motion": {"kind": "path", "points": [[0, 0], [0, 1]], "speed": 0.5,
                                          "mode": "pingpong", "spin": 0.3}})
    return compile_env(json.dumps(data, ensure_ascii=False))


def public_episode(monkeypatch: pytest.MonkeyPatch, spec: EnvSpec, body: Any, trial: Trial) -> Episode:
    """C 함수 길을 끈 Episode(만들 때 _CP_FAST를 읽어 길을 정한다)."""
    with monkeypatch.context() as mp:
        mp.setattr(episode_mod, "_CP_FAST", False)
        return Episode(spec, body, trial)


def run_fast_and_public(monkeypatch: pytest.MonkeyPatch, spec: EnvSpec, layout: GenomeLayout, genome: np.ndarray,
                        trial: Trial, *, values: bool, max_steps: int | None = None) -> int:
    body = body_of(layout, genome)
    a = Episode(spec, body, trial)
    b = public_episode(monkeypatch, spec, body, trial)
    assert all(r is not None for r in a._raw.values()) and all(r is None for r in b._raw.values())
    same_static(a, b)
    same_now(a, b, deep=True)
    brain = unpack(layout.brain, layout.brain_params(genome[None, :]))
    obs = np.zeros((1, layout.brain.n_in), dtype=np.float64)
    while not b.done and (max_steps is None or b.steps < max_steps):
        oa, ob = a.observe(), b.observe()
        assert oa.tobytes() == ob.tobytes(), b.steps
        obs[0] = ob
        act = layout.to_actions(brain.forward(obs))
        if values:
            a.step_values(act.tolist()[0])
            b.step_values(act.tolist()[0])
        else:
            a.step(act[0])
            b.step(act[0])
        same_now(a, b, deep=True)
    assert repr(a.result()) == repr(b.result())
    return int(b.steps)


@pytest.mark.parametrize("name", NAMES)
def test_t5_fast_and_public_paths_match_step_by_step(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """T5: 내장 환경마다 챔피언 × 검증 시험 0, 무작위 유전자 2 × 학습 시험."""
    spec, layout = builtin(name)
    total = run_fast_and_public(monkeypatch, spec, layout, champion(name), seeds.validation_trials(1)[0],
                                values=False, max_steps=1500)
    for i, g in enumerate(random_genomes(layout, 2, seed=55)):
        total += run_fast_and_public(monkeypatch, spec, layout, g, seeds.train_trials(5, i, 1)[0], values=i == 1)
    assert total > 300


def test_t5_parts_world_fast_and_public_paths_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """T5: 모든 부품 + 도는 정해진 움직임 발판, 여러 몸 구조(없는 부품 포함)."""
    spec = parts_world()
    layout = layout_for(spec)
    rng = np.random.default_rng(56)
    genomes = layout.initial()[None, :] + rng.uniform(-1.5, 1.5, (6, layout.dim))
    absent = set()
    for i, g in enumerate(genomes):
        absent.add(Episode(spec, body_of(layout, g), Trial(i)).absent)
        run_fast_and_public(monkeypatch, spec, layout, g, Trial(seeds.derive(56, i)), values=i % 2 == 0)
    assert len(absent) >= 3


NAMES_V62 = ("running",)
"""6.2.0 형식 4의 내장 환경(밀기·균형·세계 감쇠). 옛 환경의 대조 시험(NAMES)은 건드리지 않는다."""


def moves_world(damping: float = 0.6) -> EnvSpec:
    """형식 4 부품 시험 세계: 밀기 셋(x·xy·heading) + 균형 + 세계 감쇠 < 1 + 움직이는 공."""
    from evolab.spec import model as m

    data = {
        "version": m.CURRENT_VERSION, "name": "moves",
        "world": {"gravity": [0.0, -9.81], "damping": damping, "max_time": 4.0, "bounds": [-20, -5, 20, 20]},
        "bodies": [
            {"id": "ground", "owner": "world", "type": "static", "tags": ["ground"],
             "shapes": [{"kind": "chain", "points": [[-20, 0], [20, 0]], "radius": 0.05}]},
            {"id": "a", "pos": [0.0, 1.0], "shapes": [{"kind": "box", "w": 0.6, "h": 0.4}]},
            {"id": "b", "pos": [0.0, 1.6], "shapes": [{"kind": "circle", "r": 0.2}]},
            {"id": "c", "pos": [0.0, 2.2], "shapes": [{"kind": "box", "w": 0.4, "h": 0.2}]},
            {"id": "ball", "owner": "world", "pos": [2.0, 0.5], "shapes": [{"kind": "circle", "r": 0.25}]},
        ],
        "joints": [{"kind": "pivot", "id": "j1", "a": "a", "b": "b", "anchor_a": [0.0, 0.3], "anchor_b": [0.0, -0.3]},
                   {"kind": "pivot", "id": "j2", "a": "b", "b": "c", "anchor_a": [0.0, 0.3], "anchor_b": [0.0, -0.3]}],
        "actuators": [
            {"kind": "move", "id": "mx", "body": "a", "mode": "x", "max_force": 40.0, "max_speed": 3.0},
            {"kind": "move", "id": "mxy", "body": "b", "mode": "xy", "max_force": 20.0, "max_speed": 2.0},
            {"kind": "move", "id": "mh", "body": "c", "mode": "heading", "max_force": 10.0, "max_speed": 2.0,
             "max_turn": 3.0},
            {"kind": "balance", "id": "bal", "body": "a", "stiffness": 30.0, "damping": 2.0, "max_torque": 20.0},
        ],
        "sensors": [{"kind": "body_state", "id": "st", "body": "a", "fields": ["x", "y", "vx", "vy", "angle"]}],
        "scoring": {"blocks": [{"kind": "progress", "id": "prog", "body": "a", "direction": 0.0, "weight": 1.0}]},
    }
    return compile_env(json.dumps(data, ensure_ascii=False))


@pytest.mark.parametrize("name", NAMES_V62)
def test_t5_v62_builtins_fast_and_public_paths_match(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """T5(6.2.0, 검토 고침): 형식 4 내장 환경도 무작위 유전자 2 × 학습 시험에서 두 길이 스텝마다 같다."""
    spec, layout = builtin(name)
    total = 0
    for i, g in enumerate(random_genomes(layout, 2, seed=57)):
        total += run_fast_and_public(monkeypatch, spec, layout, g, seeds.train_trials(5, i, 1)[0], values=i == 1,
                                     max_steps=600)
    assert total > 50  # 무작위 뇌의 달리기는 곧 넘어진다(2시험 합계 100스텝 남짓)


@pytest.mark.parametrize("damping", [0.6, 1.0])
def test_t5_moves_world_fast_and_public_paths_match(damping: float, monkeypatch: pytest.MonkeyPatch) -> None:
    """T5(6.2.0, 검토 고침): 밀기 x·xy·heading, 균형, 세계 감쇠."""
    spec = moves_world(damping)
    layout = layout_for(spec)
    rng = np.random.default_rng(58)
    for i, g in enumerate(rng.normal(0.0, 1.0, (4, layout.dim))):
        run_fast_and_public(monkeypatch, spec, layout, g, Trial(seeds.derive(58, i)), values=i % 2 == 0)


def test_t6_v62_move_pointers_are_on() -> None:
    assert episode_mod._CP_FAST
    for spec in (builtin("running")[0], moves_world()):  # 달리기는 모터·균형(밀기 없음), 시험 세계는 밀기 셋·균형
        ep = Episode(spec, None, seeds.display_trial())
        assert ep._balances
        assert all(mv.raw is not None for mv in ep._moves)
        assert ep._motor_raw is not None and len(ep._motor_raw) == len(ep._motors)
        assert set(ep._raw) == set(ep.bodies) and all(r is not None for r in ep._raw.values())
    assert len(Episode(moves_world(), None, seeds.display_trial())._moves) == 3


# ---------------------------------------------------------------------------
# T6 빠른 길이 켜짐
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_t6_fast_paths_are_on_with_the_pinned_pymunk(name: str) -> None:
    """T6: 고정한 pymunk(uv.lock)에서 _CP_FAST가 참이고, 모든 물체·힘 칸·에이전트·차량·점프·모터에 C 포인터가 있다."""
    assert episode_mod._CP_FAST
    spec, layout = builtin(name)
    ep = Episode(spec, body_of(layout, champion(name)), seeds.display_trial())
    assert set(ep._raw) == set(ep.bodies) and all(r is not None for r in ep._raw.values())
    assert all(raw is not None for _bid, _b, raw in ep._force_raw)
    assert len(ep._agent_raw) == len(ep._agent_ids) and all(r is not None for r in ep._agent_raw)
    assert all(v.raw is not None for v in ep._vehicles) and all(j.raw is not None for j in ep._jumps)
    assert ep._motor_raw is not None and len(ep._motor_raw) == len(ep._motors)
    assert all(raw is not None for raw in ep._moving_raw)


def test_t6_raw_space_is_used_from_the_second_step() -> None:
    """T6: 첫 스텝은 공개 Space.step(질량 검사), 그 뒤로는 cpSpace 포인터로 cpSpaceStep을 바로 부른다."""
    spec, layout = builtin("lander")
    ep = Episode(spec, None, seeds.display_trial())
    assert ep._raw_space is None
    ep.step([0.0] * len(ep.outputs))
    assert ep._raw_space is not None and ep._raw_space == ep.space._space


def test_t6_public_path_never_uses_the_raw_space(monkeypatch: pytest.MonkeyPatch) -> None:
    """만들 때 _CP_FAST가 거짓이었던 Episode는 그 뒤 깃발이 바뀌어도 공개 Space.step만 쓴다."""
    spec, _layout = builtin("lander")
    ep = public_episode(monkeypatch, spec, None, seeds.display_trial())
    calls = []
    real = ep.space.step
    monkeypatch.setattr(ep.space, "step", lambda dt: (calls.append(dt), real(dt)))
    for _ in range(5):
        ep.step([0.0] * len(ep.outputs))
    assert ep._raw_space is None and len(calls) == 5


# ---------------------------------------------------------------------------
# T7 cpSpaceStep을 바로 부르는 전제
# ---------------------------------------------------------------------------


def space_is_quiet(space: Any) -> bool:
    return (not space._add_later and not space._remove_later and not space._post_step_callbacks
            and not space._bodies_to_check and not space._removed_shapes and not space._locked)


@pytest.mark.parametrize("name", [*NAMES, "parts"])
def test_t7_space_python_lists_stay_empty_for_a_whole_episode(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """T7: 한 에피소드 내내(접촉 콜백이 도는 동안에도) Space의 파이썬 쪽 목록이 비어 있고 잠기지 않았다. 접촉 콜백 안에서도
    목록이 비어 있음을 콜백을 감싸 확인한다(콜백이 공간을 바꾸지 않음)."""
    if name == "parts":
        spec = parts_world()
        layout = layout_for(spec)
        genome = layout.initial()
    else:
        spec, layout = builtin(name)
        genome = champion(name)
    ep = Episode(spec, body_of(layout, genome), seeds.validation_trials(2)[1])
    seen = {"begin": 0, "separate": 0}
    begin, separate = ep._on_begin, ep._on_separate

    def check(kind: str, real: Any) -> Any:
        def wrapped(arbiter: Any, space: Any, data: Any) -> None:
            seen[kind] += 1
            real(arbiter, space, data)
            assert not space._add_later and not space._remove_later and not space._post_step_callbacks
        return wrapped

    # pymunk는 on_collision에 넘긴 함수를 붙잡으므로, Episode의 메서드를 바꿔 끼우려면 처리기를 다시 단다.
    ep.space.on_collision(None, None, begin=check("begin", begin), separate=check("separate", separate))
    brain = unpack(layout.brain, layout.brain_params(genome[None, :]))
    obs = np.zeros((1, layout.brain.n_in))
    assert ep.space._bodies_to_check  # 첫 스텝 전: pymunk가 첫 Space.step에서 질량을 검사하고 비운다
    while not ep.done and ep.steps < 8000:
        obs[0] = ep.observe()
        ep.step(layout.to_actions(brain.forward(obs))[0])
        assert space_is_quiet(ep.space), ep.steps
    assert ep.steps > 100
    if name != "track_car":  # 위에서 보는 트랙 자동차 챔피언은 벽에 닿지 않는다
        assert seen["begin"] > 0, name  # 접촉 콜백이 실제로 돌았다


# ---------------------------------------------------------------------------
# T8 몸 상태 값·센서 특수화
# ---------------------------------------------------------------------------

STATE_FIELDS = ("x", "y", "vx", "vy", "speed", "angle", "omega", "v_fwd", "v_lat")


def special_states(n: int, seed: int) -> list[tuple[float, ...]]:
    """무작위 (x, y, 각도, vx, vy, 각속도) n개. 칸마다 가끔 NaN·±무한대·±0.0·아주 큰/작은 값·±π 근처."""
    rng = np.random.default_rng(seed)
    specials = [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1e308, -1e-308, np.pi, -np.pi, 3 * np.pi,
                5e-324, 1e16]
    out = []
    for _ in range(n):
        row = [float(x) for x in rng.normal(0.0, 10.0, 6)]
        row[2] = float(rng.uniform(-20.0, 20.0))
        for k in range(6):
            if rng.random() < 0.08:
                row[k] = specials[int(rng.integers(len(specials)))]
        out.append(tuple(row))
    return out


def outcome(fn: Any) -> Any:
    """fn()의 값, 또는 예외면 ("raise", 예외 이름, 글)."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return ("raise", type(exc).__name__, str(exc))


def test_t8_specialised_body_state_values_match_the_old_formulas() -> None:
    """T8: 수식 값 "<물체>.<항목>"(sy.BODY_STATE)의 특수화 함수 = 옛 Episode._state(오라클), 무작위 상태 1만 개."""
    from evolab.spec import symbols as sy
    from tests.oracle import episode_v31 as old

    spec, _layout = builtin("lander")
    ref = old.Episode(spec, None, Trial(1))
    states = special_states(10000, seed=8)
    box: list[tuple[float, ...]] = [states[0]]
    ref._bstate = lambda bid: box[0]  # type: ignore[method-assign]
    bstate = lambda bid: box[0]  # noqa: E731
    getters = {f: episode_mod._state_getter(bstate, "hull", f) for f in sy.BODY_STATE}
    for st in states:
        box[0] = st
        for f, g in getters.items():
            assert repr(outcome(g)) == repr(outcome(lambda: ref._state("hull", f))), (f, st)


@pytest.mark.parametrize("frame", ["world", "body"])
def test_t8_specialised_body_state_sensors_match_the_old_sensor(frame: str) -> None:
    """T8: 몸 상태 센서의 특수화 함수 = 오라클의 body_state, 항목 조합(하나, 둘, 모두, 순서 섞음, 겹침) × 좌표계, 무작위 상태 1만 개."""
    from evolab.spec import model as m
    from tests.oracle import episode_v31 as old

    spec, _layout = builtin("lander")
    ref = old.Episode(spec, None, Trial(1))
    rng = np.random.default_rng(81)
    combos: list[tuple[str, ...]] = [(f,) for f in STATE_FIELDS] + [STATE_FIELDS, STATE_FIELDS[::-1],
                                                                    ("vx", "vy", "angle", "omega"), ("v_fwd", "v_lat"),
                                                                    ("y", "vy"), ("x", "x", "angle", "angle")]
    combos += [tuple(rng.permutation(STATE_FIELDS)[: int(rng.integers(1, 6))]) for _ in range(20)]
    states = special_states(10000 // len(combos) + 1, seed=82)
    box: list[tuple[float, ...]] = [states[0]]
    ref._bstate = lambda bid: box[0]  # type: ignore[method-assign]
    for fields in combos:
        sensor = m.BodyState(kind="body_state", id="s", body="hull", fields=list(fields), frame=frame)
        _names, _scales, old_fn = ref._sensor_fn(sensor)
        new_fn = episode_mod._body_state_fn(lambda bid: box[0], "hull", tuple(fields), frame)
        for st in states:
            box[0] = st
            got, want = outcome(new_fn), outcome(old_fn)
            assert type(got) is tuple and (got[:1] == ("raise",) or len(got) == len(fields))
            assert repr(got) == repr(want), (fields, frame, st)  # 같은 값, 또는 같은 예외(각도 ±무한대의 cos·sin)


# ---------------------------------------------------------------------------
# T10 사건 한 루프
# ---------------------------------------------------------------------------


def events_world() -> EnvSpec:
    """조건을 몸 수치 값(p.k1..k4, 테스트가 값 사전에 직접 씀)으로 움직이는 사건 다섯: hold 0·1·여러 스텝, 앞 사건의 발생·
    횟수·경과를 읽는 사건, 내장 사건을 읽는 사건, count·since가 필요한 것과 아닌 것(블록·수식이 읽는지)."""
    d = json.loads((ENVS / "lander.evoenv").read_text(encoding="utf-8"))
    dt = d["world"]["dt"]
    d["params"] += [{"id": f"k{i}", "value": 0.0} for i in range(1, 5)]
    d["events"] = [
        {"id": "e0", "when": "p.k1 > 0.5"},
        {"id": "e1", "when": "p.k2 > 0.5 and e0", "hold": dt},
        {"id": "e2", "when": "e1.count >= 2 or p.k3 > 0.5", "hold": 3 * dt},
        {"id": "e3", "when": "e0.since > 0.05 and p.k4 > 0.5"},
        {"id": "e4", "when": "out_of_bounds or timeout.count > 0 or e2", "hold": 5 * dt},
    ]
    d["termination"] = [{"event": "timeout", "outcome": "neutral"}]
    d["scoring"]["blocks"] = [{"kind": "event", "id": "b2", "event": "e2", "value": "count"},
                              {"kind": "event", "id": "b4", "event": "e4"}]
    return compile_env(json.dumps(d, ensure_ascii=False))


def test_t10_event_loop_matches_the_old_fire_order() -> None:
    """T10: 무작위 조건 열(값 몇 스텝씩 유지), 경계 안·밖·경계 위 위치, 스텝 수가 상한을 넘는 구간에서 지금 _update_events와
    오라클의 _fire 순서가 사건 상태(연속 수·발생·횟수·마지막 시각)와 값 사전에서 같다."""
    from tests.oracle import episode_v31 as old

    spec = events_world()
    a, b = Episode(spec, None, Trial(10)), old.Episode(spec, None, Trial(10))
    assert {st.id for st in a._events} == {"e0", "e1", "e2", "e3", "e4"}
    assert a._count_needed == b._count_needed and a._since_needed == b._since_needed
    assert "e0" in a._since_needed and "e1" in a._count_needed and "e2" in a._count_needed and "e0" not in a._count_needed
    assert "e3" not in a._count_needed | a._since_needed
    rng = np.random.default_rng(10)
    x0, y0, x1, y1 = a.bounds
    xs = [x0, x1, 0.5 * (x0 + x1), x0 - 1e-9, x1 + 5.0]
    ys = [y0, y1, 0.5 * (y0 + y1), y1 + 1e-9, y0 - 3.0]
    ks = [0.0] * 4
    pos = (0.0, 0.0)
    fired = 0
    for step in range(1, 3001):
        if rng.random() < 0.25:
            ks = [float(rng.random() < 0.5) for _ in range(4)]
        if rng.random() < 0.05:
            pos = (xs[int(rng.integers(len(xs)))], ys[int(rng.integers(len(ys)))])
        for ep in (a, b):
            ep.steps = step
            ep.time = step * ep.dt
            if step == 2000:
                ep.max_steps = 1990
            for i, k in enumerate(ks):
                ep._values[f"p.k{i + 1}"] = k
            for body in ep.bodies.values():
                if body.body_type == body.DYNAMIC:
                    body.position = pos
            ep._cache.clear()
            ep._pos_cache.clear()
            ep._update_events()
        sa = [(s.id, s.consec, s.occurred, s.count, repr(s.last_time)) for s in [*a._builtin.values(), *a._events]]
        sb = [(s.id, s.consec, s.occurred, s.count, repr(s.last_time)) for s in [*b._builtin.values(), *b._events]]
        assert sa == sb, step
        assert repr(sorted(a._values.items())) == repr(sorted(b._values.items())), step
        fired = sum(s.count for s in a._events)
    assert fired > 100 and a._builtin["out_of_bounds"].count > 3 and a._builtin["timeout"].occurred
    assert all(s.count > 0 for s in a._events)


# ---------------------------------------------------------------------------
# T13 접촉 맞음 기억
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [*NAMES, "parts"])
def test_t13_contact_match_memo_equals_matches(name: str) -> None:
    """T13: 접촉 채널마다 기억한 「모양 → 선택자에 맞음」이 모든 모양에서 옛 _matches(오라클 규칙)와 같다. 챔피언 한 판을
    돌린 뒤의 기억(실제로 닿은 모양)과, 모든 모양을 채널마다 다시 판정한 값을 둘 다 본다."""
    from tests.oracle import episode_v31 as old

    if name == "parts":
        spec = parts_world()
        layout = layout_for(spec)
        genome = layout.initial()
    else:
        spec, layout = builtin(name)
        genome = champion(name)
    body = body_of(layout, genome)
    ep = Episode(spec, body, seeds.validation_trials(1)[0])
    ref = old.Episode(spec, body, seeds.validation_trials(1)[0])
    brain = unpack(layout.brain, layout.brain_params(genome[None, :]))
    obs = np.zeros((1, layout.brain.n_in))
    while not ep.done and ep.steps < 4000:
        obs[0] = ep.observe()
        ep.step(layout.to_actions(brain.forward(obs))[0])
    assert ep._contacts
    shapes_a = list(ep._shape_info)
    shapes_b = list(ref._shape_info)
    assert len(shapes_a) == len(shapes_b)
    remembered = 0
    for (key, ch), (key_b, ch_b) in zip(ep._contacts.items(), ref._contacts.items(), strict=True):
        assert key == key_b
        for s, hit in ch.memo.items():
            assert hit == ep._matches(s, ch.selectors)
            remembered += 1
        for sa, sb in zip(shapes_a, shapes_b, strict=True):
            assert ep._matches(sa, ch.selectors) == ref._matches(sb, ch_b.selectors)
    if name not in ("track_car", "glider"):
        assert remembered > 0, name


# ---------------------------------------------------------------------------
# T12 공기역학
# ---------------------------------------------------------------------------


def aero_cases() -> list[tuple[str, EnvSpec, list[np.ndarray | None]]]:
    from tests.test_build_failure import gene, poly_wing_glider

    out: list[tuple[str, EnvSpec, list[np.ndarray | None]]] = []
    spec, layout = builtin("glider")
    out.append(("glider", spec, [body_of(layout, champion("glider")),
                                 *[body_of(layout, g) for g in random_genomes(layout, 3, seed=12, scale=0.8)]]))
    spec = compile_env(poly_wing_glider(max_time=30.0))
    layout = layout_for(spec)
    out.append(("poly_wing", spec, [body_of(layout, gene(layout, {"flap": f})) for f in (0.0, 0.02, 0.055)]))
    spec = parts_world(lift=False)
    layout = layout_for(spec)
    out.append(("parts", spec, [body_of(layout, layout.initial())]))
    from tests.test_aero import extreme_glider

    for parts in (1, 3):  # 한 스텝 상한이 걸리는 극단 값(공기역학 부품 셋을 단 물체 포함, tests/test_aero.py)
        spec = compile_env(json.dumps(extreme_glider(parts)))
        layout = layout_for(spec)
        out.append((f"extreme{parts}", spec, [body_of(layout, layout.initial())]))
    return out


def test_t12_aero_forces_match_the_oracle_right_before_each_physics_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """T12: 물리 스텝 바로 전(공기힘을 더한 뒤)의 모든 물체 힘·토크·상태와 aero_clamped가 오라클과 1,000스텝 같다."""
    from tests.oracle import episode_v31 as old
    from tests.test_speed_identity import body_states

    real_c = episode_mod._CP_SPACE_STEP
    rec_a: list[str] = []
    rec_b: list[str] = []
    current: list[Any] = [None]
    monkeypatch.setattr(episode_mod, "_CP_SPACE_STEP",
                        lambda raw, dt: (rec_a.append(body_states(current[0])), real_c(raw, dt)))
    clamped = edges = 0
    for label, spec, bodies in aero_cases():
        rng = np.random.default_rng(len(label))
        for k, body in enumerate(bodies):
            trial = seeds.validation_trials(4)[k % 4]
            a, b = Episode(spec, body, trial), old.Episode(spec, body, trial)
            current[0] = a
            for ep, rec in ((a, rec_a), (b, rec_b)):
                real = ep.space.step
                monkeypatch.setattr(ep.space, "step", lambda dt, ep=ep, rec=rec, real=real: (rec.append(body_states(ep)),
                                                                                             real(dt)))
            assert a._aero and a.aero_edge_count == b.aero_edge_count > 0
            edges += a.aero_edge_count
            n_out = len(a.outputs)
            while not b.done and b.steps < 1000:
                act = [float(x) for x in rng.uniform(-1.0, 1.0, n_out)]
                a.step(act)
                b.step(act)
                assert len(rec_a) == len(rec_b) == 1, (label, k, b.steps)
                assert rec_a.pop() == rec_b.pop(), (label, k, b.steps)
                assert a.aero_clamped == b.aero_clamped
                same_now(a, b, deep=b.steps % 50 == 0)
            assert b.steps > 50, (label, k)
            clamped += b.aero_clamped
    assert edges > 20 and clamped > 100


# ---------------------------------------------------------------------------
# T11 레이 필터 한 번
# ---------------------------------------------------------------------------


def shape_filters(ep: Any) -> list[tuple[Any, ...]]:
    return [(bid, sorted(tags), s.filter.group, s.filter.categories, s.filter.mask) for s, (bid, tags) in
            ep._shape_info.items()]


def many_rays_world(n_targets: int) -> str:
    """레이 꼬리표 묶음 n_targets개(묶음마다 다른 꼬리표, 같은 묶음 하나는 두 번)를 쓰는 점프 게임."""
    d = json.loads((ENVS / "jump.evoenv").read_text(encoding="utf-8"))
    rays = next(s for s in d["sensors"] if s["kind"] == "rays")
    extra = []
    for k in range(n_targets):
        extra.append(dict(rays, id=f"rr{k}", hits=[f"tag.t{k}"], count=1))
    extra.append(dict(rays, id="rr_again", hits=["tag.t0"], count=1))
    d["sensors"] = [s for s in d["sensors"] if s["kind"] != "rays"] + extra
    for b in d["bodies"]:  # 땅에 짝수·3의 배수 꼬리표, 달리는 몸에 나머지(모든 꼬리표가 어딘가에 있어야 검증을 지난다)
        on_ground = [f"t{k}" for k in range(n_targets) if k % 2 == 0 or k % 3 == 0]
        mine = on_ground if b["id"] == "ground" else [f"t{k}" for k in range(n_targets) if f"t{k}" not in on_ground]
        b["tags"] = [*b.get("tags", []), *mine]
    return json.dumps(d, ensure_ascii=False)


@pytest.mark.parametrize("name", [*NAMES, "parts"])
def test_t11_ray_filters_match_the_oracle(name: str) -> None:
    """T11: 모든 모양의 (그룹, 범주, 마스크)와 모양 수가 오라클 Episode와 같다(없는 물체에 단 레이 센서가 있는 몸 포함)."""
    from tests.oracle import episode_v31 as old

    if name == "parts":
        spec = parts_world()
        layout = layout_for(spec)
        genomes = layout.initial()[None, :] + np.random.default_rng(11).uniform(-1.5, 1.5, (8, layout.dim))
    else:
        spec, layout = builtin(name)
        genomes = np.vstack([champion(name)[None, :], random_genomes(layout, 3, seed=11, scale=1.2)])
    absent = set()
    for g in genomes:
        body = body_of(layout, g)
        a, b = Episode(spec, body, Trial(11)), old.Episode(spec, body, Trial(11))
        absent.add(a.absent)
        assert len(a.space.shapes) == len(b.space.shapes) == len(a._shape_info)
        assert shape_filters(a) == shape_filters(b)
        assert getattr(a, "_ray_bits", None) == getattr(b, "_ray_bits", None)
        # 만든 뒤 처음 보는 묶음을 부르면 옛 길처럼 다음 비트를 붙인다
        assert a._ray_filter(["tag.nothing_here"]) == b._ray_filter(["tag.nothing_here"])
        assert shape_filters(a) == shape_filters(b) and a._ray_bits == b._ray_bits
    if name == "parts":
        assert len(absent) >= 3


@pytest.mark.parametrize("n", [29, 30, 31])
def test_t11_too_many_ray_targets_is_the_same_error(n: int) -> None:
    """T11: 레이 꼬리표 묶음 30개까지는 만들어지고(필터가 오라클과 같음), 31개면 두 런타임 모두 RT_TOO_MANY_RAY_TARGETS."""
    from evolab.app import strings as S
    from tests.oracle import episode_v31 as old

    spec = compile_env(many_rays_world(n))
    try:
        a = Episode(spec, None, Trial(12))
    except episode_mod.EpisodeBuildError as exc:
        a = ("raise", str(exc))
    try:
        b = old.Episode(spec, None, Trial(12))
    except old.EpisodeBuildError as exc:
        b = ("raise", str(exc))
    if n <= 30:
        assert not isinstance(a, tuple) and not isinstance(b, tuple)
        assert shape_filters(a) == shape_filters(b) and len(a._ray_bits) == n
    else:
        assert a == b == ("raise", S.RT_TOO_MANY_RAY_TARGETS)

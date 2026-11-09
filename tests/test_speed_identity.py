"""v6.0 오라클 바이트 비교, Episode 부분 (v6 설계 8.1 T1·T2·T9, 3.2·3.4, R 묶음).

지금 런타임(world/physics2d/episode.py)을 v3.1 출시판의 얼린 사본(tests/oracle/episode_v31.py)과 같은 프로세스에서 나란히
돌려 바이트로 비교한다. 정답 해시를 박지 않는다(명세 「같은 머신에서 두 번 돌려 비교」).

- T1(Episode): 내장 환경 6개 × (무작위 유전자 6 × 학습 시험, 픽스처 챔피언 × 검증 시험 6). 두 Episode를 같은 행동으로 한 스텝씩
  돌려 관측·실제 출력·끝남을 스텝마다, 「깊이」 쌍은 모든 물체의 위치·속도·각도·각속도, 궤적 자세, 센서 원래 값, 레이,
  블록 값, 값 사전까지 스텝마다 비교한다. 행동은 공개 step(ndarray 행)과 새 내부 함수 step_values(tolist() 행)로 둘 다 넣는다.
  같은 비교를 얼린 평가기(evaluate_v31)의 Episode만 바꿔 끼워 궤적 기록(3쌍마다 1)·관찰 코드 0·1·2(+ 격자 틀) 길로도 한다
  (평가기 쪽 비교 evaluate 대 evaluate_v31은 E 묶음의 tests/test_eval_window.py·T1 블록).
- T2(Episode): 몸 수치 때문에 조립되지 않는 쌍(꼬인 착륙선, 다각형 날개 글라이더 — tests/test_build_failure.py의 시험 환경)이
  섞인 묶음과 처음 몸 실패(오류)가 오라클과 같은 오류 글·같은 결과.
- T9: 재생 길 — engine/replay.replay(추적 없음·"brain"·True), app/play/record.replay(사람 기록), 직접 조종 스텝 열(키 조합
  목록, NaN·무한대·범위 밖 값 포함)이 오라클 Episode로 돌린 같은 재생과 궤적·점수·추적 바이트가 같다.

오라클의 EvalResult·EpisodeResult·EpisodeBuildError는 지금 코드와 다른 클래스라 칸마다 tobytes()/repr로만 비교한다.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tests.oracle.evaluate_v31 as ev31
from evolab import paths
from evolab.engine import observe as observing
from evolab.engine import seeds
from evolab.policy.brain import unpack
from evolab.policy.genome import GenomeLayout, layout_for
from evolab.spec.compiled import EnvSpec
from evolab.spec.io import compile_env, env_hash
from evolab.spec.io import load_env as _load_env
from evolab.world.physics2d import episode as cur
from tests.oracle import episode_v31 as old

ROOT = paths.builtin_env_dir()
NAMES = ("lander", "track_car", "hill_car", "jump", "glider", "wheels")


@functools.cache
def load(name: str) -> tuple[EnvSpec, GenomeLayout]:
    spec = _load_env(ROOT / f"{name}.evoenv")
    return spec, layout_for(spec)


def champion(name: str) -> np.ndarray:
    return np.load(Path(__file__).resolve().parent / "data" / "v6_champions" / f"{name}.npz")["genome"]


def random_genomes(layout: GenomeLayout, n: int, seed: int, scale: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return layout.initial()[None, :] + scale * rng.standard_normal((n, layout.dim))


def body_of(layout: GenomeLayout, genome: np.ndarray) -> np.ndarray | None:
    return layout.body_values(genome[None, :])[0] if layout.n_body else None


# ---------------------------------------------------------------------------
# 두 Episode를 나란히
# ---------------------------------------------------------------------------


def body_states(ep: Any) -> str:
    """모든 물체의 공개 API 상태(위치, 속도, 각도, 각속도, 힘, 토크)를 repr로(−0.0·NaN 비트까지 가른다)."""
    out = []
    for bid, b in ep.bodies.items():
        p, v, f = b.position, b.velocity, b.force
        out.append((bid, p.x, p.y, v.x, v.y, b.angle, b.angular_velocity, f.x, f.y, b.torque))
    return repr(out)


def same_static(a: Any, b: Any) -> None:
    """만든 직후 바뀌지 않는 것: 그림, 이름 목록, 만들기 값, 모양 필터."""
    assert repr(a.geometry()) == repr(b.geometry())
    for name in ("moving_bodies", "inputs", "outputs", "output_channel_index", "absent", "absent_joints",
                 "dead_channels", "anchor_body", "aero_edge_count", "max_steps", "dt", "bounds", "gravity", "wind",
                 "randoms", "params", "gen_values"):
        assert repr(getattr(a, name)) == repr(getattr(b, name)), name
    assert a.actuator_channels() == b.actuator_channels()
    assert list(a.bodies) == list(b.bodies)
    fa = [(bid, sorted(tags), s.filter.group, s.filter.categories, s.filter.mask, s.friction, s.elasticity)
          for s, (bid, tags) in a._shape_info.items()]
    fb = [(bid, sorted(tags), s.filter.group, s.filter.categories, s.filter.mask, s.friction, s.elasticity)
          for s, (bid, tags) in b._shape_info.items()]
    assert fa == fb
    assert len(a.space.shapes) == len(b.space.shapes) and len(a.space.constraints) == len(b.space.constraints)
    assert body_states(a) == body_states(b)


def same_now(a: Any, b: Any, deep: bool) -> None:
    assert (a.done, a.steps, repr(a.time)) == (b.done, b.steps, repr(b.time))
    assert repr(a.applied_outputs) == repr(b.applied_outputs)
    if not deep:
        return
    sa, sb = a.snapshot(), b.snapshot()
    assert sa.poses.tobytes() == sb.poses.tobytes() and sa.outputs.tobytes() == sb.outputs.tobytes()
    assert repr(sa.time) == repr(sb.time)
    assert body_states(a) == body_states(b)
    assert repr(a.channel_values()) == repr(b.channel_values())
    assert repr(a.ray_view()) == repr(b.ray_view())
    assert repr(a.block_values()) == repr(b.block_values())
    assert repr(sorted(a._values.items())) == repr(sorted(b._values.items()))
    assert a.aero_clamped == b.aero_clamped
    for ch in a.actuator_channels():
        assert a.jump_ready(ch) == b.jump_ready(ch)


def run_pair(spec: EnvSpec, layout: GenomeLayout, genome: np.ndarray, trial: cur.Trial, *, deep: bool,
             values: bool) -> int:
    """유전자 하나 × 시험 하나를 두 Episode로 끝까지. 신경망 행동(오라클 쪽 관측으로 계산)을 둘 다에 넣는다.
    values면 지금 쪽은 step_values(tolist() 행), 아니면 공개 step(ndarray 행). 돌린 스텝 수를 돌려준다."""
    body = body_of(layout, genome)
    a = cur.Episode(spec, body, trial)
    b = old.Episode(spec, body, trial)
    same_static(a, b)
    same_now(a, b, deep)
    brain = unpack(layout.brain, layout.brain_params(genome[None, :]))
    obs = np.zeros((1, layout.brain.n_in), dtype=np.float64)
    while not b.done:
        oa, ob = a.observe(), b.observe()
        assert oa.dtype == ob.dtype and oa.tobytes() == ob.tobytes(), b.steps
        obs[0] = ob
        act = layout.to_actions(brain.forward(obs))
        if values:
            a.step_values(act.tolist()[0])
        else:
            a.step(act[0])
        b.step(act[0])
        same_now(a, b, deep)
    assert a.done
    a.step(np.zeros(len(a.outputs)))  # 끝난 뒤의 step은 아무 일도 안 한다
    assert a.steps == b.steps
    assert repr(a.result()) == repr(b.result())
    assert repr(a.observe().tobytes()) == repr(b.observe().tobytes())
    return int(b.steps)


@pytest.mark.parametrize("name", NAMES)
def test_t1_random_genomes_match_the_oracle_step_by_step(name: str) -> None:
    """T1: 무작위 유전자 6 × 학습 시험(세대 0·3) — 셋은 깊이 비교, 반은 step_values 길."""
    spec, layout = load(name)
    genomes = random_genomes(layout, 6, seed=61)
    trials = seeds.train_trials(1, 0, 1) + seeds.train_trials(1, 3, 1)
    total = 0
    for i, g in enumerate(genomes):
        total += run_pair(spec, layout, g, trials[i % 2], deep=i < 3, values=i % 2 == 1)
    assert total > 50


@pytest.mark.parametrize("name", NAMES)
def test_t1_champion_matches_the_oracle_on_validation_trials(name: str) -> None:
    """T1: 픽스처 챔피언 × 검증 시험 6(긴 에피소드) — 시험 0은 깊이 비교, 홀수 시험은 step_values 길."""
    spec, layout = load(name)
    g = champion(name)
    total = 0
    for k, t in enumerate(seeds.validation_trials(6)):
        total += run_pair(spec, layout, g, t, deep=k == 0, values=k % 2 == 1)
    assert total > 600


def test_t1_wheels_bodies_with_absent_parts_match_the_oracle() -> None:
    """T1: 바퀴 고르는 언덕차의 여러 몸 구조(없는 바퀴·부품, 죽은 채널)."""
    spec, layout = load("wheels")
    genomes = random_genomes(layout, 12, seed=62, scale=1.2)
    absent = set()
    for i, g in enumerate(genomes):
        body = body_of(layout, g)
        absent.add(cur.Episode(spec, body, seeds.display_trial()).absent)
        run_pair(spec, layout, g, seeds.train_trials(2, i, 1)[0], deep=i < 4, values=i % 2 == 0)
    assert len(absent) >= 4


def test_t1_parts_world_with_every_kind_of_part_matches_the_oracle() -> None:
    """T1: 모든 부품(바퀴·모터, 추진기·연료, 점프, 공기역학, 차량, 바람 항력, 몸 상태·목표·접촉·레이 센서)을 없어질 수 있는
    물체에 단 시험 세계(tests/test_structure_runtime.PARTS)."""
    import json

    from tests.test_structure_runtime import PARTS

    spec = compile_env(json.dumps(PARTS))
    layout = layout_for(spec)
    rng = np.random.default_rng(63)
    genomes = layout.initial()[None, :] + rng.uniform(-1.5, 1.5, (8, layout.dim))
    for i, g in enumerate(genomes):
        run_pair(spec, layout, g, cur.Trial(seeds.derive(63, i)), deep=True, values=i % 2 == 0)


# ---------------------------------------------------------------------------
# 얼린 평가기에 Episode만 바꿔 끼움(궤적 기록·관찰·조립 실패 길)
# ---------------------------------------------------------------------------


def with_current_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev31, "Episode", cur.Episode)
    monkeypatch.setattr(ev31, "EpisodeBuildError", cur.EpisodeBuildError)
    monkeypatch.setattr(ev31, "EpisodeResult", cur.EpisodeResult)


def same_eval(a: Any, b: Any) -> None:
    """EvalResult 두 개가 칸마다 바이트로 같다(시간 칸 빼고)."""
    for f in ("scores", "steps", "blocks", "action_abs", "weighted"):
        x, y = getattr(a, f), getattr(b, f)
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), f
    assert a.end_events == b.end_events and a.outcomes == b.outcomes
    assert (a.trajectories is None) == (b.trajectories is None)
    for u, v in zip(a.trajectories or [], b.trajectories or [], strict=True):
        assert (u is None) == (v is None)
        if u is not None:
            assert u.poses.shape == v.poses.shape and u.poses.tobytes() == v.poses.tobytes()
            assert u.outputs.shape == v.outputs.shape and u.outputs.tobytes() == v.outputs.tobytes()
            assert (u.every, repr(u.dt), u.end_step) == (v.every, repr(v.dt), v.end_step)
    assert (a.observations is None) == (b.observations is None)
    for u, v in zip(a.observations or [], b.observations or [], strict=True):
        assert (u is None) == (v is None)
        if u is not None:
            for f in ("hist", "out_sum", "cells", "counts"):
                assert getattr(u, f).tobytes() == getattr(v, f).tobytes(), f
            assert (u.steps, u.outside) == (v.steps, v.outside)


def eval_both(monkeypatch: pytest.MonkeyPatch, spec: EnvSpec, layout: GenomeLayout, genomes: np.ndarray,
              pairs: list[tuple[int, cur.Trial]], *, frame: Any = None) -> tuple[Any, Any]:
    n = len(pairs)
    record = [i % 3 == 0 for i in range(n)]
    observe = [i % 3 for i in range(n)]
    kw = dict(record=record, every=2, observe=observe, frame=frame)
    ref = ev31.evaluate(spec, layout, genomes, pairs, **kw)
    with monkeypatch.context() as mp:
        with_current_episode(mp)
        now = ev31.evaluate(spec, layout, genomes, pairs, **kw)
    return now, ref


@pytest.mark.parametrize("name", NAMES)
def test_t1_frozen_evaluator_with_the_current_episode_matches_the_oracle(name: str,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """T1: 얼린 평가기 + 지금 Episode 대 얼린 평가기 + 오라클 Episode — 무작위 유전자 6 × 학습 시험 2와 챔피언 × 검증
    시험 6, 궤적 기록 3쌍마다 1, 관찰 코드 0·1·2 섞음 + 그 세대 격자 틀."""
    spec, layout = load(name)
    trials = seeds.train_trials(5, 2, 2)
    frame, _ = observing.world_frame(spec, env_hash(spec.file)[:16], trials[0])
    fr = tuple(frame.as_array()) if frame is not None else None
    genomes = np.vstack([random_genomes(layout, 6, seed=64), champion(name)[None, :]])
    pairs = [(g, t) for g in range(6) for t in trials] + [(6, t) for t in seeds.validation_trials(6)]
    now, ref = eval_both(monkeypatch, spec, layout, genomes, pairs, frame=fr)
    same_eval(now, ref)
    assert int(ref.steps.sum()) > 600


# ---------------------------------------------------------------------------
# T2 조립 실패
# ---------------------------------------------------------------------------


def failure_cases() -> list[tuple[str, EnvSpec, GenomeLayout, np.ndarray]]:
    from tests.test_build_failure import gene, poly_wing_glider, twist_lander

    out = []
    spec = compile_env(twist_lander())
    layout = layout_for(spec)
    rng = np.random.default_rng(54)
    X = np.vstack([gene(layout, {"twist": t}) for t in (0.0, -2.0, 0.5, -1.5)])
    X[:, layout.n_body:] = 0.3 * rng.standard_normal((4, layout.dim - layout.n_body))
    out.append(("twist_lander", spec, layout, X))
    spec = compile_env(poly_wing_glider())
    layout = layout_for(spec)
    X = np.vstack([gene(layout, {"flap": f}) for f in (0.02, -0.03, 0.05, -0.01)])
    out.append(("poly_wing_glider", spec, layout, X))
    return out


def build_outcome(cls: Any, spec: EnvSpec, body: Any, trial: Any) -> tuple[str, str] | Any:
    try:
        return cls(spec, body, trial)
    except Exception as exc:  # noqa: BLE001 - 두 런타임의 예외 클래스가 다르다(이름과 글로 비교)
        return (type(exc).__name__, str(exc))


def test_t2_build_failures_raise_the_same_error_and_the_rest_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """T2: 조립되지 않는 몸은 두 런타임이 같은 오류(이름·글)를 내고, 조립되는 몸은 스텝마다 같다. 실패가 섞인 묶음을
    얼린 평가기로 돌린 결과(실패 쌍 NaN·failure·궤적 0프레임·관찰 0)도 같다."""
    trials = [cur.Trial(seeds.derive(9, k)) for k in range(2)]
    for label, spec, layout, X in failure_cases():
        failed = 0
        for g in X:
            body = body_of(layout, g)
            for t in trials:
                a, b = build_outcome(cur.Episode, spec, body, t), build_outcome(old.Episode, spec, body, t)
                if isinstance(b, tuple):
                    assert a == b and b[0] == "EpisodeBuildError", label
                    failed += 1
                else:
                    assert not isinstance(a, tuple), (label, a)
                    run_pair(spec, layout, g, t, deep=True, values=failed % 2 == 0)
        assert failed == 4, label
        pairs = [(i, t) for i in range(len(X)) for t in trials]
        now, ref = eval_both(monkeypatch, spec, layout, X, pairs)
        same_eval(now, ref)
        assert np.isnan(ref.scores).sum() == 4


def test_t2_first_body_failure_is_the_same_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """T2: 처음 몸으로도 조립되지 않는 시험 환경은 두 런타임 모두 같은 글의 EpisodeBuildError(평가기도 오류를 올림)."""
    from tests.test_build_failure import twist_lander

    spec = compile_env(twist_lander(shift=-3.0))
    layout = layout_for(spec)
    trial = cur.Trial(seeds.derive(9, 0))
    a, b = build_outcome(cur.Episode, spec, None, trial), build_outcome(old.Episode, spec, None, trial)
    assert isinstance(b, tuple) and a == b
    pairs = [(0, trial)]
    genomes = layout.initial()[None, :]
    with pytest.raises(old.EpisodeBuildError) as ref:
        ev31.evaluate(spec, layout, genomes, pairs)
    with monkeypatch.context() as mp:
        with_current_episode(mp)
        with pytest.raises(cur.EpisodeBuildError) as now:
            ev31.evaluate(spec, layout, genomes, pairs)
    assert str(now.value) == str(ref.value)


# ---------------------------------------------------------------------------
# T9 재생 길
# ---------------------------------------------------------------------------


def same_replay(a: Any, b: Any) -> None:
    ta, tb = a.trajectory, b.trajectory
    assert ta.poses.tobytes() == tb.poses.tobytes() and ta.outputs.tobytes() == tb.outputs.tobytes()
    assert (ta.every, repr(ta.dt), ta.end_step) == (tb.every, repr(tb.dt), tb.end_step)
    assert repr(a.result) == repr(b.result)
    assert repr(a.geometry) == repr(b.geometry) and repr(a.agent) == repr(b.agent)
    assert a.channels == b.channels and a.inputs == b.inputs
    for f in ("observations", "rays"):
        x, y = getattr(a, f), getattr(b, f)
        assert (x is None) == (y is None), f
        if x is not None:
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), f
    assert (a.activations is None) == (b.activations is None)
    for x, y in zip(a.activations or [], b.activations or [], strict=True):
        assert x.shape == y.shape and x.tobytes() == y.tobytes()
    assert repr(a.raw) == repr(b.raw)


@pytest.mark.parametrize("name", NAMES)
def test_t9_engine_replay_matches_the_oracle_episode(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """T9: engine/replay.replay(추적 없음·"brain"·True)를 지금 Episode와 오라클 Episode로 — 챔피언 × 전시 시험, 무작위
    유전자 × 학습 시험."""
    from evolab.engine import replay as replay_mod

    spec, layout = load(name)
    cases = [(champion(name), seeds.display_trial()), (random_genomes(layout, 1, seed=91)[0], seeds.train_trials(9, 1, 1)[0])]
    for k, (g, t) in enumerate(cases):
        for trace in (False, "brain", True):
            every = 1 + (k + len(str(trace))) % 3
            now = replay_mod.replay(spec, layout, g, t, every=every, trace=trace)
            with monkeypatch.context() as mp:
                mp.setattr(replay_mod, "Episode", old.Episode)
                ref = replay_mod.replay(spec, layout, g, t, every=every, trace=trace)
            same_replay(now, ref)
            assert now.trajectory.frames > 2


def key_actions(n_out: int, steps: int, seed: int) -> list[list[float]]:
    """직접 조종처럼 키 조합으로 만든 행동 목록(-1, 0, 1, 그 합). 가끔 범위 밖·NaN·무한대 값(기록 파일을 손으로 고친 경우)."""
    rng = np.random.default_rng(seed)
    choices = [-1.0, 0.0, 0.0, 1.0, 1.0, 0.5, -0.5, 2.0, -2.0, 1, 0]
    out: list[list[float]] = []
    held = [0.0] * n_out
    for k in range(steps):
        if k % 7 == 0:
            held = [choices[int(rng.integers(len(choices)))] for _ in range(n_out)]
        row = list(held)
        if k % 97 == 13 and n_out:
            row[int(rng.integers(n_out))] = [math.nan, math.inf, -math.inf, 1e300][k % 4]
        out.append(row)
    return out


@pytest.mark.parametrize("name", ["lander", "track_car", "hill_car", "jump", "wheels"])
def test_t9_direct_play_steps_match_the_oracle(name: str) -> None:
    """T9: 직접 조종 창처럼 키 조합 목록(list)을 step()에 넣은 스텝 열이 오라클과 스텝마다 같다(점프 준비 표시 포함)."""
    spec, _layout = load(name)
    trial = cur.Trial(seeds.derive(77, len(name)))
    a, b = cur.Episode(spec, None, trial), old.Episode(spec, None, trial)
    same_static(a, b)
    for row in key_actions(len(spec.outputs), 900, seed=len(name)):
        if b.done:
            break
        a.step(list(row))
        b.step(list(row))
        same_now(a, b, deep=a.steps % 5 == 0)
    assert repr(a.result()) == repr(b.result())


@pytest.mark.parametrize("name", ["lander", "jump", "track_car"])
def test_t9_human_record_replay_matches_the_oracle(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """T9: app/play/record.replay(「내 최고 기록」 확인)가 지금 Episode와 오라클 Episode로 같은 결과."""
    from evolab.app.play import record as record_mod

    spec, _layout = load(name)
    rec = record_mod.new_record(spec, seeds.derive(78, len(name)))
    rec.actions = key_actions(len(spec.outputs), 400, seed=7 + len(name))
    now = record_mod.replay(spec, rec)
    with monkeypatch.context() as mp:
        mp.setattr(record_mod, "Episode", old.Episode)
        ref = record_mod.replay(spec, rec)
    assert repr(now) == repr(ref) and now.steps > 0

"""v1.2 관찰 기록: 엔진·저장 (설계 2·3·4.0절, D3~D13, 테스트 T1~T11, T15~T22).

관찰 기록(세대 대표 파일의 obs_ 키)은 학습이 읽지 않는 파생 기록이다. 여기서는
- 워커 관찰기와 관찰을 켠 학습이 학습 결과(통계, 챔피언, 세대 대표 유전자, 체크포인트)를 비트 단위로 바꾸지 않는지,
- 관찰 기록 자체가 워커 수, 관전, 이어하기, 되감기, 분기, 재현 묶음과 무관하게 같은지,
- 요약이 실패해도 학습이 이어지는지, 관찰하지 않는 가짜 평가기와 이전 실행이 그대로 도는지
를 본다. 읽는 쪽 형식 검사는 test_observe_format.py(T13·T14), v1.1 실제 실행은 test_v11_fixture.py(T12).
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import pickle
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab import cli, paths
from evolab.app import strings as S
from evolab.engine import evaluate as evaluate_mod
from evolab.engine import observe as observing
from evolab.engine import pool as pool_mod
from evolab.engine import trainer as trainer_mod
from evolab.engine.evaluate import ACTION_BINS, POSITION_EVERY, EvalResult, PairObservation, evaluate
from evolab.engine.pool import PoolError, WorkerPool
from evolab.engine.runsetup import create_run
from evolab.engine.session import PAUSED, Session, SessionRequest
from evolab.engine.trainer import TrainConfig, Trainer, training_trials
from evolab.policy.brain import unpack
from evolab.policy.genome import layout_for
from evolab.runs import observe as fmt
from evolab.runs.bundle import compare_stats, export_bundle, reproduce
from evolab.runs.catalog import branch_run
from evolab.runs.observe import GridFrame, consistent, decode
from evolab.runs.store import CheckpointPolicy, RunStore
from evolab.spec.io import compile_env, env_hash
from evolab.world.physics2d.episode import Episode, Trial

ENVS = paths.builtin_env_dir()
LANDER = ENVS / "lander.evoenv"
LANDER_TEXT = LANDER.read_text(encoding="utf-8")
LANDER_SPEC = compile_env(LANDER_TEXT)
LAYOUT = layout_for(LANDER_SPEC)
TIME = set(trainer_mod.TIME_COLUMNS)
KEEP_ALL = CheckpointPolicy(keep_last=10**6)
V10_RUN = Path(__file__).parent / "data" / "v10_run_ga" / "run"
BASE_KEYS = ["env_version", "genome", "score"]
ACTION_KEYS = {fmt.ACT_HIST, fmt.ACT_MEAN}
DENSITY_KEYS = {fmt.FRAME, fmt.WORLD_KEY, fmt.DENS_CELLS, fmt.DENS_COUNTS}

OBS_CONFIG: dict[str, Any] = {
    "method": "cmaes", "run_seed": 4, "popsize": 6, "trials": 2, "validate_every": 24, "validation_seeds": 4,
    "validation_candidates": 1, "checkpoint_every": 1, "restart_patience": 24, "restart_from": "alternate",
    "qualify_entrants": 2, "qualify_trials": 24, "max_generations": 7,
}
"""작은 착륙선 실행(세대당 학습 시험 12회). 예선 다음 출전(2세대), 예선 끝(4세대), 자동 재시작(6세대)이 7세대 안에
일어난다(기준 실행에서 assert). 검증 세트는 4개로 줄였다."""
GENS = OBS_CONFIG["max_generations"]


def text_of(name: str) -> str:
    return (ENVS / f"{name}.evoenv").read_text(encoding="utf-8")


def config(**over: Any) -> TrainConfig:
    return TrainConfig(**{**OBS_CONFIG, **over})


def strip(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in r.items() if k not in TIME} for r in rows]


def make_store(base: Path, name: str, text: str = LANDER_TEXT, policy: CheckpointPolicy | None = KEEP_ALL) -> RunStore:
    store = RunStore.create(base, name, {"env_name": "착륙선"}, policy=policy)
    store.add_env_version(text, 0)
    return store


def obs_of(store: RunStore, g: int) -> dict[str, np.ndarray] | None:
    return store.load_observation(g)


def assert_same_arrays(a: dict[str, np.ndarray] | None, b: dict[str, np.ndarray] | None, what: str = "") -> None:
    assert (a is None) == (b is None), what
    if a is None or b is None:
        return
    assert sorted(a) == sorted(b), what
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), f"{what} {k}"


def decoded(arrays: dict[str, np.ndarray] | None, g: int, layout: Any = LAYOUT) -> fmt.Observation:
    assert arrays is not None, g
    obs, reason = decode(arrays, dim=layout.dim, n_body=layout.n_body, n_out=layout.brain.n_out, generation=g)
    assert reason == "" and obs is not None, (g, reason)
    return obs


def same_results(a: EvalResult, b: EvalResult) -> None:
    """EvalResult의 학습 결과 필드가 모두 비트 단위로 같다(시간 필드, 관찰은 뺌)."""
    assert a.scores.tobytes() == b.scores.tobytes()
    assert a.steps.tobytes() == b.steps.tobytes()
    assert a.end_events == b.end_events and a.outcomes == b.outcomes
    assert a.blocks.tobytes() == b.blocks.tobytes()
    assert a.action_abs.tobytes() == b.action_abs.tobytes()
    assert a.weighted is not None and b.weighted is not None and a.weighted.tobytes() == b.weighted.tobytes()
    assert (a.trajectories is None) == (b.trajectories is None)
    for x, y in zip(a.trajectories or [], b.trajectories or [], strict=True):
        assert (x is None) == (y is None)
        if x is not None and y is not None:
            assert x.poses.tobytes() == y.poses.tobytes() and x.outputs.tobytes() == y.outputs.tobytes()
            assert (x.every, x.dt, x.end_step) == (y.every, y.dt, y.end_step)


def same_pair_observations(a: list[PairObservation | None] | None, b: list[PairObservation | None] | None) -> None:
    assert a is not None and b is not None and len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert (x is None) == (y is None)
        if x is None or y is None:
            continue
        assert x.steps == y.steps and x.outside == y.outside
        for name in ("hist", "out_sum", "cells", "counts"):
            u, v = getattr(x, name), getattr(y, name)
            assert u.dtype == v.dtype and u.tobytes() == v.tobytes(), name


@pytest.fixture(scope="module")
def pool2() -> Any:
    with WorkerPool(2) as pool:
        yield pool


# ---------------------------------------------------------------------------
# 기준 실행(끊지 않음, 워커 0, 관찰 켬): 모듈에서 한 번
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Straight:
    store: RunStore
    rows: list[dict[str, Any]]
    messages: dict[int, dict[str, np.ndarray] | None]
    """세대마다 trainer.last_observation(세션이 화면에 보내는 것)."""
    trainer: Trainer


@pytest.fixture(scope="module")
def straight(tmp_path_factory: pytest.TempPathFactory) -> Straight:
    # create_run: run.json에 설정이 있어야 재현 묶음이 같은 설정으로 다시 돈다(T10)
    created = create_run(tmp_path_factory.mktemp("straight"), LANDER_TEXT, LANDER_SPEC, config(), 0)
    store = RunStore.open(created.root, policy=KEEP_ALL)
    messages: dict[int, dict[str, np.ndarray] | None] = {}
    with WorkerPool(0) as pool:
        t = Trainer(LANDER_TEXT, config(), pool, store)
        assert t.observe  # 저장소가 있으면 기본으로 켬
        rows: list[dict[str, Any]] = []
        while t.check_stop() is None:
            rows.append(t.step())
            messages[t.state.generation] = t.last_observation
        t.finish()
    switches = [(int(r["generation"]), int(r["switch"])) for r in rows if int(r["switch"])]
    assert switches == [(2, trainer_mod.SWITCH_QUALIFY_NEXT), (4, trainer_mod.SWITCH_QUALIFY_PICK),
                        (6, trainer_mod.SWITCH_RESTART_FRESH)]
    assert t.observe_problem is None
    return Straight(store, rows, messages, t)


def all_obs(store: RunStore, gens: int = GENS) -> list[dict[str, np.ndarray] | None]:
    return [obs_of(store, g) for g in range(1, gens + 1)]


def test_straight_run_records_every_generation(straight: Straight) -> None:
    """기준 실행 자체: 세대마다 형식 1 관찰이 있고, 메시지 = 디스크, 통계 줄과 맞는다(D10)."""
    rows = straight.store.read_stats()
    for g in range(1, GENS + 1):
        arrays = obs_of(straight.store, g)
        assert_same_arrays(arrays, straight.messages[g], f"message {g}")
        obs = decoded(arrays, g)
        base = straight.store.load_champion(g, "representative")
        assert consistent(obs, rows[g - 1], base["score"])
        assert obs.has_actions and obs.has_density and obs.genomes is not None and obs.population == 6
        assert obs.trials == 2 and obs.genome_stride == 1
        assert obs.genomes[obs.representative].tobytes() == np.asarray(base["genome"]).astype(np.float16).tobytes()
        assert obs.frame is not None and obs.world_key is not None and len(obs.world_key) == observing.KEY_LENGTH
        assert int(obs.act_hist[0].sum(axis=1)[0]) == int(rows[g - 1]["steps"])  # 모든 쌍 · 모든 스텝
        assert obs.trial_seed == training_trials(LANDER_SPEC, 4, g - 1, 2)[0].seed and obs.track is None


# ---------------------------------------------------------------------------
# T1~T3: 워커 관찰기
# ---------------------------------------------------------------------------


def _batch(name: str) -> tuple[Any, Any, str, np.ndarray, list[tuple[int, Trial]], GridFrame, list[int]]:
    text = text_of(name)
    spec = compile_env(text)
    layout = layout_for(spec)
    rng = np.random.default_rng(11)
    T = 2
    genomes = [layout.initial(), layout.initial() + 0.05 * rng.standard_normal(layout.dim),
               layout.initial() + 0.6 * rng.standard_normal(layout.dim)]
    if name == "lander":  # 학습된 개체(v1.0 자료의 챔피언): 연료가 떨어져 실제 출력이 0이 되는 긴 에피소드
        with np.load(V10_RUN / "champions" / "gen_000004_champion.npz") as z:
            genomes.append(np.asarray(z["genome"], dtype=np.float64))
    X = np.vstack(genomes)
    trials = training_trials(spec, 3, 1, spec.track_set_size or T)[:T]
    pairs = [(i, t) for i in range(X.shape[0]) for t in trials]
    frame, _ = observing.world_frame(spec, env_hash(spec.file)[:16], trials[0])
    assert frame is not None
    codes = [evaluate_mod.OBSERVE_POSITIONS if k == 0 else evaluate_mod.OBSERVE_ACTIONS
             for _ in range(X.shape[0]) for k in range(T)]
    return spec, layout, text, X, pairs, frame, codes


@pytest.mark.parametrize("name", ["lander", "track_car", "hill_car"])
def test_observer_does_not_change_results(name: str, pool2: WorkerPool) -> None:
    """T1: 관찰 켬·끔 × 워커 0·2개, 궤적 기록 켬·끔에서 EvalResult가 비트 단위로 같고, 관찰은 워커 수와 무관하다."""
    spec, layout, text, X, pairs, frame, codes = _batch(name)
    v = env_hash(spec.file)[:16]  # 워커는 환경 버전별로 컴파일해 둔다: 환경마다 다른 글
    record = [k % 3 == 0 for k in range(len(pairs))]
    single = WorkerPool(0)
    plain = single.evaluate(v, text, layout, X, pairs)
    plain_rec = single.evaluate(v, text, layout, X, pairs, record=record)
    assert plain.observations is None
    on0 = single.evaluate(v, text, layout, X, pairs, observe=codes, frame=frame)
    on2 = pool2.evaluate(v, text, layout, X, pairs, observe=codes, frame=frame)
    rec2 = pool2.evaluate(v, text, layout, X, pairs, record=record, observe=codes, frame=frame)
    off2 = pool2.evaluate(v, text, layout, X, pairs)
    for r in (on0, on2, off2):
        same_results(r, plain)
    same_results(rec2, plain_rec)
    same_pair_observations(on0.observations, on2.observations)
    same_pair_observations(on0.observations, rec2.observations)
    # 일부 쌍만(코드 0이 섞임): 관찰은 그 쌍만, 결과는 그대로
    some = [c if k % 2 == 0 else 0 for k, c in enumerate(codes)]
    part = pool2.evaluate(v, text, layout, X, pairs, observe=some, frame=frame)
    same_results(part, plain)
    assert part.observations is not None
    assert [o is not None for o in part.observations] == [bool(c) for c in some]
    none = pool2.evaluate(v, text, layout, X, pairs, observe=[0] * len(pairs), frame=frame)
    assert none.observations is None


def _reference(spec: Any, layout: Any, genome: np.ndarray, trial: Trial, frame: GridFrame | None
               ) -> tuple[list[int], list[float], int, dict[int, int], int]:
    """같은 유전자를 한 스텝씩 직접 재생해 파이썬으로 센 값(설계 4.0의 관찰기 규칙 그대로)."""
    body = layout.body_values(genome[None, :])[0] if layout.n_body else None
    ep = Episode(spec, body, trial)
    brain = unpack(layout.brain, layout.brain_params(genome[None, :]))
    idx = ep.output_channel_index
    U = layout.brain.n_out
    hist, total, n, cells, outside = [0] * (U * ACTION_BINS), [0.0] * U, 0, {}, 0
    anchor = ep.anchor_body
    while not ep.done:
        ep.step(layout.to_actions(brain.forward(ep.observe()[None, :].astype(np.float64)))[0])
        u = ep.applied_outputs
        for c, (lo, hi) in enumerate(layout.output_ranges):
            x = u[idx[c]]
            total[c] += x
            b = int((x - lo) * (ACTION_BINS / (hi - lo)))
            hist[c * ACTION_BINS + min(ACTION_BINS - 1, max(0, b))] += 1
        n += 1
        if frame is not None and n % POSITION_EVERY == 0:
            p = ep.bodies[anchor].position
            k = frame.cell_at(p.x, p.y)
            if k is None:
                outside += 1
            else:
                cells[k] = cells.get(k, 0) + 1
    return hist, total, n, cells, outside


@pytest.mark.parametrize("name", ["lander", "hill_car"])
def test_observer_counts_match_steps(name: str) -> None:
    """T2: 쌍별 채널마다 hist 합 = 스텝, 코드 2 쌍은 칸 수 + 틀 밖 = 스텝 // 4, 값은 직접 재생해 센 것과 비트 단위로 같다."""
    spec, layout, text, X, pairs, frame, codes = _batch(name)
    res = evaluate(spec, layout, X, pairs, observe=codes, frame=tuple(frame))
    assert res.observations is not None
    U = layout.brain.n_out
    for k, ((g, trial), o, code) in enumerate(zip(pairs, res.observations, codes, strict=True)):
        assert o is not None and o.steps == int(res.steps[k])
        h = o.hist.reshape(U, ACTION_BINS)
        assert o.hist.dtype == np.uint32 and h.sum(axis=1).tolist() == [o.steps] * U
        hist, total, n, cells, outside = _reference(spec, layout, X[g], trial, frame if code == 2 else None)
        assert n == o.steps and o.hist.tolist() == hist
        assert o.out_sum.dtype == np.float64 and o.out_sum.tobytes() == np.array(total).tobytes()
        if code == 2:
            assert int(o.counts.sum()) + o.outside == o.steps // POSITION_EVERY
            assert o.cells.dtype == np.int32 and o.counts.dtype == np.uint32
            assert o.cells.tolist() == sorted(cells) and o.counts.tolist() == [cells[c] for c in sorted(cells)]
            assert o.outside == outside
        else:
            assert o.cells.size == 0 and o.outside == 0


class _FakeBody:
    def __init__(self) -> None:
        self.position = (0.0, 0.0)


class _FakeEpisode:
    """관찰기가 읽는 것만 있는 가짜 Episode(채널 순서가 신경망 출력 순서와 다르게)."""

    def __init__(self) -> None:
        self.output_channel_index = [1, 0]
        self.applied_outputs = [0.0, 0.0]
        self.anchor_body: str | None = "agent"
        self.bodies = {"agent": _FakeBody()}
        self.steps = 0


def _feed(ob: Any, ep: _FakeEpisode, u0: float, u1: float, pos: tuple[float, float]) -> None:
    """스텝 하나: 출력 0 = 채널 1, 출력 1 = 채널 0."""
    ep.steps += 1
    ep.applied_outputs = [u1, u0]
    ep.bodies["agent"].position = pos
    ob.after_step(ep)


def test_observer_bin_edges() -> None:
    """T3: x = low → 0칸, x = high → 19칸, 칸 경계 바로 아래 → 앞 칸, 틀 왼쪽 0.5칸 밖은 outside(int() 버림 함정).
    세는 시점(flush)을 어디서 끊어도 결과가 같다."""
    ranges = [(0.0, 1.0), (-1.0, 1.0)]  # 출력 0(0..1) = 채널 1, 출력 1(-1..1) = 채널 0
    frame = (-10.0, 0.0, 2.0, 10, 5)
    ep = _FakeEpisode()
    ob = evaluate_mod._make_observer(ep, ranges, evaluate_mod.OBSERVE_POSITIONS, frame)  # type: ignore[arg-type]
    assert ob is not None and ob.body is not None
    below = float(np.nextafter(0.25, -1.0))
    steps = [  # (출력 0 값, 출력 1 값, 위치) → 기대 칸 (출력 0, 출력 1)
        ((0.0, -1.0, (-10.0, 0.0)), (0, 0)),
        ((1.0, 1.0, (-11.0, 0.5)), (19, 19)),  # 4번째 스텝이 아니라 위치는 세지 않는다
        ((0.25, -0.5, (0.0, 0.0)), (5, 5)),
        ((below, float(np.nextafter(-0.5, -2.0)), (-11.0, 0.5)), (4, 4)),  # 4스텝째: x0 - 0.5칸 → outside
        ((0.999, 0.999, (-10.0, 0.0)), (19, 19)),
        ((0.5, 0.0, (0.0, 0.0)), (10, 10)),
        ((-0.25, -2.0, (0.0, 0.0)), (0, 0)),  # 범위 밖(이론상) → 끝 칸
        ((0.05, 0.95, (9.99, 9.99)), (1, 19)),  # 8스텝째: 위치 (9.99, 9.99) → 칸 (ix 9, iy 4)
    ]
    want = np.zeros((2, ACTION_BINS), dtype=np.int64)
    for (u0, u1, pos), (b0, b1) in steps:
        _feed(ob, ep, u0, u1, pos)
        want[0, b0] += 1
        want[1, b1] += 1
    o = ob.build()
    assert o.steps == len(steps) and o.hist.reshape(2, ACTION_BINS).tolist() == want.tolist()
    sums = [0.0, 0.0]
    for (u0, u1, _), _ in steps:  # 스텝 순서로 더함(파이썬 3.12의 sum()은 보정 합이라 쓰지 않는다)
        sums[0] += u0
        sums[1] += u1
    assert o.out_sum.tolist() == sums
    assert o.outside == 1 and o.cells.tolist() == [4 * 10 + 9] and o.counts.tolist() == [1]
    # 틀이 없거나 기준 물체가 없으면 행동만, 코드 0이면 관찰기 없음
    assert evaluate_mod._make_observer(ep, ranges, 2, None).body is None  # type: ignore[arg-type, union-attr]
    assert evaluate_mod._make_observer(ep, ranges, 1, frame).body is None  # type: ignore[arg-type, union-attr]
    ep.anchor_body = None
    assert evaluate_mod._make_observer(ep, ranges, 2, frame).body is None  # type: ignore[arg-type, union-attr]
    assert evaluate_mod._make_observer(ep, ranges, 0, frame) is None  # type: ignore[arg-type]
    # 길게: 중간에 아무 데서나 세어도(flush) 끝에 한 번 센 것과 비트 단위로 같다
    ep.anchor_body = "agent"
    n = 3 * evaluate_mod.FLUSH_EVERY + 5
    vals = np.random.default_rng(3).uniform(0.0, 1.0, n)
    outs = []
    for cut in (None, 7, evaluate_mod.FLUSH_EVERY):
        ep.steps = 0
        ob = evaluate_mod._make_observer(ep, ranges, 2, frame)  # type: ignore[arg-type]
        assert ob is not None
        for k in range(n):
            _feed(ob, ep, float(vals[k]), float(-vals[k]), (-10.0 + 0.019 * k, 0.5))
            if cut is not None and (k + 1) % cut == 0:
                ob.flush()
        outs.append(ob.build())
    total0 = 0.0
    for v in vals:
        total0 += float(v)
    for o in outs:
        assert o.steps == n and o.out_sum[0] == total0 and int(o.hist[:ACTION_BINS].sum()) == n
        assert int(o.counts.sum()) + o.outside == n // POSITION_EVERY
        same_pair_observations([o], [outs[0]])


# ---------------------------------------------------------------------------
# T4: 격자 틀과 세계 열쇠
# ---------------------------------------------------------------------------

FRAME_SCRIPT = """
import json
from evolab import paths
from evolab.engine import observe
from evolab.engine.trainer import training_trials
from evolab.spec.io import compile_env, env_hash
out = {}
for name in ("lander", "track_car", "hill_car"):
    spec = compile_env((paths.builtin_env_dir() / f"{name}.evoenv").read_text(encoding="utf-8"))
    ver = env_hash(spec.file)[:16]
    out[name] = []
    for g in range(4):
        f, k = observe.world_frame(spec, ver, training_trials(spec, 1, g, spec.track_set_size or 2)[0])
        out[name].append([list(f), k])
print(json.dumps(out))
"""


def test_world_frame_is_world_only_and_deterministic() -> None:
    """T4: 틀·열쇠는 세계만의 함수(몸 수치와 무관), 새 프로세스에서도 같다. 트랙은 세대마다 열쇠가 같고(같은 트랙
    세트의 0번), 착륙선은 다르고(지형이 세대마다 다름), 언덕차는 틀이 같고 열쇠가 다르다."""
    here: dict[str, list[Any]] = {}
    for name in ("lander", "track_car", "hill_car"):
        spec = compile_env(text_of(name))
        ver = env_hash(spec.file)[:16]
        rows = []
        for g in range(4):
            trial = training_trials(spec, 1, g, spec.track_set_size or 2)[0]
            f, k = observing.world_frame(spec, ver, trial)
            assert f is not None and k is not None
            assert max(f.nx, f.ny) == observing.GRID_CELLS and min(f.nx, f.ny) >= observing.MIN_SIDE
            bx0, by0, bx1, by1 = (float(v) for v in spec.file.world.bounds)
            assert bx0 - 1e-9 <= f.x0 and by0 - 1e-9 <= f.y0
            assert f.x0 + f.nx * f.cell <= bx1 + f.cell and f.y0 + f.ny * f.cell <= by1 + f.cell
            rows.append([list(f), k])
        here[name] = rows
        frames = {tuple(r[0]) for r in rows}
        keys = {r[1] for r in rows}
        if name == "track_car":
            assert len(keys) == 1 and len(frames) == 1
        elif name == "lander":
            assert len(keys) == 4
        else:
            assert len(frames) == 1 and len(keys) == 4
    # 몸 수치가 다른 두 Episode(언덕차: 바퀴·차체가 진화)에서 계산해도 같다
    spec = compile_env(text_of("hill_car"))
    layout = layout_for(spec)
    assert layout.n_body > 0
    ver = env_hash(spec.file)[:16]
    trial = training_trials(spec, 1, 0, 2)[0]
    lo = layout.body_values(-np.ones((1, layout.dim)))[0]
    hi = layout.body_values(np.ones((1, layout.dim)))[0]
    assert not np.allclose(lo, hi)
    a = observing.frame_from_geometry(Episode(spec, lo, trial).geometry(), ver)
    b = observing.frame_from_geometry(Episode(spec, hi, trial).geometry(), ver)
    assert a == b == observing.world_frame(spec, ver, trial)
    # 환경 버전이 다르면 열쇠도 다르다(같은 세계 모양이어도 합치지 않음)
    assert observing.world_frame(spec, "other", trial)[1] != a[1]
    out = subprocess.run([sys.executable, "-c", FRAME_SCRIPT], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == json.loads(json.dumps(here))


# ---------------------------------------------------------------------------
# T5~T10: 학습 결과는 그대로, 관찰 기록은 결정적
# ---------------------------------------------------------------------------


def _run(base: Path, name: str, workers: int, gens: int = GENS, record: bool = False, observe: bool | None = None,
         pool: Any = None) -> tuple[RunStore, Trainer, list[dict[str, Any]]]:
    store = make_store(base, name)
    rows: list[dict[str, Any]] = []

    def go(p: Any) -> Trainer:
        t = Trainer(LANDER_TEXT, config(max_generations=gens), p, store, observe=observe)
        while t.check_stop() is None:
            rows.append(t.step(record=record))
            if t.observe:
                assert_same_arrays(t.last_observation, obs_of(store, t.state.generation), "message = disk")
        t.finish()
        return t

    if pool is not None:
        return store, go(pool), rows
    with WorkerPool(workers) as p:
        return store, go(p), rows


def _checkpoint_view(store: RunStore, g: int) -> tuple[bytes, ...]:
    """체크포인트에서 이후 학습을 정하는 것의 바이트: 학습기 상태(경과 시간 뺌), 예선 보관 출전자, 학습 방법 상태(pycma
    객체 뺌), 불러온 학습 방법의 다음 ask()와 best().

    pycma 객체(state["learner"]["es"]의 피클)는 벽시계 타이머·기록기를 품어 같은 실행을 두 번 돌려도 피클 바이트가
    달라진다(v1.1부터 그렇다). 그래서 그 객체는 불러와서 다음 후보가 같은지로 본다(test_restart의 next_ask와 같은 방식).
    """
    _, d = store.load_checkpoint(g)
    config = TrainConfig.from_dict(d["config"])

    def learner_view(blob: dict[str, Any], index: int) -> tuple[bytes, bytes, bytes]:
        learner = trainer_mod.make_learner(config, LAYOUT, index=index)
        learner.load_state_dict(blob)
        rest = {k: v for k, v in blob.items() if k != "es"}
        return pickle.dumps(rest, protocol=5), learner.ask().tobytes(), learner.best().tobytes()

    state = {k: v for k, v in d["state"].items() if k not in ("elapsed", "reserve")}
    reserve = d["state"]["reserve"]
    kept = None if reserve is None else (reserve["record"], learner_view(reserve["learner"], reserve["record"]["index"]))
    return (pickle.dumps(state, protocol=5), pickle.dumps(kept, protocol=5),
            *learner_view(d["learner"], int(d["state"]["current"]["index"])))


def _same_training(a: RunStore, ta: Trainer, b: RunStore, tb: Trainer, gens: int = GENS) -> None:
    assert strip(a.read_stats()) == strip(b.read_stats())
    assert ta.learner.best().tobytes() == tb.learner.best().tobytes()
    assert a.champion_generations("champion") == b.champion_generations("champion")
    for g in a.champion_generations("champion"):
        with np.load(a._champion_path(g, "champion")) as za, np.load(b._champion_path(g, "champion")) as zb:
            assert sorted(za.files) == sorted(zb.files)
            assert all(za[k].tobytes() == zb[k].tobytes() for k in za.files)
    assert a.champion_generations("representative") == list(range(1, gens + 1))
    for g in range(1, gens + 1):
        ra, rb = a.load_champion(g, "representative"), b.load_champion(g, "representative")
        assert sorted(ra) == sorted(rb) == BASE_KEYS
        assert all(ra[k].tobytes() == rb[k].tobytes() for k in ra)
    assert a.checkpoints() == b.checkpoints()
    for g in a.checkpoints():
        assert _checkpoint_view(a, g) == _checkpoint_view(b, g), g


def test_trainer_unchanged_by_observation(straight: Straight, tmp_path: Path, pool2: WorkerPool) -> None:
    """T5: 관찰 끔·켬 × 워커 0·2개(예선·자동 재시작이 일어나는 설정): 통계, 챔피언, 세대 대표 기본 값, 체크포인트
    (학습기 상태, 학습 방법 상태, 불러온 학습 방법의 다음 후보), learner.best()가 비트 단위로 같다."""
    off0, t_off0, _ = _run(tmp_path, "끔 워커 0", 0, observe=False)
    off2, t_off2, _ = _run(tmp_path, "끔 워커 2", 2, observe=False, pool=pool2)
    on2, t_on2, _ = _run(tmp_path, "켬 워커 2", 2, pool=pool2)
    assert not t_off0.observe and t_off0.last_observation is None
    for g in range(1, GENS + 1):
        assert obs_of(off0, g) is None and obs_of(off2, g) is None  # 끈 실행에는 관찰 키가 없다
    for store, t in ((off0, t_off0), (off2, t_off2), (on2, t_on2)):
        _same_training(store, t, straight.store, straight.trainer)
    assert_same_arrays_all(on2, straight.store)


def assert_same_arrays_all(a: RunStore, b: RunStore, gens: int = GENS) -> None:
    for g in range(1, gens + 1):
        assert_same_arrays(obs_of(a, g), obs_of(b, g), f"generation {g}")


def test_observation_independent_of_workers_and_spectating(straight: Straight, tmp_path: Path,
                                                           pool2: WorkerPool) -> None:
    """T6: 워커 0·1·2·3 × 관전(record) 켬·끔에서 세대별 관찰 배열이 같다(앞 4세대: 예선 경계 둘 포함)."""
    gens = 4
    pool3 = WorkerPool(3)
    try:
        runs = _t6_runs(tmp_path, gens, pool2, pool3)
    finally:
        pool3.close()
    for store, t, rows in runs:
        assert strip(rows) == strip(straight.rows[:gens])
        assert t.last_generation is not None and t.last_generation.observation is t.last_observation
        assert_same_arrays_all(store, straight.store, gens)


def _t6_runs(tmp_path: Path, gens: int, pool2: WorkerPool, pool3: WorkerPool) -> list[Any]:
    return [_run(tmp_path, "워커 0 관전", 0, gens, record=True),
            _run(tmp_path, "워커 1 관전", 1, gens, record=True),
            _run(tmp_path, "워커 2 관전", 2, gens, record=True, pool=pool2),
            _run(tmp_path, "워커 3", 3, gens, pool=pool3),
            _run(tmp_path, "워커 3 관전", 3, gens, record=True, pool=pool3)]


def test_observation_after_resume_equals_straight(straight: Straight, tmp_path: Path) -> None:
    """T7: 3세대 체크포인트 뒤 5세대까지 쓰고 끊긴 실행을 3세대에서 이어 가면, 남아 있던 4·5세대 세대 대표는 지워지고
    같은 배열로 다시 써진다."""
    store = make_store(tmp_path, "끊고 이어서")
    with WorkerPool(0) as pool:
        t = Trainer(LANDER_TEXT, config(checkpoint_every=3), pool, store)
        for _ in range(5):
            t.step()  # 끝내지 않음(체크포인트는 3세대뿐)
        assert store.checkpoints() == [3] and store.champion_generations("representative") == [1, 2, 3, 4, 5]
        for g in (4, 5):  # 남은 파일을 일부러 다르게 바꿔 둔다: 이어 가면 지워져야 한다
            base = store.load_champion(g, "representative")
            arrays = obs_of(store, g)
            assert arrays is not None
            arrays[fmt.SCORES] = arrays[fmt.SCORES] + 1.0
            store.save_champion(g, "representative", base["genome"], score=base["score"],
                                env_version=base["env_version"], **arrays)
        resumed = Trainer.resume(store, pool)
        assert store.champion_generations("representative") == [1, 2, 3]
        resumed.run()
    assert strip(store.read_stats()) == strip(straight.rows)
    assert resumed.learner.best().tobytes() == straight.trainer.learner.best().tobytes()
    assert_same_arrays_all(store, straight.store)


class FlakyPool:
    """WorkerPool(0) 감싸개(이 테스트 전용): 안쪽 평가기의 observes를 물려받고, fail_at번째 학습 평가(observe= 있음)
    에서 PoolError(워커가 죽은 것처럼)."""

    fail_at: int | None = None

    def __init__(self, workers: int | None = None, *args: Any, **kwargs: Any) -> None:
        self.inner = WorkerPool(0)
        self.calls = 0

    @property
    def observes(self) -> bool:
        return bool(getattr(self.inner, "observes", False))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def evaluate(self, *args: Any, **kwargs: Any) -> EvalResult:
        if "observe" in kwargs:
            self.calls += 1
            if self.calls == FlakyPool.fail_at:
                raise PoolError("worker died")
        return self.inner.evaluate(*args, **kwargs)

    def close(self) -> None:
        self.inner.close()

    def __enter__(self) -> FlakyPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class InProcessSession:
    """Session을 이 프로세스의 스레드로 돌린다(워커 0개: 평가도 이 프로세스, monkeypatch가 듣는다)."""

    def __init__(self, req: SessionRequest) -> None:
        self.cmd, cmd_child = mp.Pipe()
        self.events, ev_child = mp.Pipe()
        self.session = Session(cmd_child, ev_child, req)
        self.thread = threading.Thread(target=self.session.main, name="session-under-test", daemon=True)
        self.thread.start()
        self.msgs: list[dict[str, Any]] = []

    def wait(self, pred: Any, timeout: float = 180.0) -> dict[str, Any]:
        import time

        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.events.poll(0.2):
                msg = self.events.recv()
                self.msgs.append(msg)
                if pred(msg):
                    return msg
        raise AssertionError(f"timed out; got {[m.get('type') for m in self.msgs]}")

    def send(self, msg: dict[str, Any]) -> None:
        self.cmd.send(msg)

    def close(self) -> None:
        self.send({"type": "quit"})
        self.wait(lambda m: m["type"] == "closed")
        self.thread.join(timeout=60)
        assert not self.thread.is_alive()

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [m for m in self.msgs if m["type"] == kind]


def _finished(m: dict[str, Any]) -> bool:
    return m["type"] == "state" and m["state"] == "finished"


def test_pool_error_rewind_rewrites_observations(straight: Straight, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """T8: 학습 세션에서 4세대 평가 중 PoolError → 마지막 체크포인트(2세대)로 되감고 이어 간 실행의 관찰 = 끊지 않은 실행.
    3세대 세대 대표는 지워졌다가 같은 배열로 다시 써진다."""
    monkeypatch.setattr(pool_mod, "WorkerPool", FlakyPool)
    monkeypatch.setattr(FlakyPool, "fail_at", 4)
    req = SessionRequest(kind="new", env_text=LANDER_TEXT, runs_dir=str(tmp_path), workers=0, spectate=False,
                         config=config(checkpoint_every=2).to_dict())
    s = InProcessSession(req)
    try:
        started = s.wait(lambda m: m["type"] == "started")
        rewind = s.wait(lambda m: m["type"] == "rewind")
        assert rewind["generation"] == 2
        assert [m["generation"] for m in s.of("generation")] == [1, 2, 3]
        assert s.of("error") and not s.of("error")[0]["fatal"]
        s.wait(lambda m: m["type"] == "state" and m["state"] == PAUSED)
        s.send({"type": "run"})
        s.wait(_finished)
    finally:
        s.close()
    store = RunStore.open(Path(started["run_dir"]))
    assert strip(store.read_stats()) == strip(straight.rows)
    assert_same_arrays_all(store, straight.store)
    gens = s.of("generation")
    assert [m["generation"] for m in gens] == [1, 2, 3, 3, 4, 5, 6, 7]
    for m in gens[3:]:  # 되감은 뒤 메시지의 관찰 = 디스크
        assert_same_arrays(m["observation"], obs_of(store, m["generation"]), f"message {m['generation']}")


def test_branch_keeps_observations_until_generation(straight: Straight, tmp_path: Path) -> None:
    """T9: 분기한 실행의 세대 대표 ≤ g에 부모와 같은 관찰이 있고, 이어 가면 끝까지 부모와 같다."""
    g = 3
    child = RunStore.open(branch_run(straight.store.root, g, tmp_path / "분기"))
    assert child.champion_generations("representative") == [1, 2, 3]
    for k in range(1, g + 1):
        assert_same_arrays(obs_of(child, k), obs_of(straight.store, k), f"branch {k}")
    with WorkerPool(0) as pool:
        Trainer.resume(child, pool).run()
    assert strip(child.read_stats()) == strip(straight.rows)
    assert_same_arrays_all(child, straight.store)


def test_reproduce_bundle_regenerates_identical_observations(straight: Straight, tmp_path: Path) -> None:
    """T10: 재현 묶음에는 챔피언 파일이 없고, 다시 돌린 실행은 통계가 같고(compare_stats None) 관찰도 같다."""
    bundle = export_bundle(straight.store.root, tmp_path / "묶음.zip")
    with zipfile.ZipFile(bundle) as z:
        assert not any("champions" in n or n.endswith(".npz") for n in z.namelist())
    new = RunStore.open(reproduce(bundle, tmp_path / "다시", workers=0))
    assert compare_stats(straight.store.read_stats(), new.read_stats()) is None
    assert_same_arrays_all(new, straight.store)


# ---------------------------------------------------------------------------
# T11: v1.0 실행
# ---------------------------------------------------------------------------


def test_v10_run_gets_observations_from_resume_point(tmp_path: Path) -> None:
    """T11: v1.0 실행을 이어 2세대: 관찰은 이어 간 세대에만 있고, stats.csv 열 머리와 앞 줄은 그대로."""
    run = tmp_path / "v10" / "run"
    shutil.copytree(V10_RUN, run)
    store = RunStore.open(run)
    header = store.stats_columns()
    before = store.read_stats()
    last = int(before[-1]["generation"])
    for g in range(1, last + 1):
        assert obs_of(store, g) is None  # v1.0 세대: 관찰 없음(대체 그림, D12)
        assert sorted(store.load_champion(g, "representative")) == BASE_KEYS
    with WorkerPool(0) as pool:
        t = Trainer.resume(store, pool, None, max_generations=last + 2)
        t.run()
    assert store.stats_columns() == header
    rows = store.read_stats()
    assert rows[:last] == before and len(rows) == last + 2
    for g in range(1, last + 1):
        assert obs_of(store, g) is None
    for g in (last + 1, last + 2):
        obs = decoded(obs_of(store, g), g, t.layout)
        assert consistent(obs, rows[g - 1], store.load_champion(g, "representative")["score"])
        assert obs.learner == 0 and not obs.qualifying  # v1.0 실행은 학습 방법 0 하나(통계에 learner 열은 없다)


# ---------------------------------------------------------------------------
# T15~T22
# ---------------------------------------------------------------------------


def test_summary_failure_keeps_training(straight: Straight, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T15: 요약이 예외를 내면 그 세대 대표는 관찰 없이 써지고 통계·챔피언은 같으며, observe_problem은 처음 한 번만."""
    real = observing.summarize

    def flaky(**kw: Any) -> dict[str, np.ndarray]:
        if kw["generation"] in (3, 5):
            raise ValueError(f"boom {kw['generation']}")
        return real(**kw)

    monkeypatch.setattr(observing, "summarize", flaky)
    store = make_store(tmp_path, "요약 실패")
    with WorkerPool(0) as pool:
        t = Trainer(LANDER_TEXT, config(), pool, store)
        t.run()
    assert t.observe_problem == S.TRAIN_OBSERVE_FAILED.format(reason="boom 3")
    _same_training(store, t, straight.store, straight.trainer)
    for g in range(1, GENS + 1):
        if g in (3, 5):
            assert obs_of(store, g) is None
        else:
            assert_same_arrays(obs_of(store, g), obs_of(straight.store, g), f"generation {g}")


def test_summary_failure_is_reported_once_by_the_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T15(세션): 요약이 세대마다 실패해도 화면에는 비치명 오류가 한 번만 가고 학습은 끝까지 간다. 세대 메시지의
    "observation"은 None."""

    def broken(**kw: Any) -> dict[str, np.ndarray]:
        raise ValueError("boom")

    monkeypatch.setattr(observing, "summarize", broken)
    req = SessionRequest(kind="new", env_text=LANDER_TEXT, runs_dir=str(tmp_path), workers=0, spectate=False,
                         config=config(max_generations=3, qualify_entrants=1, restart_patience=None).to_dict())
    s = InProcessSession(req)
    try:
        s.wait(_finished)
    finally:
        s.close()
    errors = s.of("error")
    assert errors == [{"type": "error", "fatal": False, "message": S.TRAIN_OBSERVE_FAILED.format(reason="boom")}]
    gens = s.of("generation")
    assert [m["generation"] for m in gens] == [1, 2, 3]
    assert all("observation" in m and m["observation"] is None for m in gens)


def test_session_generation_message_carries_the_observation(tmp_path: Path) -> None:
    """D13: 세대 메시지의 "observation" = 그 세대 대표 파일의 obs_ 키(관전 모드에서도). 시작 스냅숏은 그대로."""
    req = SessionRequest(kind="new", env_text=LANDER_TEXT, runs_dir=str(tmp_path), workers=0, spectate=False,
                         config=config(max_generations=2, qualify_entrants=1, restart_patience=None).to_dict())
    s = InProcessSession(req)
    try:
        started = s.wait(lambda m: m["type"] == "started")
        s.wait(_finished)
    finally:
        s.close()
    assert "observation" not in started
    store = RunStore.open(Path(started["run_dir"]))
    for m in s.of("generation"):
        assert set(m) >= {"population", "scores", "representative", "trial", "trajectories", "champion",
                          "env_version", "observation"}
        assert_same_arrays(m["observation"], obs_of(store, m["generation"]))
        assert decoded(m["observation"], m["generation"]).representative == m["representative"]
    assert not s.of("error")


def _fake_result(P: int, T: int, U: int = 3) -> EvalResult:
    n = P * T
    return EvalResult(scores=np.arange(n, dtype=np.float64), steps=np.full(n, 10, np.int64),
                      end_events=["landed", "crash", "", "nope"] * (n // 4) + ["timeout"] * (n % 4),
                      outcomes=["success", "failure", "neutral", "failure"] * (n // 4) + ["neutral"] * (n % 4),
                      blocks=np.zeros((n, 1)), action_abs=np.zeros((n, U)), weighted=np.zeros((n, 1)))


def test_genome_budget_and_float16() -> None:
    """T16: P·D > 32,768이면 k세대마다만 obs_genomes, 큰 값은 ±65504로 잘림(inf 없음). 결과·종료 사건 번호 규칙."""
    P, T = 160, 2  # 160 × 211 = 33,760 > 32,768 → k = 2
    assert observing.genome_stride(P, LAYOUT.dim) == 2 and observing.genome_stride(32, LAYOUT.dim) == 1
    rng = np.random.default_rng(5)
    X = rng.standard_normal((P, LAYOUT.dim)) * 1e5
    agg = rng.standard_normal(P)
    res = _fake_result(P, T)
    ends = [t.event for t in LANDER_SPEC.terminations]
    kw = {"X": X, "agg": agg, "res": res, "trials_per": T, "representative": int(np.argmax(agg)), "learner": 2,
          "qualifying": True, "layout": LAYOUT, "end_ids": ends, "trial0": Trial(2**64 - 5, None), "frame": None,
          "world_key": None}
    odd = observing.summarize(generation=3, **kw)
    even = observing.summarize(generation=4, **kw)
    assert fmt.GENOMES not in odd and fmt.GENOMES in even
    g16 = even[fmt.GENOMES]
    assert g16.dtype == np.float16 and np.isfinite(g16).all() and float(np.abs(g16).max()) == fmt.FLOAT16_MAX
    small = np.abs(X) < 1000
    assert np.allclose(g16.astype(np.float64)[small], X[small], rtol=1e-3, atol=1e-3)
    for g, arrays in ((3, odd), (4, even)):
        obs = decoded(arrays, g)
        assert obs.genome_stride == 2 and obs.learner == 2 and obs.qualifying and obs.trial_seed == 2**64 - 5
        assert not obs.has_actions and not obs.has_density and obs.outside == 0
        assert obs.outcome[0].tolist() == [1, 2] and obs.outcome[1].tolist() == [0, 2]
        assert obs.end[0].tolist() == [ends.index("landed"), ends.index("crash")]
        assert obs.end[1].tolist() == [fmt.END_NONE, fmt.END_NONE]  # 사건 없음, 목록에 없는 사건
    with pytest.raises(ValueError):  # 모양이 맞지 않으면 학습기가 잡는 예외(D9)
        observing.summarize(generation=1, **{**kw, "trials_per": 3})


def test_head_matches_learner_columns(straight: Straight) -> None:
    """T17: obs_head의 학습 방법 번호·예선 값 = 통계 learner·qualifying(예선 출전 둘, 재시작 뒤 학습 방법까지)."""
    rows = straight.store.read_stats()
    seen = set()
    for g in range(1, GENS + 1):
        obs = decoded(obs_of(straight.store, g), g)
        r = rows[g - 1]
        assert (obs.learner, int(obs.qualifying)) == (int(r["learner"]), int(r["qualifying"])), g
        assert obs.generation == g
        seen.add((obs.learner, obs.qualifying))
    assert seen == {(0, True), (1, True), (1, False), (2, False)}


def test_env_change_and_trials_change_mid_run(tmp_path: Path) -> None:
    """T18: 환경 교체(지형 시드만 다른 착륙선) 뒤 그 세대부터 틀·열쇠·env_version이 새 환경 것이고, 시험 횟수를
    바꾸면 그 세대부터 obs_outcome이 (P, 3)."""
    data = json.loads(LANDER_TEXT)
    gen = next(g for g in data["generators"] if g["kind"] == "heightmap")
    gen["seed"] = "=r.terrain_seed + 7"
    other = json.dumps(data, ensure_ascii=False, indent=2)
    store = make_store(tmp_path, "환경 교체")
    with WorkerPool(0) as pool:
        t = Trainer(LANDER_TEXT, config(qualify_entrants=1, restart_patience=None, max_generations=None), pool, store)
        t.step()
        t.step()
        old_version = t.env_version
        new_version = t.replace_env(other)
        assert new_version != old_version
        store.add_env_version(other, t.state.generation)
        t.step()
        trial0 = training_trials(t.spec, t.config.run_seed, 2, 2)[0]
        t.apply_config({"trials": 3})
        t.step()
    new_spec, old_spec = compile_env(other), LANDER_SPEC
    obs2, obs3, obs4 = (decoded(obs_of(store, g), g) for g in (2, 3, 4))
    assert str(store.load_champion(2, "representative")["env_version"]) == old_version
    assert str(store.load_champion(3, "representative")["env_version"]) == new_version
    f_new, k_new = observing.world_frame(new_spec, new_version, trial0)
    f_old, k_old = observing.world_frame(old_spec, old_version, trial0)
    assert (obs3.frame, obs3.world_key) == (f_new, k_new) and k_new != k_old
    same = "same version"  # 버전 글을 같게 해도 세계 모양(지형)이 달라 열쇠가 다르다
    assert (observing.frame_from_geometry(Episode(new_spec, None, trial0).geometry(), same)[1]
            != observing.frame_from_geometry(Episode(old_spec, None, trial0).geometry(), same)[1])
    assert obs2.trials == obs3.trials == 2 and obs4.trials == 3
    assert obs4.outcome.shape == (6, 3) and obs4.end.shape == (6, 3)
    rows = store.read_stats()
    for g, obs in ((2, obs2), (3, obs3), (4, obs4)):
        assert consistent(obs, rows[g - 1], store.load_champion(g, "representative")["score"])


class PlainEvaluator:
    """observes 표시가 없는 평가기(가짜 평가기처럼): observe=·frame= 키워드를 받지 않는다."""

    def __init__(self, inner: WorkerPool) -> None:
        self.inner = inner
        self.calls = 0

    def evaluate(self, env_version: str, env_text: str, layout: Any, genomes: np.ndarray,
                 pairs: list[tuple[int, Any]], record: list[bool] | None = None) -> EvalResult:
        self.calls += 1
        return self.inner.evaluate(env_version, env_text, layout, genomes, pairs, record=record)


def test_fake_evaluator_without_observes(tmp_path: Path) -> None:
    """T19: observes가 없는 평가기는 observe= 키워드를 받지 않고(받으면 TypeError), 관찰에는 유전자·점수·결과·몸만
    있고 행동·밀도 키는 없다(언덕차: 몸 유전자가 있다)."""
    text = text_of("hill_car")
    store = make_store(tmp_path, "가짜 평가기", text)
    with WorkerPool(0) as pool:
        ev = PlainEvaluator(pool)
        assert not getattr(ev, "observes", False)
        t = Trainer(text, TrainConfig(popsize=4, trials=2, validate_every=1000, run_seed=3), ev, store)
        t.step(record=True)
        t.step()
    assert ev.calls == 2
    layout = t.layout
    for g in (1, 2):
        arrays = obs_of(store, g)
        assert arrays is not None and not (set(arrays) & (ACTION_KEYS | DENSITY_KEYS))
        assert set(arrays) == {fmt.HEAD, fmt.TRIAL, fmt.SCORES, fmt.OUTCOME, fmt.END, fmt.GENOMES, fmt.BODY}
        obs = decoded(arrays, g, layout)
        assert obs.body is not None and obs.body.shape == (4, layout.n_body)
        assert obs.outside == 0 and obs.genomes is not None
    # 관찰을 끈 학습기는 관찰 평가기에도 observe=를 넘기지 않는다
    with WorkerPool(0) as pool:
        calls: list[dict[str, Any]] = []
        real = pool.evaluate

        def spy(*a: Any, **k: Any) -> EvalResult:
            calls.append(k)
            return real(*a, **k)

        pool.evaluate = spy  # type: ignore[method-assign]
        t = Trainer(text, TrainConfig(popsize=4, trials=2, validate_every=1000, run_seed=3), pool, None)
        assert not t.observe  # 저장소가 없으면 기본으로 끔
        t.step()
        t2 = Trainer(text, TrainConfig(popsize=4, trials=2, validate_every=1000, run_seed=3), pool, None, observe=True)
        t2.step()
    assert calls[0] == {} and set(calls[1]) == {"observe", "frame"}
    assert t.last_observation is None and t2.last_observation is not None


def test_load_champion_hides_observation_keys(straight: Straight, tmp_path: Path) -> None:
    """T20: load_champion 결과 키는 v1.1과 같고(genome, score, env_version), load_observation은 obs_ 키만. 없거나
    깨진 파일, 관찰 키가 없는 파일은 None."""
    store = straight.store
    for g in (1, GENS):
        assert sorted(store.load_champion(g, "representative")) == BASE_KEYS
        arrays = store.load_observation(g)
        assert arrays is not None and all(k.startswith(fmt.OBS_PREFIX) for k in arrays)
        assert {fmt.HEAD, fmt.SCORES, fmt.GENOMES} <= set(arrays)
        with np.load(store._champion_path(g, "representative")) as z:  # 파일에는 둘 다 있다
            assert set(z.files) == set(BASE_KEYS) | set(arrays)
    champ = store.champion_generations("champion")[0]
    assert "genome" in store.load_champion(champ, "champion")
    assert store.load_observation(GENS + 5) is None
    v10 = RunStore.open(V10_RUN)
    assert v10.load_observation(1) is None and sorted(v10.load_champion(1, "representative")) == BASE_KEYS
    copy = tmp_path / "copy"
    shutil.copytree(store.root, copy)
    broken = RunStore.open(copy)
    broken._champion_path(2, "representative").write_bytes(b"PK\x03\x04 not a zip")
    assert broken.load_observation(2) is None


def _cli_train(out: Path, *extra: str) -> RunStore:
    code = cli.main(["train", str(LANDER), "--seed", "3", "--popsize", "4", "--trials", "2", "--max-generations", "3",
                     "--workers", "0", "--out", str(out), "--validate-every", "1000", *extra])
    assert code == 0
    (run,) = list(out.iterdir())
    return RunStore.open(run)


def test_cli_no_observe(tmp_path: Path) -> None:
    """T21: train --no-observe의 세대 대표에는 관찰 키가 없고, 통계·run.json 설정은 켠 실행과 같다. 이어 갈 때도 끈다."""
    on = _cli_train(tmp_path / "켬")
    off = _cli_train(tmp_path / "끔", "--no-observe")
    assert strip(on.read_stats()) == strip(off.read_stats())
    assert on.read_run_info()["config"] == off.read_run_info()["config"]
    assert "observe" not in json.dumps(off.read_run_info()["config"])
    for g in (1, 2, 3):
        assert off.load_observation(g) is None and on.load_observation(g) is not None
        a, b = on.load_champion(g, "representative"), off.load_champion(g, "representative")
        assert all(a[k].tobytes() == b[k].tobytes() for k in BASE_KEYS)
    assert cli.main(["train", "--resume", str(off.root), "--max-generations", "4", "--workers", "0",
                     "--no-observe"]) == 0
    assert off.load_observation(4) is None
    assert cli.main(["train", "--resume", str(off.root), "--max-generations", "5", "--workers", "0"]) == 0
    assert off.load_observation(5) is not None  # 켜서 이어 가면 그 세대부터 남긴다
    assert cli.main(["train", "--resume", str(on.root), "--max-generations", "5", "--workers", "0"]) == 0
    assert strip(on.read_stats()) == strip(off.read_stats())


def test_bench_prints_observe_line(capsys: pytest.CaptureFixture[str]) -> None:
    """T22: bench가 단일 코어 측정 뒤 관찰을 켠 같은 평가의 줄(BENCH_OBSERVE)을 출력한다."""
    assert cli.main(["bench", str(LANDER), "--workers", "1", "--generations", "1", "--popsize", "2"]) == 0
    out = capsys.readouterr().out
    head = S.BENCH_OBSERVE.split("{", 1)[0]
    lines = [line for line in out.splitlines() if line.startswith(head)]
    assert len(lines) == 1 and "%" in lines[0]
    single = next(i for i, line in enumerate(out.splitlines()) if line.startswith(S.BENCH_SINGLE.split("{", 1)[0]))
    assert out.splitlines()[single + 1] == lines[0]  # 단일 코어 줄 바로 다음

"""v6.0 평가 창과 워커 GC 얼림 (v6 설계 4.7·4.9·D15·D16, 8.1 T3·T4·T14·T15, E 묶음).

- T3: 창 1·3·12·64와 쌍 순서·유전자 행 순서를 섞은 평가의 쌍별 결과가 바이트 같음(조립 실패·궤적 기록·관찰 섞음).
- T4: 끝난 줄은 observe·step을 받지 않음: 쌍마다 observe 호출 수 = 스텝 수, 끝난 뒤 step 0번.
- T14: 창 크기 B일 때 동시에 살아 있는 Episode가 B개를 넘지 않음(weakref로 셈, 셀 때마다 gc.collect).
- T15: 워커 시작 GC 얼림: 워커 하나에 착륙선 14쌍 작업 30개를 보낸 뒤 살아 있는 Episode가 한 작업 쌍 수 이하, gc.freeze는
  한 번만 불림(워커 안 계측), 결과가 WORKER_GC_FREEZE = False와 같음. 화면 프로세스·WorkerPool(0)은 얼리지 않음.

계측 워커(_instrumented_worker)는 spawn으로 띄우므로 이 모듈의 맨 위 import는 가볍게 둔다(자식이 이 모듈을 다시 불러온다).
"""

from __future__ import annotations

import gc
import weakref
from typing import Any

import numpy as np
import pytest

from evolab import paths
from evolab.engine import evaluate as evaluate_mod
from evolab.engine import pool as pool_mod
from evolab.engine import seeds
from evolab.engine.evaluate import EVAL_WINDOW, EvalResult, evaluate
from evolab.policy.genome import layout_for
from evolab.spec.io import compile_env
from evolab.world.physics2d.episode import Episode, Trial

LANDER_TEXT = (paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8")
JUMP_TEXT = (paths.builtin_env_dir() / "jump.evoenv").read_text(encoding="utf-8")
TRACK_TEXT = (paths.builtin_env_dir() / "track_car.evoenv").read_text(encoding="utf-8")


def same(a: EvalResult, b: EvalResult) -> None:
    """같은 클래스의 두 EvalResult가 바이트까지 같다(궤적·관찰 포함)."""
    for name in ("scores", "steps", "blocks", "action_abs", "weighted"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), name
    assert a.end_events == b.end_events and a.outcomes == b.outcomes
    assert (a.trajectories is None) == (b.trajectories is None)
    for u, v in zip(a.trajectories or [], b.trajectories or [], strict=True):
        assert (u is None) == (v is None)
        if u is not None and v is not None:
            assert u.poses.shape == v.poses.shape and u.poses.tobytes() == v.poses.tobytes()
            assert u.outputs.shape == v.outputs.shape and u.outputs.tobytes() == v.outputs.tobytes()
            assert (u.every, u.dt, u.end_step) == (v.every, v.dt, v.end_step)
    assert (a.observations is None) == (b.observations is None)
    for u, v in zip(a.observations or [], b.observations or [], strict=True):
        assert (u is None) == (v is None)
        if u is not None and v is not None:
            for name in ("hist", "out_sum", "cells", "counts"):
                assert getattr(u, name).tobytes() == getattr(v, name).tobytes(), name
            assert (u.steps, u.outside) == (v.steps, v.outside)


def twist_batch() -> tuple[Any, Any, np.ndarray, list[tuple[int, Trial]]]:
    """꼬인 착륙선(tests/test_build_failure.py) 개체 다섯(1·3번은 조립 실패) × 시험 넷 = 20쌍, 1초 에피소드."""
    from tests.test_build_failure import gene, twist_lander

    spec = compile_env(twist_lander(max_time=1.5))
    layout = layout_for(spec)
    rng = np.random.default_rng(3)
    X = np.vstack([gene(layout, {"twist": t}) for t in (0.0, -2.0, 0.5, -1.5, 0.9)])
    X[:, layout.n_body:] = 0.4 * rng.standard_normal((len(X), layout.dim - layout.n_body))
    pairs = [(i, Trial(seeds.derive(31, k))) for i in range(len(X)) for k in range(4)]
    return spec, layout, X, pairs


def random_batch(text: str, pop: int, trials: int, seed: int) -> tuple[Any, Any, np.ndarray, list[tuple[int, Trial]]]:
    spec = compile_env(text)
    layout = layout_for(spec)
    rng = np.random.default_rng(seed)
    X = layout.initial()[None, :] + 0.6 * rng.standard_normal((pop, layout.dim))
    from evolab.engine.trainer import training_trials

    pairs = [(i, t) for i in range(pop) for t in training_trials(spec, 2, 0, trials)]
    return spec, layout, X, pairs


def mixed_kw(n: int) -> dict[str, Any]:
    return {"record": [i % 3 == 1 for i in range(n)], "every": 3, "observe": [(i * 7) % 3 for i in range(n)]}


# ---------------------------------------------------------------------------
# T3 창 크기·순서 무관
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["twist_lander", "jump", "track_car"])
def test_t3_results_do_not_depend_on_window_or_pair_order(which: str) -> None:
    if which == "twist_lander":
        spec, layout, X, pairs = twist_batch()
    elif which == "jump":
        spec, layout, X, pairs = random_batch(JUMP_TEXT, 6, 3, 5)
    else:
        spec, layout, X, pairs = random_batch(TRACK_TEXT, 4, 3, 6)
    n = len(pairs)
    from evolab.engine import observe as observing

    frame, _ = observing.world_frame(spec, "t3", pairs[0][1])
    kw = mixed_kw(n)
    fr = None if frame is None else tuple(frame.as_array())
    base = evaluate(spec, layout, X, pairs, frame=fr, **kw)
    if which == "twist_lander":
        assert np.isnan(base.scores).sum() == 8
    rng = np.random.default_rng(17)
    for window in (1, 3, 12, 64):
        for shuffled in (False, True):
            if shuffled:  # 쌍 순서와 유전자 행 순서를 함께 섞는다
                order = rng.permutation(n)
                rows = rng.permutation(len(X))
                where = np.argsort(rows)  # 옛 행 → 새 행
                X2 = X[rows]
                pairs2 = [(int(where[pairs[k][0]]), pairs[k][1]) for k in order]
                kw2 = {"record": [kw["record"][k] for k in order], "every": kw["every"],
                       "observe": [kw["observe"][k] for k in order]}
                res = evaluate(spec, layout, X2, pairs2, frame=fr, window=window, **kw2)
                back = np.argsort(order)  # 원래 쌍 i는 섞은 묶음의 back[i]번째
                res = EvalResult.concat([res], [(0, int(back[i])) for i in range(n)])
            else:
                res = evaluate(spec, layout, X, pairs, frame=fr, window=window, **kw)
            same(res, base)


def test_window_none_is_eval_window_and_bad_windows_are_refused() -> None:
    spec, layout, X, pairs = twist_batch()
    assert EVAL_WINDOW == 12
    same(evaluate(spec, layout, X, pairs), evaluate(spec, layout, X, pairs, window=EVAL_WINDOW))
    for bad in (0, -1):
        with pytest.raises(ValueError):
            evaluate(spec, layout, X, pairs, window=bad)
    with pytest.raises(TypeError):
        evaluate(spec, layout, X, pairs, window=2.5)  # type: ignore[arg-type]
    empty = evaluate(spec, layout, X, [], window=3)
    assert empty.scores.shape == (0,) and empty.blocks.shape == (0, len(spec.file.scoring.blocks))
    assert empty.trajectories is None and empty.observations is None


# ---------------------------------------------------------------------------
# T4·T14 계측 Episode
# ---------------------------------------------------------------------------


class _Counted(Episode):
    """observe·step 호출을 세고 살아 있는 것을 weakref로 모으는 Episode(이 테스트만)."""

    alive: weakref.WeakSet[Any] = weakref.WeakSet()
    peak = 0
    probe_every = 0
    calls = 0

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.n_observe = 0
        self.after_done = 0
        self.is_reference = args[1] is None and kw.get("body") is None
        type(self).alive.add(self)

    def observe(self) -> np.ndarray:
        self.n_observe += 1
        if self.done:
            self.after_done += 1
        cls = type(self)
        cls.calls += 1
        if cls.probe_every and cls.calls % cls.probe_every == 0:
            gc.collect()  # 놓은 Episode(클로저 순환)를 거둔 뒤 센다
            cls.peak = max(cls.peak, len(cls.alive))
        return super().observe()

    def step(self, action: Any) -> None:
        if self.done:
            self.after_done += 1
        super().step(action)

    def step_values(self, values: list[float]) -> None:  # 런타임에 없으면 평가기가 부르지 않는다
        if self.done:
            self.after_done += 1
        getattr(super(), "step_values")(values)


if not hasattr(Episode, "step_values"):
    del _Counted.step_values


@pytest.fixture
def counted(monkeypatch: pytest.MonkeyPatch) -> type[_Counted]:
    _Counted.alive = weakref.WeakSet()
    _Counted.peak = 0
    _Counted.calls = 0
    _Counted.probe_every = 0
    monkeypatch.setattr(evaluate_mod, "Episode", _Counted)
    return _Counted


@pytest.mark.parametrize("window", [1, 4, 12])
def test_t4_finished_rows_get_no_observe_or_step(counted: type[_Counted], window: int) -> None:
    spec, layout, X, pairs = twist_batch()
    seen: list[_Counted] = []
    real_result = _Counted.result

    def result(self: _Counted) -> Any:  # 결과를 뗄 때 호출 수를 잡아 둔다(그 뒤 Episode는 놓인다)
        if not self.is_reference:
            seen.append(self)
        return real_result(self)

    counted.result = result  # type: ignore[method-assign]
    try:
        res = evaluate(spec, layout, X, pairs, window=window, **mixed_kw(len(pairs)))
    finally:
        del counted.result  # type: ignore[attr-defined]
    built = [i for i in range(len(pairs)) if not np.isnan(res.scores[i])]
    assert len(seen) == len(built) == 12
    # 결과를 떼는 순서는 끝나는 순서라 쌍 순서와 다를 수 있다: 스텝 수의 여러 벌로 맞춘다
    assert sorted(ep.steps for ep in seen) == sorted(int(res.steps[i]) for i in built)
    for ep in seen:
        assert ep.done and ep.n_observe == ep.steps and ep.after_done == 0


@pytest.mark.parametrize("window", [3, 12])
def test_t14_live_episodes_never_exceed_the_window(counted: type[_Counted], window: int) -> None:
    spec, layout, X, pairs = random_batch(LANDER_TEXT, 8, 4, 9)  # 32쌍
    counted.probe_every = 149  # 셀 때마다 gc.collect(이 프로세스는 커서 한 번에 수십 ms)
    res = evaluate(spec, layout, X, pairs, window=window, record=[i % 2 == 0 for i in range(len(pairs))])
    assert counted.calls == int(res.steps.sum()) and counted.calls // counted.probe_every >= 12
    assert counted.peak == window  # 창이 다 찬 때가 있고 넘지 않는다
    gc.collect()
    assert len(counted.alive) == 0  # 평가가 끝나면 하나도 붙잡지 않는다


# ---------------------------------------------------------------------------
# T15 워커 시작 GC 얼림
# ---------------------------------------------------------------------------


def _instrumented_worker(task_r: Any, res_w: Any, stats_w: Any, freeze: bool) -> None:  # pragma: no cover - 자식 프로세스
    """pool._worker_main을 그대로 돌리되 gc.freeze 호출 수와 살아 있는 Episode를 센다(워커 안 계측)."""
    import gc as gc_
    import weakref as weakref_

    from evolab.engine import evaluate as ev
    from evolab.engine import pool

    freezes: list[int] = []
    real_freeze = gc_.freeze

    def counting_freeze() -> None:
        freezes.append(1)
        real_freeze()

    gc_.freeze = counting_freeze
    alive: weakref_.WeakSet[Any] = weakref_.WeakSet()
    base = ev.Episode

    class Tracked(base):  # type: ignore[valid-type, misc]
        def __init__(self, *args: Any, **kw: Any) -> None:
            super().__init__(*args, **kw)
            alive.add(self)

    ev.Episode = Tracked
    at_start: list[int] = []
    real_evaluate = pool.evaluate

    def evaluate_(*args: Any, **kw: Any) -> Any:
        at_start.append(len(alive))  # 거두지 않고 센 값(앞 작업의 쓰레기가 아직 남았을 수 있음)
        return real_evaluate(*args, **kw)

    pool.evaluate = evaluate_
    pool._worker_main(0, task_r, res_w, freeze)
    gc_.collect()
    stats_w.send({"freezes": len(freezes), "frozen": gc_.get_freeze_count(), "alive": len(alive),
                  "at_start": at_start})


def _run_instrumented(freeze: bool, tasks: list[Any]) -> tuple[list[EvalResult], dict[str, Any]]:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    task_r, task_w = ctx.Pipe(duplex=False)
    res_r, res_w = ctx.Pipe(duplex=False)
    stats_r, stats_w = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_instrumented_worker, args=(task_r, res_w, stats_w, freeze), daemon=True)
    proc.start()
    task_r.close()
    res_w.close()
    stats_w.close()
    out: list[EvalResult] = []
    try:
        for task in tasks:
            task_w.send(task)
            assert res_r.poll(120), "worker did not answer"
            task_id, status, payload = res_r.recv()
            assert task_id == task.id and status == "ok", payload
            out.append(payload)
        task_w.send(None)
        assert stats_r.poll(60)
        stats = stats_r.recv()
        proc.join(30)
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
        for c in (task_w, res_r, stats_r):
            c.close()
    return out, stats


def lander_tasks(count: int, pairs_per_task: int) -> tuple[list[Any], Any, np.ndarray, list[list[tuple[int, Trial]]]]:
    spec = compile_env(LANDER_TEXT)
    layout = layout_for(spec)
    rng = np.random.default_rng(15)
    X = layout.initial()[None, :] + 0.5 * rng.standard_normal((pairs_per_task, layout.dim))
    tasks, batches = [], []
    for k in range(count):
        trials = seeds.train_trials(4, k, 1)
        pairs = [(i, trials[0]) for i in range(pairs_per_task)]
        batches.append(pairs)
        tasks.append(pool_mod.Task(k + 1, "v", LANDER_TEXT, layout.to_dict(), X, [(g, t.seed, t.track) for g, t in pairs],
                                   [i == 0 for i in range(pairs_per_task)], 2, [1] * pairs_per_task, None))
    return tasks, layout, X, batches


@pytest.mark.slow
def test_t15_worker_freezes_once_after_the_first_task_and_holds_no_episodes() -> None:
    tasks, layout, X, batches = lander_tasks(30, 14)
    on, stats_on = _run_instrumented(True, tasks)
    off, stats_off = _run_instrumented(False, tasks)
    assert stats_on["freezes"] == 1 and stats_on["frozen"] > 0
    assert stats_off["freezes"] == 0 and stats_off["frozen"] < stats_on["frozen"]
    # 살아 있는 Episode: 거둔 뒤에는 0(얼린 것이 하나도 없음). 첫 작업의 Episode는 얼리기 전 collect가 거뒀다(둘째 작업을
    # 시작할 때 0). 거두지 않고 센 값(작업을 시작할 때 아직 안 거둔 앞 작업의 쓰레기)은 얼림 때문에 늘지 않는다: 리눅스에서
    # 둘 다 14~51개(작업 1~3.6개분)를 오르내렸다. 수거 시점은 기계마다 달라 얼리지 않은 워커 + 한 작업 쌍 수를 상한으로 둔다.
    for stats in (stats_on, stats_off):
        assert stats["alive"] == 0 and len(stats["at_start"]) == 30
    assert stats_on["at_start"][:2] == [0, 0]
    assert max(stats_on["at_start"]) <= max(stats_off["at_start"]) + 14, (stats_on["at_start"], stats_off["at_start"])
    spec = compile_env(LANDER_TEXT)
    for a, b, pairs in zip(on, off, batches, strict=True):
        same(a, b)
        same(a, evaluate(spec, layout, X, pairs, record=[i == 0 for i in range(14)], observe=[1] * 14))


def test_t15_short_worker_run_freezes_once_and_matches_no_freeze() -> None:
    """T15 빠른 판: 작업 4개(쌍 5개씩)로 얼림이 한 번, 얼리지 않은 워커와 결과가 같다."""
    tasks, layout, X, batches = lander_tasks(4, 5)
    on, stats_on = _run_instrumented(True, tasks)
    off, stats_off = _run_instrumented(False, tasks)
    assert stats_on["freezes"] == 1 and stats_off["freezes"] == 0
    assert stats_on["alive"] == stats_off["alive"] == 0
    assert stats_on["at_start"][:2] == [0, 0]  # 첫 작업의 Episode는 얼리기 전에 거둠(얼려 붙잡지 않음)
    for a, b in zip(on, off, strict=True):
        same(a, b)


class _SpyPool(pool_mod.WorkerPool):
    """워커를 띄울 때 넘기는 인수를 모으는 풀(이 테스트만)."""

    def _spawn(self, w: int) -> None:
        real: Any = self._ctx  # type: ignore[has-type]
        seen = self.__dict__.setdefault("spawn_args", [])

        class Ctx:
            def Pipe(self, duplex: bool) -> Any:  # noqa: N802 - multiprocessing 이름
                return real.Pipe(duplex=duplex)

            def Process(self, **kw: Any) -> Any:  # noqa: N802
                seen.append(kw["args"])
                return real.Process(**kw)

        self._ctx = Ctx()
        try:
            super()._spawn(w)
        finally:
            self._ctx = real


def test_pool_passes_its_freeze_flag_to_the_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert pool_mod.WORKER_GC_FREEZE is True
    with _SpyPool(1) as pool:
        assert [a[3] for a in pool.spawn_args] == [True]  # type: ignore[attr-defined]
    monkeypatch.setattr(pool_mod, "WORKER_GC_FREEZE", False)
    with _SpyPool(1) as pool:
        assert [a[3] for a in pool.spawn_args] == [False]  # type: ignore[attr-defined]
        spec, layout, X, pairs = twist_batch()
        res = pool.evaluate("v", _twist_text(), layout, X, pairs)
    same(res, evaluate(spec, layout, X, pairs))


def _twist_text() -> str:
    from tests.test_build_failure import twist_lander

    return twist_lander(max_time=1.5)


def test_single_process_pool_never_freezes(monkeypatch: pytest.MonkeyPatch) -> None:
    """WorkerPool(0)(테스트·bench 단일 코어·워커 0개 설정)과 이 프로세스는 GC를 얼리지 않는다."""
    calls: list[int] = []
    monkeypatch.setattr(gc, "freeze", lambda: calls.append(1))
    spec, layout, X, pairs = twist_batch()
    with pool_mod.WorkerPool(0, window=3) as pool:
        for _ in range(3):
            res = pool.evaluate("v", _twist_text(), layout, X, pairs)
    assert calls == []
    same(res, evaluate(spec, layout, X, pairs))


def test_pool_window_reaches_the_workers() -> None:
    """WorkerPool(window=)가 작업에 실려 워커의 evaluate에 간다(결과는 창과 무관하므로 같음만 본다)."""
    spec, layout, X, pairs = twist_batch()
    text = _twist_text()
    ref = evaluate(spec, layout, X, pairs)
    for window in (None, 2):
        with pool_mod.WorkerPool(2, window=window) as pool:
            assert pool.window == window
            same(pool.evaluate("v", text, layout, X, pairs), ref)


class _InlinePool(pool_mod.WorkerPool):
    """워커를 띄우지 않고 작업(Task)을 이 프로세스에서 워커와 같은 함수(_run_task)로 돌리는 풀(이 테스트만): 풀이 만든 작업과
    워커가 evaluate에 넘기는 값을 본다."""

    def _spawn(self, w: int) -> None:
        pass

    def _run(self, tasks: list[Any]) -> list[EvalResult]:
        out = []
        for task in tasks:
            self.__dict__.setdefault("tasks", []).append(task)
            task_id, status, payload = pool_mod._run_task(task, {}, {})
            assert task_id == task.id and status == "ok", payload
            out.append(payload)
        return out


def test_pool_window_is_forwarded_to_the_worker_evaluate(monkeypatch: pytest.MonkeyPatch) -> None:
    """T16 창 × 워커: WorkerPool(window=)가 작업마다 Task.window로 실리고, 워커 쪽(_run_task)이 그 값을 evaluate의 window=로
    넘긴다(결과는 창과 무관해 같음만으로는 전달을 알 수 없으므로 넘긴 값을 직접 본다)."""
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")  # 워커 수가 있는 풀이 이 프로세스의 환경 변수를 바꾸므로 끝나면 되돌린다
    seen: list[Any] = []
    real = pool_mod.evaluate

    def recording(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("window", "missing"))
        return real(*args, **kwargs)

    monkeypatch.setattr(pool_mod, "evaluate", recording)
    spec, layout, X, pairs = twist_batch()
    text = _twist_text()
    ref = evaluate(spec, layout, X, pairs)
    for window in (None, 2, 5):
        seen.clear()
        pool = _InlinePool(2, window=window)
        res = pool.evaluate("v", text, layout, X, pairs)
        tasks = pool.__dict__["tasks"]
        assert len(tasks) > 1 and [t.window for t in tasks] == [window] * len(tasks)
        assert seen == [window] * len(tasks)
        same(res, ref)
    seen.clear()
    with pool_mod.WorkerPool(0, window=3) as single:  # 워커 0개: 이 프로세스의 evaluate에 바로
        single.evaluate("v", text, layout, X, pairs)
    assert seen == [3]

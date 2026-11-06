"""평가 워커 풀.

- 워커는 한 번 띄워 계속 쓴다(spawn, 윈도우와 같은 방식). 세대마다 프로세스를 새로 만들지 않는다.
- 작업을 워커마다 하나씩 주고, 끝난 워커에 다음 작업을 준다(에피소드 길이가 제각각이라 동적 분배).
- 워커는 환경 사본 내용(디스크의 사용자 파일이 아님)을 받아 버전별로 한 번만 컴파일해 둔다.
- 워커 프로세스가 죽으면 새 워커를 띄워 같은 작업을 다시 평가한다(시험 결과는 순수 함수라 결정성이 유지된다).
- 환경 오류(조립 실패 등)는 결정적인 오류이므로 PoolError로 올린다.
- 부모가 죽으면 워커도 스스로 끝난다. 워커는 Ctrl+C를 무시한다(부모가 정리한다).

결과는 작업을 어떻게 나눴는지와 무관하다: 쌍마다 독립이고, 요청한 순서로 다시 맞춘다.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from evolab.app import strings as S
from evolab.engine.evaluate import EvalResult, evaluate
from evolab.policy.genome import GenomeLayout
from evolab.world.physics2d.episode import Trial

POLL_S = 0.2
MAX_RETRIES = 2


class PoolError(RuntimeError):
    """평가를 끝낼 수 없다. message는 한국어."""


def physical_cores() -> int:
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
    except Exception:  # psutil이 없거나 실패하면 논리 프로세서로
        n = None
    return n or os.cpu_count() or 2


def default_workers() -> int:
    """물리 코어 - 1, 최소 1 (명세: 평가 워커)."""
    return max(1, physical_cores() - 1)


def limit_blas_threads() -> None:
    """워커끼리 코어를 나눠 쓰므로 BLAS 스레드는 1개. 자식 프로세스는 이 환경 변수를 물려받는다."""
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


@dataclass
class Task:
    id: int
    env_version: str
    env_text: str
    layout: dict[str, Any]
    genomes: np.ndarray
    pairs: list[tuple[int, int, int | None]]
    """(genomes 안 행 번호, 시험 시드, 트랙 번호)"""


def _lower_priority() -> None:
    if sys.platform == "win32":
        try:
            import psutil

            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:
            pass
    else:
        try:
            os.nice(5)
        except OSError:
            pass


def _worker_main(worker_id: int, inbox: Any, outbox: Any) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _lower_priority()
    from evolab.spec.io import compile_env

    specs: dict[str, Any] = {}
    layouts: dict[str, GenomeLayout] = {}
    parent = mp.parent_process()
    while True:
        try:
            task = inbox.get(timeout=0.5)
        except queue.Empty:
            if parent is not None and not parent.is_alive():
                return
            continue
        if task is None:
            return
        try:
            spec = specs.get(task.env_version)
            if spec is None:
                spec = compile_env(task.env_text)
                specs[task.env_version] = spec
            key = repr(sorted(task.layout.items()))
            layout = layouts.get(key)
            if layout is None:
                layout = GenomeLayout.from_dict(task.layout)
                layouts[key] = layout
            pairs = [(g, Trial(seed, track)) for g, seed, track in task.pairs]
            result = evaluate(spec, layout, task.genomes, pairs)
            outbox.put((worker_id, task.id, "ok", result))
        except Exception as exc:
            outbox.put((worker_id, task.id, "error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


class WorkerPool:
    """상주 워커 N개. workers=0이면 워커 없이 이 프로세스에서 평가한다(테스트·디버깅용, 결과는 같다)."""

    def __init__(self, workers: int | None = None, chunks_per_worker: int = 4) -> None:
        self.workers = default_workers() if workers is None else max(0, workers)
        self.chunks_per_worker = chunks_per_worker
        self._ctx = mp.get_context("spawn")
        self._procs: list[Any] = []
        self._inboxes: list[Any] = []
        self._outbox: Any = None
        self._next_task = 0
        if self.workers:
            limit_blas_threads()
            self._outbox = self._ctx.Queue()
            for w in range(self.workers):
                self._spawn(w)

    # ---- 프로세스 관리 ----
    def _spawn(self, w: int) -> None:
        inbox = self._ctx.Queue()
        proc = self._ctx.Process(target=_worker_main, args=(w, inbox, self._outbox), daemon=False)
        proc.start()
        if w < len(self._procs):
            self._procs[w] = proc
            self._inboxes[w] = inbox
        else:
            self._procs.append(proc)
            self._inboxes.append(inbox)

    def close(self) -> None:
        for inbox in self._inboxes:
            try:
                inbox.put(None)
            except (OSError, ValueError):
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        self._procs.clear()
        self._inboxes.clear()

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- 평가 ----
    def evaluate(
        self,
        env_version: str,
        env_text: str,
        layout: GenomeLayout,
        genomes: np.ndarray,
        pairs: Sequence[tuple[int, Trial]],
    ) -> EvalResult:
        """(유전자 행, Trial) 쌍들을 평가해 요청한 순서로 돌려준다."""
        genomes = np.ascontiguousarray(genomes, dtype=np.float64)
        if not self.workers:
            from evolab.spec.io import compile_env

            return evaluate(compile_env(env_text), layout, genomes, list(pairs))
        layout_d = layout.to_dict()
        chunks = self._split(len(pairs))
        tasks: list[Task] = []
        order: list[tuple[int, int]] = [(0, 0)] * len(pairs)
        for c, idx in enumerate(chunks):
            used = sorted({pairs[i][0] for i in idx})
            local = {g: k for k, g in enumerate(used)}
            task_pairs = []
            for r, i in enumerate(idx):
                g, trial = pairs[i]
                task_pairs.append((local[g], trial.seed, trial.track))
                order[i] = (c, r)
            self._next_task += 1
            tasks.append(Task(self._next_task, env_version, env_text, layout_d, genomes[used], task_pairs))
        parts = self._run(tasks)
        return EvalResult.concat(parts, order)

    def _split(self, n: int) -> list[list[int]]:
        k = max(1, min(n, self.workers * self.chunks_per_worker))
        size = -(-n // k)
        return [list(range(i, min(n, i + size))) for i in range(0, n, size)]

    def _run(self, tasks: list[Task]) -> list[EvalResult]:
        pending = list(range(len(tasks)))
        results: list[EvalResult | None] = [None] * len(tasks)
        by_id = {t.id: k for k, t in enumerate(tasks)}
        running: dict[int, int] = {}  # 워커 -> 작업 위치
        retries: dict[int, int] = {}
        done = 0

        def feed(w: int) -> None:
            if pending:
                k = pending.pop(0)
                running[w] = k
                self._inboxes[w].put(tasks[k])

        for w in range(self.workers):
            feed(w)
        while done < len(tasks):
            try:
                w, task_id, status, payload = self._outbox.get(timeout=POLL_S)
            except queue.Empty:
                self._recover(running, pending, retries, tasks, feed)
                continue
            k = by_id.get(task_id)
            if k is None or running.get(w) != k:
                continue  # 다시 띄우기 전에 죽은 워커가 남긴 늦은 결과
            del running[w]
            if status != "ok":
                raise PoolError(S.POOL_TASK_FAILED.format(detail=payload))
            results[k] = payload
            done += 1
            feed(w)
        return [r for r in results if r is not None]

    def _recover(self, running: dict[int, int], pending: list[int], retries: dict[int, int],
                 tasks: list[Task], feed: Any) -> None:
        for w, proc in enumerate(self._procs):
            if proc.is_alive():
                continue
            code = proc.exitcode
            k = running.pop(w, None)
            if k is not None:
                retries[k] = retries.get(k, 0) + 1
                if retries[k] > MAX_RETRIES:
                    raise PoolError(S.POOL_WORKER_DIED.format(code=code))
                pending.insert(0, k)
            self._spawn(w)
            feed(w)


def timed_evaluate(pool: WorkerPool, *args: Any) -> tuple[EvalResult, float]:
    t0 = time.perf_counter()
    res = pool.evaluate(*args)
    return res, time.perf_counter() - t0

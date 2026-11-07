"""평가 워커 풀.

- 워커는 한 번 띄워 계속 쓴다(spawn, 윈도우와 같은 방식). 세대마다 프로세스를 새로 만들지 않는다.
- 작업을 워커마다 하나씩 주고, 끝난 워커에 다음 작업을 준다(에피소드 길이가 제각각이라 동적 분배).
- 워커는 환경 사본 내용(디스크의 사용자 파일이 아님)을 받아 버전별로 한 번만 컴파일해 둔다.
- 워커 프로세스가 죽으면 새 워커를 띄워 같은 작업을 다시 평가한다(시험 결과는 순수 함수라 결정성이 유지된다).
- 환경 오류(조립 실패 등)는 결정적인 오류이므로 PoolError로 올린다.
- 워커마다 파이프 하나(공유 Queue 없음, WorkerPool 설명 참고). 부모가 죽으면 워커도 스스로 끝난다. 워커는 Ctrl+C를 무시한다(부모가 정리한다).

결과는 작업을 어떻게 나눴는지와 무관하다: 쌍마다 독립이고, 요청한 순서로 다시 맞춘다. 관찰 기록(v1.2)도 쌍마다
독립이라 같다(WorkerPool.observes: 학습기는 이 표시가 있는 평가기에만 observe=·frame=을 넘긴다).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.connection import wait
from typing import Any

import numpy as np

from evolab.app import strings as S
from evolab.engine.evaluate import EvalResult, evaluate
from evolab.policy.genome import GenomeLayout
from evolab.world.physics2d.episode import Trial

POLL_S = 0.2
MAX_RETRIES = 2
WAIT_CHUNK = 60
"""윈도우 WaitForMultipleObjects는 한 번에 63개까지만 기다린다(워커 30개를 넘으면 결과 파이프 + 프로세스 신호가 넘침)."""


def _wait(objs: list[Any], timeout: float) -> list[Any]:
    """multiprocessing.connection.wait를 핸들 수 제한 없이: 많으면 나눠서 먼저 훑고, 없으면 첫 묶음만 기다린다."""
    if len(objs) <= WAIT_CHUNK:
        return list(wait(objs, timeout=timeout))
    chunks = [objs[i:i + WAIT_CHUNK] for i in range(0, len(objs), WAIT_CHUNK)]
    ready: list[Any] = []
    for chunk in chunks:
        ready += wait(chunk, timeout=0)
    if ready:
        return ready
    return list(wait(chunks[0], timeout=timeout))


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
    record: list[bool] | None = None
    """쌍마다 궤적 기록 여부(None이면 기록 안 함)."""
    every: int = 2
    observe: list[int] | None = None
    """쌍마다 관찰 코드 0·1·2(evaluate.OBSERVE_*). None이면 관찰 안 함(v1.2)."""
    frame: tuple[float, float, float, int, int] | None = None
    """관찰 코드 2의 격자 틀(x0, y0, 칸 크기, nx, ny)."""


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


def _worker_main(worker_id: int, tasks: Any, results: Any) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _lower_priority()
    from evolab.spec.io import compile_env

    specs: dict[str, Any] = {}
    layouts: dict[str, GenomeLayout] = {}
    parent = mp.parent_process()
    while True:
        try:
            if not tasks.poll(0.5):
                if parent is not None and not parent.is_alive():
                    return
                continue
            task = tasks.recv()
        except (EOFError, OSError):  # 부모가 끝났다(파이프가 닫힘)
            return
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
            res = evaluate(spec, layout, task.genomes, pairs, task.record, task.every, task.observe, task.frame)
            msg: tuple[int, str, Any] = (task.id, "ok", res)
        except Exception as exc:
            msg = (task.id, "error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        try:
            results.send(msg)
        except (EOFError, OSError):
            return


_STOP = object()


class _Link:
    """워커 하나와 잇는 파이프 두 개(작업 보내기, 결과 받기)와 보내는 스레드.

    보내기는 스레드가 한다. 윈도우 파이프는 버퍼(8KB)보다 큰 메시지를 읽는 쪽이 읽을 때까지 쓰지 못하고, 읽는
    쪽이 죽었을 때도 쓰기가 끝나지 않는 경우가 있었다(CI에서 죽은 워커에 보내다 멈춤). 그래서 평가 루프는 보내기에서
    기다리지 않고, 워커가 죽으면 파이프를 닫는다(닫으면 걸려 있던 쓰기가 취소되어 스레드도 끝난다).
    """

    def __init__(self, task_conn: Any, result_conn: Any) -> None:
        self.task_conn = task_conn
        self.result_conn = result_conn
        self.outbox: queue.Queue[Any] = queue.Queue()
        self.thread = threading.Thread(target=self._send_loop, name="pool-send", daemon=True)
        self.thread.start()

    def _send_loop(self) -> None:
        while True:
            item = self.outbox.get()
            if item is _STOP:
                return
            try:
                self.task_conn.send(item)
            except (OSError, ValueError, EOFError, TypeError):
                return

    def send(self, item: Any) -> None:
        self.outbox.put(item)

    def close(self) -> None:
        self.outbox.put(_STOP)
        for c in (self.task_conn, self.result_conn):
            try:
                c.close()
            except OSError:
                pass


class WorkerPool:
    """상주 워커 N개. workers=0이면 워커 없이 이 프로세스에서 평가한다(테스트·디버깅용, 결과는 같다).

    워커마다 파이프 두 개를 쓴다(공유 Queue를 쓰지 않는다, _Link 참고). 공유 Queue는 보내기 스레드가 데이터를 다
    쓸 때까지 프로세스 종료를 붙잡아, 죽은 워커에 보낸 작업이 부모의 종료를 영원히 막았다. 죽은 워커의 파이프는
    버리므로 반쯤 쓰인 메시지나 늦은 결과가 다른 워커 결과에 섞이지 않는다.
    """

    observes = True
    """evaluate가 observe=·frame=을 받고 EvalResult.observations를 채운다(학습기가 getattr로 확인, 설계 D8)."""

    def __init__(self, workers: int | None = None, chunks_per_worker: int = 4) -> None:
        self.workers = default_workers() if workers is None else max(0, workers)
        self.chunks_per_worker = chunks_per_worker
        self._ctx = mp.get_context("spawn")
        self._procs: list[Any] = []
        self._links: list[_Link] = []
        self._next_task = 0
        self.last_busy = 0.0
        """마지막 evaluate에서 워커들이 계산에 쓴 시간의 합(초). 워커 사용률 = last_busy / (워커 수 × 걸린 시간)."""
        if self.workers:
            limit_blas_threads()
            for w in range(self.workers):
                self._spawn(w)

    # ---- 프로세스 관리 ----
    def _spawn(self, w: int) -> None:
        if w < len(self._links):
            self._links[w].close()
        task_r, task_w = self._ctx.Pipe(duplex=False)
        res_r, res_w = self._ctx.Pipe(duplex=False)
        proc = self._ctx.Process(target=_worker_main, args=(w, task_r, res_w), daemon=True)
        proc.start()
        task_r.close()  # 워커 쪽 끝은 워커만 가진다
        res_w.close()
        link = _Link(task_w, res_r)
        if w < len(self._procs):
            self._procs[w] = proc
            self._links[w] = link
        else:
            self._procs.append(proc)
            self._links.append(link)

    def close(self, timeout: float = 5.0) -> None:
        for link in self._links:
            link.send(None)
        deadline = time.monotonic() + timeout
        live = {p.sentinel: (p, link) for p, link in zip(self._procs, self._links, strict=True)}
        while live and time.monotonic() < deadline:
            # 일하던 워커가 보내는 결과는 받아서 버린다(안 받으면 윈도우에서 보내기가 끝나지 않는다).
            conns = {link.result_conn: s for s, (p, link) in live.items()}
            try:
                ready = _wait(list(conns) + list(live), timeout=0.1)
            except OSError:
                break
            for obj in ready:
                if obj in conns:
                    try:
                        obj.recv()
                    except (EOFError, OSError):
                        pass
                else:
                    live.pop(obj, None)
        for p in self._procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=timeout)
        for link in self._links:
            link.close()
        self._procs.clear()
        self._links.clear()

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
        record: Sequence[bool] | None = None,
        every: int = 2,
        observe: Sequence[int] | None = None,
        frame: Sequence[float] | None = None,
    ) -> EvalResult:
        """(유전자 행, Trial) 쌍들을 평가해 요청한 순서로 돌려준다. record[i]가 참인 쌍은 궤적도 돌려준다.

        observe[i]가 1·2인 쌍은 관찰(EvalResult.observations)도 돌려준다. frame은 코드 2의 격자 틀(x0, y0, 칸, nx, ny).
        """
        genomes = np.ascontiguousarray(genomes, dtype=np.float64)
        self.last_busy = 0.0
        frame_t = None if frame is None else (float(frame[0]), float(frame[1]), float(frame[2]), int(frame[3]),
                                              int(frame[4]))
        if not self.workers:
            from evolab.spec.io import compile_env

            t0 = time.perf_counter()
            obs = None if observe is None else [int(c) for c in observe]
            res = evaluate(compile_env(env_text), layout, genomes, list(pairs), record, every, obs, frame_t)
            self.last_busy = time.perf_counter() - t0
            return res
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
            rec = [bool(record[i]) for i in idx] if record is not None else None
            obs = [int(observe[i]) for i in idx] if observe is not None else None
            tasks.append(Task(self._next_task, env_version, env_text, layout_d, genomes[used], task_pairs,
                              rec if rec and any(rec) else None, every, obs if obs and any(obs) else None,
                              frame_t if obs and any(obs) else None))
        parts = self._run(tasks)
        self.last_busy = sum(p.nn_seconds + p.sim_seconds for p in parts)
        return EvalResult.concat(parts, order)

    def _split(self, n: int) -> list[list[int]]:
        if n <= 0:
            return []
        k = max(1, min(n, self.workers * self.chunks_per_worker))
        size = -(-n // k)
        return [list(range(i, min(n, i + size))) for i in range(0, n, size)]

    def _run(self, tasks: list[Task]) -> list[EvalResult]:
        pending = list(range(len(tasks)))
        results: list[EvalResult | None] = [None] * len(tasks)
        running: dict[int, int] = {}  # 워커 -> 작업 위치
        retries: dict[int, int] = {}
        done = 0

        def feed(w: int) -> None:
            if not pending or w in running or not self._procs[w].is_alive():
                return  # 죽은 워커에는 보내지 않는다(아래 복구가 다시 띄우고 준다)
            k = pending.pop(0)
            self._links[w].send(tasks[k])
            running[w] = k

        for w in range(self.workers):
            feed(w)
        while done < len(tasks):
            busy = {self._links[w].result_conn: w for w in running}
            for conn in _wait(list(busy) + [p.sentinel for p in self._procs], timeout=POLL_S):
                w = busy.get(conn) if not isinstance(conn, int) else None
                if w is None:
                    continue  # 프로세스 종료 신호: 아래 복구에서 처리
                try:
                    task_id, status, payload = conn.recv()
                except (EOFError, OSError):
                    continue  # 결과를 다 보내기 전에 죽었다
                k = running.get(w)
                if k is None or tasks[k].id != task_id:
                    continue
                del running[w]
                if status != "ok":
                    raise PoolError(S.POOL_TASK_FAILED.format(detail=payload))
                results[k] = payload
                done += 1
                feed(w)
            self._recover(running, pending, retries, feed)
        return [r for r in results if r is not None]

    def _recover(self, running: dict[int, int], pending: list[int], retries: dict[int, int], feed: Any) -> None:
        for w, proc in enumerate(self._procs):
            if proc.is_alive():
                if w not in running:
                    feed(w)
                continue
            code = proc.exitcode
            k = running.pop(w, None)
            if k is not None:
                retries[k] = retries.get(k, 0) + 1
                if retries[k] > MAX_RETRIES:
                    raise PoolError(S.POOL_WORKER_DIED.format(code=code))
                pending.insert(0, k)
            proc.join(timeout=1)
            self._spawn(w)
            feed(w)


def timed_evaluate(pool: WorkerPool, *args: Any) -> tuple[EvalResult, float]:
    t0 = time.perf_counter()
    res = pool.evaluate(*args)
    return res, time.perf_counter() - t0

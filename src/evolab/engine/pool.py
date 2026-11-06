"""평가 워커 풀.

- 워커는 한 번 띄워 계속 쓴다(spawn, 윈도우와 같은 방식). 세대마다 프로세스를 새로 만들지 않는다.
- 작업을 워커마다 하나씩 주고, 끝난 워커에 다음 작업을 준다(에피소드 길이가 제각각이라 동적 분배).
- 워커는 환경 사본 내용(디스크의 사용자 파일이 아님)을 받아 버전별로 한 번만 컴파일해 둔다.
- 워커 프로세스가 죽으면 새 워커를 띄워 같은 작업을 다시 평가한다(시험 결과는 순수 함수라 결정성이 유지된다).
- 환경 오류(조립 실패 등)는 결정적인 오류이므로 PoolError로 올린다.
- 워커마다 파이프 하나(공유 Queue 없음, WorkerPool 설명 참고). 부모가 죽으면 워커도 스스로 끝난다. 워커는 Ctrl+C를 무시한다(부모가 정리한다).

결과는 작업을 어떻게 나눴는지와 무관하다: 쌍마다 독립이고, 요청한 순서로 다시 맞춘다.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
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


def _worker_main(worker_id: int, conn: Any) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _lower_priority()
    from evolab.spec.io import compile_env

    specs: dict[str, Any] = {}
    layouts: dict[str, GenomeLayout] = {}
    parent = mp.parent_process()
    while True:
        try:
            if not conn.poll(0.5):
                if parent is not None and not parent.is_alive():
                    return
                continue
            task = conn.recv()
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
            msg: tuple[int, str, Any] = (task.id, "ok", evaluate(spec, layout, task.genomes, pairs))
        except Exception as exc:
            msg = (task.id, "error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        try:
            conn.send(msg)
        except (EOFError, OSError):
            return


class WorkerPool:
    """상주 워커 N개. workers=0이면 워커 없이 이 프로세스에서 평가한다(테스트·디버깅용, 결과는 같다).

    워커마다 양방향 파이프를 하나씩 쓴다(공유 Queue를 쓰지 않는다). 공유 Queue는 보내기 스레드가
    데이터를 다 쓸 때까지 프로세스 종료를 붙잡는데, 윈도우 파이프는 버퍼(8KB)를 넘는 메시지를 읽는
    쪽이 읽을 때까지 쓰지 못한다. 그래서 죽은 워커의 받은편지함에 남은 작업이 부모의 종료를 영원히
    막았다. 파이프는 이 프로세스에서 직접 보내고, 받는 쪽이 죽으면 바로 오류가 난다. 죽은 워커의
    파이프는 버리므로 반쯤 쓰인 메시지나 늦은 결과가 다른 워커 결과에 섞이지 않는다.
    """

    def __init__(self, workers: int | None = None, chunks_per_worker: int = 4) -> None:
        self.workers = default_workers() if workers is None else max(0, workers)
        self.chunks_per_worker = chunks_per_worker
        self._ctx = mp.get_context("spawn")
        self._procs: list[Any] = []
        self._conns: list[Any] = []
        self._next_task = 0
        if self.workers:
            limit_blas_threads()
            for w in range(self.workers):
                self._spawn(w)

    # ---- 프로세스 관리 ----
    def _spawn(self, w: int) -> None:
        if w < len(self._conns):
            self._conns[w].close()
        ours, theirs = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(target=_worker_main, args=(w, theirs), daemon=True)
        proc.start()
        theirs.close()  # 워커 쪽 끝은 워커만 가진다: 워커가 죽으면 이 파이프가 바로 끊긴다
        if w < len(self._procs):
            self._procs[w] = proc
            self._conns[w] = ours
        else:
            self._procs.append(proc)
            self._conns.append(ours)

    def close(self, timeout: float = 5.0) -> None:
        for conn in self._conns:
            try:
                conn.send(None)
            except (OSError, ValueError):
                pass
        deadline = time.monotonic() + timeout
        live = {p.sentinel: (p, c) for p, c in zip(self._procs, self._conns, strict=True)}
        while live and time.monotonic() < deadline:
            # 일하던 워커가 보내는 결과는 받아서 버린다(안 받으면 윈도우에서 보내기가 끝나지 않는다).
            conns = {c: s for s, (p, c) in live.items()}
            for obj in wait(list(conns) + list(live), timeout=0.1):
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
        for c in self._conns:
            c.close()
        self._procs.clear()
        self._conns.clear()

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
        running: dict[int, int] = {}  # 워커 -> 작업 위치
        retries: dict[int, int] = {}
        done = 0

        def feed(w: int) -> None:
            if not pending or w in running:
                return
            k = pending.pop(0)
            try:
                self._conns[w].send(tasks[k])
            except (OSError, ValueError):  # 워커가 죽었다: 아래 복구가 다시 띄우고 다시 준다
                pending.insert(0, k)
                return
            running[w] = k

        for w in range(self.workers):
            feed(w)
        while done < len(tasks):
            busy = {self._conns[w]: w for w in running}
            for conn in wait(list(busy) + [p.sentinel for p in self._procs], timeout=POLL_S):
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

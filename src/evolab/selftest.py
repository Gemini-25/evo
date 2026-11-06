"""설치 상태 자체 점검: `evolab selftest`.

exe로 묶은 뒤에만 드러나는 문제(중첩 spawn, 창이 없는 exe의 표준 출력, 한글 경로, Qt 플러그인)를
마일스톤마다 일찍 잡기 위한 점검이다. 학습 엔진의 실제 워커 풀은 M2에서 만들고, 여기서는 같은
프로세스 구조(화면 → 학습 프로세스 → 워커 여러 개, 모두 spawn)만 작게 재현한다.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import platform
import queue
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evolab import __version__, paths
from evolab.app import strings as S

PROCESS_TIMEOUT_S = 180.0
SELFTEST_SEEDS: tuple[int, ...] = tuple(range(1, 13))


class SelfTestError(Exception):
    """점검 실패. 메시지는 strings.py 문구로 만든다."""


@dataclass
class CheckResult:
    name: str
    ok: bool
    seconds: float
    detail: str


# ---------------------------------------------------------------------------
# 물리: 다리 두 개 달린 상자가 울퉁불퉁한 지형에 떨어지는 짧은 시뮬레이션
# ---------------------------------------------------------------------------


def simulate(seed: int, steps: int = 240) -> np.ndarray:
    """시드 하나로 짧은 접촉 시뮬레이션을 돌려 최종 상태를 돌려준다. 이력과 무관해야 한다."""
    import pymunk

    rng = np.random.default_rng(seed)
    space = pymunk.Space()
    space.gravity = (0.0, -9.81)

    xs = np.linspace(-20.0, 20.0, 41)
    ys = rng.uniform(-0.4, 0.4, size=xs.size)
    for i in range(xs.size - 1):
        seg = pymunk.Segment(space.static_body, (float(xs[i]), float(ys[i])), (float(xs[i + 1]), float(ys[i + 1])), 0.05)
        seg.friction = 0.9
        space.add(seg)

    mass, size = 4.0, (2.0, 1.0)
    body = pymunk.Body(mass, pymunk.moment_for_box(mass, size))
    body.position = (float(rng.uniform(-3, 3)), 4.0)
    body.angle = float(rng.uniform(-0.3, 0.3))
    body.velocity = (float(rng.uniform(-1, 1)), float(rng.uniform(-2, 0)))
    hull = pymunk.Poly.create_box(body, size)
    hull.friction = 0.9
    space.add(body, hull)

    bodies = [body]
    for side in (-1.0, 1.0):
        leg_mass, leg_size = 0.3, (0.15, 1.0)
        leg = pymunk.Body(leg_mass, pymunk.moment_for_box(leg_mass, leg_size))
        anchor = body.local_to_world((side * 0.9, -0.5))
        leg.position = anchor + pymunk.Vec2d(0, -0.5).rotated(body.angle)
        leg.angle = body.angle
        leg.velocity = body.velocity
        leg_shape = pymunk.Poly.create_box(leg, leg_size)
        leg_shape.friction = 1.2
        leg_shape.filter = pymunk.ShapeFilter(group=1)
        pivot = pymunk.PivotJoint(body, leg, anchor)
        spring = pymunk.DampedRotarySpring(body, leg, 0.0, 400.0, 30.0)
        space.add(leg, leg_shape, pivot, spring)
        bodies.append(leg)
    hull.filter = pymunk.ShapeFilter(group=1)

    for _ in range(steps):
        # 단순한 결정적 제어: 기울면 옆으로, 빨리 떨어지면 위로 민다.
        vy = body.velocity.y
        thrust = 30.0 if vy < -1.0 else 0.0
        body.apply_force_at_local_point((0.0, thrust), (0.0, 0.0))
        body.apply_force_at_local_point((-8.0 * body.angle, 0.0), (0.0, 0.5))
        space.step(1.0 / 60.0)

    state = [v for b in bodies for v in (b.position.x, b.position.y, b.angle, b.velocity.x, b.velocity.y, b.angular_velocity)]
    return np.asarray(state, dtype=np.float64)


# ---------------------------------------------------------------------------
# 프로세스 구조: 화면 프로세스 → 학습 프로세스 → 워커 N개 (모두 spawn, 워커는 재사용)
# ---------------------------------------------------------------------------


def _worker_main(tasks: Any, results: Any) -> None:
    while True:
        item = tasks.get()
        if item is None:
            return
        index, seed = item
        results.put((index, simulate(seed)))


class _MiniPool:
    """상주 워커 몇 개. 시험 순서와 워커 배분이 달라도 결과는 같아야 한다."""

    def __init__(self, ctx: Any, workers: int) -> None:
        self._tasks = ctx.Queue()
        self._results = ctx.Queue()
        self._procs = [ctx.Process(target=_worker_main, args=(self._tasks, self._results)) for _ in range(workers)]
        for p in self._procs:
            p.start()

    def map(self, seeds: Sequence[int]) -> np.ndarray:
        for index, seed in enumerate(seeds):
            self._tasks.put((index, seed))
        out: dict[int, np.ndarray] = {}
        while len(out) < len(seeds):
            index, value = self._results.get(timeout=PROCESS_TIMEOUT_S)
            out[index] = value
        return np.stack([out[i] for i in range(len(seeds))])

    def close(self) -> None:
        for _ in self._procs:
            self._tasks.put(None)
        for p in self._procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


def _trainer_main(out: Any, workers: int, seeds: Sequence[int]) -> None:
    try:
        ctx = mp.get_context("spawn")
        results: dict[int, np.ndarray] = {}
        for n in sorted({1, workers}):
            pool = _MiniPool(ctx, n)
            try:
                # 같은 풀로 두 번 돌려서 워커 재사용(이력)도 결과에 영향이 없는지 본다.
                first = pool.map(seeds)
                second = pool.map(list(reversed(seeds)))[::-1]
            finally:
                pool.close()
            if first.tobytes() != second.tobytes():
                raise SelfTestError(S.SELFTEST_ERR_MISMATCH.format(detail=f"reuse, workers={n}"))
            results[n] = first
        out.put(("ok", results, os.getpid()))
    except BaseException:
        out.put(("error", traceback.format_exc(), os.getpid()))


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


# ---------------------------------------------------------------------------
# 점검 항목
# ---------------------------------------------------------------------------


def _check_imports() -> str:
    import pydantic
    import pymunk
    import pyqtgraph
    import PySide6

    from evolab.learners.cma_import import import_cma

    cma = import_cma()
    versions = {
        "numpy": np.__version__,
        "pymunk": pymunk.version,
        "cma": cma.__version__,
        "pydantic": pydantic.VERSION,
        "PySide6": PySide6.__version__,
        "pyqtgraph": pyqtgraph.__version__,
    }
    return ", ".join(f"{k} {v}" for k, v in versions.items())


def _check_physics() -> str:
    a = np.stack([simulate(s) for s in SELFTEST_SEEDS])
    b = np.stack([simulate(s) for s in reversed(SELFTEST_SEEDS)])[::-1]
    if a.tobytes() != b.tobytes():
        raise SelfTestError(S.SELFTEST_ERR_MISMATCH.format(detail="same seed, same process"))
    return f"seeds={len(SELFTEST_SEEDS)}, checksum={_checksum(a)}"


def _check_processes(workers: int) -> str:
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    trainer = ctx.Process(target=_trainer_main, args=(out, workers, list(SELFTEST_SEEDS)))
    trainer.start()
    try:
        try:
            status, payload, pid = out.get(timeout=PROCESS_TIMEOUT_S)
        except queue.Empty as exc:
            raise SelfTestError(S.SELFTEST_ERR_TIMEOUT.format(seconds=int(PROCESS_TIMEOUT_S))) from exc
    finally:
        trainer.join(timeout=30)
        if trainer.is_alive():
            trainer.terminate()
            trainer.join(timeout=5)
    if status != "ok":
        raise SelfTestError(str(payload))

    local = np.stack([simulate(s) for s in SELFTEST_SEEDS])
    for n, arr in payload.items():
        if arr.tobytes() != local.tobytes():
            raise SelfTestError(S.SELFTEST_ERR_MISMATCH.format(detail=f"workers={n} vs in-process"))
    counts = "/".join(str(n) for n in sorted(payload))
    return f"trainer pid={pid}, workers={counts}, checksum={_checksum(local)}"


def _check_learner() -> str:
    from evolab.learners.cma_import import import_cma

    cma = import_cma()

    def sphere(x: np.ndarray) -> float:
        return float(np.sum((x - 1.0) ** 2))

    def make() -> Any:
        rng = np.random.RandomState(12345)
        opts = {"seed": np.nan, "randn": rng.randn, "popsize": 12, "verbose": -9, "verb_log": 0, "verb_disp": 0}
        return cma.CMAEvolutionStrategy(np.zeros(10), 0.5, opts)

    def run(es: Any, gens: int) -> None:
        for _ in range(gens):
            xs = es.ask()
            es.tell(xs, [sphere(np.asarray(x)) for x in xs])

    straight = make()
    run(straight, 40)

    resumed = make()
    run(resumed, 20)
    resumed = pickle.loads(pickle.dumps(resumed))
    run(resumed, 20)

    if np.asarray(straight.mean).tobytes() != np.asarray(resumed.mean).tobytes():
        raise SelfTestError(S.SELFTEST_ERR_MISMATCH.format(detail="cma resume"))
    start, best = sphere(np.zeros(10)), float(straight.result.fbest)
    if not best < start * 0.05:
        raise SelfTestError(S.SELFTEST_ERR_NO_PROGRESS.format(detail=f"{start:.3g} -> {best:.3g}"))
    return f"sphere {start:.3g} -> {best:.3g}, resume identical"


def _check_gui() -> str:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from evolab.app.main import MainWindow, create_application

    app = create_application()
    window = MainWindow()
    window.show()
    app.processEvents()
    title = window.windowTitle()
    tabs = window.tabs.count()
    window.close()
    app.processEvents()
    if title != S.APP_TITLE:
        raise SelfTestError(S.SELFTEST_ERR_TITLE.format(title=title))
    return f"platform={app.platformName()}, tabs={tabs}"


def _checksum(arr: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------


def environment_info(workers: int) -> dict[str, str]:
    return {
        "evolab": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": str(os.cpu_count()),
        "workers": str(workers),
        "frozen": str(bool(getattr(sys, "frozen", False))),
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }


def run_checks(workers: int, check_gui: bool) -> list[CheckResult]:
    checks: list[tuple[str, Callable[[], str]]] = [
        (S.SELFTEST_CHECK_IMPORTS, _check_imports),
        (S.SELFTEST_CHECK_PHYSICS, _check_physics),
        (S.SELFTEST_CHECK_LEARNER, _check_learner),
        (S.SELFTEST_CHECK_PROCESSES, lambda: _check_processes(workers)),
    ]
    if check_gui:
        checks.append((S.SELFTEST_CHECK_GUI, _check_gui))

    results: list[CheckResult] = []
    for name, fn in checks:
        t0 = time.perf_counter()
        try:
            detail, ok = fn(), True
        except SelfTestError as exc:
            detail, ok = str(exc), False
        except Exception:
            detail, ok = traceback.format_exc(), False
        results.append(CheckResult(name, ok, time.perf_counter() - t0, detail))
    return results


def format_report(info: dict[str, str], results: list[CheckResult]) -> str:
    lines = [S.SELFTEST_TITLE, ""]
    lines += [f"{k}: {v}" for k, v in info.items()]
    lines.append("")
    for r in results:
        mark = S.SELFTEST_PASS if r.ok else S.SELFTEST_FAIL
        lines.append(f"[{mark}] {r.name} ({r.seconds:.2f}s)")
        lines += [f"    {line}" for line in r.detail.splitlines()]
    failed = [r.name for r in results if not r.ok]
    lines.append("")
    lines.append(S.SELFTEST_SUMMARY_FAIL.format(names=", ".join(failed)) if failed else S.SELFTEST_SUMMARY_OK)
    return "\n".join(lines) + "\n"


def run_selftest(log_path: Path | None = None, workers: int | None = None, check_gui: bool = True) -> int:
    n = workers if workers is not None else default_workers()
    info = environment_info(n)
    results = run_checks(n, check_gui)
    report = format_report(info, results)

    path = log_path if log_path is not None else paths.log_dir() / "selftest.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")

    if sys.stdout is not None:
        print(report, end="")
        print(S.SELFTEST_LOG_WRITTEN.format(path=path))
    return 0 if all(r.ok for r in results) else 1

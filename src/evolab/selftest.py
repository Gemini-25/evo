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


POLL_S = 0.5


def _parent_alive() -> bool:
    parent = mp.parent_process()
    return parent is None or parent.is_alive()


def _worker_main(tasks: Any, results: Any) -> None:
    while True:
        try:
            item = tasks.get(timeout=POLL_S)
        except queue.Empty:
            # 부모(학습 프로세스)가 죽었으면 고아로 남지 않고 끝낸다.
            if not _parent_alive():
                return
            continue
        if item is None:
            return
        index, seed = item
        try:
            results.put(("ok", index, simulate(seed)))
        except BaseException:
            results.put(("error", index, traceback.format_exc()))


def _died(codes: list[int | None]) -> str:
    return S.SELFTEST_ERR_PROCESS_DIED.format(codes=codes, log=paths.log_dir() / "stderr.log")


def _dead(procs: Sequence[Any]) -> list[int | None]:
    return [p.exitcode for p in procs if not p.is_alive()]


class _MiniPool:
    """상주 워커 몇 개. 시험 순서와 워커 배분이 달라도 결과는 같아야 한다."""

    def __init__(self, ctx: Any, workers: int, target: Callable[[Any, Any], None] = _worker_main) -> None:
        self._tasks = ctx.Queue()
        self._results = ctx.Queue()
        self._procs = [ctx.Process(target=target, args=(self._tasks, self._results)) for _ in range(workers)]
        for p in self._procs:
            p.start()

    def map(self, seeds: Sequence[int]) -> np.ndarray:
        for index, seed in enumerate(seeds):
            self._tasks.put((index, seed))
        out: dict[int, np.ndarray] = {}
        deadline = time.monotonic() + PROCESS_TIMEOUT_S
        while len(out) < len(seeds):
            try:
                status, index, value = self._results.get(timeout=POLL_S)
            except queue.Empty:
                # 워커가 일찍 죽었으면 시간 초과까지 기다리지 않고 바로 알린다.
                if dead := _dead(self._procs):
                    raise SelfTestError(_died(dead)) from None
                if not _parent_alive() or time.monotonic() > deadline:
                    raise SelfTestError(S.SELFTEST_ERR_TIMEOUT.format(seconds=int(PROCESS_TIMEOUT_S))) from None
                continue
            if status != "ok":
                raise SelfTestError(str(value))
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
    # 물리 코어 기준 워커 수는 M2의 워커 풀에서 정한다. 자체 점검은 논리 프로세서 기준이면 충분하다.
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
    deadline = time.monotonic() + PROCESS_TIMEOUT_S
    try:
        while True:
            try:
                status, payload, pid = out.get(timeout=POLL_S)
                break
            except queue.Empty:
                if not trainer.is_alive():
                    raise SelfTestError(_died([trainer.exitcode])) from None
                if time.monotonic() > deadline:
                    raise SelfTestError(S.SELFTEST_ERR_TIMEOUT.format(seconds=int(PROCESS_TIMEOUT_S))) from None
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


def _check_builtin_env() -> str:
    """exe 안에서도 내장 예제를 찾고, 같은 시드로 두 번 돌린 결과가 같은지 본다."""
    from evolab.spec.io import load_env
    from evolab.world.physics2d.episode import Episode, Trial

    path = paths.builtin_env_dir() / "lander.evoenv"
    spec = load_env(path)

    def run() -> tuple[float, bytes]:
        ep = Episode(spec, None, Trial(12345))
        for _ in range(120):
            ep.step([0.6, 0.0, 0.1])
        return ep.result().score, ep.snapshot().poses.tobytes()

    a, b = run(), run()
    if a != b:
        raise SelfTestError(S.SELFTEST_ERR_MISMATCH.format(detail="lander replay"))
    return f"{path.name}: inputs={len(spec.inputs)}, outputs={len(spec.outputs)}, checksum={_checksum(np.frombuffer(a[1]))}"


REQUIRED_EXAMPLES: tuple[str, ...] = ("lander.evoenv", "track_car.evoenv", "hill_car.evoenv", "jump.evoenv", "glider.evoenv",
                                      "wheels.evoenv")
"""자체 점검이 있어야 한다고 보는 내장 예제(v2.0: 점프 게임, 글라이더, v2.1: 바퀴 고르는 언덕차 추가)."""


def _check_bundle_files() -> str:
    """내장 예제(착륙선, 트랙 자동차, 언덕차, 점프 게임, 글라이더, 바퀴 고르는 언덕차)를 모두 컴파일하고, v2 예제를 같은 시드로 두 번 돌려
    같은지 보고(_check_v2_examples), exe면 아이콘과 라이선스 고지가 있는지 본다."""
    import sys

    from evolab.spec.io import load_env

    names = [p.name for p in paths.builtin_envs()]
    for name in names:
        load_env(paths.builtin_env_dir() / name)
    need = set(REQUIRED_EXAMPLES)
    if not need <= set(names):
        raise SelfTestError(S.SELFTEST_ERR_MISSING.format(what=", ".join(sorted(need - set(names)))))
    parts = [f"examples={','.join(names)}", _check_v2_examples()]
    if getattr(sys, "frozen", False):
        for what in (paths.icon_path(), paths.bundle_dir("licenses") / "THIRD_PARTY_NOTICES.txt"):
            if not what.is_file():
                raise SelfTestError(S.SELFTEST_ERR_MISSING.format(what=what))
        parts.append("icon, licenses ok")
    return "; ".join(parts)


def _check_v2_examples() -> str:
    """v2 내장 예제를 같은 시드로 두 번 돌려 같은지 본다: 점프 게임(정해진 움직임 장애물 줄, 점프 추진기),
    글라이더(공기역학, 날개 모양, 모양별 밀도, 뇌 없음), (v2.1) 바퀴 고르는 언덕차(있음 칸: 바퀴 자리 넷 중 둘을 켠 몸이라
    없는 바퀴의 관절·모터·센서를 빼는 길을 지난다). exe 안에서도 v2 부품이 결정적인지 확인한다."""
    from evolab.spec.io import load_env
    from evolab.world.physics2d.episode import Episode, Trial

    parts = []
    for name, steps in (("jump.evoenv", 600), ("glider.evoenv", 600), ("wheels.evoenv", 600)):
        path = paths.builtin_env_dir() / name
        spec = load_env(path)
        body = _two_wheel_body(spec) if name == "wheels.evoenv" else None

        def run() -> tuple[float, int, bytes]:
            ep = Episode(spec, body, Trial(12345))
            n_out = len(spec.outputs)
            for k in range(steps):
                if ep.done:
                    break
                ep.step([1.0 if k % 50 < 3 else 0.0] * n_out)  # 점프 게임: 가끔 뛴다. 글라이더: 출력 없음
            obs = ep.observe()
            return ep.result().score, ep.steps, ep.snapshot().poses.tobytes() + obs.tobytes()

        a, b = run(), run()
        if a != b:
            raise SelfTestError(S.SELFTEST_ERR_MISMATCH.format(detail=f"{name} replay"))
        parts.append(f"{name}: steps={a[1]}, checksum={_checksum(np.frombuffer(a[2], dtype=np.uint8))}")
    return "; ".join(parts)


def _two_wheel_body(spec: Any) -> np.ndarray:
    """바퀴 고르는 언덕차의 처음 몸에서 바퀴 자리 1·4를 켜고 2·3을 끈 몸 수치(몸 유전자 순서)."""
    from evolab.spec import symbols as sy

    on = {"w1_on": 0.75, "w2_on": 0.25, "w3_on": 0.25, "w4_on": 0.75}
    return np.array([on.get(p.id, p.value) for p in sy.evolved_params(spec.file)], dtype=np.float64)


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


class _Report:
    """점검 기록. 줄마다 바로 파일에 쓴다. 도중에 멈추거나 죽어도 어디까지 왔는지 남게 하려고."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("w", encoding="utf-8")

    def line(self, text: str = "") -> None:
        self._file.write(text + "\n")
        self._file.flush()
        if sys.stdout is not None:
            print(text, flush=True)

    def close(self) -> None:
        self._file.close()


def _run_check(name: str, fn: Callable[[], str]) -> CheckResult:
    t0 = time.perf_counter()
    try:
        detail, ok = fn(), True
    except SelfTestError as exc:
        detail, ok = str(exc), False
    except Exception:
        detail, ok = traceback.format_exc(), False
    return CheckResult(name, ok, time.perf_counter() - t0, detail)


def run_selftest(log_path: Path | None = None, workers: int | None = None, check_gui: bool = True) -> int:
    n = workers if workers is not None else default_workers()
    checks: list[tuple[str, Callable[[], str]]] = [
        (S.SELFTEST_CHECK_IMPORTS, _check_imports),
        (S.SELFTEST_CHECK_PHYSICS, _check_physics),
        (S.SELFTEST_CHECK_LEARNER, _check_learner),
        (S.SELFTEST_CHECK_PROCESSES, lambda: _check_processes(n)),
        (S.SELFTEST_CHECK_ENV, _check_builtin_env),
        (S.SELFTEST_CHECK_BUNDLE, _check_bundle_files),
    ]
    if check_gui:
        checks.append((S.SELFTEST_CHECK_GUI, _check_gui))

    report = _Report(log_path if log_path is not None else paths.log_dir() / "selftest.log")
    failed: list[str] = [name for name, _ in checks]
    try:
        report.line(S.SELFTEST_TITLE)
        report.line()
        for key, value in environment_info(n).items():
            report.line(f"{key}: {value}")
        report.line()

        results: list[CheckResult] = []
        for name, fn in checks:
            report.line(f"[{S.SELFTEST_RUNNING}] {name}")
            r = _run_check(name, fn)
            results.append(r)
            report.line(f"[{S.SELFTEST_PASS if r.ok else S.SELFTEST_FAIL}] {r.name} ({r.seconds:.2f}s)")
            for detail_line in r.detail.splitlines():
                report.line(f"    {detail_line}")

        failed = [r.name for r in results if not r.ok]
        report.line()
        report.line(S.SELFTEST_SUMMARY_FAIL.format(names=", ".join(failed)) if failed else S.SELFTEST_SUMMARY_OK)
    finally:
        report.close()
    if sys.stdout is not None:
        print(S.SELFTEST_LOG_WRITTEN.format(path=report.path))
    return 0 if not failed else 1

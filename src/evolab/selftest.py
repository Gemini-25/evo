"""설치 상태 자체 점검: `evolab selftest`.

exe로 묶은 뒤에만 드러나는 문제(중첩 spawn, 창이 없는 exe의 표준 출력, 한글 경로, Qt 플러그인)를
마일스톤마다 일찍 잡기 위한 점검이다. 학습 엔진의 실제 워커 풀은 M2에서 만들고, 여기서는 같은
프로세스 구조(화면 → 학습 프로세스 → 워커 여러 개, 모두 spawn)만 작게 재현한다.

v3.0 점검 넷(설계 4.11, D16, D25): 도움말 쪽, 기록 폴더 쓰기와 오류 보고 묶음(EVOLAB_SELFTEST_CRASH=1이면 exe 자식의 일부러
세그폴트가 faults.log에 남는지도), Qt 한국어 번역, 처음 실행 안내 자동 진행(창 → 1세대 재생 초, `--tour full`이면 → 10세대).
새 점검은 데이터·상태 폴더를 임시 폴더로 바꿔(EVOLAB_DATA_DIR·EVOLAB_STATE_DIR, 설계 D5) 사용자의 문서 폴더·settings.ini를
건드리지 않고, 끝나면 지운다.

v3.1 점검 하나(설계 D59): 「내보내기」 — 착륙선 내장 예제를 영 유전자로 다시 시뮬레이션한 앞 2초를 내보내기 자식 프로세스(화면의
「내보내기…」와 같은 spawn 자식, 묶인 exe에서는 같은 exe)로 내장 MP4·폰 재생 파일로 만들어 상자 구조·프레임 수·자료 왕복을
확인하고, Media Foundation 시험 결과를 정보로 적는다(없어도 통과). 창이 필요 없어 --no-gui에서도 돈다.

v6.0 점검 하나(v6 설계 D19, 5.6): 「빠른 계산 길」 — 내장 예제 여섯의 같은 평가를 물리 엔진 C 함수를 바로 부르는 길과 보통 길
(공개 pymunk API)로 두 번 돌려 결과가 바이트까지 같은지 본다. 길이 꺼져 있으면(이 pymunk 판에 내부 함수가 없음) 안내하고
통과하되, CI(EVOLAB_SELFTEST_REQUIRE_FASTPATH=1 또는 GitHub Actions)에서는 실패로 본다. 창이 필요 없어 --no-gui에서도 돈다.

v6.1 점검 하나(v6 설계 5.6·8.4 W2): 「자라는 뇌」 — 착륙선 틀에서 자라는 뇌 학습 방법(learners/grow.py)을 세대마다 자라게 하며
자라기 직전·직후 평균을 같은 시험으로 평가해 모든 결과 칸이 바이트까지 같은지(함수 보존), 꺼진 좌표가 정확히 +0.0인지, 틀까지
다 자라는지 보고, 같은 설정의 짧은 학습(Trainer, 이 프로세스 안 평가) 두 번이 같은지 본다. 창이 필요 없어 --no-gui에서도 돈다.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import pickle
import platform
import queue
import shutil
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from evolab import __version__, paths
from evolab.app import strings as S

PROCESS_TIMEOUT_S = 180.0
SELFTEST_SEEDS: tuple[int, ...] = tuple(range(1, 13))
TOUR_MODES: tuple[str, ...] = ("gen1", "full")
TOUR_LIMIT_S: dict[str, float] = {"gen1": 60.0, "full": 150.0}
"""처음 실행 안내 기준(설계 9.3·D16): 창 → 1세대 재생 60초, 창 → 10세대 150초."""
TOUR_WAIT_S: dict[str, float] = {"gen1": 240.0, "full": 480.0}
"""안내가 끝나기를 기다리는 한도(기준보다 길게: 느린 PC에서 기준을 넘은 초를 적으려고)."""
CRASH_TIMEOUT_S = 120.0


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


# ---------------------------------------------------------------------------
# v6.0 점검: 빠른 계산 길 (물리 엔진 C 함수를 바로 부르는 길과 보통 길의 결과 비교)
# ---------------------------------------------------------------------------


FASTPATH_REQUIRED_ENV: Final = "EVOLAB_SELFTEST_REQUIRE_FASTPATH"
"""1이면 「빠른 계산 길」이 꺼진 것도 실패로 본다. GitHub Actions 안(GITHUB_ACTIONS=true)에서도 그렇다(v6 설계 8.6: 꺼지면
사용자에게는 안내, CI에서는 실패. 묶은 exe에 pymunk._chipmunk_cffi가 빠지면 W2가 잡는다)."""
FASTPATH_GENOMES: Final = 3
"""내장 예제마다 처음 유전자 둘레의 무작위 유전자 수(학습 시험 2개씩, 3쌍마다 궤적, 2쌍마다 행동 관찰)."""
FASTPATH_TRIALS: Final = 2
FASTPATH_SEED: Final = 6


def fastpath_required() -> bool:
    return (os.environ.get(FASTPATH_REQUIRED_ENV, "").strip() == "1"
            or os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true")


def _eval_fingerprint(res: Any) -> list[tuple[str, bytes]]:
    """EvalResult의 결과 칸 전부(점수·스텝·블록·가중 블록·행동 평균·종료 사건·결과·궤적·관찰)를 칸 이름과 바이트로.
    걸린 시간(nn_seconds·sim_seconds)만 뺀다."""
    out = [(name, getattr(res, name).tobytes()) for name in ("scores", "steps", "blocks", "action_abs", "weighted")]
    out += [("end_events", repr(res.end_events).encode()), ("outcomes", repr(res.outcomes).encode())]
    for k, tr in enumerate(res.trajectories or []):
        if tr is not None:
            meta = repr((tr.every, tr.dt, tr.end_step)).encode()
            out.append((f"trajectory {k}", tr.poses.tobytes() + tr.outputs.tobytes() + meta))
    for k, ob in enumerate(res.observations or []):
        if ob is not None:
            arrays = b"".join(getattr(ob, n).tobytes() for n in ("hist", "out_sum", "cells", "counts"))
            out.append((f"observation {k}", arrays + repr((ob.steps, ob.outside)).encode()))
    return out


def _fastpath_in_use(spec: Any) -> bool:
    """이 환경의 Episode가 정말 C 함수 길을 쓰는가(episode.fastpath_in_use: 모든 물체의 cpBody 포인터, 둘째 스텝부터 cpSpaceStep,
    모터가 있으면 모터 포인터). _CP_FAST가 참이어도 pymunk 내부 이름이 바뀐 판이면 Episode가 공개 API 길로 돌아간다(결과는 같고
    느림). bench 「pymunk C 함수 바로 부르기」 줄과 같은 판단이다."""
    from evolab.world.physics2d.episode import Trial, fastpath_in_use

    return fastpath_in_use(spec, Trial(FASTPATH_SEED))


def _check_fastpath() -> str:
    """「빠른 계산 길」(v6 설계 D19, 5.6): 내장 예제 여섯마다 같은 평가(무작위 유전자 × 학습 시험, 궤적·관찰 섞음)를 C 함수
    길과 보통 길(공개 pymunk API, Episode를 만들 때 _CP_FAST를 거짓으로)로 두 번 돌려 모든 결과 칸이 바이트까지 같은지 본다.
    길이 꺼져 있으면 안내만 하고 통과한다(결과는 같고 느릴 뿐). CI(fastpath_required)에서는 꺼짐도 실패."""
    import pymunk

    from evolab.engine.evaluate import evaluate
    from evolab.engine.trainer import training_trials
    from evolab.policy.genome import layout_for
    from evolab.spec.io import load_env
    from evolab.world.physics2d import episode as episode_mod

    specs = [(name, load_env(paths.builtin_env_dir() / name)) for name in REQUIRED_EXAMPLES]
    if not episode_mod._CP_FAST or not all(_fastpath_in_use(spec) for _name, spec in specs):
        if fastpath_required():
            raise SelfTestError(S.SELFTEST_FASTPATH_OFF)
        return f"{S.SELFTEST_FASTPATH_OFF}\npymunk {pymunk.version}"
    lines = [S.SELFTEST_FASTPATH_SAME, f"pymunk {pymunk.version}"]
    for name, spec in specs:
        layout = layout_for(spec)
        rng = np.random.default_rng(FASTPATH_SEED)
        genomes = layout.initial()[None, :] + 0.5 * rng.standard_normal((FASTPATH_GENOMES, layout.dim))
        trials = training_trials(spec, FASTPATH_SEED, 0, FASTPATH_TRIALS)[:FASTPATH_TRIALS]
        pairs = [(i, t) for i in range(FASTPATH_GENOMES) for t in trials]
        record = [k % 3 == 0 for k in range(len(pairs))]
        observe = [k % 2 for k in range(len(pairs))]
        fast = evaluate(spec, layout, genomes, pairs, record=record, observe=observe)
        episode_mod._CP_FAST = False
        try:
            public = evaluate(spec, layout, genomes, pairs, record=record, observe=observe)
        finally:
            episode_mod._CP_FAST = True
        a, b = _eval_fingerprint(fast), _eval_fingerprint(public)
        diff = [na for (na, va), (nb, vb) in zip(a, b, strict=False) if na != nb or va != vb]
        if diff or len(a) != len(b):
            raise SelfTestError(S.SELFTEST_FASTPATH_DIFF.format(detail=f"{name}: {', '.join(diff) or 'fields'}"))
        lines.append(f"{name}: pairs={len(pairs)}, steps={int(fast.steps.sum())}, "
                     f"checksum={_checksum(np.frombuffer(b''.join(v for _n, v in a), dtype=np.uint8))}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# v6.1 점검: 자라는 뇌 (자라기 직전·직후 뇌가 같은 결과를 내는지, 짧은 학습 두 번이 같은지)
# ---------------------------------------------------------------------------


GROW_CHECK_ENV: Final = "lander.evoenv"
GROW_CHECK_SEED: Final = 6
GROW_CHECK_POPSIZE: Final = 6
GROW_CHECK_TRIALS: Final = 2
GROW_CHECK_GENERATIONS: Final = 4
"""짧은 학습 두 번의 세대 수(간격 1이라 1·2·3세대 끝에 자라 2 → 4 → 8 → 16, 4세대는 다 자란 뒤)."""


def _grow_learner_check(spec: Any, layout: Any) -> tuple[list[int], int, str]:
    """세대마다 자라는 학습 방법(간격 1, 가짜 점수)으로 틀까지 자라며 자람마다 직전·직후 평균을 같은 시험(궤적 포함)으로 평가해
    비교한다. 직전 평균 = 직후 평균의 옛 켜진 좌표만(자람은 새 뉴런의 들어오는 좌표만 채운다). (켜진 수 경로, 쌍 수, 체크섬)."""
    from evolab.engine.evaluate import evaluate
    from evolab.engine.trainer import training_trials
    from evolab.learners.cmaes import CMAESConfig
    from evolab.learners.grow import GrowConfig, GrowingCMAES

    learner = GrowingCMAES(layout.initial(), layout, CMAESConfig(popsize=GROW_CHECK_POPSIZE, seed=GROW_CHECK_SEED,
                                                                 bounded_dims=layout.n_body), GrowConfig(every=1))
    frame = learner.frame
    trials = training_trials(spec, GROW_CHECK_SEED, 0, GROW_CHECK_TRIALS)[:GROW_CHECK_TRIALS]
    pairs = [(0, t) for t in trials]
    record = [True] * len(pairs)
    path = [learner.counts[0]]
    target = np.linspace(-0.3, 0.3, layout.dim)
    parts: list[bytes] = []
    for _ in range(4 * len(frame.sizes) + 8):
        if learner.counts == frame.sizes:
            break
        old = learner.counts
        X = learner.ask()
        outside = np.ones(layout.dim, dtype=bool)
        outside[frame.active_coords(old)] = False
        if np.any(X[:, outside] != 0.0) or np.any(np.signbit(X[:, outside])):
            raise SelfTestError(S.SELFTEST_GROW_DIFF.format(detail="candidates outside the active coordinates"))
        learner.tell(X, -np.sum((X - target) ** 2, axis=1))
        if learner.last_growth is None:
            continue
        after = learner.best()
        before = np.zeros(layout.dim)
        keep = frame.active_coords(old)
        before[keep] = after[keep]
        if np.array_equal(before, after):
            raise SelfTestError(S.SELFTEST_GROW_DIFF.format(detail="growth did not add incoming weights"))
        a = _eval_fingerprint(evaluate(spec, layout, before[None, :], pairs, record=record))
        b = _eval_fingerprint(evaluate(spec, layout, after[None, :], pairs, record=record))
        diff = [na for (na, va), (nb, vb) in zip(a, b, strict=False) if na != nb or va != vb]
        if diff or len(a) != len(b):
            raise SelfTestError(S.SELFTEST_GROW_DIFF.format(
                detail=f"{GROW_CHECK_ENV} {old} -> {learner.counts}: {', '.join(diff) or 'fields'}"))
        parts.extend(v for _n, v in a)
        path.append(learner.counts[0])
    if learner.counts != frame.sizes:
        raise SelfTestError(S.SELFTEST_GROW_DIFF.format(detail=f"counts {learner.counts} != frame {frame.sizes}"))
    return path, len(pairs), _checksum(np.frombuffer(b"".join(parts), dtype=np.uint8))


def _grow_training_run(text: str) -> tuple[list[dict[str, str]], bytes, int]:
    """짧은 자라는 뇌 학습 한 번(이 프로세스 안 평가, 저장소 없음): (통계 줄(시간 열 빼고, 글자로), 챔피언 바이트, 자란 수)."""
    from evolab.engine.pool import WorkerPool
    from evolab.engine.trainer import TIME_COLUMNS, TrainConfig, Trainer

    config = TrainConfig(method="grow", run_seed=GROW_CHECK_SEED, popsize=GROW_CHECK_POPSIZE, trials=1, grow_every=1,
                         validate_every=GROW_CHECK_POPSIZE, validation_seeds=2, validation_candidates=1,
                         max_generations=GROW_CHECK_GENERATIONS)
    with WorkerPool(0) as pool:
        trainer = Trainer(text, config, pool, None, observe=False)
        rows = [trainer.step() for _ in range(GROW_CHECK_GENERATIONS)]
    stats = [{k: str(v) for k, v in r.items() if k not in TIME_COLUMNS} for r in rows]
    champion = trainer.state.champion.genome
    return stats, b"" if champion is None else champion.tobytes(), sum(int(r["grow"]) for r in rows)


def _check_grow() -> str:
    """「자라는 뇌」(v6 설계 5.6·8.4 W2): 자라기 직전·직후 평균이 같은 결과(함수 보존)를 내고, 짧은 학습 두 번이 같은지."""
    from evolab.policy.genome import layout_for
    from evolab.spec.io import load_env

    path_file = paths.builtin_env_dir() / GROW_CHECK_ENV
    spec = load_env(path_file)
    layout = layout_for(spec)
    path, n_pairs, checksum = _grow_learner_check(spec, layout)
    text = path_file.read_text(encoding="utf-8-sig")
    first, second = _grow_training_run(text), _grow_training_run(text)
    if first != second:
        gen = next((k + 1 for k, (a, b) in enumerate(zip(first[0], second[0], strict=False)) if a != b), None)
        raise SelfTestError(S.SELFTEST_GROW_DIFF.format(detail=f"training run differs (generation {gen})"))
    stats, champion, grown = first
    if grown != len(path) - 1:
        raise SelfTestError(S.SELFTEST_GROW_DIFF.format(detail=f"training grew {grown} times, expected {len(path) - 1}"))
    return "\n".join([
        S.SELFTEST_GROW_SAME,
        f"{GROW_CHECK_ENV}: {' -> '.join(str(n) for n in path)}, pairs={n_pairs}, checksum={checksum}",
        f"training: generations={len(stats)}, growths={grown}, "
        f"checksum={_checksum(np.frombuffer(champion + repr(stats).encode(), dtype=np.uint8))}",
    ])


def _headless_offscreen() -> None:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _check_gui() -> str:
    _headless_offscreen()
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


# ---------------------------------------------------------------------------
# v3.0 점검: 도움말 쪽, 기록 폴더와 오류 보고 묶음, Qt 한국어, 처음 실행 안내
# ---------------------------------------------------------------------------


KEPT_LOGS: Final = ("errors.jsonl", "faults.log")
"""temporary_user_dirs(keep_logs=…)가 지우기 전에 진짜 기록 폴더로 옮겨 두는 임시 기록 파일."""


def _keep_logs(base: Path, tag: str) -> None:
    """임시 상태 폴더의 오류 기록을 진짜 기록 폴더에 selftest-<tag>-<이름>으로 남긴다(없으면 지난번 것을 지운다).
    창·안내 점검 도중 Qt 슬롯의 예외는 오류 기록에만 남으므로, 임시 폴더를 지우면 까닭을 잃는다(CI는 기록 폴더를 올린다)."""
    from evolab import paths as p

    for name in KEPT_LOGS:
        src = base / "state" / "logs" / name
        dest = p.log_dir() / f"selftest-{tag}-{name}"
        try:
            if src.is_file() and src.stat().st_size:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
            elif dest.exists():
                dest.unlink()
        except OSError:
            pass


@contextlib.contextmanager
def temporary_user_dirs(keep_logs: str | None = None) -> Iterator[Path]:
    """데이터·상태 폴더를 빈 임시 폴더로 바꾼다(EVOLAB_DATA_DIR·EVOLAB_STATE_DIR, 설계 D5). spawn 자식은 물려받는다.
    끝나면 환경 변수를 되돌리고 임시 폴더를 지운다(윈도우에서 아직 쥔 파일이 있으면 남은 것은 그대로 둔다).
    keep_logs(이름표)를 주면 지우기 전에 임시 오류 기록을 진짜 기록 폴더에 남긴다(_keep_logs)."""
    from evolab import paths as p

    base = Path(tempfile.mkdtemp(prefix="evolab-selftest-"))
    keys = (p.DATA_DIR_ENV, p.STATE_DIR_ENV)
    saved = {k: os.environ.get(k) for k in keys}
    os.environ[p.DATA_DIR_ENV] = str(base / "data")
    os.environ[p.STATE_DIR_ENV] = str(base / "state")
    try:
        yield base
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if keep_logs is not None:
            _keep_logs(base, keep_logs)
        for _ in range(10):  # 윈도우: 막 끝난 자식 프로세스의 파일 핸들이 늦게 풀리면 한 번에 지워지지 않는다
            shutil.rmtree(base, ignore_errors=True)
            if not base.exists():
                break
            time.sleep(0.3)


def _check_help() -> str:
    """도움말 쪽(strings.py HELP_PAGES)을 모두 열어 자리가 다 풀리고 낱말 풀이 쪽에 낱말이 모두 있는지(Qt 없이)."""
    from evolab.app.help import content

    for page in content.PAGES:
        if not content.load_page(page).strip():
            raise SelfTestError(S.SELFTEST_ERR_MISSING.format(what=page))
        unknown = content.unknown_placeholders(page)
        if unknown:
            raise SelfTestError(S.SELFTEST_ERR_MISSING.format(what=f"{page}: {', '.join(unknown)}"))
    words = content.load_page(content.GLOSSARY)
    missing = [term for term in S.HELP_GLOSSARY_TERMS if f"**{term}**" not in words]
    if missing:
        raise SelfTestError(S.SELFTEST_ERR_MISSING.format(what=f"{content.GLOSSARY}: {', '.join(missing)}"))
    return S.SELFTEST_HELP_DETAIL.format(pages=len(content.PAGES))


def _check_presets() -> str:
    """새 프리셋(6.2.0, 최종 설계 T40): 내장 틀 전부를 읽고 내장 조리법 전부를 만들어 L1·L2를 보고 달리기 한 세대를 돌린다(Qt 없이).

    실패하면 깨진 틀 파일이나 조리법의 이름이 든 글(presets.trainable.selftest_check)."""
    from evolab.presets.trainable import selftest_check

    ok, text = selftest_check()
    if not ok:
        raise SelfTestError(text)
    return text


def _check_logs() -> str:
    """빈 임시 상태 폴더의 로그 폴더에 써 보고 오류 보고 묶음을 만들어 확인한다(Qt 없이, report/bundle.self_check).

    EVOLAB_SELFTEST_CRASH=1이면(윈도우 CI W2만) 같은 exe로 spawn한 자식을 일부러 세그폴트시켜 종료 코드와 faults.log의
    치명적 오류 스택을 확인한다(설계 4.9). 사용자 PC에서는 돌지 않는다.
    """
    from evolab import diagnostics
    from evolab.app.report import bundle

    with temporary_user_dirs() as base:
        work = base / "work"
        work.mkdir()
        try:
            detail = bundle.self_check(work)
        except RuntimeError as exc:
            raise SelfTestError(S.SELFTEST_ERR_MISMATCH.format(detail=exc)) from None
        if os.environ.get(diagnostics.SELFTEST_CRASH_ENV, "").strip() == "1":
            ok, crash = diagnostics.crash_check(timeout=CRASH_TIMEOUT_S)
            if not ok:
                raise SelfTestError(S.SELFTEST_ERR_CRASH_NOT_LOGGED.format(detail=crash))
            detail += f"; crash child: {crash}"
    return detail


EXPORT_SECONDS: Final = 2.0
"""「내보내기」 점검이 쓰는 재생 앞부분(초, 설계 D59)."""
EXPORT_TIMEOUT_S: Final = 300.0


def _first_seconds(rep: Any, seconds: float) -> Any:
    """재생의 앞 seconds초(궤적 프레임만 자름, 세계·모양·결과는 그대로)."""
    import dataclasses

    tr = rep.trajectory
    n = min(tr.frames, int(seconds / (tr.every * tr.dt) + 1e-9) + 1)
    short = dataclasses.replace(tr, poses=tr.poses[:n].copy(), outputs=tr.outputs[:n].copy(),
                                end_step=None if n < tr.frames else tr.end_step)
    return dataclasses.replace(rep, trajectory=short, observations=None, activations=None, raw=None, rays=None)


def _check_export() -> str:
    """「내보내기」 점검(v3.1, 설계 D59): 내보내기 자식 프로세스로 내장 MP4(640×360, 고정 카메라)와 폰 재생 파일을 임시 폴더에
    만들고 MP4 상자 구조·크기·프레임 수, 폰 파일 자료 왕복(궤적 1mm 안)을 본다. MF 시험 결과는 정보로(윈도우면 작은 자식)."""
    import dataclasses

    from evolab.app.export import job as export_job
    from evolab.app.export import mp4, phone, runner
    from evolab.app.export.frames import frame_count
    from evolab.engine import seeds
    from evolab.engine.replay import replay
    from evolab.policy.genome import layout_for
    from evolab.spec.io import load_env

    spec = load_env(paths.builtin_env_dir() / "lander.evoenv")
    layout = layout_for(spec)
    rep = _first_seconds(replay(spec, layout, np.zeros(layout.dim), seeds.display_trial()), EXPORT_SECONDS)
    items = [(rep, export_job.NOW_COLOR, S.LAB_LABEL_REP.format(gen=0))]
    meta = {export_job.META_ENV: str(spec.file.name)}
    parts: dict[str, str] = {}
    first: list[float] = []
    with temporary_user_dirs(keep_logs="export") as base:
        work = base / "data" / S.VIDEO_SAVE_TITLE  # 한글 폴더 이름(묶인 exe의 한글 경로 점검)
        job = dataclasses.replace(export_job.job_from_replays(items, "mp4", size="small", camera="fixed",
                                                              title=S.SELFTEST_CHECK_EXPORT, meta=meta),
                                  encoder="builtin")
        try:
            done = runner.run_in_child(job, work / "selftest.mp4", timeout=EXPORT_TIMEOUT_S)
        except runner.ExportFailed as exc:
            raise SelfTestError(S.SELFTEST_ERR_MISMATCH.format(detail=f"mp4: {exc.message}\n{exc.detail}")) from None
        first.append(float(done.get("first_message_s") or 0.0))
        info = mp4.read_info(done["path"])
        want = frame_count(job)
        if (info["boxes"][:1] != ["ftyp"] or "moov" not in info["boxes"] or info["frames"] != want
                or (info["width"], info["height"]) != job.size or done["encoder"] != "builtin"):
            raise SelfTestError(S.SELFTEST_ERR_MISMATCH.format(detail=f"mp4 {info} (frames {want}, size {job.size})"))
        parts["mp4"] = S.SELFTEST_EXPORT_PART.format(frames=info["frames"], kb=done["bytes"] / 1024)
        job = export_job.job_from_replays(items, "html", size="small", camera="fixed", title=S.SELFTEST_CHECK_EXPORT,
                                          meta=meta)
        try:
            done = runner.run_in_child(job, work / "selftest.html", timeout=EXPORT_TIMEOUT_S)
        except runner.ExportFailed as exc:
            raise SelfTestError(S.SELFTEST_ERR_MISMATCH.format(detail=f"html: {exc.message}\n{exc.detail}")) from None
        first.append(float(done.get("first_message_s") or 0.0))
        entry = phone.decode_payload(phone.read_phone_html(done["path"]))["entries"][0]
        src = np.asarray(rep.trajectory.poses, dtype=np.float64)
        got = np.asarray(entry["poses"])
        if got.shape != src.shape or float(np.abs(got[..., :2] - src[..., :2]).max()) > phone.POS_UNIT:
            raise SelfTestError(S.SELFTEST_ERR_MISMATCH.format(detail=f"html poses {got.shape} vs {src.shape}"))
        parts["html"] = S.SELFTEST_EXPORT_PART.format(frames=got.shape[0], kb=done["bytes"] / 1024)
        # 시험 자식(윈도우)도 임시 상태 폴더를 물려받게 이 안에서(자식의 faults.log가 실제 로그 폴더에 생기지 않게, 설계 D5)
        kind, detail = runner.probe_in_child() if runner.probe_needs_child() else runner.probe_in_process()
    mf = S.EXP_ENCODER_MF.format(name=detail) if kind == "mf" else detail
    line = S.SELFTEST_EXPORT_DETAIL.format(mp4=parts["mp4"], html=parts["html"], mf=mf)
    return f"{line}\nchild first message: {' / '.join(f'{v:.2f}s' for v in first)}"


def _has_hangul(text: str) -> bool:
    return any(0xAC00 <= ord(ch) <= 0xD7A3 for ch in text)


def qt_button_texts() -> list[str]:
    """표준 단추 저장·닫기·취소·예·아니요의 글(번역기를 넣은 앱에서)."""
    from PySide6.QtWidgets import QDialogButtonBox

    b = QDialogButtonBox.StandardButton
    order = (b.Save, b.Close, b.Cancel, b.Yes, b.No)
    box = QDialogButtonBox(b.Save | b.Close | b.Cancel | b.Yes | b.No)
    try:
        return [box.button(x).text() for x in order]
    finally:
        box.deleteLater()


def _check_qt_ko() -> str:
    """Qt 기본 문구가 한국어인지(qtbase_ko 번역기, 설계 D25): 표준 단추 다섯 개의 글에 한글이 있어야 한다."""
    _headless_offscreen()
    from PySide6.QtCore import QTranslator

    from evolab.app.main import QT_TRANSLATOR_NAME, create_application

    app = create_application()
    texts = qt_button_texts()
    if not all(_has_hangul(t) for t in texts):
        raise SelfTestError(S.SELFTEST_ERR_QT_KO.format(text=", ".join(texts)))
    tr = app.findChild(QTranslator, QT_TRANSLATOR_NAME)
    name = Path(tr.filePath()).name if tr is not None else "?"
    return f"{' · '.join(texts)} ({name})"


def run_tour_check(until: str = "gen1") -> dict[str, Any]:
    """빈 데이터·상태 폴더에서 주 창을 띄우고 처음 실행 안내를 자동 모드로(단추를 스스로 누름) 돌린다(설계 D16).

    돌려주는 값: result(안내 결과), seconds(until에 따라 창 → 1세대 재생 또는 창 → 목표 세대), gen1, goal_s, clicks,
    first·goal(1세대·목표 세대 평균 점수), workers, log(단계 기록). 학습은 끝내고 창은 닫는다.
    """
    _headless_offscreen()
    from PySide6.QtCore import QEventLoop, QTimer

    from evolab.app.help import tour
    from evolab.app.main import MainWindow, create_application

    with temporary_user_dirs(keep_logs="tour"):
        app = create_application()
        window = MainWindow()
        try:
            window.resize(1280, 800)
            window.show()
            app.processEvents()
            overlay = tour.start_tour(window, window.lab.settings, auto=True, until=until)  # type: ignore[arg-type]
            if overlay is None:
                raise SelfTestError(S.SELFTEST_ERR_TOUR_FAILED.format(detail="start_tour=None"))
            if overlay.result is None:
                loop = QEventLoop()
                overlay.finished.connect(loop.quit)
                QTimer.singleShot(int(TOUR_WAIT_S[until] * 1000), loop.quit)
                loop.exec()
            t = overlay.times
            out: dict[str, Any] = {
                "result": overlay.result, "clicks": t.clicks, "first": t.first_mean, "goal": t.goal_mean,
                "workers": window.lab.workers.value(),
                "gen1": None if t.gen1_playing is None else t.gen1_playing - t.shown,
                "goal_s": None if t.goal_reached is None else t.goal_reached - t.shown,
                "log": [k for _t, k in getattr(overlay, "log", [])],
            }
            out["seconds"] = out["gen1"] if until == "gen1" else out["goal_s"]
            return out
        finally:
            window.lab.end_session()
            window.close()
            window.deleteLater()
            app.processEvents()


def _check_tour(until: str = "gen1") -> str:
    """처음 실행 안내(1세대까지 자동 진행, `--tour full`이면 10세대까지 4배속): 결과·초·누른 수를 적고 기준을 넘으면 실패."""
    from evolab.app.help import tour

    r = run_tour_check(until)
    limit = TOUR_LIMIT_S[until]
    if r["result"] != "done" or r["seconds"] is None:
        if r["result"] is None:
            raise SelfTestError(S.SELFTEST_ERR_TIMEOUT.format(seconds=int(TOUR_WAIT_S[until])))
        raise SelfTestError(S.SELFTEST_ERR_TOUR_FAILED.format(detail=f"result={r['result']}, steps={'/'.join(r['log'])}"))
    lines = [S.SELFTEST_TOUR_DETAIL.format(seconds=r["gen1"], clicks=r["clicks"])]
    what = S.SELFTEST_TOUR_WHAT_GEN1
    if until == "full":
        first = r["first"] if r["first"] is not None else float("nan")
        now = r["goal"] if r["goal"] is not None else float("nan")
        lines.append(S.SELFTEST_TOUR_FULL_DETAIL.format(goal=tour.TOUR_GOAL, seconds=r["seconds"], first=first, now=now))
        what = S.SELFTEST_TOUR_WHAT_GOAL.format(goal=tour.TOUR_GOAL)
    lines.append(f"workers={r['workers']}")
    if r["seconds"] > limit:
        raise SelfTestError("\n".join([S.SELFTEST_ERR_TOUR_SLOW.format(what=what, seconds=r["seconds"],
                                                                         limit=int(limit)), *lines]))
    return "\n".join(lines)


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


def selftest_checks(workers: int, check_gui: bool = True, tour: str = "gen1") -> list[tuple[str, Callable[[], str]]]:
    """점검 목록(이름, 함수). 창이 필요 없는 것 먼저, 그다음 창 점검(--no-gui면 뺀다)."""
    if tour not in TOUR_MODES:
        raise ValueError(tour)
    checks: list[tuple[str, Callable[[], str]]] = [
        (S.SELFTEST_CHECK_IMPORTS, _check_imports),
        (S.SELFTEST_CHECK_PHYSICS, _check_physics),
        (S.SELFTEST_CHECK_LEARNER, _check_learner),
        (S.SELFTEST_CHECK_GROW, _check_grow),
        (S.SELFTEST_CHECK_PROCESSES, lambda: _check_processes(workers)),
        (S.SELFTEST_CHECK_ENV, _check_builtin_env),
        (S.SELFTEST_CHECK_BUNDLE, _check_bundle_files),
        (S.SELFTEST_CHECK_FASTPATH, _check_fastpath),
        (S.SELFTEST_CHECK_EXPORT, _check_export),
        (S.SELFTEST_CHECK_HELP, _check_help),
        (S.SELFTEST_CHECK_PRESETS, _check_presets),
        (S.SELFTEST_CHECK_LOGS, _check_logs),
    ]
    if check_gui:
        from evolab.app.help.tour import TOUR_GOAL

        tour_name = S.SELFTEST_CHECK_TOUR if tour == "gen1" else S.SELFTEST_CHECK_TOUR_FULL.format(goal=TOUR_GOAL)
        checks += [
            (S.SELFTEST_CHECK_GUI, _check_gui),
            (S.SELFTEST_CHECK_QT_KO, _check_qt_ko),
            (tour_name, lambda: _check_tour(tour)),
        ]
    return checks


def run_selftest(log_path: Path | None = None, workers: int | None = None, check_gui: bool = True,
                 tour: str = "gen1") -> int:
    n = workers if workers is not None else default_workers()
    checks = selftest_checks(n, check_gui, tour)

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

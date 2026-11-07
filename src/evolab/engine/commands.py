"""명령줄 학습 명령: evolab train / bench / judge.

화면 없이 학습하고(train), 성능을 재고(bench), 완료 기준을 판정한다(judge). 출력은 모두 한국어이고 strings.py에 있다.
"""

from __future__ import annotations

import math
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from evolab import paths
from evolab.app import strings as S
from evolab.engine import seeds
from evolab.engine.pool import PoolError, WorkerPool, default_workers
from evolab.engine.runsetup import create_run
from evolab.engine.trainer import TrainConfig, Trainer, training_trials
from evolab.policy.genome import layout_for
from evolab.runs.store import RunStoreError
from evolab.spec.compiled import EnvFileError
from evolab.spec.io import compile_env, env_hash, load_env

METHOD_NAMES = S.METHOD_NAMES
RESUME_OVERRIDES = ("max_generations", "max_evaluations", "max_time", "target_score", "stagnation_generations")


def say(text: str) -> None:
    if sys.stdout is not None:
        print(text, flush=True)


def warn(text: str) -> None:
    if sys.stderr is not None:
        print(text, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


class _StopFlag:
    """Ctrl+C 한 번이면 이번 세대를 마치고 멈춘다(체크포인트 남김). 두 번이면 바로 끝낸다."""

    def __init__(self) -> None:
        self.requested = False
        self._old: Any = None

    def __enter__(self) -> _StopFlag:
        def handler(signum: int, frame: Any) -> None:
            if self.requested:
                raise KeyboardInterrupt
            self.requested = True
            warn(S.TRAIN_STOPPING)

        self._old = signal.signal(signal.SIGINT, handler)
        return self

    def __exit__(self, *exc: object) -> None:
        signal.signal(signal.SIGINT, self._old)


def _config_from_args(args: Any, base: TrainConfig | None = None) -> TrainConfig:
    c = base.to_dict() if base is not None else TrainConfig().to_dict()
    for key, attr in (
        ("method", "method"), ("run_seed", "seed"), ("popsize", "popsize"), ("sigma0", "sigma"),
        ("max_generations", "max_generations"), ("max_evaluations", "max_evaluations"), ("max_time", "max_time"),
        ("target_score", "target"), ("stagnation_generations", "stagnation"), ("trials", "trials"),
        ("validate_every", "validate_every"), ("diagonal_threshold", "diagonal_threshold"), ("ipop", "ipop"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            c[key] = value
    if base is None and getattr(args, "popsize", None) is None and c.get("method") == "ga":
        c["popsize"] = 64  # 명세: 유전 알고리즘 기본 개체 수 64(CMA-ES는 32)
    return TrainConfig.from_dict(c)


def _progress(trainer: Trainer, state: dict[str, float]) -> Any:
    def on_generation(row: dict[str, Any]) -> None:
        now = time.perf_counter()
        if now - state["last"] < 2.0:
            return
        state["last"] = now
        champ = row["champion_score"]
        say(S.TRAIN_PROGRESS.format(
            gen=row["generation"],
            evals=row["evaluations"],
            best=row["score_best"],
            mean=row["score_mean"],
            champ=f"{champ:.1f}" if champ != "" else "-",
            elapsed=row["elapsed"],
            sps=row["steps_per_second"],
        ))

    return on_generation


def train_command(args: Any) -> int:
    workers = args.workers if args.workers is not None else default_workers()
    try:
        if args.resume is not None:
            from evolab.runs.store import RunStore

            store = RunStore.open(Path(args.resume))
            pool = WorkerPool(workers)
            gen, ckpt = store.load_checkpoint(None)
            config = _config_from_args(args, TrainConfig.from_dict(ckpt["config"]))
            # 이어할 때는 정지 조건만 바꿀 수 있다(학습 방법·시드·개체 수를 바꾸면 다른 실험이 된다).
            overrides = {k: getattr(config, k) for k in RESUME_OVERRIDES}
            trainer = Trainer.resume(store, pool, None, **overrides)
            store.append_event("resume", gen, workers=workers)
            say(S.TRAIN_RESUMED.format(gen=gen, path=store.root))
        else:
            if args.env is None:
                warn(S.TRAIN_NEED_ENV)
                return 2
            spec = load_env(args.env)
            env_text = Path(args.env).read_text(encoding="utf-8-sig")
            config = _config_from_args(args)
            from evolab.runs.store import RunStore

            base_dir = Path(args.out) if args.out else paths.runs_dir()
            config.validate()
            layout = layout_for(spec)
            store = create_run(base_dir, env_text, spec, config, workers)
            pool = WorkerPool(workers)
            trainer = Trainer(env_text, config, pool, store)
            say(S.TRAIN_STARTED.format(name=spec.file.name, method=METHOD_NAMES.get(config.method), dim=layout.dim,
                                       pop=config.popsize, trials=trainer.trials, workers=workers, path=store.root))
    except (EnvFileError, RunStoreError, ValueError, OSError) as exc:
        warn(S.TRAIN_CANNOT_START.format(reason=exc))
        return 2
    try:
        with pool, _StopFlag() as flag:
            reason = trainer.run(_progress(trainer, {"last": 0.0}), lambda: flag.requested)
    except PoolError as exc:
        # 이번 세대는 중간에 멈췄으므로 체크포인트를 새로 남기지 않는다(마지막 정상 체크포인트에서 이어간다).
        warn(str(exc))
        return 3
    except KeyboardInterrupt:
        warn(S.TRAIN_ABORTED)
        return 130
    s = trainer.state
    store.append_event("stop", s.generation, reason=reason)
    say(S.TRAIN_FINISHED.format(
        reason=S.TRAIN_STOP_REASONS.get(reason, reason), gen=s.generation, evals=s.evaluations,
        champ=s.champion.score if math.isfinite(s.champion.score) else float("nan"),
        champ_gen=s.champion.generation, elapsed=s.elapsed, path=store.root,
    ))
    return 0


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------


def judge_command(args: Any) -> int:
    from evolab.runs.store import RunStore

    try:
        store = RunStore.open(Path(args.run))
    except (OSError, ValueError, RunStoreError) as exc:
        warn(S.JUDGE_OPEN_FAILED.format(reason=exc))
        return 2
    gens = store.champion_generations("champion")
    if args.generation is not None:
        gens = [g for g in gens if g <= args.generation]
    if not gens:
        warn(S.JUDGE_NO_CHAMPION)
        return 2
    gen = gens[-1]
    champ = store.load_champion(gen, "champion")
    info = store.read_run_info()
    # 챔피언은 gen세대를 계산할 때 쓴 환경으로 잰다: 적용 세대가 gen - 1 이하인 마지막 버전
    # (gen세대 경계에서 바꾼 환경은 gen + 1세대부터 쓴다).
    versions = [v for v in info["env_versions"] if v["from_generation"] <= gen - 1] or info["env_versions"][:1]
    env_text = store.env_text(versions[-1]["id"])
    want = str(champ["env_version"]) if "env_version" in champ else None
    if want is not None and env_hash(compile_env(env_text).file)[:16] != want:
        for v in info["env_versions"]:  # 챔피언 파일에 적힌 환경 버전이 있으면 그것으로
            text = store.env_text(v["id"])
            if env_hash(compile_env(text).file)[:16] == want:
                env_text = text
                break
    spec = compile_env(env_text)
    layout = layout_for(spec)
    trials = seeds.judge_trials(args.seeds)
    workers = args.workers if args.workers is not None else default_workers()
    try:
        with WorkerPool(workers) as pool:
            res = pool.evaluate(env_hash(spec.file)[:16], env_text, layout, champ["genome"][None, :],
                                [(0, t) for t in trials])
    except PoolError as exc:
        warn(str(exc))
        return 3
    success = sum(1 for o in res.outcomes if o == "success")
    labels = {t.event: (t.label or t.event) for t in spec.terminations}
    counts: dict[str, int] = {}
    for e in res.end_events:
        counts[labels.get(e, e)] = counts.get(labels.get(e, e), 0) + 1
    say(S.JUDGE_RESULT.format(
        gen=gen, n=len(trials), success=success, rate=100.0 * success / len(trials),
        mean=float(res.scores.mean()), ends=", ".join(f"{k} {v}" for k, v in counts.items()),
    ))
    store.append_event("judge", gen, seeds=len(trials), success=success, mean_score=float(res.scores.mean()))
    return 0


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def bench_command(args: Any) -> int:
    try:
        spec = load_env(args.env)
    except EnvFileError as exc:
        warn(str(exc))
        return 2
    env_text = Path(args.env).read_text(encoding="utf-8-sig")
    version = env_hash(spec.file)[:16]
    layout = layout_for(spec)
    rng = np.random.default_rng(0)
    pop, trials_n = args.popsize, spec.track_set_size or spec.file.evaluation.trials
    genomes = layout.initial()[None, :] + 0.5 * rng.standard_normal((pop, layout.dim))
    trials = training_trials(spec, 1, 0, trials_n)
    pairs = [(i, t) for i in range(pop) for t in trials]
    say(S.BENCH_TITLE.format(name=spec.file.name, pop=pop, trials=trials_n, dim=layout.dim))

    # 단일 코어(이 프로세스)
    single = WorkerPool(0)
    t0 = time.perf_counter()
    res = single.evaluate(version, env_text, layout, genomes, pairs)
    el = time.perf_counter() - t0
    steps = int(res.steps.sum())
    nn_share = res.nn_seconds / max(res.nn_seconds + res.sim_seconds, 1e-12)
    say(S.BENCH_SINGLE.format(sps=steps / el, us=el / steps * 1e6, nn=100 * nn_share, steps=steps, sec=el))

    counts = [int(x) for x in args.workers.split(",")] if args.workers else sorted({1, default_workers()})
    for w in counts:
        with WorkerPool(w) as pool:
            pool.evaluate(version, env_text, layout, genomes[:2], pairs[: 2 * trials_n])  # 워커 기동·컴파일 시간 빼기
            times = []
            for g in range(args.generations):
                trials_g = training_trials(spec, 1, g, trials_n)
                pairs_g = [(i, t) for i in range(pop) for t in trials_g]
                t0 = time.perf_counter()
                r = pool.evaluate(version, env_text, layout, genomes, pairs_g)
                times.append(time.perf_counter() - t0)
                steps_g = int(r.steps.sum())
            med = statistics.median(times)
            say(S.BENCH_POOL.format(workers=w, sec=med, sps=steps_g / times[-1]))
    worst = pop * trials_n * math.ceil(spec.file.world.max_time / spec.file.world.dt)
    say(S.BENCH_WORST.format(steps=worst, sec=worst / (steps / el) / max(1, default_workers()), workers=default_workers()))
    return 0


# ---------------------------------------------------------------------------
# 내보내기: bundle / reproduce / csv
# ---------------------------------------------------------------------------


def bundle_command(args: Any) -> int:
    from evolab.runs.bundle import export_bundle

    run = Path(args.run)
    out = Path(args.out) if args.out else run / "exports" / S.BUNDLE_FILE_NAME.format(run=run.name)
    try:
        path = export_bundle(run, out)
    except (RunStoreError, OSError) as exc:
        warn(str(exc))
        return 2
    say(S.BUNDLE_DONE.format(path=path))
    return 0


def reproduce_command(args: Any) -> int:
    from evolab.runs.bundle import compare_stats, read_bundle, reproduce
    from evolab.runs.store import RunStore

    base = Path(args.out) if args.out else paths.runs_dir()
    state = {"last": 0.0}

    def progress(row: dict[str, Any]) -> None:
        now = time.perf_counter()
        if now - state["last"] >= 2.0:
            state["last"] = now
            say(S.BUNDLE_PROGRESS.format(gen=row["generation"]))

    try:
        original = read_bundle(Path(args.bundle))["stats"]
        new_dir = reproduce(Path(args.bundle), base, args.workers, progress)
    except PoolError as exc:
        warn(str(exc))
        return 3
    except (RunStoreError, EnvFileError, ValueError, OSError) as exc:
        warn(str(exc))
        return 2
    diff = compare_stats(original, RunStore.open(new_dir).read_stats())
    say(S.BUNDLE_REPRODUCED.format(path=new_dir))
    if diff is None:
        say(S.BUNDLE_SAME.format(gens=len(original)))
        return 0
    say(S.BUNDLE_DIFFERENT.format(gen=diff))
    return 1


def csv_command(args: Any) -> int:
    from evolab.runs.bundle import export_csv

    run = Path(args.run)
    out = Path(args.out) if args.out else run / "exports" / "stats.csv"
    try:
        path = export_csv(run, out)
    except (RunStoreError, OSError) as exc:
        warn(str(exc))
        return 2
    say(S.CSV_DONE.format(path=path))
    return 0

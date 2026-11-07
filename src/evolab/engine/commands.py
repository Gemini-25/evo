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
from evolab.engine.trainer import (
    SWITCH_QUALIFY_NEXT,
    SWITCH_RESTART_CHAMPION,
    SWITCH_RESTART_FRESH,
    TrainConfig,
    Trainer,
    qualify_budget_problem,
    training_trials,
)
from evolab.policy.genome import layout_for
from evolab.runs.store import RunStoreError
from evolab.spec.compiled import EnvFileError
from evolab.spec.io import compile_env, env_hash, load_env

METHOD_NAMES = S.METHOD_NAMES
RESUME_OVERRIDES = ("max_generations", "max_evaluations", "max_time", "target_score", "stagnation_generations")
RESUME_LIVE = ("restart_patience", "restart_from")
"""이어 할 때 플래그로 바꿀 수 있는 학습 중 변경(화면의 「적용」과 같은 길: apply_config → 체크포인트 → 보호 → 사건)."""
RESUME_FIXED_FLAGS = (("--qualify", "qualify"), ("--qualify-trials", "qualify_trials"))
"""이어 할 때 바꿀 수 없는 플래그(예선은 실행 처음에만 뜻이 있다)."""


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
    """명령줄 인수로 설정을 만든다. base가 없으면 새 실행: 자동 재시작 같은 권장값(NEW_RUN_DEFAULTS) 위에 인수."""
    c = base.to_dict() if base is not None else TrainConfig.for_new_run().to_dict()
    for key, attr in (
        ("method", "method"), ("run_seed", "seed"), ("popsize", "popsize"), ("sigma0", "sigma"),
        ("max_generations", "max_generations"), ("max_evaluations", "max_evaluations"), ("max_time", "max_time"),
        ("target_score", "target"), ("stagnation_generations", "stagnation"), ("trials", "trials"),
        ("validate_every", "validate_every"), ("diagonal_threshold", "diagonal_threshold"), ("ipop", "ipop"),
        ("restart_patience", "restart"), ("restart_from", "restart_from"), ("qualify_entrants", "qualify"),
        ("qualify_trials", "qualify_trials"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            c[key] = value
    if getattr(args, "no_restart", False):
        c["restart_patience"] = None
    if base is None and getattr(args, "popsize", None) is None and c.get("method") == "ga":
        c["popsize"] = 64  # 명세: 유전 알고리즘 기본 개체 수 64(CMA-ES는 32)
    return TrainConfig.from_dict(c)


def _live_changes(args: Any, config: TrainConfig) -> dict[str, Any]:
    """이어 할 때 실제로 준 재시작 플래그만(RESUME_LIVE). 주지 않은 설정은 체크포인트 값 그대로."""
    given = {
        "restart_patience": getattr(args, "restart", None) is not None or bool(getattr(args, "no_restart", False)),
        "restart_from": getattr(args, "restart_from", None) is not None,
    }
    return {k: getattr(config, k) for k in RESUME_LIVE if given[k]}


def _stability_text(config: TrainConfig) -> str:
    """새 실행 시작 때 한 줄: 자동 재시작과 예선 설정."""
    if config.restart_patience is not None:
        restart = S.TRAIN_STABILITY_RESTART_ON.format(
            p=config.restart_patience, where=S.RESTART_FROM_NAMES.get(config.restart_from, config.restart_from))
    else:
        restart = S.TRAIN_STABILITY_OFF
    qualify = (S.TRAIN_STABILITY_QUALIFY_ON.format(k=config.qualify_entrants, b=config.qualify_trials)
               if config.qualify_entrants > 1 else S.TRAIN_STABILITY_OFF)
    return S.TRAIN_STABILITY.format(restart=restart, qualify=qualify)


def _changes_text(changes: dict[str, Any]) -> str:
    """이어 할 때 바꾼 재시작 설정(RESUME_LIVE)을 실험실 칸 이름으로. 끔은 '끔', 출발점은 화면의 선택 이름."""
    parts: list[str] = []
    for k, v in changes.items():
        if k == "restart_patience":
            parts.append(f"{S.LAB_RESTART} {S.TRAIN_STABILITY_OFF}" if v is None
                         else f"{S.LAB_RESTART_PATIENCE} {v:,}{S.LAB_UNIT_TRIALS.strip()}")
        elif k == "restart_from":
            parts.append(f"{S.LAB_RESTART_FROM} {S.RESTART_FROM_NAMES.get(v, v)}")
        else:  # RESUME_LIVE 밖의 키는 오지 않는다(오면 이름=값으로라도 알린다)
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _score_text(value: Any) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) and math.isfinite(value) else "-"


def _switch_text(trainer: Trainer, row: dict[str, Any]) -> str | None:
    """이 세대 끝에 학습 방법을 바꿨으면 알림 한 줄. 옛 열 머리로 시작한 실행은 열이 없어 None."""
    code = row.get("switch")
    if not isinstance(code, int) or code <= 0:
        return None
    gen = row["generation"]
    if code in (SWITCH_RESTART_FRESH, SWITCH_RESTART_CHAMPION):
        where = S.RESTART_FROM_NAMES_SHORT["champion" if code == SWITCH_RESTART_CHAMPION else "fresh"]
        return S.TRAIN_SWITCH_RESTART.format(gen=gen, stall=row["learner_stall"], where=where, n=row["restarts"] + 2)
    if code == SWITCH_QUALIFY_NEXT:
        return S.TRAIN_SWITCH_QUALIFY_NEXT.format(gen=gen, n=row["learner"] + 1, score=_score_text(row["learner_best"]),
                                                  next=row["learner"] + 2)
    cur = trainer.state.current  # 예선 끝: 통과자는 이 세대 끝에 이미 불러왔다
    return S.TRAIN_SWITCH_QUALIFY_PICK.format(gen=gen, winner=cur.index + 1, score=_score_text(cur.best))


def _progress_suffix(trainer: Trainer, row: dict[str, Any]) -> str:
    """진행 줄 끝: 예선 중이면 출전 번호, 아니면 시도 번호(옛 열 머리 실행은 없음)."""
    if "qualifying" not in row:
        return ""
    if row["qualifying"]:
        return S.TRAIN_PROGRESS_QUALIFY.format(n=row["learner"] + 1, k=trainer.config.qualify_entrants)
    return S.TRAIN_PROGRESS_ATTEMPT.format(n=row["restarts"] + 1)


def _progress(trainer: Trainer, state: dict[str, float]) -> Any:
    def on_generation(row: dict[str, Any]) -> None:
        note = _switch_text(trainer, row)
        if note is not None:  # 학습 방법 교체는 진행 줄 간격과 상관없이 바로 알린다
            say(note)
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
        ) + _progress_suffix(trainer, row))

    return on_generation


def train_command(args: Any) -> int:
    workers = args.workers if args.workers is not None else default_workers()
    pool: WorkerPool | None = None
    try:
        if args.resume is not None:
            from evolab.runs.store import RunStore

            fixed = [flag for flag, attr in RESUME_FIXED_FLAGS if getattr(args, attr, None) is not None]
            if fixed:
                warn(S.TRAIN_RESUME_QUALIFY_FIXED.format(flags=", ".join(fixed)))
                return 2
            store = RunStore.open(Path(args.resume))
            gen, ckpt = store.load_checkpoint(None)
            config = _config_from_args(args, TrainConfig.from_dict(ckpt["config"]))
            # 이어할 때는 정지 조건과 자동 재시작 설정만 바꿀 수 있다(학습 방법·시드·개체 수를 바꾸면 다른 실험이 된다).
            overrides = {k: getattr(config, k) for k in RESUME_OVERRIDES}
            live = _live_changes(args, config)
            # 잘못된 값(--max-generations 0, 검증 주기보다 짧은 --restart 등)은 워커를 띄우고 체크포인트 뒤를 잘라 내기
            # 전에 거른다.
            TrainConfig.from_dict({**ckpt["config"], **overrides, **live}).validate()
            pool = WorkerPool(workers)
            trainer = Trainer.resume(store, pool, None, **overrides)
            store.append_event("resume", gen, workers=workers)
            say(S.TRAIN_RESUMED.format(gen=gen, path=store.root))
            if live:
                # 화면의 「적용」(세션)과 같은 순서: 트레이너에 적용 → 체크포인트(바뀐 상태) → 보호 → 사건 줄
                changes = trainer.apply_config(live)
                if changes:
                    g = trainer.state.generation
                    trainer.checkpoint()
                    store.protect(g)
                    store.append_event("config_change", g, changes=changes)
                    say(S.TRAIN_CONFIG_CHANGED.format(gen=g, changes=_changes_text(changes)))
        else:
            if args.env is None:
                warn(S.TRAIN_NEED_ENV)
                return 2
            spec = load_env(args.env)
            env_text = Path(args.env).read_text(encoding="utf-8-sig")
            config = _config_from_args(args)
            from evolab.runs.store import RunStore

            base_dir = Path(args.out) if args.out else paths.runs_dir()
            if (getattr(args, "restart", None) is None and not getattr(args, "no_restart", False)
                    and config.restart_patience is not None and config.restart_patience < config.validate_every):
                # 재시작을 따로 주지 않았는데 새 실행 권장값이 검증 주기보다 짧다(v1.0에서 되던 명령): 고칠 방법을 알린다
                warn(S.TRAIN_CANNOT_START.format(reason=S.TRAIN_RESTART_TOO_SHORT.format(
                    value=config.restart_patience, every=config.validate_every) + " " + S.CLI_RESTART_TOO_SHORT_HINT))
                return 2
            config.validate()
            problem = qualify_budget_problem(config)
            if problem is not None:  # 예선만으로 예산을 다 쓰는 실행은 시작하지 않는다(폴더도 만들지 않음)
                warn(problem)
                return 2
            layout = layout_for(spec)
            store = create_run(base_dir, env_text, spec, config, workers)
            pool = WorkerPool(workers)
            trainer = Trainer(env_text, config, pool, store)
            say(S.TRAIN_STARTED.format(name=spec.file.name, method=METHOD_NAMES.get(config.method), dim=layout.dim,
                                       pop=config.popsize, trials=trainer.trials, workers=workers, path=store.root))
            say(_stability_text(config))
    except (EnvFileError, RunStoreError, ValueError, OSError) as exc:
        if pool is not None:
            pool.close()  # 워커를 띄운 뒤 실패(깨진 체크포인트 등): 워커를 남기지 않는다
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
    champ = (S.TRAIN_CHAMPION.format(score=s.champion.score, gen=s.champion.generation)
             if math.isfinite(s.champion.score) else S.TRAIN_NO_CHAMPION_YET)
    text = S.TRAIN_FINISHED.format(
        reason=S.TRAIN_STOP_REASONS.get(reason, reason), gen=s.generation, evals=s.evaluations,
        champ=champ, elapsed=s.elapsed, path=store.root,
    )
    if s.restarts:  # 요약 줄(첫 줄) 끝에 붙인다. 실행 폴더 줄은 그대로 마지막 줄
        first, sep, rest = text.partition("\n")
        text = first + S.TRAIN_FINISHED_RESTARTS.format(n=s.restarts) + sep + rest
    say(text)
    return 0


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------


def judge_command(args: Any) -> int:
    from evolab.runs.store import RunStore

    if args.seeds < 1:  # 명령줄 파서가 먼저 거른다. 판정 시드가 없으면 성공률을 낼 수 없다.
        warn(S.RVM_CLI_OPTION_TOO_SMALL.format(name="--seeds", minimum=1, value=args.seeds))
        return 2
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
    # 명령줄 파서가 먼저 거른다(측정을 시작한 뒤 0으로 나누기·빈 중앙값으로 죽지 않게).
    counts = list(args.workers or []) or sorted({1, default_workers()})
    for name, value, minimum in (("--popsize", args.popsize, 2), ("--generations", args.generations, 1),
                                 ("--workers", min(counts), 1)):
        if value < minimum:
            warn(S.RVM_CLI_OPTION_TOO_SMALL.format(name=name, minimum=minimum, value=value))
            return 2
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

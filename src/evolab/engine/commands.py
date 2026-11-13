"""명령줄 학습 명령: evolab train / bench / judge.

화면 없이 학습하고(train), 성능을 재고(bench), 완료 기준을 판정한다(judge). 출력은 모두 한국어이고 strings.py에 있다.
train --no-observe는 그래프용 관찰 기록(v1.2)만 끈다(Trainer 인자, 설정·run.json에는 남지 않음). bench는 단일 코어
측정 뒤 관찰을 켠 같은 평가를 한 번 더 돌려 차이를 알린다(D32). v6.0(v6 설계 4.10·D17): 단일 코어 줄은 학습과 같은 평가 창
(evaluate.EVAL_WINDOW)으로 재고 문구에 창 크기를 적는다. 새 줄: 「평가 창」(BENCH_WINDOWS를 번갈아), 「관전 재생 길이」(이미 잰
평가의 시험 0번 스텝 수, 계산이 더 들지 않음), 「pymunk C 함수 바로 부르기」(episode._CP_FAST).
v6.1 자라는 뇌(v6 설계 4.16·5.5): train --method grow --grow-start --grow-every. 새 실행은 시작 줄 뒤에 TRAIN_GROW_SUMMARY 한 줄
(처음부터 다 자란 크기면 TRAIN_GROW_SUMMARY_FULL), 자란 세대마다 CLI_TRAIN_GROWN 한 줄(진행 줄 간격과 상관없이. 자란 세대 끝에
자동 재시작·예선으로 그 학습 방법을 버리면 알리지 않는다: Trainer.last_growth). 은닉층이 없는 뇌면 실행 폴더를 만들기 전에 거절.
이어 할 때는 grow_*를 바꿀 수 없다(학습 방법의 일부: --grow-start·--grow-every를 주면 TRAIN_RESUME_GROW_FIXED, 종료 코드 2).
"""

from __future__ import annotations

import math
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Final

import numpy as np

from evolab import paths
from evolab.app import strings as S
from evolab.engine import observe as observing
from evolab.engine import seeds
from evolab.engine.evaluate import EVAL_WINDOW, OBSERVE_ACTIONS, OBSERVE_POSITIONS, mean_of_built
from evolab.engine.pool import PoolError, WorkerPool, default_workers
from evolab.engine.runsetup import create_run
from evolab.engine.trainer import (
    SWITCH_QUALIFY_NEXT,
    SWITCH_RESTART_CHAMPION,
    SWITCH_RESTART_FRESH,
    TrainConfig,
    Trainer,
    grow_problem,
    qualify_budget_problem,
    structure_note,
    structure_range_note,
    training_trials,
)
from evolab.policy.genome import layout_for
from evolab.runs.store import RunStoreError
from evolab.spec.compiled import EnvFileError, EnvSpec
from evolab.spec.io import compile_env, env_hash, load_env

METHOD_NAMES = S.METHOD_NAMES
RESUME_OVERRIDES = ("max_generations", "max_evaluations", "max_time", "target_score", "stagnation_generations")
RESUME_LIVE = ("restart_patience", "restart_from")
"""이어 할 때 플래그로 바꿀 수 있는 학습 중 변경(화면의 「적용」과 같은 길: apply_config → 체크포인트 → 보호 → 사건)."""
RESUME_FIXED_FLAGS = (("--qualify", "qualify"), ("--qualify-trials", "qualify_trials"))
"""이어 할 때 바꿀 수 없는 플래그(예선은 실행 처음에만 뜻이 있다)."""
RESUME_GROW_FLAGS = (("--grow-start", "grow_start"), ("--grow-every", "grow_every"))
"""이어 할 때 바꿀 수 없는 자라는 뇌 플래그(v6.1: 학습 방법의 일부)."""


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


def _config_from_args(args: Any, base: TrainConfig | None = None, spec: EnvSpec | None = None) -> TrainConfig:
    """명령줄 인수로 설정을 만든다. base가 없으면 새 실행: 자동 재시작 같은 권장값(NEW_RUN_DEFAULTS)과, spec에 있음 유전자가
    있으면 구조 뒤집기 권장값(v2.1, for_new_run(spec)) 위에 인수. --structure-flip은 GA·담금질 둘 다의 확률이다."""
    given: dict[str, Any] = {}
    for key, attr in (
        ("method", "method"), ("run_seed", "seed"), ("popsize", "popsize"), ("sigma0", "sigma"),
        ("max_generations", "max_generations"), ("max_evaluations", "max_evaluations"), ("max_time", "max_time"),
        ("target_score", "target"), ("stagnation_generations", "stagnation"), ("trials", "trials"),
        ("validate_every", "validate_every"), ("diagonal_threshold", "diagonal_threshold"), ("ipop", "ipop"),
        ("restart_patience", "restart"), ("restart_from", "restart_from"), ("qualify_entrants", "qualify"),
        ("qualify_trials", "qualify_trials"), ("grow_start", "grow_start"), ("grow_every", "grow_every"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            given[key] = value
    if getattr(args, "no_restart", False):
        given["restart_patience"] = None
    # 새 실행이면 인수를 for_new_run에 넘긴다: 프리셋 권장값(editor.train)은 준 검증 주기 등과 맞지 않으면 빠져야 한다(설계 D31).
    c = base.to_dict() if base is not None else TrainConfig.for_new_run(spec, **given).to_dict()
    c.update(given)
    flip = getattr(args, "structure_flip", None)
    if flip is not None:
        c["ga_structure_flip"] = c["sa_structure_flip"] = flip
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


def _grow_text(config: TrainConfig, layout: Any) -> str | None:
    """자라는 뇌 새 실행의 시작 줄(v6.1 TRAIN_GROW_SUMMARY): 은닉 뉴런 합 처음 → 다 자란 크기, 간격. 크기 고정이면 None.
    처음 뉴런 수가 모든 은닉층을 이미 채우면(틀로 잘림) 자라지 않으므로 「두 배」를 약속하지 않고 TRAIN_GROW_SUMMARY_FULL
    (실험실 안내 줄 LAB_GROW_HINT_ALREADY와 같은 경우)."""
    if config.method != "grow" or not layout.brain.hidden:
        return None
    sizes = layout.brain.hidden
    start = sum(min(config.grow_start, s) for s in sizes)
    full = sum(sizes)
    if start >= full:
        return S.TRAIN_GROW_SUMMARY_FULL.format(full=full)
    return S.TRAIN_GROW_SUMMARY.format(start=start, full=full, every=config.grow_every)


def _grown_text(trainer: Trainer, row: dict[str, Any]) -> str | None:
    """이 세대 끝에 뇌가 자랐으면 한 줄(v6.1 CLI_TRAIN_GROWN), 아니면 None."""
    growth = getattr(trainer, "last_growth", None)
    if growth is None:
        return None
    return S.CLI_TRAIN_GROWN.format(gen=row["generation"], old=growth[0], new=growth[1])


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


def _progress(trainer: Trainer, state: dict[str, Any]) -> Any:
    def on_generation(row: dict[str, Any]) -> None:
        grown = _grown_text(trainer, row)
        if grown is not None:  # 자람(v6.1)도 진행 줄 간격과 상관없이 바로 알린다(학습 방법 교체보다 먼저 일어남: tell 안)
            say(grown)
        note = _switch_text(trainer, row)
        if note is not None:  # 학습 방법 교체는 진행 줄 간격과 상관없이 바로 알린다
            say(note)
        problem = trainer.observe_problem
        if problem and problem != state.get("observe"):
            # 관찰 기록 실패(D9): 학습은 그대로 이어진다. 처음 한 번(더 무거운 실패로 바뀌면 한 번 더) 바로 알린다
            state["observe"] = problem
            warn(problem)
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
    observe = False if getattr(args, "no_observe", False) else None  # None: 저장소가 있으니 켬
    pool: WorkerPool | None = None
    try:
        if args.resume is not None:
            from evolab.runs.store import RunStore

            fixed = [flag for flag, attr in RESUME_FIXED_FLAGS if getattr(args, attr, None) is not None]
            if fixed:
                warn(S.TRAIN_RESUME_QUALIFY_FIXED.format(flags=", ".join(fixed)))
                return 2
            if getattr(args, "structure_flip", None) is not None:  # 학습 방법의 일부(v2.1): 이어 할 때는 못 바꿈
                warn(S.TRAIN_RESUME_STRUCTURE_FIXED)
                return 2
            if any(getattr(args, attr, None) is not None for _flag, attr in RESUME_GROW_FLAGS):  # 학습 방법의 일부(v6.1)
                warn(S.TRAIN_RESUME_GROW_FIXED)
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
            trainer = Trainer.resume(store, pool, None, observe=observe, **overrides)
            store.append_event("resume", gen, workers=workers)
            say(S.TRAIN_RESUMED.format(gen=gen, path=store.root))
            note = structure_note(trainer.config, trainer.layout, trainer.spec)
            if note is not None:  # v2.0 GA·담금질 실행을 구조 환경에서 이어 감(뒤집기 꺼짐): 한 번 알린다
                warn(note)
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
            config = _config_from_args(args, spec=spec)
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
            problem = grow_problem(config, layout)
            if problem is not None:  # 자라는 뇌인데 은닉층이 없는 뇌(v6.1): 폴더를 만들기 전에 거절
                warn(S.TRAIN_CANNOT_START.format(reason=problem))
                return 2
            store = create_run(base_dir, env_text, spec, config, workers)
            pool = WorkerPool(workers)
            trainer = Trainer(env_text, config, pool, store, observe=observe)
            say(S.TRAIN_STARTED.format(name=spec.file.name, method=METHOD_NAMES.get(config.method), dim=layout.dim,
                                       pop=config.popsize, trials=trainer.trials, workers=workers, path=store.root))
            say(_stability_text(config))
            grow_line = _grow_text(config, layout)
            if grow_line is not None:  # v6.1 자라는 뇌: 시작 요약 한 줄(설계 5.5)
                say(grow_line)
            note = structure_range_note(config, spec)
            if note is not None:  # 범위 때문에 구조 뒤집기가 건드리지 않는 있음 칸(v2.1): 시작할 때 한 번
                warn(note)
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
    ends = ends_text(spec, res.end_events)
    mean = judge_mean(res.scores)
    if has_success(spec):
        say(S.JUDGE_RESULT.format(gen=gen, n=len(trials), success=success, rate=100.0 * success / len(trials),
                                  mean=mean, ends=ends))
    else:
        # 성공 판정이 없는 환경(글라이더): 성공률 대신 평균 점수와 진행 거리 블록(v2.0 설계 4.10, D21)
        say(S.JUDGE_RESULT_NO_SUCCESS.format(gen=gen, n=len(trials), mean=mean, ends=ends))
        for line in progress_lines(spec, res.blocks, failed=np.isnan(res.scores)):
            say(line)
    store.append_event("judge", gen, seeds=len(trials), success=success, mean_score=mean)
    return 0


def ends_text(spec: Any, end_events: list[str]) -> str:
    """종료 원인 글 「이름 수, 이름 수」(처음 나온 순서). 조립 실패 쌍(v2.0 설계 4.9)의 종료 사건은 빈 글이라
    RT_BUILD_FAILED_END로 쓴다(timeout이 늘 종료 목록에 있어 다른 쌍은 빈 글이 없다)."""
    labels = {"": S.RT_BUILD_FAILED_END} | {t.event: (t.label or t.event) for t in spec.terminations}
    counts: dict[str, int] = {}
    for e in end_events:
        counts[labels.get(e, e)] = counts.get(labels.get(e, e), 0) + 1
    return ", ".join(f"{k} {v}" for k, v in counts.items())


def judge_mean(scores: np.ndarray) -> float:
    """판정 평균 점수. 조립 실패 쌍(점수 NaN, v2.0 설계 4.9)은 빼고 평균한다(모두 실패면 NaN). 실패가 없으면 v1과 같은 계산.

    챔피언 비교 점수(Trainer._compare)와 같은 규칙(evaluate.mean_of_built)."""
    return mean_of_built(scores)


def has_success(spec: Any) -> bool:
    """종료 목록에 결과가 success인 줄이 있는가. 없으면(글라이더) 성공률 자리는 「—」(v2.0 설계 D21)."""
    return any(t.outcome == "success" for t in spec.terminations)


def progress_lines(spec: Any, blocks: np.ndarray, failed: np.ndarray | None = None) -> list[str]:
    """진행 거리 블록마다 판정 시드들의 중앙값·하위 10% 한 줄(CLI_JUDGE_PROGRESS). blocks: (쌍 수, 블록 수) 가중치 전 값.

    failed: 쌍마다 조립 실패 여부(점수 NaN). 실패 쌍의 블록은 0이라(evaluate.build_failed_result) 평균 점수(judge_mean)처럼
    빼고 센다. 없으면(None) 모든 쌍을 센다(실패가 없으면 같은 결과)."""
    out: list[str] = []
    values = np.asarray(blocks, dtype=np.float64).reshape(len(blocks), -1)
    if failed is not None:
        values = values[~np.asarray(failed, dtype=bool)]
    for k, b in enumerate(spec.file.scoring.blocks):
        if b.kind != "progress" or k >= values.shape[1]:
            continue
        col = values[:, k]
        col = col[np.isfinite(col)]
        if not col.size:
            continue
        out.append(S.CLI_JUDGE_PROGRESS.format(block=b.label or b.id, median=float(np.median(col)),
                                               p10=float(np.percentile(col, 10))))
    return out


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


BENCH_OBSERVE_ROUNDS = 3
"""bench가 단일 코어 평가를 관찰 끔·켬으로 번갈아 재는 횟수(중앙값으로 비교)."""
BENCH_WINDOWS: Final = (4, 8, 12, 16, 24)
"""bench 「평가 창」 줄이 재는 창 크기(v6 설계 D17). 사용자 PC에서 EVAL_WINDOW 기본값을 고르는 근거(H1, 되돌릴 지점 3)."""
BENCH_WINDOW_ROUNDS: Final = 2
"""창마다 잰 횟수. 회차마다 순서를 뒤집어(4→24, 24→4) 기계가 점점 빨라지거나 느려져도 한쪽으로 쏠리지 않게 한다."""
PLAYBACK_TAIL_S: Final = 0.5
"""관전 재생이 가장 긴 궤적 뒤에 더 보여 주는 시간(초). app/lab/view.py의 재생 끝 규칙(가장 긴 궤적 + 0.5초)과 같은 값."""


def bench_windows(version: str, env_text: str, layout: Any, genomes: np.ndarray, pairs: list[tuple[int, Any]],
                  windows: tuple[int, ...] = BENCH_WINDOWS,
                  rounds: int = BENCH_WINDOW_ROUNDS) -> tuple[dict[int, float], int]:
    """단일 코어 평가를 창 크기마다 번갈아 rounds번 잰 µs/스텝 중앙값과 가장 빠른 창(같으면 작은 창).

    결과는 창과 상관없이 같다(evaluate 머리말). 스텝 수는 첫 평가의 값을 쓴다."""
    times: dict[int, list[float]] = {w: [] for w in windows}
    steps = 0
    for k in range(rounds):
        for w in (windows if k % 2 == 0 else tuple(reversed(windows))):
            pool = WorkerPool(0, window=w)
            t0 = time.perf_counter()
            r = pool.evaluate(version, env_text, layout, genomes, pairs)
            times[w].append(time.perf_counter() - t0)
            steps = steps or int(r.steps.sum())
    us = {w: statistics.median(t) / max(steps, 1) * 1e6 for w, t in times.items()}
    best = min(windows, key=lambda w: (us[w], w))
    return us, best


def playback_seconds(spec: Any, steps: np.ndarray, trials_n: int) -> float:
    """무작위 유전자 1세대의 시험 0번 가장 긴 궤적 길이(초): 쌍이 개체마다 시험 trials_n개씩 놓인 평가의 스텝 수에서."""
    first = np.asarray(steps)[0::max(1, trials_n)]
    return float(first.max()) * float(spec.file.world.dt) if first.size else 0.0


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

    # 단일 코어(이 프로세스). 같은 평가를 관찰 기록을 끄고·켜고 번갈아 BENCH_OBSERVE_ROUNDS번씩 재서 중앙값으로 비교한다:
    # 한 번씩만 재면 회차 사이 흔들림(같은 평가가 ±10~15%, v1.2 측정)이 관찰 비용(학습에서 +0.4%)보다 훨씬 커서 부호까지
    # 틀린다(한 번씩 잰 v1.2 초판은 −5~−23%로 나왔다). 켠 쪽에는 학습기가 세대마다 더하는 것(틀 계산, 쌍별 관찰, 세대 끝 요약)을 넣고 쓰기(세대 대표 파일 하나에
    # 더하는 바이트)는 뺀다.
    single = WorkerPool(0)
    codes = [OBSERVE_POSITIONS if k == 0 else OBSERVE_ACTIONS for _ in range(pop) for k in range(trials_n)]
    single.evaluate(version, env_text, layout, genomes[:2], pairs[: 2 * trials_n])  # 환경 컴파일·첫 호출 시간 빼기

    def plain() -> Any:
        return single.evaluate(version, env_text, layout, genomes, pairs)

    def observed() -> Any:
        frame, world_key = observing.world_frame(spec, version, trials[0])
        r = single.evaluate(version, env_text, layout, genomes, pairs, observe=codes, frame=frame)
        agg = r.scores.reshape(pop, trials_n).mean(axis=1)
        observing.summarize(generation=1, X=genomes, agg=agg, res=r, trials_per=trials_n,
                            representative=int(np.argmax(agg)), learner=0, qualifying=False, layout=layout,
                            end_ids=[t.event for t in spec.terminations], trial0=trials[0], frame=frame,
                            world_key=world_key)
        return r

    off_times: list[float] = []
    on_times: list[float] = []
    for k in range(BENCH_OBSERVE_ROUNDS):  # 끔·켬, 켬·끔, 끔·켬 …: 기계가 점점 빨라지거나 느려져도 한쪽으로 쏠리지 않게
        for mode in (("off", "on") if k % 2 == 0 else ("on", "off")):
            t0 = time.perf_counter()
            if mode == "off":
                res = plain()
                off_times.append(time.perf_counter() - t0)
            else:
                res_obs = observed()
                on_times.append(time.perf_counter() - t0)
    steps = int(res.steps.sum())
    el = statistics.median(off_times)
    nn_share = res.nn_seconds / max(res.nn_seconds + res.sim_seconds, 1e-12)
    say(S.BENCH_SINGLE.format(window=EVAL_WINDOW, sps=steps / el, us=el / steps * 1e6, nn=100 * nn_share, steps=steps,
                              sec=el))
    el_obs = statistics.median(on_times)
    ratios = [100.0 * (off / on - 1.0) for off, on in zip(off_times, on_times, strict=True)]
    say(S.BENCH_OBSERVE.format(sps=int(res_obs.steps.sum()) / el_obs, pct=100.0 * (el / el_obs - 1.0),
                               rounds=BENCH_OBSERVE_ROUNDS, lo=min(ratios), hi=max(ratios)))
    us, best = bench_windows(version, env_text, layout, genomes, pairs)
    rows = S.BENCH_WINDOW_SEP.join(S.BENCH_WINDOW_ROW.format(w=w, us=us[w]) for w in BENCH_WINDOWS)
    say(S.BENCH_WINDOW.format(rows=rows, best=best, rounds=BENCH_WINDOW_ROUNDS))
    sec = playback_seconds(spec, res.steps, trials_n)
    say(S.BENCH_PLAYBACK.format(sec=sec, wait=sec + PLAYBACK_TAIL_S))
    from evolab.world.physics2d import episode as runtime

    # 깃발(cp* 함수를 불러옴)만이 아니라 이 환경의 Episode가 정말 모든 포인터를 얻는지(자체 점검 「빠른 계산 길」과 같은 판단)
    fast = runtime._CP_FAST and runtime.fastpath_in_use(spec, trials[0])
    say(S.BENCH_FAST_PATHS.format(state=S.BENCH_FAST_ON if fast else S.BENCH_FAST_OFF))

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
    parts = v2_parts(spec, layout, genomes, pairs)
    if parts is not None:
        movers, edges, clamped = parts
        say(S.BENCH_V2_PARTS.format(movers=movers, edges=edges, clamped=clamped))
    return 0


def v2_parts(spec: Any, layout: Any, genomes: np.ndarray, pairs: list[tuple[int, Any]]) -> tuple[int, int, int] | None:
    """bench의 v2 부품 줄(설계 D48): (정해진 움직임 물체 수, 공기역학 변 수, 한 스텝 상한이 걸린 스텝 수의 합).

    물체·변 수는 처음 몸으로 센다. 둘 다 없는 환경(v1 환경)이면 None(줄을 쓰지 않는다). 상한 발동은 같은 쌍들을 화면
    재생과 같은 길(Episode + 순전파, engine.replay와 같은 계산)로 다시 돌려 센다(조립할 수 없는 몸은 건너뜀). 공기역학이
    없으면 늘 0이라 돌리지 않는다. 시간 측정이 끝난 뒤에 부른다."""
    from evolab.policy.brain import unpack
    from evolab.world.physics2d.episode import Episode, EpisodeBuildError

    if not pairs:
        return None
    body = layout.body_values(genomes) if layout.n_body else None
    ep = Episode(spec, None, pairs[0][1])
    movers = sum(1 for g in ep.geometry().bodies if g.motion is not None)
    edges = ep.aero_edge_count
    if not movers and not edges:
        return None
    clamped = 0
    if edges:
        obs = np.zeros((1, layout.brain.n_in), dtype=np.float64)
        for i, trial in pairs:
            g = np.ascontiguousarray(genomes[i:i + 1], dtype=np.float64)
            brain = unpack(layout.brain, layout.brain_params(g))
            try:
                ep = Episode(spec, body[i] if body is not None else None, trial)
            except EpisodeBuildError:
                continue
            while not ep.done:
                obs[0] = ep.observe()
                ep.step(layout.to_actions(brain.forward(obs))[0])
            clamped += ep.aero_clamped
    return movers, edges, clamped


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

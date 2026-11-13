"""「늘 올바른 파일」의 세 층 (6.2.0, 최종 설계 D38, Qt 없음).

- L1 quick_check: 구조(검사 오류 0·컴파일). 안내 창이 닿는 모든 조리법에서 성립해야 한다(T15·T23).
- L2 alive: 죽지 않음. 행동 0으로 전시 시험 2초를 돌려 0스텝에 끝나지 않는지, 선수의 상태에 기대는 점수 항이 하나 이상인지,
  입력과 출력이 하나 이상인지(뇌 없이 몸 수치만 진화하는 몸은 출력 0도 된다)를 본다. 「실험실로 보내기」의 관문이고 까닭과 한 번에
  고치는 길(fix)이 붙는다. 「가만히 두면 2초 서 있음」은 관문이 아니다(still_end는 알림뿐: 보조 100의 몸도 행동 0이면 넘어진다).
- L3 probe: 무작위 뇌 n개의 점수가 서로 다른가. 알림뿐이다.

selftest_check는 내장 조리법 전부를 만들어 L1·L2를 보고 달리기 한 세대(개체 8)를 돌린다(자체 점검 「프리셋」: selftest.py가
등록). 난수는 시드를 준 생성기만 쓴다(numpy 전역 난수 없음).
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evolab.app import strings as S
from evolab.spec import model as m
from evolab.spec.compiled import EnvFileError, EnvSpec, Issue
from evolab.spec.io import compile_env

ALIVE_SECONDS = 2.0
"""L2가 행동 0으로 돌리는 시간(초)."""
PROBE_N = 16
PROBE_SCALE = 0.5
"""L3의 무작위 유전자 = PROBE_SCALE x 표준 정규(default_rng([7, seed]))."""
FLAT_SPREAD = 1e-9
"""점수가 이 폭 안이면 「모두 같다」."""
SELFTEST_POPSIZE = 8
SELFTEST_RECIPE = "running"


@dataclass(frozen=True)
class Quick:
    ok: bool
    errors: list[Issue] = field(default_factory=list)
    inputs: int = 0
    outputs: int = 0
    body_genes: int = 0
    spec: EnvSpec | None = None
    """컴파일한 환경(ok일 때). alive·probe에 넘긴다."""


def quick_check(data: Mapping[str, Any] | str | bytes) -> Quick:
    """L1: 읽고 검사하고 컴파일한다. 오류가 있으면 ok=False와 오류 목록."""
    try:
        spec = compile_env(copy.deepcopy(dict(data)) if isinstance(data, Mapping) else data)
    except EnvFileError as exc:
        return Quick(False, [i for i in exc.issues if i.severity == "error"] or list(exc.issues))
    return Quick(True, [], len(spec.inputs), len(spec.outputs), spec.body_param_count, spec)


@dataclass(frozen=True)
class Alive:
    ok: bool
    reason: str
    """ok가 아니면 까닭(PRESET_ALIVE_*·PRESET_PROBE_BUILD), ok면 빈 글."""
    fix: str | None
    """한 번에 고치는 길: "add_rule"(점수 규칙 더하기) | "move_part"(시작 자리 옮기기) | None."""
    still_end: tuple[float, str, str] | None
    """행동 0으로 ALIVE_SECONDS 안에 끝났으면 (끝난 시각, 종료 줄 이름, 사건 id). 알림뿐(ok와 무관)."""

    def notice(self) -> str:
        """still_end의 알림 글(없으면 빈 글)."""
        if self.still_end is None:
            return ""
        return S.PRESET_ALIVE_STILL_END.format(t=self.still_end[0], end=self.still_end[1])


_STATE_BLOCKS = frozenset({"progress", "distance_to", "final_speed", "tilt", "energy", "fuel_used", "contact_time",
                           "smoothness", "checkpoints", "lap_time", "expr"})
"""그 값이 선수의 움직임에 따라 달라지는 점수 블록 종류."""


def has_state_score(env: m.EnvFile) -> bool:
    """선수의 상태에 기대는 점수 항이 하나 이상인가.

    점수 수식이 있거나, 위 종류의 블록이 있거나, 시간 끝 밖의 사건(내장 out_of_bounds 포함)을 세는 사건 블록이 있거나, 버틴
    시간 블록이 있고 시간 끝 밖의 종료 줄이 있으면(끝나는 시각이 움직임에 달림) 그렇다. 가중치가 0인 블록은 세지 않는다.
    """
    if env.scoring.formula:
        return True
    ends = [t for t in env.termination if t.event != "timeout"]
    for b in env.scoring.blocks:
        if float(b.weight) == 0.0:
            continue
        if b.kind in _STATE_BLOCKS:
            return True
        if isinstance(b, m.EventBlock) and b.event != "timeout":
            return True
        if isinstance(b, m.SurvivalBlock) and ends:
            return True
    return False


def _io_ok(spec: EnvSpec) -> bool:
    if spec.outputs:
        return bool(spec.inputs)
    return spec.body_param_count > 0


def _rule_name(spec: EnvSpec, event: str) -> str:
    """그 사건을 만든 규칙의 이름(규칙 표), 없으면 종료 줄·사건의 이름, 그것도 없으면 사건 id."""
    from evolab.presets.rules import find_rules
    from evolab.spec.io import canonical_dict

    try:
        uses, _ = find_rules(canonical_dict(spec.file))
    except Exception:  # noqa: BLE001 - 이름 찾기는 알림용이라 실패해도 id로 둔다
        uses = []
    for u in uses:
        if u.event == event:
            return u.title
    for t in spec.terminations:
        if t.event == event and t.label:
            return t.label
    for e in spec.file.events:
        if e.id == event and e.label:
            return e.label
    return event


def alive(spec: EnvSpec) -> Alive:
    """L2: 행동 0으로 전시 시험 ALIVE_SECONDS초. 입출력·점수 항·조립·0스텝 종료만 관문이다."""
    from evolab.engine.seeds import display_trial
    from evolab.world.physics2d.episode import Episode, EpisodeBuildError

    if not _io_ok(spec):
        return Alive(False, S.PRESET_ALIVE_NO_IO, None, None)
    if not has_state_score(spec.file):
        return Alive(False, S.PRESET_ALIVE_NO_SCORE, "add_rule", None)
    try:
        ep = Episode(spec, None, display_trial())
    except EpisodeBuildError:
        return Alive(False, S.PRESET_PROBE_BUILD, None, None)
    zero = [0.0] * len(spec.outputs)
    steps = max(1, int(math.ceil(ALIVE_SECONDS / spec.file.world.dt - 1e-9)))
    for _ in range(steps):
        if ep.done:
            break
        ep.step(zero)
    if not ep.done:
        return Alive(True, "", None, None)
    res = ep.result()
    if res.end_event == "timeout":
        return Alive(True, "", None, None)
    if res.steps <= 1:
        return Alive(False, S.PRESET_ALIVE_INSTANT.format(rule=_rule_name(spec, res.end_event)), "move_part",
                     (float(res.time), res.end_label or res.end_event, res.end_event))
    return Alive(True, "", None, (float(res.time), res.end_label or res.end_event, res.end_event))


@dataclass(frozen=True)
class Probe:
    ok: bool
    reason: str
    """PRESET_PROBE_OK·PRESET_PROBE_FLAT·PRESET_PROBE_BUILD를 채운 글."""
    score_spread: float
    """무작위 뇌 점수의 최댓값 − 최솟값(조립된 것만)."""
    first_end_s: float
    """가장 일찍 끝난 시험의 시각(초)."""
    build_failures: int


def probe(spec: EnvSpec, *, n: int = PROBE_N, seed: int = 0) -> Probe:
    """L3: 무작위 뇌 n개(유전자 = 0.5 x 표준 정규, default_rng([7, seed]))를 전시 시험으로 끝까지 한 번씩. 알림뿐."""
    from evolab.engine.evaluate import evaluate
    from evolab.engine.seeds import display_trial
    from evolab.policy.genome import layout_for
    from evolab.world.physics2d.episode import EpisodeBuildError

    layout = layout_for(spec)
    rng = np.random.default_rng([7, int(seed)])
    genomes = PROBE_SCALE * rng.standard_normal((int(n), layout.dim))
    trial = display_trial()
    try:
        res = evaluate(spec, layout, genomes, [(i, trial) for i in range(int(n))])
    except EpisodeBuildError:
        return Probe(False, S.PRESET_PROBE_BUILD, 0.0, 0.0, int(n))
    built = np.isfinite(res.scores)
    failures = int(np.count_nonzero(~built))
    if not built.any():
        return Probe(False, S.PRESET_PROBE_BUILD, 0.0, 0.0, failures)
    scores = res.scores[built]
    spread = float(np.max(scores) - np.min(scores))
    first = float(np.min(res.steps[built])) * float(spec.file.world.dt)
    if spread <= FLAT_SPREAD:
        return Probe(False, S.PRESET_PROBE_FLAT.format(n=int(n)), spread, first, failures)
    return Probe(True, S.PRESET_PROBE_OK.format(n=int(n)), spread, first, failures)


def _generation(text: str) -> None:
    """달리기 한 세대(개체 SELFTEST_POPSIZE, 이 프로세스 안 평가, 저장소 없음)."""
    from evolab.engine.pool import WorkerPool
    from evolab.engine.trainer import TrainConfig, Trainer

    config = TrainConfig(run_seed=1, popsize=SELFTEST_POPSIZE, trials=1, validate_every=10**9, max_generations=1)
    with WorkerPool(0) as pool:
        trainer = Trainer(text, config, pool, None, observe=False)
        trainer.step()


def selftest_check() -> tuple[bool, str]:
    """자체 점검 「프리셋」: 틀 전부를 읽고, 내장 조리법 전부를 만들어 L1·L2를 보고, 달리기 한 세대를 돌린다.

    (통과, 글). 실패하면 그 틀·조리법의 이름이 든 글(SELFTEST_ERR_PRESET)."""
    from evolab.presets.library import Library
    from evolab.presets.recipe import RecipeError, build, builtin_names, load_builtin
    from evolab.presets.template import TemplateError
    from evolab.spec.io import dump_env, parse_env
    from evolab import paths

    try:
        lib = Library(paths.template_dirs()[:1])
    except TemplateError as exc:
        return False, S.SELFTEST_ERR_PRESET.format(name="templates", reason=exc)
    names = builtin_names()
    worst = 0.0
    texts: dict[str, str] = {}
    for name in names:
        t0 = time.perf_counter()
        try:
            data = build(load_builtin(name), lib)
        except RecipeError as exc:
            return False, S.SELFTEST_ERR_PRESET.format(name=name, reason=exc)
        quick = quick_check(data)
        if not quick.ok or quick.spec is None:
            first = quick.errors[0] if quick.errors else ""
            return False, S.SELFTEST_ERR_PRESET.format(name=name, reason=first)
        verdict = alive(quick.spec)
        if not verdict.ok:
            return False, S.SELFTEST_ERR_PRESET.format(name=name, reason=verdict.reason)
        texts[name] = dump_env(parse_env(data))
        worst = max(worst, (time.perf_counter() - t0) * 1000.0)
    if SELFTEST_RECIPE in texts:
        try:
            _generation(texts[SELFTEST_RECIPE])
        except Exception as exc:  # noqa: BLE001 - 자체 점검은 까닭을 글로 돌려준다
            return False, S.SELFTEST_ERR_PRESET.format(name=SELFTEST_RECIPE, reason=exc)
    return True, S.SELFTEST_PRESETS_DETAIL.format(n=len(names), t=len(lib.all()), ms=worst)


__all__ = ["ALIVE_SECONDS", "Alive", "Probe", "Quick", "alive", "has_state_score", "probe", "quick_check",
           "selftest_check"]

"""나란히 경주의 자료 (명세: 비교 > 나란히 경주, 설계 4.6·D26~D31). Qt 없이 쓴다.

- 코스 n(1~32)은 비교 평가 세트의 n번째 시험(seeds.compare_trials()[n-1])이다. 챔피언 선정(검증 세트)과 학습에 쓰지
  않은 시드라 어느 칸에도 유리하지 않다(명세: 공정한 비교). 「다른 코스」는 다음 번호(32 다음 1)라 난수가 없다.
- 칸마다 그 유전자를 학습한 환경 버전으로 달린다(D28): 세대 g의 유전자는 적용 세대(끝낸 세대 수)가 g−1 이하인 마지막
  환경 기록(실험실 재생과 같은 규칙). 파일에 적힌 env_version이 그 환경의 해시와 다르면(드문 경우) run.json의 모든
  환경 버전에서 해시가 같은 것을 쓴다. 모든 칸의 환경 해시가 같을 때만 순위를 매기고 한 화면에 겹쳐 볼 수 있다.
- 누구를(D29): 「마지막 챔피언」은 마지막 통계 줄의 champion_generation, 「같은 학습 시험 수일 때의 챔피언」은
  N = 고른 실행들의 마지막 누적 학습 시험(evaluations)의 최솟값에서 실행마다 evaluations ≤ N인 마지막 통계 줄의
  champion_generation. 그때 챔피언이 없으면 그 줄 세대의 세대 대표로 달리고 알린다.
- 정확성 고리: 같은 실행의 챔피언을 자기 환경에서 코스 1..32로 달린 점수 32개를 np.array(...).mean()하면 그 챔피언
  파일의 compare_score와 비트 단위로 같다. Trainer._compare가 같은 32개 시험을 같은 순서로 float(scores.mean())하고,
  재생 점수는 워커 점수와 비트 단위로 같기 때문이다(tests/test_race.py가 지킨다).
"""

from __future__ import annotations

import csv
import math
import os
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import numpy as np

from evolab.app import strings as S
from evolab.engine import seeds
from evolab.runs.catalog import env_versions_until
from evolab.runs.store import PENDING_SUFFIX, STATS_FILE, RunStore, RunStoreError
from evolab.spec.compiled import EnvSpec
from evolab.spec.io import EnvFileError, compile_env, env_hash
from evolab.world.physics2d.episode import Trial

MAX_LANES: Final = 6
COURSES: Final = seeds.COMPARE_SET
"""코스 수(32): 비교 평가 세트 크기."""
WHO_MODES: Final = ("last", "budget")
GENERATION_FRACTIONS: Final = ((1, 4), (1, 2), (3, 4))
"""세대끼리 경주: 1세대, ⌈G/4⌉, ⌈G/2⌉, ⌈3G/4⌉, G세대 대표."""

_READ_ERRORS = (RunStoreError, OSError, ValueError, KeyError, csv.Error, UnicodeDecodeError, EnvFileError)
_SPEC_CACHE_SIZE = 16
_STATS_CACHE_SIZE = 16


@dataclass(frozen=True, eq=False)
class RaceEntry:
    """경주 칸 하나."""

    label: str
    """칸 제목(RACE_LANE_RUN·RACE_LANE_RUN_ENV 또는 RACE_LANE_GEN·RACE_LANE_NOW)."""
    sub: str
    """칸 둘째 줄(RACE_LANE_CHAMPION·RACE_LANE_REP_INSTEAD). 없으면 빈 글."""
    color: str
    env_text: str
    """그 유전자를 학습한 환경 버전의 글(D28)."""
    env_hash: str
    """env_hash(spec.file)[:16]: 같은 환경 판정(챔피언 파일의 env_version과 같은 값)."""
    genome: np.ndarray
    """float64 1차원(읽기 전용 사본)."""
    generation: int
    """유전자를 고른 세대(챔피언 세대 또는 세대 대표 세대)."""
    run: Path | None = None
    """유전자를 읽은 실행 폴더(경주 영상의 기본 저장 폴더). 모르면 None."""

    def __post_init__(self) -> None:
        g = np.array(self.genome, dtype=np.float64).reshape(-1)  # 사본: 부르는 쪽 배열이 바뀌어도 칸은 그대로
        g.setflags(write=False)
        object.__setattr__(self, "genome", g)


# ---------------------------------------------------------------------------
# 코스
@lru_cache(maxsize=1)
def _courses() -> tuple[Trial, ...]:
    return tuple(seeds.compare_trials())


def course_trial(n: int) -> Trial:
    """코스 n(1~32) = 비교 평가 세트의 n번째 시험(seeds.compare_trials()[n-1])."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 1 <= int(n) <= COURSES:
        raise ValueError(f"course must be 1..{COURSES}, got {n!r}")
    return _courses()[int(n) - 1]


def next_course(n: int) -> int:
    """「다른 코스」: 다음 번호, 32 다음은 1."""
    return int(n) % COURSES + 1


def lane_grid(n: int) -> tuple[int, int]:
    """칸 수 → (줄, 열): 1 → 1×1, 2 → 1×2, 3 → 1×3, 4 → 2×2, 5·6 → 2×3."""
    n = max(1, int(n))
    if n <= 3:
        return 1, n
    if n == 4:
        return 2, 2
    return math.ceil(n / 3), 3


# ---------------------------------------------------------------------------
# 환경
_spec_lock = threading.Lock()
_spec_cache: OrderedDict[str, EnvSpec] = OrderedDict()


def compile_cached(env_text: str) -> EnvSpec:
    """같은 환경 글은 한 번만 컴파일한다(경주 준비 스레드와 화면 스레드가 같이 쓴다). 최근 16개를 기억한다."""
    with _spec_lock:
        spec = _spec_cache.get(env_text)
        if spec is not None:
            _spec_cache.move_to_end(env_text)
            return spec
    spec = compile_env(env_text)
    with _spec_lock:
        _spec_cache[env_text] = spec
        while len(_spec_cache) > _SPEC_CACHE_SIZE:
            _spec_cache.popitem(last=False)
    return spec


def env_hash_of(env_text: str) -> str:
    """환경 글의 env_hash 앞 16글자(챔피언 파일 env_version과 같은 값)."""
    return env_hash(compile_cached(env_text).file)[:16]


def _versions(store: RunStore) -> list[dict[str, Any]]:
    versions = store.read_run_info().get("env_versions") or []
    return [v for v in versions if isinstance(v, dict)]


def env_text_at(store: RunStore, generation: int) -> str:
    """세대 generation을 계산할 때 쓴 환경 글: 적용 세대가 generation−1 이하인 마지막 기록(없으면 첫 기록).

    실험실 재생(LabView.env_text_for_generation)과 같은 규칙이다. 환경 기록이 없으면 RunStoreError.
    """
    usable = env_versions_until(store.read_run_info(), int(generation) - 1)  # 체크포인트 generation−1까지의 환경
    if not usable:
        raise RunStoreError(S.RUNS_ENV_VERSION_MISSING.format(version="-"))
    return store.env_text(str(usable[-1].get("id")))


def _env_for(store: RunStore, generation: int, file_version: str | None) -> tuple[str, str]:
    """(환경 글, 해시). 파일의 env_version이 env_text_at의 해시와 다르면 run.json의 모든 버전에서 같은 해시를 찾는다."""
    text = env_text_at(store, generation)
    h = env_hash_of(text)
    if not file_version or file_version == h:
        return text, h
    seen: set[str] = set()
    for v in _versions(store):
        vid = str(v.get("id"))
        if vid in seen:
            continue
        seen.add(vid)
        try:
            other = store.env_text(vid)
            if env_hash_of(other) == file_version:
                return other, file_version
        except _READ_ERRORS:
            continue
    return text, h  # 같은 해시가 없으면 그 세대 규칙대로(파일이 손으로 옮겨진 경우 등)


def _file_version(data: dict[str, np.ndarray]) -> str | None:
    v = data.get("env_version")
    if v is None:
        return None
    s = str(np.asarray(v)[()])
    return s or None


def same_env(entries: Sequence[RaceEntry]) -> bool:
    """모든 칸의 환경 해시가 같은가(칸이 없거나 하나면 참)."""
    return len({e.env_hash for e in entries}) <= 1


def ranks(entries: Sequence[RaceEntry], scores: Sequence[float]) -> list[tuple[int, bool]] | None:
    """칸마다 (순위, 공동인가). 점수 내림차순, 같은 점수는 같은 순위(공동 n위, 다음 순위는 건너뜀).

    환경이 섞이면 None(같은 코스 번호라도 세계가 다를 수 있어 순위를 매기지 않는다). 유한하지 않은 점수는 꼴찌로 본다.
    """
    if len(entries) != len(scores):
        raise ValueError(f"{len(entries)} entries but {len(scores)} scores")
    if not same_env(entries):
        return None
    vals = [float(s) if math.isfinite(float(s)) else -math.inf for s in scores]
    return [(1 + sum(1 for w in vals if w > v), sum(1 for w in vals if w == v) > 1) for v in vals]


# ---------------------------------------------------------------------------
# 실행 경주: 누구를
def _num(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


@dataclass
class _Run:
    path: Path
    color: str
    store: RunStore
    info: dict[str, Any]
    rows: list[dict[str, str]]

    @property
    def env_name(self) -> str:
        name = self.info.get("env_name")
        return name if isinstance(name, str) and name else self.path.name

    @property
    def config(self) -> dict[str, Any]:
        config = self.info.get("config")
        return config if isinstance(config, dict) else {}

    @property
    def method_name(self) -> str:
        method = self.info.get("method")
        if not isinstance(method, str) or not method:
            method = str(self.config.get("method", ""))
        return S.METHOD_NAMES.get(method, method)

    @property
    def seed(self) -> str:
        seed = self.config.get("run_seed")
        return str(seed) if isinstance(seed, int) and not isinstance(seed, bool) else S.CMP_NONE

    def last_evaluations(self) -> float:
        for r in reversed(self.rows):
            v = _num(r.get("evaluations"))
            if math.isfinite(v):
                return v
        return math.nan


_stats_lock = threading.Lock()
_stats_cache: OrderedDict[str, tuple[Any, list[dict[str, str]]]] = OrderedDict()


def _file_key(path: Path) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _stats_rows(store: RunStore) -> list[dict[str, str]]:
    """통계 줄(세대 번호가 있는 줄만). stats.csv와 옆 파일의 (수정 시각, 크기)가 같으면 앞서 읽은 것을 다시 쓴다
    (경주 창을 열 때 칸 고르기와 「같은 학습 시험 수」 풍선 도움말이 같은 파일을 두 번 읽지 않게)."""
    root = store.root
    key = (_file_key(root / STATS_FILE), _file_key(root / (STATS_FILE + PENDING_SUFFIX)))
    name = str(root)
    with _stats_lock:
        hit = _stats_cache.get(name)
        if hit is not None and hit[0] == key:
            _stats_cache.move_to_end(name)
            return hit[1]
    rows = [r for r in store.read_stats() if math.isfinite(_num(r.get("generation")))]
    with _stats_lock:
        _stats_cache[name] = (key, rows)
        while len(_stats_cache) > _STATS_CACHE_SIZE:
            _stats_cache.popitem(last=False)
    return rows


def _open_run(path: Path, color: str) -> _Run:
    store = RunStore.open(Path(path))
    info = store.read_run_info()
    return _Run(Path(path), color, store, info, _stats_rows(store))


def _reason(exc: BaseException) -> str:
    return str(exc).strip() or type(exc).__name__


def common_budget(runs: Sequence[tuple[Path, str]]) -> int | None:
    """「같은 학습 시험 수」의 N: 읽을 수 있는 실행들 마지막 누적 학습 시험의 최솟값(앞 MAX_LANES개). 없으면 None."""
    values: list[float] = []
    for path, color in list(runs)[:MAX_LANES]:
        try:
            v = _open_run(path, color).last_evaluations()
        except _READ_ERRORS:
            continue
        if math.isfinite(v):
            values.append(v)
    return int(min(values)) if values else None


class _Skip(Exception):
    """이 실행은 칸이 될 수 없다(사용자에게 보여 줄 이유)."""


def _pick(run: _Run, who: str, budget: float | None, name: str) -> tuple[RaceEntry, str | None]:
    """실행 하나의 칸과 알림 글(그때 챔피언이 없어 세대 대표로 달릴 때)."""
    rows = run.rows
    if not rows:
        raise _Skip(S.RACE_NO_GENERATIONS)
    row = rows[-1]
    if who == "budget" and budget is not None:
        fit = [r for r in rows if _num(r.get("evaluations")) <= budget]
        # 첫 세대부터 N을 넘는 실행(세대가 아주 큰 방법)은 가장 이른 시점으로 본다
        row = fit[-1] if fit else rows[0]
    gen = int(_num(row.get("generation")))
    champ_value = _num(row.get("champion_generation"))
    available = run.store.champion_generations("champion")
    if math.isfinite(champ_value):
        champ = int(champ_value)
    else:  # 열이 없는 예전 통계: 그 세대까지의 마지막 챔피언 파일
        champ = max((g for g in available if g <= gen), default=-1)
    if champ >= 1 and champ not in available:  # 파일이 빠졌으면 그 앞의 마지막 챔피언 파일
        champ = max((g for g in available if g <= champ), default=-1)
    evals_by_gen = {int(_num(r.get("generation"))): _num(r.get("evaluations")) for r in rows}
    note: str | None = None
    if champ >= 1:
        data = run.store.load_champion(champ, "champion")
        used = champ
        evals = evals_by_gen.get(champ, _num(row.get("evaluations")))
        sub = S.RACE_LANE_CHAMPION.format(gen=champ, evals=int(evals) if math.isfinite(evals) else 0)
    else:
        data = run.store.load_champion(gen, "representative")
        used = gen
        sub = S.RACE_LANE_REP_INSTEAD.format(gen=gen)
        note = S.RACE_NO_CHAMPION.format(name=name, gen=gen)
    text, h = _env_for(run.store, used, _file_version(data))
    entry = RaceEntry(label=name, sub=sub, color=run.color, env_text=text, env_hash=h, genome=data["genome"],
                      generation=used, run=run.path)
    return entry, note


def entries_from_runs(runs: Sequence[tuple[Path, str]], who: str) -> tuple[list[RaceEntry], list[str], int | None]:
    """비교 화면의 실행 경주 칸. runs = (실행 폴더, 그래프 색) 목록(그래프 순서).

    돌려주는 것: (칸들, 알림 글들, 같은 학습 시험 수 N(who="budget"일 때만, 아니면 None)).
    7개 이상이면 앞 6개 + RACE_TOO_MANY. 읽을 수 없는 실행은 빼고 RACE_SKIPPED.
    """
    if who not in WHO_MODES:
        raise ValueError(f"unknown race mode {who!r}")
    warnings: list[str] = []
    runs = list(runs)
    if len(runs) > MAX_LANES:
        warnings.append(S.RACE_TOO_MANY.format(max=MAX_LANES))
        runs = runs[:MAX_LANES]
    loaded: list[_Run] = []
    for path, color in runs:
        try:
            loaded.append(_open_run(path, color))
        except _READ_ERRORS as exc:
            warnings.append(S.RACE_SKIPPED.format(name=Path(path).name, reason=_reason(exc)))
    budget: int | None = None
    if who == "budget":
        values = [v for r in loaded if math.isfinite(v := r.last_evaluations())]
        budget = int(min(values)) if values else None
    several_envs = len({r.env_name for r in loaded}) > 1
    entries: list[RaceEntry] = []
    for r in loaded:
        if several_envs:
            name = S.RACE_LANE_RUN_ENV.format(env=r.env_name, method=r.method_name, seed=r.seed)
        else:
            name = S.RACE_LANE_RUN.format(method=r.method_name, seed=r.seed)
        try:
            entry, note = _pick(r, who, budget, name)
        except _Skip as exc:
            warnings.append(S.RACE_SKIPPED.format(name=name, reason=str(exc)))
            continue
        except _READ_ERRORS as exc:
            warnings.append(S.RACE_SKIPPED.format(name=name, reason=_reason(exc)))
            continue
        if note:
            warnings.append(note)
        entries.append(entry)
    return entries, warnings, budget


# ---------------------------------------------------------------------------
# 세대끼리 경주
def race_generations(last_gen: int) -> list[int]:
    """1, ⌈G/4⌉, ⌈G/2⌉, ⌈3G/4⌉, G (같은 세대는 하나만, 오름차순)."""
    g = int(last_gen)
    if g < 1:
        return []
    picks = {1, g} | {-(-g * a // b) for a, b in GENERATION_FRACTIONS}
    return sorted(picks)


def entries_from_generations(run_dir: Path, last_gen: int, champion: tuple[int, np.ndarray] | None,
                             colors: Sequence[str]) -> tuple[list[RaceEntry], list[str]]:
    """실험실 「세대끼리 경주」의 칸: race_generations(last_gen)의 세대 대표 + 지금 챔피언(있으면). 최대 6칸.

    세대 대표가 2세대 이상 없으면 ([], [RACE_NEED_GENS]). 읽을 수 없는 파일은 빼고 RACE_SKIPPED.
    색은 칸 순서대로 colors를 돌려 쓴다.
    """
    if int(last_gen) < 2:
        return [], [S.RACE_NEED_GENS]
    palette = list(colors) or ["#ffffff"]
    warnings: list[str] = []
    try:
        store = RunStore.open(Path(run_dir))
    except _READ_ERRORS as exc:
        return [], [S.RACE_FAILED.format(reason=_reason(exc))]
    picked: list[tuple[str, int, np.ndarray, str | None]] = []
    for g in race_generations(last_gen):
        name = S.RACE_LANE_GEN.format(gen=g)
        try:
            data = store.load_champion(g, "representative")
            picked.append((name, g, data["genome"], _file_version(data)))
        except _READ_ERRORS as exc:
            warnings.append(S.RACE_SKIPPED.format(name=name, reason=_reason(exc)))
    if champion is not None:
        cg, genome = int(champion[0]), np.asarray(champion[1], dtype=np.float64)
        version = None
        try:
            if cg in store.champion_generations("champion"):
                version = _file_version(store.load_champion(cg, "champion"))
        except _READ_ERRORS:
            version = None
        picked.append((S.RACE_LANE_NOW.format(gen=cg), max(1, cg), genome, version))
    entries: list[RaceEntry] = []
    for name, g, genome, version in picked[:MAX_LANES]:
        try:
            text, h = _env_for(store, g, version)
        except _READ_ERRORS as exc:
            warnings.append(S.RACE_SKIPPED.format(name=name, reason=_reason(exc)))
            continue
        entries.append(RaceEntry(label=name, sub="", color=palette[len(entries) % len(palette)], env_text=text,
                                 env_hash=h, genome=genome, generation=g, run=Path(run_dir)))
    return entries, warnings


__all__ = [
    "COURSES", "MAX_LANES", "WHO_MODES", "RaceEntry", "common_budget", "compile_cached", "course_trial",
    "entries_from_generations", "entries_from_runs", "env_hash_of", "env_text_at", "lane_grid", "next_course",
    "race_generations", "ranks", "same_env",
]

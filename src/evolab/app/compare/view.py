"""비교 화면 (명세: 화면과 조작 > 비교, 학습 방법 > 공정한 비교).

- 왼쪽: 실행 목록(catalog.list_runs, 최근 것부터). 반복 실행 묶음(run.json의 "group": {"id", "name", "index",
  "count"})은 묶음 줄 아래에 실행을 모아 보이고, 묶음 체크박스로 묶음 전체를 고른다. 기록 화면은 고른 실행을
  set_selected_runs로 넘긴다. 읽을 수 없는 폴더는 회색으로 보이고 고를 수 없다.
- 가운데: 고른 실행을 그래프 하나에 겹쳐 그린다. 세로축 기본값은 챔피언을 비교 평가 세트(선정에 쓰지 않은 시드
  32개)로 잰 점수(compare_score)다. 챔피언이 바뀔 때만 새로 재므로 앞 값을 이어 가는 계단선으로 그리고, 처음 잴
  때까지는 비어 있다. 성공률(compare_success, %), 챔피언 검증 점수(champion_score), 세대 최고 학습 점수(score_best)로
  바꿀 수 있다. 가로축 기본값은 누적 학습 시험 횟수(방법마다 세대 크기가 달라서)이고 경과 시간과 세대로 바꿀 수 있다.
- 묶음은 평균 선과 최소~최대 띠로 그린다. 묶음 실행들의 가로축 값을 모두 합친 공통 격자에 실행마다 값을 옮긴다
  (계단 값은 그 지점까지의 마지막 값, 세대 최고 점수는 선형 보간. 실행의 첫 값 앞과 마지막 줄 뒤는 값 없음).
  모든 실행에 값이 있는 구간은 굵은 실선 평균과 띠로 그린다. 일부 실행에만 값이 있는 구간(먼저 끝난 실행이 있거나
  아직 도는 실행이 있을 때)은 값이 있는 실행들만의 평균을 가는 점선으로 잇고 띠는 그리지 않는다. 묶음 평균을 끄면
  묶음 실행도 하나씩 가는 선으로 그린다.
- 아래: 표. 최종 점수(마지막 compare_score, 묶음은 평균 ± 표본 표준편차), 목표 점수 도달까지 시험 횟수(compare_score가
  처음 목표 이상이 된 줄의 누적 시험 횟수, 못 닿으면 '-', 묶음은 닿은 실행들의 평균과 닿은 수), 소요 시간(마지막
  elapsed, 묶음은 평균), 성공률(마지막 compare_success). 목표 점수는 사용자가 정하고, 정하기 전에는 고른 실행의
  환경 파일 evaluation.target_score(없으면 학습 설정의 목표 점수)를 쓴다.
- 통계는 실행마다 stats.csv를 읽어 숫자 열로 바꿔 둔다(파일 크기·수정 시각이 같으면 다시 읽지 않는다). 없는 열,
  빈칸, 읽을 수 없는 파일은 NaN으로 읽어 예전 실행이나 쓰는 중인 실행도 화면이 멈추지 않는다.
- 반복 실행(repeat.py)이 도는 동안 새 실행 폴더가 생기면 목록에 바로 보이고, 묶음을 고른 상태면 새 실행도 고른다.
  그래프는 세대가 끝날 때마다(1초에 한 번까지) 다시 읽는다.
- 「나란히 경주」(v1.2): 고른 실행(묶음은 구성 실행) 2개 이상의 챔피언을 같은 코스에서 나란히 재생하는 경주 창
  (app/race)을 연다. 칸 색은 그래프에서 그 실행에 준 색(묶음 구성 실행은 그 묶음 색), 7개 이상이면 앞 6개만.
"""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPen, QPixmap, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from evolab import paths
from evolab.app import strings as S
from evolab.app import style
from evolab.app.compare.repeat import RepeatDialog, RepeatPlan, RepeatRunner
from evolab.app.lab.dashboard import (
    AXIS_WIDTH,
    GLYPH_DASH,
    GLYPH_FILL,
    GLYPH_LINE,
    Theme,
    fmt_value,
    theme_for,
    to_float,
)
from evolab.runs.catalog import RunSummary, list_runs
from evolab.runs.store import PENDING_SUFFIX, STATS_FILE, RunStore, RunStoreError

X_AXES: tuple[str, ...] = ("evaluations", "elapsed", "generation")
Y_METRICS: tuple[str, ...] = ("compare_score", "compare_success", "champion_score", "score_best")
STEP_METRICS = frozenset({"compare_score", "compare_success", "champion_score"})
"""챔피언이 바뀔 때만 바뀌는 값: 앞 값을 이어 가는 계단선."""
PERCENT_METRICS = frozenset({"compare_success"})
STAT_COLUMNS: tuple[str, ...] = (
    "generation", "evaluations", "elapsed", "compare_score", "compare_success", "champion_score", "score_best",
)
DEFAULT_TARGET = 80.0
MAX_GRID = 4000
"""묶음 공통 격자의 최대 점 수. 넘으면 고르게 줄인다."""
LIVE_REFRESH_MS = 1000
SETTINGS_GROUP = "compare"
SETTINGS_FILE = "settings.ini"
PATH_ROLE = Qt.ItemDataRole.UserRole
GROUP_ROLE = Qt.ItemDataRole.UserRole + 1
LIST_COL_ENV, LIST_COL_METHOD, LIST_COL_SEED, LIST_COL_GENERATIONS, LIST_COL_GROUP = range(5)
(TABLE_COL_NAME, TABLE_COL_METHOD, TABLE_COL_SEED, TABLE_COL_FINAL, TABLE_COL_TARGET, TABLE_COL_TIME,
 TABLE_COL_SUCCESS) = range(7)

_NAN = math.nan


# ---------------------------------------------------------------------------
# 목록
@dataclass
class RunEntry:
    """목록의 실행 하나: 기록 요약 + run.json에서 더 읽은 시드와 묶음."""

    summary: RunSummary
    seed: int | None = None
    group: dict[str, Any] | None = None
    """{"id": str, "name": str, "index": int, "count": int} 또는 None(묶음이 아니거나 형식이 틀림)."""

    @property
    def path(self) -> Path:
        return self.summary.path

    @property
    def key(self) -> str:
        return str(self.summary.path)

    @property
    def ok(self) -> bool:
        return self.summary.ok

    @property
    def method_name(self) -> str:
        return S.METHOD_NAMES.get(self.summary.method, self.summary.method)

    @property
    def env_name(self) -> str:
        return self.summary.env_name or self.summary.name

    def display_name(self) -> str:
        if self.seed is None:
            return S.CMP_RUN_NAME_NO_SEED.format(env=self.env_name, method=self.method_name)
        return S.CMP_RUN_NAME.format(env=self.env_name, method=self.method_name, seed=self.seed)


def _group_of(info: dict[str, Any]) -> dict[str, Any] | None:
    g = info.get("group")
    if not isinstance(g, dict):
        return None
    gid = g.get("id")
    if not isinstance(gid, str) or not gid:
        return None

    def as_int(v: Any, default: int) -> int:
        return v if isinstance(v, int) and not isinstance(v, bool) else default

    name = g.get("name")
    return {
        "id": gid,
        "name": name if isinstance(name, str) else "",
        "index": as_int(g.get("index"), 0),
        "count": max(1, as_int(g.get("count"), 1)),
    }


def read_entry(summary: RunSummary) -> RunEntry:
    entry = RunEntry(summary)
    if not summary.ok:
        return entry
    try:
        info = RunStore(summary.path).read_run_info()
    except (RunStoreError, OSError):
        return entry
    config = info.get("config") if isinstance(info.get("config"), dict) else {}
    seed = config.get("run_seed")
    entry.seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else None
    entry.group = _group_of(info)
    return entry


def list_entries(runs_dir: Path) -> list[RunEntry]:
    return [read_entry(s) for s in list_runs(runs_dir)]


def env_target(run: Path) -> float | None:
    """실행의 목표 점수: 마지막 환경 버전의 evaluation.target_score, 없으면 학습 설정의 target_score."""
    try:
        store = RunStore(run)
        info = store.read_run_info()
        versions = info.get("env_versions") or []
        if versions:
            from evolab.spec.io import parse_env

            env = parse_env(store.env_text(str(versions[-1]["id"])))
            if env.evaluation.target_score is not None:
                return float(env.evaluation.target_score)
        config = info.get("config") if isinstance(info.get("config"), dict) else {}
        t = config.get("target_score")
        if isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t):
            return float(t)
    except Exception:  # 읽을 수 없는 실행은 목표 점수 없음
        return None
    return None


_SUCCESS_CACHE: dict[tuple[str, str], bool] = {}
"""(실행 폴더, 마지막 환경 버전 id) → 그 환경에 성공 판정이 있는가. 환경 글은 버전마다 바뀌지 않는다."""


def env_has_success(run: Path) -> bool:
    """실행의 마지막 환경 버전 종료 목록에 결과가 success인 줄이 있는가(v2.0 설계 D21). 없으면 표의 성공률 칸은 「—」.
    읽을 수 없는 실행은 True(지금처럼 숫자를 보인다)."""
    try:
        store = RunStore(Path(run))
        versions = store.read_run_info().get("env_versions") or []
        if not versions:
            return True
        vid = str(versions[-1]["id"])
        key = (str(run), vid)
        hit = _SUCCESS_CACHE.get(key)
        if hit is None:
            from evolab.spec.io import parse_env

            env = parse_env(store.env_text(vid))
            hit = _SUCCESS_CACHE[key] = any(t.outcome == "success" for t in env.termination)
        return hit
    except Exception:  # 읽을 수 없는 실행은 지금처럼(성공률 숫자)
        return True


# ---------------------------------------------------------------------------
# 통계 읽기
@dataclass
class RunData:
    """한 실행의 통계 열(실수 배열, 줄 순서). 없는 열과 빈칸은 NaN."""

    cols: dict[str, np.ndarray]
    error: str | None = None

    @property
    def n(self) -> int:
        return len(self.cols["generation"])

    def col(self, name: str) -> np.ndarray:
        return self.cols[name]


def _empty_data(error: str | None = None) -> RunData:
    return RunData({c: np.empty(0) for c in STAT_COLUMNS}, error)


def load_stats(run: Path) -> RunData:
    """stats.csv(엑셀 잠금 옆 파일 포함)를 숫자 열로. 읽을 수 없으면 빈 자료와 오류 글."""
    try:
        rows = RunStore(Path(run)).read_stats()
    except (RunStoreError, OSError, csv.Error, ValueError, UnicodeDecodeError) as exc:
        return _empty_data(str(exc) or type(exc).__name__)
    n = len(rows)
    cols = {c: np.fromiter((to_float(r.get(c)) for r in rows), dtype=np.float64, count=n) for c in STAT_COLUMNS}
    gen = cols["generation"]
    bad = ~np.isfinite(gen)
    if bad.any():
        gen[bad] = np.arange(1, n + 1, dtype=np.float64)[bad]
    return RunData(cols)


def _file_key(path: Path) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


class StatsCache:
    """실행 폴더 → RunData. stats.csv와 옆 파일의 (수정 시각, 크기)가 같으면 다시 읽지 않는다."""

    def __init__(self, loader: Callable[[Path], RunData] = load_stats) -> None:
        self._loader = loader
        self._items: dict[str, tuple[Any, RunData]] = {}

    def get(self, run: Path) -> RunData:
        run = Path(run)
        key = (_file_key(run / STATS_FILE), _file_key(run / (STATS_FILE + PENDING_SUFFIX)))
        hit = self._items.get(str(run))
        if hit is not None and hit[0] == key:
            return hit[1]
        data = self._loader(run)
        self._items[str(run)] = (key, data)
        return data

    def clear(self) -> None:
        self._items.clear()


# ---------------------------------------------------------------------------
# 계열 계산 (화면 없이 쓸 수 있다)
def ffill(y: np.ndarray) -> np.ndarray:
    """NaN을 앞의 마지막 유한한 값으로 채운다. 첫 유한한 값 앞은 NaN 그대로."""
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0:
        return y.copy()
    idx = np.where(np.isfinite(y), np.arange(len(y)), -1)
    np.maximum.accumulate(idx, out=idx)
    return np.where(idx >= 0, y[np.maximum(idx, 0)], _NAN)


def series(data: RunData, x_axis: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """(x, y). 가로축 값이 없는 줄은 뺀다. 계단 값은 앞 값을 이어 가고, 성공률은 %로."""
    if x_axis not in X_AXES:
        raise ValueError(f"unknown x axis {x_axis!r}")
    if metric not in Y_METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    x = data.col(x_axis)
    y = data.col(metric)
    if metric in PERCENT_METRICS:
        y = y * 100.0
    if metric in STEP_METRICS:
        y = ffill(y)
    keep = np.isfinite(x)
    x, y = x[keep], y[keep]
    if len(x) > 1 and np.any(np.diff(x) < 0):  # 예전 실행의 이상한 값: 가로축 순서로
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
    return x, y


def step_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """오른쪽으로 이어 가는 계단선의 꺾은선 점: y[i]는 x[i]부터 x[i+1]까지."""
    n = len(x)
    if n < 2:
        return np.asarray(x, dtype=np.float64).copy(), np.asarray(y, dtype=np.float64).copy()
    sx = np.empty(2 * n - 1)
    sy = np.empty(2 * n - 1)
    sx[0::2] = x
    sx[1::2] = x[1:]
    sy[0::2] = y
    sy[1::2] = y[:-1]
    return sx, sy


def sample(x: np.ndarray, y: np.ndarray, grid: np.ndarray, step: bool) -> np.ndarray:
    """실행 하나의 값을 격자로 옮긴다. 실행의 범위 밖(첫 값 앞, 마지막 줄 뒤)은 NaN."""
    out = np.full(len(grid), _NAN)
    if len(x) == 0:
        return out
    if step:
        idx = np.searchsorted(x, grid, side="right") - 1
        ok = (idx >= 0) & (grid <= x[-1])
        out[ok] = y[idx[ok]]
        return out
    fin = np.isfinite(y)
    if not fin.any():
        return out
    xf, yf = x[fin], y[fin]
    ok = (grid >= xf[0]) & (grid <= xf[-1])
    out[ok] = np.interp(grid[ok], xf, yf)
    return out


@dataclass
class GroupCurve:
    """묶음의 공통 격자 위 평균·최소·최대와 값이 있는 실행 수."""

    x: np.ndarray
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    count: np.ndarray
    n: int
    """이 지표에 값이 하나라도 있는 실행 수(값이 전혀 없는 실행은 '모든 실행에 값이 있다'를 막지 않게 뺀다)."""

    @property
    def full(self) -> np.ndarray:
        """모든 실행에 값이 있는 격자 점(값이 있는 실행이 하나도 없으면 없다)."""
        return (self.count == self.n) & (self.count > 0)

    @property
    def partial(self) -> np.ndarray:
        """일부 실행에만 값이 있는 격자 점."""
        return (self.count > 0) & (self.count < self.n)


def aggregate(members: Sequence[tuple[np.ndarray, np.ndarray]], step: bool, max_grid: int = MAX_GRID) -> GroupCurve:
    """묶음 실행들의 (x, y)를 공통 격자(가로축 값을 모두 합친 것)에 옮겨 평균·최소·최대를 낸다."""
    xs = [x for x, _ in members if len(x)]
    grid = np.unique(np.concatenate(xs)) if xs else np.empty(0)
    if len(grid) > max_grid:
        grid = np.linspace(grid[0], grid[-1], max_grid)
    n = len(members)
    if n == 0 or len(grid) == 0:
        e = np.empty(0)
        return GroupCurve(e, e, e, e, np.zeros(0, dtype=np.int64), n)
    values = np.vstack([sample(x, y, grid, step) for x, y in members])
    fin = np.isfinite(values)
    # 값이 하나도 없는 실행(0세대로 끝난 실행, 첫 비교 평가 전에 멈춘 실행)은 세지 않는다: 세면 어느 격자 점도
    # '모든 실행에 값이 있다'가 되지 못해 평균 선과 띠가 사라진다
    n = int(fin.any(axis=1).sum())
    count = fin.sum(axis=0)
    have = count > 0
    mean = np.full(len(grid), _NAN)
    lo = np.full(len(grid), _NAN)
    hi = np.full(len(grid), _NAN)
    v = values[:, have]
    f = fin[:, have]
    mean[have] = np.where(f, v, 0.0).sum(axis=0) / count[have]
    lo[have] = np.where(f, v, np.inf).min(axis=0)
    hi[have] = np.where(f, v, -np.inf).max(axis=0)
    return GroupCurve(grid, mean, lo, hi, count, n)


# ---------------------------------------------------------------------------
# 표 값
def _last_finite(a: np.ndarray) -> float:
    fin = np.flatnonzero(np.isfinite(a))
    return float(a[fin[-1]]) if fin.size else _NAN


@dataclass
class RunMetrics:
    final: float
    """마지막 비교 점수(compare_score)."""
    to_target: float
    """compare_score가 처음 목표 이상이 된 줄의 누적 시험 횟수. 못 닿으면 NaN."""
    elapsed: float
    success: float
    """마지막 비교 평가 성공률(%)."""
    generations: int


def run_metrics(data: RunData, target: float) -> RunMetrics:
    cs = data.col("compare_score")
    reach = np.flatnonzero(np.isfinite(cs) & (cs >= target))
    to_target = float(data.col("evaluations")[reach[0]]) if reach.size else _NAN
    gens = _last_finite(data.col("generation"))
    return RunMetrics(
        final=_last_finite(cs),
        to_target=to_target,
        elapsed=_last_finite(data.col("elapsed")),
        success=_last_finite(data.col("compare_success")) * 100.0,
        generations=int(gens) if math.isfinite(gens) else 0,
    )


def mean_std(values: Iterable[float]) -> tuple[float, float, int]:
    """(평균, 표본 표준편차(값이 하나면 NaN), 유한한 값 수). 값이 없으면 (NaN, NaN, 0)."""
    v = np.array([x for x in values if math.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return _NAN, _NAN, 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else _NAN, int(v.size)


@dataclass
class TableRow:
    """표 한 줄(실행 하나 또는 묶음 하나)."""

    name: str
    color: str
    method: str
    seeds: list[int | None]
    runs: int
    final: float
    final_std: float
    to_target: float
    reached: int
    elapsed: float
    success: float
    success_std: float
    is_group: bool
    paths: list[Path]
    errors: list[str] = field(default_factory=list)
    has_success: bool = True
    """실행(묶음은 하나라도)의 환경에 성공 판정이 있는가. 없으면 성공률 칸은 「—」(v2.0 설계 D21)."""


def table_row(name: str, color: str, entries: Sequence[RunEntry], datas: Sequence[RunData], target: float,
              is_group: bool) -> TableRow:
    ms = [run_metrics(d, target) for d in datas]
    final, final_std, _ = mean_std(m.final for m in ms)
    to_target, _, reached = mean_std(m.to_target for m in ms)
    elapsed, _, _ = mean_std(m.elapsed for m in ms)
    success, success_std, _ = mean_std(m.success for m in ms)
    methods = sorted({e.method_name for e in entries})
    return TableRow(
        name=name,
        color=color,
        method=", ".join(methods),
        seeds=[e.seed for e in entries],
        runs=len(entries),
        final=final,
        final_std=final_std if is_group else _NAN,
        to_target=to_target,
        reached=reached,
        elapsed=elapsed,
        success=success,
        success_std=success_std if is_group else _NAN,
        is_group=is_group,
        paths=[e.path for e in entries],
        errors=[f"{e.summary.name}: {d.error}" for e, d in zip(entries, datas, strict=True) if d.error],
        has_success=not entries or any(env_has_success(e.path) for e in entries),
    )


def format_count(v: float) -> str:
    return f"{round(v):,}" if math.isfinite(v) else S.CMP_NONE


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return S.CMP_NONE
    if seconds < 60:
        return S.CMP_SECONDS.format(s=seconds)
    total = round(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return S.CMP_HOURS.format(h=h, m=m)
    return S.CMP_MINUTES.format(m=m, s=s)


def format_percent(v: float) -> str:
    return S.CMP_PERCENT.format(v=v) if math.isfinite(v) else S.CMP_NONE


def format_seeds(seeds: Sequence[int | None]) -> str:
    known = sorted(s for s in seeds if s is not None)
    if len(seeds) == 1:
        return str(seeds[0]) if seeds[0] is not None else S.CMP_NONE
    if not known:
        return S.CMP_SEED_COUNT.format(n=len(seeds))
    if known == list(range(known[0], known[0] + len(known))) and len(known) == len(seeds):
        return S.CMP_SEED_RANGE.format(first=known[0], last=known[-1], n=len(seeds))
    return S.CMP_SEED_COUNT.format(n=len(seeds))


def format_mean_std(mean: float, std: float) -> str:
    """'평균 ± 표준편차'. 두 값을 평균의 크기에 맞춘 같은 자릿수로 쓴다. 표준편차가 없으면 평균만."""
    if not math.isfinite(mean):
        return S.CMP_NONE
    if not math.isfinite(std):
        return fmt_value(mean)
    a = abs(mean)
    digits = 0 if a >= 1000 else 1 if a >= 100 else 2 if a >= 1 else 3
    return S.CMP_MEAN_STD.format(mean=f"{mean:,.{digits}f}", std=f"{std:,.{digits}f}")


def row_texts(r: TableRow) -> list[str]:
    """표에 보이는 칸 글자(TABLE_COL_* 순서)."""
    if r.is_group:
        final = format_mean_std(r.final, r.final_std)
        target = (S.CMP_REACHED_SOME.format(value=format_count(r.to_target), k=r.reached, n=r.runs)
                  if r.reached else S.CMP_NONE)
        success = (S.CMP_MEAN_STD.format(mean=format_percent(r.success), std=format_percent(r.success_std))
                   if math.isfinite(r.success_std) else format_percent(r.success))
    else:
        final = fmt_value(r.final)
        target = format_count(r.to_target)
        success = format_percent(r.success)
    if not r.has_success:  # 성공 판정이 없는 환경(글라이더): 0%가 실패처럼 보이지 않게
        success = S.NO_SUCCESS_CELL
    return [r.name, r.method, format_seeds(r.seeds), final, target, format_duration(r.elapsed), success]


def _csv_number(v: float) -> str:
    if not math.isfinite(v):
        return ""
    return repr(round(float(v), 6))


def row_csv(r: TableRow, target: float) -> list[str]:
    """CSV 한 줄(S.CMP_CSV_HEADERS 순서). 숫자는 엑셀에서 계산할 수 있게 글자 꾸밈 없이."""
    seeds = " ".join(str(s) for s in r.seeds if s is not None)
    return [
        r.name, r.method, seeds, str(r.runs),
        _csv_number(r.final), _csv_number(r.final_std), _csv_number(target),
        _csv_number(r.to_target), str(r.reached), _csv_number(r.elapsed),
        _csv_number(r.success) if r.has_success else "", _csv_number(r.success_std) if r.has_success else "",
    ]


# ---------------------------------------------------------------------------
# 그리기 항목
@dataclass
class Shown:
    """그래프에 그린 것 하나(실행 하나 또는 묶음 하나). 테스트와 표가 같이 쓴다."""

    kind: str
    """"run" 또는 "group"."""
    name: str
    color: str
    style: Qt.PenStyle
    entries: list[RunEntry]
    datas: list[RunData]
    x: np.ndarray = field(default_factory=lambda: np.empty(0))
    """실행: 통계 줄의 가로축 값. 묶음: 공통 격자."""
    y: np.ndarray = field(default_factory=lambda: np.empty(0))
    """실행: 값(계단 값은 이어 간 것). 묶음: 평균."""
    curve: GroupCurve | None = None
    group_id: str = ""


def _pen(color: str | QColor, width: float, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> QPen:
    pen = pg.mkPen(color=color, width=width)
    pen.setStyle(style)
    return pen


def _qcolor(color: str, alpha: int) -> QColor:
    c = QColor(color)
    c.setAlpha(alpha)
    return c


def _blend(color: str, surface: str, amount: float) -> str:
    a, b = QColor(color), QColor(surface)
    mix = [round(x * amount + y * (1 - amount)) for x, y in ((a.red(), b.red()), (a.green(), b.green()), (a.blue(), b.blue()))]
    return QColor(*mix).name()


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _swatch(color: str) -> QIcon:
    pm = QPixmap(12, 12)
    pm.fill(QColor(color))
    return QIcon(pm)


_STYLES = (Qt.PenStyle.SolidLine, Qt.PenStyle.DashLine, Qt.PenStyle.DotLine, Qt.PenStyle.DashDotLine)


class _PlotWidget(pg.PlotWidget):
    def sizeHint(self) -> QSize:
        return QSize(480, 300)


# ---------------------------------------------------------------------------
# 화면
class CompareView(QWidget):
    """비교 화면."""

    repeat_started = Signal(str)
    """반복 실행 묶음 id."""
    repeat_run_started = Signal(Path)
    """반복 실행의 새 실행 폴더(학습 중: 기록 화면에서 지우면 안 된다)."""
    repeat_finished = Signal(bool)
    """True: 모두 끝냄, False: 취소·실패."""

    def __init__(
        self,
        runs_dir_provider: Callable[[], Path] = paths.runs_dir,
        parent: QWidget | None = None,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self._runs_dir_provider = runs_dir_provider
        if settings is None:
            settings = QSettings(str(paths.user_state_dir() / SETTINGS_FILE), QSettings.Format.IniFormat)
        self._settings = settings
        self.theme: Theme = theme_for(self)
        self.cache = StatsCache()
        self.entries: list[RunEntry] = []
        self._by_key: dict[str, RunEntry] = {}
        self._checked: set[str] = set()
        self._collapsed: set[str] = set()
        self.shown: list[Shown] = []
        self._no_success_hidden = False
        """성공률 그래프에서 성공 판정이 없는 환경의 실행을 뺐는가(안내 CMP_NO_SUCCESS_RATE)."""
        self.rows: list[TableRow] = []
        self.x_axis = "evaluations"
        self.y_metric = "compare_score"
        self.group_mean = True
        self._target_user_set = False
        self._setting_target = False
        self._filling = False
        self._loading = True
        self.runner: RepeatRunner | None = None
        self._follow_group: str | None = None
        self._last_repeat_message = ""

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(0)
        self._redraw_timer.timeout.connect(self.redraw)
        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(LIVE_REFRESH_MS)
        self._live_timer.timeout.connect(lambda: self.redraw(keep_view=True))

        # -- 왼쪽: 실행 목록 ----------------------------------------------------
        self.folder_label = QLabel()
        self.folder_label.setWordWrap(True)
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # 줄을 바꿔도 윈도우 경로(\로 이어진 한 덩어리)는 끊을 자리가 없어 그 너비가 창의 최소 너비가 됐다
        self.folder_label.setMinimumWidth(1)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(list(S.CMP_LIST_COLUMNS))
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tree.setAlternatingRowColors(False)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(LIST_COL_ENV, QHeaderView.ResizeMode.Stretch)
        for col in (LIST_COL_METHOD, LIST_COL_SEED, LIST_COL_GENERATIONS, LIST_COL_GROUP):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemExpanded.connect(lambda it: self._collapsed.discard(str(it.data(0, GROUP_ROLE))))
        self.tree.itemCollapsed.connect(lambda it: self._collapsed.add(str(it.data(0, GROUP_ROLE))))
        self.empty_label = QLabel(S.CMP_LIST_EMPTY)
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.repeat_button = QPushButton(S.CMP_REPEAT_BUTTON)
        self.repeat_button.setToolTip(S.CMP_REPEAT_BUTTON_TIP)
        self.repeat_button.clicked.connect(self.open_repeat_dialog)
        self.refresh_button = QPushButton(S.CMP_REFRESH)
        self.refresh_button.clicked.connect(self.refresh)
        self.clear_button = QPushButton(S.CMP_CLEAR_SELECTION)
        self.clear_button.clicked.connect(lambda: self.set_selected_runs([]))

        self.repeat_label = QLabel()
        self.repeat_label.setWordWrap(True)
        self.repeat_bar = QProgressBar()
        self.repeat_bar.setRange(0, 1000)
        self.repeat_bar.setTextVisible(False)
        self.repeat_cancel = QPushButton(S.CMP_REPEAT_CANCEL)
        self.repeat_cancel.clicked.connect(self.cancel_repeat)
        self.repeat_box = QFrame()
        self.repeat_box.setFrameShape(QFrame.Shape.StyledPanel)
        rb = QVBoxLayout(self.repeat_box)
        rb.setContentsMargins(6, 4, 6, 4)
        rb.addWidget(self.repeat_label)
        rrow = QHBoxLayout()
        rrow.addWidget(self.repeat_bar, 1)
        rrow.addWidget(self.repeat_cancel)
        rb.addLayout(rrow)
        self.repeat_box.hide()
        self.repeat_status = QLabel()
        self.repeat_status.setWordWrap(True)
        self.repeat_status.hide()

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        lb = QHBoxLayout()
        lb.addWidget(self.repeat_button)
        lb.addStretch(1)
        lb.addWidget(self.clear_button)
        lb.addWidget(self.refresh_button)
        ll.addLayout(lb)
        ll.addWidget(self.repeat_box)
        ll.addWidget(self.repeat_status)
        ll.addWidget(self.tree, 1)
        ll.addWidget(self.empty_label)
        ll.addWidget(self.folder_label)

        # -- 가운데: 그래프 ----------------------------------------------------
        self.y_box = QComboBox()
        for m in Y_METRICS:
            self.y_box.addItem(S.CMP_Y_NAMES[m], m)
        self.y_box.setToolTip(S.CMP_Y_TIP)
        self.y_box.currentIndexChanged.connect(lambda _i: self.set_y_metric(str(self.y_box.currentData())))
        self.x_box = QComboBox()
        for a in X_AXES:
            self.x_box.addItem(S.CMP_X_NAMES[a], a)
        self.x_box.setToolTip(S.CMP_X_TIP)
        self.x_box.currentIndexChanged.connect(lambda _i: self.set_x_axis(str(self.x_box.currentData())))
        self.mean_check = QCheckBox(S.CMP_GROUP_MEAN)
        self.mean_check.setToolTip(S.CMP_GROUP_MEAN_TIP)
        self.mean_check.setChecked(True)
        self.mean_check.toggled.connect(self.set_group_mean)
        self.target_spin = QDoubleSpinBox()
        self.target_spin.setRange(-1e9, 1e9)
        self.target_spin.setDecimals(1)
        self.target_spin.setValue(DEFAULT_TARGET)
        self.target_spin.setToolTip(S.CMP_TARGET_TIP)
        self.target_spin.valueChanged.connect(self._on_target_changed)
        self.png_button = QPushButton(S.CMP_SAVE_PNG)
        self.png_button.clicked.connect(self._on_save_png)
        self.csv_button = QPushButton(S.CMP_SAVE_CSV)
        self.csv_button.clicked.connect(self._on_save_csv)
        self.race_button = QPushButton(S.CMP_RACE_BUTTON)
        self.race_button.setToolTip(S.CMP_RACE_BUTTON_TIP)
        self.race_button.clicked.connect(self.open_race)
        self.race_dialog: Any = None
        """마지막으로 연 경주 창(RaceDialog). 다시 누르면 같은 창의 내용을 바꾼다."""

        bar = QHBoxLayout()
        bar.addWidget(QLabel(S.CMP_Y_LABEL))
        bar.addWidget(self.y_box)
        bar.addSpacing(8)
        bar.addWidget(QLabel(S.CMP_X_LABEL))
        bar.addWidget(self.x_box)
        bar.addSpacing(8)
        bar.addWidget(self.mean_check)
        bar.addStretch(1)
        bar.addWidget(self.race_button)
        bar.addWidget(self.png_button)

        t = self.theme
        self.graph_frame = QFrame()
        self.graph_frame.setObjectName("compareGraph")
        self.graph_frame.setStyleSheet(f"#compareGraph {{ background: {t.surface}; }}")
        gl = QVBoxLayout(self.graph_frame)
        gl.setContentsMargins(6, 4, 6, 4)
        gl.setSpacing(2)
        self.title_label = QLabel()
        title_font = QFont(self.title_label.font())
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet(f"color: {t.ink};")
        self.legend = QLabel()
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        self.legend.setWordWrap(True)
        self.legend.setStyleSheet(f"color: {t.ink2};")
        self.note_label = QLabel()
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet(f"color: {t.tick};")
        self.note_label.hide()
        self.plot_widget = _PlotWidget(background=t.surface)
        self.plot = self.plot_widget.getPlotItem()
        self._style_plot()
        self.message = QLabel(S.CMP_PICK_RUNS)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setWordWrap(True)
        self.message.setStyleSheet(f"color: {t.ink2};")
        self.stack = QStackedWidget()
        self.stack.addWidget(self.plot_widget)
        self.stack.addWidget(self.message)
        gl.addWidget(self.title_label)
        gl.addWidget(self.legend)
        gl.addWidget(self.stack, 1)
        gl.addWidget(self.note_label)

        graph = QWidget()
        gv = QVBoxLayout(graph)
        gv.setContentsMargins(0, 0, 0, 0)
        gv.addLayout(bar)
        gv.addWidget(self.graph_frame, 1)

        # -- 아래: 표 ----------------------------------------------------------
        self.table = QTableWidget(0, len(S.CMP_TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(S.CMP_TABLE_COLUMNS))
        for col, tip in enumerate(S.CMP_TABLE_TIPS):
            item = self.table.horizontalHeaderItem(col)
            if item is not None:
                item.setToolTip(tip)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setWordWrap(False)
        th = self.table.horizontalHeader()
        th.setSectionResizeMode(TABLE_COL_NAME, QHeaderView.ResizeMode.Stretch)
        for col in range(1, len(S.CMP_TABLE_COLUMNS)):
            th.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        tbar = QHBoxLayout()
        tbar.addWidget(QLabel(S.CMP_TARGET_LABEL))
        tbar.addWidget(self.target_spin)
        tbar.addStretch(1)
        tbar.addWidget(self.csv_button)
        table_box = QWidget()
        tv = QVBoxLayout(table_box)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.addLayout(tbar)
        tv.addWidget(self.table, 1)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(graph)
        right.addWidget(table_box)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 1)
        right.setChildrenCollapsible(False)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([440, 840])
        split.setChildrenCollapsible(False)
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.addWidget(split)

        self._load_settings()
        self._loading = False
        self.refresh()

    # ------------------------------------------------------------------
    # 설정
    # ------------------------------------------------------------------
    def _load_settings(self) -> None:
        s = self._settings
        s.beginGroup(SETTINGS_GROUP)
        try:
            x = s.value("x_axis")
            y = s.value("metric")
            mean = s.value("group_mean")
            target = s.value("target")
        finally:
            s.endGroup()
        if isinstance(x, str) and x in X_AXES:
            self.set_x_axis(x)
        if isinstance(y, str) and y in Y_METRICS:
            self.set_y_metric(y)
        if mean is not None:
            self.set_group_mean(str(mean).lower() in ("true", "1"))
        if target is not None:
            try:
                value = float(str(target))
            except ValueError:
                value = _NAN
            if math.isfinite(value):
                self._set_target_quietly(value)
                self._target_user_set = True

    def save_settings(self) -> None:
        if self._loading:
            return
        s = self._settings
        s.beginGroup(SETTINGS_GROUP)
        s.setValue("x_axis", self.x_axis)
        s.setValue("metric", self.y_metric)
        s.setValue("group_mean", self.group_mean)
        if self._target_user_set:
            s.setValue("target", self.target)
        s.endGroup()
        s.sync()

    # ------------------------------------------------------------------
    # 목록
    # ------------------------------------------------------------------
    def runs_dir(self) -> Path:
        return Path(self._runs_dir_provider())

    def refresh(self) -> None:
        """runs 폴더를 다시 읽는다. 고른 실행은 그대로 고른 채로 둔다."""
        base = self.runs_dir()
        self.entries = list_entries(base)
        self._by_key = {e.key: e for e in self.entries}
        self._checked &= {e.key for e in self.entries if e.ok}
        self._fill_tree()
        self.folder_label.setText(S.CMP_FOLDER.format(count=len(self.entries), path=base))
        self.empty_label.setVisible(not self.entries)
        self.redraw(keep_view=True)

    def _layout_entries(self) -> list[tuple[str, Any]]:
        """목록 순서: ("run", RunEntry) 또는 ("group", (묶음 정보, [RunEntry …])). 묶음은 가장 최근 실행 자리에."""
        out: list[tuple[str, Any]] = []
        groups: dict[str, list[RunEntry]] = {}
        for e in self.entries:
            if e.group is None:
                out.append(("run", e))
                continue
            gid = e.group["id"]
            if gid not in groups:
                groups[gid] = []
                out.append(("group", gid))
            groups[gid].append(e)
        result: list[tuple[str, Any]] = []
        for kind, value in out:
            if kind == "group":
                members = sorted(groups[value], key=lambda m: (m.group["index"] if m.group else 0, m.summary.started))
                result.append(("group", (members[0].group, members)))
            else:
                result.append((kind, value))
        return result

    @staticmethod
    def group_name(info: dict[str, Any], members: Sequence[RunEntry]) -> str:
        if info.get("name"):
            return str(info["name"])
        first = members[0]
        seeds = [m.seed for m in members if m.seed is not None]
        if seeds:
            return S.CMP_GROUP_NAME.format(env=first.env_name, method=first.method_name,
                                                 first=min(seeds), last=max(seeds))
        return S.CMP_GROUP_NAME_PLAIN.format(env=first.env_name, method=first.method_name)

    def _fill_tree(self) -> None:
        self._filling = True
        try:
            self.tree.clear()
            for kind, value in self._layout_entries():
                if kind == "run":
                    self.tree.addTopLevelItem(self._run_item(value, in_group=False))
                    continue
                info, members = value
                top = QTreeWidgetItem()
                top.setData(0, GROUP_ROLE, info["id"])
                name = self.group_name(info, members)
                top.setText(LIST_COL_ENV, name)
                font = QFont(top.font(0))
                font.setBold(True)
                top.setFont(LIST_COL_ENV, font)
                top.setText(LIST_COL_METHOD, ", ".join(sorted({m.method_name for m in members})))
                top.setText(LIST_COL_SEED, format_seeds([m.seed for m in members]))
                top.setText(LIST_COL_GROUP, S.CMP_GROUP_MEMBERS.format(n=len(members), count=info["count"]))
                top.setToolTip(LIST_COL_ENV, S.CMP_GROUP_TIP.format(name=name, id=info["id"]))
                ok_members = [m for m in members if m.ok]
                flags = Qt.ItemFlag.ItemIsEnabled
                if ok_members:
                    flags |= Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate
                top.setFlags(flags)
                for m in members:
                    top.addChild(self._run_item(m, in_group=True))
                self.tree.addTopLevelItem(top)
                top.setExpanded(info["id"] not in self._collapsed)
        finally:
            self._filling = False

    def _run_item(self, e: RunEntry, in_group: bool) -> QTreeWidgetItem:
        item = QTreeWidgetItem()
        s = e.summary
        item.setData(0, PATH_ROLE, e.key)
        item.setText(LIST_COL_ENV, e.env_name if s.ok else s.name)
        item.setText(LIST_COL_METHOD, e.method_name if s.ok else S.CMP_UNREADABLE)
        item.setText(LIST_COL_SEED, str(e.seed) if e.seed is not None else S.CMP_NONE)
        item.setText(LIST_COL_GENERATIONS, str(s.generations) if s.ok else S.CMP_NONE)
        if in_group and e.group is not None:
            item.setText(LIST_COL_GROUP, S.CMP_GROUP_INDEX.format(i=e.group["index"] + 1, count=e.group["count"]))
        for col in (LIST_COL_SEED, LIST_COL_GENERATIONS):
            item.setTextAlignment(col, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        tip = [S.CMP_TIP_FOLDER.format(path=s.path), S.CMP_TIP_STARTED.format(started=s.started.replace("T", " "))]
        if s.error:
            tip.append(s.error)
        for col in range(5):
            item.setToolTip(col, "\n".join(tip))
        if s.ok:
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemNeverHasChildren)
            item.setCheckState(0, Qt.CheckState.Checked if e.key in self._checked else Qt.CheckState.Unchecked)
        else:
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemNeverHasChildren)
            for col in range(5):
                item.setForeground(col, QBrush(QColor(style.ui_for().muted)))
        return item

    def _leaf_items(self) -> list[QTreeWidgetItem]:
        out: list[QTreeWidgetItem] = []
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            if top is None:
                continue
            if top.data(0, GROUP_ROLE):
                out.extend(top.child(j) for j in range(top.childCount()))
            else:
                out.append(top)
        return out

    def group_item(self, group_id: str) -> QTreeWidgetItem | None:
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            if top is not None and top.data(0, GROUP_ROLE) == group_id:
                return top
        return None

    def run_item(self, path: Path) -> QTreeWidgetItem | None:
        key = str(Path(path))
        for item in self._leaf_items():
            if item.data(0, PATH_ROLE) == key:
                return item
        return None

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._filling or column != 0:
            return
        self._sync_checked()
        self._redraw_timer.start()

    def _sync_checked(self) -> None:
        checked = set()
        for item in self._leaf_items():
            key = item.data(0, PATH_ROLE)
            if key and item.flags() & Qt.ItemFlag.ItemIsUserCheckable and item.checkState(0) == Qt.CheckState.Checked:
                checked.add(str(key))
        self._checked = checked

    def selected_runs(self) -> list[Path]:
        """고른 실행 폴더(목록 순서)."""
        return [e.path for e in self.entries if e.key in self._checked]

    def set_selected_runs(self, runs: Iterable[Path]) -> None:
        """이 실행들만 고르고 다시 그린다(기록 화면의 '비교'). 목록을 먼저 다시 읽는다. 없는 폴더는 무시한다."""
        wanted = {str(Path(p)) for p in runs}
        self.entries = list_entries(self.runs_dir())
        self._by_key = {e.key: e for e in self.entries}
        if wanted:
            resolved = {_resolved(Path(k)): k for k in wanted}
            for e in self.entries:
                if e.key not in wanted and _resolved(e.path) in resolved:
                    wanted.add(e.key)
        self._checked = {e.key for e in self.entries if e.ok and e.key in wanted}
        self._fill_tree()
        self.folder_label.setText(S.CMP_FOLDER.format(count=len(self.entries), path=self.runs_dir()))
        self.empty_label.setVisible(not self.entries)
        self.redraw()

    def set_group_checked(self, group_id: str, checked: bool = True) -> bool:
        """묶음 전체를 고르거나 푼다. 목록에 없으면 False."""
        keys = {e.key for e in self.entries if e.ok and e.group is not None and e.group["id"] == group_id}
        if not keys:
            return False
        self._checked = (self._checked | keys) if checked else (self._checked - keys)
        self._apply_checked_to_tree()
        self.redraw()
        return True

    def _apply_checked_to_tree(self) -> None:
        self._filling = True
        try:
            for item in self._leaf_items():
                if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    key = str(item.data(0, PATH_ROLE))
                    item.setCheckState(0, Qt.CheckState.Checked if key in self._checked else Qt.CheckState.Unchecked)
        finally:
            self._filling = False

    # ------------------------------------------------------------------
    # 축, 목표 점수
    # ------------------------------------------------------------------
    def set_x_axis(self, axis: str) -> None:
        if axis not in X_AXES:
            raise ValueError(f"unknown x axis {axis!r}")
        if self.x_box.currentData() != axis:
            self.x_box.blockSignals(True)
            self.x_box.setCurrentIndex(X_AXES.index(axis))
            self.x_box.blockSignals(False)
        if axis == self.x_axis:
            return
        self.x_axis = axis
        self.save_settings()
        if not self._loading:
            self.redraw()

    def set_y_metric(self, metric: str) -> None:
        if metric not in Y_METRICS:
            raise ValueError(f"unknown metric {metric!r}")
        if self.y_box.currentData() != metric:
            self.y_box.blockSignals(True)
            self.y_box.setCurrentIndex(Y_METRICS.index(metric))
            self.y_box.blockSignals(False)
        if metric == self.y_metric:
            return
        self.y_metric = metric
        self.save_settings()
        if not self._loading:
            self.redraw()

    def set_group_mean(self, on: bool) -> None:
        on = bool(on)
        if self.mean_check.isChecked() != on:
            self.mean_check.blockSignals(True)
            self.mean_check.setChecked(on)
            self.mean_check.blockSignals(False)
        if on == self.group_mean:
            return
        self.group_mean = on
        self.save_settings()
        if not self._loading:
            self.redraw()

    @property
    def target(self) -> float:
        return float(self.target_spin.value())

    def set_target(self, value: float) -> None:
        """목표 점수를 정한다(사용자가 정한 값으로 본다: 고른 실행이 바뀌어도 바꾸지 않는다)."""
        self._target_user_set = True
        if self.target_spin.value() != float(value):
            self.target_spin.setValue(float(value))  # _on_target_changed
        else:
            self._on_target_changed(float(value))

    def _set_target_quietly(self, value: float) -> None:
        self._setting_target = True
        try:
            self.target_spin.setValue(float(value))
        finally:
            self._setting_target = False

    def _on_target_changed(self, _value: float) -> None:
        if not self._setting_target and not self._loading:
            self._target_user_set = True
            self.save_settings()
        self._update_target_line()
        self._fill_table()

    def _maybe_default_target(self) -> None:
        """사용자가 아직 목표 점수를 정하지 않았으면 고른 첫 실행의 환경 목표 점수를 쓴다."""
        if self._target_user_set or not self.shown:
            return
        for s in self.shown:
            for e in s.entries:
                t = env_target(e.path)
                if t is not None:
                    if t != self.target:
                        self._set_target_quietly(t)
                    return

    # ------------------------------------------------------------------
    # 그리기
    # ------------------------------------------------------------------
    def _style_plot(self) -> None:
        t = self.theme
        p = self.plot
        p.setMenuEnabled(False)
        p.showGrid(x=False, y=True, alpha=0.35)
        for name in ("left", "bottom"):
            ax = p.getAxis(name)
            ax.setPen(_pen(t.axis, 1))
            ax.setTextPen(_pen(t.tick, 1))
            ax.enableAutoSIPrefix(False)
        p.getAxis("left").setWidth(AXIS_WIDTH)

    def _items_to_show(self) -> list[tuple[str, str, list[RunEntry], str]]:
        """(종류, 이름, 실행들, 묶음 id) 목록 순서대로. 묶음 평균을 끄면 묶음 실행도 하나씩."""
        out: list[tuple[str, str, list[RunEntry], str]] = []
        for kind, value in self._layout_entries():
            if kind == "run":
                if value.key in self._checked:
                    out.append(("run", value.display_name(), [value], ""))
                continue
            info, members = value
            chosen = [m for m in members if m.key in self._checked]
            if not chosen:
                continue
            if self.group_mean:
                out.append(("group", self.group_name(info, members), chosen, info["id"]))
            else:
                out.extend(("run", m.display_name(), [m], info["id"]) for m in chosen)
        return out

    def redraw(self, keep_view: bool = False) -> None:
        """고른 실행을 다시 읽어(바뀐 파일만) 그래프와 표를 다시 그린다."""
        self._redraw_timer.stop()
        t = self.theme
        step = self.y_metric in STEP_METRICS
        # 성공률: 성공 판정이 없는 환경(글라이더)의 실행은 그리지 않는다(통계의 0%가 실패처럼 보이지 않게, 표의 「—」와
        # 같은 규칙: v2.0 설계 D21). 범례는 남기고 안내를 보인다.
        rate = self.y_metric in PERCENT_METRICS
        self._no_success_hidden = False
        shown: list[Shown] = []
        for k, (kind, name, entries, gid) in enumerate(self._items_to_show()):
            color = t.series[k % len(t.series)]
            style = _STYLES[(k // len(t.series)) % len(_STYLES)]
            datas = [self.cache.get(e.path) for e in entries]
            item = Shown(kind, name, color, style, entries, datas, group_id=gid)
            keep = [not rate or env_has_success(e.path) for e in entries]
            self._no_success_hidden |= not all(keep)
            if kind == "run":
                if keep[0]:
                    item.x, item.y = series(datas[0], self.x_axis, self.y_metric)
            else:
                members = [d for d, ok in zip(datas, keep, strict=True) if ok]
                curve = aggregate([series(d, self.x_axis, self.y_metric) for d in members], step)
                item.curve = curve
                item.x, item.y = curve.x, curve.mean
            shown.append(item)
        first_selection = not self.shown and bool(shown)
        self.shown = shown
        if first_selection:
            self._maybe_default_target()
        self._draw(keep_view)
        self._fill_table()
        has = bool(shown)
        self.png_button.setEnabled(has)
        self.csv_button.setEnabled(has)
        self._update_race_button()

    def _draw(self, keep_view: bool) -> None:
        t = self.theme
        p = self.plot
        p.clear()
        step = self.y_metric in STEP_METRICS
        any_value = False
        legend: list[str] = []
        for s in self.shown:
            if s.kind == "run":
                if np.isfinite(s.y).any():
                    any_value = True
                    x, y = step_xy(s.x, s.y) if step else (s.x, s.y)
                    p.addItem(pg.PlotCurveItem(x, y, pen=_pen(s.color, 1.5, s.style), connect="finite"))
                legend.append(self._legend_entry(s.color, GLYPH_LINE if s.style == Qt.PenStyle.SolidLine else GLYPH_DASH,
                                                 s.name))
                continue
            curve = s.curve
            assert curve is not None
            full = curve.full
            if full.any():
                any_value = True
                a, b = int(np.flatnonzero(full)[0]), int(np.flatnonzero(full)[-1])
                gx = curve.x[a:b + 1]
                lo, hi = curve.lo[a:b + 1], curve.hi[a:b + 1]
                if step:
                    ux, uy = step_xy(gx, hi)
                    _lx, ly = step_xy(gx, lo)
                else:
                    ux, uy, ly = gx, hi, lo
                if curve.n > 1:
                    band = pg.PlotCurveItem(np.concatenate((ux, ux[::-1])), np.concatenate((uy, ly[::-1])),
                                            pen=pg.mkPen(None), brush=_qcolor(s.color, 55), fillLevel="enclosed")
                    band.setZValue(-10)
                    p.addItem(band)
                mean_full = np.where(full, curve.mean, _NAN)
                x, y = step_xy(curve.x, mean_full) if step else (curve.x, mean_full)
                line = pg.PlotCurveItem(x, y, pen=_pen(s.color, 2.6, s.style), connect="finite")
                line.setZValue(10)
                p.addItem(line)
            partial = curve.partial
            if partial.any():
                any_value = True
                part = np.where(partial, curve.mean, _NAN)
                # 실선 평균과 이어지게, 일부 구간이 있는 쪽의 경계 점을 넣는다
                if full.any():
                    a, b = int(np.flatnonzero(full)[0]), int(np.flatnonzero(full)[-1])
                    if partial[:a].any():
                        part[a] = curve.mean[a]
                    if partial[b + 1:].any():
                        part[b] = curve.mean[b]
                x, y = step_xy(curve.x, part) if step else (curve.x, part)
                dashed = pg.PlotCurveItem(x, y, pen=_pen(_blend(s.color, t.surface, 0.85), 1.6, Qt.PenStyle.DashLine),
                                          connect="finite")
                dashed.setZValue(5)
                p.addItem(dashed)
            legend.append(self._legend_entry(s.color, GLYPH_LINE, S.CMP_LEGEND_GROUP.format(
                name=s.name, n=curve.n if curve.n else len(s.entries))))
            if curve.n > 1 and full.any():  # 띠를 그렸을 때만
                legend.append(self._legend_entry(_blend(s.color, t.surface, 0.35), GLYPH_FILL, S.CMP_LEGEND_BAND))
            if partial.any():
                legend.append(self._legend_entry(_blend(s.color, t.surface, 0.85), GLYPH_DASH, S.CMP_LEGEND_PARTIAL))
        self._target_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=_pen(t.ink2, 1.2, Qt.PenStyle.DashLine),
            label="",
            labelOpts={"position": 0.02, "color": t.ink2, "fill": _qcolor(t.surface, 210), "anchors": [(0, 1), (0, 1)]},
        )
        self._target_line.setZValue(20)
        p.addItem(self._target_line, ignoreBounds=True)
        self._update_target_line()
        p.setLabel("left", S.CMP_Y_AXIS_NAMES[self.y_metric], color=t.ink2)
        p.setLabel("bottom", S.CMP_X_NAMES[self.x_axis], color=t.ink2)
        self.title_label.setText(S.CMP_TITLE.format(y=S.CMP_Y_NAMES[self.y_metric], x=S.CMP_X_NAMES[self.x_axis]))
        self.legend.setText("&nbsp;&nbsp; ".join(legend))
        self.stack.setCurrentWidget(self.plot_widget if self.shown else self.message)
        if self.shown and self._no_success_hidden:
            self.note_label.setText(S.CMP_NO_SUCCESS_RATE)
            self.note_label.show()
        elif self.shown and not any_value:
            self.note_label.setText(S.CMP_NO_VALUES.format(y=S.CMP_Y_NAMES[self.y_metric]))
            self.note_label.show()
        else:
            self.note_label.hide()
        # 사용자가 확대·이동한 것은 실시간 갱신(keep_view)에서는 그대로 둔다. 성공률은 0~100%로 고정.
        if not keep_view:
            p.enableAutoRange()
        if self.y_metric in PERCENT_METRICS:
            p.enableAutoRange(axis="y", enable=False)
            p.setYRange(0, 100, padding=0.04)

    @staticmethod
    def _legend_entry(color: str, glyph: str, name: str) -> str:
        return f'<span style="color:{color}; font-weight:bold">{glyph}</span>&nbsp;{_escape(name)}'

    def _update_target_line(self) -> None:
        line = getattr(self, "_target_line", None)
        if line is None:
            return
        if self.y_metric in ("compare_score", "champion_score", "score_best") and self.shown:
            line.setPos(self.target)
            if line.label is not None:
                line.label.setFormat(S.CMP_TARGET_LINE.format(v=fmt_value(self.target)).replace("{", "{{").replace("}", "}}"))
            line.show()
        else:
            line.hide()

    # ------------------------------------------------------------------
    # 표
    # ------------------------------------------------------------------
    def _fill_table(self) -> None:
        target = self.target
        self.rows = [table_row(s.name, s.color, s.entries, s.datas, target, s.kind == "group") for s in self.shown]
        self.table.setRowCount(len(self.rows))
        for r, row in enumerate(self.rows):
            texts = row_texts(row)
            tip_lines = [S.CMP_TIP_FOLDER.format(path=p) for p in row.paths[:10]]
            if len(row.paths) > 10:
                tip_lines.append(S.CMP_TIP_MORE.format(n=len(row.paths) - 10))
            tip_lines.extend(row.errors)
            tip = "\n".join(tip_lines)
            for c, text in enumerate(texts):
                item = QTableWidgetItem(text)
                item.setToolTip(tip)
                if c == TABLE_COL_NAME:
                    item.setIcon(_swatch(row.color))
                if c >= TABLE_COL_SEED:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)

    def table_texts(self) -> list[list[str]]:
        """표에 보이는 글자(줄마다)."""
        out: list[list[str]] = []
        for r in range(self.table.rowCount()):
            out.append([(self.table.item(r, c).text() if self.table.item(r, c) is not None else "")  # type: ignore[union-attr]
                        for c in range(self.table.columnCount())])
        return out

    # ------------------------------------------------------------------
    # 내보내기
    # ------------------------------------------------------------------
    def save_plot_png(self, path: Path) -> bool:
        """그래프(제목, 범례 포함)를 PNG로. 실패하면 False."""
        pm = self.graph_frame.grab()
        if pm.isNull() or pm.width() == 0:
            return False
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return bool(pm.save(str(path), "PNG"))

    def save_table_csv(self, path: Path) -> None:
        """표를 CSV로(BOM 붙은 UTF-8: 한국어 윈도우의 엑셀에서 바로 열린다). 숫자는 꾸밈 없이."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, lineterminator="\r\n")
            w.writerow(S.CMP_CSV_HEADERS)
            for row in self.rows:
                w.writerow(row_csv(row, self.target))

    def _export_dir(self) -> Path:
        return self.runs_dir().parent

    def ask_save_path(self, title: str, name: str, file_filter: str) -> Path | None:
        """저장할 파일을 묻는다. 테스트는 이 메서드를 바꿔 끼운다."""
        path, _ = QFileDialog.getSaveFileName(self, title, str(self._export_dir() / name), file_filter)
        return Path(path) if path else None

    def show_error(self, message: str) -> None:
        """오류 상자(다음 할 일 + 오류 보고 묶음 단추, errors.show_error). 테스트는 이 메서드를 바꿔 끼운다."""
        from evolab.app import errors

        errors.show_error(self, S.CMP_ERROR_TITLE, message)

    def _on_save_png(self) -> None:
        path = self.ask_save_path(S.CMP_SAVE_PNG, S.CMP_PNG_NAME, S.CMP_PNG_FILTER)
        if path is None:
            return
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")
        if not self.save_plot_png(path):
            self.show_error(S.CMP_SAVE_FAILED.format(path=path, reason=""))

    def _on_save_csv(self) -> None:
        path = self.ask_save_path(S.CMP_SAVE_CSV, S.CMP_CSV_NAME, S.CMP_CSV_FILTER)
        if path is None:
            return
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        try:
            self.save_table_csv(path)
        except OSError as exc:
            from evolab.app import errors

            why = errors.reason(exc)  # 날것 예외 글 대신 한국어 까닭(v3.1, v3.0 고침 13)
            self.show_error(S.CMP_SAVE_FAILED.format(path=path, reason=why))

    # ------------------------------------------------------------------
    # 나란히 경주
    # ------------------------------------------------------------------
    def race_runs(self) -> list[tuple[Path, str]]:
        """경주에 넣을 실행(그래프 순서, 묶음은 구성 실행)과 그래프에서 그 실행에 준 색(묶음 구성 실행은 묶음 색)."""
        return [(e.path, s.color) for s in self.shown for e in s.entries]

    def _update_race_button(self) -> None:
        enough = len(self.race_runs()) >= 2
        self.race_button.setEnabled(enough)
        self.race_button.setToolTip(S.CMP_RACE_BUTTON_TIP if enough else
                                    S.CMP_RACE_BUTTON_TIP + "\n" + S.RACE_TOO_FEW)

    def open_race(self) -> Any:
        """고른 실행들의 마지막 챔피언으로 경주 창을 연다(이미 열려 있으면 그 창을 바꿔 앞으로). 열지 못하면 None."""
        from evolab.app.race.model import entries_from_runs
        from evolab.app.race.view import show_race

        runs = self.race_runs()
        if len(runs) < 2:
            self.show_error(S.RACE_TOO_FEW)
            return None
        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            entries, warnings, _n = entries_from_runs(runs, "last")
        finally:
            self.unsetCursor()
        if len(entries) < 2:  # 읽을 수 없는 실행을 빼고 나니 모자람
            self.show_error("\n".join([*warnings, S.RACE_TOO_FEW]))
            return None
        self.race_dialog = show_race(self, entries, title=S.RACE_TITLE, warnings=warnings, runs=runs)
        return self.race_dialog

    # ------------------------------------------------------------------
    # 반복 실행
    # ------------------------------------------------------------------
    @property
    def repeat_running(self) -> bool:
        return self.runner is not None and self.runner.running

    def make_repeat_dialog(self) -> RepeatDialog:
        """반복 실행 대화상자. 테스트는 이 메서드를 바꿔 끼운다."""
        return RepeatDialog(self, runs_dir=self.runs_dir())

    def open_repeat_dialog(self) -> RepeatRunner | None:
        if self.repeat_running:
            return None
        dlg = self.make_repeat_dialog()
        try:
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return None
            plan = dlg.plan()
        finally:
            dlg.deleteLater()
        return self.start_repeat(plan)

    def start_repeat(self, plan: RepeatPlan) -> RepeatRunner | None:
        """반복 실행을 시작한다. 이미 도는 중이면 None. runs 폴더를 정하지 않은 계획은 이 화면의 runs 폴더에."""
        if self.repeat_running:
            return None
        if plan.runs_dir is None:
            plan.runs_dir = self.runs_dir()
        runner = RepeatRunner(plan, self)
        runner.run_started.connect(self._on_repeat_run_started)
        runner.progress.connect(self._on_repeat_progress)
        runner.run_finished.connect(lambda _i, _p: self.refresh())
        runner.failed.connect(self._on_repeat_failed)
        runner.finished.connect(self._on_repeat_finished)
        self.runner = runner
        self._follow_group = str(runner.group["id"])
        self._last_repeat_message = ""
        self.repeat_status.hide()
        self.repeat_box.show()
        self.repeat_button.setEnabled(False)
        self._update_repeat_label()
        self.repeat_started.emit(self._follow_group)
        runner.start()
        return runner

    def cancel_repeat(self) -> None:
        if self.runner is not None and self.runner.running:
            self.setCursor(Qt.CursorShape.WaitCursor)
            try:
                self.runner.cancel()
            finally:
                self.unsetCursor()

    def shutdown(self) -> None:
        """프로그램을 닫을 때: 도는 반복 실행을 깨끗이 멈추고, 경주 창의 준비 스레드도 끝날 때까지 기다린다(주 창이
        닫혀도 경주 창은 남아 있고, 프로그램이 끝날 때 돌던 QThread가 지워지면 abort)."""
        from evolab.app.race.view import shutdown_race

        if self.runner is not None:
            self.runner.shutdown()
        shutdown_race(self.race_dialog)

    def _on_repeat_run_started(self, index: int, run_dir: Path) -> None:
        gid = self._follow_group
        follow = index == 0 or any(
            e.group is not None and e.group["id"] == gid and e.key in self._checked for e in self.entries
        )
        self.entries = list_entries(self.runs_dir())
        self._by_key = {e.key: e for e in self.entries}
        if follow:
            new = {e.key for e in self.entries if e.ok and _same(e.path, run_dir)}
            self._checked |= new
        self._checked &= {e.key for e in self.entries if e.ok}
        self._fill_tree()
        self.folder_label.setText(S.CMP_FOLDER.format(count=len(self.entries), path=self.runs_dir()))
        self.empty_label.setVisible(not self.entries)
        self.redraw(keep_view=index > 0)
        self._update_repeat_label()
        self.repeat_run_started.emit(Path(run_dir))

    def _on_repeat_progress(self, _index: int, _generation: int, _elapsed: float) -> None:
        self._update_repeat_label()
        if not self._live_timer.isActive():
            self._live_timer.start()

    def _update_repeat_label(self) -> None:
        r = self.runner
        if r is None:
            return
        row = r.last_row
        index = max(0, r.index)
        seed = r.plan.base_seed + index
        g = to_float(row.get("generation")) if row else _NAN
        gen = int(g) if math.isfinite(g) else 0
        evals = to_float(row.get("evaluations")) if row else _NAN
        elapsed = to_float(row.get("elapsed")) if row else _NAN
        self.repeat_label.setText(S.CMP_REPEAT_PROGRESS.format(
            name=r.group["name"], i=index + 1, n=r.plan.count, seed=seed, gen=gen,
            evals=format_count(evals) if math.isfinite(evals) else "0",
            time=format_duration(elapsed) if math.isfinite(elapsed) else format_duration(0.0),
        ))
        self.repeat_bar.setValue(round(r.fraction() * 1000))

    def _on_repeat_failed(self, message: str) -> None:
        self._last_repeat_message = S.CMP_REPEAT_FAILED.format(reason=message)

    def _on_repeat_finished(self, completed: bool) -> None:
        r = self.runner
        self.repeat_box.hide()
        self.repeat_button.setEnabled(True)
        self._live_timer.stop()
        if r is not None:
            if completed:
                msg = S.CMP_REPEAT_DONE.format(name=r.group["name"], n=r.plan.count)
            elif r.state == "cancelled":
                msg = S.CMP_REPEAT_CANCELLED.format(name=r.group["name"], done=max(0, r.index), n=r.plan.count)
            else:
                msg = self._last_repeat_message or S.CMP_REPEAT_FAILED.format(reason=r.error or "")
            self.repeat_status.setText(msg)
            self.repeat_status.show()
        self.refresh()
        self.repeat_finished.emit(bool(completed))
        if r is not None and not completed and r.state == "failed":
            self.show_error(self.repeat_status.text())

    # ------------------------------------------------------------------
    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh()  # 화면에 돌아올 때마다(새 실행이 생겼을 수 있다)


def _resolved(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _same(a: Path, b: Path) -> bool:
    return _resolved(Path(a)) == _resolved(Path(b))


__all__ = [
    "X_AXES", "Y_METRICS", "CompareView", "GroupCurve", "RunData", "RunEntry", "RunMetrics", "Shown", "TableRow",
    "aggregate", "ffill", "load_stats", "run_metrics", "sample", "series", "step_xy",
]

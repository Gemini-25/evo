"""실험실 「보는 재미」의 화면 쪽 기록 (v1.2 설계 2.5절, 4.1·4.2·4.4·4.5, D10·D12·D18·D19). 메모리만 쓴다(디스크에 쓰지 않음).

- RunHistory: 세대 → GenEntry. 관찰 기록(runs.observe.Observation)이 있는 세대와, 기록이 없어 세대 대표만 있는 세대(D12:
  v1.0·v1.1 세대, --no-observe, 옛 프로그램이 이어 간 세대)를 세대마다 따로 다룬다. 학습 중 받은 최근 KEEP_EXACT(64)세대의
  정확한 float64 집단(지도 점 재생, D19), 챔피언 목록(세대, 유전자), 파생 계열(몸 수치, 행동 비율, 궤적 밀도 창)과 유전자
  지도 축(GenePCA, D18)을 갖는다. 파생 값은 세대를 넣을 때 한 번 계산해 두고(GenEntry), 계열은 기록 판(version)이 바뀔
  때만 다시 잇는다.
- HistoryLoader(QThread): 실행을 열 때 세대 대표 파일(champions/gen_XXXXXX_representative.npz)을 배경에서 읽는다.
  관찰 키는 runs.observe.decode로 검사하고 같은 세대 통계 줄과 consistent(D10)일 때만 쓴다. 관찰 키가 없거나 검사에서
  떨어진 세대는 그 파일의 세대 대표 유전자로 대신한다. 통계 줄이 없는 세대(통계보다 뒤인 파일)는 읽지 않는다.
  LOAD_CHUNK(64)세대씩 묶어 보내고, 화면은 꼬리표(tag)가 바뀌었으면(되감기, 다른 실행) 앞 결과를 버린다.
- 지도 점은 float16 유전자(디스크와 세대 메시지의 obs_genomes는 같은 값)로 그린다: 학습 중 그림 = 다시 연 그림.

이 모듈의 계산(GenEntry, RunHistory, GenePCA, 순수 함수)은 Qt가 없어 배경 스레드에서 불러도 된다. 무작위 PCA의 난수는
np.random.default_rng(0)(설계 3.2: 전역 난수를 쓰지 않음).
"""

from __future__ import annotations

import bisect
import math
import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from PySide6.QtCore import QThread, Signal

from evolab.policy.genome import GenomeLayout
from evolab.runs import observe as ob
from evolab.runs.observe import GridFrame, Observation

KEEP_EXACT: Final = 64
"""학습 중 받은 정확한 float64 집단을 화면 메모리에 두는 최근 세대 수(D19)."""
LOAD_CHUNK: Final = 64
"""배경 읽기가 한 번에 보내는 세대 수."""
PCA_SMALL_DIM: Final = 256
"""이 차원 이하는 누적 공분산(ΣX, ΣXᵀX) + eigh, 넘으면 표본의 무작위 PCA(D18)."""
PCA_SAMPLE_ROWS: Final = 4096
PCA_RANK: Final = 4
"""무작위 범위 찾기의 무작위 벡터 수(두 축 + 여유 둘)."""
PCA_POWER_ITERS: Final = 2
PCA_SEED: Final = 0
DENS_RECENT: Final = 20
"""궤적 밀도 「최근 n세대」의 n."""
DENSITY_WINDOWS: Final = ("one", "recent", "all")
ACTION_MODES: Final = ("pop", "rep")
"""행동 분포: 집단 전체(obs_act_hist[0]) / 세대 대표(obs_act_hist[1])."""
SCORE_RANGE_SAMPLE: Final = 100_000
"""지도 점 색 범위(2~98% 백분위)를 셀 점수 수의 상한. 넘으면 고른 간격 표본으로 센다."""
REST_FRACTION: Final = 0.05
"""「쓰는 시간」의 쉬는 칸: 0이 범위 안이면 0 ± 범위의 5%에 걸친 칸(ACT_SUMMARY)."""


# ---------------------------------------------------------------------------
# 세대 하나
@dataclass(frozen=True, eq=False)
class GenEntry:
    """화면 메모리의 세대 하나. obs가 있으면 관찰 기록, 없으면 세대 대표만(D12). 배열은 읽기 전용으로 다룬다."""

    generation: int
    obs: Observation | None
    rep_genome16: np.ndarray | None
    """(D,) float16 세대 대표: 관찰이 없는 세대의 지도 점(관찰 유전자와 같은 정밀도). 관찰이 있으면 None."""
    rep_score: float
    """세대 대표의 학습 점수(파일의 score, 학습 중이면 통계 score_best)."""
    rep_index: int | None
    """세대 대표의 개체 번호(관찰이 있으면 obs.representative, 학습 중 메시지면 그 번호, 옛 파일이면 None)."""
    learner: int | None
    """그 세대를 만든 학습 방법 번호(관찰 머리 또는 통계 learner 열, 모르면 None). 지도 길을 여기서 끊는다."""
    body_mean: np.ndarray | None = None
    """(NB,) 몸 수치 집단 평균(관찰이 없으면 세대 대표 값). 몸 유전자가 없으면 None."""
    body_min: np.ndarray | None = None
    body_max: np.ndarray | None = None
    body_rep: np.ndarray | None = None
    act_frac: np.ndarray | None = None
    """(2, U, ACTION_BINS) float64: [모드, 채널]마다 칸 비율(합 1, 그 채널 표본이 없으면 0)."""
    act_mean_pop: np.ndarray | None = None
    """(U,) 개체별 평균 실제 출력의 집단 평균."""
    act_mean_rep: np.ndarray | None = None
    """(U,) 세대 대표의 평균 실제 출력."""
    mean_genome: np.ndarray | None = None
    """(D,) float16 지도 길의 점: 관찰 유전자의 집단 평균, 관찰이 없으면 세대 대표. 투영이 선형이라 지도 위 점은
    집단 점들의 평균과 같다. float16(지도 점과 같은 정밀도)이라 큰 신경망의 긴 실행에서도 메모리가 작다."""

    @property
    def observed(self) -> bool:
        return self.obs is not None

    @property
    def map_genomes(self) -> np.ndarray | None:
        """지도에 찍을 유전자 (N, D) float16: 관찰 유전자 전체, 관찰이 없으면 세대 대표 한 줄. 유전자 간격으로 건너뛴
        세대(관찰은 있으나 유전자가 없음)는 None."""
        if self.obs is not None:
            return self.obs.genomes
        if self.rep_genome16 is not None:
            return self.rep_genome16.reshape(1, -1)
        return None

    @property
    def map_rep(self) -> int:
        """map_genomes 안에서 세대 대표의 줄 번호."""
        return self.obs.representative if self.obs is not None else 0

    @property
    def scores(self) -> np.ndarray:
        """지도 점의 점수(map_genomes 줄 순서)."""
        if self.obs is not None:
            return self.obs.scores
        return np.array([self.rep_score], dtype=np.float64)


def _clip16(genome: np.ndarray) -> np.ndarray:
    return np.clip(genome, -ob.FLOAT16_MAX, ob.FLOAT16_MAX).astype(np.float16)


def _genome64(value: Any, dim: int) -> np.ndarray | None:
    """파일·메시지의 유전자 하나를 검사한 float64 복사본(모양이 다르거나 유한하지 않으면 None)."""
    try:
        g = np.array(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if g.shape != (dim,) or not np.isfinite(g).all():
        return None
    return g


def make_entry(generation: int, obs: Observation | None, rep_genome: np.ndarray | None, rep_score: float,
               layout: GenomeLayout, *, rep_index: int | None = None, learner: int | None = None) -> GenEntry:
    """세대 하나의 GenEntry: 파생 값(몸 수치 요약, 행동 비율, 지도 길의 점)을 한 번에 계산한다. obs가 없으면 rep_genome
    (float64 세대 대표)로 대신한다(D12). 둘 다 없으면 빈 항목(점수만)."""
    rep64 = None if rep_genome is None else _genome64(rep_genome, layout.dim)
    body = None
    if obs is not None and obs.body is not None and obs.body.shape[1] > 0:
        b = np.asarray(obs.body, dtype=np.float64)
        body = (b.mean(axis=0), b.min(axis=0), b.max(axis=0), b[obs.representative].copy())
    elif obs is None and rep64 is not None and layout.n_body > 0:
        try:
            v = layout.body_values(rep64)
        except ValueError:  # GenomeError(NaN)는 _genome64가 이미 거른다
            v = None
        if v is not None:
            body = (v, v.copy(), v.copy(), v.copy())
    act_frac = act_pop = act_rep = None
    if obs is not None and obs.act_hist is not None and obs.act_mean is not None:
        h = np.asarray(obs.act_hist, dtype=np.float64)
        tot = h.sum(axis=2, keepdims=True)
        act_frac = np.divide(h, tot, out=np.zeros_like(h), where=tot > 0)
        am = np.asarray(obs.act_mean, dtype=np.float64)
        act_pop = am.mean(axis=0)
        act_rep = am[obs.representative].copy()
    rep16 = mean = None
    if obs is not None:
        if obs.genomes is not None:
            mean = _clip16(np.asarray(obs.genomes, dtype=np.float64).mean(axis=0))
    elif rep64 is not None:
        rep16 = _clip16(rep64)
        rep16.flags.writeable = False
        mean = rep16
    if mean is not None:
        mean.flags.writeable = False
    if obs is not None:
        learner = obs.learner
        rep_index = obs.representative
    return GenEntry(
        generation=int(generation), obs=obs, rep_genome16=rep16, rep_score=float(rep_score), rep_index=rep_index,
        learner=None if learner is None else int(learner),
        body_mean=None if body is None else body[0], body_min=None if body is None else body[1],
        body_max=None if body is None else body[2], body_rep=None if body is None else body[3],
        act_frac=act_frac, act_mean_pop=act_pop, act_mean_rep=act_rep, mean_genome=mean,
    )


def _finite_int(value: Any) -> int | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return int(v) if math.isfinite(v) and v == int(v) else None


def _finite_float(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.nan
    return v


def live_entry(generation: int, observation: Mapping[str, Any] | None, population: np.ndarray | None,
               representative: int, layout: GenomeLayout, *, learner: Any = None,
               rep_score: Any = math.nan) -> tuple[GenEntry | None, np.ndarray | None]:
    """학습 중 세대 메시지 → (GenEntry 또는 None, 정확한 float64 집단 또는 None). 관찰 기록은 디스크와 같은 검사(decode)를
    거친다. 관찰이 없으면(관찰을 끈 학습, 요약 실패) 집단의 세대 대표로 대신한다."""
    obs = None
    if observation is not None:
        obs, _reason = ob.decode(observation, generation=int(generation), dim=layout.dim, n_body=layout.n_body,
                                 n_out=layout.brain.n_out)
    exact = None
    rep64 = None
    rep = obs.representative if obs is not None else int(representative)
    if population is not None:
        try:
            pop = np.array(population, dtype=np.float64)
        except (TypeError, ValueError):
            pop = None
        if pop is not None and pop.ndim == 2 and pop.shape[1] == layout.dim and pop.shape[0] > 0 and \
                np.isfinite(pop).all():
            pop.flags.writeable = False
            exact = pop
            if 0 <= rep < pop.shape[0]:
                rep64 = pop[rep]
    if obs is None and rep64 is None:
        return None, exact
    score = float(obs.scores[obs.representative]) if obs is not None else _finite_float(rep_score)
    return make_entry(int(generation), obs, rep64, score, layout, rep_index=rep if rep64 is not None else None,
                      learner=_finite_int(learner)), exact


# ---------------------------------------------------------------------------
# 유전자 지도 축 (D18)
def covariance_axes(s1: np.ndarray, s2: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """누적 합 ΣX(s1), ΣXᵀX(s2), 행 수 n → (평균 (D,), 두 축 (2, D), 설명 분산 비율 (2,)). 부호는 정하지 않는다."""
    mean = s1 / n
    cov = s2 / n - np.outer(mean, mean)
    cov = (cov + cov.T) / 2.0
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    d = len(w)
    axes = np.zeros((2, d))
    explained = np.zeros(2)
    total = float(w.sum())
    for k in range(min(2, d)):
        axes[k] = v[:, d - 1 - k]
        explained[k] = float(w[d - 1 - k]) / total if total > 0 else 0.0
    return mean, axes, explained


def randomized_axes(sample: np.ndarray, *, seed: int = PCA_SEED, rank: int = PCA_RANK,
                    power: int = PCA_POWER_ITERS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """큰 차원(D > PCA_SMALL_DIM)의 표본 (m, D) → (평균, 두 축 (2, D), 설명 분산 비율 (2,)). 무작위 범위 찾기(rank개
    무작위 벡터, 거듭제곱 power회) + 작은 SVD. 난수는 default_rng(seed)라 같은 표본이면 결과가 같다."""
    x = np.asarray(sample, dtype=np.float64)
    m, d = x.shape
    mean = x.mean(axis=0)
    xc = x - mean
    k = max(1, min(rank, d, m))
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(xc @ rng.standard_normal((d, k)))
    for _ in range(power):
        z, _ = np.linalg.qr(xc.T @ q)
        q, _ = np.linalg.qr(xc @ z)
    _, s, vt = np.linalg.svd(q.T @ xc, full_matrices=False)
    axes = np.zeros((2, d))
    explained = np.zeros(2)
    total = float((xc * xc).sum())
    for j in range(min(2, len(s))):
        axes[j] = vt[j]
        explained[j] = float(s[j] * s[j]) / total if total > 0 else 0.0
    return mean, axes, explained


def align_signs(axes: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    """축 부호: 앞 축이 있으면 내적이 음수인 축을 뒤집고, 처음이면 절댓값이 가장 큰 성분이 양수가 되게."""
    out = np.array(axes, dtype=np.float64)
    for k in range(out.shape[0]):
        if previous is not None and previous.shape == out.shape and float(previous[k] @ out[k]) != 0.0:
            if float(previous[k] @ out[k]) < 0.0:
                out[k] = -out[k]
        elif out.shape[1] and out[k, int(np.argmax(np.abs(out[k])))] < 0.0:
            out[k] = -out[k]
    return out


def sample_rows(blocks: Sequence[np.ndarray], limit: int = PCA_SAMPLE_ROWS) -> np.ndarray:
    """세대별 유전자 블록들에서 고르게 뽑은 ≤ limit행(float64). 모두 합쳐 limit 이하면 전부."""
    sizes = np.array([len(b) for b in blocks], dtype=np.int64)
    total = int(sizes.sum())
    if total == 0:
        return np.zeros((0, blocks[0].shape[1] if blocks else 0))
    if total <= limit:
        return np.concatenate([np.asarray(b, dtype=np.float64) for b in blocks])
    pick = np.unique(np.linspace(0, total - 1, limit).round().astype(np.int64))
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    which = np.searchsorted(starts, pick, side="right") - 1
    rows = [np.asarray(blocks[b][p - starts[b]], dtype=np.float64) for b, p in zip(which.tolist(), pick.tolist(),
                                                                                    strict=True)]
    return np.vstack(rows)


class GenePCA:
    """유전자 지도의 두 축(D18): 지금까지 기록한 모든 세대의 PCA.

    D ≤ PCA_SMALL_DIM이면 세대가 들어올 때 ΣX, ΣXᵀX, 행 수를 누적하고 맞출 때 eigh. 더 크면 누적하지 않고 표본
    (sample_rows)의 무작위 PCA를 쓰는 쪽(화면이 배경 스레드에서 randomized_axes를 돌려 set_axes로 넣음).
    맞추는 때(wants_fit): 첫 세대, 기록 세대 수 ≥ 2 × 마지막으로 맞춘 세대 수. 단추는 force.
    """

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.big = self.dim > PCA_SMALL_DIM
        self.s1 = None if self.big else np.zeros(self.dim)
        self.s2 = None if self.big else np.zeros((self.dim, self.dim))
        self.rows = 0
        self.generations = 0
        self.fitted_generations = 0
        """마지막으로 맞춘 때의 기록 세대 수(0 = 아직)."""
        self.fits = 0
        self.mean: np.ndarray | None = None
        self.axes: np.ndarray | None = None
        """(2, D) 단위 벡터."""
        self.explained = np.zeros(2)
        self.version = 0
        """축이 바뀔 때마다 늘어난다(투영 캐시 무효화)."""

    def add(self, genomes: np.ndarray) -> None:
        """세대 하나의 유전자 (N, D)(float16 가능)를 더한다."""
        x = np.asarray(genomes, dtype=np.float64).reshape(-1, self.dim)
        if self.s1 is not None and self.s2 is not None:
            self.s1 += x.sum(axis=0)
            self.s2 += x.T @ x
        self.rows += int(x.shape[0])
        self.generations += 1

    def wants_fit(self) -> bool:
        return self.generations > 0 and (self.fitted_generations == 0 or
                                         self.generations >= 2 * self.fitted_generations)

    def fit(self) -> bool:
        """누적 공분산으로 지금 맞춘다(작은 차원만). 맞췄으면 True."""
        if self.big or self.rows == 0 or self.s1 is None or self.s2 is None:
            return False
        mean, axes, explained = covariance_axes(self.s1, self.s2, self.rows)
        self.set_axes(mean, axes, explained)
        return True

    def set_axes(self, mean: np.ndarray, axes: np.ndarray, explained: np.ndarray,
                 generations: int | None = None) -> None:
        """새 축을 넣는다(부호는 앞 축 쪽으로). generations: 그 축을 맞출 때의 기록 세대 수(무작위 경로가 배경에서
        계산하는 동안 세대가 더 들어와도 두 배 규칙이 맞게)."""
        self.axes = align_signs(axes, self.axes)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.explained = np.asarray(explained, dtype=np.float64)
        self.fitted_generations = self.generations if generations is None else int(generations)
        self.fits += 1
        self.version += 1

    def project(self, genomes: np.ndarray) -> np.ndarray:
        """(N, D) → (N, 2) float32 지도 좌표. 축이 없으면 (N, 2) NaN."""
        x = np.asarray(genomes, dtype=np.float64).reshape(-1, self.dim)
        if self.axes is None or self.mean is None:
            return np.full((x.shape[0], 2), np.nan, dtype=np.float32)
        return ((x - self.mean) @ self.axes.T).astype(np.float32)


# ---------------------------------------------------------------------------
# 파생 계열
@dataclass(frozen=True, eq=False)
class BodySeries:
    gens: np.ndarray
    """(n,) float64 세대."""
    mean: np.ndarray
    """(n, NB) 물리 단위."""
    lo: np.ndarray
    hi: np.ndarray
    rep: np.ndarray
    observed: np.ndarray
    """(n,) bool: 집단 기록이 있는 세대(없으면 세대 대표 값만)."""


@dataclass(frozen=True, eq=False)
class ActionSeries:
    first: int
    """첫 열의 세대(열 = first, first + 1, …, 빈 세대는 NaN 열)."""
    frac: np.ndarray
    """(열 수, ACTION_BINS) 시간 비율(열 합 1), 기록 없는 세대는 NaN."""
    mean: np.ndarray
    """(열 수,) 평균 실제 출력(모드에 따라 집단 평균 또는 세대 대표), 없으면 NaN."""

    @property
    def gens(self) -> np.ndarray:
        return np.arange(self.first, self.first + self.frac.shape[0], dtype=np.float64)


@dataclass(frozen=True, eq=False)
class DensityView:
    """보는 세대의 궤적 밀도(같은 세계 열쇠·틀인 세대만 합친 것)."""

    generation: int
    frame: GridFrame
    world_key: str
    counts: np.ndarray
    """(ny, nx) float64 칸별 표본 수(행 = iy, 0행이 y0 쪽)."""
    merged: tuple[int, ...]
    """합친 세대(오름차순)."""
    outside: int
    """합친 세대의 틀 밖 표본 수."""
    population: int
    """보는 세대의 개체 수."""
    cannot_merge: bool
    """창이 넓은데 세계가 달라 그 세대만 보임(DENS_CANNOT_MERGE)."""

    @property
    def samples(self) -> float:
        return float(self.counts.sum()) + self.outside


def rest_bins(low: float, high: float, bins: int = ob.ACTION_BINS) -> list[int]:
    """「쓰는 시간」에서 쉬는 칸: 0이 범위 안이면 0 ± 범위의 5%에 걸친 칸(0..1 → [0], −1..1 → [9, 10]), 아니면 [0]."""
    if not low <= 0.0 <= high:
        return [0]
    w = (high - low) / bins
    r = REST_FRACTION * (high - low)
    out = [b for b in range(bins) if low + b * w < r - 1e-12 and low + (b + 1) * w > -r + 1e-12]
    return out or [0]


def on_fraction(frac: np.ndarray, low: float, high: float) -> float:
    """칸 비율 (ACTION_BINS,) → 쓰는 시간 = 1 − 쉬는 칸의 비율."""
    return float(1.0 - frac[rest_bins(low, high, len(frac))].sum())


# ---------------------------------------------------------------------------
class RunHistory:
    """실행 하나의 「보는 재미」 기록(화면 메모리). 세대는 순서와 상관없이 넣어도 된다(배경 읽기와 학습 중 세대가 섞임)."""

    def __init__(self) -> None:
        self.layout: GenomeLayout | None = None
        self.run_dir: Path | None = None
        self.load_summary: dict[str, int] = {}
        """마지막 배경 읽기 결과 수(observed, fallback, rejected, unreadable)."""
        self.version = 0
        """기록이 바뀔 때마다 늘어난다(타일의 계열 캐시 무효화)."""
        self.epoch = 0
        """clear할 때만 늘어난다(배경에서 계산한 결과가 같은 기록의 것인지 보는 꼬리표)."""
        self.pca: GenePCA | None = None
        self.clear()

    # -- 넣기 -------------------------------------------------------------------
    def clear(self) -> None:
        self._entries: dict[int, GenEntry] = {}
        self.gens: list[int] = []
        """기록한 세대(오름차순)."""
        self._exact: OrderedDict[int, np.ndarray] = OrderedDict()
        self._champions: list[tuple[int, np.ndarray, float]] = []
        self.run_dir = None
        self.load_summary = {}
        self.pca = GenePCA(self.layout.dim) if self.layout is not None else None
        self._pca_stale = False
        self._cache: dict[Any, Any] = {}
        self.version += 1
        self.epoch += 1

    def set_layout(self, layout: GenomeLayout | None) -> None:
        """유전자 구조. 바뀌면 기록을 비운다."""
        same = (layout is None and self.layout is None) or (
            layout is not None and self.layout is not None and layout.signature() == self.layout.signature())
        self.layout = layout
        if not same:
            self.clear()

    def add(self, entry: GenEntry, *, exact: np.ndarray | None = None) -> None:
        """세대 하나를 넣는다(같은 세대가 있으면 바꾼다). exact: 그 세대의 정확한 float64 집단(최근 64세대만 둔다)."""
        g = int(entry.generation)
        old = self._entries.get(g)
        self._entries[g] = entry
        if old is None:
            bisect.insort(self.gens, g)
            genomes = entry.map_genomes
            if self.pca is not None and genomes is not None and genomes.shape[1] == self.pca.dim:
                self.pca.add(genomes)
        else:
            self._pca_stale = True
        if exact is not None:
            self._exact[g] = exact
            self._exact.move_to_end(g)
            while len(self._exact) > KEEP_EXACT:
                self._exact.popitem(last=False)
        self.version += 1

    def add_many(self, entries: Iterable[GenEntry]) -> None:
        for e in entries:
            self.add(e)

    def add_champion(self, generation: int, genome: Any, score: float = math.nan) -> None:
        """챔피언 (뽑힌 세대, 유전자 float64, 검증 점수). 같은 세대면 바꾼다."""
        if self.layout is None:
            return
        g = _genome64(genome, self.layout.dim)
        if g is None:
            return
        g.flags.writeable = False
        gen = int(generation)
        keys = [c[0] for c in self._champions]
        i = bisect.bisect_left(keys, gen)
        item = (gen, g, _finite_float(score))
        if i < len(keys) and keys[i] == gen:
            old = self._champions[i]
            if old[1].tobytes() == g.tobytes() and (math.isnan(item[2]) or old[2] == item[2]):
                return
            if math.isnan(item[2]):
                item = (gen, g, old[2])
            self._champions[i] = item
        else:
            self._champions.insert(i, item)
        self.version += 1

    # -- 읽기 -------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.gens)

    def entry(self, generation: int | None) -> GenEntry | None:
        return None if generation is None else self._entries.get(int(generation))

    @property
    def latest(self) -> int | None:
        return self.gens[-1] if self.gens else None

    def _cached(self, key: Any, make: Any) -> Any:
        hit = self._cache.get(key)
        if hit is not None and hit[0] == self.version:
            return hit[1]
        value = make()
        self._cache[key] = (self.version, value)
        return value

    @property
    def first_observed(self) -> int | None:
        """관찰 기록이 있는 첫 세대(없으면 None)."""
        return self._cached("first_obs", lambda: next((g for g in self.gens if self._entries[g].obs is not None), None))

    @property
    def unobserved_count(self) -> int:
        return self._cached("unobs", lambda: sum(1 for e in self._entries.values() if e.obs is None))

    def genome_gens(self) -> list[int]:
        """지도 점이 있는 세대(관찰 유전자 또는 세대 대표)."""
        return self._cached("ggens", lambda: [g for g in self.gens if self._entries[g].map_genomes is not None])

    def nearest_genome_gen(self, generation: int) -> int | None:
        """지도 점이 있는 가장 가까운 세대(같으면 앞 세대)."""
        gg = self.genome_gens()
        if not gg:
            return None
        i = bisect.bisect_left(gg, generation)
        if i < len(gg) and gg[i] == generation:
            return generation
        cands = [gg[j] for j in (i - 1, i) if 0 <= j < len(gg)]
        return min(cands, key=lambda c: (abs(c - generation), c))

    def previous_genome_gens(self, generation: int, k: int = 3) -> list[int]:
        """generation 앞(더 작은) 지도 점 세대 k개, 가까운 것부터."""
        gg = self.genome_gens()
        i = bisect.bisect_left(gg, generation)
        return list(reversed(gg[max(0, i - k):i]))

    def exact_genome(self, generation: int, index: int) -> np.ndarray | None:
        pop = self._exact.get(int(generation))
        if pop is None or not 0 <= int(index) < pop.shape[0]:
            return None
        return np.array(pop[int(index)], dtype=np.float64)

    def has_exact(self, generation: int) -> bool:
        return int(generation) in self._exact

    def rep_genome(self, generation: int) -> np.ndarray | None:
        """세대 대표의 정확한 유전자(float64 복사본): 화면 메모리의 집단, 없으면 세대 대표 파일의 genome."""
        g = int(generation)
        e = self._entries.get(g)
        pop = self._exact.get(g)
        if pop is not None and e is not None and e.rep_index is not None and 0 <= e.rep_index < pop.shape[0]:
            return np.array(pop[e.rep_index], dtype=np.float64)
        if self.run_dir is None or self.layout is None:
            return None
        got = ob.read_representative(Path(self.run_dir) / "champions" / f"gen_{g:06d}_representative.npz")
        if got is None:
            return None
        return _genome64(got[0].get("genome"), self.layout.dim)

    @property
    def champions(self) -> list[tuple[int, np.ndarray, float]]:
        return list(self._champions)

    def champion_entry_at(self, generation: int) -> tuple[int, np.ndarray, float] | None:
        """generation 이하에서 마지막으로 뽑힌 챔피언 (세대, 유전자, 검증 점수)."""
        keys = [c[0] for c in self._champions]
        i = bisect.bisect_right(keys, int(generation)) - 1
        return self._champions[i] if i >= 0 else None

    def champion_at(self, generation: int) -> tuple[int, np.ndarray] | None:
        c = self.champion_entry_at(generation)
        return None if c is None else (c[0], np.array(c[1], dtype=np.float64))

    # -- 유전자 지도 ---------------------------------------------------------
    def ensure_pca(self) -> GenePCA | None:
        """세대를 바꿔 넣었으면 누적 합을 처음부터 다시 만든다(축은 그대로 두고 부호 기준으로 씀)."""
        if self.pca is not None and self._pca_stale and self.layout is not None:
            old = self.pca
            pca = GenePCA(self.layout.dim)
            for g in self.gens:
                genomes = self._entries[g].map_genomes
                if genomes is not None and genomes.shape[1] == pca.dim:
                    pca.add(genomes)
            pca.axes, pca.mean, pca.explained = old.axes, old.mean, old.explained
            pca.fitted_generations, pca.fits, pca.version = old.fitted_generations, old.fits, old.version + 1
            self.pca = pca
            self._pca_stale = False
        return self.pca

    def pca_sample(self) -> np.ndarray:
        """무작위 PCA 표본: 지도 점 세대들에서 고르게 뽑은 ≤ PCA_SAMPLE_ROWS행."""
        blocks = [self._entries[g].map_genomes for g in self.genome_gens()]
        return sample_rows([b for b in blocks if b is not None])

    def score_range(self) -> tuple[float, float] | None:
        """지도 점 색의 점수 범위: 읽은 모든 세대 점수의 2~98% 백분위(세대마다 색이 바뀌지 않게)."""
        def make() -> tuple[float, float] | None:
            parts = [self._entries[g].scores for g in self.gens]
            if not parts:
                return None
            s = np.concatenate(parts)
            if len(s) > SCORE_RANGE_SAMPLE:  # 아주 긴 실행: 고른 간격 표본(결정적)으로 백분위
                s = s[:: math.ceil(len(s) / SCORE_RANGE_SAMPLE)]
            s = s[np.isfinite(s)]
            if not len(s):
                return None
            lo, hi = (float(v) for v in np.percentile(s, [2.0, 98.0]))
            if not hi > lo:
                pad = max(abs(hi) * 0.05, 1e-6)
                lo, hi = lo - pad, hi + pad
            return lo, hi

        return self._cached("score_range", make)

    def trail(self) -> list[tuple[int, int | None, np.ndarray]]:
        """지도 길의 점들: (세대, 학습 방법 번호, 집단 가운데 (D,) float16), 세대 순서. 화면은 학습 방법 번호가 바뀐
        곳에서 길을 끊고, 세대마다 투영한 점을 축 판(GenePCA.version)마다 한 번만 계산해 둔다."""
        return self._cached("trail", lambda: [(g, e.learner, e.mean_genome) for g in self.gens
                                              if (e := self._entries[g]).mean_genome is not None])

    # -- 몸 수치·행동·밀도 ----------------------------------------------------
    def body_series(self) -> BodySeries | None:
        def make() -> BodySeries | None:
            es = [self._entries[g] for g in self.gens if self._entries[g].body_mean is not None]
            if not es:
                return None
            return BodySeries(
                gens=np.array([e.generation for e in es], dtype=np.float64),
                mean=np.vstack([e.body_mean for e in es]), lo=np.vstack([e.body_min for e in es]),
                hi=np.vstack([e.body_max for e in es]), rep=np.vstack([e.body_rep for e in es]),
                observed=np.array([e.obs is not None for e in es], dtype=bool),
            )

        return self._cached("body", make)

    def action_series(self, channel: int, mode: str) -> ActionSeries | None:
        """채널 하나의 세대 × 20칸 비율과 평균 출력(mode = "pop" | "rep"). 기록이 하나도 없으면 None."""
        m = ACTION_MODES.index(mode)

        def make() -> ActionSeries | None:
            have = [g for g in self.gens if self._entries[g].act_frac is not None]
            if not have:
                return None
            first, last = self.gens[0], self.gens[-1]
            n = last - first + 1
            frac = np.full((n, ob.ACTION_BINS), np.nan)
            mean = np.full(n, np.nan)
            for g in have:
                e = self._entries[g]
                assert e.act_frac is not None and e.act_mean_pop is not None and e.act_mean_rep is not None
                if not 0 <= channel < e.act_frac.shape[1]:
                    continue
                frac[g - first] = e.act_frac[m, channel]
                mean[g - first] = (e.act_mean_pop if m == 0 else e.act_mean_rep)[channel]
            return ActionSeries(first=first, frac=frac, mean=mean)

        return self._cached(("act", channel, mode), make)

    def density(self, generation: int | None, window: str, recent: int = DENS_RECENT) -> DensityView | None:
        """보는 세대의 궤적 밀도. window: "one"(이 세대만) | "recent"(최근 recent세대) | "all"(처음부터). 창 안에서 보는
        세대와 세계 열쇠·틀이 같은 세대만 칸별로 더한다(정수 합이라 순서와 무관하게 정확). 보는 세대에 기록이 없으면 None."""
        if window not in DENSITY_WINDOWS:
            raise ValueError(f"unknown density window {window!r}")
        e = self.entry(generation)
        if e is None or e.obs is None or e.obs.frame is None or e.obs.world_key is None:
            return None
        gen = e.generation

        def make() -> DensityView:
            obs = e.obs
            assert obs is not None and obs.frame is not None and obs.world_key is not None
            frame, key = obs.frame, obs.world_key
            if window == "one":
                cand = [gen]
            else:
                lo = gen - recent + 1 if window == "recent" else -math.inf
                i0 = bisect.bisect_left(self.gens, lo) if window == "recent" else 0
                i1 = bisect.bisect_right(self.gens, gen)
                cand = self.gens[i0:i1]
            with_density: list[int] = []
            merged: list[int] = []
            for g in cand:
                o = self._entries[g].obs
                if o is None or o.frame is None:
                    continue
                with_density.append(g)
                if o.world_key == key and o.frame == frame:
                    merged.append(g)
            size = frame.nx * frame.ny
            obs_list = [o for g in merged if (o := self._entries[g].obs) is not None and o.dens_cells is not None
                        and o.dens_counts is not None]
            if obs_list:  # 원래 형(int32·uint32)대로 한 번에 이은 뒤 바꾼다(세대마다 바꾸면 긴 실행에서 느림)
                cells = np.concatenate([o.dens_cells for o in obs_list]).astype(np.int64)  # type: ignore[misc]
                counts = np.concatenate([o.dens_counts for o in obs_list]).astype(np.float64)  # type: ignore[misc]
                total = np.bincount(cells, weights=counts, minlength=size)
            else:
                total = np.zeros(size)
            outside = int(sum(o.outside for o in obs_list))
            return DensityView(
                generation=gen, frame=frame, world_key=key, counts=total.reshape(frame.ny, frame.nx),
                merged=tuple(merged), outside=outside, population=obs.population,
                cannot_merge=window != "one" and len(merged) == 1 and len(with_density) > 1,
            )

        return self._cached(("dens", gen, window, recent), make)


# ---------------------------------------------------------------------------
# 배경 읽기
def _row_generation(row: Mapping[str, Any]) -> int | None:
    return _finite_int(row.get("generation"))


def _listing(folder: Path, kind: str) -> set[int]:
    try:
        names = os.listdir(folder)
    except OSError:
        return set()
    suffix = f"_{kind}.npz"
    out: set[int] = set()
    for n in names:
        if n.startswith("gen_") and n.endswith(suffix):
            digits = n[4:-len(suffix)]
            if digits.isdigit():
                out.add(int(digits))
    return out


def load_generation(path: Path, generation: int, row: Mapping[str, Any] | None,
                    layout: GenomeLayout) -> tuple[GenEntry | None, str]:
    """세대 대표 파일 하나 → (GenEntry 또는 None, 결과 종류). 종류: "observed"(관찰 기록을 씀), "fallback"(관찰 키가
    없어 세대 대표로), "rejected"(관찰 키가 있으나 검사·일관성에서 떨어져 세대 대표로), "unreadable"(파일을 못 읽음)."""
    got = ob.read_representative(path)
    if got is None:
        return None, "unreadable"
    base, keys = got
    obs, reason = ob.decode(keys, generation=generation, dim=layout.dim, n_body=layout.n_body,
                            n_out=layout.brain.n_out)
    kind = "observed"
    if obs is not None and not ob.consistent(obs, row, base.get("score")):
        obs = None
    if obs is None:
        kind = "fallback" if reason == ob.REASON_NONE else "rejected"
    genome = _genome64(base.get("genome"), layout.dim)
    score = _finite_float(np.asarray(base["score"]).reshape(()) if "score" in base else math.nan)
    if obs is None and genome is None:
        return None, "unreadable"
    learner = None if row is None else _finite_int(row.get("learner"))
    return make_entry(generation, obs, genome if obs is None else None, score, layout, learner=learner), kind


class HistoryLoader(QThread):
    """실행 폴더의 세대 대표 파일을 배경에서 읽는다(설계 D13). 통계 줄이 있는 세대만, 세대 순서로.

    신호(모두 첫 인수가 꼬리표 tag): champions(tag, [(세대, 유전자, 검증 점수)]) — 챔피언 파일(먼저 한 번),
    chunk(tag, [GenEntry], 읽은 세대 수, 전체 세대 수) — LOAD_CHUNK세대마다, loaded(tag, 결과 수 사전) — 끝.
    requestInterruption()이면 그만 읽는다(보낸 결과는 화면이 꼬리표로 버린다).
    """

    champions = Signal(int, object)
    chunk = Signal(int, object, int, int)
    loaded = Signal(int, object)

    def __init__(self, run_dir: Path, rows: Sequence[Mapping[str, Any]], layout: GenomeLayout, tag: int,
                 chunk_size: int = LOAD_CHUNK) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.rows = [dict(r) for r in rows]
        self.layout = layout
        self.tag = int(tag)
        self.chunk_size = max(1, int(chunk_size))
        self.summary: dict[str, int] = {}
        _LIVE_LOADERS.add(self)
        self.finished.connect(self._forget)

    def _forget(self) -> None:
        # finished는 스레드가 완전히 끝나기 직전에 온다: 마지막 참조를 놓기 전에 끝나기를 기다린다
        self.wait(2000)
        _LIVE_LOADERS.discard(self)

    def run(self) -> None:  # 배경 스레드
        folder = self.run_dir / "champions"
        by_gen: dict[int, Mapping[str, Any]] = {}
        for r in self.rows:
            g = _row_generation(r)
            if g is not None:
                by_gen[g] = r
        gens = sorted(by_gen)
        reps = _listing(folder, "representative")
        last = gens[-1] if gens else -1
        champs: list[tuple[int, np.ndarray, float]] = []
        for g in sorted(_listing(folder, "champion")):
            if g > last or self.isInterruptionRequested():
                continue
            got = ob.read_representative(folder / f"gen_{g:06d}_champion.npz")
            genome = None if got is None else _genome64(got[0].get("genome"), self.layout.dim)
            if genome is not None and got is not None:
                score = _finite_float(np.asarray(got[0]["score"]).reshape(())) if "score" in got[0] else math.nan
                champs.append((g, genome, score))
        self.champions.emit(self.tag, champs)
        summary = {"observed": 0, "fallback": 0, "rejected": 0, "unreadable": 0, "missing": 0}
        batch: list[GenEntry] = []
        done = 0
        total = len(gens)
        for g in gens:
            if self.isInterruptionRequested():
                return
            done += 1
            if g not in reps:
                summary["missing"] += 1
            else:
                entry, kind = load_generation(folder / f"gen_{g:06d}_representative.npz", g, by_gen[g], self.layout)
                summary[kind] += 1
                if entry is not None:
                    batch.append(entry)
            if len(batch) >= self.chunk_size:
                self.chunk.emit(self.tag, batch, done, total)
                batch = []
        if batch or total == 0:
            self.chunk.emit(self.tag, batch, done, total)
        self.summary = summary
        self.loaded.emit(self.tag, summary)


_LIVE_LOADERS: set[HistoryLoader] = set()
"""도는 중인 읽기 스레드를 끝날 때까지 붙잡아 둔다(대시보드가 먼저 사라져도 QThread가 돌던 중에 지워지지 않게)."""

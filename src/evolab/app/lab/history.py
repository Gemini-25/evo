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
- GuiPriority(GUI_PRIORITY): 화면 스레드가 무거운 일(펌프 한 단계, 기록 묶음 넣기, 실행 열기)을 하는 동안 화면 프로세스의
  배경 작업(이 읽기, 유전자 지도 PCA, 뇌 연결선, 다시 시뮬레이션·세계 만들기)이 쉬게 한다. 둘은 GIL 하나를 나눠 쓴다.
- 지도 점은 float16 유전자(디스크와 세대 메시지의 obs_genomes는 같은 값)로 그린다: 학습 중 그림 = 다시 연 그림.
- v6.1 자라는 뇌: 통계 열 brain_neurons·brain_links·grow(GROW_COLUMNS, 자라는 실행만)를 읽는 grow_columns·grow_series.
  열이 없는 실행(크기 고정·옛 실행)은 빈 값(None·빈 사전)이다. 「뇌 크기」 타일(grow_tile.py)이 쓴다.

이 모듈의 계산(GenEntry, RunHistory, GenePCA, 순수 함수)은 Qt가 없어 배경 스레드에서 불러도 된다. 무작위 PCA의 난수는
np.random.default_rng(0)(설계 3.2: 전역 난수를 쓰지 않음).
"""

from __future__ import annotations

import bisect
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.ma  # noqa: F401  np.percentile(score_range)이 처음 불릴 때 불러온다: 화면 스레드의 첫 지도 갱신에서 7~13ms
from PySide6.QtCore import QThread, Signal

from evolab.policy.genome import GenomeLayout
from evolab.runs import observe as ob
from evolab.runs.observe import GridFrame, Observation

KEEP_EXACT: Final = 64
"""학습 중 받은 정확한 float64 집단을 화면 메모리에 두는 최근 세대 수(D19)."""
EXACT_BUDGET: Final = 64 * 1024 * 1024
"""정확한 float64 집단에 쓰는 화면 메모리 상한(바이트). 세대 수(KEEP_EXACT)와 함께 지킨다: 큰 신경망·큰 집단이면 더
적은 세대만 둔다(내장 환경 기본값은 64세대가 3~8MB). 한 세대만으로 넘으면 두지 않는다(지도 점 재생은 세대 대표로 대신)."""
LOAD_CHUNK: Final = 64
"""배경 읽기가 한 번에 보내는 세대 수."""
LOAD_YIELD_S: Final = 0.0005
"""배경 읽기가 파일 하나를 읽기 전에 자는 시간(초). 파일 읽기는 GIL을 자주 놓았다가 곧바로 다시 잡는데, 기다리는 화면
스레드는 그 사이에 깨어나지 못해 GIL을 얻지 못한다(이벤트 루프가 깨어날 때마다, Qt 호출마다 수 ms씩 굶음). 자면 GIL을
확실히 넘긴다. 750세대에 약 0.4초가 더 걸린다."""
GUI_QUIET_S: Final = 0.03
"""화면 스레드가 무거운 일(펌프 한 단계, 기록 묶음 넣기, 실행 열기)을 마친 뒤에도 배경 읽기가 더 쉬는 시간(초).
pyqtgraph는 setData 뒤 그림을 다음 이벤트 루프 차례에 그리므로 그 그림까지 GIL을 다투지 않게 한다."""
GUI_WAIT_MAX_S: Final = 0.25
"""배경 읽기가 화면을 기다리는 최대 시간(초). 화면이 계속 바빠도 이만큼마다 파일 하나는 읽는다(굶지 않게)."""
UNIT_WAIT_MAX_S: Final = 0.03
"""화면 스레드가 무거운 일(busy)을 시작할 때 배경 스레드가 하던 쉴 수 없는 일 하나(GuiPriority.working: Episode 하나 만들기,
전체 GC)가 끝나기를 기다리는 최대 시간(초, v6.0). 그 일은 혼자 돌면 수 ms(바퀴 세계 하나 6ms 안팎)다."""
PCA_SMALL_DIM: Final = 256
"""이 차원 이하는 누적 공분산(ΣX, ΣXᵀX) + eigh, 넘으면 표본의 무작위 PCA(D18)."""
PCA_SAMPLE_ROWS: Final = 4096
PCA_BATCH: Final = 64
"""유전자 지도 누적 합에 한 번에 쌓아 더하는 세대 수(배경 읽기 묶음과 같은 크기)."""
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


def decode_observation(arrays: Mapping[str, Any], generation: int,
                       layout: GenomeLayout) -> tuple[Observation | None, str]:
    """관찰 키 → (Observation 또는 None, 이유). runs.observe.decode를 그 실행의 유전자 구조로 부른다.

    구동기 출력이 없는 환경(뇌 없는 글라이더, v2.0 설계 4.8)은 decode가 n_out ≥ 1만 받으므로 행동 묶음(출력 0개라
    빈 배열)을 빼고 읽는다: 행동 분포는 어차피 비어 있고(DASH_EMPTY_NO_OUTPUTS) 나머지 기록은 그대로 검사한다."""
    n_out = layout.brain.n_out
    if n_out == 0:
        arrays = {k: v for k, v in arrays.items() if k not in (ob.ACT_HIST, ob.ACT_MEAN)}
        n_out = 1  # 행동 묶음이 없으면 decode는 n_out을 쓰지 않는다
    return ob.decode(arrays, generation=generation, dim=layout.dim, n_body=layout.n_body, n_out=n_out)


def live_entry(generation: int, observation: Mapping[str, Any] | None, population: np.ndarray | None,
               representative: int, layout: GenomeLayout, *, learner: Any = None,
               rep_score: Any = math.nan) -> tuple[GenEntry | None, np.ndarray | None]:
    """학습 중 세대 메시지 → (GenEntry 또는 None, 정확한 float64 집단 또는 None). 관찰 기록은 디스크와 같은 검사(decode)를
    거친다. 관찰이 없으면(관찰을 끈 학습, 요약 실패) 집단의 세대 대표로 대신한다."""
    obs = None
    if observation is not None:
        obs, _reason = decode_observation(observation, int(generation), layout)
    exact = None
    rep64 = None
    rep = obs.representative if obs is not None else int(representative)
    if population is not None:
        try:
            pop = np.asarray(population)
            # 화면 메모리 상한(EXACT_BUDGET)을 넘는 집단은 통째로 복사하지 않는다: 세대 대표 줄만
            keep = pop.ndim == 2 and pop.shape[0] * pop.shape[1] * 8 <= EXACT_BUDGET
            pop = np.array(pop if keep else pop[rep:rep + 1], dtype=np.float64)
        except (TypeError, ValueError, IndexError):
            pop, keep = None, False
        if pop is not None and pop.ndim == 2 and pop.shape[1] == layout.dim and pop.shape[0] > 0 and \
                np.isfinite(pop).all():
            pop.flags.writeable = False
            if keep:
                exact = pop
                if 0 <= rep < pop.shape[0]:
                    rep64 = pop[rep]
            elif 0 <= rep:
                rep64 = pop[0]
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
        self.add_many([genomes])

    def add_many(self, blocks: Sequence[np.ndarray]) -> None:
        """여러 세대의 유전자를 한 번에 더한다(세대마다 더한 합과는 부동소수 반올림만 다름). 배경 읽기의 묶음(64세대)을
        numpy 몇 번으로 끝내, 화면 스레드가 세대마다 numpy를 부르며 읽기 스레드와 GIL을 수백 번 주고받지 않게 한다(T39)."""
        for i in range(0, len(blocks), PCA_BATCH):  # 한 번에 쌓는 크기를 묶어 둔다(750세대를 한꺼번에 받아도)
            parts = [np.asarray(b).reshape(-1, self.dim) for b in blocks[i:i + PCA_BATCH]]
            x = (parts[0] if len(parts) == 1 else np.concatenate(parts)).astype(np.float64, copy=False)
            if self.s1 is not None and self.s2 is not None:
                self.s1 += x.sum(axis=0)
                self.s2 += x.T @ x
            self.rows += int(x.shape[0])
            self.generations += len(parts)

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
# 자라는 뇌(v6.1)의 통계 열
GROW_COLUMNS: Final = ("brain_neurons", "brain_links", "grow")
"""자라는 뇌 실행(학습 방법 "grow")에만 있는 통계 열(engine/trainer.GROW_COLUMNS와 같은 이름, v6 설계 D35·7.3). 뉴런·연결은 그
세대 후보를 만든 크기(은닉층 합, 편향 뺀 가중치 수), grow는 그 세대 끝에 자랐으면 1. 크기 고정·옛 실행에는 없다."""


@dataclass(frozen=True, eq=False)
class GrowSeries:
    """「뇌 크기」 타일(app/lab/grow_tile.py)이 그리는 자라는 뇌 계열. 줄 번호는 통계 표(대시보드 StatsTable)의 줄."""

    rows: np.ndarray
    """(n,) intp: 뉴런 수가 있는 통계 줄 번호(오름차순)."""
    neurons: np.ndarray
    """(n,) float64 은닉 뉴런 수(층 합)."""
    links: np.ndarray
    """(n,) float64 연결 수(값이 없으면 NaN)."""
    events: np.ndarray
    """자란(grow = 1) 통계 줄 번호(오름차순 intp)."""


def grow_columns(rows: Iterable[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """통계 줄(학습 중 숫자 사전이나 RunStore.read_stats()의 글자 사전)에서 GROW_COLUMNS를 float64 배열로. 그 열이 없는 줄
    (크기 고정·옛 실행)과 읽을 수 없는 값은 NaN. 어느 줄에도 없는 열은 빼고 돌려준다(빈 사전 = 자라는 뇌 실행이 아님)."""
    rows = list(rows)
    out: dict[str, np.ndarray] = {}
    for name in GROW_COLUMNS:
        if not any(name in r for r in rows):
            continue
        out[name] = np.array([_finite_float(r.get(name)) for r in rows], dtype=np.float64)
    return out


def grow_series(columns: Mapping[str, np.ndarray]) -> GrowSeries | None:
    """열 이름 → 줄마다 값(같은 길이, 없는 값 NaN)에서 GrowSeries. brain_neurons 열이 없거나 값이 하나도 없으면 None
    (크기 고정·옛 실행: 「뇌 크기」 타일은 GROW_TILE_FIXED)."""
    neurons = columns.get("brain_neurons")
    if neurons is None:
        return None
    n = np.asarray(neurons, dtype=np.float64)
    ok = np.isfinite(n)
    if not ok.any():
        return None
    rows = np.flatnonzero(ok).astype(np.intp)
    links = columns.get("brain_links")
    lk = np.asarray(links, dtype=np.float64)[rows] if links is not None else np.full(len(rows), math.nan)
    grow = columns.get("grow")
    events = np.zeros(0, dtype=np.intp) if grow is None else np.flatnonzero(np.asarray(grow) == 1.0).astype(np.intp)
    return GrowSeries(rows=rows, neurons=n[rows], links=lk, events=events)


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
        blocks: list[np.ndarray] = []
        self._put(entry, exact, blocks)
        if blocks and self.pca is not None:
            self.pca.add_many(blocks)
        self.version += 1

    def add_many(self, entries: Iterable[GenEntry]) -> None:
        """여러 세대를 넣는다(배경 읽기의 묶음). 유전자 지도 누적 합은 새 세대들을 모아 한 번에 더한다."""
        blocks: list[np.ndarray] = []
        for e in entries:
            self._put(e, None, blocks)
        if blocks and self.pca is not None:
            self.pca.add_many(blocks)
        self.version += 1

    def _put(self, entry: GenEntry, exact: np.ndarray | None, blocks: list[np.ndarray]) -> None:
        """add·add_many의 몸통: 기록에 넣고, 처음 들어온 세대의 지도 유전자를 blocks에 모은다."""
        g = int(entry.generation)
        old = self._entries.get(g)
        self._entries[g] = entry
        if old is None:
            bisect.insort(self.gens, g)
            genomes = entry.map_genomes
            if self.pca is not None and genomes is not None and genomes.shape[1] == self.pca.dim:
                blocks.append(genomes)
        else:
            self._pca_stale = True
        if exact is not None:
            self._exact.pop(g, None)
            if exact.nbytes <= EXACT_BUDGET:
                self._exact[g] = exact
            total = sum(a.nbytes for a in self._exact.values())
            while self._exact and (len(self._exact) > KEEP_EXACT or total > EXACT_BUDGET):
                total -= self._exact.popitem(last=False)[1].nbytes

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

    def _cached(self, key: Any, make: Any, tag: Any = None) -> Any:
        """기록 판(version)과 tag가 같으면 저장해 둔 값. 열쇠마다 값 하나만 둔다(세대처럼 많아지는 것은 tag로)."""
        hit = self._cache.get(key)
        if hit is not None and hit[0] == self.version and hit[1] == tag:
            return hit[2]
        value = make()
        self._cache[key] = (self.version, tag, value)
        return value

    def exact_bytes(self) -> int:
        """정확한 float64 집단이 쓰는 화면 메모리(바이트)."""
        return sum(a.nbytes for a in self._exact.values())

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
            blocks = [b for b in (self._entries[g].map_genomes for g in self.gens)
                      if b is not None and b.shape[1] == pca.dim]
            pca.add_many(blocks)
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
            # 같은 길이 1차원 배열 목록 → (n, NB): np.array가 vstack(원소마다 atleast_2d)보다 3배쯤 빠르고 값은 같다
            # (750세대 네 배열에 3.6ms → 1ms 남짓, 기록을 읽는 동안 묶음마다 다시 만든다)
            return BodySeries(
                gens=np.array([e.generation for e in es], dtype=np.float64),
                mean=_rows([e.body_mean for e in es]), lo=_rows([e.body_min for e in es]),
                hi=_rows([e.body_max for e in es]), rep=_rows([e.body_rep for e in es]),
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

        # 보는 세대는 tag로: 세대마다 열쇠를 따로 두면 학습·넘겨 보기 동안 밀도 격자가 세대 수만큼 쌓인다
        return self._cached(("dens", window, recent), make, tag=gen)


def _rows(arrays: Sequence[np.ndarray | None]) -> np.ndarray:
    """세대마다의 (NB,) float64 배열 목록 → (n, NB) float64(np.vstack과 같은 값)."""
    out = np.array(arrays, dtype=np.float64)  # 길이가 다르면 vstack처럼 ValueError
    if out.ndim != 2:  # 2차원이 아니면 vstack에 맡긴다(같은 결과·오류)
        return np.vstack(arrays)  # type: ignore[arg-type]
    return out


# ---------------------------------------------------------------------------
# 배경 읽기
class GuiPriority:
    """화면 스레드가 무거운 일을 하는 동안 배경 읽기를 쉬게 한다(GIL 다툼 줄이기).

    배경 읽기(파이썬 코드가 많은 npz 해석)와 화면 스레드는 GIL 하나를 나눠 쓴다. 화면 스레드는 Qt를 부를 때와 이벤트를
    기다릴 때 GIL을 놓는데, 그때마다 배경 읽기가 GIL을 잡으면 화면은 수 ms씩 기다린다: 혼자면 몇 ms인 pyqtgraph 그림이
    50~100ms가 된다. 화면 스레드는 무거운 일을 busy()로 감싸고, 배경 읽기는 파일마다 wait()로 그동안(과 끝난 뒤
    GUI_QUIET_S) 잔다. 자는 동안은 GIL을 놓는다. 화면 스레드에서만 busy()를 부르고, 배경 스레드는 값만 읽는다.

    quiet는 늘 시간으로만 끝난다. 이벤트 루프가 잠들려 할 때(aboutToBlock) 끝내지 않는 까닭: 그 신호가 리눅스(glib)에서는
    이벤트 루프가 한 바퀴 돌 때마다, 윈도우에서는 정말 한가할 때만 와서, 운영체제마다 다르게 쉬게 된다. 시간으로만 끝내면
    어디서나 같고, 테스트(T39)가 재는 것이 그대로 앱의 동작이다.

    hold(v6.0, v3.0 남은 문제 「2세대가 올 때 GIL 다툼」): 곧 무거운 일이 올 것을 아는 화면 스레드(대시보드 펌프: 새 세대가 와서
    보이는 타일이 dirty)가 그 일이 끝날 때까지(최대 seconds초) 배경 작업이 새 일을 시작하지 않게 잡아 둔다. 배경 작업은 일 하나
    (Episode 하나 만들기 등)를 시작한 뒤에는 그 안에서 쉬지 못해, 펌프가 그 사이에 시작되면 끝날 때까지 GIL을 다툰다(바퀴 세계
    하나 만들기가 언덕 모양을 수천 번 더하는 동안 첫 타일 그림 40ms가 100ms 넘게 늘었다). 주인마다 따로 잡고 release로 푼다.
    시간이 지나면 저절로 풀린다(펌프가 돌지 못해도 배경 작업이 굶지 않게).

    working(v6.0): 배경 스레드가 그 안에서 쉴 수 없는 일 하나(Episode 하나 만들기, 전체 GC)를 감싼다. 이런 일은 프로세스에서
    보통 한 번에 하나만 돈다(배경 스레드끼리도 GIL을 다투지 않게: 관전 장면·세계 그림·다시 시뮬레이션이 함께 돌면 셋 다 느려지고
    화면 몫도 줄었다). 시작할 때 화면이 바쁘면 먼저 기다린다(wait). 화면 스레드는 busy()에 들어갈 때(겹친 안쪽이 아니면) 그런
    일이 돌고 있으면 끝나기를 잠깐(최대 UNIT_WAIT_MAX_S) 자며 기다린다: 그동안 배경 스레드가 GIL을 혼자 써서 금방 끝나고, 다음
    일은 busy를 보고 시작하지 않으므로 화면의 무거운 일은 GIL을 다투지 않는다. 함께 돌면 서로 GIL을 넘길 때마다 간격(1ms)씩
    기다려 둘 다 몇 배 느려졌다(첫 타일 그림 40ms가 100ms 넘게). 계산 결과는 같다(쉬는 때만 바뀐다).
    """

    def __init__(self) -> None:
        self._busy = 0
        self._quiet_until = 0.0
        self._holds: dict[int, float] = {}
        """hold 주인(id) → 풀리는 시각(time.monotonic)."""
        self._units = 0
        """돌고 있는 배경 working() 수(보통 0 또는 1)."""
        self._count_lock = threading.Lock()
        self._unit_lock = threading.RLock()

    @contextmanager
    def busy(self, quiet: float = GUI_QUIET_S) -> Iterator[None]:
        """화면 스레드: 이 안에서는 배경 읽기가 쉰다(끝난 뒤 quiet초 더). 겹쳐 불러도 된다. 바깥 busy에 들어갈 때 배경의
        쉴 수 없는 일(working)이 돌고 있으면 끝나기를 최대 UNIT_WAIT_MAX_S초 기다린다."""
        self._busy += 1
        try:
            if self._busy == 1 and self._units:
                end = time.monotonic() + UNIT_WAIT_MAX_S
                while self._units and time.monotonic() < end:
                    time.sleep(0.0005)
            yield
        finally:
            self._busy -= 1
            self._quiet_until = max(self._quiet_until, time.monotonic() + quiet)

    @contextmanager
    def working(self, stop: Callable[[], bool] | None = None) -> Iterator[None]:
        """배경 스레드: 그 안에서 쉴 수 없는 일 하나(프로세스에서 한 번에 하나: 앞 일이 GUI_WAIT_MAX_S 넘게 끝나지 않으면 그냥
        함께 돈다). 화면이 바쁘면 시작하기 전에 기다리고(wait), 화면 스레드가 busy에 들어가면 이것이 끝나기를 잠깐 기다린다.
        같은 스레드에서 겹쳐 써도 된다."""
        got = self._unit_lock.acquire(timeout=GUI_WAIT_MAX_S)
        try:
            self.wait(stop)
            with self._count_lock:
                self._units += 1
            try:
                yield
            finally:
                with self._count_lock:
                    self._units -= 1
        finally:
            if got:
                self._unit_lock.release()

    def hold(self, owner: object, seconds: float) -> None:
        """화면 스레드: owner의 무거운 일이 곧 온다. 지금부터 seconds초 동안(또는 release까지) 배경 작업을 쉬게 한다."""
        self._holds[id(owner)] = time.monotonic() + seconds

    def release(self, owner: object) -> None:
        """화면 스레드: owner가 잡아 둔 것을 푼다(다른 주인의 hold와 busy·quiet는 그대로)."""
        self._holds.pop(id(owner), None)

    @property
    def is_busy(self) -> bool:
        if self._busy > 0:
            return True
        now = time.monotonic()
        if now < self._quiet_until:
            return True
        holds = self._holds
        if holds:
            try:
                return any(now < until for until in list(holds.values()))
            except RuntimeError:  # 배경 스레드가 읽는 사이 화면 스레드가 바꿨다: 바쁜 쪽으로
                return True
        return False

    def wait(self, stop: Callable[[], bool] | None = None, limit: float = GUI_WAIT_MAX_S) -> None:
        """배경 스레드: 화면이 바쁘면 한가해질 때까지(최대 limit초, stop()이 참이면 바로) 잔다."""
        end = time.monotonic() + limit
        while self.is_busy and (stop is None or not stop()):
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(0.002, left))


GUI_PRIORITY = GuiPriority()
"""화면 스레드 하나에 하나(실험실 대시보드가 busy로 감싸고 HistoryLoader가 기다린다)."""

GUI_SWITCH_INTERVAL_S: Final = 0.001
"""화면 프로세스의 GIL 넘김 간격(sys.setswitchinterval, 파이썬 기본 0.005초). 배경 스레드가 한 번에 오래 도는 일(실험실
다시 시뮬레이션 하나는 바퀴 차 30초에 140ms 안팎, 도중에 쉴 곳이 없다)을 하는 동안 화면 스레드는 GIL을 놓을 때마다
(Qt 호출) 다시 얻으려고 이 간격까지 기다린다. 5ms면 그림 한 번이 세 배 넘게 늘었다(학습 중 「몸 수치 추이」 갱신+그림
최대 80ms → 1ms에서 39ms, v3.0 측정). 학습은 다른 프로세스라 영향이 없다."""


def favor_gui_thread() -> None:
    """화면 프로세스에서 한 번: GIL 넘김 간격을 GUI_SWITCH_INTERVAL_S로 줄인다(이미 더 짧으면 그대로). 계산 결과는 같고
    스레드가 번갈아 도는 간격만 바뀐다."""
    if sys.getswitchinterval() > GUI_SWITCH_INTERVAL_S:
        sys.setswitchinterval(GUI_SWITCH_INTERVAL_S)


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
    obs, reason = decode_observation(keys, generation, layout)
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
        stop = self.isInterruptionRequested
        for g in sorted(_listing(folder, "champion")):
            if g > last or self.isInterruptionRequested():
                continue
            self._yield(stop)
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
                self._yield(stop)
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

    @staticmethod
    def _yield(stop: Callable[[], bool]) -> None:
        """파일 하나 앞에서: 잠깐 자서 GIL을 화면 스레드에 넘기고, 화면이 바쁘면(GUI_PRIORITY) 더 기다린다."""
        time.sleep(LOAD_YIELD_S)
        GUI_PRIORITY.wait(stop)


_LIVE_LOADERS: set[HistoryLoader] = set()
"""도는 중인 읽기 스레드를 끝날 때까지 붙잡아 둔다(대시보드가 먼저 사라져도 QThread가 돌던 중에 지워지지 않게)."""


def stop_loaders() -> None:
    """프로그램을 끝낼 때: 도는 배경 읽기를 모두 멈추라고 하고 끝날 때까지 기다린다. 읽기 스레드는 주인이 없는 파이썬
    객체라 프로그램이 끝나며 지워지는데, 그때 아직 돌고 있으면 Qt가 프로세스를 끝내 버린다(QThread 오류로 abort).
    멈추라고 하면 세대 대표 파일 하나(약 0.6ms)를 읽는 사이에 끝난다."""
    loaders = list(_LIVE_LOADERS)
    for loader in loaders:
        loader.requestInterruption()
    for loader in loaders:
        loader.wait()
        _LIVE_LOADERS.discard(loader)

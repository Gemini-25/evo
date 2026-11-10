"""자라는 뇌 학습 방법: GrowingCMAES (v6.1, v6 설계 4.15·2.3·3.3, D26~D30·D33·D34).

틀 안에서 자라는 뇌(policy/frame.py): 유전자는 늘 틀 길이(layout.dim)이고, 이 학습 방법은 켜진 좌표만 안쪽 CMA-ES(learners/cmaes.CMAES
그대로)로 탐색한다. 켜진 좌표 밖은 정확히 +0.0이다(후보·추천해 모두 np.zeros 위에 켜진 좌표만 채운다). 그래서 워커·재생·판정·
기록은 지금 코드 그대로 틀 전체를 계산하고, 자람은 이 학습 방법 안의 일이다.

- 만들 때: 은닉층마다 켜진 뉴런 수 counts = max(x0의 counts_of, min(start, 틀 크기)). 처음 상태(뇌 0)면 min(start, 틀), 챔피언에서 다시
  시작하면 챔피언의 켜진 수부터(D33). 평균 = x0의 켜진 좌표(몸 유전자는 [-1, 1]로 자름). 켜진 뉴런 가운데 들어오는·나가는 가중치가
  모두 0인 뉴런(처음 상태면 전부, 챔피언에서면 start가 챔피언보다 클 때 더해진 뉴런)은 들어오는 가중치를 init_in × 표준 정규로
  채운다(층 순서, 뉴런 번호 순서. 나가는 가중치가 0이라 처음 함수는 그대로). 안쪽 CMA-ES는 학습 방법 시드로 만든다.
- ask(): 안쪽 ask()(켜진 좌표 차원)를 np.zeros((P, D))의 켜진 좌표에 넣어 돌려준다(사본, tell 전에 다시 부르면 같은 후보).
- tell(): 마지막 ask()와 같은 후보인지 보고(check_tell) 안쪽 tell(켜진 좌표, 점수) → 평균 = np.zeros 위에 안쪽 best() →
  since += 1 → since ≥ every이고 어느 층이든 틀보다 작으면 자람(D29: 정해진 간격, 막힘 기준 없음).
- 자람: (1) 새 counts = 층마다 min(틀, 2 × 지금). (2) 새 뉴런마다(층 순서, 뉴런 번호 순서) 들어오는 가중치(새 counts 기준: 앞층의 켜진
  뉴런 전부, 첫 층은 입력 전부) = init_in × 표준 정규(이 학습 방법의 PCG64), 편향 0. 새 뉴런에서 옛 뉴런·출력으로 나가는 가중치는 0
  그대로(켜지지 않았던 좌표라 이미 0). 그래서 자라기 직전·직후 평균의 출력은 같은 실수 연산이다(더해지는 항이 정확히 0, 함수 보존 D28).
  (3) 좌표별 처음 폭: 옛 켜진 좌표 = 안쪽 pycma es.stds, 새 좌표 = 옛 폭의 중앙값. (4) 안쪽 CMA-ES를 새로 만든다: 출발점 = (2)의
  평균(켜진 좌표만), CMAESConfig(..., stds=(3), seed=자람 시드). 개체 수 그대로. (5) since = 0, growths += 1, events에 한 줄.
- best(): 평균의 사본. 자란 직후에는 (2)에서 만든 평균 그대로다(안쪽 pycma가 경계 변환으로 몸 유전자를 아주 조금 옮길 수 있어
  안쪽 best()를 쓰지 않는다: 함수 보존이 바이트 단위로 성립하게). 다음 tell부터 안쪽 best()를 틀에 넣은 것.
- stats(): 안쪽 CMAES.stats()의 키 전부. 단 "sigma" = 안쪽 es.stds의 중앙값(D34: 자라서 pycma sigma가 1.0으로 다시 시작해도 그래프가
  튀지 않음), "generation"·"evaluations" = 이 학습 방법 전체(자람을 넘어 누적), "mean_step" = 마지막 tell에서 평균이 움직인 거리(자람
  전), "diversity" = 마지막 ask 후보의 켜진 좌표별 표준편차 평균. 더해서 "neurons"·"links"(이번 세대 후보를 만든 크기 = 자라기 전),
  "growths", "grown"(이 tell에서 자랐으면 1.0).
- 난수: 안쪽 pycma RandomState(지금 CMAES처럼 안쪽 상태에 피클), 새 뉴런 가중치용 PCG64(상태 "rng"). 시드: 안쪽 처음 = 학습 방법
  시드(cma.seed), k번째 자람(1부터) = seeds.derive(학습 방법 시드, GROW_SEED_WORD, k) & 0xFFFFFFFF, PCG64 =
  seeds.derive(학습 방법 시드, GROW_RNG_WORD). 낱말을 나눠 섞는다(명세 A27과 같은 이유). numpy 전역 난수 없음.
- 상태(형식 GROW_STATE_FORMAT, 설계 2.3 + 아래 덧붙인 키): 피클할 수 있고 다른 프로세스에서 그대로 이어진다. ask()와 tell() 사이에
  저장하면 그 ask() 직전 상태(안쪽 CMAES 규칙, 이 학습 방법의 다른 값은 ask에서 바뀌지 않음)이고, 불러온 뒤 ask()는 같은 후보를
  낸다. 체크포인트는 세대 끝에 저장되고 자람은 tell 안에서 일어나므로 「자라야 하는데 아직 안 자란」 상태는 저장되지 않는다(3.3).
  상태를 읽다 실패하면 ValueError이고 이 학습 방법은 그대로다.
- IPOP과 함께 쓰지 않는다(D33: 개체 수가 바뀌면 「10세대」의 시험 수 뜻이 바뀜). cma.ipop이 참이면 ValueError(TRAIN_GROW_NO_IPOP).
  diagonal_threshold는 안쪽 CMAES가 자기 차원(켜진 좌표 수)으로 판단한다(지금 규칙 그대로).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Final

import numpy as np

from evolab.app import strings as S
from evolab.engine import seeds
from evolab.learners.base import (
    check_tell,
    generator_from_state,
    generator_state,
    initial_genome,
    is_int,
    is_real,
    new_generator,
    state_array,
)
from evolab.learners.cmaes import CMAES, CMAESConfig, config_dict
from evolab.policy.frame import BrainFrame
from evolab.policy.genome import GenomeLayout

GROW_STATE_FORMAT: Final = 1
GROW_SEED_WORD: Final = 0x67726F77
"""「grow」: k번째 자람의 안쪽 CMA-ES 시드 낱말."""
GROW_RNG_WORD: Final = 0x726E67
"""「rng」: 새 뉴런 들어오는 가중치용 PCG64 시드 낱말."""
_SEED_MASK: Final = 0xFFFFFFFF


@dataclass(frozen=True)
class GrowConfig:
    start: int = 2
    """은닉층마다 처음 뉴런 수(틀보다 크면 틀)."""
    every: int = 10
    """이 학습 방법 세대 수(tell 횟수)마다 은닉층마다 두 배(틀에서 멈춤)."""
    init_in: float = 0.5
    """새 뉴런 들어오는 가중치 = init_in × 표준 정규(학습 방법 난수)."""

    def __post_init__(self) -> None:
        if not is_int(self.start) or self.start < 1:
            raise ValueError(S.LEARNER_ERR_GROW_START.format(value=self.start))
        if not is_int(self.every) or self.every < 1:
            raise ValueError(S.LEARNER_ERR_GROW_EVERY.format(value=self.every))
        if not is_real(self.init_in) or not math.isfinite(self.init_in) or self.init_in <= 0:
            raise ValueError(S.LEARNER_ERR_GROW_INIT.format(value=self.init_in))


def growth_schedule(sizes: Sequence[int], start: int, every: int) -> list[tuple[int, tuple[int, ...]]]:
    """처음 상태에서 시작한 학습 방법이 자라는 일정: [(학습 방법 세대, 그 세대 끝 뒤 은닉층마다 켜진 수), ...]. 첫 줄은
    (0, 처음 크기)이고 다 자란 크기(sizes)에서 끝난다(처음부터 다 자랐으면 한 줄). GrowingCMAES의 규칙(층마다 min(틀, 2 × 지금),
    every 세대마다) 그대로라 화면 안내 줄(실험실 LAB_GROW_HINT)이 같은 값을 쓴다. 자동 재시작은 셈하지 않는다."""
    if not is_int(start) or start < 1:
        raise ValueError(S.LEARNER_ERR_GROW_START.format(value=start))
    if not is_int(every) or every < 1:
        raise ValueError(S.LEARNER_ERR_GROW_EVERY.format(value=every))
    full = tuple(int(s) for s in sizes)
    counts = tuple(min(start, s) for s in full)
    out = [(0, counts)]
    while counts != full:
        counts = tuple(min(s, 2 * c) for c, s in zip(counts, full, strict=True))
        out.append((out[-1][0] + every, counts))
    return out


def grow_seed(learner_seed: int, k: int) -> int:
    """k번째 자람(1부터)의 안쪽 CMA-ES 시드(32비트)."""
    return seeds.derive(learner_seed, GROW_SEED_WORD, k) & _SEED_MASK


def grow_rng_seed(learner_seed: int) -> int:
    """새 뉴런 들어오는 가중치용 PCG64 시드(64비트)."""
    return seeds.derive(learner_seed, GROW_RNG_WORD)


class GrowingCMAES:
    """Learner 구현(learners/base.Learner 규격). 점수는 높을수록 좋다(안쪽 CMAES가 pycma에 부호를 뒤집어 넘긴다)."""

    name = "grow"

    def __init__(self, x0: np.ndarray, layout: GenomeLayout, cma: CMAESConfig, grow: GrowConfig) -> None:
        if cma.ipop:
            raise ValueError(S.TRAIN_GROW_NO_IPOP)
        frame = BrainFrame(layout)  # 은닉층이 없거나 뇌가 없으면 FrameError(한국어)
        if cma.bounded_dims > layout.n_body:  # 경계는 몸 유전자(늘 켜진 앞쪽 좌표)에만
            raise ValueError(S.LEARNER_ERR_BOUNDED_DIMS_TOO_MANY.format(value=cma.bounded_dims, dim=layout.n_body))
        x = initial_genome(x0, cma.bounded_dims)
        have = frame.counts_of(x)  # 유전자 모양이 틀과 다르면 FrameError
        self._frame = frame
        self._dim = int(layout.dim)
        self._cma = replace(cma, ipop=False, stds=())  # 바깥 설정(좌표별 폭은 자랄 때 이 학습 방법이 정한다)
        self._grow = grow
        self._rng = new_generator(grow_rng_seed(cma.seed))
        counts = tuple(max(h, min(grow.start, s)) for h, s in zip(have, frame.sizes, strict=True))
        active = frame.active_coords(counts)
        mean = np.zeros(self._dim, dtype=np.float64)
        mean[active] = x[active]  # counts ≥ counts_of(x)라 x의 0이 아닌 좌표는 모두 켜진 좌표 안
        for layer, n in enumerate(counts):
            for j in range(n):
                idx = frame.neuron_in(layer, j, counts)
                # 들어오는 가중치도 나가는 가중치도 모두 0인 켜진 뉴런만(처음 상태면 전부): 나가는 쪽이 0이라 채워도 함수는 그대로
                if (idx.size and not np.any(mean[idx] != 0.0)
                        and not np.any(mean[frame.neuron_out(layer, j, counts)] != 0.0)):
                    mean[idx] = grow.init_in * self._rng.standard_normal(idx.size)
        self._counts = counts
        self._active = active
        self._mean = mean
        self._inner = CMAES(mean[active], self._cma)
        self._growths = 0
        self._since = 0
        self._events: list[list[int]] = []
        self._generation = 0
        self._evaluations = 0
        self._told = counts
        """마지막 tell 세대의 후보를 만든 크기(stats의 neurons·links). tell 전에는 지금 크기."""
        self._last_growth: tuple[int, int, int, int] | None = None
        self._mean_step = math.nan
        self._diversity = math.nan
        self._pending: np.ndarray | None = None

    # ------------------------------------------------------------------ 읽기 전용
    @property
    def popsize(self) -> int:
        return self._inner.popsize

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def config(self) -> CMAESConfig:
        return self._cma

    @property
    def grow_config(self) -> GrowConfig:
        return self._grow

    @property
    def frame(self) -> BrainFrame:
        return self._frame

    @property
    def counts(self) -> tuple[int, ...]:
        """은닉층마다 지금 켜진 뉴런 수(다음 ask()의 크기)."""
        return self._counts

    @property
    def growths(self) -> int:
        return self._growths

    @property
    def since(self) -> int:
        """마지막 자람(또는 시작) 뒤 tell 횟수."""
        return self._since

    @property
    def generation(self) -> int:
        """지금까지 tell()한 세대 수(자람을 넘어 누적)."""
        return self._generation

    @property
    def events(self) -> list[list[int]]:
        """자람 기록 [학습 방법 세대, 뉴런 합, 연결 수]의 사본(자란 뒤 크기)."""
        return [list(e) for e in self._events]

    @property
    def last_growth(self) -> tuple[int, int, int, int] | None:
        """마지막 tell()에서 자랐으면 (옛 뉴런 합, 새 뉴런 합, 옛 연결, 새 연결), 아니면 None."""
        return self._last_growth

    @property
    def inner(self) -> CMAES:
        """켜진 좌표 위의 안쪽 CMA-ES(읽기용: diagonal, dim, coordinate_stds)."""
        return self._inner

    def active_coords(self) -> np.ndarray:
        """지금 켜진 좌표(유전자 번호, 오름차순)의 사본."""
        return self._active.copy()

    # ------------------------------------------------------------------ ask/tell
    def ask(self) -> np.ndarray:
        """(popsize, 틀 길이) 후보. 켜진 좌표 밖은 정확히 +0.0. tell() 전에 다시 부르면 같은 후보(난수를 더 쓰지 않음)."""
        if self._pending is None:
            inner = self._inner.ask()
            X = np.zeros((inner.shape[0], self._dim), dtype=np.float64)
            X[:, self._active] = inner
            self._pending = X
        return self._pending.copy()

    def tell(self, candidates: np.ndarray, scores: np.ndarray) -> None:
        check_tell(self._pending, candidates)
        assert self._pending is not None
        pending = self._pending
        self._inner.tell(pending[:, self._active], scores)  # 점수 모양·NaN은 안쪽(sanitize_scores)이 본다
        previous = self._mean
        mean = np.zeros(self._dim, dtype=np.float64)
        mean[self._active] = self._inner.best()
        self._pending = None
        self._mean = mean
        self._mean_step = float(np.linalg.norm(mean - previous))
        self._diversity = self._inner.diversity
        self._generation += 1
        self._evaluations += pending.shape[0]
        self._told = self._counts
        self._last_growth = None
        self._since += 1
        if self._since >= self._grow.every and any(c < s for c, s in zip(self._counts, self._frame.sizes, strict=True)):
            self._grow_now()

    def best(self) -> np.ndarray:
        """지금 평균(경계 안으로 옮긴 안쪽 best()를 틀에 넣은 것, 자란 직후에는 자람이 만든 평균)의 사본."""
        return self._mean.copy()

    def stats(self) -> dict[str, float]:
        out = dict(self._inner.stats())
        told = self._told
        out.update({
            "sigma": float(np.median(self._inner.coordinate_stds())),
            "mean_step": self._mean_step,
            "diversity": self._diversity,
            "generation": float(self._generation),
            "evaluations": float(self._evaluations),
            "neurons": float(sum(told)),
            "links": float(self._frame.links(told)),
            "growths": float(self._growths),
            "grown": 1.0 if self._last_growth is not None else 0.0,
        })
        return out

    # ------------------------------------------------------------------ 자람
    def _grow_now(self) -> None:
        frame = self._frame
        old = self._counts
        new = tuple(min(s, 2 * c) for c, s in zip(old, frame.sizes, strict=True))
        mean = self._mean.copy()
        for layer in range(len(new)):
            for j in range(old[layer], new[layer]):
                idx = frame.neuron_in(layer, j, new)
                mean[idx] = self._grow.init_in * self._rng.standard_normal(idx.size)
        old_active = self._active
        new_active = frame.active_coords(new)
        old_stds = self._inner.coordinate_stds()
        full = np.full(self._dim, float(np.median(old_stds)), dtype=np.float64)
        full[old_active] = old_stds
        stds = tuple(float(v) for v in full[new_active])
        k = self._growths + 1
        inner = CMAES(mean[new_active], replace(self._cma, seed=grow_seed(self._cma.seed, k), stds=stds))
        links_old, links_new = frame.links(old), frame.links(new)
        self._inner = inner
        self._mean = mean
        self._counts = new
        self._active = new_active
        self._growths = k
        self._since = 0
        self._last_growth = (sum(old), sum(new), links_old, links_new)
        self._events.append([self._generation, sum(new), links_new])

    # ------------------------------------------------------------------ 상태
    def state_dict(self) -> dict[str, Any]:
        """설계 2.3의 키 + 덧붙인 키(dim, generation, evaluations, told, last_growth, mean_step, diversity). 사본이다."""
        return {
            "format": GROW_STATE_FORMAT,
            "name": self.name,
            "config": {"cma": config_dict(self._cma), "grow": asdict(self._grow)},
            "dim": self._dim,
            "counts": list(self._counts),
            "growths": self._growths,
            "since": self._since,
            "rng": generator_state(self._rng),
            "mean": self._mean.tolist(),
            "inner": self._inner.state_dict(),
            "events": [list(e) for e in self._events],
            "generation": self._generation,
            "evaluations": self._evaluations,
            "told": list(self._told),
            "last_growth": None if self._last_growth is None else list(self._last_growth),
            "mean_step": self._mean_step,
            "diversity": self._diversity,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """저장된 상태로 바꾼다. 상태 안의 설정(시드 포함)이 생성자 값보다 우선한다. 틀(유전자 배치)은 생성자의 것이고 차원이 다르면
        ValueError. 상태가 잘못되었으면(빠진 항목, 틀과 안 맞는 크기, 꺼진 좌표에 0이 아닌 값, 안쪽 상태 불일치) ValueError이고, 그때
        이 학습 방법은 그대로다."""
        if not isinstance(d, Mapping):
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=type(d).__name__))
        if d.get("name") != self.name:
            raise ValueError(S.LEARNER_ERR_STATE_KIND.format(got=d.get("name"), want=self.name))
        if d.get("format") != GROW_STATE_FORMAT:
            raise ValueError(S.LEARNER_ERR_STATE_FORMAT.format(got=d.get("format")))
        try:
            dim = int(d["dim"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=e)) from e
        if dim != self._dim:
            raise ValueError(S.LEARNER_ERR_STATE_DIM.format(got=dim, want=self._dim))
        frame = self._frame
        try:
            cfg = d["config"]
            cma = CMAESConfig(**cfg["cma"])
            grow = GrowConfig(**cfg["grow"])
            if cma.ipop or cma.stds:
                raise ValueError("cma config")
            counts = _counts(frame, d["counts"])
            told = _counts(frame, d["told"])
            growths, since = _count(d["growths"]), _count(d["since"])
            generation, evaluations = _count(d["generation"]), _count(d["evaluations"])
            events = [[_count(v) for v in e] for e in d["events"]]
            if len(events) != growths or any(len(e) != 3 for e in events):
                raise ValueError("events")
            last = d["last_growth"]
            last_growth = None if last is None else tuple(_count(v) for v in last)
            if last_growth is not None and len(last_growth) != 4:
                raise ValueError("last_growth")
            mean_step, diversity = float(d["mean_step"]), float(d["diversity"])
            mean = state_array(d["mean"], (self._dim,))
            active = frame.active_coords(counts)
            outside = np.ones(self._dim, dtype=bool)
            outside[active] = False
            if np.any(mean[outside] != 0.0):
                raise ValueError("mean outside the active coordinates")
            mean[outside] = 0.0  # -0.0도 +0.0으로(켜진 좌표 밖은 정확히 +0.0)
            rng = generator_from_state(d["rng"])
            inner_state = d["inner"]
        except (KeyError, TypeError, ValueError, IndexError) as e:  # FrameError 포함
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=e)) from e
        inner = CMAES(mean[active], replace(cma, stds=()))  # 차원만 맞으면 된다: 설정·분포는 불러온 상태가 우선
        inner.load_state_dict(inner_state)  # 차원이 켜진 좌표 수와 다르거나 깨졌으면 ValueError(이 학습 방법은 그대로)
        self._cma = cma
        self._grow = grow
        self._rng = rng
        self._counts = counts
        self._active = active
        self._mean = mean
        self._inner = inner
        self._growths = growths
        self._since = since
        self._events = events
        self._generation = generation
        self._evaluations = evaluations
        self._told = told
        self._last_growth = last_growth  # type: ignore[assignment]
        self._mean_step = mean_step
        self._diversity = diversity
        self._pending = None


def _count(v: Any) -> int:
    """상태의 0 이상 정수(bool 제외)."""
    if not is_int(v) or v < 0:
        raise ValueError(f"count: {v!r}")
    return int(v)


def _counts(frame: BrainFrame, value: Any) -> tuple[int, ...]:
    """상태의 켜진 뉴런 수(층마다 1..틀 크기). 틀과 맞지 않으면 FrameError(ValueError)."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"counts: {value!r}")
    frame.active_coords(value)  # 검사
    return tuple(int(c) for c in value)

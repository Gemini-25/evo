# 출처: 커밋 1e59b26(v6.0 출시판)의 src/evolab/learners/cmaes.py를 바이트 그대로 복사. 이 줄만 더함(tests/data/v60_snapshot/SOURCES.txt, v6 설계 7.1 T40·T44). 고치지 말 것.
"""CMA-ES 학습 방법: pycma(cma 4.x) 감싸기 (명세: 학습 방법, 결정성, 검토 반영 결정).

pycma 4.5에서 실험으로 확인한 것:
- 전용 난수: opts['randn']에 RandomState(seed).randn을 넘기고 'seed'=nan이면 pycma가 numpy 전역 난수를
  건드리지 않는다(시드도 넣지 않는다). 표본 생성기(es.sm.randn)도 같은 함수를 쓴다.
- 'signals_filename'은 빈 문자열로 끈다. 기본값이면 stop()이 실행 폴더의 cma_signals.in을 읽어 설정을 바꾼다
  (verbose=-9에서도 읽음). None을 넘기면 경고가 나서 ''를 쓴다.
- 경계: opts['bounds'] = [[-1]*k + [None]*(n-k), [1]*k + [None]*(n-k)] → BoundTransform. ask()가 돌려주는
  후보는 경계 안(경계값 ±1 그대로 나오기도 함). es.mean은 내부 좌표라 경계 밖일 수 있고, 추천해는
  es.gp.pheno(es.mean, into_bounds=repair)로 경계 안으로 옮긴다(es.result.xfavorite와 같은 계산).
  es.result는 stop()을 부르므로 쓰지 않는다.
- 대각 공분산: CMA_diagonal=True → 표본기 GaussStandardConstant + 대각 디코딩(sigma_vec). 시간·메모리 선형.
- stop()은 부작용이 있다(평탄 점수 세대 수를 센다). 그래서 tell() 뒤 정확히 한 번씩 부르고 다른 곳에서는 부르지 않는다.
  pycma의 종료 조건이 차도 ask/tell은 계속 동작한다. ipop이 켜져 있으면 그때 개체 수 2배로 재시작한다.
- tell()은 ask()가 돌려준 배열 객체 그대로 넘긴다. pycma는 후보 바이트의 hash로 내부 좌표를 찾는데,
  bytes hash는 프로세스마다 달라서 ask와 tell 사이의 pycma 객체를 저장하면 다른 프로세스에서 이어갈 수 없다.
  그래서 ask()는 pycma ask 직전에 객체를 피클해 두고(211차원 전체 공분산 약 0.7ms), ask와 tell 사이의
  state_dict()는 그 세대 경계 상태를 돌려준다. 불러온 뒤 ask()는 같은 후보를 다시 낸다. (워커 오류로 세대 도중에
  멈춰도 Trainer가 체크포인트를 남길 수 있다. Trainer의 세대 수도 그 세대를 끝내지 않은 상태다.)
- 상태: es 객체를 통째로 피클한다(전용 RandomState가 randn을 통해 함께 담긴다). 211차원 전체 공분산 약 0.9MB.
"""

from __future__ import annotations

import math
import pickle
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np

from evolab.app import strings as S
from evolab.learners.base import sanitize_scores
from evolab.learners.cma_import import import_cma

STATE_FORMAT: Final = 1
# IPOP 재시작에서 개체 수를 두 배로 늘리는 최대 횟수. 그 뒤로는 같은 개체 수로 재시작만 한다
# (평탄한 점수가 이어질 때 개체 수가 끝없이 커지는 것을 막는다). 기본 32 → 최대 1024.
IPOP_MAX_DOUBLINGS: Final = 5
_PICKLE_PROTOCOL: Final = 5
_SEED_LIMIT: Final = 2**32  # RandomState 시드 범위
_EIG_EPS: Final = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class CMAESConfig:
    popsize: int = 32
    sigma0: float = 0.5
    diagonal_threshold: int = 1000
    """차원이 이 값보다 크면 대각 공분산(sep-CMA, pycma CMA_diagonal=True)."""
    ipop: bool = False
    """pycma 종료 조건이 차면 지금 추천해에서 개체 수 2배로 재시작."""
    seed: int = 1
    """학습 방법 전용 RandomState 시드(0 ≤ seed < 2**32). Trainer가 실행 시드에서 만든다."""
    bounded_dims: int = 0
    """앞쪽 k차원(몸 유전자)을 [-1, 1] 경계 안에 둔다. 나머지(뇌 가중치)는 경계 없음."""

    def __post_init__(self) -> None:
        if not _is_int(self.popsize) or self.popsize < 2:
            raise ValueError(S.LEARNER_ERR_POPSIZE.format(value=self.popsize))
        if not _is_real(self.sigma0) or not math.isfinite(self.sigma0) or self.sigma0 <= 0:
            raise ValueError(S.LEARNER_ERR_SIGMA.format(value=self.sigma0))
        if not _is_int(self.diagonal_threshold) or self.diagonal_threshold < 0:
            raise ValueError(S.LEARNER_ERR_DIAGONAL_THRESHOLD.format(value=self.diagonal_threshold))
        if not isinstance(self.ipop, bool):
            raise ValueError(S.LEARNER_ERR_IPOP.format(value=self.ipop))
        if not _is_int(self.seed) or not 0 <= self.seed < _SEED_LIMIT:
            raise ValueError(S.LEARNER_ERR_SEED.format(value=self.seed))
        if not _is_int(self.bounded_dims) or self.bounded_dims < 0:
            raise ValueError(S.LEARNER_ERR_BOUNDED_DIMS.format(value=self.bounded_dims))


def _mean_pheno(es: Any, bounded_dims: int) -> np.ndarray:
    """분포 평균을 경계 안으로 옮긴 표현형(pycma result.xfavorite와 같은 계산)."""
    x = np.array(es.gp.pheno(es.mean, into_bounds=es.boundary_handler.repair), dtype=np.float64, copy=True)
    if bounded_dims:
        np.clip(x[:bounded_dims], -1.0, 1.0, out=x[:bounded_dims])
    return x


def _is_int(v: object) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _is_real(v: object) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


@contextmanager
def _quiet_cma() -> Iterator[None]:
    """pycma가 내는 경고(전역 verbosity라 opts로 안 꺼짐)와 그 안의 numpy 경고를 숨긴다."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module=r"cma(\.|$)")
        yield


class CMAES:
    """Learner 구현. 점수는 높을수록 좋고 pycma에는 부호를 뒤집어 넘긴다."""

    name = "cmaes"

    def __init__(self, x0: np.ndarray, config: CMAESConfig) -> None:
        x = np.array(x0, dtype=np.float64, copy=True)
        if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError(S.LEARNER_ERR_X0)
        if config.bounded_dims > x.size:
            raise ValueError(S.LEARNER_ERR_BOUNDED_DIMS_TOO_MANY.format(value=config.bounded_dims, dim=x.size))
        self._cma = import_cma()
        self._config = config
        self._dim = int(x.size)
        k = config.bounded_dims
        x[:k] = np.clip(x[:k], -1.0, 1.0)
        self._rng = np.random.RandomState(config.seed)
        self._popsize = config.popsize
        self._restarts = 0
        self._generation = 0  # tell 횟수(재시작을 넘어 누적)
        self._evaluations = 0
        self._es: Any = self._new_es(x, self._popsize)
        self._pending: list[np.ndarray] | None = None  # pycma ask()가 돌려준 객체 그대로
        self._pending_array: np.ndarray | None = None
        # ask와 tell 사이에만 있다: ask 직전 pycma 객체(피클)와 그때의 diversity. state_dict()가 쓴다.
        self._boundary: tuple[bytes, float] | None = None
        self._diversity = math.nan
        self._mean_step = math.nan
        self._stop_reasons: tuple[str, ...] = ()
        self._best = self._mean_pheno()
        self._axis_cache: tuple[tuple[int, int], float] | None = None

    # ------------------------------------------------------------------ 읽기 전용
    @property
    def popsize(self) -> int:
        return self._popsize

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def config(self) -> CMAESConfig:
        return self._config

    @property
    def diagonal(self) -> bool:
        """대각 공분산(sep-CMA)으로 도는가."""
        return self._dim > self._config.diagonal_threshold

    @property
    def restarts(self) -> int:
        return self._restarts

    @property
    def generation(self) -> int:
        """지금까지 tell()한 세대 수(재시작을 넘어 누적)."""
        return self._generation

    @property
    def stop_reasons(self) -> tuple[str, ...]:
        """마지막 tell() 뒤 pycma 종료 조건 이름(예: 'tolfun'). ipop이면 이 조건으로 재시작했다."""
        return self._stop_reasons

    # ------------------------------------------------------------------ ask/tell
    def ask(self) -> np.ndarray:
        """(popsize, 차원) 후보. tell() 전에 다시 부르면 같은 후보를 다시 돌려준다(난수를 더 쓰지 않음)."""
        if self._pending_array is None:
            boundary = (pickle.dumps(self._es, protocol=_PICKLE_PROTOCOL), self._diversity)
            with _quiet_cma():
                solutions = self._es.ask()
            arr = np.array(solutions, dtype=np.float64)
            k = self._config.bounded_dims
            if k:
                np.clip(arr[:, :k], -1.0, 1.0, out=arr[:, :k])  # BoundTransform 결과는 이미 경계 안
            self._pending = solutions
            self._pending_array = arr
            self._boundary = boundary
            self._diversity = float(np.mean(np.std(arr, axis=0)))
        return self._pending_array.copy()

    def tell(self, candidates: np.ndarray, scores: np.ndarray) -> None:
        if self._pending is None or self._pending_array is None:
            raise RuntimeError(S.LEARNER_ERR_TELL_WITHOUT_ASK)
        cand = np.asarray(candidates)
        if cand.shape != self._pending_array.shape or not np.array_equal(cand, self._pending_array):
            raise ValueError(S.LEARNER_ERR_TELL_CANDIDATES)
        fitness = -sanitize_scores(scores, self._popsize)  # pycma는 최소화
        previous = self._best
        with _quiet_cma():
            self._es.tell(self._pending, fitness.tolist())
            # 종료 조건 확인은 부작용(평탄 점수 세대 수)이 있어 세대마다 정확히 한 번.
            self._stop_reasons = tuple(str(key) for key in self._es.stop())
        self._pending = None
        self._pending_array = None
        self._boundary = None
        self._generation += 1
        self._evaluations += self._popsize
        self._best = self._mean_pheno()
        self._mean_step = float(np.linalg.norm(self._best - previous))
        if self._config.ipop and self._stop_reasons:
            self._restart()

    def best(self) -> np.ndarray:
        """분포 평균을 경계 안으로 옮긴 것(pycma result.xfavorite와 같은 계산)."""
        return self._best.copy()

    def stats(self) -> dict[str, float]:
        """sigma: pycma 단계 크기, axis_ratio: 탐색 분포 공분산의 가장 긴 축 ÷ 가장 짧은 축(전체 공분산은
        고윳값 계산 정밀도 때문에 최대 1/√eps ≈ 6.7e7),
        std_mean: 좌표별 표준편차 평균(sigma × 대각 디코딩 × √C 대각), mean_step: 마지막 tell에서 추천해가 움직인 거리,
        diversity: 마지막 ask 후보의 유전자별 표준편차 평균, popsize, restarts, generation, evaluations.
        """
        es = self._es
        return {
            "sigma": float(es.sigma),
            "axis_ratio": self._axis_ratio(),
            "std_mean": float(np.mean(es.stds)),
            "mean_step": self._mean_step,
            "diversity": self._diversity,
            "popsize": float(self._popsize),
            "restarts": float(self._restarts),
            "generation": float(self._generation),
            "evaluations": float(self._evaluations),
        }

    # ------------------------------------------------------------------ 상태
    def state_dict(self) -> dict[str, Any]:
        """세대 경계 상태. pycma 객체는 피클한 bytes로 담겨 이후 진행과 무관한 사본이다.

        ask()와 tell() 사이에 부르면 그 ask() 직전 상태(이번 세대를 시작하기 전)를 돌려준다. 불러와서 ask()하면
        같은 후보가 나온다.
        """
        if self._boundary is not None:
            es_blob, diversity = self._boundary
        else:
            es_blob, diversity = pickle.dumps(self._es, protocol=_PICKLE_PROTOCOL), self._diversity
        return {
            "format": STATE_FORMAT,
            "name": self.name,
            "config": asdict(self._config),
            "dim": self._dim,
            "popsize": self._popsize,
            "restarts": self._restarts,
            "generation": self._generation,
            "evaluations": self._evaluations,
            "diversity": diversity,
            "mean_step": self._mean_step,
            "stop_reasons": list(self._stop_reasons),
            "pycma_version": str(self._cma.__version__),
            "es": es_blob,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """저장된 상태로 바꾼다. 설정(개체 수, 시그마 등)도 저장된 값이 생성자 값보다 우선한다.
        설정을 바꿔 이어 가려면 CMAES(이전 best(), 새 설정)로 새로 만든다. 유전자 차원이 다르면 ValueError.
        상태가 잘못되었으면(빠진 항목, 깨진 피클, 서로 안 맞는 값) ValueError이고, 그때 이 학습 방법은 그대로다.
        """
        if not isinstance(d, Mapping):
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=type(d).__name__))
        if d.get("name") != self.name:
            raise ValueError(S.LEARNER_ERR_STATE_KIND.format(got=d.get("name"), want=self.name))
        if d.get("format") != STATE_FORMAT:
            raise ValueError(S.LEARNER_ERR_STATE_FORMAT.format(got=d.get("format")))
        # 모두 읽고 검사한 뒤에야 바꾼다(중간에 실패해도 반쯤 불러온 상태가 남지 않게).
        try:
            dim = int(d["dim"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=e)) from e
        if dim != self._dim:
            raise ValueError(S.LEARNER_ERR_STATE_DIM.format(got=dim, want=self._dim))
        try:
            config = CMAESConfig(**d["config"])
            popsize = int(d["popsize"])
            restarts = int(d["restarts"])
            generation = int(d["generation"])
            evaluations = int(d["evaluations"])
            diversity = float(d["diversity"])
            mean_step = float(d["mean_step"])
            stop_reasons = tuple(str(s) for s in d["stop_reasons"])
        except (KeyError, TypeError, ValueError) as e:  # 빠진 항목, 모르는 설정 이름, 잘못된 값
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=e)) from e
        try:
            with _quiet_cma():
                es = pickle.loads(d["es"])
            rng = es.opts["randn"].__self__
            es_dim, es_popsize = int(es.N), int(es.popsize)
            best = _mean_pheno(es, config.bounded_dims)
        except Exception as e:  # 다른 pycma 버전에서 만든 상태, pycma 객체가 아님 등
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=e)) from e
        if not isinstance(rng, np.random.RandomState):
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=type(rng).__name__))
        if (es_dim, es_popsize) != (self._dim, popsize) or best.shape != (self._dim,):
            raise ValueError(S.LEARNER_ERR_STATE_BROKEN.format(reason=f"N={es_dim}, popsize={es_popsize}/{popsize}"))
        self._config = config
        self._es = es
        self._rng = rng  # 재시작도 같은 난수 흐름을 이어 쓴다
        self._popsize = popsize
        self._restarts = restarts
        self._generation = generation
        self._evaluations = evaluations
        self._diversity = diversity
        self._mean_step = mean_step
        self._stop_reasons = stop_reasons
        self._pending = None
        self._pending_array = None
        self._boundary = None
        self._best = best
        self._axis_cache = None

    # ------------------------------------------------------------------ 내부
    def _new_es(self, x0: np.ndarray, popsize: int) -> Any:
        n, k = self._dim, self._config.bounded_dims
        opts: dict[str, Any] = {
            "popsize": popsize,
            "randn": self._rng.randn,
            "seed": np.nan,
            "verbose": -9,
            "verb_log": 0,
            "verb_disp": 0,
            "signals_filename": "",
            "CMA_diagonal": True if self.diagonal else 0,
        }
        if k:
            opts["bounds"] = [[-1.0] * k + [None] * (n - k), [1.0] * k + [None] * (n - k)]
        with _quiet_cma():
            return self._cma.CMAEvolutionStrategy(x0, self._config.sigma0, opts)

    def _restart(self) -> None:
        """IPOP: 지금 추천해에서 개체 수 2배(상한 있음), 초기 시그마로 새로 시작. 난수 흐름은 이어 쓴다."""
        limit = self._config.popsize * 2**IPOP_MAX_DOUBLINGS
        self._popsize = min(self._popsize * 2, limit)
        self._es = self._new_es(self._best.copy(), self._popsize)
        self._restarts += 1
        self._best = self._mean_pheno()
        self._axis_cache = None

    def _mean_pheno(self) -> np.ndarray:
        return _mean_pheno(self._es, self._config.bounded_dims)

    def _axis_ratio(self) -> float:
        """√(공분산 고윳값 최대 ÷ 최소). 공분산 = sigma² · S·C·S (S = 대각 디코딩). 세대마다 한 번만 계산."""
        key = (self._generation, self._restarts)
        if self._axis_cache is not None and self._axis_cache[0] == key:
            return self._axis_cache[1]
        es = self._es
        scaling = np.asarray(es.sigma_vec.scaling, dtype=np.float64) * np.ones(self._dim)
        top = float(np.max(scaling))
        if top > 0 and math.isfinite(top):
            # 비는 공통 배율과 무관하다. 경계가 있으면 pycma가 scaling을 약 1/sigma0로 두어 큰 sigma0에서 제곱이 0이 된다.
            scaling = scaling / top
        c = getattr(es.sm, "C", None)
        if isinstance(c, np.ndarray) and c.ndim == 2:
            ev = np.linalg.eigvalsh(c * scaling[:, None] * scaling[None, :])
            lo, hi = float(ev[0]), float(ev[-1])
            # 고윳값 오차는 약 eps × 최대 고윳값이라 그보다 작은 최솟값은 0이나 음수로도 나온다(무한대 대신 상한).
            ratio = math.sqrt(hi / max(lo, hi * _EIG_EPS)) if hi > 0 else math.nan
        else:
            sd = scaling * np.sqrt(np.asarray(es.sm.variances, dtype=np.float64))
            ratio = float(sd.max() / sd.min()) if sd.min() > 0 else math.inf
        self._axis_cache = (key, ratio)
        return ratio

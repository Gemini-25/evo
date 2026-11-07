"""관찰 기록 형식 (v1.2 「보는 재미」, 명세 부록 A33·A34, 설계 2.2절과 D10).

학습기가 세대마다 쓰는 세대 대표 파일(champions\\gen_XXXXXX_representative.npz)에 `obs_`로 시작하는 키를 함께
담는다. 학습, 이어하기, 재현, 판정은 이 키를 읽지 않는다: 결과의 결정성과 이전 실행 호환은 그래서 지켜진다. 이
모듈은 그 형식을 정한다. 쓰는 쪽(engine/observe.summarize), 읽는 쪽(화면 app/lab/history.py), 테스트가 같이 쓴다.
numpy만 쓴다(Qt 없음, 배경 스레드에서 불러도 된다).

키 (OBS_FORMAT = 1). P = 개체 수, T = 학습 시험 수, D = 유전자 차원, NB = 몸 수치 수, U = 신경망 출력 수,
K = 지나간 칸 수. P·T는 세대마다 다를 수 있으므로(IPOP, 학습 중 시험 횟수 변경) 파일마다 모양을 본다.

| 키 | 형 | 내용 | 언제 |
| --- | --- | --- | --- |
| obs_head | int64 (8,) | HEAD_FIELDS: 형식(1), 세대, 학습 방법 번호(통계 learner), 예선 중(0/1), 대표 번호, 유전자 간격 k, 틀 밖 표본 수, T | 늘 |
| obs_trial | uint64 (2,) | 시험 0번 시드, 트랙 번호 + 1(0 = 트랙 없음) | 늘 |
| obs_scores | float64 (P,) | 개체별 집계 점수(정확한 값) | 늘 |
| obs_outcome | uint8 (P, T) | 시험별 결과 OUTCOME_CODES(0 중립, 1 성공, 2 실패) | 늘 |
| obs_end | uint8 (P, T) | 시험별 종료 사건 번호(환경 종료 목록 순서), END_NONE(255) = 없음 | 늘 |
| obs_genomes | float16 (P, D) | clip(X, ±FLOAT16_MAX) | 세대 % k == 0일 때만(그때는 늘) |
| obs_body | float64 (P, NB) | layout.body_values(X) | NB > 0 |
| obs_act_hist | uint32 (2, U, ACTION_BINS) | [0] 집단의 모든 학습 시험 합, [1] 세대 대표의 모든 시험 합. U 순서 = spec.outputs | 평가기가 관찰할 때 |
| obs_act_mean | float32 (P, U) | 개체별 모든 시험의 평균 실제 출력(부호 있음) | 같음 |
| obs_frame | float64 (5,) | 격자 틀 x0, y0, 칸 크기(m), nx, ny | 틀을 만들었을 때 |
| obs_world_key | str (0차원, <U16) | 세계 열쇠 | 같음 |
| obs_dens_cells | int32 (K,) | 지나간 칸 번호(iy·nx + ix), 엄격한 오름차순 | 같음 |
| obs_dens_counts | uint32 (K,) | 칸별 표본 수(1 이상) | 같음 |

묶음 규칙: obs_act_hist·obs_act_mean은 둘 다 있거나 둘 다 없다. obs_frame·obs_world_key·obs_dens_cells·
obs_dens_counts는 넷 다 있거나 넷 다 없다(관찰을 돌려주지 않는 가짜 평가기, 틀을 만들지 못한 세대는 없음).

읽기: read_representative(파일) → (기본 키, obs_ 키), decode(obs_ 키, dim=…, n_body=…, n_out=…) →
(Observation | None, 이유). 화면은 쓰기 전에 consistent(관찰, 같은 세대 통계 줄, 파일의 score)로 D10 검사를 한다.
읽기 함수는 잘못된 기록에 예외를 내지 않는다(None과 짧은 이유, 이유는 개발자용 영어 글).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np

OBS_PREFIX: Final = "obs_"
OBS_FORMAT: Final = 1
"""obs_head[0]. 이보다 큰 값은 새 버전 프로그램이 만든 기록이다(REASON_NEWER)."""
ACTION_BINS: Final = 20
"""행동 분포 칸 수: 구동기 채널 범위(low..high)를 똑같이 나눈다. x == high는 마지막 칸."""
MAX_GRID: Final = 4096
"""격자 틀 nx, ny의 상한(읽기 검사)."""
FLOAT16_MAX: Final = 65504.0
"""obs_genomes를 쓰기 전에 자르는 범위(float16의 가장 큰 유한값)."""

HEAD: Final = "obs_head"
TRIAL: Final = "obs_trial"
SCORES: Final = "obs_scores"
OUTCOME: Final = "obs_outcome"
END: Final = "obs_end"
GENOMES: Final = "obs_genomes"
BODY: Final = "obs_body"
ACT_HIST: Final = "obs_act_hist"
ACT_MEAN: Final = "obs_act_mean"
FRAME: Final = "obs_frame"
WORLD_KEY: Final = "obs_world_key"
DENS_CELLS: Final = "obs_dens_cells"
DENS_COUNTS: Final = "obs_dens_counts"
KEYS: Final = (HEAD, TRIAL, SCORES, OUTCOME, END, GENOMES, BODY, ACT_HIST, ACT_MEAN, FRAME, WORLD_KEY, DENS_CELLS,
               DENS_COUNTS)
"""형식 1의 모든 키. 이 밖의 obs_ 키는 읽을 때 무시한다."""

HEAD_FIELDS: Final = ("format", "generation", "learner", "qualifying", "representative", "genome_stride", "outside",
                      "trials")
"""obs_head의 칸 순서."""
OUTCOME_CODES: Final = {"neutral": 0, "success": 1, "failure": 2}
"""EvalResult.outcomes 글 → obs_outcome 값."""
OUTCOME_SUCCESS: Final = 1
END_NONE: Final = 255
"""obs_end: 종료 사건이 없음(환경 종료 목록에 없는 사건 포함)."""

REASON_NONE: Final = "no observation keys"
"""decode의 이유: 관찰 키가 하나도 없다(v1.0·v1.1 세대, --no-observe, 옛 프로그램이 이어 간 세대). 잘못이 아니다."""
REASON_NEWER: Final = "written by a newer program"
"""decode의 이유: obs_head[0]이 OBS_FORMAT보다 크다."""


class GridFrame(NamedTuple):
    """궤적 밀도의 격자 틀: 왼쪽 아래 (x0, y0), 정사각 칸 크기 cell(m), 칸 수 nx × ny. 칸 번호 = iy·nx + ix."""

    x0: float
    y0: float
    cell: float
    nx: int
    ny: int

    @property
    def rect(self) -> tuple[float, float, float, float]:
        """세계 좌표의 (x0, y0, 너비, 높이)."""
        return (self.x0, self.y0, self.nx * self.cell, self.ny * self.cell)

    def as_array(self) -> np.ndarray:
        """obs_frame 값(float64 (5,))."""
        return np.array([self.x0, self.y0, self.cell, self.nx, self.ny], dtype=np.float64)

    def cell_at(self, x: float, y: float) -> int | None:
        """세계 좌표가 든 칸 번호. 틀 밖이면 None(워커 관찰기와 같은 규칙: 비교를 먼저 해서 int() 버림 함정을 피함)."""
        fx, fy = (x - self.x0) / self.cell, (y - self.y0) / self.cell
        if 0.0 <= fx < self.nx and 0.0 <= fy < self.ny:
            return int(fy) * self.nx + int(fx)
        return None

    def cell_center(self, index: int) -> tuple[float, float]:
        """칸 번호의 가운데 세계 좌표."""
        iy, ix = divmod(int(index), self.nx)
        return (self.x0 + (ix + 0.5) * self.cell, self.y0 + (iy + 0.5) * self.cell)


@dataclass(frozen=True, eq=False)
class Observation:
    """세대 하나의 관찰 기록(decode가 검사를 통과시킨 것). 배열은 읽기 전용이다(고치려면 복사할 것).

    없는 묶음은 None: genomes(유전자 간격 k로 건너뛴 세대), body(몸 유전자 없음), act_*(평가기가 관찰하지 않음),
    frame·world_key·dens_*(틀 없음).
    """

    generation: int
    learner: int
    """이 세대를 만든 학습 방법 번호(통계 learner 열과 같음)."""
    qualifying: bool
    """예선 중에 만든 세대인가(통계 qualifying 열과 같음)."""
    representative: int
    """세대 대표 = argmax(scores)(동점이면 앞쪽)의 개체 번호."""
    genome_stride: int
    """유전자 간격 k: 세대 % k == 0인 세대에만 genomes가 있다."""
    outside: int
    """시험 0번 위치 표본 중 격자 틀 밖이었던 수."""
    trial_seed: int
    """시험 0번 시드."""
    track: int | None
    """시험 0번 트랙 번호(트랙 없는 환경이면 None)."""
    scores: np.ndarray
    """(P,) float64 개체별 집계 점수."""
    outcome: np.ndarray
    """(P, T) uint8 OUTCOME_CODES."""
    end: np.ndarray
    """(P, T) uint8 종료 사건 번호(환경 종료 목록 순서, END_NONE = 없음). 읽는 쪽이 범위를 확인할 것."""
    genomes: np.ndarray | None = None
    """(P, D) float16."""
    body: np.ndarray | None = None
    """(P, NB) float64 물리 단위 몸 수치."""
    act_hist: np.ndarray | None = None
    """(2, U, ACTION_BINS) uint32: [0] 집단 합, [1] 세대 대표 합."""
    act_mean: np.ndarray | None = None
    """(P, U) float32 개체별 평균 실제 출력."""
    frame: GridFrame | None = None
    world_key: str | None = None
    dens_cells: np.ndarray | None = None
    """(K,) int32 지나간 칸 번호, 엄격한 오름차순."""
    dens_counts: np.ndarray | None = None
    """(K,) uint32 칸별 표본 수."""

    @property
    def population(self) -> int:
        return int(self.scores.shape[0])

    @property
    def trials(self) -> int:
        return int(self.outcome.shape[1])

    @property
    def has_actions(self) -> bool:
        return self.act_hist is not None

    @property
    def has_density(self) -> bool:
        return self.frame is not None

    def successes(self) -> np.ndarray:
        """(P,) int64 개체별 성공한 학습 시험 수."""
        return np.count_nonzero(self.outcome == OUTCOME_SUCCESS, axis=1).astype(np.int64)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """저장할 obs_ 키 사전(decode의 반대). save_champion(..., **obs.to_arrays())로 쓸 수 있다.

        묶음이 반만 있으면(act_hist만, frame만 등) ValueError: 형식은 묶음째 있거나 없다.
        """
        if (self.act_hist is None) != (self.act_mean is None):
            raise ValueError("act_hist and act_mean must both be set or both be None")
        if (self.frame is None) != (self.world_key is None) or (self.frame is None and (
                self.dens_cells is not None or self.dens_counts is not None)):
            raise ValueError("frame and world_key must both be set (density needs a frame) or all be None")
        head = np.array([OBS_FORMAT, self.generation, self.learner, int(bool(self.qualifying)), self.representative,
                         self.genome_stride, self.outside, self.trials], dtype=np.int64)
        trial = np.array([self.trial_seed, 0 if self.track is None else self.track + 1], dtype=np.uint64)
        out: dict[str, np.ndarray] = {
            HEAD: head,
            TRIAL: trial,
            SCORES: np.ascontiguousarray(self.scores, dtype=np.float64),
            OUTCOME: np.ascontiguousarray(self.outcome, dtype=np.uint8),
            END: np.ascontiguousarray(self.end, dtype=np.uint8),
        }
        if self.genomes is not None:
            out[GENOMES] = np.ascontiguousarray(self.genomes, dtype=np.float16)
        if self.body is not None:
            out[BODY] = np.ascontiguousarray(self.body, dtype=np.float64)
        if self.act_hist is not None and self.act_mean is not None:
            out[ACT_HIST] = np.ascontiguousarray(self.act_hist, dtype=np.uint32)
            out[ACT_MEAN] = np.ascontiguousarray(self.act_mean, dtype=np.float32)
        if self.frame is not None and self.world_key is not None:
            out[FRAME] = self.frame.as_array()
            out[WORLD_KEY] = np.array(str(self.world_key))
            cells = np.zeros(0, np.int32) if self.dens_cells is None else self.dens_cells
            counts = np.zeros(0, np.uint32) if self.dens_counts is None else self.dens_counts
            out[DENS_CELLS] = np.ascontiguousarray(cells, dtype=np.int32)
            out[DENS_COUNTS] = np.ascontiguousarray(counts, dtype=np.uint32)
        return out


class _Bad(Exception):
    """decode 안에서만: 검사 실패(이유)."""


def _readonly(a: np.ndarray) -> np.ndarray:
    v = a.view()
    v.flags.writeable = False
    return v


def _array(arrays: Mapping[str, Any], key: str, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    """키의 배열이 정확히 그 형(바이트 순서 포함)과 모양인지 본다."""
    a = arrays[key]
    if not isinstance(a, np.ndarray):
        raise _Bad(f"{key}: not an array ({type(a).__name__})")
    if a.dtype != np.dtype(dtype):
        raise _Bad(f"{key}: dtype {a.dtype}, want {np.dtype(dtype)}")
    if a.shape != shape:
        raise _Bad(f"{key}: shape {a.shape}, want {shape}")
    return a


def _group(arrays: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    """묶음 키가 모두 있으면 참, 모두 없으면 거짓, 일부만 있으면 잘못된 기록."""
    present = [k in arrays for k in keys]
    if any(present) and not all(present):
        missing = [k for k, p in zip(keys, present, strict=True) if not p]
        raise _Bad(f"incomplete group, missing {', '.join(missing)}")
    return all(present)


def decode(arrays: Mapping[str, Any], *, dim: int, n_body: int, n_out: int,
           generation: int | None = None) -> tuple[Observation | None, str]:
    """obs_ 키(다른 키가 섞여 있어도 됨)를 검사해 Observation으로. 어기면 (None, 짧은 이유), 예외를 내지 않는다.

    dim, n_body, n_out은 그 실행의 유전자 구조(GenomeLayout.dim, n_body, brain.n_out). generation을 주면
    obs_head의 세대가 그 값(파일 이름의 세대)과 같아야 한다. 성공하면 (관찰, "").

    관찰 키가 하나도 없으면 (None, REASON_NONE): 기록이 없는 세대(D12)로, 잘못된 기록이 아니다. 형식이 더 새로우면
    (None, REASON_NEWER). 이 검사는 기록 자체만 본다. 같은 세대 통계 줄과 맞는지는 consistent가 본다(D10).
    """
    for name, value, lo in (("dim", dim, 1), ("n_body", n_body, 0), ("n_out", n_out, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < lo:
            raise ValueError(f"{name} must be an int >= {lo}, got {value!r}")
    if n_body > dim:
        raise ValueError(f"n_body {n_body} > dim {dim}")
    if not any(isinstance(k, str) and k.startswith(OBS_PREFIX) for k in arrays):
        return None, REASON_NONE
    try:
        return _decode(arrays, int(dim), int(n_body), int(n_out), generation), ""
    except _Bad as exc:
        return None, str(exc)
    except Exception as exc:  # 예상 못 한 모양의 값(이상한 형, 깨진 배열): 기록을 버리고 화면은 계속
        return None, f"invalid record ({type(exc).__name__}: {exc})"


def _decode(arrays: Mapping[str, Any], dim: int, n_body: int, n_out: int, generation: int | None) -> Observation:
    for key in (HEAD, TRIAL, SCORES, OUTCOME, END):
        if key not in arrays:
            raise _Bad(f"{key}: missing")
    head = [int(v) for v in _array(arrays, HEAD, "int64", (len(HEAD_FIELDS),))]
    fmt, gen, learner, qualifying, rep, stride, outside, trials = head
    if fmt > OBS_FORMAT:
        raise _Bad(REASON_NEWER)
    if fmt != OBS_FORMAT:
        raise _Bad(f"format {fmt}")
    if gen < 0 or (generation is not None and gen != int(generation)):
        raise _Bad(f"generation {gen}" + ("" if generation is None else f", file says {int(generation)}"))
    if learner < 0 or qualifying not in (0, 1) or stride < 1 or outside < 0 or trials < 1:
        raise _Bad(f"bad head {head}")
    trial = _array(arrays, TRIAL, "uint64", (2,))
    scores = arrays[SCORES]
    if not isinstance(scores, np.ndarray) or scores.ndim != 1 or scores.shape[0] < 1:
        raise _Bad(f"{SCORES}: want a non-empty 1-D array")
    pop = int(scores.shape[0])
    scores = _array(arrays, SCORES, "float64", (pop,))
    if not 0 <= rep < pop:
        raise _Bad(f"representative {rep} outside 0..{pop - 1}")
    outcome = _array(arrays, OUTCOME, "uint8", (pop, trials))
    if outcome.size and int(outcome.max()) > max(OUTCOME_CODES.values()):
        raise _Bad(f"{OUTCOME}: unknown code {int(outcome.max())}")
    end = _array(arrays, END, "uint8", (pop, trials))

    genomes = None
    if gen % stride == 0:
        if GENOMES not in arrays:
            raise _Bad(f"{GENOMES}: missing (generation {gen}, stride {stride})")
        genomes = _array(arrays, GENOMES, "float16", (pop, dim))
        if not np.isfinite(genomes).all():
            raise _Bad(f"{GENOMES}: not finite")
    elif GENOMES in arrays:
        raise _Bad(f"{GENOMES}: present although generation {gen} is not a multiple of stride {stride}")

    body = None
    if n_body > 0:
        if BODY not in arrays:
            raise _Bad(f"{BODY}: missing")
        body = _array(arrays, BODY, "float64", (pop, n_body))
    elif BODY in arrays:
        _array(arrays, BODY, "float64", (pop, 0))  # 몸 유전자가 없으면 빈 배열만 받는다

    act_hist = act_mean = None
    if _group(arrays, (ACT_HIST, ACT_MEAN)):
        act_hist = _array(arrays, ACT_HIST, "uint32", (2, n_out, ACTION_BINS))
        act_mean = _array(arrays, ACT_MEAN, "float32", (pop, n_out))

    frame = world_key = cells = counts = None
    if _group(arrays, (FRAME, WORLD_KEY, DENS_CELLS, DENS_COUNTS)):
        f = _array(arrays, FRAME, "float64", (5,))
        if not np.isfinite(f).all():
            raise _Bad(f"{FRAME}: not finite")
        x0, y0, cell, fnx, fny = (float(v) for v in f)
        if not cell > 0.0 or fnx != math.floor(fnx) or fny != math.floor(fny):
            raise _Bad(f"{FRAME}: bad cell size or grid {f.tolist()}")
        nx, ny = int(fnx), int(fny)
        if not (1 <= nx <= MAX_GRID and 1 <= ny <= MAX_GRID):
            raise _Bad(f"{FRAME}: grid {nx}x{ny} outside 1..{MAX_GRID}")
        frame = GridFrame(x0, y0, cell, nx, ny)
        key = arrays[WORLD_KEY]
        if not isinstance(key, np.ndarray) or key.dtype.kind != "U" or key.ndim != 0 or not str(key[()]):
            raise _Bad(f"{WORLD_KEY}: want a non-empty str")
        world_key = str(key[()])
        raw = arrays[DENS_CELLS]
        k = int(raw.shape[0]) if isinstance(raw, np.ndarray) and raw.ndim == 1 else -1
        if k < 0:
            raise _Bad(f"{DENS_CELLS}: want a 1-D array")
        cells = _array(arrays, DENS_CELLS, "int32", (k,))
        counts = _array(arrays, DENS_COUNTS, "uint32", (k,))
        if k:
            if int(cells[0]) < 0 or int(cells[-1]) >= nx * ny:
                raise _Bad(f"{DENS_CELLS}: outside 0..{nx * ny - 1}")
            if k > 1 and not bool((np.diff(cells.astype(np.int64)) > 0).all()):
                raise _Bad(f"{DENS_CELLS}: not strictly ascending")
            if int(counts.min()) < 1:
                raise _Bad(f"{DENS_COUNTS}: zero count")

    def ro(a: np.ndarray | None) -> np.ndarray | None:
        return None if a is None else _readonly(a)

    return Observation(
        generation=gen, learner=learner, qualifying=bool(qualifying), representative=rep, genome_stride=stride,
        outside=outside, trial_seed=int(trial[0]), track=int(trial[1]) - 1 if int(trial[1]) > 0 else None,
        scores=_readonly(scores), outcome=_readonly(outcome), end=_readonly(end), genomes=ro(genomes), body=ro(body),
        act_hist=ro(act_hist), act_mean=ro(act_mean), frame=frame, world_key=world_key, dens_cells=ro(cells),
        dens_counts=ro(counts),
    )


def read_representative(path: Path | str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """세대 대표 npz를 한 번 열어 (기본 키, obs_ 키)로 나눈다. 없거나 읽을 수 없으면(깨진 zip·npy, 0바이트, 피클,
    잠긴 파일) None. 예외를 내지 않는다.

    기본 키는 RunStore.load_champion이 돌려주던 것(genome, score, env_version)이다. 파일을 잡는 시간은 읽는 동안뿐이다
    (with: 윈도우에서 학습기의 자르기·덮어쓰기를 막지 않게).
    """
    try:
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
    except Exception:  # FileNotFoundError, PermissionError, BadZipFile, EOFError, ValueError(피클·깨진 머리) …
        return None
    base = {k: v for k, v in data.items() if not k.startswith(OBS_PREFIX)}
    obs = {k: v for k, v in data.items() if k.startswith(OBS_PREFIX)}
    return base, obs


def scores_text(scores: Any) -> str:
    """통계 scores 칸과 같은 글: 점수마다 repr(round(v, 4))를 빈칸으로 이음(Trainer._stats_row와 같은 규칙)."""
    return " ".join(repr(round(float(v), 4)) for v in np.asarray(scores, dtype=np.float64).reshape(-1))


def consistent(obs: Observation | None, row: Mapping[str, Any] | None, file_score: Any) -> bool:
    """디스크에서 읽은 관찰을 화면이 써도 되는가(D10). 하나라도 어기면 거짓, 예외를 내지 않는다.

    row는 같은 세대의 통계 줄(stats.csv에서 읽은 글자 칸이거나 학습 중 받은 값), file_score는 같은 세대 대표 파일의
    score. 검사: ① 줄이 있고 세대가 같음 ② float(score_best) == obs.scores.max() ③ scores_text(obs.scores) ==
    줄의 scores 칸 ④ obs.scores[대표] == file_score. 통계 실수는 repr로 저장되므로 정확히 비교한다. 손으로 옮긴 파일,
    덜 끝난 자르기 같은 예외를 거른다.
    """
    if obs is None or row is None or file_score is None:
        return False
    try:
        if int(str(row["generation"])) != obs.generation:
            return False
        if float(row["score_best"]) != float(obs.scores.max()):
            return False
        if str(row["scores"]) != scores_text(obs.scores):
            return False
        return float(np.asarray(file_score, dtype=np.float64).reshape(())) == float(obs.scores[obs.representative])
    except (KeyError, TypeError, ValueError, IndexError):
        return False

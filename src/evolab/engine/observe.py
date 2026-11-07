"""세대 끝 관찰 요약 (v1.2 「보는 재미」, 설계 2.2·4.0절, D4~D7).

학습기가 세대마다 부른다. 계산만 하고 아무것도 쓰지 않는다(쓰기는 Trainer가 세대 대표 파일에 obs_ 키로).
- world_frame: 시험 0번 세계의 격자 틀과 세계 열쇠. 학습기 안에서 따로 만든 Episode(별도 pymunk Space) 하나로
  계산한다. 개체와 학습 방법 난수에 기대지 않는다.
- summarize: 이미 계산된 X(후보), agg(집계 점수), EvalResult(쌍별 관찰 포함)만으로 obs_ 키 사전을 만든다. 정수 합과
  정해진 순서의 덧셈만 써서 워커 수, 분배, 관전·고속과 무관하다.

형식(키, 형, 묶음 규칙)은 runs/observe.py가 정한다. 학습, 이어하기, 재현은 이 값을 읽지 않는다.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from evolab.engine.evaluate import EvalResult
from evolab.policy.genome import GenomeLayout
from evolab.runs import observe as fmt
from evolab.runs.observe import GridFrame
from evolab.spec.compiled import EnvSpec
from evolab.world.physics2d.episode import Episode, EpisodeBuildError, Trial, WorldGeometry

GRID_CELLS: Final = 128
"""격자 틀 긴 변의 칸 수(칸은 정사각형)."""
FRAME_MARGIN: Final = 0.4
"""세계 범위를 긴 변의 이 비율만큼 넓힌다."""
FRAME_MIN_MARGIN: Final = 5.0
"""넓히는 최소 여유(m)."""
MIN_SIDE: Final = 8
"""짧은 변의 최소 칸 수(가늘고 긴 세계도 칸이 한두 줄이 되지 않게, 가운데에 맞춰 늘린다)."""
GENOME_BUDGET: Final = 32768
"""세대당 obs_genomes 값 수 상한: 넘으면 k = ⌈P·D / 이 값⌉세대마다만 남긴다."""
KEY_LENGTH: Final = 16
"""세계 열쇠 길이(sha256 16진수 앞 글자)."""
_MM: Final = 1000.0
"""세계 열쇠는 1mm 단위로 반올림한 모양으로 만든다."""
_ANGLE: Final = 1e6
"""자세 각도는 1마이크로라디안 단위로 반올림."""


def genome_stride(population: int, dim: int) -> int:
    """유전자 간격 k = max(1, ⌈P·D / GENOME_BUDGET⌉)."""
    return max(1, math.ceil(int(population) * int(dim) / GENOME_BUDGET))


def _rounded(values: Sequence[float] | np.ndarray, scale: float) -> bytes:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return np.rint(a * scale).astype("<i8").tobytes()


def world_frame(spec: EnvSpec, env_version: str, trial: Trial) -> tuple[GridFrame | None, str | None]:
    """시험 0번 세계(세계 소유 물체의 시작 자세 모양 + 영역)의 범위 → 격자 틀과 세계 열쇠.

    Episode(spec, None, trial) 하나를 만든다(몸 수치는 세계에 영향이 없다: 세계 단계 값은 r.*와 gen.*만 쓴다).
    범위를 긴 변의 FRAME_MARGIN(최소 FRAME_MIN_MARGIN m)만큼 넓혀 세계 경계로 자르고, 긴 변 GRID_CELLS칸의 정사각
    칸으로 나눈다. 세계 열쇠 = sha256(환경 버전 + 경계·세계 물체·영역 모양을 1mm로 반올림한 값) 앞 16글자: 같은
    세계면 같은 열쇠다(트랙 세트의 같은 트랙). 세계를 만들 수 없으면 (None, None): 잘못된 환경이면 이어지는 평가가
    알린다.
    """
    try:
        ep = Episode(spec, None, trial)
    except (EpisodeBuildError, ValueError):
        return None, None
    return frame_from_geometry(ep.geometry(), env_version)


def frame_from_geometry(geo: WorldGeometry, env_version: str) -> tuple[GridFrame | None, str | None]:
    """world_frame의 계산 부분: Episode.geometry()의 세계 소유 물체(시작 자세)·영역·경계만 본다(에이전트 물체는
    보지 않으므로 몸 수치와 무관하다). 범위가 비었거나 유한하지 않으면 (None, None)."""
    h = hashlib.sha256()
    h.update(b"evolab-world-key-1\0")
    h.update(str(env_version).encode("utf-8") + b"\0")
    h.update(_rounded(geo.bounds, _MM))
    lo_x = lo_y = math.inf
    hi_x = hi_y = -math.inf
    for b in geo.bodies:
        if b.owner != "world":
            continue
        x, y, a = b.pose
        h.update(repr((b.id, b.moving, len(b.shapes))).encode("utf-8"))
        h.update(_rounded((x, y), _MM) + _rounded((a,), _ANGLE))
        c, s = math.cos(a), math.sin(a)
        for sh in b.shapes:
            h.update(repr((sh.kind, len(sh.points))).encode("utf-8"))
            h.update(_rounded([v for p in sh.points for v in p], _MM) + _rounded((sh.radius,), _MM))
            if not sh.points:
                continue
            p = np.asarray(sh.points, dtype=np.float64).reshape(-1, 2)
            wx = x + c * p[:, 0] - s * p[:, 1]
            wy = y + s * p[:, 0] + c * p[:, 1]
            r = max(0.0, float(sh.radius))
            lo_x, hi_x = min(lo_x, float(wx.min()) - r), max(hi_x, float(wx.max()) + r)
            lo_y, hi_y = min(lo_y, float(wy.min()) - r), max(hi_y, float(wy.max()) + r)
    for reg in geo.regions:
        h.update(repr((reg.id, reg.kind, len(reg.points))).encode("utf-8"))
        h.update(_rounded([v for p in reg.points for v in p], _MM))
        if reg.points:
            p = np.asarray(reg.points, dtype=np.float64).reshape(-1, 2)
            lo_x, hi_x = min(lo_x, float(p[:, 0].min())), max(hi_x, float(p[:, 0].max()))
            lo_y, hi_y = min(lo_y, float(p[:, 1].min())), max(hi_y, float(p[:, 1].max()))
    bx0, by0, bx1, by1 = (float(v) for v in geo.bounds)
    if math.isfinite(lo_x + lo_y + hi_x + hi_y):
        m = max(FRAME_MIN_MARGIN, FRAME_MARGIN * max(hi_x - lo_x, hi_y - lo_y))
        x0, y0 = max(bx0, lo_x - m), max(by0, lo_y - m)
        x1, y1 = min(bx1, hi_x + m), min(by1, hi_y + m)
        if not (x1 > x0 and y1 > y0):  # 세계 물체가 모두 경계 밖: 경계 전체
            x0, y0, x1, y1 = bx0, by0, bx1, by1
    else:  # 세계 물체도 영역도 없다: 경계 전체
        x0, y0, x1, y1 = bx0, by0, bx1, by1
    w, hgt = x1 - x0, y1 - y0
    if not (math.isfinite(w) and math.isfinite(hgt) and w > 0.0 and hgt > 0.0):
        return None, None
    cell = max(w, hgt) / GRID_CELLS
    nx = max(1, math.ceil(w / cell - 1e-9))
    ny = max(1, math.ceil(hgt / cell - 1e-9))
    if nx < MIN_SIDE:
        x0 -= (MIN_SIDE * cell - w) / 2.0
        nx = MIN_SIDE
    if ny < MIN_SIDE:
        y0 -= (MIN_SIDE * cell - hgt) / 2.0
        ny = MIN_SIDE
    return GridFrame(float(x0), float(y0), float(cell), int(nx), int(ny)), h.hexdigest()[:KEY_LENGTH]


def summarize(
    *, generation: int, X: np.ndarray, agg: np.ndarray, res: EvalResult, trials_per: int, representative: int,
    learner: int, qualifying: bool, layout: GenomeLayout, end_ids: Sequence[str], trial0: Trial,
    frame: GridFrame | Sequence[float] | None, world_key: str | None,
) -> dict[str, np.ndarray]:
    """세대 끝 요약 → obs_ 키 사전(runs/observe.py 형식 1).

    res의 쌍 순서는 학습기와 같다: 개체 i의 시험 k = i·T + k. 행동(obs_act_*)은 res.observations가 있을 때만,
    밀도(obs_frame·obs_world_key·obs_dens_*)는 그에 더해 틀과 열쇠가 있을 때만 담는다. 모양이 맞지 않으면 ValueError
    (학습기는 이 세대를 관찰 없이 쓴다, D9).
    """
    X = np.asarray(X, dtype=np.float64)
    agg = np.asarray(agg, dtype=np.float64)
    P, T = int(X.shape[0]), int(trials_per)
    if X.ndim != 2 or X.shape[1] != layout.dim or agg.shape != (P,) or T < 1 or len(res.scores) != P * T:
        raise ValueError(f"summary shapes: X {X.shape}, agg {agg.shape}, trials {T}, pairs {len(res.scores)}")
    if not 0 <= int(representative) < P:
        raise ValueError(f"representative {representative} outside 0..{P - 1}")
    rep = int(representative)
    gen = int(generation)
    stride = genome_stride(P, layout.dim)
    out: dict[str, np.ndarray] = {}
    out[fmt.TRIAL] = np.array([int(trial0.seed), 0 if trial0.track is None else int(trial0.track) + 1],
                              dtype=np.uint64)
    out[fmt.SCORES] = agg.copy()
    if len(res.outcomes) != P * T or len(res.end_events) != P * T:
        raise ValueError("outcomes or end events do not match the pairs")
    out[fmt.OUTCOME] = np.array([fmt.OUTCOME_CODES.get(o, 0) for o in res.outcomes], dtype=np.uint8).reshape(P, T)
    end_index: dict[str, int] = {}
    for k, e in enumerate(end_ids[:fmt.END_NONE]):
        end_index.setdefault(e, k)  # 같은 사건이 두 번 있으면 앞 번호(통계 end_ 열과 같은 목록)
    out[fmt.END] = np.array([end_index.get(e, fmt.END_NONE) for e in res.end_events], dtype=np.uint8).reshape(P, T)
    if gen % stride == 0:
        out[fmt.GENOMES] = np.clip(X, -fmt.FLOAT16_MAX, fmt.FLOAT16_MAX).astype(np.float16)
    if layout.n_body:
        out[fmt.BODY] = np.ascontiguousarray(layout.body_values(X), dtype=np.float64).reshape(P, layout.n_body)

    outside = 0
    obs = res.observations
    if obs is not None:
        if len(obs) != P * T or any(o is None for o in obs):
            raise ValueError("observations are missing for some pairs")
        U = layout.brain.n_out
        width = U * fmt.ACTION_BINS
        total = np.zeros(width, dtype=np.uint64)
        mine = np.zeros(width, dtype=np.uint64)
        mean = np.zeros((P, U), dtype=np.float32)
        for i in range(P):
            sums = [0.0] * U
            steps = 0
            for k in range(T):
                o: Any = obs[i * T + k]
                h = np.asarray(o.hist)
                if h.shape != (width,) or len(o.out_sum) != U:
                    raise ValueError(f"observation shapes: hist {h.shape}, out_sum {len(o.out_sum)}, outputs {U}")
                total += h.astype(np.uint64)
                if i == rep:
                    mine += h.astype(np.uint64)
                for c in range(U):  # 시험 순서(k = 0..T-1)로 더한다
                    sums[c] += float(o.out_sum[c])
                steps += int(o.steps)
            if steps:
                mean[i] = np.array([v / steps for v in sums], dtype=np.float64).astype(np.float32)
        top = np.uint64(np.iinfo(np.uint32).max)
        out[fmt.ACT_HIST] = np.minimum(np.stack([total, mine]), top).astype(np.uint32).reshape(2, U, fmt.ACTION_BINS)
        out[fmt.ACT_MEAN] = mean
        if frame is not None and world_key is not None:
            f = GridFrame(*(float(v) for v in frame[:3]), int(frame[3]), int(frame[4]))
            n_cells = f.nx * f.ny
            first = [obs[i * T] for i in range(P)]
            cells = np.concatenate([np.asarray(o.cells, dtype=np.int64) for o in first])
            counts = np.concatenate([np.asarray(o.counts, dtype=np.int64) for o in first])
            if cells.size and (int(cells.min()) < 0 or int(cells.max()) >= n_cells):
                raise ValueError(f"density cells outside 0..{n_cells - 1}")
            acc = np.zeros(n_cells, dtype=np.int64)
            np.add.at(acc, cells, counts)  # 정수 합: 순서와 무관
            nz = np.flatnonzero(acc)
            outside = sum(int(o.outside) for o in first)
            out[fmt.FRAME] = f.as_array()
            out[fmt.WORLD_KEY] = np.array(str(world_key))
            out[fmt.DENS_CELLS] = nz.astype(np.int32)
            out[fmt.DENS_COUNTS] = np.minimum(acc[nz], np.iinfo(np.uint32).max).astype(np.uint32)
    head = np.array([fmt.OBS_FORMAT, gen, int(learner), 1 if qualifying else 0, rep, stride, outside, T],
                    dtype=np.int64)
    return {fmt.HEAD: head, **out}

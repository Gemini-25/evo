"""개체 묶음 평가 (워커 안에서 돈다).

맡은 (유전자, 시험) 쌍들의 Episode를 한 스텝씩 함께 진행하고, 신경망은 묶어서 한 번에 계산한다.
결과는 쌍마다 독립적이다: 묶음 구성이 달라도(워커 수, 분배) 각 쌍의 결과는 비트 단위로 같다.
- Episode 결과는 (환경, 몸 수치, Trial)만의 함수
- BatchBrain.forward는 행마다 독립(묶음 크기, 다른 행과 무관)

관찰 기록(v1.2, 설계 D3~D5): observe[i]가 1·2인 쌍은 스텝마다 실제 구동기 출력을 20칸 히스토그램으로 세고(1·2),
2이면 4스텝마다 기준 물체 위치를 격자 틀의 칸으로 센다. 관찰기는 Episode의 읽기 전용 접근자(applied_outputs,
output_channel_index, anchor_body)와 pymunk Body.position만 읽고 난수를 쓰지 않는다. 그래서 점수·궤적은 관찰을
켜도 같고, 쌍별 관찰은 (환경, 유전자, 시험, 틀)만의 함수다.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evolab.policy.brain import BatchBrain, unpack
from evolab.policy.genome import GenomeLayout
from evolab.runs.observe import ACTION_BINS
from evolab.spec.compiled import EnvSpec
from evolab.world.physics2d.episode import Episode, Trial

POSITION_EVERY = 4
"""관찰 코드 2인 쌍의 위치 표본 간격(스텝). 표본 수 = 스텝 // 4."""
OBSERVE_NONE, OBSERVE_ACTIONS, OBSERVE_POSITIONS = 0, 1, 2
"""쌍마다 관찰 코드: 0 안 봄, 1 행동, 2 행동 + 위치(격자 틀과 기준 물체가 있을 때)."""


@dataclass
class Trajectory:
    """관전·재생용 궤적. 프레임 k는 k*every 스텝(시각 k*every*dt), 마지막 프레임은 끝난 상태."""

    poses: np.ndarray
    """(F, 움직이는 물체 수, 3) float32: x, y, 각도. 순서는 Episode.moving_bodies."""
    outputs: np.ndarray
    """(F, 구동기 출력 수) float32: 실제 출력(연료가 떨어지면 0)."""
    every: int
    dt: float

    @property
    def frames(self) -> int:
        return int(self.poses.shape[0])

    @property
    def duration(self) -> float:
        """마지막 프레임의 시각(초). 마지막 프레임은 every 간격보다 가까울 수 있다."""
        return float(self.times()[-1]) if self.frames else 0.0

    def times(self) -> np.ndarray:
        t = np.arange(self.frames, dtype=np.float64) * (self.every * self.dt)
        if self.frames and self.end_step is not None:
            t[-1] = self.end_step * self.dt
        return t

    end_step: int | None = None
    """끝난 스텝 수(마지막 프레임의 시각)."""


class _Recorder:
    def __init__(self, every: int, dt: float) -> None:
        self.every = every
        self.dt = dt
        self.poses: list[np.ndarray] = []
        self.outputs: list[np.ndarray] = []
        self.steps = 0

    def add(self, ep: Episode) -> None:
        f = ep.snapshot()
        self.poses.append(f.poses.astype(np.float32))
        self.outputs.append(f.outputs.astype(np.float32))

    def after_step(self, ep: Episode) -> None:
        self.steps += 1
        if ep.done or self.steps % self.every == 0:
            self.add(ep)

    def build(self) -> Trajectory:
        b = self.poses[0].shape[0]
        u = self.outputs[0].shape[0]
        return Trajectory(
            np.stack(self.poses).reshape(len(self.poses), b, 3),
            np.stack(self.outputs).reshape(len(self.outputs), u),
            self.every, self.dt, self.steps,
        )


@dataclass
class PairObservation:
    """쌍 하나의 관찰(v1.2). 정수 횟수는 순서와 무관하고, out_sum은 스텝 순서로 더한 값이다."""

    hist: np.ndarray
    """(U * ACTION_BINS,) uint32 실제 출력 횟수. 신경망 출력 순서(spec.outputs), 출력마다 20칸."""
    out_sum: np.ndarray
    """(U,) float64 실제 출력(부호 있음)의 합."""
    steps: int
    """센 스텝 수(= 에피소드 스텝 수). 출력마다 hist 합과 같다."""
    cells: np.ndarray
    """(K,) int32 지나간 격자 칸 번호(iy·nx + ix), 엄격한 오름차순. 코드 2가 아니면 빈 배열."""
    counts: np.ndarray
    """(K,) uint32 칸별 표본 수."""
    outside: int
    """격자 틀 밖이었던 위치 표본 수. 코드 2 쌍은 counts.sum() + outside == steps // POSITION_EVERY."""


class _Observer:
    """쌍 하나의 행동 관찰(코드 1). ep.step 뒤에 부른다. 상태를 읽기만 한다(applied_outputs).

    스텝마다 하는 일은 그 스텝의 실제 출력 리스트를 붙이는 것뿐이다(Episode는 스텝마다 새 리스트로 바꾸고 옛 리스트를
    고치지 않는다). FLUSH 스텝마다와 끝에 numpy로 센다: 칸 = int((x - low) · 20 / (high - low))를 0..19로 자름(파이썬
    int()와 같은 버림), 합은 cumsum이라 스텝 순서로 더한 값과 비트 단위로 같다. 파이썬 루프로 스텝마다 세는 것보다 싸다
    (착륙선 0.53 → 0.28µs/스텝, 부하 걸린 기계에서 잼, 보고서 M2).
    """

    __slots__ = ("idx", "lo", "k", "offs", "buf", "hist", "total", "done")
    FLUSH = 128
    """버퍼 크기(스텝). 쌍마다 출력 리스트 이만큼만 붙잡는다(약 20KB)."""

    def __init__(self, ep: Episode, ranges: Sequence[tuple[float, float]]) -> None:
        self.idx = np.asarray(ep.output_channel_index, dtype=np.intp)
        self.lo = np.array([float(lo) for lo, _ in ranges], dtype=np.float64)
        self.k = np.array([ACTION_BINS / (float(hi) - float(lo)) for lo, hi in ranges], dtype=np.float64)
        self.offs = np.arange(len(ranges), dtype=np.int64) * ACTION_BINS
        self.buf: list[list[float]] = []
        self.hist = np.zeros(len(ranges) * ACTION_BINS, dtype=np.int64)
        self.total = np.zeros((1, len(ranges)), dtype=np.float64)
        self.done = 0

    def after_step(self, ep: Episode) -> None:
        buf = self.buf
        buf.append(ep.applied_outputs)
        if len(buf) >= self.FLUSH:
            self._flush()

    def _flush(self) -> None:
        if not self.buf:
            return
        a = np.array(self.buf, dtype=np.float64).reshape(len(self.buf), -1)[:, self.idx]  # (스텝, 출력) 신경망 순서
        self.done += len(self.buf)
        self.buf = []
        b = ((a - self.lo) * self.k).astype(np.int64)  # x == high면 20 → 19칸, low보다 작으면 0칸
        np.clip(b, 0, ACTION_BINS - 1, out=b)
        b += self.offs
        self.hist += np.bincount(b.ravel(), minlength=self.hist.size)
        self.total = np.cumsum(np.concatenate((self.total, a)), axis=0)[-1:]  # 스텝 순서로 더함

    def _positions(self) -> tuple[np.ndarray, np.ndarray, int]:
        return np.zeros(0, np.int32), np.zeros(0, np.uint32), 0

    def build(self) -> PairObservation:
        self._flush()
        cells, counts, outside = self._positions()
        return PairObservation(self.hist.astype(np.uint32), self.total[0].copy(), self.done, cells, counts, outside)


class _PositionObserver(_Observer):
    """행동 + 위치 관찰(코드 2): POSITION_EVERY 스텝마다 기준 물체 위치(pymunk Body.position)를 붙여 두었다가 격자
    칸으로 센다. 칸 = int(fy)·nx + int(fx), fx = (x - x0) / 칸 크기. 틀 밖(0 ≤ fx < nx, 0 ≤ fy < ny가 아님, NaN 포함)은
    outside로 센다(비교를 먼저 해서 int() 버림 함정을 피함, GridFrame.cell_at과 같은 규칙)."""

    __slots__ = ("body", "frame", "pos", "wait", "outside", "cells")

    def __init__(self, ep: Episode, ranges: Sequence[tuple[float, float]], frame: Sequence[float], body: str) -> None:
        super().__init__(ep, ranges)
        x0, y0, cell, nx, ny = frame
        self.frame = (float(x0), float(y0), float(cell), int(nx), int(ny))
        self.body = ep.bodies[body]
        self.pos: list[Any] = []
        self.wait = POSITION_EVERY
        self.outside = 0
        self.cells: list[np.ndarray] = []

    def after_step(self, ep: Episode) -> None:
        buf = self.buf
        buf.append(ep.applied_outputs)
        w = self.wait - 1
        if w:
            self.wait = w
        else:
            self.wait = POSITION_EVERY
            self.pos.append(self.body.position)
        if len(buf) >= self.FLUSH:
            self._flush()

    def _flush(self) -> None:
        super()._flush()
        if not self.pos:
            return
        x0, y0, cell, nx, ny = self.frame
        p = np.array(self.pos, dtype=np.float64).reshape(len(self.pos), 2)
        self.pos = []
        fx = (p[:, 0] - x0) / cell
        fy = (p[:, 1] - y0) / cell
        inside = (fx >= 0.0) & (fx < nx) & (fy >= 0.0) & (fy < ny)
        self.outside += int(len(p) - np.count_nonzero(inside))
        self.cells.append(fy[inside].astype(np.int64) * nx + fx[inside].astype(np.int64))

    def _positions(self) -> tuple[np.ndarray, np.ndarray, int]:
        if not self.cells:
            return super()._positions()
        cells, counts = np.unique(np.concatenate(self.cells), return_counts=True)
        return cells.astype(np.int32), counts.astype(np.uint32), self.outside


def _make_observer(ep: Episode, ranges: Sequence[tuple[float, float]], code: int,
                   frame: Sequence[float] | None) -> _Observer | None:
    """관찰 코드 → 관찰기. 코드 2라도 틀이나 기준 물체가 없으면 행동만 센다."""
    if code == OBSERVE_NONE:
        return None
    anchor = ep.anchor_body if code == OBSERVE_POSITIONS and frame is not None else None
    if anchor is not None and frame is not None:
        return _PositionObserver(ep, ranges, frame, anchor)
    return _Observer(ep, ranges)


@dataclass
class EvalResult:
    """쌍마다 한 줄. 순서는 요청한 쌍 순서."""

    scores: np.ndarray
    """(n,) float64 시험 점수."""
    steps: np.ndarray
    """(n,) int64 스텝 수."""
    end_events: list[str]
    """종료 사건 id (예: landed, crash, timeout)."""
    outcomes: list[str]
    """success / failure / neutral"""
    blocks: np.ndarray
    """(n, 블록 수) 가중치 전 블록 값, 환경 파일의 블록 순서."""
    action_abs: np.ndarray
    """(n, 출력 수) 구동기 출력 크기의 평균."""
    weighted: np.ndarray | None = None
    """(n, 블록 수) 가중치를 곱한 블록 값(점수에 더해지는 몫). 수식 점수 환경이면 NaN."""
    trajectories: list[Trajectory | None] | None = None
    """요청한 쌍만 궤적, 나머지는 None. 하나도 요청하지 않았으면 None."""
    nn_seconds: float = 0.0
    sim_seconds: float = 0.0
    observations: list[PairObservation | None] | None = None
    """관찰을 요청한(코드 1·2) 쌍만 관찰, 나머지는 None. 하나도 요청하지 않았으면 None(v1.2)."""

    @classmethod
    def concat(cls, parts: Sequence[EvalResult], order: Sequence[tuple[int, int]]) -> EvalResult:
        """여러 조각을 원래 순서로 합친다. order[i] = (조각 번호, 조각 안 줄 번호)."""
        def pick(get):  # noqa: ANN001, ANN202
            return [get(parts[p])[r] for p, r in order]

        return cls(
            scores=np.array(pick(lambda x: x.scores), dtype=np.float64),
            steps=np.array(pick(lambda x: x.steps), dtype=np.int64),
            end_events=pick(lambda x: x.end_events),
            outcomes=pick(lambda x: x.outcomes),
            blocks=np.array(pick(lambda x: x.blocks), dtype=np.float64).reshape(len(order), -1),
            action_abs=np.array(pick(lambda x: x.action_abs), dtype=np.float64).reshape(len(order), -1),
            weighted=np.array(pick(lambda x: x.weighted), dtype=np.float64).reshape(len(order), -1),
            trajectories=(pick(lambda x: x.trajectories or [None] * len(x.scores))
                          if any(x.trajectories for x in parts) else None),
            nn_seconds=sum(x.nn_seconds for x in parts),
            sim_seconds=sum(x.sim_seconds for x in parts),
            observations=(pick(lambda x: x.observations or [None] * len(x.scores))
                          if any(x.observations for x in parts) else None),
        )


@dataclass
class _Active:
    index: np.ndarray
    brain: BatchBrain
    obs: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.obs = np.zeros((len(self.index), self.brain.layout.n_in), dtype=np.float64)


def evaluate(
    spec: EnvSpec, layout: GenomeLayout, genomes: np.ndarray, pairs: Sequence[tuple[int, Trial]],
    record: Sequence[bool] | None = None, every: int = 2, observe: Sequence[int] | None = None,
    frame: Sequence[float] | None = None,
) -> EvalResult:
    """genomes: (G, dim). pairs: (genomes의 행 번호, Trial) 목록.

    record[i]가 참인 쌍은 every 스텝마다 궤적을 남긴다. 궤적 기록은 상태를 읽기만 하므로 점수는 같다.
    observe[i]는 쌍마다 관찰 코드(OBSERVE_*). 코드 2이고 frame(x0, y0, 칸 크기, nx, ny)과 기준 물체가 있을 때만
    위치를 센다. 행동 칸은 layout.output_ranges를 나눈다. 관찰도 상태를 읽기만 하므로 점수는 같다.
    """
    genomes = np.ascontiguousarray(genomes, dtype=np.float64)
    n = len(pairs)
    body = layout.body_values(genomes) if layout.n_body else None
    all_brains = unpack(layout.brain, layout.brain_params(genomes))
    rows = np.array([g for g, _ in pairs], dtype=np.int64)
    t_start = time.perf_counter()  # 세계 만들기도 시뮬레이션 시간에 넣는다
    episodes = [Episode(spec, body[g] if body is not None else None, trial) for g, trial in pairs]
    recorders: dict[int, _Recorder] = {}
    if record is not None:
        for i, on in enumerate(record):
            if on:
                recorders[i] = _Recorder(max(1, int(every)), episodes[i].dt)
                recorders[i].add(episodes[i])
    observers: list[_Observer | None] | None = None
    if observe is not None and any(observe):
        ranges = layout.output_ranges
        observers = [_make_observer(episodes[i], ranges, int(code), frame) for i, code in enumerate(observe)]

    nn_t = 0.0
    active = _Active(np.arange(n), all_brains.select(rows)) if n else None
    while active is not None and len(active.index):
        idx = active.index
        for j, i in enumerate(idx):
            active.obs[j] = episodes[i].observe()
        t0 = time.perf_counter()
        actions = layout.to_actions(active.brain.forward(active.obs))
        nn_t += time.perf_counter() - t0
        still: list[int] = []
        for j, i in enumerate(idx):
            ep = episodes[i]
            if ep.done:
                continue  # 끝난 줄(묶음을 줄이기 전까지 남아 있음)
            ep.step(actions[j])
            rec = recorders.get(i)
            if rec is not None:
                rec.after_step(ep)
            if observers is not None:
                ob = observers[i]
                if ob is not None:
                    ob.after_step(ep)
            if not ep.done:
                still.append(j)
        if len(still) == len(idx):
            continue
        if not still:
            break
        keep = np.array(still, dtype=np.int64)
        if len(keep) * 2 <= len(idx):  # 절반 넘게 끝나면 신경망 묶음을 줄인다(행마다 독립이라 결과는 같다)
            active = _Active(idx[keep], active.brain.select(keep))
        else:
            # 끝난 줄도 계속 계산하되 결과는 쓰지 않는다(끝난 에피소드의 step은 아무 일도 안 한다)
            pass

    results = [ep.result() for ep in episodes]
    block_ids = [b.id for b in spec.file.scoring.blocks]
    return EvalResult(
        scores=np.array([r.score for r in results], dtype=np.float64),
        steps=np.array([r.steps for r in results], dtype=np.int64),
        end_events=[r.end_event for r in results],
        outcomes=[r.outcome for r in results],
        blocks=np.array([[r.blocks[b] for b in block_ids] for r in results], dtype=np.float64).reshape(n, len(block_ids)),
        action_abs=np.array([[r.action_mean_abs[c] for c in spec.outputs] for r in results], dtype=np.float64).reshape(
            n, len(spec.outputs)
        ),
        weighted=np.array([[r.weighted.get(b, np.nan) for b in block_ids] for r in results],
                          dtype=np.float64).reshape(n, len(block_ids)),
        trajectories=[recorders[i].build() if i in recorders else None for i in range(n)] if recorders else None,
        nn_seconds=nn_t,
        sim_seconds=time.perf_counter() - t_start - nn_t,
        observations=None if observers is None else [None if ob is None else ob.build() for ob in observers],
    )

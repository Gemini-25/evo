"""개체 묶음 평가 (워커 안에서 돈다).

맡은 (유전자, 시험) 쌍들의 Episode를 한 스텝씩 함께 진행하고, 신경망은 묶어서 한 번에 계산한다.
결과는 쌍마다 독립적이다: 묶음 구성이 달라도(워커 수, 분배) 각 쌍의 결과는 비트 단위로 같다.
- Episode 결과는 (환경, 몸 수치, Trial)만의 함수
- BatchBrain.forward는 행마다 독립(묶음 크기, 다른 행과 무관)

관찰 기록(v1.2, 설계 D3~D5): observe[i]가 1·2인 쌍은 스텝마다 실제 구동기 출력을 20칸 히스토그램으로 세고(1·2),
2이면 4스텝마다 기준 물체 위치를 격자 틀의 칸으로 센다. 관찰기는 Episode의 읽기 전용 접근자(applied_outputs,
output_channel_index, anchor_body)와 pymunk Body.position만 읽고 난수를 쓰지 않는다. 그래서 점수·궤적은 관찰을
켜도 같고, 쌍별 관찰은 (환경, 유전자, 시험, 틀)만의 함수다.

조립 실패(v2.0, 설계 4.9·D22): 몸 수치 때문에 Episode를 만들 수 없는(EpisodeBuildError) 쌍은 학습을 멈추지 않고 그 쌍만
실패로 센다(점수 NaN, 결과 failure, 종료 사건 BUILD_FAILED_EVENT, 스텝·블록·행동 0, 궤적 프레임 0개, 관찰 0). 그 쌍은
스텝 루프와 신경망 묶음에 넣지 않는다(행마다 독립이라 다른 쌍의 결과는 그대로). 같은 시험을 처음 몸으로도 만들 수 없으면
몸이 아니라 환경의 문제라 v1.2처럼 오류를 올린다. 실패는 (환경, 몸 수치, 시험)의 순수 함수라 어느 워커에서나 같고,
실패가 없으면 v1.2와 같은 길을 그대로 지난다.
"""

from __future__ import annotations

import math
import time
from array import array
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, cast

import numpy as np

from evolab.policy.brain import BatchBrain, unpack
from evolab.policy.genome import GenomeLayout
from evolab.runs.observe import ACTION_BINS
from evolab.spec.compiled import EnvSpec
from evolab.world.physics2d.episode import Episode, EpisodeBuildError, EpisodeResult, Trial

BUILD_FAILED_EVENT: Final = ""
"""조립 실패 쌍의 종료 사건(점수 NaN, 결과 failure, 스텝 0). 어느 종료 사건 id와도 겹치지 않아 통계 end_* 열에 들지 않는다
(그래서 그 세대의 end_* 합이 1보다 작아진다, 설계 4.9)."""
BUILD_FAILED_OUTCOME: Final = "failure"

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
    """쌍 하나의 관찰 버퍼와 세기. 상태를 읽기만 한다(applied_outputs, 기준 물체 Body.position).

    스텝마다 하는 일(이 스텝의 실제 출력을 버퍼에 잇기, POSITION_EVERY 스텝마다 기준 물체 위치 잇기)은 evaluate 루프가
    메서드를 부르지 않고 바로 한다(after_step과 같은 두 줄): 파이썬 함수 호출과 스텝마다 세는 루프가 착륙선 스텝 비용의
    몇 %라서다(보고서 M2). FLUSH_EVERY 루프마다와 끝(build)에 numpy로 센다:
    - 칸 = int((x - low) · 20 / (high - low))를 0..19로 자름(파이썬 int()와 같은 0 쪽 버림, x == high는 19칸)
    - 합 = cumsum(스텝 순서로 더한 값과 비트 단위로 같다)
    - 위치 칸 = int(fy)·nx + int(fx), fx = (x - x0) / 칸 크기. 0 ≤ fx < nx, 0 ≤ fy < ny가 아니면(NaN 포함) 틀 밖
      (비교를 먼저 해서 int() 버림 함정을 피함, GridFrame.cell_at과 같은 규칙)
    """

    __slots__ = ("acts", "pos", "body", "C", "idx", "lo", "k", "offs", "hist", "total", "done", "frame", "outside",
                 "cells")

    def __init__(self, ep: Episode, ranges: Sequence[tuple[float, float]], frame: Sequence[float] | None = None,
                 body: str | None = None) -> None:
        self.acts = array("d")
        """이어 붙인 실제 출력(스텝마다 채널 C개, Episode의 채널 순서). C double로 바로 옮긴다: 파이썬 float 객체를
        붙잡지 않아야 스텝마다 새 메모리를 건드리지 않는다(캐시가 덜 밀림)."""
        self.pos = array("d")
        """이어 붙인 기준 물체 위치(x, y)."""
        self.body = ep.bodies[body] if frame is not None and body is not None else None
        self.C = len(ep.applied_outputs)
        self.idx = np.asarray(ep.output_channel_index, dtype=np.intp)
        self.lo = np.array([float(lo) for lo, _ in ranges], dtype=np.float64)
        self.k = np.array([ACTION_BINS / (float(hi) - float(lo)) for lo, hi in ranges], dtype=np.float64)
        self.offs = np.arange(len(ranges), dtype=np.int64) * ACTION_BINS
        self.hist = np.zeros(len(ranges) * ACTION_BINS, dtype=np.int64)
        self.total = np.zeros((1, len(ranges)), dtype=np.float64)
        self.done = 0
        self.frame = None if self.body is None or frame is None else (
            float(frame[0]), float(frame[1]), float(frame[2]), int(frame[3]), int(frame[4]))
        self.outside = 0
        self.cells: list[np.ndarray] = []

    def after_step(self, ep: Episode) -> None:
        """한 스텝(evaluate 루프가 같은 일을 바로 한다). ep.steps는 방금 끝낸 스텝 수."""
        self.acts.fromlist(ep.applied_outputs)
        if self.body is not None and ep.steps % POSITION_EVERY == 0:
            self.pos.extend(self.body.position)

    def flush(self) -> None:
        """버퍼를 세고 비운다."""
        if self.acts:
            a = np.frombuffer(self.acts, dtype=np.float64).reshape(-1, self.C)[:, self.idx]  # 색인이 사본을 만든다
            self.acts = array("d")
            self.done += a.shape[0]
            b = ((a - self.lo) * self.k).astype(np.int64)  # 0 쪽 버림(int()와 같음)
            np.clip(b, 0, ACTION_BINS - 1, out=b)
            b += self.offs
            self.hist += np.bincount(b.ravel(), minlength=self.hist.size)
            self.total = np.cumsum(np.concatenate((self.total, a)), axis=0)[-1:]  # 스텝 순서로 더함
        if self.pos and self.frame is not None:
            x0, y0, cell, nx, ny = self.frame
            p = np.frombuffer(self.pos, dtype=np.float64).reshape(-1, 2).copy()
            self.pos = array("d")
            fx = (p[:, 0] - x0) / cell
            fy = (p[:, 1] - y0) / cell
            inside = (fx >= 0.0) & (fx < nx) & (fy >= 0.0) & (fy < ny)
            self.outside += int(p.shape[0] - np.count_nonzero(inside))
            self.cells.append(fy[inside].astype(np.int64) * nx + fx[inside].astype(np.int64))

    def build(self) -> PairObservation:
        self.flush()
        if self.cells:
            cells, counts = np.unique(np.concatenate(self.cells), return_counts=True)
        else:
            cells, counts = np.zeros(0, np.int64), np.zeros(0, np.int64)
        return PairObservation(self.hist.astype(np.uint32), self.total[0].copy(), self.done,
                               cells.astype(np.int32), counts.astype(np.uint32), self.outside)


FLUSH_EVERY = 512
"""관찰 버퍼를 세는 간격(묶음 루프 횟수). 쌍마다 실제 출력 512스텝분까지만 붙잡는다(착륙선 출력 3개면 12KB)."""


def _make_observer(ep: Episode, ranges: Sequence[tuple[float, float]], code: int,
                   frame: Sequence[float] | None) -> _Observer | None:
    """관찰 코드 → 관찰기. 코드 2라도 틀이나 기준 물체가 없으면 행동만 센다."""
    if code == OBSERVE_NONE:
        return None
    anchor = ep.anchor_body if code == OBSERVE_POSITIONS and frame is not None else None
    return _Observer(ep, ranges, frame if anchor is not None else None, anchor)


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


def build_failed_result(spec: EnvSpec) -> EpisodeResult:
    """조립 실패 쌍의 결과(설계 4.9): 점수 NaN, 결과 failure, 종료 사건 BUILD_FAILED_EVENT, 시간·스텝 0, 블록·행동 0.

    가중치 블록 값은 가중치 점수 환경이면 0(통계 기여 열이 NaN이 되지 않게), 수식 점수 환경이면 Episode.result()처럼
    비어 있다(EvalResult.weighted는 NaN). 화면(재생·경주)이 실패한 칸의 결과로도 쓸 수 있다.
    """
    from evolab.app import strings as S

    blocks = {b.id: 0.0 for b in spec.file.scoring.blocks}
    weighted = dict(blocks) if spec.formula is None else {}
    channels = [c.name for c in spec.actuator_channels.values()]
    return EpisodeResult(
        score=math.nan, blocks=blocks, weighted=weighted, outcome=BUILD_FAILED_OUTCOME, end_event=BUILD_FAILED_EVENT,
        end_label=S.RT_BUILD_FAILED_END, time=0.0, steps=0,
        action_mean_abs={c: 0.0 for c in [*channels, *spec.outputs]},
    )


def mean_of_built(scores: np.ndarray) -> float:
    """조립된 쌍들의 평균 점수. 조립 실패 쌍(점수 NaN, 설계 4.9)은 빼고 평균하고, 모두 실패면 NaN.

    판정 평균(commands.judge_mean)과 챔피언 비교 점수(Trainer._compare)가 함께 쓰는 규칙이다. NaN이 없으면
    float(scores.mean())과 같은 계산이라 v1·v1.2 결과와 비트 단위로 같다.
    """
    s = np.asarray(scores)
    nan = np.isnan(s)
    if nan.any() and not nan.all():
        return float(s[~nan].mean())
    return float(s.mean())


def _empty_trajectory(spec: EnvSpec, every: int) -> Trajectory:
    """조립 실패 쌍의 궤적: 프레임 0개(물체 0개, 출력 열은 구동기 채널 수), 끝난 스텝 0."""
    u = len(spec.actuator_channels)
    return Trajectory(np.zeros((0, 0, 3), dtype=np.float32), np.zeros((0, u), dtype=np.float32), every,
                      float(spec.file.world.dt), 0)


def _empty_observation(n_out: int) -> PairObservation:
    """조립 실패 쌍의 관찰: 아무것도 세지 않음(스텝 0, 칸 없음). 세대 요약에 0을 더한다."""
    return PairObservation(np.zeros(n_out * ACTION_BINS, dtype=np.uint32), np.zeros(n_out, dtype=np.float64), 0,
                           np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.uint32), 0)


def _build(spec: EnvSpec, body: np.ndarray | None, trial: Trial) -> Episode | EpisodeBuildError:
    """쌍 하나의 Episode. 조립할 수 없으면 그 EpisodeBuildError(돌려줌). 다른 예외는 코드 결함이라 그대로 올린다."""
    try:
        return Episode(spec, body, trial)
    except EpisodeBuildError as exc:
        return exc


def _only_body_failures(spec: EnvSpec, pairs: Sequence[tuple[int, Trial]],
                        errors: dict[int, EpisodeBuildError]) -> None:
    """조립 실패가 그 몸 수치 때문인지 본다(설계 D22: 처음 몸은 조립되는데 특정 몸 수치로만 안 될 때만 그 쌍의 실패).

    같은 시험을 처음 몸(Episode(spec, None, 시험): 몸 수치 = 환경 파일의 처음 값)으로도 조립할 수 없으면 몸이 아니라
    환경·시험의 문제라 v1.2처럼 그 오류를 올린다(학습이 멈추고 워커 풀은 PoolError로 알린다). 실패가 있을 때만
    시험마다 한 번 더 만들어 본다.
    """
    reference: dict[tuple[int, int | None], bool] = {}
    for i in sorted(errors):
        trial = pairs[i][1]
        key = (trial.seed, trial.track)
        if key not in reference:
            reference[key] = not isinstance(_build(spec, None, trial), EpisodeBuildError)
        if not reference[key]:
            raise errors[i]


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
    made = [_build(spec, body[g] if body is not None else None, trial) for g, trial in pairs]
    errors = {i: ep for i, ep in enumerate(made) if isinstance(ep, EpisodeBuildError)}
    if errors:
        _only_body_failures(spec, pairs, errors)
    built: list[Episode | None] = [None if i in errors else cast(Episode, ep) for i, ep in enumerate(made)]
    failed = sorted(errors)
    del made, errors  # 예외의 traceback이 만들다 만 세계를 붙잡지 않게
    episodes = cast("list[Episode]", built)  # 스텝 루프는 만든 쌍(active.index)만 다룬다
    recorders: dict[int, _Recorder] = {}
    if record is not None:
        for i, on in enumerate(record):
            ep = built[i]
            if on and ep is not None:
                recorders[i] = _Recorder(max(1, int(every)), ep.dt)
                recorders[i].add(ep)
    observers: list[_Observer | None] | None = None
    if observe is not None and any(observe):
        ranges = layout.output_ranges
        observers = [None if ep is None else _make_observer(ep, ranges, int(code), frame)
                     for ep, code in zip(built, observe, strict=True)]

    nn_t = 0.0
    sweeps = 0
    if failed:
        live = np.array([i for i, ep in enumerate(built) if ep is not None], dtype=np.int64)
        active = _Active(live, all_brains.select(rows[live])) if len(live) else None
    else:  # v1.2와 같은 길(실패가 없으면 결과가 비트 단위로 같다)
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
                if ob is not None:  # = ob.after_step(ep), 메서드 호출 없이(상태를 읽기만 한다)
                    ob.acts.fromlist(ep.applied_outputs)
                    if ob.body is not None and ep.steps % POSITION_EVERY == 0:
                        ob.pos.extend(ob.body.position)
            if not ep.done:
                still.append(j)
        if observers is not None:
            sweeps += 1
            if sweeps % FLUSH_EVERY == 0:
                for ob in observers:
                    if ob is not None:
                        ob.flush()
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

    failed_result = build_failed_result(spec) if failed else None  # 실패 쌍이 모두 같은 결과를 함께 쓴다(읽기만 함)
    results = [ep.result() if ep is not None else cast(EpisodeResult, failed_result) for ep in built]
    block_ids = [b.id for b in spec.file.scoring.blocks]
    trajectories: list[Trajectory | None] | None = None
    if recorders:
        trajectories = [recorders[i].build() if i in recorders else None for i in range(n)]
    if record is not None and failed and any(record[i] for i in failed):  # 요청한 조립 실패 쌍: 프레임 0개
        trajectories = trajectories or [None] * n
        for i in failed:
            if record[i]:
                trajectories[i] = _empty_trajectory(spec, max(1, int(every)))
    observations: list[PairObservation | None] | None = None
    if observers is not None:
        observations = [None if ob is None else ob.build() for ob in observers]
        codes = list(observe) if observe is not None else []
        for i in failed:
            if int(codes[i]) != OBSERVE_NONE:  # 요청한 조립 실패 쌍: 0을 센 관찰(세대 요약이 그 쌍을 세지 않음)
                observations[i] = _empty_observation(len(layout.output_ranges))
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
        trajectories=trajectories,
        nn_seconds=nn_t,
        sim_seconds=time.perf_counter() - t_start - nn_t,
        observations=observations,
    )

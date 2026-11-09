"""개체 묶음 평가 (워커 안에서 돈다).

맡은 (유전자, 시험) 쌍들의 Episode를 한 스텝씩 함께 진행하고, 신경망은 묶어서 한 번에 계산한다.
결과는 쌍마다 독립적이다: 묶음 구성이 달라도(워커 수, 분배, 평가 창 크기, 쌍 순서) 각 쌍의 결과는 비트 단위로 같다.
- Episode 결과는 (환경, 몸 수치, Trial)만의 함수
- BatchBrain.forward는 행마다 독립(묶음 크기, 다른 행과 무관)

평가 창(v6.0, v6 설계 4.7·D15): 살아 있는 Episode를 많아야 EVAL_WINDOW개만 두고, 끝난 자리에 다음 쌍을 쌍 순서대로 바로
채운다. 쌍이 많으면(bench 단일 코어 160쌍, 워커 1~2개, 큰 개체 수) 스텝마다 훑는 세계(파이썬 객체 + Chipmunk 공간)가
캐시(L2)를 넘어 한 스텝이 20~27% 비싸졌다(v2.0 열린 단서, 원인은 GC가 아니라 캐시). 끝난 쌍은 그 자리에서 결과·궤적·
관찰을 떼고 Episode를 놓으므로 observe·step을 더 받지 않고, 동시에 살아 있는 Episode도 창 크기 이하다. 행동은 묶음마다
ndarray.tolist() 한 번 뒤 쌍마다 Episode.step_values(목록)로 넘긴다(v6 런타임, 없으면 공개 step). 결과는 v3.1 출시판의
평가기(tests/oracle/evaluate_v31.py)와 바이트까지 같다(tests/test_eval_identity.py, test_eval_window.py).

관찰 기록(v1.2, 설계 D3~D5): observe[i]가 1·2인 쌍은 스텝마다 실제 구동기 출력을 20칸 히스토그램으로 세고(1·2),
2이면 4스텝마다 기준 물체 위치를 격자 틀의 칸으로 센다. 관찰기는 Episode의 읽기 전용 접근자(applied_outputs,
output_channel_index, anchor_body)와 pymunk Body.position만 읽고 난수를 쓰지 않는다. 그래서 점수·궤적은 관찰을
켜도 같고, 쌍별 관찰은 (환경, 유전자, 시험, 틀)만의 함수다.

조립 실패(v2.0, 설계 4.9·D22): 몸 수치 때문에 Episode를 만들 수 없는(EpisodeBuildError) 쌍은 학습을 멈추지 않고 그 쌍만
실패로 센다(점수 NaN, 결과 failure, 종료 사건 BUILD_FAILED_EVENT, 스텝·블록·행동 0, 궤적 프레임 0개, 관찰 0). 그 쌍은
창에 넣지 않는다(행마다 독립이라 다른 쌍의 결과는 그대로). 같은 시험을 처음 몸으로도 만들 수 없으면 몸이 아니라 환경의
문제라 v1.2처럼 오류를 올린다. 실패는 (환경, 몸 수치, 시험)의 순수 함수라 어느 워커에서나 같다.
"""

from __future__ import annotations

import math
import operator
import time
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, cast

import numpy as np

from evolab.policy.brain import BatchBrain, unpack
from evolab.policy.genome import GenomeLayout
from evolab.runs.observe import ACTION_BINS
from evolab.spec.compiled import EnvSpec
from evolab.world.physics2d.episode import Episode, EpisodeBuildError, EpisodeResult, Trial

EVAL_WINDOW: Final[int] = 12
"""평가 창: evaluate가 동시에 살려 두는 Episode 수의 상한(결과와 무관, v6 설계 D15). 리눅스 측정에서 창 8·16보다 대체로
빨랐다(설계 9.1-3). 사용자 PC의 bench 「평가 창」 줄에서 12가 가장 빠른 창보다 10% 넘게 느리면 그 값으로 바꾼다(되돌릴 지점 3)."""

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


def _check_body_failure(spec: EnvSpec, trial: Trial, error: EpisodeBuildError,
                        reference: dict[tuple[int, int | None], bool]) -> None:
    """조립 실패가 그 몸 수치 때문인지 본다(v2.0 설계 D22: 처음 몸은 조립되는데 특정 몸 수치로만 안 될 때만 그 쌍의 실패).

    같은 시험을 처음 몸(Episode(spec, None, 시험): 몸 수치 = 환경 파일의 처음 값)으로도 조립할 수 없으면 몸이 아니라
    환경·시험의 문제라 v1.2처럼 그 오류를 올린다(학습이 멈추고 워커 풀은 PoolError로 알린다). 실패가 있을 때만 시험마다
    한 번 더 만들어 본다(reference에 기억). 평가 창은 쌍 순서대로 만들므로 올리는 오류는 v3.1(모두 만든 뒤 번호 순서로
    처음 것)과 같은 쌍의 것이다.
    """
    key = (trial.seed, trial.track)
    ok = reference.get(key)
    if ok is None:
        ok = reference[key] = not isinstance(_build(spec, None, trial), EpisodeBuildError)
    if not ok:
        raise error


def evaluate(
    spec: EnvSpec, layout: GenomeLayout, genomes: np.ndarray, pairs: Sequence[tuple[int, Trial]],
    record: Sequence[bool] | None = None, every: int = 2, observe: Sequence[int] | None = None,
    frame: Sequence[float] | None = None, window: int | None = None,
) -> EvalResult:
    """genomes: (G, dim). pairs: (genomes의 행 번호, Trial) 목록.

    record[i]가 참인 쌍은 every 스텝마다 궤적을 남긴다. 궤적 기록은 상태를 읽기만 하므로 점수는 같다.
    observe[i]는 쌍마다 관찰 코드(OBSERVE_*). 코드 2이고 frame(x0, y0, 칸 크기, nx, ny)과 기준 물체가 있을 때만
    위치를 센다. 행동 칸은 layout.output_ranges를 나눈다. 관찰도 상태를 읽기만 하므로 점수는 같다.

    평가 창(v6 설계 4.7·D15): 살아 있는 Episode를 많아야 window개(None이면 EVAL_WINDOW)만 두고, 끝난 자리에 다음 쌍을
    쌍 순서대로 바로 채운다. 끝난 Episode는 그 자리에서 결과·궤적·관찰을 떼고 놓는다(observe·step을 더 받지 않음).
    결과(EvalResult의 모든 칸)는 window와 무관하고(쌍마다 독립, 신경망은 행마다 독립) nn_seconds·sim_seconds의 뜻은
    v3.1과 같다(세계 만들기도 시뮬레이션 시간).
    """
    width = EVAL_WINDOW if window is None else operator.index(window)
    if width < 1:
        raise ValueError(f"window must be >= 1, got {width}")
    genomes = np.ascontiguousarray(genomes, dtype=np.float64)
    n = len(pairs)
    every_n = max(1, int(every))
    body = layout.body_values(genomes) if layout.n_body else None
    all_brains = unpack(layout.brain, layout.brain_params(genomes))
    recording = [False] * n
    if record is not None:
        for i, on in enumerate(record):
            if on:
                recording[i] = True
    codes: list[int] | None = None
    if observe is not None and any(observe):
        codes = [int(c) for c in observe]
        if len(codes) != n:
            raise ValueError(f"observe has {len(codes)} codes for {n} pairs")
    ranges = layout.output_ranges

    results: list[EpisodeResult | None] = [None] * n
    trajectories: list[Trajectory | None] | None = [None] * n if any(recording) else None
    observations: list[PairObservation | None] | None = [None] * n if codes is not None else None
    failed_result: EpisodeResult | None = None  # 실패 쌍이 모두 같은 결과를 함께 쓴다(읽기만 함)
    reference: dict[tuple[int, int | None], bool] = {}

    # 슬롯: 살아 있는 쌍이 0..live-1 자리에 빈틈없이 있다. 신경망 가중치는 슬롯 자리의 연속 배열(그 쌍 유전자의 행을 복사).
    width = min(width, n)
    weights = [np.empty((width, fi, fo), dtype=np.float64) for fi, fo in layout.brain.shapes]
    biases = [np.empty((width, fo), dtype=np.float64) for _, fo in layout.brain.shapes]
    inputs = np.zeros((width, layout.brain.n_in), dtype=np.float64)
    pair_of: list[int] = []
    eps: list[Episode] = []
    steppers: list[Callable[[list[float]], None]] = []
    recorders: list[_Recorder | None] = []
    observers: list[_Observer | None] = []
    t_start = time.perf_counter()  # 세계 만들기도 시뮬레이션 시간에 넣는다
    nxt = 0

    def take() -> tuple[int, Episode] | None:
        """다음 쌍을 만든다(쌍 순서대로). 조립 실패 쌍은 그 자리에서 실패 결과로 두고 다음 쌍으로 간다."""
        nonlocal nxt, failed_result
        while nxt < n:
            i = nxt
            nxt += 1
            g, trial = pairs[i]
            made = _build(spec, body[g] if body is not None else None, trial)
            if not isinstance(made, EpisodeBuildError):
                return i, made
            _check_body_failure(spec, trial, made, reference)
            del made  # 예외의 traceback이 만들다 만 세계를 붙잡지 않게
            if failed_result is None:
                failed_result = build_failed_result(spec)
            results[i] = failed_result
            if recording[i]:  # 요청한 조립 실패 쌍: 프레임 0개
                cast("list[Trajectory | None]", trajectories)[i] = _empty_trajectory(spec, every_n)
            if codes is not None and codes[i] != OBSERVE_NONE:  # 0을 센 관찰(세대 요약이 그 쌍을 세지 않음)
                cast("list[PairObservation | None]", observations)[i] = _empty_observation(len(ranges))
        return None

    def place(s: int, i: int, ep: Episode) -> None:
        g = pairs[i][0]
        for w, src in zip(weights, all_brains.weights, strict=True):
            w[s] = src[g]
        for b, src in zip(biases, all_brains.biases, strict=True):
            b[s] = src[g]
        rec = None
        if recording[i]:
            rec = _Recorder(every_n, ep.dt)
            rec.add(ep)
        ob = None if codes is None else _make_observer(ep, ranges, codes[i], frame)
        stepper = getattr(ep, "step_values", None) or ep.step  # v6 런타임의 목록 길(없으면 공개 step)
        if s == len(eps):
            pair_of.append(i)
            eps.append(ep)
            steppers.append(stepper)
            recorders.append(rec)
            observers.append(ob)
        else:
            pair_of[s], eps[s], steppers[s], recorders[s], observers[s] = i, ep, stepper, rec, ob

    def finish(s: int) -> None:
        """끝난 슬롯의 결과를 떼어 둔다(끝난 Episode의 상태는 더 바뀌지 않으므로 v3.1이 끝에 모은 값과 같다)."""
        i = pair_of[s]
        results[i] = eps[s].result()
        rec = recorders[s]
        if rec is not None:
            cast("list[Trajectory | None]", trajectories)[i] = rec.build()
        ob = observers[s]
        if ob is not None:
            cast("list[PairObservation | None]", observations)[i] = ob.build()

    def brain_of(live: int) -> BatchBrain:
        """살아 있는 슬롯 0..live-1의 신경망(슬롯 배열의 보기: 다시 채운 자리의 가중치가 그대로 보인다)."""
        return BatchBrain(layout.brain, [w[:live] for w in weights], [b[:live] for b in biases])

    live = 0
    while live < width:
        got = take()
        if got is None:
            break
        place(live, *got)
        live += 1
    brain = brain_of(live)
    nn_t = 0.0
    sweeps = 0
    while live:
        for s in range(live):
            inputs[s] = eps[s].observe()
        t0 = time.perf_counter()
        actions = layout.to_actions(brain.forward(inputs[:live] if live < width else inputs))
        nn_t += time.perf_counter() - t0
        rows = actions.tolist()  # 묶음마다 한 번(ndarray.tolist()의 float는 행 tolist()와 같은 값)
        ended: list[int] | None = None
        for s in range(live):
            ep = eps[s]
            steppers[s](rows[s])
            rec = recorders[s]
            if rec is not None:
                rec.after_step(ep)
            ob = observers[s]
            if ob is not None:  # = ob.after_step(ep), 메서드 호출 없이(상태를 읽기만 한다)
                ob.acts.fromlist(ep.applied_outputs)
                if ob.body is not None and ep.steps % POSITION_EVERY == 0:
                    ob.pos.extend(ob.body.position)
            if ep.done:
                if ended is None:
                    ended = []
                ended.append(s)
        if codes is not None:
            sweeps += 1
            if sweeps % FLUSH_EVERY == 0:
                for ob in observers:
                    if ob is not None:
                        ob.flush()
        if ended is None:
            continue
        del ep, rec, ob  # 끝난 Episode를 루프 변수가 붙잡지 않게(놓은 뒤 바로 거둘 수 있게)
        holes: list[int] = []
        for s in ended:
            finish(s)
            got = take()
            if got is None:
                holes.append(s)
            else:
                place(s, *got)
        if not holes:
            continue
        # 남은 쌍이 없어 빈 슬롯: 살아 있는 슬롯을 앞으로 당긴다(순서는 결과와 무관, 행마다 독립)
        keep = [s for s in range(live) if s not in holes]
        for p, s in enumerate(keep):
            if p != s:
                for arr in (*weights, *biases):
                    arr[p] = arr[s]
                pair_of[p], eps[p], steppers[p] = pair_of[s], eps[s], steppers[s]
                recorders[p], observers[p] = recorders[s], observers[s]
        live = len(keep)
        del pair_of[live:], eps[live:], steppers[live:], recorders[live:], observers[live:]
        brain = brain_of(live)

    block_ids = [b.id for b in spec.file.scoring.blocks]
    done = cast("list[EpisodeResult]", results)
    return EvalResult(
        scores=np.array([r.score for r in done], dtype=np.float64),
        steps=np.array([r.steps for r in done], dtype=np.int64),
        end_events=[r.end_event for r in done],
        outcomes=[r.outcome for r in done],
        blocks=np.array([[r.blocks[b] for b in block_ids] for r in done], dtype=np.float64).reshape(n, len(block_ids)),
        action_abs=np.array([[r.action_mean_abs[c] for c in spec.outputs] for r in done], dtype=np.float64).reshape(
            n, len(spec.outputs)
        ),
        weighted=np.array([[r.weighted.get(b, np.nan) for b in block_ids] for r in done],
                          dtype=np.float64).reshape(n, len(block_ids)),
        trajectories=trajectories,
        nn_seconds=nn_t,
        sim_seconds=time.perf_counter() - t_start - nn_t,
        observations=observations,
    )

"""개체 묶음 평가 (워커 안에서 돈다).

맡은 (유전자, 시험) 쌍들의 Episode를 한 스텝씩 함께 진행하고, 신경망은 묶어서 한 번에 계산한다.
결과는 쌍마다 독립적이다: 묶음 구성이 달라도(워커 수, 분배) 각 쌍의 결과는 비트 단위로 같다.
- Episode 결과는 (환경, 몸 수치, Trial)만의 함수
- BatchBrain.forward는 행마다 독립(묶음 크기, 다른 행과 무관)
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from evolab.policy.brain import BatchBrain, unpack
from evolab.policy.genome import GenomeLayout
from evolab.spec.compiled import EnvSpec
from evolab.world.physics2d.episode import Episode, Trial


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
    record: Sequence[bool] | None = None, every: int = 2,
) -> EvalResult:
    """genomes: (G, dim). pairs: (genomes의 행 번호, Trial) 목록.

    record[i]가 참인 쌍은 every 스텝마다 궤적을 남긴다. 궤적 기록은 상태를 읽기만 하므로 점수는 같다.
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
    )

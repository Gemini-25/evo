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
    spec: EnvSpec, layout: GenomeLayout, genomes: np.ndarray, pairs: Sequence[tuple[int, Trial]]
) -> EvalResult:
    """genomes: (G, dim). pairs: (genomes의 행 번호, Trial) 목록."""
    genomes = np.ascontiguousarray(genomes, dtype=np.float64)
    n = len(pairs)
    body = layout.body_values(genomes) if layout.n_body else None
    all_brains = unpack(layout.brain, layout.brain_params(genomes))
    rows = np.array([g for g, _ in pairs], dtype=np.int64)
    episodes = [Episode(spec, body[g] if body is not None else None, trial) for g, trial in pairs]

    nn_t = 0.0
    t_start = time.perf_counter()
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
            ep.step(actions[j])
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
        nn_seconds=nn_t,
        sim_seconds=time.perf_counter() - t_start - nn_t,
    )

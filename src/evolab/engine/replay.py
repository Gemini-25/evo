"""화면 프로세스의 다시 시뮬레이션 (명세: 재생은 저장된 유전자와 시험 시드로 다시 돌린다).

타임라인 재생, "1세대 vs 지금" 겹쳐 보기, 개체 클릭 검사, 영상 내보내기가 쓴다. 워커와 같은 Episode와 같은
순전파(BatchBrain.forward, 행마다 독립)를 쓰므로 점수가 워커 점수와 비트 단위로 같다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evolab.engine.evaluate import Trajectory, _Recorder
from evolab.policy.brain import unpack
from evolab.policy.genome import GenomeLayout
from evolab.spec.compiled import EnvSpec
from evolab.world.physics2d.episode import BodyGeom, Episode, EpisodeResult, Trial, WorldGeometry


@dataclass
class Replay:
    trajectory: Trajectory
    result: EpisodeResult
    geometry: WorldGeometry
    agent: list[BodyGeom]
    """움직이는 물체의 모양(Episode.moving_bodies 순서)."""
    channels: list[str]
    """구동기 출력 채널 이름(궤적 outputs의 열 순서)."""
    inputs: list[str]
    observations: np.ndarray | None = None
    """trace일 때 (스텝, 입력 수) float32: 스텝마다 신경망에 들어간 값(정규화 후)."""
    activations: list[np.ndarray] | None = None
    """trace일 때 층마다 (스텝, 폭) float32."""
    raw: list[dict[str, float]] | None = None
    """trace일 때 스텝마다 센서 원래 값(정규화 전)."""


def agent_geoms(geometry: WorldGeometry, moving: list[str]) -> list[BodyGeom]:
    by_id = {b.id: b for b in geometry.bodies}
    return [by_id[i] for i in moving]


def replay(spec: EnvSpec, layout: GenomeLayout, genome: np.ndarray, trial: Trial, every: int = 2,
           trace: bool = False) -> Replay:
    """유전자 하나를 시험 하나로 끝까지 돌린다. trace면 스텝마다 입력·활성화·센서 원래 값을 남긴다."""
    g = np.ascontiguousarray(np.asarray(genome, dtype=np.float64).reshape(1, -1))
    body = layout.body_values(g)[0] if layout.n_body else None
    brain = unpack(layout.brain, layout.brain_params(g))
    ep = Episode(spec, body, trial)
    rec = _Recorder(max(1, int(every)), ep.dt)
    rec.add(ep)
    geometry = ep.geometry()
    obs_log: list[np.ndarray] = []
    act_log: list[list[np.ndarray]] = []
    raw_log: list[dict[str, float]] = []
    obs = np.zeros((1, layout.brain.n_in), dtype=np.float64)
    while not ep.done:
        obs[0] = ep.observe()
        layers: list[np.ndarray] | None = [] if trace else None
        out = brain.forward(obs, layers)
        if trace:
            obs_log.append(obs[0].astype(np.float32))
            act_log.append([a[0].astype(np.float32) for a in layers or []])
            raw_log.append(ep.channel_values())
        ep.step(layout.to_actions(out)[0])
        rec.after_step(ep)
    observations = activations = raw = None
    if trace:
        n_in = layout.brain.n_in
        observations = np.stack(obs_log) if obs_log else np.zeros((0, n_in), dtype=np.float32)
        widths = [fo for _, fo in layout.brain.shapes]
        activations = [np.stack([step[k] for step in act_log]) if act_log else np.zeros((0, w), dtype=np.float32)
                       for k, w in enumerate(widths)]
        raw = raw_log
    return Replay(rec.build(), ep.result(), geometry, agent_geoms(geometry, ep.moving_bodies),
                  list(ep.actuator_channels()), list(spec.inputs), observations, activations, raw)

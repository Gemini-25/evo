"""화면 프로세스의 다시 시뮬레이션 (명세: 재생은 저장된 유전자와 시험 시드로 다시 돌린다).

타임라인 재생, "1세대 vs 지금" 겹쳐 보기, 개체 클릭 검사, 영상 내보내기, 뇌 들여다보기가 쓴다. 워커와 같은
Episode와 같은 순전파(BatchBrain.forward, 행마다 독립)를 쓰므로 점수가 워커 점수와 비트 단위로 같다.

추적(trace)은 시뮬레이션을 바꾸지 않고 읽기만 더한다: True면 입력·층별 활성화·센서 원래 값·레이를, "brain"(v1.2
뇌 들여다보기의 가벼운 추적, 설계 D25)이면 입력과 층별 활성화만 남긴다(channel_values()·ray_view()를 부르지 않음).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    """trace(True 또는 "brain")일 때 (스텝, 입력 수) float32: 스텝마다 신경망에 들어간 값(정규화 후)."""
    activations: list[np.ndarray] | None = None
    """trace(True 또는 "brain")일 때 층마다 (스텝, 폭) float32."""
    raw: list[dict[str, float]] | None = None
    """trace=True일 때 스텝마다 센서 원래 값(정규화 전). "brain"이면 None."""
    rays: np.ndarray | None = None
    """trace=True이고 레이 센서가 있으면 (스텝, 레이 수, 5) float32: 시작 x, y, 끝 x, y, 맞은 비율. "brain"이면 None."""


def agent_geoms(geometry: WorldGeometry, moving: list[str]) -> list[BodyGeom]:
    by_id = {b.id: b for b in geometry.bodies}
    return [by_id[i] for i in moving]


TraceMode = bool | Literal["brain"]
TRACE_BRAIN: Literal["brain"] = "brain"


def replay(spec: EnvSpec, layout: GenomeLayout, genome: np.ndarray, trial: Trial, every: int = 2,
           trace: TraceMode = False) -> Replay:
    """유전자 하나를 시험 하나로 끝까지 돌린다.

    trace=True면 스텝마다 입력·층별 활성화·센서 원래 값·레이를 남기고, trace="brain"이면 입력·층별 활성화만 남긴다
    (뇌 그림용 가벼운 추적). 어느 쪽이든 시뮬레이션 경로와 결과(궤적, 점수)는 trace=False와 같다.
    """
    if isinstance(trace, str) and trace != TRACE_BRAIN:
        raise ValueError(f"trace must be True, False or {TRACE_BRAIN!r}, got {trace!r}")
    neurons = bool(trace)  # 입력·층별 활성화
    full = neurons and not isinstance(trace, str)  # 센서 원래 값·레이까지
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
    ray_log: list[list[tuple[float, float, float, float, float]]] = []
    obs = np.zeros((1, layout.brain.n_in), dtype=np.float64)
    while not ep.done:
        o = ep.observe()  # 새 float32 배열(스텝마다 새로 만든다): 추적은 그대로 담고 끝에 한 번에 쌓는다
        obs[0] = o
        layers: list[np.ndarray] | None = [] if neurons else None
        out = brain.forward(obs, layers)  # layers: 층마다 (1, 폭) float64 사본
        if neurons:
            obs_log.append(o)
            act_log.append(layers or [])
        if full:
            raw_log.append(ep.channel_values())
            view = ep.ray_view()
            if view:
                ray_log.append([(a[0], a[1], b[0], b[1], al) for a, b, al in view])
        ep.step(layout.to_actions(out)[0])
        rec.after_step(ep)
    observations = activations = raw = None
    if neurons:
        n_in = layout.brain.n_in
        observations = np.stack(obs_log).astype(np.float32, copy=False) if obs_log else \
            np.zeros((0, n_in), dtype=np.float32)
        widths = [fo for _, fo in layout.brain.shapes]
        activations = [np.concatenate([step[k] for step in act_log]).astype(np.float32) if act_log
                       else np.zeros((0, w), dtype=np.float32) for k, w in enumerate(widths)]
    if full:
        raw = raw_log
    rays = np.asarray(ray_log, dtype=np.float32) if ray_log else None
    return Replay(rec.build(), ep.result(), geometry, agent_geoms(geometry, ep.moving_bodies),
                  list(ep.actuator_channels()), list(spec.inputs), observations, activations, raw, rays)

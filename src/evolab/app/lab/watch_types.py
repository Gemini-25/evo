"""실험실 「보는 재미」의 화면 쪽 계약 (v1.2 설계 7.1·7.3): 대시보드(app/lab/dashboard.py, watch_tiles.py)와
실험실·뇌 그림(app/lab/view.py, brain_view.py)이 주고받는 값.

- WatchMeta: 실행(환경 + 유전자 구조) 하나의 이름표. 실험실이 실행을 열거나 학습을 시작할 때 한 번 만들어
  Dashboard.set_watch_meta로 넘기고, 타일(몸 수치·행동 분포·뇌 그림)과 기록 읽기(runs.observe.decode)가 읽는다.
- BrainSource: 뇌 들여다보기 타일이 그릴 개체 하나(설계 D24의 원천 셋). Dashboard.set_brain_source로 넘긴다.

Qt 위젯이 없는 얼린 dataclass라 배경 스레드에서 만들어도 된다. 이름표의 글은 strings.py의 규칙(차량 구동기는
ACT_CHANNEL_NAME)으로 여기서 한 번만 정한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, NamedTuple

import numpy as np

from evolab.app import strings as S
from evolab.policy.genome import GenomeLayout, layout_for
from evolab.spec import model as m
from evolab.spec import symbols as sy

if TYPE_CHECKING:
    from evolab.engine.replay import Replay
    from evolab.spec.compiled import EnvSpec


class BodyParam(NamedTuple):
    """진화하는 몸 수치 하나(유전자 순서 = GenomeLayout.body 순서 = obs_body의 열 순서)."""

    id: str
    name: str
    """표시 이름: 환경 파일 매개변수 label(없으면 id)."""
    low: float
    """진화 범위(물리 단위)."""
    high: float


class OutputChannel(NamedTuple):
    """신경망 출력 하나(순서 = spec.outputs = obs_act_hist·obs_act_mean의 U 축 순서)."""

    name: str
    """채널 이름(spec.outputs의 글, 예: main, car.throttle)."""
    label: str
    """표시 이름: 구동기 label(없으면 id). 차량은 ACT_CHANNEL_NAME(「가속」·「조향」)."""
    low: float
    """구동기 고유 범위: (0, 1) 또는 (-1, 1). 행동 분포 20칸은 이 범위를 똑같이 나눈다."""
    high: float


class NodeName(NamedTuple):
    """뇌 그림의 입력·출력 뉴런 이름표: (채널 이름, 풍선 글에 쓸 센서·구동기 표시 이름)."""

    name: str
    label: str


@dataclass(frozen=True)
class WatchMeta:
    """실행 하나의 이름표. 유전자 구조는 실행 내내 같다(환경을 바꿔도 서명이 같아야 이어 감, A8)."""

    layout: GenomeLayout
    body: tuple[BodyParam, ...]
    """layout.body 순서. 몸 유전자가 없으면 빈 튜플(몸 수치 타일은 BODY_EMPTY_NONE)."""
    channels: tuple[OutputChannel, ...]
    """신경망 출력 순서, 길이 = layout.brain.n_out."""
    inputs: tuple[NodeName, ...]
    """신경망 입력 순서, 길이 = layout.brain.n_in."""
    outputs: tuple[NodeName, ...]
    """신경망 출력 순서(channels의 name, label과 같은 값)."""
    env_name: str
    """환경 이름(환경 파일 name)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "body", tuple(BodyParam(*b) for b in self.body))
        object.__setattr__(self, "channels", tuple(OutputChannel(*c) for c in self.channels))
        object.__setattr__(self, "inputs", tuple(NodeName(*n) for n in self.inputs))
        object.__setattr__(self, "outputs", tuple(NodeName(*n) for n in self.outputs))
        lay = self.layout
        for what, got, want in (("body", len(self.body), lay.n_body), ("channels", len(self.channels), lay.brain.n_out),
                                ("inputs", len(self.inputs), lay.brain.n_in),
                                ("outputs", len(self.outputs), lay.brain.n_out)):
            if got != want:
                raise ValueError(f"WatchMeta.{what}: {got} entries for a layout with {want}")
        ids = tuple(g.id for g in lay.body)
        if tuple(b.id for b in self.body) != ids:
            raise ValueError(f"WatchMeta.body ids {[b.id for b in self.body]} do not match the layout {list(ids)}")

    @property
    def dim(self) -> int:
        return self.layout.dim

    @property
    def n_body(self) -> int:
        return self.layout.n_body

    @property
    def n_out(self) -> int:
        return self.layout.brain.n_out

    def decode_args(self) -> dict[str, int]:
        """runs.observe.decode(arrays, **meta.decode_args())의 유전자 구조 인수."""
        return {"dim": self.dim, "n_body": self.n_body, "n_out": self.n_out}

    @classmethod
    def from_spec(cls, spec: EnvSpec, layout: GenomeLayout | None = None) -> WatchMeta:
        """환경(EnvSpec)에서 만든다. layout을 주지 않으면 layout_for(spec)."""
        lay = layout if layout is not None else layout_for(spec)
        env = spec.file
        params = {p.id: p for p in sy.evolved_params(env)}
        body = tuple(BodyParam(g.id, (params[g.id].label if g.id in params else "") or g.id, g.low, g.high)
                     for g in lay.body)
        actuators = {a.id: a for a in env.actuators}
        sensors = {s.id: s for s in env.sensors}
        channels: list[OutputChannel] = []
        for name in spec.outputs:
            ch = spec.actuator_channels[name]
            act = actuators.get(ch.owner)
            label = (act.label if act is not None else "") or ch.owner
            if isinstance(act, m.Vehicle):
                part = name.rsplit(".", 1)[-1]
                label = S.ACT_CHANNEL_NAME.format(label=label, suffix=S.ACT_CHANNEL_SUFFIX.get(part, part))
            channels.append(OutputChannel(name, label, float(ch.low), float(ch.high)))
        inputs: list[NodeName] = []
        for name in spec.inputs:
            ch = spec.sensor_channels.get(name)
            owner = ch.owner if ch is not None else name
            sensor = sensors.get(owner)
            inputs.append(NodeName(name, (sensor.label if sensor is not None else "") or owner))
        outputs = tuple(NodeName(c.name, c.label) for c in channels)
        return cls(lay, body, tuple(channels), tuple(inputs), outputs, env.name)


BrainKind = Literal["playing", "live", "champion"]
BRAIN_KINDS: Final = ("playing", "live", "champion")
"""뇌 원천(D24): playing = 캔버스가 추적 재생 중인 개체(타임라인, 1세대 vs 지금의 "지금", 고속 챔피언, 클릭 검사,
지도 점), live = 관전 중 그 세대 대표를 시험 0번으로 따로 다시 돌린 것, champion = 보는 세대까지의 챔피언 멈춘 그림."""


@dataclass(frozen=True, eq=False)
class BrainSource:
    """뇌 들여다보기 타일이 그릴 개체 하나. 유전자는 읽기 전용 복사본(float64 1차원)으로 갖는다."""

    label: str
    """제목에 쓸 이름. playing이면 재생 이름(BRAIN_TITLE_PLAYING의 label: LAB_LABEL_* 글), live·champion은 타일이
    generation으로 제목(BRAIN_TITLE_LIVE, BRAIN_TITLE_CHAMPION)을 만들므로 비워 둬도 된다."""
    kind: BrainKind
    generation: int | None
    """그 개체의 세대(모르면 None)."""
    genome: np.ndarray
    """(dim,) float64. 연결선 그림은 이 값(layout.brain_params)으로 만든다."""
    replay: Replay | None = None
    """추적 재생 결과(replay(trace=True 또는 "brain")의 observations, activations). champion이거나 아직 계산 중이면
    None(BRAIN_PREPARING은 playing·live인데 None일 때)."""

    def __post_init__(self) -> None:
        if self.kind not in BRAIN_KINDS:
            raise ValueError(f"BrainSource.kind must be one of {BRAIN_KINDS}, got {self.kind!r}")
        g = np.array(self.genome, dtype=np.float64).reshape(-1)  # 복사: 원본이 바뀌어도 그림이 그대로
        g.flags.writeable = False
        object.__setattr__(self, "genome", g)
        if self.generation is not None:
            object.__setattr__(self, "generation", int(self.generation))

    @property
    def traced(self) -> bool:
        """뉴런 값을 칠할 추적이 있는가(입력·층별 활성화)."""
        return self.replay is not None and self.replay.observations is not None and self.replay.activations is not None

    def same_genome(self, other: BrainSource | None) -> bool:
        """유전자가 비트 단위로 같은가(같으면 연결선 그림을 다시 만들지 않는다)."""
        return other is not None and self.genome.shape == other.genome.shape and \
            self.genome.tobytes() == other.genome.tobytes()

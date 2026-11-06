"""직접 조종 입력 기록. 같은 환경, 같은 시드, 같은 입력이면 결과가 비트 단위로 같아야 한다.

Qt를 쓰지 않는다(테스트와 명령줄에서 재생 확인용).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evolab.spec.compiled import EnvSpec
from evolab.spec.io import env_hash
from evolab.world.physics2d.episode import Episode, EpisodeResult, Trial

FORMAT = "evolab-play-record"
FORMAT_VERSION = 1


@dataclass
class PlayRecord:
    env_name: str
    env_hash: str
    seed: int
    outputs: list[str]
    actions: list[list[float]]
    """스텝마다 brain.outputs 순서의 구동기 출력."""
    score: float | None = None
    outcome: str | None = None
    steps: int | None = None

    def to_json(self) -> str:
        data: dict[str, Any] = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "env_name": self.env_name,
            "env_hash": self.env_hash,
            "seed": self.seed,
            "outputs": self.outputs,
            "result": {"score": self.score, "outcome": self.outcome, "steps": self.steps},
            "actions": self.actions,
        }
        return json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"

    @classmethod
    def from_json(cls, text: str) -> PlayRecord:
        data = json.loads(text)
        if data.get("format") != FORMAT:
            raise ValueError(FORMAT)
        result = data.get("result") or {}
        return cls(
            env_name=str(data["env_name"]),
            env_hash=str(data["env_hash"]),
            seed=int(data["seed"]),
            outputs=[str(x) for x in data["outputs"]],
            actions=[[float(v) for v in row] for row in data["actions"]],
            score=result.get("score"),
            outcome=result.get("outcome"),
            steps=result.get("steps"),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PlayRecord:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def new_record(spec: EnvSpec, seed: int) -> PlayRecord:
    return PlayRecord(spec.file.name, env_hash(spec.file), seed, list(spec.outputs), [])


def replay(spec: EnvSpec, record: PlayRecord) -> EpisodeResult:
    """기록된 입력으로 화면 없이 다시 돌린다. 기록이 끝나도 안 끝났으면 출력 0으로 끝까지 돌린다."""
    episode = Episode(spec, None, Trial(record.seed))
    zero = np.zeros(len(spec.outputs))
    for row in record.actions:
        if episode.done:
            break
        episode.step(row)
    while not episode.done:
        episode.step(zero)
    return episode.result()


def verify_record(spec: EnvSpec, path: str | Path) -> tuple[bool, EpisodeResult, PlayRecord]:
    """화면 없이 기록을 다시 돌려 기록된 결과(점수와 스텝 수)와 비트 단위로 비교한다."""
    rec = PlayRecord.load(path)
    res = replay(spec, rec)
    same = rec.score is not None and res.score == rec.score and res.steps == rec.steps
    return same, res, rec

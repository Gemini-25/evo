"""직접 조종 입력 기록. 같은 환경, 같은 시드, 같은 입력이면 결과가 비트 단위로 같아야 한다.

Qt를 쓰지 않는다(테스트와 명령줄에서 재생 확인용).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evolab.app import strings as S
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
        """형식이 틀리면 한국어 설명이 담긴 ValueError."""
        try:
            data = json.loads(text)
            if not isinstance(data, dict) or data.get("format") != FORMAT:
                raise ValueError(S.PLAY_RECORD_NOT_RECORD)
            result = data.get("result") or {}
            outputs = [str(x) for x in data["outputs"]]
            actions = [[float(v) for v in row] for row in data["actions"]]
            if any(len(row) != len(outputs) for row in actions):
                raise ValueError(S.PLAY_RECORD_ROW_SIZE)
            score = result.get("score")
            steps = result.get("steps")
            return cls(
                env_name=str(data["env_name"]),
                env_hash=str(data["env_hash"]),
                seed=int(data["seed"]),
                outputs=outputs,
                actions=actions,
                score=float(score) if score is not None else None,
                outcome=result.get("outcome"),
                steps=int(steps) if steps is not None else None,
            )
        except ValueError:
            raise
        except (TypeError, KeyError, AttributeError, OverflowError) as exc:
            raise ValueError(S.PLAY_RECORD_BROKEN.format(detail=type(exc).__name__)) from None

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PlayRecord:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def new_record(spec: EnvSpec, seed: int) -> PlayRecord:
    return PlayRecord(spec.file.name, env_hash(spec.file), seed, list(spec.outputs), [])


def align_record(spec: EnvSpec, record: PlayRecord) -> PlayRecord:
    """기록의 출력 채널을 이 환경의 출력 순서로 맞춘다. 채널 이름이 다르면 ValueError."""
    if record.outputs == spec.outputs:
        return record
    if sorted(record.outputs) != sorted(spec.outputs):
        raise ValueError(S.PLAY_REPLAY_OUTPUTS_DIFF.format(recorded=", ".join(record.outputs), now=", ".join(spec.outputs)))
    index = [record.outputs.index(n) for n in spec.outputs]
    return PlayRecord(
        record.env_name, record.env_hash, record.seed, list(spec.outputs),
        [[row[i] for i in index] for row in record.actions], record.score, record.outcome, record.steps,
    )


def replay(spec: EnvSpec, record: PlayRecord) -> EpisodeResult:
    """기록된 입력으로 화면 없이 다시 돌린다. 기록이 끝나도 안 끝났으면 출력 0으로 끝까지 돌린다."""
    record = align_record(spec, record)
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
    """화면 없이 기록을 다시 돌려 기록된 결과(점수와 스텝 수)와 비트 단위로 비교한다.

    파일을 읽을 수 없거나, 형식이 틀리거나, 결과가 없는 기록(끝나기 전에 저장)이면 ValueError 또는 OSError.
    """
    rec = PlayRecord.load(path)
    if rec.score is None or rec.steps is None:
        raise ValueError(S.PLAY_RECORD_NO_RESULT)
    res = replay(spec, rec)
    same = res.score == rec.score and res.steps == rec.steps
    return same, res, rec


RECORD_PATTERNS = ("*.evoplay", "*.json")
"""기록 파일 이름. 화면은 .evoplay로 저장하고, 예전 기록이나 손으로 옮긴 기록은 .json일 수 있다."""


def best_record_score(spec: EnvSpec, folders: list[Path]) -> float | None:
    """여러 폴더의 기록 가운데 같은 환경(내용 해시)의 최고 점수. 읽을 수 없는 파일은 건너뛴다."""
    want = env_hash(spec.file)
    best: float | None = None
    seen: set[Path] = set()
    for folder in folders:
        for pattern in RECORD_PATTERNS:
            try:
                files = sorted(folder.glob(pattern))
            except OSError:
                continue
            for p in files:
                if p in seen:
                    continue
                seen.add(p)
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not isinstance(data, dict) or data.get("env_hash") != want:
                    continue
                result = data.get("result")
                score = result.get("score") if isinstance(result, dict) else None
                if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
                    best = float(score) if best is None else max(best, float(score))
    return best

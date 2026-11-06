"""M1: 망가진 환경 파일에 대한 견고성.

숫자 칸을 무작위로 바꿔도 결과는 둘 중 하나여야 한다: 불러올 때 한국어 오류(EnvFileError),
조립할 때 오류(EpisodeBuildError), 또는 예외 없이 끝까지 돈다. NaN이 관측값으로 새어 나오면 안 된다.
"""

import copy
import json
from typing import Any

import numpy as np
import pytest

from evolab import paths
from evolab.spec.compiled import EnvFileError
from evolab.spec.io import compile_env
from evolab.world.physics2d.episode import Episode, EpisodeBuildError, Trial

BASE = json.loads((paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8"))
WEIRD = [0.0, -1.0, 1e-9, 1e9, -1e9, 0.5, 3.0, 1e300]


def numeric_paths(node: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out += numeric_paths(v, (*path, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += numeric_paths(v, (*path, i))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out.append(path)
    return out


PATHS = [p for p in numeric_paths(BASE) if p[0] != "version"]


def set_at(data: Any, path: tuple[Any, ...], value: Any) -> None:
    for k in path[:-1]:
        data = data[k]
    data[path[-1]] = value


@pytest.mark.parametrize("case", range(150))
def test_mutated_lander_never_crashes(case: int) -> None:
    rng = np.random.default_rng(case)
    data = copy.deepcopy(BASE)
    for _ in range(int(rng.integers(1, 4))):
        path = PATHS[int(rng.integers(len(PATHS)))]
        value = WEIRD[int(rng.integers(len(WEIRD)))]
        set_at(data, path, value)
    try:
        spec = compile_env(data)
    except EnvFileError as exc:
        assert exc.issues and all(i.message for i in exc.issues)
        return
    try:
        ep = Episode(spec, None, Trial(case))
    except EpisodeBuildError:
        return
    for _ in range(120):
        if ep.done:
            break
        obs = ep.observe()
        assert np.all(np.isfinite(obs))
        ep.step(rng.random(len(spec.outputs)).tolist())
    res = ep.result()
    assert np.isfinite(res.score)

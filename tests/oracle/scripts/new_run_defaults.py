"""얼린 나무·지금 나무에서 함께 도는 스크립트: 새 실행의 설정 사전(TrainConfig.for_new_run)을 내장 환경마다 적는다.

    python new_run_defaults.py <결과.npz> <내장 환경 줄기 이름>…

결과 "configs" = JSON 글 {"": 환경 없이 for_new_run(), "<줄기>": for_new_run(spec).to_dict(), …}(적은 차례 그대로).
tests/test_foundation_v62.py(T7의 바탕 몫)가 쓴다: editor.train이 없는 옛 환경은 얼린 v6.1.0과 같은 사전이다.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from evolab import paths
from evolab.engine.trainer import TrainConfig
from evolab.spec.io import load_env


def main(out: str, *stems: str) -> None:
    files = {p.stem: p for p in paths.builtin_envs()}
    configs = {"": TrainConfig.for_new_run().to_dict()}
    for stem in stems:
        configs[stem] = TrainConfig.for_new_run(load_env(files[stem])).to_dict()
    np.savez(out, configs=np.array(json.dumps(configs, ensure_ascii=False)))


if __name__ == "__main__":
    main(*sys.argv[1:])

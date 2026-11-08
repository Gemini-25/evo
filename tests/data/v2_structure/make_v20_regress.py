"""v2.1 A' 묶음: v2.0 예제(점프 게임, 글라이더) 결과 참조 자료를 v2.0 코드(커밋 4d3e3e6)로 만든다(설계 T62).

이 스크립트는 저장소의 지금 코드가 아니라 v2.0 코드 사본으로 돌린다(그래야 "v2.0 런타임의 결과"가 된다):

    git archive 4d3e3e6 | tar -x -C <사본 폴더>
    PYTHONPATH=<사본 폴더>/src .venv/bin/python tests/data/v2_structure/make_v20_regress.py <사본 폴더> tests/data/v2_structure

만드는 것: v20_regress.npz. 점프 게임·글라이더에서 무작위 유전자 6개 x 시험 2개의 점수, 스텝, 종료 사건, 결과, 블록 값,
구동기 출력 평균, 매 스텝 궤적(자세·실제 출력, float32)의 sha256과 끝 자세. 유전자 = layout.initial() + N(0, 0.5),
np.random.default_rng(20261108)에서 환경 순서(jump, glider)대로 뽑는다. 시험 = Trial(seeds.derive(9, k)), k = 0, 1
(tests/data/make_v1_reference.py의 T29 자료와 같은 시험 공간). 환경 글과 유전자도 함께 담는다. 숫자 비교는 자료를 만든
플랫폼에서만 한다(npz의 platform_info: runsetup.platform_info의 python, numpy, pymunk, cma, cpu, T29와 같은 규칙).
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

PLATFORM_KEYS = ("python", "numpy", "pymunk", "cma", "cpu")
NAMES = ("jump", "glider")


def main() -> None:
    base = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()

    import evolab

    assert Path(evolab.__file__).resolve().is_relative_to(base), evolab.__file__
    from evolab.engine import seeds
    from evolab.engine.evaluate import evaluate
    from evolab.engine.runsetup import platform_info
    from evolab.policy.genome import layout_for
    from evolab.spec import model as m
    from evolab.spec.io import compile_env
    from evolab.world.physics2d.episode import Trial

    assert m.CURRENT_VERSION == 2
    rng = np.random.default_rng(20261108)
    trials = [Trial(seeds.derive(9, k)) for k in range(2)]
    arrays: dict[str, np.ndarray] = {
        "platform": np.array(f"{sys.platform}-{platform.machine()}"),
        "platform_info": np.array(json.dumps({k: platform_info(0)[k] for k in PLATFORM_KEYS}, sort_keys=True)),
        "trial_seeds": np.array([t.seed for t in trials], dtype=np.uint64),
    }
    for name in NAMES:
        text = (base / "environments" / f"{name}.evoenv").read_text(encoding="utf-8")
        spec = compile_env(text)
        layout = layout_for(spec)
        genomes = layout.initial()[None, :] + rng.normal(0.0, 0.5, (6, layout.dim))
        pairs = [(g, t) for g in range(len(genomes)) for t in trials]
        res = evaluate(spec, layout, genomes, pairs, record=[True] * len(pairs), every=1)
        assert res.trajectories is not None
        pose_sha = np.array([np.frombuffer(hashlib.sha256(tr.poses.tobytes()).digest(), np.uint8)
                             for tr in res.trajectories])
        out_sha = np.array([np.frombuffer(hashlib.sha256(tr.outputs.tobytes()).digest(), np.uint8)
                            for tr in res.trajectories])
        arrays.update({
            f"{name}.env_text": np.array(text),
            f"{name}.genomes": genomes,
            f"{name}.scores": res.scores,
            f"{name}.steps": res.steps,
            f"{name}.end_events": np.array(res.end_events),
            f"{name}.outcomes": np.array(res.outcomes),
            f"{name}.blocks": res.blocks,
            f"{name}.action_abs": res.action_abs,
            f"{name}.frames": np.array([tr.frames for tr in res.trajectories], dtype=np.int64),
            f"{name}.poses_sha256": pose_sha,
            f"{name}.outputs_sha256": out_sha,
            f"{name}.final_poses": np.concatenate([tr.poses[-1] for tr in res.trajectories]),
        })
        print(name, "steps", int(res.steps.sum()), "score mean", float(res.scores.mean()))
    np.savez_compressed(out_dir / "v20_regress.npz", **arrays)
    digest = hashlib.sha256()
    for k in sorted(arrays):
        digest.update(k.encode())
        digest.update(np.ascontiguousarray(arrays[k]).tobytes())
    print("v20_regress digest", digest.hexdigest())


if __name__ == "__main__":
    main()

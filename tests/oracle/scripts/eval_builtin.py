"""얼린 나무·지금 나무에서 함께 도는 견본 스크립트: 내장 환경 하나를 정해진 유전자로 평가한다(tests/oracle/frozen.py의 약속).

    python eval_builtin.py <결과.npz> <내장 환경 줄기 이름, 예 lander> [유전자 수 = 2] [시험 수 = 2]

유전자는 시드를 준 생성기로 만든 작은 값(0.3 × 표준 정규)이고 시험은 검증 시험의 앞쪽이다. 결과: 프로그램 버전, 환경 해시·형식,
유전자 차원, 평가 결과의 칸(scores·steps·blocks·action_abs·weighted), 종료 사건·결과 글, 0번 쌍의 궤적.
tests/test_frozen_oracle.py(T0)가 쓰고, 묶음 R의 넓은 비교(T3)가 본보기로 삼는다.
"""

from __future__ import annotations

import json
import sys

import numpy as np

import evolab
from evolab import paths
from evolab.engine.evaluate import evaluate
from evolab.engine.seeds import validation_trials
from evolab.policy.genome import layout_for
from evolab.spec.io import canonical_dict, env_hash, load_env


def main(out: str, stem: str, n_genomes: str = "2", n_trials: str = "2") -> None:
    files = {p.stem: p for p in paths.builtin_envs()}
    spec = load_env(files[stem])
    layout = layout_for(spec)
    rng = np.random.default_rng([62, int(n_genomes), layout.dim])
    genomes = 0.3 * rng.standard_normal((int(n_genomes), layout.dim))
    trials = validation_trials(int(n_trials))
    pairs = [(g, t) for g in range(int(n_genomes)) for t in trials]
    res = evaluate(spec, layout, genomes, pairs, record=[i == 0 for i in range(len(pairs))])
    traj = res.trajectories[0]
    np.savez(
        out,
        version=np.array(evolab.__version__),
        builtin=np.array(sorted(files)),
        env_hash=np.array(env_hash(spec.file)),
        env_format=np.array(int(canonical_dict(spec.file)["version"])),
        dim=np.array(layout.dim),
        genomes=genomes,
        scores=res.scores, steps=res.steps, blocks=res.blocks, action_abs=res.action_abs, weighted=res.weighted,
        ends=np.array(json.dumps([list(res.end_events), list(res.outcomes)], ensure_ascii=False)),
        poses=traj.poses, outputs=traj.outputs,
    )


if __name__ == "__main__":
    main(*sys.argv[1:])

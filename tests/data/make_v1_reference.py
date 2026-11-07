"""v2.0 0단계: v1 참조 자료를 v2 이전 코드(v1.2, 커밋 3dbcb95)로 만든다.

이 스크립트는 저장소의 지금 코드가 아니라 v1.2 코드 사본으로 돌린다(그래야 "옛 프로그램이 쓴 글"이 된다):

    git archive 3dbcb95 | tar -x -C <사본 폴더>
    PYTHONPATH=<사본 폴더>/src .venv/bin/python tests/data/make_v1_reference.py <사본 폴더> tests/data

만드는 것(모두 tests/data 아래, 테스트는 읽기만 한다):

- schema_fields_v1.json: v1 환경 파일 모델의 칸 목록("모델.칸", 칸은 파일에 쓰는 이름이라 별칭 with 포함).
  EnvFile에서 시작해 칸의 형(합집합, 목록, 튜플, 사전, Annotated)을 따라 닿는 모든 모델(T5).
- v1_dump/<이름>.json: 저장소 environments/*.evoenv와 tests/data 실행 폴더의 환경 사본(envs/v000.evoenv)을
  v1.2 코드의 dump_env로 쓴 글(T2). v1_dump/env_hash.txt: 같은 파일의 env_hash(T1).
- v1_regress.npz: 착륙선·트랙 자동차·언덕 넘는 자동차에서 무작위 유전자 8개 x 시험 3개의 점수, 스텝, 종료 사건,
  결과, 블록 값, 구동기 출력 평균, 매 스텝 궤적(자세·실제 출력, float32)의 sha256과 끝 자세(T29).
  유전자 = layout.initial() + N(0, 0.5), np.random.default_rng(20261107)에서 환경 순서(lander, track_car, hill_car)대로
  뽑는다. 시험 = Trial(seeds.derive(9, k)), k = 0, 1, 2(태그 9는 학습·검증·비교·판정·학습 방법 어느 공간에도 없음).
  유전자와 환경 글도 함께 담아 테스트가 같은 입력을 쓴다. 숫자 비교는 자료를 만든 플랫폼에서만 한다
  (명세 "정답 해시를 박지 않음"): 만든 플랫폼은 npz의 "platform" 칸.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import typing
from pathlib import Path

import numpy as np


def model_field_names(root: type) -> list[str]:
    """root 모델에서 닿는 모든 모델의 칸 "모델.칸"(칸 이름은 by_alias 이름). tests/test_format_v2.py와 같은 규칙."""
    from pydantic import BaseModel

    seen: set[type] = set()
    out: set[str] = set()

    def visit(tp: object) -> None:
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            if tp in seen:
                return
            seen.add(tp)
            for name, f in tp.model_fields.items():
                out.add(f"{tp.__name__}.{f.alias or name}")
                visit(f.annotation)
            return
        for arg in typing.get_args(tp):
            visit(arg)

    visit(root)
    return sorted(out)


def main() -> None:
    base = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    repo = Path(__file__).resolve().parents[2]

    import evolab

    assert Path(evolab.__file__).resolve().is_relative_to(base), evolab.__file__
    from evolab.engine import seeds
    from evolab.engine.evaluate import evaluate
    from evolab.policy.genome import layout_for
    from evolab.spec import model as m
    from evolab.spec.io import compile_env, dump_env, env_hash, parse_env
    from evolab.world.physics2d.episode import Trial

    assert m.CURRENT_VERSION == 1

    # T5: v1 모델 칸 목록
    fields = model_field_names(m.EnvFile)
    (out_dir / "schema_fields_v1.json").write_text(
        json.dumps({"made_by": "tests/data/make_v1_reference.py", "commit": "3dbcb95", "fields": fields},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # T1, T2: v1 파일의 정규 글과 해시. 원본은 저장소의 파일(사본 폴더가 아니라)이다.
    sources = {p.stem: p for p in sorted((repo / "environments").glob("*.evoenv"))}
    for run_env in sorted((repo / "tests" / "data").glob("*/run/envs/v000.evoenv")):
        sources[run_env.parents[2].name] = run_env
    dump_dir = out_dir / "v1_dump"
    dump_dir.mkdir(exist_ok=True)
    hashes = []
    for stem, path in sources.items():
        env = parse_env(path.read_bytes())
        (dump_dir / f"{stem}.json").write_bytes(dump_env(env).encode("utf-8"))
        hashes.append(f"{stem} {env_hash(env)} {path.relative_to(repo).as_posix()}")
    (dump_dir / "env_hash.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")

    # T29: v1 결과 회귀 자료
    rng = np.random.default_rng(20261107)
    trials = [Trial(seeds.derive(9, k)) for k in range(3)]
    arrays: dict[str, np.ndarray] = {
        "platform": np.array(f"{sys.platform}-{platform.machine()}"),
        "trial_seeds": np.array([t.seed for t in trials], dtype=np.uint64),
    }
    for name in ("lander", "track_car", "hill_car"):
        text = (repo / "environments" / f"{name}.evoenv").read_text(encoding="utf-8")
        spec = compile_env(text)
        layout = layout_for(spec)
        genomes = layout.initial()[None, :] + rng.normal(0.0, 0.5, (8, layout.dim))
        pairs = [(g, t) for g in range(8) for t in trials]
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
    np.savez_compressed(out_dir / "v1_regress.npz", **arrays)
    digest = hashlib.sha256()
    for k in sorted(arrays):
        digest.update(k.encode())
        digest.update(np.ascontiguousarray(arrays[k]).tobytes())
    print("v1_regress digest", digest.hexdigest())


if __name__ == "__main__":
    main()

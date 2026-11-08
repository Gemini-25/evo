"""v2.1 0'단계: v2.0 참조 자료를 v2.0 코드(커밋 4d3e3e6)로 만든다(v2 설계 T3의 v2.1 추가분).

이 스크립트는 저장소의 지금 코드가 아니라 v2.0 코드 사본으로 돌린다(그래야 "v2.0 프로그램이 쓴 글"이 된다):

    git archive 4d3e3e6 | tar -x -C <사본 폴더>
    PYTHONPATH=<사본 폴더>/src .venv/bin/python tests/data/make_v2_reference.py <사본 폴더> tests/data

만드는 것(모두 tests/data/v2_dump 아래, 테스트는 읽기만 한다):

- <이름>.json: 형식 2 파일을 v2.0 코드의 dump_env로 쓴 글. jump·glider는 저장소 environments/의 내장 예제,
  all_v2_features는 v2.0의 새 칸·종류·값(모양별 밀도 넷, 날개 모양, 정해진 움직임 물체와 path·wave 움직임, 점프 추진기,
  공기역학, 장애물 줄)을 모두 쓴 파일(이 스크립트 안의 사전)이다.
- env_hash.txt: 같은 파일의 v2.0 env_hash("이름 해시 원본 경로"; all_v2_features의 원본은 그 글 자체).

v2.1 코드는 이 글을 읽어 다시 쓰면 글자까지 같고 해시도 같아야 한다(형식 3 칸 exists는 형식 2 글에 나타나지 않음).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ALL_V2_FEATURES: dict[str, Any] = {
    "version": 2,
    "name": "v2.0 features",
    "params": [{"id": "wing_x", "value": 0.2, "evolve": True, "min": -0.5, "max": 0.5}],
    "randomize": [{"id": "oseed", "dist": "int", "low": 0, "high": 1000000}],
    "generators": [{"kind": "obstacles", "id": "obs", "seed": "=r.oseed", "count": 12, "x": 20.0}],
    "bodies": [
        {"id": "ground", "owner": "world", "type": "static", "tags": ["ground"],
         "shapes": [{"kind": "box", "w": 60.0, "h": 1.0, "offset": [0.0, -0.5]}]},
        {"id": "lift", "owner": "world", "type": "kinematic", "pos": [5.0, 1.0],
         "shapes": [{"kind": "box", "w": 2.0, "h": 0.2}],
         "motion": {"kind": "path", "points": [[0.0, 0.0], [0.0, 3.0]], "speed": 1.0, "mode": "pingpong"}},
        {"id": "bob", "owner": "world", "type": "kinematic", "pos": [-5.0, 2.0],
         "shapes": [{"kind": "circle", "r": 0.4}],
         "motion": {"kind": "wave", "amplitude": [0.0, 0.5], "period": 2.0, "phase": 0.25}},
        {"id": "hull", "pos": [0.0, 2.0],
         "shapes": [{"kind": "box", "w": 1.0, "h": 1.0, "density": 30.0},
                    {"kind": "circle", "r": 0.3, "offset": [0.0, 0.6], "density": "=2 * 60"},
                    {"kind": "poly", "points": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], "density": 0.5},
                    {"kind": "radial", "radii": [0.5, 0.6, 0.7], "density": 2.5},
                    {"kind": "wing", "le": ["=p.wing_x", 0.1], "chord": 1.2, "camber": 0.03}]},
    ],
    "actuators": [
        {"kind": "thruster", "id": "main", "body": "hull", "max_force": 100.0, "keys": ["Up"]},
        {"kind": "jump", "id": "jump", "body": "hull", "keys": ["Space"]},
        {"kind": "aero", "id": "air", "body": "hull", "surfaces": "wings"},
    ],
    "sensors": [{"kind": "time_left", "id": "tl"}],
    "editor": {"view": [-3.0, -1.0, 22.0, 6.0]},
}


def main() -> None:
    base = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    repo = Path(__file__).resolve().parents[2]

    import evolab

    assert Path(evolab.__file__).resolve().is_relative_to(base), evolab.__file__
    from evolab.spec import model as m
    from evolab.spec.io import compile_env, dump_env, env_hash, feature_version, parse_env

    assert m.CURRENT_VERSION == 2

    dump_dir = out_dir / "v2_dump"
    dump_dir.mkdir(exist_ok=True)
    hashes = []
    for stem in ("jump", "glider"):
        path = repo / "environments" / f"{stem}.evoenv"
        env = parse_env(path.read_bytes())
        compile_env(env)
        assert feature_version(env) == 2
        (dump_dir / f"{stem}.json").write_bytes(dump_env(env).encode("utf-8"))
        hashes.append(f"{stem} {env_hash(env)} {path.relative_to(repo).as_posix()}")
    env = parse_env(json.dumps(ALL_V2_FEATURES))
    compile_env(env)
    assert feature_version(env) == 2
    path = dump_dir / "all_v2_features.json"
    path.write_bytes(dump_env(env).encode("utf-8"))
    hashes.append(f"all_v2_features {env_hash(env)} {path.relative_to(repo).as_posix()}")
    (dump_dir / "env_hash.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    print("\n".join(hashes))


if __name__ == "__main__":
    main()

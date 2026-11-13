"""내장 종목 프리셋(environments/<이름>.evoenv)을 내장 조리법(src/evolab/assets/recipes/<이름>.json)으로 만든다(6.2.0, 설계 D29·D30).

environments/의 종목 파일은 이 스크립트의 출력만이다(손으로 고치지 않는다). 조리법이나 틀을 고친 묶음이 이것을 돌려 파일을
다시 쓰고, 그 커밋이 전체 테스트를 다시 돌린다.

    python packaging/make_presets.py            # SPORTS의 파일을 다시 쓴다
    python packaging/make_presets.py --check    # 다시 쓰지 않고 견준다: 파일이 없거나 환경 해시가 다르면 종료 코드 1
    python packaging/make_presets.py running    # 그 이름만

견주기는 env_hash(editor 구역 밖의 내용)로 한다. 다르면 무엇이 다른지(물리 칸 경로) 몇 줄을 적는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SPORTS: tuple[str, ...] = ("running",)
"""environments/에 내장 파일로 싣는 종목 조리법(D36: 「미룸」 hidden은 싣지 않는다). 6.3.0에 boxing·soccer(관문을 넘은 것만)."""


def env_dir() -> Path:
    return ROOT / "environments"


def render(name: str) -> str:
    """조리법 하나로 만든 환경 파일 글(정규 JSON, editor.recipe 표지 포함)."""
    from evolab.presets.recipe import build, load_builtin
    from evolab.spec.io import dump_env, parse_env

    return dump_env(parse_env(build(load_builtin(name))))


def _diff(a: Any, b: Any, path: str = "", out: list[str] | None = None, limit: int = 8) -> list[str]:
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'built' if k in a else 'file'}")
            else:
                _diff(a[k], b[k], f"{path}.{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            _diff(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def check(name: str) -> list[str]:
    """견주기: 문제 줄(없으면 빈 목록)."""
    from evolab.presets.recipe import physics_view
    from evolab.spec.compiled import EnvFileError
    from evolab.spec.io import env_hash, parse_env

    path = env_dir() / f"{name}.evoenv"
    if not path.is_file():
        return [f"{path.name}: missing"]
    built = parse_env(render(name))
    try:
        have = parse_env(path.read_bytes())
    except EnvFileError as exc:
        return [f"{path.name}: {exc}"]
    if env_hash(built) == env_hash(have):
        return []
    import json

    lines = [f"{path.name}: env_hash differs from build(load_builtin({name!r}))"]
    lines += _diff(physics_view(json.loads(render(name))), physics_view(json.loads(path.read_text(encoding="utf-8"))))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build environments/<sport>.evoenv from assets/recipes/<sport>.json")
    parser.add_argument("names", nargs="*", help="sport recipe names (default: all of SPORTS)")
    parser.add_argument("--check", action="store_true", help="compare only; exit 1 when a file is missing or differs")
    args = parser.parse_args(argv)
    names = args.names or list(SPORTS)
    problems: list[str] = []
    for name in names:
        if args.check:
            problems += check(name)
            continue
        text = render(name)
        path = env_dir() / f"{name}.evoenv"
        path.write_bytes(text.encode("utf-8"))
        print(f"wrote {path.relative_to(ROOT).as_posix()}")
    for line in problems:
        print(line, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

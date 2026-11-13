"""달리기 프리셋의 넘어짐 감점(s_fall의 weight)만 바꾼 환경 파일을 만든다 (측정용, 배포본과 무관).

    uv run python research/running_tune/make_env.py --weight -3 --out /tmp/running_w3.evoenv

내장 파일 environments/running.evoenv를 읽어 id가 "s_fall"인 점수 블록의 weight만 바꿔 쓴다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _set_weight(node: object, weight: float) -> int:
    n = 0
    if isinstance(node, dict):
        if node.get("id") == "s_fall" and "weight" in node:
            node["weight"] = weight
            n += 1
        for v in node.values():
            n += _set_weight(v, weight)
    elif isinstance(node, list):
        for v in node:
            n += _set_weight(v, weight)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--src", type=Path, default=ROOT / "environments" / "running.evoenv")
    a = ap.parse_args()
    data = json.loads(a.src.read_text(encoding="utf-8-sig"))
    if _set_weight(data, a.weight) != 1:
        raise SystemExit("s_fall block not found exactly once")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.out} (s_fall weight {a.weight})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

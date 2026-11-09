"""v6.0 통합 (v6 설계 7.2 I 묶음, 9.3 v6.0-9): 문자열 정리와 묶음 사이 연결.

- strings.py 「# v6.0 빠르게」 블록의 상수가 모두 쓰인다(v3.1 test_every_v31_string_is_used와 같은 방식. 도움말 자리 {{이름}}과
  HELP_TIPS 항목처럼 strings.py 안에서 다른 값이 그 상수를 쓰는 것도 쓰임으로 센다).
- 평가기(E)가 런타임(R)의 step_values를 정말 쓴다: 지금 Episode로 평가하면 공개 step을 한 번도 부르지 않고 스텝마다
  step_values를 부른다(R이 이름을 바꾸면 결과는 같고 느려지기만 해서 다른 테스트로는 드러나지 않는다).
- bench 「관전 재생 길이」 줄(E, engine/commands.py)의 꼬리 초가 실험실 재생 끝 규칙(P, app/lab/view.py)과 같다(엔진은 Qt 쪽
  모듈을 부를 수 없어 값을 따로 둔다).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab import paths
from evolab.engine import commands
from evolab.engine import evaluate as evaluate_mod
from evolab.engine import seeds
from evolab.policy.genome import layout_for
from evolab.spec.io import load_env
from evolab.world.physics2d.episode import Episode

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
STRINGS = SRC / "evolab" / "app" / "strings.py"
V60_HEADER = "# v6.0 빠르게"
NEXT_RELEASE_HEADER = "# v6.1"


def _v60_constant_names() -> list[str]:
    text = STRINGS.read_text(encoding="utf-8")
    start = text.index(V60_HEADER)
    end = text.find(f"\n{NEXT_RELEASE_HEADER}", start)
    block = text[start:end] if end != -1 else text[start:]
    return re.findall(r"^([A-Z][A-Z0-9_]+): Final", block, flags=re.MULTILINE)


def _strings_code_without_definitions(names: list[str]) -> str:
    """strings.py에서 주석 줄과 그 상수들의 정의 줄을 뺀 나머지(다른 값이 상수를 쓰는 곳: HELP_TIPS 항목, 도움말 자리)."""
    keep = []
    for line in STRINGS.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") or any(stripped.startswith(f"{n}: Final") for n in names):
            continue
        keep.append(line.split("  # ")[0])
    return "\n".join(keep)


def test_every_v60_string_is_used() -> None:
    names = _v60_constant_names()
    assert len(names) == 18  # 설계 6절 v6.0 새 상수(이 검사가 블록 머리를 놓치면 0이 된다)
    sources = [p for p in [*SRC.rglob("*.py"), *(ROOT / "packaging").rglob("*.py")]
               if p.name != "strings.py" and "__pycache__" not in p.parts]
    text = "\n".join(p.read_text(encoding="utf-8") for p in sources) + "\n" + _strings_code_without_definitions(names)
    unused = [n for n in names if not re.search(rf"\b{n}\b", text)]
    assert unused == []


def test_v60_names_are_defined_once() -> None:
    """v6.0 새 이름은 strings.py에서 한 번만 정의한다(제자리에서 바꾼 옛 이름은 블록 밖에 그대로, test_rules·v3 테스트도 봄)."""
    text = STRINGS.read_text(encoding="utf-8")
    names = _v60_constant_names()
    for n in names:
        assert len(re.findall(rf"^{n}: Final", text, flags=re.MULTILINE)) == 1, n


@pytest.mark.parametrize("name", ["lander", "jump", "wheels"])
def test_evaluate_drives_the_runtime_through_step_values(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """E 묶음 평가기 + R 묶음 런타임: 공개 step은 부르지 않고(부르면 실패) step_values를 스텝마다 한 번씩 부른다."""
    spec = load_env(paths.builtin_env_dir() / f"{name}.evoenv")
    layout = layout_for(spec)
    rng = np.random.default_rng(60)
    genomes = layout.initial()[None, :] + 0.5 * rng.standard_normal((3, layout.dim))
    pairs = [(i, t) for i in range(3) for t in seeds.train_trials(60, 0, 2)]
    calls: list[int] = []
    real = Episode.step_values

    def counting(self: Episode, values: list[float]) -> None:
        assert isinstance(values, list) and all(type(v) is float for v in values)
        calls.append(1)
        real(self, values)

    def forbidden(self: Episode, action: Any) -> None:
        raise AssertionError("evaluate called the public Episode.step")

    monkeypatch.setattr(Episode, "step_values", counting)
    monkeypatch.setattr(Episode, "step", forbidden)
    res = evaluate_mod.evaluate(spec, layout, genomes, pairs, window=4)
    assert len(calls) == int(res.steps.sum()) > 0


def test_bench_playback_tail_is_the_lab_playback_tail() -> None:
    from evolab.app.lab import view as lab_view

    assert commands.PLAYBACK_TAIL_S == lab_view.SKIP_TAIL_S == 0.5

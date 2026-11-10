"""v6.1 통합 (v6 설계 7.2 I' 묶음, 9.3 v6.1-6): 문자열 정리와 묶음 사이 연결.

- strings.py 「# v6.1 자라는 뇌」 블록부터 끝까지(0'·L·C 묶음 블록)의 상수가 모두 쓰인다(v6.0 test_every_v60_string_is_used와 같은
  방식. 도움말 자리 {{이름}}과 HELP_TIPS 항목처럼 strings.py 안에서 다른 값이 그 상수를 쓰는 것도 쓰임으로 센다).
- 화면 묶음(C) 테스트가 쓰는 가짜 통계 줄(test_grow_screens.grow_rows)이 진짜 학습기(L)의 통계 열과 같다: 같은 간격의 진짜 자라는
  실행에서 brain_neurons·brain_links·grow가 줄마다 같고, 대시보드가 진짜 줄로 긋는 「뇌가 자람」 세로선·풍선·「뇌 크기」 타일이
  가짜 줄로 그은 것과 같다. 세로선 풍선의 새 크기(다음 줄의 brain_neurons)가 세대 메시지 "growth"의 값(Trainer.last_growth)과 같다.
- 실험실 안내 줄(C, LAB_GROW_HINT)·명령줄 시작 줄(L, TRAIN_GROW_SUMMARY)·진짜 학습 방법이 같은 크기와 세대를 말한다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from test_dashboard import make_dashboard
from test_grow_screens import grow_rows, make_lab

from evolab import paths
from evolab.app import strings as S
from evolab.engine import commands
from evolab.engine.pool import WorkerPool
from evolab.engine.trainer import GROW_COLUMNS, TIME_COLUMNS, TrainConfig, Trainer
from evolab.learners.cmaes import CMAESConfig
from evolab.learners.grow import GrowConfig, GrowingCMAES
from evolab.policy.frame import BrainFrame
from evolab.policy.genome import layout_for
from evolab.spec.io import compile_env

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
STRINGS = SRC / "evolab" / "app" / "strings.py"
V61_HEADER = "# v6.1 자라는 뇌"
LANDER_TEXT = (paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8")


def _v61_constant_names() -> list[str]:
    text = STRINGS.read_text(encoding="utf-8")
    block = text[text.index(V61_HEADER):]
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


def test_every_v61_string_is_used() -> None:
    names = _v61_constant_names()
    # 0'단계 49개 중: 설계 6절 37개 + FRAME_* 5개, L 묶음 6개, C 묶음 1개(GROW_TILE_EMPTY). 블록 머리를 놓치면 0이 된다
    assert len(names) == 49
    sources = [p for p in [*SRC.rglob("*.py"), *(ROOT / "packaging").rglob("*.py")]
               if p.name != "strings.py" and "__pycache__" not in p.parts]
    text = "\n".join(p.read_text(encoding="utf-8") for p in sources) + "\n" + _strings_code_without_definitions(names)
    unused = [n for n in names if not re.search(rf"\b{n}\b", text)]
    assert unused == []


def test_v61_names_are_defined_once() -> None:
    text = STRINGS.read_text(encoding="utf-8")
    for n in _v61_constant_names():
        assert len(re.findall(rf"^{n}: Final", text, flags=re.MULTILINE)) == 1, n


# ---------------------------------------------------------------------------
# 화면 테스트의 가짜 통계 줄 = 진짜 학습기의 통계 열
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_grow_run() -> tuple[list[dict[str, Any]], list[tuple[int, int, int, int] | None]]:
    """진짜 짧은 자라는 실행(착륙선, 개체 6, 시험 1, 3세대마다 자람, 이 프로세스 안 평가): (통계 줄, 세대마다 last_growth)."""
    config = TrainConfig(method="grow", run_seed=5, popsize=6, trials=1, grow_every=3, validate_every=60,
                         validation_seeds=2, validation_candidates=1, max_generations=11)
    rows: list[dict[str, Any]] = []
    growths: list[tuple[int, int, int, int] | None] = []
    with WorkerPool(0) as pool:
        trainer = Trainer(LANDER_TEXT, config, pool, None, observe=False)
        for _ in range(11):
            rows.append(trainer.step())
            growths.append(trainer.last_growth)
    return rows, growths


def test_screen_test_rows_are_the_real_grow_columns(real_grow_run: Any) -> None:
    rows, growths = real_grow_run
    frame = BrainFrame(layout_for(compile_env(LANDER_TEXT)))
    fake = grow_rows(frame, 11, every=3)
    assert [{k: r[k] for k in GROW_COLUMNS} for r in rows] == [{k: r[k] for k in GROW_COLUMNS} for r in fake]
    keys = list(rows[0])
    assert keys[keys.index("switch") + 1:keys.index("switch") + 4] == list(GROW_COLUMNS)  # switch 바로 뒤(설계 D35)
    # 세대 메시지 "growth"(Trainer.last_growth)의 옛 크기 = 그 줄, 새 크기 = 다음 줄(대시보드 풍선이 다음 줄에서 읽는 값)
    for i, g in enumerate(growths):
        assert (g is None) == (rows[i]["grow"] == 0)
        if g is not None:
            old, new, old_links, new_links = g
            assert (old, old_links) == (rows[i]["brain_neurons"], rows[i]["brain_links"])
            assert (new, new_links) == (rows[i + 1]["brain_neurons"], rows[i + 1]["brain_links"])


def test_dashboard_draws_real_rows_like_the_screen_test_rows(qtbot: QtBot, tmp_path: Path, real_grow_run: Any) -> None:
    from evolab.app.lab.watch_types import WatchMeta

    rows, growths = real_grow_run
    spec = compile_env(LANDER_TEXT)
    lay = layout_for(spec)
    meta = WatchMeta.from_spec(spec, lay)
    fake = grow_rows(BrainFrame(lay), 11, every=3)
    seen = []
    for name, data in (("real", rows), ("csv", [{k: str(v) for k, v in r.items() if k not in TIME_COLUMNS}
                                                 for r in rows]), ("fake", fake)):
        dash = make_dashboard(qtbot, tmp_path, f"{name}.ini")
        dash.set_watch_meta(meta)
        for r in data:  # 학습 중처럼 한 줄씩
            dash.add_row(r)
        dash.set_page("fun")
        dash.flush()
        tile = dash.tile("brainsize")
        assert dash.brain_grows and tile.series is not None
        seen.append(([(g, k) for g, k, _ in dash.auto_marks], list(dash._auto_tips),
                     [int(tile.gens[i]) for i in tile.series.events], tile.series.neurons.tolist(),
                     tile.series.links.tolist()))
    assert seen[0] == seen[1] == seen[2]
    marks, tips, events, _neurons, _links = seen[0]
    assert marks == [(3, "grow"), (6, "grow"), (9, "grow")] and events == [3, 6, 9]
    want = [S.DASH_MARK_GROW.format(gen=i + 1, old=g[0], new=g[1]) for i, g in enumerate(growths) if g is not None]
    assert tips == want


# ---------------------------------------------------------------------------
# 실험실 안내 줄 · 명령줄 시작 줄 · 진짜 학습 방법
# ---------------------------------------------------------------------------
def _lander_with(hidden: list[int]) -> str:
    data = json.loads(LANDER_TEXT)
    data["brain"]["hidden"] = hidden
    return json.dumps(data, ensure_ascii=False)


def _learner_path(text: str, start: int, every: int) -> tuple[list[int], int]:
    """진짜 학습 방법(가짜 점수)이 다 자랄 때까지: (뉴런 합 경로, 다 자란 세대)."""
    lay = layout_for(compile_env(text))
    lr = GrowingCMAES(lay.initial(), lay, CMAESConfig(popsize=4, seed=1, bounded_dims=lay.n_body),
                      GrowConfig(start=start, every=every))
    path, gen = [sum(lr.counts)], 0
    target = np.linspace(-0.2, 0.2, lay.dim)
    while lr.counts != lr.frame.sizes:
        X = lr.ask()
        lr.tell(X, -np.sum((X - target) ** 2, axis=1))
        gen += 1
        if lr.last_growth is not None:
            path.append(sum(lr.counts))
    return path, gen


@pytest.mark.parametrize(("hidden", "start", "every"), [([16], 2, 10), ([16], 3, 4), ([8, 4], 2, 5), ([6, 5], 1, 2),
                                                         ([16], 16, 10)])
def test_lab_hint_cli_summary_and_learner_agree(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                hidden: list[int], start: int, every: int) -> None:
    text = _lander_with(hidden)
    path, gen = _learner_path(text, start, every)
    view = make_lab(qtbot, tmp_path, monkeypatch)
    try:
        assert view.use_env_text(text, "test", None)
        view.grow_start.setValue(start)
        view.grow_every.setValue(every)
        hint = view.grow_hint_text()
        if len(path) == 1:
            assert hint == S.LAB_GROW_HINT_ALREADY
        else:
            assert hint == S.LAB_GROW_HINT.format(path=" → ".join(map(str, path)), gen=gen)
        config = TrainConfig(method="grow", grow_start=start, grow_every=every)
        assert commands._grow_text(config, view.layout) == S.TRAIN_GROW_SUMMARY.format(start=path[0], full=path[-1],
                                                                                        every=every)
    finally:
        view.shutdown()

"""v6.0 `evolab bench` 줄 (v6 설계 4.10·5.6·D17, 8.1 T17, E 묶음).

bench --popsize 4 --generations 1 --workers 1 출력에 「창 12」 단일 코어 줄, 「평가 창」 줄(창 5개), 「관전 재생 길이」 줄,
「pymunk C 함수」 줄이 있다. 줄 순서: 제목, 단일 코어, 관찰을 켠 단일 코어(v1.2, 바로 다음), 평가 창, 관전 재생 길이,
pymunk C 함수, 워커 줄, 최악의 세대, (v2 부품). v2 부품 줄이 맨 끝이고 최악의 세대가 그 앞인 것은 기존 테스트
(test_judge_v2.py)가 지킨다.
"""

from __future__ import annotations

import contextlib
import io
import re
from typing import Any

import numpy as np
import pytest

from evolab import cli, paths
from evolab.app import strings as S
from evolab.engine import commands
from evolab.engine.commands import BENCH_WINDOW_ROUNDS, BENCH_WINDOWS, PLAYBACK_TAIL_S
from evolab.engine.evaluate import EVAL_WINDOW
from evolab.engine.pool import WorkerPool
from evolab.engine.trainer import training_trials
from evolab.policy.genome import layout_for
from evolab.spec.io import load_env
from evolab.world.physics2d import episode as runtime

LANDER = paths.builtin_env_dir() / "lander.evoenv"


def head(template: str, **kw: Any) -> str:
    """서식 문자열에서 처음 남은 자리 앞까지. kw의 자리({이름}, 서식 지정 없음)는 먼저 채운다."""
    for name, value in kw.items():
        template = template.replace("{" + name + "}", str(value))
    return template.split("{", 1)[0]


def bench_lines(*args: str) -> list[str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["bench", str(LANDER), *args]) == 0
    return buf.getvalue().strip().split("\n")


@pytest.fixture(scope="module")
def lander_out() -> list[str]:
    """T17의 명령: bench --popsize 4 --generations 1 --workers 1 (한 번 돌려 여러 테스트가 읽음)."""
    return bench_lines("--popsize", "4", "--generations", "1", "--workers", "1")


def bench_batch(env: Any, pop: int) -> tuple[Any, Any, np.ndarray, list[Any], int]:
    """bench_command와 같은 무작위 유전자 개체·시험(1세대)."""
    spec = load_env(env)
    layout = layout_for(spec)
    rng = np.random.default_rng(0)
    genomes = layout.initial()[None, :] + 0.5 * rng.standard_normal((pop, layout.dim))
    trials_n = spec.track_set_size or spec.file.evaluation.trials
    trials = training_trials(spec, 1, 0, trials_n)
    return spec, layout, genomes, [(i, t) for i in range(pop) for t in trials], trials_n


def test_t17_bench_prints_the_v6_lines_in_order(lander_out: list[str]) -> None:
    out = lander_out
    single = head(S.BENCH_SINGLE, window=EVAL_WINDOW)
    assert single.startswith("단일 코어(창 12)")
    starts = [single, head(S.BENCH_OBSERVE), head(S.BENCH_WINDOW), head(S.BENCH_PLAYBACK), head(S.BENCH_FAST_PATHS),
              head(S.BENCH_POOL, workers=1), head(S.BENCH_WORST)]
    assert len(out) == 1 + len(starts)  # 착륙선: v2 부품 줄 없음
    assert out[0].startswith(head(S.BENCH_TITLE))
    for line, start in zip(out[1:], starts, strict=True):
        assert line.startswith(start), (line, start)


def test_t17_window_line_has_five_windows_and_the_fastest(lander_out: list[str]) -> None:
    out = lander_out
    line = next(x for x in out if x.startswith(head(S.BENCH_WINDOW)))
    row = re.escape(S.BENCH_WINDOW_ROW).replace(r"\{w\}", r"(\d+)").replace(r"\{us:\.1f\}", r"([0-9.]+)")
    sep = re.escape(S.BENCH_WINDOW_SEP)
    pattern = (re.escape(S.BENCH_WINDOW).replace(r"\{rows\}", f"((?:{row}{sep})*{row})")
               .replace(r"\{best\}", r"(\d+)").replace(r"\{rounds\}", str(BENCH_WINDOW_ROUNDS)))
    m = re.fullmatch(pattern, line)
    assert m is not None, line
    rows = re.findall(row, m.group(1))
    assert [int(w) for w, _ in rows] == list(BENCH_WINDOWS) == [4, 8, 12, 16, 24]
    us = {int(w): float(u) for w, u in rows}
    best = int(m.groups()[-1])
    assert best in BENCH_WINDOWS and us[best] == min(us.values())
    assert all(u > 0 for u in us.values())


def test_t17_playback_and_fast_path_lines(lander_out: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    out = lander_out
    spec, layout, genomes, pairs, trials_n = bench_batch(LANDER, 4)
    res = WorkerPool(0).evaluate("v", LANDER.read_text(encoding="utf-8"), layout, genomes, pairs)
    sec = float(res.steps[0::trials_n].max()) * spec.file.world.dt  # 시험 0번 가장 긴 궤적
    assert S.BENCH_PLAYBACK.format(sec=sec, wait=sec + PLAYBACK_TAIL_S) in out
    assert PLAYBACK_TAIL_S == 0.5
    assert runtime._CP_FAST  # 고정한 pymunk에서는 빠른 길이 켜짐(v6 설계 D19)
    assert S.BENCH_FAST_PATHS.format(state=S.BENCH_FAST_ON) in out
    monkeypatch.setattr(runtime, "_CP_FAST", False)  # 이 프로세스의 단일 코어 평가가 공개 API 길로 돈다(결과는 같음)
    off = bench_lines("--popsize", "2", "--generations", "1", "--workers", "1")
    assert S.BENCH_FAST_PATHS.format(state=S.BENCH_FAST_OFF) in off
    assert S.BENCH_FAST_PATHS.format(state=S.BENCH_FAST_ON) not in off


def test_fast_path_line_says_off_when_episodes_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """v6.0 검토: 「pymunk C 함수 바로 부르기」 줄은 깃발(_CP_FAST, cp* 함수를 불러옴)만이 아니라 이 환경의 Episode가 정말 포인터를
    얻는지를 본다(자체 점검 「빠른 계산 길」과 같은 판단, episode.fastpath_in_use). pymunk 내부 이름이 바뀌어 물체 포인터를 얻지
    못하면 깃발이 참이어도 「꺼짐」."""
    assert runtime._CP_FAST
    monkeypatch.setattr(runtime, "_body_c", lambda body: None)
    spec = load_env(LANDER)
    assert not runtime.fastpath_in_use(spec)
    out = bench_lines("--popsize", "2", "--generations", "1", "--workers", "1")
    assert S.BENCH_FAST_PATHS.format(state=S.BENCH_FAST_OFF) in out
    assert S.BENCH_FAST_PATHS.format(state=S.BENCH_FAST_ON) not in out


def test_bench_windows_measures_every_window_and_picks_the_fastest(monkeypatch: pytest.MonkeyPatch) -> None:
    spec, layout, genomes, pairs, _ = bench_batch(LANDER, 2)
    text = LANDER.read_text(encoding="utf-8")
    seen: list[int | None] = []
    real = WorkerPool.evaluate

    def spy(self: WorkerPool, *args: Any, **kw: Any) -> Any:
        seen.append(self.window)
        return real(self, *args, **kw)

    monkeypatch.setattr(WorkerPool, "evaluate", spy)
    us, best = commands.bench_windows("v", text, layout, genomes, pairs, windows=(1, 3, 5), rounds=3)
    assert seen == [1, 3, 5, 5, 3, 1, 1, 3, 5]  # 회차마다 순서를 뒤집어 번갈아
    assert set(us) == {1, 3, 5} and best == min(us, key=lambda w: (us[w], w))


def test_playback_seconds_takes_trial_zero_of_every_individual() -> None:
    spec = load_env(LANDER)
    steps = np.array([10, 99, 99, 30, 5, 99, 40, 1, 1])  # 개체 3 × 시험 3: 시험 0번 = 10, 30, 40
    assert commands.playback_seconds(spec, steps, 3) == pytest.approx(40 * spec.file.world.dt)
    assert commands.playback_seconds(spec, np.zeros(0, dtype=np.int64), 3) == 0.0

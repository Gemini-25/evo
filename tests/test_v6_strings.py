"""v6.0 0단계 문자열 (v6 설계 6절 v6.0 줄, 관전 기본 「학습 먼저」 = 사용자 결정 G1·Q1).

새 상수 18개가 있고 format 자리 이름이 R·E·P·I 묶음이 넘길 인수와 같다(묶음 사이 계약). 제자리에서 바꾼 값(BENCH_SINGLE,
ONB_STEP_WATCH, 도움말 두 절, 「관전」 풍선)이 기본값 「학습 먼저」와 맞는다. 도움말 자리·「」 규칙·이름 두 번 쓰기는 v3 테스트
(test_v3_strings.py, test_help_content.py)가 그대로 본다.
"""

from __future__ import annotations

import re
import string

from evolab.app import strings as S

V60_FIELDS: dict[str, set[str]] = {
    "LAB_PACE": set(),
    "LAB_PACE_SKIP": set(),
    "LAB_PACE_EVERY": set(),
    "LAB_PACE_TIP": set(),
    "LAB_HUD_LIVE_SKIP": {"gen", "t", "speed", "latest"},
    "LAB_PACE_WAIT_HINT": {"times"},
    "LAB_PACE_CHANGED_NOTICE": set(),
    "BENCH_WINDOW": {"rows", "best", "rounds"},
    "BENCH_WINDOW_ROW": {"w", "us"},
    "BENCH_WINDOW_SEP": set(),
    "BENCH_PLAYBACK": {"sec", "wait"},
    "BENCH_FAST_PATHS": {"state"},
    "BENCH_FAST_ON": set(),
    "BENCH_FAST_OFF": set(),
    "SELFTEST_CHECK_FASTPATH": set(),
    "SELFTEST_FASTPATH_SAME": set(),
    "SELFTEST_FASTPATH_OFF": set(),
    "SELFTEST_FASTPATH_DIFF": {"detail"},
}
"""v6 설계 6절의 v6.0 새 상수(18개)와 format 자리 이름."""


def _fields(text: str) -> set[str]:
    return {f.split(".")[0].split("[")[0] for _, f, _, _ in string.Formatter().parse(text) if f}


def _section(page: str, heading: str) -> str:
    md = S.HELP_PAGES[page][1]
    return md.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]


def test_v60_constants_exist_with_their_format_fields() -> None:
    assert len(V60_FIELDS) == 18
    for name, fields in V60_FIELDS.items():
        value = getattr(S, name)
        assert isinstance(value, str) and value.strip(), name
        assert _fields(value) == fields, name
        value.format(**dict.fromkeys(fields, 1.0))


def test_changed_constants_follow_the_skip_default() -> None:
    # bench 단일 코어 줄은 창 평가 기준(설계 4.10): engine/commands.py가 window=를 넘긴다
    assert _fields(S.BENCH_SINGLE) == {"window", "sps", "us", "nn", "steps", "sec"}
    assert S.BENCH_SINGLE.startswith("단일 코어(창 {window}): ")
    # 처음 실행 안내 ④: 자리는 그대로 {speed} 하나, 학습이 재생을 기다리지 않는다는 말
    assert _fields(S.ONB_STEP_WATCH) == {"speed"} and "학습은 재생과 상관없이 계속됩니다" in S.ONB_STEP_WATCH
    # 「학습 먼저」는 1세대를 길어야 SKIP_MAX_VIEW(15초) 보이고 새 세대로 넘어간다: ④(배속 버튼은 30초 뒤)가 「지금 보이는 것이
    # 1세대」라고 하면 곧 틀린 말이 된다(v6.0 검토)
    for text in (S.ONB_STEP_WATCH, S.ONB_STEP_WATCH_FAST):
        assert text.startswith("처음 보이는 것이 1세대입니다.") and "지금 보이는" not in text
    assert "새 세대로 넘어갑니다" in S.ONB_STEP_WATCH
    assert "기다" not in S.HELP_TIPS["lab.spectate"].replace("기다릴지", "")
    assert S.LAB_PACE in S.HELP_TIPS["lab.spectate"]


def test_help_sections_describe_both_paces() -> None:
    watch = _section("lab", "관전과 고속")
    assert all(f"「{{{{{n}}}}}」" in watch for n in ("LAB_SPECTATE", "LAB_PACE", "LAB_PACE_SKIP", "LAB_PACE_EVERY",
                                                    "LAB_SPEED", "LAB_FAST"))
    assert "evolab://show/lab.speed" in watch
    slow = _section("trouble", "느릴 때")
    first = slow.strip().split("\n\n", 1)[0]
    assert re.search(r"「\{\{LAB_PACE_EVERY\}\}」", first) and "{{LAB_PACE_SKIP}}" in first and "{{LAB_FAST}}" in first
    assert "워커 수" in slow  # 예전 문단은 그대로 뒤에 남는다


def test_pace_labels_are_distinct_and_named_in_the_hints() -> None:
    assert len({S.LAB_PACE_SKIP, S.LAB_PACE_EVERY, S.LAB_PACE}) == 3
    short_skip = S.LAB_PACE_SKIP.split(" (", 1)[0]
    assert short_skip == "학습 먼저" and short_skip in S.LAB_PACE_WAIT_HINT and short_skip in S.LAB_PACE_CHANGED_NOTICE
    assert S.LAB_PACE in S.LAB_PACE_WAIT_HINT and S.LAB_VIEW_GROUP in S.LAB_PACE_CHANGED_NOTICE
    assert S.LAB_HUD_LIVE_SKIP.startswith(S.LAB_HUD_LIVE)  # 「학습 먼저」 HUD = 지금 HUD + 학습 세대


def test_no_particle_right_after_a_help_placeholder() -> None:
    """v3 D19: 「{{자리}}」 바로 뒤에 조사를 붙이지 않는다(채운 글의 받침에 따라 을/를·과/와·으로/로가 어긋남). v6 두 절이 이
    규칙을 지키게 하려고 모든 쪽에서 본다(v3.1까지의 쪽은 이미 지킴)."""
    bad = {pid: re.findall(r"「\{\{[A-Za-z0-9_.]+\}\}」[가-힣]", md) for pid, (_t, md) in S.HELP_PAGES.items()}
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}

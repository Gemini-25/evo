"""T39 (6.2.0, 묶음 P): 조사 고르기 — 받침 있음·없음·ㄹ 받침·한글 아님, 규칙 문장 30개."""

from __future__ import annotations

import re

import pytest

from evolab.app import strings as S
from evolab.presets.josa import final_consonant, fill, josa, particle


@pytest.mark.parametrize(("word", "pair", "expected"), [
    # 받침 있음
    ("몸통", "이/가", "몸통이"), ("공", "을/를", "공을"), ("땅", "은/는", "땅은"), ("벽", "와/과", "벽과"),
    ("오른쪽", "(으)로", "오른쪽으로"), ("바깥", "(으)로", "바깥으로"), ("몸통", "이(가)", "몸통이"),
    # 받침 없음
    ("골대", "이/가", "골대가"), ("머리", "을/를", "머리를"), ("다리", "은/는", "다리는"), ("허벅지", "와/과", "허벅지와"),
    ("위", "(으)로", "위로"), ("아래", "(으)로", "아래로"), ("골대", "을(를)", "골대를"),
    # ㄹ 받침: 「(으)로」만 「로」, 나머지는 받침 있음과 같다
    ("왼발", "(으)로", "왼발로"), ("오른발", "이/가", "오른발이"), ("결승선", "(으)로", "결승선으로"), ("골", "(으)로", "골로"),
    ("골", "을/를", "골을"), ("팔", "와/과", "팔과"),
    # 한글이 아님: 두 꼴을 함께
    ("torso", "이/가", "torso이(가)"), ("ball_2", "을/를", "ball_2을(를)"), ("x", "(으)로", "x(으)로"), ("", "이/가", "이(가)"),
    ("3", "은/는", "3은(는)"),
    # 닫는 기호 뒤의 끝 글자
    ("「몸통」", "이/가", "「몸통」이"), ("공 (ball)", "이/가", "공 (ball)이(가)"), ("땅 ", "이/가", "땅 이"),
])
def test_josa_picks_the_particle_by_the_final_consonant(word: str, pair: str, expected: str) -> None:
    assert josa(word, pair) == expected
    assert josa(word, pair) == word + particle(word, pair)


def test_final_consonant_table() -> None:
    assert final_consonant("가") == 0 and final_consonant("각") == 1 and final_consonant("갈") == 8
    assert final_consonant("힣") == 27 and final_consonant("A") is None and final_consonant("") is None
    with pytest.raises(ValueError):
        josa("몸통", "까")


SENTENCE_CASES: list[tuple[str, dict[str, object], str]] = [
    ("distance", {"part": "몸통", "direction": "오른쪽", "per_m": 1.0}, "몸통이 오른쪽으로 간 거리만큼 점수 (1m에 1점)"),
    ("distance", {"part": "머리", "direction": "위", "per_m": 2.5}, "머리가 위로 간 거리만큼 점수 (1m에 2.5점)"),
    ("distance", {"part": "왼발", "direction": "왼쪽", "per_m": 1.0}, "왼발이 왼쪽으로 간 거리만큼 점수 (1m에 1점)"),
    ("distance", {"part": "disc", "direction": "아래", "per_m": 1.0}, "disc이(가) 아래로 간 거리만큼 점수 (1m에 1점)"),
    ("fall", {"parts": "몸통, 허벅지", "what": "땅", "root": "몸통", "height": 0.8, "penalty": 1.0},
     "몸통, 허벅지가 땅에 닿거나 몸통이 0.8m보다 낮아지면 넘어짐 (실패로 끝, 1점 감점)"),
    ("fall", {"parts": "머리", "what": "바닥", "root": "가슴", "height": 1.2, "penalty": 0.0},
     "머리가 바닥에 닿거나 가슴이 1.2m보다 낮아지면 넘어짐 (실패로 끝, 0점 감점)"),
    ("fall", {"parts": "팔", "what": "땅", "root": "허리", "height": 0.5, "penalty": 2.0},
     "팔이 땅에 닿거나 허리가 0.5m보다 낮아지면 넘어짐 (실패로 끝, 2점 감점)"),
    ("touch_ends", {"part": "몸통", "what": "땅", "outcome": "끝(실패)"}, "몸통이 땅에 닿으면 끝(실패)"),
    ("touch_ends", {"part": "차체", "what": "언덕", "outcome": "끝(실패)"}, "차체가 언덕에 닿으면 끝(실패)"),
    ("touch_ends", {"part": "공", "what": "벽", "outcome": "성공"}, "공이 벽에 닿으면 성공"),
    ("touch_penalty", {"part": "몸통", "what": "땅", "per_s": 1.0}, "몸통이 땅에 닿아 있는 동안 1초에 1점 감점"),
    ("touch_penalty", {"part": "머리", "what": "벽", "per_s": 0.5}, "머리가 벽에 닿아 있는 동안 1초에 0.5점 감점"),
    ("touch_penalty", {"part": "왼발", "what": "물", "per_s": 3.0}, "왼발이 물에 닿아 있는 동안 1초에 3점 감점"),
    ("reach", {"part": "몸통", "where": "결승선", "outcome": "성공", "bonus": 100.0},
     "몸통이 결승선에 들어가면 성공 (+100점)"),
    ("reach", {"part": "공", "where": "골", "outcome": "성공", "bonus": 50.0}, "공이 골에 들어가면 성공 (+50점)"),
    ("reach", {"part": "차", "where": "구덩이", "outcome": "끝(실패)", "bonus": 0.0}, "차가 구덩이에 들어가면 끝(실패) (+0점)"),
    ("near", {"thing": "공", "where": "골", "per_m": 1.0, "when": "가장 가까웠을 때"},
     "공이 골에 가까울수록 점수 (1m에 1점, 가장 가까웠을 때)"),
    ("near", {"thing": "선수", "where": "공", "per_m": 0.5, "when": "끝났을 때"},
     "선수가 공에 가까울수록 점수 (1m에 0.5점, 끝났을 때)"),
    ("near", {"thing": "ball", "where": "goal", "per_m": 2.0, "when": "끝났을 때"},
     "ball이(가) goal에 가까울수록 점수 (1m에 2점, 끝났을 때)"),
    ("time_limit", {"seconds": 15.0}, "제한 시간 15초"),
    ("time_limit", {"seconds": 7.5}, "제한 시간 7.5초"),
    ("survive", {"per_s": 1.0}, "오래 버틸수록 점수 (1초에 1점)"),
    ("survive", {"per_s": 0.25}, "오래 버틸수록 점수 (1초에 0.25점)"),
    ("upright", {"part": "몸통", "weight": 1.0}, "몸통이 기울수록 감점 (세기 1)"),
    ("upright", {"part": "머리", "weight": 0.5}, "머리가 기울수록 감점 (세기 0.5)"),
    ("upright", {"part": "골반", "weight": 2.0}, "골반이 기울수록 감점 (세기 2)"),
    ("save_energy", {"weight": 0.05}, "힘을 아끼면 가산점 (세기 0.05)"),
    ("save_energy", {"weight": 0.02}, "힘을 아끼면 가산점 (세기 0.02)"),
    ("out_ends", {"outcome": "끝(실패)"}, "경기장 밖으로 나가면 끝(실패)"),
    ("out_ends", {"outcome": "끝(중립)"}, "경기장 밖으로 나가면 끝(중립)"),
]


def test_thirty_rule_sentences() -> None:
    assert len(SENTENCE_CASES) == 30
    assert {kind for kind, _, _ in SENTENCE_CASES} == set(S.RULE_SENTENCES)  # 규칙 종류마다 적어도 하나
    for kind, values, expected in SENTENCE_CASES:
        got = fill(S.RULE_SENTENCES[kind], values)
        assert got == expected, (kind, got)
        assert "{" not in got and not re.search("[가-힣](이\\(가\\)|을\\(를\\)|\\(으\\)로)", got), got


def test_fill_leaves_unknown_fields_and_plain_text() -> None:
    assert fill("{a}이(가) {b}을(를)", {"a": "공"}) == "공이 {b}을(를)"
    assert fill("점수 {n}점", {"n": 3}) == "점수 3점"
    assert fill("{x}(으)로 {y}으로", {"x": "길", "y": "아래"}) == "길로 아래로"
    assert fill("{a}와(과) {b}은(는)", {"a": "벽", "b": "다리"}) == "벽과 다리는"
    assert fill("", {}) == ""


def test_same_input_same_output() -> None:
    for kind, values, _ in SENTENCE_CASES:
        assert fill(S.RULE_SENTENCES[kind], values) == fill(S.RULE_SENTENCES[kind], dict(values))

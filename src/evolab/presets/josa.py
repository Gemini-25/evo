"""조사 고르기 (6.2.0, 설계 DU 7.3·T39).

규칙 문장(strings.RULE_SENTENCES)의 빈칸에는 사용자가 고른 부품의 표시 이름이 들어간다. 「몸통이」「공이」「골대가」처럼 빈칸 바로 뒤의
조사를 그 이름의 끝 받침에 맞춰 고른다.

- josa(word, pair): pair는 「이/가」「을/를」「은/는」「와/과」처럼 「받침 있음/받침 없음」 꼴이거나 「(으)로」다.
  받침이 있으면 앞의 것, 없으면 뒤의 것. 「(으)로」는 받침이 없거나 ㄹ 받침이면 「로」, 그 밖의 받침이면 「으로」.
  끝 글자가 한글이 아니면(영문 id, 숫자) 두 꼴을 함께 적는다: 「이(가)」, 「(으)로」.
- fill(sentence, values): 문장 틀의 「{이름}이(가)」「{이름}을(를)」「{이름}(으)로」「{이름}으로」(사이에 닫는 낫표 하나가 있어도:
  「「{이름}」을(를)」)를 채운 값에 맞춰 고르고 나머지
  「{이름}」은 값 그대로 채운다. 숫자 값은 「g」 꼴(1.0 → 「1」)로 적는다.

이 파일의 글자는 코드용 이스케이프만 쓴다(화면 글은 strings.py에만: tests/test_rules.py).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_HANGUL_FIRST = 0xAC00
_HANGUL_LAST = 0xD7A3
_RIEUL = 8
"""종성 번호 8 = ㄹ 받침."""
_EU = "\uc73c"
"""「으」"""
_RO = "\ub85c"
"""「로」"""
_TRAILING = ")]}」』\"' 〉》"
"""끝 글자를 찾을 때 건너뛰는 닫는 기호(괄호, 낫표, 따옴표, 빈칸)."""

_SYLLABLE = "[\uac00-\ud7a3]"
_CLOSE = "\u300d\u300f\"')\\]"
_PAIR_RE = re.compile(r"\{(\w+)\}([" + _CLOSE + r"]?)(" + _SYLLABLE + r"\(" + _SYLLABLE + r"\)|\(" + _EU + r"\)" + _RO
                      + "|" + _EU + _RO + ")")
"""「{이름}」 바로 뒤(닫는 낫표·따옴표 하나를 사이에 둘 수 있음)의 조사 꼴: 「이(가)」 같은 꼴, 「(으)로」, 「으로」."""
_FIELD_RE = re.compile(r"\{(\w+)\}")


def _last_syllable(word: str) -> int | None:
    """끝 글자(닫는 기호를 건너뛴)의 한글 음절 번호(0부터), 한글이 아니면 None."""
    text = str(word).rstrip(_TRAILING)
    if not text:
        return None
    code = ord(text[-1])
    if _HANGUL_FIRST <= code <= _HANGUL_LAST:
        return code - _HANGUL_FIRST
    return None


def final_consonant(word: str) -> int | None:
    """끝 글자의 종성 번호(0 = 받침 없음, 8 = ㄹ). 한글이 아니면 None."""
    n = _last_syllable(word)
    return None if n is None else n % 28


_WITH_FINAL = frozenset({"\uc774", "\uc744", "\uc740", "\uacfc", "\uc544"})
"""받침 뒤에 붙는 꼴: 「이」「을」「은」「과」「아」. 「와/과」처럼 받침 없는 꼴을 앞에 적은 짝도 알아본다."""


def _split(pair: str) -> tuple[str, str, bool]:
    """(받침 있을 때, 받침 없을 때, 「(으)로」 꼴인가)."""
    if pair in ("(" + _EU + ")" + _RO, _EU + _RO):
        return _EU + _RO, _RO, True
    m = re.fullmatch(r"([^/()]+)/([^/()]+)", pair) or re.fullmatch(r"([^/()]+)\(([^/()]+)\)", pair)
    if not m:
        raise ValueError(f"unknown josa pair: {pair!r}")
    a, b = m.group(1), m.group(2)
    if a not in _WITH_FINAL and b in _WITH_FINAL:
        a, b = b, a
    return a, b, False


def _bracket(pair: str, with_final: str, without: str) -> str:
    """한글이 아닌 낱말 뒤: 두 꼴을 적은 순서 그대로 「이(가)」「와(과)」."""
    first = pair.split("/", 1)[0] if "/" in pair else pair.split("(", 1)[0]
    return f"{with_final}({without})" if first == with_final else f"{without}({with_final})"


def particle(word: str, pair: str) -> str:
    """word 뒤에 붙일 조사만. josa(word, pair) == word + particle(word, pair)."""
    with_final, without, is_ro = _split(pair)
    jong = final_consonant(word)
    if jong is None:
        return "(" + _EU + ")" + _RO if is_ro else f"{_bracket(pair, with_final, without)}"
    if is_ro:
        return _RO if jong in (0, _RIEUL) else _EU + _RO
    return with_final if jong else without


def josa(word: str, pair: str) -> str:
    """josa("몸통", "이/가") -> "몸통이", josa("공", "을/를") -> "공을", josa("골대", "이/가") -> "골대가",
    josa("바깥", "(으)로") -> "바깥으로", josa("오른쪽", "(으)로") -> "오른쪽으로", josa("왼발", "(으)로") -> "왼발로",
    josa("torso", "이/가") -> "torso이(가)"."""
    return f"{word}{particle(word, pair)}"


def _text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def fill(sentence: str, values: Mapping[str, Any]) -> str:
    """문장 틀을 채운다. 빈칸 바로 뒤 조사는 채운 값의 받침에 맞춘다. 값이 없는 빈칸은 그대로 둔다."""

    def pair(m: re.Match[str]) -> str:
        name, close, form = m.group(1), m.group(2), m.group(3)
        if name not in values:
            return m.group(0)
        word = _text(values[name])
        return word + close + particle(word, form)

    out = _PAIR_RE.sub(pair, sentence)
    return _FIELD_RE.sub(lambda m: _text(values[m.group(1)]) if m.group(1) in values else m.group(0), out)


__all__ = ["final_consonant", "fill", "josa", "particle"]

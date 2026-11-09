"""도움말 쪽 읽기: 자리 채우기, 제목, 찾기 (v3 설계 2.5, 4.4, D6, D19).

도움말 글은 strings.py `HELP_PAGES`에 있다(`id → (제목, Markdown)`, 사전 순서 = 쪽 목록 순서). 글 안의 `{{NAME}}`은
`strings.NAME`, `{{NAME.key}}`는 `strings.NAME[key]`로 바꾼다. 모르는 자리는 그대로 두고 테스트(T11)가 잡는다.
채운 글은 처음 읽을 때 한 번 만들어 기억한다(strings.py는 실행 중에 바뀌지 않는다).

이 모듈은 Qt를 불러오지 않는다(자체 점검·명령줄에서도 쪽을 확인할 수 있게).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Final

from evolab.app import strings as S

PAGES: Final[tuple[str, ...]] = tuple(S.HELP_PAGES)
"""쪽 id(HELP_PAGES 순서 = 쪽 목록 순서)."""

HOME: Final = "start"
"""처음 쪽(도움말 창의 「처음」 단추, 모르는 쪽 대신)."""

GLOSSARY: Final = "words"
"""낱말 풀이 쪽(찾은 곳이 없을 때 권함)."""

PLACEHOLDER: Final = re.compile(r"\{\{([A-Z0-9_]+)(?:\.([a-z0-9_]+))?\}\}")
"""글 안의 자리: {{NAME}} 또는 {{NAME.key}}(설계 2.5)."""

LINK: Final = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")
"""Markdown 링크 [글](주소)."""

HEADING: Final = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")

SNIPPET: Final = 30
"""찾기 결과에서 맞은 곳 앞뒤로 보여 줄 글자 수(설계 4.4)."""

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_LIST = re.compile(r"^(\s*)(?:[-*+]|\d+\.)\s+")
_SPACES = re.compile(r"\s+")


def fill(text: str) -> str:
    """자리를 strings.py 값으로 바꾼다. 모르는 자리(이름이 없거나, 사전이 아니거나, 키가 없음)는 그대로 둔다."""

    def sub(m: re.Match[str]) -> str:
        name, key = m.group(1), m.group(2)
        value = getattr(S, name, None)
        if key is not None:
            if isinstance(value, dict) and key in value and isinstance(value[key], str):
                return value[key]
            return m.group(0)
        return value if isinstance(value, str) else m.group(0)

    return PLACEHOLDER.sub(sub, text)


def has_page(page: str) -> bool:
    return page in S.HELP_PAGES


def title(page: str) -> str:
    """쪽 제목(쪽 목록에 보이는 이름). 모르는 쪽이면 KeyError."""
    return fill(S.HELP_PAGES[page][0])


@lru_cache(maxsize=None)
def load_page(page: str) -> str:
    """자리를 채운 Markdown. 모르는 쪽이면 KeyError."""
    return fill(S.HELP_PAGES[page][1])


def unknown_placeholders(page: str) -> list[str]:
    """채운 뒤에도 남은 자리(있으면 strings.py와 도움말 글이 어긋남)."""
    return [m.group(0) for m in PLACEHOLDER.finditer(load_page(page))]


@lru_cache(maxsize=None)
def _headings(page: str) -> tuple[str, ...]:
    return tuple(m.group(2) for m in map(HEADING.match, load_page(page).splitlines()) if m)


def headings(page: str) -> list[str]:
    """쪽의 제목 글(# 쪽 제목, ## 절 제목) 순서대로. 링크 `evolab://help/<쪽>#<제목>`과 F1이 이 글로 제목을 찾는다."""
    return list(_headings(page))


def links(page: str) -> list[str]:
    """쪽 안의 링크 주소(나온 순서)."""
    return [m.group(2) for m in LINK.finditer(load_page(page))]


def _plain_line(line: str) -> str:
    m = HEADING.match(line)
    if m:
        line = m.group(2)
    line = _LIST.sub(r"\1", line)
    line = LINK.sub(r"\1", line)
    return _BOLD.sub(r"\1", line)


@lru_cache(maxsize=None)
def plain_text(page: str) -> str:
    """찾기용 글: Markdown 표시(제목 #, 목록 -, 굵게 **, 링크 주소)를 뺀 글. 줄은 빈칸 하나로 잇는다."""
    lines = (_plain_line(ln) for ln in load_page(page).splitlines())
    return _SPACES.sub(" ", " ".join(ln for ln in lines if ln.strip())).strip()


@lru_cache(maxsize=None)
def _folded(page: str) -> str:
    return plain_text(page).casefold()


def search(query: str) -> list[tuple[str, int, str]]:
    """모든 쪽에서 query를 찾는다(대소문자 무시 부분 일치, 앞뒤 빈칸 무시).

    결과 = (쪽, 맞은 수, 첫 맞은 곳 앞뒤 SNIPPET자 글), 맞은 수가 많은 쪽부터(같으면 쪽 순서). 빈 query는 빈 목록.
    """
    q = _SPACES.sub(" ", query).strip().casefold()
    if not q:
        return []
    found: list[tuple[int, int, str, str]] = []
    for order, page in enumerate(PAGES):
        text = plain_text(page)
        folded = _folded(page)
        count = folded.count(q)
        if count == 0:
            continue
        at = folded.find(q)
        start = max(0, at - SNIPPET)
        end = min(len(text), at + len(q) + SNIPPET)
        found.append((-count, order, page, text[start:end].strip()))
    found.sort()
    return [(page, -neg, snippet) for neg, _order, page, snippet in found]


def total_matches(results: list[tuple[str, int, str]]) -> int:
    """찾은 곳 수(쪽마다 맞은 수의 합: 「찾은 곳 n군데」)."""
    return sum(count for _page, count, _snippet in results)


__all__ = ["GLOSSARY", "HOME", "PAGES", "PLACEHOLDER", "SNIPPET", "fill", "has_page", "headings", "links", "load_page",
           "plain_text", "search", "title", "total_matches", "unknown_placeholders"]

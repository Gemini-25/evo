"""틀 보관함 (6.2.0, 설계 DU 4.2, Qt 없음).

paths.template_dirs()의 폴더(내장 assets/templates → 사용자 문서\\진화실험실\\templates)에서 `*.evotpl`을 읽는다. 내장 틀이 먼저이고,
사용자 폴더에 같은 id가 있으면 그것은 「내 틀」로 따로 보인다(get은 내장 것을 돌려준다). 읽지 못한 사용자 틀은 problems()에
까닭과 함께 남고 보관함은 계속 뜬다. 더 새 판의 틀(TPL_ERR_NEWER)도 그 틀만 건너뛴다.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from evolab import paths
from evolab.presets.template import Template, TemplateError, load_template, thumbnail

TEMPLATE_SUFFIX = ".evotpl"


class Library:
    """틀 보관함. dirs의 첫 폴더는 내장 틀(source "builtin"), 나머지는 사용자 틀(source "user")."""

    def __init__(self, dirs: Sequence[Path], *, builtin_count: int = 1) -> None:
        self.dirs = [Path(d) for d in dirs]
        self._all: list[Template] = []
        self._problems: list[tuple[Path, str]] = []
        for k, folder in enumerate(self.dirs):
            source = "builtin" if k < builtin_count else "user"
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*" + TEMPLATE_SUFFIX)):
                try:
                    self._all.append(load_template(path, source=source))
                except TemplateError as exc:
                    if source == "builtin":
                        raise  # 내장 틀이 깨진 것은 프로그램 결함(자체 점검이 그 이름으로 알린다)
                    self._problems.append((path, str(exc)))

    def all(self) -> list[Template]:
        """모든 틀(내장 먼저, 폴더 안에서는 파일 이름순)."""
        return list(self._all)

    def templates(self, kind: str, world: str) -> list[Template]:
        """그 종류(body·prop·arena)에서 그 세계(side·top)에 쓸 수 있는 틀(틀의 world가 같거나 any)."""
        return [t for t in self._all if t.kind == kind and t.world in (world, "any")]

    def get(self, template_id: str) -> Template:
        """id의 틀. 내장 것이 먼저다. 없으면 KeyError."""
        for t in self._all:
            if t.id == template_id and t.source == "builtin":
                return t
        for t in self._all:
            if t.id == template_id:
                return t
        raise KeyError(template_id)

    def has(self, template_id: str) -> bool:
        return any(t.id == template_id for t in self._all)

    def problems(self) -> list[tuple[Path, str]]:
        """읽지 못한 사용자 틀과 까닭(한국어)."""
        return list(self._problems)


_DEFAULT: Library | None = None
_DEFAULT_DIRS: list[Path] | None = None


def default_library() -> Library:
    """paths.template_dirs()로 만든 보관함 하나(폴더가 바뀌면 다시 만든다)."""
    global _DEFAULT, _DEFAULT_DIRS
    dirs = paths.template_dirs()
    if _DEFAULT is None or _DEFAULT_DIRS != dirs:
        _DEFAULT = Library(dirs)
        _DEFAULT_DIRS = dirs
    return _DEFAULT


def reset_default_library() -> None:
    """다음 default_library()가 폴더를 다시 읽게 한다(사용자 틀을 더한 뒤)."""
    global _DEFAULT, _DEFAULT_DIRS
    _DEFAULT = None
    _DEFAULT_DIRS = None


__all__ = ["TEMPLATE_SUFFIX", "Library", "default_library", "reset_default_library", "thumbnail"]

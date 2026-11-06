"""환경 파일 읽기, 쓰기, 버전 변환.

- 파일은 UTF-8 JSON이다.
- version이 지금보다 낮으면 MIGRATIONS를 차례로 적용하고, 높으면 '새 버전에서 만든 파일'로 거부한다.
- 저장은 기본값까지 모두 쓰는 정규 형식이다. 같은 내용이면 늘 같은 글자열이 나온다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel, ValidationError

from evolab.app import strings as S
from evolab.spec import model as m
from evolab.spec.compiled import EnvFileError, EnvSpec, Issue

Migration = Callable[[dict[str, Any]], dict[str, Any]]

# MIGRATIONS[n]은 version n 파일을 version n+1로 바꾼다.
MIGRATIONS: dict[int, Migration] = {}


def migrate(data: dict[str, Any]) -> dict[str, Any]:
    version = data.get("version", m.CURRENT_VERSION)
    if not isinstance(version, int) or version < 1:
        raise EnvFileError([Issue("error", "version", S.SPEC_BAD_VERSION.format(version=version))])
    if version > m.CURRENT_VERSION:
        raise EnvFileError([Issue("error", "version", S.SPEC_NEWER_VERSION.format(version=version))])
    while version < m.CURRENT_VERSION:
        data = MIGRATIONS[version](data)
        version += 1
        data["version"] = version
    return data


def parse_env(data: dict[str, Any] | str | bytes) -> m.EnvFile:
    """JSON 글자열이나 사전을 검증된 모델로 바꾼다(구조 검사만, 참조와 수식은 compile_env)."""
    if isinstance(data, (str, bytes)):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise EnvFileError([Issue("error", "", S.SPEC_BAD_JSON.format(line=exc.lineno, col=exc.colno))]) from None
    if not isinstance(data, dict):
        raise EnvFileError([Issue("error", "", S.SPEC_NOT_OBJECT)])
    data = migrate(dict(data))
    try:
        return m.EnvFile.model_validate(data)
    except ValidationError as exc:
        raise EnvFileError(translate_validation_error(exc)) from None


def compile_env(data: dict[str, Any] | str | bytes | m.EnvFile) -> EnvSpec:
    """읽고, 검사하고, 컴파일한다."""
    env = data if isinstance(data, m.EnvFile) else parse_env(data)
    return EnvSpec.build(env)


def load_env(path: str | Path) -> EnvSpec:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise EnvFileError([Issue("error", "", S.SPEC_READ_FAILED.format(path=p, reason=exc.strerror or exc))]) from None
    return compile_env(text)


def dump_env(env: m.EnvFile) -> str:
    """정규 JSON. 기본값까지 모두 쓰고, 칸 순서는 모델 순서를 따른다."""
    data = env.model_dump(mode="json", by_alias=True)
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def save_env(env: m.EnvFile, path: str | Path) -> None:
    Path(path).write_text(dump_env(env), encoding="utf-8")


def env_hash(env: m.EnvFile) -> str:
    """에디터 전용 칸을 뺀 내용의 해시. 실행 폴더의 환경 버전을 가리킬 때 쓴다."""
    data = env.model_dump(mode="json", by_alias=True, exclude={"editor"})
    text = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# pydantic 오류를 한국어로
# ---------------------------------------------------------------------------

_MESSAGES: dict[str, str] = {
    "missing": S.SPEC_ERR_MISSING,
    "extra_forbidden": S.SPEC_ERR_EXTRA,
    "literal_error": S.SPEC_ERR_LITERAL,
    "union_tag_invalid": S.SPEC_ERR_KIND,
    "union_tag_not_found": S.SPEC_ERR_KIND_MISSING,
    "string_pattern_mismatch": S.SPEC_ERR_PATTERN,
    "string_too_long": S.SPEC_ERR_TOO_LONG,
    "string_too_short": S.SPEC_ERR_TOO_SHORT,
    "too_short": S.SPEC_ERR_LIST_SHORT,
    "too_long": S.SPEC_ERR_LIST_LONG,
    "greater_than": S.SPEC_ERR_RANGE,
    "greater_than_equal": S.SPEC_ERR_RANGE,
    "less_than": S.SPEC_ERR_RANGE,
    "less_than_equal": S.SPEC_ERR_RANGE,
    "int_parsing": S.SPEC_ERR_INT,
    "int_type": S.SPEC_ERR_INT,
    "int_from_float": S.SPEC_ERR_INT,
    "float_parsing": S.SPEC_ERR_NUMBER,
    "float_type": S.SPEC_ERR_NUMBER,
    "bool_parsing": S.SPEC_ERR_BOOL,
    "bool_type": S.SPEC_ERR_BOOL,
    "string_type": S.SPEC_ERR_STRING,
    "list_type": S.SPEC_ERR_LIST,
    "tuple_type": S.SPEC_ERR_LIST,
    "dict_type": S.SPEC_ERR_OBJECT,
    "model_type": S.SPEC_ERR_OBJECT,
    "model_attributes_type": S.SPEC_ERR_OBJECT,
}

# 합집합 타입의 갈래 이름. 경로에서 빼고 한 칸의 오류로 합친다.
_UNION_BRANCHES = ("float", "constrained-str", "str", "int", "bool", "list", "tuple", "none")


def _kind_tags() -> frozenset[str]:
    """판별 필드(kind)의 값 전부. pydantic은 오류 경로에 이 값을 끼워 넣는다(shapes.0.box.w)."""
    tags: set[str] = set()
    for obj in vars(m).values():
        if isinstance(obj, type) and issubclass(obj, BaseModel) and "kind" in obj.model_fields:
            ann = obj.model_fields["kind"].annotation
            tags.update(a for a in get_args(ann) if isinstance(a, str))
    return frozenset(tags)


_KIND_TAGS = _kind_tags()


def _format_loc(loc: tuple[int | str, ...]) -> tuple[str, bool]:
    parts: list[str] = []
    in_union = False
    prev: int | str | None = None
    for item in loc:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        elif item in _UNION_BRANCHES or item.startswith(("function-", "list[", "tuple[", "literal[", "union[")):
            in_union = True
        elif item in _KIND_TAGS and (isinstance(prev, int) or prev == "shape"):
            pass
        else:
            parts.append(("." if parts else "") + item)
        prev = item
    return "".join(parts), in_union


def translate_validation_error(exc: ValidationError) -> list[Issue]:
    issues: list[Issue] = []
    seen: set[str] = set()
    for err in exc.errors(include_url=False):
        where, in_union = _format_loc(tuple(err["loc"]))
        if in_union:
            if where in seen:
                continue
            message = S.SPEC_ERR_NUM_OR_FORMULA
        else:
            template = _MESSAGES.get(err["type"], S.SPEC_ERR_GENERIC)
            ctx = err.get("ctx") or {}
            message = template.format(
                expected=ctx.get("expected", ""),
                limit=ctx.get("gt", ctx.get("ge", ctx.get("lt", ctx.get("le", "")))),
                min=ctx.get("min_length", ""),
                max=ctx.get("max_length", ""),
                tag=ctx.get("tag", ""),
                type=err["type"],
            )
        seen.add(where)
        issues.append(Issue("error", where, message))
    return issues

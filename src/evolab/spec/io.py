"""환경 파일 읽기, 쓰기, 버전 변환.

- 파일은 UTF-8 JSON이다.
- version이 지금보다 낮으면 MIGRATIONS를 차례로 적용하고, 높으면 '새 버전에서 만든 파일'로 거부한다.
- 저장은 기본값까지 모두 쓰는 정규 형식이다. 같은 내용이면 늘 같은 글자열이 나온다.
- 저장과 해시는 그 내용을 담을 수 있는 가장 낮은 형식으로 한다(v2 설계 D2): 칸·종류·값마다 생긴 형식을
  FIELD_SINCE·KIND_SINCE·VALUE_SINCE에 적어 두고, 쓰인 것 중 가장 큰 형식(feature_version)보다 새 칸(정의상 기본값)은
  글에서 지운다. 그래서 v1 기능만 쓴 파일은 v1.x 프로그램이 쓴 글·해시와 글자까지 같다(챔피언의 env_version, 판정,
  「내 최고 기록」, 경주가 그대로 이어진다). 새 칸은 반드시 다음 형식 번호로 등록한다(tests/test_format_v2.py가 지킴).
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel, ValidationError

from evolab.app import strings as S
from evolab.spec import model as m
from evolab.spec.compiled import EnvFileError, EnvSpec, Issue

Migration = Callable[[dict[str, Any]], dict[str, Any]]


def _fields_added_only(data: dict[str, Any]) -> dict[str, Any]:
    """칸·종류·값을 더하기만 한 형식 변경. 새 칸은 모두 기본값이 있고 기본값이면 옛 뜻과 같아 사전을 그대로 쓴다."""
    return data


# MIGRATIONS[n]은 version n 파일을 version n+1로 바꾼다.
MIGRATIONS: dict[int, Migration] = {
    1: _fields_added_only,  # 1 -> 2 (v2.0): 정해진 움직임, 장애물 줄, 점프, 공기역학, 날개 모양, 모양별 밀도
    2: _fields_added_only,  # 2 -> 3 (v2.1): 물체의 있음 칸(exists)
}

# ---------------------------------------------------------------------------
# 형식 번호: 칸·종류·값이 생긴 형식 (설계 2.2)
# ---------------------------------------------------------------------------

FIELD_SINCE: dict[tuple[type[BaseModel], str], int] = {
    (m.Box, "density"): 2, (m.Circle, "density"): 2, (m.Poly, "density"): 2, (m.Radial, "density"): 2,
    (m.Body, "motion"): 2,
    (m.Body, "exists"): 3,  # v2.1
}
"""옛 모델에 더한 칸(모델, 파이썬 칸 이름) -> 생긴 형식. 기본값이 아니면 그 형식이 필요하고, 그보다 낮은 형식 글에서는
지운다. 새로 더하는 칸은 반드시 다음 형식 번호로(이미 나간 형식에 더하면 그 형식 파일들의 해시가 바뀐다)."""
KIND_SINCE: dict[type[BaseModel], int] = {
    m.Wing: 2, m.Jump: 2, m.Aero: 2, m.ObstacleGen: 2, m.PathMotion: 2, m.WaveMotion: 2,
}
"""새 모델(종류) -> 생긴 형식. 파일에 하나라도 있으면 그 형식이다. 그 모델의 칸은 따로 등록하지 않는다."""
VALUE_SINCE: dict[tuple[type[BaseModel], str, object], int] = {
    (m.Body, "type", "kinematic"): 2,
}
"""옛 칸의 새 값(모델, 칸, 값) -> 생긴 형식."""


@functools.cache
def _model_fields(cls: type[BaseModel]) -> tuple[str, ...]:
    """모델 인스턴스를 담을 수 있는 칸(형에 모델이 들어 있는 칸). 숫자 목록 같은 칸은 걷지 않는다."""
    def has_model(tp: object) -> bool:
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            return True
        return any(has_model(a) for a in get_args(tp))

    return tuple(name for name, f in cls.model_fields.items() if has_model(f.annotation))


_FieldRule = tuple[str, str, int, Any]
"""(파이썬 칸 이름, 파일 칸 이름, 생긴 형식, 기본값)"""
_ValueRule = tuple[str, object, int]
"""(파이썬 칸 이름, 값, 생긴 형식)"""


@functools.cache
def _rules(cls: type[BaseModel]) -> tuple[int, tuple[_FieldRule, ...], tuple[_ValueRule, ...]]:
    """(종류 형식, 칸 규칙들, 값 규칙들). 부모 모델에 등록한 규칙도 따른다."""
    kind = max((v for c, v in KIND_SINCE.items() if issubclass(cls, c)), default=1)
    fields = tuple(
        (name, f.alias or name, since, f.get_default(call_default_factory=True))
        for (c, name), since in FIELD_SINCE.items() if issubclass(cls, c)
        for f in (cls.model_fields[name],)
    )
    values = tuple((name, value, since) for (c, name, value), since in VALUE_SINCE.items() if issubclass(cls, c))
    return kind, fields, values


def _children(value: Any) -> Iterator[Any]:
    """value 안의 모델·목록·사전 값(모델 칸만 따라간다)."""
    if isinstance(value, BaseModel):
        for name in _model_fields(type(value)):
            yield getattr(value, name)
    elif isinstance(value, (list, tuple)):
        yield from value
    elif isinstance(value, dict):
        yield from value.values()


def _models(value: Any) -> Iterator[BaseModel]:
    """나무 안의 모든 모델 인스턴스(자기 자신 포함)."""
    if isinstance(value, BaseModel):
        yield value
    for child in _children(value):
        if isinstance(child, (BaseModel, list, tuple, dict)):
            yield from _models(child)


def feature_version(env: m.EnvFile) -> int:
    """이 내용을 담을 수 있는 가장 낮은 형식 번호(최소 1).

    모델 나무를 걸으며 쓰인 KIND_SINCE 종류, 기본값이 아닌 FIELD_SINCE 칸, 쓰인 VALUE_SINCE 값의 형식 중 가장 큰 것.
    v2 기능을 하나도 안 쓰면 1이라 v1.x 프로그램도 그 파일을 연다.
    """
    version = 1
    for inst in _models(env):
        kind, fields, values = _rules(type(inst))
        version = max(version, kind)
        for name, _key, since, default in fields:
            if since > version and getattr(inst, name) != default:
                version = since
        for name, value, since in values:
            if since > version and getattr(inst, name) == value:
                version = since
    return version


def _strip_newer(value: Any, data: Any, version: int) -> None:
    """모델 나무와 그 덤프(사전)를 함께 걸으며 version보다 새 칸을 사전에서 지운다."""
    if isinstance(value, BaseModel):
        if not isinstance(data, dict):
            return
        for _name, key, since, _default in _rules(type(value))[1]:
            if since > version:
                data.pop(key, None)
        fields = type(value).model_fields
        for name in _model_fields(type(value)):
            key = fields[name].alias or name
            if key in data:
                _strip_newer(getattr(value, name), data[key], version)
    elif isinstance(value, (list, tuple)) and isinstance(data, list):
        for v, d in zip(value, data, strict=False):
            _strip_newer(v, d, version)
    elif isinstance(value, dict) and isinstance(data, dict):
        for k, v in value.items():
            if k in data:
                _strip_newer(v, data[k], version)


def canonical_dict(env: m.EnvFile, exclude: set[str] | None = None) -> dict[str, Any]:
    """저장·해시용 사전: model_dump(mode="json", by_alias=True, exclude=exclude)에서 FIELD_SINCE 형식이
    feature_version보다 큰 칸을 지우고 version = feature_version.

    지우는 칸은 정의상 모두 기본값이다(기본값이 아니면 feature_version이 그만큼 크다). 칸 순서는 모델 순서 그대로다.
    """
    data = env.model_dump(mode="json", by_alias=True, exclude=exclude)
    version = feature_version(env)
    _strip_newer(env, data, version)
    data["version"] = version
    return data


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
    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise EnvFileError([Issue("error", "", S.SPEC_NOT_UTF8)]) from None
    if isinstance(data, str):
        try:
            data = json.loads(data, parse_constant=_reject_constant)
        except json.JSONDecodeError as exc:
            raise EnvFileError([Issue("error", "", S.SPEC_BAD_JSON.format(line=exc.lineno, col=exc.colno))]) from None
        except ValueError as exc:  # NaN / Infinity
            raise EnvFileError([Issue("error", "", S.SPEC_JSON_NAN.format(value=exc))]) from None
        except RecursionError:
            raise EnvFileError([Issue("error", "", S.SPEC_TOO_DEEP)]) from None
    if not isinstance(data, dict):
        raise EnvFileError([Issue("error", "", S.SPEC_NOT_OBJECT)])
    data = migrate(dict(data))
    try:
        return m.EnvFile.model_validate(data)
    except ValidationError as exc:
        raise EnvFileError(translate_validation_error(exc)) from None
    except RecursionError:
        raise EnvFileError([Issue("error", "", S.SPEC_TOO_DEEP)]) from None


def _reject_constant(name: str) -> float:
    raise ValueError(name)


def compile_env(data: dict[str, Any] | str | bytes | m.EnvFile) -> EnvSpec:
    """읽고, 검사하고, 컴파일한다."""
    env = data if isinstance(data, m.EnvFile) else parse_env(data)
    return EnvSpec.build(env)


def load_env(path: str | Path) -> EnvSpec:
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise EnvFileError([Issue("error", "", S.SPEC_READ_FAILED.format(path=p, reason=exc.strerror or exc))]) from None
    return compile_env(raw)


def dump_env(env: m.EnvFile) -> str:
    """정규 JSON. 기본값까지 모두 쓰고, 칸 순서는 모델 순서를 따른다. 형식은 내용을 담을 수 있는 가장 낮은 것."""
    data = canonical_dict(env)
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def save_env(env: m.EnvFile, path: str | Path) -> None:
    Path(path).write_text(dump_env(env), encoding="utf-8")


def env_hash(env: m.EnvFile) -> str:
    """에디터 전용 칸을 뺀 내용의 해시. 실행 폴더의 환경 버전을 가리킬 때 쓴다.

    가장 낮은 형식의 정규 사전으로 계산하므로 v1 기능만 쓴 파일은 v1.x와 같은 해시다.
    """
    data = canonical_dict(env, exclude={"editor"})
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
    "finite_number": S.SPEC_ERR_FINITE,
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
# 판별 필드(kind)로 고르는 목록 칸. pydantic은 이 칸들의 오류 경로에 kind 값을 끼워 넣는다.
_TAGGED_LISTS = frozenset({"shapes", "actuators", "sensors", "generators", "blocks"})


def _kind_tags() -> frozenset[str]:
    """판별 필드(kind)의 값 전부. pydantic은 오류 경로에 이 값을 끼워 넣는다(shapes.0.box.w)."""
    tags: set[str] = set()
    for obj in vars(m).values():
        if isinstance(obj, type) and issubclass(obj, BaseModel) and "kind" in obj.model_fields:
            ann = obj.model_fields["kind"].annotation
            tags.update(a for a in get_args(ann) if isinstance(a, str))
    return frozenset(tags)


_KIND_TAGS = _kind_tags()


def _format_loc(loc: tuple[int | str, ...]) -> tuple[str, str | None]:
    """(경로, 합집합 갈래 종류). 갈래 종류는 'num'(숫자·수식 칸), 'names'(이름 목록이거나 all) 또는 None."""
    parts: list[str] = []
    branch: str | None = None
    prev: int | str | None = None
    prev2: int | str | None = None
    for item in loc:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        elif item in _UNION_BRANCHES or item.startswith(("function-", "list[", "tuple[", "literal[", "union[")):
            if branch is None:
                branch = "names" if item.startswith(("list[", "literal[")) else "num"
        elif item in _KIND_TAGS and (
            (isinstance(prev, int) and prev2 in _TAGGED_LISTS) or prev in ("shape", "motion")
        ):
            pass
        else:
            parts.append(("." if parts else "") + item)
        prev2, prev = prev, item
    return "".join(parts), branch


def translate_validation_error(exc: ValidationError) -> list[Issue]:
    issues: list[Issue] = []
    seen: set[str] = set()
    for err in exc.errors(include_url=False):
        where, branch = _format_loc(tuple(err["loc"]))
        if branch is not None and err["type"] != "finite_number":
            if where in seen:
                continue
            message = S.SPEC_ERR_NUM_OR_FORMULA if branch == "num" else S.SPEC_ERR_NAMES_OR_ALL
        else:
            template = _MESSAGES.get(err["type"], S.SPEC_ERR_GENERIC)
            ctx = err.get("ctx") or {}
            expected = str(ctx.get("expected") or ctx.get("expected_tags") or "").replace(" or ", ", ")
            message = template.format(
                expected=expected,
                limit=ctx.get("gt", ctx.get("ge", ctx.get("lt", ctx.get("le", "")))),
                min=ctx.get("min_length", ""),
                max=ctx.get("max_length", ""),
                tag=ctx.get("tag", ""),
                type=err["type"],
            )
        seen.add(where)
        issues.append(Issue("error", where, message))
    return issues

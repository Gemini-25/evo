"""속성 폼 엔진: 환경 파일 모델(spec/model.py)의 pydantic 정의에서 편집 칸을 만든다 (명세: 화면과 조작 > 에디터).

- 폼은 문서(EnvDocument)의 사전 한 군데(JSON 경로)를 보여 주고, 고칠 때는 늘 문서의 편집 함수를 거친다.
  자기 사본은 두지 않는다. 같은 칸을 이어서 타이핑하면 doc.set_value가 되돌리기 한 단계로 합친다.
- 칸 위젯은 모델의 타입으로 고른다: 숫자·수식(Num), 숫자(float), 정수, 참·거짓, 고르기(Literal), 참조(Ref),
  두 수(Vec2), 쉼표 목록, 표(꼭짓점 목록), 키·값 표(dict), 묶음(중첩 모델), 종류를 고르는 합집합(모양, 센서,
  구동기, 점수 블록, 생성기).
- 문서가 바뀌면(doc.changed) 위젯을 다시 만들지 않고 값만 고친다. 타이핑 중인 칸은 건드리지 않는다. 구조(종류,
  모양 개수, 있는지 없는지)가 바뀔 때만 다시 만든다.
- 검사 결과(doc.issues)는 경로(where)의 앞부분이 칸의 경로와 같으면 그 칸에 빨간 표시와 풍선 도움말로 보인다.
- 에이전트 쪽 숫자 칸(에이전트 물체, 관절, 구동기, 센서)에는 '진화' 체크박스를 둔다. 켜면 그 값이 몸 유전자
  (evolve 매개변수)가 되고 최솟값·최댓값을 적는다.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import inspect
import math
import re
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from annotated_types import Ge, Gt, Le, Lt, MaxLen, MinLen
from pydantic import BaseModel, StringConstraints, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from evolab.app import strings as S
from evolab.app.editor.document import EnvDocument, JsonPath, Ref, get_path, set_path
from evolab.spec import model as m
from evolab.spec import symbols as sy
from evolab.spec.compiled import Issue

_MISSING = object()
_NONE = type(None)
_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REF_RE = re.compile(r"^[A-Za-z_가-힣][A-Za-z0-9_.가-힣]*$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z_가-힣][A-Za-z0-9_가-힣]*$")
_PARAM_REF_RE = re.compile(r"=\s*p\.([A-Za-z_][A-Za-z0-9_]*)\s*")
ISSUE_COLOR = "#d9534f"

SECTION_TYPES: dict[str, Any] = {
    "generators": m.Generator,
    "bodies": m.Body,
    "joints": m.Joint,
    "actuators": m.Actuator,
    "sensors": m.Sensor,
    "regions": m.Region,
    "checkpoints": m.Checkpoints,
    "events": m.Event,
    "randomize": m.RandVar,
    "params": m.Param,
    "scoring.blocks": m.Block,
    "termination": m.End,
}
"""구역 이름 -> 그 목록 한 칸의 타입(모델 또는 종류로 고르는 합집합)."""


# ---------------------------------------------------------------------------
# 경로와 검사 결과
# ---------------------------------------------------------------------------


def path_to_where(path: JsonPath) -> str:
    """("bodies", 1, "shapes", 0, "h") -> "bodies[1].shapes[0].h" (검사 결과의 where 형식)."""
    out = ""
    for key in path:
        if isinstance(key, int):
            out += f"[{key}]"
        else:
            out += ("." if out else "") + str(key)
    return out


def issues_at(issues: list[Issue], path: JsonPath, exact: bool = False) -> list[Issue]:
    """이 경로(또는 그 안쪽)를 가리키는 검사 결과."""
    w = path_to_where(path)
    if exact:
        return [i for i in issues if i.where == w]
    if not w:
        return list(issues)
    return [i for i in issues if i.where == w or i.where.startswith((w + ".", w + "["))]


# ---------------------------------------------------------------------------
# 타입 읽기
# ---------------------------------------------------------------------------

_SC_ID = get_args(m.Id)[1]
_SC_REF = get_args(m.Ref)[1]
_SC_FORMULA = get_args(m.Formula)[1]
_SC_EXPR = get_args(m.Expr)[1]
_SC_LABEL = get_args(m.Label)[1]


@dataclass(frozen=True)
class TypeInfo:
    """칸 하나의 타입을 폼이 쓰기 좋게 줄인 것.

    kind: num(숫자·수식) float int bool literal ref id label expr text vec(고정 길이 수 묶음) line(두 점)
    vec_list num_list lines list(쉼표 목록) literal_list names_or_all(목록 또는 "all") dict model union
    union_list model_list skip(폼에 안 보임)
    """

    kind: str
    optional: bool = False
    formula: bool = False
    choices: tuple[Any, ...] = ()
    item: TypeInfo | None = None
    key: TypeInfo | None = None
    size: int = 0
    model: type[BaseModel] | None = None
    models: tuple[type[BaseModel], ...] = ()
    low: float | None = None
    high: float | None = None
    low_open: bool = False
    min_len: int = 0
    max_len: int | None = None


def _unwrap(ann: Any) -> tuple[Any, list[Any]]:
    meta: list[Any] = []
    while get_origin(ann) is Annotated:
        args = get_args(ann)
        ann = args[0]
        meta.extend(args[1:])
    return ann, meta


def _flat_meta(meta: list[Any]) -> list[Any]:
    out: list[Any] = []
    for x in meta:
        if isinstance(x, FieldInfo):
            out.extend(x.metadata)
        else:
            out.append(x)
    return out


def _is_model(t: Any) -> bool:
    return isinstance(t, type) and issubclass(t, BaseModel)


def type_info(ann: Any, metadata: tuple[Any, ...] | list[Any] = ()) -> TypeInfo:
    base, meta = _unwrap(ann)
    meta = _flat_meta([*metadata, *meta])
    low = high = None
    low_open = False
    min_len, max_len = 0, None
    sc: StringConstraints | None = None
    for x in meta:
        if isinstance(x, Ge):
            low = float(x.ge)  # type: ignore[arg-type]
        elif isinstance(x, Gt):
            low, low_open = float(x.gt), True  # type: ignore[arg-type]
        elif isinstance(x, Le):
            high = float(x.le)  # type: ignore[arg-type]
        elif isinstance(x, Lt):
            high = float(x.lt)  # type: ignore[arg-type]
        elif isinstance(x, MinLen):
            min_len = int(x.min_length)
        elif isinstance(x, MaxLen):
            max_len = int(x.max_length)
        elif isinstance(x, StringConstraints):
            sc = x
    origin = get_origin(base)
    if origin in (Union, types.UnionType):
        all_args = get_args(base)
        args = [a for a in all_args if a is not _NONE]
        optional = len(args) != len(all_args)
        unwrapped = [_unwrap(a) for a in args]
        if args and all(_is_model(b) for b, _ in unwrapped):
            models = tuple(b for b, _ in unwrapped)
            if len(models) == 1:
                return TypeInfo("model", optional, model=models[0])
            return TypeInfo("union", optional, models=models)
        has_float = any(b is float for b, _ in unwrapped)
        has_formula = any(b is str and _SC_FORMULA in mm for b, mm in unwrapped)
        if has_float and has_formula and len(args) == 2:
            return TypeInfo("num", optional, formula=True)
        if len(args) == 1:
            return dataclasses.replace(type_info(args[0], meta), optional=True)
        lits = [b for b, _ in unwrapped if get_origin(b) is Literal]
        lists = [a for a in args if get_origin(_unwrap(a)[0]) is list]
        if len(args) == 2 and lits and lists and get_args(lits[0]) == ("all",):
            inner = type_info(lists[0])
            return TypeInfo("names_or_all", optional, item=inner.item)
        return TypeInfo("skip")
    if base is bool:
        return TypeInfo("bool")
    if base is int:
        return TypeInfo("int", low=low, high=high, low_open=low_open)
    if base is float:
        return TypeInfo("float", low=low, high=high, low_open=low_open)
    if base is str:
        kind = "text"
        if sc is not None:
            if sc == _SC_ID:
                kind = "id"
            elif sc == _SC_REF:
                kind = "ref"
            elif sc == _SC_LABEL:
                kind = "label"
            elif sc == _SC_EXPR:
                kind = "expr"
        return TypeInfo(kind, max_len=sc.max_length if sc is not None else None)
    if origin is Literal:
        return TypeInfo("literal", choices=get_args(base))
    if origin is tuple:
        items = [type_info(a) for a in get_args(base)]
        if items and all(i.kind in ("num", "float") for i in items):
            return TypeInfo("vec", size=len(items), item=items[0])
        if items and all(i.kind == "vec" for i in items):
            return TypeInfo("line", size=len(items), item=items[0])
        return TypeInfo("skip")
    if origin is list:
        item = type_info(get_args(base)[0])
        kinds = {"vec": "vec_list", "line": "lines", "num": "num_list", "literal": "literal_list",
                 "union": "union_list", "model": "model_list"}
        kind = kinds.get(item.kind)
        if kind is None and item.kind in ("id", "ref", "text", "label", "int", "float"):
            kind = "list"
        return TypeInfo(kind or "skip", item=item, min_len=min_len, max_len=max_len)
    if origin is dict:
        k, v = get_args(base)
        ki, vi = type_info(k), type_info(v)
        if ki.kind == "literal" and vi.kind in ("float", "num", "int"):
            return TypeInfo("dict", key=ki, item=vi)
        return TypeInfo("skip")
    if _is_model(base):
        return TypeInfo("model", model=base)
    return TypeInfo("skip")


@dataclass(frozen=True)
class FieldSpec:
    """모델의 칸 하나."""

    model: type[BaseModel]
    name: str
    """파이썬 이름(with_)."""
    key: str
    """사전(파일) 이름(with)."""
    info: TypeInfo
    required: bool
    default: Any
    """파일 형식(JSON) 기본값. 필수 칸이면 None."""


def _jsonable(v: Any) -> Any:
    if isinstance(v, BaseModel):
        return v.model_dump(mode="json", by_alias=True)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


@functools.cache
def fields_of(model: type[BaseModel]) -> tuple[FieldSpec, ...]:
    out: list[FieldSpec] = []
    for name, f in model.model_fields.items():
        required = f.is_required()
        default = None if required else _jsonable(f.get_default(call_default_factory=True))
        out.append(FieldSpec(model, name, f.alias or name, type_info(f.annotation, f.metadata), required, default))
    return tuple(out)


def spec_for(model: type[BaseModel], key: str) -> FieldSpec | None:
    return next((s for s in fields_of(model) if s.key == key), None)


@functools.cache
def kinds_of(models: tuple[type[BaseModel], ...]) -> dict[str, type[BaseModel]]:
    """합집합의 종류(kind) 값 -> 모델, 선언 순서."""
    out: dict[str, type[BaseModel]] = {}
    for cls in models:
        ann = cls.model_fields["kind"].annotation
        for value in get_args(ann):
            out[str(value)] = cls
    return out


def model_for(info: TypeInfo, value: Any) -> type[BaseModel] | None:
    """이 값에 맞는 모델(합집합이면 kind로 고른다)."""
    if info.kind == "model":
        return info.model
    if info.kind == "union" and isinstance(value, dict):
        return kinds_of(info.models).get(str(value.get("kind")))
    return None


def section_info(section: str) -> TypeInfo:
    return type_info(SECTION_TYPES[section])


# ---------------------------------------------------------------------------
# 이름과 도움말 (strings.py)
# ---------------------------------------------------------------------------


@functools.cache
def _docstrings() -> dict[str, str]:
    """model.py의 칸 설명(칸 바로 아래 문자열)과 클래스 설명. 소스가 없으면(묶은 exe) 빈 사전."""
    try:
        tree = ast.parse(inspect.getsource(m))
    except (OSError, TypeError, SyntaxError):
        return {}
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        doc = ast.get_docstring(node)
        if doc:
            out[node.name] = doc
        for a, b in zip(node.body, node.body[1:], strict=False):
            if (isinstance(a, ast.AnnAssign) and isinstance(a.target, ast.Name) and isinstance(b, ast.Expr)
                    and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str)):
                out[f"{node.name}.{a.target.id}"] = inspect.cleandoc(b.value.value)
    return out


def field_name(model: type[BaseModel] | None, key: str) -> str:
    """칸의 한국어 이름. 모델별 이름(Model.key)이 먼저, 없으면 칸 이름, 그것도 없으면 원래 이름."""
    names = S.EDP_FIELD_NAMES
    if model is not None:
        for cls in model.__mro__:
            hit = names.get(f"{cls.__name__}.{key}")
            if hit:
                return hit
    return names.get(key, key)


def field_tip(model: type[BaseModel] | None, name: str, key: str | None = None) -> str:
    """도움말: strings의 설명 + model.py의 칸 설명."""
    key = key or name
    tips = S.EDP_FIELD_TIPS
    parts: list[str] = []
    if model is not None:
        tip = next((tips[f"{c.__name__}.{key}"] for c in model.__mro__ if f"{c.__name__}.{key}" in tips), None)
        parts.append(tip or tips.get(key, ""))
        docs = _docstrings()
        for cls in model.__mro__:
            doc = docs.get(f"{cls.__name__}.{name}")
            if doc:
                parts.append(doc)
                break
    else:
        parts.append(tips.get(key, ""))
    return "\n".join(p for p in parts if p)


def model_tip(model: type[BaseModel] | None) -> str:
    return _docstrings().get(model.__name__, "") if model is not None else ""


def value_name(key: str, value: Any) -> str:
    """고르기 칸 값의 한국어 이름(칸별 이름 key.value가 먼저). 없으면 원래 값."""
    names = S.EDP_VALUE_NAMES
    return names.get(f"{key}.{value}") or names.get(str(value)) or str(value)


def kind_name(kind: str) -> str:
    return value_name("kind", kind)


# ---------------------------------------------------------------------------
# 참조 후보 (사전에서 바로 읽는다: 구조 오류가 있어도 보이게)
# ---------------------------------------------------------------------------


def _items(data: Any, *keys: str) -> list[dict[str, Any]]:
    cur = data
    for k in keys:
        if not isinstance(cur, dict):
            return []
        cur = cur.get(k)
    return [x for x in cur if isinstance(x, dict)] if isinstance(cur, list) else []


def _ids(items: list[dict[str, Any]]) -> list[str]:
    return [str(x["id"]) for x in items if isinstance(x.get("id"), str)]


def _generator_outputs(data: Any) -> tuple[list[str], list[str], list[str], list[str]]:
    """(물체, 영역, 체크포인트, 태그) — symbols.generator_* 와 같은 규칙."""
    bodies: list[str] = []
    regions: list[str] = []
    cps: list[str] = []
    tags: list[str] = []
    for g in _items(data, "generators"):
        gid = g.get("id")
        if not isinstance(gid, str):
            continue
        if g.get("kind") == "heightmap":
            bodies.append(gid)
            tags.append("terrain")
            if g.get("pad") is not None:
                regions.append(f"{gid}.pad")
                tags.append("pad")
        elif g.get("kind") == "track":
            bodies.append(f"{gid}.walls")
            cps.append(f"{gid}.cp")
            tags.append("wall")
    return bodies, regions, cps, tags


_REF_KIND_BY_MODEL = {
    ("Thruster", "body"): "agent_body",
    ("Vehicle", "body"): "agent_body",
}
_REF_KIND_BY_KEY = {
    "body": "body", "a": "body", "b": "body", "target": "target", "joint": "joint", "place_by": "joint",
    "tank": "tank", "event": "event", "checkpoints": "checkpoint", "with": "hit", "hits": "hit",
    "actuators": "actuator",
}


def ref_kind(model: type[BaseModel] | None, key: str) -> str:
    """참조 칸이 가리키는 것: body agent_body target joint tank event checkpoint hit actuator any."""
    if model is not None:
        hit = _REF_KIND_BY_MODEL.get((model.__name__, key))
        if hit:
            return hit
    return _REF_KIND_BY_KEY.get(key, "any")


def ref_candidates(data: dict[str, Any], kind: str) -> list[str]:
    """그 참조 칸에 쓸 수 있는 이름들(에이전트 물체가 먼저)."""
    bodies = _items(data, "bodies")
    agent = [str(b["id"]) for b in bodies if isinstance(b.get("id"), str) and b.get("owner", "agent") == "agent"]
    world = [str(b["id"]) for b in bodies if isinstance(b.get("id"), str) and b.get("owner", "agent") != "agent"]
    dynamic = [str(b["id"]) for b in bodies if isinstance(b.get("id"), str) and b.get("type", "dynamic") == "dynamic"]
    gen_bodies, gen_regions, gen_cps, gen_tags = _generator_outputs(data)
    actuators = _items(data, "actuators")
    if kind == "body":
        return agent + world + gen_bodies
    if kind == "agent_body":
        return [b for b in agent if b in dynamic]
    if kind == "target":
        return _ids(_items(data, "regions")) + gen_regions + [b for b in agent + world if b in dynamic]
    if kind == "joint":
        return _ids(_items(data, "joints"))
    if kind == "tank":
        return [str(a["id"]) for a in actuators if a.get("kind") == "fuel_tank" and isinstance(a.get("id"), str)]
    if kind == "event":
        return _ids(_items(data, "events")) + list(sy.BUILTIN_EVENTS)
    if kind == "checkpoint":
        return _ids(_items(data, "checkpoints")) + gen_cps
    if kind == "actuator":
        return [str(a["id"]) for a in actuators if a.get("kind") != "fuel_tank" and isinstance(a.get("id"), str)]
    if kind == "hit":
        tags: list[str] = []
        for b in bodies:
            for t in b.get("tags", []) if isinstance(b.get("tags"), list) else []:
                if isinstance(t, str) and t not in tags:
                    tags.append(t)
        for t in gen_tags:
            if t not in tags:
                tags.append(t)
        return [f"tag.{t}" for t in tags] + agent + world + gen_bodies
    out: list[str] = []
    for section in SECTION_TYPES:
        out += _ids(_items(data, *section.split(".")))
    return out


# ---------------------------------------------------------------------------
# 새 부품, 종류 바꾸기
# ---------------------------------------------------------------------------

_DEFAULT_NUMBERS: dict[str, float] = {
    "w": 1.0, "h": 1.0, "r": 0.5, "max_force": 100.0, "max_accel": 8.0, "max_steer": 0.5, "max_torque": 10.0,
    "max_speed": 5.0, "spread": 1.5, "length": 10.0, "width": 8.0, "x": 0.0, "value": 1.0, "capacity": 100.0,
}
_DEFAULT_TEXT: dict[str, str] = {"when": "time > 10", "expr": "time"}
_DEFAULT_LITERAL_LISTS: dict[str, list[str]] = {"fields": ["vx", "vy"]}


def blank_value(spec: FieldSpec, data: dict[str, Any]) -> Any:
    """필수 칸을 채울 무난한 값(새 부품, 종류 바꾸기)."""
    info = spec.info
    k = info.kind
    if k == "literal":
        return info.choices[0]
    if k in ("num", "float"):
        v = _DEFAULT_NUMBERS.get(spec.key, 1.0)
        if info.low is not None and (v < info.low or (info.low_open and v <= info.low)):
            v = info.low + 1.0
        return v
    if k == "int":
        if spec.key == "count":
            return 5
        return int(info.low) if info.low is not None else 1
    if k == "bool":
        return False
    if k == "ref":
        cands = ref_candidates(data, ref_kind(spec.model, spec.key))
        return cands[0] if cands else spec.key
    if k == "list":
        if info.item is not None and info.item.kind == "ref":
            cands = ref_candidates(data, ref_kind(spec.model, spec.key))
            return cands[:1] or [spec.key]
        return []
    if k == "literal_list":
        default = _DEFAULT_LITERAL_LISTS.get(spec.key)
        return list(default) if default else list(info.item.choices[: max(1, info.min_len)]) if info.item else []
    if k == "vec_list":
        if info.min_len >= 3:
            return [[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]]
        return [[-1.0, 0.0], [1.0, 0.0]]
    if k == "num_list":
        return [0.5] * max(6, info.min_len)
    if k == "lines":
        return [[[0.0, -1.0], [0.0, 1.0]]]
    if k == "expr":
        return _DEFAULT_TEXT.get(spec.key, "0")
    if k in ("id", "label", "text"):
        return ""
    if k == "vec":
        return [0.0] * info.size
    if k == "dict":
        return {}
    if k == "union_list" and info.item is not None:
        return [new_item(info.item.models[0], data)]
    if k == "union":
        return new_item(info.models[0], data)
    if k == "model" and info.model is not None:
        return new_item(info.model, data)
    return None


def new_item(model: type[BaseModel], data: dict[str, Any], kind: str | None = None, **overrides: Any) -> dict[str, Any]:
    """필수 칸만 채운 새 부품 사전(나머지는 모델 기본값)."""
    out: dict[str, Any] = {}
    for spec in fields_of(model):
        if spec.key in overrides:
            out[spec.key] = overrides[spec.key]
        elif spec.key == "kind" and kind is not None:
            out["kind"] = kind
        elif spec.required:
            out[spec.key] = blank_value(spec, data)
    for key, value in overrides.items():
        out.setdefault(key, value)
    return out


@functools.cache
def _adapter(model: type[BaseModel], name: str) -> TypeAdapter[Any]:
    f = model.model_fields[name]
    ann = Annotated[(f.annotation, *f.metadata)] if f.metadata else f.annotation  # type: ignore[valid-type]
    return TypeAdapter(ann)


def value_fits(spec: FieldSpec, value: Any) -> bool:
    try:
        _adapter(spec.model, spec.name).validate_python(value)
    except (ValidationError, TypeError, ValueError):
        return False
    return True


def convert_item(item: dict[str, Any], info: TypeInfo, new_kind: str, data: dict[str, Any]) -> dict[str, Any]:
    """종류를 바꾼 새 사전. 새 종류에도 있고 값이 맞는 칸(id, label, body 등)은 그대로 둔다."""
    model = kinds_of(info.models)[new_kind]
    out: dict[str, Any] = {}
    for spec in fields_of(model):
        if spec.key == "kind":
            out["kind"] = new_kind
        elif spec.key in item and value_fits(spec, item[spec.key]):
            out[spec.key] = item[spec.key]
        elif spec.required:
            out[spec.key] = blank_value(spec, data)
    return out


# ---------------------------------------------------------------------------
# 진화 대상과 이름 바꾸기 (document.py를 고치지 않고 이 모듈에서 돕는다)
# ---------------------------------------------------------------------------


def evolvable(data: dict[str, Any], path: JsonPath) -> bool:
    """이 부품의 숫자 칸이 몸 유전자가 될 수 있나(에이전트 단계: 에이전트 물체, 관절, 구동기, 센서)."""
    if len(path) < 2 or not isinstance(path[1], int):
        return False
    section = path[0]
    if section in ("actuators", "sensors"):
        return True
    try:
        item = get_path(data, path[:2])
    except (KeyError, IndexError, TypeError):
        return False
    if not isinstance(item, dict):
        return False
    if section == "bodies":
        return item.get("owner", "agent") == "agent"
    if section == "joints":
        owners = {str(b.get("id")): b.get("owner", "agent") for b in _items(data, "bodies")}
        return owners.get(str(item.get("a")), "world") == "agent" or owners.get(str(item.get("b")), "world") == "agent"
    return False


def plain_param(value: Any) -> str | None:
    """값이 매개변수 하나("=p.<id>")만 가리키면 그 id."""
    if isinstance(value, str):
        mt = _PARAM_REF_RE.fullmatch(value)
        if mt:
            return mt.group(1)
    return None


def _all_strings(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for v in value.values():
            yield from _all_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _all_strings(v)
    elif isinstance(value, str):
        yield value


def param_uses(data: dict[str, Any], pid: str) -> int:
    """수식과 조건에서 p.<pid>를 쓰는 횟수."""
    rx = re.compile(rf"(?<![A-Za-z0-9_.])p\.{re.escape(pid)}(?![A-Za-z0-9_])")
    return sum(len(rx.findall(s)) for s in _all_strings({k: v for k, v in data.items() if k != "editor"}))


def default_range(value: float) -> tuple[float, float]:
    """진화를 켤 때 처음 범위: 지금 값의 절반 ~ 1.5배 (0이면 -1..1)."""
    if value == 0:
        return -1.0, 1.0
    a, b = value * 0.5, value * 1.5
    lo, hi = min(a, b), max(a, b)
    return float(f"{lo:.6g}"), float(f"{hi:.6g}")


_EXPR_KEYS = frozenset({"when", "expr", "formula"})
"""늘 수식인 칸(사건 조건, expr 점수 블록, 점수 수식)."""
_TEXT_KEYS = frozenset({"label", "description", "name", "id", "tags", "kind", "editor"})
"""이름 바꾸기가 들어가지 않는 칸."""
_REF_KEYS = frozenset({*_REF_KIND_BY_KEY, "inputs", "outputs"})
"""참조(부품 id, <id>.<출력>, 센서 채널)를 담는 칸."""


def rename_error(doc: EnvDocument, ref: Ref, new_id: str) -> str | None:
    """이름을 바꿀 수 없는 이유(바꿀 수 있으면 None)."""
    if new_id == ref[1]:
        return None
    if not _ID_RE.match(new_id):
        return S.EDP_BAD_ID
    if new_id in doc.all_ids():
        return S.EDP_ID_TAKEN.format(id=new_id)
    return None


def rename(doc: EnvDocument, ref: Ref, new_id: str) -> bool:
    """부품 이름(id)을 바꾸고, 그 이름을 가리키는 곳을 함께 고친다(되돌리기 한 단계).

    doc.rename과 같지만 앞에 점이 붙는 이름도 고친다: 매개변수 p.<id>, 랜덤 값 r.<id>, 생성기 결과
    gen.<id>.* 와 <id>.pad / <id>.walls / <id>.cp, 센서 채널 <id>.<항목>(뇌 입력 목록). 태그(tag.<이름>)는
    다른 이름 공간이라 건드리지 않는다.
    """
    if new_id == ref[1]:
        return True
    if rename_error(doc, ref, new_id) is not None or doc.find(ref) is None:
        return False
    section, old = ref
    esc = re.escape(old)
    if section == "params":
        rx_expr = re.compile(rf"(?<![A-Za-z0-9_.])p\.{esc}(?![A-Za-z0-9_])")
        expr_sub: Callable[[str], str] = lambda s: rx_expr.sub(f"p.{new_id}", s)  # noqa: E731
        plain_sub: Callable[[str], str] | None = None
    elif section == "randomize":
        rx_expr = re.compile(rf"(?<![A-Za-z0-9_.])r\.{esc}(?![A-Za-z0-9_])")
        expr_sub = lambda s: rx_expr.sub(f"r.{new_id}", s)  # noqa: E731
        plain_sub = None
    else:
        rx_word = re.compile(rf"(?<![A-Za-z0-9_.])(gen\.)?{esc}(?![A-Za-z0-9_])")

        def expr_sub(s: str) -> str:
            return rx_word.sub(lambda mt: (mt.group(1) or "") + new_id, s)

        def plain_sub(s: str) -> str:
            if s == old:
                return new_id
            if s.startswith(old + "."):
                return new_id + s[len(old):]
            return s

    def fix(value: Any, key: str | None) -> Any:
        if isinstance(value, dict):
            return {k: (v if k in _TEXT_KEYS else fix(v, k)) for k, v in value.items()}
        if isinstance(value, list):
            return [fix(v, key) for v in value]
        if isinstance(value, str):
            if value.startswith("=") or key in _EXPR_KEYS:
                return expr_sub(value)
            if plain_sub is not None and key in _REF_KEYS:
                return plain_sub(value)
        return value

    def mutate(d: dict[str, Any]) -> None:
        path = doc.item_path(ref)
        assert path is not None
        for k in list(d):
            if k not in _TEXT_KEYS:
                d[k] = fix(d[k], k)
        get_path(d, path)["id"] = new_id

    doc.edit(S.ED_UNDO_RENAME, mutate)
    doc.select([(section, new_id) if r == ref else r for r in doc.selection])
    return True


# ---------------------------------------------------------------------------
# 값 보이기·읽기
# ---------------------------------------------------------------------------


def fmt_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        text = repr(v)
        return text if len(text) <= 12 else f"{v:.10g}"
    return str(v)


def parse_number(text: str, *, formula: bool = True, optional: bool = False,
                 integer: bool = False) -> tuple[bool, Any, str | None]:
    """(맞나, 값, 틀린 이유). 숫자 또는 '='로 시작하는 수식."""
    t = text.strip()
    if not t:
        return (True, None, None) if optional else (False, None, S.EDP_REQUIRED)
    bad = S.EDP_BAD_NUMBER if formula else S.EDP_BAD_PLAIN_NUMBER
    if t.startswith("="):
        if not formula:
            return False, None, bad
        if len(t) > 1000:
            return False, None, S.EDP_TOO_LONG.format(n=1000)
        return True, t, None
    try:
        v = float(t)
    except ValueError:
        return False, None, bad
    if not math.isfinite(v):
        return False, None, bad
    if integer:
        if not v.is_integer():
            return False, None, S.EDP_BAD_INT
        return True, int(v), None
    return True, v, None


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return float(a) == float(b)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    return a == b


def _qt_class(w: QWidget) -> str:
    mo = w.metaObject()
    while mo is not None and not mo.className().startswith("Q"):
        mo = mo.superClass()
    return mo.className() if mo is not None else "QWidget"


def _mark(w: QWidget, invalid: bool, issue: bool) -> None:
    """빨간 테두리(검사 결과)와 옅은 빨간 바탕(잘못 쓴 값)."""
    w.setProperty("invalid", invalid)
    if not invalid and not issue:
        w.setStyleSheet("")
        return
    rules = f"border: 1px solid {ISSUE_COLOR};"
    if invalid:
        rules += " background-color: rgba(217, 83, 79, 50);"
    w.setStyleSheet(f"{_qt_class(w)} {{ {rules} }}")


class _NoWheelSpin(QSpinBox):
    """포커스가 없을 때는 휠을 무시한다(스크롤하다 값이 바뀌지 않게)."""

    def wheelEvent(self, e: Any) -> None:  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class _NoWheelCombo(QComboBox):
    def wheelEvent(self, e: Any) -> None:  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class NumEdit(QLineEdit):
    """숫자 또는 '='로 시작하는 수식 한 칸. 올바른 값이 될 때마다(타이핑 중에도) edited를 낸다."""

    edited = Signal(object)
    problem_changed = Signal(str)

    def __init__(self, *, formula: bool = True, optional: bool = False, integer: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.formula = formula
        self.optional = optional
        self.integer = integer
        self.problem: str | None = None
        self._shown = ""
        self.setMinimumWidth(44)
        self.textEdited.connect(self._on_edited)
        self.editingFinished.connect(self._on_finished)

    def parse(self, text: str | None = None) -> tuple[bool, Any, str | None]:
        return parse_number(self.text() if text is None else text, formula=self.formula, optional=self.optional,
                            integer=self.integer)

    def _set_problem(self, problem: str | None) -> None:
        if problem != self.problem:
            self.problem = problem
            self.problem_changed.emit(problem or "")

    def _on_edited(self, text: str) -> None:
        ok, value, err = self.parse(text)
        self._set_problem(err)
        if ok:
            self.edited.emit(value)

    def _on_finished(self) -> None:
        ok, value, _err = self.parse()
        if ok and self.text() != self._shown:
            self.edited.emit(value)

    def show_value(self, value: Any) -> None:
        """문서 값을 보인다. 타이핑 중이고 이미 같은 값(또는 아직 틀린 값)이면 건드리지 않는다."""
        text = fmt_value(value)
        if self.hasFocus():
            ok, cur, _ = self.parse()
            if self.problem is not None or (ok and _same(cur, value)):
                return
        if self.text() != text:
            self.setText(text)
            self.setCursorPosition(0)
        self._shown = text
        self._set_problem(None)


# ---------------------------------------------------------------------------
# 칸 위젯
# ---------------------------------------------------------------------------


class FormContext:
    """한 폼이 같이 쓰는 것: 문서, 진화 체크박스를 둘지, 부품 구역과 경로(이름 바꾸기, 진화 이름)."""

    def __init__(self, doc: EnvDocument, *, evolvable: bool = False, section: str | None = None,
                 root: JsonPath = (), compact: bool = False) -> None:
        self.doc = doc
        self.evolvable = evolvable
        self.section = section
        self.root = tuple(root)
        self.compact = compact

    def owner(self) -> tuple[str, str]:
        """(부품 id, 표시 이름) — 진화 매개변수 이름을 지을 때."""
        item = self.doc.get(self.root[:2]) if len(self.root) >= 2 else None
        if isinstance(item, dict):
            return str(item.get("id", "")), str(item.get("label") or "")
        return "", ""


class FieldWidget(QWidget):
    """칸 하나(또는 칸 묶음). path는 문서 사전 안의 경로."""

    spans = False
    """폼에서 이름표 없이 한 줄을 다 쓰는가(묶음)."""

    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec | None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.path: JsonPath = tuple(path)
        self.spec = spec
        self.default: Any = spec.default if spec is not None else None
        self.label: QLabel | None = None
        self.editor: QWidget | None = None
        self._issues: list[str] = []
        self._invalid: str | None = None
        self._tip = field_tip(spec.model, spec.name, spec.key) if spec is not None else ""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    @property
    def doc(self) -> EnvDocument:
        return self.ctx.doc

    def display_name(self) -> str:
        return field_name(self.spec.model, self.spec.key) if self.spec is not None else ""

    def value(self) -> Any:
        v = self.doc.get(self.path, _MISSING)
        return self.default if v is _MISSING else v

    def write(self, value: Any, merge: bool = True) -> None:
        self.doc.set_value(self.path, value, label=S.EDP_UNDO_FIELD.format(name=self.display_name()), merge=merge)

    def refresh(self) -> None:
        """문서 값에 맞춘다(위젯은 그대로)."""

    def leaves(self) -> Iterator[FieldWidget]:
        yield self

    # ---- 표시 ----
    def set_issues(self, issues: list[Issue]) -> None:
        msgs = [i.message for i in issues_at(issues, self.path)]
        if msgs != self._issues:
            self._issues = msgs
            self._restyle()

    @property
    def issue_messages(self) -> list[str]:
        return list(self._issues)

    def set_invalid(self, message: str | None) -> None:
        message = message or None
        if message != self._invalid:
            self._invalid = message
            self._restyle()

    @property
    def invalid(self) -> str | None:
        return self._invalid

    def tooltip(self) -> str:
        parts = [self._tip]
        if self._invalid:
            parts.append(self._invalid)
        parts += [f"⚠ {msg}" for msg in self._issues]
        return "\n".join(p for p in parts if p)

    def style_targets(self) -> list[QWidget]:
        return [self.editor] if self.editor is not None else []

    def _restyle(self) -> None:
        tip = self.tooltip()
        for w in self.style_targets():
            _mark(w, bool(self._invalid), bool(self._issues))
            w.setToolTip(tip)
        if self.label is not None:
            self.label.setToolTip(tip)
            self.label.setStyleSheet(f"color: {ISSUE_COLOR};" if self._issues else "")


class NumField(FieldWidget):
    """숫자·수식 칸. 진화할 수 있는 칸이면 옆에 '진화' 체크박스, 켜면 아래에 최솟값·최댓값."""

    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec | None, info: TypeInfo | None = None,
                 *, evolve: bool | None = None, parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, spec, parent)
        self.info = info or (spec.info if spec is not None else TypeInfo("num", formula=True))
        self.component: int | None = None
        self.writer: Callable[[Any], None] = self.write
        self.ensure: Callable[[], None] | None = None
        self._pid: str | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(4)
        self.editor = NumEdit(formula=self.info.formula, optional=self.info.optional, integer=self.info.kind == "int")
        self.editor.edited.connect(self._on_edited)
        self.editor.problem_changed.connect(self.set_invalid)
        top.addWidget(self.editor, 1)
        lay.addLayout(top)
        if evolve is None:
            evolve = ctx.evolvable and self.info.kind == "num" and not ctx.compact
        self.evolve_box: QCheckBox | None = None
        self.min_edit: NumEdit | None = None
        self.max_edit: NumEdit | None = None
        self._range: QWidget | None = None
        if evolve:
            self.evolve_box = QCheckBox(S.EDP_EVOLVE)
            self.evolve_box.clicked.connect(self._on_evolve)
            top.addWidget(self.evolve_box)
            self._range = QWidget()
            r = QHBoxLayout(self._range)
            r.setContentsMargins(0, 0, 0, 0)
            r.setSpacing(4)
            self.min_edit = NumEdit(formula=False, optional=True)
            self.max_edit = NumEdit(formula=False, optional=True)
            self.min_edit.edited.connect(lambda v: self._set_param("min", v))
            self.max_edit.edited.connect(lambda v: self._set_param("max", v))
            r.addWidget(QLabel(S.EDP_MIN))
            r.addWidget(self.min_edit, 1)
            r.addWidget(QLabel(S.EDP_MAX))
            r.addWidget(self.max_edit, 1)
            lay.addWidget(self._range)
            self._range.hide()
        self.refresh()
        self._restyle()

    @property
    def evolving(self) -> str | None:
        """진화 중이면 매개변수 id."""
        return self._pid

    def gene_name(self) -> str:
        owner, _label = self.ctx.owner()
        key = self.spec.key if self.spec is not None else "value"
        if self.component is not None:
            key += "_" + "xyzw"[self.component % 4]
        return f"{owner}_{key}" if owner else key

    def gene_label(self) -> str:
        owner, label = self.ctx.owner()
        name = self.display_name()
        if self.component is not None:
            name += " " + "xyzw"[self.component % 4]
        return S.EDP_EVOLVE_PARAM_LABEL.format(owner=label or owner, field=name)

    def refresh(self) -> None:
        value = self.value()
        pid = self.doc.evolve_param(self.path) if self.evolve_box is not None else None
        self._pid = pid
        assert isinstance(self.editor, NumEdit)
        if pid is not None and self.evolve_box is not None and self._range is not None:
            param = self.doc.item(("params", pid)) or {}
            self.editor.formula = False
            self.editor.optional = False
            self.editor.show_value(param.get("value"))
            assert self.min_edit is not None and self.max_edit is not None
            self.min_edit.show_value(param.get("min"))
            self.max_edit.show_value(param.get("max"))
            self._range.show()
            self.evolve_box.setChecked(True)
            self.evolve_box.setEnabled(True)
            self.evolve_box.setToolTip(S.EDP_EVOLVE_ON_TIP.format(id=pid))
            self._tip_extra = S.EDP_EVOLVE_VALUE_TIP.format(id=pid)
        else:
            self.editor.formula = self.info.formula
            self.editor.optional = self.info.optional
            self.editor.show_value(value)
            self._tip_extra = ""
            if self.evolve_box is not None and self._range is not None:
                self._range.hide()
                self.evolve_box.setChecked(False)
                number = isinstance(value, (int, float)) and not isinstance(value, bool)
                linked = plain_param(value)
                ok = number or (linked is not None and self.doc.find(("params", linked)) is not None)
                self.evolve_box.setEnabled(ok)
                self.evolve_box.setToolTip(S.EDP_EVOLVE_TIP if ok else S.EDP_EVOLVE_FORMULA_TIP)

    def tooltip(self) -> str:
        base = super().tooltip()
        extra = getattr(self, "_tip_extra", "")
        return "\n".join(p for p in (extra, base) if p)

    def _on_edited(self, value: Any) -> None:
        if self._pid is not None:
            idx = self.doc.find(("params", self._pid))
            if idx is not None:
                self.doc.set_value(("params", idx, "value"), value,
                                   label=S.EDP_UNDO_FIELD.format(name=self.display_name()))
            return
        self.writer(value)

    def _set_param(self, key: str, value: Any) -> None:
        if self._pid is None:
            return
        idx = self.doc.find(("params", self._pid))
        if idx is not None:
            self.doc.set_value(("params", idx, key), value, label=S.EDP_UNDO_FIELD.format(name=field_name(m.Param, key)))

    def _on_evolve(self, checked: bool) -> None:
        doc = self.doc
        if checked:
            value = self.value()
            linked = plain_param(value)
            if linked is not None and doc.find(("params", linked)) is not None:
                _enable_param(doc, linked)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if self.ensure is not None:
                    self.ensure()
                low, high = default_range(float(value))
                doc.make_evolvable(self.path, low, high, name=self.gene_name(), label=self.gene_label())
        else:
            pid = doc.evolve_param(self.path)
            if pid is not None:
                if param_uses(doc.data, pid) > 1:  # 다른 칸도 이 매개변수를 쓴다: 매개변수는 두고 진화만 끈다
                    idx = doc.find(("params", pid))
                    doc.edit(S.ED_UNDO_EVOLVE_OFF, lambda d: set_path(d, ("params", idx, "evolve"), False))
                else:
                    doc.make_fixed(self.path)
        self.refresh()
        self._restyle()


def _enable_param(doc: EnvDocument, pid: str) -> None:
    """이미 있는 매개변수를 진화 대상으로(범위가 없으면 값 기준으로 정한다)."""
    idx = doc.find(("params", pid))
    if idx is None:
        return
    param = doc.items("params")[idx]
    low, high = default_range(float(param.get("value", 0.0)))

    def mutate(d: dict[str, Any]) -> None:
        p = d["params"][idx]
        p["evolve"] = True
        if p.get("min") is None:
            p["min"] = low
        if p.get("max") is None:
            p["max"] = high

    doc.edit(S.ED_UNDO_EVOLVE_ON, mutate)


class VecField(FieldWidget):
    """고정 길이 수 묶음(Vec2, 경계 4개, 각도 제한). 선택 칸이면 '사용' 체크박스."""

    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec | None, info: TypeInfo | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, spec, parent)
        self.info = info or (spec.info if spec is not None else TypeInfo("vec", size=2))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.use_box: QCheckBox | None = None
        if self.info.optional:
            self.use_box = QCheckBox(S.EDP_USE)
            self.use_box.clicked.connect(self._on_use)
            lay.addWidget(self.use_box)
        item = self.info.item or TypeInfo("num", formula=True)
        defaults = self._defaults()
        self.parts: list[NumField] = []
        for i in range(self.info.size):
            part = NumField(ctx, (*self.path, i), spec, dataclasses.replace(item, optional=False))
            part.default = defaults[i]
            part.component = i
            part.writer = functools.partial(self._write_part, i)
            part.ensure = self._ensure
            part._tip = ("xyzw"[i] if self.info.size == 2 else f"{i + 1}") + (": " + self._tip if self._tip else "")
            part._restyle()
            lay.addWidget(part, 1)
            self.parts.append(part)
        self.editor = None
        self.refresh()
        self._restyle()

    def _defaults(self) -> list[Any]:
        d = self.spec.default if self.spec is not None else None
        if isinstance(d, list) and len(d) == self.info.size:
            return list(d)
        if self.spec is not None and self.spec.key == "limit":
            return [-0.5, 0.5]
        return [0.0] * self.info.size

    def _current(self) -> list[Any]:
        v = self.value()
        if isinstance(v, (list, tuple)) and len(v) == self.info.size:
            return list(v)
        return self._defaults()

    def _write_part(self, i: int, value: Any) -> None:
        cur = self._current()
        cur[i] = value
        self.write(cur)

    def _ensure(self) -> None:
        if not isinstance(self.doc.get(self.path), list):
            self.write(self._current(), merge=False)

    def _on_use(self, checked: bool) -> None:
        self.write(self._defaults() if checked else None, merge=False)

    def refresh(self) -> None:
        present = self.value() is not None
        if self.use_box is not None:
            self.use_box.setChecked(present)
        for p in self.parts:
            p.setEnabled(present)
            if present:
                p.refresh()
            else:
                assert isinstance(p.editor, NumEdit)
                p.editor.show_value(None)

    def leaves(self) -> Iterator[FieldWidget]:
        yield self
        for p in self.parts:
            yield from p.leaves()

    def style_targets(self) -> list[QWidget]:
        return [self.use_box] if self.use_box is not None else []

    def set_issues(self, issues: list[Issue]) -> None:
        # 묶음 자체를 가리키는 결과만(칸별 결과는 칸이 보인다). 이름표는 어느 쪽이든 빨갛게.
        exact = [i.message for i in issues_at(issues, self.path, exact=True)]
        inner = [i.message for i in issues_at(issues, self.path) if i.message not in exact]
        msgs = exact + inner
        if msgs != self._issues:
            self._issues = msgs
            self._restyle()


class IntField(FieldWidget):
    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, spec, parent)
        info = spec.info
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        spin = _NoWheelSpin()
        spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        low = int(info.low) + (1 if info.low_open else 0) if info.low is not None else -2_000_000_000
        high = int(info.high) if info.high is not None else 2_000_000_000
        spin.setRange(low, high)
        spin.valueChanged.connect(self._on_changed)
        lay.addWidget(spin)
        self.editor = spin
        self._updating = False
        self.refresh()
        self._restyle()

    def _on_changed(self, v: int) -> None:
        if not self._updating:
            self.write(int(v))

    def refresh(self) -> None:
        spin = self.editor
        assert isinstance(spin, QSpinBox)
        v = self.value()
        try:
            iv = int(v)
        except (TypeError, ValueError):
            iv = 0
        if spin.value() != iv:
            if spin.hasFocus() and spin.lineEdit().text().strip() in ("", "-"):
                return
            self._updating = True
            spin.setValue(iv)
            self._updating = False


class BoolField(FieldWidget):
    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, spec, parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        box = QCheckBox()
        box.clicked.connect(lambda checked: self.write(bool(checked), merge=False))
        lay.addWidget(box)
        if ctx.compact:
            lay.setAlignment(box, Qt.AlignmentFlag.AlignCenter)
        self.editor = box
        self.refresh()
        self._restyle()

    def refresh(self) -> None:
        box = self.editor
        assert isinstance(box, QCheckBox)
        box.blockSignals(True)
        box.setChecked(bool(self.value()))
        box.blockSignals(False)


class LiteralField(FieldWidget):
    """고르기 칸(Literal). 보이는 이름은 strings.EDP_VALUE_NAMES."""

    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, spec, parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        combo = _NoWheelCombo()
        combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.choices = tuple(spec.info.choices)
        for c in self.choices:
            combo.addItem(value_name(spec.key, c), c)
        combo.activated.connect(self._on_activated)
        lay.addWidget(combo)
        self.editor = combo
        self.refresh()
        self._restyle()

    def _on_activated(self, index: int) -> None:
        combo = self.editor
        assert isinstance(combo, QComboBox)
        value = combo.itemData(index)
        if value in self.choices and value != self.value():
            self.write(value, merge=False)

    def refresh(self) -> None:
        combo = self.editor
        assert isinstance(combo, QComboBox)
        v = self.value()
        combo.blockSignals(True)
        idx = combo.findData(v)
        if idx < 0:
            while combo.count() > len(self.choices):
                combo.removeItem(combo.count() - 1)
            combo.addItem(S.EDP_UNKNOWN_VALUE.format(value=v), v)
            idx = combo.count() - 1
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)


class UnionKindField(FieldWidget):
    """종류 고르기(모양, 센서, 구동기, 점수 블록, 생성기). 바꾸면 부품을 새 종류로 바꾼다(공통 칸은 둔다)."""

    def __init__(self, ctx: FormContext, path: JsonPath, info: TypeInfo, parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, None, parent)
        self.info = info
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        combo = _NoWheelCombo()
        combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.kinds = list(kinds_of(info.models))
        for k in self.kinds:
            combo.addItem(kind_name(k), k)
        combo.activated.connect(self._on_activated)
        lay.addWidget(combo)
        self.editor = combo
        self.refresh()
        self._restyle()

    def display_name(self) -> str:
        return field_name(None, "kind")

    def _on_activated(self, index: int) -> None:
        combo = self.editor
        assert isinstance(combo, QComboBox)
        kind = combo.itemData(index)
        item_path = self.path[:-1]
        old = self.doc.get(item_path)
        if not isinstance(old, dict) or old.get("kind") == kind or kind not in self.kinds:
            return
        new = convert_item(old, self.info, kind, self.doc.data)
        self.doc.edit(S.EDP_UNDO_KIND, lambda d: set_path(d, item_path, new))

    def refresh(self) -> None:
        combo = self.editor
        assert isinstance(combo, QComboBox)
        v = self.doc.get(self.path)
        combo.blockSignals(True)
        idx = combo.findData(v)
        if idx < 0:
            while combo.count() > len(self.kinds):
                combo.removeItem(combo.count() - 1)
            combo.addItem(S.EDP_UNKNOWN_VALUE.format(value=v), v)
            idx = combo.count() - 1
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)
        self._tip = model_tip(kinds_of(self.info.models).get(str(v)))
        self._restyle()


class RefField(FieldWidget):
    """참조 칸: 쓸 수 있는 id를 고르거나 직접 쓴다."""

    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, spec, parent)
        self.kind = ref_kind(spec.model, spec.key)
        self.optional = spec.info.optional
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        combo = _NoWheelCombo()
        combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(6)
        combo.activated.connect(lambda _i: self._commit(combo.currentText()))
        line = combo.lineEdit()
        line.editingFinished.connect(lambda: self._commit(combo.currentText()))
        line.textEdited.connect(self._check)
        lay.addWidget(combo)
        self.editor = combo
        self._cands: list[str] | None = None
        self.refresh()
        self._restyle()

    def _check(self, text: str) -> str | None:
        t = text.strip()
        err = None
        if not t:
            err = None if self.optional else S.EDP_REQUIRED
        elif not _REF_RE.match(t) or len(t) > 80:
            err = S.EDP_BAD_REF
        self.set_invalid(err)
        return err

    def _commit(self, text: str) -> None:
        if self._check(text) is not None:
            return
        t = text.strip()
        value = t if t else None
        if value != self.value():
            self.write(value, merge=False)

    def refresh(self) -> None:
        combo = self.editor
        assert isinstance(combo, QComboBox)
        v = self.value()
        text = "" if v is None else str(v)
        focused = combo.lineEdit().hasFocus()
        cands = ref_candidates(self.doc.data, self.kind)
        if self.optional:
            cands = ["", *cands]
        if cands != self._cands:
            typed = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            for c in cands:
                combo.addItem(c)
                if c in sy.BUILTIN_EVENTS:
                    combo.setItemData(combo.count() - 1, S.EDP_BUILTIN_EVENT.format(name=value_name("event", c)),
                                      Qt.ItemDataRole.ToolTipRole)
            combo.setEditText(typed if focused else text)
            combo.blockSignals(False)
            self._cands = cands
        if not focused and combo.currentText() != text:
            combo.blockSignals(True)
            combo.setEditText(text)
            combo.blockSignals(False)
            self.set_invalid(None)


class TextField(FieldWidget):
    """글자 칸: 표시 이름, 설명, 조건·점수 수식(Expr), 이름."""

    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec, *, multiline: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, spec, parent)
        self.info = spec.info
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.multiline = multiline
        if multiline:
            edit: QWidget = QPlainTextEdit()
            assert isinstance(edit, QPlainTextEdit)
            edit.setTabChangesFocus(True)
            edit.setFixedHeight(edit.fontMetrics().lineSpacing() * 3 + 12)
            edit.textChanged.connect(lambda: self._on_text(edit.toPlainText()))
        else:
            edit = QLineEdit()
            assert isinstance(edit, QLineEdit)
            if self.info.max_len:
                edit.setMaxLength(self.info.max_len)
            edit.textEdited.connect(self._on_text)
        lay.addWidget(edit)
        self.editor = edit
        self.refresh()
        self._restyle()

    def text(self) -> str:
        e = self.editor
        return e.toPlainText() if isinstance(e, QPlainTextEdit) else e.text()  # type: ignore[union-attr]

    def _problem(self, text: str) -> str | None:
        k = self.info.kind
        if not text.strip():
            if self.info.optional or k in ("label", "text"):
                return None
            return S.EDP_REQUIRED
        if k == "id" and not _MODEL_ID_RE.match(text):
            return S.EDP_BAD_ID
        if k == "ref" and not _REF_RE.match(text):
            return S.EDP_BAD_REF
        if self.info.max_len and len(text) > self.info.max_len:
            return S.EDP_TOO_LONG.format(n=self.info.max_len)
        return None

    def _on_text(self, text: str) -> None:
        err = self._problem(text)
        self.set_invalid(err)
        if err is not None:
            return
        value: Any = None if (self.info.optional and not text.strip()) else text
        if value != self.value():
            self.write(value)

    def refresh(self) -> None:
        v = self.value()
        text = "" if v is None else str(v)
        e = self.editor
        assert e is not None
        if e.hasFocus() and (self._invalid or self.text() == text):
            return
        if self.text() != text:
            e.blockSignals(True)
            if isinstance(e, QPlainTextEdit):
                e.setPlainText(text)
            else:
                assert isinstance(e, QLineEdit)
                e.setText(text)
                e.setCursorPosition(0)
            e.blockSignals(False)
        self.set_invalid(None)


class IdField(FieldWidget):
    """부품 이름(id). Enter나 칸을 떠날 때 이름을 바꾸고 이 이름을 가리키는 곳도 고친다."""

    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec | None, parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, spec, parent)
        self._tip = "\n".join(p for p in (self._tip, S.EDP_RENAME_TIP) if p)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setMaxLength(40)
        edit.textEdited.connect(self._check)
        edit.editingFinished.connect(self._commit)
        lay.addWidget(edit)
        self.editor = edit
        self.last_error: str | None = None
        self.refresh()
        self._restyle()

    def display_name(self) -> str:
        return field_name(None, "id")

    def ref(self) -> Ref | None:
        v = self.doc.get(self.path)
        return (self.ctx.section, str(v)) if self.ctx.section and isinstance(v, str) else None

    def _check(self, text: str) -> str | None:
        ref = self.ref()
        err = rename_error(self.doc, ref, text.strip()) if ref is not None else None
        self.set_invalid(err)
        return err

    def _commit(self) -> None:
        edit = self.editor
        assert isinstance(edit, QLineEdit)
        ref = self.ref()
        new = edit.text().strip()
        if ref is None or new == ref[1]:
            self.set_invalid(None)
            if ref is not None and edit.text() != ref[1]:
                edit.setText(ref[1])
            return
        err = rename_error(self.doc, ref, new)
        if err is None and rename(self.doc, ref, new):
            self.last_error = None
            self.set_invalid(None)
            return
        self.last_error = S.EDP_RENAME_FAILED.format(reason=err or S.EDP_BAD_ID)
        edit.blockSignals(True)
        edit.setText(ref[1])
        edit.blockSignals(False)
        self.set_invalid(None)
        if edit.isVisible():
            QToolTip.showText(edit.mapToGlobal(edit.rect().bottomLeft()), self.last_error, edit)

    def refresh(self) -> None:
        edit = self.editor
        assert isinstance(edit, QLineEdit)
        v = self.doc.get(self.path)
        text = "" if v is None else str(v)
        if edit.hasFocus() and edit.text() != text:
            return
        if edit.text() != text:
            edit.setText(text)
        self.set_invalid(None)


def _parse_item(info: TypeInfo, text: str) -> tuple[bool, Any]:
    k = info.kind
    if k == "int":
        ok, v, _ = parse_number(text, formula=False, integer=True)
        if not ok:
            return False, None
        if (info.low is not None and v < info.low) or (info.high is not None and v > info.high):
            return False, None
        return True, v
    if k == "float":
        ok, v, _ = parse_number(text, formula=False)
        return ok, v
    if k == "id":
        return bool(_MODEL_ID_RE.match(text)) and len(text) <= 40, text
    if k == "ref":
        return bool(_REF_RE.match(text)) and len(text) <= 80, text
    return bool(text), text


def parse_list(info: TypeInfo, text: str) -> tuple[bool, Any, str | None]:
    """쉼표 목록 -> (맞나, 목록, 틀린 이유)."""
    item = info.item or TypeInfo("text")
    parts = [p.strip() for p in text.split(",")]
    parts = [p for p in parts if p]
    if not parts and info.optional:
        return True, None, None
    out: list[Any] = []
    for p in parts:
        ok, v = _parse_item(item, p)
        if not ok:
            return False, None, S.EDP_BAD_LIST_ITEM.format(item=p)
        out.append(v)
    if len(out) < info.min_len:
        return False, None, S.EDP_LIST_TOO_SHORT.format(n=info.min_len)
    return True, out, None


class ListField(FieldWidget):
    """쉼표로 나눈 목록(키, 태그, 감지 대상, 은닉층 크기, 고를 값). 참조 목록이면 '+' 단추로 골라 더한다."""

    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec, info: TypeInfo | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, spec, parent)
        self.info = info or spec.info
        self._tip = "\n".join(p for p in (self._tip, S.EDP_LIST_TIP) if p)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        edit = QLineEdit()
        edit.textEdited.connect(self._on_text)
        lay.addWidget(edit, 1)
        self.editor = edit
        self.pick: QToolButton | None = None
        item = self.info.item
        if item is not None and item.kind == "ref":
            self.kind = ref_kind(spec.model, spec.key)
            self.pick = QToolButton()
            self.pick.setText("+")
            self.pick.setToolTip(S.EDP_PICK)
            self.pick.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            menu = QMenu(self.pick)
            menu.aboutToShow.connect(lambda: self._fill_menu(menu))
            self.pick.setMenu(menu)
            lay.addWidget(self.pick)
        self.refresh()
        self._restyle()

    def _fill_menu(self, menu: QMenu) -> None:
        menu.clear()
        current = self.value()
        current = current if isinstance(current, list) else []
        for c in ref_candidates(self.doc.data, self.kind):
            act = menu.addAction(c)
            act.setEnabled(c not in current)
            act.triggered.connect(functools.partial(self._append, c))

    def _append(self, name: str) -> None:
        cur = self.value()
        cur = list(cur) if isinstance(cur, list) else []
        if name not in cur:
            self.write([*cur, name], merge=False)

    def _on_text(self, text: str) -> None:
        ok, value, err = parse_list(self.info, text)
        self.set_invalid(err)
        if ok and value != self.value():
            self.write(value)

    @staticmethod
    def format(value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(fmt_value(v) for v in value)
        return "" if value is None else str(value)

    def refresh(self) -> None:
        edit = self.editor
        assert isinstance(edit, QLineEdit)
        v = self.value()
        text = self.format(v)
        if edit.hasFocus():
            ok, cur, _ = parse_list(self.info, edit.text())
            if self._invalid or (ok and cur == v):
                return
        if edit.text() != text:
            edit.setText(text)
            edit.setCursorPosition(0)
        self.set_invalid(None)


class LiteralListField(FieldWidget):
    """고르기 값 목록(몸 상태 항목): 체크박스. 켠 순서를 지킨다."""

    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, spec, parent)
        assert spec.info.item is not None
        self.choices = tuple(spec.info.item.choices)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        self.boxes: dict[Any, QCheckBox] = {}
        cols = 3
        for i, c in enumerate(self.choices):
            box = QCheckBox(str(c))
            box.setToolTip(value_name(spec.key, c))
            box.clicked.connect(functools.partial(self._on_click, c))
            grid.addWidget(box, i // cols, i % cols)
            self.boxes[c] = box
        self.editor = None
        self.refresh()
        self._restyle()

    def _on_click(self, choice: Any, *_args: Any) -> None:
        checked = self.boxes[choice].isChecked()
        cur = self.value()
        cur = [x for x in cur if x in self.choices] if isinstance(cur, list) else []
        if checked:
            new = cur if choice in cur else [*cur, choice]  # 켠 순서를 지킨다
        else:
            new = [x for x in cur if x != choice]
        min_len = self.spec.info.min_len if self.spec is not None else 0
        if len(new) < min_len:  # 마지막 하나는 끌 수 없다
            self.refresh()
            return
        self.write(new, merge=False)

    def refresh(self) -> None:
        v = self.value()
        chosen = set(v) if isinstance(v, list) else set()
        for c, box in self.boxes.items():
            box.blockSignals(True)
            box.setChecked(c in chosen)
            box.blockSignals(False)


class NamesOrAllField(FieldWidget):
    """이름 목록 또는 "all"(점수 블록의 구동기, 뇌 입력·출력)."""

    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, spec, parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.all_box = QCheckBox(S.EDP_ALL)
        self.all_box.clicked.connect(self._on_all)
        lay.addWidget(self.all_box)
        self.list_info = TypeInfo("list", item=spec.info.item or TypeInfo("text"), min_len=0)
        self.edit = QLineEdit()
        self.edit.textEdited.connect(self._on_text)
        lay.addWidget(self.edit, 1)
        self.editor = self.edit
        self.kind = ref_kind(spec.model, spec.key)
        self.refresh()
        self._restyle()

    def _on_all(self, checked: bool) -> None:
        if checked:
            self.write("all", merge=False)
        else:
            self.write(ref_candidates(self.doc.data, self.kind), merge=False)

    def _on_text(self, text: str) -> None:
        ok, value, err = parse_list(self.list_info, text)
        self.set_invalid(err)
        if ok and value != self.value():
            self.write(value)

    def refresh(self) -> None:
        v = self.value()
        is_all = v == "all"
        self.all_box.setChecked(is_all)
        self.edit.setEnabled(not is_all)
        text = "" if is_all else ListField.format(v)
        if self.edit.hasFocus():
            ok, cur, _ = parse_list(self.list_info, self.edit.text())
            if self._invalid or (ok and cur == v):
                return
        if self.edit.text() != text:
            self.edit.setText(text)
        self.set_invalid(None)


class DictField(FieldWidget):
    """키·값 표(몸 상태 센서의 항목별 정규화 크기)."""

    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, spec, parent)
        info = spec.info
        assert info.key is not None and info.item is not None
        self.keys = tuple(info.key.choices)
        self.value_info = info.item
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        self.rows_widget = QWidget()
        self.grid = QGridLayout(self.rows_widget)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setVerticalSpacing(2)
        outer.addWidget(self.rows_widget)
        add_row = QHBoxLayout()
        self.add_combo = _NoWheelCombo()
        self.add_button = QPushButton(S.EDP_ADD)
        self.add_button.clicked.connect(self._on_add)
        add_row.addWidget(self.add_combo, 1)
        add_row.addWidget(self.add_button)
        outer.addLayout(add_row)
        self.edits: dict[str, NumEdit] = {}
        self._built: tuple[str, ...] | None = None
        self.editor = None
        self.refresh()
        self._restyle()

    def _current(self) -> dict[str, Any]:
        v = self.value()
        return dict(v) if isinstance(v, dict) else {}

    def _rebuild(self, keys: tuple[str, ...]) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.blockSignals(True)
                w.hide()
                w.deleteLater()
        self.edits = {}
        for r, key in enumerate(keys):
            label = QLabel(str(key))
            label.setToolTip(value_name(key, key))
            edit = NumEdit(formula=self.value_info.kind == "num")
            edit.edited.connect(functools.partial(self._on_value, key))
            remove = QToolButton()
            remove.setText("✕")
            remove.setToolTip(S.EDP_REMOVE)
            remove.clicked.connect(functools.partial(self._on_remove, key))
            self.grid.addWidget(label, r, 0)
            self.grid.addWidget(edit, r, 1)
            self.grid.addWidget(remove, r, 2)
            self.edits[key] = edit
        self.add_combo.clear()
        for k in self.keys:
            if k not in keys:
                self.add_combo.addItem(str(k), k)
        self.add_combo.setEnabled(self.add_combo.count() > 0)
        self.add_button.setEnabled(self.add_combo.count() > 0)
        self._built = keys

    def _on_value(self, key: str, value: Any) -> None:
        if value is None:
            return
        self.doc.set_value((*self.path, key), value, label=S.EDP_UNDO_FIELD.format(name=self.display_name()))

    def _on_remove(self, key: str) -> None:
        cur = self._current()
        cur.pop(key, None)
        self.write(cur, merge=False)

    def _on_add(self) -> None:
        key = self.add_combo.currentData()
        if key is None:
            return
        cur = self._current()
        cur[key] = sy.DEFAULT_STATE_SCALE.get(str(key), 1.0)
        self.write(cur, merge=False)

    def refresh(self) -> None:
        cur = self._current()
        keys = tuple(str(k) for k in cur)
        if keys != self._built:
            self._rebuild(keys)
        for k, edit in self.edits.items():
            edit.show_value(cur.get(k))


class TableField(FieldWidget):
    """수 목록 표: 꼭짓점(x, y), 반지름들, 체크포인트 선(x1, y1, x2, y2), 트랙 중심선."""

    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, spec, parent)
        self.info = spec.info
        k = self.info.kind
        self.cols = {"vec_list": ["x", "y"], "num_list": [S.EDP_COL_VALUE], "lines": ["x1", "y1", "x2", "y2"]}[k]
        inner = self.info.item
        while inner is not None and inner.kind in ("vec", "line"):
            inner = inner.item
        self.formula = bool(inner and inner.formula)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        table = QTableWidget(0, len(self.cols))
        table.setHorizontalHeaderLabels(self.cols)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setDefaultSectionSize(table.fontMetrics().height() + 8)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setMaximumHeight(table.fontMetrics().height() * 9)
        table.setMinimumHeight(table.fontMetrics().height() * 5)
        table.itemChanged.connect(self._on_item)
        lay.addWidget(table)
        row = QHBoxLayout()
        self.add_button = QPushButton(S.EDP_ADD_ROW)
        self.remove_button = QPushButton(S.EDP_REMOVE_ROW)
        self.add_button.clicked.connect(self._on_add)
        self.remove_button.clicked.connect(self._on_remove)
        row.addWidget(self.add_button)
        row.addWidget(self.remove_button)
        row.addStretch(1)
        lay.addLayout(row)
        self.table = table
        self.editor = table
        self._rows: list[list[Any]] | None = None
        self.refresh()
        self._restyle()

    def _rows_from(self, value: Any) -> list[list[Any]]:
        if not isinstance(value, list):
            return []
        k = self.info.kind
        out: list[list[Any]] = []
        for v in value:
            if k == "num_list":
                out.append([v])
            elif k == "vec_list":
                out.append(list(v) if isinstance(v, (list, tuple)) else [v, None])
            else:
                flat: list[Any] = []
                for pt in v if isinstance(v, (list, tuple)) else []:
                    flat += list(pt) if isinstance(pt, (list, tuple)) else [pt]
                out.append(flat)
        return out

    def _value_from(self, rows: list[list[Any]]) -> list[Any]:
        k = self.info.kind
        if k == "num_list":
            return [r[0] for r in rows]
        if k == "vec_list":
            return [list(r) for r in rows]
        return [[[r[0], r[1]], [r[2], r[3]]] for r in rows]

    def _on_item(self, item: QTableWidgetItem) -> None:
        ok, _v, err = parse_number(item.text(), formula=self.formula)
        item.setBackground(QColor(0, 0, 0, 0) if ok else QColor(217, 83, 79, 70))
        item.setToolTip(err or "")
        rows: list[list[Any]] = []
        for r in range(self.table.rowCount()):
            row: list[Any] = []
            for c in range(len(self.cols)):
                it = self.table.item(r, c)
                good, value, _ = parse_number(it.text() if it else "", formula=self.formula)
                if not good:
                    self.set_invalid(S.EDP_BAD_CELL)
                    return
                row.append(value)
            rows.append(row)
        self.set_invalid(None)
        self.write(self._value_from(rows))

    def _on_add(self) -> None:
        rows = self._rows_from(self.value())
        if rows:
            new = list(rows[-1])
            if all(isinstance(x, (int, float)) for x in new):
                new = [x + 1.0 for x in new]
        else:
            new = [0.0] * len(self.cols)
        self.write(self._value_from([*rows, new]), merge=False)

    def _on_remove(self) -> None:
        rows = self._rows_from(self.value())
        r = self.table.currentRow()
        if not rows:
            return
        r = r if 0 <= r < len(rows) else len(rows) - 1
        del rows[r]
        if not rows and self.info.optional:
            self.write(None, merge=False)
        else:
            self.write(self._value_from(rows), merge=False)

    def refresh(self) -> None:
        rows = self._rows_from(self.value())
        if rows == self._rows:
            return
        self._rows = rows
        t = self.table
        t.blockSignals(True)
        t.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c in range(len(self.cols)):
                text = fmt_value(row[c]) if c < len(row) else ""
                it = t.item(r, c)
                if it is None:
                    t.setItem(r, c, QTableWidgetItem(text))
                elif it.text() != text:
                    it.setText(text)
                    it.setBackground(QColor(0, 0, 0, 0))
        t.blockSignals(False)
        self.remove_button.setEnabled(len(rows) > (0 if self.info.optional else self.info.min_len))
        self.add_button.setEnabled(self.info.max_len is None or len(rows) < self.info.max_len)


class FormRows(QWidget):
    """모델 하나의 칸들(QFormLayout). kind_info를 주면 맨 위에 종류 고르기를 둔다."""

    def __init__(self, ctx: FormContext, path: JsonPath, model: type[BaseModel], *, exclude: tuple[str, ...] = (),
                 only: tuple[str, ...] | None = None, kind_info: TypeInfo | None = None,
                 kind_extra: QWidget | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.path = tuple(path)
        self.model = model
        lay = QFormLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        lay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        lay.setVerticalSpacing(4)
        self.layout_ = lay
        self.fields: dict[str, FieldWidget] = {}
        if kind_info is not None:
            kf = UnionKindField(ctx, (*self.path, "kind"), kind_info)
            self._add_row(field_name(None, "kind"), "", kf, kind_extra)
            self.fields["kind"] = kf
        for spec in fields_of(model):
            if spec.key in exclude or spec.key in ("version", "editor") or (only is not None and spec.key not in only):
                continue
            if kind_info is not None and spec.key == "kind":
                continue
            w = make_field(ctx, (*self.path, spec.key), spec)
            if w is None:
                continue
            self.fields[spec.key] = w
            if w.spans:
                lay.addRow(w)
            else:
                self._add_row(field_name(model, spec.key), w.tooltip(), w)

    def _add_row(self, text: str, tip: str, w: FieldWidget, extra: QWidget | None = None) -> None:
        label = QLabel(text)
        label.setToolTip(tip)
        w.label = label
        if extra is None:
            self.layout_.addRow(label, w)
        else:
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(w, 1)
            hl.addWidget(extra)
            self.layout_.addRow(label, holder)
        w._restyle()

    def refresh(self) -> None:
        for w in self.fields.values():
            w.refresh()

    def leaves(self) -> Iterator[FieldWidget]:
        for w in self.fields.values():
            yield from w.leaves()


class ModelGroup(FieldWidget):
    """묶음 칸(재질, 착륙 패드, 시작 위치): 제목 있는 상자 안에 칸들. 선택 칸이면 상자에 체크박스."""

    spans = True

    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, spec, parent)
        info = spec.info
        assert info.model is not None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.box = QGroupBox(field_name(spec.model, spec.key))
        self.box.setToolTip(self._tip)
        lay.addWidget(self.box)
        inner = QVBoxLayout(self.box)
        inner.setContentsMargins(6, 4, 6, 4)
        present = self.value() is not None
        self.rows: FormRows | None = None
        if info.optional:
            self.box.setCheckable(True)
            self.box.setChecked(present)
            self.box.clicked.connect(self._on_toggle)
        if present or not info.optional:
            self.rows = FormRows(ctx, self.path, info.model)
            inner.addWidget(self.rows)
        self.editor = None

    def _on_toggle(self, checked: bool) -> None:
        assert self.spec is not None and self.spec.info.model is not None
        self.write(new_item(self.spec.info.model, self.doc.data) if checked else None, merge=False)

    def refresh(self) -> None:
        if self.spec is not None and self.spec.info.optional:
            self.box.blockSignals(True)
            self.box.setChecked(self.value() is not None)
            self.box.blockSignals(False)
        if self.rows is not None:
            self.rows.refresh()

    def leaves(self) -> Iterator[FieldWidget]:
        yield self
        if self.rows is not None:
            yield from self.rows.leaves()

    def style_targets(self) -> list[QWidget]:
        return []

    def _restyle(self) -> None:
        self.box.setToolTip(self.tooltip())
        self.box.setStyleSheet(f"QGroupBox {{ color: {ISSUE_COLOR}; }}" if self._issues else "")

    def set_issues(self, issues: list[Issue]) -> None:
        # 묶음 자체를 가리키는 결과만(안쪽 칸의 결과는 그 칸이 보인다).
        msgs = [i.message for i in issues_at(issues, self.path, exact=True)]
        if msgs != self._issues:
            self._issues = msgs
            self._restyle()


class UnionGroup(FieldWidget):
    """종류를 고르는 묶음 하나(영역의 모양)."""

    spans = True

    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, spec, parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.box = QGroupBox(field_name(spec.model, spec.key))
        lay.addWidget(self.box)
        inner = QVBoxLayout(self.box)
        inner.setContentsMargins(6, 4, 6, 4)
        value = self.value()
        model = model_for(spec.info, value)
        self.rows: FormRows | None = None
        if model is not None:
            self.rows = FormRows(ctx, self.path, model, kind_info=spec.info)
            inner.addWidget(self.rows)
        else:
            kf = UnionKindField(ctx, (*self.path, "kind"), spec.info)
            inner.addWidget(kf)
            self._kind_only = kf
        self.editor = None

    def refresh(self) -> None:
        if self.rows is not None:
            self.rows.refresh()

    def leaves(self) -> Iterator[FieldWidget]:
        yield self
        if self.rows is not None:
            yield from self.rows.leaves()

    def style_targets(self) -> list[QWidget]:
        return []

    def set_issues(self, issues: list[Issue]) -> None:
        msgs = [i.message for i in issues_at(issues, self.path, exact=True)]
        if msgs != self._issues:
            self._issues = msgs
            self.box.setToolTip(self.tooltip())
            self.box.setStyleSheet(f"QGroupBox {{ color: {ISSUE_COLOR}; }}" if msgs else "")


class UnionListField(FieldWidget):
    """종류를 고르는 묶음 목록(물체의 모양들): 칸마다 종류 고르기와 지우기, 아래에 더하기."""

    spans = True

    def __init__(self, ctx: FormContext, path: JsonPath, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(ctx, path, spec, parent)
        info = spec.info
        assert info.item is not None
        self.item_info = info.item
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.items: list[FormRows | None] = []
        self.remove_buttons: list[QToolButton] = []
        value = self.value()
        value = value if isinstance(value, list) else []
        title = field_name(spec.model, spec.key)
        for i, item in enumerate(value):
            box = QGroupBox(S.EDP_ITEM_TITLE.format(name=title, n=i + 1))
            inner = QVBoxLayout(box)
            inner.setContentsMargins(6, 4, 6, 4)
            remove = QToolButton()
            remove.setText(S.EDP_REMOVE)
            remove.setToolTip(S.EDP_REMOVE_ITEM_TIP.format(name=title))
            remove.setEnabled(len(value) > info.min_len)
            remove.clicked.connect(functools.partial(self._on_remove, i))
            self.remove_buttons.append(remove)
            model = model_for(self.item_info, item)
            rows: FormRows | None = None
            if model is not None:
                rows = FormRows(ctx, (*self.path, i), model, kind_info=self.item_info, kind_extra=remove)
                inner.addWidget(rows)
            else:
                head = QHBoxLayout()
                head.addWidget(UnionKindField(ctx, (*self.path, i, "kind"), self.item_info), 1)
                head.addWidget(remove)
                inner.addLayout(head)
            self.items.append(rows)
            lay.addWidget(box)
        self.add_button = QToolButton()
        self.add_button.setText(S.EDP_ADD_ITEM.format(name=title))
        self.add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.add_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        menu = QMenu(self.add_button)
        for kind in kinds_of(self.item_info.models):
            act = menu.addAction(kind_name(kind))
            act.triggered.connect(functools.partial(self._on_add, kind))
        self.add_button.setMenu(menu)
        self.add_button.setEnabled(info.max_len is None or len(value) < info.max_len)
        lay.addWidget(self.add_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.editor = None

    def _on_add(self, kind: str) -> None:
        model = kinds_of(self.item_info.models)[kind]
        item = new_item(model, self.doc.data, kind=kind)
        path = self.path

        def mutate(d: dict[str, Any]) -> None:
            try:
                lst = get_path(d, path)
            except (KeyError, IndexError, TypeError):
                lst = None
            if not isinstance(lst, list):
                set_path(d, path, [])
                lst = get_path(d, path)
            lst.append(item)

        self.doc.edit(S.EDP_UNDO_ADD_ITEM.format(name=self.display_name()), mutate)

    def _on_remove(self, index: int) -> None:
        path = self.path
        self.doc.edit(S.EDP_UNDO_REMOVE_ITEM.format(name=self.display_name()),
                      lambda d: get_path(d, path).pop(index))

    def refresh(self) -> None:
        for rows in self.items:
            if rows is not None:
                rows.refresh()

    def leaves(self) -> Iterator[FieldWidget]:
        yield self
        for rows in self.items:
            if rows is not None:
                yield from rows.leaves()

    def style_targets(self) -> list[QWidget]:
        return []

    def set_issues(self, issues: list[Issue]) -> None:
        msgs = [i.message for i in issues_at(issues, self.path, exact=True)]
        if msgs != self._issues:
            self._issues = msgs
            self.add_button.setToolTip(self.tooltip())
            self.add_button.setStyleSheet(f"QToolButton {{ color: {ISSUE_COLOR}; }}" if msgs else "")


def make_field(ctx: FormContext, path: JsonPath, spec: FieldSpec) -> FieldWidget | None:
    """칸 타입에 맞는 위젯. 폼에 안 보이는 칸이면 None."""
    k = spec.info.kind
    if k in ("num", "float"):
        return NumField(ctx, path, spec)
    if k == "int":
        return IntField(ctx, path, spec)
    if k == "bool":
        return BoolField(ctx, path, spec)
    if k == "literal":
        return LiteralField(ctx, path, spec)
    if k == "ref":
        return RefField(ctx, path, spec)
    if k == "id" and spec.key == "id" and ctx.section is not None and tuple(path) == (*ctx.root, "id"):
        return IdField(ctx, path, spec)
    if k in ("id", "label", "text", "expr"):
        return TextField(ctx, path, spec, multiline=spec.key == "description" and not ctx.compact)
    if k == "vec":
        return VecField(ctx, path, spec)
    if k in ("vec_list", "num_list", "lines"):
        return TableField(ctx, path, spec)
    if k == "list":
        return ListField(ctx, path, spec)
    if k == "literal_list":
        return LiteralListField(ctx, path, spec)
    if k == "names_or_all":
        return NamesOrAllField(ctx, path, spec)
    if k == "dict":
        return DictField(ctx, path, spec)
    if k == "model":
        return ModelGroup(ctx, path, spec)
    if k == "union":
        return UnionGroup(ctx, path, spec)
    if k == "union_list":
        return UnionListField(ctx, path, spec)
    return None


# ---------------------------------------------------------------------------
# 폼 (부품 하나 또는 설정 구역 하나)
# ---------------------------------------------------------------------------

_STRUCTURAL = frozenset({"model", "union", "union_list", "model_list"})


def structure_of(value: Any, info: TypeInfo, exclude: tuple[str, ...] = (), only: tuple[str, ...] | None = None) -> Any:
    """위젯을 다시 만들어야 하는 구조(종류, 목록 길이, 묶음이 있는지). 값이 바뀌어도 같다."""
    k = info.kind
    if k in ("model", "union"):
        if value is None:
            return None
        model = model_for(info, value)
        if model is None:
            return ("?", value.get("kind") if isinstance(value, dict) else None)
        v = value if isinstance(value, dict) else {}
        parts = tuple(
            structure_of(v.get(s.key, s.default), s.info)
            for s in fields_of(model)
            if s.info.kind in _STRUCTURAL and s.key not in exclude and (only is None or s.key in only)
        )
        return (v.get("kind") if k == "union" else None, parts)
    if k in ("union_list", "model_list"):
        if not isinstance(value, list) or info.item is None:
            return ()
        return tuple(structure_of(v, info.item) for v in value)
    return None


class ModelForm(QWidget):
    """문서의 한 경로를 보여 주는 폼. doc.changed에 맞춰 값만 고치고, 구조가 바뀌면 다시 만든다.

    model: 모델 클래스, 또는 종류로 고르는 합집합(m.Actuator 같은 Annotated 타입). 합집합이면 맨 위에
    종류 고르기를 둔다(exclude에 "kind"를 주면 빼고).
    """

    def __init__(self, doc: EnvDocument, path: JsonPath | None = None, model: Any = None, *,
                 exclude: tuple[str, ...] = (), only: tuple[str, ...] | None = None, section: str | None = None,
                 auto: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc = doc
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self.rows: FormRows | None = None
        self._content: QWidget | None = None
        self._sig: Any = None
        self.path: JsonPath | None = None
        self.info: TypeInfo | None = None
        self.exclude: tuple[str, ...] = ()
        self.only: tuple[str, ...] | None = None
        self.section: str | None = None
        self.rebuilds = 0
        if auto:
            doc.changed.connect(self.refresh)
            doc.checked.connect(self.apply_issues)
        if path is not None:
            self.set_target(path, model, exclude=exclude, only=only, section=section)

    def set_target(self, path: JsonPath | None, model: Any = None, *, exclude: tuple[str, ...] = (),
                   only: tuple[str, ...] | None = None, section: str | None = None) -> None:
        info = type_info(model) if model is not None else None
        target = (tuple(path) if path is not None else None, info, tuple(exclude), only, section)
        if target == (self.path, self.info, self.exclude, self.only, self.section) and self._sig == self._signature():
            return
        self.path, self.info, self.exclude, self.only, self.section = target
        self._rebuild()

    def clear(self) -> None:
        self.set_target(None)

    def _signature(self) -> Any:
        if self.path is None or self.info is None:
            return None
        value = self.doc.get(self.path)
        return (self.path, self.info, evolvable(self.doc.data, self.path),
                structure_of(value, self.info, self.exclude, self.only))

    def _rebuild(self) -> None:
        old = self._content
        if old is not None:
            for w in [old, *old.findChildren(QWidget)]:
                w.blockSignals(True)
            old.hide()
            self._lay.removeWidget(old)
            old.deleteLater()
        self._content = None
        self.rows = None
        self._sig = self._signature()
        self.rebuilds += 1
        if self.path is None or self.info is None:
            return
        value = self.doc.get(self.path)
        model = model_for(self.info, value)
        content = QWidget(self)
        lay = QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 0, 0)
        ctx = FormContext(self.doc, evolvable=evolvable(self.doc.data, self.path), section=self.section,
                          root=self.path)
        union = self.info.kind == "union"
        kind_info = self.info if union and "kind" not in self.exclude else None
        if model is not None:
            self.rows = FormRows(ctx, self.path, model, exclude=self.exclude, only=self.only, kind_info=kind_info)
            lay.addWidget(self.rows)
        elif union and isinstance(value, dict):
            lay.addWidget(QLabel(S.EDP_UNKNOWN_VALUE.format(value=value.get("kind"))))
            if kind_info is not None:
                lay.addWidget(UnionKindField(ctx, (*self.path, "kind"), self.info))
        self._lay.addWidget(content)
        content.show()  # 레이아웃은 보이기를 다음 이벤트로 미룬다: 바로 보이게
        self._content = content
        self.apply_issues()

    def refresh(self) -> None:
        if self._signature() != self._sig:
            self._rebuild()
            return
        if self.rows is not None:
            self.rows.refresh()

    def apply_issues(self, issues: list[Issue] | None = None) -> None:
        if self.path is None:
            return
        rel = issues_at(self.doc.issues if issues is None else issues, self.path)
        for leaf in self.leaves():
            leaf.set_issues(rel)

    def leaves(self) -> Iterator[FieldWidget]:
        if self.rows is not None:
            yield from self.rows.leaves()

    def field(self, *keys: str | int) -> FieldWidget | None:
        """경로(폼 경로 뒤쪽)로 칸 위젯 찾기. 예: field("material", "density"), field("shapes", 0, "w")."""
        target = (*(self.path or ()), *keys)
        for leaf in self.leaves():
            if leaf.path == target:
                return leaf
        return None


class _FocusRelay(QObject):
    """칸 위젯 안쪽이 포커스를 받으면 알린다(표의 현재 줄 맞추기)."""

    def __init__(self, callback: Callable[[], None], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._callback = callback

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.FocusIn:
            self._callback()
        return False


def watch_focus(widget: QWidget, callback: Callable[[], None]) -> None:
    relay = _FocusRelay(callback, widget)
    for w in [widget, *widget.findChildren(QWidget)]:
        w.installEventFilter(relay)

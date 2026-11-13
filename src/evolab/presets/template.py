"""틀 파일 `.evotpl` 읽기·펼치기·넣기 (6.2.0, 설계 DU 2.2·4.2 + 최종 설계 3.1·D44·D55, Qt 없음).

틀은 슬라이더 값 목록(params)과 환경 파일과 같은 모양의 부품 묶음(parts)이다. 숫자 칸에 `"=t.size * 0.3"`처럼 슬라이더 값을 쓰는
수식을 적을 수 있고, 넣을 때 한 번 계산해 보통 숫자(소수 6자리)로 적는다. `t.*` 밖의 이름(`r.tilt` 같은 랜덤 값)을 쓰는 수식은
`t.*`만 숫자로 바꿔 수식으로 남긴다. 그래서 틀로 만든 환경 파일은 틀을 모르는 보통 환경 파일이다.

- 몸 틀(`body`)의 좌표는 발바닥(위에서 본 몸은 가운데)이 (0, 0)이고 +x를 본다. 몸 틀은 `spawn`(시작 자세: 기울기·속도 수식)과
  `randoms`(그 수식이 쓰는 랜덤 값)를 가질 수 있다(달리기의 시작 기울기·밀기).
- 물건(`prop`)·경기장(`arena`)은 틀의 가운데가 (0, 0)인 세계 물체·영역이다. 경기장은 세계 경계(`bounds`), 기본 보기(`view`),
  자리 이름표(`spawns`: 이름 → 좌표 목록)를 더 가진다.
- 빈자리(`slots`): 틀 안의 `$ground` 같은 이름은 넣을 때 장면에서 그 태그가 있는 것으로 채운다. 태그는 여럿을 앞의 것부터
  본다(`"tags": ["ground", "terrain"]`, D44). 장면에 없으면 첫 태그 `tag.<태그>`로 둔다(못 채운 것은 resolve_slots가 알림).
- 내장 틀의 화면 글은 strings.py를 id로 가리킨다(D55): `name_id` → TPL_NAMES·TPL_BLURBS, 슬라이더의 `label_id` → TPL_PARAM_LABELS,
  `unit_id` → TPL_UNIT_NAMES, 빈자리·키 묶음의 `label_id`. 사용자 틀은 `name`·`label`·`unit` 글을 직접 담는다.

읽을 때 검사(TemplateError, 한국어 글): 모양, 모르는 칸·종류, 뒤집힌 슬라이더 범위, 틀 안에 없는 것을 가리키는 참조, 빈자리 표에
없는 `$이름`, 슬라이더의 기본·최소·최대 값에서 펼친 결과가 환경 파일 모델에 맞지 않음.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evolab.app import strings as S
from evolab.scoring.expr import ExprError, Sym, compile_expr
from evolab.spec import model as m

TEMPLATE_FORMAT = "evolab-template"
TEMPLATE_VERSION = 1
"""이 프로그램이 읽는 틀 판. 더 큰 판은 TPL_ERR_NEWER로 건너뛴다."""
KINDS = ("body", "prop", "arena")
WORLDS = ("side", "top", "any")
SECTIONS = ("bodies", "joints", "actuators", "sensors", "regions", "generators")
"""부품 묶음의 구역(환경 파일의 같은 이름 목록에 들어간다)."""
MAX_MAIN_PARAMS = 3
"""겉에 보이는 슬라이더는 틀마다 셋까지(DU D15)."""

TEMPLATE_KEYS = frozenset({
    "format", "version", "id", "kind", "world", "name", "name_id", "description", "params", "root", "parts", "slots", "keys",
    "evolvable", "spawns", "rules", "randoms", "spawn", "bounds", "view",
})
PARAM_KEYS = frozenset({"id", "label", "label_id", "value", "min", "max", "step", "unit", "unit_id", "main"})
SLOT_KEYS = frozenset({"tags", "tag", "label", "label_id"})
CHORD_KEYS = frozenset({"key", "label", "label_id", "outputs"})

REF_KEYS = frozenset({"a", "b", "body", "joint", "place_by", "target", "tank", "checkpoints"})
"""부품 안에서 다른 부품을 가리키는 칸(하나)."""
REF_LISTS = frozenset({"with", "hits", "ground", "actuators"})
"""부품 안에서 다른 부품을 가리키는 칸(목록)."""
KEY_FIELDS = ("keys", "keys_pos", "keys_neg", "keys_up", "keys_down", "keys_left", "keys_right", "keys_forward", "keys_back")
"""직접 조종 키 칸(keys=False로 펼치면 비운다)."""

_SLOT_NAME = re.compile(r"^\$[a-z_][a-z0-9_]*$")
_T_NAME = re.compile(r"\bt\.([A-Za-z_][A-Za-z0-9_]*)")
_ID = re.compile(r"^[A-Za-z_가-힣][A-Za-z0-9_가-힣]*$")

Point = tuple[float, float]
Ref = tuple[str, str]
"""(구역, id): 넣은 부품 하나."""
Parts = dict[str, Any]
"""expand의 결과: SECTIONS의 목록(숫자만) + "randomize"(랜덤 값 목록), "spawn"(몸 틀의 시작 자세 사전, 아니면 None),
"root"(뿌리 물체의 id, 아니면 None), "params"(진화 슬라이더의 몸 수치: 6.2.0은 늘 빈 목록), "keys"(키 묶음)."""


class TemplateError(ValueError):
    """틀을 읽거나 펼칠 수 없음. 글은 한국어(strings.TPL_ERR_*)."""


@dataclass(frozen=True)
class Param:
    """슬라이더 하나. label·unit은 화면 글(내장 틀은 label_id·unit_id로 strings.py에서 찾은 것)."""

    id: str
    label: str
    value: float
    min: float
    max: float
    step: float = 0.05
    unit: str = ""
    main: bool = True
    label_id: str | None = None

    @property
    def tip(self) -> str:
        """풍선 도움말(흔한 id는 TPL_PARAM_TIPS, 없으면 빈 글)."""
        return S.TPL_PARAM_TIPS.get(self.label_id or self.id, "")

    def clamp(self, value: float) -> float:
        return min(self.max, max(self.min, float(value)))


@dataclass(frozen=True)
class Slot:
    """빈자리: 넣을 때 장면에서 태그로 찾는 대상. tags는 앞의 것부터 본다(D44)."""

    name: str
    tags: tuple[str, ...]
    label: str = ""
    label_id: str | None = None


@dataclass(frozen=True)
class KeyChord:
    """조종 키 묶음(editor.keys, 6.3.0-B에서 쓴다). 6.2.0은 읽어 두기만 한다."""

    key: str
    label: str
    outputs: Mapping[str, float]
    label_id: str | None = None


@dataclass(frozen=True)
class Template:
    id: str
    kind: str
    world: str
    name: str
    description: str
    params: tuple[Param, ...]
    parts: Mapping[str, Any]
    slots: Mapping[str, Slot]
    keys: tuple[KeyChord, ...] = ()
    root: str | None = None
    evolvable: tuple[str, ...] = ()
    spawns: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    rules: tuple[Mapping[str, Any], ...] = ()
    source: str = "user"
    path: Path | None = None
    randoms: tuple[Mapping[str, Any], ...] = ()
    spawn: Mapping[str, Any] | None = None
    name_id: str | None = None
    bounds: tuple[Any, ...] | None = None
    view: tuple[Any, ...] | None = None
    sha256: str = ""
    """틀 파일 글의 sha256(template_hash). 파일 없이 사전에서 읽은 틀은 정규 JSON의 sha256."""

    def param(self, pid: str) -> Param:
        for p in self.params:
            if p.id == pid:
                return p
        raise KeyError(pid)

    def defaults(self) -> dict[str, float]:
        return {p.id: p.value for p in self.params}

    def corners(self) -> list[tuple[str, dict[str, float]]]:
        """(which, 값): 기본값, 모든 슬라이더 최솟값, 모든 슬라이더 최댓값. which는 TPL_EXPAND_WHICH의 키."""
        return [("value", self.defaults()), ("min", {p.id: p.min for p in self.params}),
                ("max", {p.id: p.max for p in self.params})]

    def main_params(self) -> tuple[Param, ...]:
        return tuple(p for p in self.params if p.main)


def template_hash(tpl: Template) -> str:
    """틀 파일 글의 sha256. 조리법 표지(editor.recipe.templates)에 적는다(D43)."""
    return tpl.sha256


# ---------------------------------------------------------------------------
# 수식
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _compiled(source: str, names: tuple[str, ...]) -> Any:
    syms = {f"t.{k}": Sym("num", "build") for k in names}
    return compile_expr(source, syms, want="num", max_phase="build")


def _num(x: float) -> float:
    return round(float(x), 6) + 0.0


def _fmt(x: float) -> str:
    return repr(float(x))


def _eval_formula(text: str, values: Mapping[str, float]) -> Any:
    """`=` 수식 하나. t.*과 pi만 쓰면 숫자(소수 6자리), 다른 이름을 쓰면 t.*만 숫자로 바꾼 수식."""
    body = text[1:]
    names = tuple(sorted(values))
    try:
        fn = _compiled(body, names)
    except ExprError:
        if not _T_NAME.search(body):
            return text
        return "=" + _T_NAME.sub(lambda mt: f"({_fmt(values[mt.group(1)])})" if mt.group(1) in values else mt.group(0),
                                 body)
    return _num(fn({f"t.{k}": float(v) for k, v in values.items()}))


def evaluate(value: Any, values: Mapping[str, float]) -> Any:
    """사전·목록 안의 모든 `=` 수식을 슬라이더 값으로 계산한다(새 사본)."""
    if isinstance(value, str):
        return _eval_formula(value, values) if value.startswith("=") else value
    if isinstance(value, list):
        return [evaluate(v, values) for v in value]
    if isinstance(value, tuple):
        return [evaluate(v, values) for v in value]
    if isinstance(value, Mapping):
        return {k: evaluate(v, values) for k, v in value.items()}
    return value


def _strings(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _strings(v, f"{path}[{i}]")
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield from _strings(v, f"{path}.{k}" if path else str(k))


# ---------------------------------------------------------------------------
# 읽기
# ---------------------------------------------------------------------------


def _field_error(where: str, reason: str) -> TemplateError:
    return TemplateError(S.TPL_ERR_FIELD.format(where=where, reason=reason))


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise _field_error(where, S.SPEC_ERR_NUMBER)
    return float(value)


def _text_of(table: Mapping[str, str], key: Any, where: str) -> str:
    if not isinstance(key, str) or key not in table:
        raise _field_error(where, S.SPEC_ERR_LITERAL.format(expected=", ".join(sorted(table))))
    return table[key]


def _parse_params(raw: Any, builtin: bool) -> tuple[Param, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise _field_error("params", S.SPEC_ERR_LIST)
    out: list[Param] = []
    seen: set[str] = set()
    for i, p in enumerate(raw):
        where = f"params[{i}]"
        if not isinstance(p, Mapping):
            raise _field_error(where, S.SPEC_ERR_OBJECT)
        extra = sorted(set(p) - PARAM_KEYS)
        if extra:
            raise _field_error(f"{where}.{extra[0]}", S.SPEC_ERR_EXTRA)
        pid = p.get("id")
        if not isinstance(pid, str) or not _ID.match(pid):
            raise _field_error(f"{where}.id", S.SPEC_ERR_PATTERN if isinstance(pid, str) else S.SPEC_ERR_MISSING)
        if pid in seen:
            raise _field_error(f"{where}.id", S.SPEC_ID_DUPLICATE.format(id=pid, other="params"))
        seen.add(pid)
        for k in ("value", "min", "max"):
            if k not in p:
                raise _field_error(f"{where}.{k}", S.SPEC_ERR_MISSING)
        value, lo, hi = (_number(p[k], f"{where}.{k}") for k in ("value", "min", "max"))
        if not lo <= value <= hi:
            raise TemplateError(S.TPL_ERR_PARAM_RANGE.format(id=pid))
        step = _number(p.get("step", 0.05), f"{where}.step")
        if step <= 0:
            raise _field_error(f"{where}.step", S.SPEC_ERR_RANGE.format(limit="> 0"))
        label_id = p.get("label_id")
        if label_id is not None:
            label = _text_of(S.TPL_PARAM_LABELS, label_id, f"{where}.label_id")
        elif isinstance(p.get("label"), str) and p["label"]:
            label = p["label"]
        elif builtin and pid in S.TPL_PARAM_LABELS:
            label, label_id = S.TPL_PARAM_LABELS[pid], pid
        else:
            raise _field_error(f"{where}.label", S.SPEC_ERR_MISSING)
        if p.get("unit_id") is not None:
            unit = _text_of(S.TPL_UNIT_NAMES, p["unit_id"], f"{where}.unit_id")
        else:
            unit = p.get("unit", "")
            if not isinstance(unit, str):
                raise _field_error(f"{where}.unit", S.SPEC_ERR_STRING)
        main = p.get("main", True)
        if not isinstance(main, bool):
            raise _field_error(f"{where}.main", S.SPEC_ERR_BOOL)
        out.append(Param(pid, label, value, lo, hi, step, unit, main, label_id))
    if sum(1 for p in out if p.main) > MAX_MAIN_PARAMS:
        raise _field_error("params", S.SPEC_ERR_LIST_LONG.format(max=MAX_MAIN_PARAMS))
    return tuple(out)


def _parse_slots(raw: Any) -> dict[str, Slot]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise _field_error("slots", S.SPEC_ERR_OBJECT)
    out: dict[str, Slot] = {}
    for name, spec in raw.items():
        where = f"slots.{name}"
        if not isinstance(name, str) or not _SLOT_NAME.match(name):
            raise _field_error(where, S.SPEC_ERR_PATTERN)
        if isinstance(spec, str):  # 줄임: "$ground": "tag.ground"
            spec = {"tags": [spec[4:] if spec.startswith("tag.") else spec]}
        if not isinstance(spec, Mapping):
            raise _field_error(where, S.SPEC_ERR_OBJECT)
        extra = sorted(set(spec) - SLOT_KEYS)
        if extra:
            raise _field_error(f"{where}.{extra[0]}", S.SPEC_ERR_EXTRA)
        tags = spec.get("tags", [spec["tag"]] if "tag" in spec else None)
        if isinstance(tags, str):
            tags = [tags]
        if not isinstance(tags, list) or not tags or not all(isinstance(t, str) and _ID.match(t) for t in tags):
            raise _field_error(f"{where}.tags", S.SPEC_ERR_LIST_SHORT.format(min=1))
        label_id = spec.get("label_id")
        if label_id is not None:
            label = _text_of(S.TPL_NAMES, label_id, f"{where}.label_id")
        else:
            label = spec.get("label", "") if isinstance(spec.get("label", ""), str) else ""
        out[name] = Slot(name, tuple(tags), label or tags[0], label_id)
    return out


def _parse_keys(raw: Any) -> tuple[KeyChord, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise _field_error("keys", S.SPEC_ERR_LIST)
    out: list[KeyChord] = []
    for i, k in enumerate(raw):
        where = f"keys[{i}]"
        if not isinstance(k, Mapping):
            raise _field_error(where, S.SPEC_ERR_OBJECT)
        extra = sorted(set(k) - CHORD_KEYS)
        if extra:
            raise _field_error(f"{where}.{extra[0]}", S.SPEC_ERR_EXTRA)
        key = k.get("key")
        if not isinstance(key, str) or not key:
            raise _field_error(f"{where}.key", S.SPEC_ERR_MISSING)
        outputs = k.get("outputs", {})
        if not isinstance(outputs, Mapping):
            raise _field_error(f"{where}.outputs", S.SPEC_ERR_OBJECT)
        label_id = k.get("label_id")
        label = (_text_of(getattr(S, "KEY_CHORD_LABELS", {}), label_id, f"{where}.label_id") if label_id is not None
                 else str(k.get("label", "")))
        out.append(KeyChord(key, label, {str(a): _number(v, f"{where}.outputs.{a}") for a, v in outputs.items()},
                            label_id))
    return tuple(out)


def _part_ids(parts: Mapping[str, Any]) -> dict[str, str]:
    """틀 안의 부품 id → 구역."""
    out: dict[str, str] = {}
    for sec in SECTIONS:
        for i, item in enumerate(parts.get(sec, [])):
            if not isinstance(item, Mapping):
                raise _field_error(f"parts.{sec}[{i}]", S.SPEC_ERR_OBJECT)
            pid = item.get("id")
            if sec == "generators" and pid is None:
                continue
            if not isinstance(pid, str) or not _ID.match(pid):
                raise _field_error(f"parts.{sec}[{i}].id",
                                   S.SPEC_ERR_PATTERN if isinstance(pid, str) else S.SPEC_ERR_MISSING)
            if pid in out:
                raise _field_error(f"parts.{sec}[{i}].id", S.SPEC_ID_DUPLICATE.format(id=pid, other=out[pid]))
            out[pid] = sec
    return out


def _check_refs(parts: Mapping[str, Any], ids: Mapping[str, str], slots: Mapping[str, Slot]) -> None:
    def check(value: Any, where: str) -> None:
        if not isinstance(value, str):
            return
        if value.startswith("$"):
            if value not in slots:
                raise TemplateError(S.TPL_ERR_SLOT.format(name=value))
        elif value.startswith("tag.") or value == "all":
            return
        elif value not in ids:
            raise TemplateError(S.TPL_ERR_REF.format(name=value, where=where))

    for sec in SECTIONS:
        for i, item in enumerate(parts.get(sec, [])):
            for k, v in item.items():
                where = f"parts.{sec}[{i}].{k}"
                if k in REF_KEYS:
                    check(v, where)
                elif k in REF_LISTS and isinstance(v, list):
                    for j, x in enumerate(v):
                        check(x, f"{where}[{j}]")
    # 빈자리를 쓰는 곳이 참조 칸 밖이어도(모양 안의 글 등) 빈자리 표에 있어야 한다
    for where, text in _strings(parts):
        if text.startswith("$") and text not in slots:
            raise TemplateError(S.TPL_ERR_SLOT.format(name=text))


def _check_formulas(data: Mapping[str, Any], params: Sequence[Param]) -> None:
    names = {p.id for p in params}
    for key in ("parts", "spawn", "bounds", "view", "spawns", "randoms"):
        for where, text in _strings(data.get(key), key):
            if not text.startswith("="):
                continue
            for mt in _T_NAME.finditer(text):
                if mt.group(1) not in names:
                    raise _field_error(where, S.EXPR_ERR_UNKNOWN_NAME.format(name=mt.group(0)))
            uses_other = bool(re.search(r"\b(?!t\.)[A-Za-z_]+\.[A-Za-z_]", text[1:]))
            if not uses_other:
                try:
                    _compiled(text[1:], tuple(sorted(names)))
                except ExprError as exc:
                    raise _field_error(where, str(exc)) from None


def _sha256_of(data: Mapping[str, Any]) -> str:
    text = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_template(data: Mapping[str, Any], *, source: str = "user", path: Path | None = None,
                   sha256: str | None = None) -> Template:
    """틀 사전을 검사해 Template으로. 잘못됐으면 TemplateError(한국어). source: "builtin" | "user"."""
    if not isinstance(data, Mapping):
        raise TemplateError(S.TPL_ERR_FORMAT.format(reason=S.SPEC_NOT_OBJECT))
    if data.get("format") != TEMPLATE_FORMAT:
        raise TemplateError(S.TPL_ERR_FORMAT.format(reason=S.SPEC_ERR_LITERAL.format(expected=f'"{TEMPLATE_FORMAT}"')))
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise _field_error("version", S.SPEC_ERR_INT)
    if version > TEMPLATE_VERSION:
        raise TemplateError(S.TPL_ERR_NEWER.format(version=version))
    extra = sorted(k for k in data if k not in TEMPLATE_KEYS)
    if extra:
        raise _field_error(extra[0], S.SPEC_ERR_EXTRA)
    tid = data.get("id")
    if not isinstance(tid, str) or not _ID.match(tid):
        raise _field_error("id", S.SPEC_ERR_PATTERN if isinstance(tid, str) else S.SPEC_ERR_MISSING)
    kind = data.get("kind")
    if kind not in KINDS:
        raise _field_error("kind", S.SPEC_ERR_LITERAL.format(expected=", ".join(KINDS)))
    world = data.get("world", "any")
    if world not in WORLDS:
        raise _field_error("world", S.SPEC_ERR_LITERAL.format(expected=", ".join(WORLDS)))
    builtin = source == "builtin"
    name_id = data.get("name_id")
    if name_id is not None:
        name = _text_of(S.TPL_NAMES, name_id, "name_id")
        description = S.TPL_BLURBS.get(name_id, "")
    else:
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise _field_error("name", S.SPEC_ERR_MISSING)
        description = data.get("description", "")
        if not isinstance(description, str):
            raise _field_error("description", S.SPEC_ERR_STRING)
    params = _parse_params(data.get("params"), builtin)
    slots = _parse_slots(data.get("slots"))
    keys = _parse_keys(data.get("keys"))
    parts = data.get("parts", {})
    if not isinstance(parts, Mapping):
        raise _field_error("parts", S.SPEC_ERR_OBJECT)
    bad = sorted(k for k in parts if k not in SECTIONS)
    if bad:
        raise _field_error(f"parts.{bad[0]}", S.SPEC_ERR_EXTRA)
    for sec in SECTIONS:
        if not isinstance(parts.get(sec, []), list):
            raise _field_error(f"parts.{sec}", S.SPEC_ERR_LIST)
    ids = _part_ids(parts)
    _check_refs(parts, ids, slots)
    root = data.get("root")
    if root is not None and ids.get(root) != "bodies":
        raise TemplateError(S.TPL_ERR_REF.format(name=root, where="root"))
    if kind == "body":
        if root is None:
            agent = [b["id"] for b in parts.get("bodies", []) if b.get("owner", "agent") == "agent"]
            if not agent:
                raise _field_error("root", S.SPEC_ERR_MISSING)
            root = agent[0]
    randoms = data.get("randoms") or []
    if not isinstance(randoms, list):
        raise _field_error("randoms", S.SPEC_ERR_LIST)
    for i, r in enumerate(randoms):
        try:
            m.RandVar.model_validate(r)
        except ValidationError as exc:
            raise _field_error(f"randoms[{i}]", _first_error(exc)) from None
    spawn = data.get("spawn")
    if spawn is not None:
        if not isinstance(spawn, Mapping):
            raise _field_error("spawn", S.SPEC_ERR_OBJECT)
        bad = sorted(set(spawn) - set(m.Spawn.model_fields))
        if bad:
            raise _field_error(f"spawn.{bad[0]}", S.SPEC_ERR_EXTRA)
    spawns_raw = data.get("spawns") or {}
    if not isinstance(spawns_raw, Mapping):
        raise _field_error("spawns", S.SPEC_ERR_OBJECT)
    spawns: dict[str, tuple[Any, ...]] = {}
    for sname, pts in spawns_raw.items():
        if not isinstance(pts, list) or not pts or not all(isinstance(p, list) and len(p) == 2 for p in pts):
            raise _field_error(f"spawns.{sname}", S.SPEC_ERR_LIST_SHORT.format(min=1))
        spawns[str(sname)] = tuple(tuple(p) for p in pts)
    for key in ("bounds", "view"):
        box = data.get(key)
        if box is not None and (not isinstance(box, list) or len(box) != 4):
            raise _field_error(key, S.SPEC_ERR_LIST)
    evolvable = data.get("evolvable") or []
    if not isinstance(evolvable, list) or not all(isinstance(e, str) and e in {p.id for p in params} for e in evolvable):
        raise _field_error("evolvable", S.SPEC_ERR_LITERAL.format(expected=", ".join(p.id for p in params)))
    rules = data.get("rules") or []
    if not isinstance(rules, list) or not all(isinstance(r, Mapping) for r in rules):
        raise _field_error("rules", S.SPEC_ERR_LIST)
    _check_formulas(data, params)
    tpl = Template(
        id=tid, kind=kind, world=world, name=name, description=description, params=params,
        parts=copy.deepcopy(dict(parts)), slots=slots, keys=keys, root=root, evolvable=tuple(evolvable), spawns=spawns,
        rules=tuple(copy.deepcopy(list(rules))), source=source, path=path,
        randoms=tuple(copy.deepcopy(list(randoms))), spawn=copy.deepcopy(dict(spawn)) if spawn is not None else None,
        name_id=name_id, bounds=tuple(data["bounds"]) if data.get("bounds") is not None else None,
        view=tuple(data["view"]) if data.get("view") is not None else None,
        sha256=sha256 or _sha256_of(data),
    )
    _check_corners(tpl)
    return tpl


def _first_error(exc: ValidationError) -> str:
    from evolab.spec.io import translate_validation_error

    issues = translate_validation_error(exc)
    return f"{issues[0].where}: {issues[0].message}" if issues else str(exc)


def corner_env(tpl: Template, values: Mapping[str, float] | None = None, *, mirror: bool = False) -> dict[str, Any]:
    """틀 하나를 펼쳐 넣은 가장 작은 환경 사전(빈자리는 그 태그를 가진 정적 땅 하나로 채움). 검사·테스트용."""
    data: dict[str, Any] = {"version": m.CURRENT_VERSION, "name": tpl.id,
                            "world": {"gravity": [0.0, 0.0] if tpl.world == "top" else [0.0, -9.81]}}
    if tpl.world == "top":
        data["world"]["view"] = "top"
    stubs = []
    for slot in tpl.slots.values():
        stubs.append({"id": f"slot_{slot.tags[0]}", "owner": "world", "type": "static", "tags": [slot.tags[0]],
                      "pos": [0.0, -50.0], "shapes": [{"kind": "box", "w": 1.0, "h": 1.0}]})
    data["bodies"] = stubs
    parts = expand(tpl, values, mirror=mirror)
    insert(data, parts)
    if tpl.bounds is not None:
        data["world"]["bounds"] = evaluate(list(tpl.bounds), _values(tpl, values))
    return data


def _check_corners(tpl: Template) -> None:
    """슬라이더의 기본·최소·최대 값에서 펼친 결과가 환경 파일 모델에 맞는가(구조 검사)."""
    for which, values in tpl.corners():
        try:
            data = corner_env(tpl, values)
            m.EnvFile.model_validate(data)
        except ValidationError as exc:
            raise TemplateError(S.TPL_ERR_EXPAND.format(which=S.TPL_EXPAND_WHICH[which],
                                                        reason=_first_error(exc))) from None
        except (ExprError, TypeError, ValueError, KeyError) as exc:
            if isinstance(exc, TemplateError):
                raise
            raise TemplateError(S.TPL_ERR_EXPAND.format(which=S.TPL_EXPAND_WHICH[which], reason=exc)) from None


def load_template(path: Path, *, source: str = "user") -> Template:
    """틀 파일 하나. sha256은 파일 글(바이트)의 것."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise TemplateError(S.TPL_ERR_FORMAT.format(reason=exc.strerror or exc)) from None
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        reason = (S.SPEC_BAD_JSON.format(line=exc.lineno, col=exc.colno) if isinstance(exc, json.JSONDecodeError)
                  else S.SPEC_NOT_UTF8)
        raise TemplateError(S.TPL_ERR_FORMAT.format(reason=reason)) from None
    except RecursionError:
        raise TemplateError(S.TPL_ERR_FORMAT.format(reason=S.SPEC_TOO_DEEP)) from None
    return parse_template(data, source=source, path=Path(path), sha256=hashlib.sha256(raw).hexdigest())


# ---------------------------------------------------------------------------
# 펼치기
# ---------------------------------------------------------------------------


def _values(tpl: Template, values: Mapping[str, float] | None) -> dict[str, float]:
    """기본값 위에 values를 얹고 슬라이더 범위로 자른다(늘 조립되는 몸만: D15). 모르는 이름은 버린다."""
    out = tpl.defaults()
    for k, v in (values or {}).items():
        if k in out and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            out[k] = tpl.param(k).clamp(float(v))
    return out


def _negate(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _num(-float(value))
    if isinstance(value, str) and value.startswith("="):
        return f"=-({value[1:]})"
    return value


def _mirror_dir(value: Any) -> Any:
    """방향 각 d → π − d."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(math.pi - float(value), 9)
    if isinstance(value, str) and value.startswith("="):
        return f"=pi - ({value[1:]})"
    return value


def _mirror_vec(v: Any) -> Any:
    if isinstance(v, list) and len(v) == 2:
        return [_negate(v[0]), v[1]]
    return v


def _mirror_shape(shape: dict[str, Any]) -> dict[str, Any]:
    s = dict(shape)
    kind = s.get("kind")
    if "offset" in s:
        s["offset"] = _mirror_vec(s["offset"])
    if kind == "box" and "angle" in s:
        s["angle"] = _negate(s["angle"])
    elif kind in ("poly", "chain") and isinstance(s.get("points"), list):
        pts = [_mirror_vec(p) for p in s["points"]]
        s["points"] = pts[::-1] if kind == "poly" else pts
    elif kind == "radial" and isinstance(s.get("radii"), list) and s["radii"]:
        r = s["radii"]
        s["radii"] = [r[0], *r[:0:-1]]
        s["angle"] = _mirror_dir(s.get("angle", 0.0))
    elif kind == "wing":
        s["le"] = _mirror_vec(s.get("le", [0.0, 0.0]))
        s["angle"] = _mirror_dir(s.get("angle", 0.0))
        s["camber"] = _negate(s.get("camber", 0.0))
    return s


def _mirror_item(sec: str, item: dict[str, Any]) -> None:
    if "pos" in item:
        item["pos"] = _mirror_vec(item["pos"])
    if sec == "bodies":
        if "angle" in item:
            item["angle"] = _negate(item["angle"])
        if isinstance(item.get("shapes"), list):
            item["shapes"] = [_mirror_shape(s) if isinstance(s, dict) else s for s in item["shapes"]]
    elif sec == "regions" and isinstance(item.get("shape"), dict):
        item["shape"] = _mirror_shape(item["shape"])
    elif sec == "joints":
        for k in ("anchor_a", "anchor_b"):
            if k in item:
                item[k] = _mirror_vec(item[k])
        if isinstance(item.get("limit"), list) and len(item["limit"]) == 2:
            lo, hi = item["limit"]
            item["limit"] = [_negate(hi), _negate(lo)]
        if "rest_angle" in item:
            item["rest_angle"] = _negate(item["rest_angle"])
    elif sec == "actuators":
        kind = item.get("kind")
        if kind == "thruster":
            item["dir"] = _mirror_dir(item.get("dir", math.pi / 2))
            if "at" in item:
                item["at"] = _mirror_vec(item["at"])
        elif kind == "jump":
            item["dir"] = _mirror_dir(item.get("dir", math.pi / 2))
        elif kind == "motor":  # 같은 키가 「앞으로」가 되게 ＋/－ 키를 맞바꾼다
            item["keys_pos"], item["keys_neg"] = item.get("keys_neg", []), item.get("keys_pos", [])
        elif kind == "balance" and "target" in item:
            item["target"] = _negate(item["target"])
    elif sec == "sensors" and item.get("kind") == "rays":
        if "origin" in item:
            item["origin"] = _mirror_vec(item["origin"])
        item["center"] = _mirror_dir(item.get("center", 0.0))


def _shift(pos: Any, at: Point) -> Any:
    if not (isinstance(pos, list) and len(pos) == 2):
        pos = [0.0, 0.0]
    out = []
    for v, d in zip(pos, at, strict=True):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(_num(float(v) + d))
        elif d:
            out.append(f"=({v[1:]}) + {_fmt(d)}")
        else:
            out.append(v)
    return out


def expand(tpl: Template, values: Mapping[str, float] | None = None, *, at: Point = (0.0, 0.0), mirror: bool = False,
           prefix: str = "", label_prefix: str = "", slots: Mapping[str, str] | None = None, keys: bool = True) -> Parts:
    """틀을 부품 묶음(숫자만)으로: 슬라이더 값 계산(범위로 자름), 좌우 뒤집기, id 앞붙임, 빈자리 채우기, 자리 옮기기.

    세계 물체·영역은 at만큼 옮긴다. 몸 틀의 에이전트 물체는 에이전트 좌표 그대로 두고 시작 자세 "spawn"의 x·y에 at을 더한다
    (insert가 장면에 맞춰 놓는다). 같은 입력이면 같은 결과(순수 함수).
    """
    vals = _values(tpl, values)
    fill = {name: "tag." + slot.tags[0] for name, slot in tpl.slots.items()}
    fill.update(slots or {})
    parts = evaluate(copy.deepcopy(dict(tpl.parts)), vals)
    ids = _part_ids(tpl.parts)
    at = (float(at[0]), float(at[1]))

    def ref(v: Any) -> Any:
        if isinstance(v, str):
            if v in fill:
                return fill[v]
            if v in ids:
                return prefix + v
        return v

    out: Parts = {sec: [] for sec in SECTIONS}
    world_kind = tpl.kind != "body"
    for sec in SECTIONS:
        for raw in parts.get(sec, []):
            item = dict(raw)
            if "id" in item:
                item["id"] = prefix + item["id"]
            if label_prefix and item.get("label"):
                item["label"] = f"{label_prefix} {item['label']}"
            for k in list(item):
                if k in REF_KEYS:
                    item[k] = ref(item[k])
                elif k in REF_LISTS and isinstance(item[k], list):
                    item[k] = [ref(x) for x in item[k]]
            if mirror:
                _mirror_item(sec, item)
            owner_world = sec == "regions" or world_kind or item.get("owner", "agent") == "world"
            if sec in ("bodies", "regions") and owner_world and not (sec == "bodies" and item.get("place_by")):
                item["pos"] = _shift(item.get("pos", [0.0, 0.0]), at)
            if not keys and sec == "actuators":
                for k in KEY_FIELDS:
                    if k in item:
                        item[k] = []
            out[sec].append(item)
    out["randomize"] = [dict(r) for r in tpl.randoms]
    spawn: dict[str, Any] | None = None
    if tpl.kind == "body":
        spawn = evaluate(dict(tpl.spawn or {}), vals)
        if mirror:
            spawn = {k: (_negate(v) if k in ("x", "angle", "vx", "omega") else v) for k, v in spawn.items()}
        x, y = _shift([spawn.get("x", 0.0), spawn.get("y", 0.0)], at)
        spawn["x"], spawn["y"] = x, y
    out["spawn"] = spawn
    out["root"] = prefix + tpl.root if tpl.root else None
    out["params"] = []
    out["keys"] = [] if not keys else [
        {"key": c.key, "label": c.label, "outputs": {prefix + a: v for a, v in c.outputs.items()}} for c in tpl.keys]
    return out


# ---------------------------------------------------------------------------
# 빈자리와 넣기
# ---------------------------------------------------------------------------


def scene_tags(data: Mapping[str, Any]) -> set[str]:
    """장면의 물체·영역에 붙은 태그."""
    tags: set[str] = set()
    for sec in ("bodies", "regions"):
        for item in data.get(sec, []) or []:
            if isinstance(item, Mapping):
                tags.update(t for t in item.get("tags", []) or [] if isinstance(t, str))
    return tags


def resolve_slots(tpl: Template, data: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    """(채운 것 {"$ground": "tag.ground"}, 못 채운 빈자리 이름). 태그는 앞의 것부터 본다(D44)."""
    have = scene_tags(data)
    filled: dict[str, str] = {}
    missing: list[str] = []
    for name, slot in tpl.slots.items():
        tag = next((t for t in slot.tags if t in have), None)
        if tag is None:
            missing.append(name)
        else:
            filled[name] = "tag." + tag
    return filled, missing


ID_SECTIONS = ("bodies", "joints", "actuators", "sensors", "regions", "events", "checkpoints", "params")
"""id가 서로 겹치면 안 되는 구역(환경 파일의 모든 부품 id는 서로 다르다)."""


def all_ids(data: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for sec in ID_SECTIONS:
        for item in data.get(sec, []) or []:
            if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                out.add(item["id"])
    for b in (data.get("scoring") or {}).get("blocks", []) or []:
        if isinstance(b, Mapping) and isinstance(b.get("id"), str):
            out.add(b["id"])
    return out


def _rename(parts: Parts, ren: Mapping[str, str]) -> Parts:
    out = copy.deepcopy(parts)
    for sec in SECTIONS:
        for item in out.get(sec, []):
            if item.get("id") in ren:
                item["id"] = ren[item["id"]]
            for k in list(item):
                if k in REF_KEYS and item[k] in ren:
                    item[k] = ren[item[k]]
                elif k in REF_LISTS and isinstance(item[k], list):
                    item[k] = [ren.get(x, x) if isinstance(x, str) else x for x in item[k]]
    if out.get("root") in ren:
        out["root"] = ren[out["root"]]
    for chord in out.get("keys", []):
        chord["outputs"] = {ren.get(a, a): v for a, v in chord["outputs"].items()}
    return out


def _rewrite_random(value: Any, old: str, new: str) -> Any:
    pattern = re.compile(r"\br\." + re.escape(old) + r"\b")
    if isinstance(value, str):
        return pattern.sub("r." + new, value) if value.startswith("=") else value
    if isinstance(value, list):
        return [_rewrite_random(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: _rewrite_random(v, old, new) for k, v in value.items()}
    return value


def unique_suffix(used: set[str], ids: Sequence[str]) -> str:
    """묶음 전체에 붙일 같은 번호 꼬리("", "_2", "_3", …): 꼬리를 붙인 id가 하나도 겹치지 않는 가장 작은 것."""
    if not any(i in used for i in ids):
        return ""
    k = 2
    while any(f"{i}_{k}" in used for i in ids):
        k += 1
    return f"_{k}"


def agent_bodies(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [b for b in data.get("bodies", []) or [] if isinstance(b, Mapping) and b.get("owner", "agent") == "agent"]


def _brain_channels(data: Mapping[str, Any]) -> tuple[list[str], list[str]] | None:
    from evolab.spec import symbols as sy

    try:
        env = m.EnvFile.model_validate({k: v for k, v in data.items() if k in m.EnvFile.model_fields})
    except ValidationError:
        return None
    return [c.name for c in sy.sensor_channels(env)], [c.name for c in sy.actuator_channels(env)]


def insert(data: dict[str, Any], parts: Parts) -> list[Ref]:
    """부품 묶음을 환경 사전에 넣는다(사전을 고침). 넣은 부품의 (구역, id) 목록을 돌려준다.

    - id가 장면과 겹치면 묶음 전체에 같은 번호 꼬리("_2")를 붙인다(같은 틀을 두 번 넣으면 둘째 묶음이 모두 _2).
    - 랜덤 값은 같은 id·같은 정의가 있으면 그대로 쓰고, 정의가 다르면 번호를 붙여 수식도 바꾼다.
    - 몸 틀: 장면에 에이전트 물체가 없으면 시작 자세(agent.spawn)를 그 틀의 것으로 두고, 있으면 시작 자세는 그대로 두고 물체를
      그 자리(spawn의 x·y − 지금 시작 자리)로 옮긴다.
    - 뇌 입력·출력을 손으로 고른 파일이면 새 채널을 끝에 더한다(전부(all)이면 따로 할 일 없음).
    """
    used = all_ids(data)
    mine = [item["id"] for sec in SECTIONS for item in parts.get(sec, []) if "id" in item]
    suffix = unique_suffix(used, mine)
    if suffix:
        parts = _rename(parts, {i: i + suffix for i in mine})
    else:
        parts = copy.deepcopy(parts)
    before = _brain_channels(data)
    # 랜덤 값
    have = {r.get("id"): r for r in data.get("randomize", []) or [] if isinstance(r, Mapping)}
    new_randoms: list[dict[str, Any]] = []
    for r in parts.get("randomize", []):
        rid = r.get("id")
        if rid in have:
            if dict(have[rid]) == r:
                continue
            k = 2
            while f"{rid}_{k}" in have:
                k += 1
            new = f"{rid}_{k}"
            r = dict(r, id=new)
            for sec in SECTIONS:
                parts[sec] = _rewrite_random(parts.get(sec, []), rid, new)
            if parts.get("spawn"):
                parts["spawn"] = _rewrite_random(parts["spawn"], rid, new)
        new_randoms.append(r)
        have[r["id"]] = r
    if new_randoms:
        data.setdefault("randomize", []).extend(new_randoms)
    spawn = parts.get("spawn")
    if spawn is not None:
        if not agent_bodies(data):
            data.setdefault("agent", {})["spawn"] = dict(spawn)
        else:
            cur = (data.get("agent") or {}).get("spawn") or {}
            dx = _float_or_zero(spawn.get("x", 0.0)) - _float_or_zero(cur.get("x", 0.0))
            dy = _float_or_zero(spawn.get("y", 0.0)) - _float_or_zero(cur.get("y", 0.0))
            for b in parts.get("bodies", []):
                if b.get("owner", "agent") == "agent" and not b.get("place_by"):
                    b["pos"] = _shift(b.get("pos", [0.0, 0.0]), (dx, dy))
    refs: list[Ref] = []
    for sec in SECTIONS:
        items = parts.get(sec, [])
        if items:
            data.setdefault(sec, []).extend(items)
            refs.extend((sec, it["id"]) for it in items if "id" in it)
    after = _brain_channels(data)
    brain = data.get("brain")
    if isinstance(brain, dict) and before is not None and after is not None:
        for key, old, new in (("inputs", before[0], after[0]), ("outputs", before[1], after[1])):
            if isinstance(brain.get(key), list):
                brain[key] = list(brain[key]) + [c for c in new if c not in old and c not in brain[key]]
    return refs


def _float_or_zero(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


# ---------------------------------------------------------------------------
# 미리 그리기(그림 목록: Qt 없음)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrawOp:
    """thumbnail의 그리기 하나. 좌표는 그림 픽셀(왼쪽 위 (0, 0), y는 아래로). role: "agent" | "world" | "region"."""

    kind: str
    """"poly" | "circle" | "line"."""
    points: tuple[tuple[float, float], ...]
    radius: float = 0.0
    role: str = "agent"


def outline(tpl: Template, values: Mapping[str, float] | None = None) -> list[tuple[str, str, list[Point], float]]:
    """기본값으로 펼친 부품의 세계 좌표 윤곽: (kind, role, 점들, 반지름). 관절로 놓인 물체(place_by)는 관절 앵커로 놓는다."""
    parts = expand(tpl, values)
    spawn = parts.get("spawn") or {}
    sx, sy0 = _float_or_zero(spawn.get("x", 0.0)), _float_or_zero(spawn.get("y", 0.0))
    poses: dict[str, tuple[float, float, float]] = {}
    joints = {j["id"]: j for j in parts.get("joints", [])}
    bodies = {b["id"]: b for b in parts.get("bodies", [])}

    def pose(bid: str, depth: int = 0) -> tuple[float, float, float]:
        if bid in poses:
            return poses[bid]
        b = bodies.get(bid, {})
        agent = b.get("owner", "agent") == "agent" and tpl.kind == "body"
        jid = b.get("place_by")
        if jid in joints and depth < 32:
            j = joints[jid]
            ax, ay, aa = pose(j.get("a", ""), depth + 1)
            pa = [_float_or_zero(v) for v in j.get("anchor_a", [0.0, 0.0])]
            pb = [_float_or_zero(v) for v in j.get("anchor_b", [0.0, 0.0])]
            ang = aa + _float_or_zero(j.get("rest_angle", 0.0))
            wx = ax + pa[0] * math.cos(aa) - pa[1] * math.sin(aa)
            wy = ay + pa[0] * math.sin(aa) + pa[1] * math.cos(aa)
            x = wx - (pb[0] * math.cos(ang) - pb[1] * math.sin(ang))
            y = wy - (pb[0] * math.sin(ang) + pb[1] * math.cos(ang))
            poses[bid] = (x, y, ang)
        else:
            p = [_float_or_zero(v) for v in b.get("pos", [0.0, 0.0])]
            poses[bid] = (p[0] + (sx if agent else 0.0), p[1] + (sy0 if agent else 0.0), _float_or_zero(b.get("angle", 0.0)))
        return poses[bid]

    out: list[tuple[str, str, list[Point], float]] = []

    def place(shape: Mapping[str, Any], x: float, y: float, a: float, role: str) -> None:
        def tf(px: float, py: float) -> Point:
            return (x + px * math.cos(a) - py * math.sin(a), y + px * math.sin(a) + py * math.cos(a))

        kind = shape.get("kind")
        off = [_float_or_zero(v) for v in shape.get("offset", [0.0, 0.0])]
        if kind == "circle":
            out.append(("circle", role, [tf(off[0], off[1])], _float_or_zero(shape.get("r", 0.1))))
        elif kind == "box":
            w, h = _float_or_zero(shape.get("w", 0.1)) / 2, _float_or_zero(shape.get("h", 0.1)) / 2
            sa = _float_or_zero(shape.get("angle", 0.0))
            pts = []
            for cx, cy in ((-w, -h), (w, -h), (w, h), (-w, h)):
                rx, ry = cx * math.cos(sa) - cy * math.sin(sa), cx * math.sin(sa) + cy * math.cos(sa)
                pts.append(tf(off[0] + rx, off[1] + ry))
            out.append(("poly", role, pts, 0.0))
        elif kind in ("poly", "chain"):
            pts = [tf(_float_or_zero(p[0]), _float_or_zero(p[1])) for p in shape.get("points", []) if len(p) == 2]
            out.append(("poly" if kind == "poly" else "line", role, pts, 0.0))

    for b in parts.get("bodies", []):
        x, y, a = pose(b["id"])
        role = "agent" if b.get("owner", "agent") == "agent" and tpl.kind == "body" else "world"
        for shape in b.get("shapes", []):
            if isinstance(shape, Mapping):
                place(shape, x, y, a, role)
    for r in parts.get("regions", []):
        p = [_float_or_zero(v) for v in r.get("pos", [0.0, 0.0])]
        if isinstance(r.get("shape"), Mapping):
            place(r["shape"], p[0], p[1], 0.0, "region")
    return out


def thumbnail(tpl: Template, size: tuple[int, int] = (96, 72), values: Mapping[str, float] | None = None) -> list[DrawOp]:
    """보관함 카드 그림: 기본값으로 펼친 부품을 그림 크기에 맞춘 그리기 목록(Qt 없음, cards.py가 QPixmap으로 그린다)."""
    shapes = outline(tpl, values)
    xs: list[float] = []
    ys: list[float] = []
    for kind, _, pts, r in shapes:
        for x, y in pts:
            xs += [x - r, x + r]
            ys += [y - r, y + r]
    if not xs:
        return []
    w, h = size
    pad = 4.0
    scale = min((w - 2 * pad) / max(max(xs) - min(xs), 1e-6), (h - 2 * pad) / max(max(ys) - min(ys), 1e-6))
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2

    def px(p: Point) -> tuple[float, float]:
        return (round(w / 2 + (p[0] - cx) * scale, 3), round(h / 2 - (p[1] - cy) * scale, 3))

    return [DrawOp(kind, tuple(px(p) for p in pts), round(r * scale, 3), role) for kind, role, pts, r in shapes]


__all__ = [
    "KINDS", "SECTIONS", "TEMPLATE_FORMAT", "TEMPLATE_VERSION", "WORLDS", "DrawOp", "KeyChord", "Param", "Parts", "Point",
    "Ref", "Slot", "Template", "TemplateError", "agent_bodies", "all_ids", "corner_env", "evaluate", "expand", "insert",
    "load_template", "outline", "parse_template", "resolve_slots", "scene_tags", "template_hash", "thumbnail",
    "unique_suffix",
]

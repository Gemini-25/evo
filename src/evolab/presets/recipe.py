"""조리법: 안내 창의 고른 것 → 환경 사전 (6.2.0, 설계 DU 2.3·4.1 + 최종 설계 3.1·D28·D29·D31·D43, Qt 없음).

조리법(recipe 판 1)은 「무엇을 만들까요」 카드(start), 세계 종류, 경기장 틀, 역할마다 몸 틀, 물건 틀, 규칙 목록, 선수 자리와
(D28) 뇌·평가·세계 값·랜덤 값·그대로 넣는 조각(extra)·권장 새 실행 설정(train)이다. build는 순수 함수다(Qt·난수·시계 없음):
같은 조리법과 같은 틀 파일이면 같은 사전이 나온다.

build의 순서: 빈 환경(세계 종류에 맞는 중력) → 경기장 틀(세계 경계·기본 보기·자리 이름표) → 물건 틀(조리법 순서) → 역할의 몸 틀
(자리) → 「보는 것」 센서(규칙이 요구하는 것만) → 규칙(제한 시간은 끝) → 그대로 넣는 조각 → 세계 값·랜덤 값·뇌·평가 →
editor.view·editor.train·editor.rules·editor.recipe(+ built: 만든 직후의 환경 해시, templates: 틀 id → sha256). 결과는
정규 사전(spec.io.canonical_dict)이라 내용을 담는 가장 낮은 형식으로 적힌다(형식 1 부품만 쓰면 형식 1: 얼린 v6.1도 연다).

adjust는 맞지 않게 된 선택(세계와 몸, 규칙에 필요한 물건, 몸에 없는 부품, 범위 밖 슬라이더)을 고치고 바꾼 것을 한국어 줄로
알린다. 멱등이고 build(adjust(r))은 예외가 없다(T15).
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from evolab import paths
from evolab.app import strings as S
from evolab.presets import rules as R
from evolab.presets.josa import fill
from evolab.presets.library import Library, default_library
from evolab.presets.template import (
    Template, all_ids, evaluate, expand, insert, resolve_slots, scene_tags, template_hash,
)
from evolab.spec import model as m
from evolab.spec.compiled import EnvFileError
from evolab.spec.editor_meta import RECIPE_KEY, RULES_KEY, TRAIN_KEY, VIEW_KEY
from evolab.spec.io import canonical_dict, env_hash, parse_env

RECIPE_VERSION = 1
WORLDS = ("side", "top")
DEFAULT_BODY = {"side": "biped", "top": "disc"}
"""세계 종류마다 처음 몸 틀(안내 창 ②의 빈 세계 처음 값)."""
DEFAULT_ARENA = {"side": "track_side", "top": "yard_top"}
DEFAULT_SECONDS = 15.0
"""제한 시간 고정 줄이 없을 때 넣는 값."""
STARTER_ORDER = ("running", "drill", "blank_side", "blank_top")
"""첫 카드 순서(그 뒤는 이름순). 6.3.0에 권투·축구(관문을 넘은 것만)."""
STARTER_TEXT: dict[str, tuple[str, str]] = {
    "running": ("ED_BUILTIN_RUNNING", "SPORT_BLURBS.running"),
    "drill": ("PRESET_START_DRILL", "PRESET_START_DRILL_SUB"),
    "blank_side": ("PRESET_START_BLANK_SIDE", "PRESET_START_BLANK_SIDE_SUB"),
    "blank_top": ("PRESET_START_BLANK_TOP", "PRESET_START_BLANK_TOP_SUB"),
}
"""첫 카드의 제목·한 줄 글: strings.py의 이름(「이름.키」는 사전의 키). text_of가 글로 바꾼다."""
STATUSES = ("ok", "experimental", "hidden")
RECIPE_KEYS = frozenset({"recipe", "start", "name", "description", "world", "arena", "roles", "props", "rules", "players",
                         "brain", "evaluation", "world_values", "randoms", "match", "extra", "train", "status", "built",
                         "templates"})
EXTRA_KEYS = ("events", "terminations", "blocks", "sensors")
TARGET_SCALE = 20.0
"""「보는 것」 센서(목표 센서)의 눈금(m)."""


class RecipeError(ValueError):
    """조리법을 읽거나 만들 수 없음. 글은 한국어(strings.RECIPE_ERR_*)."""


def _bad(reason: str) -> RecipeError:
    return RecipeError(S.RECIPE_ERR_FORMAT.format(reason=reason))


@dataclass
class Use:
    """틀 하나를 쓰는 곳. at: 놓을 자리(없으면 경기장의 자리 이름 spawn, 그것도 없으면 (0, 0))."""

    template: str
    values: dict[str, float] = field(default_factory=dict)
    at: list[float] | None = None
    spawn: str | None = None
    mirror: bool = False

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"template": self.template, "values": dict(self.values)}
        if self.at is not None:
            out["at"] = list(self.at)
        if self.spawn is not None:
            out["spawn"] = self.spawn
        if self.mirror:
            out["mirror"] = True
        return out

    @classmethod
    def from_json(cls, data: Any, where: str) -> Use:
        if not isinstance(data, Mapping) or not isinstance(data.get("template"), str):
            raise _bad(f"{where}: {S.SPEC_ERR_OBJECT}")
        extra = set(data) - {"template", "values", "at", "spawn", "mirror", "evolve"}
        if extra:
            raise _bad(f"{where}.{sorted(extra)[0]}: {S.SPEC_ERR_EXTRA}")
        values = data.get("values", {}) or {}
        if not isinstance(values, Mapping) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) for v in values.values()):
            raise _bad(f"{where}.values: {S.SPEC_ERR_NUMBER}")
        at = data.get("at")
        if at is not None and not (isinstance(at, list) and len(at) == 2 and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in at)):
            raise _bad(f"{where}.at: {S.SPEC_ERR_LIST}")
        if data.get("evolve"):  # 「이 값들도 진화」는 6.2.0에 없다(D5)
            raise _bad(f"{where}.evolve: {S.SPEC_ERR_EXTRA}")
        return cls(data["template"], {str(k): float(v) for k, v in values.items()},
                   [float(at[0]), float(at[1])] if at is not None else None,
                   data.get("spawn") if isinstance(data.get("spawn"), str) else None, bool(data.get("mirror", False)))


@dataclass
class Role:
    id: str
    label: str
    body: Use

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "body": self.body.to_json()}


@dataclass
class Seat:
    role: str
    count: int = 1
    spawn: str = "start"

    def to_json(self) -> dict[str, Any]:
        return {"role": self.role, "count": self.count, "spawn": self.spawn}


@dataclass
class Players:
    """선수와 자리. 6.2.0은 혼자(solo, 자리 하나·선수 하나)만 만든다(1대1·2대2는 6.3.0)."""

    mode: str = "solo"
    seats: list[Seat] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"mode": self.mode, "seats": [s.to_json() for s in self.seats]}


@dataclass(frozen=True)
class Starter:
    """첫 카드. title_id·blurb_id는 strings.py의 이름(text_of로 글). status: "ok" | "experimental" | "hidden"."""

    id: str
    title_id: str
    blurb_id: str
    status: str = "ok"

    @property
    def title(self) -> str:
        return text_of(self.title_id)

    @property
    def blurb(self) -> str:
        return text_of(self.blurb_id)


def text_of(name: str) -> str:
    """strings.py의 이름(「사전.키」 꼴 포함)을 글로. 없으면 빈 글."""
    head, _, key = name.partition(".")
    value = getattr(S, head, None)
    if key:
        return str(value.get(key, "")) if isinstance(value, Mapping) else ""
    return value if isinstance(value, str) else ""


@dataclass
class Recipe:
    start: str
    name: str
    world: str
    arena: Use
    roles: list[Role]
    props: list[Use]
    rules: list[dict[str, Any]]
    players: Players
    brain: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    world_values: dict[str, Any] | None = None
    randoms: list[dict[str, Any]] | None = None
    match: dict[str, Any] | None = None
    extra: dict[str, list[dict[str, Any]]] | None = None
    train: dict[str, Any] | None = None
    description: str = ""
    """환경의 설명(환경 내용이 되는 글: D55). 비우면 적지 않는다."""

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"recipe": RECIPE_VERSION, "start": self.start, "name": self.name, "world": self.world,
                               "arena": self.arena.to_json(), "roles": [r.to_json() for r in self.roles],
                               "props": [p.to_json() for p in self.props], "rules": copy.deepcopy(self.rules),
                               "players": self.players.to_json()}
        if self.description:
            out["description"] = self.description
        for key in ("brain", "evaluation", "world_values", "randoms", "match", "extra", "train"):
            value = getattr(self, key)
            if value is not None:
                out[key] = copy.deepcopy(value)
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Recipe:
        """조리법 사전 → Recipe. 잘못됐으면 RecipeError(한국어). build가 적는 built·templates와 내장 파일의 status는 읽고 버린다."""
        if not isinstance(data, Mapping):
            raise _bad(S.SPEC_NOT_OBJECT)
        version = data.get("recipe")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise _bad(f"recipe: {S.SPEC_ERR_INT}")
        if version > RECIPE_VERSION:
            raise RecipeError(S.RECIPE_ERR_NEWER.format(version=version))
        extra = sorted(set(data) - RECIPE_KEYS)
        if extra:
            raise _bad(f"{extra[0]}: {S.SPEC_ERR_EXTRA}")
        for key in ("start", "world", "arena", "roles"):
            if key not in data:
                raise _bad(f"{key}: {S.SPEC_ERR_MISSING}")
        if data["world"] not in WORLDS:
            raise _bad(f"world: {S.SPEC_ERR_LITERAL.format(expected=', '.join(WORLDS))}")
        if not isinstance(data.get("start"), str) or not isinstance(data.get("name", ""), str):
            raise _bad(f"start: {S.SPEC_ERR_STRING}")
        roles_raw = data.get("roles")
        if not isinstance(roles_raw, list) or not roles_raw:
            raise _bad(f"roles: {S.SPEC_ERR_LIST_SHORT.format(min=1)}")
        roles: list[Role] = []
        for i, r in enumerate(roles_raw):
            if not isinstance(r, Mapping) or not isinstance(r.get("id"), str):
                raise _bad(f"roles[{i}]: {S.SPEC_ERR_OBJECT}")
            roles.append(Role(r["id"], str(r.get("label", "")), Use.from_json(r.get("body"), f"roles[{i}].body")))
        props_raw = data.get("props", []) or []
        if not isinstance(props_raw, list):
            raise _bad(f"props: {S.SPEC_ERR_LIST}")
        props = [Use.from_json(p, f"props[{i}]") for i, p in enumerate(props_raw)]
        rules = data.get("rules", []) or []
        if not isinstance(rules, list) or not all(isinstance(r, Mapping) and isinstance(r.get("rule"), str)
                                                  for r in rules):
            raise _bad(f"rules: {S.SPEC_ERR_LIST}")
        pl = data.get("players") or {"mode": "solo", "seats": [{"role": roles[0].id, "count": 1, "spawn": "start"}]}
        if not isinstance(pl, Mapping) or not isinstance(pl.get("seats", []), list):
            raise _bad(f"players: {S.SPEC_ERR_OBJECT}")
        seats = []
        for i, s in enumerate(pl.get("seats", [])):
            if not isinstance(s, Mapping) or not isinstance(s.get("role"), str):
                raise _bad(f"players.seats[{i}]: {S.SPEC_ERR_OBJECT}")
            count = s.get("count", 1)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise _bad(f"players.seats[{i}].count: {S.SPEC_ERR_INT}")
            seats.append(Seat(s["role"], count, str(s.get("spawn", "start"))))
        players = Players(str(pl.get("mode", "solo")), seats)
        for key in ("brain", "evaluation", "world_values", "match", "train"):
            if data.get(key) is not None and not isinstance(data[key], Mapping):
                raise _bad(f"{key}: {S.SPEC_ERR_OBJECT}")
        if data.get("randoms") is not None and not isinstance(data["randoms"], list):
            raise _bad(f"randoms: {S.SPEC_ERR_LIST}")
        ex = data.get("extra")
        if ex is not None and (not isinstance(ex, Mapping) or set(ex) - set(EXTRA_KEYS) or not all(
                isinstance(v, list) for v in ex.values())):
            raise _bad(f"extra: {S.SPEC_ERR_OBJECT}")
        desc = data.get("description", "")
        return cls(
            start=data["start"], name=str(data.get("name", "")), world=data["world"],
            arena=Use.from_json(data["arena"], "arena"), roles=roles, props=props,
            rules=[copy.deepcopy(dict(r)) for r in rules], players=players,
            brain=copy.deepcopy(dict(data["brain"])) if data.get("brain") is not None else None,
            evaluation=copy.deepcopy(dict(data["evaluation"])) if data.get("evaluation") is not None else None,
            world_values=copy.deepcopy(dict(data["world_values"])) if data.get("world_values") is not None else None,
            randoms=copy.deepcopy(list(data["randoms"])) if data.get("randoms") is not None else None,
            match=copy.deepcopy(dict(data["match"])) if data.get("match") is not None else None,
            extra={k: copy.deepcopy(list(v)) for k, v in ex.items()} if ex is not None else None,
            train=copy.deepcopy(dict(data["train"])) if data.get("train") is not None else None,
            description=desc if isinstance(desc, str) else "",
        )


# ---------------------------------------------------------------------------
# 내장 조리법과 첫 카드
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise _bad(S.SPEC_READ_FAILED.format(path=path, reason=exc.strerror or exc)) from None
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _bad(S.SPEC_BAD_JSON.format(line=exc.lineno, col=exc.colno) if isinstance(exc, json.JSONDecodeError)
                   else S.SPEC_NOT_UTF8) from None
    if not isinstance(data, dict):
        raise _bad(S.SPEC_NOT_OBJECT)
    return data


def builtin_names() -> list[str]:
    """내장 조리법 이름(assets/recipes/<이름>.json): STARTER_ORDER 순서, 그 뒤 이름순."""
    folder = paths.recipe_dir()
    names = sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []
    return [n for n in STARTER_ORDER if n in names] + [n for n in names if n not in STARTER_ORDER]


def builtin_status(name: str) -> str:
    status = _read_json(paths.recipe_dir() / f"{name}.json").get("status", "ok")
    return status if status in STATUSES else "ok"


def load_builtin(name: str) -> Recipe:
    """assets/recipes/<name>.json. 없거나 잘못됐으면 RecipeError."""
    return Recipe.from_json(_read_json(paths.recipe_dir() / f"{name}.json"))


def _templates_of(recipe: Recipe) -> list[str]:
    return [recipe.arena.template, *(p.template for p in recipe.props), *(r.body.template for r in recipe.roles)]


def starters(lib: Library | None = None) -> list[Starter]:
    """첫 카드: 내장 조리법(STARTER_ORDER 순서). 숨김(hidden)·대전(match)·틀이 없는 조리법은 뺀다(D29·D36·D47)."""
    lib = lib or default_library()
    out: list[Starter] = []
    for name in builtin_names():
        try:
            data = _read_json(paths.recipe_dir() / f"{name}.json")
            recipe = Recipe.from_json(data)
        except RecipeError:
            continue
        status = data.get("status", "ok")
        if status not in STATUSES or status == "hidden" or recipe.match is not None:
            continue
        if not all(lib.has(t) for t in _templates_of(recipe)):
            continue
        title, blurb = STARTER_TEXT.get(name, (f"TPL_NAMES.{name}", f"TPL_BLURBS.{name}"))
        out.append(Starter(name, title, blurb, status))
    return out


def default_recipe(start: str, lib: Library | None = None) -> Recipe:
    """첫 카드의 조리법(내장 조리법 그대로의 사본). 이름은 비어 있을 수 있다(안내 창이 채운다)."""
    del lib
    return load_builtin(start)


# ---------------------------------------------------------------------------
# 만들기
# ---------------------------------------------------------------------------


def _get(lib: Library, template_id: str) -> Template:
    try:
        return lib.get(template_id)
    except KeyError:
        raise RecipeError(S.RECIPE_ERR_TEMPLATE.format(id=template_id)) from None


def _spawn_point(arena: Template, values: Mapping[str, float], name: str | None, index: int = 0) -> tuple[float, float]:
    if not name or name not in arena.spawns:
        return (0.0, 0.0)
    pts = arena.spawns[name]
    pt = evaluate(list(pts[min(index, len(pts) - 1)]), _clamped(arena, values))
    return (float(pt[0]), float(pt[1]))


def _clamped(tpl: Template, values: Mapping[str, float]) -> dict[str, float]:
    out = tpl.defaults()
    for k, v in values.items():
        if k in out:
            out[k] = tpl.param(k).clamp(v)
    return out


def _place(data: dict[str, Any], tpl: Template, use: Use, at: tuple[float, float]) -> dict[str, str]:
    """틀 하나를 넣고 틀 id → 환경 id 표를 돌려준다(겹쳐서 번호가 붙은 것까지)."""
    filled, _missing = resolve_slots(tpl, data)
    parts = expand(tpl, use.values, at=at, mirror=use.mirror, slots=filled)
    originals = [(sec, item["id"]) for sec in ("bodies", "joints", "actuators", "sensors", "regions", "generators")
                 for item in parts.get(sec, []) if "id" in item]
    refs = insert(data, parts)
    mapping = {orig: new for (_, orig), (_, new) in zip(originals, refs, strict=True)}
    for name, value in filled.items():
        mapping[name] = value
    for name, slot in tpl.slots.items():
        mapping.setdefault(name, "tag." + slot.tags[0])
    return mapping


@dataclass
class _Built:
    data: dict[str, Any]
    maps: dict[str, dict[str, str]]
    """역할 id·"arena"·"prop<i>" → (틀 id → 환경 id)."""
    roots: dict[str, str]


def _resolve(value: Any, who: str | None, built: _Built) -> str:
    """조리법 규칙의 참조 → 환경 id: tag.* 그대로, $빈자리·틀 id는 그 역할의 몸 → 경기장 → 물건 순서로 찾는다."""
    if not isinstance(value, str) or not value:
        raise _bad(S.SPEC_REF_BODY.format(ref=value))
    if value.startswith("tag."):
        return value
    order = ([who] if who else []) + ["arena"] + [k for k in built.maps if k.startswith("prop")]
    for key in order:
        mp = built.maps.get(key, {})
        if value in mp:
            return mp[value]
    raise _bad(S.SPEC_REF_BODY_OR_TAG.format(ref=value))


def _rule_args(rule: Mapping[str, Any], built: _Built, default_who: str) -> tuple[str, dict[str, Any]]:
    kind = rule.get("rule")
    if kind not in R.RULES:
        raise _bad(S.SPEC_ERR_LITERAL.format(expected=", ".join(R.RULES)))
    who = rule.get("who", default_who)
    if who not in built.roots:
        who = default_who
    args: dict[str, Any] = {}
    for slot in R.RULES[kind].slots:
        if slot.id not in rule:
            continue
        v = rule[slot.id]
        if slot.kind == "parts":
            args[slot.id] = [_resolve(x, who, built) for x in (v if isinstance(v, list) else [v])]
        elif slot.kind in ("part", "what", "where", "thing"):
            args[slot.id] = _resolve(v, who, built)
        else:
            args[slot.id] = v
    root = built.roots.get(who)
    for slot in R.RULES[kind].slots:  # 빈칸이 없으면 그 역할의 뿌리 물체
        if slot.id not in args and root is not None:
            if slot.kind == "part" or (slot.kind == "thing"):
                args[slot.id] = root
            elif slot.kind == "parts":
                args[slot.id] = [root]
    return kind, args


def _see(data: dict[str, Any], body: str, target: str) -> None:
    """목표 센서 「body에서 target이 어디 있나」를 하나 더한다(같은 것이 있으면 그대로)."""
    for s in data.get("sensors", []) or []:
        if s.get("kind") == "target" and s.get("body") == body and s.get("target") == target:
            return
    sid = R._fresh(data, f"see_{target}", set())
    label = S.TPL_SEE_LABEL.format(target=R.display_name(data, target))
    data.setdefault("sensors", []).append({"kind": "target", "id": sid, "label": label, "body": body, "target": target,
                                           "scale": TARGET_SCALE})


def _is_agent(data: Mapping[str, Any], ref: str) -> bool:
    for b in data.get("bodies", []) or []:
        if b.get("id") == ref:
            return b.get("owner", "agent") == "agent"
    return False


def build(recipe: Recipe, lib: Library | None = None) -> dict[str, Any]:
    """조리법 → 환경 사전(정규 사전, editor.recipe·editor.train·editor.rules·editor.view 포함). 순수 함수.

    틀이 없거나 참조를 풀 수 없거나 결과가 환경 파일 모델에 맞지 않으면 RecipeError(한국어).
    """
    lib = lib or default_library()
    if recipe.world not in WORLDS:
        raise _bad(f"world: {S.SPEC_ERR_LITERAL.format(expected=', '.join(WORLDS))}")
    if recipe.match is not None:  # 대전은 6.3.0
        raise _bad(f"match: {S.SPEC_ERR_EXTRA}")
    world: dict[str, Any] = {"gravity": [0.0, -9.81] if recipe.world == "side" else [0.0, 0.0]}
    if recipe.world == "top":
        world["view"] = "top"
    data: dict[str, Any] = {"version": m.CURRENT_VERSION, "name": recipe.name.strip() or S.PRESET_NAME_FALLBACK}
    if recipe.description:
        data["description"] = recipe.description
    data["world"] = world
    used_templates: dict[str, str] = {}
    built = _Built(data, {}, {})
    # 경기장
    arena = _get(lib, recipe.arena.template)
    used_templates[arena.id] = template_hash(arena)
    arena_values = _clamped(arena, recipe.arena.values)
    built.maps["arena"] = _place(data, arena, Use(arena.id, dict(recipe.arena.values)), (0.0, 0.0))
    if arena.bounds is not None:
        world["bounds"] = evaluate(list(arena.bounds), arena_values)
    view = evaluate(list(arena.view), arena_values) if arena.view is not None else None
    seats = recipe.players.seats or [Seat(recipe.roles[0].id)]
    # 물건: 자리 이름으로 놓되, 선수 자리나 앞서 놓은 물건과 같은 점이면 PROP_GAP씩 +x로 비킨다(시작부터 겹치지 않게).
    # 겹치지 않는 조리법(내장 drill·running 등)은 예전과 같은 결과다.
    taken: list[tuple[float, float]] = []
    for seat in seats:
        role = next((r for r in recipe.roles if r.id == seat.role), None)
        if role is not None:
            taken.append(tuple(role.body.at) if role.body.at is not None  # type: ignore[arg-type]
                         else _spawn_point(arena, arena_values, seat.spawn))
    for i, use in enumerate(recipe.props):
        tpl = _get(lib, use.template)
        used_templates[tpl.id] = template_hash(tpl)
        if use.at is not None:
            at = (float(use.at[0]), float(use.at[1]))
        else:
            at = _spawn_point(arena, arena_values, use.spawn or tpl.id)
            while any(abs(at[0] - x) < 1e-6 and abs(at[1] - y) < 1e-6 for x, y in taken):
                at = (at[0] + PROP_GAP, at[1])
        taken.append(at)
        built.maps[f"prop{i}"] = _place(data, tpl, use, (float(at[0]), float(at[1])))
    # 역할의 몸(6.2.0: 혼자. 자리 하나에 선수 하나)
    placed: set[str] = set()
    for seat in seats:
        role = next((r for r in recipe.roles if r.id == seat.role), None)
        if role is None or role.id in placed:
            continue
        placed.add(role.id)
        tpl = _get(lib, role.body.template)
        used_templates[tpl.id] = template_hash(tpl)
        at = tuple(role.body.at) if role.body.at is not None else _spawn_point(arena, arena_values, seat.spawn)
        mp = _place(data, tpl, role.body, (float(at[0]), float(at[1])))
        built.maps[role.id] = mp
        if tpl.root:
            built.roots[role.id] = mp[tpl.root]
    default_who = recipe.roles[0].id
    # 규칙(제한 시간은 끝)
    plain = [r for r in recipe.rules if r.get("rule") != "time_limit"]
    limits = [r for r in recipe.rules if r.get("rule") == "time_limit"]
    translated = [_rule_args(r, built, default_who) for r in plain]
    root = built.roots.get(default_who)
    for kind, args in translated:  # 「보는 것」 센서: 몸 밖의 것을 재는 규칙만
        if root is None:
            break
        if kind == "near":
            thing, where = args.get("thing"), args.get("where")
            if isinstance(thing, str) and not _is_agent(data, thing):
                _see(data, root, thing)
            if isinstance(where, str) and not where.startswith("tag.") and not _is_agent(data, where):
                _see(data, thing if isinstance(thing, str) and not _is_agent(data, thing) else root, where)
        elif kind == "reach":
            where = args.get("where")
            if isinstance(where, str) and not where.startswith("tag.") and not _is_agent(data, where):
                _see(data, root, where)
    for kind, args in translated:
        try:
            R.add_rule(data, kind, **args)
        except ValueError as exc:
            raise _bad(str(exc)) from None
    seconds = limits[-1].get("seconds", DEFAULT_SECONDS) if limits else DEFAULT_SECONDS
    try:
        R.add_rule(data, "time_limit", seconds=seconds)
    except ValueError as exc:
        raise _bad(str(exc)) from None
    # 그대로 넣는 조각
    for key, items in (recipe.extra or {}).items():
        for item in copy.deepcopy(items):
            if key == "events":
                data.setdefault("events", []).append(item)
            elif key == "terminations":
                R._insert_end(data, item)
            elif key == "blocks":
                data.setdefault("scoring", {}).setdefault("blocks", []).append(item)
            elif key == "sensors":
                data.setdefault("sensors", []).append(item)
    if recipe.randoms:
        have = {r.get("id") for r in data.get("randomize", [])}
        data.setdefault("randomize", []).extend(copy.deepcopy(r) for r in recipe.randoms if r.get("id") not in have)
    if recipe.world_values:
        world.update(copy.deepcopy(recipe.world_values))
    if recipe.brain is not None:
        data["brain"] = copy.deepcopy(recipe.brain)
    if recipe.evaluation is not None:
        data["evaluation"] = copy.deepcopy(recipe.evaluation)
    editor = data.setdefault("editor", {})
    rules_rows = editor.pop(RULES_KEY, None)
    out_editor: dict[str, Any] = {}
    if view is not None:
        out_editor[VIEW_KEY] = view
    if recipe.train:
        out_editor[TRAIN_KEY] = copy.deepcopy(recipe.train)
    if rules_rows:
        out_editor[RULES_KEY] = rules_rows
    data["editor"] = out_editor
    try:
        env = parse_env(data)
    except EnvFileError as exc:
        raise _bad(str(exc.issues[0]) if exc.issues else str(exc)) from None
    marker = recipe.to_json()
    marker["built"] = env_hash(env)
    marker["templates"] = dict(sorted(used_templates.items()))
    out = canonical_dict(env)
    out["editor"] = dict(out.get("editor", {}))
    out["editor"][RECIPE_KEY] = marker
    return out


def physics_view(data: Mapping[str, Any]) -> dict[str, Any]:
    """물리로 견줄 사전(D29): canonical_dict(parse_env(data), exclude={"editor"})에서 version·name·description과 모든 label·
    keys*를 뺀 것. 두 쪽을 모델로 읽어 기본값을 채운 뒤 견주므로 한쪽에 빠진 기본 칸은 차이가 아니다."""
    env = parse_env(copy.deepcopy(dict(data)))
    out = canonical_dict(env, exclude={"editor"})
    for key in ("version", "name", "description"):
        out.pop(key, None)
    return _strip_labels(out)


def _strip_labels(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_labels(v) for k, v in value.items() if k != "label" and not k.startswith("keys")}
    if isinstance(value, list):
        return [_strip_labels(v) for v in value]
    return value


def matches(data: Mapping[str, Any]) -> bool:
    """문서가 자기 조리법 그대로인가: editor.recipe.built == 지금 환경 해시(설치된 틀과 무관, D43)."""
    editor = data.get("editor")
    marker = editor.get(RECIPE_KEY) if isinstance(editor, Mapping) else None
    if not isinstance(marker, Mapping) or not isinstance(marker.get("built"), str):
        return False
    try:
        return env_hash(parse_env(copy.deepcopy(dict(data)))) == marker["built"]
    except EnvFileError:
        return False


# ---------------------------------------------------------------------------
# 맞추기
# ---------------------------------------------------------------------------


def _fits(tpl: Template, kind: str, world: str) -> bool:
    return tpl.kind == kind and tpl.world in (world, "any")


def _scene_for(recipe: Recipe, lib: Library) -> tuple[dict[str, Any], _Built]:
    """규칙을 빼고 만든 장면(참조를 풀고 태그를 보려고)."""
    trial = copy.deepcopy(recipe)
    trial.rules = []
    trial.extra = None
    data: dict[str, Any] = {"bodies": [], "regions": []}
    built = _Built(data, {}, {})
    arena = lib.get(trial.arena.template)
    built.maps["arena"] = _place(data, arena, trial.arena, (0.0, 0.0))
    for i, use in enumerate(trial.props):
        built.maps[f"prop{i}"] = _place(data, lib.get(use.template), use, (0.0, 0.0))
    role = trial.roles[0]
    tpl = lib.get(role.body.template)
    mp = _place(data, tpl, role.body, (0.0, 0.0))
    built.maps[role.id] = mp
    if tpl.root:
        built.roots[role.id] = mp[tpl.root]
    return data, built


def _clean_values(tpl: Template, values: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in values.items():
        if any(p.id == k for p in tpl.params) and isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[k] = tpl.param(k).clamp(float(v))
    return out


def adjust(recipe: Recipe, lib: Library | None = None) -> tuple[Recipe, list[str]]:
    """맞지 않는 선택을 고친 새 조리법과 바꾼 것의 한국어 줄(PRESET_ADJUST_*). 멱등이고 build(adjust(r))은 예외가 없다."""
    lib = lib or default_library()
    r = copy.deepcopy(recipe)
    notes: list[str] = []
    if r.world not in WORLDS:
        r.world = "side"
    r.match = None
    # 경기장
    if not lib.has(r.arena.template) or not _fits(lib.get(r.arena.template), "arena", r.world):
        r.arena = Use(DEFAULT_ARENA[r.world])
    arena = lib.get(r.arena.template)
    r.arena.values = _clean_values(arena, r.arena.values)
    r.arena.at = None
    r.arena.mirror = False
    # 몸(6.2.0: 혼자 하는 역할 하나)
    if not r.roles:
        r.roles = [Role("player", "", Use(DEFAULT_BODY[r.world]))]
    r.roles = r.roles[:1]
    role = r.roles[0]
    if not lib.has(role.body.template) or not _fits(lib.get(role.body.template), "body", r.world):
        role.body = Use(DEFAULT_BODY[r.world])
        notes.append(fill(S.PRESET_ADJUST_BODY, {"body": lib.get(role.body.template).name}))
    body = lib.get(role.body.template)
    role.body.values = _clean_values(body, role.body.values)
    r.players = Players("solo", [Seat(role.id, 1, _seat_spawn(r, arena))])
    # 물건
    kept: list[Use] = []
    for use in r.props:
        if lib.has(use.template) and _fits(lib.get(use.template), "prop", r.world):
            use.values = _clean_values(lib.get(use.template), use.values)
            if use.at is None and (use.spawn or use.template) not in arena.spawns:
                use.spawn = prop_spawn(arena, use.template)  # 경기장을 바꿔 없는 자리면(또는 자리 없이 (0, 0)이면) 맞는 자리로
            kept.append(use)
    r.props = kept
    # 규칙
    r.rules = _adjust_rules(r, lib, notes)
    if r.train is not None:
        from evolab.engine.trainer import PRESET_TRAIN_KEYS

        r.train = {k: v for k, v in r.train.items() if k in PRESET_TRAIN_KEYS} or None
    return r, notes


def _seat_spawn(r: Recipe, arena: Template) -> str:
    want = r.players.seats[0].spawn if r.players.seats else "start"
    if want in arena.spawns or not arena.spawns:
        return want
    return "start" if "start" in arena.spawns else next(iter(arena.spawns))


PROP_GAP = 1.5
"""build: 같은 자리에 놓이는 물건(또는 선수 자리와 같은 점)을 비키는 거리(m, +x 쪽)."""

PROP_SPAWNS: Final[dict[str, tuple[str, ...]]] = {"goal": ("goal", "finish", "ahead", "center")}
"""물건 틀마다 찾을 경기장 자리 이름(앞의 것부터). 없는 틀은 (틀 id, "ahead", "center")."""


def prop_spawn(arena: Template, tid: str) -> str | None:
    """물건 tid를 놓을 경기장 자리 이름(선수의 "start"는 쓰지 않는다). 맞는 자리가 없으면 None."""
    for name in PROP_SPAWNS.get(tid, (tid, "ahead", "center")):
        if name in arena.spawns and name != "start":
            return name
    return None


def _prop_with(r: Recipe, lib: Library, region: bool) -> Use | None:
    """규칙에 필요해 넣을 물건: 영역이 필요하면 골대(경기장의 골 자리, 없으면 결승점·앞쪽: 선수 자리에 놓지 않는다)."""
    if region and lib.has("goal") and _fits(lib.get("goal"), "prop", r.world):
        return Use("goal", {}, None, prop_spawn(lib.get(r.arena.template), "goal"))
    return None


def _why(text: str) -> str:
    """괄호 안에 넣는 까닭: 끝의 마침표를 뗀다."""
    return text.rstrip(". ")


def _adjust_rules(r: Recipe, lib: Library, notes: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_kinds: set[str] = set()
    seconds = DEFAULT_SECONDS
    role = r.roles[0]
    for raw in r.rules:
        rule = dict(raw)
        kind = rule.get("rule")
        if kind not in R.RULES:
            continue
        title = S.RULE_TITLES.get(kind, kind)
        if kind == "time_limit":
            s = rule.get("seconds")
            if isinstance(s, (int, float)) and not isinstance(s, bool) and math.isfinite(float(s)) and s > 0:
                seconds = min(float(s), 3600.0)
            continue
        rk = R.RULES[kind]
        if rk.world != "any" and rk.world != r.world:
            notes.append(fill(S.PRESET_ADJUST_RULE_OFF, {"rule": title, "why": _why(
                S.TPL_WRONG_WORLD.format(world=S.TPL_WORLD_NAMES[rk.world]))}))
            continue
        if kind == "out_ends" and "out_ends" in seen_kinds:
            continue
        rule["who"] = role.id
        data, built = _scene_for(r, lib)
        if ("region" in rk.needs and not data.get("regions")) or ("target" in rk.needs and not R.target_pool(data)):
            prop = _prop_with(r, lib, True)
            if prop is None:
                notes.append(fill(S.PRESET_ADJUST_RULE_OFF, {"rule": title, "why": _why(fill(
                    S.RULES_NEEDS, {"what": S.RULE_NEEDS_WHAT["region"]}))}))
                continue
            r.props.append(prop)
            notes.append(fill(S.PRESET_ADJUST_PROP_ON, {"rule": title, "prop": lib.get(prop.template).name}))
            data, built = _scene_for(r, lib)
        fixed = _fix_rule(rule, kind, data, built, role.id)
        if fixed is None:
            notes.append(fill(S.PRESET_ADJUST_RULE_OFF, {"rule": title, "why": _why(fill(
                S.RULES_NEEDS, {"what": S.RULE_NEEDS_WHAT["target"]}))}))
            continue
        out.append(fixed)
        seen_kinds.add(kind)
    out.append({"rule": "time_limit", "seconds": seconds})
    return out


def _resolvable(value: Any, who: str, built: _Built, data: Mapping[str, Any]) -> bool:
    try:
        ref = _resolve(value, who, built)
    except RecipeError:
        return False
    if ref.startswith("tag."):
        return ref[4:] in scene_tags(data)
    return ref in all_ids(data)


def _allowed_target(value: Any, kind: str, slot: R.RuleSlot, who: str, built: _Built, data: Mapping[str, Any]) -> bool:
    """where·thing 빈칸 값이 화면이 고를 수 있는 것(R.candidates)인가: 있는 id라도 고정 물체나, 「들어가면 성공」의 영역 자리에
    놓인 물체는 컴파일이 거절하므로 받지 않는다(태그도 받지 않음)."""
    try:
        ref = _resolve(value, who, built)
    except RecipeError:
        return False
    if ref.startswith("tag."):
        return False
    return ref in {v for v, _ in R.candidates(data, slot, kind)}


def _fix_rule(rule: dict[str, Any], kind: str, data: Mapping[str, Any], built: _Built, who: str) -> dict[str, Any] | None:
    """규칙 한 줄의 빈칸을 장면에 맞춘다(없는 부품 → 뿌리, 없는 태그 → 장면의 첫 태그, 범위 밖 숫자 → 처음 값)."""
    root_tpl_id = next((k for k, v in built.maps.get(who, {}).items() if v == built.roots.get(who)), None)
    body_parts = [k for k, v in built.maps.get(who, {}).items() if not k.startswith("$")
                  and any(b.get("id") == v for b in data.get("bodies", []))]
    out: dict[str, Any] = {"rule": kind, "who": who}
    for slot in R.RULES[kind].slots:
        v = rule.get(slot.id)
        if slot.kind in ("part", "parts"):
            vals = v if isinstance(v, list) else ([v] if isinstance(v, str) else [])
            vals = [x for x in vals if x in body_parts]
            if not vals and root_tpl_id is not None:
                vals = [root_tpl_id]
            if not vals:
                return None
            out[slot.id] = vals if slot.kind == "parts" else vals[0]
        elif slot.kind == "what":
            if not _resolvable(v, who, built, data):
                tags = sorted(scene_tags({"bodies": [b for b in data.get("bodies", [])
                                                     if b.get("owner", "agent") == "world"]}))
                pick = next((t for t in ("ground", "terrain", "wall") if t in tags), tags[0] if tags else None)
                if pick is None:
                    return None
                v = "tag." + pick
            out[slot.id] = v
        elif slot.kind in ("where", "thing"):
            if not _allowed_target(v, kind, slot, who, built, data):
                pick = _first_target(kind, slot.id, data, built, who, out)
                if pick is None:
                    return None
                v = pick
            out[slot.id] = v
        elif slot.kind == "choice":
            out[slot.id] = v if v in slot.choices else slot.default
        else:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
            if ok and slot.low is not None and float(v) < slot.low:
                ok = False
            out[slot.id] = round(float(v), 6) + 0.0 if ok else slot.default
    if kind == "near" and out.get("thing") == out.get("where"):
        pick = _first_target(kind, "where", data, built, who, out)  # 같은 것끼리의 거리는 늘 0: 다른 대상으로(없으면 끈다)
        if pick is None:
            return None
        out["where"] = pick
    return out


def _first_target(kind: str, slot: str, data: Mapping[str, Any], built: _Built, who: str,
                  so_far: Mapping[str, Any]) -> str | None:
    """where·thing 빈칸의 처음 값(틀 id 꼴): 영역 → 세계의 움직이는 물체 → 뿌리."""
    inv = {}
    for key, mp in built.maps.items():
        for tid, eid in mp.items():
            if not tid.startswith("$"):
                inv.setdefault(eid, tid)
    regions = [inv.get(x["id"]) for x in data.get("regions", []) if x.get("id") in inv]
    movers = [inv.get(b["id"]) for b in data.get("bodies", [])
              if b.get("owner", "agent") == "world" and b.get("type", "dynamic") == "dynamic" and b.get("id") in inv]
    root = next((k for k, v in built.maps.get(who, {}).items() if v == built.roots.get(who)), None)
    if slot == "where":
        pool = regions + ([] if kind == "reach" else movers)
    else:  # thing
        pool = movers + ([root] if root else [])
    pool = [p for p in pool if p and p != so_far.get("thing")]
    return pool[0] if pool else None


__all__ = [
    "DEFAULT_ARENA", "DEFAULT_BODY", "RECIPE_VERSION", "STARTER_ORDER", "Players", "Recipe", "RecipeError", "Role", "Seat",
    "Starter", "Use", "adjust", "build", "builtin_names", "builtin_status", "default_recipe", "load_builtin", "matches",
    "physics_view", "prop_spawn", "starters", "text_of",
]

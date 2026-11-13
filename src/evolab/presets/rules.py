"""규칙: 쉬운 문장 ↔ 사건·종료 조건·점수 블록 조각 (6.2.0, 설계 DU 4.3.5 + 최종 설계 D11·D39, Qt 없음).

규칙 하나는 환경 사전에 정해진 모양의 조각을 만든다(표는 아래 _make). 사건·블록 id는 규칙 표의 평범한 이름이다(`fall`, `touch`,
`reach`; 블록은 `s_<이름>`: `s_dist`, `s_fall`, `s_energy` …). 겹치면 `_2`, `_3`.

- add_rule은 조각과 함께 `editor.rules`(해시 밖)에 `{"kind", "event", "blocks"}` 한 줄을 적는다. 이 줄이 있고 닫힌 모양 검사를
  넘는 규칙만 「자기 규칙」이라 고칠 수 있다(RuleUse.editable, D39). 닫힌 검사: 그 사건·블록을 가리키는 것(종료 조건, 사건 블록,
  다른 사건의 조건, 수식)이 모두 그 규칙의 것이고, 모든 칸이 종류별 흰 목록 안이다.
- 적혀 있지 않지만 모양이 맞는 것(내장 예제)은 읽기 전용 문장으로 알아본다. 그 밖은 「고급 탭에 있는 규칙 n개」로 센다.
- 제한 시간(time_limit)은 늘 있는 고정 줄이다(world.max_time): 뺄 수 없고 숫자만 고치며 editor.rules에 적지 않는다.
- set_rule은 빈칸 뒤의 칸만 고친다: 값이 같으면 사전은 바이트까지 그대로다.
- 부호는 규칙이 정한다(「가까울수록 점수」「감점」「가산점」이 문장에 들어 있다).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from evolab.app import strings as S
from evolab.presets.josa import fill
from evolab.spec.editor_meta import RULES_KEY

Ref = tuple[str, str]
"""(구역, id). 구역은 "events" | "termination" | "blocks". 종료 줄의 id는 그 줄의 사건 id."""

DIRECTIONS: dict[str, float] = {"right": 0.0, "left": math.pi, "up": math.pi / 2, "down": -math.pi / 2}
"""distance의 방향 이름 → 진행 블록의 direction(라디안)."""
OUTCOMES = ("failure", "success", "neutral")
WHEN = ("min", "final")


@dataclass(frozen=True)
class RuleSlot:
    """규칙 문장의 빈칸. kind: "part"(에이전트 물체 하나) | "parts"(여럿) | "what"(닿는 대상: 태그나 물체) | "where"(영역이나
    물체) | "thing"(물체) | "choice"(choices 가운데 하나) | "number"(숫자, low 이상)."""

    id: str
    kind: str
    default: Any = None
    choices: tuple[str, ...] = ()
    low: float | None = None


@dataclass(frozen=True)
class RuleKind:
    id: str
    world: str
    """"any" | "side": 쓸 수 있는 세계 종류."""
    slots: tuple[RuleSlot, ...]
    needs: tuple[str, ...] = ()
    """장면에 있어야 하는 것: "agent"(에이전트 물체), "region"(영역), "target"(거리를 잴 다른 것)."""
    scoring: bool = False
    """점수를 정하는 규칙인가(점수 수식이 있는 파일에서는 더하지 않는다)."""


def _slots(*items: RuleSlot) -> tuple[RuleSlot, ...]:
    return items


RULES: dict[str, RuleKind] = {
    "distance": RuleKind("distance", "any", _slots(
        RuleSlot("part", "part"), RuleSlot("direction", "choice", "right", tuple(DIRECTIONS)),
        RuleSlot("per_m", "number", 1.0, low=0.0)), ("agent",), True),
    "fall": RuleKind("fall", "side", _slots(
        RuleSlot("parts", "parts"), RuleSlot("what", "what", "tag.ground"), RuleSlot("root", "part"),
        RuleSlot("height", "number", 0.0, low=0.0), RuleSlot("penalty", "number", 1.0, low=0.0)), ("agent",), True),
    "touch_ends": RuleKind("touch_ends", "any", _slots(
        RuleSlot("part", "part"), RuleSlot("what", "what", "tag.ground"),
        RuleSlot("outcome", "choice", "failure", OUTCOMES)), ("agent",)),
    "touch_penalty": RuleKind("touch_penalty", "any", _slots(
        RuleSlot("part", "part"), RuleSlot("what", "what", "tag.ground"), RuleSlot("per_s", "number", 1.0, low=0.0)),
        ("agent",), True),
    "reach": RuleKind("reach", "any", _slots(
        RuleSlot("part", "part"), RuleSlot("where", "where"), RuleSlot("outcome", "choice", "success", OUTCOMES),
        RuleSlot("bonus", "number", 100.0, low=0.0)), ("agent", "region"), True),
    "near": RuleKind("near", "any", _slots(
        RuleSlot("thing", "thing"), RuleSlot("where", "where"), RuleSlot("per_m", "number", 1.0, low=0.0),
        RuleSlot("when", "choice", "min", WHEN)), ("agent", "target"), True),
    "time_limit": RuleKind("time_limit", "any", _slots(RuleSlot("seconds", "number", 15.0, low=0.0))),
    "survive": RuleKind("survive", "any", _slots(RuleSlot("per_s", "number", 1.0, low=0.0)), (), True),
    "upright": RuleKind("upright", "side", _slots(RuleSlot("part", "part"), RuleSlot("weight", "number", 1.0, low=0.0)),
                        ("agent",), True),
    "save_energy": RuleKind("save_energy", "any", _slots(RuleSlot("weight", "number", 0.05, low=0.0)), (), True),
    "out_ends": RuleKind("out_ends", "any", _slots(RuleSlot("outcome", "choice", "failure", OUTCOMES))),
}
"""6.2.0의 규칙 열하나(strings.RULE_TITLES와 같은 순서). 대전 규칙 넷은 6.3.0-B가 더한다."""

EVENT_IDS: dict[str, str] = {"fall": "fall", "touch_ends": "touch", "reach": "reach"}
"""규칙이 만드는 사건의 id(D11: 종목 표의 평범한 이름)."""
BLOCK_IDS: dict[str, str] = {"distance": "s_dist", "fall": "s_fall", "touch_penalty": "s_touch", "reach": "s_reach",
                             "near": "s_near", "survive": "s_survive", "upright": "s_upright", "save_energy": "s_energy"}
"""규칙이 만드는 점수 블록의 id."""


@dataclass(frozen=True)
class RuleUse:
    """장면 안의 규칙 하나. args는 규칙 말(방향 이름, 양수 가중치)이고 참조는 환경 id다."""

    kind: str
    args: Mapping[str, Any]
    refs: tuple[Ref, ...] = ()
    editable: bool = False
    """editor.rules에 적혀 있고 닫힌 검사를 넘었다(규칙 탭이 고칠 수 있다, D39)."""
    fixed: bool = False
    """늘 있는 고정 줄(제한 시간): 뺄 수 없다."""
    event: str | None = None
    blocks: tuple[str, ...] = ()
    recorded: bool = False
    """editor.rules에 적혀 있다(닫힌 검사를 못 넘었어도)."""

    @property
    def title(self) -> str:
        return S.RULE_TITLES.get(self.kind, self.kind)


# ---------------------------------------------------------------------------
# 작은 도구
# ---------------------------------------------------------------------------


def _num(x: Any) -> float:
    return round(float(x), 6) + 0.0


def num_text(x: float) -> str:
    """수식에 적는 숫자: 0.8 → "0.8", 1.0 → "1"."""
    v = _num(x)
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else repr(v)


def _items(data: Mapping[str, Any], sec: str) -> list[dict[str, Any]]:
    if sec == "blocks":
        return [b for b in ((data.get("scoring") or {}).get("blocks") or []) if isinstance(b, dict)]
    if sec == "termination":
        return [t for t in (data.get("termination") or []) if isinstance(t, dict)]
    return [e for e in (data.get(sec) or []) if isinstance(e, dict)]


def _used_ids(data: Mapping[str, Any]) -> set[str]:
    from evolab.presets.template import all_ids

    return all_ids(data)


def _fresh(data: Mapping[str, Any], base: str, taken: set[str]) -> str:
    used = _used_ids(data) | taken
    if base not in used:
        return base
    k = 2
    while f"{base}_{k}" in used:
        k += 1
    return f"{base}_{k}"


def _defaults(data: Mapping[str, Any], kind: str) -> dict[str, Any]:
    """빈칸의 처음 값: 몸의 뿌리 물체, 첫 영역 같은 장면에서 고른 것."""
    root = root_body(data)
    out: dict[str, Any] = {}
    for slot in RULES[kind].slots:
        if slot.kind in ("part",) and root is not None:
            out[slot.id] = root
        elif slot.kind == "parts" and root is not None:
            out[slot.id] = [root]
        elif slot.kind == "where":
            regions = [r["id"] for r in _items(data, "regions")]
            others = [b["id"] for b in _items(data, "bodies") if b.get("owner", "agent") == "world"]
            pick = (regions + others)[:1]
            if pick:
                out[slot.id] = pick[0]
        elif slot.kind == "thing" and root is not None:
            out[slot.id] = root
        elif slot.default is not None:
            out[slot.id] = slot.default
    if kind == "time_limit":
        out["seconds"] = _num((data.get("world") or {}).get("max_time", 20.0))
    return out


def root_body(data: Mapping[str, Any]) -> str | None:
    """뿌리 물체(선언 순서 첫 에이전트 물체)."""
    for b in _items(data, "bodies"):
        if b.get("owner", "agent") == "agent" and isinstance(b.get("id"), str):
            return b["id"]
    return None


# ---------------------------------------------------------------------------
# 조각 만들기
# ---------------------------------------------------------------------------


def _fall_when(parts: Sequence[str], what: str, root: str, height: float) -> str:
    terms = [f"contact({p}, {what})" for p in parts]
    if height > 0:
        terms.append(f"{root}.y < {num_text(height)}")
    return " or ".join(terms)


def _fragments(kind: str, a: Mapping[str, Any], event: str | None, blocks: Mapping[str, str]) -> dict[str, Any]:
    """규칙의 조각: {"event": 사건 사전 | None, "end": 종료 줄 | None, "blocks": [블록 사전]}. blocks: 역할 → id."""
    title = S.RULE_TITLES[kind]
    ev: dict[str, Any] | None = None
    end: dict[str, Any] | None = None
    out_blocks: list[dict[str, Any]] = []
    if kind == "distance":
        out_blocks.append({"kind": "progress", "id": blocks["main"], "label": title, "body": a["part"],
                           "direction": DIRECTIONS[a["direction"]], "weight": _num(a["per_m"])})
    elif kind == "fall":
        ev = {"id": event, "label": title, "when": _fall_when(a["parts"], a["what"], a["root"], float(a["height"]))}
        end = {"event": event, "outcome": "failure", "label": title}
        if float(a["penalty"]) > 0:
            out_blocks.append({"kind": "event", "id": blocks["main"], "label": title, "event": event,
                               "weight": _num(-float(a["penalty"]))})
    elif kind == "touch_ends":
        ev = {"id": event, "label": title, "when": f"contact({a['part']}, {a['what']})"}
        end = {"event": event, "outcome": a["outcome"], "label": title}
    elif kind == "touch_penalty":
        out_blocks.append({"kind": "contact_time", "id": blocks["main"], "label": title, "body": a["part"],
                           "with": [a["what"]], "weight": _num(-float(a["per_s"]))})
    elif kind == "reach":
        ev = {"id": event, "label": title, "when": f"inside({a['part']}, {a['where']})"}
        end = {"event": event, "outcome": a["outcome"], "label": title}
        if float(a["bonus"]) != 0:
            out_blocks.append({"kind": "event", "id": blocks["main"], "label": title, "event": event,
                               "weight": _num(a["bonus"])})
    elif kind == "near":
        out_blocks.append({"kind": "distance_to", "id": blocks["main"], "label": title, "body": a["thing"],
                           "target": a["where"], "agg": a["when"], "weight": _num(-float(a["per_m"]))})
    elif kind == "survive":
        out_blocks.append({"kind": "survival_time", "id": blocks["main"], "label": title, "weight": _num(a["per_s"])})
    elif kind == "upright":
        out_blocks.append({"kind": "tilt", "id": blocks["main"], "label": title, "body": a["part"], "agg": "mean",
                           "weight": _num(-float(a["weight"]))})
    elif kind == "save_energy":
        out_blocks.append({"kind": "energy", "id": blocks["main"], "label": title, "weight": _num(-float(a["weight"]))})
    elif kind == "out_ends":
        end = {"event": "out_of_bounds", "outcome": a["outcome"], "label": title}
    return {"event": ev, "end": end, "blocks": out_blocks}


def _check_args(kind: str, args: Mapping[str, Any]) -> dict[str, Any]:
    if kind not in RULES:
        raise ValueError(f"unknown rule kind: {kind!r}")
    out: dict[str, Any] = {}
    for slot in RULES[kind].slots:
        if slot.id not in args:
            raise ValueError(f"rule {kind!r} needs {slot.id!r}")
        v = args[slot.id]
        if slot.kind == "number":
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError(f"rule {kind!r}: {slot.id} must be a number")
            if slot.low is not None and float(v) < slot.low:
                raise ValueError(f"rule {kind!r}: {slot.id} must be >= {slot.low}")
            v = _num(v)
        elif slot.kind == "choice":
            if v not in slot.choices:
                raise ValueError(f"rule {kind!r}: {slot.id} must be one of {slot.choices}")
        elif slot.kind == "parts":
            if isinstance(v, str):
                v = [v]
            if not isinstance(v, (list, tuple)) or not v or not all(isinstance(x, str) and x for x in v):
                raise ValueError(f"rule {kind!r}: {slot.id} must be a non-empty list of ids")
            v = list(v)
        elif not isinstance(v, str) or not v:
            raise ValueError(f"rule {kind!r}: {slot.id} must be an id")
        out[slot.id] = v
    extra = set(args) - set(out)
    if extra:
        raise ValueError(f"rule {kind!r}: unknown fields {sorted(extra)}")
    if kind == "time_limit" and out["seconds"] <= 0:
        raise ValueError("time_limit: seconds must be > 0")
    return out


def _rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    editor = data.get("editor")
    if not isinstance(editor, Mapping) or not isinstance(editor.get(RULES_KEY), list):
        return []
    return [r for r in editor[RULES_KEY] if isinstance(r, dict)]


def _insert_end(data: dict[str, Any], end: dict[str, Any]) -> None:
    """종료 줄을 시간 끝(timeout) 줄 앞에 넣는다(없으면 끝에)."""
    ends = data.setdefault("termination", [])
    for i, t in enumerate(ends):
        if isinstance(t, dict) and t.get("event") == "timeout":
            ends.insert(i, end)
            return
    ends.append(end)


def add_rule(data: dict[str, Any], kind: str, **args: Any) -> RuleUse:
    """규칙 하나를 더한다(사전을 고침). 빈 칸은 장면에서 고른 처음 값. 제한 시간은 world.max_time을 고친다."""
    full = _defaults(data, kind)
    full.update(args)
    a = _check_args(kind, full)
    if kind == "time_limit":
        data.setdefault("world", {})["max_time"] = a["seconds"]
        if not any(isinstance(t, dict) and t.get("event") == "timeout" for t in data.get("termination", []) or []):
            data.setdefault("termination", []).append({"event": "timeout", "outcome": "neutral", "label": S.END_TIMEOUT})
        return RuleUse("time_limit", a, (), True, True)
    event = _fresh(data, EVENT_IDS[kind], set()) if kind in EVENT_IDS else None
    taken = {event} if event else set()
    block = _fresh(data, BLOCK_IDS[kind], taken) if kind in BLOCK_IDS else None
    frag = _fragments(kind, a, event, {"main": block or ""})
    refs: list[Ref] = []
    if frag["event"] is not None:
        data.setdefault("events", []).append(frag["event"])
        refs.append(("events", frag["event"]["id"]))
    if frag["end"] is not None:
        _insert_end(data, frag["end"])
        refs.append(("termination", frag["end"]["event"]))
    for b in frag["blocks"]:
        data.setdefault("scoring", {}).setdefault("blocks", []).append(b)
        refs.append(("blocks", b["id"]))
    if kind == "out_ends":
        event = "out_of_bounds"
    row = {"kind": kind, "event": event, "blocks": [b["id"] for b in frag["blocks"]]}
    editor = data.setdefault("editor", {})
    editor.setdefault(RULES_KEY, []).append(row)
    return RuleUse(kind, a, tuple(refs), True, False, event, tuple(row["blocks"]), True)


# ---------------------------------------------------------------------------
# 알아보기
# ---------------------------------------------------------------------------

_CONTACT = r"contact\(\s*([A-Za-z_\uac00-\ud7a3][\w.\uac00-\ud7a3]*)\s*,\s*([A-Za-z_\uac00-\ud7a3][\w.\uac00-\ud7a3]*)\s*\)"
_INSIDE = r"inside\(\s*([A-Za-z_\uac00-\ud7a3][\w.\uac00-\ud7a3]*)\s*,\s*([A-Za-z_\uac00-\ud7a3][\w.\uac00-\ud7a3]*)\s*\)"
_HEIGHT = r"([A-Za-z_\uac00-\ud7a3][\w\uac00-\ud7a3]*)\.y\s*<\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_EVENT_KEYS = frozenset({"id", "label", "when", "hold"})
_END_KEYS = frozenset({"event", "outcome", "label"})
_BLOCK_KEYS: dict[str, frozenset[str]] = {
    "progress": frozenset({"kind", "id", "label", "weight", "body", "direction"}),
    "event": frozenset({"kind", "id", "label", "weight", "event", "value"}),
    "contact_time": frozenset({"kind", "id", "label", "weight", "body", "with"}),
    "distance_to": frozenset({"kind", "id", "label", "weight", "body", "target", "agg"}),
    "survival_time": frozenset({"kind", "id", "label", "weight"}),
    "tilt": frozenset({"kind", "id", "label", "weight", "body", "agg"}),
    "energy": frozenset({"kind", "id", "label", "weight", "actuators"}),
}


def _parse_when(when: str) -> tuple[str, Any] | None:
    """사건 조건의 모양: ("contact", (a, b)) | ("inside", (a, b)) | ("fall", (parts, what, root, height)) | None."""
    text = when.strip()
    m1 = re.fullmatch(_CONTACT, text)
    if m1:
        return "contact", (m1.group(1), m1.group(2))
    m2 = re.fullmatch(_INSIDE, text)
    if m2:
        return "inside", (m2.group(1), m2.group(2))
    terms = [t.strip() for t in re.split(r"\s+or\s+", text)]
    if len(terms) < 2:
        return None
    parts: list[str] = []
    what: str | None = None
    root: str | None = None
    height = 0.0
    for k, t in enumerate(terms):
        mc = re.fullmatch(_CONTACT, t)
        if mc:
            if root is not None:
                return None  # 높이 조각은 끝에만
            if what is not None and mc.group(2) != what:
                return None
            what = mc.group(2)
            parts.append(mc.group(1))
            continue
        mh = re.fullmatch(_HEIGHT, t)
        if mh and k == len(terms) - 1 and root is None:
            root, height = mh.group(1), float(mh.group(2))
            if height <= 0:
                return None
            continue
        return None
    if not parts or what is None:
        return None
    return "fall", (parts, what, root or parts[0], height)


def _formula_texts(data: Mapping[str, Any]) -> list[str]:
    texts = [str(e.get("when", "")) for e in _items(data, "events")]
    texts += [str(b.get("expr", "")) for b in _items(data, "blocks") if b.get("kind") == "expr"]
    formula = (data.get("scoring") or {}).get("formula")
    if isinstance(formula, str):
        texts.append(formula)
    return texts


def _mentioned(name: str, texts: Sequence[str]) -> int:
    pat = re.compile(r"(?<![\w.])" + re.escape(name) + r"(?![\w])")
    return sum(len(pat.findall(t)) for t in texts)


class _Scene:
    def __init__(self, data: Mapping[str, Any]) -> None:
        self.data = data
        self.events = {e["id"]: e for e in _items(data, "events") if isinstance(e.get("id"), str)}
        self.ends = _items(data, "termination")
        self.blocks = {b["id"]: b for b in _items(data, "blocks") if isinstance(b.get("id"), str)}
        self.texts = _formula_texts(data)
        self.agent = {b["id"] for b in _items(data, "bodies") if b.get("owner", "agent") == "agent" and "id" in b}

    def ends_of(self, event: str) -> list[dict[str, Any]]:
        return [t for t in self.ends if t.get("event") == event]

    def event_blocks(self, event: str) -> list[dict[str, Any]]:
        return [b for b in self.blocks.values() if b.get("kind") == "event" and b.get("event") == event]

    def event_closed(self, event: str, blocks: Sequence[str]) -> bool:
        """그 사건을 가리키는 것이 종료 줄 하나와 blocks의 사건 블록뿐인가(다른 사건 조건·수식에 쓰이지 않음)."""
        if len(self.ends_of(event)) != 1:
            return False
        if {b["id"] for b in self.event_blocks(event)} - set(blocks):
            return False
        own = str(self.events[event].get("when", "")) if event in self.events else ""
        others = [t for t in self.texts if t is not own]
        return _mentioned(event, others) == 0

    def block_closed(self, block: str) -> bool:
        return _mentioned(block, self.texts) == 0


def _block_ok(b: Mapping[str, Any]) -> bool:
    keys = _BLOCK_KEYS.get(str(b.get("kind")))
    return keys is not None and set(b) <= keys


def _event_ok(e: Mapping[str, Any]) -> bool:
    return set(e) <= _EVENT_KEYS and float(e.get("hold", 0.0) or 0.0) == 0.0


def _direction_name(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    for name, d in DIRECTIONS.items():
        if abs(float(value) - d) < 1e-9:
            return name
    return None


def _from_block(scene: _Scene, b: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """사건과 무관한 점수 블록 하나의 규칙 모양."""
    if not _block_ok(b) or not scene.block_closed(b["id"]):
        return None
    kind = b.get("kind")
    w = float(b.get("weight", 1.0))
    if kind == "progress":
        d = _direction_name(b.get("direction", 0.0))
        if d is not None and w >= 0 and isinstance(b.get("body"), str):
            return "distance", {"part": b["body"], "direction": d, "per_m": _num(w)}
    elif kind == "contact_time":
        with_ = b.get("with")
        if isinstance(with_, list) and len(with_) == 1 and w <= 0:
            return "touch_penalty", {"part": b.get("body"), "what": with_[0], "per_s": _num(-w)}
    elif kind == "distance_to":
        if w <= 0 and b.get("agg", "final") in WHEN:
            return "near", {"thing": b.get("body"), "where": b.get("target"), "per_m": _num(-w),
                            "when": b.get("agg", "final")}
    elif kind == "survival_time":
        if w >= 0:
            return "survive", {"per_s": _num(w)}
    elif kind == "tilt":
        if w <= 0 and b.get("agg", "final") == "mean":
            return "upright", {"part": b.get("body"), "weight": _num(-w)}
    elif kind == "energy":
        if w <= 0 and b.get("actuators", "all") == "all":
            return "save_energy", {"weight": _num(-w)}
    return None


def _from_event(scene: _Scene, e: Mapping[str, Any], prefer: str | None = None) -> tuple[str, dict[str, Any], list[str]] | None:
    """사건 하나(와 그 종료 줄, 사건 블록)의 규칙 모양: (종류, 규칙 말, 그 규칙의 블록 id).

    접촉 하나뿐인 조건은 touch_ends다. editor.rules가 fall이라고 적었으면(prefer) fall(높이 없음)로 본다. 접촉 하나에 사건 블록이
    붙은 것(언덕차의 crash와 −20 블록)은 적혀 있지 않으면 알아보지 않는다(D39)."""
    if not _event_ok(e) or not isinstance(e.get("when"), str):
        return None
    eid = e["id"]
    ends = scene.ends_of(eid)
    if len(ends) != 1 or not set(ends[0]) <= _END_KEYS:
        return None
    outcome = ends[0].get("outcome")
    eblocks = scene.event_blocks(eid)
    if len(eblocks) > 1 or any(not _block_ok(b) or b.get("value", "occurred") != "occurred" for b in eblocks):
        return None
    if not scene.event_closed(eid, [b["id"] for b in eblocks]) or any(not scene.block_closed(b["id"]) for b in eblocks):
        return None
    shape = _parse_when(e["when"])
    if shape is None:
        return None
    w = float(eblocks[0].get("weight", 1.0)) if eblocks else 0.0
    ids = [b["id"] for b in eblocks]
    kind, vals = shape
    if kind == "contact" and prefer == "fall":
        kind, vals = "fall", ([vals[0]], vals[1], vals[0], 0.0)
    if kind == "contact" and not eblocks:
        return "touch_ends", {"part": vals[0], "what": vals[1], "outcome": outcome}, ids
    if kind == "inside" and w >= 0:
        return "reach", {"part": vals[0], "where": vals[1], "outcome": outcome, "bonus": _num(w)}, ids
    if kind == "fall" and outcome == "failure" and w <= 0:
        parts, what, root, height = vals
        return "fall", {"parts": list(parts), "what": what, "root": root, "height": _num(height),
                        "penalty": _num(-w)}, ids
    return None


def find_rules(data: Mapping[str, Any]) -> tuple[list[RuleUse], int]:
    """(알아본 규칙, 알아보지 못한 사건·종료·블록 줄 수). 제한 시간 고정 줄은 늘 마지막에 있다.

    editor.rules에 적힌 줄을 먼저 보고(모양이 그 종류와 맞고 닫혀 있으면 editable), 남은 것을 모양으로 알아본다(읽기 전용).
    """
    scene = _Scene(data)
    claimed: set[Ref] = set()
    out: list[RuleUse] = []
    recorded_events: set[str] = set()
    recorded_blocks: set[str] = set()
    for row in _rows(data):
        kind = row.get("kind")
        event = row.get("event")
        blocks = [b for b in row.get("blocks", []) or [] if isinstance(b, str)]
        if kind not in RULES or kind == "time_limit":
            continue
        use = _recorded(scene, kind, event, blocks, claimed)
        if use is not None:
            out.append(use)
            claimed.update(use.refs)
            if use.event:
                recorded_events.add(use.event)
            recorded_blocks.update(use.blocks)
    # 모양으로 알아보기(읽기 전용)
    for eid, e in scene.events.items():
        if ("events", eid) in claimed:
            continue
        got = _from_event(scene, e)
        if got is None:
            continue
        kind, args, bids = got
        refs = (("events", eid), ("termination", eid), *(("blocks", b) for b in bids))
        out.append(RuleUse(kind, args, refs, False, False, eid, tuple(bids)))
        claimed.update(refs)
    for bid, b in scene.blocks.items():
        if ("blocks", bid) in claimed:
            continue
        got2 = _from_block(scene, b)
        if got2 is None:
            continue
        kind, args = got2
        out.append(RuleUse(kind, args, (("blocks", bid),), False, False, None, (bid,)))
        claimed.add(("blocks", bid))
    outs = [t for t in scene.ends if t.get("event") == "out_of_bounds"]
    if len(outs) == 1 and ("termination", "out_of_bounds") not in claimed and set(outs[0]) <= _END_KEYS:
        out.append(RuleUse("out_ends", {"outcome": outs[0].get("outcome")}, (("termination", "out_of_bounds"),),
                           False, False, "out_of_bounds"))
        claimed.add(("termination", "out_of_bounds"))
    claimed.add(("termination", "timeout"))
    seconds = _num((data.get("world") or {}).get("max_time", 20.0))
    out.append(RuleUse("time_limit", {"seconds": seconds}, (), True, True))
    rest = 0
    rest += sum(1 for eid in scene.events if ("events", eid) not in claimed)
    rest += sum(1 for t in scene.ends if ("termination", t.get("event")) not in claimed)
    rest += sum(1 for bid in scene.blocks if ("blocks", bid) not in claimed)
    return _order(out), rest


def _order(uses: list[RuleUse]) -> list[RuleUse]:
    """editor.rules 순서(적힌 것) 먼저, 그다음 읽기 전용, 제한 시간은 끝."""
    head = [u for u in uses if u.recorded]
    mid = [u for u in uses if not u.recorded and not u.fixed]
    tail = [u for u in uses if u.fixed]
    return head + mid + tail


def _recorded(scene: _Scene, kind: str, event: Any, blocks: list[str], claimed: set[Ref]) -> RuleUse | None:
    """editor.rules의 한 줄: 적힌 사건·블록이 그 종류의 모양이고 닫혀 있으면 editable. 모양이 깨졌으면 None(모양으로 다시 봄)."""
    if kind in EVENT_IDS:
        if not isinstance(event, str) or event not in scene.events or ("events", event) in claimed:
            return None
        got = _from_event(scene, scene.events[event], kind)
        if got is None or got[0] != kind or sorted(got[2]) != sorted(blocks):
            return None
        _, args, bids = got
        refs = (("events", event), ("termination", event), *(("blocks", b) for b in bids))
        return RuleUse(kind, args, refs, True, False, event, tuple(bids), True)
    if kind == "out_ends":
        outs = [t for t in scene.ends if t.get("event") == "out_of_bounds"]
        if len(outs) != 1 or ("termination", "out_of_bounds") in claimed or not set(outs[0]) <= _END_KEYS:
            return None
        return RuleUse(kind, {"outcome": outs[0].get("outcome")}, (("termination", "out_of_bounds"),), True, False,
                       "out_of_bounds", (), True)
    if len(blocks) != 1 or blocks[0] not in scene.blocks or ("blocks", blocks[0]) in claimed:
        return None
    got2 = _from_block(scene, scene.blocks[blocks[0]])
    if got2 is None or got2[0] != kind:
        return None
    return RuleUse(kind, got2[1], (("blocks", blocks[0]),), True, False, None, (blocks[0],), True)


# ---------------------------------------------------------------------------
# 고치기·빼기
# ---------------------------------------------------------------------------


def _find(data: Mapping[str, Any], sec: str, key: str, value: str) -> dict[str, Any] | None:
    for item in _items(data, sec):
        if item.get(key) == value:
            return item
    return None


def _assign(target: dict[str, Any], key: str, value: Any) -> None:
    """값이 다를 때만 적는다(같은 값이면 사전이 그대로)."""
    if key not in target or target[key] != value or type(target[key]) is not type(value):
        if key in target and _same(target[key], value):
            return
        target[key] = value


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return float(a) == float(b)
    return bool(a == b)


def _current_use(data: Mapping[str, Any], use: RuleUse) -> RuleUse:
    for u in find_rules(data)[0]:
        if u.kind == use.kind and u.event == use.event and u.blocks == use.blocks:
            return u
        if use.fixed and u.fixed:
            return u
    raise ValueError(f"rule not found in the scene: {use.kind}")


def set_rule(data: dict[str, Any], use: RuleUse, **args: Any) -> RuleUse:
    """빈칸 뒤의 칸만 고친다(값이 같으면 사전은 그대로). editable이 아니면 ValueError."""
    if not use.editable:
        raise ValueError("this rule is read-only")
    cur = _current_use(data, use)
    if not cur.editable:
        raise ValueError("this rule is read-only")
    full = dict(cur.args)
    full.update(args)
    a = _check_args(use.kind, full)
    if use.kind == "time_limit":
        world = data.setdefault("world", {})
        if not _same(world.get("max_time", 20.0), a["seconds"]) or "max_time" not in world:
            world["max_time"] = a["seconds"]
        return RuleUse("time_limit", a, (), True, True)
    block_id = cur.blocks[0] if cur.blocks else None
    want_block = BLOCK_IDS.get(use.kind)
    frag = _fragments(use.kind, a, cur.event, {"main": block_id or want_block or ""})
    if frag["event"] is not None:
        ev = _find(data, "events", "id", cur.event or "")
        if ev is not None:
            _assign(ev, "when", frag["event"]["when"])
    if frag["end"] is not None:
        end = _find(data, "termination", "event", frag["end"]["event"])
        if end is not None:
            _assign(end, "outcome", frag["end"]["outcome"])
    new_blocks = list(frag["blocks"])
    blocks = _items(data, "blocks")
    if block_id is not None:
        b = _find(data, "blocks", "id", block_id)
        if new_blocks:
            for k, v in new_blocks[0].items():
                if k not in ("id", "label", "kind") and b is not None:
                    _assign(b, k, v)
        elif b is not None:  # 벌점·보너스를 0으로: 블록을 뺀다
            data["scoring"]["blocks"] = [x for x in blocks if x is not b]
            _drop_empty_scoring(data)
            block_id = None
    elif new_blocks:  # 벌점·보너스를 0에서 올림: 블록을 더한다
        nb = dict(new_blocks[0])
        nb["id"] = _fresh(data, want_block or "s_rule", set())
        if nb.get("kind") == "event":
            nb["event"] = cur.event
        data.setdefault("scoring", {}).setdefault("blocks", []).append(nb)
        block_id = nb["id"]
    bids = (block_id,) if block_id else ()
    for row in _rows(data):
        if row.get("kind") == use.kind and row.get("event") == cur.event and list(row.get("blocks", [])) == list(
                cur.blocks):
            if list(row.get("blocks", [])) != list(bids):
                row["blocks"] = list(bids)
            break
    refs = tuple(r for r in cur.refs if r[0] != "blocks") + tuple(("blocks", b) for b in bids)
    return RuleUse(use.kind, a, refs, True, False, cur.event, bids, True)


def _drop_empty_scoring(data: dict[str, Any]) -> None:
    scoring = data.get("scoring")
    if isinstance(scoring, dict) and scoring.get("blocks") == []:
        del scoring["blocks"]
        if not scoring:
            del data["scoring"]


def remove_rule(data: dict[str, Any], use: RuleUse) -> None:
    """규칙의 사건·종료 줄·블록과 editor.rules 줄을 뺀다. editable이 아니거나 고정 줄이면 ValueError."""
    if use.fixed or use.kind == "time_limit":
        raise ValueError("the time limit cannot be removed")
    if not use.editable:
        raise ValueError("this rule is read-only")
    cur = _current_use(data, use)
    if not cur.editable:
        raise ValueError("this rule is read-only")
    events = {i for s, i in cur.refs if s == "events"}
    ends = {i for s, i in cur.refs if s == "termination"}
    blocks = {i for s, i in cur.refs if s == "blocks"}
    if events and "events" in data:
        data["events"] = [e for e in data["events"] if not (isinstance(e, dict) and e.get("id") in events)]
        if not data["events"]:
            del data["events"]
    if ends and "termination" in data:
        data["termination"] = [t for t in data["termination"] if not (isinstance(t, dict) and t.get("event") in ends)]
        if not data["termination"]:
            del data["termination"]
    if blocks and isinstance(data.get("scoring"), dict) and "blocks" in data["scoring"]:
        data["scoring"]["blocks"] = [b for b in data["scoring"]["blocks"]
                                     if not (isinstance(b, dict) and b.get("id") in blocks)]
        _drop_empty_scoring(data)
    editor = data.get("editor")
    if isinstance(editor, dict) and isinstance(editor.get(RULES_KEY), list):
        rows = editor[RULES_KEY]
        for k, row in enumerate(rows):
            if isinstance(row, dict) and row.get("kind") == cur.kind and row.get("event") == cur.event and list(
                    row.get("blocks", [])) == list(cur.blocks):
                del rows[k]
                break
        if not rows:
            del editor[RULES_KEY]
            if not editor:
                del data["editor"]


# ---------------------------------------------------------------------------
# 화면이 쓰는 것: 고를 것, 쓸 수 있나, 문장, 요약
# ---------------------------------------------------------------------------


def _label_of(item: Mapping[str, Any]) -> str:
    return str(item.get("label") or item.get("id"))


def display_name(data: Mapping[str, Any], ref: str) -> str:
    """참조의 표시 이름: 물체·영역의 label, tag.<이름>은 그 태그가 붙은 첫 물체의 label(없으면 태그 이름)."""
    if ref.startswith("tag."):
        tag = ref[4:]
        for sec in ("bodies", "regions"):
            for item in _items(data, sec):
                if tag in (item.get("tags") or []) and item.get("owner", "world" if sec == "regions" else "agent") == "world":
                    return _label_of(item)
        return tag
    for sec in ("bodies", "regions"):
        item = _find(data, sec, "id", ref)
        if item is not None:
            return _label_of(item)
    return ref


def candidates(data: Mapping[str, Any], slot: RuleSlot) -> list[tuple[str, str]]:
    """빈칸에서 고를 것: (값, 표시 이름). 고르기 빈칸은 (이름, 화면 말)."""
    bodies = _items(data, "bodies")
    regions = _items(data, "regions")
    agent = [(b["id"], _label_of(b)) for b in bodies if b.get("owner", "agent") == "agent" and "id" in b]
    world = [(b["id"], _label_of(b)) for b in bodies if b.get("owner", "agent") == "world" and "id" in b]
    if slot.kind in ("part", "parts"):
        return agent
    if slot.kind == "what":
        tags: list[str] = []
        for b in bodies:
            if b.get("owner", "agent") == "world":
                tags += [t for t in b.get("tags", []) or [] if isinstance(t, str) and t not in tags]
        return [("tag." + t, display_name(data, "tag." + t)) for t in tags] + world
    if slot.kind == "where":
        return [(r["id"], _label_of(r)) for r in regions if "id" in r] + world + agent
    if slot.kind == "thing":
        return agent + world
    if slot.kind == "choice":
        table = {"direction": S.RULE_DIRECTIONS, "outcome": S.RULE_OUTCOMES, "when": S.RULE_WHEN}.get(slot.id, {})
        return [(c, table.get(c, c)) for c in slot.choices]
    return []


def scene_world(data: Mapping[str, Any]) -> str:
    world = data.get("world") or {}
    return "top" if world.get("view") == "top" else "side"


def available(data: Mapping[str, Any], kind: str) -> tuple[bool, str]:
    """지금 장면에서 이 규칙을 더할 수 있나와 못 하면 그 까닭(한국어). 이미 있는 고정 줄·점수 수식도 본다."""
    rk = RULES[kind]
    if kind == "time_limit":
        return False, S.RULES_FIXED_TIP
    if rk.world != "any" and scene_world(data) != rk.world:
        return False, S.TPL_WRONG_WORLD.format(world=S.TPL_WORLD_NAMES[rk.world])
    if rk.scoring and (data.get("scoring") or {}).get("formula"):
        return False, S.RULES_FORMULA_NOTE
    for need in rk.needs:
        if need == "agent" and root_body(data) is None:
            return False, fill(S.RULES_NEEDS, {"what": S.RULE_NEEDS_WHAT["agent"]})
        if need == "region" and not _items(data, "regions"):
            return False, fill(S.RULES_NEEDS, {"what": S.RULE_NEEDS_WHAT["region"]})
        if need == "target" and not (_items(data, "regions") or any(
                b.get("owner", "agent") == "world" for b in _items(data, "bodies"))):
            return False, fill(S.RULES_NEEDS, {"what": S.RULE_NEEDS_WHAT["target"]})
    if kind == "out_ends" and any(t.get("event") == "out_of_bounds" for t in _items(data, "termination")):
        return False, S.RULES_ALREADY
    return True, ""


def _display_args(data: Mapping[str, Any], use: RuleUse) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for slot in RULES[use.kind].slots:
        v = use.args.get(slot.id)
        if slot.kind == "parts":
            out[slot.id] = ", ".join(display_name(data, x) for x in v or [])
        elif slot.kind in ("part", "what", "where", "thing"):
            out[slot.id] = display_name(data, str(v))
        elif slot.kind == "choice":
            table = {"direction": S.RULE_DIRECTIONS, "outcome": S.RULE_OUTCOMES, "when": S.RULE_WHEN}[slot.id]
            out[slot.id] = table.get(v, v)
        else:
            out[slot.id] = v
    return out


def sentence(data: Mapping[str, Any], use: RuleUse) -> str:
    """규칙 문장(조사까지 맞춘 것): 「몸통이 오른쪽으로 간 거리만큼 점수 (1m에 1점)」."""
    return fill(S.RULE_SENTENCES[use.kind], _display_args(data, use))


def _w(x: float) -> float:
    return _num(x)


def summary(data: Mapping[str, Any]) -> str:
    """「점수 = …」의 오른쪽: 규칙마다 한 항, 알아보지 못한 블록은 「고급 블록 n개」. 수식이 있으면 RULES_FORMULA_NOTE."""
    if (data.get("scoring") or {}).get("formula"):
        return S.RULES_FORMULA_NOTE
    uses, _ = find_rules(data)
    order = {b.get("id"): k for k, b in enumerate(_items(data, "blocks"))}
    uses = sorted((u for u in uses if u.blocks), key=lambda u: min(order.get(b, 1 << 30) for b in u.blocks))
    terms: list[str] = []
    seen: set[str] = set()
    for u in uses:
        if not u.blocks:
            continue
        seen.update(u.blocks)
        a = u.args
        if u.kind == "distance":
            terms.append(S.RULE_SUMMARY_TERMS["distance"].format(w=_w(a["per_m"])))
        elif u.kind == "fall":
            terms.append(S.RULE_SUMMARY_TERMS["fall"].format(w=_w(a["penalty"])))
        elif u.kind == "near":
            terms.append(S.RULE_SUMMARY_TERMS["near"].format(thing=display_name(data, a["thing"]),
                                                             where=display_name(data, a["where"]), w=_w(a["per_m"])))
        elif u.kind == "reach":
            terms.append(S.RULE_SUMMARY_TERMS["reach"].format(where=display_name(data, a["where"]), w=_w(a["bonus"])))
        elif u.kind == "survive":
            terms.append(S.RULE_SUMMARY_TERMS["survive"].format(w=_w(a["per_s"])))
        elif u.kind == "upright":
            terms.append(S.RULE_SUMMARY_TERMS["upright"].format(w=_w(a["weight"])))
        elif u.kind == "save_energy":
            terms.append(S.RULE_SUMMARY_TERMS["save_energy"].format(w=_w(a["weight"])))
        elif u.kind == "touch_penalty":
            terms.append(S.RULE_SUMMARY_TERMS["touch_penalty"].format(w=_w(a["per_s"])))
    other = [b for b in _items(data, "blocks") if b.get("id") not in seen]
    if other:
        terms.append(S.RULE_SUMMARY_TERMS["other"].format(n=len(other)))
    if not terms:
        return S.RULE_SUMMARY_NONE
    text = terms[0]
    for t in terms[1:]:
        text += (" " + t) if t.startswith("−") else (" + " + t)
    return text


def rule_needs(kind: str) -> tuple[str, ...]:
    return RULES[kind].needs


@dataclass(frozen=True)
class RuleRowInfo:
    """화면 한 줄의 정보(규칙 탭·안내 창이 함께 씀)."""

    use: RuleUse
    text: str
    tip: str = field(default="")


def rows(data: Mapping[str, Any]) -> tuple[list[RuleRowInfo], int]:
    """(규칙 줄, 고급 탭에 남은 줄 수). 읽기 전용 줄의 풍선은 RULES_READONLY_TIP, 고정 줄은 RULES_FIXED_TIP."""
    uses, rest = find_rules(data)
    out = []
    for u in uses:
        tip = S.RULES_FIXED_TIP if u.fixed else ("" if u.editable else S.RULES_READONLY_TIP)
        out.append(RuleRowInfo(u, sentence(data, u), tip))
    return out, rest


__all__ = [
    "BLOCK_IDS", "DIRECTIONS", "EVENT_IDS", "RULES", "RuleKind", "RuleRowInfo", "RuleSlot", "RuleUse", "add_rule",
    "available", "candidates", "display_name", "find_rules", "num_text", "remove_rule", "root_body", "rows",
    "scene_world", "sentence", "set_rule", "summary",
]

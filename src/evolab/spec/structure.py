"""몸 구조(v2.1): 물체의 있음 칸(exists)과 관절 이어짐 규칙. 런타임·검사·화면·에디터가 함께 쓰는 순수 함수.

- 있음 칸은 에이전트 물체에만 쓴다(세계 물체의 exists는 검사 오류라 여기서는 보지 않는다). 수식은 p.*만 쓰는 만들기 단계
  structure에서 계산하므로(spec/symbols.build_symbols) 몸 구조는 몸 수치만의 함수다. 그래서 화면과 에디터는 시뮬레이션
  없이 구조를 계산하고, 같은 개체는 어느 시험·세대·워커에서나 같은 몸이다.
- 꺼진 물체(OFF): 있음 값이 0.5 이하. 수식 결과가 유한하지 않으면 0으로 본다(런타임 만들기 단계 수식과 같은 규칙).
- 이어짐 규칙(v2 설계 D34): 관절 그래프(관절 a·b가 모두 에이전트 물체인 관절)에서 뿌리(선언 순서 첫 에이전트 물체)와
  이어진 에이전트 물체를 "모두 있을 때"(FULL)와 "OFF를 뺐을 때"(KEPT) 구해 없는 물체 = OFF ∪ (FULL − KEPT). 꺼진 부품에
  관절로 매달린 부품도 함께 없어지고(공중에 떠다니지 않음), 처음부터 뿌리와 이어지지 않은 에이전트 물체는 자기 있음 칸만
  따른다(v1 파일에 그런 물체가 있을 수 있다).
- exists를 쓰는 물체가 하나도 없으면 없는 물체가 없다(v1·v2.0 파일은 이 길만 지난다).

pymunk를 쓰지 않는다. spec/compiled.py가 이 모듈을 가져다 쓸 수 있게(없어질 수 있는 물체 검사) EnvSpec은 형 검사 때만
가져온다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Final

import numpy as np

from evolab.scoring.expr import formula_body
from evolab.spec import model as m

if TYPE_CHECKING:
    from evolab.spec.compiled import EnvSpec

EXISTS_THRESHOLD: Final = 0.5
"""있음 값이 이보다 커야 물체를 만든다(같으면 없음)."""
STRUCTURE_STAGE: Final = "structure"
"""있음 칸 수식의 만들기 단계(EnvSpec.formulas의 열쇠 (원문, 단계)의 단계)."""


def _agent_ids(env: m.EnvFile) -> list[str]:
    return [b.id for b in env.bodies if b.owner == "agent"]


def root_body(env: m.EnvFile) -> str | None:
    """뿌리 물체: 선언 순서 첫 에이전트 물체. 늘 있다(있음 칸을 쓸 수 없음). 에이전트 물체가 없으면 None."""
    for b in env.bodies:
        if b.owner == "agent":
            return b.id
    return None


def connected(env: m.EnvFile, removed: AbstractSet[str]) -> frozenset[str]:
    """removed를 뺀 관절 그래프(관절 a·b, 에이전트 물체끼리)에서 뿌리와 이어진 에이전트 물체(뿌리 포함).

    뿌리가 removed에 있거나 에이전트 물체가 없으면 빈 집합. 관절의 방향(부모·자식)과 종류는 따지지 않는다.
    """
    root = root_body(env)
    if root is None or root in removed:
        return frozenset()
    nodes = {i for i in _agent_ids(env) if i not in removed}
    adjacent: dict[str, list[str]] = {}
    for j in env.joints:
        if j.a in nodes and j.b in nodes and j.a != j.b:
            adjacent.setdefault(j.a, []).append(j.b)
            adjacent.setdefault(j.b, []).append(j.a)
    seen = {root}
    todo = [root]
    while todo:
        for nxt in adjacent.get(todo.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return frozenset(seen)


def absent_from(env: m.EnvFile, off: AbstractSet[str]) -> frozenset[str]:
    """이어짐 규칙(D34): off ∪ (connected(env, ∅) − connected(env, off)). off는 꺼진 에이전트 물체 id.

    off가 클수록 결과도 크다(단조): 더 끄면 뿌리와 이어진 물체가 줄어들 뿐이다.
    """
    off = frozenset(off)
    if not off:
        return off
    return off | (connected(env, frozenset()) - connected(env, off))


def _may_be_off(body: m.Body) -> bool:
    """있음 칸이 수식이거나 상수 0.5 이하인 에이전트 물체(그 칸만으로 꺼질 수 있음)."""
    if body.owner != "agent" or body.exists is None:
        return False
    return isinstance(body.exists, str) or not float(body.exists) > EXISTS_THRESHOLD


def optional_bodies(env: m.EnvFile) -> frozenset[str]:
    """없어질 수 있는 물체(D37): 있음 칸이 수식이거나 상수 0.5 이하인 에이전트 물체를 모두 끈 absent_from.

    검사(없어질 수 있는 물체를 가리키는 사건·점수 등은 오류), 런타임(Episode.absent는 늘 이 집합 안), 화면(몸 구조 타일의
    범례 열쇠), 에디터(구조 탭)가 같은 집합을 쓴다. EnvSpec 없이 모델만으로 계산하므로 오류가 있는 파일에도 쓸 수 있다.
    """
    return absent_from(env, {b.id for b in env.bodies if _may_be_off(b)})


def exists_param_id(body: m.Body) -> str | None:
    """있음 칸이 정확히 "=p.<id>" 하나(공백 무시)이면 그 <id>, 아니면 None(그런 매개변수가 있는지는 보지 않는다).

    구조 뒤집기의 있음 유전자(policy/genome.layout_for의 on_genes)와 아래 off_centre_exists가 같은 규칙을 쓴다.
    "=(p.a)"나 "=p.a > 0.5"처럼 다른 모양은 None.
    """
    if not isinstance(body.exists, str):
        return None
    text = "".join(formula_body(body.exists).split())
    return text[2:] if text.startswith("p.") and len(text) > 2 else None


def flip_threshold(low: float, high: float) -> float:
    """범위 low..high 매개변수에서 있음 문턱(물리 값 0.5)의 유전자 공간(-1..1) 위치. 범위 가운데가 0.5면 0.

    구조 뒤집기(learners/structure.flip_structure)는 유전자의 부호를 뒤집으므로 이 값이 0인 유전자만 뒤집기가 곧 구조
    바꾸기다(on_genes 규칙). 0..1과 -1..2 같은 범위가 그렇다.
    """
    return 2.0 * (EXISTS_THRESHOLD - low) / (high - low) - 1.0


FLIP_CENTRE_TOLERANCE: Final = 1e-9
"""flip_threshold가 이 값 이하이면 범위 가운데가 0.5라고 본다."""


def off_centre_exists(env: m.EnvFile) -> tuple[tuple[str, str], ...]:
    """구조 뒤집기가 건드리지 않는 있음 칸: (물체 id, 매개변수 id), 선언 순서.

    에이전트 물체의 있음 칸이 정확히 "=p.<id>"이고 그 매개변수가 진화하는데(범위 있음) 범위 가운데가 0.5가 아닌 것.
    on_genes 규칙에서 범위 때문에 빠진 물체다(예: 범위 0..2). 그대로 쓸 수 있지만(CMA-ES와 가우시안 돌연변이는 다룬다)
    GA·담금질이 구조를 바꾸지 못하므로 검사 안내와 실험실·명령줄 안내가 알린다(v2.1 검토 반영: 조용히 빠지지 않게).
    """
    params = {p.id: p for p in env.params}
    out: list[tuple[str, str]] = []
    for b in env.bodies:
        if b.owner != "agent":
            continue
        pid = exists_param_id(b)
        p = params.get(pid) if pid is not None else None
        if p is None or not p.evolve or p.min is None or p.max is None or not float(p.max) > float(p.min):
            continue
        if abs(flip_threshold(float(p.min), float(p.max))) > FLIP_CENTRE_TOLERANCE:
            out.append((b.id, p.id))
    return tuple(out)


def exists_value(spec: EnvSpec, body: m.Body, params: Mapping[str, float]) -> float:
    """물체 하나의 있음 값(비었으면 1). 수식은 spec이 단계 structure로 컴파일해 둔 것을 params로 계산한다."""
    value = body.exists
    if value is None:
        return 1.0
    if isinstance(value, str):
        x = float(spec.formulas[(value, STRUCTURE_STAGE)](params))
    else:
        x = float(value)
    return x if math.isfinite(x) else 0.0


def present(spec: EnvSpec, params: Mapping[str, float]) -> frozenset[str]:
    """있는 에이전트 물체: 에이전트 물체 전체 − absent_from(off).

    params = {"p.<id>": 물리 값}(Episode.params, param_values와 같은 사전. 있음 수식이 쓰는 p.*가 모두 있어야 한다).
    off = 있음 값이 0.5 이하인 에이전트 물체. 있음 칸을 쓰는 물체가 없으면 모든 에이전트 물체.
    """
    env = spec.file
    agents = frozenset(_agent_ids(env))
    off = {b.id for b in env.bodies
           if b.owner == "agent" and b.exists is not None and not exists_value(spec, b, params) > EXISTS_THRESHOLD}
    if not off:
        return agents
    return agents - absent_from(env, off)


def structure_key(spec: EnvSpec, params: Mapping[str, float]) -> tuple[str, ...]:
    """「몸 구조」 타일의 범례 열쇠: optional_bodies 중 있는 것(선언 순서). 없어질 수 있는 물체가 없으면 ()."""
    optional = optional_bodies(spec.file)
    if not optional:
        return ()
    here = present(spec, params)
    return tuple(b.id for b in spec.file.bodies if b.id in optional and b.id in here)


def param_values(env: m.EnvFile, body: Sequence[float] | np.ndarray | None = None) -> dict[str, float]:
    """present·structure_key에 넘길 사전 {"p.<id>": 물리 값}. Episode.params와 같은 규칙이다.

    body가 None이면 매개변수 값(처음 몸). 아니면 진화하는 매개변수(선언 순서 = 몸 유전자 순서 = 관찰 기록 obs_body의 열
    순서)를 그 값으로 바꾼다: [min, max]로 자르고, 유한하지 않으면 매개변수 값. 개수가 다르면 ValueError.
    """
    out = {f"p.{p.id}": float(p.value) for p in env.params}
    if body is None:
        return out
    evolved = [p for p in env.params if p.evolve]
    values = [float(x) for x in np.asarray(body, dtype=np.float64).ravel()]
    if len(values) != len(evolved):
        raise ValueError(f"body values: got {len(values)}, want {len(evolved)}")
    for p, x in zip(evolved, values, strict=True):
        lo = p.min if p.min is not None else x
        hi = p.max if p.max is not None else x
        out[f"p.{p.id}"] = min(max(x, lo), hi) if math.isfinite(x) else float(p.value)
    return out

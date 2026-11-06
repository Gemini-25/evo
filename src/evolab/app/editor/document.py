"""에디터의 문서 모델: 편집 중인 환경 파일 하나 (명세: 화면과 조작 > 에디터).

- 문서의 진짜 내용은 환경 파일과 같은 모양의 JSON 사전(`data`)이다. 화면 부품(캔버스, 속성 패널, 아래 탭)은 이
  사전을 읽고, 바꿀 때는 반드시 이 클래스의 편집 함수를 거친다. 그래야 되돌리기·다시하기와 실시간 검사가 맞는다.
- 편집 하나 = 되돌리기 단계 하나. 단계는 바꾸기 전·후 사전 전체를 갖는다(환경 파일은 작아서 충분히 싸다).
  같은 칸을 이어서 바꾸면(끌기, 숫자 칸 타이핑) 한 단계로 합친다(merge_key).
- 경로는 사전 안의 위치다: ("bodies", 0, "shapes", 0, "w"). 부품은 (구역, id)로 가리킨다: ("bodies", "hull").
- 진화 대상: 숫자 칸을 진화 대상으로 켜면 매개변수 p.<id>(evolve, 최솟값·최댓값)를 만들고 칸을 "=p.<id>"로 바꾼다.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from evolab import paths
from evolab.app import strings as S
from evolab.spec import model as m
from evolab.spec.compiled import EnvFileError, EnvSpec, Issue, check_env
from evolab.spec.io import dump_env, parse_env

JsonPath = tuple[str | int, ...]
Ref = tuple[str, str]
"""(구역 이름, id). 구역: bodies, joints, actuators, sensors, regions, checkpoints, events, generators, randomize,
params, scoring.blocks"""

SECTIONS = ("generators", "bodies", "joints", "actuators", "sensors", "regions", "checkpoints", "events",
            "randomize", "params")
CLIPBOARD_FORMAT = "evolab-components"
_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WHERE_RE = re.compile(r"^([a-z_.]+)\[(\d+)\]")


def empty_env(name: str | None = None) -> dict[str, Any]:
    """새로 만들기: 빈 2D 세계(중력 아래, 경계, 시간 초과 종료)."""
    return {
        "version": 1,
        "name": name or S.ED_NEW_NAME,
        "world": {"kind": "physics2d", "gravity": [0.0, -9.81], "dt": 1.0 / 60.0, "max_time": 20.0,
                  "bounds": [-50.0, -20.0, 50.0, 60.0]},
        "termination": [{"event": "timeout", "outcome": "neutral", "label": S.ED_DEFAULT_TIMEOUT_LABEL}],
    }


def get_path(data: Any, path: JsonPath) -> Any:
    cur = data
    for key in path:
        cur = cur[key]
    return cur


def set_path(data: Any, path: JsonPath, value: Any) -> None:
    """경로의 값을 바꾼다. 중간 사전이 없으면 만든다(목록 칸은 있어야 한다)."""
    cur = data
    for key in path[:-1]:
        if isinstance(key, str) and isinstance(cur, dict) and key not in cur:
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


def del_path(data: Any, path: JsonPath) -> None:
    cur = get_path(data, path[:-1])
    if isinstance(cur, dict):
        cur.pop(path[-1], None)
    else:
        del cur[path[-1]]


def where_to_ref(data: dict[str, Any], where: str) -> Ref | None:
    """검사 결과의 경로(bodies[1].shapes[0].h)를 부품(구역, id)으로. 강조 표시에 쓴다."""
    mt = _WHERE_RE.match(where)
    if mt is None:
        return None
    section, index = mt.group(1), int(mt.group(2))
    try:
        items = get_path(data, tuple(section.split(".")))
        return (section, str(items[index].get("id", index)))
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


class _Edit(QUndoCommand):
    def __init__(self, doc: EnvDocument, label: str, before: dict[str, Any], after: dict[str, Any],
                 merge_key: str | None) -> None:
        super().__init__(label)
        self.doc = doc
        self.before = before
        self.after = after
        self.merge_key = merge_key
        self._first = True

    def id(self) -> int:  # noqa: A003 (Qt 이름)
        return 1 if self.merge_key else -1

    def mergeWith(self, other: QUndoCommand) -> bool:  # noqa: N802
        if not isinstance(other, _Edit) or other.merge_key is None or other.merge_key != self.merge_key:
            return False
        self.after = other.after
        return True

    def redo(self) -> None:
        if self._first:  # push 할 때 이미 바꿔 두었다
            self._first = False
            return
        self.doc._replace(copy.deepcopy(self.after))

    def undo(self) -> None:
        self.doc._replace(copy.deepcopy(self.before))


class EnvDocument(QObject):
    """편집 중인 환경 파일."""

    changed = Signal()
    """내용이 바뀌었다(편집, 되돌리기, 불러오기)."""
    selection_changed = Signal(list)
    checked = Signal(list)
    """실시간 검사 결과(list[Issue]). 바뀐 뒤 잠깐 기다렸다가 한 번 한다."""
    file_changed = Signal()
    """경로나 저장 상태가 바뀌었다(창 제목용)."""

    def __init__(self, data: dict[str, Any] | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.data: dict[str, Any] = copy.deepcopy(data) if data is not None else empty_env()
        self.path: Path | None = None
        self.selection: list[Ref] = []
        self.undo_stack = QUndoStack(self)
        self.undo_stack.cleanChanged.connect(self._on_clean_changed)
        self.issues: list[Issue] = []
        self._check_timer = QTimer(self)
        self._check_timer.setSingleShot(True)
        self._check_timer.setInterval(250)
        self._check_timer.timeout.connect(self.run_check)
        self._spec: EnvSpec | None = None
        self._spec_for: str | None = None
        self.run_check()

    def _on_clean_changed(self, _clean: bool) -> None:
        self.file_changed.emit()

    # ------------------------------------------------------------------
    # 파일
    # ------------------------------------------------------------------
    @property
    def dirty(self) -> bool:
        return not self.undo_stack.isClean()

    def title(self) -> str:
        name = self.path.name if self.path else S.ED_UNTITLED
        return f"{name}{' *' if self.dirty else ''}"

    def load_text(self, text: str, path: Path | None = None) -> None:
        """파일 내용으로 바꾼다(되돌리기 기록은 지운다). 구조가 잘못된 파일이면 EnvFileError."""
        env = parse_env(text)  # 구조 검사(참조·수식 오류는 검사 결과 탭에 보인다)
        self.data = json.loads(dump_env(env))
        self.path = path
        self.selection = []
        self.undo_stack.clear()
        self.undo_stack.setClean()
        self._invalidate()
        self.changed.emit()
        self.selection_changed.emit([])
        self.file_changed.emit()
        self.run_check()

    def open(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            raise EnvFileError([Issue("error", "", S.SPEC_NOT_UTF8)]) from None
        except OSError as exc:
            raise EnvFileError([Issue("error", "", S.SPEC_READ_FAILED.format(path=path, reason=exc.strerror or exc))]) from None
        self.load_text(text, path)

    def new(self) -> None:
        self.load_text(json.dumps(empty_env(), ensure_ascii=False), None)

    def from_builtin(self, name: str) -> None:
        """내장 예제에서 시작하기: 내용만 가져오고 저장 위치는 비운다(내장 파일을 덮어쓰지 않게)."""
        self.load_text((paths.builtin_env_dir() / name).read_text(encoding="utf-8"), None)

    def text(self) -> str:
        """저장할 내용. 구조가 맞으면 정규 형식(dump_env), 아니면 사전 그대로."""
        try:
            return dump_env(parse_env(self.data))
        except EnvFileError:
            return json.dumps(self.data, ensure_ascii=False, indent=2) + "\n"

    def save(self, path: Path | None = None) -> Path:
        target = path or self.path
        if target is None:
            raise ValueError("no path")
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(self.text(), encoding="utf-8")
        tmp.replace(target)
        self.path = target
        self.undo_stack.setClean()
        self.file_changed.emit()
        return target

    # ------------------------------------------------------------------
    # 편집
    # ------------------------------------------------------------------
    def edit(self, label: str, mutate: Callable[[dict[str, Any]], Any], merge_key: str | None = None) -> Any:
        """사전 사본을 mutate로 바꾸고 되돌리기 단계로 남긴다. mutate의 반환값을 돌려준다."""
        before = copy.deepcopy(self.data)
        after = copy.deepcopy(self.data)
        result = mutate(after)
        if after == before:
            return result
        self.data = after
        self._invalidate()
        self.undo_stack.push(_Edit(self, label, before, copy.deepcopy(after), merge_key))
        self._after_change()
        return result

    def _replace(self, data: dict[str, Any]) -> None:
        self.data = data
        self._invalidate()
        self.selection = [r for r in self.selection if self.find(r) is not None]
        self._after_change()
        self.selection_changed.emit(list(self.selection))

    def _after_change(self) -> None:
        self.changed.emit()
        self._check_timer.start()

    def _invalidate(self) -> None:
        self._spec = None
        self._spec_for = None

    def get(self, path: JsonPath, default: Any = None) -> Any:
        try:
            return get_path(self.data, path)
        except (KeyError, IndexError, TypeError):
            return default

    def set_value(self, path: JsonPath, value: Any, label: str | None = None, merge: bool = True) -> None:
        key = "set:" + "/".join(str(k) for k in path) if merge else None
        self.edit(label or S.ED_UNDO_SET, lambda d: set_path(d, path, value), key)

    def items(self, section: str) -> list[dict[str, Any]]:
        value = self.get(tuple(section.split(".")), [])
        return value if isinstance(value, list) else []

    def find(self, ref: Ref) -> int | None:
        section, cid = ref
        for i, item in enumerate(self.items(section)):
            if str(item.get("id")) == cid:
                return i
        return None

    def item(self, ref: Ref) -> dict[str, Any] | None:
        i = self.find(ref)
        return None if i is None else self.items(ref[0])[i]

    def item_path(self, ref: Ref) -> JsonPath | None:
        i = self.find(ref)
        return None if i is None else (*ref[0].split("."), i)

    def all_ids(self, data: dict[str, Any] | None = None) -> set[str]:
        d = self.data if data is None else data
        out: set[str] = set()
        for section in (*SECTIONS, "scoring.blocks"):
            try:
                items = get_path(d, tuple(section.split(".")))
            except (KeyError, TypeError):
                continue
            out |= {str(x.get("id")) for x in items if isinstance(x, dict) and "id" in x}
        return out

    def unique_id(self, base: str, taken: set[str] | None = None) -> str:
        base = base if _ID_RE.match(base) else "part"
        used = self.all_ids() | (taken or set())
        if base not in used:
            return base
        stem = re.sub(r"_?\d+$", "", base) or "part"
        k = 1
        while f"{stem}_{k}" in used:
            k += 1
        return f"{stem}_{k}"

    def add(self, section: str, item: dict[str, Any], label: str | None = None, select: bool = True) -> Ref:
        """부품 하나를 더한다(id가 겹치면 바꾼다). 더한 부품의 Ref."""
        item = copy.deepcopy(item)
        item["id"] = self.unique_id(str(item.get("id") or section.rstrip("s")))
        keys = tuple(section.split("."))

        def mutate(d: dict[str, Any]) -> None:
            try:
                lst = get_path(d, keys)
            except KeyError:
                set_path(d, keys, [])
                lst = get_path(d, keys)
            lst.append(item)

        self.edit(label or S.ED_UNDO_ADD, mutate)
        ref = (section, item["id"])
        if select:
            self.select([ref])
        return ref

    def remove(self, refs: list[Ref], label: str | None = None) -> None:
        """부품을 지운다. 그 부품을 가리키던 관절·구동기·센서 같은 것은 지우지 않는다(검사 결과에 참조 오류로 보인다)."""
        def mutate(d: dict[str, Any]) -> None:
            for section, cid in refs:
                keys = tuple(section.split("."))
                try:
                    lst = get_path(d, keys)
                except KeyError:
                    continue
                lst[:] = [x for x in lst if str(x.get("id")) != cid]

        self.edit(label or S.ED_UNDO_REMOVE, mutate)
        self.select([r for r in self.selection if r not in refs])

    def rename(self, ref: Ref, new_id: str) -> bool:
        """id를 바꾸고, 이 id를 가리키는 곳(참조 칸, 수식, 태그 선택자 제외)도 함께 바꾼다."""
        if not _ID_RE.match(new_id) or new_id in self.all_ids() or self.find(ref) is None:
            return False
        old = ref[1]
        word = re.compile(rf"(?<![A-Za-z0-9_.]){re.escape(old)}(?![A-Za-z0-9_])")

        def fix(value: Any, top: bool) -> Any:
            if isinstance(value, dict):
                return {k: fix(v, False) for k, v in value.items()}
            if isinstance(value, list):
                return [fix(v, False) for v in value]
            if isinstance(value, str):
                if value == old:
                    return new_id
                if value.startswith("=") or (" " in value or "(" in value):
                    return word.sub(new_id, value)
            return value

        def mutate(d: dict[str, Any]) -> None:
            path = self.item_path(ref)
            assert path is not None
            get_path(d, path)["id"] = new_id
            for section in [*SECTIONS, "scoring", "termination", "brain", "agent", "world"]:
                if section in d:
                    d[section] = fix(d[section], True)

        self.edit(S.ED_UNDO_RENAME, mutate)
        self.select([(ref[0], new_id) if r == ref else r for r in self.selection])
        return True

    # ------------------------------------------------------------------
    # 진화 대상
    # ------------------------------------------------------------------
    def evolve_param(self, path: JsonPath) -> str | None:
        """이 칸이 진화 대상 매개변수 하나("=p.<id>")를 그대로 가리키면 그 id."""
        value = self.get(path)
        if isinstance(value, str):
            mt = re.fullmatch(r"=\s*p\.([A-Za-z_][A-Za-z0-9_]*)\s*", value)
            if mt:
                pid = mt.group(1)
                for p in self.items("params"):
                    if p.get("id") == pid and p.get("evolve"):
                        return pid
        return None

    def make_evolvable(self, path: JsonPath, low: float, high: float, name: str | None = None,
                       label: str = "") -> str:
        """숫자 칸을 진화 대상으로: 매개변수를 만들고(초깃값 = 지금 값) 칸을 "=p.<id>"로 바꾼다."""
        value = self.get(path)
        current = float(value) if isinstance(value, (int, float)) else (low + high) / 2.0
        pid = self.unique_id(name or "_".join(str(k) for k in path[-3:] if isinstance(k, str)) or "param")

        def mutate(d: dict[str, Any]) -> None:
            d.setdefault("params", []).append({"id": pid, "label": label, "value": current, "evolve": True,
                                               "min": float(low), "max": float(high)})
            set_path(d, path, f"=p.{pid}")

        self.edit(S.ED_UNDO_EVOLVE_ON, mutate)
        return pid

    def make_fixed(self, path: JsonPath) -> None:
        """진화 대상을 끈다: 칸을 매개변수 초깃값으로 되돌리고, 다른 데서 안 쓰면 매개변수도 지운다."""
        pid = self.evolve_param(path)
        if pid is None:
            return

        def mutate(d: dict[str, Any]) -> None:
            params = d.get("params", [])
            value = next(p.get("value", 0.0) for p in params if p.get("id") == pid)
            set_path(d, path, value)
            still_used = any(pid in s for s in _strings(d) if s.startswith("="))
            if not still_used:
                d["params"] = [p for p in params if p.get("id") != pid]

        self.edit(S.ED_UNDO_EVOLVE_OFF, mutate)

    # ------------------------------------------------------------------
    # 선택, 복사·붙여넣기
    # ------------------------------------------------------------------
    def select(self, refs: list[Ref]) -> None:
        refs = [r for r in refs if self.find(r) is not None]
        if refs != self.selection:
            self.selection = refs
            self.selection_changed.emit(list(refs))

    def copy_text(self, refs: list[Ref] | None = None) -> str:
        refs = self.selection if refs is None else refs
        items = [{"section": r[0], "item": self.item(r)} for r in refs if self.item(r) is not None]
        return json.dumps({"format": CLIPBOARD_FORMAT, "items": items}, ensure_ascii=False)

    def paste_text(self, text: str, offset: tuple[float, float] = (1.0, -1.0)) -> list[Ref]:
        """복사한 부품을 붙인다. id가 겹치면 바꾸고, 붙인 부품끼리의 참조는 새 id로 이어 준다. 위치는 조금 옮긴다."""
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return []
        if not isinstance(data, dict) or data.get("format") != CLIPBOARD_FORMAT:
            return []
        entries = [e for e in data.get("items", []) if isinstance(e, dict) and isinstance(e.get("item"), dict)
                   and e.get("section") in (*SECTIONS, "scoring.blocks")]
        if not entries:
            return []
        taken: set[str] = set()
        renames: dict[str, str] = {}
        for e in entries:
            old = str(e["item"].get("id", "part"))
            new = self.unique_id(old, taken)
            taken.add(new)
            renames[old] = new
        new_refs: list[Ref] = []

        def remap(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: remap(v) for k, v in value.items()}
            if isinstance(value, list):
                return [remap(v) for v in value]
            if isinstance(value, str) and value in renames:
                return renames[value]
            return value

        def mutate(d: dict[str, Any]) -> None:
            for e in entries:
                item = remap(copy.deepcopy(e["item"]))
                item["id"] = renames[str(e["item"].get("id", "part"))]
                pos = item.get("pos")
                if isinstance(pos, list) and len(pos) == 2 and all(isinstance(v, (int, float)) for v in pos):
                    item["pos"] = [pos[0] + offset[0], pos[1] + offset[1]]
                keys = tuple(e["section"].split("."))
                try:
                    get_path(d, keys).append(item)
                except KeyError:
                    set_path(d, keys, [item])
                new_refs.append((e["section"], item["id"]))

        self.edit(S.ED_UNDO_PASTE, mutate)
        self.select(new_refs)
        return new_refs

    # ------------------------------------------------------------------
    # 검사와 컴파일
    # ------------------------------------------------------------------
    def run_check(self) -> list[Issue]:
        """구조 검사 + 참조·수식 검사 + 에디터 검사(아무 데도 안 붙은 부품, 겹친 물체)."""
        self._check_timer.stop()
        try:
            env = parse_env(self.data)
        except EnvFileError as exc:
            self.issues = list(exc.issues)
        else:
            self.issues = check_env(env)
            if not any(i.severity == "error" for i in self.issues):
                self.issues += editor_checks(env, self.spec())
        self.checked.emit(list(self.issues))
        return self.issues

    def spec(self) -> EnvSpec | None:
        """지금 내용을 컴파일한 것(오류가 있으면 None). 같은 내용이면 다시 하지 않는다."""
        key = json.dumps(self.data, sort_keys=True, ensure_ascii=False)
        if self._spec_for == key:
            return self._spec
        self._spec_for = key
        try:
            from evolab.spec.io import compile_env

            self._spec = compile_env(copy.deepcopy(self.data))
        except EnvFileError:
            self._spec = None
        return self._spec

    def issues_for(self, ref: Ref) -> list[Issue]:
        return [i for i in self.issues if where_to_ref(self.data, i.where) == ref]


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, str):
        yield value


def editor_checks(env: m.EnvFile, spec: EnvSpec | None) -> list[Issue]:
    """에디터만 하는 검사(경고): 관절로 안 이어진 에이전트 물체, 시작할 때 겹친 물체."""
    out: list[Issue] = []
    agent = [i for i, b in enumerate(env.bodies) if b.owner == "agent"]
    if len(agent) > 1:
        linked: set[str] = set()
        for j in env.joints:
            linked |= {j.a, j.b}
        for i in agent:
            if env.bodies[i].id not in linked:
                out.append(Issue("warning", f"bodies[{i}]", S.ED_CHECK_UNATTACHED.format(id=env.bodies[i].id)))
    if spec is not None:
        out += _overlap_checks(env, spec)
    return out


def _overlap_checks(env: m.EnvFile, spec: EnvSpec) -> list[Issue]:
    """미리보기 시험으로 세계를 만들어, 시작 자세에서 서로 부딪히는 물체 쌍을 찾는다(관절로 이은 쌍과 에이전트끼리 제외)."""
    from evolab.engine import seeds
    from evolab.policy.genome import layout_for
    from evolab.world.physics2d.episode import Episode, EpisodeBuildError

    try:
        layout = layout_for(spec)
        body = layout.body_values(layout.initial()[None, :])[0] if layout.n_body else None
        trial = seeds.display_trial()
        if spec.track_set_size:
            from evolab.world.physics2d.episode import Trial

            trial = Trial(trial.seed, 0)
        ep = Episode(spec, body, trial)
    except (EpisodeBuildError, ValueError):
        return []
    joined = {frozenset((j.a, j.b)) for j in env.joints}
    index = {b.id: i for i, b in enumerate(env.bodies)}
    seen: set[frozenset[str]] = set()
    out: list[Issue] = []
    for shape, (bid, _tags) in ep._shape_info.items():
        for hit in ep.space.shape_query(shape):
            other = ep._shape_info.get(hit.shape)
            if other is None:
                continue
            pair = frozenset((bid, other[0]))
            if len(pair) < 2 or pair in seen or pair in joined:
                continue
            if hit.contact_point_set.points and min(p.distance for p in hit.contact_point_set.points) < -0.05:
                seen.add(pair)
                a, b = sorted(pair)
                where = f"bodies[{index[a]}]" if a in index else (f"bodies[{index[b]}]" if b in index else "")
                out.append(Issue("warning", where, S.ED_CHECK_OVERLAP.format(a=a, b=b)))
    return out

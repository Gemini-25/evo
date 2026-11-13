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
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from evolab import paths
from evolab.app import strings as S
from evolab.spec import model as m
from evolab.spec import symbols as sy
from evolab.spec.compiled import EnvFileError, EnvSpec, Issue, check_env
from evolab.spec.io import dump_env, parse_env

JsonPath = tuple[str | int, ...]
Ref = tuple[str, str]
"""(구역 이름, id). 구역: bodies, joints, actuators, sensors, regions, checkpoints, events, generators, randomize,
params, scoring.blocks"""

SECTIONS = ("generators", "bodies", "joints", "actuators", "sensors", "regions", "checkpoints", "events",
            "randomize", "params")
CLIPBOARD_FORMAT = "evolab-components"
_ID_RE = re.compile(str(get_args(m.Id)[1].pattern))
"""부품 id 규칙: 환경 파일 모델과 같다(한글 허용)."""
_ID_MAX = 40
_EVOLVE_RE = re.compile(r"=\s*p\.([A-Za-z_\uac00-\ud7a3][A-Za-z0-9_\uac00-\ud7a3]*)\s*")
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


def _clean(x: float) -> float:
    """파일에 적을 좌표: 부동소수 잡음을 지운다(tools.clean과 같다)."""
    return round(float(x), 6) + 0.0


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
    tool_request = Signal(str, object)
    """화면 부품(속성 패널의 「캔버스에서 길 그리기」 같은 단추)이 캔버스 도구를 켜 달라고 한다: (도구 id, 부품 Ref).
    문서 내용은 바꾸지 않는다. 캔버스가 받는다."""
    fix_request = Signal(str)
    """「고치기」가 화면을 열어 달라고 한다(6.2.0, fixes.OPEN_*: 몸 틀 보관함, 규칙 더하기). 문서 내용은 바꾸지 않는다.
    에디터 화면(view.EditorView)이 받는다."""

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
        """지금 내용. 구조가 맞으면 정규 형식(dump_env), 아니면 사전 그대로(저장은 하지 않는다: save)."""
        try:
            return dump_env(parse_env(self.data))
        except EnvFileError:
            return json.dumps(self.data, ensure_ascii=False, indent=2) + "\n"

    def save(self, path: Path | None = None) -> Path:
        """파일로 쓴다(정규 형식). 구조가 잘못된 내용이면 쓰지 않고 EnvFileError를 낸다.

        구조 오류(모르는 종류, 범위 밖 숫자 ...)가 있는 파일은 다시 열 수 없다(load_text가 거절한다). 그런 파일을
        쓰거나 멀쩡한 파일을 덮어쓰지 않으려고 저장을 거절한다. 참조·수식 오류만 있으면 저장하고 다시 열 수 있다.
        """
        target = path or self.path
        if target is None:
            raise ValueError("no path")
        text = dump_env(parse_env(self.data))
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
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
        """base(또는 base_1, base_2 ...) 가운데 아직 안 쓴 id. 모델 규칙에 맞고 예약된 이름·밑줄 두 개를 피한다."""
        base = base[:_ID_MAX] if _ID_RE.match(base) and "__" not in base else "part"
        used = self.all_ids() | (taken or set()) | sy.RESERVED
        if base not in used:
            return base
        stem = re.sub(r"_?\d+$", "", base) or "part"
        k = 1
        while True:
            suffix = f"_{k}"
            cand = (stem[: _ID_MAX - len(suffix)].rstrip("_") or "part") + suffix
            if cand not in used:
                return cand
            k += 1

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
        """id를 바꾸고, 이 id를 가리키는 참조 칸과 수식(p.*, r.*, gen.*, 센서 채널 이름 포함)도 함께 바꾼다.

        종류(kind)·소유(owner) 같은 고정값 칸과 태그는 건드리지 않는다. 구현은 속성 패널과 같다(forms.rename).
        """
        from evolab.app.editor.forms import rename

        return rename(self, ref, new_id)

    # ------------------------------------------------------------------
    # 진화 대상
    # ------------------------------------------------------------------
    def evolve_param(self, path: JsonPath) -> str | None:
        """이 칸이 진화 대상 매개변수 하나("=p.<id>")를 그대로 가리키면 그 id."""
        value = self.get(path)
        if isinstance(value, str):
            mt = _EVOLVE_RE.fullmatch(value)
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

        from evolab.app.editor.forms import param_uses

        def mutate(d: dict[str, Any]) -> None:
            params = d.get("params", [])
            value = next(p.get("value", 0.0) for p in params if p.get("id") == pid)
            set_path(d, path, value)
            # 이름 단위로 센다: 'p.hull_w'는 'p.hull_w_1' 안에 들어 있지 않다(글자 포함 검사는 안 쓴 유전자를 남겼다).
            if param_uses(d, pid) == 0:
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
        """복사한 부품을 붙인다. id가 겹치면 바꾸고, 붙인 부품끼리의 참조는 새 id로 이어 준다. 위치는 조금 옮긴다.

        - 참조 칸과 수식만 고친다(forms.id_renamer, 이름 바꾸기와 같은 규칙). 종류(kind)·태그 같은 고정값 칸은 id와
          글자가 같아도 그대로 둔다(센서 'fuel'의 종류 'fuel', 물체 'hull'의 태그 'hull').
        - 관절로 놓인 물체(place_by)를 그 관절 없이 붙이면 place_by를 지우고, 원본이 놓인 자리(+ 조금 옮김)를
          pos·angle로 적는다. 관절은 원본 물체를 붙잡고 있으므로 사본을 놓을 수 없다.
        - 단, 문서의 그 관절이 붙인 물체의 id를 붙잡고 있으면(복사 → 원본 지우기 → 붙여넣기로 옮김, 관절만 있는
          다른 문서에 붙임) place_by를 그대로 둔다. 관절이 그 물체를 놓으므로 전과 같은 자리·같은 시뮬레이션이다.
        """
        from evolab.app.editor.forms import id_renamer

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
        new_ids: list[str] = []
        renames: dict[Ref, str] = {}
        for e in entries:
            old = str(e["item"].get("id", "part"))
            new = self.unique_id(old, taken)
            taken.add(new)
            new_ids.append(new)
            renames.setdefault((str(e["section"]), old), new)
        fix = id_renamer(renames)

        def held(joint_id: str, body_id: str) -> bool:
            """문서에 이미 있는 관절 joint_id가 물체 body_id를 붙잡고 있나(그 관절의 b)."""
            joint = self.item(("joints", joint_id))
            return isinstance(joint, dict) and joint.get("b") == body_id

        loose = [e["item"] for e, new_id in zip(entries, new_ids, strict=True)
                 if e["section"] == "bodies" and isinstance(e["item"].get("place_by"), str)
                 and ("joints", e["item"]["place_by"]) not in renames and not held(e["item"]["place_by"], new_id)]
        placed = self._placements(loose) if loose else {}
        new_refs: list[Ref] = []

        def mutate(d: dict[str, Any]) -> None:
            for e, new_id in zip(entries, new_ids, strict=True):
                original = e["item"]
                item = fix(copy.deepcopy(original), None)
                item["id"] = new_id
                pose = None
                if any(original is x for x in loose):
                    item.pop("place_by", None)
                    pose = placed.get(str(original.get("id")))
                pos = item.get("pos")
                if pose is not None:
                    item["pos"] = [_clean(pose[0] + offset[0]), _clean(pose[1] + offset[1])]
                    item["angle"] = _clean(pose[2])
                elif isinstance(pos, list) and len(pos) == 2 and all(isinstance(v, (int, float)) for v in pos):
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

    def _placements(self, bodies: list[dict[str, Any]]) -> dict[str, tuple[float, float, float]]:
        """관절로 놓인 물체들이 지금 문서에서 놓인 자리를 그 물체의 pos·angle 좌표로(에이전트 물체는 시작 자세 기준).

        같은 문서에서 복사한 물체(id와 place_by가 같은 물체가 있다)만. 미리보기 장면(canvas.build_scene)에서 읽는다.
        """
        from evolab.app.editor.canvas import build_scene
        from evolab.app.editor.tools import to_local

        mine = [b for b in bodies if (self.item(("bodies", str(b.get("id")))) or {}).get("place_by") == b.get("place_by")]
        if not mine:
            return {}
        scene = build_scene(self)
        out: dict[str, tuple[float, float, float]] = {}
        for b in mine:
            bid = str(b.get("id"))
            shown = scene.body(bid)
            if shown is None:
                continue
            x, y, angle = shown.pose
            if b.get("owner", "agent") == "agent":
                (x, y), angle = to_local(scene.spawn, (x, y)), angle - scene.spawn[2]
            out[bid] = (x, y, angle)
        return out

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


EDITOR_LINTS: tuple[str, ...] = ("ground_is_body", "score_far_is_good", "no_keys", "motor_twice", "joint_apart",
                                  "never_stops")
"""켜는 뜻 검사(설계 D40의 여섯). 내장 것에서 뜨는 코드는 여기서 뺀다(되돌릴 지점 B5)."""
EDITOR_CODES: tuple[str, ...] = ("unattached", "overlap", *EDITOR_LINTS)
"""에디터만 하는 검사의 코드 전부(Issue.code). 엔진 검사의 코드는 spec/compiled.py."""
IGNORABLE: frozenset[str] = frozenset({"ground_is_body", "score_far_is_good", "no_keys"})
"""「괜찮습니다」로 숨길 수 있는 살펴볼 것(어림이라 틀려도 되는 것)."""
IGNORED_KEY = "ignored"
"""editor 사전에서 숨긴 살펴볼 것의 키(해시 밖): ["<code>@<where>", …]."""
APART_GAP = 0.01
"""관절의 두 앵커가 이만큼(m) 넘게 떨어져 있으면 「어긋난 관절」."""
GROUND_RATIO = 4.0
"""에이전트 물체 하나가 나머지 몸 전체보다 이 배수 넘게 넓고 맨 아래면 「땅이 몸이 됨」."""
KEYED_KINDS = ("motor", "thruster", "jump", "move")
"""키가 없으면 「키 없는 구동기」로 세는 구동기 종류(D40)."""


def ignore_key(issue: Issue) -> str:
    return f"{issue.code}@{issue.where}"


def ignored(data: Any) -> set[str]:
    """editor.ignored에 적힌 숨긴 살펴볼 것. 모양이 틀리면 빈 집합."""
    editor = data.get("editor") if isinstance(data, dict) else None
    rows = editor.get(IGNORED_KEY) if isinstance(editor, dict) else None
    return {r for r in rows if isinstance(r, str)} if isinstance(rows, list) else set()


def _label(item: Any) -> str:
    return str(getattr(item, "label", "") or getattr(item, "id", ""))


def _number(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def editor_checks(env: m.EnvFile, spec: EnvSpec | None) -> list[Issue]:
    """에디터만 하는 검사: 관절로 안 이어진 에이전트 물체, 시작할 때 겹친 물체(경고)와 뜻 검사 여섯(EDITOR_LINTS, D40).

    모든 항목에 코드가 있다(EDITOR_CODES). editor.ignored에 적힌 살펴볼 것은 빼고 돌려준다.
    """
    out: list[Issue] = []
    agent = [i for i, b in enumerate(env.bodies) if b.owner == "agent"]
    if len(agent) > 1:
        linked: set[str] = set()
        for j in env.joints:
            linked |= {j.a, j.b}
        for i in agent:
            if env.bodies[i].id not in linked:
                out.append(Issue("warning", f"bodies[{i}]", S.ED_CHECK_UNATTACHED.format(id=env.bodies[i].id),
                                 "unattached"))
    out += _lint_file(env)
    if spec is not None:
        ep = _start_episode(spec)
        if ep is not None:
            out += _overlap_checks(env, ep)
            out += _lint_scene(env, ep)
    hidden = ignored(env.editor)
    return [i for i in out if not (i.code in IGNORABLE and ignore_key(i) in hidden)]


def _lint_file(env: m.EnvFile) -> list[Issue]:
    """파일만 보고 하는 뜻 검사: 거리 점수의 부호, 키 없는 구동기, 한 관절에 모터 둘, 멈추지 않는 물체."""
    from evolab.presets.josa import fill
    from evolab.presets.rules import display_name

    out: list[Issue] = []
    data = {"bodies": [b.model_dump(by_alias=True) for b in env.bodies],
            "regions": [r.model_dump(by_alias=True) for r in env.regions]}
    if "score_far_is_good" in EDITOR_LINTS and env.scoring is not None:
        for k, b in enumerate(env.scoring.blocks):
            if isinstance(b, m.DistanceBlock) and b.weight > 0:
                out.append(Issue("info", f"scoring.blocks[{k}]", fill(S.ED_CHECK_SCORE_FAR, {
                    "thing": display_name(data, b.body), "where": display_name(data, b.target)}), "score_far_is_good"))
    if "no_keys" in EDITOR_LINTS:
        keyless = [k for k, a in enumerate(env.actuators) if a.kind in KEYED_KINDS and not _keys_of(a)]
        if keyless:
            where = f"actuators[{keyless[0]}]" if len(keyless) == 1 else "actuators"
            out.append(Issue("info", where, S.ED_CHECK_NO_KEYS.format(n=len(keyless)), "no_keys"))
    if "motor_twice" in EDITOR_LINTS:
        seen: dict[str, int] = {}
        joints = {j.id: j for j in env.joints}
        for k, a in enumerate(env.actuators):
            if isinstance(a, m.Motor):
                seen[a.joint] = seen.get(a.joint, 0) + 1
                n = sum(1 for x in env.actuators if isinstance(x, m.Motor) and x.joint == a.joint)
                if seen[a.joint] == 2:
                    joint = joints.get(a.joint)
                    out.append(Issue("warning", f"actuators[{k}]", S.ED_CHECK_MOTOR_TWICE.format(
                        joint=_label(joint) if joint is not None else a.joint, n=n), "motor_twice"))
    if "never_stops" in EDITOR_LINTS and env.world.view != "top" and env.world.damping >= 1.0:
        g = tuple(_number(v) for v in env.world.gravity)
        if g == (0.0, 0.0):
            for k, b in enumerate(env.bodies):
                if b.owner == "world" and b.type == "dynamic" and _number(b.drag) == 0.0:
                    out.append(Issue("warning", f"bodies[{k}]", fill(S.ED_CHECK_NEVER_STOPS, {"name": _label(b)}),
                                     "never_stops"))
    return out


def _keys_of(actuator: Any) -> list[str]:
    keys: list[str] = []
    for name in ("keys", "keys_pos", "keys_neg", "keys_up", "keys_down", "keys_left", "keys_right"):
        keys += list(getattr(actuator, name, None) or [])
    return keys


def _start_episode(spec: EnvSpec) -> Any:
    """미리보기 시험으로 시작 자세의 세계를 만든다(겹침·어긋난 관절·땅이 된 몸 검사가 함께 씀). 못 만들면 None."""
    from evolab.engine import seeds
    from evolab.policy.genome import layout_for
    from evolab.world.physics2d.episode import Episode, EpisodeBuildError, Trial

    try:
        layout = layout_for(spec)
        body = layout.body_values(layout.initial()[None, :])[0] if layout.n_body else None
        trial = seeds.display_trial()
        if spec.track_set_size:
            trial = Trial(trial.seed, 0)
        return Episode(spec, body, trial)
    except (EpisodeBuildError, ValueError):
        return None


def _lint_scene(env: m.EnvFile, ep: Any) -> list[Issue]:
    """시작 자세의 세계를 보고 하는 뜻 검사: 어긋난 관절, 땅처럼 보이는 에이전트 물체."""
    from evolab.presets.josa import fill

    out: list[Issue] = []
    if "joint_apart" in EDITOR_LINTS:
        for k, j in enumerate(env.joints):
            if j.kind not in ("pivot", "rot_spring", "weld") or j.a not in ep.bodies or j.b not in ep.bodies:
                continue
            gap = _anchor_gap(ep, j)
            if gap is not None and gap > APART_GAP:
                out.append(Issue("warning", f"joints[{k}]", S.ED_CHECK_JOINT_APART.format(joint=_label(j), gap=gap),
                                 "joint_apart"))
    if "ground_is_body" in EDITOR_LINTS:
        boxes: dict[str, tuple[float, float, float, float]] = {}
        for shape, (bid, _tags) in ep._shape_info.items():
            bb = shape.cache_bb()
            old = boxes.get(bid)
            box = (bb.left, bb.bottom, bb.right, bb.top)
            boxes[bid] = box if old is None else (min(old[0], box[0]), min(old[1], box[1]), max(old[2], box[2]),
                                                   max(old[3], box[3]))
        agent = [(k, b) for k, b in enumerate(env.bodies) if b.owner == "agent" and b.id in boxes]
        if len(agent) > 1:
            for k, b in agent:
                others = [boxes[o.id] for _k, o in agent if o.id != b.id]
                width = max(x[2] for x in others) - min(x[0] for x in others)
                bottom = min(x[1] for x in others)
                mine = boxes[b.id]
                if mine[2] - mine[0] > GROUND_RATIO * max(width, 1e-6) and mine[3] <= bottom + 0.05:
                    out.append(Issue("info", f"bodies[{k}]", fill(S.ED_CHECK_GROUND_IS_BODY, {"name": _label(b)}),
                                     "ground_is_body"))
    return out


def _anchor_gap(ep: Any, j: m.Joint) -> float | None:
    """시작할 때 관절의 두 앵커 사이 거리(m). 앵커가 수식이면 만든 값으로."""
    try:
        st = sy.joint_stage(ep.env, j)
        aa, ab = ep._vec(j.anchor_a, st), ep._vec(j.anchor_b, st)
    except Exception:  # noqa: BLE001 (계산 못 하는 앵커는 검사하지 않는다)
        return None
    wa = ep.bodies[j.a].local_to_world(aa)
    wb = ep.bodies[j.b].local_to_world(ab)
    return float((wa - wb).length)


def _overlap_checks(env: m.EnvFile, ep: Any) -> list[Issue]:
    """시작 자세에서 서로 부딪히는 물체 쌍을 찾는다(관절로 이은 쌍과 에이전트끼리 제외).

    v2: 정해진 움직임(kinematic) 물체와 움직이지 않는 물체(kinematic·static)의 쌍은 뺀다. 물리 엔진에서 둘 다 질량이
    무한이라 서로 부딪히지 않는다(설계 D26). 장애물 줄(생성기 물체)도 kinematic이라 같다.
    """
    import pymunk

    joined = {frozenset((j.a, j.b)) for j in env.joints}
    index = {b.id: i for i, b in enumerate(env.bodies)}
    seen: set[frozenset[str]] = set()
    out: list[Issue] = []
    kinematic, dynamic = pymunk.Body.KINEMATIC, pymunk.Body.DYNAMIC
    for shape, (bid, _tags) in ep._shape_info.items():
        for hit in ep.space.shape_query(shape):
            other = ep._shape_info.get(hit.shape)
            if other is None:
                continue
            pair = frozenset((bid, other[0]))
            if len(pair) < 2 or pair in seen or pair in joined:
                continue
            types = (shape.body.body_type, hit.shape.body.body_type)
            if kinematic in types and dynamic not in types:
                continue
            if hit.contact_point_set.points and min(p.distance for p in hit.contact_point_set.points) < -0.05:
                seen.add(pair)
                a, b = sorted(pair)
                where = f"bodies[{index[a]}]" if a in index else (f"bodies[{index[b]}]" if b in index else "")
                out.append(Issue("warning", where, S.ED_CHECK_OVERLAP.format(a=a, b=b), "overlap"))
    return out

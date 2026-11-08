"""v2.0 0단계: 환경 파일 형식 2 (설계 2.1·2.2·4.1·D25, 테스트 T1~T8). v2.1(0'단계)에서 형식 3(물체의 있음 칸)을 더함.

- 저장(dump_env)과 해시(env_hash)는 내용을 담을 수 있는 가장 낮은 형식으로 한다. v1 기능만 쓴 파일은 v1.2 코드가 쓴
  글·해시와 바이트까지 같다(tests/data/v1_dump, tests/data/make_v1_reference.py로 v1.2 코드에서 떠 둠). 형식 2 파일은
  v2.0 코드가 쓴 글·해시와 같다(tests/data/v2_dump, tests/data/make_v2_reference.py로 v2.0 코드에서 떠 둠).
- 모든 모델 칸 = v1 칸 목록(tests/data/schema_fields_v1.json) + 형식 표(FIELD_SINCE, KIND_SINCE)에 등록한 칸.
- 움직임은 시간만의 함수(pose_at), 기본 보기(editor.view)는 해시를 바꾸지 않는다.
"""

from __future__ import annotations

import copy
import json
import math
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from evolab.app import strings as S
from evolab.spec import model as m
from evolab.spec.compiled import EnvFileError
from evolab.spec.editor_meta import editor_view_box, view_box, with_view_box
from evolab.spec.io import (
    FIELD_SINCE,
    KIND_SINCE,
    MIGRATIONS,
    VALUE_SINCE,
    canonical_dict,
    dump_env,
    env_hash,
    feature_version,
    migrate,
    parse_env,
)
from evolab.world.physics2d.motion import MotionPlan, path_length, period_of, pose_at

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
DUMP = DATA / "v1_dump"
DUMP_V2 = DATA / "v2_dump"


def _v1_files() -> list[tuple[str, str, Path]]:
    """(이름, v1.2 코드의 env_hash, 원본 파일). tests/data/v1_dump/env_hash.txt에서."""
    out = []
    for line in (DUMP / "env_hash.txt").read_text(encoding="utf-8").splitlines():
        stem, digest, rel = line.split()
        out.append((stem, digest, ROOT / rel))
    return out


V1_FILES = _v1_files()


def model_field_names(root: type) -> set[str]:
    """root 모델에서 닿는 모든 모델의 칸 "모델.칸"(별칭 포함). tests/data/make_v1_reference.py와 같은 규칙."""
    seen: set[type] = set()
    out: set[str] = set()

    def visit(tp: object) -> None:
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            if tp in seen:
                return
            seen.add(tp)
            for name, f in tp.model_fields.items():
                out.add(f"{tp.__name__}.{f.alias or name}")
                visit(f.annotation)
            return
        for arg in typing.get_args(tp):
            visit(arg)

    visit(root)
    return out


# ---------------------------------------------------------------------------
# T1, T2: v1 파일의 해시와 글
# ---------------------------------------------------------------------------


def test_v1_reference_files_are_all_present() -> None:
    stems = {stem for stem, _, _ in V1_FILES}
    assert {"lander", "track_car", "hill_car", "v10_run_ga", "v11_run_restart"} <= stems
    assert {p.stem for p in (ROOT / "environments").glob("*.evoenv") if p.stem in ("lander", "track_car", "hill_car")} \
        == {"lander", "track_car", "hill_car"}
    for stem, _, src in V1_FILES:
        assert src.is_file(), src
        assert (DUMP / f"{stem}.json").is_file()


@pytest.mark.parametrize(("stem", "digest", "src"), V1_FILES, ids=[f[0] for f in V1_FILES])
def test_t1_v1_env_hash_is_unchanged(stem: str, digest: str, src: Path) -> None:
    env = parse_env(src.read_bytes())
    assert env_hash(env) == digest
    run_dir = src.parent.parent
    if src.parent.name == "envs":
        champions = sorted((run_dir / "champions").glob("*.npz"))
        assert champions
        for path in champions:
            with np.load(path, allow_pickle=False) as d:
                assert str(d["env_version"]) == env_hash(env)[:16], path.name


@pytest.mark.parametrize(("stem", "digest", "src"), V1_FILES, ids=[f[0] for f in V1_FILES])
def test_t2_v1_dump_text_is_byte_identical(stem: str, digest: str, src: Path) -> None:
    env = parse_env(src.read_bytes())
    assert feature_version(env) == 1
    text = dump_env(env)
    assert text.encode("utf-8") == (DUMP / f"{stem}.json").read_bytes()
    data = json.loads(text)
    assert data["version"] == 1
    for body in data["bodies"]:
        assert "motion" not in body
        assert all("density" not in sh for sh in body["shapes"])
    # 다시 읽어도 같은 글(v1 글은 읽을 때 지금 형식으로 변환되지만 쓸 때는 다시 1)
    again = parse_env(text)
    assert again.version == m.CURRENT_VERSION
    assert dump_env(again) == text and env_hash(again) == digest


# ---------------------------------------------------------------------------
# T3, T4: v2 칸·종류·값을 하나씩 켜고 끄기
# ---------------------------------------------------------------------------

BASE: dict[str, Any] = {
    "version": 1,
    "name": "format test",
    "bodies": [
        {"id": "ground", "owner": "world", "type": "static", "tags": ["ground"],
         "shapes": [{"kind": "box", "w": 20.0, "h": 1.0}]},
        {"id": "lift", "owner": "world", "type": "static", "pos": [5.0, 1.0],
         "shapes": [{"kind": "box", "w": 2.0, "h": 0.2}]},
        {"id": "hull", "pos": [0.0, 2.0],
         "shapes": [{"kind": "box", "w": 1.0, "h": 1.0}, {"kind": "circle", "r": 0.3, "offset": [0.0, 0.6]},
                    {"kind": "poly", "points": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]},
                    {"kind": "radial", "radii": [0.5, 0.6, 0.7]}]},
    ],
    "actuators": [{"kind": "thruster", "id": "main", "body": "hull", "max_force": 100.0, "keys": ["Up"]}],
    "sensors": [{"kind": "time_left", "id": "tl"}],
}
LIFT, HULL = 1, 2


def _set(path: tuple[Any, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def apply(d: dict[str, Any]) -> None:
        cur = d
        for k in path[:-1]:
            cur = cur[k]
        cur[path[-1]] = value
    return apply


def _append(section: str, item: dict[str, Any]) -> Callable[[dict[str, Any]], None]:
    def apply(d: dict[str, Any]) -> None:
        d.setdefault(section, []).append(copy.deepcopy(item))
    return apply


def _pop_last(section: str) -> Callable[[dict[str, Any]], None]:
    def apply(d: dict[str, Any]) -> None:
        d[section].pop()
    return apply


def _append_shape(item: dict[str, Any]) -> Callable[[dict[str, Any]], None]:
    def apply(d: dict[str, Any]) -> None:
        d["bodies"][HULL]["shapes"].append(copy.deepcopy(item))
    return apply


def _pop_shape(d: dict[str, Any]) -> None:
    d["bodies"][HULL]["shapes"].pop()


# 있음 칸(형식 3)은 뿌리가 아닌 에이전트 물체에만 쓴다(뿌리 hull에 쓰면 검사 오류): 관절로 붙은 바퀴를 더해 켜고 지워 끈다.
POD = {"id": "pod", "pos": [0.0, 1.2], "exists": "=p.pod_on", "shapes": [{"kind": "circle", "r": 0.3}]}
POD_JOINT = {"id": "pod_axle", "kind": "pivot", "a": "hull", "b": "pod", "anchor_a": [0.0, -0.8]}


def _add_pod(d: dict[str, Any]) -> None:
    d["bodies"].append(copy.deepcopy(POD))
    d.setdefault("joints", []).append(copy.deepcopy(POD_JOINT))


def _remove_pod(d: dict[str, Any]) -> None:
    d["bodies"].pop()
    d["joints"].pop()


PATH = {"kind": "path", "points": [[0.0, 0.0], [0.0, 3.0]], "speed": 1.0, "mode": "pingpong"}
WAVE = {"kind": "wave", "amplitude": [0.0, 0.5], "period": 2.0}

# 표 항목 -> (켜기, 끄기). 끄기는 칸을 기본값으로 되돌리거나 더한 부품을 지운다.
CASES: dict[tuple[Any, ...], tuple[Callable[[dict[str, Any]], None], Callable[[dict[str, Any]], None]]] = {
    (m.Box, "density"): (_set(("bodies", HULL, "shapes", 0, "density"), 30.0),
                         _set(("bodies", HULL, "shapes", 0, "density"), None)),
    (m.Circle, "density"): (_set(("bodies", HULL, "shapes", 1, "density"), "=2 * 60"),
                            _set(("bodies", HULL, "shapes", 1, "density"), None)),
    (m.Poly, "density"): (_set(("bodies", HULL, "shapes", 2, "density"), 0.0),
                          _set(("bodies", HULL, "shapes", 2, "density"), None)),
    (m.Radial, "density"): (_set(("bodies", HULL, "shapes", 3, "density"), 2.5),
                            _set(("bodies", HULL, "shapes", 3, "density"), None)),
    (m.Body, "motion"): (_set(("bodies", LIFT, "motion"), PATH), _set(("bodies", LIFT, "motion"), None)),
    (m.Body, "type", "kinematic"): (_set(("bodies", LIFT, "type"), "kinematic"),
                                    _set(("bodies", LIFT, "type"), "static")),
    (m.Wing,): (_append_shape({"kind": "wing", "le": [0.5, 0.1], "chord": 1.2, "camber": 0.03}), _pop_shape),
    (m.Jump,): (_append("actuators", {"kind": "jump", "id": "jump", "body": "hull", "keys": ["Space"]}),
                _pop_last("actuators")),
    (m.Aero,): (_append("actuators", {"kind": "aero", "id": "air", "body": "hull", "surfaces": "wings"}),
                _pop_last("actuators")),
    (m.ObstacleGen,): (_append("generators", {"kind": "obstacles", "id": "obs", "seed": "=r.oseed", "count": 40}),
                       _pop_last("generators")),
    (m.PathMotion,): (_set(("bodies", LIFT, "motion"), PATH), _set(("bodies", LIFT, "motion"), None)),
    (m.WaveMotion,): (_set(("bodies", LIFT, "motion"), WAVE), _set(("bodies", LIFT, "motion"), None)),
    (m.Body, "exists"): (_add_pod, _remove_pod),
}


def _registered() -> dict[tuple[Any, ...], int]:
    out: dict[tuple[Any, ...], int] = {}
    out.update({(c, f): v for (c, f), v in FIELD_SINCE.items()})
    out.update({(c,): v for c, v in KIND_SINCE.items()})
    out.update({(c, f, val): v for (c, f, val), v in VALUE_SINCE.items()})
    return out


def _case_id(key: tuple[Any, ...]) -> str:
    return ".".join([key[0].__name__, *map(str, key[1:])])


def test_every_registered_feature_has_a_t3_case() -> None:
    """형식 표에 칸·종류·값을 등록하면 아래 T3·T4 경우도 더해야 한다(v2.1의 exists 포함)."""
    assert set(CASES) == set(_registered())


def _base() -> dict[str, Any]:
    return copy.deepcopy(BASE)


def _shape_keys(data: dict[str, Any]) -> list[tuple[str, set[str]]]:
    return [(sh["kind"], set(sh)) for b in data["bodies"] for sh in b["shapes"]]


@pytest.mark.parametrize("key", list(CASES), ids=_case_id)
def test_t3_one_v2_feature_makes_a_version_2_file(key: tuple[Any, ...]) -> None:
    since = _registered()[key]
    base_env = parse_env(_base())
    assert feature_version(base_env) == 1
    data = _base()
    CASES[key][0](data)
    env = parse_env(data)
    assert feature_version(env) == since
    text = dump_env(env)
    saved = json.loads(text)
    assert saved["version"] == since
    assert canonical_dict(env)["version"] == since
    # 형식 2 글에는 형식 2까지의 칸이 모두(기본값이어도) 있고, 그보다 새 칸은 없다.
    for body in saved["bodies"]:
        assert ("motion" in body) == (FIELD_SINCE[(m.Body, "motion")] <= since)
    for kind, keys in _shape_keys(saved):
        cls = {"box": m.Box, "circle": m.Circle, "poly": m.Poly, "radial": m.Radial}.get(kind)
        if cls is not None:
            assert ("density" in keys) == (FIELD_SINCE[(cls, "density")] <= since)
    for (cls, name), v in FIELD_SINCE.items():
        if v > since:
            assert f'"{cls.model_fields[name].alias or name}"' not in text
    # 다시 읽으면 같은 모델, 다시 쓰면 같은 글
    again = parse_env(text)
    assert again == env
    assert dump_env(again) == text
    assert env_hash(again) == env_hash(env) != env_hash(base_env)


@pytest.mark.parametrize("key", list(CASES), ids=_case_id)
def test_t4_turning_a_feature_back_off_restores_text_and_hash(key: tuple[Any, ...]) -> None:
    on, off = CASES[key]
    t0 = dump_env(parse_env(_base()))
    h0 = env_hash(parse_env(_base()))
    # 에디터처럼 정규 글의 사전에서 고친다
    data = json.loads(t0)
    on(data)
    t1 = dump_env(parse_env(data))
    assert t1 != t0 and json.loads(t1)["version"] == _registered()[key]
    data = json.loads(t1)
    off(data)
    assert dump_env(parse_env(data)) == t0
    assert env_hash(parse_env(data)) == h0


def test_t4_removing_a_new_field_key_is_the_same_as_its_default() -> None:
    data = _base()
    data["bodies"][HULL]["shapes"][0]["density"] = None
    data["bodies"][LIFT]["motion"] = None
    assert dump_env(parse_env(data)) == dump_env(parse_env(_base()))
    assert env_hash(parse_env(data)) == env_hash(parse_env(_base()))


def test_t4_editor_undo_redo_round_trip(qtbot: Any) -> None:
    from evolab.app.editor.document import EnvDocument

    doc = EnvDocument()
    doc.from_builtin("lander.evoenv")
    t0, h0 = doc.text(), env_hash(parse_env(doc.data))
    assert t0.encode("utf-8") == (DUMP / "lander.json").read_bytes()
    i = doc.find(("bodies", "hull"))
    path = ("bodies", i, "shapes", 0, "density")
    doc.set_value(path, 120.0)
    t1 = doc.text()
    assert json.loads(t1)["version"] == 2 and env_hash(parse_env(doc.data)) != h0
    doc.undo_stack.undo()
    assert doc.text() == t0 and env_hash(parse_env(doc.data)) == h0
    doc.undo_stack.redo()
    assert doc.text() == t1
    doc.set_value(path, None, merge=False)  # 손으로 기본값(비움)으로 되돌림
    assert doc.text() == t0 and env_hash(parse_env(doc.data)) == h0
    ref = doc.add("actuators", {"kind": "jump", "id": "jump", "body": "hull", "keys": ["Space"]})
    assert ref == ("actuators", "jump") and json.loads(doc.text())["version"] == 2
    doc.undo_stack.undo()
    assert doc.text() == t0


# ---------------------------------------------------------------------------
# T5: 등록 빠짐 막기
# ---------------------------------------------------------------------------


def test_t5_all_model_fields_are_v1_fields_or_registered() -> None:
    v1 = set(json.loads((DATA / "schema_fields_v1.json").read_text(encoding="utf-8"))["fields"])
    registered = {f"{c.__name__}.{c.model_fields[name].alias or name}" for (c, name) in FIELD_SINCE}
    kinds = {f"{c.__name__}.{f.alias or name}" for c in KIND_SINCE for name, f in c.model_fields.items()}
    now = model_field_names(m.EnvFile)
    assert now == v1 | registered | kinds, (
        f"등록 안 한 칸: {sorted(now - v1 - registered - kinds)}, 없어진 칸: {sorted((v1 | registered | kinds) - now)}"
    )
    # 등록한 칸은 새 칸이고 기본값이 있다(옛 파일이 그대로 읽힘). 종류는 새 모델이다.
    assert not registered & v1
    assert not {name.split(".")[0] for name in kinds} & {name.split(".")[0] for name in v1}
    for (c, name), since in FIELD_SINCE.items():
        assert name in c.model_fields and not c.model_fields[name].is_required(), (c, name)
        assert 2 <= since <= m.CURRENT_VERSION
    for c, since in KIND_SINCE.items():
        assert 2 <= since <= m.CURRENT_VERSION and "kind" in c.model_fields
    for (c, name, value), since in VALUE_SINCE.items():
        assert value in typing.get_args(c.model_fields[name].annotation)
        assert 2 <= since <= m.CURRENT_VERSION
        assert f"{c.__name__}.{name}" in v1  # 옛 칸의 새 값


def test_t5_new_aliased_fields_are_counted_by_alias() -> None:
    assert "ContactSensor.with" in model_field_names(m.EnvFile)
    assert "ContactTimeBlock.with" in model_field_names(m.EnvFile)


# ---------------------------------------------------------------------------
# T6: 형식 번호와 변환
# ---------------------------------------------------------------------------


def test_t6_newer_version_is_refused_with_the_update_message() -> None:
    data = _base()
    data["version"] = m.CURRENT_VERSION + 1
    with pytest.raises(EnvFileError) as exc:
        parse_env(data)
    assert exc.value.issues[0].message == S.SPEC_NEWER_VERSION.format(version=m.CURRENT_VERSION + 1)
    assert "(파일 버전 4)" in exc.value.issues[0].message


def test_t6_older_versions_are_migrated_without_changing_the_dict() -> None:
    assert m.CURRENT_VERSION == 3
    assert sorted(MIGRATIONS) == list(range(1, m.CURRENT_VERSION))
    original = json.loads((DUMP / "lander.json").read_text(encoding="utf-8"))
    for v, step in MIGRATIONS.items():
        d = copy.deepcopy(original)
        d["version"] = v
        assert step(copy.deepcopy(d)) == d
    before = copy.deepcopy(original)
    migrated = migrate(copy.deepcopy(original))
    assert migrated["version"] == m.CURRENT_VERSION
    assert {k: v for k, v in migrated.items() if k != "version"} == {k: v for k, v in before.items() if k != "version"}
    env = parse_env(original)
    assert original == before  # 읽기가 호출한 쪽의 사전을 고치지 않음
    assert env.version == m.CURRENT_VERSION and feature_version(env) == 1


def test_t6_version_missing_or_bad() -> None:
    data = _base()
    del data["version"]
    assert parse_env(data).version == m.CURRENT_VERSION
    for bad in (0, "2", 1.5):
        data["version"] = bad
        with pytest.raises(EnvFileError) as exc:
            parse_env(data)
        assert exc.value.issues[0].message == S.SPEC_BAD_VERSION.format(version=bad)


def test_t6_v2_features_in_a_file_labelled_version_1_are_saved_as_version_2() -> None:
    data = _base()
    data["actuators"].append({"kind": "jump", "id": "jump", "body": "hull"})
    env = parse_env(data)
    assert json.loads(dump_env(env))["version"] == 2


def _v2_files() -> list[tuple[str, str, Path]]:
    """(이름, v2.0 코드의 env_hash, 원본 파일). tests/data/v2_dump/env_hash.txt에서."""
    out = []
    for line in (DUMP_V2 / "env_hash.txt").read_text(encoding="utf-8").splitlines():
        stem, digest, rel = line.split()
        out.append((stem, digest, ROOT / rel))
    return out


V2_FILES = _v2_files()


@pytest.mark.parametrize(("stem", "digest", "src"), V2_FILES, ids=[f[0] for f in V2_FILES])
def test_t3_v2_files_keep_the_v20_text_and_hash(stem: str, digest: str, src: Path) -> None:
    """T3 (v2.1 추가): 형식 3 칸(exists)이 기본값인 형식 2 파일의 글·해시가 v2.0 코드가 쓴 것과 바이트까지 같다."""
    assert {s for s, _, _ in V2_FILES} == {"jump", "glider", "all_v2_features"}
    env = parse_env(src.read_bytes())
    assert feature_version(env) == 2
    text = dump_env(env)
    assert text.encode("utf-8") == (DUMP_V2 / f"{stem}.json").read_bytes()
    assert env_hash(env) == digest
    assert json.loads(text)["version"] == 2 and '"exists"' not in text
    again = parse_env(text)
    assert dump_env(again) == text and env_hash(again) == digest


def test_motion_error_path_hides_the_kind_tag() -> None:
    data = _base()
    data["bodies"][LIFT]["motion"] = {"kind": "path", "points": [[0.0, 0.0]]}
    with pytest.raises(EnvFileError) as exc:
        parse_env(data)
    assert [i.where for i in exc.value.issues] == ["bodies[1].motion.points"]
    data["bodies"][HULL]["shapes"].append({"kind": "wing", "chord": "x"})
    data["bodies"][LIFT]["motion"] = None
    with pytest.raises(EnvFileError) as exc:
        parse_env(data)
    assert [i.where for i in exc.value.issues] == ["bodies[2].shapes[4].chord"]
    assert exc.value.issues[0].message == S.SPEC_ERR_NUM_OR_FORMULA


# ---------------------------------------------------------------------------
# T7: 움직임 = 시간만의 함수
# ---------------------------------------------------------------------------


def _close(a: tuple[float, ...], b: tuple[float, ...], tol: float = 1e-12) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


def test_t7_once_stops_at_the_end_and_start_waits() -> None:
    p = MotionPlan.path((10.0, 1.0), 0.3, [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0)], 2.0, 0.0, 1.5, "once", 0.0)
    assert p.points == ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0)) and p.cum == (0.0, 4.0, 7.0)
    assert _close(pose_at(p, 0.0), (10.0, 1.0, 0.3))
    assert _close(pose_at(p, 1.5), (10.0, 1.0, 0.3))  # 출발 전에는 첫 점
    assert _close(pose_at(p, -5.0), pose_at(p, 0.0))  # 음의 시각은 0
    assert _close(pose_at(p, 2.5), (12.0, 1.0, 0.3))
    assert _close(pose_at(p, 4.0), (14.0, 2.0, 0.3))
    for t in (5.0, 6.0, 100.0):
        assert _close(pose_at(p, t), (14.0, 4.0, 0.3))
    assert path_length(p) == 7.0 and period_of(p) is None


def test_t7_loop_is_a_closed_path() -> None:
    p = MotionPlan.path((0.0, 0.0), 0.0, [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)], 1.0, 0.0, 0.0, "loop", 0.0)
    assert p.points[-1] == p.points[0] and path_length(p) == 12.0  # 끝점 -> 첫 점 구간(5m) 포함
    assert period_of(p) == 12.0
    assert _close(pose_at(p, 9.5), (1.5, 2.0, 0.0))  # 닫는 구간 위
    for t in (0.0, 1.0, 3.5, 7.0, 11.0):
        assert _close(pose_at(p, t + 12.0), pose_at(p, t), 1e-9)
    # 이미 닫힌 길은 점을 더하지 않는다
    q = MotionPlan.path((0.0, 0.0), 0.0, [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)], 1.0, 0.0, 0.0, "loop", 0.0)
    assert len(q.points) == 3 and path_length(q) == 2.0


def test_t7_pingpong_comes_back() -> None:
    p = MotionPlan.path((1.0, 0.0), 0.0, [(0.0, 0.0), (2.0, 0.0)], 1.0, 0.0, 0.0, "pingpong", 0.0)
    xs = [pose_at(p, t)[0] for t in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
    assert xs == [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0]
    assert path_length(p) == 2.0 and period_of(p) == 4.0


def test_t7_negative_accel_stops_and_positive_accel_speeds_up() -> None:
    p = MotionPlan.path((0.0, 0.0), 0.0, [(0.0, 0.0), (100.0, 0.0)], 4.0, -2.0, 0.0, "once", 0.0)
    assert _close(pose_at(p, 1.0), (3.0, 0.0, 0.0))  # 4 - 1
    for t in (2.0, 3.0, 50.0):
        assert _close(pose_at(p, t), (4.0, 0.0, 0.0))  # 2초에 멈춤: 4·2 - 2·4/2
    assert period_of(p) is None
    q = MotionPlan.path((0.0, 0.0), 0.0, [(0.0, 0.0), (100.0, 0.0)], 1.0, 0.5, 2.0, "loop", 0.0)
    assert _close(pose_at(q, 4.0), (3.0, 0.0, 0.0))  # 출발 2초 뒤: 1·2 + 0.5·4/2
    assert period_of(q) is None  # 가속하면 되풀이되지 않음
    r = MotionPlan.path((0.0, 0.0), 0.0, [(0.0, 0.0), (5.0, 0.0)], -3.0, 0.0, -1.0, "once", 0.0)
    assert r.speed == 0.0 and r.start == 0.0 and _close(pose_at(r, 10.0), (0.0, 0.0, 0.0))


def test_t7_wave_phase_and_spin() -> None:
    w = MotionPlan.wave((2.0, 3.0), 0.5, (1.0, 2.0), 4.0, 0.25, 0.1)
    assert _close(pose_at(w, 0.0), (3.0, 5.0, 0.5))  # 위상 1/4 = 사인 꼭대기
    assert _close(pose_at(w, 2.0), (1.0, 1.0, 0.7), 1e-12)
    assert _close(pose_at(w, 1.0), (2.0, 3.0, 0.6), 1e-12)
    assert _close(pose_at(w, 4.0), pose_at(w, 0.0)[:2] + (0.9,), 1e-12)
    assert period_of(w) == 4.0 and path_length(w) == pytest.approx(2.0 * math.sqrt(5.0))
    # 주기가 0 이하로 계산되어도 유한
    z = MotionPlan.wave((0.0, 0.0), 0.0, (1.0, 1.0), 0.0, 0.0, 0.0)
    assert z.period > 0.0 and all(math.isfinite(v) for v in pose_at(z, 1.234))
    # path의 spin은 출발 시각과 무관하게 0초부터
    p = MotionPlan.path((0.0, 0.0), 1.0, [(0.0, 0.0), (1.0, 0.0)], 1.0, 0.0, 5.0, "once", 2.0)
    assert _close(pose_at(p, 1.0), (0.0, 0.0, 3.0))


def test_t7_degenerate_paths_are_finite() -> None:
    cases = [
        MotionPlan.path((1.0, 2.0), 0.0, [(0.0, 0.0), (0.0, 0.0)], 1.0, 0.0, 0.0, mode, 0.0)  # 길이 0
        for mode in ("once", "loop", "pingpong")
    ] + [
        MotionPlan.path((1.0, 2.0), 0.0, [(0.0, 0.0), (3.0, 0.0)], 0.0, 0.0, 0.0, mode, 0.0)  # 속력 0
        for mode in ("once", "loop", "pingpong")
    ] + [
        MotionPlan.path((1.0, 2.0), 0.0, [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (2.0, 0.0)], 1.0, 0.0, 0.0, "pingpong", 0.0),
    ]
    for p in cases[:6]:
        for t in (0.0, 0.5, 1.0, 7.25, 1e6):
            x, y, a = pose_at(p, t)
            assert (x, y, a) == (1.0, 2.0, 0.0)
        assert period_of(p) is None
    dup = cases[6]  # 길이 0인 구간이 끼어 있어도 이어짐
    assert path_length(dup) == 2.0 and period_of(dup) == 4.0
    for t in np.linspace(0.0, 8.0, 161):
        x, y, _ = pose_at(dup, float(t))
        assert math.isfinite(x) and y == 2.0
        expect = (float(t) % 4.0) if (float(t) % 4.0) <= 2.0 else 4.0 - (float(t) % 4.0)
        assert abs(x - (1.0 + expect)) < 1e-9
    with pytest.raises(ValueError):
        MotionPlan.path((0.0, 0.0), 0.0, [(0.0, 0.0), (1.0, 0.0)], 1.0, 0.0, 0.0, "teleport", 0.0)


def test_t7_pose_is_a_pure_function_of_time() -> None:
    p = MotionPlan.path((0.0, 0.0), 0.0, [(0.0, 0.0), (5.0, 1.0), (2.0, 4.0)], 1.3, 0.2, 0.4, "pingpong", 0.7)
    times = [k / 60.0 for k in range(0, 2400, 7)]
    first = [pose_at(p, t) for t in times]
    for t in reversed(times):  # 어떤 순서로 물어도 같은 값
        pose_at(p, t)
    assert [pose_at(p, t) for t in times] == first
    assert hash(p) == hash(MotionPlan.path((0.0, 0.0), 0.0, [(0.0, 0.0), (5.0, 1.0), (2.0, 4.0)], 1.3, 0.2, 0.4,
                                           "pingpong", 0.7))


# ---------------------------------------------------------------------------
# T8: 기본 보기 (editor.view)
# ---------------------------------------------------------------------------


def test_t8_view_box_does_not_change_the_hash_or_version() -> None:
    env = parse_env((DUMP / "lander.json").read_bytes())
    h0, t0 = env_hash(env), dump_env(env)
    assert view_box(env) is None
    editor = with_view_box(env.editor, (-4.0, -1.0, 16.0, 6.0))
    assert editor is not env.editor and "view" not in env.editor
    viewed = env.model_copy(update={"editor": editor})
    assert view_box(viewed) == (-4.0, -1.0, 16.0, 6.0)
    assert env_hash(viewed) == h0
    text = dump_env(viewed)
    assert text != t0 and json.loads(text)["version"] == 1
    assert json.loads(text)["editor"]["view"] == [-4.0, -1.0, 16.0, 6.0]
    again = parse_env(text)
    assert view_box(again) == (-4.0, -1.0, 16.0, 6.0) and env_hash(again) == h0
    cleared = again.model_copy(update={"editor": with_view_box(again.editor, None)})
    assert view_box(cleared) is None and env_hash(cleared) == h0 and dump_env(cleared) == t0
    # 다른 editor 키는 그대로 둔다
    other = with_view_box({"color": "x", "view": [0, 0, 1, 1]}, None)
    assert other == {"color": "x"}
    assert with_view_box(None, (0, 0, 1, 2)) == {"view": [0.0, 0.0, 1.0, 2.0]}


@pytest.mark.parametrize("bad", [
    [0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0, 2.0], [0.0, float("nan"), 1.0, 1.0], [0.0, 0.0, float("inf"), 1.0],
    [1.0, 0.0, 1.0, 2.0], [2.0, 0.0, 1.0, 2.0], [0.0, 3.0, 1.0, 2.0], [0.0, 0.0, "1", 2.0], [0.0, 0.0, True, 2.0],
    "0 0 1 1", None, {"x0": 0},
])
def test_t8_invalid_view_box_reads_as_none(bad: Any) -> None:
    env = m.EnvFile(name="t", editor={"view": bad})
    assert view_box(env) is None
    assert editor_view_box({"view": bad}) is None
    if bad is not None:
        with pytest.raises(ValueError):
            with_view_box({}, bad)


def test_t8_view_box_accepts_ints_and_numpy_numbers() -> None:
    assert editor_view_box({"view": [0, 0, 2, 1]}) == (0.0, 0.0, 2.0, 1.0)
    box = (np.float32(-1.5), np.int64(0), np.float64(3.0), 2)
    assert editor_view_box(with_view_box({}, box)) == (-1.5, 0.0, 3.0, 2.0)
    assert editor_view_box("view") is None

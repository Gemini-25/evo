"""M5a: 에디터 문서 모델(편집, 되돌리기, 진화 대상, 복사·붙여넣기, 이름 바꾸기, 검사)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolab import paths
from evolab.app.editor.document import EnvDocument, where_to_ref
from evolab.spec.io import compile_env, env_hash, load_env, parse_env


def test_new_document_is_valid_and_has_no_errors(qtbot: Any) -> None:
    doc = EnvDocument()
    assert not [i for i in doc.issues if i.severity == "error"]
    assert parse_env(doc.data)


def test_edit_undo_redo_and_merge(qtbot: Any) -> None:
    doc = EnvDocument()
    doc.from_builtin("lander.evoenv")
    i = doc.find(("bodies", "hull"))
    path = ("bodies", i, "material", "density")
    for v in (51.0, 52.0, 53.0):
        doc.set_value(path, v)  # 같은 칸을 이어서 바꾸면 한 단계
    assert doc.get(path) == 53.0 and doc.undo_stack.count() == 1 and doc.dirty
    doc.undo_stack.undo()
    assert doc.get(path) == 50.0 and not doc.dirty
    doc.undo_stack.redo()
    assert doc.get(path) == 53.0


def test_add_remove_unique_ids(qtbot: Any) -> None:
    doc = EnvDocument()
    a = doc.add("bodies", {"id": "box", "shapes": [{"kind": "box", "w": 1, "h": 1}]})
    b = doc.add("bodies", {"id": "box", "shapes": [{"kind": "box", "w": 1, "h": 1}]})
    assert a == ("bodies", "box") and b == ("bodies", "box_1") and doc.selection == [b]
    doc.remove([a])
    assert doc.find(a) is None and doc.find(b) is not None
    doc.undo_stack.undo()
    assert doc.find(a) is not None


def test_evolvable_field_roundtrip(qtbot: Any) -> None:
    doc = EnvDocument()
    doc.from_builtin("lander.evoenv")
    i = doc.find(("bodies", "hull"))
    path = ("bodies", i, "shapes", 0, "w")
    pid = doc.make_evolvable(path, 1.0, 3.0, name="hull_w")
    assert doc.get(path) == "=p.hull_w" and doc.evolve_param(path) == "hull_w"
    spec = compile_env(doc.data)
    assert spec.body_param_count == 1
    doc.make_fixed(path)
    assert doc.get(path) == 2.0 and all(p["id"] != pid for p in doc.items("params"))


def test_copy_paste_remaps_internal_refs(qtbot: Any) -> None:
    doc = EnvDocument()
    doc.from_builtin("lander.evoenv")
    text = doc.copy_text([("bodies", "leg_l"), ("joints", "j_leg_l")])
    refs = doc.paste_text(text)
    assert len(refs) == 2
    new_joint = doc.item(refs[1])
    assert new_joint["b"] == refs[0][1] and new_joint["a"] == "hull"  # 함께 복사한 다리를 가리키고 본체는 그대로
    assert doc.paste_text("not json") == []


def test_rename_updates_references_and_formulas(qtbot: Any) -> None:
    doc = EnvDocument()
    doc.from_builtin("lander.evoenv")
    assert doc.rename(("bodies", "hull"), "body")
    spec = compile_env(doc.data)  # 참조와 수식이 모두 새 이름으로
    assert "body" in [b.id for b in spec.file.bodies]
    assert any("contact(body, tag.terrain)" in e.when for e in spec.file.events)
    assert not doc.rename(("bodies", "body"), "leg_l")  # 이미 있는 이름


def test_save_open_roundtrip_matches_builtin(qtbot: Any, tmp_path: Path) -> None:
    doc = EnvDocument()
    doc.from_builtin("track_car.evoenv")
    out = tmp_path / "차.evoenv"
    doc.save(out)
    assert env_hash(load_env(out).file) == env_hash(load_env(paths.builtin_env_dir() / "track_car.evoenv").file)
    doc2 = EnvDocument()
    doc2.open(out)
    assert doc2.path == out and not doc2.dirty


def test_checks_point_at_components(qtbot: Any) -> None:
    doc = EnvDocument()
    doc.from_builtin("lander.evoenv")
    i = doc.find(("sensors", "pad_rel"))
    doc.set_value(("sensors", i, "body"), "nope")
    issues = doc.run_check()
    errs = [x for x in issues if x.severity == "error"]
    assert errs and where_to_ref(doc.data, errs[0].where) == ("sensors", "pad_rel")
    assert doc.spec() is None
    # 겹친 물체 경고
    doc2 = EnvDocument()
    doc2.add("bodies", {"id": "a", "owner": "world", "type": "dynamic", "shapes": [{"kind": "box", "w": 2, "h": 2}]})
    doc2.add("bodies", {"id": "b", "owner": "world", "type": "dynamic", "pos": [0.5, 0], "shapes": [{"kind": "box", "w": 2, "h": 2}]})
    msgs = [x.message for x in doc2.run_check()]
    assert any("겹쳐" in m for m in msgs)


def test_rename_keeps_kinds_tags_and_updates_dotted_names(qtbot: Any) -> None:
    doc = EnvDocument()
    doc.from_builtin("lander.evoenv")
    assert doc.rename(("sensors", "fuel"), "fuel_ratio")
    assert doc.item(("sensors", "fuel_ratio"))["kind"] == "fuel"  # 종류 값은 그대로
    assert doc.rename(("bodies", "hull"), "body")
    assert "hull" in doc.item(("bodies", "body"))["tags"]  # 태그는 다른 이름 공간
    assert compile_env(doc.data) is not None
    doc2 = EnvDocument()
    doc2.from_builtin("track_car.evoenv")
    assert doc2.rename(("generators", "track"), "road")
    spec = compile_env(doc2.data)  # gen.road.*, road.cp.* 로 따라 바뀐다
    assert spec.track_set_size == 5

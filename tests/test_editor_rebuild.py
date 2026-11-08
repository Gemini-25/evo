"""M5a·M5b 완료 기준: 에디터만으로 내장 예제(착륙선, 트랙 자동차, 언덕 넘는 자동차)를 처음부터 다시 만든다.

빈 문서에서 시작해 부품은 모두 에디터 도구 함수(캔버스 도구가 부르는 것과 같은 함수)로 만들고, 이름 바꾸기와 값은
속성 패널·아래 탭이 쓰는 문서 편집(doc.rename, doc.set_value)으로만 채운다. 저장한 파일이 내장 예제와 내용 해시까지
같으면, 결정성(같은 파일·시드 → 같은 결과) 때문에 학습도 같다. 짧게 학습해 통계가 같은지도 직접 본다.

v2.0(설계 T41): 점프 게임·글라이더도 같은 방법으로 만든다. v2 부품은 v2 도구로 만든다: 장애물 줄(add_obstacles),
점프 추진기(add_jump), 공기역학(add_aero), 날개 모양(add_wing), 정해진 움직임(make_moving). 예제 파일이 저장소에 아직
없으면(묶음 A 전) 설계 원형 파일(scratchpad/v2/final/envs의 jump_B·glider_t120, v2.1 칸 exists를 뺀 것)로 한다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from evolab import paths
from evolab.app.editor import tools as T
from evolab.app.editor.document import EnvDocument
from evolab.engine import trainer as trainer_mod
from evolab.engine.pool import WorkerPool
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.spec import model as m
from evolab.spec.io import compile_env, env_hash, load_env, parse_env

SECTIONS = ("generators", "bodies", "joints", "actuators", "sensors", "regions", "checkpoints")
PROTOTYPES = {"jump.evoenv": "jump_B.evoenv", "glider.evoenv": "glider_t120.evoenv"}
PROTOTYPE_DIR = Path("/tmp/claude-0/-home-user-evo/6ff254e5-e86e-5068-a687-5ac6a7065af0/scratchpad/v2/final/envs")


def _drop_exists(value: Any) -> Any:
    """원형 파일의 v2.1 칸(물체 exists: null)을 뺀다(v2.0 모델에는 없는 칸)."""
    if isinstance(value, dict):
        return {k: _drop_exists(v) for k, v in value.items() if k != "exists"}
    if isinstance(value, list):
        return [_drop_exists(v) for v in value]
    return value


def example_text(name: str) -> str:
    """내장 예제 글. v2 예제가 저장소에 아직 없으면 설계 원형 파일."""
    path = paths.builtin_env_dir() / name
    if path.exists() or name not in PROTOTYPES:
        return path.read_text(encoding="utf-8")
    proto = PROTOTYPE_DIR / PROTOTYPES[name]
    if not proto.exists():
        pytest.skip(f"no {name} yet and no prototype file")
    return json.dumps(_drop_exists(json.loads(proto.read_text(encoding="utf-8"))), ensure_ascii=False)
TOP_FIELDS = ("name", "description", "world", "params", "randomize", "agent", "events", "termination", "scoring",
              "brain", "evaluation")


def num(v: Any, default: float = 0.0) -> float:
    return float(v) if isinstance(v, (int, float)) else default


def create_with_tool(doc: EnvDocument, section: str, item: dict[str, Any]) -> tuple[str, str]:
    """내장 예제의 부품 하나를 에디터 도구로 만든다(위치·크기는 대강, 정확한 값은 나중에 속성 패널로)."""
    kind = item.get("kind")
    if section == "generators":
        if kind == "heightmap":
            return T.add_heightmap(doc, pad=item.get("pad") is not None)
        if kind == "track":
            return T.add_generated_track(doc) if item.get("source", "generated") == "generated" else T.add_drawn_track(
                doc, [tuple(p) for p in item["centerline"]], item.get("width"))
        if kind == "hills":
            return T.add_hills(doc)
        if kind == "obstacles":
            return T.add_obstacles(doc)
    if section == "bodies":
        shapes = item["shapes"]
        first = shapes[0]
        owner = item.get("owner", "agent")
        if first["kind"] == "box":
            ref = T.add_box(doc, owner, (0.0, 0.0), num(first["w"], 1.0), num(first["h"], 1.0))
        elif first["kind"] == "circle":
            ref = T.add_circle(doc, owner, (0.0, 0.0), num(first["r"], 0.5))
        else:
            pts = [(num(x), num(y)) for x, y in first["points"]]
            if any(isinstance(v, str) for xy in first["points"] for v in xy):
                # 수식 꼭짓점: 사람은 모양을 대강 그리고 꼭짓점 칸에 수식을 적는다
                pts = [(-1.0, -0.25), (1.0, -0.25), (1.0, 0.35), (-1.0, 0.35)]
            ref = T.add_poly(doc, owner, pts)
        for extra in shapes[1:]:
            if extra["kind"] == "wing":  # 날개 도구: 앞전에서 뒷전까지 끌기(대강, 정확한 값은 속성 패널로)
                T.add_wing(doc, ref[1], (0.5, 0.0), (-0.5, 0.0))
                continue
            T.add_shape(doc, ref, {"kind": extra["kind"], **({"r": 0.1} if extra["kind"] == "circle" else
                                                               {"w": 0.1, "h": 0.1})})
        return ref
    if section == "joints":
        return T.add_joint(doc, item["kind"], item["a"], item["b"], (0.0, 0.0), (0.0, 0.0))
    if section == "actuators":
        if kind == "fuel_tank":
            return T.add_fuel_tank(doc, None)
        if kind == "thruster":
            return T.add_thruster(doc, item["body"], (0.0, 0.0), num(item.get("dir"), math.pi / 2))
        if kind == "vehicle":
            return T.add_vehicle(doc, item["body"])
        if kind == "motor":
            return T.add_motor(doc, item["joint"])
        if kind == "jump":
            return T.add_jump(doc, item["body"], num(item.get("dir"), math.pi / 2))
        if kind == "aero":
            return T.add_aero(doc, item["body"])
    if section == "sensors":
        if kind == "rays":
            return T.add_rays(doc, item["body"], (0.0, 0.0))
        if kind == "body_state":
            return T.add_body_state(doc, item["body"])
        if kind == "contact":
            return T.add_contact(doc, item["body"])
        if kind == "target":
            return T.add_target(doc, item["body"], item["target"])
        if kind == "fuel":
            return T.add_fuel_sensor(doc, item["tank"])
        if kind == "joint_angle":
            return T.add_joint_angle(doc, item["joint"])
        if kind == "time_left":
            return T.add_time_left(doc)
    if section == "regions":
        return T.add_region(doc, item["kind"], (0.0, 0.0), 2.0, 2.0)
    if section == "checkpoints":
        return T.add_checkpoints(doc, [tuple(map(tuple, ln)) for ln in item["lines"]], item.get("loop", True))
    raise AssertionError(f"no tool for {section}/{kind}")


def rebuild(name: str) -> tuple[EnvDocument, dict[str, int]]:
    target = json.loads(json.dumps(parse_env(example_text(name)).model_dump(mode="json", by_alias=True)))
    doc = EnvDocument()
    doc.new()
    counts = {"tool": 0, "rename": 0, "field": 0}
    # 매개변수와 랜덤 범위가 먼저 있어야 수식 칸이 컴파일된다(아래 탭: 매개변수, 랜덤 범위)
    for key in ("params", "randomize"):
        doc.set_value((key,), target[key])
        counts["field"] += 1
    for section in SECTIONS:
        for i, item in enumerate(target[section]):
            ref = create_with_tool(doc, section, item)
            counts["tool"] += 1
            if ref[1] != item["id"]:
                assert doc.rename(ref, item["id"]), (ref, item["id"])
                counts["rename"] += 1
            motion = item.get("motion") if section == "bodies" else None
            if isinstance(motion, dict):  # 길 그리기 도구(정해진 움직임). 점은 대강, 정확한 값은 속성 패널로
                pts = [(num(x), num(y)) for x, y in motion.get("points", [])[1:]] or [(1.0, 0.0)]
                T.make_moving(doc, item["id"], pts, motion["kind"])
    # 속성 패널: 부품마다 내장 예제와 다른 칸만 고친다(관절 도구가 자식 물체에 place_by를 넣는 것처럼 나중 도구가
    # 앞 부품을 바꿀 수 있으므로 부품을 다 만든 뒤에 한 번에 고친다).
    for section in SECTIONS:
        for item in target[section]:
            path = doc.item_path((section, item["id"]))
            assert path is not None
            current = doc.get(path)
            for field, value in item.items():
                if current.get(field) != value:
                    doc.set_value((*path, field), value)
                    counts["field"] += 1
            for field in [f for f in current if f not in item]:
                doc.edit("", lambda d, p=path, f=field: d[p[0]][p[1]].pop(f))
    for key in TOP_FIELDS:
        if doc.get((key,)) != target[key]:
            doc.set_value((key,), target[key])
            counts["field"] += 1
    return doc, counts


@pytest.mark.parametrize("name", ["lander.evoenv", "track_car.evoenv", "hill_car.evoenv", "jump.evoenv",
                                  "glider.evoenv"])
def test_rebuild_builtin_example_with_editor_only(qtbot: Any, tmp_path: Path, name: str) -> None:
    doc, counts = rebuild(name)
    out = tmp_path / name
    doc.save(out)
    built = load_env(out)
    original = parse_env(example_text(name))
    assert env_hash(built.file) == env_hash(original)
    assert counts["tool"] == sum(len(getattr(original, s)) for s in SECTIONS)
    assert not [i for i in doc.run_check() if i.severity == "error"]
    print(f"\n{name}: tools {counts['tool']}, renames {counts['rename']}, field edits {counts['field']}")


@pytest.mark.parametrize("name", ["lander.evoenv", "hill_car.evoenv"])
def test_rebuilt_file_trains_identically(qtbot: Any, tmp_path: Path, name: str) -> None:
    """같은 파일이면 학습도 같다: 다시 만든 파일과 내장 예제를 같은 시드로 3세대 학습해 통계를 비교."""
    doc, _ = rebuild(name)
    rebuilt = doc.text()
    original = (paths.builtin_env_dir() / name).read_text(encoding="utf-8")
    assert compile_env(rebuilt).body_param_count == compile_env(original).body_param_count
    rows = []
    for text in (rebuilt, original):
        config = TrainConfig(popsize=6, trials=2, validate_every=24, max_generations=3, run_seed=4)
        with WorkerPool(0) as pool:
            t = Trainer(text, config, pool)
            got: list[dict[str, Any]] = []
            t.run(got.append)
        rows.append([{k: v for k, v in r.items() if k not in trainer_mod.TIME_COLUMNS} for r in got])
    assert rows[0] == rows[1]
    if name == "hill_car.evoenv":
        assert compile_env(rebuilt).body_param_count == 5  # 몸 수치(바퀴 반지름, 차체 꼭짓점) 진화
        assert isinstance(compile_env(rebuilt).file.generators[0], m.HillGen)

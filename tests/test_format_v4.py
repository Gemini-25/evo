"""형식 4 등록·저장·검사 (스포츠 설계 6.1 T2·T6·T9, T26의 엔진 몫, 묶음 R, DE 2.1·2.6).

- T2: World.view·World.damping·Move·Balance를 하나씩 켠 파일은 형식 4, 끄면 원래 형식(1·2·3)과 처음 글. 얼린 v6.1.0의 모델과
  견주어 새로 생긴 칸·종류는 모두 FIELD_SINCE·KIND_SINCE에 4로 등록되어 있다(등록하지 않은 새 칸이 있으면 실패).
- T6: 형식 4 검사(위에서 본 세계의 중력·옆 세계용 부품·바닥 마찰, 밀기 최고 속도, 음수, 구동기 물체)의 글이 상수와 같고, 모든 검사
  글에 코드(Issue.code)가 있다.
- T9: 얼린 v6.1이 형식 4 파일(달리기)을 「더 새로운 프로그램」 글로 거절한다.
- T26(엔진 몫): editor.train·rules·recipe를 단 형식 1 파일을 얼린 v6.1이 오류 없이 열고 다시 저장해도 editor가 남는다.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

import pytest

from evolab import paths
from evolab.app import strings as S
from evolab.spec import model as m
from evolab.spec import symbols as sy
from evolab.spec.compiled import check_env
from evolab.spec.io import FIELD_SINCE, KIND_SINCE, compile_env, dump_env, env_hash, feature_version, parse_env
from tests.oracle.frozen import frozen_envs, frozen_run

RELEASE = "v6.1.0"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
MEASURED = DATA / "presets" / "running_measured.evoenv"
OLD_SIX = ("lander", "track_car", "hill_car", "jump", "glider", "wheels")
SPECIAL_CODES = {"unknown_ref", "unknown_tag", "unknown_event", "unknown_channel", "no_scoring", "expr"}
"""문구 상수 이름이 아닌 코드(계약 C11): 화면의 고치기가 묶어 보는 이름과 수식 오류."""


def builtin(name: str) -> dict[str, Any]:
    return json.loads((paths.builtin_env_dir() / f"{name}.evoenv").read_text(encoding="utf-8"))


def agent_body(data: dict[str, Any]) -> str:
    return next(b["id"] for b in data["bodies"] if b.get("owner", "agent") == "agent" and b.get("type", "dynamic") == "dynamic")


def top_world(**world: Any) -> dict[str, Any]:
    """위에서 본 작은 세계: 원 선수 하나(밀기 xy), 공 하나, 목표 영역."""
    return {
        "version": 4, "name": "위", "world": {"view": "top", "gravity": [0.0, 0.0], "damping": 0.5,
                                             "bounds": [-20.0, -20.0, 20.0, 20.0], "max_time": 5.0, **world},
        "bodies": [
            {"id": "disc", "shapes": [{"kind": "circle", "r": 0.4}], "material": {"density": 1.0}},
            {"id": "ball", "owner": "world", "pos": [3.0, 0.0], "shapes": [{"kind": "circle", "r": 0.3}]},
        ],
        "actuators": [{"kind": "move", "id": "push", "body": "disc", "max_force": 20.0, "max_speed": 3.0,
                       "keys_up": ["Up"], "keys_down": ["Down"], "keys_left": ["Left"], "keys_right": ["Right"]}],
        "sensors": [{"kind": "body_state", "id": "me", "body": "disc", "fields": ["x", "y", "vx", "vy"]}],
        "scoring": {"blocks": [{"kind": "progress", "id": "go", "body": "disc", "weight": 1.0}]},
    }


def issues_of(data: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    return [(i.severity, i.where, i.message, i.code) for i in check_env(parse_env(data))]


# ---------------------------------------------------------------------------
# T2: 하나씩 켜면 4, 끄면 원래 형식
# ---------------------------------------------------------------------------

def _features(data: dict[str, Any]) -> dict[str, Any]:
    body = agent_body(data)
    return {
        "view": lambda d: d["world"].__setitem__("view", "top"),
        "damping": lambda d: d["world"].__setitem__("damping", 0.9),
        "move": lambda d: d["actuators"].append({"kind": "move", "id": "push62", "body": body, "mode": "x",
                                                 "max_force": 10.0, "max_speed": 2.0}),
        "balance": lambda d: d["actuators"].append({"kind": "balance", "id": "bal62", "body": body, "stiffness": 50.0,
                                                    "max_torque": 80.0}),
    }


@pytest.mark.parametrize("name, low", [("lander", 1), ("jump", 2), ("wheels", 3)])
@pytest.mark.parametrize("feature", ["view", "damping", "move", "balance"])
def test_t2_each_format_4_part_alone_makes_a_format_4_file(name: str, low: int, feature: str) -> None:
    base = builtin(name)
    env = parse_env(base)
    assert feature_version(env) == low
    text = dump_env(env)
    data = json.loads(text)
    _features(data)[feature](data)
    env4 = parse_env(data)
    saved = json.loads(dump_env(env4))
    assert feature_version(env4) == 4 and saved["version"] == 4
    assert env_hash(env4) != env_hash(env)
    # 끄면(기본값으로 되돌리거나 부품을 빼면) 원래 형식과 처음 글
    if feature in ("view", "damping"):
        saved["world"][feature] = m.World.model_fields[feature].default
    else:
        saved["actuators"].pop()
    back = parse_env(saved)
    assert feature_version(back) == low and dump_env(back) == text and env_hash(back) == env_hash(env)


def test_t2_a_format_4_file_keeps_its_parts_through_save_and_load(tmp_path: Path) -> None:
    data = top_world()
    data["actuators"].append({"kind": "balance", "id": "bal", "body": "disc", "stiffness": 5.0, "damping": 0.5,
                              "max_torque": 3.0, "target": 0.25})
    env = parse_env(data)
    text = dump_env(env)
    again = parse_env(text)
    assert dump_env(again) == text and env_hash(again) == env_hash(env)
    saved = json.loads(text)
    assert saved["version"] == 4 and (saved["world"]["view"], saved["world"]["damping"]) == ("top", 0.5)
    kinds = [a["kind"] for a in saved["actuators"]]
    assert kinds == ["move", "balance"]


MODEL_FIELDS = """
import json
import sys

import numpy as np
from pydantic import BaseModel

from evolab.spec import model

out = {}
for name in sorted(dir(model)):
    cls = getattr(model, name)
    if isinstance(cls, type) and issubclass(cls, BaseModel) and cls.__module__ == model.__name__:
        out[name] = list(cls.model_fields)
np.savez(sys.argv[1], fields=np.array(json.dumps(out)))
"""


def test_t2_every_field_and_kind_new_since_the_frozen_release_is_registered_at_format_4(tmp_path: Path) -> None:
    script = tmp_path / "fields.py"
    script.write_text(textwrap.dedent(MODEL_FIELDS).lstrip(), encoding="utf-8")
    old = json.loads(str(frozen_run(RELEASE, script)["fields"]))
    from pydantic import BaseModel

    now = {name: list(cls.model_fields) for name in dir(m) if isinstance(cls := getattr(m, name), type)
           and issubclass(cls, BaseModel) and cls.__module__ == m.__name__}
    new_kinds = sorted(set(now) - set(old))
    assert set(old) <= set(now)
    unregistered = [k for k in new_kinds if KIND_SINCE.get(getattr(m, k)) != 4]
    assert unregistered == [], f"형식 4로 등록하지 않은 새 종류: {unregistered}"
    missing: list[str] = []
    for name, fields in old.items():
        assert fields == now[name][:len(fields)], f"{name}: 옛 칸의 순서가 바뀜"
        for f in now[name][len(fields):]:
            if FIELD_SINCE.get((getattr(m, name), f)) != 4:
                missing.append(f"{name}.{f}")
    assert missing == [], f"형식 4로 등록하지 않은 새 칸: {missing}"
    assert new_kinds == ["Balance", "Move"]


# ---------------------------------------------------------------------------
# T6: 형식 4 검사와 코드
# ---------------------------------------------------------------------------

def test_t6_top_world_checks_use_the_constants_and_codes() -> None:
    assert issues_of(top_world()) == []
    bad = top_world(gravity=[0.0, -9.81], damping=1.0)
    found = issues_of(bad)
    assert ("error", "world.gravity", S.SPEC_TOP_GRAVITY, "top_gravity") in found
    assert ("warning", "world.damping", S.SPEC_TOP_NO_DAMPING, "top_no_damping") in found
    # 중력의 한 칸만 0이 아니어도, 수식이면 알 수 없으니 묻지 않는다
    assert any(c == "top_gravity" for *_, c in issues_of(top_world(gravity=[1.0, 0.0])))
    assert not any(c == "top_gravity" for *_, c in issues_of(top_world(gravity=["=r.g", 0.0])))
    # 바닥 마찰이 1이어도 움직이는 물체마다 항력이 있으면 멈춘다
    dragged = top_world(damping=1.0)
    for b in dragged["bodies"]:
        b["drag"] = 0.5
    assert not any(c == "top_no_damping" for *_, c in issues_of(dragged))
    # 옆에서 본 세계는 위의 셋을 보지 않는다(중력이 있어도, 바닥 마찰이 1이어도)
    side = top_world(view="side", gravity=[0.0, -9.81], damping=1.0)
    assert not {c for *_, c in issues_of(side)} & {"top_gravity", "top_no_damping", "top_part"}


def test_t6_side_parts_in_a_top_world_are_a_warning() -> None:
    data = top_world()
    data["actuators"] += [
        {"kind": "jump", "id": "hop", "body": "disc", "speed": 3.0},
        {"kind": "aero", "id": "air", "body": "disc"},
    ]
    found = [x for x in issues_of(data) if x[3] == "top_part"]
    assert found == [
        ("warning", "actuators[1]", S.SPEC_TOP_PART.format(id="hop", kind=S.EDP_VALUE_NAMES["jump"]), "top_part"),
        ("warning", "actuators[2]", S.SPEC_TOP_PART.format(id="air", kind=S.EDP_VALUE_NAMES["aero"]), "top_part"),
    ]


def test_t6_move_and_balance_values() -> None:
    data = top_world()
    data["actuators"][0].update(max_speed=0.0, max_force=-1.0, max_turn=-2.0)
    data["actuators"].append({"kind": "balance", "id": "bal", "body": "disc", "stiffness": -1.0, "damping": -2.0,
                              "max_torque": -3.0})
    found = issues_of(data)
    assert ("error", "actuators[0].max_speed", S.SPEC_MOVE_SPEED.format(id="push"), "move_speed") in found
    for where, name in (("actuators[0]", "max_force"), ("actuators[0]", "max_turn"), ("actuators[1]", "stiffness"),
                        ("actuators[1]", "damping"), ("actuators[1]", "max_torque")):
        assert ("error", f"{where}.{name}", S.SPEC_NEGATIVE.format(name=name), "negative") in found
    # 수식 칸은 런타임이 범위를 잡는다(검사 글 없음)
    ok = top_world()
    ok["params"] = [{"id": "f", "value": 5.0}]
    ok["actuators"][0].update(max_speed="=p.f", max_force="=p.f")
    assert issues_of(ok) == []
    neg_speed = top_world()
    neg_speed["actuators"][0]["max_speed"] = -1.0
    assert [x[3] for x in issues_of(neg_speed)] == ["move_speed"]


def test_t6_move_and_balance_need_a_moving_agent_body() -> None:
    data = top_world()
    data["actuators"] = [
        {"kind": "move", "id": "push", "body": "ball", "max_force": 1.0, "max_speed": 1.0},
        {"kind": "balance", "id": "bal", "body": "ball", "stiffness": 1.0, "max_torque": 1.0},
        {"kind": "move", "id": "ghost", "body": "nobody", "max_force": 1.0, "max_speed": 1.0},
    ]
    found = issues_of(data)
    assert ("error", "actuators[0].body", S.SPEC_ACTUATOR_BODY.format(ref="ball"), "actuator_body") in found
    assert ("error", "actuators[1].body", S.SPEC_ACTUATOR_BODY.format(ref="ball"), "actuator_body") in found
    assert ("error", "actuators[2].body", S.SPEC_REF_BODY.format(ref="nobody"), "unknown_ref") in found


def test_t6_move_keys_are_checked_like_other_keys() -> None:
    data = top_world()
    data["actuators"][0].update(keys_up=["Space"], keys_down=["R"], keys_left=["Nope"], keys_right=["D"])
    found = issues_of(data)
    assert [x[1:] for x in found] == [
        ("actuators[0].keys[0]", S.SPEC_KEY_SPACE_JUMP_ONLY.format(id="push"), "key_space_jump_only"),
        ("actuators[0].keys[1]", S.SPEC_KEY_RESERVED.format(key="R"), "key_reserved"),
        ("actuators[0].keys[2]", S.SPEC_KEY_UNKNOWN.format(key="Nope"), "key_unknown"),
    ]


def test_t6_balance_alone_is_not_an_output() -> None:
    data = top_world()
    data["actuators"] = [{"kind": "balance", "id": "bal", "body": "disc", "stiffness": 1.0, "max_torque": 1.0}]
    found = issues_of(data)
    assert ("warning", "actuators", S.SPEC_NO_ACTUATORS, "no_actuators") in found
    spec = compile_env(data)
    assert spec.outputs == [] and list(spec.actuator_channels) == []


@pytest.mark.parametrize("mode, tails", [("xy", ["x", "y"]), ("x", ["x"]), ("heading", ["forward", "turn"])])
def test_t6_move_channels(mode: str, tails: list[str]) -> None:
    data = top_world()
    data["actuators"][0]["mode"] = mode
    data["actuators"].append({"kind": "balance", "id": "bal", "body": "disc", "stiffness": 1.0, "max_torque": 1.0})
    env = parse_env(data)
    chans = sy.actuator_channels(env)
    assert [(c.name, c.owner, c.low, c.high) for c in chans] == [(f"push.{t}", "push", -1.0, 1.0) for t in tails]
    spec = compile_env(data)
    assert spec.outputs == [f"push.{t}" for t in tails]
    # 채널은 수식에서 읽을 수 있다(에너지·사건)
    data["events"] = [{"id": "fast", "when": f"abs(push.{tails[0]}) > 0.5"}]
    assert compile_env(data).events


def _env_files() -> list[Path]:
    out = [paths.builtin_env_dir() / f"{n}.evoenv" for n in OLD_SIX]
    out += [ROOT / p for p in frozen_envs(RELEASE)]
    out += [MEASURED]
    return out


def _bad_envs() -> list[dict[str, Any]]:
    """검사 글을 여럿 내는 손 환경: 옛 검사의 대부분(참조·키·범위·수식·사건)과 형식 4 검사."""
    lander = builtin("lander")
    lander["bodies"].append({"id": lander["bodies"][0]["id"], "shapes": [{"kind": "box", "w": 1, "h": 1}]})
    lander["sensors"].append({"kind": "contact", "id": "touch", "body": agent_body(lander), "with": ["tag.nothing", "nobody"]})
    lander["events"].append({"id": "bad", "when": "nobody.x >"})
    lander["termination"].append({"event": "nowhere", "outcome": "failure"})
    lander["brain"]["outputs"] = ["nope"]
    lander["scoring"] = {"blocks": []}
    lander["randomize"].append({"id": "q", "dist": "uniform", "low": 2.0, "high": 1.0})
    empty = {"version": 1, "name": "빈"}
    return [lander, empty, top_world(gravity=[0.0, -1.0], damping=1.0)]


def test_t6_every_issue_has_a_code_named_after_its_constant() -> None:
    seen: set[str] = set()
    datas = [json.loads(p.read_text(encoding="utf-8")) for p in _env_files()] + _bad_envs()
    for data in datas:
        for i in check_env(parse_env(data)):
            assert i.code, f"코드 없는 검사 글: {i}"
            assert re.fullmatch(r"[a-z0-9_]+", i.code), i.code
            if i.code not in SPECIAL_CODES:
                assert hasattr(S, "SPEC_" + i.code.upper()), i.code
            seen.add(i.code)
    for code in ("unknown_ref", "unknown_tag", "unknown_event", "unknown_channel", "no_scoring", "expr",
                 "no_sensors", "no_actuators", "top_gravity", "top_no_damping", "id_duplicate", "rand_range"):
        assert code in seen, code


def test_t6_issue_text_and_order_are_unchanged_by_the_codes() -> None:
    """코드는 칸 하나를 더할 뿐이다: 글·자리·심각도·차례는 v6.1과 같다(바이트 비교는 test_frozen_identity의 T1)."""
    data = _bad_envs()[0]
    found = check_env(parse_env(data))
    assert [str(i) for i in found] == [f"{i.where}: {i.message}" if i.where else i.message for i in found]
    assert any(i.code == "unknown_tag" and "tag.nothing" in i.message for i in found)
    assert any(i.code == "unknown_ref" and "'nobody'" in i.message for i in found)


# ---------------------------------------------------------------------------
# T9·T26: 얼린 v6.1과 형식 4 파일, editor 칸
# ---------------------------------------------------------------------------

OPEN_IN = """
import json
import sys

import numpy as np

from evolab.spec.compiled import EnvFileError
from evolab.spec.io import dump_env, load_env

rows = {}
for path in sys.argv[2:]:
    try:
        spec = load_env(path)
    except EnvFileError as exc:
        rows[path] = ["refused", [[i.severity, i.where, i.message] for i in exc.issues]]
    else:
        rows[path] = ["opened", json.loads(dump_env(spec.file)), [[i.severity, i.message] for i in spec.issues]]
np.savez(sys.argv[1], rows=np.array(json.dumps(rows, ensure_ascii=False)))
"""


def test_t9_t26_the_frozen_release_refuses_format_4_and_keeps_editor_data(tmp_path: Path) -> None:
    running4 = tmp_path / "running4.evoenv"
    env4 = parse_env(MEASURED.read_text(encoding="utf-8"))
    running4.write_text(dump_env(env4), encoding="utf-8")
    assert json.loads(running4.read_text(encoding="utf-8"))["version"] == 4
    editor = {"train": {"qualify_entrants": 3, "qualify_trials": 9600},
              "rules": [{"kind": "fall", "event": "fall", "blocks": ["s_fall"]}],
              "recipe": {"recipe": 1, "start": "running", "built": "0" * 64, "templates": {"biped": "1" * 64}},
              "view": [-2.0, -1.0, 12.0, 6.0]}
    plain = builtin("lander")
    plain["editor"] = editor
    plain_path = tmp_path / "plain.evoenv"
    plain_path.write_text(dump_env(parse_env(plain)), encoding="utf-8")
    assert json.loads(plain_path.read_text(encoding="utf-8"))["version"] == 1
    script = tmp_path / "open_in.py"
    script.write_text(textwrap.dedent(OPEN_IN).lstrip(), encoding="utf-8")
    rows = json.loads(str(frozen_run(RELEASE, script, running4, plain_path)["rows"]))
    status, issues = rows[str(running4)]
    assert status == "refused"
    assert issues == [["error", "version", S.SPEC_NEWER_VERSION.format(version=4)]]
    status, saved, found = rows[str(plain_path)]
    assert status == "opened" and saved["editor"] == editor and saved["version"] == 1
    assert [x for x in found if x[0] == "error"] == []
    # 지금 프로그램에서도 editor는 해시 밖이고 다시 저장해도 남는다
    now = parse_env(plain_path.read_text(encoding="utf-8"))
    assert json.loads(dump_env(now))["editor"] == editor
    assert env_hash(now) == env_hash(parse_env(builtin("lander")))
    assert json.loads(dump_env(parse_env(running4.read_text(encoding="utf-8"))))["version"] == 4

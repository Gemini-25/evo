"""M1: 환경 파일 읽기, 검증, 오류 메시지, 정규 저장."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from evolab import paths
from evolab.app import strings as S
from evolab.spec import model as m
from evolab.spec.compiled import EnvFileError, check_env
from evolab.spec.io import compile_env, dump_env, env_hash, load_env, parse_env

LANDER = paths.builtin_env_dir() / "lander.evoenv"


def lander_data() -> dict[str, Any]:
    return json.loads(LANDER.read_text(encoding="utf-8"))


def messages(exc: EnvFileError) -> list[str]:
    return [str(i) for i in exc.issues]


def test_builtin_lander_loads_without_issues() -> None:
    spec = load_env(LANDER)
    assert spec.issues == []
    assert len(spec.inputs) == 9
    assert spec.outputs == ["main", "side_l", "side_r"]
    assert spec.file.evaluation.trials == 5


def test_canonical_dump_round_trips() -> None:
    env = load_env(LANDER).file
    text = dump_env(env)
    again = parse_env(text)
    assert dump_env(again) == text
    assert again == env


def test_env_hash_ignores_editor_section_only() -> None:
    env = load_env(LANDER).file
    edited = env.model_copy(update={"editor": {"zoom": 3}})
    assert env_hash(edited) == env_hash(env)
    changed = env.model_copy(update={"name": "다른 이름"})
    assert env_hash(changed) != env_hash(env)


def test_bad_json_reports_line() -> None:
    with pytest.raises(EnvFileError) as exc:
        parse_env('{"name": "x",\n  "world": }')
    assert "2" in messages(exc.value)[0]


def test_newer_version_is_rejected() -> None:
    data = lander_data()
    data["version"] = m.CURRENT_VERSION + 1
    with pytest.raises(EnvFileError) as exc:
        parse_env(data)
    assert S.SPEC_NEWER_VERSION.format(version=data["version"]) in messages(exc.value)[0]


def test_unknown_field_gets_korean_message_with_path() -> None:
    data = lander_data()
    data["bodies"][0]["colour"] = "red"
    with pytest.raises(EnvFileError) as exc:
        parse_env(data)
    assert f"bodies[0].colour: {S.SPEC_ERR_EXTRA}" in messages(exc.value)


def test_number_or_formula_field_message() -> None:
    data = lander_data()
    data["bodies"][0]["shapes"][0]["w"] = "wide"
    with pytest.raises(EnvFileError) as exc:
        parse_env(data)
    assert f"bodies[0].shapes[0].w: {S.SPEC_ERR_NUM_OR_FORMULA}" in messages(exc.value)


def test_unknown_kind_message() -> None:
    data = lander_data()
    data["sensors"][0]["kind"] = "sonar"
    with pytest.raises(EnvFileError) as exc:
        parse_env(data)
    assert any(msg.startswith("sensors[0]") for msg in messages(exc.value))


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda d: d["sensors"][1].__setitem__("body", "nobody"), S.SPEC_REF_BODY.format(ref="nobody")),
        (lambda d: d["actuators"][1].__setitem__("tank", "nope"), S.SPEC_REF_TANK.format(ref="nope")),
        (lambda d: d["termination"][0].__setitem__("event", "flying"), S.SPEC_REF_EVENT.format(ref="flying")),
        (lambda d: d["bodies"][1].__setitem__("id", "hull"), S.SPEC_ID_DUPLICATE.format(id="hull", other="bodies")),
        (lambda d: d["events"][0].__setitem__("id", "time"), S.SPEC_ID_RESERVED.format(id="time")),
        (lambda d: d["world"].__setitem__("kind", "grid"), S.SPEC_GRID_NOT_YET),
        (lambda d: d["params"][0].update(evolve=True, min=1.5), S.SPEC_PARAM_VALUE_OUT.format(id="leg_len")),
        (lambda d: d["brain"].__setitem__("inputs", ["st.vx", "nope.x"]), S.SPEC_REF_CHANNEL.format(ref="nope.x")),
    ],
)
def test_semantic_errors(mutate: Any, expected: str) -> None:
    data = lander_data()
    mutate(data)
    with pytest.raises(EnvFileError) as exc:
        compile_env(data)
    assert any(expected in msg for msg in messages(exc.value)), messages(exc.value)


def test_build_formula_cannot_use_step_values() -> None:
    data = lander_data()
    data["bodies"][0]["shapes"][0]["w"] = "=hull.speed"
    with pytest.raises(EnvFileError) as exc:
        compile_env(data)
    assert any(msg.startswith("bodies[0].shapes[0].w") for msg in messages(exc.value))


def test_generator_cannot_use_body_params() -> None:
    data = lander_data()
    data["generators"][0]["amplitude"] = "=p.leg_len"
    with pytest.raises(EnvFileError) as exc:
        compile_env(data)
    assert any(msg.startswith("generators[0].amplitude") for msg in messages(exc.value))


def test_event_condition_must_be_boolean() -> None:
    data = lander_data()
    data["events"][1]["when"] = "hull.speed + 1"
    with pytest.raises(EnvFileError) as exc:
        compile_env(data)
    assert any(msg.startswith("events[1].when") for msg in messages(exc.value))


def test_event_cycle_is_detected() -> None:
    data = lander_data()
    data["events"][1]["when"] = "off_pad and hull.speed > 1"
    data["events"][2]["when"] = "crash"
    with pytest.raises(EnvFileError) as exc:
        compile_env(data)
    assert S.SPEC_EVENT_CYCLE in " ".join(messages(exc.value))


def test_query_function_arguments_are_checked() -> None:
    data = lander_data()
    data["events"][1]["when"] = "contact(hull, tag.lava)"
    with pytest.raises(EnvFileError) as exc:
        compile_env(data)
    assert any(msg.startswith("events[1].when") and "tag.lava" in msg for msg in messages(exc.value))


def test_not_yet_features_are_reported_with_milestone() -> None:
    data = lander_data()
    data["bodies"][0]["type"] = "kinematic"  # 움직이는 정적 물체는 2차
    with pytest.raises(EnvFileError) as exc:
        compile_env(data)
    assert S.SPEC_NOT_YET.format(milestone="2") in " ".join(messages(exc.value))


def test_warnings_do_not_block_loading() -> None:
    data = lander_data()
    data["scoring"] = {"blocks": []}
    env = parse_env(data)
    issues = check_env(env)
    assert [i.severity for i in issues] == ["warning"]
    compile_env(env)  # 경고만 있으면 쓸 수 있다


def test_implicit_timeout_end_is_added() -> None:
    data = lander_data()
    data["termination"] = [t for t in data["termination"] if t["event"] != "timeout"]
    spec = compile_env(data)
    assert spec.terminations[-1].event == "timeout"


def test_read_failure_is_reported(tmp_path: Path) -> None:
    with pytest.raises(EnvFileError):
        load_env(tmp_path / "없는 파일.evoenv")


def test_korean_path_round_trip(tmp_path: Path) -> None:
    from evolab.spec.io import save_env

    env = load_env(LANDER).file
    path = tmp_path / "한글 폴더" / "착륙선.evoenv"
    path.parent.mkdir()
    save_env(env, path)
    assert load_env(path).file == env


def test_model_copy_does_not_share_mutable_defaults() -> None:
    a = m.EnvFile(name="a")
    b = m.EnvFile(name="b")
    a.bodies.append(m.Body(id="x", shapes=[m.Circle(kind="circle", r=1.0)]))
    assert b.bodies == []
    assert copy.deepcopy(a) == a

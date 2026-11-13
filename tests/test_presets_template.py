"""6.2.0 묶음 P: 틀 읽기·펼치기·넣기 (스포츠 설계 6.1 T10·T11·T12·T13, T19의 틀 쪽).

내장 몸 틀(biped·disc)은 묶음 PM의 것이다. 아직 없으면 그 칸의 시험은 공용 시험 자료의 몸 틀(fixture_walker·fixture_puck)로 돈다.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from evolab import paths
from evolab.app import strings as S
from evolab.presets import template as T
from evolab.presets.library import Library, default_library, reset_default_library
from evolab.presets.trainable import alive, quick_check
from evolab.spec.io import compile_env, feature_version, parse_env
from evolab.world.physics2d.episode import Episode, Trial

ROOT = Path(__file__).resolve().parents[1]
BUILTIN = ROOT / "src" / "evolab" / "assets" / "templates"
FIXTURES = ROOT / "tests" / "fixtures" / "recipes"
HANGUL = re.compile("[가-힣]")
GENERIC = ("ground_flat", "wall", "ball", "goal", "track_side", "yard_top")
"""묶음 P의 일반 틀(설계 4.1)."""
EXPECTED_FORMAT = {"ground_flat": 1, "wall": 1, "ball": 1, "goal": 1, "track_side": 1, "yard_top": 4, "biped": 4,
                   "disc": 4}
"""틀마다 기대 형식(설계 3.1 끝, T12): 기본값으로 그 세계에 넣은 결과. biped는 균형 보조를 끄면 1."""


def builtin_ids() -> list[str]:
    return sorted(p.stem for p in BUILTIN.glob("*.evotpl"))


def lib() -> Library:
    return Library([BUILTIN, FIXTURES])


def world_env(world: str) -> dict[str, Any]:
    data: dict[str, Any] = {"version": 4, "name": "t", "world": {"gravity": [0.0, -9.81] if world == "side" else [0.0, 0.0]}}
    if world == "top":
        data["world"]["view"] = "top"
        data["world"]["damping"] = 0.5
    return data


def body_for(world: str) -> T.Template:
    """그 세계의 몸 틀: 내장(PM)이 있으면 그것, 없으면 공용 시험 자료."""
    library = lib()
    want = {"side": "biped", "top": "disc"}[world]
    if library.has(want) and library.get(want).source == "builtin":
        return library.get(want)
    return library.get({"side": "fixture_walker", "top": "fixture_puck"}[world])


# ---------------------------------------------------------------------------
# T10 틀 읽기
# ---------------------------------------------------------------------------


def test_generic_templates_are_present_and_load() -> None:
    assert set(GENERIC) <= set(builtin_ids())
    library = Library([BUILTIN])
    assert library.problems() == []
    for tid in builtin_ids():
        tpl = library.get(tid)
        assert tpl.source == "builtin" and tpl.id == tid and tpl.sha256 == T.template_hash(tpl)
        assert len(tpl.sha256) == 64


def test_builtin_templates_name_screen_text_by_id() -> None:
    """D55·T19: 내장 틀은 화면 글을 strings.py의 id로 가리키고, 슬라이더마다 이름과 풍선이 있다."""
    for tid in builtin_ids():
        raw = json.loads((BUILTIN / f"{tid}.evotpl").read_text(encoding="utf-8"))
        assert "name" not in raw and "description" not in raw and raw["name_id"] in S.TPL_NAMES, tid
        assert raw["name_id"] in S.TPL_BLURBS
        for p in raw.get("params", []):
            assert "label" not in p and "unit" not in p, (tid, p)
            assert p["label_id"] in S.TPL_PARAM_LABELS and p["label_id"] in S.TPL_PARAM_TIPS, (tid, p)
            if "unit_id" in p:
                assert p["unit_id"] in S.TPL_UNIT_NAMES
        for slot in (raw.get("slots") or {}).values():
            assert "label" not in slot or not HANGUL.search(str(slot["label"])) or "label_id" in slot, tid
        for chord in raw.get("keys", []):
            assert "label" not in chord, tid
        # 자산의 한글은 환경 내용이 되는 칸(부품 표시 이름)에만(D55)
        for where, text in T._strings(raw):
            if HANGUL.search(text):
                assert where.startswith("parts.") and where.endswith(".label"), (tid, where, text)


def _bad_templates() -> list[tuple[str, Any, str]]:
    good = json.loads((FIXTURES / "fixture_walker.evotpl").read_text(encoding="utf-8"))

    def edit(fn: Any) -> dict[str, Any]:
        d = copy.deepcopy(good)
        fn(d)
        return d

    return [
        ("not_object", [1, 2], S.TPL_ERR_FORMAT.split("{")[0]),
        ("format", edit(lambda d: d.update(format="evolab-env")), S.TPL_ERR_FORMAT.split("{")[0]),
        ("newer", edit(lambda d: d.update(version=2)), S.TPL_ERR_NEWER.split("(")[0]),
        ("version_type", edit(lambda d: d.update(version="1")), "version"),
        ("unknown_field", edit(lambda d: d.update(colour="red")), "colour"),
        ("bad_id", edit(lambda d: d.update(id="1abc")), "id"),
        ("kind", edit(lambda d: d.update(kind="robot")), "kind"),
        ("world", edit(lambda d: d.update(world="front")), "world"),
        ("range", edit(lambda d: d["params"][0].update(min=2.0)), S.TPL_ERR_PARAM_RANGE.split("「")[0]),
        ("ref", edit(lambda d: d["parts"]["joints"][0].update(b="leg_x")), "leg_x"),
        ("slot", edit(lambda d: d["parts"]["sensors"][3].update({"with": ["$floor"]})), "$floor"),
        ("expand", edit(lambda d: d["parts"]["bodies"][0]["shapes"][0].update(w="=t.size * 0 - 1")),
         S.TPL_EXPAND_WHICH["value"]),
        ("formula_name", edit(lambda d: d["parts"]["bodies"][0]["shapes"][0].update(w="=t.width * 2")), "t.width"),
        ("too_many_main", edit(lambda d: d["params"].extend(
            [{"id": f"x{k}", "label": "x", "value": 1.0, "min": 0.0, "max": 2.0} for k in range(2)])), "params"),
        ("root", edit(lambda d: d.update(root="leg_x")), "leg_x"),
        ("if", edit(lambda d: d["parts"]["sensors"][0].update({"if": "nope"})), "if"),
    ]


@pytest.mark.parametrize(("name", "data", "needle"), _bad_templates(), ids=[c[0] for c in _bad_templates()])
def test_broken_templates_fail_with_a_korean_message(name: str, data: Any, needle: str) -> None:
    with pytest.raises(T.TemplateError) as err:
        T.parse_template(data)
    text = str(err.value)
    assert HANGUL.search(text), (name, text)
    assert needle in text, (name, text)


def test_broken_user_template_is_listed_and_skipped(tmp_path: Path) -> None:
    user = tmp_path / "user"
    user.mkdir()
    (user / "broken.evotpl").write_text("{ not json", encoding="utf-8")
    newer = json.loads((FIXTURES / "fixture_puck.evotpl").read_text(encoding="utf-8"))
    newer.update(id="later", version=9)
    (user / "later.evotpl").write_text(json.dumps(newer), encoding="utf-8")
    mine = json.loads((BUILTIN / "ball.evotpl").read_text(encoding="utf-8"))  # 내장과 같은 id: 「내 틀」로 따로
    (user / "ball.evotpl").write_text(json.dumps(mine), encoding="utf-8")
    library = Library([BUILTIN, user, tmp_path / "missing"])
    problems = dict((p.name, why) for p, why in library.problems())
    assert set(problems) == {"broken.evotpl", "later.evotpl"}
    assert all(HANGUL.search(why) for why in problems.values())
    assert S.TPL_ERR_NEWER.format(version=9) == problems["later.evotpl"]
    assert library.get("ball").source == "builtin"
    assert [t.source for t in library.templates("prop", "top") if t.id == "ball"] == ["builtin", "user"]
    with pytest.raises(KeyError):
        library.get("nope")


def test_default_library_follows_the_template_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    reset_default_library()
    try:
        first = default_library()
        assert default_library() is first
        assert {t.id for t in first.all()} == set(builtin_ids())
        user = tmp_path / "data" / "templates"
        user.mkdir(parents=True)
        (user / "puck.evotpl").write_text((FIXTURES / "fixture_puck.evotpl").read_text(encoding="utf-8"),
                                          encoding="utf-8")
        reset_default_library()
        assert default_library().get("fixture_puck").source == "user"
    finally:
        reset_default_library()


def test_kinds_and_worlds_filter() -> None:
    library = Library([BUILTIN])
    assert {t.id for t in library.templates("arena", "side")} >= {"track_side"}
    assert "yard_top" not in {t.id for t in library.templates("arena", "side")}
    assert {"wall", "ball", "goal"} <= {t.id for t in library.templates("prop", "top")}
    assert "ground_flat" not in {t.id for t in library.templates("prop", "top")}


# ---------------------------------------------------------------------------
# T11 귀퉁이에서 L1, 60스텝 값이 유한, 몸 틀은 L2
# ---------------------------------------------------------------------------


def _scene(tpl: T.Template, values: dict[str, float], mirror: bool = False) -> dict[str, Any]:
    """틀 하나를 그 세계에 넣은 장면: 몸 틀이 아니면 그 세계의 몸을, 땅이 필요한 몸이면 평평한 땅을 함께."""
    world = "top" if tpl.world == "top" else "side"
    data = world_env(world)
    if tpl.kind != "body":
        T.insert(data, T.expand(tpl, values, mirror=mirror, at=(0.0, 0.0) if tpl.kind == "arena" else (3.0, 0.0)))
        if tpl.bounds is not None:
            data["world"]["bounds"] = T.evaluate(list(tpl.bounds), T._values(tpl, values))
        body = body_for(world)
    else:
        body = tpl
    if world == "side" and "ground" not in T.scene_tags(data):
        T.insert(data, T.expand(lib().get("ground_flat"), at=(0.0, 0.0)))
    filled, _ = T.resolve_slots(body, data)
    T.insert(data, T.expand(body, values if tpl.kind == "body" else None, mirror=mirror if tpl.kind == "body" else False,
                            slots=filled))
    data["scoring"] = {"blocks": [{"kind": "progress", "id": "s_go", "body": body.root, "weight": 1.0}]}
    return data


def _all_templates() -> list[T.Template]:
    library = lib()
    return [library.get(t) for t in builtin_ids()] + [library.get("fixture_walker"), library.get("fixture_puck")]


@pytest.mark.parametrize("tid", [*builtin_ids(), "fixture_walker", "fixture_puck"])
def test_every_template_corner_is_l1_and_runs_finite(tid: str) -> None:
    tpl = lib().get(tid)
    for which, values in tpl.corners():
        for mirror in (False, True):
            data = _scene(tpl, values, mirror)
            quick = quick_check(data)
            assert quick.ok, (tid, which, mirror, [str(i) for i in quick.errors])
            spec = quick.spec
            assert spec is not None
            ep = Episode(spec, None, Trial(3))
            zero = [0.0] * len(spec.outputs)
            for _ in range(60):
                if ep.done:
                    break
                ep.step(zero)
            snap = ep.snapshot()
            assert np_finite(snap.poses), (tid, which, mirror)
            if tpl.kind == "body":
                verdict = alive(spec)
                assert verdict.ok, (tid, which, mirror, verdict.reason)


def np_finite(arr: Any) -> bool:
    return all(math.isfinite(float(x)) for x in arr.ravel())


# ---------------------------------------------------------------------------
# T12 펼친 글
# ---------------------------------------------------------------------------


def _numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [x for v in value for x in _numbers(v)]
    if isinstance(value, dict):
        return [x for v in value.values() for x in _numbers(v)]
    return []


@pytest.mark.parametrize("tid", [*builtin_ids(), "fixture_walker", "fixture_puck"])
def test_expanded_text_has_no_slider_names_and_is_stable(tid: str) -> None:
    tpl = lib().get(tid)
    for _, values in tpl.corners():
        a = T.expand(tpl, values, at=(1.25, -0.5))
        b = T.expand(tpl, dict(values), at=(1.25, -0.5))
        text = json.dumps(a, ensure_ascii=False, sort_keys=True)
        assert text == json.dumps(b, ensure_ascii=False, sort_keys=True)
        assert "t." not in re.sub(r'"[^"]*"\s*:', "", text).replace("tag.", "")  # 값에 t.* 이름이 남지 않음
        assert "$" not in text
        literal = set(_numbers(dict(tpl.parts)))  # 틀에 숫자로 적힌 값은 그대로(π 같은)
        for x in _numbers({k: v for k, v in a.items() if k in T.SECTIONS}):
            assert round(x, 6) == x or x in literal or -x in literal, (tid, x)  # 계산한 값은 소수 6자리


def test_values_outside_the_range_are_clamped() -> None:
    tpl = lib().get("fixture_walker")
    hi = T.expand(tpl, {"size": 99.0, "power": -5.0, "unknown": 3.0})
    edge = T.expand(tpl, {"size": tpl.param("size").max, "power": tpl.param("power").min})
    assert hi == edge


@pytest.mark.parametrize("tid", [t for t in EXPECTED_FORMAT if t in builtin_ids()])
def test_expected_format_per_template(tid: str) -> None:
    tpl = lib().get(tid)
    data = _scene(tpl, tpl.defaults())
    assert feature_version(parse_env(data)) == EXPECTED_FORMAT[tid], tid
    if tid == "biped" and any(p.id == "balance" for p in tpl.params):
        off = _scene(tpl, dict(tpl.defaults(), balance=0.0))
        assert feature_version(parse_env(off)) == 1  # 균형 보조를 끄면 부품을 넣지 않아 형식 1(D27)


def test_fixture_bodies_stay_format_1_in_a_side_world() -> None:
    for tid in ("fixture_walker",):
        tpl = lib().get(tid)
        for _, values in tpl.corners():
            assert feature_version(parse_env(_scene(tpl, values))) == 1


def test_optional_parts_follow_their_checkbox_slider() -> None:
    raw = json.loads((FIXTURES / "fixture_walker.evotpl").read_text(encoding="utf-8"))
    raw["params"].append({"id": "balance", "label": "보조", "value": 1.0, "min": 0.0, "max": 1.0, "step": 1.0,
                          "main": True})
    raw["parts"]["actuators"].append({"kind": "balance", "id": "bal", "body": "torso", "stiffness": 100.0,
                                      "damping": 12.5, "max_torque": 100.0, "if": "balance"})
    tpl = T.parse_template(raw)
    on = T.expand(tpl)
    off = T.expand(tpl, {"balance": 0.0})
    assert [a["id"] for a in on["actuators"]][-1] == "bal" and "if" not in on["actuators"][-1]
    assert "bal" not in [a["id"] for a in off["actuators"]]
    assert feature_version(parse_env(_scene(tpl, tpl.defaults()))) == 4
    assert feature_version(parse_env(_scene(tpl, {"balance": 0.0}))) == 1


# ---------------------------------------------------------------------------
# T13 넣기
# ---------------------------------------------------------------------------


def test_one_insert_is_one_change_and_undo_restores_it() -> None:
    tpl = lib().get("fixture_walker")
    data = world_env("side")
    T.insert(data, T.expand(lib().get("ground_flat")))
    before = copy.deepcopy(data)
    refs = T.insert(data, T.expand(tpl, slots=T.resolve_slots(tpl, data)[0]))
    assert refs[0] == ("bodies", "torso") and len(refs) == sum(len(tpl.parts.get(s, [])) for s in T.SECTIONS)
    assert data != before
    after = copy.deepcopy(data)
    data.clear()
    data.update(before)  # 되돌리기 = 넣기 전 사전
    assert data == before and after != before


def test_same_template_twice_numbers_the_whole_second_group() -> None:
    tpl = lib().get("fixture_walker")
    data = world_env("side")
    T.insert(data, T.expand(lib().get("ground_flat")))
    first = T.insert(data, T.expand(tpl))
    second = T.insert(data, T.expand(tpl, at=(3.0, 0.0)))
    assert [i for _, i in second] == [f"{i}_2" for _, i in first]
    third = T.insert(data, T.expand(tpl, at=(6.0, 0.0)))
    assert {i[-2:] for _, i in third} == {"_3"}
    joints = {j["id"]: j for j in data["joints"]}
    assert (joints["hip_l_2"]["a"], joints["hip_l_2"]["b"]) == ("torso_2", "leg_l_2")  # 묶음 안의 참조도 같은 번호
    assert [r["id"] for r in data["randomize"]] == ["tilt"]  # 같은 랜덤 값은 하나
    assert quick_check(data).ok


def test_slot_resolves_to_the_first_tag_the_scene_has() -> None:
    tpl = lib().get("fixture_walker")
    assert tpl.slots["$ground"].tags == ("ground", "terrain")
    data = world_env("side")
    assert T.resolve_slots(tpl, data) == ({}, ["$ground"])
    data["bodies"] = [{"id": "hill", "owner": "world", "type": "static", "tags": ["terrain"],
                       "shapes": [{"kind": "box", "w": 10.0, "h": 1.0}], "pos": [0.0, -0.5]}]
    assert T.resolve_slots(tpl, data) == ({"$ground": "tag.terrain"}, [])
    data["bodies"].append({"id": "floor", "owner": "world", "type": "static", "tags": ["ground"],
                           "shapes": [{"kind": "box", "w": 10.0, "h": 1.0}], "pos": [0.0, -2.0]})
    assert T.resolve_slots(tpl, data) == ({"$ground": "tag.ground"}, [])  # 앞의 태그가 먼저(D44)
    parts = T.expand(tpl, slots=T.resolve_slots(tpl, data)[0])
    assert {tuple(s["with"]) for s in parts["sensors"] if s["kind"] == "contact"} == {("tag.ground",)}


def test_body_template_places_the_agent_by_its_spawn() -> None:
    tpl = lib().get("fixture_walker")
    data = world_env("side")
    T.insert(data, T.expand(tpl, at=(2.0, 0.0)))
    assert data["agent"]["spawn"] == {"y": 0.65, "angle": "=r.tilt", "x": 2.0}
    assert data["randomize"] == [dict(r) for r in tpl.randoms]
    mirrored = T.expand(tpl, mirror=True)
    assert mirrored["spawn"]["angle"] == "=-(r.tilt)"
    joint = next(j for j in mirrored["joints"] if j["id"] == "hip_l")
    assert joint["limit"] == [-1.2, 0.9]  # [lo, hi] → [−hi, −lo]
    motor = next(a for a in mirrored["actuators"] if a["id"] == "m_hip_l")
    assert (motor["keys_pos"], motor["keys_neg"]) == (["A"], ["Q"])


def test_manual_brain_lists_get_the_new_channels() -> None:
    data = world_env("top")
    data["brain"] = {"inputs": [], "outputs": []}
    T.insert(data, T.expand(lib().get("fixture_puck")))
    assert data["brain"]["inputs"] == ["st.x", "st.y", "st.vx", "st.vy"]
    assert data["brain"]["outputs"] == ["t_right", "t_left", "t_up", "t_down"]


def test_thumbnail_fits_the_card() -> None:
    for tpl in _all_templates():
        ops = T.thumbnail(tpl)
        assert ops, tpl.id
        for op in ops:
            for x, y in op.points:
                assert -0.5 <= x <= 96.5 and -0.5 <= y <= 72.5, (tpl.id, op)


def test_expand_and_insert_are_fast_enough() -> None:
    """M7의 한 조각(어림의 회귀 지킴이): 틀 넣기 + 검사 중앙값이 넉넉히 50ms 아래."""
    import time

    tpl = body_for("side")
    times = []
    for _ in range(9):
        t0 = time.perf_counter()
        data = world_env("side")
        T.insert(data, T.expand(lib().get("track_side")))
        T.insert(data, T.expand(tpl, slots=T.resolve_slots(tpl, data)[0]))
        compile_env(data)
        times.append(time.perf_counter() - t0)
    assert sorted(times)[4] < 0.05

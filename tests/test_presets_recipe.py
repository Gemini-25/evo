"""6.2.0 묶음 P: 조리법 → 환경 (스포츠 설계 6.1 T14·T15, D28·D29·D43).

내장 몸 틀(biped·disc)과 종목 조리법(running·drill)은 묶음 PM의 것이다. 아직 없으면 빈 세계 조리법은 공용 시험 자료의 몸 틀을 그 id로
베낀 대역으로 돈다(대역 보관함: stand_in_library). PM의 파일이 들어오면 같은 시험이 진짜 틀로 돈다.
"""

from __future__ import annotations

import copy
import itertools
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from evolab import paths
from evolab.app import strings as S
from evolab.presets import recipe as RC
from evolab.presets import rules as R
from evolab.presets.library import Library
from evolab.presets.trainable import alive, quick_check
from evolab.spec.editor_meta import editor_recipe, editor_rules, editor_train, editor_view_box
from evolab.spec.io import compile_env, env_hash, feature_version, parse_env

ROOT = Path(__file__).resolve().parents[1]
BUILTIN = ROOT / "src" / "evolab" / "assets" / "templates"
RECIPES = ROOT / "src" / "evolab" / "assets" / "recipes"
FIXTURES = ROOT / "tests" / "fixtures" / "recipes"
HANGUL = re.compile("[가-힣]")


def stand_in_library(tmp: Path) -> Library:
    """내장 틀 + 공용 시험 자료 + (PM의 biped·disc가 아직 없으면) 그 id로 베낀 대역."""
    extra = tmp / "stand_in"
    extra.mkdir(exist_ok=True)
    for body, source in (("biped", "fixture_walker"), ("disc", "fixture_puck")):
        if not (BUILTIN / f"{body}.evotpl").exists():
            raw = json.loads((FIXTURES / f"{source}.evotpl").read_text(encoding="utf-8"))
            raw["id"] = body
            (extra / f"{body}.evotpl").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return Library([BUILTIN, FIXTURES, extra])


@pytest.fixture()
def lib(tmp_path: Path) -> Library:
    return stand_in_library(tmp_path)


def fixture_recipe(name: str) -> RC.Recipe:
    return RC.Recipe.from_json(json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# 조리법 모양
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["fixture_side", "fixture_top", *[p.stem for p in sorted(RECIPES.glob("*.json"))]])
def test_recipe_json_round_trip(name: str) -> None:
    path = FIXTURES / f"{name}.json" if name.startswith("fixture") else RECIPES / f"{name}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    recipe = RC.Recipe.from_json(raw)
    again = RC.Recipe.from_json(recipe.to_json())
    assert again == recipe
    expect = {k: v for k, v in raw.items() if k != "status"}
    assert recipe.to_json() == expect or {k: v for k, v in recipe.to_json().items()} == expect


@pytest.mark.parametrize(("change", "needle"), [
    (lambda d: d.update(recipe=2), S.RECIPE_ERR_NEWER.split("(")[0]),
    (lambda d: d.update(recipe="1"), "recipe"),
    (lambda d: d.update(colour=1), "colour"),
    (lambda d: d.update(world="front"), "world"),
    (lambda d: d.update(roles=[]), "roles"),
    (lambda d: d["roles"][0]["body"].update(values={"size": "big"}), "values"),
    (lambda d: d["roles"][0]["body"].update(evolve=["size"]), "evolve"),
    (lambda d: d.update(rules=[{"who": "runner"}]), "rules"),
    (lambda d: d["players"]["seats"][0].update(count=0), "count"),
    (lambda d: d.update(extra={"joints": []}), "extra"),
    (lambda d: d.pop("arena"), "arena"),
])
def test_bad_recipes_fail_in_korean(change: Any, needle: str) -> None:
    raw = json.loads((FIXTURES / "fixture_side.json").read_text(encoding="utf-8"))
    change(raw)
    with pytest.raises(RC.RecipeError) as err:
        RC.Recipe.from_json(raw)
    assert HANGUL.search(str(err.value)) and needle in str(err.value)


def test_missing_template_and_builtin(lib: Library) -> None:
    recipe = fixture_recipe("fixture_side")
    recipe.roles[0].body.template = "nobody"
    with pytest.raises(RC.RecipeError) as err:
        RC.build(recipe, lib)
    assert str(err.value) == S.RECIPE_ERR_TEMPLATE.format(id="nobody")
    with pytest.raises(RC.RecipeError):
        RC.load_builtin("no_such_recipe")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def test_build_fixture_side_is_format_1_with_the_editor_markers(lib: Library) -> None:
    recipe = fixture_recipe("fixture_side")
    data = RC.build(recipe, lib)
    assert data == RC.build(fixture_recipe("fixture_side"), lib)  # 순수 함수: 같은 조리법 → 같은 사전
    assert json.dumps(data, ensure_ascii=False) == json.dumps(RC.build(recipe, lib), ensure_ascii=False)
    env = parse_env(data)
    assert data["version"] == 1 and feature_version(env) == 1  # 형식 1 부품만: 얼린 v6.1이 연다(T26의 짝)
    marker = editor_recipe(data["editor"])
    assert marker is not None and marker["built"] == env_hash(env)
    assert marker["templates"] == {t: lib.get(t).sha256 for t in ("fixture_walker", "track_side")}
    assert {k: v for k, v in marker.items() if k not in ("built", "templates")} == recipe.to_json()
    assert editor_train(data["editor"]) == {"qualify_entrants": 2, "qualify_trials": 4000}
    assert editor_view_box(data["editor"]) == (-3.0, -1.0, 9.0, 4.0)
    assert [r["kind"] for r in editor_rules(data["editor"])] == ["distance", "touch_ends"]
    assert RC.matches(data)
    spec = compile_env(data)
    assert [i for i in spec.issues if i.severity == "error"] == []
    assert len(spec.inputs) == 9 and len(spec.outputs) == 2
    assert data["world"]["max_time"] == 10.0 and data["world"]["bounds"] == [-10.0, -5.0, 150.0, 30.0]
    assert data["agent"]["spawn"]["angle"] == "=r.tilt" and [r["id"] for r in data["randomize"]] == ["tilt"]
    uses, rest = R.find_rules(data)
    assert [(u.kind, u.editable) for u in uses] == [("distance", True), ("touch_ends", True), ("time_limit", True)]
    assert rest == 0


def test_build_fixture_top_is_format_4_because_of_the_world_kind(lib: Library) -> None:
    data = RC.build(fixture_recipe("fixture_top"), lib)
    assert data["version"] == 4 and data["world"]["view"] == "top" and data["world"]["gravity"] == [0.0, 0.0]
    assert data["brain"]["hidden"] == [8] and data["evaluation"]["trials"] == 2
    assert data["agent"]["spawn"]["x"] == -3.5  # 빈 마당의 자리 start = (−길이/4, 0)
    assert "train" not in data["editor"]


def test_matches_turns_false_when_the_physics_changes(lib: Library) -> None:
    data = RC.build(fixture_recipe("fixture_side"), lib)
    data["name"] = "다른 이름"
    assert not RC.matches(data)  # 이름도 해시에 든다
    data = RC.build(fixture_recipe("fixture_side"), lib)
    data["editor"]["view"] = [0, 0, 1, 1]
    assert RC.matches(data)  # editor는 해시 밖
    data["bodies"][1]["material"]["density"] = 1.0
    assert not RC.matches(data)
    assert not RC.matches({"name": "x"})


def test_props_use_arena_spawns_and_rules_add_see_sensors(lib: Library) -> None:
    recipe = RC.Recipe(
        start="blank_top", name="공 넣기 시험", world="top", arena=RC.Use("yard_top"),
        roles=[RC.Role("player", "", RC.Use("fixture_puck"))],
        props=[RC.Use("ball", spawn="center"), RC.Use("goal", spawn="goal")],
        rules=[{"rule": "near", "thing": "ball", "where": "goal", "per_m": 1.0, "when": "min"},
               {"rule": "reach", "part": "ball", "where": "goal", "outcome": "success", "bonus": 100.0},
               {"rule": "time_limit", "seconds": 15.0}],
        players=RC.Players("solo", [RC.Seat("player")]), world_values={"damping": 0.5})
    # reach의 part는 몸의 부품이어야 하므로 adjust가 뿌리로 바꾼다. 여기서는 그대로 build가 받는 것만 본다
    recipe.rules[1]["part"] = "disc"
    data = RC.build(recipe, lib)
    bodies = {b["id"]: b for b in data["bodies"]}
    assert bodies["ball"]["pos"] == [0.0, 0.0]
    regions = {r["id"]: r for r in data["regions"]}
    assert regions["goal"]["pos"] == [6.5, 0.0]
    sees = [(s["body"], s["target"]) for s in data["sensors"] if s["kind"] == "target"]
    assert sees == [("disc", "ball"), ("ball", "goal"), ("disc", "goal")]
    assert quick_check(data).ok


def test_physics_view_ignores_names_labels_keys_and_defaults() -> None:
    measured = json.loads((ROOT / "tests" / "data" / "presets" / "running_measured.evoenv").read_text(encoding="utf-8"))
    base = RC.physics_view(measured)
    dressed = copy.deepcopy(measured)
    dressed.update(name="달리기", description="두 발로 달립니다", version=4,
                   editor={"train": {"qualify_entrants": 3}, "view": [0, 0, 1, 1]})
    for b in dressed["bodies"]:
        b["label"] = "부품"
    for a in dressed["actuators"]:
        if a["kind"] == "motor":
            a["keys_pos"], a["keys_neg"] = ["Q"], ["A"]
    dressed["world"]["view"] = "side"  # 기본값을 적어도 같다
    dressed["evaluation"]["target_score"] = None
    assert RC.physics_view(dressed) == base
    changed = copy.deepcopy(measured)
    changed["actuators"][-1]["stiffness"] = 99.0
    assert RC.physics_view(changed) != base
    assert "version" not in base and "name" not in base and "label" not in json.dumps(base)


# ---------------------------------------------------------------------------
# 내장 조리법(T14) — 빈 세계 둘은 묶음 P, 종목은 묶음 PM
# ---------------------------------------------------------------------------


def _builtin_recipes() -> list[str]:
    return [p.stem for p in sorted(RECIPES.glob("*.json"))]


@pytest.mark.parametrize("name", _builtin_recipes())
def test_builtin_recipes_build_l1_l2_and_no_issue(name: str, lib: Library, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = json.loads((RECIPES / f"{name}.json").read_text(encoding="utf-8"))
    assert raw.get("status", "ok") in RC.STATUSES
    recipe = RC.load_builtin(name)
    data = RC.build(recipe, lib)
    quick = quick_check(data)
    assert quick.ok and quick.spec is not None, [str(i) for i in quick.errors]
    assert quick.spec.issues == [], [str(i) for i in quick.spec.issues]  # 엔진 검사의 어떤 심각도도 0(D40의 엔진 쪽)
    verdict = alive(quick.spec)
    assert verdict.ok, verdict.reason
    fixed, notes = RC.adjust(recipe, lib)  # 내장 조리법은 이미 맞춰져 있다: 바꾼 것 없음, 같은 환경
    assert notes == [] and env_hash(parse_env(RC.build(fixed, lib))) == env_hash(parse_env(data))
    for where, text in _strings(raw):  # 자산의 한글은 환경 내용이 되는 칸에만(D55)
        if HANGUL.search(text):
            assert where in ("name", "description") or where.endswith(".label"), (name, where)


def _strings(value: Any, path: str = "") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, list):
        return [x for i, v in enumerate(value) for x in _strings(v, f"{path}[{i}]")]
    if isinstance(value, dict):
        return [x for k, v in value.items() for x in _strings(v, f"{path}.{k}" if path else k)]
    return []


def test_blank_recipes_are_the_documented_starting_points(lib: Library) -> None:
    side, top = RC.load_builtin("blank_side"), RC.load_builtin("blank_top")
    assert (side.world, side.arena.template, side.roles[0].body.template) == ("side", "track_side", "biped")
    assert (top.world, top.arena.template, top.roles[0].body.template) == ("top", "yard_top", "disc")
    assert [r["rule"] for r in side.rules] == ["distance", "fall", "out_ends", "time_limit"]
    assert side.rules[-1]["seconds"] == top.rules[-1]["seconds"] == 15.0
    assert side.name == top.name == ""
    data = RC.build(side, lib)
    assert data["name"] == S.PRESET_NAME_FALLBACK


def test_starters_follow_the_order_and_hide_what_cannot_be_made(lib: Library, tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    cards = RC.starters(lib)
    ids = [c.id for c in cards]
    assert {"blank_side", "blank_top"} <= set(ids)
    assert ids == [i for i in RC.STARTER_ORDER if i in ids] + sorted(i for i in ids if i not in RC.STARTER_ORDER)
    for c in cards:
        assert c.status in ("ok", "experimental")
        if c.id.startswith("blank") or c.id == "drill":
            assert c.title and c.blurb
    # 숨김·대전·틀이 없는 조리법은 카드가 없다(D36·D47)
    folder = tmp_path / "assets" / "recipes"
    folder.mkdir(parents=True)
    base = json.loads((RECIPES / "blank_side.json").read_text(encoding="utf-8"))
    for name, change in (("hidden_one", {"status": "hidden"}), ("match_one", {"match": {"swap_sides": True}}),
                         ("no_body", {"roles": [{"id": "p", "label": "", "body": {"template": "ghost"}}]}),
                         ("exp_one", {"status": "experimental"})):
        (folder / f"{name}.json").write_text(json.dumps(dict(base, start=name, **change)), encoding="utf-8")
    monkeypatch.setattr(paths, "recipe_dir", lambda: folder)
    assert [(c.id, c.status) for c in RC.starters(lib)] == [("exp_one", "experimental")]


def test_cli_preset_writes_the_env(tmp_path: Path) -> None:
    from evolab.cli import main

    out = tmp_path / "out" / "그림.evoenv"
    recipe = tmp_path / "r.json"
    raw = json.loads((FIXTURES / "fixture_side.json").read_text(encoding="utf-8"))
    raw["arena"] = {"template": "track_side", "values": {"length": 60.0}}
    raw["roles"][0]["body"] = {"template": "ball", "values": {}}  # 내장 틀만 쓰는 조리법(물건을 몸 자리에: adjust 없음)
    recipe.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    assert main(["preset", str(recipe), "-o", str(out)]) == 1  # 몸 틀이 아니면 만들 수 없다(뿌리가 없어 규칙을 못 풂)
    raw["roles"][0]["body"] = {"template": "fixture_walker", "values": {}}
    recipe.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    assert main(["preset", str(recipe), "-o", str(out)]) == 1  # 기본 보관함에는 시험 자료 틀이 없다
    assert main(["preset", str(tmp_path / "none.json")]) == 2
    assert not out.exists()


def test_cli_preset_with_a_template_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.cli import main
    from evolab.presets.library import reset_default_library

    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    user = tmp_path / "data" / "templates"
    user.mkdir(parents=True)
    (user / "fixture_walker.evotpl").write_text((FIXTURES / "fixture_walker.evotpl").read_text(encoding="utf-8"),
                                                encoding="utf-8")
    reset_default_library()
    try:
        out = tmp_path / "side.evoenv"
        assert main(["preset", str(FIXTURES / "fixture_side.json"), "-o", str(out)]) == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["version"] == 1 and data["name"] == "시험용 옆 세계"
        assert env_hash(parse_env(data)) == data["editor"]["recipe"]["built"]
    finally:
        reset_default_library()


# ---------------------------------------------------------------------------
# 내장 달리기(T14의 PM 쪽): 파일이 있을 때만
# ---------------------------------------------------------------------------


RUNNING_ENV = ROOT / "environments" / "running.evoenv"


@pytest.mark.skipif(not (RECIPES / "running.json").exists() or not RUNNING_ENV.exists(),
                    reason="running.json·running.evoenv are added by group PM")
def test_running_env_is_the_recipe_output() -> None:
    data = json.loads(RUNNING_ENV.read_text(encoding="utf-8"))
    assert env_hash(parse_env(data)) == env_hash(parse_env(RC.build(RC.load_builtin("running"))))
    done = subprocess.run([sys.executable, str(ROOT / "packaging" / "make_presets.py"), "--check"],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr


# ---------------------------------------------------------------------------
# T15 늘 올바른 파일: 고를 수 있는 것의 조합 → adjust → build → L1
# ---------------------------------------------------------------------------


RULE_CHOICES: list[dict[str, Any]] = [
    {"rule": "distance", "direction": "right", "per_m": 1.0}, {"rule": "distance", "direction": "up", "per_m": 0.0},
    {"rule": "fall", "what": "$ground", "height": 0.4, "penalty": 1.0},
    {"rule": "fall", "parts": ["nope"], "what": "tag.wall", "height": 0.0, "penalty": 0.0},
    {"rule": "touch_ends", "what": "$ground", "outcome": "failure"},
    {"rule": "touch_ends", "what": "tag.wall", "outcome": "success"},
    {"rule": "touch_penalty", "what": "tag.ground", "per_s": 2.0},
    {"rule": "reach", "where": "goal", "outcome": "success", "bonus": 100.0},
    {"rule": "reach", "where": "nowhere", "outcome": "neutral", "bonus": 0.0},
    {"rule": "near", "thing": "ball", "where": "goal", "per_m": 1.0, "when": "min"},
    {"rule": "near", "where": "ball", "per_m": 3.0, "when": "final"},
    {"rule": "survive", "per_s": 1.0}, {"rule": "upright", "weight": 0.5},
    {"rule": "save_energy", "weight": 0.05}, {"rule": "out_ends", "outcome": "failure"},
    {"rule": "out_ends", "outcome": "neutral"}, {"rule": "distance", "direction": "sideways", "per_m": -2.0},
    {"rule": "made_up"},
]
PROP_CHOICES = ["ground_flat", "wall", "ball", "goal", "ghost"]


def _choice_space(lib: Library) -> dict[str, list[Any]]:
    starts = ["blank_side", "blank_top", *[n for n in RC.builtin_names() if n not in ("blank_side", "blank_top")]]
    bodies = [t.id for t in lib.all() if t.kind == "body"]
    return {"start": starts, "world": ["side", "top"], "body": bodies, "corner": ["value", "min", "max"]}


def _combo(lib: Library, rng: random.Random) -> RC.Recipe:
    space = _choice_space(lib)
    try:
        recipe = RC.default_recipe(rng.choice(space["start"]), lib)
    except RC.RecipeError:
        recipe = RC.load_builtin("blank_side")
    recipe.world = rng.choice(space["world"])
    body = rng.choice(space["body"])
    corner = rng.choice(space["corner"])
    tpl = lib.get(body)
    recipe.roles[0].body = RC.Use(body, {p.id: getattr(p, corner) for p in tpl.params})
    arena = lib.get(recipe.arena.template) if lib.has(recipe.arena.template) else None
    if arena is not None:
        recipe.arena.values = {p.id: getattr(p, rng.choice(space["corner"])) for p in arena.params}
    recipe.props = [RC.Use(p, {}, spawn=rng.choice([None, "center", "goal", "start"]))
                    for p in rng.sample(PROP_CHOICES, rng.randint(0, 3))]
    rules = [copy.deepcopy(r) for r in rng.sample(RULE_CHOICES, rng.randint(0, 6))]
    if rng.random() < 0.8:
        rules.append({"rule": "time_limit", "seconds": rng.choice([0.5, 15.0, 60.0, -1.0, 5000.0])})
    recipe.rules = rules
    return recipe


def _check_combo(recipe: RC.Recipe, lib: Library) -> None:
    fixed, notes = RC.adjust(recipe, lib)
    again, notes2 = RC.adjust(fixed, lib)
    assert again == fixed and notes2 == [], (recipe.to_json(), notes2)  # 멱등
    assert all(HANGUL.search(n) and "{" not in n for n in notes)
    data = RC.build(fixed, lib)  # 예외 없음
    quick = quick_check(data)
    assert quick.ok, (fixed.to_json(), [str(i) for i in quick.errors])  # L1
    assert [r.get("rule") for r in fixed.rules][-1] == "time_limit"


def test_random_choices_always_give_an_l1_file(lib: Library) -> None:
    rng = random.Random(6200)
    for _ in range(300):
        _check_combo(_combo(lib, rng), lib)


def test_adjust_reports_what_it_changed(lib: Library) -> None:
    recipe = RC.load_builtin("blank_side")
    recipe.world = "top"
    recipe.rules.insert(0, {"rule": "reach", "where": "goal", "outcome": "success", "bonus": 100.0})
    fixed, notes = RC.adjust(recipe, lib)
    from evolab.presets.josa import fill

    assert notes[0] == fill(S.PRESET_ADJUST_BODY, {"body": lib.get("disc").name})
    assert "(으)로" not in notes[0] and "을(를)" not in notes[1]
    assert notes[1] == fill(S.PRESET_ADJUST_PROP_ON, {"rule": S.RULE_TITLES["reach"], "prop": lib.get("goal").name})
    assert fixed.arena.template == "yard_top" and fixed.roles[0].body.template == "disc"
    assert any(n == S.PRESET_ADJUST_PROP_ON.replace("{rule}", S.RULE_TITLES["reach"]).replace(
        "「{prop}」을(를)", f"「{lib.get('goal').name}」를") for n in notes), notes
    assert any(S.RULE_TITLES["fall"] in n for n in notes)  # 넘어지면 끝은 위에서 본 세계에서 꺼짐
    assert [r["rule"] for r in fixed.rules] == ["reach", "distance", "out_ends", "time_limit"]
    assert [p.template for p in fixed.props] == ["goal"]


@pytest.mark.slow
def test_every_choice_combination_gives_an_l1_file(tmp_path: Path) -> None:
    """S2: T15의 조합 전부 — 첫 카드 × 세계 × 몸 틀 × 슬라이더 귀퉁이 × 물건 부분집합(둘까지) × 규칙(없음·하나), 그리고 규칙 둘의
    모든 짝 × 세계 × 몸 틀 × (물건 없음·넷 다). 상한 5분(리눅스 4코어·부하 3.7~4.0에서 57초)."""
    import time

    lib = stand_in_library(tmp_path)
    space = _choice_space(lib)
    t0 = time.perf_counter()
    count = 0

    def run(start: str, world: str, body: str, corner: str, props: list[str], rs: tuple[int, ...]) -> None:
        nonlocal count
        recipe = RC.default_recipe(start, lib)
        recipe.world = world
        tpl = lib.get(body)
        recipe.roles[0].body = RC.Use(body, {p.id: getattr(p, corner) for p in tpl.params})
        recipe.props = [RC.Use(p, {}, spawn="center") for p in props]
        recipe.rules = [copy.deepcopy(RULE_CHOICES[i]) for i in rs] + [{"rule": "time_limit", "seconds": 15.0}]
        _check_combo(recipe, lib)
        count += 1
        assert time.perf_counter() - t0 < 300, count

    prop_sets = [list(c) for k in range(3) for c in itertools.combinations(PROP_CHOICES[:4], k)]
    singles = [()] + [(i,) for i in range(len(RULE_CHOICES))]
    for start, world, body, corner in itertools.product(space["start"], space["world"], space["body"], space["corner"]):
        for props in prop_sets:
            for rs in singles:
                run(start, world, body, corner, props, rs)
    for world, body in itertools.product(space["world"], space["body"]):
        for props in ([], PROP_CHOICES[:4]):
            for rs in itertools.combinations(range(len(RULE_CHOICES)), 2):
                run(space["start"][0], world, body, "value", props, rs)
    assert count > 5000


# ---------------------------------------------------------------------------
# G1의 모양(D29 (1)): 측정에 쓴 몸을 틀로 옮기면 조리법이 원형 파일과 물리 동일한 환경을 만든다.
# 진짜 biped·running 조리법의 같은 확인은 묶음 PM의 T8이다. 여기서는 build·규칙 표·track_side가 그 모양을 낼 수 있는지만 본다.
# ---------------------------------------------------------------------------


def test_measured_running_shape_is_reachable_by_a_recipe(tmp_path: Path) -> None:
    measured = json.loads((ROOT / "tests" / "data" / "presets" / "running_measured.evoenv").read_text(encoding="utf-8"))
    body = copy.deepcopy(measured)
    for s in body["sensors"]:
        if s["kind"] == "contact":
            s["with"] = ["$ground"]
    tpl = {"format": "evolab-template", "version": 1, "id": "measured_biped", "kind": "body", "world": "side",
           "name": "측정한 몸", "params": [], "root": "torso",
           "parts": {"bodies": [b for b in body["bodies"] if b["id"] != "ground"], "joints": body["joints"],
                     "actuators": body["actuators"], "sensors": body["sensors"]},
           "slots": {"$ground": {"tags": ["ground"], "label": "땅"}},
           "randoms": body["randomize"], "spawn": body["agent"]["spawn"]}
    folder = tmp_path / "t"
    folder.mkdir()
    (folder / "measured_biped.evotpl").write_text(json.dumps(tpl, ensure_ascii=False), encoding="utf-8")
    library = Library([BUILTIN, folder])
    recipe = RC.Recipe.from_json({
        "recipe": 1, "start": "running", "name": "달리기", "world": "side",
        "arena": {"template": "track_side", "values": {}},
        "roles": [{"id": "runner", "label": "", "body": {"template": "measured_biped", "values": {}}}], "props": [],
        "rules": [{"rule": "distance", "part": "torso", "direction": "right", "per_m": 1.0},
                  {"rule": "fall", "parts": ["torso", "thigh_l", "thigh_r"], "what": "$ground", "root": "torso",
                   "height": 0.8, "penalty": 1.0},
                  {"rule": "save_energy", "weight": 0.02}, {"rule": "out_ends", "outcome": "failure"},
                  {"rule": "time_limit", "seconds": 15.0}],
        "players": {"mode": "solo", "seats": [{"role": "runner", "count": 1, "spawn": "start"}]},
        "evaluation": {"trials": 3, "aggregate": "mean"},
        "train": {"qualify_entrants": 3, "qualify_trials": 9600}})
    data = RC.build(recipe, library)
    assert RC.physics_view(data) == RC.physics_view(measured)
    assert data["version"] == 4 and data["editor"]["train"] == {"qualify_entrants": 3, "qualify_trials": 9600}
    assert [r["kind"] for r in data["editor"]["rules"]] == ["distance", "fall", "save_energy", "out_ends"]
    ground = data["bodies"][0]
    assert (ground["id"], ground["tags"]) == ("ground", ["ground"])  # track_side의 땅(설계 3.1)
    assert RC.adjust(recipe, library)[1] == []

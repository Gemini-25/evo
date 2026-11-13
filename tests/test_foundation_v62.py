"""6.2.0 바탕 단계(0)의 테스트 (스포츠 최종 설계 4.1 「바탕(0)」, 6.1의 T7·T18·T19 가운데 바탕의 몫).

- 형식 4 모델과 등록: `World.view`·`World.damping`·`Move`·`Balance`의 칸·기본값(3.1의 서명), 가장 낮은 형식으로 저장. 넓은 검사와
  런타임은 묶음 R의 tests/test_format_v4.py·test_parts_v4.py.
- `Issue.code`: 얼린 dataclass에 기본값 있는 칸 하나(옛 호출·`__str__`·`EnvFileError` 글 그대로).
- T7(바탕 몫): `TrainConfig.for_new_run`이 `editor.train`의 흰 목록 키만 얹고 틀린 값·모르는 키는 뺀다. 옛 환경은 얼린 v6.1과 같은 사전.
- T18: `paths.safe_file_stem`. 새 폴더 함수 셋과 `BUILTIN_ORDER`.
- T19(바탕 몫): 문자열 장부(tests/data/strings_ledger.json)의 「6.2.0」 이름이 모두 있고 「뺌」과 뒤 단계의 새 이름은 없다. 새 문자열에
  쓰지 않기로 한 말이 없고, 같은 글의 새 상수 쌍이 없다.
- 공용 시험 자료: 측정에 쓴 달리기 원형 파일, 시험용 틀 둘과 조리법 둘의 모양.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import operator
import re
import string
from pathlib import Path
from typing import Any

import pytest

from evolab import paths
from evolab.app import strings as S
from evolab.engine.trainer import MAX_QUALIFY_ENTRANTS, NEW_RUN_DEFAULTS, PRESET_TRAIN_KEYS, TrainConfig
from evolab.policy.genome import layout_for
from evolab.spec import editor_meta
from evolab.spec import model as m
from evolab.spec.compiled import EnvFileError, Issue
from evolab.spec.io import (
    FIELD_SINCE,
    KIND_SINCE,
    MIGRATIONS,
    compile_env,
    dump_env,
    env_hash,
    feature_version,
    load_env,
    parse_env,
)
from tests.oracle import frozen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "evolab"
DATA = ROOT / "tests" / "data"
FIXTURES = ROOT / "tests" / "fixtures" / "recipes"
MEASURED = DATA / "presets" / "running_measured.evoenv"
OLD_BUILTINS = ("lander", "track_car", "hill_car", "jump", "glider", "wheels")
RELEASE = "v6.1.0"


def _lander() -> dict[str, Any]:
    return json.loads((paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 형식 4 모델과 등록 (설계 3.1: 서명 고정)
# ---------------------------------------------------------------------------
def test_format_4_is_registered() -> None:
    assert m.CURRENT_VERSION == 4 and sorted(MIGRATIONS) == [1, 2, 3]
    assert FIELD_SINCE[(m.World, "view")] == 4 and FIELD_SINCE[(m.World, "damping")] == 4
    assert KIND_SINCE[m.Move] == 4 and KIND_SINCE[m.Balance] == 4
    d = {"version": 3, "name": "x"}
    assert MIGRATIONS[3](d) is d  # 칸·종류를 더하기만 한 변경: 사전 그대로


def test_world_view_and_damping_fields() -> None:
    w = m.World()
    assert (w.view, w.damping) == ("side", 1.0)
    assert list(m.World.model_fields)[-2:] == ["view", "damping"]  # 옛 칸 뒤에(옛 형식 글의 칸 순서 그대로)
    assert m.World(view="top", gravity=(0.0, 0.0), damping=0.5).view == "top"
    for bad in ({"view": "front"}, {"damping": 0.0}, {"damping": 1.5}, {"damping": -0.1}):
        with pytest.raises(ValueError):
            m.World(**bad)  # type: ignore[arg-type]


def test_move_and_balance_models_have_the_contract_fields() -> None:
    assert list(m.Move.model_fields) == ["kind", "id", "label", "body", "mode", "max_force", "max_speed", "max_turn",
                                         "keys_up", "keys_down", "keys_left", "keys_right"]
    assert list(m.Balance.model_fields) == ["kind", "id", "label", "body", "stiffness", "damping", "max_torque", "target"]
    mv = m.Move(kind="move", id="push", body="disc", max_force=40.0, max_speed=3.0)
    assert (mv.mode, mv.max_turn, mv.label) == ("xy", 6.0, "")
    assert mv.keys_up == mv.keys_down == mv.keys_left == mv.keys_right == []
    assert m.Move(kind="move", id="p", body="b", mode="heading", max_force="=p.f", max_speed=2.0).max_force == "=p.f"
    bal = m.Balance(kind="balance", id="bal", body="torso", stiffness=100.0, max_torque=100.0)
    assert (bal.damping, bal.target) == (0.0, 0.0)
    for cls, kw in ((m.Move, {"kind": "move", "id": "p", "body": "b", "max_force": 1.0}),          # max_speed 없음
                    (m.Move, {"kind": "move", "id": "p", "body": "b", "max_force": 1.0, "max_speed": 1.0, "mode": "z"}),
                    (m.Balance, {"kind": "balance", "id": "p", "body": "b", "stiffness": 1.0}),      # max_torque 없음
                    (m.Balance, {"kind": "balance", "id": "p", "body": "b", "stiffness": 1.0, "max_torque": 1.0, "x": 1})):
        with pytest.raises(ValueError):
            cls(**kw)


def test_new_parts_make_a_format_4_file_and_old_files_stay_low() -> None:
    base = _lander()
    env = parse_env(base)
    assert env.version == 4 and feature_version(env) == 1 and json.loads(dump_env(env))["version"] == 1
    assert "view" not in json.loads(dump_env(env))["world"] and "damping" not in json.loads(dump_env(env))["world"]
    body = next(b["id"] for b in base["bodies"] if b.get("owner", "agent") == "agent")
    with_balance = json.loads(json.dumps(base))
    with_balance["actuators"].append({"kind": "balance", "id": "bal62", "body": body, "stiffness": 50.0, "max_torque": 80.0})
    env4 = parse_env(with_balance)
    saved = json.loads(dump_env(env4))
    assert feature_version(env4) == 4 and saved["version"] == 4
    assert saved["world"]["view"] == "side" and saved["world"]["damping"] == 1.0  # 형식 4 글에는 형식 4 칸이 기본값이어도 있다
    assert env_hash(env4) != env_hash(env)
    saved["actuators"].pop()  # 부품을 지우면 다시 형식 1, 글도 처음과 같다
    assert dump_env(parse_env(saved)) == dump_env(env)


def test_error_path_of_new_actuators_hides_the_kind_tag() -> None:
    data = _lander()
    data["actuators"].append({"kind": "move", "id": "push", "body": "x", "max_force": 1.0, "max_speed": 1.0, "mode": "fly"})
    with pytest.raises(EnvFileError) as exc:
        parse_env(data)
    wheres = [i.where for i in exc.value.issues]
    assert wheres == [f"actuators[{len(data['actuators']) - 1}].mode"], wheres  # "…[n].move.mode"가 아니다


# ---------------------------------------------------------------------------
# Issue.code (설계 3.1: dataclass 한 줄)
# ---------------------------------------------------------------------------
def test_issue_code_is_an_optional_last_field() -> None:
    assert [f.name for f in dataclasses.fields(Issue)] == ["severity", "where", "message", "code"]
    old = Issue("error", "bodies[0]", "글")
    assert old.code == "" and str(old) == "bodies[0]: 글" and str(Issue("warning", "", "글", "no_sensors")) == "글"
    assert Issue("info", "a", "b", code="top_gravity").code == "top_gravity"
    assert old == Issue("error", "bodies[0]", "글", "")  # 코드가 비면 옛 값과 같다
    with pytest.raises(dataclasses.FrozenInstanceError):
        old.code = "x"  # type: ignore[misc]
    err = EnvFileError([Issue("error", "w", "하나", "unknown_ref"), Issue("warning", "w", "둘")])
    assert str(err) == S.SPEC_INVALID_FILE.format(count=1) + "\nw: 하나"  # 글에는 코드가 들지 않는다


# ---------------------------------------------------------------------------
# T7(바탕 몫): for_new_run과 editor.train (설계 D31)
# ---------------------------------------------------------------------------
def _spec_with_train(train: Any) -> Any:
    data = _lander()
    data.setdefault("editor", {})["train"] = train
    return compile_env(data)


def test_preset_train_keys_are_the_white_list() -> None:
    assert PRESET_TRAIN_KEYS == ("qualify_entrants", "qualify_trials", "restart_patience", "restart_from")
    assert set(PRESET_TRAIN_KEYS) <= set(TrainConfig.__dataclass_fields__) and set(PRESET_TRAIN_KEYS) == set(NEW_RUN_DEFAULTS)


def test_for_new_run_without_editor_train_is_unchanged() -> None:
    plain = TrainConfig.for_new_run()
    assert plain.to_dict() == TrainConfig(**NEW_RUN_DEFAULTS).to_dict()
    for stem in OLD_BUILTINS:
        spec = load_env(paths.builtin_env_dir() / f"{stem}.evoenv")
        assert "train" not in spec.file.editor
        got = TrainConfig.for_new_run(spec).to_dict()
        if stem == "wheels":  # v2.1: 있음 유전자가 있는 환경만 구조 뒤집기 권장값
            assert {k: v for k, v in got.items() if got[k] != plain.to_dict()[k]} == {
                "ga_structure_flip": 0.05, "sa_structure_flip": 0.05}
        else:
            assert got == plain.to_dict(), stem


def test_for_new_run_matches_the_frozen_release_for_the_old_builtins() -> None:
    """예제 6개는 얼린 v6.1.0과 같은 사전(설정 키도 같다: 혼자 하는 실행의 run.json 키가 그대로)."""
    script = "tests/oracle/scripts/new_run_defaults.py"
    old = json.loads(str(frozen.frozen_run(RELEASE, script, *OLD_BUILTINS)["configs"]))
    new = json.loads(str(frozen.live_run(script, *OLD_BUILTINS)["configs"]))
    assert list(old) == ["", *OLD_BUILTINS] and old == new
    assert set(old["lander"]) == set(TrainConfig.for_new_run().to_dict())


def test_editor_train_adds_only_white_listed_keys() -> None:
    spec = _spec_with_train({"qualify_entrants": 3, "qualify_trials": 9600, "restart_patience": 20000,
                             "restart_from": "champion", "popsize": 64, "method": "ga", "vs_pool_size": 8, "": 1})
    c = TrainConfig.for_new_run(spec)
    assert (c.qualify_entrants, c.qualify_trials, c.restart_patience, c.restart_from) == (3, 9600, 20000, "champion")
    assert (c.popsize, c.method) == (TrainConfig().popsize, TrainConfig().method)  # 흰 목록 밖의 키는 얹지 않는다
    c.validate()
    assert c.to_dict()["qualify_entrants"] == 3 and c.to_dict()["qualify_trials"] == 9600  # run.json의 config에 그대로 적힌다
    assert set(c.to_dict()) == set(TrainConfig.for_new_run().to_dict())  # 키는 늘지 않는다
    # 부르는 쪽 값이 프리셋 권장값보다 먼저다(실험실 폼, 명령줄 인자)
    c2 = TrainConfig.for_new_run(spec, qualify_entrants=1, run_seed=7)
    assert (c2.qualify_entrants, c2.qualify_trials, c2.run_seed) == (1, 9600, 7)
    # editor는 해시 밖: 권장 설정을 고쳐도 환경 버전이 그대로다
    assert env_hash(spec.file) == env_hash(compile_env(_lander()).file)


@pytest.mark.parametrize(("train", "expect"), [
    ({"qualify_entrants": 0}, {}), ({"qualify_entrants": MAX_QUALIFY_ENTRANTS + 1}, {}), ({"qualify_entrants": 2.0}, {}),
    ({"qualify_entrants": True}, {}), ({"qualify_entrants": "3"}, {}), ({"qualify_trials": 0}, {}),
    ({"qualify_trials": None}, {}), ({"restart_patience": -5}, {}), ({"restart_patience": 1.5}, {}),
    ({"restart_from": "nope"}, {}), ({"restart_from": 3}, {}),
    ({"restart_patience": None}, {"restart_patience": None}),            # None = 끔(올바른 값)
    ({"restart_from": "alternate"}, {"restart_from": "alternate"}),
    ({"qualify_entrants": 3, "qualify_trials": "many"}, {"qualify_entrants": 3}),   # 틀린 키만 빠진다
    ({"restart_patience": 500}, {}),                                      # 검증 주기(1,000)보다 짧음: validate()가 거절할 값
    ({"qualify_entrants": 3, "qualify_trials": 500}, {}),                # 출전자마다 한 번은 검증받아야 함: 예선 통째로 뺌
    ({"qualify_trials": 500}, {"qualify_trials": 500}),                   # 예선이 꺼져 있으면(출전자 1) 그 값은 쓰이지 않아 무해
])
def test_wrong_editor_train_values_are_dropped_silently(train: dict[str, Any], expect: dict[str, Any]) -> None:
    c = TrainConfig.for_new_run(_spec_with_train(train))
    want = {**TrainConfig.for_new_run().to_dict(), **expect}
    assert c.to_dict() == want
    c.validate()  # 권장값 때문에 학습 시작이 거절되는 일은 없다


@pytest.mark.parametrize("train", [None, [], "x", 3, {"train": {}}, [["qualify_entrants", 3]]])
def test_editor_train_of_the_wrong_shape_is_ignored(train: Any) -> None:
    assert TrainConfig.for_new_run(_spec_with_train(train)).to_dict() == TrainConfig.for_new_run().to_dict()


def test_editor_train_respects_the_callers_validation_period() -> None:
    spec = _spec_with_train({"qualify_entrants": 3, "qualify_trials": 9600, "restart_patience": 5000})
    c = TrainConfig.for_new_run(spec, validate_every=6000)   # 재시작 기준 5,000 < 6,000: 그 키만 빠진다
    assert (c.qualify_entrants, c.qualify_trials, c.restart_patience) == (3, 9600, None)
    c.validate()
    c = TrainConfig.for_new_run(spec, validate_every=20000)  # 예선 9,600 < 20,000: 예선도 빠진다
    assert (c.qualify_entrants, c.restart_patience) == (1, None)
    c.validate()
    c = TrainConfig.for_new_run(spec, qualify_trials=300)    # 부르는 쪽이 준 값과 권장 출전자 수가 맞지 않으면 권장 예선을 뺀다
    assert (c.qualify_entrants, c.qualify_trials) == (1, 300)
    c.validate()


def test_only_for_new_run_reads_editor_train() -> None:
    """이 칸을 읽는 곳은 for_new_run 하나뿐이다(런타임·학습기·저장된 실행은 읽지 않는다: 설계 D31)."""
    readers = [p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py")
               if "editor_train(" in p.read_text(encoding="utf-8") and p.name != "editor_meta.py"]
    assert readers == ["engine/trainer.py"]
    runtime = [p for p in (SRC / "world").rglob("*.py")] + [p for p in (SRC / "learners").rglob("*.py")]
    assert not [p for p in runtime if "editor_meta" in p.read_text(encoding="utf-8")]


def test_editor_meta_readers_never_raise() -> None:
    assert (editor_meta.TRAIN_KEY, editor_meta.RULES_KEY, editor_meta.RECIPE_KEY) == ("train", "rules", "recipe")
    for bad in (None, [], "x", {"train": 3, "rules": {"a": 1}, "recipe": []}):
        assert editor_meta.editor_train(bad) == {}  # type: ignore[arg-type]
        assert editor_meta.editor_rules(bad) == []  # type: ignore[arg-type]
        assert editor_meta.editor_recipe(bad) is None  # type: ignore[arg-type]
    editor = {"train": {"qualify_entrants": 3}, "rules": [{"kind": "fall", "event": "fall", "blocks": ["s_fall"]}, "x", 3],
              "recipe": {"recipe": 1, "built": "abc"}, "view": [0, 0, 1, 1]}
    train, rules, recipe = editor_meta.editor_train(editor), editor_meta.editor_rules(editor), editor_meta.editor_recipe(editor)
    assert train == {"qualify_entrants": 3} and rules == [{"kind": "fall", "event": "fall", "blocks": ["s_fall"]}]
    assert recipe == {"recipe": 1, "built": "abc"}
    train["x"] = 1  # 돌려준 것은 사본이다
    rules[0]["kind"] = "y"
    assert editor["train"] == {"qualify_entrants": 3} and editor["rules"][0]["kind"] == "fall"
    assert editor_meta.editor_view_box(editor) == (0.0, 0.0, 1.0, 1.0)  # 옛 읽기 함수는 그대로


# ---------------------------------------------------------------------------
# T18: paths
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("name", "stem"), [
    ("내 달리기", "내 달리기"), ("  내 달리기  ", "내 달리기"), ("a/b\\c:d*e?f\"g<h>i|j", "a_b_c_d_e_f_g_h_i_j"),
    ("탭\t줄\n끝\x00", "탭_줄_끝_"), ("끝의 점...", "끝의 점"), ("끝의 빈칸과 점 . . ", "끝의 빈칸과 점"), ("...", None),
    ("", None), ("   ", None), ("CON", "CON_"), ("con", "con_"), ("NUL.txt", "NUL_.txt"), ("Com1", "Com1_"), ("LPT9 ", "LPT9_"),
    ("COM0", "COM0"), ("CONSOLE", "CONSOLE"), ("달리기 2", "달리기 2"), ("a.b", "a.b"), ("???", "___"),
])
def test_safe_file_stem(name: str, stem: str | None) -> None:
    want = S.PRESET_NAME_FALLBACK if stem is None else stem
    got = paths.safe_file_stem(name)
    assert got == want
    assert paths.safe_file_stem(got) == got  # 다시 넣어도 그대로
    assert not set(got) & set('\\/:*?"<>|') and not any(ord(ch) < 32 for ch in got)
    assert got == got.strip() and not got.endswith(".") and 1 <= len(got) <= paths.SAFE_STEM_MAX


def test_safe_file_stem_cuts_long_names_at_80() -> None:
    assert paths.SAFE_STEM_MAX == 80
    long = "가" * 200
    assert paths.safe_file_stem(long) == "가" * 80
    assert paths.safe_file_stem("x" * 79 + ". 뒤") == "x" * 79  # 자른 끝이 점·빈칸이면 뗀다
    assert paths.safe_file_stem("aux." + "y" * 200) == ("aux_." + "y" * 200)[:80]
    assert paths.safe_file_stem(S.PRESET_NAME_FALLBACK) == S.PRESET_NAME_FALLBACK  # 대신 쓰는 이름도 안전한 이름이다


def test_template_and_recipe_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    assert paths.user_template_dir() == tmp_path / "data" / "templates"
    assert paths.template_dirs() == [paths.bundle_dir("assets") / "templates", tmp_path / "data" / "templates"]  # 내장 → 사용자
    assert paths.recipe_dir() == paths.bundle_dir("assets") / "recipes"
    assert not (tmp_path / "data").exists()  # 폴더를 만들지 않는다(읽기만)
    monkeypatch.setattr("sys._MEIPASS", str(tmp_path / "bundle"), raising=False)  # exe로 묶였을 때는 배포본 안
    assert paths.recipe_dir() == tmp_path / "bundle" / "assets" / "recipes"
    assert paths.template_dirs()[0] == tmp_path / "bundle" / "assets" / "templates"


def test_builtin_order_puts_the_sport_presets_after_the_old_six() -> None:
    assert paths.BUILTIN_ORDER == (*OLD_BUILTINS, "running")
    stems = [p.stem for p in paths.builtin_envs()]
    assert stems[:6] == list(OLD_BUILTINS)
    assert set(stems[6:]) <= {"running"}  # 6.2.0의 종목 프리셋은 달리기 하나(묶음 PM이 넣는다. 그 전에는 없다)


# ---------------------------------------------------------------------------
# T19(바탕 몫): 문자열 장부
# ---------------------------------------------------------------------------
LEDGER = json.loads((DATA / "strings_ledger.json").read_text(encoding="utf-8"))
"""설계 3.5의 장부(scratchpad synth/strings/strings_ledger.json의 사본). 사본에서 고친 것 둘: RULE_TITLES·RULE_SENTENCES의 「값 바꿈」 줄의
단계 6.4.0 → 6.2.0(장부를 만드는 스크립트가 원문의 「# 6.4」 꼬리표를 따라 단계를 잘못 붙였다. 설계 3.5의 표는 이 둘을 6.2.0 바탕에 적는다)."""
LATER = ("6.3.0-A", "6.3.0-EP", "6.3.0-B", "6.4.0")
NOT_YET_IN_HELP_TIPS = "HELP_TIPS.update()"
"""장부의 이 줄(HELP_TIPS에 열세 키를 제자리로 더함)은 묶음 H가 연결표(app/help/anchors.py)와 함께 한다: 키가 연결표에 있어야 하고
(tests/test_v3_strings.py) 연결표의 위젯은 W·ED1·ED2·Hb가 만든다. 값이 될 상수는 이미 있다(아래 테스트)."""


def _names(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in tree.body:
        targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets if isinstance(node, ast.Assign) else []
        out |= {t.id for t in targets if isinstance(t, ast.Name)}
    return out


def _old_names() -> set[str]:
    old = frozen.frozen_tree(RELEASE) / "src" / "evolab" / "app" / "strings.py"
    return _names(ast.parse(old.read_text(encoding="utf-8")))


def _new_names() -> set[str]:
    now = _names(ast.parse((SRC / "app" / "strings.py").read_text(encoding="utf-8")))
    return {n for n in now - _old_names() if n.isupper()}


def _has(row_name: str) -> bool:
    """장부의 이름이 strings.py에 있는가. `사전[키]`·`EDP_FIELD[모델.칸]`·`사전[키.*·키.*]` 꼴은 그 사전의 항목으로 본다."""
    match = re.fullmatch(r"([A-Z_]+)\[(.+)\]", row_name)
    if match is None:
        return hasattr(S, row_name)
    base, key = match.groups()
    if base == "EDP_FIELD":
        return key in S.EDP_FIELD_NAMES and key in S.EDP_FIELD_TIPS
    table = getattr(S, base, None)
    if not isinstance(table, dict):
        return False
    prefixes = [k.removesuffix("*") for k in key.split("·")]
    return all(any(name.startswith(p) for name in table) if p.endswith(".") else p in table for p in prefixes)


def test_ledger_has_the_design_stages() -> None:
    stages = {r["stage"] for r in LEDGER}
    assert stages == {"6.2.0", "6.2.0 통합", *LATER, "6.3.0 통합", "뺌"}
    assert sum(r["stage"] == "6.2.0" for r in LEDGER) == 197  # 설계 3.5의 195줄 + 단계를 바로잡은 둘(RULE_TITLES·RULE_SENTENCES)
    assert {r["name"] for r in LEDGER if r["stage"] == "6.2.0" and r["name"].startswith("RULE_")} >= {"RULE_TITLES", "RULE_SENTENCES"}


def test_every_620_ledger_name_exists() -> None:
    rows = [r for r in LEDGER if r["stage"] == "6.2.0"]
    missing = [r["name"] for r in rows if r["name"] != NOT_YET_IN_HELP_TIPS and not _has(r["name"])]
    assert missing == []
    assert "PRESET_PROBE_INSTANT" not in vars(S) and hasattr(S, "PRESET_ALIVE_INSTANT")  # 이름 바꿈
    # HELP_TIPS에 더할 열세 키의 값이 될 상수는 모두 있다(키는 묶음 H가 연결표와 함께)
    for name in ("ED_NEW_PRESET_TIP", "RULES_ADD_TIP", "EDP_ISSUES_HINT", "EDC_OWNER_AGENT_TIP", "EDC_OWNER_WORLD_TIP",
                 "EDP_WORLD_KIND_TIP", "PRESET_STEPS_TIP", "PRESET_PREVIEW_TIP", "PRESET_NAME_TIP", "PRESET_BTN_EDIT_TIP",
                 "PRESET_BTN_PLAY_TIP", "PRESET_BTN_LAB_TIP"):
        assert isinstance(getattr(S, name), str) and getattr(S, name).strip(), name


def test_dropped_and_later_stage_names_are_not_in_strings_yet() -> None:
    new = _new_names()
    now_or_earlier = {r["name"].split("[")[0].split(".")[0] for r in LEDGER if r["stage"] in ("6.2.0", "6.2.0 통합")}
    dropped = {r["name"] for r in LEDGER if r["stage"] == "뺌"}
    later = {r["name"].split("[")[0].split(".")[0] for r in LEDGER if r["stage"] in LATER} - now_or_earlier
    assert sorted(new & dropped) == []
    assert sorted(new & later) == []
    # 뒤 단계의 사전 항목도 아직 없다(6.3.0의 다른 선수 보기·승패 블록, 6.4.0의 잡기·무작위 신호)
    for key in ("other", "result", "grip", "signal", "per.trial", "side.team", "sides.symmetric"):
        assert key not in S.EDP_VALUE_NAMES, key
    for key in ("Match.sides", "Other.who", "RandVar.per", "Region.side", "Grip.joint", "Player.team"):
        assert key not in S.EDP_FIELD_NAMES and key not in S.EDP_FIELD_TIPS, key


def _texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [x for v in value.values() for x in _texts(v)]
    if isinstance(value, (tuple, list)):
        return [x for v in value for x in _texts(v)]
    return []


def test_new_strings_follow_the_word_table() -> None:
    """설계 3.4: 틀의 조절 값은 「슬라이더」(「미끄럼」 아님), 뇌가 움직이는 몸은 「에이전트」(「선수 몸」 아님), Move는 「밀기(이동)」."""
    new = sorted(_new_names())
    assert len(new) >= 180  # 장부의 새 이름 + 다듬기 U4의 새 상수
    for name in new:
        for text in _texts(getattr(S, name)):
            assert "미끄럼" not in text and "선수 몸" not in text, name
    for text in [S.EDP_VALUE_NAMES["move"], S.EDC_TOOL_MOVE, S.SPEC_MOVE_SPEED, S.EDP_FIELD_NAMES["Move.mode"]]:
        assert "움직임" not in text  # 「움직임」은 물체 종류(dynamic)의 말
    assert S.EDP_VALUE_NAMES["move"] == S.EDC_TOOL_MOVE == "밀기(이동)"
    assert S.EDP_FIELD_NAMES["World.view"] == S.EDP_WORLD_KIND == "세계 종류" and "바닥 마찰" in S.EDP_FIELD_NAMES["World.damping"]
    assert S.EDP_VALUE_NAMES["view.side"] == S.EDP_WORLD_KINDS["side"] == S.TPL_WORLD_NAMES["side"]
    assert S.EDP_VALUE_NAMES["view.top"] == S.EDP_WORLD_KINDS["top"] == S.TPL_WORLD_NAMES["top"]


def test_no_two_new_constants_hold_the_same_text() -> None:
    """같은 글을 가진 상수는 하나만 둔다(설계 3.4: 「실험실로 보내기」는 지금의 ED_TO_LAB, 창 제목·메뉴·단추는 PRESET_TITLE 하나)."""
    by_text: dict[str, list[str]] = {}
    for name in sorted(_new_names()):
        value = getattr(S, name)
        if isinstance(value, str):
            by_text.setdefault(value, []).append(name)
    assert {t: n for t, n in by_text.items() if len(n) > 1} == {}
    for gone in ("PRESET_BTN_LAB", "MENU_NEW_PRESET", "ED_BUILTIN_DRILL"):  # 장부의 「뺌」: 같은 글의 겹친 상수
        assert not hasattr(S, gone), gone


def _fields(text: str) -> set[str]:
    return {f.split(".")[0].split("[")[0] for _, f, _, _ in string.Formatter().parse(text) if f}


def test_new_strings_fill_with_dummy_values_and_keep_their_placeholders() -> None:
    errors: list[str] = []
    for name in sorted(_new_names()):
        for text in _texts(getattr(S, name)):
            try:
                text.format(**dict.fromkeys(_fields(text), 1))  # 정수 1은 d·f·, 꼴 모두 받는다
            except (KeyError, ValueError, IndexError) as exc:
                errors.append(f"{name}: {exc!r}")
    assert errors == []
    assert _fields(S.ED_STATUS) == {"errors", "warnings", "infos"}
    assert _fields(S.HINT_EDITOR_EMPTY) == {"preset", "shapes", "builtin"}
    assert _fields(S.PRESET_STATUS_OK) == {"inputs", "outputs"} and _fields(S.PRESET_ALIVE_STILL_END) == {"t", "end"}
    assert _fields(S.LAB_NOTICE_FIRST_GEN) == {"pop"} and _fields(S.ED_WHY_ERRORS) == {"n"}
    assert _fields(S.MAIN_TITLE_RUN) == {"env", "state", "gen", "app"} and _fields(S.MAIN_CHIP) == {"state", "gen"}


def test_texts_that_name_a_button_follow_that_buttons_name() -> None:
    """단추·탭 이름을 적은 글은 그 이름과 어긋나지 않는다(이어 붙였거나, 장부의 글 그대로면 여기서 지킨다)."""
    for holder, named in (
        (S.ED_STATUS, S.EDP_TAB_ISSUES), (S.ED_HAS_ERRORS, S.EDP_TAB_ISSUES), (S.ED_HAS_ERRORS, S.FIX_BUTTON),
        (S.EDP_ISSUES_HINT, S.FIX_BUTTON), (S.EDP_ISSUES_HEADERS[3], S.FIX_BUTTON), (S.EDP_BRAIN_NO_CHANNELS, S.EDP_TAB_ISSUES),
        (S.ED_WHY_ERRORS, S.EDP_TAB_ISSUES), (S.RULES_EMPTY, S.RULES_ADD), (S.RULES_FORMULA_NOTE, S.EDP_TAB_SCORING),
        (S.PRESET_BTN_LAB_TIP, S.LAB_START), (S.TRAIN_GROW_NO_IPOP, S.LAB_IPOP), (S.VIDEO_NOTHING, S.LAB_VERSUS),
        (S.RVL_CONTINUE_BUSY_TIP, S.LAB_VERSUS), (S.LAB_ENV_EDIT_TIP, S.ED_TO_LAB), (S.MAIN_ENV_SENT, S.LAB_CONTINUE),
        (S.LAB_TIMELINE_TIP, S.LAB_TIMELINE_PLAY), (S.LAB_TIMELINE_TIP, S.DASH_PAGES["fun"]),
        (S.LAB_STOP_APPLY_NOTE, S.LAB_APPLY), (S.LAB_WHY_SPECTATE_ONLY, S.LAB_SPECTATE), (S.LAB_WHY_FAST_ONLY, S.LAB_FAST),
        (S.FIX_TIPS["add_rule"], S.EDP_TAB_RULES), (S.LAB_ENV_GROUPS_TIP["mine"], S.ED_NEW_PRESET.rstrip("…")),
        (S.SETTINGS_PRESET_USAGE_LOG, S.ED_NEW_PRESET.rstrip("…")),
    ):
        assert named in holder, (holder, named)
    assert S.LAB_CONTINUE == S.REC_RESUME  # 같은 일은 화면이 달라도 같은 말(다듬기 1.5)
    assert S.LAB_START == "▶ 학습 시작" and S.LAB_START != S.LAB_RESTART
    assert S.LAB_HUD_LIVE_SKIP.startswith(S.LAB_HUD_LIVE)


def test_rule_fix_and_template_tables_have_matching_keys() -> None:
    rules = ("distance", "fall", "touch_ends", "touch_penalty", "reach", "near", "time_limit", "survive", "upright",
             "save_energy", "out_ends")
    assert tuple(S.RULE_TITLES) == rules and set(S.RULE_SENTENCES) == set(rules)  # 6.2.0의 규칙 열하나(대전 규칙은 6.3.0-B)
    slots = set().union(*(_fields(t) for t in S.RULE_SENTENCES.values()))
    assert slots == set(S.RULE_SLOT_TIPS)  # 문장의 빈칸마다 풍선
    assert set(S.RULE_SUMMARY_TERMS) - {"other"} <= set(rules)
    assert set(S.FIX_LABELS) == set(S.FIX_TIPS) and len(S.FIX_LABELS) == 14
    assert set(S.TPL_NAMES) == set(S.TPL_BLURBS) == {"biped", "disc", "ground_flat", "wall", "ball", "goal", "track_side",
                                                     "yard_top"}
    assert set(S.TPL_PARAM_LABELS) == set(S.TPL_PARAM_TIPS)  # 슬라이더마다 이름과 풍선(D55)
    assert len(S.PRESET_STEPS) == len(S.PRESET_STEP_TIPS) == 5 and len(S.PRESET_COACH) == 3
    assert set(S.EDP_WORLD_KINDS) == set(S.EDP_WORLD_KIND_TIPS) == {"side", "top", "custom"}
    assert set(S.LAB_ENV_GROUPS) == set(S.LAB_ENV_GROUPS_TIP) == {"mine"}
    assert set(S.PRESET_ALIVE_FIXES) == {"add_rule", "move_part"}
    assert set(S.EDC_WORLD_MOTION) == {"static", "dynamic"} and set(S.EDC_CURSOR_OWNER) == {"agent", "world"}
    assert len(S.EDP_ISSUES_HEADERS) == 4 and set(S.EDP_ISSUES_SEVERITY) == {"error", "warning", "info"}
    from evolab.app.lab.dashboard import TILE_IDS

    assert set(S.DASH_TILE_EMPTY) == set(TILE_IDS)  # 「학습 그래프」 쪽 타일마다 빈 문장(다듬기 U18)


def test_every_new_model_field_has_a_name_and_a_tooltip() -> None:
    """새 칸마다 풍선: 형식 4의 칸은 「모델.칸」 또는 칸 이름으로 화면 이름과 도움말이 있다(에디터 폼이 같은 순서로 찾는다)."""
    for cls, names in ((m.World, ["view", "damping"]), (m.Move, list(m.Move.model_fields)),
                       (m.Balance, list(m.Balance.model_fields))):
        for name in names:
            if name == "kind":
                continue
            keys = (f"{cls.__name__}.{name}", name)
            assert any(k in S.EDP_FIELD_NAMES for k in keys), keys
            assert any(k in S.EDP_FIELD_TIPS for k in keys), keys
    for value in ("move", "balance", "mode.xy", "mode.x", "mode.heading", "view.side", "view.top"):
        assert value in S.EDP_VALUE_NAMES, value


def test_help_has_the_preset_page_and_the_new_glossary_terms() -> None:
    assert list(S.HELP_PAGES).index("preset") == list(S.HELP_PAGES).index("editor") + 1 and len(S.HELP_PAGES) == 13
    title, text = S.HELP_PAGES["preset"]
    assert title == S.PRESET_TITLE and "## 다섯 단계" in text and "{{ED_TO_LAB}}" in text
    assert {"프리셋", "몸 틀", "규칙", "위에서 본 세계", "균형 보조"} <= set(S.HELP_GLOSSARY_TERMS)
    editor = S.HELP_PAGES["editor"][1]
    for heading in ("## {{EDP_TAB_RULES}}", "## {{EDP_WORLD_KIND}}", "## {{EDP_TAB_ISSUES}}", "## 처음 만들 때"):
        assert heading in editor, heading
    assert "evolab://help/preset" in editor and "## 달리기" in S.HELP_PAGES["examples"][1]


# ---------------------------------------------------------------------------
# 공용 시험 자료
# ---------------------------------------------------------------------------
def test_measured_running_file_is_the_prototype_byte_for_byte() -> None:
    """측정에 쓴 원형 파일 그대로다(리드 결정 LD1의 2: px/PX0b/envs/biped_bal100_fall5.evoenv, 넘어짐 감점 −5). G1의 견본:
    조리법으로 만든 달리기가 이것과 물리 동일해야 한다(D29)."""
    import hashlib

    raw = MEASURED.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "f28fdfe212aabf06b0b2dfa32fd76e8e81b99121db8f20add3c7c8345b532069"
    data = json.loads(raw)
    assert data["version"] == 1 and "editor" not in data  # scratch 패치로 만든 파일: 적힌 형식은 1, 키·권장 설정 없음
    spec = load_env(MEASURED)
    assert feature_version(spec.file) == 4  # 균형 보조 → 형식 4(얼린 v6.1은 열지 못한다)
    balance = [a for a in spec.file.actuators if isinstance(a, m.Balance)]
    assert [(b.body, b.stiffness, b.damping, b.max_torque, b.target) for b in balance] == [("torso", 100.0, 12.5, 100.0, 0.0)]
    assert len(spec.inputs) == 13 and len(spec.outputs) == 6 and layout_for(spec).dim == 326  # DE 4.9-(1)
    assert len(spec.file.bodies) == 8 and len(spec.file.joints) == 6  # 땅 + 몸 일곱, 관절 여섯
    assert [e.id for e in spec.file.events] == ["fall"] and "s_fall" in [b.id for b in spec.file.scoring.blocks]
    ground = spec.file.bodies[0]
    assert (ground.id, ground.tags, ground.owner) == ("ground", ["ground"], "world")
    assert spec.file.world.max_time == 15.0 and not [i for i in spec.issues if i.severity == "error"]
    # 얼린 v6.1.0은 이 파일을 「모르는 종류」로 거절한다(적힌 형식이 1이라 형식 문구가 아니다: 형식 4로 저장한 글의 거절은 묶음 R의 T9)
    assert MEASURED.relative_to(ROOT).as_posix() not in frozen.frozen_envs(RELEASE)


class _T:
    """틀 수식의 이름 `t.<슬라이더 id>`."""

    def __init__(self, values: dict[str, float]) -> None:
        self.__dict__.update(values)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _calc(node: ast.AST, t: _T) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_calc(node.left, t), _calc(node.right, t))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_calc(node.operand, t))
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "t":
        return float(getattr(t, node.attr))
    raise AssertionError(ast.dump(node))


def _expand(value: Any, t: _T, slots: dict[str, str]) -> Any:
    """시험용 펼치기(묶음 P의 presets.template.expand가 할 일의 가장 작은 본보기): `=t.*` 수식은 숫자로, `$빈자리`는 태그 참조로."""
    if isinstance(value, str):
        if value in slots:
            return slots[value]
        if value.startswith("=") and "t." in value:
            return round(_calc(ast.parse(value[1:], mode="eval").body, t), 6)
        return value
    if isinstance(value, list):
        return [_expand(v, t, slots) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v, t, slots) for k, v in value.items()}
    return value


TEMPLATE_KEYS = {"format", "version", "id", "kind", "world", "name", "description", "params", "root", "parts", "slots", "keys",
                 "evolvable", "spawns", "rules", "randoms", "spawn"}
"""틀 파일의 칸: 부분 설계 DU 2.2 + 최종 설계 3.1(randoms, spawn). 내장 틀은 여기에 name_id(와 슬라이더의 label_id)."""
PARAM_KEYS = {"id", "label", "value", "min", "max", "step", "unit", "main"}
GROUND = {"id": "ground", "label": "땅", "owner": "world", "type": "static", "tags": ["ground"],
          "shapes": [{"kind": "chain", "points": [[-10.0, 0.0], [60.0, 0.0]], "radius": 0.05}]}


def _template(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.evotpl").read_text(encoding="utf-8"))


def _corners(tpl: dict[str, Any]) -> list[dict[str, float]]:
    params = tpl["params"]
    return [{p["id"]: p[which] for p in params} for which in ("value", "min", "max")]


def _env_from(tpl: dict[str, Any], values: dict[str, float], world: dict[str, Any], extra_bodies: list[dict[str, Any]]) -> dict:
    slots = {name: "tag." + slot["tags"][0] for name, slot in tpl["slots"].items()}
    parts = _expand(tpl["parts"], _T(values), slots)
    spawn = _expand(tpl.get("spawn") or {}, _T(values), slots)
    return {"version": 1, "name": tpl["name"], "world": world, "randomize": list(tpl.get("randoms", [])),
            "agent": {"spawn": spawn}, "bodies": [*extra_bodies, *parts["bodies"]], "joints": parts["joints"],
            "actuators": parts["actuators"], "sensors": parts["sensors"],
            "scoring": {"blocks": [{"kind": "progress", "id": "s_go", "body": tpl["root"], "weight": 1.0}]}}


@pytest.mark.parametrize("name", ["fixture_walker", "fixture_puck"])
def test_fixture_templates_have_the_documented_shape(name: str) -> None:
    tpl = _template(name)
    assert set(tpl) <= TEMPLATE_KEYS and (tpl["format"], tpl["version"], tpl["id"], tpl["kind"]) == (
        "evolab-template", 1, name, "body")
    assert tpl["world"] == {"fixture_walker": "side", "fixture_puck": "top"}[name]
    assert "name_id" not in tpl and tpl["name"] and tpl["description"]  # 사용자 틀처럼 글을 직접 담는다(D55)
    for p in tpl["params"]:
        assert set(p) == PARAM_KEYS and p["min"] <= p["value"] <= p["max"] and p["main"] is True, p
    assert [p["id"] for p in tpl["params"]] == ["size", "power"]
    assert set(tpl["parts"]) <= {"bodies", "joints", "actuators", "sensors", "regions", "generators"}
    assert tpl["root"] in [b["id"] for b in tpl["parts"]["bodies"]]
    used = set(re.findall(r'"(\$[a-z_]+)"', json.dumps(tpl["parts"])))
    assert used == set(tpl["slots"])  # 쓰인 빈자리 = 빈자리 표
    for slot in tpl["slots"].values():
        assert set(slot) == {"tags", "label"} and isinstance(slot["tags"], list) and slot["tags"]  # 태그 여럿(D44)
    names = set(re.findall(r"\bt\.([a-z_]+)", json.dumps(tpl)))
    assert names == {"size", "power"}  # 수식의 이름은 슬라이더뿐


def test_fixture_walker_expands_to_a_format_1_env_at_every_slider_corner() -> None:
    tpl = _template("fixture_walker")
    assert tpl["slots"]["$ground"]["tags"] == ["ground", "terrain"]
    assert [r["id"] for r in tpl["randoms"]] == ["tilt"] and tpl["spawn"]["angle"] == "=r.tilt"
    for values in _corners(tpl):
        data = _env_from(tpl, values, {"max_time": 10.0}, [GROUND])
        assert "t." not in json.dumps(data) and "$" not in json.dumps(data)
        spec = compile_env(data)
        assert [i for i in spec.issues if i.severity == "error"] == []
        assert feature_version(spec.file) == 1  # 형식 1 부품만: 얼린 v6.1도 연다
        assert len(spec.inputs) == 9 and len(spec.outputs) == 2
    assert not [a for a in tpl["parts"]["actuators"] if a["kind"] in ("move", "balance")]


def test_fixture_puck_uses_format_1_parts_in_a_top_world() -> None:
    tpl = _template("fixture_puck")
    assert tpl["slots"] == {} and "spawn" not in tpl and "randoms" not in tpl
    assert [a["kind"] for a in tpl["parts"]["actuators"]] == ["thruster"] * 4
    assert sorted(k for a in tpl["parts"]["actuators"] for k in a["keys"]) == ["Down", "Left", "Right", "Up"]
    for values in _corners(tpl):
        parts_only = _env_from(tpl, values, {"gravity": [0.0, 0.0], "max_time": 8.0}, [])
        spec = compile_env(parts_only)
        assert [i for i in spec.issues if i.severity == "error"] == []
        assert feature_version(spec.file) == 1 and len(spec.outputs) == 4
        top = _env_from(tpl, values, {"gravity": [0.0, 0.0], "max_time": 8.0, "view": "top"}, [])
        assert feature_version(parse_env(top)) == 4  # 세계 종류만으로 형식 4(부품은 형식 1 그대로)
    assert tpl["parts"]["bodies"][0]["drag"] > 0  # 바닥 마찰 대신 물체의 항력(형식 1)으로 멈춘다


RECIPE_KEYS = {"recipe", "start", "name", "world", "arena", "roles", "props", "rules", "players", "brain", "evaluation",
               "world_values", "randoms", "match", "extra", "train", "status"}
"""조리법의 칸: DU 2.3 + 최종 설계 D28(build가 적는 built·templates와 내장 조리법 파일의 status는 여기 견본에 없다)."""


@pytest.mark.parametrize(("name", "world", "arena", "body"), [("fixture_side", "side", "track_side", "fixture_walker"),
                                                              ("fixture_top", "top", "yard_top", "fixture_puck")])
def test_fixture_recipes_have_the_documented_shape(name: str, world: str, arena: str, body: str) -> None:
    recipe = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    assert set(recipe) <= RECIPE_KEYS and {"recipe", "start", "name", "world", "arena", "roles", "props", "rules",
                                           "players"} <= set(recipe)
    assert (recipe["recipe"], recipe["world"], recipe["start"]) == (1, world, f"blank_{world}")
    assert recipe["arena"] == {"template": arena, "values": {}} and arena in S.TPL_NAMES  # 경기장은 묶음 P의 일반 틀
    (role,) = recipe["roles"]
    assert set(role) == {"id", "label", "body"} and role["body"]["template"] == body
    tpl = _template(body)
    assert tpl["world"] == world and set(role["body"]["values"]) == {p["id"] for p in tpl["params"]}
    assert recipe["players"] == {"mode": "solo", "seats": [{"role": role["id"], "count": 1, "spawn": "start"}]}  # D41: mirror 없음
    part_ids = {b["id"] for b in tpl["parts"]["bodies"]}
    for rule in recipe["rules"]:
        kind = rule["rule"]
        assert kind in S.RULE_SENTENCES, kind
        args = set(rule) - {"rule", "who"}
        assert args <= _fields(S.RULE_SENTENCES[kind]), (kind, args)  # 칸 이름 = 문장의 빈칸 이름
        assert rule.get("who", role["id"]) == role["id"]
        if "part" in rule:
            assert rule["part"] in part_ids
        if str(rule.get("what", "")).startswith("$"):
            assert rule["what"] in tpl["slots"]
    assert [r["rule"] for r in recipe["rules"]][-1] == "time_limit"
    if "train" in recipe:
        assert set(recipe["train"]) <= set(PRESET_TRAIN_KEYS)
    if "brain" in recipe:
        m.Brain(**recipe["brain"])
    if "evaluation" in recipe:
        m.Evaluation(**recipe["evaluation"])

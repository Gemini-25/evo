"""6.2.0 통합(최종 설계 4.1의 묶음 I): T41과 문자열 끝 손질, 윈도우 CI의 프리셋 단계.

- T41: 버전 6.2.0, 배포본에 assets/templates·assets/recipes와 내장 달리기, 종목 글 규칙(설계 5.2: 등급 ok·experimental·hidden이 카드·
  실험실 목록·글에 어떻게 나오는가).
- 종목 글(D47)은 잰 값으로만 쓴다(리드 결정 LD1의 5: 달리기는 새 시드 13~20의 가운데 값 5.5~19.8m·넘어짐 0~9%, 「10m」를 약속하지
  않는다). 3.5 장부의 「6.2.0 통합」 이름이 있다.
- strings.py 「# 6.2.0 프리셋 만들기」 블록부터 끝까지(바탕과 묶음 블록)의 상수는 모두 어딘가에서 쓰인다. 아직 쓰는 화면이 없는 것은
  PENDING에 까닭과 함께 적고(통합 보고서의 「다른 묶음이 할 일」), 그 목록 밖의 안 쓰인 상수는 실패다.
- 윈도우 CI(W2·W3): 콘솔 exe가 만든 달리기가 배포본의 running.evoenv와 같고 그 파일로 bench가 돈다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from evolab import __version__, paths
from evolab.app import strings as S
from evolab.presets import recipe as RC

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "evolab"
STRINGS = SRC / "app" / "strings.py"
WORKFLOW = ROOT / ".github" / "workflows" / "windows.yml"
LEDGER = json.loads((ROOT / "tests" / "data" / "strings_ledger.json").read_text(encoding="utf-8"))
HEADER = "\n# 6.2.0 프리셋 만들기"
SPORTS = ("running",)
"""6.2.0의 내장 종목(설계 2.1). 공 넣기 연습(drill)은 내장 파일이 없는 안내 창 카드다."""

PENDING = {
    "HINT_LAB_IDLE_SPORT": "Ha: help/hints.lab_idle_text가 내장 종목 환경이면 이 글을 쓴다",
    "SETTINGS_PRESET_USAGE_LOG": "Ha: 설정 메뉴의 체크(preset.wizard.set_usage_log_enabled, D49 「설정으로 끔」)",
    "SETTINGS_PRESET_USAGE_LOG_TIP": "Ha: 위 체크의 풍선",
    "PRESET_COACH": "Ha: 안내 창 처음 풍선(DU 4.8, 6.2.0에서 하지 않음)",
    "PRESET_SAVED": "Ha·ED2: 안내 창이 저장한 뒤 상태 줄 한 줄",
    "TPL_BUILTIN": "W: 보관함 창에서 내장 틀과 사용자 틀을 가르는 꼬리표",
    "PRESET_MORE_PARAMS": "W: 슬라이더가 셋보다 많은 틀의 「더 보기」(내장 틀은 모두 셋 이하라 6.2.0에는 보일 곳이 없음)",
    "PRESET_MORE_PARAMS_TIP": "W: 위 단추의 풍선",
    "MENU_FILE": "ED2: 에디터 「파일」 메뉴(6.2.0 에디터는 도구 줄만 씀)",
}
"""장부의 6.2.0 이름 가운데 아직 쓰는 화면이 없는 것: 상수는 장부대로 두고(T19) 쓰는 묶음이 생기면 여기서 지운다."""


# ---------------------------------------------------------------------------
# T41 버전·배포본
# ---------------------------------------------------------------------------
def test_version_is_620_everywhere() -> None:
    assert __version__ == "6.2.0"
    assert re.search(r'^version = "6\.2\.0"$', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), flags=re.M)
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    block = lock.split('[[package]]\nname = "evolab"\n', 1)[1].split("\n\n", 1)[0]
    assert 'version = "6.2.0"' in block


def test_bundle_spec_ships_templates_recipes_and_the_running_preset() -> None:
    spec = (ROOT / "packaging" / "evolab.spec").read_text(encoding="utf-8")
    assert '(str(ROOT / "src" / "evolab" / "assets"), "assets")' in spec  # 틀·조리법(assets 아래 전부)
    assert '(str(ROOT / "environments"), "environments")' in spec  # 내장 달리기
    templates = sorted(p.stem for p in (SRC / "assets" / "templates").glob("*.evotpl"))
    assert templates == ["ball", "biped", "disc", "goal", "ground_flat", "track_side", "wall", "yard_top"]
    recipes = sorted(p.stem for p in (SRC / "assets" / "recipes").glob("*.json"))
    assert recipes == ["blank_side", "blank_top", "drill", "running"]
    assert (paths.builtin_env_dir() / "running.evoenv").is_file()
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "src/evolab/assets/templates/*.evotpl text eol=lf" in attributes  # 해시가 줄 끝에 따라 바뀌지 않게(T14)
    assert "src/evolab/assets/recipes/*.json text eol=lf" in attributes


# ---------------------------------------------------------------------------
# T41 종목 글 규칙(설계 5.2)
# ---------------------------------------------------------------------------
def _status(name: str) -> str:
    return str(json.loads((paths.recipe_dir() / f"{name}.json").read_text(encoding="utf-8")).get("status", "ok"))


def test_statuses_and_where_each_sport_shows() -> None:
    starters = {s.id: s for s in RC.starters()}
    builtin_names = {p.stem for p in paths.builtin_envs()}
    examples = S.HELP_PAGES["examples"][1]
    for name in RC.builtin_names():
        status = _status(name)
        assert status in RC.STATUSES, name
        if status == "hidden":  # 카드 없음, 내장 파일 없음, 글 상수 없음
            assert name not in starters and name not in builtin_names
            assert name not in S.SPORT_BLURBS and name not in S.HINT_LAB_IDLE_SPORT
            continue
        assert name in starters, name
        card = starters[name]
        assert card.title and card.blurb, name  # 카드 제목과 한 줄(strings.py의 글)
        assert card.title == RC.text_of(card.title_id) and card.blurb == RC.text_of(card.blurb_id)
        assert card.status == status
    for sport in SPORTS:  # 내장 파일로 싣는 종목: 실험실 목록·도움말 머리·잰 값의 글
        assert _status(sport) in ("ok", "experimental") and sport in builtin_names
        name = json.loads((paths.builtin_env_dir() / f"{sport}.evoenv").read_text(encoding="utf-8"))["name"]
        assert f"\n## {name}\n" in examples
        assert sport in S.SPORT_BLURBS and sport in S.HINT_LAB_IDLE_SPORT
        assert RC.STARTER_TEXT[sport][1] == f"SPORT_BLURBS.{sport}"


def _section(md: str, heading: str) -> str:
    return md.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]


def test_running_texts_use_only_the_measured_values() -> None:
    """LD1의 5: 카드 한 줄·실험실 안내·도움말 종목 절은 C3의 잰 값으로만(새 시드 13~20: 8개 모두 통과, 가운데 값 5.5~19.8m,
    넘어짐 0~9%). 「10m」를 약속하지 않는다. 감점 −5·15초·예선 3 × 9,600은 조리법 그대로."""
    section = _section(S.HELP_PAGES["examples"][1], S.ED_BUILTIN_RUNNING)
    texts = [S.SPORT_BLURBS["running"], S.HINT_LAB_IDLE_SPORT["running"], section]
    for text in texts:
        assert not re.search(r"\b10\s*m", text), text
    assert "5.5~19.8m" in section and "0~9%" in section and "8개 모두" in section
    recipe = json.loads((paths.recipe_dir() / "running.json").read_text(encoding="utf-8"))
    fall = next(r for r in recipe["rules"] if r["rule"] == "fall")
    seconds = next(r for r in recipe["rules"] if r["rule"] == "time_limit")["seconds"]
    assert f"{fall['penalty']:g}점을 잃" in section and f"{seconds:g}초" in section
    assert f"{recipe['train']['qualify_trials']:,}회" in section and "셋" in section
    assert "15초에 5~20m" in S.SPORT_BLURBS["running"]  # 카드는 잰 범위를 반올림한 값
    assert S.HINT_LAB_IDLE_SPORT["running"].format(start=S.LAB_START, pop=32)


def test_integration_ledger_rows_exist() -> None:
    rows = [r["name"] for r in LEDGER if r["stage"] == "6.2.0 통합"]
    assert sorted(rows) == ["HINT_LAB_IDLE_SPORT", "SPORT_BLURBS"]
    assert all(isinstance(getattr(S, n), dict) for n in rows)


# ---------------------------------------------------------------------------
# 문자열 끝 손질: 6.2.0 블록의 상수는 모두 쓰인다
# ---------------------------------------------------------------------------
def _block_names() -> list[str]:
    text = STRINGS.read_text(encoding="utf-8")
    block = text[text.index(HEADER):]
    return list(dict.fromkeys(re.findall(r"^([A-Z][A-Z0-9_]+): Final", block, flags=re.MULTILINE)))


def _strings_code_without_definitions(names: list[str]) -> str:
    keep = []
    for line in STRINGS.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") or any(stripped.startswith(f"{n}: Final") for n in names):
            continue
        keep.append(line.split("  # ")[0])
    return "\n".join(keep)


def test_every_620_string_is_used_or_named_pending() -> None:
    names = _block_names()
    assert len(names) > 200 and {"SPORT_BLURBS", "PRESET_TITLE", "REP_SAVE_WHY_OFF", "CMP_MIXED_ENVS"} <= set(names)
    sources = [p for p in [*SRC.rglob("*.py"), *(ROOT / "packaging").rglob("*.py")]
               if p.name != "strings.py" and "__pycache__" not in p.parts]
    code = "\n".join(p.read_text(encoding="utf-8") for p in sources)
    # 조리법·틀은 화면 글을 이름(「사전.키」·name_id)으로 가리킨다(D55): 그 파일도 쓰임으로 센다
    assets = "\n".join(p.read_text(encoding="utf-8") for p in (SRC / "assets").rglob("*") if p.suffix in (".json", ".evotpl"))
    text = code + "\n" + assets + "\n" + _strings_code_without_definitions(names)
    unused = {n for n in names if not re.search(rf"\b{n}\b", text)}
    assert unused <= set(PENDING), sorted(unused - set(PENDING))
    assert set(PENDING) <= set(names)


def test_620_names_are_defined_once() -> None:
    text = STRINGS.read_text(encoding="utf-8")
    for n in _block_names():
        assert len(re.findall(rf"^{n}: Final", text, flags=re.MULTILINE)) == 1, n


def test_stage0_note_about_help_tips_is_current() -> None:
    text = STRINGS.read_text(encoding="utf-8")
    assert "HELP_TIPS에는 아직 더하지 않았다" not in text
    for key in ("editor.new_preset", "preset.lab", "lab.new_preset"):
        assert key in S.HELP_TIPS


# ---------------------------------------------------------------------------
# 윈도우 CI(W2·W3)
# ---------------------------------------------------------------------------
PRESET_STEP = "Sports preset files, preset command and bench with the console exe (Korean path)"


def test_workflow_has_the_sports_preset_exe_step() -> None:
    """PyYAML 없이 글로 본다(구조는 tests/test_installer_template.py가 PyYAML로). 한글 경로의 콘솔 exe로: 틀·조리법·running.evoenv가
    배포본에 있고, `preset running`의 결과가 배포본의 running.evoenv와 SHA-256까지 같고, 그 파일로 bench가 돌며, 앞 단계 자체
    점검 기록에 「프리셋」 점검의 통과 줄이 있다. 네이티브 명령 뒤에는 늘 종료 코드를 본다."""
    text = WORKFLOW.read_text(encoding="utf-8")
    package = text.split("\n  package:\n", 1)[1].split("\n  release:\n", 1)[0]
    assert package.count(f"- name: {PRESET_STEP}\n") == 1
    before, step = package.split(f"- name: {PRESET_STEP}\n", 1)
    step = step.split("\n      - name:", 1)[0]
    assert "- name: Self-test console exe (evolab-cli.exe)" in before and "- name: Copy exe folder to a Korean path" in before
    assert "if: ${{ !cancelled() && steps.build.outcome == 'success' }}" in step and "shell: pwsh" in step
    lines = [ln.strip() for ln in step.splitlines()]
    assert '$cli = Join-Path $dest "evolab\\evolab-cli.exe"' in lines
    assert "& $cli preset running -o $made" in lines
    assert any(ln.startswith("& $cli bench $envfile ") for ln in lines)
    assert "if ($a -ne $b) {" in step and "Get-FileHash -Algorithm SHA256" in step
    for k, ln in enumerate(lines):
        if ln.startswith("& $cli "):
            assert lines[k + 1] == "if ($LASTEXITCODE -ne 0) { exit 1 }", ln
        if ln.startswith(("& $cli", "$out = & $cli")):
            assert not re.search("[가-힣]", re.sub(r'"[^"]*"', "", ln)), ln
    passed = f"[{S.SELFTEST_PASS}] {S.SELFTEST_CHECK_PRESETS.split(' (')[0]}"
    assert f'-SimpleMatch -Quiet "{passed}"' in step  # 프로그램이 쓰는 글 그대로
    tests_job = text.split("\n  tests:\n", 1)[1].split("\n  package:\n", 1)[0]
    assert "timeout-minutes: 60\n        run: uv run pytest -q\n" in tests_job  # W1: 빠른 테스트 전체(6.2.0은 파일 148개)

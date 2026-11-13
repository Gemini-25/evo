"""6.2.0 통합(최종 설계 4.1의 묶음 I): 처음 쓰는 사람의 길과 호환.

- 기계가 흉내 내는 처음 쓰는 사람(I2, 설계 T22의 꼴을 주 창 전체로): 글자를 치지 않고, 눈에 띄는 것(으뜸 단추·카드·체크)만 누른다.
  ① 실험실 「새 프리셋…」 → (첫 카드 달리기가 골라진 채) 으뜸 단추 「실험실로 보내기」: 누름 2번으로 실험실에 걷는 사람이 오고, 프리셋 권장
  설정(예선 3 × 9,600)이 폼에 들어가며 「학습 시작」이 켜지고, 저장된 파일이 「내 프리셋」 묶음에 나온다. 그 파일은 L2·L3을 넘는다.
  ② 공과 골대: 빈 세계(위에서 본)에서 원형 선수 · 공 · 골대 · 「골에 들어가면」 규칙을 골라 실험실로: 설계의 상한 15번 안. 누를 때마다 L1.
  느린 쪽(S)은 두 길 끝에서 「학습 시작」을 눌러 진짜 학습 프로세스가 2세대를 낸다.
- 호환(I4): 형식 4 파일(내장 달리기)을 얼린 v6.1.0이 SPEC_NEWER_VERSION 글로 거절하고, 안내 창이 만든 형식 1 파일(빈 세계 옆)은 연다.
  v1.0~v6.1의 시험 자료 실행은 열리고(목록·요약) 지금 프로그램이 그 환경을 다시 컴파일한다. 한글 경로의 자체 점검 「프리셋」.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QWidget

from evolab import paths
from evolab.app import strings as S
from evolab.app import style
from evolab.presets import rules as R
from evolab.presets import trainable as TR
from evolab.spec.io import parse_env

ROOT = Path(__file__).resolve().parents[1]
STEP_LIMIT = 15
"""설계 T22: 빈 세계(위)에서 단계마다 하나 이상 바꾸는 길 15번 이하(「새 프리셋…」 포함)."""


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from evolab.app.editor.view import EditorView
    from evolab.app.preset.wizard import PresetWizard

    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv("EVOLAB_NO_TOUR", "1")
    monkeypatch.setattr(PresetWizard, "ask_discard", lambda self: True)
    monkeypatch.setattr(EditorView, "maybe_save", lambda self: True)
    yield tmp_path


class NaiveUser:
    """글자를 치지 않고 보이고 켜진 것만 마우스로 누르는 사람. 누름 수를 세고, 안내 창이 열려 있는 동안 누를 때마다 L1을 본다."""

    def __init__(self, qtbot: Any, window: Any) -> None:
        self.qtbot = qtbot
        self.window = window
        self.presses = 0
        self.l1_always = True

    @property
    def wizard(self) -> Any:
        w = getattr(self.window.editor, "wizard", None)
        try:
            return w if w is not None and w.isVisible() else None
        except RuntimeError:  # 닫혀 지워진 창
            return None

    def press(self, widget: QWidget) -> None:
        assert widget.isVisible() and widget.isEnabled(), (widget.objectName(), getattr(widget, "text", lambda: "")())
        wizard = self.wizard
        if wizard is not None and wizard.scroll.viewport().isAncestorOf(widget):
            wizard.scroll.ensureWidgetVisible(widget)
        pos = widget.rect().center()
        if isinstance(widget, QCheckBox):
            pos.setX(min(8, widget.width() // 2))
        self.qtbot.mouseClick(widget, Qt.MouseButton.LeftButton, pos=pos)
        QApplication.processEvents()
        self.presses += 1
        wizard = self.wizard
        if wizard is not None:
            self.l1_always = self.l1_always and TR.quick_check(wizard.data).ok

    def press_primary(self, root: QWidget) -> None:
        """눈에 띄는 단추(켜진 으뜸이 하나뿐이어야 한다)를 누른다."""
        on = [b for b in style.primaries(root) if b.isVisible() and b.isEnabled()]
        assert len(on) == 1, [b.text() for b in on]
        self.press(on[0])


def main_window(qtbot: Any) -> Any:
    from evolab.app.main import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1366, 768)
    window.show()
    QTest.qWaitForWindowExposed(window)
    window.tabs.setCurrentWidget(window.lab)
    QTest.qWait(30)
    return window


def _mine(lab: Any) -> list[str]:
    """실험실 환경 목록의 「내 프리셋」 묶음 항목(머리 줄 다음부터 다음 머리 전까지)."""
    combo = lab.env_combo
    texts = [combo.itemText(i) for i in range(combo.count())]
    head = S.LAB_ENV_GROUPS["mine"]
    if head not in texts:
        return []
    out = []
    for i in range(texts.index(head) + 1, len(texts)):
        if combo.model().item(i) is not None and not combo.model().item(i).isSelectable():
            break
        out.append(texts[i])
    return out


def walker_path(qtbot: Any, window: Any) -> tuple[NaiveUser, Any]:
    user = NaiveUser(qtbot, window)
    lab = window.lab
    user.press(lab.btn_new_preset)
    wizard = user.wizard
    assert wizard is not None and wizard.isModal()
    assert wizard.pages[0].cards["running"].isChecked()  # 첫 카드가 골라진 채 열림
    user.press_primary(wizard)  # 「실험실로 보내기」
    assert user.wizard is None
    return user, lab


def test_naive_user_gets_a_walker_into_the_lab_in_two_presses(qtbot: Any, isolated: Path) -> None:
    window = main_window(qtbot)
    user, lab = walker_path(qtbot, window)
    assert user.presses == 2 and user.l1_always
    assert window.tabs.currentWidget() is lab
    saved = sorted(paths.user_env_dir().glob("*.evoenv"))
    names = [json.loads(p.read_text(encoding="utf-8"))["name"] for p in saved]
    preset = S.PRESET_NAME_DEFAULT.format(sport=S.ED_BUILTIN_RUNNING)
    assert preset in names
    assert lab.spec is not None and lab.spec.file.name == preset
    assert lab.qualify_entrants.value() == 3  # 프리셋 권장 설정이 폼에
    assert S.LAB_PRESET_TRAIN_APPLIED.split("{", 1)[0] in lab.status.text()
    assert lab.btn_start.isEnabled() and lab.btn_start in style.primaries(lab)
    lab.refresh_user_envs()
    assert preset in _mine(lab)
    spec = parse_env(saved[names.index(preset)].read_text(encoding="utf-8"))
    quick = TR.quick_check(json.loads(saved[names.index(preset)].read_text(encoding="utf-8")))
    assert quick.ok and TR.alive(quick.spec).ok and TR.probe(quick.spec).ok  # L1·L2·L3: 학습할 수 있는 걷는 사람
    assert spec.editor.get("train") == {"qualify_entrants": 3, "qualify_trials": 9600}


def ball_and_goal_path(qtbot: Any, window: Any) -> tuple[NaiveUser, Any]:
    user = NaiveUser(qtbot, window)
    user.press(window.lab.btn_new_preset)
    wizard = user.wizard
    assert wizard is not None
    user.press(wizard.pages[0].cards["blank_top"])        # ① 빈 세계(위에서 본)
    user.press(wizard.btn_next)
    user.press(wizard.pages[1].cards["disc"])             # ② 원형 선수
    user.press(wizard.btn_next)
    for prop in ("ball", "goal"):                         # ③ 공과 골대
        user.press(wizard.pages[2].checks[prop])
    user.press(wizard.btn_next)
    user.press(wizard.pages[3].rows["reach"].check)       # ④ 골에 들어가면 성공
    user.press(wizard.btn_next)                           # ⑤ 해 보기
    data = json.loads(json.dumps(wizard.data))
    user.press_primary(wizard)                            # 「실험실로 보내기」
    assert user.wizard is None
    return user, data


def test_naive_user_builds_ball_and_goal_within_the_step_limit(qtbot: Any, isolated: Path) -> None:
    window = main_window(qtbot)
    user, data = ball_and_goal_path(qtbot, window)
    assert user.presses <= STEP_LIMIT and user.l1_always, user.presses
    kinds = [r["rule"] for r in data["editor"]["recipe"]["rules"]]
    assert "reach" in kinds and "time_limit" in kinds
    ids = {b["id"] for b in data["bodies"]} | {r["id"] for r in data.get("regions", [])}
    assert any("ball" in i for i in ids) and any("goal" in i for i in ids), ids
    reach = next(u for u in R.find_rules(data)[0] if u.kind == "reach")
    goals = {r["id"] for r in data.get("regions", []) if "goal" in r["id"]}
    assert reach.args["where"] in goals, (reach.args, goals)  # 「들어가면 성공」이 그 골대를 가리킨다
    quick = TR.quick_check(data)
    assert quick.ok and TR.alive(quick.spec).ok and TR.probe(quick.spec).ok
    lab = window.lab
    assert lab.spec is not None and lab.btn_start.isEnabled()


def _train_two_generations(qtbot: Any, lab: Any) -> None:
    lab.btn_start.click()
    qtbot.waitUntil(lambda: lab.generation >= 2, timeout=300_000)


@pytest.mark.slow
@pytest.mark.parametrize("path", ["walker", "ball_and_goal"])
def test_s_naive_paths_then_real_training(qtbot: Any, isolated: Path, path: str) -> None:
    """두 길 끝에서 「학습 시작」을 눌러 진짜 학습 프로세스(워커 포함)가 2세대를 낸다."""
    window = main_window(qtbot)
    if path == "walker":
        walker_path(qtbot, window)
    else:
        ball_and_goal_path(qtbot, window)
    lab = window.lab
    lab.workers.setValue(1)
    _train_two_generations(qtbot, lab)
    assert lab.generation >= 2 and lab.client.active


# ---------------------------------------------------------------------------
# I4 호환
# ---------------------------------------------------------------------------
REFUSE = r"""
import sys
from pathlib import Path

import numpy as np

from evolab.spec.compiled import EnvFileError
from evolab.spec.io import parse_env

out, files = sys.argv[1], sys.argv[2:]
result = {}
for k, name in enumerate(files):
    try:
        parse_env(Path(name).read_text(encoding="utf-8"))
        result[f"ok{k}"] = np.array(True)
        result[f"msg{k}"] = np.array([""])
    except EnvFileError as exc:
        result[f"ok{k}"] = np.array(False)
        result[f"msg{k}"] = np.array([i.message for i in exc.issues])
np.savez(out, **result)
"""


def test_frozen_v61_refuses_format4_and_opens_a_format1_preset(tmp_path: Path) -> None:
    """C12: 얼린 v6.1.0이 형식 4 파일(내장 달리기)을 한국어로 거절하고, 형식 1로 끝나는 조리법의 파일(빈 세계 옆)은 연다(T9·T26)."""
    from evolab.presets import recipe as RC
    from evolab.spec.io import feature_version, load_env
    from tests.oracle import frozen

    running = tmp_path / "달리기.evoenv"
    shutil.copyfile(paths.builtin_env_dir() / "running.evoenv", running)
    side = tmp_path / "빈 세계.evoenv"
    recipe = RC.load_builtin("blank_side")
    recipe.roles[0].body.values["balance"] = 0.0  # 균형 보조 「끔」: 안내 창의 체크와 같은 길로 형식 1(D27)
    side.write_text(json.dumps(RC.build(recipe), ensure_ascii=False, indent=2), encoding="utf-8")
    assert feature_version(load_env(side).file) == 1 and feature_version(load_env(running).file) == 4
    script = tmp_path / "refuse.py"
    script.write_text(REFUSE, encoding="utf-8")
    result = frozen.frozen_run("v6.1.0", script, running, side)
    assert not bool(result["ok0"]) and list(result["msg0"]) == [S.SPEC_NEWER_VERSION.format(version=4)]
    assert bool(result["ok1"])


def test_old_fixture_runs_still_list_and_compile(tmp_path: Path) -> None:
    """v1.0(GA)·v1.1(재시작)·v6.1(자라는 뇌) 시험 자료 실행이 목록에 읽히고, 실행에 든 환경 사본을 지금 프로그램이 컴파일한다."""
    from evolab.runs.catalog import list_runs
    from evolab.spec.io import compile_env

    runs = tmp_path / "기록"
    sources = sorted(p for p in (ROOT / "tests" / "data").glob("*/run") if (p / "run.json").is_file())
    assert {p.parent.name for p in sources} >= {"v10_run_ga", "v11_run_restart", "v61_grow_run"}
    for src in sources:
        shutil.copytree(src, runs / src.parent.name)
    summaries = {s.name: s for s in list_runs(runs)}
    assert set(summaries) >= {p.parent.name for p in sources}
    for src in sources:
        envs = sorted((runs / src.parent.name / "envs").glob("*.evoenv"))
        assert envs, src
        for env in envs:
            compile_env(env.read_text(encoding="utf-8"))


def test_selftest_presets_check_in_a_korean_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab import selftest

    korean = tmp_path / "진화 실험실 자료"
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(korean / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(korean / "state"))
    check = dict(selftest.selftest_checks(0, check_gui=False))[S.SELFTEST_CHECK_PRESETS]
    detail = check()
    assert isinstance(detail, str) and detail

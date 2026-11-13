"""6.2.0 실험실의 「내 프리셋」과 프리셋 권장 설정 (스포츠 설계 3.1 H, 6.1 T35, 묶음 Hb).

- 「내 프리셋」 = 내 환경 폴더의 *.evoenv에서, 같은 줄기 이름의 내장 파일과 env_hash가 같은 것(첫 실행이 복사한 손대지 않은
  사본)을 뺀 것. 고친 사본·새 파일은 있다. 저장하면(폴더가 바뀌면) 묶음이 다시 채워진다.
- 「새 프리셋…」 단추(objectName = 속성 이름)는 new_preset_requested를 낸다.
- 프리셋(editor.train)을 고르면 그 권장값이 「학습 안정성」 칸에 들어가고 상태 글에 한 줄. 권장값 없는 환경으로 바꾸면 칸은
  새 실행 기본값으로 돌아가고 줄은 없어진다. 복제한 실행은 그 실행의 설정(권장값을 얹지 않음).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QSettings, Qt
from pytestqt.qtbot import QtBot

from evolab import paths
from evolab.app import strings as S
from evolab.app.lab import view as lab_view
from evolab.app.lab.view import LabView, preset_train_text, user_presets
from evolab.engine.trainer import NEW_RUN_DEFAULTS, TrainConfig

ENV = paths.builtin_env_dir()
LANDER = ENV / "lander.evoenv"
RUNNING = ENV / "running.evoenv"


@pytest.fixture
def mine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "environments"
    folder.mkdir()
    monkeypatch.setattr(paths, "user_env_dir", lambda: folder)
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path / "runs")
    return folder


@pytest.fixture
def lab(qtbot: QtBot, tmp_path: Path, mine: Path) -> Any:
    view = LabView(settings=QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(view)
    try:
        yield view
    finally:
        view.shutdown()


def write(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def lander_data(**editor: Any) -> dict[str, Any]:
    data = json.loads(LANDER.read_text(encoding="utf-8"))
    data["editor"] = {**(data.get("editor") or {}), **editor}
    return data


def group(lab: LabView) -> list[str]:
    """「내 프리셋」 묶음의 파일 경로(머리 줄 뒤). 묶음이 없으면 []."""
    combo = lab.env_combo
    start = 1 + len(paths.builtin_envs())
    if combo.count() == start:
        return []
    assert combo.itemData(start) is None and combo.itemText(start) == S.LAB_ENV_GROUPS["mine"]
    return [combo.itemData(i) for i in range(start + 1, combo.count())]


# ---------------------------------------------------------------------------
# 「내 프리셋」의 정의
def test_first_run_copies_are_not_my_presets_but_changed_copies_are(mine: Path) -> None:
    copied = paths.install_examples(mine)
    assert sorted(p.name for p in copied) == sorted(p.name for p in paths.builtin_envs())
    assert user_presets() == []
    # 에디터 칸(기본 보기)만 바꾼 사본은 env_hash가 같다: 여전히 손대지 않은 사본
    write(mine / "lander.evoenv", lander_data(view=[-10.0, -1.0, 10.0, 12.0]))
    assert user_presets() == []
    # 내용을 고친 사본은 내 것
    data = lander_data()
    data["world"]["gravity"] = [0.0, -5.0]
    write(mine / "lander.evoenv", data)
    assert user_presets() == [mine / "lander.evoenv"]
    # 내장 이름이 아닌 파일, 읽을 수 없는 파일도 내 것(열면 오류를 알린다), 다른 확장자는 아님
    shutil.copyfile(RUNNING, mine / "내 달리기.evoenv")
    (mine / "깨진.evoenv").write_text("{", encoding="utf-8")
    (mine / "메모.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in user_presets()] == sorted(["lander.evoenv", "내 달리기.evoenv", "깨진.evoenv"],
                                                      key=lambda n: (n.lower(), n))


def test_missing_folder_gives_no_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "user_env_dir", lambda: tmp_path / "없음")
    assert user_presets() == []


def test_lab_list_has_a_my_presets_group_after_the_builtins(qtbot: QtBot, mine: Path, lab: LabView) -> None:
    want = [str(p) for p in paths.builtin_envs()]
    combo = lab.env_combo
    assert [combo.itemData(i) for i in range(1, 1 + len(want))] == want
    assert group(lab) == []  # 빈 폴더: 묶음 머리도 없음
    paths.install_examples(mine)
    lab.refresh_user_envs()
    assert group(lab) == []  # 첫 실행이 복사한 사본은 묶음에 없다

    path = write(mine / "내 착륙.evoenv", {**lander_data(), "name": "내 착륙선"})
    lab.refresh_user_envs()
    assert group(lab) == [str(path)]
    start = 1 + len(want)
    head = combo.model().item(start)  # type: ignore[attr-defined]
    assert not head.isEnabled() and head.toolTip() == S.LAB_ENV_GROUPS_TIP["mine"]
    assert combo.itemText(start + 1) == "내 착륙선"
    # 묶음의 파일을 고르면 그 환경으로 새 실행을 준비한다
    combo.setCurrentIndex(start + 1)
    combo.activated.emit(start + 1)
    assert lab.env_path == path and lab.spec is not None and lab.spec.file.name == "내 착륙선"
    # 다시 채워도 고른 줄은 그대로
    write(mine / "가 먼저.evoenv", {**lander_data(), "name": "가"})
    lab.refresh_user_envs()
    assert combo.currentData() == str(path)
    assert group(lab) == sorted([str(path), str(mine / "가 먼저.evoenv")], key=lambda s: (Path(s).name.lower(), s))


def test_saving_into_the_folder_refills_the_group(qtbot: QtBot, mine: Path, lab: LabView) -> None:
    lab.show()
    qtbot.waitExposed(lab)
    assert group(lab) == []
    path = write(mine / "새 프리셋.evoenv", {**lander_data(), "name": "새 것"})
    # 폴더 지켜보기(저장한 프로그램과 무관), 아니면 실험실이 다시 보일 때
    try:
        qtbot.waitUntil(lambda: group(lab) == [str(path)], timeout=5000)
    except AssertionError:
        lab.hide()
        lab.show()
        assert group(lab) == [str(path)]
    path.unlink()
    lab.hide()
    lab.show()
    assert group(lab) == []


def test_experimental_builtin_title(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qtbot: QtBot, mine: Path) -> None:
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    (recipes / "running.json").write_text(json.dumps({"recipe": 1, "status": "experimental"}), encoding="utf-8")
    monkeypatch.setattr(paths, "recipe_dir", lambda: recipes)
    assert lab_view.builtin_status("running") == "experimental" and lab_view.builtin_status("lander") == "ok"
    view = LabView(settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(view)
    try:
        i = view.env_combo.findData(str(RUNNING))
        assert view.env_combo.itemText(i) == S.LAB_ENV_EXPERIMENTAL.format(name="달리기")
        assert view.env_combo.itemText(view.env_combo.findData(str(LANDER))) == "달 착륙선"
    finally:
        view.shutdown()


# ---------------------------------------------------------------------------
# 「새 프리셋…」 단추
def test_new_preset_button(qtbot: QtBot, lab: LabView) -> None:
    btn = lab.btn_new_preset
    assert btn.objectName() == "btn_new_preset" and lab.env_combo.objectName() == "env_combo"
    assert btn.text() == S.ED_NEW_PRESET and btn.toolTip() == S.ED_NEW_PRESET_TIP
    assert btn.parentWidget() is lab.env_combo.parentWidget()  # 「환경」 묶음 안
    with qtbot.waitSignal(lab.new_preset_requested, timeout=1000):
        qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)


# ---------------------------------------------------------------------------
# 프리셋 권장 설정
def stability(lab: LabView) -> tuple[Any, ...]:
    c = lab.config_from_form()
    return c.qualify_entrants, c.qualify_trials, c.restart_patience, c.restart_from


DEFAULTS = (NEW_RUN_DEFAULTS["qualify_entrants"], TrainConfig.for_new_run().qualify_trials,
            NEW_RUN_DEFAULTS["restart_patience"], NEW_RUN_DEFAULTS["restart_from"])


def test_choosing_the_running_preset_fills_the_form_and_says_so(lab: LabView) -> None:
    assert lab.load_env(LANDER) and stability(lab) == DEFAULTS
    assert S.LAB_PRESET_TRAIN_APPLIED.split("{")[0] not in lab.status.text()
    assert lab.load_env(RUNNING)
    assert lab.qualify_entrants.value() == 3 and lab.qualify_trials.value() == 9600
    assert stability(lab)[:2] == (3, 9600)
    what = " · ".join([preset_train_text("qualify_entrants", 3), preset_train_text("qualify_trials", 9600)])
    line = S.LAB_PRESET_TRAIN_APPLIED.format(what=what)
    assert lab.status.text().split("\n") == [S.LAB_STATE_NAMES["idle"], line]
    assert "9,600" in line
    # 새 실행이 쓰는 설정 = for_new_run(spec)의 값(run.json에 적힌다: 엔진 쪽 T7)
    assert lab.spec is not None
    fresh = TrainConfig.for_new_run(lab.spec)
    assert stability(lab) == (fresh.qualify_entrants, fresh.qualify_trials, fresh.restart_patience, fresh.restart_from)
    # 권장값 없는 환경으로 바꾸면 기본값으로 돌아가고 줄은 없다
    assert lab.load_env(LANDER)
    assert stability(lab) == DEFAULTS and lab.status.text() == S.LAB_STATE_NAMES["idle"]


def test_user_values_survive_old_envs_and_wrong_preset_keys_are_dropped(lab: LabView, mine: Path) -> None:
    assert lab.load_env(LANDER)
    lab.qualify_entrants.setValue(4)
    lab.qualify_trials.setValue(5000)
    other = ENV / "jump.evoenv"
    assert lab.load_env(other)  # 옛 환경끼리는 v6.1처럼 칸을 건드리지 않는다
    assert stability(lab)[:2] == (4, 5000)
    path = write(mine / "잘못.evoenv", lander_data(train={"qualify_entrants": "많이", "restart_patience": 30000,
                                                           "restart_from": "champion", "popsize": 999}))
    assert lab.load_env(path)
    assert stability(lab) == (NEW_RUN_DEFAULTS["qualify_entrants"], TrainConfig.for_new_run().qualify_trials, 30000,
                              "champion")
    assert lab.restart_on.isChecked() and lab.popsize.value() != 999
    what = " · ".join([preset_train_text("restart_patience", 30000), preset_train_text("restart_from", "champion")])
    assert lab.status.text().endswith(S.LAB_PRESET_TRAIN_APPLIED.format(what=what))


def test_cloned_run_keeps_its_own_settings(lab: LabView) -> None:
    text = RUNNING.read_text(encoding="utf-8")
    config = TrainConfig.for_new_run(qualify_entrants=2, qualify_trials=4000).to_dict()
    assert lab.use_env_text(text, "복제", config)
    assert stability(lab)[:2] == (2, 4000)
    assert S.LAB_PRESET_TRAIN_APPLIED.split("{")[0] not in lab.status.text()
    # 복제한 칸은 프리셋 값이 아니다: 다음에 옛 환경을 골라도 그대로(v6.1)
    assert lab.load_env(LANDER) and stability(lab)[:2] == (2, 4000)


def test_starting_clears_the_preset_line(lab: LabView, monkeypatch: pytest.MonkeyPatch) -> None:
    assert lab.load_env(RUNNING)
    assert len(lab.status.text().split("\n")) == 2
    monkeypatch.setattr(lab.client, "start", lambda req: None)
    assert lab.start_new()
    assert lab.status.text().split("\n")[0] == S.LAB_STATE_NAMES["starting"]
    assert S.LAB_PRESET_TRAIN_APPLIED.split("{")[0] not in lab.status.text()

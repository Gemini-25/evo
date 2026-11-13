"""「새 프리셋」 창 (6.2.0, 묶음 W): 최종 설계 6.1의 T20·T22·T24·T25·T26·T27과 보관함 창.

- T20 창: 첫 카드가 골라진 채 열림, 다섯 단계를 오가도 값이 남음, adjust가 바꾼 것이 상태 줄에, 바꾼 것이 있을 때만 닫기 물음,
  창에 묶인 모달, 1024×600·1366×728 작업 영역에서 잘리는 자식 위젯 0, 최소 크기, 처음 크기가 화면 안.
- T22 회귀 지킴(쓰기 쉬움의 증거가 아님): 「달리기」 카드 3번, 빈 세계(위)에서 단계마다 하나 이상 바꾸는 길 15번 이하, 글자 입력 0,
  누를 때마다 L1.
- T24 나가기: 「직접 해 보기」는 저장도 닫기도 하지 않음. 나머지 둘은 safe_file_stem 이름으로 저장, 같은 이름이면 번호(덮어쓰지 않음).
  can_leave가 거짓이면 창이 남고 파일을 쓰지 않음.
- T25 슬라이더 → 미리보기 문서가 바뀜(100ms 아래, CI 여유), 미리보기가 내용 상자에 맞음.
- T26 표지(editor.recipe의 built·templates)와 얼린 v6.1: 읽고 다시 써도 editor.*가 남고, 형식 1 조리법의 결과를 v6.1이 연다.
- T27 쓰임 기록: 상태 폴더에만, 설정으로 꺼짐, 이름·경로가 없음, 오류 보고 묶음에 안 들어감.

데이터·상태 폴더는 테스트마다 tmp로 돌린다(monkeypatch, 전체 실행에는 걸지 않음).
"""

from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QSlider,
    QWidget,
)

from evolab import paths
from evolab.app import strings as S
from evolab.app import style
from evolab.app.preset import wizard as WZ
from evolab.app.preset.cards import TemplatePicker
from evolab.app.preset.wizard import PresetWizard, open_wizard
from evolab.presets import recipe as RC
from evolab.presets import trainable as TR
from evolab.presets.library import Library
from evolab.spec.io import env_hash, parse_env

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "recipes"


@pytest.fixture()
def iso(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("EVOLAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EVOLAB_STATE_DIR", str(tmp_path / "state"))
    return tmp_path


def make(qtbot: Any, **kw: Any) -> PresetWizard:
    w = PresetWizard(**kw)
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    return w


def env_files() -> list[Path]:
    folder = paths.user_env_dir()
    return sorted(folder.glob("*.evoenv")) if folder.is_dir() else []


def l1(w: PresetWizard) -> bool:
    return TR.quick_check(w.data).ok


# ---------------------------------------------------------------------------
# T20 창
# ---------------------------------------------------------------------------


def test_t20_opens_with_the_first_card_and_three_exit_buttons(qtbot: Any, iso: Path) -> None:
    w = make(qtbot)
    first = RC.starters()[0].id
    assert w.start == first == "running"
    card = w.pages[0].cards[first]
    assert card.isChecked()
    assert w.btn_edit.isEnabled() and w.btn_play.isEnabled() and w.btn_lab.isEnabled()
    assert w.name_edit.text() == S.PRESET_NAME_DEFAULT.format(sport=RC.load_builtin("running").name)
    assert l1(w)
    assert w.data["editor"]["train"] == {"qualify_entrants": 3, "qualify_trials": 9600}
    # 첫 카드 넷(숨김 없는 내장 조리법), 화면 글은 strings.py에서
    assert list(w.pages[0].cards) == [s.id for s in RC.starters()]
    assert w.pages[0].cards["blank_top"].title == S.PRESET_START_BLANK_TOP


def test_t20_window_modal_and_sizes(qtbot: Any, iso: Path) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.show()
    w = open_wizard(parent)
    assert w.isVisible()
    assert w.windowModality() == Qt.WindowModality.WindowModal
    assert (w.minimumWidth(), w.minimumHeight()) == WZ.WIZARD_MIN
    assert w.isSizeGripEnabled()
    extra = w.frameGeometry().size() - w.geometry().size()
    for area, want in ((QRect(0, 0, 1366, 728), (1180, 700)), (QRect(100, 50, 1920, 1040), (1180, 700))):
        w.fit_to(area)
        assert (w.width(), w.height()) == want
        assert area.contains(w.frameGeometry())  # 창틀까지 작업 영역 안
    small = QRect(0, 0, 1024, 600)  # 최소 크기와 제목 줄을 더하면 넘친다: 크기는 최소, 자리는 영역의 왼쪽 위부터
    w.fit_to(small)
    assert w.width() == max(WZ.WIZARD_MIN[0], 1024 - extra.width())
    assert w.height() == max(WZ.WIZARD_MIN[1], 600 - extra.height())
    assert w.frameGeometry().topLeft() == small.topLeft()
    assert WZ.initial_size(QRect(0, 0, 800, 500)).toTuple() == WZ.WIZARD_MIN
    w.reject()


def test_t20_values_survive_moving_between_steps(qtbot: Any, iso: Path) -> None:
    w = make(qtbot)
    w.go(1)
    slider = w.pages[1].sliders.widget_for("size")
    assert isinstance(slider, QSlider)
    slider.setValue(slider.maximum())
    size = w.recipe.roles[0].body.values["size"]
    assert size == pytest.approx(1.15)
    w.go(3)
    w.pages[3].rows["survive"].check.setChecked(True)
    kinds = [r["rule"] for r in w.recipe.rules]
    assert "survive" in kinds
    for step in (0, 4, 2, 3, 1, 0, 3):
        w.go(step)
    w.go(1)
    assert w.pages[1].sliders.values()["size"] == pytest.approx(1.15)
    assert w.recipe.roles[0].body.values["size"] == pytest.approx(1.15)
    w.go(3)
    assert w.pages[3].rows["survive"].check.isChecked()
    assert [r["rule"] for r in w.recipe.rules] == kinds
    assert w.start == "running"
    # 단계 줄: 지난 단계에 ✔, 바꾼 단계는 풍선에 표시
    w.go(4)
    assert w.rail.item(1).text() == S.PRESET_STEP_DONE.format(step=S.PRESET_STEPS[1])
    assert S.PRESET_STEP_CHANGED_TIP in w.rail.item(1).toolTip()
    assert S.PRESET_STEP_CHANGED_TIP in w.rail.item(3).toolTip()
    assert S.PRESET_STEP_CHANGED_TIP not in w.rail.item(2).toolTip()


def test_t20_adjust_changes_show_in_the_status_line(qtbot: Any, iso: Path) -> None:
    w = make(qtbot)
    w.choose_start("blank_top")
    w.go(3)
    reach = w.pages[3].rows["reach"].check
    assert reach.isEnabled()  # 영역이 없지만 켜면 골대가 함께 들어간다
    reach.setChecked(True)
    assert any(p.template == "goal" for p in w.recipe.props)
    note = RC.adjust(RC.default_recipe("blank_top"))[1]
    assert note == []
    assert w.notes and w.notes[0] in w.status.text()
    assert l1(w)
    # 그 규칙이 쓰는 물건은 끌 수 없다(까닭은 풍선의 꺼진 까닭으로)
    w.go(2)
    goal_box = w.pages[2].checks["goal"]
    assert goal_box.isChecked() and not goal_box.isEnabled()
    assert S.RULE_TITLES["reach"] in style.why_off(goal_box)
    # 바뀐 것이 없으면 상태 줄은 L1·L2 결과만
    w.go(3)
    w.pages[3].rows["reach"].check.setChecked(False)
    assert not w.notes
    assert w.status.text().startswith("✔")


def test_t20_close_asks_only_when_something_changed(qtbot: Any, iso: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[bool] = []
    answer = {"value": False}

    def ask(self: PresetWizard) -> bool:
        asked.append(True)
        return answer["value"]

    monkeypatch.setattr(PresetWizard, "ask_discard", ask)
    w = make(qtbot)
    w.go(2)
    w.go(0)
    w.reject()
    assert not asked and not w.isVisible()

    w2 = make(qtbot)
    w2.go(3)
    w2.pages[3].rows["survive"].check.setChecked(True)
    w2.reject()
    assert asked == [True] and w2.isVisible()
    qtbot.keyClick(w2, Qt.Key.Key_Escape)
    assert asked == [True, True] and w2.isVisible()
    answer["value"] = True
    w2.reject()
    assert not w2.isVisible()
    assert env_files() == []


def _clipped(w: QWidget, scroll: Any) -> str | None:
    """가로로 잘렸으면 까닭(설계 다듬기 C2의 정의: sizeHint 폭 > 제 폭, 스크롤 안은 보이게 한 뒤 보이는 폭 < 제 폭)."""
    if isinstance(w, QLabel) and (w.wordWrap() or not w.text()):
        return None
    if isinstance(w, QLabel | QAbstractButton) and w.sizeHint().width() > w.width() + 1:
        return f"hint {w.sizeHint().width()} > {w.width()}"
    if scroll.viewport().isAncestorOf(w):
        scroll.ensureWidgetVisible(w, 0, 0)
    vis = w.visibleRegion().boundingRect()
    if vis.width() + 1 < w.width():
        return f"visible {vis.width()} < {w.width()}"
    return None


@pytest.mark.parametrize("area", [(1024, 600), (1366, 728)])
def test_t20_no_clipped_child_widgets(qtbot: Any, iso: Path, area: tuple[int, int]) -> None:
    w = make(qtbot)
    w.fit_to(QRect(0, 0, *area))
    qtbot.wait(10)
    bad: list[str] = []
    for start in ("running", "drill", "blank_side", "blank_top"):
        w.choose_start(start)
        for step in range(WZ.STEP_COUNT):
            w.go(step)
            qtbot.wait(1)
            for child in w.findChildren(QWidget):
                if not child.isVisible() or not isinstance(
                        child, QLabel | QAbstractButton | QComboBox | QLineEdit | QAbstractSpinBox | QSlider):
                    continue
                if child.window() is not w:
                    continue
                why = _clipped(child, w.scroll)
                if why:
                    bad.append(f"{start}/{step}: {type(child).__name__} {child.objectName()!r} "
                               f"{getattr(child, 'text', lambda: '')()!r}: {why}")
    assert not bad, "\n".join(bad[:20])
    assert w.scroll.horizontalScrollBar().maximum() == 0


# ---------------------------------------------------------------------------
# T22 정해진 길의 누름 수(회귀 지킴)
# ---------------------------------------------------------------------------


class Clicks:
    def __init__(self, qtbot: Any, w: PresetWizard) -> None:
        self.qtbot = qtbot
        self.w = w
        self.n = 1  # 창 밖의 「새 프리셋…」 한 번
        self.ok = True

    def __call__(self, widget: QWidget) -> None:
        assert widget.isVisible() and widget.isEnabled(), widget.objectName()
        if self.w.scroll.viewport().isAncestorOf(widget):
            self.w.scroll.ensureWidgetVisible(widget)
        self.qtbot.mouseClick(widget, Qt.MouseButton.LeftButton, pos=widget.rect().center())
        self.n += 1
        if self.w.isVisible():
            self.ok = self.ok and l1(self.w)


def test_t22_running_card_is_three_clicks(qtbot: Any, iso: Path) -> None:
    w = make(qtbot)
    got: list[tuple[str, str, Any]] = []
    w.finished_with.connect(lambda text, what, path: got.append((text, what, path)))
    click = Clicks(qtbot, w)
    click(w.pages[0].cards["running"])
    click(w.btn_lab)
    assert click.n == 3 and click.ok
    assert len(got) == 1 and got[0][1] == "lab"
    assert parse_env(got[0][0]).name == S.PRESET_NAME_DEFAULT.format(sport=RC.load_builtin("running").name)


def test_t22_blank_top_path_with_a_change_per_step(qtbot: Any, iso: Path) -> None:
    w = make(qtbot)
    got: list[tuple[str, str, Any]] = []
    w.finished_with.connect(lambda text, what, path: got.append((text, what, path)))
    click = Clicks(qtbot, w)
    click(w.pages[0].cards["blank_top"])                  # ①
    click(w.btn_next)
    click(w.pages[1].cards["disc"])                       # ② (이미 원형 선수)
    click(w.btn_next)
    for prop in ("ball", "goal", "wall"):                 # ③
        click(w.pages[2].checks[prop])
    click(w.btn_next)
    click(w.pages[3].rows["reach"].check)                 # ④
    click(w.pages[3].rows["near"].check)
    click(w.btn_next)                                     # ⑤
    click(w.btn_lab)
    assert click.n <= 15 and click.ok, click.n
    assert len(got) == 1 and got[0][1] == "lab"
    data = json.loads(got[0][0])
    kinds = [r["rule"] for r in data["editor"]["recipe"]["rules"]]
    assert {"reach", "near", "time_limit"} <= set(kinds)
    quick = TR.quick_check(data)
    assert quick.ok and TR.alive(quick.spec).ok
    assert TR.probe(quick.spec).ok
    assert Path(got[0][2]).read_text(encoding="utf-8") == got[0][0]


# ---------------------------------------------------------------------------
# T24 나가기
# ---------------------------------------------------------------------------


def test_t24_play_neither_saves_nor_closes(qtbot: Any, iso: Path) -> None:
    w = make(qtbot)
    got: list[Any] = []
    w.finished_with.connect(lambda *a: got.append(a))
    qtbot.mouseClick(w.btn_play, Qt.MouseButton.LeftButton)
    assert w.isVisible() and not got
    assert len(w.play_windows) == 1
    play = w.play_windows[0]
    assert play.isVisible() and play.parent() is w and play.isWindow()
    assert env_files() == []
    play.close()


def test_t24_edit_and_lab_save_with_safe_unique_names(qtbot: Any, iso: Path) -> None:
    name = 'a/b:c*? '
    w = make(qtbot)
    got: list[tuple[str, str, Any]] = []
    w.finished_with.connect(lambda text, what, path: got.append((text, what, path)))
    w.name_edit.setText(name)
    w.btn_edit.click()
    assert not w.isVisible()
    path1 = got[0][2]
    assert got[0][1] == "edit" and path1.name == paths.safe_file_stem(name) + ".evoenv"
    assert path1.parent == paths.user_env_dir()
    first = path1.read_bytes()

    w2 = make(qtbot)
    got2: list[tuple[str, str, Any]] = []
    w2.finished_with.connect(lambda text, what, path: got2.append((text, what, path)))
    w2.name_edit.setText(name)
    w2.btn_lab.click()
    path2 = got2[0][2]
    numbered = S.PRESET_NAME_NUMBERED.format(name=name.strip(), n=2)
    assert path2.name == paths.safe_file_stem(numbered) + ".evoenv"
    assert path1.read_bytes() == first  # 덮어쓰지 않음
    assert json.loads(got2[0][0])["name"] == numbered
    assert path2.read_text(encoding="utf-8") == got2[0][0]


def test_t24_can_leave_false_keeps_the_window_and_writes_nothing(qtbot: Any, iso: Path) -> None:
    asked: list[str] = []

    def can_leave(what: str) -> bool:
        asked.append(what)
        return False

    w = make(qtbot, can_leave=can_leave)
    got: list[Any] = []
    w.finished_with.connect(lambda *a: got.append(a))
    w.btn_edit.click()
    w.btn_lab.click()
    qtbot.keyClick(w, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert asked == ["edit", "lab", "lab"]
    assert w.isVisible() and not got and env_files() == []


def test_t24_save_failure_keeps_the_window(qtbot: Any, iso: Path) -> None:
    folder = paths.user_env_dir()
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.write_text("not a folder", encoding="utf-8")
    w = make(qtbot)
    got: list[Any] = []
    w.finished_with.connect(lambda *a: got.append(a))
    w.btn_lab.click()
    assert w.isVisible() and not got
    assert w.status.text().startswith(S.PRESET_SAVE_FAILED.split("{")[0])


def test_t24_lab_button_follows_l2(qtbot: Any, iso: Path) -> None:
    """「실험실로 보내기」는 L2일 때만 켜지고, 꺼지면 까닭이 풍선에(D38)."""
    w = make(qtbot)
    w.choose_start("blank_side")
    w.go(3)
    for kind in ("distance", "fall"):
        w.pages[3].rows[kind].check.setChecked(False)
    assert l1(w)
    assert not w.alive.ok and w.alive.reason == S.PRESET_ALIVE_NO_SCORE
    assert not w.btn_lab.isEnabled() and style.why_off(w.btn_lab) == S.PRESET_ALIVE_NO_SCORE
    assert S.PRESET_ALIVE_NO_SCORE in w.status.text()
    assert w.btn_fix.isVisible() and w.btn_fix.text() == S.PRESET_ALIVE_FIXES["add_rule"]
    assert w.btn_edit.isEnabled()  # 에디터로는 갈 수 있다(L1)
    w.go(0)
    w.btn_fix.click()
    assert w.step == 3
    w.pages[3].rows["survive"].check.setChecked(True)
    assert w.alive.ok and w.btn_lab.isEnabled() and not w.btn_fix.isVisible()


# ---------------------------------------------------------------------------
# T25 슬라이더 → 미리보기
# ---------------------------------------------------------------------------


def test_t25_slider_updates_the_preview_quickly(qtbot: Any, iso: Path) -> None:
    w = make(qtbot)
    w.fit_to(QRect(0, 0, 1366, 728))
    w.go(1)
    slider = w.pages[1].sliders.widget_for("size")
    assert isinstance(slider, QSlider)
    before_text, before_spec = w.text, w.preview.spec
    times: list[float] = []
    for value in (slider.maximum(), slider.minimum(), slider.maximum() // 2):
        t0 = time.perf_counter()
        slider.setValue(value)
        times.append((time.perf_counter() - t0) * 1000.0)
    assert w.text != before_text and w.preview.spec is not before_spec
    assert w.preview.canvas.world is not None
    assert sorted(times)[1] < 100.0, times  # 가운데 값(CI 여유. 목표 p95 50ms는 측정 M6)
    # 미리보기는 남는 자리를 쓰고 캔버스가 그 안에 맞는다
    qtbot.wait(10)
    assert w.preview.width() > 300
    assert w.preview.rect().contains(w.preview.canvas.geometry())


def test_t25_still_run_moves_the_body_with_zero_actions(qtbot: Any, iso: Path) -> None:
    w = make(qtbot)
    w.preview.btn_run.click()
    assert w.preview.running
    traj = w.preview.trajectory
    assert traj is not None and traj.frames > 10
    assert abs(float(traj.poses[-1, 0, 1]) - float(traj.poses[0, 0, 1])) > 1e-3  # 몸이 움직인다(가만히 두면 쓰러짐)
    w.preview.btn_run.click()
    assert not w.preview.running
    # 문서가 바뀌면 멈추고 시작 자세로
    w.preview.btn_run.click()
    w.go(1)
    w.pages[1].sliders.widget_for("size").setValue(0)
    assert not w.preview.running and w.preview.trajectory is None
    assert env_files() == []


# ---------------------------------------------------------------------------
# T26 표지와 얼린 v6.1
# ---------------------------------------------------------------------------


def test_t26_marker_has_built_hash_and_template_hashes(qtbot: Any, iso: Path) -> None:
    w = make(qtbot)
    marker = w.data["editor"]["recipe"]
    assert marker["built"] == env_hash(parse_env(json.loads(w.text)))
    assert set(marker["templates"]) == {"track_side", "biped"}
    assert all(len(h) == 64 for h in marker["templates"].values())
    assert RC.matches(json.loads(w.text))
    assert RC.Recipe.from_json(marker).to_json()["rules"] == w.recipe.to_json()["rules"]


FROZEN_OPEN = """
import json, sys
import numpy as np
from evolab.spec.io import compile_env, dump_env, parse_env
from evolab.engine.seeds import display_trial
from evolab.world.physics2d.episode import Episode

rows = []
for path in sys.argv[2:]:
    text = open(path, encoding="utf-8").read()
    env = parse_env(text)
    spec = compile_env(text)
    Episode(spec, None, display_trial())
    again = json.loads(dump_env(env))
    rows.append({"version": again["version"], "editor": sorted(again.get("editor", {})),
                 "recipe_same": again["editor"].get("recipe") == json.loads(text)["editor"].get("recipe")})
np.savez(sys.argv[1], rows=np.array(json.dumps(rows)))
"""


def test_t26_frozen_v61_opens_a_format1_wizard_file_and_keeps_editor(qtbot: Any, iso: Path, tmp_path: Path) -> None:
    from tests.oracle.frozen import frozen_run

    lib = Library([paths.template_dirs()[0], FIXTURES])
    recipe = RC.Recipe.from_json(json.loads((FIXTURES / "fixture_side.json").read_text(encoding="utf-8")))
    w = make(qtbot, recipe=recipe, library=lib)
    assert w.name_edit.text() == recipe.name
    got: list[tuple[str, str, Any]] = []
    w.finished_with.connect(lambda text, what, path: got.append((text, what, path)))
    w.btn_edit.click()
    data = json.loads(got[0][0])
    assert data["version"] == 1
    assert {"recipe", "rules", "train"} <= set(data["editor"])
    script = tmp_path / "frozen_open.py"
    script.write_text(textwrap.dedent(FROZEN_OPEN).lstrip(), encoding="utf-8")
    rows = json.loads(str(frozen_run("v6.1.0", script, got[0][2])["rows"]))
    assert rows == [{"version": 1, "editor": sorted(data["editor"]), "recipe_same": True}]


# ---------------------------------------------------------------------------
# T27 쓰임 기록
# ---------------------------------------------------------------------------


def _log_rows() -> list[dict[str, Any]]:
    path = WZ.usage_log_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_t27_usage_log_rows_have_no_names_or_paths(qtbot: Any, iso: Path) -> None:
    assert WZ.usage_log_enabled()  # 기본 켬
    w = make(qtbot)
    w.name_edit.setText("비밀 이름")
    w.go(1)
    w.next()
    w.pages[2].checks["ground_flat"].setChecked(True)
    w.btn_lab.click()
    rows = _log_rows()
    assert WZ.usage_log_path().parent == paths.user_state_dir()
    assert [r["action"] for r in rows][:1] == ["open"] and rows[-1]["action"] == "lab"
    assert {"step", "next", "prop", "lab"} <= {r["action"] for r in rows}
    for r in rows:
        assert set(r) == {"at", "session", "step", "action", "ms"}
        assert r["action"] in WZ.STEP_ACTIONS and isinstance(r["ms"], int) and r["ms"] >= 0
    text = WZ.usage_log_path().read_text(encoding="utf-8")
    assert "비밀" not in text and "evoenv" not in text and str(paths.data_dir()) not in text
    assert len({r["session"] for r in rows}) == 1
    # 데이터 폴더(환경·기록)에는 쓰지 않는다
    assert not list(paths.data_dir().rglob("wizard_log*"))


def test_t27_usage_log_can_be_turned_off(qtbot: Any, iso: Path) -> None:
    WZ.set_usage_log_enabled(False)
    assert not WZ.usage_log_enabled()
    w = make(qtbot)
    w.go(2)
    w.reject()
    assert not WZ.usage_log_path().exists()
    WZ.set_usage_log_enabled(True)
    assert WZ.usage_log_enabled()


def test_t27_usage_log_is_not_in_the_error_report_bundle(qtbot: Any, iso: Path) -> None:
    from evolab.app.report.bundle import BundleOptions, collect

    w = make(qtbot)
    w.go(1)
    w.reject()
    assert WZ.usage_log_path().exists()
    items = collect(BundleOptions())
    names = [i.arcname for i in items]
    assert names and not any("wizard_log" in n for n in names), names


# ---------------------------------------------------------------------------
# 보관함 창(에디터 팔레트 「틀」 단추가 함께 씀: DU 7.3)
# ---------------------------------------------------------------------------


def test_template_picker_choice_and_needs(qtbot: Any, iso: Path) -> None:
    empty: dict[str, Any] = {"version": 1, "name": "x", "bodies": []}
    picker = TemplatePicker("body", "side", empty)
    qtbot.addWidget(picker)
    assert [c.template.id for c in picker.cards] == ["biped"]
    assert picker.cards[0].isChecked()
    assert picker.btn_insert.isDefault()
    # 땅이 없으면 까닭과 「함께 넣기」
    assert not picker.needs_note.isHidden() and not picker.needs_box.isHidden()
    assert [t.id for t in picker.needs_add()] == ["ground_flat"]
    picker.sliders.widget_for("size").setValue(0)
    picker.mirror.setChecked(True)
    picker.btn_insert.click()
    tpl, values, mirror, evolve = picker.choice()
    assert tpl.id == "biped" and values["size"] == pytest.approx(0.85) and mirror and evolve == ()
    assert values["balance"] == 1.0

    picker2 = TemplatePicker("prop", "top", empty)
    qtbot.addWidget(picker2)
    assert {c.template.id for c in picker2.cards} == {"ball", "goal", "wall"}
    assert picker2.needs_add() == []
    picker2.reject()
    assert picker2.choice() is None


def test_balance_switch_drops_to_format_1(qtbot: Any, iso: Path) -> None:
    """몸 틀의 「균형 보조」 체크를 끄면 부품이 빠지고 파일이 형식 1로 내려간다(D27, 3.1 끝의 표)."""
    w = make(qtbot)
    w.choose_start("blank_side")
    assert w.data["version"] == 4
    w.go(1)
    box = w.pages[1].sliders.widget_for("balance")
    box.setChecked(False)
    assert w.data["version"] == 1
    assert not any(a.get("kind") == "balance" for a in w.data.get("actuators", []))
    assert l1(w)

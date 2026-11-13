"""「새 프리셋」 창 무작위 누르기 (6.2.0, 묶음 W, 최종 설계 T23·D38·C5).

진짜 창에 고정 시드로 300번(카드·체크·슬라이더·규칙 빈칸·단계 줄·이전/다음)을 누른다. 누를 때마다:
- 창이 가진 환경이 L1(검사 오류 0·컴파일)을 넘는다.
- 맨 예외가 없다(pytest-qt가 Qt 신호 안의 예외도 실패로 잡는다).
- 「실험실로 보내기」는 L2(alive)일 때만 켜진다(따로 다시 계산한 값과 견줌).
- 창은 닫히지 않고 파일을 쓰지 않는다(나가기 단추는 누르지 않는다).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QSlider, QToolButton, QWidget

from evolab import paths
from evolab.app.preset.pages import PartsButton
from evolab.app.preset.wizard import PresetWizard
from evolab.presets import trainable as TR

SEED = 20261113
ACTIONS = 300


@pytest.fixture()
def iso(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("EVOLAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EVOLAB_STATE_DIR", str(tmp_path / "state"))
    return tmp_path


def _targets(w: PresetWizard) -> list[QWidget]:
    """지금 쪽에서 누를 수 있는 것 + 단계 줄 + 이전/다음."""
    page = w.pages[w.step]
    out = [c for c in page.controls() if c.isVisible() and c.isEnabled()]
    out += [b for b in (w.btn_back, w.btn_next, w.btn_reset_step) if b.isEnabled()]
    out.append(w.rail)
    return out


def _poke(rng: random.Random, w: PresetWizard, target: QWidget) -> str:
    if target is w.rail:
        row = rng.randrange(w.rail.count())
        w.rail.setCurrentRow(row)
        return f"rail {row}"
    if isinstance(target, QSlider):
        target.setValue(rng.randint(target.minimum(), target.maximum()))
        return f"slider {target.value()}"
    if isinstance(target, QDoubleSpinBox):
        lo, hi = target.minimum(), min(target.maximum(), 200.0)
        target.setValue(round(rng.uniform(lo, hi), 2))
        return f"spin {target.objectName()} {target.value()}"
    if isinstance(target, QComboBox):
        if target.count():
            i = rng.randrange(target.count())
            target.setCurrentIndex(i)
            target.activated.emit(i)
        return f"combo {target.objectName()} {target.currentData()}"
    if isinstance(target, PartsButton):
        acts = target.menu_.actions()
        if acts:
            rng.choice(acts).trigger()
        return f"parts {target.chosen()}"
    if isinstance(target, QCheckBox | QToolButton):
        target.click()
        return f"click {target.objectName() or type(target).__name__}"
    target.click()  # type: ignore[attr-defined]
    return f"button {target.objectName()}"


@pytest.mark.parametrize("seed", [SEED])
def test_t23_random_presses_keep_the_file_valid(qtbot: Any, iso: Path, seed: int) -> None:
    w = PresetWizard()
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    got: list[Any] = []
    w.finished_with.connect(lambda *a: got.append(a))
    rng = random.Random(seed)
    trail: list[str] = []
    lab_on = lab_off = 0
    for k in range(ACTIONS):
        targets = _targets(w)
        target = rng.choice(targets)
        trail.append(_poke(rng, w, target))
        quick = TR.quick_check(w.data)
        assert quick.ok, (k, trail[-5:], quick.errors[:3])
        assert w.quick is not None and w.quick.ok
        alive = TR.alive(quick.spec)
        assert w.btn_lab.isEnabled() == alive.ok, (k, trail[-5:], alive)
        lab_on += alive.ok
        lab_off += not alive.ok
        assert w.isVisible() and not got
    folder = paths.user_env_dir()
    assert not (folder.is_dir() and list(folder.glob("*.evoenv")))
    # 길이 실제로 여러 상태를 지났나(누름이 헛돌지 않았나)
    assert len(set(trail)) > 100
    assert lab_on > 0
    print(f"T23: {ACTIONS} presses, lab on {lab_on}, off {lab_off}")

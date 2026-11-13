"""6.2.0 화면 다듬기: 직접 조종 창 (polish/plan.md U13의 6.2.0 몫, 묶음 Hb).

① HUD는 실험실 캔버스와 같은 둥근 상자 ② 한 판이 끝나면 캔버스 가운데에 결과 띠(다시 시작하면 사라짐) ③ 점수 표에 「-0.00」 없음
④ 끈 「입력 기록 저장」의 까닭은 풍선에만(toolTip()은 그대로) ⑤ 「조종 키」 글 상자가 옆 칸 맨 위 ⑥ 센서가 없는 환경은 빈 표 대신 한 줄.
"""

from __future__ import annotations

import json
import math
from typing import Any

from PySide6.QtGui import QColor
from pytestqt.qtbot import QtBot

from evolab import paths
from evolab.app import strings as S
from evolab.app import style
from evolab.app.play import window as play_window
from evolab.app.play.window import COLOR_BG, PlayWindow
from evolab.spec.io import compile_env, load_env

ENV = paths.builtin_env_dir()
LANDER = ENV / "lander.evoenv"
GLIDER = ENV / "glider.evoenv"


class FakeClock:
    def __init__(self, dt: float) -> None:
        self.step_ms = play_window.MAX_STEPS_PER_TICK * dt * 1000.0
        self.ms = 0.0

    def elapsed(self) -> int:
        self.ms += self.step_ms
        return int(math.ceil(self.ms))

    def restart(self) -> int:
        self.ms = 0.0
        return 0

    def start(self) -> None:
        self.ms = 0.0


def make(qtbot: QtBot, spec: Any, show: bool = False) -> PlayWindow:
    w = PlayWindow(spec, seed=5)
    qtbot.addWidget(w)
    w.timer.stop()
    if show:
        w.show()
        qtbot.waitExposed(w)
        w.timer.stop()
    return w


def finish(w: PlayWindow) -> None:
    assert w.episode is not None
    w.clock = FakeClock(w.episode.dt)  # type: ignore[assignment]
    w._steps_since_clock = 0
    for _ in range(100000):
        if w.episode.done:
            return
        w._tick()
    raise AssertionError("episode did not finish")


def far(a: QColor, b: QColor) -> bool:
    return abs(a.red() - b.red()) + abs(a.green() - b.green()) + abs(a.blue() - b.blue()) > 6


def test_hud_box_and_result_band(qtbot: QtBot) -> None:
    w = make(qtbot, load_env(LANDER), show=True)
    w._tick()
    assert w.canvas.hud
    img = w.canvas.grab().toImage()
    # 둥근 상자의 왼쪽 테두리(x = 10): 바탕색이 아니다(상자 안쪽은 바탕과 같은 색이라 테두리로 본다)
    assert any(far(img.pixelColor(x, 20), COLOR_BG) for x in (9, 10, 11))
    assert w.canvas.result is None
    finish(w)
    res = w.last_result
    assert res is not None
    assert w.canvas.result == (S.PLAY_RESULT_BAND.format(label=res.end_label, score=style.fmt_score(res.score)),
                               S.PLAY_RESULT_KEYS)
    w.canvas.repaint()
    img = w.canvas.grab().toImage()
    cx, cy = img.width() // 2, img.height() // 2
    # 띠 가장자리 안쪽 몇 점은 띠 색이 깔려 있다(세계 그림과 섞여도 바탕색 그대로인 점은 없다)
    assert any(far(img.pixelColor(cx + dx, cy), COLOR_BG) for dx in (-40, 0, 40))
    w.restart()
    assert w.canvas.result is None


def test_score_table_has_no_negative_zero(qtbot: QtBot) -> None:
    assert play_window._signed(-0.0001) == "+0.00" and play_window._signed(-0.01) == "-0.01"
    assert play_window._signed(0.0) == "+0.00" and play_window._signed(12.345) == "+12.35"
    w = make(qtbot, load_env(LANDER))
    w._fill_tables(force=True)
    cells = [w.blocks.item(r, c).text() for r in range(w.blocks.rowCount()) for c in range(3)
             if w.blocks.item(r, c) is not None]
    assert "-0.00" not in cells


def test_save_button_reason_lives_only_in_the_tip(qtbot: QtBot) -> None:
    w = make(qtbot, load_env(LANDER))
    b = w.save_button
    assert not b.isEnabled() and b.toolTip() == "" and style.tip_text(b) == S.PLAY_WHY_SAVE
    finish(w)
    assert b.isEnabled() and style.tip_text(b) == ""
    w.restart()
    assert not b.isEnabled() and style.why_off(b) == S.PLAY_WHY_SAVE
    glider = make(qtbot, load_env(GLIDER))  # 지켜보기만: 제 풍선이 까닭
    assert glider.save_button.toolTip() == S.PLAY_SAVE_WATCH_ONLY
    assert style.tip_text(glider.save_button) == S.PLAY_SAVE_WATCH_ONLY


def test_key_box_is_at_the_top_of_the_side_panel(qtbot: QtBot) -> None:
    w = make(qtbot, load_env(LANDER), show=True)
    side = w.help_label.parentWidget()
    assert side is w.blocks.parentWidget()
    assert w.help_label.y() < w.status.y() < w.blocks.y()
    assert w.help_label.frameShape() != w.help_label.Shape.NoFrame
    assert w.help_label.text() == w.help_text()  # 글은 그대로


def test_env_without_sensors_shows_a_line_instead_of_an_empty_table(qtbot: QtBot) -> None:
    data = json.loads(LANDER.read_text(encoding="utf-8"))
    data["sensors"] = []
    w = make(qtbot, compile_env(data), show=True)
    assert not w.sensors.isVisible() and not w.sensors_title.isVisible()
    assert w.no_sensors.isVisible() and w.no_sensors.text() == S.PLAY_NO_SENSORS
    lander = make(qtbot, load_env(LANDER), show=True)
    assert lander.sensors.isVisible() and not lander.no_sensors.isVisible()

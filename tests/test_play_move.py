"""6.2.0 직접 조종의 밀기 키 (스포츠 설계 3.1 H, 6.1 T36, 묶음 Hb).

- 밀기(Move)의 키: xy → 오른쪽·왼쪽 키가 "<id>.x" +1/−1, 위·아래 키가 "<id>.y" +1/−1 · x → 좌우만 · heading → 위·아래가
  "<id>.forward" +1/−1, 왼쪽·오른쪽이 "<id>.turn" +1/−1. 채널 이름은 spec.symbols.MOVE_FIELDS를 따른다.
- 창에서 키를 누르면 그 출력이 나오고, 밀기 물체가 그 방향으로 움직인다. 도움말 글에 밀기 키 줄이 있다.
- 옛 환경(내장 예제 여섯)의 키 묶음은 얼린 v6.1.0과 같고, 얼린 v6.1.0이 만든 입력 기록은 지금 창에서 같은 결과로 재생된다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox
from pytestqt.qtbot import QtBot

from evolab import paths
from evolab.app import strings as S
from evolab.app.play import window as play_window
from evolab.app.play.record import verify_record
from evolab.app.play.window import PlayWindow, key_bindings, qt_key
from evolab.spec.io import compile_env, load_env
from evolab.spec.symbols import MOVE_FIELDS
from tests.oracle.frozen import frozen_run

DT = 1.0 / 60.0
OLD = ("lander", "track_car", "hill_car", "jump", "glider", "wheels")


def move_env(mode: str, **keys: list[str]) -> dict[str, Any]:
    """위에서 본 세계의 원 하나와 밀기 하나(키는 인자로)."""
    move = {"kind": "move", "id": "push", "label": "선수", "body": "disc", "mode": mode, "max_force": 30.0,
            "max_speed": 2.5, **keys}
    return {
        "version": 4, "name": "밀기",
        "world": {"view": "top", "gravity": [0.0, 0.0], "damping": 0.5, "bounds": [-50.0, -50.0, 50.0, 50.0],
                  "max_time": 10.0, "dt": DT},
        "agent": {"spawn": {"x": 0.0, "y": 0.0, "angle": 0.0}},
        "bodies": [{"id": "disc", "shapes": [{"kind": "circle", "r": 0.4}], "material": {"density": 2.0}}],
        "actuators": [move],
        "sensors": [{"kind": "body_state", "id": "me", "body": "disc", "fields": ["x", "y", "vx", "vy"]}],
        "scoring": {"blocks": [{"kind": "progress", "id": "go", "body": "disc", "weight": 1.0}]},
    }


ARROWS = {"keys_up": ["Up", "W"], "keys_down": ["Down", "S"], "keys_left": ["Left", "A"], "keys_right": ["Right", "D"]}


def table(mode: str) -> list[tuple[str, list[str], float]]:
    spec = compile_env(move_env(mode, **ARROWS))
    return [(spec.outputs[kb.output_index], kb.key_names, kb.value) for kb in key_bindings(spec)]


def test_move_xy_keys() -> None:
    assert table("xy") == [
        ("push.x", ["Right", "D"], 1.0), ("push.x", ["Left", "A"], -1.0),
        ("push.y", ["Up", "W"], 1.0), ("push.y", ["Down", "S"], -1.0),
    ]


def test_move_x_keys_are_left_and_right_only() -> None:
    assert table("x") == [("push.x", ["Right", "D"], 1.0), ("push.x", ["Left", "A"], -1.0)]


def test_move_heading_keys() -> None:
    assert table("heading") == [
        ("push.forward", ["Up", "W"], 1.0), ("push.forward", ["Down", "S"], -1.0),
        ("push.turn", ["Left", "A"], 1.0), ("push.turn", ["Right", "D"], -1.0),
    ]


@pytest.mark.parametrize("mode", sorted(MOVE_FIELDS))
def test_move_channels_follow_move_fields_and_labels_name_the_part(mode: str) -> None:
    spec = compile_env(move_env(mode, **ARROWS))
    kb = key_bindings(spec)
    assert sorted({spec.outputs[b.output_index] for b in kb}) == sorted(f"push.{t}" for t in MOVE_FIELDS[mode])
    assert all(b.label.startswith("선수 ") for b in kb)
    labels = {b.label for b in kb}
    assert len(labels) == len(kb)  # 키 줄마다 다른 설명


def test_move_without_keys_has_no_bindings_and_one_sided_keys_give_one_line() -> None:
    assert key_bindings(compile_env(move_env("xy"))) == []
    kb = key_bindings(compile_env(move_env("x", keys_right=["Right"], keys_up=["Up"])))  # x 방식은 위·아래 키를 듣지 않음
    assert [(b.key_names, b.value) for b in kb] == [(["Right"], 1.0)]


def make_window(qtbot: QtBot, data: dict[str, Any]) -> PlayWindow:
    w = PlayWindow(compile_env(data), seed=3)
    qtbot.addWidget(w)
    w.timer.stop()
    return w


@pytest.mark.parametrize(("mode", "key", "want"), [
    ("xy", Qt.Key.Key_Right, [1.0, 0.0]), ("xy", Qt.Key.Key_A, [-1.0, 0.0]),
    ("xy", Qt.Key.Key_Up, [0.0, 1.0]), ("xy", Qt.Key.Key_S, [0.0, -1.0]),
    ("x", Qt.Key.Key_Left, [-1.0]), ("x", Qt.Key.Key_Up, [0.0]),
    ("heading", Qt.Key.Key_W, [1.0, 0.0]), ("heading", Qt.Key.Key_Down, [-1.0, 0.0]),
    ("heading", Qt.Key.Key_Left, [0.0, 1.0]), ("heading", Qt.Key.Key_D, [0.0, -1.0]),
])
def test_window_keys_give_the_move_outputs(qtbot: QtBot, mode: str, key: Qt.Key, want: list[float]) -> None:
    w = make_window(qtbot, move_env(mode, **ARROWS))
    assert w.current_action() == [0.0] * len(want)
    qtbot.keyPress(w.canvas, key)
    assert w.current_action() == want
    qtbot.keyRelease(w.canvas, key)
    assert w.current_action() == [0.0] * len(want)


def test_window_opposite_keys_cancel_and_diagonals_combine(qtbot: QtBot) -> None:
    w = make_window(qtbot, move_env("xy", **ARROWS))
    qtbot.keyPress(w.canvas, Qt.Key.Key_Right)
    qtbot.keyPress(w.canvas, Qt.Key.Key_Up)
    assert w.current_action() == [1.0, 1.0]
    qtbot.keyPress(w.canvas, Qt.Key.Key_Left)
    assert w.current_action() == [0.0, 1.0]


def test_help_text_lists_the_move_keys(qtbot: QtBot) -> None:
    w = make_window(qtbot, move_env("xy", **ARROWS))
    text = w.help_label.text()
    for keys, label in (("Right / D", S.PLAY_KEY_RIGHT), ("Left / A", S.PLAY_KEY_LEFT), ("Up / W", S.PLAY_KEY_UP),
                        ("Down / S", S.PLAY_KEY_DOWN)):
        assert S.PLAY_KEY_LINE.format(keys=keys, label=label.format(name="선수")) in text
    assert w.space_pauses  # 밀기 키에 Space가 없으면 Space는 일시정지 그대로


def test_holding_a_move_key_pushes_the_disc_that_way(qtbot: QtBot) -> None:
    w = make_window(qtbot, move_env("xy", **ARROWS))
    ep = w.episode
    assert ep is not None
    qtbot.keyPress(w.canvas, Qt.Key.Key_Right)
    qtbot.keyPress(w.canvas, Qt.Key.Key_Down)
    for _ in range(120):
        ep.step(w.current_action())
    b = ep.bodies["disc"]
    x, y = float(b.position.x), float(b.position.y)
    assert x > 1.0 and y < -1.0 and math.isclose(x, -y, rel_tol=0.05)
    assert 2.0 < b.velocity.x <= 2.5 and b.velocity.y == pytest.approx(-b.velocity.x)  # 바닥 마찰 0.5가 조금 늦춤


# ---------------------------------------------------------------------------
# 옛 환경: 키 묶음과 입력 기록이 얼린 v6.1.0과 같다
_FROZEN_SCRIPT = r'''
import json, random, sys
from pathlib import Path
import numpy as np
from evolab import paths
from evolab.app.play.record import new_record
from evolab.app.play.window import key_bindings
from evolab.spec.io import load_env
from evolab.world.physics2d.episode import Episode, Trial

out, folder = sys.argv[1], Path(sys.argv[2])
names = sys.argv[3:]
envs = {p.stem: p for p in paths.builtin_envs()}
rows, scores = {}, []
for name in names:
    spec = load_env(envs[name])
    rows[name] = [[kb.output_index, list(kb.keys), kb.label, list(kb.key_names), kb.value] for kb in key_bindings(spec)]
    rng = random.Random(name)
    seed = 1000 + len(name)
    rec = new_record(spec, seed)
    ep = Episode(spec, None, Trial(seed))
    hold = [0.0] * len(spec.outputs)
    while not ep.done and len(rec.actions) < 100000:
        if len(rec.actions) % 15 == 0:
            hold = [float(rng.choice((-1.0, 0.0, 1.0))) if spec.actuator_channels[c].low < 0 else float(rng.choice((0.0, 1.0)))
                    for c in spec.outputs]
        rec.actions.append(list(hold))
        ep.step(list(hold))
    res = ep.result()
    rec.score, rec.outcome, rec.steps = res.score, res.outcome, res.steps
    rec.save(folder / (name + ".evoplay"))
    scores.append(res.score)
np.savez(out, bindings=np.array(json.dumps(rows, ensure_ascii=False)), scores=np.array(scores, dtype=float))
'''


@pytest.fixture(scope="module")
def frozen_records(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    folder = tmp_path_factory.mktemp("frozen_records")
    script = folder / "make_records.py"
    script.write_text(_FROZEN_SCRIPT, encoding="utf-8")
    got = frozen_run("v6.1.0", script, folder, *OLD)
    return folder, {"bindings": json.loads(str(got["bindings"])), "scores": got["scores"]}


def test_old_envs_keep_the_frozen_key_bindings(frozen_records: tuple[Path, dict[str, Any]]) -> None:
    _folder, got = frozen_records
    envs = {p.stem: p for p in paths.builtin_envs()}
    for name in OLD:
        now = [[kb.output_index, list(kb.keys), kb.label, list(kb.key_names), kb.value]
               for kb in key_bindings(load_env(envs[name]))]
        assert now == got["bindings"][name], name


class FakeClock:
    """창의 실제 시계 대신: elapsed()를 부를 때마다 MAX_STEPS_PER_TICK 스텝만큼 흐른다(test_play_v2와 같은 것)."""

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


def test_frozen_records_replay_the_same_in_the_window(qtbot: QtBot, frozen_records: tuple[Path, dict[str, Any]],
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    folder, got = frozen_records
    envs = {p.stem: p for p in paths.builtin_envs()}
    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda _p, _t, text, *a, **k: shown.append(text)))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda _p, _t, text, *a, **k: shown.append(text)))
    for name, score in zip(OLD, got["scores"], strict=True):
        spec = load_env(envs[name])
        same, res, _rec = verify_record(spec, folder / f"{name}.evoplay")
        assert same and np.float64(res.score).tobytes() == np.float64(score).tobytes(), name
        if not spec.outputs:
            continue
        w = PlayWindow(spec, seed=1)
        qtbot.addWidget(w)
        w.timer.stop()
        w.play_record(folder / f"{name}.evoplay")
        assert w.replaying is not None and w.episode is not None
        w.clock = FakeClock(w.episode.dt)  # type: ignore[assignment]
        w._steps_since_clock = 0
        for _ in range(100000):
            if w.episode.done:
                break
            w._tick()
        assert w.last_result is not None, name
        assert w.last_result.score == res.score and w.last_result.steps == res.steps, name
        assert shown[-1] == S.PLAY_REPLAY_SAME.format(score=res.score), name
        w.close()


def test_qt_key_names_used_by_move_files_exist() -> None:
    for names in ARROWS.values():
        assert all(qt_key(n) is not None for n in names)
    # Space는 점프 추진기에만(검사가 거절): 밀기 키가 있어도 Space는 일시정지 그대로
    assert not play_window.uses_space(key_bindings(compile_env(move_env("xy", **ARROWS))))

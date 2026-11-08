"""v2.1 검토 반영: 독립 검토·검증이 확인한 화면 결함의 회귀 시험.

- 직접 조종(바퀴 고르는 언덕차): 처음 몸에 없는 부품 안내, 없는 부품의 키는 도움말에서 빠짐, 센서 표 「(없음)」.
- 「배열로 늘리기」: 있음 진화를 켜면 사본 수식의 p.<원본 있음>이 사본 자기 있음으로 바뀜(사본 모터가 살아 있음), 번호가
  이미 쓰이면 가장 큰 번호 다음부터(wheel5·바퀴5), 원본 몸 수치를 함께 쓰는 수식은 알림(shared_params).
- 구조 탭: 「개수를 몸 수치 하나로」 줄의 진화 칸은 켜진 채 잠기고 그 몸 수치를 알림. 「없는 부품도 보기」를 끈 채 숨은
  물체 줄을 고르면 보기가 다시 켜짐(보이지 않는 부품이 고른 채 남지 않음).
- 캔버스 오른쪽 메뉴: 몸통·세계 물체는 「배열로 늘리기…」가 흐리고 이유가 풍선 도움말.
- 바퀴 고르는 언덕차의 기본 보기(editor.view): 에디터가 차 크기로 연다(해시·형식 버전은 그대로).
- 「몸 구조」 타일: 구조의 색이 다른 구조가 들고 나도 바뀌지 않음.
- T51(설계 9.3 v2.1 기준 5): 바퀴 구조가 서로 다른(앞부분이 겹치지 않는) 개체들로 관전 캔버스·경주(나란히·겹쳐)·경주 GIF.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from evolab import paths
from evolab.app import strings as S
from evolab.app.editor import tools as T
from evolab.app.editor.canvas import EditorCanvas
from evolab.app.editor.document import EnvDocument
from evolab.app.editor.structure import COL_EVOLVE, StructureTab, structure_rows
from evolab.app.editor.view import EditorView
from evolab.app.play.window import PlayWindow
from evolab.engine import seeds
from evolab.engine.replay import replay
from evolab.policy.genome import layout_for
from evolab.spec.editor_meta import view_box
from evolab.spec.io import compile_env, env_hash, feature_version, load_env
from evolab.world.physics2d.episode import Episode, Trial

ENVS = paths.builtin_env_dir()
WHEELS = ENVS / "wheels.evoenv"
LANDER = ENVS / "lander.evoenv"
WHEELS_HASH = "1a980d79bb796e5d23adda8cf00c77907662d07aa625603a6a92d176636c1cf2"
"""바퀴 고르는 언덕차의 환경 해시(기본 보기를 더해도 그대로여야 한다: editor는 해시 밖)."""


def play_window(qtbot: QtBot, path: Path, seed: int = 5) -> PlayWindow:
    w = PlayWindow(load_env(path), seed=seed)
    qtbot.addWidget(w)
    w.timer.stop()
    return w


def wheels_doc() -> EnvDocument:
    doc = EnvDocument()
    doc.from_builtin("wheels.evoenv")
    return doc


def editor_view(qtbot: QtBot, builtin: str) -> EditorView:
    view = EditorView()
    qtbot.addWidget(view)
    view.resize(1400, 900)
    view.show()
    view.doc.from_builtin(builtin)
    view.tabs.setCurrentWidget(view.tabs.structure_tab)
    QApplication.processEvents()
    return view


def motor_force(ep: Episode, motor_id: str) -> float | None:
    names = ep.actuator_channels()
    return next((float(mo.constraint.max_force) for mo in ep._motors if names[mo.channel_index] == motor_id), None)


# ---------------------------------------------------------------------------
# 직접 조종: 바퀴 고르는 언덕차의 처음 몸(바퀴 하나, 설계 D44)


def test_play_wheels_explains_the_starting_body_and_hides_dead_keys(qtbot: QtBot) -> None:
    w = play_window(qtbot, WHEELS)
    ep = w.episode
    assert ep is not None
    assert ep.absent == {"wheel2", "wheel3", "wheel4"}
    assert set(ep.dead_channels) == {"m2", "m3", "m4"}
    assert w.dead_outputs == {w.spec.outputs.index(m) for m in ("m2", "m3", "m4")}
    help_text = w.help_text()
    assert w.help_label.text() == help_text
    labels = {a.id: a.label for a in w.spec.file.actuators}
    assert S.PLAY_KEY_CCW.format(name=labels["m1"]) in help_text
    for dead in ("m2", "m3", "m4"):
        assert S.PLAY_KEY_CCW.format(name=labels[dead]) not in help_text
        assert S.PLAY_KEY_CW.format(name=labels[dead]) not in help_text
    assert not w.structure_label.isHidden()
    assert w.structure_label.text() == S.PLAY_STRUCTURE_NOTE.format(names="바퀴2, 바퀴3, 바퀴4")
    rows = {w.sensors.item(r, 0).text() for r in range(w.sensors.rowCount())}
    for k in (2, 3, 4):
        assert f"c{k} {S.INSPECT_ABSENT}" in rows
    assert "c1" in rows and f"c1 {S.INSPECT_ABSENT}" not in rows
    # 듣지 않는 키를 눌러도 실제 출력은 0(런타임 규칙 그대로), 사는 키는 그대로 기록된다
    m2 = next(kb for kb in w.bindings if w.spec.outputs[kb.output_index] == "m2")
    w.pressed.add(m2.keys[0])
    action = w.current_action()
    ep.step(action)
    assert ep.applied_outputs[ep.actuator_channels().index("m2")] == 0.0
    # 다시 시작해도(다른 시드) 같은 몸이라 같은 안내
    w.start(7)
    assert w.structure_label.text() == S.PLAY_STRUCTURE_NOTE.format(names="바퀴2, 바퀴3, 바퀴4")


def test_play_without_structure_is_unchanged(qtbot: QtBot) -> None:
    w = play_window(qtbot, LANDER)
    assert w.structure_label.isHidden() and w.structure_label.text() == ""
    assert w.dead_outputs == frozenset() and w.absent_inputs == frozenset()
    keys = "\n".join(S.PLAY_KEY_LINE.format(keys=" / ".join(kb.key_names), label=kb.label) for kb in w.bindings)
    assert w.help_text() == S.PLAY_HELP.format(keys=keys, pause=S.PLAY_PAUSE_KEYS["both"])
    rows = [w.sensors.item(r, 0).text() for r in range(w.sensors.rowCount())]
    assert rows == list(w.episode.inputs)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 「배열로 늘리기」


def test_replicate_with_evolve_gives_the_copy_its_own_on_gate(tmp_path: Path) -> None:
    doc = wheels_doc()
    refs = T.replicate(doc, "wheel4", 2, {"kind": "line", "end": [1.6, -0.25]}, True, False)
    assert refs == [("bodies", "wheel5"), ("joints", "j5"), ("actuators", "m5"), ("sensors", "c5"),
                    ("params", "wheel5_on")]
    assert doc.item(("bodies", "wheel5"))["exists"] == "=p.wheel5_on"  # type: ignore[index]
    torque = doc.item(("actuators", "m5"))["max_torque"]  # type: ignore[index]
    assert torque.endswith("if p.wheel5_on > 0.5 else 0") and "p.w4_on" not in torque
    assert doc.item(("actuators", "m4"))["max_torque"].endswith("if p.w4_on > 0.5 else 0")  # type: ignore[index]
    assert not [i for i in doc.run_check() if i.severity == "error"]
    # 사본이 원본과 함께 쓰는 몸 수치(위치·반지름·모터 힘 나누기의 다른 바퀴)는 알린다. 새로 만든 있음 수치는 아니다
    shared = T.shared_params(doc, refs)
    assert {"w4_x", "w4_r"} <= set(shared) and "wheel5_on" not in shared and "w4_on" not in shared
    # 바퀴4는 없고 바퀴5만 있는 몸: 바퀴5의 모터가 힘을 쓴다(고치기 전에는 0이었다)
    path = tmp_path / "wheels5.evoenv"
    doc.save(path)
    spec = load_env(path)
    layout = layout_for(spec)
    names = [g.id for g in layout.body]
    body = np.array([g.initial for g in layout.body])
    body[names.index("w4_on")] = 0.25
    body[names.index("wheel5_on")] = 0.75
    ep = Episode(spec, body, seeds.display_trial())
    assert ep.absent == {"wheel2", "wheel3", "wheel4"}
    assert motor_force(ep, "m5") == pytest.approx(300.0)  # 사본 모터가 산다(고치기 전에는 0이었다)
    # 힘 나누기 합은 사본마다 고칠 몫이다(shared_params가 알림): m1의 합은 wheel5를 세지 않고, m5의 합은 w4 대신 wheel5를 센다
    assert motor_force(ep, "m1") == pytest.approx(600.0)


def test_replicate_numbers_after_the_highest_taken_number() -> None:
    assert T.copy_shift("wheel1", 2, {"wheel1", "wheel2", "wheel3", "wheel4"}) == 3
    assert T.copy_shift("wheel1", 3, {"wheel1", "wheel3"}) == 2  # 하나라도 겹치면 모두 민다(wheel4, wheel5)
    assert T.copy_shift("wheel1", 3, {"wheel1", "wheel9x"}) == 0
    assert T.copy_shift("wheel", 2, {"wheel", "wheel2", "wheel7"}) == 6  # 끝 숫자가 없으면 원본이 1번
    assert T.copy_shift("바퀴1", 2, {"바퀴1", "바퀴2", "바퀴4"}, " ") == 3
    assert T.copy_shift("", 3, {"a"}) == 0
    doc = wheels_doc()
    refs = T.replicate(doc, "wheel1", 3, {"kind": "line", "end": [1.6, -0.25]}, True, False)
    ids = [cid for _s, cid in refs]
    assert ids == ["wheel5", "wheel6", "j5", "j6", "m5", "m6", "c5", "c6", "wheel5_on", "wheel6_on"]
    label = {cid: (doc.item((s, cid)) or {}).get("label") for s, cid in refs}
    assert [label[i] for i in ids] == ["바퀴5", "바퀴6", "바퀴5 축", "바퀴6 축", "바퀴5 모터", "바퀴6 모터", "바퀴5 접지",
                                       "바퀴6 접지", "바퀴5 있음", "바퀴6 있음"]
    assert doc.item(("joints", "j5"))["b"] == "wheel5"  # type: ignore[index]
    assert doc.item(("actuators", "m6"))["joint"] == "j6"  # type: ignore[index]
    all_labels = [str(x.get("label")) for sec in ("bodies", "joints", "actuators", "sensors", "params")
                  for x in doc.items(sec) if x.get("label")]
    assert len(all_labels) == len(set(all_labels))  # 겹치는 표시 이름이 없다
    assert not [i for i in doc.run_check() if i.severity == "error"]


# ---------------------------------------------------------------------------
# 구조 탭


def test_count_rows_show_evolve_on_and_locked(qtbot: QtBot) -> None:
    view = editor_view(qtbot, "hill_car.evoenv")
    T.replicate(view.doc, "wheel_r", 3, {"kind": "line", "end": [1.1, -0.25]}, True, True)
    tab = view.tabs.structure_tab
    assert isinstance(tab, StructureTab)
    qtbot.waitUntil(lambda: tab.table.rowCount() == 3, timeout=2000)
    rows = structure_rows(view.doc)
    assert [(r.param, r.driven_by, r.evolvable) for r in rows] == [(None, "wheel_r_count", False)] * 3
    for r in range(3):
        cell = tab.table.item(r, COL_EVOLVE)
        assert cell.checkState() == Qt.CheckState.Checked
        assert not cell.flags() & Qt.ItemFlag.ItemIsEnabled
        assert cell.toolTip() == S.ED_STRUCT_EVOLVE_BY_PARAM_TIP.format(id="wheel_r_count")
    before = view.doc.text()
    tab.set_evolve("wheel_r2", False)  # 잠긴 칸: 문서를 바꾸지 않는다
    QApplication.processEvents()
    assert view.doc.text() == before
    # 진화하는 몸 수치를 쓰지 않는 일반 수식은 예전처럼 꺼진 채 잠기고 수식 안내
    view.doc.set_value((*view.doc.item_path(("bodies", "wheel_r3")), "exists"), "=0.2 + 0.1")  # type: ignore[misc]
    tab.refresh()
    cell = tab.table.item(tab.row_of("wheel_r3"), COL_EVOLVE)
    assert cell.checkState() == Qt.CheckState.Unchecked and cell.toolTip() == S.EDP_EVOLVE_FORMULA_TIP


def test_selecting_a_hidden_row_turns_show_absent_back_on(qtbot: QtBot) -> None:
    view = editor_view(qtbot, "wheels.evoenv")
    tab = view.tabs.structure_tab
    canvas = view.canvas
    tab.show_absent.setChecked(False)
    assert not canvas.show_absent and not canvas.visible_ref(("bodies", "wheel3"))
    tab.table.selectRow(tab.row_of("wheel3"))
    assert view.doc.selection == [("bodies", "wheel3")]
    assert canvas.show_absent and tab.show_absent.isChecked()
    assert canvas.visible_ref(("bodies", "wheel3")) and canvas.visible_ref(("joints", "j3"))
    assert ("bodies", "wheel3") in canvas.structure_highlight
    # 다시 끄면 숨는 부품의 고르기를 푼다(예전 규칙 그대로)
    tab.show_absent.setChecked(False)
    assert view.doc.selection == [] and tab.current_body() is None
    # 보이는 부품을 고르는 것은 보기를 건드리지 않는다
    view.doc.select([("bodies", "wheel1")])
    assert not canvas.show_absent


# ---------------------------------------------------------------------------
# 캔버스 오른쪽 메뉴


def test_body_menu_disables_replicate_with_the_reason(qtbot: QtBot) -> None:
    doc = wheels_doc()
    doc.add("bodies", {"id": "rock", "owner": "world", "type": "static", "pos": [5.0, 0.5],
                       "shapes": [{"kind": "box", "w": 1.0, "h": 1.0}]}, select=False)
    c = EditorCanvas(doc)
    qtbot.addWidget(c)
    for body, reason in (("chassis", S.ED_REPLICATE_ROOT), ("rock", S.ED_REPLICATE_NEED_AGENT)):
        menu = c.body_menu(body)
        (act,) = menu.actions()
        assert menu.toolTipsVisible() and not act.isEnabled() and act.toolTip() == reason
    menu = c.body_menu("wheel2")
    (act,) = menu.actions()
    assert act.isEnabled() and act.toolTip() == S.ED_REPLICATE_HINT.format(name="wheel2")


# ---------------------------------------------------------------------------
# 바퀴 고르는 언덕차의 기본 보기


def test_wheels_default_view_frames_the_car_and_keeps_hash_and_version(qtbot: QtBot) -> None:
    spec = load_env(WHEELS)
    assert env_hash(spec.file) == WHEELS_HASH and feature_version(spec.file) == 3
    assert json.loads(WHEELS.read_text(encoding="utf-8"))["version"] == 3
    box = view_box(spec.file)
    assert box is not None
    ep = Episode(spec, None, seeds.display_trial())
    for b in ep.geometry().bodies:  # 처음 몸이 상자 안
        if b.moving:
            assert box[0] < b.pose[0] < box[2] and box[1] < b.pose[1] < box[3], b.id
    view = editor_view(qtbot, "wheels.evoenv")
    view.canvas.repaint()
    assert view.canvas.scale > 20.0  # 440m 언덕 전체(약 1.5px/m)가 아니라 차가 보이는 배율


# ---------------------------------------------------------------------------
# 「몸 구조」 타일의 색


def test_structure_colours_stay_with_their_structure(qtbot: QtBot, tmp_path: Path) -> None:
    from test_structure_tile import feed, struct_dashboard, tile, wheels, wheels_on

    spec = wheels()
    dash = struct_dashboard(qtbot, tmp_path, spec, "colours.ini")
    lay = dash.watch_meta.layout  # type: ignore[union-attr]
    t = tile(dash)

    def pop(structs: list[tuple[int, ...]]) -> Any:
        return lambda g: np.stack([wheels_on(lay, s) for s in structs])

    six = [(1,), (1, 4), (2, 3), (1, 2, 3), (1, 2, 3, 4), (4,)]
    seen: dict[tuple[str, ...], set[str]] = {}

    def record() -> None:
        dash.flush()
        t.update_data()
        for key, color in zip(t.series, t.colors, strict=True):
            if key is not None:
                seen.setdefault(key, set()).add(color)

    for g in range(1, 4):
        feed(dash, [g], pop(six * 4))
        record()
    first = dict(zip(t.series, t.colors, strict=True))
    # (1, 3)이 많이 나와 6개에 들어온다: 순서로 색을 주면 그 뒤 구조들의 색이 밀렸다
    for g in range(4, 10):
        feed(dash, [g], pop([(1, 3)] * 12 + six * 2))
        record()
    assert (("wheel1", "wheel3")) in t.series
    for key in t.series:
        if key is not None and key in first:
            assert seen[key] == {first[key]}, key  # 6개에 남은 구조는 처음 색 그대로
    assert len(set(t.colors)) == len(t.colors)  # 보이는 계열끼리 색이 겹치지 않는다
    assert [e.color for e in t.entries()] == t.colors


# ---------------------------------------------------------------------------
# T51: 바퀴 구조가 서로 다른 개체들의 관전·경주·GIF


def _wheels_genomes() -> tuple[Any, Any, list[np.ndarray], list[tuple[int, ...]]]:
    spec = compile_env(WHEELS.read_text(encoding="utf-8"))
    layout = layout_for(spec)
    names = [g.id for g in layout.body]
    structs: list[tuple[int, ...]] = [(4,), (2, 3), (1, 2, 3, 4), ()]
    genomes = []
    for on in structs:
        g = layout.initial().copy()
        for k in range(1, 5):
            g[names.index(f"w{k}_on")] = 0.5 if k in on else -0.5
        genomes.append(g)
    return spec, layout, genomes, structs


def test_t51_wheels_individuals_with_different_structures_in_spectator_race_and_gif(qtbot: QtBot) -> None:
    from evolab.app.export import video
    from evolab.app.lab.canvas import SpectatorCanvas
    from evolab.app.lab.view import build_scene
    from evolab.app.race.model import RaceEntry
    from evolab.app.race.view import RaceView

    spec, layout, genomes, structs = _wheels_genomes()
    trial = seeds.display_trial()
    moving = []
    for g, on in zip(genomes, structs, strict=True):
        ep = Episode(spec, layout.body_values(g), trial)
        assert ep.moving_bodies == ["chassis", *[f"wheel{k}" for k in on]]
        moving.append(list(ep.moving_bodies))
    # 관전 캔버스: 개체마다 자기 몸(앞부분이 겹치지 않는 목록)
    world, agents = build_scene(spec, layout, np.stack(genomes), trial)
    assert agents is not None and [[b.id for b in a] for a in agents] == moving
    trajs = []
    for g, want in zip(genomes, moving, strict=True):
        rep = replay(spec, layout, g, trial)
        assert [b.id for b in rep.agent] == want
        assert rep.trajectory.poses.shape[1] == len(want)  # 궤적도 그 개체의 움직이는 물체 수만큼
        trajs.append(rep.trajectory)
    c = SpectatorCanvas()
    qtbot.addWidget(c)
    c.resize(640, 400)
    c.set_scene(world, agents, channels=Episode(spec, None, trial).actuator_channels())
    c.set_population(trajs, ranks=np.arange(len(trajs)), champion=0)
    for t in (0.0, 0.5, 2.0):
        c.set_time(t)
        c.grab()
    # 그리는 순서(순위)와 상관없이 개체마다 자기 물체 수만큼의 모양(바퀴·차체는 모양 하나씩)
    assert {a.index: len(a.shapes) for a in c._agents} == {i: len(m) for i, m in enumerate(moving)}
    # 경주: 나란히와 겹쳐 그리기
    text = WHEELS.read_text(encoding="utf-8")
    h = env_hash(spec.file)[:16]
    entries = [RaceEntry(f"e{k}", "", color, text, h, genomes[k], k + 1)
               for k, color in enumerate(["#e76f51", "#2a9d8f", "#e9c46a", "#264653"])]
    view = RaceView()
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.show()
    try:
        view.set_race(entries)
        qtbot.waitUntil(lambda: view.ready, timeout=60000)
        view.pause()
        assert [[b.id for b in lane.replay.agent] for lane in view.lanes] == moving
        for mode in ("split", "overlay"):
            view.set_mode(mode)
            for t in (0.0, 1.0, 3.0):
                view.seek(t)
                view.grab()
        assert all(view.overlay_canvas.overlay_frame(k) is not None for k in range(len(entries)))
        # 경주 GIF 프레임(앞 몇 장만)
        reps = [(lane.replay, QColor(e.color), e.label) for lane, e in zip(view.lanes, entries, strict=True)]
        frames = video.render_race_frames(reps, 2, progress=lambda done, _total: done < 4)
        assert frames
    finally:
        view.shutdown()


def test_t51_wheels_replays_are_deterministic_per_structure() -> None:
    """같은 개체는 어느 시점에 다시 재생해도 같은 몸(구조는 몸 수치만의 함수)."""
    spec, layout, genomes, structs = _wheels_genomes()
    for g, on in zip(genomes, structs, strict=True):
        a = replay(spec, layout, g, Trial(seeds.derive(3, 1)))
        b = replay(spec, layout, g, Trial(seeds.derive(3, 1)))
        assert [x.id for x in a.agent] == [x.id for x in b.agent] == ["chassis", *[f"wheel{k}" for k in on]]
        assert a.trajectory.poses.tobytes() == b.trajectory.poses.tobytes()


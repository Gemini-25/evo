"""v1.2 뇌 들여다보기 (설계 4.3, D23·D25, 8.3절 T37): BrainView, BrainTile, replay(trace="brain")."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from test_dashboard import make_dashboard

from evolab import paths
from evolab.app import strings as S
from evolab.app.lab import brain_view as B
from evolab.app.lab.dashboard import DARK, LIGHT, _blend
from evolab.app.lab.watch_types import BrainSource, WatchMeta
from evolab.engine import seeds
from evolab.engine.replay import replay
from evolab.policy.genome import layout_for
from evolab.spec.io import compile_env
from evolab.world.physics2d.episode import OBS_CLIP, Trial


def env_text(name: str) -> str:
    return (paths.builtin_env_dir() / f"{name}.evoenv").read_text(encoding="utf-8")


def big_relu_text() -> str:
    """착륙선과 같은 세계에 은닉층 64-64, relu인 뇌(층마다 연결이 200개를 넘는다)."""
    data = json.loads(env_text("lander"))
    data["brain"] = {**data["brain"], "hidden": [64, 64], "activation": "relu"}
    return json.dumps(data, ensure_ascii=False)


def setup(text: str) -> tuple[Any, Any, WatchMeta]:
    spec = compile_env(text)
    lay = layout_for(spec)
    return spec, lay, WatchMeta.from_spec(spec, lay)


def genome(lay: Any, seed: int = 1, scale: float = 0.5) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, lay.dim)


def view_for(qtbot: QtBot, lay: Any, meta: WatchMeta, g: np.ndarray, theme: Any = LIGHT,
             size: tuple[int, int] = (480, 260)) -> B.BrainView:
    v = B.BrainView(theme)
    qtbot.addWidget(v)
    v.resize(*size)
    v.set_network(lay.brain, lay.brain_params(g[None, :])[0], list(meta.inputs), list(meta.outputs))
    return v


# ---------------------------------------------------------------------------
# replay(trace="brain") (D25)
@pytest.mark.parametrize("env", ["lander", "track_car", "hill_car"])
def test_brain_trace_replay_matches_full_trace(env: str) -> None:
    spec, lay, _meta = setup(env_text(env))
    g = genome(lay, 3)
    full = replay(spec, lay, g, seeds.display_trial(), trace=True)
    brain = replay(spec, lay, g, seeds.display_trial(), trace="brain")
    plain = replay(spec, lay, g, seeds.display_trial())
    for r in (brain, plain):
        assert r.result.score == full.result.score and r.result.steps == full.result.steps
        assert r.trajectory.poses.tobytes() == full.trajectory.poses.tobytes()
        assert r.trajectory.outputs.tobytes() == full.trajectory.outputs.tobytes()
    assert full.observations is not None and brain.observations is not None and full.activations is not None
    assert brain.observations.tobytes() == full.observations.tobytes()
    assert brain.activations is not None and len(brain.activations) == len(full.activations)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(brain.activations, full.activations, strict=True))
    assert brain.raw is None and brain.rays is None
    assert full.raw is not None and len(full.raw) == full.result.steps
    if env != "lander":  # 레이 센서가 있는 환경: 전체 추적만 레이를 남긴다
        assert full.rays is not None
    assert plain.observations is None and plain.activations is None and plain.raw is None
    with pytest.raises(ValueError):
        replay(spec, lay, g, seeds.display_trial(), trace="all")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# BrainView (T37)
def test_edge_layer_is_built_once_per_genome_and_size(qtbot: QtBot) -> None:
    spec, lay, meta = setup(env_text("lander"))
    g = genome(lay)
    v = view_for(qtbot, lay, meta, g)
    assert lay.brain.sizes == (9, 16, 3) and v.edges_total == 9 * 16 + 16 * 3 and not v.capped
    v.grab()
    assert v.edge_builds == 1 and v.frames >= 1
    rep = replay(spec, lay, g, seeds.display_trial(), trace="brain")
    v.set_trace(rep.observations, rep.activations, rep.trajectory.dt)
    for k in range(0, rep.result.steps, 7):
        v.set_time(k * rep.trajectory.dt)
        v.grab()
    v.set_network(lay.brain, lay.brain_params(g[None, :])[0], list(meta.inputs), list(meta.outputs))  # 같은 유전자
    v.grab()
    assert v.edge_builds == 1  # 재생 시각·같은 유전자로는 다시 그리지 않는다
    g2 = g.copy()
    g2[-1] += 0.25
    v.set_network(lay.brain, lay.brain_params(g2[None, :])[0], list(meta.inputs), list(meta.outputs))
    v.grab()
    assert v.edge_builds == 2
    v.set_theme(DARK)
    v.grab()
    assert v.edge_builds == 3  # 테마가 바뀌면 색이 바뀐다
    v.resize(500, 300)
    v._resize_timer.stop()  # 크기를 바꾸는 동안 늦추는 것을 건너뛴다
    v.grab()
    assert v.edge_builds == 4


def test_set_time_changes_only_neurons(qtbot: QtBot) -> None:
    spec, lay, meta = setup(env_text("lander"))
    g = genome(lay, 5)
    rep = replay(spec, lay, g, seeds.display_trial(), trace="brain")
    v = view_for(qtbot, lay, meta, g)
    assert v.neuron_values() is None and v.step_index() is None  # 추적 없음: 회색
    v.grab()
    v.set_trace(rep.observations, rep.activations, rep.trajectory.dt)
    dt = rep.trajectory.dt
    assert v.step_index(0.0) == 0 and v.step_index(2.5 * dt) == 2 and v.step_index(1e9) == rep.result.steps - 1
    assert v.step_index(-1.0) == 0
    v.set_time(0.0)
    first = v.neuron_values()
    assert first is not None and len(first) == 3
    assert np.array_equal(first[0], rep.observations[0].astype(np.float64))
    k = rep.result.steps // 2
    v.set_time(k * dt + dt / 3)
    mid = v.neuron_values()
    assert mid is not None and v.step_index() == k
    assert all(np.array_equal(m, a[k].astype(np.float64)) for m, a in zip(mid[1:], rep.activations, strict=True))
    assert any(not np.array_equal(a, b) for a, b in zip(first, mid, strict=True))
    levels = v.neuron_levels()
    assert levels is not None
    assert np.allclose(levels[0], np.clip(mid[0] / OBS_CLIP, -1, 1)) and v.scales == [OBS_CLIP, 1.0, 1.0]
    v.grab()
    assert v.edge_builds == 1
    # 같은 스텝 안의 시각은 다시 그리지 않는다(초당 30번 이하)
    frames = v.frames
    v.set_time(k * dt + dt / 2)
    qtbot.wait(5)
    assert v.frames == frames


def test_relu_layers_use_the_replay_maximum_and_edges_are_capped(qtbot: QtBot) -> None:
    spec, lay, meta = setup(big_relu_text())
    assert lay.brain.sizes == (9, 64, 64, 3) and lay.brain.activation == "relu"
    g = genome(lay, 7, 1.0)
    rep = replay(spec, lay, g, seeds.display_trial(), trace="brain")
    assert rep.activations is not None and float(np.abs(rep.activations[1]).max()) > 1.0  # relu 값이 1을 넘는다
    v = view_for(qtbot, lay, meta, g, DARK)
    assert v.capped and v.edges_total == 9 * 64 + 64 * 64 + 64 * 3
    assert v.edges_drawn == min(576, B.EDGE_CAP) + B.EDGE_CAP + min(192, B.EDGE_CAP)
    assert v.edges_drawn > B.BACKGROUND_EDGES  # 큰 신경망은 배경 스레드
    assert not v.build_static()
    assert v.wait_edges() and v.edge_builds == 1
    v.set_trace(rep.observations, rep.activations, rep.trajectory.dt)
    assert v.relu_scaled
    tops = [float(np.abs(a).max()) for a in rep.activations[:2]]
    assert v.scales[1:3] == pytest.approx(tops) and v.scales[3] == 1.0
    for k in range(0, rep.result.steps, 5):
        v.set_time(k * rep.trajectory.dt)
        levels = v.neuron_levels()
        assert levels is not None and all(np.all(np.abs(lv) <= 1.0) for lv in levels)
    v.grab()
    assert v.edge_builds == 1
    # 층마다 |w|가 큰 EDGE_CAP개(결정적)
    w = v.weights[1]
    rows, cols, levels = B.select_edges(w)
    assert len(rows) == B.EDGE_CAP and set(np.abs(levels)) <= set(range(1, B.EDGE_LEVELS + 1))
    chosen = np.abs(w[rows, cols])
    assert chosen.min() >= np.sort(np.abs(w).ravel())[-B.EDGE_CAP]
    assert np.all(np.sign(levels) == np.sign(w[rows, cols]))
    assert np.all(np.diff(chosen) >= 0)  # 약한 선부터 그려 센 선이 위에


def test_select_edges_of_a_zero_brain_draws_nothing() -> None:
    rows, cols, levels = B.select_edges(np.zeros((9, 16)))
    assert len(rows) == len(cols) == len(levels) == 0


def test_frame_paint_time_9_16_3(qtbot: QtBot) -> None:
    spec, lay, meta = setup(env_text("lander"))
    g = genome(lay, 2)
    rep = replay(spec, lay, g, seeds.display_trial(), trace="brain")
    v = view_for(qtbot, lay, meta, g, size=(480, 300))
    v.set_trace(rep.observations, rep.activations, rep.trajectory.dt)
    v.grab()  # 첫 그림(연결선 층 만들기 포함)은 빼고 잰다
    times = []
    for k in range(min(rep.result.steps, 200)):
        v.set_time(k * rep.trajectory.dt)
        frames = v.frames
        v.grab()  # paintEvent를 바로 부른다(오프스크린에서 repaint는 창이 노출되기 전이면 그리지 않는다)
        assert v.frames == frames + 1
        times.append(v.last_paint_ms)
    med, p95 = float(np.median(times)), float(np.percentile(times, 95))
    print(f"\n[brain] 9-16-3 frame paint: median {med:.3f} ms, p95 {p95:.3f} ms, n={len(times)}, builds {v.edge_builds}")
    assert v.edge_builds == 1
    assert med < 2.0  # 목표(설계 T37): 2ms 미만


def test_node_tooltips(qtbot: QtBot) -> None:
    spec, lay, meta = setup(env_text("lander"))
    g = genome(lay, 4)
    v = view_for(qtbot, lay, meta, g)
    v.grab()
    name, label = meta.inputs[2]
    assert v.node_tip(0, 2) == S.BRAIN_NODE_INPUT_TIP.format(name=name, label=label, value="-")
    bias = f"{float(v.biases[0][4]):+.2f}"
    assert v.node_tip(1, 4) == S.BRAIN_NODE_HIDDEN_TIP.format(k=1, j=5, value="-", bias=bias)
    rep = replay(spec, lay, g, seeds.display_trial(), trace="brain")
    v.set_trace(rep.observations, rep.activations, rep.trajectory.dt)
    v.set_time(0.0)
    out_name, out_label = meta.outputs[1]
    value = f"{float(rep.activations[-1][0, 1]):+.2f}"
    bias = f"{float(v.biases[1][1]):+.2f}"
    assert v.node_tip(2, 1) == S.BRAIN_NODE_OUTPUT_TIP.format(name=out_name, label=out_label, value=value, bias=bias)
    x, y = v._pos[1][4]
    assert v.node_at(float(x), float(y)) == (1, 4) and v.node_at(1.0, 1.0) is None
    assert B.short_name("abcdefghijklmnop") == "abcdefghijk" + B.ELLIPSIS and B.short_name("pad.x") == "pad.x"


# ---------------------------------------------------------------------------
# BrainTile: 원천(D24)과 제목·범례
def brain_dashboard(qtbot: QtBot, tmp_path: Path, meta: WatchMeta) -> Any:
    dash = make_dashboard(qtbot, tmp_path, "brain.ini")
    dash.set_watch_meta(meta)
    dash.set_page("fun")
    dash.set_tile_visible("genemap", False)  # 이 시험은 지도를 보지 않는다(축 맞추기를 건너뛴다)
    return dash


def test_brain_tile_sources_titles_and_time(qtbot: QtBot, tmp_path: Path) -> None:
    spec, lay, meta = setup(env_text("lander"))
    dash = make_dashboard(qtbot, tmp_path, "brain0.ini")
    dash.set_page("fun")
    tile = dash.tile("brain")
    assert isinstance(tile, B.BrainTile)
    dash.flush()
    assert tile.stack.currentWidget() is tile.message and tile.message.text() == S.BRAIN_EMPTY  # 이름표 없음
    dash.set_watch_meta(meta)
    dash.flush()
    assert tile.message.text() == S.BRAIN_EMPTY and tile.source_kind is None  # 챔피언도 원천도 없음

    # 챔피언 멈춘 그림: 보는 세대까지의 마지막 챔피언
    c1, c3 = genome(lay, 11), genome(lay, 13)
    dash.history.add_champion(1, c1, 1.0)
    dash.history.add_champion(3, c3, 2.0)
    dash.set_brain_source(None)
    dash.flush()
    assert tile.source_kind == "champion" and tile.source_generation == 3
    assert tile.title_text == S.BRAIN_TITLE_CHAMPION.format(gen=3)
    assert tile.legend.text() == S.BRAIN_TITLE_CHAMPION.format(gen=3)
    assert S.BRAIN_LEGEND_STILL in tile.note.text() and not tile.brain.traced
    assert np.array_equal(tile.brain.weights[0].ravel(), lay.brain_params(c3[None, :])[0][: 9 * 16])
    dash.set_focus(2)
    dash.flush()
    assert tile.source_generation == 1  # 보는 세대 2까지의 챔피언은 1세대

    # 재생 중(①): 추적이 오기 전에는 계산 중, 오면 뉴런이 재생 시각을 따른다
    g = genome(lay, 21)
    dash.set_brain_source(BrainSource(S.LAB_LABEL_REP.format(gen=2), "playing", 2, g, None))
    dash.flush()
    assert tile.source_kind == "playing" and S.BRAIN_PREPARING in tile.note.text()
    builds = tile.brain.edge_builds
    rep = replay(spec, lay, g, seeds.display_trial(), trace="brain")
    dash.set_brain_source(BrainSource(S.LAB_LABEL_REP.format(gen=2), "playing", 2, g, rep))
    dash.show_time(0.0)
    dash.flush()
    assert tile.brain.traced and S.BRAIN_PREPARING not in tile.note.text() and S.BRAIN_LEGEND in tile.note.text()
    assert tile.title_text == S.BRAIN_TITLE_PLAYING.format(label=S.LAB_LABEL_REP.format(gen=2), t=0.0)
    tile.brain.grab()
    a = tile.brain.neuron_values()
    dash.show_time(1.0)
    b = tile.brain.neuron_values()
    assert a is not None and b is not None and any(not np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert tile.title_text == S.BRAIN_TITLE_PLAYING.format(label=S.LAB_LABEL_REP.format(gen=2), t=1.0)
    dash.show_time(1e6)  # 제목의 시각은 재생 길이에서 멈춘다
    assert tile.title_text == S.BRAIN_TITLE_PLAYING.format(label=S.LAB_LABEL_REP.format(gen=2),
                                                          t=rep.result.steps * rep.trajectory.dt)
    tile.brain.grab()
    assert tile.brain.edge_builds <= builds + 1  # 같은 유전자: 연결선 층은 한 번만

    # 관전 중 대표(②)
    dash.set_brain_source(BrainSource("", "live", 4, g, rep))
    dash.show_time(0.5)
    dash.flush()
    assert tile.source_kind == "live"
    assert tile.title_text == S.BRAIN_TITLE_LIVE.format(gen=4, t=0.5)
    # 유전자 크기가 맞지 않는 원천은 챔피언 멈춘 그림으로
    dash.set_brain_source(BrainSource("", "live", 4, np.zeros(lay.dim + 1), None))
    dash.flush()
    assert tile.source_kind == "champion"


def test_brain_tile_notes_for_relu_and_capped_networks(qtbot: QtBot, tmp_path: Path) -> None:
    spec, lay, meta = setup(big_relu_text())
    dash = brain_dashboard(qtbot, tmp_path, meta)
    tile = dash.tile("brain")
    g = genome(lay, 9, 1.0)
    rep = replay(spec, lay, g, Trial(5, None), trace="brain")
    dash.set_brain_source(BrainSource("x", "playing", 1, g, rep))
    dash.flush()
    notes = tile.note.text()
    assert S.BRAIN_RELU_NOTE in notes
    assert S.BRAIN_EDGES_CAPPED.format(total=tile.brain.edges_total, per=B.EDGE_CAP) in notes
    # 추적이 없으면 relu 알림은 없다(뉴런을 칠하지 않으므로)
    dash.set_brain_source(BrainSource("x", "playing", 1, g, None))
    dash.flush()
    assert S.BRAIN_RELU_NOTE not in tile.note.text()


def test_brain_tile_shows_the_viewed_representative_until_a_champion_exists(qtbot: QtBot, tmp_path: Path) -> None:
    """검증 전(고속 모드의 첫 세대들)에는 챔피언이 없다: 빈 안내 대신 보는 세대 대표의 멈춘 그림, 챔피언이 생기면 그것."""
    spec, lay, meta = setup(env_text("lander"))
    dash = make_dashboard(qtbot, tmp_path, "brain_rep.ini")
    dash.set_page("fun")
    dash.set_watch_meta(meta)
    tile = dash.tile("brain")
    pops = {g: np.stack([genome(lay, 100 * g + k) for k in range(4)]) for g in (1, 2, 3)}
    for g, pop in pops.items():
        dash.add_generation(g, None, pop, 2, None)
    dash.flush()
    assert tile.source_kind == "rep" and tile.source_generation == 3
    assert tile.title_text == S.BRAIN_TITLE_REP.format(gen=3) and S.BRAIN_LEGEND_STILL in tile.note.text()
    assert np.array_equal(tile.brain.weights[0].ravel(), lay.brain_params(pops[3][2][None, :])[0][: 9 * 16])
    dash.set_focus(1)
    dash.flush()
    assert tile.source_kind == "rep" and tile.source_generation == 1
    dash.add_generation(3, None, pops[3], 2, (2, pops[2][0], 1.5))  # 2세대에 뽑힌 챔피언
    dash.set_focus(None)
    dash.flush()
    assert tile.source_kind == "champion" and tile.source_generation == 2
    assert tile.title_text == S.BRAIN_TITLE_CHAMPION.format(gen=2)


def test_brain_tile_survives_meta_change_and_dark_theme_tokens(qtbot: QtBot, tmp_path: Path) -> None:
    _spec, lay, meta = setup(env_text("lander"))
    dash = brain_dashboard(qtbot, tmp_path, meta)
    tile = dash.tile("brain")
    dash.history.add_champion(2, genome(lay, 1), 1.0)
    dash.set_brain_source(None)
    dash.flush()
    assert tile.brain.has_network
    _spec2, _lay2, meta2 = setup(env_text("hill_car"))  # 다른 유전자 구조: 기록·그림을 비운다
    dash.set_watch_meta(meta2)
    dash.flush()
    assert not tile.brain.has_network and tile.message.text() == S.BRAIN_EMPTY
    # 연결선 색은 테마 토큰(+ series[0], − series[1])을 바탕(surface)과 섞은 것, 가장 센 단계는 토큰 그대로
    assert B._edge_pen(DARK, 8, True).color().name() == DARK.series[0]
    assert B._edge_pen(DARK, -8, False).color().name() == DARK.series[1]
    weak = B._edge_pen(DARK, 2, True).color()
    assert weak.alpha() == 255 and weak.name() == _blend(DARK.series[0], DARK.surface, B.edge_strength(2))

"""v1.2 대시보드 「보는 재미」: 쪽 탭, 갱신 펌프, 새 타일 넷, 설계 7.3 API (설계 8.2절 T28·T30~T36)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QColor, QPalette, QWheelEvent
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot
from test_dashboard import make_dashboard, synthetic_rows
from test_history import TRACK_KEY, fake_generation, meta_for, stats_row, write_run

from evolab import paths
from evolab.app import strings as S
from evolab.app.lab import dashboard as D
from evolab.app.lab import watch_tiles as W
from evolab.app.lab.canvas import SpectatorCanvas
from evolab.app.lab.watch_types import BrainSource, WatchMeta
from evolab.engine import seeds
from evolab.runs import observe as ob
from evolab.runs.store import RunStore
from evolab.spec.io import compile_env
from evolab.world.physics2d.episode import Episode

DATA = Path(__file__).parent / "data"
FRAME = ob.GridFrame(-10.0, -5.0, 0.8, 32, 24)


def feed(dash: D.Dashboard, env: str, gens: range | list[int], *, pop: int = 6, frame: ob.GridFrame | None = None,
         key: Any = None, learner: Any = 0, champion_every: int = 0) -> dict[int, tuple[ob.Observation, np.ndarray]]:
    """학습 중처럼 add_row + add_generation. key: 세계 열쇠(고정 글자 또는 세대 → 글자 함수)."""
    lay = meta_for(env).layout
    out = {}
    for g in gens:
        wk = key(g) if callable(key) else key
        lr = learner(g) if callable(learner) else learner
        obs, X = fake_generation(g, lay, pop=pop, frame=frame, world_key=wk, learner=lr)
        out[g] = (obs, X)
        champ = (g, X[obs.representative]) if champion_every and g % champion_every == 1 else None
        extra = {"champion_score": repr(float(obs.scores.max()))} if champ is not None else {}
        dash.add_row({**stats_row(obs), "evaluations": str(g * pop * 2), "elapsed": str(g * 1.5), **extra})
        dash.add_generation(g, obs.to_arrays(), X, obs.representative, champ)
    return out


def fun_dashboard(qtbot: QtBot, tmp_path: Path, env: str = "lander", name: str = "fun.ini", *,
                  genemap: bool = True) -> D.Dashboard:
    """「보는 재미」 쪽 대시보드(화면에 띄우지 않아 바로 갱신). genemap=False면 유전자 지도 타일을 숨긴다: 그 타일을
    보지 않는 테스트에서 세대마다 축 맞추기(eigh)를 하지 않게(테스트 프로세스는 BLAS 스레드를 묶지 않아 부하가 큰
    기계에서 eigh 한 번이 1초 가까이 걸린 적이 있다. 앱은 BLAS 스레드 1개)."""
    dash = make_dashboard(qtbot, tmp_path, name)
    dash.set_watch_meta(meta_for(env))
    dash.set_page("fun")
    if not genemap:
        dash.set_tile_visible("genemap", False)
    return dash


# ---------------------------------------------------------------------------
# 쪽 (T30)
def test_new_tiles_live_on_the_fun_page_and_learning_page_is_unchanged(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    assert D.TILE_PAGES == {"learning": D.TILE_IDS, "fun": D.FUN_TILE_IDS}
    assert D.FUN_TILE_IDS == ("genemap", "brain", "density", "actions", "body", "structure")  # v2.1 「몸 구조」는 끝
    assert "brain" in D.TILE_PAGES["fun"] and D.PAGE_OF["brain"] == "fun"
    assert dash.page == "learning" and dash.page_tabs.currentIndex() == 0
    assert dash.tile_order() == list(D.TILE_IDS) == dash.tile_order("learning")
    assert dash.tile_order("fun") == list(D.FUN_TILE_IDS)
    assert [dash.page_tabs.tabText(i) for i in range(2)] == [S.DASH_PAGES["learning"], S.DASH_PAGES["fun"]]
    for tid in D.FUN_TILE_IDS:
        tile = dash.tile(tid)
        assert tile.title.text() == S.DASH_TILE_TITLES[tid] and tile.title.toolTip() == S.DASH_TILE_HELP[tid]
        assert not tile.isVisibleTo(dash)  # 지금 쪽이 아니면 숨김(사용자가 숨긴 것과는 별개)
        assert dash.is_tile_visible(tid)
    assert all(dash.tile(t).isVisibleTo(dash) for t in D.TILE_IDS)
    # 「보는 재미」는 고를 때까지 굵게, 풍선 도움말 끝에 새 그래프 안내
    assert dash.page_tabs.bold_index == 1 and not dash.fun_seen
    assert not dash.page_tabs.grab().isNull()  # 굵은 탭 그리기
    assert dash.page_tabs.tabSizeHint(1).width() >= dash.page_tabs.fontMetrics().horizontalAdvance(S.DASH_PAGES["fun"])
    assert dash.page_tabs.tabToolTip(1).endswith(S.DASH_FUN_NEW_TIP)
    assert dash.page_tabs.tabToolTip(0) == S.DASH_PAGES_TIP["learning"]
    dash.page_tabs.setCurrentIndex(1)  # 탭을 눌러도 같다
    assert dash.page == "fun" and dash.fun_seen and dash.page_tabs.bold_index == -1
    assert dash.page_tabs.tabToolTip(1) == S.DASH_PAGES_TIP["fun"]
    assert all(dash.tile(t).isVisibleTo(dash) for t in D.FUN_TILE_IDS)
    assert not any(dash.tile(t).isVisibleTo(dash) for t in D.TILE_IDS)
    with pytest.raises(ValueError):
        dash.set_page("nope")


def test_page_and_fun_seen_are_saved_and_old_layouts_get_the_new_tiles(qtbot: QtBot, tmp_path: Path) -> None:
    # v1.1이 저장한 6개 배치
    s = QSettings(str(tmp_path / "old.ini"), QSettings.Format.IniFormat)
    s.beginGroup("dashboard")
    s.setValue("order", "perf,trend,heatmap,diversity,blocks,ends")
    s.setValue("hidden", "heatmap")
    s.setValue("columns", 3)
    s.endGroup()
    s.sync()
    dash = make_dashboard(qtbot, tmp_path, "old.ini")
    assert dash.tile_order("learning") == ["perf", "trend", "heatmap", "diversity", "blocks", "ends"]
    assert dash.tile_order("fun") == list(D.FUN_TILE_IDS)
    assert not dash.is_tile_visible("heatmap") and all(dash.is_tile_visible(t) for t in D.FUN_TILE_IDS)
    assert dash.page == "learning" and dash.columns == 3
    dash.set_page("fun")
    dash.move_tile("body", 0)
    dash.set_tile_visible("density", False)
    dash.move_tile_by("genemap", 1)  # 같은 쪽의 보이는 타일 사이에서
    order = dash.tile_order("fun")
    assert order[0] == "body" and order.index("genemap") == order.index("brain") + 1
    dash.tile("actions").set_mode("rep")
    dash.tile("genemap").set_color_mode("success")
    dash.tile("density").set_window("all")

    again = make_dashboard(qtbot, tmp_path, "old.ini")
    assert again.page == "fun" and again.fun_seen and again.page_tabs.currentIndex() == 1
    assert again.page_tabs.bold_index == -1
    assert again.tile_order("fun") == order and again.tile_order("learning")[0] == "perf"
    assert not again.is_tile_visible("density") and not again.is_tile_visible("heatmap")
    assert again.tile("actions").mode == "rep" and again.tile("genemap").color_mode == "success"
    assert again.tile("density").window == "all"
    # v1.1 프로그램이 읽어도 모르는 타일은 거른다(같은 "order" 열쇠에 두 쪽이 차례로)
    s = QSettings(str(tmp_path / "old.ini"), QSettings.Format.IniFormat)
    saved = s.value("dashboard/order")
    assert str(saved).split(",")[:6] == again.tile_order("learning")
    # 기본 배치로 되돌리기는 두 쪽 모두
    again.reset_layout()
    assert again.tile_order("fun") == list(D.FUN_TILE_IDS) and again.tile_order("learning") == list(D.TILE_IDS)
    assert all(again.is_tile_visible(t) for t in D.ALL_TILE_IDS)


def test_tiles_menu_lists_only_the_current_page(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    dash._fill_tiles_menu()
    titles = [a.text() for a in dash.tiles_menu.actions() if a.isCheckable()]
    assert titles == [S.DASH_TILE_TITLES[t] for t in D.TILE_IDS]
    dash.set_page("fun")
    dash._fill_tiles_menu()
    actions = [a for a in dash.tiles_menu.actions() if a.isCheckable()]
    assert [a.text() for a in actions] == [S.DASH_TILE_TITLES[t] for t in D.FUN_TILE_IDS]
    actions[1].setChecked(False)
    assert not dash.is_tile_visible("brain") and not dash.brain_visible()


# ---------------------------------------------------------------------------
# 갱신 펌프 (T31)
def test_pump_updates_visible_tiles_one_by_one_and_scrolled_out_tiles_later(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    dash.resize(900, 420)  # 두 줄만 보인다: 아래 줄 타일은 스크롤 밖
    dash.show()
    qtbot.waitExposed(dash)
    rows = synthetic_rows(41)
    dash.set_rows(rows[:40])
    qtbot.waitUntil(lambda: not any(dash.tile(t).dirty for t in dash.tile_order() if dash._in_view(dash.tile(t))))
    in_view = [t for t in dash.tile_order() if dash._in_view(dash.tile(t))]
    out_of_view = [t for t in dash.tile_order() if t not in in_view]
    assert in_view and out_of_view, (in_view, out_of_view)
    assert all(dash.tile(t).dirty for t in out_of_view)
    steps = dash.pump_steps
    dash.add_row(rows[40])
    # add_row는 표시만 하고 그리지 않는다(이벤트 루프 차례마다 하나씩)
    assert all(dash.tile(t).dirty for t in in_view)
    qtbot.waitUntil(lambda: not any(dash.tile(t).dirty for t in in_view))
    assert dash.pump_steps - steps == len(in_view)
    assert all(dash.tile(t).dirty for t in out_of_view)
    # 스크롤하면 들어온 타일을 갱신한다
    bar = dash.scroll.verticalScrollBar()
    bar.setValue(bar.maximum())
    qtbot.waitUntil(lambda: not any(dash.tile(t).dirty for t in out_of_view if dash._in_view(dash.tile(t))))
    assert any(not dash.tile(t).dirty for t in out_of_view)
    # 다른 쪽 타일은 갱신하지 않는다
    assert all(dash.tile(t).dirty for t in D.FUN_TILE_IDS)
    # flush는 지금 쪽의 dirty 타일을 모두 바로
    dash.add_row(synthetic_rows(42)[-1])
    dash.flush()
    assert not any(dash.tile(t).dirty for t in dash.tile_order())
    assert len(dash.tile("trend").lines["score_best"].getData()[0]) == 42


def test_dashboard_that_is_not_shown_updates_right_away(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    assert not dash.isVisible()
    dash.set_rows(synthetic_rows(10))
    dash.add_row(synthetic_rows(11)[-1])
    assert not any(dash.tile(t).dirty for t in D.TILE_IDS)
    assert dash.pump_steps == 0
    assert all(dash.tile(t).dirty for t in D.FUN_TILE_IDS)  # 다른 쪽
    dash.set_watch_meta(meta_for("lander"))
    dash.set_page("fun")
    assert not any(dash.tile(t).dirty for t in D.FUN_TILE_IDS)


def test_hidden_page_is_refreshed_when_switched_to(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    dash.set_watch_meta(meta_for("lander"))
    dash.resize(1200, 900)
    dash.show()
    qtbot.waitExposed(dash)
    feed(dash, "lander", range(1, 6))
    assert all(dash.tile(t).dirty for t in D.FUN_TILE_IDS)
    dash.set_page("fun")
    qtbot.waitUntil(lambda: not dash.tile("genemap").dirty)
    assert dash.tile("genemap").shown_generation == 5


# ---------------------------------------------------------------------------
# 배경 읽기(대시보드 쪽, T28)
def test_load_history_reads_in_background_and_drops_stale_results(qtbot: QtBot, tmp_path: Path) -> None:
    lay = meta_for("lander").layout
    big = tmp_path / "big"
    small = tmp_path / "small"
    rows_big = write_run(big, [fake_generation(g, lay, pop=3, seed=1) for g in range(1, 150)])
    rows_small = write_run(small, [fake_generation(g, lay, pop=3, seed=2) for g in range(1, 6)])
    dash = fun_dashboard(qtbot, tmp_path)
    dash.load_history(big, rows_big)
    assert dash.history_loading is not None
    dash.load_history(small, rows_small)  # 되감기·다른 실행: 앞 읽기의 결과는 버린다
    assert dash.wait_history()
    qtbot.wait(50)
    assert dash.history.gens == [1, 2, 3, 4, 5]
    assert dash.history.load_summary["observed"] == 5
    tag = dash._load_tag
    dash._on_history_chunk(tag - 1, [], 1, 1)  # 늦게 온 앞 꼬리표 결과
    assert dash.history.gens == [1, 2, 3, 4, 5]
    # 이름표보다 먼저 온 load_history는 이름표가 오면 시작한다
    late = make_dashboard(qtbot, tmp_path, "late.ini")
    late.load_history(small, rows_small)
    assert late.history_loading is None and late._pending_load is not None
    late.set_watch_meta(meta_for("lander"))
    assert late.wait_history() and late.history.gens == [1, 2, 3, 4, 5]
    late.load_history(None, [])
    assert late.history.gens == [] and late.history_loading is None
    # clear()도 기록과 보는 세대를 비운다
    dash.set_focus(3)
    dash.clear()
    assert dash.history.gens == [] and dash.focus is None


def test_live_generations_and_api(qtbot: QtBot, tmp_path: Path) -> None:
    """add_generation, set_focus, exact_genome·rep_genome·champion_at, 이름표가 없으면 무시."""
    dash = make_dashboard(qtbot, tmp_path)
    obs, X = fake_generation(1, meta_for("lander").layout)
    dash.add_generation(1, obs.to_arrays(), X, obs.representative, None)
    assert dash.history.gens == []  # 이름표 전
    dash.set_watch_meta(meta_for("lander"))
    data = feed(dash, "lander", range(1, 71), pop=4, champion_every=20)
    assert dash.history.gens == list(range(1, 71)) and dash.view_generation() == 70
    obs70, X70 = data[70]
    assert dash.exact_genome(70, 2).tobytes() == X70[2].tobytes()
    assert dash.exact_genome(5, 0) is None  # 64세대 밖
    assert dash.rep_genome(70).tobytes() == X70[obs70.representative].tobytes()
    assert dash.rep_index(70) == obs70.representative
    c = dash.champion_at(45)
    assert c is not None and c[0] == 41 and c[1].tobytes() == data[41][1][data[41][0].representative].tobytes()
    assert dash.champion_at(0) is None
    assert dash.history.champion_entry_at(45)[2] == pytest.approx(float(data[41][0].scores.max()))  # 통계 줄 값
    dash.add_generation(70, data[70][0].to_arrays(), X70, obs70.representative, (70, X70[0], 12.5))
    assert dash.history.champion_entry_at(70)[2] == 12.5 and dash.champion_at(70)[1].tobytes() == X70[0].tobytes()
    assert dash.observation(70) is not None and dash.observation(70).trial_seed == 1070
    dash.set_focus(10)
    assert dash.focus == 10 and dash.view_generation() == 10
    dash.set_focus(None)
    assert dash.view_generation() == 70
    # 관찰 기록이 없는 세대(관찰을 끈 학습): 집단의 세대 대표로
    o71, X71 = fake_generation(71, meta_for("lander").layout, pop=4)
    dash.add_row(stats_row(o71))
    dash.add_generation(71, None, X71, 1, None)
    e = dash.history.entry(71)
    assert e is not None and e.obs is None and e.rep_index == 1
    assert dash.rep_genome(71).tobytes() == X71[1].tobytes()
    # 이름표의 유전자 구조가 바뀌면 기록을 비운다
    dash.set_watch_meta(meta_for("lander"))
    assert len(dash.history) == 71
    dash.set_watch_meta(meta_for("hill_car"))
    assert len(dash.history) == 0


def test_brain_source_show_time_and_visibility(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    dash.set_watch_meta(meta_for("lander"))
    assert not dash.brain_visible()  # 떠 있지 않음
    dash.resize(1000, 700)
    dash.show()
    qtbot.waitExposed(dash)
    assert not dash.brain_visible()  # 「학습 그래프」 쪽
    dash.set_page("fun")
    assert dash.brain_visible()
    dash.set_tile_visible("brain", False)
    assert not dash.brain_visible()
    dash.set_tile_visible("brain", True)
    calls: list[float] = []
    tile = dash.tile("brain")
    tile.show_time = calls.append  # type: ignore[attr-defined]
    dash.show_time(1.25)
    assert calls == [1.25] and dash.play_time == 1.25
    src = BrainSource("", "champion", 3, np.zeros(meta_for("lander").dim))
    dash.flush()
    dash.set_brain_source(src)
    assert dash.brain_source is src and tile.dirty
    qtbot.waitUntil(lambda: not tile.dirty)


def test_tile_classes_of_both_pages() -> None:
    from evolab.app.lab.brain_view import BrainTile

    assert D.tile_class("brain") is BrainTile
    assert D.tile_class("genemap") is W.GeneMapTile and D.tile_class("trend") is D._TrendTile


# ---------------------------------------------------------------------------
# 유전자 지도 (T32)
def test_gene_map_points_ghosts_trail_and_clicks(qtbot: QtBot, tmp_path: Path) -> None:
    dash = fun_dashboard(qtbot, tmp_path)
    feed(dash, "lander", range(1, 11), pop=6, learner=lambda g: 0 if g <= 6 else 1, champion_every=4)
    tile = dash.tile("genemap")
    assert isinstance(tile, W.GeneMapTile) and not tile.dirty
    assert tile.stack.currentWidget() is tile.view
    assert len(tile.points.data) == 6 and tile.shown_generation == 10
    assert [len(g.data) for g in tile.ghosts] == [6, 6, 6]  # 9, 8, 7세대
    assert len(tile.rep_ring.data) == 1 and len(tile.champion.data) == 1 and len(tile.start.data) == 1
    tx, ty = tile.trail.getData()
    assert len(tx) == 11 and np.isnan(tx).sum() == 1  # 10세대 + 학습 방법이 바뀐 곳에서 끊음
    assert S.GMAP_LEGEND.format(gen=10, n=6) in tile.legend.text()
    pca = dash.history.pca
    assert pca is not None and pca.fits >= 4  # 1, 2, 4, 8세대에 맞춤
    # 보는 세대를 바꾸면 그 세대 점
    dash.set_focus(5)
    assert tile.shown_generation == 5 and [len(g.data) for g in tile.ghosts] == [6, 6, 6]
    # 잔상(4세대) 점을 한 번 누르면 그 세대로, 두 번 누르면 그 개체
    ghost = tile._drawn[1]
    gen, pts = ghost[0], ghost[1]
    assert gen == 4
    with qtbot.waitSignal(dash.focus_requested, timeout=1000) as sig:
        assert tile.click_at(float(pts[2, 0]), float(pts[2, 1]), double=False)
    assert sig.args == [4]
    now_pts = tile._drawn[0][1]
    with qtbot.waitSignal(dash.individual_requested, timeout=1000) as sig2:
        assert tile.click_at(float(now_pts[3, 0]), float(now_pts[3, 1]), double=True)
    gen2, idx2 = sig2.args
    assert gen2 == 5 and 0 <= idx2 < 6
    assert np.allclose(tile._drawn[0][1][idx2], now_pts[3])
    tile.hover(float(now_pts[idx2, 0]), float(now_pts[idx2, 1]), 0)
    assert "5세대" in tile.legend.text()
    tile.unhover()
    # 다시 맞추기 단추
    fits = pca.fits
    tile.refit_button.click()
    assert dash.history.pca.fits == fits + 1 and S.GMAP_REFITTED.format(n=10) in tile.note.text()
    # 색 고르기
    tile.set_color_mode("success")
    assert tile.color_mode == "success"


def test_gene_map_of_a_v10_run_shows_representatives_only(qtbot: QtBot, tmp_path: Path) -> None:
    run = tmp_path / "run"
    shutil.copytree(DATA / "v10_run_ga" / "run", run)
    store = RunStore.open(run)
    spec = compile_env((run / "envs" / "v000.evoenv").read_text(encoding="utf-8"))
    dash = make_dashboard(qtbot, tmp_path)
    rows = store.read_stats()
    dash.set_rows(rows)
    dash.set_watch_meta(WatchMeta.from_spec(spec))
    dash.set_page("fun")
    dash.load_history(run, rows)
    assert dash.wait_history()
    dash.flush()
    tile = dash.tile("genemap")
    assert len(tile.points.data) == 1 and tile.shown_generation == 4
    assert S.DASH_OLD_RUN_NOTE in tile.note.text()
    tx, _ = tile.trail.getData()
    assert len(tx) == 4
    assert dash.tile("density").message.text() == S.DENS_NONE
    assert dash.tile("actions").message.text() == S.ACT_NONE
    # 대표 점을 두 번 누르면 (세대, -1): 대표 그 자체
    p = tile._drawn[0][1][0]
    with qtbot.waitSignal(dash.individual_requested, timeout=1000) as sig:
        tile.click_at(float(p[0]), float(p[1]), double=True)
    assert sig.args == [4, -1]
    assert dash.rep_genome(4).tobytes() == store.load_champion(4, "representative")["genome"].tobytes()


def test_gene_map_trail_stops_at_the_viewed_generation(qtbot: QtBot, tmp_path: Path) -> None:
    """「집단 가운데가 지나온 길」은 보는 세대까지만 그린다. 그 뒤 길은 보이지 않는 선으로 화면 범위에만 들어가 세대를
    넘겨 봐도 지도 범위가 그대로다."""
    dash = fun_dashboard(qtbot, tmp_path)
    feed(dash, "lander", range(1, 9))
    tile = dash.tile("genemap")
    assert isinstance(tile, W.GeneMapTile)
    tx, _ = tile.trail.getData()
    rx, _ = tile.trail_rest.getData()
    assert len(tx) == 8 and len(rx) == 0
    dash.set_focus(2)
    assert tile.shown_generation == 2
    tx, ty = tile.trail.getData()
    rx, ry = tile.trail_rest.getData()
    assert len(tx) == 2 and len(rx) == 7  # 1·2세대 + (2세대에서 이어) 3~8세대
    assert (rx[0], ry[0]) == (tx[-1], ty[-1])
    assert tile.trail_rest.opts["pen"].style() == Qt.PenStyle.NoPen
    rest_bounds = tile.trail_rest.dataBounds(0)
    assert rest_bounds[0] is not None and np.isfinite(rest_bounds).all()


def test_gene_map_note_for_v12_generations_without_records(qtbot: QtBot, tmp_path: Path) -> None:
    """관찰을 끈 학습(--no-observe)·요약 실패처럼 v1.2 세대에 기록이 없으면 'v1.2보다 먼저'라고 하지 않는다."""
    lay = meta_for("lander").layout
    root = tmp_path / "run"
    rows = write_run(root, [fake_generation(g, lay) for g in range(1, 4)], with_obs=False)
    dash = fun_dashboard(qtbot, tmp_path)
    dash.set_rows(rows)  # v1.1 열 머리(learner 열 있음)
    dash.load_history(root, rows)
    assert dash.wait_history()
    dash.flush()
    tile = dash.tile("genemap")
    assert tile.shown_generation == 3
    assert S.DASH_NO_OBS_NOTE in tile.note.text() and S.DASH_OLD_RUN_NOTE not in tile.note.text()


def _open_with_page(qtbot: QtBot, tmp_path: Path, name: str, root: Path, rows: list[dict[str, str]],
                    page: str) -> tuple[Any, str]:
    """화면에 띄운 대시보드에서 page 쪽을 보며 기록을 다 읽은 뒤 「보는 재미」로 와서 유전자 지도를 그린다."""
    dash = make_dashboard(qtbot, tmp_path, f"{name}.ini")
    dash.resize(1200, 700)
    dash.show()
    qtbot.waitExposed(dash)
    dash.set_page(page)
    dash.set_watch_meta(meta_for("lander"))
    seen: list[int] = []
    tile = dash.tile("genemap")
    real = tile.update_data

    def spy() -> None:
        if dash.history_loading is not None:
            seen.append(len(dash.history))
        real()

    tile.update_data = spy  # type: ignore[method-assign]
    dash.load_history(root, rows)
    assert dash.wait_history(60000)
    dash.set_page("fun")
    qtbot.wait(100)
    dash.flush()
    if page == "fun":
        assert seen  # 읽는 동안 지도가 갱신됐다(임시 첫 축)
    return dash.history.pca, tile.note.text()


def test_gene_map_axes_of_an_opened_run_do_not_depend_on_the_page_during_loading(qtbot: QtBot,
                                                                                  tmp_path: Path) -> None:
    """다시 연 실행의 지도 축은 읽는 동안 어느 쪽을 봤는지와 상관없다(설계 2.5·11.2): 읽는 동안에는 첫 축만 임시로
    맞추고, 다 읽은 뒤 읽은 모든 세대로 한 번 맞춘다(알림 없음). 두 배 규칙은 거기서부터 이어진다."""
    lay = meta_for("lander").layout
    root = tmp_path / "run"
    rows = write_run(root, [fake_generation(g, lay) for g in range(1, 201)])
    fun, fun_note = _open_with_page(qtbot, tmp_path, "fun", root, rows, "fun")
    learn, learn_note = _open_with_page(qtbot, tmp_path, "learn", root, rows, "learning")
    assert fun.fitted_generations == learn.fitted_generations == 200
    assert np.array_equal(fun.axes, learn.axes) and np.array_equal(fun.explained, learn.explained)
    refit_head = S.GMAP_REFITTED.split("{", 1)[0]
    assert refit_head not in fun_note and refit_head not in learn_note
    assert not fun.wants_fit()


def test_stale_big_pca_job_does_not_forget_the_running_one(qtbot: QtBot, tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """큰 신경망(차원 > 256)의 배경 PCA: 앞 기록(되감기·다른 실행)의 작업이 늦게 끝나도 지금 도는 작업을 잊지 않는다
    (같은 축을 맞추는 작업이 겹쳐 뜨거나 '다시 맞췄습니다' 알림이 잘못 나오지 않게)."""
    import threading

    from evolab.app.lab import history as hs

    text = (paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8").replace(
        '"hidden": [16]', '"hidden": [64, 64]', 1)
    meta = WatchMeta.from_spec(compile_env(text))
    assert meta.layout.dim > hs.PCA_SMALL_DIM
    gates = [threading.Event(), threading.Event()]
    calls: list[int] = []
    real = hs.randomized_axes

    def gated(sample: Any) -> Any:
        k = len(calls)
        calls.append(k)
        if k < 2:
            gates[k].wait(60)
        return real(sample)

    monkeypatch.setattr(hs, "randomized_axes", gated)
    dash = make_dashboard(qtbot, tmp_path)
    dash.set_watch_meta(meta)
    dash.set_page("fun")
    tile = dash.tile("genemap")
    assert isinstance(tile, W.GeneMapTile)

    def feed_big(gens: range) -> None:
        for g in gens:
            obs, X = fake_generation(g, meta.layout)
            dash.add_row(stats_row(obs))
            dash.add_generation(g, obs.to_arrays(), X, obs.representative, None)

    try:
        feed_big(range(1, 4))
        qtbot.waitUntil(lambda: len(calls) == 1, timeout=10000)
        j1 = tile._job
        dash.load_history(None, [])  # 되감기: 앞 작업이 도는 중에 기록을 비운다
        feed_big(range(1, 4))
        qtbot.waitUntil(lambda: len(calls) == 2, timeout=10000)
        j2 = tile._job
        assert j1 is not None and j2 is not None and j2 is not j1
        gates[0].set()  # 앞 작업이 끝난다
        qtbot.waitUntil(j1.isFinished, timeout=10000)
        qtbot.wait(50)
        assert tile._job is j2
        feed_big(range(4, 5))
        qtbot.wait(100)
        assert len(calls) == 2  # 겹쳐 뜬 작업이 없다
    finally:
        gates[0].set()
        gates[1].set()
    # J2(1세대 기준)가 끝나면 두 배 규칙대로 한 번만 더 맞춘다(4세대 ≥ 2 × 1): 앞 작업의 결과는 쓰지 않았다
    qtbot.waitUntil(lambda: len(calls) == 3 and tile._job is None, timeout=20000)
    qtbot.wait(100)
    pca = dash.history.ensure_pca()
    assert len(calls) == 3 and pca.fits == 2 and pca.fitted_generations == 4


# ---------------------------------------------------------------------------
# 몸 수치 (T33)
def test_body_tile_explains_how_to_enable_and_draws_hill_car_params(qtbot: QtBot, tmp_path: Path) -> None:
    dash = fun_dashboard(qtbot, tmp_path, "lander", genemap=False)
    feed(dash, "lander", range(1, 4))
    body = dash.tile("body")
    assert body.stack.currentWidget() is body.message and body.message.text() == S.BODY_EMPTY_NONE
    assert dash.is_tile_visible("body") and "body" not in dash._hidden and body.isVisibleTo(dash)
    assert dash._hidden == {"genemap"}  # 몸 수치 타일은 숨지 않는다(사용자 숨김 그대로)

    dash2 = fun_dashboard(qtbot, tmp_path, "hill_car", "hill.ini", genemap=False)
    meta = meta_for("hill_car")
    data = feed(dash2, "hill_car", range(1, 9), pop=5)
    body2 = dash2.tile("body")
    assert isinstance(body2, W.BodyTile) and body2.stack.currentWidget() is body2.view
    assert len(body2.lines) == meta.n_body == 5 and body2.selected == -1
    assert body2.pick.count() == 6 and body2.pick.itemText(0) == S.BODY_ALL
    for j, line in enumerate(body2.lines):
        x, y = line.getData()
        assert len(x) == 8
        p = meta.body[j]
        b = meta.layout.body_values(data[8][1])[:, j]
        assert y[-1] == pytest.approx((b.mean() - p.low) / (p.high - p.low) * 100.0)
    assert body2._fixed_y[id(body2.plot)] == (0.0, 100.0)
    body2.pick.setCurrentIndex(2)  # 수치 하나(앞 높이)
    p = meta.body[1]
    assert body2.selected == 1 and body2._fixed_y[id(body2.plot)] == (p.low, p.high)
    assert [line.value() for line in body2.limits] == [p.low, p.high]
    rx, ry = body2.rep_line.getData()
    obs8, X8 = data[8]
    assert ry[-1] == pytest.approx(meta.layout.body_values(X8)[obs8.representative, 1])
    # 가로축 전환을 따른다
    dash2.set_axis("evaluations")
    x, _ = body2.lines[1].getData()
    assert x[-1] == 8 * 5 * 2
    # 보는 세대 세로선(따라가는 중이면 숨김)
    assert not body2.focus_line.isVisible()
    dash2.set_focus(4)
    assert body2.focus_line.value() == 4 * 5 * 2


# ---------------------------------------------------------------------------
# 행동 분포 (T34)
def test_actions_tile_image_channels_modes_and_summary(qtbot: QtBot, tmp_path: Path) -> None:
    dash = fun_dashboard(qtbot, tmp_path, "lander", genemap=False)
    data = feed(dash, "lander", range(1, 13))
    tile = dash.tile("actions")
    assert isinstance(tile, W.ActionsTile)
    assert tile.image_array().shape == (12, ob.ACTION_BINS)
    assert np.allclose(np.asarray(tile.image_array()).sum(axis=1), 1.0)
    meta = meta_for("lander")
    assert [tile.channel_box.itemText(i) for i in range(tile.channel_box.count())] == [c.label for c in meta.channels]
    obs12 = data[12][0]
    hist = obs12.act_hist[0, 0].astype(np.float64)
    on = 1.0 - hist[0] / hist.sum()  # 추진기 0..1: 0번 칸이 쉬는 칸
    want = S.ACT_SUMMARY.format(name=meta.channels[0].label, on=on * 100, mean=float(obs12.act_mean[:, 0].mean()), gen=12)
    assert tile.summary == want and want in tile.note.text()
    before = tile.image_array().copy()
    tile.set_channel(2)
    assert tile.channel == 2 and not np.array_equal(before, tile.image_array())
    tile.set_mode("rep")
    h = obs12.act_hist[1, 2].astype(np.float64)
    assert np.allclose(tile.image_array()[-1], h / h.sum())
    assert tile.summary.startswith(meta.channels[2].label)
    tile.hover(12.0, 0.01, 0)
    assert "12세대" in tile.legend.text()
    dash.set_focus(3)
    assert tile.focus_line.isVisible() and tile.focus_line.value() == 3
    assert "3세대" in tile.summary


# ---------------------------------------------------------------------------
# 궤적 밀도 (T35)
def test_density_tile_asks_for_the_world_and_draws_the_layer(qtbot: QtBot, tmp_path: Path) -> None:
    dash = fun_dashboard(qtbot, tmp_path, "track_car", genemap=False)
    asked: list[tuple[int, str]] = []
    dash.geometry_needed.connect(lambda g, k: asked.append((g, k)))
    feed(dash, "track_car", range(1, 6), frame=FRAME, key=TRACK_KEY)
    tile = dash.tile("density")
    assert isinstance(tile, W.DensityTile)
    assert asked == [(1, TRACK_KEY)]  # 같은 열쇠는 한 번만
    assert tile.message.text() == S.DENS_PREPARING
    spec = compile_env((Path(__file__).parents[1] / "environments" / "track_car.evoenv").read_text(encoding="utf-8"))
    geometry = Episode(spec, None, seeds.display_trial()).geometry()
    dash.set_world_geometry(TRACK_KEY, geometry)
    assert dash.world_geometry(TRACK_KEY) is geometry
    assert tile.stack.currentWidget() is tile.host
    layer = tile.canvas.density_layer()
    assert layer is not None
    image, rect = layer
    assert (image.width(), image.height()) == (FRAME.nx, FRAME.ny)
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == pytest.approx(FRAME.rect)
    assert S.DENS_LEGEND.format(gen=5, n=6, cell=FRAME.cell) in tile.legend.text()
    tile.set_window("recent")
    assert tile.density_view.merged == (1, 2, 3, 4, 5)
    assert S.DENS_LEGEND_MERGED.format(first=1, last=5, count=5, cell=FRAME.cell) in tile.legend.text()
    # 정적 그림에 밀도 층이 들어간다: 칸 하나 자리의 화면 색이 밀도 색
    tile.canvas.resize(400, 300)
    dv = tile.density_view
    iy, ix = np.argwhere(dv.counts == dv.counts.max())[0]
    cx, cy = FRAME.cell_center(int(iy) * FRAME.nx + int(ix))
    shot = tile.canvas.grab().toImage()
    pt = tile.canvas.world_to_screen(cx, cy)
    with_layer = shot.pixelColor(int(pt.x()), int(pt.y()))
    tile.canvas.set_density(None)
    without = tile.canvas.grab().toImage().pixelColor(int(pt.x()), int(pt.y()))
    assert with_layer != without
    # 마우스: 칸의 횟수
    tile.refresh()
    tile.hover_world(cx, cy)
    assert f"{int(dv.counts.max())}" in tile.legend.text()
    # 대시보드는 세계 그림을 최근 GEOMETRY_CACHE개만 기억한다: 밀려난 세계를 다시 보면 다시 청한다(DENS_PREPARING에
    # 멈추지 않는다)
    for k in range(D.GEOMETRY_CACHE):
        dash.set_world_geometry(f"other{k:02d}", geometry)
    assert dash.world_geometry(TRACK_KEY) is None
    tile.refresh()
    assert asked[-1] == (5, TRACK_KEY) and len(asked) == 2
    assert tile.message.text() == S.DENS_PREPARING
    dash.set_world_geometry(TRACK_KEY, geometry)
    assert tile.stack.currentWidget() is tile.host


def test_density_of_changing_worlds_cannot_merge(qtbot: QtBot, tmp_path: Path) -> None:
    dash = fun_dashboard(qtbot, tmp_path, "lander", genemap=False)
    asked: list[tuple[int, str]] = []
    spec = compile_env((Path(__file__).parents[1] / "environments" / "lander.evoenv").read_text(encoding="utf-8"))
    geometry = Episode(spec, None, seeds.display_trial()).geometry()

    def answer(gen: int, key: str) -> None:  # 실험실처럼 바로 답한다(관전 중 같은 세대)
        asked.append((gen, key))
        dash.set_world_geometry(key, geometry)

    dash.geometry_needed.connect(answer)
    tile = dash.tile("density")
    tile.set_window("recent")
    feed(dash, "lander", range(1, 5), frame=FRAME, key=lambda g: f"lander-world-{g:04d}")
    assert [k for _, k in asked] == [f"lander-world-{g:04d}" for g in range(1, 5)]
    assert tile.stack.currentWidget() is tile.host and tile.density_view.merged == (4,)
    assert S.DENS_CANNOT_MERGE in tile.note.text()


# ---------------------------------------------------------------------------
# 테마 토큰 (T36)
def _colors_of(item: Any) -> list[QColor]:
    out: list[QColor] = []
    opts = getattr(item, "opts", {}) or {}
    for key in ("pen", "brush", "symbolPen", "symbolBrush"):
        v = opts.get(key)
        for obj in v if isinstance(v, (list, tuple, np.ndarray)) else [v]:
            if obj is None or isinstance(obj, (str, int, float)):
                continue
            if hasattr(obj, "style") and obj.style() in (Qt.PenStyle.NoPen, Qt.BrushStyle.NoBrush):
                continue
            if hasattr(obj, "color"):
                out.append(obj.color())
    if isinstance(item, pg.InfiniteLine):
        out.append(item.pen.color())
    if isinstance(item, pg.ScatterPlotItem):
        for p in item.points():
            for obj in (p.pen(), p.brush()):
                if obj.style() not in (Qt.PenStyle.NoPen, Qt.BrushStyle.NoBrush):
                    out.append(obj.color())
    return out


def _on_ramp(c: QColor, ramp: tuple[str, ...]) -> bool:
    """순차 색(ramp) 토큰 사이를 잇는 선 위의 색인가(점수 색표는 ramp 토큰을 이어 칠한다)."""
    x = np.array([c.red(), c.green(), c.blue()], dtype=float)
    pts = [np.array([QColor(r).red(), QColor(r).green(), QColor(r).blue()], dtype=float) for r in ramp]
    for a, b in zip(pts, pts[1:], strict=False):
        d = b - a
        f = float(np.clip((x - a) @ d / max(float(d @ d), 1e-9), 0.0, 1.0))
        if np.abs(a + f * d - x).max() <= 2.0:
            return True
    return False


def test_new_tiles_use_only_theme_tokens_in_the_dark_theme(qtbot: QtBot, tmp_path: Path) -> None:
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    old = app.palette()
    dark = QPalette(old)
    dark.setColor(QPalette.ColorRole.Window, QColor("#202020"))
    app.setPalette(dark)
    try:
        dash = fun_dashboard(qtbot, tmp_path, "hill_car", "dark.ini")
    finally:
        app.setPalette(old)
    assert dash.theme is D.DARK
    feed(dash, "hill_car", range(1, 6), frame=FRAME, key=TRACK_KEY, champion_every=2)
    t = D.DARK
    tokens = {QColor(c).name() for c in (t.surface, t.ink, t.ink2, t.muted, t.grid, t.axis, *t.series, *t.ramp)}
    for tid in ("genemap", "body", "actions"):
        tile = dash.tile(tid)
        for plot in tile.plots:
            for item in plot.items:
                for c in _colors_of(item):
                    scored = tid == "genemap" and item is dash.tile("genemap").points
                    assert c.name() in tokens or (scored and _on_ramp(c, t.ramp)), (tid, type(item).__name__, c.name())
    for cmap in (dash.tile("genemap").cmap, dash.tile("actions").cmap):
        for c in cmap.getColors(mode="qcolor"):
            assert c.name() in tokens
    # 궤적 밀도 색표는 늘 어두운 관전 바탕 위라 DARK 토큰으로
    lut = W.density_lut()
    for p, i in W.DENSITY_STOPS:
        v = int(lut[max(1, round(p * 255))])
        want = QColor(t.series[i] if i is not None else t.ink)
        got = ((v >> 16) & 255, (v >> 8) & 255, v & 255)
        assert max(abs(a - b) for a, b in zip(got, (want.red(), want.green(), want.blue()), strict=True)) <= 2
    assert int(lut[0]) == 0


# ---------------------------------------------------------------------------
# 관전 캔버스 추가분
def test_canvas_wheel_needs_ctrl_and_density_layer(qtbot: QtBot) -> None:
    canvas = SpectatorCanvas()
    qtbot.addWidget(canvas)
    spec = compile_env((Path(__file__).parents[1] / "environments" / "lander.evoenv").read_text(encoding="utf-8"))
    canvas.set_scene(Episode(spec, None, seeds.display_trial()).geometry())
    canvas.resize(400, 300)
    canvas.set_wheel_needs_ctrl(True)

    def wheel(mods: Qt.KeyboardModifier) -> QWheelEvent:
        pos = QPointF(200, 150)
        return QWheelEvent(pos, QPointF(canvas.mapToGlobal(pos.toPoint())), QPoint(0, 0), QPoint(0, 120),
                           Qt.MouseButton.NoButton, mods, Qt.ScrollPhase.NoScrollPhase, False)

    zoom = canvas._zoom
    plain = wheel(Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(canvas, plain)
    assert not plain.isAccepted() and canvas._zoom == zoom
    QApplication.sendEvent(canvas, wheel(Qt.KeyboardModifier.ControlModifier))
    assert canvas._zoom > zoom
    img = W.density_image(np.arange(12, dtype=np.float64).reshape(3, 4))
    assert img.pixelColor(0, 0).alpha() == 0 and img.pixelColor(3, 2).alpha() == 255
    canvas.set_density(img, (0.0, 0.0, 4.0, 3.0), fit=True)
    assert canvas._fit_target() == (0.0, 0.0, 4.0, 3.0)
    canvas.set_scene(Episode(spec, None, seeds.display_trial()).geometry())
    assert canvas.density_layer() is None  # 다른 세계의 층은 버린다


def test_render_fun_page_screenshot(qtbot: QtBot, tmp_path: Path) -> None:
    import os

    dash = fun_dashboard(qtbot, tmp_path, "hill_car", "shot.ini")
    dash.resize(1400, 1000)
    dash.show()
    qtbot.waitExposed(dash)
    feed(dash, "hill_car", range(1, 31), pop=8, frame=FRAME, key=TRACK_KEY, champion_every=10)
    spec = compile_env((Path(__file__).parents[1] / "environments" / "hill_car.evoenv").read_text(encoding="utf-8"))
    dash.set_world_geometry(TRACK_KEY, Episode(spec, None, seeds.display_trial()).geometry())
    dash.flush()
    qtbot.wait(30)
    image = dash.grab().toImage()
    assert image.width() > 0 and image.height() > 0
    out = os.environ.get("EVOLAB_DASHBOARD_V12_PNG")
    assert image.save(str(Path(out) if out else tmp_path / "dashboard_v12.png"))


# ---------------------------------------------------------------------------
# 열기·관전 지연 고침 (T39·M4b): 열마다 읽는 set_rows, 기록을 읽는 동안의 펌프 간격, 한 번에 그리는 펌프 단계
def _row_by_row(dash: D.Dashboard, rows: list[dict[str, Any]]) -> None:
    """고치기 전 set_rows(줄마다 _append)."""
    dash.table.clear()
    dash.hist.clear()
    dash._clear_learner()
    for row in rows:
        dash._append(row, bulk=True)
    dash.hist.finish()


def _table_state(dash: D.Dashboard) -> dict[str, Any]:
    t, h = dash.table, dash.hist
    return {
        "names": t.names(), "n": t.n, "cap": t._cap, "version": t.columns_version,
        "cols": {k: t.col(k).copy() for k in t.names()},
        "hist": (h.n, repr(h.lo), repr(h.hi), h.rebuilds, h.matrix().copy()),
        "learner": repr((dash.auto_marks, dash._auto_rows, dash._auto_tips, dash.learner_max, dash.qualify_rows)),
    }


def _assert_same_state(a: dict[str, Any], b: dict[str, Any]) -> None:
    assert a["names"] == b["names"] and (a["n"], a["cap"], a["version"]) == (b["n"], b["cap"], b["version"])
    for k in a["names"]:
        assert np.array_equal(a["cols"][k], b["cols"][k], equal_nan=True), k
    assert a["hist"][:4] == b["hist"][:4] and np.array_equal(a["hist"][4], b["hist"][4])
    assert a["learner"] == b["learner"]


def test_set_rows_by_columns_matches_row_by_row_append(qtbot: QtBot, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """set_rows는 열 이름이 모두 같은 줄(stats.csv)을 열마다 한 번에 읽는다: 줄마다 _append한 것과 같은 표·점수 분포·
    학습 방법 교체 세로선이어야 한다(빈 칸, 공백, 읽을 수 없는 글자, 세대 빈칸·세대 열 없음, 숫자 줄 포함)."""
    from test_dashboard_v11 import stability_rows, switch_rows

    messy = synthetic_rows(60, strings=True)
    messy[2]["generation"] = ""
    messy[9]["score_best"] = "  "
    messy[10]["evaluations"] = "abc"
    messy[11]["scores"] = ""
    messy[12]["scores"] = "1.5 x 3"
    messy[19]["sigma"] = " 0.5 "
    messy[30]["diversity"] = None
    messy[31]["score_mean"] = "inf"
    no_gen = [{k: v for k, v in r.items() if k != "generation"} for r in synthetic_rows(20, strings=True)]
    ragged = synthetic_rows(20, strings=True)
    ragged[5]["extra_column"] = "1.0"
    cases = {
        "v11 strings": stability_rows(200, strings=True), "v11 numbers": stability_rows(200),
        "switch rows": switch_rows(), "messy strings": messy, "no generation column": no_gen,
        "ragged keys": ragged, "one row": synthetic_rows(1, strings=True), "empty": [],
    }
    dash = make_dashboard(qtbot, tmp_path)
    calls: list[int] = []
    real_append = dash._append
    monkeypatch.setattr(dash, "_append", lambda row, *, bulk: (calls.append(1), real_append(row, bulk=bulk))[1])
    for name, rows in cases.items():
        _row_by_row(dash, rows)
        expected = _table_state(dash)
        calls.clear()
        dash.set_rows(iter(rows))  # 한 번만 도는 반복자도 받는다
        assert len(calls) == (len(rows) if name == "ragged keys" else 0), name  # 열마다 읽는 길을 탔다
        _assert_same_state(_table_state(dash), expected)
    # 열마다 읽은 뒤 학습 중 줄을 더해도 줄마다 넣은 것과 같다(용량·열 순서가 같다)
    rows = stability_rows(300, strings=True)
    dash.set_rows(rows[:256])
    for r in rows[256:]:
        dash.add_row(r)
    got = _table_state(dash)
    _row_by_row(dash, rows)
    _assert_same_state(got, _table_state(dash))


def _fun_tiles_dirty(dash: D.Dashboard) -> None:
    for tid in D.FUN_TILE_IDS:
        dash.tile(tid).dirty = True


def test_pump_refreshes_fun_tiles_at_most_every_interval_while_history_loads(
        qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """기록을 배경으로 읽는 동안 펌프는 「보는 재미」 타일 하나를 LOADING_REFRESH_S에 한 번만 갱신한다(남은 dirty는 그때
    펌프 타이머로). 다 읽으면 바로 갱신하고, flush와 「학습 그래프」 쪽은 간격을 따르지 않는다."""
    clock = [100.0]
    monkeypatch.setattr(D, "_clock", lambda: clock[0])
    dash = make_dashboard(qtbot, tmp_path, "coalesce.ini")
    dash.set_watch_meta(meta_for("lander"))
    dash.resize(1200, 900)
    dash.show()
    qtbot.waitExposed(dash)
    dash.set_page("fun")
    dash._pump_timer.stop()
    in_view = [t for t in dash.tile_order() if dash._in_view(dash.tile(t))]
    assert "genemap" in in_view and len(in_view) >= 2
    dash.history_loading = (64, 750)  # 읽는 중(배경 읽기 없이 상태만)
    _fun_tiles_dirty(dash)
    refreshed: list[str] = []
    for tid in D.FUN_TILE_IDS:
        tile = dash.tile(tid)
        real = tile.refresh
        monkeypatch.setattr(tile, "refresh", lambda t=tid, r=real: (refreshed.append(t), r())[1])
    for _ in in_view:  # 처음에는 하나씩 바로
        dash._pump_step()
    assert refreshed == in_view
    _fun_tiles_dirty(dash)
    clock[0] += 0.1
    dash._pump_timer.stop()  # 앞 단계가 걸어 둔 다음 차례 펌프(이 시험은 펌프를 손으로 돌린다)
    dash._pump_step()  # 모두 간격 안: 아무것도 갱신하지 않고 남은 시간 뒤로 펌프를 미룬다
    assert refreshed == in_view and all(dash.tile(t).dirty for t in in_view)
    assert dash._pump_timer.isActive() and 150 <= dash._pump_timer.remainingTime() <= D.LOADING_REFRESH_S * 1000
    clock[0] += D.LOADING_REFRESH_S
    dash._pump_step()
    assert refreshed[-1] == in_view[0]
    # flush는 간격과 상관없이 지금 쪽 dirty 타일을 모두
    dash.flush()
    assert not any(dash.tile(t).dirty for t in D.FUN_TILE_IDS if dash.is_tile_visible(t))
    # 다 읽으면(loaded) 간격 없이 바로
    tag = dash._load_tag
    dash._on_history_loaded(tag, {"observed": 0})
    assert dash.history_loading is None and dash._load_refreshed == {}
    refreshed.clear()
    dash._pump_timer.stop()
    for _ in in_view:
        dash._pump_step()
    assert refreshed == in_view


def test_coalesced_loading_ends_with_the_final_data_drawn(qtbot: QtBot, tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """화면에 띄운 「보는 재미」 쪽에서 실제 배경 읽기(작은 묶음 15개): 읽는 동안 지도 갱신은 묶음 수보다 적고
    LOADING_REFRESH_S보다 촘촘하지 않으며, flush 없이도 마지막 세대 자료로 끝난다."""
    import time

    from evolab.app.lab import history as hs

    lay = meta_for("lander").layout
    root = tmp_path / "run"
    rows = write_run(root, [fake_generation(g, lay, pop=3) for g in range(1, 121)])

    class SmallChunks(hs.HistoryLoader):
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, chunk_size=8, **k)

    real_load = hs.load_generation

    def slow_load(*a: Any, **k: Any) -> Any:
        time.sleep(0.004)
        return real_load(*a, **k)

    monkeypatch.setattr(hs, "HistoryLoader", SmallChunks)
    monkeypatch.setattr(hs, "load_generation", slow_load)
    dash = make_dashboard(qtbot, tmp_path, "final.ini")
    dash.set_watch_meta(meta_for("lander"))
    dash.resize(1200, 900)
    dash.show()
    qtbot.waitExposed(dash)
    dash.set_page("fun")
    tile = dash.tile("genemap")
    seen: list[tuple[float, bool, int]] = []
    real = tile.refresh
    monkeypatch.setattr(tile, "refresh", lambda: (seen.append((time.monotonic(), dash.history_loading is not None,
                                                               len(dash.history))), real())[1])
    chunks: list[int] = []
    dash.load_history(root, rows)
    assert dash._loader is not None
    dash._loader.chunk.connect(lambda *a: chunks.append(1))
    assert dash.wait_history(60000)
    qtbot.waitUntil(lambda: not any(dash.tile(t).dirty for t in dash.tile_order() if dash._in_view(dash.tile(t))))
    during = [t for t, loading, n in seen if loading and n > 0]
    assert len(chunks) == 15 and len(during) < len(chunks)
    assert all(b - a >= D.LOADING_REFRESH_S - 0.02 for a, b in zip(during, during[1:], strict=False))
    assert not seen[-1][1] and seen[-1][2] == 120
    assert tile.shown_generation == 120


def test_pump_step_paints_a_graph_tile_once(qtbot: QtBot, tmp_path: Path) -> None:
    """펌프 한 단계는 그래프를 한 번만 그린다: pyqtgraph가 다음 차례로 미뤄 두는 자동 범위·장면 갱신을 repaint 앞에서
    처리한다(prepare_paint). 미루면 같은 그래프를 다음 차례에 한 번 더 그렸다."""
    from PySide6.QtCore import QEvent, QObject

    dash = make_dashboard(qtbot, tmp_path)
    dash.resize(1200, 900)
    dash.show()
    qtbot.waitExposed(dash)
    rows = synthetic_rows(41)
    dash.set_rows(rows[:40])
    qtbot.waitUntil(lambda: not any(dash.tile(t).dirty for t in dash.tile_order() if dash._in_view(dash.tile(t))))
    qtbot.wait(100)

    class Count(QObject):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0

        def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
            if event.type() == QEvent.Type.Paint:
                self.n += 1
            return False

    counter = Count()
    viewport = dash.tile("diversity").view.viewport()
    viewport.installEventFilter(counter)
    dash.add_row(rows[40])
    qtbot.waitUntil(lambda: not dash.tile("diversity").dirty)
    qtbot.wait(150)
    viewport.removeEventFilter(counter)
    assert counter.n == 1

"""6.2.0 다듬기 묶음 PL: 대시보드 축(U18, C10)·빈 타일 글(U18)·관전 캔버스 알림 글(U6의 set_notice, 다듬기 3.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QImage
from pytestqt.qtbot import QtBot

from evolab import paths
from evolab.app import strings as S
from evolab.app.lab import dashboard as D
from evolab.app.lab.canvas import SpectatorCanvas
from evolab.app.lab.dashboard import TILE_IDS, Dashboard, _Axis
from evolab.engine import seeds
from evolab.spec.io import load_env
from evolab.world.physics2d.episode import Episode
from tests.test_dashboard import BLOCKS, END_EVENTS, synthetic_rows


def make_dashboard(qtbot: QtBot, tmp_path: Path) -> Dashboard:
    dash = Dashboard(settings=QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(dash)
    dash.set_meta(end_events=END_EVENTS, blocks=BLOCKS, stagnation_generations=40)
    dash.resize(1200, 900)
    dash.show()
    qtbot.waitExposed(dash)
    return dash


def tick_texts(axis: Any) -> list[str]:
    """그 축이 지금 그리는 눈금 글(모든 단계)."""
    lo, hi = axis.range
    size = axis.geometry().width() if axis.orientation in ("top", "bottom") else axis.geometry().height()
    out: list[str] = []
    for spacing, values in axis.tickValues(lo, hi, size):
        out.extend(axis.tickStrings(values, axis.scale, spacing))
    return out


def refresh_all(dash: Dashboard) -> None:
    for tid in TILE_IDS:
        dash.tile(tid).refresh()
        dash.tile(tid).prepare_paint()


# ---------------------------------------------------------------------------
# _Axis 단위


def test_integer_axis_never_spaces_ticks_below_one(qtbot: QtBot) -> None:
    ax = _Axis("bottom", integer=True)
    for lo, hi, size in ((0.6, 1.4, 400.0), (0.5, 2.5, 600.0), (1.0, 5.0, 800.0), (0.9, 1.1, 300.0)):
        levels = ax.tickSpacing(lo, hi, size)
        assert len(levels) == 1, levels  # 작은 눈금 없음
        assert levels[0][0] >= 1.0 and float(levels[0][0]).is_integer()
        texts = [t for spacing, vals in ax.tickValues(lo, hi, size) for t in ax.tickStrings(vals, 1.0, spacing)]
        assert texts and all("." not in t for t in texts), texts
    assert ax.tickValues(0.6, 1.4, 400.0) == [(1.0, [1.0])]
    # 넓은 범위는 pyqtgraph의 간격(1·2·5·10배) 그대로
    plain = _Axis("bottom")
    assert ax.tickSpacing(0.0, 1000.0, 400.0)[0] == plain.tickSpacing(0.0, 1000.0, 400.0)[0]


def test_tick_strings_group_thousands(qtbot: QtBot) -> None:
    ax = _Axis("left")
    assert ax.tickStrings([40000.0, 45000.0], 1.0, 5000.0) == ["40,000", "45,000"]
    assert ax.tickStrings([-1500.0, 0.0, 1500.0], 1.0, 1500.0) == ["-1,500", "0", "1,500"]
    assert ax.tickStrings([999.0, 1000.0], 1.0, 1.0) == ["999", "1,000"]
    assert ax.tickStrings([1200.5, 1201.0], 1.0, 0.5) == ["1,200.5", "1,201.0"]
    assert ax.tickStrings([0.5, 1.0, 1.5], 1.0, 0.5) == ["0.5", "1.0", "1.5"]  # 1,000 아래는 원래 꼴
    assert ax.tickStrings([2e-17, -0.0, -1e-12], 1.0, 0.2) == ["0", "0", "0"]  # 반올림 찌꺼기·−0
    assert ax.tickStrings([1e7], 1.0, 1e6) == ["10,000,000"]  # %g의 「1e+07」이 아님
    integer = _Axis("bottom", integer=True)
    assert integer.tickStrings([1.0, 2.0, 12000.0], 1.0, 1.0) == ["1", "2", "12,000"]
    integer.set_integer(False)
    assert integer.tickStrings([1.5], 1.0, 0.5) == ["1.5"]


# ---------------------------------------------------------------------------
# 대시보드 (C10)


def test_generation_axis_ticks_are_whole_numbers_for_first_generations(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows = synthetic_rows(5)
    for n, row in enumerate(rows, start=1):
        dash.add_row(row)
        refresh_all(dash)
        qtbot.wait(1)
        for tid in TILE_IDS:
            tile = dash.tile(tid)
            for plot in tile.plots:
                ax = plot.getAxis("bottom")
                assert isinstance(ax, _Axis) and ax.integer, (tid, n)
                if not ax.style["showValues"]:
                    continue
                texts = tick_texts(ax)
                assert texts, (tid, n)
                assert all("." not in t for t in texts), (tid, n, texts)
                assert all(1 <= int(t) <= n + 1 for t in texts), (tid, n, texts)
            assert isinstance(tile.plots[0].getAxis("left"), _Axis)


def test_axis_switch_releases_integer_and_groups_thousands(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    rows = synthetic_rows(30)
    for r in rows:  # 점수를 천 단위로: 세로축 눈금에도 구분 기호
        for k in ("score_best", "score_mean", "score_median", "score_min", "score_q25", "score_q75"):
            r[k] = float(r[k]) * 100.0
        if r["champion_score"] != "":
            r["champion_score"] = float(r["champion_score"]) * 100.0
    dash.set_rows(rows)
    refresh_all(dash)
    trend = dash.tile("trend")
    left = [t for t in tick_texts(trend.plot.getAxis("left")) if abs(float(t.replace(",", ""))) >= 1000]
    assert left and all("," in t for t in left), left

    dash.set_axis("evaluations")
    refresh_all(dash)
    bottom = trend.plot.getAxis("bottom")
    assert not bottom.integer
    texts = tick_texts(bottom)
    big = [t for t in texts if abs(float(t.replace(",", ""))) >= 1000]
    assert big and all("," in t for t in big), texts
    assert dash.tile("heatmap").plots[0].getAxis("bottom").integer  # 세대 고정 타일은 그대로
    assert dash.tile("blocks").plots[0].getAxis("bottom").integer
    dash.set_axis("elapsed")
    assert not trend.plot.getAxis("bottom").integer
    dash.set_axis("generation")
    assert all(p.getAxis("bottom").integer for tid in TILE_IDS for p in dash.tile(tid).plots)


def test_fun_tiles_use_the_axis_and_genemap_is_not_integer(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    assert not dash.tile("genemap").plots[0].getAxis("bottom").integer
    for tid in ("actions", "structure", "body", "brainsize"):
        ax = dash.tile(tid).plots[0].getAxis("bottom")
        assert isinstance(ax, _Axis) and ax.integer, tid
    # 「뇌 크기」의 오른쪽 축(연결 수)도 천 단위 구분
    assert isinstance(dash.tile("brainsize").plot.getAxis("right"), _Axis)


# ---------------------------------------------------------------------------
# 빈 타일 (U18)


def test_empty_tiles_show_one_sentence_until_first_row(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    refresh_all(dash)
    assert set(S.DASH_TILE_EMPTY) == set(TILE_IDS)
    for tid in TILE_IDS:
        tile = dash.tile(tid)
        assert tile.stack.currentWidget() is tile.message, tid
        assert tile.message.text() == S.DASH_TILE_EMPTY[tid]
    dash.add_row(synthetic_rows(1)[0])
    refresh_all(dash)
    for tid in TILE_IDS:
        tile = dash.tile(tid)
        assert tile.stack.currentWidget() is tile.view, tid
    dash.clear()
    refresh_all(dash)
    for tid in TILE_IDS:
        assert dash.tile(tid).stack.currentWidget() is dash.tile(tid).message, tid
    # 「보는 재미」 타일은 저마다의 글(빈 타일 글 없음)
    assert all(dash.tile(tid).empty_text == "" for tid in ("genemap", "body", "structure"))


def test_blocks_tile_keeps_its_no_blocks_sentence(qtbot: QtBot, tmp_path: Path) -> None:
    dash = make_dashboard(qtbot, tmp_path)
    blocks = dash.tile("blocks")
    blocks.refresh()
    assert blocks.message.text() == S.DASH_TILE_EMPTY["blocks"]
    dash.set_rows(synthetic_rows(3, contrib=False))
    blocks.refresh()
    assert blocks.stack.currentWidget() is blocks.message
    assert blocks.message.text() == S.DASH_BLOCKS_NONE
    dash.clear()
    blocks.refresh()
    assert blocks.message.text() == S.DASH_TILE_EMPTY["blocks"]
    dash.set_rows(synthetic_rows(3))
    blocks.refresh()
    assert blocks.stack.currentWidget() is blocks.view


# ---------------------------------------------------------------------------
# 관전 캔버스 알림 글 (U6의 set_notice)


def grab(canvas: SpectatorCanvas) -> QImage:
    canvas.repaint()
    return canvas.grab().toImage()


def test_canvas_notice_replaces_empty_text_and_clears(qtbot: QtBot) -> None:
    canvas = SpectatorCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(640, 400)
    canvas.show()
    qtbot.waitExposed(canvas)
    assert canvas.notice == ""
    assert canvas.notice_rect().isEmpty()
    plain = grab(canvas)
    text = S.LAB_NOTICE_FIRST_GEN.format(pop=32)
    canvas.set_notice(text)
    assert canvas.notice == text
    box = canvas.notice_rect()
    assert not box.isEmpty()
    assert abs(box.center().x() - 320) < 1.5 and abs(box.center().y() - 200) < 1.5
    assert box.width() < 600
    shown = grab(canvas)
    assert shown != plain
    # 상자 밖은 그대로 바탕(empty_text는 그리지 않는다): 상자 바깥 줄 하나를 바탕과 비교
    y = int(box.top()) - 4
    assert all(shown.pixel(x, y) == plain.pixel(x, y) for x in range(0, 640, 7))
    canvas.set_notice("")
    assert canvas.notice == "" and grab(canvas) == plain


def test_canvas_notice_wraps_on_narrow_canvas(qtbot: QtBot) -> None:
    canvas = SpectatorCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(320, 240)
    canvas.show()
    qtbot.waitExposed(canvas)
    canvas.set_notice(S.LAB_NOTICE_FIRST_GEN.format(pop=1000))
    box = canvas.notice_rect()
    assert box.left() >= 0 and box.right() <= 320
    assert box.height() > canvas._font_metrics.height() * 1.5  # 좁으면 두 줄 이상으로 꺾임
    before = canvas._notice_box
    canvas.set_notice(canvas.notice)  # 같은 글: 아무것도 하지 않음(자리 기억도 그대로)
    assert canvas._notice_box is before
    canvas.resize(500, 300)
    assert abs(canvas.notice_rect().center().x() - 250) < 1.5


def test_canvas_notice_draws_over_world(qtbot: QtBot) -> None:
    spec = load_env(paths.builtin_env_dir() / "lander.evoenv")
    ep = Episode(spec, None, seeds.train_trials(1, 0, 1)[0])
    canvas = SpectatorCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(480, 320)
    canvas.show()
    qtbot.waitExposed(canvas)
    canvas.set_scene(ep.geometry(), channels=ep.actuator_channels())
    for _ in range(3):  # 정적 세계 그림 기억(같은 배율 두 프레임 뒤)이 자리 잡게
        plain = grab(canvas)
    canvas.set_notice(S.LAB_NOTICE_NEXT_GEN)
    shown = _bits(grab(canvas)).reshape(320, 480, 4)
    box = canvas.notice_rect().toAlignedRect()
    before = _bits(plain).reshape(320, 480, 4)
    inside = (shown[box.top() : box.bottom(), box.left() : box.right()] != before[box.top() : box.bottom(), box.left() : box.right()])
    assert inside.any()
    outside = shown.copy()
    outside[box.top() - 1 : box.bottom() + 2, box.left() - 1 : box.right() + 2] = before[box.top() - 1 : box.bottom() + 2, box.left() - 1 : box.right() + 2]
    assert np.array_equal(outside, before)  # 상자 밖은 그대로
    canvas.set_notice("")
    assert np.array_equal(_bits(grab(canvas)), _bits(plain))


def _bits(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB32)
    return np.frombuffer(img.constBits(), dtype=np.uint8, count=img.sizeInBytes()).copy()


# ---------------------------------------------------------------------------
# 점수 추이의 위계 (U19)와 Tab 사슬 (U20): 하면 좋음


def test_trend_champion_is_the_heaviest_line_and_legend_order(qtbot: QtBot, tmp_path: Path) -> None:
    from PySide6.QtCore import Qt

    dash = make_dashboard(qtbot, tmp_path)
    dash.set_rows(synthetic_rows(20))
    trend = dash.tile("trend")
    trend.refresh()
    t = dash.theme
    widths = {k: c.opts["pen"].widthF() for k, c in trend.lines.items()}
    champ = trend.champion.opts["pen"].widthF()
    assert champ == 2.5 and all(w < champ for w in widths.values())
    assert widths == {"score_min": 1.0, "score_mean": 2.0, "score_median": 1.5, "score_best": 2.0}
    low = trend.lines["score_min"].opts["pen"]
    assert low.style() == Qt.PenStyle.DashLine and low.color().alpha() == 120
    # 색 값은 그대로(P5)
    for key, _name, slot, _w in trend.LINES:
        assert trend.lines[key].opts["pen"].color().name() == QColor(t.series[slot]).name()
    names = [e.name for e in trend.entries()]
    assert names[:3] == [S.DASH_TREND_CHAMPION, S.DASH_TREND_BEST, S.DASH_TREND_MEAN]
    assert [e.dim for e in trend.entries()][:3] == [False, False, False]
    assert all(e.dim for e in trend.entries()[3:])
    html = trend.legend.text()
    assert html.count(f"font-size:{D.SWATCH_PX}px") == len(names)  # 견본은 모두 채운 네모
    assert D.GLYPH_LINE not in html and D.GLYPH_DASH not in html
    assert f'color:{t.tick}">{S.DASH_TREND_MEDIAN}' in html


def test_tile_buttons_graphs_and_scroll_area_are_out_of_the_tab_chain(qtbot: QtBot, tmp_path: Path) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QToolButton

    dash = make_dashboard(qtbot, tmp_path)
    assert dash.scroll.focusPolicy() == Qt.FocusPolicy.NoFocus
    for tid in D.ALL_TILE_IDS:
        tile = dash.tile(tid)
        assert tile.view.focusPolicy() == Qt.FocusPolicy.NoFocus, tid
        assert tile.view.viewport().focusPolicy() == Qt.FocusPolicy.NoFocus, tid
        heads = [b for b in tile.findChildren(QToolButton) if b.text() in (D.GLYPH_EARLIER, D.GLYPH_LATER, D.GLYPH_HIDE)]
        assert len(heads) == 3 and all(b.focusPolicy() == Qt.FocusPolicy.NoFocus for b in heads), tid
    # 키보드로는 「타일」 메뉴 단추(그대로 Tab 사슬 안)
    assert dash.tiles_button.focusPolicy() != Qt.FocusPolicy.NoFocus

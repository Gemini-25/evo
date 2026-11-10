"""v6.1 자라는 뇌 화면 (v6 설계 4.17~4.20, 5.3~5.5, D36~D38, 8.2 T45~T53; T54는 v3 도움말·풍선 테스트가 지킨다).

- T45 실험실 「뇌」 줄: CMA-ES에서만, 「자라기」면 두 칸·안내 두 줄·IPOP 숨김, 은닉층·뇌 없는 환경이면 항목이 꺼지고 풍선, 고른 값이
  TrainConfig(method "grow", grow_*)로, 학습 중 잠금, 설정에 기억.
- T46 가짜 세대 메시지 "growth" → 상태 끝줄 LAB_STATUS_GROWN 한 줄.
- T47·T48 뇌 그림: 꺼진 자리 점선 원(연결선 없음), 제목 크기 글·아래 줄, 최근 5세대 안에 켜진 뉴런 고리.
- T49 크기 고정·옛 실행: 그림이 지금과 바이트 같고(active=None과 모두 켜짐이 같은 그림), 제목에 크기 글 없음, 점선 원 없음.
- T50 「뇌 크기」 타일: 계단 = 통계 열, 세로선 = grow 열, 누르면 보는 세대, 빈 상태 셋.
- T51 점수 추이 「grow」 세로선·풍선·범례, 같은 세대 재시작 세로선과 함께, 한꺼번에 읽은 것 = 한 줄씩 더한 것.
- T52 기록 목록·비교·경주·반복 실행의 「CMA-ES · 자라는 뇌」, 반복 실행이 grow_*를 복사.
- T53 갱신 비용(넉넉한 상한, 실제 값은 측정 M12로 보고).
- 학습 프로세스와 함께: 진짜 짧은 자라는 실행(학습 먼저 L 묶음)에서 상태 글·타일·뇌 그림.

학습 프로세스 없는 시험은 통계 줄·세대 메시지를 v6 설계 7.3 계약 그대로 가짜로 만든다.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor
from pytestqt.qtbot import QtBot
from test_dashboard import make_dashboard

from evolab import paths
from evolab.app import strings as S
from evolab.app.lab import brain_view as B
from evolab.app.lab import dashboard as D
from evolab.app.lab import history as hs
from evolab.app.lab.grow_tile import GrowTile, frame_totals
from evolab.app.lab.view import BRAIN_FIXED, BRAIN_GROW, GROW_METHOD, LabView
from evolab.app.lab.watch_types import BrainSource, WatchMeta
from evolab.engine.trainer import TrainConfig
from evolab.policy.frame import BrainFrame
from evolab.policy.genome import layout_for
from evolab.runs.store import RunStore
from evolab.spec.io import compile_env

DATA = Path(__file__).parent / "data"
LANDER = paths.builtin_env_dir() / "lander.evoenv"
GLIDER = paths.builtin_env_dir() / "glider.evoenv"


def env_text(name: str) -> str:
    return (paths.builtin_env_dir() / f"{name}.evoenv").read_text(encoding="utf-8")


def with_brain(name: str, **brain: Any) -> str:
    """내장 환경의 뇌 설정만 바꾼 환경 글(예: 은닉층 없음 hidden=[], 두 층 hidden=[8, 4])."""
    data = json.loads(env_text(name))
    data["brain"] = {**data["brain"], **brain}
    return json.dumps(data, ensure_ascii=False)


def lander() -> tuple[Any, Any, WatchMeta, BrainFrame]:
    spec = compile_env(env_text("lander"))
    lay = layout_for(spec)
    return spec, lay, WatchMeta.from_spec(spec, lay), BrainFrame(lay)


def grown_genome(lay: Any, frame: BrainFrame, counts: list[int], seed: int = 1) -> np.ndarray:
    """켜진 좌표(counts)만 0이 아닌 유전자(자라는 실행의 후보와 같은 모양, 꺼진 좌표는 정확히 0.0)."""
    g = np.zeros(lay.dim)
    coords = frame.active_coords(counts)
    g[coords] = np.random.default_rng(seed).normal(0.0, 0.5, len(coords))
    return g


def grow_rows(frame: BrainFrame, gens: int, *, every: int = 10, start: int = 2,
              extra: dict[int, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """자라는 실행의 통계 줄(설계 D35): brain_neurons·brain_links = 그 세대 후보를 만든 크기, grow = 그 세대 끝에 자랐나.
    은닉층 하나 틀 기준(내장 예제), 학습 중 메시지처럼 숫자 값."""
    size = frame.sizes[0]
    n, out = min(start, size), []
    for g in range(1, gens + 1):
        grew = g % every == 0 and n < size
        row = {"generation": g, "evaluations": g * 64, "elapsed": g * 0.5, "champion_score": float(g),
               "score_best": float(g), "brain_neurons": n, "brain_links": frame.links([n]), "grow": int(grew)}
        row.update((extra or {}).get(g, {}))
        out.append(row)
        if grew:
            n = min(size, 2 * n)
    return out


def fun_dashboard(qtbot: QtBot, tmp_path: Path, rows: list[dict[str, Any]] | None, meta: WatchMeta) -> D.Dashboard:
    dash = make_dashboard(qtbot, tmp_path, "grow.ini")
    dash.set_watch_meta(meta)
    if rows is not None:
        dash.set_rows(rows)
    dash.set_page("fun")
    dash.flush()
    return dash


# ---------------------------------------------------------------------------
# 실험실 (T45, T46)
# ---------------------------------------------------------------------------
def make_lab(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: str = "lab.ini") -> LabView:
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path / "runs")
    view = LabView(settings=QSettings(str(tmp_path / settings), QSettings.Format.IniFormat))
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.show()
    qtbot.waitExposed(view)
    return view


@pytest.fixture()
def lab(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    view = make_lab(qtbot, tmp_path, monkeypatch)
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(view.client, "send", lambda msg: sent.append(dict(msg)) or False)
    assert view.load_env(LANDER)
    try:
        yield view
    finally:
        view.client._proc = None  # 잠금 시험이 넣은 가짜 학습 프로세스
        view.shutdown()


def shown(view: LabView, widget: Any) -> bool:
    label = view._method_form.labelForField(widget)
    return widget.isVisibleTo(view) and (label is None or label.isVisibleTo(view))


def test_brain_row_only_for_cmaes_and_grow_shows_fields_hints_and_hides_ipop(lab: LabView) -> None:
    assert lab.method.currentData() == "cmaes" and lab.brain_combo.currentData() == BRAIN_FIXED  # 새 실행 기본(D39)
    assert [lab.brain_combo.itemText(i) for i in range(2)] == [S.LAB_BRAIN_FIXED, S.LAB_BRAIN_GROW]
    assert lab._method_form.labelForField(lab.brain_combo).text() == S.LAB_BRAIN
    assert shown(lab, lab.brain_combo) and shown(lab, lab.ipop)
    for w in (lab.grow_start, lab.grow_every, lab.grow_hint, lab.grow_honest):
        assert not w.isVisibleTo(lab)
    # 패널 순서(설계 5.1): 학습 방법 → 뇌 → 두 칸 → 안내 두 줄 → 실행 시드
    form = lab._method_form
    rows = [form.getWidgetPosition(w)[0] for w in (lab.method, lab.brain_combo, lab.grow_start, lab.grow_every,
                                                   lab.grow_hint, lab.grow_honest, lab.seed)]
    assert rows == sorted(rows) and len(set(rows)) == len(rows)
    lab.brain_combo.setCurrentIndex(lab.brain_combo.findData(BRAIN_GROW))
    for w in (lab.grow_start, lab.grow_every, lab.grow_hint, lab.grow_honest):
        assert shown(lab, w), w
    assert not lab.ipop.isVisibleTo(lab)
    assert (lab.grow_start.value(), lab.grow_every.value()) == (2, 10)
    assert lab.grow_start.maximum() == 16 and lab.grow_every.maximum() == 100  # 가장 작은 은닉층, 100세대
    assert lab.grow_hint.text() == S.LAB_GROW_HINT.format(path="2 → 4 → 8 → 16", gen=30)
    assert lab.grow_honest.text() == S.LAB_GROW_HINT_HONEST
    lab.grow_every.setValue(7)
    lab.grow_start.setValue(5)
    assert lab.grow_hint.text() == S.LAB_GROW_HINT.format(path="5 → 10 → 16", gen=14)
    lab.grow_start.setValue(16)
    assert lab.grow_hint.text() == S.LAB_GROW_HINT_ALREADY
    # 다른 학습 방법이면 「뇌」 줄이 통째로 숨고, CMA-ES로 돌아오면 「자라기」 그대로
    for method in ("ga", "sa"):
        lab.method.setCurrentIndex(lab.method.findData(method))
        for w in (lab.brain_combo, lab.grow_start, lab.grow_every, lab.grow_hint, lab.grow_honest, lab.ipop):
            assert not w.isVisibleTo(lab), (method, w)
        assert lab.config_from_form().method == method
    lab.method.setCurrentIndex(lab.method.findData("cmaes"))
    assert shown(lab, lab.grow_start) and not lab.ipop.isVisibleTo(lab)


def test_grow_item_is_off_with_a_reason_without_hidden_layers_or_brain(lab: LabView) -> None:
    lab.brain_combo.setCurrentIndex(lab.brain_combo.findData(BRAIN_GROW))
    item = lab.brain_combo.model().item(1)
    assert item.isEnabled() and lab.brain_combo.toolTip() == S.LAB_BRAIN_TIP
    for text, reason in ((with_brain("lander", hidden=[]), S.LAB_BRAIN_GROW_NO_HIDDEN),
                         (env_text("glider"), S.LAB_BRAIN_GROW_NO_BRAIN)):
        assert lab.use_env_text(text, "test", None)
        assert lab.grow_blocked() == reason
        assert not item.isEnabled() and item.toolTip() == reason
        assert lab.brain_combo.toolTip().startswith(S.LAB_BRAIN_TIP) and lab.brain_combo.toolTip().endswith(reason)
        assert lab.brain_combo.currentData() == BRAIN_FIXED and not lab.brain_grows()  # 고를 수 없는 값은 놓는다
        assert not lab.grow_start.isVisibleTo(lab) and lab.grow_hint.text() == ""
        assert lab.config_from_form().method == "cmaes"
    # 은닉층 두 개(가장 작은 층 4): 처음 뉴런 수 최댓값 4, 안내 줄은 은닉층 합
    assert lab.use_env_text(with_brain("lander", hidden=[8, 4]), "test", None)
    assert item.isEnabled() and lab.grow_blocked() is None and lab.brain_combo.toolTip() == S.LAB_BRAIN_TIP
    lab.brain_combo.setCurrentIndex(lab.brain_combo.findData(BRAIN_GROW))
    assert lab.grow_start.maximum() == 4 and lab.grow_start.value() == 2
    assert lab.grow_hint.text() == S.LAB_GROW_HINT.format(path="4 → 8 → 12", gen=20)  # (2,2)→(4,4)→(8,4)


def test_form_makes_a_grow_config_and_opened_runs_fill_the_row(lab: LabView) -> None:
    lab.ipop.setChecked(True)
    lab.brain_combo.setCurrentIndex(lab.brain_combo.findData(BRAIN_GROW))
    lab.grow_start.setValue(4)
    lab.grow_every.setValue(5)
    c = lab.config_from_form()
    assert c.method == GROW_METHOD and c.ipop is False  # 숨은 IPOP 체크는 넘기지 않는다(함께 쓸 수 없음, D33)
    assert (c.grow_start, c.grow_every) == (4, 5)
    c.validate()
    d = c.to_dict()
    assert d["method"] == "grow" and (d["grow_start"], d["grow_every"]) == (4, 5)
    lab.brain_combo.setCurrentIndex(lab.brain_combo.findData(BRAIN_FIXED))
    fixed = lab.config_from_form()
    assert fixed.method == "cmaes" and fixed.ipop is True and "grow_start" not in fixed.to_dict()
    # 실행을 열면(설정으로 칸을 채움) 자라는 실행은 CMA-ES + 「자라기」, 크기 고정 실행은 「크기 고정」
    lab._form_from_config({**fixed.to_dict(), "method": "grow", "ipop": False, "grow_start": 3, "grow_every": 12})
    assert lab.method.currentData() == "cmaes" and lab.brain_combo.currentData() == BRAIN_GROW
    assert (lab.grow_start.value(), lab.grow_every.value()) == (3, 12) and not lab.ipop.isVisibleTo(lab)
    again = lab.config_from_form()
    assert again.method == "grow" and (again.grow_start, again.grow_every) == (3, 12)
    lab._form_from_config(fixed.to_dict())
    assert lab.brain_combo.currentData() == BRAIN_FIXED and lab.config_from_form().method == "cmaes"
    lab._form_from_config({**fixed.to_dict(), "method": "ga"})
    assert lab.method.currentData() == "ga" and not lab.brain_combo.isVisibleTo(lab)


def test_brain_row_locks_while_training_or_resuming(lab: LabView) -> None:
    lab.brain_combo.setCurrentIndex(lab.brain_combo.findData(BRAIN_GROW))
    fields = (lab.method, lab.brain_combo, lab.grow_start, lab.grow_every)
    assert all(w.isEnabled() for w in fields)
    lab.client._proc = object()  # 학습 프로세스가 도는 중(새 실행이든 이어 하는 실행이든 같은 규칙, 명세 A7)
    lab._update_buttons()
    assert not any(w.isEnabled() for w in fields)
    lab.client._proc = None
    lab._update_buttons()
    assert all(w.isEnabled() for w in fields)


def test_brain_choice_is_remembered_but_opened_runs_do_not_overwrite_it(qtbot: QtBot, tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    view = make_lab(qtbot, tmp_path, monkeypatch, "remember.ini")
    try:
        assert view.load_env(LANDER)
        view.brain_combo.setCurrentIndex(view.brain_combo.findData(BRAIN_GROW))
        view.brain_combo.activated.emit(1)  # 사용자가 고름
        view.grow_start.setValue(4)
        view.grow_every.setValue(20)
        view._form_from_config(TrainConfig().to_dict())  # 크기 고정 실행을 열었다
        assert view.brain_combo.currentData() == BRAIN_FIXED
        s = view.settings
        assert (s.value("lab/brain"), int(s.value("lab/grow_start")), int(s.value("lab/grow_every"))) == ("grow", 4, 20)
    finally:
        view.shutdown()
    again = make_lab(qtbot, tmp_path, monkeypatch, "remember.ini")
    try:
        assert again.brain_combo.currentData() == BRAIN_GROW
        assert (again.grow_start.value(), again.grow_every.value()) == (4, 20)
    finally:
        again.shutdown()
    fresh = make_lab(qtbot, tmp_path, monkeypatch, "new_user.ini")  # 처음 쓰는 사람: 「크기 고정」(D39)
    try:
        assert fresh.brain_combo.currentData() == BRAIN_FIXED and fresh.settings.value("lab/brain") is None
    finally:
        fresh.shutdown()


def generation_message(view: LabView, gen: int, growth: dict[str, int] | None = None) -> dict[str, Any]:
    """학습 프로세스의 generation 알림(설계 7.3 계약): 관전 궤적 없이(재생하지 않음), 자란 세대만 "growth"."""
    assert view.layout is not None
    X = np.repeat(view.layout.initial()[None, :], 4, axis=0)
    msg: dict[str, Any] = {
        "row": {"generation": str(gen), "evaluations": str(gen * 8), "champion_score": "1.0"}, "generation": gen,
        "population": X, "representative": 0, "scores": np.zeros(4), "trial": (0, None), "trajectories": None,
        "champion": None, "env_version": None, "observation": None, "waited": 0.0}
    if growth is not None:
        msg["growth"] = growth
    return msg


def test_growth_message_puts_one_status_line(lab: LabView) -> None:
    lab._on_generation(generation_message(lab, 1))
    assert "뇌가 자랐습니다" not in lab.status.text()
    lab._on_generation(generation_message(lab, 2, {"old": 4, "new": 8, "old_links": 71, "new_links": 139}))
    line = S.LAB_STATUS_GROWN.format(gen=2, old=4, new=8, old_links=71, new_links=139)
    assert lab.status.text().count(line) == 1 and lab.status.text().endswith(line)
    lab._on_generation(generation_message(lab, 3))  # 다음 세대가 와도 한 줄(쌓이지 않음)
    assert lab.status.text().count(line) == 1
    lab._on_generation(generation_message(lab, 4, {"old": 8, "new": 16, "old_links": 139, "new_links": 1234}))
    newer = S.LAB_STATUS_GROWN.format(gen=4, old=8, new=16, old_links=139, new_links=1234)
    assert newer in lab.status.text() and line not in lab.status.text() and "1,234" in newer
    lab._on_generation(generation_message(lab, 5, {"old": "x"}))  # type: ignore[dict-item]  # 이상한 값은 무시
    assert newer in lab.status.text()
    lab._reset_run_view()  # 다른 실행
    assert lab._grow_note == ""


# ---------------------------------------------------------------------------
# 뇌 그림 (T47, T48, T49)
# ---------------------------------------------------------------------------
def node_pixel(view: B.BrainView, layer: int, j: int) -> QColor:
    img = view.grab().toImage()
    x, y = view._pos[layer][j]
    dpr = img.devicePixelRatio()
    return img.pixelColor(round(float(x) * dpr), round(float(y) * dpr))


def test_brain_view_draws_empty_slots_without_edges_and_the_size_title(qtbot: QtBot, tmp_path: Path) -> None:
    _spec, lay, meta, frame = lander()
    g = grown_genome(lay, frame, [4])
    dash = fun_dashboard(qtbot, tmp_path, grow_rows(frame, 25), meta)
    dash.set_brain_source(BrainSource("", "live", 25, g))
    dash.flush()
    tile = dash.tile("brain")
    v = tile.brain
    assert v.active is not None and [m.tolist() for m in v.active] == [[True] * 4 + [False] * 12]
    assert v.empty_slots == 12 and v.node_state(1, 3) == "on" and v.node_state(1, 4) == "empty"
    # 꺼진 자리에는 연결선이 없다(가중치가 정확히 0이라 select_edges가 뺀다)
    into, _c, _l = v._selected[0][0], v._selected[0][1], v._selected[0][2]
    cols = v._selected[0][1]
    rows_out = v._selected[1][0]
    assert len(into) and set(cols.tolist()) <= {0, 1, 2, 3} and set(rows_out.tolist()) <= {0, 1, 2, 3}
    assert v.edges_drawn == 4 * 9 + 4 * 3
    label = S.BRAIN_SIZE_LABEL.format(n=4, total=16, links=frame.links([4]))
    assert tile.size_label == label and tile.title_text.endswith(label) and tile.title_text.startswith(
        S.BRAIN_TITLE_LIVE.format(gen=25, t=0.0))
    assert S.BRAIN_GROW_NOTE in tile.note.text()
    assert v.node_tip(1, 9).endswith(S.BRAIN_NODE_EMPTY) and S.BRAIN_NODE_EMPTY not in v.node_tip(1, 2)
    # 그림: 꺼진 자리 가운데는 바탕(채움 없음), 켜진 자리 가운데는 회색 채움(추적 없음)
    v.resize(480, 300)
    v.wait_edges()
    surface = QColor(tile.theme.surface)
    assert node_pixel(v, 1, 10) == surface and node_pixel(v, 1, 1) != surface
    # 다 자라면 크기 글·아래 줄이 사라진다(점선 원 없음), 제목은 v6.0 제목 그대로
    dash.set_brain_source(BrainSource("", "live", 26, grown_genome(lay, frame, [16])))
    dash.flush()
    assert v.empty_slots == 0 and tile.size_label == "" and tile.title_text == S.BRAIN_TITLE_LIVE.format(gen=26, t=0.0)
    assert S.BRAIN_GROW_NOTE not in tile.note.text()


def test_fresh_rings_mark_neurons_that_appeared_in_the_last_five_generations(qtbot: QtBot, tmp_path: Path) -> None:
    _spec, lay, meta, frame = lander()
    dash = fun_dashboard(qtbot, tmp_path, grow_rows(frame, 30), meta)
    for gen, counts in ((15, [4]), (20, [4]), (21, [8])):  # 세대 대표(학습 중 메시지처럼)
        X = grown_genome(lay, frame, counts, seed=gen)[None, :]
        dash.add_generation(gen, None, X, 0, None)
    tile = dash.tile("brain")
    v = tile.brain
    dash.set_brain_source(BrainSource("", "live", 25, grown_genome(lay, frame, [8], seed=3)))
    dash.flush()
    assert v.fresh is not None and v.fresh_count == 4  # 20세대(25 − 5)에는 4개, 지금 8개
    assert [v.node_state(1, j) for j in (0, 3, 4, 7, 8)] == ["on", "on", "fresh", "fresh", "empty"]
    assert v.node_tip(1, 5).endswith(S.BRAIN_NODE_NEW.format(gen=B.FRESH_GENERATIONS))
    dash.set_brain_source(BrainSource("", "live", 20, grown_genome(lay, frame, [4], seed=4)))
    dash.flush()
    assert v.fresh is not None and v.fresh_count == 0  # 15세대와 같은 크기
    dash.set_brain_source(BrainSource("", "live", 24, grown_genome(lay, frame, [8], seed=5)))
    dash.flush()
    assert v.fresh is None and v.fresh_count == 0  # 19세대 대표가 없으면 고리 없음
    # 고리를 그리면 그 자리 테두리가 범주 색 3
    dash.set_brain_source(BrainSource("", "live", 25, grown_genome(lay, frame, [8], seed=3)))
    dash.flush()
    v.resize(480, 300)
    v.wait_edges()
    img = v.grab().toImage()
    x, y = v._pos[1][5]
    r = v._radius[1]
    ring = QColor(tile.theme.series[3])
    edge = [img.pixelColor(round(float(x) + dx), round(float(y))) for dx in np.arange(r - 1.0, r + 1.5, 0.5)]
    assert any(abs(c.red() - ring.red()) < 40 and abs(c.green() - ring.green()) < 40 and abs(c.blue() - ring.blue()) < 40
               for c in edge), [c.name() for c in edge]


def test_fixed_size_and_old_runs_draw_exactly_as_before(qtbot: QtBot, tmp_path: Path) -> None:
    _spec, lay, meta, frame = lander()
    g = grown_genome(lay, frame, [16], seed=8)
    params = lay.brain_params(g[None, :])[0]
    images = []
    for kw in ({}, {"active": None, "fresh": None}, {"active": [np.ones(16, bool)]}):
        v = B.BrainView(D.LIGHT)
        qtbot.addWidget(v)
        v.resize(480, 260)
        v.set_network(lay.brain, params, list(meta.inputs), list(meta.outputs), **kw)
        v.grab()
        v._resize_timer.stop()  # 크기를 바꾸는 동안 늦추는 그림(시간에 따라 다른 길)을 건너뛴다(test_brain_view와 같은 방법)
        assert v.wait_edges()
        img = v.grab().toImage()
        images.append(bytes(img.constBits()))
    assert images[0] == images[1] == images[2]  # active=None과 모두 켜짐(고리 없음)은 v6.0과 같은 그림
    # 크기 고정 실행(통계에 brain_neurons 열 없음): GA 첫 개체처럼 뇌가 모두 0이어도 점선 원이 아니다(유전자로 판단하지 않음)
    rows = [{"generation": 1, "champion_score": 0.0, "evaluations": 64}]
    dash = fun_dashboard(qtbot, tmp_path, rows, meta)
    assert not dash.brain_grows
    zero = lay.initial()
    assert all(not a.any() for a in frame.active_neurons(zero))
    dash.set_brain_source(BrainSource("", "live", 1, zero))
    dash.flush()
    tile = dash.tile("brain")
    assert tile.brain.active is None and tile.brain.fresh is None and tile.brain.empty_slots == 0
    assert tile.title_text == S.BRAIN_TITLE_LIVE.format(gen=1, t=0.0) and S.BRAIN_GROW_NOTE not in tile.note.text()
    # v1.0 시험 자료 실행(유전 알고리즘): 챔피언 멈춘 그림, 연결선 = select_edges 그대로, 크기 글 없음
    run = tmp_path / "v10"
    shutil.copytree(DATA / "v10_run_ga" / "run", run)
    store = RunStore.open(run)
    spec10 = compile_env((run / "envs" / "v000.evoenv").read_text(encoding="utf-8"))
    meta10 = WatchMeta.from_spec(spec10)
    dash10 = make_dashboard(qtbot, tmp_path, "v10.ini")
    stats = store.read_stats()
    dash10.set_rows(stats)
    dash10.set_watch_meta(meta10)
    dash10.set_page("fun")
    dash10.load_history(run, stats)
    assert dash10.wait_history()
    dash10.flush()
    t10 = dash10.tile("brain")
    assert t10.source_kind == "champion" and t10.brain.active is None and t10.size_label == ""
    champ = store.load_champion(4, "champion")["genome"]
    want = [B.select_edges(w) for w in [x[0] for x in __import__("evolab.policy.brain", fromlist=["unpack"]).unpack(
        meta10.layout.brain, meta10.layout.brain_params(np.asarray(champ, np.float64)[None, :])[0]).weights]]
    for (r1, c1, l1), (r2, c2, l2) in zip(t10.brain._selected, want, strict=True):
        assert r1.tobytes() == r2.tobytes() and c1.tobytes() == c2.tobytes() and l1.tobytes() == l2.tobytes()
    assert t10.title_text == S.BRAIN_TITLE_CHAMPION.format(gen=4)
    assert dash10.tile("brainsize").message.text() == S.GROW_TILE_FIXED.format(
        neurons=sum(meta10.layout.brain.hidden), links=frame_totals(meta10.layout.brain)[1])


def test_set_network_rejects_masks_that_do_not_match_the_layers(qtbot: QtBot) -> None:
    _spec, lay, meta, _frame = lander()
    v = B.BrainView(D.LIGHT)
    qtbot.addWidget(v)
    params = lay.brain_params(lay.initial()[None, :])[0]
    with pytest.raises(ValueError):
        v.set_network(lay.brain, params, list(meta.inputs), list(meta.outputs), active=[np.ones(15, bool)])
    with pytest.raises(ValueError):
        v.set_network(lay.brain, params, list(meta.inputs), list(meta.outputs), fresh=[np.ones(16, bool)] * 2)
    v.set_network(lay.brain, params, list(meta.inputs), list(meta.outputs), active=[np.zeros(16, bool)])
    builds = v.edge_builds
    v.grab()
    v.set_network(lay.brain, params, list(meta.inputs), list(meta.outputs), active=[np.ones(16, bool)])
    v.grab()
    assert v.edge_builds == builds + 1 and v.empty_slots == 0  # 켜짐만 바뀌면 연결선 층은 다시 그리지 않는다(첫 그림 1번)


# ---------------------------------------------------------------------------
# 「뇌 크기」 타일 (T50)
# ---------------------------------------------------------------------------
def test_history_reads_grow_columns_and_gives_nothing_for_fixed_runs() -> None:
    rows = [{"generation": "1", "brain_neurons": "2", "brain_links": "24", "grow": "0"},
            {"generation": "2", "brain_neurons": "2", "brain_links": "24", "grow": "1"},
            {"generation": "3", "brain_neurons": "4", "brain_links": "", "grow": "0"}]
    cols = hs.grow_columns(rows)
    assert list(cols) == list(hs.GROW_COLUMNS)
    s = hs.grow_series(cols)
    assert s is not None and s.rows.tolist() == [0, 1, 2] and s.neurons.tolist() == [2.0, 2.0, 4.0]
    assert s.links[:2].tolist() == [24.0, 24.0] and np.isnan(s.links[2]) and s.events.tolist() == [1]
    assert hs.grow_columns([{"generation": "1", "champion_score": "3"}]) == {}
    assert hs.grow_series({}) is None and hs.grow_series({"brain_neurons": np.full(3, np.nan)}) is None


def test_grow_tile_steps_marks_click_and_empty_states(qtbot: QtBot, tmp_path: Path) -> None:
    _spec, lay, meta, frame = lander()
    rows = grow_rows(frame, 40)
    dash = fun_dashboard(qtbot, tmp_path, rows, meta)
    assert dash.tile_order("fun")[:3] == ["genemap", "brain", "brainsize"]  # 「뇌 들여다보기」 바로 뒤(설계 4.19)
    tile = dash.tile("brainsize")
    assert isinstance(tile, GrowTile) and tile.title.text() == S.DASH_TILE_TITLES["brainsize"]
    assert tile.title.toolTip() == S.DASH_TILE_HELP["brainsize"]
    assert tile.stack.currentWidget() is tile.view and tile.total == 16 and tile.full_links == 192
    x, y = tile.neurons_curve.getData()
    assert np.asarray(x).tolist() == list(range(1, 41)) and np.asarray(y).tolist() == [r["brain_neurons"] for r in rows]
    _x, ly = tile.links_curve.getData()  # 오른쪽 눈금(0..192)에 맞춰 뉴런 좌표로 바꾼 연결 수
    assert np.allclose(np.asarray(ly) * 192 / 16, [r["brain_links"] for r in rows])
    assert [int(tile.gens[i]) for i in tile.series.events] == [10, 20, 30]
    # 세로선: 대시보드 자동 세로선(종류 grow)이 자란 세대에, 펜은 범주 색 3 점선
    applied = [a for a in tile._mark_applied if a is not None]
    assert [(a[0], a[3]) for a in applied] == [(10.0, "grow"), (20.0, "grow"), (30.0, "grow")]
    assert applied[0][1] == S.DASH_MARK_GROW.format(gen=10, old=2, new=4)
    entries = [e.name for e in tile.entries()]
    assert entries == [S.GROW_TILE_NEURONS, S.GROW_TILE_LINKS, S.GROW_TILE_EVENT]
    tile.hover(24.6, 0.0, 0)
    assert tile.legend.text() == S.GROW_TILE_TIP.format(x=S.DASH_HOVER_X["generation"].format(gen=25, x="25"),
                                                        neurons=8, total=16, links=96)
    tile.unhover()
    with qtbot.waitSignal(dash.focus_requested, timeout=1000) as sig:
        assert tile.click_at(12.4) == 12
    assert sig.args == [12]
    dash.set_focus(12)
    dash.flush()
    assert tile.focus_line.isVisible() and tile.focus_line.value() == 12.0
    # 가로축을 누적 시험 횟수로 바꾸면 계단·세로선도 그 값에
    dash.set_axis("evaluations")
    dash.flush()
    x, _y = tile.neurons_curve.getData()
    assert np.asarray(x).tolist() == [r["evaluations"] for r in rows]
    assert [a[0] for a in tile._mark_applied if a is not None] == [640.0, 1280.0, 1920.0]
    dash.set_axis("generation")
    # 크기 고정·옛 실행: 틀 합과 다 자란 연결 수
    dash.set_rows([{k: v for k, v in r.items() if k not in hs.GROW_COLUMNS} for r in rows])
    dash.flush()
    assert tile.series is None and tile.stack.currentWidget() is tile.message
    assert tile.message.text() == S.GROW_TILE_FIXED.format(neurons=16, links=192) and tile.entries() == []
    # 학습 전(통계 없음)
    dash.set_rows([])
    dash.flush()
    assert tile.message.text() == S.GROW_TILE_EMPTY
    # 뇌 없는 환경(글라이더): v2.0 빈 상태
    glider = WatchMeta.from_spec(compile_env(env_text("glider")))
    dash.set_watch_meta(glider)
    dash.set_rows([{"generation": 1, "champion_score": 1.0}])
    dash.flush()
    assert tile.message.text() == S.DASH_EMPTY_NO_BRAIN


# ---------------------------------------------------------------------------
# 점수 추이 (T51)
# ---------------------------------------------------------------------------
def test_score_trend_grow_marks_tips_legend_and_a_restart_in_the_same_generation(qtbot: QtBot,
                                                                                 tmp_path: Path) -> None:
    _spec, _lay, meta, frame = lander()
    extra = {g: {"learner": 0, "switch": 0, "learner_best": float(g), "learner_stall": 0} for g in range(1, 26)}
    extra[20]["switch"] = 1  # 20세대 끝: 자람 + 자동 재시작(처음 상태에서)
    extra[20]["learner_stall"] = 20000
    for g in range(21, 26):
        extra[g]["learner"] = 1
    rows = grow_rows(frame, 25, extra=extra)
    for r in rows[20:]:  # 재시작한 새 학습 방법은 처음 크기부터
        r["brain_neurons"], r["brain_links"] = 2, frame.links([2])
    dash = make_dashboard(qtbot, tmp_path, "trend.ini")
    dash.set_watch_meta(meta)
    dash.set_rows(rows)
    dash.flush()
    marks = [(g, k) for g, k, _info in dash.auto_marks]
    assert marks == [(10, "grow"), (20, "grow"), (20, "restart")] and dash.grow_marks == 2
    xs, tips, labels, kinds = dash.mark_positions(True)
    assert xs.tolist() == [10.0, 20.0, 20.0] and labels[:2] == [S.DASH_MARK_LABELS["grow"]] * 2
    assert tips[0] == S.DASH_MARK_GROW.format(gen=10, old=2, new=4)
    assert tips[1] == S.DASH_MARK_GROW.format(gen=20, old=4, new=2)  # 다음 줄 크기(재시작으로 처음 크기)
    assert D.mark_style("grow") == "grow" and kinds[2] == "restart"
    pen = D.mark_pen(dash.theme, "grow")
    assert pen.color().name() == QColor(dash.theme.series[3]).name()
    trend = dash.tile("trend")
    assert S.DASH_TREND_GROW in [e.name for e in trend.entries()]
    styles = [a[3] for a in trend._mark_applied if a is not None]
    assert styles == ["grow", "grow", "restart"]
    # 한 줄씩 더해도 같은 세로선·풍선. 마지막 세대에 자랐으면 다음 줄 전에는 어림(두 배, 틀에서 멈춤)
    inc = make_dashboard(qtbot, tmp_path, "trend2.ini")
    inc.set_watch_meta(meta)
    for r in rows[:10]:
        inc.add_row(r)
    assert inc._auto_tips[-1] == S.DASH_MARK_GROW.format(gen=10, old=2, new=4)
    for r in rows[10:]:
        inc.add_row({k: str(v) for k, v in r.items()})  # stats.csv처럼 글자 값
    assert [(g, k) for g, k, _ in inc.auto_marks] == marks and inc._auto_tips == dash._auto_tips
    # 크기 고정 실행: 세로선·범례 없음
    fixed = make_dashboard(qtbot, tmp_path, "trend3.ini")
    fixed.set_rows([{k: v for k, v in r.items() if k not in hs.GROW_COLUMNS} for r in rows])
    fixed.flush()
    assert [k for _g, k, _ in fixed.auto_marks] == ["restart"] and fixed.grow_marks == 0
    assert S.DASH_TREND_GROW not in [e.name for e in fixed.tile("trend").entries()]


def test_qualify_marks_still_find_the_winner_when_entrants_grow(qtbot: QtBot, tmp_path: Path) -> None:
    """예선 출전자마다 처음부터 자란다: 「뇌가 자람」 세로선이 예선 다음·예선 끝 사이에 끼어도 통과자 풍선이 그대로."""
    _spec, _lay, meta, frame = lander()
    rows = []
    for g in range(1, 7):
        learner = (g - 1) // 2
        rows.append({"generation": g, "champion_score": 1.0, "learner": learner, "qualifying": 1 if g <= 6 else 0,
                     "learner_best": [5.0, 5.0, 9.0, 9.0, 7.0, 7.0][g - 1], "learner_stall": 0,
                     "switch": {2: 3, 4: 3, 6: 4}.get(g, 0), "brain_neurons": 2, "brain_links": frame.links([2]),
                     "grow": 1 if g % 2 == 1 else 0})
    dash = make_dashboard(qtbot, tmp_path, "qualify.ini")
    dash.set_watch_meta(meta)
    for r in rows:
        dash.add_row(r)
    kinds = [k for _g, k, _ in dash.auto_marks]
    assert kinds == ["grow", "qualify_next", "grow", "qualify_next", "grow", "qualify_pick"]
    assert dash._auto_tips[-1] == S.DASH_MARK_TIP_QUALIFY_PICK.format(gen=6, winner=2)  # 다음 줄 전: 9.0을 낸 출전 2
    dash.add_row({"generation": 7, "champion_score": 1.0, "learner": 1, "qualifying": 0, "learner_best": 9.0,
                  "learner_stall": 0, "switch": 0, "brain_neurons": 4, "brain_links": frame.links([4]), "grow": 0})
    assert dash._auto_tips[-1] == S.DASH_MARK_TIP_QUALIFY_PICK.format(gen=6, winner=2)


# ---------------------------------------------------------------------------
# 기록·비교·경주·반복 실행 (T52)
# ---------------------------------------------------------------------------
def test_method_name_in_records_compare_race_and_repeat(qtbot: QtBot, tmp_path: Path) -> None:
    from evolab.app.compare import repeat as R
    from evolab.app.compare.view import RunEntry
    from evolab.app.race.model import _Run
    from evolab.app.records.view import RecordsView
    from evolab.engine.runsetup import run_name
    from evolab.runs.catalog import read_summary

    name = "CMA-ES · 자라는 뇌"
    assert S.METHOD_NAMES["grow"] == name
    run = tmp_path / "run"
    shutil.copytree(DATA / "v10_run_ga" / "run", run)
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    info["config"] = {**info["config"], "method": "grow", "grow_start": 2, "grow_every": 10}
    info.pop("method", None)
    (run / "run.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    summary = read_summary(run)
    assert summary.method == "grow"
    assert name in [text for text, _key in RecordsView._cells(summary)]
    assert RunEntry(summary).method_name == name
    store = RunStore.open(run)
    assert _Run(run, "#000000", store, store.read_run_info(), []).method_name == name
    spec = compile_env(env_text("lander"))
    assert run_name(spec, TrainConfig(method="grow")).endswith(f"_{spec.file.name}_{name}")
    # 반복 실행: 목록 끝에 자라는 뇌, 묶음의 실행마다 grow_*를 같이 넘긴다
    assert R.METHODS[-1] == "grow"
    dlg = R.RepeatDialog(runs_dir=tmp_path, workers=1)
    qtbot.addWidget(dlg)
    assert dlg.methods()[-1] == "grow" and dlg.method.itemText(dlg.method.count() - 1) == name
    assert not dlg.method_note.isVisibleTo(dlg) or name not in dlg.method_note.text()
    assert dlg.set_method("grow")
    plan = dlg.plan()
    assert plan.config["method"] == "grow" and (plan.config["grow_start"], plan.config["grow_every"]) == (2, 10)
    assert plan.config["popsize"] == 32 and plan.config["ipop"] is False
    for i in range(plan.count):
        c = plan.config_for(i)
        assert (c["method"], c["grow_start"], c["grow_every"], c["run_seed"]) == ("grow", 2, 10, 1 + i)
    assert dlg.group_name.placeholderText() == R.default_group_name(dlg.env_name, "grow", 1, 5)
    # 은닉층이 없는 환경: 목록에서 빠지되 「아직 쓸 수 없는 방법」에는 넣지 않는다
    flat = tmp_path / "flat.evoenv"
    flat.write_text(with_brain("lander", hidden=[]), encoding="utf-8")
    assert dlg.set_env_file(flat)
    assert "grow" not in dlg.methods() and name not in dlg.method_note.text()
    sa_ok = R.can_grow(layout_for(compile_env(env_text("lander")))) and not R.can_grow(None)
    assert sa_ok


# ---------------------------------------------------------------------------
# 에디터 「뇌 설정」 탭 안내 한 줄
# ---------------------------------------------------------------------------
def test_editor_brain_tab_has_the_grow_note(qtbot: QtBot) -> None:
    from evolab.app.editor.document import EnvDocument
    from evolab.app.editor.tabs import BottomTabs

    doc = EnvDocument()
    doc.from_builtin("lander.evoenv")
    tabs = BottomTabs(doc)
    qtbot.addWidget(tabs)
    tab = tabs.brain_tab
    assert tab.grow_note.text() == S.EDP_BRAIN_GROW_NOTE and not tab.grow_note.isHidden()
    doc.from_builtin("glider.evoenv")  # 뇌 없는 환경: 안내 없음
    tab.update_count()
    assert tab.grow_note.isHidden() and not tab.none_label.isHidden()


# ---------------------------------------------------------------------------
# 갱신 비용 (T53, 측정 M12와 같은 일을 넉넉한 상한으로)
# ---------------------------------------------------------------------------
def test_active_check_and_grow_tile_refresh_cost(qtbot: QtBot, tmp_path: Path) -> None:
    _spec, lay, meta, frame = lander()
    dash = fun_dashboard(qtbot, tmp_path, grow_rows(frame, 300), meta)
    for gen in range(290, 301):
        dash.add_generation(gen, None, grown_genome(lay, frame, [16], seed=gen)[None, :], 0, None)
    brain = dash.tile("brain")
    grow = dash.tile("brainsize")
    g = grown_genome(lay, frame, [8])
    brain.active_at(295)  # 세대 대표 켜짐은 한 번 계산해 기억한다
    times_active, times_tile = [], []
    for _ in range(30):
        t0 = time.perf_counter()
        brain._growth_marks(lay, g, 300)
        times_active.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        grow.refresh()
        times_tile.append(time.perf_counter() - t0)
    active_ms = float(np.median(times_active)) * 1000
    tile_ms = float(np.median(times_tile)) * 1000
    print(f"\n[T53] active check {active_ms:.3f} ms, grow tile refresh {tile_ms:.3f} ms (median of 30)")
    assert active_ms < 5.0 and tile_ms < 20.0  # 목표 0.5ms(M12에서 잰다). 공유 CI 기계라 열 배 넘게 넉넉히


# ---------------------------------------------------------------------------
# 학습 프로세스와 함께(L 묶음의 trainer·session): 진짜 짧은 자라는 실행
# ---------------------------------------------------------------------------
def test_real_growing_run_reaches_the_lab_status_tile_and_brain(qtbot: QtBot, tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    view = make_lab(qtbot, tmp_path, monkeypatch, "real.ini")
    try:
        assert view.load_env(LANDER)
        view.rb_fast.setChecked(True)
        view.popsize.setValue(6)
        view.trials.setValue(2)
        view.workers.setValue(1)
        view.max_generations.setValue(7)
        view.brain_combo.setCurrentIndex(view.brain_combo.findData(BRAIN_GROW))
        view.grow_every.setValue(2)
        assert view.start_new()
        qtbot.waitUntil(lambda: view.state == "finished", timeout=180000)
        grows = [int(float(r["grow"])) for r in view.rows]
        neurons = [int(float(r["brain_neurons"])) for r in view.rows]
        assert grows == [0, 1, 0, 1, 0, 1, 0] and neurons == [2, 2, 4, 4, 8, 8, 16]
        assert view.status.text().count("뇌가 자랐습니다") == 1
        assert S.LAB_STATUS_GROWN.format(gen=6, old=8, new=16, old_links=frame_links(8), new_links=frame_links(16)) \
            in view.status.text()
        dash = view.dashboard
        dash.set_page("fun")
        dash.flush()
        tile = dash.tile("brainsize")
        assert tile.series is not None and [int(tile.gens[i]) for i in tile.series.events] == [2, 4, 6]
        assert [g for g, k, _ in dash.auto_marks if k == "grow"] == [2, 4, 6]
        assert dash.brain_grows and dash.tile("brain").brain.active is not None
        run_dir = view.run_dir
        view.end_session()
        assert view.open_run(run_dir)  # 다시 열면 「뇌」 줄이 그 실행 설정
        assert view.brain_combo.currentData() == BRAIN_GROW and view.grow_every.value() == 2
    finally:
        view.client.abandon(timeout=60)
        view.shutdown()


def frame_links(n: int) -> int:
    """착륙선(입력 9, 출력 3) 은닉 n개 뇌의 연결 수."""
    return 9 * n + n * 3

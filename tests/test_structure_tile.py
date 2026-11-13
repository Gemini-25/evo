"""v2.1 「몸 구조」 타일과 실험실 연결 (v2 설계 4.13·5.3·5.4, 8.6절 T74~T76, 9.2 M13).

- T74: obs_body로 센 구조 비율 = structure_key를 개체마다 부른 직접 계산, 많은 6개 + 기타, 범례 글, 관찰 기록이 없는
  세대는 세대 대표 하나, 세대마다 그 세대의 환경 버전, 챔피언 몸 그림(없는 부품은 점선).
- T75: 빈 상태(없어질 수 있는 부품이 없는 환경), 「보는 재미」 쪽 순서, 보이지 않으면 갱신하지 않음(v1.2 펌프).
- T76: 클릭 검사 「(없음)」(런타임 Episode.absent와 같은 채널), 옛 GA 실행 안내 LAB_GA_NO_STRUCTURE.
- M13(slow): 750세대 기록에서 세대 하나가 올 때 갱신 + 그리기 시간.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from pytestqt.qtbot import QtBot
from test_dashboard import make_dashboard
from test_history import fake_generation, stats_row, write_run

from evolab import paths
from evolab.app import strings as S
from evolab.app.lab import dashboard as D
from evolab.app.lab import structure_tile as ST
from evolab.app.lab.view import InspectPanel, LabView, absent_inputs, structure_notice
from evolab.app.lab.watch_types import WatchMeta
from evolab.engine import seeds
from evolab.engine.replay import replay
from evolab.engine.runsetup import create_run
from evolab.engine.trainer import STRUCTURE_FLIP_NEW_RUN, TrainConfig
from evolab.policy.genome import GenomeLayout, layout_for
from evolab.runs import observe as ob
from evolab.spec import structure as st
from evolab.spec.compiled import EnvSpec
from evolab.spec.io import compile_env
from evolab.world.physics2d.episode import Episode

ENVS = paths.builtin_env_dir()
WHEELS = ENVS / "wheels.evoenv"
WHEELS_TEXT = WHEELS.read_text(encoding="utf-8")
WHEEL_IDS = ("wheel1", "wheel2", "wheel3", "wheel4")


def wheels() -> EnvSpec:
    return compile_env(WHEELS_TEXT)


def builtin(name: str) -> EnvSpec:
    return compile_env((ENVS / f"{name}.evoenv").read_text(encoding="utf-8"))


def variant() -> EnvSpec:
    """있음 칸의 모든 종류: 열("=p.w1_on"), 일반 수식(wheel3), 상수 0.3(wheel4, 늘 없음), 관절로만 매달린 물체(hub,
    wheel2에 붙음: wheel2가 없으면 함께 없음), 없어질 수 있는 관절의 관절 각도 센서(ja2)와 hub의 몸 상태 센서(hs)."""
    d = json.loads(WHEELS_TEXT)
    bodies = {b["id"]: b for b in d["bodies"]}
    bodies["wheel3"]["exists"] = "=p.w3_on * 2 - p.w3_r"
    bodies["wheel4"]["exists"] = 0.3
    hub = copy.deepcopy(bodies["wheel2"])
    hub.update(id="hub", label="허브")
    hub.pop("exists")
    d["bodies"].append(hub)
    d["joints"].append({"id": "jh", "kind": "pivot", "a": "wheel2", "b": "hub", "anchor_a": [0, 0]})
    d["sensors"].append({"kind": "joint_angle", "id": "ja2", "label": "바퀴2 각도", "joint": "j2"})
    d["sensors"].append({"kind": "body_state", "id": "hs", "label": "허브 상태", "body": "hub", "fields": ["vx"]})
    return compile_env(d)


def without_exists(spec: EnvSpec) -> EnvSpec:
    """같은 매개변수(유전자 배치)인데 있음 칸이 없는 환경(구조를 쓰기 전 버전)."""
    env = spec.file
    return compile_env(env.model_copy(update={"bodies": [b.model_copy(update={"exists": None}) for b in env.bodies]}))


def genome_with(layout: GenomeLayout, base: np.ndarray | None = None, **values: float) -> np.ndarray:
    """처음 유전자(또는 base)에서 몸 수치 몇 개를 물리 값으로 바꾼 유전자."""
    g = layout.initial() if base is None else np.array(base, dtype=np.float64)
    for k, gene in enumerate(layout.body):
        if gene.id in values:
            g[k] = 2.0 * (values[gene.id] - gene.low) / (gene.high - gene.low) - 1.0
    return g


def wheels_on(layout: GenomeLayout, on: tuple[int, ...], base: np.ndarray | None = None) -> np.ndarray:
    return genome_with(layout, base, **{f"w{k}_on": (0.75 if k in on else 0.25) for k in range(1, 5)})


def direct_ratios(spec: EnvSpec, body: np.ndarray) -> dict[tuple[str, ...], float]:
    """개체마다 structure_key(spec, param_values(몸 수치))를 불러 센 비율(타일과 독립인 계산)."""
    keys = [st.structure_key(spec, st.param_values(spec.file, row)) for row in np.atleast_2d(body)]
    uniq, counts = np.unique(np.array([",".join(k) for k in keys], dtype=object), return_counts=True)
    return {tuple(u.split(",")) if u else (): int(c) / len(keys) for u, c in zip(uniq, counts, strict=True)}


def obs_for(gen: int, layout: GenomeLayout, X: np.ndarray) -> ob.Observation:
    """X(정해 둔 집단)의 관찰 기록: 가짜 기록의 점수·결과에 그 유전자·몸 수치를 넣는다."""
    obs, _ = fake_generation(gen, layout, pop=len(X))
    return dataclasses.replace(obs, genomes=np.clip(X, -ob.FLOAT16_MAX, ob.FLOAT16_MAX).astype(np.float16),
                               body=layout.body_values(X))


def struct_dashboard(qtbot: QtBot, tmp_path: Path, spec: EnvSpec, name: str = "structure.ini",
                     versions: list[tuple[int, Any]] | None = None) -> D.Dashboard:
    """「보는 재미」 쪽 대시보드(띄우지 않아 바로 갱신). 유전자 지도 축 맞추기(eigh)는 이 시험과 무관해 숨긴다."""
    dash = make_dashboard(qtbot, tmp_path, name)
    dash.set_watch_meta(WatchMeta.from_spec(spec))
    dash.set_page("fun")
    dash.set_tile_visible("genemap", False)
    tile(dash).set_env_versions(versions if versions is not None else [(0, spec)])
    return dash


def tile(dash: D.Dashboard) -> ST.StructureTile:
    t = dash.tile("structure")
    assert isinstance(t, ST.StructureTile)
    return t


def feed(dash: D.Dashboard, gens: range | list[int], X_of: Any, *, observe: bool = True,
         champion_every: int = 0) -> dict[int, np.ndarray]:
    """학습 중처럼 add_row + add_generation. X_of(g) → 집단 X."""
    lay = dash.watch_meta.layout  # type: ignore[union-attr]
    out = {}
    for g in gens:
        X = X_of(g)
        obs = obs_for(g, lay, X)
        rep = obs.representative
        champ = (g, X[rep]) if champion_every and (g - 1) % champion_every == 0 else None
        extra = {"champion_score": repr(float(obs.scores.max()))} if champ is not None else {}
        dash.add_row({**stats_row(obs), **extra})
        dash.add_generation(g, obs.to_arrays() if observe else None, X, rep, champ)
        out[g] = X
    return out


def random_X(layout: GenomeLayout, pop: int = 32) -> Any:
    return lambda g: fake_generation(g, layout, pop=pop)[1]


def wait_body(qtbot: QtBot, dash: D.Dashboard) -> None:
    qtbot.waitUntil(lambda: not tile(dash).body_busy, timeout=30000)
    dash.flush()
    ST.wait_jobs()


# ---------------------------------------------------------------------------
# T74: 구조 비율, 많은 6개 + 기타, 범례 글, 대표만 있는 세대, 환경 버전, 챔피언 몸
def test_t74_ratios_equal_direct_counting_with_top_six_and_other(qtbot: QtBot, tmp_path: Path) -> None:
    spec = wheels()
    dash = struct_dashboard(qtbot, tmp_path, spec)
    lay = dash.watch_meta.layout  # type: ignore[union-attr]
    pops = feed(dash, range(1, 13), random_X(lay))
    t = tile(dash)
    assert not t.dirty and t.stack.currentWidget() is t.host
    assert t.gens.tolist() == list(range(1, 13)) and t.observed.all()
    direct = {g: direct_ratios(spec, lay.body_values(X)) for g, X in pops.items()}
    for g in pops:
        got = t.structure_at(g)
        assert got.keys() == direct[g].keys() and all(got[k] == pytest.approx(direct[g][k], abs=1e-12) for k in got)
    every = sorted({k for d in direct.values() for k in d})
    assert len(every) > ST.TOP_STRUCTURES  # 이 자료는 구조가 많다: 기타가 생긴다
    share = {k: sum(d.get(k, 0.0) for d in direct.values()) for k in every}
    shown = [k for k in t.series if k is not None]
    assert len(shown) == ST.TOP_STRUCTURES and t.series[-1] is ST.OTHER_KEY and t.names[-1] == S.STRUCT_OTHER
    assert min(share[k] for k in shown) >= max(share[k] for k in every if k not in shown)
    assert shown == sorted(shown, key=lambda k: (len(k), [WHEEL_IDS.index(b) for b in k]))  # 쌓는 순서: 바퀴 수
    for i, g in enumerate(pops):
        for j, k in enumerate(shown):
            assert t.ratios[j, i] == pytest.approx(direct[g].get(k, 0.0), abs=1e-12)
        assert t.ratios[-1, i] == pytest.approx(sum(v for k, v in direct[g].items() if k not in shown), abs=1e-12)
    assert np.allclose(t.ratios.sum(axis=0), 1.0)
    theme = dash.theme
    # 색은 구조마다 지키는 자리(검토 반영: 순서가 아니라 열쇠에 붙는다) — 6개가 범주 색 앞 6개를 하나씩 나눠 갖는다
    assert sorted(t.colors[:-1]) == sorted(theme.series[:ST.TOP_STRUCTURES]) and t.colors[-1] == theme.muted
    assert [e.name for e in t.entries()] == t.names and [e.color for e in t.entries()] == t.colors
    assert t.area.layer_count == len(t.series)
    x, y = t.area.layer_data(-1)
    assert np.allclose(y, 100.0) and len(x) == len(y) + 1  # 세대마다 한 칸(계단), 맨 위 누적 = 100%
    assert t.plot.getAxis("bottom").labelText == S.DASH_AXIS_NAMES["generation"]
    # 보는 세대 세로선
    dash.set_focus(4)
    dash.flush()
    assert t.focus_line.isVisible() and t.focus_line.value() == 4.0
    # 마우스를 올리면 그 세대의 구조별 비율
    t.hover(5.2, 50.0, 0)
    assert t._hover_text.startswith(S.DASH_HOVER_X["generation"].format(gen=5, x=""))
    t.unhover()


def test_t74_long_runs_draw_an_area_through_the_generations(qtbot: QtBot, tmp_path: Path) -> None:
    spec = wheels()
    dash = struct_dashboard(qtbot, tmp_path, spec)
    lay = dash.watch_meta.layout  # type: ignore[union-attr]
    n = ST.STEP_LIMIT + 5
    feed(dash, range(1, n + 1), random_X(lay, pop=3))
    t = tile(dash)
    x, y = t.area.layer_data(-1)
    assert len(x) == len(y) == n and np.allclose(y, 100.0) and x[0] == 1 and x[-1] == n  # 계단 대신 세대 점을 잇는다
    assert np.allclose(t.ratios.sum(axis=0), 1.0)


def test_t74_legend_counts_parts_and_names_them_when_counts_repeat(qtbot: QtBot, tmp_path: Path) -> None:
    labels = {f"wheel{k}": f"바퀴{k}" for k in range(1, 5)}
    assert ST.base_name("바퀴12") == "바퀴" and ST.base_name("wheel_2") == "wheel" and ST.base_name("7") == "7"
    assert ST.structure_label(("wheel1", "wheel4"), labels, ["바퀴"], False) == S.STRUCT_COUNT.format(name="바퀴", n=2)
    assert ST.structure_label(("wheel1", "wheel4"), labels, ["바퀴"], True) == \
        S.STRUCT_COUNT_WHICH.format(name="바퀴", n=2, which="바퀴1·바퀴4")
    assert ST.structure_label((), labels, ["바퀴"], False) == S.STRUCT_NONE
    two = {**labels, "leg1": "다리1"}
    text = ST.structure_label(("leg1", "wheel2"), two, ["바퀴", "다리"], False)
    assert S.STRUCT_COUNT.format(name="바퀴", n=1) in text and S.STRUCT_COUNT.format(name="다리", n=1) in text

    spec = wheels()
    dash = struct_dashboard(qtbot, tmp_path, spec)
    lay = dash.watch_meta.layout  # type: ignore[union-attr]
    X = np.vstack([wheels_on(lay, (1, 4))] * 3 + [wheels_on(lay, (2, 3))] * 2 + [wheels_on(lay, (1,))] * 2
                  + [wheels_on(lay, ())])
    feed(dash, range(1, 4), lambda g: X)
    t = tile(dash)
    assert t.series == [(), ("wheel1",), ("wheel1", "wheel4"), ("wheel2", "wheel3")]
    assert t.names == [S.STRUCT_NONE, S.STRUCT_COUNT.format(name="바퀴", n=1),
                       S.STRUCT_COUNT_WHICH.format(name="바퀴", n=2, which="바퀴1·바퀴4"),
                       S.STRUCT_COUNT_WHICH.format(name="바퀴", n=2, which="바퀴2·바퀴3")]
    assert np.allclose(t.ratios[:, 0], [1 / 8, 2 / 8, 3 / 8, 2 / 8])
    for name in t.names:
        assert name in t.legend.text()


def test_t74_generations_without_records_count_the_representative_only(qtbot: QtBot, tmp_path: Path) -> None:
    spec = wheels()
    dash = struct_dashboard(qtbot, tmp_path, spec)
    lay = dash.watch_meta.layout  # type: ignore[union-attr]
    X = np.vstack([wheels_on(lay, (1, 2))] * 5 + [wheels_on(lay, (3,))])
    feed(dash, [1], lambda g: X)  # 관찰 기록이 있는 세대
    dash.add_row(stats_row(obs_for(2, lay, X)))
    dash.add_generation(2, None, X, 5, None)  # 관찰을 끈 학습: 세대 대표(5번, 바퀴3 하나)만
    dash.flush()
    t = tile(dash)
    assert t.structure_at(1) == pytest.approx({("wheel1", "wheel2"): 5 / 6, ("wheel3",): 1 / 6})
    assert t.structure_at(2) == {("wheel3",): 1.0}
    assert t.observed.tolist() == [True, False]
    assert S.STRUCT_REPS_ONLY in t.note.text() and t.note.isVisibleTo(t)

    # 기록에서 연 실행: 관찰 키가 없는 세대 대표 파일(v1.0·v1.1 꼴)도 대표 하나로 센다
    root = tmp_path / "run"
    gens = []
    for g in range(1, 6):
        Xg = np.vstack([wheels_on(lay, (1, 4))] * 3 + [wheels_on(lay, (g % 4 + 1,))])
        o = obs_for(g, lay, Xg)
        gens.append((o, Xg))
    rows = write_run(root, gens, with_obs=False)
    dash.load_history(root, rows)
    assert dash.wait_history(30000)
    dash.flush()
    assert t.gens.tolist() == [1, 2, 3, 4, 5] and not t.observed.any()
    for o, Xg in gens:
        rep_key = st.structure_key(spec, st.param_values(spec.file, lay.body_values(Xg[o.representative])))
        assert t.structure_at(o.generation) == {rep_key: 1.0}


def test_t74_counter_matches_present_for_every_kind_of_exists() -> None:
    """열·일반 수식·상수 있음 칸과 관절로만 매달린 물체(이어짐 규칙): 타일의 numpy 세기 = structure.present."""
    spec = variant()
    lay = layout_for(spec)
    counter = ST.StructureCounter(spec)
    assert counter.optional == st.optional_bodies(spec.file) == {"wheel1", "wheel2", "wheel3", "wheel4", "hub"}
    rng = np.random.default_rng(5)
    body = lay.body_values(rng.uniform(-1.2, 1.2, (600, lay.dim)))
    body[:5] = np.nan  # 유한하지 않은 값은 매개변수 값(param_values와 같은 규칙)
    body[5:10] *= 3.0  # 범위 밖은 자른다
    order = [b.id for b in spec.file.bodies if b.id in counter.optional]
    codes = counter.codes(body)
    for row, code, key in zip(body, codes, counter.keys(body, order), strict=True):
        params = st.param_values(spec.file, row)
        assert counter.present_of(int(code)) == st.present(spec, params)
        assert key == st.structure_key(spec, params)
    seen = {k for k in counter.keys(body, order)}
    assert any("wheel2" not in k and "hub" not in k for k in seen) and all("wheel4" not in k for k in seen)
    assert all(("hub" in k) == ("wheel2" in k) for k in seen)  # hub는 wheel2와 함께 있고 없어진다
    # 여러 세대를 한 번에 센 것 = 세대마다 센 것
    blocks = [body[10:42], body[42:43], body[43:100]]
    for (got_codes, got_counts), block in zip(counter.count(blocks), blocks, strict=True):
        u, c = np.unique(counter.codes(block), return_counts=True)
        assert got_codes == u.tolist() and got_counts.tolist() == c.tolist()


def test_t74_each_generation_is_counted_with_its_environment_version(qtbot: QtBot, tmp_path: Path) -> None:
    spec = wheels()
    before = without_exists(spec)  # 구조를 쓰기 전 버전: 모든 바퀴가 있다
    dash = struct_dashboard(qtbot, tmp_path, spec, versions=[(0, before), (3, WHEELS_TEXT)])
    t = tile(dash)
    assert [t.version_of(g) for g in (1, 3, 4, 9)] == [0, 0, 1, 1]
    lay = dash.watch_meta.layout  # type: ignore[union-attr]
    X = np.vstack([wheels_on(lay, (1,))] * 4)
    feed(dash, range(1, 7), lambda g: X)
    for g in (1, 2, 3):
        assert t.structure_at(g) == {WHEEL_IDS: 1.0}
    for g in (4, 5, 6):
        assert t.structure_at(g) == {("wheel1",): 1.0}
    t.add_env_version(5, before)  # 학습 중 환경 바꾸기: 6세대부터
    dash.flush()
    assert t.version_of(6) == 2 and t.structure_at(6) == {WHEEL_IDS: 1.0} and t.structure_at(5) == {("wheel1",): 1.0}


def test_t74_champion_body_draws_present_parts_and_dashed_absent_parts(qtbot: QtBot, tmp_path: Path) -> None:
    spec = wheels()
    lay = layout_for(spec)
    g14 = wheels_on(lay, (1, 4))
    body = lay.body_values(g14)
    full = ST.full_structure_spec(spec)
    assert full is not None and ST.full_structure_spec(builtin("lander")) is None
    parts = ST.champion_bodies(spec, body, full)
    roles = {b.id: role for b, role in parts}
    assert roles == {"chassis": "body", "wheel1": "part", "wheel4": "part", "wheel2": "absent", "wheel3": "absent"}
    real = {b.id: b for b in Episode(spec, body, seeds.display_trial()).geometry().bodies if b.owner == "agent"}
    whole = {b.id: b for b in Episode(full, body, seeds.display_trial()).geometry().bodies}
    for b, role in parts:
        if role != "absent":
            assert b == real[b.id]  # Episode(spec, 몸, 전시 시드).geometry()의 에이전트 물체 그대로
            continue
        # 뿌리(차체) 기준 자리는 모든 부품이 있는 몸과 같다
        rc, rw = real["chassis"].pose, whole["chassis"].pose
        rel = ST._relative_pose(b.pose, rc, rw)
        assert rel == pytest.approx(whole[b.id].pose, abs=1e-9)

    dash = struct_dashboard(qtbot, tmp_path, spec)
    X = np.vstack([g14] * 3 + [wheels_on(lay, (2,))])
    X1 = np.vstack([wheels_on(lay, (1,))] * 4)
    feed(dash, [1], lambda g: X1, champion_every=1)
    feed(dash, [2, 3], lambda g: X, champion_every=1)
    t = tile(dash)
    wait_body(qtbot, dash)
    cg = dash.champion_at(3)
    assert cg is not None
    want = st.structure_key(spec, st.param_values(spec.file, lay.body_values(cg[1])))
    assert t.champion == (cg[0], want)
    assert t.caption.text() == f"{S.STRUCT_CHAMPION.format(gen=cg[0])} · {t.key_names[want]}"
    assert sorted(role for _, _, role in t.picture.items).count("absent") == 4 - len(want)
    assert t.picture.part_color == t.colors[t.series.index(want)]
    assert not t.picture.grab().isNull()
    # 앞 세대를 보면 그 세대까지의 챔피언(1세대: 바퀴1 하나)
    dash.set_focus(1)
    dash.flush()
    wait_body(qtbot, dash)
    assert t.champion == (1, ("wheel1",))
    assert [r for _, _, r in t.picture.items].count("absent") == 3


# ---------------------------------------------------------------------------
# T75: 빈 상태, 쪽 순서, 보일 때만 갱신
@pytest.mark.parametrize("env", ["lander", "jump", "glider"])
def test_t75_environments_without_optional_parts_show_the_empty_state(qtbot: QtBot, tmp_path: Path,
                                                                      env: str) -> None:
    spec = builtin(env)
    dash = struct_dashboard(qtbot, tmp_path, spec, f"{env}.ini")
    lay = dash.watch_meta.layout  # type: ignore[union-attr]
    feed(dash, range(1, 3), random_X(lay, pop=4))
    t = tile(dash)
    assert t.stack.currentWidget() is t.message and t.message.text() == S.STRUCT_EMPTY
    assert t.series == [] and t.gens.size == 0


def test_t75_waiting_and_no_environment_states(qtbot: QtBot, tmp_path: Path) -> None:
    spec = wheels()
    dash = struct_dashboard(qtbot, tmp_path, spec)
    t = tile(dash)
    assert t.stack.currentWidget() is t.message and t.message.text() == S.STRUCT_WAIT  # 학습 전
    t.set_env_versions([])
    dash.flush()
    assert t.message.text() == S.STRUCT_EMPTY  # 환경을 모르면(실험실이 버전을 주기 전)


def test_t75_tile_is_last_on_the_fun_page_and_old_layouts_get_it(qtbot: QtBot, tmp_path: Path) -> None:
    assert D.FUN_TILE_IDS[-1] == "structure" and D.PAGE_OF["structure"] == "fun"
    assert D.tile_class("structure") is ST.StructureTile and "structure" not in D.TILE_IDS
    # v1.2가 저장한 「보는 재미」 쪽 배치(다섯 타일): 새 타일 v2.1 「몸 구조」는 그 뒤에 붙고, v6.1 「뇌 크기」는 「뇌 들여다보기」
    # 바로 뒤에 낀다(v6.1 고침: 설계 4.19의 자리. 통합 때는 맨 뒤 「몸 구조」 앞에 붙는 것으로 적었는데 그것이 결함이었다)
    s = QSettings(str(tmp_path / "v12.ini"), QSettings.Format.IniFormat)
    s.beginGroup("dashboard")
    s.setValue("order", ",".join([*D.TILE_IDS, "body", "genemap", "brain", "density", "actions"]))
    s.setValue("page", "fun")
    s.endGroup()
    s.sync()
    dash = make_dashboard(qtbot, tmp_path, "v12.ini")
    assert dash.tile_order("fun") == ["body", "genemap", "brain", "brainsize", "density", "actions", "structure"]
    t = dash.tile("structure")
    assert t.title.text() == S.DASH_TILE_TITLES["structure"] and t.title.toolTip() == S.DASH_TILE_HELP["structure"]
    assert dash.is_tile_visible("structure") and t.isVisibleTo(dash)
    dash._fill_tiles_menu()
    titles = [a.text() for a in dash.tiles_menu.actions() if a.isCheckable()]
    assert titles[-1] == S.DASH_TILE_TITLES["structure"]


def test_t75_tile_is_not_refreshed_while_it_cannot_be_seen(qtbot: QtBot, tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    spec = wheels()
    dash = make_dashboard(qtbot, tmp_path, "pump.ini")
    dash.set_watch_meta(WatchMeta.from_spec(spec))
    dash.set_tile_visible("genemap", False)
    dash.set_columns(3)
    dash.resize(1500, 1000)
    dash.show()
    qtbot.waitExposed(dash)
    t = tile(dash)
    t.set_env_versions([(0, spec)])
    calls: list[int] = []
    real = t.refresh
    monkeypatch.setattr(t, "refresh", lambda: (calls.append(1), real())[1])
    lay = dash.watch_meta.layout  # type: ignore[union-attr]
    feed(dash, range(1, 4), random_X(lay, pop=8))  # 「학습 그래프」 쪽: 그리지 않는다
    qtbot.wait(50)
    assert dash.page == "learning" and t.dirty and calls == []
    dash.set_page("fun")
    qtbot.waitUntil(lambda: not t.dirty, timeout=10000)  # 보이면 펌프가 그린다
    assert len(calls) >= 1 and t.gens.tolist() == [1, 2, 3]
    dash.set_tile_visible("structure", False)
    calls.clear()
    feed(dash, [4], random_X(lay, pop=8))
    qtbot.wait(50)
    assert t.dirty and calls == []
    dash.set_tile_visible("structure", True)
    qtbot.waitUntil(lambda: not t.dirty, timeout=10000)
    assert calls and t.gens.tolist() == [1, 2, 3, 4]
    qtbot.waitUntil(lambda: not t.body_busy, timeout=30000)
    ST.wait_jobs()


# ---------------------------------------------------------------------------
# T76: 클릭 검사 「(없음)」, 옛 GA 실행 안내
def test_t76_absent_inputs_match_the_runtime_and_the_panel_marks_them(qtbot: QtBot) -> None:
    spec = variant()
    lay = layout_for(spec)
    genome = wheels_on(lay, (1, 3))  # wheel2 없음(→ hub도), wheel4는 상수 0.3이라 늘 없음
    body = lay.body_values(genome)
    ep = Episode(spec, body, seeds.display_trial())
    assert ep.absent == {"wheel2", "hub", "wheel4"} and ep.absent_joints == {"j2", "j4", "jh"}
    absent = absent_inputs(spec, lay, genome)
    assert absent == {"c2", "c4", "ja2", "hs.vx"}
    rep = replay(spec, lay, genome, seeds.display_trial(), trace=True)
    assert rep.observations is not None
    for name in absent:  # 런타임에서 그 채널은 늘 0
        assert not rep.observations[:, rep.inputs.index(name)].any()
    panel = InspectPanel()
    qtbot.addWidget(panel)
    panel.set_replay(2, rep, absent)
    panel.show_time(1.0)
    lines = {line.split()[0]: line for line in panel.body.text().splitlines() if line and line.split()[0] in rep.inputs}
    assert lines.keys() == set(rep.inputs)
    for name, line in lines.items():
        assert line.endswith(S.INSPECT_ABSENT) == (name in absent), line
        if name in absent:  # 늘 0인 값 대신 이름 바로 옆(좁은 패널에서도 가로로 밀지 않고 보인다)
            assert line.split() == [name, S.INSPECT_ABSENT], line
    panel.set_replay(2, rep)  # 「(없음)」을 주지 않으면 표시 없음(v2.0과 같음)
    panel.show_time(1.0)
    assert S.INSPECT_ABSENT not in panel.body.text()
    # 있음 칸이 없는 환경, 모든 부품이 있는 개체
    lander = builtin("lander")
    assert absent_inputs(lander, layout_for(lander), layout_for(lander).initial()) == frozenset()
    w = wheels()
    assert absent_inputs(w, layout_for(w), wheels_on(layout_for(w), (1, 2, 3, 4))) == frozenset()


@pytest.fixture()
def lab(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path / "runs")
    view = LabView(settings=QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.show()
    qtbot.waitExposed(view)
    assert view.load_env(WHEELS)
    try:
        yield view
    finally:
        view.client.abandon(timeout=60)
        view.shutdown()


def test_t76_lab_click_inspect_marks_sensors_of_absent_parts(qtbot: QtBot, lab: Any) -> None:
    lay = lab.layout
    pop = np.vstack([wheels_on(lay, (1, 3)), wheels_on(lay, (1, 2, 3, 4))])
    lab.view_mode = "live"
    lab._last_played = {"row": {}, "generation": 1, "population": pop, "representative": 0, "scores": np.zeros(2),
                        "trial": (1, None), "trajectories": None, "champion": None, "env_version": None,
                        "observation": None}
    lab._inspect(0)
    qtbot.waitUntil(lambda: lab.inspect.isVisible(), timeout=60000)
    assert lab.inspect.absent == {"c2", "c4"}
    text = lab.inspect.body.text()
    marked = [line.split()[0] for line in text.splitlines() if line.endswith(S.INSPECT_ABSENT)]
    assert marked == ["c2", "c4"]
    lab._close_inspect()
    lab._inspect(1)  # 바퀴가 모두 있는 개체: 표시 없음
    qtbot.waitUntil(lambda: lab.inspect.isVisible() and lab.inspect.absent == frozenset(), timeout=60000)
    assert S.INSPECT_ABSENT not in lab.inspect.body.text()


def test_t76_old_ga_run_on_a_structure_env_is_noticed_once(qtbot: QtBot, lab: Any, tmp_path: Path) -> None:
    spec, lander = wheels(), builtin("lander")
    old = {"method": "ga", "popsize": 6}  # v2.0이 저장한 설정: 구조 뒤집기 키가 없다(0)
    assert structure_notice(old, spec) == S.LAB_GA_NO_STRUCTURE
    assert structure_notice({**old, "method": "sa"}, spec) == S.LAB_GA_NO_STRUCTURE
    assert structure_notice({**old, "ga_structure_flip": 0.05}, spec) is None
    assert structure_notice({**old, "method": "cmaes"}, spec) is None
    assert structure_notice(old, lander) is None and structure_notice(old, None) is None

    store = create_run(tmp_path / "runs", WHEELS_TEXT, spec, TrainConfig(method="ga", popsize=6), 0)
    info = store.read_run_info()
    msg = {"run_dir": str(store.root), "run_info": info, "env_text": WHEELS_TEXT, "env_version": None,
           "generation": 0, "config": info["config"], "workers": 1, "layout": info["layout"], "stats": [],
           "events": [], "champion": None, "spectate": False}
    lab._on_started(msg)
    text = lab.status.text()
    assert S.LAB_STARTED_RUN.format(path=lab.run_dir) in text and text.endswith(S.LAB_GA_NO_STRUCTURE)
    lab._update_status()
    assert lab.status.text().count(S.LAB_GA_NO_STRUCTURE) == 1  # 상태 글을 다시 만들어도 남는다
    noted = list(lab._structure_noted)
    lab._on_started(msg)  # 같은 실행은 한 번만: 다시 알리지 않는다(남아 있는 끝줄은 하나)
    assert lab._structure_noted == set(noted) == {str(store.root)}
    assert lab.status.text().count(S.LAB_GA_NO_STRUCTURE) <= 1
    # 몸 구조 타일은 이 실행의 환경 버전(지금 환경과 같은 글이면 이미 컴파일한 spec)
    t = tile(lab.dashboard)
    assert t._versions == [(0, lab.spec)]
    # 학습 중 환경 바꾸기: 그 버전을 더한다
    changed = WHEELS_TEXT.replace('"name": "', '"name": "x', 1)
    assert changed != WHEELS_TEXT
    lab._on_applied({"ok": True, "what": "env", "generation": 4, "env_text": changed, "changes": ["name"]})
    assert [g for g, _ in t._versions] == [0, 4] and t._versions[-1][1] is lab.spec


def test_new_runs_from_the_lab_get_the_structure_flip_default(qtbot: QtBot, lab: Any) -> None:
    lab.method.setCurrentIndex(lab.method.findData("ga"))
    c = lab.config_from_form()
    assert c.ga_structure_flip == STRUCTURE_FLIP_NEW_RUN and c.sa_structure_flip == STRUCTURE_FLIP_NEW_RUN
    # 복제한 v2.0 설정(키 없음)도 새 실행이면 기본값, 복제한 값이 있으면 그 값
    lab._base_config = {"method": "ga", "popsize": 6}
    assert lab.config_from_form().ga_structure_flip == STRUCTURE_FLIP_NEW_RUN
    lab._base_config = {"method": "ga", "popsize": 6, "ga_structure_flip": 0.0}
    assert lab.config_from_form().ga_structure_flip == 0.0
    lab._base_config = None
    assert lab.load_env(ENVS / "lander.evoenv")
    assert lab.config_from_form().ga_structure_flip == 0.0  # 있음 유전자가 없는 환경


# ---------------------------------------------------------------------------
# M13: 750세대 기록에서 세대 하나가 올 때 갱신 + 그리기
@pytest.mark.slow
def test_m13_generation_arrival_update_and_paint_on_a_750_generation_run(qtbot: QtBot, tmp_path: Path) -> None:
    import os

    spec = wheels()
    dash = struct_dashboard(qtbot, tmp_path, spec, "m13.ini")
    dash.resize(1400, 1100)
    dash.set_columns(2)
    dash.show()
    qtbot.waitExposed(dash)
    lay = dash.watch_meta.layout  # type: ignore[union-attr]
    root = tmp_path / "run"
    rows = write_run(root, [fake_generation(g, lay, pop=32) for g in range(1, 751)])
    dash.load_history(root, rows)
    assert dash.wait_history(120000)
    t = tile(dash)
    dash.flush()
    assert t.gens.size == 750
    wait_body(qtbot, dash)
    # 펌프처럼 읽은 기록을 먼저 그려 둔다: dash.flush()는 갱신만 하고 그리지 않아, 이것이 없으면 첫 세대 도착에 기록 전체의
    # 첫 그림(세대 750개 다각형, 범례·제목·몸 그림 배치)이 섞인다. 그 첫 그림은 따로 잰다(열기 쪽 비용, v1.2 이벤트 루프 목표)
    t0 = time.perf_counter()
    t.prepare_paint()
    t.repaint()
    first = (time.perf_counter() - t0) * 1000.0
    times: list[float] = []
    for g in range(751, 771):
        obs, X = fake_generation(g, lay, pop=32)
        dash.add_row(stats_row(obs))
        dash.add_generation(g, obs.to_arrays(), X, obs.representative, None)
        t0 = time.perf_counter()
        t.refresh()
        t.prepare_paint()
        t.repaint()
        times.append((time.perf_counter() - t0) * 1000.0)
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
    print(f"\n[M13] structure tile update+paint per generation on 750 generations: median {np.median(times):.1f} ms, "
          f"worst {max(times):.1f} ms, first paint of the loaded history {first:.1f} ms (load {load:.2f})")
    assert t.gens.size == 770
    assert np.median(times) < 100.0  # 목표 10ms(M13)는 보고서에 잰 값으로. 여기서는 큰 퇴행만

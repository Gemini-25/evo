"""v3.1 P 폰 재생 파일 (v3 설계 8.1 T42·T43, 2.8, 4.18, D56·D57, 7.3 「app/export/phone.py」).

- T42: decode_payload(build_payload(job))의 궤적이 원래와 1mm·1e-4rad 안, 출력 0..100, CSP 메타, 바깥 주소·src=(data: 말고)·
  <link·fetch·XMLHttpRequest·import 없음, 환경 이름 </script><b>에도 구조가 그대로, 문구 = PHONE_TEXT, 재생기 틀은 ASCII만,
  크기(언덕차 최대 시간·착륙선 1세대 vs 지금 ≤ 300KB, 경주 6칸 ≤ 1MB), 칸마다 세계, v2 움직이는 세계 물체 자세 = 프레임
  시각의 pose_at, v2.1 없는 물체, 상한.
- T43: 헤드리스 크롬(Playwright, node의 require("playwright"))이 있으면 390×844 폰 화면으로 열어 콘솔 오류 0, 바깥 요청 0,
  캔버스 폭 ≤ CSS 폭 × 2, 캔버스에 그림이 그려지고 끝 장면 그림이 숨음, 스크립트를 끄면 끝 장면 그림이 보임. 없으면 건너뜀
  (윈도우 CI의 tests job 포함: 윈도우 CI는 W9가 헤드리스 Edge로 따로 연다).

크기 시험은 base64 Int32 자료라 값과 무관하게 프레임·물체·출력 수로 정해지므로, 환경 최대 시간까지 가는 합성 궤적을 쓴다
(학습한 챔피언 측정값은 보고서 M9).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from evolab import __version__, paths
from evolab.app import strings as S
from evolab.app.export import job as J
from evolab.app.export import phone as P
from evolab.app.export.job import ExportJob
from evolab.app.race import model as race_model
from evolab.app.race.model import RaceEntry
from evolab.engine import seeds
from evolab.engine.evaluate import Trajectory
from evolab.engine.replay import Replay, replay
from evolab.policy.genome import layout_for
from evolab.spec.editor_meta import view_box
from evolab.world.physics2d.episode import Episode
from evolab.world.physics2d.motion import pose_at

DATA = Path(__file__).resolve().parent / "data"
CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:")
LANE_COLORS = ("#ff7828", "#508cff", "#3ddc84", "#ffd166", "#c678dd", "#56b6c2")


# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------
def _text(stem: str) -> str:
    return next(p for p in paths.builtin_envs() if p.stem == stem).read_text(encoding="utf-8-sig")


def _replay(stem: str, seed: int | None = None) -> Replay:
    """내장 환경을 전시 시드로 다시 시뮬레이션(seed가 있으면 그 시드의 작은 무작위 유전자)."""
    spec = race_model.compile_cached(_text(stem))
    layout = layout_for(spec)
    genome = np.zeros(layout.dim) if seed is None else np.random.default_rng(seed).normal(0.0, 0.6, layout.dim)
    return replay(spec, layout, genome, seeds.display_trial())


def _view(stem: str) -> tuple[float, float, float, float] | None:
    return view_box(race_model.compile_cached(_text(stem)).file)


def _env_name(stem: str) -> str:
    return str(race_model.compile_cached(_text(stem)).file.name)


def _overlay(reps: list[tuple[Replay, str, str]], stem: str, *, title: str = "지금 챔피언 (3세대)",
             env: str | None = None) -> ExportJob:
    return J.job_from_replays(reps, "html", size="medium", camera="fixed", title=title,
                              meta=J.make_meta({J.META_ENV: env if env is not None else _env_name(stem),
                                                J.META_RUN: "run 1", J.META_GENERATION: "3"}),
                              view=_view(stem))


def _single(stem: str, seed: int | None = 3) -> ExportJob:
    return _overlay([(_replay(stem, seed), "#ff7828", "지금 챔피언 (3세대)")], stem)


def _long(rep: Replay, frames: int, seed: int = 0) -> Replay:
    """같은 세계·물체에 프레임 수만 frames인 합성 궤적(크기·상한 시험용: base64 Int32 크기는 값과 무관)."""
    tr = rep.trajectory
    rng = np.random.default_rng(seed)
    nb, nc = tr.poses.shape[1], tr.outputs.shape[1]
    poses = np.cumsum(rng.normal(0.0, 0.05, (frames, nb, 3)), axis=0).astype(np.float32)
    outs = rng.uniform(-1.0, 1.0, (frames, nc)).astype(np.float32)
    end_step = (frames - 1) * tr.every
    return dataclasses.replace(rep, trajectory=Trajectory(poses, outs, tr.every, tr.dt, end_step))


def _max_frames(stem: str) -> int:
    spec = race_model.compile_cached(_text(stem))
    rep = _replay(stem)
    steps = round(float(spec.file.world.max_time) / rep.trajectory.dt)
    return steps // rep.trajectory.every + 1


def _wrap(a: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(a, dtype=np.float64) + math.pi, 2.0 * math.pi) - math.pi


def _assert_round_trip(job: ExportJob) -> dict[str, Any]:
    payload = P.build_payload(job)
    dec = P.decode_payload(json.loads(json.dumps(payload, allow_nan=False)))
    assert len(dec["entries"]) == len(job.items)
    for item, e in zip(job.items, dec["entries"], strict=True):
        tr = item.replay.trajectory
        want = np.nan_to_num(tr.poses.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        got = e["poses"]
        assert got.shape == want.shape == (e["frames"], e["nb"], 3)
        assert np.abs(got[..., :2] - want[..., :2]).max(initial=0.0) <= 0.5e-3 + 1e-9  # 1mm 칸의 반올림
        assert np.abs(_wrap(got[..., 2] - want[..., 2])).max(initial=0.0) <= 0.5e-4 + 1e-9
        assert (got[..., 2] >= -math.pi - 1e-4).all() and (got[..., 2] <= math.pi + 1e-4).all()
        outs = np.clip(np.nan_to_num(tr.outputs.astype(np.float64)), 0.0, 1.0)
        assert e["out"].shape == outs.shape
        assert np.abs(e["out"] - outs).max(initial=0.0) <= 0.005 + 1e-9
        assert e["step"] == tr.every * tr.dt and e["last"] == tr.duration
        assert e["label"] == item.label and e["color"] == item.color
        assert e["roles"] == [P._role(b) for b in item.replay.agent]
        assert len(e["bodies"]) == len(item.replay.agent)
    return payload


def _html_checks(text: str, *, names_hold_at: bool = False) -> None:
    """바깥 요청이 없는 구조(T42): CSP 메타, 바깥 주소·<link·fetch·XMLHttpRequest·import 없음, src=는 data:뿐, 틀 자리가
    남지 않음(names_hold_at이면 이름에 일부러 넣은 @@...@@는 자료·제목에 그대로 있으므로 이 검사만 뺀다)."""
    assert f'<meta http-equiv="Content-Security-Policy" content="{CSP}">' in text
    assert text.index("Content-Security-Policy") < text.index("<script")
    for word in ("http:", "https:", "<link", "fetch", "XMLHttpRequest", "import", "<iframe", "<object", "<embed",
                 "@import", "url("):
        assert word not in text, word
    for m in re.finditer(r"src\s*=", text):
        assert text[m.end():m.end() + 6] == '"data:', text[m.start():m.start() + 30]
    assert text.count("<script") == 2 and text.count("</script>") == 2
    if not names_hold_at:
        assert not re.search(r"@@[A-Z]+@@", text)


# ---------------------------------------------------------------------------
# 틀
# ---------------------------------------------------------------------------
def test_template_is_ascii_without_hangul_and_holds_each_placeholder() -> None:
    raw = P.TEMPLATE_PATH.read_bytes()
    assert all(b < 128 for b in raw), "phone_player.html must be ASCII only (words come from PHONE_TEXT)"
    text = raw.decode("ascii")
    found = re.findall(r"@@([A-Z]+)@@", text)
    assert sorted(set(found)) == sorted(P.PLACEHOLDERS)
    assert found.count("TITLE") == 2 and all(found.count(n) == 1 for n in P.PLACEHOLDERS if n != "TITLE")
    assert P.template_text() == text
    assert f'content="{CSP}"' in text
    # W9(헤드리스 Edge --dump-dom)는 <canvas가 있으면 스크립트가 돌았다고 본다: 틀에는 <canvas가 없고 스크립트가 만든다
    assert "<canvas" not in text
    assert '<img id="poster" ' in text and '<div id="player" hidden>' in text
    assert text.index("<noscript>") < text.index('<img id="poster"')  # 안내 「아래 그림은 마지막 장면」
    for word in ("http:", "https:", "<link", "fetch", "XMLHttpRequest", "import", "eval(", "new Function"):
        assert word not in text, word


def test_colors_match_spectator_canvas() -> None:
    from PySide6.QtGui import QColor

    from evolab.app.lab import canvas
    from evolab.app.play import window

    def css(c: QColor) -> str:
        if c.alpha() == 255:
            return c.name().lower()
        return f"rgba({c.red()},{c.green()},{c.blue()},{round(c.alpha() / 255, 3):g})"

    want = {"bg": window.COLOR_BG, "terrain_fill": window.COLOR_TERRAIN_FILL, "terrain_line": window.COLOR_TERRAIN_LINE,
            "pad": window.COLOR_PAD, "region": window.COLOR_REGION, "hazard": canvas.COLOR_HAZARD,
            "zone": canvas.COLOR_ZONE, "bounds": window.COLOR_BOUNDS, "world_body": window.COLOR_WORLD_BODY,
            "flame": window.COLOR_FLAME, "hud": window.COLOR_HUD, "hud_box": canvas.COLOR_HUD_BOX,
            "hud_edge": canvas.COLOR_HUD_EDGE}
    assert P.COLORS == {k: css(v) for k, v in want.items()}


def test_phone_module_does_not_import_qt() -> None:
    code = ("import sys, evolab.app.export.phone as p; "
            "assert not any(m.startswith('PySide6') for m in sys.modules), sorted(m for m in sys.modules if 'PySide' in m)")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


# ---------------------------------------------------------------------------
# 자료 왕복 (T42)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stem", ["lander", "track_car", "hill_car", "jump", "glider", "wheels"])
def test_round_trip_within_1mm_and_1e4_rad(stem: str) -> None:
    payload = _assert_round_trip(_single(stem))
    assert payload["v"] == P.PHONE_FORMAT == 1
    assert payload["unit"] == [0.001, 0.0001]
    assert payload["layout"] == "overlay" and len(payload["worlds"]) == 1
    assert payload["text"] == S.PHONE_TEXT
    assert payload["app"] == __version__
    assert payload["title"] == S.PHONE_TITLE.format(env=_env_name(stem), label="지금 챔피언 (3세대)")
    assert payload["worlds"][0]["view"] == (None if _view(stem) is None else [P._n(v) for v in _view(stem)])


def test_spinning_wheels_and_long_runs_keep_angles_within_1e4_rad() -> None:
    """바퀴처럼 여러 바퀴 도는 각도(감기 전 수백 rad)와 큰 위치도 왕복한다."""
    rep = _long(_replay("hill_car"), 400, seed=5)
    tr = rep.trajectory
    poses = tr.poses.copy()
    poses[:, :, 2] = np.linspace(-900.0, 1200.0, poses.shape[0], dtype=np.float32)[:, None]
    poses[:, :, 0] += 1500.0
    rep = dataclasses.replace(rep, trajectory=dataclasses.replace(tr, poses=poses))
    _assert_round_trip(_overlay([(rep, "#ff7828", "a")], "hill_car"))


def test_thrusters_and_outputs_follow_the_canvas_rules() -> None:
    rep = _replay("lander", 4)
    payload = P.build_payload(_overlay([(rep, "#ff7828", "a")], "lander"))
    e = payload["entries"][0]
    ids = [b.id for b in rep.agent]
    want = [(ids.index(t.body), rep.channels.index(t.channel)) for t in rep.geometry.thrusters]
    assert [(t["b"], t["ch"]) for t in e["thrusters"]] == want and want
    for t, g in zip(e["thrusters"], rep.geometry.thrusters, strict=True):
        assert t["s"] == pytest.approx(min(1.0, g.max_force / 200.0 + 0.3), abs=1e-4)
        assert t["dir"] == pytest.approx(g.dir, abs=1e-5)
    assert e["nc"] == len(rep.channels) == 3


def test_overlay_entries_share_the_first_world_and_take_canvas_shades() -> None:
    from PySide6.QtGui import QColor

    a, b = _replay("lander", 1), _replay("lander", 2)
    job = _overlay([(a, "#508cff", "1세대 대표"), (b, "#ff7828", "지금 챔피언 (9세대)")], "lander", title="1세대 vs 지금")
    payload = _assert_round_trip(job)
    assert [e["world"] for e in payload["entries"]] == [0, 0]
    assert len(payload["worlds"]) == 1
    for e in payload["entries"]:
        c = QColor(e["color"])
        assert e["leg"] == c.darker(125).name() and e["edge"] == c.lighter(160).name()
        assert e["rank"] is None


def test_movers_are_pose_at_at_the_frame_times() -> None:
    """v2 움직이는 세계 물체(점프 게임 장애물 줄): 세계 표본 k의 자세 = pose_at(움직임, k × 프레임 간격) 1mm·1e-4rad 안,
    표본 시각 = 칸 프레임 시각, 가장 긴 재생 + 꼬리까지 덮음."""
    rep = _replay("jump", 7)
    movers = [b for b in rep.geometry.bodies if b.motion is not None and not b.moving]
    assert movers
    payload = P.build_payload(_overlay([(rep, "#ff7828", "a")], "jump"))
    dec = P.decode_payload(payload)
    w = dec["worlds"][0]
    e = dec["entries"][0]
    assert w["mstep"] == e["step"]
    assert (w["mframes"] - 1) * w["mstep"] >= e["last"] + P.TAIL - 1e-9
    frame_times = rep.trajectory.times()[:-1]
    assert np.array_equal(np.arange(w["mframes"])[: len(frame_times)] * w["mstep"], frame_times)
    assert len(w["movers"]) == len(movers)
    for mv, body in zip(w["movers"], movers, strict=True):
        want = np.array([pose_at(body.motion, k * w["mstep"]) for k in range(w["mframes"])])
        assert np.abs(mv["poses"][:, :2] - want[:, :2]).max() <= 0.5e-3 + 1e-9
        assert np.abs(_wrap(mv["poses"][:, 2] - want[:, 2])).max() <= 0.5e-4 + 1e-9
        assert len(mv["shapes"]) == len(body.shapes)
    # 움직이는 세계 물체는 정적 모양에 들어가지 않는다
    static = [b for b in rep.geometry.bodies if not b.moving and b.motion is None]
    assert len(payload["worlds"][0]["static"]) == sum(len(b.shapes) for b in static)


def test_absent_bodies_of_v21_structure_are_left_out() -> None:
    """바퀴 고르는 언덕차 영 유전자: 없는 물체(v2.1)는 칸 물체·추진기에 없다(Replay가 이미 뺀 그대로)."""
    spec = race_model.compile_cached(_text("wheels"))
    layout = layout_for(spec)
    g = np.zeros((1, layout.dim))
    ep = Episode(spec, layout.body_values(g)[0] if layout.n_body else None, seeds.display_trial())
    assert ep.absent
    rep = _replay("wheels")
    payload = _assert_round_trip(_overlay([(rep, "#ff7828", "a")], "wheels"))
    e = payload["entries"][0]
    assert e["nb"] == len(rep.agent) == len(ep.moving_bodies)
    assert not ({b.id for b in rep.agent} & ep.absent)
    assert all(t["b"] < e["nb"] for t in e["thrusters"])


def test_lanes_get_their_own_worlds_and_equal_worlds_merge() -> None:
    lanes = [(_replay(stem, 10 + k), LANE_COLORS[k], f"칸 {k + 1}") for k, stem in
             enumerate(["lander", "jump", "lander", "hill_car"])]
    job = J.job_from_race(lanes, race_model.lane_grid(4)[1], "html", size="medium", camera="follow", title="경주")
    payload = _assert_round_trip(job)
    assert payload["layout"] == "lanes"
    assert [e["world"] for e in payload["entries"]] == [0, 1, 0, 2]
    assert len(payload["worlds"]) == 3
    assert len(payload["worlds"][1]["movers"]) >= 1  # 점프 게임 칸의 세계는 자기 장애물을 가진다
    assert all(e["rank"] is None for e in payload["entries"])


def test_race_ranks_go_to_the_entries() -> None:
    text = _text("lander")

    def lane(k: int) -> SimpleNamespace:
        spec = race_model.compile_cached(text)
        entry = RaceEntry(label=f"칸 {k + 1}", sub="", color=LANE_COLORS[k], env_text=text,
                          env_hash=race_model.env_hash_of(text), genome=np.zeros(layout_for(spec).dim), generation=1,
                          run=None)
        return SimpleNamespace(replay=_replay("lander", 30 + k), entry=entry)

    lanes = [lane(k) for k in range(3)]
    job = J.job_from_race(lanes, race_model.lane_grid(3)[1], "html", size="medium", camera="follow", title="경주")
    assert job.ranks is not None
    payload = P.build_payload(job)
    assert [e["rank"] for e in payload["entries"]] == list(job.ranks)
    assert len(payload["worlds"]) == 1
    assert payload["title"] == S.PHONE_TITLE.format(env=_env_name("lander"), label="경주")


def _jump_glider_lanes() -> ExportJob:
    """경주 칸 둘: 점프 게임(움직이는 장애물, 영 유전자는 곧 끝남)과 글라이더(영 유전자로 오래 감)."""
    lanes = [(_replay("jump"), LANE_COLORS[0], "점프"), (_replay("glider"), LANE_COLORS[1], "글라이더")]
    return J.job_from_race(lanes, race_model.lane_grid(2)[1], "html", size="medium", camera="follow", title="경주")


def test_lane_movers_keep_moving_until_the_whole_file_ends() -> None:
    """경주 칸: 짧은 칸(점프 게임)의 장애물도 파일 전체 길이(가장 긴 칸 + 꼬리 = 재생기 total)까지 pose_at 표본을 가진다
    (경주 창은 짧은 칸의 장애물도 전체 시각으로 움직인다: 칸 자기 길이까지만 담아 장애물이 멈추던 v3.1 검토 결함)."""
    job = _jump_glider_lanes()
    payload = P.build_payload(job)
    dec = P.decode_payload(payload)
    entries = dec["entries"]
    end = max(e["last"] for e in entries) + P.TAIL
    assert entries[0]["last"] + P.TAIL < end - 5.0  # 점프 칸이 몇 초 넘게 먼저 끝난다(이 시험이 뜻을 가지려면)
    assert len(dec["worlds"]) == 2
    jump_world = dec["worlds"][entries[0]["world"]]
    assert jump_world["movers"]
    for w in dec["worlds"]:
        assert (w["mframes"] - 1) * w["mstep"] >= end - 1e-9
    movers = [b for b in job.items[0].replay.geometry.bodies if b.motion is not None and not b.moving]
    for mv, body in zip(jump_world["movers"], movers, strict=True):
        want = np.array([pose_at(body.motion, k * jump_world["mstep"]) for k in range(jump_world["mframes"])])
        assert np.abs(mv["poses"][:, :2] - want[:, :2]).max() <= 0.5e-3 + 1e-9
    # SVG 끝 장면도 같은 끝 시각의 장애물 자리(마지막 표본)를 쓴다
    assert P.svg_poster(payload).count("<g transform=") >= len(movers)


def _env_with(stem: str, change: Any) -> str:
    data = json.loads(_text(stem))
    change(data)
    return json.dumps(data, ensure_ascii=False)


def _replay_text(text: str) -> Replay:
    spec = race_model.compile_cached(text)
    layout = layout_for(spec)
    return replay(spec, layout, np.zeros(layout.dim), seeds.display_trial())


def test_lanes_with_different_regions_or_obstacle_shapes_get_their_own_worlds() -> None:
    """세계를 합치는 열쇠는 세계 JSON이 그리는 것 모두: 영역만 다른 두 환경(언덕차 ± 목표 영역), 장애물 모양만 다른 두 환경
    (점프 게임 생성기 높이)은 다른 세계(같은 환경은 여전히 하나로: test_lanes_get_their_own_worlds_and_equal_worlds_merge)."""
    plain = _text("hill_car")
    goal = _env_with("hill_car", lambda d: d.setdefault("regions", []).append(
        {"id": "goal_zone", "kind": "goal", "shape": {"kind": "box", "w": 6.0, "h": 6.0}, "pos": [150.0, 10.0]}))
    a, b = _replay_text(plain), _replay_text(goal)
    assert len(b.geometry.regions) == len(a.geometry.regions) + 1
    for first, second in ((a, b), (b, a)):
        job = J.job_from_race([(first, LANE_COLORS[0], "A"), (second, LANE_COLORS[1], "B")],
                              race_model.lane_grid(2)[1], "html", size="medium", camera="follow", title="경주")
        payload = P.build_payload(job)
        assert [e["world"] for e in payload["entries"]] == [0, 1]
        regions = [len(it.replay.geometry.regions) for it in job.items]
        assert [len(payload["worlds"][e["world"]]["regions"]) for e in payload["entries"]] == regions

    def taller(d: dict[str, Any]) -> None:
        d["generators"][0]["height"] = [1.0, 2.0]

    a, b = _replay_text(_text("jump")), _replay_text(_env_with("jump", taller))

    def mover_shapes(rep: Replay) -> list[Any]:
        return [P._shape_json(s) for m in rep.geometry.bodies if m.motion is not None and not m.moving
                for s in m.shapes]

    assert mover_shapes(a) != mover_shapes(b)
    job = J.job_from_race([(a, LANE_COLORS[0], "A"), (b, LANE_COLORS[1], "B")], race_model.lane_grid(2)[1], "html",
                          size="medium", camera="follow", title="경주")
    payload = P.build_payload(job)
    assert [e["world"] for e in payload["entries"]] == [0, 1]
    for e, it in zip(payload["entries"], job.items, strict=True):
        drawn = [s for mv in payload["worlds"][e["world"]]["movers"] for s in mv["shapes"]]
        assert drawn == mover_shapes(it.replay)


def test_job_from_run_versus_round_trips(tmp_path: Path) -> None:
    run = tmp_path / "v11_run_restart"
    shutil.copytree(DATA / "v11_run_restart" / "run", run)
    job = J.job_from_run(run, "html", generation=None, what="versus", size="medium", camera="fixed", encoder="auto")
    payload = _assert_round_trip(job)
    assert [e["color"] for e in payload["entries"]] == [J.OLD_COLOR, J.NOW_COLOR]
    assert payload["lines"] == [f"{run.name} · {job.meta[J.META_DATE]}"]


# ---------------------------------------------------------------------------
# HTML (T42)
# ---------------------------------------------------------------------------
def test_html_structure_has_csp_and_no_outside_requests(tmp_path: Path) -> None:
    job = _single("jump")
    out = P.write_phone_html(tmp_path / "점프 게임.html", job, None)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>") and '<meta charset="utf-8">' in text[:200]
    _html_checks(text)
    assert P.read_phone_html(out) == json.loads(json.dumps(P.build_payload(job)))
    title = S.PHONE_TITLE.format(env=_env_name("jump"), label="지금 챔피언 (3세대)")
    assert f"<title>{title}</title>" in text and f'<h1 id="title">{title}</h1>' in text
    assert S.PHONE_TEXT["noscript"] in text
    assert S.PHONE_TEXT["footer"].format(version=__version__) in text
    assert '<img id="poster" alt="" src="data:image/svg+xml;base64,' in text
    assert not list(tmp_path.glob("*.tmp"))


def test_hostile_names_do_not_break_the_page() -> None:
    evil = '</script><b>x</b><!-- @@DATA@@ & "q"'
    rep = _replay("lander", 2)
    job = _overlay([(rep, "#ff7828", evil)], "lander", title=evil, env=evil)
    text = P.phone_html(job)
    _html_checks(text, names_hold_at=True)
    assert "<b>x</b>" not in text and "<!-- @@" not in text
    data = P.payload_from_html(text)
    assert data["entries"][0]["label"] == evil
    assert data["title"] == S.PHONE_TITLE.format(env=evil, label=evil)
    # 끝 장면 SVG도 이름을 담지 않는다(자료로 그린 그림)
    assert text.count("<title>") == 1


def test_svg_poster_is_well_formed_and_draws_each_lane() -> None:
    lanes = [(_replay(stem, 40 + k), LANE_COLORS[k], f"칸 {k + 1}") for k, stem in
             enumerate(["lander", "jump", "track_car", "hill_car", "glider", "wheels"])]
    job = J.job_from_race(lanes, race_model.lane_grid(6)[1], "html", size="medium", camera="follow", title="경주")
    root = ET.fromstring(P.svg_poster(P.build_payload(job)))
    ns = "{http://www.w3.org/2000/svg}"
    assert root.tag == f"{ns}svg" and root.get("width") == "640" and root.get("height") == "400"
    assert len(root.findall(f"{ns}svg")) == 6
    one = ET.fromstring(P.svg_poster(P.build_payload(_single("jump"))))
    assert len(one.findall(f"{ns}svg")) == 1
    assert one.findall(f".//{ns}g")  # 장애물 줄과 개체 물체


def test_png_poster_from_a_canvas_frame_has_at_most_64_colors(qtbot: Any, tmp_path: Path) -> None:
    import base64
    import io

    from PIL import Image

    from evolab.app.export import video

    rep = _replay("hill_car", 3)
    frames = video.render_frames([(rep, "#ff7828", "a")], size=J.GIF_SIZE, fps=5)
    out = P.write_phone_html(tmp_path / "언덕.html", _overlay([(rep, "#ff7828", "a")], "hill_car"), frames[-1])
    text = out.read_text(encoding="utf-8")
    m = re.search(r'<img id="poster" alt="" src="data:image/png;base64,([A-Za-z0-9+/=]+)"', text)
    assert m is not None
    png = base64.b64decode(m.group(1))
    im = Image.open(io.BytesIO(png))
    assert im.format == "PNG" and im.size == J.GIF_SIZE and im.mode == "P"
    assert len(im.getcolors(256) or []) <= P.POSTER_COLORS
    assert len(png) < 20_000
    _html_checks(text)


def test_too_big_replays_are_refused_without_touching_the_file(tmp_path: Path) -> None:
    rep = _replay("lander", 1)
    ok = _long(rep, P.PHONE_MAX_FRAMES + 1)  # 120초 ÷ 2스텝 + 0초 프레임
    P.build_payload(_overlay([(ok, "#ff7828", "a")], "lander"))
    big = _long(rep, P.PHONE_MAX_FRAMES + 2)
    target = tmp_path / "a.html"
    target.write_text("old", encoding="utf-8")
    with pytest.raises(P.PhoneTooBig) as err:
        P.write_phone_html(target, _overlay([(big, "#ff7828", "a")], "lander"), None)
    assert str(err.value) == S.PHONE_TOO_BIG.format(frames=P.PHONE_MAX_FRAMES + 2, limit=P.PHONE_MAX_FRAMES + 1)
    assert target.read_text(encoding="utf-8") == "old" and not list(tmp_path.glob("*.tmp"))
    # 물체 33개
    body = rep.agent[0]
    many = dataclasses.replace(rep, agent=[body] * (P.PHONE_MAX_BODIES + 1), trajectory=Trajectory(
        np.zeros((10, P.PHONE_MAX_BODIES + 1, 3), np.float32), rep.trajectory.outputs[:10], 2, rep.trajectory.dt, 18))
    with pytest.raises(P.PhoneTooBig):
        P.build_payload(_overlay([(many, "#ff7828", "a")], "lander"))


def test_exploded_frames_become_zero_and_json_stays_strict() -> None:
    rep = _replay("lander", 1)
    tr = rep.trajectory
    poses = tr.poses.copy()
    poses[5, 0, :] = (np.nan, np.inf, -np.inf)
    outs = tr.outputs.copy()
    outs[3, 0] = np.nan
    res = dataclasses.replace(rep.result, score=float("nan"))
    rep = dataclasses.replace(rep, trajectory=dataclasses.replace(tr, poses=poses, outputs=outs), result=res)
    payload = P.build_payload(_overlay([(rep, "#ff7828", "a")], "lander"))
    json.dumps(payload, allow_nan=False)
    dec = P.decode_payload(payload)
    assert np.array_equal(dec["entries"][0]["poses"][5, 0], [0.0, 0.0, 0.0])
    assert dec["entries"][0]["out"][3, 0] == 0.0
    assert payload["entries"][0]["score"] is None
    assert payload["entries"][0]["box"] is not None and all(math.isfinite(v) for v in payload["entries"][0]["box"])


def test_fit_box_rule() -> None:
    assert P.fit_box([[0, 0, 100, 10]], None, (-1, -1, 1, 1)) == pytest.approx((-8.0, -0.8, 108.0, 10.8))
    assert P.fit_box([[0, 0, 100, 2]], None, (-1, -1, 1, 1)) == pytest.approx((-8.0, -3.0, 108.0, 5.0))
    assert P.fit_box([[0, 0, 1, 1]], [5, 5, 6, 6], (-1, -1, 1, 1)) == pytest.approx((-1.0, -1.0, 7.0, 7.0))
    assert P.fit_box([None], None, (0, 0, 20, 10)) == pytest.approx((-1.6, -0.8, 21.6, 10.8))


# ---------------------------------------------------------------------------
# 크기 (T42, M9 기준)
# ---------------------------------------------------------------------------
def _size(job: ExportJob) -> int:
    return len(P.phone_html(job).encode("utf-8"))  # 끝 장면 SVG(PNG보다 큰 쪽)로 잰다


def test_sizes_hill_car_full_time_and_lander_versus_under_300kb(tmp_path: Path) -> None:
    hill = _long(_replay("hill_car", 3), _max_frames("hill_car"))
    assert _size(_overlay([(hill, "#ff7828", "a")], "hill_car")) <= 300_000
    run = tmp_path / "v10_run_ga"
    shutil.copytree(DATA / "v10_run_ga" / "run", run)
    versus = J.job_from_run(run, "html", generation=None, what="versus", size="medium", camera="fixed", encoder="auto")
    assert _size(versus) <= 300_000
    a, b = _replay("lander", 1), _replay("lander", 2)
    lander = _long(a, _max_frames("lander")), _long(b, _max_frames("lander"), seed=1)
    assert _size(_overlay([(lander[0], "#508cff", "1"), (lander[1], "#ff7828", "2")], "lander")) <= 300_000
    glider = _long(_replay("glider", 3), _max_frames("glider"))  # 120초: 3,601프레임(상한 안)
    assert glider.trajectory.frames == P.PHONE_MAX_FRAMES + 1
    assert _size(_overlay([(glider, "#ff7828", "a")], "glider")) <= 300_000


def test_size_race_of_six_full_time_lanes_under_1mb() -> None:
    stems = ["lander", "track_car", "hill_car", "jump", "glider", "wheels"]
    mixed = [(_long(_replay(s, 50 + k), _max_frames(s), seed=k), LANE_COLORS[k], f"칸 {k + 1}")
             for k, s in enumerate(stems)]
    job = J.job_from_race(mixed, race_model.lane_grid(6)[1], "html", size="medium", camera="follow", title="경주")
    assert _size(job) <= 1_000_000
    hills = [(_long(_replay("hill_car", 60 + k), _max_frames("hill_car"), seed=k), LANE_COLORS[k], f"칸 {k + 1}")
             for k in range(6)]
    job = J.job_from_race(hills, race_model.lane_grid(6)[1], "html", size="medium", camera="follow", title="경주")
    assert len(P.build_payload(job)["worlds"]) == 1
    assert _size(job) <= 1_000_000


# ---------------------------------------------------------------------------
# 헤드리스 크롬 (T43)
# ---------------------------------------------------------------------------
_BROWSERS = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "msedge", "microsoft-edge")

_RUNNER = r"""
const { chromium } = require('playwright');
(async () => {
  const exe = process.argv[2], files = process.argv.slice(3), out = [];
  const browser = await chromium.launch({ executablePath: exe, args: ['--no-sandbox', '--disable-gpu'] });
  try {
    for (const f of files) {
      for (const js of [true, false]) {
        const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true,
                                               hasTouch: true, javaScriptEnabled: js, offline: true });
        const page = await ctx.newPage();
        const errors = [], requests = [];
        page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errors.push(m.text()); });
        page.on('pageerror', e => errors.push(String(e)));
        page.on('request', r => { const u = r.url(); if (!u.startsWith('file:') && !u.startsWith('data:')) requests.push(u); });
        await page.goto(f, { waitUntil: 'load' });
        await page.waitForTimeout(js ? 900 : 200);
        const r = { file: f, js, errors, requests, poster: await page.locator('#poster').isVisible(),
                    player: await page.locator('#player').isVisible(), canvases: await page.locator('canvas').count() };
        if (js) {
          Object.assign(r, await page.evaluate(() => {
            const cv = document.querySelector('#stage canvas'), P = window.EVOLAB_PLAYER;
            const px = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
            let ink = 0;
            for (let i = 0; i < px.length; i += 4 * 5) {
              if (Math.abs(px[i] - 11) + Math.abs(px[i + 1] - 14) + Math.abs(px[i + 2] - 20) > 24) ink++;
            }
            const before = P.mode;
            document.getElementById('cam').click();
            const after = P.mode;
            P.play(false); P.seek(P.total);
            const card = document.getElementById('card');
            return { width: cv.width, height: cv.height, cssWidth: cv.clientWidth, cssHeight: cv.clientHeight, ink,
                     t: P.t, drawn: P.drawn, views: P.views, total: P.total, modes: [before, after],
                     card: !card.hidden, cardRows: card.querySelectorAll('.who').length,
                     legend: document.querySelectorAll('#legend button').length };
          }));
        }
        out.push(r);
        await ctx.close();
      }
    }
  } finally { await browser.close(); }
  process.stdout.write(JSON.stringify(out));
})().catch(e => { console.error(String(e)); process.exit(3); });
"""


def _browser() -> tuple[str, str] | None:
    """(node, 브라우저 실행 파일) 또는 None: node, require("playwright"), 크롬·엣지가 모두 있어야 한다."""
    node = shutil.which("node")
    exe = os.environ.get("EVOLAB_TEST_CHROME") or next((p for p in map(shutil.which, _BROWSERS) if p), None)
    if not node or not exe:
        return None
    try:
        ok = subprocess.run([node, "-e", "require('playwright')"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (node, exe) if ok else None


def test_headless_chrome_plays_the_file_offline(tmp_path: Path) -> None:
    found = _browser()
    if found is None:
        pytest.skip("node + playwright + Chrome/Edge are not all available")
    node, exe = found
    versus = _overlay([(_replay("lander", 1), "#508cff", "1세대 대표"), (_replay("lander", 2), "#ff7828", "지금 챔피언")],
                      "lander", title="1세대 vs 지금")
    lanes = [(_replay(stem, 20 + k), LANE_COLORS[k], f"칸 {k + 1}") for k, stem in
             enumerate(["lander", "jump", "hill_car", "glider"])]
    race = J.job_from_race(lanes, race_model.lane_grid(4)[1], "html", size="medium", camera="follow", title="경주")
    files = [P.write_phone_html(tmp_path / "1세대 vs 지금.html", versus, None),
             P.write_phone_html(tmp_path / "점프.html", _single("jump"), None),
             P.write_phone_html(tmp_path / "경주.html", race, None)]
    script = tmp_path / "runner.js"
    script.write_text(_RUNNER, encoding="utf-8")
    uris = [f.resolve().as_uri() for f in files]
    proc = subprocess.run([node, str(script), exe, *uris], capture_output=True, text=True, timeout=240,
                          encoding="utf-8")
    assert proc.returncode == 0, proc.stderr[-2000:]
    results = json.loads(proc.stdout)
    assert len(results) == 2 * len(files)
    expect = {uris[0]: (1, 2, 2), uris[1]: (1, 1, 0), uris[2]: (4, 4, 0)}  # (화면 칸, 점수 카드 줄, 범례 단추)
    for r in results:
        assert r["errors"] == [] and r["requests"] == [], r
        if not r["js"]:
            assert r["poster"] and not r["player"] and r["canvases"] == 0, r
            continue
        views, rows, legend = expect[r["file"]]
        assert not r["poster"] and r["player"] and r["canvases"] == 1, r
        assert r["width"] <= 2 * r["cssWidth"] + 1 and r["height"] <= 2 * r["cssHeight"] + 1, r
        assert r["width"] >= r["cssWidth"], r  # 화소 비율 3 화면에서 상한 2까지는 씀
        assert r["ink"] > 500, r  # 배경이 아닌 화소(세계·개체)가 그려졌다
        assert r["drawn"] > 2 and r["views"] == views, r
        assert r["modes"][0] != r["modes"][1], r
        assert r["card"] and r["cardRows"] == rows and r["legend"] == legend, r


_PROBE_RUNNER = r"""
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const exe = process.argv[2], specs = JSON.parse(fs.readFileSync(process.argv[3], 'utf8')), out = [];
  const browser = await chromium.launch({ executablePath: exe, args: ['--no-sandbox', '--disable-gpu'] });
  try {
    for (const spec of specs) {
      const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, offline: true });
      const page = await ctx.newPage();
      const errors = [];
      page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errors.push(m.text()); });
      page.on('pageerror', e => errors.push(String(e)));
      await page.goto(spec.file, { waitUntil: 'load' });
      await page.waitForTimeout(300);
      const r = await page.evaluate((s) => {
        const P = window.EVOLAB_PLAYER;
        P.play(false);
        const poses = s.poses.map(q => P.pose(q[0], q[1], q[2]));
        const movers = s.movers.map(q => P.mover(q[0], q[1], q[2]));
        P.seek(P.total);
        const card = Array.from(document.querySelectorAll('#card .who')).map(n => n.textContent);
        return { total: P.total, poses, movers, card };
      }, spec);
      r.errors = errors;
      out.push(r);
      await ctx.close();
    }
  } finally { await browser.close(); }
  process.stdout.write(JSON.stringify(out));
})().catch(e => { console.error(String(e)); process.exit(3); });
"""


def _short_turn(a: float, b: float) -> float:
    """a에서 b로 짧은 쪽으로 도는 각도 차."""
    return float(_wrap(np.array([b - a]))[0])


def test_headless_chrome_player_poses_and_race_ranks(tmp_path: Path) -> None:
    """T43 덧붙임: 재생기(JS)가 개체를 궤적 자리에, 움직이는 장애물을 pose_at 자리에 그리고(경주 칸에서 먼저 끝난 칸의
    장애물도 끝까지 움직임), 「한 화면에 겹쳐서」 경주 파일의 끝 카드가 순위 차례로 「n등 · 이름」을 보인다."""
    found = _browser()
    if found is None:
        pytest.skip("node + playwright + Chrome/Edge are not all available")
    node, exe = found
    # 1) 경주 칸 [점프 게임, 글라이더]: 개체 자세(프레임 시각·프레임 사이 가운데)와 장애물 자세
    lanes = _jump_glider_lanes()
    payload = P.build_payload(lanes)
    glider = lanes.items[1].replay.trajectory
    times = glider.times()
    k = min(200, glider.frames - 3)
    pose_q: list[list[float]] = []
    pose_want: list[tuple[float, float, float]] = []
    for b in range(glider.poses.shape[1]):
        p0 = glider.poses[k, b].astype(np.float64)
        p1 = glider.poses[k + 1, b].astype(np.float64)
        pose_q.append([1, b, float(times[k])])
        pose_want.append((p0[0], p0[1], p0[2]))
        pose_q.append([1, b, float((times[k] + times[k + 1]) / 2)])
        pose_want.append(((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, p0[2] + _short_turn(p0[2], p1[2]) / 2))
    w_jump = payload["entries"][0]["world"]
    world = payload["worlds"][w_jump]
    step = float(world["mstep"])
    total = max(e["last"] for e in payload["entries"]) + P.TAIL
    body = next(b for b in lanes.items[0].replay.geometry.bodies if b.motion is not None and not b.moving)
    k_end = math.floor((total - 0.1) / step)
    assert k_end * step > lanes.items[0].replay.trajectory.duration + 5.0  # 점프 칸이 끝난 뒤 한참
    mover_q: list[list[float]] = []
    mover_want: list[tuple[float, float, float]] = []
    for kk in (30, k_end):
        mover_q.append([w_jump, 0, kk * step])
        mover_want.append(tuple(pose_at(body.motion, kk * step)))
        a, c = np.array(pose_at(body.motion, kk * step)), np.array(pose_at(body.motion, (kk - 1) * step))
        mover_q.append([w_jump, 0, (kk - 0.5) * step])  # 표본 사이: 재생기는 선형으로 잇는다
        mover_want.append(((a[0] + c[0]) / 2, (a[1] + c[1]) / 2, c[2] + _short_turn(c[2], a[2]) / 2))
    # 2) 경주 「한 화면에 겹쳐서」(대화 상자 build_job과 같은 길): 순위는 칸 작업 그대로
    text = _text("lander")

    def lane(k: int) -> SimpleNamespace:
        spec = race_model.compile_cached(text)
        entry = RaceEntry(label=f"칸 {k + 1}", sub="", color=LANE_COLORS[k], env_text=text,
                          env_hash=race_model.env_hash_of(text), genome=np.zeros(layout_for(spec).dim), generation=1,
                          run=None)
        return SimpleNamespace(replay=_replay("lander", 40 + k), entry=entry)

    race = J.job_from_race([lane(k) for k in range(3)], race_model.lane_grid(3)[1], "html", size="medium",
                           camera="follow", title="경주")
    assert race.ranks is not None and sorted(race.ranks) == [1, 2, 3]
    merged = J.job_from_replays([(it.replay, it.color, it.label) for it in race.items], "html", size="medium",
                                camera="follow", title="경주", meta=race.meta, view=race.items[0].view)
    overlay = dataclasses.replace(merged, ranks=race.ranks)
    order = sorted(range(3), key=lambda i: (race.ranks[i], i))
    card_want = [f"{S.PHONE_TEXT['rank'].format(n=race.ranks[i])} · {race.items[i].label}" for i in order]
    files = [P.write_phone_html(tmp_path / "경주 칸.html", lanes, None),
             P.write_phone_html(tmp_path / "경주 겹쳐서.html", overlay, None)]
    specs = [{"file": files[0].resolve().as_uri(), "poses": pose_q, "movers": mover_q},
             {"file": files[1].resolve().as_uri(), "poses": [], "movers": []}]
    (tmp_path / "specs.json").write_text(json.dumps(specs), encoding="utf-8")
    script = tmp_path / "probe.js"
    script.write_text(_PROBE_RUNNER, encoding="utf-8")
    proc = subprocess.run([node, str(script), exe, str(tmp_path / "specs.json")], capture_output=True, text=True,
                          timeout=240, encoding="utf-8", check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    got_lanes, got_overlay = json.loads(proc.stdout)
    assert got_lanes["errors"] == [] and got_overlay["errors"] == []
    assert got_lanes["total"] == pytest.approx(total, abs=1e-9)
    for q, want, got in zip(pose_q + mover_q, pose_want + mover_want, got_lanes["poses"] + got_lanes["movers"],
                            strict=True):
        assert abs(got[0] - want[0]) <= 1e-3 and abs(got[1] - want[1]) <= 1e-3, (q, want, got)
        assert abs(_short_turn(want[2], got[2])) <= 1e-4, (q, want, got)
    assert got_overlay["card"] == card_want

"""v3.1 0' 내보내기 작업 (v3 설계 8.1 T32, 2.6, 7.3 「app/export/job.py」).

job_from_replays(실험실 지금 재생)·job_from_race(나란히 경주)·job_from_run(명령줄, 다시 시뮬레이션)이 맞는 칸·크기·fps·색·
이름·기본 보기·순위·meta를 만들고, ExportJob이 재생을 담은 채 pickle 왕복한다(같은 프로세스, spawn 자식). 명령줄 작업의
재생은 실험실 「1세대 대표와 지금 챔피언」과 같은 규칙(전시 시드, 그 세대 환경, run.json의 유전자 배치)으로 직접 다시
시뮬레이션한 것과 비트 단위로 같다.

이 모듈은 맨 위에서 Qt를 불러오지 않는다(spawn 자식이 이 모듈을 다시 불러온다).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import multiprocessing
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from evolab import __version__, paths
from evolab.app import strings as S
from evolab.app.export import job as J
from evolab.app.export.job import ExportItem, ExportJob
from evolab.app.race import model as race_model
from evolab.app.race.model import RaceEntry, lane_grid
from evolab.engine import seeds
from evolab.engine.replay import Replay, replay
from evolab.policy.genome import GenomeLayout, layout_for
from evolab.runs.store import RunStore, RunStoreError
from evolab.spec.io import compile_env

DATA = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------
def _builtin_text(stem: str) -> str:
    path = next(p for p in paths.builtin_envs() if p.stem == stem)
    return path.read_text(encoding="utf-8-sig")


def _replay_of(env_text: str, seed: int | None = None) -> Replay:
    """내장 환경 하나를 전시 시드로 다시 시뮬레이션한 재생. seed가 있으면 그 시드의 작은 무작위 유전자(점수가 칸마다 다르게)."""
    spec = race_model.compile_cached(env_text)
    layout = layout_for(spec)
    genome = (np.zeros(layout.dim) if seed is None
              else np.random.default_rng(seed).normal(0.0, 0.6, layout.dim))
    return replay(spec, layout, genome, seeds.display_trial())


def _entry(env_text: str, color: str, label: str, run: Path | None = None) -> RaceEntry:
    spec = race_model.compile_cached(env_text)
    return RaceEntry(label=label, sub="", color=color, env_text=env_text, env_hash=race_model.env_hash_of(env_text),
                     genome=np.zeros(layout_for(spec).dim), generation=1, run=run)


def _lane(env_text: str, k: int, run: Path | None = None) -> SimpleNamespace:
    """경주 창 칸(_Lane)처럼 .replay·.entry를 가진 것."""
    return SimpleNamespace(replay=_replay_of(env_text, seed=100 + k),
                           entry=_entry(env_text, f"#{(40 * k) % 256:02X}80C0", f"칸 {k + 1}", run))


def _deep_equal(a: Any, b: Any, where: str = "") -> list[str]:
    """두 객체가 값으로 같은지(데이터 클래스·배열·목록·사전을 따라 내려감). 다른 곳의 경로 목록."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        ok = (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
              and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
        return [] if ok else [where]
    if type(a) is not type(b):
        return [f"{where}: {type(a).__name__} != {type(b).__name__}"]
    if dataclasses.is_dataclass(a):
        return [d for f in dataclasses.fields(a) for d in _deep_equal(getattr(a, f.name), getattr(b, f.name),
                                                                      f"{where}.{f.name}")]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{where}: len {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b, strict=True)) for d in _deep_equal(x, y, f"{where}[{i}]")]
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return [f"{where}: keys"]
        return [d for k in a for d in _deep_equal(a[k], b[k], f"{where}[{k!r}]")]
    if isinstance(a, float) and math.isnan(a):
        return [] if math.isnan(b) else [where]
    if hasattr(a, "__dict__") and not isinstance(a, type):
        return _deep_equal(vars(a), vars(b), where)
    return [] if a == b else [f"{where}: {a!r} != {b!r}"]


def _same_job(a: ExportJob, b: ExportJob) -> list[str]:
    return _deep_equal(a, b, "job")


@pytest.fixture()
def runs(tmp_path: Path) -> dict[str, Path]:
    """v1.0(유전 알고리즘 4세대, 3세대 챔피언 없음)·v1.1(CMA-ES 12세대) 실행 사본(원본은 고치지 않는다)."""
    out = {}
    for name in ("v10_run_ga", "v11_run_restart"):
        out[name] = tmp_path / name
        shutil.copytree(DATA / name / "run", out[name])
    return out


# ---------------------------------------------------------------------------
# 지금 값과 같은 상수
# ---------------------------------------------------------------------------
def test_constants_match_current_gif_lab_and_race_values() -> None:
    from evolab.app.export import video
    from evolab.app.lab.view import COLOR_NOW, COLOR_OLD

    assert J.GIF_SIZE == video.DEFAULT_SIZE and J.GIF_FPS == video.DEFAULT_FPS and J.GIF_PANE == video.RACE_PANE
    assert J.OLD_COLOR == COLOR_OLD.name() and J.NOW_COLOR == COLOR_NOW.name()
    assert J.MAX_ITEMS == race_model.MAX_LANES
    assert J.MP4_SIZES == {"small": (640, 360), "medium": (1280, 720)} and J.MP4_FPS == 30  # 설계 D54, 2.7
    assert J.DEFAULT_SIZE_NAME == "medium"
    assert set(J.SUFFIXES) == set(J.FORMATS)
    assert [J.format_for_path(p) for p in ("a.MP4", "b.gif", "c.Html", "d.htm", "e")] == ["mp4", "gif", "html", None,
                                                                                         None]


# ---------------------------------------------------------------------------
# job_from_replays (실험실)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("fmt", "size", "want"), [
    ("mp4", "small", (640, 360, 30)), ("mp4", "medium", (1280, 720, 30)),
    ("gif", "small", (640, 400, 20)), ("gif", "medium", (640, 400, 20)),
    ("html", "medium", (640, 400, 20)),
])
def test_job_from_replays_cells_size_fps(fmt: str, size: str, want: tuple[int, int, int]) -> None:
    from PySide6.QtGui import QColor

    text = _builtin_text("lander")
    old, now = _replay_of(text), _replay_of(text, seed=1)
    view = (-10, -2, 10, 12)
    job = J.job_from_replays([(old, QColor(80, 140, 255), "1세대 대표"), (now, "#FF7828", "지금 챔피언 (9세대)")], fmt,
                             size=size, camera="follow", title="1세대 대표 vs 지금 챔피언 (9세대)",
                             meta={J.META_RUN: "실행 A", J.META_GENERATION: 9}, view=view)
    assert (job.layout, job.fmt, job.camera, job.encoder, job.ranks) == ("overlay", fmt, "follow", "auto", None)
    assert (*job.size, job.fps) == want
    assert job.title == "1세대 대표 vs 지금 챔피언 (9세대)"
    assert isinstance(job.items, tuple) and [it.replay for it in job.items] == [old, now]  # 같은 객체(사본 아님)
    assert job.items[0].replay is old and job.items[1].replay is now
    assert [(it.color, it.label) for it in job.items] == [("#508cff", "1세대 대표"), ("#ff7828", "지금 챔피언 (9세대)")]
    assert all(it.view == (-10.0, -2.0, 10.0, 12.0) for it in job.items)  # 겹쳐 보기는 한 세계: 모든 칸이 같은 보기
    assert job.meta == {J.META_RUN: "실행 A", J.META_GENERATION: "9", J.META_VERSION: __version__,
                        J.META_DATE: job.meta[J.META_DATE]}
    assert len(job.meta[J.META_DATE]) == 10 and job.meta[J.META_DATE][4] == "-"
    assert J.lane_layout(job) == (1, 1, *job.size)
    assert job.duration == max(old.trajectory.duration, now.trajectory.duration)


def test_job_from_replays_without_view_and_bad_arguments() -> None:
    rep = _replay_of(_builtin_text("hill_car"))
    job = J.job_from_replays([(rep, "#ffffff", "x")], "gif", size="medium", camera="fixed", title="", meta={})
    assert job.items[0].view is None and job.meta[J.META_VERSION] == __version__
    for kwargs in ({"fmt": "avi"}, {"size": "big"}, {"camera": "orbit"}):
        args = {"fmt": "mp4", "size": "medium", "camera": "fixed"} | kwargs
        with pytest.raises(ValueError):
            J.job_from_replays([(rep, "#ffffff", "x")], args["fmt"], size=args["size"], camera=args["camera"], title="",
                               meta={})
    with pytest.raises(ValueError, match=S.VIDEO_NO_FRAMES):
        J.job_from_replays([], "mp4", size="medium", camera="fixed", title="", meta={})
    with pytest.raises(ValueError):
        J.job_from_replays([(rep, "not a colour", "x")], "mp4", size="medium", camera="fixed", title="", meta={})


# ---------------------------------------------------------------------------
# job_from_race (나란히 경주)
# ---------------------------------------------------------------------------
def _even_floor(v: float) -> int:
    return int(v) // 2 * 2


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_job_from_race_grid_size_fps_ranks(n: int, tmp_path: Path) -> None:
    text = _builtin_text("lander")
    lanes = [_lane(text, k, run=tmp_path / "실행") for k in range(n)]
    rows, cols = lane_grid(n)
    scores = [float(lane.replay.result.score) for lane in lanes]
    for fmt, size in (("mp4", "medium"), ("mp4", "small"), ("gif", "medium"), ("html", "small")):
        job = J.job_from_race(lanes, cols, fmt, size=size, camera="fixed", title=S.VIDEO_RACE_TITLE.format(n=3))
        if fmt == "mp4":
            base = J.MP4_SIZES[size]
            pane = (_even_floor(base[0] / cols), _even_floor(base[1] / rows))
            assert job.fps == 30
        else:
            pane = J.GIF_PANE
            assert job.fps == 20
        assert job.size == (pane[0] * cols, pane[1] * rows), (fmt, size)
        assert all(v % 2 == 0 for v in (*job.size, *pane))
        assert job.size[0] <= max(J.MP4_SIZES[size][0], cols * J.GIF_PANE[0])
        assert J.lane_layout(job) == (rows, cols, *pane)
        assert (job.layout, job.fmt, job.camera, job.encoder) == ("lanes", fmt, "fixed", "auto")
        assert [it.label for it in job.items] == [lane.entry.label for lane in lanes]
        assert [it.color for it in job.items] == [lane.entry.color.lower() for lane in lanes]
        assert all(it.view is None for it in job.items)  # 착륙선은 기본 보기가 없다
        # 같은 환경: 점수 내림차순 순위(같은 점수는 같은 순위), race.model.ranks와 같은 값
        assert job.ranks == tuple(1 + sum(1 for w in scores if w > v) for v in scores)
        assert job.ranks == tuple(r for r, _tie in race_model.ranks([lane.entry for lane in lanes], scores))
        assert job.meta[J.META_ENV] == "달 착륙선" and job.meta[J.META_ENV_HASH] == lanes[0].entry.env_hash
        assert job.meta[J.META_RUN] == "실행" and job.meta[J.META_VERSION] == __version__
        assert job.title == S.VIDEO_RACE_TITLE.format(n=3)
    if n > 1:
        assert len(set(scores)) > 1  # 칸마다 다른 유전자: 순위가 실제로 갈린다


def test_job_from_race_mixed_envs_have_no_ranks_and_lanes_keep_their_view() -> None:
    lander, jump = _builtin_text("lander"), _builtin_text("jump")
    lanes = [_lane(lander, 0), _lane(jump, 1), SimpleNamespace(replay=None, entry=_entry(lander, "#ffffff", "준비 중"))]
    job = J.job_from_race(lanes, lane_grid(2)[1], "gif", size="medium", camera="fixed", title="")
    assert len(job.items) == 2 and job.ranks is None  # 재생 없는 칸은 빼고, 환경이 섞이면 순위 없음
    assert J.META_ENV not in job.meta and J.META_ENV_HASH not in job.meta and J.META_RUN not in job.meta
    from evolab.app.race.view import entry_view

    assert [it.view for it in job.items] == [entry_view(lanes[0].entry), entry_view(lanes[1].entry)]
    assert job.items[1].view == (-3.0, -1.0, 22.0, 6.0)  # 점프 게임 기본 보기
    # (재생, 색, 이름) 묶음도 받는다: 순위·보기 없음
    tuples = [(lane.replay, lane.entry.color, lane.entry.label) for lane in lanes[:2]]
    job2 = J.job_from_race(tuples, 2, "mp4", size="small", camera="follow", title="")
    assert job2.ranks is None and [it.view for it in job2.items] == [None, None] and job2.size == (640, 360)


def test_job_from_race_columns_must_be_the_race_grid() -> None:
    text = _builtin_text("lander")
    lanes = [_lane(text, k) for k in range(4)]
    assert lane_grid(4) == (2, 2)
    for cols in (1, 3, 4, True):
        with pytest.raises(ValueError):
            J.job_from_race(lanes, cols, "gif", size="medium", camera="fixed", title="")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=S.VIDEO_NO_FRAMES):
        J.job_from_race([SimpleNamespace(replay=None, entry=None)], 1, "gif", size="medium", camera="fixed", title="")
    with pytest.raises(ValueError):
        J.job_from_race(lanes * 2, 3, "gif", size="medium", camera="fixed", title="")  # 칸 8개 > 6


# ---------------------------------------------------------------------------
# job_from_run (명령줄: 다시 시뮬레이션)
# ---------------------------------------------------------------------------
def _expected_replay(run: Path, generation: int, kind: str) -> tuple[Replay, dict[str, np.ndarray]]:
    """실험실 「1세대 대표와 지금 챔피언」 규칙으로 직접 다시 시뮬레이션: 그 세대 환경, run.json의 유전자 배치, 전시 시드."""
    store = RunStore.open(run)
    info = store.read_run_info()
    data = store.load_champion(generation, kind)
    spec = compile_env(race_model.env_text_at(store, generation))
    layout = GenomeLayout.from_dict(info["layout"]) if info.get("layout") else layout_for(spec)
    return replay(spec, layout, np.asarray(data["genome"], dtype=np.float64), seeds.display_trial()), data


def test_job_from_run_champion_defaults_to_last_champion(runs: dict[str, Path]) -> None:
    run = runs["v10_run_ga"]
    job = J.job_from_run(run, "mp4", generation=None, what="champion", size="medium", camera="fixed", encoder="auto")
    assert (job.layout, job.fmt, job.size, job.fps, job.camera, job.encoder) == (
        "overlay", "mp4", (1280, 720), 30, "fixed", "auto")
    assert len(job.items) == 1 and job.ranks is None
    item = job.items[0]
    assert (item.color, item.label, item.view) == (J.NOW_COLOR, S.LAB_LABEL_CHAMPION.format(gen=4), None)
    assert job.title == item.label
    want, data = _expected_replay(run, 4, "champion")
    assert not _deep_equal(item.replay, want, "replay")  # 실험실 재생과 비트 단위로 같다
    assert job.meta[J.META_RUN] == "v10_run_ga" and job.meta[J.META_GENERATION] == "4"
    assert job.meta[J.META_ENV] == "달 착륙선" and job.meta[J.META_VERSION] == __version__
    if "env_version" in data:
        assert job.meta[J.META_ENV_HASH] == str(data["env_version"])
    assert len(job.meta[J.META_ENV_HASH]) == 16


def test_job_from_run_generation_picks_last_champion_at_or_before(runs: dict[str, Path]) -> None:
    run = runs["v10_run_ga"]
    assert RunStore.open(run).champion_generations("champion") == [1, 2, 4]
    job = J.job_from_run(run, "gif", generation=3, what="champion", size="small", camera="follow", encoder="builtin")
    assert job.items[0].label == S.LAB_LABEL_CHAMPION.format(gen=2) and job.meta[J.META_GENERATION] == "2"
    assert (job.size, job.fps, job.camera, job.encoder) == ((640, 400), 20, "follow", "builtin")
    assert not _deep_equal(job.items[0].replay, _expected_replay(run, 2, "champion")[0], "replay")
    job99 = J.job_from_run(run, "html", generation=99, what="champion", size="small", camera="fixed", encoder="mf")
    assert job99.items[0].label == S.LAB_LABEL_CHAMPION.format(gen=4) and job99.size == (640, 400)


@pytest.mark.parametrize("name", ["v10_run_ga", "v11_run_restart"])
def test_job_from_run_versus_is_first_representative_then_champion(runs: dict[str, Path], name: str) -> None:
    run = runs[name]
    last = RunStore.open(run).champion_generations("champion")[-1]
    job = J.job_from_run(run, "mp4", generation=None, what="versus", size="small", camera="fixed", encoder="auto")
    assert job.size == (640, 360) and job.fps == 30
    labels = [S.LAB_LABEL_REP.format(gen=1), S.LAB_LABEL_CHAMPION.format(gen=last)]
    assert [(it.color, it.label) for it in job.items] == [(J.OLD_COLOR, labels[0]), (J.NOW_COLOR, labels[1])]
    assert job.title == " vs ".join(labels)  # 실험실 _replay_label과 같은 제목
    assert not _deep_equal(job.items[0].replay, _expected_replay(run, 1, "representative")[0], "first")
    assert not _deep_equal(job.items[1].replay, _expected_replay(run, last, "champion")[0], "now")


def test_job_from_run_without_champion_uses_representative(runs: dict[str, Path]) -> None:
    run = runs["v10_run_ga"]
    for p in (run / "champions").glob("*_champion.npz"):
        p.unlink()
    job = J.job_from_run(run, "mp4", generation=None, what="champion", size="medium", camera="fixed", encoder="auto")
    assert job.items[0].label == S.LAB_LABEL_REP.format(gen=4) and job.meta[J.META_GENERATION] == "4"
    assert not _deep_equal(job.items[0].replay, _expected_replay(run, 4, "representative")[0], "replay")
    job2 = J.job_from_run(run, "mp4", generation=2, what="versus", size="medium", camera="fixed", encoder="auto")
    assert [it.label for it in job2.items] == [S.LAB_LABEL_REP.format(gen=1), S.LAB_LABEL_REP.format(gen=2)]
    for p in (run / "champions").glob("*.npz"):
        p.unlink()
    with pytest.raises(RunStoreError, match=S.RACE_NO_GENERATIONS):
        J.job_from_run(run, "mp4", generation=None, what="champion", size="medium", camera="fixed", encoder="auto")


def test_job_from_run_rejects_bad_arguments(runs: dict[str, Path], tmp_path: Path) -> None:
    run = runs["v10_run_ga"]
    good = {"generation": None, "what": "champion", "size": "medium", "camera": "fixed", "encoder": "auto"}
    for change in ({"what": "all"}, {"size": "huge"}, {"camera": "spin"}, {"encoder": "ffmpeg"}, {"generation": 0},
                   {"generation": True}, {"generation": 2.0}):
        with pytest.raises(ValueError):
            J.job_from_run(run, "mp4", **(good | change))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        J.job_from_run(run, "webm", **good)  # type: ignore[arg-type]
    with pytest.raises((RunStoreError, OSError)):
        J.job_from_run(tmp_path / "없는 실행", "mp4", **good)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ExportJob 자체
# ---------------------------------------------------------------------------
def test_export_job_checks_its_fields() -> None:
    rep = _replay_of(_builtin_text("lander"))
    item = ExportItem(rep, "#AbCdEf", "x", [0, 0, 4, 3])  # type: ignore[arg-type]
    assert item.color == "#abcdef" and item.view == (0.0, 0.0, 4.0, 3.0)
    base: dict[str, Any] = {"layout": "overlay", "fmt": "mp4", "items": [item], "size": (640, 360), "fps": 30,
                            "camera": "fixed", "title": "t"}
    job = ExportJob(**base, meta=MappingProxyType({"run": "r", "generation": 3}))
    assert job.items == (item,) and type(job.meta) is dict and job.meta == {"run": "r", "generation": "3"}
    assert ExportJob(**base).meta == {} and ExportJob(**base).encoder == "auto"
    bad = [{"layout": "grid"}, {"fmt": "webm"}, {"camera": "x"}, {"size": (641, 360)}, {"size": (0, 360)},
           {"size": (640,)}, {"size": (640.0, 360)}, {"fps": 0}, {"fps": True}, {"items": []}, {"items": [item] * 7},
           {"items": [rep]}, {"title": None}]
    for change in bad:
        with pytest.raises(ValueError):
            ExportJob(**(base | change))
    with pytest.raises(ValueError):
        ExportJob(**base, encoder="x264")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ExportJob(**base, ranks=(1, 2))
    with pytest.raises(ValueError):
        ExportJob(**base, ranks=(0,))
    assert ExportJob(**base, ranks=[1]).ranks == (1,)  # type: ignore[arg-type]
    for color in ("red", "#12345", "#1234567", 0xFFFFFF):
        with pytest.raises(ValueError):
            ExportItem(rep, color, "x")  # type: ignore[arg-type]
    for view in ((0, 0, 0, 1), (0, 1, 1, 1), (0, 0, math.inf, 1), (1, 2, 3), "abcd"):
        with pytest.raises(ValueError):
            ExportItem(rep, "#ffffff", "x", view)  # type: ignore[arg-type]
    with pytest.raises(dataclasses.FrozenInstanceError):
        job.fps = 20  # type: ignore[misc]


# ---------------------------------------------------------------------------
# pickle 왕복 (설계 D49: 화면 → 자식 프로세스)
# ---------------------------------------------------------------------------
def _sample_jobs(runs: dict[str, Path]) -> dict[str, ExportJob]:
    versus = J.job_from_run(runs["v10_run_ga"], "mp4", generation=None, what="versus", size="medium", camera="fixed",
                            encoder="auto")
    lanes = [_lane(_builtin_text("lander"), k) for k in range(3)] + [_lane(_builtin_text("jump"), 3)]
    race = J.job_from_race(lanes, lane_grid(4)[1], "html", size="medium", camera="fixed", title="경주")
    jump = _replay_of(_builtin_text("jump"), seed=7)
    movers = J.job_from_replays([(jump, "#ffffff", "점프")], "gif", size="medium", camera="fixed", title="점프",
                                meta={J.META_ENV: "점프 게임"}, view=(-3, -1, 22, 6))
    return {"versus": versus, "race": race, "movers": movers}


def test_export_job_pickle_round_trip_keeps_replays(runs: dict[str, Path]) -> None:
    jobs = _sample_jobs(runs)
    # 점프 게임 재생에는 v2 움직이는 세계 물체(BodyGeom.motion)가 있다: 폰 파일·MP4가 그 자세를 그린다
    jump_rep = jobs["movers"].items[0].replay
    assert any(b.motion is not None for b in jump_rep.geometry.bodies)
    sizes = {}
    for name, job in jobs.items():
        blob = pickle.dumps(job, protocol=pickle.HIGHEST_PROTOCOL)
        sizes[name] = len(blob)
        back = pickle.loads(blob)
        assert isinstance(back, ExportJob) and back is not job
        assert not _same_job(job, back), name
        assert not _same_job(back, pickle.loads(pickle.dumps(back))), name  # 두 번 건너도 같다
    # 착륙선 「1세대 vs 지금」 두 재생: 설계가 잰 언덕차 챔피언 72.6KB와 같은 규모(넉넉한 상한)
    assert sizes["versus"] < 512 * 1024, sizes
    assert jobs["race"].ranks is None and len(jobs["race"].items) == 4


def _child_summary(job: ExportJob, conn: Any) -> None:
    """spawn 자식: 받은 작업을 요약해 돌려준다(내보내기 자식 export_main이 받는 것과 같은 길)."""
    conn.send({
        "layout": job.layout, "fmt": job.fmt, "size": job.size, "fps": job.fps, "title": job.title,
        "meta": dict(job.meta), "ranks": job.ranks,
        "items": [(it.color, it.label, it.view, it.replay.trajectory.frames,
                   hashlib.sha256(it.replay.trajectory.poses.tobytes()).hexdigest(),
                   float(it.replay.result.score), sum(b.motion is not None for b in it.replay.geometry.bodies))
                  for it in job.items],
    })
    conn.close()


def test_export_job_crosses_a_spawn_process(runs: dict[str, Path]) -> None:
    jobs = _sample_jobs(runs)
    ctx = multiprocessing.get_context("spawn")
    for name in ("versus", "movers"):
        job = jobs[name]
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_child_summary, args=(job, child))
        proc.start()
        child.close()
        try:
            assert parent.poll(120), name
            got = parent.recv()
        finally:
            proc.join(60)
        assert proc.exitcode == 0
        want = [(it.color, it.label, it.view, it.replay.trajectory.frames,
                 hashlib.sha256(it.replay.trajectory.poses.tobytes()).hexdigest(), float(it.replay.result.score),
                 sum(b.motion is not None for b in it.replay.geometry.bodies)) for it in job.items]
        assert got == {"layout": job.layout, "fmt": job.fmt, "size": job.size, "fps": job.fps, "title": job.title,
                       "meta": dict(job.meta), "ranks": job.ranks, "items": want}
    assert got["items"][0][6] >= 1  # 움직이는 세계 물체도 건너감


def test_job_module_does_not_import_qt() -> None:
    """자식이 QApplication을 만들기 전에 작업을 풀 수 있고, 폰 파일·작성기 쪽이 Qt 없이 작업을 다룬다."""
    code = ("import sys, json; import evolab.app.export.job as j; "
            "print(json.dumps(['PySide6' in sys.modules, j.FORMATS]))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True)
    assert json.loads(out.stdout) == [False, list(J.FORMATS)]

"""v6.0 P 묶음: v3.0 남은 문제 「보는 재미」 8배 위 줄 — 2세대가 올 때 화면 프로세스의 GIL 다툼 (V30.md 「남은 문제」).

v3.0 측정에서 학습 8배 관전(바퀴 고르는 언덕차)의 위 줄 첫 타일 그림이 19번 중 1번 112ms였다. 그때 뇌 추적 다시 시뮬레이션,
관전 장면 만들기(바퀴 Episode 32개), 세계 그림이 함께 돌았다. v6.0의 고침과 이 파일이 지키는 것:
- engine/replay.py: 스텝마다 한 번 부르는 양보 자리(pause, 스레드마다 pausing). 시뮬레이션을 읽지도 쓰지도 않아 결과가
  pause가 없을 때와 바이트 같다(T: 궤적·점수·추적·관측·활성화·레이, 내장 환경 6개 × 추적 셋).
- app/lab/history.py GuiPriority: working()(배경의 쉴 수 없는 일 하나, 프로세스에서 한 번에 하나, 화면이 바쁘면 시작 전에
  기다림), busy()에 들어갈 때 도는 working()이 끝나기를 잠깐 기다림, hold/release(곧 올 펌프).
- app/lab/dashboard.py: 보이는 타일이 dirty가 되면 펌프가 다 그릴 때까지 hold, 다 그리면 release(기록을 읽는 동안은 잡지 않음).
- app/lab/view.py: 뇌 추적 배경 작업(_ReplayThread yield_steps)은 화면이 바쁜 동안 스텝 사이에서 쉰다. 사용자가 누른 재생(클릭
  검사·타임라인·1세대 vs 지금·고속 챔피언)은 v3.0처럼 작업마다 한 번만 기다린다(스텝마다 쉬면 8배 학습 중 클릭 검사가 두 배 넘게
  늦었다, v6.0 검토). 세대 메시지·장면 넣기를 busy로.
측정(8배 위 줄, 고치기 전·뒤 번갈아)은 보고서에 적는다. 이 파일은 기계 속도에 기대지 않는 동작만 단언하고, 느린 시험 하나가
실제 학습 8배 관전의 이벤트 루프 틈을 넉넉한 한계로 본다.
"""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QElapsedTimer, QSettings, QTimer
from pytestqt.qtbot import QtBot

from evolab import paths
from evolab.app.lab import dashboard as D
from evolab.app.lab import history as hs
from evolab.app.lab import view as lab_view
from evolab.app.lab.view import LabView
from evolab.engine import replay as rp
from evolab.engine import seeds
from evolab.policy.genome import layout_for
from evolab.spec.io import compile_env

ENVS = ("lander", "track_car", "hill_car", "jump", "glider", "wheels")
CHAMPIONS = Path(__file__).resolve().parent / "data" / "v6_champions"


def _spec(stem: str) -> Any:
    return compile_env((paths.builtin_env_dir() / f"{stem}.evoenv").read_text(encoding="utf-8"))


def _same_replay(a: rp.Replay, b: rp.Replay) -> None:
    ta, tb = a.trajectory, b.trajectory
    assert (ta.poses.tobytes(), ta.outputs.tobytes(), ta.every, ta.dt, ta.end_step) == \
        (tb.poses.tobytes(), tb.outputs.tobytes(), tb.every, tb.dt, tb.end_step)
    assert repr(a.result) == repr(b.result)
    assert repr(a.geometry) == repr(b.geometry) and repr(a.agent) == repr(b.agent)
    assert (a.channels, a.inputs) == (b.channels, b.inputs)
    for x, y in ((a.observations, b.observations), (a.rays, b.rays)):
        assert (x is None) == (y is None)
        if x is not None:
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
    assert (a.activations is None) == (b.activations is None)
    if a.activations is not None:
        assert [x.tobytes() for x in a.activations] == [y.tobytes() for y in b.activations]
    assert repr(a.raw) == repr(b.raw)


# ---------------------------------------------------------------------------
# 양보 자리는 결과를 바꾸지 않는다
@pytest.mark.parametrize("stem", ENVS)
def test_replay_pause_points_keep_results_byte_identical(stem: str) -> None:
    spec = _spec(stem)
    layout = layout_for(spec)
    genome = np.load(CHAMPIONS / f"{stem}.npz")["genome"]
    trial = seeds.display_trial()
    for trace in (False, "brain", True):
        plain = rp.replay(spec, layout, genome, trial, trace=trace)
        calls: list[int] = []

        def pause() -> None:
            calls.append(1)
            if len(calls) % 97 == 0:
                time.sleep(0.001)  # 정말 쉬는 자리(다른 스레드가 GIL을 잡는다)

        paused = rp.replay(spec, layout, genome, trial, trace=trace, pause=pause)
        _same_replay(plain, paused)
        assert len(calls) == plain.result.steps  # 스텝마다 한 번(관측 앞)
        calls.clear()
        with rp.pausing(pause):  # 스레드마다: 바꿔 끼운 replay(감싼 함수)도 같은 스레드면 쓴다
            hooked = rp.replay(spec, layout, genome, trial, trace=trace)
        _same_replay(plain, hooked)
        assert len(calls) == plain.result.steps
    calls.clear()
    rp.replay(spec, layout, genome, trial)  # with 밖: 쓰지 않는다
    assert calls == []


def test_pausing_is_per_thread_and_nests() -> None:
    spec = _spec("lander")
    layout = layout_for(spec)
    genome = np.load(CHAMPIONS / "lander.npz")["genome"]
    trial = seeds.display_trial()
    outer: list[int] = []
    inner: list[int] = []
    other: list[int] = []
    with rp.pausing(lambda: outer.append(1)):
        th = threading.Thread(target=lambda: other.append(rp.replay(spec, layout, genome, trial).result.steps))
        th.start()
        th.join(60)
        with rp.pausing(lambda: inner.append(1)):
            steps = rp.replay(spec, layout, genome, trial).result.steps
        rp.replay(spec, layout, genome, trial)
    assert len(inner) == steps and len(outer) == steps and other == [steps]  # 다른 스레드는 쓰지 않음, 안쪽이 이김


def test_replay_unit_covers_only_the_episode_construction() -> None:
    spec = _spec("wheels")
    layout = layout_for(spec)
    genome = np.load(CHAMPIONS / "wheels.npz")["genome"]
    trial = seeds.display_trial()
    state = {"in": False}
    entered: list[int] = []
    inside: list[bool] = []

    @contextlib.contextmanager
    def unit() -> Any:
        entered.append(1)
        state["in"] = True
        try:
            yield
        finally:
            state["in"] = False

    with rp.pausing(lambda: inside.append(state["in"]), unit=unit):
        got = rp.replay(spec, layout, genome, trial, trace="brain")
    assert entered == [1] and len(inside) == got.result.steps and not any(inside)  # 첫 스텝 앞에서 끝났다
    _same_replay(got, rp.replay(spec, layout, genome, trial, trace="brain"))


# ---------------------------------------------------------------------------
# GuiPriority: working(), busy가 working을 잠깐 기다림, hold/release
def test_working_units_run_one_at_a_time_and_wait_for_a_busy_screen() -> None:
    gp = hs.GuiPriority()
    inside: list[int] = []
    overlap: list[int] = []
    lock = threading.Lock()

    def unit() -> None:
        for _ in range(20):
            with gp.working():
                with lock:
                    inside.append(1)
                    if len(inside) > 1:
                        overlap.append(1)
                time.sleep(0.001)
                with lock:
                    inside.pop()

    threads = [threading.Thread(target=unit) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert overlap == [] and gp._units == 0  # 프로세스에서 한 번에 하나
    with gp.working():
        with gp.working():  # 같은 스레드에서 겹쳐도 된다
            assert gp._units == 2
        assert gp._units == 1
    assert gp._units == 0
    # 화면이 바쁘면 시작 전에 기다린다
    started: list[float] = []

    def start_unit() -> None:
        with gp.working():
            started.append(time.monotonic())

    with gp.busy(quiet=0.0):
        th = threading.Thread(target=start_unit)
        th.start()
        time.sleep(0.1)
        assert started == []
        left = time.monotonic()
    th.join(5)
    assert started and started[0] >= left


def test_screen_waits_briefly_for_a_running_unit_before_heavy_work() -> None:
    gp = hs.GuiPriority()
    in_unit, release = threading.Event(), threading.Event()
    ended: list[float] = []

    def unit() -> None:
        with gp.working():
            in_unit.set()
            release.wait(5)
            time.sleep(0.01)
            ended.append(time.monotonic())

    th = threading.Thread(target=unit)
    th.start()
    assert in_unit.wait(5)
    release.set()
    with gp.busy(quiet=0.0):
        entered = time.monotonic()
    th.join(5)
    assert ended and entered >= ended[0]  # 도는 일 하나가 끝난 뒤에 무거운 일을 시작한다
    # 끝나지 않는 일이면 UNIT_WAIT_MAX_S만 기다린다(화면이 멈춰 있지 않게)
    in_unit.clear()
    release.clear()
    ended.clear()
    th = threading.Thread(target=unit)
    th.start()
    assert in_unit.wait(5)
    t0 = time.monotonic()
    with gp.busy(quiet=0.0):
        waited = time.monotonic() - t0
    release.set()
    th.join(5)
    assert hs.UNIT_WAIT_MAX_S - 0.005 <= waited < hs.UNIT_WAIT_MAX_S + 1.0 and not gp.is_busy


def test_a_stuck_unit_blocks_other_units_only_for_a_while() -> None:
    """앞 일이 끝나지 않으면(실험실 시험의 붙잡힌 뇌 추적 같은 경우) 다른 배경 일은 GUI_WAIT_MAX_S 뒤 그냥 함께 돈다."""
    gp = hs.GuiPriority()
    in_unit, release = threading.Event(), threading.Event()

    def stuck() -> None:
        with gp.working():
            in_unit.set()
            release.wait(10)

    th = threading.Thread(target=stuck)
    th.start()
    assert in_unit.wait(5)
    t0 = time.monotonic()
    with gp.working():
        waited = time.monotonic() - t0
        assert gp._units == 2
    release.set()
    th.join(5)
    assert hs.GUI_WAIT_MAX_S - 0.02 <= waited < hs.GUI_WAIT_MAX_S + 2.0 and gp._units == 0


def test_hold_keeps_background_waiting_until_released_or_expired() -> None:
    gp = hs.GuiPriority()
    a, b = object(), object()
    gp.hold(a, 10.0)
    gp.hold(b, 10.0)
    assert gp.is_busy
    gp.release(a)
    assert gp.is_busy  # 다른 주인의 hold는 그대로
    gp.release(b)
    assert not gp.is_busy
    gp.release(b)  # 두 번 풀어도 된다
    gp.hold(a, 0.05)
    assert gp.is_busy
    time.sleep(0.25)  # 윈도우 time.monotonic 해상도보다 넉넉히
    assert not gp.is_busy  # 시간이 지나면 저절로 풀린다
    t0 = time.monotonic()
    gp.hold(a, 10.0)
    gp.wait(limit=0.05)  # 배경의 기다림은 여전히 limit초에서 돌아온다(굶지 않게)
    assert 0.04 <= time.monotonic() - t0 < 2.0
    gp.release(a)


# ---------------------------------------------------------------------------
# 대시보드: 새 세대로 보이는 타일이 dirty면 펌프가 다 그릴 때까지 hold
def test_dashboard_holds_background_work_until_visible_tiles_are_drawn(qtbot: QtBot, tmp_path: Path) -> None:
    from test_dashboard import make_dashboard
    from test_history import fake_generation, meta_for, stats_row

    from evolab.runs import observe as ob

    dash = make_dashboard(qtbot, tmp_path)
    meta = meta_for("lander")
    layout = meta.layout
    dash.set_watch_meta(meta)
    dash.set_page("fun")
    dash.resize(1100, 700)
    dash.show()
    qtbot.waitExposed(dash)
    qtbot.waitUntil(lambda: not dash._visible_dirty(), timeout=10000)
    qtbot.waitUntil(lambda: not hs.GUI_PRIORITY.is_busy, timeout=5000)
    frame = ob.GridFrame(-10.0, -5.0, 0.8, 32, 24)
    holds: list[bool] = []
    for g in (1, 2):
        obs, X = fake_generation(g, layout, pop=8, trials=2, frame=frame, world_key=f"w{g}")
        dash.add_row({**stats_row(obs), "evaluations": str(g * 16), "elapsed": str(g)})
        dash.add_generation(g, obs.to_arrays(), X, obs.representative, None)
        holds.append(id(dash) in hs.GUI_PRIORITY._holds)
        assert hs.GUI_PRIORITY.is_busy  # 펌프가 보이는 타일을 그리기 전: 배경 작업이 새 일을 시작하지 않는다
        qtbot.waitUntil(lambda: not dash._visible_dirty(), timeout=10000)
        qtbot.waitUntil(lambda: id(dash) not in hs.GUI_PRIORITY._holds, timeout=5000)  # 다 그리면 푼다
    assert holds == [True, True]
    # 「학습 그래프」 쪽이면 보는 재미 타일이 dirty여도 보이지 않으므로 잡지 않는다
    dash.set_page("learning")
    qtbot.waitUntil(lambda: id(dash) not in hs.GUI_PRIORITY._holds, timeout=5000)
    obs, X = fake_generation(3, layout, pop=8, trials=2, frame=frame, world_key="w3")
    dash.add_generation(3, obs.to_arrays(), X, obs.representative, None)
    assert id(dash) not in hs.GUI_PRIORITY._holds


# ---------------------------------------------------------------------------
# 실험실: 다시 시뮬레이션 스레드는 화면이 바쁜 동안 스텝 사이에서 쉰다
@pytest.fixture()
def lab(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path / "runs")
    view = LabView(settings=QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.show()
    qtbot.waitExposed(view)
    assert view.load_env(paths.builtin_env_dir() / "wheels.evoenv")
    try:
        yield view
    finally:
        view.shutdown()


def test_replay_thread_yields_between_steps_while_the_screen_is_busy(qtbot: QtBot, lab: Any,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    steps: list[float] = []
    real_observe = lab_view.Episode.observe

    def observe(self: Any) -> Any:
        steps.append(time.monotonic())
        return real_observe(self)

    monkeypatch.setattr(lab_view.Episode, "observe", observe)
    assert lab.layout is not None
    genome = np.load(CHAMPIONS / "wheels.npz")["genome"]
    done: list[Any] = []
    # 뇌 추적 배경 작업과 같은 스레드(view.py _brain_check: yield_steps=True)
    th = lab_view._ReplayThread("t", [(lab.spec, lab.layout, genome, seeds.display_trial(), "brain")], lab,
                                yield_steps=True)
    th.finished_job.connect(lambda _tag, result: done.append(result))
    qtbot.waitUntil(lambda: not hs.GUI_PRIORITY.is_busy, timeout=5000)
    th.start()
    qtbot.waitUntil(lambda: len(steps) > 20, timeout=30000)
    with hs.GUI_PRIORITY.busy(quiet=0.0):  # 화면 스레드가 무거운 일을 하는 동안(GIL을 놓고 잔다)
        time.sleep(0.02)  # 지금 스텝 하나가 끝날 틈
        n0, t0 = len(steps), time.monotonic()
        # 지켜보는 시간은 배경이 굶지 않게 하는 상한(GUI_WAIT_MAX_S 0.25초)보다 넉넉히 짧게: 화면 스레드가 늦게 깨어나도
        # (윈도우 시계 한 칸 15.6ms 등) 상한이 먼저 풀리지 않는다. 스텝 하나는 1ms보다 짧아 그동안 수십 스텝이 돌 틈이다.
        time.sleep(0.05)
        during = [t for t in steps[n0:] if t > t0]
    th.wait(60000)
    qtbot.waitUntil(lambda: bool(done), timeout=30000)
    assert during == []  # 바쁜 동안 스텝이 하나도 돌지 않았다
    assert not isinstance(done[0], Exception)
    plain = rp.replay(lab.spec, lab.layout, genome, seeds.display_trial(), trace="brain")
    _same_replay(done[0][0], plain)  # 쉬어 가며 돌린 결과가 같다
    assert hs.GUI_PRIORITY._units == 0


def test_user_started_replays_do_not_pause_between_steps(qtbot: QtBot, lab: Any,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """v6.0 검토: 사용자가 누른 재생(클릭 검사·타임라인·1세대 vs 지금·고속 챔피언, yield_steps=False)은 화면이 바빠도 스텝
    사이에서 쉬지 않는다(v3.0처럼 작업 앞에서 한 번만 기다림). 실험실이 띄우는 다시 시뮬레이션 가운데 뇌 추적만 스텝마다 쉰다."""
    assert lab.layout is not None
    genome = np.load(CHAMPIONS / "wheels.npz")["genome"]
    pauses: list[bool] = []
    real_replay = lab_view.replay

    def spy(*args: Any, **kwargs: Any) -> Any:
        pauses.append(getattr(rp._local, "pause", None) is not None)
        return real_replay(*args, **kwargs)

    monkeypatch.setattr(lab_view, "replay", spy)
    done: list[Any] = []
    th = lab_view._ReplayThread("t", [(lab.spec, lab.layout, genome, seeds.display_trial(), True)], lab)
    th.finished_job.connect(lambda _tag, result: done.append(result))
    th.start()
    th.wait(60000)
    qtbot.waitUntil(lambda: bool(done), timeout=30000)
    assert pauses == [False] and not isinstance(done[0], Exception)
    # 실험실이 띄우는 스레드: 클릭 검사는 쉬지 않고, 뇌 추적 배경 작업은 쉰다
    started: list[tuple[Any, bool]] = []
    real_start = lab_view._ReplayThread.start
    monkeypatch.setattr(lab_view._ReplayThread, "start", lambda self: (started.append((self.tag, self.yield_steps)),
                                                                        real_start(self))[1])
    X = np.repeat(genome[None, :], 3, axis=0)
    rep = rp.replay(lab.spec, lab.layout, genome, seeds.display_trial())
    trial = seeds.display_trial()
    lab.dashboard.set_page("fun")
    lab._on_generation({"row": {"generation": "1"}, "generation": 1, "population": X, "representative": 0,
                        "scores": np.zeros(3), "trial": (trial.seed, trial.track), "trajectories": [rep.trajectory] * 3,
                        "champion": None, "env_version": None, "observation": None, "waited": 0.0})
    qtbot.waitUntil(lambda: lab.playing is not None, timeout=60000)
    lab._inspect(1)
    qtbot.waitUntil(lambda: lab._inspect_index == 1, timeout=60000)
    qtbot.waitUntil(lambda: any(tag[0] == "brain" for tag, _y in started), timeout=60000)
    kinds = {("brain" if tag[0] == "brain" else tag[1][0]): y for tag, y in started}
    assert kinds.get("inspect") is False and kinds.get("brain") is True, started


def test_generation_message_and_scene_adoption_mark_the_screen_busy(qtbot: QtBot, lab: Any,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    assert lab.load_env(paths.builtin_env_dir() / "lander.evoenv")  # 몸 유전자 없음: 장면을 화면 스레드에서 넣는다
    seen: dict[str, bool] = {}
    real_feed, real_set_scene = lab._feed_dashboard, lab.canvas.set_scene
    monkeypatch.setattr(lab, "_feed_dashboard", lambda msg: (seen.setdefault("generation", hs.GUI_PRIORITY._busy > 0),
                                                             real_feed(msg))[-1])
    monkeypatch.setattr(lab.canvas, "set_scene", lambda *a, **k: (seen.setdefault("scene", hs.GUI_PRIORITY._busy > 0),
                                                                  real_set_scene(*a, **k))[-1])
    X = np.repeat(lab.layout.initial()[None, :], 3, axis=0)
    rep = rp.replay(lab.spec, lab.layout, X[0], seeds.display_trial())
    trial = seeds.display_trial()
    lab._on_generation({"row": {"generation": "1"}, "generation": 1, "population": X, "representative": 0,
                        "scores": np.zeros(3), "trial": (trial.seed, trial.track), "trajectories": [rep.trajectory] * 3,
                        "champion": None, "env_version": None, "observation": None, "waited": 0.0})
    assert seen == {"generation": True, "scene": True} and lab.playing is not None


# ---------------------------------------------------------------------------
# 느린 시험: 실제 학습 8배 관전(바퀴 고르는 언덕차, 「보는 재미」 위 줄)의 이벤트 루프 틈
@pytest.mark.slow
@pytest.mark.parametrize("pace", ["every", "skip"])
def test_training_at_8x_on_the_top_fun_row_stays_responsive(qtbot: QtBot, tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch, pace: str) -> None:
    """v3.0 ML 방법(10ms 탐침, 위 줄, 8배, 처음 틈 하나 버림)을 짧게(25초). 한계는 CI 러너를 생각해 넉넉히(명세 목표 100ms는 보고서
    측정으로 본다)."""
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path / "runs")
    view = LabView(settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.show()
    qtbot.waitExposed(view)
    try:
        assert view.load_env(paths.builtin_env_dir() / "wheels.evoenv")
        view.dashboard.set_page("fun")
        bar = view.dashboard.scroll.verticalScrollBar()
        bar.setValue(0)
        view.rb_spectate.setChecked(True)
        view.pace_combo.setCurrentIndex(view.pace_combo.findData(pace))
        view.speed_combo.setCurrentIndex(view.speed_combo.findData(8))
        view.max_generations.setValue(8)
        gaps: list[float] = []
        clock = QElapsedTimer()
        probe = QTimer()
        probe.setInterval(10)
        probe.timeout.connect(lambda: gaps.append(float(clock.restart())))
        assert view.start_new()
        clock.start()
        probe.start()
        end = time.monotonic() + 25.0
        while time.monotonic() < end and view.state != "finished":
            qtbot.wait(5)
        probe.stop()
        a = np.asarray(gaps[1:])
        print(f"\n[v60 8x top {pace}] worst {a.max():.0f} ms, p99 {np.percentile(a, 99):.0f} ms, n={a.size}, "
              f"gens {view.generation}")
        assert view.generation >= 2
        assert a.max() < 250 and np.percentile(a, 99) < 80
    finally:
        view.end_session()
        view.shutdown()

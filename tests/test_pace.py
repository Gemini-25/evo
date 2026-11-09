"""v6.0 관전 방식 「학습 먼저」 (v6 설계 4.12, 5.1·5.2, D23~D25, 8.1 T19~T23).

- T19 세션: pace="skip"이면 played 없이도 세대가 나아가고 "every"면 지금처럼 기다린다. mode 명령의 pace 반영·없으면 유지·모르는 값
  무시. waited는 「학습 먼저」·고속에서 0, 「세대마다」에서 가짜 화면이 늦게 보낸 만큼.
- T20 같은 실행을 「학습 먼저」·「세대마다 다 보기」(가짜 화면이 늦게 played)·고속으로 돌려 통계(시간 열 빼고)·챔피언·세대 대표 npz·
  체크포인트(경과 시간 빼고)가 바이트 같다.
- T21 실험실(가짜 세션 메시지): 「학습 먼저」의 재생 대기열 하나, 버린 세대마다 played, 몸 유전자 환경은 마지막 세대만 장면을 만듦
  (v6.0 검토: 재생이 끝나기 조금 전에야 만들기 시작해 장면 만들기가 세대 사이에 끝나도 보인 세대마다 하나), 재생 끝 규칙(대표 시험
  0번 끝 + 0.5초 / SKIP_MAX_VIEW초 / 기다리는 세대가 없으면 가장 긴 궤적까지, 그 세대 개체를 들여다보는 중이면 자르지 않음), HUD,
  고속이면 칸이 흐림.
- T22 기다림 알림: 「세대마다」에서 최근 10세대 기다림 몫 > 0.5면 실행마다 한 번, 「학습 먼저」에서는 안 뜸, 고속으로 바꾸면 지움.
- T23 PACE_DEFAULT와 설정: lab/pace가 없으면 기본값, 있으면 그 값, 기본이 "skip"이고 lab/mode가 있으면 알림 한 번(lab/pace_notice).
"""

from __future__ import annotations

import pickle
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QSettings, QTimer

from evolab import paths
from evolab.app import strings as S
from evolab.app.lab import view as lab_view
from evolab.app.lab.client import TrainingClient
from evolab.app.lab.view import LabView
from evolab.engine import seeds
from evolab.engine import trainer as trainer_mod
from evolab.engine.evaluate import Trajectory
from evolab.engine.session import PACE_EVERY, PACE_SKIP, PACES, Session, SessionRequest
from evolab.runs.store import CHAMPIONS_DIR, RunStore
from evolab.world.physics2d.episode import Episode

LANDER = paths.builtin_env_dir() / "lander.evoenv"
HILL = paths.builtin_env_dir() / "hill_car.evoenv"
LANDER_TEXT = LANDER.read_text(encoding="utf-8")
TIME = set(trainer_mod.TIME_COLUMNS)
CONFIG = {"popsize": 6, "trials": 2, "validate_every": 24, "checkpoint_every": 2, "run_seed": 5}


# ---------------------------------------------------------------------------
# 도우미: 진짜 학습 프로세스(test_session과 같은 방식)
# ---------------------------------------------------------------------------
_CLIENTS: list[TrainingClient] = []


@pytest.fixture(autouse=True)
def _cleanup() -> Any:
    """테스트가 실패해도 학습 프로세스를 남기지 않는다."""
    yield
    while _CLIENTS:
        _CLIENTS.pop().abandon(timeout=60)


class Recorder:
    def __init__(self, client: TrainingClient) -> None:
        self.msgs: list[tuple[str, dict[str, Any]]] = []
        for name in ("started", "generation", "state_changed"):
            getattr(client, name).connect(lambda m, n=name: self.msgs.append((n, m)))
        client.failed.connect(lambda text, fatal: self.msgs.append(("failed", {"message": text, "fatal": fatal})))

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [m for k, m in self.msgs if k == kind]

    def states(self) -> list[str]:
        return [m["state"] for m in self.of("state_changed")]


def start(qtbot: Any, tmp_path: Path, **over: Any) -> tuple[TrainingClient, Recorder]:
    client = TrainingClient()
    _CLIENTS.append(client)
    rec = Recorder(client)
    config = {**CONFIG, **over.pop("config", {})}
    req = SessionRequest(**{"kind": "new", "env_text": LANDER_TEXT, "runs_dir": str(tmp_path), "workers": 1,
                            "spectate": True, "config": config, **over})
    client.start(req)
    qtbot.waitUntil(lambda: bool(rec.of("started")) or bool(rec.of("failed")), timeout=60000)
    assert not rec.of("failed"), rec.of("failed")
    return client, rec


def gens(rec: Recorder) -> list[int]:
    return [int(g["generation"]) for g in rec.of("generation")]


# ---------------------------------------------------------------------------
# T19 세션
# ---------------------------------------------------------------------------
def fake_session(**request: Any) -> Session:
    """학습 프로세스 없이 Session의 명령 처리·기다림 판단만(파이프·트레이너 대신 가짜)."""
    s = Session(None, None, SessionRequest(kind="new", **request))
    s.trainer = SimpleNamespace(state=SimpleNamespace(generation=5))
    s.state = "running"
    return s


def test_mode_command_pace_is_applied_kept_when_missing_and_unknown_values_ignored() -> None:
    assert PACES == (PACE_SKIP, PACE_EVERY) == ("skip", "every")
    assert SessionRequest(kind="new").pace == PACE_EVERY  # 데이터클래스 기본값 = 예전 동작(설계 7.3)
    assert fake_session(pace="nonsense").pace == PACE_EVERY
    s = fake_session(spectate=True, pace=PACE_SKIP)
    s.acked = 0
    assert s.pace == PACE_SKIP and not s._must_wait()  # 화면이 1세대도 다 보지 않았는데 5세대: 기다리지 않는다
    s._handle({"type": "mode", "spectate": True})  # pace가 없으면 지금 값 그대로
    s.acked = 0
    assert s.pace == PACE_SKIP and not s._must_wait()
    s._handle({"type": "mode", "spectate": True, "pace": "slow"})  # 모르는 값은 무시
    assert s.pace == PACE_SKIP
    s._handle({"type": "mode", "spectate": True, "pace": PACE_EVERY})
    assert s.pace == PACE_EVERY and s.acked == 5 and not s._must_wait()  # 바꾼 때까지는 다 본 것으로(예전 규칙)
    s.acked = 3
    assert s._must_wait()
    s._handle({"type": "mode", "spectate": True, "pace": PACE_SKIP})  # 「학습 먼저」로 바꾸면 기다림이 바로 풀린다
    s.acked = 3
    assert not s._must_wait()
    s._handle({"type": "mode", "spectate": False, "pace": PACE_EVERY})  # 고속은 방식과 상관없이 기다리지 않는다
    s.acked = 0
    assert s.pace == PACE_EVERY and not s._must_wait()


def test_waited_counts_only_running_waits() -> None:
    s = fake_session(spectate=True, pace=PACE_EVERY)
    s.acked = 0
    s._note_wait()
    assert s._wait_since is not None
    s._wait_since -= 2.0  # 2초 기다린 것처럼
    s._handle({"type": "played", "generation": 4})
    s._note_wait()
    assert s._wait_since is None and 2.0 <= s._waited < 3.0
    assert 2.0 <= s._take_waited() < 3.0 and s._waited == 0.0
    s.acked = 0
    s._note_wait()
    s._wait_since -= 1.0
    s.state = "paused"  # 멈춤은 기다림이 아니다: 멈춘 때까지만 센다
    s._note_wait()
    waited = s._take_waited()
    assert 1.0 <= waited < 2.0
    s._note_wait()
    assert s._wait_since is None and s._take_waited() == 0.0


def test_skip_session_runs_without_played_and_never_waits(qtbot: Any, tmp_path: Path) -> None:
    client, rec = start(qtbot, tmp_path, pace=PACE_SKIP, config={"max_generations": 5})
    qtbot.waitUntil(lambda: "finished" in rec.states(), timeout=180000)  # played를 한 번도 보내지 않았다
    assert gens(rec) == [1, 2, 3, 4, 5]
    msgs = rec.of("generation")
    assert all(m["trajectories"] is not None and len(m["trajectories"]) == 6 for m in msgs)  # 관전: 궤적은 보낸다
    assert [m["waited"] for m in msgs] == [0.0] * 5
    client.shutdown()


def test_every_session_waits_and_reports_waited_then_skip_releases(qtbot: Any, tmp_path: Path) -> None:
    client, rec = start(qtbot, tmp_path, pace=PACE_EVERY, config={"max_generations": 6})
    qtbot.waitUntil(lambda: len(rec.of("generation")) >= 2, timeout=120000)
    qtbot.wait(1500)  # 화면이 1세대를 늦게 다 본다
    assert gens(rec) == [1, 2]  # 지금처럼 기다린다
    client.send({"type": "played", "generation": 1})
    qtbot.waitUntil(lambda: len(rec.of("generation")) >= 3, timeout=60000)
    waited = [m["waited"] for m in rec.of("generation")]
    assert waited[:2] == [0.0, 0.0]
    assert 1.2 <= waited[2] < 30.0, waited  # 기다린 만큼(넉넉한 위 한계: 윈도우 CI)
    client.send({"type": "mode", "spectate": True, "pace": PACE_SKIP})  # 「학습 먼저」: 바로 풀리고 끝까지 간다
    qtbot.waitUntil(lambda: "finished" in rec.states(), timeout=180000)
    assert gens(rec) == [1, 2, 3, 4, 5, 6]
    # 4세대 앞의 기다림(3세대를 보낸 뒤 mode 명령이 올 때까지)은 4세대 몫이고, 그 뒤로는 기다리지 않는다
    assert all(m["waited"] == 0.0 for m in rec.of("generation")[4:])
    client.shutdown()


def test_fast_mode_never_waits(qtbot: Any, tmp_path: Path) -> None:
    client, rec = start(qtbot, tmp_path, spectate=False, pace=PACE_EVERY, config={"max_generations": 3})
    qtbot.waitUntil(lambda: "finished" in rec.states(), timeout=180000)
    msgs = rec.of("generation")
    assert gens(rec) == [1, 2, 3] and all(m["trajectories"] is None and m["waited"] == 0.0 for m in msgs)
    client.shutdown()


# ---------------------------------------------------------------------------
# T20 관전 방식은 학습 결과를 바꾸지 않는다
# ---------------------------------------------------------------------------
def _run_folder(qtbot: Any, base: Path, mode: str) -> Path:
    over: dict[str, Any] = {"fast": {"spectate": False}, "skip": {"pace": PACE_SKIP},
                            "every": {"pace": PACE_EVERY}}[mode]
    client, rec = start(qtbot, base, config={"max_generations": 5}, **over)
    if mode == "every":  # 가짜 화면: 세대마다 1초 늦게 다 봤다고 알린다(세대 계산보다 길어 학습이 기다린다)
        client.generation.connect(lambda m: QTimer.singleShot(
            1000, lambda g=int(m["generation"]): client.send({"type": "played", "generation": g})))
    qtbot.waitUntil(lambda: "finished" in rec.states(), timeout=240000)
    run_dir = Path(rec.of("started")[0]["run_dir"])
    client.shutdown()
    if mode == "every":
        assert any(m["waited"] > 0.1 for m in rec.of("generation"))  # 정말 기다렸다
    return run_dir


def _npz(path: Path) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    with np.load(path, allow_pickle=False) as z:
        return {k: (str(z[k].dtype), tuple(z[k].shape), z[k].tobytes()) for k in z.files}


def _canon(obj: Any, seen: set[int] | None = None) -> Any:
    """체크포인트를 비교할 수 있는 값으로: 배열은 (dtype, 모양, 바이트), 실수는 repr, 객체는 (클래스, 속성). 벽시계 값은 뺀다:
    학습 상태의 경과 시간("elapsed")과 pycma 학습기 피클 안의 타이머(이름에 "timer"가 든 속성: 만든 시각·누적 시간). 둘은 같은
    설정을 두 번 돌려도 다르고(같은 방식끼리도) 학습에 쓰이지 않는다. 피클 바이트 값은 풀어서 비교한다."""
    seen = set() if seen is None else seen
    if isinstance(obj, np.ndarray):
        return ("nd", obj.dtype.str, obj.shape, obj.tobytes())
    if isinstance(obj, float):
        return ("f", repr(obj))
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, bytes):
        if obj[:1] == b"\x80":
            try:
                return ("pickle", _canon(pickle.loads(obj), seen))
            except Exception:  # noqa: BLE001 - 피클이 아니면 바이트 그대로
                pass
        return obj
    if id(obj) in seen:
        return "<cycle>"
    seen.add(id(obj))
    if isinstance(obj, dict):
        return ("d", sorted((repr(k), _canon(v, seen)) for k, v in obj.items()
                            if k != "elapsed" and "timer" not in str(k)))
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, [_canon(v, seen) for v in obj])
    if isinstance(obj, np.generic):
        return ("np", obj.dtype.str, obj.tobytes())
    if hasattr(obj, "__dict__"):
        return (type(obj).__qualname__, _canon(vars(obj), seen))
    try:
        return ("p", pickle.dumps(obj, protocol=5))
    except Exception:  # noqa: BLE001 - 피클할 수 없는 것(함수 등)은 이름만
        return ("r", type(obj).__qualname__)


def test_pace_does_not_change_training_results(qtbot: Any, tmp_path: Path) -> None:
    runs = {mode: _run_folder(qtbot, tmp_path / mode, mode) for mode in ("skip", "every", "fast")}
    stores = {mode: RunStore.open(d) for mode, d in runs.items()}
    stats = {mode: [{k: v for k, v in r.items() if k not in TIME} for r in st.read_stats()]
             for mode, st in stores.items()}
    assert len(stats["skip"]) == 5
    assert stats["skip"] == stats["every"] == stats["fast"]
    files = {mode: sorted(p.name for p in (d / CHAMPIONS_DIR).iterdir()) for mode, d in runs.items()}
    assert files["skip"] == files["every"] == files["fast"] and files["skip"]
    for name in files["skip"]:
        a, b, c = (_npz(runs[m] / CHAMPIONS_DIR / name) for m in ("skip", "every", "fast"))
        assert a == b == c, name
    checkpoints = {mode: st.checkpoints() for mode, st in stores.items()}
    assert checkpoints["skip"] == checkpoints["every"] == checkpoints["fast"] and checkpoints["skip"]
    for g in checkpoints["skip"]:
        states = [stores[m].load_checkpoint(g)[1] for m in ("skip", "every", "fast")]
        assert states[0]["learner"]["es"]  # pycma 상태가 있다(비교가 빈 것끼리가 아님)
        canon = [_canon(st) for st in states]
        assert canon[0] == canon[1] == canon[2], g


# ---------------------------------------------------------------------------
# T21~T23 실험실 (가짜 세션 메시지)
# ---------------------------------------------------------------------------
def make_lab(qtbot: Any, settings_path: Path, env: Path = LANDER) -> LabView:
    view = LabView(settings=QSettings(str(settings_path), QSettings.Format.IniFormat))
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.show()
    qtbot.waitExposed(view)
    assert view.load_env(env)
    return view


@pytest.fixture()
def lab(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path / "runs")
    view = make_lab(qtbot, tmp_path / "settings.ini")
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(view.client, "send", lambda msg: sent.append(dict(msg)) or False)
    view.sent = sent  # type: ignore[attr-defined]
    view.dashboard.set_page("learning")  # 뇌 추적 배경 작업 없이(이 시험은 관전 줄만 본다)
    try:
        yield view
    finally:
        view.shutdown()


def played(view: Any) -> list[int]:
    return [int(m["generation"]) for m in view.sent if m.get("type") == "played"]


def trajectory(spec: Any, seconds: float) -> Trajectory:
    """그 환경의 움직이는 물체 수·출력 수에 맞는 정지 궤적(길이 seconds초)."""
    ep = Episode(spec, None, seeds.display_trial())
    frames = int(round(seconds / (2 * ep.dt))) + 1
    return Trajectory(np.zeros((frames, len(ep.moving_bodies), 3), np.float32),
                      np.zeros((frames, len(spec.actuator_channels)), np.float32), 2, ep.dt)


def generation_message(view: LabView, gen: int, rep_seconds: float = 3.0, longest: float = 20.0,
                       pop: int = 4, waited: float = 0.0, gen_seconds: float = 1.0,
                       trajectories: bool = True) -> dict[str, Any]:
    """학습 프로세스의 generation 알림(관전): 대표(0번)의 시험 0번 궤적 rep_seconds초, 가장 긴 궤적 longest초."""
    assert view.layout is not None and view.spec is not None
    X = np.repeat(view.layout.initial()[None, :], pop, axis=0)
    trajs = None
    if trajectories:
        trajs = [trajectory(view.spec, rep_seconds)] + [trajectory(view.spec, longest)] * (pop - 1)
    trial = seeds.display_trial()
    return {"row": {"generation": str(gen), "gen_seconds": str(gen_seconds)}, "generation": gen, "population": X,
            "representative": 0, "scores": np.arange(pop, 0, -1, dtype=np.float64), "trial": (trial.seed, trial.track),
            "trajectories": trajs, "champion": None, "env_version": None, "observation": None, "waited": waited}


def tick_at(view: LabView, t: float) -> None:
    """재생 시각을 t초에 두고 한 틱. 틱은 앞 틱 뒤로 흐른 실제 시간을 더하므로 시계를 바로 앞에서 다시 시작한다(실제 시간에 기대지
    않게: 앞 틱과 이 틱 사이에 GC 등으로 100ms가 넘게 멈춰도 경계 3.4 대 3.5가 그대로)."""
    view.play_t = t
    view._clock.restart()
    view._tick()


def freeze_clock(view: LabView) -> None:
    """재생 시각을 테스트만 움직인다(tick_at): 틱 타이머가 저절로 돌지 않게."""
    view._timer.setInterval(10**8)


def gene_message(view: LabView, gen: int, **kw: Any) -> dict[str, Any]:
    """몸 유전자 환경의 관전 세대 메시지: 가짜 집단의 마지막 유전자에 세대 번호를 적는다(장면 만들기 스파이가 읽음)."""
    msg = generation_message(view, gen, **kw)
    X = np.array(msg["population"], dtype=np.float64)
    X[:, -1] = gen / 1000.0
    msg["population"] = X
    return msg


def scene_spy(monkeypatch: pytest.MonkeyPatch, gate: threading.Event | None = None) -> tuple[list[int], list[int]]:
    """build_scene을 감싸 (시작한 세대, 끝까지 만든 세대) 목록을 돌려준다. gate가 있으면 만들기 전에 그것을 기다린다."""
    started: list[int] = []
    built: list[int] = []
    real = lab_view.build_scene

    def spy(spec: Any, layout: Any, population: Any, trial: Any, cancel: Any = None) -> Any:
        gen = int(round(float(np.asarray(population)[0, -1]) * 1000))
        started.append(gen)
        if gate is not None:
            gate.wait(30)
        out = real(spec, layout, population, trial, cancel=cancel)
        built.append(gen)
        return out

    monkeypatch.setattr(lab_view, "build_scene", spy)
    return started, built


def test_skip_keeps_only_the_newest_waiting_generation_and_ends_playback_early(qtbot: Any, lab: Any) -> None:
    assert lab.pace == PACE_SKIP == lab_view.PACE_DEFAULT and lab.pace_combo.currentData() == PACE_SKIP
    assert lab.pace_combo.isEnabled()
    lab._on_generation(generation_message(lab, 1))
    assert lab.playing is not None and int(lab.playing["generation"]) == 1
    for g in range(2, 7):  # 1세대를 보는 동안 다섯 세대가 왔다
        lab._on_generation(generation_message(lab, g))
    assert [int(m["generation"]) for m in lab.queue] == [6]  # 대기열은 가장 새 세대 하나
    assert played(lab) == [2, 3, 4, 5]  # 버린 세대마다 바로 played(세션 acked·보는 재미 따라가기)
    assert lab.generation == 6 and len(lab.rows) == 6  # 통계는 빠짐없이
    # HUD: 학습이 앞서 있다
    tick_at(lab, 1.0)
    assert lab.canvas.hud == S.LAB_HUD_LIVE_SKIP.format(gen=1, t=lab.play_t, speed=lab.speed, latest=6)
    # 새 세대가 기다린다: 대표 시험 0번 궤적(3초) 끝 + 0.5초에 끝난다(가장 긴 궤적 20초를 다 보지 않음)
    tick_at(lab, 3.0 + lab_view.SKIP_TAIL_S - 0.1)
    assert int(lab.playing["generation"]) == 1
    tick_at(lab, 3.0 + lab_view.SKIP_TAIL_S + 0.01)
    assert int(lab.playing["generation"]) == 6 and not lab.queue
    assert played(lab) == [2, 3, 4, 5, 1]
    tick_at(lab, 1.0)
    assert lab.canvas.hud == S.LAB_HUD_LIVE.format(gen=6, t=lab.play_t, speed=lab.speed)  # 학습과 같은 세대
    # 기다리는 세대가 없으면 지금처럼 가장 긴 궤적까지
    tick_at(lab, 10.0)
    assert int(lab.playing["generation"]) == 6
    tick_at(lab, 20.0 + 0.5 + 0.01)
    assert lab.playing is None and played(lab)[-1] == 6
    # 대표 궤적이 길면 실제 시간 SKIP_MAX_VIEW초에 끝난다
    lab._on_generation(generation_message(lab, 7, rep_seconds=40.0, longest=40.0))
    lab._on_generation(generation_message(lab, 8))
    assert int(lab.playing["generation"]) == 7 and [int(m["generation"]) for m in lab.queue] == [8]
    tick_at(lab, 30.0)
    assert int(lab.playing["generation"]) == 7
    lab._live_started = time.monotonic() - lab_view.SKIP_MAX_VIEW - 0.01
    tick_at(lab, 30.0)
    assert int(lab.playing["generation"]) == 8
    # 타임라인 다시 보기 등에는 적용하지 않는다(관전 재생만)
    assert lab._skip_due() is False


def test_every_mode_queues_every_generation_and_plays_to_the_longest(qtbot: Any, lab: Any) -> None:
    lab.pace_combo.setCurrentIndex(lab.pace_combo.findData(PACE_EVERY))
    assert lab.pace == PACE_EVERY
    assert lab.sent[-1] == {"type": "mode", "spectate": True, "pace": PACE_EVERY}
    lab._on_generation(generation_message(lab, 1))
    for g in range(2, 5):
        lab._on_generation(generation_message(lab, g))
    assert [int(m["generation"]) for m in lab.queue] == [2, 3, 4] and played(lab) == []
    tick_at(lab, 10.0)
    assert lab.canvas.hud == S.LAB_HUD_LIVE.format(gen=1, t=lab.play_t, speed=lab.speed)
    assert int(lab.playing["generation"]) == 1  # 기다리는 세대가 있어도 가장 긴 궤적까지
    tick_at(lab, 20.5 + 0.01)
    assert int(lab.playing["generation"]) == 2 and played(lab) == [1]
    # 「학습 먼저」로 바꾸면 기다리던 것 가운데 가장 새 것만 남는다
    n = len(lab.sent)
    lab.pace_combo.setCurrentIndex(lab.pace_combo.findData(PACE_SKIP))
    assert lab.sent[n:] == [{"type": "mode", "spectate": True, "pace": PACE_SKIP}, {"type": "played", "generation": 3}]
    assert [int(m["generation"]) for m in lab.queue] == [4] and played(lab) == [1, 3]


def test_fast_mode_greys_the_pace_field(qtbot: Any, lab: Any) -> None:
    lab.rb_fast.setChecked(True)
    assert not lab.pace_combo.isEnabled()
    assert lab.sent[-1] == {"type": "mode", "spectate": False, "pace": PACE_SKIP}
    lab.rb_spectate.setChecked(True)
    assert lab.pace_combo.isEnabled()


def test_body_gene_env_builds_a_scene_only_for_the_newest_generation(qtbot: Any, lab: Any,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """T21: 재생 중 온 세대 5개 중 마지막 하나만 장면을 만든다. 재생 중에는 지금 재생이 끝나기 조금 전(SKIP_SCENE_LEAD_S)에야
    만들기 시작하므로 그 사이 온 세대는 기다리는 자리를 바꿔 차지하고 played만 보낸다(v6.0 검토 고침)."""
    assert lab.load_env(HILL)
    assert lab.layout is not None and lab.layout.n_body > 0
    freeze_clock(lab)
    started, built = scene_spy(monkeypatch)
    lab._on_generation(gene_message(lab, 1, rep_seconds=10.0))
    qtbot.waitUntil(lambda: lab.playing is not None, timeout=60000)  # 화면이 쉬고 있으면 바로 만든다
    for g in range(2, 7):
        lab._on_generation(gene_message(lab, g, rep_seconds=10.0))
        tick_at(lab, 0.5 * (g - 1))  # 대표 궤적 10초: 재생이 끝나기(10.5초)까지 아직 멀다
        assert lab._scene_thread is None
    assert [int(m["generation"]) for m in lab._scene_jobs] == [6] and not lab.queue
    assert played(lab) == [2, 3, 4, 5]  # 버린 세대마다 바로 played
    assert started == built == [1]
    tick_at(lab, 10.5 - lab_view.SKIP_SCENE_LEAD_S / 2)  # 끝나기 조금 전: 가장 새 세대의 장면을 만든다
    qtbot.waitUntil(lambda: [int(m["generation"]) for m in lab.queue] == [6], timeout=60000)
    assert started == built == [1, 6] and int(lab.playing["generation"]) == 1
    tick_at(lab, 10.5 + 0.01)
    assert int(lab.playing["generation"]) == 6 and played(lab) == [2, 3, 4, 5, 1]


def test_skip_builds_one_scene_per_shown_generation(qtbot: Any, lab: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """v6.0 검토: 장면 만들기가 다음 세대가 오기 전에 끝나도(실제 학습: 세대 2초, 만들기 0.2초, 재생 최대 15초) 보이지 않을 장면을
    만들지 않는다. 예전에는 세대마다 만들고 다음 세대가 오면 버려 보인 세대의 몇 배를 만들었다(바퀴 차 1배 45초: 21번 중 3번)."""
    assert lab.load_env(HILL)
    freeze_clock(lab)
    started, built = scene_spy(monkeypatch)
    lab._on_generation(gene_message(lab, 1, rep_seconds=10.0))
    qtbot.waitUntil(lambda: lab.playing is not None, timeout=60000)
    gen = 1
    shown: list[int] = []
    for _cycle in range(3):
        shown.append(int(lab.playing["generation"]))
        for k in range(4):  # 재생 하나 동안 세대 넷: 만들기가 돌았다면 다음 세대 전에 끝난다
            gen += 1
            lab._on_generation(gene_message(lab, gen, rep_seconds=10.0))
            tick_at(lab, 1.0 + 2.0 * k)
            qtbot.waitUntil(lambda: lab._scene_thread is None, timeout=60000)
        tick_at(lab, 10.0)  # 재생이 끝나기 0.5초 전
        qtbot.waitUntil(lambda: bool(lab.queue), timeout=60000)
        tick_at(lab, 10.5 + 0.01)
    shown.append(int(lab.playing["generation"]))
    assert shown == [1, 5, 9, 13]
    assert started == built == shown  # 보인 세대마다 장면 하나
    assert sorted(played(lab)) == list(range(1, 13))  # 건너뛴 세대도 모두 played(학습·보는 재미 따라가기)


def test_idle_screen_finishes_the_scene_it_is_building(qtbot: Any, lab: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """화면이 쉬고 있을 때(재생 중 아님) 만들던 장면은 새 세대가 와도 끝까지 만든다: 세대가 장면 만들기보다 빨리 와도 화면이
    다음 세대를 보인다(설계 밖 보강, 보고서). 그다음 세대의 장면은 1세대 재생이 끝나기 조금 전에 만든다."""
    assert lab.load_env(HILL)
    freeze_clock(lab)
    gate = threading.Event()
    started, built = scene_spy(monkeypatch, gate)
    lab._on_generation(gene_message(lab, 1, rep_seconds=10.0))
    lab._on_generation(gene_message(lab, 2, rep_seconds=10.0))
    lab._on_generation(gene_message(lab, 3, rep_seconds=10.0))
    assert lab.playing is None and [int(m["generation"]) for m in lab._scene_jobs] == [1, 3]
    assert played(lab) == [2]
    gate.set()
    qtbot.waitUntil(lambda: lab.playing is not None, timeout=60000)
    assert built == [1] and int(lab.playing["generation"]) == 1
    tick_at(lab, 1.0)
    assert lab._scene_thread is None and [int(m["generation"]) for m in lab._scene_jobs] == [3]
    tick_at(lab, 10.2)
    qtbot.waitUntil(lambda: built == [1, 3] and [int(m["generation"]) for m in lab.queue] == [3], timeout=60000)
    assert int(lab.playing["generation"]) == 1 and started == [1, 3]


def test_skip_does_not_cut_a_generation_while_one_of_its_individuals_is_inspected(qtbot: Any, lab: Any) -> None:
    """v6.0 검토: 「학습 먼저」에서도 재생 중인 세대의 개체를 클릭해 들여다보는 동안은 그 세대를 자르지 않는다(v3.1처럼 가장 긴
    궤적까지, 학습은 기다리지 않음). 다음 세대를 재생하기 시작하면 검사 칸·레이·겹쳐 그림을 닫는다."""
    freeze_clock(lab)
    lab._on_generation(generation_message(lab, 1))
    tick_at(lab, 1.0)
    lab._inspect(2)
    qtbot.waitUntil(lambda: lab._inspect_index == 2, timeout=30000)
    assert lab.inspect.isVisible() and len(lab.canvas._overlay) == 1
    lab._on_generation(generation_message(lab, 2))
    assert [int(m["generation"]) for m in lab.queue] == [2]
    tick_at(lab, 3.0 + lab_view.SKIP_TAIL_S + 0.01)  # 대표 궤적 끝 + 0.5초: 들여다보지 않으면 여기서 끝난다
    assert int(lab.playing["generation"]) == 1 and lab._inspect_index == 2 and lab.inspect.isVisible()
    lab._live_started = time.monotonic() - lab_view.SKIP_MAX_VIEW - 0.01
    tick_at(lab, 10.0)  # 실제 시간 상한도 들여다보는 동안은 쓰지 않는다
    assert int(lab.playing["generation"]) == 1
    tick_at(lab, 20.0 + 0.5 + 0.01)  # 가장 긴 궤적 끝: 다음 세대로, 검사를 닫는다
    assert int(lab.playing["generation"]) == 2 and played(lab) == [1]
    assert lab._inspect_index is None and not lab.inspect.isVisible() and lab.canvas._overlay == []
    assert lab.canvas._rays is None
    # 검사를 닫았으면 다시 자른다
    lab._on_generation(generation_message(lab, 3))
    tick_at(lab, 3.0 + lab_view.SKIP_TAIL_S + 0.01)
    assert int(lab.playing["generation"]) == 3


def test_late_inspect_result_does_not_land_on_the_next_generation(qtbot: Any, lab: Any,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """v6.0 검토: 클릭 검사 재생이 끝나기 전에 그 세대가 캔버스에서 사라지면(다음 세대 재생) 늦게 온 결과를 붙이지 않는다.
    결과를 기다리는 동안에도 「학습 먼저」는 그 세대를 자르지 않는다."""
    freeze_clock(lab)
    gate = threading.Event()
    real = lab_view.replay
    monkeypatch.setattr(lab_view, "replay", lambda *a, **k: (gate.wait(30), real(*a, **k))[1])
    done: list[Any] = []
    real_done = lab._replay_done
    monkeypatch.setattr(lab, "_replay_done", lambda tag, result: (done.append(tag), real_done(tag, result)))
    lab._on_generation(generation_message(lab, 1))
    lab._on_generation(generation_message(lab, 2))
    tick_at(lab, 3.3)
    lab._inspect(2)  # 잘리기 0.2초 전에 클릭
    tick_at(lab, 3.0 + lab_view.SKIP_TAIL_S + 0.01)
    assert int(lab.playing["generation"]) == 1  # 검사 결과를 기다리는 동안 자르지 않는다
    tick_at(lab, 20.0 + 0.5 + 0.01)
    assert int(lab.playing["generation"]) == 2
    gate.set()
    qtbot.waitUntil(lambda: bool(done), timeout=30000)
    assert lab._inspect_index is None and not lab.inspect.isVisible()
    assert lab.canvas._overlay == [] and lab.canvas._rays is None
    assert lab._brain_play is None or not lab._brain_play.get("inspect")


# ---------------------------------------------------------------------------
# T22 기다림 알림
# ---------------------------------------------------------------------------
def test_wait_hint_once_per_run_only_in_every_mode(qtbot: Any, lab: Any) -> None:
    hint = S.LAB_PACE_WAIT_HINT.format(times=4.0)
    lab.pace_combo.setCurrentIndex(lab.pace_combo.findData(PACE_EVERY))
    for g in range(1, lab_view.WAIT_HINT_WINDOW):  # 기다림 3초 + 계산 1초: 기다린 몫 0.75지만 아직 10세대가 안 됨
        lab._on_generation(generation_message(lab, g, waited=3.0, trajectories=False))
    assert S.LAB_PACE_WAIT_HINT.split("{")[0] not in lab.status.text()
    lab._on_generation(generation_message(lab, 10, waited=3.0, trajectories=False))
    assert lab.status.text().endswith("\n" + hint)  # 전체 40초 ÷ (40 − 30) = 4.0배
    for g in range(11, 25):
        lab._on_generation(generation_message(lab, g, waited=3.0, trajectories=False))
    assert lab.status.text().count(S.LAB_PACE_WAIT_HINT.split("{")[0]) == 1  # 실행마다 한 번
    # 「학습 먼저」로 바꾸면 알림을 지운다(권한 것을 했다)
    lab.pace_combo.setCurrentIndex(lab.pace_combo.findData(PACE_SKIP))
    assert S.LAB_PACE_WAIT_HINT.split("{")[0] not in lab.status.text()
    for g in range(25, 40):  # 「학습 먼저」에서는 뜨지 않는다(기다림 값이 와도)
        lab._on_generation(generation_message(lab, g, waited=3.0, trajectories=False))
    assert S.LAB_PACE_WAIT_HINT.split("{")[0] not in lab.status.text()


def test_wait_hint_threshold_and_new_run(qtbot: Any, lab: Any) -> None:
    lab.pace_combo.setCurrentIndex(lab.pace_combo.findData(PACE_EVERY))
    for g in range(1, 21):  # 기다린 몫이 정확히 0.5: 넘지 않으면 알리지 않는다
        lab._on_generation(generation_message(lab, g, waited=1.0, gen_seconds=1.0, trajectories=False))
    assert S.LAB_PACE_WAIT_HINT.split("{")[0] not in lab.status.text()
    lab._on_generation(generation_message(lab, 21, waited=3.0, gen_seconds=1.0, trajectories=False))
    # 최근 10세대(21세대 포함): 기다림 9 + 3 = 12초, 계산 10초 → 몫 0.545, 전체 22 ÷ 계산 10 = 2.2배
    assert lab.status.text().endswith("\n" + S.LAB_PACE_WAIT_HINT.format(times=2.2))
    assert lab.load_env(LANDER)  # 다른 실행(새 실행 준비): 알림을 지우고 다시 알릴 수 있다
    assert lab._wait_note == "" and not lab._wait_hinted and not lab._wait_log
    for g in range(1, 11):
        lab._on_generation(generation_message(lab, g, waited=9.0, gen_seconds=1.0, trajectories=False))
    assert lab.status.text().endswith("\n" + S.LAB_PACE_WAIT_HINT.format(times=10.0))
    lab.rb_fast.setChecked(True)  # 고속에서는 기다림이 없다(세지 않음)
    assert not lab._wait_log
    # 흐려진 「관전 방식」 칸을 바꾸라는 알림도 지운다(v6.0 검토), 고속 세대가 와도 다시 뜨지 않는다
    assert S.LAB_PACE_WAIT_HINT.split("{")[0] not in lab.status.text() and lab._wait_note == ""
    for g in range(11, 15):
        lab._on_generation(generation_message(lab, g, waited=0.0, gen_seconds=1.0, trajectories=False))
    assert S.LAB_PACE_WAIT_HINT.split("{")[0] not in lab.status.text()


# ---------------------------------------------------------------------------
# T23 기본값과 설정
# ---------------------------------------------------------------------------
def test_pace_default_and_settings(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path / "runs")
    assert lab_view.PACE_DEFAULT == PACE_SKIP and lab_view.SKIP_MAX_VIEW == 15.0
    assert (lab_view.WAIT_HINT_WINDOW, lab_view.WAIT_HINT_RATIO) == (10, 0.5)
    # 설정이 없으면 기본값, 새 사용자에게는 알림이 없다
    fresh = tmp_path / "fresh.ini"
    v = make_lab(qtbot, fresh)
    assert v.pace == PACE_SKIP and v.pace_combo.currentData() == PACE_SKIP
    assert S.LAB_PACE_CHANGED_NOTICE not in v.status.text()
    v.speed_combo.setCurrentIndex(2)  # 보기 칸을 저장하면 lab/pace도 쓴다
    v.shutdown()
    s = QSettings(str(fresh), QSettings.Format.IniFormat)
    assert s.value("lab/pace") == PACE_SKIP and s.contains("lab/mode") and not s.contains("lab/pace_notice")
    v = make_lab(qtbot, fresh)  # v6에서 처음 쓴 사용자: 다시 켜도 알림 없음
    assert S.LAB_PACE_CHANGED_NOTICE not in v.status.text()
    v.pace_combo.setCurrentIndex(v.pace_combo.findData(PACE_EVERY))
    v.shutdown()
    assert QSettings(str(fresh), QSettings.Format.IniFormat).value("lab/pace") == PACE_EVERY
    v = make_lab(qtbot, fresh)  # 있으면 그 값
    assert v.pace == PACE_EVERY and v.pace_combo.currentData() == PACE_EVERY
    sent: list[Any] = []
    monkeypatch.setattr(v.client, "start", lambda req: sent.append(req))
    assert v.start_new()
    assert isinstance(sent[0], SessionRequest) and sent[0].pace == PACE_EVERY and sent[0].spectate
    v.shutdown()
    # 모르는 값이면 기본값
    odd = tmp_path / "odd.ini"
    QSettings(str(odd), QSettings.Format.IniFormat).setValue("lab/pace", "sometimes")
    v = make_lab(qtbot, odd)
    assert v.pace == PACE_SKIP
    v.shutdown()
    # 예전 사용자(v3.1까지: 보기 칸 lab/mode는 있고 lab/pace는 없음): 처음 한 번 알림
    old = tmp_path / "old.ini"
    s = QSettings(str(old), QSettings.Format.IniFormat)
    for key, value in (("lab/mode", "spectate"), ("lab/speed", 4), ("lab/fast_interval", 10), ("lab/workers", 2)):
        s.setValue(key, value)
    s.sync()
    v = make_lab(qtbot, old)
    assert v.pace == PACE_SKIP and v.speed == 4
    assert S.LAB_PACE_CHANGED_NOTICE in v.status.text()
    assert v.settings.value("lab/pace_notice") is not None
    v._on_closed()  # 처음 학습 한 번 동안만 남는다
    assert S.LAB_PACE_CHANGED_NOTICE not in v.status.text()
    v.shutdown()
    v = make_lab(qtbot, old)  # 두 번째 시작: 알림 없음
    assert S.LAB_PACE_CHANGED_NOTICE not in v.status.text()
    v.shutdown()
    # 예전 사용자가 방식을 바꾸면 알림을 지운다
    old2 = tmp_path / "old2.ini"
    QSettings(str(old2), QSettings.Format.IniFormat).setValue("lab/mode", "fast")
    v = make_lab(qtbot, old2)
    assert S.LAB_PACE_CHANGED_NOTICE in v.status.text() and not v.pace_combo.isEnabled()
    v.pace_combo.setCurrentIndex(v.pace_combo.findData(PACE_EVERY))
    assert S.LAB_PACE_CHANGED_NOTICE not in v.status.text()
    v.shutdown()

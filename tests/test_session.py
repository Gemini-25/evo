"""M3: 학습 프로세스(세션)와 화면 쪽 연결. 멈춤·한 세대·관전 대기·설정과 환경 바꾸기·결정성."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evolab import paths
from evolab.app import strings as S
from evolab.app.lab.client import TrainingClient
from evolab.engine import trainer as trainer_mod
from evolab.engine.pool import WorkerPool
from evolab.engine.session import SessionRequest
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.runs.store import RunStore

LANDER_TEXT = (paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8")
TIME = set(trainer_mod.TIME_COLUMNS)
CONFIG = {"popsize": 6, "trials": 2, "validate_every": 24, "checkpoint_every": 2, "run_seed": 5}


class Recorder:
    def __init__(self, client: TrainingClient) -> None:
        self.msgs: list[tuple[str, dict[str, Any]]] = []
        for name in ("started", "generation", "state_changed", "applied", "rewound"):
            getattr(client, name).connect(lambda m, n=name: self.msgs.append((n, m)))
        client.failed.connect(lambda text, fatal: self.msgs.append(("failed", {"message": text, "fatal": fatal})))
        client.closed.connect(lambda: self.msgs.append(("closed", {})))

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [m for k, m in self.msgs if k == kind]

    def states(self) -> list[str]:
        return [m["state"] for m in self.of("state_changed")]


_CLIENTS: list[TrainingClient] = []


@pytest.fixture(autouse=True)
def _cleanup() -> Any:
    """테스트가 실패해도 학습 프로세스를 남기지 않는다(남으면 pytest가 끝날 때 기다리며 멈춘다)."""
    yield
    while _CLIENTS:
        _CLIENTS.pop().abandon(timeout=60)


def new_client() -> TrainingClient:
    client = TrainingClient()
    _CLIENTS.append(client)
    return client


def start(qtbot: Any, tmp_path: Path, **over: Any) -> tuple[TrainingClient, Recorder]:
    client = new_client()
    rec = Recorder(client)
    config = {**CONFIG, **over.pop("config", {})}
    req = SessionRequest(**{"kind": "new", "env_text": LANDER_TEXT, "runs_dir": str(tmp_path), "workers": 1,
                            "spectate": False, "config": config, **over})
    client.start(req)
    qtbot.waitUntil(lambda: bool(rec.of("started")) or bool(rec.of("failed")), timeout=60000)
    assert not rec.of("failed"), rec.of("failed")
    return client, rec


def strip(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in r.items() if k not in TIME} for r in rows]


def straight_rows(gens: int, **over: Any) -> list[dict[str, str]]:
    config = TrainConfig.from_dict({**CONFIG, "max_generations": gens, **over})
    with WorkerPool(0) as pool:
        t = Trainer(LANDER_TEXT, config, pool)
        rows: list[dict[str, Any]] = []
        t.run(rows.append)
    return strip(rows)


def test_run_to_stop_condition_matches_plain_trainer(qtbot: Any, tmp_path: Path) -> None:
    client, rec = start(qtbot, tmp_path, config={"max_generations": 4})
    qtbot.waitUntil(lambda: "finished" in rec.states(), timeout=120000)
    gens = rec.of("generation")
    assert [g["generation"] for g in gens] == [1, 2, 3, 4]
    assert all(g["trajectories"] is None for g in gens)
    run_dir = Path(rec.of("started")[0]["run_dir"])
    client.shutdown()
    assert rec.of("closed")
    assert strip(RunStore.open(run_dir).read_stats()) == straight_rows(4)
    assert strip([g["row"] for g in gens]) == straight_rows(4)


def test_spectate_waits_for_playback(qtbot: Any, tmp_path: Path) -> None:
    client, rec = start(qtbot, tmp_path, spectate=True, config={"max_generations": 5})
    qtbot.waitUntil(lambda: len(rec.of("generation")) >= 2, timeout=120000)
    qtbot.wait(2500)
    assert len(rec.of("generation")) == 2  # 1세대 재생이 끝났다는 알림 전에는 3세대로 가지 않는다
    g1 = rec.of("generation")[0]
    assert len(g1["trajectories"]) == 6 and all(t is not None for t in g1["trajectories"])
    assert g1["population"].shape[0] == 6
    client.send({"type": "played", "generation": 1})
    qtbot.waitUntil(lambda: len(rec.of("generation")) >= 3, timeout=60000)
    client.send({"type": "mode", "spectate": False})  # 고속 모드: 기다리지 않는다
    qtbot.waitUntil(lambda: "finished" in rec.states(), timeout=120000)
    assert [g["generation"] for g in rec.of("generation")] == [1, 2, 3, 4, 5]
    assert rec.of("generation")[-1]["trajectories"] is None
    client.shutdown()
    assert strip([g["row"] for g in rec.of("generation")]) == straight_rows(5)


def test_pause_change_config_and_env_then_continue(qtbot: Any, tmp_path: Path) -> None:
    client, rec = start(qtbot, tmp_path, config={"max_generations": 30})
    qtbot.waitUntil(lambda: len(rec.of("generation")) >= 1, timeout=120000)
    client.send({"type": "pause"})
    qtbot.waitUntil(lambda: "paused" in rec.states(), timeout=60000)
    paused_at = rec.of("state_changed")[-1]["generation"]
    # 바꿀 수 없는 설정은 거절
    client.send({"type": "config", "changes": {"popsize": 8}})
    qtbot.waitUntil(lambda: len(rec.of("applied")) >= 1, timeout=30000)
    assert rec.of("applied")[-1]["ok"] is False
    client.send({"type": "config", "changes": {"max_generations": paused_at + 2}})
    qtbot.waitUntil(lambda: len(rec.of("applied")) >= 2, timeout=30000)
    assert rec.of("applied")[-1]["ok"] is True
    # 차원이 바뀌는 환경은 거절, 중력만 바꾼 환경은 받아들인다
    data = json.loads(LANDER_TEXT)
    data["brain"]["hidden"] = [8]
    client.send({"type": "env", "text": json.dumps(data, ensure_ascii=False)})
    qtbot.waitUntil(lambda: len(rec.of("applied")) >= 3, timeout=30000)
    assert rec.of("applied")[-1]["ok"] is False
    data = json.loads(LANDER_TEXT)
    data["world"]["gravity"] = [0.0, -1.5]
    new_text = json.dumps(data, ensure_ascii=False, indent=2)
    client.send({"type": "env", "text": new_text})
    qtbot.waitUntil(lambda: len(rec.of("applied")) >= 4, timeout=30000)
    assert rec.of("applied")[-1]["ok"] is True, rec.of("applied")[-1]
    client.send({"type": "run"})
    qtbot.waitUntil(lambda: "finished" in rec.states(), timeout=120000)
    run_dir = Path(rec.of("started")[0]["run_dir"])
    client.shutdown()
    store = RunStore.open(run_dir)
    kinds = [(e["kind"], e["generation"]) for e in store.read_events()]
    assert ("config_change", paused_at) in kinds and ("env_change", paused_at) in kinds
    assert paused_at in store.checkpoints()  # 바꾼 지점은 보호
    info = store.read_run_info()
    assert info["env_versions"][-1]["from_generation"] == paused_at
    assert [int(r["generation"]) for r in store.read_stats()] == list(range(1, paused_at + 3))
    # 이어하기는 바뀐 환경으로: 같은 체크포인트에서 Trainer.resume으로 돌린 결과와 같다
    with WorkerPool(0) as pool:
        again = Trainer.resume(store, pool, paused_at)
        assert again.env_text == new_text and again.config.max_generations == paused_at + 2


def test_single_step_and_resume_request(qtbot: Any, tmp_path: Path) -> None:
    client, rec = start(qtbot, tmp_path, start_running=False, config={"max_generations": 3})
    qtbot.waitUntil(lambda: bool(rec.states()), timeout=30000)  # 시작 알림 바로 뒤에 상태 알림이 온다
    assert rec.states()[-1] == "paused"
    client.send({"type": "step"})
    qtbot.waitUntil(lambda: rec.states().count("paused") >= 2, timeout=120000)
    assert [g["generation"] for g in rec.of("generation")] == [1]
    run_dir = Path(rec.of("started")[0]["run_dir"])
    client.shutdown()
    # 다른 프로세스에서 이어하기
    client2 = new_client()
    rec2 = Recorder(client2)
    client2.start(SessionRequest(kind="resume", run_dir=str(run_dir), workers=1, spectate=False))
    qtbot.waitUntil(lambda: "finished" in rec2.states() or bool(rec2.of("failed")), timeout=120000)
    assert not rec2.of("failed")
    assert rec2.of("started")[0]["generation"] == 1
    client2.shutdown()
    assert strip(RunStore.open(run_dir).read_stats()) == straight_rows(3)


FAST = {"validation_seeds": 4, "validation_candidates": 1}
BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

APPLY_REFERENCE = """
import json, sys
from evolab import paths
from evolab.engine.pool import WorkerPool
from evolab.engine.trainer import TIME_COLUMNS, TrainConfig, Trainer

config, before, changes, after = json.loads(sys.argv[1]), int(sys.argv[2]), json.loads(sys.argv[3]), int(sys.argv[4])
text = (paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8")
with WorkerPool(0) as pool:
    t = Trainer(text, TrainConfig.from_dict(config), pool)
    rows = [t.step() for _ in range(before)]
    t.apply_config(changes)
    rows += [t.step() for _ in range(after)]
print(json.dumps([{k: str(v) for k, v in r.items() if k not in TIME_COLUMNS} for r in rows]))
"""


def test_enabling_restart_while_paused_matches_trainer_apply_config(qtbot: Any, tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """v1.1: 멈춤 → config 명령으로 자동 재시작을 켬 → 계속한 결과가 같은 세대에 apply_config한 Trainer와 같다.

    프로그램은 모든 프로세스에서 BLAS 스레드 1개로 돈다(명세). pytest 프로세스는 numpy를 먼저 불러 스레드 수가
    다를 수 있으므로(CMA-ES 고윳값 분해 결과가 마지막 자리에서 달라진다) 학습 프로세스와 기준 계산을 모두 1개로 맞춘다.
    """
    import os
    import subprocess
    import sys

    for name in BLAS_VARS:
        monkeypatch.setenv(name, "1")
    config = {**FAST, "max_generations": 10}
    client, rec = start(qtbot, tmp_path, start_running=False, config=config)
    qtbot.waitUntil(lambda: bool(rec.states()), timeout=30000)
    for n in (2, 3):
        client.send({"type": "step"})
        qtbot.waitUntil(lambda n=n: rec.states().count("paused") >= n, timeout=120000)
    assert [g["generation"] for g in rec.of("generation")] == [1, 2]
    client.send({"type": "config", "changes": {"restart_patience": 24}})
    qtbot.waitUntil(lambda: len(rec.of("applied")) >= 1, timeout=30000)
    reply = rec.of("applied")[-1]
    assert reply["ok"] is True and reply["changes"] == {"restart_patience": 24} and reply["generation"] == 2
    client.send({"type": "run"})
    qtbot.waitUntil(lambda: "finished" in rec.states(), timeout=180000)
    run_dir = Path(rec.of("started")[0]["run_dir"])
    client.shutdown()
    store = RunStore.open(run_dir)
    assert ("config_change", 2) in [(e["kind"], e["generation"]) for e in store.read_events()]
    proc = subprocess.run([sys.executable, "-c", APPLY_REFERENCE, json.dumps({**CONFIG, **config}), "2",
                           json.dumps({"restart_patience": 24}), "8"],
                          capture_output=True, text=True, timeout=600, env=dict(os.environ))
    assert proc.returncode == 0, proc.stderr
    expect = json.loads(proc.stdout)
    rows = store.read_stats()
    assert strip(rows) == expect == strip([g["row"] for g in rec.of("generation")])
    assert any(r["switch"] in ("1", "2") for r in rows)  # 켠 뒤 재시작이 실제로 일어났다


def test_env_change_during_qualifying_is_refused(qtbot: Any, tmp_path: Path) -> None:
    client, rec = start(qtbot, tmp_path, start_running=False,
                        config={**FAST, "qualify_entrants": 2, "qualify_trials": 48, "max_generations": 6})
    qtbot.waitUntil(lambda: bool(rec.states()), timeout=30000)
    client.send({"type": "step"})
    qtbot.waitUntil(lambda: rec.states().count("paused") >= 2, timeout=120000)
    data = json.loads(LANDER_TEXT)
    data["world"]["gravity"] = [0.0, -1.5]
    client.send({"type": "env", "text": json.dumps(data, ensure_ascii=False)})
    qtbot.waitUntil(lambda: len(rec.of("applied")) >= 1, timeout=30000)
    reply = rec.of("applied")[-1]
    assert reply["ok"] is False and reply["message"] == S.TRAIN_ENV_DURING_QUALIFY.format(n=1, k=2)
    run_dir = Path(rec.of("started")[0]["run_dir"])
    client.shutdown()
    store = RunStore.open(run_dir)
    assert "env_change" not in [e["kind"] for e in store.read_events()]
    assert len(store.read_run_info()["env_versions"]) == 1


def test_bad_request_reports_error(qtbot: Any, tmp_path: Path) -> None:
    client = new_client()
    rec = Recorder(client)
    client.start(SessionRequest(kind="resume", run_dir=str(tmp_path / "없는 실행"), workers=1))
    qtbot.waitUntil(lambda: bool(rec.of("closed")), timeout=60000)
    assert rec.of("failed") and rec.of("failed")[0]["fatal"]
    assert not client.active


@pytest.mark.parametrize("how", ["shutdown", "gone"])
def test_session_ends_cleanly(qtbot: Any, tmp_path: Path, how: str) -> None:
    client, rec = start(qtbot, tmp_path, config={"max_generations": 1000})
    qtbot.waitUntil(lambda: len(rec.of("generation")) >= 1, timeout=120000)
    proc = client._proc
    run_dir = Path(rec.of("started")[0]["run_dir"])
    if how == "shutdown":
        client.shutdown()
    else:  # 화면 프로세스가 갑자기 사라진 것처럼 파이프만 닫는다
        client._cmd.close()
        client._events.close()
        proc.join(timeout=60)
    assert not proc.is_alive()
    store = RunStore.open(run_dir)
    assert store.checkpoints() and store.checkpoints()[-1] == len(store.read_stats())


LEAK_SCRIPT = '''
import sys, tempfile, time
from evolab import paths
from evolab.app.main import create_application
from evolab.app.lab.client import TrainingClient
from evolab.engine.session import SessionRequest

if __name__ == "__main__":
    app = create_application()
    c = TrainingClient()
    text = (paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8")
    c.start(SessionRequest(kind="new", env_text=text, runs_dir=sys.argv[1], workers=1, spectate=False,
                           config={"popsize": 6, "trials": 2, "max_generations": 1000}))
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5:
        app.processEvents()
        time.sleep(0.01)
    print("leaving without shutdown")
'''


def test_python_exits_even_if_a_session_was_not_shut_down(tmp_path: Path) -> None:
    """정리하지 않고 끝내도 멈추지 않는다(예전에는 multiprocessing이 끝날 때 학습 프로세스를 끝없이 기다렸다)."""
    import os
    import subprocess
    import sys

    script = tmp_path / "leak.py"
    script.write_text(LEAK_SCRIPT, encoding="utf-8")
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    proc = subprocess.run([sys.executable, str(script), str(tmp_path / "runs")], capture_output=True, text=True,
                          timeout=180, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "leaving without shutdown" in proc.stdout
    runs = list((tmp_path / "runs").iterdir())
    from evolab.runs.store import RunStore

    assert RunStore.open(runs[0]).checkpoints()  # 끝날 때 이번 세대를 마치고 체크포인트를 남겼다

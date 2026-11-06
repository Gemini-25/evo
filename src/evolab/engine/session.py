"""학습 프로세스 (명세: 전체 구조). 화면 프로세스와 파이프 두 개(명령, 결과)로만 이야기한다.

화면 → 학습 (명령): run, pause, step, mode, played, config, env, quit
학습 → 화면 (결과): started, generation, state, applied, rewind, error, closed

- 명령은 받는 스레드가 계속 읽어 큐에 넣는다. 윈도우 파이프는 읽는 쪽이 읽을 때까지 큰 메시지(환경 파일)를 다
  쓰지 못하므로, 학습이 세대를 계산하는 동안에도 화면이 멈추지 않게 하려는 것이다.
- 설정 변경과 환경 교체는 멈춘 상태에서만, 세대 경계에서 적용한다. 적용할 때 체크포인트를 남기고 보호한다
  (명세: 환경 수정 지점은 정리하지 않는다). 변경은 세대 번호와 함께 events.jsonl에 남는다.
- 관전 모드: 화면이 세대 N을 재생하는 동안 N+1을 계산하고, 재생이 끝났다는 알림(played) 없이 그보다 앞서지 않는다.
  고속 모드는 기다리지 않는다. 어느 모드든 학습 결과는 같다(궤적 기록은 읽기만 한다).
- 평가 오류(PoolError)로 세대가 중간에 끊기면 마지막 체크포인트로 되돌아가 멈춘다(rewind).
- 화면이 사라지면(파이프가 끊김) 이번 세대를 마치고 체크포인트를 남긴 뒤 끝난다. 워커는 이 프로세스가 끝나면
  스스로 끝난다.
"""

from __future__ import annotations

import queue
import signal
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evolab.app import strings as S

RUNNING = "running"
PAUSED = "paused"
FINISHED = "finished"


@dataclass
class SessionRequest:
    """학습 프로세스를 띄울 때 넘기는 것. kind = "new"(새 실행) 또는 "resume"(실행 폴더에서 이어하기)."""

    kind: str
    workers: int | None = None
    spectate: bool = True
    record_every: int = 2
    start_running: bool = True
    env_text: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    runs_dir: str = ""
    run_dir: str = ""
    generation: int | None = None


def session_main(cmd_conn: Any, event_conn: Any, request: SessionRequest) -> None:
    """학습 프로세스 진입점(spawn). Ctrl+C는 화면 프로세스가 처리한다."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass
    Session(cmd_conn, event_conn, request).main()


class Session:
    def __init__(self, cmd_conn: Any, event_conn: Any, request: SessionRequest) -> None:
        self.cmd_conn = cmd_conn
        self.event_conn = event_conn
        self.req = request
        self.cmds: queue.Queue[dict[str, Any]] = queue.Queue()
        self.quit = False
        self.gui_gone = False
        self.state = PAUSED
        self.spectate = bool(request.spectate)
        self.step_once = False
        self.acked = 0
        self.pool: Any = None
        self.store: Any = None
        self.trainer: Any = None
        self.stop_logged = False

    # ------------------------------------------------------------------
    def main(self) -> None:
        threading.Thread(target=self._read, name="commands", daemon=True).start()
        try:
            try:
                self._open()
            except Exception as exc:  # 시작 실패는 화면에 알리고 끝낸다
                self._send({"type": "error", "fatal": True, "message": S.TRAIN_CANNOT_START.format(reason=_reason(exc)),
                            "detail": traceback.format_exc()})
                return
            self._loop()
        except Exception as exc:
            self._send({"type": "error", "fatal": True, "message": S.SESSION_CRASHED.format(reason=_reason(exc)),
                        "detail": traceback.format_exc()})
        finally:
            self._close()

    def _read(self) -> None:
        while True:
            try:
                msg = self.cmd_conn.recv()
            except (EOFError, OSError):
                self.cmds.put({"type": "quit", "gone": True})
                return
            if isinstance(msg, dict):
                self.cmds.put(msg)

    def _send(self, msg: dict[str, Any]) -> None:
        if self.gui_gone:
            return
        try:
            self.event_conn.send(msg)
        except (OSError, ValueError, EOFError):
            self.gui_gone = True
            self.quit = True

    # ------------------------------------------------------------------
    def _open(self) -> None:
        from evolab.engine.pool import WorkerPool, default_workers
        from evolab.engine.runsetup import create_run
        from evolab.engine.trainer import TrainConfig, Trainer
        from evolab.runs.store import RunStore
        from evolab.spec.io import compile_env

        req = self.req
        workers = req.workers if req.workers is not None else default_workers()
        if req.kind == "new":
            spec = compile_env(req.env_text)
            config = TrainConfig.from_dict(req.config)
            config.validate()
            self.pool = WorkerPool(workers)
            self.store = create_run(Path(req.runs_dir), req.env_text, spec, config, workers)
            self.trainer = Trainer(req.env_text, config, self.pool, self.store)
        elif req.kind == "resume":
            self.store = RunStore.open(Path(req.run_dir))
            self.pool = WorkerPool(workers)
            self.trainer = Trainer.resume(self.store, self.pool, req.generation)
            self.store.append_event("resume", self.trainer.state.generation, workers=workers)
        else:
            raise ValueError(req.kind)
        self.workers = workers
        self.acked = self.trainer.state.generation
        self._send({"type": "started", **self._snapshot()})

    def _snapshot(self) -> dict[str, Any]:
        t, s = self.trainer, self.trainer.state
        champion = None
        if s.champion.genome is not None:
            champion = {"genome": s.champion.genome, "score": s.champion.score, "generation": s.champion.generation}
        return {
            "run_dir": str(self.store.root),
            "run_info": self.store.read_run_info(),
            "env_text": t.env_text,
            "env_version": t.env_version,
            "generation": s.generation,
            "config": t.config.to_dict(),
            "workers": self.workers,
            "layout": t.layout.to_dict(),
            "stats": self.store.read_stats(),
            "events": self.store.read_events(),
            "champion": champion,
            "spectate": self.spectate,
        }

    def _close(self) -> None:
        try:
            if self.trainer is not None and self.store is not None:
                if self.trainer.state.generation:
                    self.trainer.checkpoint()
                if not self.stop_logged:
                    self.store.append_event("stop", self.trainer.state.generation, reason="requested")
        except Exception as exc:  # 끝내는 중의 저장 실패는 알리기만 한다
            self._send({"type": "error", "fatal": False, "message": S.SESSION_SAVE_FAILED.format(reason=_reason(exc))})
        if self.pool is not None:
            self.pool.close()
        self._send({"type": "closed"})
        for c in (self.event_conn, self.cmd_conn):
            try:
                c.close()
            except OSError:
                pass

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        from evolab.engine.pool import PoolError

        self._set_state(RUNNING if self.req.start_running else PAUSED)
        while not self.quit:
            idle = self.state != RUNNING or self._must_wait()
            self._drain(block=idle)
            if self.quit or self.state != RUNNING or self._must_wait():
                continue
            reason = self.trainer.check_stop()
            if reason is not None:
                self._finish(reason)
                continue
            try:
                row = self.trainer.step(record=self.spectate)
            except PoolError as exc:
                self._send({"type": "error", "fatal": False, "message": str(exc)})
                self._rewind()
                continue
            self._send_generation(row)
            if self.step_once:
                self.step_once = False
                self._pause()

    def _must_wait(self) -> bool:
        return self.spectate and self.acked < self.trainer.state.generation - 1

    def _drain(self, block: bool) -> None:
        try:
            msg = self.cmds.get(timeout=0.2) if block else self.cmds.get_nowait()
        except queue.Empty:
            return
        self._handle(msg)
        while not self.quit:
            try:
                msg = self.cmds.get_nowait()
            except queue.Empty:
                return
            self._handle(msg)

    def _set_state(self, state: str, reason: str | None = None) -> None:
        self.state = state
        self._send({"type": "state", "state": state, "reason": reason, "generation": self.trainer.state.generation})

    def _pause(self) -> None:
        self.trainer.checkpoint()
        self.store.append_event("pause", self.trainer.state.generation)
        self._set_state(PAUSED)

    def _finish(self, reason: str) -> None:
        self.trainer.stop_reason = reason
        self.trainer.finish()
        self.store.append_event("stop", self.trainer.state.generation, reason=reason)
        self.stop_logged = True
        self.step_once = False
        self._set_state(FINISHED, reason)

    def _handle(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "quit":
            self.quit = True
            self.gui_gone = bool(msg.get("gone"))
        elif kind == "run" or kind == "step":
            if self.state == RUNNING:
                self.step_once = self.step_once or kind == "step"
                return
            reason = self.trainer.check_stop()
            if reason is not None:
                self._send({"type": "applied", "what": "run", "ok": False,
                            "message": S.SESSION_STILL_STOPPED.format(reason=S.TRAIN_STOP_REASONS.get(reason, reason))})
                return
            if self.state == FINISHED or self.stop_logged:
                self.stop_logged = False
            self.store.append_event("continue", self.trainer.state.generation)
            self.step_once = kind == "step"
            self._set_state(RUNNING)
        elif kind == "pause":
            if self.state == RUNNING:
                self.step_once = False
                self._pause()
        elif kind == "mode":
            self.spectate = bool(msg.get("spectate"))
            self.acked = self.trainer.state.generation
        elif kind == "played":
            self.acked = max(self.acked, int(msg.get("generation", 0)))
        elif kind == "config":
            self._apply(msg, "config")
        elif kind == "env":
            self._apply(msg, "env")

    def _apply(self, msg: dict[str, Any], what: str) -> None:
        from evolab.engine.trainer import TrainerChangeError
        from evolab.spec.compiled import EnvFileError

        if self.state == RUNNING:
            self._send({"type": "applied", "what": what, "ok": False, "message": S.SESSION_PAUSE_FIRST})
            return
        t, gen = self.trainer, self.trainer.state.generation
        try:
            if what == "config":
                changes = t.apply_config(dict(msg.get("changes") or {}))
                if changes:
                    self.store.append_event("config_change", gen, changes=changes)
                    self._mark_change_point()
                self._send({"type": "applied", "what": what, "ok": True, "generation": gen,
                            "config": t.config.to_dict(), "changes": changes})
            else:
                text = str(msg.get("text", ""))
                old = t.env_text
                t.replace_env(text)
                if text != old:
                    vid = self.store.add_env_version(text, gen)
                    self.store.append_event("env_change", gen, version=vid)
                    self._mark_change_point()
                self._send({"type": "applied", "what": what, "ok": True, "generation": gen,
                            "env_text": t.env_text, "env_version": t.env_version})
        except (TrainerChangeError, EnvFileError, ValueError) as exc:
            self._send({"type": "applied", "what": what, "ok": False, "message": _reason(exc)})

    def _mark_change_point(self) -> None:
        """바뀐 설정·환경이 담긴 체크포인트를 남기고 정리 대상에서 뺀다(0세대면 체크포인트가 없다)."""
        gen = self.trainer.state.generation
        if gen:
            self.trainer.checkpoint()
            self.store.protect(gen)

    def _rewind(self) -> None:
        """마지막 정상 체크포인트로 되돌아간다. 체크포인트가 없으면 처음부터."""
        from evolab.engine.trainer import TrainConfig, Trainer

        if self.store.checkpoints():
            self.trainer = Trainer.resume(self.store, self.pool)
        else:
            info = self.store.read_run_info()
            self.store.truncate_after(0)
            config = TrainConfig.from_dict({**info.get("config", {}), **self.trainer.config.to_dict()})
            self.trainer = Trainer(self.trainer.env_text, config, self.pool, self.store)
        self.acked = self.trainer.state.generation
        self.step_once = False
        self.state = PAUSED
        self._send({"type": "rewind", **self._snapshot()})
        self._set_state(PAUSED)

    def _send_generation(self, row: dict[str, Any]) -> None:
        g = self.trainer.last_generation
        s = self.trainer.state
        champion = None
        if g.champion_changed and s.champion.genome is not None:
            champion = {"genome": s.champion.genome, "score": s.champion.score, "generation": s.champion.generation}
        self._send({
            "type": "generation",
            "generation": g.generation,
            "row": row,
            "population": g.population,
            "scores": g.scores,
            "representative": g.representative,
            "trial": (g.trial0.seed, g.trial0.track),
            "trajectories": g.trajectories,
            "champion": champion,
            "env_version": g.env_version,
        })


def _reason(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or type(exc).__name__

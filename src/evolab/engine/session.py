"""학습 프로세스 (명세: 전체 구조). 화면 프로세스와 파이프 두 개(명령, 결과)로만 이야기한다.

화면 → 학습 (명령): run, pause, step, mode, played, config, env, quit
학습 → 화면 (결과): started, generation, state, applied, rewind, error, closed

- 명령은 받는 스레드가 계속 읽어 큐에 넣는다. 윈도우 파이프는 읽는 쪽이 읽을 때까지 큰 메시지(환경 파일)를 다
  쓰지 못하므로, 학습이 세대를 계산하는 동안에도 화면이 멈추지 않게 하려는 것이다.
- 설정 변경과 환경 교체는 멈춘 상태에서만, 세대 경계에서 적용한다. 적용할 때 체크포인트를 남기고 보호한다
  (명세: 환경 수정 지점은 정리하지 않는다). 변경은 세대 번호와 함께 events.jsonl에 남는다.
- 관전 모드: 화면이 세대 N을 재생하는 동안 N+1을 계산하고, 재생이 끝났다는 알림(played) 없이 그보다 앞서지 않는다
  (관전 방식 「세대마다 다 보기」 PACE_EVERY). v6.0 「학습 먼저」(PACE_SKIP, 설계 4.12·D23)와 고속 모드는 기다리지 않는다
  (관전이면 궤적은 지금처럼 기록해 보낸다: 화면이 그때까지 끝난 가장 새 세대를 고른다). 어느 모드·방식이든 학습 결과는 같다
  (궤적 기록은 읽기만 하고, 기다림은 세대 사이의 쉼일 뿐이다).
- 세대 메시지의 "waited"(v6.0 D25, 저장하지 않음): 그 세대를 계산하기 전에 재생을 기다린 초. 「학습 먼저」·고속이면 늘 0.
  멈춘 동안은 세지 않는다.
- 평가 오류(PoolError)로 세대가 중간에 끊기면 마지막 체크포인트로 되돌아가 멈춘다(rewind).
- 세대 메시지에는 그 세대의 관찰 기록(v1.2 "observation": 세대 대표 파일의 obs_ 키 사전 또는 None)이 함께 간다.
  관찰 기록을 만들지 못하면(학습은 그대로) 비치명 오류를 세션에서 한 번만 알린다.
- 화면이 사라지면(파이프가 끊김) 이번 세대를 마치고 체크포인트를 남긴 뒤 끝난다. 워커는 이 프로세스가 끝나면
  스스로 끝난다.
"""

from __future__ import annotations

import queue
import signal
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from evolab.app import strings as S

RUNNING = "running"
PAUSED = "paused"
FINISHED = "finished"

PACE_SKIP: Final = "skip"
"""관전 방식 「학습 먼저」(v6.0 D23): 학습은 재생을 기다리지 않고, 화면은 그때까지 끝난 가장 새 세대를 재생한다."""
PACE_EVERY: Final = "every"
"""관전 방식 「세대마다 다 보기」(v1.0~v3.1 동작): 화면이 세대 N-1 재생을 끝내기 전에는 N+1로 가지 않는다."""
PACES: Final = (PACE_SKIP, PACE_EVERY)


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
    info: dict[str, Any] = field(default_factory=dict)
    """새 실행의 run.json에 더 넣을 것(예: 반복 실행 묶음 {"group": {...}}, 메모)."""
    pace: str = PACE_EVERY
    """관전 방식(PACES). 기본은 예전 동작(데이터클래스 기본값, v6 설계 7.3). 실험실은 칸 값(lab/pace, PACE_DEFAULT)을 넘긴다.
    모르는 값이면 PACE_EVERY."""


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
        self.pace = request.pace if request.pace in PACES else PACE_EVERY
        self.step_once = False
        self.acked = 0
        self._waited = 0.0
        """다음 세대를 계산하기 전에 재생을 기다린 초(세대 메시지 "waited"로 보내고 0으로)."""
        self._wait_since: float | None = None
        """지금 재생을 기다리기 시작한 시각(time.monotonic). 기다리지 않으면 None."""
        self.pool: Any = None
        self.store: Any = None
        self.trainer: Any = None
        self.stop_logged = False
        self.change_saved = False
        """설정 변경·환경 교체 중에 바뀐 상태의 체크포인트를 쓰기 시작했는지(실패하면 되돌려 써야 한다)."""
        self.observe_sent: str | None = None
        """화면에 마지막으로 알린 관찰 기록 실패 글(trainer.observe_problem). 글이 바뀔 때만(더 무거운 실패) 다시 알린다."""

    # ------------------------------------------------------------------
    def main(self) -> None:
        threading.Thread(target=self._read, name="commands", daemon=True).start()
        self._outbox: queue.Queue[Any] = queue.Queue()
        self._writer = threading.Thread(target=self._write, name="events", daemon=True)
        self._writer.start()
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

    def _write(self) -> None:
        """결과 보내기는 스레드가 한다: 화면이 죽거나 멈추면 윈도우에서 쓰기가 끝나지 않을 수 있어서."""
        while True:
            msg = self._outbox.get()
            if msg is None:
                return
            try:
                self.event_conn.send(msg)
            except (OSError, ValueError, EOFError, TypeError):
                self.gui_gone = True
                self.cmds.put({"type": "quit", "gone": True})
                return

    def _send(self, msg: dict[str, Any]) -> None:
        if not self.gui_gone:
            self._outbox.put(msg)

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
            self.store = create_run(Path(req.runs_dir), req.env_text, spec, config, workers, extra=dict(req.info))
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
        self._outbox.put(None)
        if not self.gui_gone:
            self._writer.join(timeout=10)  # 남은 결과를 다 보낼 때까지(화면이 받고 있으면 금방 끝난다)
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
            self._note_wait()
            self._drain(block=idle)
            self._note_wait()
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
        return self.spectate and self.pace == PACE_EVERY and self.acked < self.trainer.state.generation - 1

    def _note_wait(self) -> None:
        """재생을 기다린 시간을 센다: 학습 중(RUNNING)이고 _must_wait인 동안만. 멈춤·끝남은 기다림이 아니다."""
        waiting = self.state == RUNNING and self._must_wait()
        now = time.monotonic()
        if waiting and self._wait_since is None:
            self._wait_since = now
        elif not waiting and self._wait_since is not None:
            self._waited += now - self._wait_since
            self._wait_since = None

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
            pace = msg.get("pace")
            if pace in PACES:  # 없으면 지금 값 그대로, 모르는 값은 무시(설계 4.12)
                self.pace = str(pace)
            self.acked = self.trainer.state.generation
        elif kind == "played":
            self.acked = min(self.trainer.state.generation, max(self.acked, int(msg.get("generation", 0))))
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
        snap = t.changeable_state()
        # 변경은 다 기록하고 적용하거나, 하나도 남기지 않는다. 순서: 트레이너에 적용 → 체크포인트(바뀐 상태) → 보호 표시
        # → run.json 환경 버전 → events.jsonl 사건 줄(마지막). 중간에 실패하면 트레이너와 이미 쓴 것을 되돌린다:
        # 기록과 학습이 어긋나면 이어하기·재생·재현 묶음이 실제로 학습한 것과 달라진다.
        before = self._change_records()
        self.change_saved = False
        try:
            if what == "config":
                changes = t.apply_config(dict(msg.get("changes") or {}))
                if changes:
                    self._mark_change_point()
                    self.store.append_event("config_change", gen, changes=changes)
                reply = {"config": t.config.to_dict(), "changes": changes}
            else:
                text = str(msg.get("text", ""))
                old = t.env_text
                t.replace_env(text)
                if text != old:
                    self._mark_change_point()
                    vid = self.store.add_env_version(text, gen)
                    self.store.append_event("env_change", gen, version=vid)
                reply = {"env_text": t.env_text, "env_version": t.env_version}
        except (TrainerChangeError, EnvFileError, ValueError, OSError) as exc:
            t.restore_changeable(snap)
            self._undo_change_records(before)
            self._send({"type": "applied", "what": what, "ok": False, "message": _reason(exc)})
            return
        self._send({"type": "applied", "what": what, "ok": True, "generation": gen, **reply})

    def _mark_change_point(self) -> None:
        """바뀐 설정·환경이 담긴 체크포인트를 남기고 정리 대상에서 뺀다(0세대면 체크포인트가 없다).

        change_saved = True는 체크포인트를 쓰기 전에 둔다(뒤에서 실패해 되돌릴 때 바꾸기 전 상태로 다시 써야 한다).
        save_checkpoint는 파일을 바꿔 넣은 뒤 정리 규칙을 적용하다가도(run.json 읽기) 실패할 수 있어서, 저장이 끝난
        뒤에 표시하면 바뀐 설정·환경이 담긴 체크포인트가 그대로 남는다. 파일을 바꾸기 전에 실패했어도 바꾸기 전
        상태(실제로 학습한 설정·환경)로 다시 쓰는 것은 해가 없다.
        """
        gen = self.trainer.state.generation
        if gen:
            self.change_saved = True
            self.trainer.checkpoint()
            self.store.protect(gen)

    def _change_records(self) -> dict[str, Any] | None:
        """변경을 되돌릴 때 쓰는 run.json 기록(환경 버전, 보호 목록). 읽을 수 없으면 None(되돌릴 것도 쓰지 못한다)."""
        try:
            info = self.store.read_run_info()
        except (OSError, ValueError):
            return None
        return {"env_versions": list(info.get("env_versions", [])),
                "protected_checkpoints": list(info.get("protected_checkpoints", []))}

    def _undo_change_records(self, before: dict[str, Any] | None) -> None:
        """실패한 변경이 이미 남긴 것을 될 수 있는 만큼 되돌린다(트레이너는 이미 되돌렸다).

        - 바뀐 상태로 썼을 수 있는 체크포인트(쓰기를 시작했으면)는 되돌린 트레이너 상태로 다시 쓴다(같은 세대,
          바꾸기 전 설정·환경 버전).
        - run.json의 환경 버전과 보호 목록은 바꾸기 전으로(다른 키는 건드리지 않는다: 화면이 메모를 쓸 수 있다).
        - events.jsonl 사건 줄은 맨 마지막에 쓰므로 실패했으면 남지 않는다(잘린 줄은 다음 덧붙이기 때 잘라 낸다).
        되돌리기마저 실패하면(디스크 가득 참 등) 더 할 수 있는 것이 없어 그대로 둔다.
        """
        try:
            if self.change_saved:
                self.trainer.checkpoint()
            if before is not None:
                info = self.store.read_run_info()
                restore = {k: v for k, v in before.items() if list(info.get(k, [])) != v}
                if restore:
                    self.store.update_run_info(**restore)
        except (OSError, ValueError):
            pass

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
        self._waited, self._wait_since = 0.0, None
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
            "observation": self.trainer.last_observation,
            "waited": self._take_waited(),
        })
        problem = getattr(self.trainer, "observe_problem", None)
        if problem and problem != self.observe_sent:
            self.observe_sent = problem
            # notice: 학습은 그대로 이어지므로 화면은 막는 상자 대신 상태 글로 알린다(설계 D9)
            self._send({"type": "error", "fatal": False, "notice": True, "message": problem})


    def _take_waited(self) -> float:
        waited, self._waited = self._waited, 0.0
        return float(waited)


def _reason(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or type(exc).__name__

"""화면 쪽 학습 프로세스 연결. 학습 프로세스를 띄우고, 명령을 보내고, 결과를 Qt 신호로 바꾼다.

결과 파이프는 화면 스레드의 타이머가 짧게 자주 읽는다(학습 프로세스는 세대마다 몇 개만 보낸다). 화면이 읽지
않으면 윈도우에서는 학습 프로세스의 보내기가 기다리므로, 읽기는 화면이 바빠도 밀리지 않게 타이머 하나로 한다.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from evolab.app import strings as S
from evolab.engine.session import SessionRequest, session_main

POLL_MS = 15
MAX_PER_TICK = 32


class TrainingClient(QObject):
    started = Signal(dict)
    generation = Signal(dict)
    state_changed = Signal(dict)
    applied = Signal(dict)
    rewound = Signal(dict)
    failed = Signal(str, bool)
    """(한국어 메시지, 학습 프로세스가 끝났는지)"""
    closed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._proc: Any = None
        self._cmd: Any = None
        self._events: Any = None
        self._closed_sent = True
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)

    @property
    def active(self) -> bool:
        return self._proc is not None

    def start(self, request: SessionRequest) -> None:
        if self._proc is not None:
            raise RuntimeError("session already running")
        ctx = mp.get_context("spawn")
        cmd_r, cmd_w = ctx.Pipe(duplex=False)
        ev_r, ev_w = ctx.Pipe(duplex=False)
        # 학습 프로세스는 워커를 띄우므로 daemon이 아니다(daemon 프로세스는 자식을 만들 수 없다).
        proc = ctx.Process(target=session_main, args=(cmd_r, ev_w, request), daemon=False, name="evolab-training")
        proc.start()
        cmd_r.close()
        ev_w.close()
        self._proc, self._cmd, self._events = proc, cmd_w, ev_r
        self._closed_sent = False
        self._timer.start()

    def send(self, msg: dict[str, Any]) -> bool:
        if self._cmd is None:
            return False
        try:
            self._cmd.send(msg)
            return True
        except (OSError, ValueError):
            return False

    # ------------------------------------------------------------------
    def _poll(self) -> None:
        if self._events is None:
            return
        for _ in range(MAX_PER_TICK):
            try:
                if not self._events.poll():
                    break
                msg = self._events.recv()
            except (EOFError, OSError):
                self._lost()
                return
            self._dispatch(msg)
            if self._events is None:
                return
        if self._proc is not None and not self._proc.is_alive() and not self._safe_poll():
            self._lost()

    def _safe_poll(self) -> bool:
        try:
            return bool(self._events is not None and self._events.poll())
        except (EOFError, OSError):
            return False

    def _dispatch(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "started":
            self.started.emit(msg)
        elif kind == "generation":
            self.generation.emit(msg)
        elif kind == "state":
            self.state_changed.emit(msg)
        elif kind == "applied":
            self.applied.emit(msg)
        elif kind == "rewind":
            self.rewound.emit(msg)
        elif kind == "error":
            self.failed.emit(str(msg.get("message", "")), bool(msg.get("fatal")))
        elif kind == "closed":
            self._finish()

    def _lost(self) -> None:
        code = self._proc.exitcode if self._proc is not None else None
        was_open = not self._closed_sent
        self._finish()
        if was_open and code not in (0, None):
            self.failed.emit(S.SESSION_LOST.format(code=code), True)

    def _finish(self) -> None:
        self._timer.stop()
        for c in (self._cmd, self._events):
            if c is not None:
                try:
                    c.close()
                except OSError:
                    pass
        proc = self._proc
        self._proc = self._cmd = self._events = None
        if proc is not None:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        if not self._closed_sent:
            self._closed_sent = True
            self.closed.emit()

    def shutdown(self, timeout: float = 20.0) -> None:
        """학습 프로세스에 끝내라고 하고(이번 세대를 마치고 체크포인트를 남긴다) 끝날 때까지 기다린다."""
        if self._proc is None:
            return
        self.send({"type": "quit"})
        deadline = time.monotonic() + timeout
        while self._proc is not None and time.monotonic() < deadline:
            try:
                if self._events.poll(0.05):
                    msg = self._events.recv()
                    if msg.get("type") == "closed":
                        break
                    self._dispatch(msg)
                elif not self._proc.is_alive():
                    break
            except (EOFError, OSError):
                break
        self._finish()

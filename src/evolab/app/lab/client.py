"""화면 쪽 학습 프로세스 연결. 학습 프로세스를 띄우고, 명령을 보내고, 결과를 Qt 신호로 바꾼다.

결과 파이프는 받는 스레드가 계속 읽어 큐에 넣고, 화면 스레드의 타이머가 큐를 비운다. 세대 결과(관전 궤적 포함
약 1MB)는 파이프로 여러 번 나뉘어 오므로, 화면 스레드에서 직접 받으면 다 올 때까지 창이 멈춘다(부하가 큰
PC에서 70~110ms를 쟀다). 받는 스레드는 기다리는 동안 GIL을 놓는다.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import multiprocessing.util  # noqa: F401  아래 atexit 순서 때문에 먼저 불러 둔다
import queue
import threading
import time
import weakref
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from evolab.app import strings as S
from evolab.engine.session import SessionRequest, session_main

POLL_MS = 15
MAX_PER_TICK = 32


_LIVE: weakref.WeakSet[TrainingClient] = weakref.WeakSet()


def _close_all_at_exit() -> None:
    """파이썬이 끝날 때 남은 학습 프로세스를 정리한다.

    multiprocessing은 끝날 때 daemon이 아닌 자식(학습 프로세스)을 끝없이 기다리는데, 학습 프로세스는 화면 쪽
    파이프가 닫히기를 기다리므로 서로 막힌다(예: 창이 예외로 닫히거나 테스트가 정리 없이 실패). atexit는 등록한
    역순으로 부르므로, multiprocessing.util(끝날 때 자식을 기다리는 함수를 등록한다)을 먼저 불러 두면 이 함수가
    먼저 돈다.
    """
    for client in list(_LIVE):
        client.abandon()


atexit.register(_close_all_at_exit)


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
        self._inbox: queue.Queue[Any] = queue.Queue()
        self._outbox: queue.Queue[Any] = queue.Queue()
        self._reader: threading.Thread | None = None
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
        self._inbox = queue.Queue()
        self._reader = threading.Thread(target=self._read, args=(ev_r, self._inbox), name="training-events", daemon=True)
        self._reader.start()
        # 보내기도 스레드로: 윈도우에서 학습 프로세스가 죽거나 바쁘면 큰 메시지(환경 파일) 쓰기가 끝나지 않을 수 있다.
        self._outbox = queue.Queue()
        threading.Thread(target=self._write, args=(cmd_w, self._outbox), name="training-commands", daemon=True).start()
        self._closed_sent = False
        self._timer.start()
        _LIVE.add(self)

    def abandon(self, timeout: float = 30.0) -> None:
        """파이프만 닫고 학습 프로세스가 스스로 끝나기를 기다린다(이번 세대를 마치고 체크포인트를 남긴다)."""
        proc = self._proc
        if proc is None:
            return
        self._outbox.put(_EOF)
        for c in (self._cmd, self._events):
            if c is not None:
                try:
                    c.close()
                except OSError:
                    pass
        self._cmd = self._events = None
        proc.join(timeout=timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        self._proc = None
        _LIVE.discard(self)

    @staticmethod
    def _write(conn: Any, outbox: queue.Queue[Any]) -> None:
        while True:
            msg = outbox.get()
            if msg is _EOF:
                return
            try:
                conn.send(msg)
            except (EOFError, OSError, TypeError, ValueError):
                return

    @staticmethod
    def _read(conn: Any, inbox: queue.Queue[Any]) -> None:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError, TypeError, ValueError):  # 끊김, 또는 다른 스레드가 파이프를 닫음
                inbox.put(_EOF)
                return
            inbox.put(msg)
            if isinstance(msg, dict) and msg.get("type") == "closed":
                return

    def send(self, msg: dict[str, Any]) -> bool:
        """명령을 보낸다(기다리지 않는다: 보내는 스레드가 쓴다)."""
        if self._cmd is None:
            return False
        self._outbox.put(msg)
        return True

    # ------------------------------------------------------------------
    def _poll(self) -> None:
        if self._events is None:
            return
        for _ in range(MAX_PER_TICK):
            try:
                msg = self._inbox.get_nowait()
            except queue.Empty:
                break
            if msg is _EOF:
                self._lost()
                return
            self._dispatch(msg)
            if self._events is None:
                return
        if self._proc is not None and not self._proc.is_alive() and self._inbox.empty() and (
            self._reader is None or not self._reader.is_alive()
        ):
            self._lost()

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
        self._outbox.put(_EOF)
        for c in (self._cmd, self._events):
            if c is not None:
                try:
                    c.close()
                except OSError:
                    pass
        proc = self._proc
        self._proc = self._cmd = self._events = None
        _LIVE.discard(self)
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
                msg = self._inbox.get(timeout=0.05)
            except queue.Empty:
                if not self._proc.is_alive() and (self._reader is None or not self._reader.is_alive()):
                    break
                continue
            if msg is _EOF or msg.get("type") == "closed":
                break
            self._dispatch(msg)
        self._finish()


_EOF = object()
"""받는 스레드가 파이프가 끊긴 것을 알리는 표지."""

"""내보내기 자식 프로세스 다루기 (v3 설계 D49·D51, 2.6, 4.14, 7.3 「ExportRunner」).

- ExportRunner: 화면 프로세스가 작업 하나를 spawn 자식(worker.export_main)에게 맡기고 15ms 타이머로 진행 메시지만 읽는다
  (학습 클라이언트와 같은 방식). 신호: progressed(k, n, 지금까지 바이트), noted(글), finished(done 메시지 사전), failed(까닭,
  자세히), cancelled(). 취소 = 「cancel」을 보내고 5초 안에 끝나지 않으면 자식을 끝내고 임시 파일을 지운다. 자식이 「done」 없이
  끝나면 EXP_PROCESS_DIED(종료 코드 풀이, MP4면 「GIF로 저장해 보세요.」를 붙임). 자식의 error·note는 errors.jsonl
  (kind="export")에 남긴다. 취소를 보낸 뒤라도 「done」이 오면(파일을 이미 바꾼 뒤의 늦은 취소) finished.
- 되돌릴 지점 10(설계 7.4): 자식이 Qt를 띄웠다는 첫 진행 메시지(k = 0)를 보내기 전에 끝나면(묶인 exe에서 spawn 자식이 Qt를 못
  띄우는 경우) 같은 프로세스에서 지금 GIF처럼(진행 창 + processEvents) 다시 만들고 Media Foundation은 끈다(내장 쓰기만).
  CHILD_PROCESS = False면 처음부터 같은 프로세스에서 만든다.
- EncoderProbe: 대화 상자가 열 때 한 번, MP4 인코더를 시험한다(윈도우이고 MF를 켰으면 작은 자식 worker.probe_main, 아니면
  이 프로세스에서 바로: 리눅스·MF 끔은 DLL을 열지 않는다). 결과는 화면 프로세스가 살아 있는 동안 기억한다.
- run_in_child: Qt 없이 자식으로 작업 하나를 끝까지(자체 점검 「내보내기」: 묶인 exe의 spawn 자식이 Qt를 띄우는지도 본다).
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from PySide6.QtCore import QObject, QTimer, Signal

from evolab.app import strings as S
from evolab.app.export import worker
from evolab.app.export.job import ExportItem, ExportJob

CHILD_PROCESS: Final = True
"""False면 내보내기를 화면 프로세스에서 만든다(MF 끔, 설계 7.4 되돌릴 지점 10)."""
POLL_MS: Final = 15
"""자식 메시지를 읽는 간격(학습 클라이언트와 같음)."""
CANCEL_GRACE_S: Final = 5.0
"""취소를 보낸 뒤 자식이 스스로 끝나기를 기다리는 시간(설계 2.6). 넘으면 끝낸다."""
PROBE_TIMEOUT_S: Final = 30.0
"""인코더 시험 자식을 기다리는 한도. 넘으면 내장 쓰기로 본다."""
CHILD_TIMEOUT_S: Final = 600.0
"""run_in_child의 기본 한도(자체 점검)."""


class ExportFailed(RuntimeError):
    """자식이 작업을 끝내지 못했다(run_in_child). message = 한국어 까닭, detail = traceback 또는 종료 코드."""

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


def light_job(job: ExportJob) -> ExportJob:
    """자식에게 넘길 작업: 재생의 추적 자료(신경망 입력·층별 활성화·센서 원래 값·레이)를 뺀다. 그리기와 폰 파일은 궤적·세계·
    모양·결과만 쓰므로 그림은 같고 pickle이 작아진다(실험실 재생은 뇌 그림용 추적을 함께 가질 수 있다)."""
    items = []
    changed = False
    for it in job.items:
        rep = it.replay
        if any(getattr(rep, name, None) is not None for name in ("observations", "activations", "raw", "rays")):
            rep = dataclasses.replace(rep, observations=None, activations=None, raw=None, rays=None)
            changed = True
        items.append(ExportItem(rep, it.color, it.label, it.view) if rep is not it.replay else it)
    return dataclasses.replace(job, items=tuple(items)) if changed else job


def died_text(code: int | None, fmt: str | None = None) -> str:
    """EXP_PROCESS_DIED(종료 코드 풀이, 모르면 EXIT_CODE_UNKNOWN). MP4 작업(fmt="mp4")이면 「GIF로 저장해 보세요.」를 붙인다
    (EXP_PROCESS_DIED_GIF_HINT). GIF·폰 파일 작업과 인코더 시험 까닭(fmt 없음)에는 붙이지 않는다."""
    from evolab.diagnostics import exit_code_text

    text = S.EXP_PROCESS_DIED.format(meaning=exit_code_text(code) or S.EXIT_CODE_UNKNOWN)
    return f"{text} {S.EXP_PROCESS_DIED_GIF_HINT}" if fmt == "mp4" else text


def _record(message: str, detail: str = "") -> None:
    from evolab import diagnostics

    diagnostics.record("export", message, detail)


def _start_child(target: Callable[..., None], *args: Any) -> tuple[Any, Any]:
    """spawn 자식 하나와 화면 쪽 연결. 자식 쪽 연결은 화면에서 바로 닫는다(자식이 끝나면 화면의 recv가 EOFError)."""
    ctx = mp.get_context("spawn")
    here, there = ctx.Pipe(duplex=True)
    proc = ctx.Process(target=target, args=(*args[:1], there, *args[1:]), daemon=True, name="evolab-export")
    try:
        proc.start()
    except BaseException:
        here.close()
        raise
    finally:
        there.close()
    return proc, here


class ExportRunner(QObject):
    """작업 하나를 자식 프로세스로(설계 4.14). start → progressed… → finished | failed | cancelled 중 하나가 꼭 한 번."""

    progressed = Signal(int, int, int)
    """(그린 프레임 수 k, 전체 n, 지금까지 쓴 바이트)."""
    noted = Signal(str)
    """알림 한 줄(EXP_MF_FALLBACK 등)."""
    finished = Signal(dict)
    """done 메시지 내용 {"path", "frames", "bytes", "encoder"}."""
    failed = Signal(str, str)
    """(한국어 까닭, 자세히: traceback 또는 종료 코드)."""
    cancelled = Signal()
    """취소가 끝났다(임시 파일 없음, 원래 파일 그대로). 7.3 서명에 더한 신호."""

    def __init__(self, parent: QObject | None = None, *, use_child: bool | None = None) -> None:
        super().__init__(parent)
        self.use_child = CHILD_PROCESS if use_child is None else bool(use_child)
        self.mode = ""
        """"child" 또는 "inprocess"(지금 작업을 어디서 만드나)."""
        self.fell_back = False
        """자식이 Qt를 띄우기 전에 끝나 같은 프로세스로 다시 만들었다(되돌릴 지점 10)."""
        self.encoder_hint = ""
        """MP4를 내장 쓰기로 만드는 중이라고 알게 됐으면 "builtin"(MF 대체 note, 같은 프로세스 길)."""
        self.exit_code: int | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._proc: Any = None
        self._conn: Any = None
        self._job: ExportJob | None = None
        self._out: Path | None = None
        self._started = False
        self._cancel_at: float | None = None
        self._cancel_flag = False
        self._over = True
        self._in_poll = False

    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return not self._over

    def start(self, job: ExportJob, out: Path) -> None:
        """작업을 시작한다(돌아온 뒤 신호가 온다). 이미 돌고 있으면 RuntimeError."""
        if self.running:
            raise RuntimeError("an export is already running")
        self._job, self._out = light_job(job), Path(out)
        self._started = False
        self._cancel_at = None
        self._cancel_flag = False
        self.fell_back = False
        self.encoder_hint = ""
        self.exit_code = None
        self._over = False
        if not self.use_child:
            self._begin_inprocess()
            return
        self.mode = "child"
        try:
            self._proc, self._conn = _start_child(worker.export_main, self._job, str(self._out))
        except Exception as exc:  # noqa: BLE001 - 자식을 못 띄움(pickle·프로세스 수 등): 같은 프로세스로
            _record(S.EXP_INPROCESS_NOTE, f"{type(exc).__name__}: {exc}")
            self.fell_back = True
            self._begin_inprocess()
            return
        self._timer.start()

    def cancel(self) -> None:
        """그만 만들기. 자식이면 「cancel」을 보내고 5초 안에 안 끝나면 끝낸다. 끝나면 cancelled(또는 그 사이 끝났으면 finished)."""
        if not self.running:
            return
        self._cancel_flag = True
        if self.mode != "child" or self._cancel_at is not None:
            return
        self._cancel_at = time.monotonic() + CANCEL_GRACE_S
        try:
            self._conn.send(worker.CANCEL)
        except (OSError, ValueError):
            pass

    def wait(self, timeout: float = 60.0) -> bool:
        """끝날 때까지 이벤트를 돌리며 기다린다(테스트·자체 점검용). 끝났으면 True."""
        from PySide6.QtCore import QCoreApplication

        deadline = time.monotonic() + timeout
        while self.running and time.monotonic() < deadline:
            QCoreApplication.processEvents()
            time.sleep(0.005)
        return not self.running

    # ------------------------------------------------------------------
    # 자식
    # ------------------------------------------------------------------
    def _poll(self) -> None:
        if self._over:
            self._timer.stop()
            return
        if self._in_poll:  # 진행 신호를 받은 쪽이 이벤트를 돌리면(모달 진행 창의 setValue) 다시 들어오지 않게
            return
        self._in_poll = True
        try:
            self._poll_once()
        finally:
            self._in_poll = False

    def _poll_once(self) -> None:
        eof = False
        try:
            while self._conn is not None and self._conn.poll():
                if self._handle(self._conn.recv()):
                    return
        except (EOFError, OSError):
            eof = True
        if self._over or self._proc is None:
            return
        proc = self._proc
        alive = proc.is_alive()
        if self._cancel_at is not None and alive and time.monotonic() >= self._cancel_at:
            proc.terminate()
            proc.join(2.0)
            alive = proc.is_alive()
            if alive:
                proc.kill()
                proc.join(2.0)
            self._end_cancelled()
            return
        if alive and not eof:
            return
        proc.join(1.0)
        self.exit_code = proc.exitcode
        if self._cancel_flag:
            self._end_cancelled()
        elif not self._started:
            # 되돌릴 지점 10: 자식이 Qt를 띄우기 전에 끝났다 → 같은 프로세스에서(MF 끔)
            _record(S.EXP_INPROCESS_NOTE, f"exit code {self.exit_code}")
            self._close_child()
            self.fell_back = True
            self._begin_inprocess()
        else:
            message = died_text(self.exit_code, self._job.fmt if self._job is not None else None)
            detail = f"exit code {self.exit_code}"
            _record(message, detail)
            self._close_child()
            self._cleanup_temp()
            self._finish(lambda: self.failed.emit(message, detail))

    def _handle(self, msg: Any) -> bool:
        """메시지 하나. 작업이 끝났으면(done·error) True."""
        if not isinstance(msg, dict):
            return False
        kind = msg.get("type")
        if kind == "frames":
            k = int(msg.get("k", 0))
            if k == worker.STARTED_K:
                self._started = True
            self.progressed.emit(k, int(msg.get("n", 0)), int(msg.get("bytes", 0)))
        elif kind == "note":
            text = str(msg.get("text", ""))
            self.encoder_hint = "builtin"
            _record(text)
            self.noted.emit(text)
        elif kind == "done":
            done = {k: v for k, v in msg.items() if k != "type"}
            self._close_child(join=0.5)
            self._finish(lambda: self.finished.emit(done))
            return True
        elif kind == "error":
            message, detail = str(msg.get("message", "")), str(msg.get("detail", ""))
            _record(message, detail)
            self._close_child(join=0.5)
            self._cleanup_temp()
            self._finish(lambda: self.failed.emit(message, detail))
            return True
        return False

    def _end_cancelled(self) -> None:
        self._close_child()
        self._cleanup_temp()
        self._finish(self.cancelled.emit)

    def _close_child(self, join: float = 0.0) -> None:
        self._timer.stop()
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
        if self._proc is not None:
            if join:
                self._proc.join(join)
            if self._proc.exitcode is not None:
                self.exit_code = self._proc.exitcode
        self._conn = None
        self._proc = None

    def _cleanup_temp(self) -> None:
        if self._job is not None and self._out is not None:
            try:
                worker.temp_path_for(self._job, self._out).unlink(missing_ok=True)
            except OSError:
                pass

    def _finish(self, emit: Callable[[], None]) -> None:
        self._timer.stop()
        self._over = True
        emit()

    # ------------------------------------------------------------------
    # 같은 프로세스(되돌릴 지점 10)
    # ------------------------------------------------------------------
    def _begin_inprocess(self) -> None:
        self.mode = "inprocess"
        assert self._job is not None
        if self._job.fmt == "mp4":
            self._job = dataclasses.replace(self._job, encoder="builtin")  # 같은 프로세스에서는 MF를 쓰지 않는다
            self.encoder_hint = "builtin"
        QTimer.singleShot(0, self._run_inprocess)

    def _run_inprocess(self) -> None:
        from PySide6.QtCore import QCoreApplication

        job, out = self._job, self._out
        assert job is not None and out is not None

        def progress(msg: dict[str, Any]) -> bool:
            if msg.get("type") == "frames":
                self.progressed.emit(int(msg["k"]), int(msg["n"]), int(msg["bytes"]))
            elif msg.get("type") == "note":
                self.noted.emit(str(msg.get("text", "")))
            QCoreApplication.processEvents()
            return not self._cancel_flag

        self.progressed.emit(worker.STARTED_K, worker.progress_total(job), 0)
        try:
            done = worker.run_job(job, out, progress)
        except worker.ExportCancelled:
            self._finish(self.cancelled.emit)
        except Exception as exc:  # noqa: BLE001 - 화면에 알린다(errors.jsonl에도)
            import traceback

            message, detail = worker.error_message(exc), traceback.format_exc()
            _record(message, detail)
            self._finish(lambda: self.failed.emit(message, detail))
        else:
            self._finish(lambda: self.finished.emit(done))


# ---------------------------------------------------------------------------
# 인코더 시험
# ---------------------------------------------------------------------------
_PROBED: tuple[str, str] | None = None
"""이 프로세스에서 한 번 시험한 결과 (kind, detail)."""


def probe_needs_child() -> bool:
    """MF 시험을 자식에서 해야 하나(윈도우이고 MF를 켰을 때만 DLL을 연다)."""
    from evolab.app.export import mp4

    return sys.platform == "win32" and bool(mp4.MF_ENABLED)


def probe_in_process() -> tuple[str, str]:
    from evolab.app.export.mp4 import encoder_status

    status = encoder_status()
    return status.kind, status.detail


def probe_in_child(timeout: float = PROBE_TIMEOUT_S) -> tuple[str, str]:
    """작은 자식으로 MP4 인코더를 시험한다(Qt 없이 기다림). 자식이 죽거나 시간이 넘으면 ("builtin", 까닭)."""
    from evolab.app.export import mp4

    fallback = "builtin" if mp4.BUILTIN_ENABLED else "none"
    try:
        proc, conn = _start_child(worker.probe_main)
    except Exception as exc:  # noqa: BLE001
        return fallback, f"{type(exc).__name__}: {exc}"
    try:
        if conn.poll(timeout):
            got = conn.recv()
            if isinstance(got, tuple) and len(got) == 2:
                return str(got[0]), str(got[1])
        return fallback, died_text(proc.exitcode) if not proc.is_alive() else f"timeout {timeout:.0f}s"
    except (EOFError, OSError):
        proc.join(2.0)
        return fallback, died_text(proc.exitcode)
    finally:
        conn.close()
        proc.join(2.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(2.0)


def probed() -> tuple[str, str] | None:
    """이미 시험했으면 그 결과."""
    return _PROBED


class EncoderProbe(QObject):
    """MP4 인코더 시험(대화 상자가 열 때 한 번, 결과는 프로세스마다 기억). done(kind, detail)이 꼭 한 번 온다."""

    done = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)
        self._proc: Any = None
        self._conn: Any = None
        self._deadline = 0.0

    def start(self) -> None:
        global _PROBED
        if _PROBED is not None:
            got = _PROBED
            QTimer.singleShot(0, lambda: self.done.emit(*got))
            return
        if not probe_needs_child():
            _PROBED = probe_in_process()
            got = _PROBED
            QTimer.singleShot(0, lambda: self.done.emit(*got))
            return
        try:
            self._proc, self._conn = _start_child(worker.probe_main)
        except Exception as exc:  # noqa: BLE001
            self._end(("builtin", f"{type(exc).__name__}: {exc}"))
            return
        self._deadline = time.monotonic() + PROBE_TIMEOUT_S
        self._timer.start()

    def _poll(self) -> None:
        got: tuple[str, str] | None = None
        try:
            if self._conn.poll():
                msg = self._conn.recv()
                if isinstance(msg, tuple) and len(msg) == 2:
                    got = (str(msg[0]), str(msg[1]))
        except (EOFError, OSError):
            self._proc.join(1.0)
            got = ("builtin", died_text(self._proc.exitcode))
        if got is None and not self._proc.is_alive() and not self._conn.poll():
            got = ("builtin", died_text(self._proc.exitcode))
        if got is None and time.monotonic() > self._deadline:
            got = ("builtin", f"timeout {PROBE_TIMEOUT_S:.0f}s")
        if got is not None:
            self._end(got)

    def _end(self, got: tuple[str, str]) -> None:
        global _PROBED
        from evolab.app.export import mp4

        if got[0] == "builtin" and not mp4.BUILTIN_ENABLED:
            got = ("none", got[1])
        self._timer.stop()
        if self._conn is not None:
            self._conn.close()
        if self._proc is not None:
            self._proc.join(0.5)
            if self._proc.is_alive():
                self._proc.terminate()
        self._conn = self._proc = None
        _PROBED = got
        self.done.emit(*got)


# ---------------------------------------------------------------------------
# Qt 없이 자식으로 하나(자체 점검)
# ---------------------------------------------------------------------------
def run_in_child(job: ExportJob, out: Path, *, timeout: float = CHILD_TIMEOUT_S,
                 on_message: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """자식 프로세스로 작업 하나를 끝까지 만든다(이벤트 루프 없이 기다림). done 메시지 내용을 돌려준다. 자식의 error, 「done」
    없이 끝남, 시간 초과는 ExportFailed. 첫 메시지(k = 0, 자식이 Qt를 띄움)까지 걸린 초는 결과의 "first_message_s"."""
    t0 = time.monotonic()
    proc, conn = _start_child(worker.export_main, light_job(job), str(out))
    first: float | None = None
    try:
        deadline = t0 + timeout
        while True:
            try:
                if conn.poll(0.05):
                    msg = conn.recv()
                    if on_message is not None:
                        on_message(msg)
                    kind = msg.get("type") if isinstance(msg, dict) else None
                    if kind == "frames" and first is None:
                        first = time.monotonic() - t0
                    elif kind == "done":
                        done = {k: v for k, v in msg.items() if k != "type"}
                        done["first_message_s"] = first
                        return done
                    elif kind == "error":
                        raise ExportFailed(str(msg.get("message", "")), str(msg.get("detail", "")))
                    continue
            except (EOFError, OSError):
                proc.join(5.0)
                raise ExportFailed(died_text(proc.exitcode, job.fmt), f"exit code {proc.exitcode}") from None
            if not proc.is_alive() and not conn.poll():
                raise ExportFailed(died_text(proc.exitcode, job.fmt), f"exit code {proc.exitcode}")
            if time.monotonic() > deadline:
                raise ExportFailed(S.SELFTEST_ERR_TIMEOUT.format(seconds=int(timeout)))
    finally:
        conn.close()
        proc.join(5.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(2.0)
            worker.temp_path_for(job, Path(out)).unlink(missing_ok=True)


__all__ = [
    "CANCEL_GRACE_S", "CHILD_PROCESS", "EncoderProbe", "ExportFailed", "ExportRunner", "died_text", "light_job",
    "probe_in_child", "probe_in_process", "probe_needs_child", "probed", "run_in_child",
]

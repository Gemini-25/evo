"""내보내기 실행: 작업 하나를 파일로 (v3 설계 D49·D51, 2.6, 4.14, 7.3 「app/export/worker.py (F)」).

- run_job(job, out, progress): 프레임을 그려(frames.iter_frames) 형식별로 쓴다. MP4는 mp4.write_mp4(Media Foundation이 쓰는 중
  실패하면 내장 쓰기로 처음부터 + note), GIF는 export.video의 GIF 작성기, 폰 재생 파일은 phone.write_phone_html(끝 장면 그림 =
  마지막 프레임). 쓰는 동안은 임시 파일, 끝나면 이름을 바꾼다. 취소·오류면 임시 파일을 지우고 같은 이름의 원래 파일은 그대로
  둔다. 명령줄(`evolab export`)은 자기 프로세스에서, 화면은 아래 자식 프로세스에서 같은 함수를 부른다.
- export_main(job, conn, out): 내보내기 자식 프로세스(spawn)의 진입점. 화면이 이미 가진 재생을 pickle로 받아 Qt를 띄우고
  run_job을 돌리며 2.6의 메시지를 보낸다: {"type": "frames", "k", "n", "bytes"}(k = 0은 「Qt를 띄웠고 시작함」),
  {"type": "note", "text"}, {"type": "done", "path", "frames", "bytes", "encoder"}, {"type": "error", "message", "detail"}.
  화면이 "cancel"을 보내거나 연결이 끊기면(화면이 죽음) 임시 파일을 지우고 아무 메시지 없이 끝낸다. 다만 파일을 이미 최종
  이름으로 바꾼 뒤(run_job이 돌아온 뒤)에 온 취소는 늦은 것이다: 원래 파일은 이미 바뀌었으므로 「done」을 그대로 보낸다
  (_Link.send_final, 화면은 취소 뒤에 온 done을 finished로 받는다). 그래서 폰 파일을 쓰는 사이에 취소가 와도(쓰고 난 뒤의
  진행 메시지 k = 1이 취소를 읽음) 화면은 「저장했습니다」를 보인다(설계 2.6: 「취소」는 원래 파일이 그대로라는 뜻).
- probe_main(conn): 대화 상자가 열 때 한 번 띄우는 작은 자식: MP4 인코더 시험(mp4.encoder_status, 윈도우면 MF 64×64 두 장)
  결과를 (kind, detail)로 보낸다. MF 시험의 COM 실수도 화면 프로세스를 죽이지 않게.

이 모듈은 맨 위에서 Qt를 불러오지 않는다(명령줄 video-probe·자체 점검이 가볍게 불러오고, spawn 자식이 Qt 플랫폼을 고른 뒤 Qt를
불러오게).
"""

from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from evolab.app import strings as S
from evolab.app.export.job import ExportJob

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

CANCEL: Final = "cancel"
"""화면 → 자식 메시지: 그만 만들기(설계 2.6)."""
STARTED_K: Final = 0
"""자식이 Qt를 띄운 뒤 처음 보내는 진행 메시지의 k. 화면은 이 메시지 전에 자식이 끝나면 「자식이 Qt를 못 띄움」으로 보고
같은 프로세스에서 다시 만든다(설계 7.4 되돌릴 지점 10)."""

Progress = Callable[[dict[str, Any]], Any]


class ExportCancelled(Exception):
    """진행 콜백이 False를 돌려줘 그만뒀다(임시 파일은 지웠고 원래 파일은 그대로)."""


# ---------------------------------------------------------------------------
# 작업 → 파일
# ---------------------------------------------------------------------------
def progress_total(job: ExportJob) -> int:
    """진행 메시지의 n: MP4·GIF는 프레임 수(frames.frame_count), 폰 재생 파일은 1(끝 장면 그림 하나를 그리고 파일 하나를 씀)."""
    if job.fmt == "html":
        return 1
    from evolab.app.export.frames import frame_count

    return frame_count(job)


def run_job(job: ExportJob, out: Path, progress: Progress | None = None) -> dict[str, Any]:
    """작업을 out 파일로 만든다(QApplication이 있는 스레드에서). 돌려주는 것은 「done」 메시지 내용 {"path", "frames", "bytes",
    "encoder"}(encoder = "mf" | "builtin" | "gif" | "html").

    progress(메시지 사전)는 프레임마다 {"type": "frames", "k", "n", "bytes"}, MF 대체 때 {"type": "note", "text"}로 불린다.
    False를 돌려주면 그만두고 ExportCancelled. 그 밖의 실패는 예외 그대로(OSError, phone.PhoneTooBig, mp4.EncoderUnavailable,
    mf_win.MfError(내장 쓰기도 못 할 때), MemoryError 등): 어느 쪽이든 임시 파일은 남지 않는다.
    """
    out = Path(out)
    send: Progress = progress if progress is not None else (lambda _msg: True)
    if job.fmt == "mp4":
        return _run_mp4(job, out, send)
    if job.fmt == "gif":
        return _run_gif(job, out, send)
    return _run_html(job, out, send)


def _run_mp4(job: ExportJob, out: Path, send: Progress) -> dict[str, Any]:
    from evolab.app.export import frames, mp4

    n = progress_total(job)
    width, height = job.size

    def on_frame(count: int, written: int) -> bool:
        return send({"type": "frames", "k": int(count), "n": n, "bytes": int(written)}) is not False

    def on_note(text: str) -> None:
        send({"type": "note", "text": str(text)})

    done = mp4.write_mp4(lambda: frames.iter_frames(job), out, width, height, job.fps, prefer=job.encoder,
                         progress=on_frame, note=on_note)
    if done is None:
        raise ExportCancelled
    return done


def gif_temp_path(path: Path) -> Path:
    """GIF·폰 재생 파일을 쓰는 동안의 임시 파일 `<이름>.tmp`(v3.0 GIF 규칙). MP4는 mp4.temp_path."""
    return path.with_name(path.name + ".tmp")


def temp_path_for(job: ExportJob, path: Path) -> Path:
    """그 작업이 쓰는 동안의 임시 파일(자식을 강제로 끝냈을 때 화면이 지운다)."""
    if job.fmt == "mp4":
        from evolab.app.export.mp4 import temp_path

        return temp_path(path)
    return gif_temp_path(Path(path))


def _run_gif(job: ExportJob, out: Path, send: Progress) -> dict[str, Any]:
    from evolab.app.export import frames
    from evolab.app.export.video import _GifWriter

    n = progress_total(job)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = gif_temp_path(out)
    try:
        with open(tmp, "wb") as fp, closing(frames.iter_frames(job)) as source:
            writer = _GifWriter(fp, job.fps)
            for k, img in enumerate(source, 1):
                writer.add(img)
                if send({"type": "frames", "k": k, "n": n, "bytes": fp.tell()}) is False:
                    raise ExportCancelled
            count = writer.close()
        if not count:
            raise ValueError(S.VIDEO_NO_FRAMES)
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {"path": str(out), "frames": int(count), "bytes": out.stat().st_size, "encoder": "gif"}


def _run_html(job: ExportJob, out: Path, send: Progress) -> dict[str, Any]:
    from evolab.app.export import frames, phone

    phone.build_payload(job)  # 상한(칸·물체·프레임)을 넘으면 그림을 그리기 전에 PhoneTooBig
    poster = frames.last_frame(job)
    if send({"type": "frames", "k": 0, "n": 1, "bytes": 0}) is False:
        raise ExportCancelled
    path = phone.write_phone_html(out, job, poster)
    size = path.stat().st_size
    send({"type": "frames", "k": 1, "n": 1, "bytes": size})  # 파일은 이미 바뀜: 돌려준 값(취소)은 보지 않는다
    return {"path": str(path), "frames": max(int(it.replay.trajectory.frames) for it in job.items), "bytes": size,
            "encoder": "html"}


def error_message(exc: BaseException) -> str:
    """자식·명령줄이 보이는 짧은 한국어 까닭(EXP_FAILED의 {reason}): MF를 쓸 수 없으면 EXP_INFO_BUILTIN_TIP, 그 밖은
    errors.reason(우리 한국어 글, 나눈 OS 오류 글, 모르는 예외면 종류 이름)."""
    from evolab.app.export.mp4 import EncoderUnavailable

    if isinstance(exc, EncoderUnavailable):
        return S.EXP_INFO_BUILTIN_TIP.format(reason=str(exc))
    from evolab.app import errors

    return errors.reason(exc)


# ---------------------------------------------------------------------------
# 자식 프로세스
# ---------------------------------------------------------------------------
def prepare_qt_platform() -> None:
    """창 없이 그릴 Qt 플랫폼: QT_QPA_PLATFORM이 없으면 윈도우 밖에서는 offscreen. 윈도우는 기본 플랫폼(windows) 그대로 둔다:
    창을 띄우지 않으면 보이는 것이 없고, offscreen 플랫폼은 윈도우에서 글꼴 폴더를 찾지 못해 글자가 모두 네모로 그려진다(v3.0
    윈도우 CI에서 확인, lab.view._PanelScroll). 영상 위 글(세대·시각)이 화면 프로세스의 v3.0 GIF와 같은 글꼴이 되게."""
    if sys.platform != "win32":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def ensure_qt_app() -> Any:
    """QApplication(이미 있으면 그것). 관전 캔버스가 QWidget이라 QGuiApplication으로는 모자란다."""
    prepare_qt_platform()
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([sys.argv[0] if sys.argv and sys.argv[0] else "evolab"])
    return app


def _parent_alive() -> bool:
    import multiprocessing as mp

    parent = mp.parent_process()
    return parent is None or parent.is_alive()


class _Link:
    """자식 쪽 연결: 메시지 보내기와 취소 확인. 화면이 죽거나 연결이 끊기면 취소로 본다."""

    def __init__(self, conn: Connection) -> None:
        self.conn = conn
        self.cancelled = False
        self.broken = False
        """연결이 끊겼다(보낼 수 없음). cancelled도 함께 선다."""

    def send(self, msg: dict[str, Any]) -> bool:
        """진행 메시지를 보내고 취소를 확인한다. 취소됐으면 False(그 뒤로는 보내지 않는다)."""
        if self.cancelled:
            return False
        try:
            self.conn.send(msg)
        except (OSError, EOFError, ValueError):
            self.cancelled = self.broken = True
            return False
        return self.check()

    def send_final(self, msg: dict[str, Any]) -> None:
        """파일을 최종 이름으로 바꾼 뒤의 「done」: 늦게 온 취소와 상관없이 보낸다(연결이 끊겼으면 그만)."""
        if self.broken:
            return
        try:
            self.conn.send(msg)
        except (OSError, EOFError, ValueError):
            self.cancelled = self.broken = True

    def check(self) -> bool:
        """취소가 왔거나 화면이 사라졌으면 False."""
        try:
            while self.conn.poll():
                if self.conn.recv() == CANCEL:
                    self.cancelled = True
        except (OSError, EOFError, ValueError):
            self.cancelled = self.broken = True
        if not self.cancelled and not _parent_alive():
            self.cancelled = True
        return not self.cancelled


def export_main(job: ExportJob, conn: Connection, out: Path | str) -> None:
    """내보내기 자식 진입점(spawn). 치명적 오류 스택은 faults.log에(diagnostics.enable_fault_log), 메시지는 2.6."""
    from evolab import diagnostics

    diagnostics.enable_fault_log()
    link = _Link(conn)
    try:
        ensure_qt_app()
        if link.send({"type": "frames", "k": STARTED_K, "n": progress_total(job), "bytes": 0}):
            done = run_job(job, Path(out), link.send)
            link.send_final({"type": "done", **done})  # 파일은 이미 바뀌었다: 늦은 취소여도 알린다
    except ExportCancelled:
        pass
    except BaseException as exc:  # noqa: BLE001 - 무엇이든 화면에 알리고 끝낸다(화면이 errors.jsonl에 남김)
        if not link.cancelled:
            link.send({"type": "error", "message": error_message(exc), "detail": traceback.format_exc()})
    finally:
        try:
            conn.close()
        except OSError:
            pass


def probe_main(conn: Connection) -> None:
    """MP4 인코더 시험 자식: (kind, detail)을 보내고 끝낸다(mp4.encoder_status, 윈도우면 MF 64×64 두 장을 임시 폴더에)."""
    from evolab import diagnostics

    diagnostics.enable_fault_log()
    try:
        from evolab.app.export.mp4 import encoder_status

        status = encoder_status()
        conn.send((status.kind, status.detail))
    except BaseException as exc:  # noqa: BLE001
        try:
            conn.send(("builtin", f"{type(exc).__name__}: {exc}"))
        except (OSError, ValueError):
            pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


__all__ = [
    "CANCEL", "STARTED_K", "ExportCancelled", "ensure_qt_app", "error_message", "export_main", "gif_temp_path",
    "prepare_qt_platform", "probe_main", "progress_total", "run_job", "temp_path_for",
]

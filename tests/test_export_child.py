"""내보내기 실행과 자식 프로세스 (v3 설계 D49, 2.6, 4.14, 7.4 되돌릴 지점 10, 8.1 T40: export/worker.py·runner.py).

- run_job: GIF·MP4·폰 재생 파일을 끝까지 만들고 진행 메시지(k)는 늘어나기만 한다. 취소하면 파일·임시 파일이 남지 않고 같은
  이름의 원래 파일은 그대로다.
- ExportRunner(spawn 자식): 세 형식이 끝까지 가고 진행이 늘어나기만, 취소하면 5초 안에 끝나고 파일 없음, 자식에 os._exit(3)를
  넣으면 EXP_PROCESS_DIED(종료 코드 풀이), 자식이 Qt를 띄우기 전에 죽으면 같은 프로세스에서 다시 만든다(MF 끔), 자식의 오류는
  까닭 글 + traceback으로 오고 errors.jsonl(kind="export")에 남는다.
- light_job이 재생의 추적 자료를 뺀다(그림은 같음).

이 모듈은 spawn 자식이 다시 불러온다(가짜 자식 진입점): 맨 위에서 QApplication을 만들지 않는다.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab import paths
from evolab.app import strings as S
from evolab.app.export import job as J
from evolab.app.export import mp4, phone, worker
from evolab.engine import seeds
from evolab.engine.replay import Replay, replay
from evolab.policy.genome import layout_for
from evolab.spec.io import load_env

_CACHE: dict[tuple[str, int, bool], Replay] = {}


def _replay(env: str = "lander", seed: int = 1, trace: bool = False) -> Replay:
    key = (env, seed, trace)
    if key not in _CACHE:
        spec = load_env(paths.builtin_env_dir() / f"{env}.evoenv")
        layout = layout_for(spec)
        genome = np.random.default_rng(seed).normal(0.0, 1.0, layout.dim)
        _CACHE[key] = replay(spec, layout, genome, seeds.display_trial(), trace=trace)
    return _CACHE[key]


def _short(rep: Replay, frames: int) -> Replay:
    tr = dataclasses.replace(rep.trajectory, poses=rep.trajectory.poses[:frames].copy(),
                             outputs=rep.trajectory.outputs[:frames].copy(), end_step=None)
    return dataclasses.replace(rep, trajectory=tr)


def _job(fmt: str, rep: Replay | None = None, **kw: Any) -> J.ExportJob:
    rep = rep if rep is not None else _short(_replay(), 31)  # 1초
    return J.job_from_replays([(rep, J.NOW_COLOR, "a")], fmt, size=kw.pop("size", "small"),
                              camera=kw.pop("camera", "fixed"), title="t", meta={J.META_ENV: "lander"}, **kw)


@pytest.fixture()
def user_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """데이터·상태 폴더를 임시 폴더로(자식은 물려받는다)."""
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    return tmp_path


def _check_file(job: J.ExportJob, done: dict[str, Any]) -> None:
    from PIL import Image

    from evolab.app.export.frames import frame_count

    path = Path(done["path"])
    assert path.is_file() and done["bytes"] == path.stat().st_size
    assert not list(path.parent.glob("*.tmp*"))
    if job.fmt == "mp4":
        info = mp4.read_info(path)
        assert info["frames"] == done["frames"] == frame_count(job) and (info["width"], info["height"]) == job.size
        assert done["encoder"] in ("builtin", "mf")
    elif job.fmt == "gif":
        with Image.open(path) as im:
            assert im.size == job.size and getattr(im, "n_frames", 1) >= 1
        assert done["frames"] == frame_count(job) and done["encoder"] == "gif"
    else:
        dec = phone.decode_payload(phone.read_phone_html(path))
        src = job.items[0].replay.trajectory.poses
        assert dec["entries"][0]["poses"].shape == src.shape
        assert np.abs(dec["entries"][0]["poses"][..., :2] - src[..., :2]).max() <= phone.POS_UNIT
        assert done["encoder"] == "html" and done["frames"] == src.shape[0]


# ---------------------------------------------------------------------------
# run_job (같은 프로세스)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", ["gif", "mp4", "html"])
def test_run_job_writes_each_format_with_growing_progress(qtbot: Any, tmp_path: Path, fmt: str) -> None:
    job = _job(fmt)
    seen: list[dict[str, Any]] = []
    done = worker.run_job(job, tmp_path / "내보내기" / f"a{J.SUFFIXES[fmt]}", lambda m: seen.append(m) or True)
    _check_file(job, done)
    ks = [m["k"] for m in seen if m["type"] == "frames"]
    assert ks == sorted(ks) and ks[-1] == worker.progress_total(job)
    sizes = [m["bytes"] for m in seen if m["type"] == "frames"]
    assert sizes == sorted(sizes) and sizes[-1] > 0


@pytest.mark.parametrize("fmt", ["gif", "mp4", "html"])
def test_run_job_cancel_keeps_the_old_file_and_no_temp(qtbot: Any, tmp_path: Path, fmt: str) -> None:
    job = _job(fmt)
    out = tmp_path / f"a{J.SUFFIXES[fmt]}"
    out.write_bytes(b"old")
    calls = [0]

    def stop(msg: dict[str, Any]) -> bool:  # 첫 진행 메시지에서 그만(GIF·MP4: 한 프레임을 쓴 뒤, 폰 파일: 쓰기 전)
        calls[0] += 1
        return False

    with pytest.raises(worker.ExportCancelled):
        worker.run_job(job, out, stop)
    assert calls[0] == 1
    assert out.read_bytes() == b"old" and sorted(p.name for p in tmp_path.iterdir()) == [out.name]


def test_run_job_errors_leave_no_temp(qtbot: Any, tmp_path: Path) -> None:
    rep = _replay()
    long = dataclasses.replace(rep, trajectory=dataclasses.replace(
        rep.trajectory, poses=np.repeat(rep.trajectory.poses[:1], phone.PHONE_MAX_FRAMES + 2, axis=0),
        outputs=np.repeat(rep.trajectory.outputs[:1], phone.PHONE_MAX_FRAMES + 2, axis=0), end_step=None))
    with pytest.raises(phone.PhoneTooBig) as err:
        worker.run_job(_job("html", long), tmp_path / "a.html")
    assert worker.error_message(err.value) == str(err.value)  # 한국어 까닭 그대로
    assert list(tmp_path.iterdir()) == []
    (tmp_path / "dir.gif").mkdir()  # 같은 이름의 폴더: 이름을 바꿀 수 없다
    with pytest.raises(OSError) as err2:
        worker.run_job(_job("gif"), tmp_path / "dir.gif")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dir.gif"]
    why = worker.error_message(err2.value)
    assert why and str(tmp_path) not in why  # 날것 예외 글(경로 포함)이 아닌 짧은 까닭
    if mp4.encoder_status().kind != "mf":  # MF가 없는 PC(리눅스, 윈도우 CI 러너): --encoder mf는 쓸 수 없다
        with pytest.raises(mp4.EncoderUnavailable) as err3:
            worker.run_job(dataclasses.replace(_job("mp4"), encoder="mf"), tmp_path / "m.mp4")
        assert worker.error_message(err3.value) == S.EXP_INFO_BUILTIN_TIP.format(reason=str(err3.value))
        assert not (tmp_path / "m.mp4").exists() and not mp4.temp_path(tmp_path / "m.mp4").exists()


def test_light_job_drops_trace_data_and_keeps_the_picture(qtbot: Any) -> None:
    from evolab.app.export.frames import last_frame
    from evolab.app.export.runner import light_job

    rep = _short(_replay(trace=True), 31)
    assert rep.observations is not None and rep.activations is not None
    job = _job("gif", rep)
    light = light_job(job)
    got = light.items[0].replay
    assert got.observations is None and got.activations is None and got.raw is None and got.rays is None
    assert got.trajectory is rep.trajectory and light.meta == job.meta and light.size == job.size
    assert last_frame(light) == last_frame(job)
    plain = _job("gif")
    assert light_job(plain) is plain


# ---------------------------------------------------------------------------
# ExportRunner (spawn 자식)
# ---------------------------------------------------------------------------
def _run(qtbot: Any, runner: Any, job: J.ExportJob, out: Path, timeout_ms: int = 120_000) -> dict[str, Any]:
    events: dict[str, Any] = {"k": [], "notes": []}
    runner.progressed.connect(lambda k, n, b: events["k"].append((k, n, b)))
    runner.noted.connect(events["notes"].append)
    runner.finished.connect(lambda d: events.__setitem__("done", d))
    runner.failed.connect(lambda m, d: events.__setitem__("failed", (m, d)))
    runner.cancelled.connect(lambda: events.__setitem__("cancelled", True))
    runner.start(job, out)
    qtbot.waitUntil(lambda: not runner.running, timeout=timeout_ms)
    return events


@pytest.mark.parametrize("fmt", ["gif", "mp4", "html"])
def test_child_process_makes_each_format(qtbot: Any, user_dirs: Path, fmt: str) -> None:
    from evolab.app.export.runner import ExportRunner

    job = _job(fmt)
    runner = ExportRunner()
    t0 = time.monotonic()
    ev = _run(qtbot, runner, job, user_dirs / "내보내기" / f"챔피언{J.SUFFIXES[fmt]}")
    assert "done" in ev, ev.get("failed")
    assert runner.mode == "child" and not runner.fell_back
    _check_file(job, ev["done"])
    ks = [k for k, _n, _b in ev["k"]]
    assert ks[0] == worker.STARTED_K and ks == sorted(ks) and ks[-1] == worker.progress_total(job)
    assert time.monotonic() - t0 < 60


def test_child_cancel_ends_within_five_seconds_without_a_file(qtbot: Any, user_dirs: Path) -> None:
    from evolab.app.export.runner import CANCEL_GRACE_S, ExportRunner

    job = _job("mp4", _replay("hill_car", 2), size="medium", camera="follow")  # 30초, 917프레임: 끝나기 전에 취소
    out = user_dirs / "취소.mp4"
    runner = ExportRunner()
    ev: dict[str, Any] = {"k": []}
    runner.progressed.connect(lambda k, n, b: ev["k"].append(k))
    runner.finished.connect(lambda d: ev.__setitem__("done", d))
    runner.cancelled.connect(lambda: ev.__setitem__("cancelled", time.monotonic()))
    runner.start(job, out)
    qtbot.waitUntil(lambda: len(ev["k"]) >= 3, timeout=60_000)
    t0 = time.monotonic()
    runner.cancel()
    qtbot.waitUntil(lambda: not runner.running, timeout=int((CANCEL_GRACE_S + 10) * 1000))
    assert "cancelled" in ev and "done" not in ev
    assert ev["cancelled"] - t0 < CANCEL_GRACE_S + 1.0
    assert not out.exists() and not mp4.temp_path(out).exists()


def _exit_after_start(job: J.ExportJob, conn: Any, out: str) -> None:
    """가짜 자식: Qt를 띄웠다고 알린 뒤 갑자기 끝난다(접근 위반 같은 것)."""
    conn.send({"type": "frames", "k": worker.STARTED_K, "n": 10, "bytes": 0})
    time.sleep(0.2)
    os._exit(3)


def _exit_before_start(job: J.ExportJob, conn: Any, out: str) -> None:
    """가짜 자식: 아무것도 보내지 못하고 끝난다(묶인 exe에서 자식이 Qt를 못 띄우는 경우)."""
    os._exit(3)


def test_child_that_dies_reports_process_died(qtbot: Any, user_dirs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab import diagnostics
    from evolab.app.export import runner as R

    monkeypatch.setattr(worker, "export_main", _exit_after_start)
    out = user_dirs / "a.gif"
    runner = R.ExportRunner()
    ev = _run(qtbot, runner, _job("gif"), out)
    message = R.died_text(3, "gif")
    plain = S.EXP_PROCESS_DIED.format(meaning=diagnostics.exit_code_text(3) or S.EXIT_CODE_UNKNOWN)
    assert ev["failed"][0] == message == plain  # GIF 자식이 끝났는데 「GIF로 저장해 보세요」는 맞지 않는다
    assert S.EXP_PROCESS_DIED_GIF_HINT not in message
    assert runner.exit_code == 3 and not out.exists()
    logged = [json.loads(line) for line in (paths.log_dir() / "errors.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["kind"] == "export" and e["message"] == message for e in logged)
    # MP4 자식이 끝나면 GIF로 저장해 보라는 말을 붙인다(인코더 시험 까닭에는 붙이지 않음)
    ev = _run(qtbot, R.ExportRunner(), _job("mp4"), user_dirs / "a.mp4")
    assert ev["failed"][0] == R.died_text(3, "mp4") == f"{plain} {S.EXP_PROCESS_DIED_GIF_HINT}"
    assert R.died_text(3) == R.died_text(3, "html") == plain


def test_child_that_cannot_start_falls_back_to_this_process(qtbot: Any, user_dirs: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """되돌릴 지점 10: 자식이 첫 메시지(Qt를 띄움) 전에 끝나면 같은 프로세스에서 만들고 MF는 끈다(내장 쓰기)."""
    from evolab.app.export import runner as R

    monkeypatch.setattr(worker, "export_main", _exit_before_start)
    job = _job("mp4")
    runner = R.ExportRunner()
    ev = _run(qtbot, runner, job, user_dirs / "a.mp4")
    assert "done" in ev, ev.get("failed")
    assert runner.fell_back and runner.mode == "inprocess" and ev["done"]["encoder"] == "builtin"
    _check_file(job, ev["done"])
    logged = (paths.log_dir() / "errors.jsonl").read_text(encoding="utf-8")
    assert S.EXP_INPROCESS_NOTE in logged
    # 같은 프로세스 길도 취소할 수 있다
    runner2 = R.ExportRunner(use_child=False)
    runner2.progressed.connect(lambda k, n, b: k >= 2 and runner2.cancel())
    ev2 = _run(qtbot, runner2, _job("gif"), user_dirs / "b.gif")
    assert ev2.get("cancelled") and not (user_dirs / "b.gif").exists()


def test_child_error_comes_with_reason_and_traceback(qtbot: Any, user_dirs: Path) -> None:
    from evolab.app.export.runner import ExportRunner

    folder = user_dirs / "dir.gif"
    folder.mkdir()  # 같은 이름의 폴더라 이름을 바꿀 수 없다
    runner = ExportRunner()
    ev = _run(qtbot, runner, _job("gif"), folder)
    message, detail = ev["failed"]
    assert message and str(folder) not in message and "Traceback" in detail  # 까닭은 짧게, 원문은 자세히로
    assert folder.is_dir() and not (user_dirs / "dir.gif.tmp").exists()
    logged = [json.loads(line) for line in (paths.log_dir() / "errors.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["kind"] == "export" and e["message"] == message and "Traceback" in e["detail"] for e in logged)


def test_run_in_child_without_an_event_loop(user_dirs: Path) -> None:
    """자체 점검이 쓰는 길: 이벤트 루프 없이 자식으로 끝까지, 첫 메시지까지 걸린 초도."""
    from evolab.app.export.runner import ExportFailed, run_in_child

    job = _job("html")
    done = run_in_child(job, user_dirs / "a.html", timeout=120)
    _check_file(job, done)
    assert 0.0 < done["first_message_s"] < 60.0
    (user_dirs / "dir.html").mkdir()
    with pytest.raises(ExportFailed) as err:
        run_in_child(job, user_dirs / "dir.html", timeout=120)
    assert err.value.message and "Traceback" in err.value.detail


def test_encoder_probe_reports_once_per_process(qtbot: Any, monkeypatch: pytest.MonkeyPatch, user_dirs: Path) -> None:
    # user_dirs: 시험 자식(probe_main)이 faults.log를 여는 상태 폴더를 임시 폴더로(실제 사용자 폴더에 쓰지 않게)
    from evolab.app.export import runner as R

    monkeypatch.setattr(R, "_PROBED", None)
    got: list[tuple[str, str]] = []
    probe = R.EncoderProbe()
    probe.done.connect(lambda k, d: got.append((k, d)))
    probe.start()
    qtbot.waitUntil(lambda: bool(got), timeout=60_000)
    assert got[0][0] in ("mf", "builtin") and R.probed() == got[0]
    if not R.probe_needs_child():  # 윈도우가 아니거나 MF를 껐다: DLL을 열지 않고 이 프로세스에서
        assert got[0] == R.probe_in_process() and got[0][0] == "builtin"
    again: list[tuple[str, str]] = []
    probe2 = R.EncoderProbe()
    probe2.done.connect(lambda k, d: again.append((k, d)))
    probe2.start()
    qtbot.waitUntil(lambda: bool(again), timeout=5_000)
    assert again == got
    kind, _detail = R.probe_in_child(timeout=60)  # 작은 자식으로도 같은 판단
    assert kind == got[0][0]


class _FakeConn:
    """자식 쪽 연결 흉내(export_main을 이 프로세스에서): 보낸 메시지를 남기고, inbox의 메시지를 화면이 보낸 것처럼 준다."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.inbox: list[Any] = []

    def send(self, msg: Any) -> None:
        self.sent.append(msg)

    def poll(self, timeout: float = 0.0) -> bool:
        return bool(self.inbox)

    def recv(self) -> Any:
        return self.inbox.pop(0)

    def close(self) -> None:
        pass


def test_cancel_after_the_phone_file_was_replaced_still_reports_done(qtbot: Any, user_dirs: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """폰 파일을 쓰는 사이에 온 취소: 원래 파일은 이미 바뀌었으므로 자식은 「done」을 보낸다(화면은 finished로 「저장했습니다」).
    「done」 없이 끝나 화면이 「취소됨」으로 보이던 v3.1 검토 결함(설계 2.6: 취소는 원래 파일을 그대로 둔다는 뜻)."""
    job = _job("html")
    out = user_dirs / "폰.html"
    out.write_text("OLD FILE", encoding="utf-8")
    conn = _FakeConn()
    original = phone.write_phone_html

    def write_while_cancelled(path: Path, job: J.ExportJob, poster: Any) -> Path:
        conn.inbox.append(worker.CANCEL)  # 파일을 쓰는 동안 사용자가 「취소」
        return original(path, job, poster)

    monkeypatch.setattr(phone, "write_phone_html", write_while_cancelled)
    monkeypatch.setattr(worker, "_parent_alive", lambda: True)
    worker.export_main(job, conn, out)
    kinds = [m["type"] for m in conn.sent]
    assert kinds[-1] == "done" and "error" not in kinds, conn.sent
    assert conn.sent[-1]["path"] == str(out) and out.read_text(encoding="utf-8").startswith("<!")
    # 쓰기 전에 온 취소는 그대로 취소(원래 파일 그대로, done 없음)
    from evolab.app.export import frames

    out.write_text("OLD FILE", encoding="utf-8")
    conn = _FakeConn()
    real_last_frame = frames.last_frame

    def cancel_before_writing(job: J.ExportJob) -> Any:
        conn.inbox.append(worker.CANCEL)
        return real_last_frame(job)

    monkeypatch.setattr(phone, "write_phone_html", original)
    monkeypatch.setattr(frames, "last_frame", cancel_before_writing)
    worker.export_main(job, conn, out)
    assert "done" not in [m["type"] for m in conn.sent] and out.read_text(encoding="utf-8") == "OLD FILE"
    assert not worker.gif_temp_path(out).exists()


def test_runner_turns_a_done_after_cancel_into_finished(qtbot: Any, user_dirs: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """진짜 자식: 취소를 보낸 뒤 자식이 「done」을 보내면(늦은 취소) 화면은 finished(파일 경로)를 받는다."""
    from evolab.app.export import runner as R

    monkeypatch.setattr(worker, "export_main", _done_after_cancel)
    runner = R.ExportRunner()
    out = user_dirs / "늦은 취소.html"
    runner.progressed.connect(lambda k, n, b: runner.cancel())
    ev = _run(qtbot, runner, _job("html"), out)
    assert ev.get("done", {}).get("path") == str(out) and "cancelled" not in ev, ev


def _done_after_cancel(job: J.ExportJob, conn: Any, out: str) -> None:
    """가짜 자식: 시작을 알리고, 화면의 취소를 받은 뒤에도 「done」을 보낸다(파일을 이미 바꾼 뒤의 늦은 취소)."""
    conn.send({"type": "frames", "k": worker.STARTED_K, "n": 1, "bytes": 0})
    if conn.poll(30):
        conn.recv()
    Path(out).write_text("<!doctype html>", encoding="utf-8")
    conn.send({"type": "done", "path": str(out), "frames": 1, "bytes": 15, "encoder": "html"})
    conn.close()


class _FailingMf:
    """MF 작성기 흉내: 다섯째 프레임에서 MfError(T37의 run_job 쪽: 내장 쓰기로 처음부터 + note)."""

    kind = "mf"

    def __init__(self, path: Path, width: int, height: int, fps: int, *, bitrate: int, tmp: Path) -> None:
        self.tmp, self.frames = Path(tmp), 0
        self.tmp.write_bytes(b"")

    def add(self, frame: Any) -> None:
        from evolab.app.export import mf_win

        if self.frames == 4:
            raise mf_win.MfError("IMFSinkWriter::WriteSample", 0xC00D36B4)
        self.frames += 1

    def bytes_written(self) -> int:
        return 0

    def close(self) -> int:
        raise AssertionError("never closed")

    def abort(self) -> None:
        self.tmp.unlink(missing_ok=True)


def test_run_job_rebuilds_with_the_builtin_writer_when_mf_fails(qtbot: Any, tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.export import mf_win

    monkeypatch.setattr(mp4, "_STATUS", None)
    monkeypatch.setattr(mp4, "_is_windows", lambda: True)
    monkeypatch.setattr(mf_win, "probe", lambda: (True, "Fake H264 MFT"))
    monkeypatch.setattr(mf_win, "MfWriter", _FailingMf)
    job = _job("mp4")
    seen: list[dict[str, Any]] = []
    done = worker.run_job(job, tmp_path / "a.mp4", lambda m: seen.append(m) or True)
    assert done["encoder"] == "builtin"
    _check_file(job, done)
    notes = [m["text"] for m in seen if m["type"] == "note"]
    assert notes == [S.EXP_MF_FALLBACK.format(reason="IMFSinkWriter::WriteSample 0xC00D36B4")]
    ks = [m["k"] for m in seen if m["type"] == "frames"]
    assert ks[:4] == [1, 2, 3, 4] and ks[4] == 1 and ks[-1] == worker.progress_total(job)  # 처음부터 다시
    monkeypatch.setattr(mp4, "_STATUS", None)

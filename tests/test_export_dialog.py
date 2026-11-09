"""내보내기 대화 상자 (v3 설계 D48·D50, 4.14, 5.6, 8.1 T41: export/dialog.py, lab.save_video·race.save_gif).

- 형식마다 칸이 보이고 숨는다(크기는 MP4만, 카메라는 폰 파일이면 숨김), 안내 줄(길이·프레임 수, MP4 인코더: MF면 예상 크기,
  내장 쓰기면 EXP_INFO_BUILTIN + 풍선 까닭, 시험 전에는 EXP_INFO_PROBING, 「none」이면 MP4를 고를 수 없음).
- 내장 쓰기·따라가기에서 2초 몫으로 200MB를 넘을 예상이면 한 번 묻고, 아니요면 취소(고정 카메라·MF면 묻지 않음).
- 「저장…」이 진짜 자식 프로세스로 세 형식을 만들고(확장자 고침, 「저장했습니다」 + 「폴더 열기」, 폰 파일은 「폰으로 옮기는 법」
  한 번, 다시 보지 않기), 마지막 선택을 settings.ini [export]에 남긴다. 실패는 상자 하나.
- 실험실 save_video()·경주 save_gif()는 경로가 없으면 이 대화 상자를 열고(실험실은 실행·세대·환경 정보와 재생 환경의 기본
  보기를 넘김), 경로를 주면 v3.0과 같은 GIF(기존 test_lab·test_review_lab·test_race가 지킨다)."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget

from evolab import paths
from evolab.app import strings as S
from evolab.app.export import dialog as D
from evolab.app.export import job as J
from evolab.app.export import mp4
from evolab.app.export.frames import frame_count
from evolab.engine import seeds
from evolab.engine.replay import Replay, replay
from evolab.policy.genome import layout_for
from evolab.spec.io import load_env

DATA = Path(__file__).parent / "data"
_CACHE: dict[tuple[str, int], Replay] = {}


def _replay(env: str = "lander", seed: int = 1, frames: int | None = 31) -> Replay:
    if (env, seed) not in _CACHE:
        spec = load_env(paths.builtin_env_dir() / f"{env}.evoenv")
        layout = layout_for(spec)
        _CACHE[(env, seed)] = replay(spec, layout, np.random.default_rng(seed).normal(0.0, 1.0, layout.dim),
                                     seeds.display_trial())
    rep = _CACHE[(env, seed)]
    if frames is None:
        return rep
    tr = dataclasses.replace(rep.trajectory, poses=rep.trajectory.poses[:frames].copy(),
                             outputs=rep.trajectory.outputs[:frames].copy(), end_step=None)
    return dataclasses.replace(rep, trajectory=tr)


@pytest.fixture()
def user_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    return tmp_path


@pytest.fixture()
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def _dialog(qtbot: Any, settings: QSettings, items: Any = None, **kw: Any) -> D.ExportDialog:
    items = items if items is not None else [(_replay(), QColor(J.OLD_COLOR), "1세대 대표"),
                                             (_replay("lander", 2), QColor(J.NOW_COLOR), "지금 챔피언 (4세대)")]
    dlg = D.ExportDialog(None, kind=kw.pop("kind", "replay"), items=items, title=kw.pop("title", "1세대 vs 지금"),
                         default_name=kw.pop("default_name", "영상"), meta=kw.pop("meta", {}), settings=settings,
                         probe=kw.pop("probe", False), **kw)
    qtbot.addWidget(dlg)
    dlg.show()
    return dlg


class FakeRunner(QObject):
    """진짜 자식 대신: start에서 미리 정한 메시지를 흘린다(테스트가 진행·취소·실패를 정한다)."""

    progressed = Signal(int, int, int)
    noted = Signal(str)
    finished = Signal(dict)
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(self, script: list[tuple[str, Any]]) -> None:
        super().__init__()
        self.script = script
        self.running = False
        self.encoder_hint = ""
        self.cancel_calls = 0
        self.started: list[tuple[J.ExportJob, Path]] = []

    def start(self, job: J.ExportJob, out: Path) -> None:
        from PySide6.QtCore import QTimer

        self.started.append((job, out))
        self.running = True
        QTimer.singleShot(0, self._play)

    def _play(self) -> None:
        from PySide6.QtCore import QCoreApplication

        for kind, value in self.script:
            if not self.running:
                return
            if kind == "frames":
                self.progressed.emit(*value)
            elif kind == "note":
                self.noted.emit(value)
            elif kind == "failed":
                self.running = False
                self.failed.emit(*value)
                return
            elif kind == "done":
                self.running = False
                self.finished.emit(value)
                return
            for _ in range(3):
                QCoreApplication.processEvents()
        if self.running:  # 끝나지 않는 작업: 취소를 기다린다
            return

    def cancel(self) -> None:
        self.cancel_calls += 1
        if self.running:
            self.running = False
            self.cancelled.emit()


# ---------------------------------------------------------------------------
# 칸과 안내 줄
# ---------------------------------------------------------------------------
def test_fields_follow_the_format(qtbot: Any, settings: QSettings) -> None:
    dlg = _dialog(qtbot, settings)
    assert dlg.windowTitle() == S.VIDEO_SAVE_TITLE and dlg.what_label.text() == S.EXP_WHAT_REPLAY.format(
        label="1세대 vs 지금")
    assert dlg.fmt() == "mp4" and dlg.size_name() == "medium" and dlg.camera() == "fixed"  # 처음 기본값(설계 D50·D54)
    assert [b.text() for b in dlg.format_buttons.values()] == [S.EXP_FORMAT_MP4, S.EXP_FORMAT_GIF, S.EXP_FORMAT_HTML]
    shown = {}
    for fmt in J.FORMATS:
        dlg.set_format(fmt)
        shown[fmt] = (dlg.size_combo.isVisible(), dlg.camera_combo.isVisible(), dlg.encoder_label.isVisible())
        job = dlg.build_job()
        assert job.fmt == fmt and job.layout == "overlay" and len(job.items) == 2
        seconds, frames = D.ExportDialog.length_of(job)
        assert dlg.length_label.text() == S.EXP_INFO_LENGTH.format(seconds=seconds, frames=frames)
        if fmt != "html":
            assert frames == frame_count(job) and seconds == pytest.approx((frames - 1) / job.fps)
    assert shown == {"mp4": (True, True, True), "gif": (False, True, False), "html": (False, False, False)}
    dlg.set_format("mp4")
    assert dlg.encoder_label.text() == S.EXP_INFO_PROBING  # 인코더 시험 결과 전
    dlg.size_combo.setCurrentIndex(list(J.MP4_SIZES).index("small"))
    dlg.camera_combo.setCurrentIndex(list(J.CAMERAS).index("follow"))
    job = dlg.build_job()
    assert job.size == (640, 360) and job.camera == "follow" and job.fps == J.MP4_FPS


def test_encoder_lines_mf_builtin_and_none(qtbot: Any, settings: QSettings) -> None:
    dlg = _dialog(qtbot, settings)
    dlg.set_encoder("mf", "H264 Encoder MFT")
    job = dlg.build_job()
    seconds, _frames = D.ExportDialog.length_of(job)
    mb = mp4.mf_bitrate(*job.size) * seconds / 8 / 1e6
    assert dlg.encoder_label.text() == S.EXP_INFO_MF.format(mb=mb)
    assert dlg.encoder_label.toolTip() == S.EXP_ENCODER_MF.format(name="H264 Encoder MFT")
    dlg.set_encoder("builtin", S.EXP_MF_NOT_WINDOWS)
    assert dlg.encoder_label.text() == S.EXP_INFO_BUILTIN
    assert dlg.encoder_label.toolTip() == S.EXP_INFO_BUILTIN_TIP.format(reason=S.EXP_MF_NOT_WINDOWS)
    dlg.set_encoder("none", S.EXP_MF_NOT_WINDOWS)  # Q4가 내장 쓰기를 빼고 MF도 없는 PC
    assert not dlg.format_buttons["mp4"].isEnabled() and dlg.fmt() == "gif"


def test_probe_fills_the_encoder_line(qtbot: Any, settings: QSettings, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.export import runner as R

    monkeypatch.setattr(R, "_PROBED", ("builtin", "fake reason"))
    dlg = _dialog(qtbot, settings, probe=True)
    qtbot.waitUntil(lambda: dlg.encoder is not None, timeout=10_000)
    assert dlg.encoder == ("builtin", "fake reason") and dlg.encoder_label.text() == S.EXP_INFO_BUILTIN


def test_race_kinds_lanes_and_overlay(qtbot: Any, settings: QSettings) -> None:
    lanes = [(_replay("lander", 1), "#3987e5", "a"), (_replay("lander", 2), "#d95926", "b"),
             (_replay("lander", 3), "#199e70", "c")]
    dlg = _dialog(qtbot, settings, items=lanes, kind="race", title="나란히 경주 · 코스 1")
    assert dlg.what_label.text() == S.EXP_WHAT_RACE.format(n=3)
    dlg.set_format("gif")
    job = dlg.build_job()
    assert job.layout == "lanes" and job.size == (960, 200) and [it.label for it in job.items] == ["a", "b", "c"]
    over = _dialog(qtbot, settings, items=lanes, kind="race", title="t", overlay=True)
    over.set_format("gif")
    job = over.build_job()
    assert job.layout == "overlay" and job.size == J.GIF_SIZE and len(job.items) == 3


# ---------------------------------------------------------------------------
# 큰 파일 묻기, 실패, 취소 (가짜 실행기)
# ---------------------------------------------------------------------------
def _big_script(job_n: int, fps: int) -> list[tuple[str, Any]]:
    k = 2 * fps
    return [("frames", (0, job_n, 0)), ("frames", (k, job_n, int(300e6 * k / job_n)))]


@pytest.mark.parametrize("camera,encoder,asks", [("follow", "builtin", True), ("fixed", "builtin", False),
                                                 ("follow", "mf", False)])
def test_big_builtin_follow_export_asks_once(qtbot: Any, settings: QSettings, tmp_path: Path, camera: str,
                                             encoder: str, asks: bool) -> None:
    dlg = _dialog(qtbot, settings)
    dlg.set_encoder(encoder, "x")
    dlg.camera_combo.setCurrentIndex(list(J.CAMERAS).index(camera))
    job = dlg.build_job()
    n = frame_count(job)
    asked: list[float] = []
    fake = FakeRunner(_big_script(n, job.fps))
    dlg.make_runner = lambda: fake  # type: ignore[method-assign]
    dlg.ask_big = lambda mb: asked.append(mb) or False  # type: ignore[method-assign]
    dlg.ask_save_path = lambda fmt, default: tmp_path / "큰 영상"  # type: ignore[method-assign]
    if not asks:  # 묻지 않으면 끝나지 않는 가짜 작업이므로 바로 끝을 더한다
        fake.script.append(("done", {"path": str(tmp_path / "큰 영상.mp4"), "frames": n, "bytes": 1, "encoder": encoder}))
    out = dlg.save()
    assert fake.started[0][1] == tmp_path / "큰 영상.mp4"  # 확장자를 붙였다
    if asks:
        assert len(asked) == 1 and asked[0] == pytest.approx(300.0, rel=0.01)
        assert fake.cancel_calls == 1 and out is None and dlg.saved_path is None
    else:
        assert asked == [] and out == tmp_path / "큰 영상.mp4"


@pytest.mark.parametrize("probe_result,want", [
    (None, "builtin"),  # 시험이 아직 끝나지 않음
    (("builtin", "timeout 30s"), "builtin"),  # 시험 자식이 시간 안에 답하지 않음
    (("builtin", "x"), "builtin"),  # 시험 자식이 죽음·MF 실패·윈도우 아님
    (("mf", "H264 Encoder MFT"), "auto"),  # MF를 쓸 수 있음: 자식이 MF를 열고, 실패하면 내장 쓰기로 다시
])
def test_save_hands_the_encoder_probe_result_to_the_child(qtbot: Any, settings: QSettings, tmp_path: Path,
                                                         probe_result: tuple[str, str] | None, want: str) -> None:
    """MP4 작업의 인코더 = 대화 상자의 시험 결과: MF가 아니면 "builtin"(자식이 MF 시험을 시간 한도 없이 다시 하지 않게, 안내 줄
    EXP_INFO_BUILTIN 그대로), MF면 "auto"(v3.1 검토 결함: 결과를 보이기만 하고 자식에게는 늘 "auto"를 넘겼다)."""
    dlg = _dialog(qtbot, settings)
    if probe_result is not None:
        dlg.set_encoder(*probe_result)
    dlg.set_format("mp4")
    assert dlg.build_job().encoder == want
    fake = FakeRunner([("frames", (0, 10, 0)),
                       ("done", {"path": str(tmp_path / "a.mp4"), "frames": 10, "bytes": 1, "encoder": want})])
    dlg.make_runner = lambda: fake  # type: ignore[method-assign]
    dlg.ask_save_path = lambda fmt, default: tmp_path / "a.mp4"  # type: ignore[method-assign]
    assert dlg.save() == tmp_path / "a.mp4"
    job = fake.started[0][0]
    assert job.fmt == "mp4" and job.encoder == want
    assert dlg.builtin_now(job) == (want == "builtin")
    dlg.set_format("gif")  # GIF·폰 파일 작업은 인코더를 건드리지 않는다
    assert dlg.build_job().encoder == "auto"


def test_save_keeps_auto_when_the_builtin_writer_is_left_out(qtbot: Any, settings: QSettings,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """Q4가 내장 쓰기를 뺀 판(BUILTIN_ENABLED False): "builtin"을 넘기면 자식이 EncoderUnavailable이므로 "auto" 그대로."""
    monkeypatch.setattr(mp4, "BUILTIN_ENABLED", False)
    dlg = _dialog(qtbot, settings)
    dlg.set_format("mp4")
    assert dlg.build_job().encoder == "auto"


def test_failure_shows_one_box_and_status(qtbot: Any, settings: QSettings, tmp_path: Path) -> None:
    dlg = _dialog(qtbot, settings)
    fake = FakeRunner([("frames", (0, 10, 0)), ("failed", ("디스크가 가득 찼습니다", "Traceback ..."))])
    shown: list[str] = []
    dlg.make_runner = lambda: fake  # type: ignore[method-assign]
    dlg.show_failure = shown.append  # type: ignore[method-assign]
    dlg.ask_save_path = lambda fmt, default: tmp_path / "a.mp4"  # type: ignore[method-assign]
    assert dlg.save() is None
    assert shown == ["디스크가 가득 찼습니다"]
    assert dlg.status.text() == S.EXP_FAILED.format(reason="디스크가 가득 찼습니다") and not dlg.btn_open_folder.isVisible()
    assert dlg.btn_save.isEnabled()


def test_cancelled_save_and_closed_file_dialog(qtbot: Any, settings: QSettings, tmp_path: Path) -> None:
    dlg = _dialog(qtbot, settings)
    dlg.ask_save_path = lambda fmt, default: None  # type: ignore[method-assign]
    assert dlg.save() is None
    fake = FakeRunner([("frames", (0, 10, 0)), ("frames", (1, 10, 100))])
    dlg.make_runner = lambda: fake  # type: ignore[method-assign]
    dlg.ask_save_path = lambda fmt, default: tmp_path / "a.mp4"  # type: ignore[method-assign]
    from PySide6.QtCore import QTimer

    QTimer.singleShot(200, fake.cancel)  # 진행 창의 「취소」와 같음
    assert dlg.save() is None and fake.cancel_calls >= 1 and not dlg.status.isVisible()


# ---------------------------------------------------------------------------
# 진짜 자식 프로세스로 세 형식
# ---------------------------------------------------------------------------
def test_saves_each_format_through_the_child_process(qtbot: Any, user_dirs: Path, settings: QSettings) -> None:
    folder = user_dirs / "내보내기"
    dlg = _dialog(qtbot, settings, folder=folder, default_name="1세대 vs 지금")
    told: list[bool] = []
    dlg.show_how_to = lambda: told.append(True)  # type: ignore[method-assign]
    picked: list[tuple[str, Path]] = []

    def ask(fmt: str, default: Path) -> Path:
        picked.append((fmt, default))
        return default.with_suffix("") if fmt == "gif" else default  # GIF는 확장자 없이 골라도 붙는다

    dlg.ask_save_path = ask  # type: ignore[method-assign]
    for fmt in ("gif", "html", "mp4", "html"):
        dlg.set_format(fmt)
        if fmt == "mp4":
            dlg.size_combo.setCurrentIndex(list(J.MP4_SIZES).index("small"))
        out = dlg.save()
        want = folder / f"1세대 vs 지금{J.SUFFIXES[fmt]}"
        assert out == want and out.is_file(), dlg.status.text()
        assert dlg.status.text().startswith(S.EXP_DONE.format(path=want)) and dlg.btn_open_folder.isVisible()
        assert dlg.last_result is not None and dlg.last_result["bytes"] == want.stat().st_size
    assert [f for f, _d in picked] == ["gif", "html", "mp4", "html"]
    assert picked[0][1] == folder / "1세대 vs 지금.gif"
    assert told == [True]  # 폰으로 옮기는 법은 이 대화 상자에서 한 번
    info = mp4.read_info(folder / "1세대 vs 지금.mp4")
    assert (info["width"], info["height"]) == (640, 360)
    assert settings.value(D.KEY_FORMAT) == "html" and settings.value(D.KEY_SIZE) == "small"
    again = _dialog(qtbot, settings)  # 마지막 선택을 기억한다
    assert again.fmt() == "html" and again.size_name() == "small" and again.camera() == "fixed"
    settings.setValue(D.KEY_HOW_TO_HIDDEN, True)  # 다시 보지 않기
    hidden = _dialog(qtbot, settings, folder=folder)
    hidden.show_how_to = lambda: pytest.fail("다시 보지 않기를 골랐다")  # type: ignore[method-assign]
    hidden.ask_save_path = lambda fmt, default: folder / "두번째.html"  # type: ignore[method-assign]
    hidden.set_format("html")
    assert hidden.save() == folder / "두번째.html"


def test_choose_and_export_returns_the_saved_path(qtbot: Any, settings: QSettings, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[D.ExportDialog] = []

    def fake_exec(self: D.ExportDialog) -> int:
        made.append(self)
        self.saved_path = tmp_path / "x.gif"
        return 0

    monkeypatch.setattr(D.ExportDialog, "exec", fake_exec)
    got = D.choose_and_export(None, kind="replay", items=[(_replay(), QColor(J.NOW_COLOR), "a")], title="a",
                              default_name="a", meta={J.META_RUN: "실행"}, view=(0.0, 0.0, 10.0, 10.0),
                              folder=tmp_path, settings=settings)
    assert got == tmp_path / "x.gif" and made[0].view == (0.0, 0.0, 10.0, 10.0)
    assert made[0].build_job().items[0].view == (0.0, 0.0, 10.0, 10.0)
    assert made[0].build_job().meta[J.META_RUN] == "실행"


# ---------------------------------------------------------------------------
# 실험실·경주 화면의 입구
# ---------------------------------------------------------------------------
def test_lab_save_video_without_path_opens_the_export_dialog(qtbot: Any, user_dirs: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.lab.view import COLOR_NOW, COLOR_OLD, LabView

    monkeypatch.setattr(paths, "runs_dir", lambda: user_dirs / "runs")
    lab = LabView(settings=QSettings(str(user_dirs / "settings.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(lab)
    try:
        assert lab.btn_video.text() == S.VIDEO_SAVE
        calls: list[dict[str, Any]] = []

        def fake(parent: QWidget, **kw: Any) -> Path:
            calls.append({"parent": parent, **kw})
            return user_dirs / "영상.mp4"

        monkeypatch.setattr(D, "choose_and_export", fake)
        items = [(_replay(), COLOR_OLD, "1세대 대표"), (_replay("lander", 2), COLOR_NOW, "지금 챔피언 (4세대)")]
        lab._last_video = (items, "1세대 대표 vs 지금 챔피언 (4세대)")
        lab._replay_view = (-5.0, -1.0, 5.0, 4.0)
        before = lab.status.text()
        assert lab.save_video() == user_dirs / "영상.mp4"
        (call,) = calls
        assert call["parent"] is lab and call["kind"] == "replay" and call["items"] is items
        assert call["title"] == "1세대 대표 vs 지금 챔피언 (4세대)" and call["view"] == (-5.0, -1.0, 5.0, 4.0)
        assert call["folder"] == paths.data_dir() and call["settings"] is lab.settings
        assert set(call["meta"]) <= set(J.META_KEYS)
        assert lab.status.text() == before + "\n" + S.EXP_DONE.format(path=user_dirs / "영상.mp4")
    finally:
        lab.client.abandon(timeout=60)
        lab.shutdown()


def test_race_save_gif_without_path_opens_the_export_dialog(qtbot: Any, user_dirs: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.race import model as M
    from evolab.app.race.view import show_race

    runs = []
    for name in ("v10_run_ga", "v11_run_restart"):
        runs.append(user_dirs / "runs" / name)
        shutil.copytree(DATA / name / "run", runs[-1])
    entries, _warnings, _n = M.entries_from_runs([(runs[0], "#3987e5"), (runs[1], "#d95926")], "last")
    owner = QWidget()
    qtbot.addWidget(owner)
    owner.show()
    dlg = show_race(owner, entries, title=S.RACE_TITLE)
    view = dlg.view
    qtbot.waitUntil(lambda: view.ready, timeout=60_000)
    view.pause()
    calls: list[dict[str, Any]] = []
    out = user_dirs / "경주.html"
    monkeypatch.setattr(D, "choose_and_export", lambda parent, **kw: calls.append({"parent": parent, **kw}) or out)
    assert view.gif_button.text() == S.RACE_SAVE_GIF
    assert view.save_gif() == out
    (call,) = calls
    assert call["parent"] is view and call["kind"] == "race" and call["overlay"] is False
    assert [lane.entry for lane in call["items"]] == [lane.entry for lane in view.lanes]
    assert call["title"] == S.VIDEO_RACE_TITLE.format(n=view.course)
    assert call["folder"] == runs[0] / "exports" and call["default_name"] == view.default_gif_path().stem
    assert view.message.text() == S.EXP_DONE.format(path=out)
    # 대화 상자에서도 같은 칸으로 작업이 만들어진다(재생이 있는 칸만, 칸 격자)
    real = D.ExportDialog(view, kind="race", items=call["items"], title=call["title"],
                          default_name=call["default_name"], meta={}, folder=call["folder"], probe=False,
                          settings=QSettings(str(user_dirs / "s.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(real)
    real.set_format("gif")
    job = real.build_job()
    assert job.layout == "lanes" and len(job.items) == len(view.replays) and job.size == (640, 200)
    dlg.close()

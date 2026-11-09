"""내보내기 대화 상자 (v3 설계 D48·D50, 4.14, 5.6, 7.3 「choose_and_export」).

실험실 「내보내기…」와 경주 창 「내보내기…」가 같은 대화 상자를 연다: 무엇을(지금 보는 재생 / 이 경주), 형식(MP4 / GIF / 폰 재생
파일), 크기(MP4만: 작게 640×360 / 보통 1280×720), 카메라(고정 / 따라가기, 폰 파일이면 숨김), 안내 줄(길이·프레임 수, MP4면 인코더:
열 때 작은 자식으로 한 번 시험한 결과, MF면 예상 크기 = 비트레이트 × 길이, 내장 쓰기면 EXP_INFO_BUILTIN + 풍선 까닭).
「저장…」 → 파일 대화 상자 → 내보내기 자식 프로세스(runner.ExportRunner) + 진행 창(「만드는 중… k/n프레임 · 지금까지 nMB」, 취소).
내장 쓰기·따라가기에서 2초 몫을 쓴 크기로 전체를 예상해 200MB를 넘으면 한 번 묻는다(아니요면 취소). 끝나면 「저장했습니다」 +
「폴더 열기」, 폰 파일을 이 대화 상자에서 처음 저장하면 「폰으로 옮기는 법」(다시 보지 않기). 실패는 errors.show_error 상자(까닭
글, 자식의 traceback은 errors.jsonl). 마지막 선택은 settings.ini [export].

테스트는 ExportDialog의 ask_save_path·ask_big·show_how_to·show_failure·make_runner를 바꿔 끼운다.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

from PySide6.QtCore import QEventLoop, QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from evolab import paths
from evolab.app import strings as S
from evolab.app.export import job as J
from evolab.app.export.job import ExportJob

BIG_MB: Final = 200.0
"""내장 쓰기·따라가기에서 예상 크기가 이것(MB)을 넘으면 묻는다(설계 4.14)."""
BIG_SAMPLE_S: Final = 2.0
"""예상에 쓰는 앞부분(초): 이만큼 쓴 크기로 전체를 비례해 예상한다."""
KEY_FORMAT: Final = "export/format"
KEY_SIZE: Final = "export/size"
KEY_CAMERA: Final = "export/camera"
KEY_HOW_TO_HIDDEN: Final = "export/phone_how_to_hidden"
FILTERS: Final[dict[str, str]] = {"mp4": S.EXP_FILTER_MP4, "gif": S.VIDEO_FILTER, "html": S.EXP_FILTER_HTML}
FORMAT_TEXT: Final[dict[str, str]] = {"mp4": S.EXP_FORMAT_MP4, "gif": S.EXP_FORMAT_GIF, "html": S.EXP_FORMAT_HTML}
SIZE_TEXT: Final[dict[str, str]] = {"small": S.EXP_SIZE_SMALL, "medium": S.EXP_SIZE_MEDIUM}
CAMERA_TEXT: Final[dict[str, str]] = {"fixed": S.EXP_CAMERA_FIXED, "follow": S.EXP_CAMERA_FOLLOW}


def settings_store() -> QSettings:
    """settings.ini(실험실 settings_store와 같은 파일)."""
    return QSettings(str(paths.user_state_dir() / "settings.ini"), QSettings.Format.IniFormat)


def _mb(n: float) -> float:
    return float(n) / 1e6


class ExportDialog(QDialog):
    """내보내기 대화 상자(설계 5.6). kind="replay": items = (재생, 색, 이름) 목록(겹쳐 보기, 실험실 _last_video[0]).
    kind="race": items = 경주 창의 칸(.replay, .entry) 목록, overlay면 「한 화면에 겹쳐서」."""

    def __init__(self, parent: QWidget | None, *, kind: Literal["replay", "race"], items: Any, title: str,
                 default_name: str, meta: Mapping[str, str] | None = None, view: J.ViewBox | None = None,
                 folder: Path | None = None, overlay: bool = False, settings: QSettings | None = None,
                 probe: bool = True) -> None:
        super().__init__(parent)
        if kind not in ("replay", "race"):
            raise ValueError(f"kind must be 'replay' or 'race', got {kind!r}")
        self.setWindowTitle(S.VIDEO_SAVE_TITLE)
        self.kind = kind
        self.items = list(items)
        self.title = str(title)
        self.default_name = str(default_name) or "export"
        self.meta = dict(meta or {})
        self.view = view
        self.folder = Path(folder) if folder is not None else None
        self.overlay = bool(overlay)
        self.settings = settings if settings is not None else settings_store()
        self.saved_path: Path | None = None
        self.last_result: dict[str, Any] | None = None
        self.encoder: tuple[str, str] | None = None
        """MP4 인코더 시험 결과 (kind, detail), 아직이면 None."""
        self.runner: Any = None
        self._phone_told = False

        lay = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.addWidget(QLabel(S.EXP_WHAT), 0, 0, Qt.AlignmentFlag.AlignTop)
        self.what_label = QLabel(self._what_text())
        self.what_label.setWordWrap(True)
        grid.addWidget(self.what_label, 0, 1)
        grid.addWidget(QLabel(S.EXP_FORMAT), 1, 0, Qt.AlignmentFlag.AlignTop)
        formats = QVBoxLayout()
        self.format_group = QButtonGroup(self)
        self.format_buttons: dict[str, QRadioButton] = {}
        for i, fmt in enumerate(J.FORMATS):
            rb = QRadioButton(FORMAT_TEXT[fmt])
            self.format_group.addButton(rb, i)
            self.format_buttons[fmt] = rb
            formats.addWidget(rb)
        grid.addLayout(formats, 1, 1)
        self.size_label = QLabel(S.EXP_SIZE)
        grid.addWidget(self.size_label, 2, 0)
        row = QHBoxLayout()
        self.size_combo = QComboBox()
        for name in J.MP4_SIZES:
            self.size_combo.addItem(SIZE_TEXT[name], name)
        row.addWidget(self.size_combo)
        row.addSpacing(16)
        self.camera_label = QLabel(S.EXP_CAMERA)
        row.addWidget(self.camera_label)
        self.camera_combo = QComboBox()
        for name in J.CAMERAS:
            self.camera_combo.addItem(CAMERA_TEXT[name], name)
        row.addWidget(self.camera_combo)
        row.addStretch(1)
        grid.addLayout(row, 2, 1)
        lay.addLayout(grid)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        lay.addWidget(line)
        self.length_label = QLabel()
        lay.addWidget(self.length_label)
        self.encoder_label = QLabel()
        self.encoder_label.setWordWrap(True)
        lay.addWidget(self.encoder_label)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.hide()
        lay.addWidget(self.status)
        buttons = QHBoxLayout()
        self.btn_open_folder = QPushButton(S.EXP_OPEN_FOLDER)
        self.btn_open_folder.clicked.connect(self.open_folder)
        self.btn_open_folder.hide()
        buttons.addWidget(self.btn_open_folder)
        buttons.addStretch(1)
        self.btn_save = QPushButton(S.EXP_SAVE)
        self.btn_save.setDefault(True)
        self.btn_save.clicked.connect(self.save)
        self.btn_close = QPushButton(S.EXP_CLOSE)
        self.btn_close.clicked.connect(self.reject)
        buttons.addWidget(self.btn_save)
        buttons.addWidget(self.btn_close)
        lay.addLayout(buttons)

        self._restore()
        self.format_group.idToggled.connect(lambda _i, on: on and self._refresh())
        self.size_combo.currentIndexChanged.connect(lambda _i: self._refresh())
        self.camera_combo.currentIndexChanged.connect(lambda _i: self._refresh())
        self._refresh()
        self.probe: Any = None
        if probe:
            from evolab.app.export.runner import EncoderProbe

            self.probe = EncoderProbe(self)
            self.probe.done.connect(self.set_encoder)
            self.probe.start()

    # ------------------------------------------------------------------
    # 고른 것
    # ------------------------------------------------------------------
    def _what_text(self) -> str:
        if self.kind == "race":
            return S.EXP_WHAT_RACE.format(n=len(self._race_lanes()))
        return S.EXP_WHAT_REPLAY.format(label=self.title)

    def _race_lanes(self) -> list[Any]:
        return [lane for lane in self.items if isinstance(lane, tuple) or getattr(lane, "replay", None) is not None]

    def fmt(self) -> str:
        for fmt, rb in self.format_buttons.items():
            if rb.isChecked():
                return fmt
        return "gif"

    def set_format(self, fmt: str) -> None:
        self.format_buttons[fmt].setChecked(True)

    def size_name(self) -> str:
        return str(self.size_combo.currentData())

    def camera(self) -> str:
        return str(self.camera_combo.currentData())

    def _restore(self) -> None:
        def pick(key: str, allowed: Sequence[str], default: str) -> str:
            value = str(self.settings.value(key, default) or default)
            return value if value in allowed else default

        self.set_format(pick(KEY_FORMAT, J.FORMATS, "mp4"))
        self.size_combo.setCurrentIndex(list(J.MP4_SIZES).index(pick(KEY_SIZE, tuple(J.MP4_SIZES), J.DEFAULT_SIZE_NAME)))
        self.camera_combo.setCurrentIndex(list(J.CAMERAS).index(pick(KEY_CAMERA, J.CAMERAS, "fixed")))

    def _remember(self) -> None:
        self.settings.setValue(KEY_FORMAT, self.fmt())
        self.settings.setValue(KEY_SIZE, self.size_name())
        self.settings.setValue(KEY_CAMERA, self.camera())

    def build_job(self) -> ExportJob:
        """지금 고른 형식·크기·카메라의 작업(설계 2.6). 재생이 없으면 ValueError(VIDEO_NO_FRAMES). MP4의 인코더는
        _mp4_encoder(인코더 시험 결과를 자식에게 넘김)."""
        job = self._build_job()
        if job.fmt == "mp4":
            encoder = self._mp4_encoder()
            if encoder != job.encoder:
                job = dataclasses.replace(job, encoder=encoder)
        return job

    def _mp4_encoder(self) -> Literal["auto", "builtin"]:
        """MP4 작업의 인코더: 시험이 MF를 쓸 수 있다고 했을 때만 "auto"(자식이 MF를 열고, 쓰는 중 실패하면 내장 쓰기로 다시
        만듦: open_writer(prefer="mf")는 자식의 시험이 실패하면 대체 없이 EncoderUnavailable이라 "mf"는 넘기지 않는다).
        시험 전·시간 초과·시험 자식이 죽음·MF 실패면 "builtin": 자식이 MF 시험을 (시간 한도 없이) 다시 하지 않게, 그리고
        안내 줄(EXP_INFO_BUILTIN)이 말한 대로. 내장 쓰기를 뺀 판(BUILTIN_ENABLED가 False)은 "auto" 그대로."""
        from evolab.app.export import mp4

        if self.encoder is not None and self.encoder[0] == "mf":
            return "auto"
        return "builtin" if mp4.BUILTIN_ENABLED else "auto"

    def _build_job(self) -> ExportJob:
        fmt, size, camera = self.fmt(), self.size_name(), self.camera()
        if self.kind == "replay":
            return J.job_from_replays(self.items, fmt, size=size, camera=camera, title=self.title, meta=self.meta,
                                      view=self.view)
        from evolab.app.race.model import lane_grid

        lanes = self._race_lanes()
        if not lanes:
            raise ValueError(S.VIDEO_NO_FRAMES)
        job = J.job_from_race(lanes, lane_grid(min(len(lanes), J.MAX_ITEMS))[1], fmt, size=size, camera=camera,
                              title=self.title)
        if not self.overlay:
            return job
        # 「한 화면에 겹쳐서」: 같은 칸들을 한 세계에 겹쳐(v3.0 경주 GIF의 겹친 영상), 순위·정보 줄은 칸 작업 그대로
        merged = J.job_from_replays([(it.replay, it.color, it.label) for it in job.items], fmt, size=size,
                                    camera=camera, title=self.title, meta=job.meta, view=job.items[0].view)
        return dataclasses.replace(merged, ranks=job.ranks)

    # ------------------------------------------------------------------
    # 안내 줄
    # ------------------------------------------------------------------
    def set_encoder(self, kind: str, detail: str) -> None:
        """MP4 인코더 시험 결과(EncoderProbe.done). 「none」이면 MP4를 고를 수 없다(Q4 결정이 내장 쓰기를 뺀 경우)."""
        self.encoder = (str(kind), str(detail))
        mp4_ok = kind != "none"
        self.format_buttons["mp4"].setEnabled(mp4_ok)
        self.format_buttons["mp4"].setToolTip("" if mp4_ok else S.EXP_INFO_BUILTIN_TIP.format(reason=detail))
        if not mp4_ok and self.fmt() == "mp4":
            self.set_format("gif")
        self._refresh()

    def _refresh(self) -> None:
        try:
            self._refresh_lines()
        finally:
            self._fit()

    def _fit(self) -> None:
        """보이는 줄이 늘면 창을 그만큼 키운다(줄이지는 않는다). Qt는 맨 위 창의 최소 높이에 줄 바꿈 글의 높이(heightForWidth)를
        넣지 않아, 저장 뒤 세 줄 상태 글이나 MP4의 두 줄 인코더 안내가 생기면 형식 단추 줄이 겹쳐 그려졌다(v3.1 통합 그림)."""
        lay = self.layout()
        if lay is None or not self.isVisible():
            return
        lay.activate()
        need = lay.totalHeightForWidth(self.width()) if lay.hasHeightForWidth() else lay.totalSizeHint().height()
        need = max(need, lay.totalMinimumSize().height())
        if need > self.height():
            self.resize(self.width(), need)

    def _refresh_lines(self) -> None:
        fmt = self.fmt()
        is_mp4 = fmt == "mp4"
        self.size_combo.setVisible(is_mp4)
        self.size_label.setVisible(is_mp4)
        self.camera_label.setVisible(fmt != "html")
        self.camera_combo.setVisible(fmt != "html")
        try:
            job = self.build_job()
        except ValueError as exc:
            from evolab.app import errors

            self.length_label.setText(errors.reason(exc))
            self.encoder_label.hide()
            self.btn_save.setEnabled(False)
            return
        self.btn_save.setEnabled(True)
        seconds, frames = self.length_of(job)
        self.length_label.setText(S.EXP_INFO_LENGTH.format(seconds=seconds, frames=frames))
        self.encoder_label.setVisible(is_mp4)
        self.encoder_label.setToolTip("")
        if not is_mp4:
            return
        if self.encoder is None:
            self.encoder_label.setText(S.EXP_INFO_PROBING)
            return
        kind, detail = self.encoder
        if kind == "mf":
            from evolab.app.export.mp4 import mf_bitrate

            mb = mf_bitrate(*job.size) * seconds / 8 / 1e6
            self.encoder_label.setText(S.EXP_INFO_MF.format(mb=mb))
            self.encoder_label.setToolTip(S.EXP_ENCODER_MF.format(name=detail))
        else:
            self.encoder_label.setText(S.EXP_INFO_BUILTIN)
            self.encoder_label.setToolTip(S.EXP_INFO_BUILTIN_TIP.format(reason=detail))

    @staticmethod
    def length_of(job: ExportJob) -> tuple[float, int]:
        """(길이 초, 프레임 수). MP4·GIF는 만들 프레임(끝난 뒤 보여 주는 꼬리 포함), 폰 파일은 재생 길이와 궤적 프레임."""
        if job.fmt == "html":
            return job.duration, max(int(it.replay.trajectory.frames) for it in job.items)
        from evolab.app.export.frames import frame_count

        n = frame_count(job)
        return (n - 1) / job.fps, n

    def builtin_now(self, job: ExportJob) -> bool:
        """이 MP4 작업을 내장 쓰기로 만드는 중인가(시험 결과·작업 인코더·MF 대체 알림·같은 프로세스 길)."""
        if job.fmt != "mp4":
            return False
        if job.encoder == "builtin" or (self.runner is not None and self.runner.encoder_hint == "builtin"):
            return True
        return self.encoder is None or self.encoder[0] != "mf"

    # ------------------------------------------------------------------
    # 저장
    # ------------------------------------------------------------------
    def default_path(self, fmt: str) -> Path:
        folder = self.folder if self.folder is not None else paths.data_dir()
        return folder / f"{self.default_name}{J.SUFFIXES[fmt]}"

    def ask_save_path(self, fmt: str, default: Path) -> Path | None:
        """저장할 파일을 묻는다. 테스트는 이 메서드를 바꿔 끼운다."""
        chosen, _ = QFileDialog.getSaveFileName(self, S.VIDEO_SAVE_TITLE, str(default), FILTERS[fmt])
        return Path(chosen) if chosen else None

    def ask_big(self, mb: float) -> bool:
        """예상 크기가 클 때 그대로 만들지 묻는다(기본 아니요). 테스트는 바꿔 끼운다."""
        answer = QMessageBox.question(self, S.VIDEO_SAVE_TITLE,
                                      S.EXP_BIG_ASK.format(mb=mb, fixed=S.EXP_CAMERA_FIXED),
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def show_how_to(self) -> None:
        """폰으로 옮기는 법(다시 보지 않기). 테스트는 바꿔 끼운다."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle(S.PHONE_HOW_TO_TITLE)
        box.setText(S.PHONE_HOW_TO)
        check = QCheckBox(S.PHONE_HOW_TO_DONT_SHOW)
        box.setCheckBox(check)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()
        if check.isChecked():
            self.settings.setValue(KEY_HOW_TO_HIDDEN, True)

    def show_failure(self, message: str) -> None:
        """실패 상자(다음 할 일 + 오류 보고 묶음 단추). 테스트는 바꿔 끼운다."""
        from evolab.app import errors

        errors.show_error(self, S.VIDEO_SAVE_TITLE, S.EXP_FAILED.format(reason=message))

    def make_runner(self) -> Any:
        """내보내기 실행기(runner.ExportRunner). 테스트는 가짜로 바꿔 끼운다."""
        from evolab.app.export.runner import ExportRunner

        return ExportRunner(self)

    def save(self) -> Path | None:
        """「저장…」: 파일을 묻고 만든다. 만든 경로(취소·실패면 None)."""
        try:
            job = self.build_job()
        except ValueError as exc:
            from evolab.app import errors

            self.show_failure(errors.reason(exc))
            return None
        path = self.ask_save_path(job.fmt, self.default_path(job.fmt))
        if path is None:
            return None
        suffix = J.SUFFIXES[job.fmt]
        if path.suffix.lower() != suffix:  # 다른 형식의 확장자는 바꾸고, 그 밖(없음, 「실험 1.2」)은 붙인다
            path = path.with_suffix(suffix) if J.format_for_path(path) is not None else path.with_name(path.name + suffix)
        self._remember()
        return self.export_to(job, path)

    def export_to(self, job: ExportJob, path: Path) -> Path | None:
        """작업을 path로 만든다(진행 창, 취소). 만든 경로, 취소·실패면 None."""
        self.runner = runner = self.make_runner()
        self.status.setText("")  # 앞 저장의 글은 지운다(이번 알림·결과만)
        self.status.hide()
        self.btn_open_folder.hide()
        n_hint = max(1, self.length_of(job)[1]) if job.fmt != "html" else 1
        dlg = QProgressDialog(S.EXP_PROGRESS_FRAMES.format(k=0, n=n_hint, mb=0.0), S.VIDEO_CANCEL, 0, n_hint, self)
        dlg.setWindowTitle(S.VIDEO_SAVE_TITLE)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(300)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        outcome: dict[str, Any] = {}
        loop = QEventLoop()
        asked = [False]
        sample = max(1, int(round(BIG_SAMPLE_S * job.fps)))
        follow = job.camera == "follow"

        def on_progress(k: int, n: int, written: int) -> None:
            if n > 0:
                dlg.setMaximum(n)
            dlg.setValue(min(k, max(n, 1)))
            dlg.setLabelText(S.EXP_PROGRESS_FRAMES.format(k=k, n=max(n, 1), mb=_mb(written)))
            if (follow and not asked[0] and k >= sample and n > 0 and self.builtin_now(job)
                    and _mb(written) / k * n > BIG_MB):
                asked[0] = True
                QTimer.singleShot(0, lambda: ask(_mb(written) / k * n))

        def ask(mb: float) -> None:
            if runner.running and not self.ask_big(mb):
                runner.cancel()

        def on_note(text: str) -> None:
            self.status.setText(text)
            self.status.show()
            self._fit()

        def on_finished(done: dict[str, Any]) -> None:
            outcome["done"] = done
            loop.quit()

        def on_failed(message: str, detail: str) -> None:
            outcome["failed"] = (message, detail)
            loop.quit()

        def on_cancelled() -> None:
            outcome["cancelled"] = True
            loop.quit()

        runner.progressed.connect(on_progress)
        runner.noted.connect(on_note)
        runner.finished.connect(on_finished)
        runner.failed.connect(on_failed)
        runner.cancelled.connect(on_cancelled)
        dlg.canceled.connect(runner.cancel)
        self.btn_save.setEnabled(False)
        try:
            runner.start(job, path)
            if runner.running and not outcome:
                loop.exec()
        finally:
            dlg.canceled.disconnect(runner.cancel)
            dlg.close()
            dlg.deleteLater()
            self.btn_save.setEnabled(True)
        if "done" in outcome:
            done = outcome["done"]
            self.last_result = done
            self.saved_path = Path(done.get("path", path))
            note = self.status.text() if self.status.isVisible() and self.status.text() else ""
            lines = [S.EXP_DONE.format(path=self.saved_path)] + ([note] if note else [])
            self.status.setText("\n".join(lines))
            self.status.show()
            self.btn_open_folder.show()
            self._fit()
            if job.fmt == "html" and not self._phone_told and not self._how_to_hidden():
                self._phone_told = True
                self.show_how_to()
            return self.saved_path
        if "failed" in outcome:
            message, _detail = outcome["failed"]
            self.status.setText(S.EXP_FAILED.format(reason=message))
            self.status.show()
            self._fit()
            self.show_failure(message)
        return None

    def _how_to_hidden(self) -> bool:
        value = self.settings.value(KEY_HOW_TO_HIDDEN, False)
        return value is True or str(value).lower() in ("true", "1")

    def open_folder(self) -> None:
        if self.saved_path is not None:
            from evolab.app.report.bundle import reveal

            reveal(self.saved_path)


def choose_and_export(parent: QWidget | None, *, kind: Literal["replay", "race"], items: Any, title: str,
                      default_name: str, meta: Mapping[str, str], view: J.ViewBox | None = None,
                      folder: Path | None = None, overlay: bool = False,
                      settings: QSettings | None = None) -> Path | None:
    """내보내기 대화 상자(모달). 마지막으로 만든 파일 경로, 만들지 않고 닫으면 None(설계 7.3, 키워드 view·folder·overlay·
    settings를 더함: 실험실 재생 환경의 기본 보기, 기본 폴더, 경주 「겹쳐서」, 설정 파일)."""
    dlg = ExportDialog(parent, kind=kind, items=items, title=title, default_name=default_name, meta=meta, view=view,
                       folder=folder, overlay=overlay, settings=settings)
    try:
        dlg.exec()
        return dlg.saved_path
    finally:
        dlg.deleteLater()


__all__ = ["BIG_MB", "ExportDialog", "choose_and_export", "settings_store"]

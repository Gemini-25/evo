"""「새 프리셋」 창의 미리보기 (6.2.0, 최종 설계 4.1 「W의 미리보기 ▶는 자기 것」, DU 4.1).

지금 조리법으로 만든 환경을 실험실 관전 캔버스(SpectatorCanvas)로 그린다: 세계와 몸의 시작 자세(전시 시험). 「가만히 두고 보기」를
누르면 행동 0으로 몇 초(PREVIEW_SECONDS, 에피소드가 먼저 끝나면 거기까지) 돌린 궤적을 그 자리에서 다시 그린다. 에디터의 시험 주행에
기대지 않는다. 문서가 바뀌면 돌던 재생을 멈추고 시작 자세로 돌아간다.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PySide6.QtCore import QElapsedTimer, Qt, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from evolab.app import strings as S
from evolab.app import style
from evolab.app.lab.canvas import SpectatorCanvas
from evolab.engine.evaluate import Trajectory
from evolab.engine.seeds import display_trial
from evolab.spec.compiled import EnvSpec
from evolab.spec.editor_meta import view_box
from evolab.world.physics2d.episode import Episode, EpisodeBuildError

PREVIEW_SECONDS = 4.0
"""「가만히 두고 보기」가 돌리는 길이(초)."""
EVERY = 2
"""궤적 프레임 간격(스텝)."""
FRAME_MS = 33
"""재생 타이머 간격(ms)."""


class PresetPreview(QWidget):
    """미리보기: 위에 제목 줄과 「가만히 두고 보기」 단추, 아래 캔버스(남는 자리를 모두 씀: 내용 상자에 맞춤)."""

    drawn = Signal()
    """장면을 다시 그릴 준비가 끝났다(문서가 바뀐 뒤 한 번)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec: EnvSpec | None = None
        self.trajectory: Trajectory | None = None
        """마지막으로 돌린 짧은 재생(없으면 None)."""
        self.build_ms = 0.0
        """마지막 set_spec이 장면을 만드는 데 든 시간(ms, 측정용)."""
        self.title = QLabel(S.PRESET_PREVIEW)
        self.title.setToolTip(S.PRESET_PREVIEW_TIP)
        style.tone(self.title, "ink2")
        self.btn_run = QPushButton(S.PRESET_PREVIEW_RUN)
        self.btn_run.setToolTip(S.PRESET_PREVIEW_RUN_TIP)
        self.btn_run.setAutoDefault(False)
        self.btn_run.setDefault(False)
        self.btn_run.clicked.connect(self.toggle_run)
        self.canvas = SpectatorCanvas(self)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.canvas.setMinimumSize(200, 160)
        self.canvas.setToolTip(S.PRESET_PREVIEW_TIP)
        self.canvas.set_wheel_needs_ctrl(False)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self.title, 1)
        head.addWidget(self.btn_run)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(style.SPACE[0])
        root.addLayout(head)
        root.addWidget(self.canvas, 1)
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._tick)
        self._clock = QElapsedTimer()
        self._start_pose: Trajectory | None = None
        self._set_running(False)

    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._timer.isActive()

    def set_spec(self, spec: EnvSpec | None) -> None:
        """새 환경(컴파일한 것). None이면 빈 그림. 돌던 재생은 멈춘다."""
        t0 = time.perf_counter()
        self._timer.stop()
        self._set_running(False)
        self.spec = spec
        self.trajectory = None
        self._start_pose = None
        if spec is None:
            self.canvas.world = None
            self.canvas.update()
            self.btn_run.setEnabled(False)
            self.build_ms = (time.perf_counter() - t0) * 1000.0
            return
        try:
            ep = Episode(spec, None, display_trial())
        except EpisodeBuildError:
            self.canvas.world = None
            self.canvas.update()
            style.set_enabled(self.btn_run, False, S.PRESET_PROBE_BUILD)
            self.build_ms = (time.perf_counter() - t0) * 1000.0
            return
        geometry = ep.geometry()
        self.canvas.set_scene(geometry, None, list(ep.actuator_channels()), view_box(spec.file))
        self._start_pose = self._frames(ep, 0.0)
        self.canvas.set_population([self._start_pose], ranks=np.zeros(1))
        self.canvas.set_time(0.0)
        style.set_enabled(self.btn_run, True)
        self.build_ms = (time.perf_counter() - t0) * 1000.0
        self.drawn.emit()

    @staticmethod
    def _frames(ep: Episode, seconds: float) -> Trajectory:
        """행동 0으로 seconds초(또는 끝날 때)까지 돌린 궤적. 0이면 시작 자세 한 프레임."""
        zero = np.zeros(len(ep.outputs), dtype=np.float64)
        first = ep.snapshot()
        poses = [first.poses.astype(np.float32)]
        outs = [first.outputs.astype(np.float32)]
        steps = 0
        limit = int(round(seconds / ep.dt)) if seconds > 0 else 0
        while steps < limit and not ep.done:
            ep.step(zero)
            steps += 1
            if ep.done or steps % EVERY == 0:
                f = ep.snapshot()
                poses.append(f.poses.astype(np.float32))
                outs.append(f.outputs.astype(np.float32))
        n_b = poses[0].shape[0]
        n_u = outs[0].shape[0]
        return Trajectory(np.stack(poses).reshape(len(poses), n_b, 3), np.stack(outs).reshape(len(outs), n_u),
                          EVERY, float(ep.dt), steps)

    def run_still(self) -> Trajectory | None:
        """행동 0으로 짧게 돌려 궤적을 만든다(재생은 시작하지 않음). 조립이 안 되면 None."""
        if self.spec is None:
            return None
        try:
            ep = Episode(self.spec, None, display_trial())
        except EpisodeBuildError:
            return None
        self.trajectory = self._frames(ep, PREVIEW_SECONDS)
        return self.trajectory

    def toggle_run(self) -> None:
        if self.running:
            self.stop()
            return
        traj = self.run_still()
        if traj is None:
            return
        self.canvas.set_population([traj], ranks=np.zeros(1))
        self.canvas.set_time(0.0)
        self._clock.start()
        self._timer.start()
        self._set_running(True)

    def stop(self) -> None:
        """재생을 멈추고 시작 자세로 돌아간다."""
        self._timer.stop()
        self._set_running(False)
        if self._start_pose is not None and self.canvas.world is not None:
            self.canvas.set_population([self._start_pose], ranks=np.zeros(1))
            self.canvas.set_time(0.0)

    def _tick(self) -> None:
        if self.trajectory is None:
            self.stop()
            return
        t = self._clock.elapsed() / 1000.0
        if t >= self.trajectory.duration:
            self.canvas.set_time(self.trajectory.duration)
            self._timer.stop()
            self._set_running(False)
            return
        self.canvas.set_time(t)

    def _set_running(self, on: bool) -> None:
        self.btn_run.setText(S.PRESET_PREVIEW_STOP if on else S.PRESET_PREVIEW_RUN)

    def info(self) -> dict[str, Any]:
        """테스트·측정용: 지금 그린 것."""
        return {"world": self.canvas.world is not None, "running": self.running,
                "frames": self.trajectory.frames if self.trajectory is not None else 0}


__all__ = ["PREVIEW_SECONDS", "PresetPreview"]

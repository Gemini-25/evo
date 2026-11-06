"""실험실 화면 (명세: 화면과 조작 > 실험실).

가운데 관전 화면, 오른쪽 제어 패널, 아래 통계 대시보드. 학습은 학습 프로세스(TrainingClient)가 하고, 이 화면은
결과를 받아 그리기만 한다. 다시 시뮬레이션(타임라인, 1세대 vs 지금, 고속 모드 챔피언, 개체 클릭 검사)은
백그라운드 스레드에서 같은 런타임 코드로 한다.
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QElapsedTimer, QObject, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from evolab import paths
from evolab.app import strings as S
from evolab.app.lab.canvas import SpectatorCanvas
from evolab.app.lab.client import TrainingClient
from evolab.app.lab.dashboard import Dashboard
from evolab.engine import seeds
from evolab.engine.pool import default_workers
from evolab.engine.replay import Replay, agent_geoms, replay
from evolab.engine.session import SessionRequest
from evolab.engine.trainer import TrainConfig
from evolab.policy.genome import GenomeLayout, layout_for
from evolab.spec.compiled import EnvFileError, EnvSpec
from evolab.spec.io import compile_env, env_hash
from evolab.world.physics2d.episode import Episode, EpisodeBuildError, Trial

SPEEDS = (1, 2, 4, 8)
COLOR_OLD = QColor(80, 140, 255)
COLOR_NOW = QColor(255, 120, 40)
MARK_KINDS = ("resume", "env_change", "config_change", "branch")


def settings_store() -> QSettings:
    return QSettings(str(paths.user_state_dir() / "settings.ini"), QSettings.Format.IniFormat)


# ---------------------------------------------------------------------------
# 백그라운드 다시 시뮬레이션
# ---------------------------------------------------------------------------


class _ReplayThread(QThread):
    finished_job = Signal(object, object)
    """(꼬리표, Replay 목록 또는 예외)"""

    def __init__(self, tag: Any, jobs: list[tuple[EnvSpec, GenomeLayout, np.ndarray, Trial, bool]],
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.tag = tag
        self.jobs = jobs

    def run(self) -> None:
        try:
            out = [replay(spec, layout, genome, trial, trace=trace) for spec, layout, genome, trial, trace in self.jobs]
        except Exception as exc:  # 결과를 화면 스레드로 넘겨 알린다
            self.finished_job.emit(self.tag, exc)
            return
        self.finished_job.emit(self.tag, out)


# ---------------------------------------------------------------------------
# 개체 들여다보기
# ---------------------------------------------------------------------------


class InspectPanel(QWidget):
    """클릭한 개체를 다시 시뮬레이션한 결과: 스텝마다 센서 값과 뉴런 활성화를 막대로 보여준다."""

    closed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.replay: Replay | None = None
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.time_label = QLabel()
        self.body = QLabel()
        self.body.setTextFormat(Qt.TextFormat.PlainText)
        self.body.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        font = self.body.font()
        font.setStyleHint(font.StyleHint.Monospace)
        font.setFamilies(["Consolas", "D2Coding", "Malgun Gothic", "monospace"])
        self.body.setFont(font)
        close = QPushButton(S.LAB_INSPECT_CLOSE)
        close.clicked.connect(self.closed.emit)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.body)
        lay = QVBoxLayout(self)
        lay.addWidget(self.title)
        lay.addWidget(self.time_label)
        lay.addWidget(scroll, 1)
        lay.addWidget(close)

    def set_replay(self, index: int, rep: Replay) -> None:
        self.replay = rep
        self.title.setText(S.LAB_INSPECT_TITLE.format(index=index, score=rep.result.score, end=rep.result.end_label))
        self.show_time(0.0)

    def show_time(self, t: float) -> None:
        rep = self.replay
        if rep is None or rep.observations is None or not len(rep.observations):
            return
        dt = rep.trajectory.dt
        k = int(min(len(rep.observations) - 1, max(0, math.floor(t / dt + 1e-9))))
        self.time_label.setText(S.LAB_INSPECT_TIME.format(t=k * dt, total=rep.result.steps * dt))
        lines = [S.LAB_INSPECT_SENSORS]
        raw = rep.raw[k] if rep.raw else {}
        for i, name in enumerate(rep.inputs):
            v = float(rep.observations[k, i])
            r = raw.get(name)
            lines.append(f"{name:<14} {'' if r is None else f'{r:9.3f}'} → {v:+6.2f} {_bar(v / 5.0)}")
        lines.append("")
        lines.append(S.LAB_INSPECT_LAYERS)
        acts = rep.activations or []
        for li, a in enumerate(acts):
            last = li == len(acts) - 1
            lines.append(S.LAB_INSPECT_OUTPUTS if last else S.LAB_INSPECT_LAYER.format(k=li + 1, n=a.shape[1]))
            row = a[k]
            names = rep.channels if last and len(rep.channels) == len(row) else [str(j) for j in range(len(row))]
            for name, v in zip(names, row, strict=True):
                lines.append(f"  {name:<12} {float(v):+5.2f} {_bar(float(v))}")
        self.body.setText("\n".join(lines))


def _bar(v: float, width: int = 10) -> str:
    v = max(-1.0, min(1.0, v))
    n = round(abs(v) * width)
    left = " " * (width - n) + "█" * n if v < 0 else " " * width
    right = "█" * n + " " * (width - n) if v >= 0 else " " * width
    return f"[{left}|{right}]"


# ---------------------------------------------------------------------------
# 실험실 화면
# ---------------------------------------------------------------------------


class LabView(QWidget):
    run_started = Signal(Path)

    def __init__(self, parent: QWidget | None = None, settings: QSettings | None = None) -> None:
        super().__init__(parent)
        self.settings = settings if settings is not None else settings_store()
        self.client = TrainingClient(self)
        self.client.started.connect(self._on_started)
        self.client.generation.connect(self._on_generation)
        self.client.state_changed.connect(self._on_state)
        self.client.applied.connect(self._on_applied)
        self.client.rewound.connect(self._on_rewound)
        self.client.failed.connect(self._on_failed)
        self.client.closed.connect(self._on_closed)

        # 지금 보고 있는 실행
        self.env_path: Path | None = None
        self.env_text: str | None = None
        self.spec: EnvSpec | None = None
        self.layout: GenomeLayout | None = None
        self.run_dir: Path | None = None
        self.run_info: dict[str, Any] = {}
        self.rows: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.champion: dict[str, Any] | None = None
        self.state = "idle"
        self.generation = 0
        self._spec_cache: dict[str, EnvSpec] = {}

        # 재생 상태
        self.view_mode = "live"  # live(관전), fast(고속 챔피언), replay(타임라인·겹쳐 보기)
        self.queue: collections.deque[dict[str, Any]] = collections.deque()
        self.playing: dict[str, Any] | None = None
        self.play_t = 0.0
        self.speed = 1
        self.loop_overlay = False
        self._threads: list[_ReplayThread] = []
        self._replay_tag = 0
        self._fast_last = 0
        self._inspect_index: int | None = None
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        self._restore_settings()
        self._update_buttons()

    # ------------------------------------------------------------------
    # 화면 만들기
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.canvas = SpectatorCanvas()
        self.canvas.individual_clicked.connect(self._inspect)
        self.dashboard = Dashboard(settings=self.settings)
        self.inspect = InspectPanel()
        self.inspect.hide()
        self.inspect.closed.connect(self._close_inspect)

        # 타임라인
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setMinimum(1)
        self.timeline.setMaximum(1)
        self.timeline_label = QLabel(S.LAB_TIMELINE)
        self.timeline.valueChanged.connect(lambda v: self.timeline_label.setText(f"{S.LAB_TIMELINE} {v}"))
        self.btn_timeline = QPushButton(S.LAB_TIMELINE_PLAY)
        self.btn_timeline.clicked.connect(lambda: self.play_generation(self.timeline.value()))
        self.btn_versus = QPushButton(S.LAB_VERSUS)
        self.btn_versus.clicked.connect(self.play_versus)
        self.btn_live = QPushButton(S.LAB_LIVE)
        self.btn_live.clicked.connect(self.back_to_live)
        tl = QHBoxLayout()
        tl.addWidget(self.timeline_label)
        tl.addWidget(self.timeline, 1)
        tl.addWidget(self.btn_timeline)
        tl.addWidget(self.btn_versus)
        tl.addWidget(self.btn_live)

        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(self.canvas, 1)
        row.addWidget(self.inspect)
        cl.addLayout(row, 1)
        cl.addLayout(tl)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(center)
        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setWidget(self._build_panel())
        panel_scroll.setMinimumWidth(300)
        top.addWidget(panel_scroll)
        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 0)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self.dashboard)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(split)

    def _build_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        env = QGroupBox(S.LAB_ENV_GROUP)
        el = QVBoxLayout(env)
        self.env_label = QLabel(S.LAB_ENV_NONE)
        self.env_label.setWordWrap(True)
        self.env_combo = QComboBox()
        self.env_combo.addItem(S.LAB_ENV_BUILTIN, None)
        for p in sorted(paths.builtin_env_dir().glob("*.evoenv")):
            self.env_combo.addItem(self._env_title(p), str(p))
        self.env_combo.activated.connect(self._env_combo_chosen)
        btn_open = QPushButton(S.LAB_ENV_OPEN)
        btn_open.clicked.connect(self._choose_env)
        self.btn_reload = QPushButton(S.LAB_ENV_RELOAD)
        self.btn_reload.setToolTip(S.LAB_ENV_RELOAD_TIP)
        self.btn_reload.clicked.connect(self.reload_env)
        el.addWidget(self.env_label)
        el.addWidget(self.env_combo)
        el.addWidget(btn_open)
        el.addWidget(self.btn_reload)
        lay.addWidget(env)

        st = QGroupBox(S.LAB_SETTINGS_GROUP)
        form = QFormLayout(st)
        self.method = QComboBox()
        self.method.addItem(S.METHOD_NAMES["cmaes"], "cmaes")
        for key in ("ga", "sa"):
            self.method.addItem(S.LAB_METHOD_LATER.format(name=S.METHOD_NAMES[key]), key)
            item = self.method.model().item(self.method.count() - 1)  # type: ignore[attr-defined]
            item.setEnabled(False)
        self.seed = _spin(0, 2**31 - 1, 1)
        self.popsize = _spin(2, 4096, 32)
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.001, 10.0)
        self.sigma.setDecimals(3)
        self.sigma.setSingleStep(0.05)
        self.sigma.setValue(0.5)
        self.trials = _spin(0, 1000, 0, special=S.LAB_TRIALS_ENV)
        self.workers = _spin(1, 256, default_workers())
        form.addRow(S.LAB_METHOD, self.method)
        form.addRow(S.LAB_SEED, self.seed)
        form.addRow(S.LAB_POPSIZE, self.popsize)
        form.addRow(S.LAB_SIGMA, self.sigma)
        form.addRow(S.LAB_TRIALS, self.trials)
        form.addRow(S.LAB_WORKERS, self.workers)
        lay.addWidget(st)

        stop = QGroupBox(S.LAB_STOP_GROUP)
        sf = QFormLayout(stop)
        self.max_generations = _spin(0, 10**7, 0, special=S.LAB_NONE)
        self.max_minutes = QDoubleSpinBox()
        self.max_minutes.setRange(0.0, 10**6)
        self.max_minutes.setDecimals(1)
        self.max_minutes.setSpecialValueText(S.LAB_NONE)
        self.target = QDoubleSpinBox()
        self.target.setRange(-1e9, 1e9)
        self.target.setDecimals(2)
        self.target.setMinimum(-1e9)
        self.target.setSpecialValueText(S.LAB_NONE)
        self.target.setValue(-1e9)
        self.stagnation = _spin(0, 10**7, 0, special=S.LAB_NONE)
        sf.addRow(S.LAB_MAX_GENERATIONS, self.max_generations)
        sf.addRow(S.LAB_MAX_MINUTES, self.max_minutes)
        sf.addRow(S.LAB_TARGET, self.target)
        sf.addRow(S.LAB_STAGNATION, self.stagnation)
        self.btn_apply = QPushButton(S.LAB_APPLY)
        self.btn_apply.setToolTip(S.LAB_APPLY_TIP)
        self.btn_apply.clicked.connect(self.apply_settings)
        sf.addRow(self.btn_apply)
        lay.addWidget(stop)

        ctl = QGroupBox(S.LAB_CONTROL_GROUP)
        g = QGridLayout(ctl)
        self.btn_start = QPushButton(S.LAB_START)
        self.btn_start.clicked.connect(self.start_new)
        self.btn_pause = QPushButton(S.LAB_PAUSE)
        self.btn_pause.clicked.connect(lambda: self.client.send({"type": "pause"}))
        self.btn_continue = QPushButton(S.LAB_CONTINUE)
        self.btn_continue.clicked.connect(self.continue_run)
        self.btn_step = QPushButton(S.LAB_STEP)
        self.btn_step.clicked.connect(lambda: self.client.send({"type": "step"}))
        self.btn_end = QPushButton(S.LAB_END)
        self.btn_end.clicked.connect(self.end_session)
        g.addWidget(self.btn_start, 0, 0)
        g.addWidget(self.btn_pause, 0, 1)
        g.addWidget(self.btn_continue, 1, 0)
        g.addWidget(self.btn_step, 1, 1)
        g.addWidget(self.btn_end, 2, 0, 1, 2)
        self.status = QLabel(S.LAB_STATUS_IDLE)
        self.status.setWordWrap(True)
        g.addWidget(self.status, 3, 0, 1, 2)
        lay.addWidget(ctl)

        view = QGroupBox(S.LAB_VIEW_GROUP)
        vf = QFormLayout(view)
        self.rb_spectate = QRadioButton(S.LAB_SPECTATE)
        self.rb_fast = QRadioButton(S.LAB_FAST)
        self.rb_spectate.setChecked(True)
        grp = QButtonGroup(view)
        grp.addButton(self.rb_spectate)
        grp.addButton(self.rb_fast)
        self.rb_spectate.toggled.connect(lambda on: self.set_spectate(on))
        mode = QHBoxLayout()
        mode.addWidget(self.rb_spectate)
        mode.addWidget(self.rb_fast)
        vf.addRow(mode)
        self.speed_combo = QComboBox()
        for n in SPEEDS:
            self.speed_combo.addItem(S.LAB_SPEED_ITEM.format(n=n), n)
        self.speed_combo.currentIndexChanged.connect(lambda i: self.set_speed(SPEEDS[i]))
        vf.addRow(S.LAB_SPEED, self.speed_combo)
        self.fast_interval = _spin(1, 100000, 10)
        vf.addRow(S.LAB_FAST_INTERVAL, self.fast_interval)
        lay.addWidget(view)
        lay.addStretch(1)
        return w

    @staticmethod
    def _env_title(path: Path) -> str:
        try:
            return str(json.loads(path.read_text(encoding="utf-8-sig")).get("name") or path.stem)
        except (OSError, ValueError):
            return path.stem

    def _restore_settings(self) -> None:
        s = self.settings
        try:
            speed = int(s.value("lab/speed", 1))
            if speed in SPEEDS:
                self.speed_combo.setCurrentIndex(SPEEDS.index(speed))
            if str(s.value("lab/mode", "spectate")) == "fast":
                self.rb_fast.setChecked(True)
            self.fast_interval.setValue(int(s.value("lab/fast_interval", 10)))
        except (TypeError, ValueError):
            pass

    def _save_settings(self) -> None:
        self.settings.setValue("lab/speed", self.speed)
        self.settings.setValue("lab/mode", "spectate" if self.rb_spectate.isChecked() else "fast")
        self.settings.setValue("lab/fast_interval", self.fast_interval.value())

    # ------------------------------------------------------------------
    # 환경과 설정
    # ------------------------------------------------------------------
    def _env_combo_chosen(self, index: int) -> None:
        path = self.env_combo.itemData(index)
        if path:
            self.load_env(Path(path))

    def _choose_env(self) -> None:
        start = str(self.env_path.parent if self.env_path else paths.builtin_env_dir())
        path, _ = QFileDialog.getOpenFileName(self, S.LAB_ENV_OPEN, start, S.ENV_FILE_FILTER)
        if path:
            self.load_env(Path(path))

    def load_env(self, path: Path) -> bool:
        """새 실행에 쓸 환경 파일을 고른다(학습 중이 아닐 때)."""
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            QMessageBox.warning(self, S.LAB_OPEN_ENV_FAILED, str(exc))
            return False
        return self.use_env_text(text, str(path), None, path)

    def use_env_text(self, text: str, where: str, config: dict[str, Any] | None,
                     path: Path | None = None) -> bool:
        """환경 내용으로 새 실행을 준비한다. config를 주면 설정 칸도 채운다(복제해서 새 실행)."""
        if self.client.active:
            return False
        try:
            spec = compile_env(text)
            layout = layout_for(spec)
            Episode(spec, None if not layout.n_body else layout.body_values(layout.initial()[None, :])[0],
                    seeds.display_trial())
        except (EnvFileError, EpisodeBuildError, ValueError) as exc:
            QMessageBox.warning(self, S.LAB_OPEN_ENV_FAILED, str(exc))
            return False
        self._reset_run_view()
        self.run_dir = None
        self.run_info = {}
        self.state = "idle"
        self.env_path, self.env_text, self.spec, self.layout = path, text, spec, layout
        self.env_label.setText(f"{spec.file.name}\n{where}")
        if config:
            self._form_from_config(config)
        if spec.file.evaluation.trials:
            self.trials.setSpecialValueText(f"{S.LAB_TRIALS_ENV} ({spec.file.evaluation.trials})")
        self._show_still_scene()
        self._update_buttons()
        return True

    def _show_still_scene(self) -> None:
        if self.spec is None or self.layout is None:
            return
        rep = replay(self.spec, self.layout, self.layout.initial(), seeds.display_trial())
        self.canvas.set_scene(rep.geometry, None, channels=rep.channels)
        self.canvas.set_population([])
        self.canvas.set_overlay([])
        self.canvas.set_time(0.0)

    def config_from_form(self) -> TrainConfig:
        c = TrainConfig(
            method=str(self.method.currentData()),
            run_seed=self.seed.value(),
            popsize=self.popsize.value(),
            sigma0=self.sigma.value(),
            trials=self.trials.value() or None,
        )
        self._stop_fields(c)
        return c

    def _stop_fields(self, c: TrainConfig) -> None:
        c.max_generations = self.max_generations.value() or None
        c.max_time = self.max_minutes.value() * 60.0 or None
        c.target_score = None if self.target.value() <= self.target.minimum() else self.target.value()
        c.stagnation_generations = self.stagnation.value() or None

    def _form_from_config(self, c: dict[str, Any]) -> None:
        def setv(widget: Any, value: Any, empty: Any) -> None:
            widget.setValue(empty if value is None else value)

        idx = self.method.findData(c.get("method", "cmaes"))
        if idx >= 0:
            self.method.setCurrentIndex(idx)
        setv(self.seed, c.get("run_seed"), 1)
        setv(self.popsize, c.get("popsize"), 32)
        setv(self.sigma, c.get("sigma0"), 0.5)
        setv(self.trials, c.get("trials"), 0)
        setv(self.max_generations, c.get("max_generations"), 0)
        mt = c.get("max_time")
        self.max_minutes.setValue(0.0 if mt is None else float(mt) / 60.0)
        setv(self.target, c.get("target_score"), self.target.minimum())
        setv(self.stagnation, c.get("stagnation_generations"), 0)

    # ------------------------------------------------------------------
    # 학습 시작·조작
    # ------------------------------------------------------------------
    def start_new(self) -> bool:
        if self.client.active or self.env_text is None:
            return False
        config = self.config_from_form()
        try:
            config.validate()
        except ValueError as exc:
            QMessageBox.warning(self, S.LAB_CHANGE_FAILED, str(exc))
            return False
        self._reset_run_view()
        self.state = "starting"
        req = SessionRequest(kind="new", env_text=self.env_text, config=config.to_dict(), runs_dir=str(paths.runs_dir()),
                             workers=self.workers.value(), spectate=self.rb_spectate.isChecked())
        self.client.start(req)
        self._update_buttons()
        return True

    def resume_run(self, run_dir: Path, generation: int | None = None) -> bool:
        if self.client.active:
            return False
        self._reset_run_view()
        self.state = "starting"
        req = SessionRequest(kind="resume", run_dir=str(run_dir), generation=generation, workers=self.workers.value(),
                             spectate=self.rb_spectate.isChecked())
        self.client.start(req)
        self._update_buttons()
        return True

    def continue_run(self) -> None:
        if self.client.active:
            self.client.send({"type": "run"})
        elif self.run_dir is not None:
            self.resume_run(self.run_dir)

    def end_session(self) -> None:
        if self.client.active:
            self.setCursor(Qt.CursorShape.WaitCursor)
            self.status.setText(S.LAB_CLOSE_WAIT)
            try:
                self.client.shutdown()
            finally:
                self.unsetCursor()

    def apply_settings(self) -> None:
        if not self.client.active:
            return
        c = TrainConfig()
        self._stop_fields(c)
        changes = {
            "max_generations": c.max_generations, "max_time": c.max_time, "target_score": c.target_score,
            "stagnation_generations": c.stagnation_generations, "trials": self.trials.value() or None,
        }
        self.client.send({"type": "config", "changes": changes})

    def reload_env(self) -> None:
        """멈춘 학습에 환경 파일을 다시 읽어 넣는다(에디터는 M5a; 지금은 다른 편집기로 고친 파일)."""
        if not self.client.active or self.env_path is None:
            return
        try:
            text = self.env_path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            QMessageBox.warning(self, S.LAB_OPEN_ENV_FAILED, str(exc))
            return
        self.client.send({"type": "env", "text": text})

    def set_spectate(self, on: bool) -> None:
        self.client.send({"type": "mode", "spectate": bool(on)})
        self.queue.clear()
        if self.view_mode in ("live", "fast"):
            self.view_mode = "live" if on else "fast"
            self.playing = None
            if not on:
                self._fast_last = 0
                self._maybe_fast_replay(force=True)
        self._save_settings()

    def set_speed(self, n: int) -> None:
        self.speed = n
        self._save_settings()

    # ------------------------------------------------------------------
    # 학습 프로세스에서 온 것
    # ------------------------------------------------------------------
    def _reset_run_view(self) -> None:
        self.rows = []
        self.events = []
        self.champion = None
        self.queue.clear()
        self.playing = None
        self.generation = 0
        self.view_mode = "live" if self.rb_spectate.isChecked() else "fast"
        self.dashboard.clear()
        self.timeline.setMaximum(1)
        self._close_inspect()

    def _adopt_run(self, msg: dict[str, Any]) -> None:
        self.run_dir = Path(msg["run_dir"])
        self.run_info = dict(msg.get("run_info") or {})
        self.env_text = str(msg["env_text"])
        self.spec = compile_env(self.env_text)
        self.layout = GenomeLayout.from_dict(msg["layout"]) if msg.get("layout") else layout_for(self.spec)
        self.generation = int(msg.get("generation", 0))
        self.champion = msg.get("champion")
        self.rows = list(msg.get("stats") or [])
        self.events = list(msg.get("events") or [])
        self._spec_cache.clear()
        self.env_label.setText(f"{self.spec.file.name}\n{self.run_dir}")
        self._form_from_config(msg.get("config") or {})
        self._setup_dashboard()
        self.dashboard.set_rows(self.rows)
        self.timeline.setMaximum(max(1, self.generation))
        self.timeline.setValue(max(1, self.generation))
        self._show_still_scene()

    def _setup_dashboard(self) -> None:
        if self.spec is None:
            return
        labels = {t.event: t.label for t in self.spec.terminations}
        ends = [(t.event, labels.get(t.event) or t.event) for t in self.spec.terminations]
        blocks = [(b.id, b.label or b.id) for b in self.spec.file.scoring.blocks]
        stag = self.config_value("stagnation_generations")
        self.dashboard.set_meta(end_events=ends, blocks=blocks, stagnation_generations=stag)
        self._update_marks()
        self.dashboard.set_my_best(my_best_score(self.spec))

    def config_value(self, key: str) -> Any:
        return (self.run_info.get("config") or {}).get(key)

    def _update_marks(self) -> None:
        marks = [(int(e.get("generation", 0)), str(e.get("kind"))) for e in self.events if e.get("kind") in MARK_KINDS]
        self.dashboard.set_marks(marks)

    def _on_started(self, msg: dict[str, Any]) -> None:
        self._adopt_run(msg)
        self.status.setText(S.LAB_STARTED_RUN.format(path=self.run_dir))
        if self.run_dir is not None:
            self.run_started.emit(self.run_dir)
        if not msg.get("spectate"):
            self._maybe_fast_replay(force=True)

    def _on_rewound(self, msg: dict[str, Any]) -> None:
        self._reset_run_view()
        self._adopt_run(msg)

    def _on_generation(self, msg: dict[str, Any]) -> None:
        row = msg["row"]
        self.generation = int(msg["generation"])
        self.rows.append(row)
        self.dashboard.add_row(row)
        if msg.get("champion") is not None:
            self.champion = msg["champion"]
        at_end = self.timeline.value() >= self.timeline.maximum()
        self.timeline.setMaximum(max(1, self.generation))
        if at_end:
            self.timeline.setValue(self.generation)
        self._update_status()
        if msg.get("trajectories") is not None:
            if self.view_mode == "live":
                self.queue.append(msg)
                if self.playing is None:
                    self._play_next()
            else:
                self.client.send({"type": "played", "generation": self.generation})
        elif self.view_mode == "fast":
            self.last_fast_candidate = msg
            self._maybe_fast_replay()

    def _on_state(self, msg: dict[str, Any]) -> None:
        self.state = str(msg.get("state"))
        self._update_status()
        if self.state == "finished":
            reason = S.TRAIN_STOP_REASONS.get(str(msg.get("reason")), str(msg.get("reason")))
            self.status.setText(self.status.text() + "\n" + S.LAB_STOPPED_BECAUSE.format(reason=reason))
        self._update_buttons()

    def _on_applied(self, msg: dict[str, Any]) -> None:
        if not msg.get("ok"):
            QMessageBox.warning(self, S.LAB_CHANGE_FAILED, str(msg.get("message", "")))
            return
        gen = int(msg.get("generation", self.generation))
        kind = {"config": "config_change", "env": "env_change"}.get(str(msg.get("what")))
        if kind and (msg.get("changes") or msg.get("what") == "env"):
            self.events.append({"kind": kind, "generation": gen})
            self._update_marks()
        if msg.get("what") == "config" and msg.get("config"):
            self.run_info.setdefault("config", {}).update(msg["config"])
        if msg.get("what") == "env" and msg.get("env_text"):
            self.env_text = str(msg["env_text"])
            self.spec = compile_env(self.env_text)
            self._spec_cache.clear()
        self.status.setText(self.status.text() + "\n" + S.LAB_CHANGE_DONE.format(gen=gen))

    def _on_failed(self, message: str, fatal: bool) -> None:
        QMessageBox.warning(self, S.LAB_ERROR_TITLE, message)
        if fatal:
            self.state = "closed"
            self._update_buttons()

    def _on_closed(self) -> None:
        self.state = "closed" if self.run_dir is not None else "idle"
        self.queue.clear()
        self._update_status()
        self._update_buttons()

    def _update_status(self) -> None:
        if not self.rows:
            self.status.setText(S.LAB_STATE_NAMES.get(self.state, self.state))
            return
        last = self.rows[-1]
        champ = last.get("champion_score", "")
        try:
            champ_text = f"{float(champ):.2f}" if champ not in ("", None) else "-"
            evals = int(float(last.get("evaluations", 0)))
        except (TypeError, ValueError):
            champ_text, evals = "-", 0
        self.status.setText(S.LAB_STATUS.format(state=S.LAB_STATE_NAMES.get(self.state, self.state),
                                                gen=self.generation, evals=evals, champ=champ_text))

    def _update_buttons(self) -> None:
        active = self.client.active
        running = active and self.state == "running"
        paused = active and self.state in ("paused", "finished")
        self.btn_start.setEnabled(not active and self.env_text is not None)
        self.btn_pause.setEnabled(running)
        self.btn_continue.setEnabled(paused or (not active and self.run_dir is not None))
        self.btn_step.setEnabled(paused)
        self.btn_end.setEnabled(active)
        self.btn_apply.setEnabled(paused)
        self.btn_reload.setEnabled(paused and self.env_path is not None)
        for w in (self.method, self.seed, self.popsize, self.sigma, self.workers, self.env_combo):
            w.setEnabled(not active)
        has_gen = self.generation > 0 and self.run_dir is not None
        self.btn_timeline.setEnabled(has_gen)
        self.btn_versus.setEnabled(has_gen)

    # ------------------------------------------------------------------
    # 관전 재생
    # ------------------------------------------------------------------
    def _play_next(self) -> None:
        if not self.queue or self.spec is None or self.layout is None:
            self.playing = None
            return
        msg = self.queue.popleft()
        self.playing = msg
        seed, track = msg["trial"]
        trial = Trial(int(seed), track)
        spec = self.spec_for(msg.get("env_version"), int(msg["generation"]))
        geometry, agents = self._scene_for(spec, msg["population"], trial)
        scores = np.asarray(msg["scores"])
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[np.argsort(-scores, kind="stable")] = np.arange(len(scores))
        self.canvas.set_scene(geometry, agents, channels=list(spec.actuator_channels))
        self.canvas.set_population(list(msg["trajectories"]), ranks=ranks, champion=int(msg["representative"]))
        self.canvas.set_overlay([])
        self._start_clock()

    def _scene_for(self, spec: EnvSpec, population: np.ndarray, trial: Trial) -> tuple[Any, list[Any] | None]:
        layout = self.layout
        assert layout is not None
        if layout.n_body:
            body = layout.body_values(np.asarray(population, dtype=np.float64))
            eps = [Episode(spec, body[i], trial) for i in range(len(population))]
            geoms = [e.geometry() for e in eps]
            return geoms[0], [agent_geoms(g, e.moving_bodies) for g, e in zip(geoms, eps, strict=True)]
        ep = Episode(spec, None, trial)
        return ep.geometry(), None

    def _start_clock(self) -> None:
        self.play_t = 0.0
        self._clock.restart()
        self.canvas.set_time(0.0)
        if not self._timer.isActive():
            self._timer.start()

    def _tick(self) -> None:
        dt_real = self._clock.restart() / 1000.0
        self.play_t += min(dt_real, 0.25) * (self.speed if self.view_mode == "live" else 1.0)
        self.canvas.set_time(self.play_t)
        if self.view_mode == "live" and self.playing is not None:
            self.canvas.set_hud(S.LAB_HUD_LIVE.format(gen=self.playing["generation"], t=self.play_t, speed=self.speed))
        elif self.view_mode == "fast":
            self.canvas.set_hud(S.LAB_HUD_FAST.format(gen=self._fast_last, t=self.play_t))
        elif self.view_mode == "replay":
            self.canvas.set_hud(S.LAB_HUD_REPLAY.format(label=self._replay_label, t=self.play_t))
        if self._inspect_index is not None:
            self.inspect.show_time(self.play_t)
        if self.play_t < self.canvas.duration() + 0.5:
            return
        if self.view_mode == "live" and self.playing is not None:
            self.client.send({"type": "played", "generation": int(self.playing["generation"])})
            self.playing = None
            if self.queue:
                self._play_next()
            else:
                self._timer.stop()
        elif self.loop_overlay:
            self.play_t = 0.0
        else:
            self._timer.stop()

    # ------------------------------------------------------------------
    # 다시 시뮬레이션: 고속 모드 챔피언, 타임라인, 1세대 vs 지금, 클릭 검사
    # ------------------------------------------------------------------
    def spec_for(self, version_hash: str | None, generation: int) -> EnvSpec:
        """그 세대를 계산할 때 쓴 환경. 고친 환경이면 그때 버전으로 재생한다."""
        assert self.spec is not None
        if version_hash is not None and self.spec is not None and env_hash(self.spec.file)[:16] == version_hash:
            return self.spec
        text = self.env_text_for_generation(generation)
        if text is None:
            return self.spec
        key = str(hash(text))
        if key not in self._spec_cache:
            self._spec_cache[key] = compile_env(text)
        return self._spec_cache[key]

    def env_text_for_generation(self, generation: int) -> str | None:
        if self.run_dir is None:
            return None
        from evolab.runs.store import RunStore, RunStoreError

        try:
            store = RunStore.open(self.run_dir)
            versions = list(store.read_run_info().get("env_versions", []))
            usable = [v for v in versions if int(v.get("from_generation", 0)) <= generation - 1] or versions[:1]
            return store.env_text(str(usable[-1]["id"])) if usable else None
        except (RunStoreError, OSError, KeyError, ValueError):
            return None

    def _run_replays(self, tag: Any, jobs: list[tuple[EnvSpec, GenomeLayout, np.ndarray, Trial, bool]]) -> None:
        self._replay_tag += 1
        th = _ReplayThread((self._replay_tag, tag), jobs, self)
        th.finished_job.connect(self._replay_done)
        th.finished.connect(lambda t=th: self._forget_thread(t))
        self._threads.append(th)
        th.start()

    def _forget_thread(self, th: _ReplayThread) -> None:
        self._threads = [t for t in self._threads if t is not th]
        th.deleteLater()

    def _replay_done(self, tag: Any, result: Any) -> None:
        seq, (kind, info) = tag
        if seq != self._replay_tag:
            return  # 더 새로운 요청이 있다
        if isinstance(result, Exception):
            QMessageBox.warning(self, S.LAB_ERROR_TITLE, str(result))
            return
        if kind == "inspect":
            index = info
            self._inspect_index = index
            self.inspect.set_replay(index, result[0])
            self.inspect.show()
            rep = result[0]
            self.canvas.set_overlay([(rep.trajectory, rep.agent, COLOR_NOW, str(index))])
            if self.view_mode != "live":
                self._start_clock()
            return
        reps: list[Replay] = result
        self.canvas.set_scene(reps[0].geometry, None, channels=reps[0].channels)
        self.canvas.set_population([])
        colors = [COLOR_OLD, COLOR_NOW] if len(reps) > 1 else [COLOR_NOW]
        self.canvas.set_overlay([(r.trajectory, r.agent, c, label)
                                 for r, c, label in zip(reps, colors, info, strict=True)])
        self.loop_overlay = True
        self._start_clock()

    def _maybe_fast_replay(self, force: bool = False) -> None:
        if self.view_mode != "fast" or self.spec is None or self.layout is None:
            return
        if not force and self.generation - self._fast_last < self.fast_interval.value():
            return
        genome, gen = self._current_champion()
        if genome is None:
            return
        self._fast_last = max(self.generation, 1)
        self._replay_label = S.LAB_LABEL_CHAMPION.format(gen=gen)
        self._run_replays(("fast", [self._replay_label]),
                          [(self.spec, self.layout, genome, seeds.display_trial(), False)])

    def _current_champion(self) -> tuple[np.ndarray | None, int]:
        if self.champion is not None and self.champion.get("genome") is not None:
            return np.asarray(self.champion["genome"], dtype=np.float64), int(self.champion.get("generation", 0))
        msg = getattr(self, "last_fast_candidate", None)
        if msg is not None:
            return np.asarray(msg["population"][int(msg["representative"])]), int(msg["generation"])
        return None, 0

    def _load_representative(self, generation: int) -> np.ndarray | None:
        if self.run_dir is None:
            return None
        from evolab.runs.store import RunStore, RunStoreError

        try:
            data = RunStore.open(self.run_dir).load_champion(generation, "representative")
        except (RunStoreError, OSError, KeyError, ValueError):
            return None
        return np.asarray(data["genome"], dtype=np.float64)

    def play_generation(self, generation: int) -> bool:
        """타임라인: 그 세대 대표를 전시 시드로 다시 시뮬레이션해 재생한다."""
        genome = self._load_representative(generation)
        if genome is None or self.layout is None:
            return False
        self.view_mode = "replay"
        self._ack_pending()
        self._replay_label = S.LAB_LABEL_REP.format(gen=generation)
        spec = self.spec_for(None, generation)
        self._run_replays(("timeline", [self._replay_label]), [(spec, self.layout, genome, seeds.display_trial(), False)])
        return True

    def play_versus(self) -> bool:
        """1세대 대표와 지금 챔피언을 전시 시드로 겹쳐 본다."""
        first = self._load_representative(1)
        now, gen = self._current_champion()
        if now is None:
            now = self._load_representative(self.generation)
            gen = self.generation
        if first is None or now is None or self.layout is None:
            return False
        self.view_mode = "replay"
        self._ack_pending()
        labels = [S.LAB_LABEL_REP.format(gen=1), S.LAB_LABEL_CHAMPION.format(gen=gen)]
        self._replay_label = " vs ".join(labels)
        trial = seeds.display_trial()
        self._run_replays(("versus", labels), [(self.spec_for(None, 1), self.layout, first, trial, False),
                                                (self.spec_for(None, max(1, gen)), self.layout, now, trial, False)])
        return True

    def _ack_pending(self) -> None:
        """관전 재생을 그만 볼 때 기다리는 학습이 멈추지 않게 받은 세대를 다 봤다고 알린다."""
        self.loop_overlay = False
        if self.playing is not None:
            self.client.send({"type": "played", "generation": int(self.playing["generation"])})
            self.playing = None
        while self.queue:
            msg = self.queue.popleft()
            self.client.send({"type": "played", "generation": int(msg["generation"])})

    def back_to_live(self) -> None:
        self.loop_overlay = False
        self.canvas.set_overlay([])
        self._close_inspect()
        self.view_mode = "live" if self.rb_spectate.isChecked() else "fast"
        if self.view_mode == "fast":
            self._maybe_fast_replay(force=True)
        else:
            self._timer.stop()

    def _inspect(self, index: int) -> None:
        msg = self.playing
        if msg is None or self.layout is None:
            return
        seed, track = msg["trial"]
        spec = self.spec_for(msg.get("env_version"), int(msg["generation"]))
        genome = np.asarray(msg["population"][index], dtype=np.float64)
        self._run_replays(("inspect", index), [(spec, self.layout, genome, Trial(int(seed), track), True)])

    def _close_inspect(self) -> None:
        self._inspect_index = None
        self.inspect.hide()
        if self.view_mode == "live":
            self.canvas.set_overlay([])

    # ------------------------------------------------------------------
    # 기록 화면에서 연 실행 (학습 없이 보기)
    # ------------------------------------------------------------------
    def open_run(self, run_dir: Path) -> bool:
        if self.client.active:
            return False
        from evolab.runs.store import RunStore, RunStoreError

        try:
            store = RunStore.open(run_dir)
            info = store.read_run_info()
            versions = info.get("env_versions", [])
            text = store.env_text(str(versions[-1]["id"]))
            champion = None
            gens = store.champion_generations("champion")
            if gens:
                data = store.load_champion(gens[-1], "champion")
                champion = {"genome": data["genome"], "score": float(data.get("score", np.nan)), "generation": gens[-1]}
            stats = store.read_stats()
            msg = {"run_dir": str(run_dir), "run_info": info, "env_text": text, "generation": len(stats),
                   "config": info.get("config", {}), "layout": info.get("layout"), "stats": stats,
                   "events": store.read_events(), "champion": champion}
        except (RunStoreError, OSError, KeyError, IndexError, ValueError, EnvFileError) as exc:
            QMessageBox.warning(self, S.LAB_ERROR_TITLE, str(exc))
            return False
        self._reset_run_view()
        self._adopt_run(msg)
        self.state = "closed"
        self._update_status()
        self._update_buttons()
        return True

    def shutdown(self) -> None:
        self._timer.stop()
        self.client.shutdown()
        for th in list(self._threads):
            th.wait(10000)


def _spin(lo: int, hi: int, value: int, special: str | None = None) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    if special is not None:
        s.setSpecialValueText(special)
    return s


def my_best_score(spec: EnvSpec) -> float | None:
    """직접 조종 기록 가운데 같은 환경(내용 해시)의 최고 점수."""
    try:
        from evolab.app.play.window import records_dir

        folder = records_dir()
        want = env_hash(spec.file)
        best: float | None = None
        for p in folder.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if data.get("env_hash") != want:
                continue
            score = (data.get("result") or {}).get("score")
            if isinstance(score, (int, float)) and math.isfinite(score):
                best = score if best is None else max(best, float(score))
        return best
    except OSError:
        return None

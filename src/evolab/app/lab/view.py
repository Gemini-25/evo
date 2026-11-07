"""실험실 화면 (명세: 화면과 조작 > 실험실).

가운데 관전 화면, 오른쪽 제어 패널, 아래 통계 대시보드. 학습은 학습 프로세스(TrainingClient)가 하고, 이 화면은
결과를 받아 그리기만 한다. 다시 시뮬레이션(타임라인, 1세대 vs 지금, 고속 모드 챔피언, 개체 클릭 검사)은
백그라운드 스레드에서 같은 런타임 코드로 한다.
"""

from __future__ import annotations

import collections
import dataclasses
import gc
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QElapsedTimer, QObject, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
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
SHUTDOWN_WAIT_S = 120.0
"""창을 닫을 때 이번 세대를 마치고 체크포인트를 남길 때까지 기다리는 시간(큰 개체 수는 세대가 길다)."""


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


class SceneCancelled(Exception):
    """장면 만들기를 중간에 그만두었다(그 세대를 더 볼 일이 없거나 창을 닫는 중)."""


def build_scene(spec: EnvSpec, layout: GenomeLayout, population: np.ndarray, trial: Trial,
                cancel: Callable[[], bool] | None = None) -> tuple[Any, list[Any] | None]:
    """관전 장면: 세계 모양과 (몸 유전자가 있으면) 개체마다 몸 모양.

    몸 유전자가 있으면 개체마다 Episode를 만들어야 해서 개체 수에 비례해 느리다(언덕 자동차 32개에 0.3초 안팎).
    그래서 관전 재생은 이것을 _SceneThread에서 부른다(화면 스레드를 막지 않게).
    Episode는 하나씩 만들어 모양만 남기고 바로 버린다. N개를 다 살려 두면(Episode마다 순환 참조가 있는 객체 5천 개
    안팎) 순환 GC의 전체 수집이 그것을 거듭 훑느라 GIL을 오래 쥐어, 개체 256개에서 화면 스레드가 200ms씩 멈췄다.
    버린 Episode는 순환 참조라 참조 수로는 풀리지 않으므로 바로 젊은 세대만 수집한다(1ms 안팎). 그러면 만드는 중인
    Episode가 늙은 세대로 넘어가지 않아 전체 수집이 일어나지 않는다(개체 256개: 화면 멈춤 최대 50ms → 10ms 아래).
    세계 모양은 첫 개체 것이다. cancel()이 참이 되면 다음 개체 전에 SceneCancelled로 그만둔다.
    """
    if not layout.n_body:
        return Episode(spec, None, trial).geometry(), None
    body = layout.body_values(np.asarray(population, dtype=np.float64))
    world: Any = None
    agents: list[Any] = []
    for i in range(len(population)):
        if cancel is not None and cancel():
            raise SceneCancelled
        ep = Episode(spec, body[i], trial)
        geometry = ep.geometry()
        if world is None:
            world = geometry
        agents.append(agent_geoms(geometry, ep.moving_bodies))
        del ep, geometry  # 다음 개체를 만들기 전에 놓는다(위 설명)
        gc.collect(1)
    return world, agents


class _SceneThread(QThread):
    """관전할 세대의 장면(build_scene)을 백그라운드에서 만든다. 화면에만 쓰므로 학습 결과와 무관하다.

    requestInterruption()을 부르면 개체 하나를 만드는 사이에 멈춘다(SceneCancelled를 결과로 보낸다).
    """

    finished_job = Signal(object, object)
    """(꼬리표, (세계 모양, 개체 몸 모양) 또는 예외)"""

    def __init__(self, tag: Any, spec: EnvSpec, layout: GenomeLayout, population: np.ndarray, trial: Trial,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.tag = tag
        self.args = (spec, layout, population, trial)

    def run(self) -> None:
        try:
            out = build_scene(*self.args, cancel=self.isInterruptionRequested)
        except Exception as exc:  # 결과를 화면 스레드로 넘겨 알린다(멈춘 것도: 꼬리표가 달라 버려진다)
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
    edit_env_requested = Signal(str, str)
    """(환경 내용, 이름) — 에디터에서 고치기. 멈춘 실험이면 고친 뒤 send_env_text로 이어서 학습한다."""

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
        self._threads: list[QThread] = []
        self._replay_tag = 0
        self._replay_label = ""
        self._fast_last = 0
        self.last_fast_candidate: dict[str, Any] | None = None
        self._last_video: tuple[list[Any], str] | None = None
        self._last_played: dict[str, Any] | None = None
        """관전 재생이 끝나 화면에 남은 세대(재생이 끝난 뒤에도 개체를 클릭해 들여다볼 수 있게)."""
        self._scene_jobs: collections.deque[dict[str, Any]] = collections.deque()
        """장면(몸 유전자 환경)을 백그라운드에서 만드는 중이거나 기다리는 관전 세대. 맨 앞이 만드는 중."""
        self._scene_thread: _SceneThread | None = None
        self._scene_seq = 0
        self._closing = False
        self._base_config: dict[str, Any] | None = None
        """복제한 실행의 설정 전체. 화면에 칸이 없는 값(validate_every 등)은 새 실행이 여기서 그대로 가져간다.
        복제할 때만 정한다: 열기·다른 실행 이어하기·환경 불러오기 뒤의 새 실행은 그런 값을 기본값으로 시작한다."""
        self._form_exact: dict[str, tuple[Any, Any]] = {}
        """설정 키 → (칸을 채운 직후 칸 값, 설정의 원래 값). 칸은 소수 자릿수만큼 반올림해 담으므로, 사용자가 칸을
        바꾸지 않았으면 원래 값을 쓴다(복제·열기한 실행과 같은 설정으로 학습하게)."""
        self._restoring = False
        self.run_busy: Callable[[Path], bool] | None = None
        """주 창이 넣는다: 그 실행 폴더를 지금 다른 곳(비교 화면의 반복 실행 등)에서 학습 중인가."""
        self._inspect_index: int | None = None
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        self._restore_settings()
        self._method_changed(set_defaults=False)
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
        self.btn_video = QPushButton(S.VIDEO_SAVE)
        self.btn_video.clicked.connect(lambda: self.save_video())
        tl = QHBoxLayout()
        tl.addWidget(self.timeline_label)
        tl.addWidget(self.timeline, 1)
        tl.addWidget(self.btn_timeline)
        tl.addWidget(self.btn_versus)
        tl.addWidget(self.btn_live)
        tl.addWidget(self.btn_video)

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
        self.btn_edit = QPushButton(S.LAB_ENV_EDIT)
        self.btn_edit.setToolTip(S.LAB_ENV_EDIT_TIP)
        self.btn_edit.clicked.connect(self._edit_env)
        el.addWidget(self.env_label)
        el.addWidget(self.env_combo)
        el.addWidget(btn_open)
        el.addWidget(self.btn_reload)
        el.addWidget(self.btn_edit)
        lay.addWidget(env)

        st = QGroupBox(S.LAB_SETTINGS_GROUP)
        form = QFormLayout(st)
        self.method = QComboBox()
        for key in ("cmaes", "ga", "sa"):
            self.method.addItem(S.METHOD_NAMES[key], key)
        self.seed = _spin(0, 2**31 - 1, 1)
        self.popsize = _spin(2, 4096, 32)
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.001, 10.0)
        self.sigma.setDecimals(3)
        self.sigma.setSingleStep(0.05)
        self.sigma.setValue(0.5)
        self.trials = _spin(0, 1000, 0, special=S.LAB_TRIALS_ENV)
        self.workers = _spin(1, 256, default_workers())
        self.workers.valueChanged.connect(lambda _v: self._save_settings())
        # 방법마다 화면에서 바꿀 수 있는 값(명세: 학습 방법 표)
        self.ipop = QCheckBox(S.LAB_IPOP)
        self.ga_elites = _spin(0, 1000, 2)
        self.ga_crossover = _dspin(0.0, 1.0, 0.7, 2)
        self.ga_mut_rate = _dspin(0.0, 1.0, 0.1, 3)
        self.ga_mut_sigma = _dspin(0.0001, 10.0, 0.1, 3)
        self.ga_decay = _dspin(0.5, 1.0, 1.0, 4)
        self.sa_neighbors = _spin(1, 1000, 8)
        self.sa_step = _dspin(0.0001, 10.0, 0.1, 3)
        self.sa_temp = _dspin(0.0, 1e6, 0.0, 3)
        self.sa_temp.setSpecialValueText(S.LAB_SA_TEMP_AUTO)
        self.sa_cooling = _dspin(0.5, 1.0, 0.995, 4)
        form.addRow(S.LAB_METHOD, self.method)
        form.addRow(S.LAB_SEED, self.seed)
        self._method_rows: dict[str, list[QWidget]] = {"cmaes": [], "ga": [], "sa": []}

        def row(label: str, widget: QWidget, methods: tuple[str, ...]) -> None:
            form.addRow(label, widget)
            for mname in methods:
                self._method_rows[mname].append(widget)

        row(S.LAB_POPSIZE, self.popsize, ("cmaes", "ga"))
        row(S.LAB_SIGMA, self.sigma, ("cmaes",))
        row("", self.ipop, ("cmaes",))
        row(S.LAB_GA_ELITES, self.ga_elites, ("ga",))
        row(S.LAB_GA_CROSSOVER, self.ga_crossover, ("ga",))
        row(S.LAB_GA_MUT_RATE, self.ga_mut_rate, ("ga",))
        row(S.LAB_GA_MUT_SIGMA, self.ga_mut_sigma, ("ga",))
        row(S.LAB_GA_DECAY, self.ga_decay, ("ga",))
        row(S.LAB_SA_NEIGHBORS, self.sa_neighbors, ("sa",))
        row(S.LAB_SA_STEP, self.sa_step, ("sa",))
        row(S.LAB_SA_TEMP, self.sa_temp, ("sa",))
        row(S.LAB_SA_COOLING, self.sa_cooling, ("sa",))
        self._method_form = form
        form.addRow(S.LAB_TRIALS, self.trials)
        form.addRow(S.LAB_WORKERS, self.workers)
        self.method.currentIndexChanged.connect(lambda _i: self._method_changed())
        lay.addWidget(st)

        stop = QGroupBox(S.LAB_STOP_GROUP)
        sf = QFormLayout(stop)
        self.max_generations = _spin(0, 10**7, 0, special=S.LAB_NONE)
        # 최대 시험 횟수(명세 공통 정지 조건): 반복 실행·CLI가 정한 예산도 보이고 바꿀 수 있게
        self.max_evaluations = _spin(0, 2**31 - 1, 0, special=S.LAB_NONE)
        self.max_evaluations.setSingleStep(1000)
        self.max_evaluations.setGroupSeparatorShown(True)
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
        sf.addRow(S.CMP_REPEAT_STOP_EVALS, self.max_evaluations)
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
        self.fast_interval.valueChanged.connect(lambda _v: self._save_settings())
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
        # 네 값을 먼저 다 읽고 칸에 넣는다: 칸을 바꾸면 _save_settings가 불려 아직 안 읽은 값을 기본값으로 덮기 때문.
        s = self.settings

        def number(key: str, default: int) -> int:
            try:
                return int(s.value(key, default) or default)
            except (TypeError, ValueError):
                return default

        w = number("lab/workers", 0)
        speed = number("lab/speed", 1)
        mode = str(s.value("lab/mode", "spectate"))
        interval = number("lab/fast_interval", 10)
        self._restoring = True
        try:
            if w > 0:
                self.workers.setValue(w)
            if speed in SPEEDS:
                self.speed_combo.setCurrentIndex(SPEEDS.index(speed))
            if mode == "fast":
                self.rb_fast.setChecked(True)
            self.fast_interval.setValue(interval)
        finally:
            self._restoring = False

    def _save_settings(self) -> None:
        if self._restoring:
            return
        self.settings.setValue("lab/speed", self.speed)
        self.settings.setValue("lab/mode", "spectate" if self.rb_spectate.isChecked() else "fast")
        self.settings.setValue("lab/fast_interval", self.fast_interval.value())
        self.settings.setValue("lab/workers", self.workers.value())

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
        # 복제할 때만 칸이 없는 설정까지 가져간다(검토 24). 그냥 환경을 불러오면 그런 값은 기본값으로 돌아간다.
        self._base_config = dict(config) if config else None
        if config:
            self._form_from_config(config)
        if spec.file.evaluation.trials:
            self.trials.setSpecialValueText(f"{S.LAB_TRIALS_ENV} ({spec.file.evaluation.trials})")
        self._show_still_scene()
        self._update_buttons()
        return True

    def _show_still_scene(self) -> None:
        """전시 시드의 세계만 그린다(시뮬레이션하지 않는다: 화면 스레드를 막지 않게)."""
        if self.spec is None or self.layout is None:
            return
        g = self.layout.initial()[None, :]
        ep = Episode(self.spec, self.layout.body_values(g)[0] if self.layout.n_body else None, seeds.display_trial())
        self.canvas.set_scene(ep.geometry(), None, channels=list(ep.actuator_channels()))
        self.canvas.set_population([])
        self.canvas.set_overlay([])
        self.canvas.set_time(0.0)

    def _method_changed(self, set_defaults: bool = True) -> None:
        """고른 학습 방법의 설정 칸만 보인다. 개체 수 기본값은 방법마다 다르다(CMA-ES 32, GA 64)."""
        method = str(self.method.currentData())
        for mname, widgets in self._method_rows.items():
            for w in widgets:
                visible = method == mname or (w in self._method_rows[method])
                w.setVisible(visible)
                label = self._method_form.labelForField(w)
                if label is not None:
                    label.setVisible(visible)
        if set_defaults:
            self.popsize.setValue(64 if method == "ga" else 32)

    def _form_value(self, key: str, widget: QSpinBox | QDoubleSpinBox, convert: Callable[[Any], Any]) -> Any:
        """칸 값을 설정 값으로(convert). 설정에서 칸을 채운 뒤 사용자가 바꾸지 않았으면 그 설정의 원래 값.

        칸은 소수 자릿수만큼 반올림해(시그마 0.1234 → 0.123, 최대 시간 100초 → 1.7분 = 102초) 담으므로, 칸 값을 다시
        읽으면 복제·열기한 실행과 다른 설정으로 학습하게 된다(검토 24). 멈춘 학습에 설정을 적용할 때도 같다.
        """
        held = self._form_exact.get(key)
        if held is not None and held[0] == widget.value():
            return held[1]
        return convert(widget.value())

    def config_from_form(self) -> TrainConfig:
        """설정 칸의 값. 칸이 없는 값(validate_every 등)은 복제한 실행의 설정에서, 아니면 기본값."""
        method = str(self.method.currentData())
        base = TrainConfig.from_dict(self._base_config) if self._base_config else TrainConfig()
        fields = {key: self._form_value(key, widget, convert) for key, widget, convert in (
            ("run_seed", self.seed, _same), ("popsize", self.popsize, _same), ("sigma0", self.sigma, _same),
            ("trials", self.trials, _or_none), ("ga_elites", self.ga_elites, _same),
            ("ga_crossover", self.ga_crossover, _same), ("ga_mutation_rate", self.ga_mut_rate, _same),
            ("ga_mutation_sigma", self.ga_mut_sigma, _same), ("ga_sigma_decay", self.ga_decay, _same),
            ("sa_neighbors", self.sa_neighbors, _same), ("sa_step", self.sa_step, _same),
            ("sa_initial_temperature", self.sa_temp, _or_none), ("sa_cooling", self.sa_cooling, _same),
        )}
        c = dataclasses.replace(base, method=method, ipop=self.ipop.isChecked(), **fields)
        self._stop_fields(c)
        return c

    def _stop_fields(self, c: TrainConfig) -> None:
        """정지 조건 칸(새 실행, 멈춘 학습에 설정 적용)."""
        c.max_generations = self._form_value("max_generations", self.max_generations, _or_none)
        c.max_evaluations = self._form_value("max_evaluations", self.max_evaluations, _or_none)
        c.max_time = self._form_value("max_time", self.max_minutes, lambda v: v * 60.0 or None)
        c.target_score = self._form_value("target_score", self.target,
                                          lambda v: None if v <= self.target.minimum() else v)
        c.stagnation_generations = self._form_value("stagnation_generations", self.stagnation, _or_none)

    def _form_from_config(self, c: dict[str, Any]) -> None:
        """설정을 칸에 채운다(복제, 열기, 이어하기, 학습 시작). 칸이 없는 값은 여기서 가져가지 않는다(_base_config)."""
        self._form_exact = {}

        def setv(widget: Any, key: str, empty: Any, scale: float = 1.0) -> None:
            value = c.get(key)
            if value is None:
                widget.setValue(empty)
            else:
                widget.setValue(value if scale == 1.0 else float(value) / scale)
            if key in c:
                self._form_exact[key] = (widget.value(), value)

        idx = self.method.findData(c.get("method", "cmaes"))
        if idx >= 0:
            self.method.blockSignals(True)
            self.method.setCurrentIndex(idx)
            self.method.blockSignals(False)
            self._method_changed(set_defaults=False)
        self.ipop.setChecked(bool(c.get("ipop", False)))
        for widget, key, empty in ((self.ga_elites, "ga_elites", 2), (self.ga_crossover, "ga_crossover", 0.7),
                                   (self.ga_mut_rate, "ga_mutation_rate", 0.1),
                                   (self.ga_mut_sigma, "ga_mutation_sigma", 0.1),
                                   (self.ga_decay, "ga_sigma_decay", 1.0), (self.sa_neighbors, "sa_neighbors", 8),
                                   (self.sa_step, "sa_step", 0.1), (self.sa_temp, "sa_initial_temperature", 0.0),
                                   (self.sa_cooling, "sa_cooling", 0.995)):
            setv(widget, key, empty)
        setv(self.seed, "run_seed", 1)
        setv(self.popsize, "popsize", 32)
        setv(self.sigma, "sigma0", 0.5)
        setv(self.trials, "trials", 0)
        setv(self.max_generations, "max_generations", 0)
        setv(self.max_evaluations, "max_evaluations", 0)
        setv(self.max_minutes, "max_time", 0.0, scale=60.0)
        setv(self.target, "target_score", self.target.minimum())
        setv(self.stagnation, "stagnation_generations", 0)

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

    def busy_elsewhere(self, run_dir: Path | None) -> bool:
        """그 실행 폴더를 지금 다른 세션(비교 화면의 반복 실행 등)이 학습하고 있는가(주 창이 run_busy로 알려 준다)."""
        if run_dir is None or self.run_busy is None:
            return False
        return bool(self.run_busy(Path(run_dir)))

    def busy_runs_changed(self) -> None:
        """학습 중인 실행이 바뀌었다(주 창이 부른다): '이어서' 단추를 다시 정한다."""
        self._update_buttons()

    def resume_run(self, run_dir: Path, generation: int | None = None) -> bool:
        if self.client.active:
            return False
        if self.busy_elsewhere(run_dir):
            # 두 학습 프로세스가 한 실행 폴더에 쓰면 통계·체크포인트가 뒤섞인다
            QMessageBox.information(self, S.APP_NAME, S.RVL_RESUME_BUSY)
            return False
        if self.run_dir is None or Path(run_dir) != self.run_dir:
            # 다른 실행을 이어 가면 복제한 설정은 버린다: 그 뒤 '새로 시작'은 칸에 보이는 설정만 가져간다
            self._base_config = None
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
                self.client.shutdown(timeout=SHUTDOWN_WAIT_S)
            finally:
                self.unsetCursor()

    def apply_settings(self) -> None:
        if not self.client.active:
            return
        c = TrainConfig()
        self._stop_fields(c)
        # 칸에 있는 정지 조건은 다 보낸다(바뀌지 않은 값은 학습 프로세스가 기록하지 않는다)
        changes = {
            "max_generations": c.max_generations, "max_evaluations": c.max_evaluations, "max_time": c.max_time,
            "target_score": c.target_score, "stagnation_generations": c.stagnation_generations,
            "trials": self._form_value("trials", self.trials, _or_none),
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

    def _edit_env(self) -> None:
        if self.env_text is not None and self.spec is not None:
            self.edit_env_requested.emit(self.env_text, self.spec.file.name)

    def can_take_env(self) -> bool:
        """멈춘 학습이 있어 고친 환경을 받아 이어서 학습할 수 있는가."""
        return self.client.active and self.state in ("paused", "finished")

    def send_env_text(self, text: str) -> bool:
        """에디터에서 고친 환경을 멈춘 학습에 넣는다(세대 경계에서 적용, 결과는 applied 알림으로 온다).

        학습 중이 아니면 새 실행에 쓸 환경으로 받는다.
        """
        if self.can_take_env():
            return self.client.send({"type": "env", "text": text})
        if not self.client.active:
            return self.use_env_text(text, S.LAB_ENV_FROM_EDITOR, None)
        return False

    def set_spectate(self, on: bool) -> None:
        self.client.send({"type": "mode", "spectate": bool(on)})
        self.queue.clear()
        self._drop_scene_jobs()
        self._last_played = None
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
        # 앞 실행의 재생 상태를 새 실행에 넘기지 않는다(앞 실행 유전자를 새 환경으로 재생하거나 그 영상을 저장하지 않게)
        self._drop_scene_jobs()
        self._last_played = None
        self.last_fast_candidate = None
        self._last_video = None
        self._fast_last = 0
        self._replay_tag += 1  # 아직 도는 앞 실행의 다시 시뮬레이션 결과는 버린다
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
        self._warn_other_platform()
        self.timeline.setMaximum(max(1, self.generation))
        self.timeline.setValue(max(1, self.generation))
        self._show_still_scene()

    def _warn_other_platform(self) -> None:
        """다른 PC·다른 빌드에서 만든 실행이면 비트 재현이 보장되지 않는다고 알린다(명세: 재현 범위)."""
        from evolab import __version__
        from evolab.engine.runsetup import platform_info

        then = dict(self.run_info.get("platform") or {})
        if not then:
            return
        now = platform_info(0)
        keys = ("python", "numpy", "pymunk", "cma", "os", "cpu")
        diff = [k for k in keys if k in then and str(then[k]) != str(now.get(k))]
        if str(self.run_info.get("version", __version__)) != __version__:
            diff.insert(0, "evolab")
        if diff:
            self.status.setText(S.LAB_OTHER_PLATFORM.format(what=", ".join(diff)))
            self.platform_warning = diff
        else:
            self.platform_warning = []

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
                if self.layout is not None and self.layout.n_body:
                    self._scene_jobs.append(msg)  # 장면을 백그라운드에서 만든 뒤 재생 줄에 넣는다
                    self._next_scene()
                else:
                    self.queue.append(msg)
                    if self.playing is None:
                        self._play_next()
            else:
                self.client.send({"type": "played", "generation": self.generation})
        elif self.view_mode == "fast":
            self.last_fast_candidate = msg
            self._maybe_fast_replay()
        self._update_buttons()  # 첫 세대가 오면 타임라인·1세대 vs 지금을 학습 중에도 쓸 수 있다

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
        # 학습 프로세스는 내용이 바뀐 환경만 기록한다(같은 파일을 다시 읽으면 기록 없음): 표시도 같게
        env_changed = msg.get("what") == "env" and msg.get("env_text") is not None and \
            str(msg["env_text"]) != self.env_text
        if kind and (msg.get("changes") or env_changed):
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
        self._drop_scene_jobs()
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
        text = S.LAB_STATUS.format(state=S.LAB_STATE_NAMES.get(self.state, self.state), gen=self.generation,
                                   evals=evals, champ=champ_text)
        if getattr(self, "platform_warning", None):
            text += "\n" + S.LAB_OTHER_PLATFORM.format(what=", ".join(self.platform_warning))
        self.status.setText(text)

    def _update_buttons(self) -> None:
        active = self.client.active
        running = active and self.state == "running"
        paused = active and self.state in ("paused", "finished")
        self.btn_start.setEnabled(not active and self.env_text is not None)
        self.btn_pause.setEnabled(running)
        busy = not active and self.busy_elsewhere(self.run_dir)
        self.btn_continue.setEnabled(paused or (not active and self.run_dir is not None and not busy))
        self.btn_continue.setToolTip(S.RVL_CONTINUE_BUSY_TIP if busy else "")
        self.btn_step.setEnabled(paused)
        self.btn_end.setEnabled(active)
        self.btn_apply.setEnabled(paused)
        self.btn_reload.setEnabled(paused and self.env_path is not None)
        self.btn_edit.setEnabled(self.env_text is not None and (paused or not active))
        for w in (self.method, self.seed, self.popsize, self.sigma, self.workers, self.env_combo, self.ipop,
                  self.ga_elites, self.ga_crossover, self.ga_mut_rate, self.ga_mut_sigma, self.ga_decay,
                  self.sa_neighbors, self.sa_step, self.sa_temp, self.sa_cooling):
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
        scene = msg.pop("_scene", None)
        if scene is not None:  # 백그라운드에서 만든 장면(몸 유전자 환경)
            spec, (geometry, agents) = scene
        else:
            seed, track = msg["trial"]
            spec = self.spec_for(msg.get("env_version"), int(msg["generation"]))
            geometry, agents = self._scene_for(spec, msg["population"], Trial(int(seed), track))
        scores = np.asarray(msg["scores"])
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[np.argsort(-scores, kind="stable")] = np.arange(len(scores))
        self.canvas.set_scene(geometry, agents, channels=list(spec.actuator_channels))
        self.canvas.set_population(list(msg["trajectories"]), ranks=ranks, champion=int(msg["representative"]))
        self.canvas.set_overlay([])
        self._start_clock()

    def _scene_for(self, spec: EnvSpec, population: np.ndarray, trial: Trial) -> tuple[Any, list[Any] | None]:
        assert self.layout is not None
        return build_scene(spec, self.layout, population, trial)

    def _next_scene(self) -> None:
        """기다리는 관전 세대의 장면을 하나씩 차례로 백그라운드에서 만든다(세대 순서를 지킨다)."""
        if (self._scene_thread is not None or self._closing or not self._scene_jobs or self.spec is None
                or self.layout is None):
            return
        msg = self._scene_jobs[0]
        seed, track = msg["trial"]
        spec = self.spec_for(msg.get("env_version"), int(msg["generation"]))
        th = _SceneThread((self._scene_seq, msg, spec), spec, self.layout, np.asarray(msg["population"]),
                          Trial(int(seed), track), self)
        th.finished_job.connect(self._scene_done)
        th.finished.connect(lambda t=th: self._forget_thread(t))
        self._threads.append(th)
        self._scene_thread = th
        th.start()

    def _scene_done(self, tag: Any, result: Any) -> None:
        seq, msg, spec = tag
        self._scene_thread = None
        if seq == self._scene_seq and self._scene_jobs and self._scene_jobs[0] is msg:
            self._scene_jobs.popleft()
            if isinstance(result, Exception) or self.view_mode != "live":
                # 장면을 못 만들면 이 세대는 건너뛴다(학습이 재생 끝 알림을 기다리며 멈추지 않게)
                if isinstance(result, Exception) and not isinstance(result, SceneCancelled):
                    self.status.setText(self.status.text() + "\n" + S.RVL_SCENE_FAILED.format(reason=result))
                self.client.send({"type": "played", "generation": int(msg["generation"])})
            else:
                msg["_scene"] = (spec, result)
                self.queue.append(msg)
                if self.playing is None:
                    self._play_next()
        self._next_scene()

    def _drop_scene_jobs(self) -> None:
        """기다리는 장면 만들기를 버린다. 만드는 중인 것은 멈추라고 하고, 그 결과는 꼬리표가 달라 버려진다."""
        self._scene_jobs.clear()
        self._scene_seq += 1
        if self._scene_thread is not None:
            self._scene_thread.requestInterruption()

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
            self._last_played = self.playing
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
            self.canvas.set_rays(rep.rays, rep.trajectory.dt)
            if self.view_mode != "live" or self.playing is None:  # 재생이 끝난 세대를 클릭했으면 다시 돌린다
                self._start_clock()
            return
        reps: list[Replay] = result
        self.canvas.set_scene(reps[0].geometry, None, channels=reps[0].channels)
        self.canvas.set_population([])
        colors = [COLOR_OLD, COLOR_NOW] if len(reps) > 1 else [COLOR_NOW]
        self._last_video = (list(zip(reps, colors, info, strict=True)), self._replay_label)
        self.canvas.set_overlay([(r.trajectory, r.agent, c, label)
                                 for r, c, label in zip(reps, colors, info, strict=True)])
        self.loop_overlay = True
        self._start_clock()

    def save_video(self, path: Path | None = None) -> Path | None:
        """지금 보고 있는 재생(타임라인 세대 대표, 1세대 vs 지금, 고속 모드 챔피언)을 GIF로 저장한다."""
        from PySide6.QtWidgets import QApplication, QProgressDialog

        from evolab.app.export.video import export_gif
        from evolab.runs.store import sanitize_folder_name

        last = getattr(self, "_last_video", None)
        if not last:
            QMessageBox.information(self, S.VIDEO_SAVE_TITLE, S.VIDEO_NOTHING)
            return None
        items, label = last
        if path is None:
            folder = (self.run_dir / "exports") if self.run_dir is not None else paths.data_dir()
            default = folder / f"{sanitize_folder_name(label)}.gif"
            chosen, _ = QFileDialog.getSaveFileName(self, S.VIDEO_SAVE_TITLE, str(default), S.VIDEO_FILTER)
            if not chosen:
                return None
            path = Path(chosen)
        dlg = QProgressDialog(S.VIDEO_PROGRESS, S.VIDEO_CANCEL, 0, 100, self)
        dlg.setMinimumDuration(300)

        def progress(done: int, total: int) -> bool:
            dlg.setMaximum(total)
            dlg.setValue(done)
            QApplication.processEvents()
            return not dlg.wasCanceled()

        try:
            out = export_gif(items, path, title=label, progress=progress)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, S.VIDEO_SAVE_TITLE, str(exc))
            return None
        finally:
            dlg.close()
        if out is not None:
            self.status.setText(self.status.text() + "\n" + S.VIDEO_DONE.format(path=out))
        return out

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
        # 고친 환경이면 챔피언을 뽑은 세대의 환경 버전으로 재생한다(1세대 vs 지금과 같게)
        spec = self.spec_for(None, max(1, gen))
        self._run_replays(("fast", [self._replay_label]),
                          [(spec, self.layout, genome, seeds.display_trial(), False)])

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
        self._last_played = None
        while self.queue:
            msg = self.queue.popleft()
            self.client.send({"type": "played", "generation": int(msg["generation"])})
        for msg in self._scene_jobs:
            self.client.send({"type": "played", "generation": int(msg["generation"])})
        self._drop_scene_jobs()

    def back_to_live(self) -> None:
        self.loop_overlay = False
        self._replay_tag += 1  # 아직 도는 타임라인·겹쳐 보기 결과가 실시간 화면을 덮지 않게
        self.canvas.set_overlay([])
        self._close_inspect()
        self.view_mode = "live" if self.rb_spectate.isChecked() else "fast"
        if self.view_mode == "fast":
            self._maybe_fast_replay(force=True)
        elif self.playing is None:
            # 타임라인·겹쳐 보기 재생만 멈춘다. 관전 중인 세대는 계속 재생해야 끝 알림(played)이 가서 학습이 이어진다.
            self._timer.stop()

    def _inspect(self, index: int) -> None:
        msg = self.playing
        if msg is None and self.view_mode == "live":
            msg = self._last_played  # 재생이 끝나 화면에 남은 세대
        if msg is None or self.layout is None:
            return
        seed, track = msg["trial"]
        spec = self.spec_for(msg.get("env_version"), int(msg["generation"]))
        genome = np.asarray(msg["population"][index], dtype=np.float64)
        self._run_replays(("inspect", index), [(spec, self.layout, genome, Trial(int(seed), track), True)])

    def _close_inspect(self) -> None:
        self._inspect_index = None
        self.inspect.hide()
        self.canvas.set_rays(None)
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
        # 연 실행의 칸이 없는 설정(validate_every 등)은 '새로 시작'에 가져가지 않는다. 그대로 쓰려면 복제한다.
        self._base_config = None
        self._adopt_run(msg)
        self.state = "closed"
        self._update_status()
        self._update_buttons()
        return True

    def shutdown(self) -> None:
        """창을 닫을 때: 학습 프로세스를 끝내고 백그라운드 스레드가 다 끝날 때까지 기다린다.

        스레드는 이 화면의 자식이라, 도는 채로 화면과 함께 지워지면 Qt가 프로세스를 끝내 버린다(abort). 그래서 시간
        제한 없이 기다린다. 장면 만들기(개체 수에 비례해 길다)는 멈추라고 해서 개체 하나 안에 끝나게 한다.
        """
        self._timer.stop()
        self._closing = True  # 학습 프로세스가 끝나기를 기다리는 동안 온 세대의 장면은 만들지 않는다
        try:
            self._drop_scene_jobs()
            self.client.shutdown(timeout=SHUTDOWN_WAIT_S)
            self._drop_scene_jobs()
            for th in list(self._threads):
                th.requestInterruption()
            for th in list(self._threads):
                th.wait()
        finally:
            self._closing = False


def _same(v: Any) -> Any:
    return v


def _or_none(v: Any) -> Any:
    """0(칸의 '없음'·'환경 파일대로')이면 None."""
    return v or None


def _dspin(lo: float, hi: float, value: float, decimals: int) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(10.0 ** -min(decimals, 2))
    s.setValue(value)
    return s


def _spin(lo: int, hi: int, value: int, special: str | None = None) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    if special is not None:
        s.setSpecialValueText(special)
    return s


def my_best_score(spec: EnvSpec) -> float | None:
    """직접 조종 기록(.evoplay, 예전 .json) 가운데 같은 환경(내용 해시)의 최고 점수. 데이터 폴더를 바꾸기 전 기록도 본다."""
    try:
        from evolab.app.play.record import best_record_score
        from evolab.app.play.window import record_dirs

        return best_record_score(spec, record_dirs())
    except OSError:
        return None

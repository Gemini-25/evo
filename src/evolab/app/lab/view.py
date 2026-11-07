"""실험실 화면 (명세: 화면과 조작 > 실험실).

가운데 관전 화면, 오른쪽 제어 패널, 아래 통계 대시보드. 학습은 학습 프로세스(TrainingClient)가 하고, 이 화면은
결과를 받아 그리기만 한다. 다시 시뮬레이션(타임라인, 1세대 vs 지금, 고속 모드 챔피언, 개체 클릭 검사)은
백그라운드 스레드에서 같은 런타임 코드로 한다.

v1.2 「보는 재미」(설계 4.3, 5.1, D16·D17·D19·D24):
- 실행을 열거나 학습을 시작하면 대시보드에 이름표(WatchMeta)를 주고 세대 대표 파일의 관찰 기록을 배경으로 읽게 한다
  (load_history). 학습 중에는 세대 메시지(관찰 기록 포함)를 add_generation으로 넘긴다. 되감기면 다시 읽는다.
- 타임라인 슬라이더 하나가 보는 세대를 정한다(80ms 묶음, 맨 끝이면 최신 따라가기). 「세대 넘겨 보기」는 슬라이더를
  clamp(세대 수 ÷ 20, 2, 15)세대/초로 넘기고, 손으로 잡으면 멈춘다.
- 뇌 들여다보기 원천(D24): ① 캔버스가 재생 중인 개체(타임라인, 1세대 vs 지금의 "지금", 고속 챔피언, 클릭 검사,
  지도 점) ② 관전 중이면 그 세대 대표를 시험 0번으로 따로 다시 돌린 것 ③ 없으면 대시보드가 챔피언 멈춘 그림.
  ②와 뇌 타일이 늦게 보인 경우의 추적은 _brain_tag로 따로 세는 작업이라 _replay_tag를 올리지 않는다(사용자가
  시작한 타임라인·겹쳐 보기 재생을 지우지 않게). 한 번에 하나만 돌고, 돌고 있으면 새 작업을 띄우지 않는다.
- 유전자 지도 점 두 번 누르기: 최근 64세대는 화면 메모리의 정확한 유전자, 아니면 세대 대표(GMAP_INEXACT).
- 「세대끼리 경주」: app/race의 entries_from_generations + show_race.
"""

from __future__ import annotations

import collections
import dataclasses
import gc
import json
import math
from collections.abc import Callable, Mapping, Sequence
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
from evolab.app.lab.watch_types import BrainSource, WatchMeta
from evolab.engine import seeds
from evolab.engine.pool import default_workers
from evolab.engine.replay import TRACE_BRAIN, Replay, TraceMode, agent_geoms, replay
from evolab.engine.session import SessionRequest
from evolab.engine.trainer import (
    MAX_QUALIFY_ENTRANTS,
    NEW_RUN_DEFAULTS,
    RESTART_FROM,
    SWITCH_QUALIFY_NEXT,
    SWITCH_QUALIFY_PICK,
    SWITCH_RESTART_CHAMPION,
    SWITCH_RESTART_FRESH,
    TrainConfig,
    generation_trials,
    qualify_budget_problem,
)
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
SHOWN_RESTART_PATIENCE = int(NEW_RUN_DEFAULTS["restart_patience"] or 20000)
"""자동 재시작을 끈 설정(None)일 때 '정체 판단 학습 시험' 칸에 보이는 값: 체크하면 이 값으로 켠다."""
OPENED_RUN_CHECKPOINT_KEYS = ("restart_patience", "restart_from")
"""기록에서 연 실행에서 마지막 체크포인트 값을 쓰는 설정. 멈춘 상태에서 바꾼 자동 재시작 설정은 run.json(시작 때 설정)에
남지 않고 체크포인트와 사건 줄에만 남는다. 다른 설정은 v1.0처럼 run.json 값을 보인다."""
SHOWN_QUALIFY_TRIALS = 12000
"""키가 없는 예전 설정일 때 '출전마다 학습 시험' 칸에 보이는 값(데이터클래스 기본값, 예선 없음이라 쓰이지 않는다)."""
FOCUS_BATCH_MS = 80
"""타임라인 슬라이더 → 보는 세대를 묶어 넘기는 간격(끌어 옮기는 동안 대시보드를 세대마다 다시 그리지 않게)."""
FLIP_SECONDS = 20.0
FLIP_RATE = (2.0, 15.0)
"""「세대 넘겨 보기」 속도 = clamp(기록 세대 수 ÷ FLIP_SECONDS, 최소, 최대)세대/초."""
ReplayJob = tuple[EnvSpec, GenomeLayout, np.ndarray, Trial, TraceMode]
"""다시 시뮬레이션 작업 하나: (환경, 유전자 배치, 유전자, 시험, 추적 종류 False | True | "brain")."""


def settings_store() -> QSettings:
    return QSettings(str(paths.user_state_dir() / "settings.ini"), QSettings.Format.IniFormat)


# ---------------------------------------------------------------------------
# 백그라운드 다시 시뮬레이션
# ---------------------------------------------------------------------------


class _ReplayThread(QThread):
    finished_job = Signal(object, object)
    """(꼬리표, Replay 목록 또는 예외)"""

    def __init__(self, tag: Any, jobs: list[ReplayJob], parent: QObject | None = None) -> None:
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


class _WorldThread(QThread):
    """궤적 밀도 타일의 세계 그림(그 세대 시험 0번 세계, 개체 없음)을 백그라운드에서 만든다(설계 4.4)."""

    finished_job = Signal(object, object)
    """(꼬리표, WorldGeometry 또는 예외)"""

    def __init__(self, tag: Any, spec: EnvSpec, trial: Trial, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.tag = tag
        self.args = (spec, trial)

    def run(self) -> None:
        spec, trial = self.args
        try:
            out = Episode(spec, None, trial).geometry()
        except Exception as exc:  # 결과를 화면 스레드로 넘긴다(못 만들면 밀도 타일은 계속 준비 중 글)
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
        self.client.notice.connect(self._on_notice)
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
        self._switch_at: int | None = None
        """학습 방법을 마지막으로 바꾼(switch > 0) 통계 줄의 번호."""
        self._switch_note = ""
        """상태 글 셋째 줄: 마지막 학습 방법 교체 알림(자동 재시작, 예선 다음 출전, 예선 끝)."""
        self._stopped_line = ""
        """학습이 끝난(finished) 까닭 줄. 끝난 상태에서 설정을 적용해 상태 글을 다시 만들 때도 붙인다."""
        self._session_note = ""
        """상태 글 끝줄: 학습을 멈추지 않는 학습 프로세스 알림(관찰 기록 요약 실패, TRAIN_OBSERVE_FAILED). 실행이 바뀌면 지움."""

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

        # v1.2 보는 재미: 보는 세대, 세대 넘겨 보기, 뇌 원천
        self._focus_timer = QTimer(self)
        self._focus_timer.setSingleShot(True)
        self._focus_timer.setInterval(FOCUS_BATCH_MS)
        self._focus_timer.timeout.connect(self._apply_focus)
        self._flip_timer = QTimer(self)
        self._flip_timer.timeout.connect(self._flip_step)
        self.flip_rate = 0.0
        """「세대 넘겨 보기」 속도(세대/초). 넘기는 중이 아니면 0."""
        self._brain_play: dict[str, Any] | None = None
        """뇌 원천 ①: 캔버스가 재생 중인 개체 {label, gen, genome, spec, trial, rep, tried, inspect}."""
        self._brain_live: dict[str, Any] | None = None
        """뇌 원천 ②: 관전 중인 세대의 대표 {gen, genome, spec, trial, rep, tried}."""
        self._brain_tag = 0
        """뇌 추적 작업의 꼬리표(_replay_tag와 따로 센다: 뇌 작업이 사용자 재생을 지우지 않게)."""
        self._brain_thread: QThread | None = None
        self._brain_key: tuple[Any, ...] | None = None
        self._brain_serial = 0
        self._next_brain: tuple[str, int | None] | None = None
        """_run_replays가 띄울 작업 중 마지막 개체의 (뇌 제목, 세대). 부르는 쪽이 _run_replays 바로 앞에 정한다."""
        self._replay_brain: tuple[int, str, int | None, np.ndarray, EnvSpec, Trial] | None = None
        """지금 재생 작업(_replay_tag)의 뇌 원천 후보: (꼬리표, 제목, 세대, 유전자, 환경, 시험)."""
        self._race_dialog: Any = None
        self._world_thread: _WorldThread | None = None
        self._world_pending: collections.OrderedDict[str, int] = collections.OrderedDict()
        """세계 그림을 만드는 동안 새로 청한 세계 열쇠 → 세대. 밀도 타일은 열쇠마다 한 번만 청하므로 다 답한다
        (가장 나중에 청한 것부터: 지금 보는 세대가 먼저)."""
        self.last_notice = ""
        """상태 글 끝에 덧붙인 마지막 알림(지도 점 재생의 GMAP_INEXACT 등)."""
        self._map_notice = ""
        """상태 글 끝줄: 지금 재생에 대한 알림(지도 점 재생의 GMAP_INEXACT). 다음 재생·실시간 보기·다른 실행에서 지움."""
        self._brain_job: dict[str, Any] | None = None
        """도는 뇌 추적 작업의 원천(_brain_play 또는 _brain_live 사전)."""

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
        self.dashboard.focus_requested.connect(self.focus_generation)
        self.dashboard.individual_requested.connect(self.play_individual)
        self.dashboard.geometry_needed.connect(self._geometry_needed)
        self.inspect = InspectPanel()
        self.inspect.hide()
        self.inspect.closed.connect(self._close_inspect)

        # 타임라인(보는 세대: 설계 D16·D17)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setMinimum(1)
        self.timeline.setMaximum(1)
        self.timeline.setToolTip(S.LAB_TIMELINE_TIP)
        self.timeline_label = QLabel()
        self._update_timeline_label()
        self.timeline.valueChanged.connect(self._timeline_moved)
        self.timeline.rangeChanged.connect(lambda _lo, _hi: self._update_timeline_label())
        # 손으로 잡거나 키·휠로 옮기면 넘겨 보기를 멈춘다(setValue로 옮긴 것은 actionTriggered가 오지 않는다)
        self.timeline.sliderPressed.connect(self.stop_flip)
        self.timeline.actionTriggered.connect(lambda _action: self.stop_flip())
        self.btn_flip = QPushButton(S.LAB_FLIP)
        self.btn_flip.setCheckable(True)
        self.btn_flip.setToolTip(S.LAB_FLIP_TIP)
        self.btn_flip.clicked.connect(lambda on: self.start_flip() if on else self.stop_flip())
        self.btn_timeline = QPushButton(S.LAB_TIMELINE_PLAY)
        self.btn_timeline.clicked.connect(lambda: self.play_generation(self.timeline.value()))
        self.btn_versus = QPushButton(S.LAB_VERSUS)
        self.btn_versus.clicked.connect(self.play_versus)
        self.btn_race = QPushButton(S.LAB_RACE_GENERATIONS)
        self.btn_race.setToolTip(S.LAB_RACE_GENERATIONS_TIP)
        self.btn_race.clicked.connect(lambda: self.race_generations())
        self.btn_live = QPushButton(S.LAB_LIVE)
        self.btn_live.clicked.connect(self.back_to_live)
        self.btn_video = QPushButton(S.VIDEO_SAVE)
        self.btn_video.clicked.connect(lambda: self.save_video())
        tl = QHBoxLayout()
        tl.addWidget(self.timeline_label)
        tl.addWidget(self.timeline, 1)
        tl.addWidget(self.btn_flip)
        tl.addWidget(self.btn_timeline)
        tl2 = QHBoxLayout()
        tl2.addWidget(self.btn_versus)
        tl2.addWidget(self.btn_race)
        tl2.addWidget(self.btn_live)
        tl2.addStretch(1)
        tl2.addWidget(self.btn_video)

        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(self.canvas, 1)
        row.addWidget(self.inspect)
        cl.addLayout(row, 1)
        cl.addLayout(tl)
        cl.addLayout(tl2)

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
        lay.addWidget(stop)

        # 학습 안정성(v1.1): 정체하면 새로 시작은 멈춘 학습에도 '적용'하고, 예선은 새 실행에서만 정한다
        stab = QGroupBox(S.LAB_STABILITY_GROUP)
        bf = QFormLayout(stab)
        self.restart_on = QCheckBox(S.LAB_RESTART)
        self.restart_on.setToolTip(S.LAB_RESTART_TIP)
        self.restart_on.setChecked(NEW_RUN_DEFAULTS["restart_patience"] is not None)
        self.restart_patience = _spin(1000, 10**7, SHOWN_RESTART_PATIENCE)
        self.restart_patience.setSingleStep(1000)
        self.restart_patience.setGroupSeparatorShown(True)
        self.restart_patience.setSuffix(S.LAB_UNIT_TRIALS)
        self.restart_patience.setToolTip(S.LAB_RESTART_TIP)
        self.restart_from = QComboBox()
        for key in RESTART_FROM:
            self.restart_from.addItem(S.RESTART_FROM_NAMES[key], key)
        self.restart_from.setCurrentIndex(max(0, self.restart_from.findData(NEW_RUN_DEFAULTS["restart_from"])))
        self.restart_from.setToolTip(S.LAB_RESTART_FROM_TIP)
        self.qualify_entrants = _spin(1, MAX_QUALIFY_ENTRANTS, int(NEW_RUN_DEFAULTS["qualify_entrants"]),
                                      special=S.LAB_QUALIFY_OFF)
        self.qualify_entrants.setToolTip(S.LAB_QUALIFY_TIP)
        self.qualify_trials = _spin(1000, 10**6, int(NEW_RUN_DEFAULTS["qualify_trials"]))
        self.qualify_trials.setSingleStep(1000)
        self.qualify_trials.setGroupSeparatorShown(True)
        self.qualify_trials.setSuffix(S.LAB_UNIT_TRIALS)
        self.qualify_trials.setToolTip(S.LAB_QUALIFY_TIP)
        self.stability_hint = QLabel()
        self.stability_hint.setWordWrap(True)
        self.stability_hint.setStyleSheet("color: #8a93a6;")  # 흐린 글(밝은·어두운 테마 공통)
        bf.addRow(self.restart_on)
        bf.addRow(S.LAB_RESTART_PATIENCE, self.restart_patience)
        bf.addRow(S.LAB_RESTART_FROM, self.restart_from)
        bf.addRow(S.LAB_QUALIFY, self.qualify_entrants)
        bf.addRow(S.LAB_QUALIFY_TRIALS, self.qualify_trials)
        bf.addRow(self.stability_hint)
        self.btn_apply = QPushButton(S.LAB_APPLY)
        self.btn_apply.setToolTip(S.LAB_APPLY_TIP)
        self.btn_apply.clicked.connect(self.apply_settings)
        bf.addRow(self.btn_apply)
        lay.addWidget(stab)
        # 세대당 학습 시험이 바뀌는 칸(방법, 개체 수, 이웃 수, 시험 수)과 새 칸이 바뀌면 칸 상태·어림 글을 다시 정한다
        for changed in (self.method.currentIndexChanged, self.popsize.valueChanged, self.sa_neighbors.valueChanged,
                        self.trials.valueChanged, self.stagnation.valueChanged, self.restart_on.toggled,
                        self.restart_patience.valueChanged, self.qualify_entrants.valueChanged,
                        self.qualify_trials.valueChanged):
            changed.connect(lambda *_a: self._update_stability())

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
        self._set_watch_meta()
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
        """설정 칸의 값. 칸이 없는 값(validate_every 등)은 복제한 실행의 설정에서, 아니면 새 실행 기본값(for_new_run)."""
        method = str(self.method.currentData())
        base = TrainConfig.from_dict(self._base_config) if self._base_config else TrainConfig.for_new_run()
        fields = {key: self._form_value(key, widget, convert) for key, widget, convert in (
            ("run_seed", self.seed, _same), ("popsize", self.popsize, _same), ("sigma0", self.sigma, _same),
            ("trials", self.trials, _or_none), ("ga_elites", self.ga_elites, _same),
            ("ga_crossover", self.ga_crossover, _same), ("ga_mutation_rate", self.ga_mut_rate, _same),
            ("ga_mutation_sigma", self.ga_mut_sigma, _same), ("ga_sigma_decay", self.ga_decay, _same),
            ("sa_neighbors", self.sa_neighbors, _same), ("sa_step", self.sa_step, _same),
            ("sa_initial_temperature", self.sa_temp, _or_none), ("sa_cooling", self.sa_cooling, _same),
            ("qualify_entrants", self.qualify_entrants, _same), ("qualify_trials", self.qualify_trials, _same),
        )}
        c = dataclasses.replace(base, method=method, ipop=self.ipop.isChecked(), **fields, **self._restart_fields())
        self._stop_fields(c)
        return c

    def _restart_fields(self) -> dict[str, Any]:
        """자동 재시작 칸(새 실행, 멈춘 학습에 설정 적용). 체크를 끄면 restart_patience는 None(끔)."""
        on = self.restart_on.isChecked()
        return {"restart_patience": self._form_value("restart_patience", self.restart_patience, _same) if on else None,
                "restart_from": str(self.restart_from.currentData())}

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
            if key in c and (value is None or _shows(widget, float(value) / scale)):
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
        # 학습 안정성: 키가 없는 예전(v1.0) 설정은 자동 재시작·예선 없이 돈 실행이다(데이터클래스 기본값 = 끔)
        patience = c.get("restart_patience")
        self.restart_on.setChecked(patience is not None)
        if patience is None:
            # 끔은 체크 상자가 나타낸다. 칸에는 켤 때 쓸 값을 두고 원래 값(None)으로 붙잡지 않는다(체크하면 칸 값으로 켠다)
            self.restart_patience.setValue(SHOWN_RESTART_PATIENCE)
        else:
            setv(self.restart_patience, "restart_patience", SHOWN_RESTART_PATIENCE)
        self.restart_from.setCurrentIndex(max(0, self.restart_from.findData(c.get("restart_from", "fresh"))))
        setv(self.qualify_entrants, "qualify_entrants", 1)
        setv(self.qualify_trials, "qualify_trials", SHOWN_QUALIFY_TRIALS)

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
        problem = qualify_budget_problem(config)  # 예선만으로 최대 시험 횟수를 다 쓰면 시작하지 않는다
        if problem is not None:
            QMessageBox.warning(self, S.LAB_CHANGE_FAILED, problem)
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
        # 칸에 있는 정지 조건과 자동 재시작 설정은 다 보낸다(바뀌지 않은 값은 학습 프로세스가 기록하지 않는다).
        # 예선 설정은 실행 처음에만 뜻이 있어 보내지 않는다(학습 중에는 칸도 잠근다).
        changes = {
            "max_generations": c.max_generations, "max_evaluations": c.max_evaluations, "max_time": c.max_time,
            "target_score": c.target_score, "stagnation_generations": c.stagnation_generations,
            "trials": self._form_value("trials", self.trials, _or_none),
            **self._restart_fields(),
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
            self._brain_live = None  # 관전·고속을 바꾸면 앞 원천은 캔버스에서 사라진다(고속은 새 챔피언 재생이 ①)
            self._brain_play = None
            self._set_follow(None)
            if not on:
                self._fast_last = 0
                self._maybe_fast_replay(force=True)
            self._update_brain_source()
        self._save_settings()

    def set_speed(self, n: int) -> None:
        self.speed = n
        self._save_settings()

    # ------------------------------------------------------------------
    # 학습 프로세스에서 온 것
    # ------------------------------------------------------------------
    def _reset_run_view(self) -> None:
        self.rows = []
        self._session_note = ""
        self._map_notice = ""
        self.last_notice = ""
        self.events = []
        self.champion = None
        self._switch_at = None
        self._switch_note = ""
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
        self.stop_flip()
        self._brain_tag += 1  # 앞 실행의 뇌 추적 결과도 버린다
        self._brain_play = None
        self._brain_live = None
        self._replay_brain = None
        self._world_pending.clear()
        self.dashboard.clear()  # 보는 재미 기록·보는 세대·뇌 원천도 비운다
        self._brain_key = None
        self.timeline.setMaximum(1)
        self._update_timeline_label()
        self._close_inspect()

    def _adopt_run(self, msg: dict[str, Any]) -> None:
        self.run_dir = Path(msg["run_dir"])
        self.run_info = dict(msg.get("run_info") or {})
        if msg.get("config"):
            # 지금 학습이 쓰는 설정(이어 간 실행이면 run.json 뒤에 멈춘 상태에서 바꾼 값이 체크포인트에 있다).
            # 상태 글의 예선·자동 재시작 값은 여기서 읽고, 설정을 적용하면 _on_applied가 고친다.
            self.run_info["config"] = {**(self.run_info.get("config") or {}), **msg["config"]}
        self.env_text = str(msg["env_text"])
        self.spec = compile_env(self.env_text)
        self.layout = GenomeLayout.from_dict(msg["layout"]) if msg.get("layout") else layout_for(self.spec)
        self.generation = int(msg.get("generation", 0))
        self.champion = msg.get("champion")
        self.rows = list(msg.get("stats") or [])
        self.events = list(msg.get("events") or [])
        # 통계 전체를 받았으면 마지막 학습 방법 교체 줄로 알림을 다시 만든다
        self._switch_at = next((i for i in range(len(self.rows) - 1, -1, -1)
                                if (_int_value(self.rows[i].get("switch")) or 0) > 0), None)
        self._switch_note = switch_note(self.rows, self._switch_at) if self._switch_at is not None else ""
        self._spec_cache.clear()
        self.env_label.setText(f"{self.spec.file.name}\n{self.run_dir}")
        self._form_from_config(msg.get("config") or {})
        self._setup_dashboard()
        self.dashboard.set_rows(self.rows)
        # 보는 재미: 이름표를 주고 세대 대표 파일의 관찰 기록을 배경으로 읽는다(새 실행도: 앞 세대 대표를 파일에서 읽게)
        self._set_watch_meta()
        self.dashboard.load_history(self.run_dir, self.rows)
        self._warn_other_platform()
        self.timeline.setMaximum(max(1, self.generation))
        self.timeline.setValue(max(1, self.generation))
        self._update_timeline_label()
        self._apply_focus()
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
        note = self._session_note
        self._reset_run_view()
        self._session_note = note  # 같은 학습 프로세스의 알림(한 번만 옴)은 되감아도 남긴다
        self._adopt_run(msg)

    def _on_generation(self, msg: dict[str, Any]) -> None:
        row = msg["row"]
        self.generation = int(msg["generation"])
        self.rows.append(row)
        self._note_switch()
        self.dashboard.add_row(row)
        if msg.get("champion") is not None:
            self.champion = msg["champion"]
        self._feed_dashboard(msg)
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
            self._stopped_line = S.LAB_STOPPED_BECAUSE.format(reason=reason)
            self.status.setText(self.status.text() + "\n" + self._stopped_line)
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
        # 상태 글을 바뀐 설정(정체 판단 시험, 자동 재시작 끔·켬)으로 다시 만든다: 멈춘 동안에는 다음 세대 알림이 오지 않는다
        self._update_status()
        if self.state == "finished" and self._stopped_line:
            self.status.setText(self.status.text() + "\n" + self._stopped_line)
        self.status.setText(self.status.text() + "\n" + S.LAB_CHANGE_DONE.format(gen=gen))

    def _on_notice(self, message: str) -> None:
        """학습은 그대로 이어지는 알림: 막는 상자 대신 상태 글 끝줄로(세대가 와서 상태 글을 다시 만들어도 남는다)."""
        self._session_note = message
        self._update_status()

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
            text = S.LAB_STATE_NAMES.get(self.state, self.state)
            for note in (self._session_note, self._map_notice):
                if note:
                    text += "\n" + note
            self.status.setText(text)
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
        learner = self._learner_status(last)
        if learner:
            text += "\n" + learner
        if self._switch_note:
            text += "\n" + self._switch_note
        if getattr(self, "platform_warning", None):
            text += "\n" + S.LAB_OTHER_PLATFORM.format(what=", ".join(self.platform_warning))
        if self._session_note:
            text += "\n" + self._session_note
        if self._map_notice:
            text += "\n" + self._map_notice
        self.status.setText(text)

    def _learner_status(self, row: Mapping[str, Any]) -> str | None:
        """상태 글 둘째 줄: 예선 중이면 출전 번호와 출전 시험, 본선이면 시도 번호와 정체(재시작 시계).

        v1.1 학습 방법 열이 있는 통계 줄만(예전 열 머리로 시작한 실행은 None). 값은 그 세대를 만든 학습 방법 기준이고,
        예선 출전 수·출전마다 시험·정체 판단 시험은 지금 학습 설정(run_info의 config)에서 읽는다.
        """
        values = [_int_value(row.get(k)) for k in ("learner", "learner_trials", "learner_stall", "restarts",
                                                    "qualifying")]
        if any(v is None for v in values):
            return None
        learner, trials, stall, restarts, qualifying = values
        config = self.run_info.get("config") or {}
        if qualifying:
            return S.LAB_STATUS_QUALIFY.format(n=learner + 1, k=_int_value(config.get("qualify_entrants")) or 1,
                                               t=trials, b=_int_value(config.get("qualify_trials")) or 0)
        patience = _int_value(config.get("restart_patience"))
        if patience is None:
            return S.LAB_STATUS_ATTEMPT_OFF.format(n=restarts + 1)
        return S.LAB_STATUS_ATTEMPT.format(n=restarts + 1, stall=stall, p=patience)

    def _note_switch(self) -> None:
        """방금 받은 통계 줄이 학습 방법을 바꿨으면(switch > 0) 알림을 정한다. 바로 앞 줄이 교체였으면 다시 만든다
        (예선 끝 알림의 통과자 번호는 다음 줄의 learner로 확인한다)."""
        i = len(self.rows) - 1
        if (_int_value(self.rows[i].get("switch")) or 0) > 0:
            self._switch_at = i
        elif self._switch_at is None or self._switch_at != i - 1:
            return
        self._switch_note = switch_note(self.rows, self._switch_at)

    def _update_stability(self) -> None:
        """학습 안정성 칸의 켜짐(예선 칸은 학습 중 잠금, 재시작 칸은 늘 켜 두고 '적용'은 멈췄을 때만)과 어림 글."""
        active = self.client.active
        self.restart_patience.setEnabled(self.restart_on.isChecked())
        self.qualify_entrants.setEnabled(not active)
        self.qualify_trials.setEnabled(not active and self.qualify_entrants.value() > 1)
        self.stability_hint.setText(self.stability_hint_text())

    def stability_hint_text(self) -> str:
        """칸 값으로 어림한 예선 합계와 정체 판단까지의 세대 수(학습 방법 하나의 세대당 학습 시험 기준).

        정체 멈춤(N세대)이 자동 재시작보다 먼저 오면 덧붙여 알린다. 환경이 없거나 둘 다 끄면 빈 글.
        """
        if self.spec is None:
            return ""
        c = self.config_from_form()
        try:
            per = generation_trials(c, self.spec)
        except (TypeError, ValueError):
            return ""
        if per <= 0:
            return ""
        k, b, p = c.qualify_entrants, c.qualify_trials, c.restart_patience
        rgens = math.ceil(p / per) if p is not None else 0
        if k > 1:
            qualify = {"total": k * b, "qgens": k * math.ceil(b / per)}
            text = (S.LAB_STABILITY_HINT.format(rgens=rgens, **qualify) if p is not None
                    else S.LAB_STABILITY_HINT_QUALIFY_ONLY.format(**qualify))
        elif p is not None:
            text = S.LAB_STABILITY_HINT_NO_QUALIFY.format(rgens=rgens)
        else:
            return ""
        stop = c.stagnation_generations
        if p is not None and stop is not None and stop * per < p:
            text += S.LAB_STABILITY_STOP_FIRST.format(stop=stop)
        return text

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
        self._update_stability()  # 예선 칸(qualify_entrants, qualify_trials)도 학습 중이면 끈다
        has_gen = self.generation > 0 and self.run_dir is not None
        self.btn_timeline.setEnabled(has_gen)
        self.btn_versus.setEnabled(has_gen)
        self.btn_flip.setEnabled(self.generation >= 2 or self._flip_timer.isActive())
        self.btn_race.setEnabled(self.generation >= 2 and self.run_dir is not None)

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
        if self._brain_play is not None and self._brain_play.get("inspect"):
            # 앞 세대에서 클릭 검사한 개체: 캔버스가 더는 그 개체를 재생하지 않는다(뇌는 이 세대 대표 ②로)
            self._brain_play = None
        self._set_follow(msg)
        self._start_clock()
        self._live_brain_for(msg, spec)

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
        if self._brain_wanted():
            self._brain_check()
            self.dashboard.show_time(self.play_t)
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
                live = self._brain_live
                if live is not None and live["rep"] is None and self._brain_job is not live:
                    # 재생이 끝날 때까지 추적이 없었다(뇌 그림이 안 보였다 등): 늦게 보여도 다시 돌 수 없으니 ②를 놓고
                    # 보는 세대까지의 챔피언 멈춘 그림(③)으로(설계 D24)
                    self._brain_live = None
                    self._update_brain_source()
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

    def _run_replays(self, tag: Any, jobs: list[ReplayJob]) -> None:
        self._replay_tag += 1
        self._notice("")  # 앞 재생의 알림(지도 점 재생은 이 뒤에 다시 정한다)
        brain, self._next_brain = self._next_brain, None
        if brain is not None and jobs:  # 마지막 개체가 뇌 원천 ① 후보(겹쳐 보기는 "지금" 쪽)
            spec, _layout, genome, trial, _trace = jobs[-1]
            self._replay_brain = (self._replay_tag, brain[0], brain[1], np.asarray(genome, dtype=np.float64), spec, trial)
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
        self._adopt_play_brain(seq, result[-1], inspect=kind == "inspect")
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
        except MemoryError:
            QMessageBox.warning(self, S.VIDEO_SAVE_TITLE, S.VIDEO_NO_MEMORY)
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
        self._next_brain = (self._replay_label, gen)
        self._run_replays(("fast", [self._replay_label]),
                          [(spec, self.layout, genome, seeds.display_trial(), self._brain_trace())])

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
        self._next_brain = (self._replay_label, generation)
        self._run_replays(("timeline", [self._replay_label]),
                          [(spec, self.layout, genome, seeds.display_trial(), self._brain_trace())])
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
        self._next_brain = (labels[1], gen)  # 뇌 그림은 "지금" 쪽
        self._run_replays(("versus", labels), [(self.spec_for(None, 1), self.layout, first, trial, False),
                                                (self.spec_for(None, max(1, gen)), self.layout, now, trial,
                                                 self._brain_trace())])
        return True

    def _ack_pending(self) -> None:
        """관전 재생을 그만 볼 때 기다리는 학습이 멈추지 않게 받은 세대를 다 봤다고 알린다."""
        self.loop_overlay = False
        self._brain_live = None  # 관전 재생을 그만 본다: 뇌 원천 ②도 없다
        self._set_follow(None)  # 보는 재미 그래프도 관전 세대 대신 타임라인(맨 끝이면 최신)을 따른다
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
        self._notice("")
        self._brain_play = None  # 캔버스가 그 재생을 그만 보인다
        self._update_brain_source()
        self.view_mode = "live" if self.rb_spectate.isChecked() else "fast"
        self._set_follow(self.playing if self.view_mode == "live" else None)
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
        gen = int(msg["generation"])
        spec = self.spec_for(msg.get("env_version"), gen)
        genome = np.asarray(msg["population"][index], dtype=np.float64)
        self._next_brain = (S.GMAP_LABEL_INDIVIDUAL.format(gen=gen, index=index), gen)
        self._run_replays(("inspect", index), [(spec, self.layout, genome, Trial(int(seed), track), True)])

    def _close_inspect(self) -> None:
        self._inspect_index = None
        self.inspect.hide()
        self.canvas.set_rays(None)
        if self.view_mode == "live":
            self.canvas.set_overlay([])
        if self._brain_play is not None and self._brain_play.get("inspect"):
            self._brain_play = None  # 클릭 검사한 개체를 더 보이지 않는다
            self._update_brain_source()

    # ------------------------------------------------------------------
    # 보는 재미(v1.2): 기록, 보는 세대, 세대 넘겨 보기, 뇌 원천, 지도 점, 궤적 밀도 세계, 세대끼리 경주
    # ------------------------------------------------------------------
    def _set_watch_meta(self) -> None:
        """대시보드 「보는 재미」 이름표(유전자 구조, 몸 수치, 구동기, 신경망 입출력 이름)를 지금 환경으로."""
        if self.spec is None or self.layout is None:
            return
        try:
            meta = WatchMeta.from_spec(self.spec, self.layout)
        except (ValueError, KeyError):  # 이름표를 못 만들면 보는 재미 쪽만 비어 있다(학습·재생과 무관)
            return
        self.dashboard.set_watch_meta(meta)

    def _feed_dashboard(self, msg: Mapping[str, Any]) -> None:
        """세대 메시지의 관찰 기록·집단·세대 대표·바뀐 챔피언을 대시보드로(설계 7.3 add_generation)."""
        champion: tuple[Any, ...] | None = None
        champ = msg.get("champion")
        if isinstance(champ, Mapping) and champ.get("genome") is not None:
            cg = _int_value(champ.get("generation"))
            genome = np.asarray(champ["genome"], dtype=np.float64)
            score = _float_value(champ.get("score"))
            if cg is not None:
                champion = (cg, genome) if score is None else (cg, genome, score)
        rep = _int_value(msg.get("representative"))
        self.dashboard.add_generation(self.generation, msg.get("observation"), msg.get("population"),
                                      rep if rep is not None else 0, champion)  # type: ignore[arg-type]

    # -- 보는 세대(타임라인) -----------------------------------------------------
    def _update_timeline_label(self) -> None:
        total = self.timeline.maximum() if self.generation > 0 else 0
        gen = min(self.timeline.value(), total)
        follow = self.dashboard.follow
        if follow is not None and gen >= total > 0:  # 맨 끝(따라가기): 관전 중이면 재생 중인 세대
            gen = min(follow, total)
        self.timeline_label.setText(S.LAB_TIMELINE_OF.format(gen=gen, total=total))

    def _set_follow(self, msg: Mapping[str, Any] | None) -> None:
        """보는 재미 그래프가 따라갈 세대(타임라인이 맨 끝일 때): 관전 중이면 캔버스가 재생 중인 세대 메시지, None이면
        기록의 마지막 세대. 학습은 재생 중인 세대보다 한 세대 앞서 계산하므로(관전) 이것이 없으면 그래프·궤적 밀도가
        캔버스·뇌 그림보다 한 세대 앞선다."""
        gen = _int_value(msg.get("generation")) if msg is not None else None
        self.dashboard.set_follow(gen)
        self._update_timeline_label()

    def _timeline_moved(self, _value: int) -> None:
        self._update_timeline_label()
        if not self._focus_timer.isActive():  # 끌어 옮기는 동안 FOCUS_BATCH_MS마다 한 번만 넘긴다
            self._focus_timer.start()

    def _apply_focus(self) -> None:
        """슬라이더 값 → 대시보드 보는 세대(맨 끝이면 None = 최신 따라가기)."""
        self._focus_timer.stop()
        v = self.timeline.value()
        self.dashboard.set_focus(None if self.generation <= 0 or v >= self.timeline.maximum() else v)

    def focus_generation(self, generation: int) -> None:
        """그 세대를 보는 세대로(유전자 지도의 앞 세대 점을 눌렀다): 타임라인을 옮긴다."""
        self.stop_flip()
        if self.generation <= 0:
            return
        self.timeline.setValue(max(1, min(int(generation), self.timeline.maximum())))
        self._apply_focus()

    # -- 세대 넘겨 보기(D17) -----------------------------------------------------
    def start_flip(self) -> bool:
        """슬라이더를 clamp(세대 수 ÷ 20, 2, 15)세대/초로 한 세대씩 넘긴다. 맨 끝에서 누르면 1세대부터."""
        n = self.timeline.maximum()
        if self.generation < 2 or n < 2:
            self.btn_flip.setChecked(False)
            return False
        self.flip_rate = min(max(n / FLIP_SECONDS, FLIP_RATE[0]), FLIP_RATE[1])
        if self.timeline.value() >= n:
            self.timeline.setValue(1)
        self._apply_focus()
        self.dashboard.set_flipping(True)
        self._flip_timer.setInterval(max(1, round(1000.0 / self.flip_rate)))
        self._flip_timer.start()
        self.btn_flip.setChecked(True)
        self.btn_flip.setText(S.LAB_FLIP_STOP_V2)  # 학습 「멈춤」 단추와 헷갈리지 않게
        return True

    def stop_flip(self) -> None:
        if not self._flip_timer.isActive() and not self.btn_flip.isChecked():
            return
        self._flip_timer.stop()
        self.flip_rate = 0.0
        self.dashboard.set_flipping(False)
        self.btn_flip.setChecked(False)
        self.btn_flip.setText(S.LAB_FLIP)
        self.btn_flip.setEnabled(self.generation >= 2)

    @property
    def flipping(self) -> bool:
        return self._flip_timer.isActive()

    def _flip_step(self) -> None:
        n = self.timeline.maximum()
        v = self.timeline.value() + 1
        if v >= n:
            self.timeline.setValue(n)
            self.stop_flip()
        else:
            self.timeline.setValue(v)
        self._apply_focus()  # 넘겨 보기는 정한 속도라 묶지 않고 세대마다 넘긴다

    # -- 뇌 원천(D24) ----------------------------------------------------------
    def _brain_wanted(self) -> bool:
        """뇌 그림이 보이는가(대시보드가 떠 있고 「보는 재미」 쪽이며 뇌 타일을 숨기지 않음)."""
        return self.dashboard.brain_visible()

    def _brain_trace(self) -> TraceMode:
        """캔버스 재생 작업의 추적: 뇌 그림이 보이면 가벼운 추적("brain"), 아니면 없음."""
        return TRACE_BRAIN if self._brain_wanted() else False

    def _next_serial(self) -> int:
        self._brain_serial += 1
        return self._brain_serial

    def _adopt_play_brain(self, seq: int, rep: Replay, *, inspect: bool) -> None:
        """캔버스 재생 결과가 왔다: 그 재생의 (마지막) 개체가 뇌 원천 ①."""
        cand = self._replay_brain
        if cand is None or cand[0] != seq:
            return
        _seq, label, gen, genome, spec, trial = cand
        traced = rep.observations is not None and rep.activations is not None
        self._brain_play = {"serial": self._next_serial(), "label": label, "gen": gen, "genome": genome,
                            "spec": spec, "trial": trial, "rep": rep if traced else None, "tried": traced,
                            "inspect": inspect}
        self._update_brain_source()
        if self._brain_wanted():
            self._brain_check()

    def _live_brain_for(self, msg: dict[str, Any], spec: EnvSpec) -> None:
        """관전으로 새 세대를 재생하기 시작했다: 원천 ②를 그 세대 대표(시험 0번)로. 추적은 뇌 그림이 보일 때만 따로."""
        try:
            rep_i = int(msg["representative"])
            genome = np.asarray(msg["population"][rep_i], dtype=np.float64)
            seed, track = msg["trial"]
            trial = Trial(int(seed), track)
        except (KeyError, IndexError, TypeError, ValueError):
            self._brain_live = None
        else:
            self._brain_live = {"serial": self._next_serial(), "gen": int(msg["generation"]), "genome": genome,
                                "spec": spec, "trial": trial, "rep": None, "tried": False, "msg": msg}
        self._update_brain_source()
        if self._brain_wanted():
            self._brain_check()

    def _update_brain_source(self) -> None:
        """우선순위(① 캔버스 재생 → ② 관전 중 대표 → 없음)로 뇌 원천을 정해 대시보드에 넘긴다(바뀔 때만)."""
        play, live = self._brain_play, self._brain_live
        src: BrainSource | None = None
        key: tuple[Any, ...] | None = None
        if play is not None:
            key = ("playing", play["serial"], play["rep"] is not None)
            if key != self._brain_key:
                src = BrainSource(play["label"], "playing", play["gen"], play["genome"], play["rep"])
        elif live is not None and self.view_mode == "live":
            key = ("live", live["serial"], live["rep"] is not None)
            if key != self._brain_key:
                src = BrainSource("", "live", live["gen"], live["genome"], live["rep"])
        if key == self._brain_key:
            return
        self._brain_key = key
        self.dashboard.set_brain_source(src)

    def _brain_check(self) -> None:
        """뇌 그림이 보일 때(_tick, 새 재생): 지금 원천에 추적이 없으면 따로 다시 돌린다(trace="brain").

        _run_replays를 거치지 않고 _brain_tag로 따로 센다(_replay_tag를 올리면 사용자가 시작한 재생이 버려진다).
        한 번에 하나만: 앞 작업이 돌고 있으면 띄우지 않는다(관전 세대는 그 세대를 건너뛸 수 있다).
        """
        if self._brain_thread is not None or self.layout is None or self._closing:
            return
        job: dict[str, Any] | None = None
        which = ""
        play, live = self._brain_play, self._brain_live
        if play is not None:
            if play["rep"] is None and not play["tried"]:
                job, which = play, "play"
        elif live is not None and self.view_mode == "live":
            if live["rep"] is None and not live["tried"] and self.playing is live["msg"]:
                job, which = live, "live"
        if job is None:
            return
        job["tried"] = True
        self._brain_job = job
        self._brain_tag += 1
        th = _ReplayThread(("brain", self._brain_tag, which, job),
                           [(job["spec"], self.layout, job["genome"], job["trial"], TRACE_BRAIN)], self)
        th.finished_job.connect(self._brain_done)
        th.finished.connect(lambda t=th: self._brain_thread_finished(t))
        self._threads.append(th)
        self._brain_thread = th
        th.start()

    def _brain_thread_finished(self, th: _ReplayThread) -> None:
        if self._brain_thread is th:
            self._brain_thread = None
            self._brain_job = None
        self._forget_thread(th)

    def _brain_done(self, tag: Any, result: Any) -> None:
        _kind, seq, which, job = tag
        if self._brain_job is job:
            self._brain_job = None
        if seq != self._brain_tag:
            return  # 더 새로운 작업·다른 실행
        if which == "live" and self._brain_live is job:
            # 같은 세대가 아직 재생 중이거나, 재생이 끝나 그 세대의 마지막 장면이 캔버스에 남아 있으면 붙인다
            showing = self.playing is job["msg"] or (self.playing is None and self._last_played is job["msg"])
            if isinstance(result, Exception) or self.view_mode != "live" or not showing:
                # 실패(뇌 그림은 덧붙이는 것이라 알리지 않는다)했거나 그 세대가 캔버스에서 사라졌다: 「계산하는 중」에
                # 머물지 않게 ②를 놓는다(재생 중이 아니면 ③ 챔피언 멈춘 그림)
                if self.playing is not job["msg"]:
                    self._brain_live = None
                    self._update_brain_source()
                return
            job["rep"] = result[0]
        elif which == "play" and self._brain_play is job and not isinstance(result, Exception):
            job["rep"] = result[0]
        else:
            return
        self._update_brain_source()

    @property
    def brain_busy(self) -> bool:
        """뇌 추적 작업이 도는 중인가(시험·측정용)."""
        return self._brain_thread is not None

    # -- 지도 점 재생(D19), 궤적 밀도 세계, 세대끼리 경주 ----------------------------
    def _notice(self, text: str) -> None:
        """지금 재생에 대한 알림(빈 글이면 지움): 상태 글 끝줄 자리 하나라 쌓이지 않는다."""
        self.last_notice = text
        if text == self._map_notice:
            return
        self._map_notice = text
        self._update_status()
        if self.state == "finished" and self._stopped_line:
            self.status.setText(self.status.text() + "\n" + self._stopped_line)

    def play_individual(self, generation: int, index: int) -> bool:
        """유전자 지도 점을 두 번 눌렀다: 그 개체를 전시 시드로 재생한다. 학습 중 받은 최근 64세대는 정확한 유전자,
        아니면 그 세대 대표(정확한 유전자가 대표 자신이 아니면 GMAP_INEXACT). index −1은 세대 대표 점."""
        if self.layout is None or self.spec is None:
            return False
        gen, idx = int(generation), int(index)
        genome = self.dashboard.exact_genome(gen, idx) if idx >= 0 else None
        notice = ""
        if genome is not None:
            label = S.GMAP_LABEL_INDIVIDUAL.format(gen=gen, index=idx)
        else:
            genome = self.dashboard.rep_genome(gen)
            if genome is None:
                return False
            exact = idx < 0 or idx == self.dashboard.rep_index(gen)
            label = S.GMAP_LABEL_INDIVIDUAL.format(gen=gen, index=idx) if exact and idx >= 0 else \
                S.LAB_LABEL_REP.format(gen=gen)
            if not exact:
                notice = S.GMAP_INEXACT.format(gen=gen)
        self.view_mode = "replay"
        self._ack_pending()
        self._replay_label = label
        spec = self.spec_for(None, gen)
        self._next_brain = (label, gen)
        self._run_replays(("individual", [label]),
                          [(spec, self.layout, np.asarray(genome, dtype=np.float64), seeds.display_trial(),
                            self._brain_trace())])
        self._notice(notice)
        return True

    def _geometry_needed(self, generation: int, world_key: str) -> None:
        """궤적 밀도 타일이 그 세대 시험 0번 세계 그림을 청했다: 그 세대 환경 버전으로 세계만 백그라운드에서 만들어
        dashboard.set_world_geometry로 준다(Episode 하나, 1.8~5.5ms). 한 번에 하나, 도는 동안 온 청은 모아 두었다가
        가장 나중 것부터 답한다(타일은 열쇠마다 한 번만 청한다)."""
        if self._world_thread is not None:
            self._world_pending[str(world_key)] = int(generation)
            self._world_pending.move_to_end(str(world_key))
            return
        obs = self.dashboard.observation(int(generation))
        if obs is None or self.spec is None or self._closing:
            return
        try:
            spec = self.spec_for(None, int(generation))
            trial = Trial(int(obs.trial_seed), obs.track)
        except (EnvFileError, ValueError, KeyError):
            return  # 세계를 못 만들면 밀도 타일은 계속 DENS_PREPARING(학습·재생과 무관)
        th = _WorldThread((self.dashboard.history.epoch, str(world_key)), spec, trial, self)
        th.finished_job.connect(self._world_done)
        th.finished.connect(lambda t=th: self._world_thread_finished(t))
        self._threads.append(th)
        self._world_thread = th
        th.start()

    def _world_done(self, tag: Any, result: Any) -> None:
        epoch, world_key = tag
        if isinstance(result, Exception) or epoch != self.dashboard.history.epoch:
            return  # 실패했거나 그동안 기록을 비웠다(다른 실행·되감기)
        self.dashboard.set_world_geometry(world_key, result)

    def _world_thread_finished(self, th: _WorldThread) -> None:
        if self._world_thread is th:
            self._world_thread = None
        self._forget_thread(th)
        while self._world_pending and self._world_thread is None and not self._closing:
            key, gen = self._world_pending.popitem(last=True)
            if self.dashboard.world_geometry(key) is None:
                self._geometry_needed(gen, key)

    @property
    def world_busy(self) -> bool:
        """세계 그림을 만드는 중인가(시험용)."""
        return self._world_thread is not None or bool(self._world_pending)

    def race_generations(self) -> Any:
        """「세대끼리 경주」: 1세대·¼·½·¾·마지막 세대 대표와 지금 챔피언을 나란히 경주 창으로(설계 4.6)."""
        if self.run_dir is None or self.generation < 2:
            QMessageBox.information(self, S.LAB_RACE_GENERATIONS, S.RACE_NEED_GENS)
            return None
        from evolab.app.lab.dashboard import DARK
        from evolab.app.race.model import entries_from_generations
        from evolab.app.race.view import show_race

        champion: tuple[int, np.ndarray] | None = None
        c = self.champion
        if c is not None and c.get("genome") is not None and (_int_value(c.get("generation")) or 0) >= 1:
            champion = (int(c["generation"]), np.asarray(c["genome"], dtype=np.float64))
        entries, warnings = entries_from_generations(self.run_dir, self.generation, champion, DARK.series)
        if len(entries) < 2:
            QMessageBox.information(self, S.LAB_RACE_GENERATIONS, "\n".join(warnings) or S.RACE_NEED_GENS)
            return None
        title = S.RACE_TITLE_GENERATIONS.format(run=self.run_dir.name)
        self._race_dialog = show_race(self, entries, title=title, warnings=warnings)
        return self._race_dialog

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
            config = {**(info.get("config") or {}), **_checkpoint_settings(store, OPENED_RUN_CHECKPOINT_KEYS)}
            msg = {"run_dir": str(run_dir), "run_info": info, "env_text": text, "generation": len(stats),
                   "config": config, "layout": info.get("layout"), "stats": stats,
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
        self.stop_flip()
        self._focus_timer.stop()
        self._closing = True  # 학습 프로세스가 끝나기를 기다리는 동안 온 세대의 장면은 만들지 않는다
        try:
            self._drop_scene_jobs()
            self.client.shutdown(timeout=SHUTDOWN_WAIT_S)
            self._drop_scene_jobs()
            for th in list(self._threads):
                th.requestInterruption()
            for th in list(self._threads):
                th.wait()
            # 주인이 이 화면이 아닌 배경 스레드(보는 재미 읽기·PCA·뇌 연결선, 경주 준비)도 끝날 때까지 기다린다: 주 창이
            # 닫혀도 경주 창은 남아 있고(닫힘 알림을 받지 않음), 프로그램이 끝날 때 돌던 QThread가 지워지면 abort
            from evolab.app.race.view import shutdown_race

            self.dashboard.shutdown()
            shutdown_race(self._race_dialog)
        finally:
            self._closing = False


def _checkpoint_settings(store: Any, keys: Sequence[str]) -> dict[str, Any]:
    """마지막 체크포인트 설정에서 keys 값(있는 것만). 설정을 바꾸면 늘 그 세대 체크포인트를 남긴 뒤 사건 줄을 쓰므로
    마지막 체크포인트가 지금 설정이다. 사건 줄로 되짚지 않는다: 앞 체크포인트에서 이어 가도 사건 줄은 지우지 않는다.

    체크포인트가 없거나 읽을 수 없으면 빈 dict(run.json 값으로 보인다. 깨진 체크포인트는 이어 갈 때 알린다).
    """
    from evolab.runs.store import RunStoreError

    try:
        gens = store.checkpoints()
        if not gens:
            return {}
        _, d = store.load_checkpoint(gens[-1])
    except (RunStoreError, OSError):
        return {}
    config = d.get("config")
    if not isinstance(config, dict):
        return {}
    return {k: config[k] for k in keys if k in config}


def switch_note(rows: Sequence[Mapping[str, Any]], i: int) -> str:
    """i번째 통계 줄 끝에 일어난 학습 방법 교체 알림(명령줄과 같은 문장). 교체가 없는 줄이면 빈 글.

    값은 학습 프로세스가 보낸 줄(숫자)이든 stats.csv에서 읽은 줄(글자)이든 같다. 예선 끝(switch 4)의 통과자는 다음 줄의
    learner다. 다음 줄이 아직 없으면 출전이 끝난 줄(switch 3·4)의 learner_best 가운데 첫 최댓값으로 정한다(엔진의
    선정 규칙: 출전자 검증 최고점, 동점이면 앞 번호). 알림의 점수는 통과자가 출전을 끝냈을 때의 검증 최고점이다.
    """
    row = rows[i]
    code = _int_value(row.get("switch")) or 0
    gen = _int_value(row.get("generation"))
    learner = _int_value(row.get("learner")) or 0
    if code in (SWITCH_RESTART_FRESH, SWITCH_RESTART_CHAMPION):
        where = S.RESTART_FROM_NAMES_SHORT["champion" if code == SWITCH_RESTART_CHAMPION else "fresh"]
        return S.TRAIN_SWITCH_RESTART.format(gen=gen, stall=_int_value(row.get("learner_stall")) or 0, where=where,
                                             n=(_int_value(row.get("restarts")) or 0) + 2)
    if code == SWITCH_QUALIFY_NEXT:
        return S.TRAIN_SWITCH_QUALIFY_NEXT.format(gen=gen, n=learner + 1, score=_score_text(row.get("learner_best")),
                                                  next=learner + 2)
    if code != SWITCH_QUALIFY_PICK:
        return ""
    ends: dict[int, float] = {}  # 출전 번호(learner) → 그 출전이 끝난 줄의 검증 최고점(없으면 -inf)
    for r in rows[: i + 1]:
        if _int_value(r.get("switch")) in (SWITCH_QUALIFY_NEXT, SWITCH_QUALIFY_PICK):
            best = _float_value(r.get("learner_best"))
            ends.setdefault(_int_value(r.get("learner")) or 0, best if best is not None else -math.inf)
    winner = _int_value(rows[i + 1].get("learner")) if i + 1 < len(rows) else None
    if winner is None:
        winner = max(sorted(ends), key=lambda k: ends[k]) if ends else learner  # max는 같은 값이면 앞의 것
    return S.TRAIN_SWITCH_QUALIFY_PICK.format(gen=gen, winner=winner + 1, score=_score_text(ends.get(winner)))


def _int_value(v: Any) -> int | None:
    """통계 값(숫자 또는 글자)을 정수로. 빈칸·없는 열·숫자가 아니면 None."""
    try:
        return int(float(v))
    except (TypeError, ValueError, OverflowError):
        return None


def _float_value(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _score_text(v: Any) -> str:
    x = _float_value(v)
    return f"{x:.2f}" if x is not None and math.isfinite(x) else "-"


def _same(v: Any) -> Any:
    return v


def _or_none(v: Any) -> Any:
    """0(칸의 '없음'·'환경 파일대로')이면 None."""
    return v or None


def _shows(widget: QSpinBox | QDoubleSpinBox, value: float) -> bool:
    """칸이 이 값을 (자릿수 반올림만 하고) 보여 줄 수 있는가.

    범위 밖이면 칸은 끝값으로 잘라 보이고, 아주 작은 값이 반올림되어 최솟값이 되면 '없음' 같은 특수 글자로 보인다.
    그런 값은 칸에 보이는 것과 학습이 쓰는 것이 달라지므로 원래 값을 붙잡지 않는다(칸 값대로 학습한다).
    """
    lo, hi = widget.minimum(), widget.maximum()
    if not lo <= value <= hi:
        return False
    return not (widget.specialValueText() and widget.value() == lo and value != lo)


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

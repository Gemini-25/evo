"""반복 실행: 같은 설정을 시드만 바꿔 N번, 하나씩 차례로 돌린다 (명세: 화면과 조작 > 비교).

- RepeatPlan: 환경 글, 학습 설정(TrainConfig 사전), 반복 횟수, 첫 시드, 워커 수, runs 폴더, 묶음 이름.
  시드는 첫 시드, 첫 시드 + 1, … 이다.
- RepeatRunner: 실행을 하나씩 학습 프로세스(TrainingClient)로 돌린다. 각 실행의 run.json에 같은 묶음 id·이름과
  순번을 {"group": {"id", "name", "index", "count"}}로 넣는다(SessionRequest.info). 관전하지 않고(spectate=False)
  바로 시작한다. 정지 조건에 닿으면(state finished) 학습 프로세스에 끝내라고 하고, 닫힌 뒤 다음 실행을 띄운다.
  취소하면 지금 실행을 깨끗이 멈추고(client.shutdown: 이번 세대를 마치고 체크포인트를 남긴다) 다음 실행은 띄우지
  않는다. 지금 실행이 막 뜨는 중이라 한 세대도 마치지 못했으면 그 빈 실행 폴더는 지운다(묶음에 값이 없는 실행이
  남아 평균 선과 띠를 지우지 않게). 실행이 오류로 끝나거나 평가 오류로 멈추면(되감기 뒤 멈춤) 묶음 전체를 멈추고
  failed로 알린다.
- RepeatDialog: 환경 파일(내장 예제 + 파일 열기), 학습 방법, 개체 수(담금질은 이웃 수), 시험 횟수, 정체하면 새로 시작
  (기본 켬), 반복 횟수(기본 5), 첫 시드, 정지 조건(최대 시험 횟수 기본 30,000 또는 최대 시간), 워커 수, 묶음 이름.
  설정은 새 실행 권장값(TrainConfig.for_new_run) 위에 폼 값이다. 예선이 예산에 비해 크면(예산 이상이거나 절반 넘음) 끈다.
  학습 방법 목록에는 make_learner가 실제로 받는 방법만 보인다(available_methods). 아직 만들지 않은 방법은
  ValueError·NotImplementedError를 내고, 만드는 중인 모듈은 가져오기나 인자에서 다른 예외를 낼 수 있어 모두 뺀다.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from evolab import paths
from evolab.app import strings as S
from evolab.app.lab.client import TrainingClient
from evolab.engine.session import SessionRequest
from evolab.engine.trainer import TrainConfig, make_learner, qualify_budget_problem
from evolab.runs.catalog import delete_run
from evolab.runs.store import RunStore, RunStoreError

METHODS: tuple[str, ...] = ("cmaes", "ga", "sa")
DEFAULT_COUNT = 5
DEFAULT_MAX_EVALUATIONS = 30000
DEFAULT_MAX_MINUTES = 10.0
MAX_COUNT = 100


# ---------------------------------------------------------------------------
# 학습 방법
def default_popsize(method: str) -> int:
    """방법별 기본 개체 수(담금질은 이웃 수). 명세 기본값: CMA-ES 32, 유전 알고리즘 64, 담금질 이웃 8."""
    if method == "ga":
        try:
            from evolab.learners.ga import GAConfig

            return int(GAConfig().popsize)
        except Exception:  # 만드는 중인 모듈
            return 64
    if method == "sa":
        return int(TrainConfig().sa_neighbors)
    return int(TrainConfig().popsize)


def available_methods(layout: Any) -> list[str]:
    """이 유전자 배치로 make_learner가 만들 수 있는 학습 방법 id(METHODS 순서)."""
    out: list[str] = []
    for method in METHODS:
        config = TrainConfig(method=method)
        if method == "sa":
            config.sa_neighbors = default_popsize(method)
        else:
            config.popsize = default_popsize(method)
        try:
            make_learner(config, layout)
        except Exception:  # ValueError, NotImplementedError, 만드는 중인 모듈의 ImportError·TypeError 등
            continue
        out.append(method)
    return out


def size_key(method: str) -> str:
    """개체 수 칸이 바꾸는 TrainConfig 필드. 담금질은 개체 수를 쓰지 않고 이웃 수를 쓴다."""
    return "sa_neighbors" if method == "sa" else "popsize"


def qualify_too_big(config: TrainConfig) -> int | None:
    """반복 실행에서 끌 예선이면 그 합계 학습 시험(출전 수 × 출전마다 시험), 아니면 None.

    예선이 최대 시험 횟수 이상이거나(qualify_budget_problem) 그 절반을 넘으면 본선에 남는 예산이 너무 적다(설계 11.3).
    """
    k, b, m = config.qualify_entrants, config.qualify_trials, config.max_evaluations
    if k <= 1:
        return None
    if qualify_budget_problem(config) is not None or (m is not None and k * b > m / 2):
        return k * b
    return None


def default_group_name(env_name: str, method: str, first_seed: int, count: int) -> str:
    m = S.METHOD_NAMES.get(method, method)
    if count <= 1:
        return S.CMP_GROUP_NAME_ONE.format(env=env_name, method=m, seed=first_seed)
    return S.CMP_GROUP_NAME.format(env=env_name, method=m, first=first_seed, last=first_seed + count - 1)


# ---------------------------------------------------------------------------
# 계획
@dataclass
class RepeatPlan:
    """반복 실행 한 묶음. config는 TrainConfig.to_dict() 형식이고, run_seed는 실행마다 바꿔 넣는다."""

    env_text: str
    config: dict[str, Any]
    count: int = DEFAULT_COUNT
    base_seed: int = 1
    workers: int | None = None
    """None이면 학습 프로세스의 기본값(물리 코어 - 1)."""
    runs_dir: Path | None = None
    """None이면 paths.runs_dir()."""
    group_name: str = ""
    """비우면 '환경 · 방법 · 시드 a–b'."""
    group_id: str = ""
    """비우면 RepeatRunner가 새로 만든다."""
    env_name: str = ""
    extra_info: dict[str, Any] = field(default_factory=dict)
    """run.json에 더 넣을 것(메모 등). "group"은 RepeatRunner가 채운다."""

    def seeds(self) -> list[int]:
        return [self.base_seed + i for i in range(self.count)]

    def config_for(self, index: int) -> dict[str, Any]:
        config = dict(self.config)
        config["run_seed"] = self.base_seed + index
        return config


# ---------------------------------------------------------------------------
# 차례로 돌리기
IDLE = "idle"
RUNNING = "running"
DONE = "done"
CANCELLED = "cancelled"
FAILED = "failed"


class RepeatRunner(QObject):
    """RepeatPlan의 실행들을 하나씩 차례로 돌린다. start()로 시작하고, cancel()로 멈춘다."""

    run_started = Signal(int, Path)
    """(순번 0부터, 실행 폴더). 실행 폴더가 막 만들어졌다."""
    progress = Signal(int, int, float)
    """(순번, 끝낸 세대 수, 경과 시간 초). 세대마다. 자세한 통계 줄은 last_row."""
    run_finished = Signal(int, Path)
    """(순번, 실행 폴더). 정지 조건에 닿아 끝났다."""
    failed = Signal(str)
    """한국어 메시지. 묶음은 여기서 멈춘다(뒤이어 finished(False))."""
    finished = Signal(bool)
    """True: N개를 모두 끝냈다. False: 취소했거나 실패했다."""

    def __init__(
        self,
        plan: RepeatPlan,
        parent: QObject | None = None,
        client_factory: Callable[[QObject], TrainingClient] = TrainingClient,
    ) -> None:
        super().__init__(parent)
        if plan.count < 1:
            raise ValueError("count must be >= 1")
        self.plan = plan
        self._factory = client_factory
        name = plan.group_name or default_group_name(
            plan.env_name, str(plan.config.get("method", "cmaes")), plan.base_seed, plan.count
        )
        self.group: dict[str, Any] = {"id": plan.group_id or uuid.uuid4().hex[:12], "name": name, "count": plan.count}
        self.index = -1
        """지금(또는 마지막으로) 돌린 실행의 순번."""
        self.run_dirs: list[Path] = []
        """만들어진 실행 폴더(순번 순서)."""
        self.last_row: dict[str, Any] = {}
        """마지막 세대의 통계 줄(지금 실행)."""
        self.error: str | None = None
        self._state = IDLE
        self._client: TrainingClient | None = None
        self._run_done = False
        self._failure: str | None = None
        self._warning: str | None = None

    # -- 상태 -----------------------------------------------------------------
    @property
    def state(self) -> str:
        """idle, running, done, cancelled, failed."""
        return self._state

    @property
    def running(self) -> bool:
        return self._state == RUNNING

    @property
    def client(self) -> TrainingClient | None:
        """지금 실행의 학습 프로세스 연결(없으면 None)."""
        return self._client

    def runs_dir(self) -> Path:
        return Path(self.plan.runs_dir) if self.plan.runs_dir is not None else paths.runs_dir()

    def fraction(self) -> float:
        """묶음 전체 진행률 0..1(정지 조건이 최대 시험 횟수나 최대 시간일 때 실행 안의 진행도 넣는다)."""
        if self._state == DONE:
            return 1.0
        if self.index < 0:
            return 0.0
        inner = 0.0
        c, row = self.plan.config, self.last_row
        try:
            if c.get("max_evaluations"):
                inner = float(row.get("evaluations", 0)) / float(c["max_evaluations"])
            elif c.get("max_time"):
                inner = float(row.get("elapsed", 0)) / float(c["max_time"])
            elif c.get("max_generations"):
                inner = float(row.get("generation", 0)) / float(c["max_generations"])
        except (TypeError, ValueError, ZeroDivisionError):
            inner = 0.0
        if not math.isfinite(inner):
            inner = 0.0
        return min(1.0, (self.index + min(1.0, max(0.0, inner))) / self.plan.count)

    # -- 조작 -----------------------------------------------------------------
    def start(self) -> None:
        if self._state != IDLE:
            raise RuntimeError("repeat runner already started")
        self._state = RUNNING
        self._start_next()

    def cancel(self) -> None:
        """지금 실행을 깨끗이 멈추고(이번 세대를 마치고 체크포인트를 남긴다) 다음 실행은 띄우지 않는다.

        지금 실행이 한 세대도 마치지 못하고 멈췄으면(다음 실행이 막 뜨는 중에 취소) 그 빈 폴더를 지운다.
        """
        if self._state != RUNNING:
            return
        self._state = CANCELLED
        client = self._client
        self._client = None
        if client is not None:
            client.shutdown()  # 끝날 때까지 기다린다: 그동안 온 'started'로 run_dirs에 지금 실행 폴더가 들어온다
            self._drop(client)
            self._remove_empty_current()
        self.finished.emit(False)

    def _remove_empty_current(self) -> None:
        """취소로 한 세대도 남기지 못한 지금 실행의 폴더를 지우고 run_dirs에서 뺀다. 지우지 못하면 그대로 둔다."""
        if not 0 <= self.index < len(self.run_dirs):
            return
        run = self.run_dirs[self.index]
        try:
            store = RunStore.open(run)
            if store.read_stats() or store.checkpoints():
                return
            delete_run(run, self.runs_dir())
        except (RunStoreError, OSError):
            return
        del self.run_dirs[self.index]

    def shutdown(self) -> None:
        """프로그램을 닫을 때."""
        self.cancel()

    # -- 실행 하나 ------------------------------------------------------------
    def _start_next(self) -> None:
        if self._state != RUNNING:
            return
        self.index += 1
        if self.index >= self.plan.count:
            self._state = DONE
            self.finished.emit(True)
            return
        self._run_done = False
        self._failure = None
        self._warning = None
        self.last_row = {}
        client = self._factory(self)
        client.started.connect(self._on_started)
        client.generation.connect(self._on_generation)
        client.state_changed.connect(self._on_state)
        client.failed.connect(self._on_failed)
        client.closed.connect(self._on_closed)
        self._client = client
        info = {**self.plan.extra_info, "group": {**self.group, "index": self.index}}
        req = SessionRequest(
            kind="new",
            env_text=self.plan.env_text,
            config=self.plan.config_for(self.index),
            runs_dir=str(self.runs_dir()),
            workers=self.plan.workers,
            spectate=False,
            start_running=True,
            info=info,
        )
        try:
            client.start(req)
        except Exception as exc:  # 프로세스를 못 띄움
            self._client = None
            self._drop(client)
            self._fail(S.CMP_REPEAT_START_FAILED.format(reason=str(exc) or type(exc).__name__))

    def _is_current(self) -> bool:
        return self.sender() is self._client and self._client is not None

    def _on_started(self, msg: dict[str, Any]) -> None:
        if self.sender() is not self._client and self._state == RUNNING:
            return
        run_dir = Path(str(msg.get("run_dir", "")))
        self.run_dirs.append(run_dir)
        self.run_started.emit(self.index, run_dir)

    def _on_generation(self, msg: dict[str, Any]) -> None:
        if not self._is_current():
            return
        row = msg.get("row") or {}
        self.last_row = dict(row)
        try:
            elapsed = float(row.get("elapsed", math.nan))
        except (TypeError, ValueError):
            elapsed = math.nan
        self.progress.emit(self.index, int(msg.get("generation", 0)), elapsed)

    def _on_state(self, msg: dict[str, Any]) -> None:
        if not self._is_current() or self._client is None:
            return
        state = msg.get("state")
        if state == "finished":
            self._run_done = True
            self._client.send({"type": "quit"})  # 닫히면(closed) 다음 실행
        elif state == "paused":
            # 반복 실행은 멈추라고 하지 않는다: 평가 오류로 마지막 체크포인트에 되감고 멈춘 것
            self._failure = self._warning or S.CMP_REPEAT_PAUSED
            self._client.send({"type": "quit"})

    def _on_failed(self, message: str, fatal: bool) -> None:
        if self.sender() is not self._client and self._client is not None:
            return
        if fatal:
            self._failure = message
        else:
            self._warning = message

    def _on_closed(self) -> None:
        if self.sender() is not self._client:
            return
        # 학습 프로세스가 갑자기 끝나면 TrainingClient는 closed 바로 뒤에 failed를 보낸다: 그것까지 받고 판단한다
        QTimer.singleShot(0, self._after_close)

    def _after_close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            self._drop(client)
        if self._state != RUNNING:
            return
        if self._failure is not None or not self._run_done:
            self._fail(self._failure or S.CMP_REPEAT_LOST)
            return
        if self.run_dirs and len(self.run_dirs) > self.index:
            self.run_finished.emit(self.index, self.run_dirs[self.index])
        self._start_next()

    def _fail(self, message: str) -> None:
        self._state = FAILED
        self.error = message
        self.failed.emit(message)
        self.finished.emit(False)

    @staticmethod
    def _drop(client: TrainingClient) -> None:
        for sig in (client.started, client.generation, client.state_changed, client.failed, client.closed):
            try:
                sig.disconnect()
            except (RuntimeError, TypeError):
                pass
        client.deleteLater()


# ---------------------------------------------------------------------------
# 대화상자
def env_title(path: Path) -> str:
    """환경 파일의 이름(name). 읽을 수 없으면 파일 이름."""
    try:
        return str(json.loads(path.read_text(encoding="utf-8-sig")).get("name") or path.stem)
    except (OSError, ValueError, AttributeError):
        return path.stem


class RepeatDialog(QDialog):
    """반복 실행 설정. exec()가 Accepted면 plan()으로 RepeatPlan을 받는다."""

    def __init__(
        self,
        parent: QWidget | None = None,
        runs_dir: Path | None = None,
        env_path: Path | None = None,
        workers: int | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(S.CMP_REPEAT_TITLE)
        self.setMinimumWidth(440)
        self._runs_dir = runs_dir
        self.env_path: Path | None = None
        self.env_text: str | None = None
        self.env_name = ""
        self.layout_obj: Any = None
        self._methods: list[str] = []

        self.env_combo = QComboBox()
        for p in sorted(paths.builtin_env_dir().glob("*.evoenv")):
            self.env_combo.addItem(S.CMP_REPEAT_ENV_BUILTIN.format(name=env_title(p)), str(p))
        self.env_combo.currentIndexChanged.connect(self._env_chosen)
        self.env_open = QPushButton(S.CMP_REPEAT_ENV_OPEN)
        self.env_open.clicked.connect(self._choose_env)
        self.env_error = QLabel()
        self.env_error.setWordWrap(True)
        self.env_error.hide()

        self.method = QComboBox()
        self.method.currentIndexChanged.connect(self._method_changed)
        self.method_note = QLabel()
        self.method_note.setWordWrap(True)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 4096)
        self.size_label = QLabel(S.CMP_REPEAT_POPSIZE)
        self.trials = QSpinBox()
        self.trials.setRange(0, 1000)
        self.trials.setSpecialValueText(S.CMP_REPEAT_TRIALS_ENV)
        self.restart = QCheckBox(S.CMP_REPEAT_RESTART)  # 새 실행 권장값(자동 재시작)을 쓴다. 끄면 v1.0처럼 한 학습 방법만
        self.restart.setChecked(True)
        self.qualify_note = QLabel()  # 새 실행 권장값에서 예선이 켜졌는데 예산이 작아 끌 때만 보인다
        self.qualify_note.setWordWrap(True)
        self.qualify_note.hide()
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, MAX_COUNT)
        self.count_spin.setValue(DEFAULT_COUNT)
        self.base_seed = QSpinBox()
        self.base_seed.setRange(0, 2**31 - 1 - MAX_COUNT)
        self.base_seed.setValue(1)
        self.seeds_label = QLabel()
        self.count_spin.valueChanged.connect(self._update_seeds)
        self.base_seed.valueChanged.connect(self._update_seeds)

        self.stop_evals = QRadioButton(S.CMP_REPEAT_STOP_EVALS)
        self.stop_time = QRadioButton(S.CMP_REPEAT_STOP_TIME)
        self.stop_evals.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.stop_evals)
        group.addButton(self.stop_time)
        self.max_evaluations = QSpinBox()
        self.max_evaluations.setRange(1, 2**31 - 1)
        self.max_evaluations.setSingleStep(1000)
        self.max_evaluations.setGroupSeparatorShown(True)
        self.max_evaluations.setValue(DEFAULT_MAX_EVALUATIONS)
        self.max_minutes = QDoubleSpinBox()
        self.max_minutes.setRange(0.1, 10**6)
        self.max_minutes.setDecimals(1)
        self.max_minutes.setValue(DEFAULT_MAX_MINUTES)
        self.stop_evals.toggled.connect(self._update_stop)
        self.max_evaluations.valueChanged.connect(lambda _v: self._update_qualify_note())

        from evolab.engine.pool import default_workers

        self.workers = QSpinBox()
        self.workers.setRange(1, 256)
        self.workers.setValue(workers if workers is not None else default_workers())
        self.group_name = QLineEdit()
        self.group_name.setToolTip(S.CMP_REPEAT_NAME_TIP)

        form = QFormLayout()
        env_row = QHBoxLayout()
        env_row.addWidget(self.env_combo, 1)
        env_row.addWidget(self.env_open)
        form.addRow(S.CMP_REPEAT_ENV, env_row)
        form.addRow("", self.env_error)
        form.addRow(S.CMP_REPEAT_METHOD, self.method)
        form.addRow("", self.method_note)
        form.addRow(self.size_label, self.size_spin)
        form.addRow(S.CMP_REPEAT_TRIALS, self.trials)
        form.addRow("", self.restart)
        form.addRow(S.CMP_REPEAT_COUNT, self.count_spin)
        form.addRow(S.CMP_REPEAT_BASE_SEED, self.base_seed)
        form.addRow("", self.seeds_label)
        stop = QGridLayout()
        stop.addWidget(self.stop_evals, 0, 0)
        stop.addWidget(self.max_evaluations, 0, 1)
        stop.addWidget(self.stop_time, 1, 0)
        stop.addWidget(self.max_minutes, 1, 1)
        form.addRow(S.CMP_REPEAT_STOP, stop)
        form.addRow("", self.qualify_note)
        form.addRow(S.CMP_REPEAT_WORKERS, self.workers)
        form.addRow(S.CMP_REPEAT_NAME, self.group_name)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setText(S.CMP_REPEAT_START)
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(S.CMP_REPEAT_CANCEL_DIALOG)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        note = QLabel(S.CMP_REPEAT_NOTE)
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addLayout(form)
        lay.addWidget(self.buttons)

        self._update_stop()
        if env_path is not None:
            self.set_env_file(Path(env_path))
        elif self.env_combo.count():
            default = next(
                (i for i in range(self.env_combo.count()) if Path(str(self.env_combo.itemData(i))).stem == "lander"), 0
            )
            self.env_combo.setCurrentIndex(default)
            self._env_chosen(default)
        else:
            self._load_env(None)
        self._update_seeds()

    # -- 환경 -----------------------------------------------------------------
    def _env_chosen(self, index: int) -> None:
        data = self.env_combo.itemData(index)
        self._load_env(Path(str(data)) if data else None)

    def _choose_env(self) -> None:
        start = str(self.env_path.parent if self.env_path else paths.builtin_env_dir())
        path, _ = QFileDialog.getOpenFileName(self, S.CMP_REPEAT_ENV_OPEN, start, S.ENV_FILE_FILTER)
        if path:
            self.set_env_file(Path(path))

    def set_env_file(self, path: Path) -> bool:
        """환경 파일을 고른다(목록에 없으면 더한다). 읽을 수 없으면 False."""
        idx = self.env_combo.findData(str(path))
        if idx < 0:
            self.env_combo.addItem(S.CMP_REPEAT_ENV_FILE.format(name=env_title(path), file=path.name), str(path))
            idx = self.env_combo.count() - 1
        if self.env_combo.currentIndex() != idx:
            self.env_combo.setCurrentIndex(idx)  # _env_chosen이 읽는다
        else:
            self._load_env(path)
        return self.env_text is not None

    def _load_env(self, path: Path | None) -> None:
        from evolab.policy.genome import layout_for
        from evolab.spec.io import compile_env

        self.env_path, self.env_text, self.env_name, self.layout_obj = path, None, "", None
        error = ""
        if path is None:
            error = S.CMP_REPEAT_NO_ENV
        else:
            try:
                text = path.read_bytes().decode("utf-8-sig")
                spec = compile_env(text)
                self.layout_obj = layout_for(spec)
                self.env_text, self.env_name = text, spec.file.name
            except Exception as exc:  # 파일 오류(EnvFileError 등)
                error = S.CMP_REPEAT_ENV_FAILED.format(path=path, reason=str(exc) or type(exc).__name__)
        self.env_error.setText(error)
        self.env_error.setVisible(bool(error))
        self._fill_methods()
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok) if hasattr(self, "buttons") else None
        if ok is not None:
            ok.setEnabled(self.env_text is not None and bool(self._methods))

    def _fill_methods(self) -> None:
        keep = self.method.currentData()
        self._methods = available_methods(self.layout_obj) if self.layout_obj is not None else []
        self.method.blockSignals(True)
        self.method.clear()
        for m in self._methods:
            self.method.addItem(S.METHOD_NAMES.get(m, m), m)
        idx = self.method.findData(keep)
        self.method.setCurrentIndex(max(idx, 0))
        self.method.blockSignals(False)
        missing = [S.METHOD_NAMES.get(m, m) for m in METHODS if m not in self._methods]
        self.method_note.setText(S.CMP_REPEAT_METHODS_MISSING.format(names=", ".join(missing)) if missing else "")
        self.method_note.setVisible(bool(missing) and self.layout_obj is not None)
        self._method_changed(self.method.currentIndex())

    def methods(self) -> list[str]:
        """고를 수 있는 학습 방법 id."""
        return list(self._methods)

    def set_method(self, method: str) -> bool:
        idx = self.method.findData(method)
        if idx < 0:
            return False
        self.method.setCurrentIndex(idx)
        return True

    def _method_changed(self, _index: int) -> None:
        m = str(self.method.currentData() or "cmaes")
        self.size_label.setText(S.CMP_REPEAT_NEIGHBORS if m == "sa" else S.CMP_REPEAT_POPSIZE)
        self.size_spin.setValue(default_popsize(m))
        self._update_seeds()

    def _update_seeds(self) -> None:
        seeds = [self.base_seed.value() + i for i in range(self.count_spin.value())]
        shown = ", ".join(str(s) for s in seeds[:6]) + (", …" if len(seeds) > 6 else "")
        self.seeds_label.setText(S.CMP_REPEAT_SEEDS.format(seeds=shown))
        m = str(self.method.currentData() or "cmaes")
        self.group_name.setPlaceholderText(default_group_name(self.env_name, m, self.base_seed.value(), self.count_spin.value()))

    def _update_stop(self) -> None:
        evals = self.stop_evals.isChecked()
        self.max_evaluations.setEnabled(evals)
        self.max_minutes.setEnabled(not evals)
        self._update_qualify_note()

    def _stop_config(self, c: TrainConfig) -> None:
        """정지 조건(최대 시험 횟수 또는 최대 시간)을 넣고, 예산에 비해 큰 예선은 끈다."""
        if self.stop_evals.isChecked():
            c.max_evaluations = self.max_evaluations.value()
        else:
            c.max_time = self.max_minutes.value() * 60.0
        if qualify_too_big(c) is not None:
            c.qualify_entrants = 1

    def _update_qualify_note(self) -> None:
        """예선을 끄게 되면 그렇다고 알린다(새 실행 권장값에서 예선이 꺼져 있으면 늘 숨김)."""
        c = TrainConfig.for_new_run()
        if self.stop_evals.isChecked():
            c.max_evaluations = self.max_evaluations.value()
        total = qualify_too_big(c)
        self.qualify_note.setText(S.CMP_REPEAT_QUALIFY_OFF_HINT.format(total=total) if total is not None else "")
        self.qualify_note.setVisible(total is not None)

    # -- 결과 -----------------------------------------------------------------
    def config(self) -> TrainConfig:
        """폼의 학습 설정(run_seed는 첫 시드). 값이 잘못되면 TrainerChangeError(ValueError, 한국어).

        바탕은 새 실행 권장값(자동 재시작 등)이고, '정체하면 새로 시작'을 끄면 restart_patience = None(끔)이다.
        """
        m = str(self.method.currentData() or "cmaes")
        c = TrainConfig.for_new_run(method=m, run_seed=self.base_seed.value())
        setattr(c, size_key(m), self.size_spin.value())
        c.trials = self.trials.value() or None
        if not self.restart.isChecked():
            c.restart_patience = None
        self._stop_config(c)
        c.validate()
        return c

    def plan(self) -> RepeatPlan:
        """폼으로 만든 RepeatPlan. 환경이 없거나 설정이 잘못되면 ValueError(한국어)."""
        if self.env_text is None:
            raise ValueError(self.env_error.text() or S.CMP_REPEAT_NO_ENV)
        config = self.config()
        return RepeatPlan(
            env_text=self.env_text,
            config=config.to_dict(),
            count=self.count_spin.value(),
            base_seed=self.base_seed.value(),
            workers=self.workers.value(),
            runs_dir=self._runs_dir,
            group_name=self.group_name.text().strip(),
            env_name=self.env_name,
        )

    def accept(self) -> None:
        try:
            self.plan()
        except ValueError as exc:
            self.show_error(str(exc))
            return
        super().accept()

    def show_error(self, message: str) -> None:
        """테스트는 이 메서드를 바꿔 끼운다."""
        QMessageBox.warning(self, S.CMP_REPEAT_TITLE, message)

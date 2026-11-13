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

v6.0 관전 방식(v6 설계 4.12·5.1·5.2, D23~D25):
- 「보기」 묶음의 「관전 방식」 칸(pace_combo): 「학습 먼저」(PACE_SKIP, 기본 PACE_DEFAULT)와 「세대마다 다 보기」(PACE_EVERY,
  v3.1까지의 동작). 설정 lab/pace. 학습 프로세스에는 시작 요청(SessionRequest.pace)과 mode 명령의 "pace"로 알린다.
- 「학습 먼저」: 재생 대기열은 가장 새 세대 하나만. 새 세대가 오면 기다리던(장면을 만드는 중이거나 만들어 둔) 앞 세대는 버리고
  그 세대 번호로 played를 바로 보낸다(학습 프로세스의 acked와 「보는 재미」 따라가기를 지킴). 재생 중이 아닐 때 만들던 장면은
  끝까지 만든다(세대가 장면 만들기보다 빨리 오면 화면이 영영 멈춰 있지 않게). 새 세대가 기다리면 지금 재생은 대표 시험 0번
  궤적 끝 + 0.5초와 실제 시간 SKIP_MAX_VIEW초 가운데 이른 때에 끝낸다(기다리는 세대가 없으면 가장 긴 궤적까지). HUD 「학습은
  n세대까지」. 통계·관찰 기록은 지금처럼 세대마다 모두 대시보드로 간다.
- 「세대마다 다 보기」에서 최근 WAIT_HINT_WINDOW세대 동안 재생을 기다린 몫(세대 메시지 "waited")이 WAIT_HINT_RATIO를 넘으면 실행마다
  한 번 상태 글 끝에 LAB_PACE_WAIT_HINT.
- 기본이 「학습 먼저」이고 예전 설정(lab/mode가 있고 lab/pace가 없음)이면 처음 한 번 상태 글에 LAB_PACE_CHANGED_NOTICE
  (lab/pace_notice = 1). 보기 칸을 저장할 때 lab/pace도 함께 쓰므로 v6에서 처음 쓴 사용자에게는 뜨지 않는다.
- 화면에만 쓰는 일이라 학습 결과와 무관하다(T20: 세 방식의 통계·챔피언·체크포인트가 바이트 같음).

v6.1 자라는 뇌(v6 설계 4.17·5.1·5.2, D36):
- 「학습 설정」 묶음의 「학습 방법」 바로 아래 「뇌」 줄(brain_combo, CMA-ES일 때만 보임): 「크기 고정」(기본) / 「작게 시작해
  자라기」. 「자라기」면 실행을 학습 방법 "grow"로 만들고(TrainConfig.grow_start·grow_every) 「처음 뉴런 수」(1..가장 작은
  은닉층)·「자라는 간격(세대)」(1..GROW_EVERY_MAX) 두 칸과 회색 안내 두 줄(LAB_GROW_HINT 또는 LAB_GROW_HINT_ALREADY,
  LAB_GROW_HINT_HONEST)을 보이고 IPOP 체크를 숨긴다(함께 쓸 수 없음). 은닉층이 없는 환경·뇌 없는 환경은 「자라기」 항목을
  끄고 까닭을 풍선(LAB_BRAIN_GROW_NO_HIDDEN/NO_BRAIN)으로. 학습 중·이어 하는 실행에서는 학습 방법과 같이 잠근다.
  사용자가 고른 값은 설정 lab/brain·lab/grow_start·lab/grow_every(실행을 열어 칸을 채운 값은 쓰지 않음).
- 세대 메시지의 "growth"(자란 세대만) → 상태 글 끝줄 LAB_STATUS_GROWN 한 줄(다음 자람이 바꾼다, 막는 상자 없음).
- 안내 줄의 뉴런 수는 은닉층 합(「뇌 크기」 타일·뇌 그림 제목·상태 글과 같은 수). 은닉층이 하나인 내장 예제는 그 층 그대로다.

v2.1 몸 구조(설계 4.13·5.3·D41):
- 「몸 구조」 타일(app/lab/structure_tile.py)에 실행의 환경 버전 목록을 준다(세대마다 그 세대를 계산한 환경으로 구조를
  센다). 학습 중 환경을 바꾸면 그 버전을 더한다.
- 클릭 검사 패널: 그 개체에서 없는 물체·관절에 단 센서 채널 옆에 「(없음)」(INSPECT_ABSENT).
- 구조 환경에서 구조 뒤집기가 꺼진 유전 알고리즘·담금질 실행(v2.0에서 시작한 실행 등)을 이어 가면 LAB_GA_NO_STRUCTURE를
  실행마다 한 번 알린다. 새 실행의 기본 설정은 TrainConfig.for_new_run(환경)(있음 유전자가 있으면 구조 뒤집기 0.05).
"""

from __future__ import annotations

import collections
import dataclasses
import gc
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any, Final

import numpy as np
from PySide6.QtCore import QElapsedTimer, QFileSystemWatcher, QObject, QSettings, Qt, QThread, QTimer, Signal
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
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from evolab import paths
from evolab.app import strings as S
from evolab.app import style
from evolab.app.errors import os_cause, show_error, training_error
from evolab.app.lab.canvas import SpectatorCanvas
from evolab.app.lab.client import TrainingClient
from evolab.app.lab.dashboard import Dashboard
from evolab.app.lab.history import GUI_PRIORITY
from evolab.app.lab.watch_types import BrainSource, WatchMeta
from evolab.engine import seeds
from evolab.engine.pool import default_workers
from evolab.engine.replay import TRACE_BRAIN, Replay, TraceMode, agent_geoms, pausing, replay
from evolab.engine.session import PACE_EVERY, PACE_SKIP, PACES, SessionRequest
from evolab.engine.trainer import (
    MAX_QUALIFY_ENTRANTS,
    NEW_RUN_DEFAULTS,
    PRESET_TRAIN_KEYS,
    RECOMMENDED_RESTART_PATIENCE,
    RESTART_FROM,
    SWITCH_QUALIFY_NEXT,
    SWITCH_QUALIFY_PICK,
    SWITCH_RESTART_CHAMPION,
    SWITCH_RESTART_FRESH,
    TrainConfig,
    generation_trials,
    qualify_budget_problem,
    structure_note,
)
from evolab.policy.genome import GenomeLayout, layout_for
from evolab.spec import model as m
from evolab.spec import structure as st
from evolab.spec.compiled import EnvFileError, EnvSpec
from evolab.spec.editor_meta import editor_train, view_box
from evolab.spec.io import compile_env, env_hash, parse_env
from evolab.world.physics2d.episode import Episode, EpisodeBuildError, Trial

SPEEDS = (1, 2, 4, 8)
COLOR_OLD = QColor(80, 140, 255)
COLOR_NOW = QColor(255, 120, 40)
MARK_KINDS = ("resume", "env_change", "config_change", "branch")
SHUTDOWN_WAIT_S = 120.0
"""창을 닫을 때 이번 세대를 마치고 체크포인트를 남길 때까지 기다리는 시간(큰 개체 수는 세대가 길다)."""
SHOWN_RESTART_PATIENCE = int(NEW_RUN_DEFAULTS["restart_patience"] or RECOMMENDED_RESTART_PATIENCE)
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
PACE_DEFAULT: Final = PACE_SKIP
"""관전 방식 기본값(v6.0 D24): 설정 lab/pace가 없을 때. 사용자 결정(G1·Q1)으로 「학습 먼저」. 되돌릴 지점 5(입력 지연 M7이나 v3 안내
시험이 깨짐)면 PACE_EVERY로 낸다."""
SKIP_MAX_VIEW: Final = 15.0
"""「학습 먼저」에서 새 세대가 기다릴 때 지금 세대를 실제 시간으로 이만큼(초)까지만 본다(v6 설계 4.12, M9에서 다시 정함)."""
SKIP_TAIL_S: Final = 0.5
"""「학습 먼저」에서 새 세대가 기다릴 때: 대표 시험 0번 궤적이 끝나고 이만큼(재생 시각, 초) 뒤에 다음 세대로."""
SKIP_SCENE_LEAD_S: Final = 0.8
"""「학습 먼저」·몸 유전자 환경: 관전 재생 중에는 다음 세대의 장면을 지금 재생이 끝나기 이만큼(실제 시간, 초) 전에야 만들기
시작한다(마지막으로 다 만든 장면의 실제 시간 두 배가 더 길면 그만큼). 세대가 올 때마다 만들면 재생 하나(최대 SKIP_MAX_VIEW초)
동안 온 세대 대부분의 장면을 만들고 버렸다(바퀴 차 1배 45초: 장면 21번 중 보인 것 3번, 화면 프로세스 CPU 약 5.6초, v6.0 검토).
화면에만 쓰는 일이라 학습 결과와 무관하다."""
WAIT_HINT_WINDOW: Final = 10
"""기다림 알림(D25): 최근 이만큼의 세대로 본다."""
WAIT_HINT_RATIO: Final = 0.5
"""기다림 알림(D25): 그 세대들의 (기다린 시간 합) ÷ (계산 + 기다린 시간 합)이 이보다 크면 알린다."""
BRAIN_FIXED: Final = "fixed"
BRAIN_GROW: Final = "grow"
BRAINS: Final = (BRAIN_FIXED, BRAIN_GROW)
"""「뇌」 칸 값(v6.1, 설정 lab/brain). 「자라기」 + CMA-ES = 학습 방법 GROW_METHOD."""
GROW_METHOD: Final = "grow"
"""자라는 뇌의 학습 방법 id(TrainConfig.method, CMA-ES만, v6 설계 D31·D32)."""
GROW_START_DEFAULT: Final = 2
GROW_EVERY_DEFAULT: Final = 10
GROW_EVERY_MAX: Final = 100
GROW_START_MAX: Final = 4096
"""환경이 없을 때 「처음 뉴런 수」 칸의 최댓값(환경이 있으면 가장 작은 은닉층 크기)."""
ReplayJob = tuple[EnvSpec, GenomeLayout, np.ndarray, Trial, TraceMode]
"""다시 시뮬레이션 작업 하나: (환경, 유전자 배치, 유전자, 시험, 추적 종류 False | True | "brain")."""


def settings_store() -> QSettings:
    return QSettings(str(paths.user_state_dir() / "settings.ini"), QSettings.Format.IniFormat)


# ---------------------------------------------------------------------------
# 백그라운드 다시 시뮬레이션
# ---------------------------------------------------------------------------


def gui_first(stop: Callable[[], bool]) -> bool:
    """화면 프로세스의 배경 작업(다시 시뮬레이션, 장면·세계 만들기)이 일 하나를 시작하기 전에 부른다: 화면 스레드가 무거운
    일을 하는 중이면(GUI_PRIORITY: 펌프 한 단계, 실행 열기, v3.0부터 창 다시 그리기(dashboard._PaintGuard)) 끝날 때까지
    기다린다. pymunk는 C 함수마다 GIL을 놓았다 곧바로 다시 잡아, 같이 돌면 화면 스레드가 GIL을 얻지 못하고 굶는다(궤적
    밀도 세계 만들기와 겹친 타일 그림이 8ms → 25ms). stop()(멈추라는 요청)을 돌려준다."""
    GUI_PRIORITY.wait(stop)
    return stop()


def release_episodes(stop: Callable[[], bool]) -> None:
    """배경 작업이 다 쓴 Episode를 그 배경 스레드에서 바로 거둔다(v3.0 ML 고침). 다시 시뮬레이션·세계 그림이 끝난 뒤 부른다.

    Episode(pymunk 공간·모양)는 순환 참조라 참조 수로 풀리지 않고 다음 순환 GC가 푼다. 다시 시뮬레이션은 수천 스텝을 도는
    동안 Episode가 늙은 세대로 넘어가 전체 수집이어야 풀린다. 그 수집이 화면 스레드에서 타일 그림 한가운데 일어나면 그림에
    수십~수백 ms가 더해졌다(바퀴 고르는 언덕차 8배 관전, 2세대가 올 때: 공간 하나 푸는 데 8~10ms, 재생 뒤 전체 수집
    22~196ms가 유전자 지도 첫 그림과 겹쳐 최대 틈 113~241ms). 화면이 바쁘지 않을 때(gui_first) 만든 스레드에서 거두면 그
    일이 그림과 겹치지 않고, 그동안에도 화면 스레드는 pymunk가 GIL을 놓는 틈마다 돈다. 오래 사는 객체는 얼려 두었으므로
    (freeze_long_lived) 전체 수집이 훑는 것은 그 뒤에 만든 것뿐이다. 화면에만 쓰는 일이라 학습 결과와 무관하다."""
    gui_first(stop)
    with GUI_PRIORITY.working():  # 거두는 동안은 GIL을 쥔다: 화면이 무거운 일을 시작하면 이것이 끝나기를 잠깐 기다린다(v6.0)
        gc.collect()


class _ReplayThread(QThread):
    finished_job = Signal(object, object)
    """(꼬리표, Replay 목록 또는 예외)"""

    def __init__(self, tag: Any, jobs: list[ReplayJob], parent: QObject | None = None, yield_steps: bool = False) -> None:
        super().__init__(parent)
        self.tag = tag
        self.jobs = jobs
        self.yield_steps = yield_steps
        """참이면 화면 스레드가 바쁜 동안 스텝 사이에서도 쉰다(뇌 추적 배경 작업). 거짓이면 v3.0처럼 작업마다 한 번만(gui_first)."""

    def run(self) -> None:
        stop = self.isInterruptionRequested
        # v6.0 협조 양보(v3.0 남은 문제): 뇌 추적 배경 작업(yield_steps)은 재생 하나(바퀴 차 30초 뇌 추적 140ms 안팎) 동안에도 화면
        # 스레드가 바쁘면 스텝 사이에서 쉰다. 사용자가 누른 재생(클릭 검사·타임라인·1세대 vs 지금·고속 챔피언)은 쉬지 않는다: 스텝마다
        # 쉬면 학습 8배 관전 중 클릭 검사가 두 배 넘게 늦었다(중앙값 0.53 → 1.16초, 최대 2.2초, v6.0 검토). 첫 스텝 앞의 Episode
        # 만들기는 어느 쪽이든 쉴 수 없는 일 하나로 센다(GUI_PRIORITY.working: 다른 배경 일과 번갈아, 화면이 바빠지면 그것이 끝나기를
        # 잠깐 기다림). engine/replay.py pausing. 쉬는 자리만 바뀔 뿐 결과는 같다.
        pause = (lambda: GUI_PRIORITY.wait(stop)) if self.yield_steps else None
        try:
            out = []
            with pausing(pause, unit=lambda: GUI_PRIORITY.working(stop)):
                for spec, layout, genome, trial, trace in self.jobs:
                    gui_first(stop)
                    out.append(replay(spec, layout, genome, trial, trace=trace))
                    release_episodes(stop)
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

    v2.0 개체 조립 실패(설계 4.9): 그 몸 수치로 조립할 수 없는 개체(EpisodeBuildError)는 빈 모양 목록([])으로 둔다(그
    개체의 궤적은 프레임이 없어 그리지 않는다). 세계 모양은 조립된 첫 개체 것이고, 모두 실패하면 처음 몸으로 만든다.
    빈 목록의 수는 build_failures로 센다.
    """
    if not layout.n_body:
        return Episode(spec, None, trial).geometry(), None
    body = layout.body_values(np.asarray(population, dtype=np.float64))
    world: Any = None
    agents: list[Any] = []
    for i in range(len(population)):
        if cancel is not None and cancel():
            raise SceneCancelled
        # 개체 하나(Episode 만들기, 바퀴 세계는 언덕 모양 수천 개)는 도중에 쉴 수 없다: 화면이 그동안 무거운 일을 시작하면
        # 이것이 끝나기를 잠깐 기다린다(GUI_PRIORITY.working, v6.0)
        with GUI_PRIORITY.working():
            try:
                ep = Episode(spec, body[i], trial)
            except EpisodeBuildError:
                agents.append([])
                continue
            geometry = ep.geometry()
            if world is None:
                world = geometry
            agents.append(agent_geoms(geometry, ep.moving_bodies))
            del ep, geometry  # 다음 개체를 만들기 전에 놓는다(위 설명)
        if cancel is not None:
            cancel()  # 거두는 동안(바퀴 고르는 언덕차는 8~10ms) GIL을 쥐므로 화면이 바쁘면 끝날 때까지 기다린 뒤에(v3.0 ML)
        with GUI_PRIORITY.working():
            gc.collect(1)
    if world is None:  # 모든 개체를 조립할 수 없다(학습기는 처음 몸이 조립되는 것을 시작할 때 확인한다)
        world = Episode(spec, layout.body_values(layout.initial()[None, :])[0], trial).geometry()
    return world, agents


def build_failures(agents: Sequence[Sequence[Any]] | None) -> int:
    """build_scene이 조립하지 못한 개체 수(빈 모양 목록). 몸 유전자가 없는 환경(None)은 0."""
    return 0 if agents is None else sum(1 for a in agents if not a)


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
        self.seconds: float | None = None
        """다 만드는 데 걸린 실제 시간(초, 기다린 시간 포함). 끝까지 만들었을 때만(「학습 먼저」의 SKIP_SCENE_LEAD_S)."""

    def run(self) -> None:
        stop = self.isInterruptionRequested
        t0 = time.perf_counter()
        try:
            gui_first(stop)
            out = build_scene(*self.args, cancel=lambda: gui_first(stop))
        except Exception as exc:  # 결과를 화면 스레드로 넘겨 알린다(멈춘 것도: 꼬리표가 달라 버려진다)
            self.finished_job.emit(self.tag, exc)
            return
        self.seconds = time.perf_counter() - t0
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
        gui_first(self.isInterruptionRequested)
        try:
            with GUI_PRIORITY.working():  # 세계 하나 만들기는 도중에 쉴 수 없다(v6.0, build_scene과 같음)
                out = Episode(spec, None, trial).geometry()
        except Exception as exc:  # 결과를 화면 스레드로 넘긴다(못 만들면 밀도 타일은 계속 준비 중 글)
            self.finished_job.emit(self.tag, exc)
            return
        release_episodes(self.isInterruptionRequested)
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
        self.absent: frozenset[str] = frozenset()
        """없는 물체·관절에 단 센서의 신경망 입력 채널 이름(v2.1): 그 줄은 값 대신 이름 옆에 「(없음)」."""
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.title.setFont(style.font(self.title, "title"))  # 6.2.0 U11: 「개체 n · 점수」 / 「끝난 까닭」 두 줄
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
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # 6.2.0 U11: 값이 가로 스크롤 뒤에 숨지 않게
        scroll.setWidget(self.body)
        self.setMinimumWidth(INSPECT_MIN_WIDTH)
        lay = QVBoxLayout(self)
        lay.addWidget(self.title)
        lay.addWidget(self.time_label)
        lay.addWidget(scroll, 1)
        lay.addWidget(close)

    def set_replay(self, index: int, rep: Replay, absent: AbstractSet[str] = frozenset()) -> None:
        """absent: 이 개체에서 없는 물체·관절에 단 센서의 입력 채널 이름(absent_inputs, v2.1). 그 채널은 늘 0이다."""
        self.replay = rep
        self.absent = frozenset(absent)
        self.title.setText(S.LAB_INSPECT_TITLE.format(index=index, score=rep.result.score, end=rep.result.end_label))
        self.show_time(0.0)

    def show_time(self, t: float) -> None:
        rep = self.replay
        if rep is None or rep.observations is None or not len(rep.observations):
            return
        dt = rep.trajectory.dt
        k = int(min(len(rep.observations) - 1, max(0, math.floor(t / dt + 1e-9))))
        self.time_label.setText(S.LAB_INSPECT_TIME.format(t=k * dt, total=rep.result.steps * dt))
        acts = rep.activations or []
        if not rep.channels and all(a.shape[1] == 0 for a in acts[-1:]):
            # 뇌 없는 환경(출력 없음, v2.0 설계 D20): 빈 제목 대신 안내(센서가 있으면 센서 값은 보인다)
            sensors = self._sensor_lines(rep, k) if rep.inputs else []
            self.body.setText("\n".join([*sensors, S.DASH_EMPTY_NO_BRAIN]))
            return
        lines = self._sensor_lines(rep, k)
        lines.append(S.LAB_INSPECT_LAYERS)
        for li, a in enumerate(acts):
            last = li == len(acts) - 1
            lines.append(S.LAB_INSPECT_OUTPUTS if last else S.LAB_INSPECT_LAYER.format(k=li + 1, n=a.shape[1]))
            row = a[k]
            names = rep.channels if last and len(rep.channels) == len(row) else [str(j) for j in range(len(row))]
            for name, v in zip(names, row, strict=True):
                lines.append(f"{name:<12} {float(v):+5.2f} {_bar(float(v), INSPECT_BAR)}")
        self.body.setText("\n".join(lines))

    def _sensor_lines(self, rep: Replay, k: int) -> list[str]:
        """센서 제목과 센서마다 (정규화 전 → 신경망 입력) 한 줄, 끝에 빈 줄. 없는 부품의 센서(늘 0)는 값 대신 이름 바로
        옆에 「(없음)」: 좁은 패널(1280×800 창에서 약 13글자)에서도 가로로 밀지 않고 보인다."""
        lines = [S.LAB_INSPECT_SENSORS]
        raw = rep.raw[k] if rep.raw else {}
        wide = self.width() >= INSPECT_RAW_WIDTH  # 6.2.0 U11: 정규화 전 값은 넓을 때만(좁으면 이름·입력값·막대 약 27자)
        for i, name in enumerate(rep.inputs):
            if name in self.absent:
                lines.append(f"{name} {S.INSPECT_ABSENT}")
                continue
            v = float(rep.observations[k, i])
            r = raw.get(name)
            if wide:
                lines.append(f"{name:<12} {'' if r is None else f'{r:9.3f}'} → {v:+5.2f} {_bar(v / 5.0, INSPECT_BAR)}")
            else:
                lines.append(f"{name:<12} {v:+5.2f} {_bar(v / 5.0, INSPECT_BAR)}")
        lines.append("")
        return lines


class _PanelScroll(QScrollArea):
    """실험실 오른쪽 패널의 스크롤 영역: 처음 너비가 내용의 최소 너비보다 좁지 않다(v3.0 윈도우 CI 고침, 설계 D17·T16).

    QScrollArea.sizeHint는 너비를 글자 높이의 36배에서 자르고 세로 스크롤 막대 자리를 넣지 않는다. 스플리터는 패널(늘림 0)에 그
    너비를 주므로, 글자 높이에 비해 글이 넓은 글꼴이면 내용(최소 너비)이 영역보다 넓어 가로 스크롤 막대가 생기고 「멈춤」·「재생
    배속」이 오른쪽에서 잘린다. 창 왼쪽(캔버스)에 자리가 남아도 그랬다. 윈도우 CI의 offscreen 플랫폼(글꼴 폴더가 없어 모든 글자가
    12px 너비 네모)에서 내용 672px · 영역 432px로 잘렸다. 리눅스 기본 글꼴(9pt)은 영역 536 → 550px(세로 막대 자리 14px만큼
    넓어짐, 그만큼 캔버스가 좁아짐). 바라는 너비만 고치고 최소 너비(300)는 그대로다: 사용자는 여전히 끌어서 좁힐 수 있다."""

    def sizeHint(self) -> Any:  # QSize
        hint = super().sizeHint()
        panel = self.widget()
        if panel is not None:
            need = panel.minimumSizeHint().width() + 2 * self.frameWidth() + self.verticalScrollBar().sizeHint().width()
            hint.setWidth(max(hint.width(), need))
        return hint


def absent_inputs(spec: EnvSpec, layout: GenomeLayout, genome: np.ndarray) -> frozenset[str]:
    """그 개체(유전자)에서 없는 물체·관절에 단 센서의 신경망 입력 채널 이름(v2.1 클릭 검사 「(없음)」).

    없는 물체 = 에이전트 물체 − structure.present(몸 수치)로 Episode.absent와 같은 규칙(몸 수치만의 함수, 설계 T57).
    런타임(Episode._sensor_fn)처럼 센서의 body가 없는 물체이거나 관절 각도 센서의 관절이 없는 관절(끝 하나가 없는
    물체)이면 그 센서의 모든 채널이다. 있음 칸을 쓰지 않는 환경이면 빈 집합."""
    env = spec.file
    if all(b.exists is None for b in env.bodies):
        return frozenset()
    g = np.asarray(genome, dtype=np.float64).reshape(-1)
    body = layout.body_values(g) if layout.n_body else None
    absent = frozenset(b.id for b in env.bodies if b.owner == "agent") - st.present(spec, st.param_values(env, body))
    if not absent:
        return frozenset()
    joints = {j.id for j in env.joints if j.a in absent or j.b in absent}
    sensors = {s.id for s in env.sensors
               if getattr(s, "body", None) in absent or (isinstance(s, m.JointAngle) and s.joint in joints)}
    return frozenset(name for name in spec.inputs
                     if (ch := spec.sensor_channels.get(name)) is not None and ch.owner in sensors)


def structure_notice(config: Mapping[str, Any], spec: EnvSpec | None) -> str | None:
    """구조 환경(있음 유전자가 있음)에서 구조 뒤집기가 꺼진 유전 알고리즘·담금질 실행이면 LAB_GA_NO_STRUCTURE, 범위
    가운데가 0.5가 아니라 뒤집기가 건드리지 않는 있음 칸이 있으면 LAB_STRUCTURE_RANGE(둘 다면 두 줄), 아니면 None. 키가
    없는 예전(v2.0) 설정은 데이터클래스 기본값 0(꺼짐)으로 읽는다. 판단은 명령줄과 같은 trainer.structure_note(실행 설정
    사전과 환경으로 부르는 껍데기)."""
    if spec is None:
        return None
    try:
        layout = layout_for(spec)
    except ValueError:  # 유전자 배치를 만들 수 없는 환경(학습이 따로 알린다)
        return None
    return structure_note(TrainConfig.from_dict(dict(config)), layout, spec)


INSPECT_MIN_WIDTH = 280
"""들여다보기 패널의 최소 폭(6.2.0 U11: 1280 창에서 151px이라 값이 가로 스크롤 뒤에 숨었다)."""
INSPECT_RAW_WIDTH = 360
"""패널이 이 폭 이상일 때만 센서 줄에 정규화 전 값도 보인다."""
INSPECT_BAR = 8
"""들여다보기 막대의 반쪽 칸 수."""


def _bar(v: float, width: int = 10) -> str:
    v = max(-1.0, min(1.0, v))
    n = round(abs(v) * width)
    left = " " * (width - n) + "█" * n if v < 0 else " " * width
    right = "█" * n + " " * (width - n) if v >= 0 else " " * width
    return f"[{left}|{right}]"


# ---------------------------------------------------------------------------
# 실험실 화면
# ---------------------------------------------------------------------------


_HASH_CACHE: dict[str, tuple[tuple[int, int], str | None]] = {}
"""환경 파일 경로 → ((mtime_ns, 크기), env_hash 또는 읽을 수 없으면 None). 「내 프리셋」을 다시 채울 때 바뀐 파일만 다시 읽는다."""


def _file_env_hash(path: Path) -> str | None:
    """환경 파일의 env_hash(editor 칸 밖의 내용). 읽거나 해석할 수 없으면 None."""
    try:
        st_ = path.stat()
    except OSError:
        return None
    key = (st_.st_mtime_ns, st_.st_size)
    held = _HASH_CACHE.get(str(path))
    if held is not None and held[0] == key:
        return held[1]
    try:
        value: str | None = env_hash(parse_env(path.read_bytes()))
    except (OSError, EnvFileError, ValueError):
        value = None
    _HASH_CACHE[str(path)] = (key, value)
    return value


def _untouched_copy(path: Path, builtin: Path) -> bool:
    """첫 실행이 복사한(paths.install_examples) 내장 예제의 손대지 않은 사본인가: 바이트가 같거나 env_hash가 같다."""
    try:
        if path.read_bytes() == builtin.read_bytes():
            return True
    except OSError:
        return False
    mine = _file_env_hash(path)
    return mine is not None and mine == _file_env_hash(builtin)


def user_presets(folder: Path | None = None) -> list[Path]:
    """실험실 환경 목록의 「내 프리셋」(설계 3.1 H): 내 환경 폴더의 *.evoenv 가운데, 같은 줄기 이름의 내장 파일과 env_hash가 같은
    것(첫 실행이 복사한 손대지 않은 사본)을 뺀 것. 이름순."""
    folder = paths.user_env_dir() if folder is None else folder
    try:
        files = sorted((p for p in folder.glob("*.evoenv") if p.is_file()), key=lambda p: (p.name.lower(), p.name))
    except OSError:
        return []
    builtin = {p.stem: p for p in paths.builtin_envs()}
    return [p for p in files if not (p.stem in builtin and _untouched_copy(p, builtin[p.stem]))]


def builtin_status(stem: str) -> str:
    """내장 예제의 등급(설계 5.2): 같은 이름의 내장 조리법 파일(assets/recipes/<이름>.json)의 status. 없거나 못 읽으면 "ok".

    presets 꾸러미를 불러오지 않는다(실험실은 창이 뜰 때 만들어진다: 시작 시간)."""
    try:
        data = json.loads((paths.recipe_dir() / f"{stem}.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return "ok"
    status = data.get("status", "ok") if isinstance(data, dict) else "ok"
    return status if isinstance(status, str) else "ok"


class LabView(QWidget):
    run_started = Signal(Path)
    edit_env_requested = Signal(str, str)
    env_changed = Signal()
    """새 실행을 준비할 환경을 바꿨다(use_env_text). 빈 상태 안내가 글을 바로 고친다(v3 검토 수정)."""
    state_changed = Signal(str)
    """(6.2.0 다듬기 3.3) self.state가 지난번에 낸 값과 달라졌을 때만(세대마다 내지 않는다). 세대·점수는 받는 쪽이 읽는다."""
    new_preset_requested = Signal()
    """(6.2.0 설계 3.1 H) 「새 프리셋…」 단추: 주 창이 받아 에디터의 new_preset()을 연다."""
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
        """상태 글 끝줄: 학습을 멈추지 않는 학습 프로세스 알림(관찰 기록 실패, TRAIN_OBSERVE_FAILED_ONWARD). 실행이 바뀌면 지움."""
        self._build_note = ""
        """상태 글 끝줄: 관전 중인 세대에 조립할 수 없는 개체가 있으면 LAB_BUILD_FAILED_N(v2.0 설계 4.9). 실행이 바뀌면 지움."""
        self._grow_note = ""
        """상태 글 끝줄: 마지막으로 뇌가 자란 세대(v6.1 LAB_STATUS_GROWN, 세대 메시지 "growth"). 실행이 바뀌면 지움."""
        self._brain_quiet = False
        """「뇌」 줄 칸을 코드가 채우는 중(실행 설정·환경·설정 읽기): 사용자가 고른 값으로 설정에 쓰지 않는다."""

        # 재생 상태
        self.view_mode = "live"  # live(관전), fast(고속 챔피언), replay(타임라인·겹쳐 보기)
        self.queue: collections.deque[dict[str, Any]] = collections.deque()
        self.playing: dict[str, Any] | None = None
        self.play_t = 0.0
        self.speed = 1
        self.loop_overlay = False
        self._threads: list[QThread] = []
        self._replay_tag = 0
        self._replay_view: tuple[float, float, float, float] | None = None
        """지금 재생(타임라인·겹쳐 보기·고속 챔피언) 환경의 기본 보기(editor.view)."""
        self._still_seq = 0
        """_show_still_scene_soon의 꼬리표(새로 미룰 때마다 늘어남: 늦은 앞 요청은 그리지 않음)."""
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
        self._scene_build_s = 0.0
        """마지막으로 끝까지 만든 장면의 실제 시간(초): 「학습 먼저」에서 다음 장면을 언제 만들기 시작할지(SKIP_SCENE_LEAD_S)."""
        self._closing = False
        # v6.0 관전 방식(설계 4.12)
        self.pace: str = PACE_DEFAULT
        """관전 방식(PACES): 「보기」 묶음의 pace_combo 값."""
        self._live_started = 0.0
        """관전 중인 세대 재생을 시작한 실제 시각(time.monotonic, 「학습 먼저」의 SKIP_MAX_VIEW)."""
        self._skip_end = math.inf
        """「학습 먼저」에서 새 세대가 기다릴 때 지금 재생을 끝낼 재생 시각(대표 시험 0번 궤적 끝 + SKIP_TAIL_S)."""
        self._wait_log: collections.deque[tuple[float, float]] = collections.deque(maxlen=WAIT_HINT_WINDOW)
        """최근 세대의 (재생을 기다린 초, 계산 초): 「세대마다 다 보기」로 관전한 세대만."""
        self._wait_hinted = False
        """이 실행에서 기다림 알림(LAB_PACE_WAIT_HINT)을 이미 보였는가(실행마다 한 번)."""
        self._wait_note = ""
        """상태 글 끝줄: 기다림 알림. 실행이 바뀌거나 「학습 먼저」로 바꾸면 지움."""
        self._pace_note = ""
        """상태 글 끝줄: 관전 방식 기본값이 바뀌었다는 예전 사용자 알림(LAB_PACE_CHANGED_NOTICE). 방식을 바꾸거나 학습 프로세스가
        닫히면 지움."""
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
        self._inspect_absent: tuple[int, frozenset[str]] | None = None
        """클릭 검사 재생 작업(_replay_tag)과 그 개체의 없는 부품 센서 채널(InspectPanel의 「(없음)」)."""
        self._inspect_msg: dict[str, Any] | None = None
        """클릭 검사한(또는 그 재생을 기다리는) 개체의 관전 세대 메시지. 그 세대를 재생하는 동안은 「학습 먼저」도 재생을 자르지
        않는다(_skip_due). 다른 세대를 재생하기 시작하면 검사를 닫는다."""
        self._structure_noted: set[str] = set()
        """LAB_GA_NO_STRUCTURE를 이미 알린 실행 폴더(실행마다 한 번)."""
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
        # 6.2.0: 「내 프리셋」 묶음, 프리셋 권장 설정, 상태 점과 시작 알림
        self._preset_note = ""
        """상태 글 끝줄(학습 전): 프리셋 권장 설정을 칸에 넣었다는 한 줄(LAB_PRESET_TRAIN_APPLIED). 실행이 바뀌면 지움."""
        self._preset_form = False
        """「학습 안정성」 칸에 지금 프리셋 권장값이 들어 있다(다음에 권장값 없는 환경을 고르면 새 실행 기본값으로 되돌린다)."""
        self._emitted_state: str | None = None
        self._stop_reason = ""
        """학습 프로세스가 알린 멈춘 까닭(finished의 reason: "max_generations" 등). 다른 상태면 빈 글."""
        self.start_notice = ""
        """학습을 시작하고 첫 세대가 오기까지 캔버스 가운데에 띄운 알림 글(U6). 빈 글이면 없음."""
        self._builtin_count = 0
        self._user_dir: str | None = None
        self._env_watch = QFileSystemWatcher(self)
        self._env_watch.directoryChanged.connect(lambda _p: self._user_envs_later.start())
        self._user_envs_later = QTimer(self)
        self._user_envs_later.setSingleShot(True)
        self._user_envs_later.setInterval(300)
        self._user_envs_later.timeout.connect(self.refresh_user_envs)

        self._build_ui()
        self._restore_settings()
        self._method_changed(set_defaults=False)
        self.refresh_user_envs()
        self._update_buttons()
        platform_now()  # 실행을 열 때마다 비교하는 지금 플랫폼 정보(처음 한 번 수십 ms: uname·WMI)를 시작할 때 미리
        freeze_long_lived()

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
        self.btn_timeline.setToolTip(S.LAB_TIMELINE_PLAY_TIP)
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
        panel_scroll = _PanelScroll()  # 처음 너비 = 내용 최소 너비 이상(가로로 잘리지 않게, T16)
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
        # 6.2.0 U12: 두 나눔 막대의 자리를 설정에 남기고 다음에 되살린다(_restore_splits, 접힘 규칙을 깨면 버림)
        self._split_h, self._split_v = top, split
        self._panel_scroll = panel_scroll
        self._split_check = False
        for splitter in (top, split):
            splitter.splitterMoved.connect(lambda _pos, _i: self._save_splits())
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
        self.env_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)  # 긴 경로(U5, 위와 같음)
        self.env_combo = QComboBox()
        self.env_combo.setObjectName("env_combo")
        self.env_combo.addItem(S.LAB_ENV_BUILTIN, None)
        builtins = paths.builtin_envs()
        for p in builtins:  # 정해진 순서(착륙선, 트랙, 언덕차, 점프 게임, 글라이더, 바퀴, 달리기), 그 밖은 이름순
            title = self._env_title(p)
            if builtin_status(p.stem) == "experimental":  # 설계 5.2: 「실험」 등급의 내장 프리셋
                title = S.LAB_ENV_EXPERIMENTAL.format(name=title)
            self.env_combo.addItem(title, str(p))
        self._builtin_count = len(builtins)
        self.env_combo.activated.connect(self._env_combo_chosen)
        btn_open = QPushButton(S.LAB_ENV_OPEN)
        btn_open.clicked.connect(self._choose_env)
        # 6.2.0 설계 3.1 H: 「새 프리셋…」(에디터의 같은 단추와 같은 글·풍선, D46). 주 창이 new_preset_requested를 받는다
        self.btn_new_preset = QPushButton(S.ED_NEW_PRESET)
        self.btn_new_preset.setObjectName("btn_new_preset")
        self.btn_new_preset.setToolTip(S.ED_NEW_PRESET_TIP)
        self.btn_new_preset.clicked.connect(self.new_preset_requested.emit)
        self.btn_reload = QPushButton(S.LAB_ENV_RELOAD)
        self.btn_reload.setToolTip(S.LAB_ENV_RELOAD_TIP)
        self.btn_reload.clicked.connect(self.reload_env)
        self.btn_edit = QPushButton(S.LAB_ENV_EDIT)
        self.btn_edit.setToolTip(S.LAB_ENV_EDIT_TIP)
        self.btn_edit.clicked.connect(self._edit_env)
        el.addWidget(self.env_label)
        el.addWidget(self.env_combo)
        el.addWidget(btn_open)
        el.addWidget(self.btn_new_preset)
        el.addWidget(self.btn_reload)
        el.addWidget(self.btn_edit)

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
        # 6.2.0 U9: 학습 중에는 이 묶음이 잠긴다는 한 줄(묶음 맨 위, 학습 중에만 보임. 접힘선 아래라 접힘 예산과 무관)
        self.locked_note = style.note(S.LAB_LOCKED_NOTE)
        self.locked_note.hide()
        form.addRow(self.locked_note)
        form.addRow(S.LAB_METHOD, self.method)
        # v6.1 「뇌」 줄(설계 5.1): 학습 방법 바로 아래, CMA-ES일 때만. 「자라기」면 두 칸과 회색 안내 두 줄
        self.brain_combo = QComboBox()
        self.brain_combo.addItem(S.LAB_BRAIN_FIXED, BRAIN_FIXED)
        self.brain_combo.addItem(S.LAB_BRAIN_GROW, BRAIN_GROW)
        self.brain_combo.setToolTip(S.LAB_BRAIN_TIP)
        self.grow_start = _spin(1, GROW_START_MAX, GROW_START_DEFAULT)
        self.grow_every = _spin(1, GROW_EVERY_MAX, GROW_EVERY_DEFAULT)
        self.grow_hint = QLabel()
        self.grow_honest = QLabel(S.LAB_GROW_HINT_HONEST)
        for hint in (self.grow_hint, self.grow_honest):
            hint.setWordWrap(True)
            style.tone(hint, "ink2")  # 보조 글(학습 안정성 안내와 같은 색, 6.2.0 U2)
        form.addRow(S.LAB_BRAIN, self.brain_combo)
        form.addRow(S.LAB_GROW_START, self.grow_start)
        form.addRow(S.LAB_GROW_EVERY, self.grow_every)
        form.addRow(self.grow_hint)
        form.addRow(self.grow_honest)
        self.brain_combo.currentIndexChanged.connect(lambda _i: self._update_brain_rows())
        self.brain_combo.activated.connect(lambda _i: self._save_brain_settings())
        for spin in (self.grow_start, self.grow_every):
            spin.valueChanged.connect(self._grow_field_changed)
        form.addRow(S.LAB_SEED, self.seed)
        self._method_rows: dict[str, list[QWidget]] = {"cmaes": [self.brain_combo], "ga": [], "sa": []}

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
        # 6.2.0 U9: 학습 중에 정지 조건을 바꾸면 멈춘 뒤 「설정 적용」을 눌러야 바뀐다(학습 중에만 보임)
        self.stop_apply_note = style.note(S.LAB_STOP_APPLY_NOTE)
        self.stop_apply_note.hide()
        sf.addRow(self.stop_apply_note)

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
        style.tone(self.stability_hint, "ink2")  # 보조 글(테마를 따름, 6.2.0 U2)
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
        # 6.2.0 U5: 패널이 좁아져(394px) 띄어쓰기 없는 긴 경로(실행 폴더·내보낸 파일)가 패널을 넓혀 가로 스크롤로 단추를 자르지
        # 않게, 상태 글의 폭은 패널이 정한다(넘치는 낱말은 글 상자 안에서 잘림)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        # 6.2.0 U7: 상태 글 왼쪽에 상태 색 점 하나(칩 줄은 없다: 접힘 예산). 첫 줄 높이, 위 맞춤, 사이 4px
        self.state_dot = style.StateDot()
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(style.SPACE[0])
        status_row.addWidget(self.state_dot, 0, Qt.AlignmentFlag.AlignTop)
        status_row.addWidget(self.status, 1)
        g.addLayout(status_row, 3, 0, 1, 2)

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
        # v6.0 관전 방식(설계 5.1): 관전·고속 줄 다음, 고속이면 흐리게
        self.pace_combo = QComboBox()
        for key, name in ((PACE_SKIP, S.LAB_PACE_SKIP), (PACE_EVERY, S.LAB_PACE_EVERY)):
            self.pace_combo.addItem(name, key)
        self.pace_combo.setCurrentIndex(self.pace_combo.findData(PACE_DEFAULT))
        self.pace_combo.currentIndexChanged.connect(lambda _i: self.set_pace(str(self.pace_combo.currentData())))
        vf.addRow(S.LAB_PACE, self.pace_combo)
        self.speed_combo = QComboBox()
        for n in SPEEDS:
            self.speed_combo.addItem(S.LAB_SPEED_ITEM.format(n=n), n)
        self.speed_combo.currentIndexChanged.connect(lambda i: self.set_speed(SPEEDS[i]))
        vf.addRow(S.LAB_SPEED, self.speed_combo)
        self.fast_interval = _spin(1, 100000, 10)
        self.fast_interval.valueChanged.connect(lambda _v: self._save_settings())
        vf.addRow(S.LAB_FAST_INTERVAL, self.fast_interval)
        # v3.0(설계 D17): 진행·보기 묶음을 패널 맨 위로(만드는 순서는 그대로). 묶음은 여기서 화면 차례대로 처음 단다:
        # 위젯은 부모에 달리는 차례로 Tab 차례(초점 사슬)에 들어가므로, 화면 차례와 Tab 차례가 같아진다.
        for group in (ctl, view, env, st, stop, stab):
            lay.addWidget(group)
        lay.addStretch(1)
        return w

    @staticmethod
    def _env_title(path: Path) -> str:
        try:
            return str(json.loads(path.read_text(encoding="utf-8-sig")).get("name") or path.stem)
        except (OSError, ValueError, AttributeError):
            return path.stem

    def refresh_user_envs(self) -> list[Path]:
        """환경 목록의 「내 프리셋」 묶음을 내 환경 폴더로 다시 채운다(설계 3.1 H). 묶음 머리는 고를 수 없는 줄이고, 파일이
        없으면 묶음도 없다. 고른 줄은 그대로 둔다. 실험실이 보일 때·폴더가 바뀔 때(저장) 다시 부른다."""
        combo = self.env_combo
        mine = user_presets()
        keep = 1 + self._builtin_count
        want = [str(p) for p in mine]
        have = [combo.itemData(i) for i in range(keep + 1, combo.count())]
        titles = [self._env_title(p) for p in mine]
        if want == have and titles == [combo.itemText(i) for i in range(keep + 1, combo.count())]:
            self._watch_user_dir()
            return mine
        current = combo.currentData()
        index = combo.currentIndex()
        combo.blockSignals(True)
        try:
            while combo.count() > keep:
                combo.removeItem(combo.count() - 1)
            if mine:
                combo.addItem(S.LAB_ENV_GROUPS["mine"], None)
                head = combo.model().item(keep)  # type: ignore[attr-defined]  # QStandardItemModel(QComboBox 기본)
                head.setEnabled(False)
                head.setToolTip(S.LAB_ENV_GROUPS_TIP["mine"])
                for p, title in zip(mine, titles, strict=True):
                    combo.addItem(title, str(p))
                    combo.setItemData(combo.count() - 1, str(p), Qt.ItemDataRole.ToolTipRole)
            found = combo.findData(current) if current else -1
            combo.setCurrentIndex(found if found >= 0 else (0 if index >= keep else index))
        finally:
            combo.blockSignals(False)
        self._watch_user_dir()
        return mine

    def _watch_user_dir(self) -> None:
        """내 환경 폴더를 지켜본다(에디터·안내 창·다른 프로그램이 저장하면 묶음을 다시 채움). 데이터 폴더를 바꾸면 따라간다."""
        folder = str(paths.user_env_dir())
        if folder == self._user_dir:
            return
        if self._user_dir is not None and self._user_dir in self._env_watch.directories():
            self._env_watch.removePath(self._user_dir)
        self._user_dir = folder
        if Path(folder).is_dir():
            self._env_watch.addPath(folder)
        else:
            self._user_dir = None  # 아직 없는 폴더: 다음에 다시 본다

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh_user_envs()
        if self._split_check:
            QTimer.singleShot(0, self, self._check_splits)

    # -- 6.2.0 U12: 나눔 막대 자리 기억 ----------------------------------------------
    SPLIT_KEYS: Final = ("lab/split_h", "lab/split_v")

    def _save_splits(self) -> None:
        if self._restoring:
            return
        self._split_check = False  # 사용자가 옮긴 자리다
        for key, splitter in zip(self.SPLIT_KEYS, (self._split_h, self._split_v), strict=True):
            self.settings.setValue(key, splitter.saveState())

    def _restore_splits(self) -> None:
        """설정에 남은 나눔 막대 자리를 되살린다. 다 보이는지는 처음 보일 때 _check_splits가 확인한다."""
        restored = False
        for key, splitter in zip(self.SPLIT_KEYS, (self._split_h, self._split_v), strict=True):
            state = self.settings.value(key)
            if isinstance(state, (bytes, bytearray)) or hasattr(state, "data"):
                try:
                    restored = splitter.restoreState(state) or restored
                except (TypeError, ValueError):
                    pass
        if restored and (min(self._split_h.sizes()) <= 0 or min(self._split_v.sizes()) <= 0):
            self._reset_splits()  # 접힌(크기 0) 칸은 바로 버린다
            return
        self._split_check = restored

    def _check_splits(self) -> None:
        """되살린 자리에서 「진행」·「보기」의 세 위젯(학습 시작·멈춤·재생 배속)이 스크롤 없이 다 보이지 않으면 기본 배치로
        돌아간다(접힘 규칙 T16, 다듬기 U12)."""
        if not self._split_check or not self.isVisible():
            return
        self._split_check = False
        for w in (self.btn_start, self.btn_pause, self.speed_combo, self.canvas):
            region = w.visibleRegion()
            if region.isEmpty() or region.boundingRect() != w.rect():
                self._reset_splits()
                return

    def _reset_splits(self) -> None:
        """기본 배치: 패널은 내용 폭(_PanelScroll.sizeHint), 위·아래는 늘림 비율 3:2."""
        self._split_check = False
        h = max(2, sum(self._split_h.sizes()) or self._split_h.width())
        panel = min(h - 1, self._panel_scroll.sizeHint().width())
        self._split_h.setSizes([h - panel, panel])
        v = max(5, sum(self._split_v.sizes()) or self._split_v.height())
        self._split_v.setSizes([v * 3 // 5, v - v * 3 // 5])
        for key in self.SPLIT_KEYS:
            self.settings.remove(key)

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
        pace = str(s.value("lab/pace", PACE_DEFAULT))
        brain = str(s.value("lab/brain", BRAIN_FIXED))
        grow_start = number("lab/grow_start", GROW_START_DEFAULT)
        grow_every = number("lab/grow_every", GROW_EVERY_DEFAULT)
        self._restoring = True
        self._brain_quiet = True
        try:
            if w > 0:
                self.workers.setValue(w)
            if speed in SPEEDS:
                self.speed_combo.setCurrentIndex(SPEEDS.index(speed))
            if mode == "fast":
                self.rb_fast.setChecked(True)
            self.fast_interval.setValue(interval)
            if pace in PACES:
                self.pace_combo.setCurrentIndex(self.pace_combo.findData(pace))
            if brain in BRAINS:
                self.brain_combo.setCurrentIndex(self.brain_combo.findData(brain))
            self.grow_start.setValue(grow_start)
            self.grow_every.setValue(grow_every)
        finally:
            self._restoring = False
            self._brain_quiet = False
        self._update_view_rows()
        self._restore_splits()
        self._check_pace_notice()

    def _update_view_rows(self) -> None:
        """「보기」 묶음 칸의 켜짐(6.2.0 U9): 관전 방식·재생 배속은 관전일 때만, 고속 챔피언 간격은 고속일 때만. 줄은 그대로 보인다."""
        on = self.rb_spectate.isChecked()
        style.set_enabled(self.pace_combo, on, S.LAB_WHY_SPECTATE_ONLY)  # 고속이면 관전 방식은 뜻이 없다(설계 5.1)
        style.set_enabled(self.speed_combo, on, S.LAB_WHY_SPECTATE_ONLY)
        style.set_enabled(self.fast_interval, not on, S.LAB_WHY_FAST_ONLY)

    def _check_pace_notice(self) -> None:
        """기본 관전 방식을 「학습 먼저」로 바꾼 v6.0(D24): 예전 프로그램으로 실험실을 쓴 사용자(보기 칸을 저장한 lab/mode는 있고
        lab/pace는 없음)에게 처음 한 번 상태 글로 알린다(lab/pace_notice = 1). v6에서 보기 칸을 저장하면 lab/pace도 함께 쓴다."""
        s = self.settings
        if PACE_DEFAULT != PACE_SKIP or s.contains("lab/pace") or not s.contains("lab/mode") \
                or s.contains("lab/pace_notice"):
            return
        s.setValue("lab/pace_notice", 1)
        self._pace_note = S.LAB_PACE_CHANGED_NOTICE
        self._update_status()

    def _save_brain_settings(self) -> None:
        """「뇌」 줄을 사용자가 바꿨다(v6.1): 고른 값을 설정에 남긴다. 실행 설정·환경으로 칸을 채울 때는 쓰지 않는다."""
        if self._restoring or self._brain_quiet:
            return
        self.settings.setValue("lab/brain", str(self.brain_combo.currentData()))
        self.settings.setValue("lab/grow_start", self.grow_start.value())
        self.settings.setValue("lab/grow_every", self.grow_every.value())

    def _save_settings(self) -> None:
        if self._restoring:
            return
        self.settings.setValue("lab/speed", self.speed)
        self.settings.setValue("lab/mode", "spectate" if self.rb_spectate.isChecked() else "fast")
        self.settings.setValue("lab/fast_interval", self.fast_interval.value())
        self.settings.setValue("lab/workers", self.workers.value())
        self.settings.setValue("lab/pace", self.pace)

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
            show_error(self, S.LAB_OPEN_ENV_FAILED, exc)
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
            show_error(self, S.LAB_OPEN_ENV_FAILED, exc)
            return False
        self._reset_run_view()
        self.run_dir = None
        self.run_info = {}
        self.state = "idle"
        self.env_path, self.env_text, self.spec, self.layout = path, text, spec, layout
        self._set_watch_meta()
        self._set_structure_envs()  # 몸 구조 타일: 학습 전에도 이 환경의 빈 상태·안내를 보인다
        self.env_label.setText(f"{spec.file.name}\n{where}")
        # 복제할 때만 칸이 없는 설정까지 가져간다(검토 24). 그냥 환경을 불러오면 그런 값은 기본값으로 돌아간다.
        self._base_config = dict(config) if config else None
        if config:
            self._form_from_config(config)  # 복제한 실행은 그 실행의 설정(프리셋 권장값을 얹지 않는다)
        else:
            self._update_brain_rows()  # 이 환경의 은닉층으로 「자라기」 항목·처음 뉴런 수 최댓값·안내 줄
            self._apply_preset_train(spec)
        if spec.file.evaluation.trials:
            self.trials.setSpecialValueText(f"{S.LAB_TRIALS_ENV} ({spec.file.evaluation.trials})")
        self._show_still_scene()
        self._update_status()
        self._update_buttons()
        self.env_changed.emit()
        return True

    def _apply_preset_train(self, spec: EnvSpec) -> None:
        """(6.2.0 설계 D31·3.1 H) 프리셋의 권장 새 실행 설정(editor.train)을 「학습 안정성」 칸에 넣고 상태 글에 한 줄.

        값은 TrainConfig.for_new_run(spec)이 받아들인 것(모르는 키·틀린 값은 빠짐)이다. 권장값이 없는 환경(옛 파일 전부)은
        칸을 건드리지 않는다(v6.1과 같음). 다만 앞서 고른 프리셋의 권장값이 칸에 남아 있으면 새 실행 기본값으로 되돌린다.
        """
        raw = editor_train(spec.file.editor)
        fresh = TrainConfig.for_new_run(spec)
        applied = [k for k in PRESET_TRAIN_KEYS if k in raw and raw[k] == getattr(fresh, k)]
        self._preset_note = ""
        if not applied and not self._preset_form:
            return
        values = {k: getattr(fresh, k) for k in PRESET_TRAIN_KEYS}
        for key in PRESET_TRAIN_KEYS:
            self._form_exact.pop(key, None)
        patience = values["restart_patience"]
        self.restart_on.setChecked(patience is not None)
        self.restart_patience.setValue(int(patience) if patience is not None else SHOWN_RESTART_PATIENCE)
        if patience is not None and _shows(self.restart_patience, float(patience)):
            self._form_exact["restart_patience"] = (self.restart_patience.value(), patience)
        self.restart_from.setCurrentIndex(max(0, self.restart_from.findData(values["restart_from"])))
        for widget, key in ((self.qualify_entrants, "qualify_entrants"), (self.qualify_trials, "qualify_trials")):
            widget.setValue(int(values[key]))
            if _shows(widget, float(values[key])):
                self._form_exact[key] = (widget.value(), values[key])
        self._preset_form = bool(applied)
        if applied:
            self._preset_note = S.LAB_PRESET_TRAIN_APPLIED.format(
                what=" · ".join(preset_train_text(k, values[k]) for k in applied))

    def _show_still_scene_soon(self) -> None:
        """실행을 열 때(_adopt_run): 전시 시드의 세계를 다음 이벤트 루프 차례에 그린다. 만들기(Episode·정적 그림
        10~20ms)를 open_run 밖으로 빼서 여는 동안 이벤트 루프를 짧게 막는다. 그사이 다시 열었거나(다른 실행·되감기·
        환경 바꿈) 다른 장면(관전 세대, 끝난 재생)이 캔버스를 차지했으면 그리지 않는다."""
        self._still_seq += 1
        seq, spec, layout, world = self._still_seq, self.spec, self.layout, self.canvas.world
        QTimer.singleShot(0, self, lambda: self._still_scene_due(seq, spec, layout, world))

    def _still_scene_due(self, seq: int, spec: EnvSpec | None, layout: GenomeLayout | None, world: Any) -> None:
        if (seq == self._still_seq and spec is self.spec and layout is self.layout and world is self.canvas.world
                and self.playing is None):
            with GUI_PRIORITY.busy():
                self._show_still_scene()

    def _show_still_scene(self) -> None:
        """전시 시드의 세계만 그린다(시뮬레이션하지 않는다: 화면 스레드를 막지 않게)."""
        if self.spec is None or self.layout is None:
            return
        g = self.layout.initial()[None, :]
        ep = Episode(self.spec, self.layout.body_values(g)[0] if self.layout.n_body else None, seeds.display_trial())
        self.canvas.set_scene(ep.geometry(), None, channels=list(ep.actuator_channels()), view=view_box(self.spec.file))
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
        self._update_brain_rows()
        if set_defaults:
            self.popsize.setValue(64 if method == "ga" else 32)

    # -- v6.1 「뇌」 줄 ---------------------------------------------------------
    def grow_blocked(self) -> str | None:
        """지금 환경에서 「작게 시작해 자라기」를 고를 수 없는 까닭(풍선 글), 고를 수 있으면 None(환경이 없어도 None)."""
        lay = self.layout
        if lay is None:
            return None
        if lay.brain.n_out == 0:
            return S.LAB_BRAIN_GROW_NO_BRAIN
        if not lay.brain.hidden:
            return S.LAB_BRAIN_GROW_NO_HIDDEN
        return None

    def brain_grows(self) -> bool:
        """「뇌」 줄이 새 실행을 자라는 뇌(학습 방법 GROW_METHOD)로 만드는가: CMA-ES + 「작게 시작해 자라기」."""
        return str(self.method.currentData()) == "cmaes" and self.brain_combo.currentData() == BRAIN_GROW

    def _set_row_visible(self, widget: QWidget, visible: bool) -> None:
        widget.setVisible(visible)
        label = self._method_form.labelForField(widget)
        if label is not None:
            label.setVisible(visible)

    def _update_brain_rows(self) -> None:
        """「뇌」 줄: 「자라기」 항목 켜짐(환경)과 풍선, 처음 뉴런 수 최댓값, 두 칸·안내 줄·IPOP 보이기(CMA-ES + 「자라기」)."""
        if not hasattr(self, "_method_form"):
            return  # 패널을 만드는 중
        blocked = self.grow_blocked()
        item = self.brain_combo.model().item(1)  # type: ignore[attr-defined]  # QStandardItemModel(QComboBox 기본)
        item.setEnabled(blocked is None)
        item.setToolTip(blocked or "")
        self.brain_combo.setToolTip(S.LAB_BRAIN_TIP if blocked is None else f"{S.LAB_BRAIN_TIP}\n\n{blocked}")
        quiet, self._brain_quiet = self._brain_quiet, True
        try:
            if blocked is not None and self.brain_combo.currentData() == BRAIN_GROW:
                self.brain_combo.setCurrentIndex(self.brain_combo.findData(BRAIN_FIXED))
            self._set_grow_start_max()
        finally:
            self._brain_quiet = quiet
        cmaes = str(self.method.currentData()) == "cmaes"
        grow = self.brain_grows()
        for w in (self.grow_start, self.grow_every, self.grow_hint, self.grow_honest):
            self._set_row_visible(w, grow)
        self._set_row_visible(self.ipop, cmaes and not grow)  # 자라는 뇌는 IPOP과 함께 쓸 수 없다(설계 D33)
        self._update_grow_hint()

    def _set_grow_start_max(self) -> None:
        """「처음 뉴런 수」 칸의 최댓값 = 지금 환경의 가장 작은 은닉층(환경이 없거나 은닉층이 없으면 GROW_START_MAX)."""
        lay = self.layout
        top = min(lay.brain.hidden) if lay is not None and lay.brain.hidden else GROW_START_MAX
        if self.grow_start.maximum() != top:
            self.grow_start.setMaximum(top)

    def grow_hint_text(self) -> str:
        """안내 첫 줄: 고른 처음 뉴런 수·간격과 지금 환경으로 「2 → 4 → 8 → 16개, 30세대쯤에 다 자랍니다.」. 뉴런 수는 은닉층
        합이고 층마다 두 배(틀에서 멈춤, learners/grow.growth_schedule). 처음부터 다 자란 크기면 LAB_GROW_HINT_ALREADY. 환경이
        없거나 은닉층이 없으면 빈 글."""
        from evolab.learners.grow import growth_schedule  # 학습 방법의 일정 그대로(창을 열 때 불러오지 않게 여기서)

        lay = self.layout
        if lay is None or not lay.brain.hidden or lay.brain.n_out == 0:
            return ""
        plan = growth_schedule(lay.brain.hidden, self.grow_start.value(), self.grow_every.value())
        if len(plan) == 1:
            return S.LAB_GROW_HINT_ALREADY
        return S.LAB_GROW_HINT.format(path=" \u2192 ".join(str(sum(counts)) for _gen, counts in plan), gen=plan[-1][0])

    def _grow_field_changed(self, _value: int) -> None:
        self._update_grow_hint()
        self._save_brain_settings()

    def _update_grow_hint(self) -> None:
        text = self.grow_hint_text()
        if self.grow_hint.text() != text:
            self.grow_hint.setText(text)
        self.grow_hint.setVisible(self.brain_grows() and bool(text))  # 환경이 없으면 빈 줄을 두지 않는다

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
        fresh = TrainConfig.for_new_run(self.spec)  # 있음 유전자가 있는 환경이면 구조 뒤집기 0.05(v2.1, D41)
        if self._base_config:
            base = TrainConfig.from_dict(self._base_config)
            # 복제한 설정에 구조 뒤집기 키가 없으면(v2.0까지 만든 실행) 새 실행 기본값을 쓴다
            missing = {k: getattr(fresh, k) for k in ("ga_structure_flip", "sa_structure_flip")
                       if k not in self._base_config and hasattr(fresh, k)}
            if missing:
                base = dataclasses.replace(base, **missing)
        else:
            base = fresh
        fields = {key: self._form_value(key, widget, convert) for key, widget, convert in (
            ("run_seed", self.seed, _same), ("popsize", self.popsize, _same), ("sigma0", self.sigma, _same),
            ("trials", self.trials, _or_none), ("ga_elites", self.ga_elites, _same),
            ("ga_crossover", self.ga_crossover, _same), ("ga_mutation_rate", self.ga_mut_rate, _same),
            ("ga_mutation_sigma", self.ga_mut_sigma, _same), ("ga_sigma_decay", self.ga_decay, _same),
            ("sa_neighbors", self.sa_neighbors, _same), ("sa_step", self.sa_step, _same),
            ("sa_initial_temperature", self.sa_temp, _or_none), ("sa_cooling", self.sa_cooling, _same),
            ("qualify_entrants", self.qualify_entrants, _same), ("qualify_trials", self.qualify_trials, _same),
        )}
        grow = self.brain_grows()
        if grow:  # v6.1: 자라는 뇌 칸은 그 실행에만(크기 고정 실행의 설정은 v6.0과 같은 값, 설계 D31)
            method = GROW_METHOD
            fields["grow_start"] = self._form_value("grow_start", self.grow_start, _same)
            fields["grow_every"] = self._form_value("grow_every", self.grow_every, _same)
        c = dataclasses.replace(base, method=method, ipop=self.ipop.isChecked() and not grow, **fields,
                                **self._restart_fields())
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
        self._preset_form = False  # 칸은 이제 그 실행의 설정이다(프리셋 권장값이 아님)

        def setv(widget: Any, key: str, empty: Any, scale: float = 1.0) -> None:
            value = c.get(key)
            if value is None:
                widget.setValue(empty)
            else:
                widget.setValue(value if scale == 1.0 else float(value) / scale)
            if key in c and (value is None or _shows(widget, float(value) / scale)):
                self._form_exact[key] = (widget.value(), value)

        method = c.get("method", "cmaes")
        grow = method == GROW_METHOD  # v6.1: 자라는 뇌 실행은 CMA-ES + 「뇌」 「작게 시작해 자라기」로 보인다
        quiet, self._brain_quiet = self._brain_quiet, True
        try:
            self.brain_combo.blockSignals(True)
            self.brain_combo.setCurrentIndex(self.brain_combo.findData(BRAIN_GROW if grow else BRAIN_FIXED))
            self.brain_combo.blockSignals(False)
            self._set_grow_start_max()  # 이 실행 환경의 최댓값을 먼저(앞 환경의 최댓값으로 값이 잘리지 않게)
            setv(self.grow_start, "grow_start", GROW_START_DEFAULT)
            setv(self.grow_every, "grow_every", GROW_EVERY_DEFAULT)
        finally:
            self._brain_quiet = quiet
        idx = self.method.findData("cmaes" if grow else method)
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
            show_error(self, S.LAB_CHANGE_FAILED, exc, next_step=S.ERR_SETTINGS_NEXT)
            return False
        problem = qualify_budget_problem(config)  # 예선만으로 최대 시험 횟수를 다 쓰면 시작하지 않는다
        if problem is not None:
            QMessageBox.warning(self, S.LAB_CHANGE_FAILED, problem)
            return False
        self._reset_run_view()
        self.state = "starting"
        req = SessionRequest(kind="new", env_text=self.env_text, config=config.to_dict(), runs_dir=str(paths.runs_dir()),
                             workers=self.workers.value(), spectate=self.rb_spectate.isChecked(), pace=self.pace)
        self.client.start(req)
        # 6.2.0 U6: 누른 바로 그때 「시작하는 중」과 캔버스 알림(개체 수는 칸이 아니라 지금 만든 설정에서)
        pop = config.sa_neighbors if config.method == "sa" else config.popsize
        self._set_start_notice(S.LAB_NOTICE_FIRST_GEN.format(pop=pop))
        self._update_status()
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
                             spectate=self.rb_spectate.isChecked(), pace=self.pace)
        self.client.start(req)
        self._set_start_notice(S.LAB_NOTICE_NEXT_GEN)  # 6.2.0 U6: 이어서 학습은 숫자 없는 알림
        self._update_status()
        self._update_buttons()
        return True

    def _set_start_notice(self, text: str) -> None:
        """(6.2.0 U6) 첫 세대가 오기까지 캔버스 가운데의 멈춘 알림 글. 캔버스의 set_notice(묶음 PL, 다듬기 3.3)에 맡긴다."""
        if text == self.start_notice:
            return
        old, self.start_notice = self.start_notice, text
        set_notice = getattr(self.canvas, "set_notice", None)
        if callable(set_notice):
            set_notice(text)
        elif self.canvas.hud in ("", old):  # set_notice가 아직 없는 캔버스: HUD 상자에 띄운다(재생이 HUD를 쓰기 전까지만)
            self.canvas.set_hud(text)

    def extend_and_continue(self, generations: int = 10) -> bool:
        """(6.2.0 U10, 다듬기 3.3) 「최대 세대」에 닿아 멈춘 학습을 n세대 더: 칸 +n → 설정 적용 → 이어서.

        학습이 멈춰 있지 않거나 멈춘 까닭이 최대 세대가 아니면 False(아무것도 하지 않음)."""
        if not (self.client.active and self.state == "finished" and self._stop_reason == "max_generations"):
            return False
        limit = self.config_value("max_generations")
        base = int(limit) if isinstance(limit, (int, float)) and limit else self.generation
        self.max_generations.setValue(max(base, self.generation) + int(generations))
        self.apply_settings()
        self.continue_run()
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
            show_error(self, S.LAB_OPEN_ENV_FAILED, exc)
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
        self.client.send({"type": "mode", "spectate": bool(on), "pace": self.pace})
        self._update_view_rows()
        self._wait_log.clear()
        if not on and self._wait_note:  # 고속에서는 학습이 기다리지 않는다: 흐린 칸을 바꾸라는 알림을 지운다
            self._wait_note = ""
            self._refresh_status()
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

    def set_pace(self, pace: str) -> None:
        """관전 방식(v6.0 설계 4.12)을 바꾼다: 학습 프로세스에 mode 명령으로 알리고(「학습 먼저」면 기다림이 바로 풀림) 설정에
        남긴다. 「학습 먼저」로 바꾸면 기다리던 세대 가운데 가장 새 것만 남긴다."""
        if pace not in PACES or pace == self.pace:
            return
        self.pace = pace
        if self.pace_combo.currentData() != pace:
            self.pace_combo.setCurrentIndex(self.pace_combo.findData(pace))
        self.client.send({"type": "mode", "spectate": self.rb_spectate.isChecked(), "pace": pace})
        self._wait_log.clear()
        notes = (self._pace_note, self._wait_note)
        self._pace_note = ""
        if pace == PACE_SKIP:
            self._wait_note = ""  # 알림이 권한 것을 했다
            if self.view_mode == "live":
                self._keep_newest_waiting()
        if notes != (self._pace_note, self._wait_note):
            self._refresh_status()
        self._save_settings()

    # ------------------------------------------------------------------
    # 학습 프로세스에서 온 것
    # ------------------------------------------------------------------
    def _reset_run_view(self) -> None:
        self.rows = []
        self._preset_note = ""
        self._session_note = ""
        self._map_notice = ""
        self._build_note = ""
        self._grow_note = ""
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
        style.set_enabled(self.btn_video, False, S.LAB_WHY_NO_VIDEO)
        style.set_enabled(self.btn_live, False, S.LAB_WHY_LIVE)
        self._fast_last = 0
        self._replay_tag += 1  # 아직 도는 앞 실행의 다시 시뮬레이션 결과는 버린다
        self.stop_flip()
        self._brain_tag += 1  # 앞 실행의 뇌 추적 결과도 버린다
        self._brain_play = None
        self._brain_live = None
        self._replay_brain = None
        self._world_pending.clear()
        self._wait_log.clear()
        self._wait_hinted = False
        self._wait_note = ""
        self.dashboard.clear()  # 보는 재미 기록·보는 세대·뇌 원천도 비운다
        self._brain_key = None
        self.canvas.set_hud("")  # 앞 실행의 「n세대 · 시험 0번」 글을 남기지 않는다
        self.timeline.setMaximum(1)
        self._update_timeline_label()
        self._close_inspect()

    def _adopt_run(self, msg: dict[str, Any]) -> None:
        # 이 안에서 시작하는 보는 재미 배경 읽기는 화면을 다 바꿀 때까지 기다린다(GIL을 다투지 않게)
        with GUI_PRIORITY.busy():
            self._adopt_run_now(msg)

    def _adopt_run_now(self, msg: dict[str, Any]) -> None:
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
        self._set_structure_envs()
        self._warn_other_platform()
        self.timeline.setMaximum(max(1, self.generation))
        self.timeline.setValue(max(1, self.generation))
        self._update_timeline_label()
        self._apply_focus()
        self._show_still_scene_soon()

    def _warn_other_platform(self) -> None:
        """다른 PC·다른 빌드에서 만든 실행이면 비트 재현이 보장되지 않는다고 알린다(명세: 재현 범위)."""
        from evolab import __version__

        then = dict(self.run_info.get("platform") or {})
        if not then:
            return
        now = platform_now()
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
        self._check_structure_notice()
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
        # 세대 메시지 처리(통계 줄·보는 재미 기록·장면 넣기) 동안 배경 작업(장면 만들기, 뇌 추적, 세계 그림)을 쉬게 한다(v6.0, v3.0
        # 남은 문제 「2세대가 올 때 GIL 다툼」)
        with GUI_PRIORITY.busy():
            self._on_generation_now(msg)

    def _on_generation_now(self, msg: dict[str, Any]) -> None:
        self._set_start_notice("")
        row = msg["row"]
        self.generation = int(msg["generation"])
        self.rows.append(row)
        self._note_switch()
        self._note_growth(msg)
        self.dashboard.add_row(row)
        if msg.get("champion") is not None:
            self.champion = msg["champion"]
        self._feed_dashboard(msg)
        at_end = self.timeline.value() >= self.timeline.maximum()
        self.timeline.setMaximum(max(1, self.generation))
        if at_end:
            self.timeline.setValue(self.generation)
        self._note_waited(msg, row)
        self._update_status()
        if msg.get("trajectories") is not None:
            if self.view_mode == "live":
                if self.pace == PACE_SKIP:
                    self._keep_newest_waiting(dropping_all=True)  # 재생 대기열은 가장 새 세대 하나(설계 4.12)
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

    def _note_growth(self, msg: Mapping[str, Any]) -> None:
        """v6.1: 세대 메시지의 "growth"(그 세대 끝에 뇌가 자람, 자란 세대만) → 상태 글 끝줄 LAB_STATUS_GROWN 한 줄. 키가 없거나
        값이 이상하면 아무것도 하지 않는다(크기 고정·옛 학습 프로세스)."""
        growth = msg.get("growth")
        if not isinstance(growth, Mapping):
            return
        try:
            self._grow_note = S.LAB_STATUS_GROWN.format(
                gen=int(msg["generation"]), old=int(growth["old"]), new=int(growth["new"]),
                old_links=int(growth["old_links"]), new_links=int(growth["new_links"]))
        except (KeyError, TypeError, ValueError):
            return

    def _keep_newest_waiting(self, dropping_all: bool = False) -> None:
        """「학습 먼저」의 재생 대기열 하나(설계 4.12): 기다리는 관전 세대를 버리고 그 세대 번호로 played를 보낸다.

        dropping_all(새 세대가 막 왔다, 새 세대가 기다리는 자리에 든다): 아직 장면을 만들기 시작하지 않은 세대를 모두 버린다(설계
        4.12). 몸 유전자 환경에서 장면을 다 만든 세대(대기열)와 만드는 중인 세대는 남긴다: 관전 재생 중에는 지금 재생이 끝나기
        조금 전에야 장면을 만들기 시작하므로(_scene_later) 그 장면은 곧 보일 것이고, 버리면 새 세대 장면을 다시 만드느라 늦고 한
        일을 버린다(v6.0 검토 고침: 설계는 만드는 중인 것을 멈추고 새 것을 만든다고 했다). 몸 유전자가 없는 환경의 대기열은
        장면을 재생할 때 만드므로 모두 버린다.

        아니면(「학습 먼저」로 바꿈) 가장 새 것 하나는 남긴다. 장면을 만드는 중인 세대는 재생 중이면 멈추게 하고(v1.2
        _drop_scene_jobs), 재생 중이 아니면 끝까지 만든다: 세대가 장면 만들기보다 빨리 오면 만들 때마다 멈춰 화면이 다음 세대를
        영영 보이지 못하므로.
        """
        waiting = [*self.queue, *self._scene_jobs]
        if not waiting:
            return
        building = self._scene_jobs[0] if self._scene_thread is not None and self._scene_jobs else None
        if dropping_all:
            kept = {id(building)} if building is not None else set()
            if self.layout is not None and self.layout.n_body:
                kept |= {id(m) for m in self.queue}
            for m in waiting:
                if id(m) not in kept:
                    self.client.send({"type": "played", "generation": int(m["generation"])})
            for line in (self.queue, self._scene_jobs):
                left = [m for m in line if id(m) in kept]
                line.clear()
                line.extend(left)
            return
        newest = max(waiting, key=lambda m: int(m["generation"]))
        keep = newest
        if building is not None and self.playing is None and keep is not building:
            keep_building = building  # 화면이 쉬고 있다: 만드는 것을 마저 만들어 먼저 보인다
        else:
            keep_building = None
        ready = [m for m in self.queue if m is keep]
        jobs = [m for m in self._scene_jobs if m is keep or m is keep_building]
        for m in waiting:
            if m is not keep and m is not keep_building:
                self.client.send({"type": "played", "generation": int(m["generation"])})
        self.queue.clear()
        self.queue.extend(ready)
        if building is not None and not any(m is building for m in jobs):
            self._drop_scene_jobs()  # 만드는 중인 것을 멈추게 한다(결과는 꼬리표가 달라 버려진다)
            self._scene_jobs.extend(jobs)
            self._next_scene()  # 멈춘 스레드가 끝나면(_scene_done) 그때 남은 것을 만든다
        else:
            self._scene_jobs.clear()
            self._scene_jobs.extend(jobs)

    def _note_waited(self, msg: Mapping[str, Any], row: Mapping[str, Any]) -> None:
        """기다림 알림(D25): 「세대마다 다 보기」로 관전한 세대의 (기다린 초, 계산 초)를 모아, 최근 WAIT_HINT_WINDOW세대 동안 기다린
        몫이 WAIT_HINT_RATIO를 넘으면 실행마다 한 번 상태 글 끝에 알린다. 배수 = 전체 ÷ (전체 − 기다림)."""
        if not (self.pace == PACE_EVERY and self.view_mode == "live" and self.rb_spectate.isChecked()):
            self._wait_log.clear()
            return
        waited = _float_value(msg.get("waited"))
        compute = _float_value(row.get("gen_seconds"))
        if waited is None or compute is None or not math.isfinite(waited) or not math.isfinite(compute):
            return
        self._wait_log.append((max(0.0, waited), max(0.0, compute)))
        if self._wait_hinted or len(self._wait_log) < WAIT_HINT_WINDOW:
            return
        wait = sum(w for w, _c in self._wait_log)
        total = wait + sum(c for _w, c in self._wait_log)
        if total <= 0 or wait / total <= WAIT_HINT_RATIO or total - wait <= 0:
            return
        self._wait_hinted = True
        self._wait_note = S.LAB_PACE_WAIT_HINT.format(times=total / (total - wait))

    def _refresh_status(self) -> None:
        """상태 글을 다시 만든다(끝난 학습이면 끝난 까닭 줄도)."""
        self._update_status()
        if self.state == "finished" and self._stopped_line:
            self.status.setText(self.status.text() + "\n" + self._stopped_line)

    def _on_state(self, msg: dict[str, Any]) -> None:
        self.state = str(msg.get("state"))
        self._update_status()
        self._stop_reason = str(msg.get("reason")) if self.state == "finished" else ""
        if self.state == "finished":
            reason = S.TRAIN_STOP_REASONS.get(str(msg.get("reason")), str(msg.get("reason")))
            self._stopped_line = S.LAB_STOPPED_BECAUSE.format(reason=reason)
            self.status.setText(self.status.text() + "\n" + self._stopped_line)
        self._update_buttons()

    def _on_applied(self, msg: dict[str, Any]) -> None:
        if not msg.get("ok"):
            # 학습 프로세스가 거절한 까닭(설정 검사 글, 그쪽 예외 글일 수도 있음): 다음 할 일과 묶음 단추를 함께(설계 D26)
            show_error(self, S.LAB_CHANGE_FAILED, str(msg.get("message", "")), next_step=S.ERR_SETTINGS_NEXT)
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
            if env_changed:  # 몸 구조 타일: 이 세대 뒤(gen + 1부터)는 바뀐 환경으로 센다
                add = getattr(self.dashboard.tile("structure"), "add_env_version", None)
                if callable(add):
                    add(gen, self.spec)
            self._check_structure_notice()
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
        # 학습 프로세스의 오류 글(traceback은 client가 오류 기록에 남김). OS 오류 원문이 섞였으면 그 OS 오류로 까닭과
        # 「다음에 할 일」을 고른다(원문은 「자세히」에).
        found = training_error(message, getattr(self.client, "last_error_detail", ""))
        if found is not None:
            show_error(self, S.LAB_ERROR_TITLE, found[0], message=found[1])
        else:
            show_error(self, S.LAB_ERROR_TITLE, message)
        self._set_start_notice("")
        if fatal:
            self.state = "closed"
            self._update_buttons()

    def _on_closed(self) -> None:
        self._set_start_notice("")
        self.state = "closed" if self.run_dir is not None else "idle"
        self.queue.clear()
        self._drop_scene_jobs()
        self._pace_note = ""  # 관전 방식 알림은 처음 학습 한 번 동안만(설계 D24)
        self._update_status()
        self._update_buttons()

    def _update_status(self) -> None:
        if not self.rows:
            text = S.LAB_STATE_NAMES.get(self.state, self.state)
            for note in (self._session_note, self._map_notice, self._build_note, self._pace_note, self._wait_note,
                         self._preset_note):
                if note:
                    text += "\n" + note
            self.status.setText(text)
            return
        last = self.rows[-1]
        champ = last.get("champion_score", "")
        try:
            champ_text = f"{float(champ):.2f}" if champ not in ("", None) else S.LAB_STATUS_CHAMP_PENDING
            evals = int(float(last.get("evaluations", 0)))
        except (TypeError, ValueError):
            champ_text, evals = S.LAB_STATUS_CHAMP_PENDING, 0
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
        if self._build_note:
            text += "\n" + self._build_note
        if self._pace_note:
            text += "\n" + self._pace_note
        if self._wait_note:
            text += "\n" + self._wait_note
        if self._grow_note:
            text += "\n" + self._grow_note
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
        tip = self.btn_continue.toolTip()
        if busy and tip != S.RVL_CONTINUE_BUSY_TIP:
            self._continue_tip = tip  # 도움말 풍선(v3.0 HELP_TIPS) 등 다른 곳이 단 글은 바쁨이 풀리면 되돌린다
            self.btn_continue.setToolTip(S.RVL_CONTINUE_BUSY_TIP)
        elif not busy and tip == S.RVL_CONTINUE_BUSY_TIP:
            self.btn_continue.setToolTip(getattr(self, "_continue_tip", ""))
        self.btn_step.setEnabled(paused)
        self.btn_end.setEnabled(active)
        # 6.2.0 U9: 꺼진 까닭은 풍선에만(style.set_enabled: toolTip()은 도움말 글 그대로)
        style.set_enabled(self.btn_apply, paused, S.LAB_WHY_PAUSED_ONLY)
        style.set_enabled(self.btn_reload, paused and self.env_path is not None, S.LAB_WHY_PAUSED_ONLY if not paused else "")
        self.btn_edit.setEnabled(self.env_text is not None and (paused or not active))
        for w in (self.method, self.seed, self.popsize, self.sigma, self.workers, self.env_combo, self.ipop,
                  self.ga_elites, self.ga_crossover, self.ga_mut_rate, self.ga_mut_sigma, self.ga_decay,
                  self.sa_neighbors, self.sa_step, self.sa_temp, self.sa_cooling,
                  self.brain_combo, self.grow_start, self.grow_every):  # v6.1 「뇌」 줄도 학습 방법처럼(명세 A7)
            w.setEnabled(not active)
        self._update_stability()  # 예선 칸(qualify_entrants, qualify_trials)도 학습 중이면 끈다
        has_gen = self.generation > 0 and self.run_dir is not None
        style.set_enabled(self.btn_timeline, has_gen, S.LAB_WHY_NEED_GEN)
        style.set_enabled(self.btn_versus, has_gen, S.LAB_WHY_NEED_GEN)
        style.set_enabled(self.btn_flip, self.generation >= 2 or self._flip_timer.isActive(), S.LAB_WHY_NEED_TWO)
        style.set_enabled(self.btn_race, self.generation >= 2 and self.run_dir is not None, S.LAB_WHY_NEED_TWO)
        style.set_enabled(self.btn_video, self._last_video is not None, S.LAB_WHY_NO_VIDEO)
        style.set_enabled(self.btn_live, self.view_mode == "replay", S.LAB_WHY_LIVE)
        # 학습 중에만 보이는 안내 두 줄(묶음이 접힘선 아래라 접힘 예산과 무관)
        if self.locked_note.isHidden() == active:
            self.locked_note.setVisible(active)
            self.stop_apply_note.setVisible(active)
        self._update_roles(active)

    def _update_roles(self, active: bool) -> None:
        """(6.2.0 U7) 「진행」 묶음의 으뜸 단추 하나와 상태 점. 같으면 바로 돌아간다(세대마다 불림). 상태가 바뀌었을 때만
        state_changed를 낸다."""
        if active:
            primary = self.btn_pause if self.state in ("starting", "running") else self.btn_continue
        else:
            primary = self.btn_continue if self.run_dir is not None and self.generation > 0 else self.btn_start
        for button in (self.btn_start, self.btn_pause, self.btn_continue):
            style.set_role(button, "primary" if button is primary else "normal")
        self.state_dot.set_kind(self.state)
        if self.state != self._emitted_state:
            self._emitted_state = self.state
            self.state_changed.emit(self.state)

    # ------------------------------------------------------------------
    # 관전 재생
    # ------------------------------------------------------------------
    def _play_next(self) -> None:
        with GUI_PRIORITY.busy():  # 장면·집단을 캔버스에 넣는 동안 배경 작업을 쉬게 한다(v6.0)
            self._play_next_now()

    def _play_next_now(self) -> None:
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
            try:
                geometry, agents = self._scene_for(spec, msg["population"], Trial(int(seed), track))
            except EpisodeBuildError as exc:
                # 이 시험의 세계를 만들 수 없다(랜덤 값으로 모양이 틀어짐): 이 세대는 건너뛴다(학습이 기다리며 멈추지 않게)
                self.status.setText(self.status.text() + "\n" + S.RVL_SCENE_FAILED.format(reason=exc))
                self.client.send({"type": "played", "generation": int(msg["generation"])})
                self.playing = None
                QTimer.singleShot(0, self, self._play_next)
                return
        scores = np.asarray(msg["scores"])
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[np.argsort(-scores, kind="stable")] = np.arange(len(scores))
        failed = build_failures(agents)
        note = S.LAB_BUILD_FAILED_N.format(n=failed) if failed else ""
        if note != self._build_note:  # 조립 실패 개체가 있는 세대: 상태 줄에 한 번(없는 세대가 오면 지움)
            self._build_note = note
            self._update_status()
            if self.state == "finished" and self._stopped_line:
                self.status.setText(self.status.text() + "\n" + self._stopped_line)
        self.canvas.set_scene(geometry, agents, channels=list(spec.actuator_channels), view=view_box(spec.file))
        self.canvas.set_population(list(msg["trajectories"]), ranks=ranks, champion=int(msg["representative"]))
        self.canvas.set_overlay([])
        if self._inspect_index is not None or self._inspect_msg is not None:
            # 앞 세대 개체를 들여다보던 중: 캔버스가 그 개체를 더 재생하지 않으므로 검사 칸·레이를 닫는다(늦게 온 검사 결과는
            # _replay_done이 버린다). 뇌 원천은 아래에서 함께 정리한다.
            self._inspect_index = None
            self._inspect_msg = None
            self.inspect.hide()
            self.canvas.set_rays(None)
        if self._brain_play is not None and self._brain_play.get("inspect"):
            # 앞 세대에서 클릭 검사한 개체: 캔버스가 더는 그 개체를 재생하지 않는다(뇌는 이 세대 대표 ②로)
            self._brain_play = None
        self._set_follow(msg)
        self._live_started = time.monotonic()
        self._skip_end = _trajectory_end(msg, int(msg["representative"])) + SKIP_TAIL_S
        self._start_clock()
        self._live_brain_for(msg, spec)

    def _scene_for(self, spec: EnvSpec, population: np.ndarray, trial: Trial) -> tuple[Any, list[Any] | None]:
        assert self.layout is not None
        return build_scene(spec, self.layout, population, trial)

    def _next_scene(self) -> None:
        """기다리는 관전 세대의 장면을 하나씩 차례로 백그라운드에서 만든다(세대 순서를 지킨다). 「학습 먼저」로 재생 중이면
        지금 재생이 끝나기 조금 전까지 미룬다(_scene_later: _tick이 다시 부른다)."""
        if (self._scene_thread is not None or self._closing or not self._scene_jobs or self.spec is None
                or self.layout is None or self._scene_later()):
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

    def _scene_later(self) -> bool:
        """「학습 먼저」로 관전 세대를 재생하는 중이면 다음 장면은 아직 만들지 않는다(v6.0 검토 고침, 설계 T21 「마지막 하나만 장면을
        만든다」): 다음에 보일 장면이 이미 있거나, 지금 재생이 끝나기까지(실제 시간) SKIP_SCENE_LEAD_S(또는 마지막 장면 만들기
        시간의 두 배)보다 많이 남았다. 그 사이에 온 세대는 기다리는 자리를 바꿔 차지할 뿐 장면을 만들지 않는다. 늦게 시작해 재생
        끝까지 다 만들지 못하면 재생이 그만큼(장면 만들기 시간 안) 길어질 뿐이다(_skip_due는 만든 장면이 있어야 끝낸다)."""
        if self.pace != PACE_SKIP or self.view_mode != "live" or self.playing is None:
            return False
        if self.queue:
            return True
        return self._live_left() > max(SKIP_SCENE_LEAD_S, 2.0 * self._scene_build_s)

    def _live_left(self) -> float:
        """지금 관전 재생이 끝나기까지 남은 실제 시간(초): 다음 세대가 준비되어 있다고 칠 때(_skip_due의 규칙, 들여다보는
        중이면 가장 긴 궤적까지)."""
        speed = float(max(1, self.speed))
        left = (self.canvas.duration() + 0.5 - self.play_t) / speed
        if not self._inspecting_playing():
            left = min(left, (self._skip_end - self.play_t) / speed,
                       SKIP_MAX_VIEW - (time.monotonic() - self._live_started))
        return left

    def _scene_done(self, tag: Any, result: Any) -> None:
        seq, msg, spec = tag
        th = self._scene_thread
        if th is not None and th.seconds is not None and not isinstance(result, Exception):
            self._scene_build_s = th.seconds
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
                if self.pace == PACE_SKIP:  # 재생 대기열은 가장 새 세대 하나(앞서 만든 것이 남아 있으면 버린다)
                    for old in [m for m in self.queue if int(m["generation"]) < int(msg["generation"])]:
                        self.queue.remove(old)
                        self.client.send({"type": "played", "generation": int(old["generation"])})
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
            gen = int(self.playing["generation"])
            if self.pace == PACE_SKIP and self.generation > gen:  # 학습이 앞서 있다(설계 5.2)
                self.canvas.set_hud(S.LAB_HUD_LIVE_SKIP.format(gen=gen, t=self.play_t, speed=self.speed,
                                                               latest=self.generation))
            else:
                self.canvas.set_hud(S.LAB_HUD_LIVE.format(gen=gen, t=self.play_t, speed=self.speed))
            if self._scene_jobs and self._scene_thread is None:
                self._next_scene()  # 「학습 먼저」: 지금 재생이 끝나기 조금 전이면 다음 장면을 만들기 시작한다(_scene_later)
        elif self.view_mode == "fast":
            self.canvas.set_hud(S.LAB_HUD_FAST.format(gen=self._fast_last, t=self.play_t))
        elif self.view_mode == "replay":
            self.canvas.set_hud(S.LAB_HUD_REPLAY.format(label=self._replay_label, t=self.play_t))
        if self._inspect_index is not None:
            self.inspect.show_time(self.play_t)
        if self._brain_wanted():
            self._brain_check()
            self.dashboard.show_time(self.play_t)
        if self.play_t < self.canvas.duration() + 0.5 and not self._skip_due():
            return
        if self.view_mode == "live" and self.playing is not None:
            self.client.send({"type": "played", "generation": int(self.playing["generation"])})
            self._last_played = self.playing
            self.playing = None
            if self.queue:
                self._play_next()
            else:
                self._timer.stop()
                self._next_scene()  # 미뤄 둔 장면이 있으면 지금 만든다(화면이 쉰다)
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

    def _skip_due(self) -> bool:
        """「학습 먼저」에서 새 세대가 기다린다(장면까지 준비됨): 대표 시험 0번 궤적 끝 + SKIP_TAIL_S 또는 실제 시간 SKIP_MAX_VIEW초를
        봤으면 지금 관전 재생을 끝낸다(설계 4.12). 기다리는 세대가 없으면 지금처럼 가장 긴 궤적까지. 타임라인 「다시 보기」 등에는
        쓰지 않는다."""
        if self.pace != PACE_SKIP or self.view_mode != "live" or self.playing is None or not self.queue:
            return False
        if self._inspecting_playing():
            return False  # 이 세대의 개체를 들여다보는 중: v3.1처럼 가장 긴 궤적까지 본다(학습은 기다리지 않는다)
        return self.play_t >= self._skip_end or time.monotonic() - self._live_started >= SKIP_MAX_VIEW

    def _inspecting_playing(self) -> bool:
        """재생 중인 관전 세대의 개체를 클릭해 들여다보는 중이다(검사 칸이 보이거나 그 재생을 기다림)."""
        return self._inspect_msg is not None and self._inspect_msg is self.playing

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
        self._replay_view = view_box(jobs[-1][0].file) if jobs else None  # 이 재생의 기본 보기(_replay_done이 씀)
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
        if kind == "inspect" and not self._inspect_shown():
            self._inspect_msg = None
            return  # 그 개체의 세대가 캔버스에서 사라졌다(다음 세대 재생, 고속으로 바꿈 등): 검사 결과를 붙이지 않는다
        if isinstance(result, Exception):
            # 그 몸 수치로 조립할 수 없는 개체(v2 설계 4.9)는 그렇다고 앞에 붙인다. 모르는 예외 글은 「자세히」로(v3 설계 D26)
            built = S.PLAY_BODY_BUILD_FAILED + "\n" + str(result) if isinstance(result, EpisodeBuildError) else None
            show_error(self, S.LAB_ERROR_TITLE, result, message=built)
            return
        self._adopt_play_brain(seq, result[-1], inspect=kind == "inspect")
        if kind == "inspect":
            index = info
            self._inspect_index = index
            held = self._inspect_absent
            self.inspect.set_replay(index, result[0], held[1] if held is not None and held[0] == seq else frozenset())
            self.inspect.show()
            rep = result[0]
            self.canvas.set_overlay([(rep.trajectory, rep.agent, COLOR_NOW, S.LAB_LABEL_PICKED)])  # 6.2.0 U8
            self.canvas.set_rays(rep.rays, rep.trajectory.dt)
            if self.view_mode != "live" or self.playing is None:  # 재생이 끝난 세대를 클릭했으면 다시 돌린다
                self._start_clock()
            return
        reps: list[Replay] = result
        self.canvas.set_scene(reps[0].geometry, None, channels=reps[0].channels, view=self._replay_view)
        self.canvas.set_population([])
        colors = [COLOR_OLD, COLOR_NOW] if len(reps) > 1 else [COLOR_NOW]
        self._last_video = (list(zip(reps, colors, info, strict=True)), self._replay_label)
        style.set_enabled(self.btn_video, True)  # 6.2.0 U9: 다시 보기가 하나 생겼다
        self.canvas.set_overlay([(r.trajectory, r.agent, c, label)
                                 for r, c, label in zip(reps, colors, info, strict=True)])
        self.loop_overlay = True
        self._start_clock()

    def save_video(self, path: Path | None = None) -> Path | None:
        """지금 보고 있는 재생(타임라인 세대 대표, 1세대 vs 지금, 고속 모드 챔피언)을 내보낸다.

        path가 없으면 내보내기 대화 상자(MP4·GIF·폰 재생 파일, 고정·따라가기 카메라, 내보내기 자식 프로세스: v3.1 설계 D48·D49).
        path를 주면 v3.0처럼 이 프로세스에서 GIF 하나를 쓴다(따라가기 카메라, 진행 창)."""
        from PySide6.QtWidgets import QApplication, QProgressDialog

        from evolab.app.export.video import export_gif
        from evolab.runs.store import sanitize_folder_name

        last = getattr(self, "_last_video", None)
        if not last:
            QMessageBox.information(self, S.VIDEO_SAVE_TITLE, S.VIDEO_NOTHING)
            return None
        items, label = last
        if path is None:
            from evolab.app.export import job as export_job
            from evolab.app.export.dialog import choose_and_export
            from evolab.spec.io import env_hash

            # 폰 재생 파일 정보 줄: 실행 이름, 세대·환경(이 재생을 요청할 때의 세대와 환경 버전, 없으면 지금 환경)
            meta: dict[str, str] = {}
            if self.run_dir is not None:
                meta[export_job.META_RUN] = self.run_dir.name
            asked = getattr(self, "_replay_brain", None)
            spec = asked[4] if asked is not None else self.spec
            if asked is not None and asked[2] is not None:
                meta[export_job.META_GENERATION] = str(asked[2])
            if spec is not None:
                meta[export_job.META_ENV] = str(spec.file.name)
                meta[export_job.META_ENV_HASH] = env_hash(spec.file)[:16]
            folder = (self.run_dir / "exports") if self.run_dir is not None else paths.data_dir()
            out = choose_and_export(self, kind="replay", items=items, title=label,
                                    default_name=sanitize_folder_name(label), meta=meta,
                                    view=getattr(self, "_replay_view", None), folder=folder, settings=self.settings)
            if out is not None:
                self.status.setText(self.status.text() + "\n" + S.EXP_DONE.format(path=out))
            return out
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
            show_error(self, S.VIDEO_SAVE_TITLE, exc)
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
        self._update_buttons()  # 6.2.0 U9: 「실시간 보기」가 켜진다
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
        self._update_buttons()  # 6.2.0 U9: 「실시간 보기」가 켜진다
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
        self._update_buttons()  # 6.2.0 U9: 「실시간 보기」가 꺼진다

    def _inspect_shown(self) -> bool:
        """클릭 검사한 개체의 관전 세대가 아직 캔버스에 있다(재생 중이거나, 재생이 끝나 그대로 남음)."""
        m = self._inspect_msg
        if m is None or self.view_mode != "live":
            return False
        return self.playing is m or (self.playing is None and self._last_played is m)

    def _inspect(self, index: int) -> None:
        msg = self.playing
        if msg is None and self.view_mode == "live":
            msg = self._last_played  # 재생이 끝나 화면에 남은 세대
        if msg is None or self.layout is None:
            return
        self._inspect_msg = msg  # 이 세대를 재생하는 동안은 「학습 먼저」도 재생을 자르지 않는다(_skip_due)
        seed, track = msg["trial"]
        gen = int(msg["generation"])
        spec = self.spec_for(msg.get("env_version"), gen)
        genome = np.asarray(msg["population"][index], dtype=np.float64)
        try:
            absent = absent_inputs(spec, self.layout, genome)
        except (ValueError, KeyError):  # 몸 수치를 읽을 수 없는 유전자: 재생이 같은 오류를 알린다
            absent = frozenset()
        self._next_brain = (S.GMAP_LABEL_INDIVIDUAL.format(gen=gen, index=index), gen)
        self._run_replays(("inspect", index), [(spec, self.layout, genome, Trial(int(seed), track), True)])
        self._inspect_absent = (self._replay_tag, absent)

    def _close_inspect(self) -> None:
        self._inspect_index = None
        self._inspect_msg = None
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
    def _set_structure_envs(self) -> None:
        """「몸 구조」 타일에 이 실행의 환경 버전 목록(run.json env_versions 순서)을 준다. 지금 환경과 같은 글이면 이미
        컴파일한 self.spec을, 아니면 글을 준다(타일이 쓸 때 컴파일). 기록을 읽을 수 없으면 지금 환경 하나."""
        tile = self.dashboard.tile("structure")
        setter = getattr(tile, "set_env_versions", None)
        if not callable(setter) or self.spec is None:
            return
        versions: list[tuple[int, EnvSpec | str]] = []
        records = [v for v in self.run_info.get("env_versions") or [] if isinstance(v, dict)]
        if self.run_dir is not None and records:
            from evolab.runs.store import RunStore, RunStoreError

            try:
                store = RunStore.open(self.run_dir)
                for v in records:
                    text = store.env_text(str(v["id"]))
                    versions.append((int(v.get("from_generation", 0)), self.spec if text == self.env_text else text))
            except (RunStoreError, OSError, KeyError, ValueError, TypeError):
                versions = []
        setter(versions or [(0, self.spec)])

    def _check_structure_notice(self) -> None:
        """구조 뒤집기가 꺼진 유전 알고리즘·담금질 실행을 구조 환경에서 이어 가면 LAB_GA_NO_STRUCTURE(실행마다 한 번)."""
        if self.run_dir is None:
            return
        key = str(self.run_dir)
        if key in self._structure_noted:
            return
        try:
            note = structure_notice(self.run_info.get("config") or {}, self.spec)
        except (TypeError, ValueError):
            return
        if note:  # 학습 프로세스의 알림처럼 상태 글 끝줄(세대가 와서 상태 글을 다시 만들어도 남는다)
            self._structure_noted.add(key)
            self._session_note = note
            self.status.setText(self.status.text() + "\n" + note)

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
                           [(job["spec"], self.layout, job["genome"], job["trial"], TRACE_BRAIN)], self,
                           yield_steps=True)  # 배경 작업: 화면이 바쁘면 스텝 사이에서도 쉰다(v6.0)
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
        elif which == "play" and self._brain_play is job:
            if isinstance(result, Exception):
                # 실패: 재생이 되풀이되는 동안 「계산하는 중」에 머물지 않게 ①을 놓는다(② 또는 ③으로 내려감)
                self._brain_play = None
                self._update_brain_source()
                return
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
        self._update_buttons()  # 6.2.0 U9: 「실시간 보기」가 켜진다
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
            # 실행 폴더 파일이 망가진 경우(KeyError 등)도 「다른 실행을 열거나 지우고 다시」가 다음 할 일. OS 오류는 explain이 고른다
            broken = None if os_cause(exc) else S.ERR_RUN_STORE_NEXT.format(records=S.TAB_RECORDS)
            show_error(self, S.LAB_ERROR_TITLE, exc, next_step=broken)
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

            self.dashboard.shutdown()  # 「몸 구조」 타일의 챔피언 몸 그림(배경 Episode)도 기다린다
            shutdown_race(self._race_dialog)
        finally:
            self._closing = False
            gc.unfreeze()  # freeze_long_lived를 되돌린다(테스트처럼 화면을 여러 번 만들어도 얼린 객체가 쌓이지 않게)


_PLATFORM_NOW: dict[str, Any] | None = None


def platform_now() -> dict[str, Any]:
    """이 프로세스의 runsetup.platform_info(0)(처음 한 번만 계산: 리눅스는 uname을 새 프로세스로 띄우고 윈도우는 WMI를
    물어 수십 ms가 걸린다). 프로그램이 도는 동안 바뀌지 않는다."""
    global _PLATFORM_NOW
    if _PLATFORM_NOW is None:
        from evolab.engine.runsetup import platform_info

        _PLATFORM_NOW = platform_info(0)
    return _PLATFORM_NOW


def freeze_long_lived() -> None:
    """지금 살아 있는 객체(불러온 모듈·클래스·함수, 화면 위젯, pyqtgraph 항목: 실험실을 띄운 뒤 9만 개 남짓)를 순환 GC가
    훑지 않게 옮긴다(gc.freeze). 순환 GC의 전체 수집은 추적하는 객체를 모두 훑느라 한 번에 45~70ms 동안 GIL을 쥔다(한가한
    4코어 리눅스, v1.2 측정). 실행을 열 때나 관전 중 배경 스레드에서 그것이 일어나면 이벤트 루프가 그만큼 멈춘다. 옮긴 뒤에는
    그 뒤에 만든 객체만 훑는다(전체 수집 0.1ms 아래). 먼저 한 번 수집해 쓰레기를 얼리지 않는다. 얼린 객체도 참조 수가 0이
    되면 그대로 풀린다(순환 참조만 수집하지 않음). LabView.shutdown이 gc.unfreeze로 되돌린다."""
    gc.collect()
    gc.freeze()


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


def _trajectory_end(msg: Mapping[str, Any], index: int) -> float:
    """관전 세대 메시지에서 개체 index의 시험 0번 궤적 길이(초). 궤적이 없으면(조립 실패 등) 0."""
    try:
        traj = msg["trajectories"][index]
    except (KeyError, IndexError, TypeError):
        return 0.0
    if traj is None or not getattr(traj, "frames", 0):
        return 0.0
    return float(traj.duration)


def preset_train_text(key: str, value: Any) -> str:
    """LAB_PRESET_TRAIN_APPLIED의 {what} 한 칸: 「학습 안정성」 칸 이름과 값(설계 3.1 H)."""
    if key == "qualify_entrants":
        return f"{S.LAB_QUALIFY} {S.LAB_QUALIFY_OFF if int(value) <= 1 else int(value)}"
    if key == "qualify_trials":
        return f"{S.LAB_QUALIFY_TRIALS} {style.fmt_count(int(value))}{S.LAB_UNIT_TRIALS}"
    if key == "restart_patience":
        if value is None:
            return f"{S.LAB_RESTART} {S.LAB_QUALIFY_OFF}"
        return f"{S.LAB_RESTART_PATIENCE} {style.fmt_count(int(value))}{S.LAB_UNIT_TRIALS}"
    return f"{S.LAB_RESTART_FROM} {S.RESTART_FROM_NAMES_SHORT.get(str(value), str(value))}"


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
    """직접 조종 기록(.evoplay, 예전 .json) 가운데 같은 환경(내용 해시)의 최고 점수. 데이터 폴더를 바꾸기 전 기록도 본다.

    조종할 출력이 없는 환경(글라이더: 직접 조종 창에서 지켜보기만)은 사람의 기록이 없으므로 None(예전에 저장한 것도 쓰지 않음)."""
    if not spec.outputs:
        return None
    try:
        from evolab.app.play.record import best_record_score
        from evolab.app.play.window import record_dirs

        return best_record_score(spec, record_dirs())
    except OSError:
        return None

"""처음 실행 안내: 내장 언덕차로 1세대를 보고 10세대 「1세대 대표와 지금 챔피언」까지 7단계 (v3 설계 1.2, 4.1, 5.3, 7.3).

| # | 키 | 종류 | 대상(연결표 키) | 어둡게 | 넘어가는 조건 |
| --- | --- | --- | --- | --- | --- |
| ① | welcome | 카드 | — | 예 | 「따라 하기 시작」(「나중에 볼게요」 → later) |
| ② | press_start | 누르기 | lab.start | 예 | 대상 clicked (준비: 실험실 탭, 쉬고 있으면 내장 언덕차 + 새 실행 기본값, 관전) |
| ③ | wait_gen1 | 기다림 | lab.canvas | 아니오 | 1세대 재생 시작(lab.playing). 글: 준비 중 → 1세대 계산 중(n초) → 15초 → 90초 |
| ④ | watch | 값 | lab.speed | 아니오 | 배속 ≥ 4 또는 「이 단계 건너뛰기」(2초 뒤), 30초 뒤 「4배로 바꾸기」. 이미 4배 이상이었으면 8초(읽을 틈) 뒤나 배속을 바꾸면 |
| ⑤ | progress | 기다림 + 다음 | lab.tile.trend | 아니오 | 세대 ≥ TOUR_GOAL 그리고 「다음」 |
| ⑥ | versus | 누르기(밝게) | lab.versus | 아니오 | 대상 clicked 또는 「이 단계 건너뛰기」 |
| ⑦ | done | 보기 | main.help_corner | 아니오 | 「끝내기」 |

- 안내는 학습을 사용자가 누른 진짜 「▶ 학습 시작」으로만 시작한다(자동 모드는 그 단추를 click()). 환경 파일·실행 폴더·에디터 문서를
  고치지 않는다. 바꾸는 것: 쉬고 있는 실험실의 환경(내장 언덕차를 읽어 넣음)과 설정 칸(새 실행 기본값), 관전 모드, 설정 키
  onboarding/*(자동 모드는 안 씀). 배속은 사용자가(또는 「4배로 바꾸기」 단추로) 바꾼다(D14).
- 상태(설정 onboarding/state): 단계마다 step:<키>, 끝 done · skipped · later, 실패하면 step:press_start. 다시 켜면 이어서 카드
  (이어서 = ② 단계부터: 창을 닫으면 학습도 끝나므로).
- 1세대 전에 학습 프로세스가 닫히면 실패 카드. 다만 사용자가 「학습 끝내기」를 눌러 닫은 것이면 실패가 아니라 ②로 돌아간다.
  실패 카드의 까닭은 학습 프로세스의 글에서 「학습을 시작할 수 없습니다: 」 머리를 떼고, OS 오류 원문이면 한국어 까닭으로 바꾸고
  (errors.training_error), 홈 폴더는 ~로 줄인다. 까닭 글이 오지 않았으면 실험실 오류 상자의 글, 그것도 없으면 ONB_FAILED_NO_DETAIL.
- 자동 시작은 run_gui()가 창을 띄운 뒤 offer()만 한다(설정 키가 없고 실행 기록이 없을 때). MainWindow()만 만드는 자체 점검·테스트·
  evolab play에는 없다. EVOLAB_NO_TOUR가 있으면 끈다(D12).
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from PySide6.QtCore import QObject, QSettings, QTimer
from PySide6.QtWidgets import QMainWindow, QWidget

from evolab import paths
from evolab.app import strings as S
from evolab.app.help import anchors
from evolab.app.help.coach import CoachOverlay, CoachStep, open_help_page, scroll_into_view

TOUR_ENV: Final = "hill_car"
"""paths.builtin_envs()에서 이 이름(stem)의 내장 예제(언덕 넘는 자동차, 설계 D8)."""
TOUR_GOAL: Final = 10
"""⑤ 단계 목표 세대(되돌릴 지점 4: 6으로 낮출 수 있음)."""
STATE_KEY: Final = "onboarding/state"
VERSION_KEY: Final = "onboarding/version"
TOUR_VERSION: Final = 1
"""안내를 크게 바꾸면 올린다(설정 onboarding/version)."""
NO_TOUR_ENV: Final = "EVOLAB_NO_TOUR"
STEP_KEYS: Final = ("welcome", "press_start", "wait_gen1", "watch", "progress", "versus", "done")
"""말풍선 「i/7」로 세는 일곱 단계(설계 4.1 표 순서)."""
RESUME_AT: Final = "press_start"
"""이어서 보기·실패 뒤 다시 시작하는 단계(D13)."""
FINAL_STATES: Final = ("done", "skipped", "later", "offered")
SLOW_AFTER_S: Final = 15.0
STUCK_AFTER_S: Final = 90.0
"""③: 누른 뒤 이만큼 지나면 느린 PC 글(15초), 「더 기다리기 / 문제 해결 보기」(90초, 「더 기다리기」마다 90초 더)."""
SKIP_AFTER_S: Final = 2.0
"""④: 1세대 재생이 시작되고 이만큼 뒤에 「이 단계 건너뛰기」가 켜진다."""
SPEED_HINT_AFTER_S: Final = 30.0
"""④: 이만큼 뒤에 「4배로 바꾸기」 단추가 나타난다."""
WATCH_SPEED: Final = 4
WATCH_READ_S: Final = 8.0
"""④: 배속이 이미 4배 이상이면 이만큼(1세대 설명을 읽을 틈, 어림값) 뒤에 넘어간다(설계 D10 「배속 ≥ 4」)."""
FLAT_RATIO: Final = 1.2
"""⑤: 목표 세대의 평균이 1세대 평균의 이 배보다 작으면 진행 줄을 ONB_PROGRESS_FLAT으로(평균이 음수여도 |1세대|의 20% 오름)."""
HINT_MS: Final = 10_000
"""상태 줄 알림(예전 사용자 안내, 학습 중, 나중에) 시간."""
TROUBLE_PAGE: Final = "trouble"
REPORT_REASON: Final = "tour"
SETTINGS_KEYS_TOUCHED: Final = ("onboarding/state", "onboarding/version", "lab/mode", "lab/speed", "lab/workers",
                                "lab/fast_interval")
"""안내 중 바뀔 수 있는 설정 키(T8): 안내 키와 실험실 보기 칸(관전·배속을 바꾸면 실험실이 보기 칸 넷을 함께 저장한다).
대시보드가 「보는 재미」 쪽이었으면 ⑤ 준비가 「학습 그래프」 쪽으로 바꾸며 dashboard/* 키도 저장한다."""


@dataclass
class TourTimes:
    """안내 시각(오버레이 시계, 초)과 숫자. 자동 모드면 끝난 뒤 overlay.times로 읽는다(자체 점검 M1)."""

    shown: float
    clicked: float | None = None
    """「▶ 학습 시작」을 누른 때."""
    gen1_playing: float | None = None
    """1세대 재생이 시작된 때."""
    goal_reached: float | None = None
    """세대가 TOUR_GOAL이 된 것을 본 때."""
    clicks: int = 0
    """말풍선 단추와 대상을 누른 수(자동 모드: 1세대까지 2번 = 「따라 하기 시작」 + 「▶ 학습 시작」)."""
    first_mean: float | None = None
    goal_mean: float | None = None


# ---------------------------------------------------------------------------
# 제안 조건
# ---------------------------------------------------------------------------
def tour_env_path() -> Path | None:
    """안내 예제(내장 언덕차) 파일. 없으면 None(설치가 망가짐)."""
    for path in paths.builtin_envs():
        if path.stem == TOUR_ENV:
            return path
    return None


def has_runs(runs_dir: Path) -> bool:
    """실행 기록(run.json이 있는 실행 폴더)이 하나라도 있는가. 숨긴 폴더(.으로 시작)는 세지 않는다."""
    try:
        with os.scandir(runs_dir) as it:
            for entry in it:
                if entry.name.startswith(".") or not entry.is_dir():
                    continue
                if os.path.isfile(os.path.join(entry.path, "run.json")):
                    return True
    except OSError:
        return False
    return False


def tour_disabled() -> bool:
    """EVOLAB_NO_TOUR가 비어 있지 않고 0이 아니면 안내를 자동으로 내지 않는다."""
    return os.environ.get(NO_TOUR_ENV, "").strip() not in ("", "0")


def tour_state(settings: QSettings) -> str:
    value = settings.value(STATE_KEY, "")
    return str(value).strip() if value is not None else ""


def should_offer(settings: QSettings, runs_dir: Path) -> Literal["welcome", "resume", "hint"] | None:
    """창을 띄운 뒤 무엇을 보일지(설계 4.1 offer).

    키 없음 + 실행 기록 없음 → "welcome", 키 없음 + 실행 기록 있음 → "hint"(예전 사용자: 상태 줄 알림 한 번), step:* → "resume",
    done·skipped·later·offered(그 밖) → None. EVOLAB_NO_TOUR가 있으면 늘 None.
    """
    if tour_disabled():
        return None
    state = tour_state(settings)
    if not state:
        return "hint" if has_runs(runs_dir) else "welcome"
    if state.startswith("step:"):
        return "resume"
    return None


def tour_settings(window: QWidget) -> QSettings | None:
    """안내 상태를 두는 설정(실험실과 같은 settings.ini)."""
    lab = getattr(window, "lab", None)
    settings = getattr(lab, "settings", None)
    return settings if isinstance(settings, QSettings) else None


def running_tour(window: QWidget) -> CoachOverlay | None:
    """그 창에서 돌고 있는 안내 오버레이."""
    for child in window.findChildren(CoachOverlay):
        if getattr(child, "tour", None) is not None and child.running:
            return child
    return None


def offer(window: QMainWindow, settings: QSettings | None = None) -> CoachOverlay | None:
    """run_gui()가 창을 띄운 뒤 한 번 부른다(QTimer.singleShot(0, ...)). 환영 카드·이어서 카드를 띄우거나 예전 사용자에게 상태 줄
    알림을 한 번 보인다(그 뒤 상태 offered). 띄운 오버레이(없으면 None)."""
    settings = settings if settings is not None else tour_settings(window)
    lab = getattr(window, "lab", None)
    if settings is None or lab is None:
        return None
    choice = should_offer(settings, paths.runs_dir())
    if choice == "welcome":
        return start_tour(window, settings)
    if choice == "hint":
        _save_state(settings, "offered")
        window.statusBar().showMessage(S.ONB_EXISTING_USER_HINT.format(menu=S.MENU_HELP), HINT_MS)
        return None
    if choice == "resume" and not lab.client.active:
        return start_tour(window, settings, resume=True)
    return None


def _save_state(settings: QSettings, state: str) -> None:
    settings.setValue(STATE_KEY, state)
    settings.setValue(VERSION_KEY, TOUR_VERSION)
    settings.sync()


def _mean(row: Any) -> float | None:
    try:
        value = float(row.get("score_mean"))
    except (AttributeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def is_flat(first: float, now: float) -> bool:
    """평균이 크게 오르지 않았는가: now < first × 1.2(first가 음수면 first + 0.2·|first|)."""
    return now < first + (FLAT_RATIO - 1.0) * abs(first)


def open_report(parent: QWidget | None, run_dir: Path | None, env_text: str | None) -> Path | None:
    """오류 보고 묶음 대화 상자(report.dialog.open_report_dialog, 설계 7.3). 까닭은 "tour"."""
    from evolab.app.report.bundle import BundleOptions
    from evolab.app.report.dialog import open_report_dialog

    return open_report_dialog(parent, BundleOptions(reason=REPORT_REASON, run_dir=run_dir, env_text=env_text))


# ---------------------------------------------------------------------------
# 안내
# ---------------------------------------------------------------------------
class _Tour(QObject):
    """안내 한 번: 단계 목록, 실험실 신호, 상태 저장, 자동 모드를 맡는다(오버레이는 그리기와 넘기기만)."""

    def __init__(self, window: QMainWindow, settings: QSettings | None, *, resume: bool = False, auto: bool = False,
                 until: Literal["gen1", "full"] = "full") -> None:
        super().__init__(window)
        self.window = window
        self.lab: Any = window.lab  # type: ignore[attr-defined]
        self.settings = settings
        self.auto = auto
        self.until = until
        self.resume = resume
        self.overlay: CoachOverlay | None = None
        self.times = TourTimes(shown=0.0)
        self.prepared = False
        self.started = False
        """학습 프로세스가 시작을 알렸다(③ 글: 준비 중 → 1세대 계산 중)."""
        self.clicked = False
        self.reason = ""
        """실패 카드에 보일 까닭."""
        self.last_error = ""
        self.stuck_at = STUCK_AFTER_S
        self.watch_speed0 = 1
        self._watch_entry = -2
        self._user_ended = False
        """1세대 전에 사용자가 「학습 끝내기」를 눌렀다(그 뒤의 닫힘은 실패가 아니다)."""
        self._auto_next_sent = False
        self._listening = False
        self.steps = self._build_steps()

    # -- 공통 ---------------------------------------------------------------
    def elapsed(self) -> float:
        return self.overlay.step_elapsed() if self.overlay is not None else 0.0

    def pop(self) -> int:
        """개체 수: ② 준비 전에 실험실이 쉬고 있으면 준비가 넣을 새 실행 기본값(환영 카드), 그 뒤에는 칸 값."""
        if not self.prepared and not self.lab.client.active:
            from evolab.engine.trainer import TrainConfig

            return int(TrainConfig.for_new_run().popsize)
        return int(self.lab.popsize.value())

    def trials(self) -> int:
        value = int(self.lab.trials.value())
        if value:
            return value
        spec = getattr(self.lab, "spec", None)
        try:
            return int(spec.file.evaluation.trials or 1)
        except AttributeError:
            return 1

    def workers(self) -> int:
        return int(self.lab.workers.value())

    # -- 단계 ---------------------------------------------------------------
    def _build_steps(self) -> list[CoachStep]:
        steps: list[CoachStep] = []
        if self.resume:
            steps.append(CoachStep(
                key="resume", title=S.ONB_RESUME_TITLE, text=lambda: S.ONB_RESUME_BODY, kind="welcome", dim=True,
                buttons=((S.ONB_RESUME_CONTINUE, lambda: self._go(RESUME_AT)),
                         (S.ONB_RESUME_RESTART, lambda: self._go("welcome")),
                         (S.ONB_RESUME_STOP, lambda: self._finish("skipped"))),
                counted=False))
        steps += [
            CoachStep(key="welcome", title=S.ONB_WELCOME_TITLE, text=lambda: S.ONB_WELCOME_BODY.format(pop=self.pop()),
                      kind="welcome", dim=True,
                      buttons=((S.ONB_WELCOME_START, self._next), (S.ONB_WELCOME_LATER, lambda: self._finish("later")))),
            CoachStep(key="press_start", text=lambda: S.ONB_STEP_PRESS_START.format(start=S.LAB_START, pop=self.pop()),
                      anchor="lab.start", kind="click", dim=True, prepare=self._prepare_start),
            CoachStep(key="wait_gen1", text=self._wait_text, anchor="lab.canvas", kind="wait", until=self._gen1_playing,
                      extra=self._wait_buttons),
            CoachStep(key="watch", text=self._watch_text, anchor="lab.speed",
                      kind="value", until=self._speed_set, prepare=self._prepare_watch, skip_step=True,
                      next_after_s=SKIP_AFTER_S, extra=self._watch_buttons),
            CoachStep(key="progress",
                      text=lambda: S.ONB_STEP_PROGRESS.format(pop=self.pop(), trend=S.DASH_TILE_TITLES["trend"],
                                                              goal=TOUR_GOAL),
                      anchor="lab.tile.trend", kind="next", until=self._goal_reached, prepare=self._prepare_progress,
                      status=self._progress_line),
            CoachStep(key="versus", text=lambda: S.ONB_STEP_VERSUS.format(versus=S.LAB_VERSUS), anchor="lab.versus",
                      kind="click", skip_step=True),
            CoachStep(key="done", text=lambda: S.ONB_STEP_DONE, anchor="main.help_corner", kind="welcome",
                      buttons=((S.ONB_FINISH, lambda: self._finish("done")),), escape="done"),
            CoachStep(key="failed", text=lambda: S.ONB_FAILED.format(reason=self.reason), kind="welcome",
                      buttons=((S.ERR_REPORT_BUTTON, self.report), (S.ONB_STUCK_HELP, self.open_trouble),
                               (S.ONB_CLOSE, lambda: self._finish("failed"))),
                      default=2, counted=False),
        ]
        return steps

    def main_steps(self) -> list[CoachStep]:
        return [s for s in self.steps if s.key in STEP_KEYS]

    def _go(self, key: str) -> None:
        if self.overlay is not None:
            self.overlay.go_to(key)

    def _next(self) -> None:
        if self.overlay is not None:
            self.overlay.next()

    def _finish(self, how: str) -> None:
        if self.overlay is not None:
            self.overlay.finish(how)

    # ② 학습 시작
    def _prepare_start(self) -> None:
        lab = self.lab
        anchors.show_screen(self.window, "lab")
        if not self.prepared and not lab.client.active:
            self.prepared = True
            if not self._use_tour_env():
                return
            if not lab.rb_spectate.isChecked():
                lab.rb_spectate.setChecked(True)
            if self.auto and self.until == "full":
                lab.speed_combo.setCurrentIndex(max(0, lab.speed_combo.findData(WATCH_SPEED)))
        scroll_into_view(lab.btn_start)

    def _use_tour_env(self) -> bool:
        """내장 언덕차를 새 실행 기본 설정으로 실험실에 넣는다(환경 파일은 읽기만 한다). 못 하면 실패 카드."""
        from evolab.engine.trainer import TrainConfig
        from evolab.spec.compiled import EnvFileError
        from evolab.spec.io import compile_env

        path = tour_env_path()
        text = None
        if path is not None:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except OSError:
                text = None
        if path is None or text is None:
            self._fail(S.ONB_NO_EXAMPLE.format(name=f"{TOUR_ENV}.evoenv"), card_only=True)
            return False
        try:
            config = TrainConfig.for_new_run(compile_env(text)).to_dict()
        except (EnvFileError, ValueError) as exc:
            self._fail(str(exc), card_only=True)
            return False
        # 복제한 실행처럼 설정을 함께 넣는다: 칸이 모두 새 실행 기본값(시드 1, 시험 수 = 환경 파일 값, 자동 재시작·예선 끔)이 된다
        if not self.lab.use_env_text(text, str(path), config, path):
            self._fail(S.ONB_NO_EXAMPLE.format(name=f"{TOUR_ENV}.evoenv"), card_only=True)
            return False
        return True

    # ③ 1세대 기다림
    def _gen1_playing(self) -> bool:
        lab = self.lab
        playing = lab.playing is not None or (lab.generation >= 1 and lab.view_mode != "live")
        if playing and self.times.gen1_playing is None and self.overlay is not None:
            self.times.gen1_playing = self.overlay.now()
        return playing

    def _wait_text(self) -> str:
        el = self.elapsed()
        if el >= self.stuck_at:
            return S.ONB_STEP_WAIT_STUCK.format(seconds=int(el))
        if el >= SLOW_AFTER_S:
            return S.ONB_STEP_WAIT_SLOW
        if not (self.started or self.lab.run_dir is not None):
            return S.ONB_STEP_WAIT_PROCESS.format(workers=self.workers())
        return S.ONB_STEP_WAIT_GEN1.format(pop=self.pop(), trials=self.trials(), seconds=int(el))

    def _wait_buttons(self) -> tuple[tuple[str, Callable[[], None]], ...]:
        if self.elapsed() < self.stuck_at:
            return ()
        return ((S.ONB_STUCK_WAIT, self._wait_more), (S.ONB_STUCK_HELP, self.open_trouble))

    def _wait_more(self) -> None:
        self.stuck_at = self.elapsed() + STUCK_AFTER_S

    # ④ 보기와 배속
    def _prepare_watch(self) -> None:
        entry = len(self.overlay.log) if self.overlay is not None else -1
        if entry != self._watch_entry:  # 단계에 들어올 때만(대상이 가려져 준비를 다시 할 때는 처음 배속을 그대로)
            self._watch_entry = entry
            self.watch_speed0 = int(self.lab.speed)
        scroll_into_view(self.lab.speed_combo)

    def _already_fast(self) -> bool:
        """④에 들어올 때 이미 4배 이상이었고 지금도 그렇다."""
        return self.watch_speed0 >= WATCH_SPEED and int(self.lab.speed) >= WATCH_SPEED

    def _watch_text(self) -> str:
        return S.ONB_STEP_WATCH_FAST if self._already_fast() else S.ONB_STEP_WATCH.format(speed=S.LAB_SPEED)

    def _speed_set(self) -> bool:
        """배속 ≥ 4(설계 D10). 들어올 때 이미 4배 이상이었으면 1세대 설명을 읽을 틈(WATCH_READ_S) 뒤나 배속을 바꾸면."""
        speed = int(self.lab.speed)
        if speed < WATCH_SPEED:
            return False
        if self.watch_speed0 < WATCH_SPEED:
            return True
        return speed != self.watch_speed0 or self.elapsed() >= WATCH_READ_S

    def _watch_buttons(self) -> tuple[tuple[str, Callable[[], None]], ...]:
        if int(self.lab.speed) >= WATCH_SPEED or self.elapsed() < SPEED_HINT_AFTER_S:
            return ()
        return ((S.ONB_SPEED_SET, self._set_speed),)

    def _set_speed(self) -> None:
        combo = self.lab.speed_combo
        combo.setCurrentIndex(max(0, combo.findData(WATCH_SPEED)))

    # ⑤ 진행
    def _prepare_progress(self) -> None:
        from evolab.app.lab.dashboard import PAGE_OF

        dash = self.lab.dashboard
        page = PAGE_OF.get("trend")
        if page is not None and getattr(dash, "page", page) != page:
            dash.set_page(page)

    def _goal_reached(self) -> bool:
        lab = self.lab
        reached = int(lab.generation) >= TOUR_GOAL
        if reached and self.times.goal_reached is None and self.overlay is not None:
            self.times.goal_reached = self.overlay.now()
            rows = list(lab.rows)
            if rows:
                self.times.first_mean, self.times.goal_mean = _mean(rows[0]), _mean(rows[-1])
        if self.auto and reached and not self._auto_next_sent and self.overlay is not None:
            self._auto_next_sent = True
            QTimer.singleShot(0, self, self._auto_press_next)
        # 학습이 목표 전에 끝났으면(학습 끝내기·정지 조건) 「다음」으로 넘어갈 수 있게 한다
        return reached or (not lab.client.active and int(lab.generation) >= 1)

    def _progress_line(self) -> str:
        rows = list(self.lab.rows)
        if not rows:
            return ""
        first, now = _mean(rows[0]), _mean(rows[-1])
        if first is None or now is None:
            return ""
        gen = int(self.lab.generation)
        if gen >= TOUR_GOAL and is_flat(first, now):
            return S.ONB_PROGRESS_FLAT
        return S.ONB_PROGRESS_LINE.format(gen=min(gen, TOUR_GOAL), goal=TOUR_GOAL, first=first, now=now)

    # 실패
    def _before_gen1(self) -> bool:
        ov = self.overlay
        return ov is not None and ov.running and ov.current is not None and ov.current.key in ("press_start", "wait_gen1")

    def _fail(self, reason: str, card_only: bool = False) -> None:
        ov = self.overlay
        if ov is None or not ov.running:
            return
        self.reason = reason
        if self.auto:
            ov.finish("failed")
            return
        if ov.current is not None and ov.current.key == "failed":
            ov.text_label.setText(S.ONB_FAILED.format(reason=reason))
            return
        if card_only:
            QTimer.singleShot(0, self, lambda: self._go("failed"))  # 준비(prepare) 안에서 부르면 그 단계가 끝난 뒤 옮긴다
        else:
            ov.go_to("failed")
            QTimer.singleShot(0, self, self._close_lab_error_boxes)

    def failure_reason(self, message: str | None) -> str:
        """실패 카드의 까닭(ONB_FAILED의 {reason}): 학습 프로세스 글에서 「학습을 시작할 수 없습니다: 」 머리를 떼고(카드가 이미 그
        말을 한다), OS 오류 원문이면 한국어 까닭으로 바꾼다. 글이 없으면 실험실 오류 상자의 글, 그것도 없으면 ONB_FAILED_NO_DETAIL.
        원문은 오류 기록(errors.jsonl)과 오류 보고 묶음에 남는다."""
        from evolab.app import errors

        text = (message or "").strip()
        if not text:
            text = self._lab_box_message()
            if not text:
                return S.ONB_FAILED_NO_DETAIL
        found = errors.training_error(text, getattr(self.lab.client, "last_error_detail", ""))
        if found is not None:
            return _short_home(errors.reason(found[0]))
        template, rest = errors.split_training_message(text)
        if template == S.TRAIN_CANNOT_START:
            text = rest.strip() or S.ONB_FAILED_NO_DETAIL
        return _short_home(text)

    def _lab_box_message(self) -> str:
        """이 창에 떠 있는 오류 상자(show_error) 가운데 마지막 것의 「무엇이 잘못됐나」 글."""
        from evolab.app.errors import open_error_boxes

        for box in reversed(open_error_boxes()):
            if self._in_window(box):
                friendly = getattr(box, "friendly", None)
                return str(getattr(friendly, "message", "") or "")
        return ""

    def _in_window(self, widget: QWidget) -> bool:
        w = widget.parentWidget()
        while w is not None and w is not self.window:
            w = w.parentWidget()
        return w is self.window

    def _close_lab_error_boxes(self) -> None:
        """실패 카드가 같은 까닭과 「오류 보고 묶음 만들기…」를 보이므로, 같은 실패로 실험실이 띄운 오류 상자는 닫는다(상자가 창을
        막아 카드를 누를 수 없게 되지 않게). 오류 기록(shown)은 상자를 띄울 때 이미 남았다."""
        from evolab.app.errors import open_error_boxes

        for box in open_error_boxes():
            if self._in_window(box):
                box.close()

    def report(self) -> None:
        open_report(self.window, self.lab.run_dir, self.lab.env_text)

    def open_trouble(self) -> None:
        open_help_page(TROUBLE_PAGE, None, self.window)

    # -- 오버레이와 실험실 신호 --------------------------------------------------
    def attach(self, overlay: CoachOverlay) -> None:
        self.overlay = overlay
        overlay.tour = self  # type: ignore[attr-defined]
        overlay.times = self.times  # type: ignore[attr-defined]
        overlay.step_changed.connect(self._on_step)
        overlay.target_clicked.connect(self._on_target_clicked)
        overlay.finished.connect(self._on_finished)
        client = self.lab.client
        client.started.connect(self._on_started)
        client.failed.connect(self._on_failed)
        client.closed.connect(self._on_closed)
        # 「학습 끝내기」: pressed는 clicked(학습을 끝내고 클라이언트를 닫음)보다 먼저 온다. 누른 채 단추 밖에서 놓으면(clicked
        # 없음) released 뒤에 표시를 지운다.
        self.lab.btn_end.pressed.connect(self._on_end_pressed)
        self.lab.btn_end.released.connect(self._on_end_released)
        self._listening = True

    def _detach(self) -> None:
        if not self._listening:
            return
        self._listening = False
        client = self.lab.client
        for signal, slot in ((client.started, self._on_started), (client.failed, self._on_failed),
                             (client.closed, self._on_closed), (self.lab.btn_end.pressed, self._on_end_pressed),
                             (self.lab.btn_end.released, self._on_end_released)):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    def _on_started(self, _msg: object = None) -> None:
        self.started = True
        self._user_ended = False

    def _on_end_pressed(self) -> None:
        if self._before_gen1():
            self._user_ended = True

    def _on_end_released(self) -> None:
        # clicked(학습 끝내기)는 released 바로 뒤에 같은 차례로 돈다: 다음 차례에도 학습 중이면 끝내지 않은 것
        QTimer.singleShot(0, self, self._end_not_clicked)

    def _end_not_clicked(self) -> None:
        if self._user_ended and self.lab.client.active:
            self._user_ended = False

    def _on_failed(self, message: str, fatal: bool) -> None:
        self.last_error = str(message)
        if fatal and self._before_gen1() and self.clicked:
            self._fail(self.failure_reason(self.last_error))

    def _on_closed(self) -> None:
        if not (self._before_gen1() and self.clicked):
            return
        if self._user_ended and not self.last_error:
            self._back_to_start()  # 사용자가 1세대 전에 「학습 끝내기」를 눌렀다: 실패가 아니다
            return
        self._fail(self.failure_reason(self.last_error))

    def _back_to_start(self) -> None:
        """②(「▶ 학습 시작」 누르기)로 돌아간다(환경·설정은 이미 준비됨)."""
        self.clicked = self.started = self._user_ended = False
        ov = self.overlay
        if ov is not None and ov.running:
            ov.go_to(RESUME_AT)

    def _on_target_clicked(self, key: str) -> None:
        if key != "press_start":
            return
        self.clicked = True
        if self.overlay is not None:
            self.times.clicked = self.overlay.now()
        if not self.lab.client.active:
            # 시작하지 못함(설정 오류 상자를 이미 보임): 실패 카드(까닭 = 그 상자의 글)
            QTimer.singleShot(0, self, lambda: self._fail(self.failure_reason(self.last_error)))

    def _on_step(self, key: str) -> None:
        ov = self.overlay
        if ov is None:
            return
        self.times.clicks = ov.clicks
        if not self.auto and self.settings is not None:
            if key in STEP_KEYS and key != "welcome":
                _save_state(self.settings, f"step:{key}")
            elif key == "failed":
                _save_state(self.settings, f"step:{RESUME_AT}")
        if self.auto:
            self._auto_step(key)

    def _on_finished(self, how: str) -> None:
        ov = self.overlay
        if ov is not None:
            self.times.clicks = ov.clicks
        self._detach()
        if not self.auto and self.settings is not None:
            _save_state(self.settings, f"step:{RESUME_AT}" if how == "failed" else how)
            if how == "later":
                self.window.statusBar().showMessage(
                    S.ONB_LATER_HINT.format(menu=S.MENU_HELP, item=S.MENU_HELP_TOUR), HINT_MS)

    # -- 자동 모드(자체 점검 M1, D16) -------------------------------------------
    def _auto_step(self, key: str) -> None:
        ov = self.overlay
        if ov is None:
            return
        if key == "welcome":
            QTimer.singleShot(0, self, lambda: ov.press(S.ONB_WELCOME_START))
        elif key in ("press_start", "versus"):
            QTimer.singleShot(0, self, lambda: self._auto_click(key))
        elif key == "watch":
            if self.until == "gen1":
                ov.finish("done")
            else:
                if int(self.lab.speed) < WATCH_SPEED:
                    self._set_speed()  # 값 조건이 참이 되어 다음 확인에서 넘어간다
                else:
                    QTimer.singleShot(0, self, ov.next)
        elif key == "done":
            QTimer.singleShot(0, self, lambda: ov.press(S.ONB_FINISH))
        elif key == "failed":
            ov.finish("failed")

    def _auto_click(self, key: str) -> None:
        ov = self.overlay
        if ov is None or not ov.running or ov.current is None or ov.current.key != key:
            return
        target = ov.target()
        click = getattr(target, "click", None)
        if callable(click) and target is not None and target.isEnabled():
            click()  # QAbstractButton.click(): 진짜 clicked 신호(exe에는 QtTest가 없다)
        elif key == "versus":
            ov.next()

    def _auto_press_next(self) -> None:
        ov = self.overlay
        if ov is not None and ov.running and ov.current is not None and ov.current.key == "progress":
            ov.next()


def _short_home(text: str) -> str:
    """글 속 홈 폴더 경로를 ~로 줄인다(OS 오류 원문의 repr처럼 \\를 두 번 쓴 꼴도)."""
    home = str(Path.home())
    if len(home) <= 1:
        return text
    for form in (home.replace("\\", "\\\\"), home):
        text = re.sub(re.escape(form) + r"(?=$|[\\/'\"\s])", "~", text)
    return text


def tour_steps(window: QMainWindow, settings: QSettings | None = None, *, resume: bool = False) -> list[CoachStep]:
    """일곱 단계(①~⑦)의 CoachStep. 글은 그 창의 실제 칸 값(개체 수, 시험 수, 워커 수)으로 채워진다(T10)."""
    return _Tour(window, settings, resume=resume).main_steps()


def start_tour(window: QMainWindow, settings: QSettings | None, *, resume: bool = False, auto: bool = False,
               until: Literal["gen1", "full"] = "full") -> CoachOverlay | None:
    """안내를 띄운다(설계 7.3). 그 창에 이미 돌고 있으면 그 오버레이. 실험실이 학습 중이면 상태 줄에 ONB_BUSY만 보이고 None.

    resume=True면 이어서 카드(이어서 보기 = ② 단계부터 / 처음부터 / 그만 보기)부터. auto=True면 단추를 스스로 누르고(1세대까지 2번)
    until="gen1"이면 1세대 재생이 시작되면 끝, "full"이면 처음부터 4배로 ⑦까지 간다. 끝난 뒤 overlay.times: TourTimes.
    auto=True는 설정 onboarding/*을 쓰지 않는다.
    """
    lab = getattr(window, "lab", None)
    if lab is None:
        return None
    existing = running_tour(window)
    if existing is not None:
        existing.raise_()
        return existing
    if lab.client.active:
        window.statusBar().showMessage(S.ONB_BUSY.format(end=S.LAB_END), HINT_MS)
        return None
    if settings is None:
        settings = tour_settings(window)
    tour = _Tour(window, settings, resume=resume, auto=auto, until=until)
    overlay = CoachOverlay(window, tour.steps)
    tour.attach(overlay)
    tour.times.shown = overlay.now()
    overlay.start()
    return overlay


__all__ = ["FINAL_STATES", "NO_TOUR_ENV", "STATE_KEY", "STEP_KEYS", "TOUR_ENV", "TOUR_GOAL", "TourTimes", "has_runs",
           "is_flat", "offer", "open_report", "running_tour", "should_offer", "start_tour", "tour_env_path",
           "tour_settings", "tour_steps"]

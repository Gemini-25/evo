"""v3.0 A 묶음 처음 실행 안내 (v3 설계 1.2 D7~D16, 4.1, 5.3, 7.3, 8.1 T6~T10, 느린 T31).

- T6 자동 모드 until="gen1": 진짜 학습으로 환영 → 1세대 재생 ≤ 60초, 누름 2번, 단계 기록 순서, 실험실 환경 = 내장 언덕차, 설정 = 새 실행
  기본값(시드 1). 같은 실행으로 T8 안전(환경 파일·예전 실행 폴더 해시 그대로, 바뀐 설정 키 ⊆ 허용, 학습 중이면 ONB_BUSY만)도 본다.
- T7 should_offer 표, MainWindow()만 만들면 오버레이 없음, EVOLAB_NO_TOUR, 예전 사용자 알림, 이어서 카드.
- T8 창을 닫아도 상태가 남음, 가짜 치명 실패 → 실패 카드와 묶음 단추.
- T9 가짜 학습 클라이언트(학습 프로세스를 띄우지 않음)와 주입한 시계로 ③ 15·90초 글, ④ 배속·「4배로 바꾸기」, ⑤ 「다음」·진행 줄·
  평평함, ⑥ btn_versus.clicked.
- T10 읽을 양: ①+②+③(가장 긴 글) ≤ 200자, 말풍선 하나 ≤ 120자(실제 숫자로 채운 뒤).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from evolab import paths
from evolab.app import strings as S
from evolab.app.help import coach as C
from evolab.app.help import tour as T
from evolab.app.help.coach import CoachOverlay
from evolab.app.lab.client import TrainingClient

DATA = Path(__file__).resolve().parent / "data"
OLD_RUN = DATA / "v11_run_restart" / "run"
HILL = paths.builtin_env_dir() / "hill_car.evoenv"


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """실제 데이터·설정 폴더를 건드리지 않는다(설계 D5). 안내 자동 제안 끄개는 테스트마다 정한다."""
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.delenv(T.NO_TOUR_ENV, raising=False)
    return tmp_path


def make_window(qtbot: Any) -> Any:
    from evolab.app.main import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1280, 800)
    w.show()
    qtbot.waitExposed(w)
    return w


@pytest.fixture()
def window(qtbot: Any, isolated: Path) -> Any:
    w = make_window(qtbot)
    yield w
    w.lab.client.abandon(timeout=60)  # 실패한 테스트도 학습 프로세스를 남기지 않게


class FakeProc:
    """학습 프로세스 대신(띄우지 않음). TrainingClient가 부르는 것만."""

    exitcode = 0

    def __init__(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.alive = False

    def terminate(self) -> None:
        self.alive = False


@pytest.fixture()
def fake_client(monkeypatch: pytest.MonkeyPatch) -> list[FakeProc]:
    """TrainingClient.start를 가짜로: active만 참이 되고 메시지는 오지 않는다(테스트가 실험실 속성을 바꾼다)."""
    procs: list[FakeProc] = []

    def start(self: TrainingClient, request: Any) -> None:
        proc = FakeProc()
        procs.append(proc)
        self._proc = proc
        self._closed_sent = False

    real_shutdown = TrainingClient.shutdown

    def shutdown(self: TrainingClient, timeout: float = 20.0) -> None:
        # pytest-qt가 픽스처 정리보다 먼저 창을 닫는다: 가짜 프로세스를 먼저 끝내 실험실 정리가 120초를 기다리지 않게
        if isinstance(self._proc, FakeProc):
            self._proc.alive = False
        real_shutdown(self, timeout)

    monkeypatch.setattr(TrainingClient, "start", start)
    monkeypatch.setattr(TrainingClient, "shutdown", shutdown)
    yield procs
    for proc in procs:
        proc.alive = False


@pytest.fixture()
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [1000.0]
    monkeypatch.setattr(C, "CLOCK", lambda: now[0])
    return now


@pytest.fixture()
def no_modal(monkeypatch: pytest.MonkeyPatch) -> None:
    """실험실이 띄우는 막는 상자(오류 상자)를 테스트에서 닫힌 것으로."""
    for name in ("warning", "critical", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)


def click(window: Any, pos: QPoint) -> None:
    QTest.mouseClick(window.windowHandle(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)


def at(ov: CoachOverlay, key: str) -> bool:
    return ov.current is not None and ov.current.key == key


def bubble_text(ov: CoachOverlay) -> str:
    return ov.title_label.text() + ov.text_label.text()


def tree_hashes(*folders: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for folder in folders:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def settings_snapshot(settings: QSettings) -> dict[str, Any]:
    settings.sync()
    return {k: settings.value(k) for k in settings.allKeys()}


def changed_keys(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    return {k for k in set(before) | set(after) if before.get(k) != after.get(k)}


def to_press_start(qtbot: Any, window: Any) -> CoachOverlay:
    ov = T.start_tour(window, window.lab.settings)
    assert ov is not None and at(ov, "welcome")
    assert ov.press(S.ONB_WELCOME_START)
    qtbot.waitUntil(lambda: at(ov, "press_start") and not ov.hole.isNull(), timeout=3000)
    return ov


def to_wait(qtbot: Any, window: Any) -> CoachOverlay:
    """② 구멍 안 「▶ 학습 시작」을 진짜 창 이벤트로 눌러 ③으로(가짜 클라이언트)."""
    ov = to_press_start(qtbot, window)
    click(window, ov.hole.center())
    qtbot.waitUntil(lambda: at(ov, "wait_gen1"), timeout=3000)
    assert window.lab.client.active
    return ov


# ---------------------------------------------------------------------------
# T7 제안 조건
# ---------------------------------------------------------------------------
def test_should_offer_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(T.NO_TOUR_ENV, raising=False)
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    runs = tmp_path / "runs"
    assert T.should_offer(settings, runs) == "welcome"  # 폴더 없음
    runs.mkdir()
    (runs / "not_a_run").mkdir()
    (runs / ".deleting-x").mkdir()
    (runs / ".deleting-x" / "run.json").write_text("{}", encoding="utf-8")
    (runs / "stray.json").write_text("{}", encoding="utf-8")
    assert T.should_offer(settings, runs) == "welcome"  # run.json 없는 폴더·숨긴 폴더·파일은 실행 기록이 아니다
    (runs / "r1").mkdir()
    (runs / "r1" / "run.json").write_text("{}", encoding="utf-8")
    assert T.should_offer(settings, runs) == "hint"
    for state, want in (("step:watch", "resume"), ("step:press_start", "resume"), ("done", None), ("skipped", None),
                        ("later", None), ("offered", None)):
        settings.setValue(T.STATE_KEY, state)
        assert T.should_offer(settings, runs) == want, state
        assert T.should_offer(settings, tmp_path / "empty") == want, state
    settings.setValue(T.STATE_KEY, "step:watch")
    monkeypatch.setenv(T.NO_TOUR_ENV, "1")
    assert T.should_offer(settings, runs) is None
    monkeypatch.setenv(T.NO_TOUR_ENV, "0")
    assert T.should_offer(settings, runs) == "resume"


def test_main_window_alone_never_shows_tour(qtbot: Any, window: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """MainWindow()만 만드는 자체 점검·테스트·evolab play에는 안내가 없다(D12). 끄개가 있으면 offer()도 아무것도 안 한다."""
    qtbot.wait(300)
    assert not window.findChildren(CoachOverlay)
    assert T.tour_state(window.lab.settings) == ""
    monkeypatch.setenv(T.NO_TOUR_ENV, "1")
    assert T.offer(window) is None
    assert not window.findChildren(CoachOverlay) and T.tour_state(window.lab.settings) == ""
    monkeypatch.delenv(T.NO_TOUR_ENV)
    ov = T.offer(window)  # 키 없음 + 실행 기록 없음 → 환영 카드
    assert ov is not None and at(ov, "welcome") and ov.current.dim and ov.hole.isNull()
    assert ov.buttons() == [S.ONB_WELCOME_START, S.ONB_WELCOME_LATER]
    assert T.running_tour(window) is ov and T.start_tour(window, window.lab.settings) is ov  # 두 번 띄우지 않는다
    assert T.tour_state(window.lab.settings) == ""  # 환영 카드만으로는 상태를 쓰지 않는다


def test_existing_user_gets_status_hint_once(qtbot: Any, isolated: Path) -> None:
    run = paths.runs_dir() / "old"
    shutil.copytree(OLD_RUN, run)
    w = make_window(qtbot)
    assert T.offer(w) is None
    assert not w.findChildren(CoachOverlay)
    assert T.tour_state(w.lab.settings) == "offered"
    assert w.statusBar().currentMessage() == S.ONB_EXISTING_USER_HINT.format(menu=S.MENU_HELP)
    w.statusBar().clearMessage()
    assert T.offer(w) is None and w.statusBar().currentMessage() == ""  # 다음 실행에는 없다


def test_later_and_escape_are_remembered(qtbot: Any, window: Any) -> None:
    ov = T.start_tour(window, window.lab.settings)
    assert ov.press(S.ONB_WELCOME_LATER)
    assert ov.result == "later" and T.tour_state(window.lab.settings) == "later"
    assert window.statusBar().currentMessage() == S.ONB_LATER_HINT.format(menu=S.MENU_HELP, item=S.MENU_HELP_TOUR)
    assert T.should_offer(window.lab.settings, paths.runs_dir()) is None
    ov2 = to_press_start(qtbot, window)  # 메뉴 「처음 실행 안내 다시 보기」와 같은 길
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert ov2.result == "skipped" and T.tour_state(window.lab.settings) == "skipped"
    assert int(window.lab.settings.value(T.VERSION_KEY)) == T.TOUR_VERSION


# ---------------------------------------------------------------------------
# T6 + T8 진짜 학습(자동 모드, 1세대까지)
# ---------------------------------------------------------------------------
def test_auto_tour_to_gen1_with_real_training_is_safe(qtbot: Any, window: Any) -> None:
    from evolab.engine.trainer import TrainConfig

    lab = window.lab
    paths.install_examples()  # 사용자 환경 폴더(첫 실행 복사본)도 해시를 본다
    old_run = paths.runs_dir() / "v11_old"
    shutil.copytree(OLD_RUN, old_run)
    files_before = tree_hashes(paths.builtin_env_dir(), paths.user_env_dir(), old_run)
    assert files_before
    lab.workers.setValue(2)
    lab.seed.setValue(7)  # 사용자가 바꿔 둔 칸도 새 실행 기본값으로 돌아간다
    lab.popsize.setValue(10)
    settings_before = settings_snapshot(lab.settings)

    ov = T.start_tour(window, lab.settings, auto=True, until="gen1")
    assert ov is not None
    qtbot.waitUntil(lambda: ov.result is not None, timeout=180_000)
    times = ov.times
    assert ov.result == "done"
    seconds = times.gen1_playing - times.shown
    print(f"auto tour window->gen1 {seconds:.2f}s, click->gen1 {times.gen1_playing - times.clicked:.2f}s")
    assert seconds <= 60.0  # 설계 T6·9.3: 창 → 1세대 재생 ≤ 60초
    assert times.clicks == 2 and ov.clicks == 2  # 「따라 하기 시작」 + 「▶ 학습 시작」
    assert [k for _t, k in ov.log] == ["welcome", "press_start", "wait_gen1", "watch", "finish:done"]
    assert lab.playing is not None and lab.client.active
    # 실험실 환경 = 내장 언덕차, 설정 = 새 실행 기본값(시드 1)
    assert lab.env_path is not None and lab.env_path.resolve() == HILL.resolve()
    want = TrainConfig.for_new_run(lab.spec).to_dict()
    assert lab.config_from_form().to_dict() == want
    assert want["run_seed"] == 1 and want["popsize"] == 32 and want["method"] == "cmaes" and want["trials"] is None
    used = lab.run_info.get("config") or {}
    for key in ("method", "run_seed", "popsize", "sigma0", "trials", "restart_patience", "qualify_entrants",
                "max_generations"):
        assert used.get(key) == want[key], key
    assert lab.rb_spectate.isChecked()
    # T8: 환경 파일·예전 실행 폴더는 그대로, 새 실행은 보통 실행(기록에 남음)
    assert tree_hashes(paths.builtin_env_dir(), paths.user_env_dir(), old_run) == files_before
    assert lab.run_dir is not None and lab.run_dir.parent == paths.runs_dir() and lab.run_dir != old_run
    allowed = set(T.SETTINGS_KEYS_TOUCHED)
    assert changed_keys(settings_before, settings_snapshot(lab.settings)) <= allowed
    assert T.tour_state(lab.settings) == ""  # 자동 모드는 안내 상태를 쓰지 않는다
    # T8: 학습 중이면 안내를 시작하지 않고 ONB_BUSY만
    assert T.start_tour(window, lab.settings) is None
    assert window.statusBar().currentMessage() == S.ONB_BUSY.format(end=S.LAB_END)
    assert T.running_tour(window) is None
    lab.end_session()
    assert not lab.client.active


# ---------------------------------------------------------------------------
# T8 상태 저장, 실패 카드
# ---------------------------------------------------------------------------
def test_closing_window_mid_tour_keeps_state_and_resumes_at_press_start(qtbot: Any, isolated: Path) -> None:
    w = make_window(qtbot)
    ov = to_press_start(qtbot, w)
    assert T.tour_state(w.lab.settings) == "step:press_start"
    assert w.lab.env_path is not None and w.lab.env_path.resolve() == HILL.resolve()
    w.close()
    w2 = make_window(qtbot)
    assert T.should_offer(w2.lab.settings, paths.runs_dir()) == "resume"
    ov2 = T.offer(w2)
    assert ov2 is not None and ov2 is not ov and at(ov2, "resume") and ov2.counter.text() == ""
    assert bubble_text(ov2) == S.ONB_RESUME_TITLE + S.ONB_RESUME_BODY
    assert ov2.buttons() == [S.ONB_RESUME_CONTINUE, S.ONB_RESUME_RESTART, S.ONB_RESUME_STOP]
    assert ov2.press(S.ONB_RESUME_CONTINUE)  # 이어서 = ② 단계부터
    assert at(ov2, "press_start") and ov2.counter.text() == S.ONB_PROGRESS.format(i=2, n=7)
    ov2.finish("skipped")
    ov3 = T.start_tour(w2, w2.lab.settings, resume=True)  # 이어서 카드의 「처음부터」·「그만 보기」
    assert ov3.press(S.ONB_RESUME_RESTART) and at(ov3, "welcome")
    ov3.finish("skipped")
    ov4 = T.start_tour(w2, w2.lab.settings, resume=True)
    assert ov4.press(S.ONB_RESUME_STOP) and ov4.result == "skipped"
    assert T.tour_state(w2.lab.settings) == "skipped"


def test_fatal_failure_before_gen1_shows_failure_card(qtbot: Any, window: Any, fake_client: list[FakeProc],
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    lab = window.lab
    reports: list[tuple[Any, ...]] = []
    helps: list[tuple[Any, ...]] = []
    monkeypatch.setattr(T, "open_report", lambda *a: reports.append(a))
    monkeypatch.setattr(T, "open_help_page", lambda *a: helps.append(a) or True)
    ov = to_wait(qtbot, window)
    lab.client.failed.disconnect(lab._on_failed)  # 실험실의 막는 오류 상자 없이 안내만 본다
    lab.client.failed.emit("가짜 실패", True)
    assert at(ov, "failed") and not ov.current.dim and ov.hole.isNull()
    assert ov.text_label.text() == S.ONB_FAILED.format(reason="가짜 실패")
    assert ov.buttons() == [S.ERR_REPORT_BUTTON, S.ONB_STUCK_HELP, S.ONB_CLOSE] and ov.counter.text() == ""
    assert T.tour_state(lab.settings) == "step:press_start"
    assert ov.press(S.ERR_REPORT_BUTTON)
    assert reports == [(window, lab.run_dir, lab.env_text)]
    assert ov.press(S.ONB_STUCK_HELP)
    assert helps == [("trouble", None, window)]
    QTest.keyClick(window, Qt.Key.Key_Return)  # 실패 카드의 Enter = 「닫기」
    assert ov.result == "failed" and T.tour_state(lab.settings) == "step:press_start"


def test_failure_card_closes_the_lab_error_box_of_the_same_failure(qtbot: Any, window: Any,
                                                                   fake_client: list[FakeProc]) -> None:
    """통합(I): 실험실도 같은 치명 실패를 show_error 상자로 알린다. 그 상자는 창을 막으므로 실패 카드(같은 까닭, 같은 묶음 단추)가
    뜨면 닫는다."""
    from evolab.app import errors

    lab = window.lab
    ov = to_wait(qtbot, window)
    lab.client.failed.emit("가짜 실패", True)  # 실험실의 _on_failed도 연결된 채로
    assert at(ov, "failed") and ov.text_label.text() == S.ONB_FAILED.format(reason="가짜 실패")
    qtbot.waitUntil(lambda: not [b for b in errors.open_error_boxes() if b.parentWidget() is lab], timeout=3000)
    assert ov.running and at(ov, "failed")


def test_training_closed_before_gen1_shows_failure_card(qtbot: Any, window: Any, fake_client: list[FakeProc]) -> None:
    lab = window.lab
    ov = to_wait(qtbot, window)
    lab.client.failed.disconnect(lab._on_failed)
    lab.client.failed.emit("끊김", False)  # 치명이 아닌 알림은 실패 카드가 아니다
    assert at(ov, "wait_gen1")
    fake_client[-1].alive = False
    lab.client._finish()  # 학습 프로세스가 1세대 전에 닫힘(closed 신호)
    assert at(ov, "failed") and ov.text_label.text() == S.ONB_FAILED.format(reason="끊김")


def test_start_that_does_not_launch_shows_failure_card(qtbot: Any, window: Any, monkeypatch: pytest.MonkeyPatch,
                                                       no_modal: None) -> None:
    from evolab.engine.trainer import TrainConfig

    def bad(self: Any) -> None:
        raise ValueError("bad config")

    monkeypatch.setattr(TrainConfig, "validate", bad)
    ov = to_press_start(qtbot, window)
    click(window, ov.hole.center())
    qtbot.waitUntil(lambda: at(ov, "failed"), timeout=3000)
    assert not window.lab.client.active


def test_missing_tour_example_shows_failure_card(qtbot: Any, window: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    lab = window.lab
    lab.popsize.setValue(10)
    monkeypatch.setattr(T, "tour_env_path", lambda: None)
    ov = T.start_tour(window, lab.settings)
    assert ov.text_label.text() == S.ONB_WELCOME_BODY.format(pop=32)  # 준비가 넣을 새 실행 기본값으로 말한다
    assert ov.press(S.ONB_WELCOME_START)
    qtbot.waitUntil(lambda: at(ov, "failed"), timeout=3000)
    reason = S.ONB_NO_EXAMPLE.format(name=f"{T.TOUR_ENV}.evoenv")
    assert ov.text_label.text() == S.ONB_FAILED.format(reason=reason)
    assert lab.env_path is not None and lab.env_path.name == "lander.evoenv"  # 실험실은 그대로
    assert lab.popsize.value() == 10 and not lab.client.active


def test_open_report_uses_report_dialog_with_tour_reason(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from evolab.app.report import dialog

    seen: list[Any] = []
    monkeypatch.setattr(dialog, "open_report_dialog", lambda parent, opts: seen.append((parent, opts)) or None)
    assert T.open_report(None, tmp_path, "{}") is None
    (parent, opts), = seen
    assert parent is None and opts.reason == "tour" and opts.run_dir == tmp_path and opts.env_text == "{}"


# ---------------------------------------------------------------------------
# T9 가짜 클라이언트로 단계 넘기기
# ---------------------------------------------------------------------------
def test_steps_follow_fake_training_state(qtbot: Any, window: Any, fake_client: list[FakeProc], clock: list[float],
                                          monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lab = window.lab
    helps: list[tuple[Any, ...]] = []
    monkeypatch.setattr(T, "open_help_page", lambda *a: helps.append(a) or True)
    ov = to_wait(qtbot, window)
    t0 = clock[0]
    assert ov.text_label.text() == S.ONB_STEP_WAIT_PROCESS.format(workers=lab.workers.value())
    assert T.tour_state(lab.settings) == "step:wait_gen1" and ov.times.clicked == t0
    # ③ 학습 시작 알림(실행 폴더가 생김) 뒤: 1세대 계산 중 (n초)
    lab.run_dir = tmp_path
    clock[0] = t0 + 5
    trials = lab.spec.file.evaluation.trials
    qtbot.waitUntil(lambda: ov.text_label.text() == S.ONB_STEP_WAIT_GEN1.format(pop=32, trials=trials, seconds=5),
                    timeout=2000)
    clock[0] = t0 + 15
    qtbot.waitUntil(lambda: ov.text_label.text() == S.ONB_STEP_WAIT_SLOW, timeout=2000)
    assert ov.buttons() == [S.ONB_SKIP]
    clock[0] = t0 + 90
    qtbot.waitUntil(lambda: ov.text_label.text() == S.ONB_STEP_WAIT_STUCK.format(seconds=90), timeout=2000)
    assert ov.buttons() == [S.ONB_STUCK_WAIT, S.ONB_STUCK_HELP, S.ONB_SKIP]
    assert ov.press(S.ONB_STUCK_HELP) and helps == [("trouble", None, window)]
    assert ov.press(S.ONB_STUCK_WAIT)  # 90초 더
    qtbot.waitUntil(lambda: ov.text_label.text() == S.ONB_STEP_WAIT_SLOW and ov.buttons() == [S.ONB_SKIP], timeout=2000)
    clock[0] = t0 + 181
    qtbot.waitUntil(lambda: ov.text_label.text() == S.ONB_STEP_WAIT_STUCK.format(seconds=181), timeout=2000)
    # ③ → ④: 1세대 재생 시작
    lab.playing = {"generation": 1}
    qtbot.waitUntil(lambda: at(ov, "watch"), timeout=2000)
    assert ov.times.gen1_playing == clock[0]
    lab.playing = None
    t4 = clock[0]
    skip = ov.button(S.ONB_SKIP_STEP)
    assert skip is not None and not skip.isEnabled() and ov.button(S.ONB_SPEED_SET) is None
    clock[0] = t4 + 2.0
    qtbot.waitUntil(lambda: ov.button(S.ONB_SKIP_STEP).isEnabled(), timeout=2000)  # 재생 2초 뒤
    clock[0] = t4 + 29
    qtbot.wait(250)
    assert ov.button(S.ONB_SPEED_SET) is None
    clock[0] = t4 + 30
    qtbot.waitUntil(lambda: ov.button(S.ONB_SPEED_SET) is not None, timeout=2000)  # 30초 뒤 「4배로 바꾸기」
    # 배속 칸이 가려지면 준비(스크롤)를 다시 하지만 처음 배속은 그대로다: 2배는 아직 넘어가는 조건이 아니다.
    # (검토 수정: 「보기」 묶음이 패널 맨 위라 스크롤 0에서는 늘 보인다. 맨 아래로 스크롤해야 가려진다.)
    speed0 = ov.tour.watch_speed0
    lab.speed_combo.setCurrentIndex(lab.speed_combo.findData(2))
    area = lab.speed_combo.parentWidget()
    while not hasattr(area, "verticalScrollBar"):
        area = area.parentWidget()
    bar = area.verticalScrollBar()
    bar.setValue(bar.maximum())
    assert bar.value() > 0 and not C.shown(lab.speed_combo)
    qtbot.waitUntil(lambda: C.shown(lab.speed_combo), timeout=2000)  # 다시 준비가 스크롤해 보이게 함
    assert at(ov, "watch") and ov.button(S.ONB_SPEED_SET) is not None
    assert ov.tour.watch_speed0 == speed0 != 2  # 다시 준비해도 처음 배속은 그대로
    lab.dashboard.set_page("fun")  # ⑤ 준비가 「학습 그래프」 쪽으로 돌린다
    lab.rows = [{"score_mean": 40.0}]
    lab.generation = 1
    assert ov.press(S.ONB_SPEED_SET)
    assert lab.speed == 4
    qtbot.waitUntil(lambda: at(ov, "progress"), timeout=2000)
    # ⑤ 진행: 세대 9에서 「다음」 꺼짐, 10에서 켜짐, 진행 줄 = 가짜 row.score_mean
    assert lab.dashboard.page == "learning"
    trend = lab.dashboard.tile("trend")
    qtbot.waitUntil(lambda: not ov.hole.isNull(), timeout=2000)
    assert ov.hole.contains(trend.mapTo(window, trend.rect().center()))
    lab.rows = [{"score_mean": 40.0 + 6.0 * i} for i in range(9)]
    lab.generation = 9
    qtbot.waitUntil(lambda: ov.status_label.text() == S.ONB_PROGRESS_LINE.format(gen=9, goal=T.TOUR_GOAL, first=40,
                                                                                now=88), timeout=2000)
    assert not ov.button(S.ONB_NEXT).isEnabled()
    lab.rows.append({"score_mean": 100.0})
    lab.generation = 10
    qtbot.waitUntil(lambda: ov.button(S.ONB_NEXT).isEnabled(), timeout=2000)
    assert ov.status_label.text() == S.ONB_PROGRESS_LINE.format(gen=10, goal=10, first=40, now=100)
    assert at(ov, "progress")  # 「다음」을 눌러야 넘어간다
    assert ov.times.goal_reached == clock[0] and ov.times.first_mean == 40.0 and ov.times.goal_mean == 100.0
    lab.rows[-1] = {"score_mean": 47.0}  # 1세대의 1.2배 미만
    qtbot.waitUntil(lambda: ov.status_label.text() == S.ONB_PROGRESS_FLAT, timeout=2000)
    assert ov.press(S.ONB_NEXT)
    # ⑥ btn_versus.clicked로 끝
    assert at(ov, "versus") and not ov.current.dim
    assert ov.buttons() == [S.ONB_SKIP_STEP, S.ONB_SKIP]
    lab.btn_versus.clicked.emit()
    assert at(ov, "done") and ov.buttons() == [S.ONB_FINISH]
    assert T.tour_state(lab.settings) == "step:done"
    assert ov.press(S.ONB_FINISH)
    assert ov.result == "done" and T.tour_state(lab.settings) == "done"
    assert [k for _t, k in ov.log] == ["welcome", "press_start", "wait_gen1", "watch", "progress", "versus", "done",
                                       "finish:done"]


def test_watch_step_with_speed_already_fast_moves_on_after_reading(qtbot: Any, window: Any,
                                                                   fake_client: list[FakeProc],
                                                                   clock: list[float]) -> None:
    """④ 배속이 이미 4배 이상이면(설계 D10 「배속 ≥ 4」) 1세대 설명을 읽을 틈(WATCH_READ_S) 뒤에 넘어간다. 「4배로 바꾸기」는 없고
    글에서 「4배로 바꿔 보세요」를 뺀다. 검토 수정 전에는 「이 단계 건너뛰기」로만 넘어갔다(설계와 다름)."""
    lab = window.lab
    lab.speed_combo.setCurrentIndex(lab.speed_combo.findData(8))
    ov = to_wait(qtbot, window)
    lab.playing = {"generation": 1}
    qtbot.waitUntil(lambda: at(ov, "watch"), timeout=2000)
    lab.playing = None
    assert ov.text_label.text() == S.ONB_STEP_WATCH_FAST
    t4 = clock[0]
    clock[0] = t4 + T.WATCH_READ_S - 1
    qtbot.wait(250)
    assert at(ov, "watch") and ov.button(S.ONB_SPEED_SET) is None
    clock[0] = t4 + T.WATCH_READ_S
    qtbot.waitUntil(lambda: at(ov, "progress"), timeout=2000)
    # 학습이 목표 전에 끝나면 「다음」을 켠다(학습 끝내기·정지 조건)
    lab.generation = 3
    lab.rows = [{"score_mean": 1.0}] * 3
    fake_client[-1].alive = False
    lab.client._finish()
    qtbot.waitUntil(lambda: ov.button(S.ONB_NEXT).isEnabled(), timeout=2000)
    assert ov.status_label.text() == S.ONB_PROGRESS_LINE.format(gen=3, goal=10, first=1, now=1)
    assert at(ov, "progress")  # 1세대가 지난 뒤 닫힘은 실패가 아니다


def test_watch_step_already_fast_moves_on_when_speed_changes(qtbot: Any, window: Any, fake_client: list[FakeProc],
                                                             clock: list[float]) -> None:
    lab = window.lab
    lab.speed_combo.setCurrentIndex(lab.speed_combo.findData(4))
    ov = to_wait(qtbot, window)
    lab.playing = {"generation": 1}
    qtbot.waitUntil(lambda: at(ov, "watch"), timeout=2000)
    lab.playing = None
    qtbot.wait(250)
    assert at(ov, "watch")
    # 4배보다 낮추면 다시 「4배로 바꿔 보세요」 규칙(30초 뒤 단추)
    lab.speed_combo.setCurrentIndex(lab.speed_combo.findData(2))
    clock[0] += T.SPEED_HINT_AFTER_S
    qtbot.waitUntil(lambda: ov.button(S.ONB_SPEED_SET) is not None, timeout=2000)
    assert at(ov, "watch") and ov.text_label.text() == S.ONB_STEP_WATCH.format(speed=S.LAB_SPEED)
    lab.speed_combo.setCurrentIndex(lab.speed_combo.findData(8))
    qtbot.waitUntil(lambda: at(ov, "progress"), timeout=2000)


def test_flat_rule() -> None:
    assert T.is_flat(43.9, 52.0) and not T.is_flat(43.9, 228.8)
    assert T.is_flat(100.0, 119.9) and not T.is_flat(100.0, 120.0)
    assert T.is_flat(-100.0, -85.0) and not T.is_flat(-100.0, -79.0)  # 음수 평균: |1세대|의 20% 오름


# ---------------------------------------------------------------------------
# T10 읽을 양
# ---------------------------------------------------------------------------
def test_reading_amount(qtbot: Any, window: Any, fake_client: list[FakeProc], clock: list[float],
                        tmp_path: Path) -> None:
    lab = window.lab
    texts: dict[str, set[str]] = {}

    def note(ov: CoachOverlay) -> None:
        texts.setdefault(ov.current.key, set()).add(bubble_text(ov))

    ov = T.start_tour(window, lab.settings)
    note(ov)
    ov.press(S.ONB_WELCOME_START)
    qtbot.waitUntil(lambda: at(ov, "press_start") and not ov.hole.isNull(), timeout=3000)
    note(ov)
    click(window, ov.hole.center())
    qtbot.waitUntil(lambda: at(ov, "wait_gen1"), timeout=3000)
    t0 = clock[0]
    note(ov)
    lab.run_dir = tmp_path
    for dt in (5, 15, 99):
        clock[0] = t0 + dt
        qtbot.wait(250)
        note(ov)
    assert len(texts["wait_gen1"]) == 4
    lab.playing = {"generation": 1}
    qtbot.waitUntil(lambda: at(ov, "watch"), timeout=2000)
    note(ov)
    lab.playing = None
    clock[0] += T.SPEED_HINT_AFTER_S
    qtbot.waitUntil(lambda: ov.button(S.ONB_SPEED_SET) is not None, timeout=2000)
    note(ov)
    assert ov.press(S.ONB_SKIP_STEP)
    qtbot.waitUntil(lambda: at(ov, "progress"), timeout=2000)
    note(ov)
    lab.rows = [{"score_mean": 43.9}] * 9 + [{"score_mean": 50.0}]
    lab.generation = 10
    qtbot.waitUntil(lambda: ov.status_label.text() == S.ONB_PROGRESS_FLAT, timeout=2000)
    progress_status = ov.status_label.text()
    assert ov.press(S.ONB_NEXT)
    note(ov)
    lab.btn_versus.clicked.emit()
    note(ov)
    longest = {key: max(len(t) for t in values) for key, values in texts.items()}
    print("reading amount", longest, "status", len(progress_status))
    to_gen1 = longest["welcome"] + longest["press_start"] + longest["wait_gen1"]
    assert to_gen1 <= 200, to_gen1  # ①+②+③(가장 긴 글)
    assert set(longest) == set(T.STEP_KEYS)
    assert max(longest.values()) <= 120, longest  # 말풍선 하나(제목 + 본문)
    assert len(progress_status) <= 60  # ⑤ 진행 줄(본문 아래 흐린 줄)
    for card in (S.ONB_RESUME_TITLE + S.ONB_RESUME_BODY, S.ONB_FAILED.format(reason="x" * 40)):
        assert len(card) <= 120


# ---------------------------------------------------------------------------
# 느린: 자동 모드 full (설계 T31: 창 → 10세대 ≤ 150초)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_auto_full_tour_reaches_goal(qtbot: Any, window: Any) -> None:
    lab = window.lab
    ov = T.start_tour(window, lab.settings, auto=True, until="full")
    qtbot.waitUntil(lambda: ov.result is not None, timeout=400_000)
    t = ov.times
    print(f"auto full: window->gen1 {t.gen1_playing - t.shown:.2f}s, window->gen{T.TOUR_GOAL} "
          f"{t.goal_reached - t.shown:.2f}s, mean {t.first_mean:.1f} -> {t.goal_mean:.1f}, clicks {t.clicks}, "
          f"workers {lab.workers.value()}")
    assert ov.result == "done"
    assert [k for _t, k in ov.log] == ["welcome", "press_start", "wait_gen1", "watch", "progress", "versus", "done",
                                       "finish:done"]
    assert t.goal_reached - t.shown <= 150.0
    assert lab.speed == T.WATCH_SPEED and lab.view_mode == "replay"  # 「1세대 대표와 지금 챔피언」을 눌렀다
    lab.end_session()


# ---------------------------------------------------------------------------
# 검토 수정: 밝은 단계의 Esc·Enter는 글·숫자 칸과 다른 탭(에디터)의 것
# ---------------------------------------------------------------------------
def test_light_steps_leave_editor_and_entry_keys_alone(qtbot: Any, window: Any, fake_client: list[FakeProc],
                                                       clock: list[float], tmp_path: Path) -> None:
    from PySide6.QtWidgets import QAbstractSpinBox

    lab = window.lab
    ov = to_wait(qtbot, window)
    assert not ov.current.dim
    # ③ 에디터 탭 캔버스의 Esc는 캔버스가 받는다(안내는 계속)
    window.tabs.setCurrentWidget(window.editor)
    canvas = window.editor.canvas
    canvas.setFocus()
    got: list[int] = []
    orig = type(canvas).keyPressEvent

    def spy(self: Any, ev: Any) -> None:
        got.append(ev.key())
        orig(self, ev)

    type(canvas).keyPressEvent = spy  # type: ignore[method-assign]
    try:
        QTest.keyClick(canvas, Qt.Key.Key_Escape)
    finally:
        type(canvas).keyPressEvent = orig  # type: ignore[method-assign]
    assert got == [Qt.Key.Key_Escape] and ov.running and at(ov, "wait_gen1")
    assert T.tour_state(lab.settings) == "step:wait_gen1"
    window.tabs.setCurrentWidget(lab)
    # ⑤ 목표 세대에 닿아 「다음」이 켜져도 숫자 칸의 Enter는 숫자 칸 것
    lab.run_dir = tmp_path
    lab.playing = {"generation": 1}
    qtbot.waitUntil(lambda: at(ov, "watch"), timeout=3000)
    lab.playing = None
    lab.rows = [{"score_mean": 40.0}]
    lab.generation = 1
    lab.speed_combo.setCurrentIndex(lab.speed_combo.findData(4))
    qtbot.waitUntil(lambda: at(ov, "progress"), timeout=3000)
    lab.rows = [{"score_mean": 40.0 + 6.0 * i} for i in range(11)]
    lab.generation = 11
    qtbot.waitUntil(lambda: ov.button(S.ONB_NEXT).isEnabled(), timeout=3000)
    spins = [w for w in lab.findChildren(QAbstractSpinBox) if w.isVisible() and w.isEnabled()]
    assert spins
    spins[0].setFocus()
    QTest.keyClick(spins[0], Qt.Key.Key_Return)
    QTest.keyClick(spins[0], Qt.Key.Key_Escape)
    assert ov.running and at(ov, "progress")
    # 단추가 받은 Enter(글 칸이 아님)는 여전히 「다음」
    QTest.keyClick(lab.btn_versus, Qt.Key.Key_Return)
    assert at(ov, "versus")
    lab.btn_versus.clicked.emit()
    assert at(ov, "done")
    # ⑦ 다 본 뒤의 Esc는 건너뛰기가 아니라 끝(숫자 칸에서 초점을 뺀 뒤)
    spins[0].clearFocus()
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert ov.result == "done" and T.tour_state(lab.settings) == "done"


# ---------------------------------------------------------------------------
# 검토 수정: 1세대 전 「학습 끝내기」, 실패 카드의 까닭 글
# ---------------------------------------------------------------------------
def test_end_pressed_before_gen1_goes_back_to_press_start(qtbot: Any, window: Any, fake_client: list[FakeProc]) -> None:
    """③에서 사용자가 「학습 끝내기」를 누르면(진짜 창 클릭) 실패 카드가 아니라 ②로 돌아간다. 다시 누르면 ③."""
    lab = window.lab
    ov = to_wait(qtbot, window)
    assert lab.btn_end.isEnabled()
    # 누른 채 단추 밖에서 놓음(clicked 없음): 끝내지 않았으므로 표시를 지운다
    lab.btn_end.pressed.emit()
    lab.btn_end.released.emit()
    qtbot.waitUntil(lambda: not ov.tour._user_ended, timeout=2000)
    assert lab.client.active and at(ov, "wait_gen1")
    click(window, lab.btn_end.mapTo(window, lab.btn_end.rect().center()))
    qtbot.waitUntil(lambda: not lab.client.active, timeout=3000)
    qtbot.waitUntil(lambda: at(ov, "press_start") and not ov.hole.isNull(), timeout=3000)
    assert ov.running and S.ONB_FAILED.split("{")[0] not in ov.text_label.text()
    assert T.tour_state(lab.settings) == "step:press_start"
    click(window, ov.hole.center())
    qtbot.waitUntil(lambda: at(ov, "wait_gen1"), timeout=3000)
    assert lab.client.active and ov.text_label.text() == S.ONB_STEP_WAIT_PROCESS.format(workers=lab.workers.value())
    # 끝내기를 눌렀어도 학습 프로세스가 오류로 닫혔으면 실패 카드
    lab.btn_end.pressed.emit()
    lab.client.failed.disconnect(lab._on_failed)
    lab.client.failed.emit(S.SESSION_CRASHED.format(reason="가짜"), False)
    fake_client[-1].alive = False
    lab.client._finish()
    assert at(ov, "failed")


def _fatal(lab: Any, reason: str, detail: str = "") -> None:
    """학습 프로세스가 보내는 모양 그대로의 치명 오류(engine/session.py: TRAIN_CANNOT_START + traceback)."""
    lab.client._dispatch({"type": "error", "fatal": True, "message": S.TRAIN_CANNOT_START.format(reason=reason),
                          "detail": detail})


def test_failure_card_and_lab_box_explain_os_errors(qtbot: Any, window: Any, fake_client: list[FakeProc]) -> None:
    """권한 오류: 실험실 상자는 ERR_PERMISSION + ERR_PERMISSION_NEXT(원문은 「자세히」), 실패 카드는 머리 겹침도 원문도 없다."""
    from evolab.app import errors

    lab = window.lab
    ov = to_wait(qtbot, window)
    runs = str(Path.home() / "자료" / "runs")
    detail = f"Traceback (most recent call last):\nPermissionError: [Errno 13] Permission denied: {runs!r}\n"
    shown: list[Any] = []
    real = errors.show_error

    def spy(*args: Any, **kwargs: Any) -> Any:
        box = real(*args, **kwargs)
        shown.append(box.friendly)
        return box

    import evolab.app.lab.view as lab_view

    lab_view.show_error = spy  # type: ignore[assignment]
    try:
        _fatal(lab, f"[Errno 13] Permission denied: {runs!r}", detail)
    finally:
        lab_view.show_error = real  # type: ignore[assignment]
    friendly, = shown
    assert friendly.message == S.TRAIN_CANNOT_START.format(reason=S.ERR_PERMISSION)
    assert friendly.next_step == S.ERR_PERMISSION_NEXT.format(menu=S.MENU_SETTINGS)
    assert "[Errno 13]" in friendly.detail
    assert at(ov, "failed")
    assert ov.text_label.text() == S.ONB_FAILED.format(reason=S.ERR_PERMISSION)
    assert "[Errno" not in ov.text_label.text()


def test_failure_card_drops_repeated_prefix_and_home(qtbot: Any, window: Any, fake_client: list[FakeProc]) -> None:
    """나누지 못하는 OS 오류(이미 있음): 「학습을 시작할 수 없습니다: 」 머리를 떼고 홈 폴더는 ~."""
    lab = window.lab
    ov = to_wait(qtbot, window)
    lab.client.failed.disconnect(lab._on_failed)
    runs = Path.home() / "data" / "runs"
    _fatal(lab, f"[Errno 17] File exists: {str(runs)!r}")
    assert at(ov, "failed")
    expected = "[Errno 17] File exists: " + repr("~" + str(runs)[len(str(Path.home())):])
    assert ov.text_label.text() == S.ONB_FAILED.format(reason=expected)
    assert S.TRAIN_CANNOT_START.split("{")[0] not in ov.text_label.text()


def test_failure_card_without_message_uses_fixed_sentence(qtbot: Any, window: Any,
                                                          fake_client: list[FakeProc]) -> None:
    lab = window.lab
    ov = to_wait(qtbot, window)
    lab.status.setText("학습 전")  # 예전에는 상태 글 첫 줄이 까닭으로 보였다
    fake_client[-1].alive = False
    lab.client._finish()  # 까닭 글 없이 1세대 전에 닫힘
    assert at(ov, "failed")
    assert ov.text_label.text() == S.ONB_FAILED.format(reason=S.ONB_FAILED_NO_DETAIL)

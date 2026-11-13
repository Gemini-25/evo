"""6.2.0 화면 다듬기: 실험실 (polish/plan.md U5~U10, 완료 기준 C3·C4·C5·C6·C9·C13, 묶음 Hb).

- U5·C4: 오른쪽 패널의 최소 폭 ≤ 400px, 가장 넓은 칸이 둘째보다 60px 넘게 넓지 않음, 캔버스 폭 1280에서 ≥ 860 · 1366에서 ≥ 940.
- U6·C6: 「학습 시작」을 누른 바로 그때 상태 글 첫 줄이 「시작하는 중」, 상태 점이 starting, 캔버스 알림 글. 첫 세대 뒤 빔.
- U7·C5: 상태마다 보이는 으뜸 단추가 하나이고 켜진 것은 많아야 하나. state_changed는 상태가 바뀔 때만(세대마다 아님).
- U8: 첫 검증 전 챔피언은 「아직 없음」.
- U9·C9: 꺼진 단추의 풍선 = 보통 풍선 + 빈 줄 + 까닭, 켜지면 보통 풍선만. 연결표 단추의 toolTip()은 늘 HELP_TIPS 글.
  「보기」 칸은 관전·고속에 따라 켜짐. 학습 중에만 「학습 설정」 맨 위 잠금 줄.
- U10: extend_and_continue(n)은 최대 세대에 닿아 멈춘 학습만 n세대 더 잇는다.
- C3: 상태 글이 여러 줄이어도 「▶ 학습 시작」·「멈춤」·「재생 배속」이 다섯 창 크기에서 다 보인다. 여유 px 값은 리눅스 offscreen
  글꼴에서만 단언한다(윈도우 CI의 offscreen은 글꼴이 없어 줄바꿈이 사용자 화면과 다르다: 다듬기 C3).
- C13: 실험실 화면에 보이는 글에 프로그래머 말(「시험 0번」「전시 시드」「학습 프로세스 없음」「pycma」)이 없다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QPoint, QSettings
from PySide6.QtWidgets import QAbstractButton, QAbstractSpinBox, QApplication, QComboBox, QLabel, QScrollArea, QWidget

from evolab import paths
from evolab.app import strings as S
from evolab.app import style
from evolab.app.help import anchors
from evolab.app.lab.view import LabView

LANDER = paths.builtin_env_dir() / "lander.evoenv"
SIZES = [(1280, 720), (1280, 800), (1366, 728), (1366, 768), (1536, 824)]
LINUX = sys.platform.startswith("linux")
PROGRAMMER_WORDS = ("시험 0번", "전시 시드", "학습 프로세스 없음", "pycma")


@pytest.fixture
def lab(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr(paths, "user_env_dir", lambda: tmp_path / "environments")
    view = LabView(settings=QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.show()
    qtbot.waitExposed(view)
    try:
        yield view
    finally:
        view.client.abandon(timeout=60)
        view.shutdown()


@pytest.fixture
def window(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from evolab.app.main import MainWindow

    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv("EVOLAB_NO_TOUR", "1")
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    w.tabs.setCurrentWidget(w.lab)
    try:
        yield w
    finally:
        w.lab.client.abandon(timeout=60)


def small_run(lab: LabView, generations: int) -> None:
    """착륙선, 개체 4 · 시험 1 · 작업자 1, 고속 보기(관전 재생을 기다리지 않음)."""
    assert lab.load_env(LANDER)
    lab.rb_fast.setChecked(True)
    lab.popsize.setValue(4)
    lab.trials.setValue(1)
    lab.workers.setValue(1)
    lab.max_generations.setValue(generations)


def primaries(lab: LabView) -> list[QAbstractButton]:
    panel = lab.btn_start.parentWidget().parentWidget()
    return style.primaries(panel)


def assert_primary(lab: LabView, button: QAbstractButton) -> None:
    shown = primaries(lab)
    assert shown == [button], [b.text() for b in shown]
    assert sum(b.isEnabled() for b in shown) <= 1


def visible_texts(lab: LabView) -> list[str]:
    out = [lab.status.text(), lab.canvas.hud, lab.start_notice]
    for w in lab.findChildren(QWidget):
        if not w.isVisibleTo(lab):
            continue
        if isinstance(w, (QLabel, QAbstractButton)):
            out.append(w.text())
        elif isinstance(w, QComboBox):
            out += [w.itemText(i) for i in range(w.count())]
    return out


# ---------------------------------------------------------------------------
# U5·C4: 패널 폭과 캔버스 폭
def test_panel_is_narrow_and_no_single_row_sets_its_width(window: Any, qtbot: Any) -> None:
    lab = window.lab
    panel = lab.btn_start.parentWidget().parentWidget()
    assert panel.minimumSizeHint().width() <= 400
    widths = sorted((c.minimumSizeHint().width() for c in panel.findChildren(QWidget)
                     if isinstance(c, (QLabel, QAbstractButton, QComboBox, QAbstractSpinBox)) and c.isVisibleTo(panel)
                     and not (isinstance(c, QLabel) and c.wordWrap())), reverse=True)
    assert widths[0] - widths[1] <= 60, widths[:3]
    for size, least in (((1280, 720), 860), ((1366, 768), 940)):
        window.resize(*size)
        qtbot.waitUntil(lambda s=size: (window.width(), window.height()) == s, timeout=3000)
        qtbot.wait(50)
        assert lab.canvas.width() >= least, (size, lab.canvas.width())


# ---------------------------------------------------------------------------
# U6·U7·U8·U10: 진행 상태를 따라
def test_progress_states(qtbot: Any, lab: LabView) -> None:
    states: list[str] = []
    lab.state_changed.connect(states.append)
    small_run(lab, 2)
    assert lab.state == "idle" and lab.state_dot.kind == "idle"
    assert_primary(lab, lab.btn_start)
    assert lab.btn_start.isEnabled()

    assert lab.start_new()
    QApplication.processEvents()  # 이벤트 루프 한 번: 누른 바로 그때 보이는 것
    assert lab.status.text().split("\n")[0] == S.LAB_STATE_NAMES["starting"]
    assert lab.state_dot.kind == "starting"
    assert lab.start_notice == S.LAB_NOTICE_FIRST_GEN.format(pop=4)
    if hasattr(lab.canvas, "notice"):  # 묶음 PL의 set_notice(다듬기 3.3)
        assert lab.canvas.notice == lab.start_notice
    assert_primary(lab, lab.btn_pause)
    assert lab.locked_note.isVisibleTo(lab) and lab.stop_apply_note.isVisibleTo(lab)

    qtbot.waitUntil(lambda: lab.generation >= 1, timeout=120000)
    assert lab.start_notice == ""
    if hasattr(lab.canvas, "notice"):
        assert lab.canvas.notice == ""
    qtbot.waitUntil(lambda: lab.state == "finished", timeout=120000)
    assert_primary(lab, lab.btn_continue)
    assert lab.btn_continue.isEnabled() and lab.state_dot.kind == "finished"
    # 세대가 둘 와도 상태가 바뀔 때만 신호
    assert states[0] == "starting" and states[-1] == "finished"
    assert all(a != b for a, b in zip(states, states[1:], strict=False)), states
    assert states.count("running") <= 1

    # U10: 최대 세대에 닿아 멈춤 → 1세대 더
    assert lab.extend_and_continue(1)
    assert lab.max_generations.value() == 3
    qtbot.waitUntil(lambda: lab.generation == 3 and lab.state == "finished", timeout=120000)
    assert [int(float(r["generation"])) for r in lab.rows] == [1, 2, 3]
    assert S.LAB_STATUS_CHAMP_PENDING not in lab.status.text().split("\n")[0] or not lab.champion

    lab.end_session()
    qtbot.waitUntil(lambda: not lab.client.active, timeout=120000)
    assert lab.state == "closed"
    assert not lab.extend_and_continue(5)  # 학습 프로세스가 없으면 아무것도 하지 않음
    assert_primary(lab, lab.btn_continue)  # 저장된 실행: 「이어서 학습」이 으뜸
    assert not lab.locked_note.isVisibleTo(lab)
    assert not [t for t in visible_texts(lab) for word in PROGRAMMER_WORDS if word in t]


def test_extend_only_after_reaching_max_generations(qtbot: Any, lab: LabView) -> None:
    small_run(lab, 0)
    lab.max_evaluations.setValue(8)  # 「최대 시험 횟수」로 멈춤: 「10세대 더」는 뜻이 없다
    assert lab.start_new()
    qtbot.waitUntil(lambda: lab.state == "finished", timeout=120000)
    assert not lab.extend_and_continue(10)
    assert lab.max_generations.value() == 0


def test_champion_pending_text(lab: LabView) -> None:
    lab.rows = [{"champion_score": "", "evaluations": "32"}]
    lab.generation = 1
    lab.state = "running"
    lab._update_status()
    first = lab.status.text().split("\n")[0]
    assert first.endswith(S.LAB_STATUS_CHAMP_PENDING) and not first.endswith(" -")


# ---------------------------------------------------------------------------
# U9·C9: 꺼진 까닭
U9_BUTTONS = {"versus": "btn_versus", "video": "btn_video", "live": "btn_live", "pause": "btn_pause",
              "continue": "btn_continue", "step": "btn_step", "end": "btn_end", "speed": "speed_combo",
              "pace": "pace_combo", "fast_interval": "fast_interval"}
OWN_TIPS = ("btn_timeline", "btn_flip", "btn_race", "btn_apply", "btn_reload")


def check_tips(window: Any) -> None:
    lab = window.lab
    for key, attr in U9_BUTTONS.items():
        w = getattr(lab, attr)
        assert anchors.resolve(window, f"lab.{key}") is w
        assert w.toolTip() == S.HELP_TIPS[f"lab.{key}"], key  # 까닭이 도움말 글을 밀어내지 않는다(V1)
    for attr in (*U9_BUTTONS.values(), *OWN_TIPS):
        w = getattr(lab, attr)
        base = w.toolTip()
        assert base, attr
        if w.isEnabled():
            assert style.tip_text(w) == base, attr
        elif style.why_off(w):
            assert style.tip_text(w) == f"{base}\n\n{style.why_off(w)}", attr


def test_disabled_reasons_before_and_after_the_first_generation(qtbot: Any, window: Any) -> None:
    lab = window.lab
    check_tips(window)
    for attr, why in (("btn_timeline", S.LAB_WHY_NEED_GEN), ("btn_versus", S.LAB_WHY_NEED_GEN),
                      ("btn_flip", S.LAB_WHY_NEED_TWO), ("btn_race", S.LAB_WHY_NEED_TWO),
                      ("btn_video", S.LAB_WHY_NO_VIDEO), ("btn_live", S.LAB_WHY_LIVE),
                      ("btn_apply", S.LAB_WHY_PAUSED_ONLY), ("btn_reload", S.LAB_WHY_PAUSED_ONLY)):
        w = getattr(lab, attr)
        assert not w.isEnabled() and style.why_off(w) == why, attr
    small_run(lab, 2)
    assert lab.start_new()
    qtbot.waitUntil(lambda: lab.state == "finished", timeout=120000)
    check_tips(window)
    for attr in ("btn_timeline", "btn_versus", "btn_flip", "btn_race", "btn_apply"):
        assert getattr(lab, attr).isEnabled(), attr
    # 다시 보기를 하면 「실시간 보기」·「영상 저장」이 켜지고, 돌아오면 「실시간 보기」가 꺼진다
    assert lab.play_generation(1)
    assert lab.btn_live.isEnabled()
    qtbot.waitUntil(lambda: lab.btn_video.isEnabled(), timeout=60000)
    check_tips(window)
    lab.back_to_live()
    assert not lab.btn_live.isEnabled() and style.why_off(lab.btn_live) == S.LAB_WHY_LIVE
    assert lab.btn_video.isEnabled()  # 마지막 다시 보기는 그대로 저장할 수 있다


def test_view_rows_follow_spectate_and_fast(lab: LabView) -> None:
    lab.rb_spectate.setChecked(True)
    assert lab.speed_combo.isEnabled() and lab.pace_combo.isEnabled() and not lab.fast_interval.isEnabled()
    assert style.why_off(lab.fast_interval) == S.LAB_WHY_FAST_ONLY
    lab.rb_fast.setChecked(True)
    assert not lab.speed_combo.isEnabled() and not lab.pace_combo.isEnabled() and lab.fast_interval.isEnabled()
    assert style.why_off(lab.speed_combo) == S.LAB_WHY_SPECTATE_ONLY
    assert lab.speed_combo.isVisibleTo(lab)  # 줄은 그대로 보인다


def test_view_rows_restored_from_settings(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "user_env_dir", lambda: tmp_path / "environments")
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    settings.setValue("lab/mode", "fast")
    settings.setValue("lab/pace", "skip")
    view = LabView(settings=settings)
    qtbot.addWidget(view)
    try:
        assert view.rb_fast.isChecked() and not view.speed_combo.isEnabled() and view.fast_interval.isEnabled()
    finally:
        view.shutdown()


# ---------------------------------------------------------------------------
# C3: 여러 줄 상태 글에서도 세 위젯이 보인다(상태 글은 다듬기 원형 polish/revise/scripts/m_fold2.py와 같은 글)
WINPATH = "C:\\Users\\student\\Documents\\진화 실험실\\"


def _first(state: str, gen: int = 12, evals: int = 1824, champ: str = "177.90") -> str:
    return S.LAB_STATUS.format(state=S.LAB_STATE_NAMES[state], gen=gen, evals=evals, champ=champ)


_ATTEMPT = S.LAB_STATUS_ATTEMPT.format(n=1, stall=3, p=24)
_STOP = S.LAB_STOPPED_BECAUSE.format(reason="최대 세대 30")
STATUS_CASES: dict[str, tuple[int, str]] = {
    "idle": (70, S.LAB_STATUS_IDLE),
    "starting+path": (70, S.LAB_STARTED_RUN.format(path=WINPATH + "runs\\20261113-120655-lander-cmaes-s1")),
    "running": (70, _first("running") + "\n" + _ATTEMPT),
    "paused": (70, _first("paused") + "\n" + _ATTEMPT),
    "running big": (40, _first("running", 1234, 9876543, "-1234.56") + "\n"
                    + S.LAB_STATUS_ATTEMPT.format(n=3, stall=1200, p=2400)),
    "running+wait": (40, _first("running") + "\n" + _ATTEMPT + "\n" + S.LAB_PACE_WAIT_HINT.format(times=2.2)),
    "finished+stop": (40, _first("finished", 30, 9600) + "\n" + _ATTEMPT + "\n" + _STOP),
    "closed+other PC": (40, _first("closed") + "\n" + _ATTEMPT + "\n" + S.LAB_OTHER_PLATFORM.format(what="cpu")),
    "finished+video": (0, _first("finished", 30, 9600) + "\n" + _ATTEMPT + "\n" + _STOP + "\n"
                       + S.VIDEO_DONE.format(path=WINPATH + "exports\\달 착륙선_12세대.mp4")),
}
"""상태 → (1280×720에서 「재생 배속」 아래 남아야 하는 px(글꼴 ×1.0, 다듬기 C3), 상태 글)."""
SCALED_NEED = {70: 60, 40: 28, 0: None}
"""글꼴 ×1.10의 기준(C3). 경로가 든 6줄 상태는 기록만(None)."""


def _slack(lab: LabView) -> int:
    w = lab.speed_combo.parentWidget()
    while w is not None and not isinstance(w, QScrollArea):
        w = w.parentWidget()
    assert w is not None
    vp = w.viewport()
    return vp.height() - lab.speed_combo.mapTo(vp, QPoint(0, lab.speed_combo.height())).y()


@pytest.mark.parametrize("scale", [1.0, 1.10])
def test_progress_widgets_stay_visible_with_long_status(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         scale: float) -> None:
    from PySide6.QtGui import QFont

    from evolab.app.main import MainWindow

    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv("EVOLAB_NO_TOUR", "1")
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    old = app.font()
    slack: dict[tuple[str, tuple[int, int]], int] = {}
    try:
        f = QFont(old)
        f.setPointSizeF(old.pointSizeF() * scale)
        app.setFont(f)  # 창을 만들기 전에(원형 m_fold2.py와 같은 방식)
        window = MainWindow()
        qtbot.addWidget(window)
        window.show()
        qtbot.waitExposed(window)
        window.tabs.setCurrentWidget(window.lab)
        lab = window.lab
        try:
            for size in SIZES:
                lo = window.minimumSize()
                assert lo.width() <= size[0] and lo.height() <= size[1], (size, scale)
                window.resize(*size)
                qtbot.waitUntil(lambda s=size: (window.width(), window.height()) == s, timeout=3000)
                qtbot.wait(100)
                for name, (need, text) in STATUS_CASES.items():
                    lab.status.setText(text)
                    qtbot.wait(40)
                    slack[(name, size)] = _slack(lab)
                    if need == 0 and scale != 1.0:
                        continue  # 기록만(C3 ③)
                    for attr in ("btn_start", "btn_pause", "speed_combo"):
                        w = getattr(lab, attr)
                        region = w.visibleRegion()
                        assert not region.isEmpty() and region.boundingRect() == w.rect(), (name, size, attr, scale)
        finally:
            lab.shutdown()
    finally:
        app.setFont(old)
    print(f"x{scale}", {f"{k[0]}@{k[1][0]}x{k[1][1]}": v for k, v in slack.items() if k[1] == (1280, 720)})
    if LINUX:  # px 값은 리눅스 offscreen 글꼴에서만(C3)
        for name, (need, _text) in STATUS_CASES.items():
            want = need if scale == 1.0 else SCALED_NEED[need]
            if want is not None:
                assert slack[(name, (1280, 720))] >= want, (name, slack[(name, (1280, 720))], want, scale)

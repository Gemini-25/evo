"""v1.1 학습 안정성 화면: 실험실 설정 묶음·상태 글, 반복 실행 체크 상자 (설계 11.1·11.3, 테스트 31·32).

학습 프로세스를 띄우는 것은 한 테스트뿐이다(워커 1, 개체 6 × 시험 2). 나머지는 폼을 직접 읽거나, 착륙선을 돌리지 않는
가짜 평가기로 만든 통계 줄을 화면에 넣어 본다.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox

from evolab import paths
from evolab.app import strings as S
from evolab.app.compare.repeat import RepeatDialog, qualify_too_big
from evolab.app.lab.view import SHOWN_RESTART_PATIENCE, LabView, switch_note
from evolab.engine import seeds
from evolab.engine import trainer as trainer_mod
from evolab.engine.evaluate import EvalResult
from evolab.engine.runsetup import create_run
from evolab.engine.trainer import (
    LIVE_CONFIG_KEYS,
    NEW_RUN_DEFAULTS,
    RESTART_FROM,
    SWITCH_QUALIFY_NEXT,
    SWITCH_QUALIFY_PICK,
    SWITCH_RESTART_FRESH,
    TrainConfig,
    Trainer,
    qualify_budget_problem,
)
from evolab.runs.catalog import clone_settings
from evolab.runs.store import RunStore
from evolab.spec.io import compile_env

LANDER = paths.builtin_env_dir() / "lander.evoenv"
LANDER_TEXT = LANDER.read_text(encoding="utf-8")
V10_RUN = Path(__file__).parent / "data" / "v10_run_ga" / "run"
VALIDATION_SEEDS = {seeds.derive(seeds.TAG_VALIDATION, k) for k in range(64)}
COMPARE_SEEDS = {seeds.derive(seeds.TAG_COMPARE, k) for k in range(64)}


@pytest.fixture()
def lab(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LabView:
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path / "runs")
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    view = LabView(settings=settings)
    qtbot.addWidget(view)
    view.resize(1280, 800)
    view.show()
    assert view.load_env(LANDER)
    view.popsize.setValue(6)
    view.trials.setValue(2)
    view.workers.setValue(1)
    try:
        yield view
    finally:
        view.client.abandon(timeout=60)  # 실패한 테스트도 학습 프로세스를 남기지 않게
        view.shutdown()


@pytest.fixture()
def message_boxes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """QMessageBox.information/warning을 띄우지 않고 (제목, 내용)으로 모은다."""
    boxes: list[tuple[str, str]] = []

    def record(_parent: Any, title: str, text: str, *_a: Any, **_k: Any) -> Any:
        boxes.append((title, text))
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "information", staticmethod(record))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(record))
    return boxes


def wait_state(qtbot: Any, view: LabView, state: str, timeout: int = 240000) -> None:
    qtbot.waitUntil(lambda: view.state == state, timeout=timeout)


def stability(c: TrainConfig) -> dict[str, Any]:
    return {k: getattr(c, k) for k in NEW_RUN_DEFAULTS}


# ---------------------------------------------------------------------------
# 착륙선을 돌리지 않는 가짜 평가기로 만든 실행(학습 방법 교체가 정해진 세대에 생긴다)
# ---------------------------------------------------------------------------


class FakeEvaluator:
    """학습 시험은 0점, 검증 시험([1, k])은 validation(학습 방법 번호) 점. 학습 방법마다 점수가 한 번만 올라 정체한다."""

    def __init__(self, validation: Any) -> None:
        self.validation = validation
        self.trainer: Trainer | None = None
        self._blocks = len(compile_env(LANDER_TEXT).file.scoring.blocks)

    def evaluate(self, env_version: str, env_text: str, layout: Any, genomes: np.ndarray,
                 pairs: list[tuple[int, Any]], record: list[bool] | None = None) -> EvalResult:
        n, nb = len(pairs), self._blocks
        scores = np.zeros(n)
        first = pairs[0][1].seed
        if first in VALIDATION_SEEDS:
            assert self.trainer is not None
            scores[:] = self.validation(self.trainer.state.current.index)
        elif first in COMPARE_SEEDS:
            scores[:] = 0.0
        return EvalResult(scores=scores, steps=np.ones(n, dtype=np.int64), end_events=["timeout"] * n,
                          outcomes=["failure"] * n, blocks=np.zeros((n, nb)), action_abs=np.zeros((n, 3)),
                          weighted=np.zeros((n, nb)))


def fake_run(base: Path, config: TrainConfig, validation: Any, generations: int) -> tuple[Path, list[dict[str, Any]]]:
    """train 명령과 같은 실행 폴더를 만들고 가짜 평가기로 학습한다. (실행 폴더, 학습 프로세스가 보내는 모양의 통계 줄)."""
    spec = compile_env(LANDER_TEXT)
    store = create_run(base, LANDER_TEXT, spec, config, 0)
    ev = FakeEvaluator(validation)
    t = Trainer(LANDER_TEXT, config, ev, store)
    ev.trainer = t
    rows = [t.step() for _ in range(generations)]
    return store.root, rows


def tiny(**over: Any) -> TrainConfig:
    """세대마다 학습 시험 12회(개체 6 × 시험 2), 검증 주기 12."""
    return TrainConfig.for_new_run(**{"popsize": 6, "trials": 2, "validate_every": 12, "validation_seeds": 2,
                                      "validation_candidates": 1, "checkpoint_every": 0, "run_seed": 5, **over})


def feed(view: LabView, row: dict[str, Any]) -> None:
    """학습 프로세스의 generation 알림(관전 궤적 없음)을 화면에 넣는다."""
    view._on_generation({"row": row, "generation": int(row["generation"]), "champion": None})


def status_lines(view: LabView) -> list[str]:
    return view.status.text().split("\n")


# ---------------------------------------------------------------------------
# 31. 실험실 설정: 새 실행 기본값, 폼 ↔ 설정 왕복, v1.0 실행은 끔
# ---------------------------------------------------------------------------


def test_plain_new_run_uses_new_run_defaults(qtbot: Any, lab: LabView) -> None:
    assert lab.restart_on.isChecked() == (NEW_RUN_DEFAULTS["restart_patience"] is not None)
    c = lab.config_from_form()
    assert stability(c) == NEW_RUN_DEFAULTS
    c.validate()
    assert [lab.restart_from.itemData(i) for i in range(lab.restart_from.count())] == list(RESTART_FROM)
    assert [lab.restart_from.itemText(i) for i in range(lab.restart_from.count())] == \
        [S.RESTART_FROM_NAMES[k] for k in RESTART_FROM]
    assert lab.qualify_entrants.text() == S.LAB_QUALIFY_OFF  # 1 = 예선 없음
    assert lab.restart_patience.isEnabled() and not lab.qualify_trials.isEnabled()
    # 어림 글: 세대마다 12회(개체 6 × 시험 2)
    assert lab.stability_hint.text() == S.LAB_STABILITY_HINT_NO_QUALIFY.format(rgens=math.ceil(20000 / 12))
    lab.qualify_entrants.setValue(3)
    assert lab.qualify_trials.isEnabled()
    lab.stagnation.setValue(100)  # 100세대 × 12회 = 1200회 < 20000회: 정체 멈춤이 먼저 온다
    assert lab.stability_hint.text() == (
        S.LAB_STABILITY_HINT.format(total=36000, qgens=3000, rgens=1667) + S.LAB_STABILITY_STOP_FIRST.format(stop=100))
    lab.stagnation.setValue(2000)
    lab.restart_on.setChecked(False)
    assert not lab.restart_patience.isEnabled()
    assert lab.stability_hint.text() == S.LAB_STABILITY_HINT_QUALIFY_ONLY.format(total=36000, qgens=3000)
    assert lab.config_from_form().restart_patience is None
    lab.qualify_entrants.setValue(1)
    assert lab.stability_hint.text() == ""
    # 담금질은 세대마다 (이웃 8 + 현재해 1) × 시험 2 = 18회
    lab.method.setCurrentIndex(lab.method.findData("sa"))
    lab.restart_on.setChecked(True)
    assert lab.stability_hint.text() == S.LAB_STABILITY_HINT_NO_QUALIFY.format(rgens=math.ceil(20000 / 18))


def test_stability_form_round_trip(qtbot: Any, lab: LabView) -> None:
    original = TrainConfig.for_new_run(popsize=6, trials=2, run_seed=4, validate_every=500, restart_patience=30000,
                                       restart_from="champion", qualify_entrants=3, qualify_trials=15000)
    assert lab.use_env_text(LANDER_TEXT, "착륙선", original.to_dict())
    assert lab.restart_on.isChecked() and lab.restart_patience.value() == 30000
    assert lab.restart_from.currentData() == "champion"
    assert (lab.qualify_entrants.value(), lab.qualify_trials.value()) == (3, 15000)
    assert lab.config_from_form().to_dict() == original.to_dict()
    lab.restart_from.setCurrentIndex(lab.restart_from.findData("alternate"))
    assert lab.config_from_form().restart_from == "alternate"
    lab.restart_on.setChecked(False)
    assert lab.config_from_form().restart_patience is None
    lab.restart_on.setChecked(True)
    assert lab.config_from_form().restart_patience == 30000
    lab.restart_patience.setValue(25000)
    lab.qualify_entrants.setValue(2)
    lab.qualify_trials.setValue(9000)
    assert stability(lab.config_from_form()) == {"restart_patience": 25000, "restart_from": "alternate",
                                                 "qualify_entrants": 2, "qualify_trials": 9000}

    # v1.1에서 끈 실행(None): 체크를 켜면 None이 아니라 칸 값으로 켠다
    off = TrainConfig.for_new_run(popsize=6, trials=2, restart_patience=None)
    assert lab.use_env_text(LANDER_TEXT, "착륙선", off.to_dict())
    assert not lab.restart_on.isChecked() and not lab.restart_patience.isEnabled()
    assert lab.config_from_form().to_dict() == off.to_dict()
    lab.restart_on.setChecked(True)
    assert lab.config_from_form().restart_patience == SHOWN_RESTART_PATIENCE

    # 칸이 보여 줄 수 없는 값(칸의 최솟값 아래)은 붙잡지 않고 칸에 보이는 값으로 학습한다
    small = TrainConfig.for_new_run(popsize=6, trials=2, validate_every=24, restart_patience=48, qualify_entrants=2,
                                    qualify_trials=24)
    assert lab.use_env_text(LANDER_TEXT, "착륙선", small.to_dict())
    c = lab.config_from_form()
    assert (c.restart_patience, c.qualify_trials) == (lab.restart_patience.value(), lab.qualify_trials.value()) \
        == (1000, 1000)


def test_v10_run_opened_or_cloned_shows_restart_off(qtbot: Any, lab: LabView, tmp_path: Path) -> None:
    run = tmp_path / "runs" / "v10"
    shutil.copytree(V10_RUN, run)  # 고정 자료는 고치지 않는다
    old = clone_settings(run)
    assert not set(NEW_RUN_DEFAULTS) & set(old["config"])

    assert lab.open_run(run)
    assert not lab.restart_on.isChecked() and lab.qualify_entrants.value() == 1
    assert lab.config_from_form().restart_patience is None
    # 예전 열 머리라 둘째·셋째 줄이 없다
    assert lab._learner_status(lab.rows[-1]) is None and lab._switch_note == ""
    assert S.LAB_STATUS_ATTEMPT_OFF.format(n=1) not in lab.status.text()

    lab.restart_on.setChecked(True)  # 다른 상태에서 복제해도 칸은 v1.0 실행을 따른다
    assert lab.use_env_text(old["env_text"], old.get("env_name", ""), old["config"])
    assert not lab.restart_on.isChecked()
    assert lab.config_from_form().to_dict() == TrainConfig.from_dict(old["config"]).to_dict()  # v1.0과 같은 설정
    assert lab.stability_hint.text() == ""


# ---------------------------------------------------------------------------
# 31. 학습 중: 예선 칸은 잠기고, 재시작 칸은 멈췄을 때 '적용'으로 보낸다. 예산이 모자라면 시작하지 않는다
# ---------------------------------------------------------------------------


def test_qualify_locked_while_training_restart_applied_when_paused(qtbot: Any, lab: LabView,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(lab.client, "send", lambda msg: sent.append(msg) or True)
    lab.qualify_entrants.setValue(2)
    assert lab.qualify_entrants.isEnabled() and lab.qualify_trials.isEnabled()
    lab.client._proc = object()  # 학습 프로세스가 있는 것처럼(가짜)
    try:
        lab.state = "running"
        lab._update_buttons()
        assert not lab.qualify_entrants.isEnabled() and not lab.qualify_trials.isEnabled()
        assert lab.restart_on.isEnabled() and lab.restart_patience.isEnabled() and lab.restart_from.isEnabled()
        assert not lab.btn_apply.isEnabled()
        lab.qualify_entrants.setValue(3)  # 바꿔도(코드로) 잠금은 그대로
        assert not lab.qualify_trials.isEnabled()
        lab.state = "paused"
        lab._update_buttons()
        assert lab.btn_apply.isEnabled() and not lab.qualify_entrants.isEnabled()
        lab.restart_patience.setValue(25000)
        lab.restart_from.setCurrentIndex(lab.restart_from.findData("champion"))
        lab.apply_settings()
        lab.restart_on.setChecked(False)
        assert lab.restart_patience.isEnabled() is False and lab.restart_from.isEnabled()
        lab.apply_settings()
    finally:
        lab.client._proc = None
    lab._update_buttons()
    assert lab.qualify_entrants.isEnabled() and lab.qualify_trials.isEnabled()
    first, second = (m["changes"] for m in sent)
    assert set(first) <= set(LIVE_CONFIG_KEYS) and not {"qualify_entrants", "qualify_trials"} & set(first)
    assert (first["restart_patience"], first["restart_from"]) == (25000, "champion")
    assert (second["restart_patience"], second["restart_from"]) == (None, "champion")


def test_start_refuses_qualifying_over_budget(qtbot: Any, lab: LabView, monkeypatch: pytest.MonkeyPatch,
                                              message_boxes: list[tuple[str, str]]) -> None:
    started: list[Any] = []
    monkeypatch.setattr(lab.client, "start", started.append)
    lab.qualify_entrants.setValue(2)
    lab.qualify_trials.setValue(12000)
    lab.max_evaluations.setValue(24000)  # 예선 2 × 12000 = 24000회가 예산 이상
    problem = qualify_budget_problem(lab.config_from_form())
    assert problem is not None
    assert not lab.start_new()
    assert message_boxes == [(S.LAB_CHANGE_FAILED, problem)] and started == [] and lab.state == "idle"
    lab.max_evaluations.setValue(24001)
    assert lab.start_new()
    (req,) = started
    assert stability(TrainConfig.from_dict(req.config)) == {**NEW_RUN_DEFAULTS, "qualify_entrants": 2,
                                                            "qualify_trials": 12000}
    assert len(message_boxes) == 1


# ---------------------------------------------------------------------------
# 31. 상태 글 둘째 줄(예선 출전·시도), 셋째 줄(마지막 학습 방법 교체 알림)
# ---------------------------------------------------------------------------


def expected_learner_line(row: dict[str, Any], config: TrainConfig) -> str:
    if row["qualifying"]:
        return S.LAB_STATUS_QUALIFY.format(n=row["learner"] + 1, k=config.qualify_entrants, t=row["learner_trials"],
                                           b=config.qualify_trials)
    if config.restart_patience is None:
        return S.LAB_STATUS_ATTEMPT_OFF.format(n=row["restarts"] + 1)
    return S.LAB_STATUS_ATTEMPT.format(n=row["restarts"] + 1, stall=row["learner_stall"], p=config.restart_patience)


def test_status_lines_follow_qualifying_and_restarts(qtbot: Any, lab: LabView, tmp_path: Path) -> None:
    # 출전 2개 × 24회, 자동 재시작 24회. 검증 점수 = 학습 방법 번호라 출전 2(번호 1)가 이기고, 학습 방법마다 한 번 오른 뒤 정체한다
    config = tiny(qualify_entrants=2, qualify_trials=24, restart_patience=24)
    run, rows = fake_run(tmp_path / "runs", config, float, 10)
    codes = [r["switch"] for r in rows]
    assert codes[:4] == [0, SWITCH_QUALIFY_NEXT, 0, SWITCH_QUALIFY_PICK]  # 출전마다 2세대
    assert rows[4]["learner"] == 1 and rows[4]["qualifying"] == 0
    assert codes.count(SWITCH_RESTART_FRESH) >= 2
    lab.run_info = {"config": config.to_dict()}
    note = ""
    for i, row in enumerate(rows):
        feed(lab, row)
        code = row["switch"]
        if code == SWITCH_QUALIFY_NEXT:
            note = S.TRAIN_SWITCH_QUALIFY_NEXT.format(gen=row["generation"], n=1, score="0.00", next=2)
        elif code == SWITCH_QUALIFY_PICK:
            # 다음 줄이 오기 전에도 출전이 끝난 줄로 통과자를 안다(출전 2, 검증 1.00)
            note = S.TRAIN_SWITCH_QUALIFY_PICK.format(gen=row["generation"], winner=2, score="1.00")
        elif code == SWITCH_RESTART_FRESH:
            note = S.TRAIN_SWITCH_RESTART.format(gen=row["generation"], stall=row["learner_stall"],
                                                 where=S.RESTART_FROM_NAMES_SHORT["fresh"], n=row["restarts"] + 2)
        lines = status_lines(lab)
        assert lines[1] == expected_learner_line(row, config), i
        assert (lines[2] if len(lines) > 2 else "") == note, i
    assert lab._switch_note == note

    # 기록에서 열면(통계 값은 글자) 같은 둘째·셋째 줄
    live = status_lines(lab)[1:3]
    assert lab.open_run(run)
    assert status_lines(lab)[1:3] == live
    # 예선 끝 알림은 다음 줄이 있으면 그 줄의 learner를 통과자로 쓴다
    pick = codes.index(SWITCH_QUALIFY_PICK)
    assert switch_note(lab.rows, pick) == switch_note(lab.rows[: pick + 1], pick) == \
        S.TRAIN_SWITCH_QUALIFY_PICK.format(gen=pick + 1, winner=2, score="1.00")
    # 자동 재시작을 끄면(멈춘 상태에서 적용해 run_info의 설정이 바뀌면) 다음 상태 글은 '꺼짐'
    lab.run_info["config"]["restart_patience"] = None
    lab._update_status()
    assert status_lines(lab)[1] == S.LAB_STATUS_ATTEMPT_OFF.format(n=rows[-1]["restarts"] + 1)


def test_qualifying_tie_goes_to_the_earlier_entrant(qtbot: Any, lab: LabView, tmp_path: Path) -> None:
    config = tiny(qualify_entrants=3, qualify_trials=24, restart_patience=None)
    run, rows = fake_run(tmp_path / "runs", config, lambda _index: 5.0, 7)
    pick = [r["switch"] for r in rows].index(SWITCH_QUALIFY_PICK)
    assert rows[pick + 1]["learner"] == 0  # 엔진: 동점이면 앞 번호
    lab.run_info = {"config": config.to_dict()}
    for row in rows[: pick + 1]:
        feed(lab, row)
    want = S.TRAIN_SWITCH_QUALIFY_PICK.format(gen=pick + 1, winner=1, score="5.00")
    assert status_lines(lab)[2] == want
    assert status_lines(lab)[1] == S.LAB_STATUS_QUALIFY.format(n=3, k=3, t=24, b=24)
    feed(lab, rows[pick + 1])
    assert status_lines(lab)[1:] == [S.LAB_STATUS_ATTEMPT_OFF.format(n=1), want]
    lab._reset_run_view()
    assert lab._switch_note == "" and lab._switch_at is None
    assert lab.open_run(run)
    assert lab._switch_note == want


@pytest.mark.parametrize("turn_on", [True, False])
def test_opened_run_shows_the_restart_setting_changed_after_the_start(qtbot: Any, lab: LabView, tmp_path: Path,
                                                                      turn_on: bool) -> None:
    """멈춘 상태에서(실험실 '적용', train --resume --restart) 바꾼 자동 재시작 설정은 run.json(시작 때 설정)이 아니라
    체크포인트와 사건 줄에만 남는다. 기록에서 열어도 상태 글·칸은 지금 설정(마지막 체크포인트)을 따른다."""
    validation = lambda _index: 5.0  # noqa: E731
    start = tiny(restart_patience=None if turn_on else 24, checkpoint_every=1)
    run, _ = fake_run(tmp_path / "runs", start, validation, 4)
    store = RunStore.open(run)
    ev = FakeEvaluator(validation)
    t = Trainer.resume(store, ev, max_generations=99)  # 이어 할 때 바꾼 정지 조건(명령줄 --max-generations)
    ev.trainer = t
    # 세션의 '적용'과 같은 순서: 트레이너 → 체크포인트 → 보호 → 사건 줄
    changes = t.apply_config({"restart_patience": 24 if turn_on else None, "restart_from": "champion"})
    t.checkpoint()
    store.protect(4)
    store.append_event("config_change", 4, changes=changes)
    rows = [t.step() for _ in range(10)]
    t.finish()
    assert store.read_run_info()["config"]["restart_patience"] == (None if turn_on else 24)  # 시작 때 설정 그대로
    assert any(r["switch"] for r in rows) == turn_on  # 켠 실행은 실제로 새로 시작했다

    assert lab.open_run(run)
    assert lab.restart_on.isChecked() == turn_on and lab.restart_from.currentData() == "champion"
    last = lab.rows[-1]
    n = int(last["restarts"]) + 1
    want = (S.LAB_STATUS_ATTEMPT.format(n=n, stall=int(last["learner_stall"]), p=24) if turn_on
            else S.LAB_STATUS_ATTEMPT_OFF.format(n=n))
    assert status_lines(lab)[1] == want
    # 다른 설정은 v1.0처럼 run.json 값(시작 때 설정)을 보인다
    assert lab.config_value("max_generations") is None and lab.max_generations.value() == 0


def test_applying_a_restart_change_updates_the_status_at_once(qtbot: Any, lab: LabView, tmp_path: Path) -> None:
    """멈춘 상태에서 '적용'하면 다음 세대를 기다리지 않고 상태 글 둘째 줄을 바뀐 설정으로 고친다(완료 줄은 쌓이지 않음)."""
    config = tiny(restart_patience=24)
    _, rows = fake_run(tmp_path / "runs", config, lambda _index: 5.0, 3)
    lab.run_info = {"config": config.to_dict()}
    lab.state = "paused"
    for row in rows:
        feed(lab, row)
    stall = rows[-1]["learner_stall"]
    assert status_lines(lab)[1] == S.LAB_STATUS_ATTEMPT.format(n=1, stall=stall, p=24)
    done = S.LAB_CHANGE_DONE.format(gen=3)

    def apply(patience: int | None) -> list[str]:
        new = TrainConfig.from_dict({**config.to_dict(), "restart_patience": patience})
        lab._on_applied({"type": "applied", "what": "config", "ok": True, "generation": 3, "config": new.to_dict(),
                         "changes": {"restart_patience": patience}})
        return status_lines(lab)

    lines = apply(36)
    assert lines[1] == S.LAB_STATUS_ATTEMPT.format(n=1, stall=stall, p=36) and lines[-1] == done
    lines = apply(None)
    assert lines[1] == S.LAB_STATUS_ATTEMPT_OFF.format(n=1) and lines.count(done) == 1
    # 끝난 상태에서 적용해도 끝난 까닭 줄은 남는다
    lab._on_state({"state": "finished", "reason": "max_generations"})
    stopped = S.LAB_STOPPED_BECAUSE.format(reason=S.TRAIN_STOP_REASONS["max_generations"])
    lines = apply(48)
    assert lines[1] == S.LAB_STATUS_ATTEMPT.format(n=1, stall=stall, p=48)
    assert lines[-2:] == [stopped, done] and lines.count(stopped) == 1


# ---------------------------------------------------------------------------
# 31. 실제 학습 프로세스: 둘째 줄, 멈춘 상태에서 재시작 설정 적용(사건 기록), 끝나면 예선 칸이 다시 켜짐
# ---------------------------------------------------------------------------


def test_session_status_and_paused_restart_change(qtbot: Any, lab: LabView) -> None:
    lab.rb_fast.setChecked(True)
    lab.max_generations.setValue(50)
    assert lab.start_new()
    qtbot.waitUntil(lambda: len(lab.rows) >= 1, timeout=120000)
    assert not lab.qualify_entrants.isEnabled() and lab.restart_on.isEnabled()
    lab.btn_pause.click()
    wait_state(qtbot, lab, "paused")
    last = lab.rows[-1]
    assert status_lines(lab)[1] == S.LAB_STATUS_ATTEMPT.format(n=1, stall=int(last["learner_stall"]), p=20000)

    lab.restart_patience.setValue(30000)
    lab.restart_from.setCurrentIndex(lab.restart_from.findData("champion"))
    with qtbot.waitSignal(lab.client.applied, timeout=60000) as applied:
        lab.apply_settings()
    assert applied.args[0]["ok"] and applied.args[0]["changes"] == {"restart_patience": 30000,
                                                                    "restart_from": "champion"}
    assert lab.config_value("restart_patience") == 30000
    assert [e["kind"] for e in lab.events].count("config_change") == 1
    # 멈춘 동안에는 세대 알림이 오지 않으므로 상태 글을 적용하자마자 바뀐 값으로 고친다
    assert status_lines(lab)[1] == S.LAB_STATUS_ATTEMPT.format(n=1, stall=int(lab.rows[-1]["learner_stall"]), p=30000)
    assert status_lines(lab)[-1] == S.LAB_CHANGE_DONE.format(gen=int(applied.args[0]["generation"]))
    n = len(lab.rows)
    lab.btn_step.click()
    qtbot.waitUntil(lambda: len(lab.rows) == n + 1, timeout=120000)
    assert status_lines(lab)[1] == S.LAB_STATUS_ATTEMPT.format(n=1, stall=int(lab.rows[-1]["learner_stall"]),
                                                               p=30000)
    lab.restart_on.setChecked(False)
    with qtbot.waitSignal(lab.client.applied, timeout=60000) as applied:
        lab.apply_settings()
    assert applied.args[0]["ok"] and applied.args[0]["changes"] == {"restart_patience": None}
    assert status_lines(lab)[1] == S.LAB_STATUS_ATTEMPT_OFF.format(n=1)
    lab.btn_step.click()
    qtbot.waitUntil(lambda: len(lab.rows) == n + 2, timeout=120000)
    assert status_lines(lab)[1] == S.LAB_STATUS_ATTEMPT_OFF.format(n=1)
    lab.end_session()
    qtbot.waitUntil(lambda: not lab.client.active and lab.qualify_entrants.isEnabled(), timeout=60000)


# ---------------------------------------------------------------------------
# 32. 반복 실행: 새 실행 권장값, '정체하면 새로 시작'을 끄면 None, 예산이 작으면 예선을 끈다
# ---------------------------------------------------------------------------


def test_repeat_dialog_restart_checkbox(qtbot: Any, tmp_path: Path) -> None:
    dlg = RepeatDialog(runs_dir=tmp_path, workers=1)
    qtbot.addWidget(dlg)
    assert dlg.restart.isChecked() and dlg.restart.text() == S.CMP_REPEAT_RESTART
    c = TrainConfig.from_dict(dlg.plan().config)
    assert stability(c) == NEW_RUN_DEFAULTS
    dlg.restart.setChecked(False)
    plan = dlg.plan()
    assert plan.config["restart_patience"] is None
    assert all(plan.config_for(i)["restart_patience"] is None for i in range(plan.count))
    assert not dlg.qualify_note.isVisibleTo(dlg)


def test_repeat_dialog_turns_off_qualifying_for_small_budgets(qtbot: Any, tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """예선 기본값이 켜지는 경우(실험 결과): 예선이 예산의 절반을 넘으면 끄고 알린다."""
    monkeypatch.setitem(trainer_mod.NEW_RUN_DEFAULTS, "qualify_entrants", 3)
    dlg = RepeatDialog(runs_dir=tmp_path, workers=1)
    qtbot.addWidget(dlg)
    total = 3 * NEW_RUN_DEFAULTS["qualify_trials"]
    assert dlg.max_evaluations.value() == 30000  # 기본 예산: 예선 36000회가 절반(15000회)을 넘는다
    assert dlg.qualify_note.isVisibleTo(dlg) and dlg.qualify_note.text() == \
        S.CMP_REPEAT_QUALIFY_OFF_HINT.format(total=total)
    assert dlg.config().qualify_entrants == 1
    dlg.max_evaluations.setValue(2 * total)  # 딱 절반이면 켠 채
    assert not dlg.qualify_note.isVisibleTo(dlg) and dlg.config().qualify_entrants == 3
    dlg.max_evaluations.setValue(2 * total - 1)
    assert dlg.qualify_note.isVisibleTo(dlg) and dlg.config().qualify_entrants == 1
    dlg.stop_time.setChecked(True)  # 시간 예산이면 예선을 그대로 둔다
    assert not dlg.qualify_note.isVisibleTo(dlg) and dlg.config().qualify_entrants == 3
    over = TrainConfig.for_new_run(qualify_entrants=2, qualify_trials=12000, max_evaluations=24000)
    assert qualify_too_big(over) == 24000 and qualify_budget_problem(over) is not None
    assert qualify_too_big(TrainConfig.for_new_run(qualify_entrants=1, max_evaluations=10)) is None

"""v2.0 내장 예제(설계 4.4, 4.7, 5.6, 9.3; T19·T31~T36): 점프 게임은 풀 수 있고, 글라이더 처음 몸은 원형 값,
내장 예제 순서와 첫 실행 복사, 자체 점검. T32·T33(9.3 완료 기준 절차)은 느린 테스트다(`pytest -m slow`)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from jump_planner import first_second_clear, measure_profile, plan, run_open_loop

from evolab import cli, paths, selftest
from evolab.app import strings as S
from evolab.engine import seeds
from evolab.engine.evaluate import evaluate
from evolab.engine.pool import default_workers
from evolab.policy.genome import layout_for
from evolab.spec.io import feature_version, load_env

ENVS = paths.builtin_env_dir()
JUMP = ENVS / "jump.evoenv"
GLIDER = ENVS / "glider.evoenv"


def test_examples_compile_without_warnings() -> None:
    """T31: 두 예제가 검사 경고 없이 컴파일된다(글라이더는 SPEC_BRAINLESS 안내 하나만). 형식 2 파일이다."""
    jump = load_env(JUMP)
    glider = load_env(GLIDER)
    assert jump.issues == []
    assert [(i.severity, i.message) for i in glider.issues] == [("info", S.SPEC_BRAINLESS)]
    # 형식 2 파일이다(읽은 모델의 version은 지금 형식으로 변환된 값이라 v2.1부터 3: 파일 글과 가장 낮은 형식으로 본다)
    assert feature_version(jump.file) == feature_version(glider.file) == 2
    assert json.loads(JUMP.read_bytes())["version"] == json.loads(GLIDER.read_bytes())["version"] == 2
    assert len(jump.inputs) == 11 and jump.outputs == ["jump"] and layout_for(jump).dim == 209
    assert layout_for(glider).dim == 8 and jump.file.evaluation.trials == 5 and glider.file.evaluation.trials == 3
    assert "Space" in jump.file.actuators[0].keys
    assert jump.file.editor["view"] == [-3.0, -1.0, 22.0, 6.0] and glider.file.editor["view"] == [-4.0, -2.0, 6.0, 24.0]


def test_jump_game_is_solvable_with_full_information() -> None:
    """T19: 완전 정보 계획기(여유 8cm)가 판정 시드 100개 중 95개 이상에서 계획을 찾고, 그 계획을 열린 고리로 다시 돌려
    95개 이상이 끝까지 버틴다(설계 원형 99·99). 안 뜀·늘 뜀은 모두 실패, 첫 1초에 장애물이 선수와 겹치는 시드 0."""
    spec = load_env(JUMP)
    profile = measure_profile(spec)
    assert profile.apex == pytest.approx(1.885, abs=0.01) and profile.ready_after == 76
    feasible = replayed = 0
    for trial in seeds.judge_trials(100):
        assert first_second_clear(spec, trial.seed), trial.seed
        ok, jumps = plan(spec, trial.seed, profile)
        feasible += ok
        if ok:
            replayed += run_open_loop(spec, trial.seed, jumps)[0]
        assert not run_open_loop(spec, trial.seed, const=0.0)[0]
        assert not run_open_loop(spec, trial.seed, const=1.0)[0]
    assert feasible >= 95 and replayed >= 95, (feasible, replayed)


def glider_distances(spec: Any, genome: np.ndarray, count: int = 100) -> np.ndarray:
    layout = layout_for(spec)
    res = evaluate(spec, layout, genome[None, :], [(0, t) for t in seeds.judge_trials(count)])
    k = [b.id for b in spec.file.scoring.blocks].index("s_dist")
    return res.blocks[:, k]


def test_glider_initial_body_flies_about_113_m() -> None:
    """T34: 글라이더 처음 몸(학습 없음) 판정 100의 비행 거리 중앙값이 100~130m(원형 112.8m)."""
    spec = load_env(GLIDER)
    dist = glider_distances(spec, layout_for(spec).initial())
    assert 100.0 <= float(np.median(dist)) <= 130.0


def test_builtin_order_and_unknown_files_by_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T35: 착륙선, 트랙 자동차, 언덕차, 점프 게임, 글라이더, (v2.1) 바퀴 고르는 언덕차, 그 밖은 이름순."""
    assert [p.name for p in paths.builtin_envs()] == [
        "lander.evoenv", "track_car.evoenv", "hill_car.evoenv", "jump.evoenv", "glider.evoenv", "wheels.evoenv"]
    for name in ("zebra", "glider", "apple", "lander", "jump"):
        (tmp_path / f"{name}.evoenv").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr(paths, "builtin_env_dir", lambda: tmp_path)
    assert [p.name for p in paths.builtin_envs()] == [
        "lander.evoenv", "jump.evoenv", "glider.evoenv", "apple.evoenv", "zebra.evoenv"]


def test_first_run_copies_only_new_examples(tmp_path: Path) -> None:
    """T35: v2.0을 처음 켜면 새 예제(점프 게임, 글라이더, v2.1 바퀴 고르는 언덕차)만 복사하고 사용자가 고친 옛 예제는 그대로 둔다."""
    for name in ("lander", "track_car", "hill_car"):
        (tmp_path / f"{name}.evoenv").write_text(f"고친 {name}", encoding="utf-8")
    copied = paths.install_examples(tmp_path)
    assert [p.name for p in copied] == ["jump.evoenv", "glider.evoenv", "wheels.evoenv"]
    assert (tmp_path / "jump.evoenv").read_bytes() == JUMP.read_bytes()
    for name in ("lander", "track_car", "hill_car"):
        assert (tmp_path / f"{name}.evoenv").read_text(encoding="utf-8") == f"고친 {name}"
    assert paths.install_examples(tmp_path) == []


def test_selftest_checks_the_new_examples() -> None:
    """T36: 자체 점검이 새 예제를 컴파일하고, 점프 게임(과 글라이더)을 두 번 돌린 결과가 같다."""
    bundle = selftest._check_bundle_files()
    assert "jump.evoenv" in bundle and "glider.evoenv" in bundle
    assert {"jump.evoenv", "glider.evoenv"} <= set(selftest.REQUIRED_EXAMPLES)
    first = selftest._check_v2_examples()
    assert "jump.evoenv" in first and "glider.evoenv" in first
    assert selftest._check_v2_examples() == first


# ---------------------------------------------------------------------------
# T32, T33: 9.3 v2.0 완료 기준 절차 (느림: 측정 단계에서 돌린다)
# ---------------------------------------------------------------------------


def train_and_pick_champion(env: Path, seed: int, budget: int, out: Path) -> tuple[Any, np.ndarray, int]:
    """명령줄 학습(CMA-ES 기본값, 자동 재시작 끔, 고정 실행 시드)을 학습 시험 budget회까지 돌리고, 그 예산까지의 마지막
    챔피언(stats.csv에서 evaluations ≤ budget인 마지막 줄의 champion_generation)을 고른다(설계 9.1의 판정 절차)."""
    code = cli.main(["train", str(env), "--seed", str(seed), "--workers", str(default_workers()),
                     "--max-evaluations", str(budget), "--no-restart", "--out", str(out)])
    assert code == 0
    run = next(out.iterdir())
    rows = list(csv.DictReader((run / "stats.csv").open(encoding="utf-8-sig")))
    ok = [r for r in rows if int(r["evaluations"]) <= budget and r["champion_generation"] not in ("", None)
          and int(float(r["champion_generation"])) >= 1]
    gen = int(float(ok[-1]["champion_generation"]))
    genome = np.load(run / "champions" / f"gen_{gen:06d}_champion.npz")["genome"]
    spec = load_env(sorted((run / "envs").glob("*.evoenv"))[-1])
    return spec, genome, gen


@pytest.mark.slow
def test_jump_game_completion_criterion(tmp_path: Path) -> None:
    """T32 (9.3 v2.0 기준 1): 고정 실행 시드 1·2·3, 학습 시험 12,000회 챔피언, 판정 100개에서 끝까지 버틴 비율의 중앙값 ≥ 50%."""
    rates = []
    for seed in (1, 2, 3):
        spec, genome, gen = train_and_pick_champion(JUMP, seed, 12_000, tmp_path / f"s{seed}")
        layout = layout_for(spec)
        res = evaluate(spec, layout, genome[None, :], [(0, t) for t in seeds.judge_trials(100)])
        rates.append(sum(1 for o in res.outcomes if o == "success") / 100)
        print(json.dumps({"seed": seed, "champion_generation": gen, "success": rates[-1]}))
    assert float(np.median(rates)) >= 0.5, rates


@pytest.mark.slow
def test_glider_completion_criterion(tmp_path: Path) -> None:
    """T33 (9.3 v2.0 기준 2): 고정 시드 1·2·3, 학습 시험 6,000회 챔피언, 판정 100개 비행 거리 중앙값 ≥ 190m이고
    처음 몸 판정 중앙값의 1.6배 이상."""
    base = load_env(GLIDER)
    initial = float(np.median(glider_distances(base, layout_for(base).initial())))
    medians = []
    for seed in (1, 2, 3):
        spec, genome, gen = train_and_pick_champion(GLIDER, seed, 6_000, tmp_path / f"s{seed}")
        medians.append(float(np.median(glider_distances(spec, genome))))
        print(json.dumps({"seed": seed, "champion_generation": gen, "median_m": medians[-1], "initial_m": initial}))
    median = float(np.median(medians))
    assert median >= 190.0 and median >= 1.6 * initial, (medians, initial)

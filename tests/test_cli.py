"""명령줄 학습(evolab train)의 v6.1 자라는 뇌 (v6 설계 5.5·4.16, 8.2 T42).

- train --method grow --grow-start --grow-every가 같은 설정의 학습기(실험실 학습 프로세스와 같은 길: create_run + Trainer)와 같은
  통계를 내고, 시작 요약 줄(TRAIN_GROW_SUMMARY)과 자란 세대마다 CLI_TRAIN_GROWN 한 줄을 쓴다.
- --resume은 grow_*를 바꾸지 못하고(TRAIN_RESUME_GROW_FIXED, 종료 코드 2) 그냥 이어 가면 계속 자란다. judge·bundle·reproduce가 된다.
- IPOP과 함께, 은닉층 없는 뇌, 1보다 작은 값은 실행 폴더를 만들기 전에 거절한다. 크기 고정 실행에는 새 줄·새 키가 없다.
- 처음 뉴런 수가 틀을 이미 채우면(--grow-start 16 이상) 시작 줄이 「두 배」를 약속하지 않는다(TRAIN_GROW_SUMMARY_FULL, v6.1 고침).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evolab import cli, paths
from evolab.app import strings as S
from evolab.engine import trainer as trainer_mod
from evolab.engine.pool import WorkerPool
from evolab.engine.runsetup import create_run
from evolab.engine.trainer import GROW_COLUMNS, GROW_KEYS, TrainConfig, Trainer
from evolab.runs.store import RunStore
from evolab.spec.io import compile_env

LANDER = paths.builtin_env_dir() / "lander.evoenv"
LANDER_TEXT = LANDER.read_text(encoding="utf-8")
TIME = set(trainer_mod.TIME_COLUMNS)
BASE = ["--seed", "3", "--popsize", "4", "--trials", "1", "--workers", "0"]


def strip(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in r.items() if k not in TIME} for r in rows]


def only_run(base: Path) -> Path:
    runs = [p for p in base.iterdir() if p.is_dir()]
    assert len(runs) == 1, runs
    return runs[0]


def test_parser_accepts_grow_options() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["train", str(LANDER), "--method", "grow", "--grow-start", "3", "--grow-every", "5"])
    assert (args.method, args.grow_start, args.grow_every) == ("grow", 3, 5)
    plain = parser.parse_args(["train", str(LANDER)])
    assert plain.grow_start is None and plain.grow_every is None
    text = parser._subparsers._group_actions[0].choices["train"].format_help()  # type: ignore[union-attr]
    assert "--grow-start" in text and "--grow-every" in text and "grow" in text
    with pytest.raises(SystemExit):
        parser.parse_args(["train", str(LANDER), "--grow-every", "two"])


def test_train_grow_matches_the_same_config_and_prints_growth(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cli"
    capsys.readouterr()
    code = cli.main(["train", str(LANDER), "--method", "grow", "--grow-start", "2", "--grow-every", "10",
                     "--max-generations", "12", "--out", str(out), *BASE])
    printed = capsys.readouterr().out
    assert code == 0, printed
    lines = printed.splitlines()
    summary = S.TRAIN_GROW_SUMMARY.format(start=2, full=16, every=10)
    k = lines.index(summary)
    assert lines[k - 1].startswith(S.TRAIN_STABILITY.split("{")[0])  # 자동 재시작·예선 줄 바로 뒤
    assert S.METHOD_NAMES["grow"] in printed
    assert [ln for ln in lines if ln == S.CLI_TRAIN_GROWN.format(gen=10, old=2, new=4)] == [
        S.CLI_TRAIN_GROWN.format(gen=10, old=2, new=4)]
    assert sum(1 for ln in lines if ln.endswith("개)") and "뇌가 자랐습니다" in ln) == 1
    run = only_run(out)
    store = RunStore.open(run)
    info = store.read_run_info()
    assert (info["method"], info["config"]["grow_start"], info["config"]["grow_every"]) == ("grow", 2, 10)
    header = store.stats_columns() or []
    assert tuple(header[header.index("switch") + 1:header.index("switch") + 4]) == GROW_COLUMNS
    # 같은 설정을 학습기로(실험실 학습 프로세스의 새 실행 길: TrainConfig.for_new_run → create_run → Trainer)
    spec = compile_env(LANDER_TEXT)
    config = TrainConfig.for_new_run(spec, method="grow", grow_start=2, grow_every=10, run_seed=3, popsize=4, trials=1,
                                     max_generations=12)
    assert config.to_dict() == info["config"]
    direct = create_run(tmp_path / "direct", LANDER_TEXT, spec, config, 0)
    with WorkerPool(0) as pool:
        Trainer(LANDER_TEXT, config, pool, direct).run()
    assert strip(direct.read_stats()) == strip(store.read_stats())
    rows = store.read_stats()
    assert [(r["generation"], r["brain_neurons"]) for r in rows if r["grow"] == "1"] == [("10", "2")]
    assert rows[10]["brain_neurons"] == "4"


def test_resume_cannot_change_grow_settings_but_keeps_growing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "runs"
    assert cli.main(["train", str(LANDER), "--method", "grow", "--grow-every", "3", "--max-generations", "4",
                     "--validate-every", "8", "--out", str(out), *BASE]) == 0
    run = only_run(out)
    rows_before = RunStore.open(run).read_stats()
    for extra in (["--grow-start", "4"], ["--grow-every", "5"], ["--grow-start", "2", "--grow-every", "3"]):
        capsys.readouterr()
        assert cli.main(["train", "--resume", str(run), "--workers", "0", "--max-generations", "8", *extra]) == 2
        assert S.TRAIN_RESUME_GROW_FIXED in capsys.readouterr().err
        assert RunStore.open(run).read_stats() == rows_before  # 아무것도 바꾸지 않음
    capsys.readouterr()
    assert cli.main(["train", "--resume", str(run), "--workers", "0", "--max-generations", "7"]) == 0
    printed = capsys.readouterr().out
    assert S.CLI_TRAIN_GROWN.format(gen=6, old=4, new=8) in printed
    rows = RunStore.open(run).read_stats()
    assert [(r["generation"], r["brain_neurons"], r["grow"]) for r in rows[2:]] == [
        ("3", "2", "1"), ("4", "4", "0"), ("5", "4", "0"), ("6", "4", "1"), ("7", "8", "0")]
    # 판정·재현 묶음이 된다
    capsys.readouterr()
    assert cli.main(["judge", str(run), "--seeds", "4", "--workers", "0"]) == 0
    bundle = tmp_path / "bundle.zip"
    assert cli.main(["bundle", str(run), "--out", str(bundle)]) == 0
    capsys.readouterr()
    assert cli.main(["reproduce", str(bundle), "--out", str(tmp_path / "again"), "--workers", "0"]) == 0
    assert S.BUNDLE_SAME.format(gens=7) in capsys.readouterr().out


@pytest.mark.parametrize(("extra", "env", "message"), [
    (["--ipop"], None, S.TRAIN_GROW_NO_IPOP),
    (["--grow-every", "0"], None, S.TRAIN_CONFIG_INVALID.format(name="grow_every", value=0)),
    (["--grow-start", "-1"], None, S.TRAIN_CONFIG_INVALID.format(name="grow_start", value=-1)),
    ([], "no_hidden", S.TRAIN_GROW_NO_HIDDEN),
    ([], "glider", S.FRAME_NO_BRAIN),
])
def test_bad_grow_runs_are_refused_before_creating_a_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                            extra: list[str], env: str | None, message: str) -> None:
    path = LANDER
    if env == "no_hidden":
        data = json.loads(LANDER_TEXT)
        data["brain"]["hidden"] = []
        path = tmp_path / "no_hidden.evoenv"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    elif env == "glider":
        path = paths.builtin_env_dir() / "glider.evoenv"
    out = tmp_path / "runs"
    capsys.readouterr()
    assert cli.main(["train", str(path), "--method", "grow", "--max-generations", "2", "--out", str(out),
                     *BASE, *extra]) == 2
    assert S.TRAIN_CANNOT_START.format(reason=message) in capsys.readouterr().err
    assert not out.exists() or not any(out.iterdir())


@pytest.mark.parametrize("start", [16, 40])
def test_start_that_already_fills_the_frame_does_not_promise_doubling(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                                      start: int) -> None:
    """--grow-start가 은닉층 크기(16) 이상이면 학습 방법이 틀로 잘라 처음부터 다 자란 크기라 한 번도 자라지 않는다. 시작 줄은
    「뉴런 16개 → 16개, 10세대마다 두 배」가 아니라 자라지 않는다고 말한다(실험실 안내 줄 LAB_GROW_HINT_ALREADY와 같은 경우)."""
    out = tmp_path / "runs"
    capsys.readouterr()
    assert cli.main(["train", str(LANDER), "--method", "grow", "--grow-start", str(start), "--grow-every", "1",
                     "--max-generations", "3", "--out", str(out), *BASE]) == 0
    lines = capsys.readouterr().out.splitlines()
    full = S.TRAIN_GROW_SUMMARY_FULL.format(full=16)
    k = lines.index(full)
    assert lines[k - 1].startswith(S.TRAIN_STABILITY.split("{")[0])  # 자동 재시작·예선 줄 바로 뒤(자라는 실행과 같은 자리)
    assert not [ln for ln in lines if "두 배" in ln or "→" in ln or "뇌가 자랐습니다" in ln]
    assert "자라지 않습니다" in full and "16" in full
    rows = RunStore.open(only_run(out)).read_stats()
    assert [(r["brain_neurons"], r["grow"]) for r in rows] == [("16", "0")] * 3  # 정말 자라지 않는다
    # 한 층이라도 덜 찼으면 지금처럼 자라는 줄
    spec = compile_env(LANDER_TEXT)
    from evolab.engine import commands
    from evolab.policy.genome import layout_for

    layout = layout_for(spec)
    assert commands._grow_text(TrainConfig(method="grow", grow_start=15, grow_every=4), layout) == \
        S.TRAIN_GROW_SUMMARY.format(start=15, full=16, every=4)
    assert commands._grow_text(TrainConfig(method="grow", grow_start=start), layout) == full
    assert commands._grow_text(TrainConfig(method="cmaes", grow_start=start), layout) is None


def test_fixed_size_train_has_no_grow_line_or_keys(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "runs"
    capsys.readouterr()
    assert cli.main(["train", str(LANDER), "--method", "cmaes", "--grow-start", "5", "--max-generations", "2",
                     "--out", str(out), *BASE]) == 0
    printed = capsys.readouterr().out
    assert "자라는 뇌" not in printed and "뇌가 자랐습니다" not in printed
    store = RunStore.open(only_run(out))
    assert not set(GROW_KEYS) & set(store.read_run_info()["config"])
    assert not set(GROW_COLUMNS) & set(store.stats_columns() or [])
    _gen, ckpt = store.load_checkpoint(None)
    assert not set(GROW_KEYS) & set(ckpt["config"]) and "stds" not in ckpt["learner"]["config"]

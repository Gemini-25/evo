"""새 실행 설정과 run.json (스포츠 설계 6.1 T7의 R 몫, D31, DE T7).

바탕 단계(tests/test_foundation_v62.py)가 for_new_run의 사전을 본다. 여기서는 그 설정이 실제 실행 폴더에 적히는지를 본다.

- 혼자 하는 실행을 명령줄 train으로 만들면 run.json의 키 모양·config 사전·통계 열 머리가 얼린 v6.1.0과 같다(예제 6개 중
  환경이 다른 셋).
- editor.train이 있는 환경으로 만든 실행은 run.json의 config에 그 흰 목록 값이 적히고, 부르는 쪽 값(명령줄 인자)이 먼저다.
- 형식 4 환경(균형 보조가 든 달리기 원형 파일 + editor.train)도 명령줄 학습 한 세대가 돌고, 실행 폴더의 환경 사본은 형식 4 글이다.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab import cli
from evolab.engine.trainer import TIME_COLUMNS, TrainConfig
from evolab.runs.store import RunStore
from evolab.spec.io import dump_env, parse_env
from tests.oracle.frozen import frozen_run, live_run

RELEASE = "v6.1.0"
ROOT = Path(__file__).resolve().parents[1]
MEASURED = ROOT / "tests" / "data" / "presets" / "running_measured.evoenv"
TIME = set(TIME_COLUMNS)
SMALL = ["--seed", "3", "--popsize", "4", "--trials", "1", "--max-generations", "1", "--workers", "0", "--no-observe"]


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("EVOLAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EVOLAB_STATE_DIR", str(tmp_path / "state"))
    return tmp_path


def train(env: Path, out: Path, *extra: str) -> RunStore:
    assert cli.main(["train", str(env), "--out", str(out), *SMALL, *extra]) == 0
    [run] = [p for p in out.iterdir() if p.is_dir()]
    return RunStore.open(run)


def shape(value: Any) -> Any:
    """사전의 키 모양(값은 형 이름만). 목록은 첫 원소의 모양."""
    if isinstance(value, dict):
        return {k: shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return type(value).__name__


RUN_JSON = """
import json
import sys
from pathlib import Path

import numpy as np

from evolab import cli
from evolab.runs.store import RunStore

out, work, *stems = sys.argv[1:]
rows = {}
for stem in stems:
    base = Path(work) / stem
    argv = ["train", f"environments/{stem}.evoenv", "--out", str(base), "--seed", "3", "--popsize", "4", "--trials", "1",
            "--max-generations", "1", "--workers", "0", "--no-observe"]
    assert cli.main(argv) == 0
    [run] = [p for p in base.iterdir() if p.is_dir()]
    store = RunStore.open(run)
    rows[stem] = [store.read_run_info(), store.stats_columns(), store.read_stats()]
np.savez(out, rows=np.array(json.dumps(rows, ensure_ascii=False)))
"""


def test_t7_solo_run_json_has_the_same_keys_and_config_as_the_frozen_release(isolated: Path) -> None:
    script = isolated / "run_json.py"
    script.write_text(textwrap.dedent(RUN_JSON).lstrip(), encoding="utf-8")
    stems = ("lander", "jump", "wheels")
    (isolated / "old").mkdir()
    (isolated / "new").mkdir()
    old = json.loads(str(frozen_run(RELEASE, script, isolated / "old", *stems)["rows"]))
    new = json.loads(str(live_run(script, isolated / "new", *stems)["rows"]))
    for stem in stems:
        (oi, oc, os_), (ni, nc, ns) = old[stem], new[stem]
        assert shape(oi) == shape(ni), stem
        assert oi["config"] == ni["config"], stem
        assert (oi["version"], ni["version"]) == ("6.1.0", ni["version"]) and ni["version"] != "6.1.0"
        for k in ("method", "seed_scheme", "layout", "layout_signature", "env_name", "env_versions"):
            assert oi[k] == ni[k], (stem, k)  # env_versions의 id는 환경 해시: 같은 환경 사본
        assert oc == nc, stem
        # 통계 줄도 같다(시간 열은 빼고): 같은 학습이 돌았다
        assert [{k: v for k, v in r.items() if k not in TIME} for r in os_] == \
               [{k: v for k, v in r.items() if k not in TIME} for r in ns], stem


def _with_train(data: dict[str, Any], train: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(data))
    out.setdefault("editor", {})["train"] = train
    return out


def test_t7_preset_training_values_reach_run_json(isolated: Path) -> None:
    lander = json.loads((ROOT / "environments" / "lander.evoenv").read_text(encoding="utf-8"))
    env = isolated / "preset.evoenv"
    env.write_text(dump_env(parse_env(_with_train(lander, {"qualify_entrants": 2, "qualify_trials": 2000,
                                                               "restart_patience": None, "popsize": 99}))),
                   encoding="utf-8")
    store = train(env, isolated / "runs")
    config = store.read_run_info()["config"]
    assert (config["qualify_entrants"], config["qualify_trials"], config["restart_patience"]) == (2, 2000, None)
    assert config["popsize"] == 4  # 흰 목록 밖(popsize)은 얹지 않고, 명령줄 값이 쓰인다
    assert set(config) == set(TrainConfig.for_new_run().to_dict())
    # 명령줄 인자가 프리셋 권장값보다 먼저다
    store2 = train(env, isolated / "runs2", "--qualify", "1")
    assert store2.read_run_info()["config"]["qualify_entrants"] == 1


def test_t7_a_format_4_preset_trains_and_keeps_its_format(isolated: Path) -> None:
    data = _with_train(json.loads(MEASURED.read_text(encoding="utf-8")), {"qualify_entrants": 3, "qualify_trials": 9600})
    env = isolated / "running.evoenv"
    env.write_text(dump_env(parse_env(data)), encoding="utf-8")
    store = train(env, isolated / "runs")
    info = store.read_run_info()
    assert (info["config"]["qualify_entrants"], info["config"]["qualify_trials"]) == (3, 9600)
    text = store.env_text(info["env_versions"][0]["id"])
    saved = json.loads(text)
    assert saved["version"] == 4 and [a["kind"] for a in saved["actuators"]][-1] == "balance"
    rows = store.read_stats()
    assert len(rows) == 1 and np.isfinite(float(rows[0]["score_best"]))

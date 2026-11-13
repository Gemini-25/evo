"""얼린 v6.1.0과 지금 작업 나무의 바이트 비교 (스포츠 설계 6.1 T1·T3·T4, 묶음 R, 3.2 DE).

6.2.0의 형식 4 부품(World.view·World.damping·Move·Balance)은 옛 환경(형식 1~3)의 길을 바꾸지 않는다. 그 증명은 얼린 출시
나무(tests/oracle/v6.1.0-src.tar.gz, tests/oracle/frozen.py)를 딴 프로세스에서 돌린 결과와 지금 나무를 같은 방식의 딴
프로세스에서 돌린 결과를 견주는 것이다. 정답 해시를 박지 않는다(같은 기계에서 두 번 돌려 비교).

- T1: 내장 예제 6개와 v6.1.0 커밋에 있던 tests/data의 환경(SOURCES.txt의 frozen-env 줄)을 읽어 다시 저장한 글, 해시, 검사 글
  목록(심각도·자리·글)이 얼린 v6.1과 같다.
- T3: 예제 6개 × 시드 20 × 유전자 3을 평가 창 1·4·12로 평가한 모든 칸(점수·스텝·블록·가중 블록·행동 크기·종료 사건·결과,
  궤적 자세·출력, 관찰 코드 1·2의 횟수·합·칸)이 같다.
- T4: C 빠른 길이 쓰임, 엔진 재생(추적 없음·"brain"·전부)과 사람 기록 재생, v6.1 시험 자료 실행(tests/data/v61_grow_run)의
  이어하기·재현 묶음이 같은 통계·챔피언을 낸다.

스크립트는 이 파일 안의 글을 임시 폴더에 써서 돌린다(두 나무에 모두 있는 이름만 쓴다: frozen.py의 약속).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tests.oracle.frozen import frozen_envs, frozen_run, live_run

RELEASE = "v6.1.0"
ROOT = Path(__file__).resolve().parents[1]
OLD_SIX = ("lander", "track_car", "hill_car", "jump", "glider", "wheels")


def write_script(tmp: Path, name: str, body: str) -> Path:
    path = tmp / name
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


def both(script: Path, *args: object) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return frozen_run(RELEASE, script, *args), live_run(script, *args)


def same_arrays(old: dict[str, np.ndarray], new: dict[str, np.ndarray], skip: tuple[str, ...] = ()) -> None:
    assert sorted(old) == sorted(new)
    for k in old:
        if k in skip:
            continue
        a, b = old[k], new[k]
        assert a.dtype == b.dtype and a.shape == b.shape, k
        assert a.tobytes() == b.tobytes(), k


# ---------------------------------------------------------------------------
# T1: 다시 저장한 글·해시·검사 글
# ---------------------------------------------------------------------------

RESAVE = """
import json
import sys

import numpy as np

from evolab.spec.compiled import check_env
from evolab.spec.io import dump_env, env_hash, parse_env


def main(out, *paths):
    rows = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if "env" in data and "version" not in data:  # v1_dump·v2_dump: {"env": 환경 사전, ...}
            data = data["env"]
        env = parse_env(data)
        issues = [[i.severity, i.where, i.message] for i in check_env(env)]
        rows[p] = [dump_env(env), env_hash(env), issues]
    np.savez(out, rows=np.array(json.dumps(rows, ensure_ascii=False)))


if __name__ == "__main__":
    main(*sys.argv[1:])
"""


def env_files() -> list[str]:
    out = [f"environments/{n}.evoenv" for n in OLD_SIX]
    out += sorted(frozen_envs(RELEASE))
    return out


def test_t1_old_files_resave_to_the_same_text_hash_and_issues(tmp_path: Path) -> None:
    files = env_files()
    assert len(files) >= 6 + 10
    for f in files:
        data = json.loads((ROOT / f).read_text(encoding="utf-8"))
        if "env" in data and "version" not in data:
            data = data["env"]
        assert isinstance(data, dict) and data.get("version", 1) <= 3, f
    script = write_script(tmp_path, "resave.py", RESAVE)
    old, new = both(script, *files)
    a, b = json.loads(str(old["rows"])), json.loads(str(new["rows"]))
    assert list(a) == files
    for f in files:
        assert a[f][0] == b[f][0], f"{f}: 다시 저장한 글이 다름"
        assert a[f][1] == b[f][1], f"{f}: 해시가 다름"
        assert a[f][2] == b[f][2], f"{f}: 검사 글이 다름"


# ---------------------------------------------------------------------------
# T3: 예제 6개 × 시드 20 × 유전자 3, 평가 창 1·4·12
# ---------------------------------------------------------------------------

EVAL_GRID = """
import json
import sys

import numpy as np

from evolab.engine.evaluate import evaluate
from evolab.engine.seeds import train_trials
from evolab.policy.genome import layout_for
from evolab.spec.io import load_env

WINDOWS = (1, 4, 12)


def main(out, n_genomes, n_seeds, *paths):
    res = {}
    for path in paths:
        spec = load_env(path)
        layout = layout_for(spec)
        rng = np.random.default_rng([620, layout.dim])
        genomes = layout.initial()[None, :] + 0.6 * rng.standard_normal((int(n_genomes), layout.dim))
        trials = train_trials(62, 1, int(n_seeds))
        pairs = [(g, t) for g in range(int(n_genomes)) for t in trials]
        n = len(pairs)
        stem = path.rsplit("/", 1)[-1].split(".")[0]
        for w in WINDOWS:
            r = evaluate(spec, layout, genomes, pairs, record=[i % 4 == 0 for i in range(n)], every=1 + w % 3,
                         observe=[i % 3 for i in range(n)], frame=(-50.0, -50.0, 1.0, 200, 200), window=w)
            p = f"{stem}_w{w}_"
            res[p + "scores"] = r.scores
            res[p + "steps"] = r.steps
            res[p + "blocks"] = r.blocks
            res[p + "action_abs"] = r.action_abs
            res[p + "weighted"] = r.weighted
            res[p + "ends"] = np.array(json.dumps([r.end_events, r.outcomes]))
            tr = [t for t in (r.trajectories or []) if t is not None]
            res[p + "poses"] = np.concatenate([t.poses.reshape(-1) for t in tr])
            res[p + "outputs"] = np.concatenate([t.outputs.reshape(-1) for t in tr])
            res[p + "traj"] = np.array(json.dumps([[t.every, repr(t.dt), t.end_step] for t in tr]))
            obs = [o for o in (r.observations or []) if o is not None]
            res[p + "obs"] = np.array(json.dumps([
                [o.hist.tolist(), repr(o.out_sum.tolist()), o.steps, o.cells.tolist(), o.counts.tolist(), o.outside]
                for o in obs]))
            res[p + "n_obs"] = np.array(len(obs))
    np.savez(out, **res)


if __name__ == "__main__":
    main(*sys.argv[1:])
"""


@pytest.fixture(scope="module")
def eval_grid(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    script = write_script(tmp_path_factory.mktemp("grid"), "eval_grid.py", EVAL_GRID)
    return both(script, 3, 20, *(f"environments/{n}.evoenv" for n in OLD_SIX))


def test_t3_evaluation_grid_is_byte_identical_to_the_frozen_release(
    eval_grid: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
) -> None:
    old, new = eval_grid
    same_arrays(old, new)
    for n in OLD_SIX:
        assert int(new[f"{n}_w1_steps"].sum()) > 0
        assert int(new[f"{n}_w1_n_obs"]) > 0
        assert new[f"{n}_w1_scores"].shape == (60,)


def test_t3_windows_agree_with_each_other(eval_grid: tuple[dict[str, np.ndarray], dict[str, np.ndarray]]) -> None:
    """평가 창은 결과를 바꾸지 않는다(같은 나무 안에서도). 궤적 간격(every)은 창마다 다르게 주었으므로 견주지 않는다."""
    _old, new = eval_grid
    for n in OLD_SIX:
        for f in ("scores", "steps", "blocks", "action_abs", "ends"):
            assert new[f"{n}_w1_{f}"].tobytes() == new[f"{n}_w4_{f}"].tobytes() == new[f"{n}_w12_{f}"].tobytes(), (n, f)


# ---------------------------------------------------------------------------
# T4: 빠른 길, 재생, 사람 기록, v6.1 실행의 이어하기·재현
# ---------------------------------------------------------------------------

REPLAYS = """
import json
import math
import sys

import numpy as np

from evolab.app.play import record as record_mod
from evolab.engine import replay as replay_mod
from evolab.engine import seeds
from evolab.policy.genome import layout_for
from evolab.spec.io import load_env
from evolab.world.physics2d.episode import fastpath_in_use


def key_actions(n_out, steps, seed):
    rng = np.random.default_rng(seed)
    choices = [-1.0, 0.0, 0.0, 1.0, 1.0, 0.5, -0.5, 2.0, -2.0]
    out, held = [], [0.0] * n_out
    for k in range(steps):
        if k % 7 == 0:
            held = [choices[int(rng.integers(len(choices)))] for _ in range(n_out)]
        row = list(held)
        if k % 97 == 13 and n_out:
            row[int(rng.integers(n_out))] = [math.nan, math.inf, -math.inf, 1e300][k % 4]
        out.append(row)
    return out


def main(out, *stems):
    res = {}
    for stem in stems:
        spec = load_env(f"environments/{stem}.evoenv")
        layout = layout_for(spec)
        res[f"{stem}_fast"] = np.array(bool(fastpath_in_use(spec)))
        champ = np.load(f"tests/data/v6_champions/{stem}.npz")["genome"]
        rng = np.random.default_rng([63, layout.dim])
        cases = [(champ, seeds.display_trial()),
                 (layout.initial() + 0.6 * rng.standard_normal(layout.dim), seeds.train_trials(9, 1, 1)[0])]
        for k, (g, t) in enumerate(cases):
            for trace in (False, "brain", True):
                r = replay_mod.replay(spec, layout, g, t, every=1 + k, trace=trace)
                p = f"{stem}_{k}_{trace}_"
                res[p + "poses"] = r.trajectory.poses
                res[p + "outputs"] = r.trajectory.outputs
                res[p + "result"] = np.array(repr(r.result))
                res[p + "raw"] = np.array(repr(r.raw))
                for f in ("observations", "rays"):
                    x = getattr(r, f)
                    if x is not None:
                        res[p + f] = x
                for i, a in enumerate(r.activations or []):
                    res[p + f"act{i}"] = a
        rec = record_mod.new_record(spec, seeds.derive(78, len(stem)))
        rec.actions = key_actions(len(spec.outputs), 400, seed=7 + len(stem))
        res[f"{stem}_human"] = np.array(repr(record_mod.replay(spec, rec)))
    np.savez(out, **res)


if __name__ == "__main__":
    main(*sys.argv[1:])
"""


def test_t4_fast_path_replays_and_human_records_match_the_frozen_release(tmp_path: Path) -> None:
    script = write_script(tmp_path, "replays.py", REPLAYS)
    old, new = both(script, *OLD_SIX)
    same_arrays(old, new)
    for n in OLD_SIX:
        assert bool(new[f"{n}_fast"]), f"{n}: C 빠른 길을 쓰지 않음"


RESUME = """
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from evolab.engine import trainer as trainer_mod
from evolab.engine.pool import WorkerPool
from evolab.engine.trainer import Trainer
from evolab.runs.bundle import reproduce
from evolab.runs.store import RunStore

TIME = set(trainer_mod.TIME_COLUMNS)


def strip(rows):
    return [{k: str(v) for k, v in r.items() if k not in TIME} for r in rows]


def main(out, work):
    work = Path(work)
    run = work / "run"
    shutil.copytree("tests/data/v61_grow_run/run", run)
    store = RunStore.open(run)
    with WorkerPool(0) as pool:
        Trainer.resume(store, pool, None, max_generations=14).run()
    rows = strip(store.read_stats())
    champs = [store.load_champion(g, "representative")["genome"] for g in range(1, 15)]
    bundle = work / "bundle.zip"
    shutil.copyfile("tests/data/v61_grow_run/bundle.zip", bundle)
    again = RunStore.open(reproduce(bundle, work / "again", workers=0))
    np.savez(out, rows=np.array(json.dumps(rows)), columns=np.array(json.dumps(store.stats_columns())),
             champs=np.stack(champs), again=np.array(json.dumps(strip(again.read_stats()))),
             again_columns=np.array(json.dumps(again.stats_columns())))


if __name__ == "__main__":
    main(*sys.argv[1:])
"""


def test_t4_v61_fixture_run_resumes_and_reproduces_like_the_frozen_release(tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOLAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EVOLAB_STATE_DIR", str(tmp_path / "state"))
    script = write_script(tmp_path, "resume.py", RESUME)
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    old = frozen_run(RELEASE, script, tmp_path / "old")
    new = live_run(script, tmp_path / "new")
    same_arrays(old, new)
    rows = json.loads(str(new["rows"]))
    assert [int(r["generation"]) for r in rows] == list(range(1, 15))
    assert len(json.loads(str(new["again"]))) == 12


def test_t4_frozen_scripts_see_their_own_tree(tmp_path: Path) -> None:
    """견줌이 헛돌지 않는다: 얼린 쪽은 6.1.0, 지금 쪽은 6.2.0 프로그램이고 지금 쪽만 형식 4 부품을 안다."""
    script = write_script(tmp_path, "who.py", """
        import sys

        import numpy as np

        import evolab
        from evolab.spec import model

        np.savez(sys.argv[1], version=np.array(evolab.__version__), current=np.array(model.CURRENT_VERSION),
                 move=np.array(hasattr(model, "Move")))
    """)
    old, new = both(script)
    assert (str(old["version"]), int(old["current"]), bool(old["move"])) == ("6.1.0", 3, False)
    assert int(new["current"]) == 4 and bool(new["move"]) and str(new["version"]) != "6.1.0"



# ---------------------------------------------------------------------------
# S1 (느림): 실제 학습의 통계·챔피언
# ---------------------------------------------------------------------------

TRAIN = """
import json
import sys
from pathlib import Path

import numpy as np

from evolab import cli
from evolab.engine import trainer as trainer_mod
from evolab.runs.store import RunStore

out, work, stem, *limit = sys.argv[1:]
base = Path(work) / stem
argv = ["train", f"environments/{stem}.evoenv", "--out", str(base), "--seed", "1", "--workers", "0", *limit]
assert cli.main(argv) == 0
[run] = [p for p in base.iterdir() if p.is_dir()]
store = RunStore.open(run)
time_cols = set(trainer_mod.TIME_COLUMNS)
rows = [{k: v for k, v in r.items() if k not in time_cols} for r in store.read_stats()]
gens = store.champion_generations("champion")
champ = store.load_champion(gens[-1], "champion")["genome"] if gens else np.zeros(0)
np.savez(out, rows=np.array(json.dumps(rows)), columns=np.array(json.dumps(store.stats_columns())),
         config=np.array(json.dumps(store.read_run_info()["config"], sort_keys=True)), champion=champ)
"""


@pytest.mark.slow
@pytest.mark.parametrize("stem, limit", [("lander", ("--max-generations", "100")),
                                         ("jump", ("--max-evaluations", "3000")),
                                         ("wheels", ("--max-evaluations", "2000"))])
def test_s1_real_training_matches_the_frozen_release(stem: str, limit: tuple[str, str], tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOLAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EVOLAB_STATE_DIR", str(tmp_path / "state"))
    script = write_script(tmp_path, "train.py", TRAIN)
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    old = frozen_run(RELEASE, script, tmp_path / "old", stem, *limit, timeout=3600)
    new = live_run(script, tmp_path / "new", stem, *limit, timeout=3600)
    same_arrays(old, new)
    assert len(json.loads(str(new["rows"]))) > 1

"""v2.1 L묶음: GA·담금질 구조 뒤집기 (v2 설계 4.12·D41·3.3·6절, 테스트 T66~T69).

- GenomeLayout.on_genes: exists가 정확히 "=p.<id>"이고 그 매개변수가 진화하면(범위 가운데 0.5) 그 몸 유전자 번호.
  서명·to_dict·같음 비교에 들어가지 않는다.
- flip_structure: 행 × 있음 유전자 순서로 확률 p면 −0.5·sign, 확률 0·빈 목록이면 난수를 부르지 않는다(T66).
- GA는 자식을 다 만든 뒤(엘리트 제외), 담금질은 이웃을 만든 뒤 뒤집는다. p = 0이면 v2.0 코드(커밋 4d3e3e6)와 상태 피클
  바이트까지 같고(T67, 같은 머신에서 git archive로 v2.0 코드를 꺼내 두 번 돌려 비교), p = 0.05에서 끊고 이어 가도 같다.
- 상태 호환: p = 0이면 상태 설정에 structure_* 키가 없고, 키가 없는 상태는 0으로 읽는다. 새 실행은 있음 유전자가 있는
  환경에서만 0.05(T68). 명령줄 --structure-flip(T69).
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab import paths
from evolab.app import strings as S
from evolab.engine import seeds
from evolab.engine import trainer as trainer_mod
from evolab.engine.evaluate import EvalResult
from evolab.engine.trainer import (
    LIVE_CONFIG_KEYS,
    STRUCTURE_FLIP_NEW_RUN,
    TrainConfig,
    Trainer,
    TrainerChangeError,
    make_learner,
    structure_note,
)
from evolab.learners.base import generator_state, new_generator
from evolab.learners.ga import GAConfig, GeneticAlgorithm
from evolab.learners.sa import SAConfig, SimulatedAnnealing
from evolab.learners.structure import flip_structure
from evolab.policy.brain import BrainLayout
from evolab.policy.genome import BodyGene, GenomeLayout, layout_for
from evolab.runs.store import RunStore
from evolab.spec import structure as st
from evolab.spec.io import compile_env

ROOT = Path(__file__).resolve().parents[1]
V20_COMMIT = "4d3e3e6"
"""v2.0을 끝낸 커밋. T67은 이 커밋의 src/evolab을 git archive로 꺼내 v2.0 학습 방법을 그대로 돌린다."""
TIME_COLUMNS = set(trainer_mod.TIME_COLUMNS)


def _wheel(k: int, x: float, exists: str | None) -> dict[str, Any]:
    b: dict[str, Any] = {"id": f"wheel{k}", "pos": [x, 0.6], "shapes": [{"kind": "circle", "r": "=p.wheel_r"}],
                         "tags": ["wheel"]}
    if exists is not None:
        b["exists"] = exists
    return b


# 바퀴 셋이 모두 있음 진화(차체 = 뿌리). 몸 유전자: half, w1_on, wheel_r, w2_on, w3_on → 있음 유전자 1, 3, 4.
# wheel2의 수식은 앞뒤 공백이 있어도 "=p.<id>"로 센다.
CAR: dict[str, Any] = {
    "version": 3,
    "name": "structure learner car",
    "world": {"max_time": 1.0, "bounds": [-20.0, -10.0, 60.0, 30.0]},
    "params": [
        {"id": "half", "value": 1.0, "evolve": True, "min": 0.6, "max": 1.4},
        {"id": "w1_on", "value": 0.75, "evolve": True, "min": 0.0, "max": 1.0},
        {"id": "wheel_r", "value": 0.4, "evolve": True, "min": 0.25, "max": 0.6},
        {"id": "w2_on", "value": 0.25, "evolve": True, "min": 0.0, "max": 1.0},
        {"id": "w3_on", "value": 0.25, "evolve": True, "min": 0.0, "max": 1.0},
    ],
    "bodies": [
        {"id": "ground", "owner": "world", "type": "static", "tags": ["ground"],
         "shapes": [{"kind": "box", "w": 80.0, "h": 1.0, "offset": [20.0, -0.5]}]},
        {"id": "chassis", "pos": [0.0, 1.3], "shapes": [{"kind": "box", "w": "=2 * p.half", "h": 0.3}]},
        _wheel(1, -0.8, "=p.w1_on"), _wheel(2, 0.0, "= p.w2_on "), _wheel(3, 0.8, "=p.w3_on"),
    ],
    "joints": [{"id": f"j{k}", "kind": "pivot", "a": "chassis", "b": f"wheel{k}", "anchor_a": [x, -0.7]}
               for k, x in ((1, -0.8), (2, 0.0), (3, 0.8))],
    "actuators": [{"kind": "motor", "id": f"m{k}", "joint": f"j{k}", "max_torque": 40.0, "max_speed": 10.0}
                  for k in (1, 2, 3)],
    "sensors": [{"kind": "body_state", "id": "cs", "body": "chassis", "fields": ["vx", "angle"]},
                {"kind": "contact", "id": "w2_down", "body": "wheel2", "with": ["tag.ground"]}],
    "termination": [],
    "scoring": {"blocks": [{"kind": "progress", "id": "travel", "body": "chassis"}]},
    "brain": {"hidden": [3]},
}
CAR_ON = (1, 3, 4)
CAR_TEXT = json.dumps(CAR, ensure_ascii=False)


def _data(**changes: Any) -> dict[str, Any]:
    d = copy.deepcopy(CAR)
    d.update(copy.deepcopy(changes))
    return d


def _spec(data: dict[str, Any]) -> Any:
    return compile_env(json.dumps(data, ensure_ascii=False))


def _mixed() -> dict[str, Any]:
    """있음 유전자 규칙의 경우들: 정확히 "=p.<id>"(진화, 범위 가운데 0.5)만 센다."""
    d = _data()
    d["params"] = [
        {"id": "half", "value": 1.0, "evolve": True, "min": 0.6, "max": 1.4},          # 0
        {"id": "a_on", "value": 0.75, "evolve": True, "min": 0.0, "max": 1.0},          # 1 센다
        {"id": "wheel_r", "value": 0.4, "evolve": True, "min": 0.25, "max": 0.6},      # 2
        {"id": "b_wide", "value": 1.5, "evolve": True, "min": 0.0, "max": 2.0},         # 3 가운데 1.0: 안 셈
        {"id": "c_count", "value": 1.0, "evolve": True, "min": 0.0, "max": 2.0},        # 4 다른 수식: 안 셈
        {"id": "d_on", "value": 1.25, "evolve": True, "min": -1.0, "max": 2.0},         # 5 가운데 0.5: 센다
        {"id": "e_fixed", "value": 0.75},                                               # 진화 안 함: 안 셈
    ]
    d["bodies"] = d["bodies"][:2] + [
        _wheel(1, -0.8, "=p.a_on"), _wheel(2, -0.3, "=p.b_wide"), _wheel(3, 0.2, "=p.c_count > 1.5"),
        _wheel(4, 0.5, "=p.d_on"), _wheel(5, 0.8, "=p.e_fixed"), _wheel(6, 1.0, "=(p.a_on)"),
    ]
    d["joints"] = [{"id": f"j{k}", "kind": "pivot", "a": "chassis", "b": f"wheel{k}", "anchor_a": [0.0, -0.7]}
                   for k in range(1, 7)]
    d["actuators"] = [{"kind": "motor", "id": "m1", "joint": "j1", "max_torque": 40.0, "max_speed": 10.0}]
    d["sensors"] = [{"kind": "body_state", "id": "cs", "body": "chassis", "fields": ["vx"]}]
    return d


# ---------------------------------------------------------------------------------------------- on_genes


def test_on_genes_follow_the_exact_formula_rule() -> None:
    layout = layout_for(_spec(_mixed()))
    assert [g.id for g in layout.body] == ["half", "a_on", "wheel_r", "b_wide", "c_count", "d_on"]
    assert layout.on_genes == (1, 5)  # a_on(두 물체가 같은 유전자: 한 번), d_on(범위 -1..2)
    car = layout_for(_spec(CAR))
    assert car.on_genes == CAR_ON


def test_on_genes_are_not_part_of_the_signature_or_saved_layout() -> None:
    """수식 고치기는 서명을 바꾸지 않는다: 있음 칸을 지운 같은 환경과 서명·to_dict가 같다. from_dict에는 on_genes가 없다."""
    with_exists = layout_for(_spec(CAR))
    plain = _data()
    for b in plain["bodies"]:
        b.pop("exists", None)
    without = layout_for(_spec(plain))
    assert without.on_genes == () and with_exists.on_genes == CAR_ON
    assert with_exists.signature() == without.signature()
    assert with_exists.to_dict() == without.to_dict()
    assert "on_genes" not in json.dumps(with_exists.to_dict())
    loaded = GenomeLayout.from_dict(with_exists.to_dict())
    assert loaded.on_genes == () and loaded == with_exists  # 같음 비교에도 넣지 않는다
    assert np.array_equal(loaded.initial(), with_exists.initial())


def test_on_genes_of_the_builtin_examples() -> None:
    """v1·v2.0 예제에는 있음 유전자가 없고, 바퀴 고르는 언덕차(A'의 wheels.evoenv)는 바퀴 자리 넷."""
    for path in paths.builtin_envs():
        layout = layout_for(compile_env(path.read_text(encoding="utf-8")))
        if path.name == "wheels.evoenv":
            assert [layout.body[k].id for k in layout.on_genes] == ["w1_on", "w2_on", "w3_on", "w4_on"]
        else:
            assert layout.on_genes == (), path.name


def test_on_genes_validation() -> None:
    body = (BodyGene("a", 0.0, 1.0, 0.5), BodyGene("b", 0.0, 1.0, 0.5))
    brain = BrainLayout(1, (), 1)
    assert GenomeLayout(body, brain, ((0.0, 1.0),), on_genes=[np.int64(1)]).on_genes == (1,)  # type: ignore[arg-type]
    for bad in ((2,), (-1,), (0, 0), (True,), (0.0,)):
        with pytest.raises(ValueError):
            GenomeLayout(body, brain, ((0.0, 1.0),), on_genes=bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------- T66


def _rng_bytes(rng: np.random.Generator) -> bytes:
    return pickle.dumps(generator_state(rng))


@pytest.mark.parametrize(("p", "rows", "genes"), [
    (0.0, [0, 1, 2], [1, 3]), (-0.5, [0, 1], [1]), (math.nan, [0, 1], [1]), (1.0, [], [1, 3]), (1.0, [0, 1], []),
])
def test_t66_flip_never_touches_the_rng_when_off(p: float, rows: list[int], genes: list[int]) -> None:
    rng = new_generator(5)
    before = _rng_bytes(rng)
    X = np.linspace(-0.9, 0.9, 15).reshape(3, 5)
    Y = X.copy()
    assert flip_structure(Y, rows, genes, p, rng) == 0
    assert _rng_bytes(rng) == before
    assert Y.tobytes() == X.tobytes()


def test_t66_flip_with_p_one_flips_every_on_gene() -> None:
    X = np.array([[0.3, -0.7, 0.0, 0.9], [0.1, 0.2, -0.0, -0.4], [0.5, 0.5, 0.5, 0.5]])
    Y = X.copy()
    assert flip_structure(Y, [0, 1], [1, 2], 1.0, new_generator(1)) == 4
    assert Y[0].tolist() == [0.3, 0.5, 0.5, 0.9]  # −0.7 → +0.5, 0 → +0.5
    assert Y[1].tolist() == [0.1, -0.5, 0.5, -0.4]  # 0.2 → −0.5, −0 → +0.5
    assert Y[2].tobytes() == X[2].tobytes()  # 고르지 않은 행
    assert np.all(np.abs(Y[:2, [1, 2]]) == 0.5)
    assert np.all(np.sign(Y[:2, [1, 2]]) != np.sign(np.where(X[:2, [1, 2]] > 0, 1.0, -1.0)))


def test_t66_flip_order_is_rows_times_genes() -> None:
    """난수는 행 × 유전자 순서로 rng.random()을 하나씩 부른 것과 같다(같은 상태로 끝남)."""
    rows, genes, p = [5, 1, 3, 2], [4, 0, 2], 0.4
    X = np.random.default_rng(3).uniform(-1, 1, (6, 5))
    a, b = new_generator(9), new_generator(9)
    Y = X.copy()
    count = flip_structure(Y, rows, genes, p, a)
    Z = X.copy()
    expect = 0
    for r in rows:
        for g in genes:
            if b.random() < p:
                Z[r, g] = -0.5 if Z[r, g] > 0 else 0.5
                expect += 1
    assert count == expect > 0
    assert Y.tobytes() == Z.tobytes()
    assert _rng_bytes(a) == _rng_bytes(b)


def test_ga_flips_children_after_they_are_made_and_never_elites() -> None:
    """같은 시드의 p = 0과 p = 1: 첫 tell 뒤 엘리트는 같고, 자식은 있음 유전자만 부호가 뒤집혔다(자식을 만든 난수는 같음)."""
    x0 = np.array([0.5, -0.5, 0.2, -0.5, 0.0, 0.0, 0.0])
    kw = {"popsize": 8, "elites": 2, "seed": 4, "bounded_dims": 4}
    off = GeneticAlgorithm(x0, GAConfig(**kw))
    on = GeneticAlgorithm(x0, GAConfig(**kw, structure_flip=1.0, structure_genes=(1, 3)))
    assert off.ask().tobytes() == on.ask().tobytes()  # 처음 개체군은 뒤집지 않는다(tell에서만)
    X = off.ask()
    f = -np.sum((X - 0.3) ** 2, axis=1)
    off.tell(X, f)
    on.tell(on.ask(), f)
    A, B = off.ask(), on.ask()
    assert A[:2].tobytes() == B[:2].tobytes()
    keep = [0, 2, 4, 5, 6]
    assert A[2:, keep].tobytes() == B[2:, keep].tobytes()
    assert np.array_equal(B[2:, [1, 3]], np.where(A[2:, [1, 3]] > 0, -0.5, 0.5))


def test_sa_flips_neighbours_after_they_are_made_and_never_the_current_solution() -> None:
    x0 = np.array([0.5, -0.5, 0.2, -0.5, 0.0, 0.0])
    kw = {"neighbors": 5, "seed": 4, "bounded_dims": 4}
    off = SimulatedAnnealing(x0, SAConfig(**kw))
    on = SimulatedAnnealing(x0, SAConfig(**kw, structure_flip=1.0, structure_genes=(1, 3)))
    A, B = off.ask(), on.ask()
    assert A[0].tobytes() == B[0].tobytes() == x0.tobytes()
    keep = [0, 2, 4, 5]
    assert A[1:, keep].tobytes() == B[1:, keep].tobytes()
    assert np.array_equal(B[1:, [1, 3]], np.where(A[1:, [1, 3]] > 0, -0.5, 0.5))


@pytest.mark.parametrize("cls", [GAConfig, SAConfig])
def test_structure_config_validation(cls: type) -> None:
    ok = cls(bounded_dims=4, structure_flip=0.05, structure_genes=[3, 1])
    assert ok.structure_genes == (3, 1) and isinstance(ok.structure_genes, tuple)
    assert cls(bounded_dims=4, structure_flip=1).structure_flip == 1
    bad: list[dict[str, Any]] = [
        {"structure_flip": -0.01}, {"structure_flip": 1.5}, {"structure_flip": math.nan}, {"structure_flip": math.inf},
        {"structure_flip": True}, {"structure_flip": "0.1"}, {"structure_flip": None},
        {"structure_genes": (4,)}, {"structure_genes": (-1,)}, {"structure_genes": (1, 1)}, {"structure_genes": (True,)},
        {"structure_genes": (1.0,)}, {"structure_genes": "12"}, {"structure_genes": 1},
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            cls(bounded_dims=4, **kw)


# ---------------------------------------------------------------------------------------------- T67

# v2.0 코드와 지금 코드를 같은 방식으로 돌리는 스크립트(새 프로세스). argv: [src 폴더, 출력 피클]. src 폴더의 evolab을
# 쓰는지 확인한다. 지금 코드에는 p = 0과 있음 유전자 번호를 넣어(v2.0에는 그 칸이 없다) "꺼진 뒤집기 = v2.0"을 본다.
_TRACE_SCRIPT = r'''
import dataclasses, pickle, sys
from pathlib import Path
src = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(src))
import numpy as np
import evolab
assert Path(evolab.__file__).resolve().is_relative_to(src), evolab.__file__
from evolab.learners.ga import GAConfig, GeneticAlgorithm
from evolab.learners.sa import SAConfig, SimulatedAnnealing

def fitness(X):
    return -np.round(np.sum((X - 0.3) ** 2, axis=1), 1) + 2.0 * (X[:, [1, 3]] > 0).sum(axis=1)

x0 = np.array([0.5, -0.5, 0.2, -0.5, 0.1, -0.2, 0.0, 0.3, 0.0, 0.0])
out = {"evolab": str(Path(evolab.__file__).resolve())}
for name, cls, ccls, kw in (
    ("ga", GeneticAlgorithm, GAConfig, {"popsize": 10, "elites": 2, "seed": 11, "bounded_dims": 4, "sigma_decay": 0.99}),
    ("sa", SimulatedAnnealing, SAConfig, {"neighbors": 6, "seed": 12, "bounded_dims": 4}),
):
    if "structure_flip" in {f.name for f in dataclasses.fields(ccls)}:
        kw = {**kw, "structure_flip": 0.0, "structure_genes": (1, 3)}
    learner = cls(x0, ccls(**kw))
    trace = [pickle.dumps(learner.state_dict(), protocol=4)]
    for _ in range(6):
        X = learner.ask()
        learner.tell(X, fitness(X))
        trace += [X.tobytes(), pickle.dumps(learner.state_dict(), protocol=4)]
    out[name] = trace
Path(sys.argv[2]).write_bytes(pickle.dumps(out, protocol=4))
'''


@pytest.fixture(scope="module")
def v20_src(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """v2.0 코드(커밋 4d3e3e6)의 src. git이나 그 커밋이 없으면(얕은 복제) 건너뛴다. CI(환경 변수 CI)에서는 건너뛰지 않고
    실패한다: 윈도우 CI는 전체 기록을 받으므로(.github/workflows/windows.yml fetch-depth: 0) 그 커밋이 없으면 설정이
    잘못된 것이고, 조용히 건너뛰면 v2.0과 비교하는 유일한 테스트가 빠진다."""
    def unavailable(reason: str) -> None:
        if os.environ.get("CI"):
            pytest.fail(f"T67 needs the v2.0 commit {V20_COMMIT} on CI (full clone): {reason}")
        pytest.skip(reason)

    if shutil.which("git") is None:
        unavailable("git is not available")
    try:
        done = subprocess.run(["git", "-C", str(ROOT), "archive", V20_COMMIT, "src/evolab"], capture_output=True,
                              check=True, timeout=120)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        unavailable(f"v2.0 commit {V20_COMMIT} is not available here: {exc}")
    out = tmp_path_factory.mktemp("v20")
    with tarfile.open(fileobj=io.BytesIO(done.stdout)) as tar:
        tar.extractall(out, filter="data")
    return out / "src"


def _trace(src: Path, tmp: Path, tag: str) -> dict[str, Any]:
    script, out = tmp / "trace.py", tmp / f"trace_{tag}.pkl"
    script.write_text(_TRACE_SCRIPT, encoding="utf-8")
    subprocess.run([sys.executable, str(script), str(src), str(out)], check=True, timeout=180,
                   env={**os.environ, "PYTHONHASHSEED": "1234"})
    return pickle.loads(out.read_bytes())


def test_t67_ga_and_sa_with_p0_are_bit_identical_to_v20(v20_src: Path, tmp_path: Path) -> None:
    """같은 머신에서 v2.0 코드와 지금 코드를 돌려 6세대의 후보와 상태 피클 바이트를 비교한다(정답 해시를 박지 않음)."""
    old = _trace(v20_src, tmp_path, "v20")
    new = _trace(ROOT / "src", tmp_path, "now")
    assert Path(old["evolab"]).is_relative_to(v20_src.resolve())
    assert Path(new["evolab"]).is_relative_to((ROOT / "src").resolve())
    for name in ("ga", "sa"):
        assert len(old[name]) == len(new[name]) == 13
        for k, (a, b) in enumerate(zip(old[name], new[name], strict=True)):
            assert a == b, f"{name}: item {k} differs"


@pytest.mark.parametrize("method", ["ga", "sa"])
def test_t67_p0_with_structure_genes_is_the_plain_learner(method: str) -> None:
    """어디서나 도는 대조: p = 0이면 있음 유전자 번호를 줘도 칸이 없는 학습 방법과 후보·상태 바이트가 같다."""
    x0 = np.array([0.5, -0.5, 0.2, -0.5, 0.1, -0.2, 0.0])
    cls, ccls = (GeneticAlgorithm, GAConfig) if method == "ga" else (SimulatedAnnealing, SAConfig)
    base = {"popsize": 8, "seed": 2, "bounded_dims": 4} if method == "ga" else {"neighbors": 5, "seed": 2,
                                                                               "bounded_dims": 4}
    a = cls(x0, ccls(**base))
    b = cls(x0, ccls(**base, structure_flip=0.0, structure_genes=(1, 3)))
    for _ in range(6):
        X, Y = a.ask(), b.ask()
        assert X.tobytes() == Y.tobytes()
        f = -np.sum(X**2, axis=1)
        a.tell(X, f)
        b.tell(Y, f)
        assert pickle.dumps(a.state_dict()) == pickle.dumps(b.state_dict())


class ScoreEvaluator:
    """물리를 돌리지 않는 결정적 평가기: 점수 = 유전자의 함수(있는 바퀴가 많을수록 조금 좋음) + 시험 시드 몫."""

    def evaluate(self, env_version: str, env_text: str, layout: Any, genomes: np.ndarray,
                 pairs: list[tuple[int, Any]], record: list[bool] | None = None) -> EvalResult:
        n = len(pairs)
        on = np.asarray(layout.on_genes, dtype=np.intp)
        scores = np.empty(n)
        for j, (i, trial) in enumerate(pairs):
            g = genomes[i]
            scores[j] = -float(np.sum((g - 0.2) ** 2)) + float(np.sum(g[on] > 0)) + (trial.seed % 97) / 97.0
        return EvalResult(scores=scores, steps=np.ones(n, dtype=np.int64), end_events=["timeout"] * n,
                          outcomes=["neutral"] * n, blocks=np.zeros((n, 1)), action_abs=np.zeros((n, 3)),
                          weighted=np.zeros((n, 1)))


def _config(method: str, gens: int, flip: float = 0.05, **over: Any) -> TrainConfig:
    per = {"ga": {"popsize": 8, "ga_structure_flip": flip}, "sa": {"sa_neighbors": 5, "sa_structure_flip": flip}}
    base = {"trials": 2, "validate_every": 24, "checkpoint_every": 3, "run_seed": 5, "max_generations": gens}
    return TrainConfig(method=method, **{**base, **per[method], **over})


def _store(tmp: Path, name: str) -> RunStore:
    s = RunStore.create(tmp, name, {"env_name": CAR["name"]})
    s.add_env_version(CAR_TEXT, 0)
    return s


def _strip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: str(v) for k, v in r.items() if k not in TIME_COLUMNS} for r in rows]


@pytest.mark.parametrize("method", ["ga", "sa"])
def test_t67_resume_with_p005_is_bit_identical(method: str, tmp_path: Path) -> None:
    """p = 0.05: 6세대 한 번에 = 3세대 + 체크포인트에서 이어 3세대(통계, 학습 방법 상태 피클 바이트). 구조가 실제로 바뀐다."""
    straight_store = _store(tmp_path, "straight")
    straight = Trainer(CAR_TEXT, _config(method, 6), ScoreEvaluator(), straight_store)
    seen: list[np.ndarray] = []
    straight.run(lambda row: seen.append(straight.last_generation.population.copy()))  # type: ignore[union-attr]
    assert straight.learner.config.structure_flip == 0.05  # type: ignore[attr-defined]
    assert straight.learner.config.structure_genes == CAR_ON  # type: ignore[attr-defined]
    signs = {tuple(row) for X in seen for row in (X[:, list(CAR_ON)] > 0)}
    assert len(signs) > 1, "structure never changed"

    cut = _store(tmp_path, "cut")
    Trainer(CAR_TEXT, _config(method, 3), ScoreEvaluator(), cut).run()
    resumed = Trainer.resume(cut, ScoreEvaluator(), None, max_generations=6)
    resumed.run()
    assert _strip(cut.read_stats()) == _strip(straight_store.read_stats())
    assert pickle.dumps(resumed.learner.state_dict()) == pickle.dumps(straight.learner.state_dict())
    assert resumed.state.champion.genome is not None and straight.state.champion.genome is not None
    assert resumed.state.champion.genome.tobytes() == straight.state.champion.genome.tobytes()


@pytest.mark.parametrize("method", ["ga", "sa"])
def test_t67_learner_state_round_trip_keeps_flipping_the_same_way(method: str) -> None:
    """상태(뒤집기 칸 포함)를 피클해 다른 설정으로 만든 학습 방법에 불러도, 끊지 않은 것과 같은 후보를 낸다."""
    x0 = np.array([0.5, -0.5, 0.2, -0.5, 0.0, 0.1])
    cls, ccls = (GeneticAlgorithm, GAConfig) if method == "ga" else (SimulatedAnnealing, SAConfig)
    kw = {"popsize": 10, "seed": 8} if method == "ga" else {"neighbors": 6, "seed": 8}
    config = ccls(**kw, bounded_dims=4, structure_flip=0.2, structure_genes=(1, 3))

    def go(learner: Any, n: int) -> list[bytes]:
        out = []
        for _ in range(n):
            X = learner.ask()
            out.append(X.tobytes())
            learner.tell(X, -np.sum((X - 0.1) ** 2, axis=1))
        return out

    straight = cls(x0, config)
    asked = go(straight, 8)
    first = cls(x0, config)
    go(first, 4)
    state = pickle.loads(pickle.dumps(first.state_dict()))
    assert state["config"]["structure_flip"] == 0.2 and tuple(state["config"]["structure_genes"]) == (1, 3)
    second = cls(x0, ccls(**kw))  # 뒤집기 없이 만든 학습 방법: 상태의 설정이 우선한다
    second.load_state_dict(state)
    assert second.config == config
    assert go(second, 4) == asked[4:]
    assert pickle.dumps(second.state_dict()) == pickle.dumps(straight.state_dict())


# ---------------------------------------------------------------------------------------------- T68

V20_GA_KEYS = ["popsize", "elites", "tournament", "crossover", "mutation_rate", "mutation_sigma", "sigma_decay", "seed",
               "bounded_dims"]
V20_SA_KEYS = ["neighbors", "step", "initial_temperature", "cooling", "seed", "bounded_dims"]


@pytest.mark.parametrize("method", ["ga", "sa"])
def test_t68_state_has_structure_keys_only_when_flipping(method: str) -> None:
    cls, ccls, keys = ((GeneticAlgorithm, GAConfig, V20_GA_KEYS) if method == "ga"
                       else (SimulatedAnnealing, SAConfig, V20_SA_KEYS))
    x0 = np.zeros(6)
    for kw in ({}, {"structure_flip": 0.0, "structure_genes": (0, 2)}):
        assert list(cls(x0, ccls(bounded_dims=3, **kw)).state_dict()["config"]) == keys
    on = cls(x0, ccls(bounded_dims=3, structure_flip=0.05, structure_genes=(0, 2))).state_dict()["config"]
    assert list(on) == [*keys, "structure_flip", "structure_genes"]
    assert (on["structure_flip"], tuple(on["structure_genes"])) == (0.05, (0, 2))


@pytest.mark.parametrize("method", ["ga", "sa"])
def test_t68_state_without_structure_keys_loads_as_off(method: str) -> None:
    cls, ccls = (GeneticAlgorithm, GAConfig) if method == "ga" else (SimulatedAnnealing, SAConfig)
    x0 = np.zeros(6)
    state = cls(x0, ccls(bounded_dims=3, structure_flip=0.3, structure_genes=(1,))).state_dict()
    for key in ("structure_flip", "structure_genes"):
        del state["config"][key]
    target = cls(x0, ccls(bounded_dims=3, structure_flip=0.3, structure_genes=(1,)))
    target.load_state_dict(state)
    assert (target.config.structure_flip, target.config.structure_genes) == (0.0, ())
    broken = cls(x0, ccls(bounded_dims=3, structure_flip=0.3, structure_genes=(1,))).state_dict()
    broken["config"]["structure_genes"] = [5]  # 몸 유전자 밖
    with pytest.raises(ValueError):
        cls(x0, ccls(bounded_dims=3)).load_state_dict(broken)


def test_t68_make_learner_passes_structure_fields_only_when_they_act() -> None:
    car = layout_for(_spec(CAR))
    lander = layout_for(compile_env((paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8")))
    plain_ga = GAConfig(popsize=8, seed=seeds.learner_seed(5), bounded_dims=car.n_body)
    plain_sa = SAConfig(neighbors=5, seed=seeds.learner_seed(5), bounded_dims=car.n_body)
    assert make_learner(_config("ga", 1, 0.0), car).config == plain_ga  # type: ignore[attr-defined]
    assert make_learner(_config("sa", 1, 0.0), car).config == plain_sa  # type: ignore[attr-defined]
    ga = make_learner(_config("ga", 1, 0.05), car)
    assert asdict(ga.config) == {**asdict(plain_ga), "structure_flip": 0.05, "structure_genes": CAR_ON}  # type: ignore
    sa = make_learner(_config("sa", 1, 0.05), car)
    assert asdict(sa.config) == {**asdict(plain_sa), "structure_flip": 0.05, "structure_genes": CAR_ON}  # type: ignore
    # 있음 유전자가 없는 환경은 확률을 줘도 v2.0과 같은 설정(상태 키도 없음)
    ga_lander = make_learner(_config("ga", 1, 0.5), lander)
    assert ga_lander.config == GAConfig(popsize=8, seed=seeds.learner_seed(5), bounded_dims=0)  # type: ignore
    assert "structure_flip" not in ga_lander.state_dict()["config"]
    # CMA-ES에는 뒤집기가 없다
    cma = make_learner(TrainConfig(method="cmaes", popsize=6, ga_structure_flip=0.5, sa_structure_flip=0.5), car)
    assert "structure" not in repr(cma.config)  # type: ignore[attr-defined]


def test_t68_for_new_run_puts_the_flip_only_for_environments_with_on_genes() -> None:
    car = _spec(CAR)
    c = TrainConfig.for_new_run(car)
    assert STRUCTURE_FLIP_NEW_RUN == 0.05
    assert (c.ga_structure_flip, c.sa_structure_flip) == (0.05, 0.05)
    assert c.restart_patience == TrainConfig.for_new_run().restart_patience  # 다른 권장값은 그대로
    assert TrainConfig.for_new_run(car, ga_structure_flip=0.0, method="ga").ga_structure_flip == 0.0  # 값이 우선
    for path in paths.builtin_envs():
        spec = compile_env(path.read_text(encoding="utf-8"))
        # (6.2.0, 설계 4.5·D31) editor.train이 없는 내장 환경은 기본값과 같다(종목 프리셋은 권장 설정을 가질 수 있다)
        if path.name != "wheels.evoenv" and "train" not in (spec.file.editor or {}):
            assert TrainConfig.for_new_run(spec) == TrainConfig.for_new_run()
    counted = _data()
    for b in counted["bodies"][2:]:
        b["exists"] = "=" + b["exists"].strip("= ") + " > 0.5"  # 수식이 "=p.<id>"가 아니면 있음 유전자가 아님
    assert TrainConfig.for_new_run(_spec(counted)) == TrainConfig.for_new_run()
    # 데이터클래스 기본값은 끔: 예전 run.json·체크포인트(키 없음)는 0으로 읽는다
    assert (TrainConfig().ga_structure_flip, TrainConfig().sa_structure_flip) == (0.0, 0.0)
    assert TrainConfig.from_dict({"method": "ga"}).ga_structure_flip == 0.0
    assert "ga_structure_flip" not in trainer_mod.NEW_RUN_DEFAULTS


def test_t68_train_config_checks_and_live_keys() -> None:
    TrainConfig(method="ga", ga_structure_flip=1.0, sa_structure_flip=0.0).validate()
    for kw in ({"ga_structure_flip": -0.1}, {"sa_structure_flip": 1.1}, {"ga_structure_flip": math.nan},
               {"sa_structure_flip": True}, {"ga_structure_flip": None}):
        with pytest.raises(TrainerChangeError):
            TrainConfig(**kw).validate()  # type: ignore[arg-type]
    assert not {"ga_structure_flip", "sa_structure_flip"} & set(LIVE_CONFIG_KEYS)
    t = Trainer(CAR_TEXT, _config("ga", 1), ScoreEvaluator())
    with pytest.raises(TrainerChangeError):
        t.apply_config({"ga_structure_flip": 0.2})  # 학습 방법의 일부: 새 실행으로


def test_t68_old_ga_run_on_a_structure_environment_gets_the_notice() -> None:
    car = layout_for(_spec(CAR))
    lander = layout_for(compile_env((paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8")))
    assert structure_note(TrainConfig(method="ga"), car) == S.LAB_GA_NO_STRUCTURE  # v2.0 실행: 키가 없어 0
    assert structure_note(TrainConfig(method="sa", ga_structure_flip=0.05), car) == S.LAB_GA_NO_STRUCTURE
    assert structure_note(TrainConfig(method="sa", sa_structure_flip=0.05), car) is None
    assert structure_note(TrainConfig(method="cmaes"), car) is None
    assert structure_note(TrainConfig(method="ga"), lander) is None


def test_t68_v20_style_checkpoint_resumes_without_structure_keys(tmp_path: Path) -> None:
    """v2.0 GA 실행 모양(설정·상태에 구조 키 없음)을 구조 환경에서 이어 가면 끈 채로 그대로 이어진다(v2.0과 같음)."""
    straight = Trainer(CAR_TEXT, _config("ga", 4, 0.0), ScoreEvaluator(), _store(tmp_path, "straight"))
    straight.run()
    cut = _store(tmp_path, "cut")
    Trainer(CAR_TEXT, _config("ga", 2, 0.0), ScoreEvaluator(), cut).run()
    gen, ckpt = cut.load_checkpoint(None)
    assert "structure_flip" not in ckpt["learner"]["config"]
    for key in ("ga_structure_flip", "sa_structure_flip"):  # v2.0 run.json·체크포인트에는 이 칸이 없다
        ckpt["config"].pop(key)
    cut.save_checkpoint(gen, ckpt)
    resumed = Trainer.resume(cut, ScoreEvaluator(), None, max_generations=4)
    assert resumed.config.ga_structure_flip == 0.0
    assert structure_note(resumed.config, resumed.layout) == S.LAB_GA_NO_STRUCTURE
    resumed.run()
    assert pickle.dumps(resumed.learner.state_dict()) == pickle.dumps(straight.learner.state_dict())


def test_structure_note_and_flip_fields_ignore_cma_settings() -> None:
    assert {f.name for f in fields(TrainConfig)} >= {"ga_structure_flip", "sa_structure_flip"}
    assert trainer_mod.structure_flip_of(TrainConfig(method="cmaes", ga_structure_flip=0.3)) == 0.0
    assert trainer_mod.structure_flip_of(TrainConfig(method="ga", ga_structure_flip=0.3)) == 0.3
    assert trainer_mod.structure_flip_of(TrainConfig(method="sa", sa_structure_flip=0.2)) == 0.2


def test_present_matches_the_on_gene_signs_for_the_flip_values() -> None:
    """뒤집은 값 ±0.5는 범위 0..1 있음 매개변수에서 0.25·0.75(문턱 0.5에서 멀다): 부호가 곧 구조다."""
    spec = _spec(CAR)
    layout = layout_for(spec)
    X = np.tile(layout.initial(), (64, 1))
    flip_structure(X, range(64), layout.on_genes, 0.5, new_generator(3))
    names = {1: "wheel1", 3: "wheel2", 4: "wheel3"}
    for x in X:
        params = st.param_values(spec.file, layout.body_values(x))
        here = st.present(spec, params)
        assert {names[k] for k in layout.on_genes if x[k] > 0} == here - {"chassis"}


# ---------------------------------------------------------------------------------------------- T69


def test_t69_cli_parser_takes_structure_flip(capsys: pytest.CaptureFixture[str]) -> None:
    from evolab.cli import _build_parser

    parser = _build_parser()
    assert parser.parse_args(["train", "x.evoenv", "--structure-flip", "0.2"]).structure_flip == 0.2
    assert parser.parse_args(["train", "x.evoenv"]).structure_flip is None
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["train", "x.evoenv", "--structure-flip", "abc"])
    assert exc.value.code == 2
    assert S.RVM_CLI_NOT_NUMBER.format(value="abc") in capsys.readouterr().err
    sub = next(a for a in parser._actions if a.dest == "command").choices["train"]  # type: ignore[union-attr]
    assert S.CLI_HELP_STRUCTURE_FLIP in " ".join(sub.format_help().split())


def _run_cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    from evolab.cli import main

    code = main(argv)
    out = capsys.readouterr()
    return code, out.out + out.err


def test_t69_cli_train_uses_structure_flip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env = tmp_path / "car.evoenv"
    env.write_text(CAR_TEXT, encoding="utf-8")
    base = ["train", str(env), "--method", "ga", "--popsize", "4", "--trials", "1", "--max-generations", "1",
            "--workers", "0", "--no-observe"]

    def run(extra: list[str], name: str) -> dict[str, Any]:
        out = tmp_path / name
        code, text = _run_cli([*base, *extra, "--out", str(out)], capsys)
        assert code == 0, text
        (run_dir,) = [p for p in out.iterdir() if p.is_dir()]
        return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["config"]

    given = run(["--structure-flip", "0.2"], "given")
    assert (given["ga_structure_flip"], given["sa_structure_flip"]) == (0.2, 0.2)
    default = run([], "default")
    assert (default["ga_structure_flip"], default["sa_structure_flip"]) == (STRUCTURE_FLIP_NEW_RUN,) * 2
    code, text = _run_cli([*base, "--structure-flip", "1.5", "--out", str(tmp_path / "bad")], capsys)
    assert code == 2 and "ga_structure_flip" in text
    (run_dir,) = [p for p in (tmp_path / "given").iterdir() if p.is_dir()]
    code, text = _run_cli(["train", "--resume", str(run_dir), "--structure-flip", "0.3", "--workers", "0"], capsys)
    assert code == 2 and S.TRAIN_RESUME_STRUCTURE_FIXED in text


# ---------------------------------------------------------------------------------------------- 학습 중 있음 칸 고치기
# 설계 3.5 v2.1 줄: 학습 중 있음 칸 수식을 고치면 이어진다(서명은 그대로). 고친 뒤 있음 유전자가 바뀌면 지금 GA·담금질도 새
# 목록을 뒤집는다(4.12, 검토 반영). 진화를 켜고 끄는 것은 몸 유전자가 바뀌어 거부한다.


def _car_text(**exists: str) -> str:
    d = _data()
    for b in d["bodies"]:
        if b["id"] in exists:
            b["exists"] = exists[b["id"]]
    return json.dumps(d, ensure_ascii=False)


def _store_with(tmp: Path, name: str, text: str) -> RunStore:
    s = RunStore.create(tmp, name, {"env_name": CAR["name"]})
    s.add_env_version(text, 0)
    return s


@pytest.mark.parametrize("method", ["ga", "sa"])
@pytest.mark.parametrize(("start", "edited", "before", "after"), [
    ({}, {"wheel3": "=p.w3_on > 0.5"}, CAR_ON, (1, 3)),                                   # 줄어듦
    ({"wheel3": "=p.w3_on > 0.5"}, {"wheel3": "=p.w3_on"}, (1, 3), CAR_ON),               # 늘어남
    ({"wheel1": "=p.w1_on > 0.5", "wheel2": "=p.w2_on > 0.5", "wheel3": "=p.w3_on > 0.5"},
     {"wheel1": "=p.w1_on > 0.5", "wheel2": "=p.w2_on > 0.5"}, (), (4,)),                  # 없음 -> 하나
], ids=["shrink", "grow", "from_none"])
def test_exists_edit_mid_run_moves_the_flip_genes_and_resumes_identically(
    method: str, start: dict[str, str], edited: dict[str, str], before: tuple[int, ...], after: tuple[int, ...],
    tmp_path: Path,
) -> None:
    start_text, edited_text = _car_text(**start), _car_text(**edited)
    store = _store_with(tmp_path, "straight", start_text)
    t = Trainer(start_text, _config(method, 2, flip=0.3, checkpoint_every=1), ScoreEvaluator(), store)
    t.run()
    assert t.layout.on_genes == before
    assert tuple(t.learner.config.structure_genes) == before  # type: ignore[attr-defined]
    assert (t.learner.config.structure_flip > 0) == bool(before)  # type: ignore[attr-defined]
    t.replace_env(edited_text)
    store.add_env_version(edited_text, 2)
    assert t.layout.on_genes == after
    assert t.learner.config.structure_genes == after  # type: ignore[attr-defined]
    assert t.learner.config.structure_flip == (0.3 if after else 0.0)  # type: ignore[attr-defined]
    t.apply_config({"max_generations": 6})
    pops: list[np.ndarray] = []
    t.run(lambda row: pops.append(t.last_generation.population.copy()))  # type: ignore[union-attr]
    new_genes = [k for k in after if k not in before]
    if new_genes:
        # 고친 뒤 만든 개체군(4세대부터)에서 새 있음 유전자가 실제로 뒤집혔다(뒤집은 값은 정확히 ±0.5)
        assert any(np.any(np.abs(X[1:, new_genes]) == 0.5) for X in pops[1:]), "new on-gene never flipped"
    state = t.learner.state_dict()
    assert ("structure_flip" in state["config"]) == bool(after)  # 상태 키는 뒤집기를 쓸 때만(T68 규칙 그대로)

    # 고친 뒤의 체크포인트(4세대)에서 이어 가도 끊지 않은 실행과 같다
    after_root = tmp_path / "after"
    shutil.copytree(store.root, after_root)
    after_store = RunStore.open(after_root)
    resumed = Trainer.resume(after_store, ScoreEvaluator(), 4, max_generations=6)
    assert resumed.layout.on_genes == after
    assert resumed.learner.config.structure_genes == after  # type: ignore[attr-defined]
    resumed.run()
    assert _strip(after_store.read_stats()) == _strip(store.read_stats())
    assert pickle.dumps(resumed.learner.state_dict()) == pickle.dumps(t.learner.state_dict())

    # 고치기 전의 체크포인트(2세대)에서 이어 같은 세대에 같은 수정을 해도 같다(재현 묶음이 다시 부르는 길)
    cut = _store_with(tmp_path, "cut", start_text)
    Trainer(start_text, _config(method, 2, flip=0.3, checkpoint_every=1), ScoreEvaluator(), cut).run()
    again = Trainer.resume(cut, ScoreEvaluator(), None, max_generations=6)
    again.replace_env(edited_text)
    cut.add_env_version(edited_text, 2)
    again.run()
    assert _strip(cut.read_stats()) == _strip(store.read_stats())
    assert pickle.dumps(again.learner.state_dict()) == pickle.dumps(t.learner.state_dict())


@pytest.mark.parametrize("method", ["ga", "sa"])
def test_exists_edit_with_flip_off_changes_nothing_in_the_learner(method: str) -> None:
    """구조 뒤집기를 끈 실행(v2.0 모양): 있음 칸을 고쳐도 학습 방법 설정·상태 키가 그대로다(T68)."""
    t = Trainer(CAR_TEXT, _config(method, 1, flip=0.0), ScoreEvaluator())
    t.run()
    before = pickle.dumps(t.learner.state_dict())
    t.replace_env(_car_text(wheel3="=p.w3_on > 0.5"))
    assert t.layout.on_genes == (1, 3)
    assert pickle.dumps(t.learner.state_dict()) == before
    assert "structure_flip" not in t.learner.state_dict()["config"]


@pytest.mark.parametrize("method", ["ga", "sa"])
def test_failed_env_change_rolls_the_flip_genes_back(method: str) -> None:
    """세션이 바뀐 것을 저장하지 못하면 restore_changeable로 되돌린다: 학습 방법의 있음 유전자도 바꾸기 전으로."""
    t = Trainer(CAR_TEXT, _config(method, 1, flip=0.3), ScoreEvaluator())
    t.run()
    snap = t.changeable_state()
    state = pickle.dumps(t.learner.state_dict())
    t.replace_env(_car_text(wheel3="=p.w3_on > 0.5"))
    assert t.learner.config.structure_genes == (1, 3)  # type: ignore[attr-defined]
    t.restore_changeable(snap)
    assert t.layout.on_genes == CAR_ON
    assert t.learner.config.structure_genes == CAR_ON  # type: ignore[attr-defined]
    assert pickle.dumps(t.learner.state_dict()) == state


@pytest.mark.parametrize("method", ["ga", "sa"])
def test_toggling_evolve_mid_run_is_refused(method: str) -> None:
    """진화를 켜고 끄면 몸 유전자(서명)가 바뀌므로 거부하고, 트레이너는 그대로다."""
    t = Trainer(CAR_TEXT, _config(method, 1, flip=0.3), ScoreEvaluator())
    t.run()
    toggled = _data()
    next(p for p in toggled["params"] if p["id"] == "w3_on")["evolve"] = False
    with pytest.raises(TrainerChangeError) as exc:
        t.replace_env(json.dumps(toggled, ensure_ascii=False))
    assert str(exc.value) == S.TRAIN_ENV_LAYOUT_CHANGED
    assert t.env_text == CAR_TEXT and t.layout.on_genes == CAR_ON
    assert t.learner.config.structure_genes == CAR_ON  # type: ignore[attr-defined]


@pytest.mark.parametrize("cls", [GAConfig, SAConfig])
def test_set_structure_validates_and_keeps_the_rest(cls: type) -> None:
    x0 = np.array([0.5, -0.5, 0.2, -0.5, 0.0])
    learner = (GeneticAlgorithm if cls is GAConfig else SimulatedAnnealing)(x0, cls(bounded_dims=4, seed=3))
    asked = learner.ask().tobytes()
    learner.set_structure(0.2, [3, 1])
    assert (learner.config.structure_flip, learner.config.structure_genes) == (0.2, (3, 1))
    assert learner.ask().tobytes() == asked  # 이미 만든 후보는 그대로
    for flip, genes in ((1.5, (1,)), (0.1, (4,)), (0.1, (1, 1))):
        with pytest.raises(ValueError):
            learner.set_structure(flip, genes)
    assert (learner.config.structure_flip, learner.config.structure_genes) == (0.2, (3, 1))


# ---------------------------------------------------------------------------------------------- 범위 가운데가 0.5가 아닌 있음 칸
# on_genes는 범위 가운데가 0.5인 매개변수만 센다(뒤집기 = 부호 바꾸기라 문턱이 유전자 0이어야 한다). 설계 계약보다 좁힌 이
# 점이 조용히 지나가지 않게 검사 안내(info)와 실험실·명령줄 안내가 알린다(검토 반영).


def _wide() -> dict[str, Any]:
    d = _data()
    next(p for p in d["params"] if p["id"] == "w2_on").update({"max": 2.0})
    return d


def test_off_centre_exists_params_are_reported_not_flipped() -> None:
    spec = _spec(_wide())
    assert layout_for(spec).on_genes == (1, 4)
    assert st.off_centre_exists(spec.file) == (("wheel2", "w2_on"),)
    assert st.off_centre_exists(_spec(CAR).file) == ()
    assert st.off_centre_exists(_spec(_mixed()).file) == (("wheel2", "b_wide"),)  # d_on(-1..2)은 가운데 0.5
    infos = [i for i in spec.issues if i.severity == "info"]
    assert [i.message for i in infos] == [S.SPEC_EXISTS_FLIP_RANGE.format(id="wheel2", param="w2_on")]
    assert infos[0].where == "bodies[3].exists"
    assert not [i for i in _spec(CAR).issues if i.severity == "info"]
    for path in paths.builtin_envs():  # 내장 예제에는 없다
        assert st.off_centre_exists(compile_env(path.read_text(encoding="utf-8")).file) == (), path.name


def test_structure_note_names_parts_the_flip_cannot_reach() -> None:
    spec = _spec(_wide())
    layout = layout_for(spec)
    range_note = S.LAB_STRUCTURE_RANGE.format(names="wheel2")
    assert structure_note(TrainConfig(method="ga", ga_structure_flip=0.05), layout, spec) == range_note
    assert structure_note(TrainConfig(method="ga", ga_structure_flip=0.05), layout) is None  # spec이 없으면 예전 그대로
    assert structure_note(TrainConfig(method="sa"), layout, spec) == S.LAB_GA_NO_STRUCTURE + "\n" + range_note
    assert structure_note(TrainConfig(method="cmaes"), layout, spec) is None
    assert trainer_mod.structure_range_note(TrainConfig(method="cmaes"), spec) is None
    # 범위가 모두 0..2이면 있음 유전자가 없어 새 실행 권장값도 0: 그래도 안내는 나온다(조용히 빠지지 않음)
    every = _data()
    for p in every["params"]:
        if p["id"].endswith("_on"):
            p["max"] = 2.0
    spec2 = _spec(every)
    assert layout_for(spec2).on_genes == ()
    fresh = TrainConfig.for_new_run(spec2, method="ga")
    assert fresh.ga_structure_flip == 0.0
    assert structure_note(fresh, layout_for(spec2), spec2) == S.LAB_STRUCTURE_RANGE.format(
        names="wheel1, wheel2, wheel3")


def test_cli_train_warns_about_off_centre_exists(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env = tmp_path / "wide.evoenv"
    env.write_text(json.dumps(_wide(), ensure_ascii=False), encoding="utf-8")
    code, text = _run_cli(["train", str(env), "--method", "ga", "--popsize", "4", "--trials", "1",
                           "--max-generations", "1", "--workers", "0", "--no-observe", "--out", str(tmp_path / "o")],
                          capsys)
    assert code == 0, text
    assert S.LAB_STRUCTURE_RANGE.format(names="wheel2") in text
    code, text = _run_cli(["train", str(env), "--method", "cmaes", "--popsize", "4", "--trials", "1",
                           "--max-generations", "1", "--workers", "0", "--no-observe", "--out", str(tmp_path / "c")],
                          capsys)
    assert code == 0, text
    assert S.LAB_STRUCTURE_RANGE.format(names="wheel2") not in text

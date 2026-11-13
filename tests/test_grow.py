"""v6.1 자라는 뇌: 학습 방법 GrowingCMAES와 학습기 연결 (v6 설계 4.15·4.16·2.2~2.5·3.3, D26~D35, 8.2 T31~T44, 8.3 S3).

- T31 꺼진 좌표는 정확히 +0.0(후보·추천해, 자람 전후). T32 함수 보존(자라기 직전·직후 평균의 시험 결과 바이트 같음).
- T33 정해진 간격 두 배(틀에서 멈춤). T34 결정성·numpy 전역 난수 없음. T35 저장·복원(자람 직전·직후·한참 뒤·ask와 tell 사이).
- T36 학습기 결정성(워커 0·1·3, 관전 기록·관찰 켬끔, 모든 세대 경계에서 이어 가기, 분기, 재현 묶음, 진짜 세션의 관전 방식).
- T37 통계 열, T38 설정 검사, T39 to_dict, T40 크기 고정 실행이 v6.0 사본과 같음, T41 자동 재시작·예선, T43 「탐색 폭」·대각 문턱,
  T44 옛 프로그램(v6.0 사본)이 자라는 실행의 이어 가기·재현을 거절하고 보기·판정은 됨. 자체 점검 「자라는 뇌」. S3(느림).

v6.0 사본(tests/data/v60_snapshot)은 load_v60()이 임시 꾸러미 _evolab_v60 아래로 불러온다: 서로를 부르는 import 줄만 사본 이름으로
바꾸고(SOURCES.txt), 나머지 모듈(평가·풀·저장소·문구)은 지금 것을 쓴다.

v6.1 고침 묶음(검토·검증에서 확인된 결함)에서 더하거나 고친 것:
- T34: 은닉층 2·3개 틀에서도 같은 시드 = 같은 후보(FRAMES). 켜진 좌표 300개 이상(pycma TPA가 전역 난수를 건드림)에서는 전역 난수
  시드를 바꿔도 같은 후보. T35: 은닉층 2·3개 틀, 자람 직전·직후마다, ask와 tell 사이(그 tell에서 자라는 경우 포함).
- T37: 자라는 실행의 diversity 열은 켜진 좌표만의 표준편차 평균(자랄 때 튀지 않음). T41: 자란 세대 끝에 자동 재시작·예선으로
  버려진 자람은 알리지 않고(Trainer.last_growth, 명령줄 한 줄, 세션 "growth"), 방금 자란 출전자가 통과하면 알린다.
- T40: v6.0이 만든 실행을 v6.1이, v6.1이 만든 실행을 v6.0이 이어 간다(네 짝이 모두 같음).
- T44: 재현 거절은 v6.0의 글 그대로("grow" 한 줄), v6.1 세션이 자라는 실행을 진짜로 이어 간다(자람을 지나서), 예전 체크포인트에서
  이어 하면 분기 폴더가 남는다.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import pickle
import shutil
import sys
import threading
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab import cli, paths, selftest
from evolab.app import strings as S
from evolab.engine import commands, seeds
from evolab.engine import trainer as trainer_mod
from evolab.engine.evaluate import EvalResult, evaluate
from evolab.engine.pool import WorkerPool
from evolab.engine.replay import replay
from evolab.engine.runsetup import create_run
from evolab.engine.trainer import (
    GROW_COLUMNS,
    GROW_KEYS,
    SWITCH_QUALIFY_NEXT,
    SWITCH_QUALIFY_PICK,
    SWITCH_RESTART_CHAMPION,
    SWITCH_RESTART_FRESH,
    TrainConfig,
    Trainer,
    TrainerChangeError,
    grow_problem,
    make_learner,
)
from evolab.learners.cmaes import CMAES, CMAESConfig
from evolab.learners.grow import (
    GROW_RNG_WORD,
    GROW_SEED_WORD,
    GROW_STATE_FORMAT,
    GrowConfig,
    GrowingCMAES,
    grow_rng_seed,
    grow_seed,
)
from evolab.policy.brain import unpack
from evolab.policy.frame import BrainFrame, FrameError
from evolab.policy.genome import layout_for
from evolab.runs.bundle import compare_stats, export_bundle, read_bundle, reproduce
from evolab.runs.catalog import branch_run
from evolab.runs.store import CHAMPIONS_DIR, CheckpointPolicy, RunStore
from evolab.spec.io import compile_env

ENVS = paths.builtin_env_dir()
TEXTS = {name: (ENVS / f"{name}.evoenv").read_text(encoding="utf-8")
         for name in ("lander", "track_car", "jump", "hill_car", "glider")}
LANDER_TEXT = TEXTS["lander"]
LANDER_SPEC = compile_env(LANDER_TEXT)
LAYOUT = layout_for(LANDER_SPEC)
TIME = set(trainer_mod.TIME_COLUMNS)
KEEP_ALL = CheckpointPolicy(keep_last=10**6)
SNAPSHOT = Path(__file__).parent / "data" / "v60_snapshot"
FAST = {"validation_seeds": 4, "validation_candidates": 1}
"""실제 환경 테스트용: 검증 세트를 줄여 빠르게(검증 주기·판단 규칙은 그대로)."""


def brain_text(hidden: list[int], activation: str = "tanh", base: str = LANDER_TEXT) -> str:
    """착륙선의 은닉층만 바꾼 환경(다층·은닉층 없음 시험용)."""
    data = json.loads(base)
    data["brain"]["hidden"] = hidden
    data["brain"]["activation"] = activation
    return json.dumps(data, ensure_ascii=False, indent=2)


def layout_of(text: str) -> Any:
    return layout_for(compile_env(text))


def small(**over: Any) -> TrainConfig:
    """작은 자라는 뇌 실행: 개체 6, 시험 2(세대당 12회), 검증 주기 24, 3세대마다 자람(2 → 4 → 8 → 16은 3·6·9세대 끝)."""
    return TrainConfig(**{"method": "grow", "popsize": 6, "trials": 2, "validate_every": 24, "checkpoint_every": 2,
                          "run_seed": 7, "grow_every": 3, **FAST, **over})


def learner(layout: Any = LAYOUT, *, popsize: int = 8, seed: int = 3, every: int = 3, start: int = 2,
            **cma: Any) -> GrowingCMAES:
    return GrowingCMAES(layout.initial(), layout, CMAESConfig(popsize=popsize, seed=seed, bounded_dims=layout.n_body,
                                                              **cma), GrowConfig(start=start, every=every))


def fake_scores(X: np.ndarray) -> np.ndarray:
    """정해진 가짜 점수(높을수록 좋음): 유전자마다 다른 목표점까지의 거리."""
    target = np.sin(np.arange(X.shape[1]) * 0.37) * 0.4
    return -np.sum((X - target) ** 2, axis=1)


def generation(lr: Any) -> np.ndarray:
    X = lr.ask()
    lr.tell(X, fake_scores(X))
    return X


def outside(lr: GrowingCMAES, counts: tuple[int, ...] | None = None) -> np.ndarray:
    mask = np.ones(lr.dim, dtype=bool)
    mask[lr.frame.active_coords(lr.counts if counts is None else counts)] = False
    return mask


def strip(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in r.items() if k not in TIME} for r in rows]


def npz(path: Path) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    with np.load(path, allow_pickle=False) as z:
        return {k: (str(z[k].dtype), tuple(z[k].shape), z[k].tobytes()) for k in z.files}


def champion_files(store: RunStore) -> dict[str, Any]:
    folder = store.root / CHAMPIONS_DIR
    return {p.name: npz(p) for p in sorted(folder.iterdir())}


def genomes_of(store: RunStore) -> dict[str, bytes]:
    """세대 대표·챔피언 유전자(관찰 키는 빼고): 관찰을 끈 실행과 켠 실행을 비교할 때."""
    out: dict[str, bytes] = {}
    for kind in ("representative", "champion"):
        for g in store.champion_generations(kind):
            out[f"{kind} {g}"] = store.load_champion(g, kind)["genome"].tobytes()
    return out


def canon(obj: Any, seen: dict[int, Any] | None = None) -> Any:
    """체크포인트를 비교할 수 있는 값으로(tests/test_pace.py와 같은 규칙): 배열은 (dtype, 모양, 바이트), 실수는 repr, 객체는
    (클래스, 속성). 벽시계 값(학습 상태의 "elapsed", pycma 피클 안의 "timer" 속성)은 뺀다. 피클 바이트는 풀어서 비교한다."""
    seen = {} if seen is None else seen
    if isinstance(obj, np.ndarray):
        return ("nd", obj.dtype.str, obj.shape, obj.tobytes())
    if isinstance(obj, float):
        return ("f", repr(obj))
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, bytes):
        if obj[:1] == b"\x80":
            try:
                return ("pickle", canon(pickle.loads(obj), seen))
            except Exception:  # noqa: BLE001 - 피클이 아니면 바이트 그대로
                pass
        return obj
    if id(obj) in seen:
        return "<cycle>"
    seen[id(obj)] = obj  # 비교가 끝날 때까지 살려 둔다: 풀어 낸 피클 같은 임시 값이 사라진 뒤 그 id를 새 값이 다시 받으면
    # 처음 보는 값이 「<cycle>」로 잘못 보인다(v6.1 통합: 부하가 있을 때 T40 체크포인트 비교가 그렇게 한 번 실패)
    if isinstance(obj, dict):
        return ("d", sorted((repr(k), canon(v, seen)) for k, v in obj.items()
                            if k != "elapsed" and "timer" not in str(k)))
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, [canon(v, seen) for v in obj])
    if isinstance(obj, np.generic):
        return ("np", obj.dtype.str, obj.tobytes())
    if hasattr(obj, "__dict__"):
        return (type(obj).__qualname__, canon(vars(obj), seen))
    try:
        return ("p", pickle.dumps(obj, protocol=5))
    except Exception:  # noqa: BLE001 - 피클할 수 없는 것(함수 등)은 이름만
        return ("r", type(obj).__qualname__)


def fingerprint(res: EvalResult, rows: range) -> list[tuple[str, bytes]]:
    """평가 결과에서 rows 줄의 모든 결과 칸(점수·스텝·블록·가중 블록·행동 평균·종료 사건·결과·궤적)을 바이트로."""
    out = [(name, getattr(res, name)[list(rows)].tobytes()) for name in ("scores", "steps", "blocks", "weighted",
                                                                          "action_abs")]
    out += [("end_events", repr([res.end_events[k] for k in rows]).encode()),
            ("outcomes", repr([res.outcomes[k] for k in rows]).encode())]
    for k in rows:
        tr = (res.trajectories or [None] * len(res.scores))[k]
        assert tr is not None
        out.append((f"trajectory {k - rows.start}", tr.poses.tobytes() + tr.outputs.tobytes()
                    + repr((tr.every, tr.dt, tr.end_step)).encode()))
    return out


class Spy(GrowingCMAES):
    """자라기 직전·직후 평균을 남기는 학습 방법(자람 자체는 그대로)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pairs: list[tuple[tuple[int, ...], np.ndarray, np.ndarray]] = []

    def _grow_now(self) -> None:
        before, counts = self.best(), self.counts
        super()._grow_now()
        self.pairs.append((counts, before, self.best()))


# ---------------------------------------------------------------------------
# v6.0 사본 도우미 (T40, T44)
# ---------------------------------------------------------------------------
V60_PACKAGE = "_evolab_v60"
V60_MODULES = {"cmaes": "evolab.learners.cmaes", "trainer": "evolab.engine.trainer", "bundle": "evolab.runs.bundle",
               "session": "evolab.engine.session", "commands": "evolab.engine.commands"}
"""사본 파일 이름 → v6.0 원래 모듈 이름(불러오는 순서). 사본끼리 부르는 import 줄만 사본 꾸러미로 바꾼다."""
V60_SCREEN_FILES = {"brain_view.py": "src/evolab/app/lab/brain_view.py"}
"""화면 사본(사본 파일 → 원본 경로): load_v60()이 부르지 않고 tests/test_grow_screens.py T49가 importlib로 따로 부른다(v6.0 뇌 그림)."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_v60() -> types.SimpleNamespace:
    """v6.0 출시판 사본(trainer·session·commands·bundle·cmaes)을 임시 꾸러미로 불러온다(한 번, 그 뒤로는 같은 모듈)."""
    if V60_PACKAGE in sys.modules:
        return sys.modules[V60_PACKAGE].ns  # type: ignore[attr-defined]
    package = types.ModuleType(V60_PACKAGE)
    package.__path__ = []  # type: ignore[attr-defined]
    sys.modules[V60_PACKAGE] = package
    loaded: dict[str, types.ModuleType] = {}
    for name in V60_MODULES:
        path = SNAPSHOT / f"{name}.py"
        source = path.read_text(encoding="utf-8")
        for other, real in V60_MODULES.items():
            source = source.replace(f"from {real} import", f"from {V60_PACKAGE}.{other} import")
        module = types.ModuleType(f"{V60_PACKAGE}.{name}")
        module.__file__ = str(path)
        sys.modules[module.__name__] = module
        exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102 - 저장소 안 시험 자료(출처표로 확인)
        loaded[name] = module
    package.ns = types.SimpleNamespace(**loaded)  # type: ignore[attr-defined]
    return package.ns  # type: ignore[attr-defined]


def test_v60_snapshot_files_match_their_sources() -> None:
    """사본 = 첫 줄 출처 주석 + 1e59b26 원본 바이트(SOURCES.txt의 sha256), 사본끼리 부르는 import 말고는 v6.0 모듈 그대로."""
    lines = [ln for ln in (SNAPSHOT / "SOURCES.txt").read_text(encoding="utf-8").splitlines()
             if ln and not ln.startswith("#")]
    assert lines[0].startswith("commit 1e59b26")
    table = {ln.split()[0]: ln.split()[1:] for ln in lines[1:]}
    assert set(table) == {f"{n}.py" for n in V60_MODULES} | set(V60_SCREEN_FILES)
    assert {name: table[name][0] for name in V60_SCREEN_FILES} == V60_SCREEN_FILES
    assert {p.name for p in SNAPSHOT.iterdir() if p.suffix == ".py"} == set(table)  # 출처표에 없는 사본이 없다
    for name, (src, _blob, src_sha, copy_sha) in table.items():
        data = (SNAPSHOT / name).read_bytes()
        assert _sha(data) == copy_sha, name
        head, _, body = data.partition(b"\n")
        assert head.startswith("# 출처: 커밋 1e59b26".encode()) and src.encode() in head
        assert _sha(body) == src_sha, name
    v60 = load_v60()
    assert v60.trainer.TrainConfig is not TrainConfig and "grow_start" not in v60.trainer.TrainConfig.__dataclass_fields__
    assert v60.trainer.CHECKPOINT_FORMAT == trainer_mod.CHECKPOINT_FORMAT == 2


# ---------------------------------------------------------------------------
# T31 꺼진 좌표는 정확히 +0.0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [LANDER_TEXT, TEXTS["hill_car"], brain_text([6, 5]), brain_text([4, 6, 3], "relu")],
                         ids=["lander", "hill_car-body", "two-layers", "three-layers-relu"])
def test_inactive_coordinates_are_exactly_positive_zero(text: str) -> None:
    layout = layout_of(text)
    lr = learner(layout, every=3)
    seen = set()
    for _ in range(14):
        counts = lr.counts
        seen.add(counts)
        X = lr.ask()
        off = outside(lr)
        assert np.all(X[:, off] == 0.0) and not np.any(np.signbit(X[:, off]))
        assert np.all(lr.ask()[:, off] == 0.0)  # 다시 ask해도 같은 후보
        lr.tell(X, fake_scores(X))
        for vec, cnt in ((lr.best(), lr.counts),):  # 자람 직후에도
            o = outside(lr, cnt)
            assert np.all(vec[o] == 0.0) and not np.any(np.signbit(vec[o]))
        assert np.all(np.isfinite(lr.best()))
    assert len(seen) >= 3 and lr.growths >= 2
    if layout.n_body:  # 몸 유전자는 늘 켜져 있고 [-1, 1] 안
        X = lr.ask()
        assert np.all(np.abs(X[:, :layout.n_body]) <= 1.0) and np.any(X[:, :layout.n_body] != 0.0)


def test_initial_mean_fills_incoming_weights_and_keeps_the_output_function() -> None:
    """처음 상태(뇌 0): 켜진 뉴런의 들어오는 가중치만 init_in × 정규 난수, 나가는 가중치·편향·꺼진 자리는 0(처음 함수 = 뇌 0과 같음)."""
    lr = learner()
    frame, mean = lr.frame, lr.best()
    assert lr.counts == (2,)
    for j in range(2):
        assert np.all(mean[frame.neuron_in(0, j, (2,))] != 0.0)
        assert np.all(mean[frame.neuron_out(0, j, (2,))] == 0.0) and mean[frame.bias(0, j)] == 0.0
    rng = np.random.Generator(np.random.PCG64(grow_rng_seed(3)))
    expect = np.concatenate([0.5 * rng.standard_normal(9), 0.5 * rng.standard_normal(9)])
    got = np.concatenate([mean[frame.neuron_in(0, j, (2,))] for j in range(2)])
    assert got.tobytes() == expect.tobytes()
    assert np.count_nonzero(mean) == 18
    # 출발점이 이미 켜진 뉴런을 가진 유전자(챔피언)면 그 수부터: 들어오는 가중치가 있는 뉴런은 그대로
    champ = lr.ask()[0]
    again = GrowingCMAES(champ, LAYOUT, CMAESConfig(popsize=8, seed=9), GrowConfig(start=1))
    assert again.counts == (2,) and again.best().tobytes() == champ.tobytes()
    bigger = GrowingCMAES(champ, LAYOUT, CMAESConfig(popsize=8, seed=9), GrowConfig(start=4))
    assert bigger.counts == (4,)
    b = bigger.best()
    assert b[frame.active_coords((2,))].tobytes() == champ[frame.active_coords((2,))].tobytes()
    assert np.all(b[frame.neuron_in(0, 3, (4,))] != 0.0) and np.all(b[frame.neuron_out(0, 3, (4,))] == 0.0)
    # 들어오는 가중치만 0이고 나가는 가중치가 있는 뉴런(편향만으로 출력에 닿음)은 채우지 않는다: 처음 함수가 그대로
    odd = champ.copy()
    odd[frame.neuron_in(0, 1, (2,))] = 0.0
    odd[frame.bias(0, 1)] = 0.7
    kept = GrowingCMAES(odd, LAYOUT, CMAESConfig(popsize=8, seed=9), GrowConfig(start=2))
    assert kept.counts == (2,) and kept.best().tobytes() == odd.tobytes()


# ---------------------------------------------------------------------------
# T32 함수 보존
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["lander", "track_car", "jump"])
def test_growth_preserves_the_function_bit_for_bit(name: str) -> None:
    """자라기 직전 평균과 직후 평균으로 시험 6개(궤적 포함)의 점수·스텝·블록·궤적이 바이트 같다(자람 3번: 2 → 4 → 8 → 16)."""
    spec = compile_env(TEXTS[name])
    layout = layout_for(spec)
    lr = Spy(layout.initial(), layout, CMAESConfig(popsize=8, seed=5, bounded_dims=layout.n_body), GrowConfig(every=1))
    for _ in range(3):
        generation(lr)
    assert [c for c, _b, _a in lr.pairs] == [(2,), (4,), (8,)] and lr.counts == (16,)
    trials = seeds.validation_trials(6)
    for counts, before, after in lr.pairs:
        assert not np.array_equal(before, after)  # 새 뉴런의 들어오는 가중치가 생겼다
        keep = lr.frame.active_coords(counts)
        assert after[keep].tobytes() == before[keep].tobytes()  # 옛 켜진 좌표는 그대로
        assert np.any(before[lr.frame.neuron_out(0, 0, counts)] != 0.0)  # 옛 뉴런의 출력 가중치가 배운 값(0이 아님)
        pairs = [(i, t) for i in range(2) for t in trials]
        res = evaluate(spec, layout, np.vstack([before, after]), pairs, record=[True] * len(pairs))
        assert fingerprint(res, range(0, 6)) == fingerprint(res, range(6, 12)), counts


@pytest.mark.parametrize(("hidden", "activation"), [([6, 5], "tanh"), ([4, 6, 3], "relu")])
def test_growth_preserves_multilayer_outputs(hidden: list[int], activation: str) -> None:
    """은닉층 2·3개 틀(시험 환경): 자라기 직전·직후 평균의 신경망 출력이 무작위 입력 100개에서 바이트 같고, 착륙선 시험 2개도 같다."""
    text = brain_text(hidden, activation)
    spec = compile_env(text)
    layout = layout_for(spec)
    lr = Spy(layout.initial(), layout, CMAESConfig(popsize=8, seed=5), GrowConfig(every=1))
    while lr.counts != tuple(hidden):
        generation(lr)
    assert len(lr.pairs) >= 2
    x = np.random.default_rng(1).standard_normal((100, layout.brain.n_in)) * 3.0
    for counts, before, after in lr.pairs:
        nets = [unpack(layout.brain, np.repeat(layout.brain_params(g)[None, :], 100, axis=0)) for g in (before, after)]
        ya, yb = (net.forward(x) for net in nets)
        assert ya.tobytes() == yb.tobytes(), counts
        pairs = [(i, t) for i in range(2) for t in seeds.validation_trials(2)]
        res = evaluate(spec, layout, np.vstack([before, after]), pairs, record=[True] * len(pairs))
        assert fingerprint(res, range(0, 2)) == fingerprint(res, range(2, 4)), counts


# ---------------------------------------------------------------------------
# T33 정해진 간격
# ---------------------------------------------------------------------------
def test_fixed_interval_doubling_stops_at_the_frame() -> None:
    lr = learner(every=10)
    grown: list[tuple[int, tuple[int, ...]]] = []
    for g in range(1, 46):
        generation(lr)
        if lr.last_growth is not None:
            grown.append((g, lr.counts))
        assert lr.stats()["grown"] == (1.0 if lr.last_growth is not None else 0.0)
    assert grown == [(10, (4,)), (20, (8,)), (30, (16,))]
    assert lr.growths == 3 and lr.since == 15
    assert np.array_equal(lr.active_coords(), np.arange(LAYOUT.dim))
    assert lr.events == [[10, 4, 48], [20, 8, 96], [30, 16, 192]]
    # 틀 [12], 간격 3: 2 → 4 → 8 → 12(두 배가 틀을 넘으면 틀에서 멈춤)
    lr12 = learner(layout_of(brain_text([12])), every=3)
    path = [lr12.counts]
    for _ in range(15):
        generation(lr12)
        if lr12.last_growth is not None:
            path.append(lr12.counts)
    assert path == [(2,), (4,), (8,), (12,)] and lr12.last_growth is None
    # 층마다 따로 틀에서 멈춤: [6, 5]에서 (2, 2) → (4, 4) → (6, 5)
    lr2 = learner(layout_of(brain_text([6, 5])), every=2)
    path2 = [lr2.counts]
    for _ in range(8):
        generation(lr2)
        if lr2.last_growth is not None:
            path2.append(lr2.counts)
    assert path2 == [(2, 2), (4, 4), (6, 5)]


def test_growth_schedule_matches_the_learner() -> None:
    """화면 안내 줄용 일정(growth_schedule)이 실제 학습 방법이 자라는 세대·크기와 같다."""
    from evolab.learners.grow import growth_schedule

    assert growth_schedule((16,), 2, 10) == [(0, (2,)), (10, (4,)), (20, (8,)), (30, (16,))]
    assert growth_schedule((16,), 16, 10) == [(0, (16,))] and growth_schedule((16,), 99, 3) == [(0, (16,))]
    assert growth_schedule((12,), 2, 3) == [(0, (2,)), (3, (4,)), (6, (8,)), (9, (12,))]
    assert growth_schedule((6, 5), 2, 2) == [(0, (2, 2)), (2, (4, 4)), (4, (6, 5))]
    assert growth_schedule((7,), 3, 1) == [(0, (3,)), (1, (6,)), (2, (7,))]
    for sizes, start, every in (((16,), 2, 4), ((6, 5), 1, 2), ((9,), 3, 1)):
        lr = learner(layout_of(brain_text(list(sizes))), start=start, every=every)
        seen = [(0, lr.counts)]
        for g in range(1, 40):
            generation(lr)
            if lr.last_growth is not None:
                seen.append((g, lr.counts))
        assert seen == growth_schedule(sizes, start, every)
    for bad in ((0, 1), (1, 0), (True, 1)):
        with pytest.raises(ValueError):
            growth_schedule((16,), *bad)


def test_start_is_clipped_to_the_frame_and_never_grows() -> None:
    lr = learner(start=40)
    assert lr.counts == (16,) and np.array_equal(lr.active_coords(), np.arange(LAYOUT.dim))
    for _ in range(12):
        generation(lr)
    assert lr.growths == 0 and lr.events == [] and lr.counts == (16,)
    assert learner(layout_of(brain_text([6, 3])), start=4).counts == (4, 3)


# ---------------------------------------------------------------------------
# T34 결정성
# ---------------------------------------------------------------------------
FRAMES: dict[str, tuple[str, int, int, int]] = {
    "lander": (LANDER_TEXT, 10, 40, 3),
    "two-layers": (brain_text([6, 5]), 3, 15, 2),
    "three-layers-relu": (brain_text([4, 6, 3], "relu"), 3, 15, 2),
}
"""T34·T35의 틀: 이름 → (환경 글, 자라는 간격, 세대 수, 그동안 자라는 횟수). 은닉층 2·3개 틀은 원형이 재지 않아 함수 보존(T32)과
결정성·저장 테스트만 지킨다(설계 4.15). 착륙선 (2) → (16), [6, 5]는 (2, 2) → (4, 4) → (6, 5), [4, 6, 3]은 (2, 2, 2) → (4, 4, 3) → (4, 6, 3)."""


def _same_random_state(a: Any, b: Any) -> bool:
    return bool(a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:])


@pytest.mark.parametrize("name", list(FRAMES))
def test_same_seed_same_candidates_and_no_global_numpy_random(name: str) -> None:
    text, every, gens, growths = FRAMES[name]
    layout = layout_of(text)
    assert layout.dim < 300  # 켜진 좌표 300개 미만: 전역 난수 상태가 그대로다(300개 이상은 아래 테스트)
    before = np.random.get_state()
    runs = []
    for seed in (3, 3, 4):
        lr = learner(layout, seed=seed, every=every)
        runs.append(b"".join(generation(lr).tobytes() + lr.best().tobytes() for _ in range(gens)))
        assert lr.growths == growths and lr.counts == lr.frame.sizes
    after = np.random.get_state()
    assert runs[0] == runs[1] and runs[0] != runs[2]
    assert _same_random_state(before, after)


def test_three_hundred_active_coordinates_do_not_depend_on_the_global_numpy_random() -> None:
    """켜진 좌표가 300개 이상이면(pycma TPA_dimension) 안쪽 pycma가 폭 적응을 TPA로 고르고, 그 일관성 검사가 tell마다
    np.random.randint로 검사할 좌표 번호를 뽑는다(learners/grow.py 「난수」, learners/cmaes.py). 그래서 「전역 난수 상태가 그대로」는
    이 크기에서 성립하지 않는다. 대신 결과가 전역 난수와 무관해야 한다: 전역 시드를 바꿔도 후보·추천해가 바이트 같고, 학습 방법
    시드를 바꾸면 다르다. 착륙선 은닉 24개 틀(9·24 + 24 + 24·3 + 3 = 315차원), 간격 1이라 4세대 끝에 다 자란다."""
    layout = layout_of(brain_text([24]))
    assert layout.dim == 315
    saved = np.random.get_state()
    try:
        runs: dict[tuple[int, int], bytes] = {}
        touched = []
        for global_seed, seed in ((1, 3), (2, 3), (1, 4)):
            np.random.seed(global_seed)
            start = np.random.get_state()
            lr = learner(layout, seed=seed, every=1)
            runs[global_seed, seed] = b"".join(generation(lr).tobytes() + lr.best().tobytes() for _ in range(14))
            assert lr.counts == (24,) and lr.growths == 4 and lr.inner.dim == 315
            touched.append(not _same_random_state(start, np.random.get_state()))
    finally:
        np.random.set_state(saved)
    assert runs[1, 3] == runs[2, 3] and runs[1, 3] != runs[1, 4]
    # 이 틀이 정말 그 길(pycma가 전역 난수를 뽑는 길)을 지났다. pycma가 더는 뽑지 않게 되면 여기서 실패한다: 그때는 위 테스트의
    # 300 미만 조건과 learners/grow.py의 예외 설명을 지우면 된다
    assert all(touched)


def test_seed_words() -> None:
    assert (GROW_SEED_WORD, GROW_RNG_WORD, GROW_STATE_FORMAT) == (0x67726F77, 0x726E67, 1)
    for s in (0, 1, 12345, 2**32 - 1):
        assert grow_seed(s, 1) == seeds.derive(s, GROW_SEED_WORD, 1) & 0xFFFFFFFF
        assert grow_rng_seed(s) == seeds.derive(s, GROW_RNG_WORD)
    assert len({grow_seed(7, k) for k in range(1, 5)}) == 4
    lr = learner(seed=11, every=1)
    generation(lr)
    assert lr.inner.config.seed == grow_seed(11, 1) and lr.inner.config.stds


# ---------------------------------------------------------------------------
# T35 저장·복원
# ---------------------------------------------------------------------------
def _continue(lr: GrowingCMAES, gens: int) -> bytes:
    return b"".join(generation(lr).tobytes() + lr.best().tobytes() + repr(lr.stats()).encode() for _ in range(gens))


@pytest.mark.parametrize("name", ["hill_car", "two-layers", "three-layers-relu"])
@pytest.mark.parametrize("when", [2, 3, 4, 5, 6, 7, 11])
def test_state_roundtrip_through_pickle_continues_identically(name: str, when: int) -> None:
    """간격 3: 2·5세대 끝 = 자람 직전, 3·6세대 끝 = 자람 직후, 4·7세대 = 자란 뒤 한 세대, 11 = 한참 뒤(다 자란 뒤). 몸 유전자가 있는
    언덕 자동차와 은닉층 2·3개 틀(자람마다 그 직전·직후)."""
    lr = learner(layout_of(TEXTS[name] if name in TEXTS else FRAMES[name][0]), every=3)
    for _ in range(when):
        generation(lr)
    state = pickle.loads(pickle.dumps(lr.state_dict()))
    other = GrowingCMAES(lr.frame.layout.initial(), lr.frame.layout, CMAESConfig(popsize=4, seed=99),
                         GrowConfig(start=1, every=50))
    other.load_state_dict(state)
    assert (other.counts, other.growths, other.since, other.popsize) == (lr.counts, lr.growths, lr.since, lr.popsize)
    assert other.stats() == lr.stats() or repr(other.stats()) == repr(lr.stats())
    assert other.best().tobytes() == lr.best().tobytes()
    assert _continue(other, 8) == _continue(lr, 8)


@pytest.mark.parametrize("name", list(FRAMES))
@pytest.mark.parametrize("told", [1, 2, 3])
def test_state_between_ask_and_tell_is_the_generation_boundary(name: str, told: int) -> None:
    """간격 2(2·4세대 끝에 자람). told = 1·3: 저장한 ask의 tell에서 자란다(첫째·둘째 자람 직전의 ask와 tell 사이), 2: 자란 직후 첫 ask."""
    layout = layout_of(FRAMES[name][0])
    lr = learner(layout, every=2)
    for _ in range(told):
        generation(lr)
    X = lr.ask()
    state = pickle.loads(pickle.dumps(lr.state_dict()))
    other = learner(layout, seed=77)
    other.load_state_dict(state)
    assert other.ask().tobytes() == X.tobytes()
    lr.tell(X, fake_scores(X))
    other.tell(X, fake_scores(X))
    assert (lr.last_growth is not None) == (told != 2) and other.last_growth == lr.last_growth
    assert other.counts == lr.counts and other.best().tobytes() == lr.best().tobytes()
    assert _continue(other, 5) == _continue(lr, 5)


def test_broken_states_are_refused_and_leave_the_learner_unchanged() -> None:
    lr = learner(every=2)
    for _ in range(3):
        generation(lr)
    good = lr.state_dict()
    target = learner(seed=5, every=2)
    snapshot = repr(target.state_dict())

    def broken(**changes: Any) -> dict[str, Any]:
        d = pickle.loads(pickle.dumps(good))
        for key, value in changes.items():
            d[key] = value
        return d

    off = int(np.flatnonzero(outside(lr))[0])
    bad_mean = list(good["mean"])
    bad_mean[off] = 0.25
    cases: list[tuple[Any, str]] = [
        ("not a dict", S.LEARNER_ERR_STATE_BROKEN.format(reason="str")),
        (broken(name="cmaes"), S.LEARNER_ERR_STATE_KIND.format(got="cmaes", want="grow")),
        (broken(format=2), S.LEARNER_ERR_STATE_FORMAT.format(got=2)),
        (broken(dim=10), S.LEARNER_ERR_STATE_DIM.format(got=10, want=LAYOUT.dim)),
        (broken(counts=[17]), ""), (broken(counts=[0]), ""), (broken(counts=[2, 2]), ""), (broken(counts="4"), ""),
        (broken(mean=bad_mean), ""), (broken(mean=good["mean"][:-1]), ""), (broken(growths=-1), ""),
        (broken(events=[]), ""), (broken(rng={"bit_generator": "MT"}), ""), (broken(inner={}), ""),
        (broken(counts=[8], told=[8]), ""),  # 안쪽 상태의 차원(켜진 좌표 55개)과 맞지 않음
    ]
    for state, message in cases:
        with pytest.raises(ValueError) as err:
            target.load_state_dict(state)  # type: ignore[arg-type]
        if message:
            assert str(err.value) == message
        assert repr(target.state_dict()) == snapshot  # 반쯤 불러온 상태가 남지 않음
    target.load_state_dict(good)
    assert target.ask().tobytes() == lr.ask().tobytes()


def test_state_keys_and_cma_config_without_stds() -> None:
    lr = learner(every=1)
    generation(lr)
    d = lr.state_dict()
    assert {"format", "name", "config", "counts", "growths", "since", "rng", "mean", "inner", "events"} <= set(d)
    assert d["format"] == GROW_STATE_FORMAT and d["name"] == "grow" and d["counts"] == [4] and d["since"] == 0
    assert "stds" not in d["config"]["cma"] and d["config"]["grow"] == {"start": 2, "every": 1, "init_in": 0.5}
    assert isinstance(d["mean"], list) and len(d["mean"]) == LAYOUT.dim
    assert len(d["inner"]["config"]["stds"]) == lr.inner.dim == 55  # 자란 안쪽 CMA-ES만 좌표별 폭을 가진다
    assert d["inner"]["config"]["seed"] == grow_seed(3, 1)


# ---------------------------------------------------------------------------
# T43 「탐색 폭」과 대각 문턱
# ---------------------------------------------------------------------------
def test_sigma_is_the_median_coordinate_std_and_does_not_jump_at_growth() -> None:
    lr = learner(every=5)
    sigmas = []
    for _ in range(18):
        generation(lr)
        st = lr.stats()
        assert st["sigma"] == float(np.median(lr.inner.coordinate_stds()))
        sigmas.append((st["sigma"], st["grown"]))
    for k, (sigma, grown) in enumerate(sigmas[1:-1], start=1):
        if grown:
            assert 0.5 <= sigma / sigmas[k - 1][0] <= 2.0 and 0.5 <= sigmas[k + 1][0] / sigma <= 2.0
    assert sum(g for _s, g in sigmas) == 3
    # 새 좌표의 처음 폭 = 옛 폭의 중앙값, 옛 좌표 = 옛 폭(pycma가 처음 D에 더하는 1e-5 단위 차이만)
    class StdSpy(GrowingCMAES):
        def _grow_now(self) -> None:
            self.old_stds = self.inner.coordinate_stds()
            super()._grow_now()

    lr2 = StdSpy(LAYOUT.initial(), LAYOUT, CMAESConfig(popsize=8, seed=3), GrowConfig(every=4))
    for _ in range(4):
        generation(lr2)
    assert lr2.last_growth is not None
    old = lr2.old_stds
    new = lr2.inner.coordinate_stds()
    assert new.size == 55 and lr2.inner.config.sigma0 == 0.5  # 칸은 그대로, pycma에는 1.0(CMAESConfig.stds 규칙)
    given = np.array(lr2.inner.config.stds)
    assert np.allclose(given, new, rtol=1e-3)
    full = np.full(LAYOUT.dim, np.median(old))
    full[lr2.frame.active_coords((2,))] = old
    assert given.tobytes() == full[lr2.frame.active_coords((4,))].tobytes()


def test_diagonal_threshold_applies_to_the_active_coordinates() -> None:
    lr = learner(every=1, diagonal_threshold=60)
    diag = [(lr.inner.dim, lr.inner.diagonal)]
    for _ in range(3):
        generation(lr)
        diag.append((lr.inner.dim, lr.inner.diagonal))
    assert diag == [(29, False), (55, False), (107, True), (211, True)]


def test_stats_keys_and_size_columns_before_growth() -> None:
    lr = learner(every=2)
    keys = set(CMAES(np.zeros(4), CMAESConfig(popsize=4)).stats())
    generation(lr)
    st = lr.stats()
    assert keys | {"neurons", "links", "growths", "grown"} == set(st)
    assert (st["neurons"], st["links"], st["grown"], st["generation"], st["evaluations"]) == (2, 24, 0, 1, 8)
    generation(lr)  # 2세대 끝에 자람: 이 세대 후보를 만든 크기(2)와 자랐다는 표시
    st = lr.stats()
    assert (st["neurons"], st["links"], st["grown"], st["growths"]) == (2, 24, 1, 1)
    assert lr.last_growth == (2, 4, 24, 48)
    generation(lr)
    st = lr.stats()
    assert (st["neurons"], st["links"], st["grown"], st["generation"]) == (4, 48, 0, 3)


def test_constructor_checks() -> None:
    with pytest.raises(FrameError) as no_hidden:
        learner(layout_of(brain_text([])))
    assert str(no_hidden.value) == S.FRAME_NO_HIDDEN
    with pytest.raises(FrameError) as no_brain:
        learner(layout_of(TEXTS["glider"]))
    assert str(no_brain.value) == S.FRAME_NO_BRAIN
    with pytest.raises(ValueError) as ipop:
        learner(ipop=True)
    assert str(ipop.value) == S.TRAIN_GROW_NO_IPOP
    for kwargs, message in (({"start": 0}, S.LEARNER_ERR_GROW_START.format(value=0)),
                            ({"every": 0}, S.LEARNER_ERR_GROW_EVERY.format(value=0)),
                            ({"every": True}, S.LEARNER_ERR_GROW_EVERY.format(value=True)),
                            ({"init_in": 0.0}, S.LEARNER_ERR_GROW_INIT.format(value=0.0)),
                            ({"init_in": math.inf}, S.LEARNER_ERR_GROW_INIT.format(value=math.inf))):
        with pytest.raises(ValueError) as err:
            GrowConfig(**kwargs)
        assert str(err.value) == message
    with pytest.raises(FrameError):
        GrowingCMAES(np.zeros(10), LAYOUT, CMAESConfig(), GrowConfig())


# ---------------------------------------------------------------------------
# CMAESConfig.stds (T40의 학습 방법 쪽)
# ---------------------------------------------------------------------------
def test_cmaes_config_without_stds_is_the_v60_config() -> None:
    v60 = load_v60()
    for cfg in ({}, {"popsize": 6, "sigma0": 0.3, "diagonal_threshold": 5, "ipop": True, "seed": 9, "bounded_dims": 2}):
        new, old = CMAES(np.zeros(8), CMAESConfig(**cfg)), v60.cmaes.CMAES(np.zeros(8), v60.cmaes.CMAESConfig(**cfg))
        a, b = new.state_dict(), old.state_dict()
        assert "stds" not in a["config"] and a["config"] == b["config"]
        assert pickle.dumps(a["config"]) == pickle.dumps(b["config"])
        assert sorted(a) == sorted(b)
        for _ in range(4):
            Xa, Xb = new.ask(), old.ask()
            assert Xa.tobytes() == Xb.tobytes()
            new.tell(Xa, fake_scores(Xa))
            old.tell(Xb, fake_scores(Xb))
        assert repr(new.stats()) == repr(old.stats())
        assert canon(new.state_dict()) == canon(old.state_dict())
        loaded = CMAES(np.zeros(8), CMAESConfig())
        loaded.load_state_dict(old.state_dict())  # v6.0 상태를 그대로 읽는다
        assert loaded.ask().tobytes() == new.ask().tobytes()


def test_cmaes_stds_sets_the_initial_coordinate_stds() -> None:
    stds = (0.1, 0.2, 0.4, 0.8, 1.6)
    es = CMAES(np.zeros(5), CMAESConfig(stds=list(stds), sigma0=3.0, bounded_dims=2))  # type: ignore[arg-type]
    assert es.config.stds == stds and es.state_dict()["config"]["stds"] == list(stds)
    assert np.allclose(es.coordinate_stds(), stds, rtol=1e-4)
    for bad in ((0.1, -1.0), (0.1, math.nan), (True,), "ab"):
        with pytest.raises(ValueError):
            CMAESConfig(stds=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError) as dim:
        CMAES(np.zeros(4), CMAESConfig(stds=stds))
    assert str(dim.value) == S.LEARNER_ERR_STDS_DIM.format(got=5, dim=4)
    restored = CMAES(np.zeros(5), CMAESConfig())
    restored.load_state_dict(pickle.loads(pickle.dumps(es.state_dict())))
    assert restored.config.stds == stds and restored.ask().tobytes() == es.ask().tobytes()


# ---------------------------------------------------------------------------
# 학습기 도우미
# ---------------------------------------------------------------------------
def train(text: str, config: TrainConfig, store: RunStore | None = None, *, workers: int = 0, observe: bool | None = None,
          record: bool = False, trainer_cls: Any = Trainer) -> tuple[Any, list[dict[str, Any]]]:
    """max_generations까지 돈다(관전이면 record=True로 세대마다). 트레이너와 통계 줄."""
    with WorkerPool(workers) as pool:
        t = trainer_cls(text, config, pool, store, observe=observe)
        rows = []
        while t.check_stop() is None:
            rows.append(t.step(record=record))
        t.finish()
    return t, rows


def keep_all_store(base: Path, text: str = LANDER_TEXT, name: str = "run", config: TrainConfig | None = None) -> RunStore:
    """체크포인트를 모두 남기는 실행 폴더(config를 주면 run.json에도 적는다: 재현 묶음이 읽는다)."""
    info: dict[str, Any] = {"env_name": "착륙선"}
    if config is not None:
        info.update(method=config.method, config=config.to_dict())
    store = RunStore.create(base, name, info, policy=KEEP_ALL)
    store.add_env_version(text, 0)
    return store


def grown_rows(rows: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    return [(int(r["generation"]), int(r["brain_neurons"]), int(r["brain_links"])) for r in rows if int(r["grow"])]


# ---------------------------------------------------------------------------
# T37 통계 열
# ---------------------------------------------------------------------------
def test_stats_columns_only_for_growing_runs(tmp_path: Path) -> None:
    store = keep_all_store(tmp_path)
    t, rows = train(LANDER_TEXT, small(max_generations=11), store)
    header = store.stats_columns()
    assert header is not None
    k = header.index("switch")
    assert tuple(header[k + 1:k + 4]) == GROW_COLUMNS and not set(GROW_COLUMNS) & TIME
    assert [int(r["brain_neurons"]) for r in rows] == [2] * 3 + [4] * 3 + [8] * 3 + [16] * 2
    assert [int(r["brain_links"]) for r in rows] == [24] * 3 + [48] * 3 + [96] * 3 + [192] * 2
    assert grown_rows(rows) == [(3, 2, 24), (6, 4, 48), (9, 8, 96)]  # 자란 세대 줄의 크기 = 자라기 전
    events = t.learner.events
    assert [e[0] for e in events] == [3, 6, 9] and [e[1] for e in events] == [4, 8, 16]
    saved = store.read_stats()
    assert [(r["brain_neurons"], r["grow"]) for r in saved][:4] == [("2", "0"), ("2", "0"), ("2", "1"), ("4", "0")]
    # 크기 고정 새 실행: 열 머리에 새 열이 없다(v6.0과 같음은 T40)
    fixed = keep_all_store(tmp_path, name="fixed")
    train(LANDER_TEXT, small(method="cmaes", max_generations=1), fixed)
    assert not set(GROW_COLUMNS) & set(fixed.stats_columns() or [])
    assert [c for c in header if c not in GROW_COLUMNS] == fixed.stats_columns()


def test_diversity_column_of_growing_runs_covers_only_the_active_coordinates() -> None:
    """자라는 실행의 diversity 열 = 그 세대 후보의 켜진 좌표만의 표준편차 평균. 꺼진 좌표는 늘 0이라 틀 전체로 평균하면 값이
    (켜진 좌표 수 ÷ 틀 길이)배로 줄고(뉴런 2개일 때 29/211) 자랄 때마다 두 배쯤 뛴다(고치기 전: 1.8~2.0배, 「탐색 폭」의 1/7).
    「탐색 폭」(sigma)과 같은 단위(「탐색 폭과 다양성」 타일이 한 눈금에 그린다)라 자란 세대 앞뒤로 튀지 않는다.
    크기 고정 실행은 v6.0 식(모든 유전자의 평균) 그대로다(바이트 비교는 T40)."""
    t = Trainer(LANDER_TEXT, small(max_generations=11), ConstantValidation(), None)
    frame = BrainFrame(t.layout)
    rows = []
    while t.check_stop() is None:
        row = t.step()
        assert t.last_generation is not None
        X = t.last_generation.population
        active = frame.active_coords((int(row["brain_neurons"]),))
        assert row["diversity"] == pytest.approx(float(np.mean(np.std(X[:, active], axis=0))), rel=1e-12)
        whole = float(np.mean(np.std(X, axis=0)))  # 고치기 전 값: 틀 전체(꺼진 좌표의 0 포함)
        assert whole == pytest.approx(row["diversity"] * len(active) / t.layout.dim, rel=1e-9)
        assert 0.5 < row["diversity"] / row["sigma"] < 1.5, row["generation"]
        rows.append(row)
    assert grown_rows(rows) == [(3, 2, 24), (6, 4, 48), (9, 8, 96)]
    for k, r in enumerate(rows[:-1]):
        if r["grow"]:  # 자란 세대 줄 → 다음 줄(새 크기의 첫 후보): 「탐색 폭」처럼 이어진다
            assert 0.67 < rows[k + 1]["diversity"] / r["diversity"] < 1.5, r["generation"]
    fixed = Trainer(LANDER_TEXT, small(method="cmaes", max_generations=2), ConstantValidation(), None)
    row = fixed.step()
    assert fixed.last_generation is not None
    assert row["diversity"] == float(np.mean(np.std(fixed.last_generation.population, axis=0)))


def test_trainer_last_growth(tmp_path: Path) -> None:
    with WorkerPool(0) as pool:
        t = Trainer(LANDER_TEXT, small(grow_every=2), pool, None)
        assert t.last_growth is None
        seen = []
        for _ in range(5):
            t.step()
            seen.append(t.last_growth)
    assert seen == [None, (2, 4, 24, 48), None, (4, 8, 48, 96), None]
    with WorkerPool(0) as pool:
        fixed = Trainer(LANDER_TEXT, small(method="cmaes"), pool, None)
        fixed.step()
    assert fixed.last_growth is None


# ---------------------------------------------------------------------------
# T38 설정 검사
# ---------------------------------------------------------------------------
def _invalid(name: str, value: Any) -> str:
    return S.TRAIN_CONFIG_INVALID.format(name=name, value=value)


@pytest.mark.parametrize(("over", "message"), [
    ({"grow_start": 0}, _invalid("grow_start", 0)),
    ({"grow_start": -3}, _invalid("grow_start", -3)),
    ({"grow_start": 2.0}, _invalid("grow_start", 2.0)),
    ({"grow_start": True}, _invalid("grow_start", True)),
    ({"grow_every": 0}, _invalid("grow_every", 0)),
    ({"grow_every": None}, _invalid("grow_every", None)),
    ({"ipop": True}, S.TRAIN_GROW_NO_IPOP),
])
def test_validate_refuses_bad_grow_settings(over: dict[str, Any], message: str) -> None:
    with pytest.raises(TrainerChangeError) as err:
        small(**over).validate()
    assert str(err.value) == message


def test_validate_accepts_grow_and_ignores_grow_keys_of_fixed_runs() -> None:
    small().validate()
    small(grow_start=40, grow_every=1).validate()
    for method in ("cmaes", "ga", "sa"):
        small(method=method, grow_start=0, grow_every=0).validate()  # 크기 고정 실행은 grow_*를 쓰지 않는다
    small(method="cmaes", ipop=True).validate()
    with pytest.raises(TrainerChangeError) as err:
        small(method="neat").validate()
    assert str(err.value) == _invalid("method", "neat")
    assert trainer_mod.METHODS == ("cmaes", "ga", "sa", "grow")


def test_no_hidden_layer_or_no_brain_is_refused_in_korean() -> None:
    no_hidden = brain_text([])
    for text, message in ((no_hidden, S.TRAIN_GROW_NO_HIDDEN), (TEXTS["glider"], S.FRAME_NO_BRAIN)):
        assert grow_problem(small(), layout_of(text)) == message
        assert grow_problem(small(method="cmaes"), layout_of(text)) is None
        with WorkerPool(0) as pool, pytest.raises(TrainerChangeError) as err:
            Trainer(text, small(), pool, None)
        assert str(err.value) == message
    assert grow_problem(small(), LAYOUT) is None
    with pytest.raises(FrameError):  # 반복 실행 대화 상자(available_methods)는 make_learner가 못 만들면 뺀다
        make_learner(small(), layout_of(no_hidden))


def test_start_larger_than_the_frame_is_clipped(tmp_path: Path) -> None:
    t, rows = train(LANDER_TEXT, small(grow_start=64, max_generations=4))
    assert t.learner.counts == (16,) and t.config.grow_start == 64  # 설정은 그대로 남고 학습 방법이 틀로 자른다
    assert [int(r["brain_neurons"]) for r in rows] == [16] * 4 and not grown_rows(rows)
    two = brain_text([6, 3])
    t2, _ = train(two, small(grow_start=4, max_generations=1))
    assert t2.learner.counts == (4, 3)


def test_grow_settings_cannot_change_during_training() -> None:
    with WorkerPool(0) as pool:
        t = Trainer(LANDER_TEXT, small(), pool, None)
        t.step()
        for key in GROW_KEYS:
            with pytest.raises(TrainerChangeError) as err:
                t.apply_config({key: 4})
            assert str(err.value) == S.TRAIN_CHANGE_NOT_ALLOWED.format(keys=key)
        assert not set(GROW_KEYS) & set(trainer_mod.LIVE_CONFIG_KEYS)
        assert t.apply_config({"max_generations": 9}) == {"max_generations": 9}
        assert t.config.grow_every == 3 and t.config.method == "grow"


def test_session_new_run_without_hidden_layer_fails_before_creating_a_folder(tmp_path: Path) -> None:
    from evolab.engine.session import Session, SessionRequest

    msgs = run_session(Session, SessionRequest(kind="new", env_text=brain_text([]), config=small().to_dict(),
                                               runs_dir=str(tmp_path / "runs"), workers=0))
    errors = [m for m in msgs if m["type"] == "error"]
    assert errors and errors[0]["fatal"] and errors[0]["message"] == S.TRAIN_CANNOT_START.format(
        reason=S.TRAIN_GROW_NO_HIDDEN)
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


def run_session(session_cls: Any, request: Any, timeout: float = 120.0) -> list[dict[str, Any]]:
    """학습 프로세스 없이 이 프로세스에서 세션 하나를 돌린다(시작에 실패하는 요청용): 끝까지 받은 결과 메시지."""
    cmd_r, cmd_w = mp.Pipe(duplex=False)
    ev_r, ev_w = mp.Pipe(duplex=False)
    session_cls(cmd_r, ev_w, request).main()
    msgs: list[dict[str, Any]] = []
    while ev_r.poll(timeout):
        try:
            msg = ev_r.recv()
        except EOFError:
            break
        msgs.append(msg)
        if msg.get("type") == "closed":
            break
    cmd_w.close()
    ev_r.close()
    return msgs


def drive_session(session_cls: Any, request: Any, timeout: float = 120.0) -> list[dict[str, Any]]:
    """학습 프로세스 없이 세션 하나를 스레드에서 끝까지 돌린다(시작에 성공하는 요청용): 「끝남」 상태가 오면 quit을 보내고, closed
    까지 받은 결과 메시지. 화면(TrainingClient)이 하는 일을 파이프로 직접 한다."""
    cmd_r, cmd_w = mp.Pipe(duplex=False)
    ev_r, ev_w = mp.Pipe(duplex=False)
    worker = threading.Thread(target=lambda: session_cls(cmd_r, ev_w, request).main(), daemon=True)
    worker.start()
    msgs: list[dict[str, Any]] = []
    while ev_r.poll(timeout):
        try:
            msg = ev_r.recv()
        except EOFError:
            break
        msgs.append(msg)
        if msg.get("type") == "state" and msg.get("state") == "finished":
            cmd_w.send({"type": "quit"})
        if msg.get("type") == "closed":
            break
    worker.join(30)
    assert not worker.is_alive(), [m.get("type") for m in msgs]
    cmd_w.close()
    ev_r.close()
    return msgs


# ---------------------------------------------------------------------------
# T39 to_dict
# ---------------------------------------------------------------------------
def test_to_dict_writes_grow_keys_only_for_growing_runs() -> None:
    for method in ("cmaes", "ga", "sa"):
        c = TrainConfig.for_new_run(LANDER_SPEC, method=method, grow_start=5, grow_every=7)
        d = c.to_dict()
        assert not set(GROW_KEYS) & set(d)
        assert {"restart_patience", "restart_from", "qualify_entrants", "qualify_trials", "ga_structure_flip",
                "sa_structure_flip"} <= set(d)  # 이미 출시된 키는 지금처럼 늘 쓴다
        assert list(d) == [f.name for f in dataclasses.fields(TrainConfig) if f.name not in GROW_KEYS]
        assert TrainConfig.from_dict(d) == dataclasses.replace(c, grow_start=2, grow_every=10)
    g = TrainConfig.for_new_run(LANDER_SPEC, method="grow", grow_start=4, grow_every=6)
    d = g.to_dict()
    assert (d["method"], d["grow_start"], d["grow_every"]) == ("grow", 4, 6) and list(d)[-2:] == list(GROW_KEYS)
    assert TrainConfig.from_dict(d) == g and TrainConfig.from_dict(json.loads(json.dumps(d))) == g
    assert TrainConfig.from_dict({}) == TrainConfig() and TrainConfig().to_dict() == {
        k: v for k, v in dataclasses.asdict(TrainConfig()).items() if k not in GROW_KEYS}


# ---------------------------------------------------------------------------
# T36 학습기 결정성
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lander_reference(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict[str, str]]]:
    """착륙선 자라는 실행 11세대(3·6·9세대 끝에 자람), 체크포인트 매 세대(모두 남김), 워커 0, 관찰 켬."""
    base = tmp_path_factory.mktemp("lander_ref")
    config = small(max_generations=11, checkpoint_every=1)
    store = keep_all_store(base, config=config)
    _t, rows = train(LANDER_TEXT, config, store)
    assert grown_rows(rows) == [(3, 2, 24), (6, 4, 48), (9, 8, 96)]
    assert store.checkpoints() == list(range(1, 12))
    return store.root, strip(store.read_stats())


def test_workers_record_and_observe_do_not_change_growing_runs(tmp_path: Path,
                                                               lander_reference: tuple[Path, list[Any]]) -> None:
    ref_dir, ref = lander_reference
    ref_store = RunStore.open(ref_dir)
    for k, (workers, record, observe) in enumerate(((1, True, True), (3, False, False), (0, True, False))):
        store = keep_all_store(tmp_path / str(k))
        train(LANDER_TEXT, small(max_generations=11, checkpoint_every=1), store, workers=workers, observe=observe,
              record=record)
        assert strip(store.read_stats()) == ref, (workers, record, observe)
        if observe:
            assert champion_files(store) == champion_files(ref_store)
        assert genomes_of(store) == genomes_of(ref_store)
        for g in (2, 3, 9, 11):
            assert canon(store.load_checkpoint(g)[1]) == canon(ref_store.load_checkpoint(g)[1]), g


def test_resume_at_every_generation_boundary_including_around_growth(
        tmp_path: Path, lander_reference: tuple[Path, list[Any]]) -> None:
    """2·3세대(자람 직전·직후), 5·6·7, 8·9·10 … 모든 체크포인트에서 이어 가도 끊지 않은 실행과 같다."""
    ref_dir, ref = lander_reference
    ref_genomes = genomes_of(RunStore.open(ref_dir))
    for g in range(1, 11):
        copy = tmp_path / f"from_{g}"
        shutil.copytree(ref_dir, copy)
        store = RunStore.open(copy, policy=KEEP_ALL)
        with WorkerPool(0) as pool:
            t = Trainer.resume(store, pool, g, max_generations=11)
            assert t.learner.counts == ((2,), (4,), (8,), (16,))[min(g // 3, 3)]
            t.run()
        assert strip(store.read_stats()) == ref, g
        assert genomes_of(store) == ref_genomes, g


def test_branch_and_bundle_reproduce_growing_runs(tmp_path: Path, lander_reference: tuple[Path, list[Any]]) -> None:
    ref_dir, ref = lander_reference
    child = RunStore.open(branch_run(ref_dir, 6, tmp_path / "branch"))
    with WorkerPool(0) as pool:
        Trainer.resume(child, pool, None, max_generations=11).run()
    assert strip(child.read_stats()) == ref
    bundle = export_bundle(ref_dir, tmp_path / "bundle.zip")
    again = RunStore.open(reproduce(bundle, tmp_path / "again", workers=0))
    assert compare_stats(read_bundle(bundle)["stats"], again.read_stats()) is None
    assert strip(again.read_stats()) == ref and again.read_run_info()["config"]["method"] == "grow"
    assert genomes_of(again) == genomes_of(RunStore.open(ref_dir))


def test_track_car_growing_run_is_the_same_with_three_workers(tmp_path: Path) -> None:
    config = small(max_generations=5, grow_every=2, trials=1, popsize=4)
    a, rows_a = train(TEXTS["track_car"], config, keep_all_store(tmp_path / "a", TEXTS["track_car"]))
    b, rows_b = train(TEXTS["track_car"], config, keep_all_store(tmp_path / "b", TEXTS["track_car"]), workers=3,
                      record=True)
    assert strip(rows_a) == strip(rows_b) and grown_rows(rows_a) == [(2, 2, 22), (4, 4, 44)]
    assert a.state.champion.genome is not None
    assert a.state.champion.genome.tobytes() == b.state.champion.genome.tobytes()
    assert a.learner.best().tobytes() == b.learner.best().tobytes()


# ---------------------------------------------------------------------------
# T41 자동 재시작·예선
# ---------------------------------------------------------------------------
VALIDATION_SEEDS = {seeds.derive(seeds.TAG_VALIDATION, k) for k in range(64)}


class ConstantValidation:
    """착륙선을 돌리지 않는 평가기: 학습 점수는 유전자의 가짜 점수, 검증·비교 점수는 늘 같은 값(정체가 확실히 생김)."""

    def __init__(self) -> None:
        self.blocks = len(LANDER_SPEC.file.scoring.blocks)

    def evaluate(self, env_version: str, env_text: str, layout: Any, genomes: np.ndarray, pairs: list[tuple[int, Any]],
                 record: list[bool] | None = None) -> EvalResult:
        n = len(pairs)
        if pairs[0][1].seed in VALIDATION_SEEDS:
            scores = np.full(n, 5.0)
        else:
            per = fake_scores(genomes)
            scores = np.array([per[i] for i, _ in pairs])
        return EvalResult(scores=scores, steps=np.ones(n, dtype=np.int64), end_events=["timeout"] * n,
                          outcomes=["failure"] * n, blocks=np.zeros((n, self.blocks)), action_abs=np.zeros((n, 3)),
                          weighted=np.zeros((n, self.blocks)))


@pytest.mark.parametrize(("restart_from", "switch", "counts"), [
    ("fresh", SWITCH_RESTART_FRESH, (2,)), ("champion", SWITCH_RESTART_CHAMPION, (4,))])
def test_restart_starts_fresh_from_start_or_from_the_champion_size(restart_from: str, switch: int,
                                                                   counts: tuple[int, ...]) -> None:
    """검증 점수가 늘 같아 48회 뒤 6세대 끝에 재시작. 간격 1이라 첫 검증(2세대)의 챔피언은 뉴런 4개 뇌의 후보다.
    처음 상태에서 → start(2)부터, 챔피언에서 → 챔피언의 켜진 수(4)부터 다시 자란다."""
    config = small(grow_every=1, restart_patience=48, restart_from=restart_from, max_generations=9)
    t = Trainer(LANDER_TEXT, config, ConstantValidation(), None)
    rows = [t.step() for _ in range(6)]
    assert [int(r["switch"]) for r in rows] == [0] * 5 + [switch]
    champ = t.state.champion
    assert champ.generation == 2 and BrainFrame(t.layout).counts_of(champ.genome) == (4,)
    assert isinstance(t.learner, GrowingCMAES) and t.learner.counts == counts and t.learner.growths == 0
    assert t.state.restarts == 1 and t.state.learners == 2
    if restart_from == "champion":
        keep = t.learner.frame.active_coords((4,))
        assert t.learner.best()[keep].tobytes() == champ.genome[keep].tobytes()
    rows += [t.step() for _ in range(3)]
    assert [int(r["brain_neurons"]) for r in rows[6:]] == [counts[0], 2 * counts[0], 4 * counts[0]]


def test_restart_and_qualify_runs_resume_identically(tmp_path: Path) -> None:
    """예선 2출전(출전마다 처음부터 자람) + 자동 재시작(처음·챔피언 번갈아)인 자라는 실행이 모든 체크포인트에서 같게 이어진다."""
    config = small(max_generations=12, checkpoint_every=1, grow_every=2, qualify_entrants=2, qualify_trials=36,
                   restart_patience=24, restart_from="alternate")
    store = keep_all_store(tmp_path / "ref")
    _t, rows = train(LANDER_TEXT, config, store)
    ref = strip(store.read_stats())
    assert [int(r["learner"]) for r in rows[:6]] == [0, 0, 0, 1, 1, 1]  # 출전자 둘
    assert [int(r["brain_neurons"]) for r in rows[:6]] == [2, 2, 4, 2, 2, 4]  # 출전자마다 처음부터 자람
    for g in (2, 3, 4, 6, 7, 9, 11):
        copy = tmp_path / f"from_{g}"
        shutil.copytree(store.root, copy)
        again = RunStore.open(copy, policy=KEEP_ALL)
        with WorkerPool(0) as pool:
            Trainer.resume(again, pool, g, max_generations=12).run()
        assert strip(again.read_stats()) == ref, g
    switches = [int(r["switch"]) for r in rows]
    assert 4 in switches  # 예선 끝(통과자를 저장한 상태에서 다시 만듦)


class RisingValidation(ConstantValidation):
    """검증 점수가 검증할 때마다 1씩 오르는 평가기: 예선에서 늘 뒤 출전자가 통과한다(ConstantValidation은 동점이라 앞 출전자)."""

    def __init__(self) -> None:
        super().__init__()
        self.validations = 0

    def evaluate(self, env_version: str, env_text: str, layout: Any, genomes: np.ndarray, pairs: list[tuple[int, Any]],
                 record: list[bool] | None = None) -> EvalResult:
        res = super().evaluate(env_version, env_text, layout, genomes, pairs, record)
        if pairs[0][1].seed in VALIDATION_SEEDS:
            self.validations += 1
            res.scores[:] = 5.0 + self.validations
        return res


def announced(t: Trainer, row: dict[str, Any]) -> tuple[Any, str | None, Any]:
    """한 세대 뒤 세 곳이 알리는 자람: Trainer.last_growth, 명령줄 한 줄(CLI_TRAIN_GROWN, 없으면 None), 세션 세대 메시지의
    "growth"(없으면 None). 세션은 학습 프로세스 없이 만들어 그 세대의 메시지만 만들게 한다."""
    from evolab.engine.session import Session, SessionRequest

    sent: list[dict[str, Any]] = []
    session = Session(None, None, SessionRequest(kind="resume"))
    session.trainer = t
    session._send = sent.append  # type: ignore[method-assign]
    session._send_generation(row)
    message = next(m for m in sent if m["type"] == "generation")
    assert ("growth" in message) == (message.get("growth") is not None)
    return t.last_growth, commands._grown_text(t, row), message.get("growth")


NOT_ANNOUNCED = (None, None, None)


def grew(gen: int, old: int, new: int) -> tuple[Any, str, dict[str, int]]:
    """착륙선(입력 9, 출력 3)에서 gen세대 끝에 뉴런 old → new로 자랐을 때 announced()가 돌려줄 값."""
    return ((old, new, 12 * old, 12 * new), S.CLI_TRAIN_GROWN.format(gen=gen, old=old, new=new),
            {"old": old, "new": new, "old_links": 12 * old, "new_links": 12 * new})


@pytest.mark.parametrize(("restart_from", "switch"), [("fresh", SWITCH_RESTART_FRESH),
                                                      ("champion", SWITCH_RESTART_CHAMPION)])
def test_growth_thrown_away_by_a_restart_is_not_announced(restart_from: str, switch: int) -> None:
    """간격 3, 검증 점수가 늘 같아 6세대 끝에 자동 재시작: 3세대 끝의 자람(2 → 4)은 알리고, 6세대 끝의 자람(4 → 8)은 같은 세대 끝에
    그 학습 방법이 버려지므로 어디에도 알리지 않는다(7세대는 뉴런 2개: 처음 상태 또는 2세대에 뽑힌 뉴런 2개짜리 챔피언에서).
    통계 열 grow는 그 세대 끝에 자란 사실 그대로 1이다(대시보드 세로선 둘: 자람과 자동 재시작)."""
    config = small(grow_every=3, restart_patience=48, restart_from=restart_from, max_generations=9)
    t = Trainer(LANDER_TEXT, config, ConstantValidation(), None)
    rows, told = [], []
    for _ in range(8):
        row = t.step()
        rows.append(row)
        told.append(announced(t, row))
    assert [(int(r["switch"]), int(r["grow"])) for r in rows] == [(0, 0), (0, 0), (0, 1), (0, 0), (0, 0), (switch, 1),
                                                                   (0, 0), (0, 0)]
    assert [int(r["brain_neurons"]) for r in rows] == [2, 2, 2, 4, 4, 4, 2, 2]
    assert told == [NOT_ANNOUNCED, NOT_ANNOUNCED, grew(3, 2, 4), NOT_ANNOUNCED, NOT_ANNOUNCED, NOT_ANNOUNCED,
                    NOT_ANNOUNCED, NOT_ANNOUNCED]
    assert isinstance(t.learner, GrowingCMAES) and t.state.restarts == 1


def test_growth_at_the_end_of_qualifying_is_announced_only_when_that_entrant_goes_on() -> None:
    """예선 2출전, 출전마다 2세대(시험 24회), 간격 2: 출전자마다 마지막 세대 끝에 자란다.
    2세대 끝(다음 출전으로): 자란 출전자 0은 보관되고 출전자 1이 뉴런 2개로 시작한다 → 알리지 않는다.
    4세대 끝(예선 끝): 방금 자란 출전자 1이 통과하면(검증 점수가 오르는 평가기) 그 자란 상태에서 이어 가므로 알리고, 출전자 0이
    통과하면(동점은 앞 출전자) 출전자 1의 자람은 버려지므로 알리지 않는다. 누가 통과하든 5세대는 뉴런 4개(각자 자란 상태)."""
    config = small(grow_every=2, qualify_entrants=2, qualify_trials=24, max_generations=6)
    config.validate()
    for evaluator, winner in ((RisingValidation(), 1), (ConstantValidation(), 0)):
        t = Trainer(LANDER_TEXT, config, evaluator, None)
        rows, told = [], []
        for _ in range(5):
            row = t.step()
            rows.append(row)
            told.append(announced(t, row))
        assert [int(r["switch"]) for r in rows] == [0, SWITCH_QUALIFY_NEXT, 0, SWITCH_QUALIFY_PICK, 0]
        assert [int(r["grow"]) for r in rows] == [0, 1, 0, 1, 0]
        assert [int(r["learner"]) for r in rows] == [0, 0, 1, 1, winner]
        assert [int(r["brain_neurons"]) for r in rows] == [2, 2, 2, 2, 4]
        assert told[:3] == [NOT_ANNOUNCED] * 3 and told[4] == NOT_ANNOUNCED
        assert told[3] == (grew(4, 2, 4) if winner == 1 else NOT_ANNOUNCED), winner


# ---------------------------------------------------------------------------
# T40 크기 고정 실행이 v6.0과 같다
# ---------------------------------------------------------------------------
FIXED_CASES = [
    ("lander", {"method": "cmaes"}),
    ("lander", {"method": "cmaes", "restart_patience": 24, "restart_from": "alternate", "qualify_entrants": 2,
                "qualify_trials": 24}),
    ("lander", {"method": "ga"}),
    ("lander", {"method": "sa", "sa_neighbors": 5}),
    ("hill_car", {"method": "cmaes", "trials": 1, "popsize": 4, "diagonal_threshold": 100}),
]


@pytest.mark.parametrize(("env", "over"), FIXED_CASES, ids=[f"{e}-{o['method']}-{k}" for k, (e, o) in
                                                            enumerate(FIXED_CASES)])
def test_fixed_size_runs_are_byte_identical_to_v60(tmp_path: Path, env: str, over: dict[str, Any]) -> None:
    v60 = load_v60()
    text = TEXTS[env]
    spec = compile_env(text)
    values = {"popsize": 6, "trials": 2, "validate_every": 24, "checkpoint_every": 2, "run_seed": 7, "max_generations": 6,
              **FAST, **over}
    new_cfg = TrainConfig.for_new_run(spec, **values)
    old_cfg = v60.trainer.TrainConfig.for_new_run(spec, **values)
    assert new_cfg.to_dict() == old_cfg.to_dict() and pickle.dumps(new_cfg.to_dict()) == pickle.dumps(old_cfg.to_dict())
    stores = {}
    for tag, module, cfg in (("v60", v60.trainer, old_cfg), ("v61", trainer_mod, new_cfg)):
        store = create_run(tmp_path / tag, text, spec, cfg, 0)
        train(text, cfg, store, trainer_cls=module.Trainer)
        stores[tag] = store
    old, new = stores["v60"], stores["v61"]
    assert old.read_run_info()["config"] == new.read_run_info()["config"]
    assert old.stats_columns() == new.stats_columns() and not set(GROW_COLUMNS) & set(new.stats_columns() or [])
    assert strip(old.read_stats()) == strip(new.read_stats()) and len(new.read_stats()) == 6
    assert champion_files(old) == champion_files(new)
    assert old.checkpoints() == new.checkpoints() and new.checkpoints()
    for g in new.checkpoints():
        a, b = old.load_checkpoint(g)[1], new.load_checkpoint(g)[1]
        assert pickle.dumps(a["config"]) == pickle.dumps(b["config"])
        assert sorted(a["learner"]) == sorted(b["learner"]) and a["learner"]["config"] == b["learner"]["config"]
        assert "stds" not in b["learner"]["config"]
        assert canon(a) == canon(b), g
    if over["method"] == "cmaes":
        # 서로의 체크포인트를 이어 간다: (만든 프로그램, 이어 간 프로그램) 네 짝. v6.0이 만든 6세대 체크포인트를 v6.1이, v6.1이 만든
        # 것을 v6.0이 8세대까지 이어 가도 통계·챔피언·세대 대표·마지막 체크포인트가 모두 같다(실행 폴더를 복사해서 이어 감)
        resumed: dict[tuple[str, str], RunStore] = {}
        with WorkerPool(0) as pool:
            for made, store in stores.items():
                for prog, module in (("v60", v60.trainer), ("v61", trainer_mod)):
                    copy = tmp_path / f"{made}_resumed_by_{prog}"
                    shutil.copytree(store.root, copy)
                    again = RunStore.open(copy)
                    t = module.Trainer.resume(again, pool, None, max_generations=8)
                    assert type(t).__module__ == module.__name__ and t.state.generation == 6
                    t.run()
                    resumed[made, prog] = again
        first = resumed["v60", "v60"]
        tail = strip(first.read_stats())
        assert len(tail) == 8 and tail[:6] == strip(old.read_stats())
        for pair, again in resumed.items():
            assert strip(again.read_stats()) == tail, pair
            assert champion_files(again) == champion_files(first), pair
            assert again.checkpoints()[-1] == 8, pair
            assert canon(again.load_checkpoint(8)[1]) == canon(first.load_checkpoint(8)[1]), pair


# ---------------------------------------------------------------------------
# T44 옛 프로그램(v6.0 사본) 보호
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def grow_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """v6.1이 만든 작은 착륙선 자라는 실행(6세대, 2·4·6세대 끝에 자람, 체크포인트 2·4·6)과 그 재현 묶음."""
    base = tmp_path_factory.mktemp("v61run")  # 경로에 "grow"가 없게: 재현 거절 글("grow")을 경로와 헷갈리지 않는다
    config = TrainConfig.for_new_run(LANDER_SPEC, **dataclasses.asdict(small(grow_every=2, max_generations=6)))
    store = create_run(base / "runs", LANDER_TEXT, LANDER_SPEC, config, 0)
    t, rows = train(LANDER_TEXT, config, store)
    store.append_event("stop", t.state.generation, reason="max_generations")
    assert grown_rows(rows) == [(2, 2, 24), (4, 4, 48), (6, 8, 96)] and store.checkpoints() == [2, 4, 6]
    bundle = export_bundle(store.root, base / "bundle.zip")
    return store.root, bundle


def _snapshot_of(run: Path) -> dict[str, bytes]:
    return {str(p.relative_to(run)): p.read_bytes() for p in sorted(run.rglob("*")) if p.is_file()}


def test_v60_cli_resume_refuses_a_growing_run(tmp_path: Path, grow_run: tuple[Path, Path],
                                              capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    v60 = load_v60()
    run = tmp_path / "run"
    shutil.copytree(grow_run[0], run)
    before = _snapshot_of(run)

    def no_pool(*_a: Any, **_k: Any) -> None:
        raise AssertionError("v6.0 started workers")

    monkeypatch.setattr(v60.commands, "WorkerPool", no_pool)
    args = cli._build_parser().parse_args(["train", "--resume", str(run), "--max-generations", "9", "--workers", "0"])
    capsys.readouterr()
    assert v60.commands.train_command(args) == 2
    err = capsys.readouterr().err
    assert S.TRAIN_CANNOT_START.format(reason=_invalid("method", "grow")) in err
    assert _snapshot_of(run) == before  # 아무것도 고치지 않음(체크포인트 뒤를 잘라 내지도 않음)


def test_v60_session_resume_refuses_a_growing_run(tmp_path: Path, grow_run: tuple[Path, Path]) -> None:
    v60 = load_v60()
    run = tmp_path / "run"
    shutil.copytree(grow_run[0], run)
    before = _snapshot_of(run)
    msgs = run_session(v60.session.Session, v60.session.SessionRequest(kind="resume", run_dir=str(run), workers=0))
    errors = [m for m in msgs if m["type"] == "error"]
    assert len(errors) == 1 and errors[0]["fatal"]
    assert errors[0]["message"] == S.TRAIN_CANNOT_START.format(reason="grow")
    assert msgs[-1]["type"] == "closed" and not [m for m in msgs if m["type"] in ("started", "generation")]
    assert _snapshot_of(run) == before
    # 같은 요청(같은 폴더, 같은 칸)을 v6.1 세션은 받는다: 6세대(마지막 체크포인트, 세 번째 자람 직후)에서 시작하고 오류가 없다.
    # 이 실행은 최대 6세대라 더 돌 세대 없이 「끝남」으로 간다(자람을 지나 이어 가는 것은 아래 테스트)
    from evolab.engine.session import Session, SessionRequest

    msgs = drive_session(Session, SessionRequest(kind="resume", run_dir=str(run), workers=0))
    assert [m["type"] for m in msgs if m["type"] in ("started", "error", "generation", "closed")] == ["started", "closed"]
    started = msgs[0]
    assert started["type"] == "started" and started["generation"] == 6 and started["config"]["method"] == "grow"
    assert (started["config"]["grow_start"], started["config"]["grow_every"]) == (2, 2)
    assert [(r["generation"], r["brain_neurons"], r["grow"]) for r in started["stats"]][-2:] == [("5", "8", "0"),
                                                                                                 ("6", "8", "1")]
    assert [m["state"] for m in msgs if m["type"] == "state"][-1] == "finished"
    with WorkerPool(0) as pool:
        t = Trainer.resume(RunStore.open(run), pool, None)
        assert isinstance(t.learner, GrowingCMAES) and t.learner.counts == (16,)  # 6세대 끝에 자란 뒤 상태


def test_v61_session_resumes_a_growing_run_across_a_growth(tmp_path: Path) -> None:
    """v6.1 세션의 「이어 하기」가 자라는 실행을 진짜로 이어 간다: 7세대 실행(간격 2: 2·4·6세대 끝에 자람)을 4세대 체크포인트(둘째
    자람 직후, 뉴런 8개)에서 이어 가면 시작 메시지는 4세대, 세대 메시지는 5·6·7, "growth"는 6세대에만(8 → 16), 오류가 없고
    통계가 끊지 않은 실행과 같다."""
    from evolab.engine.session import Session, SessionRequest

    config = small(grow_every=2, max_generations=7, checkpoint_every=1)
    store = keep_all_store(tmp_path / "ref", config=config)
    train(LANDER_TEXT, config, store)
    ref = strip(store.read_stats())
    assert grown_rows(store.read_stats()) == [(2, 2, 24), (4, 4, 48), (6, 8, 96)] and store.checkpoints() == list(range(1, 8))
    copy = tmp_path / "copy"
    shutil.copytree(store.root, copy)
    msgs = drive_session(Session, SessionRequest(kind="resume", run_dir=str(copy), generation=4, workers=0,
                                                 spectate=False))
    assert not [m for m in msgs if m["type"] == "error"]
    started = [m for m in msgs if m["type"] == "started"]
    assert len(started) == 1 and started[0]["generation"] == 4 and started[0]["config"]["method"] == "grow"
    assert len(started[0]["stats"]) == 4  # 4세대 뒤의 줄은 잘라 내고 다시 돈다
    gens = [m for m in msgs if m["type"] == "generation"]
    assert [int(m["generation"]) for m in gens] == [5, 6, 7]
    assert [m.get("growth") for m in gens] == [None, {"old": 8, "new": 16, "old_links": 96, "new_links": 192}, None]
    assert [int(m["row"]["brain_neurons"]) for m in gens] == [8, 8, 16]
    assert msgs[-1]["type"] == "closed" and [m["state"] for m in msgs if m["type"] == "state"][-1] == "finished"
    again = RunStore.open(copy)
    assert strip(again.read_stats()) == ref and genomes_of(again) == genomes_of(store)
    assert [e["kind"] for e in again.read_events()][-2:] == ["resume", "stop"]


def test_v60_resume_from_an_earlier_checkpoint_leaves_a_branch_folder_it_cannot_continue(
        tmp_path: Path, grow_run: tuple[Path, Path]) -> None:
    """옛 프로그램의 기록 화면에서 「이어하기」로 최신이 아닌 체크포인트를 고르면 먼저 분기 폴더를 만들고(runs/catalog.branch_run:
    v6.0과 v6.1이 같은 파일) 그 폴더로 세션을 연다. v6.0 세션은 「grow」로 거절하고 분기 폴더는 남는다(원래 실행은 그대로).
    도움말 「자라는 뇌」 절 끝 문단이 이것과 「복제」(옛 프로그램은 크기 고정 실행으로 시작)를 적는다. 남은 폴더는 v6.1이 이어 간다."""
    v60 = load_v60()
    run = tmp_path / "run"
    shutil.copytree(grow_run[0], run)
    before = _snapshot_of(run)
    child = branch_run(run, 4, tmp_path / "분기")
    msgs = run_session(v60.session.Session, v60.session.SessionRequest(kind="resume", run_dir=str(child), workers=0))
    errors = [m for m in msgs if m["type"] == "error"]
    assert len(errors) == 1 and errors[0]["fatal"] and errors[0]["message"] == S.TRAIN_CANNOT_START.format(reason="grow")
    assert not [m for m in msgs if m["type"] in ("started", "generation")]
    assert _snapshot_of(run) == before
    left = RunStore.open(child)
    assert left.checkpoints() == [4] and len(left.read_stats()) == 4
    assert left.read_run_info()["config"]["method"] == "grow"
    with WorkerPool(0) as pool:
        Trainer.resume(left, pool, None, max_generations=6).run()
    assert strip(left.read_stats()) == strip(RunStore.open(run).read_stats())
    section = S.HELP_PAGES["lab"][1].split("## 자라는 뇌", 1)[1]
    assert "{{REC_CLONE}}" in section and "{{REC_RESUME}}" in section and "v6.1" in section


def test_v60_reproduce_refuses_and_judge_open_replay_still_work(tmp_path: Path, grow_run: tuple[Path, Path],
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
    v60 = load_v60()
    run_dir, bundle = grow_run
    parser = cli._build_parser()
    capsys.readouterr()
    assert v60.commands.reproduce_command(parser.parse_args(["reproduce", str(bundle), "--out", str(tmp_path / "re"),
                                                             "--workers", "0"])) == 2
    # v6.0 make_learner의 ValueError("grow") 그대로 한 줄. 「묶음을 읽을 수 없습니다: <경로>」 같은 다른 거절(경로에 grow가 들어
    # 있으면 "grow" in err로는 구별되지 않는다)이 아니다
    assert capsys.readouterr().err.strip().splitlines() == ["grow"]
    assert "grow" not in str(bundle) and "grow" not in str(tmp_path)
    left = [p for p in (tmp_path / "re").iterdir() if p.is_dir()]  # 설계 2.4: 빈 실행 폴더 하나가 남는다(통계 줄 없음)
    assert len(left) == 1 and RunStore.open(left[0]).read_run_info()["config"]["method"] == "grow"
    assert RunStore.open(left[0]).read_stats() == []
    # 판정: v6.0 명령이 v6.1 명령과 같은 출력을 낸다
    from evolab.engine import commands as new_commands

    judge = parser.parse_args(["judge", str(run_dir), "--seeds", "6", "--workers", "0"])
    assert v60.commands.judge_command(judge) == 0
    old_out = capsys.readouterr().out
    assert new_commands.judge_command(judge) == 0
    assert capsys.readouterr().out == old_out and old_out
    # 기록 열기와 재생: 유전자가 틀 길이라 v6.0의 layout_for(spec)·replay 그대로
    store = RunStore.open(run_dir)
    rows = store.read_stats()
    assert rows[-1]["generation"] == "6"
    spec = compile_env(store.env_text(store.read_run_info()["env_versions"][0]["id"]))
    layout = layout_for(spec)
    for kind in ("champion", "representative"):
        g = store.champion_generations(kind)[-1]
        genome = store.load_champion(g, kind)["genome"]
        assert genome.shape == (layout.dim,)
        rep = replay(spec, layout, genome, seeds.display_trial())
        res = evaluate(spec, layout, genome[None, :], [(0, seeds.display_trial())])
        assert rep.result.score == float(res.scores[0])


# ---------------------------------------------------------------------------
# 세션: 관전 방식·고속에서 같은 결과, 자란 세대만 "growth" (T36 일부)
# ---------------------------------------------------------------------------
def test_session_growth_messages_and_pace_do_not_change_results(qtbot: Any, tmp_path: Path) -> None:
    from PySide6.QtCore import QTimer

    from evolab.app.lab.client import TrainingClient
    from evolab.engine.session import PACE_EVERY, PACE_SKIP, SessionRequest

    config = small(max_generations=5, grow_every=2).to_dict()
    results = {}
    for mode, over in (("skip", {"pace": PACE_SKIP}), ("every", {"pace": PACE_EVERY}), ("fast", {"spectate": False})):
        client = TrainingClient()
        msgs: list[tuple[str, dict[str, Any]]] = []
        client.started.connect(lambda m, out=msgs: out.append(("started", m)))
        client.generation.connect(lambda m, out=msgs: out.append(("generation", m)))
        client.state_changed.connect(lambda m, out=msgs: out.append(("state", m)))
        client.failed.connect(lambda text, fatal, out=msgs: out.append(("failed", {"message": text})))
        if mode == "every":  # 가짜 화면: 세대마다 0.3초 늦게 다 봤다고 알린다
            client.generation.connect(lambda m, c=client: QTimer.singleShot(
                300, lambda g=int(m["generation"]): c.send({"type": "played", "generation": g})))
        req = SessionRequest(**{"kind": "new", "env_text": LANDER_TEXT, "runs_dir": str(tmp_path / mode), "workers": 1,
                                "spectate": True, "config": config, **over})
        client.start(req)
        try:
            qtbot.waitUntil(lambda out=msgs: any(k == "state" and m["state"] == "finished" for k, m in out)
                            or any(k == "failed" for k, _ in out), timeout=240000)
            assert not [m for k, m in msgs if k == "failed"]
            gens = [m for k, m in msgs if k == "generation"]
            assert [int(m["generation"]) for m in gens] == [1, 2, 3, 4, 5]
            assert [m.get("growth") for m in gens] == [
                None, {"old": 2, "new": 4, "old_links": 24, "new_links": 48},
                None, {"old": 4, "new": 8, "old_links": 48, "new_links": 96}, None]
            assert ("growth" in gens[1]) and ("growth" not in gens[0])
            run_dir = Path(next(m for k, m in msgs if k == "started")["run_dir"])
        finally:
            client.shutdown()
        results[mode] = run_dir
    stores = {mode: RunStore.open(d) for mode, d in results.items()}
    stats = {mode: strip(st.read_stats()) for mode, st in stores.items()}
    assert stats["skip"] == stats["every"] == stats["fast"] and len(stats["skip"]) == 5
    assert champion_files(stores["skip"]) == champion_files(stores["every"]) == champion_files(stores["fast"])
    for g in stores["skip"].checkpoints():
        a, b, c = (canon(stores[m].load_checkpoint(g)[1]) for m in ("skip", "every", "fast"))
        assert a == b == c, g
    info = stores["skip"].read_run_info()
    assert info["config"]["method"] == "grow" and info["config"]["grow_every"] == 2
    assert S.METHOD_NAMES["grow"] in stores["skip"].root.name


# ---------------------------------------------------------------------------
# 자체 점검 「자라는 뇌」
# ---------------------------------------------------------------------------
def test_selftest_grow_check_passes_and_is_listed() -> None:
    for gui in (True, False):
        names = [n for n, _ in selftest.selftest_checks(2, check_gui=gui)]
        k = names.index(S.SELFTEST_CHECK_GROW)
        assert names[k - 1] == S.SELFTEST_CHECK_LEARNER
    assert dict(selftest.selftest_checks(2, check_gui=False))[S.SELFTEST_CHECK_GROW] is selftest._check_grow
    r = selftest._run_check(S.SELFTEST_CHECK_GROW, selftest._check_grow)
    assert r.ok, r.detail
    lines = r.detail.splitlines()
    assert lines[0] == S.SELFTEST_GROW_SAME and "2 -> 4 -> 8 -> 16" in lines[1] and "growths=3" in lines[2]
    assert selftest._check_grow() == r.detail  # 두 번 돌려도 같은 체크섬


def test_selftest_grow_check_reports_a_broken_growth(monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.learners import grow as grow_mod

    original = grow_mod.GrowingCMAES._grow_now

    def leaky(self: GrowingCMAES) -> None:  # 새 뉴런의 나가는 가중치를 0으로 두지 않는 잘못된 자람
        original(self)
        self._mean[self.frame.neuron_out(0, self.counts[0] - 1, self.counts)] = 0.3

    monkeypatch.setattr(grow_mod.GrowingCMAES, "_grow_now", leaky)
    r = selftest._run_check(S.SELFTEST_CHECK_GROW, selftest._check_grow)
    assert not r.ok and S.SELFTEST_GROW_DIFF.format(detail="").rstrip() in r.detail


# ---------------------------------------------------------------------------
# S3 (느림): 착륙선 시드 1 자라는 뇌 학습 시험 12,000회
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_s3_lander_growing_run_workers_and_resume_every_three_generations(tmp_path: Path) -> None:
    """설계 8.3 S3: 착륙선 시드 1 method="grow"(새 실행 기본값) 학습 시험 12,000회가 워커 1·3에서 같고, 3세대마다 끊고 이어 간
    실행도 같다(통계 시간 열 빼고, 모든 세대 대표·챔피언 유전자)."""
    config = TrainConfig.for_new_run(LANDER_SPEC, method="grow", run_seed=1, max_evaluations=12000)
    runs = {}
    for workers in (1, 3):
        store = create_run(tmp_path / f"w{workers}", LANDER_TEXT, LANDER_SPEC, config, workers)
        train(LANDER_TEXT, config, store, workers=workers)
        runs[workers] = store
    a, b = runs[1], runs[3]
    rows = strip(a.read_stats())
    assert rows == strip(b.read_stats()) and genomes_of(a) == genomes_of(b)
    assert int(rows[-1]["evaluations"]) >= 12000 and [int(r["generation"]) for r in rows if r["grow"] == "1"] == [10, 20, 30]
    chopped = create_run(tmp_path / "chopped", LANDER_TEXT, LANDER_SPEC, config, 3)
    with WorkerPool(3) as pool:
        t = Trainer(LANDER_TEXT, dataclasses.replace(config, max_generations=3), pool, chopped)
        t.run()
        while t.state.evaluations < 12000:
            t = Trainer.resume(chopped, pool, None, max_generations=t.state.generation + 3, max_evaluations=12000)
            t.run()
    assert strip(chopped.read_stats()) == rows and genomes_of(chopped) == genomes_of(a)

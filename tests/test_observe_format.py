"""v1.2 관찰 기록 형식 (runs/observe.py, 설계 2.2절·D10, 테스트 T13·T14).

세대 대표 npz의 obs_ 키를 읽는 쪽 검사: 잘못된 기록은 예외 없이 (None, 이유), 디스크의 관찰은 같은 세대 통계 줄과
맞을 때만 쓴다. 쓰는 쪽(engine/observe.summarize)은 묶음 A의 tests/test_observe.py가 본다.
"""

from __future__ import annotations

import dataclasses
import math
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evolab import paths
from evolab.engine.evaluate import EvalResult
from evolab.engine.trainer import TrainConfig, Trainer
from evolab.runs import observe as ob
from evolab.runs.observe import GridFrame, Observation, consistent, decode, read_representative, scores_text
from evolab.runs.store import RunStore
from evolab.spec.io import compile_env

DATA = Path(__file__).parent / "data"
DIM, NB, NOUT, P, T = 7, 2, 3, 5, 2
SHAPE = {"dim": DIM, "n_body": NB, "n_out": NOUT}


def make_obs(gen: int = 4, stride: int = 1, *, actions: bool = True, density: bool = True, body: bool = True,
             track: int | None = None) -> Observation:
    """모든 묶음이 있는 올바른 기록(작은 모양: 개체 5, 시험 2, 유전자 7, 몸 2, 출력 3)."""
    rng = np.random.default_rng(gen)
    scores = rng.normal(size=P) * 50.0
    return Observation(
        generation=gen, learner=1, qualifying=False, representative=int(np.argmax(scores)), genome_stride=stride,
        outside=3, trial_seed=12345678901234567890, track=track, scores=scores,
        outcome=rng.integers(0, 3, (P, T)).astype(np.uint8), end=rng.integers(0, 4, (P, T)).astype(np.uint8),
        genomes=rng.normal(size=(P, DIM)).astype(np.float16) if gen % stride == 0 else None,
        body=rng.uniform(0.2, 1.5, (P, NB)) if body else None,
        act_hist=rng.integers(0, 1000, (2, NOUT, ob.ACTION_BINS)).astype(np.uint32) if actions else None,
        act_mean=rng.uniform(-1, 1, (P, NOUT)).astype(np.float32) if actions else None,
        frame=GridFrame(-10.0, 0.0, 1.25, 16, 12) if density else None,
        world_key="0123456789abcdef" if density else None,
        dens_cells=np.array([0, 5, 17, 191], dtype=np.int32) if density else None,
        dens_counts=np.array([3, 1, 7, 2], dtype=np.uint32) if density else None,
    )


def record(**kw: Any) -> dict[str, np.ndarray]:
    return make_obs(**kw).to_arrays()


def assert_same(a: Observation, b: Observation) -> None:
    for name in ("generation", "learner", "qualifying", "representative", "genome_stride", "outside", "trial_seed",
                 "track", "frame", "world_key"):
        assert getattr(a, name) == getattr(b, name), name
    for name in ("scores", "outcome", "end", "genomes", "body", "act_hist", "act_mean", "dens_cells", "dens_counts"):
        x, y = getattr(a, name), getattr(b, name)
        assert (x is None) == (y is None), name
        if x is not None:
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), name


# ---------------------------------------------------------------------------
# 왕복: 쓰기(save_champion) → read_representative → decode
# ---------------------------------------------------------------------------


def test_round_trip_through_save_champion(tmp_path: Path) -> None:
    store = RunStore.create(tmp_path, "run", {})
    obs = make_obs(gen=4, track=2)
    genome = np.linspace(-1, 1, DIM)
    store.save_champion(4, "representative", genome, score=float(obs.scores[obs.representative]),
                        env_version="abcdef0123456789", **obs.to_arrays())
    path = store.root / "champions" / "gen_000004_representative.npz"
    got = read_representative(path)
    assert got is not None
    base, keys = got
    assert set(base) == {"genome", "score", "env_version"} and base["genome"].tobytes() == genome.tobytes()
    assert set(keys) == set(ob.KEYS) and keys[ob.WORLD_KEY].dtype == np.dtype("<U16")
    back, reason = decode(keys, generation=4, **SHAPE)
    assert reason == "" and back is not None
    assert_same(back, obs)
    assert back.track == 2 and back.trial_seed == 12345678901234567890 and back.trials == T and back.population == P
    assert back.has_actions and back.has_density
    assert back.frame == GridFrame(-10.0, 0.0, 1.25, 16, 12) and isinstance(back.frame.nx, int)
    # 다시 쓰면 같은 배열(형, 모양, 바이트)
    again = back.to_arrays()
    assert set(again) == set(keys)
    for k in keys:
        assert again[k].dtype == keys[k].dtype and again[k].tobytes() == keys[k].tobytes(), k
    # 읽은 배열은 읽기 전용이고, 성공 수는 결과 코드 1의 개수
    assert not back.scores.flags.writeable and not back.genomes.flags.writeable  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        back.scores[0] = 0.0
    assert back.successes().tolist() == [int((obs.outcome[i] == 1).sum()) for i in range(P)]
    assert consistent(back, {"generation": "4", "score_best": repr(float(obs.scores.max())),
                             "scores": scores_text(obs.scores)}, base["score"])


def test_round_trip_minimal_record_like_a_fake_evaluator() -> None:
    """관찰을 돌려주지 않는 평가기: 행동·틀·밀도 키가 없다. 유전자 간격 2의 홀수 세대: 유전자도 없다."""
    obs = make_obs(gen=5, stride=2, actions=False, density=False)
    arrays = obs.to_arrays()
    assert set(arrays) == {ob.HEAD, ob.TRIAL, ob.SCORES, ob.OUTCOME, ob.END, ob.BODY}
    back, reason = decode(arrays, **SHAPE)
    assert reason == "" and back is not None
    assert back.genomes is None and back.act_hist is None and back.frame is None and back.dens_cells is None
    assert not back.has_actions and not back.has_density and back.track is None
    # 몸 유전자가 없는 환경: obs_body가 없거나 (P, 0)이면 body는 None
    nobody = make_obs(gen=6, body=False).to_arrays()
    for arrays in (nobody, {**nobody, ob.BODY: np.zeros((P, 0))}):
        back, reason = decode(arrays, dim=DIM, n_body=0, n_out=NOUT)
        assert reason == "" and back is not None and back.body is None
    # 지나간 칸이 하나도 없어도(모두 틀 밖) 밀도 묶음은 온전하다
    empty = {**record(), ob.DENS_CELLS: np.zeros(0, np.int32), ob.DENS_COUNTS: np.zeros(0, np.uint32)}
    back, reason = decode(empty, **SHAPE)
    assert reason == "" and back is not None and back.dens_cells.shape == (0,)  # type: ignore[union-attr]
    # 묶음을 반만 채운 관찰은 쓸 수 없다(형식은 묶음째 있거나 없다)
    for half in ({"act_mean": None}, {"world_key": None}, {"frame": None, "world_key": None, "dens_counts": None}):
        with pytest.raises(ValueError):
            dataclasses.replace(make_obs(), **half).to_arrays()


def test_files_without_observation_keys_are_not_errors() -> None:
    """v1.0·v1.1 세대 대표 파일(실제 자료): 기본 키만 있고 decode는 (None, REASON_NONE)."""
    for path in (DATA / "v10_run_ga" / "run" / "champions" / "gen_000001_representative.npz",
                 DATA / "v11_run_restart" / "run" / "champions" / "gen_000008_representative.npz"):
        got = read_representative(path)
        assert got is not None, path
        base, keys = got
        assert set(base) == {"genome", "score", "env_version"} and keys == {}
        assert decode(keys, dim=211, n_body=0, n_out=3) == (None, ob.REASON_NONE)
        assert decode(base, dim=211, n_body=0, n_out=3) == (None, ob.REASON_NONE)  # 기본 키가 섞여도 같음


# ---------------------------------------------------------------------------
# T13: 잘못된 기록
# ---------------------------------------------------------------------------


def _set(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    return lambda a: a.__setitem__(key, value)


def _drop(*keys: str) -> Callable[[dict[str, Any]], None]:
    def edit(a: dict[str, Any]) -> None:
        for k in keys:
            del a[k]
    return edit


def _head(index: int, value: int) -> Callable[[dict[str, Any]], None]:
    def edit(a: dict[str, Any]) -> None:
        h = a[ob.HEAD].copy()
        h[index] = value
        a[ob.HEAD] = h
    return edit


def _frame(index: int, value: float) -> Callable[[dict[str, Any]], None]:
    def edit(a: dict[str, Any]) -> None:
        f = a[ob.FRAME].copy()
        f[index] = value
        a[ob.FRAME] = f
    return edit


BAD: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
    ("format 2 (newer)", _head(0, 2)),
    ("format 0", _head(0, 0)),
    ("negative generation", _head(1, -1)),
    ("qualifying 2", _head(3, 2)),
    ("representative out of range", _head(4, P)),
    ("stride 0", _head(5, 0)),
    ("negative outside", _head(6, -1)),
    ("head says 3 trials", _head(7, 3)),
    ("head wrong length", _set(ob.HEAD, np.zeros(7, np.int64))),
    ("head int32", lambda a: a.__setitem__(ob.HEAD, a[ob.HEAD].astype(np.int32))),
    ("head is a list", lambda a: a.__setitem__(ob.HEAD, a[ob.HEAD].tolist())),
    ("missing scores", _drop(ob.SCORES)),
    ("scores float32", lambda a: a.__setitem__(ob.SCORES, a[ob.SCORES].astype(np.float32))),
    ("scores 2-D", lambda a: a.__setitem__(ob.SCORES, a[ob.SCORES].reshape(1, P))),
    ("empty population", _set(ob.SCORES, np.zeros(0))),
    ("trial wrong dtype", lambda a: a.__setitem__(ob.TRIAL, a[ob.TRIAL].astype(np.int64))),
    ("outcome P mismatch", _set(ob.OUTCOME, np.zeros((P + 1, T), np.uint8))),
    ("outcome T mismatch", _set(ob.OUTCOME, np.zeros((P, T + 1), np.uint8))),
    ("outcome int64", lambda a: a.__setitem__(ob.OUTCOME, a[ob.OUTCOME].astype(np.int64))),
    ("outcome unknown code", _set(ob.OUTCOME, np.full((P, T), 3, np.uint8))),
    ("end T mismatch", _set(ob.END, np.zeros((P, T + 1), np.uint8))),
    ("genomes D mismatch", _set(ob.GENOMES, np.zeros((P, DIM + 1), np.float16))),
    ("genomes P mismatch", _set(ob.GENOMES, np.zeros((P - 1, DIM), np.float16))),
    ("genomes float32", lambda a: a.__setitem__(ob.GENOMES, a[ob.GENOMES].astype(np.float32))),
    ("genomes inf", _set(ob.GENOMES, np.full((P, DIM), np.inf, np.float16))),
    ("genomes missing at stride", _drop(ob.GENOMES)),
    ("genomes off stride", _head(5, 3)),  # 4세대, 간격 3: 유전자가 있으면 안 됨
    ("body NB mismatch", _set(ob.BODY, np.zeros((P, NB + 1)))),
    ("body P mismatch", _set(ob.BODY, np.zeros((P - 1, NB)))),
    ("body missing", _drop(ob.BODY)),
    ("act_hist U mismatch", _set(ob.ACT_HIST, np.zeros((2, NOUT + 1, 20), np.uint32))),
    ("act_hist 19 bins", _set(ob.ACT_HIST, np.zeros((2, NOUT, 19), np.uint32))),
    ("act_hist int64", lambda a: a.__setitem__(ob.ACT_HIST, a[ob.ACT_HIST].astype(np.int64))),
    ("act_mean U mismatch", _set(ob.ACT_MEAN, np.zeros((P, NOUT + 1), np.float32))),
    ("act_mean P mismatch", _set(ob.ACT_MEAN, np.zeros((P + 1, NOUT), np.float32))),
    ("act_hist without act_mean", _drop(ob.ACT_MEAN)),
    ("frame without density", _drop(ob.DENS_CELLS, ob.DENS_COUNTS)),
    ("density without world key", _drop(ob.WORLD_KEY)),
    ("frame NaN", _frame(0, math.nan)),
    ("frame inf cell", _frame(2, math.inf)),
    ("frame zero cell", _frame(2, 0.0)),
    ("frame nx 0", _frame(3, 0.0)),
    ("frame ny 5000", _frame(4, 5000.0)),
    ("frame nx not whole", _frame(3, 16.5)),
    ("frame wrong length", _set(ob.FRAME, np.zeros(4))),
    ("world key empty", _set(ob.WORLD_KEY, np.array(""))),
    ("world key bytes", _set(ob.WORLD_KEY, np.array(b"0123456789abcdef"))),
    ("cells out of range", _set(ob.DENS_CELLS, np.array([0, 5, 17, 16 * 12], np.int32))),
    ("cells negative", _set(ob.DENS_CELLS, np.array([-1, 5, 17, 191], np.int32))),
    ("cells descending", _set(ob.DENS_CELLS, np.array([0, 17, 5, 191], np.int32))),
    ("cells repeated", _set(ob.DENS_CELLS, np.array([0, 5, 5, 191], np.int32))),
    ("cells int64", lambda a: a.__setitem__(ob.DENS_CELLS, a[ob.DENS_CELLS].astype(np.int64))),
    ("cells 0-D", _set(ob.DENS_CELLS, np.array(3, np.int32))),
    ("counts length", _set(ob.DENS_COUNTS, np.array([3, 1, 7], np.uint32))),
    ("counts zero", _set(ob.DENS_COUNTS, np.array([3, 0, 7, 2], np.uint32))),
]


@pytest.mark.parametrize(("name", "edit"), BAD, ids=[n for n, _ in BAD])
def test_decode_rejects_bad_records(name: str, edit: Callable[[dict[str, Any]], None]) -> None:
    """T13: 하나라도 어기면 (None, 이유)이고 예외를 내지 않는다. 고치지 않은 기록은 통과한다."""
    good = record()
    assert decode(good, generation=4, **SHAPE)[0] is not None
    bad: dict[str, Any] = dict(good)
    edit(bad)
    obs, reason = decode(bad, generation=4, **SHAPE)
    assert obs is None and reason and reason != ob.REASON_NONE, name
    if name == "format 2 (newer)":
        assert reason == ob.REASON_NEWER


def test_decode_checks_the_layout_and_file_generation() -> None:
    good = record()
    for shape in ({**SHAPE, "dim": DIM + 1}, {**SHAPE, "n_body": NB - 1}, {**SHAPE, "n_out": NOUT - 1}):
        obs, reason = decode(good, **shape)
        assert obs is None and reason, shape
    obs, reason = decode(good, generation=5, **SHAPE)  # 파일 이름은 5세대인데 기록은 4세대
    assert obs is None and "generation" in reason
    with pytest.raises(ValueError):  # 인수 잘못은 부르는 쪽의 결함이라 알린다
        decode(good, dim=0, n_body=0, n_out=1)
    with pytest.raises(ValueError):
        decode(good, dim=3, n_body=4, n_out=1)


def test_only_files_with_observation_keys_are_compressed(tmp_path: Path) -> None:
    """관찰 기록이 든 세대 대표만 압축(크기 목표 M8), 챔피언과 관찰 없는 파일은 v1.1과 같은 무압축. 둘 다 그대로 읽힌다."""
    store = RunStore.create(tmp_path, "run", {})
    obs = make_obs()
    store.save_champion(4, "representative", np.ones(DIM), score=1.0, env_version="x", **obs.to_arrays())
    store.save_champion(4, "champion", np.ones(DIM), score=1.0, env_version="x")
    store.save_champion(5, "representative", np.ones(DIM), score=2.0, env_version="x")
    folder = store.root / "champions"
    for name, method in (("gen_000004_representative.npz", zipfile.ZIP_DEFLATED),
                         ("gen_000004_champion.npz", zipfile.ZIP_STORED),
                         ("gen_000005_representative.npz", zipfile.ZIP_STORED)):
        with zipfile.ZipFile(folder / name) as z:
            assert {i.compress_type for i in z.infolist()} == {method}, name
    assert np.array_equal(store.load_champion(4, "representative")["genome"], np.ones(DIM))
    assert store.load_observation(4) is not None and store.load_observation(5) is None


def test_read_representative_rejects_bad_files(tmp_path: Path) -> None:
    """T13(파일): 없음, 0바이트, 잘린 zip, 깨진 내용(CRC), 피클이 든 npz, npz가 아닌 npy는 None(예외 없음)."""
    store = RunStore.create(tmp_path, "run", {})
    obs = make_obs()
    store.save_champion(4, "representative", np.zeros(DIM), score=1.0, env_version="x", **obs.to_arrays())
    good = store.root / "champions" / "gen_000004_representative.npz"
    data = good.read_bytes()
    assert read_representative(good) is not None
    assert read_representative(tmp_path / "missing.npz") is None
    cases = {"empty.npz": b"", "half.npz": data[: len(data) // 2], "garbage.npz": b"not a zip file at all" * 10}
    # obs_scores.npy 멤버의 자료 가운데 바이트 하나를 바꾼다(CRC가 맞지 않음). 관찰 기록이 든 파일은 압축이라
    # 지역 머리 길이(30 + 이름 + 덧붙임)를 읽어 자료가 시작하는 곳을 찾는다
    with zipfile.ZipFile(good) as z:
        info = z.getinfo("obs_scores.npy")
    head = data[info.header_offset: info.header_offset + 30]
    start = info.header_offset + 30 + int.from_bytes(head[26:28], "little") + int.from_bytes(head[28:30], "little")
    flipped = bytearray(data)
    flipped[start + info.compress_size // 2] ^= 0xFF
    cases["crc.npz"] = bytes(flipped)
    for name, content in cases.items():
        (tmp_path / name).write_bytes(content)
        assert read_representative(tmp_path / name) is None, name
    with zipfile.ZipFile(tmp_path / "crc.npz") as z:
        assert z.testzip() is not None  # 정말 깨진 파일이다
    np.savez(tmp_path / "pickled.npz", genome=np.zeros(3), obs_head=np.array([{"a": 1}], dtype=object))
    assert read_representative(tmp_path / "pickled.npz") is None
    np.save(tmp_path / "plain.npy", np.zeros(3))
    assert read_representative(tmp_path / "plain.npy") is None


def test_grid_frame_cells_follow_the_observer_rule() -> None:
    f = GridFrame(-10.0, 0.0, 1.25, 16, 12)
    assert f.rect == (-10.0, 0.0, 20.0, 15.0)
    assert f.cell_at(-10.0, 0.0) == 0 and f.cell_at(-10.0 + 1.25 * 3.5, 1.25 * 2.5) == 2 * 16 + 3
    assert f.cell_at(-10.0 - 0.625, 0.5) is None  # 왼쪽 0.5칸 밖: int(-0.5) == 0으로 0번 칸에 넣지 않는다
    assert f.cell_at(10.0, 0.5) is None and f.cell_at(0.0, 15.0) is None  # 오른쪽·위 끝은 밖
    assert f.cell_center(2 * 16 + 3) == (-10.0 + 3.5 * 1.25, 2.5 * 1.25)
    assert f.as_array().dtype == np.float64 and f.as_array().tolist() == [-10.0, 0.0, 1.25, 16.0, 12.0]


# ---------------------------------------------------------------------------
# T14: 일관성 검사(D10)
# ---------------------------------------------------------------------------


class _ScoreEvaluator:
    """착륙선을 돌리지 않고 유전자와 시드로 점수를 정하는 평가기(이 테스트 전용). 반올림 자리가 다양하게 나오게."""

    def evaluate(self, env_version: str, env_text: str, layout: Any, genomes: np.ndarray,
                 pairs: list[tuple[int, Any]], record: list[bool] | None = None) -> EvalResult:
        n = len(pairs)
        nb = len(compile_env(env_text).file.scoring.blocks)
        scores = np.array([-37.123456789 * float(np.abs(genomes[i]).sum()) + (t.seed % 1000) / 7.0 for i, t in pairs])
        return EvalResult(scores=scores, steps=np.ones(n, dtype=np.int64), end_events=["timeout"] * n,
                          outcomes=["failure"] * n, blocks=np.zeros((n, nb)), action_abs=np.zeros((n, 3)),
                          weighted=np.zeros((n, nb)))


def _trainer_generation(tmp_path: Path) -> tuple[Observation, dict[str, Any], dict[str, str], float]:
    """학습기 한 세대(가짜 평가기): (그 세대의 관찰, 학습 중 통계 줄, stats.csv에서 읽은 줄, 대표 점수)."""
    text = (paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8")
    t = Trainer(text, TrainConfig(popsize=6, trials=2, run_seed=5, validate_every=1000), _ScoreEvaluator(), None)
    t.step()
    row = t.step()
    g = t.last_generation
    assert g is not None
    obs = Observation(generation=g.generation, learner=0, qualifying=False, representative=g.representative,
                      genome_stride=1, outside=0, trial_seed=g.trial0.seed, track=g.trial0.track, scores=g.scores,
                      outcome=np.zeros((len(g.scores), 2), np.uint8), end=np.zeros((len(g.scores), 2), np.uint8),
                      genomes=np.clip(g.population, -ob.FLOAT16_MAX, ob.FLOAT16_MAX).astype(np.float16))
    obs, reason = decode(obs.to_arrays(), dim=t.layout.dim, n_body=0, n_out=3)  # 디스크에서 읽은 것과 같은 모양
    assert reason == "" and obs is not None
    store = RunStore.create(tmp_path, "run", {})
    store.append_stats(row)
    (csv_row,) = store.read_stats()
    return obs, row, csv_row, float(g.scores[g.representative])


def test_consistency_check(tmp_path: Path) -> None:
    """T14: 같은 세대 통계 줄이 있고 최고 점수·scores 칸·대표 점수가 맞을 때만 쓴다."""
    obs, live, row, rep_score = _trainer_generation(tmp_path)
    assert obs.generation == 2 and scores_text(obs.scores) == row["scores"] == live["scores"]
    assert consistent(obs, row, rep_score)  # stats.csv에서 읽은 글자 칸
    assert consistent(obs, live, rep_score)  # 학습 중 받은 값
    assert consistent(obs, row, np.float64(rep_score)) and consistent(obs, row, np.array(rep_score))
    assert not consistent(obs, None, rep_score)  # 통계 줄 없음
    assert not consistent(None, row, rep_score)
    assert not consistent(obs, row, None)
    assert not consistent(obs, {**row, "generation": "3"}, rep_score)  # 다른 세대의 줄
    best = float(row["score_best"])
    assert not consistent(obs, {**row, "score_best": repr(math.nextafter(best, math.inf))}, rep_score)
    assert not consistent(obs, {**row, "score_best": ""}, rep_score)
    scores = row["scores"].split(" ")
    scores[-1] = repr(float(scores[-1]) + 0.0001)
    assert not consistent(obs, {**row, "scores": " ".join(scores)}, rep_score)  # scores 글 다름
    assert not consistent(obs, {k: v for k, v in row.items() if k != "scores"}, rep_score)
    assert not consistent(obs, row, math.nextafter(rep_score, -math.inf))  # 파일의 대표 점수 다름
    assert not consistent(obs, row, "not a number")


def test_scores_text_matches_the_stats_column_rule() -> None:
    v = np.array([-181.63814876965984, 0.1 + 0.2, 1e-05, -1e-05, 12.0, 123456.78904])
    assert scores_text(v) == "-181.6381 0.3 0.0 -0.0 12.0 123456.789"
    assert scores_text(v.tolist()) == scores_text(v) and scores_text(np.zeros(0)) == ""

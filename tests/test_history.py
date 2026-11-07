"""v1.2 「보는 재미」 화면 쪽 기록 (app/lab/history.py): 설계 8.2절 T24~T29.

가짜 관찰 기록은 runs.observe.Observation으로 만들어 세대 대표 파일(RunStore.save_champion)에 학습기와 같은 키로 쓴다.
통계 줄은 stats.csv에서 읽은 것처럼 글자 값(score_best는 repr, scores는 scores_text)이다.
"""

from __future__ import annotations

import functools
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from evolab import paths
from evolab.app.lab import history as hs
from evolab.app.lab.watch_types import WatchMeta
from evolab.policy.genome import GenomeLayout
from evolab.runs import observe as ob
from evolab.runs.store import RunStore
from evolab.spec.io import compile_env

DATA = Path(__file__).parent / "data"
TRACK_KEY = "trackworld000001"


@functools.cache
def meta_for(env: str) -> WatchMeta:
    """내장 환경(lander, hill_car, track_car)의 이름표."""
    spec = compile_env((paths.builtin_env_dir() / f"{env}.evoenv").read_text(encoding="utf-8"))
    return WatchMeta.from_spec(spec)


def fake_generation(gen: int, layout: GenomeLayout, *, pop: int = 6, trials: int = 2, seed: int = 0, stride: int = 1,
                    learner: int = 0, actions: bool = True, frame: ob.GridFrame | None = None,
                    world_key: str | None = None, outside: int = 0) -> tuple[ob.Observation, np.ndarray]:
    """세대 하나의 (관찰 기록, float64 집단 X). 점수는 서로 달라 세대 대표(argmax)가 하나로 정해진다."""
    rng = np.random.default_rng([seed, gen])
    X = rng.normal(0.05 * gen, 0.4, (pop, layout.dim))
    scores = np.round(gen + rng.permutation(pop) * 0.5 + rng.random(pop) * 0.1, 4)
    rep = int(np.argmax(scores))
    U = layout.brain.n_out
    hist = act_mean = None
    if actions:
        hist = rng.integers(0, 40, (2, U, ob.ACTION_BINS)).astype(np.uint32)
        hist[:, :, 0] += 50
        act_mean = rng.uniform(-0.5, 0.9, (pop, U)).astype(np.float32)
    cells = counts = None
    if frame is not None:
        size = frame.nx * frame.ny
        cells = np.unique(rng.integers(0, size, 30)).astype(np.int32)
        counts = rng.integers(1, 9, len(cells)).astype(np.uint32)
    obs = ob.Observation(
        generation=gen, learner=learner, qualifying=False, representative=rep, genome_stride=stride, outside=outside,
        trial_seed=1000 + gen, track=None, scores=scores, outcome=rng.integers(0, 3, (pop, trials)).astype(np.uint8),
        end=rng.integers(0, 4, (pop, trials)).astype(np.uint8),
        genomes=np.clip(X, -ob.FLOAT16_MAX, ob.FLOAT16_MAX).astype(np.float16) if gen % stride == 0 else None,
        body=layout.body_values(X) if layout.n_body else None, act_hist=hist, act_mean=act_mean,
        frame=frame, world_key=world_key if frame is not None else None, dens_cells=cells, dens_counts=counts,
    )
    return obs, X


def stats_row(obs: ob.Observation) -> dict[str, str]:
    """stats.csv에서 읽은 것 같은 그 세대 통계 줄(D10 검사에 쓰는 칸)."""
    return {"generation": str(obs.generation), "score_best": repr(float(obs.scores.max())),
            "scores": ob.scores_text(obs.scores), "learner": str(obs.learner)}


def write_run(root: Path, gens: list[tuple[ob.Observation, np.ndarray]], *, with_obs: bool = True) -> list[dict[str, str]]:
    """세대 대표 파일을 학습기처럼 쓰고 통계 줄 목록을 돌려준다."""
    root.mkdir(parents=True, exist_ok=True)
    store = RunStore(root)
    rows = []
    for obs, X in gens:
        rep = obs.representative
        extra = obs.to_arrays() if with_obs else {}
        store.save_champion(obs.generation, "representative", X[rep], score=float(obs.scores[rep]),
                            env_version="0123456789abcdef", **extra)
        rows.append(stats_row(obs))
    return rows


def entry_of(obs: ob.Observation, X: np.ndarray, layout: GenomeLayout) -> hs.GenEntry:
    entry, _exact = hs.live_entry(obs.generation, obs.to_arrays(), X, obs.representative, layout)
    assert entry is not None
    return entry


# ---------------------------------------------------------------------------
def test_body_series_matches_numpy_and_falls_back_to_representative() -> None:
    """T24: 몸 수치 계열(평균·최소·최대·대표) = numpy 직접 계산. 관찰이 없는 세대는 대표 값."""
    meta = meta_for("hill_car")
    lay = meta.layout
    assert lay.n_body == 5
    h = hs.RunHistory()
    h.set_layout(lay)
    data = [fake_generation(g, lay, pop=7) for g in range(1, 6)]
    for obs, X in data[:4]:
        h.add(entry_of(obs, X, lay))
    obs5, X5 = data[4]
    rep5 = X5[obs5.representative]
    fallback = hs.make_entry(5, None, rep5, float(obs5.scores.max()), lay)
    h.add(fallback)
    bs = h.body_series()
    assert bs is not None and bs.gens.tolist() == [1, 2, 3, 4, 5]
    for i, (obs, X) in enumerate(data[:4]):
        b = lay.body_values(X)
        assert np.array_equal(bs.mean[i], b.mean(axis=0))
        assert np.array_equal(bs.lo[i], b.min(axis=0)) and np.array_equal(bs.hi[i], b.max(axis=0))
        assert np.array_equal(bs.rep[i], b[obs.representative])
    v = lay.body_values(rep5)
    for arr in (bs.mean, bs.lo, bs.hi, bs.rep):
        assert np.array_equal(arr[4], v)
    assert bs.observed.tolist() == [True, True, True, True, False]
    # 몸 유전자가 없는 환경은 계열이 없다
    lander = meta_for("lander").layout
    h2 = hs.RunHistory()
    h2.set_layout(lander)
    o, X = fake_generation(1, lander)
    h2.add(entry_of(o, X, lander))
    assert h2.body_series() is None


def test_action_columns_sum_to_one_and_on_time_rule() -> None:
    """T25: 행동 행렬 열 합 = 1(관찰 있는 세대), 「쓰는 시간」의 쉬는 칸: 추진기 0..1 → 0번, −1..1 → 9·10번."""
    assert hs.rest_bins(0.0, 1.0) == [0]
    assert hs.rest_bins(-1.0, 1.0) == [9, 10]
    assert hs.rest_bins(0.5, 2.0) == [0]  # 0이 범위 밖
    frac = np.zeros(ob.ACTION_BINS)
    frac[0], frac[5], frac[19] = 0.25, 0.5, 0.25
    assert hs.on_fraction(frac, 0.0, 1.0) == pytest.approx(0.75)
    frac2 = np.zeros(ob.ACTION_BINS)
    frac2[9], frac2[10], frac2[0] = 0.3, 0.2, 0.5
    assert hs.on_fraction(frac2, -1.0, 1.0) == pytest.approx(0.5)

    lay = meta_for("lander").layout
    h = hs.RunHistory()
    h.set_layout(lay)
    data = [fake_generation(g, lay) for g in (1, 2, 4)]  # 3세대는 없음, 4세대는 행동 기록 없음
    data[2] = fake_generation(4, lay, actions=False)
    for obs, X in data:
        h.add(entry_of(obs, X, lay))
    for mode, m in (("pop", 0), ("rep", 1)):
        s = h.action_series(1, mode)
        assert s is not None and s.first == 1 and s.frac.shape == (4, ob.ACTION_BINS)
        assert np.allclose(s.frac[:2].sum(axis=1), 1.0)
        assert np.isnan(s.frac[2:]).all()  # 3세대(없음)와 4세대(행동 없음)는 투명
        for i, (obs, _X) in enumerate(data[:2]):
            hist = obs.act_hist[m, 1].astype(np.float64)
            assert np.allclose(s.frac[i], hist / hist.sum())
            want = obs.act_mean[:, 1].astype(np.float64).mean() if m == 0 else float(obs.act_mean[obs.representative, 1])
            assert s.mean[i] == pytest.approx(want)
        assert math.isnan(s.mean[3])


def test_density_merges_only_the_same_world() -> None:
    """T26: 트랙 흉내(세계가 그대로) 기록은 「최근 20세대」가 칸별 합. 열쇠가 다른 세대는 빠지고 합친 수가 맞다.
    착륙선 흉내(세대마다 세계가 다름)는 그 세대만 보이고 DENS_CANNOT_MERGE."""
    lay = meta_for("track_car").layout
    frame = ob.GridFrame(-10.0, -5.0, 0.8, 32, 24)
    h = hs.RunHistory()
    h.set_layout(lay)
    data = {}
    for g in range(1, 26):
        key = "other-world-0020" if g == 20 else TRACK_KEY
        obs, X = fake_generation(g, lay, frame=frame, world_key=key, outside=g)
        data[g] = obs
        h.add(entry_of(obs, X, lay))
    dv = h.density(25, "recent")
    assert dv is not None
    want_gens = [g for g in range(6, 26) if g != 20]
    assert list(dv.merged) == want_gens and not dv.cannot_merge
    dense = np.zeros(frame.nx * frame.ny)
    for g in want_gens:
        np.add.at(dense, data[g].dens_cells.astype(np.int64), data[g].dens_counts.astype(np.float64))
    assert np.array_equal(dv.counts.ravel(), dense)
    assert dv.outside == sum(want_gens)
    one = h.density(25, "one")
    assert one is not None and one.merged == (25,)
    assert h.density(25, "all").merged == tuple(g for g in range(1, 26) if g != 20)
    lone = h.density(20, "recent")
    assert lone is not None and lone.merged == (20,) and lone.cannot_merge

    # 착륙선 흉내: 세계 열쇠가 세대마다 다름
    lander = meta_for("lander").layout
    h2 = hs.RunHistory()
    h2.set_layout(lander)
    for g in range(1, 6):
        obs, X = fake_generation(g, lander, frame=frame, world_key=f"lander-world-{g:04d}")
        h2.add(entry_of(obs, X, lander))
    dv2 = h2.density(5, "recent")
    assert dv2 is not None and dv2.merged == (5,) and dv2.cannot_merge
    assert h2.density(1, "recent").cannot_merge is False  # 창에 그 세대 하나뿐이면 알리지 않는다
    # 궤적 기록이 없는 세대
    o, X = fake_generation(6, lander)
    h2.add(entry_of(o, X, lander))
    assert h2.density(6, "one") is None


def test_gene_pca_accumulates_exactly_and_keeps_signs() -> None:
    """T27: 누적 공분산 = 한 번에 계산(1e-9), 다시 맞춘 뒤 부호가 앞 축 쪽, 두 배 규칙의 맞춤 횟수, float16 입력."""
    rng = np.random.default_rng(5)
    dim = 12
    blocks = [rng.normal(0, 1, (7, dim)) * np.linspace(3, 0.2, dim) + 0.1 * g for g in range(20)]
    blocks16 = [b.astype(np.float16) for b in blocks]
    pca = hs.GenePCA(dim)
    fits_at = []
    for g, b in enumerate(blocks16, start=1):
        pca.add(b)
        if pca.wants_fit():
            pca.fit()
            fits_at.append(g)
    assert fits_at == [1, 2, 4, 8, 16] and pca.fits == 5
    full = np.concatenate([b.astype(np.float64) for b in blocks16])
    mean, axes, explained = hs.covariance_axes(pca.s1, pca.s2, pca.rows)
    assert np.allclose(mean, full.mean(axis=0), atol=1e-9, rtol=0)
    cov = (pca.s2 / pca.rows) - np.outer(mean, mean)
    assert np.allclose(cov, np.cov(full, rowvar=False, bias=True), atol=1e-9, rtol=0)
    w, v = np.linalg.eigh(np.cov(full, rowvar=False, bias=True))
    assert abs(float(axes[0] @ v[:, -1])) == pytest.approx(1.0, abs=1e-9)
    assert explained[0] == pytest.approx(w[-1] / w.sum(), abs=1e-9)
    # 처음 맞춘 축: 절댓값이 가장 큰 성분이 양수
    first = hs.GenePCA(dim)
    first.add(blocks16[0])
    first.fit()
    for k in range(2):
        assert first.axes[k, int(np.argmax(np.abs(first.axes[k])))] > 0
    # 부호: 뒤집힌 축을 넣어도 앞 축 쪽으로
    before = pca.axes.copy()
    pca.set_axes(pca.mean, -before, pca.explained)
    assert np.array_equal(pca.axes, before)
    # float16 그대로 넣은 것 = float64로 바꿔 넣은 것
    a, b = hs.GenePCA(dim), hs.GenePCA(dim)
    for blk in blocks16[:3]:
        a.add(blk)
        b.add(blk.astype(np.float64))
    assert np.array_equal(a.s2, b.s2) and np.array_equal(a.s1, b.s1)
    pts = pca.project(blocks16[-1])
    assert pts.shape == (7, 2) and pts.dtype == np.float32


def test_randomized_pca_for_big_genomes_is_repeatable() -> None:
    """T27: D > 256은 표본의 무작위 PCA(default_rng(0)): 두 번 돌려도 비트 단위로 같고, 뚜렷한 두 방향을 찾는다."""
    rng = np.random.default_rng(9)
    dim = 400
    basis = np.linalg.qr(rng.normal(size=(dim, 2)))[0].T
    blocks = [(rng.normal(size=(32, 2)) * [9.0, 4.0]) @ basis + rng.normal(0, 0.05, (32, dim)) for _ in range(150)]
    pca = hs.GenePCA(dim)
    assert pca.big and pca.s2 is None
    sample = hs.sample_rows([b.astype(np.float16) for b in blocks])
    assert sample.shape == (hs.PCA_SAMPLE_ROWS, dim)
    m1, a1, e1 = hs.randomized_axes(sample)
    m2, a2, e2 = hs.randomized_axes(sample)
    assert a1.tobytes() == a2.tobytes() and e1.tobytes() == e2.tobytes() and m1.tobytes() == m2.tobytes()
    assert abs(float(a1[0] @ basis[0])) > 0.99 and abs(float(a1[1] @ basis[1])) > 0.99
    assert e1[0] > e1[1] > 0
    # 고르게 뽑음: 앞과 뒤 세대 모두에서
    small = hs.sample_rows([np.full((10, 3), g, dtype=np.float16) for g in range(100)], limit=50)
    assert small.shape == (50, 3) and small[0, 0] == 0 and small[-1, 0] == 99
    assert hs.sample_rows([np.ones((3, 2))], limit=50).shape == (3, 2)
    pca.add(blocks[0])
    assert pca.wants_fit() and not pca.fit()  # 큰 차원은 화면이 배경에서 맞춘다
    pca.set_axes(m1, a1, e1, generations=1)
    assert pca.fitted_generations == 1 and pca.project(blocks[0]).shape == (32, 2)


def test_recent_64_generations_keep_exact_genomes(tmp_path: Path) -> None:
    """T29: exact_genome은 최근 64세대 안이면 메시지 값과 같고 밖이면 None. rep_genome은 메모리, 없으면 파일."""
    lay = meta_for("lander").layout
    h = hs.RunHistory()
    h.set_layout(lay)
    pops = {}
    for g in range(1, 71):
        obs, X = fake_generation(g, lay, pop=4)
        entry, exact = hs.live_entry(g, obs.to_arrays(), X, obs.representative, lay)
        assert entry is not None and exact is not None
        h.add(entry, exact=exact)
        pops[g] = (obs, X)
    for g in range(7, 71):
        obs, X = pops[g]
        for i in range(4):
            got = h.exact_genome(g, i)
            assert got is not None and got.tobytes() == X[i].tobytes()
        assert h.rep_genome(g).tobytes() == X[obs.representative].tobytes()
    for g in range(1, 7):
        assert h.exact_genome(g, 0) is None
        assert h.rep_genome(g) is None  # 실행 폴더를 모름
    assert h.exact_genome(70, 4) is None and h.exact_genome(70, -1) is None
    # 지도 점은 float16을 거친 값(디스크와 같음)
    e70 = h.entry(70)
    assert e70.map_genomes.dtype == np.float16
    assert np.array_equal(e70.map_genomes, pops[70][1].astype(np.float16))
    # 실행 폴더가 있으면 오래된 세대 대표는 파일에서(정확한 float64)
    obs1, X1 = pops[1]
    write_run(tmp_path, [pops[1]])
    h.run_dir = tmp_path
    assert h.rep_genome(1).tobytes() == X1[obs1.representative].tobytes()
    # 관찰이 없는 세대 메시지: 집단의 세대 대표로 대신한다
    entry, exact = hs.live_entry(71, None, pops[70][1], 2, lay, learner="1", rep_score=5.0)
    assert entry is not None and entry.obs is None and entry.rep_index == 2 and entry.learner == 1
    assert np.array_equal(entry.map_genomes[0], pops[70][1][2].astype(np.float16))
    # 챔피언
    h.add_champion(10, pops[10][1][0], 3.5)
    h.add_champion(30, pops[30][1][1])
    assert h.champion_at(9) is None
    assert h.champion_at(29)[0] == 10 and h.champion_at(70)[0] == 30
    assert h.champion_at(70)[1].tobytes() == pops[30][1][1].tobytes()


def test_density_cache_keeps_one_view_per_window_while_flipping_generations() -> None:
    """궤적 밀도 캐시는 창(이 세대만·최근·처음부터)마다 하나: 학습 중 최신 세대를 따라가거나 세대를 넘겨 봐도 밀도
    격자가 세대 수만큼 쌓이지 않는다. 같은 세대·같은 기록이면 같은 객체(다시 계산하지 않음)."""
    lay = meta_for("lander").layout
    frame = ob.GridFrame(-10.0, -5.0, 0.8, 32, 24)
    h = hs.RunHistory()
    h.set_layout(lay)
    for g in range(1, 121):
        obs, X = fake_generation(g, lay, frame=frame, world_key=TRACK_KEY)
        h.add(entry_of(obs, X, lay))
        assert h.density(g, "one") is not None  # 학습 중 최신 세대를 따라감
    for g in range(1, 121):  # 세대 넘겨 보기
        for window in hs.DENSITY_WINDOWS:
            assert h.density(g, window).generation == g
    dens = [k for k in h._cache if isinstance(k, tuple) and k[0] == "dens"]
    assert len(dens) == len(hs.DENSITY_WINDOWS)
    a = h.density(50, "recent")
    assert h.density(50, "recent") is a and h.density(51, "recent") is not a


def test_exact_populations_stay_within_the_memory_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """최근 64세대 정확한 집단은 세대 수와 바이트(EXACT_BUDGET) 둘 다로 줄인다. 한 세대만으로 넘으면 두지 않고(지도 점
    재생은 세대 대표로 대신), 그 세대의 대표 유전자는 그대로 정확하다."""
    lay = meta_for("lander").layout
    per_gen = 8 * lay.dim * 8  # 개체 8
    monkeypatch.setattr(hs, "EXACT_BUDGET", 10 * per_gen)  # 10세대만 들어가는 상한
    h = hs.RunHistory()
    h.set_layout(lay)
    pops = {}
    for g in range(1, 31):
        obs, X = fake_generation(g, lay, pop=8)
        entry, exact = hs.live_entry(g, obs.to_arrays(), X, obs.representative, lay)
        h.add(entry, exact=exact)
        pops[g] = (obs, X)
        assert h.exact_bytes() <= hs.EXACT_BUDGET
    assert [g for g in range(1, 31) if h.has_exact(g)] == list(range(21, 31))
    assert h.exact_genome(30, 5).tobytes() == pops[30][1][5].tobytes() and h.exact_genome(20, 5) is None
    # 한 세대만으로 상한을 넘으면 두지 않는다(메시지 집단을 통째로 복사하지도 않는다)
    monkeypatch.setattr(hs, "EXACT_BUDGET", per_gen // 2)
    h2 = hs.RunHistory()
    h2.set_layout(lay)
    obs, X = fake_generation(31, lay, pop=8)
    entry, exact = hs.live_entry(31, obs.to_arrays(), X, obs.representative, lay)
    assert exact is None and entry is not None
    h2.add(entry, exact=X.copy())  # 상한을 넘는 집단을 직접 넘겨도 두지 않는다
    assert not h2.has_exact(31) and h2.exact_bytes() == 0 and h2.exact_genome(31, 0) is None
    # 관찰이 없는 세대 메시지도 대표 유전자(float64 줄 하나)는 그대로
    entry2, exact2 = hs.live_entry(32, None, X, 3, lay)
    assert exact2 is None and entry2 is not None and entry2.rep_index == 3
    assert np.array_equal(entry2.map_genomes[0], X[3].astype(np.float16))


# ---------------------------------------------------------------------------
# 배경 읽기 (T28)
def _collect(loader: hs.HistoryLoader) -> dict[str, Any]:
    got: dict[str, Any] = {"chunks": [], "champions": None, "summary": None}
    loader.chunk.connect(lambda tag, entries, done, total: got["chunks"].append((tag, list(entries), done, total)))
    loader.champions.connect(lambda tag, champs: got.__setitem__("champions", (tag, champs)))
    loader.loaded.connect(lambda tag, summary: got.__setitem__("summary", (tag, summary)))
    return got


def test_loader_reads_v12_records_in_chunks_of_64(qtbot: QtBot, tmp_path: Path) -> None:
    lay = meta_for("lander").layout
    gens = [fake_generation(g, lay, pop=4) for g in range(1, 131)]
    rows = write_run(tmp_path, gens)
    loader = hs.HistoryLoader(tmp_path, rows, lay, tag=7)
    got = _collect(loader)
    with qtbot.waitSignal(loader.loaded, timeout=20000):
        loader.start()
    qtbot.waitUntil(lambda: got["summary"] is not None)
    sizes = [len(c[1]) for c in got["chunks"]]
    assert sizes == [64, 64, 2] and {c[0] for c in got["chunks"]} == {7}
    assert [c[2] for c in got["chunks"]] == [64, 128, 130] and all(c[3] == 130 for c in got["chunks"])
    assert got["summary"] == (7, {"observed": 130, "fallback": 0, "rejected": 0, "unreadable": 0, "missing": 0})
    entries = [e for c in got["chunks"] for e in c[1]]
    assert [e.generation for e in entries] == list(range(1, 131))
    for e, (obs, _X) in zip(entries, gens, strict=True):
        assert e.obs is not None and np.array_equal(e.obs.scores, obs.scores)
        assert np.array_equal(e.obs.genomes, obs.genomes)
    assert got["champions"] == (7, [])


def test_loader_falls_back_ignores_later_files_and_rejects_inconsistent(qtbot: QtBot, tmp_path: Path) -> None:
    """관찰 키가 없는 세대는 세대 대표로, 통계보다 뒤인 세대 파일은 읽지 않고, D10에 어긋나면 관찰을 버린다."""
    lay = meta_for("lander").layout
    gens = [fake_generation(g, lay, pop=5) for g in range(1, 11)]
    rows = write_run(tmp_path, gens[:4])
    rows += write_run(tmp_path, gens[4:6], with_obs=False)  # 5, 6세대: --no-observe나 옛 프로그램이 이어 감
    rows += write_run(tmp_path, gens[6:])
    rows = rows[:8]  # 9, 10세대 파일은 통계보다 뒤
    rows[2] = dict(rows[2], scores=rows[2]["scores"] + " 1.0")  # 3세대: 점수 칸이 다름(손으로 옮긴 파일 흉내)
    rows[6] = dict(rows[6], score_best="123.25")  # 7세대: 최고 점수가 다름
    obs1, X1 = gens[0]
    RunStore(tmp_path).save_champion(1, "champion", X1[0], score=4.5, compare_score=4.0, env_version="0123456789abcdef")
    RunStore(tmp_path).save_champion(9, "champion", X1[1], score=6.0, env_version="0123456789abcdef")  # 통계보다 뒤
    loader = hs.HistoryLoader(tmp_path, rows, lay, tag=1)
    got = _collect(loader)
    with qtbot.waitSignal(loader.loaded, timeout=20000):
        loader.start()
    qtbot.waitUntil(lambda: got["summary"] is not None)
    entries = {e.generation: e for c in got["chunks"] for e in c[1]}
    assert sorted(entries) == list(range(1, 9))
    assert got["summary"][1] == {"observed": 4, "fallback": 2, "rejected": 2, "unreadable": 0, "missing": 0}
    for g in (3, 5, 6, 7):
        e = entries[g]
        obs, X = gens[g - 1]
        assert e.obs is None and e.rep_index is None
        assert np.array_equal(e.rep_genome16, X[obs.representative].astype(np.float16))
        assert e.rep_score == float(obs.scores[obs.representative])
    assert all(entries[g].obs is not None for g in (1, 2, 4, 8))
    tag, champs = got["champions"]
    assert [c[0] for c in champs] == [1] and champs[0][2] == 4.5 and champs[0][1].tobytes() == X1[0].tobytes()


def test_loader_reads_the_real_v10_and_v11_runs(qtbot: QtBot, tmp_path: Path) -> None:
    """v1.0·v1.1 실제 실행 자료(관찰 키 없음): 세대마다 세대 대표로 대신하고, v1.1의 learner 열을 읽는다."""
    for name, n, champions in (("v10_run_ga", 4, [1, 2, 4]), ("v11_run_restart", 12, [1, 2, 3, 4, 5])):
        run = tmp_path / name
        shutil.copytree(DATA / name / "run", run)
        store = RunStore.open(run)
        spec = compile_env(sorted((run / "envs").glob("*.evoenv"))[0].read_text(encoding="utf-8"))
        lay = WatchMeta.from_spec(spec).layout
        loader = hs.HistoryLoader(run, store.read_stats(), lay, tag=3)
        got = _collect(loader)
        with qtbot.waitSignal(loader.loaded, timeout=20000):
            loader.start()
        qtbot.waitUntil(lambda got=got: got["summary"] is not None)
        entries = [e for c in got["chunks"] for e in c[1]]
        assert [e.generation for e in entries] == list(range(1, n + 1))
        assert got["summary"][1]["fallback"] == n and got["summary"][1]["observed"] == 0
        assert all(e.obs is None and e.map_genomes is not None and e.map_genomes.shape == (1, lay.dim) for e in entries)
        assert [c[0] for c in got["champions"][1]] == champions
        g3 = store.load_champion(3, "representative")
        assert np.array_equal(entries[2].rep_genome16, g3["genome"].astype(np.float16))
        if name == "v11_run_restart":
            assert [e.learner for e in entries] == [0] * 8 + [1] * 4  # 9세대부터 자동 재시작한 학습 방법
        else:
            assert all(e.learner is None for e in entries)


def test_loader_stops_when_interrupted(qtbot: QtBot, tmp_path: Path) -> None:
    lay = meta_for("lander").layout
    rows = write_run(tmp_path, [fake_generation(g, lay, pop=3) for g in range(1, 200)])
    loader = hs.HistoryLoader(tmp_path, rows, lay, tag=2, chunk_size=8)
    got = _collect(loader)
    # 첫 묶음을 보내는 순간(배경 스레드 안에서) 그만 읽으라고 한다: 다음 세대 전에 멈춘다
    loader.chunk.connect(lambda *_a: loader.requestInterruption(), Qt.ConnectionType.DirectConnection)
    loader.start()
    qtbot.waitUntil(loader.isFinished, timeout=20000)
    qtbot.wait(20)
    assert got["summary"] is None
    assert [len(c[1]) for c in got["chunks"]] == [8]


# ---------------------------------------------------------------------------
# 화면 스레드 우선(T39 고침): 화면이 무거운 일을 하는 동안 배경 읽기가 쉰다
def test_gui_priority_busy_quiet_and_limit() -> None:
    import time

    gp = hs.GuiPriority()
    assert not gp.is_busy
    t0 = time.monotonic()
    gp.wait()
    assert time.monotonic() - t0 < 0.05  # 한가하면 바로
    with gp.busy(quiet=10.0):
        with gp.busy(quiet=0.0):  # 겹쳐 불러도 바깥이 끝날 때까지 바쁨
            assert gp.is_busy
        assert gp.is_busy
        t0 = time.monotonic()
        gp.wait(limit=0.05)  # 계속 바빠도 limit초 뒤에는 돌아온다(배경 읽기가 굶지 않게)
        assert 0.04 <= time.monotonic() - t0 < 2.0
        t0 = time.monotonic()
        gp.wait(stop=lambda: True, limit=5.0)  # 멈추라고 하면 바로
        assert time.monotonic() - t0 < 0.5
    assert gp.is_busy  # 끝난 뒤 quiet초 동안(미뤄 둔 그림)
    assert not hasattr(gp, "watch_idle")  # quiet는 시간으로만 끝난다(운영체제마다 같게)
    gp = hs.GuiPriority()
    with gp.busy(quiet=0.05):
        pass
    assert gp.is_busy
    time.sleep(0.06)
    assert not gp.is_busy
    with gp.busy(quiet=0.0):
        pass
    assert not gp.is_busy


def test_loader_waits_while_the_gui_is_busy(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """HistoryLoader는 파일마다 잠깐 자고(LOAD_YIELD_S), 화면이 바쁘면(GUI_PRIORITY.busy) 끝날 때까지(최대
    GUI_WAIT_MAX_S) 파일을 읽지 않는다. 결과는 쉬지 않고 읽은 것과 같다."""
    import time

    assert hs.LOAD_YIELD_S > 0 and hs.GUI_WAIT_MAX_S >= 0.2
    lay = meta_for("lander").layout
    rows = write_run(tmp_path, [fake_generation(g, lay, pop=3) for g in range(1, 6)])
    reads: list[float] = []
    real = hs.load_generation

    def spy(*a: Any, **k: Any) -> Any:
        reads.append(time.monotonic())
        return real(*a, **k)

    monkeypatch.setattr(hs, "load_generation", spy)
    loader = hs.HistoryLoader(tmp_path, rows, lay, tag=3)
    got = _collect(loader)
    with hs.GUI_PRIORITY.busy(quiet=0.0):
        started = time.monotonic()
        loader.start()
        time.sleep(0.1)  # 화면 스레드가 바쁘다(이벤트를 처리하지 않음)
        released = time.monotonic()
    qtbot.waitUntil(lambda: got["summary"] is not None, timeout=20000)
    assert len(reads) == 5
    assert reads[0] >= min(released, started + 0.2)  # 바쁜 동안은 읽지 않았다(기다림 상한 0.25초 전에)
    entries = [e for c in got["chunks"] for e in c[1]]
    assert [e.generation for e in entries] == [1, 2, 3, 4, 5] and got["summary"][1]["observed"] == 5

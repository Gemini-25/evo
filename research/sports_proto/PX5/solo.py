"""PX5 「찾기」(되돌릴 지점 B17, 술래만 진화)의 학습(scratch 세계). PX4 solo.py 사본, 세계만 arena_hide.FindWorld.

제품의 혼자 하는 학습을 닮게: CMA-ES(개체 32, 시그마 0.5), 후보마다 시험 T판(같은 세대는 같은 시험 시드 [0, 실행 시드, 세대, k]),
점수 = 평균. 학습 시험 1,000회마다 검증: 후보 = learner.best()(CMA-ES 평균), 검증 시드 [2, k] 20개의 평균 점수, 챔피언 = 검증 점수가
가장 높은 후보(같으면 먼저 것). 예산 = 학습 시험 수(--trials). 끝나면 챔피언을 판정 시드 [3, k] 20개(판정)와 100개(보고)로 잰다.

쓰기: python solo.py find --seed 1 --trials 60000 --T 6 --workers 3 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_n] = "1"

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from selfplay import Evaluator, load  # noqa: E402  (sys.path에 저장소 src를 넣음)
from arena_hide import GAMES  # noqa: E402
from league import derive  # noqa: E402

from evolab.learners.cmaes import CMAES, CMAESConfig  # noqa: E402
from evolab.learners.grow import GrowConfig, GrowingCMAES  # noqa: E402
from evolab.policy.brain import BrainLayout  # noqa: E402
from evolab.policy.genome import GenomeLayout  # noqa: E402


def run_set(ev: Evaluator, genome: np.ndarray, seeds: list[int]) -> list:
    return ev.run([(s, [genome]) for s in seeds])


def summarize(res: list) -> dict:
    ends: dict = {}
    for _s, _p, info in res:
        ends[info["end"]] = ends.get(info["end"], 0) + 1
    return {"score": float(np.mean([s[0] for s, _p, _i in res])), "success": int(sum(1 for _s, p, _i in res if p[0] == 1.0)),
            "n": len(res), "ends": ends, "hidden_spawn": int(sum(1 for _s, _p, i in res if i.get("hidden_spawn")))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("game", choices=["find"])
    p.add_argument("--opts", default="{}")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--trials", type=int, default=28800)
    p.add_argument("--pop", type=int, default=32)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--T", type=int, default=3)
    p.add_argument("--hidden", type=int, nargs="*", default=[16])
    p.add_argument("--learner", default="cmaes", choices=["cmaes", "grow"])
    p.add_argument("--grow-start", type=int, default=2, dest="grow_start")
    p.add_argument("--grow-every", type=int, default=10, dest="grow_every")
    p.add_argument("--validate-every", type=int, default=1000, dest="validate_every")
    p.add_argument("--val-seeds", type=int, default=20, dest="val_seeds")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cls = GAMES[a.game]
    opts = json.loads(a.opts)
    hidden = tuple(a.hidden)
    lay = BrainLayout(cls.N_IN[0], hidden, cls.N_OUT[0])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = CMAESConfig(popsize=a.pop, sigma0=a.sigma, seed=derive(4, a.seed, 0) & 0xFFFFFFFF)
    x0 = np.zeros(lay.n_params)
    if a.learner == "grow":
        gl = GenomeLayout(body=(), brain=lay, output_ranges=tuple((-1.0, 1.0) for _ in range(lay.n_out)))
        learner = GrowingCMAES(x0, gl, cfg, GrowConfig(start=a.grow_start, every=a.grow_every))
    else:
        learner = CMAES(x0, cfg)
    meta = dict(vars(a), dims=lay.n_params, load_start=load())
    log = open(out / "log.jsonl", "w")
    log.write(json.dumps({"meta": meta}) + "\n")
    print(json.dumps(meta), flush=True)
    ev = Evaluator(a.game, opts, hidden, a.workers)
    val_seeds = [derive(2, k) for k in range(a.val_seeds)]
    judge20 = [derive(3, k) for k in range(20)]
    judge100 = [derive(3, k) for k in range(100)]
    trials = gen = 0
    champ = (0, x0.copy(), -np.inf)
    snaps = {0: x0.copy()}
    t0 = time.perf_counter()
    while trials < a.trials:
        X = learner.ask()
        seeds = [derive(0, a.seed, gen, k) for k in range(a.T)]
        matches = [(s, [X[i]]) for i in range(X.shape[0]) for s in seeds]
        res = ev.run(matches)
        sc = np.array([r[0][0] for r in res]).reshape(X.shape[0], a.T)
        succ = np.array([r[1][0] for r in res]).reshape(X.shape[0], a.T)
        agg = sc.mean(axis=1)
        learner.tell(X, agg)
        before = trials
        trials += len(matches)
        gen += 1
        row = {"gen": gen, "trials": trials, "best": float(agg.max()), "mean": float(agg.mean()), "succ": float(succ.mean()),
               "sigma": float(learner.stats()["sigma"])}
        if a.learner == "grow":
            row["neurons"] = int(learner.stats()["neurons"])
        if trials // a.validate_every > before // a.validate_every:
            cand = learner.best()
            v = summarize(run_set(ev, cand, val_seeds))
            snaps[gen] = np.array(cand, dtype=np.float64, copy=True)
            row["val"] = v["score"]
            row["val_success"] = v["success"]
            j = summarize(run_set(ev, cand, judge20))  # 보고용(챔피언 선정에 쓰지 않음)
            row["judge20_success"] = j["success"]
            if v["score"] > champ[2]:
                champ = (gen, np.array(cand, dtype=np.float64, copy=True), v["score"])
            row["champ"] = [champ[0], champ[2]]
            print(f"gen {gen} trials={trials} best={row['best']:.3f} mean={row['mean']:.3f} val={v['score']:.3f} "
                  f"val_succ={v['success']}/{len(val_seeds)} judge20={j['success']} champ={champ[0]} load={load()}", flush=True)
        row["t"] = time.perf_counter() - t0
        log.write(json.dumps(row) + "\n")
    j20 = summarize(run_set(ev, champ[1], judge20))
    j100 = summarize(run_set(ev, champ[1], judge100))
    last = learner.best()
    l20 = summarize(run_set(ev, last, judge20))
    summary = {"summary": {"gens": gen, "trials": trials, "seconds": time.perf_counter() - t0, "champ_gen": champ[0],
                           "champ_val": champ[2], "judge20": j20, "judge100": j100, "last_judge20": l20, "workers": a.workers,
                           "load_end": load(), "world_steps": ev.steps, "matches_all": ev.matches}}
    log.write(json.dumps(summary) + "\n")
    log.close()
    print(json.dumps(summary), flush=True)
    np.savez_compressed(out / "snaps.npz", **{f"g{g}": v for g, v in snaps.items()})
    np.savez_compressed(out / "champion.npz", genome=champ[1], gen=champ[0], val=champ[2])
    (out / "DONE").write_text(json.dumps(summary))
    ev.close()


if __name__ == "__main__":
    main()

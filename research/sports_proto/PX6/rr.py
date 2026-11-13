"""PX6 리그전(scratch 세계): 실행 하나의 고른 스냅숏(역할마다 20개)을 두 역할 모든 짝으로 붙인다.

쓰기: python rr.py <라벨> [--games 40] [--n 0] [--workers 3] [--out DIR]
  data/<라벨>.json·.npz를 읽어 <out>/<라벨>_rr.json을 쓴다. 자리 = [역할 0, 역할 1], 세계 시드 derive(8, k), k < games.
  --n > 0이면 처음 n개 세대만 씀(작은 시험용).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("label")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", default=".")
    a = p.parse_args()
    info = json.loads((HERE / "data" / f"{a.label}.json").read_text())
    sys.path.insert(0, str(HERE / info["dir"]))
    from league import derive, league_ratings  # noqa: E402
    from selfplay import Evaluator  # noqa: E402

    z = np.load(HERE / "data" / f"{a.label}.npz")
    gens = info["gens"][: a.n] if a.n > 0 else info["gens"]
    S = {r: {g: z[f"r{r}_g{g}"] for g in gens} for r in (0, 1)}
    seeds = [derive(8, k) for k in range(a.games)]
    ev = Evaluator(info["game"], info["opts"], tuple(info["hidden"]), a.workers)
    t0 = time.perf_counter()
    matches, idx = [], []
    for i, g0 in enumerate(gens):
        for j, g1 in enumerate(gens):
            for s in seeds:
                matches.append((s, [S[0][g0], S[1][g1]]))
                idx.append((i, j))
    res = ev.run(matches)
    ev.close()
    m = len(gens)
    P = np.zeros((m, m))
    ends: dict = {}
    for (i, j), (_s, pt, inf) in zip(idx, res):
        P[i, j] += pt[0]
        key = f"{gens[i]}_{gens[j]}"
        ends.setdefault(key, {})
        ends[key][inf["end"]] = ends[key].get(inf["end"], 0) + 1
    pts = np.zeros((2 * m, 2 * m))
    gm = np.zeros((2 * m, 2 * m))
    G = float(a.games)
    for i in range(m):
        for j in range(m):
            pts[i, m + j] = P[i, j]
            pts[m + j, i] = G - P[i, j]
            gm[i, m + j] = gm[m + j, i] = G
    rt = league_ratings(pts, gm, anchor=0, base=1000.0)
    out = {"label": a.label, "game": info["game"], "gens": gens, "games_per_pair": a.games, "workers": a.workers,
           "role0_points_sum": P.tolist(), "rr_elo_role0": rt[:m].tolist(), "rr_elo_role1": rt[m:].tolist(), "ends": ends,
           "matches": len(matches), "seconds": time.perf_counter() - t0}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{a.label}_rr.json").write_text(json.dumps(out, indent=1))
    print(a.label, len(matches), "games", round(out["seconds"], 1), "s", flush=True)


if __name__ == "__main__":
    main()

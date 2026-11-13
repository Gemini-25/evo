"""PX5 판정(scratch 세계): 숨바꼭질 실행 하나의 미리 정한 두 줄과 함께 보고할 값. 결과는 <실행 폴더>/analysis.json.

줄(PLAN.md, design.md 8절 PX5):
  (1) 술래 줄: 학습 경기 --budget판 안의 마지막 세 검증 각각에서 그때의 술래 챔피언(그때까지 가장 높게 재진 술래 스냅숏)이 가만히 있는
      숨는 쪽(시작 자리에 그대로)을 판정 시드 [3, k] 20판 중 10판 이상 잡고, 마지막(= 예산 안 챔피언)은 12판 이상.
  (2) 숨는 쪽 줄: 숨는 쪽 챔피언(예산 안 마지막 검증의 것)이 「곧장 쫓는 봇」(마지막으로 본 자리로 곧장)에게서 [3, k] 20판 중 8판 이상 달아남.
  보고: 검증마다 계열(그때 뜬 스냅숏과 그때 챔피언의 값), 챔피언 100판 값, 리그전 10 x 10([7, k] 20판)과 두 역할 한 표 엘로(PX6 참고).

쓰기: python analyze.py <실행 폴더> [--budget 60000 --workers 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from selfplay import Evaluator  # noqa: E402
from league import derive, league_ratings  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--budget", type=int, default=60000)
    p.add_argument("--workers", type=int, default=0)
    a = p.parse_args()
    run = Path(a.run)
    rows = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
    meta = rows[0]["meta"]
    vals = [r for r in rows if "elo0" in r]
    inb = [r for r in vals if r["matches"] <= a.budget]
    snaps = np.load(run / "snaps.npz")
    S: dict = {0: {}, 1: {}}
    for k in snaps.files:
        role, g = k.split("_")
        S[int(role[1:])][int(g[1:])] = snaps[k]
    ev = Evaluator(meta["game"], meta["opts"], tuple(meta["hidden"]), a.workers)
    J20 = [derive(3, k) for k in range(20)]
    J100 = [derive(3, k) for k in range(100)]
    RR = [derive(7, k) for k in range(20)]

    def games(sg, hg, seeds):
        return ev.run([(s, [sg, hg]) for s in seeds])

    def ends(res):
        e: dict = {}
        for _s, _p, info in res:
            e[info["end"]] = e.get(info["end"], 0) + 1
        return e

    def caught(res):
        return sum(1 for _s, _p, i in res if i["end"] == "caught")

    out: dict = {"run": run.name, "budget": a.budget, "validations": len(vals), "validations_in_budget": len(inb),
                 "last_matches_in_budget": inb[-1]["matches"]}
    # 검증마다: 그때 챔피언(술래)의 가만히 있는 숨는 쪽 잡기, 그때 뜬 스냅숏의 값(학습 중 기록)
    champ_cache: dict = {}
    series = []
    for r in vals:
        cg = r["champ0"][0]
        if cg not in champ_cache:
            champ_cache[cg] = caught(games(S[0][cg], "still", J20))
        hg = r["champ1"][0]
        key = ("h", hg)
        if key not in champ_cache:
            champ_cache[key] = 20 - caught(games("chase", S[1][hg], J20))
        series.append({"gen": r["gen"], "matches": r["matches"], "matches0": r["matches0"], "elo0": r["elo0"], "elo1": r["elo1"],
                       "snap_catch_still": r.get("wins_still0"), "snap_catch_random": r.get("wins_random0"),
                       "snap_escape_chase": r.get("wins_chase1"), "snap_escape_random": r.get("wins_random1"),
                       "champ0": r["champ0"], "champ1": r["champ1"], "champ_catch_still": champ_cache[cg],
                       "champ_escape_chase": champ_cache[key], "rungs0": r.get("rungs0"), "rungs1": r.get("rungs1")})
    out["series"] = series
    sin = [s for s in series if s["matches"] <= a.budget]
    line1 = [{"validation_gen": s["gen"], "matches": s["matches"], "champ_gen": s["champ0"][0], "champ_elo": s["champ0"][1],
              "catch20": s["champ_catch_still"]} for s in sin[-3:]]
    out["line1"] = line1
    out["line1_pass"] = bool(len(line1) == 3 and all(x["catch20"] >= 10 for x in line1) and line1[-1]["catch20"] >= 12)
    sc = inb[-1]["champ0"]
    hc = inb[-1]["champ1"]
    res = games("chase", S[1][hc[0]], J20)
    esc = 20 - caught(res)
    out["line2"] = {"hider_champ_gen": hc[0], "hider_champ_elo": hc[1], "escape20": esc, "ends20": ends(res)}
    out["line2_pass"] = esc >= 8
    out["pass_all"] = bool(out["line1_pass"] and out["line2_pass"])
    # 챔피언 100판 값(보고)
    out["seeker_champion"] = {"gen": sc[0], "elo": sc[1]}
    for name in ("still", "random"):
        out["seeker_champion"][f"catch_{name}_100"] = caught(games(S[0][sc[0]], name, J100))
    out["hider_champion"] = {"gen": hc[0], "elo": hc[1]}
    for name in ("chase", "random"):
        out["hider_champion"][f"escape_{name}_100"] = 100 - caught(games(name, S[1][hc[0]], J100))
    out["champions_vs_each_other_100"] = ends(games(S[0][sc[0]], S[1][hc[0]], J100))
    # 리그전(보고, PX6): 0번 + 예산 안 검증 스냅숏에서 역할마다 고르게 10개, 짝마다 [7, k] 20판
    gens = [0] + [r["gen"] for r in inb]
    n = len(gens)
    idx = sorted({round(i * (n - 1) / 9) for i in range(10)})
    sel = [gens[i] for i in idx]
    P = np.zeros((len(sel), len(sel)))
    for i, sg in enumerate(sel):
        for j, hg in enumerate(sel):
            res = games(S[0][sg], S[1][hg], RR)
            P[i, j] = float(np.mean([pt[0] for _s, pt, _i in res]))
    m = len(sel)
    pts = np.zeros((2 * m, 2 * m))
    gm = np.zeros((2 * m, 2 * m))
    for i in range(m):
        for j in range(m):
            pts[i, m + j] = P[i, j] * 20
            pts[m + j, i] = (1 - P[i, j]) * 20
            gm[i, m + j] = gm[m + j, i] = 20
    rt = league_ratings(pts, gm, anchor=0, base=1000.0)
    out["league"] = {"gens": sel, "seeker_points": P.tolist(), "league_elo_seekers": rt[:m].tolist(),
                     "league_elo_hiders": rt[m:].tolist(),
                     "last_seeker_vs_evolved_hiders": float(P[-1, 1:].mean()),
                     "last_hider_vs_evolved_seekers": float(1.0 - P[1:, -1].mean()),
                     "training_elo_seekers": [next((v["elo0"] for v in vals if v["gen"] == g), 1000.0) for g in sel],
                     "training_elo_hiders": [next((v["elo1"] for v in vals if v["gen"] == g), 1000.0) for g in sel]}
    ev.close()
    (run / "analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("line1_pass", "line2_pass", "pass_all")}), flush=True)
    print(json.dumps({"line1": [(x["champ_gen"], x["catch20"]) for x in line1], "line2": esc,
                      "series_snap_catch": [s["snap_catch_still"] for s in series],
                      "series_champ_catch": [s["champ_catch_still"] for s in series]}), flush=True)


if __name__ == "__main__":
    main()

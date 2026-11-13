"""PX4 판정(scratch 세계): 투수 대 타자 실행 하나의 미리 정한 세 줄과 함께 보고할 값을 잰다. 결과는 <실행 폴더>/analysis.json.

줄(PLAN.md, design.md 8절 PX4):
  (1) 학습 경기 --budget판 안의 마지막 세 검증 모두에서, 그때의 투수 챔피언(그때까지 가장 높게 재진 투수 스냅숏)이 가만히 있는 타자에게
      판정 시드 [3, k] 20판 중 스트라이크 14 이상.
  (2) 타자 챔피언(--budget판 안에서 가장 높게 재진 타자 스냅숏)이 「학습 경기 20,000판에 가장 가까운 검증」의 투수 스냅숏에게
      20판 중 안타(3m 넘게) 12 이상.
  (3) 리그전: 검증 스냅숏(0번 = 처음 유전자 포함, --budget판 안) 가운데 고르게 10개씩 두 역할, 짝마다 [7, k] 20판. 마지막 투수(--budget판
      안의 마지막 스냅숏)가 진화한 타자 9개(0번 빼고)에게 얻은 평균 승점 ≥ 0.3. 0번까지 넣은 값은 보고만.

쓰기: python analyze.py <실행 폴더> [--budget 100000 --workers 3]
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
    p.add_argument("--budget", type=int, default=100000)
    p.add_argument("--workers", type=int, default=0)
    a = p.parse_args()
    run = Path(a.run)
    rows = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
    meta = rows[0]["meta"]
    vals = [r for r in rows if "elo0" in r]
    inb = [r for r in vals if r["matches"] <= a.budget]
    snaps = np.load(run / "snaps.npz")
    S = {0: {}, 1: {}}
    for k in snaps.files:
        role, g = k.split("_")
        S[int(role[1:])][int(g[1:])] = snaps[k]
    hidden = tuple(meta["hidden"])
    ev = Evaluator(meta["game"], meta["opts"], hidden, a.workers)
    J20 = [derive(3, k) for k in range(20)]
    J100 = [derive(3, k) for k in range(100)]
    RR = [derive(7, k) for k in range(20)]

    def games(pg, bg, seeds):
        return ev.run([(s, [pg, bg]) for s in seeds])

    def ends(res):
        e: dict = {}
        for _s, _p, info in res:
            e[info["end"]] = e.get(info["end"], 0) + 1
        return e

    out: dict = {"run": run.name, "budget": a.budget, "validations": len(vals), "validations_in_budget": len(inb),
                 "last_matches_in_budget": inb[-1]["matches"], "final_matches": rows[-2].get("matches") if "summary" in rows[-1] else None}
    # (1) 마지막 세 검증의 투수 챔피언 대 가만히 있는 타자
    still_b = "still"
    line1 = []
    for r in inb[-3:]:
        cg = r["champ0"][0]
        res = games(S[0][cg], still_b, J20)
        strikes = sum(1 for _s, _p, i in res if i["end"] == "strike")
        logged = next((v.get("wins_still0") for v in vals if v["gen"] == cg), None)
        line1.append({"validation_gen": r["gen"], "matches": r["matches"], "champ_gen": cg, "champ_elo": r["champ0"][1],
                      "strikes20": strikes, "logged_at_snapshot": logged, "ends": ends(res)})
    out["line1"] = line1
    out["line1_pass"] = len(line1) == 3 and all(x["strikes20"] >= 14 for x in line1)
    # 투수 챔피언(예산 안 마지막)의 값들
    pc = inb[-1]["champ0"]
    out["pitcher_champion"] = {"gen": pc[0], "elo": pc[1]}
    for name in ("still", "swing"):
        res = games(S[0][pc[0]], name, J100)
        out["pitcher_champion"][f"vs_{name}_100"] = ends(res)
    # (2) 타자 챔피언 대 20,000판 무렵 투수 스냅숏
    bc = inb[-1]["champ1"]
    near = min(vals, key=lambda r: (abs(r["matches"] - 20000), r["matches"]))
    res20 = games(S[0][near["gen"]], S[1][bc[0]], J20)
    res100 = games(S[0][near["gen"]], S[1][bc[0]], J100)
    hits20 = sum(1 for _s, _p, i in res20 if i["end"] == "hit")
    out["line2"] = {"batter_champ_gen": bc[0], "batter_champ_elo": bc[1], "pitcher_snapshot_gen": near["gen"],
                    "pitcher_snapshot_matches": near["matches"], "pitcher_snapshot_elo": near["elo0"], "hits20": hits20,
                    "ends20": ends(res20), "hits100": sum(1 for _s, _p, i in res100 if i["end"] == "hit"), "ends100": ends(res100),
                    "batter_points20": float(np.mean([pt[1] for _s, pt, _i in res20]))}
    out["line2_pass"] = hits20 >= 12
    # 타자 챔피언 대 가만히 있는 투수·마지막 투수 스냅숏(보고)
    gens = [0] + [r["gen"] for r in inb]
    lastg = gens[-1]
    res = games(S[0][lastg], S[1][bc[0]], J100)
    out["batter_champion_vs_last_pitcher_100"] = ends(res)
    # (3) 리그전
    n = len(gens)
    idx = sorted({round(i * (n - 1) / 9) for i in range(10)})
    sel = [gens[i] for i in idx]
    P = np.zeros((len(sel), len(sel)))
    E: dict = {}
    for i, pg in enumerate(sel):
        for j, bg in enumerate(sel):
            res = games(S[0][pg], S[1][bg], RR)
            P[i, j] = float(np.mean([pt[0] for _s, pt, _i in res]))
            E[f"{pg}_{bg}"] = ends(res)
    last_vs_evolved = float(P[-1, 1:].mean())
    out["line3"] = {"gens": sel, "pitcher_points": P.tolist(), "last_pitcher_vs_evolved": last_vs_evolved,
                    "last_pitcher_vs_all10": float(P[-1].mean()), "pitcher_mean_by_snapshot_vs_evolved": P[:, 1:].mean(axis=1).tolist(),
                    "batter_mean_by_snapshot_vs_evolved_pitchers": (1.0 - P[1:, :]).mean(axis=0).tolist(), "ends": E}
    out["line3_pass"] = last_vs_evolved >= 0.3
    # 두 역할을 한 표에(PX6 참고용): 선수 2m개, 투수 i 대 타자 j만 붙음. 0번 투수 = 1,000
    m = len(sel)
    pts = np.zeros((2 * m, 2 * m))
    gm = np.zeros((2 * m, 2 * m))
    for i in range(m):
        for j in range(m):
            pts[i, m + j] = P[i, j] * 20
            pts[m + j, i] = (1 - P[i, j]) * 20
            gm[i, m + j] = gm[m + j, i] = 20
    rt = league_ratings(pts, gm, anchor=0, base=1000.0)
    out["line3"]["league_elo_pitchers"] = rt[:m].tolist()
    out["line3"]["league_elo_batters"] = rt[m:].tolist()
    out["training_elo_at_sel"] = {"pitcher": [next((v["elo0"] for v in vals if v["gen"] == g), 1000.0) for g in sel],
                                  "batter": [next((v["elo1"] for v in vals if v["gen"] == g), 1000.0) for g in sel]}
    # 검증마다 계열(보고): 스냅숏의 가만히 있는 타자 스트라이크, 엘로, 칸 수
    out["series"] = [{"gen": r["gen"], "matches": r["matches"], "elo0": r["elo0"], "elo1": r["elo1"],
                      "strikes_still": r.get("wins_still0"), "swing_wins": r.get("wins_swing0"), "rungs0": r.get("rungs0"),
                      "rungs1": r.get("rungs1"), "champ0": r["champ0"], "champ1": r["champ1"]} for r in vals]
    out["pass_all"] = bool(out["line1_pass"] and out["line2_pass"] and out["line3_pass"])
    ev.close()
    (run / "analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("line1_pass", "line2_pass", "line3_pass", "pass_all")}), flush=True)
    print(json.dumps({"line1": [(x["champ_gen"], x["strikes20"]) for x in line1], "line2": out["line2"]["hits20"],
                      "line3": last_vs_evolved}), flush=True)


if __name__ == "__main__":
    main()

"""PX2b 분석: 끝난 실행 하나를 예산 자름(학습 경기 60k/120k/240k)과 챔피언 규칙 셋으로 판정한다. scratch 세계.

챔피언 규칙(미리 정함, PLAN.md):
  elo    그 예산 안의 검증에서 가장 높게 재진 스냅숏(제품: 처음으로 가장 높은 것, 엘로 1,000을 넘은 것이 없으면 0번 = 가만히)
  floor  점수 재기에서 0번 칸(가만히 있는 상대)에게 12판 중 6판 이상 이긴 스냅숏 가운데 가장 높게 재진 것. 없으면 elo 규칙
  last   그 예산 안의 마지막 검증의 스냅숏
판정 시드 = derive(3, k // 2), 20판 = 10시드 × 두 자리(100판 = 50시드). 학습 중 검증에 적힌 「가만히 있는 상대에게 넣은 골」도 같은 시드라
고르는 데는 쓰지 않는다(floor는 점수 재기 시드 derive(1, ·)의 결과만 씀).
C10: (a) 가만히 있는 상대 20판 중 골로 이긴 판 ≥ 10, (b) 학습 경기 10,000판에 가장 가까운 스냅숏에게 20판 평균 승점 ≥ 0.6(챔피언이 그 스냅숏이면 미달).
「실험」: 8골 이상 + 사다리 첫 진짜 칸(0번 다음)에게 20판 승점 ≥ 0.7(그 칸이 없거나 챔피언이면 미달).
alt(과제 글의 풀이): 8골 이상 + 10,000판 스냅숏에게 0.7 이상(보고만).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arena import GAMES  # noqa: E402
from league import derive  # noqa: E402
from selfplay import Evaluator, seats_for  # noqa: E402

TAG_JUDGE = 3
FLOOR = 6


class Judge:
    def __init__(self, game: str, opts: dict, hidden: tuple, workers: int) -> None:
        self.cls = GAMES[game]
        self.ev = Evaluator(game, opts, hidden, workers)

    def play(self, focal, opp, games: int, tag: int = TAG_JUDGE) -> dict:
        ms, mines = [], []
        for k in range(games):
            seats, mine = seats_for(self.cls, 0, focal, opp, k % 2)
            ms.append((derive(tag, k // 2), seats))
            mines.append(mine)
        res = self.ev.run(ms)
        pts = [p[m] for m, (_s, p, _i) in zip(mines, res)]
        return {"points": float(np.mean(pts)), "wins": int(sum(1 for p in pts if p == 1.0)),
                "draws": int(sum(1 for p in pts if p == 0.5)), "losses": int(sum(1 for p in pts if p == 0.0)),
                "goal_ends": int(sum(1 for _s, _p, i in res if i["end"] == "goal"))}


def analyze(run: Path, cuts: list[int], workers: int = 0) -> dict:
    rows = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    meta = rows[0]["meta"]
    summary = next(r["summary"] for r in rows if "summary" in r)
    vals = [r for r in rows if "gen" in r and "elo0" in r]
    snaps_z = np.load(run / "snaps.npz")
    snaps = {int(k.split("_g")[1]): snaps_z[k] for k in snaps_z.files}
    lad_z = np.load(run / "ladder.npz")
    ladder = sorted(((int(k.split("_")[2]), int(k.split("_")[1][1:]), lad_z[k][:-1], float(lad_z[k][-1])) for k in lad_z.files))
    base = float(meta["elo_base"])
    J = Judge(meta["game"], meta["opts"], tuple(meta["hidden"]), workers)
    snap10k = min(vals, key=lambda r: (abs(r["matches"] - 10000), r["gen"]))
    cache: dict = {}

    def judge(gen: int, rung1) -> dict:
        key = (gen, None if rung1 is None else rung1[1])
        if key in cache:
            return cache[key]
        g = snaps[gen]
        s20 = J.play(g, "still", 20)
        s100 = J.play(g, "still", 100)
        n20 = J.play(g, snaps[snap10k["gen"]], 20)
        n100 = J.play(g, snaps[snap10k["gen"]], 100)
        c20 = J.play(g, "chaser", 20)
        out = {"gen": gen, "still20": s20["wins"], "own20": s20["losses"], "still100": s100["wins"], "own100": s100["losses"],
               "snap10k20": n20["points"], "snap10k20_wdl": [n20["wins"], n20["draws"], n20["losses"]], "snap10k100": n100["points"],
               "chaser20": c20["points"], "is_snap10k": gen == snap10k["gen"]}
        if rung1 is not None and gen != rung1[1]:
            r20 = J.play(g, rung1[2], 20)
            out.update({"rung1_gen": rung1[1], "rung1_20": r20["points"], "rung1_100": J.play(g, rung1[2], 100)["points"]})
        else:
            out.update({"rung1_gen": None if rung1 is None else rung1[1], "rung1_20": None, "rung1_100": None})
        cache[key] = out
        return out

    res_cuts = []
    for cut in cuts:
        vc = [r for r in vals if r["matches"] <= cut]
        if not vc:
            continue
        last_gen = vc[-1]["gen"]
        champ = (0, base)
        for r in vc:
            if r["elo0"] > champ[1]:
                champ = (r["gen"], r["elo0"])
        fl = None
        for r in vc:
            if r.get("r0wins0", 0) >= FLOOR and (fl is None or r["elo0"] > fl[1]):
                fl = (r["gen"], r["elo0"])
        choice = {"elo": champ, "floor": fl if fl is not None else champ, "last": (last_gen, vc[-1]["elo0"])}
        rung1 = ladder[1] if len(ladder) > 1 and ladder[1][1] <= last_gen else None
        rules = {}
        for name, (gen, elo) in choice.items():
            j = dict(judge(gen, rung1))
            j["elo"] = elo
            j["floor_fallback"] = name == "floor" and fl is None
            a = j["still20"] >= 10
            b = (not j["is_snap10k"]) and j["snap10k20"] >= 0.6
            e = j["still20"] >= 8 and j["rung1_20"] is not None and j["rung1_20"] >= 0.7
            alt = j["still20"] >= 8 and (not j["is_snap10k"]) and j["snap10k20"] >= 0.7
            j.update({"a": bool(a), "b": bool(b), "c10": bool(a and b), "exp": bool(e), "alt": bool(alt)})
            rules[name] = j
        res_cuts.append({"cut": cut, "last_gen": last_gen, "last_matches": vc[-1]["matches"], "rungs": sum(1 for x in ladder if x[1] <= last_gen),
                         "rules": rules})
    series = [{"gen": r["gen"], "matches": r["matches"], "elo": r["elo0"], "goals_still": r.get("wins_still0"),
               "own_still": r.get("loss_still0"), "r0wins": r.get("r0wins0"), "chaser": r.get("vs_chaser0"), "rung": bool(r.get("rung0")),
               "draw_train": r.get("draw0"), "t": r["t"]} for r in vals]
    return {"run": str(run), "meta": meta, "summary": {k: v for k, v in summary.items() if k != "rungs"},
            "ladder": [[gen, elo] for _k, gen, _g, elo in ladder], "snap10k": [snap10k["gen"], snap10k["matches"]],
            "cuts": res_cuts, "series": series, "judge_matches": J.ev.matches}


if __name__ == "__main__":
    r = analyze(Path(sys.argv[1]), [int(c) for c in sys.argv[2].split(",")], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    print(json.dumps(r["cuts"], indent=1))

"""PX3(축구 2대2, PX2b가 확인한 ks 설정) 실행 하나 = (설정, 시드): 학습(selfplay.train) → 분석(analyze.analyze). scratch 세계(pymunk 원형), 실제 런타임 아님.

쓰기: python run.py --variant ks --seed 1 --games 120000 --workers 3 --out $MEASURE_OUT
      python run.py --list            설정 표
결과: <out>/run/(log.jsonl, snaps.npz, ladder.npz, champion.npz, snap_elo.json, DONE), <out>/analysis.json, <out>/variant.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

# PX2와 같은 바탕(DE 4.9-(3) + 제품 대전 층 기본: 새 칸만, 모음 16, 가장 새 50%, 검증 1,000판, 점수 재기 칸마다 12판, 칸 간격 70, 엘로 바닥 1,000)
BASE: dict = {"game": "soccer2r", "opts": {}, "hidden": [], "learner": "cmaes", "pop": 32, "sigma": 0.5, "K": 8, "cap": 16,
              "latest": 0.5, "pool_admit": "rung", "ref_share": 0.0, "validate_every": 1000, "rate_games": 12, "rung_step": 70.0,
              "elo_base": 1000.0, "eval_games": 20, "baselines": {"0": ["still", "chaser"]}}

# PX3(PLAN.md, 미리 정함): ks = PX2b가 1대1에서 확인한 설정(K 16 = 8짝 × 두 자리, 학습 경기의 25%는 0번 칸) + 2대2 세계.
# 2대2 세계는 DE 4.9-(3) 그대로 상대 둘을 가까운 순서로 넣는 soccer2r(본 판정). 같은 자리의 두 선수는 뇌 하나(D5), 양 팀도 같은 뇌(D6).
# ks_idx = 원형 Soccer2Near(상대를 번호 순으로 넣음, DE 9.1-11의 soccer2n_rungs_s1과 같은 센서): 보고만.
VARIANTS: dict = {
    "ks": {"K": 16, "ref_share": 0.25},
    "ks_idx": {"game": "soccer2n", "K": 16, "ref_share": 0.25},
}


def settings(variant: str) -> dict:
    s = json.loads(json.dumps(BASE))
    for k, v in VARIANTS[variant].items():
        if k == "opts":
            s["opts"].update(v)
        else:
            s[k] = v
    return s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant")
    p.add_argument("--seed", type=int)
    p.add_argument("--games", type=int, default=120000)
    p.add_argument("--cuts", default="60000,120000")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out")
    p.add_argument("--list", action="store_true")
    p.add_argument("--no-analyze", action="store_true")
    a = p.parse_args()
    if a.list:
        for v in VARIANTS:
            print(v, json.dumps(settings(v)))
        return
    import selfplay
    from analyze import analyze

    s = settings(a.variant)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "variant.json").write_text(json.dumps({"variant": a.variant, "seed": a.seed, "games": a.games, "workers": a.workers,
                                                 "settings": s}, indent=1))
    ns = argparse.Namespace(game=s["game"], opts=json.dumps(s["opts"]), seed=a.seed, minutes=0.0, gens=0, pop=s["pop"], sigma=s["sigma"],
                            K=s["K"], hidden=list(s["hidden"]), cap=s["cap"], latest=s["latest"], agg="mean", pool_admit=s["pool_admit"],
                            learner=s["learner"], grow_start=2, grow_every=10, games=a.games, validate_every=s["validate_every"],
                            elo_base=s["elo_base"], ref_share=s["ref_share"], snap_every=5, rate_games=s["rate_games"],
                            rung_step=s["rung_step"], eval_every=10, eval_games=s["eval_games"], baselines=json.dumps(s["baselines"]),
                            workers=a.workers, out=str(out / "run"))
    t0 = time.perf_counter()
    selfplay.train(ns)
    t_train = time.perf_counter() - t0
    if a.no_analyze:
        return
    cuts = [int(c) for c in a.cuts.split(",") if int(c) <= a.games]
    t1 = time.perf_counter()
    res = analyze(out / "run", cuts, workers=a.workers)
    res["wall_train_s"] = t_train
    res["wall_analyze_s"] = time.perf_counter() - t1
    res["variant"] = a.variant
    (out / "analysis.json").write_text(json.dumps(res, indent=1))
    for c in res["cuts"]:
        for rule, j in c["rules"].items():
            print(f"cut {c['cut']} {rule}: champ gen {j['gen']} elo {j['elo']:.0f} goals {j['still20']}/20 ({j['still100']}/100) "
                  f"snap10k {j['snap10k20']:.3f} rung1 {j['rung1_20']} chaser {j['chaser20']:.3f} -> p8 {j['p8']} c10 {j['c10']} exp {j['exp']}", flush=True)
    print(f"# train {t_train:.0f}s analyze {res['wall_analyze_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""PX4 대전 원형(scratch 세계): 여러 선수 평가(뇌 묶음 계산) + 스스로 대전 학습(evolab CMAES·GrowingCMAES, BatchBrain을 저장소에서 읽기만).

출처: scratchpad/sports/px/PX2/proto/selfplay.py(= PX1의 것 + 자라는 뇌). PX4에서 바꾼 것:
  - 세계는 arena_pitch.py(투수 대 타자와 혼자 하는 연습 둘).
  - --validate-on role0: 검증 주기를 「역할 0(투수)의 학습 경기 수」로 센다(DE 4.6의 5: 역할 0의 기록이 검증 주기를 넘으면). 예산(--games)은
    두 역할의 학습 경기 합.
  - 끝에 모든 스냅숏(snaps.npz), 사다리(ladder.npz), 스냅숏 엘로를 남긴다(PX1·PX2와 같음: PX6이 쓴다).

쓰기: python selfplay.py <game> --seed 1 --games 100000 --out <dir> [...]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

for _n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_n] = "1"

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))  # 저장소의 src/evolab(읽기만)
from arena_pitch import GAMES  # noqa: E402
from league import Ladder, Pool, derive, performance_rating, pick_index  # noqa: E402

from evolab.learners.cmaes import CMAES, CMAESConfig  # noqa: E402
from evolab.learners.grow import GrowConfig, GrowingCMAES  # noqa: E402
from evolab.policy.genome import GenomeLayout  # noqa: E402
from evolab.policy.brain import BatchBrain, BrainLayout, unpack  # noqa: E402

WINDOW = 12
Seat = "np.ndarray | str"


def load() -> str:
    return " ".join(open("/proc/loadavg").read().split()[:3])


def play(game: str, opts: dict, hidden: tuple[int, ...], matches: list[tuple[int, list]], window: int = WINDOW):
    """matches: (세계 시드, 자리마다 유전자 배열 또는 기준 상대 이름). 돌려줌: (결과 목록, 세계 스텝 수, 시뮬 초, 신경망 초).

    살아 있는 세계를 window개만 두고 끝난 자리를 다음 대진으로 채운다. 역할마다 (자리 x 그 역할 선수 수) 행을 한 번에 계산.
    """
    cls = GAMES[game]
    seat_roles = cls.seat_roles()
    roles = sorted(set(seat_roles))
    layouts = {r: BrainLayout(cls.N_IN[r], hidden, cls.N_OUT[r]) for r in roles}
    n = len(matches)
    results: list = [None] * n
    window = min(window, n)
    slots: list = [None] * window
    probe = cls(0, **opts)
    agents = probe.agents
    per_role = {r: [a for a, (_, rr) in enumerate(agents) if rr == r] for r in roles}
    W = {r: [np.zeros((window * len(per_role[r]), fi, fo)) for fi, fo in layouts[r].shapes] for r in roles}
    Bv = {r: [np.zeros((window * len(per_role[r]), fo)) for _, fo in layouts[r].shapes] for r in roles}
    X = {r: np.zeros((window * len(per_role[r]), layouts[r].n_in)) for r in roles}
    brains = {r: BatchBrain(layouts[r], W[r], Bv[r]) for r in roles}
    # BatchBrain은 연속 배열 사본을 만들 수 있으므로 그 안의 배열에 바로 쓴다
    W = {r: brains[r].weights for r in roles}
    Bv = {r: brains[r].biases for r in roles}
    cache: dict[int, BatchBrain] = {}
    nxt = 0
    live = 0
    steps = 0
    t_nn = 0.0
    t0 = time.perf_counter()

    def start(s: int) -> bool:
        nonlocal nxt, live
        if nxt >= n:
            slots[s] = None
            return False
        i = nxt
        nxt += 1
        seed, seats = matches[i]
        w = cls(seed, **opts)
        scripted = {}
        for r in roles:
            for k, a in enumerate(per_role[r]):
                g = seats[w.agents[a][0]]
                if isinstance(g, str):
                    scripted[a] = g
                    continue
                bb = cache.get(id(g))
                if bb is None:
                    bb = cache[id(g)] = unpack(layouts[r], g)
                row = s * len(per_role[r]) + k
                for dst, src in zip(W[r], bb.weights):
                    dst[row] = src[0]
                for dst, src in zip(Bv[r], bb.biases):
                    dst[row] = src[0]
        slots[s] = (i, w, scripted)
        live += 1
        return True

    for s in range(window):
        start(s)
    while live:
        obs_all = {}
        for s, sl in enumerate(slots):
            if sl is None:
                continue
            ob = sl[1].observe()
            obs_all[s] = ob
            for r in roles:
                base = s * len(per_role[r])
                for k, a in enumerate(per_role[r]):
                    X[r][base + k] = ob[a]
        t1 = time.perf_counter()
        acts = {r: brains[r].forward(X[r]).tolist() for r in roles}
        t_nn += time.perf_counter() - t1
        for s, sl in enumerate(slots):
            if sl is None:
                continue
            i, w, scripted = sl
            actions: list = [None] * len(agents)
            for r in roles:
                base = s * len(per_role[r])
                for k, a in enumerate(per_role[r]):
                    actions[a] = w.script(a, scripted[a]) if a in scripted else acts[r][base + k]
            w.step(actions)
            steps += 1
            if w.done:
                results[i] = w.result()
                live -= 1
                start(s)
    total = time.perf_counter() - t0
    return results, steps, total - t_nn, t_nn


def _chunk(args):
    return play(*args)


class Evaluator:
    def __init__(self, game: str, opts: dict, hidden: tuple[int, ...], workers: int) -> None:
        self.game, self.opts, self.hidden, self.workers = game, opts, hidden, workers
        self.pool = mp.get_context("fork").Pool(workers) if workers > 0 else None
        self.steps = 0
        self.sim = 0.0
        self.nn = 0.0
        self.matches = 0

    def run(self, matches: list[tuple[int, list]]):
        if not matches:
            return []
        if self.pool is None:
            parts = [play(self.game, self.opts, self.hidden, matches)]
        else:
            k = self.workers * 4
            size = -(-len(matches) // k)
            chunks = [matches[i:i + size] for i in range(0, len(matches), size)]
            parts = self.pool.map(_chunk, [(self.game, self.opts, self.hidden, c) for c in chunks], chunksize=1)
        out = []
        for res, steps, sim, nn in parts:
            out += res
            self.steps += steps
            self.sim += sim
            self.nn += nn
        self.matches += len(matches)
        return out

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def seats_for(cls, role: int, focal, opp, side: int) -> tuple[list, int]:
    """(자리마다 유전자, 내 자리 번호). 대칭 게임(역할 하나)은 side로 자리를 바꾸고, 역할이 다르면 그 역할의 자리."""
    sr = cls.seat_roles()
    if len(set(sr)) == 1:
        return ([focal, opp] if side == 0 else [opp, focal]), side
    mine = sr.index(role)
    seats = [opp] * len(sr)
    seats[mine] = focal
    return seats, mine


def effective_pool(pool: Pool, recent: list) -> list:
    """대진에 쓰는 상대 목록(세대 오름차순). 「새 칸 + 최근 둘」이면 모음(새 칸) 밖의 최근 스냅숏을 더한다."""
    if not recent:
        return pool.entries
    have = {g for g, _ in pool.entries}
    ent = pool.entries + [e for e in recent if e[0] not in have]
    ent.sort(key=lambda e: e[0])
    return ent


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def train(args: argparse.Namespace) -> None:
    """PX1: 원형 train()에 더한 것 — 학습 경기 수 예산(--games), 경기 수로 정하는 검증 주기(--validate-every), 상대 모음에 넣는 규칙
    (--pool-admit rung | all | rung+latest2), 엘로 눈금 바닥(--elo-base), 챔피언(가장 높게 재진 스냅숏), 검증마다 체크포인트와 이어하기.
    대진·시드·점수 모으기·점수 재기의 셈은 원형과 같다.
    PX2: --learner grow = 저장소의 GrowingCMAES(자라는 뇌: --hidden이 틀, --grow-start에서 시작해 --grow-every 세대마다 두 배). 만드는 법은
    engine/trainer.py의 make_learner와 같다(몸 유전자 없음). 기준 상대 평가에 이긴 판·진 판 수를 더 적는다."""
    cls = GAMES[args.game]
    opts = json.loads(args.opts)
    hidden = tuple(args.hidden)
    sr = cls.seat_roles()
    roles = sorted(set(sr))
    symmetric = len(roles) == 1
    other = {r: (r if symmetric else [x for x in roles if x != r][0]) for r in roles}
    layouts = {r: BrainLayout(cls.N_IN[r], hidden, cls.N_OUT[r]) for r in roles}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        print(f"# {out.name}: already complete", flush=True)
        return
    admit = args.pool_admit
    base = args.elo_base

    def make_learner(r: int):
        cfg = CMAESConfig(popsize=args.pop, sigma0=args.sigma, seed=derive(4, args.seed, r) & 0xFFFFFFFF)
        x0 = np.zeros(layouts[r].n_params)
        if args.learner == "grow":
            gl = GenomeLayout(body=(), brain=layouts[r], output_ranges=tuple((-1.0, 1.0) for _ in range(layouts[r].n_out)))
            return GrowingCMAES(x0, gl, cfg, GrowConfig(start=args.grow_start, every=args.grow_every))
        return CMAES(x0, cfg)

    learners = {r: make_learner(r) for r in roles}
    pools = {r: Pool(cap=args.cap) for r in roles}
    ladders = {r: Ladder(step=args.rung_step) for r in roles}
    for r in roles:
        z = np.zeros(layouts[r].n_params)
        pools[r].add(0, z)
        ladders[r].add(0, z, base)
    snaps = {r: {0: np.zeros(layouts[r].n_params)} for r in roles}
    snap_elo = {r: {0: base} for r in roles}
    recent: dict = {r: [] for r in roles}
    champ = {r: (0, np.zeros(layouts[r].n_params), base) for r in roles}
    baselines = json.loads(args.baselines)  # {역할 번호: [상대 이름...]} (그 역할이 상대하는 기준 상대)
    meta = {"game": args.game, "opts": opts, "hidden": hidden, "seed": args.seed, "pop": args.pop, "sigma": args.sigma, "K": args.K,
            "cap": args.cap, "latest": args.latest, "ref_share": args.ref_share, "pool_admit": admit, "agg": args.agg,
            "snap_every": args.snap_every, "validate_every": args.validate_every, "games": args.games, "gens": args.gens,
            "rate_games": args.rate_games, "rung_step": args.rung_step, "elo_base": base, "eval_games": args.eval_games,
            "baselines": baselines, "dims": {r: layouts[r].n_params for r in roles}, "learner": args.learner}
    if args.learner == "grow":
        meta.update({"grow_start": args.grow_start, "grow_every": args.grow_every})
    meta["validate_on"] = args.validate_on
    ckpt_path = out / "ckpt.pkl"
    log_path = out / "log.jsonl"
    gen = 0
    train_matches = rate_matches = eval_matches = 0
    role0_matches = 0
    elapsed0 = 0.0
    resumes = 0
    ev = Evaluator(args.game, opts, hidden, args.workers)
    if ckpt_path.exists():
        st = pickle.loads(ckpt_path.read_bytes())
        if st["meta"] != json.loads(json.dumps(meta)):
            raise SystemExit(f"{out}: checkpoint was made with different settings")
        gen, train_matches, rate_matches, eval_matches = st["gen"], st["train_matches"], st["rate_matches"], st["eval_matches"]
        role0_matches = st["role0_matches"]
        elapsed0, resumes = st["elapsed"], st["resumes"] + 1
        for r in roles:
            learners[r].load_state_dict(st["learners"][r])
            pools[r].entries = st["pools"][r]
            ladders[r].rungs = st["ladders"][r]
            snaps[r], snap_elo[r], recent[r], champ[r] = st["snaps"][r], st["snap_elo"][r], st["recent"][r], st["champ"][r]
        ev.steps, ev.sim, ev.nn, ev.matches = st["ev"]
        lines = log_path.read_text().splitlines()[: st["rows"]]
        _atomic_write(log_path, ("\n".join(lines) + "\n").encode())
        log = open(log_path, "a")
        n_rows = st["rows"]
        log.write(json.dumps({"resume": {"at_gen": gen, "n": resumes, "workers": args.workers, "load": load()}}) + "\n")
        n_rows += 1
        print(f"# resumed {out.name} at gen {gen} ({train_matches} games), resume #{resumes}, load {load()}", flush=True)
    else:
        log = open(log_path, "w")
        log.write(json.dumps({"meta": dict(meta, workers=args.workers, load_start=load())}) + "\n")
        n_rows = 1
        print(json.dumps(dict(meta, workers=args.workers, load_start=load())), flush=True)
    t_start = time.perf_counter()

    def save_ckpt() -> None:
        log.flush()
        os.fsync(log.fileno())
        st = {"meta": json.loads(json.dumps(meta)), "gen": gen, "train_matches": train_matches, "rate_matches": rate_matches,
              "eval_matches": eval_matches, "role0_matches": role0_matches, "elapsed": elapsed0 + time.perf_counter() - t_start, "resumes": resumes,
              "learners": {r: learners[r].state_dict() for r in roles}, "pools": {r: pools[r].entries for r in roles},
              "ladders": {r: ladders[r].rungs for r in roles}, "snaps": snaps, "snap_elo": snap_elo, "recent": recent, "champ": champ,
              "ev": (ev.steps, ev.sim, ev.nn, ev.matches), "rows": n_rows}
        _atomic_write(ckpt_path, pickle.dumps(st, protocol=4))

    while True:
        elapsed = elapsed0 + time.perf_counter() - t_start
        if ((args.minutes and elapsed >= args.minutes * 60) or (args.gens and gen >= args.gens)
                or (args.games and train_matches >= args.games)):
            break
        t_gen = time.perf_counter()
        X = {r: learners[r].ask() for r in roles}
        matches, index = [], []
        opp_gens: dict = {r: [] for r in roles}
        for r in roles:
            entries = effective_pool(pools[other[r]], recent[other[r]] if admit == "rung+latest2" else [])
            for k in range(args.K):
                pair = k // 2 if symmetric else k  # 대칭 게임: 같은 시드·상대를 자리만 바꿔 두 번
                oi = pick_index(len(entries), args.seed, gen, pair, r, args.latest, args.ref_share)
                seed = derive(0, args.seed, gen, pair)
                side = k % 2
                if side == 0 or not symmetric:
                    opp_gens[r].append(entries[oi][0])
                for i in range(X[r].shape[0]):
                    seats, mine = seats_for(cls, r, X[r][i], entries[oi][1], side)
                    matches.append((seed, seats))
                    index.append((r, i, k, mine))
        res = ev.run(matches)
        before = train_matches
        before0 = role0_matches
        train_matches += len(matches)
        role0_matches += sum(1 for ix in index if ix[0] == roles[0])
        sc = {r: np.zeros((X[r].shape[0], args.K)) for r in roles}
        pt = {r: np.zeros((X[r].shape[0], args.K)) for r in roles}
        ends: dict = {r: {} for r in roles}
        for (r, i, k, mine), (scores, pts, info) in zip(index, res):
            sc[r][i, k] = scores[mine]
            pt[r][i, k] = pts[mine]
            ends[r][info["end"]] = ends[r].get(info["end"], 0) + 1
        row: dict = {"gen": gen + 1, "t": 0.0, "matches": train_matches, "matches0": role0_matches}
        for r in roles:
            if args.agg == "min":  # 상대(대진 짝)마다 평균한 뒤 가장 나쁜 상대 값
                per = sc[r].reshape(sc[r].shape[0], -1, 2 if symmetric else 1).mean(axis=2)
                agg = per.min(axis=1)
            else:
                agg = sc[r].mean(axis=1)
            learners[r].tell(X[r], agg)
            row[f"best{r}"] = float(agg.max())
            row[f"mean{r}"] = float(agg.mean())
            row[f"pts{r}"] = float(pt[r].mean())
            row[f"draw{r}"] = float((pt[r] == 0.5).mean())
            row[f"ends{r}"] = ends[r]
            row[f"opp{r}"] = opp_gens[r]
            lst = learners[r].stats()
            row[f"sigma{r}"] = float(lst["sigma"])
            if args.learner == "grow":  # 이번 세대 후보를 만든 크기(자라기 전)와 지금까지 자란 횟수
                row[f"neurons{r}"] = int(lst["neurons"])
                row[f"growths{r}"] = int(lst["growths"])
        gen += 1
        # 검증: 스냅숏을 뜨고 사다리로 점수를 잰 뒤 상대 모음에 넣는다
        if args.validate_every and args.validate_on == "role0":
            validate = role0_matches // args.validate_every > before0 // args.validate_every
        elif args.validate_every:
            validate = train_matches // args.validate_every > before // args.validate_every
        else:
            validate = gen % args.snap_every == 0
        if validate:
            cands = {r: learners[r].best() for r in roles}
            rated = {}
            for r in roles:
                lad = ladders[other[r]]
                ids = lad.play_set()
                ms, idx = [], []
                for j in ids:
                    for k in range(args.rate_games):
                        seed = derive(1, k // 2 if symmetric else k)
                        seats, mine = seats_for(cls, r, cands[r], lad.rungs[j][1], k % 2)
                        ms.append((seed, seats))
                        idx.append((j, mine))
                rr = ev.run(ms)
                rate_matches += len(ms)
                got = {j: 0.0 for j in ids}
                for (j, mine), (_s, pts, _i) in zip(idx, rr):
                    got[j] += pts[mine]
                rating = performance_rating([(lad.rungs[j][2], got[j], args.rate_games) for j in ids])
                row[f"elo{r}"] = rating
                row[f"vs_rungs{r}"] = {str(lad.rungs[j][0]): got[j] / args.rate_games for j in ids}
                rated[r] = rating
            for r in roles:  # 모든 역할을 검증 전의 사다리로 잰 뒤 한꺼번에 칸·모음·챔피언을 고친다
                g, rating = cands[r], rated[r]
                snaps[r][gen] = g
                snap_elo[r][gen] = rating
                rung = rating >= ladders[r].top() + ladders[r].step
                if rung or admit == "all":
                    pools[r].add(gen, g)
                if rung:
                    ladders[r].add(gen, g, rating)
                recent[r] = (recent[r] + [(gen, np.array(g, dtype=np.float64, copy=True))])[-2:]
                new_champ = rating > champ[r][2]
                if new_champ:
                    champ[r] = (gen, np.array(g, dtype=np.float64, copy=True), rating)
                row[f"rung{r}"] = bool(rung)
                row[f"rungs{r}"] = len(ladders[r].rungs)
                row[f"pool{r}"] = [e[0] for e in effective_pool(pools[r], recent[r] if admit == "rung+latest2" else [])]
                row[f"champ{r}"] = [champ[r][0], champ[r][2]]
            # 기준 상대(학습에 쓰지 않는 판정): 고정 시드. 방금 뜬 스냅숏으로
            for r in roles:
                for name in baselines.get(str(r), []):
                    ms, mines = [], []
                    for k in range(args.eval_games):
                        seed = derive(3, k // 2 if symmetric else k)
                        seats, mine = seats_for(cls, r, cands[r], name, k % 2)
                        ms.append((seed, seats))
                        mines.append(mine)
                    rr = ev.run(ms)
                    eval_matches += len(ms)
                    row[f"vs_{name}{r}"] = float(np.mean([pts[m] for m, (_s, pts, _i) in zip(mines, rr)]))
                    row[f"sc_{name}{r}"] = float(np.mean([s[m] for m, (s, _p, _i) in zip(mines, rr)]))
                    row[f"wins_{name}{r}"] = int(sum(1 for m, (_s, pts, _i) in zip(mines, rr) if pts[m] == 1.0))
                    row[f"loss_{name}{r}"] = int(sum(1 for m, (_s, pts, _i) in zip(mines, rr) if pts[m] == 0.0))
                    e2: dict = {}
                    for _s, _p, info in rr:
                        e2[info["end"]] = e2.get(info["end"], 0) + 1
                    row[f"ends_{name}{r}"] = e2
            row["rate_matches"] = rate_matches
            row["eval_matches"] = eval_matches
        row["t"] = elapsed0 + time.perf_counter() - t_start
        row["gen_s"] = time.perf_counter() - t_gen
        row["load"] = load()
        log.write(json.dumps(row) + "\n")
        n_rows += 1
        if validate:
            save_ckpt()
            keys = [k for k in row if k.startswith(("elo", "vs_", "wins_still", "best", "pts", "rungs", "champ", "draw", "neurons"))]
            print(f"gen {gen} games={train_matches} t={row['t']:.0f}s load={row['load']} " + " ".join(
                f"{k}={row[k]:.3f}" if isinstance(row[k], float) else f"{k}={row[k]}" for k in keys), flush=True)
    seconds = elapsed0 + time.perf_counter() - t_start
    summary = {"summary": {"gens": gen, "seconds": seconds, "matches": train_matches, "rate_matches": rate_matches,
                           "eval_matches": eval_matches, "all_matches": ev.matches, "world_steps": ev.steps, "sim_s": ev.sim, "nn_s": ev.nn,
                           "us_per_world_step": 1e6 * (ev.sim + ev.nn) / max(ev.steps, 1),
                           "nn_share": ev.nn / max(ev.sim + ev.nn, 1e-9), "load_end": load(), "workers": args.workers, "resumes": resumes,
                           "champ": {r: [champ[r][0], champ[r][2]] for r in roles},
                           "rungs": {r: [[g, rt] for g, _v, rt in ladders[r].rungs] for r in roles}}}
    log.write(json.dumps(summary) + "\n")
    log.flush()
    os.fsync(log.fileno())
    log.close()
    print(json.dumps(summary), flush=True)
    np.savez_compressed(out / "snaps.npz", **{f"r{r}_g{g}": v for r in roles for g, v in snaps[r].items()})
    np.savez_compressed(out / "ladder.npz", **{f"r{r}_g{g}_{k}": np.append(v, rt) for r in roles
                                                 for k, (g, v, rt) in enumerate(ladders[r].rungs)})
    np.savez_compressed(out / "champion.npz", **{f"r{r}_g{champ[r][0]}": np.append(champ[r][1], champ[r][2]) for r in roles})
    (out / "snap_elo.json").write_text(json.dumps({str(r): {str(g): e for g, e in snap_elo[r].items()} for r in roles}))
    ev.close()
    _atomic_write(out / "DONE", json.dumps(summary).encode())


def bench(game: str, opts: dict, n: int = 24) -> None:
    cls = GAMES[game]
    hidden = (16,)
    sr = cls.seat_roles()
    rng = np.random.default_rng(0)
    for kind in ("zero", "random"):
        ms = []
        for i in range(n):
            seats = []
            for s, r in enumerate(sr):
                d = BrainLayout(cls.N_IN[r], hidden, cls.N_OUT[r]).n_params
                seats.append(np.zeros(d) if kind == "zero" else rng.normal(0, 0.5, d))
            ms.append((1000 + i, seats))
        best = None
        for _ in range(3):
            res, steps, sim, nn = play(game, opts, hidden, ms)
            us = 1e6 * (sim + nn) / steps
            if best is None or us < best[0]:
                best = (us, steps, sim, nn)
        us, steps, sim, nn = best
        ends: dict = {}
        for _s, _p, info in res:
            ends[info["end"]] = ends.get(info["end"], 0) + 1
        print(f"{game} {kind}: {us:.1f} us/world-step (nn {100 * nn / (sim + nn):.0f}%), {steps / n:.0f} steps/match, "
              f"{1e3 * (sim + nn) / n:.1f} ms/match, ends {ends}, load {load()}", flush=True)


def main() -> None:
    if sys.argv[1] == "bench":
        bench(sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {})
        return
    p = argparse.ArgumentParser()
    p.add_argument("game")
    p.add_argument("--opts", default="{}")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--minutes", type=float, default=0.0)
    p.add_argument("--gens", type=int, default=0)
    p.add_argument("--pop", type=int, default=32)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--hidden", type=int, nargs="*", default=[16])
    p.add_argument("--cap", type=int, default=16)
    p.add_argument("--latest", type=float, default=0.5)
    p.add_argument("--agg", default="mean")
    p.add_argument("--pool-admit", default="all", dest="pool_admit", choices=["all", "rung", "rung+latest2"])
    p.add_argument("--learner", default="cmaes", choices=["cmaes", "grow"])
    p.add_argument("--grow-start", type=int, default=2, dest="grow_start")
    p.add_argument("--grow-every", type=int, default=10, dest="grow_every")
    p.add_argument("--games", type=int, default=0)
    p.add_argument("--validate-every", type=int, default=0, dest="validate_every")
    p.add_argument("--validate-on", default="all", dest="validate_on", choices=["all", "role0"])
    p.add_argument("--elo-base", type=float, default=1000.0, dest="elo_base")
    p.add_argument("--ref-share", type=float, default=0.0, dest="ref_share")
    p.add_argument("--snap-every", type=int, default=5, dest="snap_every")
    p.add_argument("--rate-games", type=int, default=8, dest="rate_games")
    p.add_argument("--rung-step", type=float, default=70.0, dest="rung_step")
    p.add_argument("--eval-every", type=int, default=10, dest="eval_every")
    p.add_argument("--eval-games", type=int, default=20, dest="eval_games")
    p.add_argument("--baselines", default="{}")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", required=True)
    train(p.parse_args())


if __name__ == "__main__":
    main()

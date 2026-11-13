"""P2·P3 원형: 물리와 무관한 대전 층(상대 모음, 대진표, 엘로 사다리). 숫자와 유전자 배열만 다룬다.

v7 보드게임이 그대로 쓸 수 있어야 하므로 이 파일은 pymunk도 arena도 불러오지 않는다. 난수는 쓰지 않고 시드 단어에서만 뽑는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TAG_MATCH = 5
"""시드 태그: 상대 고르기 [5, 실행 시드, 세대, 대진 번호, 역할]."""


def derive(*words: int) -> int:
    state = np.random.SeedSequence(list(words)).generate_state(2, np.uint32)
    return int(state[0]) | (int(state[1]) << 32)


# ---------------------------------------------------------------------------
# 상대 모음
# ---------------------------------------------------------------------------


@dataclass
class Pool:
    """지난 대표(스냅숏) 모음. 꽉 차면 새 절반은 그대로 두고 옛 절반에서 앞 스냅숏과 세대 간격이 가장 좁은 것을 뺀다."""

    cap: int = 16
    entries: list[tuple[int, np.ndarray]] = field(default_factory=list)

    def add(self, gen: int, genome: np.ndarray) -> None:
        self.entries.append((gen, np.array(genome, dtype=np.float64, copy=True)))
        if len(self.entries) > self.cap:
            old = len(self.entries) // 2
            gaps = [(self.entries[i][0] - self.entries[i - 1][0], i) for i in range(1, old)]
            drop = min(gaps)[1] if gaps else 0  # 간격이 같으면 더 옛것. 크기 1~3이면 가장 옛것
            del self.entries[drop]

    def pick(self, run_seed: int, gen: int, k: int, role: int, latest_share: float, ref_share: float = 0.0) -> int:
        """대진 k의 상대 번호. ref_share 확률로 0번(가만히 있는 기준 상대), 아니면 latest_share 확률로 가장 새 스냅숏,
        아니면 모음 전체에서 고르게. 시드 단어만의 함수."""
        return pick_index(len(self.entries), run_seed, gen, k, role, latest_share, ref_share)


def pick_index(n: int, run_seed: int, gen: int, k: int, role: int, latest_share: float, ref_share: float = 0.0) -> int:
    """Pool.pick과 같은 셈(PX1: 「새 칸 + 최근 둘」처럼 모음 밖 항목을 더한 목록에도 쓰려고 함수로 뺌)."""
    if n == 1:
        return 0
    u = derive(TAG_MATCH, run_seed, gen, k, role)
    if ref_share > 0.0 and ((u >> 48) & 0xFFFF) < ref_share * 0x10000:
        return 0
    if (u & 0xFFFFFF) < latest_share * 0x1000000:
        return n - 1
    return (u >> 24) % n


# ---------------------------------------------------------------------------
# 엘로
# ---------------------------------------------------------------------------


def expected(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def performance_rating(results: list[tuple[float, float, float]], prior: float = 0.5) -> float:
    """고정된 상대 점수에 대한 성적 점수(최대 가능도). results = (상대 점수, 얻은 승점 합, 판 수).

    상대마다 가상의 무승부 prior판을 더해 전승·전패에도 유한하다. Σ (n + 2·prior)·E(R - Rj) = Σ (s + prior)를 이분법으로 푼다.
    """
    lo, hi = -6000.0, 12000.0
    target = sum(s + prior for _, s, _ in results)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        got = sum((n + 2.0 * prior) * expected(mid - r) for r, _, n in results)
        if got < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def mle_ratings(points: np.ndarray, games: np.ndarray, anchor: int = 0, iters: int = 400) -> np.ndarray:
    """모두끼리 붙은 표에서 한꺼번에 구한 엘로(브래들리-테리, MM 반복). points[i, j] = i가 j에게 얻은 승점, anchor의 점수 = 0."""
    n = points.shape[0]
    w = points.sum(axis=1) + 0.5  # 약한 사전(전패 방지)
    g = np.ones(n)
    for _ in range(iters):
        denom = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j and games[i, j] > 0:
                    denom[i] += games[i, j] / (g[i] + g[j])
        g = w / np.maximum(denom, 1e-12)
        g /= g[anchor]
    return 400.0 * np.log10(g)


def league_ratings(points: np.ndarray, games: np.ndarray, anchor: int = 0, base: float = 0.0, prior: float = 0.5,
                   iters: int = 2000, tol: float = 1e-10) -> np.ndarray:
    """PX1의 리그전 엘로(DE 4.5 league_ratings): 브래들리-테리 최대 가능도를 MM 반복으로, 짝마다 가상 무승부 prior판(양쪽에 prior씩 승점,
    판 수 2 x prior)을 더해 전승·전패에도 유한하다(performance_rating의 사전과 같은 꼴). anchor의 점수 = base. 수렴(로그 강도 변화 < tol)까지 돈다.
    mle_ratings와 다른 점: 사전이 선수마다 0.5점이 아니라 붙은 짝마다이고(판 수에도 반영), 수렴을 확인한다."""
    n = points.shape[0]
    played = games > 0
    np.fill_diagonal(played, False)
    w = (points * played).sum(axis=1) + prior * played.sum(axis=1)
    gm = np.where(played, games + 2.0 * prior, 0.0)
    g = np.ones(n)
    for _ in range(iters):
        denom = (gm / (g[:, None] + g[None, :])).sum(axis=1)
        new = w / np.maximum(denom, 1e-300)
        new = new / new[anchor]
        done = np.max(np.abs(np.log(new) - np.log(g))) < tol
        g = new
        if done:
            break
    return base + 400.0 * np.log10(g)


@dataclass
class Ladder:
    """점수가 굳은 기준 상대 사다리. 0번 칸은 가만히 있는 상대(점수 0)."""

    rungs: list[tuple[int, np.ndarray, float]] = field(default_factory=list)
    step: float = 70.0
    max_play: int = 8

    def add(self, gen: int, genome: np.ndarray, rating: float) -> None:
        self.rungs.append((gen, np.array(genome, dtype=np.float64, copy=True), float(rating)))

    def top(self) -> float:
        return max(r for _, _, r in self.rungs)

    def play_set(self) -> list[int]:
        """점수를 잴 때 붙을 칸: max_play개 이하면 전부, 넘으면 새 절반 + 나머지에서 고르게."""
        n = len(self.rungs)
        if n <= self.max_play:
            return list(range(n))
        new = self.max_play // 2
        old = n - new
        keep = sorted({round(i * (old - 1) / (self.max_play - new - 1)) for i in range(self.max_play - new)})
        return keep + list(range(old, n))

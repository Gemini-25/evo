"""점프 게임 완전 정보 계획기(v2 설계 4.4, T19): 장애물 줄 전체를 알고 "언제 뛸지"를 끝에서부터 동적 계획으로 정한다.

내장 점프 게임이 "풀 수 있는" 게임인지 확인하는 기준이다(학습하는 뇌는 레이와 장애물 속도만 보므로 이보다 어렵다).

- 점프 궤적은 실제 Episode에서 잰다(쉬다가 한 번 뛰고, 다시 뛸 수 있을 때까지의 바닥 높이). 착지 직후 몸이 땅에 6cm쯤
  박혔다 올라오는 것을 감안해 다음 점프를 SLACK스텝 늦게 허용하고, 겹침 판정에 여유 margin(기본 8cm)을 둔다.
- 장애물 줄 자세는 런타임과 같은 motion.pose_at으로 계산한다(위치 번호 p = 스텝 p번 뒤).
- 계획을 실제 Episode에 열린 고리로 다시 돌린다(계획한 스텝부터 실제로 뛸 때까지 최대 REPLAY_HOLD스텝 누름).

`python tests/jump_planner.py [환경 파일] [판정 시드 수]`로 계획 있음·재생 성공·안 뜀·늘 뜀 결과를 출력한다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JUMP_ENV = ROOT / "environments" / "jump.evoenv"
RUNNER = "runner"
OBSTACLES = "obs"
HALF_WIDTH = 0.3
"""선수 상자 반폭(m). 선수는 x = 0에 선다."""
HEIGHT = 1.0
"""선수 상자 높이(m). 바닥은 y = 0(땅 윗면)."""
SLACK = 2
REPLAY_HOLD = 3
PROFILE_SEED = 424242
PROFILE_REST = 30


@dataclass(frozen=True)
class JumpProfile:
    heights: np.ndarray
    """뛴 호출 뒤 t스텝의 바닥 높이(t = 0..R-1, 0 = 뛴 호출 직전), 음수는 0. 끝에 SLACK스텝의 0을 더한 것."""
    ready_after: int
    """뛴 호출에서 다시 뛸 수 있는 호출까지 스텝(SLACK을 더하기 전)."""
    apex: float


def _jumped(ep, channel: int) -> bool:
    return ep.applied_outputs[channel] == 1.0


def measure_profile(spec) -> JumpProfile:
    """실제 Episode에서 점프 궤적을 잰다: 땅에서 PROFILE_REST스텝 쉬고, 출력 1을 계속 주어 다시 뛸 때까지."""
    from evolab.world.physics2d.episode import Episode, Trial

    ep = Episode(spec, None, Trial(PROFILE_SEED))
    ch = ep.output_channel_index[0]
    body = ep.bodies[RUNNER]
    for _ in range(PROFILE_REST):
        ep.step([0.0])
    y0 = body.position.y
    heights = [0.0]
    ep.step([1.0])
    assert _jumped(ep, ch), "the runner should jump from rest"
    heights.append(body.position.y - y0)
    while True:
        ep.step([1.0])
        if _jumped(ep, ch):
            break
        heights.append(body.position.y - y0)
    h = np.maximum(np.array(heights), 0.0)
    return JumpProfile(np.concatenate([h, np.zeros(SLACK)]), len(h), float(h.max()))


def obstacle_boxes(ep) -> tuple[object, np.ndarray]:
    """(장애물 줄 MotionPlan, 장애물마다 [x0, x1, y0, y1] 줄 좌표)."""
    geo = next(g for g in ep.geometry().bodies if g.id == OBSTACLES)
    boxes = np.array([[min(p[0] for p in s.points), max(p[0] for p in s.points),
                       min(p[1] for p in s.points), max(p[1] for p in s.points)] for s in geo.shapes])
    return geo.motion, boxes


def plan(spec, seed: int, profile: JumpProfile, margin: float = 0.08) -> tuple[bool, list[int]]:
    """(계획 있음, 뛸 호출 번호들). 호출 j에 뛰면 위치 j+1..j+R-1에서 높이 H[1..R-1]로 날고 j+R에 땅에 선다."""
    from evolab.world.physics2d.episode import Episode, Trial
    from evolab.world.physics2d.motion import pose_at

    dt = spec.file.world.dt
    n = int(round(spec.file.world.max_time / dt))
    heights = profile.heights
    r = len(heights)
    ep = Episode(spec, None, Trial(seed))
    motion, boxes = obstacle_boxes(ep)
    p_count = n + r + 2
    ox = np.array([pose_at(motion, p * dt)[0] for p in range(p_count)])
    x0 = ox[:, None] + boxes[None, :, 0]
    x1 = ox[:, None] + boxes[None, :, 1]
    xo = (x0 < HALF_WIDTH + margin) & (x1 > -HALF_WIDTH - margin)  # (위치, 장애물)

    def hits(bottom: float) -> np.ndarray:
        """선수 바닥이 bottom일 때 위치마다 부딪히는가."""
        yo = (boxes[:, 2] < bottom + HEIGHT + margin) & (boxes[:, 3] > bottom - margin)
        return (xo & yo[None, :]).any(axis=1)

    ground_ok = ~hits(0.0)
    flight_bad = np.zeros(n + 1, dtype=bool)  # 호출 j에 뛰면 비행 중에 부딪히는가
    for t in range(1, r):
        flight_bad |= hits(float(heights[t]))[t:t + n + 1]
    land = np.arange(n + 1) + r
    flight_ok = ~flight_bad & ((land >= p_count) | ground_ok[np.minimum(land, p_count - 1)])
    nxt = np.full(p_count + 1, p_count, dtype=int)  # nxt[p]: p보다 큰 첫 땅 충돌 위치
    for p in range(p_count - 1, -1, -1):
        nxt[p] = nxt[p + 1] if p + 1 >= p_count or ground_ok[p + 1] else p + 1
    last = n + r + 1
    surv = np.zeros(last + 1, dtype=bool)
    choice = np.full(last + 1, -2, dtype=int)
    for p in range(last, -1, -1):
        if p >= n or nxt[p] > n:
            surv[p], choice[p] = True, -1  # 끝까지 서 있어도 된다
            continue
        for j in range(min(nxt[p] - 1, n), p - 1, -1):  # 늦게 뛰는 것부터
            if flight_ok[j] and surv[min(j + r, last)]:
                surv[p], choice[p] = True, j
                break
    jumps: list[int] = []
    p = 0
    if surv[0]:
        while p < n and choice[p] >= 0:
            jumps.append(int(choice[p]))
            p = int(choice[p]) + r
    return bool(surv[0]), jumps


def run_open_loop(spec, seed: int, jumps: list[int] | None = None, const: float | None = None) -> tuple[bool, float]:
    """계획(뛸 호출 번호)이나 일정한 출력으로 실제 Episode를 끝까지 돌린다: (끝까지 버팀, 버틴 시간)."""
    from evolab.world.physics2d.episode import Episode, Trial

    ep = Episode(spec, None, Trial(seed))
    ch = ep.output_channel_index[0]
    planned = set(jumps or [])
    call = pending = 0
    while not ep.done:
        if call in planned:
            pending = REPLAY_HOLD
        u = const if const is not None else (1.0 if pending > 0 else 0.0)
        ep.step([u])
        call += 1
        if pending > 0:
            pending = 0 if _jumped(ep, ch) else pending - 1
    res = ep.result()
    return res.end_event == "timeout", res.time


def first_second_clear(spec, seed: int, seconds: float = 1.0) -> bool:
    """처음 seconds초 동안 장애물 상자가 서 있는 선수 상자와 겹치지 않는가(여유 없음)."""
    from evolab.world.physics2d.episode import Episode, Trial
    from evolab.world.physics2d.motion import pose_at

    dt = spec.file.world.dt
    motion, boxes = obstacle_boxes(Episode(spec, None, Trial(seed)))
    for p in range(int(round(seconds / dt)) + 1):
        ox = pose_at(motion, p * dt)[0]
        overlap = ((ox + boxes[:, 0] < HALF_WIDTH) & (ox + boxes[:, 1] > -HALF_WIDTH)
                   & (boxes[:, 2] < HEIGHT) & (boxes[:, 3] > 0.0))
        if overlap.any():
            return False
    return True


def main() -> None:
    from evolab.engine import seeds
    from evolab.spec.io import load_env

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else JUMP_ENV
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    spec = load_env(path)
    profile = measure_profile(spec)
    print(f"apex {profile.apex:.3f} m, ready again after {profile.ready_after} steps")
    feasible = replayed = never = always = 0
    for trial in seeds.judge_trials(count):
        ok, jumps = plan(spec, trial.seed, profile)
        feasible += ok
        replayed += ok and run_open_loop(spec, trial.seed, jumps)[0]
        never += run_open_loop(spec, trial.seed, const=0.0)[0]
        always += run_open_loop(spec, trial.seed, const=1.0)[0]
    print(f"{path.name}: plan {feasible}/{count}, replay {replayed}/{count}, never {never}/{count}, always {always}/{count}")


if __name__ == "__main__":
    main()

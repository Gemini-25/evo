"""트랙 자동차 손 조종기(학습 없이): 중심선 추종 + 앞 코너 곡률에 맞춘 속도.

트랙 크기·차량 수치가 "잘 달리면 2바퀴가 60초 안에 충분히 들어온다"를 만족하는지 재는 데 쓴다.
`python tests/track_pilot.py [트랙 수]`로 학습 트랙 5개와 새 트랙 여러 개의 결과를 출력한다.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from evolab import paths
from evolab.spec.io import load_env
from evolab.world.physics2d.episode import Episode, Trial


def curvature_radius(center: np.ndarray) -> np.ndarray:
    a = np.roll(center, 1, axis=0)
    b = center
    c = np.roll(center, -1, axis=0)
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = ab * bc * ca / (2.0 * cross)
    return np.where(np.isfinite(r), r, 1e9)


def drive(ep: Episode, lat_accel: float = 7.0, brake: float = 6.0, vmax: float = 15.5) -> None:
    from evolab.generators.track import progress_along

    track = ep._cps[0].track
    center = np.asarray(track.centerline)
    radius = curvature_radius(center)
    n = len(center)
    spacing = track.length / n
    hint = None
    while not ep.done:
        x, y, a = (float(v) for v in ep.snapshot().poses[0])
        v = ep.channel_values()["st.v_fwd"]
        s, hint = progress_along(track, x, y, hint)
        i = int(s / spacing) % n
        look = max(4.0, 0.6 * abs(v))
        j = (i + int(look / spacing)) % n
        tx, ty = center[j]
        err = math.atan2(ty - y, tx - x) - a
        err = (err + math.pi) % (2 * math.pi) - math.pi
        steer = max(-1.0, min(1.0, 2.5 * err))
        # 앞으로 제동 거리 안의 가장 좁은 코너에 맞춘 속도
        horizon = int((v * v / (2 * brake) + 10.0) / spacing)
        target = vmax
        for k in range(0, horizon, 2):
            idx = (i + k) % n
            dist = k * spacing
            v_corner = math.sqrt(lat_accel * radius[idx])
            target = min(target, math.sqrt(v_corner**2 + 2 * brake * dist))
        throttle = max(-1.0, min(1.0, 0.8 * (target - v)))
        ep.step([throttle, steer])


def main(extra: int = 20) -> None:
    spec = load_env(paths.builtin_env_dir() / "track_car.evoenv")
    rows = []
    trials = [Trial(1, k) for k in range(spec.track_set_size or 0)] + [Trial(1000 + k, None) for k in range(extra)]
    for t in trials:
        ep = Episode(spec, None, t)
        drive(ep)
        r = ep.result()
        rows.append((t, round(ep.gen_values["gen.track.length"], 1), r.end_label, round(r.time, 1), round(r.score, 1)))
    for row in rows:
        print(*row)
    done = [r for r in rows if r[2] == "완주"]
    print(f"완주 {len(done)}/{len(rows)}, 시간 중앙값 {np.median([r[3] for r in done]) if done else '-'}초, "
          f"최대 {max([r[3] for r in done]) if done else '-'}초")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)

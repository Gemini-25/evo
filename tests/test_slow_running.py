"""6.2.0 묶음 PM의 느린 테스트(-m slow): 내장 달리기가 원형 측정과 같은 산술로 배우는가.

T8은 물리 칸이 같음을 본다. 여기서는 그 결과로 학습 자체가 같은지 본다(설계 7.1 M2의 ①, D59: 시드 1·2·3은
「구현이 원형의 산술을 되풀이하는가」의 확인으로만 쓴다). 원형 런타임(균형 보조 패치가 든 scratch 사본)에서 원형 파일로
`train --seed S --max-generations 100 --no-observe`(예선 없음)를 돌린 stats.csv의 결정적 열(gen_seconds 앞의 39열, 머리 줄 포함)
sha256과 100세대의 챔피언 검증 점수를, 같은 명령을 내장 running.evoenv로 돌린 결과와 견준다. 기준값은 묶음 PX의
되풀이 확인(px/PX0/repro, 원형 eng/runs/p1의 처음 100세대와 바이트까지 같았던 것)에서 옮겨 적었다.

환경 이름·해시는 달라도(D29) block_*·end_* 열 이름이 같으므로 결정적 열 전체가 같아야 한다.
"""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNING = ROOT / "environments" / "running.evoenv"

PROTOTYPE_100 = {
    1: ("d6637e14ecd0f4408f01c7aa2bf915fec094244beaeb90e976e4d11f2918d015", "9.683942647835934"),
    2: ("255cc56b47d20773680e026492b2998504b56c815115e8bd8f20cfb7111cf452", "6.1024775404714555"),
    3: ("37e3a4ae7a685f497c5f6ab543b95820fcef4695fbccbb59e6bf08cc128f6b68", "8.381332520402548"),
}
"""시드 → (결정적 열 sha256, 100세대의 champion_score 글). 원형의 9,600회 검증 최고점 9.68·6.10·8.38(설계 7.1 M2)."""


def stats_hash(path: Path) -> tuple[str, list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    cut = rows[0].index("gen_seconds")
    text = "\n".join(",".join(r[:cut]) for r in rows)
    body = [dict(zip(rows[0], r, strict=True)) for r in rows[1:]]
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), body


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1])
def test_builtin_running_repeats_the_prototype_learning(seed: int, tmp_path: Path) -> None:
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    done = subprocess.run(
        [sys.executable, "-m", "evolab", "train", str(RUNNING), "--seed", str(seed), "--qualify", "1",
         "--max-generations", "100", "--no-observe", "--workers", "1", "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=1800, env=env, cwd=ROOT)
    assert done.returncode == 0, done.stderr[-2000:]
    (stats,) = list(tmp_path.glob("*/stats.csv"))
    digest, rows = stats_hash(stats)
    want_hash, want_champion = PROTOTYPE_100[seed]
    assert len(rows) == 100 and rows[-1]["evaluations"] == "9600"
    assert rows[-1]["champion_score"] == want_champion
    assert digest == want_hash

"""6.2.0 묶음 PM의 느린 테스트(-m slow): 내장 달리기가 원형 측정과 같은 산술로 배우는가(설계 0.0 LD1의 M2 ①′).

T8은 물리 칸이 같음을 본다. 여기서는 그 결과로 학습 자체가 같은지 본다. 원형 런타임(균형 보조 패치가 든 scratch 사본
px/PX0b/wtb)에서 측정 파일(넘어짐 감점 −5, tests/data/presets/running_measured.evoenv)로
`train --seed 5 --qualify 3 --qualify-trials 9600 --max-evaluations 48000 --workers 1`을 돌린 실행 px/PX0b/runs/f5_s5의
stats.csv 결정적 열(gen_seconds 앞의 39열, 머리 줄 포함) sha256과 마지막 줄의 champion_score를, 내장 running.evoenv로
**권장 설정만으로**(예선 칸은 editor.train에서: 명령줄에는 예산과 워커만) 돌린 결과와 견준다.

환경 이름·해시는 달라도(D29) block_*·end_* 열 이름이 같으므로 결정적 열 전체가 같아야 한다. 이 컨테이너(인텔)와 같은 CPU
계열에서만 바이트가 같다(설계 0.0b: AMD 러너는 3세대부터 1e-9쯤 다름). 그래서 기준과 다른 CPU에서는 처음 100세대만 본다.
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

PROTOTYPE_F5_S5 = ("44e9fb8b6c5120a32fb2d804b2b85a0848c0b407af7bc0495cf43fcec329de4e", 500, "10.28986483834153")
"""px/PX0b/runs/f5_s5: (결정적 열 sha256, 줄 수, 마지막 줄의 champion_score 글). 판정 시드 100개: 중앙값 12.0m · 넘어짐 4%."""

PROTOTYPE_F5_S5_100 = "4145a9ab7cc813ca760f0a45f2df6540cc6eff20dea54881d0da8446a80dcc4a"
"""같은 실행의 처음 100줄(머리 줄 포함)의 결정적 열 sha256. CPU가 달라도 처음 몇 세대는 같다."""


def stats_hash(path: Path) -> tuple[str, list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    cut = rows[0].index("gen_seconds")
    text = "\n".join(",".join(r[:cut]) for r in rows)
    body = [dict(zip(rows[0], r, strict=True)) for r in rows[1:]]
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), body


def _same_cpu_family() -> bool:
    try:
        return "GenuineIntel" in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


@pytest.mark.slow
def test_builtin_running_repeats_the_prototype_learning(tmp_path: Path) -> None:
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    done = subprocess.run(
        [sys.executable, "-m", "evolab", "train", str(RUNNING), "--seed", "5", "--max-evaluations", "48000",
         "--workers", "1", "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=7200, env=env, cwd=ROOT)
    assert done.returncode == 0, done.stderr[-2000:]
    (stats,) = list(tmp_path.glob("*/stats.csv"))
    digest, rows = stats_hash(stats)
    want_hash, want_rows, want_champion = PROTOTYPE_F5_S5
    assert len(rows) == want_rows and rows[-1]["evaluations"] == "48000"
    with stats.open(encoding="utf-8-sig", newline="") as f:
        head = list(csv.reader(f))[:101]
    cut = head[0].index("gen_seconds")
    prefix = hashlib.sha256("\n".join(",".join(r[:cut]) for r in head).encode("utf-8")).hexdigest()
    assert prefix == PROTOTYPE_F5_S5_100
    if _same_cpu_family():
        assert rows[-1]["champion_score"] == want_champion
        assert digest == want_hash

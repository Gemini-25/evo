"""M0: 자체 점검과 그 안의 결정성 확인."""

from pathlib import Path

import numpy as np

from evolab import selftest
from evolab.app import strings as S


def test_simulation_is_deterministic_and_seed_dependent() -> None:
    a = selftest.simulate(7)
    b = selftest.simulate(7)
    c = selftest.simulate(8)
    assert a.tobytes() == b.tobytes()
    assert not np.array_equal(a, c)
    assert np.all(np.isfinite(a))


def test_simulation_ignores_process_history() -> None:
    # 다른 시드를 먼저 여러 번 돌려도 결과가 같아야 한다(워커 재사용의 전제).
    fresh = selftest.simulate(3)
    for s in range(20, 30):
        selftest.simulate(s)
    assert selftest.simulate(3).tobytes() == fresh.tobytes()


def test_full_selftest_passes(tmp_path: Path) -> None:
    # 한글이 들어간 경로에 기록을 쓴다(윈도우 사용자 폴더가 한글인 경우를 흉내).
    log = tmp_path / "점검 기록" / "selftest.log"
    code = selftest.run_selftest(log_path=log, workers=2, check_gui=True)
    report = log.read_text(encoding="utf-8")
    assert code == 0, report
    assert S.SELFTEST_SUMMARY_OK in report
    assert report.count(f"[{S.SELFTEST_PASS}]") == 5

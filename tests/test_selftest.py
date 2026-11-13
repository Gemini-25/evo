"""M0: 자체 점검과 그 안의 결정성 확인."""

import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pytest

from evolab import paths, selftest
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


def _crashing_worker(tasks: object, results: object) -> None:
    os._exit(3)


def test_dead_worker_is_reported_immediately() -> None:
    # 워커가 바로 죽으면 시간 초과까지 기다리지 않고 종료 코드와 함께 실패해야 한다.
    pool = selftest._MiniPool(mp.get_context("spawn"), 2, target=_crashing_worker)
    t0 = time.monotonic()
    try:
        with pytest.raises(selftest.SelfTestError, match="3"):
            pool.map([1, 2, 3])
    finally:
        pool.close()
    assert time.monotonic() - t0 < 30


def test_full_selftest_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 한글이 들어간 경로에 기록을 쓴다(윈도우 사용자 폴더가 한글인 경우를 흉내).
    # 「창 만들기」 점검이 사용자의 진짜 설정·데이터 폴더를 건드리지 않게 둘 다 임시 폴더로(v3 설계 D5).
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "데이터"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "상태"))
    log = tmp_path / "점검 기록" / "selftest.log"
    code = selftest.run_selftest(log_path=log, workers=2, check_gui=True)
    report = log.read_text(encoding="utf-8")
    assert code == 0, report
    assert S.SELFTEST_SUMMARY_OK in report
    # v1~v2.1 점검 7개 + v3.0 점검 넷(도움말 쪽, 기록 폴더와 묶음, Qt 한국어, 처음 실행 안내: 설계 3.2 「자체 점검 항목 넷 늘어남」)
    # v3.1: 「내보내기」 점검 하나 더(설계 D59), v6.0: 「빠른 계산 길」 점검 하나 더(v6 설계 D19), v6.1: 「자라는 뇌」 점검 하나
    # 더(v6 설계 8.4 W2), 6.2.0: 「새 프리셋」 점검 하나 더(최종 설계 T40)
    assert report.count(f"[{S.SELFTEST_PASS}]") == 15
    assert f"[{S.SELFTEST_PASS}] {S.SELFTEST_CHECK_PRESETS} (" in report
    assert f"[{S.SELFTEST_PASS}] {S.SELFTEST_CHECK_FASTPATH} (" in report and S.SELFTEST_FASTPATH_SAME in report
    assert f"[{S.SELFTEST_PASS}] {S.SELFTEST_CHECK_GROW} (" in report and S.SELFTEST_GROW_SAME in report


def test_temporary_user_dirs_keeps_error_logs_when_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """창·안내 점검의 임시 상태 폴더는 끝나면 지워진다: keep_logs를 주면 그 전에 오류 기록을 진짜 기록 폴더로 옮긴다(검토 수정).
    다음 실행에 오류가 없으면 지난번 것을 지운다(지난 실행의 오류를 이번 것으로 읽지 않게)."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    kept = tmp_path / "state" / "logs" / "selftest-tour-errors.jsonl"
    with selftest.temporary_user_dirs(keep_logs="tour") as base:
        assert paths.user_state_dir() == base / "state"
        paths.log_dir().mkdir(parents=True)
        (paths.log_dir() / "errors.jsonl").write_text('{"kind": "uncaught"}\n', encoding="utf-8")
    assert not base.exists() and paths.user_state_dir() == tmp_path / "state"
    assert kept.read_text(encoding="utf-8") == '{"kind": "uncaught"}\n'
    with selftest.temporary_user_dirs(keep_logs="tour"):
        pass
    assert not kept.exists()
    with selftest.temporary_user_dirs() as base:  # 이름표가 없으면 옮기지 않는다
        paths.log_dir().mkdir(parents=True)
        (paths.log_dir() / "errors.jsonl").write_text("x\n", encoding="utf-8")
    assert not list((tmp_path / "state" / "logs").glob("selftest-*"))

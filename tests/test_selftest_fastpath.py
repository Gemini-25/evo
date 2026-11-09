"""v6.0 자체 점검 「빠른 계산 길」 (v6 설계 8.1 T18, D19, 5.6, I 묶음).

점검은 내장 예제 여섯의 같은 평가를 물리 엔진 C 함수를 바로 부르는 길과 보통 길(공개 pymunk API)로 두 번 돌려 결과 칸을
바이트로 비교한다. 고정한 pymunk에서는 통과하고, 길이 꺼져 있으면 「꺼짐」 안내로 통과하며(CI에서는 실패), 빠른 길이 결과를
바꾸면 실패한다. 점검이 끝나면(실패해도) 모듈 깃발 _CP_FAST가 처음 값으로 돌아온다.

v6.0 검토: 「꺼짐」 판단(episode.fastpath_in_use)은 물체·공간·모터 포인터를 모두 본다(bench 줄과 같은 판단). 포인터가 일부만
없는 섞인 길도 결과가 바이트 같다.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from evolab import paths, selftest
from evolab.app import strings as S
from evolab.engine import evaluate as evaluate_mod
from evolab.engine.evaluate import evaluate
from evolab.engine.trainer import training_trials
from evolab.policy.genome import layout_for
from evolab.spec.io import load_env
from evolab.world.physics2d import episode as episode_mod
from evolab.world.physics2d.episode import Episode


@pytest.fixture
def user_mode(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """사용자 PC처럼: 「꺼짐」을 실패로 보는 CI 표시를 지운다(이 테스트가 GitHub Actions에서 돌아도)."""
    monkeypatch.delenv(selftest.FASTPATH_REQUIRED_ENV, raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    return monkeypatch


def test_fastpath_check_is_listed_after_the_examples_with_and_without_gui() -> None:
    for gui in (True, False):
        names = [n for n, _ in selftest.selftest_checks(2, check_gui=gui)]
        k = names.index(S.SELFTEST_CHECK_FASTPATH)
        assert names[k - 1] == S.SELFTEST_CHECK_BUNDLE and names[k + 1] == S.SELFTEST_CHECK_EXPORT
    assert dict(selftest.selftest_checks(2, check_gui=False))[S.SELFTEST_CHECK_FASTPATH] is selftest._check_fastpath


def test_fastpath_check_passes_on_every_builtin_example(user_mode: pytest.MonkeyPatch) -> None:
    """T18: 고정한 pymunk(uv.lock)에서 빠른 길이 켜져 있고 여섯 예제 모두 두 길의 결과가 같다(안내 줄 + 예제마다 한 줄)."""
    assert episode_mod._CP_FAST
    r = selftest._run_check(S.SELFTEST_CHECK_FASTPATH, selftest._check_fastpath)
    assert r.ok, r.detail
    lines = r.detail.splitlines()
    assert lines[0] == S.SELFTEST_FASTPATH_SAME and lines[1].startswith("pymunk ")
    pairs = selftest.FASTPATH_GENOMES * selftest.FASTPATH_TRIALS
    assert [line.split(":")[0] for line in lines[2:]] == list(selftest.REQUIRED_EXAMPLES)
    assert all(f"pairs={pairs}, steps=" in line for line in lines[2:])
    assert episode_mod._CP_FAST
    # 같은 점검을 다시 돌려도 같은 글(평가는 결정적이고 깃발은 처음 값으로 돌아온다)
    assert selftest._check_fastpath() == r.detail


def test_fastpath_check_says_off_when_the_c_functions_are_missing(user_mode: pytest.MonkeyPatch) -> None:
    """T18: 이 pymunk 판에 내부 함수가 없어 _CP_FAST가 거짓이면 「꺼짐」 안내로 통과(결과는 같고 느릴 뿐)."""
    user_mode.setattr(episode_mod, "_CP_FAST", False)
    r = selftest._run_check(S.SELFTEST_CHECK_FASTPATH, selftest._check_fastpath)
    assert r.ok, r.detail
    assert r.detail.splitlines()[0] == S.SELFTEST_FASTPATH_OFF and r.detail.splitlines()[1].startswith("pymunk ")
    assert episode_mod._CP_FAST is False  # 꺼진 판에서는 깃발을 켜지 않는다


@pytest.mark.parametrize("helper", ["_body_c", "_space_c", "_constraint_c"])
def test_fastpath_check_says_off_when_episodes_fall_back_to_the_public_api(user_mode: pytest.MonkeyPatch,
                                                                         helper: str) -> None:
    """_CP_FAST는 참이어도 pymunk 내부 이름이 바뀌어 Episode가 포인터(물체·공간·모터)를 얻지 못하면(공개 API 길) 역시 「꺼짐」.
    모터 포인터만 없어도(v6.0 검토: 예전 판단은 물체·공간만 봤다) 모터가 있는 예제(언덕차·바퀴 차)의 길이 꺼진다."""
    user_mode.setattr(episode_mod, helper, lambda obj: None)
    r = selftest._run_check(S.SELFTEST_CHECK_FASTPATH, selftest._check_fastpath)
    assert r.ok and r.detail.splitlines()[0] == S.SELFTEST_FASTPATH_OFF


def test_fastpath_probe_needs_motor_pointers_only_where_there_are_motors(user_mode: pytest.MonkeyPatch) -> None:
    lander = load_env(paths.builtin_env_dir() / "lander.evoenv")
    hill = load_env(paths.builtin_env_dir() / "hill_car.evoenv")
    assert episode_mod.fastpath_in_use(lander) and episode_mod.fastpath_in_use(hill)
    assert selftest._fastpath_in_use(hill)
    user_mode.setattr(episode_mod, "_constraint_c", lambda constraint: None)
    assert episode_mod.fastpath_in_use(lander)  # 모터가 없다
    assert not episode_mod.fastpath_in_use(hill) and not selftest._fastpath_in_use(hill)


# ---------------------------------------------------------------------------
# 섞인 길: _CP_FAST는 참인데 포인터 일부만 없을 때(pymunk 내부 이름이 일부 바뀐 판)도 결과가 같다
def _some(real: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """부를 때마다 번갈아 진짜 포인터와 None(물체마다 다른 길)."""
    calls = itertools.count()
    return lambda obj: real(obj) if next(calls) % 2 else None


MIXES: dict[str, Callable[[], dict[str, Callable[[Any], Any]]]] = {
    "no_body": lambda: {"_body_c": lambda obj: None},
    "some_body": lambda: {"_body_c": _some(episode_mod._body_c)},
    "no_motor": lambda: {"_constraint_c": lambda obj: None},
    "no_space": lambda: {"_space_c": lambda obj: None},
    "some_body_no_motor": lambda: {"_body_c": _some(episode_mod._body_c), "_constraint_c": lambda obj: None},
}


@pytest.mark.parametrize("name", selftest.REQUIRED_EXAMPLES)
def test_mixed_fast_and_public_paths_give_the_same_results(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """v6.0 검토: C 함수 길과 공개 API 길이 한 Episode 안에서 섞여도(물체 포인터 없음·일부만·모터 없음·공간 없음) 모든 결과 칸이
    바이트 같다(T5·T6은 모두 빠른 길과 모두 공개 길만 비교했다. 섞인 길은 _agent_raw가 None을 쥐어 깨진 적이 있다: 자체 점검
    T18이 찾음). 자체 점검과 같은 평가(무작위 유전자 × 학습 시험, 궤적·관찰 섞음)."""
    spec = load_env(paths.builtin_env_dir() / name)
    layout = layout_for(spec)
    rng = np.random.default_rng(selftest.FASTPATH_SEED)
    genomes = layout.initial()[None, :] + 0.5 * rng.standard_normal((selftest.FASTPATH_GENOMES, layout.dim))
    trials = training_trials(spec, selftest.FASTPATH_SEED, 0, selftest.FASTPATH_TRIALS)[:selftest.FASTPATH_TRIALS]
    pairs = [(i, t) for i in range(selftest.FASTPATH_GENOMES) for t in trials]
    record = [k % 3 == 0 for k in range(len(pairs))]
    observe = [k % 2 for k in range(len(pairs))]
    ref = selftest._eval_fingerprint(evaluate(spec, layout, genomes, pairs, record=record, observe=observe))
    for mix, helpers in MIXES.items():
        with monkeypatch.context() as mp:
            for helper, fn in helpers().items():
                mp.setattr(episode_mod, helper, fn)
            ep = Episode(spec, None, trials[0])  # 이 섞인 길을 정말 지나는가
            for _ in range(2):
                ep.step([0.0] * len(spec.outputs))
            raws = list(ep._raw.values())
            if mix == "no_body":
                assert all(r is None for r in raws) and ep._agent_raw is None and ep._raw_space is not None
            elif mix.startswith("some_body") and len(raws) > 1:
                assert any(r is None for r in raws) and any(r is not None for r in raws)
            if "no_motor" in mix and ep._motors:
                assert ep._motor_raw is None
            if mix == "no_space":
                assert ep._raw_space is None and all(r is not None for r in raws)
            got = selftest._eval_fingerprint(evaluate(spec, layout, genomes, pairs, record=record, observe=observe))
        assert [n for n, _v in got] == [n for n, _v in ref], mix
        assert [n for (n, a), (_m, b) in zip(got, ref, strict=True) if a != b] == [], mix


@pytest.mark.parametrize("marker", [(selftest.FASTPATH_REQUIRED_ENV, "1"), ("GITHUB_ACTIONS", "true")])
def test_fastpath_off_is_a_failure_in_ci(user_mode: pytest.MonkeyPatch, marker: tuple[str, str]) -> None:
    """설계 8.6: 꺼지면 사용자에게는 안내, CI에서는 실패(묶은 exe에 pymunk._chipmunk_cffi가 빠진 것을 W2가 잡게)."""
    user_mode.setenv(*marker)
    assert selftest.fastpath_required()
    user_mode.setattr(episode_mod, "_CP_FAST", False)
    r = selftest._run_check(S.SELFTEST_CHECK_FASTPATH, selftest._check_fastpath)
    assert not r.ok and r.detail == S.SELFTEST_FASTPATH_OFF


def test_fastpath_check_fails_when_the_fast_path_changes_a_result(user_mode: pytest.MonkeyPatch) -> None:
    """T18: 빠른 길만 결과를 바꾸면(둘째 스텝부터 바로 부르는 cpSpaceStep의 dt를 아주 조금 바꿈) 다름으로 실패하고, 어느
    예제의 어느 칸인지 적는다. 깃발은 처음 값으로 돌아온다."""
    real = episode_mod._CP_SPACE_STEP
    user_mode.setattr(episode_mod, "_CP_SPACE_STEP", lambda raw, dt: real(raw, dt * (1.0 + 1e-9)))
    r = selftest._run_check(S.SELFTEST_CHECK_FASTPATH, selftest._check_fastpath)
    assert not r.ok
    prefix = S.SELFTEST_FASTPATH_DIFF.split("{detail}")[0]
    assert r.detail.startswith(prefix) and f"{selftest.REQUIRED_EXAMPLES[0]}: " in r.detail and "scores" in r.detail
    assert episode_mod._CP_FAST is True


def test_fastpath_check_restores_the_flag_when_the_public_run_breaks(user_mode: pytest.MonkeyPatch) -> None:
    """보통 길 평가가 도중에 예외를 내도 _CP_FAST는 참으로 돌아온다(뒤 점검·같은 프로세스의 학습이 빠른 길을 쓴다)."""
    real = evaluate_mod.evaluate
    seen: list[bool] = []

    def flaky(*args: Any, **kwargs: Any) -> Any:
        seen.append(episode_mod._CP_FAST)
        if not episode_mod._CP_FAST:
            raise RuntimeError("public path broke")
        return real(*args, **kwargs)

    user_mode.setattr(evaluate_mod, "evaluate", flaky)
    r = selftest._run_check(S.SELFTEST_CHECK_FASTPATH, selftest._check_fastpath)
    assert not r.ok and "public path broke" in r.detail
    assert seen == [True, False] and episode_mod._CP_FAST is True

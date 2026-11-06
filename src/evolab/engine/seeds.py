"""시드 공간 (명세: 결정성).

맨 앞 단어가 용도 태그라 공간끼리 겹치지 않는다.
- 학습: [0, 실행 시드, 세대, 시험 번호]. 개체 번호는 넣지 않는다(같은 세대 개체는 같은 조건).
- 검증(챔피언 선정): [1, k], 비교 평가(비교 그래프·표): [2, k], 판정(완료 기준): [3, k]. 실행 시드와 무관한 고정 세트.
- 학습 방법 전용 난수: [4, 실행 시드].
"""

from __future__ import annotations

import numpy as np

from evolab.world.physics2d.episode import Trial

SEED_SCHEME_VERSION = 1

TAG_TRAIN = 0
TAG_VALIDATION = 1
TAG_COMPARE = 2
TAG_JUDGE = 3
TAG_LEARNER = 4

VALIDATION_SET = 20
COMPARE_SET = 32
JUDGE_SET = 100


def derive(*words: int) -> int:
    """정수 단어들에서 64비트 시드를 만든다. 같은 단어면 어느 PC·프로세스에서나 같다."""
    if any(w < 0 for w in words):
        raise ValueError(words)
    state = np.random.SeedSequence(list(words)).generate_state(2, np.uint32)
    return int(state[0]) | (int(state[1]) << 32)


def train_trials(run_seed: int, generation: int, count: int) -> list[Trial]:
    """학습 시험 목록. 시험 번호 k는 트랙 세트가 있는 환경(M4)에서 트랙 번호가 된다."""
    return [Trial(derive(TAG_TRAIN, run_seed, generation, k)) for k in range(count)]


def fixed_trials(tag: int, count: int) -> list[Trial]:
    return [Trial(derive(tag, k)) for k in range(count)]


def validation_trials(count: int = VALIDATION_SET) -> list[Trial]:
    return fixed_trials(TAG_VALIDATION, count)


def compare_trials(count: int = COMPARE_SET) -> list[Trial]:
    return fixed_trials(TAG_COMPARE, count)


def judge_trials(count: int = JUDGE_SET) -> list[Trial]:
    return fixed_trials(TAG_JUDGE, count)


def display_trial() -> Trial:
    """전시 시드: 타임라인 재생, 1세대 vs 지금, 챔피언 영상(검증 세트 0번)."""
    return validation_trials(1)[0]


def learner_seed(run_seed: int) -> int:
    """학습 방법 전용 RandomState 시드(32비트)."""
    return derive(TAG_LEARNER, run_seed) & 0xFFFFFFFF

"""Trainer: 세대 루프 (명세: 학습 방법, 결정성, 저장).

한 세대:
1. 학습 방법에서 후보를 받는다(ask).
2. 같은 세대 공통 시험 조건으로 모든 후보를 시험한다(시드 = 실행 시드, 세대, 시험 번호).
3. 시험 점수를 집계해 학습 방법에 돌려준다(tell).
4. 지금 학습 방법의 학습 시험이 일정 횟수 쌓일 때마다(검증 주기는 학습 방법별 시험 수 기준) 후보(학습 점수 상위
   몇 개 + 학습 방법의 추천해)를 검증 세트로 다시 평가해 챔피언을 고른다. 정체·목표 점수 판정은 검증 점수 기준이다.
5. 세대 끝에 학습 방법을 바꿀지 정한다(v1.1): 예선 출전이 끝났거나, 지금 학습 방법의 검증 최고점이 정해 둔 학습 시험
   동안 오르지 않았으면(정체하면 자동 재시작) 새 학습 방법으로 바꾼다. 챔피언과 기록은 그대로 이어진다.
6. 통계, 세대 대표, 챔피언, 체크포인트를 남긴다(체크포인트에는 바꾼 뒤 상태가 담긴다).

결과는 워커 수와 무관하고, 체크포인트에서 이어가면 끊지 않은 실행과 같다(경과 시간 열만 다르다).

관찰 기록(v1.2 「보는 재미」, 설계 D1·D3~D13): 관찰을 켜면(저장소가 있으면 기본) 세대마다 집단 유전자(float16),
개체별 점수·시험 결과·몸 수치, 행동 분포, 시험 0번 궤적 밀도를 세대 대표 파일에 obs_ 키로 함께 쓴다
(engine/observe.summarize, 형식은 runs/observe.py). 학습 경로는 이 기록을 읽지 않고(이어하기는 체크포인트만,
재현 묶음에는 챔피언 파일이 없다), 워커 관찰기는 상태를 읽기만 하며, 요약은 이미 계산된 값으로만 만든다. 그래서 통계,
챔피언, 세대 대표 유전자, 체크포인트의 학습 상태는 관찰을 켜도 끄도 비트 단위로 같다(pycma 객체 피클 안의 벽시계
타이머는 원래 실행마다 다르다). 요약이 실패해도(코드 결함) 학습은 이어지고 그 세대 대표는 관찰 키 없이 써진다
(observe_problem에 남김: 세계 범위만 못 만든 실패와 요약 실패를 따로, 무거운 쪽으로 한 번씩).

구조 뒤집기(v2.1, 설계 4.12·D41): 유전 알고리즘·담금질은 ga_structure_flip·sa_structure_flip 확률로 있음 유전자
(GenomeLayout.on_genes)의 부호를 뒤집는다(learners/structure.py). 데이터클래스 기본값은 0(끔)이라 키가 없는 예전 설정·상태는
v2.0과 비트 단위로 같게 돌고, 새 실행은 환경에 있음 유전자가 있을 때만 for_new_run(spec)이 STRUCTURE_FLIP_NEW_RUN을 넣는다.
확률 0이거나 있음 유전자가 없으면 make_learner가 학습 방법 설정에 구조 칸을 넘기지 않는다(상태도 v2.0과 같은 키).

자라는 뇌(v6.1, v6 설계 4.16·D31~D35): method = "grow"면 learners/grow.GrowingCMAES(CMA-ES를 켜진 좌표에, 정해진 세대마다 은닉층
뉴런 두 배, 유전자는 늘 틀 길이). grow_start·grow_every는 그 실행의 설정에만 쓴다: to_dict()가 method != "grow"면 GROW_KEYS를 지워
크기 고정 실행의 config 사전(run.json, 체크포인트, 재현 묶음)이 v6.0과 바이트 같다. 통계 열 GROW_COLUMNS(brain_neurons·brain_links·
grow)는 자라는 실행만 switch 바로 뒤에 쓴다(크기 고정 실행의 열 머리는 v6.0과 같음). 학습 방법이 grow가 아니면 이 파일의 다른
길은 한 줄도 바뀌지 않는다. 옛 프로그램(v3~v6.0)은 method = "grow"를 모르므로 이어 가기·재현을 거절한다(설계 2.4).
자라는 실행의 통계 열 diversity는 켜진 좌표만의 표준편차 평균이다(학습 방법 stats()["diversity"]): 꺼진 좌표(늘 0)까지 평균하면
자랄 때마다 값이 두 배쯤 뛰고 「탐색 폭」(sigma)과 단위가 어긋난다. 크기 고정 실행은 v6.0 식(모든 유전자) 그대로다.
last_growth는 자란 학습 방법이 다음 세대에도 쓰일 때만 남긴다: 자란 세대 끝에 자동 재시작·예선 다음 출전으로 그 학습 방법을
버리면(예선에서 다른 출전자가 통과해도) None이다(통계 열 grow는 그 세대 끝에 자란 사실 그대로 1).

조립 실패(v2.0, 설계 4.9·D22): 몸 수치 때문에 조립되지 않는 (개체, 시험) 쌍은 평가가 점수 NaN으로 돌려준다(evaluate.py).
집계 점수는 NaN이 그대로 퍼지고(그 개체 = 실패), 학습 방법은 tell에서 sanitize_scores(유한한 최솟값 − 1)로 바꾼다. 통계,
세대 대표, 관찰 기록, 화면에 보내는 점수는 같은 규칙으로 바꾼 값(agg_s)이라 실패 개체는 늘 그 세대 꼴찌다. 검증 점수가
NaN인 후보는 −∞로 보아 챔피언이 될 수 없다. 몸 유전자가 있는 환경에서 처음 몸으로도 조립되지 않으면 Trainer를 만들 때
(그리고 환경을 바꿀 때) 거절한다(check_initial_body). 실패(NaN)가 없으면 agg_s는 agg 그대로의 사본이라 결과가 v1.2와
비트 단위로 같다.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Final, Protocol

import numpy as np

from evolab.engine import observe as observing
from evolab.engine import seeds
from evolab.engine.evaluate import OBSERVE_ACTIONS, OBSERVE_POSITIONS, EvalResult, mean_of_built
from evolab.learners.base import Learner, sanitize_scores
from evolab.policy.genome import GenomeLayout, layout_for
from evolab.spec.compiled import EnvSpec
from evolab.spec.io import compile_env, env_hash
from evolab.world.physics2d.episode import Episode, EpisodeBuildError, Trial

CHECKPOINT_FORMAT = 2
"""2(v1.1): 학습 방법 기록(current), 만든 학습 방법 수, 자동 재시작 수, 예선 상태, 보관 출전자를 더했다."""
READ_FORMATS = (1, 2)
"""읽을 수 있는 체크포인트 형식. 1(v1.0)은 학습 방법 0 하나로 읽는다(자동 재시작·예선은 끈 실행)."""

RESTART_FROM: tuple[str, ...] = ("fresh", "champion", "alternate")
"""자동 재시작 때 새 학습 방법의 출발점(설정 restart_from 값, 화면 선택 순서)."""
MAX_QUALIFY_ENTRANTS = 8
NEW_RUN_DEFAULTS: dict[str, Any] = {
    "restart_patience": None,  # 끔. 실험(V11 R1)으로 확정: 켜면 권장 20,000(RECOMMENDED_RESTART_PATIENCE)
    "restart_from": "fresh",  # 실험(R2)으로 확정
    "qualify_entrants": 1,  # 1 = 예선 없음. 실험(R4)으로 확정
    "qualify_trials": 12000,  # 예선을 켤 때의 권장값. 실험(R4)으로 확정
}
"""새 실행(실험실 폼, 명령줄 train, 반복 실행)에만 넣는 권장값. 데이터클래스 기본값은 '끔'이라
예전 run.json, 체크포인트, 재현 묶음처럼 키가 없으면 v1.0과 같게 돈다."""

RECOMMENDED_RESTART_PATIENCE = 20000
"""자동 재시작을 켤 때의 권장 P(시험 수). 실험실 칸의 처음 값과 명세 A25."""

STRUCTURE_FLIP_NEW_RUN: Final = 0.05
"""구조 뒤집기 확률의 새 실행 권장값(v2.1, 설계 D41). 환경에 있음 유전자가 있을 때만 for_new_run(spec)이 GA·담금질
둘 다에 넣는다(실험실 폼, 명령줄 train, 반복 실행). NEW_RUN_DEFAULTS와 달리 환경에 따라 정해진다."""

PRESET_TRAIN_KEYS: Final = ("qualify_entrants", "qualify_trials", "restart_patience", "restart_from")
"""프리셋이 권할 수 있는 새 실행 설정 키(6.2.0, 설계 D31): 환경 파일의 editor.train에서 for_new_run(spec)이 이 키만 받는다.
editor는 해시·런타임 밖이라 이 값을 고쳐도 환경 버전이 그대로다. 이 칸을 읽는 곳은 for_new_run 하나뿐이다."""

SWITCH_NONE, SWITCH_RESTART_FRESH, SWITCH_RESTART_CHAMPION, SWITCH_QUALIFY_NEXT, SWITCH_QUALIFY_PICK = 0, 1, 2, 3, 4
"""통계 열 switch 값: 이 세대 끝에 일어난 학습 방법 교체(0 없음, 1 처음 상태에서 새로 시작, 2 챔피언에서 새로 시작,
3 예선 다음 출전, 4 예선 끝)."""
LEARNER_COLUMNS = ("learner", "learner_trials", "learner_best", "learner_stall", "restarts", "qualifying", "switch")
"""학습 방법 기록 통계 열(v1.1). 결정적이라 TIME_COLUMNS에 넣지 않는다. 옛 열 머리로 시작한 실행에는 쓰이지 않는다."""
METHODS: Final = ("cmaes", "ga", "sa", "grow")
"""학습 방법 id(TrainConfig.method 값). "grow"(v6.1)는 CMA-ES · 자라는 뇌."""
GROW_KEYS: Final = ("grow_start", "grow_every")
"""자라는 뇌 설정 키(v6.1). method != "grow"면 to_dict()가 뺀다(크기 고정 실행의 config가 v6.0과 같게, 설계 D31)."""
GROW_COLUMNS: Final = ("brain_neurons", "brain_links", "grow")
"""자라는 실행(method == "grow")의 통계 열(v6.1, switch 바로 뒤, 결정적). brain_neurons·brain_links = 이번 세대 후보를 만든 은닉
뉴런 합·연결 수(자라기 전), grow = 이 세대 끝에 자랐으면 1(v1.1 learner_*·switch 열과 같은 규칙)."""


class Evaluator(Protocol):
    """평가기(WorkerPool, 테스트의 가짜 평가기). 학습기는 record=만 넘길 수 있다.

    관찰 기록(v1.2): 클래스 속성 `observes`가 참인 평가기만 학습 평가에서 observe=(쌍마다 코드 0·1·2)와
    frame=(격자 틀 (x0, y0, 칸 크기, nx, ny) 또는 None)을 받고 EvalResult.observations를 채운다. 표시가 없는 평가기는
    키워드를 받지 않는다(그래도 유전자·점수·결과·몸 수치는 기록된다).

    점수 NaN은 조립 실패 쌍이다(v2.0, evaluate.BUILD_FAILED_EVENT). 학습기는 그 개체를 꼴찌로 센다.
    """

    def evaluate(
        self, env_version: str, env_text: str, layout: GenomeLayout, genomes: np.ndarray, pairs: list[tuple[int, Trial]],
        record: list[bool] | None = None,
    ) -> EvalResult: ...


TIME_COLUMNS = ("gen_seconds", "elapsed", "steps_per_second", "worker_utilization")
"""시간에 따라 달라지는 통계 열. 결정성 비교에서 뺀다."""
OBSERVE_FAILED_FRAME = 1
"""관찰 기록 실패 종류: 시험 0번 세계 범위(격자 틀)만 못 만듦. 그 세대는 궤적 밀도만 빠진다."""
OBSERVE_FAILED_SUMMARY = 2
"""관찰 기록 실패 종류: 세대 요약 실패. 그 세대 대표는 관찰 키 없이 써진다(더 무거움)."""

LIVE_CONFIG_KEYS = (
    "max_generations", "max_evaluations", "max_time", "target_score", "stagnation_generations",
    "checkpoint_every", "trials", "aggregate", "validate_every", "restart_patience", "restart_from",
)
"""학습 중(멈춘 상태)에 바꿀 수 있는 설정. 학습 방법·시드·개체 수·시그마를 바꾸면 다른 실험이 되므로 새 실행으로 한다.
예선 설정(qualify_*)도 실행 처음에만 뜻이 있어 바꿀 수 없다."""


class TrainerChangeError(ValueError):
    """멈춘 학습에 적용할 수 없는 변경. message는 한국어."""


@dataclass
class Generation:
    """마지막 세대의 자세한 결과(화면에 보낼 것). 통계 줄과 같은 세대."""

    generation: int
    population: np.ndarray
    scores: np.ndarray
    """개체별 집계 점수."""
    representative: int
    trial0: Trial
    trajectories: list[Any] | None
    """관전 모드면 개체별 시험 0번 궤적(Trajectory), 아니면 None."""
    env_version: str
    champion_changed: bool
    observation: dict[str, np.ndarray] | None = None
    """이 세대의 관찰 기록(세대 대표 파일의 obs_ 키 사전, v1.2). 관찰을 끈 학습기·요약 실패면 None."""


@dataclass
class TrainConfig:
    method: str = "cmaes"
    run_seed: int = 1
    popsize: int = 32
    sigma0: float = 0.5
    trials: int | None = None
    """None이면 환경 파일의 evaluation.trials."""
    aggregate: str | None = None
    """None이면 환경 파일의 evaluation.aggregate."""
    diagonal_threshold: int = 1000
    ipop: bool = False
    validate_every: int = 1000
    """학습 시험이 이만큼 쌓일 때마다 검증."""
    validation_seeds: int = seeds.VALIDATION_SET
    validation_candidates: int = 3
    checkpoint_every: int = 10
    max_generations: int | None = None
    max_evaluations: int | None = None
    max_time: float | None = None
    target_score: float | None = None
    stagnation_generations: int | None = None
    # 유전 알고리즘(method="ga"). 개체 수는 popsize를 쓴다(명세 기본 64는 GAConfig().popsize).
    ga_elites: int = 2
    ga_tournament: int = 3
    ga_crossover: float = 0.7
    ga_mutation_rate: float = 0.1
    ga_mutation_sigma: float = 0.1
    ga_sigma_decay: float = 1.0
    ga_structure_flip: float = 0.0
    """구조 뒤집기 확률(v2.1): 자식마다, 있음 유전자마다. 0이면 끔(v2.0과 같음). 새 실행 권장값은 for_new_run(spec)."""
    # 담금질(method="sa"). popsize는 쓰지 않고, 한 세대 후보 수 = sa_neighbors + 1(현재해).
    sa_neighbors: int = 8
    sa_step: float = 0.1
    sa_initial_temperature: float | None = None
    """None이면 첫 이웃들 점수의 표준편차."""
    sa_cooling: float = 0.995
    sa_structure_flip: float = 0.0
    """구조 뒤집기 확률(v2.1): 이웃마다, 있음 유전자마다. 0이면 끔(v2.0과 같음). 새 실행 권장값은 for_new_run(spec)."""
    # 학습 안정성(v1.1). 기본값은 끔: 키가 없는 예전 설정은 v1.0과 같게 돈다. 새 실행은 for_new_run()으로 권장값.
    restart_patience: int | None = None
    """지금 학습 방법의 검증 최고점이 그 학습 방법의 학습 시험 이만큼 동안 오르지 않으면 새 학습 방법으로 바꾼다. None = 끔."""
    restart_from: str = "fresh"
    """fresh: 처음 유전자(지금 환경의 몸 초깃값, 뇌 0). champion: 챔피언 유전자.
    alternate: 홀수 번째 fresh, 짝수 번째 champion."""
    qualify_entrants: int = 1
    """예선 출전 수. 1 = 예선 없음. 2..MAX_QUALIFY_ENTRANTS면 처음에 출전자마다 qualify_trials씩 차례로 돌린다."""
    qualify_trials: int = 12000
    """출전자 하나가 쓰는 학습 시험 수."""
    # 자라는 뇌(v6.1, method="grow"). 이 실행에만 to_dict()가 쓴다(GROW_KEYS). 학습 중에는 바꿀 수 없다(학습 방법의 일부).
    grow_start: int = 2
    """은닉층마다 처음 뉴런 수(틀 = 환경 은닉층 크기보다 크면 틀로 자름: 처음부터 다 자란 크기)."""
    grow_every: int = 10
    """이 학습 방법 세대 수마다 은닉층마다 뉴런 두 배(틀에서 멈춤)."""

    def to_dict(self) -> dict[str, Any]:
        """저장용 사전(asdict). method != "grow"면 GROW_KEYS를 뺀다(크기 고정 실행은 v6.0과 같은 사전, 설계 D31)."""
        d = asdict(self)
        if self.method != "grow":
            for key in GROW_KEYS:
                del d[key]
        return d

    def validate(self) -> None:
        """값 범위를 확인한다. 잘못되면 TrainerChangeError(한국어)."""
        from evolab.app import strings as S

        def bad(name: str) -> TrainerChangeError:
            return TrainerChangeError(S.TRAIN_CONFIG_INVALID.format(name=name, value=getattr(self, name)))

        def is_int(v: Any) -> bool:
            return isinstance(v, int) and not isinstance(v, bool)

        def is_num(v: Any) -> bool:
            return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

        for name in ("popsize", "validate_every", "validation_seeds", "validation_candidates"):
            if not is_int(getattr(self, name)) or getattr(self, name) < 1:
                raise bad(name)
        if self.popsize < 2:
            raise bad("popsize")
        if not is_int(self.run_seed) or self.run_seed < 0:
            raise bad("run_seed")
        if not is_num(self.sigma0) or self.sigma0 <= 0:
            raise bad("sigma0")
        if not is_int(self.diagonal_threshold) or self.diagonal_threshold < 0:
            raise bad("diagonal_threshold")
        if not is_int(self.checkpoint_every) or self.checkpoint_every < 0:
            raise bad("checkpoint_every")
        for name in ("trials", "max_generations", "max_evaluations", "stagnation_generations"):
            v = getattr(self, name)
            if v is not None and (not is_int(v) or v < 1):
                raise bad(name)
        if self.max_time is not None and (not is_num(self.max_time) or self.max_time <= 0):
            raise bad("max_time")
        if self.target_score is not None and not is_num(self.target_score):
            raise bad("target_score")
        if self.aggregate not in (None, "mean", "min", "q25"):
            raise bad("aggregate")
        if self.method not in METHODS:
            raise bad("method")
        if self.method == "grow":
            for name in GROW_KEYS:
                if not is_int(getattr(self, name)) or getattr(self, name) < 1:
                    raise bad(name)
            if self.ipop:  # 개체 수가 바뀌면 「n세대마다」의 시험 수 뜻이 바뀐다(설계 D33)
                raise TrainerChangeError(S.TRAIN_GROW_NO_IPOP)
        if self.method == "ga" and self.popsize < 2:
            raise bad("popsize")
        if not is_int(self.ga_elites) or self.ga_elites < 0 or (self.method == "ga" and self.ga_elites >= self.popsize):
            raise bad("ga_elites")
        if not is_int(self.ga_tournament) or self.ga_tournament < 1 or (
            self.method == "ga" and self.ga_tournament > self.popsize
        ):
            raise bad("ga_tournament")
        for name in ("ga_crossover", "ga_mutation_rate", "ga_structure_flip", "sa_structure_flip"):
            if not is_num(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise bad(name)
        for name in ("ga_mutation_sigma", "sa_step"):
            if not is_num(getattr(self, name)) or getattr(self, name) <= 0:
                raise bad(name)
        for name in ("ga_sigma_decay", "sa_cooling"):
            if not is_num(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise bad(name)
        if not is_int(self.sa_neighbors) or self.sa_neighbors < 1:
            raise bad("sa_neighbors")
        t = self.sa_initial_temperature
        if t is not None and (not is_num(t) or t < 0):
            raise bad("sa_initial_temperature")
        p = self.restart_patience
        if p is not None:
            if not is_int(p) or p < 1:
                raise bad("restart_patience")
            if p < self.validate_every:  # 새 학습 방법이 판단 전에 반드시 한 번은 검증받게
                raise TrainerChangeError(S.TRAIN_RESTART_TOO_SHORT.format(value=p, every=self.validate_every))
        if not isinstance(self.restart_from, str) or self.restart_from not in RESTART_FROM:
            raise bad("restart_from")
        k = self.qualify_entrants
        if not is_int(k) or not 1 <= k <= MAX_QUALIFY_ENTRANTS:
            raise TrainerChangeError(S.TRAIN_QUALIFY_ENTRANTS.format(max=MAX_QUALIFY_ENTRANTS, value=k))
        if not is_int(self.qualify_trials) or self.qualify_trials < 1:
            raise bad("qualify_trials")
        if k > 1 and self.qualify_trials < self.validate_every:
            raise TrainerChangeError(S.TRAIN_QUALIFY_TOO_SHORT.format(value=self.qualify_trials,
                                                                      every=self.validate_every))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        """저장된 설정에서. 없는 키는 데이터클래스 기본값(새 키는 '끔')이라 예전 실행은 예전처럼 돈다."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def for_new_run(cls, spec: EnvSpec | None = None, /, **values: Any) -> TrainConfig:
        """새 실행의 설정: NEW_RUN_DEFAULTS(자동 재시작 등 권장값) → 구조 뒤집기 → 프리셋 권장값(editor.train) → values 순서로 얹는다.

        spec(v2.1)을 주면 그 환경에 있음 유전자(layout.on_genes)가 있을 때 GA·담금질 구조 뒤집기 확률을
        STRUCTURE_FLIP_NEW_RUN으로 둔다(values가 더 우선). 유전자 배치를 만들 수 없는 환경은 넣지 않는다(학습을 시작할 때
        Trainer가 같은 오류를 한국어로 알린다).

        (6.2.0, 설계 D31) spec의 환경 파일에 editor.train이 있으면 PRESET_TRAIN_KEYS의 키만 얹는다. 권장값일 뿐이라 모르는 키,
        형·범위가 틀린 값, 검증 주기와 맞지 않아 validate()가 거절할 값은 조용히 뺀다. editor.train이 없는 환경(옛 파일 전부)은
        v6.1과 같은 설정이다.
        """
        structure: dict[str, Any] = {}
        preset: dict[str, Any] = {}
        if spec is not None:
            try:
                on = layout_for(spec).on_genes
            except ValueError:  # GenomeError 포함
                on = ()
            if on:
                structure = {"ga_structure_flip": STRUCTURE_FLIP_NEW_RUN, "sa_structure_flip": STRUCTURE_FLIP_NEW_RUN}
            preset = _preset_train(spec, values)
        return cls(**{**NEW_RUN_DEFAULTS, **structure, **preset, **values})


def _preset_train(spec: EnvSpec, values: Mapping[str, Any]) -> dict[str, Any]:
    """spec의 editor.train에서 새 실행 설정에 얹을 키(for_new_run만 부른다). 틀린 것은 빼고 돌려준다(예외 없음)."""
    from evolab.spec.editor_meta import editor_train

    def is_int(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    raw = editor_train(getattr(getattr(spec, "file", None), "editor", None))
    ok: dict[str, Callable[[Any], bool]] = {
        "qualify_entrants": lambda v: is_int(v) and 1 <= v <= MAX_QUALIFY_ENTRANTS,
        "qualify_trials": lambda v: is_int(v) and v >= 1,
        "restart_patience": lambda v: v is None or (is_int(v) and v >= 1),
        "restart_from": lambda v: isinstance(v, str) and v in RESTART_FROM,
    }
    out = {key: raw[key] for key in PRESET_TRAIN_KEYS if key in raw and ok[key](raw[key])}
    # 검증 주기와의 관계(validate()가 보는 것): 부르는 쪽이 준 주기가 올바르면 그것, 아니면 데이터클래스 기본값
    every = values.get("validate_every")
    if not is_int(every) or every < 1:
        every = TrainConfig.__dataclass_fields__["validate_every"].default
    patience = out.get("restart_patience")
    if patience is not None and patience < every:
        del out["restart_patience"]
    merged = {**NEW_RUN_DEFAULTS, **out, **values}
    entrants, trials = merged.get("qualify_entrants"), merged.get("qualify_trials")
    if is_int(entrants) and entrants > 1 and is_int(trials) and trials < every:
        out.pop("qualify_entrants", None)  # 예선 출전자마다 한 번은 검증받아야 한다: 맞지 않는 권장 예선은 통째로 뺀다
        out.pop("qualify_trials", None)
    return out


@dataclass
class Champion:
    genome: np.ndarray | None = None
    score: float = -math.inf
    generation: int = -1


@dataclass
class LearnerRecord:
    """학습 방법 하나(처음 것, 예선 출전자, 재시작한 것)의 기록. 재시작·예선 판단은 이것과 설정만 본다."""

    index: int = 0
    """실행 안에서 만든 순서 = 학습 방법 시드 단어."""
    trials: int = 0
    """이 학습 방법이 쓴 학습 시험."""
    validations: int = 0
    """trials // validate_every 단위로 한 검증 수."""
    best: float = -math.inf
    """이 학습 방법 후보의 검증 최고점(표시·예선 선정)."""
    ref: float = -math.inf
    """재시작 기준점(재시작을 켤 때·환경 버전이 바뀔 때 다시 잡는다)."""
    ref_at: int = 0
    """ref가 마지막으로 오른(또는 다시 잡은) 때의 trials."""

    @property
    def stall(self) -> int:
        """재시작 시계: 기준점이 오르지 않은 학습 시험 수."""
        return self.trials - self.ref_at


@dataclass
class Reserve:
    """예선에서 지금까지 가장 좋은 출전자(하나만 보관)."""

    record: LearnerRecord
    learner: dict[str, Any]
    """출전이 끝난 세대 경계의 learner.state_dict()."""


@dataclass
class TrainState:
    generation: int = 0
    """끝낸 세대 수(다음 세대 번호)."""
    evaluations: int = 0
    validation_evaluations: int = 0
    elapsed: float = 0.0
    champion: Champion = field(default_factory=Champion)
    best_validation: float = -math.inf
    improved_generation: int = 0
    compare_score: float = math.nan
    """챔피언을 비교 평가 세트(선정에 쓰지 않은 시드 32개)로 잰 점수. 비교 그래프·표의 기본 값."""
    compare_success: float = math.nan
    """같은 평가에서 성공(outcome success) 비율."""
    compare_evaluations: int = 0
    current: LearnerRecord = field(default_factory=LearnerRecord)
    """지금 학습 방법의 기록."""
    learners: int = 1
    """지금까지 만든 학습 방법 수 = 다음 학습 방법 번호."""
    restarts: int = 0
    """자동 재시작 수."""
    qualifying: bool = False
    """예선 중인가."""
    reserve: Reserve | None = None
    """예선 중 지금까지 가장 좋은 출전자(끝난 출전자가 있을 때만)."""

    @property
    def validations_done(self) -> int:
        """지금 학습 방법의 검증 수(예전 코드·테스트 호환, 읽기 전용)."""
        return self.current.validations


STOP_GENERATIONS = "max_generations"
STOP_EVALUATIONS = "max_evaluations"
STOP_TIME = "max_time"
STOP_TARGET = "target_score"
STOP_STAGNATION = "stagnation"
STOP_REQUESTED = "requested"


def training_trials(spec: EnvSpec, run_seed: int, generation: int, count: int) -> list[Trial]:
    """학습 시험 목록. 학습 트랙 세트가 있으면 시험 = 트랙마다 한 번(시험 번호 = 트랙 번호, 시험 수 = 세트 크기)."""
    size = spec.track_set_size
    if size:
        return [Trial(t.seed, k) for k, t in enumerate(seeds.train_trials(run_seed, generation, size))]
    return seeds.train_trials(run_seed, generation, count)


def trials_for(spec: EnvSpec, config: TrainConfig) -> int:
    return spec.track_set_size or config.trials or spec.file.evaluation.trials


def generation_trials(config: TrainConfig, spec: EnvSpec) -> int:
    """학습 방법 하나의 세대당 학습 시험(화면 힌트·예산 검사용): (sa_neighbors+1 또는 popsize) × 시험 수."""
    per = config.sa_neighbors + 1 if config.method == "sa" else config.popsize
    return per * trials_for(spec, config)


def qualify_budget_problem(config: TrainConfig) -> str | None:
    """새 실행을 시작하기 전 검사(엔진 불변식이 아님): 예선 합계가 최대 시험 횟수 이상이면 한국어 메시지.

    validate()에 넣지 않는다: max_evaluations는 학습 중에 바꿀 수 있고, 예선이 끝난 뒤에는 뜻이 없다.
    """
    from evolab.app import strings as S

    k, b, m = config.qualify_entrants, config.qualify_trials, config.max_evaluations
    if k > 1 and m is not None and k * b >= m:
        return S.TRAIN_QUALIFY_OVER_BUDGET.format(k=k, b=b, total=k * b, max=m)
    return None


def aggregate(scores: np.ndarray, how: str) -> np.ndarray:
    """(개체, 시험) -> (개체,). 하위 25%는 아래쪽 ceil(시험 수 x 0.25)개의 평균.

    시험 하나라도 NaN(조립 실패)이면 그 개체의 집계도 NaN이다(평균·최솟값은 원래 그렇고, 하위 25%는 정렬이 NaN을 뒤로
    보내 빠뜨리므로 따로 NaN으로 둔다). NaN이 없으면 v1.2와 같은 계산이다.
    """
    if how == "mean":
        return scores.mean(axis=1)
    if how == "min":
        return scores.min(axis=1)
    k = max(1, math.ceil(scores.shape[1] * 0.25))
    out = np.sort(scores, axis=1, kind="stable")[:, :k].mean(axis=1)
    failed = np.isnan(scores).any(axis=1)
    if failed.any():
        out[failed] = np.nan
    return out


def check_initial_body(spec: EnvSpec, layout: GenomeLayout) -> None:
    """처음 몸 수치(layout.initial())로 전시 시험의 세계를 조립해 본다(설계 4.9). 안 되면 EpisodeBuildError(한국어).

    학습 중의 조립 실패는 그 쌍만 실패로 세지만, 처음 몸부터 안 되는 환경은 학습해도 뜻이 없어 시작하지 않는다
    (실험실 환경 열기와 같은 검사). 몸 유전자가 없는 환경(착륙선·트랙 자동차 등)은 몸 때문에 조립이 안 될 수 없어
    보지 않는다: v1.2와 같이 시작하고, 시험 세계를 만들 수 없으면 평가가 오류를 올린다(evaluate.py).
    """
    from evolab.app import strings as S

    if not layout.n_body:
        return
    body = layout.body_values(layout.initial()[None, :])[0]
    try:
        Episode(spec, body, seeds.display_trial())
    except EpisodeBuildError as exc:
        raise EpisodeBuildError(S.TRAIN_INITIAL_BODY_FAILED.format(reason=str(exc).strip())) from exc


def structure_flip_of(config: TrainConfig) -> float:
    """지금 학습 방법의 구조 뒤집기 확률(GA·담금질, v2.1). CMA-ES에는 없어 0."""
    return {"ga": config.ga_structure_flip, "sa": config.sa_structure_flip}.get(config.method, 0.0)


def structure_note(config: TrainConfig, layout: GenomeLayout, spec: EnvSpec | None = None) -> str | None:
    """GA·담금질 실행의 몸 구조 안내(한국어, 여러 줄이면 줄바꿈으로 잇는다). 알릴 것이 없으면 None.

    - 구조 뒤집기가 꺼져 있고 환경에 있음 유전자가 있으면 LAB_GA_NO_STRUCTURE: v2.0에서 시작한 GA·담금질 실행(설정에 키가
      없어 0)을 구조 환경에서 이어 갈 때 실험실·명령줄이 한 번 알린다(설계 6절).
    - spec을 주면 structure_range_note(범위 가운데가 0.5가 아니라 뒤집기가 건드리지 않는 있음 칸)도.
    """
    from evolab.app import strings as S

    notes: list[str] = []
    if config.method in ("ga", "sa") and layout.on_genes and not structure_flip_of(config) > 0:
        notes.append(S.LAB_GA_NO_STRUCTURE)
    if spec is not None:
        notes.append(structure_range_note(config, spec) or "")
    return "\n".join(n for n in notes if n) or None


def structure_range_note(config: TrainConfig, spec: EnvSpec) -> str | None:
    """GA·담금질 실행인데 있음 칸이 "=p.<id>"이면서 그 매개변수 범위의 가운데가 0.5가 아닌 부품이 있으면
    LAB_STRUCTURE_RANGE(부품 이름), 아니면 None. 그런 부품은 구조 뒤집기가 켜고 끄지 않는다(structure.off_centre_exists,
    on_genes와 같은 규칙). 새 실행과 이어하기 모두 알린다."""
    from evolab.app import strings as S
    from evolab.spec.structure import off_centre_exists

    if config.method not in ("ga", "sa"):
        return None
    skipped = off_centre_exists(spec.file)
    if not skipped:
        return None
    labels = {b.id: (b.label or b.id) for b in spec.file.bodies}
    return S.LAB_STRUCTURE_RANGE.format(names=", ".join(labels[b] for b, _ in skipped))


def _structure_fields(p: float, layout: GenomeLayout) -> dict[str, Any]:
    """GAConfig·SAConfig에 넘길 구조 뒤집기 칸. 확률 0이거나 있음 유전자가 없으면 비운다(학습 방법 설정·상태가 v2.0과 같음)."""
    if not p > 0 or not layout.on_genes:
        return {}
    return {"structure_flip": float(p), "structure_genes": layout.on_genes}


def grow_problem(config: TrainConfig, layout: GenomeLayout) -> str | None:
    """자라는 뇌(method == "grow")로 이 환경을 학습할 수 없으면 한국어 이유, 아니면 None(설계 4.16).

    뇌 없는 환경(출력 없음, 글라이더)은 FRAME_NO_BRAIN, 은닉층이 없는 뇌는 TRAIN_GROW_NO_HIDDEN. 실행 폴더를 만들기 전에
    (명령줄·세션의 새 실행) 부르고, Trainer를 만들 때도 같은 검사를 한다(이어 가기·재현 묶음).
    """
    from evolab.app import strings as S

    if config.method != "grow":
        return None
    if layout.brain.n_out == 0:
        return S.FRAME_NO_BRAIN
    if not layout.brain.hidden:
        return S.TRAIN_GROW_NO_HIDDEN
    return None


def make_learner(config: TrainConfig, layout: GenomeLayout, *, index: int = 0, x0: np.ndarray | None = None) -> Learner:
    """설정대로 학습 방법을 만든다. index = 실행 안에서 만든 순서(시드 단어, 0은 v1.0과 같은 시드),
    x0 = 출발 유전자(None이면 처음 유전자: 몸 초깃값, 뇌 0). GA·담금질의 구조 뒤집기는 layout.on_genes를 뒤집는다."""
    start = layout.initial() if x0 is None else x0
    seed = seeds.learner_seed(config.run_seed, index)
    if config.method == "cmaes":
        from evolab.learners.cmaes import CMAES, CMAESConfig

        return CMAES(
            start,
            CMAESConfig(
                popsize=config.popsize,
                sigma0=config.sigma0,
                diagonal_threshold=config.diagonal_threshold,
                ipop=config.ipop,
                seed=seed,
                bounded_dims=layout.n_body,
            ),
        )
    if config.method == "ga":
        from evolab.learners.ga import GAConfig, GeneticAlgorithm

        return GeneticAlgorithm(
            start,
            GAConfig(
                popsize=config.popsize,
                elites=config.ga_elites,
                tournament=config.ga_tournament,
                crossover=config.ga_crossover,
                mutation_rate=config.ga_mutation_rate,
                mutation_sigma=config.ga_mutation_sigma,
                sigma_decay=config.ga_sigma_decay,
                seed=seed,
                bounded_dims=layout.n_body,
                **_structure_fields(config.ga_structure_flip, layout),
            ),
        )
    if config.method == "sa":
        from evolab.learners.sa import SAConfig, SimulatedAnnealing

        return SimulatedAnnealing(
            start,
            SAConfig(
                neighbors=config.sa_neighbors,
                step=config.sa_step,
                initial_temperature=config.sa_initial_temperature,
                cooling=config.sa_cooling,
                seed=seed,
                bounded_dims=layout.n_body,
                **_structure_fields(config.sa_structure_flip, layout),
            ),
        )
    if config.method == "grow":
        # v6.1 자라는 뇌: CMA-ES만(GA·담금질 자라기는 미룸, 설계 D32). IPOP은 쓰지 않는다(validate가 거절, D33).
        # diagonal_threshold는 안쪽 CMA-ES가 켜진 좌표 수에 적용한다. 출발점이 챔피언이면 그 켜진 수부터 자란다.
        from evolab.learners.cmaes import CMAESConfig
        from evolab.learners.grow import GrowConfig, GrowingCMAES

        return GrowingCMAES(
            start,
            layout,
            CMAESConfig(
                popsize=config.popsize,
                sigma0=config.sigma0,
                diagonal_threshold=config.diagonal_threshold,
                ipop=False,
                seed=seed,
                bounded_dims=layout.n_body,
            ),
            GrowConfig(start=config.grow_start, every=config.grow_every),
        )
    raise ValueError(config.method)


class Trainer:
    def __init__(
        self,
        env_text: str,
        config: TrainConfig,
        evaluator: Evaluator,
        store: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
        stats_columns: list[str] | None = None,
        observe: bool | None = None,
    ) -> None:
        """stats_columns: 저장소에 아직 통계가 없을 때 쓸 열 머리(재현 묶음이 원래 열 그대로 다시 쓰게).

        observe: 세대마다 관찰 기록을 만들지(None이면 저장소가 있을 때만). 설정(TrainConfig)·run.json에 넣지 않는다:
        켜든 끄든 학습 결과는 같다. 명령줄 train --no-observe만 끈다(D11).
        """
        self.env_text = env_text
        self.spec: EnvSpec = compile_env(env_text)
        self.env_version = env_hash(self.spec.file)[:16]
        self.config = config
        self.evaluator = evaluator
        self.store = store
        self.clock = clock
        self.layout = layout_for(self.spec)
        check_initial_body(self.spec, self.layout)
        self.trials = trials_for(self.spec, config)
        self.how = config.aggregate or self.spec.file.evaluation.aggregate
        problem = grow_problem(config, self.layout)
        if problem is not None:  # 자라는 뇌인데 은닉층이 없는 뇌(v6.1): 틀 오류(FrameError)가 보이기 전에 한국어로
            raise TrainerChangeError(problem)
        self.learner = make_learner(config, self.layout)
        self.last_growth: tuple[int, int, int, int] | None = None
        """마지막 세대 끝에 뇌가 자랐고 그 자란 학습 방법으로 다음 세대를 이어 가면 (옛 뉴런 합, 새 뉴런 합, 옛 연결, 새 연결),
        아니면 None(v6.1, 자라는 실행만). 세션이 세대 메시지 "growth"로, 명령줄이 CLI_TRAIN_GROWN 한 줄로 알린다. 저장하지
        않는다. 통계 열 grow와 거의 같은 뜻이지만, 자란 세대 끝에 학습 방법이 바뀌어(자동 재시작, 예선 다음 출전, 예선에서 다른
        출전자가 통과) 자란 뇌를 더 쓰지 않으면 None이다: 「뇌가 자랐습니다」라고 알린 다음 세대가 작은 뇌로 돌면 안 된다."""
        self.state = TrainState(qualifying=config.qualify_entrants > 1)
        self.block_ids = [b.id for b in self.spec.file.scoring.blocks]
        self.end_ids = [t.event for t in self.spec.terminations]
        self.last_stats: dict[str, Any] = {}
        self.last_generation: Generation | None = None
        self.stop_reason: str | None = None
        self.weighted_blocks = self.spec.file.scoring.formula is None
        self.observe = (store is not None) if observe is None else bool(observe)
        self.last_observation: dict[str, np.ndarray] | None = None
        """마지막 세대의 관찰 기록(obs_ 키 사전). 세션이 세대 메시지 "observation"으로 화면에 보낸다."""
        self.observe_problem: str | None = None
        """관찰 기록을 만들지 못한 이유(한국어). 종류마다 처음 한 번만 남기고, 더 무거운 실패가 오면 그 글로 바꾼다
        (_observe_failed). 학습은 그대로 이어진다(D9)."""
        self._observe_level = 0
        """observe_problem의 종류(0 없음, OBSERVE_FAILED_FRAME, OBSERVE_FAILED_SUMMARY)."""
        self.last_build_failures = 0
        """마지막 세대 학습 시험 중 조립 실패(점수 NaN) 쌍 수(v2.0). 저장하지 않는다(통계 end_* 합의 모자라는 몫과 같은 뜻)."""
        self._header: list[str] | None = None
        if store is not None:
            self._header = store.stats_columns() if hasattr(store, "stats_columns") else None
        if self._header is None and stats_columns:
            self._header = list(stats_columns)

    # ------------------------------------------------------------------
    def run(
        self,
        on_generation: Callable[[dict[str, Any]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        """정지 조건이 될 때까지 세대를 돈다. 정지 원인을 돌려준다. 끝나면 마지막 검증과 체크포인트."""
        reason = self.check_stop()
        while reason is None:
            row = self.step()
            if on_generation is not None:
                on_generation(row)
            reason = self.check_stop()
            if reason is None and should_stop is not None and should_stop():
                reason = STOP_REQUESTED
        self.stop_reason = reason
        self.finish()
        return reason

    def finish(self) -> None:
        """멈출 때 체크포인트를 남긴다.

        멈출 때 따로 검증하지 않는다. 검증은 학습 시험 수로 정한 주기에만 해야, 끊었다 이어 간 실행이 끊지 않은
        실행과 같다(챔피언과 정체 판정이 멈춘 시점에 따라 달라지지 않게).
        """
        self.checkpoint()

    def check_stop(self) -> str | None:
        c, s = self.config, self.state
        if c.max_generations is not None and s.generation >= c.max_generations:
            return STOP_GENERATIONS
        if c.max_evaluations is not None and s.evaluations >= c.max_evaluations:
            return STOP_EVALUATIONS
        if c.max_time is not None and s.elapsed >= c.max_time:
            return STOP_TIME
        if c.target_score is not None and s.champion.score >= c.target_score:
            return STOP_TARGET
        # 예선 중에는 정체 멈춤을 쉰다(뒤 출전자는 처음부터 오르므로 챔피언이 오르지 않는 것이 정상). 예선이 끝날 때
        # improved_generation을 그 세대로 올려 그때부터 센다.
        if (c.stagnation_generations is not None and not s.qualifying
                and s.generation - s.improved_generation >= c.stagnation_generations):
            return STOP_STAGNATION
        return None

    def step(self, record: bool = False) -> dict[str, Any]:
        """한 세대. 통계 한 줄을 돌려준다(저장소가 있으면 stats.csv에도 쓴다).

        record면 모든 개체의 시험 0번 궤적을 남긴다(관전 모드). 기록은 결과를 바꾸지 않는다.
        """
        t0 = self.clock()
        c, s = self.config, self.state
        gen = s.generation
        # 세대가 끝까지(통계·챔피언 저장까지) 가지 않으면 이 트레이너는 반쯤 바뀐 상태다(learner.tell, 세대 번호).
        # 그 상태로는 체크포인트를 남기지 않는다(리뷰: 끝나지 않은 세대가 체크포인트되어 통계 줄이 빠짐).
        self.in_step = True
        X = self.learner.ask()
        trials = training_trials(self.spec, c.run_seed, gen, self.trials)
        pairs = [(i, t) for i in range(X.shape[0]) for t in trials]
        kw: dict[str, Any] = {}
        frame = world_key = None
        if self.observe and getattr(self.evaluator, "observes", False):
            # 평가기 능력 표시가 있을 때만 넘긴다(D8): 테스트의 가짜 평가기는 이 키워드를 받지 않는다
            try:
                frame, world_key = observing.world_frame(self.spec, self.env_version, trials[0])
            except Exception as exc:  # 파생 자료의 결함이 학습을 멈추지 않게(D9): 틀 없이 행동만 센다
                self._observe_failed(exc, OBSERVE_FAILED_FRAME)
            kw["observe"] = [OBSERVE_POSITIONS if k == 0 else OBSERVE_ACTIONS
                             for _ in range(X.shape[0]) for k in range(len(trials))]
            kw["frame"] = None if frame is None else tuple(frame)
        if record:
            kw["record"] = [k == 0 for _ in range(X.shape[0]) for k in range(len(trials))]
        t_eval = self.clock()
        res = self.evaluator.evaluate(self.env_version, self.env_text, self.layout, X, pairs, **kw)
        eval_wall = self.clock() - t_eval
        busy = float(getattr(self.evaluator, "last_busy", 0.0) or 0.0)  # 검증 평가가 덮어쓰기 전에 읽는다
        pop, T = X.shape[0], len(trials)
        scores = res.scores.reshape(pop, T)
        agg = aggregate(scores, self.how)
        self.learner.tell(X, agg)  # 학습 방법이 NaN(조립 실패)을 agg_s와 같은 규칙으로 바꾼다
        # 자라는 뇌(v6.1): 이 tell에서 자랐나. 교체 전에 읽고, 아래에서 학습 방법이 바뀌면 지운다(자란 뇌가 버려짐)
        self.last_growth = getattr(self.learner, "last_growth", None)
        # 통계·세대 대표·관찰·화면 점수: 조립 실패 개체는 유한한 최솟값 − 1(sanitize_scores). NaN이 없으면 agg 그대로의
        # 사본이라 v1.2와 비트 단위로 같다(±무한대 집계도 v1.2처럼 그대로 둔다)
        agg_s = sanitize_scores(agg, pop) if np.isnan(agg).any() else agg.copy()
        self.last_build_failures = int(np.count_nonzero(np.isnan(res.scores)))
        s.evaluations += pop * T
        cur = s.current
        cur.trials += pop * T
        rep = int(np.argmax(agg_s))  # 동점이면 앞쪽(결정적)
        # 기록용 세대 번호는 '끝낸 세대 수'(1부터): 통계 줄, 챔피언 파일, 체크포인트가 같은 번호를 쓴다.
        s.generation = gen + 1
        label = s.generation

        validated: float | None = None
        champion_before = s.champion.generation
        # 검증은 지금 학습 방법의 학습 시험이 검증 주기의 배수를 넘을 때마다(학습 방법 0은 v1.0과 같은 세대).
        # 예선 출전의 마지막 세대는 늘 검증한다(출전자마다 끝난 상태로 겨루게).
        closing = s.qualifying and cur.trials >= c.qualify_trials
        if cur.trials // c.validate_every > cur.validations or closing:
            cur.validations = cur.trials // c.validate_every
            validated = self._validate(X, agg_s)
        gen_time = self.clock() - t0
        s.elapsed += gen_time

        # 학습 방법 교체는 여기서 판단만 하고(통계 줄에 남김), 통계·대표를 쓴 뒤 체크포인트 전에 적용한다:
        # 어느 체크포인트도 '바꿔야 하는데 아직 안 바꾼' 상태가 아니다.
        switch = self._decide()
        row = self._stats_row(label, X, agg_s, res, rep, validated, gen_time, switch)
        workers = max(1, int(getattr(self.evaluator, "workers", 1) or 1))
        row["worker_utilization"] = min(1.0, busy / (workers * eval_wall)) if eval_wall > 0 else 0.0
        row = self._fit_columns(row)
        self.last_stats = row
        traj = None
        if record and res.trajectories is not None:
            traj = [res.trajectories[i * T] for i in range(pop)]
        obs: dict[str, np.ndarray] | None = None
        if self.observe:
            # 이미 계산된 X, agg, res와 교체 전 학습 방법 번호·예선 상태만 읽는다(통계 줄과 같은 시점)
            try:
                obs = observing.summarize(generation=label, X=X, agg=agg_s, res=res, trials_per=T, representative=rep,
                                          learner=cur.index, qualifying=s.qualifying, layout=self.layout,
                                          end_ids=self.end_ids, trial0=trials[0], frame=frame, world_key=world_key)
            except Exception as exc:  # 파생 자료의 결함이 학습을 멈추지 않게(D9): 이 세대 대표는 관찰 없이 쓴다
                self._observe_failed(exc, OBSERVE_FAILED_SUMMARY)
        self.last_observation = obs
        self.last_generation = Generation(label, X, agg_s, rep, trials[0], traj, self.env_version,
                                          s.champion.generation != champion_before, obs)
        if self.store is not None:
            self.store.append_stats(row)
            self.store.save_champion(label, "representative", X[rep], score=float(agg_s[rep]),
                                     env_version=self.env_version, **(obs or {}))
        self._apply_switch(switch)
        if self.last_growth is not None and s.current.index != cur.index:
            # 자란 학습 방법이 이 세대 끝에 다른 학습 방법으로 바뀌었다(자동 재시작, 예선 다음 출전, 예선에서 다른 출전자가 통과):
            # 자란 뇌는 다음 세대에 쓰이지 않으므로 알리지 않는다. 예선 끝에 방금 자란 출전자가 통과하면(번호가 같음) 그 자란
            # 상태에서 이어 가므로 그대로 알린다. 통계 줄의 grow 열은 이미 썼고 바꾸지 않는다
            self.last_growth = None
        self.in_step = False
        if self.store is not None and c.checkpoint_every and s.generation % c.checkpoint_every == 0:
            self.checkpoint()  # 바꾼 뒤 상태(새 학습 방법)가 담긴다
        return row

    def _observe_failed(self, exc: BaseException, level: int) -> None:
        """관찰 기록을 만들지 못했다: 그 종류(level)의 처음 이유만 남긴다(세션·명령줄이 바뀔 때만 한 번씩 알린다).

        OBSERVE_FAILED_FRAME(세계 범위만 못 만듦: 그 세대는 궤적 밀도만 빠짐)보다 OBSERVE_FAILED_SUMMARY(요약 실패: 그 세대
        관찰 기록이 통째로 빠짐)가 무겁다. 더 무거운 실패가 나중에 오면 그 글로 바꾼다(같거나 가벼우면 그대로).
        """
        if level <= self._observe_level:
            return
        from evolab.app import strings as S

        text = S.TRAIN_OBSERVE_FRAME_FAILED if level == OBSERVE_FAILED_FRAME else S.TRAIN_OBSERVE_FAILED_ONWARD
        self._observe_level = level
        self.observe_problem = text.format(reason=str(exc).strip() or type(exc).__name__)

    def _validate(self, X: np.ndarray, agg: np.ndarray) -> float:
        """후보를 검증 세트로 평가해 챔피언을 고른다. 이번 검증의 최고 점수를 돌려준다.

        agg는 NaN이 없는 집계 점수(agg_s). 검증 시험에서 하나라도 조립되지 않은(평균이 NaN인) 후보는 −∞로 보아 챔피언이
        될 수 없다(모든 후보가 그러면 이번 검증의 최고 점수는 −∞이고 아무것도 바뀌지 않는다).
        """
        s = self.state
        k = min(self.config.validation_candidates, X.shape[0])
        top = np.argsort(-agg, kind="stable")[:k]
        cands = np.vstack([X[top], self.learner.best()[None, :]])
        trials = seeds.validation_trials(self.config.validation_seeds)
        pairs = [(i, t) for i in range(cands.shape[0]) for t in trials]
        res = self.evaluator.evaluate(self.env_version, self.env_text, self.layout, cands, pairs)
        s.validation_evaluations += len(pairs)
        val = res.scores.reshape(cands.shape[0], len(trials)).mean(axis=1)
        failed = np.isnan(val)
        if failed.any():  # 조립 실패 후보(NaN이 없으면 v1.2와 같은 배열 그대로)
            val[failed] = -math.inf
        best = int(np.argmax(val))
        v = float(val[best])
        cur = s.current
        if v > s.champion.score:
            s.champion = Champion(cands[best].copy(), v, s.generation)
            self._compare(cands[best])
            if self.store is not None:
                self.store.save_champion(s.generation, "champion", cands[best], score=v,
                                         compare_score=s.compare_score, env_version=self.env_version,
                                         learner=cur.index)
        if v > s.best_validation:
            s.best_validation = v
            s.improved_generation = s.generation
        # 지금 학습 방법 자신의 기록(재시작 기준점·예선 선정). 챔피언과 같은 값을 같은 엄격 부등호로 따라간다.
        if v > cur.best:
            cur.best = v
        if v > cur.ref:
            cur.ref, cur.ref_at = v, cur.trials
        return v

    def _compare(self, genome: np.ndarray) -> None:
        """새 챔피언을 비교 평가 세트로 잰다(명세: 공정한 비교). 이 시험은 누적 시험 횟수와 따로 센다.

        비교 점수는 조립된 코스들의 평균이다(판정 평균과 같은 규칙, evaluate.mean_of_built). 조립 실패는 (환경, 몸, 시험)의
        함수라(설계 D22) 검증 시험에서 조립된 챔피언도 비교 코스 몇 개에서는 조립되지 않을 수 있다. 실패한 코스가 없으면
        float(scores.mean())과 같아 v1·v1.2와 비트 단위로 같고, 경주 코스 1..32 점수의 평균과 같다는 정확성
        고리(app/race/model.py)도 그때 성립한다. 성공률은 실패 코스를 실패로 세어 32로 나눈다(판정과 같음).
        """
        s = self.state
        trials = seeds.compare_trials()
        pairs = [(0, t) for t in trials]
        res = self.evaluator.evaluate(self.env_version, self.env_text, self.layout, genome[None, :], pairs)
        s.compare_evaluations += len(pairs)
        s.compare_score = mean_of_built(res.scores)
        s.compare_success = sum(1 for o in res.outcomes if o == "success") / len(pairs)

    # ------------------------------------------------------------------
    # 학습 방법 교체: 예선, 자동 재시작 (세대 끝 한 곳에서만)
    # ------------------------------------------------------------------
    def _decide(self) -> int:
        """이 세대 끝에 학습 방법을 바꿀지(SWITCH_*). 설정과 상태만 보고, 상태를 바꾸지 않는다."""
        c, s = self.config, self.state
        cur = s.current
        if s.qualifying:  # 예선 중에는 자동 재시작하지 않는다
            if cur.trials < c.qualify_trials:
                return SWITCH_NONE
            return SWITCH_QUALIFY_NEXT if cur.index + 1 < c.qualify_entrants else SWITCH_QUALIFY_PICK
        if c.restart_patience is None or cur.stall < c.restart_patience:
            return SWITCH_NONE
        r = s.restarts + 1
        use_champion = c.restart_from == "champion" or (c.restart_from == "alternate" and r % 2 == 0)
        if use_champion and s.champion.genome is not None:
            return SWITCH_RESTART_CHAMPION
        return SWITCH_RESTART_FRESH

    def _apply_switch(self, code: int) -> None:
        """_decide가 정한 교체를 적용한다. 챔피언, best_validation, compare_*는 그대로 이어진다."""
        c, s = self.config, self.state
        if code == SWITCH_NONE:
            return
        if code in (SWITCH_QUALIFY_NEXT, SWITCH_QUALIFY_PICK):
            cur = s.current
            if s.reserve is None or cur.best > s.reserve.record.best:  # 동점이면 앞 출전자
                s.reserve = Reserve(dataclasses.replace(cur), self.learner.state_dict())
            if code == SWITCH_QUALIFY_NEXT:
                self._new_learner(self.layout.initial())
                return
            # 예선 끝: 통과자는 끊지 않은 실행에서도 늘 저장한 상태에서 다시 만든다(이어 간 실행과 같은 경로)
            r = s.reserve
            learner = make_learner(c, self.layout, index=r.record.index)
            learner.load_state_dict(r.learner)
            rec = dataclasses.replace(r.record)
            rec.validations = rec.trials // c.validate_every  # 출전 중 검증 주기가 바뀌었어도 새 단위로
            self.learner, s.current, s.reserve, s.qualifying = learner, rec, None, False
            s.improved_generation = max(s.improved_generation, s.generation)  # 정체 멈춤은 여기서부터 센다
            return
        champion = s.champion.genome
        x0 = champion.copy() if code == SWITCH_RESTART_CHAMPION and champion is not None else self.layout.initial()
        self._new_learner(x0)
        s.restarts += 1

    def _new_learner(self, x0: np.ndarray) -> None:
        """다음 번호(시드 단어)로 학습 방법을 새로 만든다. 개체 수·시그마는 설정 값 그대로(IPOP으로 늘었어도)."""
        s = self.state
        idx = s.learners
        learner = make_learner(self.config, self.layout, index=idx, x0=x0)
        self.learner = learner
        s.learners = idx + 1
        s.current = LearnerRecord(index=idx)

    def _rearm(self, record: LearnerRecord | None = None) -> None:
        """재시작 기준점을 다시 잡는다(재시작을 켤 때, 환경 버전이 바뀔 때). 다음 검증부터 다시 센다.

        record가 없으면 지금 학습 방법의 기록(재시작을 켤 때는 예선의 보관 출전자 기록도 이것으로 다시 잡는다).
        """
        rec = self.state.current if record is None else record
        rec.ref, rec.ref_at = -math.inf, rec.trials

    def _stats_row(
        self, gen: int, X: np.ndarray, agg: np.ndarray, res: EvalResult, rep: int, validated: float | None,
        gen_time: float, switch: int = SWITCH_NONE,
    ) -> dict[str, Any]:
        s = self.state
        cur = s.current
        q = np.quantile(agg, [0.25, 0.5, 0.75])
        learner_stats = self.learner.stats()
        steps = int(res.steps.sum())
        row: dict[str, Any] = {
            "generation": gen,
            "evaluations": s.evaluations,
            "validation_evaluations": s.validation_evaluations,
            "score_best": float(agg.max()),
            "score_mean": float(agg.mean()),
            "score_q25": float(q[0]),
            "score_median": float(q[1]),
            "score_q75": float(q[2]),
            "score_min": float(agg.min()),
            "validation_best": validated if validated is not None else "",
            "champion_score": s.champion.score if math.isfinite(s.champion.score) else "",
            "champion_generation": s.champion.generation,
            "compare_score": s.compare_score if math.isfinite(s.compare_score) else "",
            "compare_success": s.compare_success if math.isfinite(s.compare_success) else "",
            "compare_evaluations": s.compare_evaluations,
            # 학습 방법 기록: 이번 세대를 만든(바꾸기 전) 학습 방법 기준. switch는 이 세대 끝에 일어난 교체.
            "learner": cur.index,
            "learner_trials": cur.trials,
            "learner_best": cur.best if math.isfinite(cur.best) else "",
            "learner_stall": cur.stall,
            "restarts": s.restarts,
            "qualifying": 1 if s.qualifying else 0,
            "switch": switch,
            **self._grow_columns(learner_stats),  # 자라는 실행만(v6.1 GROW_COLUMNS), 크기 고정 실행은 비어 열이 v6.0과 같음
            "sigma": float(learner_stats.get("sigma", math.nan)),
            "diversity": self._diversity(X, learner_stats),
            "distance": mean_pairwise_distance(X),
            "steps": steps,
        }
        n = len(res.end_events)
        for e in self.end_ids:
            row[f"end_{e}"] = sum(1 for x in res.end_events if x == e) / n
        for j, b in enumerate(self.block_ids):
            row[f"block_{b}"] = float(res.blocks[:, j].mean())
        if self.weighted_blocks and res.weighted is not None and self.block_ids:
            per = res.weighted.reshape(X.shape[0], -1, len(self.block_ids)).mean(axis=1)
            for j, b in enumerate(self.block_ids):
                row[f"contrib_mean_{b}"] = float(per[:, j].mean())
                row[f"contrib_rep_{b}"] = float(per[rep, j])
        row["scores"] = " ".join(repr(round(float(v), 4)) for v in agg)
        # 아래는 시간에 따라 달라지는 열(결정성 비교에서 뺀다)
        row["gen_seconds"] = gen_time
        row["elapsed"] = s.elapsed
        row["steps_per_second"] = steps / gen_time if gen_time > 0 else 0.0
        return row

    def _diversity(self, X: np.ndarray, learner_stats: dict[str, float]) -> float:
        """통계 열 diversity(유전자 표준편차 평균). 크기 고정 실행은 v6.0 식 그대로 모든 유전자의 평균이다. 자라는 실행은 켜진
        좌표만의 평균(학습 방법이 이 세대 후보에서 이미 계산한 stats()["diversity"]): 꺼진 좌표는 늘 0이라 틀 전체로 평균하면
        값이 (켜진 좌표 수 / 틀 길이)배로 줄고 자랄 때마다 두 배쯤 뛴다. 여기서 learner.active_coords()로 다시 계산하지 않는다:
        이 세대 끝에 자랐으면 그것은 이미 자란 뒤의 좌표다."""
        if self.config.method == "grow":
            return float(learner_stats["diversity"])
        return float(np.mean(np.std(X, axis=0)))

    def _grow_columns(self, learner_stats: dict[str, float]) -> dict[str, Any]:
        """자라는 실행의 통계 열(GROW_COLUMNS): 이번 세대 후보를 만든 뉴런 합·연결 수와 이 세대 끝에 자랐나(0/1). 아니면 빈 사전."""
        if self.config.method != "grow":
            return {}
        return {"brain_neurons": int(learner_stats["neurons"]), "brain_links": int(learner_stats["links"]),
                "grow": int(learner_stats["grown"])}

    def _fit_columns(self, row: dict[str, Any]) -> dict[str, Any]:
        """예전 형식으로 시작한 실행이면 그 열에 맞춘다(없는 열은 빈칸, 새 열은 뺀다)."""
        if self._header is None or list(row) == self._header:
            return row
        return {k: row.get(k, "") for k in self._header}

    # ------------------------------------------------------------------
    # 멈춘 상태에서의 변경 (세대 경계)
    # ------------------------------------------------------------------
    def apply_config(self, changes: dict[str, Any]) -> dict[str, Any]:
        """정지 조건 같은 설정을 바꾼다. 바꾼 값만 돌려준다. 체크포인트에 새 설정이 들어간다."""
        from evolab.app import strings as S

        bad = [k for k in changes if k not in LIVE_CONFIG_KEYS]
        if bad:
            raise TrainerChangeError(S.TRAIN_CHANGE_NOT_ALLOWED.format(keys=", ".join(bad)))
        trial_config = TrainConfig.from_dict({**self.config.to_dict(), **changes})
        trial_config.validate()
        done = {k: v for k, v in changes.items() if getattr(self.config, k) != v}
        was_off = self.config.restart_patience is None
        self.config = trial_config
        self.trials = trials_for(self.spec, self.config)
        self.how = self.config.aggregate or self.spec.file.evaluation.aggregate
        if "validate_every" in done:
            # 검증은 지금 학습 방법의 학습 시험 수가 새 주기의 배수를 넘을 때마다: 지금까지 한 수를 새 단위로 바꿔야
            # 바로 이어진다
            cur = self.state.current
            cur.validations = cur.trials // self.config.validate_every
        if was_off and self.config.restart_patience is not None:
            # 끔 → 켬: 기준점을 다시 잡아 켠 뒤 첫 검증부터 센다(오래 정체한 실행을 켜자마자 바꾸지 않게,
            # 통계 없이도 이어하기·재현이 같게). 예선 중이면 보관 출전자도 다시 잡는다: 통과자는 보관 출전자일 때가
            # 많은데, 끈 동안 쌓인 정체로 본선 첫 세대에 바로 바뀌면 안 된다(누가 이기든 켠 뒤부터 센다)
            self._rearm()
            if self.state.reserve is not None:
                self._rearm(self.state.reserve.record)
        return done

    def changeable_state(self) -> dict[str, Any]:
        """설정 변경·환경 교체를 되돌릴 때 쓰는 사본(저장에 실패하면 트레이너를 그대로 돌려놓는다)."""
        s = self.state
        return {"config": TrainConfig.from_dict(self.config.to_dict()), "env_text": self.env_text, "spec": self.spec,
                "env_version": self.env_version, "layout": self.layout, "trials": self.trials, "how": self.how,
                "current": dataclasses.replace(s.current),
                "reserve_record": None if s.reserve is None else dataclasses.replace(s.reserve.record),
                "learner_structure": self._learner_structure()}

    def restore_changeable(self, snap: dict[str, Any]) -> None:
        self.config = snap["config"]
        self.env_text, self.spec, self.env_version = snap["env_text"], snap["spec"], snap["env_version"]
        self.layout = snap["layout"]
        self.trials, self.how = snap["trials"], snap["how"]
        self.state.current = dataclasses.replace(snap["current"])
        if self.state.reserve is not None and snap["reserve_record"] is not None:
            self.state.reserve.record = dataclasses.replace(snap["reserve_record"])
        structure = snap.get("learner_structure")
        if structure is not None and structure != self._learner_structure():
            self.learner.set_structure(*structure)  # type: ignore[attr-defined]

    def _learner_structure(self) -> tuple[float, tuple[int, ...]] | None:
        """지금 학습 방법의 구조 뒤집기 칸 (확률, 있음 유전자). 뒤집기가 없는 학습 방법(CMA-ES)이면 None."""
        if not hasattr(self.learner, "set_structure"):
            return None
        cfg = self.learner.config  # type: ignore[attr-defined]
        return float(cfg.structure_flip), tuple(cfg.structure_genes)

    def _structure_for(self, layout: GenomeLayout) -> tuple[float, tuple[int, ...]] | None:
        """배치 layout에서 지금 학습 방법이 뒤집어야 할 (확률, 있음 유전자): make_learner가 그 배치로 새로 만들 때와 같은
        규칙(_structure_fields). 지금 값과 같거나 뒤집기가 없는 학습 방법이면 None(바꿀 것 없음)."""
        have = self._learner_structure()
        if have is None:
            return None
        fields = _structure_fields(structure_flip_of(self.config), layout)
        want = (float(fields.get("structure_flip", 0.0)), tuple(fields.get("structure_genes", ())))
        return None if want == have else want

    def replace_env(self, env_text: str) -> str:
        """환경을 바꾼다(유전자 배치, 점수 블록, 종료 조건 목록이 같아야 한다). 새 환경 버전 id(해시)를 돌려준다.

        예선 중(평가를 하나라도 했으면)에는 거부한다(출전자끼리 다른 환경에서 겨루지 않게). 환경 버전이 바뀌면 지금
        학습 방법의 재시작 기준점을 다시 잡는다(어렵게 고친 직후 옛 기준을 못 넘어 엉뚱하게 재시작하지 않게).

        있음 칸 수식을 고쳐 있음 유전자(layout.on_genes, 서명에 들어가지 않음)가 바뀌면 지금 GA·담금질의 구조 뒤집기 목록도
        새 배치를 따른다(설계 4.12·3.5: 고친 뒤 새로 만드는 학습 방법과 같은 규칙). 바뀐 목록은 학습 방법 상태의 설정에
        들어가 다음 체크포인트에 남고, 재현 묶음은 같은 세대에 replace_env를 다시 부르므로 이어하기·재현이 같다. 구조
        뒤집기를 끈 실행(확률 0)은 바뀌는 것이 없다(상태 키도 그대로). 보관 출전자(예선)는 여기서 있을 수 없다: 보관은
        예선 중 평가 뒤에만 생기고, 그때는 위에서 거부한다.
        """
        from evolab.app import strings as S

        s = self.state
        if s.qualifying and s.evaluations > 0:
            raise TrainerChangeError(S.TRAIN_ENV_DURING_QUALIFY.format(n=s.current.index + 1,
                                                                       k=self.config.qualify_entrants))
        spec = compile_env(env_text)
        layout = layout_for(spec)
        if layout.signature() != self.layout.signature():
            raise TrainerChangeError(S.TRAIN_ENV_LAYOUT_CHANGED)
        try:
            check_initial_body(spec, layout)  # 처음 몸부터 조립되지 않는 환경으로는 바꾸지 않는다(시작할 때와 같은 검사)
        except EpisodeBuildError as exc:
            raise TrainerChangeError(str(exc)) from exc
        if [b.id for b in spec.file.scoring.blocks] != self.block_ids or [t.event for t in spec.terminations] != self.end_ids:
            raise TrainerChangeError(S.TRAIN_ENV_COLUMNS_CHANGED)
        if (spec.file.scoring.formula is None) != self.weighted_blocks:
            raise TrainerChangeError(S.TRAIN_ENV_COLUMNS_CHANGED)
        version = env_hash(spec.file)[:16]
        structure = self._structure_for(layout)
        if version != self.env_version:
            self._rearm()
        self.env_text = env_text
        self.spec = spec
        # 새 환경의 배치를 쓴다. 신호(signature)가 같아 학습 상태·체크포인트·평가는 그대로이고, 몸 수치 초깃값만
        # 달라질 수 있다: 처음 상태에서 새로 시작하는 학습 방법과 다음 예선 출전자의 출발점(layout.initial())이
        # 지금 환경을 따라야, 그 뒤 체크포인트에서 이어 간 트레이너(체크포인트의 환경으로 배치를 만든다)와 같다
        self.layout = layout
        if structure is not None:
            self.learner.set_structure(*structure)  # type: ignore[attr-defined]
        self.env_version = version
        self.trials = trials_for(spec, self.config)
        self.how = self.config.aggregate or spec.file.evaluation.aggregate
        return self.env_version

    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        s = self.state
        return {
            "format": CHECKPOINT_FORMAT,
            "config": self.config.to_dict(),
            "env_version": self.env_version,
            "layout_signature": self.layout.signature(),
            "learner": self.learner.state_dict(),
            "state": {
                "generation": s.generation,
                "evaluations": s.evaluations,
                "validation_evaluations": s.validation_evaluations,
                "elapsed": s.elapsed,
                "champion_genome": None if s.champion.genome is None else s.champion.genome.copy(),
                "champion_score": s.champion.score,
                "champion_generation": s.champion.generation,
                "best_validation": s.best_validation,
                "compare_score": s.compare_score,
                "compare_success": s.compare_success,
                "compare_evaluations": s.compare_evaluations,
                "improved_generation": s.improved_generation,
                # 형식 2: 학습 방법 기록과 예선 상태
                "current": asdict(s.current),
                "learners": s.learners,
                "restarts": s.restarts,
                "qualifying": s.qualifying,
                "reserve": None if s.reserve is None else {"record": asdict(s.reserve.record),
                                                           "learner": s.reserve.learner},
            },
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """체크포인트를 불러온다(형식 1·2). 모두 읽고 검사한 다음에야 바꾼다: 실패하면(ValueError) 트레이너는 그대로다."""
        fmt = d.get("format")
        if fmt not in READ_FORMATS:
            raise ValueError("checkpoint format")
        if d["layout_signature"] != self.layout.signature():
            raise ValueError("layout")
        st = d["state"]
        try:
            if fmt == 1:
                # v1.0: 학습 방법 0 하나. best는 best_validation과 같은 값(같은 최댓값, 같은 부등호)이라 정확하다.
                # ref_at은 정확하지 않지만 쓰이지 않는다(설정이 끔이고, 켜면 apply_config가 다시 잡는다).
                if self.config.qualify_entrants != 1:
                    raise ValueError("qualify_entrants")
                ev, bv = st["evaluations"], st["best_validation"]
                current = LearnerRecord(0, ev, st["validations_done"], bv, bv, ev)
                learners, restarts, qualifying, reserve = 1, 0, False, None
            else:
                current = _learner_record(st["current"])
                learners, restarts, qualifying = st["learners"], st["restarts"], st["qualifying"]
                r = st["reserve"]
                reserve = None if r is None else Reserve(_learner_record(r["record"]), r["learner"])
            state = TrainState(
                generation=st["generation"],
                evaluations=st["evaluations"],
                validation_evaluations=st["validation_evaluations"],
                elapsed=st["elapsed"],
                champion=Champion(st["champion_genome"], st["champion_score"], st["champion_generation"]),
                best_validation=st["best_validation"],
                improved_generation=st["improved_generation"],
                compare_score=st.get("compare_score", math.nan),
                compare_success=st.get("compare_success", math.nan),
                compare_evaluations=st.get("compare_evaluations", 0),
                current=current,
                learners=learners,
                restarts=restarts,
                qualifying=qualifying,
                reserve=reserve,
            )
        except (KeyError, TypeError) as e:
            raise ValueError(f"checkpoint state: {e!r}") from e
        _check_learner_state(self.config, state)
        if reserve is not None:
            # 보관 출전자 상태를 미리 시험한다: 깨졌으면 예선이 끝날 때가 아니라 이어 갈 때 알린다
            make_learner(self.config, self.layout, index=reserve.record.index).load_state_dict(reserve.learner)
        learner = make_learner(self.config, self.layout, index=current.index)
        learner.load_state_dict(d["learner"])  # 상태 안의 설정(시드 포함)이 생성자 값보다 우선한다
        self.learner = learner
        self.state = state

    def checkpoint(self) -> None:
        if getattr(self, "in_step", False):
            return  # 끝나지 않은 세대(오류로 중간에 멈춤): 마지막 정상 체크포인트를 그대로 둔다
        if self.store is not None and self.state.generation:
            self.store.save_checkpoint(self.state.generation, self.state_dict())

    @classmethod
    def resume(
        cls, store: Any, evaluator: Evaluator, generation: int | None = None, clock: Callable[[], float] = time.perf_counter,
        *, observe: bool | None = None, **overrides: Any,
    ) -> Trainer:
        """저장소의 체크포인트에서 이어간다. overrides로 정지 조건 같은 설정을 바꿀 수 있다.

        observe는 Trainer와 같다(None이면 켬). 체크포인트에는 관찰 상태가 없다(관찰은 세대마다의 순수 함수):
        끈 실행을 켜서 이어 가면 그 세대부터 기록한다."""
        gen, d = store.load_checkpoint(generation)
        info = store.read_run_info()
        config = TrainConfig.from_dict({**d["config"], **overrides})
        versions = list(info["env_versions"])
        # 체크포인트 시점에 쓰던 환경: 적용 세대가 체크포인트 세대 이하인 마지막 기록
        usable = [v for v in versions if int(v.get("from_generation", 0)) <= gen] or versions[:1]
        version = usable[-1]["id"]
        text = store.env_text(version)
        if versions[-1]["id"] != version:
            store.add_env_version(text, gen)  # 마지막 기록이 지금 쓰는 환경이 되게
        trainer = cls(text, config, evaluator, store, clock, observe=observe)
        trainer.load_state_dict(d)
        store.truncate_after(gen)
        return trainer


def _learner_record(d: Any) -> LearnerRecord:
    """체크포인트의 학습 방법 기록. 빠졌거나 모르는 항목, 잘못된 형이면 ValueError."""
    if not isinstance(d, dict):
        raise ValueError(f"learner record: {type(d).__name__}")
    names = [f.name for f in dataclasses.fields(LearnerRecord)]
    if sorted(d) != sorted(names):
        raise ValueError(f"learner record keys: {sorted(d)}")
    rec = LearnerRecord(**d)
    for name in ("index", "trials", "validations", "ref_at"):
        v = getattr(rec, name)
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"learner record {name}: {v!r}")
    for name in ("best", "ref"):
        v = getattr(rec, name)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v):
            raise ValueError(f"learner record {name}: {v!r}")
    return rec


def _check_learner_state(config: TrainConfig, s: TrainState) -> None:
    """학습 방법 기록의 불변식. 어기면 ValueError(깨진 체크포인트, 설정과 맞지 않는 상태)."""
    cur, k = s.current, config.qualify_entrants

    def need(ok: bool, what: str) -> None:
        if not ok:
            raise ValueError(f"learner state: {what}")

    for name, v in (("learners", s.learners), ("restarts", s.restarts)):
        need(isinstance(v, int) and not isinstance(v, bool), name)
    need(isinstance(s.qualifying, bool), "qualifying")
    need(0 <= cur.index < s.learners, "index")
    need(s.restarts >= 0 and cur.trials >= 0 and cur.validations >= 0, "counts")
    if s.qualifying:
        need(k > 1, "qualify_entrants")
        need(s.restarts == 0, "restarts during qualifying")
        need(s.learners == cur.index + 1 <= k, "entrant index")
        need((s.reserve is None) == (cur.index == 0), "reserve")
        need(s.reserve is None or (0 <= s.reserve.record.index < cur.index), "reserve index")
        need(s.reserve is None or isinstance(s.reserve.learner, dict), "reserve learner")
    else:
        need(s.reserve is None, "reserve outside qualifying")
        need(k <= 1 or s.learners >= k, "learners after qualifying")


def mean_pairwise_distance(X: np.ndarray) -> float:
    """개체 사이 유클리드 거리의 평균(서로 다른 쌍)."""
    n = X.shape[0]
    if n < 2:
        return 0.0
    sq = np.einsum("ij,ij->i", X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    iu = np.triu_indices(n, 1)
    return float(np.sqrt(np.maximum(d2[iu], 0.0)).mean())

"""구조 뒤집기(v2.1, 설계 4.12·D41): 유전 알고리즘과 담금질이 몸 구조를 바꾸게 하는 돌연변이.

몸 구조는 있음 유전자(물체의 exists가 정확히 "=p.<id>"인 진화 매개변수, GenomeLayout.on_genes)로 정해진다. 있음 값이
0.5보다 크면 그 물체가 있다. 범위 0..1인 매개변수는 유전자 공간(-1..1)의 0이 물리 값 0.5라서 유전자의 부호가 곧 구조다.

GA·담금질의 가우시안 돌연변이(폭 0.1)로는 처음 ±0.5에 있는 있음 유전자가 0을 거의 넘지 못해 구조가 바뀌지 않는다(evo 원형:
25세대 동안 한 번도 안 바뀜). 그래서 자식·이웃마다, 있음 유전자마다 확률 p로 부호를 뒤집어 ∓0.5에 둔다.

- 난수는 학습 방법 전용 생성기(체크포인트에 상태가 있음)에서 행 × 유전자 순서로 뽑으므로 이어하기·재현이 같다.
- p가 0이거나 뒤집을 유전자·행이 없으면 난수를 부르지 않는다: 구조 뒤집기를 쓰지 않는 실행은 v2.0과 비트 단위로 같다.
- CMA-ES에는 없다(분포의 표준편차 0.5가 이미 구조를 뒤집는다, 설계 D38).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from evolab.app import strings as S
from evolab.learners.base import is_int, is_real

FLIP_VALUE = 0.5
"""뒤집은 있음 유전자의 크기. 범위 0..1 매개변수에서 물리 값 0.25(없음) 또는 0.75(있음)이고, 에디터가 만드는 있음
매개변수의 처음 값과 같다(설계 D38)."""


def flip_structure(X: np.ndarray, rows: Sequence[int], on_genes: Sequence[int], p: float,
                   rng: np.random.Generator) -> int:
    """rows × on_genes 순서로 rng.random() < p이면 X[r, g] = −0.5·sign(X[r, g])(0이면 +0.5). 뒤집은 수를 돌려준다.

    X는 그 자리에서 바꾼다. 난수는 (len(rows), len(on_genes)) 한 번에 뽑는다(같은 생성기에서 rng.random()을 그 순서로
    부른 것과 같은 값). p ≤ 0(NaN 포함) 또는 rows·on_genes가 비면 rng를 부르지 않는다(옛 실행과 비트 동일).
    """
    if not p > 0 or len(rows) == 0 or len(on_genes) == 0:
        return 0
    r = np.asarray(rows, dtype=np.intp)
    g = np.asarray(on_genes, dtype=np.intp)
    hit = rng.random((r.size, g.size)) < p
    if not hit.any():
        return 0
    rr, gg = np.nonzero(hit)
    rows_hit, genes_hit = r[rr], g[gg]
    X[rows_hit, genes_hit] = np.where(X[rows_hit, genes_hit] > 0.0, -FLIP_VALUE, FLIP_VALUE)
    return int(rr.size)


def check_structure_config(structure_flip: object, structure_genes: object, bounded_dims: int) -> tuple[int, ...]:
    """GAConfig·SAConfig의 구조 뒤집기 칸을 검사하고 유전자 번호를 tuple로 돌려준다. 잘못되면 ValueError(한국어).

    확률은 0 이상 1 이하. 유전자 번호는 몸 유전자(앞쪽 bounded_dims차원, [-1, 1]로 잘림) 안의 서로 다른 0 이상의 정수.
    """
    p = float(structure_flip) if is_real(structure_flip) else math.nan  # type: ignore[arg-type]
    if not 0.0 <= p <= 1.0:  # NaN도 여기서 걸린다(무한대는 1보다 큼)
        raise ValueError(S.LEARNER_ERR_STRUCTURE_FLIP.format(value=structure_flip))
    if isinstance(structure_genes, (str, bytes)) or not isinstance(structure_genes, Sequence):
        raise ValueError(S.LEARNER_ERR_STRUCTURE_GENES.format(value=structure_genes, n=bounded_dims))
    genes = tuple(structure_genes)
    if (any(not is_int(k) or not 0 <= int(k) < bounded_dims for k in genes)
            or len(set(int(k) for k in genes)) != len(genes)):
        raise ValueError(S.LEARNER_ERR_STRUCTURE_GENES.format(value=list(genes), n=bounded_dims))
    return tuple(int(k) for k in genes)


def config_for_state(config: dict[str, Any]) -> dict[str, Any]:
    """state_dict()에 담을 설정 사전: 구조 뒤집기가 꺼졌으면(확률 0) structure_* 키를 뺀다(그 자리에서 바꾸고 돌려준다).

    v2.0 프로그램은 GAConfig(**config)로 상태를 읽어 모르는 키가 있으면 실패한다. 뒤집기를 쓰지 않는 실행의 상태는 v2.0이
    쓴 것과 같은 키·같은 바이트이고, 읽을 때 키가 없으면 데이터클래스 기본값(0, 빈 tuple)이다.
    """
    if not config.get("structure_flip"):
        config.pop("structure_flip", None)
        config.pop("structure_genes", None)
    return config

"""v6.0 M1 고침: 세계 만들기 가속(결과 비트 동일)의 테스트 (v6 설계 4.6·9.3 v6.0-2, 되돌릴 지점 1).

M1 bench 장면에서 바퀴 고르는 언덕차가 v6/v3.1 0.823(기준 ≤ 0.82)이었다. 그 장면 시간의 약 40%가 세계 만들기(언덕 선분 440개)였고
그 몫은 ×0.895만 빨랐다. 고친 것(episode.py):

- 언덕길·높이 지도 꼭짓점(생성기의 순수 함수 결과)을 인자 비트를 키로 몇 벌만 담아 둔다(_cached_terrain, TERRAIN_CACHE_SIZE).
- 지형 선분의 마찰, 레이 필터는 공개 속성이 부르는 C 함수(cpShapeSetFriction, cpShapeGetFilter·cpShapeSetFilter)를 바로
  부르고, 필터 범주는 모양 정보마다 한 번 센다.

여기서 지키는 것: 담아 둔 지형으로 만든 세계가 새로 만든 세계·v3.1 오라클 세계와 모양 하나하나(종류, 끝점, 반지름, 마찰,
탄성, 필터, 센서, 충돌 종류, 공간에 더한 순서 = Chipmunk hashid)까지 같고 끝까지 돌린 결과가 같다. 키가 0.0과 -0.0을 가른다.
담는 수가 상한을 넘지 않는다. 생성기 오류는 담지 않고 v3.1과 같은 EpisodeBuildError다. 나머지 비교(스텝마다, 평가기, 재생)는
tests/test_speed_identity.py·test_eval_identity.py·test_speed_runtime.py(T5: C 함수 길 대 공개 API 길)가 이미 한다.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pymunk
import pytest

from evolab.policy.genome import layout_for
from evolab.spec.io import compile_env
from evolab.world.physics2d import episode as cur
from tests.oracle import episode_v31 as old
from tests.test_speed_identity import ROOT, body_of, load, random_genomes, run_pair, same_static

TERRAIN_ENVS = ("wheels", "hill_car", "lander")
"""언덕길(바퀴·언덕차) 또는 높이 지도(착륙선, 패드 있음) 생성기가 있는 내장 환경(글라이더는 생성기가 없다)."""


def space_layout(ep: Any) -> list[tuple[Any, ...]]:
    """공간의 모든 모양(공간에 더한 순서)을 모양 하나하나의 값으로. Chipmunk hashid는 더한 순서의 번호다."""
    out: list[tuple[Any, ...]] = []
    for s in ep.space.shapes:
        if isinstance(s, pymunk.Segment):
            geom: tuple[Any, ...] = ("segment", tuple(s.a), tuple(s.b), s.radius)
        elif isinstance(s, pymunk.Circle):
            geom = ("circle", tuple(s.offset), s.radius)
        else:
            geom = ("poly", tuple(tuple(v) for v in s.get_vertices()), s.radius)
        f = s.filter
        out.append((*geom, s.friction, s.elasticity, f.group, f.categories, f.mask, s.sensor, s.collision_type,
                    s._hashid, s.body.body_type, s.mass, s.density))
    return out


@pytest.fixture
def empty_cache(monkeypatch: pytest.MonkeyPatch) -> dict[bytes, Any]:
    """이 테스트만의 빈 지형 저장소(다른 테스트가 담은 것과 섞이지 않게)."""
    store: dict[bytes, Any] = {}
    monkeypatch.setattr(cur, "_TERRAIN_CACHE", store)
    return store


@pytest.mark.parametrize("name", TERRAIN_ENVS)
def test_cached_terrain_builds_the_same_world_as_a_fresh_build_and_the_oracle(
        name: str, empty_cache: dict[bytes, Any]) -> None:
    spec, layout = load(name)
    genomes = random_genomes(layout, 3, seed=61, scale=0.8)
    trial = cur.Trial(6061)
    fresh = cur.Episode(spec, body_of(layout, genomes[0]), trial)  # 저장소가 비어 있음: 생성기를 부른다
    assert len(empty_cache) == 1
    for g in genomes:
        body = body_of(layout, g)
        hit = cur.Episode(spec, body, trial)  # 같은 시험: 담아 둔 지형
        assert len(empty_cache) == 1
        ref = old.Episode(spec, body, trial)
        assert space_layout(hit) == space_layout(ref)
        same_static(hit, ref)
    assert repr(fresh.geometry().bodies[0]) == repr(hit.geometry().bodies[0])  # 세계 지형 그림
    # 담아 둔 지형으로 만든 Episode를 끝까지 돌려도 오라클과 스텝마다 같다(run_pair의 지금 쪽은 저장소에서 꺼낸다)
    run_pair(spec, layout, genomes[1], trial, deep=False, values=True)
    assert len(empty_cache) == 1


def test_cached_terrain_values_are_immutable_python_floats(empty_cache: dict[bytes, Any]) -> None:
    for name in TERRAIN_ENVS:
        spec, layout = load(name)
        cur.Episode(spec, body_of(layout, layout.initial()), cur.Trial(7))
    # 바퀴 고르는 언덕차와 언덕 넘는 자동차는 같은 언덕길 생성기 인자(같은 시험이면 같은 언덕 시드)라 한 벌을 함께 쓸 수 있다
    assert 2 <= len(empty_cache) <= len(TERRAIN_ENVS)
    for value in empty_cache.values():
        assert isinstance(value, tuple)
        xs, ys = value[0], value[1]
        assert isinstance(xs, tuple) and isinstance(ys, tuple) and len(xs) == len(ys) >= 2
        assert all(type(v) is float for v in (*xs, *ys))


def test_terrain_cache_stays_within_its_size(empty_cache: dict[bytes, Any]) -> None:
    spec, layout = load("hill_car")
    body = body_of(layout, layout.initial())
    for seed in range(3 * cur.TERRAIN_CACHE_SIZE + 1):
        cur.Episode(spec, body, cur.Trial(seed))
        assert 1 <= len(empty_cache) <= cur.TERRAIN_CACHE_SIZE


def _hills_world(base_y: float) -> Any:
    data = json.loads((ROOT / "hill_car.evoenv").read_text(encoding="utf-8"))
    for g in data["generators"]:
        if g["kind"] == "hills":
            g["base_y"] = base_y
            g["flat_start"] = 400.0  # 끝까지 평지: 높이 = base_y + 0 x (...)라 0.0과 -0.0의 차이가 꼭짓점에 남을 수 있다
    return compile_env(json.dumps(data, ensure_ascii=False))


def test_terrain_cache_key_tells_zero_from_negative_zero(empty_cache: dict[bytes, Any]) -> None:
    """같은 값(==)이어도 비트가 다른 인자(0.0과 -0.0)는 다른 키다: 꺼낸 지형이 그 인자로 새로 만든 것·오라클과 같다."""
    pos, neg = _hills_world(0.0), _hills_world(-0.0)
    trial = cur.Trial(5)
    lay = layout_for(pos)
    body = body_of(lay, lay.initial())
    a = cur.Episode(pos, body, trial)
    b = cur.Episode(neg, body, trial)
    assert len(empty_cache) == 2
    assert space_layout(a) == space_layout(old.Episode(pos, body, trial))
    assert repr(space_layout(b)) == repr(space_layout(old.Episode(neg, body, trial)))
    assert repr(b.geometry()) == repr(old.Episode(neg, body, trial).geometry())


def test_terrain_generator_errors_are_the_same_build_error_and_are_not_kept(empty_cache: dict[bytes, Any]) -> None:
    data = json.loads((ROOT / "hill_car.evoenv").read_text(encoding="utf-8"))
    for g in data["generators"]:
        if g["kind"] == "hills":
            g["x_range"] = [10.0, 10.0]  # 빈 구간: 생성기의 ValueError
    spec = compile_env(json.dumps(data, ensure_ascii=False))
    errors = []
    for mod in (cur, old):
        with pytest.raises(mod.EpisodeBuildError) as info:
            mod.Episode(spec, None, mod.Trial(3))
        errors.append(str(info.value))
    assert errors[0] == errors[1]
    assert not empty_cache


def test_public_path_writes_the_same_shape_values_as_the_c_path(monkeypatch: pytest.MonkeyPatch,
                                                                empty_cache: dict[bytes, Any]) -> None:
    """_CP_FAST가 거짓이면(pymunk 내부가 바뀐 판) 마찰·필터를 공개 속성으로 쓴다: 모양 값이 C 함수 길과 같다."""
    for name in ("wheels", "track_car", "lander", "jump"):
        spec, layout = load(name)
        body = body_of(layout, random_genomes(layout, 1, seed=9)[0])
        fast = cur.Episode(spec, body, cur.Trial(19))
        with monkeypatch.context() as mp:
            mp.setattr(cur, "_CP_FAST", False)
            public = cur.Episode(spec, body, cur.Trial(19))
        assert space_layout(fast) == space_layout(public) == space_layout(old.Episode(spec, body, cur.Trial(19)))


def test_track_walls_are_closed_loops_like_the_oracle() -> None:
    spec, layout = load("track_car")
    a = cur.Episode(spec, None, cur.Trial(4, 0))
    b = old.Episode(spec, None, old.Trial(4, 0))
    walls = [s for s in space_layout(a) if s[0] == "segment"]
    assert walls and walls == [s for s in space_layout(b) if s[0] == "segment"]
    assert np.isfinite(np.asarray([w[1] + w[2] for w in walls], dtype=float)).all()

v2.0 0단계(설계 T1·T2)의 기준 자료. v2 이전 코드(v1.2, 커밋 3dbcb95)의 dump_env로 쓴 글과 env_hash다.
<이름>.json은 저장소 environments/<이름>.evoenv와 tests/data/<이름>/run/envs/v000.evoenv를 그 코드로 정규 형식으로 쓴 글,
env_hash.txt는 "이름 해시 원본 파일" 줄. 만든 방법과 v1_regress.npz·schema_fields_v1.json은 tests/data/make_v1_reference.py 첫머리.
테스트(tests/test_format_v2.py)는 읽기만 하고 여기는 고치지 않는다.

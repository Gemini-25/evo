v2.1 0'단계(설계 T3의 v2.1 추가분)의 기준 자료. v2.0 코드(커밋 4d3e3e6)의 dump_env로 쓴 글과 env_hash다.
<이름>.json은 형식 2 파일(저장소 environments/jump.evoenv·glider.evoenv, v2.0의 새 칸·종류·값을 모두 쓴 all_v2_features)을
그 코드로 정규 형식으로 쓴 글, env_hash.txt는 "이름 해시 원본 파일" 줄. 만든 방법은 tests/data/make_v2_reference.py 첫머리.
테스트(tests/test_format_v2.py)는 읽기만 하고 여기는 고치지 않는다.

"""v6 오라클: v3.1 출시판(커밋 a96e3bd)의 런타임·평가기를 얼린 사본 (v6 설계 3.2·7.3·D41).

- `episode_v31.py` = 그 커밋의 `src/evolab/world/physics2d/episode.py`, `evaluate_v31.py` = `src/evolab/engine/evaluate.py`를 바이트
  그대로 복사하고 첫 줄에 출처 주석을 더한 것이다. `evaluate_v31.py`만 Episode import 한 줄을 `tests.oracle.episode_v31`로 바꿨다.
  두 파일의 sha256은 `SOURCES.txt`에 있고 `tests/test_oracle_sane.py`(T0)가 실수로 고친 것을 막는다.
- 모듈 안에 상대 import가 없다: 오라클이 부르는 다른 모듈(센서·수식·지형·신경망·문구 등)은 지금 코드의 것을 쓴다. 그래서 그 모듈이
  결과를 바꾸는 변경을 받으면 오라클 비교는 뜻을 잃는다(그 묶음에서 이 폴더를 지운다). 늦어도 v4 0단계에서 지운다(D41).
- 쓰는 법(v6 설계 7.3 계약):

      from tests.oracle.evaluate_v31 import evaluate as evaluate_v31
      from tests.oracle.episode_v31 import Episode as EpisodeV31

  오라클의 `EvalResult`·`Trajectory`·`PairObservation`·`EpisodeResult`·`EpisodeBuildError`는 지금 코드의 같은 이름과 다른 클래스다.
  결과는 칸마다 `tobytes()`·`repr`로 비교하고 데이터클래스 `==`나 `isinstance`로 섞어 비교하지 않는다. 오라클 Episode에는 지금
  코드의 `Trial`(seeds가 만든 것)을 그대로 넘겨도 된다(이름으로만 읽음). 오라클 Episode의 빠른 길 깃발은 `episode_v31._CP_FAST`다.
- `tests` 폴더를 꾸러미로 부르려고 pyproject.toml의 pytest 설정에 `pythonpath = ["."]`를 두었다(`uv run pytest`도 저장소 맨 위를
  찾음). pytest 밖 스크립트는 저장소 맨 위를 `sys.path`에 넣고 부른다. 배포본(exe)에는 묶지 않는다.
"""

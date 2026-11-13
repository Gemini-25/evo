"""v6.0 0단계 오라클 점검 (v6 설계 8.1 T0, 3.2, 7.3, D41).

- 오라클(tests/oracle: v3.1 출시판 episode.py·evaluate.py의 바이트 사본)을 불러올 수 있고, 지금 런타임과 다른 모듈이다.
- 오라클 파일의 sha256이 SOURCES.txt와 같다(실수로 고친 것을 막음, 결과 해시가 아니라 소스 파일 해시). 첫 줄 출처 주석과
  evaluate_v31의 import 한 줄 말고는 원본과 바이트가 같다(원본 sha256으로 확인, git이 그 커밋을 가지고 있으면 git과도 비교).
- 오라클 모듈 안에 상대 import가 없고, 배포 코드(src)가 오라클을 부르지 않는다.
- 픽스처 챔피언 6개(tests/data/v6_champions)가 지금 내장 환경 배치와 맞고, 오라클로 돌린 결과가 지금 평가기와 바이트 같다
  (한 시험씩, 오라클이 실제로 돈다는 확인. 넓은 비교는 R·E 묶음의 T1·T2·T9).
"""

from __future__ import annotations

import ast
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from evolab import paths

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests" / "oracle"
CHAMPIONS = ROOT / "tests" / "data" / "v6_champions"
SRC = ROOT / "src" / "evolab"

OLD_IMPORT = b"from evolab.world.physics2d.episode import Episode, EpisodeBuildError, EpisodeResult, Trial\n"
NEW_IMPORT = b"from tests.oracle.episode_v31 import Episode, EpisodeBuildError, EpisodeResult, Trial\n"
ORACLE_SOURCES = {"episode_v31.py": "src/evolab/world/physics2d/episode.py", "evaluate_v31.py": "src/evolab/engine/evaluate.py"}
"""오라클 파일 → v3.1 출시판의 원본 경로(설계 7.1 0 묶음)."""
FIXTURE_KEYS = {"genome", "env_file", "env_sha256", "layout_signature", "generation", "run_seed", "score", "source"}
OLD_BUILTINS = ("lander", "track_car", "hill_car", "jump", "glider", "wheels")
"""옛 내장 예제 여섯(v6.1까지의 내장 전부). 이 파일의 옛 오라클(v3.1 사본)과 고정 챔피언은 이 여섯만 본다(스포츠 설계 4.5):
6.2.0부터 내장에 종목 프리셋(달리기 …)이 더해지고, 그것들은 얼린 출시 나무 쪽(tests/oracle/frozen.py)이 맡는다."""


def _old_builtin_envs() -> list[Path]:
    envs = [p for p in paths.builtin_envs() if p.stem in OLD_BUILTINS]
    assert [p.stem for p in envs] == list(OLD_BUILTINS)  # 여섯이 모두 있고 정해진 순서의 맨 앞
    return envs


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sources_table() -> tuple[str, dict[str, tuple[str, str, str, str]]]:
    """SOURCES.txt → (출처 커밋, 파일 → (원본 경로, 원본 git blob, 원본 sha256, 오라클 파일 sha256))."""
    commit = ""
    rows: dict[str, tuple[str, str, str, str]] = {}
    for line in (ORACLE / "SOURCES.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0].startswith("frozen"):  # 6.2.0: 얼린 출시 나무의 줄(tests/oracle/frozen.py가 읽고 tests/test_frozen_oracle.py가 지킨다)
            continue
        if parts[0] == "commit":
            commit = parts[1]
            continue
        name, src, blob, src_sha, oracle_sha = parts
        rows[name] = (src, blob, src_sha, oracle_sha)
    return commit, rows


def _original_bytes(name: str) -> bytes:
    """오라클 파일에서 첫 줄 출처 주석을 떼고(evaluate_v31은 import 한 줄을 되돌려) 원본 바이트를 되살린다."""
    data = (ORACLE / name).read_bytes()
    head, sep, body = data.partition(b"\n")
    assert sep and head.startswith(b"# "), name
    if name == "evaluate_v31.py":
        assert body.count(NEW_IMPORT) == 1 and body.count(OLD_IMPORT) == 0, name
        body = body.replace(NEW_IMPORT, OLD_IMPORT)
    return body


# ---------------------------------------------------------------------------
# 오라클 파일
# ---------------------------------------------------------------------------
def test_oracle_modules_import_and_are_separate_from_the_runtime() -> None:
    import tests.oracle.episode_v31 as ep31
    import tests.oracle.evaluate_v31 as ev31
    from evolab.engine import evaluate as ev
    from evolab.world.physics2d import episode as ep

    assert Path(ep31.__file__).resolve().parent == ORACLE and Path(ev31.__file__).resolve().parent == ORACLE
    # 오라클 평가기는 오라클 Episode를 쓴다(import 한 줄), 지금 코드의 것과는 다른 클래스·다른 빠른 길 깃발
    assert ev31.Episode is ep31.Episode and ev31.EpisodeBuildError is ep31.EpisodeBuildError
    assert ev31.EpisodeResult is ep31.EpisodeResult and ev31.Trial is ep31.Trial
    assert ep31.Episode is not ep.Episode and ev31.evaluate is not ev.evaluate
    assert ev.Episode is ep.Episode
    assert isinstance(ep31._CP_FAST, bool)
    # 계약(설계 7.3)에 쓰인 이름
    from tests.oracle.episode_v31 import Episode as EpisodeV31
    from tests.oracle.evaluate_v31 import evaluate as evaluate_v31

    assert callable(evaluate_v31) and isinstance(EpisodeV31, type)


def test_oracle_folder_holds_only_the_listed_files() -> None:
    files = sorted(p.name for p in ORACLE.iterdir() if p.is_file())
    # 6.2.0 바탕이 더한 둘: 얼린 v6.1.0 출시 나무의 묶음과 그것을 돌리는 frozen.py(스크립트는 scripts/ 폴더, tests/test_frozen_oracle.py)
    assert files == ["SOURCES.txt", "__init__.py", "episode_v31.py", "evaluate_v31.py", "frozen.py",
                     "v6.1.0-src.tar.gz"], files
    commit, rows = _sources_table()
    assert set(rows) == set(ORACLE_SOURCES)
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), commit


def test_oracle_files_match_sources_txt() -> None:
    """오라클 파일 sha256 = SOURCES.txt, 첫 줄은 출처 주석, 그 밖은 원본과 바이트 같음(원본 sha256)."""
    commit, rows = _sources_table()
    for name, (src, _blob, src_sha, oracle_sha) in rows.items():
        assert src == ORACLE_SOURCES[name], name
        data = (ORACLE / name).read_bytes()
        assert _sha(data) == oracle_sha, f"{name}가 바뀌었습니다: 오라클 파일은 고치지 않습니다(설계 D41)"
        first = data.split(b"\n", 1)[0].decode("utf-8")
        assert first.startswith(f"# 출처: 커밋 {commit[:12]}") and src in first, first
        assert b"\r\n" not in data, name  # .gitattributes tests/oracle/** -text: 윈도우 체크아웃도 줄바꿈 그대로
        assert _sha(_original_bytes(name)) == src_sha, name


def test_sources_txt_matches_git_when_the_commit_is_present() -> None:
    """저장소가 출처 커밋을 가지고 있으면(얕은 CI 체크아웃은 없을 수 있음) git의 원본과 blob·sha256이 같다. 읽기만 한다."""
    git = shutil.which("git")
    commit, rows = _sources_table()
    if git is None:
        pytest.skip("git 없음")
    has = subprocess.run([git, "-C", str(ROOT), "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True,
                         check=False)
    if has.returncode != 0:
        pytest.skip(f"이 체크아웃에 커밋 {commit[:12]}이 없음")
    for name, (src, blob, src_sha, _oracle_sha) in rows.items():
        got_blob = subprocess.run([git, "-C", str(ROOT), "rev-parse", f"{commit}:{src}"], capture_output=True, text=True,
                                  check=True).stdout.strip()
        assert got_blob == blob, name
        original = subprocess.run([git, "-C", str(ROOT), "cat-file", "blob", blob], capture_output=True, check=True).stdout
        assert _sha(original) == src_sha and original == _original_bytes(name), name


def test_oracle_has_no_relative_imports_and_src_never_imports_it() -> None:
    for name in ORACLE_SOURCES:
        tree = ast.parse((ORACLE / name).read_bytes())
        froms = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert all(n.level == 0 for n in froms), name
        oracle_imports = [n.module for n in froms if (n.module or "").startswith("tests")]
        assert oracle_imports == (["tests.oracle.episode_v31"] if name == "evaluate_v31.py" else []), name
    offenders = [p.relative_to(SRC) for p in SRC.rglob("*.py") if "tests.oracle" in p.read_text(encoding="utf-8")]
    assert offenders == []  # 배포본(exe)에 오라클을 묶지 않는다(설계 2.1)


# ---------------------------------------------------------------------------
# 픽스처 챔피언 (tests/data/v6_champions)
# ---------------------------------------------------------------------------
def _fixture(stem: str) -> dict[str, np.ndarray]:
    with np.load(CHAMPIONS / f"{stem}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def test_one_champion_per_builtin_env_matching_the_current_layout() -> None:
    from evolab.policy.genome import layout_for
    from evolab.spec.io import load_env

    envs = _old_builtin_envs()
    assert len(envs) == 6  # 착륙선, 트랙, 언덕차, 점프, 글라이더, 바퀴 고르는 언덕차
    assert [p.stem for p in paths.builtin_envs()][:6] == list(OLD_BUILTINS)  # 종목 프리셋은 그 뒤에 온다
    assert sorted(p.stem for p in CHAMPIONS.glob("*.npz")) == sorted(p.stem for p in envs)
    readme = (CHAMPIONS / "README.txt").read_text(encoding="utf-8")
    for env_file in envs:
        z = _fixture(env_file.stem)
        assert set(z) == FIXTURE_KEYS, env_file.stem
        assert f"{env_file.stem}.npz" in readme  # 출처 한 줄
        assert str(z["env_file"]) == env_file.name
        assert str(z["env_sha256"]) == _sha(env_file.read_bytes()), (
            f"{env_file.name}가 챔피언을 고를 때와 다릅니다: 픽스처를 다시 고르세요(README)")
        layout = layout_for(load_env(env_file))
        genome = z["genome"]
        assert genome.dtype == np.float64 and genome.shape == (layout.dim,), env_file.stem
        assert np.isfinite(genome).all(), env_file.stem
        assert str(z["layout_signature"]) == layout.signature(), env_file.stem
        assert int(z["generation"]) >= 1 and np.isfinite(float(z["score"]))


def test_oracle_and_current_evaluator_agree_on_each_champion() -> None:
    """오라클이 실제로 돈다: 챔피언마다 검증 시험 0번(궤적 기록 포함)을 두 평가기로 돌려 결과 칸이 바이트 같다."""
    from evolab.engine.evaluate import evaluate
    from evolab.engine.seeds import validation_trials
    from evolab.policy.genome import layout_for
    from evolab.spec.io import load_env
    from tests.oracle.evaluate_v31 import evaluate as evaluate_v31

    trial = validation_trials(1)[0]
    for env_file in _old_builtin_envs():
        spec = load_env(env_file)
        layout = layout_for(spec)
        genomes = _fixture(env_file.stem)["genome"][None, :]
        a = evaluate(spec, layout, genomes, [(0, trial)], record=[True])
        b = evaluate_v31(spec, layout, genomes, [(0, trial)], record=[True])
        for f in ("scores", "steps", "blocks", "action_abs", "weighted"):
            assert getattr(a, f).tobytes() == getattr(b, f).tobytes(), (env_file.stem, f)
        assert a.end_events == b.end_events and a.outcomes == b.outcomes, env_file.stem
        ta, tb = a.trajectories[0], b.trajectories[0]
        assert ta is not None and tb is not None
        assert ta.poses.tobytes() == tb.poses.tobytes() and ta.outputs.tobytes() == tb.outputs.tobytes(), env_file.stem
        assert int(a.steps[0]) > 0, env_file.stem

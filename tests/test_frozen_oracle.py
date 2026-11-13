"""6.2.0 바탕 단계 T0: 얼린 v6.1.0 오라클 (스포츠 설계 D52·D56, 6.1의 T0).

- `tests/oracle/v6.1.0-src.tar.gz`의 sha256과 출시 커밋 해시(40자, 관문 G0)가 `SOURCES.txt`에 있고 파일과 맞는다.
- 묶음에는 그 커밋의 `src/evolab`과 `environments`만 있고, 풀어 보면 프로그램 버전이 6.1.0이다.
- `frozen_run`이 내장 예제 하나를 얼린 나무의 코드로 돌려 결과를 돌려준다. 같은 스크립트를 지금 나무로 돌린 것(`live_run`)과 칸마다
  바이트가 같다(견본 하나. 넓은 비교는 묶음 R의 tests/test_frozen_identity.py).
- `SOURCES.txt`의 `frozen-env` 목록(그 커밋에 있던 tests/data의 환경)이 지금도 그 바이트 그대로 있다.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

import evolab
from evolab.spec import model as m
from tests.oracle import frozen

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "v6.1.0"
SCRIPT = "tests/oracle/scripts/eval_builtin.py"
RESULT_FIELDS = ("genomes", "scores", "steps", "blocks", "action_abs", "weighted", "poses", "outputs")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_sources_name_the_release_commit_and_the_archive_hash() -> None:
    table = frozen.releases()
    assert RELEASE in table
    commit, name, sha = table[RELEASE]
    assert re.fullmatch(r"[0-9a-f]{40}", commit), commit  # 줄인 해시가 아니라 40자(G0)
    assert name == "v6.1.0-src.tar.gz" and frozen.archive_path(RELEASE) == ROOT / "tests" / "oracle" / name
    assert _sha(frozen.archive_path(RELEASE).read_bytes()) == sha
    # 옛 오라클 파일 둘은 6.2.0에도 그대로 있다(6.3.0 바탕 A가 거둔다: 설계 D52)
    assert (ROOT / "tests" / "oracle" / "episode_v31.py").is_file() and (ROOT / "tests" / "oracle" / "evaluate_v31.py").is_file()


def test_archive_holds_only_the_release_source_and_builtin_envs() -> None:
    with tarfile.open(frozen.archive_path(RELEASE), "r:gz") as tar:
        members = tar.getmembers()
    files = [mem.name for mem in members if mem.isfile()]
    assert files and all(mem.isfile() or mem.isdir() for mem in members)  # 링크·장치 파일 없음
    assert all(name.startswith(("src/evolab/", "environments/")) for name in files), files[:5]
    assert not [name for name in files if name.endswith(".pyc") or "__pycache__" in name]
    envs = sorted(Path(name).stem for name in files if name.startswith("environments/") and name.endswith(".evoenv"))
    assert envs == sorted(["lander", "track_car", "hill_car", "jump", "glider", "wheels"])  # v6.1.0의 내장 예제 여섯
    assert "src/evolab/__init__.py" in files and "src/evolab/world/physics2d/episode.py" in files


def test_archive_matches_the_release_commit_when_git_has_it() -> None:
    """git이 그 커밋을 가지고 있으면(개발 저장소) 묶음의 파일 하나하나가 그 커밋의 blob과 같다. 없으면(배포 사본, 얕은 복제) 건너뜀."""
    git = shutil.which("git")
    commit = frozen.releases()[RELEASE][0]
    if git is None or subprocess.run([git, "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT, capture_output=True,
                                     check=False).returncode != 0:
        pytest.skip("release commit is not in this checkout")
    listed = subprocess.run([git, "ls-tree", "-r", "--name-only", commit, "--", "src/evolab", "environments"], cwd=ROOT,
                            capture_output=True, check=True).stdout.decode("utf-8").split("\n")
    tree = frozen.frozen_tree(RELEASE)
    on_disk = sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert on_disk == sorted(name for name in listed if name)
    for name in ("src/evolab/world/physics2d/episode.py", "src/evolab/spec/io.py", "environments/lander.evoenv"):
        blob = subprocess.run([git, "cat-file", "blob", f"{commit}:{name}"], cwd=ROOT, capture_output=True, check=True).stdout
        assert (tree / name).read_bytes() == blob, name


def test_frozen_tree_is_the_610_program_and_is_unpacked_once() -> None:
    tree = frozen.frozen_tree(RELEASE)
    assert tree == frozen.frozen_tree(RELEASE) and (tree / "src" / "evolab" / "__init__.py").is_file()
    init = (tree / "src" / "evolab" / "__init__.py").read_text(encoding="utf-8")
    assert '__version__ = "6.1.0"' in init and evolab.__version__ != "6.1.0"
    model = (tree / "src" / "evolab" / "spec" / "model.py").read_text(encoding="utf-8")
    assert "CURRENT_VERSION = 3" in model and m.CURRENT_VERSION == 4  # 얼린 쪽은 형식 3까지, 지금은 4


def test_frozen_run_evaluates_a_builtin_env_like_the_working_tree() -> None:
    old = frozen.frozen_run(RELEASE, SCRIPT, "lander", 2, 2)
    new = frozen.live_run(SCRIPT, "lander", 2, 2)
    assert str(old["version"]) == "6.1.0" and str(new["version"]) == evolab.__version__
    assert list(old["builtin"]) == sorted(["lander", "track_car", "hill_car", "jump", "glider", "wheels"])
    assert set(old["builtin"]) <= set(new["builtin"])  # 지금 나무에는 종목 프리셋이 더 있을 수 있다
    assert int(old["env_format"]) == int(new["env_format"]) == 1 and str(old["env_hash"]) == str(new["env_hash"])
    assert int(old["dim"]) == int(new["dim"]) > 0 and old["scores"].shape == (4,)
    assert int(old["steps"].min()) > 0
    for field in RESULT_FIELDS:
        assert old[field].dtype == new[field].dtype and old[field].shape == new[field].shape, field
        assert old[field].tobytes() == new[field].tobytes(), field
    assert str(old["ends"]) == str(new["ends"])


def test_frozen_run_reports_failures_with_the_error_text(tmp_path: Path) -> None:
    script = tmp_path / "boom.py"
    script.write_text("import sys\nimport evolab\nsys.exit('frozen says ' + evolab.__version__)\n", encoding="utf-8")
    with pytest.raises(frozen.FrozenRunError, match="frozen says 6.1.0"):
        frozen.frozen_run(RELEASE, script)
    quiet = tmp_path / "quiet.py"  # 결과 파일을 쓰지 않고 끝나도 실패
    quiet.write_text("import evolab\n", encoding="utf-8")
    with pytest.raises(frozen.FrozenRunError):
        frozen.live_run(quiet)
    with pytest.raises(frozen.FrozenRunError, match="unknown frozen release"):
        frozen.frozen_run("v0.0.0", script)
    with pytest.raises(frozen.FrozenRunError, match="no such script"):
        frozen.frozen_run(RELEASE, tmp_path / "missing.py")


def test_frozen_run_passes_arguments_and_returns_arrays(tmp_path: Path) -> None:
    script = tmp_path / "echo.py"
    script.write_text(
        "import sys\nimport numpy as np\nimport evolab\n"
        "np.savez(sys.argv[1], args=np.array(sys.argv[2:]), where=np.array(evolab.__file__), n=np.arange(3))\n",
        encoding="utf-8")
    out = frozen.frozen_run(RELEASE, script, "가나다", 7, 0.5)
    assert list(out["args"]) == ["가나다", "7", "0.5"] and out["n"].tolist() == [0, 1, 2]
    assert Path(str(out["where"])).resolve().is_relative_to(frozen.frozen_tree(RELEASE).resolve())
    live = frozen.live_run(script, "x")
    assert Path(str(live["where"])).resolve().is_relative_to((ROOT / "src").resolve())


def test_frozen_env_list_is_still_on_disk_byte_for_byte() -> None:
    envs = frozen.frozen_envs(RELEASE)
    assert len(envs) >= 13 and all(name.startswith("tests/data/") for name in envs)
    assert "tests/data/v2_structure/wheels_fixed1.evoenv" in envs and "tests/data/v1_dump/lander.json" in envs
    for name, sha in envs.items():
        assert _sha((ROOT / name).read_bytes()) == sha, f"{name} changed since {RELEASE}"
    # 6.2.0이 더한 환경(측정에 쓴 달리기 원형 등)은 이 목록에 없다: 얼린 v6.1이 읽지 못하는 형식일 수 있다
    assert not [name for name in envs if "presets" in name]

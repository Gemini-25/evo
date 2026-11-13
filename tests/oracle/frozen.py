"""얼린 출시 나무를 딴 프로세스에서 돌려 견주는 오라클 (스포츠 설계 D52·D56, 6.2.0 바탕).

`tests/oracle/<출시>-src.tar.gz`는 그 출시 커밋의 `src/evolab`과 `environments`를 `git archive`로 묶은 것이다(커밋 해시와 sha256은
`SOURCES.txt`의 `frozen` 줄, tests/test_frozen_oracle.py가 지킨다). import를 고쳐 쓰지 않는다: 묶음을 임시 폴더에 풀고 그 `src`를
`PYTHONPATH` 맨 앞에 둔 하위 프로세스가 스크립트를 돌린다. 그래서 얼린 쪽은 런타임·검사·수식·학습기까지 모두 그 출시의 코드다
(옛 `episode_v31.py`·`evaluate_v31.py`는 지금의 `spec`·`policy`를 불러 썼다).

쓰는 법(묶음 사이의 계약):

    from tests.oracle.frozen import frozen_run, live_run
    old = frozen_run("v6.1.0", "tests/oracle/scripts/eval_builtin.py", "lander", 3)   # 얼린 v6.1.0
    new = live_run("tests/oracle/scripts/eval_builtin.py", "lander", 3)               # 지금 작업 나무(같은 방식의 하위 프로세스)
    assert old["scores"].tobytes() == new["scores"].tobytes()

스크립트의 약속:
- `sys.argv[1]` = 결과를 쓸 `.npz` 경로, `sys.argv[2:]` = 부르는 쪽이 준 인자(글로 바뀜). `__name__ == "__main__"`으로 돈다.
- 결과는 `numpy.savez(sys.argv[1], 이름=배열, …)`로 쓴다. 객체 배열은 쓰지 않는다(`allow_pickle=False`로 읽음). 글은 `numpy.array("글")`
  이나 글 배열로, 여러 줄 표는 JSON 글 하나로 담는다.
- 두 나무에서 같은 스크립트가 돌아야 하므로 두 출시에 모두 있는 이름만 불러 쓴다(새 이름은 `getattr`·`try`로 가른다).
- 스크립트는 `tests`를 꾸러미로 부르지 않는다(얼린 나무에는 `tests`가 없다). 같은 폴더의 다른 스크립트는 불러도 된다(스크립트 폴더가
  `sys.path` 맨 앞에 있다).
- 작업 폴더는 저장소 맨 위다: 인자로 준 `tests/data/…` 같은 상대 경로를 두 나무의 스크립트가 같은 파일로 읽는다. 내장 환경은
  `evolab.paths.builtin_envs()`로 찾는다(얼린 쪽은 묶음 안의 `environments`, 지금 쪽은 저장소의 것).

돌려주는 것은 `{이름: numpy 배열}` 사전이다. 하위 프로세스가 0이 아닌 값으로 끝나면 `FrozenRunError`(끝난 값과 표준 오류의 끝부분).
환경 변수는 그대로 물려준다(`EVOLAB_DATA_DIR`·`EVOLAB_STATE_DIR`·`QT_QPA_PLATFORM` 등): 테스트가 격리해 둔 폴더를 얼린 쪽도 쓴다.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
from pathlib import Path

import numpy as np

ORACLE = Path(__file__).resolve().parent
ROOT = ORACLE.parents[1]
SOURCES = ORACLE / "SOURCES.txt"
DEFAULT_TIMEOUT = 600.0
"""하위 프로세스 한 번의 시간 한도(초). 느린 CI도 넉넉히 넘기지 않게."""

_BOOT = """
import os, runpy, sys
tree, script = sys.argv[1], sys.argv[2]
src = os.path.join(tree, "src")
if src not in sys.path:
    sys.path.insert(0, src)
import evolab
here = os.path.realpath(os.path.dirname(evolab.__file__))
want = os.path.realpath(os.path.join(src, "evolab"))
if here != want:
    sys.exit("frozen oracle: evolab was imported from %s, not from %s" % (here, want))
sys.argv = [script] + sys.argv[3:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
"""
"""하위 프로세스의 첫 코드: 그 나무의 evolab이 불렸는지 확인한 뒤 스크립트를 `__main__`으로 돌린다."""


class FrozenRunError(RuntimeError):
    """얼린(또는 지금) 나무의 스크립트가 실패했다."""


_lock = threading.Lock()
_trees: dict[str, Path] = {}


def releases() -> dict[str, tuple[str, str, str]]:
    """SOURCES.txt의 `frozen` 줄: 출시 이름 → (커밋 해시 40자, 묶음 파일 이름, 묶음 sha256)."""
    out: dict[str, tuple[str, str, str]] = {}
    for line in SOURCES.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 7 and parts[0] == "frozen" and parts[2] == "commit" and parts[4] == "file":
            out[parts[1]] = (parts[3], parts[5], parts[6])
    return out


def frozen_envs(release: str) -> dict[str, str]:
    """SOURCES.txt의 `frozen-env` 줄: 그 출시 커밋에 있던 tests/data의 환경 파일(저장소 상대 경로) → 그때의 sha256."""
    out: dict[str, str] = {}
    for line in SOURCES.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "frozen-env" and parts[1] == release:
            out[parts[2]] = parts[3]
    return out


def archive_path(release: str) -> Path:
    """그 출시의 묶음 파일. 모르는 출시면 KeyError."""
    return ORACLE / releases()[release][1]


def frozen_tree(release: str) -> Path:
    """그 출시의 묶음을 임시 폴더에 푼 자리(`src/evolab`, `environments`가 그 아래). 프로세스마다 한 번 풀고 끝날 때 지운다.

    묶음의 sha256이 SOURCES.txt와 다르면 FrozenRunError(고쳐진 묶음으로 견주지 않는다).
    """
    with _lock:
        tree = _trees.get(release)
        if tree is not None and tree.is_dir():
            return tree
        try:
            _commit, name, want = releases()[release]
        except KeyError:
            raise FrozenRunError(f"unknown frozen release {release!r}: not in {SOURCES.name}") from None
        archive = ORACLE / name
        got = hashlib.sha256(archive.read_bytes()).hexdigest()
        if got != want:
            raise FrozenRunError(f"{name}: sha256 {got} does not match {SOURCES.name} ({want})")
        tree = Path(tempfile.mkdtemp(prefix=f"evolab-frozen-{release}-"))
        atexit.register(shutil.rmtree, tree, ignore_errors=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tree, filter="data")
        _trees[release] = tree
        return tree


def _run(tree: Path, script: str | Path, args: tuple[object, ...], timeout: float) -> dict[str, np.ndarray]:
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = ROOT / script_path
    if not script_path.is_file():
        raise FrozenRunError(f"no such script: {script_path}")
    env = dict(os.environ)
    src = str(tree / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["PYTHONIOENCODING"] = "utf-8"
    with tempfile.TemporaryDirectory(prefix="evolab-frozen-out-") as tmp:
        out = Path(tmp) / "result.npz"
        cmd = [sys.executable, "-X", "utf8", "-c", _BOOT, str(tree), str(script_path), str(out), *map(str, args)]
        try:
            proc = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise FrozenRunError(f"{script_path.name}: no result after {timeout:.0f}s") from exc
        if proc.returncode != 0 or not out.is_file():
            tail = proc.stderr.decode("utf-8", "replace")[-4000:]
            raise FrozenRunError(f"{script_path.name} failed in {tree} (exit {proc.returncode}):\n{tail}")
        with np.load(out, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}


def frozen_run(release: str, script: str | Path, *args: object, timeout: float = DEFAULT_TIMEOUT) -> dict[str, np.ndarray]:
    """얼린 출시(release, 예 "v6.1.0")의 코드로 script를 돌린 결과 `{이름: 배열}`.

    script는 저장소 맨 위 기준 경로나 절대 경로. args는 글로 바뀌어 `sys.argv[2:]`가 된다(`sys.argv[1]`은 결과 .npz 경로).
    """
    return _run(frozen_tree(release), script, args, timeout)


def live_run(script: str | Path, *args: object, timeout: float = DEFAULT_TIMEOUT) -> dict[str, np.ndarray]:
    """지금 작업 나무의 코드로 같은 script를 같은 방식(하위 프로세스, 같은 약속)으로 돌린 결과. frozen_run과 견줄 짝."""
    return _run(ROOT, script, args, timeout)


__all__ = ["DEFAULT_TIMEOUT", "FrozenRunError", "archive_path", "frozen_envs", "frozen_run", "frozen_tree", "live_run",
           "releases"]

"""코드 규칙 검사.

- 화면 문구는 strings.py에만 둔다: 다른 소스의 문자열 리터럴(문서 문자열 제외)에 한글 금지.
- 결정성: numpy 전역 난수(np.random.rand 등) 금지. 난수는 Generator나 전용 RandomState로만.
"""

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "evolab"
STRINGS_FILE = SRC / "app" / "strings.py"
HANGUL = re.compile(r"[가-힣ㄱ-ㆎ]")

# 시드를 받는 생성기와 형식만 허용한다.
ALLOWED_NP_RANDOM = {"default_rng", "Generator", "RandomState", "SeedSequence", "PCG64", "PCG64DXSM", "BitGenerator"}


def _sources() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if p != STRINGS_FILE)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def test_user_visible_text_only_in_strings_py() -> None:
    offenders: list[str] = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and HANGUL.search(node.value)
            ):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}: {node.value[:40]!r}")
    assert not offenders, "strings.py 밖의 한글 문자열:\n" + "\n".join(offenders)


def test_no_global_numpy_random() -> None:
    offenders: list[str] = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # np.random.<name> 또는 numpy.random.<name>
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "random"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id in {"np", "numpy"}
                and node.attr not in ALLOWED_NP_RANDOM
            ):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}: np.random.{node.attr}")
        text = path.read_text(encoding="utf-8")
        if re.search(r"from\s+numpy\.random\s+import|from\s+numpy\s+import\s+random", text):
            offenders.append(f"{path.relative_to(SRC)}: numpy.random import")
    assert not offenders, "numpy 전역 난수 사용:\n" + "\n".join(offenders)

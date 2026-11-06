"""코드 규칙 검사.

- 화면 문구는 strings.py에만 둔다: 다른 소스의 문자열 리터럴(문서 문자열 제외)에 한글 금지.
- 결정성: numpy 전역 난수(np.random.rand 등) 금지. 난수는 시드를 준 Generator나 전용 RandomState로만.
"""

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "evolab"
STRINGS_FILE = SRC / "app" / "strings.py"
HANGUL = re.compile(r"[가-힣ㄱ-ㆎ]")

# numpy.random에서 쓸 수 있는 이름: 시드를 받는 생성기와 형식만.
ALLOWED_NP_RANDOM = {"default_rng", "Generator", "RandomState", "SeedSequence", "PCG64", "PCG64DXSM", "BitGenerator"}
# 인자 없이 부르면 시간이나 OS 난수로 시드가 정해지는 생성자.
NEEDS_SEED = {"default_rng", "RandomState", "SeedSequence", "PCG64", "PCG64DXSM"}


def _sources() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if p != STRINGS_FILE)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """문서 문자열: 값으로 쓰이지 않고 문장으로만 놓인 문자열(모듈·클래스·함수 첫 줄, 필드 아래 설명)."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            ids.add(id(node.value))
    return ids


def numpy_random_violations(source: str) -> list[str]:
    """numpy 전역 난수를 쓰는 곳을 줄 번호와 함께 돌려준다. 별칭 import도 따라간다."""
    tree = ast.parse(source)
    numpy_names: set[str] = set()  # numpy 모듈에 묶인 이름 (np, numpy, onp ...)
    random_names: set[str] = set()  # numpy.random 모듈에 묶인 이름 (npr, random ...)
    seeded_ctor_names: set[str] = set()  # from numpy.random import default_rng 등
    out: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy":
                    numpy_names.add(alias.asname or "numpy")
                elif alias.name == "numpy.random":
                    if alias.asname:
                        random_names.add(alias.asname)
                    else:
                        numpy_names.add("numpy")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "numpy":
                for alias in node.names:
                    if alias.name == "random":
                        random_names.add(alias.asname or "random")
            elif node.module == "numpy.random":
                for alias in node.names:
                    if alias.name not in ALLOWED_NP_RANDOM:
                        out.append(f"{node.lineno}: from numpy.random import {alias.name}")
                    elif alias.name in NEEDS_SEED:
                        seeded_ctor_names.add(alias.asname or alias.name)

    def random_module_expr(expr: ast.AST) -> bool:
        """expr가 numpy.random 모듈 자체를 가리키는가 (np.random, npr)."""
        if isinstance(expr, ast.Name):
            return expr.id in random_names
        return (
            isinstance(expr, ast.Attribute)
            and expr.attr == "random"
            and isinstance(expr.value, ast.Name)
            and expr.value.id in numpy_names
        )

    attribute_bases: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attribute_bases.add(id(node.value))

    for node in ast.walk(tree):
        # np.random.<이름>, npr.<이름>
        if isinstance(node, ast.Attribute) and random_module_expr(node.value):
            if node.attr not in ALLOWED_NP_RANDOM:
                out.append(f"{node.lineno}: numpy.random.{node.attr}")
        # r = np.random 처럼 모듈을 통째로 다른 이름에 넘기는 것
        if random_module_expr(node) and isinstance(node, ast.Attribute) and id(node) not in attribute_bases:
            out.append(f"{node.lineno}: numpy.random module used as a value")
        # 시드 없이 만든 생성기
        if isinstance(node, ast.Call) and not node.args and not node.keywords:
            func = node.func
            name = None
            if isinstance(func, ast.Attribute) and random_module_expr(func.value):
                name = func.attr
            elif isinstance(func, ast.Name) and func.id in seeded_ctor_names:
                name = func.id
            if name is not None and name in NEEDS_SEED | seeded_ctor_names:
                out.append(f"{node.lineno}: {name}() without a seed")
    return sorted(set(out))


def test_user_visible_text_only_in_strings_py() -> None:
    offenders: list[str] = []
    for path in _sources():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                # 원문 기준: 글자 그대로 적힌 한글만 잡는다("\uac00" 같은 이스케이프는 코드용이라 허용).
                text = ast.get_source_segment(source, node) or ""
                if HANGUL.search(text):
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno}: {text[:40]!r}")
    assert not offenders, "strings.py 밖의 한글 문자열:\n" + "\n".join(offenders)


def test_no_global_numpy_random_in_sources() -> None:
    offenders = [
        f"{path.relative_to(SRC)}:{v}"
        for path in _sources()
        for v in numpy_random_violations(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, "numpy 전역 난수 사용:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np\nnp.random.rand(3)",
        "import numpy as onp\nonp.random.normal()",
        "import numpy.random as npr\nnpr.rand()",
        "from numpy import random\nrandom.uniform()",
        "from numpy.random import rand",
        "import numpy as np\nr = np.random\nr.uniform()",
        "import numpy as np\nnp.random.default_rng()",
        "from numpy.random import default_rng as mk\nmk()",
        "import numpy.random\nnumpy.random.seed(1)",
    ],
)
def test_rule_catches_global_random(source: str) -> None:
    assert numpy_random_violations(source)


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np\nrng = np.random.default_rng(1)\nrng.random()",
        "from numpy.random import SeedSequence, default_rng\nss = SeedSequence([1, 2])\ndefault_rng(ss)",
        "import numpy as np\nrs = np.random.RandomState(5)\nrs.randn(3)",
        "import random\nrandom.random()",  # 표준 라이브러리 random은 이 규칙의 대상이 아니다
        "x = 'np.random.rand() in a string'",
    ],
)
def test_rule_allows_seeded_generators(source: str) -> None:
    assert numpy_random_violations(source) == []

"""pycma 불러오기. 쓰지 않는 그림 기능(matplotlib) 경고를 숨긴다."""

from __future__ import annotations

import warnings
from types import ModuleType


def import_cma() -> ModuleType:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not import matplotlib")
        import cma
    return cma

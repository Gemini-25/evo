"""환경 파일의 에디터 전용 칸(editor) 읽기·쓰기 (v2 설계 D25).

editor 구역은 해시(env_hash)와 런타임에서 빠지고 형식 번호와도 무관하다. v1.x도 editor를 자유 사전으로 읽으므로
기본 보기를 저장한 v1 기능 파일은 v1.x에서 그대로 열린다.

- editor.view = [x0, y0, x1, y1]: 기본 보기(세계 좌표 상자). 에디터 「전체 보기」, 직접 조종, 실험실 관전·경주·영상의
  처음 카메라. 잘못된 값(길이가 4가 아님, 숫자가 아님, 유한하지 않음, x0 >= x1 또는 y0 >= y1)은 없는 것으로 본다.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import Any

from evolab.spec import model as m

VIEW_KEY = "view"
"""editor 사전에서 기본 보기의 키."""

ViewBox = tuple[float, float, float, float]


def _as_box(value: Any) -> ViewBox | None:
    """[x0, y0, x1, y1] 모양의 올바른 상자면 float 네 개, 아니면 None."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    out: list[float] = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            return None
        f = float(v)
        if not math.isfinite(f):
            return None
        out.append(f)
    x0, y0, x1, y1 = out
    if not (x0 < x1 and y0 < y1):
        return None
    return (x0, y0, x1, y1)


def editor_view_box(editor: Mapping[str, Any] | None) -> ViewBox | None:
    """editor 사전(에디터 문서의 data["editor"] 같은 것)에서 기본 보기. 없거나 잘못되었으면 None."""
    if not isinstance(editor, Mapping):
        return None
    return _as_box(editor.get(VIEW_KEY))


def view_box(env: m.EnvFile) -> ViewBox | None:
    """환경의 기본 보기 (x0, y0, x1, y1). 없거나 잘못되었으면 None."""
    return editor_view_box(env.editor)


def with_view_box(editor: Mapping[str, Any] | None, box: tuple[float, float, float, float] | None) -> dict[str, Any]:
    """editor 사전의 사본에 기본 보기를 넣거나(box) 지운(None) 새 사전. 다른 키는 그대로 둔다.

    box가 올바른 상자가 아니면 ValueError(호출하는 쪽의 실수: 화면의 보이는 영역은 늘 올바른 상자다).
    """
    out = dict(editor or {})
    if box is None:
        out.pop(VIEW_KEY, None)
        return out
    checked = _as_box(box)
    if checked is None:
        raise ValueError(f"invalid view box: {box!r}")
    out[VIEW_KEY] = list(checked)
    return out


__all__ = ["VIEW_KEY", "ViewBox", "editor_view_box", "view_box", "with_view_box"]

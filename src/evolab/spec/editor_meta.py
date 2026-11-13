"""환경 파일의 에디터 전용 칸(editor) 읽기·쓰기 (v2 설계 D25).

editor 구역은 해시(env_hash)와 런타임에서 빠지고 형식 번호와도 무관하다. v1.x도 editor를 자유 사전으로 읽으므로
기본 보기를 저장한 v1 기능 파일은 v1.x에서 그대로 열린다.

- editor.view = [x0, y0, x1, y1]: 기본 보기(세계 좌표 상자). 에디터 「전체 보기」, 직접 조종, 실험실 관전·경주·영상의
  처음 카메라. 잘못된 값(길이가 4가 아님, 숫자가 아님, 유한하지 않음, x0 >= x1 또는 y0 >= y1)은 없는 것으로 본다.
- (6.2.0) editor.train = {"qualify_entrants": 3, "qualify_trials": 9600, …}: 프리셋이 권하는 새 실행 설정. 읽는 곳은
  engine.trainer.TrainConfig.for_new_run 하나뿐이다(런타임·학습기·저장된 실행은 읽지 않는다: editor는 런타임 밖). 받는 키와
  값 검사는 그 함수의 PRESET_TRAIN_KEYS.
- (6.2.0) editor.rules = [{"kind": "fall", "event": "fall", "blocks": ["s_fall"]}, …]: 「규칙」 탭·안내 창이 만든 규칙의
  표지(presets/rules.py가 쓰고 읽는다).
- (6.2.0) editor.recipe = {"recipe": 1, …, "built": <만든 직후의 환경 해시>, "templates": {틀 id: sha256}}: 이 환경을 만든
  조리법(presets/recipe.py가 쓴다).
- (6.2.0) editor.ignored = ["<검사 코드>@<어디>", …]: 「문제 목록」에서 「괜찮습니다」를 누른 살펴볼 것(app/editor/fixes.py가 쓰고
  document.editor_checks가 읽어 뺀다). 해시·런타임 밖.

이 모듈의 읽기 함수는 모양이 틀린 값을 「없는 것」으로 돌려준다(예외를 던지지 않는다: 사용자가 손으로 고친 파일도 열려야 한다).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import Any

from evolab.spec import model as m

VIEW_KEY = "view"
"""editor 사전에서 기본 보기의 키."""
TRAIN_KEY = "train"
"""editor 사전에서 프리셋 권장 새 실행 설정의 키(6.2.0, 설계 D31)."""
RULES_KEY = "rules"
"""editor 사전에서 규칙 표지 목록의 키(6.2.0, 설계 D39)."""
RECIPE_KEY = "recipe"
"""editor 사전에서 조리법 표지의 키(6.2.0, 설계 D43)."""
IGNORED_KEY = "ignored"
"""editor 사전에서 「괜찮습니다」로 숨긴 살펴볼 것의 키(6.2.0, 묶음 ED2: app/editor/document.IGNORED_KEY와 같은 값)."""

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


def editor_train(editor: Mapping[str, Any] | None) -> dict[str, Any]:
    """editor.train(프리셋 권장 새 실행 설정)의 사본. 없거나 사전이 아니면 빈 사전. 키·값은 거르지 않는다(for_new_run이 거른다)."""
    if not isinstance(editor, Mapping):
        return {}
    value = editor.get(TRAIN_KEY)
    return {str(k): v for k, v in value.items()} if isinstance(value, Mapping) else {}


def editor_rules(editor: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """editor.rules(규칙 표지)의 사본: 사전인 줄만. 없거나 목록이 아니면 빈 목록."""
    if not isinstance(editor, Mapping):
        return []
    value = editor.get(RULES_KEY)
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def editor_recipe(editor: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """editor.recipe(조리법 표지)의 사본. 없거나 사전이 아니면 None."""
    if not isinstance(editor, Mapping):
        return None
    value = editor.get(RECIPE_KEY)
    return dict(value) if isinstance(value, Mapping) else None


__all__ = [
    "RECIPE_KEY", "RULES_KEY", "TRAIN_KEY", "VIEW_KEY", "ViewBox",
    "editor_recipe", "editor_rules", "editor_train", "editor_view_box", "view_box", "with_view_box",
]

"""위젯 연결표: 안내·도움말·풍선 도움말·빈 상태가 화면의 위젯을 찾는 단 하나의 표 (v3 설계 4.5, 7.3).

화면 파일(lab/view.py 등)을 고치지 않고 바깥에서 위젯을 찾는다. 키 → Anchor(화면, 속성 경로 | 단추 글 상수 | 대시보드 타일 id).
화면이 속성 이름이나 단추 글을 바꾸면 tests/test_help_anchors.py(T1)가 바로 알리고, 이 표만 고친다.

- `path`: 그 화면의 맨 위(탭 화면과 "main"은 주 창, 따로 뜨는 창은 그 창)에서 점으로 이은 속성 경로. 끝이 `()`인 마디는 인자 없이
  부른다(예: 주 창 탭 줄 모서리 `tabs.cornerWidget()`).
- `text`: strings.py 상수 이름. 그 화면 안에서 글이 이 상수와 같은 단추(QAbstractButton)를 찾는다(지역 변수로만 만든 단추).
- `tile`: 대시보드 타일 id(`Dashboard.tile(id)`).

이 모듈은 화면 클래스를 불러오지 않는다(주 창·직접 조종 창 등은 클래스 이름으로 알아본다). 그래서 아직 없는 v3 창(오류 보고, 내보내기)도
같은 표로 다루고, 불러오는 비용이 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from PySide6.QtWidgets import QAbstractButton, QApplication, QMainWindow, QWidget

from evolab.app import strings as S

TAB_SCREENS: Final = ("editor", "lab", "compare", "records")
"""주 창의 탭 화면. MainWindow에서 같은 이름의 속성이 그 탭 위젯이다(탭 순서와 같음)."""

WINDOW_SCREENS: Final[dict[str, str]] = {
    "PlayWindow": "play",
    "RaceDialog": "race",
    "RepeatDialog": "repeat",
    "ReportDialog": "report",
    "ExportDialog": "export",
}
"""따로 뜨는 창의 클래스 이름 → 화면 이름(하위 클래스도 같은 화면)."""

MAIN_WINDOW_CLASS: Final = "MainWindow"
SCREENS: Final = ("main", *TAB_SCREENS, *WINDOW_SCREENS.values())
"""화면 이름 전부. "main"은 주 창에서 탭 밖(메뉴 막대, 탭 줄 모서리, 상태 줄)."""

EDITOR_DRAWN_SECTIONS: Final = ("bodies", "generators", "regions", "checkpoints")
"""에디터 캔버스에 그려지는 부품 구역. 모두 비면 빈 문서(관절·구동기·센서는 물체에 붙으므로 물체 없이 있을 수 없다)."""


@dataclass(frozen=True)
class Anchor:
    """연결표 한 줄. path·text·tile 가운데 하나만 쓴다."""

    screen: str
    """화면 이름(SCREENS): main, editor, lab, compare, records, play, race, repeat, report, export."""
    path: str | None = None
    """그 화면 맨 위(주 창 또는 그 창)에서 속성 경로, 예 "lab.btn_start"."""
    text: str | None = None
    """strings 이름: 이 글을 가진 단추, 예 "PLAY_RESTART"."""
    tile: str | None = None
    """대시보드 타일 id."""


def _table() -> dict[str, Anchor]:
    lab = {
        # 환경과 학습 설정 칸
        "env": "env_combo",
        "method": "method",
        "seed": "seed",
        "popsize": "popsize",
        "sigma": "sigma",
        "ipop": "ipop",
        "trials": "trials",
        "workers": "workers",
        "ga_elites": "ga_elites",
        "ga_crossover": "ga_crossover",
        "ga_mut_rate": "ga_mut_rate",
        "ga_mut_sigma": "ga_mut_sigma",
        "ga_decay": "ga_decay",
        "sa_neighbors": "sa_neighbors",
        "sa_step": "sa_step",
        "sa_temp": "sa_temp",
        "sa_cooling": "sa_cooling",
        # 정지 조건, 학습 안정성
        "max_generations": "max_generations",
        "max_evaluations": "max_evaluations",
        "max_minutes": "max_minutes",
        "target": "target",
        "stagnation": "stagnation",
        "restart_on": "restart_on",
        "restart_patience": "restart_patience",
        "restart_from": "restart_from",
        "qualify_entrants": "qualify_entrants",
        "qualify_trials": "qualify_trials",
        # 진행, 보기
        "start": "btn_start",
        "pause": "btn_pause",
        "continue": "btn_continue",
        "step": "btn_step",
        "end": "btn_end",
        "spectate": "rb_spectate",
        "fast": "rb_fast",
        "speed": "speed_combo",
        "fast_interval": "fast_interval",
        # 캔버스 아래 타임라인 줄
        "versus": "btn_versus",
        "live": "btn_live",
        "video": "btn_video",
        "timeline": "timeline",
        "timeline_play": "btn_timeline",
        "flip": "btn_flip",
        "race": "btn_race",
        # 환경 묶음·학습 안정성 묶음 단추, 캔버스, 대시보드
        "reload": "btn_reload",
        "edit": "btn_edit",
        "apply": "btn_apply",
        "canvas": "canvas",
        "dashboard": "dashboard",
    }
    table: dict[str, Anchor] = {f"lab.{key}": Anchor("lab", path=f"lab.{attr}") for key, attr in lab.items()}
    # 대시보드 타일: 타일 제목 사전의 키가 곧 타일 id 전부다(Dashboard.ALL_TILE_IDS와 같음, T1이 확인)
    table.update({f"lab.tile.{tile_id}": Anchor("lab", tile=tile_id) for tile_id in S.DASH_TILE_TITLES})
    compare = {"repeat": "repeat_button", "refresh": "refresh_button", "clear": "clear_button", "png": "png_button",
               "csv": "csv_button", "race": "race_button"}
    table.update({f"compare.{key}": Anchor("compare", path=f"compare.{attr}") for key, attr in compare.items()})
    table["records.refresh"] = Anchor("records", path="records.refresh_button")
    table["records.memo"] = Anchor("records", path="records.memo_button")
    table["editor.builtin"] = Anchor("editor", text="ED_FROM_BUILTIN")
    table["play.restart"] = Anchor("play", text="PLAY_RESTART")
    table["play.new_seed"] = Anchor("play", text="PLAY_NEW_SEED")
    table["play.save"] = Anchor("play", path="save_button")
    table["play.load"] = Anchor("play", text="PLAY_LOAD_RECORD")
    # 주 창 탭 줄 오른쪽 모서리 「? 도움말」 단추(I 묶음 help/setup.py가 tabs.setCornerWidget으로 단다)
    table["main.help_corner"] = Anchor("main", path="tabs.cornerWidget()")
    return table


ANCHORS: Final[dict[str, Anchor]] = _table()
"""키 → Anchor. 키: HELP_TIPS의 키 전부 + "lab.canvas", "lab.dashboard", "lab.tile.<id>", "main.help_corner", "lab.timeline",
"lab.timeline_play", "lab.flip", "lab.race", "lab.reload", "lab.edit", "lab.apply"(설계 7.3)."""


def keys_for(screen: str) -> list[str]:
    """그 화면의 연결표 키(표 순서)."""
    return [key for key, anchor in ANCHORS.items() if anchor.screen == screen]


# ---------------------------------------------------------------------------
# 창과 화면 알아보기
# ---------------------------------------------------------------------------
def _class_names(widget: object) -> set[str]:
    return {cls.__name__ for cls in type(widget).__mro__}


def _window_screen(widget: QWidget) -> str | None:
    """따로 뜨는 창이면 그 화면 이름."""
    for cls in type(widget).__mro__:
        screen = WINDOW_SCREENS.get(cls.__name__)
        if screen is not None:
            return screen
    return None


def _is_main_window(widget: object) -> bool:
    return isinstance(widget, QMainWindow) and MAIN_WINDOW_CLASS in _class_names(widget)


def _parent(widget: QWidget) -> QWidget | None:
    """부모 위젯. 대시보드 타일은 지금 쪽이 아니면 부모가 없으므로(배치에서 빠짐) 자기 대시보드를 부모로 본다."""
    parent = widget.parentWidget()
    if parent is None and hasattr(widget, "tile_id"):
        dash = getattr(widget, "dash", None)
        if isinstance(dash, QWidget):
            return dash
    return parent


def _top_levels() -> list[QWidget]:
    """열린 창들(보이는 창 먼저)."""
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return []
    windows = list(app.topLevelWidgets())
    return [w for w in windows if w.isVisible()] + [w for w in windows if not w.isVisible()]


def _screen_root(root: QWidget, screen: str) -> QWidget | None:
    """연결표 경로의 출발점: 탭 화면·"main"이면 주 창, 따로 뜨는 창이면 그 창."""
    if screen in WINDOW_SCREENS.values():
        for w in (root, root.window()):
            if _window_screen(w) == screen:
                return w
        for w in _top_levels():  # 닫혔지만 아직 지워지지 않은 창은 고르지 않는다
            if _window_screen(w) == screen and w.isVisible():
                return w
        return None
    for w in (root, root.window()):
        if _is_main_window(w):
            return w
    for w in _top_levels():
        if _is_main_window(w):
            return w
    return None


def _follow(obj: object, path: str) -> object | None:
    for part in path.split("."):
        if obj is None:
            return None
        if part.endswith("()"):
            method = getattr(obj, part[:-2], None)
            obj = method() if callable(method) else None
        else:
            obj = getattr(obj, part, None)
    return obj


def _find_button(container: object, text: str | None) -> QWidget | None:
    if not isinstance(container, QWidget) or not text:
        return None
    found = [b for b in container.findChildren(QAbstractButton) if b.text() == text]
    visible = [b for b in found if b.isVisible()]
    return (visible or found or [None])[0]


def _tile(main: QWidget, tile_id: str) -> object | None:
    dashboard = _follow(main, "lab.dashboard")
    getter = getattr(dashboard, "tile", None)
    if not callable(getter):
        return None
    try:
        return getter(tile_id)
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# 공개 함수 (서명 고정: 설계 7.3)
# ---------------------------------------------------------------------------
def resolve(root: QWidget, key: str) -> QWidget | None:
    """연결표 키의 위젯. root는 주 창이나 따로 뜨는 창(또는 그 안의 아무 위젯). 없으면 None(그 창이 열려 있지 않음 포함)."""
    anchor = ANCHORS.get(key)
    if anchor is None or root is None:
        return None
    try:
        base = _screen_root(root, anchor.screen)
        if base is None:
            return None
        found: object | None = None
        if anchor.path is not None:
            found = _follow(base, anchor.path)
        elif anchor.text is not None:
            container = getattr(base, anchor.screen, None) if anchor.screen in TAB_SCREENS else base
            found = _find_button(container, getattr(S, anchor.text, None))
        elif anchor.tile is not None:
            found = _tile(base, anchor.tile)
    except RuntimeError:  # 이미 지워진 Qt 객체
        return None
    return found if isinstance(found, QWidget) else None


def screen_of(widget: QWidget | None) -> str | None:
    """위젯이 있는 화면: 따로 뜨는 창이면 그 창의 화면, 주 창이면 지나온 탭 화면(탭 밖이면 "main"), 우리 창 밖이면 None.

    주 창 탭 안에서 띄운 대화 상자(메시지 상자 등)는 부모를 따라 그 탭 화면으로 본다.
    """
    passed: list[QWidget] = []
    w = widget
    try:
        while w is not None:
            screen = _window_screen(w)
            if screen is not None:
                return screen
            if _is_main_window(w):
                for tab in TAB_SCREENS:
                    page = getattr(w, tab, None)
                    if isinstance(page, QWidget) and any(p is page for p in passed):
                        return tab
                return "main"
            passed.append(w)
            w = _parent(w)
    except RuntimeError:  # 이미 지워진 Qt 객체
        return None
    return None


def show_screen(window: QMainWindow, screen: str) -> bool:
    """그 화면이 보이게 주 창의 탭을 바꾼다. "main"은 할 일 없이 참, 따로 뜨는 창의 화면이나 모르는 화면이면 거짓."""
    if screen == "main":
        return True
    if screen not in TAB_SCREENS:
        return False
    page = getattr(window, screen, None)
    tabs = getattr(window, "tabs", None)
    if not isinstance(page, QWidget) or tabs is None:
        return False
    tabs.setCurrentWidget(page)
    return True


def editor_is_empty(window: QMainWindow) -> bool:
    """에디터 문서에 캔버스에 그려지는 부품(물체, 지형·트랙·언덕·장애물 생성기, 영역, 체크포인트)이 하나도 없는가."""
    doc = _follow(window, "editor.doc")
    items = getattr(doc, "items", None)
    if not callable(items):
        return False
    return not any(items(section) for section in EDITOR_DRAWN_SECTIONS)


__all__ = ["ANCHORS", "Anchor", "SCREENS", "TAB_SCREENS", "WINDOW_SCREENS", "editor_is_empty", "keys_for", "resolve",
           "screen_of", "show_screen"]

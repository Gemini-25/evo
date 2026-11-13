"""화면 다듬기의 통합 기준(polish/plan.md 5.1의 C2·C5·C8, 통합 묶음 I).

- C2 잘림: 1280×720과 1366×768 창에서, 글꼴 ×1.0과 ×1.10으로, 네 탭(에디터는 착륙선을 열고 관절 하나를 고른 채, 실험실은 내장 달리기를
  고른 채)·직접 조종(달리기)·도움말 창에서 보이는 단추·이름표·체크 상자·라디오 단추 가운데 가로로 잘린 것 0, 보이는 스크롤 영역의
  가로 스크롤 막대 0. 「가로로 잘림」 = sizeHint 폭 > 제 폭, 또는 (스크롤 영역 안이면 ensureWidgetVisible로 세로로 보이게 한 뒤)
  보이는 폭 < 제 폭. 줄바꿈 이름표는 뺀다(세로로 늘어남). 일부러 줄이는 글(경로·상태 글: 가로 크기 정책 Ignored)도 뺀다.
- C5 으뜸 행동: 상태마다 보이는 으뜸 단추는 정확히 하나(실험실 학습 전의 캔버스 카드 단추는 같은 행동이라 패널 단추와 한 쌍),
  켜진 것은 많아야 하나. 으뜸이 꺼지는 상태(에디터에 오류, 기록에서 고른 것 없음)에서는 켜진 것 0이고 까닭(tip_text)이 있다.
  높이는 같은 줄의 옆 단추와 같다.
- C8 풍선: 주 창 네 탭·안내 창·직접 조종 창의 위젯과 표·목록 칸에서 띄운 풍선 창의 폭 ≤ 440px(앱 전체 풍선 필터, 1.8).

px 값은 리눅스 offscreen 글꼴에서만 단언한다(윈도우 CI의 offscreen은 글꼴이 없어 글자가 모두 네모: 다듬기 계획 C3의 규칙). 윈도우에서는
잘림 대신 「보임」과 으뜸 수만 본다.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QAbstractListModel, QEvent, QPoint
from PySide6.QtGui import QFont, QHelpEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QApplication,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QWidget,
)

from evolab import paths
from evolab.app import strings as S
from evolab.app import style

LINUX = sys.platform.startswith("linux")
SIZES = [(1280, 720), (1366, 768)]
SCALES = [1.0, 1.10]
TIP_LIMIT = 440


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv("EVOLAB_NO_TOUR", "1")
    from evolab.app.editor.view import EditorView
    from evolab.app.preset.wizard import PresetWizard

    monkeypatch.setattr(PresetWizard, "ask_discard", lambda self: True)
    monkeypatch.setattr(EditorView, "maybe_save", lambda self: True)  # 고친 문서를 닫을 때 모달 물음이 뜨지 않게
    yield tmp_path
    QToolTip.hideText()


@pytest.fixture(params=SCALES, ids=lambda s: f"font{s:g}")
def scaled(request: pytest.FixtureRequest, qapp: QApplication) -> Iterator[float]:
    old = QFont(qapp.font())
    font = QFont(old)
    font.setPointSizeF(old.pointSizeF() * request.param)
    qapp.setFont(font)
    yield request.param
    qapp.setFont(old)


def main_window(qtbot: Any, size: tuple[int, int]) -> Any:
    from evolab.app.main import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(*size)
    window.show()
    QTest.qWaitForWindowExposed(window)
    QTest.qWait(50)
    return window


def _scroll_of(w: QWidget) -> QScrollArea | None:
    p = w.parentWidget()
    while p is not None:
        if isinstance(p, QScrollArea):
            return p
        p = p.parentWidget()
    return None


def _shrinks_on_purpose(w: QWidget) -> bool:
    """일부러 줄이는 글: 가로 크기 정책이 Ignored(실험실 상태·환경 글, Hb)이거나 elide 하는 이름표(비교의 폴더 경로, PC)."""
    if w.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored:
        return True
    return type(w).__name__ == "_ElidedLabel"


def clipped(root: QWidget) -> list[tuple[str, str, str, int, int, int]]:
    """root 아래 보이는 단추·이름표 가운데 가로로 잘린 것: (종류, objectName, 글, sizeHint 폭, 폭, 보이는 폭)."""
    bad = []
    for w in root.findChildren(QWidget):
        if not isinstance(w, (QAbstractButton, QLabel)) or not w.isVisible():
            continue
        if isinstance(w, QLabel) and (w.wordWrap() or not w.text()):
            continue
        if isinstance(w, QAbstractButton) and not w.text() and w.icon().isNull():
            continue
        if _shrinks_on_purpose(w):
            continue
        area = _scroll_of(w)
        if area is not None:
            area.ensureWidgetVisible(w, 0, 0)
            QApplication.processEvents()
        visible = w.visibleRegion().boundingRect().width()
        if w.sizeHint().width() > w.width() + 1 or visible < w.width() - 1:
            bad.append((type(w).__name__, w.objectName(), w.text()[:30], w.sizeHint().width(), w.width(), visible))
    return bad


def hscrolls(root: QWidget) -> list[str]:
    return [a.objectName() or type(a.widget()).__name__ for a in root.findChildren(QScrollArea)
            if a.isVisible() and a.horizontalScrollBar().isVisible()]


def _prepare_editor(window: Any) -> None:
    ed = window.editor
    ed.doc.load_text((paths.builtin_env_dir() / "lander.evoenv").read_text(encoding="utf-8"), None)
    joints = ed.doc.items("joints")
    if joints:
        ed.doc.select([("joints", joints[0]["id"])])
    QTest.qWait(50)


# ---------------------------------------------------------------------------
# C2 잘림
# ---------------------------------------------------------------------------
def test_walk_finds_a_clipped_button_and_a_hidden_scroll_bar(qtbot: Any) -> None:
    """걷기 자체의 대조군: 글보다 좁은 단추와 가로 스크롤 막대가 뜬 스크롤 영역은 잡힌다."""
    from PySide6.QtWidgets import QPushButton, QVBoxLayout

    root = QWidget()
    qtbot.addWidget(root)
    lay = QVBoxLayout(root)
    narrow = QPushButton("아주 긴 단추 이름이 여기에 들어갑니다")
    narrow.setFixedWidth(60)
    lay.addWidget(narrow)
    area = QScrollArea()
    wide = QLabel("가" * 200)
    area.setWidget(wide)
    area.setFixedSize(120, 80)
    lay.addWidget(area)
    root.show()
    QTest.qWaitForWindowExposed(root)
    assert [b[0] for b in clipped(root) if b[0] == "QPushButton"] == ["QPushButton"]
    assert hscrolls(root) == ["QLabel"]


@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_c2_no_horizontally_clipped_controls_in_the_main_window(qtbot: Any, isolated: Path, scaled: float,
                                                                 size: tuple[int, int]) -> None:
    window = main_window(qtbot, size)
    _prepare_editor(window)
    assert window.lab.load_env(paths.builtin_env_dir() / "running.evoenv")
    report = {}
    for tab in ("editor", "lab", "compare", "records"):
        page = getattr(window, tab)
        window.tabs.setCurrentWidget(page)
        QTest.qWait(80)
        report[tab] = (clipped(page), hscrolls(page))
    if LINUX:
        assert all(not bad and not bars for bad, bars in report.values()), report
    else:  # 글꼴 없는 offscreen: 세 위젯이 보이는지만
        assert window.editor.go_bar.isVisible() and window.lab.btn_start.isVisible()


@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_c2_play_and_help_windows(qtbot: Any, isolated: Path, scaled: float, size: tuple[int, int]) -> None:
    from evolab.app.help import window as HW
    from evolab.app.play.window import PlayWindow

    from evolab.spec.io import load_env

    play = PlayWindow(load_env(paths.builtin_env_dir() / "running.evoenv"))
    qtbot.addWidget(play)
    play.resize(*size)
    play.show()
    QTest.qWaitForWindowExposed(play)
    QTest.qWait(50)
    help_window = HW.open_help("examples")
    roots: list[QWidget] = [play]
    if isinstance(help_window, QWidget):
        help_window.resize(*size)
        QTest.qWait(50)
        roots.append(help_window)
    report = {type(r).__name__: (clipped(r), hscrolls(r)) for r in roots}
    if isinstance(help_window, QWidget):
        help_window.close()
    play.close()
    if LINUX:
        assert all(not bad and not bars for bad, bars in report.values()), report


# ---------------------------------------------------------------------------
# C5 으뜸 행동
# ---------------------------------------------------------------------------
def _primaries(root: QWidget) -> tuple[list[QAbstractButton], list[QAbstractButton]]:
    shown = [b for b in style.primaries(root) if b.isVisible()]
    return shown, [b for b in shown if b.isEnabled()]


def test_c5_one_primary_per_state(qtbot: Any, isolated: Path) -> None:
    window = main_window(qtbot, (1280, 720))
    lab = window.lab
    window.tabs.setCurrentWidget(lab)
    assert lab.load_env(paths.builtin_env_dir() / "running.evoenv")
    QTest.qWait(80)
    shown, on = _primaries(lab)
    # 학습 전: 패널의 「학습 시작」과 캔버스 카드의 같은 단추(한 쌍)
    assert lab.btn_start in shown and lab.btn_start in on
    others = [b for b in shown if b is not lab.btn_start]
    assert all(b.text() == lab.btn_start.text() for b in others) and len(others) <= 1, [b.text() for b in shown]
    row = [b for b in (lab.btn_pause,) if b.isVisible()]
    for b in row:
        assert b.height() == lab.btn_start.height()  # 옆 단추와 같은 높이

    ed = window.editor
    window.tabs.setCurrentWidget(ed)
    _prepare_editor(window)
    shown, on = _primaries(ed)
    assert len(shown) == 1 and len(on) == 1, [b.text() for b in shown]
    # 오류가 있으면 켜진 으뜸 0 + 까닭
    first = ed.doc.items("bodies")[0]["id"]
    ed.doc.set_value(("bodies", 0, "id"), first + " bad id!")
    qtbot.waitUntil(lambda: not _primaries(ed)[1], timeout=5000)
    shown, on = _primaries(ed)
    assert len(shown) == 1 and not on and style.why_off(shown[0]), [b.text() for b in shown]
    assert S.ED_WHY_ERRORS.format(n=1) in style.tip_text(shown[0])

    records = window.records
    window.tabs.setCurrentWidget(records)
    QTest.qWait(50)
    shown, on = _primaries(records)
    assert len(shown) == 1 and not on and style.why_off(shown[0])  # 고른 실행 없음


def test_c5_wizard_has_one_primary_on_every_step(qtbot: Any, isolated: Path) -> None:
    from evolab.app.preset.wizard import PresetWizard

    w = PresetWizard()
    qtbot.addWidget(w)
    w.show()
    QTest.qWaitForWindowExposed(w)
    for step in range(5):
        w.go(step)
        QTest.qWait(30)
        shown, on = _primaries(w)
        assert len(shown) == 1 and len(on) <= 1, (step, [b.text() for b in shown])
    w.close()


def test_c5_error_box_has_one_primary(qtbot: Any, isolated: Path) -> None:
    from evolab.app import errors

    box = errors.error_box(None, S.ED_OPEN_FAILED, errors.explain(FileNotFoundError(2, "No such file", "x.evoenv")))
    qtbot.addWidget(box)
    box.show()
    QTest.qWait(30)
    shown, on = _primaries(box)
    assert len(shown) == 1 and len(on) == 1 and shown[0] is box.defaultButton()
    box.close()


# ---------------------------------------------------------------------------
# C8 풍선 폭
# ---------------------------------------------------------------------------
def _tip_label() -> QWidget | None:
    labels = [w for w in QApplication.topLevelWidgets() if w.inherits("QTipLabel") and w.isVisible()]
    return labels[0] if labels else None


def _show_tip(target: QWidget, pos: QPoint) -> int | None:
    QToolTip.hideText()
    QApplication.sendEvent(target, QHelpEvent(QEvent.Type.ToolTip, pos, target.mapToGlobal(pos)))
    label = _tip_label()
    if label is None:
        return None
    WIDEST.append(label.width())
    return label.width()


WIDEST: list[int] = []
"""띄운 풍선 폭(테스트가 긴 풍선도 실제로 띄웠는지 본다)."""


def tip_widths(root: QWidget) -> tuple[int, list[tuple[int, str, str]]]:
    """root 아래 보이는 위젯과 표·목록·나무의 보이는 칸에 풍선을 띄워 (띄운 수, 너무 넓은 것)."""
    shown, wide = 0, []
    for w in root.findChildren(QWidget):
        if not w.isVisible():
            continue
        if isinstance(w, QAbstractItemView):
            model = w.model()
            if model is None:
                continue
            root_index = w.rootIndex()
            for r in range(min(model.rowCount(root_index), 30)):
                columns = 1 if isinstance(model, QAbstractListModel) else model.columnCount(root_index)
                for c in range(columns):
                    index = model.index(r, c, root_index)
                    tip = model.data(index, 3)  # Qt.ToolTipRole
                    rect = w.visualRect(index)
                    if not tip or rect.isEmpty():
                        continue
                    width = _show_tip(w.viewport(), rect.center())
                    shown += width is not None
                    if width is not None and width > TIP_LIMIT:
                        wide.append((width, w.objectName(), str(tip)[:40]))
            continue
        if not style.tip_text(w):
            continue
        width = _show_tip(w, QPoint(min(4, w.width() // 2), min(4, w.height() // 2)))
        shown += width is not None
        if width is not None and width > TIP_LIMIT:
            wide.append((width, w.objectName(), style.tip_text(w)[:40]))
    QToolTip.hideText()
    return shown, wide


def test_c8_tooltips_are_at_most_440_px_wide(qtbot: Any, isolated: Path) -> None:
    from evolab.app.preset.wizard import open_wizard

    window = main_window(qtbot, (1280, 720))
    _prepare_editor(window)
    total = 0
    wide: dict[str, list[Any]] = {}
    for tab in ("editor", "lab", "compare", "records"):
        page = getattr(window, tab)
        window.tabs.setCurrentWidget(page)
        QTest.qWait(50)
        n, bad = tip_widths(page)
        total += n
        wide[tab] = bad
    wizard = open_wizard(window)
    QTest.qWaitForWindowExposed(wizard)
    for step in range(5):
        wizard.go(step)
        QTest.qWait(20)
        n, bad = tip_widths(wizard)
        total += n
        wide[f"preset{step}"] = bad
    wizard.close()
    assert total > 200  # 풍선이 실제로 떴다(필터가 받은 것과 Qt가 띄운 것 모두)
    if LINUX:
        assert max(WIDEST) > 300  # 긴 풍선(줄바꿈된 것)도 띄워 보았다
    if LINUX:
        assert not any(wide.values()), {k: v for k, v in wide.items() if v}

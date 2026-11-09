"""v3.0 B 묶음 풍선 도움말 감사 (v3 설계 4.6, D24, 8.1 T15, 9.2 M13).

조작 위젯 = 보이는 QAbstractButton(탭 줄 단추 제외), QComboBox, QAbstractSpinBox, QSlider, QLineEdit(콤보·숫자 칸 속 글 칸 제외).
  풍선 : toolTip()이 비어 있지 않음
  이름표: 글이 있는 단추이거나, QFormLayout에서 짝 이름표가 있음(칸을 감싼 위젯의 이름표 포함)
  맨 위젯: 둘 다 없음
기준: 실험실·비교·기록·직접 조종에서 맨 위젯 0, 실험실 설정 칸(숫자·고르기·체크·라디오) 100% 풍선, `apply_tooltips`는 이미 있는
풍선을 덮지 않음. (onboard 설계의 tools/help_audit.py를 테스트로 옮김)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QSlider,
    QTabBar,
    QWidget,
)

from evolab import paths
from evolab.app import strings as S
from evolab.app.help import anchors
from evolab.app.help import tooltips as T

LANDER = paths.builtin_env_dir() / "lander.evoenv"
CONTROLS = (QAbstractButton, QComboBox, QAbstractSpinBox, QSlider, QLineEdit)
SETTING_FIELDS = (QAbstractSpinBox, QComboBox, QAbstractButton)
"""실험실 설정 칸: 숫자 칸, 고르기, 체크·라디오(패널의 단추 중 글 단추는 아래에서 뺀다)."""


@dataclass
class Audit:
    screen: str
    controls: int = 0
    with_tip: int = 0
    bare: list[str] = field(default_factory=list)

    @property
    def tip_pct(self) -> float:
        return 100.0 * self.with_tip / self.controls if self.controls else 100.0


def _has_form_label(widget: QWidget, root: QWidget) -> bool:
    """widget(또는 widget을 감싼 위젯)이 폼 줄의 칸이고 이름표 글이 있는가."""
    w: QWidget | None = widget
    while w is not None and w is not root:
        parent = w.parentWidget()
        layout = QWidget.layout(parent) if parent is not None else None  # 속성 layout을 가린 위젯도 있다
        if isinstance(layout, QFormLayout):
            label = layout.labelForField(w)
            if label is not None and getattr(label, "text", lambda: "")():
                return True
        w = parent
    return False


def audit(root: QWidget, screen: str) -> Audit:
    out = Audit(screen)
    for w in root.findChildren(QWidget):
        if not isinstance(w, CONTROLS) or isinstance(w, QTabBar) or isinstance(w.parentWidget(), QTabBar):
            continue
        if not w.isVisibleTo(root):
            continue
        if isinstance(w, QLineEdit) and isinstance(w.parentWidget(), (QComboBox, QAbstractSpinBox)):
            continue
        out.controls += 1
        tip = bool(w.toolTip())
        text = w.text() if isinstance(w, QAbstractButton) else ""
        out.with_tip += tip
        if not tip and not text and not _has_form_label(w, root):
            out.bare.append(f"{type(w).__name__} {w.objectName()} in {type(w.parentWidget()).__name__}")
    return out


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv("EVOLAB_NO_TOUR", "1")
    yield tmp_path
    app = QApplication.instance()
    if isinstance(app, QApplication):  # 직접 조종 창 풍선 필터를 다음 테스트(다른 파일)에 남기지 않는다
        T.uninstall(app)


@pytest.fixture()
def main_window(qtbot: Any, isolated: Path) -> Any:
    from evolab.app.main import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1280, 800)
    window.show()
    return window


def _strip_help_tips(window: Any) -> None:
    """HELP_TIPS에서 온 풍선을 떼어 붙이기 전 상태로(통합 단계가 주 창을 만들 때 이미 붙여도 같은 시험이 되게).

    v1·v2가 화면 파일에서 단 풍선(HELP_TIPS와 글이 다름)은 그대로 둔다.
    """
    for key, tip in S.HELP_TIPS.items():
        widget = anchors.resolve(window, key)
        if widget is None:
            continue
        for w in (widget, T.form_label(widget)):
            if w is not None and w.toolTip() == tip:
                w.setToolTip("")


def _lab_panel(lab: Any) -> QWidget:
    """실험실 오른쪽 패널(설정 묶음들이 든 위젯)."""
    group = lab.seed.parentWidget()
    assert isinstance(group, QGroupBox)
    panel = group.parentWidget()
    assert panel is not None and lab.btn_start.parentWidget().parentWidget() is panel
    return panel


def _setting_fields(lab: Any) -> list[QWidget]:
    """실험실 패널의 설정 칸(숨은 방법별 칸 포함): 숫자 칸, 고르기, 체크·라디오. 글 단추(시작·적용 등)는 빼고."""
    out: list[QWidget] = []
    for w in _lab_panel(lab).findChildren(QWidget):
        if isinstance(w, QAbstractButton) and not w.isCheckable():
            continue
        if isinstance(w, QLineEdit) or not isinstance(w, SETTING_FIELDS):
            continue
        out.append(w)
    return out


def test_audit_after_tooltips(main_window: Any, qtbot: Any) -> None:
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    w = main_window
    _strip_help_tips(w)
    before = {s: audit(getattr(w, s), s) for s in ("lab", "compare", "records")}
    keys = [k for s in T.MAIN_SCREENS for k in anchors.keys_for(s) if k in S.HELP_TIPS]
    empty = [k for k in keys if not anchors.resolve(w, k).toolTip()]  # type: ignore[union-attr]
    added = T.install(app, w)
    assert added == len(empty) and len(empty) >= 40  # 풍선이 비어 있던 연결표 위젯 모두(이미 있던 것은 그대로)
    results = []
    for screen in ("lab", "compare", "records"):
        w.tabs.setCurrentWidget(getattr(w, screen))
        qtbot.wait(10)
        results.append(audit(getattr(w, screen), screen))
    play = w.open_play(LANDER)
    assert play is not None
    qtbot.waitExposed(play)
    flt = T.window_tips()
    assert flt is not None and ("play", 4) in flt.applied  # 직접 조종 창이 처음 보일 때 붙임
    results.append(audit(play, "play"))
    play.close()
    for r in results:
        assert r.bare == [], (r.screen, r.bare)
    by = {r.screen: r for r in results}
    assert by["play"].with_tip == by["play"].controls == 4
    assert by["lab"].with_tip > before["lab"].with_tip
    assert before["lab"].bare  # 붙이기 전에는 맨 위젯이 있었다(환경 고르기 칸)


def test_lab_setting_fields_all_have_tips(main_window: Any) -> None:
    lab = main_window.lab
    _strip_help_tips(main_window)
    fields = _setting_fields(lab)
    assert len(fields) >= 30  # 환경, 방법, 시드 … 보기 묶음까지(숨은 GA·SA 칸 포함)
    missing_before = [f for f in fields if not f.toolTip()]
    assert missing_before  # 붙이기 전(v2.1)에는 풍선이 없는 칸이 많다
    T.apply_tooltips(main_window, "lab")
    missing = [f"{type(f).__name__}" for f in fields if not f.toolTip()]
    assert missing == []
    for method in ("ga", "sa", "cmaes"):  # 방법을 바꿔 보이는 칸이 달라져도
        lab.method.setCurrentIndex(lab.method.findData(method))
        assert [f for f in _setting_fields(lab) if f.isVisibleTo(lab) and not f.toolTip()] == []


def test_apply_keeps_existing_tips_and_labels_get_the_text(main_window: Any) -> None:
    lab = main_window.lab
    _strip_help_tips(main_window)
    assert lab.restart_on.toolTip() == S.LAB_RESTART_TIP  # v1.1이 단 풍선
    lab.seed.setToolTip("custom")
    first = T.apply_tooltips(main_window, "lab")
    assert lab.restart_on.toolTip() == S.LAB_RESTART_TIP and lab.seed.toolTip() == "custom"
    assert lab.popsize.toolTip() == S.HELP_TIPS["lab.popsize"]
    label = T.form_label(lab.popsize)
    assert label is not None and label.toolTip() == S.HELP_TIPS["lab.popsize"]  # 이름표에도 같은 글
    assert T.form_label(lab.seed).toolTip() == "custom"  # type: ignore[union-attr]
    assert first > 0 and T.apply_tooltips(main_window, "lab") == 0  # 두 번째는 붙일 것이 없음
    keys = [k for k in anchors.keys_for("lab") if k in S.HELP_TIPS]
    assert all(anchors.resolve(main_window, k).toolTip() for k in keys)  # type: ignore[union-attr]


def test_window_tips_apply_once_per_window(main_window: Any, qtbot: Any) -> None:
    from evolab.app.play.window import PlayWindow
    from evolab.spec.io import load_env

    app = QApplication.instance()
    assert isinstance(app, QApplication)
    T.install(app, main_window)
    flt = T.window_tips()
    assert flt is not None
    play = PlayWindow(load_env(LANDER))
    qtbot.addWidget(play)
    n = len(flt.applied)
    play.show()
    play.hide()
    play.show()
    assert flt.applied[n:] == [("play", 4)]
    restart = anchors.resolve(play, "play.restart")
    assert restart is not None and restart.toolTip() == S.HELP_TIPS["play.restart"]
    other = QWidget()  # 우리 창이 아니면 붙이지 않음
    qtbot.addWidget(other)
    other.show()
    assert flt.applied[n:] == [("play", 4)]


def test_audit_finds_bare_widgets(qtbot: Any) -> None:
    """감사 자체 확인: 이름표도 풍선도 없는 고르기 칸을 맨 위젯으로 센다."""
    from PySide6.QtWidgets import QPushButton, QVBoxLayout

    root = QWidget()
    qtbot.addWidget(root)
    lay = QVBoxLayout(root)
    lay.addWidget(QComboBox())
    lay.addWidget(QPushButton("ok"))
    form_host = QWidget()
    form = QFormLayout(form_host)
    form.addRow("label", QComboBox())
    lay.addWidget(form_host)
    root.show()
    r = audit(root, "x")
    assert r.controls == 3 and len(r.bare) == 1 and r.with_tip == 0

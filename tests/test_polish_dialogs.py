"""6.2.0 다듬기 PD: 오류 상자(U35)와 오류 보고 묶음 창의 단추 위계(다듬기 1.5·1.8).

- 오류 상자(show_error·잡히지 않은 예외 상자): 폭이 BOX_MIN_WIDTH 이상, 「확인」만 기본 단추이자 으뜸(높이·폭은 꾸미기 전과 같음),
  묶음 단추는 자세히 단추 바로 옆, 본문과 「다음에 할 일」의 각 문단이 두 줄 안(리눅스 글꼴에서만 단언: 윈도우 CI의 offscreen은
  글꼴이 없어 줄바꿈이 사용자 화면과 다르다).
- 오류 보고 묶음 창: 「저장…」이 으뜸이자 기본 단추, 켜진 으뜸은 하나, 넣을 것이 없으면 꺼지고 풍선에 까닭(toolTip()은 그대로),
  「폴더 열기」는 조용한 단추.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QPushButton, QSpacerItem, QWidget

from evolab import diagnostics, paths
from evolab.app import errors, style
from evolab.app import strings as S
from evolab.app.report.bundle import BundleOptions

LINUX = sys.platform.startswith("linux")


@pytest.fixture(autouse=True)
def _private_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """오류 기록과 보고 묶음이 실제 사용자 폴더를 건드리지 않게."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    diagnostics.uninstall()
    yield
    for box in errors.open_error_boxes():
        box.close()
    for w in QApplication.topLevelWidgets():
        if isinstance(w, QMessageBox) and w.isVisible():
            w.close()
    diagnostics.uninstall()


def raised(exc: BaseException) -> BaseException:
    try:
        raise exc
    except BaseException as caught:  # noqa: BLE001
        return caught


def cases() -> list[tuple[str, Any]]:
    """다듬기 감사 사진(light_errors.png)의 오류 상자들."""
    return [
        ("notfound", raised(FileNotFoundError(2, "No such file or directory", "/nonexistent/folder/file.evoenv"))),
        ("permission", raised(PermissionError(13, "Permission denied", "C:/Users/me/Documents/x"))),
        ("unexpected", raised(KeyError("missing_key"))),
        ("worker", "워커 프로세스가 응답하지 않습니다."),
    ]


def wrapped_lines(label: QLabel | None) -> list[int]:
    """라벨 글의 문단(\\n으로 나뉜 줄)마다 지금 폭에서 꺾여 차지하는 줄 수."""
    if label is None or not label.isVisible() or not label.text():
        return []
    fm = label.fontMetrics()
    width = label.contentsRect().width()
    flags = int(Qt.TextFlag.TextWordWrap)
    return [round(fm.boundingRect(0, 0, width, 10**5, flags, para).height() / fm.lineSpacing()) if para else 1
            for para in label.text().split("\n")]


def labels(box: QMessageBox) -> dict[str, QLabel]:
    return {lbl.objectName(): lbl for lbl in box.findChildren(QLabel)}


def details_button(box: QMessageBox) -> QPushButton | None:
    found = [b for b in box.buttons() if box.buttonRole(b) == QMessageBox.ButtonRole.ActionRole
             and b is not getattr(box, "report_button", None) and b.text() != S.ERR_REPORT_BUTTON]
    return found[0] if found else None  # type: ignore[return-value]


def native_ok_size(parent: QWidget) -> tuple[int, int]:
    """꾸미지 않은 메시지 상자의 표준 「확인」 단추 크기(같은 스타일·글꼴)."""
    plain = QMessageBox(QMessageBox.Icon.Warning, "t", "x", QMessageBox.StandardButton.Ok, parent)
    plain.show()
    QApplication.processEvents()
    ok = plain.button(QMessageBox.StandardButton.Ok)
    size = ok.sizeHint().width(), ok.sizeHint().height()
    plain.close()
    plain.deleteLater()
    return size


def check_box(box: QMessageBox, parent: QWidget) -> None:
    QApplication.processEvents()
    ok = box.button(QMessageBox.StandardButton.Ok)
    assert ok is not None
    # 폭: 격자 맨 아래 빈 칸
    assert box.width() >= errors.BOX_MIN_WIDTH
    spacers = [box.layout().itemAt(i) for i in range(box.layout().count())]
    assert sum(1 for s in spacers if isinstance(s, QSpacerItem) and s.sizeHint().width() == errors.BOX_MIN_WIDTH) == 1
    # 「확인」만 기본·으뜸, 다른 단추는 기본이 아님(autoDefault는 그대로: 초점 단추는 Enter로 누를 수 있다)
    assert box.defaultButton() is ok and ok.isDefault()
    assert style.primaries(box) == [ok] and ok.isEnabled()
    others = [b for b in box.buttons() if b is not ok]
    assert others and not any(isinstance(b, QPushButton) and b.isDefault() for b in others)
    assert all(b.autoDefault() for b in others if isinstance(b, QPushButton))
    assert box.escapeButton() in (ok, None)  # None: Qt가 띄울 때 하나뿐인 AcceptRole(「확인」)을 고른다
    assert QApplication.focusWidget() in (ok, None)  # offscreen에서 창이 활성이 아니면 초점이 없을 수 있다
    # 으뜸 단추의 크기는 꾸미기 전 그대로(줄이 어긋나지 않게, 「확인」이 좁아지지 않게)
    width, height = native_ok_size(parent)
    assert ok.height() == height and ok.width() >= width
    # 묶음 단추는 자세히 단추 바로 옆(자세히가 없으면 「확인」 옆), 「확인」이 맨 앞
    row = sorted((b for b in box.buttons() if b.isVisible()), key=lambda b: b.geometry().x())
    report = box.report_button if hasattr(box, "report_button") else next(b for b in row if b.text() == S.ERR_REPORT_BUTTON)
    details = details_button(box)
    expected = [ok, details, report] if details is not None else [ok, report]
    assert row == expected
    # 본문과 「다음에 할 일」의 문단마다 두 줄 안(리눅스 글꼴: 감사의 3~4줄)
    found = labels(box)
    lines = wrapped_lines(found.get("qt_msgbox_label")) + wrapped_lines(found.get("qt_msgbox_informativelabel"))
    assert lines
    if LINUX:
        assert max(lines) <= 2, lines


@pytest.mark.parametrize("key", [k for k, _ in cases()])
def test_error_box_is_wide_with_one_primary_ok(qtbot: Any, key: str) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.resize(1280, 720)
    parent.show()
    error = dict(cases())[key]
    box = errors.show_error(parent, S.LAB_ERROR_TITLE, error)
    qtbot.waitUntil(box.isVisible, timeout=3000)
    check_box(box, parent)
    # 보통 글 형식과 덧붙임 글의 꼴은 그대로(tests/test_errors.py가 지키는 것)
    assert box.textFormat() == Qt.TextFormat.PlainText
    assert box.informativeText().startswith(f"{S.ERR_NEXT_LABEL}: ")


def test_details_toggle_keeps_width_and_second_tidy_adds_nothing(qtbot: Any) -> None:
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.show()
    box = errors.show_error(parent, S.LAB_ERROR_TITLE, raised(KeyError("k")))
    qtbot.waitUntil(box.isVisible, timeout=3000)
    before = box.width()
    details = details_button(box)
    assert details is not None
    details.click()
    QApplication.processEvents()
    assert box.width() >= max(errors.BOX_MIN_WIDTH, before - 1)
    ok = box.button(QMessageBox.StandardButton.Ok)
    width = ok.minimumWidth()
    errors.tidy_box(box)
    count = sum(1 for i in range(box.layout().count()) if isinstance(box.layout().itemAt(i), QSpacerItem)
                and box.layout().itemAt(i).sizeHint().width() == errors.BOX_MIN_WIDTH)
    assert count == 1 and ok.minimumWidth() == width and style.primaries(box) == [ok]


def test_report_button_still_opens_bundle_dialog(qtbot: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """으뜸·폭을 바꿔도 묶음 단추가 하던 일은 그대로(상자를 닫고 다음 차례에 묶음 대화 상자)."""
    opened: list[Any] = []
    monkeypatch.setattr(errors, "open_report", lambda parent: opened.append(parent))
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.show()
    box = errors.show_error(parent, S.LAB_ERROR_TITLE, "글")
    qtbot.waitUntil(box.isVisible, timeout=3000)
    box.report_button.click()
    qtbot.waitUntil(lambda: opened == [parent], timeout=3000)


def test_crash_box_matches_error_box(qtbot: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.report import dialog

    opened: list[Any] = []
    monkeypatch.setattr(dialog, "open_report_dialog", lambda parent, opts: opened.append(opts))
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.resize(1280, 720)
    parent.show()
    dialog.show_crash_box(parent, "ValueError: example at view.py:123", BundleOptions(reason="crash"))
    (box,) = [w for w in QApplication.topLevelWidgets() if isinstance(w, QMessageBox) and w.isVisible()]
    assert box.windowModality() == Qt.WindowModality.NonModal and box.text() == S.DIAG_CRASH_TEXT
    check_box(box, parent)
    report = next(b for b in box.buttons() if b.text() == S.ERR_REPORT_BUTTON)
    report.click()
    qtbot.waitUntil(lambda: len(opened) == 1, timeout=3000)
    assert opened[0].reason == "crash"


def test_report_dialog_save_is_the_one_primary(qtbot: Any) -> None:
    from evolab.app.report import dialog

    dlg = dialog.ReportDialog(None, BundleOptions())
    qtbot.addWidget(dlg)
    dlg.show()
    QApplication.processEvents()
    assert dlg.btn_save.isDefault() and dlg.btn_save.property(style.ROLE_PROPERTY) == "primary"
    assert [b for b in style.primaries(dlg) if b.isEnabled()] == [dlg.btn_save]
    assert dlg.btn_open_folder.property(style.ROLE_PROPERTY) == "quiet"
    assert dlg.btn_save.height() == dlg.btn_close.height()  # 같은 줄의 「닫기」와 높이가 같다
    tip = dlg.btn_save.toolTip()
    # 넣을 것을 모두 빼면 꺼지고, 풍선에 까닭(toolTip()은 그대로)
    for box in dlg.checks.values():
        box.setChecked(False)
    assert not dlg.btn_save.isEnabled() and dlg.btn_save.toolTip() == tip
    assert style.why_off(dlg.btn_save) == S.REP_SAVE_WHY_OFF and S.REP_SAVE_WHY_OFF in style.tip_text(dlg.btn_save)
    assert not [b for b in style.primaries(dlg) if b.isEnabled()]
    dlg.checks["report"].setChecked(True)
    assert dlg.btn_save.isEnabled() and style.tip_text(dlg.btn_save) == tip


def test_report_dialog_save_comes_back_after_writing(qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evolab.app.report import bundle, dialog

    monkeypatch.setattr(bundle, "reveal", lambda p: None)
    dlg = dialog.ReportDialog(None, BundleOptions())
    qtbot.addWidget(dlg)
    target = tmp_path / "x.zip"
    assert dlg.save_to(target) == target
    assert dlg.btn_save.isEnabled() and style.why_off(dlg.btn_save) == ""

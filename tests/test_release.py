"""M7: 첫 실행 예제 복사, 아이콘, 정보 창, 라이선스 고지."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from evolab import paths


def test_install_examples_copies_once_and_keeps_user_edits(tmp_path: Path) -> None:
    copied = paths.install_examples(tmp_path)
    names = sorted(p.name for p in copied)
    assert {"lander.evoenv", "track_car.evoenv", "hill_car.evoenv"} <= set(names)
    (tmp_path / "lander.evoenv").write_text("{}", encoding="utf-8")  # 사용자가 고친 것처럼
    assert paths.install_examples(tmp_path) == []
    assert (tmp_path / "lander.evoenv").read_text(encoding="utf-8") == "{}"


def test_icon_and_about(qtbot: Any) -> None:
    from evolab.app.main import MainWindow, create_application

    create_application()
    assert paths.icon_path().is_file()
    w = MainWindow()
    qtbot.addWidget(w)
    assert not w.windowIcon().isNull()
    text = w.about_text()
    assert "진화 실험실" in text


def test_collect_licenses(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))
    try:
        from collect_licenses import collect
    finally:
        sys.path.pop(0)
    out = collect(tmp_path / "licenses")
    notices = (out / "THIRD_PARTY_NOTICES.txt").read_text(encoding="utf-8")
    for name in ("numpy", "pymunk", "cma", "pydantic", "PySide6", "pyqtgraph", "pillow", "psutil", "Python"):
        assert name.lower() in notices.lower()
    pyside = [p for p in out.iterdir() if p.name.lower().startswith("pyside6")]
    assert pyside and (pyside[0] / "LGPL-3.0.txt").is_file()


def test_custom_data_dir(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(paths, "user_state_dir", lambda: tmp_path / "state")
    assert paths.custom_data_dir() is None
    target = tmp_path / "내 데이터 폴더"
    paths.set_custom_data_dir(target)
    assert paths.data_dir() == target and paths.runs_dir() == target / "runs"
    paths.set_custom_data_dir(None)
    assert paths.custom_data_dir() is None

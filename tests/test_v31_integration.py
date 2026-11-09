"""v3.1 통합 검사: 묶음 사이 연결(배포본에 폰 재생기 틀, v3.1 문구가 모두 쓰임).

- PyInstaller 설정(packaging/evolab.spec)이 패키지 안의 자료 파일(.py가 아닌 것)을 모두 담고, 폰 재생 파일 틀은 코드가 찾는
  자리(phone.TEMPLATE_PATH = 모듈 옆 assets/, 배포본에서는 _MEIPASS/evolab/app/export/assets/)에 둔다. 빠지면 exe의 자체 점검
  「내보내기」(W2)·exe 내보내기(W7)·헤드리스 Edge(W9)가 윈도우 CI에서야 FileNotFoundError로 깨진다.
- strings.py의 v3.1 블록(0' 묶음과 뒤 묶음이 덧붙인 블록)의 상수는 모두 src/ 또는 packaging/ 어딘가에서 쓰인다.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SPEC = ROOT / "packaging" / "evolab.spec"
V31_HEADER = "# v3.1 나누기와 설치"


def _spec_datas() -> list[tuple[Path, str]]:
    """evolab.spec의 Analysis(datas=[...])에서 (원본 폴더, 배포본 안 자리). 원본은 ROOT / "a" / "b" 꼴의 식만 읽는다."""
    tree = ast.parse(SPEC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Analysis":
            datas = next(k.value for k in node.keywords if k.arg == "datas")
            break
    else:
        raise AssertionError("no Analysis(...) call in evolab.spec")
    out: list[tuple[Path, str]] = []
    for elt in datas.elts:  # type: ignore[attr-defined]
        src, dest = elt.elts
        parts: list[str] = []
        expr = src.args[0] if isinstance(src, ast.Call) else src  # str(ROOT / ...)
        while isinstance(expr, ast.BinOp):
            if isinstance(expr.right, ast.Constant):
                parts.insert(0, expr.right.value)
            expr = expr.left
        if isinstance(expr, ast.Name) and expr.id == "ROOT":
            out.append((ROOT.joinpath(*parts), dest.value))
    return out


def test_spec_bundles_every_package_data_file() -> None:
    datas = _spec_datas()
    data_files = [p for p in (SRC / "evolab").rglob("*")
                  if p.is_file() and p.suffix not in (".py", ".pyc") and "__pycache__" not in p.parts]
    assert data_files, "expected at least the icon and the phone player template"
    for path in data_files:
        assert any(src in path.parents for src, _dest in datas), f"{path.relative_to(ROOT)} is not in evolab.spec datas"


def test_spec_puts_the_phone_player_where_the_code_looks() -> None:
    from evolab.app.export import phone

    folder = phone.TEMPLATE_PATH.parent
    assert phone.TEMPLATE_PATH.is_file()
    dests = {src: dest for src, dest in _spec_datas()}
    assert folder in dests, "the phone player folder must be its own datas entry"
    # 배포본에서 모듈 __file__은 _MEIPASS/evolab/app/export/phone.pyc: 틀 폴더도 그 옆이어야 한다
    assert dests[folder] == folder.relative_to(SRC).as_posix()


def _v31_constant_names() -> list[str]:
    text = (SRC / "evolab" / "app" / "strings.py").read_text(encoding="utf-8")
    start = text.index(V31_HEADER)
    return re.findall(r"^([A-Z][A-Z0-9_]+): Final", text[start:], flags=re.MULTILINE)


def test_every_v31_string_is_used() -> None:
    names = _v31_constant_names()
    assert len(names) >= 60  # 0' 묶음 62개 + F 묶음 6개(이 검사가 블록 머리를 놓치면 0이 된다)
    sources = [p for p in [*SRC.rglob("*.py"), *(ROOT / "packaging").rglob("*.py")]
               if p.name != "strings.py" and "__pycache__" not in p.parts]
    text = "\n".join(p.read_text(encoding="utf-8") for p in sources)
    unused = [n for n in names if not re.search(rf"\b{n}\b", text)]
    assert unused == []


# ---------------------------------------------------------------------------
# 내보내기 대화 상자: 글 줄이 늘어도 위 칸이 겹쳐 그려지지 않는다(v3.1 통합 오프스크린 그림에서 찾음)
# ---------------------------------------------------------------------------
def _overlapping_rows(dlg: object) -> list[str]:
    """보이는 형식 단추·크기 줄·안내 줄이 서로 겹치거나 바라는 높이보다 낮게 눌렸으면 그 이름."""
    widgets = [*dlg.format_buttons.values(), dlg.size_combo, dlg.length_label, dlg.encoder_label,  # type: ignore[attr-defined]
               dlg.status]  # type: ignore[attr-defined]
    shown = [w for w in widgets if w.isVisible()]
    bad = []
    for w in shown:
        want = w.heightForWidth(w.width()) if w.hasHeightForWidth() else w.minimumSizeHint().height()
        if w.height() < want:
            bad.append(f"{type(w).__name__} {w.objectName() or w.text()[:12] if hasattr(w, 'text') else ''}: "
                       f"{w.height()} < {want}")
    for a, b in zip(shown, shown[1:], strict=False):
        ga, gb = a.geometry(), b.geometry()
        if a.parentWidget() is b.parentWidget() and ga.intersects(gb):
            bad.append(f"overlap {ga.getRect()} {gb.getRect()}")
    return bad


def test_export_dialog_grows_when_status_and_encoder_lines_appear(qtbot, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import dataclasses

    import numpy as np
    from PySide6.QtCore import QObject, QSettings, QTimer, Signal
    from PySide6.QtGui import QColor

    from evolab import paths
    from evolab.app.export import dialog as D
    from evolab.app.export import job as J
    from evolab.engine import seeds
    from evolab.engine.replay import replay
    from evolab.policy.genome import layout_for
    from evolab.spec.io import load_env

    spec = load_env(paths.builtin_env_dir() / "lander.evoenv")
    layout = layout_for(spec)
    rep = replay(spec, layout, np.zeros(layout.dim), seeds.display_trial())
    rep = dataclasses.replace(rep, trajectory=dataclasses.replace(
        rep.trajectory, poses=rep.trajectory.poses[:31].copy(), outputs=rep.trajectory.outputs[:31].copy(),
        end_step=None))

    class Done(QObject):
        progressed = Signal(int, int, int)
        noted = Signal(str)
        finished = Signal(dict)
        failed = Signal(str, str)
        cancelled = Signal()
        running = False
        encoder_hint = ""

        def start(self, job: object, out: Path) -> None:
            self.running = True
            QTimer.singleShot(0, lambda: (setattr(self, "running", False),
                                          self.finished.emit({"path": str(out), "frames": 31, "bytes": 1,
                                                              "encoder": "builtin"})))

        def cancel(self) -> None:
            pass

    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    settings.setValue(D.KEY_FORMAT, "gif")  # 지난번 GIF: 짧은 창으로 열린다
    dlg = D.ExportDialog(None, kind="replay", items=[(rep, QColor(J.NOW_COLOR), "지금 챔피언 (4세대)")],
                         title="지금 챔피언 (4세대)", default_name="영상", meta={}, settings=settings, probe=False)
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitExposed(dlg)
    assert _overlapping_rows(dlg) == []
    dlg.set_encoder("builtin", "test")
    dlg.set_format("mp4")  # 크기·카메라 줄과 두 줄 인코더 안내가 생긴다
    qtbot.wait(50)
    assert _overlapping_rows(dlg) == []
    dlg.make_runner = Done  # type: ignore[method-assign]
    deep = tmp_path.joinpath(*(["아주 긴 폴더 이름"] * 6))  # 저장 글이 세 줄쯤 된다
    dlg.ask_save_path = lambda fmt, default: deep / "영상.mp4"  # type: ignore[method-assign]
    assert dlg.save() == deep / "영상.mp4"
    qtbot.wait(50)
    assert dlg.status.isVisible() and dlg.btn_open_folder.isVisible()
    assert _overlapping_rows(dlg) == []

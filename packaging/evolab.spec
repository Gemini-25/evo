# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 설정: 하나의 폴더 배포본 (dist/evolab/evolab.exe, dist/evolab/evolab-cli.exe).
# 빌드: uv run pyinstaller packaging/evolab.spec --noconfirm

from pathlib import Path

ROOT = Path(SPECPATH).parent

# 쓰지 않는 큰 모듈은 빼서 배포본 크기를 줄인다.
EXCLUDES = [
    "tkinter",
    "matplotlib",
    "IPython",
    "pytest",
    "pyqtgraph.opengl",
    "PySide6.QtNetwork",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuickWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtPdf",
    "PySide6.QtDesigner",
    "PySide6.QtUiTools",
    "PySide6.QtHelp",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.Qt3DCore",
]

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[(str(ROOT / "environments"), "environments")],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="evolab",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # 더블클릭했을 때 검은 창이 뜨지 않게
)

# 명령줄용: 표준 출력과 종료 코드가 필요한 selftest / train / bench는 이쪽으로 실행한다.
# 창 모드 exe는 셸이 기다리지 않고 출력도 없다.
exe_cli = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="evolab-cli",
    debug=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    exe_cli,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="evolab",
)

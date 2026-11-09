# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 설정: 하나의 폴더 배포본 (dist/evolab/evolab.exe, dist/evolab/evolab-cli.exe).
# 빌드: uv run pyinstaller packaging/evolab.spec --noconfirm

from pathlib import Path

ROOT = Path(SPECPATH).parent

# 라이선스 고지를 모은다(설치된 의존성의 라이선스 파일 + 요약). 배포본의 licenses 폴더가 된다.
import sys as _sys

_sys.path.insert(0, str(ROOT / "packaging"))
from collect_licenses import collect  # noqa: E402

LICENSES = collect(ROOT / "build" / "licenses")
ICON = str(ROOT / "packaging" / "icons" / "evolab.ico")

# exe 파일 버전 정보(v3.1 설계 D45): 파일 버전 3.1.0.0처럼 pyproject 버전 네 칸, 제품 이름은 strings.py APP_NAME.
# 탐색기 속성·작업 관리자·오류 보고에서 빌드를 가린다. 리눅스 빌드에서는 PyInstaller가 무시한다(경고 한 줄).
from version_info import write_version_files  # noqa: E402

VERSION_FILES = write_version_files(ROOT / "build" / "version_info")

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
    datas=[
        (str(ROOT / "environments"), "environments"),
        (str(ROOT / "src" / "evolab" / "assets"), "assets"),
        # 폰 재생 파일 틀(v3.1 설계 D56): phone.TEMPLATE_PATH = 모듈 옆 assets/, 배포본에서는 _MEIPASS/evolab/app/export/assets/
        (str(ROOT / "src" / "evolab" / "app" / "export" / "assets"), "evolab/app/export/assets"),
        (str(LICENSES), "licenses"),
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    # 에디터 속성 패널이 칸 설명(풍선 도움말)을 모델 소스의 docstring에서 읽으므로 소스도 넣는다.
    module_collection_mode={"evolab.spec.model": "pyz+py"},
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
    icon=ICON,
    version=str(VERSION_FILES["evolab"]),
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
    icon=ICON,
    version=str(VERSION_FILES["evolab-cli"]),
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

"""프로그램이 쓰는 폴더 위치.

사용자 데이터(환경 파일, 실행 기록)는 문서 폴더 아래에 두고, 위치는 화면에서 Qt의
QStandardPaths로 정한다(OneDrive로 옮겨진 문서 폴더도 따라가기 위해서). 이 모듈은 Qt 없이도
쓰는 프로그램 내부 폴더(로그, 캐시)만 다룬다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "evolab"


def user_state_dir() -> Path:
    """로그와 캐시를 두는 사용자별 폴더. 문서 폴더가 아니다."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_STATE_HOME")
        root = Path(base) if base else Path.home() / ".local" / "state"
    return root / APP_DIR_NAME


def log_dir() -> Path:
    return user_state_dir() / "logs"


def builtin_env_dir() -> Path:
    """내장 예제 환경 파일 폴더. exe로 묶이면 배포본 안의 environments 폴더."""
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle is not None:
        return Path(bundle) / "environments"
    return Path(__file__).resolve().parents[2] / "environments"


BUILTIN_ORDER: tuple[str, ...] = ("lander", "track_car", "hill_car", "jump", "glider", "wheels")
"""내장 예제의 정해진 순서(파일 이름에서 .evoenv를 뺀 것): 착륙선, 트랙 자동차, 언덕 넘는 자동차, 점프 게임, 글라이더,
(v2.1) 바퀴 고르는 언덕차."""


def builtin_envs() -> list[Path]:
    """내장 예제 환경 파일: BUILTIN_ORDER 순서로 있는 것, 그다음 나머지 *.evoenv를 이름순으로(v2 설계 D27).

    에디터 「내장 예제에서 시작하기」, 실험실 환경 목록, 반복 실행 대화 상자, 자체 점검, 첫 실행 복사가 같은 순서를 쓴다.
    """
    folder = builtin_env_dir()
    files = {p.stem: p for p in folder.glob("*.evoenv") if p.is_file()}
    ordered = [files.pop(stem) for stem in BUILTIN_ORDER if stem in files]
    return ordered + [files[stem] for stem in sorted(files)]


def bundle_dir(name: str) -> Path:
    """배포본에 함께 들어가는 폴더(assets, licenses). exe로 묶이면 _MEIPASS 아래, 개발 중에는 저장소 안."""
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle is not None:
        return Path(bundle) / name
    root = Path(__file__).resolve().parents[2]
    return root / ("src/evolab/assets" if name == "assets" else name)


def icon_path() -> Path:
    return bundle_dir("assets") / "icon.png"


def documents_dir() -> Path:
    """문서 폴더. 윈도우에서 OneDrive로 옮겨진 문서 폴더도 따라간다(Qt QStandardPaths)."""
    try:
        from PySide6.QtCore import QStandardPaths

        docs = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        if docs:
            return Path(docs)
    except Exception:  # Qt를 못 불러오는 환경
        pass
    return Path.home() / "Documents"


PATHS_FILE = "paths.ini"
"""데이터 폴더 위치 설정. 화면 설정(settings.ini, Qt 형식)과 따로 둔다: Qt 없이(명령줄, 학습 프로세스) 같은 값을 읽기 위해."""


def custom_data_dir() -> Path | None:
    """설정에서 바꾼 데이터 폴더(없으면 None)."""
    import configparser

    cfg = configparser.ConfigParser(interpolation=None)
    try:
        cfg.read(user_state_dir() / PATHS_FILE, encoding="utf-8")
    except (OSError, configparser.Error, UnicodeDecodeError):
        return None
    value = cfg.get("paths", "data_dir", fallback="").strip()
    return Path(value) if value else None


def set_custom_data_dir(folder: Path | None) -> None:
    """데이터 폴더를 바꾼다(None이면 기본 문서\\진화실험실로)."""
    import configparser

    cfg = configparser.ConfigParser(interpolation=None)
    cfg["paths"] = {"data_dir": str(folder) if folder is not None else ""}
    path = user_state_dir() / PATHS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        cfg.write(f)


def data_dir() -> Path:
    """사용자 데이터 폴더: 기본은 문서\\진화실험실, 설정에서 바꿀 수 있다(명세: 저장, 기록, 내보내기)."""
    from evolab.app import strings as S

    return custom_data_dir() or documents_dir() / S.DATA_DIR_NAME


def runs_dir() -> Path:
    return data_dir() / "runs"


def user_env_dir() -> Path:
    """사용자 환경 파일 폴더: 문서\\진화실험실\\environments"""
    return data_dir() / "environments"


def records_dir() -> Path:
    """직접 조종 입력 기록 폴더: 데이터 폴더\\records (설정에서 데이터 폴더를 바꾸면 따라간다)."""
    return data_dir() / "records"


def default_records_dir() -> Path:
    """데이터 폴더를 바꾸기 전 기본 위치의 기록 폴더(문서\\진화실험실\\records). 예전 기록을 계속 읽기 위해 둔다."""
    from evolab.app import strings as S

    return documents_dir() / S.DATA_DIR_NAME / "records"


def install_examples(target: Path | None = None) -> list[Path]:
    """첫 실행 때 내장 예제를 사용자 환경 폴더로 복사한다. 이미 있는 파일은 건드리지 않는다(사용자가 고쳤을 수 있다).

    복사한 파일 목록을 돌려준다.
    """
    import shutil

    dest = target if target is not None else user_env_dir()
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in builtin_envs():
        out = dest / src.name
        if not out.exists():
            shutil.copyfile(src, out)
            copied.append(out)
    return copied

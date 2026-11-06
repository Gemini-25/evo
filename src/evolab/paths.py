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

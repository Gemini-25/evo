"""프로그램이 쓰는 폴더 위치.

사용자 데이터(환경 파일, 실행 기록)는 문서 폴더 아래에 두고, 위치는 화면에서 Qt의
QStandardPaths로 정한다(OneDrive로 옮겨진 문서 폴더도 따라가기 위해서). 이 모듈은 Qt 없이도
쓰는 프로그램 내부 폴더(로그, 캐시)만 다룬다.

(v3 설계 D5) 환경 변수 EVOLAB_DATA_DIR(데이터 폴더)와 EVOLAB_STATE_DIR(설정·로그 폴더)가 있으면 그 폴더를 먼저 쓴다.
자체 점검·설치 점검·테스트가 사용자의 문서 폴더와 settings.ini를 건드리지 않게 하려는 것이다. 환경 변수라서 spawn 자식
(학습 프로세스, 워커)도 그대로 물려받는다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "evolab"

DATA_DIR_ENV = "EVOLAB_DATA_DIR"
"""이 환경 변수가 있으면 데이터 폴더(환경 파일, 실행 기록, 직접 조종 기록)로 쓴다. 설정에서 바꾼 폴더(paths.ini)보다 먼저다."""
STATE_DIR_ENV = "EVOLAB_STATE_DIR"
"""이 환경 변수가 있으면 설정·로그 폴더(settings.ini, paths.ini, logs)로 쓴다."""


def _env_dir(name: str) -> Path | None:
    """환경 변수로 준 폴더(비었거나 없으면 None). 상대 경로는 지금 작업 폴더 기준의 절대 경로로 바꾼다."""
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser().absolute()


def user_state_dir() -> Path:
    """로그와 캐시를 두는 사용자별 폴더. 문서 폴더가 아니다. EVOLAB_STATE_DIR가 있으면 그 폴더."""
    override = _env_dir(STATE_DIR_ENV)
    if override is not None:
        return override
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


BUILTIN_ORDER: tuple[str, ...] = ("lander", "track_car", "hill_car", "jump", "glider", "wheels", "running")
"""내장 환경의 정해진 순서(파일 이름에서 .evoenv를 뺀 것): 내장 예제 여섯(착륙선, 트랙 자동차, 언덕 넘는 자동차, 점프 게임,
글라이더, (v2.1) 바퀴 고르는 언덕차), 그 뒤는 종목 프리셋((6.2.0) 달리기. 6.3.0에 권투·축구: 관문을 넘은 것만)."""


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


def recipe_dir() -> Path:
    """내장 조리법 폴더(assets/recipes/<이름>.json, 6.2.0). 폴더가 없을 수도 있다(읽는 쪽이 본다)."""
    return bundle_dir("assets") / "recipes"


def user_template_dir() -> Path:
    """사용자 틀 폴더: 문서\\진화실험실\\templates (6.2.0은 읽기만 한다. 만들지 않는다)."""
    return data_dir() / "templates"


def template_dirs() -> list[Path]:
    """틀(.evotpl)을 찾는 폴더: 내장(assets/templates) → 사용자 폴더 순서. 없는 폴더도 그대로 돌려준다(읽는 쪽이 건너뜀)."""
    return [bundle_dir("assets") / "templates", user_template_dir()]


_BAD_FILE_CHARS = frozenset('\\/:*?"<>|')
"""윈도우 파일 이름에 쓸 수 없는 글자."""
_RESERVED_FILE_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL", *(f"{dev}{n}" for dev in ("COM", "LPT") for n in "123456789\u00b9\u00b2\u00b3")]
)
"""윈도우 장치 이름(대소문자 무관, 첫 점 앞부분이 이것이면 확장자가 있어도 열 수 없다)."""
SAFE_STEM_MAX = 80
"""safe_file_stem이 돌려주는 줄기의 최대 글자 수."""


def safe_file_stem(name: str) -> str:
    """사용자가 적은 이름을 윈도우에서도 안전한 파일 줄기(확장자 없는 이름)로 바꾼다(6.2.0, 설계 D42).

    \\ / : * ? " < > | 와 제어 글자는 "_"로, 앞뒤 빈칸과 끝의 점은 떼고, SAFE_STEM_MAX(80)자에서 자르고, CON·NUL 같은 장치
    이름이면 그 뒤에 "_"를 붙인다. 남는 것이 없으면 S.PRESET_NAME_FALLBACK. 같은 입력이면 늘 같은 결과이고, 결과를 다시
    넣어도 그대로다.
    """
    text = "".join("_" if (ch in _BAD_FILE_CHARS or ord(ch) < 32 or ord(ch) == 127) else ch for ch in str(name))
    text = text.strip().rstrip(". ")[:SAFE_STEM_MAX].rstrip(". ")
    if not text:
        from evolab.app import strings as S

        text = S.PRESET_NAME_FALLBACK
    head, dot, tail = text.partition(".")
    if head.rstrip().upper() in _RESERVED_FILE_NAMES:
        text = (head.rstrip() + "_" + dot + tail)[:SAFE_STEM_MAX].rstrip(". ")
    return text


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
    """사용자 데이터 폴더: 기본은 문서\\진화실험실, 설정에서 바꿀 수 있다(명세: 저장, 기록, 내보내기).

    EVOLAB_DATA_DIR가 있으면 설정과 상관없이 그 폴더다(v3 설계 D5).
    """
    override = _env_dir(DATA_DIR_ENV)
    if override is not None:
        return override
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
    """데이터 폴더를 바꾸기 전 기본 위치의 기록 폴더(문서\\진화실험실\\records). 예전 기록을 계속 읽기 위해 둔다.

    EVOLAB_DATA_DIR가 있으면 그 폴더의 records(문서 폴더를 읽지 않는다).
    """
    override = _env_dir(DATA_DIR_ENV)
    if override is not None:
        return override / "records"
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

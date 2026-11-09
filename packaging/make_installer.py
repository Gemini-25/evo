"""Inno Setup 스크립트(.iss)를 틀과 strings.py·pyproject 값으로 만든다 (v3.1 설계 D35~D43, 2.10, 4.12).

    uv run python packaging/make_installer.py --dist dist\\evolab --out build\\installer\\evolab.iss
        [--version-override X.Y.Z] [--fast] [--sign] [--messages-file Korean.isl] [--names-out names.json]

- 틀은 `packaging/installer/evolab.iss.in`(ASCII만, `@NAME@` 자리). Inno는 중괄호를 상수·`{{` 이스케이프·ISPP 인라인에 쓰므로
  자리 표시는 골뱅이로 쓴다.
- 사용자에게 보이는 글은 모두 strings.py에서 온다: `APP_NAME`(프로그램·바로 가기·폴더 이름 줄), `INST_*`와
  `DATA_DIR_NAME`(`[CustomMessages]`). Inno 기본 문구는 공식 번역 `compiler:Languages\\Korean.isl`(6.5.0+).
- 버전 정본은 `pyproject.toml`의 `[project].version`. `evolab/__init__.py`의 `__version__`과 다르면 실패한다(D3).
  `--version-override`는 업그레이드 시험용 옛 판(2.99.0)을 같은 배포본으로 만들 때만 쓴다(D47).
- 결과는 UTF-8 BOM·CRLF(Inno Setup 6·7이 한글을 바르게 읽는 형식). 경로는 출력 폴더 기준 상대 경로(같은 드라이브일 때).
- `AppId`는 영영 바꾸지 않는다(D38, 테스트 T33).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path, PureWindowsPath
from typing import Final

PACKAGING: Final = Path(__file__).resolve().parent
ROOT: Path = PACKAGING.parent
TEMPLATE: Final = PACKAGING / "installer" / "evolab.iss.in"

APP_ID: Final = "{133D1A60-B1E2-4B1D-9352-37DB65C86C97}"
"""절대 바꾸지 않는다(D38): 바꾸면 윈도우가 다른 프로그램으로 보고 덮어 설치 대신 둘이 깔린다. v3 설계 때 uuid4로 한 번 만든 값."""
APP_MUTEX: Final = "evolab-app-running"
"""화면 프로세스가 쥐는 이름 있는 뮤텍스(`evolab.winapp.MUTEX_NAME`, 테스트가 같은지 본다). 설치·제거가 보고 닫으라고 한다(D40)."""
STATE_DIR_NAME: Final = "evolab"
"""설정·로그 폴더 이름 `%LOCALAPPDATA%\\evolab`(`evolab.paths.APP_DIR_NAME`, 테스트가 같은지 본다)."""
SIGN_TOOL: Final = "evolabsign"
"""`--sign`일 때 `SignTool=` 이름. ISCC에 `/Sevolabsign=<명령>`으로 실제 명령을 준다(docs/signing.md)."""
DEFAULT_MESSAGES_FILE: Final = r"compiler:Languages\Korean.isl"
COMPRESSION: Final = "lzma2/ultra64"
FAST_COMPRESSION: Final = "lzma2/fast"
OLD_TEST_VERSION: Final = "2.99.0"
"""업그레이드·내려 설치 시험에 쓰는 옛 판 번호(설치 연기 시험 installer_smoke.ps1, D47)."""

CUSTOM_MESSAGES: Final[dict[str, str]] = {
    "TasksGroup": "INST_TASKS_GROUP",
    "TaskDesktop": "INST_TASK_DESKTOP",
    "RunNow": "INST_RUN_NOW",
    "ReportShortcut": "INST_REPORT_SHORTCUT",
    "NewerInstalled": "INST_NEWER_INSTALLED",
    "AskSettings": "INST_ASK_SETTINGS",
    "DataKept": "INST_DATA_KEPT",
    "DataDirName": "DATA_DIR_NAME",
}
"""`[CustomMessages]` 이름 → strings.py 상수. `%1`·`%2`·`%n`은 Inno 자리(줄 바꿈은 `%n`으로만)."""

_VERSION_RE: Final = re.compile(r"\d+\.\d+\.\d+")
_PLACEHOLDER_RE: Final = re.compile(r"@([A-Z][A-Z0-9_]*)@")


class InstallerError(Exception):
    """버전이 맞지 않거나 틀·문구가 잘못됐다(명령은 종료 코드 1)."""


def _strings():
    """strings.py(Qt를 불러오지 않는다). 저장소의 src를 먼저 본다."""
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from evolab.app import strings as S

    return S


def read_version(root: Path | None = None) -> str:
    """pyproject.toml의 버전. `src/evolab/__init__.py`의 `__version__`과 다르거나 X.Y.Z 꼴이 아니면 InstallerError."""
    root = ROOT if root is None else root
    try:
        version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
        init = (root / "src" / "evolab" / "__init__.py").read_text(encoding="utf-8")
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        raise InstallerError(f"cannot read the version: {exc}") from exc
    found = re.search(r"""^__version__\s*=\s*["']([^"']+)["']""", init, re.MULTILINE)
    package = found.group(1) if found else None
    if package != version:
        raise InstallerError(f"version mismatch: pyproject.toml {version!r}, src/evolab/__init__.py {package!r}")
    check_version(version)
    return version


def check_version(version: str) -> str:
    """X.Y.Z(숫자 셋)만 받는다: 설치된 판과 세 칸으로 비교하고(`[Code]`) 파일 버전 네 칸을 만든다."""
    if not _VERSION_RE.fullmatch(version):
        raise InstallerError(f"version must look like 3.1.0, not {version!r}")
    return version


def numeric_version(version: str) -> str:
    """`VersionInfoVersion` 네 칸: 3.1.0 → 3.1.0.0."""
    return check_version(version) + ".0"


def _win_path(path: Path, base: Path | None) -> str:
    """Inno 스크립트에 쓸 경로(역슬래시). base가 있으면 base 기준 상대 경로, 드라이브가 달라 못 만들면 절대 경로."""
    path = Path(os.path.abspath(path))
    if base is not None:
        try:
            return str(PureWindowsPath(os.path.relpath(path, os.path.abspath(base))))
        except ValueError:
            pass
    return str(PureWindowsPath(path))


def _custom_message_lines(S) -> list[str]:
    lines = []
    for name, constant in CUSTOM_MESSAGES.items():
        value = getattr(S, constant)
        if not isinstance(value, str) or not value.strip() or any(c in value for c in "\r\n{}"):
            raise InstallerError(f"strings.{constant} cannot be an Inno custom message (one line, no braces)")
        lines.append(f"korean.{name}={value}")
    return lines


def render_iss(dist: Path, *, version: str, fast: bool = False, sign: bool = False, out_dir: Path | None = None,
               messages_file: Path | None = None) -> str:
    """`.iss` 글(CRLF). dist = PyInstaller 한 폴더 배포본(`dist/evolab`), out_dir = `.iss`를 둘 폴더(상대 경로의 기준).

    version은 X.Y.Z. fast면 압축만 `lzma2/fast`(시험용 옛 판), sign이면 `SignTool=evolabsign`·`SignedUninstaller=yes`.
    messages_file을 주면 공식 `compiler:Languages\\Korean.isl` 대신 그 파일(설계 7.4 되돌릴 지점 6: 러너에 없을 때).
    """
    S = _strings()
    check_version(version)
    if sign:
        signing = f"SignTool={SIGN_TOOL}\nSignedUninstaller=yes"
    else:
        signing = f"; Code signing is off: make_installer.py --sign adds SignTool={SIGN_TOOL}, SignedUninstaller=yes."
    values = {
        "VERSION": version,
        "VERSION4": numeric_version(version),
        "APP_ID": APP_ID,
        "APP_NAME": S.APP_NAME,
        "PUBLISHER": S.INST_PUBLISHER,
        "ICON": _win_path(ROOT / "packaging" / "icons" / "evolab.ico", out_dir),
        "DIST": _win_path(dist, out_dir),
        "COMPRESSION": FAST_COMPRESSION if fast else COMPRESSION,
        "MUTEX": APP_MUTEX,
        "SIGNING": signing,
        "MESSAGES_FILE": DEFAULT_MESSAGES_FILE if messages_file is None else _win_path(messages_file, out_dir),
        "CUSTOM_MESSAGES": "\n".join(_custom_message_lines(S)),
        "STATE_DIR_NAME": STATE_DIR_NAME,
    }
    template = TEMPLATE.read_text(encoding="utf-8")
    missing = sorted(set(_PLACEHOLDER_RE.findall(template)) - set(values))
    if missing:
        raise InstallerError(f"template placeholders without a value: {missing}")
    text = _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template)
    if any('"' in v for v in (values["DIST"], values["ICON"], S.APP_NAME, S.INST_REPORT_SHORTCUT)):
        raise InstallerError("names and paths with double quotes cannot go into the quoted Inno parameters")
    lines = text.replace("\r\n", "\n").split("\n")
    while lines and not lines[-1]:
        lines.pop()
    return "\r\n".join(lines) + "\r\n"


def write_iss(text: str, out: Path) -> Path:
    """UTF-8 BOM으로 쓴다(text는 이미 CRLF)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    return out


def names(version: str) -> dict[str, str]:
    """설치 연기 시험(installer_smoke.ps1)이 쓰는 이름들. 한글이 PowerShell 콘솔 인코딩을 거치지 않게 JSON 파일로 넘긴다."""
    S = _strings()
    return {
        "version": version,
        "app_id": APP_ID,
        "app_name": S.APP_NAME,
        "report_shortcut": S.INST_REPORT_SHORTCUT,
        "data_dir_name": S.DATA_DIR_NAME,
        "state_dir_name": STATE_DIR_NAME,
        "mutex": APP_MUTEX,
        "old_test_version": OLD_TEST_VERSION,
    }


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="make_installer.py", description="Generate the Inno Setup script for evolab.")
    p.add_argument("--dist", type=Path, default=ROOT / "dist" / "evolab", help="PyInstaller one-folder build")
    p.add_argument("--out", type=Path, default=ROOT / "build" / "installer" / "evolab.iss", help="output .iss")
    p.add_argument("--version-override", default=None, help="X.Y.Z for a test build (the old 2.99.0 setup)")
    p.add_argument("--fast", action="store_true", help="lzma2/fast compression (test builds)")
    p.add_argument("--sign", action="store_true", help="turn on SignTool=evolabsign and SignedUninstaller=yes")
    p.add_argument("--messages-file", type=Path, default=None, help="Korean.isl to use instead of the compiler's")
    p.add_argument("--names-out", type=Path, default=None, help="also write names for installer_smoke.ps1 (JSON)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        version = read_version()  # 덮어쓰기를 줘도 pyproject와 __version__이 같은지는 늘 본다
        if args.version_override is not None:
            version = check_version(args.version_override)
        out = Path(os.path.abspath(args.out))
        text = render_iss(args.dist, version=version, fast=args.fast, sign=args.sign, out_dir=out.parent,
                          messages_file=args.messages_file)
    except InstallerError as exc:
        print(f"make_installer: {exc}", file=sys.stderr)
        return 1
    write_iss(text, out)
    if args.names_out is not None:
        args.names_out.parent.mkdir(parents=True, exist_ok=True)
        args.names_out.write_text(json.dumps(names(version), ensure_ascii=True, indent=1) + "\n", encoding="ascii")
    print(f"{out}: version {version}, {'fast' if args.fast else 'ultra'} compression, signing "
          f"{'on' if args.sign else 'off'}, {text.count(chr(10))} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

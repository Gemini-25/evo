"""배포 파일: 휴대용 zip, SHA256SUMS.txt, 릴리스 글(v3.1 설계 D44, 4.20 「Portable zip and checksums」, W10).

    uv run python packaging/installer/release_files.py --dist dist\\evolab --out build\\release
        [--setup build\\installer\\Output\\evolab-setup-3.1.0.exe] [--version 3.1.0]

out 폴더에 만드는 것:
- `evolab-<버전>-portable.zip`: 맨 위 폴더 `evolab-<버전>/` 안에 배포본 그대로 + `설치 없이 쓰기.txt`(strings.py
  `PORTABLE_README_NAME`·`PORTABLE_README`, UTF-8 BOM·CRLF: 오래된 메모장도 한글을 바르게 연다).
- 설치 파일 사본(`--setup`을 주면, 이름 그대로).
- `SHA256SUMS.txt`: `sha256sum` 형식(해시, 빈칸 둘, 파일 이름), 이름순, LF. `sha256sum -c`와 PowerShell `Get-FileHash`로 확인.
- `release-notes.md`·`release-title.txt`: GitHub 릴리스 본문과 제목. 글은 strings.py에서만 온다(제목 = `APP_NAME` + 버전,
  본문 = 도움말 「설치와 제거」 쪽에서 쪽 제목과 앱 안 링크를 뺀 것 + 체크섬).
설치 파일이 없으면(설계 7.4 되돌릴 지점 6: CI에서 설치 프로그램을 끝내 만들지 못함) zip과 체크섬만 만들고 경고를 찍는다.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
SUMS_NAME: Final = "SHA256SUMS.txt"
NOTES_NAME: Final = "release-notes.md"
TITLE_NAME: Final = "release-title.txt"
_APP_LINK_RE: Final = re.compile(r"\s*\[[^\]]*\]\(evolab://[^)]*\)")


def _strings():
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from evolab.app import strings as S

    return S


def portable_name(version: str) -> str:
    return f"evolab-{version}-portable.zip"


def setup_name(version: str) -> str:
    return f"evolab-setup-{version}.exe"


def portable_zip(dist: Path, out: Path, version: str) -> Path:
    """dist(PyInstaller 한 폴더 배포본)를 `evolab-<버전>/` 아래에 담고 안내 글을 더한 zip을 out에 쓴다."""
    S = _strings()
    if not (dist / "evolab.exe").is_file() and not (dist / "evolab").is_file():
        raise FileNotFoundError(f"no evolab executable in {dist}")
    top = f"evolab-{version}"
    readme = S.PORTABLE_README.format(version=version)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(dist.rglob("*")):
            if path.is_file():
                z.write(path, f"{top}/{path.relative_to(dist).as_posix()}")
        z.writestr(f"{top}/{S.PORTABLE_README_NAME}", b"\xef\xbb\xbf" + readme.encode("utf-8"))
    tmp.replace(out)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_sums(files: Sequence[Path], out: Path) -> Path:
    """`sha256sum` 형식 체크섬 파일(이름순, LF)."""
    lines = [f"{sha256(p)}  {p.name}\n" for p in sorted(files, key=lambda p: p.name)]
    out.write_text("".join(lines), encoding="ascii", newline="\n")
    return out


def release_title(version: str) -> str:
    return f"{_strings().APP_NAME} {version}"


def release_notes(version: str, sums: Path) -> str:
    """릴리스 본문(Markdown). 쪽 제목 줄(릴리스 제목과 겹침)과 앱 안에서만 열리는 `evolab://` 링크는 뺀다."""
    S = _strings()
    _title, body = S.HELP_PAGES["install"]
    body = _APP_LINK_RE.sub("", body).strip()
    body = re.sub(r"\A# [^\n]*\n+", "", body)
    checksums = sums.read_text(encoding="ascii").strip()
    return f"{body}\n\n## SHA256\n\n```\n{checksums}\n```\n"


def build(dist: Path, out_dir: Path, version: str, setup: Path | None) -> list[Path]:
    """out_dir에 zip·설치 파일 사본·SHA256SUMS·릴리스 글을 만들고 체크섬을 넣은 파일 목록을 돌려준다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [portable_zip(dist, out_dir / portable_name(version), version)]
    if setup is not None and setup.is_file():
        if setup.name != setup_name(version):
            raise ValueError(f"setup file {setup.name} does not match version {version}")
        copied = out_dir / setup.name
        if setup.resolve() != copied.resolve():
            shutil.copy2(setup, copied)
        files.append(copied)
    else:
        print(f"release_files: WARNING no setup file ({setup}); only the portable zip is listed", file=sys.stderr)
    sums = write_sums(files, out_dir / SUMS_NAME)
    (out_dir / NOTES_NAME).write_text(release_notes(version, sums), encoding="utf-8", newline="\n")
    (out_dir / TITLE_NAME).write_text(release_title(version) + "\n", encoding="utf-8", newline="\n")
    return files


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="release_files.py", description="Portable zip, SHA256SUMS.txt and release notes.")
    p.add_argument("--dist", type=Path, default=ROOT / "dist" / "evolab")
    p.add_argument("--out", type=Path, default=ROOT / "build" / "release")
    p.add_argument("--setup", type=Path, default=None, help="the installer to list in SHA256SUMS.txt")
    p.add_argument("--version", default=None, help="X.Y.Z (default: pyproject.toml)")
    args = p.parse_args(argv)
    sys.path.insert(0, str(ROOT / "packaging"))
    from make_installer import InstallerError, check_version, read_version

    try:
        version = read_version() if args.version is None else check_version(args.version)
    except InstallerError as exc:
        print(f"release_files: {exc}", file=sys.stderr)
        return 1
    try:
        files = build(args.dist, args.out, version, args.setup)
    except (OSError, ValueError) as exc:
        print(f"release_files: {exc}", file=sys.stderr)
        return 1
    for path in [*files, args.out / SUMS_NAME, args.out / NOTES_NAME, args.out / TITLE_NAME]:
        print(f"{path.name}: {path.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""배포본 PE 파일(.exe·.dll·.pyd) 가운데 Authenticode 서명이 없는 것(v3.1 설계 D46, docs/signing.md).

    uv run python packaging/list_unsigned.py dist\\evolab            # 개수와 서명 없는 파일 목록
    uv run python packaging/list_unsigned.py dist\\evolab --paths    # 서명 없는 파일의 절대 경로만 한 줄에 하나(sign.ps1)

PE 머리의 보안 디렉터리(데이터 디렉터리 4번, 인증서 표)가 있는지만 본다. 서명이 유효한지(인증서 사슬·시각 도장)는 보지 않는다:
그것은 `signtool verify /pa`가 한다. 파이썬만 쓰므로 리눅스에서도 돈다. 종료 코드는 늘 0(정보), 폴더가 없으면 2.
"""

from __future__ import annotations

import argparse
import struct
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Final

PE_SUFFIXES: Final = frozenset({".exe", ".dll", ".pyd"})
_SECURITY_DIR: Final = 4  # IMAGE_DIRECTORY_ENTRY_SECURITY


def certificate_table_size(path: Path) -> int | None:
    """PE 파일의 인증서 표 크기(바이트). 서명이 없으면 0, PE가 아니거나 읽을 수 없으면 None."""
    try:
        with path.open("rb") as f:
            dos = f.read(0x40)
            if len(dos) < 0x40 or dos[:2] != b"MZ":
                return None
            (pe_offset,) = struct.unpack_from("<I", dos, 0x3C)
            f.seek(pe_offset)
            head = f.read(4 + 20 + 240)
    except OSError:
        return None
    if len(head) < 4 + 20 + 2 or head[:4] != b"PE\0\0":
        return None
    opt = 4 + 20  # PE 서명 + COFF 머리 뒤가 선택 머리
    (magic,) = struct.unpack_from("<H", head, opt)
    if magic == 0x10B:  # PE32
        count_at, dirs_at = opt + 92, opt + 96
    elif magic == 0x20B:  # PE32+
        count_at, dirs_at = opt + 108, opt + 112
    else:
        return None
    entry_at = dirs_at + 8 * _SECURITY_DIR
    if len(head) < entry_at + 8:
        return None
    (count,) = struct.unpack_from("<I", head, count_at)
    if count <= _SECURITY_DIR:
        return 0
    _offset, size = struct.unpack_from("<II", head, entry_at)
    return size


def scan(dist: Path) -> tuple[list[Path], list[Path]]:
    """(서명 있는 PE, 서명 없는 PE)를 dist 기준 상대 경로로(이름순)."""
    signed: list[Path] = []
    unsigned: list[Path] = []
    for path in sorted(dist.rglob("*")):
        if path.suffix.lower() not in PE_SUFFIXES or not path.is_file():
            continue
        size = certificate_table_size(path)
        if size is None:
            continue
        (signed if size > 0 else unsigned).append(path.relative_to(dist))
    return signed, unsigned


def _group(rel: Path) -> str:
    """요약용 묶음: _internal 아래 첫 폴더(PySide6, numpy …), 바로 아래 파일은 _internal, 맨 위는 그 이름."""
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "_internal":
        return f"_internal/{parts[1]}"
    return "/".join(parts[:-1]) or parts[-1]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="list_unsigned.py", description="List PE files without an Authenticode signature.")
    p.add_argument("dist", type=Path, help="PyInstaller one-folder build (dist/evolab)")
    p.add_argument("--paths", action="store_true", help="print only the absolute paths of unsigned files")
    args = p.parse_args(argv)
    if not args.dist.is_dir():
        print(f"list_unsigned: no folder {args.dist}", file=sys.stderr)
        return 2
    signed, unsigned = scan(args.dist)
    if args.paths:
        for rel in unsigned:
            print((args.dist / rel).resolve())
        return 0
    unsigned_bytes = sum((args.dist / rel).stat().st_size for rel in unsigned)
    print(f"PE files: {len(signed) + len(unsigned)}, signed: {len(signed)}, unsigned: {len(unsigned)} "
          f"({unsigned_bytes / 1e6:.1f} MB)")
    for group, n in sorted(Counter(_group(rel) for rel in unsigned).items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:4d}  {group}")
    for rel in unsigned:
        print(f"  unsigned: {rel.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

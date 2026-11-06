"""배포본에 넣을 라이선스 고지를 모은다(빌드 때 evolab.spec이 부른다).

evolab의 실행 의존성(설치된 배포본 메타데이터로 따라간 전체 목록)마다 라이선스 이름과 라이선스 파일을
licenses/<이름>-<버전>/에 복사하고, 요약을 licenses/THIRD_PARTY_NOTICES.txt에 쓴다. 파이썬 자체(PSF)와
Qt(PySide6에 들어 있는 LGPLv3, 동적 링크라 사용자가 교체할 수 있음)도 적는다.
"""

from __future__ import annotations

import re
import shutil
import sys
from importlib import metadata
from pathlib import Path

LICENSE_FILE = re.compile(r"(LICEN[CS]E|COPYING|NOTICE|AUTHORS)", re.IGNORECASE)


def dependency_closure(root: str = "evolab") -> list[metadata.Distribution]:
    seen: dict[str, metadata.Distribution] = {}
    stack = [root]
    while stack:
        name = stack.pop()
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in seen:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        seen[key] = dist
        for req in dist.requires or []:
            if "extra ==" in req:
                continue
            m = re.match(r"^\s*([A-Za-z0-9_.\-]+)", req)
            if m:
                stack.append(m.group(1))
    return [d for k, d in sorted(seen.items()) if k != "evolab"]


def license_name(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    expr = meta.get("License-Expression")
    if expr:
        return expr
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    if classifiers:
        return ", ".join(classifiers)
    lic = (meta.get("License") or "").strip()
    return lic.splitlines()[0][:120] if lic else "UNKNOWN"


def collect(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    lines = [
        "Evolution Lab (진화 실험실) - third-party software notices",
        "",
        "This program bundles the following open-source components. Their license texts are in the",
        "folders next to this file. Qt (in PySide6) is used under the LGPLv3 and is dynamically linked:",
        "the Qt DLLs in this folder can be replaced by the user.",
        "",
        f"Python {sys.version.split()[0]} - Python Software Foundation License",
    ]
    py_license = Path(sys.base_prefix) / "LICENSE.txt"
    if not py_license.exists():
        py_license = Path(sys.base_prefix) / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "LICENSE.txt"
    if py_license.exists():
        (out / "python").mkdir()
        shutil.copyfile(py_license, out / "python" / "LICENSE.txt")
    for dist in dependency_closure():
        name, version = dist.metadata["Name"], dist.version
        lines.append(f"{name} {version} - {license_name(dist)}")
        folder = out / f"{name}-{version}"
        copied = 0
        for f in dist.files or []:
            if LICENSE_FILE.search(f.name) or "licenses/" in str(f).replace("\\", "/"):
                src = Path(dist.locate_file(f))
                if src.is_file() and src.stat().st_size < 2_000_000:
                    folder.mkdir(exist_ok=True)
                    dest = folder / f.name
                    if dest.exists():
                        dest = folder / f"{f.parent.name}_{f.name}"
                    shutil.copyfile(src, dest)
                    copied += 1
        if not copied and name.lower().startswith(("pyside6", "shiboken6")):
            # PySide6 휠에는 라이선스 파일이 없다: Qt for Python은 LGPLv3(GPLv3를 참조)로 쓴다.
            folder.mkdir(exist_ok=True)
            here = Path(__file__).resolve().parent / "licenses"
            for f in ("LGPL-3.0.txt", "GPL-3.0.txt"):
                shutil.copyfile(here / f, folder / f)
            copied = 2
        if not copied:
            lic = dist.metadata.get("License")
            if lic and len(lic) > 200:
                folder.mkdir(exist_ok=True)
                (folder / "LICENSE.txt").write_text(lic, encoding="utf-8")
    (out / "THIRD_PARTY_NOTICES.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    target = collect(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/licenses"))
    print((target / "THIRD_PARTY_NOTICES.txt").read_text(encoding="utf-8"))

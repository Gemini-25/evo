"""v3.1 G 묶음: 설치 프로그램·exe 버전 정보·서명 목록·배포 파일·윈도우 CI 설정 (v3 설계 8.1 T33·T34, 4.12·4.13·4.20).

ISCC 컴파일·설치·제거는 윈도우 CI(W4·W5)에서만 된다. 여기서는 생성된 .iss 글과 도우미 스크립트를 정적으로·리눅스에서 본다.
PowerShell 구문 검사는 pwsh가 있을 때만(윈도우 CI tests job), windows.yml 구조 검사는 PyYAML이 있을 때만 돈다.
"""

from __future__ import annotations

import difflib
import importlib
import json
import re
import shutil
import struct
import subprocess
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

import pytest

import evolab
from evolab import paths, winapp
from evolab.app import strings as S

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"
sys.path.insert(0, str(PACKAGING))
sys.path.insert(0, str(PACKAGING / "installer"))

import list_unsigned  # noqa: E402
import make_installer as MI  # noqa: E402
import release_files  # noqa: E402
import version_info  # noqa: E402

HANGUL = re.compile(r"[가-힣ㄱ-ㆎ]")
FIXED_APP_ID = "{133D1A60-B1E2-4B1D-9352-37DB65C86C97}"  # 설계 D38: 이 값은 영영 바뀌지 않는다
WORKFLOW = ROOT / ".github" / "workflows" / "windows.yml"


def _render(**kw: Any) -> str:
    kw.setdefault("version", evolab.__version__)
    return MI.render_iss(ROOT / "dist" / "evolab", out_dir=ROOT / "build" / "installer", **kw)


def _sections(text: str) -> dict[str, list[str]]:
    """[섹션] → 줄 목록(머리 앞 줄은 "")."""
    out: dict[str, list[str]] = {"": []}
    current = ""
    for line in text.split("\r\n"):
        m = re.fullmatch(r"\[(\w+)\]", line)
        if m:
            current = m.group(1)
            out.setdefault(current, [])
        else:
            out[current].append(line)
    return out


def _custom_messages(text: str) -> dict[str, str]:
    lines = _sections(text)["CustomMessages"]
    found = {}
    for line in lines:
        if line.strip():
            m = re.fullmatch(r"korean\.(\w+)=(.*)", line)
            assert m, line
            assert m.group(1) not in found, f"custom message {m.group(1)} defined twice"
            found[m.group(1)] = m.group(2)
    return found


def _code_tokens(code: str) -> list[str]:
    """[Code]의 Pascal 낱말(문자열 '…'·주석 //, { }, (* *) 빼고), 소문자."""
    code = re.sub(r"'(?:[^']|'')*'", "''", code)
    code = re.sub(r"//[^\n]*", "", code)
    code = re.sub(r"\{[^}]*\}", "", code)
    code = re.sub(r"\(\*.*?\*\)", "", code, flags=re.S)
    return re.findall(r"[A-Za-z_]\w*", code.lower())


# ---------------------------------------------------------------------------
# T33 생성된 Inno Setup 스크립트
# ---------------------------------------------------------------------------
def test_app_id_never_changes() -> None:
    assert MI.APP_ID == FIXED_APP_ID
    assert f"AppId={{{FIXED_APP_ID}" in _render().split("\r\n")


def test_written_file_is_utf8_bom_with_crlf(tmp_path: Path) -> None:
    out = MI.write_iss(_render(), tmp_path / "evolab.iss")
    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    body = raw[3:]
    assert body.endswith(b"\r\n") and not body.endswith(b"\r\n\r\n")
    assert b"\n" not in body.replace(b"\r\n", b"") and b"\r" not in body.replace(b"\r\n", b"")
    body.decode("utf-8")  # 잘린 UTF-8이 없다(Inno 7.0.2는 잘못된 바이트를 오류로 본다)


def test_required_directives() -> None:
    lines = _render().split("\r\n")
    setup = _sections(_render())["Setup"]
    required = [
        f'#define AppVersion "{evolab.__version__}"',
        "AppVersion={#AppVersion}",
        f"VersionInfoVersion={evolab.__version__}.0",
        f"AppName={S.APP_NAME}",
        f"AppPublisher={S.INST_PUBLISHER}",
        f"DefaultDirName={{autopf}}\\{S.APP_NAME}",
        f"DefaultGroupName={S.APP_NAME}",
        "DisableProgramGroupPage=yes",
        "DisableDirPage=auto",
        "UsePreviousAppDir=yes",
        "PrivilegesRequired=lowest",
        "PrivilegesRequiredOverridesAllowed=commandline",
        "ArchitecturesAllowed=x64compatible",
        "ArchitecturesInstallIn64BitMode=x64compatible",
        "MinVersion=10.0.17763",
        "UninstallDisplayIcon={app}\\evolab.exe",
        "OutputBaseFilename=evolab-setup-{#AppVersion}",
        "Compression=lzma2/ultra64",
        "SolidCompression=yes",
        "LZMAUseSeparateProcess=yes",
        "AppMutex=evolab-app-running",
        "CloseApplications=yes",
        "RestartApplications=no",
        "SetupLogging=yes",
    ]
    for line in required:
        assert line in lines, line
    assert all(line in setup for line in required if "=" in line and not line.startswith("#"))
    assert not any(line.startswith(("SignTool=", "SignedUninstaller=")) for line in lines)
    assert 'Name: "korean"; MessagesFile: "compiler:Languages\\Korean.isl"' in _sections(_render())["Languages"]
    assert 'Type: filesandordirs; Name: "{app}\\_internal"' in _sections(_render())["InstallDelete"]
    assert 'Type: filesandordirs; Name: "{app}\\_internal"' in _sections(_render())["UninstallDelete"]


def test_sections_in_order() -> None:
    names = [m.group(1) for line in _render().split("\r\n") if (m := re.fullmatch(r"\[(\w+)\]", line))]
    assert names == ["Setup", "Languages", "CustomMessages", "Tasks", "InstallDelete", "Files", "Icons", "Run",
                     "UninstallDelete", "Code"]


def test_mutex_and_state_folder_match_the_program() -> None:
    assert MI.APP_MUTEX == winapp.MUTEX_NAME
    assert MI.STATE_DIR_NAME == paths.APP_DIR_NAME
    assert f"{{localappdata}}\\{paths.APP_DIR_NAME}" in "\n".join(_sections(_render())["Code"])


def test_paths_are_relative_to_the_script_folder() -> None:
    text = _render()
    assert ('Source: "..\\..\\dist\\evolab\\*"; DestDir: "{app}"; '
            'Flags: ignoreversion recursesubdirs createallsubdirs') in text
    assert "SetupIconFile=..\\..\\packaging\\icons\\evolab.ico" in text.split("\r\n")
    assert "OutputDir=Output" in text.split("\r\n")
    assert (ROOT / "packaging" / "icons" / "evolab.ico").is_file()


def test_custom_messages_are_the_strings_and_all_used_ones_exist() -> None:
    text = _render()
    defined = _custom_messages(text)
    assert defined == {name: getattr(S, const) for name, const in MI.CUSTOM_MESSAGES.items()}
    used = set(re.findall(r"\{cm:(\w+)", text)) | set(re.findall(r"CustomMessage\('(\w+)'\)", text))
    assert used <= set(defined), used - set(defined)
    assert set(defined) <= used, f"unused custom messages: {set(defined) - used}"
    inst = {n for n in dir(S) if n.startswith("INST_")}
    assert inst - {"INST_PUBLISHER"} == {c for c in MI.CUSTOM_MESSAGES.values() if c.startswith("INST_")}
    # 정의된 메시지 이름이 Korean.isl의 기본 [CustomMessages] 이름과 겹치지 않는다
    builtin = {"NameAndVersion", "AdditionalIcons", "CreateDesktopIcon", "CreateQuickLaunchIcon", "ProgramOnTheWeb",
               "UninstallProgram", "LaunchProgram", "AssocFileExtension", "AssocingFileExtension",
               "AutoStartProgramGroupDescription", "AutoStartProgram", "AddonHostProgramNotFound"}
    assert not set(defined) & builtin


def test_inno_placeholders_in_messages() -> None:
    m = _custom_messages(_render())
    assert "%1" in m["NewerInstalled"] and "%2" in m["NewerInstalled"]
    assert "%1" in m["AskSettings"] and "%1" in m["DataKept"]
    assert m["DataDirName"] == S.DATA_DIR_NAME
    for value in m.values():
        assert not re.search(r"%[3-9]", value) and "{" not in value


def test_hangul_only_in_messages_comments_and_name_lines() -> None:
    name_keys = ("AppName=", "AppVerName=", "AppPublisher=", "DefaultDirName=", "DefaultGroupName=",
                 "UninstallDisplayName=")
    bad = []
    for section, lines in _sections(_render()).items():
        for line in lines:
            if not HANGUL.search(line) or line.lstrip().startswith(";") or section == "CustomMessages":
                continue
            if section == "Setup" and line.startswith(name_keys):
                continue
            if section == "Icons" and line.startswith(f'Name: "{{group}}\\{S.APP_NAME}"') \
                    or section == "Icons" and line.startswith(f'Name: "{{autodesktop}}\\{S.APP_NAME}"'):
                continue
            bad.append(f"[{section}] {line}")
    assert not bad, bad
    assert not HANGUL.search("\n".join(_sections(_render())["Code"]))


def test_template_is_ascii_and_every_placeholder_is_filled() -> None:
    template = MI.TEMPLATE.read_bytes()
    assert template.isascii()
    names = set(re.findall(rb"@([A-Z][A-Z0-9_]*)@", template))
    assert names == {n.encode() for n in ("VERSION", "VERSION4", "APP_ID", "APP_NAME", "PUBLISHER", "ICON", "DIST",
                                          "COMPRESSION", "MUTEX", "SIGNING", "MESSAGES_FILE", "CUSTOM_MESSAGES",
                                          "STATE_DIR_NAME")}
    for kw in ({}, {"fast": True}, {"sign": True}):
        text = _render(**kw)
        assert not re.search(r"@[A-Z][A-Z0-9_]*@", text)
    # ISPP는 주석 줄의 인라인 지시어도 펼친다: 틀에는 {#AppVersion} 말고 다른 인라인 지시어가 없다
    assert set(re.findall(rb"\{#(\w*)", template)) == {b"AppVersion"}


def test_icons_tasks_and_run() -> None:
    sec = _sections(_render())
    icons = [line for line in sec["Icons"] if line.strip()]
    assert icons == [
        f'Name: "{{group}}\\{S.APP_NAME}"; Filename: "{{app}}\\evolab.exe"; WorkingDir: "{{app}}"',
        'Name: "{group}\\{cm:ReportShortcut}"; Filename: "{app}\\evolab.exe"; Parameters: "report --open"; '
        'WorkingDir: "{app}"',
        f'Name: "{{autodesktop}}\\{S.APP_NAME}"; Filename: "{{app}}\\evolab.exe"; WorkingDir: "{{app}}"; '
        'Tasks: desktopicon',
    ]
    tasks = [line for line in sec["Tasks"] if line.strip()]
    assert tasks == ['Name: "desktopicon"; Description: "{cm:TaskDesktop}"; GroupDescription: "{cm:TasksGroup}"']
    assert "unchecked" not in tasks[0]  # 바탕 화면 바로 가기는 기본 켬(D37)
    run = [line for line in sec["Run"] if line.strip()]
    assert run == ['Filename: "{app}\\evolab.exe"; Description: "{cm:RunNow}"; Flags: nowait postinstall skipifsilent']
    # .evoenv 연결은 설계가 미뤘다(0절 「범위 줄이기」): 레지스트리 섹션이 없다
    assert "Registry" not in sec


def _diff(a: str, b: str) -> tuple[list[str], list[str]]:
    removed, added = [], []
    for line in difflib.ndiff(a.split("\r\n"), b.split("\r\n")):
        if line.startswith("- "):
            removed.append(line[2:])
        elif line.startswith("+ "):
            added.append(line[2:])
    return removed, added


def test_version_override_fast_and_sign_change_only_their_lines() -> None:
    base = _render()
    v = evolab.__version__
    removed, added = _diff(base, _render(version="2.99.0"))
    assert removed == [f'#define AppVersion "{v}"', f"VersionInfoVersion={v}.0"]
    assert added == ['#define AppVersion "2.99.0"', "VersionInfoVersion=2.99.0.0"]
    removed, added = _diff(base, _render(fast=True))
    assert (removed, added) == (["Compression=lzma2/ultra64"], ["Compression=lzma2/fast"])
    removed, added = _diff(base, _render(sign=True))
    assert len(removed) == 1 and removed[0].startswith("; Code signing is off")
    assert added == ["SignTool=evolabsign", "SignedUninstaller=yes"]


def test_messages_file_override() -> None:
    text = MI.render_iss(ROOT / "dist" / "evolab", version="3.1.0", out_dir=ROOT / "build" / "installer",
                         messages_file=ROOT / "packaging" / "installer" / "Korean.isl")
    assert 'Name: "korean"; MessagesFile: "..\\..\\packaging\\installer\\Korean.isl"' in text.split("\r\n")


@pytest.mark.parametrize("bad", ["2.99", "3.1.0.0", "v3.1.0", "3.1.0rc1", ""])
def test_bad_versions_are_refused(bad: str) -> None:
    with pytest.raises(MI.InstallerError):
        _render(version=bad)
    assert MI.main(["--out", "unused.iss", "--version-override", bad]) == 1


def _fake_root(tmp_path: Path, pyproject: str, package: str) -> Path:
    (tmp_path / "src" / "evolab").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "evolab"\nversion = "{pyproject}"\n', encoding="utf-8")
    (tmp_path / "src" / "evolab" / "__init__.py").write_text(f'"""x"""\n\n__version__ = "{package}"\n', encoding="utf-8")
    return tmp_path


def test_version_comes_from_pyproject_and_must_match_the_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
    assert MI.read_version() == evolab.__version__
    assert MI.read_version(_fake_root(tmp_path / "same", "4.5.6", "4.5.6")) == "4.5.6"
    bad = _fake_root(tmp_path / "bad", "3.1.0", "3.0.0")
    with pytest.raises(MI.InstallerError, match="mismatch"):
        MI.read_version(bad)
    # 명령도 실패하고 .iss를 쓰지 않는다(덮어쓰기 버전을 줘도 pyproject와 __version__은 늘 본다)
    monkeypatch.setattr(MI, "ROOT", bad)
    out = tmp_path / "out" / "evolab.iss"
    assert MI.main(["--dist", str(tmp_path / "dist"), "--out", str(out), "--version-override", "2.99.0"]) == 1
    assert not out.exists()
    assert "mismatch" in capsys.readouterr().err


def test_command_writes_the_script_and_names(tmp_path: Path) -> None:
    out = tmp_path / "build" / "installer" / "evolab.iss"
    names_out = tmp_path / "build" / "installer" / "names.json"
    dist = tmp_path / "dist" / "evolab"
    assert MI.main(["--dist", str(dist), "--out", str(out), "--names-out", str(names_out)]) == 0
    text = out.read_bytes().decode("utf-8-sig")
    assert 'Source: "..\\..\\dist\\evolab\\*"' in text
    raw = names_out.read_bytes()
    assert raw.isascii()  # 한글은 \u 이스케이프: PowerShell 콘솔 인코딩을 거치지 않는다
    names = json.loads(raw)
    assert names == {
        "version": evolab.__version__, "app_id": FIXED_APP_ID, "app_name": S.APP_NAME,
        "report_shortcut": S.INST_REPORT_SHORTCUT, "data_dir_name": S.DATA_DIR_NAME,
        "state_dir_name": paths.APP_DIR_NAME, "mutex": winapp.MUTEX_NAME, "old_test_version": "2.99.0",
    }
    old = tmp_path / "old.iss"
    assert MI.main(["--dist", str(dist), "--out", str(old), "--version-override", "2.99.0", "--fast"]) == 0
    old_text = old.read_bytes().decode("utf-8-sig")
    assert '#define AppVersion "2.99.0"' in old_text and "Compression=lzma2/fast" in old_text


def test_code_section_reads_correctly() -> None:
    """[Code]를 컴파일할 수 없으므로(윈도우 W4) 짜임을 읽어서 본다: begin/end 짝, 함수, 묻는 상자의 기본 답, 데이터 폴더를 지우지 않음."""
    code = "\n".join(_sections(_render())["Code"])
    tokens = _code_tokens(code)
    opens = sum(tokens.count(w) for w in ("begin", "case", "try"))
    assert opens == tokens.count("end"), (opens, tokens.count("end"))
    assert code.count("(") == code.count(")")
    assert f"Uninstall\\{FIXED_APP_ID}_is1'" in code
    for name in ("NextVersionPart", "CompareVersions", "InitializeSetup", "DeleteSettingsFiles", "CurUninstallStepChanged"):
        assert re.search(rf"^(function|procedure) {name}\b", code, flags=re.M), name
    assert "function InitializeSetup(): Boolean;" in code
    assert "procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);" in code
    # 두 질문 모두 기본 단추가 「아니요」이고 조용한 모드의 답도 IDNO(설계 D39·D41)
    asks = re.findall(r"SuppressibleMsgBox\((.*?)\)\s*=\s*IDYES", code, flags=re.S)
    assert len(asks) == 2 and all("MB_YESNO or MB_DEFBUTTON2, IDNO" in a for a in asks)
    # 조용한 제거는 묻기 전에 끝내고, 모든 사용자 설치에서는 설정 질문을 하지 않는다
    body = code[code.index("procedure CurUninstallStepChanged"):]
    assert body.index("UninstallSilent()") < body.index("AskSettings")
    assert body.index("if not IsAdminInstallMode() then") < body.index("AskSettings")
    # 지우는 것은 설정 폴더 안의 알려진 파일과 logs뿐: 데이터 폴더({userdocs})나 설정 폴더 통째로는 지우지 않는다
    deletes = re.findall(r"(DelTree|DeleteFile|RemoveDir)\(([^;]*)\);", code)
    assert sorted(arg for _fn, arg in deletes) == sorted([
        "Dir + '\\settings.ini'", "Dir + '\\settings.ini.lock'", "Dir + '\\paths.ini'", "Dir + '\\logs', True, True, True",
        "Dir"])
    assert "userdocs" not in "".join(arg for _fn, arg in deletes)
    assert "DataDir" in body and "CustomMessage('DataKept')" in body


def test_version_compare_logic_matches_a_python_model() -> None:
    """[Code]의 CompareVersions를 같은 규칙의 파이썬으로 옮겨 경계값을 본다(세 칸, 숫자가 아니면 0)."""

    def next_part(s: str) -> tuple[int, str]:
        head, _, rest = s.partition(".")
        try:
            return int(head), rest
        except ValueError:
            return 0, rest

    def compare(a: str, b: str) -> int:
        for _ in range(3):
            x, a = next_part(a)
            y, b = next_part(b)
            if x != y:
                return 1 if x > y else -1
        return 0

    assert compare("3.1.0", "2.99.0") == 1 and compare("2.99.0", "3.1.0") == -1
    assert compare("3.10.0", "3.9.0") == 1 and compare("3.1.0", "3.1.0") == 0 and compare("3.1", "3.1.0") == 0
    code = "\n".join(_sections(_render())["Code"])
    assert "for I := 1 to 3 do" in code and "StrToIntDef(T, 0)" in code and "Pos('.', S)" in code


# ---------------------------------------------------------------------------
# T34 winapp(뮤텍스) — 리눅스 동작은 test_diagnostics.py에도 있다. 설치 프로그램과 같은 이름을 쓰는지 본다.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(sys.platform == "win32", reason="윈도우가 아닐 때의 동작")
def test_app_mutex_is_a_no_op_off_windows() -> None:
    assert winapp.create_app_mutex() is None
    assert winapp.MUTEX_NAME == MI.APP_MUTEX


# ---------------------------------------------------------------------------
# exe 파일 버전 정보 (D45)
# ---------------------------------------------------------------------------
def test_numeric_version() -> None:
    assert version_info.numeric_version("3.1.0") == (3, 1, 0, 0)
    assert version_info.numeric_version("2.99.0") == (2, 99, 0, 0)
    with pytest.raises(ValueError):
        version_info.numeric_version("3.70000.0")
    assert version_info.project_version() == evolab.__version__


def _pyinstaller_versioninfo(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """PyInstaller의 versioninfo 모듈. 리눅스에는 윈도우 전용 의존성(pefile, win32api)이 없어 쓰지 않는 자리만 메운다."""
    pytest.importorskip("PyInstaller")
    import PyInstaller.compat as compat

    if sys.platform != "win32":
        monkeypatch.setitem(sys.modules, "pefile", types.ModuleType("pefile"))
        monkeypatch.setattr(compat, "win32api", None, raising=False)
    monkeypatch.delitem(sys.modules, "PyInstaller.utils.win32.versioninfo", raising=False)
    module = importlib.import_module("PyInstaller.utils.win32.versioninfo")
    if sys.platform != "win32":
        # 가짜 pefile로 불러온 모듈을 남기지 않는다(sys.modules와 부모 꾸러미의 속성 둘 다)
        monkeypatch.delitem(sys.modules, "PyInstaller.utils.win32.versioninfo")
        parent = sys.modules.get("PyInstaller.utils.win32")
        if parent is not None and getattr(parent, "versioninfo", None) is module:
            delattr(parent, "versioninfo")
    return module


@pytest.mark.parametrize("exe", ["evolab", "evolab-cli"])
def test_version_info_file_is_what_pyinstaller_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exe: str) -> None:
    files = version_info.write_version_files(tmp_path)
    assert set(files) == {"evolab", "evolab-cli"}
    assert files[exe].read_bytes().startswith(b"\xef\xbb\xbf")
    vi = _pyinstaller_versioninfo(monkeypatch)
    info = vi.load_version_info_from_text_file(str(files[exe]))
    back = info
    if sys.platform == "win32":
        # 바이트 왕복은 윈도우에서만: PyInstaller가 struct 'L'(윈도우 4바이트, 리눅스 8바이트)로 쓴다
        back = vi.VSVersionInfo()
        back.fromRaw(info.toRaw())
    major, minor, patch = (int(x) for x in evolab.__version__.split("."))
    assert back.ffi.fileVersionMS == (major << 16) | minor and back.ffi.fileVersionLS == patch << 16
    assert back.ffi.productVersionMS == back.ffi.fileVersionMS
    table = back.kids[0].kids[0]
    assert table.name == "041204B0"
    found = {s.name: s.val for s in table.kids}
    assert found == {
        "CompanyName": S.INST_PUBLISHER, "FileDescription": S.APP_NAME, "FileVersion": f"{evolab.__version__}.0",
        "InternalName": exe, "OriginalFilename": f"{exe}.exe", "ProductName": S.APP_NAME,
        "ProductVersion": evolab.__version__,
    }
    assert back.kids[1].kids[0].kids == [0x0412, 1200]


def test_spec_gives_both_exes_their_version_info(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """evolab.spec을 PyInstaller 대신 가짜 Analysis·EXE로 돌려 두 EXE에 각자의 버전 정보 파일이 들어가는지 본다."""
    calls: dict[str, list[dict[str, Any]]] = {"EXE": [], "COLLECT": []}

    class Stub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args, self.kwargs = args, kwargs
            self.scripts, self.pure, self.binaries, self.datas = [], [], [], []

    def exe(*args: Any, **kwargs: Any) -> Stub:
        calls["EXE"].append(kwargs)
        return Stub(*args, **kwargs)

    fake_licenses = types.ModuleType("collect_licenses")
    fake_licenses.collect = lambda out: tmp_path / "licenses"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "collect_licenses", fake_licenses)
    real_write = version_info.write_version_files
    monkeypatch.setattr(version_info, "write_version_files", lambda out_dir, version=None: real_write(tmp_path, version))
    monkeypatch.setitem(sys.modules, "version_info", version_info)
    saved_path = list(sys.path)
    try:
        namespace = {"SPECPATH": str(PACKAGING), "Analysis": Stub, "PYZ": Stub, "EXE": exe, "COLLECT": Stub,
                     "__file__": str(PACKAGING / "evolab.spec")}
        exec(compile((PACKAGING / "evolab.spec").read_text(encoding="utf-8"), "evolab.spec", "exec"), namespace)
    finally:
        sys.path[:] = saved_path
    by_name = {kw["name"]: kw for kw in calls["EXE"]}
    assert set(by_name) == {"evolab", "evolab-cli"}
    for name, kw in by_name.items():
        path = Path(kw["version"])
        assert path.parent == tmp_path and path.is_file()
        text = path.read_text(encoding="utf-8-sig")
        assert f"StringStruct('OriginalFilename', '{name}.exe')" in text
    assert by_name["evolab"]["console"] is False and by_name["evolab-cli"]["console"] is True


# ---------------------------------------------------------------------------
# 서명 없는 PE 목록 (D46)
# ---------------------------------------------------------------------------
def _pe(magic: int, security_size: int, n_dirs: int = 16) -> bytes:
    """작은 PE 머리: DOS 머리 + "PE\\0\\0" + COFF 20바이트 + 선택 머리(데이터 디렉터리 n_dirs개)."""
    pe_offset = 0x80
    dos = bytearray(pe_offset)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, pe_offset)
    coff = struct.pack("<HHIIIHH", 0x8664, 0, 0, 0, 0, 0, 0)
    if magic == 0x10B:
        opt = bytearray(96 + 8 * n_dirs)
        struct.pack_into("<I", opt, 92, n_dirs)
        dirs_at = 96
    else:
        opt = bytearray(112 + 8 * n_dirs)
        struct.pack_into("<I", opt, 108, n_dirs)
        dirs_at = 112
    struct.pack_into("<H", opt, 0, magic)
    if n_dirs > 4:
        struct.pack_into("<II", opt, dirs_at + 8 * 4, 0x400 if security_size else 0, security_size)
    return bytes(dos) + b"PE\0\0" + coff + bytes(opt) + b"\0" * 64


def test_certificate_table_detection(tmp_path: Path) -> None:
    cases = {
        "signed64.dll": (_pe(0x20B, 0x1A00), 0x1A00),
        "unsigned64.exe": (_pe(0x20B, 0), 0),
        "signed32.pyd": (_pe(0x10B, 0x800), 0x800),
        "unsigned32.dll": (_pe(0x10B, 0), 0),
        "few_dirs.dll": (_pe(0x20B, 0, n_dirs=3), 0),
        "text.dll": (b"not a PE file at all", None),
        "short.dll": (b"MZ", None),
        "bad_pe_sig.dll": (_pe(0x20B, 0).replace(b"PE\0\0", b"XX\0\0"), None),
    }
    for name, (data, expected) in cases.items():
        (tmp_path / name).write_bytes(data)
        assert list_unsigned.certificate_table_size(tmp_path / name) == expected, name


def test_scan_and_command_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dist = tmp_path / "evolab"
    (dist / "_internal" / "PySide6" / "plugins").mkdir(parents=True)
    (dist / "evolab.exe").write_bytes(_pe(0x20B, 0))
    (dist / "evolab-cli.exe").write_bytes(_pe(0x20B, 0))
    (dist / "_internal" / "python312.dll").write_bytes(_pe(0x20B, 0x2000))
    (dist / "_internal" / "PySide6" / "Qt6Core.dll").write_bytes(_pe(0x20B, 0x2000))
    (dist / "_internal" / "PySide6" / "plugins" / "qoffscreen.dll").write_bytes(_pe(0x20B, 0))
    (dist / "_internal" / "base_library.zip").write_bytes(b"PK")
    (dist / "_internal" / "notes.dll.txt").write_bytes(_pe(0x20B, 0))
    signed, unsigned = list_unsigned.scan(dist)
    assert [p.as_posix() for p in signed] == ["_internal/PySide6/Qt6Core.dll", "_internal/python312.dll"]
    assert [p.as_posix() for p in unsigned] == ["_internal/PySide6/plugins/qoffscreen.dll", "evolab-cli.exe", "evolab.exe"]
    assert list_unsigned.main([str(dist)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("PE files: 5, signed: 2, unsigned: 3")
    assert "unsigned: _internal/PySide6/plugins/qoffscreen.dll" in out and "_internal/PySide6" in out
    assert list_unsigned.main([str(dist), "--paths"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [Path(line).name for line in lines] == ["qoffscreen.dll", "evolab-cli.exe", "evolab.exe"]
    assert all(Path(line).is_absolute() for line in lines)
    assert list_unsigned.main([str(tmp_path / "missing")]) == 2


# ---------------------------------------------------------------------------
# 휴대용 zip·SHA256SUMS·릴리스 글 (D44, W10)
# ---------------------------------------------------------------------------
def _fake_dist(root: Path) -> Path:
    dist = root / "dist" / "evolab"
    (dist / "_internal" / "PySide6").mkdir(parents=True)
    (dist / "evolab.exe").write_bytes(b"MZ gui")
    (dist / "evolab-cli.exe").write_bytes(b"MZ cli")
    (dist / "_internal" / "PySide6" / "Qt6Core.dll").write_bytes(b"MZ qt" * 100)
    return dist


def test_portable_zip_sums_and_notes(tmp_path: Path) -> None:
    dist = _fake_dist(tmp_path)
    setup = tmp_path / "Output" / "evolab-setup-3.1.0.exe"
    setup.parent.mkdir()
    setup.write_bytes(b"setup bytes")
    out = tmp_path / "release"
    files = release_files.build(dist, out, "3.1.0", setup)
    assert [p.name for p in files] == ["evolab-3.1.0-portable.zip", "evolab-setup-3.1.0.exe"]
    with zipfile.ZipFile(out / "evolab-3.1.0-portable.zip") as z:
        names = z.namelist()
        assert sorted(names) == sorted([
            "evolab-3.1.0/evolab.exe", "evolab-3.1.0/evolab-cli.exe", "evolab-3.1.0/_internal/PySide6/Qt6Core.dll",
            f"evolab-3.1.0/{S.PORTABLE_README_NAME}"])
        readme = z.getinfo(f"evolab-3.1.0/{S.PORTABLE_README_NAME}")
        assert readme.flag_bits & 0x800  # 한글 이름은 UTF-8 표시(윈도우 탐색기·Expand-Archive가 읽음)
        raw = z.read(readme)
        assert raw.startswith(b"\xef\xbb\xbf")
        assert raw[3:].decode("utf-8") == S.PORTABLE_README.format(version="3.1.0")
        assert "evolab-setup-3.1.0.exe" in raw.decode("utf-8-sig")
        assert z.read("evolab-3.1.0/_internal/PySide6/Qt6Core.dll") == b"MZ qt" * 100
    sums = (out / "SHA256SUMS.txt").read_bytes()
    assert b"\r" not in sums
    lines = sums.decode("ascii").splitlines()
    assert [line.split("  ")[1] for line in lines] == ["evolab-3.1.0-portable.zip", "evolab-setup-3.1.0.exe"]
    for line in lines:
        digest, name = line.split("  ")
        assert digest == release_files.sha256(out / name) and len(digest) == 64
    notes = (out / "release-notes.md").read_text(encoding="utf-8")
    assert "evolab://" not in notes and not notes.startswith("# ")
    assert "## SHA256" in notes and lines[0] in notes and lines[1] in notes
    page_body = S.HELP_PAGES["install"][1]
    first_section = page_body.split("\n## ")[1].split("\n")[0]
    assert f"## {first_section}" in notes
    assert (out / "release-title.txt").read_text(encoding="utf-8") == f"{S.APP_NAME} 3.1.0\n"


def test_release_files_without_installer_and_with_a_wrong_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dist = _fake_dist(tmp_path)
    files = release_files.build(dist, tmp_path / "a", "3.1.0", None)
    assert [p.name for p in files] == ["evolab-3.1.0-portable.zip"]
    assert "WARNING no setup file" in capsys.readouterr().err
    assert len((tmp_path / "a" / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines()) == 1
    wrong = tmp_path / "evolab-setup-2.99.0.exe"
    wrong.write_bytes(b"x")
    with pytest.raises(ValueError):
        release_files.build(dist, tmp_path / "b", "3.1.0", wrong)
    with pytest.raises(FileNotFoundError):
        release_files.portable_zip(tmp_path / "nothing", tmp_path / "c.zip", "3.1.0")
    assert release_files.main(["--dist", str(dist), "--out", str(tmp_path / "d"), "--version", "3.1"]) == 1


# ---------------------------------------------------------------------------
# PowerShell 스크립트 (pwsh가 있으면 구문 검사: 윈도우 CI tests job)
# ---------------------------------------------------------------------------
PS_SCRIPTS = [PACKAGING / "installer_smoke.ps1", PACKAGING / "installer" / "sign.ps1"]


@pytest.mark.parametrize("script", PS_SCRIPTS, ids=lambda p: p.name)
def test_powershell_scripts_are_utf8_bom_crlf(script: Path) -> None:
    raw = script.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # 윈도우 PowerShell 5.1도 한글 주석을 바르게 읽는다
    assert b"\n" not in raw.replace(b"\r\n", b"")
    text = raw.decode("utf-8-sig")
    assert "#Requires -Version 7.0" in text and "Set-StrictMode -Version 3.0" in text
    # 사용자 이름·한글 경로를 스크립트에 박지 않는다(이름은 names.json, 경로는 Inno 상수와 같은 윈도우 폴더 API로)
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert not HANGUL.search(code)


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh 없음(윈도우 CI tests job에서 돈다)")
@pytest.mark.parametrize("script", PS_SCRIPTS, ids=lambda p: p.name)
def test_powershell_scripts_parse(script: Path) -> None:
    # -Command 뒤의 인수는 모두 명령 글에 이어 붙으므로 경로는 명령 안에 작은따옴표로 넣는다.
    quoted = str(script).replace("'", "''")
    command = ("$e = $null; "
               f"$null = [System.Management.Automation.Language.Parser]::ParseFile('{quoted}', [ref]$null, [ref]$e); "
               "if ($e) { $e | ForEach-Object { $_.ToString() }; exit 1 }")
    result = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_smoke_script_covers_every_design_step() -> None:
    text = (PACKAGING / "installer_smoke.ps1").read_bytes().decode("utf-8-sig")
    for needle in ("/VERYSILENT", "/SUPPRESSMSGBOXES", "/CURRENTUSER", "/NORESTART", "HKCU:\\", "InstallLocation",
                   "_internal\\stale.txt", "report --open", "selftest", "Downgrade refused", "TryOpenExisting",
                   "/SECONDPHASE", "$UninstallTimeout = 120", "Start-Sleep -Seconds 1", "EVOLAB_DATA_DIR",
                   "EVOLAB_STATE_DIR", "EVOLAB_NO_TOUR", "GITHUB_ACTIONS", "step times",
                   '[ValidateSet("strict", "lenient", "skip")][string]$RunningUninstall = "strict"'):
        assert needle in text, needle
    # 이름은 names.json에서: make_installer.names()의 열쇠를 모두 쓴다
    for key in ("app_name", "report_shortcut", "data_dir_name", "state_dir_name", "mutex", "app_id", "old_test_version"):
        assert f"$n.{key}" in text, key
    # 내려 설치 확인 줄은 [Code]의 Log 글과 같아야 한다
    assert "Log('Downgrade refused.');" in _render()


def test_smoke_script_reads_shortcuts_with_the_unicode_shell_reader() -> None:
    """a96e3bd 윈도우 CI W5: WScript.Shell의 CreateShortcut은 .lnk 경로를 시스템 ANSI 코드 페이지로 바꿔, 영어 윈도우(1252)에서
    한글 이름의 바로 가기를 열지 못하고 오류 없이 빈 새 바로 가기를 돌려줬다(있는 바로 가기 셋의 대상·인수가 빈 글로 읽혀 실패 넷).
    확인 값은 유니코드 셸 자동화(Shell.Application → FolderItem.GetLink)로 읽고 대상·인수·작업 폴더를 모두 본다.
    WScript.Shell은 정보 줄(같은 파일을 다르게 읽으면 찍기)에만 남는다. pwsh가 없는 곳에서도 도는 글 검사다."""
    text = (PACKAGING / "installer_smoke.ps1").read_bytes().decode("utf-8-sig")
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))

    def function_body(name: str) -> str:
        start = code.index(f"function {name}(")
        return code[start:code.index("\n}\n", start)]

    assert "$script:shellApp = New-Object -ComObject Shell.Application" in code
    reader = function_body("Read-Shortcut")
    for needle in ("$script:shellApp.Namespace(", ".ParseName(", "$item.IsLink", "$item.GetLink",
                   "$link.Path", "$link.Arguments", "$link.WorkingDirectory", "$why = ", "Error = $why }"):
        assert needle in reader, needle
    check = function_body("Test-Shortcut")
    assert "CreateShortcut" not in check and "$lnk = Read-Shortcut $Path" in check
    for needle in ("-PathType Leaf", "if ($lnk.Error)", "Test-SamePath $lnk.Target $AppExe",
                   "$lnk.Arguments -ceq $Arguments", "Test-SamePath $lnk.WorkingDirectory $AppDir"):
        assert needle in check, needle
    # 대상·작업 폴더 비교는 예외 없이 거짓(빈 글·잘못된 경로)이나 대소문자를 무시한 같은 전체 경로
    same = function_body("Test-SamePath")
    assert "if (-not $A -or -not $B) { return $false }" in same and "OrdinalIgnoreCase" in same and "catch" in same
    # WScript.Shell의 CreateShortcut은 정보 줄 하나에만(확인 값에 쓰지 않는다)
    assert code.count("CreateShortcut(") == 1
    info = function_body("Show-WshShortcutView")
    assert "CreateShortcut(" in info and "Check " not in info and "information only" in info
    # 바로 가기 셋 모두 같은 확인을 거친다: 시작 메뉴 둘(오류 보고는 report --open), 바탕 화면 하나
    for call in ('Test-Shortcut $StartLnk "" ', 'Test-Shortcut $ReportLnk "report --open" ', 'Test-Shortcut $DeskLnk "" '):
        assert call in code, call


def test_packaging_scripts_keep_korean_text_in_strings_py() -> None:
    """test_rules.py와 같은 규칙을 이 묶음의 packaging 스크립트에도: 문서 문자열·주석 말고 한글 문자열 리터럴이 없다."""
    import ast

    files = [PACKAGING / "make_installer.py", PACKAGING / "version_info.py", PACKAGING / "list_unsigned.py",
             PACKAGING / "installer" / "release_files.py"]
    offenders = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        docstrings = {id(node.value) for node in ast.walk(tree)
                      if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                      and isinstance(node.value.value, str)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                if HANGUL.search(ast.get_source_segment(source, node) or ""):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# 윈도우 CI 설정 (W4·W5·W10·W11). PyYAML이 있을 때만(실행 의존성이 아니다).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps(workflow: dict[str, Any], job: str) -> list[dict[str, Any]]:
    return workflow["jobs"][job]["steps"]


def _step(workflow: dict[str, Any], job: str, name_part: str) -> dict[str, Any]:
    found = [s for s in _steps(workflow, job) if name_part in str(s.get("name", ""))]
    assert len(found) == 1, name_part
    return found[0]


def test_workflow_jobs(workflow: dict[str, Any]) -> None:
    jobs = workflow["jobs"]
    assert set(jobs) == {"tests", "package", "release"}
    assert "needs" not in jobs["package"] and "needs" not in jobs["tests"]
    release = jobs["release"]
    assert release["needs"] == ["tests", "package"]
    assert release["if"] == "startsWith(github.ref, 'refs/tags/v')"
    assert release["permissions"] == {"contents": "write"}
    # tests job에 EVOLAB_DATA_DIR를 두지 않는다(test_release::test_custom_data_dir가 paths.ini에 기댄다)
    assert not any(k.startswith("EVOLAB_") for k in jobs["tests"].get("env", {}))
    assert not any(k.startswith("EVOLAB_") for k in jobs["package"].get("env", {}))


def test_workflow_signing_is_skipped_without_secrets(workflow: dict[str, Any]) -> None:
    env = workflow["jobs"]["package"]["env"]
    assert env["SIGN_ENABLED"] == "${{ secrets.SIGN_CERT_BASE64 != '' }}"
    sign = _step(workflow, "package", "Sign (only when a certificate is configured)")
    assert "env.SIGN_ENABLED == 'true'" in sign["if"]
    assert sign["env"]["SIGN_CERT_BASE64"] == "${{ secrets.SIGN_CERT_BASE64 }}"
    assert "packaging/installer/sign.ps1" in sign["run"]
    # 비밀 값 자체는 서명하는 단계에만 있다
    for step in _steps(workflow, "package"):
        if "SIGN_CERT_BASE64" in str(step.get("env", {})):
            assert step.get("name") in ("Sign (only when a certificate is configured)",
                                        "Build installers (current and old 2.99.0)")
    names = [s.get("name", "") for s in _steps(workflow, "package")]
    assert names.index("Build exe") < names.index("Sign (only when a certificate is configured)") \
        < names.index("Copy exe folder to a Korean path") < names.index("Build installers (current and old 2.99.0)")


def test_workflow_installer_steps(workflow: dict[str, Any]) -> None:
    build = _step(workflow, "package", "Build installers")
    run = build["run"]
    assert ("packaging/make_installer.py --dist dist\\evolab --out build\\installer\\evolab.iss "
            "--names-out build\\installer\\names.json") in run
    assert "--version-override 2.99.0 --fast" in run
    assert "choco install innosetup" in run and "Languages\\Korean.isl" in run and "60000000" in run
    smoke = _step(workflow, "package", "Installer smoke test")
    assert smoke["if"] == "${{ !cancelled() && steps.installers.outcome == 'success' }}"
    for arg in ("./packaging/installer_smoke.ps1", "-Setup", "-OldSetup", "-Version $env:VER", "-Names", "-LogDir",
                "-RunningUninstall strict"):
        assert arg in smoke["run"], arg
    portable = _step(workflow, "package", "Portable zip and checksums")
    assert "packaging/installer/release_files.py" in portable["run"]
    uploads = {s["with"]["name"]: s for s in _steps(workflow, "package")
               if str(s.get("uses", "")).startswith("actions/upload-artifact")}
    assert {"evolab-windows", "evolab-setup", "evolab-exports", "evolab-selftest-logs"} <= set(uploads)
    assert uploads["evolab-setup"]["with"]["path"] == "build/release/"
    publish = _step(workflow, "release", "Publish the GitHub release")
    for needle in ("evolab-setup-${version}.exe", "evolab-${version}-portable.zip", "SHA256SUMS.txt", "sha256sum -c",
                   "release-notes.md", "release-title.txt", "gh release create"):
        assert needle in publish["run"], needle
    for path in ("packaging/make_installer.py", "packaging/installer_smoke.ps1", "packaging/installer/release_files.py",
                 "packaging/installer/sign.ps1", "packaging/list_unsigned.py"):
        assert (ROOT / path).is_file(), path
    # W6는 pytest 인수가 아니라 환경 변수로 SDK 출력을 넘긴다(conftest에 --mf-abi가 없으면 pytest가 인수 오류로 멈춘다)
    abi = _step(workflow, "package", "Media Foundation ABI table")
    pytest_lines = [line for line in abi["run"].splitlines() if line.startswith("uv run pytest")]
    assert "EVOLAB_MF_ABI" in abi["run"] and pytest_lines == ["uv run pytest -q tests/test_mf_abi.py"]
    assert "packaging\\ci\\mf_abi.c" in abi["run"]


def test_workflow_steps_needing_the_exe_follow_the_build(workflow: dict[str, Any]) -> None:
    """앞 단계 하나가 실패해도 exe가 있으면 배포본 점검이 끝까지 돈다(D34의 뜻): exe가 있어야 하는 단계는 build 결과를 본다."""
    for step in _steps(workflow, "package"):
        name = str(step.get("name", ""))
        if name in ("Exe file version", "Copy exe folder to a Korean path", "Unsigned PE list",
                    "Build installers (current and old 2.99.0)", "Portable zip and checksums"):
            assert "!cancelled()" in step["if"] and "steps.build.outcome == 'success'" in step["if"], name


def test_workflow_text_runs_the_yaml_checks_and_compares_the_exe_mp4() -> None:
    """PyYAML 없이도(이 저장소의 잠금 환경) 글로 확인한다: tests job이 위 윈도우 CI 설정 검사를 PyYAML을 얹어 돌리고(그렇지
    않으면 어느 CI에서도 건너뛰어짐), W7이 exe 내장 MP4의 프레임 수와 소스로 만든 같은 파일의 SHA-256을 비교하고(9.3 기준 4),
    W6의 cl이 표준 전처리기(/Zc:preprocessor)로 mf_abi.c를 빌드한다(v3.1 검토)."""
    text = WORKFLOW.read_text(encoding="utf-8")
    tests_job = text.split("\n  tests:\n", 1)[1].split("\n  package:\n", 1)[0]
    assert "uv run --with pyyaml pytest -q tests/test_installer_template.py -k workflow" in tests_job
    assert "uv run pytest -q\n" in tests_job  # 빠른 테스트 전체(W1)는 그대로
    w7 = text.split("- name: Exports with the console exe (Korean path)", 1)[1].split("\n      - name:", 1)[0]
    assert "uv run evolab export $run --format mp4 --encoder builtin --out $srcMp4" in w7
    assert "frame_count(job)" in w7 and "Get-FileHash -Algorithm SHA256 -LiteralPath $exeMp4" in w7
    assert "if ($exeHash -ne $srcHash)" in w7
    assert "cl /nologo /W3 /Zc:preprocessor packaging\\ci\\mf_abi.c" in text

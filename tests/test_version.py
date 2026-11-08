"""v3.0 0단계 기초 (v3 설계 8.1 T3·T4): 버전이 한 곳에서 같고, EVOLAB_DATA_DIR·EVOLAB_STATE_DIR가 폴더를 바꾼다."""

from __future__ import annotations

import configparser
import multiprocessing
import os
import re
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

import evolab
from evolab import paths
from evolab.app import strings as S

ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r"\d+\.\d+\.\d+")


# ---------------------------------------------------------------------------
# T3 버전
# ---------------------------------------------------------------------------
def test_pyproject_version_is_the_package_version() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = data["project"]["version"]
    assert version == evolab.__version__
    assert VERSION_RE.fullmatch(version), version


def test_uv_lock_records_the_same_evolab_version() -> None:
    """uv.lock에서 evolab 줄만 버전을 따라간다(설계 D2: 새 실행 의존성 없음, 패키지 목록은 그대로)."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    ours = [p for p in lock["package"] if p["name"] == "evolab"]
    assert len(ours) == 1 and ours[0]["version"] == evolab.__version__


def test_version_is_3_0_or_later() -> None:
    """v3.0부터 0.0.1이 아니다: 옛 실행을 열면 「다른 빌드」 안내가 맞게 뜨고, 오류 보고가 빌드를 가른다(설계 D3)."""
    major, _minor, _patch = (int(x) for x in evolab.__version__.split("."))
    assert major >= 3


# ---------------------------------------------------------------------------
# T4 데이터·상태 폴더 덮어쓰기
# ---------------------------------------------------------------------------
def _old_user_state_dir() -> Path:
    """v2.1까지의 user_state_dir 계산(덮어쓰기가 없을 때 지금과 같은지 보려고 그대로 옮김)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_STATE_HOME")
        root = Path(base) if base else Path.home() / ".local" / "state"
    return root / paths.APP_DIR_NAME


@pytest.fixture()
def no_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(paths.DATA_DIR_ENV, raising=False)
    monkeypatch.delenv(paths.STATE_DIR_ENV, raising=False)


def test_without_overrides_folders_are_unchanged(no_overrides: None, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    assert paths.user_state_dir() == _old_user_state_dir()
    assert paths.log_dir() == _old_user_state_dir() / "logs"
    # 데이터 폴더: 설정(paths.ini)이 없으면 문서\진화실험실, 있으면 그 폴더(상태 폴더를 임시로 돌려 실제 설정을 건드리지 않음)
    monkeypatch.setattr(paths, "user_state_dir", lambda: tmp_path / "state")
    assert paths.data_dir() == paths.documents_dir() / S.DATA_DIR_NAME
    assert paths.default_records_dir() == paths.documents_dir() / S.DATA_DIR_NAME / "records"
    custom = tmp_path / "내 데이터"
    paths.set_custom_data_dir(custom)
    assert paths.data_dir() == custom and paths.runs_dir() == custom / "runs"
    assert paths.default_records_dir() == paths.documents_dir() / S.DATA_DIR_NAME / "records"


def test_overrides_move_every_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data, state = tmp_path / "데이터 폴더", tmp_path / "상태"
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(data))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(state))
    assert paths.user_state_dir() == state
    assert paths.log_dir() == state / "logs"
    assert paths.data_dir() == data
    assert paths.runs_dir() == data / "runs"
    assert paths.user_env_dir() == data / "environments"
    assert paths.records_dir() == data / "records"
    # 예전 기록을 찾는 기본 위치도 문서 폴더가 아니라 덮어쓴 폴더(실제 문서 폴더를 읽지 않음)
    assert paths.default_records_dir() == data / "records"
    # 설정 파일(paths.ini)은 덮어쓴 상태 폴더에 쓰고, 데이터 폴더는 환경 변수가 이긴다
    paths.set_custom_data_dir(tmp_path / "다른 곳")
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(state / paths.PATHS_FILE, encoding="utf-8")
    assert Path(cfg["paths"]["data_dir"]) == tmp_path / "다른 곳"
    assert paths.custom_data_dir() == tmp_path / "다른 곳"
    assert paths.data_dir() == data


def test_each_override_is_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(paths.DATA_DIR_ENV, raising=False)
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "s"))
    assert paths.user_state_dir() == tmp_path / "s"
    assert paths.data_dir() == paths.documents_dir() / S.DATA_DIR_NAME  # 덮어쓴 상태 폴더에 paths.ini 없음
    monkeypatch.delenv(paths.STATE_DIR_ENV)
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "d"))
    assert paths.user_state_dir() == _old_user_state_dir()
    assert paths.data_dir() == tmp_path / "d"


def test_empty_or_relative_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.STATE_DIR_ENV, "   ")
    assert paths.user_state_dir() == _old_user_state_dir()  # 빈 값은 없는 것과 같다
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(paths.DATA_DIR_ENV, "rel")
    assert paths.data_dir() == tmp_path / "rel" and paths.data_dir().is_absolute()


def test_install_examples_goes_to_overridden_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    copied = paths.install_examples()
    assert copied and all(p.parent == tmp_path / "data" / "environments" for p in copied)


def test_spawn_child_inherits_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """학습 프로세스·워커처럼 spawn으로 띄운 자식도 같은 폴더를 쓴다(환경 변수를 물려받음)."""
    data, state = tmp_path / "d", tmp_path / "s"
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(data))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(state))
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as ex:
        got = [ex.submit(fn).result(timeout=120) for fn in (paths.data_dir, paths.user_state_dir, paths.log_dir)]
    assert got == [data, state, state / "logs"]

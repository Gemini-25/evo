"""명령줄 내보내기와 자체 점검 「내보내기」 (v3 설계 D58·D59, 4.19, 8.1 T44·T45: cli.py export·video-probe, selftest.py).

- `evolab export <실행> --format mp4|gif|html --out …` 종료 코드 0과 「만들었습니다」 한 줄, 확장자가 형식과 다르면 2(파일을
  만들지 않음), 고를 수 없는 값도 2, 실행 폴더가 아니면 1, 쓸 수 없는 인코더(`--encoder mf`, MF 없는 PC)면 1과 까닭.
- 다시 시뮬레이션은 job_from_run(실험실 「1세대 대표와 지금 챔피언」과 같은 규칙)이고 MP4 프레임 수 = frames.frame_count.
- 창 없이: QT_QPA_PLATFORM이 없어도 별도 프로세스로 돌아간다(윈도우 밖에서는 스스로 offscreen).
- `evolab video-probe`는 리눅스에서 「builtin」과 0.
- T45: 자체 점검 「내보내기」가 오프스크린에서 통과하고, 임시 데이터·상태 폴더만 쓴다.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from evolab import paths
from evolab.app import strings as S
from evolab.app.export import job as J
from evolab.app.export import mp4, phone
from evolab.app.export.frames import frame_count
from evolab.cli import main

DATA = Path(__file__).parent / "data"


@pytest.fixture()
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """v1.0 실행 사본(유전 알고리즘 착륙선 4세대, 원본은 고치지 않는다)과 임시 데이터·상태 폴더."""
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    out = tmp_path / "실행 기록" / "v10"
    shutil.copytree(DATA / "v10_run_ga" / "run", out)
    return out


def test_export_each_format_exit_0(qtbot: Any, run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "내보내기"
    for fmt, extra in (("mp4", ["--encoder", "builtin", "--size", "small"]), ("gif", []), ("html", []),
                       ("mp4", ["--what", "versus", "--size", "small", "--camera", "follow", "--generation", "2"])):
        out = folder / f"챔피언 {fmt} {len(extra)}{J.SUFFIXES[fmt]}"
        assert main(["export", str(run), "--format", fmt, *extra, "--out", str(out)]) == 0
        line = capsys.readouterr().out.strip().splitlines()[-1]
        assert out.is_file() and line.startswith(S.CLI_EXPORT_DONE.split("{")[0]) and str(out) in line
        job = J.job_from_run(run, fmt, generation=2 if "--generation" in extra else None,
                             what="versus" if "versus" in extra else "champion",
                             size="small" if "small" in extra else "medium",
                             camera="follow" if "follow" in extra else "fixed", encoder="auto")
        if fmt == "mp4":
            info = mp4.read_info(out)
            assert info["frames"] == frame_count(job) and (info["width"], info["height"]) == (640, 360)
            assert S.EXP_ENCODER_BUILTIN in line or S.EXP_ENCODER_MF.split("(")[0] in line
        elif fmt == "html":
            dec = phone.decode_payload(phone.read_phone_html(out))
            assert [e["label"] for e in dec["entries"]] == [it.label for it in job.items]
            assert S.EXP_ENCODER_HTML in line
        else:
            assert S.EXP_ENCODER_GIF in line
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        f"챔피언 {f} {n}{J.SUFFIXES[f]}" for f, n in (("mp4", 4), ("gif", 0), ("html", 0), ("mp4", 8)))


def test_export_argument_errors_exit_2(run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "a.gif"
    assert main(["export", str(run), "--format", "mp4", "--out", str(out)]) == 2  # 확장자가 형식과 다르다
    assert S.EXP_BAD_SUFFIX.format(name="a.gif") in capsys.readouterr().err and not out.exists()
    assert main(["export", str(run), "--format", "mp4", "--out", str(tmp_path / "a")]) == 2
    for bad in (["--format", "avi"], ["--format", "gif", "--size", "huge"], ["--format", "gif", "--generation", "0"],
                ["--format", "gif"]):  # 마지막: --out 없음
        with pytest.raises(SystemExit) as err:
            main(["export", str(run), *bad, *([] if bad == ["--format", "gif"] else ["--out", str(out)])])
        assert err.value.code == 2
    assert not out.exists()


def test_export_failures_exit_1(run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "a.gif"
    assert main(["export", str(tmp_path / "없는 실행"), "--format", "gif", "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert err.startswith(S.EXP_FAILED.split("{")[0]) and "Traceback" not in err and not out.exists()
    if mp4.encoder_status().kind != "mf":  # MF가 없는 PC(리눅스, 윈도우 CI 러너)
        out = tmp_path / "a.mp4"
        assert main(["export", str(run), "--format", "mp4", "--encoder", "mf", "--out", str(out)]) == 1
        reason = mp4.encoder_status().detail
        assert S.EXP_INFO_BUILTIN_TIP.format(reason=reason) in capsys.readouterr().err
        assert not out.exists() and not mp4.temp_path(out).exists()


@pytest.mark.skipif(sys.platform == "win32", reason="윈도우는 MF 시험 결과가 PC마다 다르다(W8에서 기록)")
def test_video_probe_says_builtin_off_windows(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["video-probe"]) == 0
    line = capsys.readouterr().out.strip()
    status = mp4.encoder_status()
    assert line == S.CLI_VIDEO_PROBE_LINE.format(encoder="builtin", mf=status.detail) and status.kind == "builtin"


def test_export_runs_without_a_window_in_its_own_process(run: Path, tmp_path: Path) -> None:
    """설치된 명령처럼 별도 프로세스로(QT_QPA_PLATFORM 없이): 창 없이 끝나고 종료 코드 0."""
    out = tmp_path / "프로세스" / "챔피언.html"
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    done = subprocess.run([sys.executable, "-m", "evolab", "export", str(run), "--format", "html", "--out", str(out)],
                          env=env, capture_output=True, timeout=300)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert out.is_file() and str(out) in done.stdout.decode("utf-8")


@pytest.mark.parametrize("child_probe", [False, True], ids=["probe-as-on-this-os", "probe-child-as-on-windows"])
def test_selftest_export_check_passes_with_temporary_folders(qtbot: Any, tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch, child_probe: bool) -> None:
    """T45: 자체 점검 「내보내기」가 통과하고(내장 MP4·폰 재생 파일·MF 정보), 사용자의 데이터·상태 폴더에 남는 것이 없다.
    child_probe: 윈도우(MF 켬)처럼 인코더 시험을 작은 자식으로 해도(자식의 faults.log가 임시 상태 폴더에 생겨야 한다:
    시험이 임시 폴더를 되돌린 뒤에 돌던 v3.1 검토 결함)."""
    import tempfile

    from evolab import selftest
    from evolab.app.export import runner

    real_data, real_state, tmp_root = tmp_path / "문서 데이터", tmp_path / "설정", tmp_path / "임시"
    tmp_root.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(real_data))
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(real_state))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_root))
    if child_probe:
        monkeypatch.setattr(runner, "probe_needs_child", lambda: True)
    assert (S.SELFTEST_CHECK_EXPORT, selftest._check_export) in selftest.selftest_checks(2, check_gui=False)
    detail = selftest._check_export()
    first = detail.splitlines()[0]
    assert first.startswith(S.SELFTEST_EXPORT_DETAIL.split("{")[0]) and "child first message" in detail
    want = frame_count(J.job_from_replays([(_lander_head(), J.NOW_COLOR, "a")], "mp4", size="small",
                                          camera="fixed", title="", meta={}))
    assert S.SELFTEST_EXPORT_PART.split("{")[0] in first and f"{want}" in first
    assert not real_data.exists() or not any(real_data.rglob("*"))
    assert not real_state.exists() or not any(real_state.rglob("*"))
    if sys.platform != "win32":  # 윈도우는 자식의 파일 핸들이 늦게 풀려 임시 폴더가 남을 수 있다
        assert not any(tmp_root.iterdir())


def _lander_head() -> Any:
    import numpy as np

    from evolab import selftest
    from evolab.engine import seeds
    from evolab.engine.replay import replay
    from evolab.policy.genome import layout_for
    from evolab.spec.io import load_env

    spec = load_env(paths.builtin_env_dir() / "lander.evoenv")
    layout = layout_for(spec)
    return selftest._first_seconds(replay(spec, layout, np.zeros(layout.dim), seeds.display_trial()),
                                   selftest.EXPORT_SECONDS)

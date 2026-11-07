"""재현 묶음과 통계 CSV 내보내기 (명세: 저장, 기록, 내보내기).

재현 묶음 = run.json + 환경 버전 사본 + 세대 번호가 붙은 변경 기록(events.jsonl) + 원래 통계(비교용).
다시 돌리면 처음 환경·설정으로 새 실행을 만들고, 기록된 세대 경계마다 같은 설정 변경과 환경 교체를 같은 순서로
적용하며 같은 세대 수까지 학습한다. 같은 PC·같은 빌드에서는 통계가 비트 단위로 같고(시간 열 제외), 다른 PC에서는
통계적으로 재현된다(명세: 재현 범위).
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evolab.app import strings as S
from evolab.runs import store as _st
from evolab.runs.store import RunStore, RunStoreError

BUNDLE_FORMAT = "evolab-repro"
BUNDLE_VERSION = 1


def _version_ids(info: dict[str, Any]) -> list[str]:
    """run.json에 기록된 환경 버전 id(기록 순서). 기록이 망가졌으면(id가 없는 항목 등) RunStoreError(한국어)."""
    versions = info.get("env_versions", [])
    if not isinstance(versions, list):
        raise RunStoreError(S.RUNS_ENV_VERSION_MISSING.format(version="?"))
    out: list[str] = []
    for v in versions:
        vid = v.get("id") if isinstance(v, dict) else None
        if not isinstance(vid, str):
            raise RunStoreError(S.RUNS_ENV_VERSION_MISSING.format(version="?" if vid is None else vid))
        out.append(vid)
    return out


def export_bundle(run_dir: Path, out: Path) -> Path:
    """실행 폴더에서 재현 묶음(zip)을 만든다. 실패하면 임시 파일을 지우고 원래 out은 그대로 둔다.

    환경 사본이 없거나 바뀌었거나 기록이 망가졌으면 RunStoreError(한국어), out을 다른 프로그램이 잡고 있어 바꿀 수
    없으면 RunStoreError(RUNS_FILE_BUSY).
    """
    store = RunStore.open(run_dir)
    info = store.read_run_info()
    stats = store.read_stats()
    version_ids = _version_ids(info)
    manifest = {"format": BUNDLE_FORMAT, "version": BUNDLE_VERSION, "run": store.root.name,
                "created": dt.datetime.now().isoformat(timespec="seconds"), "generations": len(stats)}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("bundle.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            z.writestr("run.json", json.dumps(info, ensure_ascii=False, indent=2))
            for vid in version_ids:
                name = f"envs/{vid}.evoenv"
                if name not in z.namelist():
                    z.writestr(name, store.env_text(vid))
            z.writestr("events.jsonl", "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in store.read_events()))
            z.writestr("stats.csv", _csv_text(stats))
            z.writestr("README.txt", S.BUNDLE_README)
        try:
            _st._replace(tmp, out)  # 윈도우에서 잠깐 잡혀 있으면 조금씩 다시 시도한다(저장소와 같은 규칙)
        except PermissionError as exc:
            raise RunStoreError(S.RUNS_FILE_BUSY.format(path=out)) from exc
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return out


def read_bundle(path: Path) -> dict[str, Any]:
    """묶음 내용: run.json, 환경 버전 사본 {id: 글}, 사건 목록, 원래 통계."""
    try:
        with zipfile.ZipFile(path) as z:
            manifest = json.loads(z.read("bundle.json"))
            if manifest.get("format") != BUNDLE_FORMAT:
                raise RunStoreError(S.BUNDLE_NOT_BUNDLE.format(path=path))
            info = json.loads(z.read("run.json"))
            envs = {Path(n).stem: z.read(n).decode("utf-8") for n in z.namelist() if n.startswith("envs/")}
            events = [json.loads(line) for line in z.read("events.jsonl").decode("utf-8").splitlines() if line.strip()]
            stats = list(csv.DictReader(io.StringIO(z.read("stats.csv").decode("utf-8-sig"))))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise RunStoreError(S.BUNDLE_READ_FAILED.format(path=path, reason=exc)) from None
    return {"manifest": manifest, "info": info, "envs": envs, "events": events, "stats": stats}


def changes_by_generation(events: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """세대 경계마다 적용할 변경(설정 변경, 환경 교체)을 기록 순서대로."""
    out: dict[int, list[dict[str, Any]]] = {}
    for e in events:
        if e.get("kind") in ("config_change", "env_change"):
            out.setdefault(int(e.get("generation", 0)), []).append(e)
    return out


def reproduce(bundle: Path, runs_dir: Path, workers: int | None = None,
              on_generation: Callable[[dict[str, Any]], None] | None = None) -> Path:
    """묶음을 다시 돌려 새 실행 폴더를 만든다. 새 실행 폴더 경로."""
    from evolab.engine.pool import WorkerPool, default_workers
    from evolab.engine.runsetup import create_run
    from evolab.engine.trainer import TrainConfig, Trainer
    from evolab.spec.io import compile_env

    data = read_bundle(bundle)
    info, envs = data["info"], data["envs"]
    versions = info.get("env_versions", [])
    if not versions or str(versions[0]["id"]) not in envs:
        raise RunStoreError(S.BUNDLE_READ_FAILED.format(path=bundle, reason="envs"))
    first = envs[str(versions[0]["id"])]
    config = TrainConfig.from_dict(info.get("config", {}))
    config.max_generations = None
    config.max_evaluations = None
    config.max_time = None
    config.target_score = None
    config.stagnation_generations = None
    target = int(data["manifest"].get("generations") or len(data["stats"]))
    changes = changes_by_generation(data["events"])
    n = workers if workers is not None else default_workers()
    store = create_run(runs_dir, first, compile_env(first), config, n,
                       extra={"reproduced_from": {"run": data["manifest"].get("run"), "bundle": str(bundle)},
                              "memo": S.BUNDLE_MEMO.format(run=data["manifest"].get("run"))})
    with WorkerPool(n) as pool:
        trainer = Trainer(first, config, pool, store)
        while True:
            g = trainer.state.generation
            for e in changes.get(g, []):
                if e["kind"] == "config_change":
                    trainer.apply_config(dict(e.get("changes") or {}))
                    store.append_event("config_change", g, changes=e.get("changes") or {})
                else:
                    text = envs[str(e["version"])]
                    trainer.replace_env(text)
                    store.add_env_version(text, g)
                    store.append_event("env_change", g, version=e["version"])
            if g >= target:
                break
            row = trainer.step()
            if on_generation is not None:
                on_generation(row)
        trainer.checkpoint()
        store.append_event("stop", trainer.state.generation, reason="reproduced")
    return store.root


def compare_stats(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> int | None:
    """두 통계가 같으면 None, 다르면 처음 다른 세대 번호. 시간 열은 비교하지 않는다."""
    from evolab.engine.trainer import TIME_COLUMNS

    skip = set(TIME_COLUMNS)
    for k, (ra, rb) in enumerate(zip(a, b, strict=False)):
        da = {key: str(v) for key, v in ra.items() if key not in skip}
        db = {key: str(v) for key, v in rb.items() if key not in skip}
        if da != db:
            return k + 1
    if len(a) != len(b):
        return min(len(a), len(b)) + 1
    return None


def export_csv(run_dir: Path, out: Path) -> Path:
    """통계 CSV(세대별 통계 전체). 한국어 윈도우 엑셀에서 바로 열리게 BOM 붙은 UTF-8."""
    rows = RunStore.open(run_dir).read_stats()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_csv_text(rows), encoding="utf-8-sig", newline="")
    return out


def _csv_text(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\r\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return buf.getvalue()

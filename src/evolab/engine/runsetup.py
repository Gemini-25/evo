"""새 실행 폴더 만들기. 명령줄 학습(evolab train)과 실험실 화면의 학습 프로세스가 같이 쓴다."""

from __future__ import annotations

import datetime as dt
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from evolab import __version__
from evolab.app import strings as S
from evolab.engine import seeds
from evolab.engine.pool import physical_cores
from evolab.engine.trainer import TrainConfig
from evolab.policy.genome import layout_for
from evolab.runs.store import RunStore
from evolab.spec.compiled import EnvSpec


def platform_info(workers: int) -> dict[str, Any]:
    import cma
    import pymunk

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pymunk": pymunk.version,
        "cma": cma.__version__,
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "physical_cores": physical_cores(),
        "workers": workers,
        "frozen": bool(getattr(sys, "frozen", False)),
    }


def run_name(spec: EnvSpec, config: TrainConfig, now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now()
    return f"{now:%Y-%m-%d_%H%M}_{spec.file.name}_{S.METHOD_NAMES.get(config.method, config.method)}"


def create_run(base_dir: Path, env_text: str, spec: EnvSpec, config: TrainConfig, workers: int,
               extra: dict[str, Any] | None = None) -> RunStore:
    """실행 폴더를 만들고 run.json, 환경 사본(v000), 시작 사건을 남긴다."""
    layout = layout_for(spec)
    info: dict[str, Any] = {
        "program": "evolab", "version": __version__, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "env_name": spec.file.name, "method": config.method, "config": config.to_dict(),
        "seed_scheme": seeds.SEED_SCHEME_VERSION, "layout": layout.to_dict(),
        "layout_signature": layout.signature(), "platform": platform_info(workers),
    }
    info.update(extra or {})
    store = RunStore.create(base_dir, run_name(spec, config), info)
    store.add_env_version(env_text, 0)
    store.append_event("start", 0, workers=workers)
    return store

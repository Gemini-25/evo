"""내보내기 작업 (v3 설계 2.6, 7.3 「app/export/job.py (0')」, D48~D58).

화면 프로세스(실험실·경주 창의 내보내기 대화 상자)와 명령줄(`evolab export`)이 ExportJob을 만들고, 내보내기 자식
프로세스(`worker.export_main`, 묶음 F)가 pickle로 받아 그리고 쓴다(MP4·GIF·폰 재생 파일). 재생(Replay)은 화면이 이미
다시 시뮬레이션한 것을 그대로 담는다: 자식은 시뮬레이션하지 않는다. 명령줄은 자기 프로세스에서 job_from_run이 실행
폴더로 다시 시뮬레이션해 같은 작업을 만든다(전시 시드, 실험실 「1세대 대표와 지금 챔피언」과 같은 규칙).

- 크기·fps(설계 2.6, D54): MP4는 30fps, 「작게」 640×360·「보통」 1280×720, 경주 칸은 그 크기를 lane_grid 열·줄로 나눠
  짝수로 내린 칸의 격자. GIF는 지금 값 그대로(겹쳐서 640×400, 경주 칸 320×200, 20fps). 폰 재생 파일은 GIF와 같은
  크기·fps를 쓴다: 재생기는 궤적 프레임을 그대로 쓰고, 이 크기는 스크립트가 없을 때 보이는 끝 장면 그림(GIF 마지막
  프레임과 같은 그림)에만 쓴다.
- 경주 칸 격자는 ExportJob에 따로 두지 않는다(7.3 필드 그대로): 늘 lane_grid(칸 수)이고 lane_layout(job)이 돌려준다.
- 이 모듈은 Qt를 맨 위에서 불러오지 않는다: 자식이 QApplication을 만들기 전에 작업을 풀 수 있고, 폰 파일·작성기
  테스트가 가볍다. QColor는 job_from_replays가 "#rrggbb"로 바꿀 때만 불러온다.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from evolab import __version__
from evolab.app import strings as S
from evolab.engine.replay import Replay

if TYPE_CHECKING:
    from PySide6.QtGui import QColor

    from evolab.runs.store import RunStore
    from evolab.spec.compiled import EnvSpec

ViewBox = tuple[float, float, float, float]

FORMATS: Final = ("mp4", "gif", "html")
LAYOUTS: Final = ("overlay", "lanes")
CAMERAS: Final = ("fixed", "follow")
ENCODERS: Final = ("auto", "mf", "builtin")
WHATS: Final = ("champion", "versus")
"""명령줄 --what: 챔피언 하나, 1세대 대표 + 챔피언."""
SUFFIXES: Final[dict[str, str]] = {"mp4": ".mp4", "gif": ".gif", "html": ".html"}
"""형식 → 파일 확장자(대화 상자의 기본 이름, 명령줄의 확장자 검사: EXP_BAD_SUFFIX)."""

MP4_SIZES: Final[dict[str, tuple[int, int]]] = {"small": (640, 360), "medium": (1280, 720)}
"""MP4 크기(설계 D54): 대화 상자 「작게」·「보통」, 명령줄 --size. 기본은 「보통」."""
DEFAULT_SIZE_NAME: Final = "medium"
MP4_FPS: Final = 30
GIF_SIZE: Final = (640, 400)
"""겹친 GIF(지금 값 = export.video.DEFAULT_SIZE). 폰 재생 파일의 끝 장면 그림도 이 크기."""
GIF_PANE: Final = (320, 200)
"""경주 GIF 칸 하나(지금 값 = export.video.RACE_PANE)."""
GIF_FPS: Final = 20
"""GIF(지금 값 = export.video.DEFAULT_FPS)."""
MAX_ITEMS: Final = 6
"""칸(재생) 수 상한: 경주 칸 상한(race.model.MAX_LANES), 폰 재생 파일 칸 상한(설계 2.8)."""

OLD_COLOR: Final = "#508cff"
"""1세대 대표 색(실험실 COLOR_OLD = QColor(80, 140, 255)): 명령줄 「1세대 vs 지금」."""
NOW_COLOR: Final = "#ff7828"
"""지금 챔피언 색(실험실 COLOR_NOW = QColor(255, 120, 40))."""

META_RUN: Final = "run"
"""meta 키: 실행 폴더 이름."""
META_GENERATION: Final = "generation"
"""meta 키: 세대(정수 글)."""
META_ENV: Final = "env"
"""meta 키: 환경 이름(환경 파일의 name)."""
META_ENV_HASH: Final = "env_hash"
"""meta 키: 환경 해시 앞 16자(챔피언 파일 env_version과 같은 값)."""
META_VERSION: Final = "version"
"""meta 키: 프로그램 버전(evolab.__version__). job_from_*가 늘 넣는다."""
META_DATE: Final = "date"
"""meta 키: 작업을 만든 날(YYYY-MM-DD, 그 PC 시각). job_from_*가 늘 넣는다(폰 파일 정보 줄의 「내보낸 날」)."""
META_KEYS: Final = (META_RUN, META_GENERATION, META_ENV, META_ENV_HASH, META_VERSION, META_DATE)

_COLOR_RE = re.compile(r"#[0-9a-fA-F]{6}")


@dataclass(frozen=True)
class ExportItem:
    """내보낼 재생 하나(겹쳐 보기의 한 개체, 경주의 한 칸)."""

    replay: Replay
    """engine.replay.Replay: 화면(또는 명령줄)이 이미 다시 시뮬레이션한 것. pickle로 자식에게 넘긴다."""
    color: str
    """"#rrggbb"(소문자로 맞춤)."""
    label: str
    """범례·칸 이름·폰 파일 칸 이름."""
    view: ViewBox | None = None
    """그 칸 환경의 기본 보기(editor.view, spec.editor_meta.view_box). 고정 카메라가 먼저 쓴다(설계 D50·4.15)."""

    def __post_init__(self) -> None:
        if not isinstance(self.color, str) or not _COLOR_RE.fullmatch(self.color):
            raise ValueError(f"color must be '#rrggbb', got {self.color!r}")
        object.__setattr__(self, "color", self.color.lower())
        if not isinstance(self.label, str):
            raise ValueError(f"label must be str, got {type(self.label).__name__}")
        object.__setattr__(self, "view", _check_view(self.view))


@dataclass(frozen=True)
class ExportJob:
    """내보내기 작업 하나(설계 2.6). 화면 → 자식 프로세스는 pickle로 넘긴다."""

    layout: Literal["overlay", "lanes"]
    """overlay: 한 세계에 겹쳐(챔피언, 1세대 vs 지금, 경주 「겹쳐서」), lanes: 경주 칸(칸마다 자기 세계·카메라)."""
    fmt: Literal["mp4", "gif", "html"]
    items: tuple[ExportItem, ...]
    size: tuple[int, int]
    """전체 그림 (너비, 높이), 둘 다 짝수. lanes면 칸 크기 × lane_grid 격자(lane_layout)."""
    fps: int
    camera: Literal["fixed", "follow"]
    title: str
    """화면 위 글(VIDEO_HUD의 title). 경주 칸이면 첫 칸 글상자 첫 줄."""
    encoder: Literal["auto", "mf", "builtin"] = "auto"
    meta: Mapping[str, str] = field(default_factory=dict)
    """META_* 키(실행 이름, 세대, 환경 이름, 환경 해시 16자, 프로그램 버전, 만든 날): 폰 파일 정보 줄. 늘 평범한 dict로
    바꿔 담는다(pickle 되게: MappingProxyType 같은 것은 pickle 되지 않는다)."""
    ranks: tuple[int, ...] | None = None
    """경주 순위(칸 순서, 공동이면 같은 수). 모든 칸 환경이 같을 때만(race.model.ranks 규칙), 아니면 None."""

    def __post_init__(self) -> None:
        _check_choice(self.layout, LAYOUTS, "layout")
        _check_choice(self.fmt, FORMATS, "fmt")
        _check_choice(self.camera, CAMERAS, "camera")
        _check_choice(self.encoder, ENCODERS, "encoder")
        items = tuple(self.items)
        if not 1 <= len(items) <= MAX_ITEMS:
            raise ValueError(f"an export job needs 1..{MAX_ITEMS} items, got {len(items)}")
        if not all(isinstance(it, ExportItem) for it in items):
            raise ValueError("items must be ExportItem")
        object.__setattr__(self, "items", items)
        size = tuple(self.size)
        if (len(size) != 2 or not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in size)
                or not all(int(v) >= 2 and int(v) % 2 == 0 for v in size)):
            raise ValueError(f"size must be two positive even integers, got {self.size!r}")
        object.__setattr__(self, "size", (int(size[0]), int(size[1])))
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, np.integer)) or int(self.fps) < 1:
            raise ValueError(f"fps must be a positive integer, got {self.fps!r}")
        object.__setattr__(self, "fps", int(self.fps))
        if not isinstance(self.title, str):
            raise ValueError(f"title must be str, got {type(self.title).__name__}")
        if not all(isinstance(k, str) for k in self.meta):
            raise ValueError("meta keys must be str")
        object.__setattr__(self, "meta", {k: str(v) for k, v in self.meta.items()})
        if self.ranks is not None:
            ranks = tuple(int(r) for r in self.ranks)
            if len(ranks) != len(items) or any(r < 1 for r in ranks):
                raise ValueError(f"ranks must be one positive rank per item, got {self.ranks!r}")
            object.__setattr__(self, "ranks", ranks)

    @property
    def duration(self) -> float:
        """가장 긴 재생의 길이(초, 궤적 마지막 프레임 시각)."""
        return max(float(it.replay.trajectory.duration) for it in self.items)


# ---------------------------------------------------------------------------
# 작업 만들기
# ---------------------------------------------------------------------------
def job_from_replays(items: Sequence[tuple[Replay, QColor | str, str]], fmt: str, *, size: str, camera: str,
                     title: str, meta: Mapping[str, str], view: ViewBox | None = None) -> ExportJob:
    """실험실의 지금 재생(타임라인 세대 대표, 1세대 vs 지금, 고속 모드 챔피언) 또는 경주 「겹쳐서」를 겹쳐 보기 작업으로.

    items = (재생, 색(QColor 또는 "#rrggbb"), 이름): 실험실 `_last_video[0]`와 같은 모양. size = "small"·"medium"(MP4만 씀,
    GIF·폰 파일이면 검사만 하고 지금 GIF 크기). view = 그 재생 환경의 기본 보기(실험실 `_replay_view`): 겹쳐 보기는 한
    세계라 모든 칸에 같은 값을 넣는다(7.3 서명에 더한 키워드, 기본 None). meta에는 프로그램 버전·만든 날을 더한다.
    """
    _check_choice(fmt, FORMATS, "fmt")
    _check_choice(size, MP4_SIZES, "size")
    _check_choice(camera, CAMERAS, "camera")
    rows = list(items)
    if not rows:
        raise ValueError(S.VIDEO_NO_FRAMES)
    out = tuple(ExportItem(rep, _hex(color), str(label), view) for rep, color, label in rows)
    width, height, fps = _overlay_size(fmt, size)
    return ExportJob(layout="overlay", fmt=fmt, items=out, size=(width, height), fps=fps, camera=camera,
                     title=str(title), meta=make_meta(meta))


def job_from_race(lanes: Sequence[Any], cols: int, fmt: str, *, size: str, camera: str, title: str) -> ExportJob:
    """나란히 경주(칸 격자)를 칸 작업으로.

    lanes = 경주 창의 칸(`RaceView.lanes`: `.replay`가 Replay 또는 None, `.entry`가 race.model.RaceEntry) 또는
    (재생, 색, 이름) 묶음. 재생이 없는 칸은 뺀다(지금 경주 GIF와 같음). 칸 객체면 색·이름·기본 보기를 그 칸의 RaceEntry에서
    가져오고, 모든 칸 환경이 같으면 순위(race.model.ranks)와 환경 이름·해시를 meta에 넣는다.

    cols는 경주 창이 쓰는 lane_grid(남은 칸 수)의 열 수여야 한다(ExportJob은 격자를 따로 담지 않고 lane_layout이 늘
    lane_grid로 다시 정한다). 다르면 ValueError. 크기: MP4는 MP4_SIZES[size]를 열·줄로 나눠 짝수로 내린 칸 × 격자,
    GIF·폰 파일은 칸 320×200 × 격자, fps는 MP4 30·GIF 20.
    """
    from evolab.app.race.model import lane_grid, ranks

    _check_choice(fmt, FORMATS, "fmt")
    _check_choice(size, MP4_SIZES, "size")
    _check_choice(camera, CAMERAS, "camera")
    kept: list[tuple[Replay, str, str, ViewBox | None, Any]] = []
    for lane in lanes:
        if isinstance(lane, tuple):
            rep, color, label = lane
            kept.append((rep, _hex(color), str(label), None, None))
            continue
        rep = getattr(lane, "replay", None)
        if rep is None:
            continue
        entry = lane.entry
        kept.append((rep, _hex(entry.color), str(entry.label), _entry_view(entry), entry))
    if not kept:
        raise ValueError(S.VIDEO_NO_FRAMES)
    if len(kept) > MAX_ITEMS:
        raise ValueError(f"a race export takes at most {MAX_ITEMS} lanes, got {len(kept)}")
    rows, want = lane_grid(len(kept))
    if isinstance(cols, bool) or int(cols) != want:
        raise ValueError(f"cols must be lane_grid({len(kept)}) columns = {want}, got {cols!r}")
    if fmt == "mp4":
        base_w, base_h = MP4_SIZES[size]
        pane = (_even_floor(base_w / want), _even_floor(base_h / rows))
        fps = MP4_FPS
    else:
        pane, fps = GIF_PANE, GIF_FPS
    items = tuple(ExportItem(rep, color, label, view) for rep, color, label, view, _e in kept)
    entries = [e for *_rest, e in kept]
    rank_tuple: tuple[int, ...] | None = None
    known: dict[str, str] = {}
    if all(e is not None for e in entries):
        got = ranks(entries, [float(rep.result.score) for rep, *_rest in kept])
        if got is not None:  # 모든 칸 환경이 같다
            rank_tuple = tuple(r for r, _tie in got)
            known[META_ENV_HASH] = str(entries[0].env_hash)
            name = _env_name(entries[0].env_text)
            if name:
                known[META_ENV] = name
        run_dirs = {e.run for e in entries}
        if len(run_dirs) == 1 and None not in run_dirs:
            known[META_RUN] = Path(next(iter(run_dirs))).name
    return ExportJob(layout="lanes", fmt=fmt, items=items, size=(pane[0] * want, pane[1] * rows),
                     fps=fps, camera=camera, title=str(title), meta=make_meta(known),
                     ranks=rank_tuple)


def job_from_run(run_dir: Path, fmt: str, *, generation: int | None, what: str, size: str, camera: str,
                 encoder: str) -> ExportJob:
    """명령줄 `evolab export`: 실행 폴더에서 그 세대(None이면 마지막)까지의 마지막 챔피언을 전시 시드로 다시 시뮬레이션해
    겹쳐 보기 작업으로(설계 4.19, D58). 이 프로세스에서 시뮬레이션한다.

    - 챔피언: generation 이하 세대의 마지막 챔피언 파일(판정 명령과 같은 고르기). 챔피언 파일이 없으면(검증 전에 멈춘
      실행) 그 세대 이하의 마지막 세대 대표로(실험실 「1세대 vs 지금」·경주와 같은 대체), 이름은 LAB_LABEL_REP.
    - what="versus"면 1세대 대표(OLD_COLOR)를 앞에, 챔피언(NOW_COLOR)을 뒤에 둔다. 제목은 이름을 " vs "로 이은 것.
    - 환경: 그 세대를 계산할 때 쓴 환경 버전(race.model.env_text_at), 파일에 적힌 env_version이 다르면 그 해시의 버전
      (판정·경주 규칙). 유전자 배치는 run.json의 layout(실험실이 연 실행과 같음), 없으면 그 환경의 layout_for.
    - 기본 보기: 챔피언 환경의 editor.view(실험실이 마지막 재생 환경의 보기를 쓰는 것과 같음).

    인수가 틀리면 ValueError, 실행 폴더를 읽을 수 없으면 RunStoreError·OSError·EnvFileError(모두 한국어 메시지),
    세대가 하나도 없으면 RunStoreError(RACE_NO_GENERATIONS).
    """
    from evolab.app.race.model import compile_cached
    from evolab.engine import seeds
    from evolab.engine.replay import replay
    from evolab.policy.genome import GenomeLayout, layout_for
    from evolab.runs.store import RunStore, RunStoreError
    from evolab.spec.editor_meta import view_box
    from evolab.spec.io import env_hash

    _check_choice(fmt, FORMATS, "fmt")
    _check_choice(what, WHATS, "what")
    _check_choice(size, MP4_SIZES, "size")
    _check_choice(camera, CAMERAS, "camera")
    _check_choice(encoder, ENCODERS, "encoder")
    if generation is not None and (isinstance(generation, bool) or not isinstance(generation, (int, np.integer))
                                   or int(generation) < 1):
        raise ValueError(f"generation must be None or a positive integer, got {generation!r}")
    run_dir = Path(run_dir)
    store = RunStore.open(run_dir)
    info = store.read_run_info()
    limit = None if generation is None else int(generation)

    def last_at_most(kind: str) -> int | None:
        gens = [g for g in store.champion_generations(kind) if limit is None or g <= limit]
        return gens[-1] if gens else None

    gen = last_at_most("champion")
    if gen is not None:
        data = store.load_champion(gen, "champion")
        label = S.LAB_LABEL_CHAMPION.format(gen=gen)
    else:
        gen = last_at_most("representative")
        if gen is None:
            raise RunStoreError(S.RACE_NO_GENERATIONS)
        data = store.load_champion(gen, "representative")
        label = S.LAB_LABEL_REP.format(gen=gen)
    stored = info.get("layout")
    run_layout = GenomeLayout.from_dict(stored) if isinstance(stored, Mapping) and stored else None
    trial = seeds.display_trial()

    picks: list[tuple[EnvSpec, np.ndarray, str, str]] = []
    if what == "versus":
        first = store.load_champion(1, "representative")
        picks.append((compile_cached(_env_text_for(store, 1, first)), np.asarray(first["genome"], dtype=np.float64),
                      OLD_COLOR, S.LAB_LABEL_REP.format(gen=1)))
    spec = compile_cached(_env_text_for(store, max(1, gen), data))
    picks.append((spec, np.asarray(data["genome"], dtype=np.float64), NOW_COLOR, label))
    view = view_box(spec.file)
    items = tuple(ExportItem(replay(s, run_layout or layout_for(s), genome, trial), color, name, view)
                  for s, genome, color, name in picks)
    width, height, fps = _overlay_size(fmt, size)
    meta = make_meta({META_RUN: run_dir.name, META_GENERATION: str(gen), META_ENV: str(spec.file.name),
                      META_ENV_HASH: env_hash(spec.file)[:16]})
    return ExportJob(layout="overlay", fmt=fmt, items=items, size=(width, height), fps=fps,
                     camera=camera, title=" vs ".join(it.label for it in items), encoder=encoder,
                     meta=meta)


# ---------------------------------------------------------------------------
# 묶음 F·P가 쓰는 도우미
# ---------------------------------------------------------------------------
def lane_layout(job: ExportJob) -> tuple[int, int, int, int]:
    """(줄, 열, 칸 너비, 칸 높이). lanes면 lane_grid(칸 수)와 size를 그 격자로 나눈 칸(job_from_race가 만든 작업이면 나머지
    없이 나뉜다), overlay면 (1, 1, 전체 너비, 전체 높이). 칸 k는 격자 (k // 열, k % 열) 자리."""
    if job.layout != "lanes":
        return 1, 1, job.size[0], job.size[1]
    from evolab.app.race.model import lane_grid

    rows, cols = lane_grid(len(job.items))
    return rows, cols, job.size[0] // cols, job.size[1] // rows


def make_meta(known: Mapping[str, Any] | None = None) -> dict[str, str]:
    """META_* 사전: known(값은 글로 바꿈)에 프로그램 버전과 오늘 날짜를 더한다(이미 있으면 그대로)."""
    out = {str(k): str(v) for k, v in (known or {}).items()}
    out.setdefault(META_VERSION, __version__)
    out.setdefault(META_DATE, dt.date.today().isoformat())
    return out


def format_for_path(path: Path | str) -> str | None:
    """파일 이름의 확장자(대소문자 무시)에 맞는 형식, 모르는 확장자면 None(명령줄·대화 상자의 EXP_BAD_SUFFIX)."""
    suffix = Path(path).suffix.lower()
    return next((fmt for fmt, s in SUFFIXES.items() if s == suffix), None)


# ---------------------------------------------------------------------------
def _check_choice(value: Any, allowed: Sequence[str] | Mapping[str, Any], name: str) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {tuple(allowed)}, got {value!r}")


def _check_view(view: Any) -> ViewBox | None:
    if view is None:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in view)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"view must be (x0, y0, x1, y1), got {view!r}") from exc
    if not all(np.isfinite((x0, y0, x1, y1))) or x1 <= x0 or y1 <= y0:
        raise ValueError(f"view must be a finite box with x1 > x0 and y1 > y0, got {view!r}")
    return (x0, y0, x1, y1)


def _hex(color: Any) -> str:
    """QColor·Qt 색 이름·"#rrggbb" → "#rrggbb"(소문자). 이미 "#rrggbb" 글이면 Qt를 부르지 않는다."""
    if isinstance(color, str) and _COLOR_RE.fullmatch(color):
        return color.lower()
    from PySide6.QtGui import QColor

    c = QColor(color)
    if not c.isValid():
        raise ValueError(f"invalid color {color!r}")
    return c.name().lower()


def _even_floor(value: float) -> int:
    return max(2, int(value) // 2 * 2)


def _overlay_size(fmt: str, size: str) -> tuple[int, int, int]:
    if fmt == "mp4":
        return (*MP4_SIZES[size], MP4_FPS)
    return (*GIF_SIZE, GIF_FPS)


def _entry_view(entry: Any) -> ViewBox | None:
    """경주 칸 환경의 기본 보기(race.view.entry_view와 같은 값: 경주 준비가 이미 컴파일한 글이라 다시 컴파일하지 않음)."""
    from evolab.app.race.model import compile_cached
    from evolab.spec.editor_meta import view_box
    from evolab.spec.io import EnvFileError

    try:
        return view_box(compile_cached(entry.env_text).file)
    except (EnvFileError, ValueError):
        return None


def _env_name(env_text: str) -> str:
    from evolab.app.race.model import compile_cached
    from evolab.spec.io import EnvFileError

    try:
        return str(compile_cached(env_text).file.name)
    except (EnvFileError, ValueError):
        return ""


def _env_text_for(store: RunStore, generation: int, data: Mapping[str, Any]) -> str:
    """그 세대를 계산할 때 쓴 환경 글. 챔피언·세대 대표 파일의 env_version이 그 글의 해시와 다르면 run.json의 버전 중
    같은 해시의 글(판정 명령·경주 칸과 같은 규칙). 같은 해시가 없으면 세대 규칙대로."""
    from evolab.app.race.model import env_hash_of, env_text_at
    from evolab.runs.store import RunStoreError
    from evolab.spec.io import EnvFileError

    text = env_text_at(store, generation)
    raw = data.get("env_version")
    want = str(np.asarray(raw)[()]) if raw is not None else ""
    if not want or env_hash_of(text) == want:
        return text
    seen: set[str] = set()
    for v in store.read_run_info().get("env_versions") or []:
        vid = str(v.get("id")) if isinstance(v, dict) else ""
        if not vid or vid in seen:
            continue
        seen.add(vid)
        try:
            other = store.env_text(vid)
            if env_hash_of(other) == want:
                return other
        except (RunStoreError, OSError, EnvFileError, ValueError):
            continue
    return text


__all__ = [
    "CAMERAS", "DEFAULT_SIZE_NAME", "ENCODERS", "FORMATS", "GIF_FPS", "GIF_PANE", "GIF_SIZE", "LAYOUTS", "MAX_ITEMS",
    "META_DATE", "META_ENV", "META_ENV_HASH", "META_GENERATION", "META_KEYS", "META_RUN", "META_VERSION", "MP4_FPS",
    "MP4_SIZES", "NOW_COLOR", "OLD_COLOR", "SUFFIXES", "WHATS", "ExportItem", "ExportJob", "ViewBox", "format_for_path",
    "job_from_race", "job_from_replays", "job_from_run", "lane_layout", "make_meta",
]

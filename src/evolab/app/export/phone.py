"""폰 재생 파일 (v3 설계 1.7 D56·D57, 2.8, 4.18, 7.3 「app/export/phone.py (P)」).

내보내기 작업(ExportJob)을 HTML 파일 하나로 쓴다: 폰 브라우저가 인터넷 없이 여는 캔버스 재생기 + 자료 JSON + 스크립트가
돌지 않는 미리 보기(메신저·파일 앱)를 위한 끝 장면 그림. 앱 없이, 바깥 요청 없이(CSP default-src 'none') 열린다.

- 틀은 assets/phone_player.html(ASCII만, 한글 없음). 재생기 문구는 strings.PHONE_TEXT를 자료 JSON "text"로 넣고, 제목·정보
  줄·noscript 안내·바닥 글은 이 모듈이 HTML로 이스케이프해 틀의 @@자리@@에 넣는다(스크립트가 없어도 보이게). 자리는 한 번에
  바꾼다(넣은 글 안의 @@...@@는 다시 바꾸지 않는다).
- 자료(형식 PHONE_FORMAT = 1, 설계 2.8): 궤적은 위치 1mm·각도 1e-4rad 정수(각도는 [-π, π)로 감음, 재생기는 짧은 쪽으로
  잇는다)를 Int32 리틀 엔디언 base64로, 출력은 0..100 정수(양수 쪽만: 관전 캔버스 불꽃과 같은 규칙)를 같은 방식으로.
  DEFLATE는 쓰지 않는다(D57). JSON 안의 "<"는 \\u003c로 써서 </script>로 끊기지 않게 한다.
- v2 움직이는 세계 물체(BodyGeom.motion)는 궤적에 없으므로 세계마다 일정한 간격(그 세계를 쓰는 칸들의 프레임 간격 중 가장
  짧은 것)으로 0초부터 파일 전체에서 가장 긴 재생 + 꼬리(TAIL)까지(= 재생기의 전체 길이: 경주 칸의 짧은 칸도 경주 창처럼
  끝까지 움직인다) pose_at을 미리 계산해 넣는다. 표본 시각이 칸 프레임 시각과 같아 재생기가 표본 사이를 선형으로 이어도
  개체와 맞물린다(움직임 규칙을 JS로 두 벌 두지 않는다: v2 D7, 심사).
- v2.1 없는 물체는 Replay.agent·geometry에 이미 빠져 있어 칸의 물체 목록이 그대로 맞다(불꽃도 있는 물체의 추진기만).
- 배치: overlay(한 세계 = 첫 칸 재생의 세계: 관전 캔버스 겹쳐 보기·GIF와 같음, 칸 색·범례), lanes(경주: 칸마다 자기 세계와
  카메라, 같은 세계는 하나로 합쳐 담음: 세계 JSON이 그리는 것(경계·기본 보기·정적 물체·영역·움직이는 물체의 모양과 움직임)이
  모두 같을 때만).
- 끝 장면 그림: 화면(자식 프로세스)이 관전 캔버스로 그린 마지막 프레임(QImage)을 64색 PNG로. 그림을 주지 않으면 이 모듈이
  자료로 그린 SVG(Qt 없이, 같은 세계·궤적·마지막 자세).
- 상한(설계 2.8): 칸 PHONE_MAX_ENTRIES, 칸마다 물체 PHONE_MAX_BODIES, 칸마다 프레임 PHONE_MAX_FRAMES 간격(0초 프레임은
  따로라 3,601프레임까지). 3,600 간격은 dt 1/60초·2스텝마다 기록(내장 환경)에서 120초(내장 환경 가운데 가장 긴 글라이더의
  최대 시간)에 맞춘 값이다. 에디터는 최대 시간 3,600초까지, dt는 더 작게도 받으므로 그런 환경의 긴 재생은 넘을 수 있다.
  넘으면 PhoneTooBig(PHONE_TOO_BIG, 프레임 수와 상한을 알림).

이 모듈은 맨 위에서 Qt를 불러오지 않는다(QColor·QImage는 쓰는 함수 안에서만).
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from evolab import __version__
from evolab.app import strings as S
from evolab.app.export.job import (
    GIF_SIZE,
    MAX_ITEMS,
    META_DATE,
    META_ENV,
    META_RUN,
    META_VERSION,
    ExportItem,
    ExportJob,
)
from evolab.world.physics2d.motion import pose_at

if TYPE_CHECKING:
    from PySide6.QtGui import QImage

    from evolab.world.physics2d.episode import BodyGeom, ShapeGeom, WorldGeometry

PHONE_FORMAT: Final = 1
"""자료 형식 번호(JSON "v"). 재생기는 1이 아니면 text.newer를 보이고 멈춘다. 칸을 더할 때는 그대로, 뜻이 바뀔 때만 올린다."""
PHONE_MAX_ENTRIES: Final = 6
"""칸(재생) 수 상한 = 경주 칸 상한 = job.MAX_ITEMS."""
PHONE_MAX_BODIES: Final = 32
"""칸 하나의 움직이는 물체 수 상한(세계마다 움직이는 세계 물체 수 상한도 같다)."""
PHONE_MAX_FRAMES: Final = 3600
"""칸 하나의 프레임 간격 수 상한. 0초 프레임이 따로 있어 프레임 수는 3,601까지. dt 1/60초·2스텝마다 기록이면 120초(내장
환경 가운데 가장 긴 글라이더의 최대 시간)에 맞춘 값이다. 환경의 최대 시간 상한(3,600초)이 아니다: 에디터가 받는 더 긴 최대
시간이나 더 작은 dt의 재생은 넘을 수 있고, 그러면 PhoneTooBig으로 알린다(칸마다 셈, 자료 전체 크기가 아님)."""

POS_UNIT: Final = 0.001
"""위치 정수 한 칸(m): 1mm."""
ANGLE_UNIT: Final = 1e-4
"""각도 정수 한 칸(rad)."""
OUT_SCALE: Final = 100
"""출력 0..1 → 0..100 정수."""
TAIL: Final = 0.5
"""가장 긴 재생이 끝난 뒤 더 보이는 시간(초): 겹친 GIF와 같다(export.video._iter_frames의 +0.5)."""
FIT_MARGIN: Final = 0.08
"""「전체 보기」 상자의 양쪽 여유(비율): 고정 카메라(설계 4.15)와 같다."""
FIT_MIN: Final = 8.0
"""「전체 보기」 상자 가로·세로 최소(m): 고정 카메라와 같다."""
POSTER_COLORS: Final = 64
"""끝 장면 PNG 색 수(설계 D56)."""
TERRAIN_DEPTH: Final = 50.0
"""지형 아래 채우기를 세계 아래쪽 경계보다 이만큼(m) 더 내린다(관전 캔버스와 같다)."""
TEMPLATE_PATH: Final = Path(__file__).with_name("assets") / "phone_player.html"
"""재생기 틀(ASCII만). PyInstaller 배포본에서는 _MEIPASS/evolab/app/export/assets/ 아래(evolab.spec datas)."""
PLACEHOLDERS: Final = ("TITLE", "INFO", "NOSCRIPT", "POSTER", "FOOTER", "DATA")
"""틀의 @@이름@@ 자리. TITLE은 <title>과 머리글 두 곳, 나머지는 정확히 한 곳(template_text가 확인)."""

_INT32 = (-(2**31), 2**31 - 1)
_ALPHA_TRAIL = 210
"""지나온 궤적 선의 불투명도(관전 캔버스 ALPHA_TRAIL)."""

COLORS: Final[dict[str, str]] = {
    "bg": "#0b0e14",
    "terrain_fill": "#2a2f3a",
    "terrain_line": "#8a93a6",
    "pad": "#3ddc84",
    "region": "rgba(61,220,132,0.157)",
    "hazard": "rgba(255,99,99,0.176)",
    "zone": "rgba(255,255,255,0.071)",
    "bounds": "rgba(255,255,255,0.157)",
    "world_body": "#81a1c1",
    "flame": "#ff9f43",
    "hud": "#e5e9f0",
    "hud_box": "rgba(11,14,20,0.745)",
    "hud_edge": "rgba(255,255,255,0.157)",
}
"""관전 캔버스(play.window·lab.canvas의 COLOR_*)와 같은 색. 재생기와 SVG 끝 장면이 같은 표를 쓴다(자료 JSON "colors")."""

_PLACEHOLDER_RE = re.compile(r"@@([A-Z]+)@@")
_DATA_RE = re.compile(r'<script id="evolab-data" type="application/json">(.*?)</script>', re.DOTALL)


class PhoneTooBig(ValueError):
    """재생이 상한(칸·물체·프레임)을 넘어 폰 재생 파일로 만들지 않는다. 메시지는 PHONE_TOO_BIG(한국어)."""


# ---------------------------------------------------------------------------
# 자료
# ---------------------------------------------------------------------------
def build_payload(job: ExportJob) -> dict[str, Any]:
    """작업 → 자료 사전(설계 2.8 JSON). 상한을 넘으면 PhoneTooBig, 궤적 모양이 물체 수와 다르면 ValueError."""
    items = list(job.items)
    _check_limits(items)
    lanes = job.layout == "lanes"
    # 세계: overlay는 첫 칸의 세계 하나, lanes는 칸마다(같은 세계는 하나로)
    world_keys: list[str] = []
    world_src: list[tuple[WorldGeometry, tuple[float, float, float, float] | None]] = []
    world_of: list[int] = []
    for k, it in enumerate(items):
        if not lanes and k > 0:
            world_of.append(0)
            continue
        geom = it.replay.geometry
        key = _world_key(geom, it.view)
        if key in world_keys:
            world_of.append(world_keys.index(key))
            continue
        world_keys.append(key)
        world_src.append((geom, it.view))
        world_of.append(len(world_src) - 1)
    ranks = list(job.ranks) if job.ranks is not None else [None] * len(items)
    entries = [_entry_json(it, w, r) for it, w, r in zip(items, world_of, ranks, strict=True)]
    # 움직이는 세계 물체는 파일 전체 길이(재생기의 total = 가장 긴 재생 + TAIL)까지: 경주 창처럼 짧은 칸의 장애물도 계속 움직임
    horizon = max(e["last"] for e in entries) + TAIL
    worlds = []
    for w, (geom, view) in enumerate(world_src):
        step = min(e["step"] for e in entries if e["world"] == w)
        worlds.append(_world_json(geom, view, step, horizon))
    meta = dict(job.meta)
    return {
        "v": PHONE_FORMAT,
        "app": meta.get(META_VERSION, __version__),
        "layout": job.layout,
        "title": _title(job),
        "lines": _info_lines(meta),
        "text": dict(S.PHONE_TEXT),
        "unit": [POS_UNIT, ANGLE_UNIT],
        "oscale": OUT_SCALE,
        "tail": TAIL,
        "fit": [FIT_MARGIN, FIT_MIN],
        "colors": dict(COLORS),
        "worlds": worlds,
        "entries": entries,
    }


def decode_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """자료 사전의 사본에서 base64 정수를 실수 배열로 푼다(테스트 왕복, 자체 점검).

    칸 "poses" → (F, B, 3) float64(x, y m, 각도 rad ∈ [-π, π)), "out" → (F, C) float64(0..1), 세계 움직이는 물체
    "poses" → (M, 3) float64. 다른 칸은 그대로(JSON 값)."""
    out = json.loads(json.dumps(payload))
    pu, au = (float(v) for v in out.get("unit", (POS_UNIT, ANGLE_UNIT)))
    oscale = float(out.get("oscale", OUT_SCALE))
    for e in out["entries"]:
        f, b, c = int(e["frames"]), int(e["nb"]), int(e["nc"])
        q = _ints(e["poses"], f * b * 3).reshape(f, b, 3).astype(np.float64)
        q[..., :2] *= pu
        q[..., 2] *= au
        e["poses"] = q
        e["out"] = _ints(e["out"], f * c).reshape(f, c).astype(np.float64) / oscale
    for w in out["worlds"]:
        m = int(w["mframes"])
        for mv in w["movers"]:
            q = _ints(mv["poses"], m * 3).reshape(m, 3).astype(np.float64)
            q[:, :2] *= pu
            q[:, 2] *= au
            mv["poses"] = q
    return out


def payload_from_html(text: str) -> dict[str, Any]:
    """폰 재생 파일 글에서 자료 사전을 꺼낸다(JSON 그대로, base64는 풀지 않음). 없으면 ValueError."""
    m = _DATA_RE.search(text)
    if m is None:
        raise ValueError("no evolab-data script in this file")
    data = json.loads(m.group(1))
    if not isinstance(data, dict):
        raise ValueError("evolab-data is not an object")
    return data


def read_phone_html(path: Path | str) -> dict[str, Any]:
    """파일에서 자료 사전을 꺼낸다(payload_from_html). 자체 점검의 왕복 확인용."""
    return payload_from_html(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def phone_html(job: ExportJob, poster: QImage | None = None) -> str:
    """작업 → 폰 재생 파일 글(UTF-8로 쓸 것). poster가 있으면 64색 PNG 끝 장면, 없으면 자료로 그린 SVG 끝 장면."""
    payload = build_payload(job)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    poster_uri = poster_png_uri(poster) if poster is not None else svg_poster_uri(payload)
    text = payload["text"]
    values = {
        "TITLE": html.escape(payload["title"]),
        "INFO": html.escape(" · ".join(payload["lines"])),
        "NOSCRIPT": html.escape(str(text.get("noscript", ""))),
        "POSTER": poster_uri,
        "FOOTER": html.escape(str(text.get("footer", "")).replace("{version}", str(payload["app"]))),
        "DATA": data,
    }
    return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template_text())


def write_phone_html(path: Path, job: ExportJob, poster: QImage | None) -> Path:
    """폰 재생 파일을 path에 쓴다(<이름>.tmp에 쓴 뒤 이름을 바꿈, 실패하면 임시 파일을 지우고 원래 파일은 그대로). path를
    돌려준다. 상한을 넘으면 PhoneTooBig(파일을 건드리지 않음)."""
    path = Path(path)
    text = phone_html(job, poster)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(text.encode("utf-8"))
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


@lru_cache(maxsize=1)
def template_text() -> str:
    """재생기 틀 글(ASCII). 자리가 PLACEHOLDERS 말고는 없고 맞는 수만큼 있는지 확인한다(아니면 ValueError)."""
    text = TEMPLATE_PATH.read_text(encoding="ascii")
    found = _PLACEHOLDER_RE.findall(text)
    for name in {*found, *PLACEHOLDERS}:
        want = 2 if name == "TITLE" else 1 if name in PLACEHOLDERS else 0
        if found.count(name) != want:
            raise ValueError(f"phone player template must hold @@{name}@@ {want} time(s), found {found.count(name)}")
    return text


def poster_png_uri(poster: QImage) -> str:
    """QImage(마지막 프레임) → 64색 PNG data URI(Pillow, 이미 의존성). 디더링 없이(작고 선명하게)."""
    from PIL import Image
    from PySide6.QtGui import QImage

    rgb = poster.convertToFormat(QImage.Format.Format_RGB888)
    w, h, stride = rgb.width(), rgb.height(), rgb.bytesPerLine()
    if w <= 0 or h <= 0:
        raise ValueError("empty poster image")
    im = Image.frombuffer("RGB", (w, h), bytes(rgb.constBits())[: stride * h], "raw", "RGB", stride, 1)
    pal = im.quantize(colors=POSTER_COLORS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    buf = io.BytesIO()
    pal.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def svg_poster_uri(payload: Mapping[str, Any]) -> str:
    """자료로 그린 끝 장면 SVG의 data URI(그림을 받지 못했을 때). base64라 글 안에 바깥 주소 모양이 생기지 않는다."""
    return "data:image/svg+xml;base64," + base64.b64encode(svg_poster(payload).encode("utf-8")).decode("ascii")


def svg_poster(payload: Mapping[str, Any], size: tuple[int, int] = GIF_SIZE) -> str:
    """끝 장면 SVG(Qt 없이): 세계(영역·지형·정적 물체·패드), 움직이는 세계 물체(마지막 시각), 지나온 궤적, 마지막 자세.
    카메라는 재생기의 「전체 보기」와 같은 상자, lanes면 칸마다(가로 3열까지)."""
    dec = decode_payload(payload)
    col = dec["colors"]
    entries, worlds = dec["entries"], dec["worlds"]
    lanes = dec["layout"] == "lanes" and len(entries) > 1
    groups = [[k] for k in range(len(entries))] if lanes else [list(range(len(entries)))]
    cols = min(len(groups), 3)
    rows = math.ceil(len(groups) / cols)
    width, height = int(size[0]), int(size[1])
    pw, ph = width / cols, height / rows
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="{col["bg"]}"/>']
    fit = dec.get("fit", (FIT_MARGIN, FIT_MIN))
    t_end = max(e["last"] for e in entries) + float(dec.get("tail", TAIL))  # 재생기의 끝 장면 시각(움직이는 물체)
    for g, idxs in enumerate(groups):
        group = [entries[i] for i in idxs]
        world = worlds[group[0]["world"]]
        x0, y0, x1, y1 = fit_box([e["box"] for e in group], world["box"], world["bounds"], *fit)
        out.append(f'<svg x="{_n((g % cols) * pw)}" y="{_n((g // cols) * ph)}" width="{_n(pw)}" height="{_n(ph)}" '
                   f'viewBox="{_n(x0)} {_n(-y1)} {_n(x1 - x0)} {_n(y1 - y0)}" preserveAspectRatio="xMidYMid meet">')
        out.extend(_svg_world(world, col, t_end))
        for e in group:
            out.extend(_svg_entry(e, col))
        out.append("</svg>")
    out.append("</svg>")
    return "".join(out)


def fit_box(boxes: Sequence[Sequence[float] | None], world_box: Sequence[float] | None,
            bounds: Sequence[float], margin: float = FIT_MARGIN, min_size: float = FIT_MIN,
            ) -> tuple[float, float, float, float]:
    """「전체 보기」 상자(재생기 fitBox와 같은 규칙): 칸 궤적 상자들과 세계 상자(패드·목표 점, 기본 보기)를 합치고, 하나도
    없으면 세계 경계. 양쪽에 margin 비율 여유, 가로·세로가 min_size보다 작으면 가운데를 두고 넓힌다."""
    got = [b for b in [*boxes, world_box] if b is not None]
    if got:
        x0 = min(float(b[0]) for b in got)
        y0 = min(float(b[1]) for b in got)
        x1 = max(float(b[2]) for b in got)
        y1 = max(float(b[3]) for b in got)
    else:
        x0, y0, x1, y1 = (float(v) for v in bounds)
    w, h = x1 - x0, y1 - y0
    x0, x1, y0, y1 = x0 - w * margin, x1 + w * margin, y0 - h * margin, y1 + h * margin
    if x1 - x0 < min_size:
        c = (x0 + x1) / 2.0
        x0, x1 = c - min_size / 2.0, c + min_size / 2.0
    if y1 - y0 < min_size:
        c = (y0 + y1) / 2.0
        y0, y1 = c - min_size / 2.0, c + min_size / 2.0
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# 안쪽
# ---------------------------------------------------------------------------
def _check_limits(items: Sequence[ExportItem]) -> None:
    if not items:
        raise ValueError(S.VIDEO_NO_FRAMES)
    if len(items) > min(PHONE_MAX_ENTRIES, MAX_ITEMS):
        raise PhoneTooBig(S.PHONE_TOO_BIG.format(frames=max(it.replay.trajectory.frames for it in items),
                                                 limit=PHONE_MAX_FRAMES + 1))
    for it in items:
        frames = int(it.replay.trajectory.frames)
        bodies = max(len(it.replay.agent), sum(1 for b in it.replay.geometry.bodies if _is_mover(b)))
        if frames < 1:
            raise ValueError(S.VIDEO_NO_FRAMES)
        if frames > PHONE_MAX_FRAMES + 1 or bodies > PHONE_MAX_BODIES:
            # 물체가 많으면 상한 물체 수로 환산한 프레임 수(같은 자료 크기)로 알린다
            equiv = frames if bodies <= PHONE_MAX_BODIES else math.ceil(frames * bodies / PHONE_MAX_BODIES)
            raise PhoneTooBig(S.PHONE_TOO_BIG.format(frames=equiv, limit=PHONE_MAX_FRAMES + 1))


def _is_mover(body: BodyGeom) -> bool:
    return not body.moving and body.motion is not None


def _role(body: BodyGeom) -> str:
    """관전 캔버스 _role과 같다: leg 표시가 있으면 leg, 에이전트 물체면 agent, 아니면 world(색 COLOR_WORLD_BODY)."""
    if "leg" in body.tags:
        return "leg"
    return "agent" if body.owner == "agent" else "world"


def _entry_json(item: ExportItem, world: int, rank: int | None) -> dict[str, Any]:
    rep = item.replay
    tr = rep.trajectory
    poses = np.asarray(tr.poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[2] != 3 or poses.shape[1] != len(rep.agent):
        raise ValueError(f"trajectory has {poses.shape[1:]} poses for {len(rep.agent)} moving bodies")
    frames, nb = int(poses.shape[0]), int(poses.shape[1])
    poses = np.nan_to_num(poses, nan=0.0, posinf=0.0, neginf=0.0)  # 관전 캔버스처럼 터진 프레임은 0
    outs = np.asarray(tr.outputs, dtype=np.float64).reshape(frames, -1)
    nc = int(outs.shape[1])
    q = np.empty((frames, nb, 3), dtype=np.float64)
    q[..., :2] = np.clip(np.rint(poses[..., :2] / POS_UNIT), *_INT32)
    q[..., 2] = np.rint(_wrap(poses[..., 2]) / ANGLE_UNIT)
    oq = np.rint(np.clip(np.nan_to_num(outs, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0) * OUT_SCALE)
    radii = [_radius(b.shapes) for b in rep.agent]
    box = None
    if nb:
        r = max(radii)
        lo = poses[..., :2].reshape(-1, 2).min(axis=0) - r
        hi = poses[..., :2].reshape(-1, 2).max(axis=0) + r
        box = [_n(lo[0]), _n(lo[1]), _n(hi[0]), _n(hi[1])]
    ids = [b.id for b in rep.agent]
    channels = list(rep.channels)
    thrusters = []
    for t in rep.geometry.thrusters:
        if t.body in ids and t.channel in channels and channels.index(t.channel) < nc:
            thrusters.append({"b": ids.index(t.body), "at": [_n(t.at[0]), _n(t.at[1])], "dir": _n(t.dir, 5),
                              "ch": channels.index(t.channel), "s": _n(min(1.0, t.max_force / 200.0 + 0.3), 4)})
    leg, edge = _shades(item.color)
    res = rep.result
    return {
        "label": item.label,
        "color": item.color,
        "leg": leg,
        "edge": edge,
        "world": world,
        "bodies": [[_shape_json(s) for s in b.shapes] for b in rep.agent],
        "roles": [_role(b) for b in rep.agent],
        "r0": _n(radii[0]) if radii else 0,
        "thrusters": thrusters,
        "frames": frames,
        "nb": nb,
        "nc": nc,
        "step": float(tr.every * tr.dt),
        "last": float(tr.duration),
        "poses": _b64(q),
        "out": _b64(oq),
        "box": box,
        "score": _finite(res.score),
        "end": str(res.end_label),
        "outcome": str(res.outcome),
        "time": _finite(res.time),
        "rank": None if rank is None else int(rank),
    }


def _world_key(geom: WorldGeometry, view: Sequence[float] | None) -> str:
    """lanes에서 같은 세계를 하나로 합치는 열쇠: 세계 JSON이 그리는 것 모두(경계, 기본 보기, 정적 물체, 영역(목표 점·따라가기
    카메라 대상도 여기서 나옴), 움직이는 물체마다 모양과 움직임). 표본(poses·mstep·mframes)은 움직임과 시간에서 나오므로 뺀다.
    영역만 다르거나 장애물 모양만 다른(생성기 높이·폭 등) 두 환경은 다른 세계다."""
    movers = [([_shape_json(s) for s in b.shapes], repr(b.motion)) for b in geom.bodies if _is_mover(b)]
    regions = [(r.kind, _flat(r.points)) for r in geom.regions]
    return json.dumps([[_n(v) for v in geom.bounds], None if view is None else [_n(v) for v in view],
                       _static_json(geom), regions, movers], default=str)


def _world_json(geom: WorldGeometry, view: Sequence[float] | None, step: float, horizon: float) -> dict[str, Any]:
    step = max(float(step), 1e-6)
    mframes = max(2, math.ceil(horizon / step - 1e-9) + 1)
    times = np.arange(mframes, dtype=np.float64) * step  # 칸 프레임 시각과 같은 식(k × every × dt)
    movers = []
    for b in geom.bodies:
        if not _is_mover(b):
            continue
        assert b.motion is not None
        p = np.array([pose_at(b.motion, float(t)) for t in times], dtype=np.float64)
        q = np.empty_like(p)
        q[:, :2] = np.clip(np.rint(np.nan_to_num(p[:, :2], nan=0.0, posinf=0.0, neginf=0.0) / POS_UNIT), *_INT32)
        q[:, 2] = np.rint(_wrap(np.nan_to_num(p[:, 2], nan=0.0, posinf=0.0, neginf=0.0)) / ANGLE_UNIT)
        movers.append({"shapes": [_shape_json(s) for s in b.shapes], "poses": _b64(q)})
    targets = [c for r in geom.regions if r.kind in ("pad", "goal") for pt in r.points for c in (_n(pt[0]), _n(pt[1]))]
    pts = [(targets[i], targets[i + 1]) for i in range(0, len(targets), 2)]
    if view is not None:
        pts += [(float(view[0]), float(view[1])), (float(view[2]), float(view[3]))]
    box = None
    if pts:
        box = [_n(min(p[0] for p in pts)), _n(min(p[1] for p in pts)), _n(max(p[0] for p in pts)),
               _n(max(p[1] for p in pts))]
    return {
        "bounds": [_n(v) for v in geom.bounds],
        "view": None if view is None else [_n(v) for v in view],
        "box": box,
        "targets": targets,
        "static": _static_json(geom),
        "regions": [{"kind": r.kind, "p": _flat(r.points)} for r in geom.regions],
        "movers": movers,
        "mstep": step,
        "mframes": mframes,
    }


def _static_json(geom: WorldGeometry) -> list[dict[str, Any]]:
    """정적 세계 물체의 모양(세계 좌표). 지형 표시 물체의 선분은 이어진 사슬 하나("chain": 관전 캔버스가 선과 그 아래 채우기로
    그리는 것과 같은 점 순서, 바로 붙은 같은 점은 하나로), 다른 선분은 "segment", 다각형 "poly", 원 "circle"."""
    out: list[dict[str, Any]] = []
    for b in geom.bodies:
        if b.moving or b.motion is not None:
            continue
        x0, y0, a0 = (float(v) for v in b.pose)
        c, s = math.cos(a0), math.sin(a0)

        def world(p: Sequence[float], x0: float = x0, y0: float = y0, c: float = c, s: float = s) -> tuple[float, float]:
            return (x0 + p[0] * c - p[1] * s, y0 + p[0] * s + p[1] * c)

        segs = [sh for sh in b.shapes if sh.kind == "segment" and len(sh.points) >= 2]
        if segs and "terrain" in b.tags:
            chain: list[tuple[float, float]] = []
            for sh in segs:
                for p in sh.points[:2]:
                    wx, wy = world(p)
                    q = (_n(wx), _n(wy))
                    if not chain or chain[-1] != q:
                        chain.append(q)
            if len(chain) == 1:
                chain.append(chain[0])
            out.append({"k": "chain", "p": [v for q in chain for v in q]})
        else:
            for sh in segs:
                out.append({"k": "segment", "p": _flat([world(sh.points[0]), world(sh.points[1])])})
        for sh in b.shapes:
            if sh.kind == "poly" and len(sh.points) >= 3:
                out.append({"k": "poly", "p": _flat([world(p) for p in sh.points])})
            elif sh.kind == "circle" and sh.points:
                out.append({"k": "circle", "p": _flat([world(sh.points[0])]), "r": _n(sh.radius)})
    return out


def _shape_json(shape: ShapeGeom) -> dict[str, Any]:
    """물체 좌표의 모양 하나."""
    d: dict[str, Any] = {"k": shape.kind, "p": _flat(shape.points)}
    if shape.kind != "poly":
        d["r"] = _n(shape.radius)
    return d


def _radius(shapes: Sequence[ShapeGeom]) -> float:
    """물체 원점에서 가장 먼 모양 점까지(관전 캔버스 _build_shape의 radius와 같은 규칙)."""
    r = 0.0
    for s in shapes:
        for x, y in s.points:
            r = max(r, math.hypot(float(x), float(y)) + (float(s.radius) if s.kind != "poly" else 0.0))
    return r


def _title(job: ExportJob) -> str:
    env = str(job.meta.get(META_ENV, "")).strip()
    label = str(job.title).strip()
    if env and label:
        return S.PHONE_TITLE.format(env=env, label=label)
    return label or env or S.APP_NAME


def _info_lines(meta: Mapping[str, str]) -> list[str]:
    """정보 줄: 실행 이름 · 내보낸 날(환경 이름은 제목에, 세대는 제목의 칸 이름에 있다)."""
    parts = [str(meta.get(k, "")).strip() for k in (META_RUN, META_DATE)]
    line = " · ".join(p for p in parts if p)
    return [line] if line else []


def _shades(color: str) -> tuple[str, str]:
    """겹쳐 보기 개체의 (다리 색, 테두리 색): 관전 캔버스 set_overlay와 같은 QColor.darker(125)·lighter(160)."""
    from PySide6.QtGui import QColor

    c = QColor(color)
    return c.darker(125).name().lower(), c.lighter(160).name().lower()


def _wrap(a: np.ndarray) -> np.ndarray:
    """각도를 [-π, π)로."""
    return np.mod(np.asarray(a, dtype=np.float64) + math.pi, 2.0 * math.pi) - math.pi


def _b64(q: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(q, dtype="<i4").tobytes()).decode("ascii")


def _ints(text: str, count: int) -> np.ndarray:
    raw = base64.b64decode(text or "", validate=True)
    if len(raw) != 4 * count:
        raise ValueError(f"expected {count} int32 values, got {len(raw)} bytes")
    return np.frombuffer(raw, dtype="<i4").astype(np.int64)


def _n(v: float, digits: int = 3) -> float | int:
    """JSON 숫자: 유한하지 않으면 0, digits 자리로 반올림, 정수면 정수로(".0"을 줄임)."""
    x = float(v)
    if not math.isfinite(x):
        return 0
    x = round(x, digits)
    return int(x) if x.is_integer() and abs(x) < 2**53 else x


def _flat(points: Sequence[Sequence[float]]) -> list[float | int]:
    return [_n(c) for p in points for c in (p[0], p[1])]


def _finite(v: float) -> float | None:
    x = float(v)
    return round(x, 4) if math.isfinite(x) else None


# ---- SVG 끝 장면 ----------------------------------------------------------
# 같은 색으로 칠하는 것은 <path> 하나로 묶는다(트랙 벽 336개를 요소 하나로: 그림 글이 몇 배 작다). y는 뒤집어 쓴다.
def _xy(x: float, y: float) -> str:
    return f"{_n(x)} {_n(-y)}"


def _d_line(flat: Sequence[float]) -> str:
    """평평한 점 목록 → 열린 꺾은선 경로 조각("M x y x y ...": M 뒤 점은 이어지는 선)."""
    pts = [_xy(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]
    return "M" + " ".join(pts) if len(pts) >= 2 else ""


def _d_shapes(shapes: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """모양 목록 → (채우기 경로: 다각형·원, 선 경로: 선분·사슬)."""
    fill: list[str] = []
    lines: list[str] = []
    for sh in shapes:
        p = sh.get("p") or []
        if sh["k"] == "poly" and len(p) >= 6:
            fill.append(_d_line(p) + "Z")
        elif sh["k"] == "circle" and len(p) >= 2:
            r = float(sh.get("r", 0.0))
            cx, cy = float(p[0]), float(p[1])
            fill.append(f"M{_xy(cx + r, cy)}A{_n(r)} {_n(r)} 0 1 0 {_xy(cx - r, cy)}"
                        f"A{_n(r)} {_n(r)} 0 1 0 {_xy(cx + r, cy)}Z")
        elif sh["k"] in ("segment", "chain") and len(p) >= 4:
            lines.append(_d_line(p))
    return "".join(fill), "".join(lines)


def _path(d: str, fill: str = "none", stroke: str = "none", width: float = 0.0, rule: str = "") -> str:
    if not d:
        return ""
    extra = f' fill-rule="{rule}"' if rule else ""
    if stroke == "none":
        return f'<path d="{d}" fill="{fill}"{extra}/>'
    return (f'<path d="{d}" fill="{fill}"{extra} stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" '
            f'stroke-linecap="round" vector-effect="non-scaling-stroke"/>')


def _g(x: float, y: float, a: float, body: str) -> str:
    return f'<g transform="translate({_xy(x, y).replace(" ", ",")}) rotate({_n(-math.degrees(a), 2)})">{body}</g>'


def _svg_world(world: Mapping[str, Any], col: Mapping[str, str], t_end: float) -> list[str]:
    out: list[str] = []
    bx0, by0, bx1, by1 = (float(v) for v in world["bounds"])
    out.append(_path(_d_line([bx0, by0, bx1, by0, bx1, by1, bx0, by1]) + "Z", stroke=col["bounds"], width=1))
    by_color: dict[str, list[str]] = {}
    for r in world["regions"]:
        if len(r["p"]) >= 6:
            color = {"pad": col["region"], "goal": col["region"], "hazard": col["hazard"]}.get(r["kind"], col["zone"])
            by_color.setdefault(color, []).append(_d_line(r["p"]) + "Z")
    out.extend(_path("".join(d), fill=c) for c, d in by_color.items())
    bottom = by0 - TERRAIN_DEPTH
    chains = [sh["p"] for sh in world["static"] if sh["k"] == "chain" and len(sh["p"]) >= 4]
    out.append(_path("".join(_d_line([*p, p[-2], bottom, p[0], bottom]) + "Z" for p in chains),
                     fill=col["terrain_fill"], rule="evenodd"))
    fill_d, _ = _d_shapes([sh for sh in world["static"] if sh["k"] in ("poly", "circle")])
    out.append(_path(fill_d, fill=col["world_body"], stroke=col["terrain_line"], width=1))
    _, line_d = _d_shapes([sh for sh in world["static"] if sh["k"] in ("chain", "segment")])
    out.append(_path(line_d, stroke=col["terrain_line"], width=1.5))
    pads = "".join(_d_line(r["p"][:4]) for r in world["regions"] if r["kind"] == "pad" and len(r["p"]) >= 4)
    out.append(_path(pads, stroke=col["pad"], width=4))
    step, m = float(world["mstep"]), int(world["mframes"])
    k = min(m - 1, max(0, int(round(t_end / step))))
    for mv in world["movers"]:
        x, y, a = (float(v) for v in mv["poses"][k])
        fd, ld = _d_shapes(mv["shapes"])
        out.append(_g(x, y, a, _path(fd, fill=col["world_body"], stroke=col["terrain_line"], width=1)
                      + _path(ld, stroke=col["world_body"], width=2)))
    return [o for o in out if o]


def _svg_entry(e: Mapping[str, Any], col: Mapping[str, str]) -> list[str]:
    out: list[str] = []
    if not e["nb"]:
        return out
    poses = np.asarray(e["poses"])
    fills = {"agent": e["color"], "leg": e["leg"], "world": col["world_body"]}
    for b, shapes in enumerate(e["bodies"]):
        x, y, a = (float(v) for v in poses[-1, b])
        fd, ld = _d_shapes(shapes)
        out.append(_g(x, y, a, _path(fd, fill=fills.get(e["roles"][b], e["color"]), stroke=e["edge"], width=1.5)
                      + _path(ld, stroke=e["edge"], width=1.5)))
    trail = poses[:, 0, :2]
    stride = max(1, math.ceil(len(trail) / 400))
    pick = np.concatenate([trail[::stride], trail[-1:]])
    r, g, b_ = (int(e["color"][i:i + 2], 16) for i in (1, 3, 5))
    out.append(_path(_d_line([float(v) for xy in pick for v in xy]),
                     stroke=f"rgba({r},{g},{b_},{_ALPHA_TRAIL / 255:.3f})", width=2))
    return [o for o in out if o]


__all__ = [
    "ANGLE_UNIT", "COLORS", "FIT_MARGIN", "FIT_MIN", "OUT_SCALE", "PHONE_FORMAT", "PHONE_MAX_BODIES", "PHONE_MAX_ENTRIES",
    "PHONE_MAX_FRAMES", "PLACEHOLDERS", "POSTER_COLORS", "POS_UNIT", "TAIL", "TEMPLATE_PATH", "PhoneTooBig",
    "build_payload", "decode_payload", "fit_box", "payload_from_html", "phone_html", "poster_png_uri", "read_phone_html",
    "svg_poster", "svg_poster_uri", "template_text", "write_phone_html",
]

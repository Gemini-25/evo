"""「새 프리셋」 창 (6.2.0, 최종 설계 3.1 W·D38·D42·D49, DU 4.1·5.1).

조리법(presets.recipe.Recipe)이 진짜 상태다. 쪽에서 무엇을 바꾸든 창은 조리법을 고치고 → `adjust`(맞지 않는 선택을 고치고 바꾼 것의
글) → `build`(환경 사전) → L1 `quick_check` → L2 `alive`를 다시 한다. 만들 수 없는 바꿈(드묾)은 받아들이지 않고 앞 상태로 둔다: 그래서
창이 가진 환경은 늘 L1을 넘는다(T23). L3 `probe`는 「해 보기」 쪽이 보일 때 뒤 스레드에서 돌고 알림뿐이다.

창 동작(D42): 창에 묶인 모달(`open()`), 처음 크기는 1,180×700과 화면 작업 영역 가운데 작은 쪽, 최소 1,000×580, 크기 조절 가능, 오른쪽
쪽은 스크롤. 모든 단추 `autoDefault` 끔(Enter가 아무 단추도 누르지 않음), 이름 칸의 Enter는 이름만 확정, Alt+←/→ 이전·다음,
Ctrl+Enter 「실험실로 보내기」. 「직접 해 보기」는 나가지 않는다(메모리의 환경으로 직접 조종 창을 자식으로 띄우고 저장하지 않음).
저장은 「에디터에서 다듬기」「실험실로 보내기」에서만: 먼저 `can_leave(무엇)`(에디터의 저장 물음, 실험실이 바쁜가)을 묻고, 참일 때만
`paths.safe_file_stem` 이름으로 내 환경 폴더에 쓰고(같은 이름이 있으면 「이름 2」…, 덮어쓰지 않음) `finished_with`를 낸 뒤 닫는다.

쓰임 기록(D49): 상태 폴더의 wizard_log.jsonl에 줄마다 {"at", "session", "step", "action", "ms"}만 적는다(이름·경로·환경 내용 없음).
설정 `preset/usage_log`(기본 켬)로 끈다. 오류 보고 묶음에는 들어가지 않는다(묶음은 settings.ini와 로그만 담는다).
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QRect, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QCloseEvent, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from evolab import paths
from evolab.app import strings as S
from evolab.app import style
from evolab.app.preset.pages import (
    ArenaPage,
    BodyPage,
    Page,
    RulesPage,
    StartPage,
    TryPage,
)
from evolab.app.preset.preview import PresetPreview
from evolab.presets import recipe as RC
from evolab.presets import trainable as TR
from evolab.presets.library import Library, default_library
from evolab.presets.recipe import Recipe, RecipeError
from evolab.presets.template import Template
from evolab.spec.compiled import EnvSpec

WIZARD_SIZE = (1180, 700)
WIZARD_MIN = (1000, 580)
RAIL_WIDTH = 180
SIDE_WIDTH = 400
STEP_COUNT = 5
USAGE_LOG_FILE = "wizard_log.jsonl"
USAGE_LOG_KEY = "preset/usage_log"
SETTINGS_FILE = "settings.ini"
STEP_ACTIONS = frozenset({"open", "step", "back", "next", "card", "body", "arena", "prop", "slider", "rule",
                          "rule_slot", "reset", "play", "edit", "lab", "close", "stay", "save_failed", "name"})
"""쓰임 기록에 적는 행동 이름(이 밖의 글은 적지 않는다)."""


# ---------------------------------------------------------------------------
# 쓰임 기록
# ---------------------------------------------------------------------------


def _settings() -> QSettings:
    return QSettings(str(paths.user_state_dir() / SETTINGS_FILE), QSettings.Format.IniFormat)


def usage_log_enabled() -> bool:
    """설정 `preset/usage_log`(기본 켬)."""
    value = _settings().value(USAGE_LOG_KEY, True)
    if isinstance(value, str):
        return value.lower() not in ("false", "0", "no")
    return bool(value)


def set_usage_log_enabled(on: bool) -> None:
    """설정 화면(S.SETTINGS_PRESET_USAGE_LOG)이 부른다."""
    s = _settings()
    s.setValue(USAGE_LOG_KEY, bool(on))
    s.sync()


def usage_log_path() -> Path:
    return paths.user_state_dir() / USAGE_LOG_FILE


class UsageLog:
    """창 하나의 쓰임 기록. 켜져 있을 때만 파일에 한 줄씩 덧붙인다(실패해도 창은 그대로)."""

    def __init__(self) -> None:
        self.enabled = usage_log_enabled()
        self.session = uuid.uuid4().hex[:12]
        self._t = time.monotonic()

    def write(self, step: int, action: str) -> None:
        if not self.enabled or action not in STEP_ACTIONS:
            return
        now = time.monotonic()
        row = {"at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(), "session": self.session,
               "step": int(step), "action": action, "ms": int(round((now - self._t) * 1000.0))}
        self._t = now
        try:
            path = usage_log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        except OSError:
            self.enabled = False


# ---------------------------------------------------------------------------
# 창
# ---------------------------------------------------------------------------


def initial_size(area: QRect) -> QSize:
    """처음 크기: 1,180×700과 작업 영역 가운데 작은 쪽(최소 크기 아래로는 안 내려감)."""
    return QSize(max(WIZARD_MIN[0], min(WIZARD_SIZE[0], area.width())),
                 max(WIZARD_MIN[1], min(WIZARD_SIZE[1], area.height())))


class PresetWizard(QDialog):
    """「새 프리셋」 창. finished_with(환경 글, "edit" | "lab", 저장한 경로)."""

    finished_with = Signal(str, str, object)
    _probe_ready = Signal(int, object)

    def __init__(self, parent: QWidget | None = None, *, start: str | None = None, recipe: Recipe | None = None,
                 library: Library | None = None, can_leave: Callable[[str], bool] | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(S.PRESET_TITLE)
        self.setObjectName("preset_wizard")
        self.setSizeGripEnabled(True)
        self.setMinimumSize(*WIZARD_MIN)
        self.library = library or default_library()
        self.can_leave = can_leave
        self.usage = UsageLog()
        self.starters = RC.starters(self.library)
        self.spec: EnvSpec | None = None
        self.data: dict[str, Any] = {}
        self.text = ""
        self.quick: TR.Quick | None = None
        self.alive: TR.Alive | None = None
        self.probe: TR.Probe | None = None
        self.notes: list[str] = []
        self.saved_path: Path | None = None
        self.play_windows: list[QWidget] = []
        self._probe_gen = 0
        self._probe_thread: threading.Thread | None = None
        self._dirty_pages: set[int] = set()
        self._visited: set[int] = {0}
        self._name_edited = False
        self.step = 0
        self._probe_ready.connect(self._got_probe)

        if recipe is not None:
            first = copy.deepcopy(recipe)
            self.start = first.start
        else:
            ids = [s.id for s in self.starters]
            self.start = start if start in ids else (ids[0] if ids else "blank_side")
            first = self._default_recipe(self.start)
        self._opened_start = self.start
        self.recipe = first

        self._build_ui()
        name = recipe.name if recipe is not None and recipe.name else self._default_name(self.start)
        self.name_edit.setText(name)
        self._name_edited = recipe is not None and bool(recipe.name)
        if not self._apply(first, None):
            self._apply(self._default_recipe("blank_side"), None)
        self._opened = self._snapshot()
        self._base = copy.deepcopy(self.recipe)
        self.go(0)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)
        screen = (parent.screen() if parent is not None else None) or QGuiApplication.primaryScreen()
        self.fit_to(screen.availableGeometry() if screen is not None else QRect(0, 0, *WIZARD_SIZE))
        self.usage.write(0, "open")

    # ------------------------------------------------------------------
    # 화면
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.rail = QListWidget()
        self.rail.setObjectName("rail")
        self.rail.setToolTip(S.PRESET_STEPS_TIP)
        self.rail.setFixedWidth(RAIL_WIDTH)
        for k, text in enumerate(S.PRESET_STEPS):
            item = QListWidgetItem(text)
            item.setToolTip(S.PRESET_STEP_TIPS[k])
            item.setSizeHint(QSize(RAIL_WIDTH - 8, 36))
            self.rail.addItem(item)
        self.rail.currentRowChanged.connect(self._rail_moved)
        name_label = QLabel(S.PRESET_NAME)
        name_label.setToolTip(S.PRESET_NAME_TIP)
        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("name_edit")
        self.name_edit.setToolTip(S.PRESET_NAME_TIP)
        self.name_edit.setMaxLength(120)
        name_label.setBuddy(self.name_edit)
        self.name_edit.textEdited.connect(self._name_typed)
        self.name_edit.returnPressed.connect(self._commit_name)
        self.name_edit.editingFinished.connect(self._commit_name)
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.addWidget(self.rail, 1)
        left.addWidget(name_label)
        left.addWidget(self.name_edit)
        left_host = QWidget()
        left_host.setLayout(left)
        left_host.setFixedWidth(RAIL_WIDTH)

        self.preview = PresetPreview()
        self.preview.setObjectName("preview")
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.pages: list[Page] = [StartPage(self), BodyPage(self), ArenaPage(self), RulesPage(self), TryPage(self)]
        self.stack = QWidget()
        stack_box = QVBoxLayout(self.stack)
        stack_box.setContentsMargins(0, 0, 0, 0)
        for page in self.pages:  # 지금 쪽만 보인다(숨은 쪽은 스크롤 높이에 들지 않는다)
            page.hide()
            stack_box.addWidget(page)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("pages")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.scroll.setWidget(self.stack)
        self.scroll.setFixedWidth(SIDE_WIDTH)

        middle = QHBoxLayout()
        middle.setSpacing(style.SPACE[2])
        middle.addWidget(left_host)
        middle.addWidget(self.preview, 1)
        middle.addWidget(self.scroll)

        self.status = QLabel()
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.btn_fix = QPushButton()
        self.btn_fix.setObjectName("btn_fix")
        self.btn_fix.hide()
        self.btn_fix.clicked.connect(self._fix)
        self.btn_reset_step = QPushButton(S.PRESET_BTN_RESET_STEP)
        self.btn_reset_step.setObjectName("btn_reset_step")
        self.btn_reset_step.setToolTip(S.PRESET_BTN_RESET_STEP_TIP)
        self.btn_reset_step.clicked.connect(self.reset_step)
        self.btn_edit = QPushButton(S.PRESET_BTN_EDIT)
        self.btn_edit.setObjectName("btn_edit")
        self.btn_edit.setToolTip(S.PRESET_BTN_EDIT_TIP)
        self.btn_edit.clicked.connect(lambda: self._leave("edit"))
        self.btn_play = QPushButton(S.PRESET_BTN_PLAY)
        self.btn_play.setObjectName("btn_play")
        self.btn_play.setToolTip(S.PRESET_BTN_PLAY_TIP)
        self.btn_play.clicked.connect(self.play)
        self.btn_lab = QPushButton(S.ED_TO_LAB)
        self.btn_lab.setObjectName("btn_lab")
        self.btn_lab.setToolTip(S.PRESET_BTN_LAB_TIP)
        style.set_role(self.btn_lab, "primary")
        self.btn_lab.clicked.connect(lambda: self._leave("lab"))
        self.btn_back = QPushButton(S.PRESET_BTN_BACK)
        self.btn_back.setObjectName("btn_back")
        self.btn_back.setToolTip(S.PRESET_BTN_BACK_TIP)
        self.btn_back.clicked.connect(self.back)
        self.btn_next = QPushButton(S.PRESET_BTN_NEXT)
        self.btn_next.setObjectName("btn_next")
        self.btn_next.setToolTip(S.PRESET_BTN_NEXT_TIP)
        self.btn_next.clicked.connect(self.next)

        status_row = QHBoxLayout()
        status_row.addWidget(self.status, 1)
        status_row.addWidget(self.btn_fix)
        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_reset_step)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_edit)
        buttons.addWidget(self.btn_play)
        buttons.addWidget(self.btn_lab)
        buttons.addSpacing(style.SPACE[2])
        buttons.addWidget(self.btn_back)
        buttons.addWidget(self.btn_next)

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SPACE[2], style.SPACE[2], style.SPACE[2], style.SPACE[2])
        root.setSpacing(style.SPACE[1])
        root.addLayout(middle, 1)
        root.addLayout(status_row)
        root.addLayout(buttons)

        for keys, slot in ((QKeySequence(Qt.Modifier.ALT | Qt.Key.Key_Left), self.back),
                           (QKeySequence(Qt.Modifier.ALT | Qt.Key.Key_Right), self.next),
                           (QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Return), self._lab_key),
                           (QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Enter), self._lab_key)):
            sc = QShortcut(keys, self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(slot)

    def fit_to(self, area: QRect) -> None:
        """작업 영역 area에 맞춘 처음 크기와 자리(가운데)."""
        extra = self.frameGeometry().size() - self.geometry().size()  # 제목 줄·테두리(띄운 뒤에만 앎)
        fw, fh = max(0, extra.width()), max(0, extra.height())
        size = initial_size(area.adjusted(0, 0, -fw, -fh))
        self.resize(size)
        x = area.x() + max(0, (area.width() - size.width() - fw) // 2)
        y = area.y() + max(0, (area.height() - size.height() - fh) // 2)
        self.move(x, y)  # 창틀의 왼쪽 위

    def tab_chain(self) -> list[QWidget]:
        """키보드 차례: 단계 줄 → 이름 → 미리보기 단추 → 지금 쪽 → 아래 단추들."""
        page = self.pages[self.step]
        chain: list[QWidget] = [self.rail, self.name_edit, self.preview.btn_run, *page.controls(),
                                self.btn_fix, self.btn_reset_step, self.btn_edit, self.btn_play, self.btn_lab,
                                self.btn_back, self.btn_next]
        return chain

    def _set_tab_order(self) -> None:
        chain = self.tab_chain()
        for a, b in zip(chain, chain[1:], strict=False):
            QWidget.setTabOrder(a, b)

    # ------------------------------------------------------------------
    # 첫 카드
    # ------------------------------------------------------------------

    def _default_recipe(self, start: str) -> Recipe:
        try:
            return RC.default_recipe(start, self.library)
        except RecipeError:
            return RC.default_recipe("blank_side", self.library)

    def _default_name(self, start: str) -> str:
        try:
            base = RC.default_recipe(start, self.library).name
        except RecipeError:
            base = ""
        return S.PRESET_NAME_DEFAULT.format(sport=base) if base else S.PRESET_NAME_FALLBACK

    def starter_body(self, start: str) -> Template | None:
        try:
            r = RC.default_recipe(start, self.library)
        except RecipeError:
            return None
        tid = r.roles[0].body.template if r.roles else None
        return self.library.get(tid) if tid and self.library.has(tid) else None

    def starter_blurb(self, start: str) -> str:
        st = next((s for s in self.starters if s.id == start), None)
        text = st.blurb if st is not None else ""
        if not text:
            try:
                text = RC.default_recipe(start, self.library).description
            except RecipeError:
                text = ""
        return text

    def can_add_region(self) -> bool:
        """영역이 필요한 규칙을 켜면 adjust가 골대를 함께 넣을 수 있나(그 세계에 쓸 골대 틀이 있음)."""
        if not self.library.has("goal"):
            return False
        return self.library.get("goal").world in (self.recipe.world, "any")

    def choose_start(self, start: str) -> None:
        if start == self.start:
            return
        old = self.start
        self.start = start
        r = self._default_recipe(start)
        if not self._name_edited:
            self.name_edit.setText(self._default_name(start))
        if self._apply(r, "card"):
            self._base = copy.deepcopy(self.recipe)
        else:
            self.start = old

    # ------------------------------------------------------------------
    # 조리법 바꾸기
    # ------------------------------------------------------------------

    def change(self, mutate: Callable[[Recipe], None], action: str) -> bool:
        r = copy.deepcopy(self.recipe)
        mutate(r)
        return self._apply(r, action)

    def _apply(self, recipe: Recipe, action: str | None) -> bool:
        """조리법을 맞추고 만들고 확인한다. 만들 수 없으면 앞 상태 그대로 두고 False."""
        r = copy.deepcopy(recipe)
        r.name = self.name_edit.text().strip()
        try:
            r, notes = RC.adjust(r, self.library)
            data = RC.build(r, self.library)
        except (RecipeError, KeyError, ValueError) as exc:
            self._show_status(S.PRESET_STATUS_BAD.format(n=1, first=str(exc)))
            self._sync_all()
            return False
        quick = TR.quick_check(data)
        if not quick.ok:  # 생기면 안 되는 상태(T15·T23): 받아들이지 않는다
            first = quick.errors[0].message if quick.errors else ""
            self._show_status(S.PRESET_STATUS_BAD.format(n=len(quick.errors), first=first))
            self._sync_all()
            return False
        self.recipe = r
        self.notes = notes
        self.data = data
        self.text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        self.quick = quick
        self.spec = quick.spec
        self.alive = TR.alive(quick.spec) if quick.spec is not None else None
        self.probe = None
        self._probe_gen += 1
        self.preview.set_spec(self.spec)
        self._dirty_pages = set(range(STEP_COUNT))
        self._sync_all()
        self._update_status()
        if action is not None:
            self.usage.write(self.step, action)
        return True

    def _sync_all(self) -> None:
        """보이는 쪽만 지금 그리고 나머지는 보일 때 그린다."""
        step = self.step
        if step in self._dirty_pages:
            self.pages[step].sync()
            self._dirty_pages.discard(step)
            self._set_tab_order()
        if step == 4:
            self._start_probe()
        self._update_rail()

    def _snapshot(self) -> tuple[str, str]:
        return json.dumps(self.recipe.to_json(), sort_keys=True), self.name_edit.text().strip()

    @property
    def changed(self) -> bool:
        """처음 연 뒤 무엇이라도 바꿨나(닫을 때 물음)."""
        return self._snapshot() != self._opened

    # ------------------------------------------------------------------
    # 이름
    # ------------------------------------------------------------------

    def _name_typed(self, _text: str) -> None:
        self._name_edited = True

    def _commit_name(self) -> None:
        name = self.name_edit.text().strip()
        if name != self.recipe.name.strip():
            self._apply(self.recipe, "name")

    # ------------------------------------------------------------------
    # 단계
    # ------------------------------------------------------------------

    def go(self, step: int) -> None:
        step = max(0, min(STEP_COUNT - 1, int(step)))
        self.step = step
        self._visited.add(step)
        for k, page in enumerate(self.pages):
            page.setVisible(k == step)
        if self.rail.currentRow() != step:
            self.rail.blockSignals(True)
            self.rail.setCurrentRow(step)
            self.rail.blockSignals(False)
        if step in self._dirty_pages or not hasattr(self, "_opened"):
            self.pages[step].sync()
            self._dirty_pages.discard(step)
        self._set_tab_order()
        if step == 4:
            self._start_probe()
            self.pages[4].sync()
        self.scroll.verticalScrollBar().setValue(0)
        style.set_enabled(self.btn_back, step > 0)
        style.set_enabled(self.btn_next, step < STEP_COUNT - 1)
        style.set_enabled(self.btn_reset_step, step < STEP_COUNT - 1, "" if step < STEP_COUNT - 1 else
                          S.PRESET_RESET_NOTHING)
        self._update_rail()

    def _rail_moved(self, row: int) -> None:
        if row >= 0 and row != self.step:
            self.go(row)
            self.usage.write(row, "step")

    def back(self) -> None:
        if self.step > 0:
            self.go(self.step - 1)
            self.usage.write(self.step, "back")

    def next(self) -> None:
        if self.step < STEP_COUNT - 1:
            self.go(self.step + 1)
            self.usage.write(self.step, "next")

    def _changed_steps(self) -> set[int]:
        """종목 틀의 처음 값에서 바꾼 단계."""
        base, now = self._base, self.recipe
        out: set[int] = set()
        if self.start != self._opened_start:
            out.add(0)
        if [r.body.to_json() for r in base.roles] != [r.body.to_json() for r in now.roles]:
            out.add(1)
        if base.arena.to_json() != now.arena.to_json() or [p.to_json() for p in base.props] != [
                p.to_json() for p in now.props]:
            out.add(2)
        if base.rules != now.rules:
            out.add(3)
        return out

    def _update_rail(self) -> None:
        if not hasattr(self, "rail"):
            return
        changed = self._changed_steps() if hasattr(self, "_base") else set()
        for k in range(self.rail.count()):
            item = self.rail.item(k)
            done = k in self._visited and k != getattr(self, "step", 0)
            item.setText(S.PRESET_STEP_DONE.format(step=S.PRESET_STEPS[k]) if done else S.PRESET_STEPS[k])
            font = item.font()
            font.setBold(k in changed)
            item.setFont(font)
            tip = S.PRESET_STEP_TIPS[k]
            item.setToolTip(f"{tip}\n{S.PRESET_STEP_CHANGED_TIP}" if k in changed else tip)

    def reset_step(self) -> None:
        """지금 단계의 고른 것을 종목 틀의 처음 값으로."""
        step = self.step
        if step == 0:
            if not self._name_edited:
                self.name_edit.setText(self._default_name(self._opened_start))
            if self.start != self._opened_start:
                self.choose_start(self._opened_start)
            self.usage.write(step, "reset")
            return
        base = self._default_recipe(self.start)

        def mutate(r: Recipe) -> None:
            if step == 1:
                r.roles = copy.deepcopy(base.roles)
            elif step == 2:
                r.arena = copy.deepcopy(base.arena)
                r.props = copy.deepcopy(base.props)
            elif step == 3:
                r.rules = copy.deepcopy(base.rules)

        self.change(mutate, "reset")

    # ------------------------------------------------------------------
    # 상태 줄과 단추
    # ------------------------------------------------------------------

    def _show_status(self, text: str) -> None:
        self.status.setText(text)

    def status_text(self) -> str:
        q, a = self.quick, self.alive
        if q is None:
            return ""
        parts: list[str] = []
        if self.notes:
            parts.append(" ".join(S.PRESET_STATUS_ADJUSTED.format(what=n) for n in self.notes))
        if not q.ok:
            first = q.errors[0].message if q.errors else ""
            parts.append(S.PRESET_STATUS_BAD.format(n=len(q.errors), first=first))
        elif a is not None and not a.ok:
            parts.append(a.reason)
        elif q.outputs == 0 and q.body_genes > 0:
            parts.append(S.PRESET_STATUS_BODY_ONLY.format(genes=q.body_genes))
        else:
            parts.append(S.PRESET_STATUS_OK.format(inputs=q.inputs, outputs=q.outputs))
        if a is not None and a.ok and a.notice():
            parts.append(a.notice())
        return " ".join(p for p in parts if p)

    def _update_status(self) -> None:
        self._show_status(self.status_text())
        ok1 = self.quick is not None and self.quick.ok
        ok2 = ok1 and self.alive is not None and self.alive.ok
        why = "" if ok2 else (self.alive.reason if self.alive is not None else "")
        style.set_enabled(self.btn_lab, ok2, why)
        style.set_enabled(self.btn_edit, ok1)
        style.set_enabled(self.btn_play, ok1 and self.spec is not None and bool(self.spec.outputs))
        fix = self.alive.fix if self.alive is not None and not self.alive.ok else None
        if fix:
            self.btn_fix.setText(S.PRESET_ALIVE_FIXES.get(fix, ""))
            self.btn_fix.setToolTip(S.PRESET_FIX_TIP)
            self.btn_fix.show()
        else:
            self.btn_fix.hide()

    def _fix(self) -> None:
        fix = self.alive.fix if self.alive is not None else None
        self.go(3 if fix == "add_rule" else 2)

    def check_text(self) -> str:
        """「해 보기」 쪽의 확인 결과: L2 알림 + L3 결과(돌고 있으면 「확인 중…」)."""
        parts: list[str] = []
        if self.alive is not None and not self.alive.ok:
            parts.append(self.alive.reason)
        elif self.alive is not None and self.alive.notice():
            parts.append(self.alive.notice())
        if self.probe is not None:
            parts.append(self.probe.reason)
        elif self.spec is not None:
            parts.append(S.PRESET_PROBE_RUNNING)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # L3(뒤 스레드)
    # ------------------------------------------------------------------

    def _start_probe(self) -> None:
        if self.spec is None or self.probe is not None:
            return
        gen = self._probe_gen
        if self._probe_thread is not None and self._probe_thread.is_alive() and getattr(
                self._probe_thread, "gen", None) == gen:
            return
        spec = self.spec

        def work() -> None:
            try:
                result: Any = TR.probe(spec)
            except Exception:  # noqa: BLE001 - 알림뿐이라 실패는 조용히
                result = None
            try:
                self._probe_ready.emit(gen, result)
            except RuntimeError:  # 창이 먼저 닫혀 지워짐
                pass

        t = threading.Thread(target=work, name="preset-probe", daemon=True)
        t.gen = gen  # type: ignore[attr-defined]
        self._probe_thread = t
        t.start()

    def _got_probe(self, gen: int, result: object) -> None:
        if gen != self._probe_gen or not isinstance(result, TR.Probe):
            return
        self.probe = result
        if self.step == 4:
            self.pages[4].sync()

    def wait_probe(self, timeout: float = 30.0) -> TR.Probe | None:
        """(테스트용) 지금 문서의 L3가 끝날 때까지 기다린다."""
        from PySide6.QtCore import QCoreApplication

        self._start_probe()
        end = time.monotonic() + timeout
        while self.probe is None and time.monotonic() < end:
            QCoreApplication.processEvents()
            t = self._probe_thread
            if t is not None:
                t.join(0.01)
        QCoreApplication.processEvents()
        return self.probe

    # ------------------------------------------------------------------
    # 나가기
    # ------------------------------------------------------------------

    def _lab_key(self) -> None:
        if self.btn_lab.isEnabled():
            self._leave("lab")

    def play(self) -> QWidget | None:
        """직접 해 보기: 메모리의 환경으로 직접 조종 창을 자식 창으로. 저장하지도 닫지도 않는다."""
        self._commit_name()
        if self.spec is None:
            return None
        from evolab.app.play.window import PlayWindow
        from evolab.world.physics2d.episode import EpisodeBuildError

        try:
            window = PlayWindow(self.spec)
        except EpisodeBuildError:
            self._show_status(S.PRESET_PROBE_BUILD)
            return None
        window.setParent(self, Qt.WindowType.Window)
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.play_windows.append(window)
        window.destroyed.connect(lambda _=None, w=window: self.play_windows.remove(w) if w in self.play_windows
                                 else None)
        window.show()
        self.usage.write(self.step, "play")
        return window

    def target_path(self, name: str | None = None) -> tuple[Path, str]:
        """저장할 경로와 환경 이름: 내 환경 폴더/<safe_file_stem(이름)>.evoenv, 있으면 「이름 2」「이름 3」…"""
        name = (name if name is not None else self.name_edit.text()).strip() or S.PRESET_NAME_FALLBACK
        folder = paths.user_env_dir()
        stem = paths.safe_file_stem(name)
        path = folder / f"{stem}.evoenv"
        n = 2
        final = name
        while path.exists():
            final = S.PRESET_NAME_NUMBERED.format(name=name, n=n)
            path = folder / f"{paths.safe_file_stem(final)}.evoenv"
            n += 1
        return path, final

    def save(self) -> Path:
        """내 환경 폴더에 새 파일로 쓴다(덮어쓰지 않음). OSError는 부르는 쪽이 받는다."""
        path, final = self.target_path()
        if final != self.recipe.name.strip():
            self.name_edit.setText(final)
            self._apply(self.recipe, None)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as f:
            f.write(self.text)
        self.saved_path = path
        return path

    def _leave(self, what: str) -> bool:
        self._commit_name()
        ok1 = self.quick is not None and self.quick.ok
        if not ok1 or (what == "lab" and not (self.alive is not None and self.alive.ok)):
            return False
        if self.can_leave is not None and not self.can_leave(what):
            self.usage.write(self.step, "stay")
            return False
        try:
            path = self.save()
        except OSError as exc:
            self._show_status(S.PRESET_SAVE_FAILED.format(reason=exc.strerror or exc))
            self.usage.write(self.step, "save_failed")
            return False
        self.usage.write(self.step, what)
        self._leaving = True
        self.preview.stop()
        self.finished_with.emit(self.text, what, path)
        self.accept()
        return True

    def ask_discard(self) -> bool:
        """「만들던 프리셋을 버릴까요?」(테스트가 바꿔 끼운다)."""
        answer = QMessageBox.question(self, S.PRESET_TITLE, S.PRESET_DISCARD,
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def reject(self) -> None:
        if getattr(self, "_leaving", False):
            super().reject()
            return
        if self.changed and not self.ask_discard():
            return
        self._leaving = True
        self.preview.stop()
        self.usage.write(self.step, "close")
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if getattr(self, "_leaving", False) or self.result() == QDialog.DialogCode.Accepted:
            event.accept()
            return
        if self.changed and not self.ask_discard():
            event.ignore()
            return
        self._leaving = True
        self.preview.stop()
        self.usage.write(self.step, "close")
        event.accept()


def open_wizard(parent: QWidget, **kw: Any) -> PresetWizard:
    """창에 묶인 모달로 띄우고 돌려준다(open(): 부르는 쪽은 막히지 않는다. 결과는 finished_with)."""
    wizard = PresetWizard(parent, **kw)
    wizard.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    wizard.open()
    return wizard


__all__ = ["PresetWizard", "UsageLog", "initial_size", "open_wizard", "set_usage_log_enabled", "usage_log_enabled",
           "usage_log_path"]

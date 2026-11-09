"""오류 문구: 예외를 사람이 읽을 글(무엇이 잘못됐나 + 다음에 할 일 + 자세히)로 바꾸고 오류 상자로 보인다 (v3 설계 D26, 4.8, 5.5).

- `explain(error)` → `Friendly(message, next_step, detail)`. 구체적인 예외부터 본다:

  | 예외 | 무엇 | 다음에 할 일 |
  | --- | --- | --- |
  | `MemoryError` | ERR_MEMORY | ERR_MEMORY_NEXT |
  | `PermissionError`, `OSError`(EACCES·EPERM·EROFS, 윈도우 5·19·32·33) | ERR_PERMISSION | ERR_PERMISSION_NEXT |
  | `OSError`(ENOSPC·EDQUOT, 윈도우 39·112) | ERR_DISK_FULL | ERR_DISK_FULL_NEXT |
  | `FileNotFoundError`(ENOENT, 윈도우 2·3) | ERR_NOT_FOUND(경로) | ERR_NOT_FOUND_NEXT |
  | `EnvFileError` | 예외 글(이미 한국어) | ERR_ENV_FILE_NEXT |
  | `EpisodeBuildError` | 예외 글 | ERR_BUILD_NEXT |
  | `RunStoreError` | 예외 글 | ERR_RUN_STORE_NEXT |
  | 그 밖에 우리 코드가 한국어 글로 낸 예외(설정 검사 등, `OSError` 말고) | 예외 글 | ERR_UNEXPECTED_NEXT |
  | 그 밖 | ERR_UNEXPECTED(예외 종류 이름) | ERR_UNEXPECTED_NEXT |
  | 글(str: 이미 한국어로 만든 알림) | 그 글 | ERR_UNEXPECTED_NEXT |

  `EnvFileError`·`RunStoreError`가 파일을 못 읽거나 못 써서 생긴 것이면(안에 `OSError`가 있으면) 그 `OSError`로 판단한다(OS 글이
  영어로 섞이지 않게). 「자세히」(detail)는 예외 원문과 traceback(안의 OSError 포함). 입력 기록 오류처럼 부르는 쪽만 뜻을 아는
  예외는 부르는 쪽이 `next_step=`·`message=`로 준다(설계 4.8). `reason(error)`은 「…하지 못했습니다: {reason}」 같은 한국어 틀에 넣을
  짧은 까닭이다(모르는 예외면 예외 종류 이름만, 원문은 「자세히」로).
- `show_error(parent, title, error, *, next_step=None, message=None)`: 제목은 부르는 쪽 그대로, 본문 = 무엇이 잘못됐나, 그 아래
  「다음에 할 일: …」, 단추 [확인] [오류 보고 묶음 만들기…], 「자세히」 = 예외 원문과 traceback. 늘 오류 기록(errors.jsonl)에
  kind="shown" 한 줄을 남긴다. 상자는 **막지 않고** 띄운다(`open()`: 그 창만 막는 창 모달): 부르는 쪽은 곧바로 이어 가고, 학습
  클라이언트 타이머 같은 슬롯 안에서 중첩 이벤트 루프가 생기지 않는다(안내·자체 점검이 상자 때문에 멈추지 않음). 상자를 띄운 창이
  닫히면(프로그램이 숨기면) 그 상자도 닫는다.
- 화면 파일은 `QMessageBox.*`에 예외(except 이름, `str(이름)`, 그 이름을 넣은 `format`)를 넣지 않는다: tests/test_errors.py가 AST로
  막는다(이 파일만 예외).
"""

from __future__ import annotations

import errno
import functools
import os
import re
import traceback
import weakref
from dataclasses import dataclass, replace
from typing import Any, Final

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Slot
from PySide6.QtWidgets import QMessageBox, QWidget

from evolab.app import strings as S

PERMISSION_ERRNOS: Final = frozenset({errno.EACCES, errno.EPERM, errno.EROFS})
PERMISSION_WINERRORS: Final = frozenset({5, 19, 32, 33})
"""윈도우: 접근 거부, 쓰기 금지 매체, 다른 프로세스가 사용 중(공유 위반), 잠금 위반."""
DISK_FULL_ERRNOS: Final = frozenset({errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)})
DISK_FULL_WINERRORS: Final = frozenset({39, 112})
"""윈도우: ERROR_HANDLE_DISK_FULL, ERROR_DISK_FULL."""
NOT_FOUND_WINERRORS: Final = frozenset({2, 3})
"""윈도우: 파일 없음, 경로 없음."""
BOX_NAME: Final = "evolabErrorBox"
"""show_error 상자의 objectName(안내·테스트가 찾는 이름)."""
REPORT_REASON: Final = "error"
"""「오류 보고 묶음 만들기…」로 만든 묶음의 까닭(REP_REASONS 키)."""

_HANGUL = re.compile("[{}-{}{}-{}]".format(chr(0x3131), chr(0x318E), chr(0xAC00), chr(0xD7A3)))
"""한글 자모·음절(우리 코드가 strings.py 글로 낸 예외인지 가른다). 코드값으로 써서 「strings.py 밖 한글 없음」 규칙을 지킨다."""
_open_boxes: dict[int, QMessageBox] = {}
"""떠 있는 상자(띄운 차례). 부모 없이 띄운 상자가 파이썬 쪽에서 사라지지 않게 닫히거나 지워질 때까지 쥔다."""


@dataclass(frozen=True)
class Friendly:
    """사람이 읽을 오류: 무엇이 잘못됐나, 다음에 할 일, 자세히(예외 원문과 traceback)."""

    message: str
    next_step: str
    detail: str


@functools.cache
def _known() -> tuple[type[BaseException], type[BaseException], type[BaseException]]:
    """우리 예외 클래스(처음 쓸 때 불러온다: 이 모듈을 가볍게, 불러오기 고리 없이)."""
    from evolab.runs.store import RunStoreError
    from evolab.spec.compiled import EnvFileError
    from evolab.world.physics2d.episode import EpisodeBuildError

    return EnvFileError, EpisodeBuildError, RunStoreError


def os_cause(error: BaseException) -> OSError | None:
    """error가 OSError면 그것, 우리 예외(환경 파일·실행 폴더)가 OSError 때문에 생겼으면 그 OSError. 아니면 None."""
    if isinstance(error, OSError):
        return error
    env_error, _build, store_error = _known()
    if isinstance(error, (env_error, store_error)):
        inner = error.__cause__ or error.__context__
        if isinstance(inner, OSError):
            return inner
    return None


def _os_kind(error: OSError) -> tuple[str, str] | None:
    """OS 오류를 권한·디스크 가득·찾을 수 없음으로 나눈다. 그 밖이면 None."""
    code = error.errno
    winerror = getattr(error, "winerror", None)
    if isinstance(error, PermissionError) or code in PERMISSION_ERRNOS or winerror in PERMISSION_WINERRORS:
        return S.ERR_PERMISSION, S.ERR_PERMISSION_NEXT.format(menu=S.MENU_SETTINGS)
    if code in DISK_FULL_ERRNOS or winerror in DISK_FULL_WINERRORS:
        return S.ERR_DISK_FULL, S.ERR_DISK_FULL_NEXT.format(records=S.TAB_RECORDS)
    if isinstance(error, FileNotFoundError) or code == errno.ENOENT or winerror in NOT_FOUND_WINERRORS:
        name = error.filename
        path = os.fsdecode(name) if isinstance(name, bytes) else ("" if name is None else str(name))
        text = S.ERR_NOT_FOUND.format(path=path) if path else S.ERR_NOT_FOUND.format(path="").rstrip().rstrip(":")
        return text, S.ERR_NOT_FOUND_NEXT.format(records=S.TAB_RECORDS)
    return None


def _classify(error: BaseException) -> tuple[str, str, bool]:
    """(무엇, 다음에 할 일, 모르는 예외인가)."""
    env_error, build_error, store_error = _known()
    if isinstance(error, MemoryError):
        return S.ERR_MEMORY, S.ERR_MEMORY_NEXT, False
    os_error = os_cause(error)
    if os_error is not None:
        kind = _os_kind(os_error)
        if kind is not None:
            return kind[0], kind[1], False
    if isinstance(error, env_error):
        return str(error), S.ERR_ENV_FILE_NEXT.format(issues=S.EDP_TAB_ISSUES), False
    if isinstance(error, build_error):
        return str(error), S.ERR_BUILD_NEXT, False
    if isinstance(error, store_error):
        return str(error), S.ERR_RUN_STORE_NEXT.format(records=S.TAB_RECORDS), False
    text = str(error).strip()
    if text and not isinstance(error, OSError) and _HANGUL.search(text):
        return text, S.ERR_UNEXPECTED_NEXT, False  # 우리 코드가 strings.py 글로 낸 예외(설정 검사, 계산 도우미 등)
    return S.ERR_UNEXPECTED.format(kind=type(error).__name__), S.ERR_UNEXPECTED_NEXT, True


def detail_text(error: BaseException) -> str:
    """「자세히」: 예외 원문과 traceback. `raise ... from None`으로 숨긴 안쪽 예외(파일을 못 읽은 OSError 등)도 붙인다."""
    text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    inner = error.__context__
    if error.__suppress_context__ and inner is not None:
        text += "\n" + "".join(traceback.format_exception(type(inner), inner, inner.__traceback__))
    return text.rstrip()


def explain(error: BaseException | str) -> Friendly:
    """예외(또는 이미 만든 한국어 알림 글)를 사람이 읽을 오류로."""
    if isinstance(error, str):
        return Friendly(error, S.ERR_UNEXPECTED_NEXT, "")
    message, next_step, _unknown = _classify(error)
    return Friendly(message, next_step, detail_text(error))


def reason(error: BaseException) -> str:
    """한국어 틀(「…하지 못했습니다: {reason}」)에 넣을 짧은 까닭: 우리 한국어 글, 나눈 OS 오류 글, 모르는 예외면 종류 이름."""
    message, _next, unknown = _classify(error)
    return type(error).__name__ if unknown else message


# ---------------------------------------------------------------------------
# 학습 프로세스가 보낸 오류 글(engine/session.py: 「…: {reason}」, reason = str(예외))
# ---------------------------------------------------------------------------
TRAINING_TEMPLATES: Final = (S.TRAIN_CANNOT_START, S.SESSION_CRASHED, S.SESSION_SAVE_FAILED)
"""학습 프로세스의 오류 글 틀. 까닭 자리에 예외 원문(str)이 들어오므로 OS 오류면 영어 원문과 경로가 섞인다."""

_OS_TEXT = re.compile(r"^(?:(?P<cls>[A-Za-z_][\w.]*): )?\[(?P<kind>Errno|WinError) (?P<code>-?\d+)\] (?P<rest>.*)$")
"""str(OSError) 모양: 「[Errno 13] Permission denied: 'C:\\…'」(traceback 마지막 줄이면 앞에 「PermissionError: 」)."""
_QUOTED = r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\""
_OS_REST = re.compile(rf"^(?P<strerror>.*?)(?:: (?P<f1>{_QUOTED})(?: -> (?P<f2>{_QUOTED}))?)?$")


def _unquote(text: str | None) -> str | None:
    if text is None:
        return None
    import ast

    try:
        value = ast.literal_eval(text)  # repr로 쓴 경로(윈도우 \\는 \ 하나로)
    except (ValueError, SyntaxError):
        return text[1:-1]
    return value if isinstance(value, str) else None


def os_error_from_text(text: str) -> OSError | None:
    """str(OSError) 글(앞에 「종류: 」가 있어도 됨)을 OSError로 되살린다. 그 모양이 아니면 None.

    errno 글은 OSError가 errno로 하위 종류(PermissionError 등)를 고르고, WinError 글은 winerror 속성을 단다(_os_kind가 읽음)."""
    m = _OS_TEXT.match(text.strip())
    if m is None:
        return None
    rest = _OS_REST.match(m.group("rest"))
    strerror = rest.group("strerror") if rest is not None else m.group("rest")
    filename = _unquote(rest.group("f1")) if rest is not None else None
    filename2 = _unquote(rest.group("f2")) if rest is not None else None
    code = int(m.group("code"))
    import builtins

    cls = getattr(builtins, m.group("cls") or "", None)
    if not (isinstance(cls, type) and issubclass(cls, OSError)):
        cls = OSError
    if m.group("kind") == "Errno":
        exc = cls(code, strerror, filename, None, filename2) if filename2 is not None else cls(code, strerror, filename)
    else:
        exc = cls(None, strerror, filename)
        exc.winerror = code  # type: ignore[attr-defined]  # 윈도우가 아니면 이 속성이 없다
    return exc


def os_error_from_traceback(detail: str) -> OSError | None:
    """traceback 글에서 마지막 OS 오류 줄(「PermissionError: [Errno 13] …」)을 OSError로. 없으면 None."""
    for line in reversed((detail or "").splitlines()):
        line = line.strip()
        if line and _OS_TEXT.match(line) and ": [" in line:
            return os_error_from_text(line)
    return None


def split_training_message(message: str) -> tuple[str | None, str]:
    """학습 오류 글 → (틀, 까닭). 틀에 맞지 않으면 (None, 글 그대로)."""
    for template in TRAINING_TEMPLATES:
        head, _sep, tail = template.partition("{reason}")
        if message.startswith(head) and message.endswith(tail):
            return template, message[len(head):len(message) - len(tail) if tail else None]
    return None, message


def training_error(message: str, detail: str = "") -> tuple[OSError, str] | None:
    """학습 프로세스의 오류 글에 OS 오류 원문(「[Errno …]」·「[WinError …]」)이 섞였고 그 OS 오류를 권한·디스크 가득·찾을 수 없음으로
    나눌 수 있으면 (OSError, 같은 틀에 한국어 까닭을 넣은 글). 아니면 None(글을 그대로 보인다).

    OS 오류는 까닭 자리의 글에서, 안 되면 학습 프로세스가 보낸 traceback(detail)에서 되살린다."""
    if "[Errno " not in message and "[WinError " not in message:
        return None
    template, rest = split_training_message(message)
    exc = os_error_from_text(rest) or os_error_from_traceback(detail)
    if exc is None or _os_kind(exc) is None:
        return None
    why = reason(exc)
    return exc, template.format(reason=why) if template is not None else why


def structure_next_step(error: BaseException) -> str | None:
    """에디터가 열지 못한 환경 파일의 다음 할 일: 파일을 못 읽었으면 None(explain이 고른 OS 오류의 다음 할 일), 구조가 망가졌으면
    ERR_ENV_STRUCTURE_NEXT(그런 파일은 에디터도 열지 못해 「검사 결과」 탭을 볼 수 없다)."""
    if os_cause(error) is not None:
        return None
    return S.ERR_ENV_STRUCTURE_NEXT.format(builtin=S.ED_FROM_BUILTIN)


def _record(title: str, friendly: Friendly) -> None:
    """오류 기록에 kind="shown" 한 줄(진단 모듈이 없거나 실패해도 상자는 띄운다)."""
    try:
        from evolab import diagnostics

        diagnostics.record("shown", f"{title}: {friendly.message}", friendly.detail)
    except Exception:  # noqa: BLE001 - 기록 때문에 오류 상자를 못 띄우면 안 된다
        pass


def error_box(parent: QWidget | None, title: str, friendly: Friendly) -> QMessageBox:
    """오류 상자를 만든다(띄우지는 않음). 글은 모두 일반 글(예외 글의 <, & 를 HTML로 읽지 않게)."""
    box = QMessageBox(parent)
    box.setObjectName(BOX_NAME)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setTextFormat(Qt.TextFormat.PlainText)  # 본문과 「다음에 할 일」 줄 둘 다
    box.setText(friendly.message)
    if friendly.next_step:
        box.setInformativeText(f"{S.ERR_NEXT_LABEL}: {friendly.next_step}")
    if friendly.detail:
        box.setDetailedText(friendly.detail)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.setDefaultButton(QMessageBox.StandardButton.Ok)
    box.setEscapeButton(QMessageBox.StandardButton.Ok)
    report = box.addButton(S.ERR_REPORT_BUTTON, QMessageBox.ButtonRole.ActionRole)
    box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    box.friendly = friendly  # type: ignore[attr-defined]  # 안내·테스트가 읽는다
    box.report_button = report  # type: ignore[attr-defined]
    return box


def open_report(parent: QWidget | None) -> Any:
    """「오류 보고 묶음 만들기…」: 오류 보고 묶음 대화 상자(D 묶음). 주 창 실험실의 지금 실행·환경을 넣는다."""
    from evolab.app.report import dialog

    open_for = getattr(dialog, "open_for", None)
    if callable(open_for):
        return open_for(parent, REPORT_REASON)
    from evolab.app.report.bundle import BundleOptions

    return dialog.open_report_dialog(parent, BundleOptions(reason=REPORT_REASON))


def _alive(widget: Any) -> Any:
    """C++ 쪽이 아직 살아 있는 위젯이면 그것, 아니면 None."""
    if widget is None:
        return None
    import shiboken6

    return widget if shiboken6.isValid(widget) else None


def _open_report_later(origin: weakref.ref[QWidget] | None) -> None:
    open_report(_alive(origin()) if origin is not None else None)


class _BoxWatcher(QObject):
    """상자의 자식: 상자가 닫히면 「오류 보고 묶음 만들기…」를 눌렀는지 본다.

    상자 신호에 부모 위젯을 붙잡은 람다를 걸지 않는다: 상자가 먼저 지워질 때 그 람다가 부모의 마지막 파이썬 참조를 놓으면
    파이썬이 가진 부모(부모 없는 창)가 상자 소멸 도중 지워져 프로세스가 죽는다(오프스크린 시험에서 확인).
    """

    def __init__(self, box: QMessageBox) -> None:
        super().__init__(box)
        box.finished.connect(self._done)
        owner = box.parentWidget()
        if owner is not None:
            QWidget.window(owner).installEventFilter(self)  # 상자를 띄운 창이 닫히면(숨으면) 상자도 닫는다

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        # 창이 닫히거나 프로그램이 숨기면(최소화 같은 창 관리자의 숨김 말고) 그 창의 오류 상자는 뜻이 없다.
        # 한 번도 보이지 않은 창(만드는 중에 오류)을 닫을 때는 숨김이 오지 않으므로 닫기 사건도 본다.
        kind = event.type()
        if kind == QEvent.Type.Close or (kind == QEvent.Type.Hide and not event.spontaneous()):
            box = self.parent()
            if isinstance(box, QMessageBox) and box.isVisible():
                box.close()
        return False

    @Slot(int)
    def _done(self, _result: int) -> None:
        box = self.parent()
        if not isinstance(box, QMessageBox) or box.clickedButton() is not getattr(box, "report_button", None):
            return
        owner = box.parentWidget()
        origin = weakref.ref(owner) if owner is not None else None
        # 상자가 다 닫힌 뒤(다음 이벤트 차례) 묶음 대화 상자를 연다
        QTimer.singleShot(0, functools.partial(_open_report_later, origin))


def _forget(key: int, *_destroyed: Any) -> None:
    _open_boxes.pop(key, None)


def show_error(parent: QWidget | None, title: str, error: BaseException | str, *, next_step: str | None = None,
               message: str | None = None) -> QMessageBox:
    """오류 상자를 막지 않고 띄운다(설계 5.5). next_step·message를 주면 explain의 그 칸 대신 쓴다. 띄운 상자를 돌려준다."""
    friendly = explain(error)
    if message is not None:
        friendly = replace(friendly, message=message)
    if next_step is not None:
        friendly = replace(friendly, next_step=next_step)
    _record(title, friendly)
    box = error_box(parent, title, friendly)
    _BoxWatcher(box)
    key = id(box)
    _open_boxes[key] = box
    # 목록에서는 상자가 실제로 지워질 때 뺀다(닫힐 때 빼면 부모 없는 상자의 마지막 참조가 닫는 도중에 사라진다)
    box.destroyed.connect(functools.partial(_forget, key))
    box.open()
    return box


def open_error_boxes() -> list[QMessageBox]:
    """지금 떠 있는 show_error 상자들(띄운 차례)."""
    return [b for b in _open_boxes.values() if _alive(b) is not None and b.isVisible()]


__all__ = ["BOX_NAME", "Friendly", "REPORT_REASON", "TRAINING_TEMPLATES", "detail_text", "error_box", "explain",
           "open_error_boxes", "open_report", "os_cause", "os_error_from_text", "os_error_from_traceback", "reason",
           "show_error", "split_training_message", "structure_next_step", "training_error"]

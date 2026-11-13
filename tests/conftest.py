import os

# 화면 테스트는 창을 실제로 띄우지 않고 메모리 안에서만 그린다(리눅스 컨테이너, 윈도우 CI 공통).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# 앱과 똑같이 numpy를 불러오기 전에 BLAS 스레드를 1개로 묶는다(명세: 모든 프로세스에서 BLAS 스레드 1개).
# 묶지 않으면 테스트 프로세스의 계산(유전자 지도의 eigh 등)이 앱과 다른 경로를 타고, 붐비는 기계에서 크게 느려진다.
from evolab.startup import limit_blas_threads  # noqa: E402

limit_blas_threads()

import sys  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _drop_app_wide_help_filters():
    """주 창(MainWindow)은 만들어질 때 앱 전체 이벤트 필터 둘(F1·Shift+F1, 따로 뜨는 창의 풍선 도움말)을 단다(v3.0
    app/help/setup.py). 테스트가 끝나면 떼어 다음 테스트가 앞 테스트의 주 창에 기대지 않게 한다(테스트 순서와 무관하게).
    그 모듈을 불러온 적이 없으면 아무것도 하지 않는다(Qt를 쓰지 않는 테스트는 Qt를 불러오지 않는다)."""
    yield
    context = sys.modules.get("evolab.app.help.context")
    tooltips = sys.modules.get("evolab.app.help.tooltips")
    if context is None and tooltips is None:
        return
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if not isinstance(app, QApplication):
        return
    if context is not None:
        context.uninstall_shortcuts(app)
    if tooltips is not None:
        tooltips.uninstall(app)


@pytest.fixture(autouse=True)
def _release_stuck_modifier_keys():
    """(6.2.0 통합) Ctrl+Enter처럼 수정 키를 누른 키 입력이 창을 닫으면 놓는 이벤트가 그 창에 닿지 못해 Qt의 수정 키 상태
    (QGuiApplication.keyboardModifiers)가 눌린 채 남는다. 그러면 다음 테스트의 selectRow가 Ctrl+클릭(골라 둔 줄에 더함)으로
    바뀐다(안내 창 테스트 뒤의 기록 목록 테스트). 테스트가 끝날 때 남은 수정 키를 놓는다. Qt를 불러온 적이 없으면 아무것도 하지 않는다."""
    yield
    if "PySide6.QtWidgets" not in sys.modules:
        return
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QWidget

    app = QApplication.instance()
    if not isinstance(app, QApplication) or app.keyboardModifiers() == Qt.KeyboardModifier.NoModifier:
        return
    probe = QWidget()
    probe.show()
    for key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
        QTest.keyRelease(probe, key)
    probe.close()
    probe.deleteLater()

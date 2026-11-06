import os

# 화면 테스트는 창을 실제로 띄우지 않고 메모리 안에서만 그린다(리눅스 컨테이너, 윈도우 CI 공통).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

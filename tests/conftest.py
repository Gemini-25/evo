import os

# 화면 테스트는 창을 실제로 띄우지 않고 메모리 안에서만 그린다(리눅스 컨테이너, 윈도우 CI 공통).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# 앱과 똑같이 numpy를 불러오기 전에 BLAS 스레드를 1개로 묶는다(명세: 모든 프로세스에서 BLAS 스레드 1개).
# 묶지 않으면 테스트 프로세스의 계산(유전자 지도의 eigh 등)이 앱과 다른 경로를 타고, 붐비는 기계에서 크게 느려진다.
from evolab.startup import limit_blas_threads  # noqa: E402

limit_blas_threads()

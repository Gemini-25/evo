"""화면과 명령줄에 보이는 모든 문구.

사용자에게 보이는 글은 이 파일에만 둔다. 다른 모듈의 문자열 리터럴에 한글이 들어가면
tests/test_rules.py가 실패한다.
"""

from typing import Final

APP_NAME: Final = "진화 실험실"
APP_TITLE: Final = "진화 실험실"

# 탭 이름. 순서가 곧 화면의 탭 순서다.
TAB_EDITOR: Final = "에디터"
TAB_LAB: Final = "실험실"
TAB_COMPARE: Final = "비교"
TAB_RECORDS: Final = "기록"

PLACEHOLDER_EDITOR: Final = "환경 에디터는 M5a에서 만듭니다."
PLACEHOLDER_LAB: Final = "실험실 화면은 M3에서 만듭니다."
PLACEHOLDER_COMPARE: Final = "비교 화면은 M6에서 만듭니다."
PLACEHOLDER_RECORDS: Final = "기록 화면은 M3에서 만듭니다."

# 명령줄
CLI_DESCRIPTION: Final = "진화 실험실. 명령 없이 실행하면 프로그램 창을 엽니다."
CLI_HELP_TRAIN: Final = "화면 없이 학습합니다."
CLI_HELP_BENCH: Final = "성능을 측정합니다."
CLI_HELP_PLAY: Final = "환경을 키보드로 직접 조종합니다."
CLI_HELP_SELFTEST: Final = "설치 상태를 점검합니다 (프로세스, 물리, 학습 라이브러리, 창)."
CLI_HELP_ENV_FILE: Final = "환경 파일 (.evoenv)"
CLI_HELP_METHOD: Final = "학습 방법"
CLI_HELP_SEED: Final = "실행 시드"
CLI_HELP_LOG: Final = "점검 결과를 쓸 파일 경로"
CLI_HELP_WORKERS: Final = "점검에 쓸 워커 수 (기본: 코어 수 - 1)"
CLI_HELP_NO_GUI: Final = "창 점검을 건너뜁니다."
CLI_NOT_IMPLEMENTED: Final = "'{command}' 명령은 {milestone}에서 구현합니다."

# 자체 점검
SELFTEST_TITLE: Final = "진화 실험실 자체 점검"
SELFTEST_RUNNING: Final = "진행"
SELFTEST_PASS: Final = "통과"
SELFTEST_FAIL: Final = "실패"
SELFTEST_SUMMARY_OK: Final = "모든 점검을 통과했습니다."
SELFTEST_SUMMARY_FAIL: Final = "실패한 점검이 있습니다: {names}"
SELFTEST_LOG_WRITTEN: Final = "점검 기록: {path}"
SELFTEST_CHECK_IMPORTS: Final = "라이브러리 불러오기"
SELFTEST_CHECK_PHYSICS: Final = "물리 결정성 (같은 시드 두 번)"
SELFTEST_CHECK_PROCESSES: Final = "프로세스 구조 (화면 → 학습 → 워커, 워커 1개와 N개 결과 비교)"
SELFTEST_CHECK_LEARNER: Final = "CMA-ES 학습 라이브러리"
SELFTEST_CHECK_GUI: Final = "창 만들기"
SELFTEST_ERR_MISMATCH: Final = "결과가 다릅니다: {detail}"
SELFTEST_ERR_TIMEOUT: Final = "{seconds}초 안에 응답이 없습니다."
SELFTEST_ERR_PROCESS_DIED: Final = "자식 프로세스가 결과 없이 끝났습니다 (종료 코드 {codes}). 창 없는 exe라면 원인이 {log}에 남습니다."
SELFTEST_ERR_NO_PROGRESS: Final = "점수가 나아지지 않았습니다: {detail}"
SELFTEST_ERR_TITLE: Final = "창 제목이 다릅니다: {title}"

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

# ---- 지형 자동 생성 (generators/terrain.py) ----
TERRAIN_ERR_SEED: Final = "지형 시드는 0 이상의 정수여야 합니다 (받은 값: {value})."
TERRAIN_ERR_RANGE: Final = "지형 범위가 잘못되었습니다: 오른쪽 끝({x_max})이 왼쪽 끝({x_min})보다 커야 하고 둘 다 유한한 수여야 합니다."
TERRAIN_ERR_SEGMENT: Final = "지형 구간 길이는 0보다 큰 유한한 수여야 합니다 (받은 값: {value})."
TERRAIN_ERR_TOO_MANY_VERTICES: Final = "지형 꼭짓점이 너무 많습니다: {count}개 (최대 {limit}개). 구간 길이를 늘리세요."
TERRAIN_ERR_BASE_Y: Final = "지형 기준 높이는 유한한 수여야 합니다 (받은 값: {value})."
TERRAIN_ERR_AMPLITUDE: Final = "지형 진폭은 0 이상의 유한한 수여야 합니다 (받은 값: {value})."
TERRAIN_ERR_ROUGHNESS: Final = "지형 거칠기는 0과 1 사이여야 합니다 (받은 값: {value})."
TERRAIN_ERR_PAD_X: Final = "착륙 패드 위치는 유한한 수여야 합니다 (받은 값: {value})."
TERRAIN_ERR_PAD_WIDTH_MISSING: Final = "착륙 패드 위치를 정했으면 패드 폭도 정해야 합니다."
TERRAIN_ERR_PAD_WIDTH: Final = "착륙 패드 폭은 0보다 큰 유한한 수여야 합니다 (받은 값: {value})."
TERRAIN_ERR_PAD_MARGIN: Final = "착륙 패드 양옆 여유는 0 이상의 유한한 수여야 합니다 (받은 값: {value})."
TERRAIN_ERR_PAD_TOO_WIDE: Final = "착륙 패드 폭 {width}m에 양옆 여유 {margin}m씩을 더하면 지형 폭 {world}m보다 넓습니다."
TERRAIN_ERR_HEIGHTMAP_SHAPE: Final = "높이맵의 x와 y는 길이가 같은 1차원 배열이고 점이 2개 이상이어야 합니다 (x {nx}, y {ny})."
TERRAIN_ERR_HEIGHTMAP_NOT_FINITE: Final = "높이맵에 유한하지 않은 값이 있습니다."
TERRAIN_ERR_HEIGHTMAP_ORDER: Final = "높이맵의 x는 계속 커져야 합니다 ({index}번째 점에서 커지지 않음)."
TERRAIN_ERR_HEIGHTMAP_PAD: Final = "착륙 패드 {pad}가 높이맵과 맞지 않습니다: 패드 양끝이 꼭짓점이어야 하고 그 사이 높이가 모두 같아야 합니다."

# ---- 수식 언어 (scoring/expr.py) ----
EXPR_ERR_AT: Final = "{message} ({column}번째 글자)"
EXPR_ERR_EMPTY: Final = "수식이 비어 있습니다."
EXPR_ERR_TOO_LONG: Final = "수식이 너무 깁니다 ({length}자, 최대 {limit}자)."
EXPR_ERR_TOO_COMPLEX: Final = "수식이 너무 복잡합니다 (요소 {count}개, 최대 {limit}개)."
EXPR_ERR_TOO_DEEP: Final = "수식이 너무 깊게 중첩되었습니다 (최대 {limit}단계)."
EXPR_ERR_SYNTAX: Final = "수식 문법이 맞지 않습니다."
EXPR_ERR_INCOMPLETE: Final = "수식이 중간에 끝났습니다 (끝에 값이나 괄호가 빠졌습니다)."
EXPR_ERR_IF_ELSE: Final = "'if'를 쓰려면 'else'도 있어야 합니다 (예: a if x > 0 else b)."
EXPR_ERR_UNCLOSED: Final = "여는 괄호 '{char}'가 닫히지 않았습니다."
EXPR_ERR_UNMATCHED: Final = "닫는 괄호 '{char}'의 짝이 없습니다."
EXPR_ERR_ASSIGN: Final = "'='는 쓸 수 없습니다. 같은지 비교하려면 '=='를 쓰세요."
EXPR_ERR_COMMENT: Final = "'#'(주석)은 수식에 쓸 수 없습니다."
EXPR_ERR_CONSTANT: Final = "수식에는 숫자와 True/False만 쓸 수 있습니다 ({what}은(는) 쓸 수 없습니다)."
EXPR_ERR_SYNTAX_KIND: Final = "{what}은(는) 수식에 쓸 수 없습니다."
EXPR_ERR_OPERATOR: Final = "연산자 '{op}'는 쓸 수 없습니다. 쓸 수 있는 연산자: + - * / ** < <= > >= == != and or not"
EXPR_ERR_ATTRIBUTE: Final = "'.'은 이름 사이에만 쓸 수 있습니다 (예: hull.speed)."
EXPR_ERR_DUNDER: Final = "'__'가 들어간 이름은 쓸 수 없습니다: '{name}'"
EXPR_ERR_UNKNOWN_NAME: Final = "모르는 이름 '{name}'입니다."
EXPR_ERR_UNKNOWN_FUNC: Final = "모르는 함수 '{name}'입니다. 쓸 수 있는 함수: {funcs}."
EXPR_DID_YOU_MEAN: Final = "'{hint}'을(를) 쓰려던 건가요?"
EXPR_ERR_CALL_TARGET: Final = "함수는 이름으로만 부를 수 있습니다 (예: sqrt(x))."
EXPR_ERR_KEYWORD: Final = "함수 인자에 이름을 붙일 수 없습니다 ('{name}=')."
EXPR_ERR_ARG_COUNT: Final = "{func}()에는 인자가 {expected}개 필요한데 {got}개를 넣었습니다."
EXPR_ERR_QUERY_UNAVAILABLE: Final = "{func}()는 여기서 쓸 수 없습니다."
EXPR_ERR_QUERY_ARG: Final = "{func}()의 인자는 부품 id여야 합니다 (예: contact(leg_l, ground))."
EXPR_ERR_NEED_BOOL: Final = "'{op}'에는 참/거짓 값이 필요한데 숫자가 왔습니다. 숫자는 비교로 바꾸세요 (예: x > 0)."
EXPR_ERR_NEED_BOOL_IF: Final = "'if' 조건에는 참/거짓 값이 필요한데 숫자가 왔습니다 (예: a if x > 0 else b)."
EXPR_ERR_NEED_NUM_COMPARE: Final = "비교 '{op}'에는 숫자가 필요한데 참/거짓 값이 왔습니다. 참/거짓은 그대로 쓰거나 not을 붙이세요."
EXPR_ERR_WANT_BOOL: Final = "조건식의 결과는 참/거짓이어야 합니다 (예: hull.speed < 0.5)."
EXPR_ERR_PHASE: Final = "'{name}'은(는) {phase} 값이라 {context}에서는 쓸 수 없습니다."
EXPR_PHASE_VALUE_BUILD: Final = "환경을 만들 때 정해지는"
EXPR_PHASE_VALUE_STEP: Final = "매 스텝 바뀌는"
EXPR_PHASE_VALUE_END: Final = "에피소드가 끝나야 정해지는"
EXPR_PHASE_CONTEXT_BUILD: Final = "환경을 만들 때 계산하는 수식"
EXPR_PHASE_CONTEXT_STEP: Final = "매 스텝 검사하는 조건"
EXPR_PHASE_CONTEXT_END: Final = "점수 수식"
EXPR_WHAT_STRING: Final = "문자열"
EXPR_WHAT_BYTES: Final = "바이트 문자열"
EXPR_WHAT_NONE: Final = "None"
EXPR_WHAT_COMPLEX: Final = "복소수"
EXPR_WHAT_ELLIPSIS: Final = "생략 기호(...)"
EXPR_WHAT_SUBSCRIPT: Final = "인덱스([ ])"
EXPR_WHAT_SLICE: Final = "슬라이스(:)"
EXPR_WHAT_LAMBDA: Final = "lambda 함수"
EXPR_WHAT_COMPREHENSION: Final = "컴프리헨션(for ... in)"
EXPR_WHAT_WALRUS: Final = "대입식(:=)"
EXPR_WHAT_STARRED: Final = "별표 인자(*, **)"
EXPR_WHAT_FSTRING: Final = "f-문자열"
EXPR_WHAT_CONTAINER: Final = "목록·튜플·집합·사전"
EXPR_WHAT_AWAIT: Final = "await·yield"
EXPR_WHAT_OTHER: Final = "이 구문"

# ---- 지형 자동 생성: 검토 보완 (generators/terrain.py) ----
TERRAIN_ERR_TOO_WIDE: Final = "지형이 너무 넓습니다: 폭 {width}m (최대 {limit}m)."
TERRAIN_ERR_HEIGHT_RANGE: Final = "지형 높이 범위가 너무 큽니다 (기준 높이 {base_y}, 진폭 {amplitude}). 더 작은 값을 쓰세요."

# ---- 환경 파일 검사 (spec/) ----
SPEC_INVALID_FILE: Final = "환경 파일에 오류가 {count}개 있습니다."
SPEC_BAD_JSON: Final = "JSON 형식이 잘못되었습니다 ({line}번째 줄, {col}번째 글자)."
SPEC_NOT_OBJECT: Final = "환경 파일의 맨 바깥은 { }로 감싼 객체여야 합니다."
SPEC_READ_FAILED: Final = "파일을 읽을 수 없습니다: {path} ({reason})"
SPEC_BAD_VERSION: Final = "version은 1 이상의 정수여야 합니다 (받은 값: {version})."
SPEC_NEWER_VERSION: Final = "이 파일은 더 새로운 프로그램에서 만들었습니다 (파일 버전 {version}). 프로그램을 업데이트하세요."
SPEC_GRID_NOT_YET: Final = "격자 세계는 2차에서 지원합니다. 지금은 physics2d만 쓸 수 있습니다."
SPEC_BOUNDS_ORDER: Final = "경계는 (왼쪽, 아래, 오른쪽, 위) 순서이고 오른쪽·위가 왼쪽·아래보다 커야 합니다."
SPEC_ID_RESERVED: Final = "'{id}'은(는) 예약된 이름이라 부품 id로 쓸 수 없습니다."
SPEC_ID_DUPLICATE: Final = "id '{id}'가 겹칩니다 ({other}에도 있음). 모든 부품 id는 서로 달라야 합니다."
SPEC_PARAM_RANGE_MISSING: Final = "진화 대상 매개변수 '{id}'에는 최솟값과 최댓값이 있어야 합니다."
SPEC_PARAM_RANGE_ORDER: Final = "매개변수 '{id}'의 최솟값이 최댓값보다 작아야 합니다."
SPEC_PARAM_VALUE_OUT: Final = "매개변수 '{id}'의 초깃값이 최솟값~최댓값 범위 밖입니다."
SPEC_RAND_RANGE: Final = "랜덤 값 '{id}'에는 low ≤ high인 low와 high가 있어야 합니다."
SPEC_RAND_NORMAL: Final = "정규분포 랜덤 값 '{id}'에는 mean과 0 이상인 std가 있어야 합니다."
SPEC_RAND_CHOICE: Final = "선택 랜덤 값 '{id}'에는 values 목록이 있어야 합니다."
SPEC_REF_BODY: Final = "물체 '{ref}'가 없습니다."
SPEC_REF_REGION: Final = "영역(또는 물체) '{ref}'가 없습니다."
SPEC_REF_BODY_OR_TAG: Final = "물체나 태그 '{ref}'가 없습니다. 태그는 tag.<이름>으로 씁니다."
SPEC_REF_JOINT: Final = "관절 '{ref}'가 없습니다."
SPEC_REF_TANK: Final = "연료 탱크 '{ref}'가 없습니다."
SPEC_REF_EVENT: Final = "사건 '{ref}'가 없습니다."
SPEC_REF_ACTUATOR: Final = "구동기 '{ref}'가 없습니다."
SPEC_REF_CHANNEL: Final = "채널 '{ref}'가 없습니다."
SPEC_NO_AGENT_BODY: Final = "구동기가 있는데 에이전트 물체가 하나도 없습니다."
SPEC_AGENT_NOT_DYNAMIC: Final = "에이전트 물체 '{id}'가 움직이지 않는 물체입니다."
SPEC_PLACE_BY_CHILD: Final = "place_by 관절 '{joint}'의 b가 이 물체('{body}')여야 합니다."
SPEC_PLACE_CYCLE: Final = "place_by 관절이 서로 꼬리를 물고 있습니다(순환)."
SPEC_JOINT_SAME_BODY: Final = "관절 '{id}'가 같은 물체를 양쪽에 연결합니다."
SPEC_NO_SENSORS: Final = "센서가 없습니다. 신경망 입력이 비게 됩니다."
SPEC_NO_ACTUATORS: Final = "구동기가 없습니다. 신경망이 할 수 있는 일이 없습니다."
SPEC_NO_SCORE: Final = "점수 블록과 점수 수식이 없습니다. 모든 개체의 점수가 0이 됩니다."
SPEC_NOT_YET: Final = "이 기능은 {milestone}단계에서 구현합니다."
SPEC_EVENT_CYCLE: Final = "사건 조건이 서로를(또는 자기 자신을) 참조합니다(순환)."
SPEC_QUERY_ARGS: Final = "{func}()에는 인자가 두 개 필요합니다."
SPEC_QUERY_FIRST_BODY: Final = "{func}()의 첫 인자는 물체 id여야 합니다 ('{name}'은(는) 물체가 아님)."
SPEC_QUERY_BAD_SECOND: Final = "{func}()의 둘째 인자 '{name}'이(가) 없습니다. {want}"
SPEC_QUERY_WANT_BODY_OR_TAG: Final = "물체 id나 tag.<이름>을 쓰세요."
SPEC_QUERY_WANT_REGION: Final = "영역 id를 쓰세요."
SPEC_QUERY_WANT_REGION_OR_BODY: Final = "영역이나 물체 id를 쓰세요."
SPEC_ERR_MISSING: Final = "꼭 있어야 하는 칸이 없습니다."
SPEC_ERR_EXTRA: Final = "모르는 칸입니다(오타일 수 있음)."
SPEC_ERR_LITERAL: Final = "다음 중 하나여야 합니다: {expected}"
SPEC_ERR_KIND: Final = "모르는 종류(kind)입니다. 다음 중 하나여야 합니다: {expected}"
SPEC_ERR_KIND_MISSING: Final = "종류(kind) 칸이 없습니다."
SPEC_ERR_PATTERN: Final = "id는 영문자·한글·밑줄로 시작하고 영문자·숫자·한글·밑줄만 쓸 수 있습니다."
SPEC_ERR_TOO_LONG: Final = "글자가 너무 깁니다(최대 {max}자)."
SPEC_ERR_TOO_SHORT: Final = "비어 있으면 안 됩니다."
SPEC_ERR_LIST_SHORT: Final = "항목이 적어도 {min}개 있어야 합니다."
SPEC_ERR_LIST_LONG: Final = "항목이 너무 많습니다(최대 {max}개)."
SPEC_ERR_RANGE: Final = "허용 범위를 벗어났습니다 (기준 {limit})."
SPEC_ERR_INT: Final = "정수여야 합니다."
SPEC_ERR_NUMBER: Final = "숫자여야 합니다."
SPEC_ERR_BOOL: Final = "true 또는 false여야 합니다."
SPEC_ERR_STRING: Final = "글자여야 합니다."
SPEC_ERR_LIST: Final = "목록 [ ]이어야 합니다."
SPEC_ERR_OBJECT: Final = "객체 {{ }}여야 합니다."
SPEC_ERR_NUM_OR_FORMULA: Final = "숫자이거나 '='로 시작하는 수식이어야 합니다 (예: 1.5 또는 \"=p.leg_len / 2\")."
SPEC_ERR_GENERIC: Final = "값이 올바르지 않습니다 ({type})."
END_TIMEOUT: Final = "시간 초과"

# ---- 런타임 (world/physics2d/) ----
RT_BODY_PARAM_COUNT: Final = "몸 수치 개수가 맞지 않습니다: {got}개를 받았는데 진화 대상 매개변수는 {want}개입니다."
RT_ACTION_SIZE: Final = "행동 벡터 길이가 맞지 않습니다: {got}개를 받았는데 구동기 출력은 {want}개입니다."
RT_DEGENERATE_SHAPE: Final = "물체 '{body}'의 모양이 잘못되었습니다(크기가 0이거나 꼬인 다각형)."

# ---- 직접 조종 (app/play/) ----
PLAY_TITLE: Final = "직접 조종 — {name}"
PLAY_BLOCKS_TITLE: Final = "점수 블록"
PLAY_SENSORS_TITLE: Final = "센서 값 (신경망 입력)"
PLAY_COL_BLOCK: Final = "블록"
PLAY_COL_VALUE: Final = "값"
PLAY_COL_SCORE: Final = "점수"
PLAY_COL_CHANNEL: Final = "채널"
PLAY_TOTAL: Final = "합계"
PLAY_RUNNING: Final = "진행 중"
PLAY_DONE: Final = "끝: {label} · 점수 {score:.2f}  (R 다시, N 새 시드)"
PLAY_SEED: Final = "시드: {seed}"
PLAY_HUD: Final = "시간 {t:.1f} / {max:.0f}초 · 속력 {speed:.2f} m/s"
PLAY_PAUSED: Final = "[일시정지]"
PLAY_REPLAYING: Final = "[기록 재생 중]"
PLAY_RESTART: Final = "다시 시작 (R)"
PLAY_NEW_SEED: Final = "새 시드 (N)"
PLAY_SAVE_RECORD: Final = "입력 기록 저장"
PLAY_LOAD_RECORD: Final = "입력 기록 재생"
PLAY_RECORD_FILTER: Final = "입력 기록 (*.evoplay)"
PLAY_SAVED: Final = "저장했습니다: {path}"
PLAY_RECORD_BAD: Final = "입력 기록을 읽을 수 없습니다: {reason}"
PLAY_REPLAY_TITLE: Final = "입력 기록 재생"
PLAY_REPLAY_SAME: Final = "재생 결과가 기록과 같습니다 (점수 {score:.6f})."
PLAY_REPLAY_DIFF: Final = "재생 결과가 기록과 다릅니다: 기록 {a}, 재생 {b}"
PLAY_REPLAY_ENV_DIFF: Final = "이 기록은 다른 내용의 환경 파일에서 만들었습니다. 결과가 다를 수 있습니다."
PLAY_KEY_LINE: Final = "{keys}: {label}"
PLAY_HELP: Final = "조종 키\n{keys}\n\nR 다시 시작 · N 새 시드 · Space 일시정지 · 마우스 휠 확대"
PLAY_OPEN_FAILED: Final = "환경 파일을 열 수 없습니다"
MENU_PLAY: Final = "직접 조종"
MENU_PLAY_LANDER: Final = "달 착륙선 몰아 보기"
MENU_PLAY_OPEN: Final = "환경 파일 열어서 몰아 보기…"
ENV_FILE_FILTER: Final = "환경 파일 (*.evoenv)"
CLI_HELP_PLAY_SEED: Final = "시험 시드 (없으면 무작위)"
CLI_HELP_PLAY_REPLAY: Final = "이 입력 기록을 화면에서 재생"
CLI_HELP_PLAY_VERIFY: Final = "이 입력 기록을 화면 없이 다시 돌려 기록된 결과와 같은지 확인 (종료 코드 0 같음, 1 다름, 2 읽을 수 없음)"
CLI_VERIFY_SAME: Final = "같음: 기록 점수 {recorded}, 재생 점수 {replayed} ({steps}스텝)"
CLI_VERIFY_DIFF: Final = "다름: 기록 점수 {recorded}, 재생 점수 {replayed} ({steps}스텝)"
SELFTEST_CHECK_ENV: Final = "내장 예제 환경 (착륙선 불러오기, 같은 시드 두 번 시뮬레이션)"
RT_BAD_MASS: Final = "물체 '{body}'의 질량이나 관성이 올바르지 않습니다(크기나 밀도가 0이거나 너무 큼)."
SPEC_RAND_HUGE: Final = "랜덤 값 '{id}'의 범위가 너무 크거나 유한하지 않습니다(±1e15 이내)."
SPEC_NEGATIVE: Final = "{name} 값은 0 이상이어야 합니다."

# ---- 환경 파일 검사: M1 검토 보완 ----
SPEC_NOT_UTF8: Final = "UTF-8로 저장된 파일이 아닙니다. 메모장이라면 '다른 이름으로 저장'에서 인코딩을 UTF-8로 골라 다시 저장하세요."
SPEC_TOO_DEEP: Final = "파일 구조가 너무 깊게 중첩되어 있습니다."
SPEC_JSON_NAN: Final = "JSON에 {value}가 있습니다. 숫자 칸에는 유한한 숫자만 쓸 수 있습니다."
SPEC_ERR_FINITE: Final = "유한한 숫자여야 합니다(NaN, 무한대 불가)."
SPEC_ERR_NAMES_OR_ALL: Final = "이름 목록 [\"a\", \"b\"]이거나 \"all\"이어야 합니다."
SPEC_ID_DUNDER: Final = "id '{id}'에 밑줄 두 개(__)가 들어 있습니다. 수식에서 쓸 수 없는 이름입니다."
SPEC_PLACE_BY_OWNER: Final = "place_by 관절 '{joint}'의 부모(a)는 '{body}'와 소유(agent/world)가 같은 물체여야 합니다(생성기 물체 불가)."
SPEC_JOINT_NO_DYNAMIC: Final = "관절 '{id}'의 양쪽 물체가 모두 움직이지 않는 물체입니다. 한쪽은 dynamic이어야 합니다."
SPEC_AGENT_COLLIDE: Final = "관절 '{id}'의 collide는 효과가 없습니다. 에이전트 물체끼리는 서로 부딪히지 않습니다."
SPEC_AGENT_SELF_CONTACT: Final = "'{a}'와 '{b}'는 모두 에이전트 물체라 서로 닿지 않습니다. 이 접촉 조건은 늘 거짓입니다."
SPEC_ACTUATOR_BODY: Final = "구동기는 에이전트의 움직이는 물체(dynamic)에만 달 수 있습니다 ('{ref}')."
SPEC_KEY_RESERVED: Final = "'{key}' 키는 직접 조종 창 제어용(R 다시 시작, N 새 시드, Space 일시정지)이라 구동기에 쓸 수 없습니다."
SPEC_KEY_UNKNOWN: Final = "모르는 키 이름 '{key}'입니다. 예: Up, Down, Left, Right, A~Z, 0~9, Shift, Control."

# ---- 직접 조종: M1 검토 보완 ----
DATA_DIR_NAME: Final = "진화실험실"
PLAY_NO_WEIGHT: Final = "—"
PLAY_BUILD_FAILED: Final = "시드 {seed}로 환경을 조립할 수 없습니다: {reason}"
PLAY_SAVE_FAILED: Final = "입력 기록을 저장하지 못했습니다: {reason}"
PLAY_RECORD_NOT_RECORD: Final = "진화 실험실 입력 기록 파일이 아닙니다."
PLAY_RECORD_ROW_SIZE: Final = "입력 기록의 행 길이가 출력 채널 수와 맞지 않습니다."
PLAY_RECORD_BROKEN: Final = "입력 기록 파일의 내용이 잘못되었습니다 ({detail})."
PLAY_RECORD_NO_RESULT: Final = "끝나기 전에 저장한 기록이라 비교할 결과가 없습니다."
PLAY_REPLAY_OUTPUTS_DIFF: Final = "기록의 출력 채널({recorded})이 이 환경({now})과 달라 재생할 수 없습니다."

# ---- 신경망·유전자 (policy/) ----
POLICY_BRAIN_SIZE: Final = "뇌 파라미터가 {got}개인데 이 신경망({shape})에는 {want}개가 필요합니다."
POLICY_GENOME_SIZE: Final = "유전자 길이가 {got}개인데 이 환경에는 {want}개(몸 수치 {body}개 + 뇌 파라미터 {brain}개)가 필요합니다."
POLICY_LAYOUT_BROKEN: Final = "유전자 구조 정보가 잘못되었습니다 ({detail})."
POLICY_GENOME_NAN: Final = "유전자에 숫자가 아닌 값(NaN)이 있어 몸 수치를 정할 수 없습니다 (몸 유전자 '{gene}')."
POLICY_BODY_RANGE_HUGE: Final = "몸 유전자 '{id}'의 범위({low} ~ {high})가 너무 넓어 계산할 수 없습니다. 최솟값과 최댓값을 줄이세요."

# ---- 학습 방법 (learners/) ----
LEARNER_ERR_POPSIZE: Final = "개체 수는 2 이상의 정수여야 합니다 (지금 {value})."
LEARNER_ERR_SIGMA: Final = "초기 시그마는 0보다 큰 유한한 숫자여야 합니다 (지금 {value})."
LEARNER_ERR_DIAGONAL_THRESHOLD: Final = "대각 공분산 전환 차원은 0 이상의 정수여야 합니다 (지금 {value})."
LEARNER_ERR_IPOP: Final = "재시작(IPOP) 설정은 켜기 또는 끄기여야 합니다 (지금 {value})."
LEARNER_ERR_SEED: Final = "학습 방법 시드는 0 이상 2³² 미만의 정수여야 합니다 (지금 {value})."
LEARNER_ERR_BOUNDED_DIMS: Final = "경계를 두는 차원 수는 0 이상의 정수여야 합니다 (지금 {value})."
LEARNER_ERR_BOUNDED_DIMS_TOO_MANY: Final = "경계를 두는 차원 수({value})가 유전자 차원({dim})보다 큽니다."
LEARNER_ERR_X0: Final = "초기 유전자는 유한한 숫자로 된 비어 있지 않은 1차원 벡터여야 합니다."
LEARNER_ERR_SCORES_SHAPE: Final = "점수 배열의 모양 {got}이 후보 수 ({count},)와 맞지 않습니다."
LEARNER_ERR_TELL_WITHOUT_ASK: Final = "후보를 받기(ask) 전에 점수를 넘겼습니다(tell)."
LEARNER_ERR_TELL_CANDIDATES: Final = "점수와 함께 넘긴 후보가 마지막으로 받은 후보(ask)와 다릅니다. 받은 후보를 같은 순서로 그대로 넘겨야 합니다."
LEARNER_ERR_STATE_KIND: Final = "다른 학습 방법('{got}')의 상태라 '{want}'에 불러올 수 없습니다."
LEARNER_ERR_STATE_FORMAT: Final = "학습 방법 상태 형식({got})을 읽을 수 없습니다."
LEARNER_ERR_STATE_DIM: Final = "저장된 학습 방법 상태의 유전자 차원({got})이 지금 환경의 차원({want})과 다릅니다."
LEARNER_ERR_STATE_BROKEN: Final = "저장된 학습 방법 상태를 읽을 수 없습니다 (다른 버전에서 만들었거나 손상됨: {reason})."

# ---- 실행 기록 저장소 (runs/) ----
RUNS_DEFAULT_FOLDER_NAME: Final = "실행"
RUNS_NOT_A_RUN: Final = "실행 기록 폴더가 아닙니다(run.json이 없습니다): {path}"
RUNS_INFO_BROKEN: Final = "실행 정보 파일을 읽을 수 없습니다: {path} ({detail})"
RUNS_NEWER_FORMAT: Final = "새 버전 프로그램에서 만든 실행 기록입니다(형식 {got}, 이 프로그램은 {want}까지 읽음). 프로그램을 업데이트하세요."
RUNS_BAD_POLICY: Final = "체크포인트 보관 설정이 잘못되었습니다. 최근 개수는 1 이상, 주기는 0 이상의 정수여야 합니다 (지금 최근 {keep_last}, 주기 {keep_every})."
RUNS_BAD_GENERATION: Final = "세대 번호는 0 이상의 정수여야 합니다 (지금 {value})."
RUNS_FILE_BUSY: Final = "다른 프로그램이 파일을 쓰고 있어 바꿀 수 없습니다: {path}. 잠시 뒤 다시 시도하세요."
RUNS_ENV_VERSION_MISSING: Final = "환경 버전 '{version}'이 이 실행 기록에 없습니다."
RUNS_ENV_VERSION_CHANGED: Final = "환경 사본의 내용이 기록된 해시와 다릅니다: {path}. 실행 폴더 안의 환경 사본은 고치지 마세요."
RUNS_STATS_KEYS: Final = "통계 열이 첫 줄과 다릅니다. 빠진 열: {missing} / 새 열: {extra}"
RUNS_STATS_NOT_UTF8: Final = "통계 파일이 UTF-8이 아닙니다: {path}. 엑셀에서 저장했다면 'CSV UTF-8(쉼표로 분리)' 형식으로 다시 저장하세요."
RUNS_STATS_LOCKED: Final = "통계 파일이 다른 프로그램(엑셀 등)에서 열려 있어 고칠 수 없습니다: {path}. 그 프로그램을 닫고 다시 시도하세요."
RUNS_STATS_PENDING: Final = "stats.csv가 다른 프로그램(엑셀 등)에서 열려 있어 통계 {count}줄을 stats.csv.pending에 따로 모아 두었습니다. 그 프로그램을 닫으면 다음 세대에 이어 씁니다."
RUNS_NO_CHECKPOINT: Final = "이어 갈 체크포인트가 없습니다: {path}"
RUNS_CHECKPOINT_MISSING: Final = "{generation}세대 체크포인트가 없습니다. 남아 있는 세대: {available}"
RUNS_CHECKPOINT_BROKEN: Final = "체크포인트 파일을 읽을 수 없습니다(손상되었거나 다른 버전에서 만듦): {path} ({detail})"
RUNS_CHAMPION_KIND_LABELS: Final = {"representative": "세대 대표", "champion": "챔피언"}
RUNS_CHAMPION_MISSING: Final = "{generation}세대 {kind} 파일이 없습니다."
RUNS_CHAMPION_BROKEN: Final = "챔피언 파일을 읽을 수 없습니다: {path} ({detail})"

# ---- 학습 엔진 (engine/) ----
POOL_TASK_FAILED: Final = "평가 중 오류가 났습니다. 환경 파일을 확인하세요.\n{detail}"
POOL_WORKER_DIED: Final = "평가 워커가 거듭 비정상 종료했습니다(종료 코드 {code})."
METHOD_NAMES: Final = {"cmaes": "CMA-ES", "ga": "유전 알고리즘", "sa": "담금질"}
TRAIN_STOP_REASONS: Final = {
    "max_generations": "최대 세대",
    "max_evaluations": "최대 시험 횟수",
    "max_time": "최대 시간",
    "target_score": "목표 점수 도달",
    "stagnation": "정체",
    "requested": "사용자가 멈춤",
}
TRAIN_STARTED: Final = "학습 시작: {name} · {method} · 유전자 {dim}차원 · 개체 {pop} · 시험 {trials}회 · 워커 {workers}개\n실행 폴더: {path}"
TRAIN_RESUMED: Final = "{gen}세대 체크포인트에서 이어갑니다: {path}"
TRAIN_PROGRESS: Final = "세대 {gen} · 시험 {evals:,}회 · 최고 {best:.1f} · 평균 {mean:.1f} · 챔피언(검증) {champ} · {elapsed:.0f}초 · 초당 {sps:,.0f}스텝"
TRAIN_FINISHED: Final = "학습 끝 ({reason}): {gen}세대, 시험 {evals:,}회, {champ}, {elapsed:.0f}초\n실행 폴더: {path}"
TRAIN_CHAMPION: Final = "챔피언 검증 점수 {score:.2f} ({gen}세대)"
TRAIN_NO_CHAMPION_YET: Final = "챔피언 없음(검증 전에 끝남: 학습 시험이 검증 주기만큼 쌓이지 않음)"
TRAIN_STOPPING: Final = "멈추는 중입니다. 이번 세대를 마치고 체크포인트를 남깁니다(한 번 더 누르면 바로 끝냄)."
TRAIN_ABORTED: Final = "바로 끝냈습니다. 마지막 체크포인트부터 이어갈 수 있습니다."
TRAIN_NEED_ENV: Final = "환경 파일이나 --resume 실행 폴더를 주세요."
JUDGE_OPEN_FAILED: Final = "실행 폴더를 열 수 없습니다: {reason}"
JUDGE_NO_CHAMPION: Final = "아직 챔피언이 없습니다(검증 전에 멈춘 실행)."
JUDGE_RESULT: Final = "판정({gen}세대 챔피언, 판정 시드 {n}개): 성공 {success}/{n} ({rate:.0f}%), 평균 점수 {mean:.2f}\n종료 원인: {ends}"
BENCH_TITLE: Final = "성능 측정: {name} · 개체 {pop} × 시험 {trials}회 · 유전자 {dim}차원 (무작위 유전자, 1세대 기준)"
BENCH_SINGLE: Final = "단일 코어: 초당 {sps:,.0f}스텝 ({us:.1f}µs/스텝), 신경망 계산 비율 {nn:.1f}%, {steps:,}스텝 {sec:.2f}초"
BENCH_POOL: Final = "워커 {workers}개: 세대당 {sec:.2f}초 (중앙값), 초당 {sps:,.0f}스텝"
BENCH_WORST: Final = "모든 시험이 최대 시간을 다 쓰는 최악의 세대: {steps:,}스텝 ≈ 워커 {workers}개로 {sec:.2f}초 (단일 코어 속도 기준 추정)"
CLI_HELP_RESUME: Final = "이 실행 폴더의 마지막 체크포인트에서 이어서 학습"
CLI_HELP_OUT: Final = "실행 폴더를 만들 곳 (기본: 문서\\진화실험실\\runs)"
CLI_HELP_TRAIN_WORKERS: Final = "평가 워커 수 (기본: 물리 코어 수 - 1)"
CLI_HELP_POPSIZE: Final = "개체 수"
CLI_HELP_SIGMA: Final = "초기 시그마"
CLI_HELP_TRIALS: Final = "개체당 시험 횟수 (기본: 환경 파일 값)"
CLI_HELP_MAX_TIME: Final = "최대 학습 시간(초)"
CLI_HELP_MAX_GENS: Final = "최대 세대"
CLI_HELP_MAX_EVALS: Final = "최대 학습 시험 횟수"
CLI_HELP_TARGET: Final = "목표 점수(챔피언 검증 점수 기준)"
CLI_HELP_STAGNATION: Final = "이 세대 수 동안 검증 점수가 오르지 않으면 멈춤"
CLI_HELP_BENCH_WORKERS: Final = "잴 워커 수 목록 (예: 1,2,3)"
CLI_HELP_BENCH_GENS: Final = "워커 수마다 잴 세대 수"
CLI_HELP_JUDGE: Final = "실행의 챔피언을 판정 시드로 평가 (완료 기준)"
CLI_HELP_RUN_DIR: Final = "실행 폴더"
CLI_HELP_JUDGE_SEEDS: Final = "판정 시드 개수"
CLI_HELP_JUDGE_GEN: Final = "이 세대까지의 마지막 챔피언으로 판정 (기본: 마지막 챔피언)"
CLI_HELP_VALIDATE_EVERY: Final = "학습 시험이 이만큼 쌓일 때마다 챔피언 검증 (기본 1000)"
TRAIN_CANNOT_START: Final = "학습을 시작할 수 없습니다: {reason}"
CLI_HELP_DIAGONAL: Final = "유전자 차원이 이 값보다 크면 대각 공분산(sep-CMA) 사용 (기본 1000)"

# M3 학습 중 변경 (lead)
TRAIN_CHANGE_NOT_ALLOWED: Final = "학습 중에는 바꿀 수 없는 설정입니다: {keys}. 새 실행으로 시작하세요."
TRAIN_ENV_LAYOUT_CHANGED: Final = (
    "센서·구동기·몸 수치·뇌 구조가 바뀌어 유전자 차원이 달라졌습니다. 이어서 학습할 수 없으니 새 실행으로 시작하세요."
)
TRAIN_ENV_COLUMNS_CHANGED: Final = "점수 블록이나 종료 조건 목록이 바뀌었습니다. 이어서 학습할 수 없으니 새 실행으로 시작하세요."
TRAIN_CONFIG_INVALID: Final = "학습 설정 값이 잘못되었습니다: {name} = {value}"

# M3 학습 프로세스 (lead)
SESSION_CRASHED: Final = "학습 프로세스에 예상하지 못한 오류가 났습니다: {reason}"
SESSION_SAVE_FAILED: Final = "학습을 끝내며 저장하지 못했습니다: {reason}"
SESSION_STILL_STOPPED: Final = "정지 조건({reason})에 걸려 있습니다. 정지 조건을 바꾼 뒤 이어 하세요."
SESSION_PAUSE_FIRST: Final = "학습을 멈춘 뒤에 바꿀 수 있습니다."
SESSION_LOST: Final = "학습 프로세스가 갑자기 끝났습니다(종료 코드 {code}). 마지막 체크포인트에서 이어 할 수 있습니다."

# M3 spectator canvas
SPEC_NO_SCENE: Final = "학습을 시작하면 집단이 여기에 나타납니다."
SPEC_LEGEND_CHAMPION: Final = "챔피언"
SPEC_LEGEND_TOP: Final = "상위 10%"
SPEC_LEGEND_REST: Final = "나머지 개체"

# M3 통계 대시보드 (app/lab/dashboard.py)
DASH_TILE_TITLES: Final = {
    "trend": "점수 추이",
    "heatmap": "점수 분포 지도",
    "diversity": "탐색 폭과 다양성",
    "blocks": "점수 블록 기여도",
    "ends": "종료 원인 비율",
    "perf": "처리 성능",
    # v1.2 「보는 재미」 쪽
    "genemap": "유전자 지도",
    "brain": "뇌 들여다보기",
    "density": "궤적 밀도",
    "actions": "행동 분포",
    "body": "몸 수치 추이",
}
DASH_AXIS_LABEL: Final = "가로축"
DASH_AXIS_NAMES: Final = {"generation": "세대", "evaluations": "누적 시험 횟수", "elapsed": "경과 시간(초)"}
DASH_AXIS_TOOLTIP: Final = "점수 추이, 탐색 폭, 종료 원인, 처리 성능 그래프의 가로축을 바꿉니다."
DASH_TILES_BUTTON: Final = "타일"
DASH_TILES_TOOLTIP: Final = "보일 그래프와 배치를 고릅니다."
DASH_TILES_COLUMNS: Final = "한 줄에 놓을 그래프 수"
DASH_TILES_COLUMNS_ITEM: Final = "{n}개"
DASH_TILES_RESET: Final = "기본 배치로 되돌리기"
DASH_TILE_MOVE_EARLIER: Final = "앞으로 옮기기"
DASH_TILE_MOVE_LATER: Final = "뒤로 옮기기"
DASH_TILE_HIDE: Final = "이 그래프 숨기기 ('타일' 단추에서 다시 켬)"
DASH_SCORE: Final = "점수"
DASH_TREND_CHAMPION: Final = "챔피언(검증)"
DASH_TREND_BEST: Final = "최고"
DASH_TREND_MEDIAN: Final = "중앙값"
DASH_TREND_MEAN: Final = "평균"
DASH_TREND_MIN: Final = "최저"
DASH_TREND_BAND: Final = "25~75% 구간"
DASH_MY_BEST: Final = "내 최고 기록"
DASH_HEATMAP_FRACTION: Final = "개체 비율"
DASH_HEATMAP_HINT: Final = "색이 진할수록 그 점수대에 개체가 많습니다."
DASH_HEATMAP_HOVER: Final = "{gen}세대 · 점수 {lo}~{hi} · 개체 {pct:.0f}%"
DASH_SIGMA: Final = "시그마"
DASH_DIVERSITY: Final = "유전자 표준편차 평균"
DASH_DISTANCE: Final = "개체 간 평균 거리"
DASH_SPREAD_AXIS: Final = "퍼짐"
DASH_DISTANCE_AXIS: Final = "거리"
DASH_BLOCKS_MODES: Final = {"rep": "세대 대표", "mean": "집단 평균"}
DASH_BLOCKS_MODE_TOOLTIP: Final = "세대 대표(그 세대 최고 개체)와 집단 평균 중 무엇의 블록별 점수를 쌓을지 고릅니다."
DASH_BLOCKS_TOTAL: Final = "합계"
DASH_BLOCKS_AXIS: Final = "점수 기여"
DASH_BLOCKS_NONE: Final = "이 환경은 점수를 수식으로 계산해서 블록별 기여도가 없습니다."
DASH_OTHER: Final = "기타"
DASH_ENDS_AXIS: Final = "시험 비율(%)"
DASH_PERF_GEN_SECONDS: Final = "세대당 시간(초)"
DASH_PERF_STEPS: Final = "초당 시뮬레이션 스텝"
DASH_PERF_UTIL: Final = "워커 사용률(%)"
DASH_PERF_AXES: Final = {"gen_seconds": "초", "steps_per_second": "스텝/초", "worker_utilization": "%"}
DASH_MARK_LABELS: Final = {"resume": "이어하기", "env_change": "환경 수정", "config_change": "설정 변경", "branch": "분기",
                            "restart": "새로 시작", "qualify_next": "예선 다음", "qualify_pick": "예선 끝"}
DASH_MARK_TOOLTIP: Final = "{gen}세대: {label}"
DASH_HOVER_X: Final = {"generation": "{gen}세대", "evaluations": "{gen}세대 · 시험 {x}회", "elapsed": "{gen}세대 · {x}초"}
# M3 records (기록 화면, 실행 목록·분기·삭제)
REC_TITLE: Final = "기록"
REC_COL_ENV: Final = "환경"
REC_COL_METHOD: Final = "방법"
REC_COL_STARTED: Final = "시작 시각"
REC_COL_GENERATIONS: Final = "세대 수"
REC_COL_BEST: Final = "최고 점수"
REC_COL_MEMO: Final = "메모"
REC_REFRESH: Final = "새로 고침"
REC_OPEN: Final = "열기"
REC_OPEN_TIP: Final = "통계와 챔피언 재생을 봅니다(목록에서 두 번 눌러도 열립니다)."
REC_RESUME: Final = "이어하기"
REC_RESUME_TIP: Final = "남아 있는 체크포인트에서 학습을 이어 갑니다. 최신이 아닌 체크포인트를 고르면 새 실행으로 분기합니다."
REC_CLONE: Final = "복제해서 새 실행"
REC_CLONE_TIP: Final = "이 실행의 환경과 학습 설정으로 새 실행을 시작합니다."
REC_EDIT_MEMO: Final = "메모 편집"
REC_DELETE: Final = "삭제"
REC_DELETE_TIP: Final = "실행 폴더를 통째로 지웁니다. 되돌릴 수 없습니다."
REC_FOLDER: Final = "실행 {count}개 · {path}"
REC_EMPTY: Final = "아직 실행 기록이 없습니다. 실험실에서 학습을 시작하면 여기에 나타납니다."
REC_NONE: Final = "—"
REC_UNREADABLE: Final = "읽을 수 없음"
REC_ENV_BRANCHED: Final = "{env} · {generation}세대 분기"
REC_TIP_FOLDER: Final = "폴더: {path}"
REC_TIP_PARENT: Final = "'{run}' 실행의 {generation}세대 체크포인트에서 분기한 실행입니다."
REC_RESUME_TITLE: Final = "이어하기"
REC_RESUME_PICK: Final = "어느 체크포인트에서 이어 갈까요?\n최신이 아닌 체크포인트를 고르면 그 세대까지의 기록을 복사한 새 실행 폴더에서 이어 갑니다."
REC_CHECKPOINT_LATEST: Final = "{generation}세대 (최신)"
REC_CHECKPOINT_BRANCH: Final = "{generation}세대 (새 실행으로 분기)"
REC_BRANCH_CONFIRM: Final = "{generation}세대는 최신 체크포인트가 아니므로 새 실행 폴더를 만들어 거기서 이어 갑니다.\n그 세대까지의 통계, 챔피언, 체크포인트를 복사하고, 원래 실행은 그대로 남습니다."
REC_RESUME_FAILED: Final = "이어 갈 수 없습니다: {reason}"
REC_RESUME_BUSY: Final = "이미 학습 중인 실행입니다."
REC_BRANCH_FOLDER: Final = "{stamp}_{env}_{method}_{generation}세대 분기"
REC_DELETE_TITLE: Final = "실행 삭제"
REC_DELETE_CONFIRM: Final = "'{name}' 실행을 삭제할까요?\n통계, 체크포인트, 챔피언, 내보낸 파일까지 폴더 전체가 지워지며 되돌릴 수 없습니다."
REC_DELETE_FAILED: Final = "삭제할 수 없습니다: {reason}"
REC_DELETE_BUSY: Final = "지금 학습 중인 실행은 삭제할 수 없습니다. 학습을 멈춘 뒤 다시 시도하세요."
REC_DELETE_OUTSIDE: Final = "실행 기록 폴더({runs_dir}) 바로 아래의 실행만 삭제할 수 있습니다: {path}"
REC_DELETE_NOT_RUN: Final = "실행 폴더가 아니어서 삭제하지 않습니다(run.json이 없습니다): {path}"
REC_DELETE_LINK: Final = "바로 가기(링크) 폴더는 삭제하지 않습니다: {path}"
REC_DELETE_FILES_BUSY: Final = "다른 프로그램(학습, 엑셀, 탐색기 등)이 실행 폴더의 파일을 쓰고 있어 삭제할 수 없습니다: {path}. 그 프로그램을 닫고 다시 시도하세요."
REC_MEMO_FAILED: Final = "메모를 저장할 수 없습니다: {reason}"

# M3 실험실 화면 (lead)
LAB_ENV_GROUP: Final = "환경"
LAB_ENV_NONE: Final = "환경을 고르세요"
LAB_ENV_BUILTIN: Final = "내장 예제"
LAB_ENV_OPEN: Final = "파일 열기…"
LAB_ENV_RELOAD: Final = "환경 다시 불러오기"
LAB_ENV_RELOAD_TIP: Final = "멈춘 상태에서 환경 파일을 다시 읽어 이어서 학습합니다(유전자 차원이 같을 때만)."
LAB_SETTINGS_GROUP: Final = "학습 설정"
LAB_METHOD: Final = "학습 방법"
LAB_SEED: Final = "실행 시드"
LAB_POPSIZE: Final = "개체 수"
LAB_SIGMA: Final = "시그마"
LAB_TRIALS: Final = "시험 횟수"
LAB_TRIALS_ENV: Final = "환경 파일 값"
LAB_WORKERS: Final = "워커 수"
LAB_STOP_GROUP: Final = "정지 조건"
LAB_MAX_GENERATIONS: Final = "최대 세대"
LAB_MAX_MINUTES: Final = "최대 시간(분)"
LAB_TARGET: Final = "목표 점수"
LAB_STAGNATION: Final = "정체 세대"
LAB_NONE: Final = "없음"
LAB_APPLY: Final = "설정 적용"
LAB_APPLY_TIP: Final = "멈춘 상태에서 정지 조건, 시험 횟수, 자동 재시작 설정을 바꿉니다."
LAB_CONTROL_GROUP: Final = "진행"
LAB_START: Final = "새로 시작"
LAB_PAUSE: Final = "멈춤"
LAB_CONTINUE: Final = "이어서"
LAB_STEP: Final = "한 세대만"
LAB_END: Final = "학습 끝내기"
LAB_VIEW_GROUP: Final = "보기"
LAB_SPECTATE: Final = "관전"
LAB_FAST: Final = "고속"
LAB_SPEED: Final = "재생 배속"
LAB_SPEED_ITEM: Final = "{n}배"
LAB_FAST_INTERVAL: Final = "챔피언 재생 간격(세대)"
LAB_TIMELINE: Final = "세대"
LAB_TIMELINE_PLAY: Final = "이 세대 대표 재생"
LAB_VERSUS: Final = "1세대 대표와 지금 챔피언"
LAB_LIVE: Final = "실시간 보기"
LAB_STATUS_IDLE: Final = "학습 전"
LAB_STATUS: Final = "{state} · {gen}세대 · 시험 {evals:,}회 · 챔피언(검증) {champ}"
LAB_STATE_NAMES: Final = {"running": "학습 중", "paused": "멈춤", "finished": "끝남", "starting": "시작하는 중", "idle": "학습 전",
                          "closed": "학습 프로세스 없음"}
LAB_STOPPED_BECAUSE: Final = "학습 끝: {reason}"
LAB_HUD_LIVE: Final = "{gen}세대 · 시험 0번 · {t:.1f}초 · {speed}배"
LAB_HUD_REPLAY: Final = "{label} · 전시 시드 · {t:.1f}초"
LAB_HUD_FAST: Final = "고속 모드 · {gen}세대 챔피언 · 전시 시드 · {t:.1f}초"
LAB_LABEL_REP: Final = "{gen}세대 대표"
LAB_LABEL_CHAMPION: Final = "지금 챔피언 ({gen}세대)"
LAB_NO_CHAMPION: Final = "아직 챔피언이 없습니다(첫 검증 전)."
LAB_ERROR_TITLE: Final = "학습 오류"
LAB_CHANGE_FAILED: Final = "바꾸지 못했습니다"
LAB_CHANGE_DONE: Final = "{gen}세대 뒤부터 적용됩니다."
LAB_OPEN_ENV_FAILED: Final = "환경 파일을 열 수 없습니다"
LAB_STARTED_RUN: Final = "실행 폴더: {path}"
LAB_RESUME: Final = "이 실행 이어서 학습"
LAB_CLOSE_WAIT: Final = "학습을 정리하는 중입니다…"
LAB_INSPECT_TITLE: Final = "개체 {index} 들여다보기 · 점수 {score:.2f} · {end}"
LAB_INSPECT_SENSORS: Final = "센서(정규화 전 → 신경망 입력)"
LAB_INSPECT_LAYERS: Final = "신경망 활성화"
LAB_INSPECT_LAYER: Final = "층 {k} ({n}개)"
LAB_INSPECT_OUTPUTS: Final = "출력"
LAB_INSPECT_CLOSE: Final = "닫기"
LAB_INSPECT_TIME: Final = "{t:.2f}초 / {total:.2f}초"
LAB_FINISHED_INFO: Final = "학습이 끝났습니다 ({reason}). 정지 조건을 바꾸고 '설정 적용' 뒤 '이어서'로 계속할 수 있습니다."
# M3 records (추가)
REC_STATS_BROKEN: Final = "통계 파일을 읽을 수 없습니다: {path} ({detail})"

# M3 주 창 (lead)
MAIN_LAB_BUSY: Final = "실험실에서 학습이 진행 중입니다. '학습 끝내기'를 누른 뒤 다시 하세요."

# M4 환경 검사 (lead)
SPEC_REF_CHECKPOINTS: Final = "체크포인트 '{ref}'이(가) 없습니다."
SPEC_TRACK_CENTERLINE: Final = "직접 그린 트랙에는 중심선 꼭짓점이 3개 이상 있어야 합니다."
SPEC_TRACK_ONE_SET: Final = "자동 생성 트랙(학습 트랙 세트)은 환경 하나에 하나만 둘 수 있습니다."
RT_TOO_MANY_RAY_TARGETS: Final = "레이 센서의 감지 대상 조합이 너무 많습니다(30가지까지)."

# M3 통계 대시보드 (2)
DASH_STAGNATION_LEGEND: Final = "정체({n}세대 이상 그대로)"

# M4 직접 조종 차량 키 (lead)
PLAY_KEY_FORWARD: Final = "{name} 앞으로"
PLAY_KEY_BACK: Final = "{name} 뒤로"
PLAY_KEY_LEFT: Final = "{name} 왼쪽"
PLAY_KEY_RIGHT: Final = "{name} 오른쪽"
CLI_HELP_IPOP: Final = "pycma 종료 조건이 차면 개체 수를 두 배로 늘려 다시 시작(IPOP-CMA-ES, 자동 재시작 --restart와 다름)"

# M4 트랙 생성기 (generators/track.py)
TRACK_ERR_SEED: Final = "트랙 시드는 0 이상의 정수여야 합니다 (받은 값: {value})."
TRACK_ERR_SEGMENTS: Final = "트랙 구간 수는 3 이상 {limit} 이하의 정수여야 합니다 (받은 값: {value})."
TRACK_ERR_CURVATURE: Final = "트랙 곡률은 0과 1 사이여야 합니다 (받은 값: {value})."
TRACK_ERR_LENGTH_RANGE: Final = "트랙 길이 범위가 잘못되었습니다: (최소, 최대) 둘 다 0보다 큰 유한한 수이고 최소가 최대보다 크지 않아야 합니다 (받은 값: {value})."
TRACK_ERR_WIDTH: Final = "트랙 폭은 0보다 큰 유한한 수여야 합니다 (받은 값: {value})."
TRACK_ERR_SPACING: Final = "트랙 중심선 점 간격은 0보다 큰 유한한 수여야 합니다 (받은 값: {value})."
TRACK_ERR_CHECKPOINTS: Final = "체크포인트 수는 1 이상 {limit} 이하의 정수여야 합니다 (받은 값: {value})."
TRACK_ERR_TOO_MANY_POINTS: Final = "트랙 중심선 점이 너무 많습니다: 길이 {length:g}m를 간격 {spacing:g}m로 나누면 {limit}개를 넘습니다. 간격을 늘리거나 트랙을 줄이세요."
TRACK_ERR_GENERATE_FAILED: Final = "트랙을 만들지 못했습니다 (시드 {seed}, {attempts}번 시도). 마지막 이유: {reason} 폭을 줄이거나, 길이 범위를 늘리거나, 구간 수를 줄여 보세요."
TRACK_ERR_POINTS_SHAPE: Final = "트랙 중심선은 (x, y) 꼭짓점의 목록이어야 합니다."
TRACK_ERR_TOO_FEW_POINTS: Final = "트랙 중심선에는 서로 다른 꼭짓점이 3개 이상 있어야 합니다 (지금 {count}개)."
TRACK_ERR_TOO_MANY_DRAWN: Final = "트랙 중심선 꼭짓점이 너무 많습니다: {count}개 (최대 {limit}개)."
TRACK_ERR_POINTS_NOT_FINITE: Final = "트랙 중심선 꼭짓점에 유한하지 않은 값이 있습니다."
TRACK_ERR_SELF_INTERSECT: Final = "트랙 중심선이 스스로 교차합니다. 8자처럼 꼬이지 않은 닫힌 길로 그리세요."
TRACK_ERR_TOO_TIGHT: Final = "트랙 굽이가 폭에 비해 너무 급합니다: ({x:.1f}, {y:.1f}) 근처 굽이의 반지름이 {radius:.2f}m인데, 폭 {width:g}m 트랙에서는 {limit:.2f}m보다 커야 합니다. 굽이를 완만하게 그리거나 폭을 줄이세요."
TRACK_ERR_WALLS_CROSS: Final = "트랙의 서로 다른 부분이 너무 가까워 벽이 겹칩니다. 길 사이를 더 띄우거나 폭을 줄이세요."
TRACK_ERR_SET_COUNT: Final = "학습 트랙 수는 0 이상의 정수여야 합니다 (받은 값: {value})."
TRACK_ERR_SET_INDEX: Final = "학습 트랙 번호는 0 이상의 정수여야 합니다 (받은 값: {value})."
TRACK_ERR_DATA: Final = "트랙 자료가 잘못되었습니다: 중심선·벽은 같은 수(3개 이상)의 유한한 (x, y) 점이고, 호 길이와 체크포인트·출발 위치가 수여야 합니다."

# M5a 에디터 문서 (lead)
ED_NEW_NAME: Final = "새 환경"
ED_UNTITLED: Final = "이름 없음"
ED_DEFAULT_TIMEOUT_LABEL: Final = "시간 초과"
ED_UNDO_SET: Final = "값 바꾸기"
ED_UNDO_ADD: Final = "부품 추가"
ED_UNDO_REMOVE: Final = "부품 지우기"
ED_UNDO_RENAME: Final = "이름 바꾸기"
ED_UNDO_PASTE: Final = "붙여넣기"
ED_UNDO_EVOLVE_ON: Final = "진화 대상으로"
ED_UNDO_EVOLVE_OFF: Final = "진화 대상 끄기"
ED_CHECK_UNATTACHED: Final = "에이전트 물체 '{id}'이(가) 관절로 다른 물체에 붙어 있지 않습니다."
ED_CHECK_OVERLAP: Final = "물체 '{a}'와(과) '{b}'이(가) 시작할 때 겹쳐 있습니다."

# M6 GA/SA
LEARNER_GA_ERR_ELITES: Final = "엘리트 수는 0 이상이고 개체 수({popsize})보다 작은 정수여야 합니다 (지금 {value})."
LEARNER_GA_ERR_TOURNAMENT: Final = "토너먼트 크기는 1 이상이고 개체 수({popsize}) 이하인 정수여야 합니다 (지금 {value})."
LEARNER_GA_ERR_CROSSOVER: Final = "교차 확률은 0 이상 1 이하의 숫자여야 합니다 (지금 {value})."
LEARNER_GA_ERR_MUTATION_RATE: Final = "돌연변이 확률은 0 이상 1 이하의 숫자여야 합니다 (지금 {value})."
LEARNER_GA_ERR_MUTATION_SIGMA: Final = "돌연변이 폭은 0보다 큰 유한한 숫자여야 합니다 (지금 {value})."
LEARNER_GA_ERR_SIGMA_DECAY: Final = "돌연변이 폭 감쇠는 0보다 크고 1 이하인 숫자여야 합니다 (지금 {value})."
LEARNER_SA_ERR_NEIGHBORS: Final = "이웃 수는 1 이상의 정수여야 합니다 (지금 {value})."
LEARNER_SA_ERR_STEP: Final = "이웃 폭은 0보다 큰 유한한 숫자여야 합니다 (지금 {value})."
LEARNER_SA_ERR_TEMPERATURE: Final = "초기 온도는 비워 두거나(첫 이웃들 점수의 표준편차) 0 이상의 유한한 숫자여야 합니다 (지금 {value})."
LEARNER_SA_ERR_COOLING: Final = "온도 감쇠율은 0보다 크고 1 이하인 숫자여야 합니다 (지금 {value})."

# M5b 모터·언덕 (lead)
SPEC_MOTOR_JOINT_KIND: Final = "모터는 회전 관절(pivot, rot_spring)에만 달 수 있습니다: '{ref}'"
SPEC_MOTOR_NOT_AGENT: Final = "모터를 단 관절 '{ref}'은(는) 에이전트의 움직이는 물체에 붙어 있어야 합니다."
TERRAIN_HILLS_RANGE: Final = "언덕길 x 범위가 잘못되었습니다: {x_min} .. {x_max}"
PLAY_KEY_CCW: Final = "{name} 반시계"
PLAY_KEY_CW: Final = "{name} 시계"

# M6 재현 묶음·CSV (lead)
BUNDLE_README: Final = (
    "진화 실험실 재현 묶음\n\n"
    "run.json(학습 설정, 실행 시드, 프로그램·의존성 버전), envs\\(환경 버전 사본), events.jsonl(세대 번호가 붙은 변경 기록),\n"
    "stats.csv(원래 통계)가 들어 있습니다.\n\n"
    "다시 돌리기: evolab-cli.exe reproduce 이_파일.zip\n"
    "같은 PC·같은 빌드에서는 통계가 비트 단위로 같고, 다른 PC에서는 통계적으로 재현됩니다.\n"
)
BUNDLE_NOT_BUNDLE: Final = "재현 묶음 파일이 아닙니다: {path}"
BUNDLE_READ_FAILED: Final = "재현 묶음을 읽을 수 없습니다: {path} ({reason})"
BUNDLE_MEMO: Final = "재현 묶음에서 다시 돌림(원래 실행: {run})"
BUNDLE_DONE: Final = "재현 묶음을 만들었습니다: {path}"
BUNDLE_PROGRESS: Final = "다시 돌리는 중: {gen}세대"
BUNDLE_REPRODUCED: Final = "다시 돌린 실행 폴더: {path}"
BUNDLE_SAME: Final = "원래 실행과 같습니다({gens}세대, 시간 열 제외 모든 통계 일치)."
BUNDLE_DIFFERENT: Final = "원래 실행과 {gen}세대부터 다릅니다(다른 PC나 다른 빌드면 비트 단위로 같지 않을 수 있습니다)."
CSV_DONE: Final = "통계 CSV를 저장했습니다: {path}"
CLI_HELP_BUNDLE: Final = "실행 폴더에서 재현 묶음(zip)을 만듭니다."
CLI_HELP_RUN_DIR: Final = "실행 폴더"
CLI_HELP_BUNDLE_OUT: Final = "만들 zip 파일 (기본: 실행 폴더\\exports)"
CLI_HELP_REPRODUCE: Final = "재현 묶음을 다시 돌리고 원래 통계와 비교합니다."
CLI_HELP_BUNDLE_FILE: Final = "재현 묶음 zip 파일"
CLI_HELP_CSV: Final = "세대별 통계를 CSV로 내보냅니다."
CLI_HELP_CSV_OUT: Final = "저장할 CSV 파일 (기본: 실행 폴더\\exports\\stats.csv)"
BUNDLE_FILE_NAME: Final = "{run}_재현묶음.zip"

# M6 영상 (lead)
VIDEO_HUD: Final = "{title} · {t:.1f}초"
VIDEO_HUD_TIME: Final = "{t:.1f}초"
VIDEO_NO_FRAMES: Final = "영상으로 만들 프레임이 없습니다."
VIDEO_SAVE: Final = "영상 저장…"
VIDEO_SAVE_TITLE: Final = "영상 저장 (GIF)"
VIDEO_FILTER: Final = "GIF 영상 (*.gif)"
VIDEO_PROGRESS: Final = "영상 만드는 중…"
VIDEO_CANCEL: Final = "취소"
VIDEO_DONE: Final = "영상을 저장했습니다: {path}"
VIDEO_NOTHING: Final = "먼저 타임라인에서 세대 대표를 재생하거나 '1세대 대표와 지금 챔피언'을 누르세요."

# M6 실험실 방법별 설정 (lead)
LAB_IPOP: Final = "pycma 종료 조건이 차면 개체 수 2배로 다시 시작(IPOP, CMA-ES만)"
LAB_GA_ELITES: Final = "엘리트 수"
LAB_GA_CROSSOVER: Final = "교차 확률"
LAB_GA_MUT_RATE: Final = "돌연변이 확률"
LAB_GA_MUT_SIGMA: Final = "돌연변이 폭"
LAB_GA_DECAY: Final = "폭 감쇠(세대마다)"
LAB_SA_NEIGHBORS: Final = "이웃 수"
LAB_SA_STEP: Final = "이웃 폭"
LAB_SA_TEMP: Final = "초기 온도"
LAB_SA_TEMP_AUTO: Final = "자동(첫 이웃 점수 표준편차)"
LAB_SA_COOLING: Final = "감쇠율(단계마다)"

# M5a 에디터 캔버스·도구·팔레트 (app/editor/canvas.py, tools.py, palette.py)
EDC_GROUP_SELECT: Final = "선택"
EDC_GROUP_SHAPES: Final = "모양"
EDC_GROUP_TERRAIN: Final = "지형·트랙"
EDC_GROUP_JOINTS: Final = "관절"
EDC_GROUP_ACTUATORS: Final = "구동기"
EDC_GROUP_SENSORS: Final = "센서"
EDC_GROUP_REGIONS: Final = "영역·체크포인트·시작 위치"
EDC_TOOL_SELECT: Final = "선택·이동·회전"
EDC_TOOL_BOX: Final = "사각형"
EDC_TOOL_CIRCLE: Final = "원"
EDC_TOOL_POLY: Final = "다각형"
EDC_TOOL_TERRAIN_LINE: Final = "지형 선"
EDC_TOOL_TERRAIN_AUTO: Final = "지형 자동 생성"
EDC_TOOL_TRACK_DRAW: Final = "트랙 그리기"
EDC_TOOL_TRACK_AUTO: Final = "트랙 자동 생성"
EDC_TOOL_JOINT_PIVOT: Final = "회전 관절"
EDC_TOOL_JOINT_ROT_SPRING: Final = "회전 스프링"
EDC_TOOL_JOINT_WELD: Final = "고정 관절"
EDC_TOOL_JOINT_SPRING: Final = "거리 스프링"
EDC_TOOL_THRUSTER: Final = "추진기"
EDC_TOOL_FUEL_TANK: Final = "연료 탱크"
EDC_TOOL_VEHICLE: Final = "차량 구동기"
EDC_TOOL_SENSOR_RAYS: Final = "레이"
EDC_TOOL_SENSOR_BODY_STATE: Final = "몸 상태"
EDC_TOOL_SENSOR_CONTACT: Final = "접촉"
EDC_TOOL_SENSOR_TARGET: Final = "목표 위치"
EDC_TOOL_SENSOR_JOINT_ANGLE: Final = "관절 각도"
EDC_TOOL_SENSOR_FUEL: Final = "남은 연료"
EDC_TOOL_SENSOR_TIME: Final = "남은 시간"
EDC_TOOL_REGION_GOAL: Final = "목표 영역"
EDC_TOOL_REGION_HAZARD: Final = "위험 영역"
EDC_TOOL_REGION_PAD: Final = "착륙 패드"
EDC_TOOL_CHECKPOINTS: Final = "체크포인트"
EDC_TOOL_SPAWN: Final = "시작 위치"
EDC_HINT_SELECT: Final = "누르면 선택(Shift: 더하기·빼기), 빈 곳을 끌면 범위 선택, 끌어서 이동, 노란 손잡이로 회전. Delete: 지우기, 화살표: 격자 한 칸 이동."
EDC_HINT_BOX: Final = "끌어서 사각형 물체를 만듭니다. Ctrl을 누른 채 그리면 선택한 물체에 모양을 더합니다. Alt: 격자 맞춤 끄기."
EDC_HINT_CIRCLE: Final = "가운데를 누르고 끌어서 반지름을 정합니다. Ctrl을 누른 채 그리면 선택한 물체에 모양을 더합니다."
EDC_HINT_POLY: Final = "꼭짓점을 차례로 누르고, 두 번 누르거나 Enter로 끝냅니다. Backspace: 마지막 점 지우기, Esc: 취소."
EDC_HINT_TERRAIN_LINE: Final = "지형 꼭짓점을 차례로 누르고, 두 번 누르거나 Enter로 끝냅니다."
EDC_HINT_TERRAIN_AUTO: Final = "높이맵 지형 생성기와 착륙 패드를 더합니다. 수치는 생성기 탭에서 고칩니다."
EDC_HINT_TRACK_DRAW: Final = "트랙 중심선 꼭짓점을 차례로 누르고(닫힌 길), 두 번 누르거나 Enter로 끝냅니다. 첫 점이 출발선입니다."
EDC_HINT_TRACK_AUTO: Final = "학습 트랙 세트를 만드는 트랙 생성기를 더합니다. 수치는 생성기 탭에서 고칩니다."
EDC_HINT_JOINT: Final = "첫 물체(a)에서 관절 자리를 누르세요."
EDC_HINT_JOINT_SECOND: Final = "둘째 물체(b)에서 관절 자리를 누르세요. 두 점이 겹치도록 b가 놓입니다."
EDC_HINT_THRUSTER: Final = "에이전트 물체를 눌러 추진기 자리를 정하고, 끌어서 미는 방향을 정합니다."
EDC_HINT_FUEL_TANK: Final = "물체를 누르면 연료 탱크를 더하고, 그 물체의 추진기가 이 탱크를 쓰게 합니다."
EDC_HINT_VEHICLE: Final = "에이전트 물체를 눌러 차량 구동기를 답니다."
EDC_HINT_SENSOR_BODY: Final = "센서를 달 물체를 누르세요(레이는 누른 자리에서 나갑니다)."
EDC_HINT_SENSOR_TARGET: Final = "센서를 달 물체를 누르세요."
EDC_HINT_SENSOR_TARGET_SECOND: Final = "목표(영역이나 움직이는 물체)를 누르세요."
EDC_HINT_SENSOR_JOINT_ANGLE: Final = "관절 표시를 누르세요."
EDC_HINT_SENSOR_FUEL: Final = "연료 탱크 표시나 탱크를 쓰는 물체를 누르세요."
EDC_HINT_SENSOR_TIME: Final = "남은 시간 센서를 더합니다."
EDC_HINT_REGION: Final = "끌어서 영역을 그립니다(그냥 누르면 기본 크기)."
EDC_HINT_CHECKPOINTS: Final = "선마다 두 점을 누르고, Enter로 끝냅니다. 첫 선이 출발선입니다. Backspace: 마지막 선 지우기."
EDC_HINT_SPAWN: Final = "눌러서 시작 위치를, 끌어서 시작 방향을 정합니다."
EDC_LABEL_BOX: Final = "사각형"
EDC_LABEL_CIRCLE: Final = "원"
EDC_LABEL_POLY: Final = "다각형"
EDC_LABEL_TERRAIN_LINE: Final = "지형 선"
EDC_LABEL_HEIGHTMAP: Final = "지형"
EDC_LABEL_TRACK: Final = "트랙"
EDC_LABEL_JOINT_PIVOT: Final = "회전 관절"
EDC_LABEL_JOINT_ROT_SPRING: Final = "회전 스프링 관절"
EDC_LABEL_JOINT_WELD: Final = "고정 관절"
EDC_LABEL_JOINT_SPRING: Final = "거리 스프링"
EDC_LABEL_THRUSTER: Final = "추진기"
EDC_LABEL_TANK: Final = "연료 탱크"
EDC_LABEL_VEHICLE: Final = "차량 구동기"
EDC_LABEL_RAYS: Final = "레이"
EDC_LABEL_BODY_STATE: Final = "몸 상태"
EDC_LABEL_CONTACT: Final = "접촉"
EDC_LABEL_TARGET: Final = "목표 상대 위치"
EDC_LABEL_JOINT_ANGLE: Final = "관절 각도"
EDC_LABEL_FUEL: Final = "남은 연료"
EDC_LABEL_TIME_LEFT: Final = "남은 시간"
EDC_LABEL_REGION_GOAL: Final = "목표"
EDC_LABEL_REGION_HAZARD: Final = "위험"
EDC_LABEL_REGION_PAD: Final = "착륙 패드"
EDC_LABEL_REGION_ZONE: Final = "영역"
EDC_LABEL_CHECKPOINTS: Final = "체크포인트"
EDC_BADGE_RAYS: Final = "레이"
EDC_BADGE_BODY_STATE: Final = "상태"
EDC_BADGE_CONTACT: Final = "접촉"
EDC_BADGE_TARGET: Final = "목표"
EDC_BADGE_JOINT_ANGLE: Final = "각도"
EDC_BADGE_FUEL: Final = "연료%"
EDC_BADGE_TIME_LEFT: Final = "시간"
EDC_BADGE_TANK: Final = "탱크"
EDC_BADGE_VEHICLE: Final = "구동"
EDC_BADGE_MOTOR: Final = "모터"
EDC_SPAWN_LABEL: Final = "시작"
EDC_COORDS: Final = "x {x:.2f}  y {y:.2f}"
EDC_FIT: Final = "전체 보기"
EDC_SNAP: Final = "격자 맞춤"
EDC_SNAP_UNIT: Final = " m"
EDC_SNAP_TIP: Final = "점과 이동을 격자에 맞춥니다. 끄는 동안 Alt를 누르면 잠시 끕니다."
EDC_OWNER_LABEL: Final = "새 물체:"
EDC_OWNER_AGENT: Final = "에이전트"
EDC_OWNER_WORLD: Final = "세계"
EDC_PREVIEW_PARTIAL: Final = "검사 오류가 있어 오류가 난 부품은 숫자만으로 그렸습니다(수식은 0으로 볼 수 있음)."
EDC_PREVIEW_BUILD_FAILED: Final = "미리보기 세계를 만들지 못해 숫자만으로 그렸습니다: {reason}"
EDC_LOCKED_FORMULA: Final = "위치나 각도가 수식이라 끌 수 없습니다. 속성 패널에서 고치세요."
EDC_LOCKED_PLACED: Final = "이 물체는 관절(place_by)이 자리를 정합니다. 관절 앵커나 쉬는 각도를 고치세요."
EDC_LOCKED_SPAWN: Final = "시작 위치의 수식 칸({fields})은 바꾸지 않았습니다."
EDC_ERR_POLY_FEW: Final = "다각형에는 서로 다른 꼭짓점이 3개 이상 있어야 합니다."
EDC_ERR_POLY_MANY: Final = "다각형 꼭짓점은 {limit}개까지입니다."
EDC_ERR_POLY_AREA: Final = "다각형 넓이가 0입니다. 꼭짓점이 한 줄에 있지 않게 그리세요."
EDC_ERR_POLY_SELF_INTERSECT: Final = "다각형 변이 서로 교차합니다. 꼬이지 않게 다시 그리세요(Backspace로 마지막 점을 지울 수 있습니다)."
EDC_ERR_LINE_FEW: Final = "선에는 서로 다른 점이 2개 이상 있어야 합니다."
EDC_ERR_TRACK_FEW: Final = "트랙 중심선에는 서로 다른 꼭짓점이 3개 이상 있어야 합니다."
EDC_ERR_TRACK_ONE_SET: Final = "학습 트랙 세트를 만드는 트랙 생성기는 하나만 둘 수 있습니다."
EDC_ERR_CHECKPOINT_FEW: Final = "체크포인트 선이 없습니다. 선마다 두 점을 누르세요."
EDC_ERR_CLICK_BODY: Final = "물체를 누르세요."
EDC_ERR_CLICK_AGENT_BODY: Final = "에이전트 물체를 누르세요."
EDC_ERR_CLICK_JOINT: Final = "관절 표시를 누르세요."
EDC_ERR_CLICK_TARGET: Final = "목표로 쓸 영역이나 움직이는 물체를 누르세요."
EDC_ERR_ACTUATOR_BODY: Final = "구동기는 에이전트의 움직이는 물체에만 달 수 있습니다('{body}')."
EDC_ERR_JOINT_SAME: Final = "관절은 서로 다른 두 물체를 잇습니다."
EDC_ERR_NO_BODY: Final = "그 물체가 없습니다."
EDC_ERR_NO_JOINT: Final = "그 관절이 없습니다."
EDC_ERR_NO_TANK: Final = "연료 탱크가 없습니다. 먼저 연료 탱크를 더하세요."
EDC_ERR_TARGET_SELF: Final = "센서를 단 물체 자신은 목표가 될 수 없습니다."
EDC_UNDO_ADD_BODY: Final = "물체 추가"
EDC_UNDO_ADD_SHAPE: Final = "모양 추가"
EDC_UNDO_ADD_TERRAIN_LINE: Final = "지형 선 추가"
EDC_UNDO_ADD_GENERATOR: Final = "생성기 추가"
EDC_UNDO_ADD_JOINT: Final = "관절 추가"
EDC_UNDO_ADD_ACTUATOR: Final = "구동기 추가"
EDC_UNDO_ADD_SENSOR: Final = "센서 추가"
EDC_UNDO_ADD_REGION: Final = "영역 추가"
EDC_UNDO_ADD_CHECKPOINTS: Final = "체크포인트 추가"
EDC_UNDO_SPAWN: Final = "시작 위치"
EDC_UNDO_MOVE: Final = "옮기기"
EDC_UNDO_ROTATE: Final = "돌리기"

# M7 배포 (lead)
MENU_HELP: Final = "도움말"
MENU_ABOUT: Final = "정보…"
MENU_OPEN_DATA: Final = "데이터 폴더 열기"
ABOUT_TEXT: Final = (
    "{name} {version}\n\n"
    "직접 만든 2D 실험 환경에서 진화 전략(CMA-ES, 유전 알고리즘, 담금질)이 해법을 찾아가는 과정을 보는 프로그램입니다.\n"
    "사용자 데이터: 문서\\진화실험실 (environments, runs, records)\n\n"
    "오픈 소스 라이선스\n"
    "--------------------\n{notices}"
)
ABOUT_NO_NOTICES: Final = "(개발 환경에서 실행 중이라 라이선스 고지 파일이 없습니다. exe 배포본의 licenses 폴더에 있습니다.)"
ABOUT_OPEN_LICENSES: Final = "라이선스 폴더 열기"
SELFTEST_CHECK_BUNDLE: Final = "내장 예제와 배포 파일"
SELFTEST_ERR_MISSING: Final = "없습니다: {what}"

# M6 compare (비교 화면, 반복 실행: app/compare/)
CMP_RUN_NAME: Final = "{env} · {method} · 시드 {seed}"
CMP_RUN_NAME_NO_SEED: Final = "{env} · {method}"
CMP_GROUP_NAME: Final = "{env} · {method} · 시드 {first}–{last}"
CMP_GROUP_NAME_ONE: Final = "{env} · {method} · 시드 {seed} (반복 1회)"
CMP_GROUP_NAME_PLAIN: Final = "{env} · {method} 반복 실행"
CMP_GROUP_TIP: Final = "반복 실행 묶음: {name}\n묶음 id: {id}\n묶음 체크박스로 묶음 전체를 고릅니다."
CMP_GROUP_MEMBERS: Final = "실행 {n}/{count}"
CMP_GROUP_INDEX: Final = "{i}/{count}"
CMP_FOLDER: Final = "실행 {count}개 · {path}"
CMP_LIST_COLUMNS: Final = ("환경", "방법", "시드", "세대", "묶음")
CMP_LIST_EMPTY: Final = "아직 실행 기록이 없습니다. 실험실에서 학습하거나 '반복 실행'으로 시드만 바꿔 여러 번 돌리면 여기에 나타납니다."
CMP_UNREADABLE: Final = "읽을 수 없음"
CMP_NONE: Final = "-"
CMP_TIP_FOLDER: Final = "폴더: {path}"
CMP_TIP_STARTED: Final = "시작: {started}"
CMP_TIP_MORE: Final = "… 외 {n}개"
CMP_REPEAT_BUTTON: Final = "반복 실행…"
CMP_REPEAT_BUTTON_TIP: Final = "같은 설정을 시드만 바꿔 여러 번 차례로 학습합니다. 묶음은 평균 선과 최소~최대 띠로 그립니다."
CMP_REFRESH: Final = "새로 고침"
CMP_CLEAR_SELECTION: Final = "모두 풀기"
CMP_REPEAT_CANCEL: Final = "취소"
CMP_Y_LABEL: Final = "세로축"
CMP_X_LABEL: Final = "가로축"
CMP_Y_NAMES: Final = {
    "compare_score": "비교 점수",
    "compare_success": "성공률(%)",
    "champion_score": "검증 점수",
    "score_best": "세대 최고 학습 점수",
}
CMP_Y_AXIS_NAMES: Final = {
    "compare_score": "비교 점수(챔피언)",
    "compare_success": "성공률(%)",
    "champion_score": "검증 점수(챔피언)",
    "score_best": "학습 점수(세대 최고)",
}
CMP_Y_TIP: Final = (
    "비교 점수: 챔피언을 선정에 쓰지 않은 비교 평가 세트(시드 32개)로 잰 점수(기본값, 공정한 비교).\n"
    "성공률: 같은 비교 평가에서 성공한 비율.\n"
    "검증 점수: 챔피언 선정에 쓴 검증 세트 점수(선정 편향이 남습니다).\n"
    "세대 최고 학습 점수: 잡음 섞인 학습 점수의 세대 최고점(개체가 많은 방법이 운으로 높게 나옵니다)."
)
CMP_X_NAMES: Final = {"evaluations": "누적 시험 횟수", "elapsed": "경과 시간(초)", "generation": "세대"}
CMP_X_TIP: Final = "방법마다 한 세대의 크기가 다르므로 기본 가로축은 누적 학습 시험 횟수입니다(검증·비교 평가 시험은 넣지 않습니다)."
CMP_GROUP_MEAN: Final = "묶음은 평균과 범위로"
CMP_GROUP_MEAN_TIP: Final = (
    "켜면 반복 실행 묶음을 평균 선과 최소~최대 띠로, 끄면 묶음 실행을 하나씩 그립니다.\n"
    "모든 실행에 값이 있는 구간만 실선과 띠로 그리고, 일부 실행에만 값이 있는 구간(먼저 끝났거나 아직 도는 실행이 있을 때)은 "
    "그 실행들만의 평균을 점선으로 잇습니다."
)
CMP_TARGET_LABEL: Final = "목표 점수"
CMP_TARGET_TIP: Final = "표의 '목표 점수 도달까지 시험 횟수'를 정하는 비교 점수입니다. 정하기 전에는 환경 파일의 목표 점수를 씁니다."
CMP_TARGET_LINE: Final = "목표 {v}"
CMP_SAVE_PNG: Final = "그래프 이미지 저장"
CMP_SAVE_CSV: Final = "표 CSV 저장"
CMP_PNG_NAME: Final = "비교.png"
CMP_CSV_NAME: Final = "비교.csv"
CMP_PNG_FILTER: Final = "PNG 이미지 (*.png)"
CMP_CSV_FILTER: Final = "CSV 파일 (*.csv)"
CMP_SAVE_FAILED: Final = "저장하지 못했습니다: {path} {reason}"
CMP_ERROR_TITLE: Final = "비교"
CMP_PICK_RUNS: Final = "왼쪽 목록에서 비교할 실행이나 반복 실행 묶음을 고르세요."
CMP_NO_VALUES: Final = (
    "고른 실행에 아직 '{y}' 값이 없습니다. 비교 평가는 학습 시험 1,000회마다 하는 검증에서 챔피언이 바뀔 때 합니다."
)
CMP_TITLE: Final = "{y} · 가로축 {x}"
CMP_LEGEND_GROUP: Final = "{name} 평균 ({n}개)"
CMP_LEGEND_BAND: Final = "최소~최대"
CMP_LEGEND_PARTIAL: Final = "일부 실행만"
CMP_TABLE_COLUMNS: Final = ("이름", "방법", "시드", "최종 점수", "목표 점수 도달까지 시험 횟수", "소요 시간", "성공률")
CMP_TABLE_TIPS: Final = (
    "실행(환경 · 방법 · 시드) 또는 반복 실행 묶음. 색은 그래프 선과 같습니다.",
    "학습 방법",
    "실행 시드. 묶음은 시드 범위와 실행 수",
    "챔피언을 비교 평가 세트(시드 32개)로 잰 마지막 점수. 묶음은 평균 ± 표본 표준편차",
    "비교 점수가 처음 목표 점수 이상이 된 때의 누적 학습 시험 횟수. 못 닿으면 '-'. 묶음은 닿은 실행들의 평균과 닿은 수",
    "학습에 걸린 시간(마지막 경과 시간). 묶음은 평균",
    "마지막 비교 평가에서 성공한 비율. 묶음은 평균 ± 표본 표준편차",
)
CMP_MEAN_STD: Final = "{mean} ± {std}"
CMP_REACHED_SOME: Final = "{value} ({k}/{n}개 도달)"
CMP_SECONDS: Final = "{s:.1f}초"
CMP_MINUTES: Final = "{m}분 {s:02d}초"
CMP_HOURS: Final = "{h}시간 {m:02d}분"
CMP_PERCENT: Final = "{v:.0f}%"
CMP_SEED_COUNT: Final = "{n}개"
CMP_SEED_RANGE: Final = "{first}–{last} ({n}개)"
CMP_CSV_HEADERS: Final = (
    "이름", "방법", "시드", "실행 수", "최종 점수(평균)", "최종 점수 표준편차", "목표 점수",
    "목표 점수 도달까지 시험 횟수(평균)", "목표 점수에 닿은 실행 수", "소요 시간(초, 평균)", "성공률(%, 평균)",
    "성공률 표준편차",
)
CMP_REPEAT_PROGRESS: Final = "반복 실행 '{name}' {i}/{n} · 시드 {seed} · {gen}세대 · 시험 {evals}회 · {time}"
CMP_REPEAT_DONE: Final = "반복 실행 '{name}'을(를) 마쳤습니다({n}개)."
CMP_REPEAT_CANCELLED: Final = "반복 실행 '{name}'을(를) 취소했습니다({n}개 중 {done}개 끝냄)."
CMP_REPEAT_FAILED: Final = "반복 실행이 멈췄습니다: {reason}"
CMP_REPEAT_LOST: Final = "학습 프로세스가 정지 조건에 닿기 전에 끝났습니다."
CMP_REPEAT_PAUSED: Final = "평가 오류로 학습이 멈췄습니다."
CMP_REPEAT_START_FAILED: Final = "학습 프로세스를 시작하지 못했습니다: {reason}"
CMP_REPEAT_TITLE: Final = "반복 실행"
CMP_REPEAT_NOTE: Final = (
    "같은 설정을 시드만 바꿔 N번 차례로 학습합니다(한 번에 하나씩, 관전 없이). 실행들은 한 묶음으로 기록되어 "
    "비교 화면에 평균 선과 최소~최대 띠로 나옵니다."
)
CMP_REPEAT_ENV: Final = "환경"
CMP_REPEAT_ENV_BUILTIN: Final = "{name} (내장 예제)"
CMP_REPEAT_ENV_FILE: Final = "{name} ({file})"
CMP_REPEAT_ENV_OPEN: Final = "파일 열기…"
CMP_REPEAT_ENV_FAILED: Final = "환경 파일을 읽을 수 없습니다: {path}\n{reason}"
CMP_REPEAT_NO_ENV: Final = "환경 파일을 고르세요."
CMP_REPEAT_METHOD: Final = "학습 방법"
CMP_REPEAT_METHODS_MISSING: Final = "아직 쓸 수 없는 방법: {names}"
CMP_REPEAT_POPSIZE: Final = "개체 수"
CMP_REPEAT_NEIGHBORS: Final = "이웃 수"
CMP_REPEAT_TRIALS: Final = "시험 횟수"
CMP_REPEAT_TRIALS_ENV: Final = "환경 파일 값"
CMP_REPEAT_COUNT: Final = "반복 횟수"
CMP_REPEAT_BASE_SEED: Final = "첫 시드"
CMP_REPEAT_SEEDS: Final = "실행 시드: {seeds}"
CMP_REPEAT_STOP: Final = "정지 조건"
CMP_REPEAT_STOP_EVALS: Final = "최대 시험 횟수"
CMP_REPEAT_STOP_TIME: Final = "최대 시간(분)"
CMP_REPEAT_WORKERS: Final = "워커 수"
CMP_REPEAT_NAME: Final = "묶음 이름"
CMP_REPEAT_NAME_TIP: Final = "비우면 '환경 · 방법 · 시드 범위'로 짓습니다."
CMP_REPEAT_START: Final = "시작"
CMP_REPEAT_CANCEL_DIALOG: Final = "닫기"
CMP_RECORDS_BUTTON: Final = "비교"
CMP_RECORDS_BUTTON_TIP: Final = "고른 실행들을 비교 화면에서 한 그래프에 겹쳐 그립니다."

# 실험실 ↔ 에디터 (lead)
LAB_ENV_EDIT: Final = "에디터에서 고치기"
LAB_ENV_EDIT_TIP: Final = "이 환경을 에디터에서 엽니다. 멈춘 학습이면 고친 뒤 '실험실에 보내기'로 이어서 학습합니다."
LAB_ENV_FROM_EDITOR: Final = "에디터에서 보낸 환경"
# M5a 속성 패널과 아래 탭 (EDP: app/editor/forms.py, properties.py, tabs.py)
EDP_FIELD_NAMES: Final = {
    # 칸 이름 -> 화면 이름. "모델.칸"이 있으면 그것이 먼저다.
    "id": "이름(id)", "label": "표시 이름", "kind": "종류", "name": "환경 이름", "description": "설명",
    "gravity": "중력 (x, y)", "wind": "바람 (x, y)", "dt": "시간 간격(초)", "max_time": "최대 시간(초)",
    "bounds": "경계 (왼쪽, 아래, 오른쪽, 위)", "contact_debounce": "접촉 디바운스(초)",
    "value": "값", "evolve": "진화", "min": "최솟값", "max": "최댓값",
    "dist": "분포", "low": "최솟값", "high": "최댓값", "mean": "평균", "std": "표준편차", "values": "고를 값들",
    "w": "너비", "h": "높이", "offset": "중심 위치 (x, y)", "angle": "각도", "r": "반지름", "points": "꼭짓점",
    "radii": "반지름들", "closed": "닫힌 선", "radius": "두께(반지름)",
    "density": "밀도", "friction": "마찰", "elasticity": "탄성",
    "owner": "소속", "type": "움직임", "pos": "위치 (x, y)", "place_by": "배치 관절", "shapes": "모양",
    "material": "재질", "tags": "태그", "drag": "바람 항력",
    "a": "물체 A (부모)", "b": "물체 B (자식)", "anchor_a": "A 쪽 앵커", "anchor_b": "B 쪽 앵커",
    "rest_angle": "쉬는 각도", "stiffness": "강성", "damping": "감쇠", "rest_length": "쉬는 길이",
    "limit": "각도 제한", "collide": "서로 부딪힘",
    "capacity": "용량", "body": "물체", "at": "붙는 위치 (x, y)", "dir": "미는 방향(라디안)", "max_force": "최대 힘",
    "tank": "연료 탱크", "fuel_per_s": "초당 연료", "keys": "조종 키",
    "max_accel": "최대 가속", "max_steer": "최대 조향각", "lateral_grip": "옆미끄럼 마찰", "wheelbase": "축간 거리",
    "max_lateral_accel": "최대 횡가속도", "keys_forward": "앞으로 키", "keys_back": "뒤로 키",
    "keys_left": "왼쪽 키", "keys_right": "오른쪽 키",
    "joint": "관절", "max_torque": "최대 토크", "max_speed": "최대 속도", "keys_pos": "+ 방향 키",
    "keys_neg": "- 방향 키",
    "fields": "항목", "frame": "좌표계", "scale": "정규화 크기", "target": "목표", "with": "닿는 대상",
    "origin": "시작점 (x, y)", "count": "개수", "spread": "부채꼴 각도", "center": "가운데 방향",
    "length": "길이", "hits": "감지 대상",
    "shape": "모양", "lines": "선", "loop": "한 바퀴로 이어짐",
    "x": "x", "y": "y", "vx": "x 속도", "vy": "y 속도", "omega": "각속도", "width": "폭",
    "region_height": "영역 높이",
    "seed": "시드", "x_range": "x 범위", "segment": "구간 길이", "base_y": "기준 높이", "amplitude": "높낮이",
    "roughness": "거칠기", "pad": "착륙 패드",
    "source": "트랙 만들기", "set_seed": "트랙 세트 시드", "set_size": "학습 트랙 수", "centerline": "중심선",
    "segments": "구간 수", "curvature": "곡률", "length_range": "길이 범위", "checkpoints": "체크포인트",
    "wall_radius": "벽 두께",
    "spawn": "시작 위치", "when": "조건", "hold": "유지 시간(초)", "event": "사건", "outcome": "결과",
    "weight": "가중치", "agg": "집계", "direction": "방향(라디안)", "actuators": "구동기", "expr": "수식",
    "blocks": "블록", "formula": "점수 수식",
    "inputs": "입력", "hidden": "은닉층 크기", "activation": "활성화 함수", "outputs": "출력",
    "trials": "시험 횟수", "aggregate": "집계 방식", "target_score": "목표 점수",
    "world": "세계", "agent": "에이전트", "scoring": "점수", "brain": "뇌", "evaluation": "평가",
    "Joint.kind": "관절 종류", "Region.kind": "영역 종류", "Vehicle.drag": "구름 저항",
    "HeightmapGen.radius": "지면 두께", "HeightmapGen.friction": "지면 마찰", "TrackGen.friction": "벽 마찰",
    "TrackGen.width": "트랙 폭", "TrackGen.checkpoints": "체크포인트 수", "PadSpec.x": "패드 가운데 x",
    "PadSpec.width": "패드 폭", "EventBlock.value": "값 종류", "Spawn.x": "x", "Spawn.y": "y",
    "Spawn.angle": "각도", "Spawn.vx": "x 속도", "Spawn.vy": "y 속도", "Spawn.omega": "각속도",
}
EDP_FIELD_TIPS: Final = {
    # 칸 도움말(모델 설명 앞에 붙는다). "모델.칸"이 있으면 그것이 먼저다.
    "id": "다른 칸과 수식에서 이 부품을 가리키는 이름입니다.",
    "label": "화면에 보이는 이름입니다.",
    "dt": "물리 시간 간격(초). 기본 1/60초.",
    "max_time": "이 시간이 지나면 timeout(시간 초과) 사건으로 끝납니다.",
    "gravity": "중력 가속도 (x, y), m/s².",
    "keys": "직접 조종 키 이름을 쉼표로 나눠 씁니다(예: Up, W). R, N, Space는 창 제어용이라 쓸 수 없습니다.",
    "keys_forward": "직접 조종 키 이름을 쉼표로 나눠 씁니다(예: Up, W).",
    "keys_back": "직접 조종 키 이름을 쉼표로 나눠 씁니다(예: Down, S).",
    "keys_left": "직접 조종 키 이름을 쉼표로 나눠 씁니다(예: Left, A).",
    "keys_right": "직접 조종 키 이름을 쉼표로 나눠 씁니다(예: Right, D).",
    "keys_pos": "직접 조종 키 이름을 쉼표로 나눠 씁니다.",
    "keys_neg": "직접 조종 키 이름을 쉼표로 나눠 씁니다.",
    "tags": "모양 태그를 쉼표로 나눠 씁니다. 조건과 센서에서 tag.<태그>로 가리킵니다.",
    "with": "물체 id나 tag.<태그>를 쉼표로 나눠 씁니다.",
    "hits": "레이가 맞는 물체 id나 tag.<태그>를 쉼표로 나눠 씁니다.",
    "when": "참·거짓 수식('=' 없이). 예: contact(hull, tag.terrain) and hull.speed < 0.5",
    "hold": "조건이 이 시간 동안 이어서 참이면 사건이 일어납니다.",
    "weight": "점수 = 블록 값 × 가중치의 합(점수 수식이 없을 때). 벌점은 음수로 씁니다.",
    "evolve": "켜면 몸 유전자가 됩니다(최솟값·최댓값이 있어야 합니다).",
    "trials": "개체 하나를 몇 번 시험할지.",
    "aggregate": "여러 시험 점수를 하나로 합치는 방법.",
    "target_score": "이 점수에 이르면 학습을 멈춥니다(비우면 없음).",
    "hidden": "은닉층마다 뉴런 수를 쉼표로 나눠 씁니다(예: 16 또는 32, 16).",
    "fields": "읽을 상태 항목. 채널 이름은 <센서 id>.<항목>입니다.",
    "values": "choice 분포에서 고를 값들(쉼표로 나눔).",
    "expr": "매 스텝 계산하는 수식('=' 없이). 참은 1입니다.",
    "inputs": "신경망 입력 순서. 켠 채널만 이 순서대로 넣습니다.",
    "outputs": "신경망 출력 순서. 켠 채널만 이 순서대로 씁니다.",
}
EDP_VALUE_NAMES: Final = {
    # 고르기 칸의 값 -> 화면 이름. "칸.값"이 있으면 그것이 먼저다.
    "physics2d": "2D 물리", "grid": "격자",
    "uniform": "균등", "int": "정수 균등", "normal": "정규분포", "choice": "목록에서 고르기",
    "box": "사각형", "circle": "원", "poly": "다각형", "radial": "방사형", "chain": "선 체인",
    "agent": "에이전트", "world": "세계", "dynamic": "움직임", "static": "고정", "kinematic": "정해진 움직임",
    "pivot": "회전", "weld": "고정", "rot_spring": "회전 스프링", "spring": "거리 스프링",
    "fuel_tank": "연료 탱크", "thruster": "추진기", "vehicle": "차량 구동기", "motor": "모터",
    "body_state": "몸 상태", "target": "목표 위치", "contact": "접촉", "joint_angle": "관절 각도",
    "fuel": "남은 연료", "time_left": "남은 시간", "rays": "레이",
    "frame.world": "세계 좌표", "frame.body": "물체 좌표",
    "goal": "목표", "hazard": "위험", "pad": "착륙 패드", "zone": "구역",
    "heightmap": "지형 자동 생성", "track": "트랙", "generated": "자동 생성", "drawn": "직접 그림",
    "success": "성공", "failure": "실패", "neutral": "중립",
    "event": "사건 발생", "distance_to": "목표까지 거리", "progress": "진행 거리", "survival_time": "생존 시간",
    "final_speed": "종료 속도", "tilt": "기울기", "energy": "에너지 사용", "fuel_used": "연료 사용",
    "contact_time": "접촉 시간", "smoothness": "부드러움", "checkpoints": "체크포인트", "lap_time": "랩 시간",
    "expr": "수식",
    "occurred": "일어났나 (0/1)", "count": "횟수",
    "final": "종료 시", "min": "최솟값", "mean": "평균", "max": "최댓값", "integral": "누적", "q25": "하위 25%",
    "tanh": "tanh", "relu": "ReLU",
    "timeout": "시간 초과", "out_of_bounds": "경계 이탈",
    "x": "x 위치", "y": "y 위치", "vx": "x 속도", "vy": "y 속도", "speed": "속력", "angle": "각도",
    "omega": "각속도", "v_fwd": "앞뒤 속도", "v_lat": "옆 속도",
}
EDP_SECTION_NAMES: Final = {
    "bodies": "물체", "joints": "관절", "actuators": "구동기", "sensors": "센서", "regions": "영역",
    "checkpoints": "체크포인트", "events": "사건", "generators": "생성기", "randomize": "랜덤 값",
    "params": "매개변수", "scoring.blocks": "점수 블록", "termination": "종료 조건", "scoring": "점수",
    "world": "세계", "agent": "시작 위치", "brain": "뇌", "evaluation": "평가",
}
EDP_BAD_NUMBER: Final = "숫자나 '='로 시작하는 수식을 쓰세요."
EDP_BAD_PLAIN_NUMBER: Final = "숫자를 쓰세요(이 칸은 수식을 쓸 수 없습니다)."
EDP_BAD_INT: Final = "정수를 쓰세요."
EDP_BAD_CELL: Final = "표에 숫자나 수식이 아닌 칸이 있습니다."
EDP_REQUIRED: Final = "비워 둘 수 없습니다."
EDP_TOO_LONG: Final = "{n}자까지 쓸 수 있습니다."
EDP_BAD_ID: Final = "이름(id)은 영문자나 _로 시작하고 영문자, 숫자, _만 쓸 수 있습니다."
EDP_ID_TAKEN: Final = "'{id}'은(는) 이미 쓰는 이름입니다."
EDP_BAD_REF: Final = "부품 이름(id)이나 tag.<태그>를 쓰세요."
EDP_LIST_TIP: Final = "쉼표로 나눠 여러 개를 씁니다."
EDP_BAD_LIST_ITEM: Final = "'{item}'은(는) 이 목록에 쓸 수 없는 값입니다."
EDP_LIST_TOO_SHORT: Final = "적어도 {n}개는 있어야 합니다."
EDP_UNKNOWN_VALUE: Final = "{value} (모르는 값)"
EDP_BUILTIN_EVENT: Final = "내장 사건: {name}"
EDP_EVOLVE: Final = "진화"
EDP_EVOLVE_TIP: Final = "켜면 이 값이 몸 유전자가 되어 최솟값과 최댓값 사이에서 진화합니다."
EDP_EVOLVE_ON_TIP: Final = "몸 유전자 p.{id}. 끄면 지금 초깃값으로 고정합니다."
EDP_EVOLVE_FORMULA_TIP: Final = "수식 칸은 진화 대상으로 바꿀 수 없습니다. 숫자로 바꾸거나 매개변수 하나(=p.이름)만 가리키게 하세요."
EDP_EVOLVE_VALUE_TIP: Final = "몸 유전자 p.{id}의 초깃값"
EDP_EVOLVE_PARAM_LABEL: Final = "{owner} {field}"
EDP_MIN: Final = "최소"
EDP_MAX: Final = "최대"
EDP_USE: Final = "사용"
EDP_ALL: Final = "전부"
EDP_PICK: Final = "목록에서 골라 더하기"
EDP_ADD: Final = "추가"
EDP_REMOVE: Final = "지우기"
EDP_UP: Final = "위로"
EDP_DOWN: Final = "아래로"
EDP_ADD_ROW: Final = "줄 추가"
EDP_REMOVE_ROW: Final = "줄 지우기"
EDP_REMOVE_ITEM_TIP: Final = "이 {name}을(를) 지웁니다."
EDP_COL_VALUE: Final = "값"
EDP_ITEM_TITLE: Final = "{name} {n}"
EDP_ADD_ITEM: Final = "{name} 추가"
EDP_UNDO_FIELD: Final = "{name} 바꾸기"
EDP_UNDO_KIND: Final = "종류 바꾸기"
EDP_UNDO_ADD_ITEM: Final = "{name} 추가"
EDP_UNDO_REMOVE_ITEM: Final = "{name} 지우기"
EDP_UNDO_MOVE: Final = "순서 바꾸기"
EDP_RENAME_TIP: Final = "Enter를 누르면 이름을 바꾸고, 이 이름을 가리키는 칸과 수식도 함께 고칩니다."
EDP_RENAME_FAILED: Final = "이름을 바꾸지 못했습니다: {reason}"
EDP_PANEL_TITLE_NONE: Final = "속성"
EDP_PANEL_TITLE_KIND: Final = "{section} · {kind}"
EDP_PANEL_HINT: Final = "캔버스나 아래 표에서 부품을 고르면 여기에서 속성을 고칩니다."
EDP_PANEL_MULTI: Final = "{n}개를 골랐습니다. 첫 번째 부품을 보여 줍니다."
EDP_PANEL_ID: Final = "이름(id)"
EDP_PANEL_LABEL: Final = "표시 이름"
EDP_PANEL_DELETE: Final = "지우기"
EDP_PANEL_DELETE_TIP: Final = "이 부품을 지웁니다(되돌리기로 살릴 수 있습니다)."
EDP_PANEL_ISSUES: Final = "이 부품의 검사 결과"
EDP_TAB_WORLD: Final = "세계 설정"
EDP_TAB_EVENTS: Final = "사건·종료 조건"
EDP_TAB_SCORING: Final = "점수 블록과 수식"
EDP_TAB_RANDOM: Final = "랜덤 범위"
EDP_TAB_GENERATORS: Final = "생성기"
EDP_TAB_EVAL: Final = "평가 설정"
EDP_TAB_BRAIN: Final = "뇌 설정"
EDP_TAB_PARAMS: Final = "매개변수"
EDP_TAB_ISSUES: Final = "검사 결과"
EDP_TAB_ISSUES_COUNT: Final = "검사 결과 ({n})"
EDP_GROUP_ENV: Final = "환경"
EDP_GROUP_WORLD: Final = "세계"
EDP_GROUP_SPAWN: Final = "시작 위치 (에이전트 좌표의 원점)"
EDP_GROUP_EVENTS: Final = "사건"
EDP_GROUP_TERMINATION: Final = "종료 조건 (위가 먼저)"
EDP_EVENTS_TIP: Final = "사건은 이름 붙은 조건입니다. 다른 수식에서 <id>, <id>.count, <id>.since로 씁니다. 내장 사건: timeout, out_of_bounds."
EDP_TERMINATION_TIP: Final = "같은 스텝에 여러 종료 조건이 겹치면 목록 위쪽이 먼저입니다. timeout(시간 초과)이 없으면 자동으로 맨 뒤에 더합니다."
EDP_GROUP_BLOCK: Final = "고른 블록의 설정"
EDP_GROUP_GENERATOR: Final = "고른 생성기의 설정"
EDP_DETAIL_PICK: Final = "표에서 줄을 고르면 종류별 설정이 여기에 나옵니다."
EDP_DETAIL_NO_FIELDS: Final = "이 종류에는 따로 설정할 칸이 없습니다."
EDP_FORMULA: Final = "점수 수식 (선택)"
EDP_FORMULA_TIP: Final = "비워 두면 블록 값 × 가중치의 합이 점수입니다. 쓰면 수식이 먼저입니다: 블록 id(가중치 전 값), time, success, failure, 사건 이름을 쓸 수 있습니다."
EDP_RANDOM_NOTE: Final = "시험마다 시험 시드로 뽑는 값입니다. 다른 칸의 수식에서 r.<id>로 씁니다(예: =r.pad_x)."
EDP_PARAMS_NOTE: Final = "수식에서 p.<id>로 쓰는 값입니다. '진화'를 켜면 몸 유전자가 되어 최솟값~최댓값 사이에서 진화합니다."
EDP_EVAL_TRACK_NOTE: Final = "학습 트랙 세트({n}개)가 있어 시험 횟수는 트랙 수로 정해집니다(트랙마다 한 번)."
EDP_BRAIN_INPUTS: Final = "입력 (센서 채널)"
EDP_BRAIN_OUTPUTS: Final = "출력 (구동기 채널)"
EDP_BRAIN_ALL: Final = "전부 (선언 순서)"
EDP_BRAIN_NET: Final = "신경망"
EDP_BRAIN_PARAMS: Final = "신경망 {shape} · 뇌 파라미터 {brain}개 · 몸 유전자 {body}개 · 유전자 차원 {dim}"
EDP_BRAIN_CHANNEL: Final = "{name}  ({label})"
EDP_BRAIN_PARAMS_NONE: Final = "검사 오류가 있어 파라미터 수를 계산할 수 없습니다."
EDP_BRAIN_NO_CHANNELS: Final = "파일 구조에 오류가 있어 채널 목록을 만들 수 없습니다. 검사 결과 탭을 보세요."
EDP_BRAIN_UNKNOWN: Final = "{name} (없는 채널)"
EDP_ISSUES_SEVERITY: Final = {"error": "오류", "warning": "경고"}
EDP_ISSUES_HEADERS: Final = ("종류", "위치", "내용")
EDP_ISSUES_NONE: Final = "문제가 없습니다."
EDP_ISSUES_HINT: Final = "두 번 누르면 그 부품을 고르고 화면을 옮깁니다."
EDP_WHERE_FILE: Final = "파일 전체"
EDP_WHERE_REF: Final = "{section} '{id}' ({where})"
EDP_WHERE_PATH: Final = "{section} ({where})"

# M5b 에디터 도구 (lead)
EDC_TOOL_HILLS_AUTO: Final = "언덕 자동 생성"
EDC_TOOL_MOTOR: Final = "모터"
EDC_HINT_HILLS_AUTO: Final = "오른쪽으로 갈수록 험해지는 언덕길을 더합니다. 수치는 아래 '생성기' 탭에서 고칩니다."
EDC_HINT_MOTOR: Final = "회전 관절(회전, 회전 스프링)을 누르면 모터를 답니다. 출력 +는 반시계."
EDC_LABEL_HILLS: Final = "언덕길"
EDC_LABEL_MOTOR: Final = "모터"
EDC_ERR_MOTOR_JOINT: Final = "모터는 회전 관절(회전, 회전 스프링)에만 달 수 있습니다."

# M5a 에디터 화면 (lead)
ED_BUILTIN_LANDER: Final = "달 착륙선"
ED_BUILTIN_TRACK: Final = "트랙 자동차"
ED_BUILTIN_HILL: Final = "언덕 넘는 자동차"
ED_COORDS: Final = "x {x:.2f} · y {y:.2f} m"
ED_NEW: Final = "새로 만들기"
ED_OPEN: Final = "열기…"
ED_SAVE: Final = "저장"
ED_SAVE_AS: Final = "다른 이름으로 저장…"
ED_FROM_BUILTIN: Final = "내장 예제에서 시작"
ED_UNDO: Final = "되돌리기"
ED_REDO: Final = "다시하기"
ED_COPY: Final = "복사"
ED_PASTE: Final = "붙여넣기"
ED_PLAY: Final = "직접 조종"
ED_TO_LAB: Final = "실험실로 보내기"
ED_STATUS: Final = "검사: 오류 {errors}개, 경고 {warnings}개 (아래 '검사 결과' 탭)"
ED_STATUS_OK: Final = "검사: 문제 없음"
ED_UNSAVED_TITLE: Final = "저장하지 않은 고침"
ED_UNSAVED: Final = "'{name}'에 저장하지 않은 고침이 있습니다. 저장할까요?"
ED_OPEN_FAILED: Final = "환경 파일을 열 수 없습니다"
ED_SAVE_FAILED: Final = "저장하지 못했습니다"
ED_PASTE_NOTHING: Final = "붙여 넣을 부품이 없습니다(진화 실험실에서 복사한 부품만 붙여 넣을 수 있습니다)."
ED_HAS_ERRORS: Final = "환경에 오류가 있습니다. 아래 '검사 결과' 탭을 보고 고친 뒤 다시 하세요."
MAIN_ENV_SENT: Final = "에디터에서 고친 환경을 멈춘 학습에 보냈습니다. 적용되면 '이어서'를 누르세요."
SPEC_CHECKPOINTS_LOOP_MIN: Final = "순환 체크포인트 '{id}'에는 선이 3개 이상 있어야 합니다(앞뒤로 흔들어 바퀴를 세지 못하게)."

# 설정 (lead)
MENU_SETTINGS: Final = "설정"
MENU_DATA_DIR: Final = "데이터 폴더 바꾸기…"
MENU_DATA_DIR_RESET: Final = "데이터 폴더를 기본(문서\\진화실험실)으로"
MAIN_DATA_DIR_SET: Final = "데이터 폴더: {path} (새 실행부터 이 폴더에 저장합니다. 예전 실행은 그대로 둡니다)"
LAB_OTHER_PLATFORM: Final = "이 실행은 다른 PC나 다른 버전에서 만들어졌습니다({what}). 이어서 학습할 수 있지만 비트 단위 재현은 보장되지 않습니다."
REC_BUNDLE: Final = "재현 묶음…"
REC_BUNDLE_TIP: Final = "이 실행을 같은 조건으로 다시 돌릴 수 있는 zip(설정, 시드, 환경 버전, 변경 기록)을 만듭니다."
REC_CSV: Final = "통계 CSV…"
REC_CSV_TIP: Final = "세대별 통계 전체를 CSV로 저장합니다(엑셀에서 바로 열림)."
REC_EXPORT_DONE: Final = "저장했습니다: {path}"
REC_EXPORT_FAILED: Final = "저장하지 못했습니다"
REC_ZIP_FILTER: Final = "재현 묶음 (*.zip)"
REC_CSV_FILTER: Final = "CSV (*.csv)"

# 리뷰 수정 lab (실험실·영상·주 창)
RVL_RESUME_BUSY: Final = "이 실행은 지금 다른 곳(비교 화면의 반복 실행 등)에서 학습 중이라 이어서 학습할 수 없습니다. 그 학습이 끝난 뒤 다시 하세요."
RVL_CONTINUE_BUSY_TIP: Final = "지금 다른 곳(비교 화면의 반복 실행 등)에서 학습 중인 실행이라 이어서 학습할 수 없습니다. 타임라인과 '1세대 대표와 지금 챔피언'으로 보기만 할 수 있습니다."
RVL_SCENE_FAILED: Final = "이 세대의 관전 장면을 만들지 못해 건너뜁니다: {reason}"

# 리뷰 수정 environment editor
RVE_BAD_ID: Final = "이름(id)은 영문자, 한글, _로 시작하고 영문자, 한글, 숫자, _만 쓸 수 있습니다(40자까지)."
RVE_RANGE_VALUE: Final = "값"
RVE_OUT_OF_RANGE: Final = "허용 범위를 벗어났습니다: {range}"
RVE_SAVE_INVALID: Final = "구조 오류가 있어 저장하지 않았습니다. 이대로 저장하면 이 파일을 다시 열 수 없습니다.\n아래 오류를 고친 뒤(또는 되돌리기로 되돌린 뒤) 다시 저장하세요.\n\n{errors}"

# 리뷰 수정 misc (직접 조종 기록 위치, 명령줄 인수 검사, 변경 지점 기록)
RVM_CLI_BAD_ARGS: Final = "명령줄 인수가 잘못되었습니다: {detail}"
RVM_CLI_NOT_INT: Final = "정수가 아닙니다: '{value}'"
RVM_CLI_TOO_SMALL: Final = "{minimum} 이상이어야 합니다: {value}"
RVM_CLI_NOT_NUMBER: Final = "숫자가 아닙니다: '{value}'"
RVM_CLI_BAD_WORKER_LIST: Final = "워커 수 목록은 1 이상의 정수를 쉼표로 이어 써야 합니다(예: 1,2,4): '{value}'"
RVM_CLI_OPTION_TOO_SMALL: Final = "{name} 값은 {minimum} 이상이어야 합니다: {value}"

# 2차 검토 수정
# argparse가 직접 만드는 문장(사용법, 도움말 제목, 인수 오류). 키는 argparse(파이썬 3.12)의 gettext 원문 그대로다.
# 명령줄 파서(cli._Parser)가 일하는 동안에만 이 번역을 쓴다.
RV2_ARGPARSE_TEXT: Final = {
    "usage: ": "사용법: ",
    "positional arguments": "위치 인수",
    "options": "옵션",
    "show this help message and exit": "이 도움말을 보여 주고 끝냅니다",
    "argument %(argument_name)s: %(message)s": "%(argument_name)s 인수: %(message)s",
    "the following arguments are required: %s": "다음 인수가 필요합니다: %s",
    "one of the arguments %s is required": "다음 인수 중 하나가 필요합니다: %s",
    "unrecognized arguments: %s": "알 수 없는 인수입니다: %s",
    "invalid choice: %(value)r (choose from %(choices)s)": "고를 수 없는 값입니다: %(value)r (고를 수 있는 값: %(choices)s)",
    "unknown parser %(parser_name)r (choices: %(choices)s)": "알 수 없는 명령입니다: %(parser_name)r (고를 수 있는 명령: %(choices)s)",
    "invalid %(type)s value: %(value)r": "%(type)s 값으로 읽을 수 없습니다: %(value)r",
    "expected one argument": "값이 하나 필요합니다",
    "expected at most one argument": "값은 하나까지만 줄 수 있습니다",
    "expected at least one argument": "값이 하나 이상 필요합니다",
    "expected %s argument": "값이 %s개 필요합니다",
    "expected %s arguments": "값이 %s개 필요합니다",
    "ambiguous option: %(option)s could match %(matches)s": "줄인 옵션 이름 %(option)s(이)가 여러 옵션과 맞습니다: %(matches)s",
    "ignored explicit argument %r": "이 옵션은 값을 받지 않습니다: %r",
    "not allowed with argument %s": "%s 인수와 함께 쓸 수 없습니다",
    "unexpected option string: %s": "예상하지 못한 옵션입니다: %s",
}
RV2_CLI_COMMAND_METAVAR: Final = "명령"

# ---------------------------------------------------------------------------
# v1.1 학습 안정성: 정체하면 새로 시작, 예선 (명세: 학습 방법, 부록 A25~A32)
# 용어: 예선 단계의 학습 방법은 "출전 1…K", 본선의 학습 방법은 "시도 n"(재시작 수 + 1). 내부 번호는 화면에 쓰지 않는다.
# ---------------------------------------------------------------------------
# 실험실 설정
LAB_STABILITY_GROUP: Final = "학습 안정성"
LAB_RESTART: Final = "정체하면 새로 시작"
LAB_RESTART_TIP: Final = ("지금 학습 방법이 낸 최고 검증 점수가 아래 학습 시험 횟수 동안 오르지 않으면, 챔피언과 기록은 그대로 두고 "
                          "학습 방법만 새 시드로 다시 만듭니다. 멈춘 상태에서 바꾸고 「적용」할 수 있고, 켜면 그 뒤 첫 검증부터 셉니다.")
LAB_RESTART_PATIENCE: Final = "정체 판단 학습 시험"
LAB_UNIT_TRIALS: Final = " 회"
LAB_RESTART_FROM: Final = "새로 시작할 곳"
RESTART_FROM_NAMES: Final = {"fresh": "처음 상태(새 시드)", "champion": "챔피언에서(새 시드)",
                             "alternate": "번갈아(처음 상태 → 챔피언)"}
RESTART_FROM_NAMES_SHORT: Final = {"fresh": "처음 상태", "champion": "챔피언"}
LAB_RESTART_FROM_TIP: Final = ("처음 상태: 몸 초깃값·뇌 0에서 다른 시드로. 챔피언에서: 챔피언 유전자를 출발점으로, 탐색 폭은 처음 값으로. "
                               "번갈아: 처음 상태와 챔피언을 차례로.")
LAB_QUALIFY: Final = "예선 출전 수"
LAB_QUALIFY_OFF: Final = "끔"
LAB_QUALIFY_TRIALS: Final = "출전마다 학습 시험"
LAB_QUALIFY_TIP: Final = ("시드만 다른 학습 방법 여러 개를 처음에 하나씩 차례로 짧게 돌려 보고, 검증 점수가 가장 높았던 하나로 "
                          "계속합니다. 운 나쁜 출발에 오래 갇히지 않게 합니다. 예선 시험도 누적 시험 횟수에 들어갑니다. "
                          "새 실행을 시작할 때만 정할 수 있습니다.")
LAB_STABILITY_HINT: Final = "예선 합계 학습 시험 {total:,}회(약 {qgens}세대) · 정체 판단 약 {rgens}세대"
LAB_STABILITY_HINT_NO_QUALIFY: Final = "정체 판단 약 {rgens}세대"
LAB_STABILITY_STOP_FIRST: Final = " · 정체 멈춤({stop}세대)이 자동 재시작보다 먼저 옵니다"
# 실험실 상태 글
LAB_STATUS_QUALIFY: Final = "예선 출전 {n}/{k} · 출전 시험 {t:,}/{b:,}회"
LAB_STATUS_ATTEMPT: Final = "시도 {n} · 정체 {stall:,}/{p:,}회"
LAB_STATUS_ATTEMPT_OFF: Final = "시도 {n} · 자동 재시작 꺼짐"
# 학습 방법 교체 알림(실험실 상태 글과 명령줄이 함께 씀)
TRAIN_SWITCH_RESTART: Final = ("{gen}세대: 학습 시험 {stall:,}회 동안 나아지지 않아 {where}에서 새로 시작했습니다(시도 {n}). "
                               "챔피언은 그대로입니다.")
TRAIN_SWITCH_QUALIFY_NEXT: Final = "{gen}세대: 예선 출전 {n} 끝(최고 검증 {score}) → 출전 {next}"
TRAIN_SWITCH_QUALIFY_PICK: Final = "{gen}세대: 예선 끝 → 출전 {winner}(검증 {score})로 계속합니다"
# 대시보드
DASH_MARK_TIP_RESTART: Final = "{gen}세대 끝: {where}에서 새로 시작 · 정체 {stall:,}회 · 이 학습 방법 최고 {best}"
DASH_MARK_TIP_QUALIFY_NEXT: Final = "{gen}세대 끝: 예선 출전 {n} 끝(최고 {best})"
DASH_MARK_TIP_QUALIFY_PICK: Final = "{gen}세대 끝: 예선 끝, 출전 {winner}로 계속"
DASH_TREND_LEARNER: Final = "지금 학습 방법 최고(검증)"
DASH_QUALIFY_LEGEND: Final = "예선"
# 설정 검사
TRAIN_RESTART_TOO_SHORT: Final = "정체 판단 학습 시험({value:,}회)은 검증 주기({every:,}회) 이상이어야 합니다."
TRAIN_QUALIFY_TOO_SHORT: Final = "출전마다 학습 시험({value:,}회)은 검증 주기({every:,}회) 이상이어야 합니다."
TRAIN_QUALIFY_ENTRANTS: Final = "예선 출전 수는 1~{max}이어야 합니다(1이면 예선 없음). 지금 {value}"
TRAIN_QUALIFY_OVER_BUDGET: Final = ("예선(출전 {k}개 × 학습 시험 {b:,}회 = {total:,}회)이 최대 시험 횟수({max:,}회) 이상입니다. "
                                    "예선을 줄이거나 끄세요.")
TRAIN_ENV_DURING_QUALIFY: Final = "예선 중에는 환경을 바꿀 수 없습니다(지금 출전 {n}/{k}). 예선이 끝난 뒤 바꾸세요."
TRAIN_RESUME_QUALIFY_FIXED: Final = "이어 할 때는 예선 설정을 바꿀 수 없습니다: {flags}. 새 실행으로 시작하세요."
# 명령줄
CLI_HELP_RESTART: Final = "지금 학습 방법의 최고 검증 점수가 이 학습 시험 횟수 동안 오르지 않으면 학습 방법만 새로 시작(새 실행 기본 20000)"
CLI_HELP_NO_RESTART: Final = "자동 재시작을 끈다"
CLI_HELP_RESTART_FROM: Final = "새로 시작할 곳: fresh(처음 상태), champion(챔피언에서), alternate(번갈아)"
CLI_HELP_QUALIFY: Final = "예선 출전 수(1이면 예선 없음, 기본 1)"
CLI_HELP_QUALIFY_TRIALS: Final = "예선 출전마다 학습 시험 횟수(기본 12000)"
TRAIN_STABILITY: Final = "자동 재시작: {restart} · 예선: {qualify}"
TRAIN_STABILITY_RESTART_ON: Final = "학습 시험 {p:,}회 정체 시 {where}에서"
TRAIN_STABILITY_OFF: Final = "끔"
TRAIN_STABILITY_QUALIFY_ON: Final = "출전 {k}개 × 학습 시험 {b:,}회"
TRAIN_PROGRESS_QUALIFY: Final = " · 예선 {n}/{k}"
TRAIN_PROGRESS_ATTEMPT: Final = " · 시도 {n}"
TRAIN_FINISHED_RESTARTS: Final = " · 새로 시작 {n}번"
TRAIN_CONFIG_CHANGED: Final = "설정 변경({gen}세대부터): {changes}"
# 반복 실행
CMP_REPEAT_RESTART: Final = "정체하면 새로 시작"
CMP_REPEAT_QUALIFY_OFF_HINT: Final = "예산이 작아 예선을 껐습니다(예선 {total:,}회가 예산의 절반을 넘음)."
LAB_STABILITY_HINT_QUALIFY_ONLY: Final = "예선 합계 학습 시험 {total:,}회(약 {qgens}세대)"
CLI_RESTART_TOO_SHORT_HINT: Final = "--restart 횟수로 늘리거나 --no-restart로 끄세요."

# ---------------------------------------------------------------------------
# v1.2 보는 재미: 2차 그래프 다섯 개와 나란히 경주 (0단계: 설계 6절 전체 목록)
# 대시보드 타일 제목은 위 DASH_TILE_TITLES에 더했다.
# ---------------------------------------------------------------------------
# v1.2 관찰 기록 (engine/trainer.py, engine/session.py, engine/commands.py, cli.py)
TRAIN_OBSERVE_FAILED: Final = ("그래프용 관찰 기록을 만들지 못했습니다. 학습은 그대로 이어지고 이 세대는 새 그래프에서 "
                               "빠집니다: {reason}")
BENCH_OBSERVE: Final = "관찰 기록을 켠 단일 코어: 초당 {sps:,.0f}스텝 ({pct:+.1f}%)"
CLI_HELP_NO_OBSERVE: Final = ("그래프용 관찰 기록(집단 유전자, 행동 분포, 궤적 밀도)을 남기지 않는다. 학습 결과는 같다. "
                              "실험실에서 이어 가면 그 세대부터 다시 남긴다")

# v1.2 대시보드 공통 (app/lab/dashboard.py)
DASH_PAGES: Final = {"learning": "학습 그래프", "fun": "보는 재미"}
DASH_PAGES_TIP: Final = {
    "learning": "점수, 탐색 폭, 종료 원인처럼 학습이 잘 되고 있는지 보는 그래프입니다.",
    "fun": "집단이 어디로 모이는지, 뇌가 어떻게 움직이는지, 어디를 지나가는지 구경하는 그래프입니다.",
}
DASH_FUN_NEW_TIP: Final = "새 그래프: 유전자 지도, 뇌 들여다보기, 궤적 밀도, 행동 분포, 몸 수치 추이"
DASH_TILE_HELP: Final = {
    "genemap": ("유전자는 수백 개의 숫자라 그대로는 그릴 수 없어서, 집단이 가장 크게 달라지는 두 방향으로 눌러 펼쳤습니다. "
                "점 하나가 개체 하나이고, 가까운 점일수록 비슷한 개체입니다. 축의 숫자에는 뜻이 없습니다."),
    "brain": ("신경망은 왼쪽의 센서 값을 받아 가운데 뉴런들을 거쳐 오른쪽 구동기 출력을 냅니다. 선은 연결이고, 학습은 이 "
              "연결의 세기를 바꾸는 일입니다. 재생 중에는 뉴런 색이 그 순간의 값을 보여 줍니다."),
    "density": ("시험 0번(관전 화면과 같은 세계)에서 집단 전체가 각 자리를 지나간 횟수를 칠했습니다. 세대가 지나며 색이 목표 "
                "쪽으로 모이는지 보세요."),
    "actions": ("구동기 출력 세기를 20칸으로 나눠, 그 세기로 쓴 시간이 얼마인지 세대마다 칠했습니다. 예를 들어 착륙선이 주 "
                "추진기를 얼마나 세게, 얼마나 자주 켜는지 보입니다."),
    "body": ("진화하는 몸 수치(바퀴 크기, 다리 길이 등)의 집단 평균(선)과 가장 작은 값~가장 큰 값(띠)입니다. 띠가 좁아지면 "
             "집단이 한 몸 모양으로 모인 것입니다."),
}
DASH_LOADING: Final = "기록을 읽는 중… ({done}/{total}세대)"
DASH_OLD_RUN_NOTE: Final = "v1.2보다 먼저 학습한 세대라 집단 기록이 없어 세대 대표만 보입니다."
DASH_OBSERVE_FROM: Final = "{gen}세대부터 집단 기록이 있습니다(그 앞은 세대 대표만)."
DASH_NO_RECORD: Final = "이 세대에는 기록이 없습니다."
DASH_GENOME_STRIDE: Final = "집단이 커서 유전자는 {k}세대마다 남깁니다. 가까운 기록: {gen}세대"
DASH_FOCUS_CURSOR: Final = "보는 세대"

# 유전자 지도 (app/lab/watch_tiles.py, app/lab/history.py)
GMAP_EMPTY: Final = "학습을 시작하면 세대마다 집단이 이 지도 위의 점으로 나타납니다. 가까운 점일수록 비슷한 개체입니다."
GMAP_AXIS_X: Final = "지도 가로: 가장 크게 달라지는 방향 ({pct:.0f}%)"
GMAP_AXIS_Y: Final = "지도 세로: 그다음으로 달라지는 방향 ({pct:.0f}%)"
GMAP_COLOR_LABEL: Final = "색"
GMAP_COLOR_MODES: Final = {"score": "점수", "success": "성공 비율"}
GMAP_COLOR_TIP: Final = "점 색을 점수(색 막대)로 칠할지, 학습 시험에서 성공한 비율로 칠할지 고릅니다."
GMAP_SCORE_BAR: Final = "점수"
GMAP_SUCCESS_BAR: Final = "성공 비율"
GMAP_REFIT: Final = "지도 다시 맞추기"
GMAP_REFIT_TIP: Final = "지금까지의 세대 전체를 보고 지도 방향을 다시 잡습니다. 기록 세대 수가 두 배가 될 때마다 저절로도 다시 잡습니다."
GMAP_REFITTED: Final = "지도를 다시 맞췄습니다({n}세대 기준)."
GMAP_BIG_DIM: Final = "유전자가 커서({dim}차원) 지도를 표본으로 맞추는 중…"
GMAP_LEGEND: Final = "{gen}세대 · 개체 {n}"
GMAP_REP: Final = "세대 대표"
GMAP_CHAMPION: Final = "그때 챔피언"
GMAP_START: Final = "출발점"
GMAP_TRAIL: Final = "집단 가운데가 지나온 길"
GMAP_GHOST: Final = "앞 세대"
GMAP_HINT: Final = "점에 마우스를 올리면 개체 정보가 보이고, 흐린 점을 누르면 그 세대로, 두 번 누르면 그 개체를 다시 봅니다."
GMAP_POINT_TIP: Final = "{gen}세대 {index}번 개체 · 점수 {score} · 시험 성공 {success}/{trials}"
GMAP_POINT_TIP_REP: Final = "{gen}세대 대표({index}번 개체) · 점수 {score}"
GMAP_CHAMPION_TIP: Final = "그때 챔피언({gen}세대에 뽑힘) · 검증 점수 {score}"
GMAP_START_TIP: Final = "출발점: 처음 유전자(몸은 처음 값, 뇌는 모두 0)"
GMAP_LABEL_INDIVIDUAL: Final = "{gen}세대 {index}번 개체"
GMAP_INEXACT: Final = "이 세대는 정확한 유전자를 메모리에 두지 않아 {gen}세대 대표를 대신 재생합니다."

# 몸 수치 추이 (app/lab/watch_tiles.py)
BODY_EMPTY_NONE: Final = ("이 환경은 몸 수치를 진화시키지 않습니다. 에디터에서 숫자 속성 옆 「진화」를 켜면 몸이 어떻게 "
                          "바뀌는지 여기에 나옵니다.")
BODY_EMPTY: Final = "학습을 시작하면 진화하는 몸 수치의 집단 평균과 범위(가장 작은 값~가장 큰 값)를 그립니다."
BODY_PICK_LABEL: Final = "수치"
BODY_PICK_TIP: Final = "볼 몸 수치를 고릅니다. '모두'는 수치마다 진화 범위를 0~100%로 맞춰 함께 그립니다."
BODY_ALL: Final = "모두(범위 대비 %)"
BODY_AXIS_ALL: Final = "진화 범위 안 위치(%)"
BODY_MEAN: Final = "평균"
BODY_RANGE: Final = "가장 작은 값~가장 큰 값"
BODY_REP: Final = "세대 대표"
BODY_LIMITS: Final = "진화 범위 끝"
BODY_HOVER: Final = "{name}: 평균 {mean} ({lo}~{hi}) · 대표 {rep}"
BODY_REPS_ONLY: Final = "집단 기록이 없는 세대는 세대 대표의 값만 보입니다."

# 뇌 들여다보기 (app/lab/brain_view.py)
BRAIN_EMPTY: Final = "학습을 시작하면 지금 가장 잘하는 개체의 뇌(신경망)를 그립니다. 재생하면 뉴런이 켜지는 모습이 보입니다."
BRAIN_TITLE_CHAMPION: Final = "그때 챔피언({gen}세대) · 멈춘 그림"
BRAIN_TITLE_LIVE: Final = "{gen}세대 대표 · 관전 화면과 같은 시험 · {t:.1f}초"
BRAIN_TITLE_PLAYING: Final = "{label} · {t:.1f}초"
BRAIN_PREPARING: Final = "뉴런 움직임을 계산하는 중…"
BRAIN_LEGEND: Final = "파랑 = 더하는 연결, 주황 = 빼는 연결, 굵을수록 셈 · 뉴런 색이 강할수록 지금 값이 큼"
BRAIN_LEGEND_STILL: Final = "파랑 = 더하는 연결, 주황 = 빼는 연결, 굵을수록 셈 · 재생하면 뉴런이 켜집니다"
BRAIN_RELU_NOTE: Final = "은닉층이 relu라 층마다 그 재생에서 가장 큰 값을 가장 진한 색으로 칠했습니다."
BRAIN_INPUTS: Final = "입력(센서)"
BRAIN_HIDDEN: Final = "층 {k}"
BRAIN_OUTPUTS: Final = "출력(구동기)"
BRAIN_EDGES_CAPPED: Final = "연결 {total:,}개 중 층마다 센 것 {per:,}개까지만 그렸습니다."
BRAIN_NODE_INPUT_TIP: Final = "입력 {name} ({label}) · 지금 값 {value}"
BRAIN_NODE_HIDDEN_TIP: Final = "층 {k} 뉴런 {j} · 지금 값 {value} · 편향 {bias}"
BRAIN_NODE_OUTPUT_TIP: Final = "출력 {name} ({label}) · 지금 값 {value} · 편향 {bias}"

# 궤적 밀도 (app/lab/watch_tiles.py)
DENS_EMPTY: Final = "학습을 시작하면 집단이 지나간 곳을 열화상처럼 칠합니다(시험 0번, 관전 화면과 같은 세계)."
DENS_WINDOW_LABEL: Final = "합치기"
DENS_WINDOWS: Final = {"one": "이 세대만", "recent": "최근 {n}세대", "all": "처음부터"}
DENS_WINDOW_TIP: Final = "시험 세계가 같은 세대끼리만 합칩니다(트랙처럼 세계가 그대로인 환경)."
DENS_CANNOT_MERGE: Final = "세대마다 시험 세계(지형·트랙)가 달라 이 세대만 보입니다. '세대 넘겨 보기'로 바뀌는 모습을 보세요."
DENS_LEGEND: Final = "{gen}세대 · 시험 0번 · 개체 {n} · 칸 {cell:.1f}m · 색 막대의 위쪽 색일수록 자주 지나간 곳"
DENS_LEGEND_MERGED: Final = "{first}~{last}세대 중 같은 세계 {count}세대 합침 · 칸 {cell:.1f}m"
DENS_OUTSIDE: Final = " · 격자 밖 {pct:.0f}%"
DENS_BAR: Final = "지나간 횟수"
DENS_HOVER: Final = "x {x:.1f}m, y {y:.1f}m · 지나간 횟수 {count} (전체의 {pct:.1f}%)"
DENS_NONE: Final = "이 세대에는 궤적 기록이 없습니다."
DENS_PREPARING: Final = "세계 모양을 만드는 중…"

# 행동 분포 (app/lab/watch_tiles.py)
ACT_EMPTY: Final = "학습을 시작하면 구동기(추진기, 모터, 바퀴)를 얼마나 세게, 얼마나 자주 쓰는지 보여 줍니다."
ACT_CHANNEL_LABEL: Final = "구동기"
ACT_CHANNEL_TIP: Final = "볼 구동기 출력을 고릅니다."
ACT_MODES: Final = {"pop": "집단 전체", "rep": "세대 대표"}
ACT_MODE_TIP: Final = "집단 전체가 쓴 시간을 셀지, 세대 대표(그 세대 학습 점수 1등) 하나만 셀지 고릅니다."
ACT_AXIS: Final = "출력 세기"
ACT_FRACTION: Final = "시간 비율"
ACT_MEAN_LINE: Final = "평균 출력"
ACT_HINT: Final = "색 막대의 위쪽 색일수록 그 세기로 오래 썼습니다. 쉬는 값(0) 둘레 칸은 거의 쓰지 않은 시간입니다."
ACT_HOVER: Final = "{gen}세대 · 출력 {lo:+.2f}~{hi:+.2f} · 시간 {pct:.0f}%"
ACT_SUMMARY: Final = "{name}: 쓰는 시간 {on:.0f}% · 평균 출력 {mean:+.2f} ({gen}세대, 모든 학습 시험)"
ACT_CHANNEL_SUFFIX: Final = {"throttle": "가속", "steer": "조향"}
ACT_CHANNEL_NAME: Final = "{label} · {suffix}"
ACT_NONE: Final = "이 세대에는 행동 기록이 없습니다."

# 실험실 타임라인 (app/lab/view.py)
LAB_TIMELINE_OF: Final = "세대 {gen} / {total}"
LAB_TIMELINE_TIP: Final = ("보는 세대를 고릅니다. '보는 재미' 그래프가 이 세대를 보여 주고, '이 세대 대표 재생'은 이 세대 대표를 "
                           "다시 돌립니다. 맨 오른쪽에 두면 새 세대를 따라갑니다.")
LAB_FLIP: Final = "세대 넘겨 보기"
LAB_FLIP_STOP: Final = "멈춤"
LAB_FLIP_TIP: Final = ("슬라이더를 자동으로 한 세대씩 넘기며 '보는 재미' 그래프가 어떻게 바뀌는지 보여 줍니다. 전체를 20초 "
                       "안팎에 넘기고, 맨 끝에서 누르면 1세대부터 시작합니다.")
LAB_RACE_GENERATIONS: Final = "세대끼리 경주"
LAB_RACE_GENERATIONS_TIP: Final = "1세대, 중간 세대들, 마지막 세대의 대표와 지금 챔피언을 같은 코스에서 나란히 달리게 합니다."

# 나란히 경주 (app/race/view.py, app/race/model.py, app/compare/view.py, app/export/video.py)
CMP_RACE_BUTTON: Final = "나란히 경주"
CMP_RACE_BUTTON_TIP: Final = "고른 실행(2~6개)의 챔피언을 같은 코스(같은 시작 조건)에서 나란히 달리게 합니다."
RACE_TITLE: Final = "나란히 경주"
RACE_TITLE_GENERATIONS: Final = "세대끼리 경주 · {run}"
RACE_COURSE_LABEL: Final = "코스"
RACE_COURSE_ITEM: Final = "코스 {n}"
RACE_COURSE_TIP: Final = ("모든 칸이 같은 시작 조건(시드)에서 달립니다. 코스 1~32는 챔피언을 고를 때 쓰지 않은 비교 평가 "
                          "시드라 어느 칸에도 유리하지 않습니다.")
RACE_NEXT_COURSE: Final = "다른 코스"
RACE_WHO_LABEL: Final = "누구를"
RACE_WHO: Final = {"last": "마지막 챔피언", "budget": "같은 학습 시험 수일 때의 챔피언"}
RACE_WHO_TIP: Final = ("같은 학습 시험 수: 가장 짧게 학습한 실행의 누적 학습 시험({trials:,}회) 시점에서 실행마다 그때의 "
                       "챔피언끼리 겨룹니다(학습 방법끼리 공정하게).")
RACE_VIEW_LABEL: Final = "보기"
RACE_VIEW: Final = {"split": "나란히", "overlay": "한 화면에 겹쳐서"}
RACE_OVERLAY_OFF_TIP: Final = "칸마다 환경이 달라 한 화면에 겹쳐 볼 수 없습니다."
RACE_SPEED_LABEL: Final = "배속"
RACE_SPEED_ITEM: Final = "{n:g}배"
RACE_START: Final = "출발"
RACE_PAUSE: Final = "일시 정지"
RACE_RESTART: Final = "처음부터"
RACE_SAVE_GIF: Final = "경주 영상 저장…"
RACE_GIF_NAME: Final = "경주_코스{n}_{stamp}.gif"
RACE_PREPARING: Final = "경주 준비 중… ({done}/{total})"
RACE_LANE_RUN: Final = "{method} · 시드 {seed}"
RACE_LANE_RUN_ENV: Final = "{env} · {method} · 시드 {seed}"
RACE_LANE_GEN: Final = "{gen}세대 대표"
RACE_LANE_NOW: Final = "지금 챔피언({gen}세대)"
RACE_LANE_CHAMPION: Final = "{gen}세대 챔피언 · 학습 시험 {evals:,}회"
RACE_LANE_REP_INSTEAD: Final = "{gen}세대 대표(그때 챔피언 없음)"
RACE_RUNNING: Final = "달리는 중 · {t:.1f}초"
RACE_LANE_DONE: Final = "끝 · {end} · {t:.1f}초 · 점수 {score:.1f}"
RACE_LANE_RANK: Final = "{rank}위 · {end} · {t:.1f}초 · 점수 {score:.1f}"
RACE_LANE_RANK_TIE: Final = "공동 {rank}위 · {end} · {t:.1f}초 · 점수 {score:.1f}"
RACE_RESULT: Final = "결과: {order}"
RACE_RESULT_ITEM: Final = "{rank}위 {name}"
RACE_RESULT_SEP: Final = " · "
RACE_DIFFERENT_ENV: Final = "칸마다 학습한 환경이 달라 같은 코스 번호라도 세계가 다를 수 있습니다. 순위는 매기지 않습니다."
RACE_TOO_FEW: Final = "경주할 실행을 2개 이상 고르세요."
RACE_TOO_MANY: Final = "경주는 한 번에 {max}개까지입니다. 앞의 {max}개만 넣었습니다."
RACE_NEED_GENS: Final = "세대 대표가 2세대 이상 있어야 세대끼리 경주할 수 있습니다."
RACE_NO_CHAMPION: Final = "{name}: 그 시점까지 챔피언이 없어 세대 대표({gen}세대)로 달립니다."
RACE_SKIPPED: Final = "{name}: 재생할 유전자를 읽지 못해 뺐습니다({reason})."
RACE_FAILED: Final = "경주를 준비하지 못했습니다: {reason}"
RACE_HUD: Final = "코스 {n} · {t:.1f}초 · {speed:g}배"
VIDEO_RACE_TITLE: Final = "나란히 경주 · 코스 {n}"

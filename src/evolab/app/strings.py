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
PLAY_HELP: Final = "조종 키\n{keys}\n\nR 다시 시작 · N 새 시드 · {pause} 일시정지 · 마우스 휠 확대"
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
SPEC_KEY_RESERVED: Final = "'{key}' 키는 직접 조종 창 제어용(R 다시 시작, N 새 시드)이라 구동기에 쓸 수 없습니다."
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
METHOD_NAMES: Final = {"cmaes": "CMA-ES", "ga": "유전 알고리즘", "sa": "담금질",
                       "grow": "CMA-ES · 자라는 뇌"}  # v6.1: 작게 시작해 자라는 뇌(CMA-ES만, learners/grow.py)
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
BENCH_SINGLE: Final = ("단일 코어(창 {window}): 초당 {sps:,.0f}스텝 ({us:.1f}µs/스텝), 신경망 계산 비율 {nn:.1f}%, "
                       "{steps:,}스텝 {sec:.2f}초")  # v6.0: 「창 {window}」(v6.0 전 보고서의 160쌍 한 묶음 값과 바로 비교하지 않게)
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
    "structure": "몸 구조",  # v2.1
    "brainsize": "뇌 크기",  # v6.1 (app/lab/grow_tile.py, 「보는 재미」 쪽 「뇌 들여다보기」 바로 뒤)
}
DASH_AXIS_LABEL: Final = "가로축"
DASH_AXIS_NAMES: Final = {"generation": "세대", "evaluations": "누적 시험 횟수", "elapsed": "경과 시간(초)"}
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
LAB_START: Final = "▶ 학습 시작"  # v3.0(설계 D17): 원래 「새로 시작」(「정체하면 새로 시작」 칸과 헷갈림)
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
VIDEO_SAVE: Final = "내보내기…"                       # v3.1(설계 6절, D48): 원래 「영상 저장…」. 형식은 내보내기 대화 상자가 고른다
VIDEO_SAVE_TITLE: Final = "내보내기"                  # v3.1: 원래 「영상 저장 (GIF)」(대화 상자·오류 상자 제목)
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
    # v2.0 움직이는 세계와 공기
    "motion": "움직임", "PathMotion.points": "지날 점(물체 위치 기준)", "speed": "속력", "accel": "가속(초당)",
    "start": "출발(초)", "mode": "방식", "spin": "회전 속도(rad/s)", "period": "주기(초)", "phase": "위상(주기 비율)",
    "WaveMotion.amplitude": "흔들림 폭 (x, y)",
    "Jump.speed": "점프 속도(m/s)", "Jump.dir": "뛰는 방향(라디안)", "ground": "땅으로 볼 것", "cooldown": "다시 뛰기까지(초)",
    "Aero.density": "공기 밀도", "span": "날개 폭(화면 안쪽)", "aspect_ratio": "가로세로비", "stall": "실속 각도",
    "Aero.friction": "표면 마찰", "surfaces": "공기를 받는 모양",
    "Wing.le": "앞전 (x, y)", "chord": "시위 길이", "Wing.angle": "붙임각(라디안)", "thick": "두께(시위 비율)",
    "camber": "휨(시위 비율)",
    "Box.density": "이 모양 밀도", "Circle.density": "이 모양 밀도", "Poly.density": "이 모양 밀도",
    "Radial.density": "이 모양 밀도", "Wing.density": "이 모양 밀도",
    "ObstacleGen.count": "장애물 수", "ObstacleGen.x": "첫 장애물 x", "ObstacleGen.y": "바닥 y", "gap": "간격(초)",
    "ObstacleGen.width": "폭 범위", "ObstacleGen.height": "높이 범위", "air_chance": "공중 장애물 비율",
    "air_y": "공중 장애물 높이", "air_height": "공중 장애물 두께", "ObstacleGen.speed": "처음 속도",
    "ObstacleGen.accel": "빨라지는 정도(초당)", "ObstacleGen.direction": "미는 방향(라디안)",
    # 물체의 type 칸은 「움직임 종류」(SPEC_MOTION_NOT_KINEMATIC 문구와 같게, 움직임 칸 「움직임」과 구별)
    "Body.type": "움직임 종류",
    # v2.1 몸 구조 진화
    "exists": "있음",
}
EDP_FIELD_TIPS: Final = {
    # 칸 도움말(모델 설명 앞에 붙는다). "모델.칸"이 있으면 그것이 먼저다.
    "id": "다른 칸과 수식에서 이 부품을 가리키는 이름입니다.",
    "label": "화면에 보이는 이름입니다.",
    "dt": "물리 시간 간격(초). 기본 1/60초.",
    "max_time": "이 시간이 지나면 timeout(시간 초과) 사건으로 끝납니다.",
    "gravity": "중력 가속도 (x, y), m/s².",
    "keys": "직접 조종 키 이름을 쉼표로 나눠 씁니다(예: Up, W). R, N은 창 제어용이라 쓸 수 없고, Space는 점프 추진기에만 쓸 수 있습니다.",
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
    # v2.0 움직이는 세계와 공기 (묶은 exe에는 model.py 설명이 없으므로 새 칸마다 둔다)
    "motion": "정해진 움직임 물체가 시간에 따라 움직이는 방법입니다. 모든 개체가 같은 움직임을 봅니다.",
    "PathMotion.points": "지나갈 점들입니다. 물체 위치를 기준으로 적고 첫 점은 보통 (0, 0)입니다.",
    "mode": "한 번: 끝에서 멈춤. 돌기: 끝에서 첫 점으로 이어 돎. 왕복: 끝에서 되돌아옴.",
    "start": "이 시각까지 출발점에 서 있습니다. 랜덤 값을 쓰면 시험마다 달라집니다.",
    "accel": "초마다 속력이 이만큼 늘어납니다. 음수면 느려지다 멈춥니다.",
    "Jump.speed": "땅을 박차고 오르는 속도입니다. 점프 높이는 대략 속도² ÷ (2 × 중력)입니다.",
    "ground": "여기에 닿아 있을 때만 뜁니다. 비우면 공중에서도 뜁니다(날갯짓).",
    "cooldown": "한 번 뛴 뒤 다시 뛸 수 있을 때까지의 시간입니다.",
    "Aero.density": "공기의 밀도(kg/m³). 바다 높이 공기는 약 1.2입니다.",
    "span": "화면에 보이지 않는 안쪽 방향 날개 폭입니다. 공기힘이 이 폭에 비례합니다.",
    "aspect_ratio": "날개 폭 ÷ 시위입니다. 클수록 같은 각도에서 양력이 크고 유도 항력이 작습니다. 0이면 끝없이 긴 날개.",
    "stall": "이 받음각을 넘으면 흐름이 떨어져 양력이 줄고 항력이 커집니다.",
    "surfaces": "모든 모양: 사각형·다각형·방사형·날개 모양 테두리가 모두 공기를 받습니다. 날개 모양만: 동체 같은 다른 모양은 받지 않습니다.",
    "Wing.le": "날개의 앞쪽 끝(공기를 먼저 만나는 쪽)입니다. 물체 좌표로 적습니다.",
    "chord": "앞전에서 뒷전까지 길이입니다.",
    "Wing.angle": "시위가 기울어진 각도입니다. 양수면 앞전이 들립니다.",
    "thick": "날개 두께를 시위 길이에 대한 비율로 적습니다. 0보다 크면 날개 모양이 늘 올바릅니다.",
    "camber": "날개가 위로 휜 정도를 시위 길이에 대한 비율로 적습니다.",
    "Box.density": "비우면 물체 재료의 밀도를 씁니다. 작고 무거운 추로 무게 중심을 옮길 때 씁니다.",
    "Circle.density": "비우면 물체 재료의 밀도를 씁니다. 작고 무거운 추로 무게 중심을 옮길 때 씁니다.",
    "Poly.density": "비우면 물체 재료의 밀도를 씁니다. 작고 무거운 추로 무게 중심을 옮길 때 씁니다.",
    "Radial.density": "비우면 물체 재료의 밀도를 씁니다. 작고 무거운 추로 무게 중심을 옮길 때 씁니다.",
    "Wing.density": "비우면 물체 재료의 밀도를 씁니다. 작고 무거운 추로 무게 중심을 옮길 때 씁니다.",
    "gap": "앞 장애물과 이 장애물이 선수에게 닿는 시각의 차이(초)입니다. 빨라져도 뛸 틈이 남습니다.",
    "air_chance": "이 비율만큼 장애물이 공중에 뜹니다. 서 있으면 지나가고 뛰면 부딪힙니다.",
    # v2.0 0단계에서 더함: 설계 6절 목록에 없던 새 칸의 도움말(새 칸마다 도움말을 둔다는 D28·T47을 채움)
    "PathMotion.speed": "길을 따라 움직이는 처음 속력(m/s)입니다.",
    "spin": "제자리에서 도는 빠르기(rad/s)입니다. +는 반시계 방향입니다.",
    "period": "한 번 오갔다 제자리로 돌아오는 시간(초)입니다.",
    "phase": "흔들기를 주기의 몇 분의 1만큼 앞서 시작할지 적습니다(0~1).",
    "WaveMotion.amplitude": "가운데에서 한쪽 끝까지 흔들리는 거리 (x, y)입니다.",
    "Jump.dir": "뛰는 방향(라디안, 세계 기준)입니다. 1.57이면 위쪽입니다. 물체가 기울어도 바뀌지 않습니다.",
    "Jump.keys": "직접 조종 키 이름을 쉼표로 나눠 씁니다(예: Up, W, Space). 점프에는 Space도 쓸 수 있습니다. R, N은 창 제어용이라 쓸 수 없습니다.",
    "Aero.friction": "공기가 표면을 스치며 생기는 마찰 계수입니다. 보통 0.01 안팎입니다.",
    "ObstacleGen.seed": "장애물 줄을 뽑는 시드입니다. 랜덤 값(=r.이름)을 쓰면 시험마다 다른 줄이 나옵니다.",
    "ObstacleGen.count": "줄에 놓을 장애물 수입니다.",
    "ObstacleGen.x": (
        "첫 장애물이 출발하는 자리입니다. 선수 자리(x = 0)에서 장애물이 오는 쪽으로 잰 거리로, 방향이 3.14면 곧 x 위치입니다. "
        "선수 시작 위치에서 2m 넘게 떨어뜨리세요."
    ),
    "ObstacleGen.y": "장애물이 놓이는 바닥 높이입니다.",
    "ObstacleGen.width": "장애물 폭을 이 범위(최소, 최대)에서 뽑습니다.",
    "ObstacleGen.height": "땅 장애물 높이를 이 범위(최소, 최대)에서 뽑습니다.",
    "air_y": "공중 장애물 아랫면이 바닥에서 떠 있는 높이 범위입니다.",
    "air_height": "공중 장애물 두께 범위입니다.",
    "ObstacleGen.speed": "장애물 줄이 처음 다가오는 속도(m/s)입니다.",
    "ObstacleGen.accel": "초마다 장애물 줄이 이만큼 빨라집니다. 음수면 느려지다 멈춥니다.",
    "ObstacleGen.direction": (
        "장애물 줄을 미는 방향(라디안)입니다. 3.14면 오른쪽에서 왼쪽으로, 0이면 왼쪽에서 오른쪽으로 밉니다. "
        "어느 방향이든 장애물은 바닥 높이 그대로 선수 자리(x = 0)를 지납니다."
    ),
    "ObstacleGen.friction": "장애물 표면의 마찰입니다.",
    "Jump.body": "점프 추진기를 다는 에이전트 물체입니다. 이 물체가 땅에 닿아 있을 때 뜁니다.",
    "Aero.body": "공기힘을 받는 에이전트 물체입니다. 이 물체의 다각형 모양 테두리가 공기를 받습니다.",
    # v2.1 몸 구조 진화
    "exists": "0.5보다 크면 이 물체를 만듭니다. 진화를 켜면 생기고 없어지는 것까지 진화합니다. 없으면 붙은 관절·구동기·센서도 없습니다.",
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
    # v2.0 움직이는 세계와 공기 ("kinematic": "정해진 움직임"은 위에 있음). all은 다른 칸("전부")과 겹치지 않게 칸 이름을 붙임
    "path": "길", "wave": "흔들기", "once": "한 번", "loop": "돌기", "pingpong": "왕복",
    "jump": "점프 추진기", "aero": "공기역학", "wing": "날개", "obstacles": "장애물 줄",
    "surfaces.all": "모든 모양", "surfaces.wings": "날개 모양만",
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
EDP_ISSUES_SEVERITY: Final = {"error": "오류", "warning": "경고", "info": "안내"}
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
BENCH_OBSERVE: Final = ("관찰 기록을 켠 단일 코어: 초당 {sps:,.0f}스텝 ({pct:+.1f}%, 끔·켬을 번갈아 {rounds}번 잰 중앙값, "
                         "회차별 {lo:+.1f}~{hi:+.1f}%)")
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
    # v2.1: 설계 6절의 STRUCT_TIP(대시보드는 「보는 재미」 타일 제목의 풍선 도움말을 이 사전에서 읽어 따로 두지 않는다)
    "structure": ("세대마다 개체들이 어떤 몸 구조(있는 부품 모음)를 가졌는지 비율로 쌓아 보여 줍니다. 아래는 보는 세대 챔피언의 "
                  "몸입니다."),
}
DASH_LOADING: Final = "기록을 읽는 중… ({done}/{total}세대)"
DASH_OLD_RUN_NOTE: Final = "v1.2보다 먼저 학습한 세대라 집단 기록이 없어 세대 대표만 보입니다."
DASH_OBSERVE_FROM: Final = "{gen}세대부터 집단 기록이 있습니다(그 앞은 세대 대표만)."
DASH_NO_RECORD: Final = "이 세대에는 기록이 없습니다."
DASH_GENOME_STRIDE: Final = "집단이 커서 유전자는 {k}세대마다 남깁니다. 가까운 기록: {gen}세대"
DASH_FOCUS_CURSOR: Final = "보는 세대"

# 유전자 지도 (app/lab/watch_tiles.py, app/lab/history.py)
GMAP_EMPTY: Final = "학습을 시작하면 세대마다 집단이 이 지도 위의 점으로 나타납니다. 가까운 점일수록 비슷한 개체입니다."
GMAP_AXIS_X: Final = "가로: 가장 크게 달라지는 방향 ({pct:.0f}%)"
GMAP_AXIS_Y: Final = "세로: 그다음 방향 ({pct:.0f}%)"
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
GMAP_POINT_TIP_OLD: Final = "{gen}세대 대표 · 점수 {score} (이 세대는 집단 기록이 없어 대표만 보입니다)"
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
BRAIN_TITLE_REP: Final = "{gen}세대 대표(그때 챔피언 없음) · 멈춘 그림"
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
RACE_SAVE_GIF: Final = "내보내기…"                    # v3.1(설계 6절, D48): 원래 「경주 영상 저장…」
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
RACE_NO_GENERATIONS: Final = "아직 끝낸 세대가 없습니다"
RACE_FAILED: Final = "경주를 준비하지 못했습니다: {reason}"
RACE_HUD: Final = "코스 {n} · {t:.1f}초 · {speed:g}배"
VIDEO_RACE_TITLE: Final = "나란히 경주 · 코스 {n}"

# ---------------------------------------------------------------------------
# v1.2 검토 뒤 고침 (관찰 기록 알림, 세대 넘겨 보기 단추, 기록 없는 세대, 가로축 도움말, 경주 칸 이름, 영상 메모리)
TRAIN_OBSERVE_FAILED_ONWARD: Final = ("그래프용 관찰 기록을 만들지 못했습니다. 학습은 그대로 이어지지만 이 세대부터 새 그래프에 "
                                      "빠지는 세대가 있을 수 있습니다: {reason}")
TRAIN_OBSERVE_FRAME_FAILED: Final = ("궤적 밀도 그래프에 쓸 세계 범위를 만들지 못했습니다. 학습과 다른 그래프는 그대로입니다: "
                                     "{reason}")
LAB_FLIP_STOP_V2: Final = "넘겨 보기 멈춤"
DASH_NO_OBS_NOTE: Final = "이 세대는 집단 기록이 없어 세대 대표만 보입니다."
DASH_AXIS_TOOLTIP_V12: Final = "점수 추이, 탐색 폭, 종료 원인, 처리 성능, 몸 수치 추이 그래프의 가로축을 바꿉니다."
RACE_LANE_DUP: Final = "{name} ({tag})"
RACE_LANE_BRANCH_TAG: Final = "{gen}세대에서 분기"
VIDEO_NO_MEMORY: Final = "메모리가 모자라 영상을 저장하지 못했습니다."

# ---------------------------------------------------------------------------
# v2.0 움직이는 세계와 공기: 점프 게임, 글라이더 (0단계: 설계 6절 v2.0 전체 목록)
# 에디터 칸 이름·도움말·고르기 값은 위 EDP_FIELD_NAMES·EDP_FIELD_TIPS·EDP_VALUE_NAMES에, 검사 심각도 「안내」는
# EDP_ISSUES_SEVERITY에 더했고, PLAY_HELP는 제자리에서 {pause} 자리로 바꿨다(PLAY_PAUSE_KEYS로 채움).
# 설계의 PLAY_BUILD_FAILED(몸 조립 실패)는 이미 있는 PLAY_BUILD_FAILED(시드로 환경 조립 실패)와 이름이 겹쳐
# PLAY_BODY_BUILD_FAILED로 둔다. 설계 6절 목록에 없던 것(구현하며 더함)은 줄 끝에 「# 6절 밖」.
# ---------------------------------------------------------------------------
# v2 환경 파일 검사 (spec/compiled.py)
SPEC_KINEMATIC_AGENT: Final = "에이전트 물체 '{id}'는 정해진 움직임이 될 수 없습니다. 세계 물체만 스스로 움직일 수 있습니다."
SPEC_MOTION_NOT_KINEMATIC: Final = "물체 '{id}'에 움직임이 있지만 움직임 종류가 '정해진 움직임'이 아닙니다."
SPEC_KINEMATIC_NO_MOTION: Final = "물체 '{id}'는 정해진 움직임인데 움직임이 없어 고정 물체와 같습니다."
SPEC_MOTION_STILL: Final = "물체 '{id}'의 길 길이나 속력이 0이라 움직이지 않습니다."
SPEC_MOTION_PERIOD: Final = "물체 '{id}'의 흔들기 주기는 0보다 커야 합니다."
SPEC_KEY_SPACE_JUMP_ONLY: Final = "Space 키는 점프 추진기에만 쓸 수 있습니다 ('{id}')."
SPEC_AERO_NO_EDGES: Final = "공기역학 '{id}': 물체 '{body}'에 공기를 받을 모양이 없습니다(원과 선은 받지 않습니다)."
SPEC_AERO_RANGE: Final = "공기역학 '{id}'의 {name} 값이 범위 밖입니다."
SPEC_WING_SIZE: Final = "물체 '{id}'의 날개 모양은 시위와 두께가 0보다 커야 합니다."
SPEC_SHAPE_DENSITY_STATIC: Final = "물체 '{id}'는 움직이지 않는 물체라 모양별 밀도를 쓰지 않습니다."
SPEC_OBSTACLES_GAP: Final = "장애물 줄 '{id}'의 간격(초)은 0보다 커야 합니다."
SPEC_OBSTACLES_START: Final = "장애물 줄 '{id}'의 첫 장애물이 에이전트 시작 위치에 너무 가깝습니다(2m 이상 떨어뜨리세요)."
SPEC_BRAINLESS: Final = "출력이 없어 뇌 없이 몸만 진화합니다."
SPEC_REGION_DENSITY: Final = "영역 '{id}'는 물체가 아니라서 모양 밀도를 쓰지 않습니다."  # 6절 밖

# v2 런타임·판정 (world/physics2d, engine/evaluate.py, engine/commands.py, cli.py)
RT_BUILD_FAILED_END: Final = "조립 실패"
JUDGE_RESULT_NO_SUCCESS: Final = ("판정({gen}세대 챔피언, 판정 시드 {n}개): 평균 점수 {mean:.2f} (이 환경에는 성공 판정이 없습니다)\n"
                                  "종료 원인: {ends}")
CLI_JUDGE_PROGRESS: Final = "진행 거리({block}) 중앙값 {median:.1f}m, 하위 10% {p10:.1f}m"
BENCH_V2_PARTS: Final = "v2 부품: 움직이는 물체 {movers}개, 공기역학 변 {edges}개, 한 스텝 상한 발동 {clamped}스텝"
NO_SUCCESS_CELL: Final = "—"
TRAIN_INITIAL_BODY_FAILED: Final = "처음 몸 수치로는 몸을 만들 수 없습니다. {reason}"  # 6절 밖

# v2 에디터 도구·캔버스 (app/editor/tools.py, palette.py, canvas.py, view.py, forms.py)
EDC_GROUP_MOTION: Final = "움직임과 공기"
EDC_TOOL_PATH: Final = "길 그리기"
EDC_TOOL_OBSTACLES: Final = "장애물 줄"
EDC_TOOL_JUMP: Final = "점프 추진기"
EDC_TOOL_WING: Final = "날개"
EDC_TOOL_AERO: Final = "공기역학"
EDC_HINT_PATH: Final = "세계 물체를 누른 뒤 지날 점을 차례로 누릅니다. 두 번 누르거나 Enter로 끝, Esc로 취소."
EDC_HINT_OBSTACLES: Final = "다가오는 장애물 줄을 더합니다. 간격·크기·빨라지는 정도는 생성기 탭에서 고칩니다."
EDC_HINT_JUMP: Final = "에이전트 물체를 눌러 점프 추진기를 달고, 끌어서 뛰는 방향을 정합니다."
EDC_HINT_WING: Final = "에이전트 물체 위에서 앞전부터 뒷전까지 끌어 날개 모양을 그립니다."
EDC_HINT_AERO: Final = "에이전트 물체를 누르면 그 물체의 테두리가 공기를 받습니다(양력과 항력)."
EDC_PATH_WORLD_ONLY: Final = "길은 세계 물체에만 그릴 수 있습니다."
EDC_PATH_TOO_MANY: Final = "길의 점은 256개까지입니다."
EDC_ACTUATOR_AGENT_ONLY: Final = "이 부품은 에이전트의 움직이는 물체에만 달 수 있습니다."
EDC_UNDO_MAKE_MOVING: Final = "움직이는 물체로 바꾸기"
EDC_UNDO_ADD_WING: Final = "날개 모양 더하기"
EDC_OBSTACLES_MORE: Final = "→ 장애물 {count}개"
EDP_JUMP_HEIGHT: Final = "점프 높이 약 {h:.1f}m (중력 {g:.1f} 기준, 어림)"
EDP_WING_AREA: Final = "날개 면적 {a:.2f}m² (시위 {c:.2f}m × 폭 {s:.2f}m)"
EDP_PATH_INFO: Final = "길 길이 {length:.1f}m, 한 바퀴 {period:.1f}초"
EDP_PATH_LENGTH: Final = "길 길이 {length:.1f}m"
EDP_PATH_DRAW: Final = "캔버스에서 길 그리기"
EDP_BRAIN_NONE: Final = "뇌가 없는 환경입니다(출력이 없음). 몸 수치만 진화합니다."
EDP_HIDDEN_NONE: Final = "은닉층 없음"
ED_PREVIEW_TIME: Final = "미리보기 시각"
ED_PREVIEW_PLAY: Final = "움직임 미리보기"
ED_PREVIEW_TIME_TIP: Final = "정해진 움직임 물체가 이 시각에 어디 있는지 봅니다(학습에는 영향 없음). 고치면 0초로 돌아갑니다."
ED_PREVIEW_TRIAL: Final = "시험 {n}"  # 6절 밖: 「미리보기 시험 바꾸기」 단추(설계 5.1), 누를 때마다 다음 시험
ED_PREVIEW_TRIAL_TIP: Final = "누를 때마다 다른 시험의 랜덤 값(장애물 줄 등)으로 미리보기를 바꿉니다. 저장과 학습에는 영향이 없습니다."
ED_VIEW_SAVE: Final = "지금 보기를 기본 보기로"
ED_VIEW_CLEAR: Final = "기본 보기 지우기"
ED_VIEW_SAVED: Final = "기본 보기를 저장했습니다. 직접 조종과 실험실도 이 보기로 시작합니다."
ED_BUILTIN_JUMP: Final = "점프 게임"
ED_BUILTIN_GLIDER: Final = "글라이더"
ED_VIEW_MENU: Final = "보기"  # 6절 밖: 보기 메뉴(도구 막대의 펼침 단추) 제목
EDC_LABEL_JUMP: Final = "점프"  # 6절 밖: 캔버스의 점프 추진기 배지
EDC_LABEL_OBSTACLE_SEED: Final = "장애물 시드"  # 6절 밖: 장애물 줄 도구가 만드는 랜덤 값 oseed의 이름

# v2 직접 조종 (app/play/window.py)
PLAY_PAUSE_KEYS: Final = {"both": "Space·Esc", "esc": "Esc"}
PLAY_JUMP_READY: Final = "점프 준비됨"
PLAY_JUMP_AIR: Final = "공중"
PLAY_WATCH_ONLY: Final = "조종할 것이 없는 환경입니다. 지켜보세요."
PLAY_BODY_BUILD_FAILED: Final = "이 몸은 만들 수 없습니다(모양이 올바르지 않음)."

# v2 실험실·대시보드 (app/lab/view.py, watch_tiles.py, brain_view.py)
LAB_BUILD_FAILED_N: Final = "이 세대에 몸을 만들 수 없는 개체가 {n}개 있습니다(가장 낮은 점수로 셈)."
DASH_EMPTY_NO_BRAIN: Final = "이 환경은 뇌가 없습니다. 몸만 진화합니다."
DASH_EMPTY_NO_OUTPUTS: Final = "구동기 출력이 없습니다."
DASH_END_BUILD_FAILED: Final = "조립 실패"

# ---- v2.0 검토 결과 고침 묶음 (검사·에디터·비교·직접 조종) ----
SPEC_BRAINLESS_HIDDEN: Final = "출력이 없어 은닉층이 쓰이지 않습니다. 뇌 탭에서 은닉층을 비우세요."
SPEC_AERO_TWICE: Final = "물체 '{body}'에 공기역학이 이미 있습니다. 하나만 두세요(공기힘이 겹쳐 커집니다)."
CMP_NO_SUCCESS_RATE: Final = "성공 판정이 없는 환경의 실행은 성공률이 없어 그리지 않았습니다."
EDP_WING_CHORD: Final = "시위 {c:.2f}m (공기역학을 붙이면 양력이 생깁니다)"
PLAY_SAVE_WATCH_ONLY: Final = "조종하지 않는 환경이라 입력 기록을 저장하지 않습니다."

# ---------------------------------------------------------------------------
# v2.1 몸 구조 진화: 바퀴 고르는 언덕차, 구조 뒤집기, 에디터 「구조」 탭, 「몸 구조」 타일 (0'단계: 설계 6절 v2.1 전체 목록)
# 에디터 칸 이름·도움말 "exists"는 위 EDP_FIELD_NAMES·EDP_FIELD_TIPS에, 타일 제목 "structure"는 DASH_TILE_TITLES에,
# 타일 풍선 도움말(설계의 STRUCT_TIP)은 DASH_TILE_HELP["structure"]에 더했다.
# ---------------------------------------------------------------------------
# v2.1 환경 파일 검사 (spec/compiled.py)
SPEC_EXISTS_WORLD: Final = "'있음'은 에이전트 물체에만 쓸 수 있습니다 ('{id}')."
SPEC_EXISTS_ROOT: Final = "몸통 '{id}'(첫 에이전트 물체)는 없앨 수 없습니다. '있음'을 비우세요."
SPEC_EXISTS_REF: Final = "'{body}'는 진화로 없어질 수 있어 {where}에서 쓸 수 없습니다(접촉만 쓸 수 있습니다)."

# v2.1 명령줄 (cli.py)
CLI_HELP_STRUCTURE_FLIP: Final = "유전 알고리즘·담금질이 '있음' 유전자를 뒤집을 확률(구조 진화 환경의 새 실행 기본 0.05)"

# v2.1 에디터 (app/editor/view.py, structure.py, tabs.py, forms.py, tools.py)
ED_BUILTIN_WHEELS: Final = "바퀴 고르는 언덕차"
ED_TAB_STRUCTURE: Final = "구조"
ED_STRUCT_EMPTY: Final = "진화로 생기고 없어지는 부품이 없습니다. 물체의 '있음' 옆 진화를 켜거나 '배열로 늘리기'를 쓰세요."
ED_STRUCT_COLUMNS: Final = ("물체", "있음", "진화", "처음 상태", "함께 사라지는 부품")
ED_STRUCT_ON: Final = "있음"
ED_STRUCT_OFF: Final = "없음"
ED_STRUCT_ROOT_LOCKED: Final = "몸통은 없앨 수 없습니다."
ED_STRUCT_SHOW_ABSENT: Final = "없는 부품도 보기"
ED_REPLICATE: Final = "배열로 늘리기…"
ED_REPLICATE_TITLE: Final = "배열로 늘리기"
ED_REPLICATE_COUNT: Final = "개수"
ED_REPLICATE_LAYOUT: Final = {"line": "직선", "circle": "원", "mirror": "좌우 거울"}
ED_REPLICATE_EVOLVE: Final = "있음/없음도 진화"
ED_REPLICATE_ONE_COUNT: Final = "개수를 몸 수치 하나로 묶기"
ED_REPLICATE_DONE: Final = "'{id}'를 {n}개로 늘렸습니다."
ED_REPLICATE_NEED_AGENT: Final = "에이전트 물체를 하나 고르세요."
ED_PARAM_ON_LABEL: Final = "{label} 있음"
ED_PARAM_COUNT_LABEL: Final = "{label} 개수"

# v2.1 실험실·대시보드 (app/lab/structure_tile.py, view.py)
STRUCT_EMPTY: Final = "이 환경에는 진화로 생기고 없어지는 부품이 없습니다. 에디터에서 부품의 '있음' 진화를 켜 보세요."
STRUCT_OTHER: Final = "기타"
STRUCT_COUNT: Final = "{name} {n}개"
STRUCT_COUNT_WHICH: Final = "{name} {n}개: {which}"
STRUCT_NONE: Final = "없음"
STRUCT_CHAMPION: Final = "{gen}세대 챔피언의 몸"
INSPECT_ABSENT: Final = "(없음)"
LAB_GA_NO_STRUCTURE: Final = "이 실행은 구조 뒤집기가 꺼져 있어 유전 알고리즘·담금질이 몸 구조를 거의 바꾸지 못합니다."

# ---------------------------------------------------------------------------
# v2.1 A' 묶음 (spec/compiled.py): SPEC_EXISTS_REF의 {where}에 넣는 자리 이름
# ---------------------------------------------------------------------------
SPEC_EXISTS_WHERE_EVENT: Final = "사건 '{id}'"
SPEC_EXISTS_WHERE_BLOCK: Final = "점수 블록 '{id}'"
SPEC_EXISTS_WHERE_FORMULA: Final = "점수 수식"
SPEC_EXISTS_WHERE_CHECKPOINTS: Final = "체크포인트 '{id}'"
SPEC_EXISTS_WHERE_TARGET: Final = "목표 센서 '{id}'"

# v2.1 L묶음(학습 방법 구조 뒤집기: learners/structure.py·ga.py·sa.py, engine/commands.py 이어하기 검사). 6절 밖
LEARNER_ERR_STRUCTURE_FLIP: Final = "구조 뒤집기 확률은 0 이상 1 이하의 숫자여야 합니다 (지금 {value})."
LEARNER_ERR_STRUCTURE_GENES: Final = "'있음' 유전자 번호는 몸 유전자 수({n})보다 작고 서로 다른 0 이상의 정수여야 합니다 (지금 {value})."
TRAIN_RESUME_STRUCTURE_FIXED: Final = "이어 할 때는 구조 뒤집기 확률(--structure-flip)을 바꿀 수 없습니다. 새 실행으로 시작하세요."

# ---------------------------------------------------------------------------
# v2.1 B'묶음 (에디터 구조 탭·배열로 늘리기: app/editor/structure.py, tools.py). 6절 밖
# ---------------------------------------------------------------------------
ED_REPLICATE_ROOT: Final = "몸통(첫 에이전트 물체)은 배열로 늘릴 수 없습니다. 몸통에 붙은 부품을 고르세요."
ED_REPLICATE_HINT: Final = "'{name}' 물체와 거기 붙은 관절·모터·구동기·센서(관절로 매달린 부품 포함)를 함께 복사합니다."
ED_REPLICATE_LAYOUT_LABEL: Final = "놓는 방식"
ED_REPLICATE_LAYOUT_TIPS: Final = {
    "line": "원본 자리(시작점)부터 끝점까지 같은 간격으로 놓습니다.",
    "circle": "가운데를 중심으로 같은 각도 간격으로 돌려 놓습니다.",
    "mirror": "세로 축에 비춰 뒤집은 사본 하나를 만듭니다.",
}
ED_REPLICATE_START: Final = "시작점 (x, y)"
ED_REPLICATE_END: Final = "끝점 (x, y)"
ED_REPLICATE_CENTER: Final = "가운데 (x, y)"
ED_REPLICATE_RADIUS: Final = "반지름"
ED_REPLICATE_AXIS: Final = "거울 축 x"
ED_REPLICATE_EVOLVE_TIP: Final = "부품마다 '있음' 몸 수치를 만들어, 생기고 없어지는 것까지 진화합니다. 처음에는 원본만 있습니다."
ED_REPLICATE_ONE_COUNT_TIP: Final = "몸 수치 하나(개수)가 몇 개를 달지 정합니다. k번째 부품은 개수가 k 이상일 때 있습니다."
ED_REPLICATE_OK: Final = "늘리기"
ED_REPLICATE_CANCEL: Final = "취소"
ED_STRUCT_HINT: Final = "진화로 생기고 없어질 수 있는 물체입니다. 줄을 고르면 함께 사라지는 부품이 캔버스에 점선으로 보입니다."
ED_STRUCT_UNKNOWN: Final = "—"
ED_STRUCT_PARTS_TIP: Final = "이 물체가 없을 때 함께 없어지는 부품 수: 관절·모터·구동기·센서와, 관절로만 매달려 있던 물체."
ED_STRUCT_SHOW_ABSENT_TIP: Final = "처음 몸에 없는 부품을 옅은 점선으로 그립니다. 끄면 처음 몸 그대로 보입니다."

# v2.1 C'묶음(「몸 구조」 타일 app/lab/structure_tile.py). 6절 밖
STRUCT_AXIS: Final = "개체 비율(%)"
STRUCT_WAIT: Final = "학습을 시작하면 세대마다 개체들의 몸 구조 비율을 그립니다."
STRUCT_REPS_ONLY: Final = "집단 기록이 없는 세대는 세대 대표의 몸 구조 하나만 셉니다."

# ---------------------------------------------------------------------------
# v2.1 검토 반영 묶음(독립 검토·검증에서 확인된 결함 고침: spec/compiled.py, engine/trainer.py, app/play/window.py,
# app/editor/tools.py·view.py·structure.py). 6절 밖
# ---------------------------------------------------------------------------
SPEC_EXISTS_FLIP_RANGE: Final = ("물체 '{id}'의 있음 몸 수치 '{param}'은 범위의 가운데가 0.5가 아니라서, 유전 알고리즘·"
                                 "담금질의 구조 뒤집기가 이 부품을 켜고 끄지 않습니다. 범위를 0..1로 하면 뒤집습니다.")
LAB_STRUCTURE_RANGE: Final = "있음 몸 수치의 범위 가운데가 0.5가 아니라서 구조 뒤집기가 켜고 끄지 않는 부품: {names}. 범위를 0..1로 하세요."
PLAY_STRUCTURE_NOTE: Final = "처음 몸에 없는 부품: {names}. 그 부품의 키는 듣지 않습니다. 처음 몸은 서툴 수 있고, 학습하면 몸 구조가 바뀝니다."
ED_REPLICATE_SHARED: Final = "사본의 수식은 원본의 몸 수치({params})를 그대로 씁니다. 사본마다 다르게 하려면 수식을 고치세요."
ED_STRUCT_EVOLVE_BY_PARAM_TIP: Final = "있음이 진화하는 몸 수치 '{id}'로 정해집니다. 진화를 끄려면 매개변수 탭에서 그 몸 수치의 진화를 끄세요."

# ---------------------------------------------------------------------------
# v3.0 처음 만나기 (0단계 묶음: v3 설계 6절의 v3.0 줄, 명세 부록 A65~). 「# v3.1」 줄(설치 프로그램, 내보내기, 폰 재생 파일,
# 명령줄 내보내기, VIDEO_SAVE·VIDEO_SAVE_TITLE·RACE_SAVE_GIF의 새 값)은 v3.1 0단계에서 더한다.
# - LAB_START는 위 제자리에서 「▶ 학습 시작」으로 바꿨다(설계 D17).
# - 설계의 새 ABOUT_TEXT(데이터·설정 폴더 자리 {data_dir}·{state_dir})는 ABOUT_TEXT_V3다. 통합(I 묶음)에서 main.py의 정보 창을
#   이것으로 옮기고 옛 ABOUT_TEXT(데이터 폴더를 「문서\진화실험실」 글자로 박은 것)를 지웠다.
# - 도움말 쪽은 v2·v2.1 뒤 화면에 맞췄다: 대시보드 타일 12개(「몸 구조」), 에디터 아래 탭 10개(「구조」), 내장 예제 6개(바퀴 고르는
#   언덕차), 낱말 21개(「몸 구조」). 「설치와 제거」 쪽과 「기록과 내보내기」 쪽 「영상」 절은 설계의 v3.0판(zip, GIF)이었고
#   v3.1 0'단계에서 v3.1 글(설치 프로그램, MP4·GIF·폰 재생 파일)로 제자리에서 바꿨다(맨 끝 v3.1 머리말 참고).
# ---------------------------------------------------------------------------
ABOUT_TEXT_V3: Final = (
    "{name} {version}\n\n"
    "직접 만든 2D 실험 환경에서 진화 전략(CMA-ES, 유전 알고리즘, 담금질)이 해법을 찾아가는 과정을 보는 프로그램입니다.\n"
    "사용자 데이터: {data_dir} (environments, runs, records)\n"
    "설정과 오류 기록: {state_dir}\n\n"
    "오픈 소스 라이선스\n"
    "--------------------\n{notices}"
)

# ---- 처음 실행 안내 (app/help/tour.py) ----
ONB_WELCOME_TITLE: Final = "진화 실험실에 오신 것을 환영합니다"
ONB_WELCOME_BODY: Final = "자동차 {pop}대가 스스로 언덕을 넘는 법을 배우는 모습을 함께 봅시다. 단추 두 번이면 시작합니다."
ONB_WELCOME_START: Final = "따라 하기 시작"
ONB_WELCOME_LATER: Final = "나중에 볼게요"
ONB_LATER_HINT: Final = "따라 하기는 「{menu} → {item}」에서 언제든 다시 볼 수 있습니다."
ONB_RESUME_TITLE: Final = "따라 하기를 마저 할까요?"
ONB_RESUME_BODY: Final = "지난번에 끝까지 보지 않았습니다. 이어서 보면 학습 시작 단계부터 다시 합니다."
ONB_RESUME_CONTINUE: Final = "이어서 보기"
ONB_RESUME_RESTART: Final = "처음부터"
ONB_RESUME_STOP: Final = "그만 보기"
ONB_STEP_PRESS_START: Final = "「{start}」을 누르세요. 자동차 {pop}대가 함께 연습을 시작합니다."
ONB_STEP_WAIT_PROCESS: Final = "학습을 준비하는 중입니다. 계산 도우미 {workers}개를 띄우고 있습니다…"
ONB_STEP_WAIT_GEN1: Final = "1세대를 계산하는 중입니다: 자동차 {pop}대 × 연습 {trials}번 ({seconds}초)"
ONB_STEP_WAIT_SLOW: Final = "처음 실행은 윈도우 보안 검사 때문에 느릴 수 있습니다. 그대로 기다려 주세요."
ONB_STEP_WAIT_STUCK: Final = "{seconds}초가 지나도 1세대가 오지 않습니다. 더 기다리거나 문제 해결 도움말을 보세요."
ONB_STUCK_WAIT: Final = "더 기다리기"
ONB_STUCK_HELP: Final = "문제 해결 보기"
ONB_STEP_WATCH: Final = ("처음 보이는 것이 1세대입니다. 아직 배운 것이 없어 대부분 멈추거나 뒤집힙니다. "
                         "학습은 재생과 상관없이 계속됩니다. 화면은 곧 새 세대로 넘어갑니다. "
                         "재생을 빨리 보려면 「{speed}」을 4배로 바꿔 보세요.")  # v6.0: 관전 기본 「학습 먼저」(15초 안에 새 세대)
ONB_SPEED_SET: Final = "4배로 바꾸기"
ONB_STEP_PROGRESS: Final = ("세대마다 멀리 간 차를 닮은 새 차 {pop}대가 다시 달립니다. 「{trend}」 그래프의 평균 선을 보세요. "
                            "{goal}세대가 되면 다음으로 넘어갑니다.")
ONB_PROGRESS_LINE: Final = "{gen}/{goal}세대 · 1세대 평균 {first:.0f}점 → 지금 평균 {now:.0f}점"
ONB_PROGRESS_FLAT: Final = "아직 평균 점수가 크게 오르지 않았습니다. 학습 빠르기는 시드와 PC에 따라 다릅니다."
ONB_STEP_VERSUS: Final = "「{versus}」을 누르면 1세대에서 가장 잘한 차와 지금 챔피언이 같은 언덕을 나란히 달립니다."
ONB_STEP_DONE: Final = ("운전하는 법을 가르친 사람은 없습니다. 멀리 간 차를 닮은 차를 세대마다 고른 결과입니다. "
                        "궁금한 곳에 마우스를 올리고 F1을 누르면 그 부분의 도움말이 열립니다.")
ONB_NEXT: Final = "다음"
ONB_SKIP: Final = "건너뛰기"
ONB_SKIP_STEP: Final = "이 단계 건너뛰기"
ONB_FINISH: Final = "끝내기"
ONB_CLOSE: Final = "닫기"
ONB_PROGRESS: Final = "{i}/{n}"
ONB_BACK_TO_LAB: Final = "실험실 탭으로 돌아가기"
ONB_FAILED: Final = "학습을 시작하지 못했습니다: {reason}"
ONB_BUSY: Final = "지금 실험실에서 학습 중이라 따라 하기를 시작할 수 없습니다. 「{end}」를 누른 뒤 다시 하세요."
ONB_NO_EXAMPLE: Final = "내장 예제 「{name}」 파일을 찾을 수 없습니다. 설치가 망가졌을 수 있으니 다시 설치하세요."
ONB_EXISTING_USER_HINT: Final = "새 기능: 화면마다 도움말(F1)과 처음 실행 안내가 생겼습니다. 「{menu}」 메뉴에서 보세요."

# ---- 도움말 메뉴·모서리 단추 (app/help/setup.py, app/main.py) ----
MENU_HELP_SCREEN: Final = "이 화면 도움말"
MENU_HELP_BROWSE: Final = "도움말 찾아보기…"
MENU_HELP_WHATS_THIS: Final = "이건 뭐예요?"
MENU_HELP_TOUR: Final = "처음 실행 안내 다시 보기"
MENU_REPORT: Final = "오류 보고 묶음 만들기…"
MENU_OPEN_LOGS: Final = "오류 기록 폴더 열기"
HELP_CORNER_BUTTON: Final = "? 도움말"
HELP_CORNER_TIP: Final = "지금 화면의 도움말을 엽니다 (F1)."

# ---- 도움말 창 (app/help/window.py, content.py) ----
HELPW_TITLE: Final = "진화 실험실 도움말"
HELPW_SEARCH_PLACEHOLDER: Final = "도움말에서 찾기 (예: 시드, 배속)"
HELPW_BACK: Final = "◀ 뒤로"
HELPW_FORWARD: Final = "앞으로 ▶"
HELPW_HOME: Final = "처음"
HELPW_FONT_BIGGER: Final = "글자 크게"
HELPW_FONT_SMALLER: Final = "글자 작게"
HELPW_RESULTS: Final = "찾은 곳 {n}군데"
HELPW_RESULT_ITEM: Final = "{page} · {count}번 · …{snippet}…"
HELPW_NO_RESULTS: Final = "「{q}」에 맞는 곳이 없습니다. 낱말을 줄이거나 「{glossary}」 쪽을 보세요."
HELPW_PAGE_MISSING: Final = "도움말 쪽을 찾을 수 없습니다: {page}"
HELPW_SHOW_MISSING: Final = "이 단추는 지금 화면에 없습니다: {what}"
HELPW_TOUR_BUSY: Final = "학습 중에는 처음 실행 안내를 다시 볼 수 없습니다. 학습을 끝낸 뒤 다시 하세요."
HELPW_LINK_BLOCKED: Final = "이 도움말은 인터넷 주소를 열지 않습니다: {url}"
WHATS_THIS_HINT: Final = "알고 싶은 곳을 누르세요 (그만하려면 Esc)."
WHATS_THIS_NONE: Final = "이 부분에는 따로 설명이 없습니다. 「{page}」 도움말을 여세요."
WHATS_THIS_MORE: Final = "도움말에서 더 보기"
HELP_GLOSSARY_TERMS: Final = ("세대", "개체", "집단", "유전자", "몸 수치", "몸 구조", "뇌", "뉴런", "시험", "시드", "세대 대표",
                              "챔피언", "검증", "관전", "고속", "CMA-ES", "유전 알고리즘", "담금질", "체크포인트", "관찰 기록",
                              "재현 묶음", "누적 시험 횟수")  # v6.1: 「뉴런」

# 도움말 쪽: id → (제목, Markdown). 「…」 안에는 {{상수}}·{{사전.키}} 자리만 쓴다(테스트). 쪽 순서 = 목록 순서.
HELP_PAGES: Final = {
"start": ("처음 시작", """# 처음 시작

진화 실험실은 직접 만든 2D 실험 환경에서 **학습이 해법을 찾아가는 과정**을 보는 프로그램입니다.
몸과 작은 신경망(뇌)을 가진 개체 여러 개를 시험하고, 잘한 개체를 닮은 새 개체를 만드는 일을 세대마다 되풀이합니다.

## 3분 따라 하기

[처음 실행 안내 시작](evolab://tour) — 언덕 넘는 자동차가 스스로 언덕을 넘는 법을 익히는 모습을 봅니다.

## 화면 넷

- **{{TAB_EDITOR}}**: 물체, 센서, 추진기, 점수 기준을 놓아 실험 환경을 만듭니다. [에디터 도움말](evolab://help/editor)
- **{{TAB_LAB}}**: 고른 환경으로 학습을 돌리고 지켜봅니다. [실험실 도움말](evolab://help/lab)
- **{{TAB_COMPARE}}**: 여러 실행을 한 그래프에 겹치고 나란히 달리게 합니다. [비교 도움말](evolab://help/compare)
- **{{TAB_RECORDS}}**: 지난 실행을 열고, 이어 하고, 내보냅니다. [기록 도움말](evolab://help/records)

## 다음에 해 볼 것

- 실험실에서 다른 내장 예제를 골라 학습해 보기 ([내장 예제](evolab://help/examples))
- 「{{MENU_PLAY}}」 메뉴로 직접 몰아 보고 AI와 비교하기
- 에디터의 「{{ED_FROM_BUILTIN}}」 단추로 예제를 열어 한 군데만 고쳐 학습해 보기

어느 화면에서든 **F1**을 누르면 마우스 아래 부분의 도움말이 열립니다. 모르는 말은 [낱말 풀이](evolab://help/words)에 있습니다.
"""),
"lab": ("실험실", """# 실험실

학습을 돌리고 지켜보는 화면입니다. 오른쪽 패널 맨 위의 「{{LAB_START}}」 단추를 누르면 고른 환경에서 개체들이 연습을 시작하고,
세대마다 잘한 개체를 바탕으로 다음 세대를 만듭니다. [단추 보여 주기](evolab://show/lab.start)

## 진행

- 「{{LAB_START}}」 단추: 지금 설정으로 새 실행을 만듭니다. 기록은 「{{TAB_RECORDS}}」 탭에 자동으로 남습니다.
- 「{{LAB_PAUSE}}」 단추: 이번 세대를 마치고 멈춥니다. 「{{LAB_CONTINUE}}」 단추로 계속하고, 「{{LAB_STEP}}」 단추는 한 세대만 더 합니다.
- 「{{LAB_END}}」 단추: 체크포인트를 남기고 학습 프로그램을 닫습니다. 나중에 기록 화면에서 이어 할 수 있습니다.

## 관전과 고속

- 「{{LAB_SPECTATE}}」: 한 세대의 개체 전체를 겹쳐 보여 줍니다. 「{{LAB_PACE}}」 칸에서 보는 방식을 고릅니다.
  - 「{{LAB_PACE_SKIP}}」: 학습은 쉬지 않고, 재생이 끝날 때마다 그때까지 끝난 가장 새 세대를 보여 줍니다.
    건너뛴 세대도 기록과 그래프에는 모두 남고, 아래 타임라인으로 언제든 다시 볼 수 있습니다.
  - 「{{LAB_PACE_EVERY}}」: 모든 세대를 차례로 보여 주고, 재생이 끝날 때까지 다음 세대를 기다립니다.
    「{{LAB_SPEED}}」 1배에서는 학습이 몇 배 느려지니 4배나 8배로 올리세요. [배속 보여 주기](evolab://show/lab.speed)
- 「{{LAB_FAST}}」: 집단 대신 정해 둔 세대마다 챔피언만 다시 보여 줍니다. 학습은 「{{LAB_PACE_SKIP}}」 방식처럼 쉬지 않습니다.

## 환경 고르기

오른쪽 패널의 「{{LAB_ENV_GROUP}}」 묶음에서 내장 예제나 내 환경 파일을 고릅니다. 학습 중에는 바꿀 수 없습니다.

## 타임라인과 비교

- 세대 막대에서 세대를 고르고 「{{LAB_TIMELINE_PLAY}}」 단추를 누르면 그 세대의 대표를 다시 재생합니다.
- 「{{LAB_VERSUS}}」 단추: 1세대에서 가장 잘한 개체와 지금 챔피언을 같은 조건에서 겹쳐 봅니다.
- 「{{LAB_RACE_GENERATIONS}}」 단추: 여러 세대의 대표를 나란히 달리게 합니다.
- 「{{LAB_LIVE}}」 단추: 다시 보기를 그만두고 지금 학습 중인 세대로 돌아갑니다.
- 「{{LAB_FLIP}}」 단추: 세대를 하나씩 자동으로 넘기며 「{{DASH_PAGES.fun}}」 그래프가 바뀌는 모습을 보여 줍니다.

## 개체 들여다보기

재생 중인 개체를 누르면 그 개체의 센서 값과 신경망 출력을 볼 수 있습니다.
몸 구조가 진화하는 환경에서는 그 개체에 없는 부품에 단 센서의 값 대신 「{{INSPECT_ABSENT}}」 표시가 나옵니다.

## 학습 중 고치기

멈춘 뒤 「{{LAB_ENV_EDIT}}」 단추로 환경을 바꾸고 다시 보낼 수 있습니다. 센서나 추진기 수가 바뀌면 이어 할 수 없고 새로 시작해야 합니다.
정지 조건과 시험 횟수는 멈춘 뒤 「{{LAB_APPLY}}」 단추로 바꿉니다.

## 설정 칸

칸 위에 마우스를 올리면 설명이 나옵니다. 처음에는 기본값 그대로 두는 것이 좋습니다.
같은 설정이라도 실행 시드에 따라 빠르기가 다릅니다. 점수가 오래 그대로면 시드만 바꿔 새로 시작해 보세요.
「{{LAB_STABILITY_GROUP}}」 묶음의 자동 재시작은 선택 기능입니다. 시험해 보니 평균 성공률을 높이지 못해 기본으로 꺼 두었습니다.

## 자라는 뇌

「{{LAB_BRAIN}}」 줄에서 「{{LAB_BRAIN_GROW}}」 항목을 고르면 뇌가 은닉층마다 「{{LAB_GROW_START}}」 칸의 수만큼 뉴런으로 시작해서
「{{LAB_GROW_EVERY}}」 칸의 세대마다 두 배로 늘어납니다. 이 줄은 학습 방법이 CMA-ES일 때만 보입니다.
늘어날 때 새 뉴런의 출력 쪽 연결은 0에서 시작하므로 그때까지 배운 행동은 그대로이고, 그다음 세대부터 새 뉴런을 쓰는 법을 배웁니다.
「{{DASH_TILE_TITLES.brain}}」 타일에서 옅은 점선 원이 채워지는 모습을, 「{{DASH_TILE_TITLES.brainsize}}」 타일에서 뉴런과 연결 수가 계단처럼
느는 모습을 봅니다. 에디터 「{{EDP_TAB_BRAIN}}」 탭의 은닉층 크기가 다 자란 크기입니다.
내장 예제로 시험해 보니 같은 학습 시험 수에서 「{{LAB_BRAIN_FIXED}}」 방식과 비슷하거나 조금 느렸습니다. 두 가지를 비교 화면에서 나란히
돌려 보면 그 환경에 맞는 쪽을 알 수 있습니다.
자라는 뇌 실행은 v6.1 이상에서만 이어 하고 재현할 수 있습니다. 그보다 옛 프로그램은 이 실행을 열어 보고 재생하고 판정할 수는 있지만
이어 하기와 재현은 거절합니다. 옛 프로그램에서 「{{REC_CLONE}}」 단추를 누르면 자라지 않는 크기 고정 실행으로 시작되고, 「{{REC_RESUME}}」
단추로 예전 체크포인트를 고르면 이어 할 수 없는 분기 폴더만 남습니다(지워도 됩니다).
"""),
"dashboard": ("그래프 읽기", """# 그래프 읽기

실험실 아래 대시보드는 「{{DASH_PAGES.learning}}」 쪽과 「{{DASH_PAGES.fun}}」 쪽으로 나뉩니다. 타일은 켜고 끄고 순서를 바꿀 수 있습니다.
타일 위에서 F1을 누르면 그 타일 설명으로 옵니다.

## {{DASH_TILE_TITLES.trend}}

세대마다 최고·평균·중앙값·최저 점수입니다. **{{DASH_TREND_CHAMPION}}** 선은 학습에 쓰지 않은 검증 시험으로 다시 잰 점수라 덜 흔들립니다.
평균 선이 오르면 집단 전체가 나아지는 중입니다.

## {{DASH_TILE_TITLES.heatmap}}

가로는 세대, 세로는 점수입니다. 진할수록 그 점수대에 개체가 많습니다. 덩어리가 위로 올라가면 집단이 나아지는 중입니다.

## {{DASH_TILE_TITLES.diversity}}

학습 방법이 얼마나 넓게 찾는지입니다. 줄어들면 한곳으로 모이는 중입니다.

## {{DASH_TILE_TITLES.blocks}}

점수가 어느 항목(성공, 거리, 연료 등)에서 나오는지입니다. 원하지 않는 꼼수가 점수를 벌고 있는지 볼 때 씁니다.

## {{DASH_TILE_TITLES.ends}}

시험이 어떻게 끝났는지(성공, 충돌, 멈춤, 시간 끝)의 비율입니다. 몸을 만들 수 없던 개체의 몫은 「{{DASH_END_BUILD_FAILED}}」 색으로 칠합니다.

## {{DASH_TILE_TITLES.perf}}

세대마다 걸린 시간, 초당 시뮬레이션 스텝, 워커 사용률입니다. PC가 바쁘면 느려집니다.

## {{DASH_TILE_TITLES.genemap}}

개체 하나하나를 점으로 찍은 지도입니다. 집단이 어디로 움직였는지 보입니다.

## {{DASH_TILE_TITLES.brain}}

신경망 연결과 재생 중 뉴런 값입니다. 뇌가 없는 환경(글라이더 등)에서는 비어 있습니다.

## {{DASH_TILE_TITLES.brainsize}}

세대마다 은닉 뉴런 수(계단)와 연결 수(점선)를 그립니다. 세로 점선은 뇌가 자란 세대입니다. 「{{LAB_BRAIN_FIXED}}」 실행에서는 크기만 알려 줍니다.

## {{DASH_TILE_TITLES.density}}

집단이 자주 지나간 곳입니다.

## {{DASH_TILE_TITLES.actions}}

추진기와 모터를 얼마나 세게, 얼마나 자주 썼는지입니다.

## {{DASH_TILE_TITLES.body}}

진화하는 몸 수치(바퀴 크기, 다리 길이 등)가 세대마다 어떻게 바뀌었는지입니다.

## {{DASH_TILE_TITLES.structure}}

부품이 생기고 없어지는 환경(바퀴 고르는 언덕차 등)에서, 세대마다 개체들이 어떤 몸 구조를 가졌는지 비율로 쌓아 보여 줍니다.
아래 그림은 보는 세대 챔피언의 몸입니다. 그런 부품이 없는 환경에서는 비어 있습니다.

## 가로축

「{{DASH_AXIS_LABEL}}」 칸에서 세대, 누적 시험 횟수, 경과 시간을 고릅니다.
"""),
"editor": ("에디터", """# 에디터

실험 환경을 만드는 화면입니다. 만든 환경은 .evoenv 파일로 저장되고 실험실에서 학습합니다.

## 처음 만들 때

1. 「{{ED_FROM_BUILTIN}}」 단추로 비슷한 예제를 엽니다.
2. 한두 군데만 고칩니다(예: 착륙선의 중력, 언덕의 높이).
3. 「{{EDP_TAB_ISSUES}}」 탭에 오류가 없는지 봅니다.
4. 「{{ED_PLAY}}」 단추로 몰아 본 뒤 「{{ED_TO_LAB}}」 단추로 실험실에 보냅니다.

## 도구와 캔버스

왼쪽에서 도구를 고른 뒤 가운데 캔버스에 놓습니다. 끌어서 옮기고, 휠로 확대합니다. 되돌리기는 Ctrl+Z, 다시하기는 Ctrl+Y입니다.
「{{EDC_GROUP_MOTION}}」 묶음에는 스스로 움직이는 물체, 장애물 줄, 점프, 날개를 만드는 도구가 있습니다.
움직이는 물체는 캔버스 아래 「{{ED_PREVIEW_TIME}}」 칸에서 미리 움직여 봅니다.

## 속성과 진화 대상

오른쪽 속성 칸 위에 마우스를 올리면 설명이 나옵니다. 숫자 옆 진화 표시를 켜면 그 값이 몸 수치가 되어 학습 중에 바뀝니다.
숫자 칸에 =로 시작하는 수식을 쓸 수 있습니다(예: =p.leg_len / 2).

## {{EDP_TAB_WORLD}}

중력, 경계, 시간 간격 같은 세계 전체의 값입니다.

## {{EDP_TAB_EVENTS}}

무엇이 일어나면 시험이 끝나는지(착지, 충돌, 시간)를 정합니다. 위에 있는 조건이 먼저입니다.

## {{EDP_TAB_SCORING}}

점수는 점수 블록의 값 × 가중치의 합입니다. 학습은 이 점수를 높이는 쪽으로만 갑니다. 원하지 않는 꼼수가 보이면 점수를 고치세요.

## {{EDP_TAB_RANDOM}}

시험마다 달라지는 값(시작 위치, 바람 등)의 범위입니다. 범위가 넓을수록 학습은 어렵지만 처음 보는 조건에 강해집니다.

## {{EDP_TAB_GENERATORS}}

지형, 트랙, 언덕, 장애물 줄을 자동으로 만드는 도구입니다.

## {{EDP_TAB_EVAL}}

개체 하나를 몇 번 시험해 평균 낼지 정합니다.

## {{EDP_TAB_BRAIN}}

신경망의 크기(은닉층)와 활성화 함수입니다. 처음에는 그대로 두세요. 출력이 없는 환경은 뇌 없이 몸만 진화합니다.

## {{ED_TAB_STRUCTURE}}

진화로 생기고 없어질 수 있는 부품의 목록입니다. 줄을 고르면 그 부품이 없을 때 함께 사라지는 부품이 캔버스에 점선으로 보입니다.
「{{ED_REPLICATE}}」 단추로 부품 하나를 여러 개로 늘리고, 몇 개를 달지도 진화하게 할 수 있습니다.

## {{EDP_TAB_PARAMS}}

진화하는 몸 수치의 이름과 범위입니다. 부품이 있는지 없는지도 몸 수치로 정할 수 있습니다.

## {{EDP_TAB_ISSUES}}

파일의 문제와 경고입니다. 줄을 누르면 그 부품으로 갑니다.
"""),
"play": ("직접 조종", """# 직접 조종

학습 전에 키보드로 환경을 직접 몰아 보는 창입니다. 사람이 해 보면 무엇이 어려운지 알 수 있습니다.

## 키

조작 키는 창 옆에 나옵니다. **Esc**는 언제나 일시정지이고, 점프 게임은 **Space**로 뜁니다.
「{{PLAY_RESTART}}」 단추는 같은 조건으로, 「{{PLAY_NEW_SEED}}」 단추는 새 시작 조건으로 다시 합니다.
몸 구조가 진화하는 환경에서 처음 몸에 없는 부품의 키는 듣지 않습니다.

## 입력 기록

「{{PLAY_SAVE_RECORD}}」 단추로 방금 판의 키 입력을 저장하고, 「{{PLAY_LOAD_RECORD}}」 단추로 똑같이 다시 재생합니다.

## 내 최고 기록

저장한 기록의 점수는 실험실 점수 그래프에 「{{DASH_MY_BEST}}」 선으로 나옵니다. 조종할 것이 없는 환경(글라이더 등)은 지켜보기만 합니다.
"""),
"compare": ("비교와 반복 실행", """# 비교와 반복 실행

## 실행 고르기

목록에서 실행을 여러 개 고르면 한 그래프에 겹쳐 그립니다. 가로축 기본은 **누적 시험 횟수**입니다.
학습 방법마다 한 세대의 크기가 달라서 세대로 비교하면 공정하지 않기 때문입니다. 세로축은 학습에도 선정에도 쓰지 않은 비교 시험으로 잰 점수입니다.

## 반복 실행

「{{CMP_REPEAT_BUTTON}}」 단추는 같은 설정을 시드만 바꿔 여러 번 돌려 평균 선과 최소~최대 띠로 보여 줍니다.
결과가 시드 운인지 설정 덕인지 가릴 때 씁니다.

## 표와 그림 저장

「{{CMP_SAVE_PNG}}」 단추는 그래프를 그림으로, 「{{CMP_SAVE_CSV}}」 단추는 표를 엑셀에서 열리는 CSV로 저장합니다.

## 나란히 달리기

「{{CMP_RACE_BUTTON}}」 단추는 고른 실행의 챔피언을 같은 코스에서 나란히 달리게 합니다. [나란히 경주](evolab://help/race)
"""),
"race": ("나란히 경주", """# 나란히 경주

고른 실행 2~6개(또는 한 실행의 여러 세대)의 챔피언을 같은 코스에서 동시에 달리게 합니다.

## 코스

「{{RACE_NEXT_COURSE}}」 단추로 코스를 바꿉니다. 코스는 학습에 쓰지 않은 비교 시험에서 고릅니다.

## 순위

모든 칸의 환경이 같을 때만 순위를 매깁니다. 환경이 다르면 칸마다 따로 봅니다.

## 저장

「{{RACE_SAVE_GIF}}」 단추로 경주를 영상으로 남깁니다.
"""),
"records": ("기록과 내보내기", """# 기록과 내보내기

실행마다 데이터 폴더의 runs 안에 폴더가 하나 생깁니다. 그 폴더만 있으면 다시 보고, 이어 하고, 재현할 수 있습니다.

## 열기와 이어하기

- 「{{REC_OPEN}}」 단추: 실험실에서 그 실행을 봅니다.
- 「{{REC_RESUME}}」 단추: 마지막(또는 고른) 체크포인트에서 계속 학습합니다. 예전 체크포인트에서 이어 가면 새 실행 폴더로 갈라집니다.
- 「{{REC_CLONE}}」 단추: 같은 설정으로 새로 시작합니다.
- 「{{REC_EDIT_MEMO}}」 단추: 목록에서 알아보기 쉽게 메모를 붙입니다.

## 재현 묶음

「{{REC_BUNDLE}}」 단추로 같은 실험을 다시 돌릴 수 있는 zip을 만듭니다. 같은 PC·같은 버전이면 결과가 비트 단위로 같고, 다른 PC에서는 비슷한 결과가 나옵니다.

## 통계 CSV

「{{REC_CSV}}」 단추로 세대별 통계 전체를 저장합니다. 엑셀에서 바로 열립니다.

## 영상

실험실과 경주 창의 「{{VIDEO_SAVE}}」 단추로 지금 보는 재생을 저장합니다.

- **MP4**: 카카오톡·폰 앨범·발표 자료에 넣기 좋은 영상입니다. 기본은 화면 전체가 들어오는 고정 카메라입니다.
- **GIF**: 어디서나 보이는 움직이는 그림입니다.
- **폰 재생 파일(.html)**: 폰 브라우저에서 인터넷 없이 열리는 파일 하나입니다. 멈추고, 돌려 보고, 배속을 바꿀 수 있습니다.
"""),
"examples": ("내장 예제", """# 내장 예제

프로그램에 들어 있는 견본 환경입니다. 실험실 환경 목록과 에디터의 「{{ED_FROM_BUILTIN}}」 단추에 있습니다.

## 달 착륙선

추진기 셋으로 패드 위에 살며시 내려앉기. 처음 착지가 보이기까지 수십 세대가 걸립니다.

## 트랙 자동차

위에서 본 차가 벽에 닿지 않고 트랙을 돌기. 레이 센서로 벽까지 거리를 봅니다.

## 언덕 넘는 자동차

울퉁불퉁한 언덕을 멀리 넘어가기. 바퀴 크기와 차체도 함께 진화합니다. 처음 실행 안내가 쓰는 예제입니다.

## 점프 게임

다가오는 장애물을 뛰어넘거나 피하며 오래 버티기.

## 글라이더

높은 곳에서 던져 멀리 날기. 뇌 없이 날개와 무게 중심만 진화합니다.

## 바퀴 고르는 언덕차

바퀴 자리 네 곳 중 어디에 몇 개를 달지와 바퀴 크기, 차체 모양이 함께 진화합니다. 몸 구조가 바뀌는 진화를 보는 예제입니다.
"""),
"trouble": ("문제가 생겼을 때", """# 문제가 생겼을 때

## 오류 보고 묶음

[오류 보고 묶음 만들기](evolab://report)를 누르면 문제를 알리는 데 필요한 기록을 zip 하나로 모읍니다.
자동으로 보내지 않으니 저장한 파일을 직접 보내 주세요. 무엇이 들어가는지는 만들기 전에 목록으로 보여 줍니다.

## 느릴 때

가장 흔한 원인은 「{{LAB_PACE}}」 칸에서 「{{LAB_PACE_EVERY}}」 항목을 고르고 1배로 관전하는 것입니다. 이때는 학습이 재생을
기다립니다. 「{{LAB_PACE}}」 칸을 「{{LAB_PACE_SKIP}}」 항목으로 바꾸거나 「{{LAB_FAST}}」 단추를 고르세요. v6부터 같은 결과를
예전보다 빨리 계산합니다(결과는 예전 실행과 비트 단위로 같습니다).

노트북은 전원을 연결하고 윈도우 전원 모드를 “최고 성능”으로 바꾸세요. 실험실 「{{LAB_SETTINGS_GROUP}}」 묶음의 워커 수는 코어 수보다 하나 적게 두는 것이 좋습니다.
처음 실행은 윈도우 보안 검사 때문에 몇십 초 느릴 수 있습니다.

## 점수가 오르지 않을 때

같은 설정이라도 시드에 따라 빠르기가 크게 다릅니다. 시드만 바꿔 새로 시작하거나 비교 화면의 「{{CMP_REPEAT_BUTTON}}」 단추로 여러 시드를 돌려 보세요.
점수 블록이 원하는 행동에 점수를 주는지도 확인하세요([에디터](evolab://help/editor)).

## 창이 멈춘 것 같을 때

학습은 따로 돌기 때문에 창이 잠깐 느려도 기록은 남습니다. 1분 넘게 움직이지 않으면 창을 닫고 다시 열어 「{{TAB_RECORDS}}」 탭에서 이어 하세요.

## 저장하지 못할 때

윈도우 보안의 “제어된 폴더 액세스”가 켜져 있으면 문서 폴더에 쓰지 못할 수 있습니다. 진화 실험실을 허용하거나 「{{MENU_SETTINGS}}」 메뉴에서 데이터 폴더를 바꾸세요.
OneDrive로 옮겨진 문서 폴더는 실행 기록이 많으면 동기화가 느려집니다. [데이터 폴더 열기](evolab://data)

## 경고 창이 뜰 때

서명이 없는 새 프로그램이라 윈도우나 백신이 경고할 수 있습니다. 설치·실행 경고는 [설치와 제거](evolab://help/install)를 보세요.

[오류 기록 폴더 열기](evolab://logs)
"""),
"words": ("낱말 풀이", """# 낱말 풀이

- **세대**: 개체 여러 개를 시험하고 다음 후보를 만드는 한 바퀴.
- **개체**: 몸 수치와 뇌 가중치를 담은 숫자 목록 하나. 화면의 차 한 대, 착륙선 한 대.
- **집단**: 한 세대의 개체 전체(기본 32개).
- **유전자**: 개체를 이루는 숫자 목록. 학습 방법이 이 숫자들을 바꿔 가며 찾습니다.
- **몸 수치**: 바퀴 크기, 다리 길이처럼 진화하는 몸의 값.
- **몸 구조**: 몸에 어떤 부품이 달렸는지(바퀴 몇 개, 어느 자리). 부품이 생기고 없어지는 것도 진화할 수 있습니다.
- **뇌**: 센서 값을 받아 추진기·모터 출력을 내는 작은 신경망.
- **뉴런**: 신경망의 계산 단위 하나. 들어오는 값에 연결 세기를 곱해 더한 뒤 보통 -1..1로 눌러 내보냅니다.
- **시험**: 개체 하나를 시작 조건 하나로 끝까지 돌리는 것. 개체마다 몇 번 시험해 평균을 냅니다.
- **누적 시험 횟수**: 지금까지 한 시험의 수. 학습 방법끼리 비교하는 기준입니다.
- **시드**: 난수의 출발점. 같은 PC·같은 버전에서 같은 시드면 같은 결과가 나옵니다.
- **세대 대표**: 그 세대에서 학습 점수가 가장 높은 개체.
- **챔피언**: 검증 시험으로 다시 재서 고른, 지금까지 가장 나은 개체.
- **검증**: 학습에 쓰지 않은 시험으로 다시 재는 것. 운 좋게 높은 점수를 걸러 냅니다.
- **관전**: 집단 전체가 움직이는 모습을 보면서 학습하기.
- **고속**: 기다리지 않고 학습하고 챔피언만 가끔 보기.
- **CMA-ES**: 기본 학습 방법. 좋은 개체 쪽으로 찾는 범위를 옮기고 좁혀 갑니다.
- **유전 알고리즘**: 좋은 개체끼리 섞고 조금씩 바꿔 자식을 만드는 방법.
- **담금질**: 처음에는 나쁜 해도 받아들이다가 점점 깐깐해지는 방법.
- **체크포인트**: 이어 하기 위해 저장한 학습 상태.
- **관찰 기록**: 그래프를 그리려고 세대마다 남기는 집단의 기록. 학습 결과에는 영향이 없습니다.
- **재현 묶음**: 같은 실험을 다시 돌릴 수 있는 zip 파일.
"""),
"install": ("설치와 제거", """# 설치와 제거

## 설치

설치 파일을 실행합니다. 관리자 권한이 필요 없고 내 계정에만 설치됩니다.
받은 설치 파일을 처음 열면 “Windows의 PC 보호” 창이 뜰 수 있습니다. 이 프로그램은 아직 코드 서명이 없어서입니다.
“추가 정보”를 누른 뒤 “실행”을 누르면 설치됩니다.

## 업데이트

새 설치 파일을 그대로 실행하면 같은 자리에 덮어 설치됩니다. 환경 파일, 실행 기록, 설정은 그대로 남습니다.
프로그램이 켜져 있으면 닫으라고 알려 줍니다. 프로그램이 새 버전을 스스로 확인하지는 않습니다.

## 제거

윈도우 설정 → 앱 → 설치된 앱에서 진화 실험실을 제거합니다. 환경 파일과 실행 기록은 **지우지 않습니다**.
설정과 오류 기록을 지울지는 제거할 때 묻습니다. [데이터 폴더 열기](evolab://data)

## 설치 없이 쓰기

zip 판을 풀어 evolab.exe를 실행해도 됩니다. 기록 위치는 설치 판과 같습니다.
"""),
}

# ---- 풍선 도움말 (app/help/tooltips.py: 연결표 키 → 글. 위젯에 이미 풍선이 있으면 그대로 둔다) ----
HELP_TIPS: Final = {
    "lab.env": "학습할 환경을 고릅니다. 내장 예제는 프로그램에 들어 있는 견본입니다.",
    "lab.method": "해법을 찾는 방법입니다. 처음에는 CMA-ES를 그대로 쓰세요. 유전 알고리즘과 담금질은 비교용입니다.",
    "lab.seed": "난수의 출발점입니다. 같은 PC에서 같은 시드면 같은 결과가 나옵니다. 학습이 막히면 시드만 바꿔 다시 해 보세요.",
    "lab.popsize": "한 세대에 시험하는 개체 수입니다. 늘리면 세대마다 오래 걸리지만 더 넓게 찾습니다.",
    "lab.sigma": "처음에 유전자를 얼마나 넓게 흔들어 볼지입니다. 보통 그대로 둡니다.",
    "lab.ipop": "더 나아지지 않는다고 판단하면 개체 수를 두 배로 늘려 처음부터 다시 찾습니다(CMA-ES만).",
    "lab.trials": "개체 하나를 몇 번 시험해 평균 낼지입니다. 맨 아래 값(환경 파일 값)이면 환경 파일에 적힌 횟수를 씁니다.",
    "lab.workers": "동시에 계산할 도우미 프로그램 수입니다. 보통 코어 수보다 하나 적게 둡니다. 줄이면 PC가 덜 바쁩니다.",
    "lab.ga_elites": "다음 세대로 그대로 넘기는 가장 좋은 개체 수입니다.",
    "lab.ga_crossover": "두 부모의 유전자를 섞어 자식을 만들 확률입니다.",
    "lab.ga_mut_rate": "유전자 하나하나가 바뀔 확률입니다.",
    "lab.ga_mut_sigma": "유전자가 바뀔 때 얼마나 크게 바뀌는지입니다.",
    "lab.ga_decay": "세대마다 돌연변이 폭에 곱하는 값입니다. 1이면 줄지 않습니다.",
    "lab.sa_neighbors": "한 단계에 함께 시험하는 이웃 해의 수입니다.",
    "lab.sa_step": "이웃 해를 지금 해에서 얼마나 멀리 뽑을지입니다.",
    "lab.sa_temp": "처음 온도입니다. 높을수록 나쁜 해도 받아들여 넓게 돌아다닙니다. 맨 아래 값(자동)이면 처음 이웃들의 점수로 정합니다.",
    "lab.sa_cooling": "단계마다 온도에 곱하는 값입니다. 1에 가까울수록 천천히 식습니다.",
    "lab.max_generations": "이 세대 수가 되면 학습을 멈춥니다. 맨 아래 값(없음)이면 세대 수로는 멈추지 않습니다.",
    "lab.max_evaluations": "누적 시험 횟수가 이 값이 되면 멈춥니다. 맨 아래 값(없음)이면 시험 횟수로는 멈추지 않습니다.",
    "lab.max_minutes": "이 시간(분)이 지나면 멈춥니다. 맨 아래 값(없음)이면 시간으로는 멈추지 않습니다.",
    "lab.target": "챔피언의 검증 점수가 이 값에 닿으면 멈춥니다. 맨 아래 값(없음)이면 점수로는 멈추지 않습니다.",
    "lab.stagnation": "챔피언이 이 세대 수 동안 나아지지 않으면 멈춥니다. 맨 아래 값(없음)이면 정체로는 멈추지 않습니다.",
    "lab.restart_on": "정체하면 학습 방법을 새로 만들어 다시 찾습니다. 시험해 보니 평균 성공률을 높이지 못해 기본으로 꺼 두었습니다.",
    "lab.restart_patience": "검증 최고점이 이 학습 시험 수 동안 오르지 않으면 정체로 봅니다.",
    "lab.restart_from": "다시 찾을 때 처음 상태에서 할지, 챔피언에서 할지입니다.",
    "lab.qualify_entrants": "처음에 학습 방법 여러 개를 짧게 겨루게 해 가장 나은 하나로 이어 갑니다. 맨 아래 값(끔)이면 예선 없음.",
    "lab.qualify_trials": "예선 출전자 하나가 쓰는 학습 시험 수입니다.",
    "lab.start": "고른 환경과 설정으로 새 실행을 만들어 학습을 시작합니다. 기록은 「기록」 탭에 자동으로 남습니다.",
    "lab.pause": "이번 세대를 마치고 멈춥니다.",
    "lab.continue": "멈춘 학습을 이어서 합니다.",
    "lab.step": "한 세대만 더 계산하고 멈춥니다.",
    "lab.end": "체크포인트를 남기고 학습 프로그램을 닫습니다. 나중에 「기록」에서 이어 할 수 있습니다.",
    "lab.spectate": "집단 전체가 움직이는 모습을 보여 줍니다. 학습이 재생을 기다릴지는 아래 「관전 방식」 칸에서 고릅니다.",
    "lab.fast": "기다리지 않고 학습만 빠르게 합니다. 정해 둔 세대마다 챔피언만 재생합니다.",
    "lab.speed": "관전 재생 속도입니다. 빠를수록 세대가 빨리 넘어갑니다.",
    "lab.fast_interval": "고속 모드에서 몇 세대마다 챔피언을 재생할지입니다.",
    "lab.versus": "1세대에서 가장 잘한 개체와 지금 챔피언을 같은 조건에서 겹쳐 봅니다.",
    "lab.live": "다시 보기를 그만두고 지금 학습 중인 세대로 돌아갑니다.",
    "lab.video": "지금 보는 재생을 영상(MP4·GIF)이나 폰 재생 파일로 저장합니다.",
    "compare.repeat": "지금 고른 설정을 시드만 바꿔 여러 번 돌립니다.",
    "compare.refresh": "실행 목록을 다시 읽습니다.",
    "compare.clear": "고른 실행을 모두 풉니다.",
    "compare.png": "지금 그래프를 그림 파일로 저장합니다.",
    "compare.csv": "아래 표를 CSV로 저장합니다(엑셀에서 열림).",
    "compare.race": "고른 실행의 챔피언들을 같은 코스에서 나란히 달리게 합니다.",
    "records.refresh": "실행 목록을 다시 읽습니다.",
    "records.memo": "고른 실행에 메모를 붙입니다. 목록에서 알아보기 쉽게 합니다.",
    "play.restart": "같은 조건으로 처음부터 다시 합니다 (R).",
    "play.new_seed": "시작 조건을 새로 뽑아 다시 합니다 (N).",
    "play.save": "방금 끝난 판의 키 입력을 저장합니다. 나중에 똑같이 다시 재생할 수 있습니다.",
    "play.load": "저장한 키 입력을 불러와 그대로 다시 재생합니다.",
    "editor.builtin": "내장 예제를 불러와 고쳐 만듭니다. 처음이라면 이것부터 해 보세요.",
}

# ---- 빈 상태 (app/help/hints.py) ----
HINT_LAB_IDLE_TITLE: Final = "학습할 준비가 됐습니다"
HINT_LAB_IDLE: Final = "「{start}」을 누르면 {env} {pop}개가 동시에 연습을 시작합니다."
HINT_LAB_TOUR: Final = "처음이라면 따라 하기로 시작"
HINT_DASH_EMPTY: Final = "학습을 시작하면 세대마다 여기에 그래프가 그려집니다."
HINT_DASH_ONE: Final = "1세대만 끝났습니다. 2세대부터 선으로 이어집니다."
HINT_EDITOR_EMPTY: Final = ("왼쪽 「{shapes}」에서 도구를 골라 캔버스에 끌어 그리세요. 처음이라면 「{builtin}」으로 예제를 불러와 "
                            "고쳐 보는 것이 쉽습니다.")

# ---- 오류 문구 (app/errors.py) ----
ERR_NEXT_LABEL: Final = "다음에 할 일"
ERR_REPORT_BUTTON: Final = "오류 보고 묶음 만들기…"
ERR_PERMISSION: Final = "파일을 쓸 수 없습니다. 다른 프로그램(엑셀, 탐색기 미리 보기, 백신 등)이 잡고 있거나 쓰기 권한이 없습니다."
ERR_PERMISSION_NEXT: Final = "그 프로그램을 닫고 다시 해 보세요. 계속되면 「{menu}」에서 데이터 폴더를 다른 곳으로 바꾸세요."
ERR_NOT_FOUND: Final = "파일이나 폴더를 찾을 수 없습니다: {path}"
ERR_NOT_FOUND_NEXT: Final = "옮기거나 지우지 않았는지 확인하세요. 실행 기록이면 「{records}」 탭에서 목록을 새로 고치세요."
ERR_DISK_FULL: Final = "디스크 공간이 부족합니다."
ERR_DISK_FULL_NEXT: Final = "「{records}」 탭에서 필요 없는 실행을 지우거나, 데이터 폴더를 공간이 넉넉한 드라이브로 옮기세요."
ERR_MEMORY: Final = "메모리가 부족합니다."
ERR_MEMORY_NEXT: Final = "다른 프로그램을 닫고, 실험실의 워커 수나 개체 수를 줄여 보세요."
ERR_ENV_FILE_NEXT: Final = "에디터에서 이 파일을 열면 「{issues}」 탭에 고칠 곳이 표시됩니다."
ERR_BUILD_NEXT: Final = "몸 수치나 수식이 만들 수 없는 모양을 만듭니다. 에디터에서 값을 바꾸고 직접 조종으로 먼저 확인하세요."
ERR_RUN_STORE_NEXT: Final = "실행 폴더가 망가졌을 수 있습니다. 「{records}」에서 다른 실행을 열거나, 이 실행을 지우고 다시 학습하세요."
ERR_RECORD_NEXT: Final = "다른 기록 파일을 고르거나, 이 환경으로 새로 몰아 보고 다시 저장하세요."
ERR_UNEXPECTED: Final = "예상하지 못한 오류가 났습니다: {kind}"
ERR_UNEXPECTED_NEXT: Final = "학습 기록은 자동으로 저장됩니다. 같은 일이 되풀이되면 오류 보고 묶음을 만들어 보내 주세요."

# ---- 잡히지 않은 예외·갑자기 닫힘·시작 실패 (diagnostics.py, app/report/) ----
DIAG_CRASH_TITLE: Final = "문제가 생겼습니다"
DIAG_CRASH_TEXT: Final = ("예상하지 못한 오류가 났습니다. 하던 일은 대부분 계속할 수 있지만, 화면이 이상하면 프로그램을 다시 여세요.\n"
                          "오류 내용은 기록에 남겼습니다. 알려 주시려면 「오류 보고 묶음 만들기」를 누르세요.")
DIAG_CRASH_AGAIN: Final = "같은 오류가 또 났습니다(기록에 남김): {line}"
DIAG_PREVIOUS_TITLE: Final = "지난번에 갑자기 닫혔습니다"
DIAG_PREVIOUS_TEXT: Final = "지난번에 진화 실험실이 정상적으로 닫히지 않았습니다. 그때 기록으로 오류 보고 묶음을 만들까요?"
DIAG_PREVIOUS_YES: Final = "묶음 만들기…"
DIAG_PREVIOUS_NO: Final = "괜찮습니다"
DIAG_STARTUP_TITLE: Final = "진화 실험실을 시작하지 못했습니다"
DIAG_STARTUP_TEXT: Final = ("프로그램 창을 띄우지 못했습니다: {reason}\n\n"
                            "바탕 화면에 오류 보고 묶음을 만들었습니다: {path}\n"
                            "이 파일을 보내 주시면 원인을 찾는 데 도움이 됩니다.")
SESSION_LOST_MEANING: Final = "{base} ({meaning})"
EXIT_CODE_TEXT: Final = {
    3221225477: "메모리 접근 오류",
    3221225725: "스택 넘침",
    3221226505: "버퍼 검사 실패",
    3221225794: "DLL 초기화 실패",
    3221225786: "Ctrl+C 또는 창 닫기로 끝남",
    3221225495: "메모리 부족",
}
EXIT_CODE_SIGNAL: Final = "신호 {name}로 끝남"
EXIT_CODE_UNKNOWN: Final = "알 수 없는 원인"

# ---- 오류 보고 묶음 (app/report/, cli.py) ----
REP_TITLE: Final = "오류 보고 묶음 만들기"
REP_INTRO: Final = "문제를 알릴 때 보낼 zip 파일을 만듭니다. 인터넷으로 보내지 않습니다. 만든 파일을 메일이나 메신저에 붙여 보내 주세요."
REP_NOTE_LABEL: Final = "무엇을 하다가 문제가 생겼나요? (적지 않아도 됩니다)"
REP_NOTE_PLACEHOLDER: Final = "예: 학습을 시작하고 몇 초 뒤에 오류 상자가 떴습니다."
REP_ITEMS_LABEL: Final = "넣을 것"
REP_ITEM_NAMES: Final = {
    "report": "프로그램·PC 정보", "errors": "최근 오류 기록", "faults": "치명적 오류 기록",
    "stderr": "오류 출력 로그(마지막 부분)", "selftest": "마지막 자체 점검 기록", "settings": "화면 설정과 데이터 폴더 설정",
    "run": "지금 실행의 설정·변경 기록·통계(마지막 50세대)·환경 사본", "env": "지금 열린 환경 파일",
}
REP_PRIVACY: Final = ("사용자 이름·컴퓨터 이름이 든 경로는 「{mask}」로 바꿔 적습니다. 학습한 유전자, 체크포인트, 다른 실행, "
                      "직접 조종 기록, 내보낸 영상, 화면 그림은 넣지 않습니다.")
REP_FILES: Final = "넣을 파일 {count}개, 약 {size}"
REP_SAVE: Final = "저장…"
REP_CANCEL: Final = "닫기"
REP_SAVE_TITLE: Final = "오류 보고 묶음 저장"
REP_FILTER: Final = "오류 보고 묶음 (*.zip)"
REP_DONE: Final = "오류 보고 묶음을 만들었습니다: {path}"
REP_OPEN_FOLDER: Final = "폴더 열기"
REP_FAILED: Final = "오류 보고 묶음을 만들지 못했습니다: {reason}"
REP_FILE_NAME: Final = "진화실험실_오류보고_{stamp}.zip"
REP_USER_MASK: Final = "<사용자>"
REP_HOST_MASK: Final = "<컴퓨터>"
REP_TXT_TITLE: Final = "진화 실험실 오류 보고"
REP_TXT_FIELDS: Final = {
    "created": "만든 시각", "version": "프로그램 버전", "frozen": "exe 배포본", "os": "운영 체제", "cpu": "CPU",
    "cores": "코어(물리/논리)", "memory": "메모리", "screen": "화면(배율)", "python": "파이썬", "qt": "Qt",
    "data_dir": "데이터 폴더", "custom_data_dir": "데이터 폴더를 바꿈", "reason": "만든 까닭", "note": "사용자 메모",
}
REP_REASONS: Final = {"menu": "메뉴", "error": "오류 상자", "crash": "예상하지 못한 오류", "previous": "지난번 갑자기 닫힘",
                      "startup": "시작 실패", "cli": "명령줄", "tour": "처음 실행 안내 실패"}
REP_YES: Final = "예"
REP_NO: Final = "아니오"
CLI_HELP_REPORT: Final = "오류 보고 묶음(zip)을 만듭니다."
CLI_HELP_REPORT_OUT: Final = "만들 zip 파일 경로 (기본: 바탕 화면)"
CLI_HELP_REPORT_RUN: Final = "함께 넣을 실행 폴더"
CLI_HELP_REPORT_OPEN: Final = "만든 뒤 그 폴더를 엽니다."
CLI_REPORT_DONE: Final = "오류 보고 묶음을 만들었습니다: {path} (파일 {count}개, {size})"
CLI_UNCAUGHT: Final = "예상하지 못한 오류로 멈췄습니다: {reason}. 기록: {path}"

# ---- 자체 점검 (selftest.py) ----
SELFTEST_CHECK_TOUR: Final = "처음 실행 안내 (1세대까지 자동 진행)"
SELFTEST_TOUR_DETAIL: Final = "창 → 1세대 재생 {seconds:.1f}초, 누름 {clicks}번"
SELFTEST_TOUR_FULL_DETAIL: Final = "창 → {goal}세대 {seconds:.1f}초, 1세대 평균 {first:.1f} → {goal}세대 평균 {now:.1f}"
SELFTEST_ERR_TOUR_SLOW: Final = "{what}까지 {seconds:.0f}초가 걸렸습니다 (기준 {limit}초)."
SELFTEST_CHECK_HELP: Final = "도움말 쪽"
SELFTEST_HELP_DETAIL: Final = "쪽 {pages}개 열림, 자리 모두 풀림"
SELFTEST_CHECK_QT_KO: Final = "Qt 한국어 번역"
SELFTEST_ERR_QT_KO: Final = "Qt 기본 단추가 한국어가 아닙니다: {text}"
SELFTEST_CHECK_LOGS: Final = "기록 폴더 쓰기와 오류 보고 묶음"
CLI_HELP_SELFTEST_TOUR: Final = "처음 실행 안내 점검을 어디까지 할지: gen1(1세대 재생까지, 기본), full(목표 세대까지 4배속)"

# ---- v3.0 B 묶음(도움말 창·F1, app/help/context.py) ----
HELP_HEADING_VIDEO: Final = "영상"                                  # 「기록과 내보내기」 쪽 절 제목: 내보내기 대화 상자(v3.1)의 F1이 여는 곳(설계 D21)

# ---- v3.0 D 묶음(진단·오류 보고 묶음: diagnostics.py, app/report/) ----
DIAG_CRASH_MORE: Final = "또 다른 오류가 났습니다(기록에 남김): {line}"          # 상자를 이미 3번 보인 뒤의 다른 오류(상태 줄)
DIAG_STARTUP_NO_BUNDLE: Final = ("프로그램 창을 띄우지 못했습니다: {reason}\n\n"
                                 "오류 보고 묶음도 만들지 못했습니다: {error}")  # 시작 실패 때 묶음까지 실패

# ---- v3.0 C 묶음(화면 손질: app/errors.py) ----
ERR_SETTINGS_NEXT: Final = "오른쪽 설정 칸의 값을 고친 뒤 다시 해 보세요."  # 학습 설정 검사·「설정 적용」이 거절됐을 때(실험실)
ERR_ENV_STRUCTURE_NEXT: Final = ("다른 프로그램으로 고친 파일이면 위에 적힌 곳을 고친 뒤 다시 여세요. "
                                 "새로 만들려면 「{builtin}」을 누르세요.")  # 에디터가 구조가 망가진 환경 파일을 열지 못했을 때

# ---- v3.0 I 묶음(통합: selftest.py 처음 실행 안내·묶음 점검) ----
SELFTEST_CHECK_TOUR_FULL: Final = "처음 실행 안내 ({goal}세대까지 4배속 자동 진행)"     # `selftest --tour full`(설계 W3)
SELFTEST_TOUR_WHAT_GEN1: Final = "1세대 재생"                                       # SELFTEST_ERR_TOUR_SLOW의 {what}
SELFTEST_TOUR_WHAT_GOAL: Final = "{goal}세대"                                       # SELFTEST_ERR_TOUR_SLOW의 {what}(--tour full)
SELFTEST_ERR_TOUR_FAILED: Final = "처음 실행 안내가 끝까지 가지 못했습니다: {detail}"
SELFTEST_ERR_CRASH_NOT_LOGGED: Final = "일부러 멈춘 자식 프로세스의 치명적 오류가 faults.log에 남지 않았습니다: {detail}"  # EVOLAB_SELFTEST_CRASH=1(W2)

# ---- v3.0 검토 수정(안내 실패 글·④ 배속·실험실 빈 상태, app/help/tour.py·hints.py) ----
ONB_STEP_WATCH_FAST: Final = ("처음 보이는 것이 1세대입니다. 아직 배운 것이 없어 대부분 멈추거나 뒤집힙니다. "
                              "잠시 뒤 다음 단계로 넘어갑니다.")           # ④: 배속이 이미 4배 이상일 때(설계 D10)
ONB_FAILED_NO_DETAIL: Final = "학습 프로세스를 띄우지 못했습니다"          # 실패 카드의 {reason}: 까닭 글이 오지 않았을 때
HINT_LAB_IDLE_ANY: Final = "「{start}」을 누르면 개체 {pop}개가 「{env}」에서 동시에 연습을 시작합니다."  # 실험실 빈 상태(모든 환경)

# ---------------------------------------------------------------------------
# v3.1 나누기와 설치 (0' 묶음: v3 설계 6절의 「# v3.1」 줄, 명세 부록 A79~A89): 설치 프로그램, 휴대용 zip, 내보내기 대화 상자,
# 폰 재생 파일, 명령줄 내보내기, 자체 점검 「내보내기」.
# - VIDEO_SAVE·VIDEO_SAVE_TITLE·RACE_SAVE_GIF는 위 제자리에서 「내보내기…」·「내보내기」로 바꿨다(같은 이름을 두 번 쓰지 않음).
# - HELP_PAGES의 「설치와 제거」 쪽(설치 프로그램 판), 「기록과 내보내기」 쪽 「영상」 절(MP4·GIF·폰 재생 파일), 「나란히 경주」 쪽
#   「저장」 절, HELP_TIPS["lab.video"]도 제자리에서 v3.1 글로 바꿨다.
# - 설계 6절에 없던 세 줄: PORTABLE_README_NAME(D44 「설치 없이 쓰기.txt」), EXP_SAVE·EXP_CLOSE(5.6의 [저장…] [닫기]).
# - 뒤 묶음(G·M·P·F)이 문구를 더하면 이 블록 뒤에 자기 묶음 이름을 머리로 단 블록으로 붙인다.
# ---------------------------------------------------------------------------
# ---- 설치 프로그램 (packaging/make_installer.py가 [CustomMessages]로 옮김. %1·%2·%n은 Inno 자리) ----
INST_PUBLISHER: Final = "진화 실험실"
INST_TASKS_GROUP: Final = "바로 가기:"
INST_TASK_DESKTOP: Final = "바탕 화면에 바로 가기 만들기"
INST_RUN_NOW: Final = "진화 실험실 실행"
INST_REPORT_SHORTCUT: Final = "진화 실험실 오류 보고 묶음 만들기"
INST_NEWER_INSTALLED: Final = ("더 새로운 버전(%1)이 이미 설치되어 있습니다. 이 설치 파일은 %2입니다.%n%n"
                               "그래도 예전 버전으로 바꿔 설치할까요? 새 버전에서 만든 환경 파일은 예전 버전에서 열리지 않을 수 있습니다.")
INST_ASK_SETTINGS: Final = ("프로그램 설정과 오류 기록(창·그래프 배치, 데이터 폴더 위치, 로그)도 지울까요?%n%n위치: %1%n%n"
                            "[아니요]를 누르면 남겨 두었다가 다시 설치할 때 그대로 씁니다.")
INST_DATA_KEPT: Final = ("실험 기록은 지우지 않았습니다.%n%n환경 파일, 실행 기록, 직접 조종 기록은 데이터 폴더(기본: %1)에 "
                         "그대로 있습니다. 필요 없으면 그 폴더를 직접 지우세요.")
PORTABLE_README_NAME: Final = "설치 없이 쓰기.txt"          # 휴대용 zip 안 PORTABLE_README 파일 이름(설계 D44)
PORTABLE_README: Final = (
    "진화 실험실 {version} (설치 없이 쓰기)\r\n\r\n"
    "이 폴더의 evolab.exe를 두 번 누르면 실행됩니다. 설치 판과 같은 데이터 폴더(문서\\진화실험실)를 씁니다.\r\n"
    "zip을 풀기 전에 zip 파일의 속성에서 「차단 해제」를 켜면 실행할 때마다 나오는 경고가 줄어듭니다.\r\n"
    "시작 메뉴 바로 가기와 제거가 필요하면 설치 파일(evolab-setup-{version}.exe)을 쓰세요.\r\n"
)

# ---- 내보내기 대화 상자 (app/export/dialog.py) ----
EXP_WHAT: Final = "무엇을"
EXP_WHAT_REPLAY: Final = "지금 보는 재생 ({label})"
EXP_WHAT_RACE: Final = "이 경주 ({n}칸)"
EXP_FORMAT: Final = "형식"
EXP_FORMAT_MP4: Final = "영상 (MP4) — 카카오톡·폰 앨범·발표 자료"
EXP_FORMAT_GIF: Final = "움직이는 그림 (GIF) — 문서·게시판"
EXP_FORMAT_HTML: Final = "폰 재생 파일 (HTML) — 아주 작고, 인터넷 없이 폰 브라우저에서 열림"
EXP_SIZE: Final = "크기"
EXP_SIZE_SMALL: Final = "작게 (640×360)"
EXP_SIZE_MEDIUM: Final = "보통 (1280×720)"
EXP_CAMERA: Final = "카메라"
EXP_CAMERA_FIXED: Final = "고정: 전체가 보이게"
EXP_CAMERA_FOLLOW: Final = "따라가기: 관전 화면처럼"
EXP_INFO_LENGTH: Final = "길이 {seconds:.0f}초 · {frames}프레임"
EXP_INFO_MF: Final = "윈도우 영상 인코더로 만듭니다 (약 {mb:.0f}MB 예상)."
EXP_INFO_BUILTIN: Final = "내장 영상 쓰기로 만듭니다. 압축하지 않은 영상이라 파일이 큽니다 (만들면서 크기를 보여 줍니다)."
EXP_INFO_BUILTIN_TIP: Final = ("이 PC에서 윈도우 영상 인코더를 쓸 수 없습니다: {reason}. 윈도우 N·KN 판은 「미디어 기능 팩」을 "
                               "설치하면 쓸 수 있습니다.")
EXP_BIG_ASK: Final = ("파일이 {mb:.0f}MB쯤 될 것 같습니다. 카메라를 「{fixed}」 쪽으로 바꾸면 훨씬 작아집니다. 그대로 만들까요?")
EXP_PROGRESS_FRAMES: Final = "만드는 중… {k}/{n}프레임 · 지금까지 {mb:.1f}MB"
EXP_DONE: Final = "저장했습니다: {path}"
EXP_OPEN_FOLDER: Final = "폴더 열기"
EXP_FAILED: Final = "만들지 못했습니다: {reason}"
EXP_MF_FALLBACK: Final = "윈도우 영상 인코더가 실패해 내장 영상 쓰기로 다시 만들었습니다 ({reason})."
EXP_PROCESS_DIED: Final = "내보내기 프로그램이 갑자기 끝났습니다({meaning})."
EXP_FILTER_MP4: Final = "MP4 영상 (*.mp4)"
EXP_FILTER_HTML: Final = "폰 재생 파일 (*.html)"
EXP_BAD_SUFFIX: Final = "파일 이름은 .mp4, .gif, .html 중 하나로 끝나야 합니다: {name}"
EXP_ENCODER_MF: Final = "윈도우 영상 인코더 ({name})"
EXP_ENCODER_BUILTIN: Final = "내장 영상 쓰기"
EXP_MF_OFF: Final = "꺼 둠"
EXP_MF_NOT_WINDOWS: Final = "윈도우가 아님"
EXP_SAVE: Final = "저장…"                                # 대화 상자 단추(설계 5.6)
EXP_CLOSE: Final = "닫기"                                # 대화 상자 단추(설계 5.6)

# ---- 폰 재생 파일 (app/export/phone.py → 재생기 안 JSON "text". 재생기 HTML 틀에는 한글이 없다) ----
PHONE_TITLE: Final = "{env} · {label}"
PHONE_TOO_BIG: Final = "재생이 너무 길거나 칸이 많아 폰 재생 파일로 만들 수 없습니다(프레임 {frames}개, 상한 {limit}개)."
PHONE_HOW_TO_TITLE: Final = "폰으로 옮기는 법"
PHONE_HOW_TO: Final = ("만든 파일 하나를 카카오톡 「나와의 채팅」, 메일, USB 등으로 폰에 옮긴 뒤 브라우저(Chrome, 삼성 인터넷, "
                       "Safari)로 여세요. 인터넷 연결 없이 재생됩니다.\n\n"
                       "메신저나 파일 앱의 미리 보기에서는 움직이지 않을 수 있습니다. 그럴 때는 「다른 앱으로 열기」에서 브라우저를 고르세요.")
PHONE_HOW_TO_DONT_SHOW: Final = "다시 보지 않기"
PHONE_TEXT: Final = {
    "play": "재생", "pause": "멈춤", "restart": "처음부터", "speed": "{x}배", "seconds": "{t}초 / {total}초",
    "camera_fit": "전체 보기", "camera_follow": "따라가기", "score": "점수 {score}", "rank": "{n}등",
    "legend_hint": "눌러서 켜고 끄기", "rotate": "가로로 돌리면 크게 보입니다",
    "newer": "더 새 판 진화 실험실에서 만든 파일입니다. 새 판으로 다시 내보내 주세요.",
    "noscript": ("이 미리 보기에서는 움직이지 않습니다. 파일을 Chrome·Safari·삼성 인터넷 같은 브라우저로 여세요. "
                 "아래 그림은 마지막 장면입니다."),
    "footer": "진화 실험실 {version}에서 만든 재생 파일 · 인터넷 연결 없이 열립니다",
}

# ---- 명령줄 내보내기 (cli.py) ----
CLI_HELP_EXPORT: Final = "실행의 챔피언 재생을 MP4·GIF·폰 재생 파일로 내보냅니다(창 없이)."
CLI_HELP_EXPORT_FORMAT: Final = "형식: mp4, gif, html"
CLI_HELP_EXPORT_GEN: Final = "몇 세대 챔피언 (기본: 마지막)"
CLI_HELP_EXPORT_WHAT: Final = "champion(챔피언) 또는 versus(1세대 대표와 챔피언)"
CLI_HELP_EXPORT_SIZE: Final = "MP4 크기: small(640×360), medium(1280×720)"
CLI_HELP_EXPORT_CAMERA: Final = "fixed(고정, 기본) 또는 follow(따라가기)"
CLI_HELP_EXPORT_ENCODER: Final = "MP4 인코더: auto(기본), mf(윈도우 영상 인코더), builtin(내장)"
CLI_HELP_EXPORT_OUT: Final = "만들 파일 경로(확장자가 형식과 맞아야 함)"
CLI_EXPORT_DONE: Final = "만들었습니다: {path} ({size}, {frames}프레임, {encoder})"
CLI_HELP_VIDEO_PROBE: Final = "이 PC에서 쓸 MP4 인코더를 알려 줍니다."
CLI_VIDEO_PROBE_LINE: Final = "MP4 인코더: {encoder} · 윈도우 영상 인코더: {mf}"
SELFTEST_CHECK_EXPORT: Final = "내보내기 (내장 MP4·폰 재생 파일·윈도우 영상 인코더 확인)"
SELFTEST_EXPORT_DETAIL: Final = "내장 MP4 {mp4} · 폰 재생 파일 {html} · 윈도우 영상 인코더: {mf}"

# ---------------------------------------------------------------------------
# v3.1 F 묶음(내보내기 연결: app/export/dialog.py·runner.py·worker.py, cli.py, selftest.py, editor/palette.py)
# ---------------------------------------------------------------------------
EXP_INFO_PROBING: Final = "이 PC에서 쓸 영상 인코더를 확인하는 중…"          # 대화 상자 안내 줄(인코더 시험 자식이 끝나기 전)
EXP_INPROCESS_NOTE: Final = "내보내기 프로그램을 따로 띄우지 못해 이 창에서 만듭니다(내장 영상 쓰기)."  # 되돌릴 지점 10(errors.jsonl)
EXP_ENCODER_GIF: Final = "GIF"                                            # CLI_EXPORT_DONE의 {encoder}(GIF)
EXP_ENCODER_HTML: Final = "폰 재생 파일"                                  # CLI_EXPORT_DONE의 {encoder}(폰 재생 파일)
SELFTEST_EXPORT_PART: Final = "{frames}프레임 · {kb:.0f}KB"               # SELFTEST_EXPORT_DETAIL의 {mp4}·{html}
EDC_SNAP_STEP_TIP: Final = "격자 한 칸의 크기입니다. 점을 찍거나 옮기면 이 간격에 맞춰집니다."  # 에디터 격자 간격 칸 풍선

# ---------------------------------------------------------------------------
# v3.1 검토 수정(내보내기: app/export/runner.py). EXP_PROCESS_DIED는 위 제자리에서 「GIF로 저장해 보세요.」를 뺐다: GIF·폰 파일
# 자식이 끝났을 때나 인코더 시험 까닭(EXP_INFO_BUILTIN_TIP의 {reason})에서는 맞지 않는 말이라 MP4 작업이 끝났을 때만 붙인다.
# ---------------------------------------------------------------------------
EXP_PROCESS_DIED_GIF_HINT: Final = "GIF로 저장해 보세요."  # MP4 내보내기 자식이 갑자기 끝났을 때 EXP_PROCESS_DIED 뒤에 붙임

# ---------------------------------------------------------------------------
# v6.0 빠르게 (0단계 묶음: v6 설계 6절의 v6.0 줄, 명세 부록 A90~). v6.1(자라는 뇌) 줄은 v6.1 0'단계에서 더한다.
# 관전 기본은 「학습 먼저」(PACE_DEFAULT = "skip", 사용자 결정 G1·Q1). 위 제자리에서 바꾼 것(같은 이름을 두 번 쓰지 않음):
# - BENCH_SINGLE: 「단일 코어(창 {window})」 — engine/commands.py bench가 window=를 넘긴다(E 묶음).
# - ONB_STEP_WATCH: 학습은 재생과 상관없이 계속된다(처음 실행 안내 ④). v6.0 검토: 「학습 먼저」는 15초 안에 새 세대로 넘어가므로
#   ONB_STEP_WATCH·ONB_STEP_WATCH_FAST의 「지금 보이는 것이 1세대」를 「처음 보이는 것이 1세대」로 바꾸고 새 세대로 넘어간다고 적었다
#   (설계 6절 글과 다름, 보고서).
# - HELP_PAGES 「실험실」 쪽 「관전과 고속」 절, 「문제가 생겼을 때」 쪽 「느릴 때」 절 맨 앞 문단. 설계 글에서 자리 바로 뒤의 조사
#   (「…」를·과·으로)만 「칸」·「항목」·「단추」·「방식」으로 바꿨다(v3 D19: 자리 뒤에 조사를 붙이지 않음, 채운 글의 받침과 어긋남).
# - HELP_TIPS["lab.spectate"]: 「학습은 재생이 끝날 때까지 기다립니다」를 뺐다(기본이 「학습 먼저」라 맞지 않음, 설계 밖).
# HELP_TIPS["lab.pace"] = LAB_PACE_TIP는 P 묶음이 「관전 방식」 칸(lab.pace_combo)과 연결표 키 lab.pace를 만들 때 함께 더한다
# (HELP_TIPS 키는 모두 연결표에 있어야 한다: tests/test_v3_strings.py, test_help_anchors.py).
# ---------------------------------------------------------------------------
# ---- 관전 방식 (app/lab/view.py) ----
LAB_PACE: Final = "관전 방식"
LAB_PACE_SKIP: Final = "학습 먼저 (새 세대만 보기)"
LAB_PACE_EVERY: Final = "세대마다 다 보기 (학습이 기다림)"
LAB_PACE_TIP: Final = ("학습 먼저: 학습은 쉬지 않고, 재생이 끝날 때마다 그때까지 끝난 가장 새 세대를 보여 줍니다. "
                       "건너뛴 세대도 기록과 그래프에 모두 남고 타임라인으로 다시 볼 수 있습니다.\n"
                       "세대마다 다 보기: 모든 세대를 차례로 보여 주고, 재생이 끝날 때까지 학습이 기다립니다. "
                       "1배속이면 학습이 몇 배 느려집니다.")
LAB_HUD_LIVE_SKIP: Final = "{gen}세대 · 시험 0번 · {t:.1f}초 · {speed}배 · 학습은 {latest}세대까지"
LAB_PACE_WAIT_HINT: Final = ("학습이 재생을 기다리느라 약 {times:.1f}배 느려졌습니다. "
                             "「관전 방식」을 「학습 먼저」로 바꾸면 학습이 쉬지 않습니다.")
LAB_PACE_CHANGED_NOTICE: Final = ("관전 방식이 「학습 먼저」로 바뀌었습니다. 학습은 쉬지 않고 화면은 가장 새 세대를 보여 줍니다. "
                                  "예전처럼 세대마다 다 보려면 「보기」 묶음에서 바꾸세요.")  # 예전 설정(lab/mode가 있음)에 한 번

# ---- 성능 측정과 자체 점검 (engine/commands.py, selftest.py) ----
BENCH_WINDOW: Final = ("평가 창(살아 있는 에피소드 수): {rows} · 가장 빠른 창 {best}개 "
                       "(번갈아 {rounds}번 잰 중앙값, 결과는 창과 상관없이 같음)")
BENCH_WINDOW_ROW: Final = "{w}개 {us:.1f}µs"
BENCH_WINDOW_SEP: Final = " · "
BENCH_PLAYBACK: Final = ("관전 재생 길이(무작위 유전자 1세대, 시험 0번 가장 긴 궤적): {sec:.1f}초 → "
                         "「세대마다 다 보기」 1배면 세대마다 최소 {wait:.1f}초")
BENCH_FAST_PATHS: Final = "pymunk C 함수 바로 부르기: {state}"
BENCH_FAST_ON: Final = "켜짐(레이·정해진 움직임·몸 상태·힘·한 스텝)"
BENCH_FAST_OFF: Final = "꺼짐 — 이 pymunk 판에는 필요한 내부 함수가 없어 보통 길을 씁니다(결과는 같고 더 느립니다)"
SELFTEST_CHECK_FASTPATH: Final = "빠른 계산 길 (물리 엔진 C 함수를 바로 부르는 길과 보통 길의 결과 비교)"
SELFTEST_FASTPATH_SAME: Final = "빠른 계산 길이 켜져 있고 보통 길과 결과가 같습니다."
SELFTEST_FASTPATH_OFF: Final = "빠른 계산 길이 꺼져 있습니다(이 물리 엔진 판에 필요한 내부 함수가 없음). 결과는 같고 계산만 느립니다."
SELFTEST_FASTPATH_DIFF: Final = "빠른 계산 길과 보통 길의 결과가 다릅니다: {detail}"

# ---------------------------------------------------------------------------
# v6.0 P 묶음 (관전 방식): 「관전 방식」 칸(lab.pace_combo)의 풍선. 연결표 키 lab.pace(app/help/anchors.py)와 함께 더한다.
# ---------------------------------------------------------------------------
HELP_TIPS["lab.pace"] = LAB_PACE_TIP

# ---------------------------------------------------------------------------
# v6.1 자라는 뇌 (0'단계 묶음: v6 설계 6절의 「# v6.1」 줄). 학습 방법 "grow"(CMA-ES만, 선택 기능이고 새 실행의
# 기본은 「크기 고정」), 실험실 「뇌」 줄, 뇌 그림의 꺼진 자리·새 뉴런, 「뇌 크기」 타일, 명령줄, 자체 점검, 에디터 안내 한 줄.
# 위 제자리에서 바꾼 것(같은 이름을 두 번 쓰지 않음):
# - METHOD_NAMES["grow"], DASH_TILE_TITLES["brainsize"](「보는 재미」 쪽 타일 id는 C 묶음이 app/lab/dashboard.py에 더한다).
# - HELP_GLOSSARY_TERMS에 「뉴런」(설계는 사전 항목으로 적었지만 지금은 낱말 묶음 + 「낱말 풀이」 쪽 줄이라 둘 다 더함). 설계 글
#   「-1..1로 눌러」를 「보통 -1..1로 눌러」로: 은닉층을 ReLU로 고를 수 있다(에디터 「뇌 설정」).
# - HELP_PAGES 「실험실」 쪽 끝 「자라는 뇌」 절, 「그래프 읽기」 쪽 「뇌 크기」 절(「뇌 들여다보기」 절 바로 뒤, 타일 순서와 같게).
#   설계 글에서 자리 바로 뒤의 조사(「…」에서·를·마다·와·의)를 줄·항목·칸·타일·탭·방식으로 바꿨다(v3 D19: 자리 뒤에 조사를 붙이지
#   않음. 「크기 고정」와처럼 받침과 어긋남). 「처음 뉴런 수」 칸과 「CMA-ES일 때만 보임」(설계 D32·D36)을 한 문장씩 더했다.
# 설계 글과 다른 것: TRAIN_GROW_NO_IPOP의 「정체하면 개체 수를 늘려 재시작」은 지금 화면에 없는 이름이라 실험실 칸(LAB_IPOP)·
# 명령줄(--ipop)과 같은 말로 바꿨다. FRAME_*는 policy/frame.py의 FrameError 글(6절 밖, 0'단계가 함께 더함).
# HELP_TIPS 세 키(lab.brain·lab.grow_start·lab.grow_every)의 연결표 키는 C 묶음이 app/help/anchors.py에 더한다(그 전까지
# test_v3_strings·test_help_anchors의 「HELP_TIPS 키 ⊆ 연결표」와 타일 id 검사가 실패한다).
# ---------------------------------------------------------------------------
# ---- 실험실 「뇌」 (app/lab/view.py) ----
LAB_BRAIN: Final = "뇌"
LAB_BRAIN_FIXED: Final = "크기 고정"
LAB_BRAIN_GROW: Final = "작게 시작해 자라기"
LAB_BRAIN_TIP: Final = (
    "크기 고정: 에디터 「뇌 설정」에서 정한 크기의 뇌로 처음부터 학습합니다.\n"
    "작게 시작해 자라기: 은닉층마다 뉴런 몇 개로 시작해서, 정해진 세대마다 뉴런을 두 배로 늘립니다. 새 뉴런은 출력 쪽 연결이 0으로 "
    "태어나므로 늘어나는 순간에는 지금까지 배운 행동이 그대로입니다. 「뇌 설정」에서 정한 크기까지 자랍니다. 자라는 모습을 보는 "
    "방법이며, 크기 고정보다 더 잘 배운다는 뜻은 아닙니다.")
LAB_BRAIN_GROW_NO_HIDDEN: Final = "이 환경의 뇌에는 은닉층이 없어서 자랄 수 없습니다. 에디터 「뇌 설정」에서 은닉층을 더하면 고를 수 있습니다."
LAB_BRAIN_GROW_NO_BRAIN: Final = "이 환경은 뇌 없이 몸만 진화해서 고를 것이 없습니다."
LAB_GROW_START: Final = "처음 뉴런 수"
LAB_GROW_START_TIP: Final = "은닉층마다 처음에 켜 둘 뉴런 수입니다. 「뇌 설정」의 은닉층 크기보다 클 수 없습니다."
LAB_GROW_EVERY: Final = "자라는 간격(세대)"
LAB_GROW_EVERY_TIP: Final = "이만큼 세대가 지날 때마다 은닉층마다 뉴런을 두 배로 늘립니다. 「뇌 설정」에서 정한 크기에 닿으면 멈춥니다."
LAB_GROW_HINT: Final = "{path}개, {gen}세대쯤에 다 자랍니다."  # {path}: 「2 → 4 → 8 → 16」(은닉층 뉴런 합, growth_schedule)
LAB_GROW_HINT_ALREADY: Final = "처음부터 다 자란 크기라 자라지 않습니다."
LAB_GROW_HINT_HONEST: Final = "같은 학습 시험 수에서 크기 고정보다 더 잘 배운다는 뜻은 아닙니다. 뇌가 자라는 모습을 볼 때 고르세요."
LAB_STATUS_GROWN: Final = "{gen}세대: 뇌가 자랐습니다 — 뉴런 {old} → {new}개, 연결 {old_links:,} → {new_links:,}개"

# ---- 학습기·명령줄 (engine/trainer.py, cli.py, engine/commands.py) ----
TRAIN_GROW_NO_HIDDEN: Final = ("이 환경의 뇌에는 은닉층이 없어서 「자라는 뇌」로 학습할 수 없습니다. 크기 고정으로 학습하거나 "
                               "에디터 「뇌 설정」에서 은닉층을 더하세요.")
TRAIN_GROW_NO_IPOP: Final = ("자라는 뇌는 IPOP(pycma 종료 조건이 차면 개체 수를 2배로 늘려 다시 시작)과 함께 쓸 수 없습니다. "
                             "둘 중 하나만 고르세요.")
TRAIN_GROW_SUMMARY: Final = "자라는 뇌: 뉴런 {start}개 → {full}개, {every}세대마다 두 배"
CLI_HELP_GROW_START: Final = "자라는 뇌(--method grow)의 은닉층마다 처음 뉴런 수(기본 2)"
CLI_HELP_GROW_EVERY: Final = "자라는 뇌가 뉴런을 두 배로 늘리는 세대 간격(기본 10)"
CLI_TRAIN_GROWN: Final = "{gen}세대: 뇌가 자랐습니다(뉴런 {old} → {new}개)"

# ---- 대시보드 (app/lab/dashboard.py, grow_tile.py, brain_view.py) ----
DASH_MARK_GROW: Final = "{gen}세대 끝: 뇌가 자람(뉴런 {old} → {new}개)"
DASH_TREND_GROW: Final = "뇌가 자란 세대"
GROW_TILE_NEURONS: Final = "은닉 뉴런 수"
GROW_TILE_LINKS: Final = "연결 수"
GROW_TILE_EVENT: Final = "자란 세대"
GROW_TILE_AXIS_NEURONS: Final = "뉴런"
GROW_TILE_AXIS_LINKS: Final = "연결"
GROW_TILE_FIXED: Final = ("이 실행의 뇌는 크기가 정해져 있습니다: 뉴런 {neurons}개, 연결 {links:,}개.\n"
                          "실험실에서 학습 방법을 CMA-ES로 두고 「뇌」 줄의 「작게 시작해 자라기」를 고르면 뇌가 자라는 모습을 볼 수 "
                          "있습니다.")  # v6.1 고침: 「뇌」 줄은 CMA-ES일 때만 보인다(GA·담금질 실행에서도 따라 할 수 있게)
GROW_TILE_TIP: Final = "{x}: 뉴런 {neurons}/{total}개, 연결 {links:,}개"
BRAIN_SIZE_LABEL: Final = " · 뉴런 {n}/{total} · 연결 {links:,}"
BRAIN_GROW_NOTE: Final = "옅은 점선 원은 아직 생기지 않은 뉴런 자리입니다. 정해진 세대마다 두 배로 채워집니다."
BRAIN_NODE_NEW: Final = "최근 {gen}세대 안에 새로 생긴 뉴런"
BRAIN_NODE_EMPTY: Final = "아직 생기지 않은 뉴런 자리"

# ---- 자체 점검 (selftest.py) ----
SELFTEST_CHECK_GROW: Final = "자라는 뇌 (자라기 직전·직후 뇌가 같은 결과를 내는지, 짧은 학습 두 번이 같은지)"
SELFTEST_GROW_SAME: Final = "자라는 뇌가 자라기 전과 같은 결과를 내고, 두 번 돌린 학습이 같습니다."
SELFTEST_GROW_DIFF: Final = "자라는 뇌 점검이 실패했습니다: {detail}"

# ---- 에디터 「뇌 설정」 탭 (app/editor/tabs.py) ----
EDP_BRAIN_GROW_NOTE: Final = ("실험실에서 「작게 시작해 자라기」를 고르면 이 크기가 뇌가 다 자란 크기입니다. "
                              "크게 잡아도 처음에는 작게 시작합니다.")

# ---- 뇌 틀 (policy/frame.py의 FrameError, 6절 밖) ----
FRAME_NO_BRAIN: Final = "이 환경은 뇌 없이 몸만 진화해서 자라는 뇌의 틀을 만들 수 없습니다."
FRAME_NO_HIDDEN: Final = "이 환경의 뇌에는 은닉층이 없어서(입력이 바로 출력으로 이어짐) 자라는 뇌의 틀을 만들 수 없습니다."
FRAME_BAD_COUNTS: Final = ("켜진 뉴런 수가 잘못되었습니다: {counts}. 은닉층 {layers}개마다 정수 하나씩, 1 이상 그 층 크기 이하여야 "
                           "합니다(은닉층 크기 {sizes}).")
FRAME_BAD_NEURON: Final = "뉴런 자리가 잘못되었습니다: 은닉층 {layer}의 {j}번째 뉴런(은닉층 크기 {sizes})."
FRAME_GENOME_SHAPE: Final = "유전자 모양이 {shape}인데 이 뇌 틀에는 숫자 {dim}개짜리 한 줄이 필요합니다."

# ---- 풍선 도움말 (app/help/tooltips.py). 연결표 키는 C 묶음이 app/help/anchors.py에 더한다 ----
HELP_TIPS["lab.brain"] = LAB_BRAIN_TIP
HELP_TIPS["lab.grow_start"] = LAB_GROW_START_TIP
HELP_TIPS["lab.grow_every"] = LAB_GROW_EVERY_TIP

# ---------------------------------------------------------------------------
# v6.1 L 묶음(학습: learners/cmaes.py, learners/grow.py, engine/trainer.py, engine/commands.py). 6절 밖에서 더 필요해진 글:
# 학습 방법 설정 검사(CMAESConfig.stds, GrowConfig)와 명령줄 이어 하기 거절 한 줄.
# ---------------------------------------------------------------------------
LEARNER_ERR_STDS: Final = "좌표별 처음 폭은 0보다 큰 유한한 숫자들이어야 합니다 (지금 {value})."
LEARNER_ERR_STDS_DIM: Final = "좌표별 처음 폭 {got}개가 유전자 차원({dim})과 맞지 않습니다."
LEARNER_ERR_GROW_START: Final = "자라는 뇌의 처음 뉴런 수는 1 이상의 정수여야 합니다 (지금 {value})."
LEARNER_ERR_GROW_EVERY: Final = "자라는 뇌가 자라는 간격은 1 이상의 정수(세대)여야 합니다 (지금 {value})."
LEARNER_ERR_GROW_INIT: Final = "새 뉴런의 들어오는 가중치 크기는 0보다 큰 유한한 숫자여야 합니다 (지금 {value})."
TRAIN_RESUME_GROW_FIXED: Final = ("이어 할 때는 자라는 뇌 설정(--grow-start, --grow-every)을 바꿀 수 없습니다. "
                                  "학습 방법의 일부라서, 바꾸려면 새 실행으로 시작하세요.")

# ---------------------------------------------------------------------------
# v6.1 C 묶음 (화면: 실험실 「뇌」 줄, 뇌 그림, 「뇌 크기」 타일, 점수 추이 세로선). 6절 밖에서 화면에 더 필요한 글.
# - DASH_TILE_HELP["brainsize"]: 「보는 재미」 타일 제목 풍선(v1.2 규칙: 「보는 재미」 타일마다 DASH_TILE_HELP, test_dashboard_v12).
# - DASH_MARK_LABELS["grow"]: 점수 추이 세로선 위 이름(다른 세로선 이름처럼 짧게). 풍선은 DASH_MARK_GROW, 범례는 DASH_TREND_GROW.
# - GROW_TILE_EMPTY: 「뇌 크기」 타일의 학습 전 빈 상태(실행이 자랄지 아직 모름: GROW_TILE_FIXED는 통계 줄이 생긴 뒤에만).
# ---------------------------------------------------------------------------
DASH_TILE_HELP["brainsize"] = ("세대마다 은닉 뉴런 수(계단, 왼쪽 눈금)와 연결 수(점선, 오른쪽 눈금)입니다. 세로 점선은 뇌가 "
                               "자란 세대입니다. 그래프를 누르면 그 세대를 봅니다. 실험실 「뇌」에서 「작게 시작해 자라기」를 "
                               "고른 실행에서만 자랍니다.")
DASH_MARK_LABELS["grow"] = "뇌 자람"
GROW_TILE_EMPTY: Final = "학습을 시작하면 세대마다 뇌의 뉴런 수와 연결 수를 그립니다."

# ---------------------------------------------------------------------------
# v6.1 고침 묶음 (검토·검증에서 확인된 결함 19건을 고치며 더 필요해진 글). 위 제자리에서 바꾼 것(같은 이름을 두 번 쓰지 않음):
# - HELP_PAGES 「실험실」 쪽 「자라는 뇌」 절 끝 문단: 옛 프로그램(v6.0 이하)이 자라는 실행을 만났을 때 일어나는 일 넷(열기·재생·판정은
#   됨, 이어 하기·재현은 거절, 복제는 크기 고정 실행으로 시작, 예전 체크포인트에서 이어 하면 분기 폴더만 남음).
# - GROW_TILE_FIXED: 「뇌」 줄이 CMA-ES일 때만 보인다는 말을 더함(GA·담금질 실행의 「뇌 크기」 타일에서도 따라 할 수 있게).
# ---------------------------------------------------------------------------
# 명령줄 train --method grow의 시작 줄: 처음 뉴런 수가 틀을 이미 채워 자라지 않을 때(TRAIN_GROW_SUMMARY의 「두 배」 대신)
TRAIN_GROW_SUMMARY_FULL: Final = "자라는 뇌: 뉴런 {full}개, 처음부터 다 자란 크기라 자라지 않습니다"
# 「뇌 들여다보기」 멈춘 그림: 그때 챔피언이 지금 보는 세대와 다른 크기의 뇌일 때(챔피언을 뽑은 뒤 뇌가 자랐거나 자동 재시작으로
# 작은 뇌에서 다시 시작) 보는 세대 대표를 그린다. BRAIN_TITLE_REP의 「그때 챔피언 없음」은 이 경우 사실이 아니라 따로 둔다.
BRAIN_TITLE_REP_SIZE: Final = "{gen}세대 대표(그때 챔피언은 {champion}세대, 뇌 크기가 다름) · 멈춘 그림"
# 「뇌 크기」 타일, 크기 고정 실행인데 은닉층이 없는 뇌: 「자라기」를 고를 수 없으므로 GROW_TILE_FIXED의 권유 대신 까닭을 적는다
GROW_TILE_FIXED_NO_HIDDEN: Final = ("이 실행의 뇌는 크기가 정해져 있습니다: 뉴런 {neurons}개, 연결 {links:,}개.\n"
                                    "이 환경의 뇌에는 은닉층이 없어서 자랄 수 없습니다. 에디터 「뇌 설정」에서 은닉층을 더하면 "
                                    "「작게 시작해 자라기」를 고를 수 있습니다.")
# 「뇌가 자람」 세로선 풍선: 자란 뒤 크기를 알 수 없을 때(다음 줄이 없고 통계 줄의 크기가 틀의 자람 규칙과 맞지 않음) 옛 크기만
DASH_MARK_GROW_FROM: Final = "{gen}세대 끝: 뇌가 자람(자라기 전 뉴런 {old}개)"

#!/bin/bash
# 측정 농장(.github/workflows/measure.yml)의 러너에서 도는 한 실행: 넘어짐 감점 w, 시드 s.
#   bash research/running_tune/run.sh <weight> <seed> [max_evaluations]
# 내장 달리기 프리셋에서 감점만 바꾼 파일로, 프리셋이 권하는 설정(예선 3 × 9,600)과 표준 예산(48,000회)으로 학습하고
# 판정 시드 100개로 판정한다. 결과는 $MEASURE_OUT (train.log, judge.log, 환경 파일, 실행 폴더: 체크포인트는 지움).
set -uo pipefail
w=$1; s=$2; maxev=${3:-48000}
out=${MEASURE_OUT:?}
env="$out/running_w${w#-}.evoenv"
uv run python research/running_tune/make_env.py --weight "$w" --out "$env" || exit 1
filter() { grep --line-buffered -v -e matplotlib -e _warnings; }
start=$(date +%s)
uv run evolab train "$env" --seed "$s" --qualify 3 --qualify-trials 9600 --max-evaluations "$maxev" --workers 3 \
  --out "$out/run" 2>&1 | filter > "$out/train.log"
rc=${PIPESTATUS[0]}
echo "# train exit $rc seconds $(( $(date +%s) - start ))" >> "$out/train.log"
tail -3 "$out/train.log"
[ "$rc" -eq 0 ] || exit "$rc"
run=$(find "$out/run" -mindepth 1 -maxdepth 1 -type d | sort | head -1)
uv run evolab judge "$run" --seeds 100 --workers 3 2>&1 | filter > "$out/judge.log"
jrc=${PIPESTATUS[0]}
cat "$out/judge.log"
rm -rf "$run/checkpoints"
ls -t "$run/champions" 2>/dev/null | tail -n +2 | while read -r f; do rm -f "$run/champions/$f"; done
exit "$jrc"

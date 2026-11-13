#!/bin/bash
# PX5 농장 한 작업(scratch 세계). 쓰기: research/sports_proto/PX5/run.sh <설정> <시드> [selfplay·solo에 더할 인자...]
#   설정: h16 | h0 | grow (숨바꼭질 두 역할, 학습 경기 합 60,000판 + 판정) · find (「찾기」: 술래만, 학습 시험 60,000회, 시험 6판, 뇌 h16)
#   결과: $MEASURE_OUT/run/ (log.jsonl, snaps.npz, ladder.npz, champion.npz, snap_elo.json, analysis.json 또는 찾기의 log·snaps·champion)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
setting=$1; seed=$2; shift 2
out=${MEASURE_OUT:-/tmp/px5_out}/run
W=${PX5_WORKERS:-3}
PY="uv run python"
[ -x "$here/../../../.venv/bin/python" ] && [ -z "${CI:-}" ] && PY="$here/../../../.venv/bin/python"
G=${PX5_GAMES:-60000}
common=(--seed "$seed" --games "$G" --pop 32 --K 6 --cap 16 --latest 0.5 --pool-admit rung
        --validate-every 1000 --validate-on role0 --rate-games 12 --rung-step 70 --elo-base 1000 --eval-games 20
        --baselines '{"0":["still","random"],"1":["chase","random"]}' --workers "$W" --out "$out")
case $setting in
  h16)  $PY "$here/selfplay.py" hide "${common[@]}" --hidden 16 "$@" ;;
  h0)   $PY "$here/selfplay.py" hide "${common[@]}" --hidden "$@" ;;
  grow) $PY "$here/selfplay.py" hide "${common[@]}" --hidden 16 --learner grow --grow-start 2 --grow-every 10 "$@" ;;
  find) exec $PY "$here/solo.py" find --seed "$seed" --trials "${PX5_TRIALS:-60000}" --T 6 --hidden 16 --workers "$W" --out "$out" "$@" ;;
  *) echo "unknown setting $setting" >&2; exit 2 ;;
esac
$PY "$here/analyze.py" "$out" --budget "$G" --workers "$W"

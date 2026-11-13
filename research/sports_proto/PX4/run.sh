#!/bin/bash
# PX4 농장 한 작업(scratch 세계). 쓰기: research/sports_proto/PX4/run.sh <설정> <시드> [추가 세계 옵션 JSON]
#   설정: h16 | h0 | grow (투수 대 타자, 학습 경기 100,000판 + 판정) · pp (투구 연습) · bp (타격 연습) (28,800회)
#   결과: $MEASURE_OUT/run/ (log.jsonl, snaps.npz, ladder.npz, champion.npz, snap_elo.json, analysis.json)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
setting=$1; seed=$2
opts=${3:-'{"ball_w":0.1,"hit_floor":0.1,"ball_loss":true}'}
out=${MEASURE_OUT:-/tmp/px4_out}/run
W=${PX4_WORKERS:-3}
PY="uv run python"
[ -x "$here/../../../.venv/bin/python" ] && [ -z "${CI:-}" ] && PY="$here/../../../.venv/bin/python"
common=(--opts "$opts" --seed "$seed" --games "${PX4_GAMES:-100000}" --pop 32 --K 6 --cap 16 --latest 0.5 --pool-admit rung
        --validate-every 1000 --validate-on role0 --rate-games 12 --rung-step 70 --elo-base 1000 --eval-games 20
        --baselines '{"0":["still","swing"],"1":["still","random"]}' --workers "$W" --out "$out")
case $setting in
  h16)  $PY "$here/selfplay.py" pitch_easy "${common[@]}" --hidden 16 ;;
  h0)   $PY "$here/selfplay.py" pitch_easy "${common[@]}" --hidden ;;
  grow) $PY "$here/selfplay.py" pitch_easy "${common[@]}" --hidden 16 --learner grow --grow-start 2 --grow-every 10 ;;
  pp)   exec $PY "$here/solo.py" pitch_practice --seed "$seed" --trials "${PX4_TRIALS:-28800}" --T 3 --hidden 16 --workers "$W" --out "$out" ;;
  bp)   exec $PY "$here/solo.py" bat_practice --seed "$seed" --trials "${PX4_TRIALS:-28800}" --T 3 --hidden 16 --workers "$W" --out "$out" ;;
  *) echo "unknown setting $setting" >&2; exit 2 ;;
esac
$PY "$here/analyze.py" "$out" --budget "${PX4_GAMES:-100000}" --workers "$W"

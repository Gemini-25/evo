#!/bin/bash
# PX6 농장 한 작업(scratch 세계): 라벨마다 리그전 하나. 쓰기: research/sports_proto/PX6/run.sh <라벨> [<라벨> ...]
#   결과: $MEASURE_OUT/<라벨>_rr.json
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${MEASURE_OUT:-/tmp/px6_out}
W=${PX6_WORKERS:-3}
PY="uv run python"
[ -x "$here/../../../.venv/bin/python" ] && [ -z "${CI:-}" ] && PY="$here/../../../.venv/bin/python"
for label in "$@"; do
  $PY "$here/rr.py" "$label" --games "${PX6_GAMES:-40}" --workers "$W" --out "$out"
done

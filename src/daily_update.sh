#!/bin/bash
# Daily refresh: pull latest games -> rebuild -> features -> retrain.
# Run by launchd every morning at 6am (see ~/Library/LaunchAgents/com.nbapredictor.daily.plist),
# or manually: bash src/daily_update.sh
set -u
cd /Users/adityakumar/orca/workspaces/nbapredictor/gorgonian
PY=/opt/homebrew/bin/python3
LOG=data/raw/daily_update.log
STATUS=data/processed/retrain_status.json

echo "{\"state\": \"running\", \"at\": \"$(date -u +%FT%TZ)\"}" > "$STATUS"
echo "===== daily update $(date) =====" >> "$LOG"
$PY -W ignore -m src.update_daily >> "$LOG" 2>&1 \
&& $PY -W ignore -m src.build_dataset >> "$LOG" 2>&1 \
&& $PY -W ignore -m src.features_rolling >> "$LOG" 2>&1 \
&& $PY -W ignore -m src.train >> "$LOG" 2>&1 \
&& $PY -W ignore -m src.snapshot >> "$LOG" 2>&1 \
&& echo "{\"state\": \"done\", \"at\": \"$(date -u +%FT%TZ)\"}" > "$STATUS" \
|| echo "{\"state\": \"failed\", \"at\": \"$(date -u +%FT%TZ)\"}" > "$STATUS"
echo "===== finished $(date) =====" >> "$LOG"

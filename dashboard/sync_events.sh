#!/usr/bin/env bash
# Pull detection events (WAV + mel.npy + JSON) from the Pi to the dashboard.
# Run periodically (or via watch / cron) on the Mac, then refresh the dashboard.
#
#   ./sync_events.sh            # one-off pull
#   watch -n 30 ./sync_events.sh   # pull every 30 s
set -euo pipefail

PI="${ALERTRECK_PI:-alertreck@alertreck.local}"
REMOTE="${ALERTRECK_REMOTE:-~/alertreck/data/evidence/}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOCAL="$HERE/events/"

mkdir -p "$LOCAL"
echo "Syncing $PI:$REMOTE  →  $LOCAL"
rsync -av --include='*/' \
  --include='*.json' --include='*.mel.npy' --include='*.wav' \
  --exclude='*' \
  "$PI:$REMOTE" "$LOCAL"
echo "Done. Refresh the dashboard."

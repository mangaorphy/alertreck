#!/usr/bin/env bash
# Resumable upload of data/ and dataset/ to the shared Drive folder.
# rclone copy is idempotent: already-uploaded files are skipped on re-run,
# so a dropped connection just resumes. The loop retries until it fully succeeds.

set -u
FOLDER_ID="1U9BwIUNQ8Snl5RxR8LHthWfdOc_EdcTM"
REMOTE="orphy:"
LOG="upload.log"

common=(
  --drive-root-folder-id="$FOLDER_ID"
  --progress                     # live progress: %, ETA, speed, transfers
  --transfers=16                 # many parallel uploads — big win for lots of small files
  --checkers=16                  # parallel existence/hash checks (skip-already-uploaded)
  --drive-chunk-size=128M        # bigger chunks = fewer round-trips on big files
  --drive-pacer-min-sleep=10ms   # default 100ms between API calls — 10x faster small-file throughput
  --drive-pacer-burst=200        # allow bursts of API calls before pacing kicks in
  --drive-upload-cutoff=256M     # files under this upload in one request (no resumable overhead)
  --low-level-retries=20         # retry within a single HTTP call on network blips
  --retries=10                   # retry whole passes
  --retries-sleep=15s
  --stats=10s
  --stats-one-line
  --log-file="$LOG"
  --log-level=INFO
)

for src in data dataset; do
  echo ">>> Uploading $src/ -> Drive:$src/"
  until rclone copy "$src" "${REMOTE}${src}" "${common[@]}"; do
    echo "!!! rclone exited non-zero (likely network). Resuming in 20s... ($(date))"
    sleep 20
  done
  echo ">>> $src/ fully uploaded and verified."
done

echo ">>> ALL DONE."

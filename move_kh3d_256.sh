#!/usr/bin/env bash
# Pull the outputs written so far by kh3d_256x1024x256 from the remote host's
# /scratch to the local /data HDD. Run this on the local machine:
#
#   ./move_kh3d_256.sh
#
# Nothing is deleted on the remote, so the live run is unaffected. Re-running
# it later copies only new or changed files.
#
# Files modified in the last $SETTLE_MIN minutes are skipped, since the run
# may still be writing them.

set -euo pipefail

REMOTE="sasi@10.42.75.224"
SRC="/scratch/sasi/simulation_outputs/kh3d_256x1024x256"
DEST_PARENT="/data/sasi/simulation_outputs"
DEST="$DEST_PARENT/$(basename "$SRC")"
SETTLE_MIN=5

command -v rsync >/dev/null || { echo "rsync is not installed locally" >&2; exit 1; }
[[ -d "$DEST_PARENT" ]] || { echo "Destination parent missing: $DEST_PARENT" >&2; exit 1; }

# List remote files that have not been touched for SETTLE_MIN minutes (paths relative to SRC).
filelist=$(mktemp)
trap 'rm -f "$filelist"' EXIT
ssh "$REMOTE" "cd '$SRC' && find . -type f -mmin +$SETTLE_MIN -printf '%P\n'" > "$filelist"
n_total=$(ssh "$REMOTE" "find '$SRC' -type f | wc -l")

n_files=$(wc -l < "$filelist")
echo "Copying $n_files files ($(( n_total - n_files )) skipped as recently modified)"

mkdir -p "$DEST"
rsync -a --info=progress2 --files-from="$filelist" "$REMOTE:$SRC/" "$DEST/"

echo "Done: $REMOTE:$SRC -> $DEST (source left intact)"

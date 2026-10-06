#!/bin/bash
# Run this on the DESTINATION machine, from the directory you want files copied into.

SRC_USER="sasi"
SRC_HOST="10.42.75.223"
SRC_DIR="/home/arnav/athenak/my_outputs/noSMR_2_3_cutoffISMcoolfn"
LIMIT_KB=$((20 * 1024 * 1024))   # 20 GB expressed in KiB (du -k units)

# Ask the source for sizes (in KiB) of each top-level subdirectory,
# keep only those below the limit, then rsync each one here.
ssh "${SRC_USER}@${SRC_HOST}" "du -k -d 1 '${SRC_DIR}'" \
  | awk -v lim="$LIMIT_KB" -v root="$SRC_DIR" '$1 < lim && $2 != root {print $2}' \
  | while read -r dir; do
      echo ">>> Copying ${dir}"
      rsync -avh --progress --partial "${SRC_USER}@${SRC_HOST}:${dir}" .
    done

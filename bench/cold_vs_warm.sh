#!/usr/bin/env bash
# Measure the task suite with the n-gram table cold vs warm in the page cache.
#
# "Cold" means the table is served from NVMe on almost every lookup. Dropping the
# page cache needs root; without it the script measures whatever state it finds and
# says so, which is still useful because the table does not warm on its own.
set -euo pipefail
# Port tracks run.sh's PORT default. Deliberately not ${HOST:-...}: HOST is commonly
# exported by shells and CI images, and inheriting it would silently retarget the run.
API="${API:-http://127.0.0.1:${PORT:-30000}}"
MODEL="${MODEL:?set MODEL to the path of shard 00001 of the GGUF}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"

drop_caches() {
  if [ "$(id -u)" = "0" ]; then sync; echo 3 > /proc/sys/vm/drop_caches; return 0; fi
  if command -v docker >/dev/null 2>&1; then
    docker run --rm --privileged alpine sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null && return 0
  fi
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null && return 0
  return 1
}

echo "== dropping page cache =="
if drop_caches; then echo "   dropped"; else echo "   could not drop (need root); measuring current state"; fi
"$PY" "$HERE/../tools/page_cache.py" "$MODEL" || true
"$PY" "$HERE/run_bench.py" --api "$API" --model "$MODEL" --label cold --skip-ctx

echo
echo "== warming the table =="
"$PY" "$HERE/../tools/warm_table.py" "$MODEL"
"$PY" "$HERE/run_bench.py" --api "$API" --model "$MODEL" --label warm --skip-ctx

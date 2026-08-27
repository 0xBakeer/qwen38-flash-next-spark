#!/usr/bin/env bash
# Qwen3.8-Flash-Next on a DGX Spark / GB10, with the 51B n-gram table left on NVMe.
#
#   ./run.sh setup    build llama.cpp (PR #27742 + patches) and fetch the model
#   ./run.sh serve    warm the n-gram table, then start llama-server
#   ./run.sh bench    quick decode/prefill measurement against a running server
#   ./run.sh all      setup, then serve
#
# Everything is configurable by environment variable; the defaults are what was measured.
set -euo pipefail

ROOT="${ROOT:-$HOME/.qwen38fn}"
SRC="${SRC:-$ROOT/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$ROOT/model}"
REPO="${REPO:-unsloth/Qwen3.8-Flash-Next-GGUF}"
QUANT="${QUANT:-UD-Q4_K_XL}"
PR="${PR:-27742}"
# The PR is active, so its tip moves. Pin a commit the patches in patches/ are known to
# apply to; PR_SHA=head tracks the branch tip instead (patches may then not apply).
PR_SHA="${PR_SHA:-035e227}"
CUDA_ARCH="${CUDA_ARCH:-121}"          # GB10 is SM 12.1; cmake promotes this to 121a
CTX="${CTX:-262144}"                   # KV is only ~24 KB/token, so full context is affordable
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
JOBS="${JOBS:-$(nproc)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

find_cuda() {
  if command -v nvcc >/dev/null 2>&1; then return; fi
  for d in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.*; do
    [ -x "$d/bin/nvcc" ] && { export PATH="$d/bin:$PATH"; export CUDA_HOME="$d"; return; }
  done
  die "nvcc not found. Install the CUDA toolkit or set PATH to include it."
}

preflight() {
  command -v git   >/dev/null || die "git not found"
  command -v cmake >/dev/null || die "cmake not found"
  command -v python3 >/dev/null || die "python3 not found"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found - is this a CUDA machine?"
  find_cuda

  local mem_gib
  mem_gib=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)
  log "detected ${mem_gib} GiB system memory, $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  [ "$mem_gib" -lt 100 ] && cat <<EOF
WARNING: this recipe was measured on a 128GB unified-memory box (~121 GiB usable).
         ${QUANT} needs roughly 77 GiB resident plus room for the page cache.
         With less memory, try a smaller quant from ${REPO}.
EOF

  # Swap must be off: we want cold n-gram pages evicted from the page cache,
  # never resident weights pushed into swap.
  if [ "$(awk 'NR==2{print $3}' /proc/swaps 2>/dev/null || echo 0)" != "" ] && [ -s /proc/swaps ] && [ "$(wc -l < /proc/swaps)" -gt 1 ]; then
    echo "NOTE: swap is enabled. Consider 'sudo swapoff -a' so file pages are evicted instead of anonymous ones."
  fi
}

setup() {
  preflight
  mkdir -p "$ROOT"

  if [ ! -d "$SRC/.git" ]; then
    log "cloning llama.cpp"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$SRC"
  fi

  log "fetching PR #${PR} (qwen4exp support - not yet merged upstream)"
  git -C "$SRC" fetch --depth 50 origin "pull/${PR}/head:qwen4exp" --force 2>/dev/null || true
  if [ "$PR_SHA" = "head" ]; then
    git -C "$SRC" checkout -q qwen4exp
  else
    git -C "$SRC" checkout -q "$PR_SHA" 2>/dev/null || {
      echo "NOTE: pinned commit $PR_SHA not reachable; falling back to the branch tip."
      echo "      The patches may not apply - check the log below."
      git -C "$SRC" checkout -q qwen4exp
    }
  fi
  log "at $(git -C "$SRC" log --oneline -1)"

  local applied=0 skipped=0
  for p in "$HERE"/patches/*.patch; do
    [ -e "$p" ] || continue
    if git -C "$SRC" apply --check "$p" 2>/dev/null; then
      log "applying $(basename "$p")"
      git -C "$SRC" apply "$p"; applied=$((applied+1))
    elif git -C "$SRC" apply --reverse --check "$p" 2>/dev/null; then
      log "$(basename "$p") already applied"; applied=$((applied+1))
    else
      echo "WARNING: $(basename "$p") does not apply to this checkout - building without it."
      echo "         See docs/sources.md; try PR_SHA=035e227 for the pinned commit."
      skipped=$((skipped+1))
    fi
  done
  [ "$skipped" -gt 0 ] && echo "WARNING: ${skipped} patch(es) skipped, ${applied} applied."

  log "building for SM ${CUDA_ARCH} with ${JOBS} jobs (this takes a while)"
  cmake -S "$SRC" -B "$SRC/build" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$SRC/build" -j"$JOBS" --target llama-server llama-cli llama-quantize

  log "sanity check: qwen4exp graph vs CPU reference"
  "$SRC/build/bin/test-llama-archs" -a qwen4exp 2>/dev/null | grep -E "qwen4exp" || \
    echo "(test-llama-archs not built; skipping)"

  if ! ls "$MODEL_DIR/$QUANT"/*00001-of-*.gguf >/dev/null 2>&1; then
    log "downloading ${REPO} ${QUANT} (~104 GiB)"
    python3 -m pip install -q --upgrade huggingface_hub >/dev/null 2>&1 || true
    HF_XET_HIGH_PERFORMANCE=1 hf download "$REPO" --include "${QUANT}/*" --local-dir "$MODEL_DIR"
  fi
  log "setup complete"
}

model_path() {
  local m
  m=$(ls "$MODEL_DIR/$QUANT"/*00001-of-*.gguf 2>/dev/null | head -1) \
    || die "model not found under $MODEL_DIR/$QUANT - run './run.sh setup' first"
  [ -n "$m" ] || die "model not found under $MODEL_DIR/$QUANT - run './run.sh setup' first"
  echo "$m"
}

serve() {
  find_cuda
  local M; M=$(model_path)

  # The n-gram table does not warm on its own: 320M rows addressed by a 3-gram hash
  # means a short workload almost never touches a row twice. One sequential read
  # cuts major faults ~6x. Skip with WARM=0.
  if [ "${WARM:-1}" = "1" ]; then
    log "warming the n-gram table (one sequential read, ~30 s)"
    python3 "$HERE/tools/warm_table.py" "$M" || echo "(warm failed; continuing)"
  fi

  # Quoting "${EXTRA_ARGS:-}" would pass an empty argument, which llama-server rejects.
  local extra=()
  [ -n "${EXTRA_ARGS:-}" ] && read -r -a extra <<< "$EXTRA_ARGS"

  # ngram-mod drafts spans from context repetition, so it needs no draft model and
  # costs no memory. It is a large win on copy-heavy work (editing a file you were
  # given) and a no-op on free-form prose. Speculation is exact: the target verifies
  # every token, so output is unchanged. Disable with SPEC=none.
  local spec=()
  [ "${SPEC:-ngram-mod}" != "none" ] && spec=(--spec-type "${SPEC:-ngram-mod}")

  log "starting llama-server on ${HOST}:${PORT}, ctx ${CTX}, spec ${SPEC:-ngram-mod}"
  # -ot per_layer_token_embd=CPU  : keep the 51B n-gram table off the GPU
  # -lm mmap                      : serve that table from NVMe via the page cache
  # --parallel 1                  : one slot. Concurrent requests queue rather than crash,
  #                                 but they do not batch - see the recipe README
  # Without an alias, llama-server reports the full GGUF path as the model id, which
  # leaks the filesystem layout to every API client and reads badly in model pickers.
  exec "$SRC/build/bin/llama-server" \
    -m "$M" \
    --alias "${ALIAS:-qwen3.8-flash-next}" \
    -lm mmap \
    -ot "per_layer_token_embd=CPU" \
    --n-gpu-layers 999 \
    --ctx-size "$CTX" \
    --parallel 1 \
    "${spec[@]}" \
    --temp 1.0 --top-p 0.95 --top-k 20 \
    --host "$HOST" --port "$PORT" \
    "${extra[@]}"
}

bench() {
  local api="http://${HOST}:${PORT}"
  curl -sf -m 5 "$api/health" >/dev/null || die "no server on $api - start it with './run.sh serve'"
  log "measuring decode (server-reported timings, prefill excluded)"
  python3 - "$api" <<'PY'
import json, sys, urllib.request
api = sys.argv[1]
prompt = "Write a detailed explanation of consistent hashing with virtual nodes."
for i in range(3):
    body = json.dumps({"messages":[{"role":"user","content":prompt}],
                       "max_tokens":300,"temperature":1.0,"top_p":0.95,"top_k":20}).encode()
    req = urllib.request.Request(api + "/v1/chat/completions", body,
                                 {"Content-Type":"application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=900))
    t = d.get("timings") or {}
    print("  run{}: decode {:6.2f} tok/s   prefill {:7.1f} tok/s".format(
        i+1, t.get("predicted_per_second", 0), t.get("prompt_per_second", 0)))
PY
}

case "${1:-all}" in
  setup) setup ;;
  serve) serve ;;
  bench) bench ;;
  all)   setup; serve ;;
  *)     sed -n '2,9p' "$0"; exit 1 ;;
esac

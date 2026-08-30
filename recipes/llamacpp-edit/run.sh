#!/usr/bin/env bash
# Qwen3.8-Flash-Next on a DGX Spark / GB10, with the 51B n-gram table left on NVMe.
#
#   ./run.sh setup    build llama.cpp and fetch the model
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
# PR #27742 (qwen4exp support) MERGED upstream on 2026-08-27, and both patches in patches/
# are now redundant: upstream implemented can_reuse() for the qwen4exp graph inputs itself,
# and rewrote the quantizer to stage rows in slabs bounded by max_buf_size, which is what
# rowband-ple-quant did. Neither patch applies to master any more, and neither needs to.
#
#   REF=pinned   (default) the exact PR commit every number in this repo was measured on,
#                with the patches applied. Reproduces the published measurements.
#   REF=master   upstream master, no patches. What you want for new work.
#   REF=<sha>    any commit; patches are attempted and skipped if they do not apply.
#
# The default is still the pin because that is what the measurements in docs/ were taken
# on, and a recipe whose numbers you cannot reproduce is not much of a recipe.
REF="${REF:-pinned}"
PR_SHA="${PR_SHA:-035e227}"
# Vision. The GGUF shards carry no vision tensors - llama.cpp ships multimodal as a separate
# projector - so image input needs an mmproj file. Unsloth publishes one in the same repo as
# the quants, so this is a 0.9 GiB extra download, not a hunt through a third repository.
# MMPROJ=none turns it off; MMPROJ=<path> points at your own.
MMPROJ="${MMPROJ:-auto}"
MMPROJ_FILE="${MMPROJ_FILE:-mmproj-F16.gguf}"
CUDA_ARCH="${CUDA_ARCH:-121}"          # GB10 is SM 12.1; cmake promotes this to 121a
CTX="${CTX:-262144}"                   # KV is only ~24 KB/token, so full context is affordable
# Slots. llama.cpp divides CTX by PARALLEL and gives each slot a private, fixed share, so this
# is not vLLM's --max-num-seqs: a slot count is a standing claim on the context whether or not
# anyone is using it. 2 costs nothing at one caller (35.02 tok/s inside the 34.12-37.43 spread
# of three one-slot repeats) and is worth 1.24-1.30x under load, leaving 131,072 tokens per
# request - four times the largest context ever measured here. See ../../docs/parallel.md.
PARALLEL="${PARALLEL:-2}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
JOBS="${JOBS:-$(nproc)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The version of the recipes, not of llama.cpp. A default changing here moves your numbers, so
# a run that records this can be placed against the right set of published figures.
RECIPE_VERSION="$(cat "$HERE/../../VERSION" 2>/dev/null || echo unknown)"

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
  # Test the contents, not the size: procfs files always stat as 0 bytes, so the
  # `[ -s /proc/swaps ]` guard this used to carry meant the NOTE could never print.
  if [ "$(awk 'NR>1' /proc/swaps 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "NOTE: swap is enabled ($(awk 'NR>1{u+=$4} END{printf "%.1f", u/1048576}' /proc/swaps) GiB in use)."
    echo "      Consider 'sudo swapoff -a' so file pages are evicted instead of anonymous ones."
  fi
}

setup() {
  preflight
  mkdir -p "$ROOT"

  if [ ! -d "$SRC/.git" ]; then
    log "cloning llama.cpp"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$SRC"
  fi

  local want="$REF"
  [ "$want" = "pinned" ] && want="$PR_SHA"

  if [ "$want" = "master" ]; then
    # qwen4exp is in master since 2026-08-27, so there is no PR to fetch.
    log "fetching upstream master (qwen4exp merged 2026-08-27 via PR #${PR})"
    git -C "$SRC" fetch --depth 50 origin master --force
    git -C "$SRC" checkout -q FETCH_HEAD
  else
    log "fetching PR #${PR} for the pinned commit ${want}"
    git -C "$SRC" fetch --depth 50 origin "pull/${PR}/head:qwen4exp" --force 2>/dev/null || true
    git -C "$SRC" checkout -q "$want" 2>/dev/null || {
      echo "NOTE: commit $want not reachable; falling back to the merged branch tip."
      git -C "$SRC" fetch --depth 50 origin master --force
      git -C "$SRC" checkout -q FETCH_HEAD
      want=master
    }
  fi
  log "at $(git -C "$SRC" log --oneline -1)"

  # Both patches are upstream as of master. Attempting them there produces two scary
  # "does not apply" warnings for work that is already in the tree, so do not attempt them.
  if [ "$want" = "master" ]; then
    log "skipping patches/: both are upstream in master (can_reuse for the qwen4exp graph"
    log "  inputs, and slab-bounded staging in the quantizer). Nothing to apply."
  else
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
        echo "         If this is master or near it, that is expected: both patches are"
        echo "         upstream. Use REF=master to skip them cleanly, or REF=pinned to"
        echo "         reproduce the measurements in docs/. See docs/sources.md."
        skipped=$((skipped+1))
      fi
    done
    if [ "$skipped" -gt 0 ]; then
      echo "WARNING: ${skipped} patch(es) skipped, ${applied} applied."
    fi
  fi

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

  if [ "$MMPROJ" != "none" ] && [ ! -f "$MODEL_DIR/$MMPROJ_FILE" ] && [ ! -f "$MMPROJ" ]; then
    log "downloading ${MMPROJ_FILE} (~0.9 GiB, gives the GGUF image input)"
    HF_XET_HIGH_PERFORMANCE=1 hf download "$REPO" --include "$MMPROJ_FILE" --local-dir "$MODEL_DIR" \
      || echo "WARNING: mmproj download failed - the server will still start, text-only."
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

# Resolves to a projector path, or to nothing at all. Nothing is a valid answer: the server
# starts text-only, which is what every measurement before 2026-08-30 was taken on.
mmproj_path() {
  case "$MMPROJ" in
    none) return ;;
    auto) [ -f "$MODEL_DIR/$MMPROJ_FILE" ] && echo "$MODEL_DIR/$MMPROJ_FILE" ;;
    *)    [ -f "$MMPROJ" ] && echo "$MMPROJ" || die "MMPROJ=$MMPROJ does not exist" ;;
  esac
}

serve() {
  find_cuda
  # serve() does not run preflight(), and the deferred warm below polls /health.
  [ "${WARM:-0}" = "1" ] && { command -v curl >/dev/null || die "curl not found (or set WARM=0)"; }
  local M; M=$(model_path)

  # Deferred until the server is serving, and OFF by default. Two separate findings.
  #
  # Ordering: warming before the load is wasted. Loading the model streams the whole GGUF
  # through a 121 GiB box and evicts the table region as it goes - measured on a GB10 as 100%
  # cached before, 1.9% by the time the server answered /health, and independently as 18%
  # established before startup reading back 0.06% afterwards. Backgrounding it keeps the exec
  # below, so this script is still replaced by llama-server and signal handling is unchanged.
  # exec preserves the PID, so $$ is llama-server's once it takes over; without the kill -0
  # check the warmer would outlive a server that failed to start, poll out its whole budget and
  # then read 26.8 GiB for nothing. WARM_WAIT counts attempts, each ~1 s (connection refused)
  # to ~3 s (packets dropped).
  #
  # Default: 0. Fixing the ordering makes the warm actually happen, and it still is not worth
  # doing. From a genuine cold start (0.06% resident) the warmer reads all 26.8 GiB at
  # 1.01 GiB/s and reaches only 25.9% - the page cache has nowhere to put the rest once the
  # model load has taken its share. Decode at 25.88% is 34.12 tok/s against 34.99 and 37.43 for
  # two runs at 0.06%, so the warm result sits inside the spread of the cold ones, which differ
  # from each other by 6.5%. WARM=1 still works, and is now correct when you ask for it.
  if [ "${WARM:-0}" = "1" ]; then
    local self=$$
    (
      for _ in $(seq 1 "${WARM_WAIT:-600}"); do
        kill -0 "$self" 2>/dev/null || exit 0
        curl -sf -m 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1 && break
        sleep 1
      done
      kill -0 "$self" 2>/dev/null || exit 0
      log "warming the n-gram table (one sequential read, ~27 s, reaches ~26%)"
      python3 "$HERE/tools/warm_table.py" "$M" || echo "(warm failed; continuing)"
    ) &
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

  local mm=() mmp
  mmp=$(mmproj_path)
  if [ -n "$mmp" ]; then
    mm=(--mmproj "$mmp")
    log "vision on: $(basename "$mmp")"
  else
    log "vision off (no projector; set MMPROJ or re-run setup)"
  fi

  log "recipe v${RECIPE_VERSION} — starting llama-server on ${HOST}:${PORT}, ctx ${CTX}, ${PARALLEL} slot(s) of $((CTX / PARALLEL)) tokens, spec ${SPEC:-ngram-mod}"
  # -ot per_layer_token_embd=CPU  : keep the 51B n-gram table off the GPU
  # -lm mmap                      : serve that table from NVMe via the page cache
  # --mmproj                      : the vision projector, when one was fetched. ~0.9 GiB of
  #                                 extra resident weights; the model scores the same on the
  #                                 atlas image eval as the NVFP4 checkpoint does - see
  #                                 ../../docs/vision.md
  # --parallel                    : slots. CTX is divided across them, so 2 means 131,072
  #                                 tokens per request. Raising it buys concurrency and spends
  #                                 context; 64 is not a configuration - see docs/parallel.md
  # Without an alias, llama-server reports the full GGUF path as the model id, which
  # leaks the filesystem layout to every API client and reads badly in model pickers.
  exec "$SRC/build/bin/llama-server" \
    -m "$M" \
    --alias "${ALIAS:-qwen3.8-flash-next}" \
    -lm mmap \
    -ot "per_layer_token_embd=CPU" \
    --n-gpu-layers 999 \
    --ctx-size "$CTX" \
    --parallel "$PARALLEL" \
    "${spec[@]}" \
    "${mm[@]}" \
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

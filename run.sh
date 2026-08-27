#!/usr/bin/env bash
# Qwen3.8-Flash-Next on a DGX Spark — recipe dispatcher.
#
#   ./run.sh edit    setup|serve|bench|all    llama.cpp + context-copying speculation
#   ./run.sh longctx setup|serve|bench        vLLM + NVFP4 + the model's MTP draft head
#
# Which one you want depends on the work, not on which is "faster" — see README.md.
# Short version: coding agents that rewrite files -> edit. Chat and long documents -> longctx.
#
# Backwards compatibility: `./run.sh setup|serve|bench|all` with no recipe named still runs
# the edit recipe, which is what this script used to do on its own.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Qwen3.8-Flash-Next on a DGX Spark

  ./run.sh edit    setup     build llama.cpp (PR #27742 + patches) and fetch the GGUF (~105 GB)
  ./run.sh edit    serve     start it on :8000
  ./run.sh edit    bench     measure decode across four task shapes
  ./run.sh edit    all       setup, then serve

  ./run.sh longctx setup     build the vLLM container and fetch the NVFP4 checkpoint (~126 GB)
  ./run.sh longctx serve     start it on :8000
  ./run.sh longctx bench     measure decode, prefill, and behaviour at depth

Both serve an OpenAI-compatible API on http://localhost:8000/v1.
Only one can run at a time — they share the GPU.

Which one?
  Rewriting whole files under a coding agent   -> edit     (88 tok/s on that shape)
  Chat, reasoning, long documents              -> longctx  (flat ~32 tok/s, 4x faster prefill)
  Undecided                                    -> longctx

Every flag is overridable by environment variable; see the recipe READMEs.
EOF
}

case "${1:-}" in
  edit)     shift; exec "$HERE/recipes/llamacpp-edit/run.sh" "${@:-serve}" ;;
  longctx)
    shift
    case "${1:-serve}" in
      setup) exec "$HERE/recipes/vllm-longctx/setup.sh" ;;
      serve) exec "$HERE/recipes/vllm-longctx/serve.sh" ;;
      bench) shift; exec "$HERE/bench/portable_bench.py" --api "http://127.0.0.1:${PORT:-8000}" \
                     --label "${LABEL:-vllm-longctx}" "$@" ;;
      *) echo "unknown longctx command: $1" >&2; usage; exit 2 ;;
    esac
    ;;
  setup|serve|bench|all)
    # legacy form, no recipe named
    exec "$HERE/recipes/llamacpp-edit/run.sh" "$@"
    ;;
  ""|-h|--help|help) usage ;;
  *) echo "unknown recipe: $1" >&2; echo; usage; exit 2 ;;
esac

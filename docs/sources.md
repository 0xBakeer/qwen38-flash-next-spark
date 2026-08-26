# Sources

Everything this recipe depends on, and where the claims in the docs come from.

## Model and weights

| What | Where |
|---|---|
| Qwen3.8-Flash-Next (BF16, original) | https://huggingface.co/Qwen/Qwen3.8-Flash-Next |
| GGUF quants used here (UD-Q4_K_XL and the rest of the ladder) | https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF |
| Official FP8 checkpoint | https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8 |
| NVFP4 checkpoint (for SGLang/vLLM; 135.3 GB, does **not** fit a single 128 GB Spark) | https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4 |
| Qwen3.5-0.8B, used to test an external draft model (vocab-compatible at 248320) | https://huggingface.co/Qwen/Qwen3.5-0.8B |

## Runtime

| What | Where |
|---|---|
| llama.cpp | https://github.com/ggml-org/llama.cpp |
| PR #27742 — `qwen4exp` support. **Required; not merged at time of writing.** | https://github.com/ggml-org/llama.cpp/pull/27742 |
| Upstream feature request that preceded it | https://github.com/ggml-org/llama.cpp/issues/27741 |

`run.sh` pins the PR to a known-good commit (`PR_SHA`) so the patches in `patches/` apply
cleanly. Set `PR_SHA=head` to track the branch tip instead, at the cost of the patches
possibly no longer applying.

## Where specific claims come from

The architectural claim that sparsely-accessed embedding tables can live in off-accelerator
storage is made by Qwen themselves in the Qwen3.8-Flash-Next tech report, in the context of
scaling the n-gram vocabulary from 20V to 200V:

> Because embedding tables are sparsely accessed and deterministically addressed, they can be
> scaled with negligible additional per-token computation and stored in off-accelerator storage.

Repository: https://github.com/QwenLM/Qwen3.8-Flash-Next

Issues documented in [the README](../README.md) were reported upstream by others and
independently reproduced here:

| Claim | Reported at |
|---|---|
| Quantizing `per_layer_token_embd` dies with `std::bad_alloc` (the `f32_conv_buf` staging path) | https://github.com/ggml-org/llama.cpp/pull/27742#issuecomment-5427556414 |
| Quantized KV cache aborts on this architecture | https://github.com/ggml-org/llama.cpp/pull/27742#issuecomment-5427655584 |
| The n-gram table is 51B params *on top of* the ~126B compute params | https://github.com/ggml-org/llama.cpp/pull/27742#issuecomment-5426608276 |
| PR runs the IQ1_S quant | https://github.com/ggml-org/llama.cpp/pull/27742#issuecomment-5426517862 |

The `ngram-mod` speculation result in this repo was prompted by a Metal report of the same
technique working well on this architecture (79% acceptance, 2.4x on a copy-heavy edit). That
report also documents an Apple-specific problem this recipe does not have — on Metal the PLE
tensor shares mmap'd regions with GPU-assigned tensors, so it gets wired for GPU access and the
offload does not happen:

https://github.com/ggml-org/llama.cpp/pull/27742#issuecomment-5429988128

## Reference points from other hardware

Useful for calibrating whether your numbers are reasonable. Not measured by this repo.

| Machine | Quant | Decode | Prefill | Source |
|---|---|---|---|---|
| M5 Max, 128 GB, macOS 26 | UD-Q4_K_XL | 34-36 tok/s short, 29 at 20k | 614 tok/s @20k | the Metal report linked above |
| DGX Spark, GB10 | UD-Q4_K_XL | ~22 tok/s | 355-662 tok/s | [docs/benchmarks.md](benchmarks.md) |

The M5 Max has roughly twice the memory bandwidth of a GB10 (~546 vs ~273 GB/s) and gets
roughly 1.6x the decode throughput, which is consistent with decode being bandwidth-bound.

## A note on benchmarks in the PR thread

The llama.cpp maintainers have asked that benchmark results not be posted in PR #27742, which
is reserved for correctness and architectural review. If you reproduce or contradict the
numbers here, open an issue on **this** repo rather than commenting there.

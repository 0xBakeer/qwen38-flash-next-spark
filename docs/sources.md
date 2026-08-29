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
| PR #27742 — `qwen4exp` support. **Merged 2026-08-27** (65 commits, merge commit `6c84c7d`). | https://github.com/ggml-org/llama.cpp/pull/27742 |
| Upstream feature request that preceded it | https://github.com/ggml-org/llama.cpp/issues/27741 |

### The PR merged, and both patches went with it

qwen4exp is in llama.cpp master as of 2026-08-27. Neither patch in `patches/` applies to
master any more, and neither is needed there:

- **`canreuse-qwen4exp.patch`** — master implements `can_reuse()` on both qwen4exp graph
  inputs itself (`llm_graph_input_qsa` and `llm_graph_input_ple` in
  `src/models/qwen4exp.cpp`), which is the same mechanism the patch added.
- **`rowband-ple-quant.patch`** — master rewrote the quantizer loop to process rows in slabs
  bounded by `max_buf_size` (`src/llama-quant.cpp`, "process the rows in slabs"), which
  removes the whole-tensor f32 staging buffer the patch existed to avoid. Both fixes are
  the same shape; upstream's is more general, since it bounds every tensor rather than the
  PLE table specifically.

Checked by `git apply --check` against master at `c841aee` on 2026-08-30: both patches fail
to apply, and the code they would have added is present.

`REF` selects what gets built:

| `REF` | What you get |
|---|---|
| `pinned` (default) | PR commit `035e227` plus the patches — the exact build every number in this repo was measured on |
| `master` | upstream master, no patches — what you want for new work |
| any sha | that commit; patches are attempted and skipped if they do not apply |

The default is still the pin, because a recipe whose published numbers you cannot reproduce
is not much of a recipe. Re-measuring on master is tracked separately.

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

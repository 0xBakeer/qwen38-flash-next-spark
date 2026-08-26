# Qwen3.8-Flash-Next on a DGX Spark

A reproducible recipe for running **Qwen3.8-Flash-Next** — a 180B-parameter model — on a single
NVIDIA DGX Spark (GB10, 128 GB unified memory) with llama.cpp, at Q4 quantization and the full
262,144-token context.

The trick that makes it fit: 51.2B of the model's 180B parameters are a single n-gram embedding
table that is never multiplied, only looked up. It can live on NVMe instead of in memory.

To be clear about what this is and isn't: a single Spark decodes at **~22 tok/s** with this setup.
That is not fast in absolute terms. The interesting result is that a 180B model runs *at all* in
128 GB, with full context, and that serving a quarter of its parameters from NVMe costs almost
nothing.

## Why the 51B-on-NVMe trick works

Qwen3.8-Flash-Next's 180B parameters split into:

| Block | Params | Role |
|---|---|---|
| Compute (MoE experts, attention, GDN, ...) | 128.8B | Matmuls — must be resident |
| n-gram / PLE embedding table | 51.2B | Pure lookup — does not need to be resident |

The table is one tensor, `per_layer_token_embd.weight`, shape `[160, 320001536]`. It is never
part of a matrix multiply: per generated token the model *gathers* about 16 rows out of
320,001,536. Embedding tables are sparsely accessed and deterministically addressed, so they can
live in off-accelerator storage — Qwen's own tech report makes this point about scaling n-gram
vocabulary. We pin the tensor to the CPU backend and let mmap serve it from NVMe:

```
-ot "per_layer_token_embd=CPU"  -lm mmap
```

Result: the model file is 103.7 GiB but only ~76.9 GiB is resident. Process RSS is ~1.4 GiB — the
table is served by the OS page cache backed by NVMe. See
[docs/how-it-works.md](docs/how-it-works.md) for the full memory arithmetic.

## Measured results

Hardware: DGX Spark (GB10, SM121), 121 GiB usable, ~273 GB/s memory bandwidth, CUDA 13.0, aarch64.
Model: `unsloth/Qwen3.8-Flash-Next-GGUF`, UD-Q4_K_XL (103.7 GiB, 4 shards).

Decode and prefill vs. context length (server-reported timings; prefill excluded from decode):

| Prompt tokens | Prefill tok/s | Decode tok/s | Major faults/token |
|---:|---:|---:|---:|
| 226 | 355 | 22.34 | 2.7 |
| 1,659 | 632 | 22.14 | 2.0 |
| 6,443 | 662 | 21.28 | 1.4 |
| 19,197 | 499 | 19.50 | 1.3 |

Decode degrades only ~13% from 226 to 19k tokens of context.

Table warming A/B (`tools/warm_table.py`):

| State | Table cached | Major faults/token | Decode tok/s |
|---|---:|---:|---:|
| Cold | 1.3% | 13.1 | 21.05 |
| Warmed | 79% | 2.1 | 22.40 |

Warming costs one sequential 26.8 GiB read (~26 s) and cuts major faults ~6x — but buys only
~+6% throughput. In other words: **keeping the table on NVMe is nearly free.** That is the
headline result. Full details and reproduction steps in [docs/benchmarks.md](docs/benchmarks.md).

Other measured facts:

- Load time: ~3m35s from NVMe.
- Steady state: ~95 GiB used, ~26 GiB page cache, process RSS ~1.4 GiB.
- Full 262,144-token context works. KV cache is only 24 KB/token (12 of 48 layers use full
  attention, GQA with 2 KV heads), so 262k tokens cost just 6 GiB.

## Quick start

1. Build llama.cpp with the unmerged PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742)
   (qwen4exp architecture support), plus the two patches in [`patches/`](patches/) (recommended,
   see below).
2. Download `unsloth/Qwen3.8-Flash-Next-GGUF` (UD-Q4_K_XL, 4 shards, 103.7 GiB).
3. Start the server:

```
./run.sh
```

4. Optionally warm the embedding table (one 26.8 GiB sequential read, ~26 s):

```
python3 tools/warm_table.py
```

### Speculative decoding: large win on copy-heavy work

`run.sh` enables `--spec-type ngram-mod` by default. It drafts spans from repetition in the
context, so it needs no draft model and no extra memory, and speculation is exact — the target
verifies every token, so output is identical.

| Task | Decode tok/s | Draft acceptance |
|---|---|---|
| Reproduce a given file with one change | 97.4 | 94.7% |
| Targeted bug fix in a given file | 68.6 | 81.4% |
| Add a function to a given file | 31.4 | 56.9% |
| Free-form prose (control) | 22.1 | 5.8% |

The gain tracks how much of the output already appears in the prompt, which is exactly the
shape of tool-driven editing. Free-form generation is unchanged. Disable with `SPEC=none`.

An external draft model was also tested (Qwen3.5-0.8B, vocab-compatible at 248320) and gave
**no** speedup: mean accepted length 2.88 yet decode stayed ~23 tok/s. Speculative decoding
normally amortizes one weight read over k tokens, but in a top-10-of-512 MoE, k tokens activate
up to k*10 different experts, so weight traffic scales with k. `ngram-mod` wins instead by
accepting very long spans (mean 50-60 tokens), which amortizes over the verify step rather than
over the weight read.

## Known issues

Documented honestly — this is early, on an unmerged architecture port.

1. **Concurrency crashes.** Any second in-flight request aborts with
   `src/models/qwen4exp.cpp:284: GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() && "the indexer cache must track the attention cache cell for cell") failed`.
   Workaround: run with `--parallel 1`. Single-stream is stable.
2. **Quantized KV cache aborts** on this architecture (`qwen4exp.cpp:544`). Keep KV at f16 — at
   24 KB/token it is cheap anyway.
3. **No speculative decoding.** The GGUF converter drops the MTP head
   (`supports_mtp_export = False`), so no MTP draft exists. An external draft model was tested
   (Qwen3.5-0.8B, vocab-compatible at 248320) and gave **no speedup**: mean accepted length 2.88,
   decode stayed ~23 tok/s. Why: speculative decoding amortizes one weight read over k tokens,
   but in a top-10-of-512 MoE, k tokens activate up to k×10 *different* experts — weight traffic
   scales with k and the amortization never happens.
4. **Requires unmerged llama.cpp PR #27742.** Expect churn until it lands.

## Patches

Two patches against llama.cpp ship in [`patches/`](patches/):

- **`rowband-ple-quant.patch`** — `llama-quant.cpp` stages the whole dequantized tensor through an
  f32 buffer; for this model's 51.2G-element table that is 204.8 GB, so quantizing it dies with
  `std::bad_alloc` on any machine under ~200 GB RAM. The patch dequantizes in ≤2 GiB row bands.
  Verified byte-identical to the unpatched output over 31,106,622,336 bytes. Only needed if you
  quantize the model yourself.
- **`canreuse-qwen4exp.patch`** — `llm_graph_input_qsa` and `llm_graph_input_ple` never override
  `can_reuse()`, whose base implementation returns false, so the whole compute graph is rebuilt
  and re-split every token and CUDA graph capture never engages (`graphs reused = 0`). The patch
  implements `can_reuse()` for both. Measured effect here: graphs reused 0 → 1543, decode +2.8%.

## Credits

- **Qwen** for the model and the tech report describing the n-gram/PLE design.
- **Unsloth** for the GGUF quants and llama.cpp PR #27742.
- **llama.cpp** for everything else.

MIT licensed.

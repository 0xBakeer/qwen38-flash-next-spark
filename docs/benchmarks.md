# Benchmarks

All numbers on this page are real measurements from a single machine. Nothing is extrapolated.
Where something was not measured, it says so.

## Provenance

| Item | Value |
|---|---|
| Hardware | NVIDIA DGX Spark (GB10, SM121), 128 GB unified memory, 121 GiB usable |
| Memory bandwidth | ~273 GB/s |
| Software | CUDA 13.0, aarch64 Linux, llama.cpp with PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) + `patches/canreuse-qwen4exp.patch` |
| Model | `unsloth/Qwen3.8-Flash-Next-GGUF`, UD-Q4_K_XL, 103.7 GiB, 4 shards |
| Serving config | `./run.sh` — table pinned to CPU backend + mmap, `--parallel 1`, f16 KV |

All decode/prefill figures are the server's own reported timings, with prefill excluded from the
decode rate. Everything is single-stream (`--parallel 1` — concurrency crashes on this
architecture, see the README's known issues).

## Decode / prefill vs. context length

| Prompt tokens | Prefill tok/s | Decode tok/s | Major faults/token |
|---:|---:|---:|---:|
| 226 | 355 | 22.34 | 2.7 |
| 1,659 | 632 | 22.14 | 2.0 |
| 6,443 | 662 | 21.28 | 1.4 |
| 19,197 | 499 | 19.50 | 1.3 |

Observations:

- Decode degrades only ~13% from 226 to 19,197 prompt tokens.
- Major faults per generated token *fall* with longer prompts — plausibly because a longer
  prefill has already faulted in more of the table rows the decode will reuse (not separately
  verified).
- Contexts beyond ~19k were not benchmarked for throughput; the full 262,144 window was verified
  to load and run (KV cost: 24 KB/token → 6 GiB at 262k).

**Reproduce:** start the server via `./run.sh` (table warmed, see below). Send single
completions with prompts of the sizes above and read the server's timing output for prefill and
decode rates. For faults/token, sample the process's major-fault counter (e.g. from
`/proc/<pid>/stat`) before and after a generation and divide the delta by the number of
generated tokens.

## Table-warming A/B

Warming = one sequential read of the 26.8 GiB `per_layer_token_embd` region, done by
`tools/warm_table.py` (~26 s on this NVMe).

| State | Table cached | Major faults/token | Decode tok/s |
|---|---:|---:|---:|
| Cold | 1.3% | 13.1 | 21.05 |
| Warmed | 79% | 2.1 | 22.40 |

**Important caveat, measured later:** the +6% above was measured with speculation OFF. With
`--spec-type ngram-mod` enabled, warming is worth far more — up to **+42%** on copy-heavy work
(52.6 -> 74.6 tok/s). Verifying a 50-60 token span touches many n-gram rows at once, so major
faults cost proportionally more than during one-token-at-a-time decode. See
[bench/results.md](../bench/results.md).


Observations:

- The table does **not** warm itself under normal use: rows are addressed by a 3-gram hash into
  320M slots, so a short workload rarely touches the same row twice. After a normal session it
  sat at 1.3% cached.
- Warming cuts major faults ~6x but buys only ~+6% decode. Keeping the table on NVMe is nearly
  free either way — that is the headline result of this repo.

**Reproduce:** for the cold row, drop caches (or reboot), start the server, run a generation and
measure as above; report the cached fraction of the table's file region (page-residency check
over the mmapped range). For the warmed row, run `tools/warm_table.py` first, then repeat.

## Memory footprint and load time

| Metric | Value |
|---|---:|
| Model file | 103.7 GiB |
| Resident weights | ~76.9 GiB |
| Table on NVMe | ~26.8 GiB |
| Steady-state memory used | ~95 GiB |
| Page cache at steady state | ~26 GiB |
| Process RSS | ~1.4 GiB |
| Load time from NVMe | ~3m35s |

**Reproduce:** start via `./run.sh`, wait for load to finish (timed from launch to server-ready),
then read overall used memory and page cache from `free`, and the process RSS from
`ps`/`/proc/<pid>/status`.

## Speculative decoding (negative result)

No MTP draft exists — the GGUF converter drops the MTP head (`supports_mtp_export = False`). An
external draft was tested instead:

| Draft | Vocab | Mean accepted length | Decode |
|---|---:|---:|---|
| Qwen3.5-0.8B | 248320 (compatible) | 2.88 | ~23 tok/s — no speedup |

Acceptance was healthy; throughput did not move. Explanation: speculative decoding pays off by
amortizing one weight read over k accepted tokens. In a top-10-of-512 MoE, k tokens activate up
to k×10 *different* experts, so weight traffic grows with k and the amortization never happens.

**Reproduce:** run the server with the draft model attached via llama.cpp's speculative options
and compare the reported decode rate and acceptance statistics against the no-draft baseline.

## Graph-reuse patch

With stock PR #27742, `llama_context` reports `graphs reused = 0`: the qsa/ple graph inputs never
override `can_reuse()`, so the graph is rebuilt and re-split every token and CUDA graph capture
never engages. With `patches/canreuse-qwen4exp.patch`:

| Metric | Before | After |
|---|---:|---:|
| Graphs reused | 0 | 1543 |
| Decode | baseline | +2.8% |

**Reproduce:** run the same generation with and without the patch and compare the
`graphs reused` counter in the server log and the reported decode rate.

## Not measured

- Throughput at contexts between 19k and 262k.
- Multi-request throughput (blocked by the concurrency crash).
- Other quantization levels, other draft models, batch/offline throughput.

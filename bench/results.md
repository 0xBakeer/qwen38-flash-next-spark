# Benchmark results

> ### Superseded — 2026-08-27
>
> This page records a run from **2026-08-26**. A re-measurement the next day, on the same
> commit and the same patches but with a harness that discards a warmup and varies the prompt
> between repetitions, did not reproduce two of its conclusions:
>
> | claim here | re-measured |
> |---|---|
> | Reproduce a file, cold: **52.6** | **88.5** |
> | Warming worth up to **+42%** on copy-heavy work | **no measurable difference** either way |
> | Free-form prose: **22.2** | **27.8** |
>
> The difference is not fully explained. The most likely factor is that the current build has
> `patches/canreuse-qwen4exp.patch` engaging CUDA graph capture, which both raises the floor
> and removes the page-cache sensitivity that warming was compensating for.
>
> Kept as the dated record of what was measured then. **Current numbers:**
> [../docs/measurements.md](../docs/measurements.md).

Hardware: DGX Spark (GB10, SM121), 121 GiB unified memory, ~273 GB/s, CUDA 13.0, aarch64.
Model: `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q4_K_XL (103.7 GiB, 4 shards), llama.cpp PR #27742
at `035e227` plus the patches in `patches/`. Context 262144, `--parallel 1`,
`--spec-type ngram-mod`, n-gram table pinned to CPU and served from NVMe.

Reproduce with `bench/cold_vs_warm.sh` (both tables) or `bench/run_bench.py` (one run).
All figures are the server's own `timings`, so prefill is never counted as decode.

## Decode vs task shape

Speculation drafts from repetition in the prompt, so throughput tracks how much of the
output already appears in the input. Both runs are identical except for how much of the
n-gram table is resident in the page cache.

| Task | Cold (17% cached) | Warm (99% cached) |
|---|---|---|
| Reproduce a file with one change | 52.6 | **74.6** |
| Targeted bug fix | 46.0 | **51.6** |
| Add a function | 32.4 | 30.4 |
| Free-form prose (control) | 22.2 | 23.3 |

Output was verified correct in every coding case (the requested change was present).

Two things to read out of this:

- **The gain tracks copying, not the task being "code".** Adding a function means writing new
  lines, so it behaves much more like prose than like reproducing a file.
- **Warming the table matters most exactly when speculation is working.** Verifying a 50-60
  token span touches many n-gram rows at once, so major faults cost proportionally more than
  they do during ordinary one-token-at-a-time decode. Measured without speculation the same
  warm/cold difference was only about 6%.

## Decode and prefill vs context length

Measured separately, cold table, speculation on but with prompts that give it nothing to
draft from, so these are close to the floor.

| Prompt tokens | Prefill tok/s | Decode tok/s |
|---|---|---|
| 95 | 77 | 23.6 |
| 303 | 440 | 21.2 |
| 1,020 | 609 | 21.9 |
| 3,252 | 671 | 20.8 |
| 9,630 | 648 | 21.0 |

Decode is essentially flat out to ~10k. A separate earlier run reached 19,197 tokens at
19.5 tok/s, so the falloff to that depth is around 13%. There is no long-context collapse.

## Page cache behaviour

The table does not warm on its own: 320,001,536 rows addressed by a 3-gram hash means a short
workload rarely touches a row twice. Observed at 262144 context:

| Moment | Table cached |
|---|---|
| After the startup warm | 100% |
| After a benchmark pass | 0.1% |
| After re-warming, under load | 50% |
| After dropping caches and re-warming with the box otherwise idle | 99% |

A full warm is one sequential 26.8 GiB read, about 26 s at ~1.0 GiB/s. At 262k context the
model's own file pages evict the table again over time. At smaller contexts there is more
headroom and it stays resident.

## Not measured

- Throughput between 19k and 262k context.
- Multiple concurrent requests. These runs are all single-stream. (An earlier version of this
  line said concurrency aborts the server. It does not: eight simultaneous requests all returned
  200, and `--parallel 2` serves two slots. Neither was measured *here*.)
- Any quant other than UD-Q4_K_XL.
- Whether a longer or repeated run converges on a stable cached fraction at 262k.

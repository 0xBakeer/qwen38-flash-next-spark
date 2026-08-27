# How it works

This document explains why a 180B-parameter model fits and runs on a 128 GB DGX Spark: what the
n-gram/PLE table is, the memory arithmetic, how the page cache serves the table from NVMe, why
warming helps (and why it barely matters), and the KV-cache math that makes 262k context cheap.

## The parameter split

Qwen3.8-Flash-Next has 180B parameters, but they are not all the same kind of parameter:

| Block | Params | Access pattern |
|---|---:|---|
| Compute: MoE experts, attention, GDN, etc. | 128.8B | Dense reads every token (per activated expert) — must be resident |
| n-gram / PLE embedding table | 51.2B | Sparse gather: ~16 rows per token — can live on NVMe |

The second block is a single tensor:

```
per_layer_token_embd.weight    shape [160, 320001536]
```

That is 160 × 320,001,536 = 51.2G elements — 28% of the model — in one tensor.

## Why an embedding table can live on NVMe

A weight matrix in a matmul is read *in its entirety* every time it is used. If it isn't resident,
every token pays the full transfer cost, and inference speed collapses to storage bandwidth.

An embedding table is different in two ways:

1. **Sparsely accessed.** Per token, the model gathers ~16 rows out of 320,001,536. It never
   touches the other 99.999995% of the table for that token.
2. **Deterministically addressed.** The rows to fetch are known from the token ids (via an n-gram
   hash) before the gather runs — this is a lookup, not a computation over the whole tensor.

So the per-token traffic to the table is a handful of small reads, not a 26.8 GiB stream. Reads
like that are exactly what an OS page cache over NVMe is good at. Qwen's own tech report makes
this point: n-gram vocabulary can be scaled up precisely because the resulting table can be
served from off-accelerator storage.

The llama.cpp incantation (from `./run.sh`):

```
-ot "per_layer_token_embd=CPU"  -lm mmap
```

`-ot "per_layer_token_embd=CPU"` pins the tensor to the CPU backend, so it is never copied to
accelerator-resident memory. With mmap, the tensor's pages are only faulted in as rows are
actually gathered.

## The memory arithmetic

At UD-Q4_K_XL the model file is **103.7 GiB** (4 shards). Of that:

| Portion | Size | Where it lives |
|---|---:|---|
| Compute weights (128.8B params) | ~76.9 GiB | Resident (unified memory) |
| PLE table (51.2B params) | ~26.8 GiB | NVMe, served via page cache |

The 26.8 GiB figure is simply 103.7 − 76.9, and matches the size of the sequential read that
warms the full table.

Measured steady state on a 121 GiB-usable GB10:

- ~95 GiB used overall
- ~26 GiB page cache (mostly the table, once warmed)
- process RSS ~1.4 GiB — the weights are mmapped, so almost nothing is process-private

This leaves room for the full-context KV cache (see below) inside 128 GB. Load time from NVMe is
~3m35s.

## Page-cache behaviour, and why the table never warms itself

You might expect the hot rows of the table to accumulate in the page cache naturally as you use
the model. Measured answer: they don't, to any useful degree.

The reason is the addressing scheme. Rows are selected by a 3-gram hash into 320M slots. A short
or medium workload produces a long stream of *distinct* 3-grams, so it rarely touches the same
row twice — there is no small hot set for the cache to converge on. After a normal session the
table was only **1.3% cached**, and decode paid **13.1 major faults per token**.

The fix looks trivial: read the table region once, sequentially. `tools/warm_table.py` does
this — one 26.8 GiB sequential read, ~26 s. Measured effect at the time:

| State | Table cached | Major faults/token | Decode tok/s |
|---|---:|---:|---:|
| Cold | 1.3% | 13.1 | 21.05 |
| Warmed | 79% | 2.1 | 22.40 |

> **Corrected 2026-08-27 — warming is no longer worth doing.** Re-measured on the
> current build with `patches/canreuse-qwen4exp.patch` applied, warming produces **no
> measurable change**. Prose is 27.8 tok/s at both 17.7% and 58.1% residency; file
> reproduction is 88.5 cold against 86.2 warm, with ranges that overlap heavily. The
> earlier "+42% on copy-heavy work" figure did not reproduce.
>
> The likely cause is the patch itself: once CUDA graph capture engages, the per-token
> cost that page faults used to sit on top of is much smaller, so residency stops
> mattering. Note that the current *cold* number (88.5) exceeds the old *warm* one (74.6).
>
> `tools/warm_table.py` is kept for A/B work. The boot-time warming ritual is not needed.

**What we believed on 2026-08-26, and no longer do.** The +6% above was measured with
speculation OFF. With `--spec-type ngram-mod` enabled, warming appeared to be worth far more —
up to +42% on copy-heavy work (52.6 -> 74.6 tok/s) — on the reasoning that verifying a 50-60
token span touches many n-gram rows at once, so major faults should cost proportionally more
than during one-token-at-a-time decode.

That reasoning is sound and the effect is real when page faults dominate. It simply stopped
applying: with CUDA graph capture engaging, faults no longer dominate. The re-measurement
above supersedes it. Original record: [bench/results.md](../bench/results.md).


Major faults drop ~6x — but decode only improves ~6%. That asymmetry is the real finding: even
*cold*, with 13 major faults per token, the NVMe lookups barely dent throughput, because decode
time is dominated by reading the activated compute weights out of ~273 GB/s unified memory, and
the few-KB random table reads largely hide behind that. Warming is worth ~26 s of your time, but the table being
on NVMe at all is nearly free.

## KV-cache math: why 262k context costs 6 GiB

Qwen3.8-Flash-Next has 48 layers, but only **12** of them use full attention with a growing KV
cache; the rest use GDN-style layers whose state does not grow with context. The full-attention
layers use GQA with only **2 KV heads**.

Net cost: **24 KB per token** of context (f16 KV). So the full 262,144-token window costs

```
262,144 tokens × 24 KB = 6 GiB
```

— which is why max context is simply enabled by default in `./run.sh` rather than being a
trade-off. Note the KV cache must stay f16: quantized KV aborts on this architecture (see the
known issues in the README).

## What this does *not* solve

Serving the table from NVMe removes the memory obstacle; it does not make a 180B model fast.
Decode is ~22 tok/s single-stream at Q4 on this hardware (~273 GB/s memory bandwidth), and
degrades only ~13% out to 19k tokens of context. Speculative
decoding does not help this architecture in practice (measured; see the README), so ~22 tok/s is
what one Spark does today.

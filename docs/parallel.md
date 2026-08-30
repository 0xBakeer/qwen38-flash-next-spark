# How many requests at once

The editing recipe shipped `--parallel 1` from the start. It now ships **2**. This page is why,
and why the answer is not the same as the long-context recipe's, even though the knob looks the
same.

## The thing that makes llama.cpp different

**`--ctx-size` is divided by `--parallel`.** Each slot gets a fixed, private share of the
context, decided at startup:

| `--parallel` | context per request | measured `n_ctx` per slot |
|---:|---:|---|
| 1 | 262,144 | — |
| **2** | **131,072** | 131072 |
| 8 | 32,768 | 32768 |
| 64 | 4,096 | 4096 |

Read off `/slots` on a running server, and the products come out exactly at 262,144 every time.

vLLM does not work this way. It keeps one shared KV pool — 654,635 tokens at `GPU_MEM=0.85` —
and hands it out on demand, which is why raising `--max-num-seqs` there costs nothing when the
server is quiet and why that recipe ships 16. Here a slot count is a **standing claim** on the
context, whether or not anyone is using it.

So the trade is real and it runs in one direction: concurrency is bought with the maximum prompt
any single request can hold.

## What two slots cost, and what they buy

Nothing, and 1.2–1.3×.

**Cost, at one caller** — three one-slot repeats bound the noise floor at 6.5%, and the two-slot
figure lands inside it:

| | `--parallel 1` | `--parallel 2` |
|---|---:|---:|
| `serve-single-i256-o256` decode | 34.12 / 34.99 / 37.43 tok/s | **35.02** |
| `serve-single` TTFT p50 | 1.05–1.10 s | 1.01 s |
| `prefill-32k` | 1,481.0 tok/s | **1,484.9** (0.3%) |
| `prefill-8k` | 1,660.9 tok/s | 1,772.1 |

**Buy, under load** — paired runs, same box, same projector, slot count the only variable:

| | 1 slot | 2 slots | |
|---|---:|---:|---|
| `serve-short-c16` aggregate | 36.40 tok/s | **45.27** | 1.24× |
| `serve-short-c16` TTFT p50 | 53.3 s | 41.1 s | |
| `serve-chat-c8` aggregate | 25.45 tok/s | **32.98** | 1.30× |
| `serve-chat-c8` TTFT p50 | 71.8 s | 48.1 s | |
| requests completed | 320/320, 200/200 | 320/320, 200/200 | |

One measurement subtlety worth carrying: the first `serve-short-c16` one-slot run started at
**0.06%** page-cache residency, straight after a server restart, against its two-slot twin's
72.22%. That confound was the same size as the effect, so the one-slot run was repeated warm
(75.26%) and came back at **36.40** against the cold run's 36.98 — 1.6% apart. Residency did not
distort it, and the repeat also bounds the noise on that workload.

131,072 tokens is still four times the largest context this recipe has ever been measured at, so
two slots give away a ceiling nobody was reaching.

## What more slots cost

More than they buy, and past a point the server stops working.

`sweep-parallel-1-64-i1k-o256-v1` offers 1→64 concurrent requests against a fixed server. Run at
8 slots and at 64:

| offered | 8 slots: per-req tok/s | TTFT p50 | | 64 slots: per-req tok/s | TTFT p50 |
|---:|---:|---:|---|---:|---:|
| 1 | 22.64 | 2.22 s | | 22.12 | 3.45 s |
| 2 | 14.92 | 2.22 s | | 12.34 | 6.31 s |
| 4 | 7.09 | 2.48 s | | 6.81 | 5.41 s |
| 8 | 5.75 | **3.12 s** | | 4.78 | 7.28 s |
| 16 | 5.46 | 50.96 s | | 3.26 — **1 of 64 completed** | 73.25 s |
| 32 | 5.27 | 141.01 s | | — | — |
| 64 | 5.59 — 68 of 76 completed | 180.84 s | | — | — |

Harness headline for the whole sweep: **43.62 tok/s at 8 slots, 27.35 at 64.**

**64 slots is worse than 8 at every single level**, and it collapsed at 16 offered requests: 63
of 64 failed with `RemoteProtocolError: peer closed connection without sending complete message
body`. It was not a crash — no assert, no CUDA error, no out-of-memory in the server log. The log
is wall-to-wall:

```
srv alloc: - making room for prompt cache entry, removing oldest entry (size = 400.407 MiB)
srv alloc: - making room for prompt cache entry, removing oldest entry (size = 405.919 MiB)
```

Prompt-cache thrashing, with generation falling to **1.3–1.9 tok/s** and three-second windows at
**0.30 t/s**, so the workload's own 300-second budget expired underneath it. (The server process
was also gone when the box was checked half an hour later, with the log ending mid-request and no
shutdown line. We could not confirm what removed it, so we are not claiming it crashed.)

Eight slots is the last setting on this list that serves its own concurrency well — up to 8
callers at ~3 s to first token — but it costs 87.5% of the context ceiling to get there, and
first-token latency past its slot count is a cliff, not a slope: 3.12 s at 8 offered, 50.96 s at
16.

## The reading

- **2 is the default.** Free when idle, 1.2–1.3× when two people or two tools are talking to it,
  and 131k of context left over.
- **Raise it only if you know your prompts are short.** 8 slots serves 8 concurrent callers
  properly, at 32k each. That is a coding-assistant configuration, not a long-document one.
- **Do not go looking for vLLM's numbers here.** The long-context recipe serves 16 concurrent at
  96–109 tok/s aggregate with first tokens under 2.7 s. This engine cannot do that at any slot
  count, because the context is partitioned rather than pooled. If you need real concurrency,
  that is the recipe for it.
- **`--parallel 64` is not a configuration**, on this model, on this box.

One thing this page cannot tell you: whether the 1.24–1.30× holds on llama.cpp master. Every
figure here was measured on the pinned pre-merge commit `035e227` plus the patches in
`patches/`. See [sources.md](sources.md).

Cell data: [inference-atlas](https://github.com/0xBakeer/inference-atlas/tree/main/results/llamacpp/Qwen/Qwen3.8-Flash-Next).

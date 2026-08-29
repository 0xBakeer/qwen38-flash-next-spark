# Long-context recipe — vLLM + NVFP4 + MTP

**For chat, reasoning, and anything that reads long documents.**

This is the even performer. It runs at roughly the same speed whatever you ask it, reads long
documents about five times faster than the other recipe, and does not slow down as the context
fills up. It is *not* the one to use if a coding agent is rewriting whole files — see
[../llamacpp-edit](../llamacpp-edit/) for that.

## Measured

| | |
|---|---|
| Free-form prose | **32.2 tok/s** |
| Rewriting a file with one change | 39.1 tok/s |
| Fixing a bug in a file | 35.0 tok/s |
| Adding a function | 33.6 tok/s |
| Spread across those four | **1.2×** (the edit recipe swings 3.2×) |
| Time to first token, short prompt | **~0.3 s** |
| Prefill | **~2,200–2,460 tok/s**, flat to 195k tokens |
| Decode at 1k / 32k / 128k context | 31.7 / 33.5 / 31.7 — **no falloff** |
| Concurrent requests | **16** served well; 64 possible for batch work |
| Aggregate decode, 16 concurrent | **96–109 tok/s**, TTFT under 2.7 s |

Full method and raw figures: [../../docs/measurements.md](../../docs/measurements.md).

## Why it behaves this way

The model ships a **trained MTP head** — a small predictor that guesses the next few tokens.
vLLM runs it; llama.cpp cannot, because the GGUF converter drops the head (`supports_mtp_export
= False`). We confirmed this directly: our GGUF contains **zero** MTP tensors across all four
shards, the NVFP4 checkpoint contains **all 31**.

Because the head is trained rather than copying from your prompt, it works the same on any text.
Hence the flatness — and hence why this recipe wins on prose and loses on file rewriting.

The gain is real but modest: across engines, prose goes 27.8 → 32.2 (**~1.16×**). We did not
run our own MTP-off A/B; the upstream recipe's in-engine one measured 17 → 27 tok/s at MTP=2,
so part of our cross-engine delta is the engine and quant, not the head. Either way it is far
from the 3–5× dense models see. On a top-10-of-512 mixture-of-experts,
verifying *k* draft tokens touches the *union* of experts across those positions, so speculation
buys far less here than the 3–5× seen on dense models. Raising `MTP` above 3 makes it worse, not
better.

## Install

```bash
./setup.sh      # builds the container, downloads ~126 GB
./serve.sh      # starts on http://localhost:8000/v1
```

`setup.sh` clones and builds [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
(Apache-2.0), whose patch serves the n-gram table from disk. That patch is the reason this fits at
all, and it stays in their repository rather than being copied into this one. See
[../../CREDITS.md](../../CREDITS.md).

First start reads ~83 GiB off disk and takes **12–15 minutes**.

## Settings

| variable | default | what it does |
|---|---|---|
| `PORT` | `8000` | host port |
| `BIND` | `127.0.0.1` | loopback only. `0.0.0.0` exposes it to your whole network |
| `CTX` | `262144` | context length. The full native window |
| `SEQS` | `16` | concurrent sequences. See *How many at once* below |
| `GPU_MEM` | `0.85` | share of the 128 GB pool for weights + KV |
| `MTP` | `3` | speculative tokens. `0` disables, for an A/B |
| `PREWARM` | `1` | stream the table at boot so the first request is not cold |

The upstream default is `GPU_MEM=0.78`, which leaves only 10.82 GiB of KV — **227,651 tokens,
less than the model's own context window**, so a single full-length request will not fit. We
raise it to 0.85, which measured 18.13 GiB of KV = **641,601 tokens**, comfortably 2.4× the
context, while still leaving ~19 GiB of headroom.

## How many at once

`SEQS` was `2`, the upstream container's default. This model and this recipe are both days
old, so there was no published figure to set it against. Two slots is a scheduler cap rather
than a memory limit: at `GPU_MEM=0.85` the server reports a KV pool of **654,635 tokens**,
while 64 concurrent ~1.3k-token requests need about **83,000**. It left roughly 87% of the
pool unused.

Measured with the pinned parallel sweeps from
[inference-atlas](https://github.com/0xBakeer/inference-atlas), 32 and 256 requests
respectively, every request completed:

| concurrent | tok/s each | aggregate | TTFT p50 | | tok/s each | aggregate | TTFT p50 |
|---:|---:|---:|---:|---|---:|---:|---:|
| | *512-token prompts* | | | | *1k-token prompts* | | |
| 1 | 27.70 | 27.7 | 0.68 s | | 25.01 | 25.0 | 1.23 s |
| 2 | 20.23 | 40.5 | 1.02 s | | 19.63 | 39.3 | 1.39 s |
| 4 | 15.06 | 60.2 | 1.10 s | | 14.04 | 56.2 | 1.48 s |
| 8 | 10.68 | 85.4 | 1.27 s | | 9.01 | 72.1 | 1.73 s |
| **16** | 6.80 | 108.8 | **2.15 s** | | 6.00 | 96.0 | **2.64 s** |
| 32 | 4.21 | 134.8 | 13.80 s | | 3.46 | 110.8 | 16.32 s |
| 64 | — | — | — | | 2.02 | 129.4 | 70.42 s |

**Aggregate throughput never plateaus** — it keeps climbing to 64, just with sharply
diminishing returns: 1→8 buys 2.9x, 8→16 buys 33%, 16→32 buys 15%, 32→64 buys 17%.

**Time to first token is where the wall is, and it is a cliff rather than a slope.** It sits
under 2.7 s all the way to 16 concurrent, then goes to 16 s at 32 and **70 s at 64**. Both
sweeps agree on where it breaks. The cause is `--max-num-batched-tokens 8192`: past a certain
number of simultaneous prefills, each one is chunked across many scheduler steps before its
first token appears.

So **16 is the last concurrency this configuration serves well**, at 96–109 tok/s aggregate,
which is already about 75% of everything the box will ever do. Going to 64 buys ~35% more
aggregate throughput and costs 27x on first-token latency.

We ship `SEQS=16` for that reason. It is a cap, not a batch size — with one caller it behaves
exactly like `SEQS=2`, so nothing is lost when the server is quiet, and a burst of up to 16 is
served without queueing. Beyond 16, callers queue rather than all being served badly: 64
requests through a 16-slot server finish in about 171 s against 127 s if batched 64-wide, but
the ones being served see 2.6 s to first token instead of 70 s.

**Set `SEQS=64` if you are running batch work** where nothing is waiting on a first token and
the extra ~35% aggregate throughput is what matters. That is also the configuration the atlas
cells for this recipe were measured at, deliberately, so that no cell is capped by the
scheduler.

## Turn thinking off

Thinking is on by default and **86% of generated tokens were reasoning** in our measurements. The
same prompt answered in **15.0 s with thinking off against 55.1 s with it on** — not because
tokens got faster, but because there were a quarter as many.

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

## Known issues

- **Prefix caching must stay off** (`--no-enable-prefix-caching`) — a GB10 GDN kernel bug. This is
  also why our prefill figures are honest: nothing is served from cache.
- **Full `torch.compile` is off** — an Inductor int64-indexing assert on sm_121.
- **The n-gram gather must stay outside CUDA graphs.** `serve.sh` declares it a splitting op and
  captures `PIECEWISE`. Do not switch to a `FULL` capture mode.
- **`VLLM_PLE_CPU_OFFLOAD=1` hangs.** The official pinned-host-RAM path registers a
  `PleOffloadLayer` and then spins a core with no disk I/O, indefinitely — it expects an offload
  worker this image does not launch. Only `VLLM_PLE_MMAP` works here.
- **Sixteen sequences, not two hundred.** A chat UI, a coding agent and a background job can
  all point at this endpoint without queueing. This is still not a multi-user server: past 16
  concurrent requests time to first token collapses from under 2.7 s to 16 s at 32 and 70 s
  at 64.

## Verifying it works

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next",
       "messages":[{"role":"user","content":"Reply with exactly: ok"}],
       "max_tokens":50,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

Then measure it yourself:

```bash
../../bench/portable_bench.py --api http://127.0.0.1:8000 --label mine
```

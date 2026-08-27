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
| Concurrent requests | 2 |

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
| `SEQS` | `2` | concurrent sequences |
| `GPU_MEM` | `0.85` | share of the 128 GB pool for weights + KV |
| `MTP` | `3` | speculative tokens. `0` disables, for an A/B |
| `PREWARM` | `1` | stream the table at boot so the first request is not cold |

The upstream default is `GPU_MEM=0.78`, which leaves only 10.82 GiB of KV — **227,651 tokens,
less than the model's own context window**, so a single full-length request will not fit. We
raise it to 0.85, which measured 18.13 GiB of KV = **641,601 tokens**, comfortably 2.4× the
context, while still leaving ~19 GiB of headroom.

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
- **Two sequences, not two hundred.** If a chat UI, a coding agent and a background job all point
  at this endpoint, they share two slots and queue.

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

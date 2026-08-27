# Editing recipe — llama.cpp + context-copying speculation

**For coding agents that rewrite files you hand them.**

This is the specialist. On the task it is built for it is more than twice as fast as anything
else measured on this box; on writing new text it is the slower of the two recipes. If you mostly
chat with the model or feed it long documents, use [../vllm-longctx](../vllm-longctx/) instead.

## Measured

Cold page cache — the n-gram table only 17.7% resident, i.e. the state you actually boot into.

| task | tok/s | correct |
|---|---|---|
| Reproduce a file with one change | **88.5** | 3/3 |
| Targeted bug fix | **46.1** | 3/3 |
| Add a function | 32.2 | 3/3 |
| Free-form prose (control) | 27.8 | n/a |

That is a **3.2× spread across four tasks with one model and one server.** The number depends
almost entirely on how much of the answer already exists in the question.

## Why

`--spec-type ngram-mod` drafts candidate spans by looking for repetition **in your prompt**. Hand
it a file and ask for one change, and nearly every token of the answer is already in front of it —
it lifts 60-token spans in a single verified step. Ask for something new and it has nothing to
copy, so you get the model's honest unaccelerated rate of about 28.

Speculation is **exact**: the model verifies every drafted token, so output is byte-identical to
running without it. You are not trading quality for speed.

## Install

```bash
./run.sh setup    # builds llama.cpp (PR #27742 + patches), fetches ~105 GB
./run.sh serve    # starts on http://localhost:8000/v1
./run.sh bench    # measures the four tasks above
```

From the repository root you can also use `../../run.sh edit setup|serve|bench|all`.

## Settings

| variable | default | what it does |
|---|---|---|
| `CTX` | `262144` | context length. KV is only ~24 KB/token, so the full window is affordable |
| `PORT` | `30000` | change to `8000` to match the other recipe |
| `SPEC` | `ngram-mod` | `none` disables speculation |
| `QUANT` | `UD-Q4_K_XL` | see below before changing this |
| `PR_SHA` | `035e227` | pinned commit the patches apply to |

## Two things that are not worth doing

**Do not bother warming the table.** The recipe used to recommend it. Re-measured on the current
build with the `canreuse-qwen4exp` patch applied, warming buys **nothing**: prose is 27.8 tok/s at
both 17.7% and 58.1% residency, identical to the decimal, and the medians moved in both directions by less than the
run-to-run spread. `tools/warm_table.py` is still here for A/B work, but the boot ritual is unnecessary.

**Do not drop to a lower-bit K-quant for speed.** `UD-Q3_K_XL` moves 19% fewer bytes per token and
is **14% slower on prose** (24.0 vs 27.8). K-quant dequantisation costs more than the memory
traffic it saves, because this configuration is not bandwidth-bound — see
[../../docs/ruled-out.md](../../docs/ruled-out.md). Choose a smaller quant for disk space or
quality reasons, not for speed.

## Known issues

- **One request at a time.** `--parallel 1` is forced. Concurrent requests do not crash — they
  queue, and we verified that: 8 simultaneous requests all returned 200, in near-perfect 3-second
  succession. But throughput does not increase, so N clients each get roughly 1/N of the speed.
- **Quantized KV aborts** on this architecture. Keep it at f16; at ~24 KB/token it is cheap.
- **No MTP.** The GGUF converter drops the model's trained draft head — we confirmed zero MTP
  tensors across all four shards. If you want that, use the other recipe.
- **Built from an unmerged PR** (#27742). Expect churn until it lands.

## Patches

Two, in [`../../patches/`](../../patches/):

- **`canreuse-qwen4exp.patch`** — implements `can_reuse()` for the qwen4exp graph inputs so CUDA
  graph capture engages at all. Without it the graph is rebuilt every token. This is why our prose
  figure (27.8) is well above the 19–25 tok/s others report for the same quant on the same box.
- **`rowband-ple-quant.patch`** — only needed if you quantize the model yourself. Staging the
  51.2B table through an f32 buffer needs 204.8 GB; this dequantizes in ≤2 GiB row bands.

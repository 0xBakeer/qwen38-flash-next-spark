# Benchmarks — how to reproduce them

**Current numbers live in [measurements.md](measurements.md).** This page is the method: how each
figure was taken, for both recipes, so you can check them on your own box.

All numbers in this repository are real measurements from a single machine. Nothing is
extrapolated. Where something was not measured, it says so.

## Provenance

| Item | Value |
|---|---|
| Hardware | NVIDIA DGX Spark (GB10, SM121), 128 GB unified memory, ~121 GiB usable |
| Memory bandwidth | ~273 GB/s |
| OS | Ubuntu 24.04.4, kernel 6.17.0-1029-nvidia aarch64, driver 580.173.02, CUDA 13.0, swap off |
| Editing recipe | llama.cpp, PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) @ `035e227` + `patches/canreuse-qwen4exp.patch`; `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q4_K_XL, 103.7 GiB |
| Long-context recipe | vLLM `0.1.dev20073+g8e685d198` (image built from upstream `82ed48d`), `RadixArk/Qwen3.8-Flash-Next-NVFP4`, MTP=3, prefix caching off (`PREFIX_CACHE=0`; it now ships on, so set this to reproduce), PIECEWISE CUDA graphs |

Nothing else contended for the GPU during any run: the host's usual API endpoint was shut off for
the duration.

---

## The harness

Both recipes are measured by **one script**, [`../bench/portable_bench.py`](../bench/portable_bench.py),
so their numbers are comparable:

```bash
./bench/portable_bench.py --api http://127.0.0.1:8000 --label mine
```

It streams responses and times **time-to-first-token separately** from decode. That matters: the
two engines report their own counters differently and are not comparable directly. llama.cpp
exposes a `timings` block; vLLM does not, and vLLM's own "generation throughput" counts accepted
draft tokens, which reads ~6 tok/s higher than what a client actually receives. Measuring
end-to-end from the client side avoids picking whichever number flatters.

Two methodology details, both learned the hard way and inherited from `bench/run_bench.py`:

- **One warmup per task is discarded.** The first request after any state change is markedly
  slower.
- **Every repetition substitutes a different token into the prompt.** Repeating an *identical*
  prompt lets the context-copying drafter draft from its own memory of the previous generation.
  Measured inflation when we did not do this: **60 → 169 tok/s.** Any benchmark that sends the
  same prompt three times and averages is measuring its own cache.

### A prefill trap worth knowing

An earlier run of ours reported ~1,500–1,660 tok/s prefill for llama.cpp. That was wrong. The
prompts shared a common prefix, so most requests hit the server's slot cache. The giveaway was the
spread: TTFT p50 of 477 ms against a max of 21 s.

**The honest cold prefill rate is the slowest request, not the aggregate.** Our current prefill
figures use `max_tokens=1` and, on the vLLM side, benefit from prefix caching being disabled on
this platform — nothing is served from cache.

---

## Decode by task shape

The four tasks are defined in `bench/portable_bench.py`: reproduce a file with one change, a
targeted bug fix, add a function, and free-form prose as a control. They are chosen to span
**how much of the answer already exists in the question**, because on this model that is the
variable that dominates throughput.

```bash
# editing recipe
./run.sh edit serve
./bench/portable_bench.py --api http://127.0.0.1:8000 --label llamacpp

# long-context recipe
./run.sh longctx serve
./bench/portable_bench.py --api http://127.0.0.1:8000 --label vllm
```

Results: [measurements.md](measurements.md#decode-by-task-shape).

## Cold vs warm page cache

Only meaningful for the **editing** recipe. Under vLLM the n-gram table stays mapped by the live
process, so `drop_caches` cannot evict it and there is no cold state to measure while it serves.

```bash
MODEL=~/.qwen38fn/model/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  bench/cold_vs_warm.sh
```

The script drops the page cache (needs root, or a privileged container), reports table residency
via `mincore(2)`, measures, then warms with `tools/warm_table.py` and measures again.

**Result, and it reverses this repo's earlier advice:** warming buys nothing on the current build.
Prose measures 27.8 tok/s at both 17.7% and 58.1% residency, and on the other tasks the medians
move in both directions by less than the run-to-run spread. Read that as no measurable effect,
not as "cold is faster". See [ruled-out.md](ruled-out.md#warming-the-n-gram-table--no-effect).

## Prefill and decode across the context window

```bash
# prompt of a known size, one output token: TTFT is prompt ingestion alone
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"<long text>"}],"max_tokens":1}'
```

Divide the reported `prompt_tokens` by the measured TTFT. For decode at depth, send the same long
prompt with a real generation request and time tokens after the first.

Results: [measurements.md](measurements.md#prefill).

## Memory and residency

```bash
free -g                                    # system view
tools/page_cache.py <shard-1.gguf>         # n-gram table residency, mincore(2), no root needed
docker logs qwen38-flash | grep -i "KV cache\|Actual usage"   # vLLM's own accounting
```

`tools/page_cache.py` locates `per_layer_token_embd.weight` across a split GGUF and reports what
fraction of its byte range is in the page cache.

**A caution.** Unified memory makes `free` confusing here — "GPU" allocations and page cache come
from the same pool, and mapped file pages do not appear where you would expect. Our own
file-level residency measurement totals more than physical RAM across the vLLM checkpoint. It
validates on single files (a control reads 0.0%, goes to 100.0% when read, files are not sparse)
but we do not trust the aggregate and do not publish it. See
[measurements.md](measurements.md#one-measurement-we-are-not-publishing).

## Concurrency

```bash
for i in $(seq 1 8); do
  curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"count to 40"}],"max_tokens":200}' \
    -o /tmp/c-$i.json -w "req$i %{http_code} %{time_total}s\n" &
done; wait
```

**Correction to an earlier claim in this repo.** This page previously said concurrent requests
crash the editing recipe. They do not. Eight simultaneous requests all returned 200, the server
survived, and completions arrived in near-perfect 3-second succession — that is `--parallel 1`
**queueing**, not batching. Aggregate throughput therefore stays at the single-stream rate and each
of N clients gets roughly 1/N of it. The long-context recipe genuinely serves 2 sequences.

---

## Editing-recipe specifics

These were measured on the editing recipe only and have not been re-run against vLLM.

### Major page faults per token

| Prompt tokens | Prefill tok/s | Decode tok/s | Major faults/token |
|---:|---:|---:|---:|
| 226 | 355 | 22.34 | 2.7 |
| 1,659 | 632 | 22.14 | 2.0 |
| 6,443 | 662 | 21.28 | 1.4 |
| 19,197 | 499 | 19.50 | 1.3 |

Faults per generated token *fall* with longer prompts — plausibly because a longer prefill has
already faulted in table rows the decode reuses. Not separately verified.

**Reproduce:** sample the process's major-fault counter from `/proc/<pid>/stat` before and after a
generation, divide the delta by tokens generated.

### Memory footprint and load time

| Metric | Value |
|---|---:|
| Model file | 103.7 GiB |
| Resident weights | ~76.9 GiB |
| Table served from NVMe | ~26.8 GiB |
| Steady-state memory used | ~95 GiB |
| Process RSS | ~1.4 GiB |
| Load time from NVMe | ~3m35s |

The RSS figure is the point of the whole design: the table is served by the OS page cache from
NVMe, so it never becomes process-resident.

### The graph-reuse patch

With stock PR #27742, `graphs reused = 0` — the qsa/ple graph inputs never override `can_reuse()`,
so the graph is rebuilt and re-split every token and CUDA graph capture never engages. With
`patches/canreuse-qwen4exp.patch`:

| Metric | Before | After |
|---|---:|---:|
| Graphs reused | 0 | 1543 |
| Decode | baseline | +2.8% |

The measured effect is larger than +2.8% in practice: our prose figure (27.8 tok/s) sits above the
19–25 tok/s others report for the same quant on the same hardware, and warming the table stopped
mattering once graphs engaged.

**Reproduce:** run the same generation with and without the patch; compare the `graphs reused`
counter in the server log.

### Draft-model speculation

An external draft model was tested before MTP was available anywhere:

| Draft | Vocab | Mean accepted length | Decode |
|---|---:|---:|---|
| Qwen3.5-0.8B | 248320 (compatible) | 2.88 | ~23 tok/s — no speedup |

Acceptance was healthy; throughput did not move. On a top-10-of-512 MoE, *k* draft tokens activate
the **union** of experts across those positions, so weight traffic grows with *k*.

**This is no longer the last word on speculation.** The model's own trained MTP head *does* help —
about 1.16× on prose across engines — but only through vLLM, because the GGUF converter drops the head
(`supports_mtp_export = False`; we confirmed zero MTP tensors across all four shards, and all 31
present in the NVFP4 checkpoint). See [ruled-out.md](ruled-out.md#mtp-speculative-decoding--real-but-modest).

---

## Not measured

- **Editing recipe** above ~19k context, and its decode-at-depth curve.
- **IQ-series quants** on either recipe. Our Q3_K result says lower-bit K-quants are *slower*
  here, but independent reports show the IQ series behaving the opposite way. Different kernels.
- **Quality.** No eval suite has been run against either recipe. Every number here is speed.
- **Batch/offline throughput**, and concurrency beyond 8 simultaneous requests.
- **Vision throughput.** Both recipes now have a measured image-eval *score*
  ([../docs/vision.md](vision.md)), but no dedicated vision throughput benchmark: the numbers
  there are eval wall-clock at concurrency 4, not a tok/s figure.

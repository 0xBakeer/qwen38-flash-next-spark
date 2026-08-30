# All measurements

Every number in this repository, how it was taken, and what it does and does not support.

**Machine.** ASUS Ascent GX10 (NVIDIA GB10, SM121), 128 GB unified memory (~121 GiB usable),
~273 GB/s, Ubuntu 24.04.4, kernel 6.17.0-1029-nvidia aarch64, driver 580.173.02, CUDA 13.0,
swap off. Nothing else was contending for the GPU: the host's usual API endpoint was shut off for
the duration of every run.

**Model.** Qwen3.8-Flash-Next. 176,943,899,520 parameters total, of which 51,200,245,760 are the
n-gram/PLE table, leaving **125,743,653,760 of compute parameters** and ~6B active per token.
512 experts, top-10 routing, 48 blocks, 262,144 native context.

**Harness.** Both engines measured by the same script ([`bench/portable_bench.py`](../bench/portable_bench.py)),
which streams responses and times TTFT separately rather than trusting either engine's own
counters. Two details matter and are inherited from `bench/run_bench.py`:

- one warmup per task is discarded — the first request after any state change is markedly slower
- every repetition substitutes a **different token** into the prompt, because repeating an
  identical prompt lets the context-copying drafter draft from its own memory of the previous
  generation. Measured inflation when we did not do this: 60 → 169 tok/s.

Medians of 3, temperature 0.

---

## Decode by task shape

| Task | llama.cpp Q4_K_XL cold | llama.cpp Q4_K_XL warm | llama.cpp Q3_K_XL cold | vLLM NVFP4 MTP=3 |
|---|---|---|---|---|
| Reproduce a file with one change | **88.5** | 86.2 | 68.3 | 39.1 |
| Targeted bug fix | 46.1 | 42.2 | **48.8** | 35.0 |
| Add a function | 32.2 | 36.9 | **41.6** | 33.6 |
| Free-form prose (control) | 27.8 | 27.8 | 24.0 | **32.2** |
| **spread** | 3.2× | 3.1× | 2.8× | **1.2×** |

Cold = 17.7% of the n-gram table resident, verified with `mincore(2)` after dropping the page
cache. Warm = 58.1%. The Q3_K_XL column booted cold at 29.0% and was never warmed — immaterial,
since residency measurably changes nothing, but stated for completeness. All coding outputs verified correct 3/3 — the drafter is not trading
correctness for speed.

The spread row is the finding. Two speculation mechanisms, opposite characters:

- **`ngram-mod`** drafts from repetition in the prompt. Brilliant when the output already exists
  in the input, worth nothing on novel text.
- **MTP** drafts from a trained head. Works equally on anything, lower ceiling.

A single tok/s figure for this model, with no task attached, is not a meaningful number.

---

## Prefill

vLLM, NVFP4, 262,144 context. TTFT with `max_tokens=1`, so the figure is prompt ingestion alone.
Honest because prefix caching was disabled for every run here — nothing is served from cache.
It now ships **on** (`PREFIX_CACHE=1`); set `PREFIX_CACHE=0` to reproduce these numbers.
That was a correctness measure on the image these numbers were taken on, not a property of the
hardware; see [../recipes/vllm-longctx/README.md](../recipes/vllm-longctx/README.md#known-issues).

| prompt tokens | TTFT | prefill tok/s |
|---:|---:|---:|
| 2,542 | 1.14 s | 2,231 |
| 19,950 | 8.10 s | 2,463 |
| 79,637 | 33.53 s | 2,375 |
| 128,104 | 55.77 s | 2,297 |
| 195,458 | 89.55 s | **2,183** |

**Essentially flat: the 2.5k and 195k endpoints differ by 2%, and even the falloff from the
peak (2,463 at 20k) to 195k is only 11%.** For comparison, llama.cpp cold measures ~486 tok/s at
10.2k, 448 at 40.3k, and 253 at 161k — so vLLM is 4–5× faster at prefill *and* degrades less.

A caution about prefill numbers generally: an earlier run of ours reported ~1,500–1,660 tok/s for
llama.cpp, which was wrong. The benchmark used shared-prefix haystack prompts, so most requests
hit the server's slot cache. The giveaway was the spread — TTFT p50 of 477 ms against a max of
21 s. The honest cold rate is the slowest request, not the aggregate.

## Decode at depth

vLLM, same configuration.

| context depth | prompt tokens | decode tok/s |
|---|---:|---:|
| ~1k | 1,044 | 31.7 |
| ~32k | 31,338 | 33.5 |
| ~128k | 125,127 | 31.7 |

**Flat.** A published single-Spark llama.cpp measurement has decode falling 22.7 → 7.2 tok/s at
230k. This configuration does not do that, which is a second reason to prefer it for long
documents beyond the prefill advantage.

---

## Memory

vLLM at 262,144 context.

| component | size | format | notes |
|---|---:|---|---|
| checkpoint on disk | 126 GiB | — | 135,195,303,851 bytes, 206 shards |
| ├ n-gram / PLE table | 47.7 GiB | **FP8** | 10 shards. More precision than the experts get |
| ├ compute weights | ~63 GB | NVFP4 | experts, attention, GDN |
| └ MTP head, embeddings, norms | 15 GB | BF16 | |
| weights + non-torch, resident | 83.47 GiB | | vLLM's own accounting |
| KV cache pool | 18.13 GiB | | **641,601 tokens** = 2.4× the context |
| system in use | 111 / 121 GiB | | |

At the upstream default `GPU_MEM=0.78` the KV pool is only 10.82 GiB = 227,651 tokens, **less
than the model's own context window**. We serve at 0.85.

llama.cpp for contrast: 105 GB on disk at Q4_K_XL, ~77 GiB resident, peak RAM 102.9 GB, KV
~24 KB/token so the full 262k window costs about 6 GiB.

### One measurement we are not publishing

Our file-level residency tool totals 125.9 GiB resident across the vLLM checkpoint, which exceeds
physical RAM while `free` reports 2 GiB of buff/cache. The tool validates correctly on single
files — a control file reads 0.0%, goes to 100.0% when read, and the files are not sparse — but
that aggregate is unphysical and we do not trust it at scale.

What is defensible and reproducible: **the PLE table did not evict.** Dropping the page cache
left it reporting 100% resident, because the pages are mapped by the live process. Decode is
also insensitive to cache state. Both are consistent with the table being effectively
RAM-resident while vLLM serves, which is a real behavioural difference from llama.cpp — where
residency swung between 0% and 73% and had to be warmed deliberately.

---

## Thinking on vs off

Same prompt, same server, llama.cpp Q3_K_XL.

| | tokens generated | of which reasoning | decode tok/s | wall clock |
|---|---:|---:|---:|---:|
| thinking on | 1,600 | 1,373 (86%) | 29.4 | 55.1 s |
| thinking off | 349 | 0 | 24.1 | **15.0 s** |

Thinking-off is **slower per token and 3.7× faster to the answer.** This is the largest
real-world improvement measured anywhere in this repository, and it is not a speed optimisation.

It also means raw tok/s flatters a thinking model: reasoning traces are formulaic, so the
context-copying drafter drafts them *better* than genuine prose. A circulating claim that "think
tokens can never be drafted" does not reproduce here — we measure the opposite.

---

## The efficiency gap

Against DeepSeek-V4-Flash measured on the same box with the same workloads:

| configuration | active params | bits/weight | bytes/token | tok/s | bandwidth efficiency |
|---|---:|---:|---:|---:|---:|
| DeepSeek-V4-Flash, exl3 3.0bpw | 20.4B | 3.0 | 7.65 GB | 27.0 | **76%** |
| DeepSeek-V4-Flash, single-stream | 20.4B | 3.0 | 7.65 GB | 32.2 | **90%** |
| Qwen3.8-Flash-Next, llama.cpp | ~6B | 5.03 | 3.77 GB | 27.8 | **38%** |

**Qwen reads half the bytes per token and runs at the same speed.** At 27.8 tok/s the step is
~36 ms; moving 3.8 GB at 273 GB/s costs ~14 ms. The other **~22 ms is fixed overhead**.

It is not mixture-of-experts sparsity — DeepSeek is equally sparse (216 experts, top-6, 2.8%
activation against Qwen's 2.0%) and does not suffer it. See
[hardware-notes.md](hardware-notes.md) for where that 22 ms is likely going.

Note the quantization figure: `UD-Q4_K_XL` is nominally 4-bit but measures **5.03 bits/parameter**
(111,323,630,080 bytes over 176,943,899,520 parameters), because the dynamic scheme holds some
tensors higher.

---

## Where this sits against everyone else

Independently published single-Spark free-form figures, all within 48 hours of the model's
release:

| source | configuration | tok/s |
|---|---|---:|
| llama.cpp Q4_K_XL + ngram | various | 21 – 27.7 |
| IQ4_XS | llama.cpp | 27 |
| **this repo, edit recipe** | Q4_K_XL + ngram | **27.8** |
| IQ1_S | llama.cpp | 33 |
| llama.cpp, unspecified quant | — | 34.5 |
| **this repo, long-context recipe** | NVFP4 + MTP=3 | **32.2** |

**No verified report of 40+ tok/s free-form on a single Spark exists.** Figures above that in
circulation are dual-Spark rigs, or copy-heavy tasks quoted without the task named. Our own 88.5
is exactly that kind of number, which is why it is never quoted here without "reproduce a file
with one change" attached to it.

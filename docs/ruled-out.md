# What we ruled out

Things that should have made this faster and did not. Kept because a negative result closes off a
direction, and because several of them contradict advice that is circulating — including advice in
earlier versions of this repository.

---

## Warming the n-gram table — no effect

**Expected:** a large win. Earlier measurements in this repo showed cold 52.6 → warm 74.6 tok/s on
file reproduction, and up to +42% with speculation on.

**Measured on the current build:**

| task | cold (17.7% resident) | warm (58.1%) |
|---|---:|---:|
| Reproduce a file with one change | **88.5** | 86.2 |
| Targeted bug fix | **46.1** | 42.2 |
| Add a function | 32.2 | **36.9** |
| Free-form prose | 27.8 | 27.8 |

Read this as **no measurable difference**, not as "cold is faster". The medians move in both directions and the ranges overlap heavily — file reproduction is 78.6–109.1 cold against 78.1–114.2 warm. What did not happen is the +42% the earlier run reported.

**Why the change:** almost certainly the `canreuse-qwen4exp` patch. With CUDA graph capture
actually engaging, the per-token cost that page faults used to sit on top of is much smaller, so
residency stops mattering. Note also that our cold figure (88.5) now exceeds the old *warm* one
(74.6).

`tools/warm_table.py` is still shipped for A/B work. The boot-time warming ritual is not needed,
and as of 2026-08-29 `WARM` defaults to `0` in `recipes/llamacpp-edit/run.sh` — the code had been
doing it by default while this document said it was pointless.

Measured on this build, published in [inference-atlas](https://github.com/0xBakeer/inference-atlas):
the warmer cannot reach a useful residency at all. From a genuine cold start (0.06% resident) it
reads all 26.8 GiB at 1.01 GiB/s and lands at **25.9%**, because once the model load has taken its
share of the 121 GiB box the page cache has nowhere to put the rest. Decode at 25.88% is 34.12
tok/s against 34.99 and 37.43 for two runs at 0.06% — inside the spread of the cold runs, which
differ from each other by 6.5%.

Also ruled out, for the same reason: **holding the table at a chosen residency**. 18% established
before startup is evicted to 0.06% by the model load; 18% after startup reaches 11.43%; 58%
overshoots to 100%; and `fadvise(DONTNEED)` cannot go below 28% while the server maps the file.
There is no stable middle to measure.

---

## Lower-bit K-quants — slower, not faster

**Expected:** fewer bytes per token → proportionally faster decode.

**Measured:** `UD-Q3_K_XL` (90.0 GB) against `UD-Q4_K_XL` (111.3 GB) — 19% fewer bytes:

| task | Q4_K_XL | Q3_K_XL |
|---|---:|---:|
| Reproduce a file with one change | **88.5** | 68.3 |
| Free-form prose | **27.8** | 24.0 |

**14% slower on prose.** Q3_K dequantisation costs more per weight than Q4_K, and on a path where
~22 ms of every 36 ms step is already non-bandwidth overhead, you pay more in unpacking than you
save in memory traffic.

This is the cleanest evidence in the repository that **this configuration is not
bandwidth-bound.** If it were, fewer bytes would have been faster.

**Caveat, and it matters:** this result is specific to the **K-quant** series. Independent reports
on the **IQ** series show the opposite — IQ4_XS at 27 tok/s against IQ1_S at 33, i.e. 23% smaller
and 23% faster. IQ quants use different kernels. We did not test them. So "lower bit is slower" is
true for Q3_K here and should not be generalised.

Pick a smaller quant for disk space or to trade quality; do not pick one expecting speed.

---

## `VLLM_PLE_CPU_OFFLOAD=1` — hangs

**Expected:** the official upstream path for keeping the n-gram table in pinned host RAM instead
of paging it from NVMe, which should sidestep a documented GB10 pathology where small *pageable*
host→device copies run ~50× slower than pinned.

**Measured:** it never serves. The engine registers `PleOffload: registered 1 PleOffloadLayer(s)`
with an IPC endpoint, completes CUDA graph capture, and then spins.

Diagnosed rather than assumed: the EngineCore thread sits in state `R` burning a full core —
6,137 CPU ticks in 60 seconds — with **zero disk I/O**. That is a busy-wait, not a slow 44 GiB
load. It expects an offload worker process that the single-Spark image does not launch.

`VLLM_PLE_MMAP=1` (the container's own patch) is the only working path here. **The hypothesis is
untested, not disproven** — pinned host RAM may well help; it would need the upstream vLLM image
and whatever starts that worker.

---

## MTP speculative decoding — real, but modest

**Expected:** the 3–5× that speculative decoding delivers on dense models.

**Measured:** ~1.16× on prose across engines (27.8 → 32.2), and a flat 1.2× spread across all four
task shapes.

**Why, and this is architectural:** on a top-10-of-512 mixture-of-experts, verifying *k* draft
tokens activates the **union** of experts across those *k* positions. Published measurements put
that at 2–3× verification traffic growth. The union grows sublinearly, so speculation is weakened
rather than destroyed — but with only 10 of 512 experts hot, this model sits close to the worst
case among MoEs for the effect.

Practical consequence: **do not raise `MTP` above 3.** Adjacent evidence puts the optimum at 2–3
for 4-bit weights, and the expert-union tax plus acceptance decay both argue against deeper drafts.

---

## Turning thinking off — works, and is not a speed optimisation

The exception on this page: this one delivers, but not the way you would expect.

| | tokens | reasoning | decode tok/s | wall clock |
|---|---:|---:|---:|---:|
| thinking on | 1,600 | 1,373 (86%) | 29.4 | 55.1 s |
| thinking off | 349 | 0 | 24.1 | **15.0 s** |

**Slower per token, 3.7× faster to the answer.** The win comes from generating a quarter as many
tokens, not from generating them faster.

Two things follow. First, this is the largest real-world improvement measured anywhere in this
repository. Second, **raw tok/s flatters a thinking model** — reasoning traces are repetitive, so
the context-copying drafter drafts them *better* than genuine prose. A claim circulating that
"think tokens can never be drafted" does not reproduce here; we measure the opposite.

---

## Still open

**The per-token n-gram gather.** At 27.8 tok/s the step is ~36 ms against a ~14 ms bandwidth cost,
so ~22 ms is fixed overhead. The prime suspect is the gather running as a CPU op plus a *pageable*
host→device copy outside CUDA graphs — a pattern NVIDIA measures at ~50× slower than pinned on
this hardware, and which GB10's coherent address space should make unnecessary entirely, since the
GPU can dereference host memory directly.

There is direct precedent: in llama.cpp PR #25962, an embedding `get_rows` that had been evicted
from the graph went **6.18 → 1.72 ms/token** once a device-side kernel brought it back — on a much
smaller model.

Nobody has published this fix for GB10. It is an engineering task, not a flag change, and on the
arithmetic it is where the missing 22 ms lives. See [hardware-notes.md](hardware-notes.md).

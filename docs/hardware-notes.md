# Hardware notes — GB10, and where the missing performance is

This page is for anyone who wants to make this faster. It is the diagnostic work, not the recipe.

## The gap, stated precisely

Measured against DeepSeek-V4-Flash on the same box, same workloads:

| configuration | active params | bits/weight | bytes/token | tok/s | bandwidth efficiency |
|---|---:|---:|---:|---:|---:|
| DeepSeek-V4-Flash, exl3 3.0bpw | 20.4B | 3.0 | 7.65 GB | 27.0 | **76%** |
| DeepSeek-V4-Flash, single-stream | 20.4B | 3.0 | 7.65 GB | 32.2 | **90%** |
| Qwen3.8-Flash-Next, llama.cpp | ~6B | 5.03 | 3.77 GB | 27.8 | **38%** |

**Qwen reads half the bytes per token and runs at the same speed.** At 27.8 tok/s the step is
~36 ms. Moving 3.8 GB at ~273 GB/s costs ~14 ms. **The other ~22 ms is fixed, non-bandwidth
overhead.**

Two hypotheses die immediately:

- **Not mixture-of-experts sparsity.** DeepSeek is equally sparse — 216 experts, top-6, 2.8%
  activation, against Qwen's 512 / top-10 / 2.0% — and reaches 76–90%.
- **Not memory bandwidth.** If it were, dropping from Q4_K to Q3_K (19% fewer bytes) would have
  been faster. It was **14% slower**.

---

## The prime suspect: the per-token n-gram gather

Every generated token gathers ~16 rows out of 320,001,536 from the PLE table. In both recipes that
table is pinned to the CPU side and served from disk. The vLLM recipe documents the consequence
plainly: the gather is a CPU op followed by a **pageable** host→device copy, and it *must* run
outside CUDA graphs — which is why the serve script declares it a splitting op and captures
`PIECEWISE`.

So on every single token: a synchronising copy, and a graph break.

Three pieces of evidence that this is expensive on this specific hardware:

**1. Pageable small copies are pathological on GB10.** NVIDIA's own developer forum documents
small host→device copies from pageable memory running **~50× slower than pinned** on DGX Spark —
2,500 small copies taking 10.3 s against 0.2 s, and a real model load going 38.5 s → 2.6 s. A
per-token gather of a few KB from an mmapped region is exactly that shape.

**2. Graph breaks around an evicted gather are measurable in llama.cpp.** In PR #25962, a
quantized embedding `get_rows` that had been "kicked out of the graph" went **6.18 → 1.72
ms/token** once a device-side kernel brought it back — on a much smaller model than this one.

**3. Our own patch already proved the mechanism matters.** `canreuse-qwen4exp` exists because the
qwen4exp graph inputs never override `can_reuse()`, so the whole graph was rebuilt and re-split
every token and CUDA graph capture never engaged. Applying it is why our prose figure (27.8) sits
above the 19–25 tok/s others report for the same quant on the same box.

---

## Why the copy may not be needed at all

GB10 is a **hardware-coherent ATS platform**. CPU and GPU share one page-table domain over
NVLink-C2C, `cudaDevAttrPageableMemoryAccessUsesHostPageTables=1`, and the GPU can dereference
**system-allocated (`malloc`/`mmap`) memory directly** — no `cudaMemcpy`, no `cudaHostRegister`.

That means the "CPU op + pageable host→device copy, outside CUDA graphs" design is treating GB10
like a discrete-GPU system. On this hardware the copy is an artifact of the porting assumption,
not a requirement.

The shape of the fix: make the gather a **GPU kernel that reads the host-resident table in
place**. That would delete the CPU op, the pageable copy, the per-token graph break, and the CUDA
graph exclusion all at once. The reverse direction — CPU reading `cudaMalloc` memory coherently —
is *not* supported, but GPU-reads-host is the supported direction.

**Nobody has published this for GB10.** On the arithmetic it is where most of the 22 ms lives. It
is real engineering, not a flag, which is presumably why it is still unclaimed.

---

## Smaller things worth trying

**Pin the staging buffer.** If any copy path survives, `cudaHostRegister` it. Cheap insurance
against the 50× pathology, superseded by the zero-copy approach above.

**Rebase onto current llama.cpp master.** Upstream now fuses softmax+top-k MoE routing and GEMV;
NVIDIA measured an average **35% MoE uplift** on DGX Spark from those changes. Our build is from
a pre-merge PR commit that may predate or miss fusion coverage for the qwen4exp graph shape
(the PR has since merged; this has not been re-measured on master).

**Try the IQ quant series.** Our Q3_K result says lower-bit K-quants are slower here, but that is
a K-quant kernel property. Independent reports show IQ4_XS at 27 tok/s against IQ1_S at 33 — 23%
smaller, 23% faster. Different kernels, opposite behaviour. Untested by us.

**Watch for llama.cpp MTP.** The GGUF converter currently drops the model's trained draft head
(`supports_mtp_export = False`) — we confirmed zero MTP tensors across all four shards. Support is
work in progress upstream. llama.cpp's baseline here is *higher* than vLLM's without speculation,
so if the head lands on that baseline it becomes the interesting configuration.

---

## What will not help

**Deeper speculation.** On a top-10-of-512 MoE, verifying *k* draft tokens activates the union of
experts across those positions — published at 2–3× verification traffic growth. `MTP=3` is around
the optimum; above it, the expert-union tax and acceptance decay both bite.

**Expert prefetch and caching papers.** Nearly all 2026 work in that area targets *offloaded* MoE,
where PCIe transfer dominates. On GB10's unified memory with weights resident there is no offload
gap to hide.

**Lower-bit weights, on their own.** Covered above and in [ruled-out.md](ruled-out.md): you are
overhead-bound, not bandwidth-bound, so cutting bytes cuts the wrong cost.

---

## Platform quirks to know

- **Prefix caching is disabled in the long-context recipe**, and not for the reason this line
  used to give. It is not a GB10 kernel bug: vLLM's engine core overwrites
  `cache_config.block_size` with the smallest KV-group block size (16, from the QSA raw-key ring)
  while the Mamba state block is 1,600, so a prefix hit computed the wrong state slot and
  restored an all-zero Mamba state. That is the community image's diagnosis, and it is patched
  there in `8347e7c` (2026-08-29). **We could not reproduce the failure, and our first
  explanation for that was wrong**: the attention block is aligned up to 1,600, but a third KV
  group — the QSA raw-key ring, block size 8 at `MTP=3` — holds the `min()` at 8, so the
  mismatch is present. Identical outputs across three calls turned out to be the wrong
  observable; the recipe README has the reason. Upstream vLLM has no such group and is
  unaffected. Prefix caching now ships **on**, worth 1.76× aggregate
  throughput on a shared-prefix workload with no accuracy change. Every measurement published
  here was taken with it off, so the prefill figures are genuinely cache-free; `PREFIX_CACHE=0`
  reproduces that configuration.
- **Full `torch.compile` is unavailable** — an Inductor int64-indexing assert on sm_121.
- **FlashInfer's `cutedsl` backend does not support SM121.**
- **`free` accounting is confusing here.** Unified memory means "GPU" allocations and page cache
  come from the same pool, and mapped file pages do not appear where you expect. We have a
  file-residency measurement that totals more than physical RAM; it validates on single files but
  we do not trust its aggregate, and we say so rather than publishing it.

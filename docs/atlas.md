# What the atlas has measured

Every figure in this repository comes from `bench/portable_bench.py` — one harness, written
here, run by the people who wrote the recipes. That is a conflict of interest, so both recipes
have also been run through the pinned workloads of
[**inference-atlas**](https://github.com/0xBakeer/inference-atlas): a different harness, a
different prompt set, fixed workload definitions, and results that live as one JSON file per
measurement with the hardware captured rather than typed.

This page is the index. It says what has been measured there, what it changed here, and what
is still missing.

## Why it is worth having

The atlas is not a leaderboard and does not rank anything. It records *evidence*: a cell is a
configuration on a piece of hardware, and its colour says how well attested that cell is, not
how fast it is. For this repository the value is narrower and more useful — it is a second
opinion on our own numbers, taken with a harness we did not write.

Three things came out of it that this repository did not have before, and one that it had
wrong.

## 1. Page-cache residency is a hidden variable, and it moves

A quarter of the GGUF checkpoint is served from NVMe by design. How much of the 26.82 GiB
n-gram table happens to be in the page cache is therefore an independent variable in every
measurement — and nothing in the recipe was reporting it. Every atlas cell for the editing
recipe now carries a `mincore(2)` reading taken immediately before the workload.

What that produced:

| | |
|---|---|
| Residency at a real cold start | **0.06%** — not the ~2% previously assumed |
| After the recipe's own warmer, from cold | **25.9%** (reads 26.8 GiB at ~1.01 GiB/s) |
| Warming *before* the server starts | fully evicted: 18% → **0.06%** |
| After one 50-request workload | 0.06% → **54.75%** → 79.1% |

The last row is the important one: **a long workload warms itself**, so it spends most of its
own duration warm no matter where it started. And the third row is why `run.sh` defers its
warmer until after the server is serving — loading the model streams the whole GGUF through a
121 GiB box and evicts the table region on the way. That ordering fix came from
[@rumi-ali](https://github.com/rumi-ali) and the atlas runs are what confirmed it.

Decode at 25.88% residency measured 34.12 tok/s, which sits *inside* the 34.99–37.43 spread of
two otherwise identical cold runs. So residency is real and worth recording, and it is not
visible above the noise on this workload — both statements at once.

**The noise floor is 6.5%** on `serve-single`, established by repeating the same cell three
times. Nothing smaller than that should be read as a difference.

## 2. The long-context recipe was capped by a scheduler default

The upstream container ships `--max-num-seqs 2`. The atlas ran both arms of the A/B — same
workloads, same box, one variable — and the result changed what this repository ships:

- All four **concurrency-1** cells are identical between the arms. The cap costs nothing when
  the server is quiet.
- Every **loaded** cell is 1.2–2.7× better with it lifted, and time-to-first-token is where it
  shows: eight callers with 2k prompts wait **150.8 s** for a first token at two slots, **2.4 s**
  at sixty-four.
- The 64-caller cell at `SEQS=2` **lost 127 of 640 requests** to the workload's own timeout —
  the only cell in either arm that failed requests.

Full table and the reasoning: [../recipes/vllm-longctx/README.md](../recipes/vllm-longctx/README.md#sixteen-sequences-not-two-hundred).
The recipe now ships `SEQS=16`.

## 3. Vision, where the docs were simply wrong

This repository said images were a reason to choose the long-context recipe, because the GGUF
has no vision tensors. The tensors part is true; the conclusion was not. With the projector
Unsloth publishes alongside the quants, the editing recipe scores **0.967** on
`eval-vision-v1` — the same 58/60 as the NVFP4 checkpoint, with identical per-category splits.

[vision.md](vision.md) has the comparison, including the two items each engine misses.

## 4. Two slots cost nothing when nobody is waiting

The editing recipe forces `--parallel 1`. Running the same workloads at two slots, with the
vision projector loaded, gives a like-for-like pair at concurrency 1:

| workload | `--parallel 1` | `--parallel 2` + projector |
|---|---:|---:|
| `serve-single-i256-o256` decode | 34.12 / 34.99 / 37.43 tok/s | **35.02** |
| `serve-single` TTFT p50 | 1.05–1.10 s | **1.01 s** |
| `prefill-8k` prefill | 1,660.9 tok/s | **1,772.1** |

The two-slot decode figure lands in the middle of the three one-slot repeats — inside the 6.5%
noise floor, so the second slot and the 0.9 GiB projector cost nothing measurable when only one
caller is present. `prefill-8k` is 6.7% higher, but there is only one run on each side and no
repeat to bound prefill noise, so that one is suggestive, not established.

What this pair does **not** answer is whether two slots *batch* under load. That needs the
loaded workloads on both sides, which is running now.

## Where the numbers live

| Recipe | Cells |
|---|---|
| Editing (llama.cpp) | [`results/llamacpp/Qwen/Qwen3.8-Flash-Next/`](https://github.com/0xBakeer/inference-atlas/tree/main/results/llamacpp/Qwen/Qwen3.8-Flash-Next) |
| Long-context (vLLM) | [`results/vllm/Qwen/Qwen3.8-Flash-Next/`](https://github.com/0xBakeer/inference-atlas/tree/main/results/vllm/Qwen/Qwen3.8-Flash-Next) |

Each file records the engine build, the exact flags, the captured hardware fingerprint, the
workload definition, and a list of gotchas — including the ones that say a given cell is *not*
comparable with another.

## Where the two harnesses disagree

They mostly do not, but the differences are honest and worth stating rather than averaging away.

| | this repo | atlas |
|---|---:|---:|
| vLLM prefill, ~32k | 2,463 tok/s | 2,230 |
| vLLM prefill, ~128k | 2,297 tok/s | 2,057 |

Flat in both, which is the claim that matters. The atlas runs 5–10% lower throughout; the
prompt content, the warmup handling and `max-num-seqs` all differ. The honest range across both
is **~2,030–2,460**.

## Still missing

- Editing-recipe cells above 32k context, and its decode-at-depth curve.
- Any cell for a quant other than `UD-Q4_K_XL`.
- Anything measured on llama.cpp **master**. Every editing-recipe cell was taken on the
  pre-merge commit `035e227` plus the patches in `patches/`; the PR
  [merged on 2026-08-27](sources.md) and nothing has been re-measured since.
- A vision *throughput* number for either recipe. `eval-vision-v1` gives a score and a wall
  clock, not tok/s.

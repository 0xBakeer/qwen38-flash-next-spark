# Speculative decoding: what it does here, and how to measure it

`run.sh` enables `--spec-type ngram-mod` by default. This document explains why that changes
throughput by 3x on some tasks and not at all on others, and records several ways of measuring
it that produce numbers which look real but are not.

Everything here was measured on a DGX Spark (GB10, 121 GiB unified memory) serving
`unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q4_K_XL through llama.cpp PR #27742.

## The mechanism

`ngram-mod` builds an n-gram model from the tokens in the current context and uses it to draft
candidate spans. The target model then verifies the whole span in one step and accepts the
prefix it agrees with. Speculation is **exact**: any drafted token the target would not have
produced is rejected, so the output is identical to running without it. Only speed changes.

The consequence is that throughput is a function of **novelty**, not of "is this code":

| Task | Decode tok/s |
|---|---|
| Return a given file with one line changed | ~70 |
| Fix a bug in a file you were given | ~60 |
| Add a function to a file you were given | ~35 |
| Write prose | ~22 |

Adding a function means writing new lines, so it behaves much more like prose than like
reproducing a file. The question is not whether the task is code, it is how much of the output
already exists in the input.

## Why an external draft model does not help

Worth recording because it is the obvious thing to reach for. A vocabulary-compatible draft
model (Qwen3.5-0.8B, vocab 248320, same as the target) gave **no speedup at all**: mean accepted
length 2.88, decode unchanged at ~23 tok/s.

Speculative decoding normally works by amortizing one weight read across k drafted tokens. That
assumption breaks on a sparse MoE. This model routes top-10 of 512 experts per token, so k
tokens can activate up to k x 10 *different* experts, and weight traffic scales with k instead
of staying flat. There is nothing to amortize.

`ngram-mod` wins for a different reason: it accepts very long spans (50-60 tokens measured), so
it amortizes over the *verify step* rather than over the weight read.

MTP is not an option **in llama.cpp**. The GGUF converter sets `supports_mtp_export = False`
and drops the head, so no MTP tensors exist in any GGUF of this model (verified across all
1224 tensors).

> **Updated 2026-08-27.** The head itself is fine — it is the GGUF conversion that loses it.
> All 31 MTP tensors are present in the NVFP4 checkpoint, and vLLM runs them. Measured gain
> on prose: about 1.16× across engines (27.8 → 32.2), flat across task shapes rather than the
> 3× swing `ngram-mod`
> shows. Modest, because on a top-10-of-512 MoE verifying *k* draft tokens activates the
> union of experts across those positions. See
> [../recipes/vllm-longctx](../recipes/vllm-longctx/) and [ruled-out.md](ruled-out.md).
>
> An in-engine MTP-off A/B has since been reported by a third party, on their own DGX Spark
> and a different, non-public checkpoint
> ([#6](https://github.com/0xBakeer/qwen38-flash-next-spark/issues/6)): **+35% at one
> caller, not measurable at 16 concurrent** — their reading being that once the batch
> saturates the box, draft tokens have no idle capacity to convert. Their numbers, on their
> stack; details in the [long-context recipe README](../recipes/vllm-longctx/README.md).

## Four ways to measure this wrong

All four produced plausible numbers that were wrong — three here, one reported from outside.
They are listed because each is a natural thing to do.

### 1. Repeating an identical prompt

The most damaging one, because repeating a measurement is normally how you *remove* noise.

n-gram speculation drafts from repetition in the context. Send the same request twice and it
begins drafting from its own memory of the previous generation. Measured, temperature 0:

| | run 1 | run 2 | run 3 | run 4 |
|---|---|---|---|---|
| identical prompt each time | 59.8 | 59.8 | **169.0** | **166.8** |
| varied prompt each time | 61.7 | 70.3 | 26.8 | 64.1 |

It needs two observations before the n-grams are usable, which is why nothing happens until the
third run, and then it is about **2.8x**. More repetitions make it worse, not better.

The same effect is visible from a chat UI. Five identical requests, byte-identical prompts and
byte-identical answers (verified by md5), gave **43.2, 95.4, 100.3, 100.3, 100.2 tok/s**. The
honest number for that task is 43, not 100.

`bench/run_bench.py` defeats this by substituting a different token into the prompt on every
repetition, keeping the task identical in shape while denying the speculator a match.

### 2. Not discarding the first run

The first request after any state change is markedly slower. Measured at temperature 0 with
byte-identical output across eight runs:

```
run1: 132.8   run2: 172.9   run3: 172.7   run4: 171.4
run5: 171.4   run6: 171.3   run7: 170.7   run8: 170.4
```

Runs 2-8 are stable to about 1%. Run 1 is 23% low. Comparing two first-runs against each other
is what once produced a "cold page cache is faster than warm" result here, which is not a thing.

> **Still true, and worth reading twice — 2026-08-27.** A properly repeated A/B (n=3, warmup
> discarded) now finds **no measurable difference either way**: 88.5 cold against 86.2 warm on
> file reproduction, ranges 78.6–109.1 and 78.1–114.2, and prose identical at 27.8. The
> warning above is exactly why that is reported as "warming does nothing" rather than as
> "cold is faster" — those medians differ by less than the run-to-run spread.

### 3. Letting `max_tokens` run past the copied region

Throughput depends on the ratio of copied to generated tokens in the *output*. Ask for 900
tokens of a 1300-token file and it is still copying, so it stays fast. Ask for 1600 and it
finishes the file and continues into freshly generated commentary at ~22 tok/s, dragging the
average down. The same task measured 171 tok/s at one `max_tokens` and 52 at another.

### 4. Calling a difference from single runs

Identical single runs on this box spread by **6.5%** with nothing changed between them,
measured here under llama.cpp; a matching **6.9%** spread has been reported under vLLM on
another DGX Spark with a different quantization and drafter
([#6](https://github.com/0xBakeer/qwen38-flash-next-spark/issues/6) — their stack, not rerun
here). So two single runs that
differ by less than roughly 10% support no conclusion: either endpoint may just be the top or
bottom of its own spread.

This one has now claimed a casualty on each side. Here, both the original "+42% from warming"
and its first retraction were single-run comparisons inside that regime. Outside, a published
"k=2 is the optimum, k=3 is past it" was withdrawn by its own author in
[#6](https://github.com/0xBakeer/qwen38-flash-next-spark/issues/6) when k=2's 38.0 turned out
to be the top of its spread and k=3's 36.8 sat inside it, above the mean. Compare means of
repeated runs, or report the spread alongside the difference.

## Reasoning tokens hide all of this

If the model emits a think block, those tokens are generated, never drafted. This chat template
defaults `reasoning_effort` to `xhigh`, so a long think block runs at ~22 tok/s and averages with
whatever the copy phase is doing. Measured on the same task: **33 tok/s with reasoning at the
default, 43 with it off**, on a cold first run.

See [open-webui.md](open-webui.md) for turning it off.

## A prompt to test with

`bench/prompts/rate-limiter-edit.txt` is a ~130 line Python file plus an instruction to return it
with one class renamed. About 95% of the output is verbatim copy. Also published as a gist:
https://gist.github.com/0xBakeer/44a2a00783b7c08806833f8022ad0f04

Vary the class name (`RateLimiterA`, `RateLimiterB`, ...) if you run it more than once.

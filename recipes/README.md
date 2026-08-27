# Recipes

Two working setups, plus the ones we tried and rejected.

| | [llamacpp-edit](llamacpp-edit/) | [vllm-longctx](vllm-longctx/) |
|---|---|---|
| Engine | llama.cpp, PR #27742 + patches | vLLM 0.1.dev20073, NVFP4 |
| Speculation | `ngram-mod` — copies from your prompt | MTP — the model's trained draft head |
| Best at | rewriting files (**88.5 tok/s**) | prose (**32.2**), prefill (**~2,300**), long context |
| Worst at | prose (27.8), prefill (253–486) | file rewriting (39.1) |
| Consistency | 3.2× swing across tasks | **1.2× swing** |
| Concurrency | 1 | 2 |
| Disk / start-up | 105 GB / ~2 min | 126 GB / 12–15 min |

Full comparison: [../docs/choosing.md](../docs/choosing.md).
All figures: [../docs/measurements.md](../docs/measurements.md).

## How to run either

From the repository root:

```bash
./run.sh edit    setup && ./run.sh edit    serve
./run.sh longctx setup && ./run.sh longctx serve
```

Both then answer on `http://localhost:8000/v1` speaking the OpenAI API. Only one at a time —
they share the GPU.

---

## Configurations we tried and did not ship

Kept here so you do not spend a day rediscovering them. Details and numbers in
[../docs/ruled-out.md](../docs/ruled-out.md).

**llama.cpp with a warmed n-gram table.** Was the recommended path in earlier versions of this
repo. Re-measured on the current build it buys **nothing** — prose is 27.8 tok/s at both 17.7% and
58.1% residency, and cold beats warm on two of four tasks. The `canreuse-qwen4exp` patch appears
to have removed the sensitivity.

**llama.cpp at Q3_K_XL.** 19% fewer bytes per token, **14% slower on prose** (24.0 vs 27.8).
K-quant dequantisation costs more than the bandwidth it saves. Ship it if you need the 21 GB of
disk back, not for speed. (The IQ quant series reportedly behaves the opposite way — untested
here.)

**vLLM with `VLLM_PLE_CPU_OFFLOAD=1`.** The official pinned-host-RAM path. It **hangs** — registers
a `PleOffloadLayer`, finishes graph capture, then spins a full core with zero disk I/O
indefinitely, waiting on an offload worker the single-Spark image never launches. Only
`VLLM_PLE_MMAP` works here.

**vLLM at `MTP=0`.** Without the draft head vLLM is *slower* than llama.cpp at decode. The head is
the entire reason to run this recipe; do not disable it except for an A/B.

**Deeper speculation (`MTP` ≥ 4).** On a top-10-of-512 MoE, verifying *k* draft tokens activates
the union of experts across those positions, so the returns fall off hard. 3 is about the optimum.

---

## Something that beats both recipes

Turning thinking off. **86% of generated tokens were reasoning** in our measurements, and the same
answer arrived in **15 seconds instead of 55** — slower per token, four times fewer tokens.

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

Leave it on for hard reasoning and maths, off for everything else. It is a bigger practical
difference than the choice of recipe.

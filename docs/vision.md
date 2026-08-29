# Vision

Both recipes see images, and they score the same.

That is a change. Until 2026-08-30 this repository said image input was a reason to choose the
long-context recipe, because the GGUF "has no vision tensors at all". The first half of that is
true and the second half was the wrong conclusion: llama.cpp ships multimodal as a separate
projector, and Unsloth publishes one **in the same repository as the quants**. It is a 0.9 GiB
download next to a 104 GiB one, not a hunt through a third-party repo.

## What was measured

`eval-vision-v1` from the [inference atlas](https://github.com/0xBakeer/inference-atlas): 60
generated images — clock faces, dice, bar charts, arrows, grids, shape counts, and OCR panels —
scored by exact match against a known answer, at concurrency 4.

| | Editing recipe (GGUF + `mmproj`) | Long-context recipe (NVFP4) |
|---|---|---|
| Score | **58/60 — 0.967** | **58/60 — 0.967** |
| Wall clock, 60 items | 598 s | 233 s |
| Latency p50 / mean | 26.9 s / 39.0 s | 9.1 s / 15.2 s |
| Output tokens generated | 19,440 | 17,224 |
| Peak host RAM | 101.0 GB | 111.6 GB |
| Energy for the run | 6.38 Wh | 2.51 Wh |
| Slots serving concurrency 4 | 2 | 64 |

Per category, the two are indistinguishable — same totals, same splits:

| | arrows | chart | clock | dice | grid | ocr | shapes |
|---|---|---|---|---|---|---|---|
| Both recipes | 8/8 | 10/10 | 7/8 | 6/6 | 6/6 | 10/10 | 11/12 |

And by difficulty, both: easy 38/38, medium 13/14, hard 7/8.

## The identical score is not the same two failures

Both miss **vis-0012** (shapes, medium — "which quadrant holds the extra square"). Both fail it
the same way: the model reasons to the 2,048-token output cap and never emits an answer, so the
scorer sees an empty string. That is a truncation failure, not a perception failure, and it is
the same item on both engines.

The second miss differs. The editing recipe loses **vis-0053** (clock, 10:35); the long-context
recipe loses **vis-0050** (clock, 11:20). Each engine reads the clock the other one misses, and
both misses are again the 2,048-token cap rather than a wrong time. Clock faces are where this
model spends its tokens: on the editing recipe the eight clock items averaged 1,051 output tokens
against 155 for OCR.

So "0.967 on both" is a real result, and "the same 58 items" is not. With a sample of 60 and a
single differing item, the honest reading is that the projector costs no measurable accuracy —
not that the two paths are token-for-token equivalent.

## The speed gap is real but smaller than it looks

598 s against 233 s is 2.6×, and part of that is queueing, not compute: the eval runs at
concurrency 4, the long-context recipe had 64 slots available, and the editing recipe had **2**.
Roughly half of its requests were waiting for a slot rather than generating. The per-item
latencies inherit that — a p50 of 26.9 s is a queue plus a generation.

The other part is real. The editing recipe also generated 13% more tokens for the same 60
questions, at a lower per-token rate on this workload, and burned 2.5× the energy doing it.

If you need image input on this hardware, the choice is no longer "which recipe can" but
"how fast do you need it". The editing recipe can.

## Turning it on

`./run.sh setup` fetches `mmproj-F16.gguf` alongside the quants and `./run.sh serve` passes
`--mmproj` when it finds it. Nothing else changes. To opt out — or to check that a measurement
was text-only — set `MMPROJ=none`.

```bash
MMPROJ=none ./run.sh serve      # text-only, as every pre-2026-08-30 number here was measured
MMPROJ=/path/to/mmproj.gguf ./run.sh serve
```

Two things worth knowing before you compare numbers across it:

- **The projector is ~0.9 GiB of extra resident weights.** On a 121 GiB box that is not free.
  Peak host RAM in the run above was 101.0 GB.
- **A cell with the projector is not comparable to one without it.** The atlas result carries
  that as an explicit gotcha, and so should any figure you quote from a vision-enabled server.

## Caveats

- This was measured at `--parallel 2`, not the recipe's default of 1. The default is unchanged;
  two slots were used because the eval runs at concurrency 4 and one slot would have serialised
  it completely.
- `mmproj-BF16.gguf` also exists in the same repo. It has not been measured here.
- No vision *throughput* benchmark exists for either recipe. The wall-clock figures above are
  eval duration at concurrency 4, which is not a tok/s number and should not be quoted as one.

Source data: [inference-atlas results for this
model](https://github.com/0xBakeer/inference-atlas/tree/main/results).

# Credits

This repository is a thin layer over other people's work. What is genuinely ours is the
measurement methodology, the `canreuse-qwen4exp` and `rowband-ple-quant` patches, the serving
configurations, and the documentation. Everything below is somebody else's.

## The model

**[Qwen](https://qwen.ai)** — Qwen3.8-Flash-Next, and the tech report describing the n-gram / PLE
design that this whole approach rests on. The weights carry Qwen's own licence, which has
conditions of its own; read it before deploying commercially.

## The weights

**[Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)** — the dynamic GGUF quants
used by the editing recipe, and llama.cpp PR
[#27742](https://github.com/ggml-org/llama.cpp/pull/27742) which adds qwen4exp architecture
support. Without that PR none of this runs at all.

**[RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)** — the NVFP4 checkpoint
used by the long-context recipe, which retains all 31 MTP tensors and is therefore the only route
to the model's trained draft head on this hardware.

## Corrections from outside

**[Ali Naeini (@rumi-ali)](https://github.com/rumi-ali)** — reproduced this recipe end to end on
their own DGX Spark and reported four independent fixes in
[#1](https://github.com/0xBakeer/qwen38-flash-next-spark/pull/1), all of which landed.

The one that mattered most: the startup warm ran *before* `exec llama-server`, and loading the
model evicts the n-gram table as it streams the GGUF through the box — so the 26.8 GiB read was
discarded before it could help. They measured the table at 100% cached right after the warm and
1.9% by the time the server answered `/health`. That diagnosis was later confirmed here by a
different method (18% established before startup reads back as 0.06%), and it is why the warm is
now deferred until the server is serving.

Also theirs: the swap-enabled warning that could never print because procfs files always stat as
0 bytes; two benchmark entry points defaulting to the wrong port; and `run_bench.py` still
claiming concurrent requests abort the server after that had been corrected everywhere else.

Their commit is squashed into `93c457c`, which GitHub attributed to the repository owner on
merge; the work and the diagnosis are theirs.

**[Jürgen Schmied (@jschmied)](https://github.com/jschmied)** — three contributions in
[#6](https://github.com/0xBakeer/qwen38-flash-next-spark/issues/6), all measured on their own
DGX Spark with a non-public checkpoint, so the docs carry the first two as their results
rather than ours. They report replicating our 6.5% single-run noise floor at 6.9% under vLLM
with a different quantization and drafting mechanism — the claim that the ~10% single-run
limit belongs to the platform rather than to llama.cpp is theirs, made possible by their run.
They ran the in-engine MTP-off A/B the long-context recipe admitted it lacked (+35% at one caller,
not measurable at 16 concurrent — speculation stops paying once the batch saturates the box).
And they reported that `VLLM_TORCH_PROFILER_DIR` is inert in the Flash-Next preview build,
along with the working `--profiler-config` form and the systemd quoting trap. They also
withdrew one of their own published claims on endpoint-versus-spread grounds in the same
report, which is the methodology being used the way it was meant to be.

## The container that makes the long-context recipe possible

**[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)** (Apache-2.0) — the
patch that serves the 51.2B n-gram table from disk under vLLM, and the GB10-specific serving
configuration around it: the `PIECEWISE` CUDA-graph capture with the gather declared a splitting
op, and the workarounds for prefix caching and `torch.compile` on sm_121.

That patch is the reason a 122 GiB checkpoint fits next to a usable KV cache on one box. It is
**not vendored into this repository** — `recipes/vllm-longctx/setup.sh` clones and builds it from
source so the code stays under its own licence and its own authorship. Our contribution on top is
the serving configuration and the measurements.

Their published figures are also what we set out to verify. Where our numbers differ from theirs
we say so, and why.

## The engines

**[llama.cpp](https://github.com/ggml-org/llama.cpp)** — the editing recipe, and `ngram-mod`
speculation.

**[vLLM](https://github.com/vllm-project/vllm)** — the long-context recipe, MTP support, and the
`release/qwen38next` recipe branch.

## Measurements we compare against

Independent single-Spark figures published by other people within days of the model's release are
listed in [docs/measurements.md](docs/measurements.md). They are what let us say honestly where
this repository sits — including that our own free-form number is *not* the fastest published, and
that no verified 40+ tok/s free-form single-Spark result exists.

**DeepSeek-V4-Flash** numbers used in the efficiency comparison were measured on the same box with
the same workloads, and are what showed that this configuration runs at roughly half the bandwidth
efficiency it should.

---

MIT licensed, except where noted above. Third-party components keep their own licences.

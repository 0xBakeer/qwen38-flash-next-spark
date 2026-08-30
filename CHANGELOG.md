# Changelog

What a version means here: this repository is not a library, and nothing imports it. What you
depend on is **the defaults the recipes ship and the measurements taken on them**. So a release
is a measurement epoch — the configuration as it stood, and the figures that belong to it.

- **MAJOR** — a recipe is added or removed, or the measurement basis changes (new hardware, a
  different model).
- **MINOR** — a shipped default changes, or a recipe gains a capability. Your numbers move.
- **PATCH** — documentation, corrections, tooling. Your numbers do not move.

Every entry leads with **Defaults that changed**, because that is the part that alters what you
would measure if you ran the recipe yourself. `./run.sh` and `serve.sh` print the version they
were launched from.

## v0.2.0 — 2026-08-30

### Defaults that changed

| Recipe | Setting | Was | Now | What it costs you |
|---|---|---|---|---|
| llamacpp-edit | `--parallel` | 1 | **2** | context per request 262,144 → 131,072 |
| llamacpp-edit | `--mmproj` | absent | **fetched and loaded** | ~0.9 GiB more resident weights; image input works |
| vllm-longctx | prefix caching | off | **on** (`PREFIX_CACHE=1`) | prefill figures become cache-assisted; `PREFIX_CACHE=0` restores the measured configuration |

Every measurement published in this repository was taken **before** these flips: `PARALLEL=1`
and `PREFIX_CACHE=0` and no projector reproduce the configuration behind the numbers in
[docs/measurements.md](docs/measurements.md).

### Added

- **Vision on the editing recipe.** `mmproj-F16.gguf` comes from the same Unsloth repo as the
  quants. Scores **0.967** on the atlas image eval — the same 58/60 as the NVFP4 checkpoint,
  with identical per-category splits. [docs/vision.md](docs/vision.md).
- **[docs/parallel.md](docs/parallel.md)** — llama.cpp partitions `--ctx-size` across slots, so a
  slot count is a standing claim on context. The full 1→64 curve, and why 64 is not a setting.
- **[docs/atlas.md](docs/atlas.md)** — index of everything measured in the inference atlas and
  what it changed here.
- **The image records what it was built from.** `setup.sh` labels the vLLM image with the
  upstream repo, ref and short sha; `serve.sh` prints it. `UPSTREAM_REF` defaults to a moving
  branch, and prefix-caching behaviour differs across those commits.
- `REF=master` builds llama.cpp from the merged upstream code; `REF=pinned` (default) stays on
  the commit every number here was measured on.

### Changed

- **llama.cpp PR #27742 merged upstream** (2026-08-27). Both patches in `patches/` are now
  redundant there: master implements `can_reuse()` for the qwen4exp graph inputs and bounds the
  quantizer's staging buffer itself. They still apply to the pinned commit.
- Prefix caching, measured rather than assumed: **1.76×** aggregate and 2.55 s against 5.86 s
  first-token latency on a shared-prefix workload, accuracy unchanged.

### Corrected

Four claims this repository made that turned out not to be supportable. Each is kept in the docs
with what replaced it, because a repository whose corrections are invisible is not more
trustworthy than one that never corrects.

- "Concurrent requests abort the server" — they queue. Two stale copies survived an earlier fix,
  including the footer `bench/matrix.sh` stamps into every results file it writes.
- "The editing recipe cannot do images" — it can, at the same score.
- "Prefix caching must stay off: a GB10 GDN kernel bug" — no source was ever recorded for it. The
  real mechanism is a `block_size` mismatch diagnosed by [@blazux](https://github.com/blazux).
- "The mismatch does not arise in this configuration" — it does. A third KV cache group holds the
  minimum at 8, and our probe could not have detected the failure either way.

### Fixed

- The comparison table still advertised the long-context recipe at two concurrent sequences after
  it moved to sixteen.

## v0.1.0 — 2026-08-29

The repository as first published: two working recipes, the measurement methodology, and the
`SEQS=2 → 16` change that came out of the first atlas A/B.

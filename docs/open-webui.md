# Serving this to Open WebUI

Two things need setting up: pointing Open WebUI at the server, and turning reasoning off so the
speculation actually shows.

## Connecting

`llama-server` speaks the OpenAI API, so it goes in as an OpenAI connection:

**Settings → Connections → Manage OpenAI API Connections → +**, URL `http://<host>:8000/v1`,
any non-empty API key (llama-server ignores it unless you set `--api-key`).

Two things worth doing on the server side:

- **`--alias`.** `run.sh` passes `--alias qwen3.8-flash-next` by default. Without it,
  `llama-server` reports the full GGUF path as the model id, which reads badly in the picker and
  advertises your filesystem layout to every client that calls `/v1/models`.
- **`--parallel 2`.** The default in `run.sh` since 2026-08-30, raised from 1 because two slots
  cost nothing at one caller and are worth 1.24–1.30× under load. That is exactly the case a
  chat UI creates: a background title-generation call landing mid-generation now has somewhere
  to go instead of queueing behind you. The cost is context — 131,072 tokens per request rather
  than 262,144 — and `PARALLEL=1` gives the full window back. See
  [parallel.md](parallel.md).

**Turn off title generation, tag generation and follow-up suggestions** in
**Settings → Interface**. Each is a separate model call, and at `PARALLEL=1` a background
call landing while you are generating goes into the queue ahead of nothing and behind you —
harmless, but it is your throughput it is spending.

## Turning reasoning off

This matters more than it sounds. The Qwen3.8 chat template defaults `reasoning_effort` to
`xhigh`, so every answer opens with a long think block. Think tokens are generated one at a
time and can never be drafted by speculation, so they run at the slow rate and average with
whatever the fast copy phase is doing. Measured on the same edit task: **33 tok/s with the
default, 43 with reasoning off**, on a cold first run.

llama.cpp accepts the standard OpenAI field and maps it (`server-common.cpp`):

| `reasoning_effort` | Effect |
|---|---|
| `none` | `enable_thinking = false`, template emits an empty think block, no reasoning |
| `low` / `high` / `xhigh` | passed to the template as-is |
| unset | `xhigh` |

Open WebUI 0.11 forwards `reasoning_effort` as a model param (`utils/payload.py`), so a custom
model is enough. No server restart needed.

**Workspace → Models → +**, base model `qwen3.8-flash-next`, then under **Advanced Params** set
**Reasoning Effort** to `none`.

Server-wide alternative, if you would rather it be the default for every client:

```bash
EXTRA_ARGS='--chat-template-kwargs {"enable_thinking":false}' ./run.sh serve
```

## Reading the speed

Open WebUI shows generation stats on the **(i)** icon of any response. That is the number to
quote, not a stopwatch, because it comes from the server's own timings and excludes prefill.

One trap when reading them: **repeating an identical prompt inflates the figure**. n-gram
speculation starts drafting from its own memory of the previous answer, and after two identical
requests the number roughly doubles. Five identical requests measured 43.2, 95.4, 100.3, 100.3,
100.2 tok/s, with byte-identical prompts and answers throughout. The honest figure there is 43.

Change one token between runs if you want a number that reproduces for other people. See
[speculative-decoding.md](speculative-decoding.md).

## What to expect

| Task | Decode tok/s |
|---|---|
| Paste a file, ask for one small change | ~70 |
| Paste a file, ask for a bug fix | ~60 |
| Paste a file, ask for a new function | ~35 |
| Chat, essays, summaries, Q&A | ~22 |

If a chat response feels slow, that is the prose path and no setting will change it. The speed
comes from having something in the context worth copying.

# Choosing a setup

Two setups. Same model, same box, and they differ by more than a factor of two depending on what
you ask. This page is the longer version of the table in the README.

## The short answer

**A coding agent drives it and mostly rewrites files** → **Editing**
**You chat with it, ask questions, or feed it long documents** → **Writing & long documents**
**Not sure** → **Writing & long documents**

You can install both. Only one runs at a time — they share the GPU.

---

## Side by side

| | Editing | Writing & long documents |
|---|---|---|
| Engine | llama.cpp | vLLM |
| Weights | Unsloth GGUF, Q4_K_XL | RadixArk NVFP4 |
| How it guesses ahead | copies repetition from your prompt | a trained predictor inside the model |
| Rewriting a file with one change | **88.5** | 39.1 |
| Fixing a bug in a file | **46.1** | 35.0 |
| Adding a function | 32.2 | 33.6 |
| Writing prose | 27.8 | **32.2** |
| Consistency across those four | 3.2× swing | **1.2× swing** |
| Time to first token | ~1–2 s | **~0.3 s** |
| Reading a 128k-token document | ~7 min | **~56 s** |
| Speed at 128k context | degrades | **flat** |
| Simultaneous requests | 1 (others queue) | 2 |
| Disk | 105 GB | 126 GB |
| Start-up | ~2–4 min | 12–15 min |

---

## Why they differ so much

Both use *speculative decoding* — guessing several tokens ahead and checking them in one pass.
Both are exact: the model verifies every guess, so neither trades quality for speed. They differ
in **where the guesses come from**, and that changes everything.

**The Editing setup guesses from your prompt.** Hand it a file and ask for one change, and almost
the entire answer is already sitting in the question. It leaps 60 tokens at a time. That is the
88.5. Ask it to write something new and there is nothing to copy — it falls to its honest
unaccelerated rate, about 28.

**The Writing setup guesses from a trained predictor** shipped inside the model — a "multi-token
prediction" head. That works identically on any text, so the number barely moves whatever you ask.
Lower ceiling, far steadier floor.

The consequence worth internalising: **any speed claim about this model is meaningless without the
task attached.** A 97 tok/s headline and a 22 tok/s headline can both be true of the same box on
the same afternoon.

---

## The cases that decide it

**A coding agent rewriting whole files.** Editing, decisively — 88.5 against 39.1, more than
twice as fast on exactly that shape.

**A coding agent making small surgical diffs.** Closer than it looks. Editing wins on the bug-fix
shape (46.1 vs 35.0), but "add a function" is nearly a tie (32.2 vs 33.6) because new code is new
text with nothing to copy. If your agent sends large contexts every turn, the long-context
setup's 4–5× faster reading may matter more than decode.

**Chat.** Writing & long documents. Faster on prose, and a 0.3 s first token against 1–2 s makes
it feel far more responsive regardless of throughput.

**Summarising or asking about a long document.** Writing & long documents, and it is not close.
It ingests 195,458 tokens in 90 seconds and holds ~2,200 tok/s across the whole window; the
editing setup runs 253–486 tok/s and slows as context grows.

**Images.** Either, now — and this used to say otherwise. Both score **0.967** on the atlas
image eval, 58 of 60, with identical per-category splits. The NVFP4 checkpoint carries the vision
tower inline (333 tensors, 449M parameters, unquantized). The GGUF has no vision tensors at all,
but llama.cpp ships multimodal as a separate projector and Unsloth publishes one in the same
repository as the quants, so `./run.sh setup` fetches it and `--mmproj` loads it. Choose on speed
instead: the same 60 images take 233 s on *Writing & long documents* and 598 s on *File editing*,
part of which is the editing recipe serving a concurrency-4 eval from two slots. Details and the
two items each one misses: [vision.md](vision.md).

**More than one person or tool at once.** Writing & long documents, and by a wide margin: it
serves **16 concurrent requests** at 96–109 tok/s aggregate with first tokens still under
2.7 s. Neither setup is a multi-user server, though — past 16, time to first token goes to
16 s at 32 callers and 70 s at 64.

---

## One setting that beats both

The model thinks before answering, and **86% of generated tokens were reasoning** in our
measurements — tokens you pay for and never see.

Turning thinking off made the same answer arrive in **15 seconds instead of 55**. Tokens were
slightly *slower*; there were a quarter as many.

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

Leave it on for hard reasoning and maths. Turn it off for everything else. This is a larger
real-world difference than the choice between the two setups.

---

## What neither one does

**Neither reaches 40 tok/s writing prose.** 27.8 and 32.2. No published single-Spark configuration
does — the best independent figure anyone has posted is 34.5, and the higher numbers in
circulation are two-Spark rigs or copy-heavy tasks quoted without the task named.

If you need faster generative output than that, this box is not the answer today. See
[hardware-notes.md](hardware-notes.md) for where the remaining performance is and why it has not
been claimed yet.

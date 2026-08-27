# Qwen3.8-Flash-Next on a DGX Spark

Run a **180-billion-parameter model** on one desktop box with 128 GB of memory, at its full
262,144-token context.

The trick that makes it fit: 51.2B of the model's 176.9B parameters are a single lookup table
that is never multiplied by anything — only read from. It can live on your SSD instead of in
memory. That leaves 125.7B parameters of actual compute to fit in the box, which they do.

---

## Start here: which setup do you want?

There are two, and **they are good at genuinely different things.** Picking the wrong one costs
you a factor of two, so it is worth thirty seconds.

|                                            | **Editing** | **Writing & long documents** |
|--------------------------------------------|-------------|------------------------------|
| Best for                                    | coding agents, rewriting files, refactors | chat, reasoning, summarising big documents |
| Rewriting a file with one change            | **88 tokens/sec** | 39 |
| Fixing a bug in a file                      | **46** | 35 |
| Writing prose                               | 28 | **32** |
| Time to start answering (short prompt)      | ~1–2 s | **~0.3 s** |
| Reading a 128,000-token document            | ~4 min | **~56 s** |
| Speed at very long context                  | degrades | **stays flat** |
| Handles more than one request at a time     | no | yes, two |
| Disk needed                                 | 105 GB | 126 GB |

**Rule of thumb**

- A coding agent drives it, and it mostly rewrites files you give it → **Editing**
- You talk to it, ask it questions, or feed it long documents → **Writing & long documents**
- Not sure → start with **Writing & long documents**. It is the more even performer, and its
  fast document reading is the difference you will feel first.

You can install both. They use the same GPU, so only one runs at a time.

---

## Install

You need an NVIDIA DGX Spark or compatible GB10 box (128 GB unified memory), a recent driver
with Docker, and free disk space per the table above.

```bash
git clone https://github.com/0xBakeer/qwen38-flash-next-spark.git
cd qwen38-flash-next-spark
```

**Editing setup** — builds llama.cpp and downloads the model (~105 GB):

```bash
./run.sh edit setup
./run.sh edit serve
```

**Writing & long-document setup** — builds a container and downloads the model (~126 GB):

```bash
./run.sh longctx setup
./run.sh longctx serve
```

Either one then answers on `http://localhost:8000/v1`, speaking the OpenAI API, so anything that
talks to OpenAI talks to this: Open WebUI, coding agents, your own scripts.

First start takes a while — it is reading 100+ GB off disk. Later starts are faster.

Full instructions per setup:
[recipes/llamacpp-edit](recipes/llamacpp-edit/) · [recipes/vllm-longctx](recipes/vllm-longctx/)

---

## One setting worth knowing

The model **thinks before it answers**, and you pay for every thought token even though you never
see them. In our measurements **86% of generated tokens were reasoning.**

Turning thinking off made the same answer arrive in **15 seconds instead of 55** — not because
tokens got faster (they got slightly slower) but because there were far fewer of them.

Add this to your request:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

Leave thinking on for hard reasoning and maths. Turn it off for everything else.

---

## What the numbers actually mean

You will see very different speeds from the same model depending on what you ask it to do. That is
not noise — it is the whole story of how these setups work, and it is worth understanding before
you compare any two figures.

**The Editing setup drafts its guesses from your prompt.** When you paste a file and ask for one
change, almost every token of the answer already exists in the question, so it can leap ahead
60 tokens at a time and verify them in one pass. That is where 88 tokens/sec comes from. Ask it to
write something new and there is nothing to copy, so it falls back to about 28.

**The Writing setup drafts from a trained predictor** shipped inside the model. That works the same
on any kind of text, so it lands near 32–39 whatever you ask — much steadier, lower at the top end.

So: **a speed figure for this model is meaningless without the task attached to it.** Ours are all
published with the task named, in [docs/measurements.md](docs/measurements.md).

---

## Documentation

**Start here**
- [Choosing a setup](docs/choosing.md) — the longer version of the table above
- [How it works](docs/how-it-works.md) — the SSD trick and the memory arithmetic

**Measurements**
- [All measurements](docs/measurements.md) — every number, how it was taken
- [What we ruled out](docs/ruled-out.md) — five things that should have helped and did not
- [Benchmarks](docs/benchmarks.md) — reproducing them yourself

**Going deeper**
- [Hardware notes](docs/hardware-notes.md) — GB10 specifics, and where the remaining performance is
- [Speculative decoding](docs/speculative-decoding.md) — why throughput varies 3× by task
- [Open WebUI](docs/open-webui.md) — connecting it to a chat interface
- [Sources](docs/sources.md) — model, weights, upstream issues

---

## Credits

This builds on other people's work — the model, the quantisations, the engines, and in particular
the vLLM container that the long-context setup depends on. See [CREDITS.md](CREDITS.md).

MIT licensed. The model weights carry Qwen's own licence, which has conditions of its own.

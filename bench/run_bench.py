#!/usr/bin/env python3
"""Benchmark a running llama-server serving Qwen3.8-Flash-Next.

Measures three things, all from the server's own `timings` so prefill is never
counted as decode:

  1. decode / prefill vs prompt length
  2. decode vs task shape, which is what actually determines speed when
     speculation is on (ngram-mod drafts from repetition in the prompt)
  3. page-cache residency of the n-gram table, if --model is given

Sends one request at a time on purpose: concurrent requests currently abort the
server on this architecture (see the README).

Usage:
  ./run_bench.py --api http://127.0.0.1:8000 [--model /path/to/shard-00001-of-N.gguf]
                 [--out results.md] [--skip-ctx] [--skip-tasks]
"""
import argparse, json, os, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FILLER = "The following is an excerpt from a technical specification document. "

# A file to hand the model for the edit tasks. Self-referential on purpose: it
# keeps the benchmark self-contained and the file is a realistic size.
def sample_file():
    p = os.path.join(HERE, os.pardir, "tools", "page_cache.py")
    with open(p) as f:
        return f.read()


def post(api, prompt, max_tokens, temperature=1.0):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature, "top_p": 0.95, "top_k": 20,
    }).encode()
    req = urllib.request.Request(api + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    wall = time.time() - t0
    t = d.get("timings") or {}
    u = d.get("usage") or {}
    msg = d["choices"][0]["message"]
    return {
        "prompt_tokens": u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
        "decode": t.get("predicted_per_second") or 0.0,
        "prefill": t.get("prompt_per_second") or 0.0,
        "wall": wall,
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
    }


def table_cached(model):
    """Percent of the n-gram table currently in page cache, via tools/page_cache.py."""
    if not model:
        return None
    tool = os.path.join(HERE, os.pardir, "tools", "page_cache.py")
    try:
        out = subprocess.run([sys.executable, tool, model], capture_output=True,
                             text=True, timeout=300).stdout
        for tok in out.split():
            if tok.endswith("%"):
                return tok
    except Exception:
        pass
    return None


def bench_ctx(api, model, rows):
    print("\n## Decode and prefill vs context length\n")
    hdr = "| Prompt tokens | Prefill tok/s | Decode tok/s |"
    sep = "|---|---|---|"
    if model:
        hdr += " Table cached |"; sep += "---|"
    print(hdr); print(sep)
    for words in (40, 300, 1200, 4000, 12000):
        prompt = (FILLER * (words // 8 + 2))[: words * 5]
        r = post(api, prompt + " Summarize the above in 60 words.", 120)
        line = "| {:,} | {:.0f} | {:.2f} |".format(
            r["prompt_tokens"], r["prefill"], r["decode"])
        if model:
            line += " {} |".format(table_cached(model) or "n/a")
        print(line)
        rows.append(("ctx", r["prompt_tokens"], r["prefill"], r["decode"]))


def bench_tasks(api, rows, repeat=3, temp=0.0):
    src = sample_file()
    tasks = [
        ("Reproduce a file with one change",
         "Here is a Python file:\n\n```python\n" + src +
         "\n```\n\nReturn the ENTIRE file with one change: add a module-level constant "
         "PAGE_FLAG_@ = 1 near the top and use it instead of the literal 1 in the "
         "mincore bit test. Output only code.", "PAGE_FLAG_"),
        ("Targeted bug fix",
         "Here is a Python file:\n\n```python\n" + src +
         "\n```\n\nReturn the ENTIRE file with the page-size lookup made robust: fall back "
         "to 4096 if os.sysconf raises, and name the fallback constant FALLBACK_@. Output only code.", "4096"),
        ("Add a function",
         "Here is a Python file:\n\n```python\n" + src +
         "\n```\n\nReturn the ENTIRE file with an added function human(n) that formats a "
         "byte count as GiB, named human_@. Output only code.", "def human_"),
        ("Free-form prose (control)",
         "Write about 250 words explaining why consistent hashing reduces key movement "
         "when a node joins or leaves. Mention the number @ somewhere.", None),
    ]
    print("\n## Decode vs task shape\n")
    print("Speculation drafts from repetition in the prompt, so throughput tracks how much of")
    print("the output already appears in the input.\n")
    print("n={} per task, temperature {}. Median with min-max range.\n".format(repeat, temp))
    print("| Task | Decode tok/s (median) | Range | Correct |")
    print("|---|---|---|---|")
    for name, prompt, needle in tasks:
        speeds, oks = [], []
        # Two things this loop must avoid, both learned the hard way:
        #
        # 1. The first request after any state change is markedly slower
        #    (~133 vs ~171 tok/s), so one warmup is discarded.
        # 2. Repeating an IDENTICAL prompt lets ngram-mod draft from its own
        #    memory of the previous generation. Measured inflation: 60 -> 169
        #    tok/s from the third identical request onward. So every repetition
        #    substitutes a different token for "@", keeping the task identical
        #    in shape while defeating that reuse.
        post(api, prompt.replace("@", "WARM"), 1600, temperature=temp)
        for i in range(repeat):
            tag = "ABCDEFGHIJ"[i % 10] * 3
            r = post(api, prompt.replace("@", tag), 1600, temperature=temp)
            body = r["content"] or r["reasoning"]
            speeds.append(r["decode"])
            oks.append("n/a" if needle is None else ("yes" if needle in body else "no"))
        speeds.sort()
        med = speeds[len(speeds) // 2]
        ok = "n/a" if needle is None else "{}/{}".format(oks.count("yes"), len(oks))
        print("| {} | **{:.1f}** | {:.1f} - {:.1f} | {} |".format(
            name, med, speeds[0], speeds[-1], ok))
        rows.append(("task", name, med, ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None,
                    help="path to shard 00001 of the GGUF, to report table residency")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None, help="tag this run, e.g. cold / warm")
    ap.add_argument("--repeat", type=int, default=3,
                    help="repetitions per task; reports median and range")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="0 = greedy. Speculation acceptance depends on the sampled text, so\n                         at temp 1.0 a single run varies by +-30 percent and means little.")
    ap.add_argument("--skip-ctx", action="store_true")
    ap.add_argument("--skip-tasks", action="store_true")
    a = ap.parse_args()

    try:
        with urllib.request.urlopen(a.api + "/health", timeout=10) as h:
            if json.load(h).get("status") != "ok":
                sys.exit("server not ready at " + a.api)
    except Exception as e:
        sys.exit("no server at {}: {}".format(a.api, e))

    if a.out:
        sys.stdout = open(a.out, "w")

    ids = json.load(urllib.request.urlopen(a.api + "/v1/models", timeout=10))["data"]
    print("# Benchmark results{}\n".format(" - " + a.label if a.label else ""))
    print("Model id: `{}`".format(ids[0]["id"]))
    if a.model:
        c = table_cached(a.model)
        if c:
            print("\nn-gram table in page cache at start: **{}**".format(c))

    rows = []
    if not a.skip_ctx:
        bench_ctx(a.api, a.model, rows)
    if not a.skip_tasks:
        bench_tasks(a.api, rows, a.repeat, a.temp)
    print("\n---\nOne request at a time; concurrent requests abort the server (see README).")


if __name__ == "__main__":
    main()

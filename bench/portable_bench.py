#!/usr/bin/env python3
"""Engine-portable decode benchmark.

Measures decode tok/s the same way on llama.cpp and vLLM by streaming and timing
TTFT separately, instead of trusting llama.cpp's `timings` block (which vLLM does
not emit). Task set and the @-substitution methodology are taken from
qwen38-flash-next-spark/bench/run_bench.py so numbers stay comparable:

  - one warmup per task is discarded (first request after a state change is slower)
  - every repetition substitutes a different token for "@", so ngram speculation
    cannot draft from its memory of the previous identical generation
"""
import argparse, json, os, statistics, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def sample_file(path):
    with open(path) as f:
        return f.read()


def post_stream(api, model, prompt, max_tokens, temperature=0.0):
    """Returns (ttft_s, total_s, n_completion_tokens, text)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        api + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    ntok = 0
    usage_ct = None
    chunks = []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            u = d.get("usage")
            if u and u.get("completion_tokens"):
                usage_ct = u["completion_tokens"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or delta.get("reasoning")
                             or delta.get("reasoning_content") or "")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    ntok += 1
                    chunks.append(piece)
    total = time.time() - t0
    n = usage_ct or ntok
    return ttft or total, total, n, "".join(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default=".", help="where to write the JSON result")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--src", default=os.path.join(HERE, os.pardir, "tools", "page_cache.py"),
                    help="file handed to the model for the edit tasks")
    ap.add_argument("--max-tokens", type=int, default=1600)
    a = ap.parse_args()

    src = sample_file(a.src)
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

    print("# %s" % a.label)
    print()
    print("| Task | Decode tok/s (median) | Range | TTFT s | Correct |")
    print("|---|---|---|---|---|")
    results = []
    for name, prompt, needle in tasks:
        try:
            post_stream(a.api, a.model, prompt.replace("@", "WARM"), a.max_tokens, a.temp)
        except Exception as e:
            print("| %s | WARMUP FAILED: %s | | | |" % (name, str(e)[:60]))
            continue
        speeds, ttfts, oks = [], [], []
        for i in range(a.repeat):
            tag = "ABCDEFGHIJ"[i % 10] * 3
            try:
                ttft, total, n, text = post_stream(
                    a.api, a.model, prompt.replace("@", tag), a.max_tokens, a.temp)
            except Exception as e:
                print("| %s | FAILED: %s | | | |" % (name, str(e)[:60]))
                speeds = []
                break
            dec = (n - 1) / (total - ttft) if n > 1 and total > ttft else 0.0
            speeds.append(dec)
            ttfts.append(ttft)
            oks.append("n/a" if needle is None else ("yes" if needle in text else "no"))
        if not speeds:
            continue
        speeds_sorted = sorted(speeds)
        med = speeds_sorted[len(speeds_sorted) // 2]
        ok = "n/a" if needle is None else "%d/%d" % (oks.count("yes"), len(oks))
        print("| %s | **%.1f** | %.1f - %.1f | %.2f | %s |" % (
            name, med, speeds_sorted[0], speeds_sorted[-1],
            statistics.median(ttfts), ok))
        results.append({"task": name, "median_decode_tok_s": round(med, 2),
                        "min": round(speeds_sorted[0], 2), "max": round(speeds_sorted[-1], 2),
                        "ttft_s": round(statistics.median(ttfts), 3), "correct": ok})

    out = os.path.join(a.out_dir, "bench-%s.json" % a.label)
    with open(out, "w") as f:
        json.dump({"label": a.label, "api": a.api, "results": results}, f, indent=2)
    print()
    print("saved: %s" % out)


if __name__ == "__main__":
    main()

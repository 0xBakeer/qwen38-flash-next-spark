#!/usr/bin/env python3
"""Pull the per_layer_token_embd region of a (possibly split) GGUF into page cache.

Rationale: the n-gram table is 320M rows addressed by a 3-gram hash, so a workload almost
never touches the same row twice early on -- it never warms naturally, even when the table
would fit in cache entirely. Reading it once sequentially is far cheaper than paying ~10
major faults/token forever.

Usage: plewarm.py <model.gguf> [tensor_name]
"""
import sys, os, importlib.util, time

spec = importlib.util.spec_from_file_location("pc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "page_cache.py"))
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)

path = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else "per_layer_token_embd.weight"

hit = None
for sh in pc.shards_for(path):
    try: off, length = pc.read_gguf_tensor(sh, want)
    except Exception: continue
    if off is not None: hit = (sh, off, length); break
if hit is None:
    print(f"tensor {want} not found"); sys.exit(1)
sh, off, length = hit

c0, n0, pg = pc.resident(sh, off, length)
print(f"before: {100*c0/n0:5.1f}% of {length/2**30:.1f} GiB cached")

fd = os.open(sh, os.O_RDONLY)
# advise first (async kernel readahead), then force with sequential reads
try: os.posix_fadvise(fd, off, length, os.POSIX_FADV_WILLNEED)
except Exception: pass
CH = 64 << 20
t0 = time.time(); done = 0
while done < length:
    n = min(CH, length - done)
    b = os.pread(fd, n, off + done)
    if not b: break
    done += len(b)
os.close(fd)
el = time.time() - t0

c1, n1, pg = pc.resident(sh, off, length)
print(f"after:  {100*c1/n1:5.1f}% cached  ({done/2**30:.1f} GiB read in {el:.1f}s = {done/2**30/max(el,0.01):.2f} GiB/s)")

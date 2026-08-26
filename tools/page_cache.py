#!/usr/bin/env python3
"""Page-cache residency of the per_layer_token_embd region of a GGUF, via mincore(2).
No root needed. Usage: plecache.py <model.gguf> [tensor_name]"""
import sys, os, mmap, ctypes, struct, re, glob

def read_gguf_tensor(path, want):
    f = open(path, "rb")
    magic, ver = struct.unpack("<II", f.read(8))
    assert magic == 0x46554747, f"not a gguf: {magic:x}"
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
    def rd_str():
        (n,) = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8","replace")
    def skip_val(t):
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t == 8: rd_str()
        elif t == 9:
            (et,) = struct.unpack("<I", f.read(4)); (n,) = struct.unpack("<Q", f.read(8))
            for _ in range(n): skip_val(et)
        else: f.read(sz[t])
    align = 32
    for _ in range(n_kv):
        k = rd_str(); (t,) = struct.unpack("<I", f.read(4))
        if k == "general.alignment":
            (vt,) = (t,); (align,) = struct.unpack("<I", f.read(4))
        else: skip_val(t)
    infos = []
    for _ in range(n_tensors):
        name = rd_str(); (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack(f"<{nd}Q", f.read(8*nd))
        (typ,) = struct.unpack("<I", f.read(4)); (off,) = struct.unpack("<Q", f.read(8))
        infos.append((name, dims, typ, off))
    data_start = f.tell()
    if data_start % align: data_start += align - (data_start % align)
    f.close()
    infos.sort(key=lambda x: x[3])
    for i,(name,dims,typ,off) in enumerate(infos):
        if name == want:
            end = infos[i+1][3] if i+1 < len(infos) else os.path.getsize(path)-data_start
            return data_start+off, end-off
    return None, None

def resident(path, off, length):
    # Fraction of the byte range [off, off+length) that is in page cache.
    # Maps via libc directly: a read-only Python mmap object refuses to hand out a
    # raw pointer (ctypes.from_buffer demands a writable buffer), and mincore needs one.
    pg = os.sysconf("SC_PAGE_SIZE")
    base = (off // pg) * pg
    span = (off - base) + length
    PROT_READ, MAP_SHARED = 1, 1
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mmap.restype = ctypes.c_void_p
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    fd = os.open(path, os.O_RDONLY)
    addr = libc.mmap(None, span, PROT_READ, MAP_SHARED, fd, base)
    if addr in (None, 0, 2**64 - 1):
        os.close(fd)
        raise OSError(ctypes.get_errno(), "mmap failed")
    npages = (span + pg - 1) // pg
    vec = (ctypes.c_ubyte * npages)()
    rc = libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(span), vec)
    if rc != 0:
        libc.munmap(ctypes.c_void_p(addr), span)
        os.close(fd)
        raise OSError(ctypes.get_errno(), "mincore failed")
    cached = sum(1 for b in vec if b & 1)
    libc.munmap(ctypes.c_void_p(addr), span)
    os.close(fd)
    return cached, npages, pg

def shards_for(path):
    """A split GGUF keeps its own tensor table per shard, and shard 1 may hold none.
    Given any shard, return every shard of the set. Plain string parsing: no regex."""
    d, base = os.path.dirname(path), os.path.basename(path)
    if "-of-" not in base or not base.endswith(".gguf"):
        return [path]
    head, _, tail = base.rpartition("-of-")
    stem = head.rsplit("-", 1)[0]
    total = tail[:-len(".gguf")]
    hits = sorted(glob.glob(os.path.join(d, stem + "-*-of-" + total + ".gguf")))
    return hits or [path]
if __name__ == "__main__":
    path = sys.argv[1]; want = sys.argv[2] if len(sys.argv)>2 else "per_layer_token_embd.weight"
    hit = None
    for sh in shards_for(path):
        try: off, length = read_gguf_tensor(sh, want)
        except Exception: continue
        if off is not None: hit = (sh, off, length); break
    if hit is None: print(f"    tensor {want} not found in any shard of {path}"); sys.exit(1)
    sh, off, length = hit
    c,n,pg = resident(sh, off, length)
    print(f"{want}: region {length/2**30:.1f} GiB  cached {c*pg/2**30:.1f} GiB / {n*pg/2**30:.1f} GiB  = {100*c/n:.1f}%  [{os.path.basename(sh)}]")

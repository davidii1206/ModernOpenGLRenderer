#!/usr/bin/env python3
"""Render a MIP4 sparse mip-chain atlas to a BMP for visual inspection.

Usage:
    mip4_to_bmp.py chain.mip4 [--rects patch_table.bin] [options]

Reconstructs the chain exactly like a GPU would (root -> child texture fetches)
for every sampled atlas texel and writes a dependency-free 24-bit BMP.

Examples:
    # Grayscale depth (channel 0 = dmin), auto-scaled to ~2048 px wide
    python3 mip4_to_bmp.py atlas_depth.bin --rects patch_table.bin

    # UV as a colour map (R = u, G = v) at full texel resolution
    python3 mip4_to_bmp.py atlas_uv.bin --rects patch_table.bin \
        --rgb --scale 1 --out uv.bmp

    # Thickness at coarse resolution, patch boundaries drawn
    python3 mip4_to_bmp.py atlas_thickness.bin --rects patch_table.bin \
        --patch-boundaries

    # 8x nearest-neighbour upscale so a small atlas is easy to inspect
    python3 mip4_to_bmp.py atlas_depth.bin --rects patch_table.bin --upscale 8
"""
import argparse
import struct
import sys

MIP_MAGIC = 0x4D495034
PATC_MAGIC = 0x50415443
MIP_MAX_TEXW = 16384


class Error(Exception):
    pass


def load_mip4(path):
    with open(path, "rb") as f:
        raw = f.read()
    pos = 0

    def u32():
        nonlocal pos
        v = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        return v

    def u64():
        nonlocal pos
        v = struct.unpack_from("<Q", raw, pos)[0]
        pos += 8
        return v

    def floats(n):
        nonlocal pos
        v = struct.unpack_from("<%df" % n, raw, pos)
        pos += 4 * n
        return list(v)

    if u32() != MIP_MAGIC:
        raise Error("bad magic (not a MIP4 file)")
    ver = u32()
    channels = u32()
    bpc = u32()
    num_levels = u32()
    leaf_tile = u32()
    atlas_w = u32()
    atlas_h = u32()
    num_patches = u32()
    qmin = floats(channels)
    qmax = floats(channels)
    if bpc != 1:
        raise Error(f"unsupported bytes-per-channel {bpc}")
    levels = []
    for _ in range(num_levels):
        w = u32()
        h = u32()
        do = u64()
        ds = u64()
        mo = u64()
        ms = u64()
        if mo + ms > len(raw) or do + ds > len(raw):
            raise Error("level data out of bounds")
        levels.append({"w": w, "h": h, "do": do, "ds": ds, "mo": mo, "ms": ms})
    return dict(version=ver, channels=channels, num_levels=num_levels,
                leaf_tile=leaf_tile, atlas_w=atlas_w, atlas_h=atlas_h,
                num_patches=num_patches, qmin=qmin, qmax=qmax, levels=levels, raw=raw)


def read_patch_table(path):
    with open(path, "rb") as f:
        raw = f.read()
    pos = 0

    def u32():
        nonlocal pos
        v = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        return v

    if u32() != PATC_MAGIC:
        raise Error("bad patch_table magic")
    version = u32()
    if version not in (1, 2):
        raise Error("unsupported patch_table version")
    n = u32()
    if u32() != n * 12:
        raise Error("bad aabb_min section")
    pos += n * 12
    if u32() != n * 12:
        raise Error("bad aabb_max section")
    pos += n * 12
    if u32() != n:
        raise Error("bad axis section")
    pos += n
    if u32() != n * 4:
        raise Error("bad texwh section")
    pos += n * 4
    if u32() != n * 16:
        raise Error("bad rects section")
    rects = []
    for _ in range(n):
        rects.append(struct.unpack_from("<4I", raw, pos))
        pos += 16
    double_sided = []
    if version >= 2:
        if u32() != n:
            raise Error("bad double_sided section")
        double_sided = [b != 0 for b in raw[pos:pos + n]]
        pos += n
    return rects, double_sided


def descend(m, patch_id, x, y, N, stop_at):
    """Walk the chain to the node containing lattice point (x, y); return
    (leaf_level, leaf_index, raw_bytes, decoded_values, covered)."""
    L = 0
    x0 = y0 = 0
    s = N
    idx = patch_id
    raw, levels = m["raw"], m["levels"]
    while True:
        meta = struct.unpack_from("<I", raw, levels[L]["mo"] + 4 * idx)[0]
        if not (meta & 0x40000000) or s <= stop_at:
            break
        hs = s >> 1
        cx = 1 if x >= x0 + hs else 0
        cy = 1 if y >= y0 + hs else 0
        idx = (meta & 0x3FFFFFFF) + cy * 2 + cx
        x0 += cx * hs
        y0 += cy * hs
        s = hs
        L += 1
    covered = bool(meta & 0x80000000)
    ch = m["channels"]
    vals = struct.unpack_from("<%dB" % ch, raw, levels[L]["do"] + idx * ch)
    dec = []
    for c in range(ch):
        lo, hi = m["qmin"][c], m["qmax"][c]
        b = vals[c]
        if b == 0 or not (hi > lo):
            dec.append(0.0)
        else:
            dec.append(lo + (hi - lo) * (b - 1) / 254.0)
    return L, idx, vals, dec, covered


def to_byte(v, lo, hi):
    if not (hi > lo):
        return 0
    t = (v - lo) / (hi - lo)
    return int(max(0.0, min(1.0, t)) * 254.0 + 0.5) + 1


def upscale(rows, W, H, k):
    """Nearest-neighbour upscale: repeat every texel k times in both axes."""
    if k <= 1:
        return rows, W, H
    nW, nH = W * k, H * k
    nrows = []
    for y in range(H):
        row = rows[y]
        nrow = []
        for x in range(W):
            nrow.extend([row[x]] * k)
        nrows.extend([nrow] * k)
    return nrows, nW, nH


def write_bmp(path, w, h, rows):
    """rows: h lists of w (r,g,b) tuples. 24-bit BGR, bottom-up, 4-byte padded."""
    row_size = (w * 3 + 3) & ~3
    data_size = row_size * h
    with open(path, "wb") as f:
        f.write(b"BM")
        f.write(struct.pack("<IHHI", 54 + data_size, 0, 0, 54))
        f.write(struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, data_size, 0, 0, 0, 0))
        pad = b"\x00" * (row_size - w * 3)
        for y in range(h - 1, -1, -1):
            row = rows[y]
            for r, g, b in row:
                f.write(bytes((b, g, r)))
            if pad:
                f.write(pad)


def render(m, rects, scale, channel, rgb, patch_lines):
    aw, ah = m["atlas_w"], m["atlas_h"]
    npat = len(rects)
    if m["num_patches"] != npat:
        raise Error(f"patch_table has {npat} patches, chain says {m['num_patches']}")
    if channel >= m["channels"]:
        raise Error(f"chain has {m['channels']} channels, --channel {channel} invalid")
    if rgb and m["channels"] < 2:
        raise Error("--rgb needs a 2-channel chain")

    W = (aw + scale - 1) // scale
    H = (ah + scale - 1) // scale
    raw, levels, leaf = m["raw"], m["levels"], m["leaf_tile"]

    rows = [[(0, 0, 0)] * W for _ in range(H)]
    for pid, (ax, ay, tw, th) in enumerate(rects):
        N = 1
        while N < max(tw, th):
            N <<= 1
        i0 = (ax + scale - 1) // scale
        j0 = (ay + scale - 1) // scale
        i1 = min((ax + tw - 1) // scale, W - 1)
        j1 = min((ay + th - 1) // scale, H - 1)
        for j in range(j0, j1 + 1):
            row = rows[j]
            ty = j * scale
            for i in range(i0, i1 + 1):
                tx = i * scale
                if tx < ax or ty < ay:
                    continue
                if tx >= ax + tw or ty >= ay + th:
                    continue
                _, _, rawb, _, covered = descend(m, pid, tx - ax, ty - ay, N, leaf)
                if not covered:
                    continue
                if rgb:
                    row[i] = (rawb[0], rawb[1], 0)
                else:
                    v = rawb[channel]
                    row[i] = (v, v, v)
        if pid and pid % 1000 == 0:
            print(f"  patch {pid}/{npat} ...", flush=True)

    if patch_lines:
        for ax, ay, tw, th in rects:
            i0 = max((ax + scale - 1) // scale, 0)
            j0 = max((ay + scale - 1) // scale, 0)
            i1 = min((ax + tw - 1) // scale, W - 1)
            j1 = min((ay + th - 1) // scale, H - 1)
            for i in range(i0, i1 + 1):
                for j in (j0, j1):
                    rows[j][i] = (255, 0, 0)
            for j in range(j0, j1 + 1):
                for i in (i0, i1):
                    rows[j][i] = (255, 0, 0)
    return rows, W, H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chain", help="atlas_*.bin (MIP4 chain)")
    ap.add_argument("--rects", metavar="PATCH_TABLE", default=None,
                    help="patch_table.bin (required; maps atlas texels to patch roots)")
    ap.add_argument("--scale", type=int, default=None,
                    help="sample every N atlas texels (default: auto from --max-width)")
    ap.add_argument("--upscale", type=int, default=1,
                    help="render each sampled texel as an NxN block (nearest-neighbour)")
    ap.add_argument("--max-width", type=int, default=2048,
                    help="auto-scale so the image is ~this many texels wide")
    ap.add_argument("--channel", type=int, default=0,
                    help="channel to render as grayscale (default 0)")
    ap.add_argument("--rgb", action="store_true",
                    help="2-channel chains: R = channel 0, G = channel 1")
    ap.add_argument("--patch-boundaries", action="store_true",
                    help="overlay red patch boundary lines")
    ap.add_argument("--out", default=None, help="output .bmp (default: chain.bmp)")
    args = ap.parse_args(argv)

    m = load_mip4(args.chain)
    if not args.rects:
        raise Error("--rects patch_table.bin is required (to map texels to patch roots)")
    rects, _double_sided = read_patch_table(args.rects)
    if args.scale is None:
        args.scale = max(1, (m["atlas_w"] + args.max_width - 1) // args.max_width)
    print(f"{args.chain}: {m['atlas_w']}x{m['atlas_h']} atlas, "
          f"{m['channels']} channel(s), {len(m['levels'])} levels, "
          f"{len(rects)} patches, scale={args.scale}")

    rows, W, H = render(m, rects, args.scale, args.channel, args.rgb,
                        args.patch_boundaries)
    if args.upscale > 1:
        rows, W, H = upscale(rows, W, H, args.upscale)
    out = args.out or (args.chain.rsplit(".", 1)[0] + ".bmp")
    write_bmp(out, W, H, rows)
    print(f"wrote {out} ({W}x{H})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

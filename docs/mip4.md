# MIP4 — Adaptive Sparse Mip-Chain Atlas Format

Produced by the CPU-only `gfx::CoverageAtlas` API (`gllib/gfx/coverage_atlas.hpp`, driven by
example 31 `examples/31_mesh_decomposition`) for Sponza-class meshes.
Replaces the old flat float32 atlas dumps with four self-describing, GPU-traversable
sparse mip chains (~30 MB total vs ~3 GB of float texels). The four chains share
one quadtree topology.

Each `atlas_*.bin` is one chain over a shared quadtree. Level 0 has one node per
patch, at the same index the patch occupies in `patch_table.bin`, `patch_tris.bin`,
and the BVH (**pack order**, not patch id). A node subdivides 1→4 while it is
**partially covered** (silhouette refinement) or while the covered region's
depth / thickness / UV range exceeds a small tolerance. Fully covered and fully
empty regions stay as leaves, so coverage edges refine down to `leaf_tile`
texels while flat regions stay coarse and detailed regions stay fine. The
finest node size is `config.mip_leaf_tile` (default 4; set to 1 to store
per-texel values wherever the mesh subdivides).

```
atlas_depth.bin      RG16 per node: (dmin, dmax)   quantised to [gdmin, gdmax]
atlas_thickness.bin  R8   per node: thmax           quantised to [0, gthmax]
atlas_uv.bin         RG8  per node: (uavg, vavg)    quantised per channel
atlas_normal.bin     RG8  per node: (nxavg, nyavg)  octahedral normal, q range [-1,1]
```

The depth chain uses **16-bit** channels (256× the discrete levels of RG8) so
runtime gradient reconstruction no longer drowns in quantisation noise on
shallow diagonal patches. The normal chain stores a baked interpolated vertex
normal per texel (octahedral-encoded, `safe_normalize` + `octahedral_encode`)
aggregated per node; the ray pass fetches it directly instead of re-deriving a
normal from quantised depth neighbours.

## File layout (all little-endian)

### Header

| offset | size | field |
|-------:|-----:|-------|
| 0      | u32  | magic `0x4D495034` ("MIP4") |
| 4      | u32  | version (2) |
| 8      | u32  | channels (1 = R8, 2 = RG8/RG16) |
| 12     | u32  | bytes per channel (1 = 8-bit, 2 = 16-bit depth) |
| 16     | u32  | num_levels |
| 20     | u32  | leaf_tile (4 = finest node size in texels) |
| 24     | u32  | atlas_w |
| 28     | u32  | atlas_h |
| 32     | u32  | num_patches |
| 36     | f32×channels | qmin per channel |
| 36+4c | f32×channels | qmax per channel |
| 36+8c | 40×num_levels | level records (below) |

Header size = `36 + 8*channels + 40*num_levels` bytes.

Each level record (40 bytes):

| field | type |
|-------|------|
| w     | u32  — value/meta texture width  |
| h     | u32  — texture height (`ceil(n/w)`) |
| data_off | u64 — byte offset of value data |
| data_size | u64 — `w*h*channels*bpc` bytes |
| meta_off | u64 — byte offset of meta data |
| meta_size | u64 — `w*h*4` bytes |

### Level data

Node *i* of a level lives at texel `(i % w, i / w)` of that level's pair of
plain GPU textures:

* **Value texture** — `data_off..+data_size`, `R8`, `RG8` or `RG16`, `bpc` bytes
  per channel:
  - 8-bit: `byte = 1 + round(clamp((v - qmin[c]) / (qmax[c] - qmin[c]), 0, 1) * 254)`,
    decoded as `v = qmin[c] + (qmax[c] - qmin[c]) * (byte - 1) / 254`.
  - 16-bit (depth chain): `uint16 = 1 + round(clamp((v - qmin[c]) / (qmax[c] - qmin[c]), 0, 1) * 65534)`,
    stored little-endian, decoded as `v = qmin[c] + (qmax[c] - qmin[c]) * (uint16 - 1) / 65534`.
  A `0` value (byte or uint16) is reserved for an **empty node** (no covered
  texels); every covered node stores `1..255` / `1..65535`, so a valid value is
  never `0`/black and consumers can test `value != 0` (or meta bit 31) before
  decoding.
* **Meta texture** — `meta_off..+meta_size`, one `u32` per node (`GL_R32UI`):

  | bit(s) | meaning |
  |--------|---------|
  | 31 | `has_data` — region contains rasterised texels |
  | 30 | `subdivided` — node has 4 children |
  | 0–29 | child offset — index of the first child in the **next** level |

  A subdivided node's four children are contiguous in the next level at
  `child .. child+3`, in quadrant order `(x,y)`, `(x+hs,y)`, `(x,y+hs)`,
  `(x+hs,y+hs)`.

## GPU-style traversal

Root node index = patch index (level 0). Root size `N = next_pow2(max(tex_w, tex_h))`
of the patch's atlas rect. To fetch the value at patch-local texel `(u,v)`:

```
idx = patch_index; s = N; x = y = 0; L = 0
while meta[idx] has bit30 set and s > leaf_tile:
    hs = s / 2
    cx = u >= x + hs; cy = v >= y + hs
    idx = (meta[idx] & 0x3FFFFFFF) + cy*2 + cx
    x += cx*hs; y += cy*hs; s = hs; L += 1
value = data[L][idx]            # channels*bpc bytes, decoded with qmin/qmax
```

Depth is a coarse–fine chain of native texture fetches — no stack, no per-level
structure.

## Tolerances / constants

| constant | value | meaning |
|----------|-------|---------|
| `MIP_LEAF_TILE` (config `mip_leaf_tile`) | 4 | finest node size (texels; 1 = per-texel) |
| `MIP_TOL_FRAC` (config `mip_tol_frac`) | 0.004 | relative range tolerance |
| `MIP_MAX_LEVELS` | 12 | tree depth cap |
| `MIP_MAX_TEXW` | 16384 | level texture width cap |

`mip_leaf_tile`, `mip_tol_frac`, `texel_density`, `min_tex`, `max_tex` and
`auto_target` are exposed on `gfx::CoverageAtlasConfig` (see
`coverage_atlas.hpp`) so the resolution and detail can be chosen per model
rather than being fixed at compile time.

A node stays a leaf (stops subdividing) when it is **not partially covered** —
`count == 0` (empty) or `count == s²` (fully covered) — and its aggregate no
longer *exceeds*:

```
(dmax - dmin) > 0.004 * (gdmax - gdmin)        # depth span
thmax           > 0.004 * gthmax                # thickness
(umax-umin) > 0.004 * (gmaxUV - gminUV)         # u span
(vmax-vmin) > 0.004 * (gmaxUV - gminUV)         # v span
```

`count` is the number of rasterised texels inside the node. Partial coverage forces
subdivision regardless of value uniformity, so `has_data` is only coarse at 4×4
leaf granularity and never loses covered texels (conservative silhouette).

## Reference data

| chain | channels | bpc | nodes (Sponza) | file size |
|-------|---------:|----:|---------------:|----------:|
| depth | 2 | 2 | 1,815,700 | 16.03 MB |
| thickness | 1 | 1 | 1,815,700 | 8.66 MB |
| uv | 2 | 1 | 1,815,700 | 10.39 MB |
| normal | 2 | 1 | 1,815,700 | 10.39 MB |

9 levels (L0 4,576 → L8 568,848 nodes). The four chains share one topology.
With partial-coverage refinement the chain runs ~18% more nodes / ~22% larger than
value-refinement alone, in exchange for sharp coverage silhouettes (over-reporting
confined to 4×4 leaves).

## Tools

* `tools/mip4_to_bmp.py chain.mip4 --rects patch_table.bin` — renders a chain to a
  BMP for visual inspection (see `--help`).
* Structure/validation scripts used during development live in the repo's build dir
  history and `/tmp/opencode/validate_mip4.py`, `/tmp/opencode/traverse_mip4.py`.

# Splat Pass Performance Plan (Stage E)

**Current state**: `splat ≈ 9-11ms` for 92M vertex-shader invocations
(90,000 gather pixels × 1024 nodes at `micro_res_scale=4`, `splat_lod=10`).
`setup=0.03ms`, `sum=0.10ms` are cheap. FS cost is minor. The VS is lean
(~5 SSBO reads + ~30 ALU) but ~50% of vertices are culled *inside* the VS
after paying the launch cost. Cost is fundamentally `pixels × 2^LOD`.

## Goal
Reduce splat time at minimal visual cost, keeping real GPU rasterization +
hardware depth sort (no return to compute traversal).

## Plan (by impact)

### 1. Default tuning — big, instant win (2–4×)
- `micro_res_scale` 4 → 6: 90K → 40K gather pixels → splat ≈4.4ms.
  Bilateral upsampling (4×4 samples) largely hides the lower-res gather.
- `splat_lod` 10 → 9 (512 nodes): halves node count → ≈5ms.
- Combined: ≈10ms → ≈2.2ms. Verify visually and pick values; both already
  have live sliders.

### 2. VS micro-optimizations (~10–25%)
- Precompute per-instance tile coords `(tx,ty)` in `cam_setup` (store 2 floats
  in `cams`), removing the per-vertex `gidx % rw` / `gidx / rw`.
- Hoist atlas NDC scale/offset into precomputed uniforms
  (`u_ndc_scale = 2/u_atlas_size`, `u_ndc_off`), replacing per-vertex divides.
- Skip `gl_PointSize = max(2*r_px,1)` clamp when `r_px < 0.5` (sub-pixel).
- No visual change.

### 3. Sub-pixel far-field culling + convolution accumulation (quality-preserving)
- VS culls splats with `r_px < 0.5` (they rasterize as 1px points — pure
  waste). Saves point setup + FS for ~half the surviving splats.
- `micro_sum_cs` re-loops the level-L nodes per gather pixel, recomputes
  `r_px`, and adds the culled (far-field) nodes' `rad × π·r_px² × warp-Jacobian`
  at their projected micro-pixel position — additive, so no energy is lost.
- Keeps LOD-10 quality; expected net ~1–2ms.

### 4. Bucketed per-pixel adaptive LOD (follow-up, 2–4× at fixed quality)
- `cam_setup` computes a per-pixel LOD (heuristic: needed granularity grows
  with proximity to other geometry).
- Compaction pass buckets gather-pixel indices per LOD level.
- One `glDrawArraysInstancedBaseInstance` per used LOD, with a per-instance
  pixel-index lookup — pixels that only need coarse LOD stop launching fine
  nodes entirely. This is the only option that reduces VS *launches* at
  constant quality. Scene-dependent gains; significant complexity.

### Not recommended
- Compute-based splatting via imageAtomics (depth-sorted packed key) would be
  ~2-3× faster but reintroduces the compute traversal that was intentionally
  replaced with real rasterization.

## Suggested execution order
1. Add temporary counters (total VS, per-cull-reason, surviving, sub-pixel)
   to quantify headroom.
2. Apply §2 (micro-opts) — safe, measure.
3. Apply §3 (sub-pixel cull + far-field accumulation) — measure.
4. Apply §1 (defaults) — visually verify.
5. Defer §4 unless more speed is needed at full quality.
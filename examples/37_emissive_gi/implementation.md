# Example 37 — Point-Sampled Emissive Lighting

Micro-rendering for scalable, parallel final gathering  ×  Ray tracing point sampled geometry.

- **A**: Ritschel, Engelhardt, Grosch, Seidel, Kautz, Dachsbacher.
  *Micro-Rendering for Scalable, Parallel Final Gathering.* ACM TOG 28(5):132, SIGGRAPH Asia 2009.
  Paper text + notes: `../../micro_rendering.md`; author PDF: `https://jankautz.com/publications/microrenderingSIGAsia09.pdf`
- **B**: Schaufler & Jensen. *Ray Tracing Point Sampled Geometry.* Rendering Techniques 2000
  (11th EG Workshop on Rendering), pp. 319–328.
  PDF: `https://cseweb.ucsd.edu/~henrik/papers/ray_tracing_point_sampled_geometry/ray_tracing_point_sampled_geometry_egwr2000.pdf`

> **Naming note.** The title "Ray Tracing Point Sampled Geometry" belongs to Schaufler & Jensen 2000.
> The frequently-misattributed "Wand et al. 2001" paper does not exist; the continuous-LOD/footprint
> algorithm people usually mean is Wand & Straßer, *Multi-Resolution Point-Sample Raytracing* (GI 2003),
> which is *not* used here. Example 37 follows Schaufler & Jensen 2000 (octree + ray-cylinder pruning +
> interpolated point disks).

This doc is the living map. Each stage is independently testable/visualizable before moving on.

---

## 1. Design summary

Combine the two techniques into **one lighting pipeline**:

- The scene is represented as **point-sampled geometry (surfels)** stored in a **BVH8 octree**
  (8-ary bounding-sphere hierarchy — Schaufler & Jensen's structure, and the modern GPU answer to
  latency-bound binary trees: depth ≈ log₈ N, SIMD-friendly child tests, cache-line-sized nodes).
- **Micro-rendering (paper A)** rasterizes this hierarchy into per-gather-point micro-buffers to
  compute final gathering of *indirect* radiance.
- **Ray tracing point-sampled geometry (paper B)** traverses the *same* octree with ray-cylinder
  pruning and interpolated point-disk intersections, used for: the on-demand post-traversal fallback,
  **directional-light soft shadow rays**, and optional **glossy specular reflection rays**.
- **Emissive-forward lighting** (our deviation from paper A): there is **no direct forward lighting
  pass on the points**. Emissive materials are the light sources — leaf surfels on emissive surfaces
  carry emitted radiance, pulled up the hierarchy. The **only** forward lighting is a single
  **directional light**, shaded in the composite pass against point-traced soft shadows.

### Per-frame pipeline

| Pass | What it does | Where |
|---|---|---|
| 1. Forward G-buffer | position / normal / albedo / emissive / linear depth (mesh raster) | `gbuf` shaders |
| 2. Octree update | leaf re-eval from (tri, barycentric), bottom-up 8-way refit | compute |
| 3. Radiance update | emissive leaves seed radiance; bottom-up pull-up; optional re-emission bounces | compute |
| 4. Micro-render gather | solid-angle cut, shared-memory micro-buffer, GGX warp, S-J fallback | compute |
| 5. S-J ray pass | directional soft-shadow rays (+ optional specular rays) | compute |
| 6. Composite | directional light × point shadow + gathered indirect + emissive self (+ bilateral upsample) + tonemap | fullscreen |

---

## 2. Why a BVH8 octree with packed SoA records

The original paper ran at ~150M gather samples/s on a GTX 285 (~0.7–1 TFLOPS, 159 GB/s) — a ~1–3%
FLOP efficiency. The workload is **latency/divergence/bandwidth-bound tree traversal**, not FLOP-bound,
so FLOP scaling barely transfers. The fix is a *layout* fix:

| Problem | Layout response |
|---|---|
| 14–20 dependent loads per micro-pixel (log₂ N) | 8-ary tree → depth ≈ 5–7 |
| 32-byte node records (ex36) streamed from DRAM | packed records (~8–16 B), fp16/fixed-point + octahedral normals |
| Reject test touches whole node | **SoA**: first slab (`childBounds`) alone decides reject → fewer cache lines per test |
| Per-thread private micro-buffer arrays (ex36) → register spill | shared-memory micro-buffer, one 8×8 workgroup per gather point, small per-thread stack |
| Random access pattern | DFS/pre-order layout (children contiguous), Morton/Z-order dispatch, L2-resident trees for N ≤ 64k |
| S-J "all points within r of ray" queries | dense cache-line-aligned **leaf blocks**; cylinder query = scan one contiguous block |

Working budget (1080p, Cornell-scale, N≈16–64k — to validate in S9):

| Mode | 680M (iGPU) | 3060 Mobile | 3070 Desktop |
|---|---|---|---|
| Full-res 8×8 gather | 15–40 fps | 150–300 fps | 250–500 fps |
| Full-res 16×16 | 3–8 fps | 40–90 fps | 60–150 fps |
| Preview (1/16 res + bilateral, 8×8) | 60–140 fps | 300–600 fps | 500–900 fps |

680M reality check: no RT hardware and ~half the GTX 285's bandwidth (shared system RAM), so the
interactive target on the iGPU is **preview mode**, never full-res high-quality.

---

## 3. Data structures

### 3.1 Surfels (leaf payload)

One surfel per scene surface element. `LeafBlock` = fixed-size (e.g. 64) surfel array, SoA-padded to
cache lines.

| Field | Storage | Bytes |
|---|---|---|
| position | scene-space quantized 16-bit ×3 (linear map over scene AABB) | 6 |
| normal | octahedral (sx, sy) 16-bit ×2 | 4 |
| radius | fp16 | 2 |
| albedo | RGB8 (linear) | 3 |
| emissive radiance | RGBA16F in separate radiance buffer (not per-surfel struct) | — |
| (tri idx, barycentric u,v) | CPU-side `LeafSource`, not uploaded per-surfel | — |
| **≈ packed surfel** | | **15 B** (~16 B padded) |

Positions are **not** re-quantized per frame; leaves are re-evaluated from barycentric coords then
quantized on write.

### 3.2 Octree (BVH8)

Complete 8-ary sphere tree, N = 8ᵏ leaves (pad with degenerate/zero-radius surfels to a power of 8).
Implicit indexing: children of node *i* at `8i+1 … 8i+8` → **no child-pointer bytes**.

SoA slabs (SSBOs):

| Slab | Contents | Size |
|---|---|---|
| `nodeBounds` | interior: 8 × (fp16 pos + fp16 radius) | 64 B/node |
| `nodeCone` | node normal-cone: octahedral axis (16×2) + half-angle (8-bit) | 5 B → pad 8 |
| `leafBlocks` | dense surfel blocks (see 3.1) | ~16 B/surfel |
| `radiance` | RGBA16F per node (leaf emissive → pull-up) | 8 B/node |
| `leafSource` | CPU: (tri, u, v) per leaf | fixed |

Interior nodes store a **normal cone** (axis + half-angle) for back-face culling during both
traversals (QSplat/micro-rendering convention).

### 3.3 Buffer list

| Buffer | Written by |
|---|---|
| `NodeBounds`, `NodeCone` | CPU build → GPU refit |
| `LeafBlocks` | CPU build / GPU leaf update |
| `Radiance` | emissive seed + pull-up |
| `LeafSource` | CPU (fixed) |
| `GatherPoint` (pos/normal/albedo) | G-buffer pass |
| `MicroOutput` (low-res indirect) | micro-render pass |
| `ShadowTex`, `SpecularTex` | S-J ray pass |
| `OutputImage` | composite |

---

## 4. Traversal A — micro-render gather (paper A)

One workgroup per gather point; 8×8 local threads = one micro-pixel each.

1. Load gather point (jittered within tile if upsampling): world pos `p`, normal `n`, albedo.
2. Build tangent frame `T,B`; pole = `warp_dir = mix(n, reflect(-V,n), 1-roughness)` (importance warp;
   roughness=1 ⇒ cosine-weighted hemisphere).
3. **Cut traversal** (iterative DFS, shared across the micro-pixels is a stretch-goal optimization;
   baseline = per-micro-pixel DFS with small stack):
   For each node: project bounding sphere center `c` into micro space → pixel `(x,y) = Φ⁻¹(ω)`;
   compute subtended solid angle `Ω` vs. per-pixel solid angle `Ω(x,y)`.
   - `Ω > Ω(x,y)` ⇒ recurse into ≤ 8 children.
   - else ⇒ rasterize: depth-test node into micro-buffer slot `(x,y)` (atomic min in shared memory).
   - back-face cull with normal cone + hemisphere test (`dot(c-p, n) <= 0` skip).
4. **On-demand S-J fallback** (paper A's post-traversal list): leaves that still cover >1 micro-pixel
   are stored in a bounded list; after the cut, each unresolved micro-pixel casts a ray and resolves it
   against that list using **interpolated point-disk** intersection (Traversal B).
5. **Convolve**: sum micro-buffer radiance × BRDF weight (or plain sum when warp table used) →
   write `MicroOutput`.

### Cut criterion (paper A, verbatim)

> "we first compute the direction ωᵢ to the node's center, the solid angle Ωᵢ that it subtends, and the
> pixel (xᵢ,yᵢ) = Φ⁻¹(ωᵢ) … If Ωᵢ > Ω(Φ⁻¹(ωᵢ)), then the node is larger than 1 pixel under the
> projective mapping, and we will proceed with the child-nodes. Otherwise, we perform a depth test."

---

## 5. Traversal B — ray tracing point-sampled geometry (paper B)

For an arbitrary ray `R(o, d, tmin, tmax, ε)` where `ε` = cylinder radius (softness):

1. Traverse the octree: node is rejected unless its bounding sphere intersects the **ray cylinder**
   (ray expanded by `ε`).
2. At leaf blocks: for each surfel, compute perpendicular distance `h` from the surfel center to the
   ray axis; if `h ≤ radius` (or `≤ ε`), it's a candidate.
3. **Interpolated hit**: weight candidates by a falloff on `h` (and on distance along the ray), blend
   position, normal, and radiance → smooth, hole-free surface (paper B's disk interpolation).
4. Return hit (position, normal, radiance) **and** an opacity/softness factor:
   - **Shadow rays**: `ε` = softness × surfel radius → occlusion = 1 − interpolated opacity.
     Wider ε ⇒ softer shadows.
   - **Specular rays**: trace 1 reflection ray per glossy gather point; ε from BRDF cone.

Same octree as the gather; no separate structure.

---

## 6. Emissive-forward lighting (our deviation)

- **No point-level direct lighting.** Paper A's "direct lighting of the point hierarchy" pass is
  replaced by: emissive leaves write their emitted radiance into `Radiance`.
- **Pull-up**: bottom-up 8-way average of child radiance → interior nodes carry approximate radiance
  for coarse-level micro-render cutoffs and cheap multi-bounce.
- **Bounces (toggle 0/1/2)**: optionally gather `Radiance` into `Radiance` again (re-emission from
  lit surfels) before the final pull-up. Multi-bounce uses the same micro-render traversal at node
  level (paper A's radiosity application).
- **The only forward light is the directional light**, applied in the composite pass:

```
color = albedo * (directional_Lambert * pointShadow)   // forward-only term
      + indirect                                       // micro-render gather of emissive radiance
      + emissive_self                                   // from G-buffer
```

---

## 7. Build order

| Stage | Deliverable / validation | Status |
|---|---|---|
| S1 | Scaffold: window, camera, hot-reload shaders, model load + toggle (CornellBox/Sponza). Mesh on screen. | ✅ |
| S2 | Forward G-buffer (pos/normal/albedo/emissive/depth) + display pass. Visualize each target. | ✅ |
| S3 | CPU surfel sampling (best-candidate) + BVH8 build + packed SoA upload + point-cloud & sphere debug overlays. | ✅ |
| S4 | GPU refit (leaf re-eval + 8-way bottom-up) + emissive radiance seed + pull-up; bounce toggle. | ✅ |
| S5 | Micro-render gather, fixed cosine mapping, full-res. Validate: wall opposite emissive panel is lit. | ✅ |
| S6 | S-J interpolated-disk fallback (post-traversal list). Compare with/without (paper A Fig. 11). | |
| S7 | S-J soft shadow rays for the directional light + composite. Soft shadows vs. hard shadow map. | |
| S8 | Glossy specular reflection rays (GGX warp + 1 ray). | |
| S9 | Bilateral upsampling + quality knobs; GPU timers; validate the perf table in §2. | 🔶 (upsampling + gather-res slider + timers done; see perf pass above) |

### S3 status (done) — notes for later stages

- **Sampling**: grid-accelerated best-candidate (Mitchell 1991), `K = 24` candidates/sample,
  uniform grid (cell = expected spacing) with expanding-ring nearest search capped at 6 cells plus
  a `stop_at` early-prune against the running best candidate. Cornell box, N=32768 builds in
  **~0.5 s** (sampling dominates).
- **Tree**: complete 8-ary (BVH8), implicit children `8i+1..8i+8`, N = 8ᵏ exactly (budgets
  `{4096, 32768, 262144}`), bottom-up Ritter sphere + merged normal cone.
- **Cache**: `cache/37_<model>_N<n>.bin` — topology + `leafSource` + static attributes.
  Invalidated on: cache magic/version, model file **stamp (mtime⊕size)**, **material hash**,
  **model-transform hash** (world-space samples are baked), N, sampling constants. Positions/bounds
  are **not** cached (S4 refits per frame → future skinned models keep working).
- **Packing (S3b)**: 16-byte leaf records (16-bit fixed-point pos, fp16-quantized radius u16,
  octahedral snorm16 normal, RGB8 albedo), 64-byte interior `childBounds`, 8-byte normal cones.
  CPU-side decode-vs-fp32 validation: max pos err `3.8e-5`, normal `0.0`, albedo `0.002`,
  radius `3e-5`.
- **Known limitation**: textured albedo/emissive are *not* sampled — surfels carry material
  *factor* colors only. Cornell (untextured) is the correctness reference; Sponza surfels will be
  factor-colored (white) until texture sampling is added.

### S4 status (done) — notes for later stages

- **Refit** (`leaf_refit.comp` + `refit_bottom_up.comp`): leaves re-evaluated from
  `TriangleBuf` + barycentric `LeafSource` (skinning-ready); interior nodes rebuilt bottom-up with
  **Ritter's algorithm over the 8 child spheres** (CPU `build_octree8` uses the identical merge, so
  GPU == CPU bit-for-bit). Passes: 1 leaf dispatch + `depth-1` level dispatches.
- **Radiance** (`radiance_seed.comp` + `radiance_pull.comp`): emissive leaves write
  `emissive × gain` into the RGBA16F per-node buffer, then bottom-up 8-way average pull-up.
- **Validation**: `validate_s4()` readback compares GPU vs CPU — Cornell N=32768:
  leaf pos err `0.0`, sphere err `0.0`, radiance err `0.0`.
- `pc.vert` gained a **radiance** color mode (leaf nodes at `[oct_interior, total)`).
- Multi-bounce (`num_bounces` 0/1/2 in UI) is **active** since S5c (see above); each bounce is a
  full leaf-level re-emission gather + copy-back + pull-up.
- **S4 refits fp32 buffers only** — the S3b packed records become stale after refit. GPU packing
  is deferred to S5, when the traversal consumes packed data.

### S5 status (done) — notes for later stages

- **Gather** (`micro_render.comp`): one 8×8 workgroup per gather pixel, thread = micro-pixel.
  Cosine-weighted **Nusselt disk mapping** (`ω = (u, v, √(1−u²−v²))`), jittered tangent frame.
- **Traversal**: per-thread iterative DFS with a 48-entry stack; hemisphere test, normal-cone
  backface cull, **angular-overlap prune** (`dot(dir,ω) < cos(θ_node+θ_pix)`), then solid-angle
  cutoff `Ω_node = 2π(1−cos θ_node)` vs `Ω_pix = 2π/m_valid` → recurse or rasterize.
- **No shared micro-buffer / no atomics**: each resolved node covers ≤1 pixel, so each thread
  keeps only its own `(depth, node)` in registers; per-workgroup shared reduction sums the 64
  micro-pixel contributions. This avoids ex36's per-thread 64-entry register-spill entirely.
- **Convolution**: with the disk mapping the cosine term cancels → `indirect = albedo · (1/m_valid) Σ rad[node]`. Radiance is the S4 emissive seed/pull-up.
- **Verified**: indirect readback scales **exactly 8×** with emissive gain (mean 0.00048→0.00383,
  max 0.047→0.373). Full-res 8×8 at 1080p.
- **Performance reality check**: full-res gather is slow on the iGPU (~1–2 s/frame) — expected;
  this is exactly what bilateral upsampling + low-res gather (S9) fixes. Morton dispatch is a
  later perf tweak.
- `display.frag` gained **view mode 5 = Indirect**; `validate_s5()` ("Indirect stats" button)
  prints mean/max.

### S5b — flat leaf splat (replaces the DFS traversal)

The per-gather-point DFS cut in `micro_render.comp` was the latency-bound hot path. It is now
replaced by a **flat brute-force leaf splat** into a per-gather-point micro-buffer atlas:

- One thread = one gather point; it loops over *all* `N` packed leaves (no hierarchy).
- Each leaf projects into the gather point's cosine-disk (sin projection, radius 1 at the rim),
  then depth-sorts into a `micro_size²` atlas tile via `imageAtomicMin` on a packed
  `(depth_q13 << 19) | node` key (r32ui atlas, cleared each frame with `glClearTexImage`).
  The convolve then sums `rad[node]` per covered micro-pixel.
- **Plane-bias cull** (`dot(d, n) <= leaf_radius` → skip): the gather surface's own leaves sit at
  grazing angles at distance ≈ spacing ≈ radius; without the bias their ~full-tile footprints
  win the depth sort at near-zero depth and black-clobber the whole micro-buffer (splotchy,
  near-black output — the old DFS avoided this only accidentally by recursing past leaves into
  out-of-bounds records). A per-leaf **backface cull** restores the old normal-cone behavior.
- Cost is O(P×N) — fast at low gather-res × low N, worse than the DFS at full res. The gather-res
  slider (1/16) is the intended interactive path. Atlas size = `(gather_w·M)×(gather_h·M)` uints
  (≈368 MB at full-res M=8, ≈1.4 MB at 1/16), so full-res is memory-heavy too.

### S5c — re-emission bounces (multi-bounce GI)

The bounce gather (`radiance_bounce.comp`) makes the `Bounces` UI do what §6 promised: one thread
per **leaf** surfel runs the same flat-splat micro-buffer gather (its own r32ui atlas,
`ceil(sqrt(N))` tiles per row), then re-seeds

```
rad_next[leaf] = emissive × gain + albedo × E,   E = (1/m_valid) Σ disk rad[winner]
```

The host copies the leaf region back into `Radiance` (`glCopyNamedBufferSubData`) and re-runs the
pull-up, so each extra bounce adds one more order of indirect light before the screen gather.
Without this pass the pipeline is single-bounce only: non-emissive leaves carry zero radiance
forever, so e.g. a ceiling micro-buffer can never see light bounced off the floor. Cost is O(N²)
per bounce (fine at N ≤ 32k; warned above that).

**GL note:** `gl::Texture::image_2d` allocates *immutable* storage — re-creating a texture at a
new size must move-assign a fresh `gl::Texture` (done for the G-buffer targets, indirect,
upsampled and both atlases; a plain re-spec is an `INVALID_OPERATION` and silently keeps the old
size).

### Performance optimization pass (P0–P4) — done

Starting point: full-res gather = **2.5 s/frame (0.4 fps)**. Root cause: per-micro-pixel DFS
(~9G node tests) + 32-byte fp32 node records + one 8×8 workgroup per pixel with 63/64 threads idle.

Changes, in order of impact:

1. **Amortized cut + one thread per gather point** (`micro_render.comp`): replace per-micro-pixel
   DFS with a single serial cut per gather point; each thread = one gather point (no barriers, no
   shared memory). ~9×.
2. **Packed traversal** (`PackedSphere` 8 B/node + `PackedCone` 8 B/node SSBOs, read directly):
   ~35% (also eliminated a `PackedCone` 6-byte-vs-8-byte struct mismatch).
3. **ALU elimination**: solid-angle cutoff reduced to an integer `r16² > thresh·d²` (no sqrt),
   cone-axis octahedral decode deferred to `cw > 0` (no normalize for wide cones), fp16-depth
   micro-buffer packed with node into one `uint` (monotonic → single compare). Together ~35%.

Measured (1600×900, Cornell box, N=32768, micro 8):

| gather res | micro-render | total | fps |
|---|---|---|---|
| full (1/1) | ~99 ms | ~99 ms | ~10 |
| 1/2 | ~46 ms | ~46 ms | ~22 |
| 1/4 | ~23 ms | ~24 ms | ~42 |
| 1/16 | ~5.8 ms | ~6 ms | ~165 |

Full-res is still latency/occupancy-bound (the serial dependent-load chain in the cut); the
gather-resolution slider is the intended interactive path, matching the paper's "preview → ground
truth" scale. Further full-res gains would need a wavefront/parallel cut (breadth-first frontier
grows exponentially → shared-memory blowup for micro > 8) or a smaller per-thread footprint.

Deferred: P4 Morton dispatch (tree is ~600 KB → fits L2, marginal expected gain).

> **Model list (S1/S9).** Currently only `CornellBoxOriginal.glb` and `sponza.glb` are wired up.
> `Stanford_Bunny.glb`, `Stanford_Dragon.glb`, `Rocks.glb` load fine but are disabled for now —
> they will be re-enabled once glTF node transforms are handled (gfx::Model currently leaves
> meshes in raw local space, so scenes built from multiple transformed nodes render mispositioned).
> The per-model transform in `main.cpp` (`model_transform`) already computes bounds from actual
> vertex positions and is robust to any unit scale (Sponza spans ~2735 units; it is scaled to
> fit a 2-unit box centered at the origin).

---

## 8. Performance model & checklist

Paper A time split (typical scene): ~60% rasterizing the hierarchy, ~18% mapping evaluation, ~8% ray
casting, ~11% upsampling/tonemap/direct light, ~2% hierarchy update. Track per-pass GPU timers and
compare against this split — it tells us where a given GPU bottlenecks.

Optimization checklist (in order of expected payoff):

1. Shared-memory micro-buffer, one workgroup per gather point (fixes ex36's register-spill).
2. BVH8 + packed/SoA records (bandwidth + depth).
3. DFS/pre-order + Morton dispatch + jitter (coherence, L2 residency).
4. Bilateral upsampling = the interactive mode (paper A's own preview strategy).
5. **Stretch**: amortized cut — traverse once per gather point and software-rasterize each resolved
   node into every covered micro-pixel instead of per-micro-pixel DFS.
6. **Stretch**: wavefront/level-parallel traversal (batched, throughput-bound instead of latency-bound).
7. **Stretch**: optional HW-RT acceleration on Ampere by voxelizing surfels into an AABB/BLAS — this
   abandons the continuous-LOD/sphere traversal, so it is a separate design, not this example.

---

## 9. Deliberate simplifications

- No bit-packing beyond the quantizations in §3 (16-byte surfels, 64-byte interior nodes are fine).
- Warp table: per-material precomputed GGX (64×64) rather than per-gather-point tables.
- Interpolation: Gaussian falloff on perpendicular distance (no full ray-cone differentials).
- Complete 8-ary tree (padded) instead of a sparse octree → implicit child indexing, at the cost of
  ~7/8 padding on sparsely populated nodes (fine for scene-scale budgets ≤ 512k surfels).

## 10. Repo layout

```
examples/37_emissive_gi/
  implementation.md    <- this file
  main.cpp             <- C++ driver
  shaders/             <- *.glsl, hot-reloaded
  CMakeLists.txt       <- copies shaders + data/*.glb
```

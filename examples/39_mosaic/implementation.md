# 39_mosaic — implementation status

MOSAIC: Micro-rendered Object-Space Amortized Irradiance Cache.
Design spec: `Mosaic Lighting.md` in the repo root. Stage numbering (S0–S8 plus
the direct-lighting addendum) follows the spec.

## Build & run

```bash
cmake -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target 39_mosaic
cd build-release/examples/39_mosaic && ./39_mosaic
```

The binary must run with its own build directory as the working directory:
models, `shaders/` and `cache/` are all resolved relative to cwd. Shaders hot
reload from the **copied** `shaders/` next to the binary, so editing a source
shader needs a rebuild (or a re-run of the copy) before the reload sees it.

Environment hooks, mirroring example 38's `SRT_*`:

| Var | Effect |
|---|---|
| `MOSAIC_SCENE=n` | 0 = Cornell Box, 1 = Sponza |
| `MOSAIC_VIEW=n` | initial debug view (see the View combo) |
| `MOSAIC_BENCH=n` | run n frames, print min/p10/p50/p90 of the tail, exit |
| `MOSAIC_SHOT=path` | write a screenshot before exiting |
| `MOSAIC_POINTS=1` | start with the surfel point-cloud overlay enabled |
| `MOSAIC_SURFEL_COLOR=n` | surfel overlay colour mode |
| `MOSAIC_SURFEL_LOD=n` | surfel overlay LOD |
| `MOSAIC_POINT_SCALE=f` | surfel overlay point size |
| `MOSAIC_MOVERS=0` | static scene only, no rigid movers |
| `MOSAIC_EYE=x,y,z` / `MOSAIC_TARGET=x,y,z` | camera override, in units of the scene radius |
| `MOSAIC_VALIDATE=1` | run the GPU-vs-CPU clipmap check and cache statistics |
| `MOSAIC_SH_TEST=L` | feed the gather a constant hemisphere of radiance L; the reconstruction must return L |
| `MOSAIC_EMISSIVE=f` | emissive boost override |
| `MOSAIC_IRR_GAIN=f` | irradiance debug-view gain |
| `MOSAIC_EXPOSURE=f` | initial exposure |
| `MOSAIC_NOGUI=1` | skip the ImGui overlay (clean screenshots) |
| `MOSAIC_REACH=f` | S7 screen-gather reach, in surfel radii |
| `MOSAIC_SHELL=n` | S5 shell radius, in cascade-0 cells |
| `MOSAIC_SPATIAL=f` | S6b spatial filter strength (0 = raw cache) |
| `MOSAIC_SKY=f` | uniform sky radiance; with `MOSAIC_EMISSIVE=0` this renders the gather's OPENNESS, which is how the coverage model is checked |
| `MOSAIC_BUDGET=n` | surfel updates per frame |
| `MOSAIC_TMAX=f` | temporal alpha upper bound (raise to see the raw per-refresh variance) |
| `MOSAIC_ESTEPS=n` | emitter cone-march steps; 0 renders the area lights unshadowed |

## Stage status

| Stage | What | Status |
|---|---|---|
| M0 | Scaffolding: instances, deferred G-buffer, display/debug views, hot reload with GLSL `#include`, per-pass timers, env hooks | ✅ |
| M1 | Surfel bake: object-space Poisson-disk sets, **texture-sampled albedo/emissive**, 4 LODs, packed records, disk cache | ✅ |
| M2 | S0 instance assembly + S1 cluster clipmap, occupancy clipmap, surfel index grid | ✅ |
| M3 | S2 surfel direct light, S3 scheduler, S5 micro-render gather, S6 SH projection + filter | ✅ |
| M4 | S7 screen gather + bilateral upsample + lit composite | ✅ |
| M4b | Matching the Cornell reference: micro-buffer solid angle, splat kernel, shell geometry, scheduler fairness | ✅ |
| M4c | S8 glossy ladder: sun GGX ✅, SSR / parallax cubemap / lobe | 🔶 |
| M5a | Emitter proxies from emissive surfels, analytic diffuse area lights, T2 occupancy cone-march visibility | ✅ |
| M5b | Portals (Sponza sky), T1 shadow atlas, CSM + PCSS, contact shadows, ReSTIR DI, glossy LTC tables | ⬜ |
| M6 | S4 surfel ReSTIR tail, LOD crossfade, LRU, scaling knobs | ⬜ |

## Layout

| File | Role |
|---|---|
| `main.cpp` | window, scene/instance setup, frame graph, ImGui |
| `surfel_bake.hpp/.cpp` | M1: triangle extraction, curvature/component annotation, Poisson sampling, texture read-back, packing, LOD parents, disk cache, `SurfelLibrary` |
| `clipmap.hpp/.cpp` | M2: S0 instance assembly, S1 cluster cascades, occupancy clipmap, GPU-vs-CPU validation |
| `cache_gi.hpp/.cpp` | M3: S2 surfel direct lighting, S3 scheduler, S5 gather, S6 projection/temporal blend |
| `mosaic.hpp` | `Instance`, `Bounds`, `Config` (cascade sizes, budgets) |
| `gpu_util.hpp/.cpp` | `PassTimer`, ImGui stacked bar/legend, `Pipeline` (hot-reloadable program), `PrefixSum`, camera control |
| `screen.hpp/.cpp` | `GBuffer`, `GeometryPass`, `DisplayPass` |
| `shaders/common/` | GLSL headers shared by every stage (`oct.glsl`, `gbuffer.glsl`) |

## Decisions and deviations from the spec

**World position is reconstructed from depth, not stored.** Examples 37 and 38
both keep a full RGBA16F position target; that is ~8 MB/frame at 1080p to store
what the depth buffer already encodes. `world_from_depth()` lives in
`shaders/common/gbuffer.glsl`.

**GLSL `#include` instead of copy-paste.** Examples 36/37/38 each duplicate
octahedral encode/decode and surfel unpacking in every shader that needs them.
`gl::HotReloadProgram::add_include_dir("shaders/common")` resolves includes, and
`gl/shader_include.hpp` additionally supports `#if/#ifdef` with a host-supplied
macro map when shader variants are needed later.

**Two octahedral mappings, deliberately distinct.** `oct_encode/oct_decode` map
the unit *sphere* (normal compression, same convention as `coverage_atlas.cpp`
and examples 32/34/35/37, so packed normals interchange). `hemi_oct_encode/decode`
map the unit *hemisphere* for the S5 micro-buffer. Unlike the disk/Nusselt
projection used by 36 and 37, the hemi-octahedral mapping does **not** cancel the
cosine term — the S6 resolve must weight each texel by `cos(theta) * solid angle`
explicitly.

**64³ cascades sit exactly on the scan limit.** `PrefixSum` scans block sums in a
single 1024-thread workgroup, capping one scan at `1024 * 256 = 262144` elements.
That is exactly `64³`, the spec's cascade resolution, with zero headroom: raising
`Config::kClipRes` requires a multi-level scan first. `PrefixSum::run` logs and
refuses rather than silently truncating.

**Cell sizes are scene-relative, not metric.** The spec's 0.5/2/8 m cascades and
4/8/25/100 cm surfel spacings assume a metric scene. Cornell is normalized to
~4 units and Sponza spans ~2735 before fitting, so `Config::cascade0_cell` and
`surfel_spacing` are derived from the fitted scene extent and the spec's ratios
are preserved.

**Movers use a separate asset.** Reusing scene meshes for the rigid movers gives
flat panels on Cornell, which cannot show whether cached irradiance travels with
a moving object. `Stanford_Bunny.glb` is loaded as a second model, which also
exercises the multi-model instance path S0 needs. Note that asset is a 7-level
LOD group laid out **side by side in Z**, so both the mesh selection and the fit
must restrict to LOD 0 or the mover ends up 6x too small.

## M1 results

| Scene | Sets | Surfels | Emissive | Bake (cold, 16 threads) | Bake (cached) |
|---|---|---|---|---|---|
| Cornell Box | 9 | 278 919 | 2 181 | 0.19 s | 0.00 s |
| Sponza | 104 | 661 376 | 0 | 2.6 s | 0.01 s |

Sponza has no emissive materials at all, which is why it has zero emissive
surfels — its light has to come from the sun and from authored portals (M5).

**LOD-0 spacing comes from a surfel budget, not from the cascade size.** This is
the single most important decision in M1. Deriving spacing from
`Config::cascade0_cell` (as the first draft did) says nothing about how much
surface area a scene contains: on Sponza every one of the 103 meshes saturated
the per-mesh cap and the bake ran past ten minutes. `scene_surface_area()` plus
`spacing_for_budget()` gives 0.0191 world units for Sponza and 0.0157 for
Cornell, and the bake drops to seconds.

**The bake is parallel across meshes.** Each mesh is independent. Texture
read-back must stay on the main thread because it issues GL calls; the sampling
that follows does not.

**Surfels are stored in object space, quantized to the mesh AABB.** A set is
keyed by (model, mesh) and shared by every instance of it. Because the target
spacing is expressed in world units, `bake_mesh` divides by the instance's
uniform scale to get the mesh's own units — Sponza's fit scale is ~0.001, so
skipping that conversion produces surfels ~1000x the intended size. Non-uniform
instance scales are detected and warned about rather than silently mishandled.

## Deferred from M1

**Per-surfel cut planes** (example 38's `compute_surfel_cuts`, `main.cpp:746`)
are not baked. They clip a surfel disk exactly at mesh feature edges, which
matters a great deal for 38's ray-disk intersection but much less for a 16x16
micro-buffer splat whose angular resolution is ~16x coarser. They also cost
192 bytes per surfel — twelve times the packed record — so at 660 k surfels that
is 127 MB. Revisit in M3 if crease artefacts actually appear in the gather.

**Alpha-masked geometry is sampled without regard to alpha.** Sponza's foliage
quads get surfels across their fully transparent regions. The G-buffer discards
those texels but the bake does not, so foliage currently carries surfels that
represent nothing. Fix by sampling the base-colour alpha at bake time and
rejecting samples below the material's alpha cutoff.

## M2 results

Frame cost, Release build, 1600x900 (whole frame, GI stages plus G-buffer):

| Scene | Live surfels | min | p50 | p90 |
|---|---|---|---|---|
| Cornell Box | 69 327 | 0.28 ms | 0.35 ms | 0.67 ms |
| Sponza | 262 102 | 0.47 ms | 0.89 ms | 1.23 ms |

S1 sub-pass breakdown on Cornell: count 0.018, scan 0.066, compact 0.020,
merge 0.050, cluster 0.557, occupancy ~0.03 ms — against the spec's 0.60 ms
budget for the whole of S1.

**Do not call `glGenerateTextureMipmap` on a 3D texture.** It cost 2.6 ms per
128^3 R8 volume on this driver, 7.9 ms for the three cascades, which was 97% of
the frame. A 2x2x2 box downsample in compute (`occupancy_mip.comp`) does the same
work in microseconds. The filter is the mean, not the max: the value is a
coverage fraction and a cone march accumulating (1 - alpha) needs a half-covered
coarse cell to read as half-occluding.

**Per-cascade scans need per-cascade base offsets.** A single prefix sum is
capped at 262144 elements, which is exactly one cascade, so the scan runs once
per cascade and `cell_start` restarts at 0 each time. Without `clip_bases.comp`
adding a base, the compacted membership ranges of different cascades overlap in
`packed_idx` and silently overwrite each other. This is invisible whenever only
one cascade is non-empty — Cornell with an exterior camera never showed it, and
only Sponza with an interior camera did. The base is folded into `cell_sc.x` at
merge time so no downstream consumer has to know about cascades.

**Budget exhaustion coarsens, it does not drop.** When the live set would exceed
`kMaxLiveSurfels`, an instance steps to a coarser LOD rather than being skipped;
a skipped instance leaves a hole that reads as missing indirect light. Only if
even the coarsest LOD does not fit is it dropped, and the UI says so. Sponza sits
right at the cap (262 102 of 262 144) with 100/2/3/1 instances across the LODs.

**A camera outside the scene wastes C0.** The cascades are camera-centred, so
with the Cornell camera outside the box, C0 (6.9 units across) reaches nothing
and every surfel lands in C1. That is correct behaviour, not a bug, but it means
exterior-camera timings understate the cost of the finest cascade.

### Validation (plan gate 2)

`MOSAIC_VALIDATE=1` rebuilds the cascade assignment on the CPU from the live
surfel buffer and compares per-cell counts, the compacted membership sets
(order-independent — the compaction slot comes from an atomic), the occupancy
bitmask, and each cluster's area-weighted centroid and total projected area.

```
Cornell: PASS (69 327 live, 0 outside all cascades)
Sponza:  PASS (262 102 live, 0 outside all cascades)
```

## M3 results

Measured on an RTX 3060 mobile, 1600x900, Release.

| Scene | Frame (p50) | GPU total | S5 gather | S1 clipmap | S2 direct | S3 sched |
|---|---|---|---|---|---|---|
| Cornell Box | 2.73 ms | 2.03 ms | 0.44 ms | 0.63 ms | 0.11 ms | 0.17 ms |
| Sponza | 6.13 ms | 4.75 ms | 0.37 ms | 1.48 ms | 0.14 ms | 0.35 ms |

The spec budgets 2.20 ms for S5 at 8k updates; both scenes come in well under.

### Frame timing was measured wrong at first

The frame counter originally timed the CPU span of the loop body. With vsync off
and no sync point that measures how long it takes to SUBMIT a frame, not how long
the frame takes — it read ~1.5 ms while Sponza was actually running at ~10 fps.
Frame time now comes from the wall-clock delta between successive `window.time()`
calls, and the UI additionally reports the summed `GL_TIME_ELAPSED` per-pass
timers (which were correct all along) plus a warning when the GPU sum exceeds the
wall clock. `MOSAIC_BENCH` prints wall-clock percentiles and a per-pass GPU
breakdown.

### What made the gather affordable: 132 ms -> 5.3 ms on Sponza

Four fixes, in order of impact:

1. **Dense cells use their cluster instead of individual surfels.** The micro-
   buffer resolves 256 directions, so splatting far more emitters than that into
   it is waste. Sponza packs columns, arches and curtains into a small volume, so
   one C0 cell can hold thousands of surfels; expanding every near cell cost
   ~27000 splats per gather. Above a threshold (64) the cell's own cluster is
   used, which is energy-preserving by construction — it is the area-weighted
   aggregate of exactly those surfels. Gather 39 ms -> 0.4 ms, and mean
   irradiance went UP, because individual splats were losing emitters to the
   depth test that the cluster accounts for.
2. **The cluster pass gets a workgroup per cell, not a thread per cell.** Most
   cells are small but the tail is long: a coarse cell can hold tens of thousands
   of surfels and one thread stalls its whole workgroup. Dispatched indirectly
   over a compacted list of non-empty cells. 26.3 ms -> 1.0 ms.
3. **The coarsest cascade must be genuinely coarse.** The gather walks every
   non-empty cell of the coarsest cascade, so when the coverage rule picked a
   single cascade for Sponza that walk became 10000 cluster splats per surfel.
   The cascade count now also requires the coarsest cell to be at least a tenth
   of the scene diameter.
4. **Near radius of one cell, and a bounded splat footprint.** The splat loop is
   O(footprint^2) LDS atomics; unbounded, one close cluster could write the whole
   tile.

### Correctness gates

| Gate | Result |
|---|---|
| 2. Clipmap vs CPU reference | PASS both scenes, 0 surfels outside coverage |
| 3. SH round trip (constant hemisphere) | L=0.5 -> 0.492, L=2.0 -> 1.969 (98.4%, the expected 16x16 quadrature error) |
| 4. Energy scaling | 8x emitter -> 7.96x and 8.02x irradiance |

Gate 3 caught a real bug: the projection multiplied radiance by cos(theta) AND
applied the cosine-convolution constants, double-counting. The SH path projects
RAW radiance; `sh_eval_irradiance` does the convolution.

### Other M3 findings

**The cache is keyed on the SOURCE surfel, not the live index.** S0 reassigns
live indices every frame as LOD selection and instance ordering change, so a
cache keyed on the live index would scramble whenever either did. `live_src[]`
is the indirection, and it is also what makes the object-space claim hold.

**The scheduler starves without a per-frame dither.** Large numbers of surfels
land in the same priority bucket; `sched_select` takes the first `budget` of them
in atomic order, which correlates with thread id, so the same low-index surfels
refreshed every frame while the rest never updated. A dither of about one bucket
width makes the choice among equals uniformly random per frame.

**Surfels are inserted into every cascade containing them, not just the finest.**
Otherwise the gather's shell walk has a hole: a neighbour further than C0's few
cell reach but nearer than C1's box boundary would be gathered by nothing. Costs
3x index memory.

**Cornell's camera has to see C0.** The cascades are camera-centred, so an
exterior viewpoint left C0 empty and the near-field path never ran. The coverage
rule now requires the coarsest cascade to span twice the scene diameter.

### Artifact fixes

Four visible artifacts, each with a distinct cause:

**Hard black patches with curved edges** — L1 SH ringing. An L1 basis cannot
represent a strongly directional distribution, and the reconstruction
`A0*C0*c0 + A1*C1*(c1 . n)` dips negative wherever the L1 term outweighs the DC
term, which is exactly what a single bright emitter produces. Clamping the result
at zero turns that dip into a hard-edged black patch that moves with the light.
`sh_eval_irradiance` now windows the L1 band per channel by the largest factor
that keeps the minimum at zero, preserving both total energy and gradient
direction.

**Objects permanently black** — scheduler starvation, and the most instructive
bug of the milestone. The mover bunnies never received a single cache update in
600 frames. Two independent causes:

  1. Importance scaled with the raw surfel radius, so it measured how finely a
     mesh happened to be sampled rather than how much screen it covered. The
     bunnies carry surfels a quarter the radius of the Cornell walls and scored
     0.44 against their 0.80. Importance is now measured in PIXELS COVERED.
  2. The age term saturated at 0.30, below the maximum importance, so a stale
     surfel could never outrank a freshly updated visible one — starvation was
     permanent rather than merely slow. `kAgeSpan` now exceeds the largest
     possible importance, which bounds staleness by construction.

  A third, related fairness bug: `sched_select` resolved the oversubscribed
  cutoff bucket by `atomicAdd` order, which correlates with thread index, so low
  live indices won every frame. Since S0 appends dynamic instances last, the
  movers sat at the top of the live range and lost every time. The cutoff bucket
  is now subsampled uniformly with a per-surfel, per-frame hash.

**Blotchy mottling** — each surfel integrates its own 16x16 micro-buffer, so
neighbours land on slightly different answers. `sh_spatial.comp` averages over
same-cell neighbours with normal and plane-distance rejection (the same rejects
the screen gather uses, and they are the correctness condition, not an
optimisation: blending across a corner is exactly the leak this technique is
supposed to be free of). It reads the raw cache and writes a separate filtered
one, because filtering in place would feed the blur back into the temporal
estimate and flatten the cache over a few seconds. 0.28-0.79 ms.

**Diagonal streaking** — the same per-surfel variance, structured by the
hemi-octahedral texel grid. Removed by the spatial filter.

### Known limitations

- **Small bright emitters are diluted by coarse clusters.** One cluster per cell
  averages a Cornell ceiling panel against the dark ceiling around it. This is
  precisely the case the spec assigns to the ReSTIR tail (S4, M6): deterministic
  micro-rendering owns the low-frequency bulk, ReSTIR owns the bright tail.
- **Only the live LOD is cached.** Non-selected LODs are never updated, so about
  75% of source surfels stay black. LOD crossfade (M6) seeds a level from its
  parent.
- **Sponza has no emissive materials**, so its only lights are the sun and the
  sky fallback. It stays dim until M5 adds portals and proper direct lighting.
- (Resolved in M4.) Indirect light was visible only through the surfel debug
  overlay. The overlay draws round point sprites, so the "circles" in it were
  the visualisation, not the lighting.
- **Sponza is dark.** It has no emissive materials, its roof blocks the sun over
  most of the atrium, and the sky fallback is a flat colour. It needs M5's proper
  sun, sky and portal lights before it reads as lit rather than as merely
  correct.

## M4 results

The cache finally reaches the screen. S7 gathers the per-surfel SH into a
half-resolution indirect buffer, a bilateral upsample takes it to full
resolution, and the display pass composites `emissive + albedo * (direct +
indirect) + sun specular`. View mode 8 shows indirect alone, mode 9 the lit
image.

Measured on an RTX 3060 mobile at 1600x900, Release, wall clock, steady-state
tail, `MOSAIC_NOGUI=1`:

| Scene | Frame p50 | GPU total | S5 gather | S6 spatial | S7 gather | S7 upsample |
|---|---|---|---|---|---|---|
| Cornell | 6.43 ms | 5.53 ms | 1.29 ms | 1.55 ms | 0.93 ms | 0.08 ms |
| Sponza | 9.40 ms | 8.03 ms | 1.32 ms | 1.80 ms | 1.14 ms | 0.09 ms |

Both numbers are stable to about +/-2% across runs once the GPU has settled.
They are not stable across *thermal* states: the same build measured 5.5 and
11.1 GPU ms on consecutive runs when the laptop was already hot. Compare only
runs taken back to back.

### Three bugs stood between "the cache is correct" and "the image is right"

Each one produced a distinct, recognisable artefact, and none of them was in the
cache itself -- M3's validation gates still pass unchanged.

**1. The screen gather subsampled the cells it searched.** It walked a
(2R+1)^3 ring and, inside each cell, took `kMaxTaps` members with a stride, so
that a dense cell contributed 8 *arbitrary* surfels rather than the 8 near the
pixel. A cascade-0 cell holds ~80 surfels in Cornell and a surfel reaches about
two of its own radii, so almost every strided candidate failed the distance
test and most of the screen found nothing at all: the image was a blotchy
half-covered mess with black holes at surfel scale. Now the gather walks the
2x2x2 block of cells around the pixel -- which by construction contains every
surfel that can reach it, because S0 picks the LOD so a radius stays well under
a cell -- home cell first, in order, with a scan ceiling instead of a stride.

**2. Cascade 0 did not cover the scene from the framing viewpoint.** C0's cell
follows the surfel spacing (spec: ~12 surfels across a cell), which gave Cornell
a 12.09-unit coverage around a camera standing 4.3 units outside a 6.9-unit
scene. The back wall fell about half a unit outside, `clip_inside` rejected every
cell the gather asked for, and the whole wall returned the sky -- a hard-edged
black rectangle exactly where the colour bleeding should be. Two changes:

- a **coverage floor** on C0 for small scenes, `2 x diameter`, capped at
  `32 x surfel_spacing` so it cannot fire on a large scene and turn the near
  field into metre-wide cells. Cornell's C0 cell goes 0.1888 -> 0.2147; Sponza is
  unchanged, its spacing-derived cell was already larger than the floor.
- a **coarse-cascade fallback** in the gather: a pixel with no per-surfel answer
  now takes the enclosing cluster of the finest cascade that does contain it.
  A cluster is the area-weighted outgoing radiance of exactly the surfaces in
  that cell, so it is coarse but not biased, and it degrades to something
  plausible instead of to the sky.

**3. The spatial filter strided too, and flattened the cache into cells.**
`sh_spatial` picked its 8 neighbours by striding across the cell. The stride is
the same for every surfel in a cell, so they all averaged the same handful of
members and converged on one value: the walls came out as flat blocks at exactly
the cascade-0 cell size, which read as a chequerboard. It now walks the cell in
order and rejects on **distance** (4 surfel radii) as well as on normal and
plane, with a distance falloff in the weight -- an actual local blur, each
surfel averaging its own neighbourhood. Cost went 0.79 -> 1.55 ms on Cornell.

### S5 shell radius

The default is now 3 cells rather than 1. At R=1 the mid field starts at 0.19
units, so everything past that is a single coarse cluster per cell and the walls
away from the light stayed nearly black. R=3 hands over to the coarsest cascade
at 0.64 units instead. Costs about 0.8 ms on Cornell and is the single largest
quality knob in the pipeline.

### S8: what is in and what is not

The **rough end** of the glossy ladder is implemented: analytic GGX for the sun,
height-correlated Smith visibility, Schlick Fresnel, metal F0 from albedo with
the diffuse lobe removed for metals. It needs no extra buffers.

The **mirror-to-glossy bands are not**. SSR needs a lit HDR target plus a
history copy to march against, and the pipeline currently composites straight
into the default framebuffer; the parallax cubemap and the dominant-lobe
fallback both need the per-surfel lobe buffer, which S6 does not extract yet.
Both are a restructure rather than a shader, so they are tracked separately
above rather than being claimed as done.

### Known limitations after M4

- **Curved surfaces still speckle.** The spatial filter needs neighbours whose
  normals agree within 0.9; the bunnies' curvature leaves few, so per-surfel
  gather variance survives there as coloured mottling.
- **Sponza is still dim.** The atrium's only lights are the sun (blocked over
  most of the floor) and a flat sky colour. M5's portals and proper direct
  lighting are what this scene is waiting for.
- **Emitter surfels stop being refreshed.** In Cornell the light panel's own
  2181 surfels report 0% fresh late in a run. They are emitters, so their own
  cached irradiance barely matters, but it is a scheduler fairness artefact
  worth revisiting with M6.

## M4b — matching the Cornell reference

`CornellBoxOriginalGroundTruth.png` next to this file is the path-traced
reference. Comparing against it turned up four separate defects, all of which
read as *shapes* in the lighting rather than as noise, and all of which were
still there after the M4 fixes.

Run the comparison with
`MOSAIC_SCENE=0 MOSAIC_VIEW=9 MOSAIC_NOGUI=1 MOSAIC_MOVERS=0`.

### 1. Diagonal chevrons — the micro-buffer quantized solid angle to whole texels

`splat()` rasterized every emitter as some number of FULLY lit texels. A cluster
whose true angular radius was a third of a texel still lit one entire texel, and
one that grew past half a texel jumped from 1 texel to 9 — a nine-fold step in
its contribution. Sweep a surfel along a wall and that step traces the octahedral
texel grid across the surface, which is the diagonal chevron and diamond banding
that made the walls look tiled.

The splat now computes the emitter's actual solid angle, `omega = A cos / d^2`,
and uses it two ways:

- **The footprint comes from omega, not from the covering radius.** A cluster's
  covering radius is its cell's half-diagonal, several times the angular extent
  of a flat wall patch inside it. Sizing the footprint from omega instead makes
  the coverage below land at 1 for anything at least a texel across.
- **Coverage accumulates per texel and clamps at 1; radiance stays
  winner-takes-all.** These cannot share the depth test. Surfels overlap heavily
  — at half a metre a wall's surfels are ~25 to a texel, each covering ~23% of
  it — so taking the fill from the single depth-sort winner would drop the other
  24 and leave 77% of a solid wall reading as open sky. Summing coverage gets the
  occlusion right while the nearest emitter still supplies the radiance.

Cluster energy is exact under this model: radiance is the area-weighted mean and
coverage is the summed area, so `mean(L) * sum(A)/d^2 = sum(L_i A_i)/d^2`
whatever the cell aggregation does.

### 2. Concentric rings around the light — an integer footprint radius

With the footprint sized in whole texels, a fixed emitter's splat still stepped
from 1 texel to 9 to 25 at fixed distances, and each step traced a circle on
every surface at that distance. The kernel is now a **disk with a one-texel soft
edge**, normalized so the splat always carries exactly omega: a texel enters the
footprint at zero weight and grows in, so the estimate is continuous in
`radius/dist`.

Flat inside the disk matters as much as the soft rim. A peaked (tent) kernel
loses whatever the centre texel clamps off while the rim stays half empty, and a
closed room measurably reads as more open than it is.

The micro-buffer's tangent frame is also rotated by a per-surfel, per-frame
angle, so whatever quantization survives is decorrelated between neighbours and
becomes noise the temporal blend and the S6 filter already remove.

### 3. Rectangular tiles one cascade-0 cell across — two order-dependent budgets

Two passes capped their work by *scan order*, which is the same for every point
inside a cell, so every point in a cell converged on the same value and the
answer changed in one step at the cell boundary:

- **S7 screen gather** stopped at the first 16 accepted taps. Now the tap count
  is unbounded and only the number of candidates INSPECTED is capped; the
  distance weights already fall off, so accepting everything that passes is both
  simpler and continuous. This was the dominant one.
- **S6 spatial filter** picked its 8 neighbours by striding across the cell (see
  M4 above); it now walks in order and rejects on distance.

A per-pixel random start would sample the cell without bias and does look
marginally smoother, but it makes neighbouring threads read unrelated addresses:
5.8 ms against 2.4 ms for the sequential walk. Not worth it once the tap cap is
gone.

Also fixed here: **the S5 shell was a box while the next cascade excluded a
sphere.** Cells in the box's corners were splatted twice, once as C0 cells and
again inside a coarse cluster, and the double-counted region was locked to the
gather point's cell. The shell is now clipped to the same sphere the next level
excludes. And a coarse cluster's **self-exclusion** is measured against half its
cell rather than its covering radius — the covering radius (0.87 cells) exceeded
the handover distance ((R-0.5) finer cells), so every coarse cluster in between
was dropped by both levels: a hole in the mid field exactly where a room's
opposite wall sits.

### 4. Half the cache was never written — the scheduler starved its own tail

`sched_priority`'s staleness bound was `kAgeSpan (0.50) > kImportanceBase +
kImportanceSpan (0.45)`. That only holds against a surfel whose own age term is
zero, and in steady state no visible surfel's is: visible surfels settled at age
~6, scoring 0.45 + 0.05, just above the starved tail's flat 0.50, and took the
entire budget.

On Cornell that left **42.8% of the cache ever written at all**. The rest sat at
zero and splatted as black into every other surfel's micro-buffer, so it cost
energy twice. Raising the budget from 8k to 32k took coverage to 77.9% and nearly
doubled mean irradiance, which measures what the bug was worth.

Past `kAgeSaturate` the age term now **preempts** importance rather than
outweighing it (`priority = 1.0`), which bounds staleness at
`kAgeSaturate + live/budget` frames and lets `sched_select`'s uniform subsample
share the budget fairly. At the unchanged 8k budget, Cornell coverage went
42.8% -> 77.8% — 217045 of 217045 LOD-0 surfels, i.e. the entire live set — and
mean irradiance 0.0239 -> 0.0405.

### Result

| Scene | Frame p50 | GPU total | S5 gather | S6 spatial | S7 gather |
|---|---|---|---|---|---|
| Cornell | 8.94 ms | 8.02 ms | 2.62 ms | 1.63 ms | 2.24 ms |
| Sponza | 10.78 ms | 9.36 ms | 2.73 ms | 1.84 ms | 1.37 ms |

Up from 6.4 / 9.4 ms before this pass; the cost is concentrated in S5's smooth
splat and S7's uncapped taps, and both buy a visible artefact class each.

All three gates still pass unchanged: clipmap PASS on both scenes with 0 surfels
outside coverage; SH round trip 0.5 -> 0.49217 and 2.0 -> 1.96868 (98.4%, the
16x16 quadrature error); 8x emissive -> 7.99x mean, 8.02x peak.

Default exposure is now 2.0. The cache carries one bounce per frame through the
S2 feedback loop rather than a converged multi-bounce solution, so a closed
diffuse room lands about a stop under a path-traced reference at unit exposure.

### 5. Low-frequency blotching — the temporal filter was fighting its own jitter

Rotating the micro-buffer's tangent frame per surfel AND per frame (fix 2 above)
turns successive refreshes into independent samples of the same integral, which
makes the temporal blend a Monte Carlo average. The variance-adaptive alpha was
working against that: it read the sampling jitter as "this surfel is still
moving", pinned alpha near its 0.5 ceiling, and averaged about two refreshes.
What was left was gather variance low-passed by S6 and S7 into soft patches
rather than removed -- smooth-edged blotches on the ceiling and the box faces
where the reference has a clean gradient.

`temporal_max` is now 0.15, averaging ~7 refreshes. Sweep it with `MOSAIC_TMAX`;
the improvement is clear from 0.5 to 0.25 to 0.15 and flattens below that.

Latency is not the cost it looks like. S2's direct term is exact and recomputed
every frame, and a hard lighting change still forces alpha to 1 through the
history reset, so only the multi-bounce tail takes ~7 refreshes to settle.

### Still visibly unlike the reference

- **The rigid movers mottle.** A mover's surfels change what they see every
  frame, so the temporal blend never converges on them, and being curved they
  have few same-normal neighbours for the spatial filter to average over. The
  static geometry around them is clean.
- **Contact shadows are soft and weak** where the reference has a definite dark
  band under each box. That is the visibility ladder, which is M5.
- **The room is still short of a full multi-bounce solution.** The feedback loop
  gains one bounce per refresh, and a refresh is 30-70 frames for the tail.
- **Straight-edged patches survive on the ceiling.** Unlike the blotching above
  these have hard, non-axis-aligned boundaries, so they are not variance: they
  are the coarse-cluster aggregation, one splat per cell carrying a cell's
  average radiance from its centroid. Cells enter and leave the horizon cull as a
  unit, and their edges are what shows. The designed fix is the S4 ReSTIR tail
  (M6), which resolves bright clusters into explicitly weighted samples instead
  of leaning on the aggregate.

## M5a — emitter proxies and analytic area lights

Until now the only path from an emissive surface to a receiver ran through the
gather: the emitter's surfels were splatted into every receiver's 16x16
micro-buffer like any other geometry. Correct in the limit, and bad in practice
for a small bright source — a Cornell ceiling panel lands on a handful of texels
out of 256, and once it is far enough to be aggregated it is averaged against the
dark ceiling around it inside one cluster.

M5a represents such sources explicitly, as oriented rectangles shaded
analytically. A rectangle's diffuse contribution has a closed form, so the
emitter itself contributes no noise and no banding at all — only its visibility
does.

### Extraction (deviation from the spec)

The spec rasterizes emissive textures into UV space, runs connected components
there, and fits a rect per island. `direct.cpp` runs the same connected
components and PCA fit over the **emissive surfels M1 already baked**: they carry
texture-sampled emissive radiance, they are already distributed over the surface,
and they cost nothing extra. Same output — centre, two half-axes, normal, area,
mean radiance — with no UV-space rasterizer.

LOD 0 only (the coarser levels resample the same surface and would emit the same
light three more times). Islands link within 2.5 spacings and a 0.9 normal dot.
The 2x2 covariance's principal axis has a closed form, so the fit needs no
iterative solver.

**Fit quality is handled by scaling, not by rejecting.** A rectangle is a poor
shape for a disc or an L, but a poor shape is not a poor light: scaling the
radiance by the coverage keeps emitted power exact whatever the outline, and what
is left is only that the light is spatially smeared over its bounding rectangle.
Rejecting the island instead sends it back through the gather and costs the
banding this path exists to remove. Cornell fits one proxy over 1661 surfels at
0.56 coverage.

### Partitioning, not adding

Proxied emission is **removed** from `live_radiance` in S2, so the gather no
longer sees it. The two paths partition the light rather than double-counting it;
the surfels stay in the micro-buffer as occluders, just with no emission. The
emissive boost applies to proxies too, or the slider would scale only the light
still going through the gather.

### Three bugs, each with a distinct signature

**1. The polygon wound the wrong way and the light switched itself off.** The
Lambert/Arvo form factor is a *signed* sum: reverse the vertex order and it comes
out negative, and the clamp then returns zero. `cross(u, v)` lands either way
depending on how the PCA axes fell out, so `refresh()` now pins it by swapping
the axes (not negating one, which would also flip the rectangle), and the shader
walks the corners in the order that winds counter-clockwise as seen from the
emitting side. Before the fix the scene was black except for a few slivers that
happened to fall on the right side.

**2. The cone march drew concentric rings around the light.** Two causes, and
raising the step count made the second one *sharper*, which is what separated
them:

- Pure cone stepping grows `t` by `(1 + spread)` per step, which for a small
  source is far too slow — at spread 0.12, sixteen steps cover 6x the start
  radius, well short of a room. The march stopped mid-air with partial occlusion
  accumulated, and where it stopped depends on distance to the light. Now the
  geometric ratio is chosen to land exactly on `t_end` in `steps`, which bounds
  the cost as well.
- The cone widens to the source's own half-extent by the time it arrives, so for
  a light set into a ceiling the last samples always straddle the plane the light
  sits in and read it as an occluder. Whether a given sample lands inside that
  layer depends on where the progression puts it — hence rings. The march now
  ends a source-radius short of the emitter.

**3. Most vexing parse.** `std::vector<std::vector<uint32_t>> groups(size_t(n));`
declares a function. Written with `resize()` instead.

### Cost

| Scene | Frame p50 | GPU total | Display (incl. area lights) | S2 direct |
|---|---|---|---|---|
| Cornell | 9.27 ms | 8.38 ms | 0.52 ms | 0.12 ms |
| Sponza | 10.80 ms | 9.41 ms | 0.10 ms | 0.14 ms |

Sponza has no emissive materials, so it fits zero proxies and pays nothing. The
cache deliberately gets 6 cone-march steps against the pixel path's 16: a
hemisphere integral low-passes the shadow, so penumbra accuracy inside the bounce
is wasted work (spec section 2).

Gates: clipmap PASS both scenes; SH round trip 0.5 -> 0.49217 and
2.0 -> 1.96868 (98.4%); 8x emissive -> 8.00x mean irradiance (peak reads 8.68x
because the peak surfel is not the same one at both settings).

### Known limitations after M5a

- **Faint arcs survive around the emitter.** Much reduced but not gone; they are
  the residual phase of the cone march's sample positions against the occupancy
  grid. Only the pixel path shows them — view 8 (indirect alone) is clean.
- **Contact shadows are soft.** A cone march is a coverage integral, so it has no
  sharp near-field term. That is what the screen-space contact shadow and the T1
  PCSS atlas are for, both M5b.
- **Sponza gains nothing yet.** It has no emissive geometry at all, so its lights
  are still the sun and a flat sky colour. Portals are the M5b item that matters
  for it.

## Notes on the gllib API

- `gl::Texture::image_2d` allocates **immutable** storage. Re-specing at a new
  size is `INVALID_OPERATION` and silently keeps the old size, so `GBuffer::create`
  move-assigns fresh textures on resize.
- `gl::BufferType` has no `dispatch_indirect`; `GL_DISPATCH_INDIRECT_BUFFER` must
  be bound with raw `glBindBuffer` (needed from M3 onward).
- `gfx::Model` discards decoded image bytes after upload, so M1's texture
  sampling reads them back with `glGetTextureImage`.
- `ModelMaterialInfo`'s texture indices are glTF **image** indices, and `Model`
  skips images it fails to decode, so an index can be out of range on a damaged
  asset. `GeometryPass` and `bake_mesh` bounds-check rather than trusting them.
- `glGetTextureImage` on an sRGB internal format returns the **stored** bytes and
  performs no sRGB decode, so the bake linearizes explicitly.
- `glBlitFramebuffer` refuses a depth copy between differently-formatted depth
  buffers ("Depth formats do not match"), which rules out blitting the DEPTH32F
  G-buffer depth into the default framebuffer. The surfel overlay instead tests
  the depth *texture* in its fragment shader for scene occlusion, and uses the
  default framebuffer's own depth buffer for point-vs-point occlusion.
- CMake's `copy`/`copy_directory` are unconditional, so a POST_BUILD data copy
  refreshes the file's mtime on every relink. The surfel cache is keyed on that
  mtime, so this silently forced a full rebake after every code change; the
  example uses `copy_if_different` / `copy_directory_if_different`.

## Fixes made outside this example

- `gl::HotReloadProgram::add_stage` did not propagate include dirs to stages
  added *after* `add_include_dir`, despite the header documenting them as
  "applied to all stages" — so `#include` silently failed depending on call
  order. `add_stage` now passes the include dirs to the `ShaderFile` constructor,
  which also avoids a spurious "include not found" error on the initial read.
- `gfx::Model` ignored **KHR_materials_emissive_strength**. glTF clamps
  `emissiveFactor` to [0,1], so any emitter brighter than one carries its real
  intensity in that extension; ignoring it caps every light in every scene at 1.
  CornellBoxOriginal.glb authors its ceiling panel at strength 17, which is why
  it rendered ~17x too dim compared with Blender in every example that used it.
  `ModelMaterialInfo::emissive_factor` is now premultiplied by the extension
  value (and may therefore exceed 1), with the raw value kept in
  `emissive_strength`.
- FidelityFX FSR2 was fetched from `/tmp/opencode/FidelityFX-FSR2-OpenGL`, a path
  that no longer exists, so any fresh CMake configure failed. It is now behind
  `option(GLLIB_ENABLE_FSR2 OFF)` with a `GLLIB_FSR2_REPOSITORY` cache variable;
  `gfx/gfx.hpp` guards the `fsr2.hpp` include with `GLLIB_HAS_FSR2`. No example
  linked it.

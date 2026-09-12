# 41_multibounce — implementation status

Brute-force reference for `multi-bounce-gbuffer.md`, **variant A** (the recursive
N³ formulation). The design document's own section 8.2 orders the work:

> 1. **Offline single camera.** Brute-force hemicube, full-res mesh, no
>    optimization. Compare integrated irradiance against a path-traced reference.
>    Validates the integration math (§4.4) before any performance work. Getting
>    solid-angle weighting wrong here poisons everything downstream and is very
>    hard to spot later.
> 2. Reuse radius experiment (§8.1).
> 3. **Compute rasterizer**, one camera, full-res mesh, correctness only.
> 4. Cluster DAG LOD. *This is where the technique either works or doesn't.*

This example is milestones 1 and 3, generalized from one camera to one per
G-buffer pixel, plus the variant A recursion on top. Milestones 2 and 4 are
deliberately absent: both are approximations, and this is the thing they are
supposed to be asserted against.

## Build & run

```bash
cmake -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target 41_multibounce
cd build-release/examples/41_multibounce && ./41_multibounce
```

The binary must run with its own build directory as the working directory: the
`.glb`, `shaders/` and both reference PNGs are resolved relative to cwd. Shaders
hot reload from the **copied** `shaders/` next to the binary, so editing a source
shader needs a build (the `41_multibounce_shaders` target is `ALL`) before the
reload sees it.

| Var | Effect |
|---|---|
| `MBG_GATE=all` or a comma list | run the analytic gates and exit. Names: `quad`, `closed`, `rect`, `occ`, `oracle`, `series`. `MBG_GATE_VERBOSE=1` prints each rasterizer/oracle disagreement |
| `MBG_MODEL=path.glb` | model to load (default `CornellBoxOriginal.glb`). The two reference PNGs only match that one |
| `MBG_BOUNCES=n` | camera levels, 1–4. 1 = direct only |
| `MBG_SCALE=n` | GI grid = framebuffer / n. **1 = one camera per pixel**, the doc's correctness reference |
| `MBG_BUDGET=n` | level-1 cameras per frame. The sweep is split into `ceil(pixels/budget)` chunks; the budget is clamped down if the camera tree would exceed 512 MB |
| `MBG_RES=n` / `MBG_RES2/3/4=n` | hemi-octahedral target edge per level, 2–32 |
| `MBG_BLOCK=n` / `MBG_BLOCK2/3=n` | spawn tile edge in texels. **1 spawns a child per texel — the unabridged recursion** |
| `MBG_SKY=f` | radiance of an uncovered texel. **0.05 is needed to reproduce the full-GI reference**; see finding 5 |
| `MBG_EMISSIVE=f` | emissive scale |
| `MBG_BIAS=f` | camera offset along its own normal, world units (default = 2.5e-4 × scene diagonal) |
| `MBG_PLANE=f` | upsample plane cutoff, world units (default = 0.01 × scene diagonal) |
| `MBG_TWOSIDED=1` | force every triangle to emit from both faces |
| `MBG_GTCAM=0\|1\|2` | free camera \| reference camera at 512² (pixel-aligned against the PNGs) \| reference camera at 1600×900 |
| `MBG_SOLVE=n` | run n complete sweeps before the first present, then hold, and print the per-sweep wall clock |
| `MBG_COMPARE=1` | RMSE/MAE against the reference selected by `MBG_REF`, in display space |
| `MBG_REF=0\|1` | which reference: direct-lighting or full-GI |
| `MBG_VIEW=n` / `MBG_TONEMAP=n` / `MBG_EXPOSURE=f` | initial display mode, tone curve, exposure |
| `MBG_BENCH=n` | run n frames, print wall-clock percentiles + per-pass GPU averages, exit |
| `MBG_SHOT=path` | write a screenshot before exiting |
| `MBG_NOGUI=1` | skip the ImGui overlay — **required** for a clean screenshot |
| `MBG_PAUSE=1` | start with the solver held |

## What is implemented

| Doc section | What | Status |
|---|---|---|
| 4.1 | One camera per G-buffer pixel of a 1/N grid, oriented along the normal | ✅ |
| 4.1 | Camera **clustering** by position and normal, world-radius cap, per-frame jitter | ⬜ out of scope by design — this is the zero-reuse reference §8.1 measures against |
| 4.3 | Hemi-octahedral hemisphere map, one square target, one pass | ✅ |
| 4.4 | Cosine × solid-angle weighted integration | ✅ integrated per texel, not point sampled |
| 4.4 | SH2 projection | ⬜ deliberately not done — see "Deviations" |
| 5.2 | Cluster DAG LOD, meshlets, boundary locking | ⬜ milestone 4, and the whole point of having this reference first |
| 5.3 | Compute rasterizer, one workgroup per camera, LDS depth buffer, atomicMin | ✅ with a 32-bit key instead of 64 — see "Deviations" |
| 5.3 | Visibility buffer resolved in a second pass | ✅ and better: it never leaves shared memory |
| 5.4 | Work list, cluster/camera binning, vertex amortization | ⬜ nothing to amortize at 32 triangles |
| 6.1 | Recursive levels, resolve deepest → primary, terminating in direct only | ✅ |
| 6.2 | Resolution falloff, clustering at depth | ✅ per-level target edge and spawn tiles |
| 6.2 | Russian roulette by throughput, importance-based spawning | ⬜ both are variance/bias trades a reference should not make |
| 7 | Variant B: object-space cache, temporal feedback | ⬜ a separate example; this is what validates it |
| 8.2 §1, §3 | Analytic integration gates, rasterizer vs oracle | ✅ 21 assertions, all passing |
| 8.2 §6 | Variant A end to end, producing reference images | ✅ |

## Layout

| File | Role |
|---|---|
| `main.cpp` | window, scene load, frame graph, ImGui, env hooks, the reference comparison |
| `scene.hpp/.cpp` | triangle extraction, the GPU triangle soup, the quadrature table |
| `solver.hpp/.cpp` | the recursive camera schedule: allocation, chunking, dispatch, the gate entry points |
| `screen.hpp/.cpp` | G-buffer, geometry pass, references, display — adapted from example 40 |
| `validate.hpp/.cpp` | the analytic gates |
| `gpu_util.hpp/.cpp` | `PassTimer`, `Pipeline`, camera control — copied from example 40 unchanged |

Shaders: `place.comp` (camera placement), `raster.comp` (the software rasterizer,
the integration and the spawn), `gather.comp` (fold the children in),
`upsample.comp` (GI grid → full res), plus the G-buffer and display raster pairs.
`shaders/common/` carries `hemi.glsl` (the map, the clipper, the packed key),
`scene.glsl` (structs, bindings, ONB), and `oct.glsl`, `gbuffer.glsl`,
`brdf.glsl`, `tonemap.glsl` copied from example 40.

## Units

```
emission   emitted RADIANCE  L_e  [W/m²/sr]
irradiance IRRADIANCE        E    [W/m²]
outgoing   L_out = L_e + albedo * E / PI
```

The `/PI` belongs to the **emitting** surface. It exists in exactly two places:
`shaders/gather.comp` (transport) and `shaders/common/brdf.glsl` (display). The
`series` gate fails immediately if either is wrong, which is the only reason to
trust that they are not — a missing or doubled PI is a constant factor, it is
invisible in an image, and it is consistent across near and far so no internal
cross-check catches it.

## Deviations from the design document, and why

**The hemi-octahedral map is piecewise PROJECTIVE, and the rasterizer uses that.**
§4.3 says "octahedral mapping is nonlinear, so triangle edges are technically
curved. With LOD holding triangles near pixel size, the linear-edge
approximation error stays well under a pixel and can be ignored." That reasoning
is unavailable here — this example rasterizes the full-resolution mesh, where a
Cornell wall is two triangles each covering a quarter of the hemisphere — and it
turns out not to be needed. Within a region of fixed `sign(d.x), sign(d.y)` the
map is `d -> (A d)/(c·d)`, a projective map of the direction, and projective maps
take great circles to straight lines. So a triangle edge projects to an exactly
straight segment provided the triangle is first clipped at the two folds. Each
quadrant is a projective view exactly the way a hemicube face is; the square
packs four of them into one target instead of five faces into five, which is
§4.3's reason for preferring it and is still the right call. See
`shaders/common/hemi.glsl`.

**A 32-bit visibility key instead of §5.3's uint64.** GLSL has no portable 64-bit
shared atomic — `NV_shader_atomic_int64` and its AMD counterpart are vendor
extensions — so the key is `depth16 << 16 | triangle16`, capping the scene at
65535 triangles (asserted on the host) and depth at 1/65535 of the far distance.
Neither limit is fundamental; the same kernel with a wider key is the production
form. The cap is why this example ships with a 32-triangle scene and why the
`Scene::build` failure path exists.

**The visibility buffer never reaches global memory.** §5.3 says "single
write-out to global memory at the end", and §5.3's separate resolve pass then
reads it back. Here the integration AND the spawn both consume the shared buffer
in place, so there is no write-out at all. At the default 3-bounce schedule that
saves about 350 MB of round-trip traffic per sweep. The visibility dump exists
only behind `u_dump`, for the `oracle` gate.

**No SH2 projection.** §4.4 says to project the resolved hemisphere to SH2 and
store 8 bytes per entry. SH2 exists so that a receiver which is not the camera
can reconstruct irradiance for its own normal — it is the representation that
makes §4.1's clustering possible. With reuse radius zero, the receiver IS the
camera and its normal IS the hemisphere's axis, so projecting to SH2 and
evaluating along +Z would band-limit a number this example already has exactly.
Adding SH2 here would import the clustering variant's error into the reference
that is supposed to bound it. It belongs with §4.1, in the example after this.

**Direct lighting comes out of the same gather.** §3 lists "no correct area-light
soft shadows from the GI pass. Direct lighting stays a separate conventional
pass" as a non-goal. Cornell's only light is emissive geometry, so a separate
direct pass would need an area-light estimator (example 40 has one, 251 lines of
`nee.glsl`) and this example would then not be testing the thing it is for. The
consequence is measured in finding 2: at 32×32 the emitter panel is resolved by
a few tens of texels and the direct term carries a few percent of quadrature
noise. That is the honest cost of taking the technique undiluted, and it is
exactly why §3 says what it says.

**Jacobi over chunks, not a per-frame N³.** The full tree does not fit in memory
at a useful resolution (finding 6), so a sweep is split into chunks of
`MBG_BUDGET` level-1 cameras, each carrying its whole subtree, with the GI image
persisting between frames. This is progressive rendering and **not** variant B:
nothing reads last frame's radiance, so a completed sweep is a correct variant-A
image rather than a converging one.

## Gates

`MBG_GATE=all`, 21 assertions, all passing:

| Gate | Asserts | Result |
|---|---|---|
| `quad` | the quadrature table integrates the hemisphere **before** normalization | `sum(dΩ)` = 2π and `sum(cos dΩ)` = π to 1e-8 relative at 8², 16², 32² |
| `closed` | a camera sealed inside an emitter of radiance L reads exactly πL, at five orientations and two resolutions | worst orientation off by 7.5e-8 — i.e. the rasterizer leaves no cracks and double coverage costs nothing |
| `rect` | a camera under a rectangle matches the analytic form factor | 20.0% at 8², 4.3% at 16², **1.1% at 32²**, and the gate asserts the halving, not just the bound |
| `occ` | an opaque panel between camera and emitter takes it to zero | exactly 0 |
| `oracle` | the compute rasterizer vs a CPU ray cast, texel for texel, 64 cameras on Cornell's own surfaces | **0 disagreements in 65536 texels** (see finding 1) |
| `series` | N camera levels in a closed box of albedo ρ read πL(1+ρ+…+ρ^(N-1)) | 4.8e-8, 2.0e-7, 1.1e-7 relative at 1, 2, 3 levels |

`closed` and `series` between them pin down everything a picture cannot: the
solid-angle weights, the absence of cracks, the `/PI`, the per-tile albedo mass,
and the deepest-level-first resolve order.

## Measurements

All at the reference camera, 512×512, one complete sweep, RMSE in display space
against the path-traced PNG. **Timings are llvmpipe (software GL, 4 CPU threads)
— they are not GPU numbers.** The camera/texel/triangle-raster counts beside them
are hardware independent and are the thing to scale.

### Bounce count (GI 64×64, targets 32/8/8/8, tile 4, sky 0)

| Bounces | vs direct ref | vs full-GI ref | cameras/sweep | texels/sweep | sweep |
|---|---|---|---|---|---|
| 1 | **0.0557** | 0.1511 | 4.1e3 | 4.2e6 | 0.11 s |
| 2 | — | 0.0885 | 2.7e5 | 2.1e7 | 1.7 s |
| 3 | — | **0.0808** | 1.3e6 | 8.8e7 | 8.2 s |
| 4 | — | 0.0821 | 5.5e6 | 3.6e8 | 34 s |

### Spawn tile size (2 bounces, GI 64×64)

| Tile | children/camera | vs full-GI ref | cameras/sweep | sweep |
|---|---|---|---|---|
| 8 | 16 | 0.0997 | 7.0e4 | 0.54 s |
| 4 | 64 | 0.0885 | 2.7e5 | 1.7 s |
| 2 | 256 | 0.0859 | 1.1e6 | 6.0 s |
| 1 | 1024 | 0.0845 | 4.2e6 | 21 s |

### The configuration these images were taken at

| Config | vs reference | cameras/sweep | texels/sweep | sweep |
|---|---|---|---|---|
| 1 bounce, GI 128×128, 32², sky 0 | **0.0577** vs direct | 1.6e4 | 1.7e7 | 0.76 s |
| 3 bounces, GI 128×128, 32/16/8, tile 4, sky 0.05 | **0.0642** vs full GI | 1.8e7 | 1.4e9 | 111 s |

### The sky term (3 bounces, GI 64×64, tile 4, vs the full-GI reference)

| `MBG_SKY` | RMSE | MAE |
|---|---|---|
| 0 | 0.0808 | 0.0600 |
| **0.05** | **0.0671** | **0.0444** |
| 0.10 | 0.0843 | 0.0649 |

### Deeper-level target resolution (3 bounces, tile 4, sky 0.05)

| Ladder | RMSE | cameras/sweep | sweep |
|---|---|---|---|
| 32/8/8 | 0.0671 | 1.3e6 | 8.2 s |
| 32/16/8 | 0.0638 | 4.5e6 | 27 s |

3.3× the cameras for 5% of the RMSE. The document's own §6.2 ladder (32² → 12² →
6²) is available through `MBG_RES2`/`MBG_RES3`; the default stays 32/8/8 because
deep bounces are low frequency and this is where that claim holds up.

### Level-1 target resolution (1 bounce, vs the direct reference)

| Target | RMSE | sweep |
|---|---|---|
| 8×8 | 0.1592 | 0.038 s |
| 16×16 | 0.0714 | 0.051 s |
| 32×32 | **0.0557** | 0.109 s |

### Camera density (1 bounce, 32×32 target, vs the direct reference)

| Scale | cameras/sweep | RMSE | sweep |
|---|---|---|---|
| 8 | 4.1e3 | **0.0557** | 0.11 s |
| 4 | 1.6e4 | 0.0577 | 0.47 s |
| 2 | 6.6e4 | 0.0564 | 3.3 s |
| 1 (per pixel) | 2.6e5 | 0.0561 | 12 s |

## Findings

### 1. The fold clip is not optional, and neither is a tolerance on it

Clipping each triangle at the two octahedral folds before projecting is what
makes straight-edge rasterization exact (see "Deviations"). Doing it exposes a
second, smaller problem: at an even target edge, texel (i,i) and texel
(i, res−1−i) have their **centres exactly on a fold**. A triangle spanning that
fold is split there, both pieces' shared edge passes exactly through the centre,
both edge functions evaluate to zero plus float noise, and if both round the
wrong way the texel is left empty — a hole in a solid wall.

Measured, before the fix: 3 empty texels in 65536 over 64 Cornell cameras, every
one of them on a fold, which `MBG_GATE_VERBOSE=1` prints. Tiny, but it is a hole
in a visibility buffer, and it carried 2.6e-4 of the hemisphere's energy.

The fix is to let the pieces **overlap** by a hair rather than risk a gap: the
edge test accepts `w >= -eps` with `eps` scaled by the piece's own area.
atomicMin makes double coverage free — both pieces compute the same depth from
the same plane — so the asymmetry is entirely in our favour. After it: 0
disagreements with the oracle, and the `series` gate's 3-level error improved
from 4.8e-6 to 1.1e-7.

### 2. The direct term's error is quadrature, and it converges as expected

Against the direct-lighting reference, which has no free parameters at all
(`MBG_BOUNCES=1`, `MBG_SKY=0`), RMSE falls 0.159 → 0.071 → 0.056 for 8², 16²,
32² targets, and the `rect` gate shows the same sequence against an analytic
answer (20.0% → 4.3% → 1.1%). Coverage is binary per texel, so the error lives on
the emitter's silhouette. At 32² the Cornell panel is resolved by a few tens of
texels, which is where the residual mottling in the image comes from — not from
noise in any stochastic sense; the solver is deterministic and a sweep is
repeatable bit for bit.

This is §3's non-goal ("direct lighting stays a separate conventional pass")
showing up as a number. An area-light estimator for the direct term would remove
essentially all of it.

### 3. Camera density is not where the error is

One camera per 8×8 block of pixels scores **the same** as one camera per pixel
(0.0557 vs 0.0561) for 64× fewer cameras and 100× less time. Irradiance is low
frequency, so the coarse grid plus a joint-bilateral upsample loses nothing
measurable — and scale 8 is fractionally *better* than scale 4 because the
upsample's blur suppresses the per-camera quadrature error of finding 2.

That is the single most useful number here for anyone planning the shipping
configuration: at Cornell's scale, spend on target resolution and bounce depth,
not on camera count. It is also a warning about §8.1 — the reuse-radius
experiment is about clustering cameras across *world-space* distance, and a
per-pixel grid at 512² over a 2 m box is already sampling every ~4 mm.

### 4. Tile clustering fails structurally, not statistically

§6.2 offers "massive clustering at depth" as a cost lever on the grounds that the
error is attenuated by albedo multiplications before it reaches the eye. It is,
but the error is **correlated across neighbouring receivers**, so it does not
look like attenuated noise — it looks like geometry.

With tile 8 (16 children per camera) the 2-bounce image carries hard-edged
rectangular patches on the back wall, and at `MBG_SCALE=1` — where there is no
upsample to blur anything — they resolve into large polygonal streaks across
every surface. The mechanism: each tile reuses one representative hit point's
irradiance for a whole solid-angle tile, and as the receiver moves the
representative **flips discontinuously** from one surface to another (ceiling to
emitter, wall to floor), so the reused value jumps along a line in screen space.

RMSE barely registers this — 0.0997 at tile 8 against 0.0845 at tile 1, a 15%
spread for 40× the cost — which is itself the finding: **RMSE is the wrong metric
for this error class.** The images are not 15% apart, they are different in kind.
Tile 4 is the default because it is where the structured artifacts stop being
visible on this scene, not because of its RMSE.

The obvious next move is to jitter the representative choice per camera, which
converts the structure into noise a temporal filter can absorb. That is the right
answer for variant B and the wrong one here: a reference has to be deterministic.

### 5. The full-GI reference needs `MBG_SKY=0.05`, the direct one needs nothing

The Cornell box is open at the front, so a camera on a side wall sees the world
through the opening. Example 40 established (its finding 6) that the full-GI PNG
was rendered with Blender's default world grey and that 0.05 reproduces it; its
corner pixel is (58,58,58) against the direct render's (0,0,0). The same value
behaves the same way here: 3 bounces at tile 4 scores 0.0808 at sky 0 and
**0.0671** at sky 0.05.

So the direct reference is the primary gate — one bounce, sky 0, no free
parameters — and the full-GI reference is the secondary one with one declared
parameter.

### 6. The N³ is real, and memory hits before time does

Camera counts at the default schedule, per level: 4.1e3 → 2.6e5 → 1.0e6 → 5.5e6.
Each level's cameras need 48 bytes of state (position, normal, irradiance, tile
mass), so a 4-bounce tree over a 64×64 GI grid is already 330 MB, and a 1600×900
framebuffer at `MBG_SCALE=4` would want 5 GB before it wanted a second of compute.
That is why the solver chunks: `MBG_BUDGET` level-1 cameras at a time, each
carrying its whole subtree, with a hard clamp at 512 MB that halves the budget
until the tree fits and says so.

Time then scales with the same tree: ×16 for the second bounce, ×4.8 for the
third, ×4.2 for the fourth — while RMSE goes 0.151 → 0.089 → 0.081 → 0.082. **The
fourth bounce costs 4× and changes nothing measurable** (0.0013 RMSE, within the
noise of the sky-term choice), which answers the question `kMaxLevels = 4` exists
to ask. Three levels is the configuration; the doc's own §6.1 stops at three for
the same reason.

### 7. Per-pass timers and `MBG_SOLVE` do not mix

A short `MBG_BENCH` run with `MBG_SOLVE=1` reports `Place` at 148 ms and
`Raster L1` at 0.97 ms for a level with 4096 cameras and 4.2 M texels, which
cannot be true. Two things combine:

- `MBG_SOLVE` runs the entire sweep inside frame 0, so every kernel's FIRST
  dispatch happens inside its own query — and under llvmpipe the first dispatch
  is where the shader is JIT-compiled. `Place` is the first compute dispatch in
  the frame and absorbs it.
- `PassTimer` double-buffers one query per pass, so when a sweep runs several
  chunks in one frame only the last chunk's query survives to be read back.

In steady state — one chunk per frame, which is how the example runs
interactively — the numbers behave: at 1600×900, 2 bounces, budget 2048, the
same timers report `Raster L1` 609 ms against `Raster L2` 3.9 ms, `Resolve`
0.3 ms and `Place` 60 ms, which is the shape the workload predicts.

So: the per-sweep wall clock printed by `MBG_SOLVE`, bracketed by `glFinish` on
both sides, is the number quoted throughout this document, and the per-pass
timers are for steady-state profiling on real hardware.

## What this does not answer

- **§8.1, the reuse radius experiment.** The gate the doc puts before everything
  else. This example is the instrument for it, not the experiment: it can place a
  camera at P and integrate exactly, so plotting error against receiver distance
  is now a small amount of host code. Finding 4 is the same failure at tile
  granularity and suggests the answer will not be generous.
- **Whether rasterization beats ray traversal.** The premise of the whole
  document. Answering it needs §5.2's LOD, §5.4's binning, and a scene where
  "every camera loops every triangle" is not a viable strategy — Cornell's 32
  triangles make this example's raster loop trivial and its occupancy terrible
  (64 threads, 32 triangles, so half the workgroup idles). The cost model here is
  texels, not triangles, and that is not the regime the document is about.
- **Glossy transport, variant B, and anything temporal.** Out of scope by design.

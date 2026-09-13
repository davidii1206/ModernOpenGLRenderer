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
| `MBG_GATE=all` or a comma list | run the analytic gates and exit. Names: `quad`, `closed`, `rect`, `occ`, `oracle`, `texeldir`, `series`, `paths`. `MBG_GATE_VERBOSE=1` prints each rasterizer/oracle disagreement |
| `MBG_MODEL=path.glb` | model to load (default `CornellBoxOriginal.glb`). The two reference PNGs only match that one |
| `MBG_BOUNCES=n` | camera levels, 1–`kMaxLevels`. **3 is the cap** (solver.hpp); 1 = direct only |
| `MBG_PATHS=n` | **single-sample continuation**: split this many ways at the primary hit, branch factor 1 below it, so the cost is `paths × bounces` instead of `K^bounces` (default 40, the split that costs exactly what the branching tree costs at 3 bounces; 20 measures the same for half the cameras, see finding 21). `0` selects the branching tile estimator; see finding 20 |
| `MBG_RR=f` | Russian-roulette threshold on path throughput (default 0.15). Below it a path survives with probability `throughput/f` and is divided by it. 0 disables |
| `MBG_IMPORTANCE=0\|1` | draw the continuation from the micro-buffer's radiance rather than from `cos × dΩ × albedo` alone (default 1) |
| `MBG_SCALE=n` | GI grid = framebuffer / n (default 4). **1 = one camera per pixel**, the doc's correctness reference. Finer resolves creases and costs silhouettes — see finding 21 |
| `MBG_BUDGET=n` | level-1 cameras per frame. The sweep is split into `ceil(pixels/budget)` chunks; the budget is clamped down if the camera tree would exceed 512 MB |
| `MBG_RES=n` / `MBG_RES2/3/4=n` | hemi-octahedral target edge per level, 2–32 (defaults 16/8/8/8) |
| `MBG_JITTER=0\|1` | rotate each receiver's tangent frame by a hash of its position (default 1). **Load-bearing**; see finding 14 |
| `MBG_FILTER=n` / `MBG_FILTER_R=n` | a-trous denoise iterations over the GI grid and taps per side (default 2, 2); see finding 17 |
| `MBG_LV_RES=n` | edge of the secondary cameras' light view (default 8) |
| `MBG_INDIRECT_ONLY=1` | composite the bounce term alone, for inspecting it |
| `MBG_BLOCK=n` / `MBG_BLOCK2/3=n` | spawn tile edge in texels. **1 spawns a child per texel — the unabridged recursion** |
| `MBG_NEE=0\|1` | evaluate the direct term analytically instead of taking it from the raster (default 1). **The single most important setting for image quality**; see finding 2 |
| `MBG_TENT=0\|1` | spread each texel's mass over the four nearest spawn tiles (default 1) |
| `MBG_DIRECT_PIXEL=0\|1` | rasterize a hemisphere **per pixel** for the image's direct term instead of taking it from the GI grid (default 1); see finding 11 |
| `MBG_DIRECT_RES=n` | edge of that pass's per-emitter light view (default 8); see finding 13 |
| `MBG_SKY=f` | constant radiance added to every uncovered texel in every direction. **0.05 is needed to reproduce the full-GI reference**; see finding 5 |
| `MBG_DAYLIGHT=1` | a sun and a sky dome through the box's open side. Off by default, so nothing above it changes; see finding 15 |
| `MBG_SKY_ZENITH=` / `_HORIZON=` / `_GROUND=` | the dome, as `f` for a grey or `r,g,b` for a colour. Override whatever `MBG_DAYLIGHT` set |
| `MBG_SKY_UP=x,y,z` | world up for the dome's gradient (default `0,1,0` — glTF is Y-up) |
| `MBG_SUN=f\|r,g,b` | irradiance on a surface facing the sun. 0 skips the sun's rasterization entirely |
| `MBG_SUN_DIR=x,y,z` | toward the sun |
| `MBG_SUN_ANGLE=deg` | the sun's angular **radius**; larger is a softer shadow at no extra cost |
| `MBG_EMISSIVE=f` | emissive scale |
| `MBG_BIAS=f` | camera offset along its own normal, world units (default = 2.5e-4 × scene diagonal) |
| `MBG_PLANE=f` | upsample plane cutoff, world units (default = 0.01 × scene diagonal) |
| `MBG_TWOSIDED=1` | force every triangle to emit from both faces |
| `MBG_GTCAM=0\|1\|2` | free camera \| reference camera at 512² (pixel-aligned against the PNGs) \| reference camera at 1600×900 |
| `MBG_SOLVE=n` | run n complete sweeps before the first present, then hold, and print the per-sweep wall clock |
| `MBG_COMPARE=1` | RMSE/MAE against the reference selected by `MBG_REF`, in display space |
| `MBG_REF=0\|1` | which reference: direct-lighting or full-GI |
| `MBG_VIEW=n` / `MBG_TONEMAP=n` / `MBG_EXPOSURE=f` | initial display mode, tone curve (0 ACES, 1 Reinhard, 2 clamp, 3 filmic, 4 AgX; default 1, see finding 8), exposure |
| `MBG_GATE_RES=n` / `MBG_GATE_BLOCK=n` | target and tile size for the `series` and `texeldir` gates, for bisecting a failure |
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
| 3 | Direct lighting as a separate conventional pass, not out of the GI gather | ✅ analytic polygon irradiance, visibility from the rasterized depth sort — per camera for the transport, per pixel for the image |
| 5.2 | Cluster DAG LOD, meshlets, boundary locking | ⬜ milestone 4, and the whole point of having this reference first |
| 5.3 | Compute rasterizer, one workgroup per camera, LDS depth buffer, atomicMin | ✅ with a 32-bit key instead of 64 — see "Deviations" |
| 5.3 | Visibility buffer resolved in a second pass | ✅ and better: it never leaves shared memory |
| 5.4 | Work list, cluster/camera binning, vertex amortization | ⬜ nothing to amortize at 32 triangles |
| 6.1 | Recursive levels, resolve deepest → primary, terminating in direct only | ✅ |
| 6.2 | Resolution falloff, clustering at depth | ✅ per-level target edge and spawn tiles |
| 6.2 | Russian roulette by throughput, importance-based spawning | ✅ both, in the path estimator — and neither is a bias trade: roulette divides survivors by their survival probability and importance sampling divides by its own pdf, so both are exactly unbiased. The `paths` gate asserts it. Finding 20 |
| — | Single-sample continuation (branch factor 1), i.e. path tracing the recursion instead of branching it | ✅ the default; `MBG_PATHS=0` restores the doc's branching variant A |
| 7 | Variant B: object-space cache, temporal feedback | ⬜ a separate example; this is what validates it |
| 8.2 §1, §3 | Analytic integration gates, rasterizer vs oracle | ✅ 35 assertions, all passing |
| 8.2 §6 | Variant A end to end, producing reference images | ✅ |
| — | Sky dome and sun (not in the doc; the doc's scene is closed) | ✅ dome from the empty texels, sun analytic with a rasterized cone — finding 15 |

## Layout

| File | Role |
|---|---|
| `main.cpp` | window, scene load, frame graph, ImGui, env hooks, the reference comparison |
| `scene.hpp/.cpp` | triangle extraction, the GPU triangle soup, the quadrature table |
| `solver.hpp/.cpp` | the camera schedule: allocation, chunking, dispatch, the gate entry points. `level_children()` is the entire difference between the branching and the path estimator |
| `screen.hpp/.cpp` | G-buffer, geometry pass, references, display — adapted from example 40 |
| `sky.hpp` | the sun and the dome: one struct, one `bind()`, the `MBG_DAYLIGHT` preset |
| `validate.hpp/.cpp` | the analytic gates |
| `gpu_util.hpp/.cpp` | `PassTimer`, `Pipeline`, camera control — copied from example 40 unchanged |

Shaders: `place.comp` (camera placement), `raster.comp` (the software rasterizer,
the integration and the spawn), `gather.comp` (fold the children in),
`upsample.comp` (GI grid → full res), `direct_pixel.comp` (the image's direct
term, per pixel), plus the G-buffer and display raster pairs. `shaders/common/`
carries `hemi.glsl` (the map, the clipper, the packed key), `raster.glsl` (the
hemisphere rasterizer and its visibility ratio, shared by every pass that needs
to know what a point can see), `lightview.glsl` (the per-emitter frustum and its
rasterizer, and the sun's cone), `emitter.glsl` (the analytic magnitude, shared
by both direct paths so they cannot drift), `sky.glsl` (the dome's gradient and
the sun's analytic irradiance), `scene.glsl` (structs, bindings, ONB), and `oct.glsl`,
`gbuffer.glsl`, `brdf.glsl`, `tonemap.glsl` from example 40 — `tonemap.glsl` with
an AgX curve added (finding 10).

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

**Direct lighting is a separate pass, because §3 says so and the images agreed.**
The first version of this example took the emitter out of the hemisphere raster
like everything else, on the grounds that it was testing the technique
undiluted. §3's non-goal — "no correct area-light soft shadows from the GI pass.
Direct lighting stays a separate conventional pass" — turned out to be the whole
ballgame for image quality, and finding 2 is the measurement. The direct term is
now exact polygon irradiance with visibility sampled against the hemisphere
buffer that is already in shared memory, so it costs one extra pass over the
emitters and no extra structure at all. `MBG_NEE=0` restores the original
behaviour, and the `rect` gate asserts both estimators separately.

**Jacobi over chunks, not a per-frame N³.** The full tree does not fit in memory
at a useful resolution (finding 6), so a sweep is split into chunks of
`MBG_BUDGET` level-1 cameras, each carrying its whole subtree, with the GI image
persisting between frames. This is progressive rendering and **not** variant B:
nothing reads last frame's radiance, so a completed sweep is a correct variant-A
image rather than a converging one.

## Gates

`MBG_GATE=all`, **35 assertions, all passing**:

| Gate | Asserts | Result |
|---|---|---|
| `quad` | the quadrature table integrates the hemisphere **before** normalization | `sum(dΩ)` = 2π and `sum(cos dΩ)` = π to 1e-8 relative at 8², 16², 32² |
| `closed` | a camera sealed inside an emitter reads exactly πL — at five orientations, on all six faces, and next to an edge | worst case off by 3.7e-5: the rasterizer leaves no cracks, and double coverage costs nothing |
| `rect` | a receiver under a rectangle matches the analytic form factor, both estimators | analytic: 0.02% at every target size, and **resolution-free to 1e-6**. Quadrature: 20.0% / 4.3% / 1.1% at 8² / 16² / 32², and the gate asserts the halving rather than the bound |
| `occ` | an opaque panel between camera and emitter takes it to zero | exactly 0 |
| `oracle` | the compute rasterizer vs a CPU ray cast, texel for texel, 64 cameras on Cornell's own surfaces | **0 disagreements in 65536 texels** (finding 1) |
| `texeldir` | the direct term evaluated at **hit points**, integrated against the analytic answer and cross-checked per texel against a CPU ray cast | mean within 2.4e-5, worst texel within 1.5e-4 (finding 8) |
| `series` | N camera levels in a closed box of albedo ρ read πL(1+ρ+…+ρ^(N-1)) | 5.6e-5, 3.1e-4, 2.7e-4 relative at 1, 2, 3 levels |
| `paths` | the same series, but measured with the **stochastic** estimator: single-sample continuation, importance-sampled and rouletted, over 512 receivers × 64 paths | within **8.7e-6** at every level up to the cap. Roulette at a threshold it cannot reach reproduces the unrouletted answer **bit for bit**; at a threshold that kills roughly half the paths per level it still lands within 1.0e-4 (finding 20) |

`closed` and `series` between them pin down everything a picture cannot: the
solid-angle weights, the absence of cracks, the `/PI`, the per-tile albedo mass,
and the deepest-level-first resolve order. The first 35 all run on the
**branching** estimator, deliberately: they assert exact numbers, and an exact
number is something only a deterministic estimator can be held to. `paths` is
the one that holds the stochastic estimator to the same analytic answer in
expectation — and it is the assertion that found the roulette bug in finding 20,
which no image would have shown.

## Measurements

All at the reference camera, 512×512, one complete sweep, against the matching
path-traced PNG in display space under the default Reinhard curve (finding 10).
RMSE is reported for continuity but **it is nearly blind to this example's
artifacts** — see finding 9, and read the roughness column instead.
**Timings are llvmpipe (software GL, 4 CPU threads) — they are not GPU numbers.**
The camera and texel counts beside them are hardware independent and are the
thing to scale.

### The default configuration

`MBG_BOUNCES=3`, `MBG_SCALE=4` (GI 128×128), 40 paths, targets 16/8/8, analytic
direct with an 8×8 light view per pixel, jitter + denoise on, no environment.

| | RMSE | roughness vs reference | cameras/sweep |
|---|---|---|---|
| 1 bounce vs the direct reference | **0.0430** | 0.91× | 1.6e4 + 262k direct |
| 3 bounces vs the full-GI reference (sky 0.05) | **0.0402** | 0.67× | 1.3e6 + 262k direct |

At one bounce there is no recursion and the two estimators are the same code.

### The two estimators, at the same budget

| 3 bounces, sky 0.05 | RMSE | ceiling roughness | cameras/sweep | sweep |
|---|---|---|---|---|
| **path**, 40 paths (default) | **0.0402** | 0.95 | 1.33e6 | 51.4 s |
| branching tiles (`MBG_PATHS=0`) | 0.0407 | 0.91 | 1.33e6 | 52.5 s |

40 is the split that costs exactly what the tree costs at three bounces, so this
is the same budget spent two ways. They are a wash on every measured axis, and
the tiebreak is that one of them is unbiased (finding 20).

Both moved with finding 16 (the light view now clips to its own frustum): the
full-GI number was **0.0532** before it and the per-frame direct pass was
**15.6 s**. Every table below this one that predates it is marked, and every one
of them was measured with the branching estimator, which is still what
`MBG_PATHS=0` selects.

The roughness column reads 0.67× at three bounces, i.e. this image is noticeably
SMOOTHER than the path-traced reference. That is the denoise, and it is the
honest cost of it — the reference's column is its own Monte Carlo noise floor,
which a filtered estimate has no reason to reproduce. Finding 17 is about the
case where the same filter stopped being honest.

### Where the time goes

One complete sweep of the indirect solve, glFinish-bracketed, and the per-frame
passes measured by holding the solver. **llvmpipe, 4 CPU threads — not GPU
numbers**; the camera and texel counts are what scale.

**READ THE WORK COLUMN, NOT THE TIME COLUMN.** These numbers were taken on a
shared container, and the same three-bounce path sweep has come back at 8.4 s and
at 82 s depending on what else was running at the time. *Within* a batch the
spread is under 1% and the comparisons are sound; *across* batches the clock
means nothing. The camera and texel counts are exact and hardware independent,
and they are the thing to scale with.

**At the shipped configuration** — 3 bounces, 40 paths, GI grid 128×128 — one
complete sweep, three runs in one batch, spread under 1%:

| | time | work |
|---|---|---|
| **indirect, 3 bounces, path 40** | **51.4 s** | 1.33e6 cameras, 8.8e7 texels, 4.2e7 triangle-rasters |
| indirect, 3 bounces, path 20 (same quality) | ~26 s | 0.67e6 cameras |
| indirect, 3 bounces, branching | 52.5 s | 1.33e6 cameras, 8.8e7 texels |
| indirect, 3 bounces, path 64 | 81.7 s | 2.11e6 cameras, 1.38e8 texels |
| indirect, 3 bounces, **with the sun** | ~57 s | + one cone raster per camera |
| direct, per frame | 7.5 s | 262k fitted light views |
| direct, per frame, **with the sun** | 8.4 s | + 262k cone views, one rasterization each |
| everything else, per frame | 13 ms | G-buffer, denoise, upsample, display |

A sweep is four chunks, so a frame in steady state is about 13 s of solve plus
the per-frame direct pass.

**Beyond the cap**, measured with `kMaxLevels` raised, for the record that
justifies the estimator:

| | time | work |
|---|---|---|
| **branching, one batch** | | |
| 4 bounces | 182 s | 5.52e6 cameras, 3.57e8 texels |
| 5 bounces | **707 s** | 2.23e7 cameras, 1.43e9 texels |
| **path 64, one batch** | | |
| 5 bounces | 13 s | 4.21e6 cameras, 2.73e8 texels |
| 8 bounces | **20 s** | 7.36e6 cameras, 4.74e8 texels |

The branching rows are ×3.9 per bounce, which is the level-2 tile count and
nothing else — the exponent, measured. The path rows are flat to within their own
noise across 5→8 bounces while the camera count rises 1.75×. Both shapes are what
the camera counts predict, which is the point: the counts are exact and the clock
is not.

The sun costs **15%** of the per-pixel direct pass, for one more light view per
pixel against the emitter's two rasterizations — and buys a light whose shadow is
resolved per pixel. `MBG_DAYLIGHT=1` raises a three-bounce sweep from 48 s to
49 s: the secondary cameras' cone view is 8×8 and is skipped outright on every
receiver facing away from the sun, which in a box lit through one aperture is
most of them.

Bounce depth used to BE the cost — ×4 per bounce, all the way up. With branch
factor 1 it is a linear term, and what is left at the top of the profile is the
per-frame direct pass. **That is now the most expensive single pass in the
renderer** — 262k per-pixel light views, rebuilt every frame because the camera
may have moved. `MBG_DIRECT_RES` and `MBG_SCALE` are the levers; the denoise and
upsample are free at 13 ms together.

The direct pass is per frame rather than per sweep, and it is a hemisphere per
pixel — 262k workgroups at 512×512, against the 16k the GI grid runs. That is the
doc's own §4.1 reference configuration for a single level, and it buys the one
term that cannot be interpolated.

Roughness is high-pass energy relative to the reference's own, so **both
directions are wrong**: far below 1.0 is an over-smoothed image, far above it is
artifacts. The same configuration with `MBG_DIRECT_PIXEL=0` scores 0.0486 and
**0.61×** at one bounce — measurably smoother than a converged path trace, which
is not a compliment: it is the shadow boundaries being blurred away by the
upsample (finding 11). The 3-bounce figure above is carried by the ceiling
(1.72×), the surface that needs the most bounces and gets its light entirely
through the clustered indirect term.

### Bounce count (GI 64×64, tile 4, sky 0.05) — branching estimator

| Bounces | RMSE vs full GI | cameras/sweep | sweep |
|---|---|---|---|
| 1 | 0.1061 | 4.1e3 | 0.18 s |
| 2 | 0.0586 | 2.7e5 | 3.0 s |
| 3 | **0.0531** | 1.3e6 | 11 s |
| 4 | 0.0530 | 5.5e6 | 47 s |

### Spawn tile size (3 bounces, GI 64×64, sky 0.05)

| Tile | RMSE | sweep | before the direct/indirect split (2 bounces) |
|---|---|---|---|
| 8 | 0.0533 | 1.9 s | 0.0997, with visible panel-shaped patches |
| 4 | 0.0531 | 11 s | 0.0885 |
| 2 | 0.0529 | 117 s | 0.0859 |

### Estimator ablation (3 bounces, GI 64×64, tile 4, sky 0.05)

| | RMSE | sweep |
|---|---|---|
| analytic direct + per-texel split | 0.0531 | 11.2 s |
| quadrature direct (`MBG_NEE=0`) | 0.0555 | 9.2 s |
| no tent weights (`MBG_TENT=0`) | 0.0530 | 10.9 s |

The analytic direct term costs 22% of a sweep and is worth far more than the
0.0024 of RMSE it shows here — see findings 2 and 9.

### Level-1 target resolution (1 bounce, vs the direct reference)

| Target | RMSE | analytic direct term | quadrature direct term |
|---|---|---|---|
| 8×8 | 0.0529 | exact (0.02% on the `rect` gate) | 20.0% |
| 16×16 | 0.0489 | exact | 4.3% |
| 32×32 | **0.0468** | exact | 1.1% |

What still improves with resolution is the *indirect* quadrature and the
visibility test's silhouette, not the direct magnitude.

### The sky term (3 bounces, GI 64×64, tile 4)

| `MBG_SKY` | RMSE vs full GI |
|---|---|
| 0 | 0.0808 |
| **0.05** | **0.0671** |
| 0.10 | 0.0843 |

(Measured under ACES before the tone curve was changed; the ordering is what
matters and it matches example 40's finding 6 exactly.)

### Camera density (3 bounces, sky 0.05)

| Scale | GI grid | cameras/sweep | RMSE | sweep |
|---|---|---|---|---|
| 8 | 64×64 | 1.3e6 | 0.0531 | 11 s |
| 4 | 128×128 | 5.3e6 | 0.0538 | 45 s |

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

### 2. The direct term had to come out of the gather, exactly as §3 says

The first version of this example took everything from the hemisphere raster,
including the emitter. The doc lists that as a non-goal -- "No correct area-light
soft shadows from the GI pass. Direct lighting stays a separate conventional
pass" -- and the images said the same thing louder: the panel is the brightest
thing in the scene by two orders of magnitude and subtends a few percent of the
hemisphere, so a texel either sees all of it or none of it, and every camera's
direct term carried a few percent of error uncorrelated with its neighbours. At
the 8x8 targets the deeper levels use, a camera that plainly sees the panel can
miss it with every texel centre, which made the second bounce blotchy on top.

The fix is the estimator now in `raster.comp`:

- **magnitude**: the exact projected solid angle of the emitter polygon clipped
  to the hemisphere (Lambert's formula). No quadrature, no resolution
  dependence. The `rect` gate now reads 0.02% against the analytic form factor
  at 8x8, 16x16 and 32x32 alike, where the quadrature reads 20.0%, 4.3% and 1.1%.
- **visibility**: a contribution-weighted fraction sampled against the depth
  already sitting in shared memory, 16 samples per emitter per camera.

So the only sampled quantity left in the direct term is a number bounded in
[0,1] that is 0 or 1 over most of the image with a ramp across penumbrae -- which
is the error budget an area light actually wants.

**RMSE cannot see any of this.** Against the direct reference the two estimators
score *identically*, 0.0468, while one image is smooth and the other is covered
in mottle. That is finding 9, and it is why this example grew a second metric.
On the high-pass measure the quadrature direct term runs at **2.45x** the
reference's own noise floor and the analytic one at **0.97x** -- at or below the
path tracer's residual.

### 3. Camera density is not where the error is

One camera per 8×8 block of pixels scores the same as one per 4×4 — 0.0531
against 0.0538 at three bounces, for a quarter of the cameras and a quarter of
the time — and at one bounce the whole ladder from scale 8 down to scale 1 (one
camera per pixel) moves RMSE by 0.0004. Irradiance is low frequency, so the
coarse grid plus a joint-bilateral upsample loses nothing measurable, and the
coarser grid is fractionally *better* because the upsample's blur suppresses
what per-camera error remains.

That is the most useful number here for anyone planning a shipping
configuration: at Cornell's scale, spend on bounce depth and on the direct
estimator, not on camera count. It is also a warning about §8.1 — the
reuse-radius experiment is about clustering cameras across *world-space*
distance, and a per-pixel grid at 512² over a 2 m box is already sampling every
~4 mm.

### 4. Tile clustering fails structurally — until the fast part is taken out of it

§6.2 offers "massive clustering at depth" as a cost lever on the grounds that the
error is attenuated by albedo multiplications before it reaches the eye. It is,
but the error is **correlated across neighbouring receivers**, so it does not
look like attenuated noise -- it looks like geometry. With tile 8 the 2-bounce
image carried hard-edged patches shaped like the light panel, and at
`MBG_SCALE=1`, where no upsample blurs anything, they resolved into polygonal
streaks across every surface.

The mechanism is not that one sample is noisy. It is that the tile stands in for
a function that varies FAST across it: the bright pool the panel throws on the
floor falls off as cos/d² over a few texels, and one sample of it smeared over a
tile, with the sample jumping from one surface to another as the receiver moves,
is a moving hard edge.

So the tile's mass is split by how fast the thing it multiplies varies:

| mass | multiplies | why it is safe |
|---|---|---|
| `M_dir` | the child's **visible fraction** | carries the unshadowed direct irradiance evaluated at **every texel's own hit point**, exactly |
| `M_ind` | the child's irradiance **minus its direct term** | the smooth remainder, which is what clustering was always safe for |

The result is that tile size almost stops mattering. Measured at 3 bounces
against the full-GI reference:

| Tile | before the split | after | cost after |
|---|---|---|---|
| 8 | 0.0997 (2-bounce) | **0.0533** | 1.9 s |
| 4 | 0.0885 | 0.0531 | 11.2 s |
| 2 | 0.0859 | 0.0529 | 117 s |
| 1 | 0.0845 | — | — |

An 18% spread became 0.8%, and the configuration that used to be visibly the
worst is now the one to ship. The clustering lever the doc proposes is real; it
just cannot be applied to the direct term.

The **tent weighting** -- spreading each texel's mass over the four nearest tiles
rather than assigning it to one -- was built for the same artifact and is kept,
but it now changes RMSE by 0.0001 (0.0530 against 0.0531). It smooths a term that
is no longer the dominant one. It stays on because the discontinuity it removes
is a property of the scheme rather than of Cornell, and because it costs 3% of a
sweep.

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

Camera counts at the default schedule, per level: 4.1e3 → 6.6e4 → 2.6e5, and a
fourth level would add 1.0e6. Each level's cameras need 96 bytes of state
(position, normal, irradiance, the direct term, and two tile masses), so a
4-bounce tree over a 64×64 GI grid runs to hundreds of megabytes and a 1600×900
framebuffer at `MBG_SCALE=4` would want gigabytes before it wanted a second of
compute.
That is why the solver chunks: `MBG_BUDGET` level-1 cameras at a time, each
carrying its whole subtree, with a hard clamp at 512 MB that halves the budget
until the tree fits and says so.

Time then scales with the same tree: ×17 for the second bounce, ×3.8 for the
third, ×4.1 for the fourth — while RMSE goes 0.106 → 0.059 → 0.053 → 0.053. **The
fourth bounce costs 4× and changes nothing measurable** (0.0001 RMSE), which
answers the question `kMaxLevels = 4` exists to ask. Three levels is the
configuration; the doc's own §6.1 stops at three for the same reason.

What the fourth bounce would have bought is visible in the 3-bounce diff as a
ceiling that is slightly too dark — the surface that needs the most bounces — and
that is the case §7's variant B exists to serve, at one camera level per frame
instead of four.

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

### 8. Two bugs in the analytic direct term, both found by gates, neither visible as "wrong"

The estimator in finding 2 was written, passed four of the six gates, and was
wrong twice. Both failures were in the same three lines, and neither would have
been diagnosable from an image.

**A receiver on the emitter's own plane reads a full hemisphere of it.** Lambert's
formula integrates whatever polygon it is given; an emitter whose plane contains
the receiver is edge-on and contributes nothing, but on the sphere that polygon
degenerates to a great circle, and if the receiver's projection falls inside the
triangle the three edge terms take the same sign and the sum comes out at 2π. So
the formula reports π·L instead of 0 — the worst available answer, at the panel's
radiance of 17, for every hit point on the panel and every child camera a tile
places there. The `series` gate caught it as a **+29% energy overshoot** at two
levels.

**Rejecting on height alone then threw away perpendicular emitters.** The obvious
fix — ignore an emitter when the receiver is within a bias of its plane — also
discards emitters that merely *pass through* the receiver while standing
perpendicular to it, and those cover half its hemisphere rather than none of it.
A symmetric box puts many texel centres precisely on its own edges, so this cost
**12% of the hemisphere at an 8×8 target**, and it survived the `closed` gate
(whose cameras sit in open space) and the `rect` gate (one emitter, no edges). It
took a gate that evaluates the direct term **at hit points** rather than at
cameras — `texeldir`, which integrates the per-texel values against the analytic
answer and cross-checks every one against a CPU ray cast — to localize it. The
fix needs both conditions: coplanar means near the plane *and* parallel normals.

A third version of the same class sat in the shadow test, and is finding 12.

### 9. RMSE is nearly blind to the artifacts, so the example measures roughness too

The two direct-term estimators of finding 2 score the **same** 0.0468 RMSE
against the direct reference. One of those images is smooth; the other is the
mottled one this work started from. Two reasons: a high-frequency error averages
to almost nothing in a per-pixel mean, and the residual is dominated by the tone
curve (finding 10), which is a large smooth offset that drowns everything else.

So `MBG_COMPARE` also reports **roughness**: each pixel minus the mean of its
15×15 neighbourhood, RMS over patches that are smooth and shadow-free in the
reference. The reference is a converged path trace, so its column is the noise
floor of the comparison and anything above it is ours.

| 1 bounce, direct reference | RMSE | roughness vs reference |
|---|---|---|
| quadrature direct term | 0.0468 | **2.45×** |
| analytic direct term | 0.0468 | **0.97×** |

The kernel size is not incidental: the artifacts live at the GI grid's scale, 8
pixels at `MBG_SCALE=8`, and a 5×5 high-pass looks straight past them — on the
same pair of images it reports 1.06 against 0.54 where the 15×15 reports 3.93
against 1.31. A metric has to be tuned to the artifact it is meant to see, and
the honest version of that statement is that **the images remain the primary
instrument** and both numbers are supporting evidence.

### 10. The references are not reproduced by AgX, and the tone curve now costs as much as the transport

Example 40 reasoned that both PNGs must be Blender AgX renders — the panel lands
near-neutral without clipping, which is AgX's signature — and called an exact
port the follow-up. This example has that port (`MBG_TONEMAP=4`, the standard
approximation of Sobotka's transform). It is **not** the closest match.

Per patch, against the direct reference:

| curve | back wall | short box top | RMSE |
|---|---|---|---|
| reference | (112, 91, 47) | (104, 86, 48) | — |
| **Reinhard** | **(105, 90, 53)** | **(99, 86, 52)** | **0.0468** |
| filmic | (124, 103, 48) | (120, 103, 53) | 0.0482 |
| clamp | (114, 95, 54) | (111, 94, 54) | 0.0486 |
| ACES | (134, 109, 47) | (133, 112, 53) | 0.0532 |
| AgX | (112, 97, 84) | (104, 93, 86) | 0.0649 |

AgX gets the red almost exactly and then desaturates the blue to nearly double
the reference's. Either the renders are not AgX, or the standard approximation
diverges from Blender's OCIO transform at this saturation. Either way the
measurement picks the default, which is now Reinhard, and all five stay
available.

This matters beyond presentation: at RMSE 0.047 against a reference whose curve
we are guessing, **the tone curve contributes about as much error as the light
transport does**. Further transport work on this scene has to be scored on
patches and on roughness rather than on a single number — and the real fix is a
reference rendered through a curve we control.

### 11. The direct term needs its own receiver, not just its own estimator

Finding 2 took the direct term out of the quadrature. That fixed its *magnitude*
and left its *resolution* wrong, because the receiver was still a secondary
camera: one per `MBG_SCALE`×`MBG_SCALE` block of pixels, 4×4 by default, followed
by a joint-bilateral upsample. Direct lighting carries the sharpest edges in the
image — contact shadows, the boundary where a box occludes the panel — and
reconstructing them from receivers four pixels apart makes them soft no matter
how good the upsample is. That is Nyquist, not a filtering failure, and it is the
same limit example 40 hit against its surfel spacing and answered the same way.

So the direct term gets its own receiver: `direct_pixel.comp` rasterizes **a
hemisphere per pixel**. Same rasterizer, same LDS buffer, same atomicMin, same
visibility rule — only the receiver changes. The solve is untouched (every camera
still computes its own direct term, because the transport reads it) and the GI
grid then carries the **indirect residual alone** (`gather.comp`'s `u_split`), so
nothing is counted twice.

| | RMSE | roughness |
|---|---|---|
| direct from the GI grid | 0.0474 | 0.69× |
| **direct per pixel** | **0.0431** | **0.91×** |

The roughness figure is the interesting one. 0.61× said the image was *smoother
than the converged path trace* — detail lost to the upsample, which is not a
compliment. Per pixel it sits at 1.02×, carrying the same high-frequency content
the reference does.

### 12. Visibility is the depth sort, and nothing else

Getting there took three attempts, and the two failures are worth recording
because both looked reasonable.

**Attempt 1 — sample the emitter, test against the stored depth.** Pick points on
the emitter, find the texel each falls in, compare its stored depth against the
sample's distance. This is a shadow-map lookup with no slope-scaled bias: the
depth belongs to the texel's *centre*, the sample ray is not that ray, and on a
surface seen at a grazing angle — the ceiling beside the panel, from anywhere on
the back wall — the distance changes by more across one texel than the panel is
far away. Samples nothing was blocking read as occluded, and the visible fraction
stepped by 1/16 as samples crossed texel boundaries, printing a fine diagonal
cross-hatch of the octahedral grid onto every surface.

**Attempt 2 — fix it with a ray cast.** Replacing the depth comparison with a
Möller-Trumbore against the scene removed the artifact and was the wrong answer:
avoiding ray traversal is the entire premise of the technique. A renderer that
rasterizes its transport and then ray casts its shadows has not tested the thing
the document is about.

**What works is a ratio taken off two rasterizations.** The emitters are
rasterized *alone* into a second LDS buffer, then:

```
visible fraction = (cosine-weighted mass of texels the emitter WON in the full buffer)
                 / (cosine-weighted mass it covers with nothing else in the scene)
```

Both come off the same texel grid, so the quantization that made the raw
quadrature estimate useless cancels exactly, and what is left is a proper
fraction: 1 in the open, 0 in umbra, a ramp across a penumbra. Nothing compares a
depth against anything — the atomicMin already did that, per texel, along the
texel's own direction, which is the only place the comparison is exact. Cost is
two extra triangles rasterized and one LDS buffer.

Two details the gates and the roughness metric forced out:

- **Coplanar emitters.** Cornell's panel lies in the ceiling's plane, so the two
  are at identical depth in every texel the panel covers and atomicMin breaks the
  tie by triangle index. The numerator therefore compares the winner's depth
  against the emitter's own, from the two buffers, with one key-step of
  tolerance — still two numbers the same rasterizer produced for the same ray.
- **Sub-texel emitters.** When the emitter wins no texel centre the denominator
  is zero and there is no ratio. Answering "not visible" there put dark speckle
  along every surface junction — **2.64×** the reference's roughness, against
  1.37× when the same case answered "visible". It now degrades to a single depth
  comparison at the texel the emitter's centroid falls in, which lands at
  **1.02×**. Sub-texel lights get a one-texel-wide shadow boundary, which is the
  honest answer at that resolution.

There is now exactly one visibility mechanism in this renderer — rasterize, let
the depth sort decide, read the winner — and every pass shares it through
`common/raster.glsl`.

### 13. Point the resolution at the light, not at the sky

The per-pixel receiver of finding 11 fixed *where* the direct term is sampled and
left *how well* it resolves the light alone, and the second one is just as
visible. Visibility off the hemisphere buffer is a ratio of texel masses, so its
precision is however many texels the emitter covers:

| receiver | panel's solid angle | texels of a 32×32 hemisphere |
|---|---|---|
| short box top | 2.84% of the hemisphere | 29 |
| floor centre | 0.71% | 7 |
| floor far corner | 0.36% | 4 |

Four to seven distinguishable values across a penumbra is not a soft shadow with
a coarse ramp — it is a hard line with a couple of steps in it, and below one
texel it is a hard line with none. A Cornell box is made of exactly the long soft
penumbrae that shows up in: the room's top edges and the boxes' shadows.

Raising the hemisphere's resolution is the wrong lever. 128×128 would put 64 KB
in shared memory to resolve a light occupying 1% of it, and the LDS buffer that
makes the whole technique affordable is the thing being spent.

The right lever is to point the resolution **at the light**. `direct_pixel.comp`
now fits a perspective frustum to each emitter, rasterizes the emitter alone and
then the whole scene into it, and takes the same ratio off those two buffers. At
a 16×16 target the emitter covers ~256 texels instead of ~5. Same rasterizer,
same atomicMin, same read-the-winner rule — the texels simply point somewhere
more useful, and triangles that miss the frustum are rejected by their bounding
box and cost nothing.

| | RMSE | roughness | direct pass |
|---|---|---|---|
| hemisphere visibility, 32×32 | 0.0439 | 1.02× | 7.4 s |
| **light view, 16×16** | **0.0431** | **0.91×** | 16 s |
| light view, 8×8 | 0.0430 | 0.91× | 5.7 s |
| light view, 4×4 | 0.0431 | 0.92× | 2.8 s |

RMSE barely moves — the penumbra is a small fraction of the image and finding 9
applies — but the shadows go from stepped to smooth. The resolution ladder is
nearly flat because the frustum is *fitted*: even 4×4 spends 16 samples on the
light, against the 4-7 a 32×32 hemisphere manages. The default is 16 because this
is a reference; 8 is free and indistinguishable on this scene.

The limitation is the frustum: an emitter spanning more than about a hemisphere
from the receiver has no bounded one, and `lv_setup` gives up and calls the light
unoccluded. That is right for a panel and wrong for a receiver sitting inside a
glowing box — which is what the hemisphere path, still used by every secondary
camera, is there for.

### 14. It was never a sampling-rate problem — it was a correlation problem

With the direct term fixed, the indirect was still wrong in a way that only
shows once it is looked at on its own (`MBG_INDIRECT_ONLY=1` at 4× exposure,
which is what that switch exists for): **horizontal banding** across the ceiling
and the upper walls, and blotchy patches elsewhere.

Two hypotheses were wrong and are worth recording because both looked right:

- **The second bounce's shadowing is binary.** At an 8×8 target a level-2 camera
  puts *half a texel* on Cornell's panel, so its visible fraction was the
  sub-texel yes/no of finding 12, flipping as a tile's representative moved.
  That is real, and fixing it — the secondary cameras now use a fitted light view
  like the per-pixel pass, so there is one visibility mechanism everywhere —
  changed the banding **not at all**.
- **The spawn tiles are too coarse.** Tile 2 puts 256 children on every level-1
  camera and costs 100 s a sweep. Also unchanged.

The third hypothesis was right about the cause and wrong about the fix. A 32×32
gather hands every receiver the **same 1024 directions**, so its quadrature error
is *correlated between neighbours*: a scene feature crosses a texel boundary at
the same place for a whole row of receivers and their sums step together. That is
why it reads as banding rather than grain, and why the pattern's spacing follows
the GI grid — at `MBG_SCALE=2` it becomes a fine diagonal weave along the
octahedral folds.

Raising the target to 64×64 does remove it, and that is the wrong lever: it
treats a correlation problem as a sampling-rate problem and pays 4× for it.
Example 40 gets clean indirect out of **8×8** buckets, and says why in one line
of `common/micro.glsl` — its frame rotation is "deterministic across frames and
runs but decorrelated between neighbours, so the coherent octahedral banding on a
flat wall becomes spatial noise instead of a pattern". Two passes, not more
texels:

1. **Rotate each receiver's tangent frame** by a hash of its position
   (`common/raster.glsl`). The error stops being shared between neighbours and
   becomes noise. Hashing the position rather than the camera index keeps a sweep
   reproducible when the chunk scheduler moves a camera to a different slot.
2. **Denoise the GI grid** with an a-trous, normal- and plane-aware filter
   (`gi_filter.comp`) before upsampling. This is only safe because of finding
   11's split: the grid carries the **indirect residual alone**, which is low
   frequency by nature, while the direct term — every sharp edge in the image —
   is added afterwards at full resolution and never touches it. Example 40 makes
   exactly the same split for exactly the same reason.

Neither works alone. Filtering a coherent pattern blurs the pattern; jittering
without averaging trades bands for grain.

With both, the target resolution stops mattering:

| level-1 target | RMSE | ceiling roughness | mean roughness | sweep |
|---|---|---|---|---|
| **16×16** | **0.0532** | 0.88× | 0.82× | 37 s |
| 32×32 | 0.0533 | 0.88× | 0.82× | 38 s |
| 64×64 | 0.0532 | 0.88× | 0.82× | 43 s |

*(measured before finding 16; the same configuration now reads 0.0404 / 0.88× /
0.81× / 48 s. The comparison between rows is unaffected — the change is in the
direct pass, which is identical in all three.)*

So the default went to **16×16** — half the doc's near tier and a sixteenth of
the directions this example briefly thought it needed. The ceiling, which is lit
entirely by bounce and was the worst surface in the image at **1.88×** the
reference's roughness, is now at 0.88×: slightly *under*, which is the denoise
doing its job and the honest cost of it.

The remaining lever is the one §6.2 predicts. Raising level 2 from 8×8 to 24×24
moves the ceiling from 1.30× to 1.12× for **6× the sweep** and 0.0001 of RMSE —
deep bounces are low frequency and do not want the resolution.

### 15. A sky is what the visibility buffer already says; a sun is not

The Cornell box has an open +z side, so adding an environment needs no new scene
and it turns out to need almost no new machinery either — but the two halves of
"sky and sunlight" land on opposite sides of this renderer's one design rule.

**The dome is free, and that is the interesting part.** A secondary camera's
hemisphere is a visibility buffer; a texel no triangle reached is a texel that
sees the sky. `raster.comp`'s quadrature loop already visits every one of them,
already knows the direction, and already has the cosine-weighted solid angle in
`quad[]`. The whole feature is replacing a constant with `mbg_sky_dome(dir)`.
Sky occlusion therefore comes out exactly as correct as the rasterizer is, which
is the same guarantee every bounce already has — no cone tracing, no AO term, no
second structure, and no ray. The Cornell box's open side is a hole in the
geometry and the renderer treats it as one.

**The sun cannot work that way, for the reason the panel could not.** A disc of
1.2° covers 5×10⁻⁴ of a hemisphere; a 16×16 target has 256 texels, so the sun is
a fifth of a texel and a quadrature would find it in one camera out of five and
miss it in the rest. That is finding 2 again, one order of magnitude worse. So
the sun takes the same split as the emitters:

| | magnitude | visibility |
|---|---|---|
| emitter | Lambert's formula, exact | ratio of two rasterizations of a fitted frustum |
| **sun** | `L·Ω·cos θ`, exact | fraction of a rasterized **cone** that nothing reached |

and it is *cheaper* than the emitter's, because a light at infinity has nothing
behind it. There is no emitter pass to divide by: a texel of the disc is lit
exactly when nothing was written into it. It also needs no bias and cannot
self-shadow numerically, because there is no depth comparison at all — the
receiver's own triangle is removed by the receiver-plane cull before a key is
ever packed, and after that, presence is the entire test.

The split between the passes follows finding 11 without modification. The sun is
a direct term, so it goes per **pixel** (`direct_pixel.comp`) where its shadow
boundary is the sharpest thing in the image, and it is subtracted out of the GI
grid by `u_split` exactly as the panel is. The dome is a hemisphere integral, so
it stays on the GI grid: a per-pixel version would be a second copy of the same
integral at forty times the cost, for a term smooth enough that a 4×4 block
resolves it. That division — one direction goes per pixel, the whole hemisphere
goes on the grid — is the same one the emitters already forced, and getting a sky
and a sun for free out of it is the strongest evidence so far that the split is
the right one rather than a Cornell-shaped convenience.

Cost: **+15%** on the per-pixel direct pass, **+1 s on a 48 s** three-bounce
sweep. `MBG_DAYLIGHT=1`, and `renders/06_daylight_3bounce.png`.

Two things are deliberately approximate and both are recorded in
`shaders/common/sky.glsl`. The cosine is clamped rather than integrated over the
part of the disc above the horizon, which is wrong by O(α²) ≈ 3×10⁻⁴ of the sun's
own irradiance and only within a degree of the terminator. And the dome is a
three-colour gradient rather than Preetham or Hosek — this example measures
transport, and all the transport needs from a dome is that its radiance vary with
direction.

### 16. The light view has to clip to its own frustum, and it was costing 2× and 0.013 RMSE

Found while building the sun, because the sun is the case that breaks loudly.

`lv_raster_tri` projected every triangle into the light view and then clamped the
resulting bounding box to the target. For a triangle *outside* the frustum that
still works — the clamp collapses it to one texel column and the edge functions
reject it — but the pixel coordinates it produces are as large as the ratio
between the triangle's angular extent and the frustum's. For an emitter that
ratio is single digits. For the **sun's cone** it is several thousand: a Cornell
wall lands at pixel coordinates of order 10⁴ on an 8×8 buffer.

Two things break there at once, and the second one was already happening on the
emitters:

- float32 edge functions built from products of 10⁴-scale coordinates carry
  absolute noise comparable to `lv_fill`'s tolerance;
- that tolerance is `1e-5 × |area2|`, i.e. a fixed fraction of the triangle's own
  area, so a triangle projected 10³× too large is dilated by a fraction of a
  texel. The occluder comes out slightly bigger than it is.

Clipping the polygon against the frustum's four side planes first bounds every
coordinate by construction. The side planes of a perspective frustum all pass
through the receiver, so they are great circles in direction space and
`mbg_clip_plane` takes them unchanged — 4 more Sutherland-Hodgman passes, and
`MBG_CLIP_MAX` from 8 to 10 to hold the result.

It is not only a correctness fix. The bounding box no longer has to be clamped
down from thousands of texels, so the rasterizer stops iterating texels it will
reject:

| | per-frame direct pass | full-GI RMSE | 1-bounce RMSE |
|---|---|---|---|
| bounding box clamped | 15.6 s | 0.0532 | 0.0430 |
| **clipped to the frustum** | **7.0 s** | **0.0404** | 0.0431 |

Half the time and a fifth off the error, from a bug that produced no visible
artifact — the dilation is sub-texel and it widens *every* occluder by the same
relative amount, so it reads as a uniformly slightly-darker room rather than as
anything wrong. The 31 gates did not catch it either: every one of them is either
unoccluded or fully occluded, and a fractional penumbra error has nowhere to show
up in a binary answer. What caught it was a light small enough to make the same
mistake three orders of magnitude bigger.

### 17. The denoise was amplifying exactly what it was supposed to remove

Reported as "horizontal light streaks on the ceiling", and it was four separate
bugs in `gi_filter.comp` stacking up. None of them was visible before there was a
sun, because none of them creates noise — they all mishandle noise, and the
emitter alone did not produce enough to notice.

The signature that identified it: the ceiling's high-pass noise was strongly
**row-correlated** (row-mean σ 0.26 against a column-mean σ 0.13) with a spectral
peak sitting on exactly **two grid cells**, and it got *worse* with more
iterations — 0.09 at one, 0.22 at three, 0.71 at five. A denoiser whose output
degrades monotonically in its own iteration count is not tuned wrong, it is
wrong.

Two cells is the grid's Nyquist, and that is the tell. An a-trous iteration with
tap spacing 2^k samples every 2^k-th cell, so it has **unity gain** at the
frequencies that alias onto DC at that spacing — Nyquist first among them. The
construction only works if each iteration annihilates what the next, more dilated
one will alias. That is the entire reason the literature's kernel is the B3
spline (1,4,6,4,1)/16, whose response is exactly zero there.

1. **The kernel was not a B3 spline.** It was `1/(1 + dx² + dy²)`, which the
   comment beside it called "B3-spline-ish". Along one axis that is
   (0.2, 0.5, 1, 0.5, 0.2): it passes **17%** of Nyquist, and the dilated
   iterations then pass 100% of it. Replaced with C(2R, R+d) — which *is*
   (1,4,6,4,1) at R=2 and generalizes to any radius, with response cos^2R(ω/2),
   identically zero at Nyquist. Separable, because the null has to hold per axis.
2. **The kernel was truncated at every edge.** A rejected tap was dropped and the
   weights renormalized, and a truncated binomial is not a binomial, so the null
   went away exactly where the artifact was: the ceiling is a thin band in screen
   space, the dilated iterations reach ±8 cells, and taps that far out land on
   the back wall or the sky. Now a rejected tap is replaced by **the centre's own
   value**: the kernel is always whole, the null always holds, and nothing leaks
   across the edge because nothing from the far side is read.
3. **The filter wrote back into the grid it was filtering.** The GI image
   accumulates across frames — a sweep is chunked — so in interactive use every
   cell the chunk cursor was not currently rewriting got re-filtered once per
   frame, forever. That is not "3 iterations", it is an unbounded number of them,
   converging on the filter's own fixed point, which is precisely the content its
   dilated passes cannot attenuate. `gi_disp_` now holds the denoised copy and
   `gi_` keeps the raw solve. (This one is invisible to `MBG_SOLVE`/`MBG_BENCH`,
   which filter once — it only bites the live view, which is where it was seen.)
4. **Three iterations was one too many.** With the kernel fixed, the third
   iteration's tap spacing of 4 still reaches 8 cells, a third of the way across
   a ceiling that is ~23 grid rows tall because it is seen nearly edge on. At
   that reach it is no longer averaging neighbours that share a neighbourhood:

   | iterations | ceiling row σ | col σ | row/col | full-GI RMSE |
   |---|---|---|---|---|
   | 3 | 0.174 | 0.107 | 1.62 | 0.0403 |
   | **2** | **0.125** | 0.113 | **1.11** | 0.0408 |

   Isotropic instead of plainly horizontal, for 0.0005 of RMSE. The default is
   now 2.

After all four, with the sun on, the ceiling's row σ is **0.119 against a column
σ of 0.279** — the anisotropy is not reduced, it is reversed, which is what
"there is no horizontal structure left" looks like in this measurement.

### 18. One visible fraction cannot serve two lights of different size

The other half of the same report, and a genuine transport error rather than a
filtering one.

Finding 4 split the spawn tile's mass in two so that the fast-varying factor
(the geometric falloff) was reconstructed per texel and only the slow one (the
visible fraction) was clustered. That rests on an assumption nobody wrote down
because nothing violated it: **an area light's visible fraction is slow.**
Cornell's panel subtends thirty degrees, its penumbrae are tens of centimetres
wide, and one sample of that per tile is fine.

A sun is not slow. A 1.2° disc is visible or it is not, across a couple of
centimetres. And the child reported a single fraction, energy weighted across
every light — so with the sun carrying the larger irradiance and therefore the
larger weight, **the panel's smooth contribution was being multiplied by the
sun's near-binary answer.** The panel bouncing off the floor is most of the light
on the ceiling, so most of the light on the ceiling inherited the sun's variance,
and finding 17's filter turned that into the streaks.

The fix is finding 4's own logic applied one level further in: one mass per
visibility *character*, not one per light and not one for all of them.

| | mass at the parent | scaled by |
|---|---|---|
| `M_ind` | albedo × solid angle | the child's indirect remainder |
| `M_emit` | unshadowed **emitter** irradiance at each texel's hit point | the child's emitter fraction |
| `M_sun` | unshadowed **sun** irradiance at each texel's hit point | the child's sun fraction |

Costs one more `vec4` per child and one more per camera (`kWeightBytes` 32→48,
`kDirectBytes` 16→32) and nothing in time. Every emitter still shares one
fraction: they all have the same soft-penumbra character, which is the property
the clustering actually depends on. The sun is the one light with a record of its
own because it is the one light that does not.

What this does **not** fix is the sun term's own variance — `M_sun` is still
scaled by one sample of a binary function per tile. Measured: quadrupling the
level-1 children (`MBG_BLOCK=2`, 4× the cost) moves the unfiltered ceiling noise
only 2.9 → 2.2, so the remaining variance is spread down the whole recursion
rather than concentrated at the top, and buying it off with more children is not
the lever. The lever that would work is a sun visibility evaluated per texel
instead of per tile, which needs something the parent does not have.

### 19. A receiver in a concave corner can be outside the room

The second artifact in the same report: a handful of white pixels running down
the red wall's back corner, at full sun, in a column where everything around
them is in shadow. Three of them in the 512² frame — and they were the sun's,
not the panel's: the indirect-only render had none.

A shading point in a corner lies on the adjacent wall as well as on its own, and
*where* it lies on that wall is decided by a depth reconstruction: the pixel's
depth, unprojected, plus a bias along the pixel's own normal — which for a
perpendicular wall moves it not at all. At the one-pixel column where the
G-buffer flips from one wall to the other, that reconstruction can land a few
microns **outside the room**. From outside, the wall is behind the receiver, its
projection misses the light view entirely, and the sun shines straight through.

The rasterizer is not wrong here; given the position it was handed it is right.
The position is wrong, and no amount of care in the projection fixes it — which
is why the first attempt (relaxing `lv_fill`'s `t <= 0` rejection when the plane
passes within the receiver's own placement tolerance) changed nothing. When the
receiver is outside, the wall does not project into the cone *at all*.

So the robust statement has to be about the plane rather than the projection: a
receiver within its placement tolerance of a triangle's plane is on that surface,
and everything on the far side of it is inside the material. `lv_raster_tri` now
marks that half-space occluded directly, at distance zero, with a bounding-box
test to keep it honest — "the plane passes near the receiver" is satisfied by any
distant triangle whose plane happens to sweep past, and only a triangle whose own
extent also contains the receiver is the surface being stood on.

Zero isolated bright pixels remain anywhere in the frame, by the same scan that
found the original three.

### 20. Branch factor 1: the recursion is a path tracer, not a tree

**This renderer is capped at three bounces** (`kMaxLevels`, solver.hpp), so what
follows is partly an argument about a constraint that is no longer binding. It is
kept because it is why the estimator below is the default at three bounces too,
and because raising the cap is now a one-line change rather than a rewrite. The
four-, five- and eight-bounce rows were measured with it at 8.

The doc's variant A branches. Every camera spawns K children, so a depth-D solve
costs K^D cameras, and every lever section 6.2 offers -- resolution falloff,
coarser tiles -- shrinks **K**. Shrinking the base of an exponent buys one or two
levels and then the exponent takes them back:

| bounces | branching, cameras per primary hit | cameras/sweep | measured sweep |
|---|---|---|---|
| 3 | 81 | 1.33e6 | 45 s |
| 4 | 337 | 5.52e6 | **182 s** |
| 5 | 1361 | 2.23e7 | **707 s** |
| 8 | 87 377 | 1.4e9 | ~12 hours, extrapolated |

Almost exactly ×4 per bounce, which is the level-2 tile count and nothing else.

**Split once, then continue single-sample.** At the primary hit the camera splits
into `paths` directions; from there each path picks exactly ONE continuation per
bounce. Branch factor 1, so the cost is `1 + paths × (D-1)` -- linear:

| bounces | path, cameras per primary hit | cameras/sweep | measured sweep |
|---|---|---|---|
| 3 | 129 | 2.11e6 | 15 s |
| 4 | 193 | 3.16e6 | 11 s |
| 5 | 257 | 4.21e6 | 13 s |
| 8 | 449 | 7.36e6 | **20 s** |

(One batch, and the clock on this machine is worth about a factor of two — the
3-bounce row reading higher than the 4-bounce one is the noise, not a result.
The camera counts are exact.)

**The crossover is at four bounces**, and that is worth stating plainly rather
than selling the change as a free win. At D=3 the tree is genuinely cheaper per
leaf -- 16 level-1 cameras serve 64 leaves, where 64 paths do not share anything
-- so 81 cameras against 129. At D=4 it is 337 against 193, at D=5 1361 against
257, and at D=8 it is **195×**. The tree is better exactly until it isn't.

Which means that **at the three-bounce cap the exponential argument decides
nothing**, and the comparison has to be made at equal cost instead. It can be
made exactly: the tree carries 1 + 16 + 64 = 81 cameras per primary hit, a path
split of S carries 1 + 2S, and 1 + 2×40 = 81. Same cameras, same texels, same
triangle-rasters, to the unit — so `MBG_PATHS=40` is the branching estimator's
own budget spent the other way.

| 3 bounces, sky 0.05, 1.33e6 cameras/sweep either way | RMSE | ceiling roughness | sweep |
|---|---|---|---|
| **path, 40 (default)** | **0.0402** | 0.95 | **51.4 s** |
| branching tiles (`MBG_PATHS=0`) | 0.0407 | 0.91 | 52.5 s |
| path, 64 | 0.0402 | 0.97 | 81.7 s |

Three runs each, same batch, spread under 1%. At its own budget the path
estimator is a hair faster, a hair more accurate, and a hair noisier on the
ceiling — which is to say the three of them are a wash, and the tiebreak is that
one estimator is unbiased and the other is not. The tile clustering is an
approximation that more cameras do not remove; see the paragraph below, and the
sun's visibility in finding 18, which the tile estimator could only approximate
and this one gets exactly right by construction.

(`MBG_PATHS=64` is the same quality for 1.6× the cameras, which is the useful
negative: past the point where the hemisphere is adequately sampled, more paths
buy nothing at a fixed bounce count. They buy depth, and depth is capped.)

Three things make one sample per bounce good enough to keep:

**Stratification and decorrelation.** The `paths` samples are drawn one per equal
stratum of the CDF, offset by a hash of the receiver's position. Stratified so
they cannot clump; hashed per receiver so the residual decorrelates between
neighbours instead of banding, which is finding 14's lesson applied to a
different sampling grid.

**Importance sampling** by `cos × dΩ × albedo × (unshadowed direct irradiance at
the hit)`. The last factor is the only guess, and it is free: the direct
irradiance at every texel's hit point is already evaluated for the mass. Which
makes it worth reporting that **on this scene it buys nothing measurable**:

| | RMSE | ceiling roughness |
|---|---|---|
| radiance-weighted (`MBG_IMPORTANCE=1`) | 0.0402 | 0.970 |
| `cos × dΩ × albedo` only | 0.0403 | 0.949 |

Within noise, and if anything the ablation is a hair smoother. That is not an
argument against importance sampling, it is an argument about Cornell: the box is
matte, its walls have similar albedo, and its one bright emitter is handled
analytically and so never appears in the continuation distribution at all. The
radiance factor varies by a factor of a few across a hemisphere here, not by
orders of magnitude, and a few is not where importance sampling pays. It stays
on -- it costs nothing and it is the factor that will matter the moment a scene
has real contrast in what a hemisphere sees -- but the measurement is here so
nobody reads a win into it that this scene cannot show.

**Russian roulette** on the accumulated throughput. Below the threshold a path
survives with probability `throughput/threshold` and is divided by that
probability, so the expectation is unchanged and the expected path length becomes
finite and scene-dependent instead of pinned to the bounce count. At 0.15 it does
not fire within three bounces off Cornell's walls (0.73³ = 0.39), which is
deliberate: the default configuration is still exactly the solve it was, and
roulette only decides things at depths that used to be unreachable.

**And the clustering error goes away entirely**, which is the part that matters
beyond cost. The tile estimator carries a whole tile's mass and scales it by
visibility measured at ONE representative texel -- the approximation that forced
a hard-edged light to get a mass of its own (finding 18) and whose variance
survived even then. Here the child sits exactly at the hit point whose mass it
carries, so `M_emit × vis` and `M_sun × vis` are not approximations at all: the
tile and its representative are the same texel by construction. At three bounces
the two estimators land in the same place -- RMSE 0.0402 against 0.0407, ceiling
roughness 0.970 against 0.909 -- so the unbiased one costs nothing in quality at
the depth where the biased one is still affordable.

#### The roulette bug, and why a gate found it and an image never would

First version: throughput was `parent × luminance(M_ind) / π`. That reads as the
running product of albedos and it is wrong by a factor of `paths`, because M_ind
already carries the 1/N of the split -- and **the split is stratification, not
attenuation**. A path that is one of 64 does not carry a 64th of the light; it
carries an unbiased estimate of all of it, and the 64th is the averaging that
happens when its siblings are summed.

So every path's throughput read `albedo/64 ≈ 0.008`, roulette fired on all of
them at the first bounce, 95% died and the survivors were scaled by 19×. The
answer stayed *correct* -- the 1/q division is unbiased whatever q is -- and the
only symptom was variance, which looks like noise, which looks like something to
tune the denoiser for. What caught it was the `paths` gate asserting that
roulette at a threshold it cannot reach must reproduce the unrouletted answer
**bit for bit**. Fixing it dropped the ceiling's high-pass noise from 1.40 to
0.97 at no cost.

#### What the depth buys -- which is what set the cap

Finding 5 recorded that the full-GI reference only matches with `MBG_SKY=0.05`
and could not say what the 0.05 was standing in for -- the obvious candidate
being the bounces variant A truncates. That question needed eight bounces, and
eight bounces cost twelve hours. Once it cost one sweep, it could be asked:

| bounces | RMSE, sky 0 | RMSE, sky 0.05 |
|---|---|---|
| 3 | 0.0668 | **0.0402** |
| 5 | 0.0626 | 0.0414 |
| 8 | 0.0622 | — |

The extra five bounces are worth 7% of the error and have clearly converged. The
constant sky is worth 40%. **So it is not truncation.** A constant added in every
direction, that no amount of transport reproduces, is a world background -- and
the Cornell box is open on +z, so the reference render's own environment shines
straight in. `MBG_SKY=0.05` was never a fudge for missing light; it was the
missing light, and the sky dome of finding 15 is the same thing spelled properly.

And that measurement is what makes the three-bounce cap a decision rather than a
concession. Bounces four through eight move the error by 7% of a number that is
already dominated by the tone curve, on the scene most flattering to deep
interreflection there is -- a closed white box. There is nothing there worth
paying for, and the cap says so.

### 21. The soft crease is the grid — and buying it with a finer grid is a bad trade

Reported as "the denoiser blends stuff across an edge", most visibly along the
top of the room, where the path-traced reference has a clean line. Half of this
finding is a diagnosis that held up and half is a change that was shipped, looked
worse, and was reverted. Both halves are the point.

**The diagnosis.** It is not the denoise. Profile down the ceiling/back-wall
crease, rows of flat approach before the drop:

| | flat rows before the crease |
|---|---|
| default (scale 4) | 6 |
| **denoise off entirely** | **8** |
| denoise, 1 iteration | 3 |
| grid 2× finer, denoise on | **2** |
| path-traced reference | 2 |

Switching the denoise off makes it *worse*. What halves it is halving the grid
block, which says the cause is resolution. The contact darkening at a corner is
two or three pixels wide; at scale 4 the grid's nearest sample sits at a block
centre two pixels away — and its value there is right, 69 against the reference's
69 — while the reference then falls 69 → 67 → 61 → 59 over the next three pixels,
where we have no sample at all. The bilateral upsample correctly rejects the tap
across the crease, so the stencil collapses to that one grid row and holds it
flat: a shelf one block wide, then a cliff at the geometric edge.

**A gradient fit does not help, and the reason is the useful part.** Fit a plane
to the taps that survive the geometric test and evaluate it at the pixel, so the
interpolation continues the trend instead of clamping against the cut. Measured:
RMSE identical to four digits, profile unchanged. The grid's own values
approaching the corner are 70, 69, 69, 69 — there is no gradient at grid scale to
extrapolate, because the falloff happens entirely *between* the last two samples.
The information is absent, not attenuated. Reverted.

**So the grid was made finer, and that was the mistake.** Paths and grid pixels
are both sampling rates and they do not saturate together — 40, 64 and 256 paths
at scale 4 all score RMSE 0.0402, while grid resolution was still moving
everything. Scale 3 with 20 paths measured better than the scale-4 default on
every number that was being watched:

| | cameras/sweep | RMSE | crease flat rows | ceiling patch |
|---|---|---|---|---|
| scale 4, 40 paths | 1.33e6 | 0.0402 | 6 | 1.40 |
| scale 3, 20 paths | 1.20e6 | 0.0375 | 2 | 1.33 |

Fewer cameras, better RMSE, a crease profile matching the reference. It was
shipped, and it looked worse — reported immediately, and correctly.

**What no aggregate metric here reports is the silhouette.** At a diagonal edge
the reconstruction is per grid block, and scale 4's softness was the only thing
hiding that. Make the grid finer and the indirect resolves sharply enough for the
block structure to become visible as stair-steps down the ceiling/side-wall
diagonal, plus a bright fringe along the vertical corner. It is not specific to
the non-power-of-two grid: scale 3 shows it at 20 *and* 40 paths, and so does
scale 2. Nor is it the path count: scale 4 at 20 paths has a clean diagonal.

So the trade is a soft crease against a stepped silhouette, and the stepped
silhouette is worse. **Default is back at scale 4, 40 paths** — the crease stays
soft, and that is now a known and measured cost rather than an unexamined one.

Two things survive the revert:

- **Paths saturate at about half the default.** 20 paths at scale 4 scores
  0.0403 against 40's 0.0402, with whole-image noise 3.86 against 3.88, for
  **half the cameras**. `MBG_PATHS=20` is a free 2× on the solve. It is not the
  default only because 40 is the split that costs exactly what the branching
  estimator costs, which is the comparison finding 20 rests on.
- **The real fix is not more grid.** The crease needs the upsample to behave at a
  silhouette and the denoise to average *along* an edge it cannot average across
  — an anisotropic kernel steered by local structure. That is a real piece of
  machinery, and it is the thing to build if this matters, rather than spending
  4× the cameras to trade one artifact for another.

### 22. The light view was local-memory bound, and the loop order was why

First measurements on real hardware (RTX 3060, against the llvmpipe numbers
everything above was taken on), and the headline is that the per-pixel direct
pass cost **2570 ms of a 3657 ms frame** — on a GPU, for 262k pixels of Cornell.
A 4-thread software rasterizer was within 3x of it. Something was wrong, and it
was not the thing it looked like.

**Two guesses that measured as no-ops**, both kept because they are strictly less
work, neither the cause:

- `direct_pixel.comp` included `raster.glsl` and so declared 10.5 KB of shared
  memory (`s_vis`, `s_cdf`, the reductions) it never touched — on paper the
  difference between ~12% and ~29% occupancy. Removing it changed nothing: the
  driver already dead-strips unused shared storage.
- the emitter pass ran `lv_raster_tri` from all 64 invocations with no `tid`
  guard, rasterizing one triangle 64 times into the same 64 shared addresses.
  Guarding it changed nothing either: NVIDIA warp-aggregates same-address shared
  atomics, so it cost about two warps, not 64x.

**What found it was a scaling measurement, not a hypothesis.** `MBG_DIRECT_RES`
scales texels per light view while leaving per-triangle setup fixed:

| light view | texels | Direct/px |
|---|---|---|
| 4x4 | 16 | 806 ms |
| 8x8 | 64 | 1967 ms |
| 16x16 | 256 | 6495 ms |

Almost linear in texels. So the cost was `lv_fill`'s inner loop — and `lv_fill`
walks a triangle's projected bounding box **serially in one lane**, while the
parallelism is spread across triangles. 32 triangles over 64 threads: half the
workgroup idles at the barrier, and the critical path is whatever the largest
triangle costs one thread.

Worse, each of those threads carried `vec3[MBG_CLIP_MAX]` arrays through five
Sutherland-Hodgman passes (near plane plus four frustum sides), dynamically
indexed — which a GPU puts in **local memory**, i.e. off-chip. Roughly 720 bytes
per thread, 46 KB of local traffic per workgroup, every access a global
transaction.

**The fix inverts the loops.** Each thread takes ONE texel and asks which
triangle is nearest along it. Perfectly balanced; the winner lives in a register,
so the shared visibility buffer, its atomics and the barriers around them all
disappear; and there is nothing left to clip, because the only directions ever
tested are texel centres, which are inside the frustum by construction.

It is the same depth sort, not a ray cast. `lv_texel_key` decides coverage with
the rasterizer's own edge test evaluated before the projection instead of after:
the planes through the receiver and the triangle's edges have normals
`cross(v_k, v_k+1)`, and a direction is inside the cone exactly when all three
dot products share the sign of the triple product `[v0,v1,v2]`. Under a
projective map that expression *is* the 2D edge function up to a positive scale.

| | Direct/px | whole frame |
|---|---|---|
| rasterized, bbox per thread | 2570 ms | 3657 ms |
| **inverted, texel per thread** | **20.7 ms** | **871 ms** |

**124x**, and the image is the same one: RMSE against the direct reference 0.0429
either way, roughness 0.91x either way, maximum pixel difference 4.3/255 with
nothing over 8, and the penumbra still ramps 14-13-12-11-10-9-8-6-5-4-3-2-1-0
instead of stepping. The residual differences are the old edge tolerance, which
the inverted test does not need.

Two things follow. The shape left behind is what culling wants — every triangle
is now an independent side-effect-free test against a direction, so a frustum or
cluster cull just shortens the inner loop. And `raster.comp` still rasterizes its
light views the old way; Raster L2 and L3 are now the entire remaining frame
cost, and the same inversion applies to them.

**A caution about the numbers above this finding.** They were all taken on
llvmpipe, where local memory is just stack and this whole effect is invisible. A
software rasterizer and a GPU do not rank the same implementation the same way.

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
- **What the path estimator costs on a scene with real occlusion.** Finding 20's
  crossover at four bounces is a Cornell number. Both sides of it move with the
  scene: the branching tree's advantage at shallow depth comes from sharing
  level-1 cameras between leaves, which is worth more when a camera is expensive
  (many triangles), and the path estimator's variance is set by how much the
  radiance varies across a hemisphere, which is worth more when there is
  something to be wrong about. The importance-sampling ablation in that finding
  is the same caveat from the other side: it measured as a wash here, and this
  is not the scene that would show it earning its keep.

# 40_surfel_fmm — implementation status

Brute-force reference for `surfel-gi-spec-r3.md`. Stage numbering (M0, M0b, M1,
M2) follows the plan; the spec's own numbering is §10 steps 1 and 2.

The spec is emphatic that this method must not be built in dependency order:

> 1. Brute-force all-pairs, no occlusion, no hierarchy. O(N²), slow, but
>    unambiguously correct for form factors. Do not proceed until this matches.
> 2. Add the microbuffer with a brute-force candidate list (still all-pairs).
>    Now you have correct occlusion. This is the algorithmic core; everything
>    after is acceleration.

Steps 3–10 (sparse grid, per-cell dispatch, P2M/M2M/M2L/L2L, trilinear `Lsh`,
temporal amortization, LOD) are pure acceleration and are meant to be asserted
against steps 1–2 — step 4 is specified as *bit-identical* against step 3. This
example exists to be the thing they are asserted against, so it deliberately
contains no grid, no interaction list and no temporal filter.

## Build & run

```bash
cmake -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target 40_surfel_fmm
cd build-release/examples/40_surfel_fmm && ./40_surfel_fmm
```

The binary must run with its own build directory as the working directory: the
`.glb`, `shaders/` and both reference PNGs are resolved relative to cwd. Shaders
hot reload from the **copied** `shaders/` next to the binary, so editing a source
shader needs `cmake --build build-release --target 40_surfel_fmm_shaders` (or any
build, since that target is `ALL`) before the reload sees it.

| Var | Effect |
|---|---|
| `SGI_GATE=all` or a comma list | run the analytic gates and exit. Names: `bake`, `tonemap`, `p2p`, `disc`, `hemi`, `scale`, `noocc`, `opaque`, `partition`, `graze`, `occ`. `occ` honours `SGI_SURFELS` and `SGI_BUCKETS` so its bias can be swept against both |
| `SGI_MODEL=path.glb` | model to load (default `CornellBoxOriginal.glb`). The two reference PNGs only match that one, so anything else is for testing the estimator, not for a numeric comparison |
| `SGI_SURFELS=n` | surfel budget (default 30000; hard cap 65535, see the packed key) |
| `SGI_BUDGET=n` | receivers solved per frame (default 2048) |
| `SGI_BOUNCES=n` | sweeps before the solve stops; **sweep k == k bounces** under per-pixel NEE, k+1 under per-surfel (finding 24). 0 = never stop |
| `SGI_BUCKETS=8\|16` | microbuffer edge: 64 or 256 buckets |
| `SGI_METHOD=0\|1` | 0 = M1 all-pairs radiance (**no occlusion**), 1 = M2 microbuffer. Keep M1: it is the gates' exact reference, and finding 32 used it to prove an artifact was not an occlusion artifact |
| `SGI_SKY=f` | radiance of an uncovered bucket. **0.05 reproduces the full-GI reference**; see finding 6 |
| `SGI_EMISSIVE=f` | emissive scale |
| `SGI_SOLVE=n` | run n complete sweeps before the first present, then **hold** |
| `SGI_PAUSE=1` | start with the solver held |

**Direct term (NEE).** The estimator is in `shaders/common/nee.glsl`, shared by
the per-surfel and per-pixel passes so they cannot drift.

| Var | Effect |
|---|---|
| `SGI_NEE=0\|1` | split the direct term out of the microbuffer (default 1) |
| `SGI_NEE_PIXEL=0\|1` | evaluate it per pixel instead of per surfel (default 0). Same estimator, G-buffer receiver; findings 24 and 28 |
| `SGI_NEE_OCC=f` | occluder radius scale, **visibility only** (default 2.0). Coverage 1.0 does not seal, and how much inflation does depends on the sampler; findings 25 and 29 |
| `SGI_NEE_CUTS=0\|1` | clip occluder discs at the mesh's feature edges (default 1); finding 27 |
| `SGI_NEE_THICK=f` | surfel slab half-thickness, in radii (default 0.25). Required, not cosmetic: a flat disc is measure-zero and blocks nothing (finding 28) |
| `SGI_NEE_BIAS=f` | receiver offset along its own normal, in radii (default 0.05) |
| `SGI_NEE_SELF=f` / `SGI_NEE_SELF_TOL=f` | same-surface rejection: normal agreement (default 0.9) and tangent-plane distance in radii (default 1.0); finding 23 |
| `SGI_SHOW_LIGHT=1` | write the visibility scalar instead of irradiance — the instrument every direct-term finding was measured with |

**Reconstruction and denoise.**

| Var | Effect |
|---|---|
| `SGI_GATHER_R/P/N/K=f` | gather radius (spacings), plane tolerance, min `dot(n_px, n_surfel)`, kernel (0 cone, 1 Wendland C2, 2 Gaussian) |
| `SGI_MLS=0\|1` | 0 = Shepard (degree 0), 1 = degree-1 moving least squares (default); finding 22 |
| `SGI_LIGHT_SIGMA=f` / `SGI_LIGHT_GRAD=f` | the `light_vis` edge stop (default **0 = off**, finding 36) and its gradient prediction; findings 19, 20 and 36 |
| `SGI_FILTER=n` / `SGI_FILTER_R=f` | object-space cache denoise iterations and radius |
| `SGI_CELL=f` | grid cell size, in surfel spacings |

**Diagnostics.**

| Var | Effect |
|---|---|
| `SGI_BIAS=f` | plane bias, in receiver radii (default 1) |
| `SGI_SOFT=f` | disc softening eps, in r² (default 1) |
| `SGI_HORIZON=f` | receiver-side `cos` floor (default 0.02) |
| `SGI_TWOSIDED=1` | force every surfel to emit from both faces — reproduces finding 10 |
| `SGI_JITTER=0\|1\|2` | tangent-frame rotation: none / static per surfel / per surfel per frame |
| `SGI_POINTS=1` | start with the raw point-cloud overlay — the cache with no reconstruction, which is what located the band in finding 31 |
| `SGI_STATS=1` | per-sweep read-back of mean/peak/flux/delta (stalls; off by default) |
| `SGI_VIEW=n` | initial display mode |
| `SGI_TONEMAP=n` / `SGI_EXPOSURE=f` | 0 ACES, 1 Reinhard, 2 clamp, 3 filmic; initial exposure |
| `SGI_GTCAM=1\|2` | reference camera at 512×512 (pixel-aligned against the PNGs) or at 1600×900 |
| `SGI_BENCH=n` | run n frames, print wall-clock percentiles + per-pass GPU averages, exit |
| `SGI_SHOT=path` | write a screenshot before exiting |
| `SGI_NOGUI=1` | skip the ImGui overlay — **required** for a clean screenshot |

> **Build the `40_surfel_fmm_shaders` target, not `40_surfel_fmm`.** The CMake
> dependency runs that way round (`add_dependencies(40_surfel_fmm_shaders
> 40_surfel_fmm)`), so building the binary alone links new C++ against the
> **previously copied** shaders. That silently validates the old kernel: a first
> run of finding 14's gates reported every number bit-identical to the
> single-layer baseline, which read as "the change does nothing" and was in fact
> "the change never reached the GPU".

## Stage status

| Stage | What | Status |
|---|---|---|
| M0 | Scaffolding: G-buffer, geometry pass, display/compare views, surfel splat, point cloud, hot reload with GLSL `#include`, per-pass timers, env hooks | ✅ |
| M0b | Surfel bake: stratified area-weighted SoA set, exact area invariant, one radius | ✅ |
| M1 | `bf_radiance.comp` — all-pairs radiance, no occlusion | ✅ |
| M2 | `bf_micro.comp` — all-pairs candidates through a per-receiver hemi-octahedral microbuffer | ✅ |
| — | §9 step-0 analytic gates, a CPU-ray-cast occlusion gate and an opacity gate (9 gates, 25 assertions) | ✅ all pass |
| — | Matched-camera compare against both path-traced references | ✅ |
| M3+ | Sparse grid, U/V lists, FMM, temporal amortization, LOD | ⬜ out of scope by design |

## Layout

| File | Role |
|---|---|
| `main.cpp` | window, scene load, frame graph, ImGui, env hooks |
| `surfels.hpp/.cpp` | M0b: triangle extraction, stratified bake, packing, the six SoA SSBOs |
| `brute.hpp/.cpp` | M1 + M2 driver: slice scheduler, sweep ping-pong, the exact bucket table |
| `screen.hpp/.cpp` | G-buffer, geometry pass, surfel splat, point cloud, references, display |
| `validate.hpp/.cpp` | the §9 step-0 gates |
| `gpu_util.hpp/.cpp` | `PassTimer`, `Pipeline`, camera control — copied from 39 minus `PrefixSum` |

Shaders: `bf_lout.comp` (the `L_out` precompute), `bf_radiance.comp` (M1),
`bf_micro.comp` (M2), plus the G-buffer / splat / points / display raster pairs.
`shaders/common/` carries `oct.glsl`, `gbuffer.glsl`, `pack.glsl`, `surfel.glsl`,
`brdf.glsl`, `micro.glsl`, `tonemap.glsl`.

## Units

Stated once, because §1.1 says getting this wrong is the most common failure and
that the error is *consistent* across near and far, so nothing internal catches
it:

```
emission   emitted RADIANCE  L_e  [W/m²/sr]
irradiance IRRADIANCE        E    [W/m²]
outgoing   L_out = L_e + albedo * E / PI
```

The `/PI` belongs to the **emitting** surfel. It exists in exactly one place in
this example — `shaders/common/brdf.glsl::sgi_outgoing` — and everything that
forms `L_out` routes through it: `bf_lout.comp` precomputes the whole array once
per sweep into binding 6, and M1, M2, the splat and the point cloud all read
that. A `/PI` in one path and not another is the bug §1.1 warns about, and
centralizing it is the only structural defence.

## Design decisions that deviate from the spec, and why

**Jacobi at sweep granularity, not Gauss-Seidel.** §7 forbids ping-ponging, but
that prohibition is specifically about updating a *subset* of receivers per
frame, where a skipped receiver would read two-sweep-old data after a swap. Here
the slice scheduler covers every receiver exactly once per sweep and the swap
happens only at sweep boundaries, so the hazard cannot arise. Two things are
bought with it: a GPU Gauss-Seidel is not merely order-dependent but
*nondeterministic* — the order is workgroup scheduling order — which would make
step 4's bit-identical assertion impossible; and Jacobi's iterate k is exactly k
bounces, so **sweep 1 is precisely what the direct-only path trace shows**, which
is what makes the direct reference a usable gate at all.

**fp32 irradiance, not RGB9E5.** The whole point of this stage is that a
discrepancy against the reference is a bug; RGB9E5's 9-bit mantissa would put a
floor under that. At 30k surfels the two irradiance buffers are 1 MB total.
Switch when the grid lands, and assert the delta is under 1% at that point.

**Uniform surfel radius.** `r = sqrt(A / (pi*N))` makes `N*pi*r² == A` an exact
invariant (gate 6 asserts it to 1e-4, measures 1e-7) and removes an entire class
of "is this a radius bug?" from the reference. It also makes
`spacing = r*sqrt(PI)` exactly, which the splat reach depends on — see below.

**Emission is front-face only; occlusion is two-sided.** A back-facing surfel
contributes no radiance and still blocks. The glTF `doubleSided` flag is baked
into the surfel record but deliberately not consulted for emission — see
finding 10 for why that distinction is load-bearing and how it was missed.

**Christensen's disc softening, not a hard `min(ω, 2π)`.** The plan called for
the spec's clamp. A hard clamp introduces a derivative discontinuity at the
distance where it engages, which traces a visible ring; `d² → d² + ε·r²`
(Christensen 2008) is smooth and agrees to within a percent everywhere the clamp
was not active. `SolveConfig::soft_eps`.

## Findings

### 1. Hemi-octahedral texels are NOT equal solid angle

The plan assumed `dw = 2π/MS²` and no per-bucket table. That is wrong. Measured
texel solid angle varies **3.2×** across an 8×8 hemi-oct map and **4.2×** across
16×16, and the uniform assumption gives `Σ cos·dw = 0.983π` (8×8) / `0.969π`
(16×16) instead of `π` — a 1.7%/3.1% energy error with a *spatial pattern*, which
is worse than a constant factor because it does not divide out.

`build_bucket_table()` therefore computes both quantities exactly on the host:
`dw` per texel by Girard's theorem (Van Oosterom & Strackee for the spherical
triangle), and `wcos = ∫cos θ dΩ` over the texel by 16×16 subsampling with the
`1/|p|³` Jacobian. Both are stored, because `dw·dir.z` — the obvious shortcut —
is off by +1.04% (8×8) / +0.27% (16×16). Gate 3 asserts `Σdw = 2π` and
`Σwcos = π` to 1e-5; both land at ~5e-9.

Note for whoever ports §3.4's `neighbourBucket()` seam handling: that warning is
about the *sphere* octahedral map's fold. This is the hemi-oct map, whose square
boundary is the horizon, not a fold — weight that falls off the square is
genuinely below the horizon and must be **dropped**, not wrapped. `bf_micro.comp`
normalizes the splat kernel over the *unclipped* box for exactly this reason;
normalizing over the clipped box piles sub-horizon energy onto the rim.

### 2. The spec's single-winner coverage is catastrophic in the brute-force regime

§4.1 takes coverage from the single depth winner, `cov = saturate(π·angR²/dw)`,
and blends the remainder against the far field. That is self-consistent inside
the U-list, where K ≈ 91 candidates over 64 buckets means ~1.4 candidates per
bucket and each subtends roughly a whole bucket. It is not self-consistent here:
at N = 30k a wall surfel one metre away subtends ~9e-4 sr against a 64-bucket
texel's 0.098 sr, so ~110 surfels share a texel and the single winner reports it
1% covered. With no far field to blend against, a closed room comes out ~100×
too dark.

Measured against the direct reference at N = 30k, 16×16, one sweep (interior
mean, sRGB):

| Coverage model | mean | MAE |
|---|---|---|
| reference | (64.7, 48.0, 17.6) | — |
| `SGI_COV=0` spec §4.1 literally | **(2.9, 2.6, 2.3)** | 40.97/255 |
| `SGI_COV=1` example 39 (summed cov, winner radiance) | (71.7, 52.8, 17.5) | 15.84/255 |
| `SGI_COV=2` accumulated cov **and** cov-weighted radiance | (54.1, 38.1, 12.2) | **11.18/255** |

22× too dark, which is the predicted order. Mode 2 is the default and is what the
acceleration stages should be compared against. Mode 0 is kept because step 3
needs the spec's own formulation to exist to be compared against once the U-list
bound makes it valid; mode 1 is kept because it is what example 39 ships.

Mode 1 lands closer in *mean* than mode 2 but has a worse MAE: winner-takes-all
radiance makes a bucket entirely the colour of its nearest surfel, which
overshoots and undershoots in different places rather than being uniformly off.
The spec flags the underlying issue itself at §5 ("a bucket fully covered by
three partial discs still reads as partially covered and leaks far field") and
proposes the two-layer microbuffer as the fix.

### 3. A bucket asks two questions, and they need two different tests

This was the longest thread in the example and it went through three wrong
answers before the right one, so it is worth the space. The visible symptom that
started it: **the floor was lit right up to the boxes' base with no contact
shadow at all**, and a stair-stepped bright band along the contact line.

Phase B has to decide, for each candidate landing in a bucket, whether it
contributes. The three attempts:

**(a) Depth window — "within a couple of surfel radii of the winner's depth".**
Wrong. A bucket is a cone of finite angular width (0.157 rad at 16×16), so the
surface it sees spans a real range of *depths*: for a plane at incidence angle φ
the depth varies by `d·α·tanφ` across one bucket, ~0.09 world units at one metre
and 60°, against a tolerance of two radii (0.016). It therefore rejects most of
the winner's own coplanar neighbours on anything but a face-on surface. The
coaxial-disc gate caught this immediately *and* diagnosed it, because the shape
of the error is informative — an error that **worsens with separation** cannot be
a constant factor, it has to be angular:

| gate | expected | (a) depth | (b) plane | (c) ray vs plane |
|---|---|---|---|---|
| disc M2 h/R = 0.5 | 1.9152 | 0.9130 (−52%) | 1.8470 (−3.6%) | 1.8470 (−3.6%) |
| disc M2 h/R = 1.0 | 1.2000 | 0.4538 (−62%) | 1.1761 (−2.0%) | 1.1761 (−2.0%) |
| disc M2 h/R = 2.0 | 0.5390 | 0.1880 (−65%) | 0.5329 (−1.1%) | 0.5329 (−1.1%) |

Cornell never showed this: it is a room of face-on flat surfaces at short range,
where a depth window is roughly adequate. The gate is a pair of *coaxial discs*
precisely because it puts one surface at every angle from head-on to grazing in a
single measurement.

**(b) Plane test — "within a couple of radii of the winner's supporting plane".**
Fixes the disc gate, and introduces the contact-shadow leak. A plane test accepts
anything near the winner's **infinite** plane, and Cornell's boxes have vertical
faces whose planes (`z = const`) pass straight through the ceiling panel, which
spans the same `z`. So panel surfels came out coplanar with a box face, were
taken for the same surface, had their radiance averaged into buckets the box
actually occludes — and lit the floor at the box's base. The lesson is small and
general: **coplanar is not co-surface**. Points genuinely further along one
surface also share its *normal*, and the panel does not, so a normal-agreement
term separates the two cases for one `dot`.

**(c) Ray against the winner's plane, plus normal agreement.** The question
occlusion actually asks is not "does this belong to the winner's surface" but
"is it *behind* the front surface **along its own ray**". Intersecting the
candidate's ray with the winner's plane and comparing depths there is
slope-correct by construction — which is what (b) was reaching for — while still
admitting a *nearer* surface that the winner's plane says nothing about. The
normal term is kept only for candidates sitting on the plane, to reject (b)'s
coplanar-but-different-surface case.

**And the separate half: opacity is not the same quantity as radiance.** Even
with (c), the microbuffer reported a closed room as 3× more open than it is (see
finding 4). Every form of "does this candidate belong to the front surface"
throws away the *other* surfaces in the bucket — and a cone in a closed room
usually straddles more than one. In a corner it sees two walls and only the
winner's was counted; the rest read as open sky. §5 names this failure exactly:
"a bucket fully covered by three partial discs still reads as partially covered
and leaks far field".

The fix is to stop making one number do both jobs. `bf_micro.comp` now keeps two
accumulators: `lds_cov_any`, which every candidate adds to and which becomes the
bucket's **opacity**, and `lds_cov_front`, restricted by test (c), which
normalizes the bucket's **radiance**. The uncovered-by-the-front-surface part of
a bucket is then attributed the front surface's radiance rather than the sky —
exact when the two surfaces are lit alike (a corner, a silhouette against its own
wall), and the reason §5 wants a second layer for when they are not.

### 4. Occlusion needed its own gate, and it had to be a ray cast

Gates 1–5 are all analytic and all *unoccluded*: p2p and disc have two surfaces
with nothing between them, hemi has no geometry at all, and noocc forces every
candidate visible on purpose. **Nothing measured whether the microbuffer
occludes** — which is the entire content of §10 step 2. Everything in finding 3
was found by eye first and only then explained, which is the wrong way round.

Gate 8 closes that. Light the scene with a uniform unit sky and no emitters; then
a surfel's irradiance is exactly `E = π · (cosine-weighted fraction of the
hemisphere that is open)`, and the same number can be ray cast against the 32
source triangles on the CPU to any precision. No reference image, no tone curve,
no camera. Results are reported *per openness band*, because a uniform bias and
one concentrated in the occluded regions are different bugs and a mean hides the
difference.

Cosine-weighted openness, N = 30k, 16×16, 4286 surfels × 2048 rays:

| band (CPU openness) | CPU | (b) plane | (c) ray | (c) + split opacity |
|---|---|---|---|---|
| deeply occluded (<0.15) | 0.0621 | 0.2221 | 0.1760 | **0.1148** |
| occluded (0.15–0.35) | 0.2237 | 0.3662 | 0.3267 | **0.2509** |
| half open (0.35–0.60) | 0.4312 | 0.5389 | 0.5073 | **0.4417** |
| mostly open (0.60–1.00) | 0.7551 | 0.7782 | 0.7746 | **0.7744** |
| mean abs error | — | 0.1476 | 0.1055 | **0.0420** |
| population bias | — | +93.7% | +67.0% | **+25.6%** |

The **sweep against N and bucket count is what identified the cause**, and it is
worth recording how, because the first hypothesis was wrong and the sweep said so
in one run. A stochastic coverage bias (a texel's coverage is a sum over the
surfel centres landing in it, so it fluctuates about 1 and `clamp(cov,1)` removes
the excess but not the deficit) predicts the error falls as surfels-per-texel
rises — so it should improve with **more surfels** *and* with **coarser
buckets**. Measured before the fix:

| | 8×8 | 16×16 |
|---|---|---|
| N = 15000 | — | 0.136 |
| N = 30000 | 0.125 | 0.114 |
| N = 60000 | — | 0.098 |

Coarser buckets were *worse*, and 4× the surfels bought only 1.2×. Both signs
wrong for a sampling artifact, both right for an angular one — which is what
pointed at silhouette and corner buckets rather than at density. After the split:

| | 8×8 | 16×16 |
|---|---|---|
| N = 15000 | — | 0.0569 |
| N = 30000 | 0.0324 | 0.0420 |
| N = 60000 | — | 0.0316 |

Now coarser buckets help, and the N scaling is 0.74× per doubling against the
0.707× the sampling model predicts. `err·sqrt(N)` = 6.97 / 7.27 / 7.74 — flat.
That is the surfel representation floor of §9 step 1, so gate 8 asserts against
`12/sqrt(N)` rather than a fixed number: it passes at any density while still
failing hard on anything structural, since the finding-3 bug sat at `25/sqrt(N)`
and did not improve with N at all.

### 5. The splat's partition of unity depends on the bake's area invariant

The splat weight falls linearly to zero at `r·reach`, and the bake makes
`spacing = r·sqrt(PI) ≈ 1.77·r` exactly. At `reach = 1` the support is therefore
*smaller than the spacing*: a pixel between four surfels collects zero weight from
all of them and the normalize divides zero by zero. That renders as a fine dark
stipple over every surface — it reads as material detail, not as a hole, which is
what makes it dangerous. `reach = 4` is 2.26 spacings and several surfels always
overlap a pixel. This is a *screen* artifact only; the surfel irradiance was
correct throughout.

### 6. The full-GI reference was rendered with a world, the direct one was not

The two PNGs disagree about the background: the direct reference's corner pixel
is (0, 0, 0), the full-GI reference's is (58, 58, 58). The converged render
therefore includes a uniform world, which is what makes the short box's
camera-facing face — which sees nothing but the open end of the box — a bright
neutral grey in the reference and near-black in a closed-room solve.

`SGI_SKY=0.05` (Blender's default world grey) reproduces it. At N = 30k, 24
sweeps, per-region sRGB:

| region | reference | sky 0 | **sky 0.05** | sky 0.10 |
|---|---|---|---|---|
| short box front (world-lit only) | (54, 48, 42) | (8, 3, 1) | **(48, 42, 38)** | (78, 72, 67) |
| tall box front | (81, 66, 43) | (54, 37, 8) | **(75, 59, 35)** | (93, 78, 56) |
| floor mid | (87, 66, 39) | (57, 35, 9) | **(75, 54, 31)** | (90, 70, 49) |
| back wall mid (panel-lit) | (144, 121, 69) | (141, 116, 45) | **(145, 122, 54)** | (149, 127, 62) |
| short box top (panel-lit) | (145, 124, 74) | (139, 116, 46) | **(143, 122, 56)** | (148, 128, 66) |
| interior mean | (121.5, 98.8, 55.7) | (110.4, 84.7, 32.0) | **(118.8, 94.6, 44.9)** | (126.3, 103.4, 56.2) |

`sky = 0.10` matches the interior *mean* slightly better in blue but gets there by
overshooting the world-lit faces by 45%. 0.05 is the principled value, it
reproduces the hardest case — the face lit by nothing but the world — to within
11%, and after finding 3 it puts the panel-lit surfaces within **1%** in R and G
(145 vs 144, 143 vs 145). What is left is blue, and blue alone, which is
finding 9.

So: **the direct reference is the primary gate** (`SGI_SKY=0`, one sweep, no free
parameters at all), and the full-GI reference is the secondary one and needs
`SGI_SKY=0.05` declared.

### 7. Screenshots must be taken BEFORE `swap_buffers`

`SGI_SHOT` originally wrote its PNG after the frame loop, i.e. after the last
`swap_buffers`. `glfwSwapBuffers` leaves the back buffer's contents **undefined**,
so `glReadPixels` there returns whatever the driver left behind. On this machine
that was intermittently an all-black or an all-white 512×512 image — roughly two
runs in five — and the failure is silent: a black screenshot of the diff view is
indistinguishable from a diff view that found no error, and a white one from a
blown exposure. Two view modes were investigated as broken before the pattern
showed up in a repeat run of a mode that had already worked.

The shot is now taken immediately before the swap on the final frame, and a run
that fails to produce one says so. Five consecutive runs of the split view now
give a bit-identical mean.

### 8. A scripted solve can trip the GPU watchdog and return flat grey

`SGI_SOLVE=n` queues n complete sweeps before the first present. At N = 30k and
16×16 a sweep is ~0.5 s, so the burst crosses the kernel's ~10 s job watchdog
somewhere past 20 sweeps: the GPU is reset, the context is lost, and every frame
after comes back as **uniform grey** rather than as any kind of error. It
reproduced exactly at the boundary — `SGI_SOLVE=20` correct, `SGI_SOLVE=24` grey,
and the mean was a suspiciously flat (1.54, 1.54, 1.53).

A `glFinish()` between sweeps bounds any single submission to one sweep and fixes
it; the path is the deliberately-blocking one, so the stall costs nothing.
`SGI_SOLVE=24`, `32` and `64` now all return bit-identical means to `20`.

Worth pairing with finding 7: both are ways for a headless scripted run to hand
back a confident, plausible-looking image that is not the render. Neither is a
graphics bug in the usual sense and neither announces itself.

### 9. The tonemap gap is the residual, and it is bounded

Both references are Blender Cycles renders, and a saturated emitter of
(17.0, 12.0, 4.0) landing near-neutral at (254, 250, 238) *without clipping* is
Blender 4.x AgX, which none of the four transforms here is:

| transform | (17, 12, 4) → | reference |
|---|---|---|
| ACES (Narkowicz) | (255, 255, 252) | (254, 250, 238) |
| Reinhard | (249, 246, 231) | " |
| Clamp | (255, 255, 255) | " |
| Filmic (Hejl) | (252, 251, 243) | " |

The `tonemap` gate prints this table so the choice is measured rather than
assumed. ACES is the default. The residual shows up exactly where AgX differs
from ACES and nowhere else: with `sky = 0.05` the panel-lit surfaces agree in R
and G to 2–3% while the blue channel runs 20% low (AgX desaturates toward neutral
much harder than ACES), and the midtones run ~15% low (AgX's midtone lift). A
linear-light comparison is not available because the references are already
tonemapped and AgX is not invertible without porting it.

Porting AgX is the obvious next accuracy step for the *comparison*, not for the
solver.

### 10. An opaque surface must not emit from its own back face

The symptom was a **one-pixel bright line exactly along every box/floor contact**
— the floor lit precisely where it should be most shadowed. Scanning down a
column through the short box's base: box face 1, then `19, 15, 11, 7, 5, 4`, then
the shadowed floor at 3. A spike to 6× the floor either side of it.

The cause: `SolveConfig::two_sided` defaulted to **true**, and both kernels
implemented it as `cosE = abs(cosE)` on every surfel. That does not make a
surface two-sided, it makes it *transmitting*: a lit surface re-emits its own
outgoing radiance out of its back. So each box's brightly lit lid emitted
downward into the sealed volume underneath it, the floor surfels under the box
were lit through the lid, and the screen splat — which gathers within four surfel
radii — pulled them onto the visible pixels at the contact.

The original reasoning for the default was that Cycles' emission BSDF is
two-sided. It is; the error was applying that globally rather than to emitters.

Two things made this hard to see:

- **The material flag is not the fix, and looks like it is.** The bake has stored
  a per-surfel `kSurfelDoubleSided` since M0b, taken from glTF `doubleSided`, and
  no shader ever read it. Wiring it up changes nothing on this scene:
  CornellBoxOriginal.glb marks **all 29997 surfels double-sided**. glTF
  `doubleSided` controls raster backface *culling*; it does not say a diffuse
  surface re-emits its front side's radiance out of its back. Emission is
  front-face only, full stop. Two-sidedness belongs in OCCLUSION, which already
  uses `abs(dot(nj, -w))` unconditionally, so a back-facing surfel emits nothing
  and still blocks — §4.1's "single most common source of light leaks".
- **Whole-image statistics are blind to it.** At the 512² reference camera the
  interior mean is *identical* before and after the fix (54.1 both) and the
  converged MAE moves 8.96 → 9.27, i.e. slightly the wrong way. A one-pixel line
  is a vanishing fraction of the pixels. Every check used up to this point —
  interior mean, ratio, MAE, and the per-band openness of gate 8 — averages over
  area and cannot see it. It was found by scanning a single column of pixels
  across the contact, and only after being told the edge still looked lit.

Gate 9 closes the hole. Three coaxial discs: an emitter, a larger opaque blocker
that fully shadows it, and a receiver behind the blocker facing its back. It
needs **three sweeps** to mean anything — on sweep 1 the blocker's outgoing
radiance is still zero, so nothing can leak through it and the test passes
vacuously. It also deliberately does *not* let `exact()` zero `two_sided`, since
a gate written to catch a wrong setting has to see the live one:

| | blocker E | receiver behind it |
|---|---|---|
| front-face emission (shipping) | 0.22014 | **0.00000** (0.00%) |
| `SGI_TWOSIDED=1` (the bug) | 0.22014 | 0.23524 (**106.9%**) |

The receiver behind an opaque blocker was receiving *more* light than the blocker
itself. That the gate fails loudly when the bug is reintroduced was verified, not
assumed — a gate that cannot fail is worth nothing, and the first version of this
one could not, because `exact()` was quietly overriding the very setting under
test.

### 12. The "painterly" look was the splat kernel, not the surfel density

The render read as soft and painterly against the crisp path trace. The natural
reading is "the surfels are too coarse", and that turned out to be wrong: at
N = 30k the cache was already fine enough, and the blur was in the *display*.

Measured on a true soft-shadow edge on the floor (chosen by masking to pixels
where the G-buffer normal is the floor on both sides, so it is a shadow boundary
and not a geometric silhouette — the silhouettes are 1 px hard in this renderer
and 3 px antialiased in the path trace, which is the opposite problem):

| | 10–90% transition |
|---|---|
| path-traced reference | 8 px |
| linear ramp over 4 radii (old) | **13 px** |
| Gaussian, sigma = 0.75 spacings | **9 px** |

The old radial weight was a linear ramp over the whole footprint, which ties
sharpness to support. The footprint has to span at least one surfel spacing or
pixels between surfels collect no weight at all (finding 5), and a linear ramp
that wide then low-passes everything over its full width. Narrowing the
footprint instead just reopens finding 5's holes — at reach 1.0 the contact rows
drop to zero.

A Gaussian in units of the **spacing** decouples them: the sprite stays wide
enough to never leave a hole, while the weight falls off over about one spacing.
Cost, at N = 30k: spurious high-frequency on a flat wall patch rises from 0.95 to
1.13 of 255, whole-image MAE is unchanged at 11.2, and no new dropouts appear.
`SGI_KERNEL` sets sigma; 0 restores the old linear ramp.

**Where the floor actually is.** With the reconstruction fixed, the edge width
tracks twice the surfel spacing, which is the cache's Nyquist limit:

| N | spacing (px) | 2 x spacing | measured edge |
|---|---|---|---|
| 10000 | 7.3 | 14.6 | 17 |
| 30000 | 4.2 | 8.5 | 9 |
| 60000 | 3.0 | 6.0 | 9 |

At N = 30k the reconstruction sits exactly on the cache limit. At 60k the cache
stops being the constraint and the edge matches the path tracer to within a pixel
(9 against 8) — at that point what remains is the reference's own penumbra and
the microbuffer's angular resolution.

So the softness is a bandwidth budget, not a property of the method. Two ways to
buy sharpness, and only one of them is cheap:

- **Raise N.** Edge width goes as `1/sqrt(N)`, and it also lowers finding 4's
  sampling bias. But this stage is O(N^2) and the packed key caps N at 65535, so
  it is the expensive lever here and would stop being one the moment the grid
  lands.
- **Stop caching direct light.** This is the architectural answer and it is what
  the spec and example 39 both do: evaluate direct light per pixel and cache only
  the bounce. Indirect light genuinely is low frequency, so a coarse cache costs
  nothing there; it is the *direct* term that carries every sharp shadow edge and
  it is the reason a cache-resolution limit is visible at all. Example 40 caches
  both on purpose — a reference that only cached the bounce could not be compared
  against a path trace of everything, and the direct-only reference at sweep 1
  would have nothing to gate against.

### 13. What is left, and what it is not

The diff view (`SGI_VIEW=11`, M2 one sweep vs the direct reference, gain 4) after
findings 3 and 4. The contact shadows that prompted the whole investigation are
gone from it. What remains, in descending order:

- **The ceiling band around the panel** is now the most visible artifact, because
  finding 12's sharper kernel stopped blurring it away. Still open (see Open).
- **A bright ring on the light panel's rim.** The emitter's own edge — one texel
  of antialiasing difference against a path-traced edge. Not worth chasing.
- **A mottled band on the ceiling around the panel, running ~25% BRIGHT**
  (measured 63.2 against the reference's 50.0 over rows 95–165). Open. Three
  hypotheses have been tested and eliminated: it is not tangent-frame
  quantization (`SGI_JITTER=0` and `1` give local sd 9.47 vs 9.25, i.e. no
  difference), it is not the plane bias (`SGI_BIAS` from 1.0 down to 0.0 leaves
  the band at exactly 63.2), and it is not a density artifact in the usual
  direction — *more* surfels makes it worse (sd 9.47 at N = 30k, 12.10 at 60k).
  The likely area is the near-field geometry of a nearly coplanar emitter:
  Cornell's panel sits at y = 1.98 under a ceiling at y = 1.99, so the gap is
  0.01 against a surfel radius of 0.0164 at N = 30k. Every ceiling receiver sees
  the panel edge-on at a distance below one radius, which is exactly where the
  disc solid angle is least trustworthy and where `soft_eps` is doing the most
  work. Not yet confirmed.
- **Warm streaks along the floor/wall and wall/wall concave seams** — §9's
  "darkening in concave corners". Smaller than before finding 3's opacity split,
  which is what the split was for, but not gone.
- The box/floor **contact lines are no longer among them** (finding 10). The
  column scan across the short box's base now reads `1, 1, 1, 2, 2, 2 … 3`,
  rising monotonically away from the contact, which is the correct direction.
- **One-pixel red outlines on the box silhouettes** — geometry antialiasing, a
  hard rasterizer edge against a path-traced one.

Everything else is deep blue. The largest *systematic* residual is no longer a
transport error: it is the tone curve (finding 9), then the surfel sampling floor
(finding 4), both of which are measured and bounded.

### 14. The second microbuffer layer, and why it cannot be `1 - cov_front`

**Default: off (`layers = 1`).** Everything below is measured and stands, but the
change moves 0.7% of the image and 89% of pixels by 2/255 or less, while the
splat's surfel mottling is ~10/255. It cannot be evaluated through that, and it
was over-claimed on first write: the penumbra numbers below are one ramp on the
near floor, not a global result. The effect is concentrated on the ceiling band.
Re-enable and re-judge once the reconstruction noise floor drops.

Spec §5's two-layer microbuffer, listed as open since finding 3. The single-layer
integration fills `cov_any` of a bucket and paints all of it with the **front**
surface's radiance, so a bucket straddling a silhouette — half the box's dark
side, half the lit ceiling behind it — comes out entirely box-coloured. That is
what quantizes a penumbra to the bucket grid.

It is not a small term. At 16x16 the mean bucket is 0.0245 sr, an angular radius
of ~5 degrees; for a floor receiver ~0.5 units from a box that is ~0.044 world
units of shadow-edge slop, about 6-7 px at the reference camera — the same order
as the cache's own 8-9 px Nyquist limit (finding 12), so the two are co-dominant.
Measured on the long soft-shadow ramp across the near floor (rows 492-504, cols
225-345, 10-90% transition, `SGI_SKY=0`, one sweep):

| | 10-90% width | contrast | profile RMS vs ref |
|---|---|---|---|
| path-traced reference | **108.0 px** | 54.8 | — |
| one layer | 77.6 px | 38.4 | 34.03 |
| two layers, gate 0.25 | 90.2 px | 43.6 | 29.58 |
| **two layers, gate 0.5** | **105.6 px** | 46.5 | **24.65** |
| two layers, gate 0.8 | 107.3 px | 47.6 | 20.03 |

The single-layer model was compressing the penumbra by **28%**. Two layers
recover it to within 2.2% of the path tracer and cut the profile RMS error by 28%.

**The trap, and gate 9 caught it.** The natural formulation is that the front
layer covers `cov_front` of the cone and the layer behind gets the rest,
`1 - cov_front`. Written that way, gate 9 failed immediately: **8.2% of an area
light came through an opaque disc that occludes it completely.** The reason is a
sentence worth keeping:

> `cov_front < 1` on a solid sheet does not mean the sheet is partly
> transparent. It means the coverage estimator undershot.

and it undershoots by a lot, systematically, and we already had the number. Gate
8 reads this closed room as 0.1977 open where a CPU ray cast says 0.1574 — 25.6%
too open. A back layer sized `1 - cov_front` draws its light from exactly that
bias, so it transmits through solid geometry. Finding 4's sampling floor and this
are the same defect seen from two sides.

So the change is scoped to be a **pure radiance attribution**: the bucket's total
opacity stays exactly `clamp(cov_any)`, the single-layer value, and only the
question of *whose* radiance fills it is answered differently. The front layer's
transmittance is a separate, calibrated quantity, `clamp(1 - cov_front/tau, 0, 1)`,
with `tau` (`SGI_BACKGATE`, `SolveConfig::back_gate`) the front coverage at which
the front surface is taken to close the bucket. Because opacity is untouched by
construction, **gate 8's openness error is bit-identical across the whole `tau`
sweep** (0.041976796 at every value tested, and at `layers = 1`) — which is the
check that the split did not leak into the opacity.

`tau` is measured, not chosen for looks:

| `tau` | gate 9 leak (tol 2%) | penumbra width | direct MAE | ceiling band vs ref |
|---|---|---|---|---|
| 1.0 (i.e. `1 - cov_front`) | **8.22% FAIL** | — | — | — |
| 0.85 | 2.36% FAIL | — | — | — |
| 0.8 | 1.62% | 107.3 px | 7.38 | -5.3% |
| 0.7 | 1.05% | — | 7.44 | — |
| **0.5 (default)** | **0.24%** | **105.6 px** | **7.59** | **-2.5%** |
| 0.35 | 0.0005% | — | 7.75 | — |
| 0.25 | 0.000% | 90.2 px | 7.88 | +0.8% |
| one layer | 0.000% | 77.6 px | 8.39 | +6.5% |

Every image metric improves monotonically with `tau` and the leak grows
monotonically with it, so this is a real trade and not a free lunch. 0.5 is the
default because it captures 98% of the penumbra-width gain (105.6 of 107.3) for
15% of the leak, and because gate 9's coaxial-disc setup is a *clean*
configuration — real geometry will leak more than it does, so spending 80% of
that gate's tolerance to buy 3% of MAE is the wrong way round.

**It also moves finding 13's ceiling band**, which four eliminated hypotheses had
left open. Converged, `SGI_SKY=0.05`, 16 sweeps, band rows 95-165:

| | interior ratio (R,G) | MAE | ceiling band | band local sd |
|---|---|---|---|---|
| one layer | 0.932, 0.911 | 8.26 | 93.2 (ref 88.6, **+5.2%**) | 3.71 |
| two layers | 0.926, 0.898 | 8.04 | 86.6 (ref 88.6, **-2.3%**) | 3.09 |

The band's absolute error drops 57% and its mottle 17%. That is consistent with
the fifth hypothesis — near-field geometry of a nearly coplanar emitter — being an
*attribution* problem rather than a geometry one: every ceiling receiver sees the
panel edge-on in buckets that also contain ceiling behind it, and the single-layer
model was handing the panel's own bright radiance to the whole bucket.

**Cost: 2-3%, not the 40-60% predicted.** N = 30k, 16x16, full sweep per frame:

| | wall clock min | p50 | GPU solve |
|---|---|---|---|
| one layer | 295.3 ms | 302.4 | 306.3 |
| two layers | 303.9 ms | 308.2 | 307.7 |

The estimate assumed the non-front path going from one atomic to five would cost
proportionally, since the majority of candidate-texels in a closed room sit behind
the winner. It does not, and the reason revises finding 3's reading: the new
atomics target *different* LDS addresses (`lds_cov_back`, `lds_rad_back`) and so
add no contention to the ones already hot. Finding 3's 16% for "one ray/plane
intersection plus a second atomicAdd" was therefore mostly the plane
intersection's ALU, not the atomic — §4.2's "the atomics will dominate" is
overstated for this kernel at this bucket count.

**What it still gets wrong.** One bucket behind, holding the average of everything
behind the front surface, including a third surface the second occludes. Depth
complexity along a cone in a closed room is 1 almost everywhere and exactly 2 at a
silhouette — the case this exists for — so the error is confined to genuinely
layered geometry and will grow on scenes that have it. Resolving it needs a second
`atomicMin` pass to give layer 2 its own winner and supporting plane (+1 KB LDS,
~+20%); the numbers above did not justify paying for it yet.

LDS at 16x16 goes 11.5 KB -> 15.5 KB. For contrast, buying the same penumbra
accuracy by raising `MS` to 32 would need 41.5 KB (one workgroup per SM), a 4x
bucket loop, and a wider `kMaxMS2` — which is the argument for layers over
resolution.

### 15. The mottling is in the cache, not in the reconstruction

"You can make out the surfel positions" was the standing complaint, and finding
12 had put it down to reconstruction bandwidth. That was wrong, and the
measurement that settles it is a **provably smooth patch**: floor pixels where
the path tracer's own 9x9 local sd is under 1.0 (6189 px, its sd there 0.810).
Any high frequency we produce on that patch is ours.

| reconstruction | local sd | x reference |
|---|---|---|
| path-traced reference | 0.810 | 1.0 |
| splat, reach 4, kernel 0.75 | 2.105 | 2.6 |
| gather R = 2.5 spacings | 1.955 | 2.4 |
| gather R = 1.0 spacings | 4.364 | 5.4 |
| **gather R = 0.6 (~ nearest surfel: the raw cached value)** | **5.998** | **7.4** |

Read the last row first. With the reconstruction reduced to "take the nearest
plane-consistent surfel", the image carries **7.4x** the reference's high
frequency on a surface that should be perfectly smooth. That is not a filter
artifact -- it is what is *in the cache*. It is finding 4's sampling floor,
err ~ 7.3/sqrt(N) per receiver, seen directly instead of through a metric.

So the splat's blur was doing two jobs, denoising and reconstructing, and tying
them together is why every attempt to sharpen one made the other worse. The
gather alone, which this stage was planned around, makes the image **worse**
(5.4x against the splat's 2.6x) for exactly that reason: a sharper reconstruction
is a more faithful view of a noisy cache. The plan's premise -- "the gather's
nearest-surfel fallback lets the radius shrink, so it can be sharper" -- is true
and useless on its own.

Splitting the two jobs is what works. `surfel_filter.comp` denoises the cache in
**object space** over the grid, and the gather then reconstructs sharply:

| | local sd x ref, gather R = 1.0 | R = 2.5 |
|---|---|---|
| no denoise | 5.4 | 2.4 |
| 1 iteration | 2.4 | 1.7 |
| **2 iterations** | **1.7** | 1.4 |
| 4 iterations | 1.3 | 1.2 |

**Gather R = 1.0 with two denoise iterations gives 1.7x against the splat's 2.6x,
at a tighter reconstruction radius** -- better on both axes at once, which neither
change could manage alone.

Object space rather than screen space is the load-bearing part. The filter width
is in surfel spacings, so it does not vary with resolution or camera distance --
a screen-space blur denoises a distant wall far harder than a near one, and the
splat's kernel is exactly that. And it rejects across a surface boundary by the
neighbour's own normal and tangent-plane distance rather than by whatever
survives projection. The filter runs on a **copy**: the solve keeps iterating on
the unfiltered buffer, so transport, convergence and all 25 gates are untouched
by construction, and they still pass.

**What it does not do: sharpen.** Measured on the near-floor penumbra ramp,
10-90% width is 77.6 px for the splat and 77.2 px for the denoised gather, against
the path tracer's 108. The shadow gradient lives in the cached `E` and no
reconstruction can change it. At N = 30k the spacing is 4.2 px and the gradients
here are far wider than that, so the reconstruction was simply never the binding
constraint on this scene -- finding 12's "edge width tracks 2 x spacing" was
measured on a sharper edge than the ones that dominate this view.

Cost, 1600x900, RTX 3060 mobile:

| | reconstruction pass |
|---|---|
| splat (30k point sprites, blended) | 1.98 ms |
| gather, no denoise | **1.63 ms** |
| gather + 2 denoise iterations | 3.33 ms |

The gather alone is *cheaper* than the splat. The grid it needs is 73x72x73 cells
(16.1% occupied), 276k entries at 9.2 per surfel, 8.2 MB, built on the CPU in
11 ms. Entries per surfel are higher than the 2-3 predicted because fat insertion
uses the bounding sphere and the surfel diameter (0.0329) exceeds a one-spacing
cell (0.0291), so a sphere straddles 2-3 cells per axis rather than 1-2.

Defaults are now `SGI_RECON=1` (gather), `SGI_GATHER_R=1.0`, `SGI_FILTER=2`,
`SGI_FILTER_R=2.0`. The splat stays as `SGI_RECON=0`, because finding 12's numbers
were taken with it and a reconstruction change has to stay comparable against the
one it replaces.

### 16. Microbuffer reprojection: built, verified, and not yet better

`E` is the INTEGRAL of a microbuffer, so interpolating two surfels' `E` gives a
linear ramp between two scalars while the true transition between them is not
linear. Everything that would say what shape it really has -- which directions
see the emitter, where the occluder is, how far away it is -- is destroyed at
integration. So: persist the microbuffers, REPROJECT them into each pixel's own
hemisphere using the per-bucket depth, and integrate afterwards.
`shaders/mb_reproject.comp`, `SGI_RECON=2`.

**It is implemented and it is correct.** The identity gate is the one that
matters, because a transposed basis or an off-by-one bucket index still produces
a plausible image: reprojecting with `k = 1` must reproduce the source surfel's
own `E`.

| | |
|---|---|
| mean ratio, k=1 reprojection vs the surfel's own E | **1.001** |
| median abs difference | 2.33 / 255 |
| whole-image ratio vs the gather | 0.837 against 0.850 |

**It is also not better.** MAE against the direct reference is 8.44 where the
gather is 8.42, and it is visibly blockier. The reason is the one flagged when
it was planned: the reprojection reads the stored microbuffers and never touches
the integrated `E`, so it never benefits from finding 15's object-space denoise
and shows the full 7.4x cache noise.

Three things were learned building it, and they are the durable part.

**1. Occupancy and reconstruction weight are different quantities.** The first
version used each sample's solid angle for both, which makes the reconstruction
weight track sample density -- and 256 samples landing in 256 buckets is a
Poisson process, so a fraction `e^-1 = 37%` of buckets receive nothing at all and
were handed to the sky. Measured hemisphere coverage was **0.48 in a closed
room**, and the image came out 13% dark (ratio 0.739 against the gather's 0.850).
Separating them -- the depth test and each sample's stored fill decide occupancy,
a normalized kernel decides reconstruction -- recovered the energy exactly
(0.852). The resampling kernel floor `u_min_fp` then sets the hole rate: 0.5
texels covers 0.8 of a texel, 1.0 covers ~3 and takes the expected hole rate to
`e^-3`.

**2. The tangent frame has to be shared, and that costs the rotation.** The
reader rebuilds each source's basis from its index alone, so `hash_u32`, `onb`
and the rotation moved into `common/micro.glsl` as `sgi_surfel_frame`. Per-frame
rotation (`SGI_JITTER=2`) cannot be reproduced at all; the static per-surfel
rotation (1) can, but then bucket `b` of two neighbours points in different
directions, which rules out any per-bucket filtering. `SGI_RECON=2` forces
`SGI_JITTER=0`, and doing so alone moved MAE from 9.15 to 8.44 -- the rotation
was costing more than it bought here.

**3. The penumbra ramp is not a bandwidth problem, and the metric said so all
along.** Comparing 10-90% transition widths between two ramps of different
contrast is meaningless, and that is what was being done. The endpoints:

| | lit end | umbra end | contrast |
|---|---|---|---|
| path-traced reference | 70.2 | 14.4 | 55.9 |
| splat | 40.5 | 0.9 | 39.6 |
| gather | 40.7 | 0.8 | 39.9 |

Our lit floor is at **58% of the reference** at that spot (35% in linear light).
The transition is not too soft, it is too shallow, and no reconstruction can
change that -- it is upstream, in the coverage estimator gate 8 measures as
reading this room 25.6% too open. Chasing a "shadow sharpness" number there was
chasing the wrong quantity.

**Cost.** `k * MS^2 = 2048` reprojections per pixel, one workgroup per pixel
(48 KB of shared memory forbids batching pixels), 270 ms/frame at 512x512.
Storage is 58.6 MB at N=30k, MS=16, written by `bf_micro.comp` under
`u_store_mb` -- one extra 256-element loop in a kernel that already costs 300 ms
per sweep, so unmeasurable.

**Where it goes next, in order.** Denoising the microbuffers per bucket
(`mb_filter.comp`, `SGI_MB_FILTER=1`) is implemented and made the image *worse*
(MAE 8.44 -> 8.78), so it is off by default and needs diagnosing before anything
else -- averaging bucket `b` across neighbours should be sound with a shared
frame, and it is not behaving as if it is. Beyond that, finding 3's coverage
estimator is the real target: it caps the reprojection exactly as it caps every
reconstruction, and this stage is the second measurement pointing at it.

### 17. The interpolation kernel was a cone, and a cone shows every sample

The gather always interpolated -- `SUM E_i w_i / SUM w_i` over the surfels around
a pixel -- so "the surfels are visible" was not a missing interpolation. It was
the shape of the weight.

Schaufler & Jensen equation (1) is `w = r - d`: a **cone**. It peaks at the
surfel with a kink at the apex and hits zero at `r` with another kink, so a field
of them reconstructs as a field of little bumps -- literally one visible bump per
surfel. Two more discontinuities sat next to it: the plane and normal tests were
hard rejects, so a surfel left the sum while still carrying weight.

Replaced with **Wendland C2**, `w = (1-q)^4 (4q+1)`, the standard compact-support
SPH kernel: flat-topped at `q = 0`, and value, slope and curvature all zero at
`q = 1`, so a surfel enters and leaves with no discontinuity in the result or its
first two derivatives. The plane and normal terms became smooth gates that reach
zero at their bound rather than being cut there. `SGI_GATHER_K` selects the
kernel (0 = the paper's cone, kept because this is a reference; 1 = Wendland,
default; 2 = Gaussian), and the default radius went from 1.0 to 2.5 spacings.

Cost at 1600x900, N = 30k:

| | reconstruction |
|---|---|
| splat | 6.76 ms |
| gather R=2.5, no cache denoise | **4.29 ms** |
| gather R=2.5 + cache denoise x2 | 7.14 ms |

Radius barely affects cost -- the fat-inserted grid rejects most candidates on
the first test -- so the radius is free to be chosen for quality.

**What it fixes and what it does not.** Wall and floor interiors are visibly
smoother, and the improvement grows with radius (3.5 spacings is smoother still).
What remains is at SURFACE BOUNDARIES: at a wall/wall corner the plane and normal
gates reject the other surface, and the boundary of that rejection is itself
jagged at surfel scale, so the corner stair-steps. That is the next thing to fix
and it is a different problem from the kernel -- a smooth kernel cannot smooth a
rejection boundary, only a better-conditioned rejection can.

Note also that the aggregate metrics could not see any of this: surfel-scale sd
on the back wall reads 5.28 for the cone and 5.43 for Wendland, i.e. slightly
*worse*, while the crops show the opposite. Every blind high-pass metric tried
here is dominated by real image content (edges and shading gradients), and the
only reliable instrument in this stage has been a 1:1 crop of a surface known to
be flat.

### 18. The penumbra structure is noise, so an edge-aware weight would lock it in

Proposal on the table: store a per-surfel scalar from the microbuffer, and where
neighbours disagree about it, sharpen the interpolation instead of blurring, to
keep shadow boundaries crisp.

The mechanism is sound -- it is an edge-stopping (bilateral) weight, cheap, one
float per surfel. But applied to THIS artifact it would make things worse, and
the denoise sweep is what shows why. Floor penumbra beside the short box, one
sweep, 1600x900:

| cache denoise | penumbra |
|---|---|
| x0 | heavy surfel-scale blotching |
| x2 | mostly clean |
| x12 | clean, and the penumbra gradient is fully intact |

The structure in the penumbra is **noise in the cached E**, not a reconstruction
failure -- more denoising removes it, and the penumbra survives untouched because
its gradient is far wider than the filter. A bilateral weight cannot tell that
noise from an edge: finding 15 measured the per-surfel value at 7.4x the path
tracer's high frequency on a provably smooth surface, so to any similarity-based
weight the noise *is* the edge signal. It would be preserved rather than removed,
which is the classic bilateral failure mode.

There is a second, more basic limit. An edge-aware weight can PRESERVE an edge
present in the surfel data; it cannot place one that is not. Two surfels one
spacing apart reading 1.0 and 0.0 give a sharp transition at an arbitrary point
between them, quantized to the surfel spacing -- trading a soft, correctly-placed
penumbra for a hard, wrongly-placed one, which reads as a jagged blocky shadow
edge. And a penumbra is genuinely smooth: the reference's is soft, so sharpening
it is the wrong target.

**What survives from the idea, and it is the good part:** a per-surfel scalar
that says whether local variation is signal or noise would let the denoise be
ADAPTIVE -- strong on flat lighting, restrained across a real boundary. For that
the scalar has to be less noisy than E itself, which rules out E and rules in
something like visibility toward the emitter: bounded in [0,1], no radiance
magnitude, no albedo or bounce mixed in.

**The established fix for "I can see the cache samples" is the irradiance
gradient** (Ward & Heckbert 1992). Store grad-E per surfel and interpolate to
first order, `E(p) = SUM w_i (E_i + grad-E_i . (p - p_i)) / SUM w_i`. It
reproduces a linear ramp exactly, which is locally what a penumbra is, so the
piecewise-constant blobbiness disappears without sharpening anything into the
wrong place. It is computable from the per-bucket radiance and depth that
finding 16 already persists -- that is exactly the hemispherical division Ward's
gradients are derived from.

**The Cornell box top edges are a different problem** and belong to finding 17:
there the two faces have different normals, the gather rejects across them, and
the rejection BOUNDARY is jagged at surfel scale. No kernel and no gradient fixes
that; a better-conditioned rejection does.

**Also in this stage: the denoise is now cached against the solve version**
(sweeps, cursor). The irradiance only changes when the solve advances, so a
converged or paused solve reuses the filtered buffer. That takes the iteration
count out of the per-frame cost entirely, at 1600x900:

| | reconstruction |
|---|---|
| splat | 7.98 ms |
| gather + denoise x6, solve running | 7.37 ms |
| gather + denoise x6, converged | **2.18 ms** |
| gather + denoise x24, converged | 2.43 ms |

which is what makes a genuinely strong denoise affordable. Defaults are now
`SGI_FILTER=6`, `SGI_FILTER_R=3.0`.

### 19. The cache denoise was bending light around corners, and light_delta fixes it

Finding 18 got this wrong. The object-space denoise averages `E` across
plane-consistent neighbours with **no lighting context whatsoever**: a floor
surfel in umbra and one in full light are coplanar and share a normal, so every
geometric test passes and they are averaged together. At six iterations over
three spacings that diffuses light straight into the shadow. Measured on the
floor umbra beside the short box (pixels the undenoised render puts below 6/255):

| | umbra mean |
|---|---|
| no denoise (the correct answer) | 0.488 |
| **denoise x6, no lighting context** | **2.482** |

A **5x overshoot**: the shadow was being filled in. Raising the denoise strength
in finding 18 made this worse, and the metric used there -- surfel-scale sd on
flat regions -- is blind to it by construction, because filling a shadow is a
*low*-frequency error.

**The fix is a per-surfel lighting descriptor**, `light_vis`: the cosine-weighted
fraction of the hemisphere covered by emitters this receiver can actually see,

    light = INTEGRAL [emitter visible] cos dOmega / PI

accumulated in `bf_micro.comp` from a coverage channel restricted to emissive,
front-layer surfels (210 emitters of 30k, so the extra atomic is nearly free) and
written to binding 16. Both the denoise and the pixel gather multiply their
weights by `exp(-(dv/sigma)^2)` on it.

It must be this and not `E`. `E` carries 7.4x the path tracer's high frequency on
a provably flat surface (finding 15), so a similarity test on `E` cannot separate
a shadow edge from sampling noise and would preserve exactly what the filter
exists to remove. `light_vis` is bounded, pure visibility geometry -- no radiance
magnitude, no albedo, no bounce -- and it does not drift as the solve converges.

**It has to be normalized, and that was the first version's bug.** Absolute
`light_vis` is scene-dependent: Cornell's panel subtends ~4 of 256 buckets, so a
fully lit floor surfel reads **0.016, not 1**, and every sigma tried was wider
than the entire range -- the edge stop moved the umbra from 2.482 to 2.470, i.e.
did nothing. `bf_micro.comp` now keeps a running maximum via one `atomicMax` on
the bit-cast float (binding 17, monotone so it needs no clear), and sigma reads
as a fraction of the umbra-to-full-light range.

| `SGI_LIGHT_SIGMA` | umbra mean |
|---|---|
| off | 2.482 |
| 0.25 | 2.437 |
| 0.12 | 2.291 |
| 0.06 | 1.831 |
| 0.03 | 1.119 |
| **0.015 (default)** | **0.554** |
| 0.008 | 0.292 |

At 0.015 the umbra is back to 0.554 against the undenoised 0.488 -- the leak is
gone while the denoise still runs everywhere the lighting is uniform.

**The honest tradeoff.** The filter now stops averaging exactly where visibility
varies, which is the penumbra, so the penumbra keeps its raw sampling noise and
the shadow boundary shows visible cell structure again. That is the correct
direction -- noisy and in the right place beats smooth and in the wrong place --
but it is a real cost and it is visible. Averaging *along* the iso-visibility
direction instead of rejecting outright is the obvious refinement and is not done.

**Method note.** Two findings in a row were wrong because the metric could not see
the defect: surfel-scale sd is blind to a filled-in shadow, and every blind
high-pass metric tried in this example has been dominated by real image content.
The reliable instruments here have been a 1:1 crop of a surface known to be flat
and a masked mean over a region known to be in umbra.

### 20. Averaging along iso-visibility: the gradient, and the limit it runs into

Finding 19's edge stop rejects any neighbour whose emitter visibility differs.
That protects the umbra, but it also stops the denoise dead in a **penumbra**,
because a penumbra is exactly where visibility varies -- so the penumbra kept its
full sampling noise (measured 10.15 against 6.66 for no denoising at all: worse
than not filtering, because the cache stayed noisy while the boundary got cell
structure).

A penumbra is not a discontinuity, it is a smooth ramp. So fit a **local linear
model of visibility** per surfel (`light_grad.comp`: weighted least squares in
the tangent plane, solved as a 2x2, stored as a world-space vector so nothing
downstream has to agree on a frame) and test the RESIDUAL against it:

    residual = v_j - v_i - grad_i . (p_j - p_i)

Along a ramp the residual is ~0, every neighbour is accepted and the penumbra is
denoised at full strength -- and averaging a linear function with a symmetric
kernel returns its centre value, so the ramp survives the averaging rather than
being flattened by it.

**Two attempts to make it discriminate failed, and the second one is the
interesting result.**

*Trusting the gradient unconditionally* put the leak back: a least-squares fit
always returns a plane, including across a hard boundary, and there it returns a
steep one that "explains" the jump. Umbra went 0.554 -> 1.819.

*Gating on the fit quality* -- reject the gradient where the linear model
predicts the neighbourhood badly -- over-corrected: `light_vis` carries its own
sampling noise, so the residual is large almost everywhere and the gradient got
thrown away in the penumbra too, the case it exists for. Umbra 0.728, penumbra
noise 9.88, i.e. back to isotropic.

*A robust reweighted refit* -- downweight the neighbours the first fit disagrees
with, so the gradient is fitted to the surfel's own side of a boundary -- changed
almost nothing (umbra 1.777 against 1.819). **That is the real finding: at surfel
resolution a sharp shadow boundary and a steep penumbra ramp are the same signal.**
The sampling smears a contact shadow over a couple of spacings, which is a steep
ramp, and no estimator can separate the two from the data. It is an information
limit, not a fitting problem, and no amount of cleverness in the fit removes it.

So it is exposed as a knob, `SGI_LIGHT_GRAD` in [0,1], scaling how far the linear
model is trusted. The trade is monotone and clean (sigma 0.015, denoise x6, floor
umbra and penumbra beside the short box):

| gradient trust | umbra mean | penumbra noise |
|---|---|---|
| no denoise at all | 0.488 | 6.66 |
| 0 (isotropic, finding 19) | 0.554 | 10.15 |
| 0.25 | 0.617 | 9.72 |
| **0.5 (default)** | **0.764** | **8.73** |
| 0.75 | 1.157 | 6.78 |
| 1.0 | 1.777 | 5.40 |
| edge stop off entirely | 2.482 | 4.68 |

0.5 removes 78% of the shadow leak the unguarded denoise produced while taking a
noticeable bite out of the penumbra noise. 0.75 matches the unfiltered penumbra
noise exactly, at a leak that is still a third of the unguarded case. The right
value depends on whether shadow fidelity or penumbra smoothness matters more, and
that is a judgement rather than a measurement, which is why it is a slider.

The gradient pass runs on the same cache-version trigger as the denoise, so a
converged solve pays for it once.

### 21. Next event estimation with a cone-bitmask, and the bugs it took to get there

The microbuffer was doing double duty: visibility structure AND light sampling
structure. Right for indirect, hopeless for direct -- Cornell's panel lands on
about 4 of 256 buckets (finding 16), so the term carrying every sharp shadow was
quantized to ~5 degrees with a penumbra built from four samples of the light.

Split it. `nee_direct.comp` samples each proxied emitter explicitly with its own
visibility query; `bf_lout.comp` removes those emitters' EMISSION so the two
paths partition the emitter set (their coverage is untouched, so they still
occlude). Doing the removal in `bf_lout.comp` rather than in `bf_micro.comp`
matters: it is the single place outgoing radiance is defined, so M1, M2 and every
gate see the same partition instead of one of them drifting.

**Why a bitmask.** Summed disc coverage double-counts overlapping occluders and
then clamps, and this example has been bitten by that twice -- finding 14 measured
a closed room reading 25.6% too open, finding 19 needed a calibration constant to
stop a second layer drawing light out of that bias. `OR` is idempotent, so
overlapping occluders at different depths compose exactly. It replaces the biased
estimator instead of calibrating it, and it is noise-free: 64 stratified samples
of the emitter with no random sampling. The mask is a `uvec2` and not a `uint64`
because `GL_ARB_gpu_shader_int64` is an extension while `bitCount` on `uint` is
core GLSL 4.0.

**Per-bit analytic contribution, not form factor times scalar visibility.** Each
bit is a patch of known area, direction and cosines, so the visible bits are
summed directly. That is not pedantry here: the panel subtends **25 degrees** from
a floor point half a unit below it, and at that width visibility correlates with
direction across the light. A scalar visibility fraction discards that.

Result at the reference camera, one sweep, against the direct path trace:

| | ratio | MAE |
|---|---|---|
| microbuffer | 0.854 | 8.46 |
| **NEE** | 1.106 | **7.74** |

and binned by reference brightness, which is the informative view:

| ref bin | reference | microbuffer | NEE |
|---|---|---|---|
| 0-5 (umbra) | 0.30 | 0.86 | 2.20 |
| 15-30 | 22.85 | 14.66 | 20.92 |
| 30-55 | 40.54 | 30.36 | **40.83** |
| 55-90 | 70.78 | 59.76 | 80.55 |
| 90-140 | 100.52 | 91.38 | 119.25 |

NEE is within **0.7%** in the 30-55 band where the microbuffer is 25% low. The
growing overshoot toward the bright end is not a magnitude error -- a magnitude
error would be constant across bands -- it is a curve SHAPE difference, i.e.
finding 9's unported AgX, which the microbuffer's coverage-loss darkness had been
accidentally masking. Two errors cancelling, the same trap finding 13 recorded for
64 buckets reading closer than 256.

**Four bugs, three of them silent, all worth recording.**

*The gate's own analytic answer was wrong.* First run reported NEE 217% high --
and the microbuffer 213% high. Two independent implementations agreeing with each
other and disagreeing with the analytic value is the signature of a wrong analytic
value: `E = pi * L * F`, and the gate had `E = L * F`. Section 9's "if you are off
by pi, stop" applies to the test as much as to the code. Corrected, NEE is +1.03%
and the microbuffer -0.28% against the closed form.

*The emitter proxy was 186% of the panel's area.* A quad built from two triangles
lists its diagonal corners twice, so a PCA over vertices is skewed toward that
diagonal and the principal axis comes out at 45 degrees. Replaced with a
**minimum-area bounding rectangle**, which is immune to vertex multiplicity and
exact for a rectangular source: ratio 1.0000. The area assertion in the fit is
what caught this, not the image.

*The visibility buffer had no storage.* `run_direct` runs BEFORE `dispatch`, and
`b_light_` was allocated inside `dispatch`, so the direct pass wrote `light_vis`
into a zero-sized buffer for the whole first solve. The direct term itself was
correct throughout, which is what made it confusing -- the image changed, so the
pass was obviously running.

*A cap of 6 cells left 59% of the frustum unscanned.* Marching the axis with a
fixed box neighbourhood has to cap the box or the cost explodes, and Cornell's
panel is 14.6 cells across at its own plane. A 25-degree light is not the sun.
Replaced with a frustum walk over a **coarse occupancy bitmask** (one bit per
4x4x4 block, built with the grid): the interior is hollow, so nearly every block
in the open volume is rejected on a single bit. Cost 1030 -> 540 ms and the
truncation is gone.

**Remaining: the umbra still leaks**, 2.20 against the reference's 0.30 at the
default `kMinAxis` 0.75 (1.13 at 1.00, which costs nothing in the lit regions --
the clamp is invisible there). In linear light both are near black, but it is a
real residual and it is the next thing to chase.

**Cost.** The direct term depends only on geometry and emission, not on the
sweep's irradiance, so it is computed once per solve and cached, not once per
sweep. Gates: 27/27, including two new partition gates.

### 22. Degree-1 moving least squares

`SUM E_i w_i / SUM w_i` is Shepard interpolation, degree-0 MLS, whose signature
artifact is a **flat spot at every data point** -- the field becomes plateaus with
transitions between them, and it provably cannot reproduce a linear ramp, which is
the one thing a penumbra locally is. "Each blob has the same colour inside of it"
and "the penumbra has no clean gradient" are one defect, not two.

Degree 1 fits `a + b*x + c*y` and evaluates it at the query point. Because the
query point is the origin of the offsets, **only `a` is needed** -- one Cramer
numerator, no full solve, and no per-surfel gradient stored anywhere; the 3x3
matrix depends only on positions so it is shared across R, G and B.

Verified in a 1-D analogue before building it (noisy ramp, Wendland weights,
R = 2.5 spacings): Shepard's error is ~0.2 **and structured** -- it lags and
flattens, which is what reads as plateaus -- while degree-1 is ~0.04 and unbiased.

Two guards are load-bearing: fall back to Shepard when the normal matrix is
ill-conditioned (too few neighbours, or collinear ones -- silhouettes and thin
geometry), and **clamp the fitted value to the range of the contributing
samples**, because a linear fit extrapolates and near a boundary it will overshoot
into a halo or negative light.

Measured on the penumbra, the fraction of pixels whose 3x3 neighbourhood is
perfectly constant: **35.8% -> 30.6%**. Real but modest, and smaller than the
1-D analogue suggested -- the remaining plateaus are not all interpolation, some
are the cache's own resolution. `SGI_MLS=0` keeps Shepard.

### 23. The edge leakage was three coupled defects in one function

The direct term leaked and over-darkened along edges at the same time, which is
the signature of a projection that is wrong in *shape* rather than in *amount*.
All three defects lived in `sgi_project_occluder`, and none of them could be
tuned out on its own.

**(a) The ellipse had no orientation.** The first version returned two scalar
semi-axes and assigned the unforeshortened one to `half_u` and the foreshortened
one to `half_v` -- whatever the disc's actual orientation. That is wrong by
construction: rotating an occluder about the line of sight leaves its shadow
unchanged, but it flipped the mask between over- and under-occluding. It leaked
along every edge whose surfels happened to foreshorten along `u`. The silhouette
of a disc seen from `P` is an ellipse whose unforeshortened axis lies along
`cross(n_o, w)`; both axes are perpendicular to `w`, so both carry through the
projection by the same construction as the centre. They are stored as **conjugate
semi-diameters** (`vec2 a0, a1`), not principal axes -- the inside test only needs
`[a0 a1]` to be invertible, and the principal axes are never needed.

The offsets are carried exactly rather than paraxially: the ray `P -> (C+E)` meets
the plane at `P + s*(v+E)` with `s = num / (denom + dot(E, n_e))`, so the shadow's
offset is `(s - s0)*v + s*E`. One divide more than `s0*E`, and `s0*E` is wrong by
a factor that grows with how obliquely the shadow lands on the light.

**(b) The minor-axis clamp was a constant over-occlusion applied where it was
least justified.** An edge-on disc projects to a line and occludes nothing, so a
disc-tiled surface goes transparent under grazing light -- hence the old
`max(|cos|, 0.75)` clamp. But a ceiling disc seen edge-on from a wall point a few
centimetres below the ceiling was then inflated to 0.75 of its full radius and
blacked out the light. **That is the ragged dark band along every concave edge.**

A surfel is better modelled as a thin oblate **spheroid** of semi-axes
`(r, r, thick*r)`, whose silhouette semi-minor axis is exactly
`r * (|cos| + thick*|sin|)`. It equals `r` face-on and degrades to `thick*r`
edge-on rather than to a constant. The distinction that makes this a model and not
a fudge: a **long** grazing path through a wall crosses many discs whose slivers
still `OR` together into an opaque mask, while a **short** one at a corner does
not -- correct in both cases.

**(c) The receiver bias was papering over a missing test.** The bias existed to
lift a receiver out of its own plane so its coplanar neighbours stopped
registering as occluders. But lifting a floor point a full surfel radius also
lifts it out from behind whatever is standing on the floor -- a leak at every
contact and every concave edge. **A flat surface does not shadow itself**, so test
for that directly (`sgi_same_surface`): near-parallel normals *and* near-zero
offset from the receiver's tangent plane. A crease fails the first test and a step
fails the second, so both still occlude. With it in place the bias drops from
**1.0 radii to 0.05**.

**Measured**, direct term only, `SGI_GTCAM=1`, against
`CornellBoxGroundTruthDirectLighting.png`:

| config | ratio | MAE | 0-5 | 5-15 | 15-30 | 30-55 | 55-90 | 90-140 | 140+ |
|---|---|---|---|---|---|---|---|---|---|
| reference | | | 0.30 | 10.57 | 22.85 | 40.54 | 70.78 | 100.52 | 239.67 |
| old-like (thick .75, bias 1.0, self off) | 1.004 | 8.00 | 0.66 | 5.73 | 16.61 | 36.68 | 78.39 | 119.76 | 193.37 |
| **new (thick .25, bias .05, self on)** | 1.052 | **6.01** | 0.90 | 8.00 | 20.35 | 40.36 | 79.48 | 118.67 | 193.37 |
| new, self rejection off | 0.990 | 7.90 | 0.85 | 5.45 | 16.58 | 35.94 | 77.58 | 117.43 | 193.37 |
| new, bias back to 1.0 | 1.059 | 6.50 | 0.91 | 7.59 | 20.05 | 40.46 | 80.52 | 120.40 | 193.37 |
| new, thick back to 0.75 | 1.043 | 6.10 | 0.65 | 7.62 | 19.98 | 40.02 | 78.84 | 118.64 | 193.37 |

The **same-surface rejection is the load-bearing one**: without it the 5-15 and
15-30 bands collapse to 5.45 and 16.58 against the reference's 10.57 and 22.85 --
that is the surface shadowing itself, and on screen it is blotchy self-shadowing
all over the tall box's face and the back wall. Once it is on, thickness barely
matters (6.01 vs 6.10 between 0.25 and 0.75), because what the clamp was mostly
inflating *was* the receiver's own plane. Sensitivity is flat in every knob --
thickness 0.15/0.25/0.35/0.50 and bias 0.0/0.05/0.2/0.5 all move the MAE by less
than 0.1, and `self_tol` 0.5 through 20.0 is bit-identical. That flatness is the
signature of a model rather than a fudge, and it is the reason no number here was
tuned to the reference.

Gate 11's partition number improved as a side effect: NEE against the analytic
rectangle went from **+1.03% to +0.07%**. Gates: 27/27.

### 24. The direct term per pixel

Per-surfel receivers make the cache the resolution limit. A surfel is ~19-37 px
across at the reference camera, so a shadow boundary is sampled at that spacing
and interpolated; the silhouette comes out jagged no matter how good the
interpolation is. That is Nyquist, not a filtering failure, and finding 22's MLS
could only ever soften it.

`nee_pixel.comp` runs the *same estimator* -- literally the same code, shared
textually through `common/nee.glsl`, because a divergence between the two would
show up as a direct/indirect mismatch that is impossible to attribute -- with the
receiver read from the G-buffer instead of the surfel set. Nothing else changes.
The per-surfel pass still runs and still feeds `bf_lout.comp`, because the
**bounce transport** needs the direct term in the cache regardless of where the
image takes it from. Indirect stays interpolated from the cache, which is the
right place for it: low frequency, and 48x cheaper there.

A pixel has no self index, and does not need one: `sgi_same_surface` is a strictly
better test than an index compare, which only ever excluded *one* of the several
discs overlapping any given point.

**Composites into the reconstruction's target** with a read-modify-write, so it
needs no extra full-resolution texture. The gather and the reprojection store `E`
directly in `.rgb`; the splat's target is a `(w*E, w)` accumulator that the
display pass divides, so it is incompatible and the host forces a gather when this
is on.

**The sweep index shifts by one, and this is not a bug.** Under per-surfel NEE the
direct term is added into the irradiance by `bf_micro.comp` at the end of a sweep,
so sweep 1 produces direct only -- its `lout` input was all zero, since emission is
zeroed under the partition. Under per-pixel NEE `bf_lout.comp` adds `direct_e`
when forming outgoing radiance, so sweep 1 already produces bounce 1. After *k*
sweeps: per-pixel is `D + B1..Bk`, per-surfel is `D + B1..B(k-1)`. **Per-surfel
`SGI_BOUNCES=k+1` matches per-pixel `SGI_BOUNCES=k`**, and per-surfel's first
sweep is a wasted O(N^2) pass. I spent a while treating this as a 33% overshoot
before noticing I was comparing different bounce counts.

**Measured.** Direct term alone (`SGI_SOLVE=0 SGI_PAUSE=1` for the per-pixel path,
which leaves the cache at zero), 512^2, against the direct reference:

| | ratio | MAE | 0-5 | 5-15 | 15-30 | 30-55 | 55-90 | 90-140 | 140+ |
|---|---|---|---|---|---|---|---|---|---|
| reference | | | 0.30 | 10.57 | 22.85 | 40.54 | 70.78 | 100.52 | 239.67 |
| per surfel | 1.053 | 6.16 | 1.16 | 8.87 | 20.68 | 40.35 | 79.27 | 117.50 | 193.37 |
| **per pixel** | 1.052 | **6.01** | **0.90** | 8.00 | 20.35 | 40.36 | 79.48 | 118.67 | 193.37 |

Better overall and **22% less umbra leak** (0.90 against 1.16, reference 0.30).

Full GI at matched bounce counts (per-surfel 8 sweeps, per-pixel 7), against
`CornellBoxOriginalGroundTruth.png`: ratio 0.946 / 0.945, MAE 13.08 / 13.04, and
every band agrees to within 0.3. The two paths are energetically identical, as
they must be -- the difference is *where* the direct term is evaluated, not how
much of it there is. The whole of the improvement is spatial.

**Visually**, at `SGI_GTCAM=2` (1600x900), 1:1 crops:

* **Box/floor contact.** Per surfel, the contact shadow is a blocky staircase
  spilling one to two surfel widths past the box, with a grey halo where the umbra
  should be black. Per pixel it hugs the base exactly, the umbra is black to the
  contact line, and the penumbra is a smooth gradient.
* **Wall/ceiling junction.** In the `SGI_SHOW_LIGHT` visibility map the per-surfel
  boundary is a sawtooth with roughly surfel-period teeth; the per-pixel one is a
  straight line. **But the ragged dark band survives in the final image, identical
  in both** -- so it is not a direct-lighting artifact at all. It is the
  microbuffer's, and per-pixel direct cannot touch it. That is worth recording
  because the band was on the list of things this change was expected to fix.
* What per pixel *adds* is a faint speckle inside contact umbrae and sliver
  "grass" along the floor/wall junction. Those are the **occluder** representation
  -- the disc soup -- now resolved rather than blurred away by the receiver
  spacing. Per-pixel direct trades receiver-resolution blur for occluder-
  resolution detail; the occluder side is the next limit.

**Cost.** 194 ms at 512^2 and 482 ms at 1600x900, against ~4.6 ms for the whole
reconstruction. That is 1.44M receivers against 30k on a flat uniform grid with a
4x4x4 occupancy bitmask and no hierarchy, so treat it as an upper bound rather
than the cost of the technique. `SGI_NEE_PIXEL=1`.

### 25. Coverage 1.0 does not seal, and Cornell cannot show it

The parameters this pass introduced are all **dimensionless** -- `thick`, `bias`
and `self_tol` are in surfel radii, `self_cos` is a cosine -- so none of them
carries a scene scale, and none of them can be fitted to Cornell in the units
sense. That is necessary and it is not sufficient. What they *can* be fitted to is
Cornell's **geometry**, and Cornell presents exactly one configuration: a small
panel directly overhead, every surface flat, every dihedral 90 degrees, and no ray
from a receiver to the light ever crossing a surface at a shallow angle. Whatever
is wrong in the cases it does not present is invisible here by construction.

**Gate 12 (`graze`) is the case it does not present.** An opaque single-layer wall
between a receiver and an emitter, with the whole configuration rotated from
head-on to nearly in the wall's plane. Leak is the fraction of the unoccluded
direct term that gets through:

```
theta from the normal        0      45      70      80      85
cov 0.66 occ 1.00 thk 0.00   31.25   31.35   31.36   31.36   31.36
cov 0.66 occ 1.00 thk 0.25   28.15   27.06    5.81    2.77    0.00
cov 1.00 occ 1.00 thk 0.00   17.19   17.27   17.29   17.29   17.29
cov 1.00 occ 1.00 thk 0.25   17.19    4.46    0.00    0.00    0.00
cov 1.00 occ 1.25 thk 0.00    0.00    0.00    0.00    0.00    0.00
cov 1.00 occ 1.25 thk 0.25    0.00    0.00    0.00    0.00    0.00
```

**17% of the light goes straight through a wall at the coverage the baker
produces, at every angle, at every thickness.** Thickness cannot touch it: a
head-on disc is already at its full radius, so there is nothing for the spheroid
model to inflate. This is not a tuning failure, it is a modelling one, and it had
been sitting under every measurement in findings 21-24.

**The occluder radius is not the energy radius.** The bake sizes surfels so that
`sum(pi r^2) == A` exactly -- gate 6 asserts it -- which is the right condition for
a disc to carry the right amount of *energy*, and the wrong one for it to *seal*.
Equal circles cannot tile a plane: the densest possible packing covers 0.9069 of
it, and the Vogel spiral the bake uses is looser still. A surface tiled at
coverage 1.0 has real holes and a shadow ray finds them.

So scale the radius used for **visibility only** (`u_occ`, `SGI_NEE_OCC`, default
**1.25**). Energy, form factors and the microbuffer keep the bake's radius, so
nothing downstream of gates 2 and 6 moves. Coverage goes as the square, so 1.25 is
coverage 1.56, and the gate reads 0.00 from 1.5 up; 1.1 (coverage 1.21) is not
enough, at 4.7%. It is the smallest value that seals, not a fitted one.

**Why Cornell never showed it.** Its shadow casters are closed solids, so a ray to
the light crosses two disc layers (lid and side) and one layer's holes are covered
by the other's; and nothing is ever behind a wall, so the walls' holes have nothing
to leak into. Single-layer geometry -- a fence, a leaf, a curtain, a wall lit from
both sides -- is the common case everywhere else.

**What it costs here**, against the two path-traced references at 512^2:

| | direct MAE | umbra (0-5) | full-GI MAE |
|---|---|---|---|
| reference | | 0.30 | |
| occ 1.00 | 6.01 | 0.90 | 13.04 |
| occ 1.25 | 6.12 | 0.66 | 13.37 |

The umbra leak drops 27% and the whole-image MAE rises ~2%, because the penumbra
was already too dark (band 5-15 reads 8.00 against the reference's 10.57 at occ
1.0) and inflating occluders makes that worse. Those are two opposite errors with
different causes and the second one is finding 26.

**The other geometric assumptions, stated as bounds rather than guessed at.**
`sgi_same_surface` drops an occluder only when it is within `self_tol` radii of the
receiver's tangent plane **and** within `acos(self_cos)` = 25.8 degrees of its
normal. Both conditions, so:

* a **crease** shallower than 25.8 degrees loses occlusion in a band of width
  `self_tol * r / sin(alpha)` along the crease line. Cornell's dihedrals are all
  90 degrees, so it never happens here; the error is self-limiting, because a
  shallow crease occludes little in the first place.
* a **convex** curved surface is safe at any curvature: every other point of it
  lies below the local tangent plane, so it cannot occlude anyway.
* **grazing light on a near-flat surface** is where it genuinely fails: relief
  below one surfel radius should self-shadow and is discarded. Cornell's light is
  directly overhead and its surfaces are exactly flat, so this is the assumption
  it most thoroughly fails to test.
* turning the test off is not an option -- the same crop that shows the teeth
  shows the entire wall covered in self-shadow (finding 23).

**And the bake can violate its own coverage guarantee.** `CornellBoxBunnyMirror.glb`
needs 98829 surfels for its 69483 triangles; the set is capped at 65535 by the
`depth16 | index16` microbuffer key and **truncated**, landing at
`sum(pi r^2)/A = 0.663`. The gate's `cov 0.66` rows are what that costs: 17% leak
even with the occluder scale on, 31% without it. Truncation is the wrong response
to the cap -- it leaves the surface with holes rather than with coarser discs --
and it means the estimator's coverage precondition is silently false on any scene
that hits the ceiling. `SGI_MODEL=path.glb` loads an alternative model, for
exactly this kind of check.

### 26. What is left is on the occluder side, not the receiver side

Finding 24 removed the receiver's resolution limit. Everything still visible at the
reference camera is now a property of the **occluder** representation, and the
three remaining artifacts are one cause seen three ways:

* **Scalloped silhouettes.** A straight edge represented by a row of discs has a
  scalloped boundary, amplitude up to `r`, period the surfel spacing. In the
  visibility map at the tall box's base this reads as a row of triangular teeth
  along the contact line. It is present identically in the per-surfel and
  per-pixel paths, so it is not a receiver artifact; it survives every setting of
  `thick`, `occ` and `bias`, so it is not a parameter; and at 65k surfels against
  30k the teeth get finer in proportion, so it is the disc size and nothing else.
* **Pixelated shadow boundaries** are the same scallop, plus the 8x8 bit grid:
  the mask changes in whole bits, so neighbouring pixels that cover the same bit
  set get exactly the same answer. The energy weighting spreads the result over
  129-151 distinct levels in a penumbra rather than 64, so level quantization is
  not the limit; the spatial jump is.
* **Penumbrae systematically too dark** (band 5-15 at 8.00 against the
  reference's 10.57). The union of discs whose centres lie inside a surface
  extends up to `r` past its boundary, so every silhouette is **dilated by one
  surfel radius** and every shadow is slightly too large. This is the exact
  opposite of finding 25's hole leak and it is why the two cannot both be fixed
  by a radius: interior discs need to be bigger to seal, boundary discs need to
  be smaller not to dilate.

That last observation is also the way out, and it is a bake change rather than a
shader one: an interior surfel (one whose neighbourhood is fully covered) can be
inflated and a boundary surfel cannot, so the flag belongs in the spare bits of
`s_albedo`. Not built.


**Correction, from findings 27 and 28.** The claim above that the scallop is
structural and that only a bake change could reach it was half right and half
wrong. The bake change exists and is better than the interior flag proposed here
(finding 27, cut planes ported from example 38). But the *dominant* term in those
teeth was not dilation at all -- it was an occluder that is not between the
receiver and the light being counted anyway (finding 28), which no amount of
representation work would have fixed.

### 27. Cut planes: clipping the disc at the mesh's own edges

Finding 26 left the two errors at an impasse: interior discs must overlap to seal
and boundary discs must not overrun the geometry, and one radius cannot do both.
Example 38 already had the answer -- per-surfel **cut planes**, derived at bake
time from the mesh's sharp and boundary edges, against which the disc is clipped.
Inflate freely, then clip; the two errors stop competing.

Ported into `cuts.hpp` / `cuts.cpp`, keeping 38's substance:

* Cuts come from **topology, not from neighbouring samples**, so the planes land
  on triangle borders however the surface was sampled.
* Which side of an edge a disc keeps is decided from the incident faces'
  **centroids**, never from the surfel centre or the sign of an offset. 38's
  comment records why: a barycentric lattice puts whole rows of samples exactly
  ON the border, where those quantities are zero and half the decisions flip. This
  example samples with R2 rather than a lattice so exact zeros are rarer, but
  "rarer" is not a reason to use a test with no defined answer at zero.
* An edge only cuts surfels of **its own object** (union-find over shared edges),
  so a box standing on the floor is not clipped by the floor.
* Cuts are only taken between **flat faces** (per-triangle curvature from smooth
  dihedrals), because a plane through a crease on a curved surface slices
  unrelated geometry -- 38 measured that beheading bunny surfels 4 cm away.
* A cut may never reject its own surfel, which is what keeps a shallow
  self-intersection from beheading a disc.

**Deviation from 38:** the plane is stored packed, `vec4(n, -dot(anchor, n))`, one
vec4 per slot rather than two. 38 keeps the anchor form because its layout is
"ready for per-frame skinned cuts"; this example's set is static, so the packed
form is exactly equivalent and halves the buffer. `kSurfelCuts = 4`.

Applied **per bit**, and only to bits the ellipse already accepted: a bit is a
known point on the light, so the ray to it meets the occluder's tangent plane at
one point, which is either inside the clipped disc or not. Exact, with the
ellipse serving only as the cheap bound on which bits to try. Skipped within ~6
degrees of edge-on where that solve is ill-conditioned -- and the safe direction
to fail there is *keeping* the bit, since an unclipped bit over-occludes slightly
while a wrongly dropped one is a leak.

Cornell: 40 feature edges, **2669 of 29997 surfels clipped (8.9%)**, none
saturating 4 planes, 1.9 MB, 4.3 ms of bake.

### 28. The occluder was never checked to be BETWEEN the receiver and the light

The teeth along the back wall's base survived the cut planes, and chasing that is
what found the real defect. The projection answers "does the occluder's silhouette
cover this direction". That is not the same question as "does the occluder block
this ray", and the difference is not academic.

A wall disc sitting a centimetre above a floor receiver at the wall/floor junction
is seen **edge-on**, so its silhouette is a sliver -- but its centre is only ~r
away, so the projection magnifies that sliver by `d_emitter / r`, about **120x**
here, and it blankets the entire panel. The floor in front of the back wall reads
as shadowed, in a row of teeth at the surfel spacing. Nothing physical is
happening: every ray from that receiver to the light goes forward, away from the
wall, and never crosses the wall's plane at all.

`sgi_same_surface` cannot catch it -- the two normals are perpendicular, so they
are emphatically not the same surface -- and no thickness, radius or bias setting
touches it, which is exactly why it survived every parameter sweep in finding 26
and why tuning could only trade it against something else.

The occluder is the oblate spheroid of finding 23, so the exact test is a **slab
crossing**: the ray `P + s*d`, `s` in (0,1), must enter
`|dot(q - C, n_o)| <= thick * r`. Solving both plane crossings gives an
`s`-interval and the occluder blocks iff that interval meets (0, 1). The
near-parallel case falls out as a limit rather than needing its own branch: a tiny
`|dn|` leaves an interval that is either the whole line (the receiver is inside
the slab, so the disc really does block) or empty (it is outside, so it cannot).
The same per-bit ray was already being computed for the cut test, so this costs
one dot and one divide.

**It also makes `thick` honest.** Gate 12 now reads **100% leak at thick = 0**, at
every angle and every coverage — correct, because a mathematically flat disc is a
measure-zero object and blocks nothing. Previously the silhouette test alone let a
zero-thickness disc occlude. `thick` is now a required part of the model rather
than a knob, and the MAE is flat across 0.10-0.50 (5.84-5.85), so there is no
cliff next to the shipped 0.25.

**Measured**, direct term only, 512^2, against the direct reference:

| | ratio | MAE | umbra (0-5) | 5-15 | 15-30 |
|---|---|---|---|---|---|
| reference | | | 0.30 | 10.57 | 22.85 |
| occ 1.00, no cuts, no slab | 1.052 | 6.01 | 0.90 | 8.00 | 20.35 |
| occ 1.25, no cuts, no slab | 1.044 | 6.12 | 0.66 | 7.68 | 20.06 |
| occ 1.25, cuts, no slab | 1.054 | 5.93 | 0.70 | 7.99 | 20.39 |
| **occ 1.25, cuts, slab** | 1.058 | **5.84** | 0.72 | 8.07 | 20.54 |

Full GI at 7 sweeps against `CornellBoxOriginalGroundTruth.png`: **13.04 -> 12.95**,
against the 13.37 that the occluder inflation alone cost. So the sealing radius is
now free: cuts and the slab test give back everything it took and a little more.

**Visually** this is the largest single step since the per-pixel pass. In the
`SGI_SHOW_LIGHT` map at 1600x900: the teeth along the back-wall/floor junction are
**gone**, the tall box's silhouette against the back wall goes from a surfel-period
sawtooth to a straight line, the speckle inside the box's contact umbra is gone,
and the mottling on the floor near the red wall is gone. What remains is a fan of
streaks where the short box's shadow meets the green wall at a very grazing view
angle, which is that boundary's own structure under perspective compression rather
than a new artifact.

**Cost** of findings 27 and 28 together: 194 -> 200 ms at 512^2 and 482 -> 493 ms
at 1600x900, about 3%. Gates: 28/28. `SGI_NEE_CUTS=0` disables the clipping; the
slab test has no switch, because there is no configuration in which counting an
occluder that is not in the way is correct.


### 29. How much inflation seals depends on the SAMPLER, and gate 12 was lying

Finding 25 set `nee_occ` to 1.25 on gate 12's evidence: coverage 1.56, leak 0.00
at every angle. The render still leaked -- a dashed bright line along the short
box's contact with the floor and a fan of bright streaks in its shadow against the
green wall, both visible in `SGI_SHOW_LIGHT` and both obvious in the image.

The gate was measuring the wrong point set. Its wall was tiled with a **Vogel
spiral**, which is far more locally uniform than the bake's **per-triangle R2 with
a Cranley-Patterson rotation**. Same nominal coverage, different worst-case gap,
and the leak lives in the worst case, not the mean. So gate 12 now sweeps both
samplers:

```
sampler / config                     0      45      70      80      85
spiral cov 1.00 occ 1.00 thk 0.25   17.19    4.46    0.00    0.00    0.00
spiral cov 1.00 occ 1.25 thk 0.25    0.00    0.00    0.00    0.00    0.00
r2     cov 1.00 occ 1.00 thk 0.25   17.18    7.67    1.36    0.00    0.00
r2     cov 1.00 occ 1.25 thk 0.25    7.83    1.41    0.00    0.00    0.00
r2     cov 1.00 occ 1.50 thk 0.25    0.00    0.00    0.00    0.00    0.00
r2     cov 1.00 occ 2.00 thk 0.25    0.00    0.00    0.00    0.00    0.00
r2     cov 0.66 occ 2.00 thk 0.25    0.00    0.00    0.00    0.00    0.00
r2     cov 1.00 occ 2.00 thk 0.00  100.00  100.00  100.00  100.00  100.00
```

The spiral's answer (1.25) leaves the real sampler at **7.83%**. A gate that
passes while the render leaks is worse than no gate, so it now uses the bake's own
sampler, and the assertion is made at the shipped configuration.

**`nee_occ` is 2.0**, not the 1.5 the gate's knee suggests. The gate's wall is a
flat plane crossed in its interior; the render's failures are grazing paths
arriving near a *clipped boundary*, where coverage is thinnest, and 2.0 is what
closes those. It is affordable only because of the cut planes: the cost of going
1.25 -> 2.0 is **MAE 5.84 -> 5.85** on the direct term and full GI 12.95 -> 13.01,
against MAE 6.47 for the same inflation before finding 27. The umbra leak goes
0.72 -> 0.66 (reference 0.30). Above 2.5 it starts to cost properly (5.93 at 2.5,
6.03 at 3.0, 6.47 at 4.0), so 2.0 is not the top of a plateau, it is the point
just past the knee.

The cut reach follows: a disc can cross an edge within its EFFECTIVE extent, so
`kCutReachRadii` covers the top of the `nee_occ` slider rather than its default.
19.1% of surfels now carry a cut, against 8.9%. A surfel with a cut it never
needed is free; one missing a cut it needed over-occludes at every silhouette.

### 30. What is left is in the microbuffer, not in NEE

> **Superseded in part by finding 32.** The attribution to the microbuffer is
> wrong: the band renders identically under M1, which has no microbuffer and no
> occlusion. What survives is the narrower claim that it is not in NEE.

With the direct term clean, the remaining artifacts can be attributed by simply
rendering it alone (`SGI_SOLVE=0 SGI_PAUSE=1`, which leaves the cache at zero).

**The ragged dark band under the ceiling is entirely indirect.** In the
direct-only render the ceiling is black -- correct, it is coplanar with the panel
and cannot see it -- and the back wall's top is a perfectly smooth gradient with
no band at all. Add the cache and the scallops appear. It is not an NEE artifact
and never was, which is why it was identical in the per-surfel and per-pixel
paths (finding 24) and why none of findings 25-29 touched it.

**The wavy contour banding on the boxes' dark faces is the same source**: the
microbuffer's irradiance field, reconstructed by degree-1 MLS from a cache whose
samples are ~19-37 px apart. Those faces read direct == 0 to within the 6x gain
the visibility map was inspected at.

None of this session's three fixes reach the microbuffer, and that is structural
rather than an oversight: `bf_micro.comp` is a **depth-buffered rasterizer** into
a hemi-octahedral buffer, not a coverage mask. It has no per-bit ray to hang a
slab test or a cut plane on -- each bucket keeps one depth winner, so an occluder
either wins its bucket or vanishes. Giving it the same treatment means giving it
the same shape of query, which is a larger change than any made here.

**One direct-term artifact does survive**: a staircase of horizontal ledges down
the tall box's narrow right-hand face, growing with `nee_occ`. That face is
vertical under an overhead light, so it is lit at grazing incidence -- the one
configuration finding 25 named as the genuine failure of the same-surface test,
where relief below one surfel radius should self-shadow and is discarded, and
where inflated discs then over-occlude what is left. It is the smallest remaining
term and it is on the list of known limits rather than being a surprise.


### 31. The ceiling band, located: one row of surfels in the CACHE

> **Superseded in part by finding 32.** The localisation here holds -- it is one
> row of surfels in the cache, and the reconstruction only smears it. The
> mechanism proposed here does not: the coverage-spread fix was built, measured,
> and did nothing.

Finding 30 attributed the remaining artifacts to the microbuffer. This narrows it
to a mechanism, by elimination.

**It is in the cache, not the reconstruction.** The band is unchanged at gather
radius 1.2 and 4.0, at `SGI_GATHER_N=0.5`, at `SGI_GATHER_P=0.3`, and with
`SGI_MLS=0`. The splat reconstruction hides it, which is blur, not a fix. The
point-cloud view (`SGI_POINTS=1`, raw cache, no reconstruction at all) shows the
answer directly: **a single row of wall surfels immediately below the ceiling
holds a much lower irradiance than the row beneath it.** The gather then smears
that row down over its radius, which is the band.

**It is not any of the obvious knobs.** Identical at 8x8, 16x16 and 32x32 buckets;
identical with two-sided emitters; unchanged at `SGI_BIAS` 0.1 and 3.0, and at
`SGI_HORIZON=0.15`. At 8k / 30k / 60k surfels the scallop PERIOD tracks the surfel
spacing but the amplitude does not move. So it is neither an angular-resolution
effect nor a sampling-density one.

**It amplifies.** Faint at one bounce, stronger at two, strongest at seven. A
small local deficit feeds itself: the darkened row is itself the light source for
its neighbours on the next sweep.

**The magnitude is roughly right; the shape is not.** The path-traced reference
has a smooth darkening at that junction -- a point in a concave corner really does
see only a quarter-space, and the Cornell ceiling is unlit -- but it has no hard
edge and no scallops. What is wrong is the per-surfel VARIANCE, not the mean.

**Which makes it finding 25 again, in the other estimator.** A wall surfel in the
corner should have its upper hemisphere sealed by ceiling discs. At the bake's
coverage of 1.0 the discs do not tile, so some surfels in that row are sealed and
some have gaps, at random, per surfel -- and the feedback loop then amplifies the
difference. Same cause as the 17% that went straight through a wall in gate 12;
different estimator, so `nee_occ` does not reach it.

**And it does not port across directly.** In NEE, visibility and energy are
separate quantities, which is exactly why the visibility radius could be inflated
alone. In `bf_micro.comp` they are the same quantity: `omega` is the solid angle,
and it drives both the coverage and the radiance. Half of that is fine --
radiance is coverage-normalised (`L = rad / cov_front`), so a uniform inflation
cancels out of `L` entirely and only moves the fill. But the fill is
`clamp(cov_any, 0, 1)`, and inflating it brightens every PARTIALLY covered bucket,
which is gate 2's coaxial-disc form factor. So the change needed is not "scale the
radius"; it is a way to close gaps between discs of one surface without inflating
partial coverage by a distant one -- for instance by making the seal decision per
bucket against the winner's own plane, where the neighbours' membership is already
known, rather than by growing every footprint.

The convex case is the same story with the sign flipped: the seams along the tall
box's edges are present at bounce 1, invariant to `SGI_BIAS`, and read as green
blotches on a face that should smoothly pick up the green wall.

**On the crude same-surface test.** `bf_micro.comp` still rejects an occluder by
`dot(dv, nP) <= u_plane_bias * r`, which is the receiver-plane offset alone --
precisely the test finding 23 replaced in NEE with `sgi_same_surface` (normal
agreement AND plane distance), worth MAE 7.75 -> 6.68 there. It is not the cause
of this band, since sweeping it does nothing, but it is the same known-inferior
test and it should go the same way when the microbuffer is reworked.

### 32. Correction: the band is not occlusion, and not the microbuffer

Findings 30 and 31 attributed the ceiling band to the microbuffer's coverage
estimator and predicted that inflating the footprint would close it. Both halves
were wrong, and the test that settles it is one line:

**`SGI_METHOD=0` renders the band identically.** M1 is the analytic
point-to-point estimator -- no microbuffer, no buckets, no depth winner, **no
occlusion at all**. The two images differ by MAE 3.83 overall, so the switch
really took effect, and the band is in both. An artifact that survives removing
occlusion entirely is not an occlusion artifact.

Everything else that was tried and did nothing, for the record: the phase-B
coverage spread finding 31 proposed (implemented, measured at 1.0 / 1.5 / 2.0 /
3.0, MAE 0.099 at the widest -- removed again); `u_soft_eps` from 0 to 8 (MAE
0.12); `u_layers`, `u_cov_mode`, `u_back_gate`; 8x8 vs 16x16 buckets; 8k / 30k /
60k surfels; every gather parameter.

**Where it actually is, measured.** The G-buffer normal at 1600x900 puts the
ceiling/wall junction exactly at row 191, and the luminance step is exactly there
too -- so the band is the top rows of the BACK WALL, not a feature part-way down
it. Against the path-traced reference at 512^2, scanning down columns 150-360:

| row | reference | ours | ref sd | our sd |
|---|---|---|---|---|
| 104 (ceiling) | 80.03 | 84.34 | 4.31 | 6.07 |
| 108 (ceiling) | 74.26 | 72.80 | 4.25 | 5.30 |
| **109 (wall top)** | **71.68** | **55.50** | **4.16** | **12.15** |
| 110 | 65.05 | 56.23 | 3.82 | 10.79 |
| 111 | 62.79 | 58.83 | 3.51 | 8.92 |
| 113 | 67.55 | 67.46 | 4.03 | 5.50 |

Two separate errors, and the second is the visible one:

* the corner dip is REAL -- the reference has it too, bottoming at 62.8 -- but
  ours is **~10 units too dark** and arrives in one row where the reference takes
  four;
* the horizontal standard deviation at the junction row is **12.15 against the
  reference's 4.16**, three times the variation. That is the scalloping, and it
  is the part that reads as an artifact rather than as shading.

Further up the ceiling ours is 4-18 units too BRIGHT (row 88: 102.3 vs 87.4), so
the contact is over-sharpened in both directions, not uniformly dark.

**What this rules in.** M1 and M2 share the receiver-side culls
(`dot(dv, nP) <= u_plane_bias * r` and `cosP <= u_horizon`), the disc solid angle
`omega = pi r^2 cos / (d^2 + eps r^2)`, `bf_lout.comp`, and the bake. The
`u_plane_bias` sweep (0.1 and 3.0, MAE 0.067 and 0.33) and the `u_soft_eps` sweep
both come back nearly flat, which leaves the near-field disc form factor itself
and the bake's placement near a crease.

**The instrument this needs.** Every measurement above is on the Cornell render,
where the corner's true value is only known from a reference PNG at one camera.
The next step is a synthetic concave-corner gate -- two perpendicular planes, a
distant emitter, irradiance read back per surfel against the analytic
two-plane-corner solution -- so the error can be plotted against height above the
crease, in radii, with no reconstruction and no camera in the way. That is the
same move that turned the direct term's guesswork into findings 25 and 29, and it
should come before any more shader changes.


### 33. Cleanup: what was removed and what that costs

The brute-force stage is finished, so the experimental alternatives that were
kept alongside it while it was being decided have been removed. Nothing in the
shipped path changed: the direct term and the full-GI render are **bit-identical**
before and after (direct MAE 5.85, full GI 13.01, 28/28 gates).

**Removed**

| | why it existed | why it is gone |
|---|---|---|
| Point-sprite splat reconstruction (`SGI_RECON=0`, `surfel_splat.vert/frag`) | the original tier-0 reconstruction | superseded by the gather; findings 12 and 17 |
| Microbuffer reprojection (`SGI_RECON=2`, `mb_reproject.comp`, `mb_filter.comp`, the 61 MB microbuffer store, `SGI_MB_*`) | tier 2: interpolate the microbuffers, integrate after | finding 16 -- "built, verified, and not yet better" |
| `CoverageMode` 0 and 1 (`SGI_COV`) | the spec's literal winner-only model, and example 39's | kept only as a comparison target for a step 3 that this example is not going to build |
| The second microbuffer layer (`SGI_LAYERS=2`, `SGI_BACKGATE`, `lds_cov_back`, `lds_rad_back`) | finding 14 | off by default, untested by any gate, and the front/back split is not what the remaining artifact turned out to be (finding 32) |

About 1200 lines, four shaders, nine env vars and 61 MB of GPU memory. The
`nee_pixel` caveat goes with them -- there is only one reconstruction target now,
it holds `E` directly, and the per-pixel direct term composites into it
unconditionally.

**What that costs.** Findings 12, 14, 16 and 17 quote numbers that can no longer
be reproduced from this tree; they are kept as written, and this note is the
record that their code is gone. `Method::Radiance` (M1) was **not** removed
despite being an alternative -- gates 1, 2 and 5 use it as the exact reference for
M2, and finding 32 is the argument for keeping it: an estimator with no occlusion
at all is the fastest way to tell whether an artifact is an occlusion artifact.
`u_no_occlusion` (gate 5) and `u_debug_constant` (gate 3) stayed for the same
reason.

**Also tidied.** `SGI_SOFT` added (the `soft_eps` config field had an ImGui
control but no env var, so it could not be swept from a script); the `Splat`
timer renamed `Reconstruct`; the "Splat weight" view mode renamed to what it
actually shows now, the gather's fallback flag; `u_splat` renamed `u_recon` in
display.frag; and main.cpp's header now carries the env-var table.

### Finding 34 — the ceiling band, half of it: the plane bias is a slab, not a same-surface test

Findings 30–32 chased the band through the reconstruction, the microbuffer and
the density, and eliminated all three. Finding 32 recorded the correction and
left the cause open. It is in the transport kernel, and M1 has it too — which is
why every microbuffer knob came back flat.

**The instrument.** Cornell cannot measure a crease, because it has no analytic
answer anywhere. Gate 13 builds one: two unit squares meeting at a right angle
along a shared edge, one a uniform emitter, the other a pure receiver, nothing
else in the scene. The receiver's irradiance is then Lambert's contour integral
over the emitter rectangle — exact for a uniform-radiance polygon, not a
quadrature — so every surfel is checked against its own true value and binned by
**height above the crease in surfel radii**, which is the axis the defect lives
on. Both planes are tiled with the bake's own R2 sampler, for the reason gate 12
had to learn twice (finding 29).

**What it found.** `u_plane_bias` rejected any emitter whose centre lay within one
receiver radius of the receiver's tangent plane, whatever its orientation:

| M1, shipped config | 0-1 | 1-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64+ |
|---|---|---|---|---|---|---|---|---|
| before | -68.2% | -12.0% | -8.9% | -2.3% | -0.7% | -0.2% | -0.1% | -0.1% |
| after  | +7.2%  | -1.0%  | -3.8% | -0.9% | -0.3% | -0.1% | -0.1% | -0.1% |

The surfel row along the crease was 68% dark. It did not shrink with density
(8k → 32k moved it to -68.7%), and it scaled with the bias (at 2.0 the 1-2 row
went to -46%), which is what identifies a structural cull rather than a
quadrature artifact.

On a flat surface that slab is the intent — it stops a surfel lighting its own
coplanar neighbours. At a concave crease the perpendicular plane's near strip
lies inside the same slab, and that strip carries most of the irradiance because
its 1/d² is the smallest.

**The fix** is the predicate `nee.glsl` has used since finding 23, ported into
`bf_radiance.comp` and both phases of `bf_micro.comp`: reject only when the
normals AGREE *and* the centre is near the receiver's plane. A crease fails the
first test, so it survives; a flat surface passes both, so nothing changes there.
What now holds the 1/d² at contact is the disc softening, which is what it was
for — with the bias at zero and no softening the first bin reads +28443%.

**Cost.** Nothing. Direct MAE 5.85 and the full-GI bands are unchanged to the
digit, gate 8's closed-room openness bias *improved* from +25.6% to +22.4%, and
30/30 gates pass. On the render the band's shoulder rows (106-108, 112-116) went
from -1.6/-1.5/-0.5 and -1.6/-0.1/-0.5 to within ±0.9 of the reference. The
band's core rows (109-111) moved by about 1 unit and are still there: they are
finding 35.

### Finding 35 — the other half is summed coverage, clamped

Gate 13's M2 rows keep a deficit the M1 rows do not:

| 16x16, 8k | 0-1 | 1-2 | 2-4 | 4-8 | 8-16 | 16-32 | 32-64 | 64+ |
|---|---|---|---|---|---|---|---|---|
| M1 | +7.2% | -1.0% | -3.8% | -0.9% | -0.3% | -0.1% | -0.1% | -0.1% |
| M2 | -61.9% | -24.9% | -19.8% | -13.2% | -7.7% | -3.4% | -1.6% | -0.7% |
| M2, `fill` unclamped | -55.2% | -4.5% | -3.5% | -0.6% | -0.4% | -0.3% | -0.3% | -0.2% |

The third row is the whole diagnosis. `fill = clamp(cov_any, 0, 1)` is the only
thing between it and the second, and without the clamp the microbuffer agrees
with the analytic corner at every height.

**The clamp is not the bug.** Removing it inflates Cornell's converged solve by
31% (ratio 0.948 → 1.309, MAE 13.01 → 29.70), because in a closed room a bucket
straddling two walls genuinely sums past 1 and must not be allowed to amplify.

Ruled out by measurement, each leaving the number unmoved: the depth tolerance
(2, 8 and 10⁴ radii are byte-identical), the front/behind classification
(`fill = cov_front` is byte-identical to `cov_any`), bucket count, surfel
density, per-texel solid angle in both the deposit and its normalization, and
clipping the normalization sum at the square's rim. Occluder radius inflation —
finding 25's fix, ported to the gather — does seal the crease and grows every
distant silhouette with it (+15% at 64 radii), trading one error for a worse one.

Two experiments locate the rest:

* **Shape.** Coverage is deposited as a circle in *texel* space, and this map is
  anisotropic by 4.2x across the square at 16x16, worst at the rim — which is the
  horizon, which is where a crease puts the neighbouring plane. Replacing the
  kernel with 32 stratified samples of the disc pushed through the exact forward
  map took the worst bias beyond 4 radii from 13.2% to 9.5%.
* **Packing.** Discs of total area A cannot tile an area-A plane (finding 25), so
  at grazing incidence they overlap in some directions and leave gaps in others.
  Coverage per bucket is bimodal around a correct mean — which is precisely why
  the unclamped path is accurate and the clamped one is dark. Dilating the sample
  disc at constant energy, the variance-reduction form of radius inflation that
  does not grow the silhouette, took 4-8 radii from 9.5% to 5.4%.

Neither is kept. Both change the estimator everywhere to fix it in one place, and
neither helps the first two radii, which is the row the render's band is on.

**Where to start next.** Coverage has to become a UNION rather than a sum, the way
finding 14 made NEE visibility a bitmask rather than summed coverage. That is the
same lesson for the third time in this example: `OR` is idempotent and a sum is
not, and every repair built on top of a summed coverage estimate — finding 19's
calibration constant, `SGI_BACKGATE`, this clamp — has been a way of apologising
for it.


### Finding 36 — the ceiling band was the edge stop, not the estimator

Findings 34 and 35 were both real and both measurable in gate 13, and neither was
the band. The band was `SGI_LIGHT_SIGMA`, the `light_vis` edge stop of findings
19-20, still on at its default of 0.015.

That stop was built when the microbuffer carried the direct term. The cache then
held a hard shadow boundary, and the gather would happily interpolate light
across it, so the blend was keyed on visibility to stop it. Under per-pixel NEE
the cache holds the INDIRECT term only. There is no boundary left to preserve —
but the stop still fires wherever `light_vis` jumps, suppressing exactly the
neighbours the gather needs, and what it leaves behind is a row of surfels
reconstructed from too few samples.

Which is why it read as a *band* and not as an offset. At the ceiling junction,
reference camera at 512², columns 150-360:

| row | reference | stop on | stop off |
|---|---|---|---|
| 109 | 71.68, sd 4.16 | 56.37, **sd 12.21** | 61.11, **sd 4.40** |
| 110 | 65.05, sd 3.82 | 57.12, sd 10.89 | 61.56, sd 4.48 |
| 111 | 62.79, sd 3.51 | 59.80, sd 9.06 | 64.10, sd 4.86 |

Three times the reference's horizontal variation, gone. The residual level error
on row 109 is finding 35's microbuffer corner deficit, which gate 13 puts at
-24.9% one to two radii from a crease.

The default is now 0. The stop is still wired up, because `SGI_NEE_PIXEL=0` does
put the direct term back in the cache and there it is doing its original job.

**Method note.** Findings 30-32 chased this through the reconstruction and
concluded it was in the cache, on the evidence that it survived every gather knob
and showed up in `SGI_POINTS`. Every one of those sweeps moved a *gather* knob;
none of them moved the edge stop, which lives in the same pass. "Invariant to
everything I varied" is only as strong as the list, and the list had a hole in it.

### Finding 37 — the cut planes were skipped in the one case that needed them

`sgi_raster_ellipse_cut` clipped an occluder disc at its mesh edge by
intersecting the shadow ray with the occluder's own plane — and skipped that test
within ~6 degrees of edge-on, where `cnum / dn` runs away as `dn -> 0`. The
comment argued the safe direction to fail is keeping the bit, since an unclipped
bit over-occludes slightly and a dropped one leaks.

Slightly, except where edge-on is the rule. A receiver on the floor beside a wall
looks up at a panel overhead, and every one of those rays runs nearly parallel to
the wall — so no wall surfel near the junction was ever clipped at the junction,
and at `u_occ = 2.0` their discs reach two radii out across the floor. That is
the floor's dark fringe along every concave corner.

Near edge-on the well-conditioned point is the one on the ray closest to the
disc's centre, projected into the disc's plane. Both branches ask the same
question and agree where they overlap; the second is the one that survives
`dn -> 0`.

**Measured**, direct only at 512²: 2176 pixels change, every one of them brighter
(mean +5.3, max +38.4), all on the floor and lower walls — and on exactly those
pixels the error against the direct reference falls from **11.13 to 7.10**.
Whole-image direct MAE 5.85 -> 5.82, full GI 12.98 -> 12.94, 30/30 gates.

### Finding 38 — the penumbra's stair is the mask, and only the mask

Everything in the direct term is analytic — the per-bit cosines, the areas, the
distances — except the visibility mask, which is 8x8. So a penumbra is
reconstructed in steps of one bit, and one bit was 1/64 = 1.6% of the light.

Down a column of the tall box's lit face at 1600x900, 90 pixels of gradient:

| mask | distinct levels | mean step | max step |
|---|---|---|---|
| 8x8 | 44 | 0.84 | 1.93 |
| 16x16 | 66 | 0.56 | 1.00 |

The max step is what makes a stair visible, and it halves to 1.0 — the 8-bit
display quantum for luminance. The banding on the box face goes with it.

256 bits is four `uvec2` in a struct, addressed by a four-way branch on the word
index; an indexed local array would go to scratch memory and cost far more. Every
helper is written against `kBitsN`, so `kBitsEdge = 8u` is a one-line way back.

**Cost is the honest part: 2.8x on the direct pass, 243 ms -> 674 ms at 512².**
Almost all of it is the occluder rasterization, whose bit-space footprint grows
with the square of the edge — not the quadrature. Measured, not assumed:
short-circuiting the quadrature with a closed-form rectangle irradiance
(Lambert's contour integral, exact, four `acos` against 256 iterations) changed
the render by *nothing* — every band identical to three decimals — and cost 60 ms
more, because a warp holding one penumbra pixel runs the loop for all its lanes
anyway. That path was written, measured, and removed.


### Finding 39 — what the direct pass actually costs, and why tiling it is the wrong fix

The first plan for making the direct term realtime was to share one occluder
candidate list across an 8x8 screen tile -- section 4.1's per-cell reuse, in
screen space, and the pass already dispatches in 8x8 workgroups so the shape was
free. The measurement said don't.

Going from an 8x8 mask to 16x16 (finding 38) changed exactly one thing: the
bit-space area of every splat, by 4x. That is a controlled experiment, and
solving `243 = C + R`, `677 = C + 4R` splits the pass into

    rasterization  ~578 ms        march + projection + everything else  ~98 ms

at 512^2. Sharing candidates across a tile shrinks the 98 and *grows* the 578,
because a tile's candidate set is a superset of any one pixel's. The rasterizer
was the target all along.

**What worked: skip bits that are already set.** `OR` is idempotent, so a bit some
earlier occluder has already set cannot change, and testing it costs one mask
lookup against a slab test, a cut test and four plane dots. `sgi_raster_ellipse_cut`
now accumulates into the caller's running mask instead of building its own and
merging, so it can skip. **677 -> 333 ms, byte-identical output** -- both the
direct render and the full-GI render, every pixel.

**What did not: an early-out per cell.** The same idea one level up -- bail out of
the grid march as soon as the mask is full, per cell rather than per macro block
-- cost **2.3x** (333 -> 756 ms). Eight `bitCount`s in the innermost loop, plus a
`return` out of a triple-nested loop, is worth more than the march it saves.
Reverted. Both results are byte-identical to each other, so this is a pure cost
measurement with no image to weigh against it.

**Where the remaining 333 ms goes**, at 512^2 and 30k surfels:

| configuration | ms |
|---|---|
| shipped (`occ` 2.0, cuts on) | 336 |
| cuts off | 272 |
| `occ` 1.0 | 228 |
| `occ` 1.0, cuts off | 217 |

So the cut test is ~19% and the 2.0 occluder inflation ~32%, and neither is a
dominant term hiding the rest: ~217 ms is the irreducible march-project-slab-set
core. Cost against surfel count is sublinear -- 10k / 30k / 60k gives 211 / 331 /
525 ms -- because the grid holds candidates per unit volume roughly constant while
the discs get smaller, so bits-rasterized is closer to conserved than to linear.

**The verdict, which is the point of the exercise.** 333 ms over 262144 pixels is
**1.27 us per visibility query**. A 2 ms budget at 1600x900 is about 1.4 ns per
query. That is three orders of magnitude, against a 2x from the best available
micro-optimization. Per-pixel NEE, as a per-pixel grid march over disc occluders,
does not get there by tuning, and no amount of FMM work touches it -- the FMM
accelerates surfel-to-surfel transport, and this is neither.

What makes the same query affordable per SURFEL is not that it is cheaper. It is
identical, 1.27 us either way. It is that the surfel cache is temporally
amortized: 2048 receivers a frame is 2.6 ms, and a pixel cannot be amortized
because every pixel must be answered every frame. That is the whole difference,
and it is the argument for putting the direct term back in the cache -- where
spec section 8's budget assumed it was all along.

The cost of doing so is not energy. Per-surfel direct measures MAE 5.98 against
per-pixel's 5.85, which is nothing. It is silhouette sharpness: the cache is
Nyquist-limited to 19-37 px at this camera at 30k surfels (finding 24). Buying
that back means camera-distance LOD tiers -- section 6.1, step 9 in section 10's
order -- which stop being a performance tier and become the thing that makes the
direct term look right. That reordering is the real result here.


### Finding 40 — one entry per surfel, not one per cell; and three ways the rig was lying

**The change.** The grid inserts a surfel into every cell its bounding sphere
touches -- 11.9 cells per surfel here -- because a range query has to find every
surfel whose disc covers the query point, whichever cell that point is in. The
NEE shadow march is not a range query. It walks a volume, so it meets the same
surfel once per cell it occupies and pays a centre fetch, a normal fetch, a
same-surface test and an ellipse projection every time.

Measured per pixel at 512² before the change: **2152 cells visited, 6480 candidate
entries iterated, 420 entries projected.** Six thousand candidates against a scene
of thirty thousand surfels is a fifth of the scene per pixel -- but divided by
11.9 entries per surfel it is about 545 distinct surfels, each tested a dozen
times.

So the bake now marks the entry whose cell holds the surfel's centre
(`kCellOwner`, the top bit of `cell_item`; the bake caps indices at 65535 so the
bit is free, and an assertion checks that exactly one entry per surfel carries
it -- none makes a surfel invisible to the march, two counts it twice). The march
takes that entry and skips the rest for the cost of a 4-byte read.

**It is not byte-identical, and the attempt to make it so is the interesting
part.** Fat insertion's cull is "does ANY cell of this surfel's span pass the cone
test" -- a union-of-cells shape. No uniform pad on a centre-cell test reproduces
it, because a pad both admits surfels the old cull missed and drops ones it
caught. Tuning the pad made the difference WORSE rather than smaller: 102
differing pixels at `pad = r*u_occ`, 375 at `pad = h + sqrt(3)r`. That is the
signature of a different shape, not a wrong size, and it is what said to stop
tuning.

The honest replacement is to cull the SURFEL rather than its cell: pad the cell
tests enough that no owner cell holding a possible occluder is skipped, then test
the surfel's own inflated sphere against the cone. Tighter, exact, and it is the
test the cell version was approximating.

That change has a real consequence: insertion only ever knew the bake radius `r`,
so a surfel between `r` and `r * u_occ` of the cone was silently dropped -- **the
grid was quietly undoing part of the occluder inflation finding 29 added to make
surfaces seal.** Those come back.

**Result**, 512², warmed:

| | Direct/px | ratio | MAE |
|---|---|---|---|
| fat insertion | 133.5 - 135.0 ms | 1.057 | 5.81 |
| owner entry only | 104.8 - 107.4 ms | 1.057 | 5.81 |

**1.27x**, with every direct band identical to two decimals and every GI band
within 0.02. 147 direct pixels differ (0.056%), 5591 GI pixels (2.1%) at a
magnitude of a few units. 30/30 gates.

### The rig was lying in three ways, and all three are fixed

None of these were bugs in the renderer. All three produced numbers that looked
entirely plausible.

**1. The reference camera was steerable.** `camera_control` ran whatever `SGI_GTCAM`
said, so a stray mouse movement while the window came up rendered a different
view and every number taken from that shot was wrong. One measurement in this
session came out at MAE 14.69 against a true 5.81 and was very nearly believed.
`SGI_GTCAM != 0` now locks the camera; use `SGI_GTCAM=0` to fly.

**2. The framebuffer was not the size it asked for.** A window created at 512x512
comes back with a 640x640 framebuffer under a 125% desktop scale. Nothing
notices: the render is correct, it is simply not the size the reference image is.
`gllib::Window::set_framebuffer_size` now forces the exact pixel size by
measuring the compositor's scale and dividing it out, and `SGI_GTCAM` logs an
error rather than proceeding if it cannot get there.

**3. `SGI_BENCH=8` measures GPU clock ramp, not the shader.** The same
configuration reports 133 ms at `SGI_BENCH=40` and anywhere from 144 to 248 ms at
`SGI_BENCH=8`, run to run. **Every absolute millisecond figure in finding 39 was
taken at `SGI_BENCH=8` and is inflated by roughly 3x.** The ratios that finding
draws its conclusions from are unaffected -- both sides of each comparison were
cold -- but its headline number is not: the per-pixel visibility query is about
**400 ns warmed, not 1.27 us**, and the gap to a 2 ms budget at 1600x900 is about
**290x, not three orders of magnitude.** Use `SGI_BENCH=40` or higher for anything
quoted as a cost.


### Finding 41 — lever 2 was a wash, and the profile says why

**What was tried.** The cone from receiver to emitter had two loose bounds. Its
far radius was `|half_u| + |half_v| = 0.425` where the rectangle's circumradius is
`|half_u + half_v| = 0.302`, and the cone's cross-section is the square of that,
so the wide end -- which lands on the ceiling around the panel, and supplies most
of the candidates -- was twice the size it needed to be.

Tightening it measured 133.5 -> 63.6 ms and changed 18 pixels. Which meant it was
not conservative, and chasing that is the whole finding.

**Three attempts, and what each one taught.**

The far cap `t <= d_e + R` clips a rectangle corner that leans away from the
receiver, since the emitter's bounding sphere reaches `t = d_e + half_w`.
Extending it changed nothing: still 18 pixels.

The taper itself is the gap. `rad = half_w * t / d_e` assumes every point of the
rect sits at `t = d_e`, but a corner nearer than the centre needs the cone to
have opened to `half_w` by `t_X`, which is as small as `d_e - half_w`. Dividing
by `d_e - half_w` fixes it and **costs 156 ms against 81** -- because for a
receiver close to the light `d_e` approaches `half_w` and the cone degenerates
into a cylinder. Correct and unusable.

The shortfall is `s * half_w * (1 - t_X/d_e)`, largest at `s = 1`, and
`d_e - t_X <= half_w`, so a constant `half_w^2 / d_e` covers it with no
degenerate case. That is what shipped.

**And it buys nothing.** 81 ms, against 81 ms before lever 2 started. The tighter
radius is almost exactly cancelled by the margin it needs and the longer far cap.
The old bound's slack was doing real work; removing it and paying for correctness
separately lands in the same place.

What it does buy is that the cull is now provably conservative, where before it
could drop an occluder near a rectangle corner for an off-axis receiver. 394 of
262144 direct pixels change, MAE 5.81 either way, GI 12.93 -> 12.94, 30/30 gates.
A latent gap closed for free, and no speed.

**Where the time actually is.** Instrumented per pixel, after lever 1:

| | per pixel |
|---|---|
| macro-block bbox iterations | 945 |
| macro blocks occupied and in the cone | 251 |
| **inner cell-loop iterations** | **16064** (251 x 64) |
| cells that pass the cell cone test | 2683 |
| owner entries examined | 881 |
| occluders projected | 48 |

The macro loop is cheap -- 945 bit tests. The cell loop is not: every occupied
macro block runs its full 4x4x4, so **16064 sphere-vs-cone tests per pixel** to
find 2683 cells worth reading. Six of every seven are spent on cells that hold
nothing or lie outside the cone.

The fix is the one the macro level already uses, one level down: a 64-bit cell
occupancy mask per macro block (one `uvec2`, 32 KB for this grid), so the inner
loop iterates set bits instead of testing all 64. That replaces ~11000 frustum
tests per pixel with a bit scan, and it is the next thing to try -- not another
attempt at tightening the cone, which this finding says is finished.


### Finding 42 — the march was bound by reads, not by tests

Finding 41's profile said the cone march ran 16064 sphere-vs-cone tests per pixel
to find 2683 cells worth reading, and pointed at a cell-occupancy mask as the
fix. It was right about the fix and wrong about why it would pay.

**A 64-bit cell mask per macro block.** The macro level already stores one bit
per 4x4x4 block; this adds one bit per CELL inside it, a `uvec2` per block, 32 KB
for this grid. The inner loop iterates set bits with `findLSB` instead of running
the full 4x4x4. That is ~11000 fewer frustum tests per pixel and it bought
**80.6 -> 71.2 ms**, byte-identical. Twelve percent, for cutting two thirds of
the arithmetic the profile was counting -- which is how you find out the
arithmetic was not the constraint.

**Owners first.** The real cost was in the entry loop. Each visited cell holds
~4.3 entries and the march read every one of them to test a single bit, about
11500 four-byte reads per pixel to find 881 owners. So the bake now places a
cell's owner entries FIRST and packs the count of them into the high half of
`cell_sc.y` -- free, since the low half tops out at 9 here and the whole `uvec2`
is read anyway. The march loops to that count and touches nothing else.

**71.2 -> 52.3 ms.** Direct output byte-identical; the GI render differs by four
pixels at a magnitude of one, because reordering entries within a cell reorders a
float sum in the gather.

**Cumulative, 512^2, warmed, against the pre-optimization march:**

| | Direct/px | |
|---|---|---|
| fat insertion, all entries, full 4x4x4 | 133.5 ms | |
| owner dedup + exact sphere cull (finding 40) | 105 ms | |
| ...with the pad fixed to r_occ | 81.2 ms | byte-identical |
| ...+ conservative cone (finding 41) | 80.6 ms | 394 px, a correctness fix |
| ...+ per-block cell mask | 71.2 ms | byte-identical |
| ...+ owner entries contiguous | **52.3 ms** | 4 px in GI |

**2.55x**, MAE 5.81 direct and 12.94 full GI throughout, 30/30 gates. The frame is
now Direct/px 53.1, reconstruct 1.9, everything else 0.05.

**The lesson, twice in two findings.** Finding 41 counted tests and predicted a
win from removing them; the win was 12%. This finding counted reads and got 27%
from removing them. Both numbers came off the same instrument in the same pass.
An operation count is not a cost model, and on this hardware the difference
between the two is most of the answer.


### Finding 43 — ask the cache before marching the grid

The per-pixel march answers the same question for every pixel, and finding 39
measured that 87% of them have a locally flat visibility field: no occluder edge
crosses them, so the answer is "wholly lit" or "wholly dark" and the march is
being run to rediscover it.

The classification is already in the cache. `light_vis` is written per surfel by
the per-surfel NEE pass -- an energy-weighted fraction, noise-free by
construction -- and the gather already interpolates it. What the gather did not
publish is whether the neighbours AGREE, which is the only part that matters: a
mean of 0 and 1 looks exactly like a half-lit surfel.

So the gather now writes the **bracket** `[min, max]` of raw `light_vis` over
every surfel that fed the pixel, into its own RG16F target. Where `min` is 1 the
emitter is unoccluded and its irradiance has a closed form -- Lambert's contour
integral over the rectangle, four `acos` against a 256-sample quadrature. Where
`max` is 0 it is zero. Otherwise, march.

Two things it must not do, and both are in the code rather than in this note:

* **Normalize.** `lnorm` is `1/max` over the set, so if nothing in the scene is
  fully lit it scales a partly-lit surfel up to 1.0 and the agreement test reads
  a shadow as open sky. The bracket carries raw `light_vis`.
* **Trust an empty cache.** A paused or unstarted solve leaves `light_vis` all
  zeros, which reads as "every neighbour agrees the light is blocked" and renders
  a black room with a lit panel in it. That is exactly what the first run did.
  The running maximum is the signal that anything has been written at all, and
  the gather falls back to `[0, 1]` -- "no idea", march it -- without it.

**Measured**, 7 bounces, warmed:

| | Direct/px | GI MAE |
|---|---|---|
| 512², march every pixel | 48.7 ms | 12.94 |
| 512², skip on agreement | **33.5 ms** | 12.93 |
| 1600x900, march every pixel | 109.2 ms | |
| 1600x900, skip on agreement | **57.9 ms** | |

**1.45x at 512², 1.88x at 1600x900** -- more at higher resolution, because more of
the screen is flat. 3755 pixels change (1.43%), and on exactly those pixels the
error against the reference *falls*, 13.36 to 12.84. The difference map puts them
all on the ceiling around the panel, which is where `u_occ`'s inflation
over-occludes; nothing moves at the contact shadows. 30/30 gates.

**What it cannot survive** is an occluder whose entire shadow falls between
surfels, since then no neighbour disagrees and the pixel is declared lit. The
gather's radius is the margin -- every surfel within it must agree -- so the
shadow has to be finer than the cache's spacing across the whole neighbourhood.
Cornell has nothing like that. A scene with wires or foliage might.

**It is off by default** (`SGI_NEE_SKIP=0`). This example is the reference the
accelerated stages get asserted against, and its shipped image should be the one
the gates and the reference comparisons were run against, not a faster one that
depends on cache state. Turn it on for the realtime path.

**And it has a dynamic-scene hazard that is not yet tested.** The classification
is only as fresh as the cache, and the cache is amortized over ~15 frames. A
moving light or a moving occluder makes `light_vis` stale, and a stale bracket
does not degrade gracefully -- it says "no need to look" about a region whose
answer has changed. Any dynamic version needs the bracket invalidated wherever
the solve is, which is the same invalidation the transport needs and should share
it.

### Where the direct term stands

| | 512² | 1600x900 |
|---|---|---|
| start of this optimization pass | 133.5 ms | ~734 ms (extrapolated) |
| owner dedup, cell mask, owners contiguous | 52.3 ms | 109.2 ms |
| + cache agreement skip | 33.5 ms | 57.9 ms |

**4.0x at 512²**, and at 1600x900 the per-pixel visibility query is now **40 ns**
against the ~1.4 ns a 2 ms budget allows -- about **29x**, from ~290x when this
started. Still not realtime, and the remaining distance is unlikely to come from
culling: the per-pixel profile no longer has a dominant term. It comes from not
answering every pixel every frame, which is temporal reuse, which is the same
argument the cache already makes for the indirect term.


### Finding 44 — sizing the occlusion pass, which is not the FMM's job

**Read this one as a measurement of the OCCLUSION pass, not a criticism of the
FMM.** The architecture splits the two on purpose: the FMM carries unoccluded
transport to whatever needs it, and resolving occlusion is a separate, independent
stage. An earlier version of this finding read the split as the FMM breaking the
image, which is a misreading -- the FMM is doing exactly its job. What the numbers
below actually size is the job left over.

The microbuffer resolves geometry within the U-list: section 4.1 dispatches per
leaf cell at `h = 1.5s` and flattens the 27 surrounding cells into at most 256
candidates, so about +/-1.5 spacings -- **0.044 units in a 2-unit room**. Past
that, occlusion has to come from somewhere else. The question worth an hour is
how much there is to do.

**Simulating it needs no FMM code.** `SGI_NEAR` sets an occlusion horizon in
spacings: beyond it a surfel still lights but never blocks and never wins a
bucket's depth test. That is the near/far split, minus the SH approximation of
the far radiance. Full GI, 7 bounces, against the path-traced reference:

| occlusion horizon | ratio | MAE |
|---|---|---|
| unlimited (shipped) | 0.949 | 12.94 |
| 200 spacings | 0.949 | 12.94 |
| 25 spacings | 0.841 | 15.03 |
| 12 spacings | 0.838 | 15.17 |
| 6 spacings | 0.838 | 15.18 |
| 3 spacings | 0.838 | 15.18 |
| **1.5 spacings (the spec's h)** | **0.838** | **15.18** |
| 0.2 spacings | 0.838 | 15.18 |

The 200-spacing row reproducing the unlimited one exactly is what says the knob
is sound. The rest is the finding: **every horizon from 0.2 to 12 spacings gives
the identical image.** Nothing in that entire range occludes anything. The
occlusion that matters starts around 25 spacings and is all at room scale -- the
boxes and the walls, 0.5 to 2 units, 17 to 69 spacings.

Which is obvious in hindsight. Surfaces are locally flat and the same-surface
cull already removes a receiver's own neighbours, so there is nothing left within
a few spacings to block anything. The occluders are the other side of the room.

Visually the near-field-only render flattens: the boxes' faces brighten, the
contact darkening weakens, and the soft indirect shading that gives the image its
depth is largely gone.

**Fair caveat.** This simulation is cruder than section 5's prescribed blend. It
averages far radiance into one `L` per bucket, back-facing zeros included, rather
than keeping `Lnear` and `Lfar` separate and blending by coverage -- so the
magnitude of the error is not exactly what the FMM would produce. The structural
conclusion does not depend on that: in a closed room the near coverage within 1.5
spacings is essentially zero, so section 5's `(1 - cov)` is essentially 1 and the
far field arrives unmasked either way. The box casts no indirect shadow.

**Enlarging the U-list is not the fix.** It spans +/-1.5 spacings at 27 cells;
reaching 25 spacings means +/-16.7 cells, 33^3 = 36000 of them, against a `MAXK`
of 256.

**The spec half-knows this.** Section 5.1 warns that a sealed room "receives the
full outdoor sky through its walls, masked only by whatever happens to sit in the
receiver's own 27-cell U-list", and prescribes a per-cell sky visibility scalar
for it. The measurement says the same hole applies to *interior* light, not just
sky, and a per-cell scalar will not close that one -- interior occlusion is
directional.

**What this sizes.** The occlusion pass has to cover 0.838 -> 0.949 in ratio and
15.18 -> 12.94 in MAE, and it has to do it at room scale rather than near scale,
because that is where all of it is. Nothing it does inside 12 spacings will
register.

The structure to build it on is already in the tree: the grid's 4x4x4 macro
occupancy bitmask is a low-resolution binary voxelization of the scene. Marched
per bucket -- a handful of bit tests along each bucket's direction -- it gives a
far-field visibility scalar at a cost that does not grow with surfel count, which
is the property the FMM has for transport and the microbuffer does not have for
visibility.

Worth building and validating BEFORE the FMM, for a reason that has nothing to do
with the FMM being at fault: the near/far blend of section 5 is the seam the two
meet at, and it is easier to get that seam right against an exact far field
(today's all-pairs) than against an SH approximation of one. Once the occlusion
pass holds the image at `SGI_NEAR=1.5`, the far field can become an expansion
without the two changes being confounded.


### Finding 45 — the far field needs cell radiance, not surfel radiance

Building the occlusion pass finding 44 sized. Three pieces landed, in order, each
one measured.

**1. Section 5's blend, properly separated.** `lds_cov_far` / `lds_rad_far` hold
the far field apart from the near, and the integrate is
`Lin = L * fill + Efar * (1 - fill)`. With no horizon set nothing is far and this
reduces to `mix(u_sky, L, fill)` exactly -- byte-identical to the shipped
reference, which is the assertion this rests on.

Two errors on the way, both worth keeping:

* **The far share is not composited by its own coverage.** `cov_far * (1 - fill)`
  leaves partially-covered buckets dark: ratio 0.533 against 0.838. The far field
  is a directional estimate for the rest of the bucket, not a second layer.
* **The far field is a SUMMED irradiance, not an averaged radiance.**
  `rad_far / cov_far` is the mean radiance of everything in the bucket -- near
  wall, far wall and back faces alike -- and it reads far darker than the truth.
  The FMM expands radiant *intensity* for exactly this reason: it sums flux, it
  does not average radiance.

And one bug that looked like neither: the far branch sat *after* the winner
block, which bails on an empty bucket -- and a bucket whose only geometry is far
IS empty, because phase A skips far candidates. Every far candidate was dropped.
The symptom was three different far-field formulas producing bit-identical black
rooms, which is what said the branch was never running at all.

With the far field working and nothing occluding it, `SGI_NEAR=1.5` gives ratio
**1.043**, against 0.949 for exact all-pairs -- too bright, dark bands twice the
reference. That is the hole, now measured from the other side.

**2. The coarse march.** `sgi_far_horizon` DDAs the grid's macro occupancy
bitmask -- one bit per 4x4x4 cell block, ~2 spacings -- from the near horizon
outwards, per bucket, and reports the distance to the first occupied block. Cost
is a DDA per bucket and does not grow with surfel count.

It needs the ray lifted a full macro block off the receiver's own plane. A block
(0.059 units) is WIDER than the near horizon (0.044), so a march started at the
receiver hits the block its own surface occupies on the first test and reports
everything as blocked: a black room. Lifted, a grazing ray travels above its own
plane's blocks and gains height as it goes.

**3. And it does not work with a distance window.** Culling far candidates at
`t_first` throws away the very surface the march just found, because the march
returns where the block STARTS and the surfels are inside it. A slack window past
the first hit trades one error against the other:

| slack, in macro blocks | ratio | MAE | band 5-15 |
|---|---|---|---|
| 1.5 | 0.707 | 21.66 | 0.55 |
| 3 | 0.830 | 16.03 | 2.77 |
| 6 | 0.869 | 14.66 | 5.80 |
| 1000 (no far occlusion) | 1.043 | 12.92 | 21.53 |
| **exact all-pairs** | **0.949** | **12.94** | **10.62** |

The 1000-block row reproducing the un-occluded result exactly is what says the
plumbing is right. The rest says no value of it is correct: a surface seen at a
grazing angle spans many blocks along the ray, so any window wide enough to keep
it also admits the surface behind it. This is the same lesson the microbuffer
already learned -- "the winner's SUPPORTING PLANE, not its depth" -- and the same
fix is not available here, because the far field is supposed to be cheap and a
per-candidate plane test is the all-pairs cost again.

Note also that MAE is the wrong instrument for this one: it reads 12.92 for the
un-occluded far field against 12.94 for exact, while the 5-15 band is twice the
reference. The bands are what to watch.

**What the measurement actually says.** The mismatch is not in the occlusion, it
is in the pairing: far *radiance* still comes from individual surfels while far
*occlusion* comes from voxels, and the two disagree about what a surface is.

In the FMM they do not. The far field's radiance comes from CELLS, so a march
that stops at the first occupied cell takes that cell's aggregate and there is no
window to tune -- the cell you hit is the surface you see. That is the version to
build: a per-block mean outgoing radiance, one cheap reduction over surfels per
sweep, sampled at the march's first hit. It is an order-0 multipole, it deletes
the far candidate loop entirely rather than making it cheaper, and it is the
first real piece of the FMM rather than a stand-in for it.


### Finding 46 — an order-0 multipole, and the far field stops needing a tuning knob

Finding 45 ended on a diagnosis: far *radiance* came from individual surfels
while far *occlusion* came from voxels, the two disagreed about where a surface
ends, and no depth window reconciled them. The fix is to make both come from the
same place.

`blk_rad.comp` reduces `lout` into one mean outgoing radiance per 4x4x4 macro
block -- an order-0 multipole -- rebuilt once per sweep alongside `lout`, one
pass over the surfels, fixed point because core GLSL has no float atomicAdd. The
per-bucket march now returns the **block** it stopped at rather than a distance,
and that block's radiance is the far field for that bucket. There is nothing left
to tune: the cell you hit is the surface you see. `SGI_FAR_SLACK` is gone.

The far candidate loop in phase B is gone with it. A surfel past the horizon
contributes nothing there -- its radiance reaches the receiver through its block.

**At the FMM's own U-list horizon, `SGI_NEAR=1.5`:**

| far-field model | ratio | MAE | band 5-15 |
|---|---|---|---|
| per-surfel, unoccluded | 1.043 | 12.92 | 21.53 |
| per-surfel + best distance window | 0.869 | 14.66 | 5.80 |
| **order-0 block radiance at the first hit** | **0.926** | **14.57** | **5.85** |
| exact all-pairs | 0.949 | 12.94 | 10.62 |

and the error keeps closing as the near field grows -- at `SGI_NEAR=25` it is
ratio 0.928, MAE 13.36, band 10.43 against 10.62.

Visually the indirect shadows, the colour bleeding and the contact darkening all
survive, which none of the earlier far-field models managed. `SGI_NEAR=0` remains
byte-identical to the brute-force reference, and 30/30 gates pass.

**And it is faster: the solve goes 34.1 -> 20.3 ms per 2048-receiver slice**,
with the all-pairs loop still *visiting* every far candidate and merely exiting
early on it. The structural win -- visiting only the U-list -- is spec section 10
step 3, and it now has somewhere for the rest of the light to come from, which is
the thing it did not have before.

**What is left, and it is the order.** Shadows at `SGI_NEAR=1.5` sit at 5.85
against 10.62: too DARK, where the unoccluded model was too bright. Two suspects,
both testable. A block reports one radiance whichever side you look at it from,
so a block holding two faces of a corner averages them -- section 1.2's order-2
`Ilm[9]` exists for exactly this. And the march stops at any block the ray
touches, including one it only clips the corner of, which over-occludes; a cone
rather than a ray, or a finer block, would soften it.

Neither is a knob. Both are the next increment of the same structure, which is
what says this is the right structure.


### Finding 47 — step 3: the U-list walk, and it is exact

Spec section 10 step 3: near field through the grid, everything else through the
far field. With finding 46's order-0 multipole carrying the far field, the near
field can finally stop being "all N candidates, most of which exit early".

The walk gathers the U-list once into LDS and both phases read it -- owner
entries only, cell span dilated by a cell plus a radius so a surfel whose centre
cell falls just outside still gets in, and the `d2 > near2` test still deciding
membership. Overflowing the list would silently drop occluders, so it does not:
the walk raises a flag and both phases fall back to the all-pairs stride, which
is slow and correct, and keeps large `SGI_NEAR` values usable as a diagnostic
even when the U-list they imply is far too big to gather.

**It is byte-identical at every horizon** -- 0, 1.5 and 6 spacings, zero pixels
differing against the all-pairs near field. Same candidate set, reached a
different way, which is the assertion step 3 exists for. 30/30 gates.

**Solve, per 2048-receiver slice:**

| | all-pairs | U-list | |
|---|---|---|---|
| 10k surfels | 11.54 ms | 1.13 ms | 10.2x |
| 30k surfels | 34.24 ms | 2.34 ms | **14.6x** |
| 60k surfels | 72.87 ms | 5.24 ms | 13.9x |

At 30k that is **~34 ms a bounce against ~500 ms**.

**The far march is not the cost.** Disabling it leaves 0.90 / 1.92 / 4.61 ms
against 1.13 / 2.34 / 5.24, so the per-bucket DDA is 0.2-0.6 ms -- a fifth of the
pass at most, and the part that does not grow with surfel count.

**What does grow is the walk, and section 4.1 already says why.** The U-list is a
constant ~30 surfels at 1.5 spacings whatever N is, and the cell span is a
constant 7^3 -- but every receiver re-reads those 343 `cell_sc` entries for
itself, out of a grid that gets larger with N, so the cache hit rate falls away.
That is exactly the redundant fetch section 4.1 describes: "every receiver in a
cell has the same 27-cell U-list and the same K candidates", re-probed and
re-fetched per receiver.

Which makes step 4 -- one workgroup per occupied cell rather than per receiver,
with the U-list gathered once and shared -- not just the performance step the
spec calls it but the fix for the one number here that does not scale. And it
comes with its own assertion: bit-identical against step 3, which is now the
thing that has to hold.


## Gate results

`SGI_GATE=all SGI_NOGUI=1 ./40_surfel_fmm` — 30 assertions, all pass.

| # | Gate | Assertion | Measured | Tol |
|---|---|---|---|---|
| 6 | bake | `Σπr² / A == 1` | rel 1.1e-7 | 1e-4 |
| 1 | p2p (M1) | `E == a·L·cosE·cosP/r²` | rel 3.0e-8 | 1e-5 |
| 1 | p2p (M2) | same, through the microbuffer | rel 3.7e-3 | 2e-2 |
| 1 | p2p | the emitter receives exactly 0 | 0 | — |
| 2 | disc (M1) | coaxial disc form factor, x = 0.5/1/2 | rel 4e-7 / 9e-8 / 4e-8 | 1e-2 |
| 2 | disc (M2) | same | rel 3.6e-2 / 2.0e-2 / 1.1e-2 | 5e-2 |
| 3 | bucket 8×8, 16×16 | `Σdw == 2π`, `Σwcos == π` | rel ~5e-9 | 1e-5 |
| 3 | hemi | constant radiance L gives `E == π·L` | rel 2.8e-8 | 2e-3 |
| 3 | hemi | linearity in L | exact | 1e-6 |
| 4 | scale | 8× emission gives 8× E, at 1 and 8 sweeps | rel 9.7e-8 / 1.6e-5 | 1e-3 |
| 5 | noocc | M1 vs M2 with every candidate forced visible | rel 1.3e-2 | 2e-2 |
| 8 | occ | cosine-weighted openness vs a CPU ray cast, mean abs | 0.042 | 12/√N = 0.069 |
| 8 | occ | same, bias in the deeply-occluded band | 0.053 | 16/√N = 0.092 |
| 9 | opaque | a lit opaque blocker is lit at all | 0.220 | — |
| 9 | opaque | light leaking through it to a receiver behind | 0.000 | 2e-2 |

Gate 3 is the one that catches a `/PI`, and it is the reason the whole set exists:
per §9, a `PI` error is consistent across the near and far paths, so it never
shows up as an internal discrepancy — only against an analytic answer. Gate 8 is
the one that catches an occlusion bug, and it exists because gates 1–5 are all
deliberately *unoccluded* and so could not (finding 4). Gate 9 is the one that
catches light passing *through* a surface, which gate 8 cannot: a uniform-sky
openness test is a pure visibility question and says nothing about which face of
a surfel emits (finding 10).

Gate 5's worst individual surfel is off by 72% while the population is off by
1.3%. That is bucket quantization on a receiver whose hemisphere is dominated by
a handful of near neighbours, and it is the expected signature rather than a
defect; the tolerance is on the population for that reason.

## Reference comparison

Camera, derived from the reference and then refined: `eye (0.004, 0.999, 3.864)`,
`target (0.004, 0.999, 0)`, `fov_y 38.75°`, 512×512. `SGI_GTCAM=1` sets all four
and sizes the framebuffer to match, so the split and diff views are
pixel-aligned. The scene is **not** fit-normalized (example 39 normalizes only
because it must also handle Sponza), so these are native glb units: floor y = 0,
ceiling y = 1.99, back wall z = −1.04, opening toward +z, light panel at y = 1.98.

Convergence, N = 30k, M2, 16×16, `SGI_SKY=0`, interior mean in sRGB:

| sweeps | mean | Δ (R) |
|---|---|---|
| 1 (direct) | (54.1, 38.1, 12.2) | — |
| 2 | (66.8, 47.0, 14.3) | +23.4% |
| 4 | (77.4, 54.1, 16.0) | +15.8% |
| 8 | (80.4, 56.0, 16.2) | +3.9% |
| 16 | (80.7, 56.1, 16.2) | +0.3% |
| 32 | (80.7, 56.1, 16.2) | +0.0% |

Converged by 16 sweeps, so the residual against the reference is *not* a
convergence artifact. `SGI_BOUNCES=16` is the honest "converged" setting.

Against the matching reference (interior mean):

| | vs reference (R, G) | MAE |
|---|---|---|
| M1, 1 sweep, vs direct | 1.14×, 1.09× | 10.22 |
| M2, 1 sweep, vs direct | 0.84×, 0.80× | 11.18 |
| M1, 8 sweeps + sky, vs full | 1.40×, 1.35× | 30.62 |
| M2, 8 sweeps + sky, vs full | **0.97×, 0.95×** | **9.27** |

**M1 (no occlusion)** renders with no shadow under either box at all and the
whole room washed out. M1 overshooting and M2 undershooting by comparable
amounts, with M1's error growing with bounce count (1.14× → 1.40×) while M2's
does not, is the expected signature: unoccluded transport compounds over bounces.
That is the visual confirmation that occlusion is the only thing M2 adds.

Bucket resolution and surfel count, direct reference, one sweep, interior mean:

| variant | mean | ratio (R) | MAE |
|---|---|---|---|
| 8×8 = 64 buckets | (59.0, 42.3, 13.6) | 0.91 | 9.76/255 |
| 16×16 = 256 buckets | (54.1, 38.1, 12.2) | 0.84 | 11.18/255 |
| N = 10000 | (53.0, 37.1, 11.8) | 0.82 | 10.81/255 |
| N = 30000 | (54.1, 38.1, 12.2) | 0.84 | 11.18/255 |
| N = 60000 | (56.3, 39.9, 13.0) | 0.87 | 11.85/255 |

Both sweeps move in the direction finding 4's sampling floor predicts, and that
is the useful part: the image gets **brighter** with more surfels (0.82 → 0.87)
and with coarser buckets (0.84 → 0.91), because both raise surfels per texel,
which lowers the coverage-clamping leak, which stops throwing away light. The
remaining direct-image deficit is therefore partly that same sampling floor and
partly the tone curve of finding 9 — not a transport error.

64 buckets landing *closer* to the reference than 256 is real but is not an
argument for 64: it is two errors of opposite sign partly cancelling (coarser
quantization against a smaller sampling leak). 256 is the more faithful
microbuffer, which is what this reference is for.

## Timing

RTX 3060 mobile, `SGI_BENCH`, one full sweep per frame (`SGI_BUDGET = N`):

| configuration | ms / sweep |
|---|---|
| M1, N = 30k | 18.2 |
| M2 8×8, N = 30k | 217 |
| M2 16×16, N = 10k | 58.7 |
| M2 16×16, N = 30k | 476 |
| M2 16×16, N = 60k | 1002 |

M2 is 12× (64 buckets) to 26× (256 buckets) the cost of M1 over the identical
9·10⁸ candidate pairs, which is §4.2's prediction that the atomics and the
footprint loops dominate rather than the ALU. Scaling is mildly sublinear in N
because the bake's radius shrinks as `1/sqrt(N)`, so each splat covers fewer
texels.

Interactive default (N = 30k, budget 2048, M2 16×16): 34.5 ms/frame p50, of which
Solve 31.3, Splat 1.1, G-buffer 0.03, Display 0.03. ~15 frames per sweep.

Finding 3's correct acceptance test costs ~16% over the plane test it replaced
(410 → 476 ms per sweep at 16×16): one ray/plane intersection per candidate-texel
and a second `atomicAdd` for the opacity accumulator. Cheap for what it buys, and
the atomics were already the bottleneck rather than the ALU.

## What is deliberately absent

No sparse grid, no U/V interaction lists, no P2M/M2M/M2L/L2L, no `Lsh` trilinear
interpolation, no temporal amortization, no LOD, no RGB9E5 irradiance, no
per-frame tangent-frame jitter by default. Every one of those is spec §10 step 3
or later, and every one of them is meant to be *asserted against what this
example produces*. Adding any of them here would remove the thing they are
supposed to be checked against.

## Open

- The ceiling band around the light panel. **Largely addressed by finding 14's
  second layer**: over rows 95-165 its error against the converged reference goes
  from +5.2% to -2.3% and its local sd from 3.71 to 3.09. What remains is the
  residual undershoot. Historical note, from before that: ~25% bright, 63.1
  against the reference's 50.0. Four hypotheses eliminated — not tangent-frame
  jitter, not the plane bias, not density (more surfels makes it worse), and not
  back-face emission (finding 10's fix leaves it at 63.1 from 63.2). The
  near-field geometry of a nearly coplanar emitter is the remaining suspect: the
  panel sits 0.01 below the ceiling against a surfel radius of 0.0164.
- AgX is not ported, so the comparison against both references carries a known
  tone-curve offset that shows up almost entirely in blue (finding 9).
- The second microbuffer layer holds ONE bucket behind (finding 14), i.e. the
  average of everything past the front surface. Exact layering needs a second
  `atomicMin` pass for layer 2's own winner and plane. Not yet justified by the
  numbers on this scene, where depth complexity along a cone is 2 at most.
- `SGI_BACKGATE` (default 0.5) calibrates the front layer's transmittance against
  a coverage estimator that gate 8 measures as 25.6% too open. It is a real
  trade: every image metric improves with it and gate 9's leak grows with it
  (finding 14). It should stop being needed if the coverage estimator's bias is
  ever fixed at source, which is finding 4's territory.
- Splitting direct light out of the cache (finding 12) is the architectural way
  to sharpen the image without paying O(N^2) for surfels. Out of scope here by
  design, and the natural thing to revisit once the grid lands.

# Renders

All at the reference camera (`MBG_GTCAM=1`, 512×512, pixel-aligned against the
two path-traced PNGs one directory up), default configuration unless noted:
3 camera levels (the cap), GI grid 128×128, targets 16/8/8, the single-sample
path estimator at 20 paths with roulette at 0.15, analytic direct term with an
8×8 light view per pixel, jitter + 2 a-trous iterations, Reinhard.

One complete sweep is **6.7e5 cameras / 4.6e7 texels**, split into 4 chunks. On
llvmpipe (software GL, 4 CPU threads, shared container) that took **14 s**; that
number is a property of this machine and not of the method, and the counts beside
it are what to scale with.

Those two are what the schedule ASKS for. What the sweep actually does, from
`MBG_COUNT=1`, is **5.34e5 live cameras** (a fifth of the slots are background
pixels or escaped paths, which write their zeros and return), **3.73e7 texels**,
and **9.92e8 triangle fetches** — 26.6 of Cornell's 32 triangles per texel, the
rest cut by the any-hit early-out. The sweep line also prints a
"triangle-rasters" figure: ignore it. It is `cameras × scene.count()` computed on
the host, it never multiplies by the texels a camera rasterizes, and on this
scene it is low by a factor of 46. See implementation.md finding 27.

| File | What | Compare against |
|---|---|---|
| `01_direct_1bounce.png` | `MBG_BOUNCES=1 MBG_SKY=0` — direct lighting only, no free parameters at all. RMSE 0.0430 | `../CornellBoxGroundTruthDirectLighting.png` |
| `02_gi_3bounce.png` | `MBG_SKY=0.05` — the full solve. RMSE 0.0402 | `../CornellBoxOriginalGroundTruth.png` |
| `03_indirect_only_3bounce_4xexposure.png` | `MBG_INDIRECT_ONLY=1 MBG_EXPOSURE=4` — the bounce term with the direct term left out, exposed 4× so it is visible on its own | nothing; it is a diagnostic |
| `04_diff_direct_vs_reference.png` | `MBG_VIEW=11`, 4× gain. Blue = agreement, warm = error | — |
| `05_diff_gi_vs_reference.png` | the same for the full solve. The only warm region left is the emitter panel, which is the tone-curve gap of finding 10, not transport | — |
| `06_daylight_3bounce.png` | `MBG_DAYLIGHT=1` — a sun and a sky dome through the box's open +z side, on top of the panel. The hard edge across the tall box is the ceiling's leading edge cutting the beam | nothing; the references are of a closed box |
| `07_daylight_indirect_only_4xexposure.png` | `MBG_DAYLIGHT=1 MBG_INDIRECT_ONLY=1 MBG_EXPOSURE=4` — the same scene with the sun's direct term removed, so what is left is the sky's hemisphere integral plus three bounces | — |
| `08_bunny_3bounce.png` | `MBG_MODEL=CornellBoxBunnyMirror.glb MBG_SKY=0.05` — the only render here that is **not** 32 triangles. 69483 of them in 1086 clusters, which is the one thing the other seven cannot test | nothing; there is no path-traced reference for this scene |

## Why there is a bunny

Every other image here is the 32-triangle Cornell box, and 32 triangles is a
single cluster — below the eight-cluster floor, so it takes the uncooperative
traversal and never runs the cluster levels at all. That made the whole render
set structurally blind to the traversal: finding 28 (the distance bound switched
off on 40 of every 41 cameras) and finding 30 (the angular cull) are both
invisible in images 01–07, and would stay invisible however many of them were
added. `08` is 69483 triangles in 1086 clusters and exercises both.

What it measures, at the default configuration:

| | |
|---|---|
| cameras | 5.14e5 live of 6.72e5 scheduled |
| texels | 3.60e7 |
| per camera-thread | 17 group tests → 8.8 entered, 481 cluster tests → **3.5 entered** |
| per texel | **196 triangles fetched** of 69483 in the scene — 0.28% |
| angular-cull violations | **0** (finding 30's assertion; it reads 1 at some other configurations) |

Three and a half clusters entered out of 481 tested is findings 28 and 30
together, and it is why this render takes 318 s on llvmpipe rather than the hours
it would have taken a week ago.

It is also the clearest demonstration that the sweep line's "triangle-rasters" is
not a work figure. Here it claims 4.67e10 against a measured 7.06e9 — **6.6×
too high**, where on Cornell the same formula is 46× too *low*. It is
`cameras × scene.count()`: it misses the texel multiplier, which makes it low,
and it ignores the cull, which makes it high, and which of the two wins is a
property of the scene.

**The tall box is black on purpose.** It carries a mirror material, and this
renderer is Lambertian everywhere — `scene.hpp` resolves a material to one
albedo and one emission — so a mirror's base colour is what it reflects, which
is nothing. The bunny is the diffuse half of the same file and is the subject.

Reproduce any of them with, e.g.:

```bash
cd build-release/examples/41_multibounce
MBG_NOGUI=1 MBG_GTCAM=1 MBG_SKY=0.05 MBG_SOLVE=1 MBG_BENCH=1 \
  MBG_SHOT=02_gi_3bounce.png ./41_multibounce

MBG_NOGUI=1 MBG_GTCAM=1 MBG_DAYLIGHT=1 MBG_SOLVE=1 MBG_BENCH=1 \
  MBG_SHOT=06_daylight_3bounce.png ./41_multibounce

MBG_NOGUI=1 MBG_GTCAM=1 MBG_MODEL=../../../data/CornellBoxBunnyMirror.glb \
  MBG_SKY=0.05 MBG_SOLVE=1 MBG_BENCH=1 MBG_COUNT=1 \
  MBG_SHOT=08_bunny_3bounce.png ./41_multibounce
```

`MBG_NOGUI=1` matters: the ImGui overlay is otherwise in the screenshot.
`MBG_SOLVE=1` completes a sweep before the first present and `MBG_BENCH=1` then
exits after one frame; without `MBG_BENCH` the app holds the solve and runs its
interactive loop forever, screenshot or not. `01` and `04` compare against the
direct reference, which is `MBG_REF=0`.

**`MBG_BENCH=1` is kept, and the reason has now been narrowed to llvmpipe.**
Finding 29 recorded that with the denoiser on, a run of more than one frame did
not reproduce -- four runs of one binary at `MBG_BENCH=3` gave four distinct
images. That was measured in a software-GL container and does NOT happen on
hardware: three runs at `MBG_BENCH=3` on an RTX 3060 are bit-identical. The
one-frame form is kept because it reproduces everywhere.

**These images are generated on an RTX 3060 (driver 610.57.04), not on
llvmpipe.** The same binary on the two produces images differing by up to 31/255
across 191420 of the 262144 pixels, so a render made on one will not compare
byte-for-byte against the other. They reproduce bit-exactly on the hardware
named here, which is the machine this example is developed on.

The daylight images carry no `MBG_EXPOSURE`: the `MBG_DAYLIGHT` preset is scaled
so that the **three-bounce** solve lands in the tone curve's usable range at
exposure 1. A white box interreflects, so a sun tuned to read at one bounce
pushes every surface into Reinhard's shoulder by the third and flattens the very
shadow it was set up to show.

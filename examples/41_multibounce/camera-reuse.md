# Camera reuse: how much of this renderer's work is redundant, and how much of that is recoverable

Every camera in this example walks the scene alone. `mbg_cull`'s only
hemisphere-wide constraint is the receiver's own tangent plane, and the comment
that says so (`shaders/common/raster.glsl:59-79`) also says what that is worth:
standing on Sponza's floor, it "keeps most of the building". Two cameras four
millimetres apart therefore build nearly the same cluster mask, walk nearly the
same clusters, and resolve nearly the same visibility — and the code shares none
of it.

This document is about whether that is recoverable. The short answer, measured
below, is: **about a factor of two, and the limit is radiometric rather than
geometric.** The redundancy is real and enormous, and almost all of it is
load-bearing.

---

## 1. The redundancy, quantified

Cornell+bunny at the default, from `MBG_COUNT=1`:

| | |
|---|---|
| live cameras per sweep | 5.14e5 of 6.72e5 scheduled |
| texels rasterized | 3.60e7 |
| triangles fetched per texel | 92.15 |
| **triangle fetches per sweep** | **~3.3e9** |
| triangles in the scene | 69 483 |

So the sweep fetches each triangle, on average, about **47 000 times**. A single
camera touches roughly 9% of the scene, and there are half a million of them.

The shape of the waste is worse than the ratio suggests, because of what a
camera keeps. A level-1 camera rasterizes a 16x16 hemisphere — 256 texels — and
its entire output is 48 bytes: one irradiance triple and two direct records
(`shaders/common/raster_body.glsl:329-343`). The per-texel visibility buffer
never leaves shared memory; `vis_out` is written only under `u_dump`, which only
the gates set. **The expensive thing is discarded by construction**, at roughly
64:1, and then the neighbouring camera computes it again.

Findings 38, 41 and 42 closed the per-kernel levers: this kernel sits at a
pressure limit where anything *added* costs more than a better algorithm saves,
and every win came from subtraction. Redundancy across cameras is the one large
inefficiency left that is a *scheduling* problem rather than a per-kernel one, so
it is not behind that wall.

---

## 2. Two questions, and they have different answers

**How many cameras are needed?** Radiometric. This is the design doc's §8.1
reuse-radius experiment, which `implementation.md`'s "What this does not answer"
names, orders first (milestone 2 of 4), and records as undone. `place.comp:6-19`
states that this example is deliberately "the version with reuse radius zero" —
the thing §8.1 measures *against*.

**How should the cameras that remain share geometry?** Geometric. Parallax,
shared visibility structures, batched traversal.

They are not the same question and the first one dominates, because a camera you
do not create costs nothing at all. Section 3 answers it. Section 4 shows that
once it is answered, the second question has much less left to win than the
fetch ratio in section 1 implies.

---

## 3. §8.1, measured: the reuse radius is not generous

`MBG_SCALE` is the reuse radius in its cheapest form — the GI grid is
`framebuffer / scale`, so scale 8 has a quarter the cameras of scale 4. Cornell,
three bounces, warm readings, against `CornellBoxOriginalGroundTruth.png`:

| `MBG_SCALE` | cameras | sweep | RMSE | roughness (mean) |
|---|---|---|---|---|
| 4 (default) | 16384 | 472.7 ms | 0.0402 | 0.540 |
| 6 | 7396 | 214.5 ms | **0.0374** | 0.525 |
| 8 | 4096 | 121.6 ms | 0.0382 | 0.484 |
| 12 | 1849 | 57.0 ms | 0.0494 | 0.954 |
| 16 | 1024 | **35.4 ms** | 0.0396 | 0.511 |

Read that table and the answer looks like 16: **thirteen times faster with a
better RMSE than the default.** Finding 3 said much the same thing in its day
("one camera per 8x8 block scores the same as one per 4x4 ... for a quarter of
the cameras"), and finding 21 is the record of what happens when this table is
believed.

### What the table cannot see

The composite image is dominated by the direct term, which is computed per pixel
(finding 11) and does not touch the grid at all. So RMSE against the full
reference is mostly scoring a term the grid does not affect. Isolate what the
grid actually computes — `MBG_INDIRECT_ONLY=1 MBG_EXPOSURE=4` — and the three
densities are plainly different:

| indirect term only, vs scale 4 | max | mean |
|---|---|---|
| scale 6 | 109 | 1.99 |
| scale 8 | 110 | 2.29 |
| scale 16 | 118 | 3.47 |

and the character of the difference is the point, not its size. At scale 4 the
residual is fine-grained mottling. At scale 6 it is slightly softer and still
fine. By scale 8 the green bleed on the back wall has broken into visible coarse
blobs, and at 16 it is **large, soft, correlated patches** — including a green
bleed on the tall box that is not in the scale-4 image at all. On the bunny the
colour bleeding coarsens the same way.

That is finding 4's failure mode exactly: "the error is **correlated across
neighbouring receivers**, so it does not look like attenuated noise -- it looks
like geometry." It is also finding 14's, which is the same thing seen from the
other side: a coarser grid means fewer independent tangent frames, and the
per-receiver rotation is what turns correlated quadrature error into noise the
denoise can remove.

### The answer

**Scale 6 is the limit, not scale 8 and certainly not 16.** Scale 6 is a 2.2x
saving with a *better* RMSE (0.0374 against 0.0402) and an indirect term that is
only slightly softer. Scale 8 already shows coarse blobs in the bounce, and 16
shows correlated patches — neither of which any aggregate in this repository can
see, because all of them are dominated by the per-pixel direct term.

So the reuse radius exists and it is worth **about a factor of two**, which is
what "What this does not answer" predicted when it said "finding 4 is the same
failure at tile granularity and suggests the answer will not be generous." It is
also the reason the 13x in the table above is not real: RMSE and roughness both
improve monotonically in a direction the image rejects, which is finding 21's
lesson arriving a second time on the same knob.

Two caveats kept deliberately:

- **Scale 12 is an unexplained outlier** — RMSE 0.0494 and roughness 0.954,
  nearly double every neighbour. 512/12 is not an integer, so the non-integer
  grid block is the suspect, and finding 21 recorded the same class of problem at
  scale 3. Not chased.
- The silhouette crops finding 21 used (the ceiling/left-wall diagonal and the
  vertical corner) are **identical** across scales 4, 8 and 16. Coarsening is
  safe there for a structural reason: those edges come from the per-pixel
  G-buffer, and a coarser grid gives the joint-bilateral upsample *more* to blur,
  where finding 21's failure came from giving it less. Coarse and fine fail in
  opposite directions, and only the fine direction shows up in a silhouette.

---

## 4. The geometric question, and why its constant is unfavourable

Suppose the camera count is settled and the remaining cameras should share the
geometry they have in common. The criterion is parallax.

Two receivers separated by `s`, looking at geometry at distance `d`, disagree
about its direction by about `s/d` radians. A texel of an `n x n`
hemi-octahedral target subtends `sqrt(2*pi)/n = 2.51/n`, so at level 1's `n = 16`
a texel is 0.157 rad — **nine degrees**. Two receivers can share a surface when

```
s/d  <  theta        =>        d  >  s / theta
```

The grid is 128 cells across the frame, so with `L` the world extent visible
across the frame, adjacent cells are `s1 = L/128` apart and

```
d*  =  (L/128) / 0.157  =  0.05 L
```

Adjacent cells see everything beyond **5% of the visible extent** identically.
That sounds generous, and it is the reason the redundancy in section 1 is so
large. But sharing needs a block, not a pair, and it needs somewhere to put the
shared answer:

| receiver block | `s` | `d*` at receiver resolution | `d*` with a 2x-finer shared map |
|---|---|---|---|
| 2x2 | `L/64` | 0.10 L | 0.20 L |
| 4x4 | `L/32` | 0.20 L | **0.40 L** |
| 8x8 | `L/16` | 0.40 L | 0.80 L |

The last column is the one that matters, and the factor of two in it is not
optional. A shared map that is the *same* resolution as the hemispheres it serves
quantizes every receiver's directions onto one grid — which re-correlates them
across the block and undoes finding 14, the thing that made the indirect term
usable. For the per-receiver jitter to survive the lookup, the map has to be
finer than the hemisphere, and 2x linear (4x the texels) is the least that
plausibly does.

**The sizing law.** A `G x G` block of `n`-texel cameras sharing an `m`-texel map
saves `G^2 n / m`, subject to `m >= 4n` and to `d*` being small enough that the
near field is cheap. At `G = 4`, `n = 256`, `m = 1024` that is a **4x** saving —
but only on geometry beyond `0.40 L`, which in a closed room is the far wall and
part of the ceiling: a minority of a receiver's hemisphere. Call it 30-40% of the
directions, and the overall figure is **about 1.3x**.

Note what this law also explains: example 40's finding 16 built exactly this with
`m = n`, and reported "built, verified, and not yet better" — MAE 8.44 against
the gather's 8.42, visibly blockier, 270 ms per frame, 58.6 MB. At `m = n` the
map *is* the limiting resolution, so the saving is nominally `G^2` and the
quality is spent paying for it. That finding also isolated the reason it cannot
be patched: "**The tangent frame has to be shared, and that costs the
rotation**" — a static per-surfel rotation makes bucket `b` of two neighbours
point in different directions, "which rules out any per-bucket filtering".

So the geometric question is worth roughly 1.3x for a shared-visibility pass,
several megabytes, a new scheduling structure, and a live risk to the one thing
findings 4 and 14 had to fix. **Section 3's grid coarsening is worth 2.2x for an
environment variable.**

That estimate survived measurement; the reasoning behind it did not. Section 5
step 2 measures the neighbour overlap directly and finds 37-63% where this
section's parallax argument implies ~95%. The number lands in the same place for
a different reason, and the reason matters: it is not that the shared answer is
hard to place, it is that **there is much less common work than the fetch counts
suggest**, because the culls have already taken it out.

---

## 5. What to build, in order

1. ~~**Change the default to `MBG_SCALE=6`.**~~ **Done.** 2.2x on the solve
   (473 ms to 220 ms), RMSE 0.0374 against 0.0402, and the only density in the
   ladder whose bounce term is still free of coarse structure. Verified before
   the change on the axes that matter rather than on RMSE: finding 21's two
   silhouette crops are unchanged, the bounce term at 4x exposure is only
   slightly softer, and the composite moves by at most 75/255 on 1.4% of pixels
   — in the direction of the reference. 36/36 gates, and the non-integer grid
   (512/6 = 86, and 267x150 at 1600x900) behaves.
2. ~~**Measure the overlap before building anything geometric.**~~ **Done, and
   it revises section 4.** `MBG_OVERLAP=1` records which clusters each level-1
   camera entered and compares adjacent grid cells. Of a camera's entered
   clusters, its immediate neighbour also enters **37%** on Cornell+bunny and
   **63%** on Sponza — not the ~95% section 4's parallax argument implies.

   | | clusters entered | Jaccard | contained |
   |---|---|---|---|
   | Cornell+bunny, 2172 clusters | 67.7 | 0.242 | **0.372** |
   | Sponza, 8196 clusters | 752.9 | 0.458 | **0.626** |

   **It is not parallax.** Tripling the neighbour separation (scale 6 to 12)
   moves containment from 0.372 to 0.378. **It is only partly the jitter**:
   `MBG_JITTER=0` raises it to 0.458, so finding 14's per-receiver rotation
   costs about nine points of coherence — real, and not the bulk.

   What it is: **the entered set is not a stable geometric property of the
   receiver.** Findings 28, 30 and 34 prune each camera to a small
   bound-sensitive set — 67.7 clusters of 2172, about 3% of the scene — and
   whichever of two neighbours finds a near hit first prunes harder. Two cameras
   that see the same room enter different thirds of it.

   So the redundancy in section 1 is real at the FETCH level and has already
   been removed at the WORK level, by the culls. Modelling a `G x G` shared
   traversal as `union ~ |A|(1 + (G-1)(1-c))`:

   | | G=2 | G=4 |
   |---|---|---|
   | Cornell+bunny | 1.23x | 1.39x |
   | Sponza | 1.46x | 1.89x |

   and that is a *ceiling* on the cluster walk alone. It does not survive
   contact with the fact that `mbg_tri_setup` subtracts `g_P`, so the per-triangle
   setup cannot be shared between cameras at all — only the fetch can, which is
   the half that is already in cache.
3. **Only if (2) contradicts section 4's estimate**, build the shared far field:
   one world-space octahedral map per receiver block, at least 2x finer linearly
   than the hemispheres it serves, built once per block; each receiver then looks
   up **its own jittered directions** in it rather than walking the scene. The
   near field (`d < d*`) stays per-receiver, bounded by the distance and slab
   culls that findings 30 and 34 already built. Budget a few MB, transient for
   one sweep, in global memory — *not* shared memory, so findings 38 and 42 do
   not apply.

   The distinction from example 40's finding 16 is the whole design: the shared
   thing is *directional visibility at higher resolution*, looked up with
   per-receiver directions, rather than a same-resolution micro-buffer
   reprojected and then integrated.

---

## 6. What not to build

- **Interpolating irradiance between receivers.** Finding 4 (correlated error
  reads as geometry), finding 11 (direct light from receivers four pixels apart
  is soft — "that is Nyquist, not a filtering failure"), finding 21 (a gradient
  fit measured as a no-op because "the information is absent, not attenuated").
- **Reprojecting a same-resolution micro-buffer.** Example 40's finding 16, built
  and measured: not better, blockier, 270 ms, 58.6 MB.
- **Sharing a tangent frame across receivers.** Finding 14: the per-receiver
  rotation is what decorrelates the quadrature error, and "neither works alone"
  with the denoise.
- **Reusing the direct term at any radius.** Finding 11, and finding 25 is the
  shape reuse must take instead: skip only where the answer is *certain* — all
  four taps agreeing on exactly 1 or exactly 0 — because that is "not an
  approximation standing in for the answer; it is the answer, reached without the
  work."
- **Anything that adds shared memory or a dynamically indexed local array.**
  Findings 38 and 42: 4 KB of shared memory cost 1.92x, and one indexed local
  array cost 1.47x with the feature switched off.

`Mosaic Lighting.md:234-242` argues the opposite of findings 4 and 11 — "the
cache does not need penumbra accuracy. Indirect light is integrated over a
hemisphere, which is a low-pass filter." Both are right about different terms,
and finding 4's mass split is the reconciliation: the direct half is evaluated at
every texel's own hit point and is never shared; the smooth remainder is what
clustering was always safe for. Any reuse scheme here has to inherit that split
rather than argue with it.

---

## 7. Extending past the first bounce

The first bounce is the clean case and it is not where the work is. On an
RTX 3060, Cornell at the default: `Raster L1` 7.8 ms, `Raster L2` 66.7 ms,
`Raster L3` 42.6 ms of a 139.5 ms frame. **Level 1 is 6% of the frame**, so a
mechanism confined to it cannot matter; levels 2 and 3 carry 40 of the 41 cameras
a primary hit spawns.

Those cameras are not on a grid. They sit at path hit points scattered over each
parent's hemisphere, so grouping them means binning hit points in world space —
a pass that does not exist. Three structural obstacles, all found in the code
rather than assumed:

- **Index identity is a pure function of the schedule at every level.** Level-1
  camera `i` is GI pixel `cursor_ + i` (`place.comp:46-48`, and again at
  `gather.comp:77-79`); a child is slot `cam * u_children + b` (the spawn at
  `raster_body.glsl:558-564`, the gather at `gather.comp:47`). Both `place.comp`
  and the spawn write inactive slots explicitly rather than compacting, and
  `scene.glsl`'s `MbgCam` comment says that is the point. Any scheme that
  introduces a compaction breaks the placement and the gather at once.
- **Masses live in the child's slot and encode the parent's quadrature weight,
  albedo, MC probability and roulette division** (`raster_body.glsl:491-564`). A
  child shared between parents needs a *set* of masses, one per referring parent.
- **The gather is per-parent with a fixed `u_children` loop and no atomics**
  (`gather.comp:45-70`). One child serving several parents needs either
  scatter-with-atomics or a per-parent child list.

None of that is a reason not to do it. It is the reason the first bounce is worth
designing against: it is the only level where the receivers already form a grid,
so the criterion in section 4 can be measured there before the binning pass
exists.

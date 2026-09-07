# Surfel Global Illumination via FMM Radiance + Microbuffer Visibility

An implementation specification for emissive-surfel direct + indirect lighting with
rasterized near-field occlusion, targeting low-to-mid-range GPUs.

**Revision 3.** The far field keeps revision 2's FMM structure but fixes its
approximation: the source-side representation is now a spherical-harmonic **radiant
intensity** vector rather than exitance-plus-normal-cone, which makes M2M exactly linear.
The near-field kernel is restructured from one workgroup per receiver to one workgroup per
**cell**, which is where most of the available parallelism was being lost. Several
accuracy claims from revision 2 are retracted as unjustified. See Appendix A for the
changelog.

---

## 0. What this is

Every surfel is simultaneously a **receiver** and a potential **emitter**. There are no
separate light objects. An emissive mesh, an area light, and a bounce surface are all the
same primitive. Lighting is solved by, for each receiver:

1. Evaluating incident radiance from all **well-separated** geometry via an FMM local
   expansion, in O(N), ignoring occlusion.
2. Resolving the **near field** — geometry in adjacent cells — with a per-receiver
   microbuffer that determines which surfel is actually visible in each direction.

The split exists because radiance superposes linearly and visibility does not. Everything
in this document follows from that one fact, and §3.2 of revision 2 violated it in one
place; see the normal-cone note below.

The critical additional fact, which is what makes the FMM half work: **irradiance is a
low-pass filter of incident radiance.** The clamped-cosine kernel `max(0, n·w)` has
spherical-harmonic coefficients that decay as `1/l^2` with the band-3 term vanishing
identically; bands 0-2 capture ~99% of the response for *any* receiver normal
(Ramamoorthi & Hanrahan 2001). So a 9-coefficient local expansion of the radiance field
over a target cell is near-exact for Lambertian receivers.

**Be precise about what that argument does and does not cover.** There are two
independent truncations in this method:

| | Truncation | Order | Error bounded by |
|---|---|---|---|
| **Target side** | local expansion of incident radiance about the target cell | order-2 SH | R&H: `ahat_3 = 0`, ~1% |
| **Source side** | expansion of the source cell about its expansion point | see below | *not* covered by R&H |

Ramamoorthi & Hanrahan bounds the target side only. Revision 2 stored a single point
source per cell — a positional monopole — and then cited R&H as though it bounded the
whole method. It does not. With the V-list stencil the source cell has circumradius
`0.87h` at a centre-to-centre distance of `2h` to `5.2h`, which is an effective opening
angle of roughly 25 degrees. That is a *much* looser separation than revision 1's
Barnes-Hut `theta_open`, not a tighter one.

The honest framing: **revision 2 traded far-field accuracy for a ~40x reduction in
operations, and the trade is a good one** because irradiance is forgiving and errors
partially cancel across 189 sources. But do not expect the far field to be more accurate
than a well-tuned Barnes-Hut descent. It is not. It is much faster, has no `theta_open`
to mistune, and is accurate *enough*. §3.2 gives an optional source-side correction if you
need a real accuracy knob.

The one thing revision 2 got outright wrong was the **normal cone**. `coneFacingFactor` is
not linear in the child cells, so merging cones at M2M does not produce the correct
aggregate emitter — a parent whose children face opposite directions gets a wide cone that
passes full exitance in both directions. Revision 3 replaces exitance-plus-cone with an SH
radiant intensity vector, which restores exact linearity at M2M and happens to reuse the
same `ahat` constants. See §3.2.

Visibility does not superpose, which is why it stays in the near field. Do not attempt to
push it into the expansion.

**What you get:** noise-free, temporally stable, soft area-light shadows and one-bounce
(or multi-bounce, if iterated) indirect light, with no ray tracing and no Monte Carlo.
Far-field cost is O(N) with a small constant and no per-receiver traversal at all.

**What you do not get:** ground-truth-matching sharp shadows, correct specular, or
occlusion at range. Light passing through a wall thicker than one cell is *not* blocked
by the far field. This is a biased method. Bias shrinks with surfel density, cell size,
and microbuffer resolution, not with time.

---

## 1. Data structures

### 1.1 Surfel

Stored **SoA, not AoS**. The visibility kernel touches position/radius/normal for every
candidate but shading terms only for the ~64 bucket winners; interleaving them wastes
2-4x of every cache line you pull.

```c
// buffer A: float4 { pos.xyz, radius }      16 B  -- one aligned fetch, hot
// buffer B: uint   { octahedral normal }     4 B  -- hot
//
// Cold: read only for bucket winners and for P2M.
// buffer C: uint albedo    RGB8 + 8b flags/roughness
// buffer D: uint emission  RGB9E5 emitted RADIANCE  (W/m^2/sr)
// buffer E: uint irradiance RGB9E5   -- see §7 on why this is NOT ping-ponged
```

**Units, stated once, because getting this wrong is the most common failure:**

- `emission` is **radiance**, `L_e`, in W/m^2/sr.
- The irradiance buffer stores **irradiance** `E`, in W/m^2 — the cosine-weighted integral
  of incident radiance, not radiance itself.
- A surfel's outgoing radiance is therefore `L_out = L_e + albedo * E / PI`.

The `/PI` belongs to the surfel that is *emitting*, and it must appear everywhere
`L_out` is formed — in the microbuffer integration, in the P2M aggregate, and in the
brute-force reference. Revision 1 had it in the receiver's write but not in the
near-field emitter term, which made every bounce 3.14x too bright *consistently*, so
energy validation would not have caught it.

Octahedral normal encoding: 16:16 snorm, ~0.1 degree error. Plenty.
RGB9E5 gives HDR range in 4 bytes and is fine for irradiance. It is **not** fine for SH
coefficients, which are signed and span several orders of magnitude across bands — but
§1.2 now sidesteps that question entirely.

### 1.2 The FMM grid

FMM wants a **level-uniform spatial subdivision with a fixed geometric neighbourhood**,
not a general BVH.

A BVH is the right structure when every query is an independent traversal, because it
adapts to density. FMM does not traverse. It evaluates a **fixed stencil** — for a
uniform octree, each cell's interaction list is exactly the 189 cells that are children
of its parent's neighbours but are not themselves adjacent to it. That is a compile-time
constant offset pattern. No pointer chasing, no divergence, no per-receiver work at all.
Losing density adaptivity is worth it to get a stencil.

```c
// Sparse octree, Morton-keyed, built to a fixed leaf depth.
// Only occupied cells are stored; a hash or sorted-key array maps Morton -> slot.

struct Cell {
    // --- multipole (upward pass), expanded about the cell CENTRE ---
    float3 Ilm[9];        //108B  order-2 SH of radiant INTENSITY, W/sr, per channel
    float3 mu;            // 12B  flux-weighted first moment about centre (optional, §3.2)
    // --- local expansion (downward pass) ---
    float3 Lsh[9];        //108B  order-2 SH of incident radiance, per channel
};
```

**Use fp32 in the Cell struct. Do not pack it.** Revision 2 agonized over fp16 versus
RGB9E5 for SH coefficients. At `h = 1.5s` and 500K surfels you have ~7000 occupied leaf
cells; 216 B each is 1.5 MB. This is a rounding error against a 250 MB surfel set. Packing
buys nothing and costs you a class of precision bugs — in particular `Ilm` is
`area * radiance * pi`, which overflows fp16 for bright emitters with large aggregate
area. Ship fp32 and stop thinking about it.

**Note what is gone.** `centroid`, `area`, `exitance`, and `normalCone` are all deleted.
The intensity SH subsumes area, exitance, and directionality in one linear quantity, and
the expansion point is now the cell centre rather than a drifting centroid — which is
what guarantees the `2h` separation bound and lets §3.2 drop the soft form-factor clamp.

Two tiers, pick one:

| Tier | Stored | Bytes/cell | Accuracy |
|------|--------|-----------|----------|
| **A — monopole** | `Ilm` + `Lsh` | ~216 B | Source cell collapsed to a point at its centre. |
| **B — first moment** | `+ mu` | ~228 B | Expansion point shifted by the clamped flux centroid. Cheap, meaningful. |

Ship tier A, measure, then add tier B if §9 step 3 shows a delta you care about. Tier B is
twelve extra bytes and about six extra ALU ops per M2L; it is close to free.

### 1.3 Choosing the leaf cell size

This is the single most important tuning parameter, and revision 2 mischaracterized it as
a one-way derivation from `K`. It is a three-way balance.

Let `s` = mean surfel spacing, `h` = leaf cell edge length, `q = h/s`.

| Quantity | Scaling | Direction |
|---|---|---|
| U-list occupancy `K ~ 27 q^3` | grows as `q^3` | larger `h` = more near-field ALU |
| Receivers per cell `~ q^3` | grows as `q^3` | larger `h` = more candidate **reuse** (§4.1) |
| Occupied cell count `~ N / q^3` | shrinks as `q^-3` | larger `h` = **cheaper** M2L |
| Far-field error | grows with `h` | larger `h` = coarser source approximation |
| Near-field radius | `h` to `2h` | larger `h` = longer contact shadows resolved |

Two consequences revision 2 missed:

**Near-field ALU and M2L cost pull in opposite directions.** Total near-field work is
`27 N q^3`; total M2L work is `189 * 27 * N / q^3`. There is a genuine minimum, and it is
not where `K` alone puts it. At 500K surfels the near field dominates by a wide margin at
any sane `q`, so you still want `q` small — but check the M2L column before going below
`q = 1.25`, because cell count explodes cubically and you will start paying for the
189-cell stencil.

**With the §4.1 restructure, memory traffic is independent of `h`.** Per-cell candidate
fetch is `K ~ 27 q^3`, cell count is `N / q^3`, so total candidate traffic is `~27 N`
regardless of `q`. Revision 2's per-receiver kernel re-fetched candidates once per
receiver, making traffic scale as `27 N q^3`. Once the candidate list is loaded to LDS
once per cell, `h` becomes a pure ALU knob. On the APU targets, which are bandwidth-bound,
this is the single most valuable change in this revision.

Target `K` in the 64-192 range. For 500K surfels this is typically `h = 1.5-2s`, giving
3000-8000 occupied leaf cells.

**Hard cap: `K <= 256`.** §4.1 packs the winner's index into 8 bits of a 32-bit atomic.
The per-cell density counter in §2.1 must enforce this, not merely encourage it.

### 1.4 Build

Radix sort of Morton codes over surfel positions, then a compaction pass to produce the
occupied-cell list per level. ~0.3 ms for 500K surfels on a 3060, ~1.5 ms on a Steam Deck.

Rebuild the whole thing per frame if geometry is dynamic. Unlike a BVH there is no
refit/rebuild quality tradeoff — the grid is geometric, so a rebuild is just a re-sort.
Static scenes can cache the leaf occupancy and skip the sort entirely, updating only
the multipole payloads.

The sort also gives you the receiver ordering for free: because pass 7 now dispatches per
cell (§4.1), the Morton-sorted surfel array *is* the per-cell receiver list, and
`gCellFirst / gCellLast` serve both the candidate walk and the receiver loop.

---

## 2. Pipeline

Per frame, in order:

| # | Pass | Threads | Notes |
|---|------|---------|-------|
| 1 | Surfel spawn / update / recycle | 1/surfel | See §2.1 |
| 2 | Morton sort + sparse grid build | 1/surfel | Dynamic geometry only |
| 3 | **P2M + M2M upsweep** | 1/cell/level | Bottom-up. Exact summation. |
| 4 | **M2L** | 1/cell/level | Fixed 189-cell stencil. The FMM core. |
| 5 | **L2L downsweep** | 1/cell/level | Top-down |
| 6 | Microbuffer visibility solve **+ far-field evaluation** | **1 wave/cell** | The expensive one |
| 7 | Apply to screen | 1/pixel | Interpolate surfels to G-buffer |

Revision 2's separate L2P pass is **gone**. It computed a single scalar irradiance per
receiver that was then blended against the near field with a `(1 - coverageFraction)`
scale — a heuristic that §5 already identified as wrong. Pass 6 now evaluates the local
expansion *per bucket*, in the direction that bucket actually looks, which is both more
correct and cheaper than a separate pass plus a heuristic blend. Keep a standalone L2P
kernel only for the lowest LOD tier (§6.1) and as the register-bound fallback.

Pass 6 is amortized temporally (§7). Passes 3-5 are cheap enough to run in full every
frame and you should, because partial FMM updates produce spatially inconsistent
expansions that look like blotches.

There is **no candidate-list pass**. Revision 1 materialized K indices per receiver to
VRAM and read them back — at 250K surfels and K=128 that is ~128 MB round-tripped per
full update, on hardware that shares LPDDR5 with the CPU. Pass 6 walks its own U-list
(27 cells, known offsets, no traversal) once per cell and holds it in LDS.

### 2.1 Surfel allocation

Genuinely the hardest part of shipping any surfel GI. Minimum viable:

- Spawn from the G-buffer: each frame, pick low-coverage screen tiles and spawn a surfel
  at the depth-projected world position, radius set from projected pixel footprint.
- Maintain a per-cell coverage counter; reject spawns into cells already at target
  density. This is what keeps `K` bounded, and `K` bounded is what keeps pass 6 bounded.
  **This is now a correctness requirement, not a performance one:** `K > 256` overflows
  the 8-bit index field in §4.1's packed atomic.
- **Clamp surfel radius against `h`.** A surfel with `radius > ~0.25h` at close range
  subtends more than the 3x3 bucket neighbourhood that §4.1's splat loop covers, and will
  leave holes in the microbuffer that read as light leaks. Either cap the radius at spawn
  or route oversized candidates through the conservative path in §4.2.
- Recycle surfels not seen for N frames, with a grace period longer than your temporal
  update interval or you will thrash.
- **Offscreen geometry still needs surfels.** Indirect light from behind the camera is
  most of what makes this look better than SSGI. Keep a ring of retained surfels beyond
  the frustum, recycled by distance rather than visibility.

EA SEED's "Global Illumination Based on Surfels" (2021) covers this properly. Budget
real time for it; it is not a footnote.

---

## 3. The math

### 3.1 Why the cosine is the only correction

The differential form factor between receiver P and emitter surfel E is

```
dF = (cos_P * cos_E) / (pi * r^2) * dA_E
```

The solid angle E subtends at P is

```
dw = (cos_E * dA_E) / r^2
```

Substituting: `dF = cos_P * dw / pi`. The emitter cosine and the inverse-square falloff
are **already inside the solid angle**. If you integrate over the hemisphere in angular
space — which is exactly what a microbuffer does — only the receiver cosine remains.

```
E(P) = SUM_over_buckets[ L_in(b) * cos(theta_b) * dw_b ]
L_out(P) = L_e(P) + albedo_P * E(P) / pi
```

**Critical: do not also apply 1/r^2 inside the microbuffer path.** Applying inverse-square
there double-counts falloff and darkens everything nearby. The FMM path (§3.2) does carry
an explicit `1/r^2`, because it works in intensity-and-direction form rather than
integrating a rasterized hemisphere. Two formulations of the same integral; never mix them
within one direction.

### 3.2 Far field: the FMM

The whole far field is built on one identity. A Lambertian surfel of area `a`, outgoing
radiance `L_out`, and normal `n` has radiant intensity

```
I(w) = a * L_out * max(0, n . w)          [W/sr]
```

and the clamped cosine `max(0, n . w)`, as a function of `w`, has the **exact** SH
projection `ahat_l * Y_lm(n)` with

```
ahat = { pi, 2pi/3, 0, pi/4 }      // bands 0,1,2,3 -- band 3 vanishes
```

These are the same Ramamoorthi-Hanrahan constants used at L2P. That is not a coincidence;
it is the same convolution applied at the emitter instead of the receiver. It means an
order-2 SH intensity vector represents a *cluster of Lambertian emitters* to within ~1%,
regardless of how their normals are distributed — which is precisely the property the
normal cone was trying and failing to provide.

#### P2M — surfel to multipole

For each leaf cell, accumulate over its surfels, expanding about the **cell centre**:

```
L_out,i = L_e,i + albedo_i * E_i / pi          // note the /PI
Ilm    += a_i * L_out,i * ahat_l * Y_lm(n_i)   // 9 coeffs, RGB
mu     += a_i * lum(L_out,i) * (pos_i - cellCentre)   // tier B only
```

`mu` is the flux-weighted first moment about the centre. Divide by the accumulated flux
weight at the end to get an offset vector.

#### M2M — child multipoles to parent

```
parent.Ilm[lm] += child.Ilm[lm]
```

That is the entire operation. Intensity is linear, so M2M is exact summation with no
merge heuristic, no cone union, no `pi/2` clamp, and no "mark the cell omnidirectional"
degenerate case. Revision 2's cone merge was the one place the design's linearity premise
was violated, and it biased bright in exactly the configuration that matters most — a
cell containing a wall and its opposite face.

For tier B, `mu` merges as a flux-weighted average of the children's shifted moments,
re-centred on the parent centre. If that bookkeeping annoys you, skip tier B above the
leaf level; the coarse levels contribute least and are furthest away.

#### M2L — multipole to local expansion

For target cell T, for each of the 189 source cells S in T's interaction list:

```
p      = S.centre + (tier B ? clampOffset(S.mu, 0.25 * h) : 0)
d      = p - T.centre
r2     = dot(d, d)
dir    = d * rsqrt(r2)

I      = max(0, evalSH(S.Ilm, -dir))       // W/sr toward T. 9 madds + 20 for Y_lm.
for (l,m) in bands 0..2:
    T.Lsh[lm] += (I / r2) * Y_lm(dir)
```

Two things that are gone and should stay gone:

- **The soft clamp `+ S.area` in the denominator.** It existed because revision 2 expanded
  about a luminance-weighted centroid that could drift up to `0.87h` from the cell centre,
  so two V-list cells nominally `2h` apart could have expansion points `0.26h` apart and
  the `1/r^2` would blow up. Expanding about the cell centre restores the stencil's
  guarantee: `r >= 2h`, always. No singularity, and no bias from a clamp that was
  suppressing real energy from the nearest and brightest sources.
- **`coneFacingFactor`.** Replaced by `evalSH(S.Ilm, -dir)`, which is a dot product with
  no branch and is linear under M2M.

`clampOffset` in tier B limits the shift so the separation guarantee survives; `0.25h`
keeps `r >= 1.5h` in the worst case, which is ample.

Cost: 9 `Y_lm` polynomial evaluations for `-dir` (~20 ops, shared with `dir` up to sign
flips on the odd band), 9 madds to evaluate `I`, then 27 madds to accumulate. Roughly
5500 FLOPs per target cell for the whole 189-cell list. No transcendentals, no branches,
no memory indirection beyond the stencil.

**Sanity check the whole chain on one surfel.** Single emitter, single receiver:

```
I(-dir)  = a * L_out * cos_E
Lsh[lm]  = (a * L_out * cos_E / r^2) * Y_lm(dir)
E(P)     = SUM_l ahat_l SUM_m Lsh[lm] Y_lm(n_P)
         = (a * L_out * cos_E / r^2) * max(0, dir . n_P)
         = a * L_out * cos_E * cos_P / r^2
```

which is the exact point-to-point transfer. No stray `pi` anywhere, because `L_out`
already carries the `/pi`. Run this as a unit test; it catches every normalization
convention error in one shot.

#### Deringing

Order-2 SH of a near-delta source has negative lobes. With many summed sources these
mostly cancel, but a single dominant emitter (a sun-like area light seen through a gap)
will produce negative irradiance on surfaces facing away.

Revision 2 prescribed an unconditional Hanning window:

```
w_l = (1 + cos(pi * l / 3)) / 2     // w0=1, w1=0.75, w2=0.25
```

**Do not apply this unconditionally.** It attenuates the band-2 term to 25% *everywhere*,
which is a permanent loss of directional contrast in every cell, to fix an artifact that
occurs in a small minority of them. Instead:

1. Clamp final irradiance to zero at evaluation time. This is free and handles the
   visible symptom.
2. Track a dominance ratio at M2L time — `maxSingleContribution / totalContribution`,
   one extra max and one extra add per source — and apply the window only in cells where
   it exceeds ~0.5.

If you need a single global setting for simplicity, window band 2 only at `w2 = 0.5` and
accept a smaller contrast loss than revision 2 prescribed.

#### L2L — parent local expansion to children

```
child.Lsh[lm] += parent.Lsh[lm]
```

**This is a value copy, not a Taylor re-centring, and it has a visible consequence that
revision 2 did not name.** The local expansion is therefore piecewise constant over each
leaf cell, so the far-field irradiance field is discontinuous at every cell boundary. The
artifact is a blocky brightness modulation on a grid — which is exactly the signature
§10 step 5 warns about and attributes to gaps or overlaps in the interaction lists. You
would chase the wrong bug.

Revision 2 offered analytic gradients (`dLsh`, 162 B/cell plus a derivation) as "tier B,
for larger cells". The fix is real but the mechanism is overkill. **Trilinearly interpolate
`Lsh` from the 8 nearest cell centres at evaluation time instead.** It removes the
discontinuity, needs no extra storage, needs no derivative derivation, and — the reason it
is nearly free here — pass 6 has *already resolved all 27 U-list cells*, so the 8 taps are
guaranteed to be in LDS. See §4.1.

Analytic gradients remain the right answer if you later want leaf cells 2-3x larger.
They are not needed to ship.

#### Evaluating the expansion at a receiver

```
E(P, w) evaluated per bucket:   L_far(w) = SUM_lm Lsh_interp[lm] * Y_lm(w)
E(P) accumulated:               SUM_buckets L_far(w_b) * cos_b * dw_b
```

Note this replaces the closed-form L2P `E = SUM_l ahat_l SUM_m Lsh[lm] Y_lm(n_P)`. The
closed form is the analytic cosine convolution over the *whole* hemisphere; the per-bucket
form integrates the same field numerically over only the *unoccluded* directions, which is
what §5 needs. The two agree when nothing occludes — worth asserting in a debug build.

Keep the closed form for the LOD tier where you skip the microbuffer entirely: nine
`Y_lm(n_P)`, 27 madds, three band scales, ~50 ALU, fully register-resident.

#### What this actually buys over Barnes-Hut

For 500K surfels with revision 1's `theta_open = 0.02`, each receiver descends into
roughly 150-300 nodes at ~30 ops each: **~3 GFLOP per full update**, all of it divergent
traversal with incoherent node fetches.

FMM at `h = 1.5s`: ~6000 leaf cells x 189 x ~30 madds, plus ~1/8 that at each coarser
level: **~35 MFLOP** of M2L, a fixed stencil with perfect coherence, plus the per-bucket
far-field evaluation folded into pass 6.

That is roughly a 40x reduction in far-field operations and a much larger reduction in
wall time, because the operations that remain are the ones GPUs are good at.

**Accuracy is a wash to slightly worse, not better.** Revision 2 claimed an improvement
and justified it with `ahat_3 = 0`, which as §0 explains bounds only the target side.
Barnes-Hut at `theta = 0.02` resolves sources far more finely than a fixed V-list stencil
with a 25-degree effective opening angle. What you actually gain is: no `theta_open` to
mistune per scene, exact linearity at M2M (revision 3), and a cost model that does not
depend on scene configuration. Those are worth having. Higher fidelity is not on the list.

### 3.3 Interaction lists

For a level-uniform octree, per cell:

- **U-list** — the cell itself plus its 26 face/edge/vertex neighbours. Adjacent, so the
  expansion is inadmissible. **This is the near field**, handled by the microbuffer.
- **V-list** — the 189 cells that are children of the parent's neighbours and are not in
  U. **This is M2L.**

Cells further away are handled at coarser levels by the parent's own V-list, which is
where the O(N) comes from. Because the tree is level-uniform, W- and X-lists (needed only
for adaptive FMM) do not exist. Keep it that way; the adaptive variants are where FMM
implementations go to die.

Both lists are constant integer offsets in Morton space. Generate them from the 3D
coordinate delta inline — the test is `max(|dx|,|dy|,|dz|) > 1 && max(...) <= 3` over the
parent-relative 6x6x6 neighbourhood. Two comparisons, no table fetch. (`6^3 - 3^3 = 189`,
which is a useful assertion to put in the build.)

### 3.4 Near-field microbuffer

The U-list surfels are the near-field candidate set: close enough that occlusion between
them matters. `K` is 64-192 by construction, hard-capped at 256 (§1.3).

Hemisphere parametrization: **hemi-octahedral**. Map the hemisphere to a unit square,
subdivide into an NxN grid. Bucket solid angles are near-uniform, including at grazing
angles, which is where contact shadows live. Polar/spherical parametrizations concentrate
buckets at the pole and starve the horizon — avoid them.

```c
// direction (in P's tangent frame, z = normal) -> bucket index
uint dirToBucket(float3 d, uint N) {
    float2 p = d.xy / (abs(d.x) + abs(d.y) + d.z);   // hemi-oct fold
    float2 uv = float2(p.x + p.y, p.x - p.y) * 0.5 + 0.5;
    uint2  c  = min(uint2(uv * N), N - 1);
    return c.y * N + c.x;
}
```

**Neighbour lookup at the seam is not trivial.** `neighbourBucket()` in §4 must handle the
diamond fold: stepping off the edge of the unit square wraps to a mirrored, reflected
position, not a toroidal one. Get this wrong and wide discs at grazing angles splat into
the wrong hemisphere half. Write it as an explicit 8-case function against the folded
coordinates and unit-test it against a brute-force angular search before you trust
anything downstream.

Precompute `bucketDir[b]` and `bucketSolidAngle[b]` into constant memory. They are
per-bucket constants in the tangent frame and never change.

---

## 4. The visibility kernel

This is where the frame time goes.

### 4.1 One workgroup per *cell*, not per receiver

Revision 2 dispatched one wave per receiver and had each wave walk its own 27-cell U-list.
That structure loses most of the machine on two independent counts:

**Lane utilization.** The inner loop `for (i = first + lane; i < last; i += WAVE)` iterates
over one cell at a time. At `h = 1.5s` a cell holds ~3.4 surfels, so a 32-lane wave runs 27
loop iterations at roughly 10% occupancy. Nine tenths of your ALU throughput is idle lanes,
in the kernel that is supposed to be ALU-bound.

**Redundant fetch.** Every receiver in a cell has the *same* 27-cell U-list and the *same*
`K` candidates. Revision 2 re-probed the cell hash 27 times per receiver (13.5M incoherent
probes at 500K surfels) and re-fetched all `K` candidate positions per receiver.

Both are fixed by dispatching per occupied leaf cell and hoisting the shared work:

```hlsl
#define BUCKETS   64          // 8x8 hemi-oct grid
#define WAVE      32          // one wave per group; see §4.3
#define MAXK      256         // hard cap, enforced by §2.1
#define EMPTY     0xFFFFFFFFu

groupshared float4 gCandPR [MAXK];      // pos.xyz, radius            4 KB
groupshared uint   gCandIdx[MAXK];      // global surfel index        1 KB
groupshared uint   gULSlot [27];        // resolved U-list cell slots
groupshared float3 gLsh    [27][9];     // U-list local expansions  2.9 KB
groupshared uint   gMicro  [BUCKETS];   // depth24 | localIdx8      256 B
groupshared uint   gK;

[numthreads(WAVE, 1, 1)]
void VisibilitySolve(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint cellSlot = gActiveCellList[gid.x];
    int3 baseCell = gCellCoord[cellSlot];

    // ================= PHASE 1: once per cell, amortized over all receivers =========
    if (lane == 0) gK = 0;
    GroupMemoryBarrierWithGroupSync();

    // 27 lanes, one U-list cell each. Resolve the hash and pull the local expansion.
    if (lane < 27) {
        int3 cc         = baseCell + int3(lane % 3, (lane / 3) % 3, lane / 9) - 1;
        uint slot       = lookupCell(mortonEncode(cc));
        gULSlot[lane]   = slot;
        [unroll] for (int k = 0; k < 9; ++k)
            gLsh[lane][k] = (slot != ~0u) ? gCellLsh[slot * 9 + k] : 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Flatten the 27 cells into one contiguous candidate array in LDS.
    // This loop IS load-imbalanced -- and that is fine, because it runs once per cell
    // instead of once per receiver, so its cost is divided by the receiver count.
    for (uint u = 0; u < 27; ++u) {
        uint slot = gULSlot[u];
        if (slot == ~0u) continue;
        uint first = gCellFirst[slot], last = gCellLast[slot];
        for (uint i = first + lane; i < last; i += WAVE) {
            uint dst;  InterlockedAdd(gK, 1, dst);
            if (dst < MAXK) { gCandPR[dst] = gSurfelPosRad[i]; gCandIdx[dst] = i; }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint K = min(gK, MAXK);

    // ================= PHASE 2: per receiver, perfectly load balanced ===============
    uint rFirst = gCellFirst[cellSlot], rLast = gCellLast[cellSlot];

    for (uint P = rFirst; P < rLast; ++P)
    {
        if (!needsUpdate(P)) continue;                  // temporal amortization, §7

        float4   pr  = gSurfelPosRad[P];
        float3   nP  = decodeOct(gSurfelNormal[P]);
        float3x3 TBN = buildTangentFrame(nP);           // world -> tangent

        // Trilinear far field. All 8 taps are already in LDS from phase 1.
        float3 Lsh[9];
        interpolateLsh(pr.xyz, baseCell, gLsh, Lsh);    // ~216 madds

        for (uint b = lane; b < BUCKETS; b += WAVE) gMicro[b] = EMPTY;
        GroupMemoryBarrierWithGroupSync();

        // ---- scatter: flat, every lane busy every iteration ----
        for (uint j = lane; j < K; j += WAVE)
        {
            uint gi = gCandIdx[j];
            if (gi == P) continue;                      // self-exclusion

            float4 c     = gCandPR[j];
            float3 d     = c.xyz - pr.xyz;
            float  r2    = dot(d, d);
            float  invR  = rsqrt(r2);
            float3 local = mul(TBN, d * invR);
            if (local.z <= 0.02) continue;              // horizon bias

            float angR  = c.w * invR;                   // angular radius
            float depth = r2 * invR - c.w;              // NEAR surface, not centre
            if (depth <= 0) continue;                   // interpenetrating

            uint e  = packEntry(depth, j);              // note: j is LOCAL, 0..K-1
            uint bc = dirToBucket(local, 8);
            InterlockedMin(gMicro[bc], e);

            if (angR > gBucketHalfAngle) {
                [unroll] for (int o = 0; o < 8; ++o) {
                    uint nb = neighbourBucket(bc, o);
                    if (dot(gBucketDir[nb], local) > cos(angR))
                        InterlockedMin(gMicro[nb], e);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // ---- integrate: near and far in ONE pass, coverage-blended ----
        float3 sum = 0;
        for (uint b = lane; b < BUCKETS; b += WAVE)
        {
            float3 bd   = gBucketDir[b];                // tangent frame
            float  cosP = bd.z;
            float  dw   = gBucketSolidAngle[b];
            float3 bw   = mul(bd, TBN);                 // tangent -> world

            float3 Lfar  = max(0, evalSH(Lsh, bw));     // far field, THIS direction
            float3 Lnear = 0;
            float  cov   = 0;

            uint e = gMicro[b];
            if (e != EMPTY) {
                uint   j  = e & 0xFF;                   // local index
                uint   gi = gCandIdx[j];
                float4 c  = gCandPR[j];
                float3 d  = c.xyz - pr.xyz;
                float  ar = c.w * rsqrt(dot(d, d));
                cov = saturate(PI * ar * ar / dw);      // disc solid angle / bucket

                // NOTE THE /PI. Stored irradiance -> outgoing radiance.
                Lnear = decodeRGB9E5(gSurfelEmission[gi])
                      + decodeRGB9E5(gSurfelIrrad[gi])
                        * decodeRGB8(gSurfelAlbedo[gi]) / PI;

                // backface: emits nothing, but STILL occludes (cov stays)
                if (dot(decodeOct(gSurfelNormal[gi]), -normalize(d)) <= 0) Lnear = 0;
            }

            sum += (Lnear * cov + Lfar * (1 - cov)) * cosP * dw;
        }

        sum = WaveActiveSum(sum);
        if (lane == 0) blendIrradianceInPlace(P, sum);  // §7 -- NOT a ping-pong write
    }
}
```

**Points that matter:**

- **The packed entry is 32 bits, not 64.** Revision 2 used a 64-bit `atomicMin` because it
  packed a *global* surfel index. The index only needs to address within the U-list, and
  `K <= 256` fits in 8 bits. So `packEntry` is
  `(min(0xFFFFFF, uint(saturate(depth / maxNearDepth) * 0xFFFFFF)) << 8) | j`,
  with `maxNearDepth = 2 * sqrt(3) * h`. That is a plain 32-bit `InterlockedMin`, supported
  everywhere, half the LDS, and a faster atomic. It also removes §4.4's entire reason to
  exist as a compatibility path. (If you do keep a 64-bit variant with a raw float depth,
  remember that IEEE positive floats compare correctly as integers — `asuint` is safe, but
  only for positive values, and only after the `depth <= 0` reject.)
- **Coverage now blends near against far symmetrically**, in one line, in one place. This
  simultaneously fixes revision 2's contradiction between §4.1's code (`farE * (1-frac)`)
  and §5's prose (per-empty-bucket SH), implements §6.3's coverage weighting, and stops a
  30%-coverage winner from killing 100% of the far field in its direction. Empty buckets
  fall out as `cov = 0`. There is no `coverageFraction` scalar anywhere any more.
- **The far field is trilinearly interpolated for free.** The 8 surrounding cell centres
  are a subset of the 27 U-list cells phase 1 already loaded. This is what removes the
  cell-boundary discontinuity described in §3.2's L2L section, at the cost of ~216 madds
  per receiver and zero extra memory traffic.
- `depth = distance - radius` uses the near surface of the disc, not the centre. Using
  the centre lets a large surfel lose to a small one it actually occludes.
- The backface check zeroes **emission but not coverage**. A surfel facing away still
  blocks. Culling it from the depth test is the single most common source of light leaks
  in this kind of system.
- `local.z <= 0.02` is a horizon bias. Without it, coplanar neighbours self-shadow and you
  get surfel-scale acne, exactly like shadow-map acne.
- Receivers are iterated **serially within the group** so all `WAVE` lanes cooperate on
  each one. With `WAVE = 32` the group is a single wave, so the barriers are
  wave-synchronous and compile to little or nothing; they are still required to stop the
  compiler reordering LDS access.

**LDS budget:** ~8.2 KB per group as written. On RDNA2 that allows 7-8 concurrent groups
per CU, which is enough to hide the atomic latency. If you are occupancy-bound, store
`gLsh` as `half3` (1.5 KB instead of 2.9 KB) — it is only ever consumed by an
interpolation, so the precision loss is harmless, unlike storing the multipoles packed.

**Traffic:** candidate fetch is now `~27 N` total, independent of `h` (§1.3). Revision 2's
was `~27 N (h/s)^3`. At `h = 1.5s` that is a 3.4x reduction; at `h = 2s`, 8x.

### 4.2 Atomic contention is still the real cost

The revision 1 budget of `receivers * K * 40` gave ~160 MFLOP for a Steam Deck frame,
which is ~0.1 ms of raw FP32 against a quoted 3.5-5 ms. That 40x gap was doing all the
work and was never explained. It is mostly `InterlockedMin` serialization.

Revision 3's restructure does not fix this. It fixes lane utilization and bandwidth, which
are different problems. Near-field surfels are spatially clustered, so with `K = 128` over
64 buckets a handful of buckets receive most of the traffic, and the neighbour-expansion
loop multiplies atomic count by up to 9x. **Profile this specific thing before trusting
any timing in §8.**

Mitigations, in order of preference:

1. **Pre-sort candidates by bucket in registers** using a wave ballot before the atomic,
   so each bucket is written once per wave by one lane rather than up to `WAVE` times.
   Costs a few ballots and a prefix sum; removes most of the contention.
2. **Conservative splat for oversized discs.** A disc whose angular radius exceeds ~2
   bucket widths should splat a fixed 3x3 block unconditionally rather than running the
   8-neighbour test. Note this only covers one ring: a surfel with `radius > ~0.25h` at
   close range subtends more than 3x3 and will leave holes. §2.1 caps radius at spawn for
   exactly this reason; if you cannot, such candidates need a separate wider splat path.
3. **Drop to 32 buckets** on the low tier. Halves the bucket count and the contention, and
   §6.1's LOD already wants this for most receivers.

### 4.3 Wave width

**RDNA2 is not natively wave64.** RDNA SIMDs are 32 lanes wide; wave64 is supported but
issues over two cycles. Revision 1's claim that Steam Deck and 680M give "64 buckets at
the same per-lane cost" is false. Assume wave32 everywhere, accept two iterations for the
64-bucket loops, and do not compile a wave64 variant expecting a win. If you want one lane
per bucket, use 32 buckets.

### 4.4 Alternative: lane-per-bucket, zero atomics

Each lane owns one or two buckets; every candidate is tested against every bucket.

Revision 2 filed this as a compatibility fallback for hardware lacking 64-bit LDS atomics.
That justification is gone — §4.1 uses a 32-bit atomic that every target supports. But the
technique is now *more* interesting than it was, not less, and it deserves reconsideration
on its merits:

- The candidates are already in LDS, so the redundancy is pure ALU, not pure bandwidth.
- No atomics at all, which removes the cost §4.2 says dominates.
- Zero divergence, perfect lane utilization, fully register-resident accumulation.
- It composes with the per-cell dispatch without any change.

Cost is roughly `K * BUCKETS` coverage tests instead of `K` plus atomics — about an order
of magnitude more ALU. Concretely: viable on the RTX 3060 tier at half update rate,
**not** viable on Steam Deck. Given that this design is deliberately ALU-heavy and
atomic-bound, build both and let the profiler pick per platform. Do not assume §4.1 wins.

A cheap middle ground: give each lane a bounding cone over the buckets it owns and reject
candidates against that cone first — one dot product kills most candidates before the
per-bucket work.

---

## 5. Combining near and far field

**In revision 3 this section has almost nothing left in it, which is the point.** FMM's
U/V-list split makes the two fields disjoint by construction: the near field is exactly
the U-list, the far field is exactly V-list-and-above, and no surfel is in both.

The only residual is that a U-list occluder can block a V-list emitter. That is handled by
the per-bucket coverage blend in §4.1:

```
sum += (Lnear * cov + Lfar * (1 - cov)) * cosP * dw
```

evaluated in the tangent-frame direction of each bucket. Revision 1's scalar
`(1 - coverageFraction)` treated the far field as isotropic, which is wrong when a bright
cluster sits at +X and the occluders sit at -X. Revision 2 identified the problem and
prescribed a per-empty-bucket SH evaluation, but left the contradictory scalar in the code
listing and still discarded the far field entirely in partially-covered buckets. The
coverage blend fixes both, costs ~30 ALU per bucket, and needs no separate combination
step, no `coverageFraction`, and no L2P pass.

Note the SH is smooth, so this does *not* recover directional detail the far field never
had — it recovers the correct *masking* of the far field by near geometry, which is the
part that shows up as missing contact occlusion against distant light.

One known gap: only the winning surfel's coverage is tracked, so a bucket fully covered by
three partial discs still reads as partially covered and leaks far field. This is the same
limitation as §6.3's silhouette artifact and has the same fix (two-layer microbuffer).

### 5.1 Sky and environment

Empty buckets in an open scene have no far-field surfel contribution at all, because the
sky is not in the surfel set. Add the environment map's order-2 SH directly into every
leaf cell's `Lsh` during the L2L downsweep. It is a constant addend, it is cheap, and it
gets masked by near-field coverage automatically.

Without this, interiors near windows will be dark and you will misdiagnose it as a form
factor bug.

**But be aware the sky is the worst case for an unoccluded far field.** It is a
full-hemisphere emitter at effectively infinite distance, so a sealed room with 1 m walls
receives the full outdoor sky through them, masked only by whatever happens to sit in the
receiver's own 27-cell U-list. This will be your most visible leak by a wide margin, and
revision 2 filed it under a general disclaimer.

The cheap mitigation, which you should build alongside §5.1 rather than after shipping: a
scalar per-cell sky visibility, computed once during the upsweep by accumulating occupancy
along +Y (or against the dominant sky direction) through the grid, and used to attenuate
the env addend per cell. It is one float per cell and one multiply, it is wrong in
detail, and it turns "the room is lit like an open field" into "the room is a bit too
bright near the window", which is a completely different class of artifact. The principled
version is Ren et al.'s SH exponentiation (§11).

---

## 6. LOD

Two independent axes. Do not conflate them.

### 6.1 Camera distance -> microbuffer resolution

Pure screen-space error budgeting. A shadow boundary covering 3 pixels does not need a
64-bucket hemisphere.

```
pixelSize     = 2 * tan(fov/2) * distToCamera / screenHeight
bucketsWanted = clamp(64 * (surfelRadius / pixelSize), 4, 64)
```

Quantize to {16, 32, 64} and dispatch three separate kernels, sorting receivers by tier
first. Branching inside one kernel destroys occupancy; three specialized kernels do not.

At the lowest tier, skip the microbuffer entirely and use the closed-form L2P from §3.2 —
nine `Y_lm(n_P)`, 27 madds, ~50 ALU, no LDS, no atomics. Distant surfels do not need
resolved near-field occlusion.

**Caveat introduced by the per-cell dispatch.** Tier is a per-*receiver* property but the
dispatch is per-*cell*, and receivers in one cell are at nearly the same camera distance,
so in practice a whole cell shares a tier. Assign the tier per cell from its centre, sort
the active cell list by tier, and dispatch three ranges. Do not try to mix tiers within a
group.

### 6.2 Emitter distance -> handled by the FMM levels, not by resolution

Do **not** vary microbuffer resolution by emitter distance. The FMM level structure
already handles emitter distance: a source cell contributes at whichever level it becomes
well-separated, and its angular footprint at that level is by construction below the
resolution the local expansion carries. There is nothing left for buffer resolution to do.

This is the point where the design diverges from Radiance Cascades. RC trades angular
resolution against spatial resolution because it caches radiance in a **shared probe grid**
that many receivers sample. Your microbuffer is per-receiver and not reused, so there is
nothing to amortize into and the cascade trade does not apply.

The wrinkle: the FMM local expansion **is** a shared radiance cache — `Lsh` per cell is
sampled by every receiver in that cell, and after §4.1 it is literally loaded into LDS once
and read by all of them. So the far half of this design has RC's amortization property
while the near half does not. If you later want the full RC benefit, the change is to push
the near field into cell-level cached directional visibility too, at which point you have
something much closer to RC and should evaluate whether to just build that instead.

### 6.3 Partial bucket coverage

One winner per bucket means a surfel covering 30% of a bucket contributes 100% of its
solid angle. At 64 buckets your penumbra has at most 64 quantization levels and edges will
crawl as surfels migrate between buckets.

This matters more here than in the prior art: Christensen used microbuffers for
**indirect** light only, where the signal is low-frequency. Using it for **direct**
area-light shadows is a harder ask than the published work supports.

**The coverage-weighted single winner is now the default**, folded into §4.1's integrate
loop as `cov = saturate(PI * angR^2 / dw)`. It costs three ALU ops, it removes silhouette
over-brightening, and because it blends against the far field rather than against black,
it does not darken. It does not fix crawling.

Remaining options if crawling is unacceptable:

- **Two-layer microbuffer.** Keep the two nearest entries per bucket (two `atomicMin`
  passes, or a 64-bit pack now that you have 8-bit indices to spare) and blend by coverage.
  Roughly 1.7x pass 6 cost, fixes crawling well, and also fixes the multiple-partial-discs
  gap noted in §5.
- **Accept it and get direct light elsewhere.** If direct area shadows are the goal and
  you have shadow-map or ray budget, spend it there and let this system own indirect only.
  This is the conservative call and what the prior art actually validates.

---

## 7. Temporal amortization

**Mandatory on Steam Deck class hardware. Strongly recommended everywhere.**

Amortize pass 6 only. Run passes 3-5 (the FMM) in full every frame — they are cheap now,
and a partially-updated local expansion produces spatially inconsistent lighting that
reads as blotching, which is much worse than latency.

Update a rotating subset of receivers per frame. Prioritize by:

```
priority = screenCoverage * (1 + lightingDelta) * framesSinceUpdate
```

- Surfels visible and large on screen: every frame.
- Visible, small: every 2-4 frames.
- Offscreen but recently visible: every 8-16 frames.
- Far offscreen: every 32+ frames or on demand.

Blend new results into the stored irradiance with a temporal hysteresis of 0.05-0.2. Raise
the blend rate when a surfel's estimate changes sharply.

**Do not ping-pong the irradiance buffer.** Revision 1 and 2 both specified
`irradianceIn / irradianceOut` with a per-frame swap, *and* specified that only a subset of
receivers is updated per frame. Those two things are incompatible: a receiver skipped this
frame never writes `irradianceOut`, so after the swap it reads whatever was in that slot
two frames ago. At a 1/4 update rate three quarters of your surfels are reading stale
garbage every frame, and the symptom — a low-frequency flicker correlated with the update
rotation — is very easy to misread as temporal instability in the solver.

Use a **single buffer with an in-place read-modify-write**:

```
E_new = lerp(E_stored, E_computed, alpha)
```

Only updated receivers touch it; everyone else keeps their value. This is safe because
pass 6 is dispatched per cell and no two groups write the same receiver. It does mean P2M
reads irradiance that is being written in the same frame by a different pass — separate
the two with a barrier, and accept that the far field is built from a mix of ages, which
is fine and consistent. If you want strict determinism, double-buffer but **copy through**
the non-updated set.

**Be realistic about the latency.** At a 1/4 update rate with hysteresis 0.1, the
effective time constant is ~40 frames, or 0.66 s at 60 fps. Revision 1's claim of "no
ghosting" is too strong — because the input is noise-free you avoid the *variance*-driven
ghosting that TAA fights, but you do not avoid lag. Moving lights and moving receivers
will smear. The delta-driven blend rate boost fixes the worst of it at the cost of some
temporal instability; tune it as a tradeoff, not a free win.

**Multi-bounce convergence.** Iterating the solve gives a Neumann series. Because the far
field is unoccluded, every bounce over-estimates slightly, and in a closed bright scene
with high albedo the leaked energy compounds across iterations. `albedo < 1` keeps it
bounded but not necessarily at the right value. Cap the effective bounce count (a
per-surfel bounce counter, or simply a global energy clamp against the direct-only
solution) rather than letting it run to a fixed point.

---

## 8. Performance budget

Cost is dominated by pass 6. The far field is a rounding error.

**Far field (passes 3-5), 500K surfels, `h = 1.5s`, ~6000 leaf cells:**

| Pass | Work | Est. cost |
|------|------|-----------|
| P2M + M2M | 500K + ~7K cells | < 0.1 ms |
| M2L | ~7K cells x 189 x ~30 madd | 0.1 - 0.3 ms |
| L2L | ~7K cells | < 0.05 ms |

Under 0.4 ms on a Steam Deck for the entire far field, versus ~2-3 ms for revision 1's
Barnes-Hut traversal. Add ~0.3-1.5 ms for pass 2 with dynamic geometry. Note that the
separate L2P pass is gone; its work moved into pass 6 and grew, because it is now
evaluated per bucket rather than once per receiver.

**Pass 6 ALU model, per receiver, summed across the wave:**

| Stage | ALU |
|---|---|
| Trilinear `Lsh` interpolation (27 values, 8 taps) | ~220 |
| Candidate scatter, `K = 91` at ~35 ops each | ~3200 |
| Neighbour splat expansion | ~1500 |
| Integrate: 64 buckets x (SH eval ~30 + near-field ~20) | ~3200 |
| **Total** | **~8000-9000** |

That is roughly double revision 2's per-receiver ALU, in exchange for deleting a VRAM
round trip, the L2P pass, and the coverage-fraction heuristic. On an ALU-heavy design that
is the trade you want.

**Near field (pass 6)** — these figures are **explicitly untrusted** until §4.2 is
profiled. The ALU model above gives ~0.2-0.4 ms of raw FP32 on every target below, against
the estimates in the table. That gap is atomic serialization, LDS bank conflicts, and
occupancy, and it is the entire question.

| Target | FP32 | Surfels | K | Update rate | Est. pass 6 |
|--------|------|---------|---|-------------|-------------|
| Steam Deck (RDNA2, 8 CU) | ~1.6 TF | 150-250K | 64-96 | 1/4 | 3.5 - 5 ms |
| Radeon 680M | ~3.4 TF | 250-350K | 96 | 1/4 | 2.5 - 4 ms |
| RTX 3060 Mobile | ~10 TF | 500K | 128-192 | 1/2 | 1.5 - 2.5 ms |
| RTX 3060 Desktop | ~12.7 TF | 500K-1M | 192 | 1/2 | 1.2 - 2 ms |

Add 0.3-0.8 ms for pass 7 (screen apply).

**Verdict on the target range: yes.**

- **RTX 3060 (both):** comfortable. Full quality, 60 fps, room for two-layer microbuffers,
  and enough ALU headroom to evaluate §4.4's atomic-free variant seriously.
- **Radeon 680M:** workable at 1080p/30 or 720p/60. Bandwidth-limited more than
  ALU-limited, which is why §4.1's per-cell candidate reuse matters more here than
  anywhere else — it is the largest single reduction in this revision.
- **Steam Deck:** viable. Budget ~4 ms of a 33 ms frame. Ship temporal amortization, 32
  buckets for most receivers, `K <= 96`, and under ~250K surfels.

The APUs remain the real constraint. Both share memory bandwidth with the CPU, so profile
bandwidth before ALU on those parts — the ALU-heavy design is the right call there for
exactly that reason, and §4.1's `h`-independent traffic is what makes the ALU/bandwidth
balance a knob you actually control.

---

## 9. Validating against path-traced ground truth

Separate integration error from representation error, or you will spend weeks tuning the
wrong knob.

**Step 0 — Check the `/PI`, then check the whole chain analytically.** Before anything
else, run §3.2's single-surfel identity as a unit test: one emitter, one receiver, known
geometry, and assert `E = a * L_out * cos_E * cos_P / r^2` through P2M, M2L, and
evaluation. It catches every SH normalization convention error at once. Then compare a
disc-to-disc case against the analytic form factor. If you are off by pi, 3.14x, or 9.87x,
stop and fix it — this error is *consistent* across both the near and far paths, so it will
not show up as a discrepancy between them, only against an external reference.

**Step 1 — Establish the ceiling.** Render your reference path trace of the *same
emissive geometry*, direct-only, Lambertian-only. Then run your solver with the FMM
disabled (brute-force all-pairs) and microbuffer resolution cranked to 256+ buckets.
Converge it. The remaining difference is **surfel representation error** — disc
approximation, gaps, density. No amount of algorithm tuning removes it.

**Step 2 — Reintroduce the microbuffer resolution you plan to ship.** The delta is
quantization error. Expect blocky shadow boundaries, softened contact shadows, and — if
you disabled the coverage blend — over-bright silhouettes (§6.3).

**Step 3 — Reintroduce the FMM.** **Expect a visible delta, and do not treat it as a
bug.** Revision 2 told you to expect near-zero here and to go hunting if the image moved.
That advice followed from the incorrect accuracy claim §0 retracts. With a 25-degree
effective opening angle you should see a few percent, concentrated in mid-range soft
shadows. Investigate only if it is large, and check in this order: the source-side
intensity SH (evaluate `evalSH(Ilm, w)` against direct summation over the cell's surfels
for a few directions), the interpolation of `Lsh`, the band scale factors `ahat`, then the
deringing window.

**Step 4 — Vary SH order.** Run with bands 0-1 (4 coeffs) and bands 0-2 (9 coeffs). The
difference should be a few percent and no more. If it is large, your scene has a
dominant near-delta emitter in the far field and you need the conditional deringing
window (§3.2) or a smaller `h`.

**Step 5 — Turn on tier B.** If the flux-moment shift moves the image noticeably, your
cells are too large relative to the intra-cell brightness variation and `h` should come
down; if it does not, stay on tier A.

**Error signature to expect overall:** darkening in concave corners, contact shadows
softer and shorter than reference, leaks through geometry thinner than one cell, sky
leaking into enclosed spaces unless §5.1's occlusion term is in, quantized penumbra edges,
and slight over-softening of distant shadows. Energy should be within a few percent
globally.

**Useful diagnostics:** render `cov` per receiver averaged over buckets, the empty-bucket
count, and the per-cell `K`. Large empty-bucket counts in enclosed scenes means `h` is too
small. `K` varying by more than ~4x across occupied cells means your surfel density
control (§2.1) is not working, and pass 6 will have terrible load balance *between groups*
— which the per-cell dispatch makes worse than the per-receiver one did, since a group's
runtime is now proportional to `K * receiverCount`. Consider splitting oversized cells
across multiple groups.

---

## 10. Implementation order

Do not build this in dependency order. Build it in **debuggability order**.

1. **Brute-force all-pairs, no occlusion, no hierarchy.** O(N^2), slow, but unambiguously
   correct for form factors. Validate energy against a path-traced reference with
   occlusion disabled. Do not proceed until this matches. Step 0 of §9 lives here.
2. **Add the microbuffer with a brute-force candidate list** (still all-pairs). Now you
   have correct occlusion. Validate shadows. This is the algorithmic core; everything
   after is acceleration.
3. **Add the sparse grid and the U-list walk, per receiver.** Near field only, far field
   still brute-force. Validates `neighbourBucket`, cell lookup, and the packed atomic
   independently of any FMM code. Build the naive per-receiver version first — it is
   simpler, and step 4 is where you make it fast.
4. **Restructure to per-cell dispatch (§4.1).** Pure performance, no image change. Assert
   bit-identical output against step 3 before and after; if the image moves, your
   candidate flattening dropped or duplicated a surfel.
5. **Add P2M/M2M with a trivial M2L** — all non-U cells, direct summation, no interaction
   lists. Validates the intensity SH and normalization against brute force while still
   being O(N^2) in cells. §9 step 0's analytic identity is the test.
6. **Add the real V-list M2L and L2L.** Now it is O(N). Render a per-cell count of "cells
   contributing" and verify it equals the total cell count for every target.
7. **Add trilinear `Lsh` interpolation.** The delta is the cell-boundary discontinuity
   disappearing. If you skip this you will see a grid artifact and blame step 6.
8. **Add temporal amortization**, with the in-place blend of §7, not a ping-pong.
9. **Add camera-distance LOD tiers (§6.1) and the sky occlusion term (§5.1).**
10. **Iterate for multi-bounce**, with the energy cap of §7.

Steps 1, 2, 4 and 6 are the ones worth being slow about. Two failure modes deserve
naming because they produce the *same* symptom — a subtle brightness modulation on a grid
pattern, easily misattributed to surfel density:

- Step 6, if a small fraction of cells are double-counted or dropped from the V-list.
- Step 7 being skipped, since a constant-per-cell local expansion is discontinuous at cell
  boundaries by construction.

Diagnose them apart by rendering the far field alone with the near field disabled: list
errors move when you change the interaction-list code, interpolation errors do not.

---

## 11. Prior art worth reading

- **Christensen, "Point-Based Approximate Color Bleeding"** (Pixar tech memo, 2008). The
  closest published relative for the near-field half — per-receiver microbuffer
  rasterization of a point cloud. Shipped in production. Read before step 2. Note it uses
  microbuffers for indirect only; see §6.3.
- **Ramamoorthi & Hanrahan, "An Efficient Representation for Irradiance Environment
  Maps"** (SIGGRAPH 2001). The source of the `ahat` coefficients, used here **twice** —
  once at the emitter, to project a Lambertian cluster's intensity distribution (§3.2), and
  once at the receiver, for the cosine convolution. The `ahat_3 = 0` result bounds the
  target-side truncation only; see §0.
- **Greengard & Rokhlin (1987)** for FMM proper. Worth reading specifically to see what
  this method does *not* have: real translation operators. The Laplace kernel admits exact
  M2M/M2L/L2L translation at any expansion order `p`, which is where classical FMM's
  accuracy knob comes from. Radiance transport has no such theory, so here the source side
  is a monopole (tier A) or monopole-plus-shift (tier B) and there is no clean way to raise
  `p`. `h` is your only real accuracy knob. That is the honest cost of the analogy.
- **Barnes & Hut (1986)** for context on what you give up (density adaptivity, and a
  tunable opening angle that can be made tight) and gain (a stencil, and no divergence).
- **Ren et al., "Real-time Soft Shadows in Dynamic Scenes using Spherical Harmonic
  Exponentiation"** (SIGGRAPH 2006). Log-space visibility makes blockers additive, which
  is the only known route to hierarchically aggregable visibility. Low-frequency only, so
  it suits the far field. **This is the natural next extension** if far-field leaks become
  the dominant artifact — and per §5.1 they probably will, through the sky term. It would
  let you attenuate `Lsh` during the downsweep rather than leaving the far field fully
  unoccluded.
- **EA SEED, "Global Illumination Based on Surfels"** (2021). For §2.1, which this
  document still underspecifies relative to its difficulty.

**On the FMM/visibility boundary:** classical FMM's O(N) depends on M2L building a local
expansion valid across a target cell, which requires the field to be smooth over that
cell. The unoccluded radiance field satisfies this. Visibility does not; a shadow boundary
is precisely the discontinuity that breaks admissibility. So you get **O(N) for radiance**
and an irreducible O(N*k) near-field visibility residual with `k` bounded by U-list
occupancy. The overall method is O(N) with a near-field constant.

Note the complexity claim is about *scaling*, not accuracy. The far field is O(N) at a
fixed and fairly coarse approximation order, and unlike classical FMM you cannot buy
accuracy back by raising `p`. Do not let the O(N) claim do work it is not entitled to do.

---

## Appendix A — Changelog from revision 2

**Correctness fixes:**

- **Normal cone removed; source-side representation is now an SH radiant intensity
  vector** (§1.2, §3.2). `coneFacingFactor` is not linear in the child cells, so revision
  2's cone union at M2M produced an incorrect aggregate emitter, biased bright, worst for
  cells containing a surface and its opposite face. Intensity superposes exactly, so M2M
  is now plain summation. Uses the same `ahat` constants as L2P.
- **Expansion point moved from the luminance-weighted centroid to the cell centre**
  (§3.2). The drifting centroid could put two V-list cells `0.26h` apart despite a `2h`
  stencil guarantee, which is why revision 2 needed the `+ S.area` soft clamp. Centre
  expansion restores `r >= 2h` unconditionally; the clamp is deleted along with the bias
  it was introducing on the nearest and brightest sources.
- **Ping-pong irradiance buffer replaced with an in-place blend** (§1.1, §7). Ping-pong
  plus partial temporal updates meant three quarters of surfels read two-frame-old data
  every frame at a 1/4 update rate. Symptom is a flicker correlated with the update
  rotation, easily misread as solver instability.
- **`(1 - coverageFraction)` scalar removed entirely** (§4.1, §5). Revision 2's code
  listing still used it while its prose prescribed per-empty-bucket SH; neither handled
  partially covered buckets, which discarded the far field wholesale in any direction with
  a winner. Replaced by a single symmetric per-bucket blend.
- **Cell-boundary discontinuity named and fixed** (§3.2, §4.1). Tier-A L2L is a value copy,
  not a Taylor re-centring, so the far field is piecewise constant per cell. Revision 2's
  §10 warned about the resulting grid artifact but attributed it solely to interaction-list
  bugs. Fixed by trilinear interpolation of `Lsh` at evaluation time, which is nearly free
  because the 8 taps are already in LDS.
- **Multi-bounce energy cap** added (§7).

**Corrections to claims:**

- **The far field is not more accurate than revision 1's Barnes-Hut. It is much faster.**
  (§0, §3.2, §9 step 3.) Revision 2 cited `ahat_3 = 0` as bounding total error at ~1%.
  That result bounds the *target-side* expansion only. The source side is a monopole with
  a ~25-degree effective opening angle, versus revision 1's `theta_open = 0.02`. The trade
  is still correct; the justification given for it was not, and it would have sent you
  bug-hunting at §9 step 3.
- **§1.3 reframed as a three-way balance.** `h` sets near-field ALU (`~q^3`), M2L cost
  (`~q^-3`), *and* far-field accuracy. Revision 2 derived it from `K` alone.
- **Unconditional Hanning deringing downgraded** (§3.2). Attenuating band 2 to 25%
  everywhere is a permanent contrast loss to fix a rare artifact. Now conditional on a
  per-cell dominance ratio, with a zero-cost negative clamp as the baseline.

**Performance:**

- **Pass 6 dispatches one workgroup per cell, not per receiver** (§4.1). Revision 2's inner
  loop ran ~27 iterations at ~10% lane occupancy. Flattening the U-list into LDS once per
  cell gives full lane utilization in the hot loop and makes candidate traffic `~27 N`,
  independent of `h`, versus `~27 N q^3` before. Biggest single win on the APU targets.
- **64-bit LDS atomics no longer required** (§4.1). The winner's index only has to address
  within the U-list, so `K <= 256` fits in 8 bits and the entry packs into a 32-bit
  `InterlockedMin`. Removes §4.4's stated reason to exist as a compatibility path.
- **Separate L2P pass deleted** (§2, §3.2). Folded into pass 6 as a per-bucket evaluation,
  which is both more correct and cheaper than a separate pass plus a blend heuristic. A
  standalone L2P kernel survives only for the lowest LOD tier.
- **Cell struct is fp32 and unpacked** (§1.2). At ~7000 cells the whole structure is 1.5 MB;
  revision 2's fp16-versus-RGB9E5 analysis was optimizing a rounding error, and `Ilm`
  overflows fp16 for bright large-area emitters.
- **§4.4 (lane-per-bucket) promoted from fallback to a real candidate** on high-ALU
  targets, since the candidates now live in LDS and the redundancy is pure ALU.

**Gaps filled:**

- Surfel radius must be clamped against `h` or oversized discs leave holes the 3x3 splat
  cannot cover (§2.1, §4.2).
- Sky visibility term for the env addend (§5.1); an unoccluded full-hemisphere emitter at
  infinity is the worst case for this method and will be the dominant leak.
- LOD tier is a per-cell property now that dispatch is per-cell (§6.1).
- Inter-group load balance is now proportional to `K * receiverCount`, making §2.1's
  density control more load-bearing than before (§9).

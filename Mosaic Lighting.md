\# MOSAIC — Micro-rendered Object-Space Amortized Irradiance Cache

\#\# 1\. Pitch

MOSAIC stores indirect light in \*\*object-space surfel sets that ship with the mesh asset\*\*, and estimates transport by \*\*micro-rasterizing a camera-centered clipmap of surfel clusters\*\* into tiny per-surfel octahedral buffers rather than tracing rays. The hemisphere integral is split by frequency: the low-frequency bulk is captured deterministically by a 16×16 micro-buffer splat pass (cheap, alias-free, no BVH, no RT cores), while the high-variance bright tail is captured by \*\*world-space ReSTIR reservoirs\*\* that importance-sample distant emissive clusters the micro-buffer would undersample. Direct lighting on surfels is re-evaluated every frame; only the bounce term is amortized, so light response latency is one frame per bounce instead of a convergence window. Target: 5.5–6 ms at 1080p on a GTX 1660 / RX 580, degrading to \~3 ms at 900p on GTX 1050-class parts.

\---

\#\# 2\. Core data structures

\*\*A. Per-asset surfel sets (object space).\*\*  
Each mesh carries a Poisson-disk surface sampling generated once at import (or on first stream-in via area-weighted triangle sampling plus spatial-hash dart throwing). Four LODs at 4 cm / 8 cm / 25 cm / 1 m spacing. Per surfel, 48 bytes:

| Field | Bits |  
|---|---|  
| Position (object space, quantized to instance AABB) | 3×16 |  
| Normal (octahedral) | 16 |  
| Radius, LOD, flags | 16 |  
| Albedo (RGB565) | 16 |  
| Skin indices/weights (skinned meshes only) | 4×8 \+ 4×8 |  
| Irradiance L1 SH (DC as RGB9E5, 3 coeffs as snorm16 relative to DC) | 80 |  
| Specular lobe (peak dir oct16, peak RGB9E5, sharpness u8) | 56 |  
| Reservoir (sample cluster ID, target pdf, W, M) | 64 |

Object space is the load-bearing choice: a rigid dynamic object's cache is never invalidated, only transformed. Skinned meshes skin their surfels in the same compute dispatch as the mesh, and there are far fewer surfels than vertices at coarse LOD.

\*\*B. Cluster clipmap (rebuilt every frame).\*\*  
Three camera-centered 64³ uniform grids at 0.5 m / 2 m / 8 m cell size (32 m / 128 m / 512 m coverage). Each non-empty cell aggregates its surfels into one cluster: centroid, normal cone (axis \+ half-angle), total projected area, outgoing radiance L1 SH. Built with atomics, compacted into a per-level dense cluster list. This replaces PBGI's octree. It is sort-free, rebuilt from scratch each frame (so dynamic geometry is free), and gives O(1) LOD selection by world distance, which is exactly the query micro-rendering wants.

\*\*C. Occupancy clipmap.\*\* 3 × 128³ × 8-bit coarse opacity, used only to shadow the C2 (far-field) cluster contributions. This is the one voxel-ish component and it carries no radiance.

\*\*D. Surfel index grid.\*\* Hash grid at C0 resolution mapping cell → surfel ID list, used for near-field splatting and for the screen-space gather.

Why this suits low-end hardware: no BVH build, no acceleration-structure refit, no HW RT, no per-pixel ray dispatch. The inner loop is a splat with an \`atomicMin\` depth test, which is ALU-bound on a shared candidate list with very high L1 hit rates.

\---

\#\# 3\. Pipeline stages

\*\*S0 — Instance assembly.\*\* Cull instances within 256 m, select surfel LOD by projected area, skin where needed, transform to world, append to the live surfel buffer. Hard cap \~250k live surfels with LRU eviction by screen importance.

\*\*S1 — Clipmap build.\*\* Scatter surfels into the three cascades, aggregate cluster attributes, compact cell lists. Also rasterize the occupancy clipmap from instance proxy geometry.

\*\*S2 — Direct lighting on surfels.\*\* Every live surfel, every frame. Clustered light list \+ shadow map taps (1 tap for CSM, 1 for the nearest 2 local lights). Writes to the surfel's \*outgoing\* radiance \= direct \+ last-frame irradiance × albedo. This feedback loop is where multi-bounce comes from, one bounce per frame.

\*\*S3 — Update scheduler.\*\* Priority \= screen importance (from a 1/8-res surfel coverage pass with atomics) \+ age \+ temporal radiance variance \+ disocclusion flag. 256-bucket LDS counting sort, select top 8k. On-screen surfels get \~25% of the budget, so visible surfels refresh every 4–8 frames; the tail refreshes in \~30 frames.

\*\*S4 — ReSTIR tail resample.\*\* For each selected surfel: 8 RIS candidates drawn from the emissive cluster list proportional to \`area × luminance / d²\`, temporal reuse from the surfel's own reservoir, spatial reuse from 3 neighbors in the same clipmap cell (normal-cone \+ plane distance rejected). Output: 1–2 explicitly-weighted bright clusters.

\*\*S5 — Micro-render gather.\*\* One 64-thread workgroup per selected surfel renders a 16×16 hemispherical octahedral micro-buffer:  
\- Near field (\< 1 m): splat \*individual\* surfels from the index grid.  
\- Mid field: splat C0/C1 clusters as oriented disks.  
\- Far field: splat C2 clusters, attenuated by an 8-step occupancy march.  
\- Reservoir samples from S4 are splatted with their RIS weight.  
\- Empty texels sample the sky cubemap.  
Depth test via packed \`atomicMin(depth\<\<24 | index)\`, then a resolve pass integrates cosine-weighted. Typical 300–600 splats per surfel.

\*\*S6 — Projection and filtering.\*\* Project the micro-buffer to L1 SH (diffuse) and extract the dominant lobe by first-moment of the luminance-weighted directions plus a sharpness from the second moment (specular). Temporal EMA with SVGF-style variance-adaptive α ∈ \[0.05, 0.5\], then a spatial pass over same-cell neighbors with plane and normal weights.

\*\*S7 — Screen integration.\*\* Half-res gather: per pixel, fetch up to 8 surfels from the index grid, weight by \`w \= max(0, dot(n, ns)) · saturate(1 − |plane\_dist| / r) · saturate(1 − d/r)\`, normalize, fall back to the enclosing C0 cluster SH when total weight is low. Bilateral upsample to full res on depth \+ normal. Multiply by full-res albedo and a cheap bent-normal AO for contact detail.

\*\*S8 — Glossy resolve.\*\* Roughness \> 0.4: surfel SH \+ lobe directly. Roughness 0.1–0.4: SSR with the surfel lobe as the miss fallback. Roughness \< 0.1: SSR, then parallax-corrected cubemap array, then lobe.

\---

\#\# 4\. Hard problems

\*\*Light leaking.\*\* Irradiance lives on surfaces with normals, not in a volume, so there is no probe-interpolation leak class at all. A pixel on the lit side of a wall cannot pick up the dark-side surfel because the gather rejects on signed plane distance. The micro-buffer depth test provides explicit occlusion for near and mid field. The residual risk is C2 clusters aggregating over 8 m; that is why C2 alone is occupancy-marched, and its contribution is clamped to a low-frequency term.

\*\*Temporal lag / ghosting.\*\* Decoupling matters more than filtering here. Direct light on surfels is exact and per-frame (S2), so flipping a light on propagates to first-bounce next frame and second-bounce the frame after. Only the multi-bounce tail settles slowly, which is perceptually forgiving. On top: variance-clamped history, plus a hard history reset when a surfel's direct term changes by \> 2× (catches shadow sweeps and light toggles).

\*\*Disocclusion.\*\* The cache is world-space and always warm behind occluders, since off-screen surfels still receive low-priority updates. Newly disoccluded surfels are flagged and jump the priority queue for 8 frames, seeded from their enclosing cluster SH rather than from black.

\*\*Specular / glossy.\*\* The single directional lobe extracted in S6 costs nothing extra because the micro-buffer is already there. It handles blurry indirect reflections (a wall lit by a window, a floor reflecting a neon sign) which is where SSR fails hardest. It cannot do sharp reflections, hence the SSR/cubemap ladder.

\*\*Scale.\*\* Cascades do the work. Interiors sit almost entirely in C0/C1 with 4–8 cm surfels. Open worlds push most content into C2 with 25 cm–1 m surfels and rely on sun-lit cluster SH for the far field. Beyond 512 m, fall back to a single sky/ground SH plus CSM-derived ambient bounce. Surfel sets stream in and out with mesh LODs; LOD transitions crossfade over 8 frames using the parent LOD's radiance so nothing pops.

\---

\#\# 5\. Budget (1080p, GTX 1660, 60 fps target)

| Stage | ms | % of GI |  
|---|---|---|  
| S0 Instance assembly / skinning | 0.35 | 6% |  
| S1 Clipmap \+ occupancy build | 0.60 | 10% |  
| S2 Direct light on surfels | 0.50 | 9% |  
| S3 Scheduler / priority sort | 0.20 | 3% |  
| S4 ReSTIR resample | 0.40 | 7% |  
| S5 Micro-render gather (8k × 16²) | 2.20 | 38% |  
| S6 Projection \+ temporal \+ spatial filter | 0.35 | 6% |  
| S7 Half-res screen gather \+ upsample | 0.90 | 15% |  
| S8 Glossy resolve | 0.30 | 5% |  
| \*\*Total\*\* | \*\*5.80\*\* | |

Scaling knobs, in the order I would pull them: surfel update budget 8k → 4k → 2k; micro-buffer 16² → 12² → 8²; screen gather half-res → quarter-res; drop ReSTIR spatial reuse; C0 cell 0.5 m → 1 m; drop C2 entirely and substitute sky SH. A GTX 1050 at 900p/30 fps lands around 3.0 ms with 2k updates and 8² buffers. Memory: \~12 MB live surfels, \~25 MB clusters, 6 MB occupancy, 40–60 MB asset surfel sets, total 85–105 MB.

Scaling up: if HW RT exists, replace the near-field splat loop with actual rays inside the C0 radius and keep micro-rendering for mid/far field. RT becomes an accelerator for one sub-stage, never a dependency.

\---

\#\# 6\. Comparison

| | MOSAIC | DDGI | Lumen (SW) | Full PT |  
|---|---|---|---|---|  
| Diffuse quality | High, no volume leak, surface-accurate | Medium, probe leak, coarse detail | High | Reference |  
| Glossy indirect | Blurry only (single lobe) \+ SSR ladder | Poor | Good | Reference |  
| Perf (1660-class) | \~5.8 ms | \~3–4 ms \+ ray cost | Not viable (\~20 ms+) | Not viable |  
| Memory | 85–105 MB | 30–60 MB | 300–600 MB | Scene BVH only |  
| HW RT required | No | Effectively yes | No (SDF), but heavy | Yes |  
| Dynamic geometry | Rigid \+ skinned free; deforming needs fallback | Good | Good | Full |  
| Light response latency | 1 frame direct, 1 frame/bounce | Several frames | Several frames | Instant |  
| Small bright emitters | Handled via ReSTIR tail | Poor | Medium | Full |

\---

\#\# 7\. Prior work and what is recombined

Builds directly on Ritschel et al. 2009 \*Micro-Rendering for Scalable, Parallel Final Gathering\* and Christensen 2008 \*Point-Based Approximate Color Bleeding\*; Ward et al. 1988 irradiance caching; Ritschel et al. 2008 \*Imperfect Shadow Maps\*; Keller 1997 \*Instant Radiosity\*; Halén & Stachowiak 2021 surfel-based GI (SEED); Bitterli et al. 2020 ReSTIR and Boissé 2021 world-space spatiotemporal reservoirs; Schied et al. 2017 SVGF for variance-adaptive temporal filtering; Silvennoinen & Lehtinen 2017 sparse radiance probes; Tanner et al. 1998 clipmaps.

New or non-obvious:

1\. \*\*Clipmap-of-clusters replaces the PBGI octree.\*\* Sort-free, rebuilt per frame, dynamic-geometry-native, O(1) LOD selection.  
2\. \*\*Frequency split of the hemisphere.\*\* Deterministic micro-rendering owns the low-frequency bulk; ReSTIR owns the bright tail. Existing work uses one or the other for all transport.  
3\. \*\*Object-space asset-resident surfel sets with skinning\*\*, so rigid and skinned motion never invalidates cache.  
4\. \*\*Per-frame exact direct light on surfels with amortized indirect\*\*, decoupling response latency from convergence rate.  
5\. \*\*Free specular lobe\*\* extracted from the micro-buffer that already exists.

\---

\#\# 8\. Failure cases and fallbacks

| Failure | Fallback |  
|---|---|  
| Thin foliage below surfel spacing (leaks, missing occlusion) | Two-sided translucent surfel flag \+ per-instance density boost; below a threshold, substitute a single per-instance SH probe for the whole plant |  
| Tiny bright emitters (candle, small neon) missed by 16² buffer | Register emitters under \~0.05 m² into an explicit analytic light list sampled by the ReSTIR path, bypassing clusters |  
| Mirror surfaces | SSR → parallax cubemap → lobe ladder; accept degradation, this is not a mirror-capable technique |  
| Cold cache on fast camera teleport | Priority burst (raise budget to 32k for 4 frames), seed from C1/C2 cluster SH, pre-warm along camera velocity vector |  
| Deforming non-skinned geometry (cloth, destruction, fluids) | Mark instance \`volatile\`; either re-sample surfels from the live vertex buffer every 4 frames (linear cost, fine for a handful of objects) or swap to a per-instance probe grid |  
| Transparent surfaces | Gather from the enclosing C0 cluster SH, no surfel representation |  
| Far-field light beyond 512 m (sun-lit valley) | World-level SH cascade \+ CSM-derived ambient bounce approximation |  
| Energy drift from the feedback loop | Clamp albedo to 0.9, clamp per-bounce gain, and cap the feedback chain at 4 effective bounces by decaying stored irradiance 2% per update |  
| Surfel budget exhaustion in dense scenes | LRU eviction by screen importance; evicted regions read the cluster SH, which visibly softens but does not go black |

The honest weak spot is sharp specular. MOSAIC is a diffuse-and-rough-glossy solution with a screen-space ladder bolted on for mirrors, and I would not try to fix that inside this architecture.

\# MOSAIC Addendum — Direct Lighting, Area Lights, and Soft Shadows

You're right that I hand-waved S2. "One CSM tap plus two shadow map taps" is fine for feeding the bounce cache but it is not a direct lighting solution, and if area lights and emissive textures are the primary light sources then direct lighting \*is\* most of the image.

The core move: \*\*separate the radiance term from the visibility term.\*\* LTC gives you exact, noise-free, analytic soft area-light shading for the unshadowed component. Visibility is a separate, much lower-frequency signal that gets its own cheaper estimator. This split (Heitz et al. 2016, and the standard \`LTC × V̄\` approximation) is what makes area lights affordable without RT cores. It is technically wrong — it decorrelates shading from occlusion — but the error is a slight over-softening near contact, which reads as acceptable and is patched by contact shadows.

\---

\#\# 1\. Emitter taxonomy

Everything that emits gets classified at import or at instance spawn into one of four tiers. Tier determines the estimator.

| Tier | What | Radiance estimator | Visibility estimator |  
|---|---|---|---|  
| \*\*T0 Sun\*\* | Directional, 0.53° disk | Analytic disk NdotL | PCSS on CSM |  
| \*\*T1 Primary area\*\* | Fitted rect/disk/tube, ≥ \~0.2 m², budget 8–16 per view | LTC, analytic | Dedicated shadow map \+ blocker search |  
| \*\*T2 Secondary emitters\*\* | Fitted proxies below the T1 budget, or beyond \~25 m | LTC, analytic | Occupancy clipmap cone march |  
| \*\*T3 Emissive residue\*\* | Everything left: small texels, distant glow, complex emissive detail | Cluster clipmap (the existing S1 path) | Already occlusion-tested by the micro-buffer |

The important structural property is that T3 is not a special case. Emissive surfels already sit in the clipmap and already carry outgoing radiance, so "faint diffuse emissive glow" falls out of the GI system for free. The tiering only exists to promote the emitters that need \*sharp\* treatment out of the low-frequency path.

\---

\#\# 2\. Emitter proxy extraction (offline / import time)

Emissive textures are the hard part, because "the light" is an arbitrary blob of texels, not a shape.

1\. Rasterize the mesh's emissive texture into UV space, threshold at some luminance floor (say 0.02 × peak).  
2\. Connected-component label the thresholded mask.  
3\. For each component, fit an oriented rectangle in 3D (PCA on the surface positions of member texels, then take the two dominant axes). Fit quality \= fraction of component area covered by the rect. If coverage \< 0.6, split the component along its major axis and retry, to a depth of 3\. If it still doesn't fit, demote all of it to T3.  
4\. Store per proxy: center, two half-axes, normal, area, average emitted radiance (area-weighted mean of member texels), plus a 2-coefficient cosine-power directionality fit if the material is anisotropic.  
5\. Merge proxies whose planes are coincident within 5° and 2 cm.

A neon sign becomes 5–20 rects. A glowing window becomes one rect. A dense particle-emissive detail texture fails the coverage test and goes to T3, which is correct — you don't want 400 tiny analytic lights.

For \*\*textured\*\* area lights (a TV, a stained-glass window, a video wall), keep the emissive texture and build a mip chain in linear space. LTC handles this: after the LTC transform, you evaluate the light's clipped polygon in its own 2D space, compute the covered area, and pick a mip level by that area (Heitz's filtered importance sampling for LTC). One texture fetch. A TV in a dark room correctly casts colored, spatially varying light.

\*\*Portals.\*\* Windows and doorways get authored (or auto-detected from hole-in-wall geometry) as portal rects whose radiance is the sky cubemap sampled through the portal solid angle. Interiors then get sun-and-sky through openings as a T1 area light rather than relying on the bounce cache, which is the single biggest quality win for indoor scenes.

\---

\#\# 3\. Runtime pipeline changes

Insert between S2 and S7:

\*\*S2a — Emitter list build.\*\* Cull proxies to the view frustum plus a 20 m margin. Score each by \`radiance × area × solidAngleAtCameraOrCluster\`. Top 16 → T1, given a shadow map slot. Remainder → T2. Emissive surfels not covered by any proxy already live in the clipmap as T3, no work needed.

\*\*S2b — T1 shadow map render.\*\* 16 slots in a 4096² atlas, tile size adaptive by screen importance (512² down to 128²). Rects use a single perspective frustum from the rect center with FOV covering the receiving volume; this is not correct for an area light (an area light has no single center of projection) but the blocker-search step below recovers the penumbra. Tube and disk lights use the same. Render cost: shared with the existing shadow pass, and these are re-rendered on a rotating schedule — static emitter \+ static geometry means the tile is cached and skipped entirely. Typically 2–4 of 16 tiles redraw per frame.

\*\*S2c — Occupancy prepass.\*\* Already built in S1; no extra cost, just noting T2 depends on it.

\*\*S3a — ReSTIR DI (per pixel, half res).\*\* Reservoirs over the combined T1+T2 emitter set. 8 candidates by \`radiance × area / d²\` with a normal-cone reject, temporal reuse with the standard M-clamp at 20, spatial reuse from 4 neighbors. Output: 2 emitters per pixel carrying explicit weights. This exists because with 40+ T2 proxies in a shop interior, evaluating all of them per pixel is wasteful, and evaluating a fixed subset produces spatial discontinuities. ReSTIR makes the selection stochastic-but-coherent.

Note: this is \*selection\* only. Once selected, the emitter is shaded with full analytic LTC, not with a sampled point. So there is no shading noise, only a small amount of selection noise in the visibility term, which the half-res \+ bilateral upsample absorbs.

\*\*S7a — Direct shading.\*\* For each pixel, for each of T0 and the 2 reservoir emitters:

\`\`\`  
L\_direct \+= LTC(rect, V, N, roughness) \* albedo\_or\_spec \* V\_term  
\`\`\`

\---

\#\# 4\. The visibility ladder

\`V\_term\` is assembled from up to three sources, cheapest first, taking the minimum:

\*\*(a) Screen-space contact shadow.\*\* 8 steps, 0.5 m max, quarter-res depth. Owns the first \~30 cm. This is the term that fixes the LTC×V decorrelation artifact at contact points, and it's why the split is tolerable.

\*\*(b) T1: PCSS blocker search on the emitter's shadow map.\*\*  
\- Blocker search: 8 taps over a search radius proportional to the light's angular size at the receiver, \`r\_search \= lightHalfWidth \* (d\_receiver \- d\_near) / d\_receiver\` in shadow-map UV.  
\- Penumbra width from average blocker depth: \`w\_pen \= lightWidth \* (d\_recv \- d\_blocker) / d\_blocker\`.  
\- 12-tap Poisson PCF at \`w\_pen\`, rotated per pixel by an interleaved-gradient noise, denoised by the existing temporal filter.  
\- Clamp \`w\_pen\` to 64 texels so a large light close to a wall doesn't blow the tap radius out.

This gives genuine penumbra that widens with distance from the contact point, which is the whole visual signature of area lighting. 20 taps per T1 emitter, at half res, for 2 emitters per pixel.

\*\*(c) T2 and long-range: occupancy clipmap cone march.\*\*  
March a cone from the shading point toward the emitter centroid, cone half-angle \= the emitter's angular radius. 12–16 steps, sampling the occupancy clipmap at a mip selected by cone radius, accumulating \`1 \- alpha\` with correlation-aware blending. Coverage is the visibility term directly — a cone through a voxel field naturally produces a soft, penumbra-like falloff, and the aperture is physically tied to the light's size, so a big light gives a soft shadow and a small one gives a tight shadow \*for the right reason\*.

Resolution is the limitation: 128³ at 0.5 m in C0 means it will not resolve a chair leg's shadow. That's acceptable for T2, which by construction is the dimmer/farther set, and the contact shadow covers the near field.

\*\*T0 sun.\*\* PCSS on the CSM with the search radius derived from the 0.53° solar disk: \`r \= 0.0093 \* (d\_recv \- d\_blocker)\` in world units. At 2 m occluder distance that's \~1.9 cm of penumbra — small, which is why plain PCF often looks acceptable and why sun soft shadows are the cheapest win here. Add a per-cascade blocker search of 6 taps and 16-tap PCF. Under overcast/sky-dominant conditions, widen the effective disk by an authored \`sunAngularRadius\` multiplier; art direction wants this control anyway.

\---

\#\# 5\. Surfel-side direct lighting (S2, revised)

The cache does \*\*not\*\* need penumbra accuracy. Indirect light is integrated over a hemisphere, which is a low-pass filter — a hard shadow boundary in the bounce contributes nearly the same irradiance as a soft one. So S2 runs a deliberately cheaper version:

\- T0: 1 CSM tap, no PCSS.  
\- T1: LTC unshadowed × 4-tap PCF, no blocker search.  
\- T2: LTC × cone march at 6 steps instead of 16\.  
\- T3: nothing (already in the clipmap).  
\- Emitter selection: the surfel's own ReSTIR reservoir (S4) already does this; reuse it rather than running a separate DI pass.

Costs roughly 0.65 ms instead of 0.50, and unlike the pixel path it runs on all live surfels including off-screen ones — which is precisely how an emissive sign behind the camera still illuminates the visible wall.

\---

\#\# 6\. Revised budget (1080p, GTX 1660\)

| Stage | ms |  
|---|---|  
| S0–S1 assembly \+ clipmaps | 0.95 |  
| S2 surfel direct (LTC \+ cheap vis) | 0.65 |  
| S2a emitter list build | 0.10 |  
| S2b T1 shadow atlas (amortized, \~3 tiles/frame) | 0.55 |  
| S3 scheduler | 0.20 |  
| S3a ReSTIR DI (half res) | 0.45 |  
| S4 surfel ReSTIR | 0.40 |  
| S5 micro-render gather | 2.20 |  
| S6 projection \+ filter | 0.35 |  
| S7 indirect screen gather | 0.90 |  
| S7a direct shading: LTC \+ PCSS \+ cone march | 1.30 |  
| S7b contact shadows | 0.20 |  
| S8 glossy resolve | 0.30 |  
| \*\*Total (direct \+ indirect)\*\* | \*\*8.55\*\* |

That's the honest number for a full lighting solution at 60 fps on a 1660, leaving \~7.5 ms for everything else. Tight but shippable at 1080p with a lean forward+ or deferred base pass.

Degradation order for lower-end parts: T1 budget 16 → 8 → 4; PCSS 12-tap → 6-tap → plain 4-tap PCF at fixed radius; cone march 16 → 8 steps; ReSTIR DI down to 4 candidates and temporal-only reuse; drop contact shadows last, since they're cheap and carry a lot of perceived grounding.

\---

\#\# 7\. New failure cases

| Failure | Fallback |  
|---|---|  
| Emissive texture with no fittable shape (dense detail, particle sheets) | Demote to T3; it becomes soft bounce-only glow with no sharp shadow. Usually correct, since such surfaces rarely cast readable shadows |  
| LTC × V decorrelation: shadow too soft right at contact under a large close light | Screen-space contact shadow term; if off-screen, clamp \`w\_pen\` by distance to nearest occupancy voxel |  
| Emitter behind the shading point's horizon but its proxy center is in front (long tube/rect straddling the plane) | LTC's polygon clipping already handles this correctly for radiance; for visibility, split proxies longer than 3 m into segments at import |  
| Highly non-uniform emissive texture (bright filament in a dim frame) causing the mean-radiance proxy to misrepresent it | Fit quality check on radiance variance as well as coverage; high-variance components split further or demote to T3 |  
| T2 cone-march self-shadowing acne on thin geometry | Bias the march start by 1.5× the C0 cell size along the normal, plus normal-offset; accept slight peter-panning at 8 m cascade |  
| More than 16 significant emitters in view (a casino, a server room) | ReSTIR DI degrades gracefully — the excess become T2 with cone-march visibility. Visibly softer shadows from the demoted set, but no popping since promotion hysteresis is 8 frames |  
| Moving emitter with baked shadow tile | Instances flagged \`dynamicEmitter\` force a tile redraw every frame; budget 4 such lights, remainder forced to T2 |

\---

\#\# 8\. What this adds to the prior-work list

Heitz, Dupuy, Hill & Neubelt 2016, \*Real-Time Polygonal-Light Shading with Linearly Transformed Cosines\*, plus the textured-light extension. Fernando 2005, \*Percentage-Closer Soft Shadows\*. Crassin et al. 2011 for the cone-march-as-visibility idea, though here the cone marches an occupancy field for shadows rather than a radiance field for GI. Bitterli et al. 2020 ReSTIR, applied to emitter selection rather than sample generation.

The recombination that matters: \*\*the same emitter proxy set feeds both the per-pixel analytic direct path and the per-surfel cache path, at two different fidelity levels, and the T3 residue is handled by the GI system with no direct-lighting code at all.\*\* One emitter representation, three estimators chosen by importance. That means an emissive material author never has to decide whether their surface is "a light" or "GI" — the tiering does it, and the tier boundary is a smooth quality gradient rather than a switch.
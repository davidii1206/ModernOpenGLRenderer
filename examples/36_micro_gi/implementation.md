# Micro-Rendering for Scalable, Parallel Final Gathering — OpenGL Implementation Plan

Reference: Ritschel, Engelhardt, Grosch, Seidel, Kautz, Dachsbacher.
*Micro-Rendering for Scalable, Parallel Final Gathering.* ACM TOG 28(5), SIGGRAPH Asia 2009.

Original implementation: CUDA. Target here: OpenGL 4.5+ compute shaders, one invocation
(or one small local workgroup) per gather sample, mirroring the paper's "one CUDA thread
per micro-rendering" design. The paper itself notes a pure OpenGL implementation should be
comparably fast since all work fits in a single kernel/shader.

This doc is the map. We implement it stage by stage; each stage should be independently
testable/visualizable before moving to the next.

---

## 0. Data structures (shared across all stages)

### 0.1 Point sample (leaf payload)
Packed to match the paper's 128-bit node record, extended slightly for GL alignment:

```c
struct PointNode {
    vec4 posRadius;     // xyz = position, w = radius
    vec4 normalAlbedo;  // xyz = normal (or packed), w = albedo (packed rgb -> could split)
    vec4 extra;         // x = cone angle (normal cone half-angle), y = reserved,
                         // z = triangle index (for barycentric re-eval), w = radiance (half2 packed or separate buffer)
};
```
Practical note: don't fight GLSL packing rules the way the paper fights CUDA bit-packing.
Use a `std430` SSBO of two `vec4`s per node for leaves+interior (32 bytes/node) and a
*separate* `vec2` (or `vec4` for RGB+pad) buffer for reflected radiance, updated per-frame
per lighting pass. This trades some memory for shader simplicity and avoids premature
bit-packing optimization.

Interior node = same struct: bounding sphere (posRadius) + normal cone (normalAlbedo.xyz
= cone axis, extra.x = cone half-angle). No albedo/radiance meaning for interior nodes.

### 0.2 Complete binary tree layout
Same trick as the paper: store the tree as an implicit **complete binary tree** in a flat
array, leaves in one contiguous range, interior nodes computed bottom-up. Concretely:

- `N` = number of leaf points (pad up to a power of two with degenerate/zero-radius points
  if needed, OR keep paper's approach: build is offline so we can choose N = 2^k exactly by
  how many points we sample).
- Node array size = `2N - 1`. Use the standard implicit heap indexing: root = 0, children
  of node `i` = `2i+1`, `2i+2` — OR follow the paper's array layout (leaves at the end,
  built bottom-up by levels) — either works; I'd use the heap-index scheme since it makes
  the GPU refit trivially parallel over levels without extra offset bookkeeping.
- Leaf `j` (0-indexed sample) lives at array index `(N - 1) + j`.

### 0.3 Buffers (SSBOs)
| Buffer | Contents | Written by |
|---|---|---|
| `TriangleBuf` | mesh triangles (positions, normals, material id) | app upload |
| `LeafSourceBuf` | per leaf: triangle index + barycentric (u,v) — fixed after preprocessing | CPU preprocess |
| `TreeNodeBuf` | `2N-1` nodes: posRadius, normal+coneAngle | GPU refit (leaves updated, interior recomputed) |
| `RadianceBuf` | per-node reflected radiance (updated by lighting pass) | GPU lighting pass |
| `WarpTableBuf` / texture | per-BRDF tabulated inverse-CDF mapping for importance warping | CPU precompute (static BRDFs) or GPU precompute pass (spatially varying) |
| `GatherPointBuf` | per-pixel (or per-sample) world pos, normal, view dir, material | G-buffer pass |
| `OutputImage` | final indirect radiance (low-res or full-res) | micro-rendering pass |

---

## 1. Pipeline stages, in build order

### Stage A — Scaffolding
- GLFW + GLAD + OpenGL 4.5 core context, debug callback, basic camera.
- Simple forward G-buffer pass (position, normal, albedo, depth) — this gives us gather
  points for free and doubles as the "direct illumination" pass target.
- **Deliverable:** textured triangle mesh on screen, G-buffer visualizable.

### Stage B — Offline point sampling + tree topology (CPU, once per mesh/load)
- Best-candidate sampling of points on triangles, density ∝ area (paper's method) — or a
  simpler stratified/area-weighted sampling to start (best-candidate is a quality
  refinement, not required for correctness).
- Store (triangleIndex, barycentric) per point — this is what lets Stage C re-evaluate
  position/normal every frame for deforming geometry.
- Build topology via recursive median split on largest-extent axis (paper's method,
  O(n log n), done once). Emit the **leaf order array** (a permutation) — this fixes which
  original point sample sits at each leaf slot. Topology itself needs no GPU work since
  it's structural only.
- **Deliverable:** a CLI/offline step producing `leaf_order.bin` + point count N (power of two).

### Stage C — GPU per-frame refit (compute shader) — "runtime GPU build"
Two compute passes, run once per frame before lighting/gathering:
1. **Leaf update pass** (`N` threads): re-evaluate world position + normal from
   `LeafSourceBuf` (triangle index, barycentric) against the current (possibly skinned/
   animated) `TriangleBuf`. Write into `TreeNodeBuf` at leaf indices.
2. **Bottom-up refit pass(es)** (`log2(N)` dispatches, one per tree level, going from
   leaves toward root; level `L` has `N / 2^(depth-L)` nodes): each thread reads its two
   children's bounding spheres + normal cones and computes the parent's minimal enclosing
   sphere (Ritter's algorithm is fine — same simplification the paper implicitly allows)
   and merged normal cone. This is the O(n) bottom-up merge from the paper.
- **Deliverable:** dump the tree to a debug SSBO readback and validate bounding spheres
  visually (draw as instanced wireframe spheres) against the mesh.

### Stage D — Direct lighting of the point hierarchy
- Shade every **leaf** point with direct light (shadow-mapped) → write to `RadianceBuf`.
- Propagate leaf radiance up through interior nodes via a **pull/average pass** (again
  `log2(N)` compute dispatches) so interior nodes hold an area-weighted average radiance —
  needed later for coarse-node shading during micro-rendering traversal cutoffs, and for
  the radiosity/photon-map applications in section 4 of the paper.
- **Deliverable:** point cloud renders with correct direct shading (instanced point/disc
  splat renderer for debug visualization).

### Stage E — Micro-rendering core (GPU rasterization point splatting)
> **IMPLEMENTED** (current code). The original design called for a compute-shader BVH
> traversal per gather sample; that has been replaced by **real GPU rasterization** of the
> point hierarchy, per the user's request. This is closer to the hardware-tessellated
> variant the paper discusses, and it gives us true hardware depth sorting for free.

Three passes, run once per frame:

1. **Per-gather-pixel micro camera setup** (`cam_setup_cs`), one thread per low-res gather
   pixel. Builds a per-pixel micro camera from the G-buffer (position, normal, albedo,
   roughness, view direction):
   - Tangent frame `T/B/N` from the geometric normal and the view direction.
   - Warped sampling direction via the GGX BRDF warp: `warp_dir = normalize(mix(N, R, 1 - roughness))`
     where `R` is the specular reflection of the view about `N`.
   - Outputs `6 × vec4` per gather pixel into `cams_buf` (SSBO binding 23):
     `pos+valid, T, B, warp_dir, albedo, normal`.
   - Also regenerates the per-roughness warp table texture when the material roughness
     slider changes.

2. **Instanced point splatting into a micro-atlas** (`splat_vs`/`splat_fs`), the heart of
   the change. One `glDrawArraysInstanced(GL_POINTS)` per atlas row-chunk:
   - Instance = one gather pixel; vertex = one hierarchy node from the selected tree level
     `L = splat_lod` (range `[2^L-1, 2^(L+1)-1)`, interior nodes carry pulled-up radiance;
     `L = LOG2N` means leaf surfels). This is the rasterized equivalent of the traversal
     "cut".
   - The vertex shader applies the **hemispherical disk mapping** Φ(x,y) =
     (x, y, sqrt(1-x²-y²)) in the gather pixel's tangent frame — non-linear, so it is
     implemented manually in the shader rather than as a literal 4×4 matrix.
   - Culls per splat: invalid gather pixel, self-epsilon (`0.5 × leaf radius`), behind the
     receiver's hemisphere, outside the warped mapping, and backfacing nodes (via the node's
     normal-cone axis).
   - Splat radius in micro pixels: `r_px = min(nodeRadius · invDist · microSize/2, microSize/2)`.
   - The fragment shader discards outside the round point footprint (`gl_PointCoord`) and
     writes `gl_FragDepth` from the warped distance — the GPU does the **depth sorting**
     (micro-buffer depth test) natively.
   - Output goes to a **micro-atlas**: one RGBA16F color texture + DEPTH_COMPONENT32F
     renderbuffer, tiled over `atlas_rows` low-res rows per chunk to bound memory
     (32M pixels per chunk budget).
   - The splat level is a runtime slider (`splat_lod`), so cost scales like
     `2^L × gatherPixels`.

3. **Convolution** (`micro_sum_cs`), one thread per gather pixel. Convolves its 8×8 micro
   tile with the GGX warp Jacobian (same math as the compute-only design) and writes the
   indirect result to `indirect_tex`. Optionally dumps the tile to `debug_tex` for the
   `show_micro_debug` overlay.

Bilateral upsampling (Stage H) runs after this to lift the low-res indirect result to full
resolution.

Measured on an RTX 3060 Laptop GPU (scale=4 → 400×225 gather pixels, LOD 10 → 1024 nodes,
≈92M splat instances): `setup ≈ 0.03ms`, `splat ≈ 9–11ms`, `sum ≈ 0.10ms`.

### Stage F — On-demand ray casting for post-traversal list
- After rasterization, for each pixel with an unresolved leaf-list entry, ray-cast (simple
  ray-disc intersection, since leaves are oriented discs) against that pixel's list only.
  Small, bounded cost (paper reports this as 8% of frame time).
- **Deliverable:** eliminates rasterization holes at silhouettes (compare with/without, like
  paper's Fig. 11).

### Stage G — BRDF importance warping
- Given view direction ω_o and a BRDF slice f_r^ωo(ω), tabulate it as a 2D PDF, compute
  inverse marginal/conditional CDFs (M⁻¹, C⁻¹) — do this as a small compute pre-pass per
  *material type* (not per gather sample, for common analytic BRDFs like Phong/GGX) storing
  the warp as a small texture (e.g. 32×32 or 64×64) sampled per gather point. Spatially
  varying BRDF params (e.g. varying roughness) can be handled by tabulating a small family
  of tables and blending, or computing the warp table per-gather-point in a lightweight
  compute pass if budget allows (this is closer to the paper's literal "every micro-render
  computes its own warp").
- Swap Φ(x,y) in Stage E's traversal for the warped mapping; Ω(x,y) becomes the Jacobian of
  the warp (paper's finite-difference/gradient approximation is fine).
- Once warped, convolution becomes a **plain sum**, no explicit BRDF/cosine multiply.
- **Deliverable:** glossy materials look correct with far fewer artifacts than the fixed
  mapping at equal micro-buffer resolution (reproduce paper's Fig. 5 comparison).

### Stage H — Bilateral upsampling
- Run Stage E/F/G at reduced resolution (e.g. 1/4 or 1/16 pixel count) with jittered/rotated
  sample positions, then upsample using depth+normal-aware bilateral weights (paper cites
  Sloan et al. 2007 image-based proxy accumulation) in a full-screen compute or fragment
  pass, combined with full-res direct lighting.
- **Deliverable:** interactive frame rates at preview quality; toggle between full-res and
  upsampled to compare.

### Stage I — Applications (pick based on what you want to demo)
1. **One-bounce indirect** — already have this after Stage E–H.
2. **Multi-bounce via instant radiosity**: generate VPLs from a shadow-map-based RSM pass,
   relight the point hierarchy leaves with VPL contributions (instead of/in addition to
   direct light) before the final micro-rendering gather pass.
3. **Radiosity-style multi-bounce**: Jacobi iteration — repeatedly micro-render *at interior
   node level* to update `RadianceBuf`, then push-pull to propagate through the hierarchy;
   finish with one full micro-render pass at every pixel.
4. **Photon-map final gathering**: offline photon pass (CPU or compute), density estimation
   into leaf radiance, pull up through hierarchy, then reuse Stage E–H unchanged for the
   interactive walkthrough.

---

## 2. Suggested build order (given your answers)

1. Stage A (scaffold + G-buffer)
2. Stage B (offline point sampling + topology — CPU tool)
3. Stage C (GPU refit compute shaders)
4. Stage D (direct lighting of hierarchy + radiance pull-up)
5. Stage E (micro-rendering core, fixed mapping, diffuse)
6. Stage F (ray-cast fallback)
7. Stage G (BRDF importance warping)
8. Stage H (bilateral upsampling)
9. Stage I (pick an application)

## 3. Things we're deliberately simplifying vs. the paper (flag if you want them exact)
- Bit-packing of node data into 128 bits: skipped in favor of plain `vec4`-aligned SSBOs.
  Revisit only if memory bandwidth becomes the bottleneck.
- Best-candidate point sampling: start with area-weighted stratified sampling; upgrade
  later if point distribution artifacts show up.
- Per-gather-sample warp table computation: start with per-material precomputed tables;
  the fully spatially-varying version is a later refinement (Stage G note above).
- Morton-order dispatch reordering for coherence (paper's space-filling-curve scheduling,
  Fig. 13): **moot now** — the micro pass is rasterized point splatting, so the GPU's own
  primitive/VS scheduling handles coherence.

## 4. Repo layout
```
microrender/
  docs/DESIGN.md          <- this file
  src/                    <- C++ app: window, GL setup, buffer management, per-stage drivers
  shaders/                <- .comp/.vert/.frag GLSL files, one subfolder per stage
  third_party/             <- GLFW/GLAD/GLM (fetched via CMake FetchContent)
  CMakeLists.txt
```
// Example 33 — MDC Coverage Atlas → Voxel Grid + DDA Ray Marcher
//
// Builds a binary 3D occupancy grid from the MDC output (the adaptive sparse
// mip-chains + per-texel coverage mask that gfx::CoverageAtlas produces for
// examples 31/32) and renders it with a GPU DDA (Amanatides & Woo) ray marcher.
// The grid is stored bit-packed: N^3 voxels use ceil(N^3/8) bytes, i.e. a 1024^3
// grid is 128 MiB, not 1 GiB.
//
// Voxelisation: every covered atlas texel is resolved to its [front, back]
// depth band by walking the patch's mip quadtree to the leaf covering that
// texel (child = first_child + cy*2 + cx, descending a quadrant per level).
// The texel's world column is a world-axis-aligned box (every patch basis is a
// world axis), and every grid cell it intersects is stamped occupied. This is
// the same sparse tree the shader ray marcher uses, so the grid is a faithful
// byproduct of the MDC pipeline, not a separate scan-conversion.
//
// Depth decode (mirrors the shader): dmin/dmax are 16-bit codes over
// [qmin, qmax] (the global front/back depth range in world units, measured
// from the model AABB minimum), code = clamp(t,0,1)*65534 + 1, so
//   d = qmin + (qmax - qmin) * (code - 1) / 65534
// and a world coordinate along the patch's depth axis is  d + dot(origin, u).
//
// Usage:
//   33_mdc_voxelize [--model PATH.glb] [--grid N] [--scale S] [--out FILE.bin]
//   --model    initial model (one of the three example-32 scenes, or any glb)
//   --grid     initial grid resolution, power of two (default 1024; 64..2048)
//   --scale    initial "resolution scale" multiplier over the 1024 base
//   --pad      half-voxel footprint overlap for neighbouring patches (default 0.5)
//   --mid      fraction of the depth band kept, re-anchored on the band midpoint
//              (default 0.5; 1.0 = full conservative band, the old behaviour)
//   --density  multiplier for the atlas texel density (default 1.0; >1 shrinks
//              the conservative depth bias ~linearly and forces an atlas rebuild)
//   --out      also write the grid to FILE.bin (MDCV v2, see below)
//
// Controls: LMB drag orbits, scroll wheel zooms.
//
// MDCV v2 output format (with --out):
//   "MDCV" u32 version(2) u32 N u64 num_occupied
//   f32[3] aabb_min f32[3] aabb_max f32[3] voxel_size
//   then ceil(N^3/8) bytes, bit-packed 0/1 occupancy,
//   idx = z*N*N + y*N + x, bit (idx & 7) of byte (idx >> 3).

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>
#include <gfx/imgui_overlay.hpp>
#include <gllib/log.hpp>
#include <imgui.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using glm::vec3;

static uint32_t next_pow2_u32(uint32_t v) {
    v--; v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

static int round_pow2(int v) {
    return int(next_pow2_u32(uint32_t(std::max(1, v))));
}

static std::string base_name(const std::string& path) {
    size_t s = path.find_last_of('/');
    size_t d = path.find_last_of('.');
    size_t b = (s == std::string::npos) ? 0 : s + 1;
    if (d == std::string::npos || d < b) d = path.size();
    return path.substr(b, d - b);
}

// --- Same scenes as example 32, same cache dirs -----------------------------

struct ModelEntry {
    const char* name;
    const char* glb;
    const char* cache;
};

static const ModelEntry kKnownModels[] = {
    {"Cornell Box",     "CornellBoxOriginal.glb", "cache/cornell"},
    {"Stanford Bunny",  "Stanford_Bunny.glb",     "cache/bunny"},
    {"Stanford Dragon", "Stanford_Dragon.glb",    "cache/dragon"},
};

// --- Bit-packed 3D occupancy grid -------------------------------------------

struct Grid {
    int N = 0;
    vec3 lo{1e30f}, hi{-1e30f}, cs{1.0f};
    std::vector<uint8_t> bits;  // ceil(N^3/8) bytes

    void alloc(int n, const vec3& minp, const vec3& maxp) {
        N = n;
        lo = minp;
        hi = maxp;
        cs = (hi - lo) / float(N);
        bits.assign((size_t(N) * N * N + 7u) / 8u, 0);
    }

    inline void set(int x, int y, int z) {
        const size_t i = size_t(z) * N * N + size_t(y) * N + size_t(x);
        bits[i >> 3] |= uint8_t(1u << (i & 7u));
    }

    inline bool get(int x, int y, int z) const {
        if ((unsigned)x >= (unsigned)N || (unsigned)y >= (unsigned)N ||
            (unsigned)z >= (unsigned)N) return false;
        const size_t i = size_t(z) * N * N + size_t(y) * N + size_t(x);
        return (bits[i >> 3] >> (i & 7u)) & 1u;
    }

    uint64_t count() const {
        uint64_t c = 0;
        for (uint8_t b : bits) c += std::popcount(b);
        return c;
    }

    double mib() const { return double(bits.size()) / (1024.0 * 1024.0); }
};

struct VoxelStats {
    uint64_t covered = 0, stamped = 0, skip_not_covered = 0, nocc = 0;
    double ms = 0.0;
};

// Voxelise the atlas into a bit grid. Single-threaded (the MDC pipeline itself
// is single-threaded; this mirrors it). The tree descent is identical to the
// ray-marcher shader's, so N only changes the rasterisation step size.
static Grid voxelize(const gfx::CoverageAtlas& atlas, int N, VoxelStats& st, float pad_frac = 0.0f,
                     float mid_frac = 0.5f) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    vec3 lo(1e30f), hi(-1e30f);
    for (const auto& p : atlas.positions()) { lo = glm::min(lo, p); hi = glm::max(hi, p); }
    for (int k = 0; k < 3; ++k) if (hi[k] - lo[k] < 1e-6f) hi[k] = lo[k] + 1.0f;

    Grid g;
    g.alloc(N, lo, hi);

    const auto& patches = atlas.patches();
    const auto& dc      = atlas.depth_chain();
    const auto& cov     = atlas.coverage();
    const int aw = atlas.atlas_width(), ah = atlas.atlas_height();
    const float gdmin = dc.qmin[0], gdmax = dc.qmax[0];

    VoxelStats s;
    for (int pid = 0; pid < (int)patches.size(); ++pid) {
        const auto& p = patches[pid];
        if (p.tex_w <= 0 || p.tex_h <= 0) continue;
        const int tw = p.tex_w, th = p.tex_h;
        const uint32_t Np = next_pow2_u32(uint32_t(std::max(tw, th)));
        const float d0 = glm::dot(lo, p.basis_u);
        const float da = p.proj_size.x / float(tw);
        const float db = p.proj_size.y / float(th);

        // World unit axes of the stored projection components (project_along
        // in coverage_atlas.cpp returns raw world components; basis_v/basis_w
        // can carry a sign flip on negative axes, so use the raw component
        // axes ex/ey here).
        vec3 ex, ey;
        switch (p.axis) {
            case 0: case 1: ex = {0, 1, 0}; ey = {0, 0, 1}; break;  // u = ±X, proj = (Y,Z)
            case 2: case 3: ex = {1, 0, 0}; ey = {0, 0, 1}; break;  // u = ±Y, proj = (X,Z)
            default:        ex = {1, 0, 0}; ey = {0, 1, 0}; break;  // u = ±Z, proj = (X,Y)
        }

        for (int j = 0; j < th; ++j) {
            for (int i = 0; i < tw; ++i) {
                const int gx = p.atlas_x + i, gy = p.atlas_y + j;
                if (gx < 0 || gy < 0 || gx >= aw || gy >= ah) continue;
                if (!cov[size_t(gy) * aw + size_t(gx)]) continue;
                s.covered++;

                // Level-0 chain node index == the patch's position in
                // patches() (pack order), NOT p.id — ids are assigned before
                // pack_atlas() sorts the vector, so the two diverge.
                int L = 0; uint32_t idx = uint32_t(pid);
                int x0 = 0, y0 = 0, stp = int(Np);
                for (;;) {
                    if (L >= int(dc.levels.size())) break;
                    const auto& lev = dc.levels[L];
                    if (idx >= lev.meta.size()) break;
                    const uint32_t m = lev.meta[idx];
                    if (!(m & 0x80000000u) || !(m & 0x40000000u)) break;
                    const int hs = stp >> 1;
                    const int cx = (i >= x0 + hs) ? 1 : 0;
                    const int cy = (j >= y0 + hs) ? 1 : 0;
                    idx = (m & 0x3FFFFFFFu) + uint32_t(cy * 2 + cx);
                    x0 += cx * hs; y0 += cy * hs; stp = hs; ++L;
                }
                if (L >= int(dc.levels.size())) continue;
                const auto& lev = dc.levels[L];
                if (idx >= lev.meta.size()) continue;
                const uint32_t m = lev.meta[idx];
                if (!(m & 0x80000000u)) { s.skip_not_covered++; continue; }
                if (lev.data.size() < size_t(idx) * 4 + 4) continue;

                const uint8_t* dv = lev.data.data() + size_t(idx) * 4;
                const uint32_t q0 = dv[0] | (uint32_t(dv[1]) << 8);
                const uint32_t q1 = dv[2] | (uint32_t(dv[3]) << 8);
                if (q0 == 0 || q1 == 0) continue;
                const float dmin = gdmin + (gdmax - gdmin) * float(q0 - 1) / 65534.0f;
                const float dmax = gdmin + (gdmax - gdmin) * float(q1 - 1) / 65534.0f;

                const float a0 = p.proj_min.x + float(i) * da, a1 = a0 + da;
                const float b0 = p.proj_min.y + float(j) * db, b1 = b0 + db;
                const float cu0 = dmin + d0, cu1 = dmax + d0;

                // The conservative per-texel band [dmin, dmax] is the surface's
                // depth extent across the whole texel square. For a tilted texel
                // the near edge sits a full thickness in front of the surface at
                // the texel centre, and at patch seams (where the surface is most
                // oblique to the depth axis) that bias piles up into a coherent
                // ridge. Re-anchor the stamped column on the band midpoint and
                // keep only a mid_frac fraction of the thickness, so the reported
                // surface tracks the true surface instead of its near edge. The
                // depth extent is additionally capped at one voxel so a steep
                // texel can never push its column more than half a voxel in front
                // of the texel-centre surface; without the cap the column front
                // would stick out a full thickness, and neighbouring patches'
                // differing thicknesses would leave a visible step at the seam.
                float bm_lo = cu0, bm_hi = cu1;
                if (mid_frac < 1.0f) {
                    const float bm_mid = 0.5f * (cu0 + cu1);
                    const float u_cell = std::abs(p.basis_u.x) * g.cs.x +
                                         std::abs(p.basis_u.y) * g.cs.y +
                                         std::abs(p.basis_u.z) * g.cs.z;
                    const float bm_hw  = std::min(0.5f * (cu1 - cu0) * mid_frac,
                                                  0.5f * u_cell);
                    bm_lo = bm_mid - bm_hw;
                    bm_hi = bm_mid + bm_hw;
                }

                // Half-voxel overlap on the projection footprint so that
                // neighbouring patches' boxes meet instead of leaving a
                // sub-voxel step at their shared face. The depth band is
                // deliberately not extended (that would just thicken slabs).
                const float pad = pad_frac > 0.0f
                    ? pad_frac * (std::abs(ex.x) * g.cs.x + std::abs(ex.y) * g.cs.y +
                                  std::abs(ex.z) * g.cs.z)
                    : 0.0f;
                const float a0p = a0 - pad, a1p = a1 + pad;
                const float b0p = b0 - pad, b1p = b1 + pad;

                // World AABB of the texel column box in the (u, ex, ey) frame.
                vec3 wlo(0.0f), whi(0.0f);
                auto add_range = [&](const vec3& ax, float cmin, float cmax) {
                    for (int k = 0; k < 3; ++k) {
                        if (std::abs(ax[k]) < 0.5f) continue;
                        if (ax[k] > 0.0f) { wlo[k] = cmin; whi[k] = cmax; }
                        else              { wlo[k] = -cmax; whi[k] = -cmin; }
                    }
                };
                add_range(p.basis_u, bm_lo, bm_hi);
                add_range(ex, a0p, a1p);
                add_range(ey, b0p, b1p);

                const int gx0 = std::clamp(int(std::floor((wlo.x - lo.x) / g.cs.x)), 0, N - 1);
                const int gx1 = std::clamp(int(std::floor((whi.x - lo.x) / g.cs.x)), 0, N - 1);
                const int gy0 = std::clamp(int(std::floor((wlo.y - lo.y) / g.cs.y)), 0, N - 1);
                const int gy1 = std::clamp(int(std::floor((whi.y - lo.y) / g.cs.y)), 0, N - 1);
                const int gz0 = std::clamp(int(std::floor((wlo.z - lo.z) / g.cs.z)), 0, N - 1);
                const int gz1 = std::clamp(int(std::floor((whi.z - lo.z) / g.cs.z)), 0, N - 1);
                for (int z = gz0; z <= gz1; ++z)
                    for (int y = gy0; y <= gy1; ++y)
                        for (int x = gx0; x <= gx1; ++x)
                            g.set(x, y, z);
                s.stamped++;
            }
        }
    }

    s.nocc = g.count();
    s.ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    st = s;
    return g;
}

// --- DDA ray marcher --------------------------------------------------------

static const char* kDdaShaderSrc = R"GLSL(
#version 450 core

layout(local_size_x = 8, local_size_y = 8) in;

layout(std430, binding = 0) readonly buffer Voxels {
    uint data[];
};
layout(rgba8, binding = 1) uniform writeonly image2D u_out;

uniform uint u_N;
uniform vec3 u_lo;     // grid AABB min (world)
uniform vec3 u_cell;   // voxel size (world)
uniform vec3 u_cam;    // eye (world)
uniform mat4 u_inv_vp; // inverse view-projection
uniform int  u_color_mode;  // 0 = normal shading, 1 = world-position colour

bool is_set(ivec3 c) {
    if (c.x < 0 || c.y < 0 || c.z < 0 ||
        c.x >= int(u_N) || c.y >= int(u_N) || c.z >= int(u_N)) return false;
    // Linear index i = z*N^2 + y*N + x can exceed 2^32 at N = 2048
    // (2047*2048^2 > 2^32), which would silently wrap in 32-bit arithmetic.
    // N is a power of two >= 64, so N^2 is a multiple of 32 and i = z*N^2 + r
    // (r = y*N + x) splits without a carry: word = z*(N^2>>5) + (r>>5),
    // bit = r&31. z*(N^2>>5) stays < 2^28 for N <= 2048.
    uint r = uint(c.y) * u_N + uint(c.x);
    uint w = uint(c.z) * (u_N * u_N >> 5u) + (r >> 5u);
    return ((data[w] >> (r & 31u)) & 1u) != 0u;
}

// Surface normal from the occupancy field: central differences over the six
// neighbours, negated so the result points from the occupied side (interior)
// towards the empty side (the surface). Out-of-range neighbours read as empty,
// so voxels on the grid boundary get their true outward normal instead of a
// ray-direction-dependent one. Falls back to 0 for fully-surrounded voxels
// (there the caller keeps the DDA crossing normal).
vec3 occupancy_normal(ivec3 c) {
    ivec3 g = ivec3(0);
    g.x = (is_set(c + ivec3(1, 0, 0)) ? 1 : 0) - (is_set(c - ivec3(1, 0, 0)) ? 1 : 0);
    g.y = (is_set(c + ivec3(0, 1, 0)) ? 1 : 0) - (is_set(c - ivec3(0, 1, 0)) ? 1 : 0);
    g.z = (is_set(c + ivec3(0, 0, 1)) ? 1 : 0) - (is_set(c - ivec3(0, 0, 1)) ? 1 : 0);
    vec3 n = -vec3(g);
    return dot(n, n) > 1e-6 ? normalize(n) : vec3(0.0);
}

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    ivec2 sz = imageSize(u_out);
    if (p.x >= sz.x || p.y >= sz.y) return;

    vec2 ndc = (vec2(p) + 0.5) / vec2(sz) * 2.0 - 1.0;
    vec4 wn = u_inv_vp * vec4(ndc, -1.0, 1.0);
    vec4 wf = u_inv_vp * vec4(ndc,  1.0, 1.0);
    vec3 ro = u_cam;
    vec3 rd = normalize(wf.xyz / wf.w - wn.xyz / wn.w);

    vec3 col = mix(vec3(0.015, 0.020, 0.035), vec3(0.09, 0.11, 0.16), ndc.y * 0.5 + 0.5);

    // Clip the ray to the grid slab; if it misses, keep the background.
    vec3 hi = u_lo + u_cell * float(u_N);
    vec3 invR = 1.0 / max(abs(rd), vec3(1e-30));
    // Multiply by sign(rd) so the t's are the true entry/exit times even when
    // a ray component is negative (plain (b - ro) * invR is wrong then).
    vec3 t0 = (u_lo - ro) * invR * sign(rd);
    vec3 t1 = (hi  - ro) * invR * sign(rd);
    vec3 tmn = min(t0, t1);
    vec3 tmx = max(t0, t1);
    float tN = max(tmn.x, max(tmn.y, tmn.z));
    float tF = min(tmx.x, min(tmx.y, tmx.z));
    if (tN > tF || tF < 0.0) {
        imageStore(u_out, p, vec4(col, 1.0));
        return;
    }
    // Start at the entry point (cell space).
    vec3 pos = (ro + rd * max(tN, 0.0) - u_lo) / u_cell;
    ivec3 cell = clamp(ivec3(floor(pos)), ivec3(0), ivec3(u_N) - 1);

    // DDA: d is in cells per world unit, so t stays in world units.
    vec3 d = rd / u_cell;
    ivec3 step = ivec3(sign(d));
    vec3 tMax, tDelta;
    for (int k = 0; k < 3; ++k) {
        float a = abs(d[k]);
        if (a < 1e-20) { tMax[k] = 1e30; tDelta[k] = 1e30; continue; }
        tDelta[k] = 1.0 / a;
        // Distance to the next grid line in the direction of travel: the
        // upper boundary (cell+1) when stepping +, the lower (cell) when -.
        tMax[k] = (step[k] > 0 ? (float(cell[k]) + 1.0 - pos[k])
                               : (float(cell[k]) - pos[k])) / d[k];
    }

    // Entry face normal: the ray enters the grid through the plane of the
    // axis that attains the max entry time tN. A hit in the very first cell
    // (an entry-cell hit) lies on the model's outer shell against that grid
    // face, so its outward normal is axis-aligned (-step along that axis)
    // rather than the diagonal -step used before.
    int entryAxis = 0;
    float bestT = tmn.x;
    if (tmn.y > bestT) { bestT = tmn.y; entryAxis = 1; }
    if (tmn.z > bestT) { entryAxis = 2; }
    ivec3 nrm = ivec3(0);
    nrm[entryAxis] = -step[entryAxis];
    bool hit = false;
    int it = 0, maxSteps = 3 * int(u_N) + 16;
    while (it++ < maxSteps) {
        if (is_set(cell)) { hit = true; break; }
        if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
            pos.x += float(step.x); cell.x += step.x; tMax.x += tDelta.x;
            nrm = ivec3(-step.x, 0, 0);
        } else if (tMax.y <= tMax.z) {
            pos.y += float(step.y); cell.y += step.y; tMax.y += tDelta.y;
            nrm = ivec3(0, -step.y, 0);
        } else {
            pos.z += float(step.z); cell.z += step.z; tMax.z += tDelta.z;
            nrm = ivec3(0, 0, -step.z);
        }
        if (cell.x < 0 || cell.y < 0 || cell.z < 0 ||
            cell.x >= int(u_N) || cell.y >= int(u_N) || cell.z >= int(u_N)) break;
    }

    if (hit) {
        if (u_color_mode == 1) {
            // World-position colour: maps the hit voxel's world position onto
            // the grid AABB to a continuous RGB ramp. Independent of the
            // occupancy normals, so any leftover banding must be geometry.
            vec3 hi = u_lo + u_cell * float(u_N);
            vec3 cpos = (u_lo + (vec3(cell) + 0.5) * u_cell - u_lo) / max(hi - u_lo, vec3(1e-6));
            col = clamp(cpos, vec3(0.0), vec3(1.0));
        } else {
            vec3 n = occupancy_normal(cell);
            if (dot(n, n) < 1e-6) n = normalize(vec3(nrm));   // interior voxel: use the crossed face
            vec3 base = n * 0.5 + 0.5;                         // normal -> RGB
            float diff = max(dot(n, normalize(vec3(0.5, 0.75, 0.35))), 0.0);
            float t = min(tMax.x, min(tMax.y, tMax.z));        // entry t of hit cell
            float fog = exp(-0.08 * max(t, 0.0));
            col = base * (0.75 + 0.35 * diff) * (0.85 + 0.15 * fog);
        }
    }
    imageStore(u_out, p, vec4(col, 1.0));
}
)GLSL";

// --- Scene loading (model + cached MDC atlas) -------------------------------

static bool load_scene(const ModelEntry& entry, gfx::Model& model,
                       gfx::CoverageAtlas& atlas, float tol_frac, float density_mul = 1.0f) {
    std::printf("Loading %s...\n", entry.glb);
    if (!model.load(entry.glb)) {
        std::fprintf(stderr, "Failed to load %s\n", entry.glb);
        return false;
    }
    // The grid resolves the model AABB to `grid`^3 cells; for the stamped
    // voxels to actually be cell-sized the atlas texels must be as dense as
    // the grid. Use the auto density (config.auto_target texels across the
    // model's longest axis) instead of the fixed default, mirroring the
    // example-32 loader.
    vec3 lo(1e30f), hi(-1e30f);
    for (size_t mi = 0; mi < model.mesh_count(); ++mi) {
        const auto& mesh = model.mesh(mi);
        for (size_t v = 0; v < mesh.vertex_count(); ++v) {
            const float* p = mesh.vertices()[v].position;
            lo = glm::min(lo, vec3(p[0], p[1], p[2]));
            hi = glm::max(hi, vec3(p[0], p[1], p[2]));
        }
    }
    float span = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
    const float want_density = span > 1e-6f
        ? float(atlas.config().auto_target) / span * density_mul : 0.0f;

    const bool loaded = atlas.load_files(entry.cache);
    const bool leaf_ok = atlas.config().mip_leaf_tile == 1;
    const bool tol_ok = std::fabs(atlas.config().mip_tol_frac - tol_frac) < 1e-5f;
    const bool density_ok = want_density <= 0.0f ||
        std::fabs(atlas.config().texel_density - want_density) < 1e-3f;
    if (!loaded || !leaf_ok || !tol_ok || !density_ok) {
        std::printf("%s cached MDC snapshot — building atlas...\n",
                    loaded ? (leaf_ok ? (tol_ok ? "Different texel density," : "Different mip tolerance,")
                                      : "Uses mip_leaf_tile != 1,")
                           : "No");
        gfx::CoverageAtlasConfig cfg = atlas.config();
        cfg.mip_leaf_tile = 1;
        cfg.mip_tol_frac = tol_frac;
        if (want_density > 0.0f) cfg.texel_density = want_density;
        atlas.set_config(cfg);
        if (!atlas.build(model)) {
            std::fprintf(stderr, "CoverageAtlas build failed\n");
            return false;
        }
        atlas.write_files(entry.cache);
    }
    std::printf("Atlas: %dx%d @ %.0f texels/unit, %zu patches, %zu triangles\n",
                atlas.atlas_width(), atlas.atlas_height(), atlas.final_density(),
                atlas.patches().size(), atlas.triangles().size());
    return true;
}

static void write_grid(const std::string& path, const Grid& g) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "Cannot open %s for writing\n", path.c_str()); return; }
    const char magic[4] = {'M', 'D', 'C', 'V'};
    uint32_t version = 2, N = uint32_t(g.N);
    uint64_t nocc = g.count();
    std::fwrite(magic, 1, 4, f);
    std::fwrite(&version, 4, 1, f);
    std::fwrite(&N, 4, 1, f);
    std::fwrite(&nocc, 8, 1, f);
    std::fwrite(&g.lo, 12, 1, f);
    std::fwrite(&g.hi, 12, 1, f);
    std::fwrite(&g.cs, 12, 1, f);
    std::fwrite(g.bits.data(), 1, g.bits.size(), f);
    std::fclose(f);
    std::printf("Wrote %s (MDCV v2: header 56 B + %zu bit-packed bytes)\n",
                path.c_str(), g.bits.size());
}

// --- main -------------------------------------------------------------------

int main(int argc, char** argv) {
    gllib::log_to_stderr(gllib::LogLevel::info);

    std::string model_path;
    std::string out_path;
    std::string dump_frame;
    int grid = 1024;
    float scale_arg = 1.0f;
    float zoom_arg = 1.0f;
    float tol_arg = 0.0f;
    float pad_arg = 0.5f;
    float mid_arg = 0.5f;
    float density_arg = 1.0f;
    bool color_mode_pos = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--model") { if (i + 1 < argc) model_path = argv[++i]; }
        else if (a == "--grid" ) { if (i + 1 < argc) grid = std::atoi(argv[++i]); }
        else if (a == "--scale") { if (i + 1 < argc) scale_arg = float(std::atof(argv[++i])); }
        else if (a == "--zoom" ) { if (i + 1 < argc) zoom_arg = float(std::atof(argv[++i])); }
        else if (a == "--tol"  ) { if (i + 1 < argc) tol_arg = float(std::atof(argv[++i])); }
        else if (a == "--pad"  ) { if (i + 1 < argc) pad_arg = float(std::atof(argv[++i])); }
        else if (a == "--mid"  ) { if (i + 1 < argc) mid_arg = float(std::atof(argv[++i])); }
        else if (a == "--density") { if (i + 1 < argc) density_arg = float(std::atof(argv[++i])); }
        else if (a == "--color-mode") { if (i + 1 < argc) color_mode_pos = std::atoi(argv[++i]) != 0; }
        else if (a == "--out"  ) { if (i + 1 < argc) out_path = argv[++i]; }
        else if (a == "--dump-frame") { if (i + 1 < argc) dump_frame = argv[++i]; }
        else { std::fprintf(stderr, "Unknown option: %s\n", a.c_str()); return 1; }
    }

    // Model list: the three example-32 scenes, optionally a custom glb.
    std::vector<ModelEntry> models;
    for (const auto& m : kKnownModels) models.push_back(m);
    int cur_model = 0;
    if (!model_path.empty()) {
        const std::string bn = base_name(model_path);
        bool known = false;
        for (int i = 0; i < (int)models.size(); ++i) {
            if (bn == base_name(models[i].glb)) { cur_model = i; known = true; break; }
        }
        if (!known) {
            std::string cache = "cache/" + bn;
            std::string glb   = model_path;
            std::string name  = "Custom: " + bn;
            // ModelEntry holds const char*; keep the strings alive.
            static std::vector<std::string> name_store, glb_store, cache_store;
            name_store.push_back(name);
            glb_store.push_back(glb);
            cache_store.push_back(cache);
            models.push_back({name_store.back().c_str(), glb_store.back().c_str(),
                              cache_store.back().c_str()});
            cur_model = int(models.size()) - 1;
        }
    }

    grid = std::clamp(round_pow2(grid), 64, 2048);
    if (scale_arg != 1.0f)
        grid = std::clamp(round_pow2(int(std::lround(float(grid) * scale_arg))), 64, 2048);

    // --- Window + scene ---
    gfx::Window window({"33 MDC Voxelizer", 1280, 720});
    window.vsync(false);

    gfx::ImGuiOverlay gui;
    if (!gui.init(window)) {
        std::fprintf(stderr, "ImGui overlay init failed\n");
        return EXIT_FAILURE;
    }

    gfx::Model model;
    gfx::CoverageAtlas atlas;
    gfx::CoverageAtlasConfig atlas_default;
    const float tol_frac = tol_arg > 0.0f ? tol_arg : atlas_default.mip_tol_frac;
    if (!load_scene(models[cur_model], model, atlas, tol_frac, density_arg)) return EXIT_FAILURE;
    std::printf("Mip tolerance: %.4g\n", tol_frac);

    // --- Camera (model-relative) ---
    gfx::Camera cam;
    vec3 cam_target(0.0f), cam_lo(0.0f), cam_hi(0.0f);
    float cam_dist = 4.0f;
    float base_grid = 1024.0f, res_scale = 1.0f;

    auto refresh_camera = [&] {
        const auto& pos = atlas.positions();
        cam_lo = vec3(1e30f); cam_hi = vec3(-1e30f);
        for (const auto& p : pos) { cam_lo = glm::min(cam_lo, p); cam_hi = glm::max(cam_hi, p); }
        cam_target = (cam_lo + cam_hi) * 0.5f;
        float span = glm::length(cam_hi - cam_lo);
        cam_dist = (2.4f * span + 1.0f) / zoom_arg;
        cam.look_at(cam_target + vec3(0.35f * cam_dist, 0.45f * cam_dist, cam_dist),
                    cam_target);
    };
    refresh_camera();

    // --- Initial voxelisation ---
    Grid grid_;
    VoxelStats vstats;
    double last_vox_ms = 0.0, avg_vox_ms = 0.0;
    uint64_t vox_runs = 0;
    int vox_grid_span = 0;

    auto do_voxelize = [&] {
        VoxelStats s;
        Grid g = voxelize(atlas, grid, s, pad_arg, mid_arg);
        grid_ = std::move(g);
        vstats = s;
        last_vox_ms = s.ms;
        avg_vox_ms = vox_runs ? (avg_vox_ms * double(vox_runs) + s.ms) / double(vox_runs + 1) : s.ms;
        vox_runs++;
        vox_grid_span = grid;
        std::printf("Voxelised %d^3 in %.1f ms: %llu texels, %llu stamped, %llu occupied (%.2f%%), grid %.1f MiB\n",
                    grid, s.ms, (unsigned long long)s.covered,
                    (unsigned long long)s.stamped, (unsigned long long)s.nocc,
                    100.0 * double(s.nocc) / double(size_t(grid) * grid * grid),
                    grid_.mib());
        if (!out_path.empty()) write_grid(out_path, grid_);
    };
    do_voxelize();

    // --- GPU resources ---
    gl::Buffer vox_ssbo(gl::BufferType::shader);
    gl::Program  dda_prog;
    {
        gl::Shader cs(gl::ShaderType::compute, kDdaShaderSrc);
        if (!cs.compiled()) {
            std::fprintf(stderr, "DDA shader failed:\n%s\n", cs.info_log().c_str());
            return EXIT_FAILURE;
        }
        dda_prog.attach(cs);
        if (!dda_prog.link()) {
            std::fprintf(stderr, "DDA program link failed:\n%s\n", dda_prog.info_log().c_str());
            return EXIT_FAILURE;
        }
    }
    const GLint uN      = dda_prog.uniform_location("u_N");
    const GLint uLo     = dda_prog.uniform_location("u_lo");
    const GLint uCell   = dda_prog.uniform_location("u_cell");
    const GLint uCam    = dda_prog.uniform_location("u_cam");
    const GLint uInvVp  = dda_prog.uniform_location("u_inv_vp");
    const GLint uColorMode = dda_prog.uniform_location("u_color_mode");

    gl::Texture out_tex(gl::TextureType::tex_2d);
    gl::Framebuffer out_fbo(gl::FramebufferType::read);
    int fb_w = 0, fb_h = 0;

    auto resize_render = [&] {
        if (fb_w == window.framebuffer_width() && fb_h == window.framebuffer_height()) return;
        fb_w = window.framebuffer_width();
        fb_h = window.framebuffer_height();
        out_tex = gl::Texture(gl::TextureType::tex_2d);
        out_tex.image_2d(0, GL_RGBA8, fb_w, fb_h, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        out_tex.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        out_tex.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        out_fbo = gl::Framebuffer(gl::FramebufferType::read);
        out_fbo.attach_texture(GL_COLOR_ATTACHMENT0, out_tex);
    };

    auto upload_grid = [&] {
        vox_ssbo = gl::Buffer(gl::BufferType::shader);
        vox_ssbo.data(grid_.bits.data(), grid_.bits.size());
        vox_ssbo.bind_base(0);
    };
    upload_grid();

    // --- Stats ---
    std::vector<float> frame_times;
    double disp_frame_period = 0.0, period_acc = 0.0;
    int period_n = 0;
    double last = window.time(), win_start = window.time();
    double prev_x = 0, prev_y = 0;
    bool prev_lmb = false;
    bool frame_dumped = false;

    // --- Main loop ---
    while (!window.should_close()) {
        double now = window.time();
        float dt = float(now - last);
        last = now;
        if (frame_times.size() >= 120) frame_times.erase(frame_times.begin());
        frame_times.push_back(dt * 1000.0f);
        period_acc += dt * 1000.0f;
        period_n++;

        window.poll_events();
        gui.begin_frame();

        // Camera input (skip when ImGui owns the mouse).
        if (!gui.wants_mouse()) {
            double cx, cy;
            window.cursor_position(cx, cy);
            if (window.mouse_down(gfx::MouseButton::left)) {
                cam.orbit(float(cx - prev_x) * 0.005f, float(prev_y - cy) * 0.005f);
            }
            prev_x = cx;
            prev_y = cy;
            double scroll = window.scroll_delta();
            if (scroll != 0.0) cam.zoom(float(scroll) * 0.1f);
        } else {
            window.scroll_delta();
        }
        prev_lmb = window.mouse_down(gfx::MouseButton::left);

        resize_render();
        int w = window.framebuffer_width(), h = window.framebuffer_height();
        if (w > 0 && h > 0) cam.set_aspect(float(w) / float(h));

        // --- DDA ray march ---
        {
            glm::mat4 inv_vp = glm::inverse(cam.view_projection());
            dda_prog.use();
            dda_prog.uniform1ui(uN, uint32_t(grid_.N));
            dda_prog.uniform3fv(uLo, glm::value_ptr(grid_.lo));
            dda_prog.uniform3fv(uCell, glm::value_ptr(grid_.cs));
            dda_prog.uniform3fv(uCam, glm::value_ptr(cam.position()));
            dda_prog.uniform_matrix4fv(uInvVp, glm::value_ptr(inv_vp));
            dda_prog.uniform1i(uColorMode, color_mode_pos ? 1 : 0);
            out_tex.bind_image(1, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
            vox_ssbo.bind_base(0);
            glDispatchCompute((uint32_t((w + 7) / 8)), (uint32_t((h + 7) / 8)), 1);
            glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
        }

        // --- Blit to screen ---
        out_fbo.blit_to(0, 0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_LINEAR);

        // --- Debug: dump one frame of the DDA output as PPM ---
        if (!dump_frame.empty() && !frame_dumped) {
            std::vector<uint8_t> px(size_t(w) * h * 4);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, out_fbo.handle());
            glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            FILE* f = std::fopen(dump_frame.c_str(), "wb");
            if (f) {
                std::fprintf(f, "P6\n%d %d\n255\n", w, h);
                for (int r = 0; r < h; ++r)
                    for (int x = 0; x < w; ++x)
                        std::fwrite(&px[size_t(r) * w * 4 + size_t(x) * 4], 3, 1, f);
                std::fclose(f);
                std::printf("Dumped DDA frame to %s (%dx%d)\n", dump_frame.c_str(), w, h);
            }
            frame_dumped = true;
        }

        // --- ImGui debug UI ---
        {
            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(330, 0), ImGuiCond_FirstUseEver);
            ImGui::Begin("33 MDC Voxelizer - Debug", nullptr, ImGuiWindowFlags_NoSavedSettings);

            ImGui::SeparatorText("Performance");
            ImGui::Text("Frametime: %.3f ms   (avg %.3f ms)", dt * 1000.0f, disp_frame_period);
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            if (frame_times.size() > 1)
                ImGui::PlotLines("frame period ms (raw)", frame_times.data(),
                                 int(frame_times.size()), 0, nullptr,
                                 0.0f, 0.0f, ImVec2(0, 48));

            ImGui::SeparatorText("Voxelization");
            ImGui::Text("Grid: %d^3  (%.2f MiB, voxel %.4f u)", grid_.N, grid_.mib(),
                        (grid_.hi.x - grid_.lo.x) / float(grid_.N));
            ImGui::Text("Occupied: %llu voxels (%.2f%%)",
                        (unsigned long long)vstats.nocc,
                        100.0 * double(vstats.nocc) /
                            double(size_t(grid_.N) * grid_.N * grid_.N));
            ImGui::Text("Voxelization: %.1f ms  (avg %.1f ms, %llu runs)",
                        last_vox_ms, avg_vox_ms, (unsigned long long)vox_runs);

            ImGui::SeparatorText("Scene");
            ImGui::Checkbox("World-position colour", &color_mode_pos);
            if (ImGui::Combo("Model", &cur_model, "Cornell Box\0Stanford Bunny\0Stanford Dragon\0")) {
                if (load_scene(models[cur_model], model, atlas, tol_frac, density_arg)) {
                    refresh_camera();
                    do_voxelize();
                    upload_grid();
                }
            }
            ImGui::Text("Atlas: %dx%d @ %.0f texels/unit",
                        atlas.atlas_width(), atlas.atlas_height(), atlas.final_density());
            int next_n = std::clamp(round_pow2(int(std::lround(base_grid * res_scale))), 64, 2048);
            ImGui::SliderFloat("Resolution scale", &res_scale, 0.25f, 2.0f, "%.2fx");
            ImGui::Text("  -> grid %d^3 (%.1f MiB)", next_n,
                        double(size_t(next_n) * next_n * next_n) / 8.0 / (1024.0 * 1024.0));
            if (ImGui::IsItemDeactivatedAfterEdit() && next_n != grid_.N) {
                grid = next_n;
                do_voxelize();
                upload_grid();
            }
            if (ImGui::Button("Revoxelize")) {
                grid = next_n;
                do_voxelize();
                upload_grid();
            }

            ImGui::SeparatorText("Controls");
            ImGui::Text("LMB drag: orbit   scroll: zoom");

            ImGui::End();
        }

        gui.render();
        window.swap_buffers();

        // Every ~0.5 s average the accumulated frame periods.
        if (window.time() - win_start >= 0.5) {
            win_start = window.time();
            disp_frame_period = period_n ? period_acc / double(period_n) : 0.0;
            period_acc = 0.0;
            period_n = 0;
        }
    }

    return EXIT_SUCCESS;
}

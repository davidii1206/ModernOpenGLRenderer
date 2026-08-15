// Example 33 — MDC Coverage Atlas → Per-Patch Voxel Grids + DDA Ray Marcher
//
// Builds one small binary occupancy grid PER ATLAS PATCH (instead of one big
// world-aligned grid for the whole model) from the MDC output — the adaptive
// sparse mip-chains + per-texel coverage mask that gfx::CoverageAtlas produces
// for examples 31/32 — and renders them with a GPU DDA (Amanatides & Woo) ray
// marcher that traces the per-patch grids through the patch BVH.
//
// Why per-patch grids: a single world-aligned grid forces every patch's depth
// band to land on the same global voxel lattice, and at a shared border two
// neighbouring patches' bands fight that rounding — the box edges floor to
// different voxels on each side and a sub-voxel step shows up as a coherent
// seam. Each patch grid lives in the patch's own frame (ex/ey = projection
// axes, u = depth axis) with its cell grid aligned to the patch's atlas texels,
// so a texel's depth band lands on exact cell boundaries and the patch is
// self-consistent. The marcher takes the nearest voxel across the per-patch
// grids, and the patch-id colour mode makes the border behaviour visible.
//
// Voxelisation per patch: every covered atlas texel is resolved to its
// [front, back] depth band by walking the patch's mip quadtree to the leaf
// covering that texel (child = first_child + cy*2 + cx, descending a quadrant
// per level), then stamped into the patch's local grid. This is the same sparse
// tree the MDC pipeline itself uses, so the grids are a faithful byproduct of
// the pipeline, not a separate scan-conversion. Grids are bit-packed and all
// concatenated into one SSBO (word-aligned back to back).
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
//   --grid     resolution of the merged --out dump grid (default 1024; 64..2048)
//   --scale    initial "resolution scale" multiplier over the 1024 base
//   --pad      half-cell footprint overlap for neighbouring patches (default 0.5)
//   --mid      fraction of the depth band kept, re-anchored on the band midpoint
//              (default 0.5; 1.0 = full conservative band, the old behaviour)
//   --density  multiplier for the atlas texel density (default 1.0; >1 shrinks
//              the conservative depth bias ~linearly and forces an atlas rebuild)
//   --color-mode 0 = normal shading, 1 = world position, 2 = patch id
//   --out      also merge the per-patch grids and write them to FILE.bin
//              (MDCV v2, see below)
//
// Controls: LMB drag orbits, scroll wheel zooms. The debug window has live
// --mid/--pad sliders (the two seam controls) and a patch inspector.
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

// --- Bit-packed 3D occupancy grid (only used for the --out merged dump) -----

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

// --- Per-patch voxelisation -------------------------------------------------
//
// Every atlas patch becomes its own local occupancy grid, aligned to the
// patch's frame (ex/ey = the two raw projection axes, u = the depth axis) with
// cell = the patch's atlas texel, so a texel's depth-band column lands on exact
// cell boundaries. All grids are bit-packed and concatenated word-aligned into
// one buffer; the layout below mirrors the GLSL `PatchGrid` struct in
// kDdaShaderSrc exactly (all vec4 fields keep std430 offsets trivial).

struct PatchVox {
    glm::vec4 ex, ey, u;             // world axes of the local frame (u = depth)
    glm::vec4 local_min;             // local grid origin (a, b, c)
    glm::vec4 cell;                  // local voxel sizes (da, db, dd)
    glm::uvec4 dims;                 // (nx, ny, nz)
    glm::uvec4 word_off_pad;         // (word offset into the packed bits buffer, 0, 0, 0)
    glm::vec4 world_min, world_max;  // world AABB of the grid box
};

struct PerPatch {
    std::vector<PatchVox> info;
    std::vector<uint32_t> words;       // all grids bit-packed, word-aligned back to back
    std::vector<uint64_t> occupied;    // per-patch occupied voxel count
    vec3 lo{1e30f}, hi{-1e30f};        // model AABB

    double mib() const { return double(words.size()) * 4.0 / (1024.0 * 1024.0); }
};

static PerPatch voxelize_patches(const gfx::CoverageAtlas& atlas, VoxelStats& st,
                                 float pad_frac = 0.0f, float mid_frac = 0.5f) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    PerPatch out;
    for (const auto& p : atlas.positions()) { out.lo = glm::min(out.lo, p); out.hi = glm::max(out.hi, p); }
    for (int k = 0; k < 3; ++k) if (out.hi[k] - out.lo[k] < 1e-6f) out.hi[k] = out.lo[k] + 1.0f;
    const vec3& lo = out.lo;

    const auto& patches = atlas.patches();
    const auto& dc      = atlas.depth_chain();
    const auto& cov     = atlas.coverage();
    const int aw = atlas.atlas_width(), ah = atlas.atlas_height();
    const float gdmin = dc.qmin[0], gdmax = dc.qmax[0];

    // Decode one texel to its (mid-anchored) world depth band along the patch's
    // depth axis. Walks the patch's mip quadtree to the leaf covering the texel
    // (child = first_child + cy*2 + cx, descending a quadrant per level) — the
    // same sparse tree the MDC pipeline itself uses.
    auto decode = [&](int pid, int i, int j, float cap, float mid_frac,
                      float& bm_lo, float& bm_hi) -> bool {
        const auto& p = patches[pid];
        const int gx = p.atlas_x + i, gy = p.atlas_y + j;
        if (gx < 0 || gy < 0 || gx >= aw || gy >= ah) return false;
        if (!cov[size_t(gy) * aw + size_t(gx)]) return false;
        const uint32_t Np = next_pow2_u32(uint32_t(std::max(p.tex_w, p.tex_h)));
        const float d0 = glm::dot(lo, p.basis_u);
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
        if (L >= int(dc.levels.size())) return false;
        const auto& lev = dc.levels[L];
        if (idx >= lev.meta.size()) return false;
        const uint32_t m = lev.meta[idx];
        if (!(m & 0x80000000u)) return false;
        if (lev.data.size() < size_t(idx) * 4 + 4) return false;
        const uint8_t* dv = lev.data.data() + size_t(idx) * 4;
        const uint32_t q0 = dv[0] | (uint32_t(dv[1]) << 8);
        const uint32_t q1 = dv[2] | (uint32_t(dv[3]) << 8);
        if (q0 == 0 || q1 == 0) return false;
        const float dmin = gdmin + (gdmax - gdmin) * float(q0 - 1) / 65534.0f;
        const float dmax = gdmin + (gdmax - gdmin) * float(q1 - 1) / 65534.0f;
        const float cu0 = dmin + d0, cu1 = dmax + d0;
        // The conservative band is the surface's depth extent across the whole
        // texel square; on a tilted texel the near edge sits up to a full
        // thickness in front of the surface, and at patch seams (where the
        // surface is most oblique to the depth axis) that bias piles into a
        // coherent ridge. Re-anchor on the midpoint and keep only mid_frac of
        // the thickness so the stamped surface tracks the true surface; the
        // extent is capped at one cell so a steep texel can never push its
        // column more than half a cell in front of the texel-centre surface.
        if (mid_frac < 1.0f) {
            const float bm_mid = 0.5f * (cu0 + cu1);
            const float bm_hw  = std::min(0.5f * (cu1 - cu0) * mid_frac, 0.5f * cap);
            bm_lo = bm_mid - bm_hw; bm_hi = bm_mid + bm_hw;
        } else {
            bm_lo = cu0; bm_hi = cu1;
        }
        return true;
    };

    VoxelStats s;
    out.info.reserve(patches.size());
    out.occupied.reserve(patches.size());

    for (int pid = 0; pid < (int)patches.size(); ++pid) {
        const auto& p = patches[pid];
        if (p.tex_w <= 0 || p.tex_h <= 0) continue;
        const int tw = p.tex_w, th = p.tex_h;
        const float da = p.proj_size.x / float(tw);
        const float db = p.proj_size.y / float(th);
        const float cap = std::max(da, db);

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
        const vec3 u = p.basis_u;

        // Pass 1: depth extent of the mid-anchored bands.
        float cmin = 1e30f, cmax = -1e30f;
        uint64_t pcov = 0;
        for (int j = 0; j < th; ++j)
            for (int i = 0; i < tw; ++i) {
                const int gx = p.atlas_x + i, gy = p.atlas_y + j;
                if (gx < 0 || gy < 0 || gx >= aw || gy >= ah) continue;
                if (!cov[size_t(gy) * aw + size_t(gx)]) continue;
                s.covered++;
                pcov++;
                float b0, b1;
                if (!decode(pid, i, j, cap, mid_frac, b0, b1)) { s.skip_not_covered++; continue; }
                cmin = std::min(cmin, b0); cmax = std::max(cmax, b1);
            }
        if (cmin > cmax) continue;
        if (cmax - cmin < 1e-9f) cmax = cmin + cap;

        // Local grid: the texel footprint padded by half a cell (so adjacent
        // patches' boxes meet instead of leaving a sub-cell step at their
        // shared face) × the depth extent at ~one cell of depth resolution.
        // The depth gets one cell per texel like the projection axes; the old
        // hard 64-slice cap compressed steep/curved surfaces (e.g. a bunny's
        // flank) into terraced steps much coarser than the texel pitch, so the
        // limit is now driven by the texel cell size with a generous memory cap.
        const int kMaxDepthSlices = 512;
        const float pad_a = pad_frac * da, pad_b = pad_frac * db;
        const float a0w = p.proj_min.x - pad_a, a1w = p.proj_min.x + p.proj_size.x + pad_a;
        const float b0w = p.proj_min.y - pad_b, b1w = p.proj_min.y + p.proj_size.y + pad_b;
        const int nx = std::max(1, int(std::ceil((a1w - a0w) / da - 1e-6f)));
        const int ny = std::max(1, int(std::ceil((b1w - b0w) / db - 1e-6f)));
        const int nz = std::clamp(int(std::lround((cmax - cmin) / cap)), 1, kMaxDepthSlices);
        const float dd = (cmax - cmin) / float(nz);

        PatchVox gv;
        gv.ex = glm::vec4(ex, 0.0f); gv.ey = glm::vec4(ey, 0.0f); gv.u = glm::vec4(u, 0.0f);
        gv.local_min = glm::vec4(a0w, b0w, cmin, 0.0f);
        gv.cell = glm::vec4(da, db, dd, 0.0f);
        gv.dims = glm::uvec4(uint32_t(nx), uint32_t(ny), uint32_t(nz), 0u);
        gv.word_off_pad.x = uint32_t(out.words.size());
        // World AABB of the grid box. The frame axes are world axes (u = ±unit
        // axis, ex/ey the other two), so the component min/max of the box
        // corners is exact.
        {
            vec3 wlo(1e30f), whi(-1e30f);
            auto grow = [&](const vec3& ax, float c0, float c1) {
                for (int k = 0; k < 3; ++k) {
                    if (std::abs(ax[k]) < 0.5f) continue;
                    if (ax[k] > 0.0f) { wlo[k] = std::min(wlo[k], c0); whi[k] = std::max(whi[k], c1); }
                    else              { wlo[k] = std::min(wlo[k], -c1); whi[k] = std::max(whi[k], -c0); }
                }
            };
            grow(ex, a0w, a1w); grow(ey, b0w, b1w); grow(u, cmin, cmax);
            gv.world_min = glm::vec4(wlo, 0.0f); gv.world_max = glm::vec4(whi, 0.0f);
        }

        const size_t nbits = size_t(nx) * ny * nz;
        const size_t nwords = (nbits + 31u) / 32u;
        const size_t woff = out.words.size();
        out.words.resize(woff + nwords, 0u);

        // Pass 2: stamp the covered texel columns into the local grid.
        auto set = [&](int x, int y, int z) {
            const size_t idx = size_t(z) * nx * ny + size_t(y) * nx + size_t(x);
            out.words[woff + (idx >> 5)] |= uint32_t(1u << (idx & 31u));
        };
        for (int j = 0; j < th; ++j)
            for (int i = 0; i < tw; ++i) {
                float bm_lo, bm_hi;
                if (!decode(pid, i, j, cap, mid_frac, bm_lo, bm_hi)) continue;
                const float a0 = p.proj_min.x + float(i) * da - pad_a;
                const float a1 = p.proj_min.x + float(i + 1) * da + pad_a;
                const float b0 = p.proj_min.y + float(j) * db - pad_b;
                const float b1 = p.proj_min.y + float(j + 1) * db + pad_b;
                const int gx0 = std::clamp(int(std::floor((a0 - a0w) / da)), 0, nx - 1);
                const int gx1 = std::clamp(int(std::floor((a1 - a0w) / da)), 0, nx - 1);
                const int gy0 = std::clamp(int(std::floor((b0 - b0w) / db)), 0, ny - 1);
                const int gy1 = std::clamp(int(std::floor((b1 - b0w) / db)), 0, ny - 1);
                const int gz0 = std::clamp(int(std::floor((bm_lo - cmin) / dd)), 0, nz - 1);
                const int gz1 = std::clamp(int(std::floor((bm_hi - cmin) / dd)), 0, nz - 1);
                for (int z = gz0; z <= gz1; ++z)
                    for (int y = gy0; y <= gy1; ++y)
                        for (int x = gx0; x <= gx1; ++x)
                            set(x, y, z);
                s.stamped++;
            }
        uint64_t pocc = 0;
        for (size_t k = woff; k < woff + nwords; ++k) pocc += std::popcount(out.words[k]);
        out.occupied.push_back(pocc);
        out.info.push_back(gv);
    }

    s.nocc = 0;
    for (uint32_t w : out.words) s.nocc += std::popcount(w);
    s.ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    st = s;
    return out;
}

// --- BVH over the per-patch grid AABBs (median split) -----------------------
// Same layout the shader expects: left = node+1, right = node + val (relative).

struct BvhNodeGpu {
    glm::vec4 amin, amax;
    uint32_t val = 0, is_leaf = 0;
    uint32_t _pad0 = 0, _pad1 = 0;  // pad to 48 B to match GLSL std430 array stride
};

struct BvhPrim {
    vec3 lo, hi;
    uint32_t index = 0;
};

static int build_bvh_rec(std::vector<BvhPrim>& prims, int begin, int end,
                         std::vector<BvhNodeGpu>& nodes) {
    vec3 lo(1e30f), hi(-1e30f);
    for (int i = begin; i < end; ++i) { lo = glm::min(lo, prims[i].lo); hi = glm::max(hi, prims[i].hi); }
    const int self = int(nodes.size());
    nodes.push_back(BvhNodeGpu{});
    nodes[self].amin = glm::vec4(lo, 0.0f);
    nodes[self].amax = glm::vec4(hi, 0.0f);
    if (end - begin == 1) {
        nodes[self].val = prims[begin].index;
        nodes[self].is_leaf = 1;
        return self;
    }
    const vec3 e = hi - lo;
    const int ax = (e.x >= e.y && e.x >= e.z) ? 0 : (e.y >= e.z) ? 1 : 2;
    auto key = [ax](const BvhPrim& p) { return ax == 0 ? p.lo.x : ax == 1 ? p.lo.y : p.lo.z; };
    std::sort(prims.begin() + begin, prims.begin() + end,
              [&](const BvhPrim& a, const BvhPrim& b) { return key(a) < key(b); });
    const int mid = begin + (end - begin) / 2;
    build_bvh_rec(prims, begin, mid, nodes);
    const int right = build_bvh_rec(prims, mid, end, nodes);
    nodes[self].val = uint32_t(right - self);
    nodes[self].is_leaf = 0;
    return self;
}

// --- Merge the per-patch grids into one world-aligned grid (--out dump) -----

static Grid build_merged_grid(const PerPatch& pp, int N) {
    Grid g;
    g.alloc(N, pp.lo, pp.hi);
    for (size_t k = 0; k < pp.info.size(); ++k) {
        const PatchVox& pv = pp.info[k];
        const vec3 ex = glm::vec3(pv.ex), ey = glm::vec3(pv.ey), u = glm::vec3(pv.u);
        const vec3 lm = glm::vec3(pv.local_min), cs = glm::vec3(pv.cell);
        const int nx = int(pv.dims.x), ny = int(pv.dims.y), nz = int(pv.dims.z);
        const size_t woff = pv.word_off_pad.x;
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x) {
                    const size_t idx = size_t(z) * nx * ny + size_t(y) * nx + size_t(x);
                    if (!((pp.words[woff + (idx >> 5)] >> (idx & 31u)) & 1u)) continue;
                    vec3 wlo(1e30f), whi(-1e30f);
                    auto grow = [&](const vec3& ax, float c0, float c1) {
                        for (int k2 = 0; k2 < 3; ++k2) {
                            if (std::abs(ax[k2]) < 0.5f) continue;
                            if (ax[k2] > 0.0f) { wlo[k2] = std::min(wlo[k2], c0); whi[k2] = std::max(whi[k2], c1); }
                            else               { wlo[k2] = std::min(wlo[k2], -c1); whi[k2] = std::max(whi[k2], -c0); }
                        }
                    };
                    const float a0 = lm.x + float(x) * cs.x, a1 = a0 + cs.x;
                    const float b0 = lm.y + float(y) * cs.y, b1 = b0 + cs.y;
                    const float c0 = lm.z + float(z) * cs.z, c1 = c0 + cs.z;
                    grow(ex, a0, a1); grow(ey, b0, b1); grow(u, c0, c1);
                    const int gx0 = std::clamp(int(std::floor((wlo.x - g.lo.x) / g.cs.x)), 0, N - 1);
                    const int gx1 = std::clamp(int(std::floor((whi.x - g.lo.x) / g.cs.x)), 0, N - 1);
                    const int gy0 = std::clamp(int(std::floor((wlo.y - g.lo.y) / g.cs.y)), 0, N - 1);
                    const int gy1 = std::clamp(int(std::floor((whi.y - g.lo.y) / g.cs.y)), 0, N - 1);
                    const int gz0 = std::clamp(int(std::floor((wlo.z - g.lo.z) / g.cs.z)), 0, N - 1);
                    const int gz1 = std::clamp(int(std::floor((whi.z - g.lo.z) / g.cs.z)), 0, N - 1);
                    for (int z2 = gz0; z2 <= gz1; ++z2)
                        for (int y2 = gy0; y2 <= gy1; ++y2)
                            for (int x2 = gx0; x2 <= gx1; ++x2)
                                g.set(x2, y2, z2);
                }
    }
    return g;
}

// --- DDA ray marcher --------------------------------------------------------

static const char* kDdaShaderSrc = R"GLSL(
#version 450 core

layout(local_size_x = 8, local_size_y = 8) in;

// One local grid per atlas patch, aligned to the patch's frame (ex/ey = the
// projection axes, u = the depth axis) with cell = atlas texel, so the depth
// band of a texel column lands on exact cell boundaries instead of a
// world-aligned rounding that fought the neighbouring patch at the border.
struct PatchGrid {
    vec4 ex;
    vec4 ey;
    vec4 u;
    vec4 local_min;      // local (a, b, c) of the grid origin
    vec4 cell;           // local voxel sizes (da, db, dd)
    uvec4 dims;          // (nx, ny, nz)
    uvec4 word_off_pad;  // word offset of this grid's bits in the packed buffer
    vec4 world_min;      // world AABB of the grid box
    vec4 world_max;
};

struct BvhNode {
    vec4 amin;
    vec4 amax;
    uint val;            // leaf: patch index, internal: relative right offset
    uint is_leaf;
};

layout(std430, binding = 0) readonly buffer GridBuf { PatchGrid grids[]; };
layout(std430, binding = 1) readonly buffer WordBuf  { uint words[]; };
layout(std430, binding = 2) readonly buffer BvhBuf   { BvhNode bvh[]; };
layout(rgba8, binding = 3) uniform writeonly image2D u_out;

uniform uint u_grid_count;
uniform vec3 u_cam;
uniform mat4 u_inv_vp;
uniform int  u_color_mode;   // 0 = normal shading, 1 = world position, 2 = patch id
uniform vec3 u_lo;           // model AABB min (world-position colour)
uniform vec3 u_hi;           // model AABB max

bool is_set(uint g, ivec3 c) {
    uvec3 dim = grids[g].dims.xyz;
    if (c.x < 0 || c.y < 0 || c.z < 0 ||
        c.x >= int(dim.x) || c.y >= int(dim.y) || c.z >= int(dim.z)) return false;
    // A patch grid's dims stay well under 2^16 in projection, so the linear
    // index z*nx*ny + y*nx + x fits 32 bits comfortably.
    uint r   = uint(c.y) * dim.x + uint(c.x);
    uint idx = uint(c.z) * (dim.x * dim.y) + r;
    uint w   = grids[g].word_off_pad.x + (idx >> 5u);
    return ((words[w] >> (idx & 31u)) & 1u) != 0u;
}

bool ray_aabb(vec3 ro, vec3 rd, vec3 inv, vec3 amin, vec3 amax,
              out float t_in, out float t_out) {
    vec3 t0 = (amin - ro) * inv * sign(rd);
    vec3 t1 = (amax - ro) * inv * sign(rd);
    vec3 tmn = min(t0, t1);
    vec3 tmx = max(t0, t1);
    t_in  = max(tmn.x, max(tmn.y, tmn.z));
    t_out = min(tmx.x, min(tmx.y, tmx.z));
    return t_in <= t_out;
}

// Surface normal in the patch's local frame: central differences over the six
// neighbours, negated so the result points from the occupied side (interior)
// towards the empty side (the surface). Out-of-range neighbours read as empty,
// so voxels on the grid boundary get their true outward normal. Mapped to world
// via the patch basis (orthonormal, so reflections are handled by the mapping).
// Falls back to 0 for fully-surrounded voxels (there the caller keeps the DDA
// crossing normal).
vec3 occupancy_normal_world(uint g, ivec3 c) {
    ivec3 gr = ivec3(0);
    gr.x = (is_set(g, c + ivec3(1, 0, 0)) ? 1 : 0) - (is_set(g, c - ivec3(1, 0, 0)) ? 1 : 0);
    gr.y = (is_set(g, c + ivec3(0, 1, 0)) ? 1 : 0) - (is_set(g, c - ivec3(0, 1, 0)) ? 1 : 0);
    gr.z = (is_set(g, c + ivec3(0, 0, 1)) ? 1 : 0) - (is_set(g, c - ivec3(0, 0, 1)) ? 1 : 0);
    vec3 n = -vec3(gr);
    if (dot(n, n) < 1e-6) return vec3(0.0);
    vec3 nw = grids[g].ex.xyz * n.x + grids[g].ey.xyz * n.y + grids[g].u.xyz * n.z;
    return normalize(nw);
}

vec3 patch_color(uint g) {
    uint h = g * 2654435761u;
    vec3 c = vec3(float(h & 0xFFu), float((h >> 8) & 0xFFu), float((h >> 16) & 0xFFu)) / 255.0;
    return mix(c, vec3(1.0), 0.35);
}

// DDA through one patch's local grid between the world-space entry/exit t's.
// The frame transform is orthonormal, so t stays in world units and is
// comparable across patches. Returns the entry t of the first occupied cell.
bool march_grid(uint g, vec3 ro, vec3 rd, float tN, float tF,
                out ivec3 cell, out ivec3 nrm, out float ht) {
    vec3 ex = grids[g].ex.xyz, ey = grids[g].ey.xyz, u = grids[g].u.xyz;
    vec3 lm = grids[g].local_min.xyz, ce = grids[g].cell.xyz;
    uvec3 dim = grids[g].dims.xyz;

    vec3 roL = vec3(dot(ro, ex), dot(ro, ey), dot(ro, u));
    vec3 rdL = vec3(dot(rd, ex), dot(rd, ey), dot(rd, u));
    vec3 pos = (roL + rdL * max(tN, 0.0) - lm) / ce;
    cell = clamp(ivec3(floor(pos)), ivec3(0), ivec3(dim) - 1);

    // DDA: d is in cells per world unit, so t stays in world units.
    vec3 d = rdL / ce;
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
    // axis that attains the max entry time. A hit in the very first cell (an
    // entry-cell hit) lies on the patch shell against that grid face, so its
    // outward normal is axis-aligned (-step along that axis).
    vec3 hi = lm + ce * vec3(dim);
    vec3 invR = 1.0 / max(abs(rdL), vec3(1e-30));
    vec3 t0 = (lm - roL) * invR * sign(rdL);
    vec3 t1 = (hi - roL) * invR * sign(rdL);
    vec3 tmn = min(t0, t1);
    int entryAxis = 0;
    float bestT = tmn.x;
    if (tmn.y > bestT) { bestT = tmn.y; entryAxis = 1; }
    if (tmn.z > bestT) { entryAxis = 2; }
    nrm = ivec3(0);
    nrm[entryAxis] = -step[entryAxis];

    float base_t = max(tN, 0.0);
    float cur_t = base_t;
    int it = 0;
    int maxSteps = 3 * (int(dim.x) + int(dim.y) + int(dim.z)) + 16;
    while (it++ < maxSteps) {
        if (is_set(g, cell)) { ht = cur_t; return true; }
        if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
            cur_t = base_t + tMax.x; pos.x += float(step.x); cell.x += step.x; tMax.x += tDelta.x;
            nrm = ivec3(-step.x, 0, 0);
        } else if (tMax.y <= tMax.z) {
            cur_t = base_t + tMax.y; pos.y += float(step.y); cell.y += step.y; tMax.y += tDelta.y;
            nrm = ivec3(0, -step.y, 0);
        } else {
            cur_t = base_t + tMax.z; pos.z += float(step.z); cell.z += step.z; tMax.z += tDelta.z;
            nrm = ivec3(0, 0, -step.z);
        }
        if (cell.x < 0 || cell.y < 0 || cell.z < 0 ||
            cell.x >= int(dim.x) || cell.y >= int(dim.y) || cell.z >= int(dim.z)) break;
    }
    return false;
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

    bool hit = false;
    float best_t = 1e30;
    uint hit_g = 0u;
    ivec3 hit_cell = ivec3(0), hit_nrm = ivec3(0);

    if (u_grid_count > 0u) {
        const int MAX_STACK = 32;
        int st_node[MAX_STACK];
        float st_tin[MAX_STACK];
        int sp = 0;

        vec3 invR = 1.0 / max(abs(rd), vec3(1e-30));
        float rt0, rt1;
        // Trace the patch BVH, marching each candidate patch's local grid and
        // keeping the nearest voxel across all grids. Nodes whose entry is
        // already behind the current best hit are pruned.
        if (ray_aabb(ro, rd, invR, bvh[0].amin.xyz, bvh[0].amax.xyz, rt0, rt1) && rt1 >= 0.0) {
            int node = 0; float tin = rt0;
            for (;;) {
                BvhNode b = bvh[node];
                if (b.is_leaf != 0u) {
                    uint g = b.val;
                    float g0, g1;
                    if (ray_aabb(ro, rd, invR, grids[g].world_min.xyz, grids[g].world_max.xyz, g0, g1) &&
                        g1 >= 0.0 && g0 <= best_t) {
                        ivec3 cell, nrm; float ht;
                        if (march_grid(g, ro, rd, max(tin, g0), g1, cell, nrm, ht) && ht < best_t) {
                            best_t = ht; hit = true; hit_g = g; hit_cell = cell; hit_nrm = nrm;
                        }
                    }
                    if (sp == 0) break;
                    --sp; node = st_node[sp]; tin = st_tin[sp];
                    continue;
                }
                int left = node + 1;
                int right = node + int(b.val);
                float t0, t1, t2, t3;
                bool hl = ray_aabb(ro, rd, invR, bvh[left].amin.xyz, bvh[left].amax.xyz, t0, t1);
                bool hr = ray_aabb(ro, rd, invR, bvh[right].amin.xyz, bvh[right].amax.xyz, t2, t3);
                int near_n = -1, far_n = -1;
                float near_t = 0.0, far_t = 0.0;
                bool has_near = false, has_far = false;
                if (hl && hr) {
                    if (t0 <= t2) { near_n = left; near_t = t0; far_n = right; far_t = t2; }
                    else { near_n = right; near_t = t2; far_n = left; far_t = t0; }
                    has_near = true; has_far = true;
                } else if (hl) { near_n = left; near_t = t0; has_near = true; }
                else if (hr) { near_n = right; near_t = t2; has_near = true; }
                if (has_far && far_t <= best_t && sp < MAX_STACK) {
                    st_node[sp] = far_n; st_tin[sp] = far_t; ++sp;
                }
                if (!has_near || near_t > best_t) {
                    if (sp == 0) break;
                    --sp; node = st_node[sp]; tin = st_tin[sp];
                    continue;
                }
                node = near_n; tin = near_t;
            }
        }
    }

    if (hit) {
        uint g = hit_g;
        if (u_color_mode == 1) {
            // World-position colour: maps the hit voxel's world position onto
            // the model AABB to a continuous RGB ramp. Independent of the
            // occupancy normals, so any leftover banding must be geometry.
            vec3 ex = grids[g].ex.xyz, ey = grids[g].ey.xyz, u = grids[g].u.xyz;
            vec3 lm = grids[g].local_min.xyz, ce = grids[g].cell.xyz;
            vec3 w = ex * (lm.x + (float(hit_cell.x) + 0.5) * ce.x) +
                     ey * (lm.y + (float(hit_cell.y) + 0.5) * ce.y) +
                     u  * (lm.z + (float(hit_cell.z) + 0.5) * ce.z);
            col = clamp((w - u_lo) / max(u_hi - u_lo, vec3(1e-6)), vec3(0.0), vec3(1.0));
        } else if (u_color_mode == 2) {
            // Patch-id colour: each patch's voxel contribution in its own
            // colour, so the seam behaviour on a border is attributable.
            col = patch_color(g);
        } else {
            vec3 n = occupancy_normal_world(g, hit_cell);
            if (dot(n, n) < 1e-6) {
                vec3 ex = grids[g].ex.xyz, ey = grids[g].ey.xyz, u = grids[g].u.xyz;
                n = normalize(ex * float(hit_nrm.x) + ey * float(hit_nrm.y) + u * float(hit_nrm.z));
            }
            vec3 base = n * 0.5 + 0.5;                          // normal -> RGB
            float diff = max(dot(n, normalize(vec3(0.5, 0.75, 0.35))), 0.0);
            float fog = exp(-0.08 * max(best_t, 0.0));
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
    // The per-patch grids resolve each patch's atlas texels, so for the voxels
    // to be cell-sized the atlas texels must be as dense as the grid. Use the
    // auto density (config.auto_target texels across the model's longest axis)
    // instead of the fixed default, mirroring the example-32 loader.
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
    int color_mode = 0;

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
        else if (a == "--color-mode") { if (i + 1 < argc) color_mode = std::atoi(argv[++i]); }
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
    PerPatch pp_;
    VoxelStats vstats;
    double last_vox_ms = 0.0, avg_vox_ms = 0.0;
    uint64_t vox_runs = 0;
    std::vector<BvhNodeGpu> bvh_cpu;

    auto rebuild_bvh = [&] {
        std::vector<BvhPrim> prims(pp_.info.size());
        for (size_t i = 0; i < pp_.info.size(); ++i) {
            prims[i].lo = glm::vec3(pp_.info[i].world_min);
            prims[i].hi = glm::vec3(pp_.info[i].world_max);
            prims[i].index = uint32_t(i);
        }
        bvh_cpu.clear();
        if (!prims.empty()) build_bvh_rec(prims, 0, int(prims.size()), bvh_cpu);
    };

    auto write_merged_if = [&] {
        if (!out_path.empty()) {
            Grid m = build_merged_grid(pp_, grid);
            write_grid(out_path, m);
        }
    };

    // --- GPU resources ---
    gl::Buffer grids_ssbo(gl::BufferType::shader);
    gl::Buffer words_ssbo(gl::BufferType::shader);
    gl::Buffer bvh_ssbo(gl::BufferType::shader);
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

    auto upload_grid = [&] {
        const std::vector<uint32_t> dummy(1, 0u);
        grids_ssbo = gl::Buffer(gl::BufferType::shader);
        words_ssbo = gl::Buffer(gl::BufferType::shader);
        bvh_ssbo   = gl::Buffer(gl::BufferType::shader);
        grids_ssbo.data(pp_.info.empty()  ? (const void*)dummy.data() : (const void*)pp_.info.data(),
                        pp_.info.empty()  ? sizeof(uint32_t)          : sizeof(PatchVox) * pp_.info.size());
        words_ssbo.data(pp_.words.empty() ? (const void*)dummy.data() : (const void*)pp_.words.data(),
                        pp_.words.empty() ? sizeof(uint32_t)          : sizeof(uint32_t) * pp_.words.size());
        bvh_ssbo.data(  bvh_cpu.empty()   ? (const void*)dummy.data() : (const void*)bvh_cpu.data(),
                        bvh_cpu.empty()   ? sizeof(uint32_t)          : sizeof(BvhNodeGpu) * bvh_cpu.size());
        grids_ssbo.bind_base(0);
        words_ssbo.bind_base(1);
        bvh_ssbo.bind_base(2);
    };

    auto do_voxelize = [&] {
        VoxelStats s;
        PerPatch pp = voxelize_patches(atlas, s, pad_arg, mid_arg);
        pp_ = std::move(pp);
        vstats = s;
        last_vox_ms = s.ms;
        avg_vox_ms = vox_runs ? (avg_vox_ms * double(vox_runs) + s.ms) / double(vox_runs + 1) : s.ms;
        vox_runs++;
        rebuild_bvh();
        write_merged_if();
        upload_grid();
        std::printf("Voxelised %zu patch grids in %.1f ms: %llu texels, %llu stamped, %llu occupied (%.2f%% of the model's texel volume), grids %.2f MiB\n",
                    pp_.info.size(), s.ms, (unsigned long long)s.covered,
                    (unsigned long long)s.stamped, (unsigned long long)s.nocc,
                    100.0 * double(s.nocc) / (double(size_t(grid) * grid * grid)),
                    pp_.mib());
    };
    do_voxelize();
    {
        std::printf("DBG cam_pos=(%.3f %.3f %.3f) target=(%.3f %.3f %.3f)\n",
                    cam.position().x, cam.position().y, cam.position().z,
                    cam_target.x, cam_target.y, cam_target.z);
        std::printf("DBG model aabb lo=(%.3f %.3f %.3f) hi=(%.3f %.3f %.3f)\n",
                    pp_.lo.x, pp_.lo.y, pp_.lo.z, pp_.hi.x, pp_.hi.y, pp_.hi.z);
        if (!bvh_cpu.empty()) {
            glm::vec3 rl = glm::vec3(bvh_cpu[0].amin), rh = glm::vec3(bvh_cpu[0].amax);
            std::printf("DBG bvh root lo=(%.3f %.3f %.3f) hi=(%.3f %.3f %.3f)\n",
                        rl.x, rl.y, rl.z, rh.x, rh.y, rh.z);
        }
        {
            FILE* f = std::fopen("/tmp/opencode/dbg_pp.bin", "wb");
            if (f) {
                uint32_t n = uint32_t(pp_.info.size());
                std::fwrite(&n, 4, 1, f);
                for (const auto& pv : pp_.info)
                    std::fwrite(&pv, sizeof(PatchVox), 1, f);
                std::fwrite(pp_.words.data(), 4, pp_.words.size(), f);
                uint32_t nb = uint32_t(bvh_cpu.size());
                std::fwrite(&nb, 4, 1, f);
                for (const auto& b : bvh_cpu) {
                    std::fwrite(&b.amin, 16, 1, f);
                    std::fwrite(&b.amax, 16, 1, f);
                    std::fwrite(&b.val, 4, 1, f);
                    std::fwrite(&b.is_leaf, 4, 1, f);
                }
                std::fclose(f);
                std::printf("DBG dumped %u grids, %zu words, %u bvh nodes\n", n, pp_.words.size(), nb);
            }
        }
    }

    // --- GPU resources ---
    const GLint uGridCount = dda_prog.uniform_location("u_grid_count");
    const GLint uLo        = dda_prog.uniform_location("u_lo");
    const GLint uHi        = dda_prog.uniform_location("u_hi");
    const GLint uCam       = dda_prog.uniform_location("u_cam");
    const GLint uInvVp     = dda_prog.uniform_location("u_inv_vp");
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

    upload_grid();

    // --- Stats ---
    std::vector<float> frame_times;
    double disp_frame_period = 0.0, period_acc = 0.0;
    int period_n = 0;
    double last = window.time(), win_start = window.time();
    double prev_x = 0, prev_y = 0;
    bool prev_lmb = false;
    bool frame_dumped = false;
    int insp_patch = 0;
    float mid_last = mid_arg, pad_last = pad_arg;

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

        // Model switch via the number keys (edge-triggered; the rebuild can be
        // slow, so never fire on held keys).
        static bool prev_model_keys[3] = {};
        const gfx::Key model_keys[3] = {gfx::Key::_1, gfx::Key::_2, gfx::Key::_3};
        for (int m = 0; m < 3; ++m) {
            const bool down = window.key_down(model_keys[m]);
            const bool press = down && !prev_model_keys[m];
            prev_model_keys[m] = down;
            if (press && m != cur_model) {
                if (load_scene(models[m], model, atlas, tol_frac, density_arg)) {
                    cur_model = m;
                    refresh_camera();
                    do_voxelize();
                } else {
                    std::fprintf(stderr, "Failed to switch to %s\n", models[m].name);
                }
            }
        }

        resize_render();
        int w = window.framebuffer_width(), h = window.framebuffer_height();
        if (w > 0 && h > 0) cam.set_aspect(float(w) / float(h));

        // --- DDA ray march through the per-patch grids ---
        {
            glm::mat4 inv_vp = glm::inverse(cam.view_projection());
            dda_prog.use();
            dda_prog.uniform1ui(uGridCount, uint32_t(pp_.info.size()));
            dda_prog.uniform3fv(uLo, glm::value_ptr(pp_.lo));
            dda_prog.uniform3fv(uHi, glm::value_ptr(pp_.hi));
            dda_prog.uniform3fv(uCam, glm::value_ptr(cam.position()));
            dda_prog.uniform_matrix4fv(uInvVp, glm::value_ptr(inv_vp));
            dda_prog.uniform1i(uColorMode, color_mode);
            out_tex.bind_image(3, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
            grids_ssbo.bind_base(0);
            words_ssbo.bind_base(1);
            bvh_ssbo.bind_base(2);
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
            ImGui::SetNextWindowSize(ImVec2(340, 0), ImGuiCond_FirstUseEver);
            ImGui::Begin("33 MDC Voxelizer - Debug", nullptr, ImGuiWindowFlags_NoSavedSettings);

            ImGui::SeparatorText("Performance");
            ImGui::Text("Frametime: %.3f ms   (avg %.3f ms)", dt * 1000.0f, disp_frame_period);
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            if (frame_times.size() > 1)
                ImGui::PlotLines("frame period ms (raw)", frame_times.data(),
                                 int(frame_times.size()), 0, nullptr,
                                 0.0f, 0.0f, ImVec2(0, 48));

            ImGui::SeparatorText("Voxelization");
            ImGui::Text("Patch grids: %zu  (%.2f MiB)", pp_.info.size(), pp_.mib());
            ImGui::Text("Occupied: %llu voxels",
                        (unsigned long long)vstats.nocc);
            ImGui::Text("Voxelization: %.1f ms  (avg %.1f ms, %llu runs)",
                        last_vox_ms, avg_vox_ms, (unsigned long long)vox_runs);
            if (ImGui::SliderFloat("Mid band", &mid_arg, 0.05f, 1.0f, "%.2f") && mid_arg != mid_last) {
                mid_last = mid_arg;
                do_voxelize();
            }
            if (ImGui::SliderFloat("Footprint pad", &pad_arg, 0.0f, 1.0f, "%.2f") && pad_arg != pad_last) {
                pad_last = pad_arg;
                do_voxelize();
            }
            ImGui::TextWrapped("mid re-anchors each depth band on its midpoint "
                               "(1.0 = full band); pad overlaps neighbouring "
                               "patches' footprints by that fraction of a cell.");
            if (ImGui::Button("Revoxelize")) do_voxelize();

            ImGui::SeparatorText("Scene");
            ImGui::Combo("Colour mode", &color_mode, "Normal\0World position\0Patch id\0");
            int cur_model_prev = cur_model;
            if (ImGui::Combo("Model", &cur_model, "Cornell Box\0Stanford Bunny\0Stanford Dragon\0")) {
                if (load_scene(models[cur_model], model, atlas, tol_frac, density_arg)) {
                    refresh_camera();
                    do_voxelize();
                } else {
                    std::fprintf(stderr, "Failed to switch to %s\n", models[cur_model].name);
                    cur_model = cur_model_prev;
                }
            }
            ImGui::Text("Atlas: %dx%d @ %.0f texels/unit",
                        atlas.atlas_width(), atlas.atlas_height(), atlas.final_density());

            if (!pp_.info.empty()) {
                ImGui::SeparatorText("Patch inspector");
                insp_patch = std::clamp(insp_patch, 0, int(pp_.info.size()) - 1);
                ImGui::SliderInt("Patch", &insp_patch, 0, int(pp_.info.size()) - 1);
                const PatchVox& pv = pp_.info[insp_patch];
                ImGui::Text("cells %ux%ux%u, occupied %llu, depth slices %u",
                            pv.dims.x, pv.dims.y, pv.dims.z,
                            (unsigned long long)pp_.occupied[insp_patch],
                            uint32_t(pv.dims.z));
            }

            if (!out_path.empty()) {
                ImGui::SeparatorText("Merged --out grid");
                ImGui::SliderFloat("Resolution scale", &res_scale, 0.25f, 2.0f, "%.2fx");
                const int next_n = std::clamp(round_pow2(int(std::lround(base_grid * res_scale))), 64, 2048);
                ImGui::Text("  -> grid %d^3 (%.1f MiB)", next_n,
                            double(size_t(next_n) * next_n * next_n) / 8.0 / (1024.0 * 1024.0));
                if (ImGui::IsItemDeactivatedAfterEdit() && next_n != grid) {
                    grid = next_n;
                    write_merged_if();
                }
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

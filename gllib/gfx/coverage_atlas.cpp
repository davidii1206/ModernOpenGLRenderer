// Mesh -> occlusion-free patch atlas pipeline.
//
// CPU-only: welds and clusters a loaded model into occlusion-free patches,
// packs them into a texture atlas, builds a BVH, rasterises depth/thickness/UV
// coverage and emits three adaptive MIP4 sparse mip-chains (docs/mip4.md).
// No OpenGL is required; see examples/31_mesh_decomposition for a thin driver.

#include <gfx/coverage_atlas.hpp>
#include <gfx/mesh.hpp>
#include <gfx/model.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <filesystem>

namespace {

using Vec2 = glm::vec2;
using Vec3 = glm::vec3;

// ============================================================ constants ==

constexpr int   TOPOLOGY_TEST_RES  = 64;
constexpr int   MAX_GRID_DIM       = 512;
constexpr int   SAH_BINS           = 16;
constexpr int   BVH_MAX_LEAF       = 1;

constexpr int   MIP_MAX_LEVELS = 12;
constexpr int   MIP_MAX_TEXW   = 16384;    // max level texture width
constexpr uint32_t MIP_LEAF    = 0xFFFFFFFFu;

const Vec3 axis_dirs[6] = {
    {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
};
const char* axis_names[6] = { "+X", "-X", "+Y", "-Y", "+Z", "-Z" };

// ================================================================ math ==

struct AABB { Vec3 lo = {1e30f, 1e30f, 1e30f}, hi = {-1e30f, -1e30f, -1e30f}; };
static void grow(AABB& b, Vec3 p) { b.lo = glm::min(b.lo, p); b.hi = glm::max(b.hi, p); }
static void grow_aabb(AABB& b, const AABB& o) { grow(b, o.lo); grow(b, o.hi); }
static Vec3 center(AABB b) { return (b.lo + b.hi) * 0.5f; }
static Vec3 extent(AABB b) { return b.hi - b.lo; }
static float surface_area(AABB b) {
    Vec3 e = extent(b);
    return 2.0f * (e.x*e.y + e.y*e.z + e.x*e.z);
}
static Vec3 safe_normalize(Vec3 v) {
    float l = glm::length(v);
    return l > 1e-8f ? v * (1.0f / l) : Vec3{0, 1, 0};
}

static Vec2 project_along(int axis, Vec3 p) {
    switch (axis) {
        case 0: case 1: return {p.y, p.z};
        case 2: case 3: return {p.x, p.z};
        case 4: case 5: return {p.x, p.y};
    }
    return {0, 0};
}
// Depth is measured from the model's AABB minimum along the projection axis
// (not from world-space origin), so the stored values are relative to the
// model itself and land in [0, extent]. All occlusion/thickness tests only use
// depth *differences*, so the constant offset never changes their results.
static Vec3 g_depth_origin = {0.0f, 0.0f, 0.0f};
static float depth_along(int axis, Vec3 p) {
    return glm::dot(p - g_depth_origin, axis_dirs[axis]);
}
static float edge_fn_2d(Vec2 a, Vec2 b, Vec2 p) {
    return (b.x-a.x)*(p.y-a.y) - (b.y-a.y)*(p.x-a.x);
}

// ========================================================= structures ==

using Patch = gfx::CoverageAtlas::Patch;
using BVHNode = gfx::CoverageAtlas::BVHNode;
using MipLevel = gfx::CoverageAtlas::MipLevel;
using MipChain = gfx::CoverageAtlas::MipChain;
using CoverageAtlasConfig = gfx::CoverageAtlasConfig;

struct Triangle {
    unsigned int v[3];
    Vec3  centroid;
    Vec3  normal;
    float axis_pref[6];     // dot(normal, axis_dir), sorted descending
    int   axis_idx[6];      // corresponding DOMINANT_AXES index
    int   patch_id;         // -1 = unassigned
};

// Clustering patch under construction (occupancy grid).
struct Cluster {
    int  id;
    int  axis;
    std::vector<int> tris;
    int  grid_w = 0, grid_h = 0;
    Vec2 grid_origin = {0, 0};
    float scale = 100.0f;
    std::vector<bool> grid;
};

struct DepthCell {
    bool  has_sample = false;
    float min_depth  = 0;
    float max_depth  = 0;
    int   tri_id     = -1;
};

struct UnionFind {
    std::vector<int> parent, rank;
    explicit UnionFind(int n) : parent(n), rank(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    int find(int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    int unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return a;
        if (rank[a] < rank[b]) std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b]) rank[a]++;
        return a;
    }
};

// ================================================= adjacency + affinity ==

static uint64_t edge_key(unsigned int a, unsigned int b) {
    if (a > b) std::swap(a, b);
    return (uint64_t(a) << 32) | uint64_t(b);
}

static void build_adjacency(const std::vector<Triangle>& tris,
                            std::vector<std::vector<int>>& adj,
                            std::vector<int>& boundary_flags) {
    size_t n = tris.size();
    adj.assign(n, {});
    boundary_flags.assign(n, 0);

    std::unordered_map<uint64_t, std::vector<int>> edge_map;
    edge_map.reserve(n * 3);
    for (size_t i = 0; i < n; ++i) {
        for (int e = 0; e < 3; ++e) {
            uint64_t key = edge_key(tris[i].v[e], tris[i].v[(e+1)%3]);
            edge_map[key].push_back(int(i));
        }
    }
    for (auto& [key, tris_on_edge] : edge_map) {
        if (tris_on_edge.size() == 2) {
            adj[tris_on_edge[0]].push_back(tris_on_edge[1]);
            adj[tris_on_edge[1]].push_back(tris_on_edge[0]);
        } else if (tris_on_edge.size() == 1) {
            boundary_flags[tris_on_edge[0]] |= 1;
        } else {
            for (int ti : tris_on_edge) boundary_flags[ti] |= 2;
        }
    }
}

static void compute_axis_affinity(Triangle& t) {
    for (int i = 0; i < 6; ++i) {
        t.axis_pref[i] = glm::dot(t.normal, axis_dirs[i]);
        t.axis_idx[i]  = i;
    }
    // Sort descending by affinity
    for (int i = 0; i < 5; ++i)
        for (int j = i+1; j < 6; ++j)
            if (t.axis_pref[j] > t.axis_pref[i]) {
                std::swap(t.axis_pref[i], t.axis_pref[j]);
                std::swap(t.axis_idx[i],  t.axis_idx[j]);
            }
}

// ================================================== conservative raster ==

// Conservative rasterise one triangle into a DepthCell grid.
// Returns the list of cell indices touched.
static std::vector<int> conservative_rasterize(
        Vec2 v0, Vec2 v1, Vec2 v2,
        float d0, float d1, float d2,
        Vec2 grid_origin, float scale, int grid_w, int grid_h)
{
    auto to_fx = [&](float x) { return (x - grid_origin.x) * scale; };
    auto to_fy = [&](float y) { return (y - grid_origin.y) * scale; };

    float fx0=to_fx(v0.x), fy0=to_fy(v0.y);
    float fx1=to_fx(v1.x), fy1=to_fy(v1.y);
    float fx2=to_fx(v2.x), fy2=to_fy(v2.y);

    int gx0 = std::max(0,        (int)std::floor(std::min({fx0,fx1,fx2})));
    int gx1 = std::min(grid_w-1, (int)std::ceil (std::max({fx0,fx1,fx2})));
    int gy0 = std::max(0,        (int)std::floor(std::min({fy0,fy1,fy2})));
    int gy1 = std::min(grid_h-1, (int)std::ceil (std::max({fy0,fy1,fy2})));

    float area2 = edge_fn_2d({fx0,fy0},{fx1,fy1},{fx2,fy2});
    if (std::abs(area2) < 1e-10f) return {};
    float inv = 1.0f / area2;

    std::vector<int> cells;
    for (int gy = gy0; gy <= gy1; ++gy) {
        float py = float(gy) + 0.5f;
        for (int gx = gx0; gx <= gx1; ++gx) {
            float px = float(gx) + 0.5f;
            Vec2 fp = {px, py};
            float w0 = edge_fn_2d({fx1,fy1},{fx2,fy2},fp) * inv;
            float w1 = edge_fn_2d({fx2,fy2},{fx0,fy0},fp) * inv;
            float w2 = 1.0f - w0 - w1;
            if (w0 > 1e-6f && w1 > 1e-6f && w2 > 1e-6f)
                cells.push_back(gy * grid_w + gx);
        }
    }
    return cells;
}

// Barycentric-interpolated depth at a sample point inside the triangle.
static float interp_depth(Vec2 fp, Vec2 v0, Vec2 v1, Vec2 v2,
                           float d0, float d1, float d2)
{
    float fx0=v0.x, fy0=v0.y, fx1=v1.x, fy1=v1.y, fx2=v2.x, fy2=v2.y;
    float area2 = (fx1-fx0)*(fy2-fy0) - (fx2-fx0)*(fy1-fy0);
    if (std::abs(area2) < 1e-10f) return d0;
    float inv = 1.0f / area2;
    float w2 = ((fx1-fx0)*(fp.y-fy0) - (fp.x-fx0)*(fy1-fy0)) * inv;
    float w0 = ((fx2-fx1)*(fp.y-fy1) - (fp.x-fx1)*(fy2-fy1)) * inv;
    return w0*d0 + (1.0f-w0-w2)*d1 + w2*d2;
}

// Grow a DepthCell grid to cover a new 2D triangle.
struct DepthGrid {
    std::vector<DepthCell> cells;
    int gw = 0, gh = 0;
    Vec2 origin = {0, 0};
    float scale = 1.0f;
};

static void grow_depth_grid(DepthGrid& dg, Vec2 v0, Vec2 v1, Vec2 v2) {
    auto to_gx = [&](float x) { return (int)std::floor((x - dg.origin.x) * dg.scale); };
    auto to_gy = [&](float y) { return (int)std::floor((y - dg.origin.y) * dg.scale); };

    if (dg.cells.empty()) {
        float min_x = std::min({v0.x,v1.x,v2.x});
        float max_x = std::max({v0.x,v1.x,v2.x});
        float min_y = std::min({v0.y,v1.y,v2.y});
        float max_y = std::max({v0.y,v1.y,v2.y});
        float span = std::max(max_x-min_x, max_y-min_y);
        float pad = span*0.1f + 0.01f;
        dg.origin = {min_x-pad, min_y-pad};
        dg.gw = std::max(1,(int)std::ceil((max_x-min_x+2*pad)*dg.scale));
        dg.gh = std::max(1,(int)std::ceil((max_y-min_y+2*pad)*dg.scale));
        dg.cells.resize(size_t(dg.gw)*dg.gh);
        return;
    }

    int mgx_min = std::min({to_gx(v0.x),to_gx(v1.x),to_gx(v2.x)});
    int mgx_max = std::max({to_gx(v0.x),to_gx(v1.x),to_gx(v2.x)});
    int mgy_min = std::min({to_gy(v0.y),to_gy(v1.y),to_gy(v2.y)});
    int mgy_max = std::max({to_gy(v0.y),to_gy(v1.y),to_gy(v2.y)});

    if (mgx_min >= 0 && mgx_max < dg.gw && mgy_min >= 0 && mgy_max < dg.gh) return;

    int new_x0 = std::min(0, mgx_min);
    int new_x1 = std::max(dg.gw-1, mgx_max);
    int new_y0 = std::min(0, mgy_min);
    int new_y1 = std::max(dg.gh-1, mgy_max);
    int nw = new_x1 - new_x0 + 1;
    int nh = new_y1 - new_y0 + 1;

    // Keep the grid bounded: growth beyond the cap coarsens the scale and
    // resamples existing samples (merging depth ranges stays conservative).
    if (nw > MAX_GRID_DIM || nh > MAX_GRID_DIM) {
        float unit_w = float(nw) / dg.scale;
        float unit_h = float(nh) / dg.scale;
        float ns = std::min(MAX_GRID_DIM / unit_w, MAX_GRID_DIM / unit_h);
        int nwn = std::max(1, (int)std::ceil(unit_w * ns));
        int nhn = std::max(1, (int)std::ceil(unit_h * ns));
        Vec2 new_origin = {dg.origin.x + float(new_x0)/dg.scale,
                           dg.origin.y + float(new_y0)/dg.scale};
        std::vector<DepthCell> ng(size_t(nwn)*nhn);
        for (int y = 0; y < dg.gh; ++y) {
            for (int x = 0; x < dg.gw; ++x) {
                const DepthCell& c = dg.cells[size_t(y)*dg.gw+x];
                if (!c.has_sample) continue;
                float ux = dg.origin.x + (float(x)+0.5f)/dg.scale;
                float uy = dg.origin.y + (float(y)+0.5f)/dg.scale;
                int nx = std::max(0, std::min(nwn-1, (int)std::floor((ux-new_origin.x)*ns)));
                int ny = std::max(0, std::min(nhn-1, (int)std::floor((uy-new_origin.y)*ns)));
                DepthCell& t = ng[size_t(ny)*nwn+nx];
                if (!t.has_sample) {
                    t = c;
                } else {
                    t.min_depth = std::min(t.min_depth, c.min_depth);
                    t.max_depth = std::max(t.max_depth, c.max_depth);
                }
            }
        }
        dg.origin = new_origin;
        dg.scale = ns;
        dg.gw = nwn; dg.gh = nhn;
        dg.cells = std::move(ng);
        return;
    }

    std::vector<DepthCell> ng(size_t(nw)*nh);
    int sx = -new_x0, sy = -new_y0;
    for (int y = 0; y < dg.gh; ++y)
        for (int x = 0; x < dg.gw; ++x)
            ng[size_t(y+sy)*nw+(x+sx)] = dg.cells[size_t(y)*dg.gw+x];
    dg.origin.x += float(new_x0)/dg.scale;
    dg.origin.y += float(new_y0)/dg.scale;
    dg.gw = nw; dg.gh = nh;
    dg.cells = std::move(ng);
}

// Incremental occlusion-free test: check one new triangle against an existing grid.
static bool occlusion_free_incremental(std::vector<DepthCell>& cells,
                                        int gw, int gh,
                                        Vec2 grid_origin, float grid_scale,
                                        Vec2 v0, Vec2 v1, Vec2 v2,
                                        float d0, float d1, float d2,
                                        int tri_id, float epsilon)
{
    auto touched = conservative_rasterize(v0,v1,v2,d0,d1,d2, grid_origin,grid_scale,gw,gh);
    for (int ci : touched) {
        float depth = interp_depth({float(ci%gw)+0.5f, float(ci/gw)+0.5f},
                                    v0,v1,v2, d0,d1,d2);
        auto& c = cells[ci];
        if (c.has_sample) {
            if (std::abs(depth - c.min_depth) > epsilon && c.tri_id != tri_id)
                return false;
        }
    }
    // All clear — mark cells
    for (int ci : touched) {
        float depth = interp_depth({float(ci%gw)+0.5f, float(ci/gw)+0.5f},
                                    v0,v1,v2, d0,d1,d2);
        auto& c = cells[ci];
        if (!c.has_sample) {
            c.min_depth = c.max_depth = depth;
            c.tri_id = tri_id;
            c.has_sample = true;
        } else {
            c.min_depth = std::min(c.min_depth, depth);
            c.max_depth = std::max(c.max_depth, depth);
        }
    }
    return true;
}

// ============================================== greedy clustering (§3) ==

static void build_patches(std::vector<Triangle>& tris,
                           const std::vector<Vec3>& positions,
                           const std::vector<std::vector<int>>& adj,
                           std::vector<Cluster>& clusters,
                           float density, float epsilon)
{
    int n = int(tris.size());
    std::vector<bool> visited(n, false);

    constexpr int TOPO_RES = TOPOLOGY_TEST_RES;

    // Sort by flatness (best seeds first — same as spec §3 priority)
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return tris[a].axis_pref[0] > tris[b].axis_pref[0];
    });

    int patch_count = 0;

    for (int seed : order) {
        if (visited[seed]) continue;
        int axis = tris[seed].axis_idx[0];

        Cluster patch;
        patch.id = patch_count++;
        patch.axis = axis;

        // Build topology grid for this patch
        DepthGrid dg;
        AABB bb2d;
        {
            Vec2 gv[3]; float gd[3];
            for (int e = 0; e < 3; ++e) {
                gv[e] = project_along(axis, positions[tris[seed].v[e]]);
                gd[e] = depth_along(axis, positions[tris[seed].v[e]]);
                grow(bb2d, {gv[e].x, gv[e].y, 0});
            }
            float span_x = bb2d.hi.x - bb2d.lo.x;
            float span_y = bb2d.hi.y - bb2d.lo.y;
            float span = std::max(span_x, span_y);
            if (span < 1e-6f) span = 1.0f;
            dg.scale = std::min(float(TOPO_RES) / span, 50.0f);
            grow_depth_grid(dg, gv[0], gv[1], gv[2]);
            auto touched = conservative_rasterize(gv[0],gv[1],gv[2],gd[0],gd[1],gd[2],
                                                   dg.origin, dg.scale, dg.gw, dg.gh);
            for (int ci : touched) {
                float depth = interp_depth({float(ci%dg.gw)+0.5f, float(ci/dg.gw)+0.5f},
                                            gv[0],gv[1],gv[2], gd[0],gd[1],gd[2]);
                dg.cells[ci] = {true, depth, depth, seed};
            }
            patch.tris.push_back(seed);
            visited[seed] = true;
            tris[seed].patch_id = patch.id;
        }

        // BFS from seed
        std::queue<int> bfs;
        for (int nb : adj[seed])
            if (!visited[nb]) bfs.push(nb);

        while (!bfs.empty()) {
            int ti = bfs.front(); bfs.pop();
            if (visited[ti]) continue;

            // Axis alignment check (§3: dot with chosen axis must exceed threshold)
            float ndot = glm::dot(tris[ti].normal, axis_dirs[axis]);
            if (ndot < 0.3f) continue;

            // Occlusion-free incremental test
            Vec2 gv[3]; float gd[3];
            for (int e = 0; e < 3; ++e) {
                gv[e] = project_along(axis, positions[tris[ti].v[e]]);
                gd[e] = depth_along(axis, positions[tris[ti].v[e]]);
            }
            grow_depth_grid(dg, gv[0], gv[1], gv[2]);
            if (!occlusion_free_incremental(dg.cells, dg.gw, dg.gh, dg.origin, dg.scale,
                                            gv[0],gv[1],gv[2], gd[0],gd[1],gd[2],
                                            ti, epsilon))
                continue;

            // Accept triangle
            visited[ti] = true;
            tris[ti].patch_id = patch.id;
            patch.tris.push_back(ti);

            for (int nb : adj[ti])
                if (!visited[nb]) bfs.push(nb);
        }

        if (!patch.tris.empty())
            clusters.push_back(std::move(patch));
    }

    printf("  Clustering: %zu final patches (BFS + union-find)\n", clusters.size());
}

// ================================================ cleanup merge (§3.5) ==

// Build a depth grid rasterised from all triangles of a cluster (on its axis).
static void build_cluster_grid(DepthGrid& dg, const Cluster& patch,
                               const std::vector<Triangle>& tris,
                               const std::vector<Vec3>& positions, int axis)
{
    AABB bb2d;
    for (int ti : patch.tris) {
        for (int e = 0; e < 3; ++e) {
            Vec2 gv = project_along(axis, positions[tris[ti].v[e]]);
            grow(bb2d, {gv.x, gv.y, 0});
        }
    }
    float span_x = bb2d.hi.x - bb2d.lo.x;
    float span_y = bb2d.hi.y - bb2d.lo.y;
    float span = std::max(span_x, span_y);
    if (span < 1e-6f) span = 1.0f;

    dg = DepthGrid{};
    dg.scale = std::min(float(TOPOLOGY_TEST_RES) / span, 50.0f);
    for (int ti : patch.tris) {
        Vec2 gv[3]; float gd[3];
        for (int e = 0; e < 3; ++e) {
            gv[e] = project_along(axis, positions[tris[ti].v[e]]);
            gd[e] = depth_along(axis, positions[tris[ti].v[e]]);
        }
        grow_depth_grid(dg, gv[0], gv[1], gv[2]);
        auto touched = conservative_rasterize(gv[0],gv[1],gv[2],gd[0],gd[1],gd[2],
                                               dg.origin, dg.scale, dg.gw, dg.gh);
        for (int ci : touched) {
            float depth = interp_depth({float(ci%dg.gw)+0.5f, float(ci/dg.gw)+0.5f},
                                        gv[0],gv[1],gv[2], gd[0],gd[1],gd[2]);
            dg.cells[ci] = {true, depth, depth, ti};
        }
    }
}

// Test-and-commit: can all triangles in tri_list be added to dg without violating
// occlusion?  The grid is only mutated on success (committed after a full pass),
// so a failed candidate never poisons the shared grid.
static bool occlusion_free_add(DepthGrid& dg,
                               const std::vector<int>& tri_list,
                               const std::vector<Triangle>& tris,
                               const std::vector<Vec3>& positions,
                               int axis, float epsilon)
{
    struct Pending { bool has = false; float min_d = 0, max_d = 0; };
    std::unordered_map<int, Pending> pending;
    pending.reserve(tri_list.size() * 8);

    for (int ti : tri_list) {
        Vec2 gv[3]; float gd[3];
        for (int e = 0; e < 3; ++e) {
            gv[e] = project_along(axis, positions[tris[ti].v[e]]);
            gd[e] = depth_along(axis, positions[tris[ti].v[e]]);
        }
        grow_depth_grid(dg, gv[0], gv[1], gv[2]);
        auto touched = conservative_rasterize(gv[0],gv[1],gv[2],gd[0],gd[1],gd[2],
                                               dg.origin, dg.scale, dg.gw, dg.gh);
        for (int ci : touched) {
            float depth = interp_depth({float(ci%dg.gw)+0.5f, float(ci/dg.gw)+0.5f},
                                        gv[0],gv[1],gv[2], gd[0],gd[1],gd[2]);
            auto& c = dg.cells[ci];
            if (c.has_sample && std::abs(depth - c.min_depth) > epsilon) return false;
            auto it = pending.find(ci);
            if (it != pending.end()) {
                if (std::abs(depth - it->second.min_d) > epsilon) return false;
            } else {
                pending[ci] = {true, depth, depth};
            }
        }
    }

    for (auto& [ci, p] : pending) {
        auto& c = dg.cells[ci];
        if (!c.has_sample) {
            c = DepthCell{true, p.min_d, p.max_d, -1};
        } else {
            c.min_depth = std::min(c.min_depth, p.min_d);
            c.max_depth = std::max(c.max_depth, p.max_d);
        }
    }
    return true;
}

static void merge_pass(std::vector<Cluster>& clusters,
                       std::vector<Triangle>& tris,
                       const std::vector<Vec3>& positions,
                       float epsilon, int min_patch_size, float axis_threshold)
{
    constexpr int MAX_PASSES = 10;
    int total_merged = 0;

    for (int pass = 0; pass < MAX_PASSES; ++pass) {
        size_t n = clusters.size();
        if (n < 2) break;

        // Build cluster adjacency from shared edges
        std::vector<std::vector<int>> patch_adj(n);
        {
            std::unordered_map<uint64_t, int> edge_to_tri;
            for (size_t i = 0; i < tris.size(); ++i) {
                for (int e = 0; e < 3; ++e) {
                    uint64_t key = edge_key(tris[i].v[e], tris[i].v[(e+1)%3]);
                    auto it = edge_to_tri.find(key);
                    if (it != edge_to_tri.end()) {
                        int other = it->second;
                        int pa = tris[i].patch_id;
                        int pb = tris[other].patch_id;
                        if (pa >= 0 && pb >= 0 && pa != pb) {
                            patch_adj[pa].push_back(pb);
                            patch_adj[pb].push_back(pa);
                        }
                        edge_to_tri.erase(it);
                    } else {
                        edge_to_tri[key] = int(i);
                    }
                }
            }
            for (auto& adj : patch_adj) {
                std::sort(adj.begin(), adj.end());
                adj.erase(std::unique(adj.begin(), adj.end()), adj.end());
            }
        }

        // Targets: largest first (absorb into the biggest compatible neighbour)
        std::vector<int> target_order(n);
        std::iota(target_order.begin(), target_order.end(), 0);
        std::sort(target_order.begin(), target_order.end(), [&](int a, int b) {
            return clusters[a].tris.size() > clusters[b].tris.size();
        });

        std::vector<bool> dissolved(n, false);
        int merged_this_pass = 0;
        int fail_axis = 0, fail_occ = 0;

        for (int ni : target_order) {
            if (dissolved[ni]) continue;

            int axis = clusters[ni].axis;

            // Build the target's depth grid ONCE per pass; candidates commit into it.
            DepthGrid dg;
            build_cluster_grid(dg, clusters[ni], tris, positions, axis);

            // Candidate clusters: adjacent, small, undissolved (smallest first)
            std::vector<int> cands;
            for (int nb : patch_adj[ni]) {
                if (dissolved[nb] || nb == ni) continue;
                if (int(clusters[nb].tris.size()) >= min_patch_size) continue;
                cands.push_back(nb);
            }
            if (cands.empty()) continue;
            std::sort(cands.begin(), cands.end(), [&](int a, int b) {
                return clusters[a].tris.size() < clusters[b].tris.size();
            });

            for (int pi : cands) {
                if (dissolved[pi]) continue;

                Vec3 avg_normal = {0, 0, 0};
                for (int ti : clusters[pi].tris) avg_normal += tris[ti].normal;
                avg_normal = safe_normalize(avg_normal);

                float ndot = glm::dot(avg_normal, axis_dirs[axis]);
                if (std::abs(ndot) < axis_threshold) { fail_axis++; continue; }

                if (occlusion_free_add(dg, clusters[pi].tris, tris, positions, axis, epsilon)) {
                    for (int ti : clusters[pi].tris) {
                        tris[ti].patch_id = ni;
                        clusters[ni].tris.push_back(ti);
                    }
                    dissolved[pi] = true;
                    merged_this_pass++;
                } else {
                    fail_occ++;
                }
            }
        }

        printf("  Merge pass %d: absorbed %d patches, %zu remaining"
               "  [fail axis=%d occ=%d]\n",
               pass+1, merged_this_pass, n - merged_this_pass, fail_axis, fail_occ);

        total_merged += merged_this_pass;
        if (merged_this_pass == 0) break;

        std::vector<Cluster> remaining;
        std::vector<int> remap(n, -1);
        int new_id = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!dissolved[i]) {
                remap[i] = new_id++;
                clusters[i].id = remap[i];
                remaining.push_back(std::move(clusters[i]));
            }
        }
        for (auto& t : tris)
            if (t.patch_id >= 0) t.patch_id = remap[t.patch_id];
        clusters = std::move(remaining);
    }
    printf("  Merge pass total: %d absorbed, %zu remaining\n", total_merged, clusters.size());
}

// ================================================= patch finalisation ==

static Vec3 orthonormal_tangent(int axis) {
    switch (axis) {
        case 0: case 1: return {0, 1, 0};
        case 2: case 3: return {1, 0, 0};
        case 4: case 5: return {1, 0, 0};
    }
    return {1,0,0};
}

static Vec3 orthonormal_bitangent(int axis) {
    Vec3 n = axis_dirs[axis];
    Vec3 t = orthonormal_tangent(axis);
    return glm::cross(n, t);
}

static std::vector<Patch> finalize_patches(
        const std::vector<Cluster>& clusters,
        const std::vector<Triangle>& tris,
        const std::vector<Vec3>& positions)
{
    std::vector<Patch> result;
    result.reserve(clusters.size());
    for (auto& p : clusters) {
        Patch fp;
        fp.id = p.id;
        fp.axis = p.axis;
        AABB bb;
        fp.basis_u = axis_dirs[p.axis];
        fp.basis_v = orthonormal_tangent(p.axis);
        fp.basis_w = orthonormal_bitangent(p.axis);

        Vec2 min_proj = {1e30f, 1e30f};
        Vec2 max_proj = {-1e30f, -1e30f};
        for (int ti : p.tris) {
            for (int e = 0; e < 3; ++e) {
                Vec3 pos = positions[tris[ti].v[e]];
                grow(bb, pos);
                Vec2 proj = project_along(p.axis, pos);
                min_proj = {std::min(min_proj.x, proj.x), std::min(min_proj.y, proj.y)};
                max_proj = {std::max(max_proj.x, proj.x), std::max(max_proj.y, proj.y)};
            }
        }
        fp.aabb_min = bb.lo; fp.aabb_max = bb.hi;
        fp.proj_min = min_proj;
        fp.proj_size = {max_proj.x - min_proj.x, max_proj.y - min_proj.y};
        fp.tex_w = fp.tex_h = 0;
        fp.atlas_x = fp.atlas_y = 0;
        fp.tris = p.tris;
        result.push_back(std::move(fp));
    }
    return result;
}

// ================================================= texture sizing (§5) ==

static float size_textures(std::vector<Patch>& patches, float density,
                           int min_tex, int max_tex) {
    float total = 0;
    for (auto& p : patches) {
        p.tex_w = std::clamp((int)std::ceil(p.proj_size.x * density), min_tex, max_tex);
        p.tex_h = std::clamp((int)std::ceil(p.proj_size.y * density), min_tex, max_tex);
        total += float(p.tex_w) * p.tex_h;
    }
    return total;
}

// Scales `base_density` (texels per unit) down only as far as needed to fit
// `budget` texels. Returns the effective density actually applied.
static float fit_to_budget(std::vector<Patch>& patches, float base_density,
                           float budget, int min_tex, int max_tex) {
    float total = size_textures(patches, base_density, min_tex, max_tex);
    if (total <= budget) {
        printf("  Texture budget: %.1f M texels used of %.1f M — within budget\n",
               total/1e6f, budget/1e6f);
        return base_density;
    }
    float lo = 0.001f, hi = 1.0f;
    for (int it = 0; it < 32; ++it) {
        float mid = (lo+hi)*0.5f;
        float t = size_textures(patches, base_density*mid, min_tex, max_tex);
        if (t > budget) hi = mid; else lo = mid;
    }
    float final_density = base_density * lo;
    size_textures(patches, final_density, min_tex, max_tex);
    printf("  Texture budget: %.1f M texels used of %.1f M (density scale=%.4f)\n",
           size_textures(patches, final_density, min_tex, max_tex)/1e6f, budget/1e6f, lo);
    return final_density;
}

// =================================================== atlas packing (§6) ==

struct SkylineNode { int x, y, w; };

static void pack_atlas(std::vector<Patch>& patches, int& atlas_w, int& atlas_h) {
    // Sort by max side descending: keeps the layout compact and near-square.
    std::sort(patches.begin(), patches.end(),
              [](auto& a, auto& b) {
                  return std::max(a.tex_w, a.tex_h) > std::max(b.tex_w, b.tex_h);
              });

    atlas_w = 1024; atlas_h = 1024;

    // On failure the whole atlas is re-packed from scratch at a larger size.
    // The previous code cleared the skyline mid-loop but kept the already-placed
    // patches at their old (smaller-atlas) coordinates, which made later patches
    // overlap them and forced the atlas to grow to ~2.7x the real texel budget.
    for (;;) {
        std::vector<SkylineNode> skyline;
        skyline.push_back({0, 0, atlas_w});

        bool ok = true;
        for (auto& p : patches) {
            // Find best position
            int best_i = -1;
            int best_y = atlas_h + 1;
            for (int i = 0; i < (int)skyline.size(); ++i) {
                if (skyline[i].w >= p.tex_w && skyline[i].y + p.tex_h <= atlas_h) {
                    if (skyline[i].y < best_y) {
                        best_y = skyline[i].y;
                        best_i = i;
                    }
                }
            }
            if (best_i < 0) { ok = false; break; }

            int px = skyline[best_i].x;
            int py = skyline[best_i].y;
            p.atlas_x = px;
            p.atlas_y = py;

            // Update skyline
            std::vector<SkylineNode> new_skyline;
            for (auto& s : skyline) {
                int sx = s.x, se = s.x + s.w;
                int ox = std::max(sx, px), oe = std::min(se, px + p.tex_w);
                if (oe <= ox) {
                    new_skyline.push_back(s);
                } else {
                    if (sx < ox) new_skyline.push_back({sx, s.y, ox - sx});
                    new_skyline.push_back({ox, py + p.tex_h, oe - ox});
                    if (oe < se) new_skyline.push_back({oe, s.y, se - oe});
                }
            }
            std::sort(new_skyline.begin(), new_skyline.end(),
                      [](auto& a, auto& b) { return a.x < b.x; });
            skyline.clear();
            for (auto& n : new_skyline) {
                if (!skyline.empty() && skyline.back().y == n.y &&
                    skyline.back().x + skyline.back().w == n.x) {
                    skyline.back().w += n.w;
                } else {
                    skyline.push_back(n);
                }
            }
        }

        if (ok) break;
        atlas_w *= 2;
        atlas_h *= 2;
        if (atlas_w > (1 << 16) || atlas_h > (1 << 16)) {
            fprintf(stderr, "Atlas packing failed\n");
            break;
        }
    }

    printf("  Atlas size: %d × %d (%.1f M texels)\n",
           atlas_w, atlas_h, float(atlas_w)*atlas_h/1e6f);
}

// ============================================================ BVH (§7) ==

struct BVHPrim { AABB aabb; int index; };

// Recursive SAH build — returns (root_index, total_nodes).
static int sah_build(std::vector<BVHPrim>& prims, int begin, int end,
                     std::vector<BVHNode>& nodes)
{
    AABB total_bb;
    for (int i = begin; i < end; ++i) grow_aabb(total_bb, prims[i].aabb);
    int count = end - begin;

    auto make_leaf = [&](int pos) {
        BVHNode n;
        n.aabb_min = prims[pos].aabb.lo;
        n.aabb_max = prims[pos].aabb.hi;
        n.patch_index = uint32_t(prims[pos].index);
        n.is_leaf = 1;
        nodes.push_back(n);
        return int(nodes.size()) - 1;
    };

    if (count == 1) {
        return make_leaf(begin);
    }

    // Bin-based SAH sweep: pick the best (axis, split) in [begin, end).
    float best_cost = 1e30f;
    int best_axis = 0, best_split = begin + count / 2;
    bool have_split = false;

    for (int axis = 0; axis < 3; ++axis) {
        auto cmp = [axis](const BVHPrim& a, const BVHPrim& b) {
            float ca = (axis==0)?a.aabb.lo.x:(axis==1)?a.aabb.lo.y:a.aabb.lo.z;
            float cb = (axis==0)?b.aabb.lo.x:(axis==1)?b.aabb.lo.y:b.aabb.lo.z;
            return ca < cb;
        };
        std::sort(prims.begin()+begin, prims.begin()+end, cmp);

        int nb = SAH_BINS;
        std::vector<AABB> bin_bb(nb, AABB{});
        std::vector<int> bin_count(nb, 0);
        for (int i = begin; i < end; ++i) {
            float c = (axis==0)?prims[i].aabb.lo.x:(axis==1)?prims[i].aabb.lo.y:prims[i].aabb.lo.z;
            float tmin = (axis==0)?total_bb.lo.x:(axis==1)?total_bb.lo.y:total_bb.lo.z;
            float tmax = (axis==0)?total_bb.hi.x:(axis==1)?total_bb.hi.y:total_bb.hi.z;
            int b = (tmax > tmin) ? std::min(nb-1, int((c-tmin)/(tmax-tmin)*nb)) : 0;
            grow(bin_bb[b], prims[i].aabb.lo);
            grow(bin_bb[b], prims[i].aabb.hi);
            bin_count[b]++;
        }

        // Evaluate splits
        AABB left_bb;
        int left_count = 0;
        for (int i = 0; i < nb-1; ++i) {
            if (bin_count[i] == 0) continue;
            grow(left_bb, bin_bb[i].lo);
            grow(left_bb, bin_bb[i].hi);
            left_count += bin_count[i];
            int right_count = count - left_count;
            if (right_count <= 0) break;
            AABB right_bb;
            int rc = 0;
            for (int j = i+1; j < nb; ++j) {
                if (bin_count[j] > 0) {
                    grow(right_bb, bin_bb[j].lo);
                    grow(right_bb, bin_bb[j].hi);
                    rc += bin_count[j];
                }
            }
            float cost = surface_area(left_bb) * left_count +
                         surface_area(right_bb) * right_count;
            if (cost < best_cost) {
                best_cost = cost;
                best_axis = axis;
                best_split = begin + left_count;
                have_split = true;
            }
        }
    }

    if (have_split) {
        // Re-sort by the winning axis (the sweep left prims sorted by the last
        // axis evaluated, which may differ from best_axis).
        auto cmp = [best_axis](const BVHPrim& a, const BVHPrim& b) {
            float ca = (best_axis==0)?a.aabb.lo.x:(best_axis==1)?a.aabb.lo.y:a.aabb.lo.z;
            float cb = (best_axis==0)?b.aabb.lo.x:(best_axis==1)?b.aabb.lo.y:b.aabb.lo.z;
            return ca < cb;
        };
        std::sort(prims.begin()+begin, prims.begin()+end, cmp);
    } else {
        // Every prim is coincident on all axes (degenerate or identical AABBs),
        // so the sweep found no split. Fall back to a median split along the
        // widest axis — this guarantees each prim lands in exactly one leaf
        // instead of being silently dropped.
        Vec3 e = extent(total_bb);
        best_axis = (e.x >= e.y && e.x >= e.z) ? 0 : (e.y >= e.z) ? 1 : 2;
        auto cmp = [best_axis](const BVHPrim& a, const BVHPrim& b) {
            float ca = (best_axis==0)?a.aabb.lo.x:(best_axis==1)?a.aabb.lo.y:a.aabb.lo.z;
            float cb = (best_axis==0)?b.aabb.lo.x:(best_axis==1)?b.aabb.lo.y:b.aabb.lo.z;
            return ca < cb;
        };
        std::sort(prims.begin()+begin, prims.begin()+end, cmp);
        best_split = begin + count / 2;
    }

    int self_idx = int(nodes.size());
    nodes.push_back(BVHNode{}); // placeholder
    int left = sah_build(prims, begin, best_split, nodes);
    int right = sah_build(prims, best_split, end, nodes);

    // Fill in the placeholder
    nodes[self_idx].aabb_min = total_bb.lo;
    nodes[self_idx].aabb_max = total_bb.hi;
    nodes[self_idx].right_offset = uint32_t(right - self_idx);
    nodes[self_idx].is_leaf = 0;

    return self_idx;
}

static std::vector<BVHNode> build_bvh(const std::vector<Patch>& patches) {
    std::vector<BVHPrim> prims(patches.size());
    for (size_t i = 0; i < patches.size(); ++i) {
        prims[i].aabb = AABB{patches[i].aabb_min, patches[i].aabb_max};
        prims[i].index = int(i);
    }
    std::vector<BVHNode> nodes;
    if (!prims.empty()) sah_build(prims, 0, int(prims.size()), nodes);

    // Quantise AABBs to 16-bit
    if (!nodes.empty()) {
        AABB root_bb;
        for (auto& n : nodes) {
            grow(root_bb, n.aabb_min);
            grow(root_bb, n.aabb_max);
        }
        Vec3 rlo = root_bb.lo, rhi = root_bb.hi;
        Vec3 span = rhi - rlo;
        for (auto& n : nodes) {
            for (int c = 0; c < 3; ++c) {
                float t0 = span[c] > 1e-10f ? (n.aabb_min[c]-rlo[c])/span[c] : 0;
                float t1 = span[c] > 1e-10f ? (n.aabb_max[c]-rlo[c])/span[c] : 0;
                n.q_min[c] = uint16_t(std::clamp(t0, 0.0f, 1.0f) * 65535.0f);
                n.q_max[c] = uint16_t(std::clamp(t1, 0.0f, 1.0f) * 65535.0f);
            }
        }
    }

    printf("  BVH: %zu nodes\n", nodes.size());
    return nodes;
}

// ================================================== vertex welding ===

// Weld vertices by position. Triangle soup (e.g. Stanford_Bunny) shares zero
// vertices, which makes index-based adjacency useless. Deduplicate positions so
// edges match again. Normals at weld points are averaged; the first UV wins.
static void weld_vertices(std::vector<Vec3>& positions, std::vector<Vec3>& normals,
                          std::vector<Vec2>& uvs, std::vector<unsigned int>& indices)
{
    struct WeldKey { int64_t x, y, z;
        bool operator==(const WeldKey& o) const { return x==o.x && y==o.y && z==o.z; } };
    struct WeldHash {
        size_t operator()(const WeldKey& k) const {
            size_t h1 = std::hash<int64_t>()(k.x);
            size_t h2 = std::hash<int64_t>()(k.y);
            size_t h3 = std::hash<int64_t>()(k.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    AABB bb;
    for (auto& p : positions) grow(bb, p);
    Vec3 ext = extent(bb);
    float max_span = std::max({ext.x, ext.y, ext.z});
    float weld_eps = std::max(max_span * 1e-7f, 1e-9f);

    std::unordered_map<WeldKey, unsigned int, WeldHash> weld;
    weld.reserve(positions.size());
    std::vector<unsigned int> remap(positions.size());
    std::vector<Vec3> wpos;
    std::vector<Vec3> wnorm;
    std::vector<Vec2> wuv;
    wpos.reserve(positions.size() / 3);
    wnorm.reserve(positions.size() / 3);
    wuv.reserve(positions.size() / 3);

    auto quantize = [&](Vec3 p) -> WeldKey {
        return {std::llround(p.x / weld_eps),
                std::llround(p.y / weld_eps),
                std::llround(p.z / weld_eps)};
    };

    for (size_t v = 0; v < positions.size(); ++v) {
        WeldKey key = quantize(positions[v]);
        auto it = weld.find(key);
        if (it != weld.end()) {
            remap[v] = it->second;
            wnorm[it->second] += normals[v];
        } else {
            unsigned int idx = unsigned(wpos.size());
            weld[key] = idx;
            remap[v] = idx;
            wpos.push_back(positions[v]);
            wnorm.push_back(normals[v]);
            wuv.push_back(uvs[v]);
        }
    }

    for (size_t i = 0; i < indices.size(); ++i)
        indices[i] = remap[indices[i]];
    positions = std::move(wpos);
    normals = std::move(wnorm);
    uvs = std::move(wuv);
}

// =========================================== atlas rasterisation (§8) ==

// Fills empty texels that are fully enclosed by covered texels (interior
// rasterisation cracks) so a fine leaf tile does not produce noisy black
// holes inside the surface. Texels connected to the patch border through
// empty texels are real "outside the silhouette" and are left empty. Filled
// cells copy the average of their covered orthogonal neighbours.
static void fill_atlas_holes(std::vector<float>& atlas_depth,
                             std::vector<float>& atlas_thickness,
                             std::vector<float>& atlas_uv,
                             int atlas_w, int ax, int ay, int tw, int th)
{
    const size_t n = size_t(tw) * th;
    std::vector<uint8_t> st(n, 0);            // 0=covered, 1=empty-border, 2=interior
    auto at = [&](int x, int y) -> bool {     // true if covered
        return atlas_depth[size_t(ay + y) * atlas_w + (ax + x)] > -1e20f;
    };
    for (int y = 0; y < th; ++y)
        for (int x = 0; x < tw; ++x)
            if (!at(x, y)) st[size_t(y) * tw + x] = 1;

    // Flood from the rect border through empty cells (the true outside).
    std::vector<int> q;
    auto push = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= tw || y >= th) return;
        size_t i = size_t(y) * tw + x;
        if (st[i] == 1) { st[i] = 0; q.push_back(int(i)); }  // 0 = reached outside
    };
    for (int x = 0; x < tw; ++x) { push(x, 0); push(x, th - 1); }
    for (int y = 0; y < th; ++y) { push(0, y); push(tw - 1, y); }
    while (!q.empty()) {
        int i = q.back(); q.pop_back();
        int x = i % tw, y = i / tw;
        push(x - 1, y); push(x + 1, y); push(x, y - 1); push(x, y + 1);
    }

    // Every cell still marked empty (1) is an interior hole; fill inward.
    std::vector<int> fill;
    bool changed = true;
    while (changed) {
        changed = false;
        fill.clear();
        for (int y = 0; y < th; ++y) {
            for (int x = 0; x < tw; ++x) {
                size_t i = size_t(y) * tw + x;
                if (st[i] != 1) continue;
                float d = 0, t = 0, u = 0, v = 0; int cnt = 0;
                const int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};
                for (int k = 0; k < 4; ++k) {
                    int nx = x + dx[k], ny = y + dy[k];
                    if (nx < 0 || ny < 0 || nx >= tw || ny >= th) continue;
                    size_t j = size_t(ay + ny) * atlas_w + (ax + nx);
                    if (atlas_depth[j] <= -1e20f) continue;
                    d += atlas_depth[j]; t += atlas_thickness[j];
                    u += atlas_uv[j * 2]; v += atlas_uv[j * 2 + 1]; cnt++;
                }
                if (cnt > 0) { st[i] = 0; fill.push_back(int(i)); }
            }
        }
        if (!fill.empty()) {
            changed = true;
            for (int i : fill) {
                int x = i % tw, y = i / tw;
                float d = 0, t = 0, u = 0, v = 0; int cnt = 0;
                const int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};
                for (int k = 0; k < 4; ++k) {
                    int nx = x + dx[k], ny = y + dy[k];
                    if (nx < 0 || ny < 0 || nx >= tw || ny >= th) continue;
                    size_t j = size_t(ay + ny) * atlas_w + (ax + nx);
                    if (atlas_depth[j] <= -1e20f) continue;
                    d += atlas_depth[j]; t += atlas_thickness[j];
                    u += atlas_uv[j * 2]; v += atlas_uv[j * 2 + 1]; cnt++;
                }
                if (cnt == 0) continue;
                size_t j = size_t(ay + y) * atlas_w + (ax + x);
                atlas_depth[j]     = d / cnt;
                atlas_thickness[j] = t / cnt;
                atlas_uv[j * 2]    = u / cnt;
                atlas_uv[j * 2 + 1] = v / cnt;
            }
        }
    }
}

static void rasterize_atlas_textures(
        const std::vector<Patch>& patches,
        const std::vector<Triangle>& tris,
        const std::vector<Vec3>& positions,
        const std::vector<Vec2>& uvs,
        std::vector<float>& atlas_depth,
        std::vector<float>& atlas_thickness,
        std::vector<float>& atlas_uv,
        int atlas_w, int atlas_h)
{
    atlas_depth.assign(size_t(atlas_w)*atlas_h, -1e30f);
    atlas_thickness.assign(size_t(atlas_w)*atlas_h, 0.0f);
    atlas_uv.assign(size_t(atlas_w)*atlas_h*2, 0.0f);

    for (auto& fp : patches) {
        if (fp.tex_w <= 0 || fp.tex_h <= 0) continue;
        int ax = fp.atlas_x, ay = fp.atlas_y;
        int tw = fp.tex_w, th = fp.tex_h;
        Vec2 pmn = fp.proj_min, pmx = fp.proj_min + fp.proj_size;
        auto to_atlas = [&](Vec2 p) -> Vec2 {
            float u = (pmx.x>pmn.x) ? (p.x-pmn.x)/(pmx.x-pmn.x) : 0.5f;
            float v = (pmx.y>pmn.y) ? (p.y-pmn.y)/(pmx.y-pmn.y) : 0.5f;
            return {float(ax)+u*tw, float(ay)+v*th};
        };

        // Pass A+C: depth + UV rasterisation (patch triangles only)
        for (int ti : fp.tris) {
            Vec2 gv[3]; float gd[3]; Vec2 guv[3];
            for (int e = 0; e < 3; ++e) {
                Vec3 pos = positions[tris[ti].v[e]];
                gv[e] = project_along(fp.axis, pos);
                gd[e] = depth_along(fp.axis, pos);
                guv[e] = (tris[ti].v[e] < uvs.size()) ? uvs[tris[ti].v[e]] : Vec2{0,0};
            }
            Vec2 av[3] = {to_atlas(gv[0]), to_atlas(gv[1]), to_atlas(gv[2])};
            float fx0=av[0].x,fy0=av[0].y,fx1=av[1].x,fy1=av[1].y,fx2=av[2].x,fy2=av[2].y;
            int x0=std::max(ax,(int)std::floor(std::min({fx0,fx1,fx2})));
            int x1=std::min(ax+tw-1,(int)std::ceil(std::max({fx0,fx1,fx2})));
            int y0=std::max(ay,(int)std::floor(std::min({fy0,fy1,fy2})));
            int y1=std::min(ay+th-1,(int)std::ceil(std::max({fy0,fy1,fy2})));
            float area2=(fx1-fx0)*(fy2-fy0)-(fx2-fx0)*(fy1-fy0);
            if(std::abs(area2)<1e-10f) continue;
            float inv=1.0f/area2;
            bool cw = area2 < 0.0f;
            for(int py=y0;py<=y1;++py){
                for(int px=x0;px<=x1;++px){
                    float ppx=float(px)+0.5f,ppy=float(py)+0.5f;
                    float e0=(fx1-fx0)*(ppy-fy0)-(ppx-fx0)*(fy1-fy0);
                    float e1=(fx2-fx1)*(ppy-fy1)-(ppx-fx1)*(fy2-fy1);
                    float e2=(fx0-fx2)*(ppy-fy2)-(ppx-fx2)*(fy0-fy2);
                    bool inside = cw ? (e0<=1e-6f&&e1<=1e-6f&&e2<=1e-6f)
                                     : (e0>=-1e-6f&&e1>=-1e-6f&&e2>=-1e-6f);
                    if(inside){
                        float w2=e0*inv;
                        float w0=e1*inv;
                        float w1=1.0f-w0-w2;
                        size_t idx=size_t(py)*atlas_w+px;
                        float depth=w0*gd[0]+w1*gd[1]+w2*gd[2];
                        if(atlas_depth[idx] <= -1e20f || depth < atlas_depth[idx]) atlas_depth[idx]=depth;
                        atlas_uv[idx*2+0]=w0*guv[0].x+w1*guv[1].x+w2*guv[2].x;
                        atlas_uv[idx*2+1]=w0*guv[0].y+w1*guv[1].y+w2*guv[2].y;
                    }
                }
            }
        }

        // Thickness: for each valid texel, accumulate depth range
        for (int ti : fp.tris) {
            Vec2 gv[3]; float gd[3];
            for (int e = 0; e < 3; ++e) {
                Vec3 pos = positions[tris[ti].v[e]];
                gv[e] = project_along(fp.axis, pos);
                gd[e] = depth_along(fp.axis, pos);
            }
            Vec2 av[3]={to_atlas(gv[0]),to_atlas(gv[1]),to_atlas(gv[2])};
            float fx0=av[0].x,fy0=av[0].y,fx1=av[1].x,fy1=av[1].y,fx2=av[2].x,fy2=av[2].y;
            int x0=std::max(ax,(int)std::floor(std::min({fx0,fx1,fx2})));
            int x1=std::min(ax+tw-1,(int)std::ceil(std::max({fx0,fx1,fx2})));
            int y0=std::max(ay,(int)std::floor(std::min({fy0,fy1,fy2})));
            int y1=std::min(ay+th-1,(int)std::ceil(std::max({fy0,fy1,fy2})));
            float area2=(fx1-fx0)*(fy2-fy0)-(fx2-fx0)*(fy1-fy0);
            if(std::abs(area2)<1e-10f) continue;
            bool cw = area2 < 0.0f;
            for(int py=y0;py<=y1;++py){
                for(int px=x0;px<=x1;++px){
                    float ppx=float(px)+0.5f,ppy=float(py)+0.5f;
                    float e0=(fx1-fx0)*(ppy-fy0)-(ppx-fx0)*(fy1-fy0);
                    float e1=(fx2-fx1)*(ppy-fy1)-(ppx-fx1)*(fy2-fy1);
                    float e2=(fx0-fx2)*(ppy-fy2)-(ppx-fx2)*(fy0-fy2);
                    bool inside = cw ? (e0<=1e-6f&&e1<=1e-6f&&e2<=1e-6f)
                                     : (e0>=-1e-6f&&e1>=-1e-6f&&e2>=-1e-6f);
                    if(inside){
                        size_t idx=size_t(py)*atlas_w+px;
                        if(atlas_depth[idx]>-1e20f){
                            float dmin=std::min({gd[0],gd[1],gd[2]});
                            float dmax=std::max({gd[0],gd[1],gd[2]});
                            atlas_thickness[idx]=std::max(atlas_thickness[idx], dmax-dmin);
                        }
                    }
                }
            }
        }

        // Close interior coverage cracks so a fine leaf tile has no holes.
        fill_atlas_holes(atlas_depth, atlas_thickness, atlas_uv,
                         atlas_w, ax, ay, tw, th);
    }

    if (getenv("MDC_RASTER_DBG")) {
        fprintf(stderr, "[RASTER] g_depth_origin = (%.4f %.4f %.4f)\n",
                g_depth_origin.x, g_depth_origin.y, g_depth_origin.z);
        for (int pid = 0; pid < (int)patches.size() && pid < 13; ++pid) {
            auto& fp = patches[pid];
            // center texel of the patch rect
            int cx = fp.atlas_x + fp.tex_w / 2, cy = fp.atlas_y + fp.tex_h / 2;
            float d = atlas_depth[size_t(cy) * atlas_w + cx];
            float th = atlas_thickness[size_t(cy) * atlas_w + cx];
            float u = atlas_uv[(size_t(cy) * atlas_w + cx) * 2];
            Vec2 mid2 = fp.proj_min + (fp.proj_size * 0.5f);
            fprintf(stderr,
                    "[RASTER] patch %d axis=%d center_tex=(%d,%d) depth=%8.4f "
                    "thick=%7.4f uv=(%.3f,%.3f) aabb_mid=%.2f/%.2f/%.2f\n",
                    pid, fp.axis, cx, cy, d, th, u, 0.0f,
                    mid2.x, mid2.y, 0.0f);
        }
    }
}

// ===================================================== mip chains (§8b) ==

static int next_pow2_int(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

struct MipCell {                      // aggregated value of one quadtree node
    float dmin =  1e30f, dmax = -1e30f;
    float thmax = 0.0f;
    float umin = 1e30f, umax = -1e30f;
    float vmin = 1e30f, vmax = -1e30f;
    float uavg = 0.0f, vavg = 0.0f;
    int   count = 0;
};

struct MipNodeOut {
    MipCell  agg;
    uint32_t child = MIP_LEAF;
};

struct MipTolerances {
    float depth_lo, depth_hi;
    float thick_max;
    float uv_lo, uv_hi;
    float frac = 0.004f;   // subdivide tolerance as a fraction of global range
};

// Per-patch value pyramid over a power-of-two lattice. levels[0] cells are
// `leaf_tile` texels (config.mip_leaf_tile); each higher level downsamples
// 2x2. Because node boundaries in the quadtree are exact powers of two from
// the patch origin, a node of size s reads its aggregate from pyramid level
// log2(s/leaf_tile) in O(1).
struct PatchPyramid {
    int n4 = 0;                                  // finest-level cells per side (N/leaf_tile)
    int leaf_tile = 1;
    std::vector<std::vector<MipCell>> levels;    // levels[0] = size-leaf_tile cells

    MipCell lookup(int x, int y, int s) const {
        int t = s / leaf_tile, k = 0;
        while (t > 1) { t >>= 1; ++k; }
        int w = n4 >> k;
        return levels[k][size_t(y / s) * w + (x / s)];
    }
};

static PatchPyramid build_patch_pyramid(
        const Patch& fp,
        const std::vector<float>& atlas_depth,
        const std::vector<float>& atlas_thickness,
        const std::vector<float>& atlas_uv,
        int atlas_w, int leaf_tile)
{
    int tw = fp.tex_w, th = fp.tex_h;
    int N = std::max(next_pow2_int(std::max(tw, th)), leaf_tile);
    int n4 = N / leaf_tile;
    int nl = 1;
    for (int s = N; s > leaf_tile; s >>= 1) nl++;

    PatchPyramid pyr;
    pyr.n4 = n4;
    pyr.leaf_tile = leaf_tile;
    pyr.levels.assign(nl, {});
    pyr.levels[0].assign(size_t(n4) * n4, MipCell{});
    int ax = fp.atlas_x, ay = fp.atlas_y;
    for (int j = 0; j < n4; ++j) {
        for (int i = 0; i < n4; ++i) {
            MipCell c;
            int x0 = i * leaf_tile, y0 = j * leaf_tile;
            for (int v = y0; v < y0 + leaf_tile && v < th; ++v) {
                for (int u = x0; u < x0 + leaf_tile && u < tw; ++u) {
                    size_t idx = size_t(ay + v) * atlas_w + (ax + u);
                    float d = atlas_depth[idx];
                    if (d <= -1e20f) continue;
                    c.dmin = std::min(c.dmin, d);
                    c.dmax = std::max(c.dmax, d);
                    c.thmax = std::max(c.thmax, atlas_thickness[idx]);
                    float uu = atlas_uv[idx * 2], vv = atlas_uv[idx * 2 + 1];
                    c.umin = std::min(c.umin, uu);
                    c.umax = std::max(c.umax, uu);
                    c.vmin = std::min(c.vmin, vv);
                    c.vmax = std::max(c.vmax, vv);
                    c.uavg += uu;
                    c.vavg += vv;
                    c.count++;
                }
            }
            if (c.count) { c.uavg /= float(c.count); c.vavg /= float(c.count); }
            pyr.levels[0][size_t(j) * n4 + i] = c;
        }
    }

    for (int k = 0; k + 1 < nl; ++k) {
        int w = n4 >> k, wn = w >> 1;
        pyr.levels[k + 1].assign(size_t(wn) * wn, MipCell{});
        for (int j = 0; j < wn; ++j) {
            for (int i = 0; i < wn; ++i) {
                MipCell c;
                for (int dj = 0; dj < 2; ++dj) {
                    for (int di = 0; di < 2; ++di) {
                        const MipCell& s = pyr.levels[k][size_t(j * 2 + dj) * w + (i * 2 + di)];
                        c.count += s.count;
                        c.dmin = std::min(c.dmin, s.dmin);
                        c.dmax = std::max(c.dmax, s.dmax);
                        c.thmax = std::max(c.thmax, s.thmax);
                        c.umin = std::min(c.umin, s.umin);
                        c.umax = std::max(c.umax, s.umax);
                        c.vmin = std::min(c.vmin, s.vmin);
                        c.vmax = std::max(c.vmax, s.vmax);
                        c.uavg += s.uavg * float(s.count);
                        c.vavg += s.vavg * float(s.count);
                    }
                }
                if (c.count) { c.uavg /= float(c.count); c.vavg /= float(c.count); }
                pyr.levels[k + 1][size_t(j) * wn + i] = c;
            }
        }
    }

    if (getenv("MDC_RASTER_DBG")) {
        const MipCell& root = pyr.levels[nl - 1][0];
        fprintf(stderr,
                "[PYR] patch N=%d n4=%d nl=%d root dmin=%8.4f dmax=%8.4f "
                "thmax=%7.4f count=%d\n",
                N, n4, nl, root.dmin, root.dmax, root.thmax, root.count);
        for (int i = 0; i < n4 && i < 4; ++i) {
            const MipCell& c = pyr.levels[0][size_t(n4 / 2) * n4 + i];
            size_t idx = size_t(ay + (n4 / 2) * leaf_tile) * atlas_w + (ax + i * leaf_tile);
            fprintf(stderr, "[PYR]   finest (i=%d) dmin=%8.4f dmax=%8.4f count=%d raw_atlas=%8.4f\n",
                    i, c.dmin, c.dmax, c.count, atlas_depth[idx]);
        }
    }
    return pyr;
}

static bool node_exceeds(const MipCell& c, const MipTolerances& tol) {
    float dtol = tol.frac * std::max(tol.depth_hi - tol.depth_lo, 1e-6f);
    float ttol = tol.frac * std::max(tol.thick_max, 1e-6f);
    float utol = tol.frac * std::max(tol.uv_hi - tol.uv_lo, 1e-6f);
    return (c.dmax - c.dmin) > dtol ||
           c.thmax > ttol ||
           (c.umax - c.umin) > utol ||
           (c.vmax - c.vmin) > utol;
}

// Builds one patch's quadtree directly into the shared per-level node lists.
// Children are appended in quadrant order, so the parent's child offset is
// simply the level's size at the moment the parent is pushed.
static void build_mip_tree_rec(std::vector<std::vector<MipNodeOut>>& levels,
                               const PatchPyramid& pyr,
                               const MipTolerances& tol,
                               int level, int x, int y, int s)
{
    MipNodeOut node;
    node.agg = pyr.lookup(x, y, s);
    bool has = node.agg.count > 0;
    // Subdivide while the region is partially covered (silhouette refinement)
    // or while its value range exceeds the tolerance. Full and empty regions
    // stay as leaves, so coverage edges refine down to `leaf_tile` texels.
    bool partial = has && node.agg.count < s * s;
    bool sub = has && s > pyr.leaf_tile && level + 1 < MIP_MAX_LEVELS &&
               (node_exceeds(node.agg, tol) || partial);
    if (sub) {
        node.child = uint32_t(levels[level + 1].size());
        levels[level].push_back(node);
        int hs = s >> 1;
        build_mip_tree_rec(levels, pyr, tol, level + 1, x,     y,     hs);
        build_mip_tree_rec(levels, pyr, tol, level + 1, x + hs, y,     hs);
        build_mip_tree_rec(levels, pyr, tol, level + 1, x,     y + hs, hs);
        build_mip_tree_rec(levels, pyr, tol, level + 1, x + hs, y + hs, hs);
    } else {
        levels[level].push_back(node);
    }
}

enum MipChanType { MIP_DEPTH, MIP_THICK, MIP_UV };

static uint8_t mip_qbyte(float v, float lo, float hi) {
    // Map the valid range [lo, hi] onto [1, 255] so byte 0 is reserved for
    // "empty" (no covered texels) and a covered value is never black.
    float t = (hi > lo) ? (v - lo) / (hi - lo) : 0.0f;
    int q = int(std::lround(std::clamp(t, 0.0f, 1.0f) * 254.0f)) + 1;
    return uint8_t(std::clamp(q, 1, 255));
}

static std::array<float, 2> mip_values(MipChanType t, const MipCell& c) {
    switch (t) {
        case MIP_DEPTH: return {c.dmin, c.dmax};
        case MIP_THICK: return {c.thmax, 0.0f};
        case MIP_UV:    return {c.uavg, c.vavg};
    }
    return {0, 0};
}

static void emit_mip_chain(MipChanType type,
                           const std::vector<std::vector<MipNodeOut>>& nodes,
                           const float qmin[2], const float qmax[2],
                           int leaf_tile, MipChain& out)
{
    out.channels = (type == MIP_THICK) ? 1 : 2;
    out.leaf_tile = leaf_tile;
    out.qmin[0] = qmin[0]; out.qmax[0] = qmax[0];
    out.qmin[1] = qmin[1]; out.qmax[1] = qmax[1];

    int used = int(nodes.size());
    while (used > 1 && nodes[size_t(used) - 1].empty()) used--;
    out.levels.resize(size_t(used));

    for (int L = 0; L < used; ++L) {
        const auto& nlist = nodes[size_t(L)];
        if (nlist.empty()) continue;
        auto& lev = out.levels[size_t(L)];
        uint32_t n = uint32_t(nlist.size());
        lev.w = std::min<uint32_t>(n, MIP_MAX_TEXW);
        lev.h = (n + lev.w - 1) / lev.w;
        lev.data.assign(size_t(n) * out.channels, 0);
        lev.meta.assign(n, 0);
        for (uint32_t i = 0; i < n; ++i) {
            const MipCell& c = nlist[i].agg;
            if (c.count > 0) {
                auto v = mip_values(type, c);
                uint8_t* p = lev.data.data() + size_t(i) * out.channels;
                p[0] = mip_qbyte(v[0], qmin[0], qmax[0]);
                if (out.channels == 2) p[1] = mip_qbyte(v[1], qmin[1], qmax[1]);
            }

            lev.meta[i] = (c.count > 0 ? 0x80000000u : 0u) |
                          (nlist[i].child != MIP_LEAF ? 0x40000000u : 0u) |
                          (nlist[i].child & 0x3FFFFFFFu);
        }
    }
}

static void build_mip_chains(
        const std::vector<Patch>& patches,
        const std::vector<float>& atlas_depth,
        const std::vector<float>& atlas_thickness,
        const std::vector<float>& atlas_uv,
        int atlas_w, int atlas_h,
        float tol_frac, int leaf_tile,
        MipChain& depth_chain, MipChain& thick_chain, MipChain& uv_chain)
{
    printf("Building adaptive mip chains...\n");

    // Global quantisation ranges over all valid atlas texels.
    float gdmin =  1e30f, gdmax = -1e30f;
    float gthmax = 0.0f;
    float gumin =  1e30f, gumax = -1e30f;
    float gvmin =  1e30f, gvmax = -1e30f;
    size_t n = size_t(atlas_w) * atlas_h;
    for (size_t i = 0; i < n; ++i) {
        float d = atlas_depth[i];
        if (d <= -1e20f) continue;
        gdmin = std::min(gdmin, d); gdmax = std::max(gdmax, d);
        gthmax = std::max(gthmax, atlas_thickness[i]);
        float u = atlas_uv[i * 2], v = atlas_uv[i * 2 + 1];
        gumin = std::min(gumin, u); gumax = std::max(gumax, u);
        gvmin = std::min(gvmin, v); gvmax = std::max(gvmax, v);
    }
    if (gdmax <= gdmin) { gdmin = 0.0f; gdmax = 1.0f; }
    if (gumax <= gumin) { gumin = 0.0f; gumax = 1.0f; }
    if (gvmax <= gvmin) { gvmin = 0.0f; gvmax = 1.0f; }
    if (gthmax <= 0.0f) gthmax = 1.0f;
    printf("  Depth range %.4f..%.4f, thickness max %.4f, UV u %.4f..%.4f v %.4f..%.4f\n",
           gdmin, gdmax, gthmax, gumin, gumax, gvmin, gvmax);

    MipTolerances tol{gdmin, gdmax, gthmax,
                      std::min(gumin, gvmin), std::max(gumax, gvmax),
                      tol_frac};

    // Level-0 node index must match the order of the other outputs (patch
    // table / patch tris / BVH all iterate `patches` in this exact order,
    // which is the pack order after pack_atlas() sorted by size), so process
    // the patches in-place rather than sorting by id.
    std::vector<std::vector<MipNodeOut>> nodes(MIP_MAX_LEVELS);
    for (auto& fp : patches) {
        if (fp.tex_w <= 0 || fp.tex_h <= 0) continue;
        PatchPyramid pyr = build_patch_pyramid(fp, atlas_depth, atlas_thickness,
                                               atlas_uv, atlas_w, leaf_tile);
        build_mip_tree_rec(nodes, pyr, tol, 0, 0, 0, pyr.n4 * leaf_tile);
    }

    // Emit the three chains (they share the same tree topology).
    // qmin[i]/qmax[i] = quantisation range for channel i.
    float dqlo[2] = {gdmin, gdmin};
    float dqhi[2] = {gdmax, gdmax};
    float tqlo[2] = {0.0f, 0.0f};
    float tqhi[2] = {gthmax, gthmax};
    float uvlo[2] = {gumin, gvmin};
    float uvhi[2] = {gumax, gvmax};
    emit_mip_chain(MIP_DEPTH, nodes, dqlo, dqhi, leaf_tile, depth_chain);
    emit_mip_chain(MIP_THICK, nodes, tqlo, tqhi, leaf_tile, thick_chain);
    emit_mip_chain(MIP_UV, nodes, uvlo, uvhi, leaf_tile, uv_chain);

    int deepest = 0;
    for (int L = 0; L < (int)nodes.size(); ++L)
        if (!nodes[size_t(L)].empty()) deepest = L;
    size_t total = 0;
    for (int L = 0; L <= deepest; ++L) {
        printf("  Level %2d: %7zu nodes  (node size %dx%d atlas texels)\n",
               L, nodes[size_t(L)].size(),
               leaf_tile << (deepest - L), leaf_tile << (deepest - L));
        total += nodes[size_t(L)].size();
    }
    printf("  Mip-chain nodes: %zu total\n", total);
}

// ========================================================= file writers ==

static void write_u32(FILE* f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void write_u64(FILE* f, uint64_t v) { fwrite(&v, 8, 1, f); }

static bool write_patch_table(const char* path,
                              const std::vector<Patch>& patches)
{
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return false; }
    write_u32(f, 0x50415443);  // "PATC"
    write_u32(f, 2);           // version (2 adds per-patch double_sided)
    write_u32(f, uint32_t(patches.size()));

    // aabb_min
    write_u32(f, uint32_t(patches.size()*12));
    for (auto& p : patches) fwrite(&p.aabb_min, 12, 1, f);
    // aabb_max
    write_u32(f, uint32_t(patches.size()*12));
    for (auto& p : patches) fwrite(&p.aabb_max, 12, 1, f);
    // axis
    write_u32(f, uint32_t(patches.size()));
    for (auto& p : patches) { uint8_t a = uint8_t(p.axis); fwrite(&a, 1, 1, f); }
    // tex_w, tex_h (uint16)
    write_u32(f, uint32_t(patches.size()*4));
    for (auto& p : patches) { uint16_t w=p.tex_w, h=p.tex_h; fwrite(&w,2,1,f); fwrite(&h,2,1,f); }
    // atlas_rect (x,y,w,h as uint32)
    write_u32(f, uint32_t(patches.size()*16));
    for (auto& p : patches) {
        uint32_t r[4] = {uint32_t(p.atlas_x),uint32_t(p.atlas_y),uint32_t(p.tex_w),uint32_t(p.tex_h)};
        fwrite(r, 4, 4, f);
    }
    // double_sided (u8 per patch)
    write_u32(f, uint32_t(patches.size()));
    for (auto& p : patches) {
        uint8_t d = p.double_sided ? 1 : 0;
        fwrite(&d, 1, 1, f);
    }
    fclose(f);
    printf("  Wrote %s\n", path);
    return true;
}

static bool write_patch_tris(const char* path,
                             const std::vector<gfx::CoverageAtlas::Vec3Tri>& tris,
                             const std::vector<Patch>& patches)
{
    // Build reordered index buffer: patches in order, triangles contiguous
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    write_u32(f, 0x54524953);  // "TRIS"
    uint32_t total = uint32_t(tris.size());
    write_u32(f, total);
    for (auto& p : patches) {
        for (int ti : p.tris) {
            uint32_t idx[3] = {tris[size_t(ti)].v[0], tris[size_t(ti)].v[1], tris[size_t(ti)].v[2]};
            fwrite(idx, 4, 3, f);
        }
    }
    fclose(f);
    printf("  Wrote %s\n", path);
    return true;
}

static bool write_bvh(const char* path, const std::vector<BVHNode>& nodes) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    write_u32(f, 0x42564830);  // "BVH0"
    write_u32(f, uint32_t(nodes.size()));
    fwrite(nodes.data(), sizeof(BVHNode), nodes.size(), f);
    fclose(f);
    printf("  Wrote %s\n", path);
    return true;
}

static uint64_t write_mip_chain(const char* path, const MipChain& ch,
                                int atlas_w, int atlas_h, int num_patches)
{
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    uint32_t nl = uint32_t(ch.levels.size());
    write_u32(f, 0x4D495034u);                    // "MIP4"
    write_u32(f, 1u);                             // version
    write_u32(f, uint32_t(ch.channels));
    write_u32(f, 1u);                             // bytes per channel
    write_u32(f, nl);
    write_u32(f, uint32_t(ch.leaf_tile));
    write_u32(f, uint32_t(atlas_w));
    write_u32(f, uint32_t(atlas_h));
    write_u32(f, uint32_t(num_patches));
    fwrite(ch.qmin, 4, size_t(ch.channels), f);
    fwrite(ch.qmax, 4, size_t(ch.channels), f);

    uint64_t header_bytes = uint64_t(36 + 8 * ch.channels + nl * 40);
    uint64_t off = header_bytes;
    uint64_t total_nodes = 0;
    std::vector<uint64_t> data_off(nl), data_size(nl), meta_off(nl), meta_size(nl);
    for (uint32_t L = 0; L < nl; ++L) {
        uint64_t dn = ch.levels[L].meta.size();
        total_nodes += dn;
        data_off[L] = off;  data_size[L] = dn * uint64_t(ch.channels); off += data_size[L];
        meta_off[L] = off;  meta_size[L] = dn * 4;                     off += meta_size[L];
    }
    for (uint32_t L = 0; L < nl; ++L) {
        write_u32(f, ch.levels[L].w);
        write_u32(f, ch.levels[L].h);
        write_u64(f, data_off[L]);
        write_u64(f, data_size[L]);
        write_u64(f, meta_off[L]);
        write_u64(f, meta_size[L]);
    }
    for (uint32_t L = 0; L < nl; ++L) {
        if (ch.levels[L].meta.empty()) continue;
        fwrite(ch.levels[L].data.data(), 1, ch.levels[L].data.size(), f);
        fwrite(ch.levels[L].meta.data(), 4, ch.levels[L].meta.size(), f);
    }
    fclose(f);
    printf("  Wrote %s (%.2f MB, %u levels, %llu nodes)\n",
           path, double(off) / 1048576.0, nl,
           (unsigned long long)total_nodes);
    return off;
}

static void write_i32(FILE* f, int32_t v) { fwrite(&v, 4, 1, f); }
static void write_f32(FILE* f, float v) { fwrite(&v, 4, 1, f); }

// Full-state snapshot (atlas_state.bin): everything needed to reconstruct the
// atlas on the next run without re-running the pipeline. Versioned; load_files
// refuses to read an unknown version (forcing a rebuild).
static bool write_atlas_state(const char* path,
                              const CoverageAtlasConfig& cfg,
                              const std::vector<glm::vec3>& positions,
                              const std::vector<glm::vec3>& normals,
                              const std::vector<glm::vec2>& uvs,
                              const std::vector<gfx::CoverageAtlas::Vec3Tri>& tris,
                              const std::vector<Patch>& patches,
                              const std::vector<BVHNode>& bvh,
                              int atlas_w, int atlas_h, float density,
                              const MipChain& depth, const MipChain& thick, const MipChain& uv)
{
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return false; }
    write_u32(f, 0x41544C53u);       // "ATLS"
    write_u32(f, 1u);                // version
    write_f32(f, cfg.texel_density); write_i32(f, cfg.auto_target);
    write_f32(f, cfg.budget_texels); write_i32(f, cfg.min_tex); write_i32(f, cfg.max_tex);
    write_f32(f, cfg.mip_tol_frac);  write_i32(f, cfg.mip_leaf_tile);
    write_i32(f, cfg.min_patch_size); write_f32(f, cfg.epsilon);
    write_f32(f, cfg.axis_threshold); write_u32(f, cfg.rotate_model_x ? 1u : 0u);
    write_i32(f, atlas_w); write_i32(f, atlas_h);
    write_f32(f, density);

    auto write_vecs = [&](const auto& vec, size_t stride) {
        write_u32(f, uint32_t(vec.size()));
        for (auto& v : vec) fwrite(&v, 1, stride, f);
    };
    write_vecs(positions, 12);
    write_vecs(normals, 12);
    write_vecs(uvs, 8);
    write_u32(f, uint32_t(tris.size()));
    for (auto& t : tris) fwrite(t.v, 4, 3, f);

    write_u32(f, uint32_t(patches.size()));
    for (auto& p : patches) {
        write_i32(f, p.id); write_i32(f, p.axis);
        fwrite(&p.aabb_min, 12, 1, f); fwrite(&p.aabb_max, 12, 1, f);
        fwrite(&p.basis_u, 12, 1, f); fwrite(&p.basis_v, 12, 1, f); fwrite(&p.basis_w, 12, 1, f);
        fwrite(&p.proj_min, 8, 1, f); fwrite(&p.proj_size, 8, 1, f);
        write_i32(f, p.tex_w); write_i32(f, p.tex_h);
        write_i32(f, p.atlas_x); write_i32(f, p.atlas_y);
        uint8_t ds = p.double_sided ? 1 : 0;
        fwrite(&ds, 1, 1, f);
        write_u32(f, uint32_t(p.tris.size()));
        for (int ti : p.tris) write_i32(f, ti);
    }

    write_u32(f, uint32_t(bvh.size()));
    for (auto& b : bvh) {
        fwrite(&b.aabb_min, 12, 1, f); fwrite(&b.aabb_max, 12, 1, f);
        fwrite(b.q_min, 2, 3, f); fwrite(b.q_max, 2, 3, f);
        write_u32(f, b.is_leaf ? b.patch_index : b.right_offset);
        write_u32(f, b.is_leaf);
    }

    auto write_chain = [&](const MipChain& ch) {
        write_i32(f, ch.channels); write_i32(f, ch.leaf_tile);
        fwrite(ch.qmin, 4, 2, f); fwrite(ch.qmax, 4, 2, f);
        write_u32(f, uint32_t(ch.levels.size()));
        for (auto& lv : ch.levels) {
            write_u32(f, lv.w); write_u32(f, lv.h);
            write_u64(f, lv.data.size());
            write_u64(f, lv.meta.size());
            if (!lv.data.empty()) fwrite(lv.data.data(), 1, lv.data.size(), f);
            if (!lv.meta.empty()) fwrite(lv.meta.data(), 4, lv.meta.size(), f);
        }
    };
    write_chain(depth); write_chain(thick); write_chain(uv);
    fclose(f);
    printf("  Wrote %s\n", path);
    return true;
}

static bool read_u32(FILE* f, uint32_t& v) { return fread(&v, 4, 1, f) == 1; }
static bool read_i32(FILE* f, int32_t& v) { return fread(&v, 4, 1, f) == 1; }
static bool read_f32(FILE* f, float& v) { return fread(&v, 4, 1, f) == 1; }
static bool read_u64(FILE* f, uint64_t& v) { return fread(&v, 8, 1, f) == 1; }
static bool read_blob(FILE* f, void* dst, size_t n) { return fread(dst, 1, n, f) == n; }

static bool read_atlas_state(const char* path,
                             CoverageAtlasConfig& cfg,
                             std::vector<glm::vec3>& positions,
                             std::vector<glm::vec3>& normals,
                             std::vector<glm::vec2>& uvs,
                             std::vector<gfx::CoverageAtlas::Vec3Tri>& tris,
                             std::vector<Patch>& patches,
                             std::vector<BVHNode>& bvh,
                             int& atlas_w, int& atlas_h, float& density,
                             MipChain& depth, MipChain& thick, MipChain& uv)
{
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    uint32_t magic = 0, version = 0;
    if (!read_u32(f, magic) || magic != 0x41544C53u) { fclose(f); return false; }
    if (!read_u32(f, version) || version != 1u) { fclose(f); return false; }

    auto read_cfg = [&]() -> bool {
        return read_f32(f, cfg.texel_density) && read_i32(f, cfg.auto_target) &&
               read_f32(f, cfg.budget_texels) && read_i32(f, cfg.min_tex) &&
               read_i32(f, cfg.max_tex) && read_f32(f, cfg.mip_tol_frac) &&
               read_i32(f, cfg.mip_leaf_tile) && read_i32(f, cfg.min_patch_size) &&
               read_f32(f, cfg.epsilon) && read_f32(f, cfg.axis_threshold);
    };
    uint32_t rot = 0;
    if (!read_cfg() || !read_u32(f, rot)) { fclose(f); return false; }
    cfg.rotate_model_x = rot != 0;
    if (!read_i32(f, atlas_w) || !read_i32(f, atlas_h) || !read_f32(f, density)) {
        fclose(f); return false;
    }

    auto read_vecs = [&](auto& vec, size_t stride) -> bool {
        uint32_t n = 0;
        if (!read_u32(f, n)) return false;
        vec.resize(n);
        for (auto& v : vec)
            if (!read_blob(f, &v, stride)) return false;
        return true;
    };
    if (!read_vecs(positions, 12) || !read_vecs(normals, 12) || !read_vecs(uvs, 8)) {
        fclose(f); return false;
    }
    {
        uint32_t n = 0;
        if (!read_u32(f, n)) { fclose(f); return false; }
        tris.resize(n);
        for (auto& t : tris)
            if (!read_blob(f, t.v, 12)) { fclose(f); return false; }
    }
    {
        uint32_t n = 0;
        if (!read_u32(f, n)) { fclose(f); return false; }
        patches.resize(n);
        for (auto& p : patches) {
            if (!read_i32(f, p.id) || !read_i32(f, p.axis) ||
                !read_blob(f, &p.aabb_min, 12) || !read_blob(f, &p.aabb_max, 12) ||
                !read_blob(f, &p.basis_u, 12) || !read_blob(f, &p.basis_v, 12) ||
                !read_blob(f, &p.basis_w, 12) || !read_blob(f, &p.proj_min, 8) ||
                !read_blob(f, &p.proj_size, 8) ||
                !read_i32(f, p.tex_w) || !read_i32(f, p.tex_h) ||
                !read_i32(f, p.atlas_x) || !read_i32(f, p.atlas_y)) {
                fclose(f); return false;
            }
            uint8_t ds = 0;
            if (!read_blob(f, &ds, 1)) { fclose(f); return false; }
            p.double_sided = ds != 0;
            uint32_t ntr = 0;
            if (!read_u32(f, ntr)) { fclose(f); return false; }
            p.tris.resize(ntr);
            for (auto& ti : p.tris)
                if (!read_i32(f, ti)) { fclose(f); return false; }
        }
    }
    {
        uint32_t n = 0;
        if (!read_u32(f, n)) { fclose(f); return false; }
        bvh.resize(n);
        for (auto& b : bvh) {
            if (!read_blob(f, &b.aabb_min, 12) || !read_blob(f, &b.aabb_max, 12) ||
                !read_blob(f, b.q_min, 6) || !read_blob(f, b.q_max, 6)) {
                fclose(f); return false;
            }
            uint32_t val = 0, leaf = 0;
            if (!read_u32(f, val) || !read_u32(f, leaf)) { fclose(f); return false; }
            b.is_leaf = leaf;
            if (leaf) b.patch_index = val; else b.right_offset = val;
        }
    }
    auto read_chain = [&](MipChain& ch) -> bool {
        if (!read_i32(f, ch.channels) || !read_i32(f, ch.leaf_tile) ||
            !read_blob(f, ch.qmin, 8) || !read_blob(f, ch.qmax, 8)) return false;
        uint32_t nl = 0;
        if (!read_u32(f, nl)) return false;
        ch.levels.resize(nl);
        for (auto& lv : ch.levels) {
            uint64_t ds = 0, ms = 0;
            if (!read_u32(f, lv.w) || !read_u32(f, lv.h) ||
                !read_u64(f, ds) || !read_u64(f, ms)) return false;
            lv.data.resize(size_t(ds));
            lv.meta.resize(size_t(ms));
            if (!lv.data.empty() && !read_blob(f, lv.data.data(), lv.data.size())) return false;
            if (!lv.meta.empty() && !read_blob(f, lv.meta.data(), lv.meta.size() * 4)) return false;
        }
        return true;
    };
    if (!read_chain(depth) || !read_chain(thick) || !read_chain(uv)) { fclose(f); return false; }

    fclose(f);
    printf("  Read %s\n", path);
    return true;
}

} // namespace

// ========================================================================
//  gfx::CoverageAtlas
// ========================================================================

namespace gfx {

CoverageAtlas::CoverageAtlas(const CoverageAtlasConfig& config) : config_(config) {}

bool CoverageAtlas::build(const gfx::Model& model) {
    auto t0 = std::chrono::high_resolution_clock::now();
    g_depth_origin = {0.0f, 0.0f, 0.0f};

    printf("=== Mesh → Occlusion-Free Patch Atlas Pipeline ===\n");
    if (config_.texel_density > 0.0f)
        printf("Texel density: %.0f texels/unit\n", config_.texel_density);
    else
        printf("Texel density: auto (~%d texels across longest axis)\n",
               config_.auto_target);
    printf("Budget:        %.1f M texels\n", config_.budget_texels/1e6f);
    printf("Patch tex:     %d .. %d texels, mip tol %.4f, leaf tile %d\n",
           config_.min_tex, config_.max_tex, config_.mip_tol_frac,
           config_.mip_leaf_tile);
    printf("Epsilon:       %.6f\n", config_.epsilon);
    printf("\n");

    // Use only LOD0 meshes. Meshes that belong to a LOD group contribute
    // their finest level (mesh_indices[0]); any mesh outside a group is used
    // as-is. Without this, duplicate LODs get concatenated into one surface.
    std::vector<int> use_meshes;
    {
        std::vector<bool> in_lod_group(model.mesh_count(), false);
        for (size_t g = 0; g < model.lod_group_count(); ++g)
            for (int mi : model.lod_group(g).mesh_indices)
                in_lod_group[size_t(mi)] = true;
        for (size_t m = 0; m < model.mesh_count(); ++m)
            if (!in_lod_group[m]) use_meshes.push_back(int(m));
        for (size_t g = 0; g < model.lod_group_count(); ++g)
            use_meshes.push_back(model.lod_group(g).mesh_indices[0]);
        printf("Using %zu mesh(es) (LOD0 only, %zu lod groups)\n",
               use_meshes.size(), model.lod_group_count());
    }

    // Process each LOD0 mesh as an independent submesh. Interiors (e.g. Sponza)
    // store separate objects as distinct glTF meshes. Welding and the
    // decomposition must run per submesh so patches never span independent
    // objects; the shared atlas still collects every submesh's patches.
    positions_.clear();
    normals_.clear();
    uvs_.clear();
    triangles_.clear();
    patches_.clear();
    bvh_nodes_.clear();
    depth_chain_ = MipChain{};
    thickness_chain_ = MipChain{};
    uv_chain_ = MipChain{};

    std::vector<Vec3> positions, normals;
    std::vector<Vec2> uvs;
    std::vector<Triangle> tris;
    std::vector<Patch> final_patches;

    for (int m : use_meshes) {
        const auto& mesh = model.mesh(size_t(m));
        printf("\n-- Submesh %d/%zu: %zu triangles --\n",
               m, use_meshes.size(), mesh.index_count() / 3);

        // Double-sidedness is a property of the source submesh's material
        // (glTF material.doubleSided). Patches never span submeshes, so every
        // patch built from this submesh inherits its flag.
        bool sub_double_sided = false;
        {
            int mi = model.mesh_material(size_t(m));
            if (mi >= 0) sub_double_sided = model.material_info(size_t(mi)).double_sided;
            printf("  double_sided: %s (material %d)\n",
                   sub_double_sided ? "true" : "false", mi);
        }

        std::vector<Vec3> lpos, lnorm;
        std::vector<Vec2> luv;
        std::vector<unsigned int> lindices;
        lpos.reserve(mesh.vertex_count());
        lnorm.reserve(mesh.vertex_count());
        luv.reserve(mesh.vertex_count());
        lindices.reserve(mesh.index_count());
        for (size_t v = 0; v < mesh.vertex_count(); ++v) {
            const gfx::Vertex& vert = mesh.vertices()[v];
            lpos.push_back({vert.position[0], vert.position[1], vert.position[2]});
            lnorm.push_back({vert.normal[0], vert.normal[1], vert.normal[2]});
            luv.push_back({vert.texcoord[0], vert.texcoord[1]});
        }
        for (size_t i = 0; i < mesh.index_count(); ++i)
            lindices.push_back(mesh.indices()[i]);

        // Weld within the submesh only (never across independent objects).
        weld_vertices(lpos, lnorm, luv, lindices);

        // Rotate 90° about X so the scene stands upright:
        // R_x(90°) * (x,y,z) = (x, -z, y)
        if (config_.rotate_model_x) {
            for (auto& p : lpos) { float y = p.y, z = p.z; p.y = -z; p.z = y; }
            for (auto& n : lnorm) { float y = n.y, z = n.z; n.y = -z; n.z = y; }
        }

        // Vertex normals may be absent — compute face normals if so.
        bool has_normals = false;
        for (auto& n : lnorm)
            if (glm::dot(n, n) > 1e-10f) { has_normals = true; break; }
        if (!has_normals) {
            printf("  No vertex normals — computing face normals\n");
            for (size_t i = 0; i < lindices.size() / 3; ++i) {
                Vec3 p0 = lpos[lindices[i*3+0]];
                Vec3 p1 = lpos[lindices[i*3+1]];
                Vec3 p2 = lpos[lindices[i*3+2]];
                Vec3 fn = safe_normalize(glm::cross(p1-p0, p2-p0));
                lnorm[lindices[i*3+0]] += fn;
                lnorm[lindices[i*3+1]] += fn;
                lnorm[lindices[i*3+2]] += fn;
            }
            for (auto& n : lnorm) n = safe_normalize(n);
        }

        // Local triangle data (indices into lpos).
        size_t ltris_count = lindices.size() / 3;
        std::vector<Triangle> ltris(ltris_count);
        for (size_t i = 0; i < ltris_count; ++i) {
            ltris[i].v[0] = lindices[i*3+0];
            ltris[i].v[1] = lindices[i*3+1];
            ltris[i].v[2] = lindices[i*3+2];
            Vec3 p0 = lpos[ltris[i].v[0]];
            Vec3 p1 = lpos[ltris[i].v[1]];
            Vec3 p2 = lpos[ltris[i].v[2]];
            ltris[i].centroid = (p0+p1+p2) * (1.0f/3.0f);
            ltris[i].normal = safe_normalize(glm::cross(p1-p0, p2-p0));
            compute_axis_affinity(ltris[i]);
            ltris[i].patch_id = -1;
        }

        // --- §3  Greedy clustering ---
        std::vector<std::vector<int>> adj;
        std::vector<int> boundary_flags;
        build_adjacency(ltris, adj, boundary_flags);
        std::vector<Cluster> clusters;
        build_patches(ltris, lpos, adj, clusters, config_.texel_density, config_.epsilon);

        // --- §3.5  Merge pass (threshold scales with this submesh's size) ---
        if (config_.min_patch_size > 0) {
            int mps = std::max(1, std::min(config_.min_patch_size, int(ltris.size()) / 200));
            merge_pass(clusters, ltris, lpos, config_.epsilon, mps, config_.axis_threshold);
        }

        // --- §4  Patch finalisation (local indices) ---
        auto sub_final = finalize_patches(clusters, ltris, lpos);

        // --- Accumulate into the shared buffers/atlas ---
        unsigned int base_pos = unsigned(positions.size());
        int base_tri = int(tris.size());
        int id_base = int(final_patches.size());
        for (auto& p : lpos) positions.push_back(p);
        for (auto& n : lnorm) normals.push_back(n);
        for (auto& u : luv) uvs.push_back(u);
        for (auto& t : ltris) {
            t.v[0] += base_pos; t.v[1] += base_pos; t.v[2] += base_pos;
            if (t.patch_id >= 0) t.patch_id += id_base;
            tris.push_back(t);
        }
        for (auto& fp : sub_final) {
            fp.id += id_base;
            fp.double_sided = sub_double_sided;
            for (auto& ti : fp.tris) ti += base_tri;
            final_patches.push_back(std::move(fp));
        }
    }

    size_t tri_count = tris.size();
    printf("\nVertices: %zu, Triangles: %zu across %zu submeshes\n",
           positions.size(), tri_count, use_meshes.size());

    AABB mesh_bb;
    for (auto& p : positions) grow(mesh_bb, p);
    Vec3 mesh_ext = extent(mesh_bb);
    printf("Mesh bbox: [%.4f %.4f %.4f] - [%.4f %.4f %.4f]  extent %.4f x %.4f x %.4f\n",
           mesh_bb.lo.x, mesh_bb.lo.y, mesh_bb.lo.z,
           mesh_bb.hi.x, mesh_bb.hi.y, mesh_bb.hi.z,
           mesh_ext.x, mesh_ext.y, mesh_ext.z);

    // Depth is stored relative to the model's AABB (not world space): zero the
    // origin now, before rasterisation, so every patch shares the same basis.
    g_depth_origin = mesh_bb.lo;

    // --- §5  Texture sizing ---
    printf("Sizing textures...\n");
    float density = config_.texel_density;
    if (density <= 0.0f) {
        // Auto: fit `config_.auto_target` texels across the model's longest
        // axis, so a huge building and a tiny low-detail prop both land on a
        // comparable absolute resolution without any manual tuning.
        float span = std::max({mesh_ext.x, mesh_ext.y, mesh_ext.z});
        if (span < 1e-6f) span = 1.0f;
        density = float(config_.auto_target) / span;
        printf("  Auto density: %.1f texels/unit (span %.3f -> ~%d texels)\n",
               density, span, config_.auto_target);
    }
    final_density_ = fit_to_budget(final_patches, density, config_.budget_texels,
                                   config_.min_tex, config_.max_tex);

    // --- §6  Atlas packing ---
    printf("Packing atlas...\n");
    pack_atlas(final_patches, atlas_w_, atlas_h_);

    // Compute atlas dimensions
    atlas_w_ = 0; atlas_h_ = 0;
    for (auto& p : final_patches) {
        atlas_w_ = std::max(atlas_w_, p.atlas_x + p.tex_w);
        atlas_h_ = std::max(atlas_h_, p.atlas_y + p.tex_h);
    }
    printf("  Final atlas: %d × %d\n", atlas_w_, atlas_h_);

    // --- §7  BVH build ---
    printf("Building BVH...\n");
    bvh_nodes_ = build_bvh(final_patches);

    // --- §8  Atlas texture rasterisation ---
    printf("Rasterising atlas textures...\n");
    std::vector<float> atlas_depth, atlas_thickness, atlas_uv;
    rasterize_atlas_textures(final_patches, tris, positions, uvs,
                              atlas_depth, atlas_thickness, atlas_uv,
                              atlas_w_, atlas_h_);

    // --- Statistics ---
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1-t0).count();
    printf("\n=== Results ===\n");
    printf("Time:      %.3f s\n", elapsed);
    printf("Patches:   %zu\n", final_patches.size());

    size_t assigned = 0;
    size_t min_tri = SIZE_MAX, max_tri = 0;
    for (auto& p : final_patches) {
        assigned += p.tris.size();
        min_tri = std::min(min_tri, p.tris.size());
        max_tri = std::max(max_tri, p.tris.size());
    }
    printf("Assigned:  %zu / %zu triangles\n", assigned, tri_count);
    if (final_patches.size() > 0)
        printf("Per patch: min=%zu, max=%zu, avg=%.0f\n",
               min_tri, max_tri, double(assigned)/final_patches.size());

    int axis_counts[6] = {};
    for (auto& p : final_patches) axis_counts[p.axis]++;
    printf("By axis:   ");
    for (int a = 0; a < 6; ++a)
        if (axis_counts[a] > 0) printf("%s=%d ", axis_names[a], axis_counts[a]);
    printf("\n");

    // --- Mip chains ---
    printf("\nBuilding mip chains...\n");
    build_mip_chains(final_patches, atlas_depth, atlas_thickness, atlas_uv,
                     atlas_w_, atlas_h_, config_.mip_tol_frac,
                     config_.mip_leaf_tile,
                     depth_chain_, thickness_chain_, uv_chain_);

    // Publish results.
    positions_ = std::move(positions);
    normals_   = std::move(normals);
    uvs_       = std::move(uvs);
    triangles_.resize(tris.size());
    for (size_t i = 0; i < tris.size(); ++i)
        triangles_[i] = {tris[i].v[0], tris[i].v[1], tris[i].v[2]};
    patches_   = std::move(final_patches);

    printf("\nPipeline complete in %.3f s.\n", elapsed);
    return tri_count > 0;
}

bool CoverageAtlas::write_files(const std::string& dir) const {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        fprintf(stderr, "Cannot create output dir %s\n", dir.c_str());
        return false;
    }
    auto join = [&](const char* name) {
        return (std::filesystem::path(dir) / name).string();
    };

    printf("\nWriting outputs...\n");
    bool ok = true;
    ok &= write_patch_table(join("patch_table.bin").c_str(), patches_);
    ok &= write_patch_tris(join("patch_tris.bin").c_str(), triangles_, patches_);
    ok &= write_bvh(join("bvh_nodes.bin").c_str(), bvh_nodes_);

    write_mip_chain(join("atlas_depth.bin").c_str(), depth_chain_,
                    atlas_w_, atlas_h_, int(patches_.size()));
    write_mip_chain(join("atlas_thickness.bin").c_str(), thickness_chain_,
                    atlas_w_, atlas_h_, int(patches_.size()));
    write_mip_chain(join("atlas_uv.bin").c_str(), uv_chain_,
                    atlas_w_, atlas_h_, int(patches_.size()));

    ok &= write_atlas_state(join("atlas_state.bin").c_str(), config_,
                            positions_, normals_, uvs_, triangles_, patches_,
                            bvh_nodes_, atlas_w_, atlas_h_, final_density_,
                            depth_chain_, thickness_chain_, uv_chain_);

    // --- Patch summary ---
    {
        std::string path = join("patch_summary.txt");
        FILE* f = fopen(path.c_str(), "w");
        if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); return false; }
        size_t tri_count = triangles_.size();
        size_t assigned = 0;
        for (auto& p : patches_) assigned += p.tris.size();
        fprintf(f, "patches %zu\n", patches_.size());
        fprintf(f, "triangles %zu\n", tri_count);
        fprintf(f, "assigned %zu\n", assigned);
        fprintf(f, "atlas %d %d\n", atlas_w_, atlas_h_);
        fprintf(f, "density %.2f\n", final_density_);
        fprintf(f, "bvh_nodes %zu\n", bvh_nodes_.size());
        {
            size_t mip_nodes = 0;
            for (auto& lv : depth_chain_.levels) mip_nodes += lv.meta.size();
            fprintf(f, "mip_levels %zu\n", depth_chain_.levels.size());
            fprintf(f, "mip_nodes %zu\n", mip_nodes);
            for (size_t L = 0; L < depth_chain_.levels.size(); ++L)
                fprintf(f, "mip_level %zu nodes %zu tex %ux%u\n", L,
                        depth_chain_.levels[L].meta.size(),
                        depth_chain_.levels[L].w, depth_chain_.levels[L].h);
        }
        for (auto& p : patches_) {
            fprintf(f, "patch %d axis %s tris %zu tex %dx%d atlas (%d,%d) double_sided %d\n",
                    p.id, axis_names[p.axis],
                    patches_[size_t(p.id)].tris.size(),
                    p.tex_w, p.tex_h, p.atlas_x, p.atlas_y,
                    p.double_sided ? 1 : 0);
        }
        fclose(f);
        printf("  Wrote %s\n", path.c_str());
    }
    return ok;
}

bool CoverageAtlas::load_files(const std::string& dir) {
    auto path = (std::filesystem::path(dir) / "atlas_state.bin").string();

    CoverageAtlasConfig cfg;
    std::vector<glm::vec3> positions, normals;
    std::vector<glm::vec2> uvs;
    std::vector<Vec3Tri> tris;
    std::vector<Patch> patches;
    std::vector<BVHNode> bvh;
    int aw = 0, ah = 0;
    float density = 0.0f;
    MipChain depth, thick, uv;
    if (!read_atlas_state(path.c_str(), cfg, positions, normals, uvs, tris, patches,
                          bvh, aw, ah, density, depth, thick, uv))
        return false;

    config_ = cfg;
    positions_ = std::move(positions);
    normals_   = std::move(normals);
    uvs_       = std::move(uvs);
    triangles_ = std::move(tris);
    patches_   = std::move(patches);
    bvh_nodes_ = std::move(bvh);
    atlas_w_ = aw; atlas_h_ = ah;
    final_density_ = density;
    depth_chain_ = std::move(depth);
    thickness_chain_ = std::move(thick);
    uv_chain_ = std::move(uv);

    printf("Loaded cached atlas from %s: %zu patches, %zu triangles, atlas %dx%d @ %.0f texels/unit\n",
           path.c_str(), patches_.size(), triangles_.size(), atlas_w_, atlas_h_, final_density_);
    return true;
}

} // namespace gfx

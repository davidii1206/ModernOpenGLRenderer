// Example 27 — MDC (Mesh Decomposition Clustering)
// GPU-driven: normals/centers via compute, adjacency via sort+sweep,
// union-find + compaction entirely in SSBOs.  Single-scalar readback only.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <gllib/log.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <gfx/imgui_overlay.hpp>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <bit>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#ifndef GL_TEXTURE_MIN_REDUCTION_MODE_EXT
#define GL_TEXTURE_MIN_REDUCTION_MODE_EXT 0x9366
#endif

// ---------------------------------------------------------------------------
// Per-triangle GPU data layout (must match GLSL TriData)
// ---------------------------------------------------------------------------

struct GPUTri {
    float nx, ny, nz, _pad0;
    float cx, cy, cz, _pad1;
    int32_t face_mask;
    int32_t patch_id;
    int32_t _pad2[2];
};
static_assert(sizeof(GPUTri) == 48, "GPUTri size");

struct Scalars {
    uint32_t changed = 0;
    uint32_t edge_count = 0;
    uint32_t patch_count = 0;
    uint32_t flips = 0;
};

struct PatchMetaGPU {
    int32_t dominant_axis; // 0:+X,1:-X,2:+Y,3:-Y,4:+Z,5:-Z
    int32_t atlas_x, atlas_y;
    int32_t atlas_w, atlas_h;
    float proj_origin_x, proj_origin_y;
    float height_min, height_max; // normalization range for depth
    int32_t _pad[3];
};
static_assert(sizeof(PatchMetaGPU) == 48, "PatchMetaGPU size");

// ======================= COMPUTE SHADER SOURCES ===========================

// --- normals / centers / initial face mask ---
static const char* cs_normals_src = R"(
#version 460 core
layout(local_size_x = 256) in;

struct TriData {
    vec3 normal;
    float _pad0;
    vec3 center;
    float _pad1;
    int  face_mask;
    int  patch_id;
    int  _pad2[2];
};

layout(std430, binding = 0) readonly buffer VertBuf { float verts[]; };
layout(std430, binding = 1) buffer   TriBuf  { TriData tris[]; };
layout(std430, binding = 2) readonly buffer IdxBuf  { uint    idx[]; };

uniform uint u_tri_count;

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_tri_count) return;

    uint i0 = idx[t * 3];
    uint i1 = idx[t * 3 + 1];
    uint i2 = idx[t * 3 + 2];

    vec3 p0 = vec3(verts[i0 * 12], verts[i0 * 12 + 1], verts[i0 * 12 + 2]);
    vec3 p1 = vec3(verts[i1 * 12], verts[i1 * 12 + 1], verts[i1 * 12 + 2]);
    vec3 p2 = vec3(verts[i2 * 12], verts[i2 * 12 + 1], verts[i2 * 12 + 2]);

    vec3 n = normalize(cross(p1 - p0, p2 - p0));
    vec3 c = (p0 + p1 + p2) / 3.0;

    float d[6] = float[](n.x, -n.x, n.y, -n.y, n.z, -n.z);
    int best = 0;
    for (int j = 1; j < 6; ++j)
        if (d[j] > d[best]) best = j;
    int mask = (d[best] > 0.0) ? (1 << best) : 0;

    tris[t].normal    = n;
    tris[t]._pad0     = 0.0;
    tris[t].center    = c;
    tris[t]._pad1     = 0.0;
    tris[t].face_mask = mask;
    tris[t].patch_id  = -1;
    tris[t]._pad2[0]  = 0;
    tris[t]._pad2[1]  = 0;
}
)";

// --- emit edges (one thread per triangle, writes 3 packed keys) ---
static const char* cs_emit_edges_src = R"(
#version 460 core
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 256) in;

layout(std430, binding = 2) readonly buffer IdxBuf { uint idx[]; };
layout(std430, binding = 3) buffer   EdgeBuf { uint64_t keys[]; };

uniform uint u_tri_count;

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_tri_count) return;

    uint i0 = idx[t * 3];
    uint i1 = idx[t * 3 + 1];
    uint i2 = idx[t * 3 + 2];
    uint v[3] = {i0, i1, i2};

    for (int e = 0; e < 3; ++e) {
        uint a = v[e];
        uint b = v[(e + 1) % 3];
        if (a > b) { uint tmp = a; a = b; b = tmp; }
        // Pack: a(22 bits) << 42 | b(22 bits) << 20 | tri_id(20 bits)
        uint64_t key = (uint64_t(a) << 42) | (uint64_t(b) << 20) | uint64_t(t);
        keys[t * 3 + e] = key;
    }
}
)";

// --- bitonic sort (uint64 keys, tri_id packed in low 20 bits) ---
static const char* cs_sort_src = R"(
#version 460 core
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 256) in;

layout(std430, binding = 3) buffer EdgeBuf { uint64_t keys[]; };

uniform uint u_j;
uniform uint u_k;
uniform uint u_N;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= u_N) return;
    uint ixj = i ^ u_j;
    if (ixj > i) {
        bool asc = (i & u_k) == 0;
        if ((keys[i] > keys[ixj]) == asc) {
            uint64_t t = keys[i]; keys[i] = keys[ixj]; keys[ixj] = t;
        }
    }
}
)";

// --- bitonic sort with base offset (for per-bin Morton-code sorting) ---
static const char* cs_sort_flat_src = R"(
#version 460 core
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 256) in;

layout(std430, binding = 3) buffer FlatBuf { uint64_t keys[]; };

uniform uint u_N;
uniform uint u_j;
uniform uint u_k;
uniform uint u_base;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= u_N) return;
    uint gi = i + u_base;
    uint gj = (i ^ u_j) + u_base;
    if (gj > gi) {
        bool asc = (i & u_k) == 0;
        if ((keys[gi] > keys[gj]) == asc) {
            uint64_t t = keys[gi]; keys[gi] = keys[gj]; keys[gj] = t;
        }
    }
}
)";

// --- sweep sorted keys for duplicates → compacted adjacency pairs ---
static const char* cs_sweep_src = R"(
#version 460 core
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 256) in;

layout(std430, binding = 3) readonly buffer EdgeBuf { uint64_t keys[]; };
layout(std430, binding = 4) buffer   PairBuf  { uvec2  pairs[]; };
layout(std430, binding = 5) buffer   CtrBuf {
    uint changed;
    uint edge_count;
    uint patch_count;
    uint flips;
};

uniform uint u_padded;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i + 1 >= u_padded) return;
    if (keys[i] == uint64_t(-1)) return;

    // Compare vertices (high 44 bits); ignore tri_id (low 20 bits)
    if ((keys[i] >> 20) == (keys[i + 1] >> 20)) {
        uint tri_a = uint(keys[i]      & 0xFFFFFu);
        uint tri_b = uint(keys[i + 1]  & 0xFFFFFu);
        uint slot = atomicAdd(edge_count, 1u);
        pairs[slot] = uvec2(tri_a, tri_b);
    }
}
)";

// --- union-find init ---
static const char* cs_uf_init_src = R"(
#version 460 core
layout(local_size_x = 256) in;

struct TriData {
    vec3 normal;
    float _pad0;
    vec3 center;
    float _pad1;
    int  face_mask;
    int  patch_id;
    int  _pad2[2];
};

layout(std430, binding = 1) readonly buffer TriBuf  { TriData tris[]; };
layout(std430, binding = 6) buffer   ParentBuf { int parent[]; };

uniform uint u_tri_count;

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_tri_count) return;
    parent[t] = (tris[t].face_mask != 0) ? int(t) : -1;
}
)";

// --- union-find hook (multiple iterations converge) ---
// Always link larger root index → smaller root index to prevent cycles.
static const char* cs_uf_hook_src = R"(
#version 460 core
layout(local_size_x = 256) in;

struct TriData {
    vec3 normal;
    float _pad0;
    vec3 center;
    float _pad1;
    int  face_mask;
    int  patch_id;
    int  _pad2[2];
};

layout(std430, binding = 1) readonly buffer TriBuf   { TriData tris[]; };
layout(std430, binding = 4) readonly buffer PairBuf  { uvec2  pairs[]; };
layout(std430, binding = 6) buffer   ParentBuf { int    parent[]; };

uniform uint  u_pair_max;
uniform float u_normal_threshold;

void main() {
    uint e = gl_GlobalInvocationID.x;
    if (e >= u_pair_max) return;
    uvec2 p = pairs[e];
    if (p.x == p.y) return;

    int a = int(p.x);
    int b = int(p.y);
    if (tris[a].face_mask != tris[b].face_mask) return;
    if (u_normal_threshold > 0.0f && dot(tris[a].normal, tris[b].normal) < u_normal_threshold) return;

    int ra = a;
    while (parent[ra] != ra && parent[ra] >= 0) { parent[ra] = parent[parent[ra]]; ra = parent[ra]; }
    int rb = b;
    while (parent[rb] != rb && parent[rb] >= 0) { parent[rb] = parent[parent[rb]]; rb = parent[rb]; }
    if (ra < 0 || rb < 0 || ra == rb) return;

    // Link higher index root → lower index root → never a cycle
    if (ra < rb)  { int expected = rb; atomicCompSwap(parent[rb], expected, ra); }
    else          { int expected = ra; atomicCompSwap(parent[ra], expected, rb); }
}
)";

// --- union-find compress ---
static const char* cs_uf_compress_src = R"(
#version 460 core
layout(local_size_x = 256) in;

struct TriData {
    vec3 normal;
    float _pad0;
    vec3 center;
    float _pad1;
    int  face_mask;
    int  patch_id;
    int  _pad2[2];
};

layout(std430, binding = 1) buffer   TriBuf   { TriData tris[]; };
layout(std430, binding = 6) buffer   ParentBuf { int parent[]; };
layout(std430, binding = 9) buffer   SizeBuf   { uint patch_size[]; };

uniform uint u_tri_count;

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_tri_count) return;

    if (tris[t].face_mask == 0) {
        tris[t].patch_id = -1;
        return;
    }

    int r = int(t);
    while (parent[r] != r && parent[r] >= 0) { r = parent[r]; }

    int cur = int(t);
    while (parent[cur] != cur && parent[cur] >= 0) {
        int nxt = parent[cur];
        parent[cur] = r;
        cur = nxt;
    }

    tris[t].patch_id = r;
    atomicAdd(patch_size[r], 1u);
}
)";

// --- patch compaction pass A: roots allocate compacted IDs ---
static const char* cs_compact_a_src = R"(
#version 460 core
layout(local_size_x = 256) in;

struct TriData {
    vec3 normal;
    float _pad0;
    vec3 center;
    float _pad1;
    int  face_mask;
    int  patch_id;
    int  _pad2[2];
};

layout(std430, binding = 1) buffer TriBuf   { TriData tris[]; };
layout(std430, binding = 5) buffer CtrBuf {
    uint changed;
    uint edge_count;
    uint patch_count;
    uint flips;
};
layout(std430, binding = 6) buffer RootBuf { int root_id[]; };

uniform uint u_tri_count;

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_tri_count) return;
    if (tris[t].face_mask == 0 || tris[t].patch_id != int(t)) return;
    root_id[t] = int(atomicAdd(patch_count, 1u));
}
)";

// --- patch compaction pass B: scatter compacted IDs ---
static const char* cs_compact_b_src = R"(
#version 460 core
layout(local_size_x = 256) in;

struct TriData {
    vec3 normal;
    float _pad0;
    vec3 center;
    float _pad1;
    int  face_mask;
    int  patch_id;
    int  _pad2[2];
};

layout(std430, binding = 1) buffer TriBuf { TriData tris[]; };
layout(std430, binding = 6) readonly buffer RootBuf { int root_id[]; };

uniform uint u_tri_count;

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_tri_count) return;
    if (tris[t].face_mask == 0) return;

    int r = tris[t].patch_id;
    if (r >= 0) tris[t].patch_id = root_id[r];
}
)";

// --- AdjCSR degree pass ---
static const char* cs_adj_degree_src = R"(
#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 4) readonly buffer PairBuf { uvec2 pairs[]; };
layout(std430, binding = 7) buffer   DegBuf  { uint  degree[]; };

uniform uint u_edge_count;

void main() {
    uint e = gl_GlobalInvocationID.x;
    if (e >= u_edge_count) return;
    atomicAdd(degree[pairs[e].x], 1u);
    atomicAdd(degree[pairs[e].y], 1u);
}
)";

// --- AdjCSR scatter neighbors ---
static const char* cs_adj_scatter_src = R"(
#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 4) readonly buffer PairBuf { uvec2 pairs[]; };
layout(std430, binding = 7) buffer   CurBuf { uint cursor[]; };
layout(std430, binding = 8) buffer   NbrBuf { uint neighbors[]; };

uniform uint u_edge_count;

void main() {
    uint e = gl_GlobalInvocationID.x;
    if (e >= u_edge_count) return;
    uint a = pairs[e].x;
    uint b = pairs[e].y;
    neighbors[atomicAdd(cursor[a], 1u)] = b;
    neighbors[atomicAdd(cursor[b], 1u)] = a;
}
)";

// --- patch size histogram ---
static const char* cs_patch_size_src = R"(
#version 460 core
layout(local_size_x = 256) in;

struct TriData {
    vec3 normal;
    float _pad0;
    vec3 center;
    float _pad1;
    int  face_mask;
    int  patch_id;
    int  _pad2[2];
};

layout(std430, binding = 1) readonly buffer TriBuf  { TriData tris[]; };
layout(std430, binding = 9) buffer   SizeBuf { uint patch_size[]; };

uniform uint u_tri_count;

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_tri_count) return;
    int p = tris[t].patch_id;
    if (p >= 0) atomicAdd(patch_size[p], 1u);
}
)";

// --- GPU greedy relabel ---
static const char* cs_relabel_src = R"(
#version 460 core
layout(local_size_x = 256) in;

struct TriData {
    vec3 normal;
    float _pad0;
    vec3 center;
    float _pad1;
    int  face_mask;
    int  patch_id;
    int  _pad2[2];
};

layout(std430, binding = 1) buffer   TriBuf  { TriData tris[]; };
layout(std430, binding = 7) readonly buffer OffBuf  { uint csr_off[]; };
layout(std430, binding = 8) readonly buffer NbrBuf  { uint csr_nb[]; };
layout(std430, binding = 9) readonly buffer SizeBuf { uint patch_size[]; };
layout(std430, binding = 5) buffer   CtrBuf  { uint changed; uint edge_count; uint patch_count; uint flips; };

const vec3 AXES[6] = vec3[](
    vec3(1,0,0), vec3(-1,0,0),
    vec3(0,1,0), vec3(0,-1,0),
    vec3(0,0,1), vec3(0,0,-1)
);

uniform float u_cos_epsilon;
uniform uint  u_tri_count;

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_tri_count) return;
    if (tris[t].face_mask == 0 || tris[t].patch_id < 0) return;

    vec3 n = tris[t].normal;
    float best = -1.0, second = -1.0;
    int best_a = -1, second_a = -1;
    for (int a = 0; a < 6; ++a) {
        float d = dot(n, AXES[a]);
        if (d > best) { second = best; second_a = best_a; best = d; best_a = a; }
        else if (d > second) { second = d; second_a = a; }
    }
    if (best - second >= u_cos_epsilon || second_a < 0) return;

    int second_mask = 1 << second_a;
    int cur_patch = tris[t].patch_id;
    int cur_size = int(patch_size[cur_patch]);

    uint start = csr_off[t];
    uint end = csr_off[t + 1];
    for (uint j = start; j < end; ++j) {
        uint nb = csr_nb[j];
        if (tris[nb].face_mask == second_mask && tris[nb].patch_id >= 0) {
            if (int(patch_size[tris[nb].patch_id]) > cur_size) {
                tris[t].face_mask = second_mask;
                atomicAdd(flips, 1u);
                break;
            }
        }
    }
}
)";

// --- GPU orphan adoption ---
static const char* cs_orphan_src = R"(
#version 460 core
layout(local_size_x = 256) in;

struct TriData {
    vec3 normal;
    float _pad0;
    vec3 center;
    float _pad1;
    int  face_mask;
    int  patch_id;
    int  _pad2[2];
};

layout(std430, binding = 1) buffer   TriBuf  { TriData tris[]; };
layout(std430, binding = 7) readonly buffer OffBuf  { uint csr_off[]; };
layout(std430, binding = 8) readonly buffer NbrBuf  { uint csr_nb[]; };
layout(std430, binding = 9) readonly buffer SizeBuf { uint patch_size[]; };
layout(std430, binding = 5) buffer   CtrBuf  { uint changed; uint edge_count; uint patch_count; uint flips; };

uniform uint u_tri_count;

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_tri_count) return;
    if (tris[t].face_mask == 0 || tris[t].patch_id < 0) return;

    int cur_patch = tris[t].patch_id;
    int cur_size = int(patch_size[cur_patch]);
    const int MIN_PATCH = 3;
    if (cur_size > MIN_PATCH) return;

    vec3 n = tris[t].normal;
    int best_nb_mask = 0;
    int best_nb_size = 0;

    uint start = csr_off[t];
    uint end = csr_off[t + 1];
    for (uint j = start; j < end; ++j) {
        uint nb = csr_nb[j];
        if (tris[nb].patch_id < 0) continue;
        int nb_size = int(patch_size[tris[nb].patch_id]);
        if (nb_size <= MIN_PATCH) continue;
        int nb_mask = tris[nb].face_mask;
        float d = 0.0;
        if      (nb_mask == 1)  d = n.x;
        else if (nb_mask == 2)  d = -n.x;
        else if (nb_mask == 4)  d = n.y;
        else if (nb_mask == 8)  d = -n.y;
        else if (nb_mask == 16) d = n.z;
        else if (nb_mask == 32) d = -n.z;
        if (d <= 0.0) continue;
        if (nb_size > best_nb_size) { best_nb_size = nb_size; best_nb_mask = nb_mask; }
    }

    if (best_nb_mask != 0 && best_nb_mask != tris[t].face_mask) {
        tris[t].face_mask = best_nb_mask;
        atomicAdd(flips, 1u);
    }
}
)";

// --- per-patch AABB + height range (GPU) ---
static const char* cs_patch_aabb_src = R"(
#version 460 core
layout(local_size_x = 256) in;

struct TriData {
    vec3 normal;
    float _pad0;
    vec3 center;
    float _pad1;
    int  face_mask;
    int  patch_id;
    int  _pad2[2];
};

layout(std430, binding = 0) readonly buffer VertBuf { float verts[]; };
layout(std430, binding = 1) readonly buffer TriBuf  { TriData tris[]; };
layout(std430, binding = 2) readonly buffer IdxBuf  { uint idx[]; };
layout(std430, binding = 11) buffer   AABBBuf { uint aabb[]; };

uniform uint u_tri_count;
uniform int  u_patch_count;

uint sortable_uint(float f) {
    uint u = floatBitsToUint(f);
    // IEEE 754 total-order sortable: flip all bits for negatives, set sign=1 for positives
    return (u & 0x80000000u) != 0 ? ~u : (u | 0x80000000u);
}

int dominant_axis_from_mask(int mask) {
    if (mask == 1)  return 0;
    if (mask == 2)  return 1;
    if (mask == 4)  return 2;
    if (mask == 8)  return 3;
    if (mask == 16) return 4;
    if (mask == 32) return 5;
    return -1;
}

float height_val(vec3 p, int axis) {
    if (axis == 0) return  p.x;
    if (axis == 1) return -p.x;
    if (axis == 2) return  p.y;
    if (axis == 3) return -p.y;
    if (axis == 4) return  p.z;
    return -p.z;
}

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_tri_count) return;

    int pid = tris[t].patch_id;
    if (pid < 0 || pid >= u_patch_count) return;

    uint base = uint(pid) * 9u;
    int axis = dominant_axis_from_mask(tris[t].face_mask);

    // Store dominant_axis mask via atomicOr of face_mask bits
    atomicOr(aabb[base + 8u], uint(tris[t].face_mask));

    uint i0 = idx[t * 3u];
    uint i1 = idx[t * 3u + 1u];
    uint i2 = idx[t * 3u + 2u];

    vec3 p0 = vec3(verts[i0 * 12u], verts[i0 * 12u + 1u], verts[i0 * 12u + 2u]);
    vec3 p1 = vec3(verts[i1 * 12u], verts[i1 * 12u + 1u], verts[i1 * 12u + 2u]);
    vec3 p2 = vec3(verts[i2 * 12u], verts[i2 * 12u + 1u], verts[i2 * 12u + 2u]);

    vec3 p[3] = vec3[](p0, p1, p2);

    for (int v = 0; v < 3; ++v) {
        uint sx = sortable_uint(p[v].x);
        uint sy = sortable_uint(p[v].y);
        uint sz = sortable_uint(p[v].z);

        atomicMin(aabb[base + 0u], sx);
        atomicMin(aabb[base + 1u], sy);
        atomicMin(aabb[base + 2u], sz);
        atomicMax(aabb[base + 3u], sx);
        atomicMax(aabb[base + 4u], sy);
        atomicMax(aabb[base + 5u], sz);

        if (axis >= 0) {
            float h = height_val(p[v], axis);
            uint sh = sortable_uint(h);
            atomicMin(aabb[base + 6u], sh);
            atomicMax(aabb[base + 7u], sh);
        }
    }
}
)";

// --- BVH: count patches per dominant-axis bin ---
static const char* cs_bvh_count_src = R"(
#version 460 core
layout(local_size_x = 256) in;

struct PatchMeta {
    int  dominant_axis;
    int  atlas_x, atlas_y;
    int  atlas_w, atlas_h;
    float proj_origin_x, proj_origin_y;
    float height_min, height_max;
    int  _pad[3];
};

layout(std430, binding = 10) readonly buffer MetaBuf { PatchMeta patches[]; };
layout(std430, binding = 12) buffer   CtrBuf  { uint ctr[18]; };

uniform int u_patch_count;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_patch_count)) return;
    int a = patches[i].dominant_axis;
    if (a >= 0) atomicAdd(ctr[a], 1u);
}
)";

// --- BVH: compute Morton codes + scatter patches into per-bin arrays ---
static const char* cs_bvh_scatter_src = R"(
#version 460 core
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 256) in;

struct PatchMeta {
    int  dominant_axis;
    int  atlas_x, atlas_y;
    int  atlas_w, atlas_h;
    float proj_origin_x, proj_origin_y;
    float height_min, height_max;
    int  _pad[3];
};

layout(std430, binding = 10) readonly buffer MetaBuf { PatchMeta patches[]; };
layout(std430, binding = 12) buffer   CtrBuf  { uint ctr[18]; };
layout(std430, binding = 13) readonly buffer AABBBuf { float aabb_float[]; };
layout(std430, binding = 14) buffer   FlatBuf { uint64_t data[]; };

uniform int  u_patch_count;
uniform vec3 u_mesh_min;
uniform vec3 u_mesh_inv; // 1.0 / (mesh_max - mesh_min)

uint spread(uint v) {
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v <<  8)) & 0x0300F00Fu;
    v = (v | (v <<  4)) & 0x030C30C3u;
    v = (v | (v <<  2)) & 0x09249249u;
    return v;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_patch_count)) return;
    int a = patches[i].dominant_axis;
    if (a < 0) return;

    float mx = aabb_float[i * 6u + 0] + aabb_float[i * 6u + 3];
    float my = aabb_float[i * 6u + 1] + aabb_float[i * 6u + 4];
    float mz = aabb_float[i * 6u + 2] + aabb_float[i * 6u + 5];
    vec3 c = vec3(mx * 0.5, my * 0.5, mz * 0.5);

    uint ux = uint(clamp((c.x - u_mesh_min.x) * u_mesh_inv.x * 1024.0, 0.0, 1023.0));
    uint uy = uint(clamp((c.y - u_mesh_min.y) * u_mesh_inv.y * 1024.0, 0.0, 1023.0));
    uint uz = uint(clamp((c.z - u_mesh_min.z) * u_mesh_inv.z * 1024.0, 0.0, 1023.0));

    uint morton = spread(ux) | (spread(uy) << 1) | (spread(uz) << 2);

    uint pos = atomicAdd(ctr[12 + a], 1u);
    data[ctr[6 + a] + pos] = (uint64_t(morton) << 32) | uint64_t(i);
}
)";

// --- BVH: level-by-level 8-way hierarchy construction ---
static const char* cs_bvh_build_src = R"(
#version 460 core
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 256) in;

struct BVHNode {
    float min_x, min_y, min_z;
    int   child_base;
    float max_x, max_y, max_z;
    int   meta;
    int   l, r;
    int   pad0, pad1;
};

layout(std430, binding = 12) readonly buffer CtrBuf { uint ctr[18]; };
layout(std430, binding = 14) readonly buffer FlatBuf { uint64_t data[]; };
layout(std430, binding = 15) buffer NodeBuf { BVHNode nodes[]; };
layout(std430, binding = 17) readonly buffer WorkBuf  { uint wc; int wi[]; };
layout(std430, binding = 18) buffer       NextBuf  { uint nc; int ni[]; };
layout(std430, binding = 19) buffer       CntBuf   { uint ncnt; };

uniform int u_bin;

#define LEAF_FLAG 0x80000000u

int find_first_group(int lo, int hi, int shift, int min_g) {
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        uint m = uint(data[ctr[6 + u_bin] + uint(mid)] >> 32);
        int g = int((m >> shift) & 0x7u);
        if (g >= min_g) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= wc) return;

    int node_idx = wi[idx * 3];
    int l = wi[idx * 3 + 1];
    int r = wi[idx * 3 + 2];

    uint bo = ctr[6 + u_bin];
    uint ml = uint(data[bo + uint(l)] >> 32);
    uint mr = uint(data[bo + uint(r)] >> 32);

    uint diff = ml ^ mr;
    int common_bits = 30;
    if (diff != 0u) common_bits = 29 - findMSB(diff);

    int shift = 29 - common_bits - 2;
    if (shift < 0) {
        uint64_t d_first = data[bo + uint(l)];
        int pid = int(uint(d_first & 0xFFFFFFFFu));
        nodes[node_idx].child_base = pid;
        nodes[node_idx].meta = int(LEAF_FLAG | uint(r - l + 1));
        return;
    }

    int active_firsts[8];
    int active_lasts[8];
    int num_children = 0;

    for (int g = 0; g < 8; g++) {
        int first = find_first_group(l, r + 1, shift, g);
        if (first > r) continue;
        uint mfirst = uint(data[bo + uint(first)] >> 32);
        if (int((mfirst >> shift) & 0x7u) != g) continue;

        int next_first = find_first_group(l, r + 1, shift, g + 1);
        int last = next_first - 1;

        active_firsts[num_children] = first;
        active_lasts[num_children] = last;
        num_children++;
    }

    if (num_children > 0) {
        int child_base = int(atomicAdd(ncnt, uint(num_children)));

        for (int i = 0; i < num_children; i++) {
            int child = child_base + i;
            int first = active_firsts[i];
            int last = active_lasts[i];

            nodes[child].l = first;
            nodes[child].r = last;

            if (first == last) {
                uint64_t d = data[bo + uint(first)];
                int pid = int(uint(d & 0xFFFFFFFFu));
                nodes[child].child_base = pid;
                nodes[child].meta = int(LEAF_FLAG | 1u);
            } else {
                uint wpos = atomicAdd(nc, 1u);
                ni[wpos * 3] = child;
                ni[wpos * 3 + 1] = first;
                ni[wpos * 3 + 2] = last;
            }
        }
        nodes[node_idx].child_base = child_base;
        nodes[node_idx].meta = num_children;
    } else {
        nodes[node_idx].child_base = -1;
        nodes[node_idx].meta = 0;
    }
}
)";

// --- Bottom-up AABB computation for the BVH ---
static const char* cs_bvh_aabb_src = R"(
#version 460 core
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 256) in;

layout(std430, binding = 11) readonly buffer AABBSort { uint sa[]; };

struct BVHNode {
    float min_x, min_y, min_z;
    int   child_base;
    float max_x, max_y, max_z;
    int   meta;
    int   l, r;
    int   pad0, pad1;
};

layout(std430, binding = 14) readonly buffer FlatBuf { uint64_t data[]; };
layout(std430, binding = 15) buffer NodeBuf { BVHNode nodes[]; };

layout(std430, binding = 19) buffer CntBuf { uint ncnt; };

uniform int u_node_count;
uniform int u_root_idx;
uniform int u_bin;

#define LEAF_FLAG 0x80000000u

float from_sortable(uint s) {
    if ((s & 0x80000000u) != 0)
        return uintBitsToFloat(s ^ 0x80000000u);
    else
        return uintBitsToFloat(~s);
}

uniform int u_base_offset;

void main() {
    // Process high-to-low so children (larger indices) are computed before parents
    int n = u_root_idx + u_node_count - 1 - int(gl_GlobalInvocationID.x);
    if (n < u_root_idx) return;

    bool leaf = (nodes[n].meta & int(LEAF_FLAG)) != 0;
    if (leaf) {
        float mnx = 1e30, mny = 1e30, mnz = 1e30;
        float mxx = -1e30, mxy = -1e30, mxz = -1e30;
        for (int i = nodes[n].l; i <= nodes[n].r; i++) {
            uint64_t entry = data[u_base_offset + i];
            int pid = int(uint(entry & 0xFFFFFFFFu));
            int base = pid * 9;
            float x0 = from_sortable(sa[base]);
            float y0 = from_sortable(sa[base + 1]);
            float z0 = from_sortable(sa[base + 2]);
            float x1 = from_sortable(sa[base + 3]);
            float y1 = from_sortable(sa[base + 4]);
            float z1 = from_sortable(sa[base + 5]);
            if (i == nodes[n].l) {
                mnx = x0; mny = y0; mnz = z0;
                mxx = x1; mxy = y1; mxz = z1;
            } else {
                mnx = min(mnx, x0); mny = min(mny, y0); mnz = min(mnz, z0);
                mxx = max(mxx, x1); mxy = max(mxy, y1); mxz = max(mxz, z1);
            }
        }
        nodes[n].min_x = mnx;
        nodes[n].min_y = mny;
        nodes[n].min_z = mnz;
        nodes[n].max_x = mxx;
        nodes[n].max_y = mxy;
        nodes[n].max_z = mxz;
    } else {
        int cc = nodes[n].meta;
        int cb = nodes[n].child_base;
        float mnx = 1e30, mny = 1e30, mnz = 1e30;
        float mxx = -1e30, mxy = -1e30, mxz = -1e30;
        for (int i = 0; i < cc; i++) {
            int c = cb + i;
            mnx = min(mnx, nodes[c].min_x);
            mny = min(mny, nodes[c].min_y);
            mnz = min(mnz, nodes[c].min_z);
            mxx = max(mxx, nodes[c].max_x);
            mxy = max(mxy, nodes[c].max_y);
            mxz = max(mxz, nodes[c].max_z);
        }
        nodes[n].min_x = mnx;
        nodes[n].min_y = mny;
        nodes[n].min_z = mnz;
        nodes[n].max_x = mxx;
        nodes[n].max_y = mxy;
        nodes[n].max_z = mxz;
    }
}
)";

// --- GPU ray-height field pick: BVH traversal + depth culling ---
static const char* cs_pick_src = R"(
#version 460 core
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct PatchMeta {
    int  dominant_axis;
    int  atlas_x, atlas_y;
    int  atlas_w, atlas_h;
    float proj_origin_x, proj_origin_y;
    float height_min, height_max;
    int  _pad[3];
};

struct BVHNode {
    float min_x, min_y, min_z;
    int   child_base;
    float max_x, max_y, max_z;
    int   meta;
    int   l, r;
    int   pad0, pad1;
};

layout(std430, binding = 10) readonly buffer MetaBuf { PatchMeta patches[]; };
layout(std430, binding = 13) readonly buffer AABBBuf { float aabb_float[]; };
layout(std430, binding = 14) readonly buffer FlatBuf { uint64_t data[]; };
layout(std430, binding = 15) readonly buffer NodeBuf { BVHNode nodes[]; };
layout(std430, binding = 16) readonly buffer RootBuf { int bvh_roots[6]; };
uniform int u_bin_offsets[6];

uniform sampler2D u_height_sampler;
uniform sampler2D u_depth_tex;
uniform vec3 u_ro;
uniform vec3 u_rd;
uniform int  u_patch_count;
uniform float u_texel_density;
uniform mat4 u_mvp;
uniform ivec2 u_view_size;

layout(std430, binding = 20) buffer OutBuf {
    int  hit_patch;
    float hit_t;
    float hit_px, hit_py, hit_pz;
};

#define LEAF_FLAG 0x80000000

float height_val(vec3 p, int axis) {
    if (axis == 0) return  p.x;
    if (axis == 1) return -p.x;
    if (axis == 2) return  p.y;
    if (axis == 3) return -p.y;
    if (axis == 4) return  p.z;
    return -p.z;
}

vec2 project(vec3 p, int axis) {
    if (axis == 0 || axis == 1) return vec2(p.y, p.z);
    if (axis == 2 || axis == 3) return vec2(p.x, p.z);
    return vec2(p.x, p.y);
}

// Bilinear height sample with on-the-fly gradient (thickness) computation.
// The gradient is computed analytically from the 4 corner heights, naturally
// capturing both intra-triangle slope and inter-triangle height discontinuities.
vec2 sample_height_bilinear(vec2 tx_ty, PatchMeta pm, inout bool valid) {
    valid = false;
    float tx = tx_ty.x - 0.5;
    float ty = tx_ty.y - 0.5;
    int ix0 = int(floor(tx)), iy0 = int(floor(ty));
    int ix1 = ix0 + 1, iy1 = iy0 + 1;
    float fx = tx - float(ix0), fy = ty - float(iy0);

    bool c00 = false, c10 = false, c01 = false, c11 = false;
    float h00 = 0.0, h10 = 0.0, h01 = 0.0, h11 = 0.0;
    int ax = pm.atlas_x, ay = pm.atlas_y, aw = pm.atlas_w, ah = pm.atlas_h;

    if (ix0 >= ax && ix0 < ax + aw && iy0 >= ay && iy0 < ay + ah) {
        vec2 hc = texelFetch(u_height_sampler, ivec2(ix0, iy0), 0).rg;
        if (hc.g > 1e-6) { h00 = hc.r; c00 = true; }
    }
    if (ix1 >= ax && ix1 < ax + aw && iy0 >= ay && iy0 < ay + ah) {
        vec2 hc = texelFetch(u_height_sampler, ivec2(ix1, iy0), 0).rg;
        if (hc.g > 1e-6) { h10 = hc.r; c10 = true; }
    }
    if (ix0 >= ax && ix0 < ax + aw && iy1 >= ay && iy1 < ay + ah) {
        vec2 hc = texelFetch(u_height_sampler, ivec2(ix0, iy1), 0).rg;
        if (hc.g > 1e-6) { h01 = hc.r; c01 = true; }
    }
    if (ix1 >= ax && ix1 < ax + aw && iy1 >= ay && iy1 < ay + ah) {
        vec2 hc = texelFetch(u_height_sampler, ivec2(ix1, iy1), 0).rg;
        if (hc.g > 1e-6) { h11 = hc.r; c11 = true; }
    }

    int nc = int(c00) + int(c10) + int(c01) + int(c11);
    if (nc < 1) return vec2(0.0, 0.0);
    if (nc == 1) {
        valid = true;
        float h = c00 ? h00 : (c01 ? h01 : (c10 ? h10 : h11));
        return vec2(h, 0.0);
    }
    valid = true;

    float sum_h = 0.0, sum_w = 0.0;
    if (c00) { float w = (1.0-fx)*(1.0-fy); sum_h += h00 * w; sum_w += w; }
    if (c10) { float w = fx*(1.0-fy); sum_h += h10 * w; sum_w += w; }
    if (c01) { float w = (1.0-fx)*fy; sum_h += h01 * w; sum_w += w; }
    if (c11) { float w = fx*fy; sum_h += h11 * w; sum_w += w; }
    float h = sum_h / sum_w;

    float dh_dx = 0.0; int nx = 0;
    float dh_dy = 0.0; int ny = 0;
    if (c00 && c10) { dh_dx += (1.0-fy) * (h10 - h00); nx++; }
    if (c01 && c11) { dh_dx += fy * (h11 - h01); nx++; }
    if (c00 && c01) { dh_dy += (1.0-fx) * (h01 - h00); ny++; }
    if (c10 && c11) { dh_dy += fx * (h11 - h10); ny++; }
    if (nx == 0 && ny == 0) {
        if (c00 && c11) { float d = h11 - h00; dh_dx = d; dh_dy = d; }
        else if (c10 && c01) { float d = h01 - h10; dh_dx = d; dh_dy = d; }
    }

    float grad = max(length(vec2(dh_dx, dh_dy)), 1e-6);
    return vec2(h, grad);
}

vec3 dominant_axis_normal(int a) {
    if (a == 0) return vec3( 1.0,  0.0,  0.0);
    if (a == 1) return vec3(-1.0,  0.0,  0.0);
    if (a == 2) return vec3( 0.0,  1.0,  0.0);
    if (a == 3) return vec3( 0.0, -1.0,  0.0);
    if (a == 4) return vec3( 0.0,  0.0,  1.0);
    return vec3(0.0, 0.0, -1.0);
}

void march_patch(int pid, vec3 ro, vec3 rd, float pt0, float pt1,
                 ivec2 asz, float td,
                 inout float best_t, inout int best_pid, inout vec3 best_pos) {
    PatchMeta pm = patches[pid];
    int axis = pm.dominant_axis;
    if (dot(rd, dominant_axis_normal(axis)) >= 0.0) return;
    float hmin = pm.height_min, hmax = pm.height_max;
    float hr = max(hmax - hmin, 1e-10);
    float ox = pm.proj_origin_x, oy = pm.proj_origin_y;

    vec2 rd_uv = project(rd, axis);
    float uv_speed = max(length(rd_uv), 1e-4);
    float base_step = 0.5 / td;
    float step = min(base_step, (1.0 / td) / uv_speed);
    step = max(step, 1e-8);
    float h_step = step * abs(height_val(rd, axis));
    float min_thick_w = h_step * 2.0;

    // Jitter the start to break up banding
    float jitter = float((pid * 37 + 13) & 63) / 64.0;
    float t = pt0 + jitter * step;
    float tmax = min(pt1, best_t);
    bool entered = false;

    while (t <= tmax) {
        vec3 p = ro + rd * t;
        vec2 pr = project(p, axis);
        float tx = (pr.x - ox) * td + 0.5 + float(pm.atlas_x);
        float ty = (pr.y - oy) * td + 0.5 + float(pm.atlas_y);

        if (tx < float(pm.atlas_x) + 0.5 || tx >= float(pm.atlas_x + pm.atlas_w) - 0.5 ||
            ty < float(pm.atlas_y) + 0.5 || ty >= float(pm.atlas_y + pm.atlas_h) - 0.5) {
            t += step;
            continue;
        }

        bool samp_valid;
        vec2 ht = sample_height_bilinear(vec2(tx, ty), pm, samp_valid);
        if (samp_valid) {
            float hs = ht.r * hr + hmin;
            float thick_w = max(ht.g * hr, min_thick_w);
            float hr2 = height_val(p, axis);
            if (!entered) {
                entered = true;
                if (hr2 < hs - thick_w * 0.5) {
                    return;
                }
            }
            if (abs(hr2 - hs) <= thick_w * 0.5) {
                float ta = t - step; if (ta < pt0) ta = pt0;
                float tb = t;
                for (int it = 0; it < 8; it++) {
                    float tm = (ta + tb) * 0.5;
                    p = ro + rd * tm;
                    pr = project(p, axis);
                    tx = (pr.x - ox) * td + 0.5 + float(pm.atlas_x);
                    ty = (pr.y - oy) * td + 0.5 + float(pm.atlas_y);
                    bool sv2;
                    vec2 htf = sample_height_bilinear(vec2(tx, ty), pm, sv2);
                    hs = sv2 ? (htf.r * hr + hmin) : -1e30;
                    hr2 = height_val(p, axis);
                    if (hr2 <= hs) tb = tm; else ta = tm;
                }
                t = (ta + tb) * 0.5;
                if (t < best_t) { best_t = t; best_pid = pid; best_pos = ro + rd * t; }
                break;
            }
        }
        t += step;
    }
}

void main() {
    hit_patch = -1;
    hit_t = 1e30;
    hit_px = hit_py = hit_pz = 0.0;

    vec3 ro = u_ro;
    vec3 rd = normalize(u_rd);
    ivec2 asz = textureSize(u_height_sampler, 0);
    float td = u_texel_density;
    float best_t = 1e30;
    int best_pid = -1;
    vec3 best_pos = vec3(0.0);

    // ---------- BVH traversal ----------
    int stack[128]; int sp = 0;
    for (int bi = 0; bi < 6; bi++) { int r = bvh_roots[bi]; if (r >= 0) stack[sp++] = (r << 3) | bi; }

    while (sp > 0) {
        int nidx_packed = stack[--sp];
        int nidx = nidx_packed >> 3;
        int b = nidx_packed & 7;
        if (sp > 120) break;

        vec3 mn = vec3(nodes[nidx].min_x, nodes[nidx].min_y, nodes[nidx].min_z);
        vec3 mx = vec3(nodes[nidx].max_x, nodes[nidx].max_y, nodes[nidx].max_z);

        float t0 = -1e30, t1 = 1e30;
        for (int a = 0; a < 3; a++) {
            float inv = 1.0 / rd[a];
            float ta = (mn[a] - ro[a]) * inv, tb = (mx[a] - ro[a]) * inv;
            if (ta > tb) { float tt = ta; ta = tb; tb = tt; }
            t0 = max(t0, ta); t1 = min(t1, tb);
        }
        if (t0 > t1 || t1 < 0.0 || t0 >= best_t) continue;

        // Depth culling: check if the AABB entry point is behind the visible surface
        {
            vec3 p0 = ro + rd * t0;
            vec4 clip = u_mvp * vec4(p0, 1.0);
            vec3 ndc = clip.xyz / clip.w;
            vec2 duv = ndc.xy * 0.5 + 0.5;
            if (duv.x >= 0.0 && duv.x <= 1.0 && duv.y >= 0.0 && duv.y <= 1.0) {
                float stored_depth = textureLod(u_depth_tex, duv, 0.0).r;
                float node_depth = ndc.z * 0.5 + 0.5;
                if (node_depth > stored_depth + 0.0005) continue;
            }
        }

        int meta = nodes[nidx].meta;
        int cb = nodes[nidx].child_base;

        if ((meta & LEAF_FLAG) != 0) {
            int lc = meta & ~LEAF_FLAG;
            int bo = u_bin_offsets[b];
            for (int li = 0; li < lc; li++) {
                int pid = int(data[bo + nodes[nidx].l + li] & 0xFFFFFFFFu);
                int base = pid * 6;
                vec3 pmn = vec3(aabb_float[base], aabb_float[base+1], aabb_float[base+2]);
                vec3 pmx = vec3(aabb_float[base+3], aabb_float[base+4], aabb_float[base+5]);
                float pt0 = -1e30, pt1 = 1e30;
                for (int a = 0; a < 3; a++) {
                    float inv = 1.0 / rd[a];
                    float ta = (pmn[a] - ro[a]) * inv, tb = (pmx[a] - ro[a]) * inv;
                    if (ta > tb) { float tt = ta; ta = tb; tb = tt; }
                    pt0 = max(pt0, ta); pt1 = min(pt1, tb);
                }
                if (pt0 > pt1 || pt1 < 0.0 || pt0 >= best_t) continue;
                if (pt0 < 0.0) pt0 = 0.0;
                march_patch(pid, ro, rd, pt0, pt1, asz, td, best_t, best_pid, best_pos);
            }
        } else {
            int cc = meta;
            for (int i = cc - 1; i >= 0 && sp < 128; i--) {
                int child = cb + i;
                stack[sp++] = (child << 3) | b;
            }
        }
    }

    // ---------- Linear fallback (if BVH missed) ----------
    if (best_pid < 0) {
        for (int pid = 0; pid < u_patch_count; pid++) {
            int b = pid * 6;
            vec3 pmn = vec3(aabb_float[b], aabb_float[b+1], aabb_float[b+2]);
            vec3 pmx = vec3(aabb_float[b+3], aabb_float[b+4], aabb_float[b+5]);
            float pt0 = -1e30, pt1 = 1e30;
            for (int a = 0; a < 3; a++) {
                float inv = 1.0 / rd[a];
                float ta = (pmn[a] - ro[a]) * inv, tb = (pmx[a] - ro[a]) * inv;
                if (ta > tb) { float tt = ta; ta = tb; tb = tt; }
                pt0 = max(pt0, ta); pt1 = min(pt1, tb);
            }
            if (pt0 > pt1 || pt1 < 0.0 || pt0 >= best_t) continue;
            if (pt0 < 0.0) pt0 = 0.0;
            march_patch(pid, ro, rd, pt0, pt1, asz, td, best_t, best_pid, best_pos);
        }
    }

    if (best_pid >= 0) {
        hit_patch = best_pid;
        hit_t = best_t;
        hit_px = best_pos.x; hit_py = best_pos.y; hit_pz = best_pos.z;
    }
}
)";

static const char* cs_recon_src = R"(
#version 460 core
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 16, local_size_y = 16) in;
struct PatchMeta {
    int  dominant_axis;
    int  atlas_x, atlas_y;
    int  atlas_w, atlas_h;
    float proj_origin_x, proj_origin_y;
    float height_min, height_max;
    int  _pad[3];
};
struct BVHNode {
    float min_x, min_y, min_z;
    int   child_base;
    float max_x, max_y, max_z;
    int   meta;
    int   l, r;
    int   pad0, pad1;
};
layout(std430, binding = 10) readonly buffer MetaBuf { PatchMeta patches[]; };
layout(std430, binding = 13) readonly buffer AABBBuf { float aabb_float[]; };
layout(std430, binding = 14) readonly buffer FlatBuf { uint64_t data[]; };
layout(std430, binding = 15) readonly buffer NodeBuf { BVHNode nodes[]; };
layout(std430, binding = 16) readonly buffer RootBuf { int bvh_roots[6]; };
uniform int u_bin_offsets[6];
uniform sampler2D u_height_sampler;
uniform sampler2D u_height_linear;
uniform sampler2D u_recon_depth;
uniform sampler2D u_diag_sampler;
uniform mat4 u_inv_view_proj;
uniform mat4 u_inv_model;
uniform mat4 u_model;
uniform vec3 u_mesh_aabb_min;
uniform vec3 u_mesh_aabb_max;
uniform int  u_patch_count;
uniform float u_texel_density;
uniform ivec2 u_view_size;
uniform bool u_use_fallback;
uniform int u_highlight_patch;
layout(rgba32f, binding = 0) uniform image2D u_out_pos;
layout(rgba16f, binding = 1) uniform image2D u_out_normal;
#define LEAF_FLAG 0x80000000
float height_val(vec3 p, int axis) {
    if (axis == 0) return  p.x; if (axis == 1) return -p.x;
    if (axis == 2) return  p.y; if (axis == 3) return -p.y;
    if (axis == 4) return  p.z; return -p.z;
}
vec2 project(vec3 p, int axis) {
    if (axis == 0 || axis == 1) return vec2(p.y, p.z);
    if (axis == 2 || axis == 3) return vec2(p.x, p.z);
    return vec2(p.x, p.y);
}
vec3 dominant_normal(int axis) {
    if (axis == 0) return vec3( 1.0,  0.0,  0.0);
    if (axis == 1) return vec3(-1.0,  0.0,  0.0);
    if (axis == 2) return vec3( 0.0,  1.0,  0.0);
    if (axis == 3) return vec3( 0.0, -1.0,  0.0);
    if (axis == 4) return vec3( 0.0,  0.0,  1.0);
    return vec3(0.0, 0.0, -1.0);
}

vec3 compute_normal_smooth(vec2 hit_uv, PatchMeta pm, float hr, ivec2 asz) {
    float tx = hit_uv.x - 0.5;
    float ty = hit_uv.y - 0.5;
    int ix0 = int(floor(tx)), iy0 = int(floor(ty));
    int ix1 = ix0 + 1, iy1 = iy0 + 1;
    float fx = tx - float(ix0), fy = ty - float(iy0);

    // Clamp to patch atlas region so we never read from a neighbouring patch
    int ax = pm.atlas_x, ay = pm.atlas_y;
    int aw = pm.atlas_w, ah = pm.atlas_h;
    int x0 = clamp(ix0, ax, ax + aw - 1);
    int x1 = clamp(ix1, ax, ax + aw - 1);
    int y0 = clamp(iy0, ay, ay + ah - 1);
    int y1 = clamp(iy1, ay, ay + ah - 1);

    // Only interpolate texels with non-zero coverage
    vec2 hc00 = texelFetch(u_height_sampler, ivec2(x0, y0), 0).rg;
    vec2 hc10 = texelFetch(u_height_sampler, ivec2(x1, y0), 0).rg;
    vec2 hc01 = texelFetch(u_height_sampler, ivec2(x0, y1), 0).rg;
    vec2 hc11 = texelFetch(u_height_sampler, ivec2(x1, y1), 0).rg;

    bool c00 = hc00.g > 1e-6, c10 = hc10.g > 1e-6;
    bool c01 = hc01.g > 1e-6, c11 = hc11.g > 1e-6;
    float h00 = hc00.r, h10 = hc10.r, h01 = hc01.r, h11 = hc11.r;

    int nc = int(c00) + int(c10) + int(c01) + int(c11);
    if (nc < 2) return dominant_normal(pm.dominant_axis);

    float dh_dx = 0.0; int nx = 0;
    float dh_dy = 0.0; int ny = 0;
    if (c00 && c10) { dh_dx += (1.0-fy) * (h10 - h00); nx++; }
    if (c01 && c11) { dh_dx += fy * (h11 - h01); nx++; }
    if (c00 && c01) { dh_dy += (1.0-fx) * (h01 - h00); ny++; }
    if (c10 && c11) { dh_dy += fx * (h11 - h10); ny++; }
    if (nx == 0 && ny == 0) {
        if (c00 && c11) { float d = h11 - h00; dh_dx = d; dh_dy = d; }
        else if (c10 && c01) { float d = h01 - h10; dh_dx = d; dh_dy = d; }
    }

    float dhdu = dh_dx * hr * u_texel_density;
    float dhdv = dh_dy * hr * u_texel_density;
    vec3 u_dir, v_dir, h_dir;
    int axis = pm.dominant_axis;
    if (axis == 0) { u_dir = vec3(0,1,0); v_dir = vec3(0,0,1); h_dir = vec3(1,0,0); }
    else if (axis == 1) { u_dir = vec3(0,1,0); v_dir = vec3(0,0,1); h_dir = vec3(-1,0,0); }
    else if (axis == 2) { u_dir = vec3(1,0,0); v_dir = vec3(0,0,1); h_dir = vec3(0,1,0); }
    else if (axis == 3) { u_dir = vec3(1,0,0); v_dir = vec3(0,0,1); h_dir = vec3(0,-1,0); }
    else if (axis == 4) { u_dir = vec3(1,0,0); v_dir = vec3(0,1,0); h_dir = vec3(0,0,1); }
    else { u_dir = vec3(1,0,0); v_dir = vec3(0,1,0); h_dir = vec3(0,0,-1); }
    return normalize(-dhdu * u_dir - dhdv * v_dir + h_dir);
}

vec2 sample_height_safe(vec2 tx_ty, PatchMeta pm) {
    float tx = tx_ty.x - 0.5;
    float ty = tx_ty.y - 0.5;
    int ix0 = int(floor(tx)), iy0 = int(floor(ty));
    int ix1 = ix0 + 1, iy1 = iy0 + 1;
    float fx = tx - float(ix0), fy = ty - float(iy0);

    // Clamp to patch atlas region so we never read from a neighbouring patch
    int ax = pm.atlas_x, ay = pm.atlas_y;
    int aw = pm.atlas_w, ah = pm.atlas_h;
    ix0 = clamp(ix0, ax, ax + aw - 1);
    ix1 = clamp(ix1, ax, ax + aw - 1);
    iy0 = clamp(iy0, ay, ay + ah - 1);
    iy1 = clamp(iy1, ay, ay + ah - 1);

    // Only interpolate texels with non-zero coverage
    vec2 hc00 = texelFetch(u_height_sampler, ivec2(ix0, iy0), 0).rg;
    vec2 hc10 = texelFetch(u_height_sampler, ivec2(ix1, iy0), 0).rg;
    vec2 hc01 = texelFetch(u_height_sampler, ivec2(ix0, iy1), 0).rg;
    vec2 hc11 = texelFetch(u_height_sampler, ivec2(ix1, iy1), 0).rg;

    bool c00 = hc00.g > 1e-6, c10 = hc10.g > 1e-6;
    bool c01 = hc01.g > 1e-6, c11 = hc11.g > 1e-6;

    float h00 = hc00.r, h10 = hc10.r, h01 = hc01.r, h11 = hc11.r;

    int nc = int(c00) + int(c10) + int(c01) + int(c11);
    if (nc < 1) return vec2(-1e30, 0.0);
    if (nc == 1) {
        float h = c00 ? h00 : (c01 ? h01 : (c10 ? h10 : h11));
        return vec2(h, 0.0);
    }

    float sum_h = 0.0, sum_w = 0.0;
    float w00 = (1.0-fx)*(1.0-fy), w10 = fx*(1.0-fy), w01 = (1.0-fx)*fy, w11 = fx*fy;
    if (c00) { sum_h += h00 * w00; sum_w += w00; }
    if (c10) { sum_h += h10 * w10; sum_w += w10; }
    if (c01) { sum_h += h01 * w01; sum_w += w01; }
    if (c11) { sum_h += h11 * w11; sum_w += w11; }
    float h = sum_h / sum_w;

    float dh_dx = 0.0; int nx = 0;
    float dh_dy = 0.0; int ny = 0;
    if (c00 && c10) { dh_dx += (1.0-fy) * (h10 - h00); nx++; }
    if (c01 && c11) { dh_dx += fy * (h11 - h01); nx++; }
    if (c00 && c01) { dh_dy += (1.0-fx) * (h01 - h00); ny++; }
    if (c10 && c11) { dh_dy += fx * (h11 - h10); ny++; }
    if (nx == 0 && ny == 0) {
        if (c00 && c11) { float d = h11 - h00; dh_dx = d; dh_dy = d; }
        else if (c10 && c01) { float d = h01 - h10; dh_dx = d; dh_dy = d; }
    }
    float grad = max(length(vec2(dh_dx, dh_dy)), 1e-6);
    return vec2(h, grad);
}

// Pure height-field ray-march with thickness-envelope sampling
bool dda_patch_march(PatchMeta pm, vec3 ro, vec3 rd, float td,
                     float pt0, float pt1, float best_t, ivec2 asz,
                     out float hit_t, out vec3 hit_pos, out vec3 hit_n, out vec2 hit_uv) {
    int axis = pm.dominant_axis;
    float ox = pm.proj_origin_x, oy = pm.proj_origin_y;
    float hmin = pm.height_min, hmax = pm.height_max;
    float hr = max(hmax - hmin, 1e-10);

    vec2 rd_uv = project(rd, axis);
    float step = min(0.5 / td, (1.0 / td) / max(length(rd_uv), 1e-4));
    float h_step = step * abs(height_val(rd, axis));
    float min_thick_w = h_step * 2.0;

    bool entered = false;
    int underground_count = 0;
    float t = pt0;
    float tmax = min(pt1, best_t);
    while (t <= tmax) {
        vec3 p = ro + rd * t;
        vec2 pr = project(p, axis);
        float tx = (pr.x - ox) * td + 0.5 + float(pm.atlas_x);
        float ty = (pr.y - oy) * td + 0.5 + float(pm.atlas_y);
        if (tx < float(pm.atlas_x) || tx >= float(pm.atlas_x + pm.atlas_w) ||
            ty < float(pm.atlas_y) || ty >= float(pm.atlas_y + pm.atlas_h)) {
            t += step; continue;
        }
        vec2 ht = sample_height_safe(vec2(tx, ty), pm);
        float hf_val = ht.x;
        if (hf_val < -5e5) { t += step; continue; }
        float thick = ht.y;
        float hf_w = hf_val * hr + hmin;
        float thick_w = (thick < 1e-6) ? min_thick_w : max(thick * hr, min_thick_w);
        float hr_w = height_val(p, axis);
        if (!entered) { entered = true; }
        if (hr_w < hf_w - thick_w * 0.5) {
            underground_count++;
            if (underground_count >= 3) return false;
        } else {
            underground_count = 0;
        }
        if (abs(hr_w - hf_w) <= thick_w * 0.5) {
            float ta = max(t - step, pt0);
            float tb = t;
            for (int it = 0; it < 8; it++) {
                float tm = (ta + tb) * 0.5;
                p = ro + rd * tm;
                pr = project(p, axis);
                tx = (pr.x - ox) * td + 0.5 + float(pm.atlas_x);
                ty = (pr.y - oy) * td + 0.5 + float(pm.atlas_y);
                ht = sample_height_safe(vec2(tx, ty), pm);
                hf_w = ht.x * hr + hmin;
                thick = ht.y;
                thick_w = (thick < 1e-6) ? (min_thick_w * 2.0) : max(thick * hr, min_thick_w);
                hr_w = height_val(p, axis);
                if (hr_w <= hf_w) tb = tm; else ta = tm;
            }
            hit_t = (ta + tb) * 0.5;
            hit_pos = ro + rd * hit_t;
            pr = project(hit_pos, axis);
            tx = (pr.x - ox) * td + 0.5 + float(pm.atlas_x);
            ty = (pr.y - oy) * td + 0.5 + float(pm.atlas_y);
            hit_uv = vec2(tx, ty);
            hit_n = compute_normal_smooth(hit_uv, pm, hr, asz);
            return true;
        }
        t += step;
    }
    return false;
}

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    if (px.x >= u_view_size.x || px.y >= u_view_size.y) return;

    float truth_depth = texelFetch(u_recon_depth, px, 0).r;
    if (truth_depth >= 0.9999) { imageStore(u_out_pos, px, vec4(0.0)); imageStore(u_out_normal, px, vec4(0.0)); return; }

    vec2 ndc = (vec2(px) + 0.5) / vec2(u_view_size) * 2.0 - 1.0;
    vec4 nh = u_inv_view_proj * vec4(ndc, -1.0, 1.0);
    vec4 fh = u_inv_view_proj * vec4(ndc,  1.0, 1.0);
    vec3 ro_w = nh.xyz / nh.w;
    vec3 rd_w = normalize(fh.xyz / fh.w - ro_w);
    vec3 ro = (u_inv_model * vec4(ro_w, 1.0)).xyz;
    vec3 rd = normalize((u_inv_model * vec4(rd_w, 0.0)).xyz);

    {
        float t0 = -1e30, t1 = 1e30;
        for (int a = 0; a < 3; a++) {
            float inv = 1.0 / rd[a];
            float ta = (u_mesh_aabb_min[a] - ro[a]) * inv;
            float tb = (u_mesh_aabb_max[a] - ro[a]) * inv;
            if (ta > tb) { float tmp = ta; ta = tb; tb = tmp; }
            t0 = max(t0, ta); t1 = min(t1, tb);
        }
        if (t0 > t1 || t1 < 0.0) { imageStore(u_out_pos, px, vec4(0.0)); imageStore(u_out_normal, px, vec4(0.0)); return; }
    }

    float td = u_texel_density;
    float best_t = 1e30;
    vec3 best_pos = vec3(0.0);
    vec2 best_uv = vec2(0.0);
    vec3 best_n = vec3(0.0);
    int best_pid = -1;

    // Use bilinear marching (same as pick shader) to find hit on each patch
    ivec2 asz = textureSize(u_height_sampler, 0);
    int stack[128];
    int sp = 0;
    for (int bi = 0; bi < 6; bi++) { int r = bvh_roots[bi]; if (r >= 0) stack[sp++] = (r << 3) | bi; }
    while (sp > 0) {
        int nidx_packed = stack[--sp];
        int nidx = nidx_packed >> 3;
        int b = nidx_packed & 7;
        if (sp > 120) break;
        vec3 mn = vec3(nodes[nidx].min_x, nodes[nidx].min_y, nodes[nidx].min_z);
        vec3 mx = vec3(nodes[nidx].max_x, nodes[nidx].max_y, nodes[nidx].max_z);
        float t0 = -1e30, t1 = 1e30;
        for (int a = 0; a < 3; a++) {
            float inv = 1.0 / rd[a];
            float ta = (mn[a] - ro[a]) * inv, tb = (mx[a] - ro[a]) * inv;
            if (ta > tb) { float tt = ta; ta = tb; tb = tt; }
            t0 = max(t0, ta); t1 = min(t1, tb);
        }
        if (t0 > t1 || t1 < 0.0 || t0 >= best_t) continue;
        int meta = nodes[nidx].meta;
        int cb = nodes[nidx].child_base;
        if ((meta & LEAF_FLAG) != 0) {
            int lc = meta & ~LEAF_FLAG;
            int bo = u_bin_offsets[b];
            for (int li = 0; li < lc; li++) {
                uint64_t d = data[uint(bo + nodes[nidx].l + li)];
                int pid = int(uint(d & 0xFFFFFFFFu));
                if (u_highlight_patch >= 0 && pid != u_highlight_patch) continue;
                int base = pid * 6;
                vec3 pmn = vec3(aabb_float[base], aabb_float[base+1], aabb_float[base+2]);
                vec3 pmx = vec3(aabb_float[base+3], aabb_float[base+4], aabb_float[base+5]);
                float pt0 = -1e30, pt1 = 1e30;
                for (int a = 0; a < 3; a++) {
                    float inv = 1.0 / rd[a];
                    float ta = (pmn[a] - ro[a]) * inv, tb = (pmx[a] - ro[a]) * inv;
                    if (ta > tb) { float tt = ta; ta = tb; tb = tt; }
                    pt0 = max(pt0, ta); pt1 = min(pt1, tb);
                }
                if (pt0 > pt1 || pt1 < 0.0 || pt0 >= best_t) continue;
                if (pt0 < 0.0) pt0 = 0.0;
                PatchMeta pm = patches[pid];
                int axis = pm.dominant_axis;
                if (dot(rd, dominant_normal(axis)) >= 0.0) continue;

                float hit_t; vec3 hit_pos, hit_n; vec2 hit_uv;
                if (dda_patch_march(pm, ro, rd, td, pt0, pt1, best_t, asz,
                                    hit_t, hit_pos, hit_n, hit_uv)) {
                    if (hit_t < best_t) {
                        best_t = hit_t; best_pos = hit_pos;
                        best_uv = hit_uv; best_pid = pid; best_n = hit_n;
                    }
                }
            }
        } else {
            int cc = meta;
            for (int i = cc - 1; i >= 0 && sp < 128; i--) {
                int child = cb + i;
                stack[sp++] = (child << 3) | b;
            }
        }
    }
    if (best_t >= 1e29 && u_use_fallback) {
        for (int pid = 0; pid < u_patch_count; pid++) {
            if (u_highlight_patch >= 0 && pid != u_highlight_patch) continue;
            int b = pid * 6;
            vec3 pmn = vec3(aabb_float[b], aabb_float[b+1], aabb_float[b+2]);
            vec3 pmx = vec3(aabb_float[b+3], aabb_float[b+4], aabb_float[b+5]);
            float pt0 = -1e30, pt1 = 1e30;
            for (int a = 0; a < 3; a++) {
                float inv = 1.0 / rd[a];
                float ta = (pmn[a] - ro[a]) * inv, tb = (pmx[a] - ro[a]) * inv;
                if (ta > tb) { float tt = ta; ta = tb; tb = tt; }
                pt0 = max(pt0, ta); pt1 = min(pt1, tb);
            }
            if (pt0 > pt1 || pt1 < 0.0 || pt0 >= best_t) continue;
            if (pt0 < 0.0) pt0 = 0.0;
            PatchMeta pm = patches[pid];
            int axis = pm.dominant_axis;
            if (dot(rd, dominant_normal(axis)) >= 0.0) continue;

            float hit_t; vec3 hit_pos, hit_n; vec2 hit_uv;
            if (dda_patch_march(pm, ro, rd, td, pt0, pt1, best_t, asz,
                                hit_t, hit_pos, hit_n, hit_uv)) {
                if (hit_t < best_t) {
                    best_t = hit_t; best_pos = hit_pos;
                    best_uv = hit_uv; best_pid = pid; best_n = hit_n;
                }
            }
        }
    }
    if (best_t < 1e29) {
        // Cliff check: reject recon hits that are far from the truth surface
        float truth_z_ndc = truth_depth * 2.0 - 1.0;
        vec4 truth_w_clip = u_inv_view_proj * vec4(ndc, truth_z_ndc, 1.0);
        vec3 truth_w = truth_w_clip.xyz / truth_w_clip.w;
        vec3 recon_w = (u_model * vec4(best_pos, 1.0)).xyz;
        float scene_scale = length(u_mesh_aabb_max - u_mesh_aabb_min);
        if (distance(recon_w, truth_w) > 0.002 * scene_scale) {
            imageStore(u_out_pos, px, vec4(0.0));
            imageStore(u_out_normal, px, vec4(0.0));
            return;
        }

        vec3 pos_w = recon_w;
        PatchMeta bpm = patches[best_pid];
        float bhr = max(bpm.height_max - bpm.height_min, 1e-10);
        ivec2 asz = textureSize(u_height_sampler, 0);
        best_n = compute_normal_smooth(best_uv, bpm, bhr, asz);
        vec3 n_world = normalize(transpose(mat3(u_inv_model)) * best_n);
        imageStore(u_out_pos, px, vec4(pos_w, 1.0));
        imageStore(u_out_normal, px, vec4(n_world * 0.5 + 0.5, 1.0));
    } else {
        imageStore(u_out_pos, px, vec4(0.0));
        imageStore(u_out_normal, px, vec4(0.0));
    }
}
)";

// --- Compare reconstructed normals with ground truth ---

// --- Compare reconstructed normals with ground truth ---
static const char* cs_compare_src = R"(
#version 460 core
layout(local_size_x = 16, local_size_y = 16) in;
uniform ivec2 u_view_size;
layout(rgba16f, binding = 1) readonly uniform image2D u_recon_normal;
layout(rgba16f, binding = 2) readonly uniform image2D u_truth_normal;
layout(std430, binding = 30) buffer AccBuf {
    int  hit_pixels;
    int  match_pixels;
    float hit_ratio;
    float match_ratio;
};
shared uint sh_hit;
shared uint sh_match;
void main() {
    if (gl_LocalInvocationIndex == 0u) { sh_hit = 0u; sh_match = 0u; }
    barrier();
    memoryBarrierShared();
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    if (px.x < u_view_size.x && px.y < u_view_size.y) {
        vec3 rn = imageLoad(u_recon_normal, px).xyz * 2.0 - 1.0;
        vec3 tn = imageLoad(u_truth_normal, px).xyz * 2.0 - 1.0;
        if (length(rn) > 0.01 && length(tn) > 0.01) {
            atomicAdd(sh_hit, 1u);
            if (dot(normalize(rn), normalize(tn)) > 0.95) atomicAdd(sh_match, 1u);
        }
    }
    barrier();
    memoryBarrierShared();
    if (gl_LocalInvocationIndex == 0u) {
        atomicAdd(hit_pixels, int(sh_hit));
        atomicAdd(match_pixels, int(sh_match));
    }
}
)";

// ==================== RASTERIZE HEIGHT & UV ATLASES ===========================
static const char* cs_rasterize_src = R"(
#version 460 core
layout(local_size_x = 256) in;

struct TriData {
    vec3 normal;
    float _pad0;
    vec3 center;
    float _pad1;
    int  face_mask;
    int  patch_id;
    int  _pad2[2];
};

struct PatchMeta {
    int  dominant_axis; // 0:+X,1:-X,2:+Y,3:-Y,4:+Z,5:-Z
    int  atlas_x, atlas_y;
    int  atlas_w, atlas_h;
    float proj_origin_x, proj_origin_y;
    float height_min, height_max; // for normalized depth [0..1]
    int  _pad[3];
};

layout(std430, binding = 0) readonly buffer VertBuf { float verts[]; };
layout(std430, binding = 1) readonly buffer TriBuf  { TriData tris[]; };
layout(std430, binding = 2) readonly buffer IdxBuf  { uint idx[]; };
layout(std430, binding = 10) readonly buffer PatchMetaBuf { PatchMeta patches[]; };

layout(binding = 0, rg16f) uniform image2D u_height_atlas;
layout(binding = 1, rg32f) uniform image2D u_uv_atlas;

uniform uint  u_tri_count;
uniform int   u_patch_count;
uniform float u_texel_density;

vec2 project(vec3 p, int axis) {
    if (axis == 0 || axis == 1) return vec2(p.y, p.z); // +X / -X -> YZ
    if (axis == 2 || axis == 3) return vec2(p.x, p.z); // +Y / -Y -> XZ
    return vec2(p.x, p.y); // +Z / -Z -> XY
}

float height_val(vec3 p, int axis) {
    if (axis == 0) return  p.x; // +X
    if (axis == 1) return -p.x; // -X
    if (axis == 2) return  p.y; // +Y
    if (axis == 3) return -p.y; // -Y
    if (axis == 4) return  p.z; // +Z
    return -p.z;                // -Z
}

ivec2 to_texel(PatchMeta pm, vec3 p, float texel_density) {
    vec2 proj = project(p, pm.dominant_axis);
    int x = int((proj.x - pm.proj_origin_x) * texel_density + 0.5) + pm.atlas_x;
    int y = int((proj.y - pm.proj_origin_y) * texel_density + 0.5) + pm.atlas_y;
    return ivec2(x, y);
}

float edge(vec2 a, vec2 b, vec2 p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_tri_count) return;

    int pid = tris[t].patch_id;
    if (pid < 0 || pid >= u_patch_count) return;

    PatchMeta pm = patches[pid];
    if (pm.dominant_axis < 0) return;

    uint i0 = idx[t * 3];
    uint i1 = idx[t * 3 + 1];
    uint i2 = idx[t * 3 + 2];

    vec3 p0 = vec3(verts[i0 * 12], verts[i0 * 12 + 1], verts[i0 * 12 + 2]);
    vec3 p1 = vec3(verts[i1 * 12], verts[i1 * 12 + 1], verts[i1 * 12 + 2]);
    vec3 p2 = vec3(verts[i2 * 12], verts[i2 * 12 + 1], verts[i2 * 12 + 2]);

    float h0 = height_val(p0, pm.dominant_axis);
    float h1 = height_val(p1, pm.dominant_axis);
    float h2 = height_val(p2, pm.dominant_axis);

    ivec2 t0 = to_texel(pm, p0, u_texel_density);
    ivec2 t1 = to_texel(pm, p1, u_texel_density);
    ivec2 t2 = to_texel(pm, p2, u_texel_density);

    ivec2 bb_min = min(t0, min(t1, t2));
    ivec2 bb_max = max(t0, max(t1, t2));

    ivec2 region_min = ivec2(pm.atlas_x, pm.atlas_y);
    ivec2 region_max = ivec2(pm.atlas_x + pm.atlas_w - 1,
                              pm.atlas_y + pm.atlas_h - 1);
    bb_min = max(bb_min, region_min);
    bb_max = min(bb_max, region_max);
    if (bb_min.x > bb_max.x || bb_min.y > bb_max.y) return;

    vec2 f0 = vec2(t0);
    vec2 f1 = vec2(t1);
    vec2 f2 = vec2(t2);

    float area2 = edge(f0, f1, f2);
    if (abs(area2) < 1e-10) return;
    float inv_area2 = 1.0 / area2;

    // Barycentric derivatives w.r.t. texel x,y (constant per triangle)
    float dw0_dx = (f1.y - f2.y) * inv_area2;
    float dw0_dy = (f2.x - f1.x) * inv_area2;
    float dw1_dx = (f2.y - f0.y) * inv_area2;
    float dw1_dy = (f0.x - f2.x) * inv_area2;
    float dw2_dx = (f0.y - f1.y) * inv_area2;
    float dw2_dy = (f1.x - f0.x) * inv_area2;

    float h_range = max(pm.height_max - pm.height_min, 1e-10);

    for (int y = bb_min.y; y <= bb_max.y; ++y) {
        for (int x = bb_min.x; x <= bb_max.x; ++x) {
            vec2 p = vec2(float(x) + 0.5, float(y) + 0.5);

            float w0 = edge(f1, f2, p) * inv_area2;
            float w1 = edge(f2, f0, p) * inv_area2;
            float w2 = edge(f0, f1, p) * inv_area2;

            if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) {
                float h = w0 * h0 + w1 * h1 + w2 * h2;
                float h_norm = (h - pm.height_min) / h_range;

                // Thickness = height gradient magnitude (normalized)
                float dh_dx = dw0_dx * h0 + dw1_dx * h1 + dw2_dx * h2;
                float dh_dy = dw0_dy * h0 + dw1_dy * h1 + dw2_dy * h2;
                float thickness = max(length(vec2(dh_dx, dh_dy)) / h_range, 1e-6);

                imageStore(u_height_atlas, ivec2(x, y), vec4(h_norm, thickness, 0.0, 0.0));
                imageStore(u_uv_atlas,    ivec2(x, y), vec4(w0, w1, 0.0, 0.0));
            }
        }
    }
}
)";

// ==================== DIAGONAL DIRECTION COMPUTATION =============
// For each 2x2 texel block, picks the diagonal that minimizes height difference
static const char* cs_diag_src = R"(
#version 460 core
layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0, r8) uniform image2D u_diag_img;
uniform sampler2D u_height_sampler;
uniform ivec2 u_atlas_size;
void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    if (px.x >= u_atlas_size.x || px.y >= u_atlas_size.y) return;
    // Check if we're inside the valid cell range (need i+1, j+1 to exist)
    if (px.x >= u_atlas_size.x - 1 || px.y >= u_atlas_size.y - 1) {
        imageStore(u_diag_img, px, vec4(0.0));
        return;
    }
    float m00 = texelFetch(u_height_sampler, px + ivec2(0,0), 0).g;
    float m10 = texelFetch(u_height_sampler, px + ivec2(1,0), 0).g;
    float m01 = texelFetch(u_height_sampler, px + ivec2(0,1), 0).g;
    float m11 = texelFetch(u_height_sampler, px + ivec2(1,1), 0).g;
    if (m00 < 1e-6 || m10 < 1e-6 || m01 < 1e-6 || m11 < 1e-6) {
        imageStore(u_diag_img, px, vec4(0.0));
        return;
    }
    float h00 = texelFetch(u_height_sampler, px + ivec2(0,0), 0).r;
    float h10 = texelFetch(u_height_sampler, px + ivec2(1,0), 0).r;
    float h01 = texelFetch(u_height_sampler, px + ivec2(0,1), 0).r;
    float h11 = texelFetch(u_height_sampler, px + ivec2(1,1), 0).r;
    float d_even = abs(h00 - h11);
    float d_odd  = abs(h10 - h01);
    imageStore(u_diag_img, px, vec4(d_even <= d_odd ? 0.0 : 1.0, 0.0, 0.0, 0.0));
}
)";

// ==================== VERTEX / FRAGMENT SHADERS ===========================

static const char* pbr_vs = R"(
#version 430 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

uniform mat4 u_model;
uniform mat4 u_view_proj;
uniform mat3 u_normal_mat;

out vec3 v_pos;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    vec4 world = u_model * vec4(a_pos, 1.0);
    gl_Position = u_view_proj * world;
    v_pos = world.xyz;
    v_normal = normalize(u_normal_mat * a_normal);
    v_uv = a_uv;
}
)";

static const char* simple_fs = R"(
#version 430 core
in vec3 v_pos;
in vec3 v_normal;
in vec2 v_uv;
    layout(location = 0) out vec4 frag_color;

    uniform vec3 u_color;
uniform vec3 u_light_dir;
uniform int u_debug_mode;
uniform int u_face_filter;
uniform int u_highlight_patch;

struct TriData {
    vec3 normal;
    float _pad0;
    vec3 center;
    float _pad1;
    int  face_mask;
    int  patch_id;
    int  _pad2[2];
};

layout(std430, binding = 1) buffer TriBuf { TriData tris[]; };

const vec3 face_colors[6] = vec3[](
    vec3(1.0, 0.2, 0.2),
    vec3(0.2, 0.6, 1.0),
    vec3(0.2, 1.0, 0.2),
    vec3(1.0, 1.0, 0.2),
    vec3(1.0, 0.6, 0.2),
    vec3(0.8, 0.2, 1.0)
);

uint hash_int(int v) {
    uint x = uint(v) * 0x9e3779b1u;
    x = (x ^ (x >> 16)) * 0x85ebca6bu;
    x = x ^ (x >> 13);
    return x;
}

void main() {
    int mask = tris[gl_PrimitiveID].face_mask;
    if (u_face_filter > 0) {
        int bit = 1 << (u_face_filter - 1);
        if ((mask & bit) == 0) discard;
    }

    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    float shade = max(dot(N, -L), 0.0) * 0.6 + 0.4;

    if (u_debug_mode == 2) {
        vec3 col = vec3(0.0);
        if      ((mask & 1)  != 0) col = face_colors[0];
        else if ((mask & 2)  != 0) col = face_colors[1];
        else if ((mask & 4)  != 0) col = face_colors[2];
        else if ((mask & 8)  != 0) col = face_colors[3];
        else if ((mask & 16) != 0) col = face_colors[4];
        else if ((mask & 32) != 0) col = face_colors[5];
        else col = vec3(0.15);
        frag_color = vec4(col * shade, 1.0);
        return;
    }

    if (u_debug_mode == 1) {
        frag_color = vec4(N * 0.5 + 0.5, 1.0);
        return;
    }

    if (u_debug_mode == 3) {
        int pid = tris[gl_PrimitiveID].patch_id;
        if (pid < 0) discard;
        uint h = hash_int(pid + 1);
        vec3 col = vec3(
            float((h      ) & 0xFFu) / 255.0,
            float((h >> 8 ) & 0xFFu) / 255.0,
            float((h >> 16) & 0xFFu) / 255.0
        );
        frag_color = vec4(col * shade, 1.0);
        return;
    }

    if (u_highlight_patch >= 0) {
        int pid = tris[gl_PrimitiveID].patch_id;
        if (pid == u_highlight_patch) {
            vec3 col = vec3(0.0);
            if      ((mask & 1)  != 0) col = face_colors[0];
            else if ((mask & 2)  != 0) col = face_colors[1];
            else if ((mask & 4)  != 0) col = face_colors[2];
            else if ((mask & 8)  != 0) col = face_colors[3];
            else if ((mask & 16) != 0) col = face_colors[4];
            else if ((mask & 32) != 0) col = face_colors[5];
            else col = vec3(0.15);
            frag_color = vec4(col * shade, 1.0);
        } else {
            frag_color = vec4(vec3(1.0) * shade, 1.0);
        }
        return;
    }

    vec3 color = u_color * shade;
    frag_color = vec4(color, 1.0);
}
)";

// ============================ HELPERS =====================================

static uint32_t next_pow2(uint32_t x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}

struct SSBO {
    GLuint handle = 0;
    void create(const void* data, size_t bytes) {
        glCreateBuffers(1, &handle);
        glNamedBufferStorage(handle, bytes, data, GL_DYNAMIC_STORAGE_BIT);
    }
    void bind(int binding) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, handle);
    }
    ~SSBO() { if (handle) glDeleteBuffers(1, &handle); }
    SSBO() = default;
    SSBO(const SSBO&) = delete;
    SSBO& operator=(const SSBO&) = delete;
    SSBO(SSBO&& other) noexcept : handle(other.handle) { other.handle = 0; }
    SSBO& operator=(SSBO&& other) noexcept {
        if (this != &other) {
            if (handle) glDeleteBuffers(1, &handle);
            handle = other.handle; other.handle = 0;
        }
        return *this;
    }
};

static gl::Program* make_program(const char* vs_src, const char* fs_src) {
    gl::Shader vs(gl::ShaderType::vertex, vs_src);
    gl::Shader fs(gl::ShaderType::fragment, fs_src);
    if (!vs.compiled() || !fs.compiled()) return nullptr;
    auto* prog = new gl::Program;
    prog->attach(vs);
    prog->attach(fs);
    if (!prog->link()) { delete prog; return nullptr; }
    return prog;
}

static gl::Program* make_compute(const char* cs_src, const char* name) {
    gl::Shader cs(gl::ShaderType::compute, cs_src);
    if (!cs.compiled()) {
        gllib::logf(gllib::LogLevel::error, "Compute shader '%s' compile failed", name);
        return nullptr;
    }
    auto* prog = new gl::Program;
    prog->attach(cs);
    if (!prog->link()) { gllib::logf(gllib::LogLevel::error, "Compute shader '%s' link failed", name); delete prog; return nullptr; }
    return prog;
}

// ======================= CAMERA CONTROLS ==================================

static void fps_control(gfx::Window& window, gfx::Camera& cam,
                         float dt, float& yaw, float& pitch, bool& captured) {
    static double prev_x = 0, prev_y = 0;
    static bool first = true;
    float speed = 3.0f * dt;
    float sens = 0.002f;

    if (window.mouse_down(gfx::MouseButton::right)) {
        if (!captured) {
            glfwSetInputMode((GLFWwindow*)window.native_handle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            captured = true;
            first = true;
        }
    } else if (captured) {
        glfwSetInputMode((GLFWwindow*)window.native_handle(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        captured = false;
    }

    if (captured) {
        double cx, cy;
        window.cursor_position(cx, cy);
        if (first) { prev_x = cx; prev_y = cy; first = false; }
        yaw -= float(cx - prev_x) * sens;
        pitch = glm::clamp(pitch + float(prev_y - cy) * sens, -1.5f, 1.5f);
        prev_x = cx;
        prev_y = cy;
    }

    float cy = std::cos(yaw), sy = std::sin(yaw);
    glm::vec3 fwd(sy, 0, cy), rgt(cy, 0, -sy);
    glm::vec3 vel(0);
    if (window.key_down(gfx::Key::w)) vel += fwd;
    if (window.key_down(gfx::Key::s)) vel -= fwd;
    if (window.key_down(gfx::Key::a)) vel += rgt;
    if (window.key_down(gfx::Key::d)) vel -= rgt;
    if (window.key_down(gfx::Key::space)) vel.y += 1;
    if (window.key_down(gfx::Key::shift)) vel.y -= 1;
    if (glm::length(vel) > 0.0f) vel = glm::normalize(vel) * speed;

    glm::vec3 new_pos = cam.position() + vel;
    glm::vec3 dir(std::cos(pitch) * sy, std::sin(pitch), std::cos(pitch) * cy);
    cam.look_at(new_pos, new_pos + dir);
}

// ===================== CPU GREEDY RELABEL (optional, uses readback) =======

struct AdjEdge {
    uint32_t tri_a, tri_b;
};

static void greedy_relabel(std::vector<GPUTri>& tris,
                           const std::vector<AdjEdge>& edges,
                           float cos_epsilon,
                           float normal_threshold) {
    size_t n = tris.size();
    const glm::vec3 axes[6] = {
        { 1, 0, 0}, {-1, 0, 0},
        { 0, 1, 0}, { 0,-1, 0},
        { 0, 0, 1}, { 0, 0,-1}
    };

    std::vector<int> degree(n, 0);
    for (auto& e : edges) { degree[e.tri_a]++; degree[e.tri_b]++; }
    std::vector<uint32_t> adj_off(n + 1, 0);
    for (size_t i = 0; i < n; ++i) adj_off[i + 1] = adj_off[i] + degree[i];
    std::vector<uint32_t> adj_nb(adj_off[n]);
    {
        std::vector<uint32_t> cur = adj_off;
        for (auto& e : edges) {
            adj_nb[cur[e.tri_a]++] = e.tri_b;
            adj_nb[cur[e.tri_b]++] = e.tri_a;
        }
    }

    // Single-pass patching helper
    auto compute_patches = [&](const std::vector<GPUTri>& t) {
        size_t tn = t.size();
        std::vector<int32_t> parent(tn, -1);
        std::vector<int32_t> rank(tn, 0);
        for (size_t i = 0; i < tn; ++i)
            if (t[i].face_mask != 0) parent[i] = int32_t(i);

        auto find = [&](int32_t x) {
            int32_t r = x;
            while (parent[r] != r) r = parent[r];
            while (x != r) { int32_t tmp = parent[x]; parent[x] = r; x = tmp; }
            return r;
        };

        for (auto& e : edges) {
            if (parent[e.tri_a] >= 0 && parent[e.tri_b] >= 0) {
                if (t[e.tri_a].face_mask != t[e.tri_b].face_mask) continue;
                if (normal_threshold > 0.0f) {
                    glm::vec3 na(t[e.tri_a].nx, t[e.tri_a].ny, t[e.tri_a].nz);
                    glm::vec3 nb(t[e.tri_b].nx, t[e.tri_b].ny, t[e.tri_b].nz);
                    if (glm::dot(na, nb) < normal_threshold) continue;
                }
                int32_t ra = find(int32_t(e.tri_a));
                int32_t rb = find(int32_t(e.tri_b));
                if (ra != rb) {
                    if (rank[ra] < rank[rb]) parent[ra] = rb;
                    else if (rank[ra] > rank[rb]) parent[rb] = ra;
                    else { parent[rb] = ra; rank[ra]++; }
                }
            }
        }

        for (size_t i = 0; i < tn; ++i)
            if (parent[i] >= 0) parent[i] = find(int32_t(i));

        std::vector<int32_t> id_of_root(tn, -1);
        int32_t next = 0;
        std::vector<int32_t> ids(tn, -1);
        for (size_t i = 0; i < tn; ++i) {
            if (parent[i] >= 0) {
                if (id_of_root[parent[i]] < 0) id_of_root[parent[i]] = next++;
                ids[i] = id_of_root[parent[i]];
            }
        }
        return ids;
    };

    for (int iter = 0; iter < 5; ++iter) {
        auto ids = compute_patches(tris);
        int max_id = -1;
        for (size_t i = 0; i < n; ++i) if (ids[i] > max_id) max_id = ids[i];
        int np = max_id + 1;
        std::vector<int> sz(np, 0);
        for (size_t i = 0; i < n; ++i) if (ids[i] >= 0) sz[ids[i]]++;

        int flips = 0;
        std::vector<int> new_masks(n);
        for (size_t i = 0; i < n; ++i) new_masks[i] = tris[i].face_mask;

        for (size_t i = 0; i < n; ++i) {
            if (tris[i].face_mask == 0 || ids[i] < 0) continue;
            glm::vec3 nrm(tris[i].nx, tris[i].ny, tris[i].nz);
            float best = -1.0f, second = -1.0f;
            int best_a = -1, second_a = -1;
            for (int a = 0; a < 6; ++a) {
                float d = glm::dot(nrm, axes[a]);
                if (d > best) { second = best; second_a = best_a; best = d; best_a = a; }
                else if (d > second) { second = d; second_a = a; }
            }
            if (best - second >= cos_epsilon) continue;
            if (second_a < 0) continue;
            int second_mask = 1 << second_a;
            int cur_patch = ids[i];
            int cur_size = sz[cur_patch];
            for (uint32_t j = adj_off[i]; j < adj_off[i + 1]; ++j) {
                uint32_t nb = adj_nb[j];
                if (tris[nb].face_mask == second_mask && ids[nb] >= 0) {
                    if (sz[ids[nb]] > cur_size) {
                        new_masks[i] = second_mask;
                        flips++;
                        break;
                    }
                }
            }
        }
        if (flips == 0) break;
        for (size_t i = 0; i < n; ++i) tris[i].face_mask = new_masks[i];
    }

    // Orphan adoption
    for (int cleanup = 0; cleanup < 3; ++cleanup) {
        auto ids = compute_patches(tris);
        int max_id = -1;
        for (size_t i = 0; i < n; ++i) if (ids[i] > max_id) max_id = ids[i];
        int np = max_id + 1;
        std::vector<int> sz(np, 0);
        for (size_t i = 0; i < n; ++i) if (ids[i] >= 0) sz[ids[i]]++;
        const int MIN_PATCH = 3;
        int flips = 0;
        std::vector<int> new_masks(n);
        for (size_t i = 0; i < n; ++i) new_masks[i] = tris[i].face_mask;
        for (size_t i = 0; i < n; ++i) {
            if (tris[i].face_mask == 0 || ids[i] < 0) continue;
            int cur_size = sz[ids[i]];
            if (cur_size > MIN_PATCH) continue;
            glm::vec3 nrm(tris[i].nx, tris[i].ny, tris[i].nz);
            int best_nb_mask = 0;
            int best_nb_size = 0;
            for (uint32_t j = adj_off[i]; j < adj_off[i + 1]; ++j) {
                uint32_t nb = adj_nb[j];
                if (ids[nb] < 0) continue;
                int nb_size = sz[ids[nb]];
                if (nb_size <= MIN_PATCH) continue;
                int nb_mask = tris[nb].face_mask;
                float d = 0.0f;
                if      (nb_mask == 1)  d = nrm.x;
                else if (nb_mask == 2)  d = -nrm.x;
                else if (nb_mask == 4)  d = nrm.y;
                else if (nb_mask == 8)  d = -nrm.y;
                else if (nb_mask == 16) d = nrm.z;
                else if (nb_mask == 32) d = -nrm.z;
                if (d <= 0.0f) continue;
                if (nb_size > best_nb_size) { best_nb_size = nb_size; best_nb_mask = nb_mask; }
            }
            if (best_nb_mask != 0 && best_nb_mask != tris[i].face_mask) {
                new_masks[i] = best_nb_mask; flips++;
            }
        }
        if (flips == 0) break;
        for (size_t i = 0; i < n; ++i) tris[i].face_mask = new_masks[i];
    }
}

struct BVHNodeRaw {
    float min_x, min_y, min_z;
    int   child_base;
    float max_x, max_y, max_z;
    int   meta;
    int   l, r;
    int   pad0, pad1;
};
static_assert(sizeof(BVHNodeRaw) == 48, "BVHNodeRaw must be 48 bytes");

// ====================================================================
//  main
// ====================================================================

int main() {
    gllib::log_to_stderr(gllib::LogLevel::info);
    gfx::Window window({"27 MDC — Mesh Decomposition Clustering", 1400, 900});
    window.vsync(false);
    const char* gpu_str = (const char*)glGetString(GL_RENDERER);
    gllib::logf(gllib::LogLevel::info, "GPU: %s", gpu_str ? gpu_str : "unknown");

    gfx::ImGuiOverlay gui;
    if (!gui.init(window)) {
        gllib::log(gllib::LogLevel::error, "ImGui init failed");
        return EXIT_FAILURE;
    }

    // --- Load model ---
    gfx::Model model;
    bool loaded = model.load("Stanford_Dragon.glb");
    if (!loaded) loaded = model.load("Stanford_Bunny.glb");
    if (!loaded) {
        gllib::log(gllib::LogLevel::error, "Failed to load model");
        return EXIT_FAILURE;
    }
    gllib::logf(gllib::LogLevel::info, "Loaded model: %zu meshes", model.mesh_count());

    const gfx::Mesh& first_mesh = model.mesh(0);
    size_t tri_count = first_mesh.index_count() / 3;
    size_t vert_count = first_mesh.vertex_count();
    gllib::logf(gllib::LogLevel::info, "Mesh: %zu triangles, %zu vertices", tri_count, vert_count);

    // --- Bind mesh VBO/EBO as SSBOs (read-only for compute) ---
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, first_mesh.vbo_handle());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, first_mesh.ebo_handle());

    // --- Create TriBuf SSBO (binding 1) ---
    SSBO tri_ssbo;
    tri_ssbo.create(nullptr, tri_count * sizeof(GPUTri));
    tri_ssbo.bind(1);

    // --- Compute adjacency once: EdgeKeyBuf (bind 3), PairBuf (bind 4), ScalarBuf (bind 5) ---
    uint32_t edge_keys_per_tri = 3;
    uint32_t raw_edge_count = uint32_t(tri_count * edge_keys_per_tri);
    uint32_t padded = next_pow2(raw_edge_count);

    // EdgeKeyBuf: single uint64_t array padded to pow2, sentinels = UINT64_MAX
    std::vector<uint64_t> init_keys(padded, UINT64_MAX);
    SSBO edge_key_buf;
    edge_key_buf.create(init_keys.data(), init_keys.size() * sizeof(uint64_t));
    edge_key_buf.bind(3);

    // EdgePairBuf: enough space for all possible edge pairs
    uint32_t max_pairs = uint32_t(tri_count * 2);
    std::vector<uint64_t> empty_pairs(max_pairs, 0ULL);
    SSBO edge_pair_buf;
    edge_pair_buf.create(empty_pairs.data(), empty_pairs.size() * sizeof(uint64_t));
    edge_pair_buf.bind(4);

    // ScalarBuf: atomic counters
    Scalars init_scalars{};
    SSBO scalar_buf;
    scalar_buf.create(&init_scalars, sizeof(Scalars));
    scalar_buf.bind(5);

    // RootId / parent buffer (binding 6) — sized for tri_count ints
    SSBO parent_buf;
    parent_buf.create(nullptr, tri_count * sizeof(int32_t));
    parent_buf.bind(6);

    // AdjCSR buffers: OffBuf (binding 7), NbrBuf (binding 8), SizeBuf (binding 9)
    SSBO csr_off_buf;
    csr_off_buf.create(nullptr, (tri_count + 1) * sizeof(uint32_t));
    csr_off_buf.bind(7);

    SSBO csr_nbr_buf;
    csr_nbr_buf.create(nullptr, 2 * max_pairs * sizeof(uint32_t));
    csr_nbr_buf.bind(8);

    SSBO size_buf;
    size_buf.create(nullptr, tri_count * sizeof(uint32_t));
    size_buf.bind(9);

    // Temporary cursor buffer for AdjCSR scatter (same size as csr_off)
    SSBO cursor_buf;
    cursor_buf.create(nullptr, (tri_count + 1) * sizeof(uint32_t));

    // PatchMeta SSBO (binding 10) — created/destroyed dynamically
    SSBO patch_meta_ssbo;
    // Per-patch AABB SSBO (binding 11) — filled by cs_patch_aabb, read back to CPU
    SSBO patch_aabb_ssbo;
    // Per-patch float AABB SSBO (binding 13) — for BVH build
    SSBO patch_aabb_float_ssbo;
    // BVH SSBOs
    SSBO bvh_counters_ssbo;   // binding 12: counters[6] + offsets[6] + cursors[6]
    SSBO bvh_flat_ssbo;       // binding 14: uint64_t (morton<<32|patch_id) per patch, per-bin sorted
    SSBO bvh_nodes_ssbo;      // binding 15: BVHNode array (max 2*patch_count across all bins)
    SSBO bvh_roots_ssbo;      // binding 16: 6 int32_t root node indices
    SSBO bvh_work_ssbo;       // binding 17: work list input (count + items)
    SSBO bvh_next_ssbo;       // binding 18: work list output (count + items)
    SSBO bvh_node_counter_ssbo; // binding 19: next free node index (starts at 1, node 0 = root)

    // Atlas texture handles — dynamically created when patches are rasterized
    GLuint height_atlas_tex = 0;
    GLuint edge_atlas_tex = 0;
    GLuint diag_atlas_tex = 0;
    GLuint uv_atlas_tex = 0;

    // --- Compile compute shaders ---
    gl::Program* prog_normals   = make_compute(cs_normals_src, "cs_normals_src");
    gl::Program* prog_emit      = make_compute(cs_emit_edges_src, "cs_emit_edges_src");
    gl::Program* prog_sort      = make_compute(cs_sort_src, "cs_sort_src");
    gl::Program* prog_sort_flat = make_compute(cs_sort_flat_src, "cs_sort_flat_src");
    gl::Program* prog_sweep     = make_compute(cs_sweep_src, "cs_sweep_src");
    gl::Program* prog_uf_init   = make_compute(cs_uf_init_src, "cs_uf_init_src");
    gl::Program* prog_uf_hook   = make_compute(cs_uf_hook_src, "cs_uf_hook_src");
    gl::Program* prog_compress  = make_compute(cs_uf_compress_src, "cs_uf_compress_src");
    gl::Program* prog_compact_a = make_compute(cs_compact_a_src, "cs_compact_a_src");
    gl::Program* prog_compact_b = make_compute(cs_compact_b_src, "cs_compact_b_src");

    gl::Program* prog_adj_degree  = make_compute(cs_adj_degree_src, "cs_adj_degree_src");
    gl::Program* prog_adj_scatter = make_compute(cs_adj_scatter_src, "cs_adj_scatter_src");
    gl::Program* prog_relabel     = make_compute(cs_relabel_src, "cs_relabel_src");
    gl::Program* prog_orphan      = make_compute(cs_orphan_src, "cs_orphan_src");
    gl::Program* prog_rasterize   = make_compute(cs_rasterize_src, "cs_rasterize_src");
    gl::Program* prog_patch_aabb  = make_compute(cs_patch_aabb_src, "cs_patch_aabb_src");
    gl::Program* prog_bvh_count   = make_compute(cs_bvh_count_src, "cs_bvh_count_src");
    gl::Program* prog_bvh_scatter = make_compute(cs_bvh_scatter_src, "cs_bvh_scatter_src");
    gl::Program* prog_bvh_build   = make_compute(cs_bvh_build_src, "cs_bvh_build_src");
    gl::Program* prog_bvh_aabb    = make_compute(cs_bvh_aabb_src, "cs_bvh_aabb_src");
    gl::Program* prog_pick        = make_compute(cs_pick_src, "cs_pick_src");
    gl::Program* prog_recon       = make_compute(cs_recon_src, "cs_recon_src");
    gl::Program* prog_compare     = make_compute(cs_compare_src, "cs_compare_src");
    gl::Program* prog_diag        = make_compute(cs_diag_src, "cs_diag_src");

    if (!prog_normals || !prog_emit || !prog_sort || !prog_sort_flat || !prog_sweep ||
        !prog_uf_init || !prog_uf_hook || !prog_compress ||
        !prog_compact_a || !prog_compact_b ||
        !prog_adj_degree || !prog_adj_scatter ||
        !prog_relabel || !prog_orphan || !prog_rasterize || !prog_patch_aabb ||
        !prog_bvh_count || !prog_bvh_scatter || !prog_bvh_build || !prog_bvh_aabb ||
        !prog_pick || !prog_recon || !prog_compare || !prog_diag) {
        gllib::log(gllib::LogLevel::error, "Compute shader compilation failed");
        return EXIT_FAILURE;
    }

    // Pick result SSBO (binding 20)
    SSBO pick_result_ssbo;
    {
        int init[5] = {-1, 0, 0, 0, 0};
        pick_result_ssbo.create(init, sizeof(init));
        pick_result_ssbo.bind(20);
    }

    // Recon (debug accuracy view) resources — dynamically sized to match window
    int recon_w = 1, recon_h = 1;
    GLuint recon_pos_img = 0, recon_normal_img = 0, recon_truth_tex = 0;
    GLuint recon_fbo = 0, recon_depth_tex = 0, recon_display_tex = 0;
    GLuint depth_copy_tex = 0, depth_copy_fbo = 0;
    SSBO recon_accuracy_ssbo;
    {
        int acc_init[4] = {0, 0, 0, 0};
        recon_accuracy_ssbo.create(acc_init, sizeof(acc_init));
        recon_accuracy_ssbo.bind(30);
    }

    auto recreate_recon = [&](int w, int h) {
        if (recon_pos_img)          glDeleteTextures(1, &recon_pos_img);
        if (recon_normal_img)       glDeleteTextures(1, &recon_normal_img);
        if (recon_truth_tex)        glDeleteTextures(1, &recon_truth_tex);
        if (recon_display_tex)      glDeleteTextures(1, &recon_display_tex);
        if (recon_fbo)              glDeleteFramebuffers(1, &recon_fbo);
        if (recon_depth_tex)        glDeleteTextures(1, &recon_depth_tex);
        recon_w = w > 0 ? std::max(1, w/2) : 1; recon_h = h > 0 ? std::max(1, h/2) : 1;

        glCreateTextures(GL_TEXTURE_2D, 1, &recon_pos_img);
        glTextureStorage2D(recon_pos_img, 1, GL_RGBA32F, recon_w, recon_h);
        glTextureParameteri(recon_pos_img, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(recon_pos_img, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glCreateTextures(GL_TEXTURE_2D, 1, &recon_normal_img);
        glTextureStorage2D(recon_normal_img, 1, GL_RGBA16F, recon_w, recon_h);
        glTextureParameteri(recon_normal_img, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(recon_normal_img, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glCreateTextures(GL_TEXTURE_2D, 1, &recon_truth_tex);
        glTextureStorage2D(recon_truth_tex, 1, GL_RGBA16F, recon_w, recon_h);
        glTextureParameteri(recon_truth_tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(recon_truth_tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glCreateTextures(GL_TEXTURE_2D, 1, &recon_display_tex);
        glTextureStorage2D(recon_display_tex, 1, GL_SRGB8_ALPHA8, recon_w, recon_h);
        glTextureParameteri(recon_display_tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(recon_display_tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glCreateTextures(GL_TEXTURE_2D, 1, &recon_depth_tex);
        glTextureStorage2D(recon_depth_tex, 1, GL_DEPTH_COMPONENT24, recon_w, recon_h);
        glTextureParameteri(recon_depth_tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(recon_depth_tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureParameteri(recon_depth_tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(recon_depth_tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glCreateFramebuffers(1, &recon_fbo);
        glNamedFramebufferTexture(recon_fbo, GL_COLOR_ATTACHMENT0, recon_truth_tex, 0);
        glNamedFramebufferTexture(recon_fbo, GL_DEPTH_ATTACHMENT, recon_depth_tex, 0);
    };

    recreate_recon(window.width(), window.height());

    auto recreate_depth_copy = [&](int w, int h) {
        if (depth_copy_tex) glDeleteTextures(1, &depth_copy_tex);
        if (depth_copy_fbo) glDeleteFramebuffers(1, &depth_copy_fbo);
        w = w > 0 ? w : 1; h = h > 0 ? h : 1;
        glCreateTextures(GL_TEXTURE_2D, 1, &depth_copy_tex);
        glTextureStorage2D(depth_copy_tex, 1, GL_DEPTH_COMPONENT24, w, h);
        glTextureParameteri(depth_copy_tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(depth_copy_tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(depth_copy_tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(depth_copy_tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glCreateFramebuffers(1, &depth_copy_fbo);
        glNamedFramebufferTexture(depth_copy_fbo, GL_DEPTH_ATTACHMENT, depth_copy_tex, 0);
        glNamedFramebufferDrawBuffer(depth_copy_fbo, GL_NONE);
        glNamedFramebufferReadBuffer(depth_copy_fbo, GL_NONE);
    };
    recreate_depth_copy(window.width(), window.height());

    // --- Render shader ---
    gl::Program* render_prog = make_program(pbr_vs, simple_fs);
    if (!render_prog) {
        gllib::log(gllib::LogLevel::error, "Render shader compilation failed");
        return EXIT_FAILURE;
    }

    // ===== INITIAL COMPUTE PASSES =====

    auto dispatch_normals = [&]() {
        prog_normals->use();
        GLint loc = prog_normals->uniform_location("u_tri_count");
        if (loc >= 0) prog_normals->uniform1ui(loc, uint32_t(tri_count));
        gl::dispatch_compute(uint32_t((tri_count + 255) / 256), 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };

    auto dispatch_edges = [&]() {
        prog_emit->use();
        GLint loc = prog_emit->uniform_location("u_tri_count");
        if (loc >= 0) prog_emit->uniform1ui(loc, uint32_t(tri_count));
        gl::dispatch_compute(uint32_t((tri_count + 255) / 256), 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };

    auto dispatch_sort = [&]() {
        prog_sort->use();
        GLint u_j = prog_sort->uniform_location("u_j");
        GLint u_k = prog_sort->uniform_location("u_k");
        GLint u_N = prog_sort->uniform_location("u_N");
        if (u_N >= 0) prog_sort->uniform1ui(u_N, padded);
        uint32_t groups = (padded + 255) / 256;
        for (uint32_t k = 2; k <= padded; k <<= 1) {
            for (uint32_t j = k >> 1; j > 0; j >>= 1) {
                if (u_j >= 0) prog_sort->uniform1ui(u_j, j);
                if (u_k >= 0) prog_sort->uniform1ui(u_k, k);
                gl::dispatch_compute(groups, 1, 1);
                gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }
        }
    };

    auto dispatch_sweep = [&]() {
        // Reset edge_count before sweep
        uint32_t zero_val = 0;
        glNamedBufferSubData(scalar_buf.handle, offsetof(Scalars, edge_count), 4, &zero_val);

        prog_sweep->use();
        GLint loc = prog_sweep->uniform_location("u_padded");
        if (loc >= 0) prog_sweep->uniform1ui(loc, padded);
        uint32_t groups = (padded + 255) / 256;
        gl::dispatch_compute(groups, 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };

    // Core union-find (no compaction; compress also populates patch_size for relabel)
    auto dispatch_patches_core = [&](float nt, int hook_iters) {
        uint32_t tn = uint32_t(tri_count);
        uint32_t groups = uint32_t((tri_count + 255) / 256);
        uint32_t zero_val = 0;

        glClearNamedBufferData(size_buf.handle, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero_val);

        prog_uf_init->use();
        GLint loc = prog_uf_init->uniform_location("u_tri_count");
        if (loc >= 0) prog_uf_init->uniform1ui(loc, tn);
        gl::dispatch_compute(groups, 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        prog_uf_hook->use();
        loc = prog_uf_hook->uniform_location("u_pair_max");
        if (loc >= 0) prog_uf_hook->uniform1ui(loc, max_pairs);
        loc = prog_uf_hook->uniform_location("u_normal_threshold");
        if (loc >= 0) prog_uf_hook->uniform1f(loc, nt);
        uint32_t hook_groups = (max_pairs + 255) / 256;
        for (int iter = 0; iter < hook_iters; ++iter) {
            gl::dispatch_compute(hook_groups, 1, 1);
            gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        prog_compress->use();
        loc = prog_compress->uniform_location("u_tri_count");
        if (loc >= 0) prog_compress->uniform1ui(loc, tn);
        gl::dispatch_compute(groups, 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };

    // Full variant including compaction (only needed once at the end)
    auto dispatch_patches_full = [&](float nt, int hook_iters) {
        dispatch_patches_core(nt, hook_iters);

        uint32_t tn = uint32_t(tri_count);
        uint32_t groups = uint32_t((tri_count + 255) / 256);
        uint32_t zero_val = 0;
        glNamedBufferSubData(scalar_buf.handle, offsetof(Scalars, patch_count), 4, &zero_val);

        prog_compact_a->use();
        GLint loc = prog_compact_a->uniform_location("u_tri_count");
        if (loc >= 0) prog_compact_a->uniform1ui(loc, tn);
        gl::dispatch_compute(groups, 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        prog_compact_b->use();
        loc = prog_compact_b->uniform_location("u_tri_count");
        if (loc >= 0) prog_compact_b->uniform1ui(loc, tn);
        gl::dispatch_compute(groups, 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };

    // --- Run initial compute passes: normals/centers, then adjacency ---
    dispatch_normals();
    dispatch_edges();
    dispatch_sort();
    dispatch_sweep();

    // Edge adjacency count for info
    uint32_t edge_pair_count = 0;
    glGetNamedBufferSubData(scalar_buf.handle, offsetof(Scalars, edge_count), 4, &edge_pair_count);
    gllib::logf(gllib::LogLevel::info, "Mesh: %u adjacency edges (GPU built)", edge_pair_count);

    // --- Build AdjCSR (one-time, from EdgePairBuf) ---
    if (edge_pair_count > 0) {
        // 1. Degree pass: zero-init csr_off_buf (binding 7), count per-triangle degree
        uint32_t zero_deg = 0;
        glClearNamedBufferData(csr_off_buf.handle, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero_deg);
        prog_adj_degree->use();
        GLint deg_loc = prog_adj_degree->uniform_location("u_edge_count");
        if (deg_loc >= 0) prog_adj_degree->uniform1ui(deg_loc, edge_pair_count);
        gl::dispatch_compute((edge_pair_count + 255) / 256, 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // 2. CPU prefix-sum degree → offsets (tri_count+1 elements, ~1MB — one-time cost)
        std::vector<uint32_t> offsets(tri_count + 1, 0);
        glGetNamedBufferSubData(csr_off_buf.handle, 0, tri_count * sizeof(uint32_t), offsets.data());
        for (size_t i = 0; i < tri_count; ++i) offsets[i + 1] += offsets[i];
        glNamedBufferSubData(csr_off_buf.handle, 0, (tri_count + 1) * sizeof(uint32_t), offsets.data());

        // 3. Upload cursors for scatter (same data as offsets, mutated by scatter)
        glNamedBufferSubData(cursor_buf.handle, 0, (tri_count + 1) * sizeof(uint32_t), offsets.data());

        // 4. Scatter neighbors: cursor_buf on binding 7, csr_nbr_buf on binding 8
        cursor_buf.bind(7);
        prog_adj_scatter->use();
        GLint sca_loc = prog_adj_scatter->uniform_location("u_edge_count");
        if (sca_loc >= 0) prog_adj_scatter->uniform1ui(sca_loc, edge_pair_count);
        gl::dispatch_compute((edge_pair_count + 255) / 256, 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // 5. Restore csr_off_buf on binding 7
        csr_off_buf.bind(7);
    }

    // --- Camera ---
    gfx::Camera cam;
    cam.perspective(45.0f, float(window.width()) / window.height(), 0.1f, 100.0f);
    cam.look_at({3, 1.5f, -2}, {0, 0, 0});

    // Load vertex/index data for AABB/center + rasterization
    std::vector<gfx::Vertex> mesh_verts(first_mesh.vertex_count());
    glGetNamedBufferSubData(first_mesh.vbo_handle(), 0,
                            mesh_verts.size() * sizeof(gfx::Vertex), mesh_verts.data());
    std::vector<uint32_t> mesh_idx(first_mesh.index_count());
    glGetNamedBufferSubData(first_mesh.ebo_handle(), 0,
                            mesh_idx.size() * sizeof(uint32_t), mesh_idx.data());

    glm::vec3 center(0);
    glm::vec3 mesh_aabb_min( FLT_MAX);
    glm::vec3 mesh_aabb_max(-FLT_MAX);
    for (auto& v : mesh_verts) {
        glm::vec3 pos(v.position[0], v.position[1], v.position[2]);
        center += pos;
        mesh_aabb_min = glm::min(mesh_aabb_min, pos);
        mesh_aabb_max = glm::max(mesh_aabb_max, pos);
    }
    center /= float(mesh_verts.size());

    glm::vec3 light_dir = glm::normalize(glm::vec3(1, -1.5f, 1));

    float yaw = 0, pitch = 0;
    bool captured = false;
    int debug_mode = 0;
    bool show_bvh_debug[6] = {};
    float compute_ms = 0.0f;
    float bvh_build_ms = 0.0f;
    std::vector<uint8_t> bvh_debug_nodes;
    std::vector<int> bvh_per_bin_depth(6, 0);
    std::vector<uint32_t> bvh_debug_bin_counts(6, 0);
    std::vector<uint32_t> bvh_debug_bin_offsets(6, 0);
    int bvh_debug_total_nodes = 0;
    int highlight_patch = -1;
    float texel_density = 0.0f;
    glm::ivec2 atlas_size = {0, 0};
    glm::vec3 picked_position;
    std::vector<glm::vec3> patch_aabb_min, patch_aabb_max;
    float model_rot_x = 90.0f;
    int face_filter = 0;
    int patch_count = 0;
    float normal_threshold = 0.0f;
    float greedy_epsilon = 0.5f;
    int relabel_iters = 3;
    int orphan_iters = 1;
    int hook_iters = 6;
    static const char* face_names[] = { "All", "+X", "-X", "+Y", "-Y", "+Z", "-Z" };

    // Rasterization controls
    bool rasterize_enabled = true;
    int atlas_target_size = 1024;
    int atlas_max_width = 8192;
    int atlas_texel_count = 0;
    int atlas_current_w = 0;
    int atlas_current_h = 0;
    float rasterize_ms = 0.0f;
    float clear_ms = 0.0f, compute_ms_gpu = 0.0f, mip_ms = 0.0f, cpu_meta_ms = 0.0f;
    bool show_height_debug = false;
    GLuint debug_display_tex = 0;
    int dbg_tex_w = 0, dbg_tex_h = 0;
    std::vector<PatchMetaGPU> debug_patch_meta;

    GLuint gpu_query = 0, gpu_clear_q = 0, gpu_raster_q = 0, gpu_mip_q = 0;
    glGenQueries(1, &gpu_query);
    glGenQueries(1, &gpu_clear_q);
    glGenQueries(1, &gpu_raster_q);
    glGenQueries(1, &gpu_mip_q);

    gfx::Renderer renderer;
    renderer.set_clear_color(0.1f, 0.15f, 0.2f, 1.0f);
    gl::enable(GL_DEPTH_TEST);

    gfx::DebugDraw dd;
    gfx::GpuTimer bvh_timer;

    double last = window.time();

    while (!window.should_close()) {
        double now = window.time();
        float dt = float(now - last);
        last = now;

        window.poll_events();
        fps_control(window, cam, dt, yaw, pitch, captured);
        cam.set_aspect(float(window.width()) / window.height());

        // Recreate recon + depth-copy resources if window size changed
        {   int ww = window.width(), wh = window.height();
            if (ww != recon_w || wh != recon_h) {
                recreate_recon(ww, wh);
                recreate_depth_copy(ww, wh);
            }
        }

        glm::mat4 vp = cam.view_projection();

        glm::mat4 model_mat = glm::mat4(1.0f);
        model_mat = glm::translate(model_mat, -center);
        model_mat = glm::rotate(model_mat, glm::radians(model_rot_x), glm::vec3(1, 0, 0));
        model_mat = glm::scale(model_mat, glm::vec3(1.5f));

        // Ray picking — GPU Hi-Z ray march through height atlas
        {
            static bool prev_lmb = false;
            bool lmb = window.mouse_down(gfx::MouseButton::left);
            bool do_pick = false;
            double px = 0, py = 0;
            if (!captured && lmb && !prev_lmb) {
                window.cursor_position(px, py);
                do_pick = true;
            }
            prev_lmb = lmb;
            if (do_pick && patch_count > 0 && pick_result_ssbo.handle) {
                glm::mat4 inv_model = glm::inverse(model_mat);
                glm::ivec4 vp_win(0, 0, window.width(), window.height());
                auto unproject = [&](float win_z) -> glm::vec3 {
                    glm::vec3 win(float(px), float(window.height() - int(py)), win_z);
                    glm::mat4 inv_vp = glm::inverse(cam.view_projection());
                    glm::vec4 h = inv_vp * glm::vec4(
                        2.0f * win.x / float(vp_win.z) - 1.0f,
                        2.0f * win.y / float(vp_win.w) - 1.0f,
                        2.0f * win.z - 1.0f, 1.0f);
                    return glm::vec3(h) / h.w;
                };
                glm::vec3 ro_w = unproject(0.0f);
                glm::vec3 rd_w = glm::normalize(unproject(1.0f) - ro_w);
                glm::vec3 ro = glm::vec3(inv_model * glm::vec4(ro_w, 1.0f));
                glm::vec3 rd = glm::normalize(glm::vec3(inv_model * glm::vec4(rd_w, 0.0f)));

                glm::mat4 mvp = vp * model_mat;
                glm::ivec2 rsize(window.width(), window.height());

                if (bvh_flat_ssbo.handle) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, bvh_flat_ssbo.handle);
                if (bvh_nodes_ssbo.handle) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, bvh_nodes_ssbo.handle);
                if (bvh_roots_ssbo.handle) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 16, bvh_roots_ssbo.handle);

                prog_pick->use();
                auto _ploc = [&](const char* n) { return prog_pick->uniform_location(n); };
                GLint _pl;
                if ((_pl = _ploc("u_bin_offsets")) >= 0) {
                    GLint offsets_int[6];
                    for (int i = 0; i < 6; ++i) offsets_int[i] = (int)bvh_debug_bin_offsets[i];
                    glUniform1iv(_pl, 6, offsets_int);
                }

                if ((_pl = _ploc("u_ro")) >= 0) prog_pick->uniform3f(_pl, ro.x, ro.y, ro.z);
                if ((_pl = _ploc("u_rd")) >= 0) prog_pick->uniform3f(_pl, rd.x, rd.y, rd.z);
                if ((_pl = _ploc("u_patch_count")) >= 0) prog_pick->uniform1i(_pl, patch_count);
                if ((_pl = _ploc("u_texel_density")) >= 0) prog_pick->uniform1f(_pl, texel_density);
                if ((_pl = _ploc("u_height_sampler")) >= 0) prog_pick->uniform1i(_pl, 0);
                if ((_pl = _ploc("u_depth_tex")) >= 0) prog_pick->uniform1i(_pl, 1);
                if ((_pl = _ploc("u_mvp")) >= 0) prog_pick->uniform_matrix4fv(_pl, glm::value_ptr(mvp));
                if ((_pl = _ploc("u_view_size")) >= 0) prog_pick->uniform2iv(_pl, glm::value_ptr(rsize));
                glBindTextureUnit(0, height_atlas_tex);
                glBindTextureUnit(1, depth_copy_tex);
                gl::dispatch_compute(1, 1, 1);
                gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

                struct { int patch; float t; float pos[3]; } result;
                glGetNamedBufferSubData(pick_result_ssbo.handle, 0, sizeof(result), &result);
                highlight_patch = result.patch;
                if (result.patch >= 0) {
                    picked_position = glm::vec3(result.pos[0], result.pos[1], result.pos[2]);
                }
            }
        }

        // --- Render ---
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, window.width(), window.height());
        renderer.clear();

        render_prog->use();
        auto rloc = [&](const char* n) { return render_prog->uniform_location(n); };

        GLint loc = rloc("u_view_proj");  if (loc >= 0) render_prog->uniform_matrix4fv(loc, glm::value_ptr(vp));
        loc = rloc("u_model");       if (loc >= 0) render_prog->uniform_matrix4fv(loc, glm::value_ptr(model_mat));
        glm::mat3 nm = glm::transpose(glm::inverse(glm::mat3(model_mat)));
        loc = rloc("u_normal_mat");  if (loc >= 0) render_prog->uniform_matrix3fv(loc, glm::value_ptr(nm));
        loc = rloc("u_color");       if (loc >= 0) render_prog->uniform3f(loc, 0.6f, 0.55f, 0.5f);
        loc = rloc("u_light_dir");   if (loc >= 0) render_prog->uniform3fv(loc, glm::value_ptr(light_dir));
        loc = rloc("u_debug_mode");  if (loc >= 0) render_prog->uniform1i(loc, debug_mode);
        loc = rloc("u_face_filter"); if (loc >= 0) render_prog->uniform1i(loc, face_filter);
        loc = rloc("u_highlight_patch"); if (loc >= 0) render_prog->uniform1i(loc, highlight_patch);

        first_mesh.draw();

        // Copy depth buffer for next frame's pick culling
        {
            int ww = window.width(), wh = window.height();
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, depth_copy_fbo);
            glBlitFramebuffer(0, 0, ww, wh, 0, 0, ww, wh, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        }

        // --- Debug recon view (height-field accuracy) ---
        static bool show_recon = false;
        static bool recon_use_fallback = true;
        static float recon_accuracy = 0.0f, recon_coverage = 0.0f;
        static int recon_hit = 0, recon_match = 0;

        // --- ImGui ---
        gui.begin_frame();
        {
            ImGui::Begin("AABB Face Filter");
            ImGui::Text("Triangles: %zu", tri_count);
            ImGui::Text("Compute: %.3f ms", compute_ms);
            if (patch_count > 0)
                ImGui::Text("Patches: %d", patch_count);
            ImGui::Separator();
            ImGui::SliderFloat("Rotate X", &model_rot_x, -180.0f, 180.0f);
            ImGui::Separator();
            ImGui::Text("Face Filter");
            ImGui::Combo("Face", &face_filter, face_names, 7);
            ImGui::Separator();
            ImGui::Text("Debug Mode");
            ImGui::RadioButton("Flat", &debug_mode, 0);
            ImGui::RadioButton("Normals", &debug_mode, 1);
            ImGui::RadioButton("Faces", &debug_mode, 2);
            ImGui::RadioButton("Patches", &debug_mode, 3);
            ImGui::Separator();

            if (ImGui::Button("Re-assign Faces")) {
                auto t0 = std::chrono::steady_clock::now();
                dispatch_normals();
                auto t1 = std::chrono::steady_clock::now();
                compute_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
                patch_count = 0;
            }

            ImGui::Separator();
            ImGui::Text("Patches");
            ImGui::SliderFloat("Normal Threshold", &normal_threshold, 0.0f, 1.0f, "%.2f (%.0f deg)");
            if (normal_threshold > 0.0f)
                ImGui::Text("  Merging only edges with dot >= %.2f", normal_threshold);
            ImGui::SliderFloat("Greedy Epsilon", &greedy_epsilon, 0.0f, 0.5f, "%.3f");
            if (greedy_epsilon > 0.0f)
                ImGui::Text("  Flipping ambiguous tris where bestDot - secondDot < %.3f", greedy_epsilon);
            ImGui::SliderInt("Hook Iters", &hook_iters, 1, 20);
            ImGui::SliderInt("Relabel Iters", &relabel_iters, 0, 10);
            ImGui::SliderInt("Orphan Iters", &orphan_iters, 0, 5);

            ImGui::Separator();
            ImGui::Text("Patch Rasterization");
            ImGui::Checkbox("Rasterize", &rasterize_enabled);
            ImGui::SliderInt("Atlas Target Size", &atlas_target_size, 128, 4096);
            if (atlas_texel_count > 0) {
                ImGui::Text("Atlas texels: %d", atlas_texel_count);
                float mem_mb = atlas_texel_count * 10.0f / (1024.0f * 1024.0f);
                ImGui::Text("Atlas memory: %.2f MB", mem_mb);
                if (rasterize_ms > 0.0f) {
                    ImGui::Text("Rasterize: %.3f ms", rasterize_ms);
                    ImGui::Text("  clear: %.3f  compute: %.3f  mip: %.3f  cpu: %.3f",
                                clear_ms, compute_ms_gpu, mip_ms, cpu_meta_ms);
                }
            }

            if (ImGui::Button("Compute Patches")) {
                auto t0 = std::chrono::steady_clock::now();

                glBeginQuery(GL_TIME_ELAPSED, gpu_query);

                if (greedy_epsilon > 0.0f && edge_pair_count > 0 && relabel_iters > 0) {
                    for (int iter = 0; iter < relabel_iters; ++iter) {
                        dispatch_patches_core(normal_threshold, hook_iters);

                        prog_relabel->use();
                        GLint loc = prog_relabel->uniform_location("u_tri_count");
                        if (loc >= 0) prog_relabel->uniform1ui(loc, uint32_t(tri_count));
                        loc = prog_relabel->uniform_location("u_cos_epsilon");
                        if (loc >= 0) prog_relabel->uniform1f(loc, greedy_epsilon);
                        gl::dispatch_compute(uint32_t((tri_count + 255) / 256), 1, 1);
                        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    }

                    for (int cleanup = 0; cleanup < orphan_iters; ++cleanup) {
                        dispatch_patches_core(normal_threshold, hook_iters);

                        prog_orphan->use();
                        GLint loc = prog_orphan->uniform_location("u_tri_count");
                        if (loc >= 0) prog_orphan->uniform1ui(loc, uint32_t(tri_count));
                        gl::dispatch_compute(uint32_t((tri_count + 255) / 256), 1, 1);
                        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    }
                } else {
                    // epsilon=0 or no iterations: just one core pass
                    dispatch_patches_core(normal_threshold, hook_iters);
                }

                dispatch_patches_full(normal_threshold, hook_iters);

                glEndQuery(GL_TIME_ELAPSED);

                uint32_t pc = 0;
                glGetNamedBufferSubData(scalar_buf.handle,
                                        offsetof(Scalars, patch_count), 4, &pc);
                patch_count = int(pc);

                auto t1 = std::chrono::steady_clock::now();
                compute_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

                GLuint64 gpu_ns = 0;
                glGetQueryObjectui64v(gpu_query, GL_QUERY_RESULT, &gpu_ns);
                gllib::logf(gllib::LogLevel::info,
                            "Compute Patches → CPU: %.3f ms  GPU: %.3f ms",
                            compute_ms, gpu_ns / 1e6);

                // ---- Rasterize patches into height/UV atlas ----
                if (rasterize_enabled && patch_count > 0) {
                    auto rt0 = std::chrono::steady_clock::now();

                    // Per-patch AABB + height range via GPU compute
                    static const uint32_t SORT_INF  = 0xFF800000u; // sortable(+inf)
                    static const uint32_t SORT_NINF = 0x007FFFFFu; // sortable(-inf)
                    std::vector<uint32_t> aabb_init(patch_count * 9);
                    for (int i = 0; i < patch_count; ++i) {
                        auto* r = &aabb_init[i * 9];
                        r[0] = SORT_INF;   // min_x
                        r[1] = SORT_INF;   // min_y
                        r[2] = SORT_INF;   // min_z
                        r[3] = SORT_NINF;  // max_x
                        r[4] = SORT_NINF;  // max_y
                        r[5] = SORT_NINF;  // max_z
                        r[6] = SORT_INF;   // height_min
                        r[7] = SORT_NINF;  // height_max
                        r[8] = 0;          // dominant_axis (stored +1, 0 = unset)
                    }
                    if (patch_aabb_ssbo.handle)
                        glDeleteBuffers(1, &patch_aabb_ssbo.handle);
                    patch_aabb_ssbo.create(aabb_init.data(), aabb_init.size() * 4);
                    patch_aabb_ssbo.bind(11);

                    prog_patch_aabb->use();
                    GLint _aabb_loc;
                    if ((_aabb_loc = prog_patch_aabb->uniform_location("u_tri_count"))  >= 0)
                        prog_patch_aabb->uniform1ui(_aabb_loc, uint32_t(tri_count));
                    if ((_aabb_loc = prog_patch_aabb->uniform_location("u_patch_count")) >= 0)
                        prog_patch_aabb->uniform1i(_aabb_loc, patch_count);
                    gl::dispatch_compute(uint32_t((tri_count + 255) / 256), 1, 1);
                    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

                    // Read back per-patch results (~36KB, no stall)
                    std::vector<uint32_t> aabb_raw(patch_count * 9);
                    glGetNamedBufferSubData(patch_aabb_ssbo.handle, 0,
                                            aabb_raw.size() * 4, aabb_raw.data());

                    auto from_sortable = [](uint32_t s) -> float {
                        return (s & 0x80000000u) != 0
                            ? std::bit_cast<float>(s ^ 0x80000000u)
                            : std::bit_cast<float>(~s);
                    };

                    struct PatchMetaCPU {
                        glm::vec3 aabb_min, aabb_max;
                        int dominant_axis = -1;
                        float height_min, height_max;
                    };
                    std::vector<PatchMetaCPU> patch_cpu(patch_count);
                    for (int i = 0; i < patch_count; ++i) {
                        uint32_t* r = &aabb_raw[i * 9];
                        patch_cpu[i].aabb_min = glm::vec3(from_sortable(r[0]), from_sortable(r[1]), from_sortable(r[2]));
                        patch_cpu[i].aabb_max = glm::vec3(from_sortable(r[3]), from_sortable(r[4]), from_sortable(r[5]));
                        patch_cpu[i].height_min = from_sortable(r[6]);
                        patch_cpu[i].height_max = from_sortable(r[7]);
                        uint32_t mask = r[8];
                        int da = -1;
                        if (mask & 1u)  da = 0;
                        else if (mask & 2u)  da = 1;
                        else if (mask & 4u)  da = 2;
                        else if (mask & 8u)  da = 3;
                        else if (mask & 16u) da = 4;
                        else if (mask & 32u) da = 5;
                        patch_cpu[i].dominant_axis = da;
                    }
                    patch_aabb_min.resize(patch_count);
                    patch_aabb_max.resize(patch_count);
                    for (int i = 0; i < patch_count; ++i) {
                        patch_aabb_min[i] = patch_cpu[i].aabb_min;
                        patch_aabb_max[i] = patch_cpu[i].aabb_max;
                    }
                    highlight_patch = -1;

                    // Upload float AABBs to binding 13 for scatter shader
                    {
                        std::vector<float> aabb_float(patch_count * 6);
                        for (int i = 0; i < patch_count; ++i) {
                            auto& pc = patch_cpu[i];
                            aabb_float[i * 6 + 0] = pc.aabb_min.x;
                            aabb_float[i * 6 + 1] = pc.aabb_min.y;
                            aabb_float[i * 6 + 2] = pc.aabb_min.z;
                            aabb_float[i * 6 + 3] = pc.aabb_max.x;
                            aabb_float[i * 6 + 4] = pc.aabb_max.y;
                            aabb_float[i * 6 + 5] = pc.aabb_max.z;
                        }
                        if (patch_aabb_float_ssbo.handle)
                            glDeleteBuffers(1, &patch_aabb_float_ssbo.handle);
                        patch_aabb_float_ssbo.create(aabb_float.data(), aabb_float.size() * sizeof(float));
                        patch_aabb_float_ssbo.bind(13);
                    }

                    glm::vec3 mesh_extent = mesh_aabb_max - mesh_aabb_min;
                    float max_extent = glm::max(mesh_extent.x, glm::max(mesh_extent.y, mesh_extent.z));
                    texel_density = float(atlas_target_size) / max_extent;

                    auto project_2d = [](glm::vec3 p, int axis) -> glm::vec2 {
                        if (axis == 0 || axis == 1) return glm::vec2(p.y, p.z);
                        if (axis == 2 || axis == 3) return glm::vec2(p.x, p.z);
                        return glm::vec2(p.x, p.y);
                    };

                    struct PatchLayout {
                        int w, h;
                        float ox, oy;
                    };
                    std::vector<PatchLayout> layouts(patch_count);
                    for (int i = 0; i < patch_count; ++i) {
                        auto& pc = patch_cpu[i];
                        if (pc.dominant_axis < 0) continue;
                        glm::vec2 pmin = project_2d(pc.aabb_min, pc.dominant_axis);
                        glm::vec2 pmax = project_2d(pc.aabb_max, pc.dominant_axis);
                        glm::vec2 ext = pmax - pmin;
                        layouts[i].ox = pmin.x;
                        layouts[i].oy = pmin.y;
                        layouts[i].w = std::max(1, int(ceil(ext.x * texel_density)));
                        layouts[i].h = std::max(1, int(ceil(ext.y * texel_density)));
                    }

                    // Try multiple atlas widths and pick the one with minimal total texels
                    int max_patch_w = 0;
                    for (auto& pl : layouts) max_patch_w = std::max(max_patch_w, pl.w);
                    int min_width = std::max(256, int(next_pow2(max_patch_w)));
                    std::vector<int> width_candidates;
                    for (int w = min_width; w <= atlas_max_width; w *= 2)
                        width_candidates.push_back(w);

                    struct PackTrial {
                        int width, height, total_texels, used_texels;
                        std::vector<PatchMetaGPU> meta;
                    };
                    std::vector<PackTrial> trials;

                    for (int trial_width : width_candidates) {
                        std::vector<int> order(patch_count);
                        std::iota(order.begin(), order.end(), 0);
                        std::sort(order.begin(), order.end(), [&](int a, int b) {
                            if (layouts[a].h != layouts[b].h)
                                return layouts[a].h > layouts[b].h;
                            return layouts[a].w > layouts[b].w;
                        });

                        int cur_x = 0, cur_y = 0, row_h = 0;
                        int texel_sum = 0;
                        std::vector<PatchMetaGPU> meta(patch_count);

                        for (int idx : order) {
                            auto& pm = meta[idx];
                            auto& pc = patch_cpu[idx];
                            auto& pl = layouts[idx];

                            pm.dominant_axis = pc.dominant_axis;
                            pm.proj_origin_x = pl.ox;
                            pm.proj_origin_y = pl.oy;
                            pm.atlas_w = pl.w;
                            pm.atlas_h = pl.h;
                            pm.height_min = pc.height_min;
                            pm.height_max = pc.height_max;

                            if (cur_x + pl.w > trial_width) {
                                cur_x = 0;
                                cur_y += row_h;
                                row_h = 0;
                            }
                            pm.atlas_x = cur_x;
                            pm.atlas_y = cur_y;
                            cur_x += pl.w;
                            row_h = std::max(row_h, pl.h);
                            texel_sum += pl.w * pl.h;
                        }

                        int h = cur_y + row_h;
                        trials.push_back({trial_width, h, trial_width * h, texel_sum, std::move(meta)});
                    }

                    auto best_it = std::min_element(trials.begin(), trials.end(),
                        [](const PackTrial& a, const PackTrial& b) {
                            return a.total_texels < b.total_texels;
                        });

                    int atlas_w = best_it->width;
                    int atlas_h = best_it->height;
                    atlas_texel_count = best_it->used_texels;
                    std::vector<PatchMetaGPU> patch_meta = std::move(best_it->meta);

                    // Create / update PatchMeta SSBO (binding 10)
                    if (patch_meta_ssbo.handle) {
                        glDeleteBuffers(1, &patch_meta_ssbo.handle);
                        patch_meta_ssbo.handle = 0;
                    }
                    patch_meta_ssbo.create(patch_meta.data(),
                                           patch_meta.size() * sizeof(PatchMetaGPU));
                    patch_meta_ssbo.bind(10);
                    atlas_current_w = atlas_w;
                    atlas_current_h = atlas_h;
                    atlas_size = {atlas_w, atlas_h};

                    // --- Create / update BVH SSBOs ---
                    bvh_timer.begin();
                    int bvh_work_cap = std::max(8, patch_count) + 64;
                    int bvh_node_cap = std::max(2, 2 * patch_count) + 64;

                    auto rebind_ssbo = [&](SSBO& ssbo, const void* data, size_t bytes, GLuint b) {
                        if (ssbo.handle) glDeleteBuffers(1, &ssbo.handle);
                        ssbo.create(data, bytes);
                        ssbo.bind(b);
                    };

                    std::vector<uint32_t> bvh_ctr_init(18, 0);
                    rebind_ssbo(bvh_counters_ssbo, bvh_ctr_init.data(), 18 * sizeof(uint32_t), 12);

                    rebind_ssbo(bvh_flat_ssbo, nullptr, bvh_work_cap * sizeof(uint64_t), 14);
                    rebind_ssbo(bvh_nodes_ssbo, nullptr, bvh_node_cap * 48, 15);

                    std::vector<int32_t> roots_init(6, -1);
                    rebind_ssbo(bvh_roots_ssbo, roots_init.data(), 6 * sizeof(int32_t), 16);

                    // Work input (binding 17): count + items[cap][3]
                    rebind_ssbo(bvh_work_ssbo, nullptr, (1 + bvh_work_cap * 3) * sizeof(int32_t), 17);
                    // Work output (binding 18): same layout
                    rebind_ssbo(bvh_next_ssbo, nullptr, (1 + bvh_work_cap * 3) * sizeof(int32_t), 18);

                    uint32_t node_counter_init = 1; // node 0 = root
                    rebind_ssbo(bvh_node_counter_ssbo, &node_counter_init, sizeof(uint32_t), 19);

                    // --- BVH bin counting + scatter ---
                    prog_bvh_count->use();
                    GLint _bvh_loc = prog_bvh_count->uniform_location("u_patch_count");
                    if (_bvh_loc >= 0) prog_bvh_count->uniform1i(_bvh_loc, patch_count);
                    gl::dispatch_compute(uint32_t((patch_count + 255) / 256), 1, 1);
                    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

                    // CPU: compute per-bin prefix sums
                    std::vector<uint32_t> bin_counts(6, 0);
                    glGetNamedBufferSubData(bvh_counters_ssbo.handle, 0, 6 * sizeof(uint32_t), bin_counts.data());
                    uint32_t flat_offset = 0;
                    std::vector<uint32_t> bin_offsets(6, 0);
                    for (int b = 0; b < 6; ++b) {
                        bin_offsets[b] = flat_offset;
                        flat_offset += bin_counts[b];
                    }
                    bvh_debug_bin_offsets = bin_offsets;
                    // Write offsets + reset cursors to 0
                    std::vector<uint32_t> ctr_data(18);
                    for (int b = 0; b < 6; ++b) {
                        ctr_data[b]      = bin_counts[b];
                        ctr_data[6 + b]  = bin_offsets[b];
                        ctr_data[12 + b] = 0; // cursor
                    }
                    glNamedBufferSubData(bvh_counters_ssbo.handle, 0, 18 * sizeof(uint32_t), ctr_data.data());

                    // Scatter patches into per-bin flat arrays
                    prog_bvh_scatter->use();
                    auto _sloc = [&](const char* n) { return prog_bvh_scatter->uniform_location(n); };
                    if ((_bvh_loc = _sloc("u_patch_count")) >= 0)
                        prog_bvh_scatter->uniform1i(_bvh_loc, patch_count);
                    if ((_bvh_loc = _sloc("u_mesh_min")) >= 0)
                        prog_bvh_scatter->uniform3f(_bvh_loc, mesh_aabb_min.x, mesh_aabb_min.y, mesh_aabb_min.z);
                    glm::vec3 mesh_inv = 1.0f / glm::max(mesh_extent, glm::vec3(1e-10f));
                    if ((_bvh_loc = _sloc("u_mesh_inv")) >= 0)
                        prog_bvh_scatter->uniform3f(_bvh_loc, mesh_inv.x, mesh_inv.y, mesh_inv.z);
                    gl::dispatch_compute(uint32_t((patch_count + 255) / 256), 1, 1);
                    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

                    // GPU bitonic sort of each bin's Morton-coded data
                    {
                        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bvh_flat_ssbo.handle);
                        prog_sort_flat->use();
                        GLint _sort_N    = prog_sort_flat->uniform_location("u_N");
                        GLint _sort_j    = prog_sort_flat->uniform_location("u_j");
                        GLint _sort_k    = prog_sort_flat->uniform_location("u_k");
                        GLint _sort_base = prog_sort_flat->uniform_location("u_base");
                        for (int b = 0; b < 6; ++b) {
                            uint32_t cnt = bin_counts[b];
                            if (cnt < 2) continue;
                            uint32_t off = bin_offsets[b];
                            uint32_t N = 1;
                            while (N < cnt) N <<= 1;
                            if (_sort_N >= 0) prog_sort_flat->uniform1ui(_sort_N, N);
                            if (_sort_base >= 0) prog_sort_flat->uniform1ui(_sort_base, off);
                            uint32_t groups = (N + 255) / 256;
                            for (uint32_t k = 2; k <= N; k <<= 1) {
                                for (uint32_t j = k >> 1; j > 0; j >>= 1) {
                                    if (_sort_j >= 0) prog_sort_flat->uniform1ui(_sort_j, j);
                                    if (_sort_k >= 0) prog_sort_flat->uniform1ui(_sort_k, k);
                                    gl::dispatch_compute(groups, 1, 1);
                                    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
                                }
                            }
                        }
                    }

                    // Destroy old atlas textures
                    if (height_atlas_tex) { glDeleteTextures(1, &height_atlas_tex); height_atlas_tex = 0; }
                    if (edge_atlas_tex)   { glDeleteTextures(1, &edge_atlas_tex);   edge_atlas_tex = 0; }
                    if (diag_atlas_tex)   { glDeleteTextures(1, &diag_atlas_tex);   diag_atlas_tex = 0; }
                    if (uv_atlas_tex)     { glDeleteTextures(1, &uv_atlas_tex);     uv_atlas_tex = 0; }

                    // Mip level count for height atlas
                    int mip_levels = 1 + int(std::floor(std::log2(std::max(atlas_w, atlas_h))));

                    // Create height atlas (R16F) with mip maps
                    glCreateTextures(GL_TEXTURE_2D, 1, &height_atlas_tex);
                    glTextureStorage2D(height_atlas_tex, mip_levels, GL_RG16F, atlas_w, atlas_h);
                    glTextureParameteri(height_atlas_tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
                    glTextureParameteri(height_atlas_tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                    glTextureParameteri(height_atlas_tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTextureParameteri(height_atlas_tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

                    // Create diagonal direction atlas (R8) — per-cell triangle split, single level
                    glCreateTextures(GL_TEXTURE_2D, 1, &diag_atlas_tex);
                    glTextureStorage2D(diag_atlas_tex, 1, GL_R8, atlas_w, atlas_h);
                    glTextureParameteri(diag_atlas_tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                    glTextureParameteri(diag_atlas_tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                    glTextureParameteri(diag_atlas_tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTextureParameteri(diag_atlas_tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

                    // Keep edge atlas (unused by new DDA code, but retained for potential debugging)

                    // Create UV atlas (RG32F)
                    glCreateTextures(GL_TEXTURE_2D, 1, &uv_atlas_tex);
                    glTextureStorage2D(uv_atlas_tex, 1, GL_RG32F, atlas_w, atlas_h);
                    glTextureParameteri(uv_atlas_tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                    glTextureParameteri(uv_atlas_tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                    glTextureParameteri(uv_atlas_tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTextureParameteri(uv_atlas_tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

                    // Clear textures (timed)
                    glBeginQuery(GL_TIME_ELAPSED, gpu_clear_q);
                    float clear_hv[2] = {0.0f, 0.0f};
                    for (int m = 0; m < mip_levels; ++m)
                        glClearTexImage(height_atlas_tex, m, GL_RG, GL_FLOAT, clear_hv);
                    float clear_uv[2] = {-1.0f, -1.0f};
                    glClearTexImage(uv_atlas_tex, 0, GL_RG, GL_FLOAT, clear_uv);
                    glEndQuery(GL_TIME_ELAPSED);

                    // Bind images for compute
                    glBindImageTexture(0, height_atlas_tex, 0, GL_FALSE, 0,
                                       GL_READ_WRITE, GL_RG16F);
                    glBindImageTexture(1, uv_atlas_tex, 0, GL_FALSE, 0,
                                       GL_READ_WRITE, GL_RG32F);

                    // Dispatch rasterization (timed)
                    prog_rasterize->use();
                    auto rloc = [&](const char* n) { return prog_rasterize->uniform_location(n); };
                    GLint _loc;
                    if ((_loc = rloc("u_tri_count"))     >= 0) prog_rasterize->uniform1ui(_loc, uint32_t(tri_count));
                    if ((_loc = rloc("u_patch_count"))    >= 0) prog_rasterize->uniform1i(_loc, patch_count);
                    if ((_loc = rloc("u_texel_density"))  >= 0) prog_rasterize->uniform1f(_loc, texel_density);

                    glBeginQuery(GL_TIME_ELAPSED, gpu_raster_q);
                    gl::dispatch_compute(uint32_t((tri_count + 255) / 256), 1, 1);
                    gl::memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
                    glEndQuery(GL_TIME_ELAPSED);

                    glBeginQuery(GL_TIME_ELAPSED, gpu_mip_q);
                    glTextureParameteri(height_atlas_tex, GL_TEXTURE_MIN_REDUCTION_MODE_EXT, GL_MAX_EXT);
                    glGenerateTextureMipmap(height_atlas_tex);
                    glEndQuery(GL_TIME_ELAPSED);

                    // Diagonal direction computed from height gradient
                    glBindImageTexture(0, diag_atlas_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R8);
                    glBindTextureUnit(0, height_atlas_tex);
                    prog_diag->use();
                    GLint _dloc;
                    GLint dsz[2] = {atlas_w, atlas_h};
                    if ((_dloc = prog_diag->uniform_location("u_atlas_size")) >= 0)
                        prog_diag->uniform2iv(_dloc, dsz);
                    if ((_dloc = prog_diag->uniform_location("u_height_sampler")) >= 0)
                        prog_diag->uniform1i(_dloc, 0);
                    gl::dispatch_compute(uint32_t((atlas_w + 15) / 16), uint32_t((atlas_h + 15) / 16), 1);
                    gl::memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
                    glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R8);

                    atlas_current_w = atlas_w;
                    atlas_current_h = atlas_h;
                    debug_patch_meta = patch_meta;

                    GLuint64 clear_ns = 0, raster_ns = 0, mip_ns = 0;
                    glGetQueryObjectui64v(gpu_clear_q, GL_QUERY_RESULT, &clear_ns);
                    glGetQueryObjectui64v(gpu_raster_q, GL_QUERY_RESULT, &raster_ns);
                    glGetQueryObjectui64v(gpu_mip_q, GL_QUERY_RESULT, &mip_ns);

                    auto rt1 = std::chrono::steady_clock::now();
                    rasterize_ms = std::chrono::duration<float, std::milli>(rt1 - rt0).count();
                    clear_ms = clear_ns / 1e6f;
                    compute_ms_gpu = raster_ns / 1e6f;
                    mip_ms = mip_ns / 1e6f;
                    cpu_meta_ms = rasterize_ms - (clear_ms + compute_ms_gpu + mip_ms);

                    gllib::logf(gllib::LogLevel::info,
                                "Rasterize: clear=%.3fms  compute=%.3fms  mip=%.3fms  cpu_meta=%.3fms  total=%.3fms",
                                clear_ns / 1e6, raster_ns / 1e6, mip_ns / 1e6,
                                rasterize_ms - (clear_ns + raster_ns + mip_ns) / 1e6,
                                rasterize_ms);
                    gllib::logf(gllib::LogLevel::info,
                                "Rasterized %d patches into %dx%d atlas (%.1fM texels)",
                                patch_count, atlas_w, atlas_h, atlas_texel_count / 1e6);

                    // --- Build per-bin 8-way LBVHs ---
                    {
                        int bvh_roots_data[6];
                        std::vector<uint32_t> bin_node_bases(6, 0);
                        uint32_t total_nodes_needed = 0;
                        for (int b = 0; b < 6; ++b) {
                            bin_node_bases[b] = total_nodes_needed;
                            uint32_t cnt = bin_counts[b];
                            uint32_t max_nodes = cnt > 0 ? (2 * cnt) : 1;
                            total_nodes_needed += max_nodes;
                            bvh_roots_data[b] = cnt > 0 ? (int)bin_node_bases[b] : -1;
                        }
                        glNamedBufferSubData(bvh_roots_ssbo.handle, 0,
                                             sizeof(bvh_roots_data), bvh_roots_data);

                        std::vector<uint64_t> flat_data(patch_count, 0);
                        if (patch_count > 0 && bvh_flat_ssbo.handle) {
                            glGetNamedBufferSubData(bvh_flat_ssbo.handle, 0,
                                                    patch_count * sizeof(uint64_t), flat_data.data());
                        }

                        uint32_t zero32 = 0;
                        for (int b = 0; b < 6; ++b) {
                            uint32_t cnt = bin_counts[b];
                            if (cnt == 0) continue;

                            uint32_t root_idx = bin_node_bases[b];
                            uint32_t max_nodes = 2 * cnt;
                            uint32_t node_counter_init = root_idx + 1;

                            glNamedBufferSubData(bvh_node_counter_ssbo.handle, 0, 4, &node_counter_init);

                            // Initialize root node in node array: n_l, n_r, clear others
                            const size_t node_sz = 48;
                            int root_lr[2] = {0, int(cnt) - 1};
                            glNamedBufferSubData(bvh_nodes_ssbo.handle, root_idx * node_sz + 32, 8, root_lr);
                            int root_zeros[2] = {0, 0};
                            glNamedBufferSubData(bvh_nodes_ssbo.handle, root_idx * node_sz + 12, 8, root_zeros);
                            float huge[6] = {1e30f, 1e30f, 1e30f, -1e30f, -1e30f, -1e30f};
                            glNamedBufferSubData(bvh_nodes_ssbo.handle, root_idx * node_sz, 24, huge);
                            int pad_zero[2] = {0, 0};
                            glNamedBufferSubData(bvh_nodes_ssbo.handle, root_idx * node_sz + 40, 8, pad_zero);

                            // Work list: {wc=1, node_idx, l=0, r=cnt-1}
                            int work_init[4] = {1, int(root_idx), 0, int(cnt) - 1};
                            glNamedBufferSubData(bvh_work_ssbo.handle, 0, sizeof(work_init), work_init);
                            glNamedBufferSubData(bvh_next_ssbo.handle, 0, 4, &zero32);

                            uint32_t wc = 1;
                            prog_bvh_build->use();
                            GLint _b_bin = prog_bvh_build->uniform_location("u_bin");
                            if (_b_bin >= 0) prog_bvh_build->uniform1i(_b_bin, b);
                            while (wc > 0) {
                                gl::dispatch_compute((wc + 255) / 256, 1, 1);
                                gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

                                uint32_t nc;
                                glGetNamedBufferSubData(bvh_next_ssbo.handle, 0, 4, &nc);
                                if (nc == 0) break;

                                size_t copy_bytes = (1 + nc * 3) * sizeof(int32_t);
                                glCopyNamedBufferSubData(bvh_next_ssbo.handle,
                                                         bvh_work_ssbo.handle, 0, 0, copy_bytes);
                                glNamedBufferSubData(bvh_next_ssbo.handle, 0, 4, &zero32);
                                wc = nc;
                            }

                            // Bottom-up AABB on CPU to prevent parallel GPU race condition
                            uint32_t post_count;
                            glGetNamedBufferSubData(bvh_node_counter_ssbo.handle, 0, 4, &post_count);
                            uint32_t bin_nc = post_count - root_idx;
                            if (bin_nc > 0) {
                                std::vector<BVHNodeRaw> cpu_nodes(bin_nc);
                                glGetNamedBufferSubData(bvh_nodes_ssbo.handle, root_idx * sizeof(BVHNodeRaw),
                                                        bin_nc * sizeof(BVHNodeRaw), cpu_nodes.data());

                                for (int i = (int)bin_nc - 1; i >= 0; --i) {
                                    BVHNodeRaw& node = cpu_nodes[i];
                                    bool leaf = (node.meta & 0x80000000) != 0;
                                    if (leaf) {
                                        float mnx = 1e30f, mny = 1e30f, mnz = 1e30f;
                                        float mxx = -1e30f, mxy = -1e30f, mxz = -1e30f;
                                        int lc = node.meta & 0x7FFFFFFF;
                                        for (int j = 0; j < lc; ++j) {
                                            int data_idx = (int)bin_offsets[b] + node.l + j;
                                            if (data_idx >= 0 && data_idx < (int)flat_data.size()) {
                                                uint64_t entry = flat_data[data_idx];
                                                int pid = int(entry & 0xFFFFFFFFu);
                                                if (pid >= 0 && pid < (int)patch_aabb_min.size()) {
                                                    glm::vec3 pmn = patch_aabb_min[pid];
                                                    glm::vec3 pmx = patch_aabb_max[pid];
                                                    if (j == 0) {
                                                        mnx = pmn.x; mny = pmn.y; mnz = pmn.z;
                                                        mxx = pmx.x; mxy = pmx.y; mxz = pmx.z;
                                                    } else {
                                                        mnx = std::min(mnx, pmn.x); mny = std::min(mny, pmn.y); mnz = std::min(mnz, pmn.z);
                                                        mxx = std::max(mxx, pmx.x); mxy = std::max(mxy, pmx.y); mxz = std::max(mxz, pmx.z);
                                                    }
                                                }
                                            }
                                        }
                                        node.min_x = mnx; node.min_y = mny; node.min_z = mnz;
                                        node.max_x = mxx; node.max_y = mxy; node.max_z = mxz;
                                    } else {
                                        float mnx = 1e30f, mny = 1e30f, mnz = 1e30f;
                                        float mxx = -1e30f, mxy = -1e30f, mxz = -1e30f;
                                        int cc = node.meta;
                                        int cb = node.child_base;
                                        for (int j = 0; j < cc; ++j) {
                                            int c_local = (cb + j) - (int)root_idx;
                                            if (c_local >= 0 && c_local < (int)cpu_nodes.size()) {
                                                const BVHNodeRaw& child = cpu_nodes[c_local];
                                                mnx = std::min(mnx, child.min_x); mny = std::min(mny, child.min_y); mnz = std::min(mnz, child.min_z);
                                                mxx = std::max(mxx, child.max_x); mxy = std::max(mxy, child.max_y); mxz = std::max(mxz, child.max_z);
                                            }
                                        }
                                        node.min_x = mnx; node.min_y = mny; node.min_z = mnz;
                                        node.max_x = mxx; node.max_y = mxy; node.max_z = mxz;
                                    }
                                }

                                glNamedBufferSubData(bvh_nodes_ssbo.handle, root_idx * sizeof(BVHNodeRaw),
                                                     bin_nc * sizeof(BVHNodeRaw), cpu_nodes.data());
                            }
                        }
                        gllib::logf(gllib::LogLevel::info,
                                    "Built BVH: %d total nodes across %d active bins",
                                    total_nodes_needed,
                                    std::count_if(bin_counts.begin(), bin_counts.end(),
                                                  [](uint32_t c) { return c > 0; }));
                        bvh_timer.end();
                        bvh_build_ms = (float)bvh_timer.elapsed_ms();

                        // Read back BVH nodes for debug view + depth info
                        bvh_debug_total_nodes = (int)total_nodes_needed;
                        bvh_debug_bin_counts = bin_counts;
                        bvh_debug_nodes.resize(total_nodes_needed * 48);
                        glGetNamedBufferSubData(bvh_nodes_ssbo.handle, 0,
                                                bvh_debug_nodes.size(), bvh_debug_nodes.data());

                        auto* raw = (BVHNodeRaw*)bvh_debug_nodes.data();
                        bvh_per_bin_depth.assign(6, 0);
                        for (int b = 0; b < 6; ++b) {
                            if (bin_counts[b] == 0) continue;
                            int ri = bvh_roots_data[b];
                            if (ri < 0) continue;
                            std::vector<int> depths(total_nodes_needed, 0);
                            std::vector<int> queue = {ri};
                            int max_d = 0;
                            while (!queue.empty()) {
                                int n = queue.back(); queue.pop_back();
                                int d = depths[n];
                                if (d > max_d) max_d = d;
                                int meta = raw[n].meta;
                                if (meta & 0x80000000) continue;
                                int cc = meta;
                                int cb = raw[n].child_base;
                                for (int c = 0; c < cc; ++c) {
                                    int ch = cb + c;
                                    depths[ch] = d + 1;
                                    queue.push_back(ch);
                                }
                            }
                            bvh_per_bin_depth[b] = max_d;
                        }
                    }
                }
            }

            ImGui::Checkbox("Show Height Debug", &show_height_debug);

            if (bvh_debug_total_nodes > 0) {
                ImGui::Separator();
                ImGui::Text("BVH: %.3f ms  (%d nodes)", bvh_build_ms, bvh_debug_total_nodes);
                const char* axis_names[] = {"+X","-X","+Y","-Y","+Z","-Z"};
                for (int b = 0; b < 6; ++b) {
                    if (bvh_debug_bin_counts[b] == 0) continue;
                    ImGui::Checkbox(axis_names[b], &show_bvh_debug[b]);
                    ImGui::SameLine();
                    ImGui::Text(": %u patches, depth %d",
                                bvh_debug_bin_counts[b], bvh_per_bin_depth[b]);
                }
            }

            ImGui::Separator();
            ImGui::Checkbox("Show Recon View", &show_recon);

            ImGui::End();
        }

        // Separate ImGui window for recon view
        if (show_recon && recon_normal_img) {
            ImGui::Begin("Reconstruction View", &show_recon);
            int rays = recon_w * recon_h;
            ImGui::Text("Rays: %d  (%d x %d)", rays, recon_w, recon_h);
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            ImGui::Text("Coverage: %.1f%%  (%d / %d hit)",
                        recon_coverage * 100.0f, recon_hit, rays);
            ImGui::Text("Normal Accuracy: %.1f%%  (%d / %d matched)",
                        recon_accuracy * 100.0f, recon_match, recon_hit);
            ImGui::Checkbox("Fallback Scan", &recon_use_fallback);
            static int recon_display_mode = 1; // 0=position, 1=normal
            ImGui::RadioButton("Position", &recon_display_mode, 0); ImGui::SameLine();
            ImGui::RadioButton("Normal",   &recon_display_mode, 1);
            ImVec2 rsize = ImGui::GetContentRegionAvail();
            float aspect = float(recon_w) / recon_h;
            if (rsize.x / aspect < rsize.y) rsize.y = rsize.x / aspect;
            else rsize.x = rsize.y * aspect;
            GLuint display_img = recon_display_mode == 0 ? recon_pos_img : recon_normal_img;
            ImGui::Image((ImTextureID)(intptr_t)display_img, rsize, ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();
        }

        // Debug height atlas view (temporarily disabled)
        // if (show_height_debug && height_atlas_tex && atlas_current_h > 0) {
        //     ... (full readback + display)
        // }

        // BVH debug visualization
        dd.clear();
        if (bvh_debug_total_nodes > 0) {

            auto* raw = (BVHNodeRaw*)bvh_debug_nodes.data();
            glm::vec4 colors[6] = {
                {1,0,0,1}, {0.5f,0,0,1},
                {0,1,0,1}, {0,0.5f,0,1},
                {0,0,1,1}, {0,0,0.5f,1}
            };
            for (int b = 0; b < 6; ++b) {
                if (bvh_debug_bin_counts[b] == 0 || !show_bvh_debug[b]) continue;
                int ri = 0;
                for (int p = 0; p < b; ++p)
                    ri += bvh_debug_bin_counts[p] > 0 ? 2 * (int)bvh_debug_bin_counts[p] : 1;
                if (ri >= bvh_debug_total_nodes) continue;
                std::vector<int> queue = {ri};
                while (!queue.empty()) {
                    int n = queue.back(); queue.pop_back();
                    int meta = raw[n].meta;
                    glm::vec3 mn(raw[n].min_x, raw[n].min_y, raw[n].min_z);
                    glm::vec3 mx(raw[n].max_x, raw[n].max_y, raw[n].max_z);
                    if (mn.x < mx.x) dd.draw_box(mn, mx, colors[b]);
                    if (!(meta & 0x80000000)) {
                        int cc = meta;
                        int cb = raw[n].child_base;
                        for (int c = 0; c < cc; ++c) queue.push_back(cb + c);
                    }
                }
            }
        }
        if (highlight_patch >= 0) {
            dd.draw_sphere(picked_position, 0.02f, {0.0f, 1.0f, 1.0f, 1.0f});
        }
        dd.render(vp * model_mat);

        {
            // Render ground truth normals to recon FBO
            glBindFramebuffer(GL_FRAMEBUFFER, recon_fbo);
            glViewport(0, 0, recon_w, recon_h);
            glClearColor(0, 0, 0, 1);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            render_prog->use();
            auto rl2 = [&](const char* n) { return render_prog->uniform_location(n); };
            GLint l2;
            if ((l2 = rl2("u_view_proj")) >= 0) render_prog->uniform_matrix4fv(l2, glm::value_ptr(vp));
            if ((l2 = rl2("u_model")) >= 0) render_prog->uniform_matrix4fv(l2, glm::value_ptr(model_mat));
            glm::mat3 nm2 = glm::transpose(glm::inverse(glm::mat3(model_mat)));
            if ((l2 = rl2("u_normal_mat")) >= 0) render_prog->uniform_matrix3fv(l2, glm::value_ptr(nm2));
            if ((l2 = rl2("u_color")) >= 0) render_prog->uniform3f(l2, 0.6f, 0.55f, 0.5f);
            if ((l2 = rl2("u_light_dir")) >= 0) render_prog->uniform3fv(l2, glm::value_ptr(light_dir));
            if ((l2 = rl2("u_debug_mode")) >= 0) render_prog->uniform1i(l2, 1);
            if ((l2 = rl2("u_face_filter")) >= 0) render_prog->uniform1i(l2, 0);
            if ((l2 = rl2("u_highlight_patch")) >= 0) render_prog->uniform1i(l2, -1);
            first_mesh.draw();
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

            // Compute pass
            if (show_recon && patch_count > 0 && height_atlas_tex && recon_pos_img) {
                int zero[4] = {0, 0, 0, 0};
                glNamedBufferSubData(recon_accuracy_ssbo.handle, 0, sizeof(zero), zero);

                glm::mat4 inv_vp = glm::inverse(vp);
                glm::mat4 inv_model = glm::inverse(model_mat);
                glm::vec3 cam_pos = cam.position();
                GLint rsize[2] = {recon_w, recon_h};

                glBindImageTexture(0, recon_pos_img, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
                glBindImageTexture(1, recon_normal_img, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
                prog_recon->use();
                auto _rcl = [&](const char* n) { return prog_recon->uniform_location(n); };
                GLint _rc;
                if ((_rc = _rcl("u_inv_view_proj")) >= 0) prog_recon->uniform_matrix4fv(_rc, glm::value_ptr(inv_vp));
                if ((_rc = _rcl("u_inv_model")) >= 0) prog_recon->uniform_matrix4fv(_rc, glm::value_ptr(inv_model));
                if ((_rc = _rcl("u_model")) >= 0) prog_recon->uniform_matrix4fv(_rc, glm::value_ptr(model_mat));
                if ((_rc = _rcl("u_mesh_aabb_min")) >= 0) prog_recon->uniform3fv(_rc, glm::value_ptr(mesh_aabb_min));
                if ((_rc = _rcl("u_mesh_aabb_max")) >= 0) prog_recon->uniform3fv(_rc, glm::value_ptr(mesh_aabb_max));
                if ((_rc = _rcl("u_patch_count")) >= 0) prog_recon->uniform1i(_rc, patch_count);
                if ((_rc = _rcl("u_texel_density")) >= 0) prog_recon->uniform1f(_rc, texel_density);
                if ((_rc = _rcl("u_view_size")) >= 0) prog_recon->uniform2iv(_rc, rsize);
                if ((_rc = _rcl("u_height_sampler")) >= 0) prog_recon->uniform1i(_rc, 0);
                if ((_rc = _rcl("u_height_linear")) >= 0) prog_recon->uniform1i(_rc, 3);
                if ((_rc = _rcl("u_recon_depth")) >= 0) prog_recon->uniform1i(_rc, 1);
                if ((_rc = _rcl("u_diag_sampler")) >= 0) prog_recon->uniform1i(_rc, 2);
                if ((_rc = _rcl("u_use_fallback")) >= 0) prog_recon->uniform1i(_rc, recon_use_fallback ? 1 : 0);
                if ((_rc = _rcl("u_highlight_patch")) >= 0) prog_recon->uniform1i(_rc, highlight_patch);
                glBindTextureUnit(0, height_atlas_tex);
                glBindTextureUnit(1, recon_depth_tex);
                glBindTextureUnit(2, diag_atlas_tex);
                glBindTextureUnit(3, height_atlas_tex);
                if (bvh_flat_ssbo.handle) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, bvh_flat_ssbo.handle);
                if (bvh_nodes_ssbo.handle) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, bvh_nodes_ssbo.handle);
                if (bvh_roots_ssbo.handle) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 16, bvh_roots_ssbo.handle);
                if ((_rc = _rcl("u_bin_offsets")) >= 0) {
                    GLint offsets_int[6];
                    for (int i = 0; i < 6; ++i) offsets_int[i] = (int)bvh_debug_bin_offsets[i];
                    glUniform1iv(_rc, 6, offsets_int);
                }
                gl::dispatch_compute(uint32_t((recon_w + 15) / 16), uint32_t((recon_h + 15) / 16), 1);
                gl::memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

                glBindImageTexture(1, recon_normal_img, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
                glBindImageTexture(2, recon_truth_tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
                prog_compare->use();
                auto _cpl = [&](const char* n) { return prog_compare->uniform_location(n); };
                GLint _cp;
                if ((_cp = _cpl("u_view_size")) >= 0) prog_compare->uniform2iv(_cp, rsize);
                gl::dispatch_compute(uint32_t((recon_w + 15) / 16), uint32_t((recon_h + 15) / 16), 1);
                gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

                int acc[4];
                glGetNamedBufferSubData(recon_accuracy_ssbo.handle, 0, sizeof(acc), acc);
                recon_hit = acc[0]; recon_match = acc[1];
                int total = recon_w * recon_h;
                recon_coverage = total > 0 ? float(recon_hit) / total : 0.0f;
                recon_accuracy = recon_hit > 0 ? float(recon_match) / recon_hit : 0.0f;
            }
        }

        gui.render();

        window.swap_buffers();
        window.poll_events();
    }

    delete render_prog;
    delete prog_normals;
    delete prog_emit;
    delete prog_sort;
    delete prog_sort_flat;
    delete prog_sweep;
    delete prog_uf_init;
    delete prog_uf_hook;
    delete prog_compress;
    delete prog_compact_a;
    delete prog_compact_b;
    delete prog_adj_degree;
    delete prog_adj_scatter;
    delete prog_relabel;
    delete prog_orphan;
    delete prog_rasterize;
    delete prog_patch_aabb;
    delete prog_bvh_count;
    delete prog_bvh_scatter;
    delete prog_bvh_build;
    delete prog_bvh_aabb;
    delete prog_pick;
    delete prog_recon;

    delete prog_compare;
    delete prog_diag;
    if (pick_result_ssbo.handle) glDeleteBuffers(1, &pick_result_ssbo.handle);
    if (recon_accuracy_ssbo.handle) glDeleteBuffers(1, &recon_accuracy_ssbo.handle);
    if (patch_meta_ssbo.handle) glDeleteBuffers(1, &patch_meta_ssbo.handle);
    if (patch_aabb_ssbo.handle) glDeleteBuffers(1, &patch_aabb_ssbo.handle);
    if (patch_aabb_float_ssbo.handle) glDeleteBuffers(1, &patch_aabb_float_ssbo.handle);
    if (bvh_counters_ssbo.handle) glDeleteBuffers(1, &bvh_counters_ssbo.handle);
    if (bvh_flat_ssbo.handle) glDeleteBuffers(1, &bvh_flat_ssbo.handle);
    if (bvh_nodes_ssbo.handle) glDeleteBuffers(1, &bvh_nodes_ssbo.handle);
    if (bvh_roots_ssbo.handle) glDeleteBuffers(1, &bvh_roots_ssbo.handle);
    if (bvh_work_ssbo.handle) glDeleteBuffers(1, &bvh_work_ssbo.handle);
    if (bvh_next_ssbo.handle) glDeleteBuffers(1, &bvh_next_ssbo.handle);
    if (bvh_node_counter_ssbo.handle) glDeleteBuffers(1, &bvh_node_counter_ssbo.handle);
    if (height_atlas_tex) glDeleteTextures(1, &height_atlas_tex);
    if (uv_atlas_tex)     glDeleteTextures(1, &uv_atlas_tex);
    if (debug_display_tex) glDeleteTextures(1, &debug_display_tex);
    if (recon_pos_img)          glDeleteTextures(1, &recon_pos_img);
    if (recon_normal_img)       glDeleteTextures(1, &recon_normal_img);
    if (recon_truth_tex)        glDeleteTextures(1, &recon_truth_tex);
    if (recon_display_tex)      glDeleteTextures(1, &recon_display_tex);
    if (recon_fbo)              glDeleteFramebuffers(1, &recon_fbo);
    if (recon_depth_tex)        glDeleteTextures(1, &recon_depth_tex);
    if (depth_copy_tex)         glDeleteTextures(1, &depth_copy_tex);
    if (depth_copy_fbo)         glDeleteFramebuffers(1, &depth_copy_fbo);
    glDeleteQueries(1, &gpu_query);
    glDeleteQueries(1, &gpu_clear_q);
    glDeleteQueries(1, &gpu_raster_q);
    glDeleteQueries(1, &gpu_mip_q);

    return EXIT_SUCCESS;
}

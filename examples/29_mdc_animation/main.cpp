// Example 29 — MDC + Animation: skinned character (MDC every frame) + static dragon.
// GPU-driven MDC with dirty-bone gating: triangles whose dominant bone hasn't
// moved significantly reuse their last frame's face_mask/normal/center.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <gllib/log.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <gfx/imgui_overlay.hpp>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// ======================= COMPUTE SHADER SOURCES ===========================

// --- normals / centers / initial face mask (with skinning support) ---
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

layout(std430, binding = 0) readonly buffer VertBuf    { float verts[]; };
layout(std430, binding = 1) buffer TriBuf { TriData tris[]; };

layout(std430, binding = 2) readonly buffer IdxBuf     { uint    idx[]; };
layout(std430, binding =10) readonly buffer BoneVBO    { uint    bone_data[]; };
layout(std430, binding =11) readonly buffer BoneMat    { mat4    bones[]; };
layout(std430, binding =12) readonly buffer BoneDirty  { uint    dirty[]; };
layout(std430, binding =13) readonly buffer DomBone    { uint    dom_bone[]; };

uniform uint  u_tri_count;
uniform bool  u_has_skin;

// Per-vertex: bone_data[vid*6] = packed uint16 joints[4] + vec4 weights
vec4 get_skinned_pos(uint vid, vec3 bind_pos) {
    uint base = vid * 6u;
    uint lo = bone_data[base];
    uint hi = bone_data[base + 1];
    uvec4 j;
    j.x = lo & 0xFFFFu;
    j.y = (lo >> 16) & 0xFFFFu;
    j.z = hi & 0xFFFFu;
    j.w = (hi >> 16) & 0xFFFFu;
    vec4 w = vec4(
        uintBitsToFloat(bone_data[base + 2]),
        uintBitsToFloat(bone_data[base + 3]),
        uintBitsToFloat(bone_data[base + 4]),
        uintBitsToFloat(bone_data[base + 5])
    );
    mat4 sk = mat4(0);
    for (int i = 0; i < 4; ++i) sk += w[i] * bones[j[i]];
    return sk * vec4(bind_pos, 1.0);
}

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= u_tri_count) return;

    // Dirty-bone skip
    if (u_has_skin && dirty[dom_bone[t]] == 0u) return;

    uint i0 = idx[t * 3u];
    uint i1 = idx[t * 3u + 1u];
    uint i2 = idx[t * 3u + 2u];

    vec3 p0 = vec3(verts[i0 * 12u], verts[i0 * 12u + 1u], verts[i0 * 12u + 2u]);
    vec3 p1 = vec3(verts[i1 * 12u], verts[i1 * 12u + 1u], verts[i1 * 12u + 2u]);
    vec3 p2 = vec3(verts[i2 * 12u], verts[i2 * 12u + 1u], verts[i2 * 12u + 2u]);

    if (u_has_skin) {
        p0 = get_skinned_pos(i0, p0).xyz;
        p1 = get_skinned_pos(i1, p1).xyz;
        p2 = get_skinned_pos(i2, p2).xyz;
    }

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

// --- sweep sorted keys for duplicates -> compacted adjacency pairs ---
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

layout(std430, binding = 1) buffer TriBuf { TriData tris[]; };
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

// ==================== VERTEX / FRAGMENT SHADERS ===========================

// Debug render shader (reads TriBuf for face/patch coloring, with skinning support)
static const char* debug_vs = R"(
#version 430 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

layout(std430, binding = 10) readonly buffer BoneVBO { uint bone_data[]; };
layout(std430, binding = 11) readonly buffer BoneMat { mat4 bones[]; };

uniform mat4 u_model;
uniform mat4 u_view_proj;
uniform mat3 u_normal_mat;
uniform bool u_has_skin;

out vec3 v_pos;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    vec3 pos = a_pos;
    vec3 nrm = a_normal;
    if (u_has_skin) {
        uint vid = uint(gl_VertexID);
        uint base = vid * 6u;
        uint lo = bone_data[base];
        uint hi = bone_data[base + 1];
        uvec4 j;
        j.x = lo & 0xFFFFu;
        j.y = (lo >> 16) & 0xFFFFu;
        j.z = hi & 0xFFFFu;
        j.w = (hi >> 16) & 0xFFFFu;
        vec4 w = vec4(
            uintBitsToFloat(bone_data[base + 2]),
            uintBitsToFloat(bone_data[base + 3]),
            uintBitsToFloat(bone_data[base + 4]),
            uintBitsToFloat(bone_data[base + 5])
        );
        mat4 sk = mat4(0);
        for (int i = 0; i < 4; ++i) sk += w[i] * bones[j[i]];
        pos = (sk * vec4(a_pos, 1.0)).xyz;
        nrm = normalize(mat3(sk) * a_normal);
    }
    vec4 world = u_model * vec4(pos, 1.0);
    gl_Position = u_view_proj * world;
    v_pos = world.xyz;
    v_normal = normalize(u_normal_mat * nrm);
    v_uv = a_uv;
}
)";

static const char* debug_fs = R"(
#version 430 core
in vec3 v_pos;
in vec3 v_normal;
in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

uniform vec3 u_color;
uniform vec3 u_light_dir;
uniform int u_debug_mode;
uniform int u_face_filter;

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

    vec3 color = u_color * shade;
    frag_color = vec4(color, 1.0);
}
)";

// --- Self-occlusion detection: depth-peeling vertex shader (same as debug VS) ---
static const char* occ_vs = debug_vs;

// Pass 1: write nearest patch_id (0 = empty)
static const char* occ_fs_pass1 = R"(
#version 430 core
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
layout(location = 0) out uint out_patch;
void main() {
    int pid = tris[gl_PrimitiveID].patch_id;
    out_patch = pid >= 0 ? uint(pid + 1) : 0u;
}
)";

// Pass 2: depth-peel, detect same-patch overlap
static const char* occ_fs_pass2 = R"(
#version 430 core
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
layout(std430, binding = 14) coherent buffer OccCount { uint occ_count; };

uniform sampler2D u_depth_tex;
uniform usampler2D u_patch_tex;

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    float d1 = texelFetch(u_depth_tex, coord, 0).r;
    // Skip the nearest layer — we want fragments behind it
    if (gl_FragCoord.z <= d1 + 1e-6) discard;
    uint pid1 = texelFetch(u_patch_tex, coord, 0).r;
    if (pid1 == 0u) discard;
    int pid2 = tris[gl_PrimitiveID].patch_id;
    if (pid2 >= 0 && pid1 == uint(pid2 + 1)) {
        atomicAdd(occ_count, 1u);
    }
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

static gl::Program* make_compute(const char* cs_src) {
    gl::Shader cs(gl::ShaderType::compute, cs_src);
    if (!cs.compiled()) {
        gllib::logf(gllib::LogLevel::error, "Compute shader compile failed");
        return nullptr;
    }
    auto* prog = new gl::Program;
    prog->attach(cs);
    if (!prog->link()) { delete prog; return nullptr; }
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

// ====================================================================
//  main
// ====================================================================

int main() {
    gllib::log_to_stderr(gllib::LogLevel::info);
    gfx::Window window({"29 MDC + Animation", 1400, 900});
    window.vsync(false);

    gfx::ImGuiOverlay gui;
    if (!gui.init(window)) {
        gllib::log(gllib::LogLevel::error, "ImGui init failed");
        return EXIT_FAILURE;
    }

    // ---- Load models ----
    gfx::Model dragon_model;
    if (!dragon_model.load("Stanford_Dragon.glb")) {
        gllib::log(gllib::LogLevel::error, "Failed to load Stanford_Dragon.glb");
        return EXIT_FAILURE;
    }
    gllib::logf(gllib::LogLevel::info, "Dragon: %zu meshes", dragon_model.mesh_count());
    const gfx::Mesh& dragon_mesh = dragon_model.mesh(0);
    size_t dragon_tri_count = dragon_mesh.index_count() / 3;

    gfx::Model anim_model;
    if (!anim_model.load("BrainStem.glb")) {
        gllib::log(gllib::LogLevel::error, "Failed to load BrainStem.glb");
        return EXIT_FAILURE;
    }
    gllib::logf(gllib::LogLevel::info, "Animated: %zu meshes, skin=%s, %zu animations",
                anim_model.mesh_count(), anim_model.has_skin() ? "yes" : "no",
                anim_model.animation_count());

    if (!anim_model.has_skin() || anim_model.animation_count() == 0) {
        gllib::log(gllib::LogLevel::error, "Animated model has no skin or animation");
        return EXIT_FAILURE;
    }

    gfx::Skeleton& skeleton = anim_model.skeleton();
    gfx::AnimationClip& anim = anim_model.animation(0);

    // ---- Concatenate all animated meshes into one combined buffer ----
    struct CombinedMesh {
        std::vector<gfx::Vertex> verts;
        std::vector<uint32_t> indices;
        std::vector<gfx::BoneWeight> bone_data;
        std::vector<uint16_t> dom_bone;   // dominant bone per triangle
        size_t tri_count = 0;
        size_t vert_count = 0;
        GLuint vao = 0;
        GLuint vbo = 0;
        GLuint ebo = 0;
        GLuint bone_ssbo = 0;
        GLuint dom_bone_ssbo = 0;
    };

    CombinedMesh cmesh;
    {
        size_t total_verts = 0, total_indices = 0;
        for (size_t i = 0; i < anim_model.mesh_count(); ++i) {
            const auto& m = anim_model.mesh(int(i));
            total_verts += m.vertex_count();
            total_indices += m.index_count();
        }
        cmesh.verts.reserve(total_verts);
        cmesh.indices.reserve(total_indices);
        cmesh.bone_data.reserve(total_verts);

        size_t base_vert = 0;
        for (size_t i = 0; i < anim_model.mesh_count(); ++i) {
            const auto& m = anim_model.mesh(int(i));
            // Read back vertex/index data from GPU buffers
            std::vector<gfx::Vertex> mv(m.vertex_count());
            glGetNamedBufferSubData(m.vbo_handle(), 0, mv.size() * sizeof(gfx::Vertex), mv.data());
            cmesh.verts.insert(cmesh.verts.end(), mv.begin(), mv.end());

            if (m.has_bone_data()) {
                std::vector<gfx::BoneWeight> mb(m.vertex_count());
                glGetNamedBufferSubData(m.bone_vbo_handle(), 0, mb.size() * sizeof(gfx::BoneWeight), mb.data());
                cmesh.bone_data.insert(cmesh.bone_data.end(), mb.begin(), mb.end());
            } else {
                cmesh.bone_data.resize(cmesh.bone_data.size() + m.vertex_count());
            }

            std::vector<uint32_t> mi(m.index_count());
            glGetNamedBufferSubData(m.ebo_handle(), 0, mi.size() * sizeof(uint32_t), mi.data());
            for (auto& idx : mi) idx += uint32_t(base_vert);
            cmesh.indices.insert(cmesh.indices.end(), mi.begin(), mi.end());

            base_vert += m.vertex_count();
        }

        cmesh.vert_count = cmesh.verts.size();
        cmesh.tri_count = cmesh.indices.size() / 3;
        gllib::logf(gllib::LogLevel::info, "Combined animated: %zu triangles, %zu vertices",
                    cmesh.tri_count, cmesh.vert_count);

        // Compute dominant bone per triangle (highest weight across 3 vertices)
        cmesh.dom_bone.resize(cmesh.tri_count, 0);
        for (size_t t = 0; t < cmesh.tri_count; ++t) {
            uint32_t i0 = cmesh.indices[t * 3];
            uint32_t i1 = cmesh.indices[t * 3 + 1];
            uint32_t i2 = cmesh.indices[t * 3 + 2];

            // Accumulate weights per bone across the 3 vertices
            float best_w = -1.0f;
            uint16_t best_b = 0;
            for (uint32_t v : {i0, i1, i2}) {
                if (v >= cmesh.bone_data.size()) continue;
                const auto& bw = cmesh.bone_data[v];
                for (int c = 0; c < 4; ++c) {
                    if (bw.weights[c] > best_w) {
                        best_w = bw.weights[c];
                        best_b = bw.joints[c];
                    }
                }
            }
            cmesh.dom_bone[t] = best_b;
        }

        // Upload combined buffers to GPU
        glCreateBuffers(1, &cmesh.vbo);
        glNamedBufferData(cmesh.vbo,
                          cmesh.verts.size() * sizeof(gfx::Vertex),
                          cmesh.verts.data(), GL_STATIC_DRAW);

        glCreateBuffers(1, &cmesh.ebo);
        glNamedBufferData(cmesh.ebo,
                          cmesh.indices.size() * sizeof(uint32_t),
                          cmesh.indices.data(), GL_STATIC_DRAW);

        glCreateBuffers(1, &cmesh.bone_ssbo);
        glNamedBufferData(cmesh.bone_ssbo,
                          cmesh.bone_data.size() * sizeof(gfx::BoneWeight),
                          cmesh.bone_data.data(), GL_STATIC_DRAW);

        glCreateBuffers(1, &cmesh.dom_bone_ssbo);
        glNamedBufferData(cmesh.dom_bone_ssbo,
                          cmesh.dom_bone.size() * sizeof(uint16_t),
                          cmesh.dom_bone.data(), GL_STATIC_DRAW);

        // Create VAO for debug rendering of combined mesh
        glCreateVertexArrays(1, &cmesh.vao);
        glVertexArrayVertexBuffer(cmesh.vao, 0, cmesh.vbo, 0, sizeof(gfx::Vertex));

        glVertexArrayAttribFormat(cmesh.vao, 0, 3, GL_FLOAT, GL_FALSE,
                                  offsetof(gfx::Vertex, position));
        glVertexArrayAttribBinding(cmesh.vao, 0, 0);
        glEnableVertexArrayAttrib(cmesh.vao, 0);

        glVertexArrayAttribFormat(cmesh.vao, 1, 3, GL_FLOAT, GL_FALSE,
                                  offsetof(gfx::Vertex, normal));
        glVertexArrayAttribBinding(cmesh.vao, 1, 0);
        glEnableVertexArrayAttrib(cmesh.vao, 1);

        glVertexArrayAttribFormat(cmesh.vao, 2, 2, GL_FLOAT, GL_FALSE,
                                  offsetof(gfx::Vertex, texcoord));
        glVertexArrayAttribBinding(cmesh.vao, 2, 0);
        glEnableVertexArrayAttrib(cmesh.vao, 2);

        glVertexArrayElementBuffer(cmesh.vao, cmesh.ebo);


    }

    // ---- Bind combined mesh buffers as SSBOs ----
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cmesh.vbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, cmesh.ebo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, cmesh.bone_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, cmesh.dom_bone_ssbo);

    size_t tri_count = cmesh.tri_count;

    // ---- Create TriBuf SSBO (binding 1) ----
    SSBO tri_ssbo;
    tri_ssbo.create(nullptr, tri_count * sizeof(GPUTri));
    tri_ssbo.bind(1);

    // ---- Compute adjacency buffers ----
    uint32_t edge_keys_per_tri = 3;
    uint32_t raw_edge_count = uint32_t(tri_count * edge_keys_per_tri);
    uint32_t padded = next_pow2(raw_edge_count);

    std::vector<uint64_t> init_keys(padded, UINT64_MAX);
    SSBO edge_key_buf;
    edge_key_buf.create(init_keys.data(), init_keys.size() * sizeof(uint64_t));
    edge_key_buf.bind(3);

    uint32_t max_pairs = uint32_t(tri_count * 2);
    std::vector<uint64_t> empty_pairs(max_pairs, 0ULL);
    SSBO edge_pair_buf;
    edge_pair_buf.create(empty_pairs.data(), empty_pairs.size() * sizeof(uint64_t));
    edge_pair_buf.bind(4);

    Scalars init_scalars{};
    SSBO scalar_buf;
    scalar_buf.create(&init_scalars, sizeof(Scalars));
    scalar_buf.bind(5);

    SSBO parent_buf;
    parent_buf.create(nullptr, tri_count * sizeof(int32_t));
    parent_buf.bind(6);

    SSBO csr_off_buf;
    csr_off_buf.create(nullptr, (tri_count + 1) * sizeof(uint32_t));
    csr_off_buf.bind(7);

    SSBO csr_nbr_buf;
    csr_nbr_buf.create(nullptr, 2 * max_pairs * sizeof(uint32_t));
    csr_nbr_buf.bind(8);

    SSBO size_buf;
    size_buf.create(nullptr, tri_count * sizeof(uint32_t));
    size_buf.bind(9);

    SSBO cursor_buf;
    cursor_buf.create(nullptr, (tri_count + 1) * sizeof(uint32_t));

    // --- Bone dirty flags SSBO (binding 12) — recreated per frame ---
    SSBO bone_dirty_buf;
    int bone_count = skeleton.joint_count();
    bone_dirty_buf.create(nullptr, bone_count * sizeof(uint32_t)); // reusable

    // --- Occlusion counter SSBO (binding 14) ---
    SSBO occ_counter_buf;
    {
        uint32_t zero = 0;
        occ_counter_buf.create(&zero, 4);
        occ_counter_buf.bind(14);
    }

    // --- Compile compute shaders ---
    gl::Program* prog_normals   = make_compute(cs_normals_src);
    gl::Program* prog_emit      = make_compute(cs_emit_edges_src);
    gl::Program* prog_sort      = make_compute(cs_sort_src);
    gl::Program* prog_sweep     = make_compute(cs_sweep_src);
    gl::Program* prog_uf_init   = make_compute(cs_uf_init_src);
    gl::Program* prog_uf_hook   = make_compute(cs_uf_hook_src);
    gl::Program* prog_compress  = make_compute(cs_uf_compress_src);
    gl::Program* prog_compact_a = make_compute(cs_compact_a_src);
    gl::Program* prog_compact_b = make_compute(cs_compact_b_src);

    gl::Program* prog_adj_degree  = make_compute(cs_adj_degree_src);
    gl::Program* prog_adj_scatter = make_compute(cs_adj_scatter_src);
    gl::Program* prog_relabel     = make_compute(cs_relabel_src);
    gl::Program* prog_orphan      = make_compute(cs_orphan_src);

    if (!prog_normals || !prog_emit || !prog_sort || !prog_sweep ||
        !prog_uf_init || !prog_uf_hook || !prog_compress ||
        !prog_compact_a || !prog_compact_b ||
        !prog_adj_degree || !prog_adj_scatter ||
        !prog_relabel || !prog_orphan) {
        gllib::log(gllib::LogLevel::error, "Compute shader compilation failed");
        return EXIT_FAILURE;
    }

    // --- PBR material (for both dragon and animated model) ---
    gfx::PBRMaterial pbr;
    if (!pbr.valid()) {
        gllib::log(gllib::LogLevel::error, "PBR program failed");
        return EXIT_FAILURE;
    }

    // --- IBL ---
    gfx::IBLProbe ibl;
    ibl.generate_procedural(256);
    ibl.bake();

    // --- Debug render shader (for combined mesh) ---
    gl::Program* debug_prog = make_program(debug_vs, debug_fs);
    if (!debug_prog) {
        gllib::log(gllib::LogLevel::error, "Debug shader compilation failed");
        return EXIT_FAILURE;
    }

    // --- Self-occlusion detection programs + FBO ---
    gl::Program* occ_prog_pass1 = make_program(occ_vs, occ_fs_pass1);
    gl::Program* occ_prog_pass2 = make_program(occ_vs, occ_fs_pass2);
    if (!occ_prog_pass1 || !occ_prog_pass2) {
        gllib::log(gllib::LogLevel::error, "Occlusion shader compilation failed");
        return EXIT_FAILURE;
    }
    GLuint occ_fbo1 = 0;
    glCreateFramebuffers(1, &occ_fbo1);
    GLuint occ_color_tex = 0, occ_depth_tex = 0;

    // ===== INITIAL COMPUTE PASSES =====

    auto dispatch_normals = [&](bool has_skin) {
        prog_normals->use();
        GLint loc;
        loc = prog_normals->uniform_location("u_tri_count");
        if (loc >= 0) prog_normals->uniform1ui(loc, uint32_t(tri_count));
        loc = prog_normals->uniform_location("u_has_skin");
        if (loc >= 0) prog_normals->uniform1i(loc, has_skin ? 1 : 0);
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
        uint32_t zero_val = 0;
        glNamedBufferSubData(scalar_buf.handle, offsetof(Scalars, edge_count), 4, &zero_val);

        prog_sweep->use();
        GLint loc = prog_sweep->uniform_location("u_padded");
        if (loc >= 0) prog_sweep->uniform1ui(loc, padded);
        uint32_t groups = (padded + 255) / 256;
        gl::dispatch_compute(groups, 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };

    uint32_t total_edge_pairs = 0;

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
        if (loc >= 0) prog_uf_hook->uniform1ui(loc, total_edge_pairs);
        loc = prog_uf_hook->uniform_location("u_normal_threshold");
        if (loc >= 0) prog_uf_hook->uniform1f(loc, nt);
        uint32_t hook_groups = (total_edge_pairs + 255) / 256;
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

    // --- Run initial MDC compute (no skinning for first pass — bind-pose normals) ---
    dispatch_normals(false);
    dispatch_edges();
    dispatch_sort();
    dispatch_sweep();

    glGetNamedBufferSubData(scalar_buf.handle, offsetof(Scalars, edge_count), 4, &total_edge_pairs);
    gllib::logf(gllib::LogLevel::info, "Animated mesh: %u adjacency edges (GPU built)", total_edge_pairs);

    // --- Build AdjCSR (one-time) ---
    if (total_edge_pairs > 0) {
        uint32_t zero_deg = 0;
        glClearNamedBufferData(csr_off_buf.handle, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero_deg);
        prog_adj_degree->use();
        GLint deg_loc = prog_adj_degree->uniform_location("u_edge_count");
        if (deg_loc >= 0) prog_adj_degree->uniform1ui(deg_loc, total_edge_pairs);
        gl::dispatch_compute((total_edge_pairs + 255) / 256, 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        std::vector<uint32_t> offsets(tri_count + 1, 0);
        glGetNamedBufferSubData(csr_off_buf.handle, 0, tri_count * sizeof(uint32_t), offsets.data());
        for (size_t i = 0; i < tri_count; ++i) offsets[i + 1] += offsets[i];
        glNamedBufferSubData(csr_off_buf.handle, 0, (tri_count + 1) * sizeof(uint32_t), offsets.data());

        glNamedBufferSubData(cursor_buf.handle, 0, (tri_count + 1) * sizeof(uint32_t), offsets.data());
        cursor_buf.bind(7);
        prog_adj_scatter->use();
        GLint sca_loc = prog_adj_scatter->uniform_location("u_edge_count");
        if (sca_loc >= 0) prog_adj_scatter->uniform1ui(sca_loc, total_edge_pairs);
        gl::dispatch_compute((total_edge_pairs + 255) / 256, 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
        csr_off_buf.bind(7);
    }

    // --- Center model positions using CPU readback of combined verts ---
    glm::vec3 anim_center(0);
    for (auto& v : cmesh.verts)
        anim_center += glm::vec3(v.position[0], v.position[1], v.position[2]);
    anim_center /= float(cmesh.verts.size());

    // Dragon center
    std::vector<gfx::Vertex> dv(dragon_mesh.vertex_count());
    glGetNamedBufferSubData(dragon_mesh.vbo_handle(), 0,
                            dv.size() * sizeof(gfx::Vertex), dv.data());
    glm::vec3 dragon_center(0);
    for (auto& v : dv) dragon_center += glm::vec3(v.position[0], v.position[1], v.position[2]);
    dragon_center /= float(dv.size());
    dv.clear(); dv.shrink_to_fit();

    // --- Camera ---
    gfx::Camera cam;
    cam.perspective(45.0f, float(window.width()) / window.height(), 0.1f, 100.0f);
    cam.look_at({0, 1.5f, 5}, {0, 1, 0});

    glm::vec3 light_dir = glm::normalize(glm::vec3(1, -1.5f, 1));

    float yaw = 0, pitch = 0;
    bool captured = false;

    // Animation state
    float anim_time = 0.0f;
    float anim_speed = 1.0f;
    bool playing = true;

    // MDC debug state
    int debug_mode = 3;  // start with patch coloring
    int face_filter = 0;
    int patch_count = 0;
    int bad_patch_count = 0;
    float compute_ms = 0.0f;
    float gpu_ms = 0.0f;
    float normal_threshold = 0.0f;
    float greedy_epsilon = 0.0f;
    int relabel_iters = 0;
    int orphan_iters = 0;
    int hook_iters = 6;
    float dirty_threshold_deg = 3.0f;
    bool show_debug = false;
    int instance_count = 3;
    static const char* face_names[] = { "All", "+X", "-X", "+Y", "-Y", "+Z", "-Z" };

    // Track last bone matrices for dirty detection
    std::vector<glm::mat4> last_bone_matrices;
    // Init bone_dirty_buf to all-1s for first frame
    {
        std::vector<uint32_t> all_dirty(bone_count, 1u);
        glNamedBufferSubData(bone_dirty_buf.handle, 0, bone_count * sizeof(uint32_t), all_dirty.data());
    }

    GLuint gpu_query = 0;
    glGenQueries(1, &gpu_query);

    gfx::Renderer renderer;
    renderer.set_clear_color(0.1f, 0.15f, 0.2f, 1.0f);
    gl::enable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    double last = window.time();

    while (!window.should_close()) {
        double now = window.time();
        float dt = float(now - last);
        last = now;

        window.poll_events();
        fps_control(window, cam, dt, yaw, pitch, captured);
        cam.set_aspect(float(window.width()) / window.height());

        glm::mat4 vp = cam.view_projection();
        glm::mat4 light_vp = gfx::compute_light_vp(cam.view_projection(), light_dir, 3.0f, 6.0f);

        // --- Animation update ---
        if (playing) {
            anim_time += dt * anim_speed;
            if (anim_time > anim.duration) anim_time = 0.0f;
        }
        anim.sample(anim_time, skeleton);
        skeleton.update();

        // --- Dirty-bone gating ---
        bool any_dirty = true; // default to true if first frame
        if (!last_bone_matrices.empty() && last_bone_matrices.size() == size_t(bone_count)) {
            any_dirty = false;
            std::vector<uint32_t> bone_dirty_vec(bone_count, 0);
            float thresh_rad = glm::radians(dirty_threshold_deg);
            for (int b = 0; b < bone_count; ++b) {
                // extract rotation from bone matrices to compare angular change
                const glm::mat4& cur = skeleton.bone_matrices()[b];
                const glm::mat4& prev = last_bone_matrices[b];
                // Compare only the 3x3 rotation submatrix — translation doesn't affect normals
                float diff = 0.0f;
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        diff += std::abs(cur[c][r] - prev[c][r]);
                // Convert to approximate angle using matrix diff -> angle heuristic
                // A rotation of theta around any axis gives a mat3 diff of ~2*sin(theta/2)
                // For small angles, diff ~= theta
                float angle_rad = diff * 0.5f;
                bone_dirty_vec[b] = (angle_rad > thresh_rad) ? 1u : 0u;
                any_dirty |= (bone_dirty_vec[b] != 0);
            }
            glNamedBufferSubData(bone_dirty_buf.handle, 0,
                                 bone_count * sizeof(uint32_t), bone_dirty_vec.data());
        }
        last_bone_matrices.assign(skeleton.bone_matrices().begin(),
                                  skeleton.bone_matrices().end());

        // Bind skeleton SSBO for compute (binding 11) and for render
        GLuint sk_ssbo = skeleton.palette_ssbo();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, sk_ssbo);

        // --- MDC compute (every frame with dirty-bone gating) ---
        auto t0 = std::chrono::steady_clock::now();
        glBeginQuery(GL_TIME_ELAPSED, gpu_query);

    // Track GPU query polling
    static int frame_counter = 0;
    ++frame_counter;

    if (any_dirty) {
        bone_dirty_buf.bind(12);

        dispatch_normals(true);

        // Edge topology is static — edges/sort/sweep run once at load time.
        // Only patches need recomputing when normals change.

        dispatch_patches_full(normal_threshold, hook_iters);

        // Patch count readback (throttled ~2 Hz)
        if (frame_counter % 30 == 0) {
            uint32_t pc = 0;
            glGetNamedBufferSubData(scalar_buf.handle,
                                    offsetof(Scalars, patch_count), 4, &pc);
            patch_count = int(pc);
        }
    }

    // --- Self-occlusion detection via depth peel ---
    {
        static int occ_tex_w = 0, occ_tex_h = 0;
        int w = window.width(), h = window.height();
        if (w != occ_tex_w || h != occ_tex_h) {
            occ_tex_w = w; occ_tex_h = h;
            if (occ_color_tex) glDeleteTextures(1, &occ_color_tex);
            if (occ_depth_tex) glDeleteTextures(1, &occ_depth_tex);
            glCreateTextures(GL_TEXTURE_2D, 1, &occ_color_tex);
            glTextureStorage2D(occ_color_tex, 1, GL_R32UI, w, h);
            glTextureParameteri(occ_color_tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTextureParameteri(occ_color_tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glCreateTextures(GL_TEXTURE_2D, 1, &occ_depth_tex);
            glTextureStorage2D(occ_depth_tex, 1, GL_DEPTH_COMPONENT32F, w, h);
            glNamedFramebufferTexture(occ_fbo1, GL_COLOR_ATTACHMENT0, occ_color_tex, 0);
            glNamedFramebufferTexture(occ_fbo1, GL_DEPTH_ATTACHMENT, occ_depth_tex, 0);
        }

        // Reset counter
        {
            uint32_t zero = 0;
            glNamedBufferSubData(occ_counter_buf.handle, 0, 4, &zero);
        }

        // Instance transforms for this frame
        static std::vector<glm::mat4> ixforms;
        ixforms.resize(instance_count);
        for (int i = 0; i < instance_count; ++i) {
            float x = (i - (instance_count - 1) * 0.5f) * 1.8f;
            glm::mat4 m = glm::translate(glm::mat4(1.0f), -anim_center);
            m = glm::translate(m, glm::vec3(x, 0, 0));
            m = glm::scale(m, glm::vec3(1.5f));
            ixforms[i] = m;
        }

        // Pass 1: render nearest pixels (color = patch_id + 1)
        GLenum draw_bufs[] = { GL_COLOR_ATTACHMENT0 };
        glNamedFramebufferDrawBuffers(occ_fbo1, 1, draw_bufs);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, occ_fbo1);
        glViewport(0, 0, w, h);
        GLuint clr0[] = { 0 };
        glClearBufferuiv(GL_COLOR, 0, clr0);
        glClear(GL_DEPTH_BUFFER_BIT);
        occ_prog_pass1->use();
        {
            GLint loc = occ_prog_pass1->uniform_location("u_view_proj");
            if (loc >= 0) occ_prog_pass1->uniform_matrix4fv(loc, glm::value_ptr(vp));
            loc = occ_prog_pass1->uniform_location("u_has_skin");
            if (loc >= 0) occ_prog_pass1->uniform1i(loc, 1);
        }
        glBindVertexArray(cmesh.vao);
        for (int i = 0; i < instance_count; ++i) {
            glm::mat3 nm = glm::transpose(glm::inverse(glm::mat3(ixforms[i])));
            GLint loc = occ_prog_pass1->uniform_location("u_model");
            if (loc >= 0) occ_prog_pass1->uniform_matrix4fv(loc, glm::value_ptr(ixforms[i]));
            loc = occ_prog_pass1->uniform_location("u_normal_mat");
            if (loc >= 0) occ_prog_pass1->uniform_matrix3fv(loc, glm::value_ptr(nm));
            glDrawElements(GL_TRIANGLES, GLsizei(cmesh.indices.size()), GL_UNSIGNED_INT, 0);
        }

        // Pass 2: detect fragments behind the nearest with the same patch
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glViewport(0, 0, w, h);
        glDisable(GL_DEPTH_TEST);
        GLboolean color_mask[4];
        glGetBooleanv(GL_COLOR_WRITEMASK, color_mask);
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

        occ_prog_pass2->use();
        {
            GLint loc = occ_prog_pass2->uniform_location("u_view_proj");
            if (loc >= 0) occ_prog_pass2->uniform_matrix4fv(loc, glm::value_ptr(vp));
            loc = occ_prog_pass2->uniform_location("u_has_skin");
            if (loc >= 0) occ_prog_pass2->uniform1i(loc, 1);
        }
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, occ_depth_tex);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, occ_color_tex);
        {
            GLint loc = occ_prog_pass2->uniform_location("u_depth_tex");
            if (loc >= 0) occ_prog_pass2->uniform1i(loc, 0);
            loc = occ_prog_pass2->uniform_location("u_patch_tex");
            if (loc >= 0) occ_prog_pass2->uniform1i(loc, 1);
        }

        for (int i = 0; i < instance_count; ++i) {
            glm::mat3 nm = glm::transpose(glm::inverse(glm::mat3(ixforms[i])));
            GLint loc = occ_prog_pass2->uniform_location("u_model");
            if (loc >= 0) occ_prog_pass2->uniform_matrix4fv(loc, glm::value_ptr(ixforms[i]));
            loc = occ_prog_pass2->uniform_location("u_normal_mat");
            if (loc >= 0) occ_prog_pass2->uniform_matrix3fv(loc, glm::value_ptr(nm));
            glDrawElements(GL_TRIANGLES, GLsizei(cmesh.indices.size()), GL_UNSIGNED_INT, 0);
        }

        // Restore state
        glColorMask(color_mask[0], color_mask[1], color_mask[2], color_mask[3]);
        glEnable(GL_DEPTH_TEST);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);

        // Read back counter (throttled ~2 Hz)
        if (frame_counter % 30 == 0) {
            uint32_t val = 0;
            glGetNamedBufferSubData(occ_counter_buf.handle, 0, 4, &val);
            bad_patch_count = int(val);
        }
    }

    glEndQuery(GL_TIME_ELAPSED);
    auto t1 = std::chrono::steady_clock::now();
    compute_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    // Poll GPU query — don't stall if not ready
    GLint gpu_avail = 0;
    glGetQueryObjectiv(gpu_query, GL_QUERY_RESULT_AVAILABLE, &gpu_avail);
    if (gpu_avail) {
        GLuint64 gpu_ns = 0;
        glGetQueryObjectui64v(gpu_query, GL_QUERY_RESULT, &gpu_ns);
        gpu_ms = float(gpu_ns / 1e6);
    }

        // --- Render ---
        glViewport(0, 0, window.width(), window.height());
        renderer.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (show_debug) {
            // Debug render: visualize MDC on combined mesh
            debug_prog->use();
            auto rloc = [&](const char* n) { return debug_prog->uniform_location(n); };
            glm::mat4 m = glm::translate(glm::mat4(1.0f), -anim_center);
            m = glm::scale(m, glm::vec3(1.5f));
            glm::mat3 nm = glm::transpose(glm::inverse(glm::mat3(m)));

            GLint loc;
            loc = rloc("u_view_proj");  if (loc >= 0) debug_prog->uniform_matrix4fv(loc, glm::value_ptr(vp));
            loc = rloc("u_model");       if (loc >= 0) debug_prog->uniform_matrix4fv(loc, glm::value_ptr(m));
            loc = rloc("u_normal_mat");  if (loc >= 0) debug_prog->uniform_matrix3fv(loc, glm::value_ptr(nm));
            loc = rloc("u_color");       if (loc >= 0) debug_prog->uniform3f(loc, 0.6f, 0.55f, 0.5f);
            loc = rloc("u_light_dir");   if (loc >= 0) debug_prog->uniform3fv(loc, glm::value_ptr(light_dir));
            loc = rloc("u_debug_mode");  if (loc >= 0) debug_prog->uniform1i(loc, debug_mode);
            loc = rloc("u_face_filter"); if (loc >= 0) debug_prog->uniform1i(loc, face_filter);
            loc = rloc("u_has_skin");    if (loc >= 0) debug_prog->uniform1i(loc, 1);

            // Render combined mesh with TriBuf data
            glBindVertexArray(cmesh.vao);
            glDrawElements(GL_TRIANGLES, uint32_t(cmesh.tri_count * 3), GL_UNSIGNED_INT, nullptr);
        } else {
            // --- Render Dragon (static, no skin) ---
            pbr.begin(cam.view(), cam.projection(), cam.position());
            {
                glm::mat4 m = glm::translate(glm::mat4(1.0f), -dragon_center);
                m = glm::translate(m, glm::vec3(-2.5f, 0, 0));
                m = glm::rotate(m, glm::radians(90.0f), glm::vec3(1, 0, 0));
                m = glm::scale(m, glm::vec3(1.5f));
                pbr.set_model_matrix(m);
                pbr.set_ibl(ibl);
                pbr.set_ambient_hemi({0.3f, 0.4f, 0.6f}, {0.1f, 0.08f, 0.06f}, 0.15f);
                pbr.set_skin(0); // disable skinning
                for (size_t i = 0; i < dragon_model.mesh_count(); ++i) {
                    int mi = dragon_model.mesh_material(int(i));
                    if (mi >= 0 && size_t(mi) < dragon_model.material_count())
                        pbr.set_material(dragon_model.material_info(size_t(mi)), dragon_model);
                    pbr.draw(dragon_model.mesh(int(i)));
                }
            }

            // --- Render animated model(s) (skinned), N instances ---
            {
                pbr.set_ibl(ibl);
                pbr.set_ambient_hemi({0.3f, 0.4f, 0.6f}, {0.1f, 0.08f, 0.06f}, 0.15f);
                pbr.set_skin(sk_ssbo);
                for (int inst = 0; inst < instance_count; ++inst) {
                    float x = (inst - (instance_count - 1) * 0.5f) * 1.8f;
                    glm::mat4 m = glm::translate(glm::mat4(1.0f), -anim_center);
                    m = glm::translate(m, glm::vec3(x, 0, 0));
                    m = glm::scale(m, glm::vec3(1.5f));
                    pbr.set_model_matrix(m);
                    for (size_t i = 0; i < anim_model.mesh_count(); ++i) {
                        int mi = anim_model.mesh_material(int(i));
                        pbr.set_material(anim_model.material_info(size_t(mi >= 0 ? mi : 0)), anim_model);
                        pbr.draw(anim_model.mesh(int(i)));
                    }
                }
            }

            pbr.end();

            // Restore PairBuf at binding 4 — PBR::set_skin(sk_ssbo) overwrote it
            edge_pair_buf.bind(4);
        }

        // --- ImGui ---
        gui.begin_frame();
        {
            ImGui::Begin("MDC + Animation Controls");

            ImGui::Text("Animated: %zu tris x%d, %d bones",
                        tri_count, instance_count, bone_count);
            ImGui::Text("Dragon: %zu tris", dragon_tri_count);
            ImGui::Text("Total rendered: %zu tris",
                        dragon_tri_count + tri_count * size_t(instance_count));
            ImGui::Separator();

            ImGui::Text("Animation: %s", anim.name.c_str());
            ImGui::Text("Duration: %.2f s", anim.duration);
            ImGui::Separator();

            if (ImGui::Button(playing ? "Pause" : "Play"))
                playing = !playing;
            ImGui::SameLine();
            ImGui::SliderFloat("Speed", &anim_speed, 0.0f, 3.0f, "%.2fx");

            bool scrubbing = ImGui::SliderFloat("Time", &anim_time, 0.0f, anim.duration, "%.3f s");
            if (scrubbing) {
                playing = false;
                anim.sample(anim_time, skeleton);
                skeleton.update();
            }

            ImGui::Separator();
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            ImGui::SliderInt("Instances", &instance_count, 1, 50);

            ImGui::Separator();
            ImGui::Text("MDC Timing");
            ImGui::Text("CPU: %.3f ms  GPU: %.3f ms", compute_ms, gpu_ms);
            if (patch_count > 0)
                ImGui::Text("Patches: %d (animated mesh only)", patch_count);
            if (bad_patch_count > 0)
                ImGui::TextColored(ImVec4(1,0.2f,0.2f,1), "Self-occluding fragments: %d", bad_patch_count);
            else
                ImGui::Text("Self-occluding fragments: 0");

            ImGui::Separator();
            ImGui::Text("Dirty-Bone Gating");
            ImGui::SliderFloat("Threshold (deg)", &dirty_threshold_deg, 0.5f, 20.0f, "%.1f");

            ImGui::Separator();
            ImGui::Checkbox("Show MDC Debug", &show_debug);
            if (show_debug) {
                ImGui::Combo("Face Filter", &face_filter, face_names, 7);
                ImGui::RadioButton("Flat", &debug_mode, 0);
                ImGui::RadioButton("Normals", &debug_mode, 1);
                ImGui::RadioButton("Faces", &debug_mode, 2);
                ImGui::RadioButton("Patches", &debug_mode, 3);
            }

            ImGui::Separator();
            ImGui::Text("MDC Params");
            ImGui::SliderFloat("Normal Threshold", &normal_threshold, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Greedy Epsilon", &greedy_epsilon, 0.0f, 0.5f, "%.3f");
            ImGui::SliderInt("Hook Iters", &hook_iters, 1, 20);
            ImGui::SliderInt("Relabel Iters", &relabel_iters, 0, 10);
            ImGui::SliderInt("Orphan Iters", &orphan_iters, 0, 5);

            ImGui::End();
        }
        gui.render();

        window.swap_buffers();
        window.poll_events();
    }

    // Cleanup
    delete debug_prog;
    delete prog_normals;
    delete prog_emit;
    delete prog_sort;
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
    delete occ_prog_pass1;
    delete occ_prog_pass2;
    if (occ_fbo1) glDeleteFramebuffers(1, &occ_fbo1);
    if (occ_color_tex) glDeleteTextures(1, &occ_color_tex);
    if (occ_depth_tex) glDeleteTextures(1, &occ_depth_tex);
    glDeleteQueries(1, &gpu_query);
    if (cmesh.vao) glDeleteVertexArrays(1, &cmesh.vao);
    if (cmesh.vbo) glDeleteBuffers(1, &cmesh.vbo);
    if (cmesh.ebo) glDeleteBuffers(1, &cmesh.ebo);
    if (cmesh.bone_ssbo) glDeleteBuffers(1, &cmesh.bone_ssbo);
    if (cmesh.dom_bone_ssbo) glDeleteBuffers(1, &cmesh.dom_bone_ssbo);

    return EXIT_SUCCESS;
}

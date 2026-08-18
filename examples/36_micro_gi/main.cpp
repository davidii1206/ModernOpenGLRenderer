// Example 36 — Micro Rendering: Stage A+B
//
// Stage A: G-buffer geometry pass (position, normal, albedo, emissive, depth)
//          with fullscreen display and ImGui visualization.
// Stage B: Offline best-candidate point sampling (Mitchell 1991) + complete
//          binary tree hierarchy (recursive median split, bounding spheres,
//          normal cones).  Debug point-cloud overlay.

#include <gl/gl.hpp>
#include <gl/query.hpp>
#include <gfx/gfx.hpp>
#include <gllib/log.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int LOG2N = 14;
constexpr int N = 1 << LOG2N;       // 16384
constexpr int BEST_K = 32;
constexpr float COVERAGE = 1.4f;
constexpr float LOD_THRESHOLD = 0.15f;

// Morton code helpers for space-filling curve dispatch
uint32_t part1by2(uint32_t x) {
    x &= 0x000003ff;
    x = (x | (x << 16)) & 0x000f000f;
    x = (x | (x << 8)) & 0x00f000f0;
    x = (x | (x << 4)) & 0x0c0c0c0c;
    x = (x | (x << 2)) & 0x22222222;
    return x;
}
uint32_t morton2d(uint32_t x, uint32_t y) {
    return part1by2(x) | (part1by2(y) << 1);
}

// ===========================================================================
// Shaders
// ===========================================================================

const char* gbuf_vs = R"(
#version 460 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

uniform mat4 u_model;
uniform mat4 u_view_proj;
uniform mat3 u_normal_mat;

out vec3 v_pos;
out vec3 v_normal;

void main() {
    vec4 world = u_model * vec4(a_pos, 1.0);
    gl_Position = u_view_proj * world;
    v_pos = world.xyz;
    v_normal = normalize(u_normal_mat * a_normal);
}
)";

const char* gbuf_fs = R"(
#version 460 core
in vec3 v_pos;
in vec3 v_normal;

layout(location = 0) out vec4 out_albedo;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_position;
layout(location = 3) out vec4 out_emissive;
layout(location = 4) out float out_depth;

uniform vec3 u_albedo;
uniform vec3 u_emissive;
uniform mat4 u_view;

void main() {
    vec3 N = normalize(v_normal);
    vec4 view_p = u_view * vec4(v_pos, 1.0);
    out_albedo   = vec4(u_albedo, 1.0);
    out_normal   = vec4(N, 1.0);
    out_position = vec4(v_pos, 1.0);
    out_emissive = vec4(u_emissive, 1.0);
    out_depth    = -view_p.z;
}
)";

const char* display_vs = R"(
#version 460 core
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

const char* display_fs = R"(
#version 460 core
in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_tex;
uniform int u_mode;
uniform float u_far;
uniform float u_exposure;
uniform float u_gamma;

vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec4 c = texture(u_tex, v_uv);

    if (u_mode == 2) {
        c = vec4(c.rgb * 0.5 + 0.5, 1.0);
    } else if (u_mode == 3) {
        c = vec4(clamp(c.rgb * 0.5 + 0.5, 0.0, 1.0), 1.0);
    } else if (u_mode == 4) {
        // emissive — pass through
    } else if (u_mode == 5) {
        c = vec4(vec3(c.r / u_far), 1.0);
    } else if (u_mode == 6) {
        // indirect — pass through
    }

    vec3 col = c.rgb * u_exposure;
    col = aces(col);
    col = pow(max(col, vec3(0.0)), vec3(1.0 / u_gamma));
    frag_color = vec4(col, 1.0);
}
)";

// Point cloud debug overlay — reads positions/normals/colors from SSBOs.
const char* pc_vs = R"(
#version 460 core
layout(std430, binding = 0) readonly buffer PGeom { vec4 pgeom[]; };
layout(std430, binding = 1) readonly buffer PNrm  { vec4 pnrm[];  };
layout(std430, binding = 2) readonly buffer PAlb  { vec4 palb[];  };
layout(std430, binding = 3) readonly buffer PEmit { vec4 pemit[]; };
layout(std430, binding = 8) readonly buffer Rad   { vec4 rad[];   };

uniform mat4 u_view_proj;
uniform float u_point_size;
uniform int u_color_mode;  // 0=albedo 1=emissive 2=normal 3=position 4=radiance
uniform uint u_num_leaves;

out vec3 v_color;

void main() {
    gl_Position = u_view_proj * vec4(pgeom[gl_VertexID].xyz, 1.0);
    gl_PointSize = u_point_size;

    if (u_color_mode == 0)      v_color = palb[gl_VertexID].rgb;
    else if (u_color_mode == 1) v_color = pemit[gl_VertexID].rgb;
    else if (u_color_mode == 2) v_color = pnrm[gl_VertexID].rgb * 0.5 + 0.5;
    else if (u_color_mode == 3) v_color = pgeom[gl_VertexID].xyz;
    else                        v_color = rad[u_num_leaves - 1u + gl_VertexID].rgb;
}
)";

const char* pc_fs = R"(
#version 460 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
)";

// ===========================================================================
// Stage C — Compute shaders: leaf update + tree refit
// ===========================================================================

const char* leaf_update_cs = R"(
#version 460 core
layout(local_size_x = 256) in;

struct GpuTri {
    vec4 pos[3];
    vec4 nrm[3];
};

struct LeafSrc {
    uint tri_idx;
    float u, v, pad;
};

layout(std430, binding = 0) writeonly buffer PGeom   { vec4 pgeom[];   };
layout(std430, binding = 1) writeonly buffer PNrm    { vec4 pnrm[];    };
layout(std430, binding = 4) readonly buffer TriBuf   { GpuTri tris[];  };
layout(std430, binding = 5) readonly buffer LeafSrcB { LeafSrc leaves[]; };
layout(std430, binding = 6) writeonly buffer Sphere   { vec4 sphere[];  };
layout(std430, binding = 7) writeonly buffer Cone     { vec4 cone[];    };
layout(std430, binding = 13) writeonly buffer AabbMin { vec4 aabb_min[]; };
layout(std430, binding = 14) writeonly buffer AabbMax { vec4 aabb_max[]; };

uniform uint u_num_leaves;
uniform float u_leaf_radius;
uniform int u_tree_offset;   // N - 1

void main() {
    uint leaf = gl_GlobalInvocationID.x;
    if (leaf >= u_num_leaves) return;

    LeafSrc src = leaves[leaf];
    GpuTri tri = tris[src.tri_idx];

    float w2 = 1.0 - src.u - src.v;
    vec3 pos = src.u * tri.pos[0].xyz + src.v * tri.pos[1].xyz + w2 * tri.pos[2].xyz;
    vec3 nrm = normalize(src.u * tri.nrm[0].xyz + src.v * tri.nrm[1].xyz + w2 * tri.nrm[2].xyz);

    pgeom[leaf] = vec4(pos, u_leaf_radius);
    pnrm[leaf]  = vec4(nrm, 0.0);

    // Leaf-level tree node entries
    sphere[u_tree_offset + leaf] = vec4(pos, u_leaf_radius);
    cone[u_tree_offset + leaf]   = vec4(nrm, 1.0);

    // Degenerate AABB (min == max == pos) for tight refit
    aabb_min[u_tree_offset + leaf] = vec4(pos, 0.0);
    aabb_max[u_tree_offset + leaf] = vec4(pos, 0.0);
}
)";

const char* tree_refit_cs = R"(
#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 6) buffer Sphere  { vec4 sphere[];  };
layout(std430, binding = 7) buffer Cone    { vec4 cone[];    };
layout(std430, binding = 13) buffer AabbMin { vec4 aabb_min[]; };
layout(std430, binding = 14) buffer AabbMax { vec4 aabb_max[]; };

uniform uint u_count;      // nodes at this level
uniform uint u_level_start; // first node index at this level

const float PI = 3.14159265358979;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= u_count) return;

    uint node = u_level_start + idx;
    uint left  = 2 * node + 1;
    uint right = 2 * node + 2;

    // --- AABB merge (tight bounding volume) ---
    vec3 bmin = min(aabb_min[left].xyz, aabb_min[right].xyz);
    vec3 bmax = max(aabb_max[left].xyz, aabb_max[right].xyz);
    aabb_min[node] = vec4(bmin, 0.0);
    aabb_max[node] = vec4(bmax, 0.0);

    // --- Tight bounding sphere from AABB ---
    vec3 center = (bmin + bmax) * 0.5;
    float radius = length(bmax - bmin) * 0.5;
    sphere[node] = vec4(center, radius);

    // --- Normal cone merge ---
    vec3 a1 = cone[left].xyz,  a2 = cone[right].xyz;
    float w1 = cone[left].w,  w2 = cone[right].w;
    float b1 = acos(clamp(w1, -1.0, 1.0));
    float b2 = acos(clamp(w2, -1.0, 1.0));
    float ang = acos(clamp(dot(a1, a2), -1.0, 1.0));
    vec3 axis; float cw;
    if (ang + b1 + b2 >= PI - 1e-5 || length(a1 + a2) < 1e-4) {
        axis = a1;
        cw = -1.0;
    } else {
        axis = normalize(a1 + a2);
        cw = cos((ang + b1 + b2) * 0.5);
    }
    cone[node] = vec4(axis, cw);
}
)";

// ===========================================================================
// Stage C+ — Sphere/cone reorder: heap order → DFS order
// ===========================================================================

const char* sphere_reorder_cs = R"(
#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 6) readonly buffer SphHeap  { vec4 sph_heap[];  };
layout(std430, binding = 7) readonly buffer ConeHeap { vec4 cone_heap[]; };
layout(std430, binding = 16) writeonly buffer SphDfs  { vec4 sph_dfs[];  };
layout(std430, binding = 17) writeonly buffer ConeDfs { vec4 cone_dfs[]; };
layout(std430, binding = 21) readonly buffer N2O     { uint new_to_old[]; };

uniform uint u_tree_size;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= u_tree_size) return;
    uint h = new_to_old[i];
    sph_dfs[i]  = sph_heap[h];
    cone_dfs[i] = cone_heap[h];
}
)";

// ===========================================================================
// Stage D — Radiance reorder: heap order → DFS order
// ===========================================================================

const char* radiance_reorder_cs = R"(
#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 8) readonly buffer RadHeap { vec4 rad_heap[]; };
layout(std430, binding = 20) writeonly buffer RadDfs { vec4 rad_dfs[]; };
layout(std430, binding = 21) readonly buffer N2O    { uint new_to_old[]; };

uniform uint u_tree_size;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= u_tree_size) return;
    rad_dfs[i] = rad_heap[new_to_old[i]];
}
)";

// ===========================================================================
// Stage D — Direct lighting (emissive point sources) + radiance pull-up
// ===========================================================================

const char* direct_lighting_cs = R"(
#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PGeom     { vec4 pgeom[];    };
layout(std430, binding = 1) readonly buffer PNrm      { vec4 pnrm[];     };
layout(std430, binding = 2) readonly buffer PAlb      { vec4 palb[];     };
layout(std430, binding = 3) readonly buffer PEmit      { vec4 pemit[];    };
layout(std430, binding = 8) writeonly buffer Radiance  { vec4 radiance[]; };
layout(std430, binding = 9) readonly buffer Emitters   { uint emitters[]; };

uniform uint u_num_leaves;
uniform uint u_num_emitters;
uniform float u_leaf_area;
uniform float u_emissive_gain;

const float PI = 3.14159265358979;

void main() {
    uint recv = gl_GlobalInvocationID.x;
    if (recv >= u_num_leaves) return;

    vec3 pos = pgeom[recv].xyz;
    vec3 nrm = normalize(pnrm[recv].xyz);
    vec3 emissive = pemit[recv].rgb * u_emissive_gain;

    vec3 incoming = vec3(0.0);
    for (uint i = 0; i < u_num_emitters; i++) {
        uint e = emitters[i];

        vec3 e_pos = pgeom[e].xyz;
        vec3 e_nrm = normalize(pnrm[e].xyz);
        vec3 e_emit = pemit[e].rgb * u_emissive_gain;

        vec3 dir = e_pos - pos;
        float dist2 = max(dot(dir, dir), 1e-4);
        float dist = sqrt(dist2);
        vec3 wi = dir / dist;

        float cos_recv = max(dot(nrm, wi), 0.0);
        float cos_emit = max(dot(e_nrm, -wi), 0.0);

        incoming += e_emit * cos_emit * cos_recv / dist2 * u_leaf_area;
    }

    radiance[u_num_leaves - 1u + recv] = vec4(emissive + palb[recv].rgb / PI * incoming, u_leaf_area);
}
)";

// ===========================================================================
// Stage D — Hierarchical multi-bounce gather (DFS traversal, O(N log N))
// ===========================================================================

const char* hierarchical_bounce_cs = R"(
#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PGeom    { vec4 pgeom[];    };
layout(std430, binding = 1) readonly buffer PNrm     { vec4 pnrm[];     };
layout(std430, binding = 2) readonly buffer PAlb     { vec4 palb[];     };
layout(std430, binding = 3) readonly buffer PEmit    { vec4 pemit[];    };
layout(std430, binding = 16) readonly buffer Spheres { vec4 sph[];      };
layout(std430, binding = 17) readonly buffer Cones   { vec4 cone[];     };
layout(std430, binding = 18) readonly buffer LChild  { uint left_ch[];  };
layout(std430, binding = 19) readonly buffer RChild  { uint right_ch[]; };
layout(std430, binding = 20) readonly buffer RadDfs  { vec4 rad_dfs[];  };
layout(std430, binding = 10) writeonly buffer RadDst { vec4 rad_dst[];  };

uniform uint u_num_leaves;
uniform uint u_tree_offset;
uniform float u_leaf_area;
uniform float u_emissive_gain;

const float PI = 3.14159265358979;
const uint STACK_DEPTH = 32;

void main() {
    uint recv = gl_GlobalInvocationID.x;
    if (recv >= u_num_leaves) return;

    vec3 pos = pgeom[recv].xyz;
    vec3 nrm = normalize(pnrm[recv].xyz);
    vec3 alb = palb[recv].rgb;
    vec3 emissive = pemit[recv].rgb * u_emissive_gain;

    vec3 incoming = vec3(0.0);

    uint stack[STACK_DEPTH];
    int sp = 0;
    stack[sp++] = 0u;

    while (sp > 0) {
        uint node = stack[--sp];

        vec3 center = sph[node].xyz;
        float radius = sph[node].w;
        vec3 to_center = center - pos;
        float dist2 = dot(to_center, to_center);

        if (dot(to_center, nrm) < 0.0) continue;

        vec4 cn = cone[node];
        float dot_oc_axis = dot(to_center, cn.xyz);
        if (dot_oc_axis > 0.0 && cn.w > 0.0 &&
            dot_oc_axis * dot_oc_axis > dist2 * (1.0 - cn.w * cn.w))
            continue;

        float cz = dot(to_center, nrm);
        if (cz <= 0.0) continue;

        bool is_leaf = (left_ch[node] == 0xFFFFFFFFu);

        if (is_leaf) {
            if (dist2 < u_leaf_area * 0.25) continue;
            vec3 wi = to_center / sqrt(dist2);
            vec3 leaf_nrm = cone[node].xyz;
            float cos_recv = max(dot(nrm, wi), 0.0);
            float cos_emit = max(dot(leaf_nrm, -wi), 0.0);
            incoming += rad_dfs[node].rgb * cos_emit * cos_recv / dist2 * u_leaf_area;
        } else {
            float angular_size = radius / max(sqrt(dist2), 1e-10);
            if (angular_size < 0.15) {
                if (dist2 < 1e-8) continue;
                vec3 wi = to_center / sqrt(dist2);
                vec3 node_axis = cn.xyz;
                float cos_recv = max(dot(nrm, wi), 0.0);
                float cos_emit = max(dot(node_axis, -wi), 0.0);
                float node_area = rad_dfs[node].w;
                incoming += rad_dfs[node].rgb * cos_emit * cos_recv / dist2 * node_area;
            } else {
                if (sp + 2 <= STACK_DEPTH) {
                    stack[sp++] = left_ch[node];
                    stack[sp++] = right_ch[node];
                }
            }
        }
    }

    rad_dst[u_tree_offset + recv] = vec4(emissive + alb / PI * incoming, u_leaf_area);
}
)";

const char* radiance_pullup_cs = R"(
#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 8) buffer Radiance { vec4 radiance[]; };

uniform uint u_count;
uniform uint u_level_start;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= u_count) return;

    uint node = u_level_start + idx;
    uint left  = 2 * node + 1;
    uint right = 2 * node + 2;

    float aL = radiance[left].w;
    float aR = radiance[right].w;
    float aTotal = aL + aR;

    vec3 avg = (radiance[left].rgb * aL + radiance[right].rgb * aR) / max(aTotal, 1e-10);
    radiance[node] = vec4(avg, aTotal);
}
)";

// ===========================================================================
// Stage E — Micro-rendering compute shader
// ===========================================================================

const char* micro_render_cs = R"(
#version 460 core
#extension GL_ARB_shader_clock : require
layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0) uniform sampler2D gbuf_pos;
layout(binding = 1) uniform sampler2D gbuf_nrm;
layout(binding = 2) uniform sampler2D gbuf_alb;
layout(binding = 3) uniform sampler2D u_warp_tex;

layout(std430, binding = 16) readonly buffer Spheres { vec4 sphere_buf[]; };
layout(std430, binding = 17) readonly buffer Cones   { vec4 cone_buf[];   };
layout(std430, binding = 18) readonly buffer LChild  { uint left_child[];  };
layout(std430, binding = 19) readonly buffer RChild  { uint right_child[]; };
layout(std430, binding = 20) readonly buffer Rad     { vec4 rad_buf[];    };
layout(std430, binding = 22) readonly buffer MortonTbl { uvec4 morton_tbl[]; };

layout(std430, binding = 10) writeonly buffer DebugPos { vec4 debug_pos[]; };
layout(std430, binding = 11) writeonly buffer DebugCol { vec4 debug_col[]; };
layout(std430, binding = 12) writeonly buffer ProfBuf { uint prof_data[]; };
layout(std430, binding = 15) writeonly buffer VisitStats { uvec4 visit_stats[]; };
// visit_stats: x = total nodes visited (DFS pops), y = max stack depth reached
//   (== STACK_DEPTH means the stack overflowed and children were dropped),
//   z = unused, w = unused.

layout(binding = 0, rgba16f) writeonly uniform image2D u_output;
layout(binding = 1, rgba16f) writeonly uniform image2D u_debug_img;

uniform uint u_num_leaves;
uniform ivec2 u_screen_size;
uniform uint u_micro_size;
uniform uint u_m_valid;
uniform uint u_scale;
uniform float u_gain;
uniform ivec2 u_debug_pixel;
uniform float u_self_eps2;
uniform int u_low_res_w;
uniform int u_low_res_h;
uniform float u_roughness;
uniform vec3 u_camera_pos;
uniform uint u_jitter_seed;

#define NPHASES 4
#define STACK_DEPTH 32
#define POST_LIST_CAP 16

void main() {
    ivec2 local = ivec2(gl_LocalInvocationID.xy);
    uint lid = uint(local.x) + uint(local.y) * 8u;

    // Space-filling curve dispatch: map workgroup index → tile via Morton table
    uvec4 morton_entry = morton_tbl[gl_WorkGroupID.x];
    ivec2 tile = ivec2(morton_entry.y, morton_entry.z);
    ivec2 pixel_lr = tile * 8 + local;

    if (pixel_lr.x >= u_low_res_w || pixel_lr.y >= u_low_res_h) return;

    ivec2 pixel = pixel_lr * int(u_scale);
    if (pixel.x >= u_screen_size.x || pixel.y >= u_screen_size.y) return;

    uint g_idx = uint(pixel_lr.x) + uint(pixel_lr.y) * uint(u_low_res_w);
    uint base = g_idx * NPHASES;

    // Jittered gather point within tile for bilateral upsample (Stage H)
    uint seed = u_jitter_seed + g_idx * 2654435761u;
    float jx = float(seed % 65536u) / 65536.0 * float(u_scale);
    float jy = float((seed / 65536u) % 65536u) / 65536.0 * float(u_scale);
    ivec2 pixel_sample = pixel_lr * int(u_scale) + ivec2(int(jx), int(jy));
    pixel_sample = min(pixel_sample, ivec2(u_screen_size) - 1);

    vec3 pos = texelFetch(gbuf_pos, pixel_sample, 0).xyz;
    vec3 nrm = normalize(texelFetch(gbuf_nrm, pixel_sample, 0).xyz);
    vec3 alb = texelFetch(gbuf_alb, pixel_sample, 0).rgb;

    if (dot(pos, pos) < 1e-6) {
        imageStore(u_output, pixel_lr, vec4(0.0));
        visit_stats[g_idx] = uvec4(0u);
        prof_data[base + 0] = uint(clockARB());
        prof_data[base + 1] = uint(clockARB());
        prof_data[base + 2] = uint(clockARB());
        prof_data[base + 3] = uint(clockARB());
        return;
    }

    vec3 up = abs(nrm.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, nrm));
    vec3 B = cross(nrm, T);

    // BRDF importance warping: blend tangent-frame pole toward reflection direction
    vec3 view_dir = normalize(u_camera_pos - pos);
    vec3 refl_dir = reflect(-view_dir, nrm);
    vec3 warp_dir = normalize(mix(nrm, refl_dir, 1.0 - u_roughness));
    vec3 up_w = abs(warp_dir.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 Tw = normalize(cross(up_w, warp_dir));
    vec3 Bw = cross(warp_dir, Tw);

    float half_ms = float(u_micro_size) * 0.5;
    float ms_f = float(u_micro_size);
    uint micro_total = u_micro_size * u_micro_size;

    prof_data[base + 0] = uint(clockARB());

    // ---- Per-thread micro-buffer (private arrays → local memory) ----
    // Each thread tracks the closest surfel for each micro-pixel independently.
    float my_depth[64];
    uint  my_node[64];
    for (uint i = 0u; i < 64u; i++) {
        my_depth[i] = 1e30;
        my_node[i] = 0u;
    }

    // ---- Stage F: Post-traversal leaf list for ray-disc fallback ----
    uint post_list[POST_LIST_CAP];
    uint post_count = 0u;

    // ---- Per-thread iterative DFS using DFS-ordered BVH (no shared state, no barriers) ----
    uint my_stack[STACK_DEPTH];
    int  my_sp = 0;
    my_stack[my_sp++] = 0u;

    uint visited = 0u;
    uint max_sp = 0u;

    while (my_sp > 0) {
        max_sp = max(max_sp, uint(my_sp));
        uint node = my_stack[--my_sp];
        visited++;

        vec4 sph = sphere_buf[node];
        vec3 center = sph.xyz;
        float radius = sph.w;

        vec3 to_center = center - pos;
        float dist2 = dot(to_center, to_center);

        if (dot(to_center, nrm) < 0.0) continue;

        vec4 cn = cone_buf[node];
        float dot_oc_axis = dot(to_center, cn.xyz);
        if (dot_oc_axis > 0.0 && cn.w > 0.0 &&
            dot_oc_axis * dot_oc_axis > dist2 * (1.0 - cn.w * cn.w))
            continue;

        float cz = dot(to_center, nrm);
        if (cz <= 0.0) continue;

        float distSafe2 = max(dist2, 1e-10);
        float invDist = inversesqrt(distSafe2);

        float dx = dot(to_center, Tw) * invDist;
        float dy = dot(to_center, Bw) * invDist;
        float cosTheta = dot(to_center, warp_dir) * invDist;

        float px_f = (dx + 1.0) * half_ms;
        float py_f = (dy + 1.0) * half_ms;

        float angularRadius = radius * invDist;
        float r_proj = min(angularRadius * half_ms, half_ms);

        if (px_f + r_proj < 0.0 || px_f - r_proj >= ms_f ||
            py_f + r_proj < 0.0 || py_f - r_proj >= ms_f)
            continue;

        // DFS-ordered leaf check: leaf has no children
        bool is_leaf = (left_child[node] == 0xFFFFFFFFu);
        if (is_leaf || r_proj <= 0.5) {
            float dist = sqrt(dist2);

            // Self-epsilon: only skip individual LEAVES too close to receiver,
            // NOT internal BVH nodes (which would cull entire subtrees).
            if (is_leaf && dist2 < u_self_eps2) continue;

            int ms = int(u_micro_size);
            int ir = max(1, int(ceil(r_proj)));
            int px_i = int(px_f);
            int py_i = int(py_f);
            int xmin = max(0, px_i - ir);
            int xmax = min(ms - 1, px_i + ir);
            int ymin = max(0, py_i - ir);
            int ymax = min(ms - 1, py_i + ir);
            float r2 = r_proj * r_proj;

            for (int my = ymin; my <= ymax; my++) {
                for (int mx = xmin; mx <= xmax; mx++) {
                    float ddx = float(mx) + 0.5 - px_f;
                    float ddy = float(my) + 0.5 - py_f;
                    if (ddx * ddx + ddy * ddy <= r2) {
                        uint mid = uint(mx) + uint(my) * u_micro_size;
                        if (dist < my_depth[mid]) {
                            my_depth[mid] = dist;
                            my_node[mid] = node;
                        }
                    }
                }
            }

            // Stage F: collect leaves into post-traversal list
            if (is_leaf && post_count < POST_LIST_CAP) {
                post_list[post_count++] = node;
            }
        } else {
            // DFS-ordered children via explicit child-pointer arrays
            uint lc = left_child[node];
            uint rc = right_child[node];
            if (my_sp + 2 <= STACK_DEPTH) {
                my_stack[my_sp++] = lc;
                my_stack[my_sp++] = rc;
            }
        }
    }

    prof_data[base + 1] = uint(clockARB());

    // ---- Stage F: Ray-disc intersection for unresolved micro-pixels ----
    // For any micro-pixel not hit by rasterization, ray-cast against collected leaves.
    for (uint i = 0u; i < micro_total; i++) {
        if (my_node[i] > 0u) continue;  // already resolved

        // Reconstruct ray direction from micro-pixel center via warp table
        float mu = (float(i % uint(u_micro_size)) + 0.5 - half_ms) / half_ms;
        float mv = (float(i / uint(u_micro_size)) + 0.5 - half_ms) / half_ms;
        vec2 warp_uv = vec2(mu, mv) * 0.5 + 0.5;
        vec3 warp_dir_tan = texture(u_warp_tex, warp_uv).xyz;
        vec3 ray_dir = normalize(Tw * warp_dir_tan.x + Bw * warp_dir_tan.y + warp_dir * warp_dir_tan.z);

        for (uint li = 0u; li < post_count; li++) {
            uint lnode = post_list[li];
            vec4 lsph = sphere_buf[lnode];
            vec3 lc = lsph.xyz;
            float lr = lsph.w;

            // Ray-sphere intersection
            vec3 oc = pos - lc;
            float b = dot(oc, ray_dir);
            float c = dot(oc, oc) - lr * lr;
            float disc = b * b - c;
            if (disc < 0.0) continue;

            float t = -b - sqrt(disc);
            if (t < 0.0) t = -b + sqrt(disc);
            if (t < 0.0) continue;

            vec3 hit = pos + ray_dir * t;

            // Oriented disc test: project onto plane, check radius
            vec4 lc_data = cone_buf[lnode];
            vec3 leaf_nrm = lc_data.xyz;
            float cosHalf = lc_data.w;
            float hdotn = dot(hit - lc, leaf_nrm);
            if (hdotn < 0.0) continue;
            vec3 proj = (hit - lc) - leaf_nrm * hdotn;
            if (dot(proj, proj) > lr * lr) continue;

            // Self-epsilon
            if (t * t < u_self_eps2) continue;

            if (t < my_depth[i]) {
                my_depth[i] = t;
                my_node[i] = lnode;
            }
        }
    }

    // ---- Read radiance for each micro-pixel's closest surfel, with Jacobian weighting ----
    vec3 sum = vec3(0.0);
    for (uint i = 0u; i < micro_total; i++) {
        if (my_node[i] > 0u) {
            float mu = (float(i % uint(u_micro_size)) + 0.5 - half_ms) / half_ms;
            float mv = (float(i / uint(u_micro_size)) + 0.5 - half_ms) / half_ms;
            vec2 warp_uv = vec2(mu, mv) * 0.5 + 0.5;
            float jacobian = texture(u_warp_tex, warp_uv).w;
            sum += rad_buf[my_node[i]].rgb * jacobian;
        }
    }

    prof_data[base + 2] = uint(clockARB());

    // ---- Debug pixel overlay ----
    if (u_debug_pixel.x >= 0 && pixel_lr == u_debug_pixel) {
        for (uint i = 0u; i < micro_total; i++) {
            vec3 r = my_node[i] > 0u ? rad_buf[my_node[i]].rgb : vec3(0.3, 0.0, 0.0);
            debug_col[i] = vec4(r, 1.0);
            debug_pos[i] = my_node[i] > 0u ? vec4(sphere_buf[my_node[i]].xyz, 1.0) : vec4(0.0);
        }
        // Write all 64 micro-pixel radiance values to the 8×8 debug image
        for (uint dy = 0u; dy < 8u; dy++) {
            for (uint dx = 0u; dx < 8u; dx++) {
                uint mid = dx + dy * 8u;
                vec3 r = my_node[mid] > 0u ? rad_buf[my_node[mid]].rgb : vec3(0.3, 0.0, 0.0);
                imageStore(u_debug_img, ivec2(dx, dy), vec4(r, 1.0));
            }
        }
    }

    // ---- Write output ----
    vec3 indirect = u_m_valid > 0u
        ? u_gain * alb * sum / float(u_m_valid)
        : vec3(0.0);
    imageStore(u_output, pixel_lr, vec4(indirect, 1.0));

    prof_data[base + 3] = uint(clockARB());

    visit_stats[g_idx] = uvec4(visited, max_sp, post_count, 0u);
}
)";

// ===========================================================================
// Stage H — Bilateral upsampling compute shader
// ===========================================================================

const char* bilateral_upsample_cs = R"(
#version 460 core
layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0, rgba16f) readonly  uniform image2D u_low_res;
layout(binding = 1, rgba16f) writeonly uniform image2D u_hi_res;

layout(binding = 0) uniform sampler2D gbuf_pos;
layout(binding = 1) uniform sampler2D gbuf_nrm;

uniform ivec2 u_low_size;
uniform ivec2 u_hi_size;
uniform float u_depth_sigma;
uniform float u_normal_exp;

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    if (px.x >= u_hi_size.x || px.y >= u_hi_size.y) return;

    vec2 uv = (vec2(px) + 0.5) / vec2(u_hi_size);
    vec3 pos_hi = texture(gbuf_pos, uv).xyz;
    vec3 nrm_hi = normalize(texture(gbuf_nrm, uv).xyz);

    // Map full-res pixel to low-res continuous coordinate
    vec2 low_f = (vec2(px) + 0.5) / vec2(u_hi_size) * vec2(u_low_size) - 0.5;
    ivec2 base = ivec2(floor(low_f));

    float depth_sq_inv = 1.0 / max(u_depth_sigma * u_depth_sigma, 1e-10);
    float normal_exp = u_normal_exp;

    vec3  sum  = vec3(0.0);
    float wsum = 0.0;

    for (int dy = -1; dy <= 2; dy++) {
        for (int dx = -1; dx <= 2; dx++) {
            ivec2 lpx = base + ivec2(dx, dy);
            if (lpx.x < 0 || lpx.x >= u_low_size.x ||
                lpx.y < 0 || lpx.y >= u_low_size.y) continue;

            vec3 val = imageLoad(u_low_res, lpx).rgb;
            float brightness = dot(val, vec3(0.2126, 0.7152, 0.0722));
            if (brightness < 1e-6) continue;

            // Bilateral weights
            vec2 diff = vec2(lpx) - low_f;
            float ws = exp(-dot(diff, diff) * 0.5);

            vec2 sample_uv = (vec2(lpx) + 0.5) / vec2(u_low_size);
            vec3 pos_low = texture(gbuf_pos, sample_uv).xyz;
            vec3 nrm_low = normalize(texture(gbuf_nrm, sample_uv).xyz);

            float depth_diff = length(pos_hi - pos_low);
            float wd = exp(-depth_diff * depth_diff * depth_sq_inv);

            float wn = pow(max(0.0, dot(nrm_hi, nrm_low)), normal_exp);

            float w = ws * wd * wn;
            sum  += val * w;
            wsum += w;
        }
    }

    vec3 result = wsum > 0.0 ? sum / wsum : vec3(0.0);
    imageStore(u_hi_res, px, vec4(result, 1.0));
}
)";

// ===========================================================================
// Stage C — Bounding sphere wireframe debug overlay
// ===========================================================================

const char* sphere_vs = R"(
#version 460 core
layout(location = 0) in vec3 a_sphere_vert;
layout(location = 1) in vec4 a_instance;   // center.xyz, radius

uniform mat4 u_view_proj;

void main() {
    vec3 center = a_instance.xyz;
    float radius = a_instance.w;
    gl_Position = u_view_proj * vec4(center + a_sphere_vert * radius, 1.0);
}
)";

const char* sphere_fs = R"(
#version 460 core
uniform vec4 u_color;
out vec4 frag_color;
void main() {
    frag_color = u_color;
}
)";

// ===========================================================================
// Stage E — Debug surfel overlay (64 colored dots for micro-buffer samples)
// ===========================================================================

const char* debug_surfel_vs = R"(
#version 460 core
layout(std430, binding = 10) readonly buffer DebugPos { vec4 d_pos[]; };
layout(std430, binding = 11) readonly buffer DebugCol { vec4 d_col[]; };
uniform mat4 u_view_proj;
out vec4 v_color;
void main() {
    v_color = d_col[gl_VertexID];
    gl_Position = u_view_proj * vec4(d_pos[gl_VertexID].xyz, 1.0);
    gl_PointSize = 10.0;
}
)";

const char* debug_surfel_fs = R"(
#version 460 core
in vec4 v_color;
out vec4 frag_color;
void main() {
    vec2 pc = gl_PointCoord * 2.0 - 1.0;
    if (dot(pc, pc) > 1.0) discard;
    frag_color = v_color;
}
)";

// ===========================================================================
// Orbit camera
// ===========================================================================
void camera_control(gfx::Window& w, gfx::Camera& cam, float dt, bool allow, bool& captured) {
    static double px0 = 0, py0 = 0;
    static bool first = true;

    bool rmb = w.mouse_down(gfx::MouseButton::right);
    if (allow && rmb) {
        if (!captured) {
            glfwSetInputMode((GLFWwindow*)w.native_handle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            captured = true;
            first = true;
        }
    } else if (captured) {
        glfwSetInputMode((GLFWwindow*)w.native_handle(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        captured = false;
    }

    if (captured) {
        double cx, cy;
        w.cursor_position(cx, cy);
        if (first) { px0 = cx; py0 = cy; first = false; }
        cam.orbit(float(cx - px0) * 0.008f, float(cy - py0) * 0.008f);
        px0 = cx;
        py0 = cy;
    }

    if (allow) {
        double scroll = w.scroll_delta();
        if (scroll != 0.0) cam.zoom(float(-scroll) * 0.05f);
    }

    float speed = 2.0f * dt;
    glm::vec3 fwd = glm::normalize(cam.target() - cam.position());
    glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0, 1, 0)));
    glm::vec3 vel(0.0f);
    if (w.key_down(gfx::Key::w)) vel += fwd;
    if (w.key_down(gfx::Key::s)) vel -= fwd;
    if (w.key_down(gfx::Key::a)) vel -= right;
    if (w.key_down(gfx::Key::d)) vel += right;
    if (w.key_down(gfx::Key::space)) vel.y += 1;
    if (w.key_down(gfx::Key::shift)) vel.y -= 1;
    if (glm::length(vel) > 0.0f) vel = glm::normalize(vel) * speed;
    if (glm::length(vel) > 0.0f || captured) cam.set_target(cam.target() + vel);

    cam.orbit(0.0f, 0.0f);
}

// ===========================================================================
// Stage B — Data structures
// ===========================================================================

constexpr float PI = 3.14159265358979f;

struct Tri {
    glm::vec3 p[3], n[3];
    float area;
    glm::vec3 alb, emi;
};

struct SamplePoint {
    glm::vec3 pos, nrm, alb, emi;
    int tri_idx;
    float u, v;  // barycentric
};

struct PointHierarchy {
    std::vector<SamplePoint> pts;
    std::vector<glm::vec4> sphere;    // (center.xyz, radius) — 2N-1 nodes
    std::vector<glm::vec4> cone;      // (axis.xyz, cos(half-angle))
    std::vector<uint32_t> nodeleaf;   // leftmost leaf index per node
    float leaf_radius = 0.0f;
    float total_area = 0.0f;
    int N = 0;
};

// GPU-friendly triangle layout (std430 compatible): 6 × vec4 = 96 bytes per tri
struct GpuTri {
    glm::vec4 pos[3];   // xyz = vertex position, w = 0
    glm::vec4 nrm[3];   // xyz = vertex normal,   w = 0
};

// Per-leaf source data for GPU refit: 16 bytes
struct LeafSrc {
    uint32_t tri_idx;
    float u, v, pad;
};

// ===========================================================================
// Stage B — Extract triangles from model
// ===========================================================================
std::vector<Tri> extract_triangles(const gfx::Model& model, const glm::mat4& model_mat) {
    std::vector<Tri> tris;
    for (size_t mi = 0; mi < model.mesh_count(); ++mi) {
        const gfx::Mesh& mesh = model.mesh(int(mi));
        int mati = model.mesh_material(int(mi));
        const auto& mat = model.material_info(size_t(mati >= 0 ? mati : 0));
        glm::vec3 alb(mat.base_color_factor[0], mat.base_color_factor[1], mat.base_color_factor[2]);
        glm::vec3 emi(mat.emissive_factor[0], mat.emissive_factor[1], mat.emissive_factor[2]);
        const auto& vs = mesh.vertices();
        const auto& is = mesh.indices();

        auto add_tri = [&](const gfx::Vertex& a, const gfx::Vertex& b, const gfx::Vertex& c) {
            glm::vec3 pa(a.position[0], a.position[1], a.position[2]);
            glm::vec3 pb(b.position[0], b.position[1], b.position[2]);
            glm::vec3 pc(c.position[0], c.position[1], c.position[2]);
            glm::vec3 na(a.normal[0], a.normal[1], a.normal[2]);
            glm::vec3 nb(b.normal[0], b.normal[1], b.normal[2]);
            glm::vec3 nc(c.normal[0], c.normal[1], c.normal[2]);
            float area = 0.5f * glm::length(glm::cross(pb - pa, pc - pa));
            if (area < 1e-9f) return;
            // Transform to world space
            pa = glm::vec3(model_mat * glm::vec4(pa, 1.0f));
            pb = glm::vec3(model_mat * glm::vec4(pb, 1.0f));
            pc = glm::vec3(model_mat * glm::vec4(pc, 1.0f));
            na = glm::normalize(glm::mat3(model_mat) * na);
            nb = glm::normalize(glm::mat3(model_mat) * nb);
            nc = glm::normalize(glm::mat3(model_mat) * nc);
            tris.push_back({{pa, pb, pc}, {na, nb, nc}, area, alb, emi});
        };

        if (is.empty()) {
            for (size_t v = 0; v + 2 < vs.size(); v += 3)
                add_tri(vs[v], vs[v + 1], vs[v + 2]);
        } else {
            for (size_t k = 0; k + 2 < is.size(); k += 3)
                add_tri(vs[is[k]], vs[is[k + 1]], vs[is[k + 2]]);
        }
    }
    return tris;
}

// ===========================================================================
// Stage B — Best-candidate sampling (Mitchell 1991)
//   K candidates per sample; pick the one with maximum minimum-distance to
//   any previously accepted sample.  Proper Poisson-disk distribution.
// ===========================================================================
SamplePoint sample_on_tri(const Tri& tri, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float su = dist(rng), sv = dist(rng);
    if (su + sv > 1.0f) { su = 1.0f - su; sv = 1.0f - sv; }
    float sw = 1.0f - su - sv;
    glm::vec3 pos = su * tri.p[0] + sv * tri.p[1] + sw * tri.p[2];
    glm::vec3 nrm = glm::normalize(su * tri.n[0] + sv * tri.n[1] + sw * tri.n[2]);
    return {pos, nrm, tri.alb, tri.emi, 0, su, sv};
}

std::vector<SamplePoint> sample_points_mitchell(
    const std::vector<Tri>& tris, float total_area, int N, int K)
{
    // Build CDF for area-proportional triangle selection
    std::vector<float> cdf(tris.size());
    float acc = 0.0f;
    for (size_t i = 0; i < tris.size(); ++i) {
        acc += tris[i].area;
        cdf[i] = acc;
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::uniform_int_distribution<int> tri_pick(0, int(tris.size()) - 1);

    std::vector<SamplePoint> accepted;
    accepted.reserve(N);

    // First sample: random
    {
        int ti = tri_pick(rng);
        SamplePoint sp = sample_on_tri(tris[ti], rng);
        sp.tri_idx = ti;
        accepted.push_back(sp);
    }

    for (int i = 1; i < N; ++i) {
        float best_dist = -1.0f;
        SamplePoint best_sp{};

        for (int k = 0; k < K; ++k) {
            // Pick a random triangle weighted by area
            float r = unit(rng) * total_area;
            int ti = int(std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin());
            ti = std::clamp(ti, 0, int(tris.size()) - 1);

            SamplePoint cand = sample_on_tri(tris[ti], rng);
            cand.tri_idx = ti;

            // Find minimum distance to any accepted sample
            float min_d = 1e30f;
            for (const auto& a : accepted) {
                float d2 = glm::dot(cand.pos - a.pos, cand.pos - a.pos);
                if (d2 < min_d) {
                    min_d = d2;
                    if (min_d < best_dist) break;  // early exit
                }
            }

            if (min_d > best_dist) {
                best_dist = min_d;
                best_sp = cand;
            }
        }

        accepted.push_back(best_sp);

        if ((i & 0xFFF) == 0) {
            gllib::logf(gllib::LogLevel::info, "  Mitchell sampling: %d / %d", i, N);
        }
    }

    gllib::logf(gllib::LogLevel::info, "  Mitchell sampling complete: %d points", N);
    return accepted;
}

// ===========================================================================
// Stage G — GGX BRDF importance-sampled warp table generation
//   For a given roughness alpha, tabulates the inverse CDF of the GGX NDF
//   to map uniform (u,v) → importance-sampled hemisphere direction + Jacobian.
//   For Lambertian BRDF: jacobian = (1/pi) * cos(theta_i) / p(omega)
//   where p(omega) = D(H)*cos(theta_H) / (4 * wo.H)
// ===========================================================================

void compute_brdf_warp_table(float alpha, int W, std::vector<float>& data) {
    data.resize(W * W * 4);
    float a2 = alpha * alpha;

    for (int j = 0; j < W; j++) {
        for (int i = 0; i < W; i++) {
            float u = (float(i) + 0.5f) / float(W);
            float v = (float(j) + 0.5f) / float(W);

            // GGX importance sampling: inverse CDF for polar angle of half-vector
            float cos_h = std::sqrt((1.0f - u) / (1.0f + (a2 - 1.0f) * u + 1e-10f));
            float sin_h = std::sqrt(std::max(0.0f, 1.0f - cos_h * cos_h));
            float phi_h = 2.0f * PI * v;

            // Half-vector in tangent space
            glm::vec3 H(sin_h * std::cos(phi_h), sin_h * std::sin(phi_h), cos_h);

            // View direction along +Z (tangent space)
            glm::vec3 Wo(0.0f, 0.0f, 1.0f);
            float VoH = glm::max(glm::dot(Wo, H), 0.0f);

            // Reflect view direction around half-vector
            glm::vec3 Wi = 2.0f * VoH * H - Wo;

            // Ensure upper hemisphere
            if (Wi.z < 0.0f) {
                Wi = -Wi;
                H = -H;
                VoH = glm::max(glm::dot(Wo, H), 0.0f);
            }

            // GGX NDF: D(H)
            float d = cos_h * cos_h * (a2 - 1.0f) + 1.0f;
            float D = a2 / (PI * d * d);

            // Sampling PDF: p(omega) = D(H) * cos(theta_H) / (4 * wo.H)
            float pdf = D * cos_h / (4.0f * VoH + 1e-10f);

            // For Lambertian BRDF (1/pi), Jacobian = f_r * cos(theta_i) / pdf
            float jacobian = (1.0f / PI) * glm::max(Wi.z, 0.0f) / (pdf + 1e-10f);

            int idx = (j * W + i) * 4;
            data[idx + 0] = Wi.x;
            data[idx + 1] = Wi.y;
            data[idx + 2] = Wi.z;
            data[idx + 3] = jacobian;
        }
    }
}

// ===========================================================================
// Stage B — Tree hierarchy build (recursive median split)
//   - N must be a power of two
//   - Leaf j at heap index (N-1)+j
//   - Interior nodes computed bottom-up: bounding sphere + normal cone
// ===========================================================================
PointHierarchy build_hierarchy(
    const std::vector<SamplePoint>& points, float coverage_factor)
{
    int N = int(points.size());
    PointHierarchy ph;
    ph.N = N;
    ph.pts = points;

    float total_area = 0.0f;
    for (auto& p : ph.pts) {
        (void)p;
    }
    // Compute total area from the distribution (approximate from leaf radius)
    // Actually, we'll compute it properly from the points' density
    // leaf_radius = coverage * sqrt(total_area / (N * PI))
    // We need total_area from the extraction step. Pass it in.
    // For now compute from average density:

    // Compute positions-only for sorting
    std::vector<glm::vec3> P(N);
    for (int i = 0; i < N; ++i) P[i] = ph.pts[i].pos;

    // Compute bounding box to derive leaf radius
    glm::vec3 mn(1e30f), mx(-1e30f);
    for (int i = 0; i < N; ++i) {
        mn = glm::min(mn, P[i]);
        mx = glm::max(mx, P[i]);
    }
    float diag = glm::length(mx - mn);
    ph.leaf_radius = coverage_factor * diag / std::sqrt(float(N));

    // Recursive median split to determine leaf order
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);

    std::function<void(int, int)> rec = [&](int lo, int hi) {
        if (hi - lo <= 1) return;
        glm::vec3 bmin(1e30f), bmax(-1e30f);
        for (int k = lo; k < hi; ++k) {
            bmin = glm::min(bmin, P[order[k]]);
            bmax = glm::max(bmax, P[order[k]]);
        }
        glm::vec3 ext = bmax - bmin;
        int axis = (ext.y > ext.x && ext.y > ext.z) ? 1
                 : ((ext.z > ext.x && ext.z > ext.y) ? 2 : 0);
        int mid = (lo + hi) / 2;
        std::nth_element(
            order.begin() + lo, order.begin() + mid, order.begin() + hi,
            [&](int a, int b) { return P[a][axis] < P[b][axis]; });
        rec(lo, mid);
        rec(mid, hi);
    };
    rec(0, N);

    // Permute points into leaf order
    std::vector<SamplePoint> ordered(N);
    for (int i = 0; i < N; ++i) ordered[i] = ph.pts[order[i]];
    ph.pts = std::move(ordered);

    // Build heap-layout tree: leaves at [N-1, 2N-2], interior at [0, N-2]
    int tree_size = 2 * N - 1;
    ph.sphere.assign(tree_size, glm::vec4(0.0f));
    ph.cone.assign(tree_size, glm::vec4(0.0f));
    ph.nodeleaf.assign(tree_size, 0u);

    // Initialize leaves
    for (int i = 0; i < N; ++i) {
        int node = (N - 1) + i;
        ph.sphere[node] = glm::vec4(ph.pts[i].pos, ph.leaf_radius);
        ph.cone[node] = glm::vec4(ph.pts[i].nrm, 1.0f);  // cos(0) = 1
        ph.nodeleaf[node] = uint32_t(i);
    }

    // Bottom-up refit using AABB-based tight bounding spheres
    // First pass: compute per-node AABBs from actual leaf positions
    std::vector<glm::vec3> bmin(tree_size, glm::vec3(1e30f));
    std::vector<glm::vec3> bmax(tree_size, glm::vec3(-1e30f));

    for (int i = 0; i < N; ++i) {
        int node = (N - 1) + i;
        bmin[node] = ph.pts[i].pos;
        bmax[node] = ph.pts[i].pos;
    }

    for (int node = N - 2; node >= 0; --node) {
        int l = 2 * node + 1;
        int r = 2 * node + 2;

        bmin[node] = glm::min(bmin[l], bmin[r]);
        bmax[node] = glm::max(bmax[l], bmax[r]);

        // Tight bounding sphere from AABB
        glm::vec3 center = (bmin[node] + bmax[node]) * 0.5f;
        float radius = glm::length(bmax[node] - bmin[node]) * 0.5f;
        ph.sphere[node] = glm::vec4(center, radius);

        // Normal cone merge
        glm::vec3 a1(ph.cone[l].x, ph.cone[l].y, ph.cone[l].z);
        glm::vec3 a2(ph.cone[r].x, ph.cone[r].y, ph.cone[r].z);
        float w1 = ph.cone[l].w, w2 = ph.cone[r].w;
        float b1 = std::acos(std::clamp(w1, -1.0f, 1.0f));
        float b2 = std::acos(std::clamp(w2, -1.0f, 1.0f));
        float ang = std::acos(std::clamp(glm::dot(a1, a2), -1.0f, 1.0f));
        glm::vec3 axis;
        float cw;
        if (ang + b1 + b2 >= PI - 1e-5f || glm::length(a1 + a2) < 1e-4f) {
            axis = a1;
            cw = -1.0f;
        } else {
            axis = glm::normalize(a1 + a2);
            cw = std::cos((ang + b1 + b2) * 0.5f);
        }
        ph.cone[node] = glm::vec4(axis, cw);
        ph.nodeleaf[node] = ph.nodeleaf[l];
    }

    return ph;
}

// ===========================================================================
// Stage B — DFS-reorder BVH for cache-friendly traversal
//   Reorders nodes so that DFS traversal visits them in sequential memory
//   order.  Also produces explicit child-pointer arrays (heap layout uses
//   2*i+1/2*i+2 addressing which is cache-hostile for DFS).
//   Leaves are identified by left_child == 0xFFFFFFFF.
// ===========================================================================
struct DfsBvh {
    std::vector<glm::vec4> sphere;       // DFS-reordered (tree_size)
    std::vector<glm::vec4> cone;         // DFS-reordered (tree_size)
    std::vector<uint32_t>  left_child;   // new-index of left child  (0xFFFFFFFF = leaf)
    std::vector<uint32_t>  right_child;  // new-index of right child
    std::vector<uint32_t>  new_to_old;   // DFS index → heap index mapping
    int tree_size = 0;
};

DfsBvh dfs_reorder_bvh(const PointHierarchy& ph) {
    int N = ph.N;
    int tree_size = 2 * N - 1;
    DfsBvh r;
    r.tree_size = tree_size;
    r.sphere.resize(tree_size);
    r.cone.resize(tree_size);
    r.left_child.resize(tree_size, 0xFFFFFFFFu);
    r.right_child.resize(tree_size, 0xFFFFFFFFu);
    r.new_to_old.resize(tree_size);

    std::vector<int> old_to_new(tree_size);
    int next = 0;

    // Pass 1: DFS to assign new sequential indices
    std::function<void(int)> dfs = [&](int old_node) {
        old_to_new[old_node] = next++;
        if (old_node < N - 1) {
            dfs(2 * old_node + 1);
            dfs(2 * old_node + 2);
        }
    };
    dfs(0);

    // Compute inverse mapping: new_to_old[dfs_idx] = heap_idx
    for (int i = 0; i < tree_size; i++) {
        r.new_to_old[old_to_new[i]] = uint32_t(i);
    }

    // Pass 2: copy data into new order, build child pointers
    for (int old_node = 0; old_node < tree_size; old_node++) {
        int new_node = old_to_new[old_node];
        r.sphere[new_node] = ph.sphere[old_node];
        r.cone[new_node]   = ph.cone[old_node];
        if (old_node < N - 1) {
            r.left_child[new_node]  = uint32_t(old_to_new[2 * old_node + 1]);
            r.right_child[new_node] = uint32_t(old_to_new[2 * old_node + 2]);
        }
    }

    gllib::logf(gllib::LogLevel::info,
        "DFS reorder: %d nodes, tree depth = %d",
        tree_size, int(std::ceil(std::log2(double(N)))));
    return r;
}

// ===========================================================================
// Cache save/load (binary)
// ===========================================================================

constexpr uint32_t CACHE_MAGIC   = 0x4D494352;  // "MICR"
constexpr uint32_t CACHE_VERSION = 2;

std::string cache_path_for(const std::string& model_name) {
    return model_name + ".micro_cache";
}

bool save_cache(const std::string& path, const PointHierarchy& ph,
                const std::vector<Tri>& tris)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    auto w32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto wf  = [&](float v)    { f.write(reinterpret_cast<const char*>(&v), 4); };

    w32(CACHE_MAGIC);
    w32(CACHE_VERSION);
    w32(uint32_t(ph.N));
    wf(ph.leaf_radius);
    wf(ph.total_area);
    w32(uint32_t(tris.size()));

    f.write(reinterpret_cast<const char*>(ph.pts.data()),
            ph.pts.size() * sizeof(SamplePoint));
    f.write(reinterpret_cast<const char*>(ph.sphere.data()),
            ph.sphere.size() * sizeof(glm::vec4));
    f.write(reinterpret_cast<const char*>(ph.cone.data()),
            ph.cone.size() * sizeof(glm::vec4));
    f.write(reinterpret_cast<const char*>(ph.nodeleaf.data()),
            ph.nodeleaf.size() * sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(tris.data()),
            tris.size() * sizeof(Tri));

    bool ok = f.good();
    if (ok) gllib::logf(gllib::LogLevel::info, "Cache: saved %s", path.c_str());
    return ok;
}

bool load_cache(const std::string& path, PointHierarchy& ph, std::vector<Tri>& tris) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    auto r32 = [&]() -> uint32_t {
        uint32_t v = 0; f.read(reinterpret_cast<char*>(&v), 4); return v;
    };
    auto rf = [&]() -> float {
        float v = 0; f.read(reinterpret_cast<char*>(&v), 4); return v;
    };

    if (r32() != CACHE_MAGIC) return false;
    if (r32() != CACHE_VERSION) {
        gllib::log(gllib::LogLevel::info, "Cache: version mismatch, rebuilding");
        return false;
    }

    int n = int(r32());
    if (n != N) {
        gllib::log(gllib::LogLevel::info, "Cache: N mismatch, rebuilding");
        return false;
    }

    ph.N = n;
    ph.leaf_radius = rf();
    ph.total_area = rf();
    int ntris = int(r32());

    ph.pts.resize(n);
    f.read(reinterpret_cast<char*>(ph.pts.data()), n * sizeof(SamplePoint));

    int tree_size = 2 * n - 1;
    ph.sphere.resize(tree_size);
    f.read(reinterpret_cast<char*>(ph.sphere.data()), tree_size * sizeof(glm::vec4));
    ph.cone.resize(tree_size);
    f.read(reinterpret_cast<char*>(ph.cone.data()), tree_size * sizeof(glm::vec4));
    ph.nodeleaf.resize(tree_size);
    f.read(reinterpret_cast<char*>(ph.nodeleaf.data()), tree_size * sizeof(uint32_t));

    tris.resize(ntris);
    f.read(reinterpret_cast<char*>(tris.data()), ntris * sizeof(Tri));

    if (!f.good()) return false;

    gllib::logf(gllib::LogLevel::info, "Cache: loaded %s (N=%d, lr=%.4f, ta=%.4f)",
                path.c_str(), n, ph.leaf_radius, ph.total_area);
    return true;
}

} // namespace

// ===========================================================================
// GPU / CPU profiling (double-buffered GL_TIME_ELAPSED queries)
// ===========================================================================

class PassTimer {
public:
    explicit PassTimer(const char* name, bool gpu = true)
        : name_(name), gpu_(gpu), q_(gl::QueryType::time_elapsed),
          q_prev_(gl::QueryType::time_elapsed) {}

    void begin() {
        t0_ = std::chrono::steady_clock::now();
        if (gpu_) q_.begin();
    }
    void end() {
        if (gpu_) q_.end();
        cpu_ms_ = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0_).count();
        ran_ = true;
        if (gpu_) std::swap(q_, q_prev_);
    }
    void skip() { ran_ = false; cpu_ms_ = 0.0; }

    void readback() {
        gpu_ms_ = 0.0;
        if (ran_) {
            if (gpu_) gpu_ms_ = double(q_prev_.result()) * 1.0e-6;
            cpu_acc_ += cpu_ms_;
            gpu_acc_ += gpu_ms_;
            n_++;
        }
        win_cpu_acc_ += cpu_ms_;
        win_gpu_acc_ += gpu_ms_;
        win_n_++;
    }

    void flush_window() {
        disp_cpu_ = win_n_ ? win_cpu_acc_ / double(win_n_) : 0.0;
        disp_gpu_ = win_n_ ? win_gpu_acc_ / double(win_n_) : 0.0;
        win_cpu_acc_ = win_gpu_acc_ = 0.0;
        win_n_ = 0;
        cpu_hist_.push_back(float(disp_cpu_));
        if (cpu_hist_.size() > kHist) cpu_hist_.erase(cpu_hist_.begin());
        if (gpu_) {
            gpu_hist_.push_back(float(disp_gpu_));
            if (gpu_hist_.size() > kHist) gpu_hist_.erase(gpu_hist_.begin());
        }
    }

    const char* name() const { return name_; }
    bool gpu() const { return gpu_; }
    double disp_cpu() const { return disp_cpu_; }
    double disp_gpu() const { return disp_gpu_; }
    const std::vector<float>& cpu_hist() const { return cpu_hist_; }
    const std::vector<float>& gpu_hist() const { return gpu_hist_; }

private:
    static constexpr size_t kHist = 120;
    const char* name_;
    bool gpu_;
    gl::Query q_, q_prev_;
    std::chrono::steady_clock::time_point t0_;
    double cpu_ms_ = 0.0, gpu_ms_ = 0.0;
    double cpu_acc_ = 0.0, gpu_acc_ = 0.0;
    double win_cpu_acc_ = 0.0, win_gpu_acc_ = 0.0;
    double disp_cpu_ = 0.0, disp_gpu_ = 0.0;
    int n_ = 0, win_n_ = 0;
    bool ran_ = false;
    std::vector<float> cpu_hist_, gpu_hist_;
};

static void imgui_stacked_bar(const ImVec2& pos, const ImVec2& size,
                              const float* vals, const ImU32* cols, int n) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float total = 0.0f;
    for (int i = 0; i < n; ++i) total += vals[i];
    if (total <= 0.0f) {
        dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y),
                          IM_COL32(40, 40, 40, 255));
        return;
    }
    float x = pos.x;
    for (int i = 0; i < n; ++i) {
        float w = size.x * vals[i] / total;
        if (w > 0.0f)
            dl->AddRectFilled(ImVec2(x, pos.y), ImVec2(x + w, pos.y + size.y), cols[i]);
        x += w;
    }
    dl->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y),
                IM_COL32(255, 255, 255, 90));
}

static void imgui_stacked_legend(const char* id, const char* const* names,
                                 const float* ms, const ImU32* cols, int n, float total) {
    if (ImGui::BeginTable(id, 2)) {
        for (int i = 0; i < n; ++i) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImVec4 c = ImGui::ColorConvertU32ToFloat4(cols[i]);
            ImGui::ColorButton("##sw", c,
                               ImGuiColorEditFlags_NoTooltip |
                               ImGuiColorEditFlags_NoInputs |
                               ImGuiColorEditFlags_NoPicker, ImVec2(10, 10));
            ImGui::SameLine();
            ImGui::Text("%s", names[i]);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f ms  (%.1f%%)", ms[i],
                        total > 0.0f ? 100.0f * ms[i] / total : 0.0f);
        }
        ImGui::EndTable();
    }
}

// ===========================================================================
// Stage C — Unit sphere mesh for wireframe bounding-sphere debug rendering
// ===========================================================================

struct SphereMesh {
    gl::VertexArray vao;
    gl::Buffer vbo, ebo;
    int index_count = 0;
};

SphereMesh create_wireframe_sphere(int lat_seg, int lon_seg) {
    SphereMesh sm;

    std::vector<glm::vec3> verts;
    std::vector<uint32_t> idx;

    // Top pole = index 0
    verts.emplace_back(0.0f, 1.0f, 0.0f);
    for (int i = 1; i < lat_seg; ++i) {
        float phi = PI * float(i) / float(lat_seg);
        float sp = std::sin(phi), cp = std::cos(phi);
        for (int j = 0; j < lon_seg; ++j) {
            float theta = 2.0f * PI * float(j) / float(lon_seg);
            verts.emplace_back(sp * std::cos(theta), cp, sp * std::sin(theta));
        }
    }
    // Bottom pole
    verts.emplace_back(0.0f, -1.0f, 0.0f);

    // Top cap
    for (int j = 0; j < lon_seg; ++j) {
        idx.push_back(0);
        idx.push_back(1 + j);
        idx.push_back(1 + (j + 1) % lon_seg);
    }
    // Middle bands
    for (int i = 0; i < lat_seg - 2; ++i) {
        int row0 = 1 + i * lon_seg;
        int row1 = 1 + (i + 1) * lon_seg;
        for (int j = 0; j < lon_seg; ++j) {
            int j1 = (j + 1) % lon_seg;
            idx.push_back(row0 + j);
            idx.push_back(row0 + j1);
            idx.push_back(row1 + j1);
            idx.push_back(row0 + j);
            idx.push_back(row1 + j1);
            idx.push_back(row1 + j);
        }
    }
    // Bottom cap
    int bp = 1 + (lat_seg - 1) * lon_seg;
    for (int j = 0; j < lon_seg; ++j) {
        idx.push_back(bp);
        idx.push_back(bp + (j + 1) % lon_seg);
        idx.push_back(bp + j);
    }

    sm.index_count = int(idx.size());

    sm.vbo = gl::Buffer(gl::BufferType::vertex, gl::BufferUsage::static_draw);
    sm.vbo.data(verts.data(), verts.size() * sizeof(glm::vec3));
    sm.ebo = gl::Buffer(gl::BufferType::index, gl::BufferUsage::static_draw);
    sm.ebo.data(idx.data(), idx.size() * sizeof(uint32_t));

    return sm;
}

void setup_sphere_vao(SphereMesh& sm, const gl::Buffer& instance_buf) {
    sm.vao.bind();

    // Attrib 0: unit sphere vertices
    sm.vbo.bind();
    sm.vao.attrib_pointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (const void*)0);
    sm.vao.enable_attrib(0);

    // Attrib 1: per-instance data from sphere_buf (center.xyz, radius)
    instance_buf.bind();
    sm.vao.attrib_pointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (const void*)0);
    sm.vao.enable_attrib(1);
    glVertexAttribDivisor(1, 1);

    // Element buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sm.ebo.handle());

    gl::VertexArray::unbind();
}

// ===========================================================================
// G-buffer FBO
// ===========================================================================
struct GBuffer {
    int w = 0, h = 0;
    gl::Texture albedo{gl::TextureType::tex_2d};
    gl::Texture normal{gl::TextureType::tex_2d};
    gl::Texture position{gl::TextureType::tex_2d};
    gl::Texture emissive{gl::TextureType::tex_2d};
    gl::Texture depth{gl::TextureType::tex_2d};
    gl::Renderbuffer depth_rbo;
    gl::Framebuffer fbo;
};

void create_gbuffer(GBuffer& g, int w, int h) {
    g.w = w;
    g.h = h;

    auto tex2 = [&](gl::Texture& t, GLenum internal) {
        t.image_2d(0, internal, w, h,
                   internal == GL_RGBA16F ? GL_RGBA : GL_RED,
                   internal == GL_RGBA16F ? GL_FLOAT : GL_FLOAT, nullptr, 1);
        t.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        t.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        t.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        t.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };

    tex2(g.albedo,   GL_RGBA16F);
    tex2(g.normal,   GL_RGBA16F);
    tex2(g.position, GL_RGBA16F);
    tex2(g.emissive, GL_RGBA16F);
    tex2(g.depth,    GL_R32F);

    g.depth_rbo.storage(GL_DEPTH_COMPONENT24, w, h);

    g.fbo.bind();
    g.fbo.attach_texture(GL_COLOR_ATTACHMENT0, g.albedo);
    g.fbo.attach_texture(GL_COLOR_ATTACHMENT1, g.normal);
    g.fbo.attach_texture(GL_COLOR_ATTACHMENT2, g.position);
    g.fbo.attach_texture(GL_COLOR_ATTACHMENT3, g.emissive);
    g.fbo.attach_texture(GL_COLOR_ATTACHMENT4, g.depth);
    g.fbo.attach_renderbuffer(GL_DEPTH_ATTACHMENT, g.depth_rbo);
    GLenum bufs[5] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
                      GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3,
                      GL_COLOR_ATTACHMENT4};
    glDrawBuffers(5, bufs);
    if (!g.fbo.check())
        gllib::log(gllib::LogLevel::error, "G-buffer framebuffer incomplete");
    gl::Framebuffer::unbind(gl::FramebufferType::both);
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    gllib::log_to_stderr(gllib::LogLevel::info);
    gfx::Window window({"36 Micro Rendering — Stage A+B+C+D+E", 1600, 900});
    window.vsync(false);

    gfx::ImGuiOverlay gui;
    if (!gui.init(window)) {
        gllib::log(gllib::LogLevel::error, "ImGui init failed");
        return EXIT_FAILURE;
    }

    // --- Compile shaders ---
    auto make_program = [](const char* vs_src, const char* fs_src) -> gl::Program {
        gl::Shader vs(gl::ShaderType::vertex, vs_src);
        gl::Shader fs(gl::ShaderType::fragment, fs_src);
        gl::Program prog;
        prog.attach(vs);
        prog.attach(fs);
        if (!prog.link())
            gllib::log(gllib::LogLevel::error, prog.info_log().c_str());
        return prog;
    };

    gl::Program gbuf_prog = make_program(gbuf_vs, gbuf_fs);
    gl::Program display_prog = make_program(display_vs, display_fs);
    gl::Program pc_prog = make_program(pc_vs, pc_fs);

    // Stage C — compute shaders
    auto make_compute = [](const char* src) -> gl::Program {
        gl::Shader cs(gl::ShaderType::compute, src);
        gl::Program prog;
        prog.attach(cs);
        if (!prog.link())
            gllib::log(gllib::LogLevel::error, prog.info_log().c_str());
        return prog;
    };
    gl::Program leaf_update_prog = make_compute(leaf_update_cs);
    gl::Program tree_refit_prog  = make_compute(tree_refit_cs);
    gl::Program sph_reorder_prog = make_compute(sphere_reorder_cs);
    gl::Program rad_reorder_prog = make_compute(radiance_reorder_cs);
    gl::Program direct_light_prog = make_compute(direct_lighting_cs);
    gl::Program bounce_gather_prog = make_compute(hierarchical_bounce_cs);
    gl::Program radiance_pull_prog = make_compute(radiance_pullup_cs);
    gl::Program micro_render_prog  = make_compute(micro_render_cs);
    gl::Program bilateral_prog     = make_compute(bilateral_upsample_cs);
    gl::Program sphere_prog      = make_program(sphere_vs, sphere_fs);
    gl::Program debug_surfel_prog = make_program(debug_surfel_vs, debug_surfel_fs);

    if (!gbuf_prog.linked() || !display_prog.linked() || !pc_prog.linked() ||
        !leaf_update_prog.linked() || !tree_refit_prog.linked() ||
        !sph_reorder_prog.linked() || !rad_reorder_prog.linked() ||
        !direct_light_prog.linked() || !bounce_gather_prog.linked() || !radiance_pull_prog.linked() ||
        !micro_render_prog.linked() || !bilateral_prog.linked() || !sphere_prog.linked() ||
        !debug_surfel_prog.linked()) {
        gllib::log(gllib::LogLevel::error, "Shader compilation failed");
        return EXIT_FAILURE;
    }

    // --- Load model ---
    gfx::Model model;
    if (!model.load("CornellBoxOriginal.glb")) {
        gllib::log(gllib::LogLevel::error, "Failed to load CornellBoxOriginal.glb");
        return EXIT_FAILURE;
    }
    gllib::logf(gllib::LogLevel::info, "Loaded: %zu meshes, %zu materials",
                model.mesh_count(), model.material_count());

    glm::mat4 model_mat = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));

    // --- Scene bounds → camera placement ---
    glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);
    for (size_t i = 0; i < model.mesh_count(); ++i) {
        glm::vec4 bs = model.mesh_bounding_sphere(i);
        glm::vec4 c = model_mat * glm::vec4(bs.x, bs.y, bs.z, 1.0f);
        for (int j = 0; j < 3; ++j) {
            lo[j] = std::min(lo[j], c[j] - bs.w);
            hi[j] = std::max(hi[j], c[j] + bs.w);
        }
    }
    glm::vec3 cam_target = (lo + hi) * 0.5f;
    float cam_dist = 0.5f * glm::length(hi - lo) + 1.8f;

    gfx::Camera cam;
    cam.perspective(45.0f, float(window.framebuffer_width()) / float(window.framebuffer_height()),
                    0.1f, 1000.0f);
    cam.look_at(cam_target + glm::vec3(0, 0, cam_dist), cam_target);

    // ===================================================================
    // Stage B — Build or load point hierarchy
    // ===================================================================
    std::string cache_file = cache_path_for("CornellBoxOriginal.glb");
    PointHierarchy ph;
    std::vector<Tri> tris;
    double build_ms = 0.0;
    bool cached = load_cache(cache_file, ph, tris);

    if (!cached) {
        gllib::log(gllib::LogLevel::info, "Extracting triangles...");
        tris = extract_triangles(model, model_mat);
        float total_area = 0.0f;
        for (const auto& t : tris) total_area += t.area;
        gllib::logf(gllib::LogLevel::info, "  %zu triangles, total area = %.4f",
                    tris.size(), total_area);

        gllib::logf(gllib::LogLevel::info, "Building hierarchy: N=%d, K=%d, coverage=%.2f",
                    N, BEST_K, COVERAGE);

        auto t0 = std::chrono::steady_clock::now();
        auto sample_pts = sample_points_mitchell(tris, total_area, N, BEST_K);
        ph = build_hierarchy(sample_pts, COVERAGE);
        auto t1 = std::chrono::steady_clock::now();
        build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        gllib::logf(gllib::LogLevel::info, "  Hierarchy built in %.1f ms  leaf_radius=%.6f",
                    build_ms, ph.leaf_radius);
        ph.total_area = total_area;

        save_cache(cache_file, ph, tris);
    }

    // --- Upload point data to SSBOs ---
    std::vector<glm::vec4> vp(N), vn(N), va(N), ve(N);
    for (int i = 0; i < N; ++i) {
        const auto& p = ph.pts[i];
        vp[i] = glm::vec4(p.pos, ph.leaf_radius);
        vn[i] = glm::vec4(p.nrm, 0.0f);
        va[i] = glm::vec4(p.alb, 1.0f);
        ve[i] = glm::vec4(p.emi, 1.0f);
    }

    gl::Buffer pgeom_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer pnrm_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer palb_buf(gl::BufferType::shader, gl::BufferUsage::static_draw);
    gl::Buffer pemit_buf(gl::BufferType::shader, gl::BufferUsage::static_draw);
    gl::Buffer sphere_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer cone_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);

    // AABB buffers for tight bounding sphere computation (binding 13, 14)
    int tree_size = 2 * N - 1;
    std::vector<glm::vec4> aabb_min_data(tree_size, glm::vec4(1e30f));
    std::vector<glm::vec4> aabb_max_data(tree_size, glm::vec4(-1e30f));
    for (int i = 0; i < N; ++i) {
        int node = (N - 1) + i;
        aabb_min_data[node] = glm::vec4(ph.pts[i].pos, 0.0f);
        aabb_max_data[node] = glm::vec4(ph.pts[i].pos, 0.0f);
    }
    for (int node = N - 2; node >= 0; --node) {
        int l = 2 * node + 1, r = 2 * node + 2;
        aabb_min_data[node] = glm::vec4(
            glm::min(glm::vec3(aabb_min_data[l]), glm::vec3(aabb_min_data[r])), 0.0f);
        aabb_max_data[node] = glm::vec4(
            glm::max(glm::vec3(aabb_max_data[l]), glm::vec3(aabb_max_data[r])), 0.0f);
    }
    gl::Buffer aabb_min_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer aabb_max_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    aabb_min_buf.data(aabb_min_data.data(), aabb_min_data.size() * sizeof(glm::vec4));
    aabb_max_buf.data(aabb_max_data.data(), aabb_max_data.size() * sizeof(glm::vec4));

    pgeom_buf.data(vp.data(), vp.size() * sizeof(glm::vec4));
    pnrm_buf.data(vn.data(), vn.size() * sizeof(glm::vec4));
    palb_buf.data(va.data(), va.size() * sizeof(glm::vec4));
    pemit_buf.data(ve.data(), ve.size() * sizeof(glm::vec4));
    sphere_buf.data(ph.sphere.data(), ph.sphere.size() * sizeof(glm::vec4));
    cone_buf.data(ph.cone.data(), ph.cone.size() * sizeof(glm::vec4));

    // --- Stage C: Upload TriangleBuf + LeafSourceBuf ---
    std::vector<GpuTri> gpu_tris(tris.size());
    for (size_t i = 0; i < tris.size(); ++i) {
        for (int v = 0; v < 3; ++v) {
            gpu_tris[i].pos[v] = glm::vec4(tris[i].p[v], 0.0f);
            gpu_tris[i].nrm[v] = glm::vec4(tris[i].n[v], 0.0f);
        }
    }
    gl::Buffer tri_buf(gl::BufferType::shader, gl::BufferUsage::static_draw);
    tri_buf.data(gpu_tris.data(), gpu_tris.size() * sizeof(GpuTri));

    std::vector<LeafSrc> leaf_src(N);
    for (int i = 0; i < N; ++i) {
        leaf_src[i] = { uint32_t(ph.pts[i].tri_idx), ph.pts[i].u, ph.pts[i].v, 0.0f };
    }
    gl::Buffer leaf_src_buf(gl::BufferType::shader, gl::BufferUsage::static_draw);
    leaf_src_buf.data(leaf_src.data(), leaf_src.size() * sizeof(LeafSrc));

    gllib::logf(gllib::LogLevel::info, "Stage C: uploaded %zu triangles (%zu bytes), %d leaf sources",
                tris.size(), gpu_tris.size() * sizeof(GpuTri), N);

    // Bind sphere_buf and cone_buf to fixed binding points for compute + wireframe
    sphere_buf.bind_base(6);
    cone_buf.bind_base(7);

    // --- DFS-reorder BVH for cache-friendly traversal ---
    DfsBvh dfs_bvh = dfs_reorder_bvh(ph);

    gl::Buffer dfs_sphere_buf(gl::BufferType::shader, gl::BufferUsage::static_draw);
    gl::Buffer dfs_cone_buf(gl::BufferType::shader, gl::BufferUsage::static_draw);
    gl::Buffer dfs_left_buf(gl::BufferType::shader, gl::BufferUsage::static_draw);
    gl::Buffer dfs_right_buf(gl::BufferType::shader, gl::BufferUsage::static_draw);

    dfs_sphere_buf.data(dfs_bvh.sphere.data(), dfs_bvh.sphere.size() * sizeof(glm::vec4));
    dfs_cone_buf.data(dfs_bvh.cone.data(), dfs_bvh.cone.size() * sizeof(glm::vec4));
    dfs_left_buf.data(dfs_bvh.left_child.data(), dfs_bvh.left_child.size() * sizeof(uint32_t));
    dfs_right_buf.data(dfs_bvh.right_child.data(), dfs_bvh.right_child.size() * sizeof(uint32_t));

    dfs_sphere_buf.bind_base(16);
    dfs_cone_buf.bind_base(17);
    dfs_left_buf.bind_base(18);
    dfs_right_buf.bind_base(19);

    // --- New_to_old mapping SSBO (binding 21) ---
    gl::Buffer new_to_old_buf(gl::BufferType::shader, gl::BufferUsage::static_draw);
    new_to_old_buf.data(dfs_bvh.new_to_old.data(), dfs_bvh.new_to_old.size() * sizeof(uint32_t));
    new_to_old_buf.bind_base(21);

    // --- DFS-ordered radiance buffer (binding 20) ---
    gl::Buffer dfs_rad_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    dfs_rad_buf.data(nullptr, size_t(2 * N - 1) * sizeof(glm::vec4));
    dfs_rad_buf.bind_base(20);

    // --- Morton dispatch table SSBO (binding 22) ---
    gl::Buffer morton_buf(gl::BufferType::shader, gl::BufferUsage::static_draw);

    // --- Stage D: Extract emitter leaf indices + create radiance buffer ---
    std::vector<uint32_t> emitter_indices;
    for (int i = 0; i < N; ++i) {
        if (glm::dot(ph.pts[i].emi, ph.pts[i].emi) > 1e-6f)
            emitter_indices.push_back(uint32_t(i));
    }
    gllib::logf(gllib::LogLevel::info, "Stage D: %zu emitter leaves out of %d",
                emitter_indices.size(), N);

    gl::Buffer emitters_buf(gl::BufferType::shader, gl::BufferUsage::static_draw);
    emitters_buf.data(emitter_indices.data(), emitter_indices.size() * sizeof(uint32_t));
    emitters_buf.bind_base(9);

    gl::Buffer radiance_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    radiance_buf.data(nullptr, size_t(2 * N - 1) * sizeof(glm::vec4));
    radiance_buf.bind_base(8);

    // Ping-pong buffer for multi-bounce gather
    gl::Buffer radiance_b_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    radiance_b_buf.data(nullptr, size_t(2 * N - 1) * sizeof(glm::vec4));

    int num_bounces = 2;

    float leaf_area = (emitter_indices.empty() || N == 0) ? 0.0f : ph.total_area / float(N);

    // --- Empty VAO for point cloud (shader reads from SSBO via gl_VertexID) ---
    gl::VertexArray pc_vao;

    // --- Stage C: wireframe bounding sphere debug mesh ---
    SphereMesh sphere_mesh = create_wireframe_sphere(8, 16);
    setup_sphere_vao(sphere_mesh, sphere_buf);

    // --- G-buffer ---
    GBuffer gbuf;
    create_gbuffer(gbuf, window.framebuffer_width(), window.framebuffer_height());

    // --- Indirect illumination texture (Stage E) ---
    int micro_res_scale = 1;
    gl::Texture indirect_tex{gl::TextureType::tex_2d};
    auto create_indirect_tex = [&](int w, int h) {
        int rw = std::max(1, w / micro_res_scale);
        int rh = std::max(1, h / micro_res_scale);
        indirect_tex.image_2d(0, GL_RGBA16F, rw, rh, GL_RGBA, GL_FLOAT, nullptr, 1);
        indirect_tex.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        indirect_tex.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        indirect_tex.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        indirect_tex.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };
    create_indirect_tex(window.framebuffer_width(), window.framebuffer_height());

    // --- Full-res bilateral upsample output texture (Stage H) ---
    gl::Texture indirect_upsampled_tex{gl::TextureType::tex_2d};
    auto create_upsampled_tex = [&](int w, int h) {
        indirect_upsampled_tex.image_2d(0, GL_RGBA16F, w, h, GL_RGBA, GL_FLOAT, nullptr, 1);
        indirect_upsampled_tex.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        indirect_upsampled_tex.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        indirect_upsampled_tex.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        indirect_upsampled_tex.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };
    create_upsampled_tex(window.framebuffer_width(), window.framebuffer_height());

    bool use_bilateral_upsample = true;
    float bilateral_depth_sigma = 0.01f;
    float bilateral_normal_exp = 32.0f;

    // --- Debug micro-buffer texture (8×8) ---
    gl::Texture debug_tex{gl::TextureType::tex_2d};
    debug_tex.image_2d(0, GL_RGBA16F, 8, 8, GL_RGBA, GL_FLOAT, nullptr, 1);
    debug_tex.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    debug_tex.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    debug_tex.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    debug_tex.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // --- BRDF importance warp table (64×64 RGBA32F) ---
    // GGX importance-sampled warp: maps (u,v) to hemisphere direction + Jacobian.
    // For roughness=1 (alpha=1), this degenerates to cosine-weighted hemisphere.
    gl::Texture warp_tex{gl::TextureType::tex_2d};
    constexpr int WARP_SIZE = 64;
    std::vector<float> warp_data(WARP_SIZE * WARP_SIZE * 4);
    compute_brdf_warp_table(1.0f, WARP_SIZE, warp_data);
    warp_tex.image_2d(0, GL_RGBA32F, WARP_SIZE, WARP_SIZE, GL_RGBA, GL_FLOAT, warp_data.data(), 1);
    warp_tex.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    warp_tex.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    warp_tex.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    warp_tex.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // --- Debug surfel SSBOs (64 samples for micro-buffer overlay) ---
    gl::Buffer debug_pos_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    debug_pos_buf.data(nullptr, 64 * sizeof(glm::vec4));
    gl::Buffer debug_col_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    debug_col_buf.data(nullptr, 64 * sizeof(glm::vec4));

    // --- Micro-rendering intra-shader profiling SSBO ---
    gl::Buffer prof_buf(gl::BufferType::shader, gl::BufferUsage::stream_read);
    std::vector<uint32_t> prof_readback;

    gl::Buffer visit_stats_buf(gl::BufferType::shader, gl::BufferUsage::stream_read);
    std::vector<uint32_t> visit_stats_readback; // uvec4 per workgroup, flattened
    double visit_avg = 0.0, visit_max = 0.0;
    double iters_avg = 0.0, iters_maxed_pct = 0.0; // % of pixels that hit MAX_BFS_ITERS
    double frontier_avg = 0.0, frontier_saturated_pct = 0.0; // % that hit Q_CAP

    // --- Fullscreen triangle VAO ---
    gl::VertexArray fsq_vao;

    // --- State ---
    int view_mode = 0;
    float exposure = 1.0f;
    float gamma = 2.2f;
    const float far_plane = 1000.0f;
    bool captured = false;

    // Point cloud debug state
    bool show_points = true;
    float point_size = 5.0f;
    int pc_color_mode = 0;  // 0=albedo 1=emissive 2=normal 3=position

    // Stage C debug state
    bool show_spheres = false;
    bool run_refit = true;
    float max_pos_err = 0.0f;
    float max_cone_err = 0.0f;
    glm::vec4 sphere_color(0.0f, 1.0f, 0.0f, 0.3f);
    int sphere_lod = 0;  // 0=all, 1=leaves only, 2=interior only

    // Stage D state
    float emissive_gain = 8.0f;

    // Stage E state
    bool run_micro_render = true;
    int micro_size = 8;
    float micro_gain = 1.0f;
    float micro_roughness = 1.0f;
    float last_roughness = -1.0f;  // track for warp table regeneration
    int tile_size = 2;
    bool show_micro_debug = false;
    uint64_t frame_counter = 0;
    // Even with the debug panel open, only pay for the profiling stall
    // periodically -- readback data doesn't need to be per-frame to be useful,
    // and this keeps the debug view from itself becoming the bottleneck.
    constexpr uint64_t kProfileEveryNFrames = 30;
    int debug_pixel_x = -1, debug_pixel_y = -1;

    // Intra-shader micro profiling results (ticks, displayed as %)
    static const char* phase_names[] = {"traversal", "radiance", "accumulate"};
    double phase_avg[3] = {};
    double phase_pct[3] = {};
    uint64_t prof_clock_period_ns = 1;

    // Profiling timers
    PassTimer t_frame("frame", false);
    PassTimer t_geo("geometry");
    PassTimer t_refit("gpu refit");
    PassTimer t_direct("direct light");
    PassTimer t_micro("micro render");
    PassTimer t_display("display");
    PassTimer t_pointcloud("point cloud");
    PassTimer t_imgui("imgui", false);
    double win_start = window.time();

    double last = window.time();

    while (!window.should_close()) {
        ++frame_counter;
        const bool profile_this_frame = (frame_counter % kProfileEveryNFrames) == 0;
        double now = window.time();
        float dt = float(now - last);
        last = now;

        window.poll_events();
        camera_control(window, cam, dt, !gui.wants_mouse(), captured);
        cam.set_aspect(float(window.framebuffer_width()) / float(window.framebuffer_height()));

        int fw = window.framebuffer_width(), fh = window.framebuffer_height();
        static int prev_scale = micro_res_scale;
        if (fw != gbuf.w || fh != gbuf.h || micro_res_scale != prev_scale) {
            create_gbuffer(gbuf, fw, fh);
            create_indirect_tex(fw, fh);
            create_upsampled_tex(fw, fh);
            prev_scale = micro_res_scale;
        }

        glm::mat4 vp = cam.view_projection();

        t_frame.begin();

        // ===================================================================
        // 1. Geometry pass
        // ===================================================================
        t_geo.begin();
        gbuf.fbo.bind();
        gl::viewport(0, 0, gbuf.w, gbuf.h);
        gl::clear_color(0.0f, 0.0f, 0.0f, 0.0f);
        gl::clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        gl::enable(GL_DEPTH_TEST);
        gl::depth_func(GL_LESS);

        gbuf_prog.use();
        auto g_loc = [&](const char* n) { return gbuf_prog.uniform_location(n); };
        GLint loc;
        loc = g_loc("u_view_proj"); if (loc >= 0) gbuf_prog.uniform_matrix4fv(loc, glm::value_ptr(vp));
        loc = g_loc("u_view");      if (loc >= 0) gbuf_prog.uniform_matrix4fv(loc, glm::value_ptr(cam.view()));
        loc = g_loc("u_model");     if (loc >= 0) gbuf_prog.uniform_matrix4fv(loc, glm::value_ptr(model_mat));
        glm::mat3 normal_mat = glm::transpose(glm::inverse(glm::mat3(model_mat)));
        loc = g_loc("u_normal_mat"); if (loc >= 0) gbuf_prog.uniform_matrix3fv(loc, glm::value_ptr(normal_mat));

        for (size_t i = 0; i < model.mesh_count(); ++i) {
            int mi = model.mesh_material(int(i));
            const auto& mat = model.material_info(size_t(mi >= 0 ? mi : 0));
            loc = g_loc("u_albedo");   if (loc >= 0) gbuf_prog.uniform3fv(loc, mat.base_color_factor);
            loc = g_loc("u_emissive"); if (loc >= 0) gbuf_prog.uniform3fv(loc, mat.emissive_factor);
            model.mesh(i).draw();
        }

        gl::Framebuffer::unbind(gl::FramebufferType::both);
        glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
        t_geo.end();

    // ===================================================================
        // 1b. Stage C — GPU refit: leaf update + bottom-up tree refit
        // ===================================================================
        t_refit.begin();
        if (run_refit) {
            // Bind compute-writeable buffers
            pgeom_buf.bind_base(0);
            pnrm_buf.bind_base(1);
            palb_buf.bind_base(2);
            pemit_buf.bind_base(3);
            tri_buf.bind_base(4);
            leaf_src_buf.bind_base(5);
            sphere_buf.bind_base(6);
            cone_buf.bind_base(7);
            aabb_min_buf.bind_base(13);
            aabb_max_buf.bind_base(14);

            // Pass 1: Leaf update — re-evaluate positions from triangle barycentrics
            leaf_update_prog.use();
            GLint loc_c;
            loc_c = leaf_update_prog.uniform_location("u_num_leaves");
            if (loc_c >= 0) glProgramUniform1ui(leaf_update_prog.handle(), loc_c, GLuint(N));
            loc_c = leaf_update_prog.uniform_location("u_leaf_radius");
            if (loc_c >= 0) glProgramUniform1f(leaf_update_prog.handle(), loc_c, ph.leaf_radius);
            loc_c = leaf_update_prog.uniform_location("u_tree_offset");
            if (loc_c >= 0) glProgramUniform1i(leaf_update_prog.handle(), loc_c, N - 1);
            gl::dispatch_compute((N + 255) / 256, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            // Pass 2: Bottom-up refit — log2(N) levels, from leaves to root
            tree_refit_prog.use();
            for (int level = 0; level < LOG2N; ++level) {
                uint32_t count = uint32_t(N) >> (level + 1);
                if (count == 0) break;
                uint32_t level_start = count - 1;
                loc_c = tree_refit_prog.uniform_location("u_count");
                if (loc_c >= 0) glProgramUniform1ui(tree_refit_prog.handle(), loc_c, count);
                loc_c = tree_refit_prog.uniform_location("u_level_start");
                if (loc_c >= 0) glProgramUniform1ui(tree_refit_prog.handle(), loc_c, level_start);
                gl::dispatch_compute((count + 255) / 256, 1, 1);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }

            t_refit.end();

            // --- Stage D: Direct lighting from emissive leaves ---
            t_direct.begin();
            radiance_buf.bind_base(8);
            emitters_buf.bind_base(9);

            direct_light_prog.use();
            loc_c = direct_light_prog.uniform_location("u_num_leaves");
            if (loc_c >= 0) glProgramUniform1ui(direct_light_prog.handle(), loc_c, GLuint(N));
            loc_c = direct_light_prog.uniform_location("u_num_emitters");
            if (loc_c >= 0) glProgramUniform1ui(direct_light_prog.handle(), loc_c, GLuint(emitter_indices.size()));
            loc_c = direct_light_prog.uniform_location("u_leaf_area");
            if (loc_c >= 0) glProgramUniform1f(direct_light_prog.handle(), loc_c, leaf_area);
            loc_c = direct_light_prog.uniform_location("u_emissive_gain");
            if (loc_c >= 0) glProgramUniform1f(direct_light_prog.handle(), loc_c, emissive_gain);
            gl::dispatch_compute((N + 255) / 256, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            // --- Stage D: Radiance pull-up (bottom-up, log2(N) levels) ---
            radiance_pull_prog.use();
            for (int level = 0; level < LOG2N; ++level) {
                uint32_t count = uint32_t(N) >> (level + 1);
                if (count == 0) break;
                uint32_t level_start = count - 1;
                loc_c = radiance_pull_prog.uniform_location("u_count");
                if (loc_c >= 0) glProgramUniform1ui(radiance_pull_prog.handle(), loc_c, count);
                loc_c = radiance_pull_prog.uniform_location("u_level_start");
                if (loc_c >= 0) glProgramUniform1ui(radiance_pull_prog.handle(), loc_c, level_start);
                gl::dispatch_compute((count + 255) / 256, 1, 1);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }

            float self_eps = 0.5f * ph.leaf_radius;

            // --- Multi-bounce: hierarchical DFS gather (paper Fig. 3 right half) ---
            // First, reorder heap data → DFS order (sphere/cone don't change, only once)
            sph_reorder_prog.use();
            loc_c = sph_reorder_prog.uniform_location("u_tree_size");
            if (loc_c >= 0) glProgramUniform1ui(sph_reorder_prog.handle(), loc_c, GLuint(2 * N - 1));
            gl::dispatch_compute((2 * N - 1 + 255) / 256, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            for (int bounce = 0; bounce < num_bounces; ++bounce) {
                // Reorder radiance to DFS order
                rad_reorder_prog.use();
                loc_c = rad_reorder_prog.uniform_location("u_tree_size");
                if (loc_c >= 0) glProgramUniform1ui(rad_reorder_prog.handle(), loc_c, GLuint(2 * N - 1));
                gl::dispatch_compute((2 * N - 1 + 255) / 256, 1, 1);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                // Ping-pong: src at binding 8, dst at binding 10
                bounce_gather_prog.use();
                loc_c = bounce_gather_prog.uniform_location("u_num_leaves");
                if (loc_c >= 0) glProgramUniform1ui(bounce_gather_prog.handle(), loc_c, GLuint(N));
                loc_c = bounce_gather_prog.uniform_location("u_leaf_area");
                if (loc_c >= 0) glProgramUniform1f(bounce_gather_prog.handle(), loc_c, leaf_area);
                loc_c = bounce_gather_prog.uniform_location("u_emissive_gain");
                if (loc_c >= 0) glProgramUniform1f(bounce_gather_prog.handle(), loc_c, emissive_gain);
                loc_c = bounce_gather_prog.uniform_location("u_self_eps2");
                if (loc_c >= 0) glProgramUniform1f(bounce_gather_prog.handle(), loc_c, self_eps * self_eps);

                if (bounce % 2 == 0) {
                    radiance_buf.bind_base(8);
                    radiance_b_buf.bind_base(10);
                } else {
                    radiance_b_buf.bind_base(8);
                    radiance_buf.bind_base(10);
                }
                gl::dispatch_compute((N + 255) / 256, 1, 1);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                // Pullup on the destination buffer
                gl::Buffer& dst_buf = (bounce % 2 == 0) ? radiance_b_buf : radiance_buf;
                dst_buf.bind_base(8);
                radiance_pull_prog.use();
                for (int level = 0; level < LOG2N; ++level) {
                    uint32_t count = uint32_t(N) >> (level + 1);
                    if (count == 0) break;
                    uint32_t level_start = count - 1;
                    loc_c = radiance_pull_prog.uniform_location("u_count");
                    if (loc_c >= 0) glProgramUniform1ui(radiance_pull_prog.handle(), loc_c, count);
                    loc_c = radiance_pull_prog.uniform_location("u_level_start");
                    if (loc_c >= 0) glProgramUniform1ui(radiance_pull_prog.handle(), loc_c, level_start);
                    gl::dispatch_compute((count + 255) / 256, 1, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                }
            }
            // Ensure binding 8 has the final radiance for micro-render
            if (num_bounces > 0 && (num_bounces % 2 == 1))
                radiance_b_buf.bind_base(8);
            else
                radiance_buf.bind_base(8);

            t_direct.end();
        }

        // ===================================================================
        // 1e. Micro-rendering (Stage E)
        // ===================================================================
        t_micro.begin();
        if (run_micro_render) {
            int rw = std::max(1, fw / micro_res_scale);
            int rh = std::max(1, fh / micro_res_scale);

            indirect_tex.bind_image(0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

            gbuf.position.bind(0);
            gbuf.normal.bind(1);
            gbuf.albedo.bind(2);

            // Regenerate GGX warp table if roughness changed
            if (std::abs(micro_roughness - last_roughness) > 0.001f) {
                float alpha = std::max(0.001f, micro_roughness * micro_roughness);
                compute_brdf_warp_table(alpha, WARP_SIZE, warp_data);
                warp_tex.image_2d(0, GL_RGBA32F, WARP_SIZE, WARP_SIZE, GL_RGBA, GL_FLOAT, warp_data.data(), 1);
                last_roughness = micro_roughness;
            }
            warp_tex.bind(3);

            // Reorder heap → DFS for micro-render
            sphere_buf.bind_base(6);
            cone_buf.bind_base(7);
            sph_reorder_prog.use();
            loc = sph_reorder_prog.uniform_location("u_tree_size");
            if (loc >= 0) glProgramUniform1ui(sph_reorder_prog.handle(), loc, GLuint(2 * N - 1));
            gl::dispatch_compute((2 * N - 1 + 255) / 256, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            // Reorder radiance heap → DFS for micro-render
            // Binding 8 is the final radiance from the bounce loop (or direct lighting)
            rad_reorder_prog.use();
            loc = rad_reorder_prog.uniform_location("u_tree_size");
            if (loc >= 0) glProgramUniform1ui(rad_reorder_prog.handle(), loc, GLuint(2 * N - 1));
            gl::dispatch_compute((2 * N - 1 + 255) / 256, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            // Bind DFS-ordered buffers for micro-render
            dfs_sphere_buf.bind_base(16);
            dfs_cone_buf.bind_base(17);
            dfs_left_buf.bind_base(18);
            dfs_right_buf.bind_base(19);
            dfs_rad_buf.bind_base(20);
            morton_buf.bind_base(22);
            // radiance already in dfs_rad_buf (binding 20) after bounce loop reorder

            debug_pos_buf.bind_base(10);
            debug_col_buf.bind_base(11);

            // Intra-shader profiling buffer
            size_t prof_count = size_t(rw) * size_t(rh) * 4;
            prof_buf.data(nullptr, prof_count * sizeof(uint32_t));
            prof_buf.bind_base(12);

            // Node-visit instrumentation buffer (uvec4 per workgroup)
            size_t visit_count = size_t(rw) * size_t(rh) * 4;
            visit_stats_buf.data(nullptr, visit_count * sizeof(uint32_t));
            visit_stats_buf.bind_base(15);

            debug_tex.bind_image(1, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

            // Clear debug tex to black before dispatch so stale data doesn't persist
            {
                float zeros[4] = {0, 0, 0, 0};
                glClearTexImage(debug_tex.handle(), 0, GL_RGBA, GL_FLOAT, zeros);
            }

            micro_render_prog.use();
            loc = micro_render_prog.uniform_location("u_num_leaves");
            if (loc >= 0) glProgramUniform1ui(micro_render_prog.handle(), loc, GLuint(N));
            loc = micro_render_prog.uniform_location("u_screen_size");
            if (loc >= 0) glProgramUniform2i(micro_render_prog.handle(), loc, fw, fh);
            loc = micro_render_prog.uniform_location("u_micro_size");
            if (loc >= 0) glProgramUniform1ui(micro_render_prog.handle(), loc, GLuint(micro_size));
            loc = micro_render_prog.uniform_location("u_scale");
            if (loc >= 0) glProgramUniform1ui(micro_render_prog.handle(), loc, GLuint(micro_res_scale));
            loc = micro_render_prog.uniform_location("u_gain");
            if (loc >= 0) glProgramUniform1f(micro_render_prog.handle(), loc, micro_gain);

            loc = micro_render_prog.uniform_location("u_low_res_w");
            if (loc >= 0) glProgramUniform1i(micro_render_prog.handle(), loc, rw);
            loc = micro_render_prog.uniform_location("u_low_res_h");
            if (loc >= 0) glProgramUniform1i(micro_render_prog.handle(), loc, rh);

            // Squared threshold below which a leaf surfel is treated as coincident
            // with the receiver and skipped.  Use 0.5× the leaf radius — only
            // prevents exact self-intersection, not neighbor culling.
            float self_eps = 0.5f * ph.leaf_radius;
            loc = micro_render_prog.uniform_location("u_self_eps2");
            if (loc >= 0) glProgramUniform1f(micro_render_prog.handle(), loc, self_eps * self_eps);

            int m_valid = 0;
            for (int ly = 0; ly < micro_size; ly++) {
                for (int lx = 0; lx < micro_size; lx++) {
                    float mu = (2.0f * lx + 1.0f) / micro_size - 1.0f;
                    float mv = (2.0f * ly + 1.0f) / micro_size - 1.0f;
                    if (mu * mu + mv * mv <= 1.0f) m_valid++;
                }
            }
            loc = micro_render_prog.uniform_location("u_m_valid");
            if (loc >= 0) glProgramUniform1ui(micro_render_prog.handle(), loc, GLuint(m_valid));

            loc = micro_render_prog.uniform_location("gbuf_pos");
            if (loc >= 0) glProgramUniform1i(micro_render_prog.handle(), loc, 0);
            loc = micro_render_prog.uniform_location("gbuf_nrm");
            if (loc >= 0) glProgramUniform1i(micro_render_prog.handle(), loc, 1);
            loc = micro_render_prog.uniform_location("gbuf_alb");
            if (loc >= 0) glProgramUniform1i(micro_render_prog.handle(), loc, 2);
            loc = micro_render_prog.uniform_location("u_warp_tex");
            if (loc >= 0) glProgramUniform1i(micro_render_prog.handle(), loc, 3);

            loc = micro_render_prog.uniform_location("u_roughness");
            if (loc >= 0) glProgramUniform1f(micro_render_prog.handle(), loc, micro_roughness);
            loc = micro_render_prog.uniform_location("u_camera_pos");
            if (loc >= 0) {
                glm::vec3 cp = cam.position();
                glProgramUniform3f(micro_render_prog.handle(), loc, cp.x, cp.y, cp.z);
            }
            loc = micro_render_prog.uniform_location("u_jitter_seed");
            if (loc >= 0) glProgramUniform1ui(micro_render_prog.handle(), loc, GLuint(frame_counter));

            if (show_micro_debug && debug_pixel_x >= 0) {
                loc = micro_render_prog.uniform_location("u_debug_pixel");
                if (loc >= 0) glProgramUniform2i(micro_render_prog.handle(), loc,
                    debug_pixel_x / micro_res_scale, (fh - 1 - debug_pixel_y) / micro_res_scale);
            } else {
                loc = micro_render_prog.uniform_location("u_debug_pixel");
                if (loc >= 0) glProgramUniform2i(micro_render_prog.handle(), loc, -1, -1);
            }

            int dispatch_w = (rw + 7) / 8;
            int dispatch_h = (rh + 7) / 8;
            int num_workgroups = dispatch_w * dispatch_h;

            // Generate Morton order dispatch table: uvec4(morton_code, tile_x, tile_y, 0)
            {
                std::vector<uint32_t> morton_entries(num_workgroups * 4);
                for (int ty = 0; ty < dispatch_h; ty++) {
                    for (int tx = 0; tx < dispatch_w; tx++) {
                        uint32_t mc = morton2d(uint32_t(tx), uint32_t(ty));
                        int idx = (ty * dispatch_w + tx) * 4;
                        morton_entries[idx + 0] = mc;
                        morton_entries[idx + 1] = uint32_t(tx);
                        morton_entries[idx + 2] = uint32_t(ty);
                        morton_entries[idx + 3] = 0;
                    }
                }
                // Sort by Morton code for cache-friendly dispatch ordering
                struct MortonEntry { uint32_t code, tx, ty, pad; };
                auto* entries = reinterpret_cast<MortonEntry*>(morton_entries.data());
                std::sort(entries, entries + num_workgroups,
                    [](const MortonEntry& a, const MortonEntry& b) { return a.code < b.code; });
                morton_buf.data(morton_entries.data(), num_workgroups * 4 * sizeof(uint32_t));
            }

            gl::dispatch_compute(num_workgroups, 1, 1);

            // Profiling readback: always run for console logging, but
            // only do the expensive glFinish+map when profiling or logging.
            if (profile_this_frame) {
                glMemoryBarrier(GL_ALL_BARRIER_BITS);
                glFinish();

                prof_readback.resize(prof_count);
                void* ptr = prof_buf.map_range(0, prof_count * sizeof(uint32_t), GL_MAP_READ_BIT);
                if (ptr) {
                    memcpy(prof_readback.data(), ptr, prof_count * sizeof(uint32_t));
                    prof_buf.unmap();
                }
                double sums[3] = {};
                uint64_t total = 0;
                for (size_t wg = 0; wg < size_t(rw) * size_t(rh); ++wg) {
                    uint32_t t0 = prof_readback[wg * 4 + 0];
                    uint32_t t1 = prof_readback[wg * 4 + 1];
                    uint32_t t2 = prof_readback[wg * 4 + 2];
                    uint32_t t3 = prof_readback[wg * 4 + 3];
                    sums[0] += (t1 - t0);
                    sums[1] += (t2 - t1);
                    sums[2] += (t3 - t2);
                    total   += (t3 - t0);
                }
                size_t nwg = size_t(rw) * size_t(rh);
                for (int i = 0; i < 3; ++i) {
                    phase_avg[i] = sums[i] / double(nwg);
                    phase_pct[i] = total > 0 ? sums[i] * 100.0 / double(total) : 0.0;
                }

                // --- Node-visit instrumentation summary (DFS) ---
                visit_stats_readback.resize(visit_count);
                void* vptr = visit_stats_buf.map_range(0, visit_count * sizeof(uint32_t), GL_MAP_READ_BIT);
                if (vptr) {
                    memcpy(visit_stats_readback.data(), vptr, visit_count * sizeof(uint32_t));
                    visit_stats_buf.unmap();
                }
                uint64_t visitedSum = 0, visitedMax = 0;
                uint64_t stackSum = 0, stackOverflowCount = 0;
                for (size_t wg = 0; wg < nwg; ++wg) {
                    uint32_t visited  = visit_stats_readback[wg * 4 + 0];
                    uint32_t maxSp    = visit_stats_readback[wg * 4 + 1];
                    visitedSum += visited;
                    visitedMax = std::max<uint64_t>(visitedMax, visited);
                    stackSum += maxSp;
                    if (maxSp >= 32) stackOverflowCount++; // hit STACK_DEPTH
                }
                visit_avg    = double(visitedSum) / double(nwg);
                visit_max    = double(visitedMax);
                frontier_avg = double(stackSum) / double(nwg);  // reuse as avg max stack depth
                frontier_saturated_pct = 100.0 * double(stackOverflowCount) / double(nwg); // reuse as stack overflow %
            } else {
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
            }
        }
        t_micro.end();

        // ===================================================================
        // 1f. Bilateral upsampling (Stage H)
        // ===================================================================
        if (run_micro_render && use_bilateral_upsample && micro_res_scale > 1) {
            glMemoryBarrier(GL_ALL_BARRIER_BITS);

            int rw = std::max(1, fw / micro_res_scale);
            int rh = std::max(1, fh / micro_res_scale);

            indirect_tex.bind_image(0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
            indirect_upsampled_tex.bind_image(1, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

            gbuf.position.bind(0);
            gbuf.normal.bind(1);

            bilateral_prog.use();
            loc = bilateral_prog.uniform_location("u_low_size");
            if (loc >= 0) glProgramUniform2i(bilateral_prog.handle(), loc, rw, rh);
            loc = bilateral_prog.uniform_location("u_hi_size");
            if (loc >= 0) glProgramUniform2i(bilateral_prog.handle(), loc, fw, fh);
            loc = bilateral_prog.uniform_location("u_depth_sigma");
            if (loc >= 0) glProgramUniform1f(bilateral_prog.handle(), loc, bilateral_depth_sigma);
            loc = bilateral_prog.uniform_location("u_normal_exp");
            if (loc >= 0) glProgramUniform1f(bilateral_prog.handle(), loc, bilateral_normal_exp);

            gl::dispatch_compute((fw + 7) / 8, (fh + 7) / 8, 1);
            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
        }

    // ===================================================================
        // 2. Display pass — fullscreen triangle showing selected G-buffer target
        // ===================================================================
        t_display.begin();
        gl::disable(GL_DEPTH_TEST);
        gl::viewport(0, 0, fw, fh);
        gl::clear_color(0.02f, 0.02f, 0.03f, 1.0f);
        gl::clear(GL_COLOR_BUFFER_BIT);

        display_prog.use();
        fsq_vao.bind();
        gl::Texture* targets[] = { &gbuf.albedo, &gbuf.normal, &gbuf.position,
                                    &gbuf.emissive, &gbuf.depth };
        if (view_mode == 5) {
            if (use_bilateral_upsample && micro_res_scale > 1)
                indirect_upsampled_tex.bind(0);
            else
                indirect_tex.bind(0);
        } else {
            targets[view_mode]->bind(0);
        }
        loc = display_prog.uniform_location("u_tex");      if (loc >= 0) display_prog.uniform1i(loc, 0);
        loc = display_prog.uniform_location("u_mode");     if (loc >= 0) display_prog.uniform1i(loc, view_mode + 1);
        loc = display_prog.uniform_location("u_far");      if (loc >= 0) display_prog.uniform1f(loc, far_plane);
        loc = display_prog.uniform_location("u_exposure"); if (loc >= 0) display_prog.uniform1f(loc, exposure);
        loc = display_prog.uniform_location("u_gamma");    if (loc >= 0) display_prog.uniform1f(loc, gamma);
        gl::draw_arrays(GL_TRIANGLES, 0, 3);
        t_display.end();

        // ===================================================================
        // 2b. Point cloud debug overlay
        // ===================================================================
        t_pointcloud.begin();
        if (show_points) {
            pc_prog.use();
            loc = pc_prog.uniform_location("u_view_proj");
            if (loc >= 0) pc_prog.uniform_matrix4fv(loc, glm::value_ptr(vp));
            loc = pc_prog.uniform_location("u_point_size");
            if (loc >= 0) pc_prog.uniform1f(loc, point_size);
            loc = pc_prog.uniform_location("u_color_mode");
            if (loc >= 0) pc_prog.uniform1i(loc, pc_color_mode);
            loc = pc_prog.uniform_location("u_num_leaves");
            if (loc >= 0) glProgramUniform1ui(pc_prog.handle(), loc, GLuint(N));

            pgeom_buf.bind_base(0);
            pnrm_buf.bind_base(1);
            palb_buf.bind_base(2);
            pemit_buf.bind_base(3);
            radiance_buf.bind_base(8);

            gl::enable(GL_PROGRAM_POINT_SIZE);
            gl::clear(GL_DEPTH_BUFFER_BIT);
            gl::enable(GL_DEPTH_TEST);
            gl::depth_func(GL_LESS);
            pc_vao.bind();
            gl::draw_arrays(GL_POINTS, 0, N);
            gl::disable(GL_PROGRAM_POINT_SIZE);
        }
        t_pointcloud.end();

        // ===================================================================
        // 2c. Bounding sphere wireframe debug overlay
        // ===================================================================
        if (show_spheres) {
            sphere_prog.use();
            loc = sphere_prog.uniform_location("u_view_proj");
            if (loc >= 0) sphere_prog.uniform_matrix4fv(loc, glm::value_ptr(vp));
            loc = sphere_prog.uniform_location("u_color");
            if (loc >= 0) sphere_prog.uniform4fv(loc, glm::value_ptr(sphere_color));

            gl::enable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

            sphere_mesh.vao.bind();

            int inst_first = 0, inst_count = 2 * N - 1;
            if (sphere_lod == 1) { inst_first = N - 1; inst_count = N; }
            else if (sphere_lod == 2) { inst_first = 0; inst_count = N - 1; }

            // Redirect instance attribute binding to the right offset in sphere_buf
            glVertexArrayVertexBuffer(sphere_mesh.vao.handle(), 1,
                                      sphere_buf.handle(),
                                      inst_first * sizeof(glm::vec4),
                                      sizeof(glm::vec4));

            glDrawElementsInstanced(GL_TRIANGLES, sphere_mesh.index_count,
                                    GL_UNSIGNED_INT, nullptr, inst_count);

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            gl::disable(GL_BLEND);
        }

        // ===================================================================
        // 2d. Debug surfel overlay (micro-buffer sample points)
        // ===================================================================
        if (show_micro_debug && run_micro_render) {
            debug_surfel_prog.use();
            loc = debug_surfel_prog.uniform_location("u_view_proj");
            if (loc >= 0) debug_surfel_prog.uniform_matrix4fv(loc, glm::value_ptr(vp));

            debug_pos_buf.bind_base(10);
            debug_col_buf.bind_base(11);

            gl::enable(GL_PROGRAM_POINT_SIZE);
            pc_vao.bind();
            glDrawArrays(GL_POINTS, 0, 64);
            gl::disable(GL_PROGRAM_POINT_SIZE);
        }

        // ===================================================================
        // 3. ImGui
        // ===================================================================
        t_imgui.begin();
        gui.begin_frame();
        {
            ImGui::Begin("Stage A+B+C+D+E — Micro Rendering");
            ImGui::Text("FPS: %.1f  Frame: %.2f ms", 1.0f / std::max(dt, 1e-6f), dt * 1000.0f);
            ImGui::Text("Resolution: %d x %d", gbuf.w, gbuf.h);
            ImGui::Separator();

            // G-buffer view
            ImGui::Combo("G-Buffer View", &view_mode,
                         "Albedo\0Normal\0Position\0Emissive\0Depth\0Indirect\0");
            ImGui::SliderFloat("Exposure", &exposure, 0.05f, 5.0f);
            ImGui::SliderFloat("Gamma", &gamma, 1.0f, 3.0f);

            // G-buffer previews
            {
                float aw = ImGui::GetContentRegionAvail().x;
                if (aw < 64.0f) aw = 256.0f;
                float ih = aw * float(gbuf.h) / float(gbuf.w);
                ImVec2 uv0(0, 1), uv1(1, 0);
                ImGui::Separator();
                ImGui::Text("G-Buffer Previews");
                ImGui::Image((ImTextureID)(intptr_t)gbuf.albedo.handle(), ImVec2(aw, ih), uv0, uv1);
                ImGui::Image((ImTextureID)(intptr_t)gbuf.normal.handle(), ImVec2(aw, ih), uv0, uv1);
                ImGui::Image((ImTextureID)(intptr_t)gbuf.position.handle(), ImVec2(aw, ih), uv0, uv1);
                ImGui::Image((ImTextureID)(intptr_t)gbuf.emissive.handle(), ImVec2(aw, ih), uv0, uv1);
                ImGui::Image((ImTextureID)(intptr_t)gbuf.depth.handle(), ImVec2(aw, ih), uv0, uv1);
            }

            // Point hierarchy debug
            ImGui::Separator();
            ImGui::Text("Point Hierarchy (Stage B)");
            ImGui::Text("  Points: %d (2^%d)", ph.N, LOG2N);
            ImGui::Text("  Triangles: %zu", tris.size());
            ImGui::Text("  Total area: %.4f", ph.total_area);
            ImGui::Text("  Leaf radius: %.6f", ph.leaf_radius);
            ImGui::Text("  Build time: %.1f ms", build_ms);

            ImGui::Separator();
            ImGui::Checkbox("Show point cloud", &show_points);
            ImGui::SliderFloat("Point size", &point_size, 1.0f, 20.0f);
            ImGui::Combo("Point color", &pc_color_mode, "Albedo\0Emissive\0Normal\0Position\0Radiance\0");

            // Stage C — GPU refit
            ImGui::Separator();
            ImGui::Text("Stage C — GPU Refit");
            ImGui::Checkbox("Run refit", &run_refit);
            ImGui::SameLine();
            ImGui::Text("  %.2f ms", t_refit.disp_gpu());
            ImGui::Checkbox("Show bounding spheres", &show_spheres);
            if (show_spheres) {
                ImGui::Combo("Sphere LOD", &sphere_lod, "All\0Leaves\0Interior\0");
                ImGui::ColorEdit4("Sphere color", &sphere_color.x,
                                  ImGuiColorEditFlags_NoInputs);
            }

            if (ImGui::Button("Validate GPU vs CPU")) {
                // Read back GPU sphere/cone data and compare with CPU reference
                int tree_size = 2 * N - 1;
                std::vector<glm::vec4> gpu_sphere(tree_size), gpu_cone(tree_size);
                void* p = sphere_buf.map_range(0, tree_size * sizeof(glm::vec4), GL_MAP_READ_BIT);
                if (p) { memcpy(gpu_sphere.data(), p, tree_size * sizeof(glm::vec4)); sphere_buf.unmap(); }
                p = cone_buf.map_range(0, tree_size * sizeof(glm::vec4), GL_MAP_READ_BIT);
                if (p) { memcpy(gpu_cone.data(), p, tree_size * sizeof(glm::vec4)); cone_buf.unmap(); }

                max_pos_err = 0.0f;
                max_cone_err = 0.0f;
                for (int i = 0; i < tree_size; ++i) {
                    glm::vec3 gp(gpu_sphere[i]), cp(ph.sphere[i]);
                    float pos_d = glm::length(gp - cp);
                    float rad_d = std::abs(gpu_sphere[i].w - ph.sphere[i].w);
                    max_pos_err = std::max(max_pos_err, std::max(pos_d, rad_d));

                    float ca = std::acos(std::clamp(glm::dot(glm::normalize(glm::vec3(gpu_cone[i])),
                                                             glm::normalize(glm::vec3(ph.cone[i]))),
                                                    -1.0f, 1.0f));
                    float cd = std::abs(gpu_cone[i].w - ph.cone[i].w);
                    max_cone_err = std::max(max_cone_err, std::max(ca, cd));
                }
                gllib::logf(gllib::LogLevel::info,
                            "Validation: max pos err = %.8f, max cone err = %.8f rad",
                            max_pos_err, max_cone_err);
            }
            if (max_pos_err > 0.0f || max_cone_err > 0.0f) {
                ImGui::Text("Max pos err: %.8f", max_pos_err);
                ImGui::Text("Max cone err: %.8f rad", max_cone_err);
            }

            // Stage D — Direct lighting
            ImGui::Separator();
            ImGui::Text("Stage D — Direct Lighting");
            ImGui::Text("  Emitters: %zu / %d", emitter_indices.size(), N);
            ImGui::Text("  Leaf area: %.6f", leaf_area);
            ImGui::SliderFloat("Emissive gain", &emissive_gain, 0.1f, 50.0f, "%.1f");
            ImGui::SliderInt("Bounces", &num_bounces, 0, 8);
            ImGui::Text("  Switch to 'Radiance' color mode to visualize");

            // Stage E — Micro-rendering
            ImGui::Separator();
            ImGui::Text("Stage E — Micro-Rendering");
            ImGui::Checkbox("Run micro-render", &run_micro_render);
            if (run_micro_render) {
                ImGui::SameLine();
                ImGui::Text("  %.2f ms", t_micro.disp_gpu());
            }
            ImGui::Text("  Micro-res: %dx%d (%d valid disk px)", micro_size, micro_size, micro_size * micro_size);
            ImGui::SliderInt("Render scale", &micro_res_scale, 1, 8);
            ImGui::SliderInt("Tile size", &tile_size, 1, 8);
            ImGui::SliderFloat("Gain", &micro_gain, 0.1f, 20.0f, "%.1f");
            ImGui::SliderFloat("Roughness", &micro_roughness, 0.0f, 1.0f, "%.2f");
            if (micro_res_scale > 1) {
                ImGui::Checkbox("Bilateral upsample", &use_bilateral_upsample);
                if (use_bilateral_upsample) {
                    ImGui::SliderFloat("Depth sigma", &bilateral_depth_sigma, 0.001f, 0.5f, "%.3f");
                    ImGui::SliderFloat("Normal exp", &bilateral_normal_exp, 1.0f, 128.0f, "%.0f");
                }
            }
            ImGui::Text("  Internal: %dx%d", std::max(1, fw / micro_res_scale), std::max(1, fh / micro_res_scale));
            ImGui::Text("  Select 'Indirect' in G-Buffer View to visualize");
            ImGui::Checkbox("Debug micro-buffer", &show_micro_debug);
            if (show_micro_debug) {
                auto& io = ImGui::GetIO();
                debug_pixel_x = int(io.MousePos.x);
                debug_pixel_y = int(io.MousePos.y);
                ImGui::Text("  Hover pixel: (%d, %d)", debug_pixel_x, debug_pixel_y);
                if (debug_pixel_x >= 0 && debug_pixel_y >= 0) {
                    ImGui::Text("  Leaf albedo (R/G/B):");
                    ImVec2 sz(192, 192);
                    ImGui::Image((ImTextureID)(intptr_t)debug_tex.handle(), sz);
                }
            }

            if (run_micro_render && !prof_readback.empty()) {
                ImGui::Separator();
                ImGui::Text("  Shader phases:");
                ImU32 pcols[] = {
                    IM_COL32(100,180,255,255), IM_COL32(255,200,60,255), IM_COL32(100,220,120,255)
                };
                ImVec2 bp = ImGui::GetCursorScreenPos();
                ImVec2 bs(ImGui::GetContentRegionAvail().x, 16.0f);
                float pvals[3] = { float(phase_pct[0]), float(phase_pct[1]), float(phase_pct[2]) };
                imgui_stacked_bar(bp, bs, pvals, pcols, 3);
                ImGui::Dummy(bs);
                float tot_avg = float(phase_avg[0] + phase_avg[1] + phase_avg[2]);
                imgui_stacked_legend("##microphases", phase_names, pvals, pcols, 3, tot_avg);

                ImGui::Separator();
                ImGui::Text("  DFS traversal instrumentation:");
                ImGui::Text("    Nodes visited/px: avg %.0f, max %.0f",
                             visit_avg, visit_max);
                ImGui::Text("    Max stack depth: avg %.1f  |  hit STACK_DEPTH(32): %.1f%% of px",
                             frontier_avg, frontier_saturated_pct);
                if (frontier_saturated_pct > 5.0) {
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.3f, 1),
                        "    -> stack overflow on a meaningful fraction of pixels!");
                }
            }

            ImGui::Separator();
            ImGui::Text("Profiling");
            {
                static const PassTimer* timers[] = {
                    &t_geo, &t_refit, &t_direct, &t_micro, &t_display, &t_pointcloud, &t_imgui, &t_frame
                };
                static const ImU32 colors[] = {
                    IM_COL32(100,180,255,255), IM_COL32(255,120,100,255), IM_COL32(255,200,60,255),
                    IM_COL32(100,220,120,255), IM_COL32(180,130,255,255), IM_COL32(255,160,200,255),
                    IM_COL32(120,220,220,255), IM_COL32(200,200,200,255)
                };
                static const char* names[] = {
                    "geo", "refit", "direct", "micro", "display", "pointcloud", "imgui", "frame"
                };
                constexpr int NT = sizeof(timers) / sizeof(timers[0]);
                float total = timers[NT-1]->disp_cpu();
                float bar_vals[NT];
                for (int i = 0; i < NT; ++i)
                    bar_vals[i] = (i < NT-1) ? float(timers[i]->disp_gpu()) : float(timers[i]->disp_cpu());

                ImVec2 bp = ImGui::GetCursorScreenPos();
                ImVec2 bs(ImGui::GetContentRegionAvail().x, 20.0f);
                imgui_stacked_bar(bp, bs, bar_vals, colors, NT);
                ImGui::Dummy(bs);
                imgui_stacked_legend("##proflegend", names, bar_vals, colors, NT, total);
            }

            ImGui::End();
        }
        gui.render();
        t_imgui.end();
        t_frame.end();

        // Readback all timers
        t_refit.readback();
        t_direct.readback();
        t_micro.readback();
        t_display.readback();
        t_pointcloud.readback();
        t_frame.readback();
        t_imgui.readback();

        if (now - win_start >= 0.5) {
            t_refit.flush_window();
            t_direct.flush_window();
            t_micro.flush_window();
            t_display.flush_window();
            t_pointcloud.flush_window();
            t_frame.flush_window();
            t_imgui.flush_window();
            win_start = now;

            gllib::logf(gllib::LogLevel::info,
                "PERF  frame=%.1fms  geo=%.1f  refit=%.1f  direct=%.1f  micro=%.1f  display=%.1f  imgui=%.1f  |  "
                "phase: trav=%.2fms(%.0f%%) rad=%.2fms(%.0f%%) accum=%.2fms(%.0f%%)  |  "
                "nodes: avg=%.0f max=%.0f  max_stack: avg=%.0f  |  "
                "scale=%d  dispatch=%dx%d  (%d wg)  screen=%dx%d",
                t_frame.disp_cpu(),
                t_geo.disp_gpu(), t_refit.disp_gpu(), t_direct.disp_gpu(),
                t_micro.disp_gpu(), t_display.disp_gpu(), t_imgui.disp_cpu(),
                t_micro.disp_gpu() * phase_pct[0] / 100.0, phase_pct[0],
                t_micro.disp_gpu() * phase_pct[1] / 100.0, phase_pct[1],
                t_micro.disp_gpu() * phase_pct[2] / 100.0, phase_pct[2],
                visit_avg, visit_max,
                frontier_avg,
                micro_res_scale,
                std::max(1, (fw / micro_res_scale + 7) / 8),
                std::max(1, (fh / micro_res_scale + 7) / 8),
                std::max(1, (fw / micro_res_scale + 7) / 8) * std::max(1, (fh / micro_res_scale + 7) / 8),
                fw, fh);
        }

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
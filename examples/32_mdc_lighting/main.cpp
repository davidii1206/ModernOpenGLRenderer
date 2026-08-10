// Example 32 — MDC Lighting: deferred G-Buffer lighting of the Cornell Box.
//
// Loads data/CornellBoxOriginal.glb and rotates the box so its open face points
// at the camera. A deferred pipeline renders it in two passes:
//   geometry pass  → G-buffer (albedo / normal+roughness+metallic /
//                               emissive+AO / depth)
//   lighting pass  → fullscreen quad composites the G-buffer with a hemisphere
//                    ambient fill and the emissive light panel, ACES tonemapping.
// The scene carries no point light — the emissive panel is the only light.
// The ImGui debug window shows per-frame timing, all four G-buffer targets and
// a toggle that renders the MDC (mesh decomposition clustering) patches from
// gfx::CoverageAtlas with per-patch colors.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>
#include <gfx/gbuffer.hpp>
#include <gllib/log.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <gfx/imgui_overlay.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Geometry pass — writes albedo, normal/roughness/metallic, emissive/AO, depth
// ---------------------------------------------------------------------------

static const char* geo_vs = R"(
#version 430 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;

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

static const char* geo_fs = R"(
#version 430 core
in vec3 v_pos;
in vec3 v_normal;

layout(location = 0) out vec4 out_albedo;
layout(location = 1) out vec4 out_normal_rm;
layout(location = 2) out vec4 out_emissive_ao;

uniform vec3  u_albedo;
uniform float u_metallic;
uniform float u_roughness;
uniform float u_ao;
uniform vec3  u_emissive;
uniform float u_emissive_strength;

void main() {
    vec3 N = normalize(v_normal);
    out_albedo = vec4(u_albedo, 1.0);
    out_normal_rm = vec4(N.xy * 0.5 + 0.5, u_roughness, u_metallic);
    out_emissive_ao = vec4(u_emissive * u_emissive_strength, u_ao);
}
)";

// ---------------------------------------------------------------------------
// Lighting pass — fullscreen quad composites the G-buffer
// ---------------------------------------------------------------------------

static const char* light_vs = R"(
#version 430 core
layout(location = 0) in vec2 a_pos;
out vec2 v_uv;
void main() {
    v_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
)";

static const char* light_fs = R"(
#version 430 core
in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

uniform sampler2D u_albedo_tex;
uniform sampler2D u_normal_rm_tex;
uniform sampler2D u_emissive_ao_tex;
uniform sampler2D u_depth_tex;

uniform int u_debug_view;

const vec3  AMBIENT_TOP       = vec3(0.32, 0.30, 0.28);
const vec3  AMBIENT_BOTTOM    = vec3(0.04, 0.04, 0.05);
const float AMBIENT_INTENSITY = 0.8;

vec3 aces(vec3 x) {
    float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    ivec2 tex_size = textureSize(u_depth_tex, 0);
    vec2 uv = gl_FragCoord.xy / vec2(tex_size);

    vec3 albedo = texture(u_albedo_tex, uv).rgb;
    if (u_debug_view == 1) { frag_color = vec4(albedo, 1.0); return; }

    vec4 nrm = texture(u_normal_rm_tex, uv);
    if (u_debug_view == 2) { frag_color = vec4(nrm.rgb, 1.0); return; }

    vec4 emao = texture(u_emissive_ao_tex, uv);
    if (u_debug_view == 3) { frag_color = vec4(emao.rgb, 1.0); return; }

    float depth = texture(u_depth_tex, uv).r;
    if (u_debug_view == 4) { frag_color = vec4(vec3(depth), 1.0); return; }

    if (depth >= 1.0) {
        frag_color = vec4(0.03, 0.03, 0.05, 1.0);
        return;
    }

    vec3 N;
    N.xy = nrm.xy * 2.0 - 1.0;
    N.z = sqrt(max(1.0 - dot(N.xy, N.xy), 0.0));
    float ao = emao.a;
    vec3 emissive = emao.rgb;

    vec3 ambient = mix(AMBIENT_BOTTOM, AMBIENT_TOP, N.y * 0.5 + 0.5) * AMBIENT_INTENSITY;
    vec3 color = albedo * ambient * ao + emissive;

    color = aces(color);
    color = pow(color, vec3(1.0 / 2.2));
    frag_color = vec4(color, 1.0);
}
)";

// ---------------------------------------------------------------------------
// Patch view — draws MDC patches with per-patch colors (gl_DrawID → SSBO)
// ---------------------------------------------------------------------------

static const char* patch_vs = R"(
#version 460 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
uniform mat4 u_vp;
out vec3 v_normal;
flat out int v_patch_id;
void main() {
    gl_Position = u_vp * vec4(a_pos, 1.0);
    v_normal = a_normal;
    v_patch_id = gl_DrawID;
}
)";

static const char* patch_fs = R"(
#version 460 core
in vec3 v_normal;
flat in int v_patch_id;
layout(std430, binding = 0) readonly buffer ColorBuf {
    vec4 patch_colors[];
};
out vec4 frag_color;
void main() {
    vec3 n = normalize(v_normal);
    vec3 light_dir = normalize(vec3(0.0, 1.0, 0.6));
    float light = 0.55 + 0.45 * abs(dot(n, light_dir));
    frag_color = vec4(patch_colors[v_patch_id].rgb * light, 1.0);
}
)";

// ---------------------------------------------------------------------------
// Atlas viewer — GPU reconstruction of a MIP4 chain (UV / thickness / depth)
// ---------------------------------------------------------------------------

static const char* atlas_vs = R"(
#version 430 core
layout(location = 0) in vec2 a_pos;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
)";

static const char* atlas_fs = R"(
#version 430 core
layout(location = 0) out vec4 frag_color;

uniform int u_patch_count;
uniform int u_chain;             // 0 = UV, 1 = thickness, 2 = depth, 3 = normal
uniform int u_debug_meta;        // TEMP: 0=normal, 1=pid, 2=level, 3=idx
uniform int u_level_off[32];     // meta SSBO offsets per level (shared topology)
uniform int u_value_off[32];     // value texture offsets per level (selected chain)

layout(std430, binding = 0) readonly buffer PatchBuf { uvec4 rects[]; };
layout(std430, binding = 1) readonly buffer MetaBuf   { uint  meta[]; };

uniform sampler2D u_values;      // RG8 / RG16 normalized

const int LEAF_TILE = 1;
const int TEX_W = 16384;

float decode_value(float b, float lo, float hi, int bits) {
    float n = (bits >= 16) ? 65535.0 : 255.0;
    float d = (bits >= 16) ? 65534.0 : 254.0;
    return lo + (hi - lo) * (b * n - 1.0) / d;
}

vec3 octahedral_decode(vec2 f) {
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) {
        n.x = (1.0 - abs(f.y)) * (f.x >= 0.0 ? 1.0 : -1.0);
        n.y = (1.0 - abs(f.x)) * (f.y >= 0.0 ? 1.0 : -1.0);
    }
    return normalize(n);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int pid = -1;
    for (int i = 0; i < u_patch_count; ++i) {
        uvec4 r = rects[i];
        if (p.x >= int(r.x) && p.x < int(r.x + r.z) &&
            p.y >= int(r.y) && p.y < int(r.y + r.w)) { pid = i; break; }
    }
    if (pid < 0) { frag_color = vec4(0.0); return; }

    uvec4 r = rects[pid];
    int x = p.x - int(r.x);
    int y = p.y - int(r.y);
    int maxd = int(max(r.z, r.w));
    int N = 1;
    while (N < maxd) N <<= 1;

    int L = 0, x0 = 0, y0 = 0, s = N, idx = pid;
    while (true) {
        uint m = meta[u_level_off[L] + idx];
        if ((m & 0x40000000u) == 0u || s <= LEAF_TILE) break;
        int hs = s >> 1;
        int cx = (x >= x0 + hs) ? 1 : 0;
        int cy = (y >= y0 + hs) ? 1 : 0;
        idx = int(m & 0x3FFFFFFFu) + cy * 2 + cx;
        x0 += cx * hs;
        y0 += cy * hs;
        s = hs;
        L += 1;
    }

    // Normalized (0..1) value bytes; byte 0 = empty, covered nodes are 1..255.
    // Matches the reference renderer (tools/mip4_to_bmp.py shows the raw bytes).
    if (u_debug_meta == 1) { frag_color = vec4(vec3(float(pid) / float(u_patch_count)), 1.0); return; }
    if (u_debug_meta == 2) { frag_color = vec4(vec3(float(L) / 8.0), 1.0); return; }
    if (u_debug_meta == 3) { frag_color = vec4(vec3(float(idx) / 9000.0), 1.0); return; }
    if (u_debug_meta == 4) { frag_color = vec4(vec3(float(u_level_off[1]) / 13.0), 1.0); return; }
    if (u_debug_meta == 5) { frag_color = vec4(vec3(float(u_level_off[2]) / 65.0), 1.0); return; }
    if (u_debug_meta == 6) { uint mm = meta[u_level_off[1] + pid]; frag_color = vec4(vec3(float(mm) / float(0xFFFFFFFFu)) * 600.0, 1.0); return; }
    if (u_debug_meta == 7) { frag_color = vec4(vec3(float(rects[0].x) / 2043.0), 1.0); return; }
    if (u_debug_meta == 8) { frag_color = vec4(vec3(float(rects[1].x) / 2043.0), 1.0); return; }
    if (u_debug_meta == 9) { frag_color = vec4(vec3(float(rects[6].x) / 2043.0), 1.0); return; }
    if (u_debug_meta == 10) { frag_color = vec4(vec3(float(rects[7].x) / 2043.0), 1.0); return; }
    if (u_debug_meta == 11) { frag_color = vec4(vec3(gl_FragCoord.x / 2043.0), 1.0); return; }
    if (u_debug_meta == 12) { frag_color = vec4(vec3(gl_FragCoord.y / 1040.0), 1.0); return; }
    int vo = u_value_off[L] + idx;
    vec2 q = texelFetch(u_values, ivec2(vo % TEX_W, vo / TEX_W), 0).rg;

    if (u_chain == 0)
        frag_color = vec4(q, 0.0, 1.0);         // UV: u,v
    else if (u_chain == 3) {
        // Normal: decode the octahedral coords back to a viewable normal.
        vec2 oct = vec2(decode_value(q.x, -1.0, 1.0, 8),
                        decode_value(q.y, -1.0, 1.0, 8));
        frag_color = vec4(octahedral_decode(oct) * 0.5 + 0.5, 1.0);
    } else
        frag_color = vec4(vec3(q.x), 1.0);      // thickness / depth: channel 0
}
)";

// ---------------------------------------------------------------------------
// Primary-ray pass — traces a ray through the BVH, then refines the hit with
// the patch's atlas mip chain (depth + thickness), writing surface normals.
// ---------------------------------------------------------------------------

static const char* ray_cs = R"(
#version 460 core
layout(local_size_x = 16, local_size_y = 16) in;

const int LEAF_TILE = 1;
const int TEX_W = 16384;
const int MAX_STACK = 64;
const int SAMPLES = 8;

struct PatchInfo {
    vec2  proj_min;
    vec2  proj_size;
    vec3  basis_u;   // depth axis
    vec3  basis_v;   // world direction of the atlas-u axis
    vec3  basis_w;   // world direction of the atlas-v axis
    uint  axis;
    uint  tex_w;
    uint  tex_h;
    uint  double_sided;   // 1 = material has no backface culling (glTF doubleSided)
};

struct BVHGPU {
    vec3  amin;
    uint  pad0;
    vec3  amax;
    uint  pad1;
    uint  val;       // leaf: patch index, internal: right_offset
    uint  is_leaf;
};

layout(std430, binding = 0) readonly buffer PatchBuf     { uvec4 rects[]; };
layout(std430, binding = 1) readonly buffer MetaBuf      { uint  meta[]; };
layout(std430, binding = 2) readonly buffer PatchInfoBuf { PatchInfo patches[]; };
layout(std430, binding = 3) readonly buffer BvhBuf       { BVHGPU bvh[]; };

uniform sampler2D u_depth_tex;     // depth chain value texture (RG16)
uniform sampler2D u_thick_tex;     // thickness chain value texture (RG8)
uniform sampler2D u_normal_tex;    // normal chain value texture (RG8, octahedral)
uniform int  u_level_off[32];      // meta SSBO offsets per level (shared)
uniform int  u_value_off_depth[32];
uniform int  u_value_off_thick[32];
uniform int  u_value_off_normal[32];
uniform int  u_patch_count;
uniform mat4 u_vp_inv;
uniform vec2 u_res;
uniform vec3 u_ro;
uniform vec3 u_depth_origin;       // g_depth_origin (model AABB min)
uniform float u_depth_lo;          // depth chain qmin[0]
uniform float u_depth_hi;          // depth chain qmax[0]
uniform float u_thick_max;         // thickness chain qmax[0]
uniform float u_connect_tol;       // normal connectivity tolerance (world units)
uniform float u_eps;               // slab test epsilon (quantisation slack)
uniform int  u_dbg;                // 0=normals, 1=ray dir, 2=root-aabb test

layout(binding = 0, rgba8) uniform image2D u_out;

float decode_value(float b, float lo, float hi, int bits) {
    float n = (bits >= 16) ? 65535.0 : 255.0;
    float d = (bits >= 16) ? 65534.0 : 254.0;
    return lo + (hi - lo) * (b * n - 1.0) / d;
}

// Octahedral decode of a unit normal stored in [-1,1]^2 (matches the CPU-side
// octahedral_encode in coverage_atlas.cpp, including the z<0 fold convention).
vec3 octahedral_decode(vec2 f) {
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) {
        n.x = (1.0 - abs(f.y)) * (f.x >= 0.0 ? 1.0 : -1.0);
        n.y = (1.0 - abs(f.x)) * (f.y >= 0.0 ? 1.0 : -1.0);
    }
    return normalize(n);
}

bool ray_aabb(vec3 ro, vec3 rd, vec3 amin, vec3 amax,
              out float t_in, out float t_out) {
    t_in = -1.0e30;
    t_out = 1.0e30;
    for (int c = 0; c < 3; ++c) {
        float o = ro[c], d = rd[c], lo = amin[c], hi = amax[c];
        if (abs(d) < 1.0e-9) {
            if (o < lo || o > hi) return false;
        } else {
            float inv = 1.0 / d;
            float t1 = (lo - o) * inv;
            float t2 = (hi - o) * inv;
            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
            t_in = max(t_in, t1);
            t_out = min(t_out, t2);
        }
    }
    if (t_out < 0.0) return false;
    t_in = max(t_in, 0.0);
    return t_in <= t_out;
}

// Descend the shared quadtree to the finest node covering atlas texel (tx,ty)
// of patch pid. Returns true if that node is covered (meta bit 0x80000000).
bool query_node(ivec2 txy, int pid,
                out int L, out int idx, out int x0, out int y0, out int s,
                out float dmin, out float dmax, out float thick) {
    uvec4 r = rects[pid];
    int x = txy.x - int(r.x);
    int y = txy.y - int(r.y);
    if (x < 0 || y < 0 || x >= int(r.z) || y >= int(r.w)) return false;
    int maxd = int(max(r.z, r.w));
    int N = 1;
    while (N < maxd) N <<= 1;

    L = 0; x0 = 0; y0 = 0; s = N; idx = pid;
    while (true) {
        uint m = meta[u_level_off[L] + idx];
        if ((m & 0x40000000u) == 0u || s <= LEAF_TILE) break;
        int hs = s >> 1;
        int cx = (x >= x0 + hs) ? 1 : 0;
        int cy = (y >= y0 + hs) ? 1 : 0;
        idx = int(m & 0x3FFFFFFFu) + cy * 2 + cx;
        x0 += cx * hs;
        y0 += cy * hs;
        s = hs;
        L += 1;
    }
    uint m = meta[u_level_off[L] + idx];
    if ((m & 0x80000000u) == 0u) return false;

    int vo = u_value_off_depth[L] + idx;
    vec2 dq = texelFetch(u_depth_tex, ivec2(vo % TEX_W, vo / TEX_W), 0).rg;
    dmin = decode_value(dq.x, u_depth_lo, u_depth_hi, 16);
    dmax = decode_value(dq.y, u_depth_lo, u_depth_hi, 16);
    int vt = u_value_off_thick[L] + idx;
    float tq = texelFetch(u_thick_tex, ivec2(vt % TEX_W, vt / TEX_W), 0).r;
    thick = decode_value(tq, 0.0, u_thick_max, 8);
    return true;
}

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= int(u_res.x) || p.y >= int(u_res.y)) {
        imageStore(u_out, p, vec4(0.0));
        return;
    }

    vec2 ndc = vec2(2.0 * (float(p.x) + 0.5) / u_res.x - 1.0,
                    2.0 * (float(p.y) + 0.5) / u_res.y - 1.0);
    vec4 cw = u_vp_inv * vec4(ndc, 1.0, 1.0);
    vec3 wpos = cw.xyz / cw.w;
    vec3 ro = u_ro;
    vec3 rd = normalize(wpos - ro);

    if (u_dbg == 1) {
        imageStore(u_out, p, vec4(rd * 0.5 + 0.5, 1.0));
        return;
    }
    if (u_dbg == 2) {
        float t0, t1;
        bool hit = (bvh.length() > 0) &&
                   ray_aabb(ro, rd, bvh[0].amin, bvh[0].amax, t0, t1);
        imageStore(u_out, p, hit ? vec4(0.2, 0.9, 0.2, 1.0) : vec4(0.0));
        return;
    }
    if (u_dbg == 3) {
        // Brute-force over all leaves (bypasses BVH traversal): 2=slab hit,
        // 1=covered node found but slab rejected, 0=nothing.
        int state = 0;
        float tbest = 1.0e30;
        for (int i = 0; i < int(bvh.length()); ++i) {
            if (bvh[i].is_leaf == 0u) continue;
            int pid = int(bvh[i].val);
            float t0, t1;
            if (!ray_aabb(ro, rd, bvh[i].amin, bvh[i].amax, t0, t1)) continue;
            PatchInfo pi = patches[pid];
            float D0 = dot(ro - u_depth_origin, pi.basis_u);
            float dD = dot(rd, pi.basis_u);
            float step = max((t1 - t0) / float(SAMPLES), 0.0);
            bool any_covered = false;
            for (int k = 0; k < SAMPLES; ++k) {
                float t = t0 + step * float(k);
                vec3 P = ro + rd * t;
                vec2 proj = vec2(dot(P, pi.basis_v), dot(P, pi.basis_w));
                vec2 nrm = (proj - pi.proj_min) / pi.proj_size;
                if (nrm.x < 0.0 || nrm.y < 0.0 || nrm.x > 1.0 || nrm.y > 1.0) continue;
                uvec4 r = rects[pid];
                ivec2 txy = ivec2(int(r.x) + int(nrm.x * float(r.z)),
                                  int(r.y) + int(nrm.y * float(r.w)));
                int L, idx, x0, y0, s;
                float dmin, dmax, thick;
                if (!query_node(txy, pid, L, idx, x0, y0, s, dmin, dmax, thick)) continue;
                any_covered = true;
                float D = dot(P - u_depth_origin, pi.basis_u);
                if (D >= dmin - u_eps && D <= dmin + thick) {
                    float th = t;
                    if (abs(dD) > 1.0e-6) {
                        float ts = (dmin - D0) / dD;
                        if (ts >= t0 && ts <= t1) th = ts;
                    }
                    if (th < tbest) { tbest = th; state = 2; }
                    break;
                }
            }
            if (state < 2 && any_covered) state = 1;
        }
        vec3 col = state == 2 ? vec3(0.1, 0.4, 1.0)
                 : (state == 1 ? vec3(1.0, 0.2, 0.1) : vec3(0.0));
        imageStore(u_out, p, vec4(col, 1.0));
        return;
    }
    if (u_dbg == 4) {
        // Diagnostics at the FIRST covered sample: R=t0 mapped to [0,5],
        // G=leaf index/30, B=pid/13
        bool captured = false;
        for (int i = 0; i < int(bvh.length()); ++i) {
            if (captured) break;
            if (bvh[i].is_leaf == 0u) continue;
            int pid = int(bvh[i].val);
            float t0, t1;
            if (!ray_aabb(ro, rd, bvh[i].amin, bvh[i].amax, t0, t1)) continue;
            PatchInfo pi = patches[pid];
            float step = max((t1 - t0) / float(SAMPLES), 0.0);
            for (int k = 0; k < SAMPLES; ++k) {
                float t = t0 + step * float(k);
                vec3 P = ro + rd * t;
                vec2 proj = vec2(dot(P, pi.basis_v), dot(P, pi.basis_w));
                vec2 nrm = (proj - pi.proj_min) / pi.proj_size;
                if (nrm.x < 0.0 || nrm.y < 0.0 || nrm.x > 1.0 || nrm.y > 1.0) continue;
                uvec4 r = rects[pid];
                ivec2 txy = ivec2(int(r.x) + int(nrm.x * float(r.z)),
                                  int(r.y) + int(nrm.y * float(r.w)));
                int L, idx, x0, y0, s;
                float dmin, dmax, thick;
                if (!query_node(txy, pid, L, idx, x0, y0, s, dmin, dmax, thick)) continue;
                imageStore(u_out, p, vec4(t0 / 5.0,
                                          float(i) / 30.0,
                                          float(pid) / 13.0, 1.0));
                captured = true;
                break;
            }
        }
        if (!captured) imageStore(u_out, p, vec4(0.5, 0.5, 1.0, 1.0));
        return;
    }
    if (u_dbg == 5) {
        // Row y = node y: R=low byte of bvh[y].val, G=low byte of bvh[y].is_leaf
        if (bvh.length() == 0) { imageStore(u_out, p, vec4(0.0)); return; }
        int i = min(p.y, int(bvh.length()) - 1);
        uint v = bvh[i].val, il = bvh[i].is_leaf;
        imageStore(u_out, p, vec4(float(v & 0xFFu) / 255.0,
                                  float(il & 0xFFu) / 255.0, 0.0, 1.0));
        return;
    }

    float best_t = 1.0e30;
    int best_pid = -1;
    int best_L = 0, best_idx = 0, best_x0 = 0, best_y0 = 0, best_s = 0;

    if (u_patch_count == 0 || int(bvh.length()) == 0) {
        imageStore(u_out, p, vec4(0.0));
        return;
    }

    int dbg_leaf_visits = 0;
    int dbg_covered = 0;
    int dbg_reject = 0;
    bool dbg_captured = false;
    vec2 miss_nrm_f = vec2(0.0);
    int miss_L = 0;
    bool dbg_pushed = false;

    int   st_node[MAX_STACK];
    float st_tin[MAX_STACK];
    float st_tout[MAX_STACK];
    int sp = 0;

    float rt0, rt1;
    if (ray_aabb(ro, rd, bvh[0].amin, bvh[0].amax, rt0, rt1)) {
        st_node[sp] = 0; st_tin[sp] = rt0; st_tout[sp] = rt1; sp++;
    }

    while (sp > 0) {
        sp--;
        int node = st_node[sp];
        float tin = st_tin[sp];
        float tout = st_tout[sp];
        if (tin > best_t && u_dbg != 7) continue;

        BVHGPU b = bvh[node];
        if (b.is_leaf != 0u) {
            dbg_leaf_visits++;
            int pid = int(b.val);
            PatchInfo pi = patches[pid];
            uvec4 r = rects[pid];

            float dD = dot(rd, pi.basis_u);
            if (abs(dD) <= 1.0e-6) continue;

            // Single-sided materials cull their back face. basis_u points along
            // the patch's surface normals (the clustering axis), so a ray hitting
            // the front face approaches from the +basis_u side (dD < 0) and a
            // back-side hit has dD > 0. Double-sided patches accept both sides.
            if (pi.double_sided == 0u && dD > 0.0) continue;

            float D0 = dot(ro - u_depth_origin, pi.basis_u);
            float qstep = (u_depth_hi - u_depth_lo) / 65534.0;   // 16-bit depth
            float htol = max(2.0 * qstep, 1.0e-4);   // depth quantisation slack

            // ---- Texel-stepped heightfield march (robust at grazing angles) ----
            // The old code solved the crossing analytically with th=(dmin-D0)/dD
            // sampled 16 times over the slab. For grazing rays dD -> 0, which
            // amplifies the depth-quantisation error by 1/|dD| and can skip a
            // thin band entirely, making slanted surfaces see-through/noisy.
            // Instead, advance the projection by at most ~1 texel per step
            // (like example 27) and bisect the band crossing.
            float tex_s = min(pi.proj_size.x / float(max(r.z, 1u)),
                              pi.proj_size.y / float(max(r.w, 1u)));
            vec2 rd_uv = vec2(dot(rd, pi.basis_v), dot(rd, pi.basis_w));
            float uv_speed = max(length(rd_uv), 1.0e-4);
            float step = min(0.5 * tex_s, tex_s / uv_speed);
            step = clamp(step, 1.0e-6, tout - tin);

            float t = tin;
            bool have_prev = false;
            float t_prev = tin;
            bool prev_before = false;   // previous sample still in front of the band
            int guard = 0;
            while (t <= tout && guard < 2048) {
                guard++;
                vec3 P = ro + rd * t;
                vec2 proj = vec2(dot(P, pi.basis_v), dot(P, pi.basis_w));
                vec2 nrm = (proj - pi.proj_min) / pi.proj_size;
                bool inside = nrm.x >= 0.0 && nrm.y >= 0.0 && nrm.x <= 1.0 && nrm.y <= 1.0;
                float dm = 0.0, dmax = 0.0, thk = 0.0;
                int L, idx, x0, y0, s;
                if (inside) {
                    ivec2 txy = ivec2(int(r.x) + int(nrm.x * float(r.z)),
                                      int(r.y) + int(nrm.y * float(r.w)));
                    if (query_node(txy, pid, L, idx, x0, y0, s, dm, dmax, thk)) {
                        dbg_covered++;
                    } else {
                        inside = false;
                    }
                }
                float D = dot(P - u_depth_origin, pi.basis_u);
                // A ray that enters the leaf already inside the band (a head-on
                // ray hits the AABB front face right where the surface sits) never
                // produces a "before -> in-band" crossing, so accept it directly.
                // The band is the leaf's own depth extent [dm, dmax], capped by
                // the per-texel thickness. A node that stopped subdividing as
                // "flat enough" spans a merged region whose dmax can reach far
                // past the local surface (exactly at a cube edge), letting a ray
                // register "inside the band" all the way toward the back face;
                // the thickness cap keeps the band tight to the local surface.
                float band_hi = inside ? min(dmax, dm + thk) : 0.0;
                if (inside && D >= dm - htol && D <= band_hi + htol) {
                    best_t = t;
                    best_pid = pid;
                    best_L = L; best_idx = idx;
                    best_x0 = x0; best_y0 = y0; best_s = s;
                    break;
                }
                // "before" = the ray's depth is still above the surface band
                // (D > band_hi), i.e. it has not reached the surface yet.
                bool cur_before = !inside || (D > band_hi + htol);

                // The band crossing happened between the previous and current
                // sample. Detect it in either direction (front or back face).
                if (have_prev && (prev_before != cur_before)) {
                    float ta = t_prev;
                    float tb = t;
                    for (int it = 0; it < 8; ++it) {
                        float tm = (ta + tb) * 0.5;
                        vec3 Pm = ro + rd * tm;
                        vec2 proj_m = vec2(dot(Pm, pi.basis_v), dot(Pm, pi.basis_w));
                        vec2 nrm_m = (proj_m - pi.proj_min) / pi.proj_size;
                        if (nrm_m.x >= 0.0 && nrm_m.y >= 0.0 && nrm_m.x <= 1.0 && nrm_m.y <= 1.0) {
                            ivec2 txy_m = ivec2(int(r.x) + int(nrm_m.x * float(r.z)),
                                                int(r.y) + int(nrm_m.y * float(r.w)));
                            int Lm, idxm, x0m, y0m, sm;
                            float dm_m, dmax_m, thk_m;
                            if (query_node(txy_m, pid, Lm, idxm, x0m, y0m, sm, dm_m, dmax_m, thk_m)) {
                                float Dm = dot(Pm - u_depth_origin, pi.basis_u);
                                if (Dm <= dm_m) tb = tm; else ta = tm;
                                continue;
                            }
                        }
                        // Fell off the patch or hit an uncovered node: keep the
                        // lower bound so the bracket stays inside the surface.
                        ta = tm;
                    }
                    float th = (ta + tb) * 0.5;

                    // Final acceptance: the refined hit must sit on the surface
                    // band of the cell it lands in.
                    vec3 P_f = ro + rd * th;
                    vec2 proj_f = vec2(dot(P_f, pi.basis_v), dot(P_f, pi.basis_w));
                    vec2 nrm_f = (proj_f - pi.proj_min) / pi.proj_size;
                    if (nrm_f.x < 0.0 || nrm_f.y < 0.0 || nrm_f.x > 1.0 || nrm_f.y > 1.0) { if (!dbg_captured) { dbg_captured = true; dbg_reject = 4; miss_nrm_f = nrm_f; } }
                    else {
                        ivec2 txy_f = ivec2(int(r.x) + int(nrm_f.x * float(r.z)),
                                            int(r.y) + int(nrm_f.y * float(r.w)));
                        int Lf, idxf, x0f, y0f, sf;
                        float dmf, dmxf, thkf;
                        if (!query_node(txy_f, pid, Lf, idxf, x0f, y0f, sf, dmf, dmxf, thkf)) { if (!dbg_captured) { dbg_captured = true; dbg_reject = 5; miss_nrm_f = nrm_f; miss_L = Lf; } }
                        else {
                            float D_f = dot(ro + rd * th - u_depth_origin, pi.basis_u);
                            float band_hi_f = min(dmxf, dmf + thkf);
                            if (D_f < dmf - htol || D_f > band_hi_f + htol) { if (!dbg_captured) { dbg_captured = true; dbg_reject = 3; miss_nrm_f = nrm_f; miss_L = Lf; } }
                            else if (th < best_t) {
                                best_t = th;
                                best_pid = pid;
                                best_L = Lf; best_idx = idxf;
                                best_x0 = x0f; best_y0 = y0f; best_s = sf;
                            }
                        }
                    }
                }

                t_prev = t;
                prev_before = cur_before;
                have_prev = true;
                if (t > best_t) break;
                t += step;
            }
        } else {
            dbg_pushed = true;
            int left = node + 1;
            int right = node + int(b.val);
            float t0, t1, t2, t3;
            bool hit_l = ray_aabb(ro, rd, bvh[left].amin,  bvh[left].amax,  t0, t1);
            bool hit_r = ray_aabb(ro, rd, bvh[right].amin, bvh[right].amax, t2, t3);
            if (hit_l && hit_r) {
                if (t0 <= t2) {   // push the farther child first
                    st_node[sp] = right; st_tin[sp] = t2; st_tout[sp] = t3; sp++;
                    st_node[sp] = left;  st_tin[sp] = t0; st_tout[sp] = t1; sp++;
                } else {
                    st_node[sp] = left;  st_tin[sp] = t0; st_tout[sp] = t1; sp++;
                    st_node[sp] = right; st_tin[sp] = t2; st_tout[sp] = t3; sp++;
                }
            } else if (hit_l) {
                st_node[sp] = left; st_tin[sp] = t0; st_tout[sp] = t1; sp++;
            } else if (hit_r) {
                st_node[sp] = right; st_tin[sp] = t2; st_tout[sp] = t3; sp++;
            }
        }
    }

    if (u_dbg == 6 || u_dbg == 7) {
        imageStore(u_out, p, vec4(float(dbg_leaf_visits) / 30.0,
                                  dbg_pushed ? 1.0 : 0.0,
                                  float(dbg_covered) / 30.0, 1.0));
        return;
    }

    if (best_pid < 0) {
        if (u_dbg == 11) {
            // Miss causality: R=covered-sample count/30, G=reject reason
            // (1=range-front, 2=range-behind, 3=final-band, 4=final-out-of-rect,
            //  5=final-query-false), B=leaf visits/30
            imageStore(u_out, p, vec4(float(dbg_covered) / 30.0,
                                      float(dbg_reject) / 8.0,
                                      float(dbg_leaf_visits) / 30.0, 1.0));
            return;
        }
        if (u_dbg == 12) {
            // For reject 4/5 misses, dump the final projection and node level:
            // R=nrm_f.x, G=nrm_f.y, B=final level/30. For other misses: old codes.
            if (dbg_reject == 4 || dbg_reject == 5) {
                vec2 nf = miss_nrm_f;
                imageStore(u_out, p, vec4(clamp(nf.x, 0.0, 2.0) / 2.0,
                                          clamp(nf.y, 0.0, 2.0) / 2.0,
                                          float(miss_L) / 30.0, 1.0));
                return;
            }
            imageStore(u_out, p, vec4(float(dbg_covered) / 30.0,
                                      float(dbg_reject) / 8.0,
                                      float(dbg_leaf_visits) / 30.0, 1.0));
            return;
        }
        imageStore(u_out, p, vec4(0.0));
        return;
    }

    if (u_dbg == 8) {
        // First covered sample anywhere (brute force):
        // R = (dmin-lo)/(hi-lo), G = (D-lo)/(hi-lo), B = thick/thick_max
        for (int i = 0; i < int(bvh.length()); ++i) {
            if (bvh[i].is_leaf == 0u) continue;
            int pid = int(bvh[i].val);
            float t0, t1;
            if (!ray_aabb(ro, rd, bvh[i].amin, bvh[i].amax, t0, t1)) continue;
            PatchInfo pi = patches[pid];
            float D0 = dot(ro - u_depth_origin, pi.basis_u);
            float dD = dot(rd, pi.basis_u);
            if (pi.double_sided == 0u && dD > 0.0) continue;
            float step = max((t1 - t0) / float(SAMPLES), 0.0);
            for (int k = 0; k < SAMPLES; ++k) {
                float t = t0 + step * float(k);
                vec3 P = ro + rd * t;
                vec2 proj = vec2(dot(P, pi.basis_v), dot(P, pi.basis_w));
                vec2 nrm = (proj - pi.proj_min) / pi.proj_size;
                if (nrm.x < 0.0 || nrm.y < 0.0 || nrm.x > 1.0 || nrm.y > 1.0) continue;
                uvec4 r = rects[pid];
                ivec2 txy = ivec2(int(r.x) + int(nrm.x * float(r.z)),
                                  int(r.y) + int(nrm.y * float(r.w)));
                int L, idx, x0, y0, s;
                float dmin, dmax, thick;
                if (!query_node(txy, pid, L, idx, x0, y0, s, dmin, dmax, thick)) continue;
                float D = dot(P - u_depth_origin, pi.basis_u);
                imageStore(u_out, p, vec4((dmin - u_depth_lo) / (u_depth_hi - u_depth_lo),
                                          (D - u_depth_lo) / (u_depth_hi - u_depth_lo),
                                          thick / u_thick_max, 1.0));
                return;
            }
        }
        imageStore(u_out, p, vec4(0.5, 0.5, 1.0, 1.0));
        return;
    }

    // ---- Surface normal from the baked octahedral normal chain ----
    // The normal was rasterised once (interpolated vertex normal, octahedral-
    // encoded) at atlas build time and aggregated per quadtree node, so fetching
    // it at the resolved node is exact and independent of the view angle /
    // march step that decided which node the ray landed on. No runtime finite
    // differencing over quantised depth samples, no neighbour-dependent wobble.
    PatchInfo pi = patches[best_pid];
    int vo = u_value_off_normal[best_L] + best_idx;
    vec2 qn = texelFetch(u_normal_tex, ivec2(vo % TEX_W, vo / TEX_W), 0).rg;
    vec2 oct = vec2(decode_value(qn.x, -1.0, 1.0, 8),
                    decode_value(qn.y, -1.0, 1.0, 8));
    vec3 n = octahedral_decode(oct);
    // Keep the baked normal facing the ray. Double-sided patches are genuinely
    // ambiguous (backfaces mirror the normal); single-sided patches already
    // cull the back face, but the flip is a free safety net against the normal
    // inaccuracies the rasteriser can introduce at silhouette edges.
    if (dot(n, rd) > 0.0) n = -n;

    if (u_dbg == 9) {
        imageStore(u_out, p, vec4(float(best_x0) / 512.0,
                                  float(best_y0) / 512.0,
                                  float(best_s) / 30.0, 1.0));
        return;
    }

    if (u_dbg == 10) {
        // TEMP: encode best_pid (R) and best_t (G) for sweep analysis
        imageStore(u_out, p, vec4(float(best_pid + 1) / 64.0,
                                  best_t / 8.0, 1.0, 1.0));
        return;
    }

    imageStore(u_out, p, vec4(n * 0.5 + 0.5, 1.0));
}
)";

static const char* ray_display_fs = R"(
#version 430 core
in vec2 v_uv;
layout(location = 0) out vec4 frag_color;
uniform sampler2D u_tex;
void main() {
    frag_color = vec4(texture(u_tex, v_uv).rgb, 1.0);
}
)";

struct PatchGPU {
    GLuint vao = 0, vbo = 0, ebo = 0, indirect = 0, color_ssbo = 0;
    GLsizei draw_count = 0;
    size_t patch_count = 0, tri_count = 0;
};

struct AtlasView {
    GLuint rects_ssbo = 0, meta_ssbo = 0;
    GLuint value_tex[4] = {0, 0, 0, 0};   // UV, thickness, depth, normal
    GLuint fbo = 0, view_tex = 0;
    int view_w = 0, view_h = 0;
    int level_count = 0;
    int level_off[32] = {};
    int value_off[4][32] = {};
};

// std430 layouts matching the primary-ray compute shader.
struct PatchInfoGPU {
    float proj_min[2];
    float proj_size[2];
    float basis_u[3]; float pad0;
    float basis_v[3]; float pad1;
    float basis_w[3]; float pad2;
    uint32_t axis;
    uint32_t tex_w;
    uint32_t tex_h;
    uint32_t double_sided;   // 1 = no backface culling (glTF material.doubleSided)
};
static_assert(sizeof(PatchInfoGPU) == 80);

struct BvhGPU {
    float amin[3]; float pad0;
    float amax[3]; float pad1;
    uint32_t val;
    uint32_t is_leaf;
    uint32_t pad2[2];
};
static_assert(sizeof(BvhGPU) == 48);
static_assert(offsetof(BvhGPU, val) == 32);
static_assert(offsetof(BvhGPU, is_leaf) == 36);

struct RayPassGPU {
    GLuint patch_info_ssbo = 0, bvh_ssbo = 0;
    GLuint ray_tex = 0;
    int w = 0, h = 0;
};

// The patch atlas axes come from project_along() (see gfx/coverage_atlas.cpp):
// the projection drops the depth coordinate, so the atlas-u / atlas-v world
// directions depend only on which two coordinates are kept.
static glm::vec3 patch_depth_axis(int axis) {
    switch (axis) {
        case 0: return { 1, 0, 0};
        case 1: return {-1, 0, 0};
        case 2: return { 0, 1, 0};
        case 3: return { 0,-1, 0};
        case 4: return { 0, 0, 1};
        default: return { 0, 0,-1};
    }
}
static void patch_axes(int axis, glm::vec3& bv, glm::vec3& bw) {
    switch (axis) {
        case 0: case 1: bv = {0,1,0}; bw = {0,0,1}; break;
        case 2: case 3: bv = {1,0,0}; bw = {0,0,1}; break;
        default:        bv = {1,0,0}; bw = {0,1,0}; break;   // 4,5
    }
}

static void upload_ray_pass_buffers(const gfx::CoverageAtlas& atlas, RayPassGPU& rp) {
    const auto& patches = atlas.patches();
    std::vector<PatchInfoGPU> pinfo(patches.size());
    for (size_t i = 0; i < patches.size(); ++i) {
        const auto& p = patches[i];
        PatchInfoGPU& g = pinfo[i];
        g.proj_min[0] = p.proj_min.x; g.proj_min[1] = p.proj_min.y;
        g.proj_size[0] = p.proj_size.x; g.proj_size[1] = p.proj_size.y;
        glm::vec3 bu = patch_depth_axis(p.axis), bv, bw;
        patch_axes(p.axis, bv, bw);
        g.basis_u[0] = bu.x; g.basis_u[1] = bu.y; g.basis_u[2] = bu.z;
        g.basis_v[0] = bv.x; g.basis_v[1] = bv.y; g.basis_v[2] = bv.z;
        g.basis_w[0] = bw.x; g.basis_w[1] = bw.y; g.basis_w[2] = bw.z;
        g.axis = uint32_t(p.axis);
        g.tex_w = uint32_t(p.tex_w); g.tex_h = uint32_t(p.tex_h);
        // Carry the decomposition's double_sided flag in the last uint slot;
        // the ray shader rejects back-side hits on single-sided patches only.
        g.double_sided = p.double_sided ? 1u : 0u;
    }
    if (!rp.patch_info_ssbo) glGenBuffers(1, &rp.patch_info_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, rp.patch_info_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, pinfo.size() * sizeof(PatchInfoGPU),
                 pinfo.data(), GL_STATIC_DRAW);

    const auto& nodes = atlas.bvh_nodes();
    std::vector<BvhGPU> bgpu(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& n = nodes[i];
        BvhGPU& g = bgpu[i];
        g.amin[0] = n.aabb_min.x; g.amin[1] = n.aabb_min.y; g.amin[2] = n.aabb_min.z;
        g.amax[0] = n.aabb_max.x; g.amax[1] = n.aabb_max.y; g.amax[2] = n.aabb_max.z;
        g.val = n.is_leaf ? n.patch_index : n.right_offset;
        g.is_leaf = n.is_leaf;
    }
    if (!rp.bvh_ssbo) glGenBuffers(1, &rp.bvh_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, rp.bvh_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bgpu.size() * sizeof(BvhGPU),
                 bgpu.data(), GL_STATIC_DRAW);
}

static void upload_patch_gpu(const gfx::CoverageAtlas& atlas, PatchGPU& g) {
    const auto& patch_pos = atlas.positions();
    const auto& patch_nrm = atlas.normals();
    const auto& tris = atlas.triangles();
    const auto& patches = atlas.patches();

    struct RenderVertex { float position[3]; float normal[3]; };
    std::vector<RenderVertex> render_verts(patch_pos.size());
    for (size_t i = 0; i < patch_pos.size(); ++i)
        render_verts[i] = {{patch_pos[i].x, patch_pos[i].y, patch_pos[i].z},
                           {patch_nrm[i].x, patch_nrm[i].y, patch_nrm[i].z}};

    std::vector<GLuint> render_indices;
    render_indices.reserve(tris.size() * 3);
    for (auto& p : patches)
        for (int ti : p.tris) {
            render_indices.push_back(tris[size_t(ti)].v[0]);
            render_indices.push_back(tris[size_t(ti)].v[1]);
            render_indices.push_back(tris[size_t(ti)].v[2]);
        }

    std::vector<gl::DrawElementsIndirectCommand> draw_cmds;
    {
        GLuint index_offset = 0;
        for (auto& p : patches) {
            GLuint count = GLuint(p.tris.size() * 3);
            draw_cmds.push_back({count, 1, index_offset, 0, 0});
            index_offset += count;
        }
    }

    if (!g.vao) glGenVertexArrays(1, &g.vao);
    if (!g.vbo) glGenBuffers(1, &g.vbo);
    if (!g.ebo) glGenBuffers(1, &g.ebo);
    if (!g.indirect) glGenBuffers(1, &g.indirect);
    if (!g.color_ssbo) glGenBuffers(1, &g.color_ssbo);

    glBindVertexArray(g.vao);
    glBindBuffer(GL_ARRAY_BUFFER, g.vbo);
    glBufferData(GL_ARRAY_BUFFER, render_verts.size() * sizeof(RenderVertex),
                 render_verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, render_indices.size() * sizeof(GLuint),
                 render_indices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex),
                          (void*)offsetof(RenderVertex, normal));
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, g.indirect);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, draw_cmds.size() * sizeof(gl::DrawElementsIndirectCommand),
                 draw_cmds.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);

    struct alignas(16) PatchColor { float r, g, b, a; };
    std::vector<PatchColor> patch_colors(patches.size());
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> s_dist(0.45f, 1.0f);
    std::uniform_real_distribution<float> v_dist(0.7f, 1.0f);
    auto rand_hue = [&rng]() {
        return std::fmod(std::fmod(float(rng()) * 2.3283064e-10f, 1.0f) + 1.0f, 1.0f);
    };
    for (size_t i = 0; i < patches.size(); ++i) {
        float h = rand_hue(), s = s_dist(rng), v = v_dist(rng);
        float c = v * s, hp = h * 6.0f, x = c * (1.0f - std::fabs(std::fmod(hp, 2.0f) - 1.0f));
        float r, gg, b;
        switch (int(hp)) {
            case 0: r = c; gg = x; b = 0; break;
            case 1: r = x; gg = c; b = 0; break;
            case 2: r = 0; gg = c; b = x; break;
            case 3: r = 0; gg = x; b = c; break;
            case 4: r = x; gg = 0; b = c; break;
            default: r = c; gg = 0; b = x; break;
        }
        float m = v - c;
        patch_colors[i] = {r + m, gg + m, b + m, 1.0f};
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, g.color_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, patch_colors.size() * sizeof(PatchColor),
                 patch_colors.data(), GL_STATIC_DRAW);

    g.draw_count = GLsizei(draw_cmds.size());
    g.patch_count = patches.size();
    g.tri_count = tris.size();
}

static void upload_chain_tex(const gfx::CoverageAtlas::MipChain& chain, GLuint tex) {
    const int MAX_W = 16384;
    int bpc = chain.bytes_per_channel;
    size_t ch = size_t(chain.channels);
    size_t bytes_per_node = ch * size_t(bpc);
    size_t total_nodes = 0;
    for (const auto& lv : chain.levels)
        total_nodes += lv.data.size() / bytes_per_node;
    size_t texel_bytes = (bpc == 2) ? 4 : 2;   // RG8 texel = 2 B, RG16 = 4 B
    GLsizei w = MAX_W;
    GLsizei h = GLsizei((total_nodes + MAX_W - 1) / MAX_W);
    if (h == 0) h = 1;
    std::vector<uint8_t> buf(size_t(w) * size_t(h) * texel_bytes, 0);
    size_t o = 0;
    for (const auto& lv : chain.levels) {
        size_t n = lv.data.size() / bytes_per_node;
        if (bpc == 2) {
            // Raw 16-bit pairs (dmin, dmax) already laid out as RG16 texels.
            std::memcpy(buf.data() + o, lv.data.data(), lv.data.size());
            o += lv.data.size();
        } else {
            for (size_t i = 0; i < n; ++i) {
                buf[o++] = lv.data[i * ch];
                buf[o++] = (ch == 2) ? lv.data[i * ch + 1] : 0;
            }
        }
    }
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (bpc == 2)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16, w, h, 0,
                     GL_RG, GL_UNSIGNED_SHORT, buf.data());
    else
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, w, h, 0,
                     GL_RG, GL_UNSIGNED_BYTE, buf.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
}

static void upload_atlas_view(const gfx::CoverageAtlas& atlas, AtlasView& v) {
    const auto& patches = atlas.patches();

    std::vector<glm::uvec4> rects;
    rects.reserve(patches.size());
    for (const auto& p : patches)
        rects.push_back({uint32_t(p.atlas_x), uint32_t(p.atlas_y),
                         uint32_t(p.tex_w), uint32_t(p.tex_h)});
    if (!v.rects_ssbo) glGenBuffers(1, &v.rects_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, v.rects_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, rects.size() * sizeof(glm::uvec4),
                 rects.data(), GL_STATIC_DRAW);

    // The three chains share one quadtree topology -> upload meta once.
    const gfx::CoverageAtlas::MipChain& ref = atlas.uv_chain();
    std::vector<uint32_t> meta;
    v.level_count = 0;
    for (const auto& lv : ref.levels) {
        if (v.level_count < 32) v.level_off[v.level_count] = int(meta.size());
        v.level_count++;
        meta.insert(meta.end(), lv.meta.begin(), lv.meta.end());
    }
    if (!v.meta_ssbo) glGenBuffers(1, &v.meta_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, v.meta_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, meta.size() * sizeof(uint32_t),
                 meta.data(), GL_STATIC_DRAW);

    const gfx::CoverageAtlas::MipChain* chains[4] = {&atlas.uv_chain(),
                                      &atlas.thickness_chain(),
                                      &atlas.depth_chain(),
                                      &atlas.normal_chain()};
    for (int c = 0; c < 4; ++c) {
        if (!v.value_tex[c]) glGenTextures(1, &v.value_tex[c]);
        upload_chain_tex(*chains[c], v.value_tex[c]);
        // value_off counts texels: each node is one RG texel regardless of bpc.
        uint32_t off = 0;
        for (int L = 0; L < v.level_count && L < 32; ++L) {
            v.value_off[c][L] = int(off);
            off += uint32_t(chains[c]->levels[size_t(L)].data.size() /
                            (size_t(chains[c]->channels) *
                             size_t(chains[c]->bytes_per_channel)));
        }
    }

    if (!v.view_tex) {
        glGenTextures(1, &v.view_tex);
        glGenFramebuffers(1, &v.fbo);
    }
    v.view_w = atlas.atlas_width();
    v.view_h = atlas.atlas_height();
    glBindTexture(GL_TEXTURE_2D, v.view_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, v.view_w, v.view_h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindFramebuffer(GL_FRAMEBUFFER, v.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           v.view_tex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

static bool build_program(gl::Program& prog, const char* vs_src, const char* fs_src) {
    gl::Shader vs(gl::ShaderType::vertex, vs_src);
    gl::Shader fs(gl::ShaderType::fragment, fs_src);
    if (!vs.compiled() || !fs.compiled()) {
        std::fprintf(stderr, "Shader compile failed:\n%s\n",
                     vs.compiled() ? fs.info_log().c_str() : vs.info_log().c_str());
        return false;
    }
    prog.attach(vs);
    prog.attach(fs);
    if (!prog.link()) {
        std::fprintf(stderr, "Program link failed:\n%s\n", prog.info_log().c_str());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// BVH debug: wireframe highlight + mouse ray-casting
// ---------------------------------------------------------------------------

struct RayHit {
    bool   leaf = false;          // true if a leaf node was hit
    int    node = -1;             // best (closest) hit node index
    int    patch = -1;            // patch owned by that leaf
    float  t = 1e30f;             // entry distance along the ray
    int    entered = -1;          // deepest entered node (fallback if no leaf)
    int    entered_depth = -1;
    std::vector<int> chain;       // working traversal path
    std::vector<int> best_chain;  // node path to the best leaf hit
};

static bool ray_aabb(const glm::vec3& ro, const glm::vec3& rd,
                     const glm::vec3& amin, const glm::vec3& amax,
                     float& t_in, float& t_out) {
    t_in = -1e30f;
    t_out = 1e30f;
    for (int c = 0; c < 3; ++c) {
        float o = ro[c], d = rd[c], lo = amin[c], hi = amax[c];
        if (std::fabs(d) < 1e-9f) {
            if (o < lo || o > hi) return false;
        } else {
            float inv = 1.0f / d;
            float t0 = (lo - o) * inv;
            float t1 = (hi - o) * inv;
            if (t0 > t1) std::swap(t0, t1);
            t_in  = std::max(t_in, t0);
            t_out = std::min(t_out, t1);
            if (t_in > t_out) return false;
        }
    }
    if (t_out < 0.0f) return false;   // box fully behind the ray origin
    t_in = std::max(t_in, 0.0f);
    return true;
}

static void bvh_trace(const std::vector<gfx::CoverageAtlas::BVHNode>& nodes,
                      int idx, const glm::vec3& ro, const glm::vec3& rd,
                      int depth, RayHit& out) {
    if (idx < 0 || idx >= int(nodes.size())) return;
    const auto& n = nodes[size_t(idx)];
    float ti, to;
    if (!ray_aabb(ro, rd, n.aabb_min, n.aabb_max, ti, to)) return;

    out.chain.push_back(idx);
    if (depth > out.entered_depth) { out.entered_depth = depth; out.entered = idx; }

    if (n.is_leaf) {
        if (ti < out.t) {
            out.t = ti;
            out.node = idx;
            out.patch = int(n.patch_index);
            out.leaf = true;
            out.best_chain = out.chain;
        }
    } else {
        int right = idx + int(n.right_offset);
        bvh_trace(nodes, idx + 1, ro, rd, depth + 1, out);
        if (right > idx + 1 && right < int(nodes.size()))
            bvh_trace(nodes, right, ro, rd, depth + 1, out);
    }
    out.chain.pop_back();
}

// Builds a world-space ray from a cursor position (GLFW top-left origin).
static glm::vec3 screen_ray(const gfx::Camera& cam, double mx, double my,
                            int w, int h, glm::vec3& origin) {
    float ndc_x = float(2.0 * mx / double(w) - 1.0);
    float ndc_y = float(1.0 - 2.0 * my / double(h));
    glm::vec4 clip(ndc_x, ndc_y, 1.0f, 1.0f);
    glm::vec4 world = glm::inverse(cam.view_projection()) * clip;
    world /= world.w;
    origin = cam.position();
    return glm::normalize(glm::vec3(world) - origin);
}

// ---------------------------------------------------------------------------
// Model switching
// ---------------------------------------------------------------------------

struct ModelEntry {
    const char* name;       // ImGui label
    const char* glb_path;   // relative to the example working dir
    const char* cache_dir;  // where the preprocessed MDC outputs are cached
};

static const ModelEntry kModels[] = {
    {"Cornell Box",    "CornellBoxOriginal.glb", "cache/cornell"},
    {"Stanford Bunny", "Stanford_Bunny.glb",     "cache/bunny"},
    {"Stanford Dragon","Stanford_Dragon.glb",    "cache/dragon"},
};
static constexpr int kModelCount = int(sizeof(kModels) / sizeof(kModels[0]));

// Must match LEAF_TILE in the ray + atlas GLSL. Caches built with a different
// leaf tile have a different quadtree layout and must be rebuilt.
static constexpr int kShaderLeafTile = 1;

// GPU/CPU state that is (re)built whenever the active model changes.
struct SceneState {
    gfx::CoverageAtlas atlas;
    PatchGPU  patch_gpu;
    AtlasView atlas_view;
    RayPassGPU ray_pass;

    float    ray_depth_lo = 0.0f, ray_depth_hi = 0.0f, ray_thick_max = 0.0f;
    glm::vec3 ray_depth_origin{0.0f};

    std::vector<int> lod0_meshes;   // meshes the geometry pass draws
    glm::vec3 cam_target{0.0f, 1.0f, 0.0f};
    float     cam_dist = 3.8f;
};

// Mirrors CoverageAtlas::build(): every mesh not inside a LOD group, plus each
// group's LOD0 mesh. Prevents the bunny's LOD duplicates from z-fighting.
static std::vector<int> lod0_mesh_list(const gfx::Model& model) {
    std::vector<bool> in_lod(model.mesh_count(), false);
    for (size_t g = 0; g < model.lod_group_count(); ++g)
        for (int mi : model.lod_group(g).mesh_indices)
            in_lod[size_t(mi)] = true;
    std::vector<int> use;
    for (size_t m = 0; m < model.mesh_count(); ++m)
        if (!in_lod[m]) use.push_back(int(m));
    for (size_t g = 0; g < model.lod_group_count(); ++g)
        use.push_back(model.lod_group(g).mesh_indices[0]);
    return use;
}

static const gfx::ModelMaterialInfo& mesh_material_or_default(const gfx::Model& model,
                                                              int mesh_idx) {
    int mi = model.mesh_material(size_t(mesh_idx));
    if (mi >= 0 && size_t(mi) < model.material_count())
        return model.material_info(size_t(mi));
    static gfx::ModelMaterialInfo def;   // white, opaque, double_sided=false
    return def;
}

// Loads a model's glb (fast) and builds the MDC atlas only when no cached
// preprocessed snapshot exists. Uploads every GPU buffer and re-derives the
// ray-pass constants + camera framing.
static bool load_scene(const ModelEntry& entry, gfx::Model& model, SceneState& st) {
    if (!model.load(entry.glb_path)) {
        std::fprintf(stderr, "Failed to load %s\n", entry.glb_path);
        return false;
    }
    std::printf("Loaded %s: %zu meshes, %zu materials, %zu textures, %zu LOD groups\n",
                entry.name, model.mesh_count(), model.material_count(),
                model.texture_count(), model.lod_group_count());

    // Per-model texel density: scale to the model's own size so every model
    // (big or small) lands on a comparable absolute resolution. A fixed
    // per-unit density leaves a small object (e.g. the bunny) with sub-texel
    // triangles, whose rasterisation never marks a texel "covered" — then
    // query_node reports a miss and the ray pass shows a literal gap. The
    // extent comes from the same LOD0 mesh list the atlas build uses (the
    // rotate_model_x rotation is isometric and does not change the span).
    float want_density = 0.0f;
    {
        std::vector<int> use = lod0_mesh_list(model);
        glm::vec3 lo(1e30f), hi(-1e30f);
        for (int mi : use) {
            const auto& mesh = model.mesh(size_t(mi));
            for (size_t v = 0; v < mesh.vertex_count(); ++v) {
                const float* pos = mesh.vertices()[v].position;
                glm::vec3 p(pos[0], pos[1], pos[2]);
                lo = glm::min(lo, p);
                hi = glm::max(hi, p);
            }
        }
        float span = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
        if (span > 1e-6f)
            want_density = float(st.atlas.config().auto_target) / span;
    }

    bool loaded = st.atlas.load_files(entry.cache_dir);
    bool leaf_ok = st.atlas.config().mip_leaf_tile == kShaderLeafTile;
    bool density_ok = want_density <= 0.0f ||
                      std::fabs(st.atlas.config().texel_density - want_density) < 1e-3f;
    if (!loaded || !leaf_ok || !density_ok) {
        if (!loaded) {
            std::printf("No cached MDC snapshot for %s — building atlas...\n", entry.name);
        } else if (!leaf_ok) {
            std::printf("Cached MDC snapshot for %s uses leaf_tile=%d but the "
                        "shader needs %d — rebuilding...\n", entry.name,
                        st.atlas.config().mip_leaf_tile, kShaderLeafTile);
        } else {
            std::printf("Cached MDC snapshot for %s was built at %.0f texels/unit "
                        "but the model needs %.0f — rebuilding...\n", entry.name,
                        st.atlas.config().texel_density, want_density);
        }
        gfx::CoverageAtlasConfig cfg = st.atlas.config();
        if (!leaf_ok) cfg.mip_leaf_tile = kShaderLeafTile;
        if (want_density > 0.0f) cfg.texel_density = want_density;
        st.atlas.set_config(cfg);
        if (!st.atlas.build(model)) {
            std::fprintf(stderr, "CoverageAtlas build failed\n");
            return false;
        }
        st.atlas.write_files(entry.cache_dir);
    }

    st.lod0_meshes = lod0_mesh_list(model);

    upload_patch_gpu(st.atlas, st.patch_gpu);
    upload_atlas_view(st.atlas, st.atlas_view);
    upload_ray_pass_buffers(st.atlas, st.ray_pass);

    st.ray_depth_lo = st.atlas.depth_chain().qmin[0];
    st.ray_depth_hi = st.atlas.depth_chain().qmax[0];
    st.ray_thick_max = st.atlas.thickness_chain().qmax[0];
    st.ray_depth_origin = glm::vec3(1e30f);
    for (const auto& v : st.atlas.positions())
        st.ray_depth_origin = glm::min(st.ray_depth_origin, v);

    // Frame the camera on the model's AABB.
    if (!st.atlas.positions().empty()) {
        glm::vec3 lo(1e30f), hi(-1e30f);
        for (const auto& v : st.atlas.positions()) { lo = glm::min(lo, v); hi = glm::max(hi, v); }
        st.cam_target = (lo + hi) * 0.5f;
        float extent = glm::length(hi - lo);
        st.cam_dist = 0.5f * extent + 3.0f;
    }
    std::printf("Scene ready: %zu patches, %zu triangles, atlas %dx%d @ %.0f texels/unit\n",
                st.atlas.patches().size(), st.atlas.triangles().size(),
                st.atlas.atlas_width(), st.atlas.atlas_height(), st.atlas.final_density());
    return true;
}

int main() {
    gllib::log_to_stderr(gllib::LogLevel::info);

    gfx::Window window({"32 MDC Lighting", 1280, 720});
    window.vsync(false);

    // --- Scene: load glb + build/load MDC atlas, upload every GPU buffer ---
    gfx::Model model;
    SceneState st;
    int cur_model = 0;
    if (const char* me = getenv("MODEL")) cur_model = std::max(0, std::min(atoi(me), kModelCount - 1));
    if (!load_scene(kModels[cur_model], model, st)) return EXIT_FAILURE;

    // Aliases so the render loop below reads cleanly.
    auto& atlas            = st.atlas;
    auto& patch_gpu        = st.patch_gpu;
    auto& atlas_view       = st.atlas_view;
    auto& ray_pass         = st.ray_pass;
    auto& ray_depth_lo     = st.ray_depth_lo;
    auto& ray_depth_hi     = st.ray_depth_hi;
    auto& ray_thick_max    = st.ray_thick_max;
    auto& ray_depth_origin = st.ray_depth_origin;
    auto& cam_target       = st.cam_target;
    auto& cam_dist         = st.cam_dist;

    // TEMP: dump triangle geometry + patch mapping for the CPU reference tracer.
    if (getenv("DUMPTRIS")) {
        const auto& pos = atlas.positions();
        const auto& tris = atlas.triangles();
        const auto& pats = atlas.patches();
        std::vector<uint32_t> tri_patch(tris.size(), 0);
        for (size_t i = 0; i < pats.size(); ++i)
            for (int ti : pats[i].tris) tri_patch[size_t(ti)] = uint32_t(i);
        FILE* f = fopen("/tmp/opencode/tris.bin", "wb");
        uint32_t np = uint32_t(pos.size()), nt = uint32_t(tris.size());
        fwrite(&np, 4, 1, f);
        fwrite(&nt, 4, 1, f);
        for (const auto& v : pos) fwrite(glm::value_ptr(v), 4, 3, f);
        for (const auto& t : tris) fwrite(t.v, 4, 3, f);
        fwrite(tri_patch.data(), 4, tri_patch.size(), f);
        for (size_t i = 0; i < pats.size(); ++i) {
            uint32_t axis = uint32_t(pats[i].axis);
            fwrite(&axis, 4, 1, f);
            fwrite(glm::value_ptr(pats[i].aabb_min), 4, 3, f);
            fwrite(glm::value_ptr(pats[i].aabb_max), 4, 3, f);
            fwrite(glm::value_ptr(pats[i].proj_min), 4, 2, f);
            fwrite(glm::value_ptr(pats[i].proj_size), 4, 2, f);
        }
        // BVH nodes: 3f min, 3f max, u32 val, u32 is_leaf
        uint32_t nb = uint32_t(atlas.bvh_nodes().size());
        fwrite(&nb, 4, 1, f);
        for (const auto& n : atlas.bvh_nodes()) {
            fwrite(glm::value_ptr(n.aabb_min), 4, 3, f);
            fwrite(glm::value_ptr(n.aabb_max), 4, 3, f);
            uint32_t val = n.is_leaf ? n.patch_index : n.right_offset;
            fwrite(&val, 4, 1, f);
            fwrite(&n.is_leaf, 4, 1, f);
        }
        fclose(f);
        std::printf("DUMPTRIS: %u pos, %u tris, %zu patches -> /tmp/opencode/tris.bin\n",
                    np, nt, pats.size());
    }

    // --- Shaders ---
    gl::Program geo_prog, light_prog, patch_prog, atlas_prog;
    if (!build_program(geo_prog, geo_vs, geo_fs)) return EXIT_FAILURE;
    if (!build_program(light_prog, light_vs, light_fs)) return EXIT_FAILURE;
    if (!build_program(patch_prog, patch_vs, patch_fs)) return EXIT_FAILURE;
    if (!build_program(atlas_prog, atlas_vs, atlas_fs)) return EXIT_FAILURE;

    // Primary-ray compute pass + its fullscreen display.
    gl::Shader ray_cs_shader(gl::ShaderType::compute, ray_cs);
    if (!ray_cs_shader.compiled()) {
        std::fprintf(stderr, "Compute shader compile failed:\n%s\n",
                     ray_cs_shader.info_log().c_str());
        return EXIT_FAILURE;
    }
    gl::Program ray_prog;
    ray_prog.attach(ray_cs_shader);
    if (!ray_prog.link()) {
        std::fprintf(stderr, "Compute program link failed:\n%s\n",
                     ray_prog.info_log().c_str());
        return EXIT_FAILURE;
    }
    gl::Program ray_disp_prog;
    if (!build_program(ray_disp_prog, light_vs, ray_display_fs)) return EXIT_FAILURE;

    // --- Fullscreen quad ---
    const float quad_verts[] = {-1, -1, 1, -1, 1, 1, -1, 1};
    const unsigned int quad_idx[] = {0, 1, 2, 0, 2, 3};
    gl::Buffer quad_vbo(gl::BufferType::vertex);
    quad_vbo.data(quad_verts, sizeof(quad_verts));
    gl::Buffer quad_ebo(gl::BufferType::index);
    quad_ebo.data(quad_idx, sizeof(quad_idx));
    gl::VertexArray quad_vao;
    quad_vao.bind();
    quad_vbo.bind();
    quad_ebo.bind();
    quad_vao.attrib_pointer(0, 2, GL_FLOAT, false, 8, (void*)0);
    quad_vao.enable_attrib(0);
    gl::VertexArray::unbind();

    // --- GBuffer ---
    gfx::GBuffer gbuf;
    gbuf.create(window.framebuffer_width(), window.framebuffer_height());

    // --- ImGui ---
    gfx::ImGuiOverlay gui;
    if (!gui.init(window)) {
        std::fprintf(stderr, "ImGui init failed\n");
        return EXIT_FAILURE;
    }

    // --- Camera looking into the scene (framed on the active model) ---
    gfx::Camera cam;
    cam.perspective(45.0f, float(window.framebuffer_width()) / float(window.framebuffer_height()), 0.1f, 100.0f);
    cam.look_at(cam_target + glm::vec3(0.0f, 0.0f, cam_dist), cam_target);

    // Rotate the box so its open face (+Y in model space) points at the camera.
    glm::mat4 model_mat = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));
    glm::mat3 normal_mat = glm::transpose(glm::inverse(glm::mat3(model_mat)));

    bool show_patches = false;
    int debug_view = 0;
    const char* chain_env = getenv("ATLAS_CHAIN");
    int atlas_chain = chain_env ? atoi(chain_env) : 0;   // 0 = UV, 1 = thickness, 2 = depth, 3 = normal
    float density_scale = 1.0f;
    float base_density = atlas.config().texel_density;

    // --- BVH debug visualization ---
    gfx::DebugDraw bvh_draw;
    bool show_bvh = true;
    bool show_bvh_occ = true;     // occlude boxes by the scene depth
    RayHit pick;                  // last mouse pick result
    bool left_prev = false;
    bool press_in_view = false;
    double press_x = 0.0, press_y = 0.0;

    // --- Primary-ray pass ---
    bool show_ray_pass = true;    // fullscreen normal view
    float ray_connect_tol = 0.05f;
    float ray_eps = 0.02f;
    int ray_dbg = 0;
    const char* rdbg_env = getenv("RAYDBG");
    if (rdbg_env) ray_dbg = atoi(rdbg_env);

    double prev_x, prev_y;
    window.cursor_position(prev_x, prev_y);

    gl::enable(GL_DEPTH_TEST);

    std::vector<float> frame_times;
    double last = window.time();
    int frame = 0;

    // TEMP: scripted orbit sweep for grazing-angle diagnostics (SWEEP=<frames>).
    int sweep_total = 0;
    int sweep_frame = 0;
    if (const char* se = getenv("SWEEP")) sweep_total = std::max(1, atoi(se));

    while (!window.should_close()) {
        ++frame;
        double now = window.time();
        float dt = float(now - last);
        last = now;

        window.poll_events();
        gui.begin_frame();

        if (sweep_total && sweep_frame >= sweep_total) break;
        if (sweep_total) {
            float ang = 2.0f * 3.14159265f * float(sweep_frame) / float(sweep_total);
            float r = 3.8f;
            glm::vec3 eye(r * std::cos(ang), 1.0f, r * std::sin(ang));
            cam.look_at(eye, glm::vec3(0.0f, 1.0f, 0.0f));
        }

        if (frame_times.size() >= 120) frame_times.erase(frame_times.begin());
        frame_times.push_back(dt * 1000.0f);

        // Orbit (left drag) + zoom (scroll), unless the ImGui window grabs mouse
        if (!gui.wants_mouse()) {
            if (window.mouse_down(gfx::MouseButton::left)) {
                double cx, cy;
                window.cursor_position(cx, cy);
                cam.orbit(float(cx - prev_x) * 0.005f, float(prev_y - cy) * 0.005f);
                prev_x = cx;
                prev_y = cy;
            } else {
                window.cursor_position(prev_x, prev_y);
            }
            double scroll = window.scroll_delta();
            if (scroll != 0.0) cam.zoom(float(scroll) * 0.1f);
        } else {
            window.scroll_delta();
        }

        // Left click in the 3D view: ray-cast against the BVH. Only active
        // while the wireframe is shown. Clicks that press or release over an
        // ImGui window never pick.
        bool left_now = window.mouse_down(gfx::MouseButton::left);
        if (left_now && !left_prev && !gui.wants_mouse() && show_bvh) {
            press_in_view = true;
            window.cursor_position(press_x, press_y);
        } else if (!left_now && left_prev) {
            if (press_in_view && !gui.wants_mouse()) {
                double rx, ry;
                window.cursor_position(rx, ry);
                if (std::hypot(rx - press_x, ry - press_y) < 6.0) {
                    glm::vec3 ro, rd;
                    rd = screen_ray(cam, rx, ry, window.width(), window.height(), ro);
                    RayHit hit;
                    bvh_trace(atlas.bvh_nodes(), 0, ro, rd, 0, hit);
                    if (hit.leaf) {
                        pick = hit;
                    } else if (hit.entered >= 0) {
                        pick = RayHit{};
                        pick.node = hit.entered;
                        pick.entered = hit.entered;
                        pick.entered_depth = hit.entered_depth;
                        pick.leaf = false;
                        pick.patch = -1;
                    } else {
                        pick = RayHit{};
                    }
                }
            }
            press_in_view = false;
        }
        left_prev = left_now;

        cam.set_aspect(float(window.framebuffer_width()) / float(window.framebuffer_height()));

        if (window.framebuffer_width() != gbuf.width() || window.framebuffer_height() != gbuf.height())
            gbuf.create(window.framebuffer_width(), window.framebuffer_height());

        glm::mat4 vp = cam.view_projection();

        // --- Geometry pass ---
        gbuf.bind_for_geometry();
        geo_prog.use();
        auto geo_loc = [&](const char* n) { return geo_prog.uniform_location(n); };
        GLint loc;

        loc = geo_loc("u_view_proj");
        if (loc >= 0) geo_prog.uniform_matrix4fv(loc, glm::value_ptr(vp));

        for (int mi : st.lod0_meshes) {
            const auto& mat = mesh_material_or_default(model, mi);

            loc = geo_loc("u_model");
            if (loc >= 0) geo_prog.uniform_matrix4fv(loc, glm::value_ptr(model_mat));
            loc = geo_loc("u_normal_mat");
            if (loc >= 0) geo_prog.uniform_matrix3fv(loc, glm::value_ptr(normal_mat));
            loc = geo_loc("u_albedo");
            if (loc >= 0) geo_prog.uniform3fv(loc, mat.base_color_factor);
            loc = geo_loc("u_metallic");
            if (loc >= 0) geo_prog.uniform1f(loc, mat.metallic_factor);
            loc = geo_loc("u_roughness");
            if (loc >= 0) geo_prog.uniform1f(loc, mat.roughness_factor);
            loc = geo_loc("u_ao");
            if (loc >= 0) geo_prog.uniform1f(loc, 1.0f);
            loc = geo_loc("u_emissive");
            if (loc >= 0) geo_prog.uniform3fv(loc, mat.emissive_factor);

            bool emissive = mat.emissive_factor[0] > 0.0f ||
                            mat.emissive_factor[1] > 0.0f ||
                            mat.emissive_factor[2] > 0.0f;
            loc = geo_loc("u_emissive_strength");
            if (loc >= 0) geo_prog.uniform1f(loc, emissive ? 17.0f : 0.0f);

            model.mesh(size_t(mi)).draw();
        }

        gl::disable(GL_DEPTH_TEST);

        // --- Lighting / patch pass ---
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        gl::viewport(0, 0, window.framebuffer_width(), window.framebuffer_height());
        gl::clear_color(0.03f, 0.03f, 0.05f, 1.0f);
        gl::clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (show_patches) {
            gl::enable(GL_DEPTH_TEST);
            patch_prog.use();
            loc = patch_prog.uniform_location("u_vp");
            if (loc >= 0) patch_prog.uniform_matrix4fv(loc, glm::value_ptr(vp));
            glBindVertexArray(patch_gpu.vao);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, patch_gpu.color_ssbo);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, patch_gpu.indirect);
            gl::multi_draw_elements_indirect(GL_TRIANGLES, GL_UNSIGNED_INT,
                                             nullptr, patch_gpu.draw_count, 0);
            glBindVertexArray(0);
        } else {
            gbuf.bind_for_lighting(3, 4, 5, 6);

            light_prog.use();
            auto light_loc = [&](const char* n) { return light_prog.uniform_location(n); };
            loc = light_loc("u_albedo_tex");      if (loc >= 0) light_prog.uniform1i(loc, 3);
            loc = light_loc("u_normal_rm_tex");    if (loc >= 0) light_prog.uniform1i(loc, 4);
            loc = light_loc("u_emissive_ao_tex");  if (loc >= 0) light_prog.uniform1i(loc, 5);
            loc = light_loc("u_depth_tex");        if (loc >= 0) light_prog.uniform1i(loc, 6);
            loc = light_loc("u_debug_view");       if (loc >= 0) light_prog.uniform1i(loc, debug_view);

            quad_vao.bind();
            gl::draw_elements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        }

        // --- Primary-ray pass: BVH traversal + atlas mip-chain refinement ---
        {
            if (!ray_pass.ray_tex || ray_pass.w != window.framebuffer_width() ||
                ray_pass.h != window.framebuffer_height()) {
                if (!ray_pass.ray_tex) glGenTextures(1, &ray_pass.ray_tex);
                ray_pass.w = window.framebuffer_width();
                ray_pass.h = window.framebuffer_height();
                glBindTexture(GL_TEXTURE_2D, ray_pass.ray_tex);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ray_pass.w, ray_pass.h, 0,
                             GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glBindTexture(GL_TEXTURE_2D, 0);
            }

            ray_prog.use();
            auto r_loc = [&](const char* n) { return ray_prog.uniform_location(n); };
            loc = r_loc("u_vp_inv");
            if (loc >= 0) ray_prog.uniform_matrix4fv(loc, glm::value_ptr(glm::inverse(vp)));
            loc = r_loc("u_res");        if (loc >= 0) ray_prog.uniform2f(loc, float(ray_pass.w), float(ray_pass.h));
            loc = r_loc("u_ro");         if (loc >= 0) ray_prog.uniform3fv(loc, glm::value_ptr(cam.position()));
            loc = r_loc("u_depth_origin"); if (loc >= 0) ray_prog.uniform3fv(loc, glm::value_ptr(ray_depth_origin));
            loc = r_loc("u_depth_lo");   if (loc >= 0) ray_prog.uniform1f(loc, ray_depth_lo);
            loc = r_loc("u_depth_hi");   if (loc >= 0) ray_prog.uniform1f(loc, ray_depth_hi);
            loc = r_loc("u_thick_max");  if (loc >= 0) ray_prog.uniform1f(loc, ray_thick_max);
            loc = r_loc("u_connect_tol"); if (loc >= 0) ray_prog.uniform1f(loc, ray_connect_tol);
            loc = r_loc("u_eps");        if (loc >= 0) ray_prog.uniform1f(loc, ray_eps);
            loc = r_loc("u_dbg");        if (loc >= 0) ray_prog.uniform1i(loc, ray_dbg);
            loc = r_loc("u_patch_count"); if (loc >= 0) ray_prog.uniform1i(loc, int(patch_gpu.patch_count));
            loc = r_loc("u_level_off");  if (loc >= 0) glUniform1iv(loc, 32, atlas_view.level_off);
            loc = r_loc("u_value_off_depth");  if (loc >= 0) glUniform1iv(loc, 32, atlas_view.value_off[2]);
            loc = r_loc("u_value_off_thick");  if (loc >= 0) glUniform1iv(loc, 32, atlas_view.value_off[1]);
            loc = r_loc("u_value_off_normal"); if (loc >= 0) glUniform1iv(loc, 32, atlas_view.value_off[3]);
            loc = r_loc("u_depth_tex");  if (loc >= 0) ray_prog.uniform1i(loc, 0);
            loc = r_loc("u_thick_tex");  if (loc >= 0) ray_prog.uniform1i(loc, 1);
            loc = r_loc("u_normal_tex"); if (loc >= 0) ray_prog.uniform1i(loc, 2);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, atlas_view.value_tex[2]);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, atlas_view.value_tex[1]);
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, atlas_view.value_tex[3]);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, atlas_view.rects_ssbo);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, atlas_view.meta_ssbo);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ray_pass.patch_info_ssbo);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ray_pass.bvh_ssbo);
            glBindImageTexture(0, ray_pass.ray_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
            gl::dispatch_compute((ray_pass.w + 15) / 16, (ray_pass.h + 15) / 16, 1);
            gl::memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                               GL_TEXTURE_FETCH_BARRIER_BIT);
            glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);

            // TEMP: dump the ray-pass output for the sweep.
            if (sweep_total) {
                char path[128];
                std::snprintf(path, sizeof(path), "/tmp/opencode/sweep_%03d.ppm", sweep_frame);
                glBindTexture(GL_TEXTURE_2D, ray_pass.ray_tex);
                glPixelStorei(GL_PACK_ALIGNMENT, 1);
                std::vector<uint8_t> px(size_t(ray_pass.w) * ray_pass.h * 3);
                glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, px.data());
                FILE* f = fopen(path, "wb");
                fprintf(f, "P6\n%d %d\n255\n", ray_pass.w, ray_pass.h);
                fwrite(px.data(), 1, px.size(), f);
                fclose(f);
                ++sweep_frame;
                if (sweep_frame >= sweep_total) {
                    FILE* done = fopen("/tmp/opencode/sweep_done", "wb");
                    if (done) fclose(done);
                }
            }

            GLint upc = -1, ures[2] = {0, 0};
            GLint loc2 = r_loc("u_patch_count");
            if (loc2 >= 0) glGetUniformiv(ray_prog.handle(), loc2, &upc);
            loc2 = r_loc("u_res");
            if (loc2 >= 0) glGetUniformiv(ray_prog.handle(), loc2, ures);
            GLint64 sz3 = 0, sz0 = 0;
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, atlas_view.rects_ssbo);
            glGetBufferParameteri64v(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &sz0);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ray_pass.bvh_ssbo);
            glGetBufferParameteri64v(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &sz3);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            //printf("CHECK: u_patch_count=%d u_res=%d,%d rects_ssbo=%lld bvh_ssbo=%lld\n",
            //       upc, ures[0], ures[1], (long long)sz0, (long long)sz3);

            if (getenv("RAYDUMP") && frame <= 2) {
                char path[128];
                std::snprintf(path, sizeof(path), "/tmp/opencode/ray_%d.ppm", frame);
                std::vector<uint8_t> px(size_t(ray_pass.w) * ray_pass.h * 3);
                glBindTexture(GL_TEXTURE_2D, ray_pass.ray_tex);
                glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, px.data());
                FILE* f = fopen(path, "wb");
                fprintf(f, "P6\n%d %d\n255\n", ray_pass.w, ray_pass.h);
                fwrite(px.data(), 1, px.size(), f);
                fclose(f);
                printf("RAYDUMP: wrote %s\n", path);
                if (frame == 1) {
                    std::vector<BvhGPU> bv(atlas.bvh_nodes().size());
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ray_pass.bvh_ssbo);
                    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, atlas.bvh_nodes().size() * sizeof(BvhGPU), bv.data());
                    FILE* bf = fopen("/tmp/opencode/gpu_bvh.bin", "wb");
                    fwrite(bv.data(), sizeof(BvhGPU), bv.size(), bf);
                    fclose(bf);
                    std::vector<PatchInfoGPU> pi(patch_gpu.patch_count);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ray_pass.patch_info_ssbo);
                    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, patch_gpu.patch_count * sizeof(PatchInfoGPU), pi.data());
                    FILE* pf = fopen("/tmp/opencode/gpu_patch.bin", "wb");
                    fwrite(pi.data(), sizeof(PatchInfoGPU), pi.size(), pf);
                    fclose(pf);
                    glm::mat4 vpinv = glm::inverse(vp);
                    FILE* cf = fopen("/tmp/opencode/gpu_cam.bin", "wb");
                    fwrite(glm::value_ptr(vpinv), 16, 4, cf);
                    glm::vec3 cpos = cam.position();
                    fwrite(glm::value_ptr(cpos), 3, 4, cf);
                    glm::vec3 dor = ray_depth_origin;
                    fwrite(glm::value_ptr(dor), 3, 4, cf);
                    fwrite(&ray_depth_lo, 4, 1, cf);
                    fwrite(&ray_depth_hi, 4, 1, cf);
                    fwrite(&ray_thick_max, 4, 1, cf);
                    fwrite(&ray_eps, 4, 1, cf);
                    fclose(cf);
                    printf("DUMP: wrote gpu_bvh/gpu_patch/gpu_cam.bin\n");
                }
            }

            if (show_ray_pass) {
                gl::disable(GL_DEPTH_TEST);
                ray_disp_prog.use();
                loc = ray_disp_prog.uniform_location("u_tex");
                if (loc >= 0) ray_disp_prog.uniform1i(loc, 0);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, ray_pass.ray_tex);
                quad_vao.bind();
                gl::draw_elements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
            }
        }

        // --- Atlas texture reconstruction (full MIP4 chain walk, into view FBO) ---
        if (atlas_view.view_tex) {
            glBindFramebuffer(GL_FRAMEBUFFER, atlas_view.fbo);
            gl::viewport(0, 0, atlas_view.view_w, atlas_view.view_h);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            gl::disable(GL_DEPTH_TEST);

            atlas_prog.use();
            auto a_loc = [&](const char* n) { return atlas_prog.uniform_location(n); };
            loc = a_loc("u_patch_count"); if (loc >= 0) atlas_prog.uniform1i(loc, int(patch_gpu.patch_count));
            loc = a_loc("u_chain");       if (loc >= 0) atlas_prog.uniform1i(loc, atlas_chain);
            loc = a_loc("u_debug_meta");  if (loc >= 0) atlas_prog.uniform1i(loc, getenv("DUMP") ? atoi(getenv("DUMPMETA") ? getenv("DUMPMETA") : "0") : 0);
            loc = a_loc("u_level_off");   if (loc >= 0) glUniform1iv(loc, 32, atlas_view.level_off);
            loc = a_loc("u_value_off");   if (loc >= 0) glUniform1iv(loc, 32, atlas_view.value_off[atlas_chain]);
            loc = a_loc("u_values");      if (loc >= 0) atlas_prog.uniform1i(loc, 0);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, atlas_view.value_tex[atlas_chain]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, atlas_view.rects_ssbo);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, atlas_view.meta_ssbo);
            quad_vao.bind();
            if (frame == 0) {
                GLint vb[4] = {};
                glGetIntegerv(GL_VIEWPORT, vb);
                printf("DUMP: atlas draw viewport=%d,%d,%d,%d\n", vb[0], vb[1], vb[2], vb[3]);
            }
            gl::draw_elements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            gl::viewport(0, 0, window.framebuffer_width(), window.framebuffer_height());
        }

        // --- TEMP: dump atlas view texture for debugging ---
        if (getenv("DUMP") && frame < 3) {
            if (frame == 1) {
                {
                    const auto& uv = atlas.uv_chain();
                    printf("UV CHAIN: channels=%d levels=%zu leaf_tile=%d qmin=%.3f,%.3f qmax=%.3f,%.3f\n",
                           uv.channels, uv.levels.size(), uv.leaf_tile, uv.qmin[0], uv.qmin[1], uv.qmax[0], uv.qmax[1]);
                    for (size_t L = 0; L < uv.levels.size(); ++L) {
                        const auto& lv = uv.levels[L];
                        size_t nn = lv.data.size() / 2;
                        int nz = 0, mx = 0;
                        for (size_t i = 0; i < nn; ++i) {
                            int v = lv.data[i*2] | lv.data[i*2+1];
                            if (v) { nz++; mx = std::max<int>(mx, lv.data[i*2]); }
                        }
                        printf("  UV level %zu: %zu nodes, nz=%zu/%zu max=%d\n", L, nn, size_t(nz), nn, mx);
                    }
                }
                printf("CPU PATCHES: %zu\n", atlas.patches().size());
                for (int i = 0; i < (int)atlas.patches().size(); ++i) {
                    const auto& p = atlas.patches()[size_t(i)];
                    printf("  CPU patch %d: x=%d y=%d w=%d h=%d axis=%d amin=(%.3f,%.3f,%.3f) amax=(%.3f,%.3f,%.3f) pmn=(%.3f,%.3f) psz=(%.3f,%.3f)\n",
                           i, p.atlas_x, p.atlas_y, p.tex_w, p.tex_h, p.axis,
                           p.aabb_min.x, p.aabb_min.y, p.aabb_min.z,
                           p.aabb_max.x, p.aabb_max.y, p.aabb_max.z,
                           p.proj_min.x, p.proj_min.y, p.proj_size.x, p.proj_size.y);
                }
                const auto& cbn = atlas.bvh_nodes();
                for (size_t i = 0; i < cbn.size(); ++i)
                    printf("  CPU BVH %zu: amin=(%.3f,%.3f,%.3f) amax=(%.3f,%.3f,%.3f) leaf=%d pid=%d\n",
                           i, cbn[i].aabb_min.x, cbn[i].aabb_min.y, cbn[i].aabb_min.z,
                           cbn[i].aabb_max.x, cbn[i].aabb_max.y, cbn[i].aabb_max.z,
                           int(cbn[i].is_leaf), int(cbn[i].is_leaf ? cbn[i].patch_index : -1));
                std::vector<glm::uvec4> rbuf(atlas_view.rects_ssbo ? patch_gpu.patch_count : 0);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, atlas_view.rects_ssbo);
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, patch_gpu.patch_count * sizeof(glm::uvec4), rbuf.data());
                for (size_t i = 0; i < rbuf.size(); ++i)
                    printf("RECT %zu: x=%u y=%u w=%u h=%u\n", i, rbuf[i].x, rbuf[i].y, rbuf[i].z, rbuf[i].w);
                std::vector<uint32_t> mbuf(atlas_view.meta_ssbo ? 16 : 0);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, atlas_view.meta_ssbo);
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 16 * sizeof(uint32_t), mbuf.data());
                for (size_t i = 0; i < mbuf.size() && i < patch_gpu.patch_count; ++i)
                    printf("META0 %zu: 0x%08x\n", i, mbuf[i]);
                printf("PATCHCOUNT: %zu  LEVELS: %d\n", patch_gpu.patch_count, atlas_view.level_count);
                {
                    int tw = 0, th = 0;
                    glBindTexture(GL_TEXTURE_2D, atlas_view.view_tex);
                    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &tw);
                    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &th);
                    GLint vb[4] = {};
                    glGetIntegerv(GL_VIEWPORT, vb);
                    GLint fbs[2] = {};
                    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                                          GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, fbs);
                    printf("DUMP: view_tex=%dx%d view_w/h=%dx%d vp=%d,%d,%d,%d\n",
                           tw, th, atlas_view.view_w, atlas_view.view_h, vb[0], vb[1], vb[2], vb[3]);
                }
                {
                    int tex_w = 0, tex_h = 0;
                    glBindTexture(GL_TEXTURE_2D, atlas_view.value_tex[0]);
                    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &tex_w);
                    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &tex_h);
                    std::vector<uint8_t> tbuf(size_t(tex_w) * size_t(tex_h) * 2);
                    glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_UNSIGNED_BYTE, tbuf.data());
                    int nz = 0; int mx = 0;
                    for (size_t i = 0; i < tbuf.size(); i += 2) {
                        int v = tbuf[i] | tbuf[i + 1];
                        if (v) { nz++; mx = std::max<int>(mx, tbuf[i]); }
                    }
                    printf("VALUE TEX0: width=%d non-zero texels=%d/%zu max=%d\n",
                           tex_w, nz, tbuf.size() / 2, mx);
                }
                for (int i = 0; i < atlas_view.level_count; ++i)
                    printf("  level %d: level_off=%d value_off=%d\n", i, atlas_view.level_off[i], atlas_view.value_off[atlas_chain][i]);
            }
            char path[128];
            std::snprintf(path, sizeof(path), "/tmp/opencode/gpu_chain%d_f%d.ppm",
                          atlas_chain, frame);
            glBindFramebuffer(GL_FRAMEBUFFER, atlas_view.fbo);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            std::vector<uint8_t> px(size_t(atlas_view.view_w) * atlas_view.view_h * 3);
            glReadPixels(0, 0, atlas_view.view_w, atlas_view.view_h, GL_RGB,
                         GL_UNSIGNED_BYTE, px.data());
            FILE* f = fopen(path, "wb");
            fprintf(f, "P6\n%d %d\n255\n", atlas_view.view_w, atlas_view.view_h);
            fwrite(px.data(), 1, px.size(), f);
            fclose(f);
            printf("DUMP: wrote %s (%dx%d)\n", path, atlas_view.view_w, atlas_view.view_h);
                {
                    FILE* dd = fopen("/tmp/opencode/inmem_depth.bin", "wb");
                    const auto& dc = atlas.depth_chain();
                    for (const auto& lv : dc.levels)
                        fwrite(lv.data.data(), 1, lv.data.size(), dd);
                    fclose(dd);
                    int tex_w = 0, tex_h = 0;
                    glBindTexture(GL_TEXTURE_2D, atlas_view.value_tex[2]);
                    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &tex_w);
                    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &tex_h);
                    std::vector<uint8_t> tbuf(size_t(tex_w) * size_t(tex_h) * 2);
                    glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_UNSIGNED_BYTE, tbuf.data());
                    FILE* td = fopen("/tmp/opencode/gpu_depth_tex.bin", "wb");
                    for (size_t i = 0; i < size_t(tex_w) * size_t(tex_h); ++i)
                        fwrite(&tbuf[i*2], 1, 1, td);
                    fclose(td);
                    printf("DUMP: gpu depth tex width=%d height=%d (%u bytes R)\n", tex_w, tex_h, unsigned(size_t(tex_w) * size_t(tex_h)));
                    glBindTexture(GL_TEXTURE_2D, atlas_view.value_tex[1]);
                    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &tex_w);
                    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &tex_h);
                    std::vector<uint8_t> tbuf2(size_t(tex_w) * size_t(tex_h) * 2);
                    glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_UNSIGNED_BYTE, tbuf2.data());
                    FILE* tf = fopen("/tmp/opencode/gpu_thick_tex.bin", "wb");
                    for (size_t i = 0; i < size_t(tex_w) * size_t(tex_h); ++i)
                        fwrite(&tbuf2[i*2], 1, 1, tf);
                    fclose(tf);
                    printf("DUMP: gpu thick tex width=%d height=%d (%u bytes R)\n", tex_w, tex_h, unsigned(size_t(tex_w) * size_t(tex_h)));
                }
                FILE* md = fopen("/tmp/opencode/inmem_meta.bin", "wb");
                const auto& uc = atlas.uv_chain();
                uint32_t lv_count = uint32_t(uc.levels.size());
                fwrite(&lv_count, 4, 1, md);
                for (const auto& lv : uc.levels) {
                    uint32_t n = uint32_t(lv.meta.size());
                    fwrite(&n, 4, 1, md);
                    fwrite(lv.meta.data(), 4, lv.meta.size(), md);
                }
                fclose(md);
                printf("LEVELOFF:");
                for (int L = 0; L < atlas_view.level_count && L < 32; ++L) printf(" %d", atlas_view.level_off[L]);
                printf("\n");
                for (int c = 0; c < 4; ++c) {
                    printf("VALUEOFF %d:", c);
                    for (int L = 0; L < atlas_view.level_count && L < 32; ++L) printf(" %d", atlas_view.value_off[c][L]);
                    printf("\n");
                }
                {
                    std::vector<uint32_t> gm(21277);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, atlas_view.meta_ssbo);
                    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                                       gm.size() * sizeof(uint32_t), gm.data());
                    FILE* gd = fopen("/tmp/opencode/gpu_meta.bin", "wb");
                    fwrite(gm.data(), 4, gm.size(), gd);
                    fclose(gd);
                }
                printf("DUMP: wrote inmem depth + meta\n");
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            gl::viewport(0, 0, window.framebuffer_width(), window.framebuffer_height());
        }

        // --- BVH wireframe overlay ---
        if (show_bvh && !atlas.bvh_nodes().empty()) {
            if (show_bvh_occ) {
                // The lighting quad overwrote the default FB depth, so restore
                // the scene depth from the G-buffer to get correct occlusion.
                glBindFramebuffer(GL_READ_FRAMEBUFFER, gbuf.fbo_handle());
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
                glBlitFramebuffer(0, 0, gbuf.width(), gbuf.height(),
                                  0, 0, window.framebuffer_width(), window.framebuffer_height(),
                                  GL_DEPTH_BUFFER_BIT, GL_NEAREST);
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
            }
            gl::enable(GL_DEPTH_TEST);
            bvh_draw.clear();
            const auto& nodes = atlas.bvh_nodes();
            for (size_t i = 0; i < nodes.size(); ++i) {
                const auto& n = nodes[i];
                glm::vec4 col;
                if (int(i) == pick.node) col = {1.0f, 0.1f, 0.1f, 1.0f};
                else if (std::find(pick.best_chain.begin(), pick.best_chain.end(),
                                   int(i)) != pick.best_chain.end())
                    col = {1.0f, 0.6f, 0.1f, 1.0f};
                else if (n.is_leaf) col = {0.35f, 0.85f, 1.0f, 1.0f};
                else col = {0.45f, 1.0f, 0.45f, 1.0f};
                bvh_draw.draw_box(n.aabb_min, n.aabb_max, col);
            }
            if (pick.node >= 0) {
                // Inflated bright box on top of the picked node so it always reads.
                const auto& n = nodes[size_t(pick.node)];
                glm::vec3 c = (n.aabb_min + n.aabb_max) * 0.5f;
                glm::vec3 half = (n.aabb_max - n.aabb_min) * 0.5f + glm::vec3(0.012f);
                bvh_draw.draw_box(c - half, c + half, {1.0f, 0.05f, 0.25f, 1.0f});
            }
            bvh_draw.render(vp);
            gl::disable(GL_DEPTH_TEST);
        }

        // --- ImGui debug window ---
        {
            ImGui::Begin("MDC Lighting - Debug");
            ImGui::Text("Frametime: %.3f ms", dt * 1000.0f);
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            ImGui::PlotLines("ms", frame_times.data(), int(frame_times.size()),
                             0, nullptr, 0.0f, 40.0f, ImVec2(0, 60));
            ImGui::Separator();

            int cur_model_prev = cur_model;
            if (ImGui::Combo("Model", &cur_model,
                             "Cornell Box\0Stanford Bunny\0Stanford Dragon\0")) {
                if (load_scene(kModels[cur_model], model, st)) {
                    cam.look_at(st.cam_target + glm::vec3(0.0f, 0.0f, st.cam_dist), st.cam_target);
                    base_density = atlas.config().texel_density;
                    pick = RayHit{};
                    std::printf("Switched to %s\n", kModels[cur_model].name);
                } else {
                    std::fprintf(stderr, "Failed to switch to %s\n", kModels[cur_model].name);
                    cur_model = cur_model_prev;
                }
            }
            ImGui::Text("Scene: %s", kModels[cur_model].name);
            ImGui::Separator();

            ImGui::Checkbox("Show MDC patches", &show_patches);
            if (show_patches) {
                ImGui::Text("Rendering %zu patches", patch_gpu.patch_count);
            }
            ImGui::Separator();

            ImGui::Checkbox("Show BVH wireframe", &show_bvh);
            ImGui::Checkbox("Occlude BVH by scene depth", &show_bvh_occ);
            ImGui::Separator();

            ImGui::Checkbox("Show primary-ray normals", &show_ray_pass);
            if (show_ray_pass) {
                ImGui::SliderFloat("Connectivity tol", &ray_connect_tol, 0.001f, 0.2f, "%.3f");
                ImGui::SliderFloat("Slab eps", &ray_eps, 0.0f, 0.1f, "%.3f");
            }
            if (ray_pass.ray_tex) {
                float rsz = float(ImGui::GetContentRegionAvail().x);
                if (rsz < 64) rsz = 256;
                ImGui::Image((ImTextureID)(intptr_t)ray_pass.ray_tex,
                             ImVec2(rsz, rsz * 0.5625f), ImVec2(0, 1), ImVec2(1, 0));
            }
            ImGui::Separator();

            if (pick.node >= 0) {
                const auto& n = atlas.bvh_nodes()[size_t(pick.node)];
                ImGui::Text("BVH pick:");
                if (pick.leaf) {
                    ImGui::Text("  node %d (leaf)  patch %d  depth %d",
                                pick.node, pick.patch, int(pick.best_chain.size()) - 1);
                } else {
                    int right = pick.node + int(n.right_offset);
                    ImGui::Text("  node %d (internal, deepest entered)  left %d  right %d",
                                pick.node, pick.node + 1, right);
                }
                ImGui::Text("  aabb min (%.2f, %.2f, %.2f)",
                            n.aabb_min.x, n.aabb_min.y, n.aabb_min.z);
                ImGui::Text("  aabb max (%.2f, %.2f, %.2f)",
                            n.aabb_max.x, n.aabb_max.y, n.aabb_max.z);
                if (pick.leaf) {
                    ImGui::Text("  t=%.3f", pick.t);
                    std::string chain;
                    for (size_t k = 0; k < pick.best_chain.size(); ++k)
                        chain += std::to_string(pick.best_chain[k]) +
                                 (k + 1 < pick.best_chain.size() ? " -> " : "");
                    ImGui::TextWrapped("  chain: %s", chain.c_str());
                }
            } else {
                ImGui::Text("Left-click the viewport to ray-cast the BVH");
            }
            ImGui::Separator();

            ImGui::Text("Coverage atlas: %dx%d @ %.0f texels/unit",
                        atlas.atlas_width(), atlas.atlas_height(), atlas.final_density());
            ImGui::Combo("Atlas texture", &atlas_chain, "UV\0Thickness\0Depth\0Normal\0");
            ImGui::SliderFloat("Resolution scale", &density_scale, 0.25f, 4.0f, "%.2fx");
            if (ImGui::Button("Recompute atlas")) {
                gfx::CoverageAtlasConfig cfg = atlas.config();
                cfg.texel_density = base_density * density_scale;
                atlas.set_config(cfg);
                if (atlas.build(model)) {
                    upload_patch_gpu(atlas, patch_gpu);
                    upload_atlas_view(atlas, atlas_view);
                    upload_ray_pass_buffers(atlas, ray_pass);
                    ray_depth_lo = atlas.depth_chain().qmin[0];
                    ray_depth_hi = atlas.depth_chain().qmax[0];
                    ray_thick_max = atlas.thickness_chain().qmax[0];
                    ray_depth_origin = glm::vec3(1e30f);
                    for (const auto& v : atlas.positions())
                        ray_depth_origin = glm::min(ray_depth_origin, v);
                    atlas.write_files(kModels[cur_model].cache_dir);
                    // base_density stays the per-model auto density so the
                    // resolution slider keeps scaling relative to it instead of
                    // compounding on every recompute.
                    pick = RayHit{};
                    std::printf("Rebuilt atlas: %dx%d @ %.0f texels/unit, %zu patches\n",
                                atlas.atlas_width(), atlas.atlas_height(),
                                atlas.final_density(), atlas.patches().size());
                }
            }
            if (atlas_view.view_tex) {
                float asz = float(ImGui::GetContentRegionAvail().x);
                if (asz < 64) asz = 256;
                float ah = asz * float(atlas_view.view_h) / float(atlas_view.view_w);
                ImGui::Image((ImTextureID)(intptr_t)atlas_view.view_tex,
                             ImVec2(asz, ah), ImVec2(0, 1), ImVec2(1, 0));
            }
            ImGui::Separator();

            ImGui::Combo("Debug view", &debug_view,
                         "Composite\0Albedo\0Normal+RM\0Emissive+AO\0Depth\0");

            float sz = float(ImGui::GetContentRegionAvail().x);
            if (sz < 64) sz = 256;
            ImGui::Image((ImTextureID)(intptr_t)gbuf.albedo_handle(), ImVec2(sz, sz * 0.5f), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::Image((ImTextureID)(intptr_t)gbuf.normal_rm_handle(), ImVec2(sz, sz * 0.5f), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::Image((ImTextureID)(intptr_t)gbuf.emissive_ao_handle(), ImVec2(sz, sz * 0.5f), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::Image((ImTextureID)(intptr_t)gbuf.depth_handle(), ImVec2(sz, sz * 0.5f), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();
        }
        gui.render();

        window.swap_buffers();
    }

    return EXIT_SUCCESS;
}

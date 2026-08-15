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
#include <gl/query.hpp>
#include <gfx/gfx.hpp>
#include <gfx/gbuffer.hpp>
#include <gllib/log.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <gfx/imgui_overlay.hpp>

#include <algorithm>
#include <chrono>
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
uniform int u_sel_patch;
layout(std430, binding = 0) readonly buffer ColorBuf {
    vec4 patch_colors[];
};
out vec4 frag_color;
void main() {
    vec3 n = normalize(v_normal);
    vec3 light_dir = normalize(vec3(0.0, 1.0, 0.6));
    float light = 0.55 + 0.45 * abs(dot(n, light_dir));
    vec3 col = patch_colors[v_patch_id].rgb * light;
    if (v_patch_id == u_sel_patch)
        col = mix(col, vec3(1.0, 1.0, 1.0), 0.7);
    frag_color = vec4(col, 1.0);
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
uniform int u_sel_patch;         // patch picked in the 3D view (-1 = none)
uniform int u_sel_on;            // 1 = highlight the picked patch's atlas rect
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

vec4 highlight_patch(vec4 col, int x, int y, int w, int h) {
    if (u_sel_on == 0 || u_sel_patch < 0) return col;
    bool border = (x < 3 || y < 3 || x >= w - 3 || y >= h - 3);
    if (border) return vec4(mix(col.rgb, vec3(1.0, 0.15, 0.12), 0.95), 1.0);
    return vec4(mix(col.rgb, vec3(1.0, 0.30, 0.25), 0.50), 1.0);
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
    if (u_debug_meta == 1) {
        vec4 c = vec4(vec3(float(pid) / float(u_patch_count)), 1.0);
        if (pid == u_sel_patch) c = highlight_patch(c, x, y, int(r.z), int(r.w));
        frag_color = c; return;
    }
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

    vec4 col;
    if (u_chain == 0)
        col = vec4(q, 0.0, 1.0);          // UV: u,v
    else if (u_chain == 3) {
        // Normal: decode the octahedral coords back to a viewable normal.
        vec2 oct = vec2(decode_value(q.x, -1.0, 1.0, 8),
                        decode_value(q.y, -1.0, 1.0, 8));
        col = vec4(octahedral_decode(oct) * 0.5 + 0.5, 1.0);
    } else
        col = vec4(vec3(q.x), 1.0);       // thickness / depth: channel 0

    if (pid == u_sel_patch)
        col = highlight_patch(col, x, y, int(r.z), int(r.w));
    frag_color = col;
}
)";

// ---------------------------------------------------------------------------
// Primary-ray pass — traces a ray through the BVH, then refines the hit with
// the patch's atlas mip chain (depth + thickness), writing surface normals.
// ---------------------------------------------------------------------------

static const char* ray_common_glsl = R"(
#ifdef PERF_ENABLED
#extension GL_ARB_shader_clock : require
#endif
// 8x8 = 64 threads/group instead of 16x16 = 256: this kernel is register- and
// divergence-heavy (big local arrays + per-ray quadtree walks), so a smaller
// group gives the scheduler finer granularity to interleave work while one
// group stalls on memory, and a single divergent group can't occupy the SM.
layout(local_size_x = 8, local_size_y = 8) in;

const int LEAF_TILE = 1;
// Sizes of the two per-invocation local arrays (BVH traversal stack and the
// quadtree descent cache). Both live in registers/scratch, so oversized arrays
// are the dominant register-pressure / spill cost. Verified with the max-stack
// and max-depth counters (perf[12]/perf[13]) that these bounds are never hit:
// the SAH tree over ~16 patches is ~5 deep, and the descent stops early on
// depth-flat nodes (FLAT_STOP_CODES), so the worst-case atlas depth is well
// below the old MAX_MIP = 12.
const int MAX_STACK = 16;
const int MAX_MIP = 7;
// A node whose 16-bit depth span (dmax-dmin) is within this many codes is
// depth-flat: its aggregate dmin/dmax/thick are within the marching band slack
// of every texel it contains, so the descent can stop there. Flat walls and
// silhouettes resolve in one fetch; curved/creased regions keep descending.
const int FLAT_STOP_CODES = 4;
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

// Per-texel coverage mask packed 4 bytes per uint. u_cover_texel() resolves
// the exact texel's coverage in one read so the descent may stop at a coarse
// depth-converged node without losing the empty-texel distinction.
layout(std430, binding = 5) readonly buffer CoverBuf { uint cover[]; };
uniform int u_atlas_w;

// Merged meta+values buffer texture: one RGBA32UI texel per quadtree node.
// Meta and the depth/thick/normal values share the quadtree topology, so one
// index (u_value_off_dt[L] + idx) fetches the traversal metadata AND the
// values in a single texelFetch (replacing the old meta SSBO read + RGBA16UI
// value read):
//   R = meta32  (0x80000000 covered, 0x40000000 has children, low30 = child)
//   G = (dmin16 << 16) | dmax16
//   B = (thick8 << 24) | (nrm_x8 << 8) | nrm_y8
uniform usamplerBuffer u_dtm_buf;
uniform int  u_level_off[32];      // meta SSBO offsets per level (shared)
uniform int  u_value_off_dt[32];   // value offsets per level (shared topology)
uniform int  u_patch_count;
uniform mat4 u_vp_inv;           // debug kernel only; lean uses the ray basis
uniform vec2 u_res;
uniform vec3 u_ro;
uniform vec2 u_ndc_scale;        // ndc = u_ndc_scale*p + u_ndc_bias  (p = pixel)
uniform vec2 u_ndc_bias;
uniform vec3 u_rd0;              // rd = normalize(u_rd0 + ndc.x*u_rd_dx + ndc.y*u_rd_dy)
uniform vec3 u_rd_dx;            // exact per-pixel ray basis precomputed on CPU
uniform vec3 u_rd_dy;
uniform vec3 u_root_amin;        // BVH root AABB, avoids the bvh[0] SSBO read
uniform vec3 u_root_amax;
uniform vec3 u_depth_origin;       // g_depth_origin (model AABB min)
uniform float u_depth_lo;          // depth chain qmin[0]
uniform float u_depth_hi;          // depth chain qmax[0]
uniform float u_thick_max;         // thickness chain qmax[0]
uniform float u_eps;               // slab test epsilon (debug mode 3 only)
uniform int  u_dbg;                // debug only: 0=normals, 1=ray dir, ...

layout(binding = 0, rgba8) uniform image2D u_out;

// Perf counters are compiled out of the production kernel (lean) unless
// PERF_ENABLED is defined; the debug kernel keeps them behind the same macro
// and gates them at runtime with u_perf instead of carrying a uniform branch
// in the hot path.
#ifdef PERF_ENABLED
layout(std430, binding = 4) buffer PerfBuf { uint perf[]; };
// perf[]: 0=leaf visits, 1=texel queries, 2=march iters, 3=bisect iters,
//         4=guard-limited rays, 5=rays traced
//        6=cycles setup, 7=cycles traversal, 8=cycles leaf setup,
//        9=cycles march loop, 10=cycles bisection, 11=cycles finalize
//        12=max BVH stack depth (sp after push), 13=max quadtree level reached
uniform int u_perf;    // runtime gate for the debug kernel's counters
#endif

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

bool ray_aabb(vec3 ro, vec3 rd, vec3 rd_inv, vec3 amin, vec3 amax,
              out float t_in, out float t_out) {
    t_in = -1.0e30;
    t_out = 1.0e30;
    for (int c = 0; c < 3; ++c) {
        float o = ro[c], d = rd[c], inv = rd_inv[c], lo = amin[c], hi = amax[c];
        if (abs(d) < 1.0e-9) {
            if (o < lo || o > hi) return false;
        } else {
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

// Per-invocation cache of the last quadtree descent path. Consecutive queries
// along a ray march land on the same node or a descendant / sibling, so we
// avoid re-walking the tree from the root on every sample: start from the
// deepest cached ancestor that contains the queried texel (a scan over the
// already-cached chain, no memory traffic) and descend only the levels that
// actually change.
struct QCache {
    int pid;             // patch the cached chain belongs to (-1 = invalid)
    int depth;           // number of valid chain entries
    int sz;              // root lattice size N; node size at level L is N >> L
    int lev[MAX_MIP];    // level of each chain node
    int id[MAX_MIP];     // node index within that level
    int x0[MAX_MIP];
    int y0[MAX_MIP];
};

// Descend the shared quadtree to the finest node covering atlas texel (tx,ty)
// of patch pid. Returns true if that node is covered (meta bit 0x80000000).
// Coverage of the atlas texel (tx,ty) from the byte-packed mask.
bool u_cover_texel(ivec2 txy) {
    uint idx = uint(txy.y) * uint(u_atlas_w) + uint(txy.x);
    return ((cover[idx >> 2u] >> ((idx & 3u) << 3u)) & 0xFFu) != 0u;
}

bool query_node(inout QCache qc, ivec2 txy, int pid,
                out int L, out int idx, out int x0, out int y0, out int s,
                out float dmin, out float dmax, out float thick) {
    uvec4 r = rects[pid];
    int x = txy.x - int(r.x);
    int y = txy.y - int(r.y);
    if (x < 0 || y < 0 || x >= int(r.z) || y >= int(r.w)) return false;
    int maxd = int(max(r.z, r.w));
    int N = 1;
    while (N < maxd) N <<= 1;
    qc.sz = N;

    // Start from the deepest cached ancestor that contains the texel. The
    // chain is walked top-down here; each cached entry is just the node rect,
    // so containment tests never touch the SSBOs. The cached node size is
    // derived from the level (s = N >> lev), so it is not stored.
    int start_k = -1;
    if (qc.pid == pid) {
        for (int k = qc.depth - 1; k >= 0; --k) {
            int cx0 = qc.x0[k], cy0 = qc.y0[k], cs = qc.sz >> qc.lev[k];
            if (x >= cx0 && y >= cy0 && x < cx0 + cs && y < cy0 + cs) {
                start_k = k;
                break;
            }
        }
    }
    if (start_k >= 0) {
        L = qc.lev[start_k];
        idx = qc.id[start_k];
        x0 = qc.x0[start_k];
        y0 = qc.y0[start_k];
        s = qc.sz >> qc.lev[start_k];
    } else {
        L = 0; idx = pid; x0 = 0; y0 = 0; s = N;
        qc.depth = 0;
    }
    int qc_top = start_k + 1;

    // A single texelFetch per node returns both the traversal metadata (R)
    // and the values (G/B), so the old path's final-node meta re-read and the
    // separate value fetch collapse into the same load. The descent also stops
    // as soon as the node's depth span (dmax-dmin) is within the marching band
    // slack: depth-flat regions (walls, flat silhouettes) resolve at a coarse
    // node in one fetch, while curved/creased regions keep descending.
    uvec4 q;
    while (true) {
        q = texelFetch(u_dtm_buf, u_value_off_dt[L] + idx);
        uint m = q.r;
        if ((m & 0x40000000u) == 0u || s <= LEAF_TILE) break;
        // Depth-flat early stop: the node's aggregate depth band is within the
        // marching slack of every texel it contains, so its dmin/dmax/thick
        // (and, for a flat region, its aggregate normal) are usable directly.
        uint dmin16 = (q.g >> 16u) & 0xFFFFu;
        uint dmax16 = q.g & 0xFFFFu;
        if (dmax16 - dmin16 <= uint(FLAT_STOP_CODES)) break;
        if (qc_top < MAX_MIP) {
            qc.lev[qc_top] = L; qc.id[qc_top] = idx;
            qc.x0[qc_top] = x0; qc.y0[qc_top] = y0;
            qc_top++;
        }
        int hs = s >> 1;
        int cx = (x >= x0 + hs) ? 1 : 0;
        int cy = (y >= y0 + hs) ? 1 : 0;
        idx = int(m & 0x3FFFFFFFu) + cy * 2 + cx;
        x0 += cx * hs;
        y0 += cy * hs;
        s = hs;
        L += 1;
    }
#ifdef PERF_ENABLED
    atomicMax(perf[13], uint(L));   // deepest level the descent reached
#endif
    if (qc_top < MAX_MIP) {
        qc.lev[qc_top] = L; qc.id[qc_top] = idx;
        qc.x0[qc_top] = x0; qc.y0[qc_top] = y0;
        qc_top++;
    }
    qc.depth = qc_top;
    qc.pid = pid;

    // The aggregate meta bit only says "some texel in this node is covered".
    // A coarse early-stopped node needs the per-texel coverage mask; a leaf
    // (s == LEAF_TILE) is exactly one texel, so its bit is already exact.
    if ((q.r & 0x80000000u) == 0u) return false;
    if (s > LEAF_TILE && !u_cover_texel(txy)) return false;
#ifdef PERF_ENABLED
    atomicAdd(perf[1], 1u);
#endif

    // Integer buffer texture: decode_value expects the same normalized [0,1]
    // inputs the old RG16/RG8 textures produced, so scale by 2^bits-1 first.
    dmin = decode_value(float((q.g >> 16u) & 0xFFFFu) / 65535.0, u_depth_lo, u_depth_hi, 16);
    dmax = decode_value(float(q.g & 0xFFFFu) / 65535.0, u_depth_lo, u_depth_hi, 16);
    uint th8 = (q.b >> 24u) & 0xFFu;
    thick = decode_value(float(th8) / 255.0, 0.0, u_thick_max, 8);
    return true;
}

// Fast re-query for points near a node that was just descended (used by the
// bisection refinement). Three paths, in order:
//   1. the texel sits inside the "hot" node of a neighbouring sample — resume
//      the descent from it (zero memory when the hot node is already a leaf);
//   2. otherwise walk the cache chain read-only, resuming from the deepest
//      frozen ancestor that contains the texel (the cache is left untouched so
//      every midpoint query starts from the same warm chain);
//   3. otherwise fall back to the full query_node.
bool query_hot(inout QCache qc, ivec2 txy, int pid,
               int hL, int hidx, int hx0, int hy0, int hs, bool hv,
               out int L, out int idx, out int x0, out int y0, out int s,
               out float dmin, out float dmax, out float thick) {
    int qL = 0, qidx = 0, qx0 = 0, qy0 = 0, qs = 0;
    bool have = false;
    if (hv && txy.x >= hx0 && txy.y >= hy0 && txy.x < hx0 + hs && txy.y < hy0 + hs) {
        qL = hL; qidx = hidx; qx0 = hx0; qy0 = hy0; qs = hs;
        have = true;
    } else if (qc.pid == pid) {
        for (int k = qc.depth - 1; k >= 0; --k) {
            int cx0 = qc.x0[k], cy0 = qc.y0[k], cs = qc.sz >> qc.lev[k];
            if (txy.x >= cx0 && txy.y >= cy0 && txy.x < cx0 + cs && txy.y < cy0 + cs) {
                qL = qc.lev[k]; qidx = qc.id[k]; qx0 = cx0; qy0 = cy0; qs = cs;
                have = true;
                break;
            }
        }
    }
    if (have) {
        L = qL; idx = qidx; x0 = qx0; y0 = qy0; s = qs;
        uvec4 q;
        if (s > LEAF_TILE) {
            while (true) {
                q = texelFetch(u_dtm_buf, u_value_off_dt[L] + idx);
                uint m = q.r;
                if ((m & 0x40000000u) == 0u || s <= LEAF_TILE) break;
                uint dmin16 = (q.g >> 16u) & 0xFFFFu;
                uint dmax16 = q.g & 0xFFFFu;
                if (dmax16 - dmin16 <= uint(FLAT_STOP_CODES)) break;
                int hss = s >> 1;
                int cx = (txy.x >= x0 + hss) ? 1 : 0;
                int cy = (txy.y >= y0 + hss) ? 1 : 0;
                idx = int(m & 0x3FFFFFFFu) + cy * 2 + cx;
                x0 += cx * hss; y0 += cy * hss; s = hss; L += 1;
            }
#ifdef PERF_ENABLED
            atomicMax(perf[13], uint(L));
#endif
            if ((q.r & 0x80000000u) == 0u) return false;
            if (s > LEAF_TILE && !u_cover_texel(txy)) return false;
        } else {
            q = texelFetch(u_dtm_buf, u_value_off_dt[L] + idx);
        }
#ifdef PERF_ENABLED
        atomicAdd(perf[1], 1u);
#endif
        dmin = decode_value(float((q.g >> 16u) & 0xFFFFu) / 65535.0, u_depth_lo, u_depth_hi, 16);
        dmax = decode_value(float(q.g & 0xFFFFu) / 65535.0, u_depth_lo, u_depth_hi, 16);
        uint th8 = (q.b >> 24u) & 0xFFu;
        thick = decode_value(float(th8) / 255.0, 0.0, u_thick_max, 8);
        return true;
    }
    return query_node(qc, txy, pid, L, idx, x0, y0, s, dmin, dmax, thick);
}

// Refine a bracketed march interval [ta, tb] (ta in front of the surface) to the
// point where the ray reaches the RECONSTRUCTED surface. The conservative depth
// band stored per texel is [dmin, dmax]: dmin is the footprint MINIMUM so the
// band never lets a ray tunnel through the near half of a slanted texel, but
// that front sits a full thickness in front of the true surface at the texel
// centre. Resolving the band front would therefore reconstruct the footprint's
// near edge instead of the surface, stepping out by up to a texel thickness —
// and at patch seams, where the slant and texel size change between patches,
// those steps pile up into a coherent ridge. The band MIDPOINT crosses the
// surface through the texel centres, which stays continuous across patch
// borders. This mirrors the voxelizer's --mid re-anchoring (example 33).
float refine_band_mid(inout QCache qc, int pid, vec3 ro, vec3 rd, PatchInfo pi, uvec4 r,
                      float ta, float tb) {
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
            if (query_node(qc, txy_m, pid, Lm, idxm, x0m, y0m, sm, dm_m, dmax_m, thk_m)) {
                float Dm = dot(Pm - u_depth_origin, pi.basis_u);
                if (Dm <= 0.5 * (dm_m + min(dmax_m, dm_m + thk_m))) tb = tm; else ta = tm;
                continue;
            }
        }
        ta = tm;   // fell off the patch or hit an uncovered node: keep the lower bound
    }
    return (ta + tb) * 0.5;
}
)";

// Production kernel: no debug modes, no runtime perf uniform, far BVH children
// are never pushed once they are behind the best hit so far.
static const char* ray_lean_main = R"(
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= int(u_res.x) || p.y >= int(u_res.y)) {
        imageStore(u_out, p, vec4(0.0));
        return;
    }
#ifdef PERF_ENABLED
    atomicAdd(perf[5], 1u);
    uint t_setup0 = clock2x32ARB().x;
#endif

    vec2 ndc = u_ndc_scale * vec2(p) + u_ndc_bias;
    vec3 ro = u_ro;
    vec3 rd = normalize(u_rd0 + ndc.x * u_rd_dx + ndc.y * u_rd_dy);
    vec3 rd_inv = 1.0 / rd;

    float best_t = 1.0e30;
    int best_pid = -1;
    int best_L = 0, best_idx = 0, best_x0 = 0, best_y0 = 0, best_s = 0;

    if (u_patch_count == 0 || int(bvh.length()) == 0) {
        imageStore(u_out, p, vec4(0.0));
        return;
    }

    QCache qc;
    qc.pid = -1;
    qc.depth = 0;

    int   st_node[MAX_STACK];
    float st_tin[MAX_STACK];
    float st_tout[MAX_STACK];
    int sp = 0;

    float rt0, rt1;
    if (ray_aabb(ro, rd, rd_inv, u_root_amin, u_root_amax, rt0, rt1)) {
        st_node[sp] = 0; st_tin[sp] = rt0; st_tout[sp] = rt1; sp++;
        // Nothing beyond the root slab exit can contain a hit, so seed best_t
        // with it instead of +inf. Miss rays now get the far-child prune
        // (far_t <= best_t) for their whole traversal instead of only after
        // they find a surface, which is the cheap-but-common case here.
        best_t = rt1;
    }
#ifdef PERF_ENABLED
    atomicMax(perf[12], uint(sp));
    atomicAdd(perf[6], clock2x32ARB().x - t_setup0);
    uint t_trav0 = clock2x32ARB().x;
#endif

    while (sp > 0) {
#ifdef PERF_ENABLED
        t_trav0 = clock2x32ARB().x;
#endif
        sp--;
        int node = st_node[sp];
        float tin = st_tin[sp];
        float tout = st_tout[sp];
        for (;;) {
            if (tin > best_t) break;

            BVHGPU b = bvh[node];
            if (b.is_leaf != 0u) {
#ifdef PERF_ENABLED
                atomicAdd(perf[0], 1u);
                uint t_leaf0 = clock2x32ARB().x;
                atomicAdd(perf[7], t_leaf0 - t_trav0);
#endif
            int pid = int(b.val);
            PatchInfo pi = patches[pid];
            uvec4 r = rects[pid];

            float dD = dot(rd, pi.basis_u);
            if (abs(dD) <= 1.0e-6) break;

            // Single-sided materials cull their back face.
            if (pi.double_sided == 0u && dD > 0.0) break;

            float D0 = dot(ro - u_depth_origin, pi.basis_u);
            float qstep = (u_depth_hi - u_depth_lo) / 65534.0;   // 16-bit depth
            float htol = max(2.0 * qstep, 1.0e-4);   // depth quantisation slack

            // ---- Texel-stepped heightfield march (robust at grazing angles) ----
            float tex_s = min(pi.proj_size.x / float(max(r.z, 1u)),
                              pi.proj_size.y / float(max(r.w, 1u)));
            vec2 rd_uv = vec2(dot(rd, pi.basis_v), dot(rd, pi.basis_w));
            float uv_speed = max(length(rd_uv), 1.0e-4);
            // Loop-invariant fine step (hoisted; the original recomputed this
            // clamp() on every iteration with identical operands).
            float base_step = clamp(min(0.5 * tex_s, tex_s / uv_speed),
                                    1.0e-6, tout - tin);

            float t = tin;
            bool have_prev = false;
            float t_prev = tin;
            bool prev_before = false;   // previous sample still in front of the band
            int pL = 0, pidx = 0, px0 = 0, py0 = 0, ps = 0;
            bool pvalid = false;   // previous covered sample's query node (bisect reuse)
            int guard = 0;
#ifdef PERF_ENABLED
            uint t_march0 = clock2x32ARB().x;
            atomicAdd(perf[8], t_march0 - t_leaf0);
#endif
            while (guard < 2048 && t <= tout) {
                guard++;
#ifdef PERF_ENABLED
                atomicAdd(perf[2], 1u);
#endif
                vec3 P = ro + rd * t;
                vec2 proj = vec2(dot(P, pi.basis_v), dot(P, pi.basis_w));
                vec2 nrm = (proj - pi.proj_min) / pi.proj_size;
                bool inside = nrm.x >= 0.0 && nrm.y >= 0.0 && nrm.x <= 1.0 && nrm.y <= 1.0;
                float dm = 0.0, dmax = 0.0, thk = 0.0;
                int L, idx, x0, y0, s;
                bool covered = false;
                if (inside) {
                    ivec2 txy = ivec2(int(r.x) + int(nrm.x * float(r.z)),
                                      int(r.y) + int(nrm.y * float(r.w)));
                    covered = query_node(qc, txy, pid, L, idx, x0, y0, s, dm, dmax, thk);
                    if (!covered) inside = false;
                }
                float D = dot(P - u_depth_origin, pi.basis_u);
                float step = base_step;
                if (inside && covered && s > LEAF_TILE) {
                    float stride = clamp(min(0.5 * tex_s * float(s),
                                             tex_s * float(s) / uv_speed),
                                         1.0e-6, tout - tin);
                    if (D - abs(dD) * stride > dmax + htol) step = stride;
                }
                float band_hi = inside ? min(dmax, dm + thk) : 0.0;
                if (inside && D >= dm - htol && D <= band_hi + htol) {
                    // Snap the hit to the band midpoint (texel-centre surface)
                    // instead of leaving it at the first sample inside the
                    // conservative band; see refine_band_mid.
                    float th = t;
                    if (have_prev && t > t_prev) th = refine_band_mid(qc, pid, ro, rd, pi, r, t_prev, t);
                    best_t = th;
                    best_pid = pid;
                    best_L = L; best_idx = idx;
                    best_x0 = x0; best_y0 = y0; best_s = s;
                    break;
                }
                bool cur_before = !inside || (D > band_hi + htol);

                if (have_prev && (prev_before != cur_before)) {
#ifdef PERF_ENABLED
                    uint t_bisect0 = clock2x32ARB().x;
#endif
                    float ta = t_prev;
                    float tb = t;
                    int rL = 0, ridx = 0, rx0 = 0, ry0 = 0, rs = 0;
                    bool rvalid = false;   // last successful midpoint query node
                    for (int it = 0; it < 8; ++it) {
#ifdef PERF_ENABLED
                        atomicAdd(perf[3], 1u);
#endif
                        float tm = (ta + tb) * 0.5;
                        vec3 Pm = ro + rd * tm;
                        vec2 proj_m = vec2(dot(Pm, pi.basis_v), dot(Pm, pi.basis_w));
                        vec2 nrm_m = (proj_m - pi.proj_min) / pi.proj_size;
                        if (nrm_m.x >= 0.0 && nrm_m.y >= 0.0 && nrm_m.x <= 1.0 && nrm_m.y <= 1.0) {
                            ivec2 txy_m = ivec2(int(r.x) + int(nrm_m.x * float(r.z)),
                                                int(r.y) + int(nrm_m.y * float(r.w)));
                            int Lm, idxm, x0m, y0m, sm;
                            float dm_m, dmax_m, thk_m;
                            // Reuse the already-descended node of a bracket
                            // endpoint (current or previous sample) instead of
                            // re-walking the quadtree for every midpoint.
                            int hL = L, hidx = idx, hx0 = x0, hy0 = y0, hs = s;
                            bool hv = covered &&
                                      txy_m.x >= hx0 && txy_m.y >= hy0 &&
                                      txy_m.x < hx0 + hs && txy_m.y < hy0 + hs;
                            if (!hv && pvalid &&
                                txy_m.x >= px0 && txy_m.y >= py0 &&
                                txy_m.x < px0 + ps && txy_m.y < py0 + ps) {
                                hL = pL; hidx = pidx; hx0 = px0; hy0 = py0; hs = ps; hv = true;
                            }
                            if (query_hot(qc, txy_m, pid, hL, hidx, hx0, hy0, hs, hv,
                                          Lm, idxm, x0m, y0m, sm, dm_m, dmax_m, thk_m)) {
                                rL = Lm; ridx = idxm; rx0 = x0m; ry0 = y0m; rs = sm; rvalid = true;
                                float Dm = dot(Pm - u_depth_origin, pi.basis_u);
                                // Converge on the band midpoint (texel-centre
                                // surface) rather than the band front (footprint
                                // near edge); see refine_band_mid.
                                if (Dm <= 0.5f * (dm_m + min(dmax_m, dm_m + thk_m))) tb = tm; else ta = tm;
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
                    if (nrm_f.x >= 0.0 && nrm_f.y >= 0.0 && nrm_f.x <= 1.0 && nrm_f.y <= 1.0) {
                        ivec2 txy_f = ivec2(int(r.x) + int(nrm_f.x * float(r.z)),
                                            int(r.y) + int(nrm_f.y * float(r.w)));
                        int Lf, idxf, x0f, y0f, sf;
                        float dmf, dmxf, thkf;
                        // Final query: reuse the last bisect midpoint's node (or
                        // a bracket endpoint) before falling back to a descent.
                        int hfL = rL, hfidx = ridx, hfx0 = rx0, hfy0 = ry0, hfs = rs;
                        bool hfv = rvalid &&
                                   txy_f.x >= hfx0 && txy_f.y >= hfy0 &&
                                   txy_f.x < hfx0 + hfs && txy_f.y < hfy0 + hfs;
                        if (!hfv) {
                            hfL = L; hfidx = idx; hfx0 = x0; hfy0 = y0; hfs = s;
                            hfv = covered &&
                                  txy_f.x >= hfx0 && txy_f.y >= hfy0 &&
                                  txy_f.x < hfx0 + hfs && txy_f.y < hfy0 + hfs;
                        }
                        if (!hfv && pvalid &&
                            txy_f.x >= px0 && txy_f.y >= py0 &&
                            txy_f.x < px0 + ps && txy_f.y < py0 + ps) {
                            hfL = pL; hfidx = pidx; hfx0 = px0; hfy0 = py0; hfs = ps; hfv = true;
                        }
                        if (query_hot(qc, txy_f, pid, hfL, hfidx, hfx0, hfy0, hfs, hfv,
                                      Lf, idxf, x0f, y0f, sf, dmf, dmxf, thkf)) {
                            float D_f = dot(ro + rd * th - u_depth_origin, pi.basis_u);
                            float band_hi_f = min(dmxf, dmf + thkf);
                            float slack = max(htol, 2.0 * abs(dD) * step);
                            if (D_f >= dmf - htol - slack && D_f <= band_hi_f + htol + slack &&
                                th < best_t) {
                                best_t = th;
                                best_pid = pid;
                                best_L = Lf; best_idx = idxf;
                                best_x0 = x0f; best_y0 = y0f; best_s = sf;
                            }
                        }
                    }
#ifdef PERF_ENABLED
                atomicAdd(perf[10], clock2x32ARB().x - t_bisect0);
#endif
                }

                t_prev = t;
                prev_before = cur_before;
                have_prev = true;
                pvalid = covered;
                if (covered) {
                    pL = L; pidx = idx; px0 = x0; py0 = y0; ps = s;
                }
                if (t > best_t) break;
                if (t >= tout) break;      // the slab exit was just sampled
                t += step;
                if (t > tout) t = tout;    // land the final sample on the exit
            }
#ifdef PERF_ENABLED
            atomicAdd(perf[9], clock2x32ARB().x - t_march0);
            if (guard >= 2048) atomicAdd(perf[4], 1u);
#endif
            break;
        } else {
            int left = node + 1;
            int right = node + int(b.val);
            float t0, t1, t2, t3;
            bool hit_l = ray_aabb(ro, rd, rd_inv, bvh[left].amin,  bvh[left].amax,  t0, t1);
            bool hit_r = ray_aabb(ro, rd, rd_inv, bvh[right].amin, bvh[right].amax, t2, t3);
            // Descend into the near child inline and push only the far child,
            // so each level costs one push/pop round-trip instead of two.
            int near_n = -1, far_n = -1;
            float near_t = 0.0, near_x = 0.0, far_t = 0.0, far_x = 0.0;
            bool has_near = false, has_far = false;
            if (hit_l && hit_r) {
                if (t0 <= t2) {
                    near_n = left;  near_t = t0; near_x = t1;
                    far_n  = right; far_t  = t2; far_x  = t3;
                } else {
                    near_n = right; near_t = t2; near_x = t3;
                    far_n  = left;  far_t  = t0; far_x  = t1;
                }
                has_near = true;
                has_far = true;
            } else if (hit_l) {
                near_n = left; near_t = t0; near_x = t1; has_near = true;
            } else if (hit_r) {
                near_n = right; near_t = t2; near_x = t3; has_near = true;
            }
            // A child whose entry distance is already behind the best hit can
            // never contain a closer surface, so skip it entirely.
            if (has_far && far_t <= best_t) {
                st_node[sp] = far_n; st_tin[sp] = far_t; st_tout[sp] = far_x; sp++;
#ifdef PERF_ENABLED
                atomicMax(perf[12], uint(sp));
#endif
            }
            if (!has_near || near_t > best_t) break;
            node = near_n; tin = near_t; tout = near_x;
        }
        }
    }
#ifdef PERF_ENABLED
    atomicAdd(perf[7], clock2x32ARB().x - t_trav0);
    uint t_final0 = clock2x32ARB().x;
#endif

    if (best_pid < 0) {
        imageStore(u_out, p, vec4(0.0));
        return;
    }

    // ---- Surface normal from the baked octahedral normal chain ----
    // The normal was rasterised once and aggregated per quadtree node, so
    // fetching it at the resolved node is exact and view-independent.
    PatchInfo pi = patches[best_pid];
    int vo = u_value_off_dt[best_L] + best_idx;
    uvec4 qn = texelFetch(u_dtm_buf, vo);
    uint nx8 = (qn.b >> 8u) & 0xFFu;
    uint ny8 = qn.b & 0xFFu;
    vec2 oct = vec2(decode_value(float(nx8) / 255.0, -1.0, 1.0, 8),
                    decode_value(float(ny8) / 255.0, -1.0, 1.0, 8));
    vec3 n = octahedral_decode(oct);
    if (dot(n, rd) > 0.0) n = -n;

    imageStore(u_out, p, vec4(n * 0.5 + 0.5, 1.0));
#ifdef PERF_ENABLED
    atomicAdd(perf[11], clock2x32ARB().x - t_final0);
#endif
}
)";

// Full diagnostic kernel: keeps every u_dbg mode (0..12) and the runtime
// perf gate, plus the original un-culled BVH traversal so the debug visual
// shows the whole visited set.
static const char* ray_debug_main = R"(
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= int(u_res.x) || p.y >= int(u_res.y)) {
        imageStore(u_out, p, vec4(0.0));
        return;
    }
#ifdef PERF_ENABLED
    if (u_perf != 0) atomicAdd(perf[5], 1u);
#endif

    QCache qc;
    qc.pid = -1;
    qc.depth = 0;

    vec2 ndc = vec2(2.0 * (float(p.x) + 0.5) / u_res.x - 1.0,
                    2.0 * (float(p.y) + 0.5) / u_res.y - 1.0);
    vec4 cw = u_vp_inv * vec4(ndc, 1.0, 1.0);
    vec3 wpos = cw.xyz / cw.w;
    vec3 ro = u_ro;
    vec3 rd = normalize(wpos - ro);
    vec3 rd_inv = 1.0 / rd;

    if (u_dbg == 1) {
        imageStore(u_out, p, vec4(rd * 0.5 + 0.5, 1.0));
        return;
    }
    if (u_dbg == 2) {
        float t0, t1;
        bool hit = (bvh.length() > 0) &&
                   ray_aabb(ro, rd, rd_inv, bvh[0].amin, bvh[0].amax, t0, t1);
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
            if (!ray_aabb(ro, rd, rd_inv, bvh[i].amin, bvh[i].amax, t0, t1)) continue;
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
                if (!query_node(qc, txy, pid, L, idx, x0, y0, s, dmin, dmax, thick)) continue;
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
            if (!ray_aabb(ro, rd, rd_inv, bvh[i].amin, bvh[i].amax, t0, t1)) continue;
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
                if (!query_node(qc, txy, pid, L, idx, x0, y0, s, dmin, dmax, thick)) continue;
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
    if (ray_aabb(ro, rd, rd_inv, bvh[0].amin, bvh[0].amax, rt0, rt1)) {
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
#ifdef PERF_ENABLED
            if (u_perf != 0) atomicAdd(perf[0], 1u);
#endif
            int pid = int(b.val);
            PatchInfo pi = patches[pid];
            uvec4 r = rects[pid];

            float dD = dot(rd, pi.basis_u);
            if (abs(dD) <= 1.0e-6) continue;

            if (pi.double_sided == 0u && dD > 0.0) continue;

            float D0 = dot(ro - u_depth_origin, pi.basis_u);
            float qstep = (u_depth_hi - u_depth_lo) / 65534.0;   // 16-bit depth
            float htol = max(2.0 * qstep, 1.0e-4);   // depth quantisation slack

            float tex_s = min(pi.proj_size.x / float(max(r.z, 1u)),
                              pi.proj_size.y / float(max(r.w, 1u)));
            vec2 rd_uv = vec2(dot(rd, pi.basis_v), dot(rd, pi.basis_w));
            float uv_speed = max(length(rd_uv), 1.0e-4);
            float base_step = clamp(min(0.5 * tex_s, tex_s / uv_speed),
                                    1.0e-6, tout - tin);

            float t = tin;
            bool have_prev = false;
            float t_prev = tin;
            bool prev_before = false;
            int guard = 0;
            while (guard < 2048 && t <= tout) {
                guard++;
#ifdef PERF_ENABLED
                if (u_perf != 0) atomicAdd(perf[2], 1u);
#endif
                vec3 P = ro + rd * t;
                vec2 proj = vec2(dot(P, pi.basis_v), dot(P, pi.basis_w));
                vec2 nrm = (proj - pi.proj_min) / pi.proj_size;
                bool inside = nrm.x >= 0.0 && nrm.y >= 0.0 && nrm.x <= 1.0 && nrm.y <= 1.0;
                float dm = 0.0, dmax = 0.0, thk = 0.0;
                int L, idx, x0, y0, s;
                bool covered = false;
                if (inside) {
                    ivec2 txy = ivec2(int(r.x) + int(nrm.x * float(r.z)),
                                      int(r.y) + int(nrm.y * float(r.w)));
                    covered = query_node(qc, txy, pid, L, idx, x0, y0, s, dm, dmax, thk);
                    if (covered) dbg_covered++;
                    if (!covered) inside = false;
                }
                float D = dot(P - u_depth_origin, pi.basis_u);
                float step = base_step;
                if (inside && covered && s > LEAF_TILE) {
                    float stride = clamp(min(0.5 * tex_s * float(s),
                                             tex_s * float(s) / uv_speed),
                                         1.0e-6, tout - tin);
                    if (D - abs(dD) * stride > dmax + htol) step = stride;
                }
                float band_hi = inside ? min(dmax, dm + thk) : 0.0;
                if (inside && D >= dm - htol && D <= band_hi + htol) {
                    // Snap the hit to the band midpoint (texel-centre surface)
                    // instead of leaving it at the first sample inside the
                    // conservative band; see refine_band_mid.
                    float th = t;
                    if (have_prev && t > t_prev) th = refine_band_mid(qc, pid, ro, rd, pi, r, t_prev, t);
                    best_t = th;
                    best_pid = pid;
                    best_L = L; best_idx = idx;
                    best_x0 = x0; best_y0 = y0; best_s = s;
                    break;
                }
                bool cur_before = !inside || (D > band_hi + htol);

                if (have_prev && (prev_before != cur_before)) {
                    float ta = t_prev;
                    float tb = t;
                    for (int it = 0; it < 8; ++it) {
#ifdef PERF_ENABLED
                        if (u_perf != 0) atomicAdd(perf[3], 1u);
#endif
                        float tm = (ta + tb) * 0.5;
                        vec3 Pm = ro + rd * tm;
                        vec2 proj_m = vec2(dot(Pm, pi.basis_v), dot(Pm, pi.basis_w));
                        vec2 nrm_m = (proj_m - pi.proj_min) / pi.proj_size;
                        if (nrm_m.x >= 0.0 && nrm_m.y >= 0.0 && nrm_m.x <= 1.0 && nrm_m.y <= 1.0) {
                            ivec2 txy_m = ivec2(int(r.x) + int(nrm_m.x * float(r.z)),
                                                int(r.y) + int(nrm_m.y * float(r.w)));
                            int Lm, idxm, x0m, y0m, sm;
                            float dm_m, dmax_m, thk_m;
                            if (query_node(qc, txy_m, pid, Lm, idxm, x0m, y0m, sm, dm_m, dmax_m, thk_m)) {
                                float Dm = dot(Pm - u_depth_origin, pi.basis_u);
                                // Converge on the band midpoint (texel-centre
                                // surface) rather than the band front (footprint
                                // near edge); see refine_band_mid.
                                if (Dm <= 0.5f * (dm_m + min(dmax_m, dm_m + thk_m))) tb = tm; else ta = tm;
                                continue;
                            }
                        }
                        ta = tm;
                    }
                    float th = (ta + tb) * 0.5;

                    vec3 P_f = ro + rd * th;
                    vec2 proj_f = vec2(dot(P_f, pi.basis_v), dot(P_f, pi.basis_w));
                    vec2 nrm_f = (proj_f - pi.proj_min) / pi.proj_size;
                    if (nrm_f.x < 0.0 || nrm_f.y < 0.0 || nrm_f.x > 1.0 || nrm_f.y > 1.0) { if (!dbg_captured) { dbg_captured = true; dbg_reject = 4; miss_nrm_f = nrm_f; } }
                    else {
                        ivec2 txy_f = ivec2(int(r.x) + int(nrm_f.x * float(r.z)),
                                            int(r.y) + int(nrm_f.y * float(r.w)));
                        int Lf, idxf, x0f, y0f, sf;
                        float dmf, dmxf, thkf;
                        if (!query_node(qc, txy_f, pid, Lf, idxf, x0f, y0f, sf, dmf, dmxf, thkf)) { if (!dbg_captured) { dbg_captured = true; dbg_reject = 5; miss_nrm_f = nrm_f; miss_L = Lf; } }
                        else {
                            float D_f = dot(ro + rd * th - u_depth_origin, pi.basis_u);
                            float band_hi_f = min(dmxf, dmf + thkf);
                            float slack = max(htol, 2.0 * abs(dD) * step);
                            if (D_f < dmf - htol - slack || D_f > band_hi_f + htol + slack) { if (!dbg_captured) { dbg_captured = true; dbg_reject = 3; miss_nrm_f = nrm_f; miss_L = Lf; } }
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
                if (t >= tout) break;
                t += step;
                if (t > tout) t = tout;
            }
#ifdef PERF_ENABLED
            if (guard >= 2048 && u_perf != 0) atomicAdd(perf[4], 1u);
#endif
        } else {
            dbg_pushed = true;
            int left = node + 1;
            int right = node + int(b.val);
            float t0, t1, t2, t3;
            bool hit_l = ray_aabb(ro, rd, rd_inv, bvh[left].amin,  bvh[left].amax,  t0, t1);
            bool hit_r = ray_aabb(ro, rd, rd_inv, bvh[right].amin, bvh[right].amax, t2, t3);
            if (hit_l && hit_r) {
                if (t0 <= t2) {
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
            imageStore(u_out, p, vec4(float(dbg_covered) / 30.0,
                                      float(dbg_reject) / 8.0,
                                      float(dbg_leaf_visits) / 30.0, 1.0));
            return;
        }
        if (u_dbg == 12) {
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
            if (!ray_aabb(ro, rd, rd_inv, bvh[i].amin, bvh[i].amax, t0, t1)) continue;
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
                if (!query_node(qc, txy, pid, L, idx, x0, y0, s, dmin, dmax, thick)) continue;
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
    PatchInfo pi = patches[best_pid];
    int vo = u_value_off_dt[best_L] + best_idx;
    uvec4 qn = texelFetch(u_dtm_buf, vo);
    uint nx8 = (qn.b >> 8u) & 0xFFu;
    uint ny8 = qn.b & 0xFFu;
    vec2 oct = vec2(decode_value(float(nx8) / 255.0, -1.0, 1.0, 8),
                    decode_value(float(ny8) / 255.0, -1.0, 1.0, 8));
    vec3 n = octahedral_decode(oct);
    if (dot(n, rd) > 0.0) n = -n;

    if (u_dbg == 9) {
        imageStore(u_out, p, vec4(float(best_x0) / 512.0,
                                  float(best_y0) / 512.0,
                                  float(best_s) / 30.0, 1.0));
        return;
    }

    if (u_dbg == 10) {
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
    GLuint dtm_ssbo = 0, dtm_tex = 0;     // merged meta+depth+thick+normal buffer texture
    GLuint cover_ssbo = 0;                // packed per-texel coverage mask (binding 5)
    GLuint fbo = 0, view_tex = 0;
    int view_w = 0, view_h = 0;
    int level_count = 0;
    int level_off[32] = {};
    int value_off[4][32] = {};
    int value_off_dt[32] = {};            // per-level offsets into the merged buffer
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

// Per-pass GPU/CPU timing with double-buffered GL_TIME_ELAPSED queries.
// Only one time-elapsed query may be active at once, so passes are timed
// sequentially (never nested); the frame timer is CPU-only.
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

    // Reads the just-finished frame's query result (never re-begun until the
    // next frame, so the read cannot block a live query). Samples accumulate
    // into the current window for the 0.5 s averaged display.
    void readback() {
        gpu_ms_ = 0.0;
        if (ran_) {
            if (gpu_) gpu_ms_ = double(q_prev_.result()) * 1.0e-6;   // ns -> ms
            cpu_acc_ += cpu_ms_;
            gpu_acc_ += gpu_ms_;
            n_++;
        }
        win_cpu_acc_ += cpu_ms_;
        win_gpu_acc_ += gpu_ms_;
        win_n_++;
    }

    // Called every ~0.5 s: averages the accumulated samples into disp_* (what
    // the ImGui window shows) and appends that average to the history plots.
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
    double cpu_ms() const { return cpu_ms_; }
    double gpu_ms() const { return gpu_ms_; }
    double disp_cpu() const { return disp_cpu_; }
    double disp_gpu() const { return disp_gpu_; }
    double avg_cpu() const { return n_ ? cpu_acc_ / double(n_) : 0.0; }
    double avg_gpu() const { return n_ ? gpu_acc_ / double(n_) : 0.0; }
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

// Per-phase shader-clock breakdown of the ray kernel, mirrored on PassTimer's
// 0.5 s display window: readback() per dispatch, flush_window() publishes the
// averaged cycles/ray values the ImGui window shows so the readout is stable.
class PhaseAccum {
public:
    void readback(const GLuint pv[12]) {
        for (int k = 0; k < 6; ++k) {
            cyc_acc_[k] += pv[6 + k];
            win_cyc_acc_[k] += pv[6 + k];
        }
        rays_acc_ += pv[5];
        win_rays_acc_ += pv[5];
    }

    void flush_window() {
        if (win_rays_acc_ > 0.0)
            for (int k = 0; k < 6; ++k) disp_cyc_[k] = win_cyc_acc_[k] / win_rays_acc_;
        for (int k = 0; k < 6; ++k) win_cyc_acc_[k] = 0.0;
        win_rays_acc_ = 0.0;
    }

    double disp_cyc(int k) const { return disp_cyc_[k]; }
    double avg_cyc(int k) const {
        return rays_acc_ > 0.0 ? cyc_acc_[k] / rays_acc_ : 0.0;
    }

private:
    double cyc_acc_[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double win_cyc_acc_[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double disp_cyc_[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double rays_acc_ = 0.0, win_rays_acc_ = 0.0;
};

// Horizontal stacked bar (segment widths ∝ time) drawn with the ImGui draw list.
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

// Two-column legend next to a stacked bar: color swatch + name | ms + share.
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

    // Merged meta + depth + thickness + normal buffer texture for the ray
    // pass: one RGBA32UI texel per quadtree node
    //   R = meta32, G = (dmin16 << 16) | dmax16, B = (thick8 << 24) | (nrmx8 << 8) | nrm_y8
    // All four chains share the quadtree topology, so the per-level offsets
    // equal the depth chain's (value_off[2]) and every descent/finalize step
    // becomes a single texelFetch instead of an SSBO meta read + value fetch.
    {
        const auto& dc = atlas.depth_chain();
        const auto& tc = atlas.thickness_chain();
        const auto& nc = atlas.normal_chain();
        std::vector<glm::uvec4> dtm;
        size_t tot = 0;
        for (const auto& lv : dc.levels)
            tot += lv.data.size() / (size_t(dc.channels) * size_t(dc.bytes_per_channel));
        dtm.reserve(tot);
        for (int L = 0; L < v.level_count && L < 32; ++L) {
            const auto& dl = dc.levels[size_t(L)];
            const auto& tl = tc.levels[size_t(L)];
            const auto& nl = nc.levels[size_t(L)];
            const auto& ml = ref.levels[size_t(L)];
            size_t n = dl.data.size() / (size_t(dc.channels) * size_t(dc.bytes_per_channel));
            const uint16_t* dp = reinterpret_cast<const uint16_t*>(dl.data.data());
            for (size_t i = 0; i < n; ++i) {
                uint16_t dmin = dp[i * 2];
                uint16_t dmax = dp[i * 2 + 1];
                uint8_t th = uint8_t(tl.data[i]);
                uint8_t nx = uint8_t(nl.data[i * 2]);
                uint8_t ny = uint8_t(nl.data[i * 2 + 1]);
                dtm.push_back(glm::uvec4(
                    ml.meta[i],
                    (uint32_t(dmin) << 16) | uint32_t(dmax),
                    (uint32_t(th) << 24) | (uint32_t(nx) << 8) | uint32_t(ny),
                    0u));
            }
        }
        if (!v.dtm_ssbo) glGenBuffers(1, &v.dtm_ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, v.dtm_ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, dtm.size() * sizeof(glm::uvec4),
                     dtm.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        if (!v.dtm_tex) glGenTextures(1, &v.dtm_tex);
        glBindTexture(GL_TEXTURE_BUFFER, v.dtm_tex);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32UI, v.dtm_ssbo);
        glBindTexture(GL_TEXTURE_BUFFER, 0);
        for (int L = 0; L < v.level_count && L < 32; ++L)
            v.value_off_dt[L] = v.value_off[2][L];
    }

    // Per-texel coverage mask, packed 4 bytes per uint (binding 5 in the ray
    // shader). Resolves the exact texel's coverage after an early descent stop.
    {
        const auto& cov = atlas.coverage();
        std::vector<uint32_t> packed((cov.size() + 3) / 4, 0u);
        for (size_t i = 0; i < cov.size(); ++i)
            packed[i >> 2] |= uint32_t(cov[i]) << ((i & 3u) << 3u);
        if (!v.cover_ssbo) glGenBuffers(1, &v.cover_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, v.cover_ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, packed.size() * sizeof(uint32_t),
                     packed.data(), GL_STATIC_DRAW);
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

// Assembles a ray compute program from the shared preamble + a body. The perf
// counters compile in only when `perf` is set (PERF_ENABLED macro), so the
// hot production path carries no uniform branch.
static bool build_ray_program(gl::Program& prog, const char* common, const char* body,
                              bool perf) {
    std::string src = "#version 460 core\n";
    if (perf) src += "#define PERF_ENABLED 1\n";
    src += common;
    src += "\n";
    src += body;
    gl::Shader cs(gl::ShaderType::compute, src);
    if (!cs.compiled()) {
        std::fprintf(stderr, "Compute shader compile failed:\n%s\n",
                     cs.info_log().c_str());
        return false;
    }
    prog.attach(cs);
    if (!prog.link()) {
        std::fprintf(stderr, "Compute program link failed:\n%s\n",
                     prog.info_log().c_str());
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

// Exact patch under the cursor: brute-force ray-triangle test over every
// patch's own triangles (the scene is small, runs once per click). The BVH
// pick above only finds the nearest *AABB* entry, which on a curved surface is
// often a neighbour patch whose box is entered first but whose surface is
// behind the clicked one. Returns the patch id of the nearest actual surface
// hit, or -1.
static int pick_patch_exact(const gfx::CoverageAtlas& atlas,
                            const glm::vec3& ro, const glm::vec3& rd) {
    const auto& patches = atlas.patches();
    const auto& tris    = atlas.triangles();
    const auto& pos     = atlas.positions();
    int best_patch = -1;
    float best_t = 1e30f;
    for (size_t pi = 0; pi < patches.size(); ++pi) {
        for (int ti : patches[pi].tris) {
            const auto& tr = tris[size_t(ti)];
            const glm::vec3& a = pos[tr.v[0]];
            const glm::vec3& b = pos[tr.v[1]];
            const glm::vec3& c = pos[tr.v[2]];
            glm::vec3 e1 = b - a, e2 = c - a;
            glm::vec3 h = glm::cross(rd, e2);
            float det = glm::dot(e1, h);
            if (std::fabs(det) < 1e-12f) continue;
            float inv = 1.0f / det;
            glm::vec3 s = ro - a;
            float u = glm::dot(s, h) * inv;
            if (u < 0.0f || u > 1.0f) continue;
            glm::vec3 q = glm::cross(s, e1);
            float v = glm::dot(rd, q) * inv;
            if (v < 0.0f || u + v > 1.0f) continue;
            float t = glm::dot(e2, q) * inv;
            if (t < 0.0f || t >= best_t) continue;
            best_t = t;
            best_patch = int(pi);
        }
    }
    return best_patch;
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
    glm::vec3 model_aabb_min{0.0f}, model_aabb_max{0.0f};  // global model bounds
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
        st.model_aabb_min = lo;
        st.model_aabb_max = hi;
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

    // --- Primary-ray pass + its fullscreen display ---
    // Two kernels share the preamble: the lean production kernel (no debug
    // modes, no runtime perf branch) and the full diagnostic kernel. Each is
    // built with and without perf counters, so the common path is branch-free.
    const int ray_dbg = ([] {
        const char* e = getenv("RAYDBG");
        return e ? atoi(e) : 0;
    })();
    const int perf_enabled = getenv("PERF") ? 1 : 0;

    gl::Program ray_lean_prog, ray_lean_perf_prog, ray_debug_prog, ray_debug_perf_prog;
    if (!build_ray_program(ray_lean_prog, ray_common_glsl, ray_lean_main, false))
        return EXIT_FAILURE;
    if (perf_enabled &&
        !build_ray_program(ray_lean_perf_prog, ray_common_glsl, ray_lean_main, true))
        return EXIT_FAILURE;
    if (ray_dbg != 0 &&
        !build_ray_program(ray_debug_prog, ray_common_glsl, ray_debug_main, false))
        return EXIT_FAILURE;
    if (ray_dbg != 0 && perf_enabled &&
        !build_ray_program(ray_debug_perf_prog, ray_common_glsl, ray_debug_main, true))
        return EXIT_FAILURE;
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
    int selected_patch = -1;      // leaf patch id picked in the 3D view (-1 = none)
    bool highlight_atlas_patch = true;  // tint the picked patch in the atlas view
    bool left_prev = false;
    bool press_in_view = false;
    double press_x = 0.0, press_y = 0.0;

    // --- Primary-ray pass ---
    bool show_atlas = true;       // full MIP-chain reconstruction into the view FBO
    bool show_ray_pass = true;    // fullscreen normal view
    float ray_connect_tol = 0.05f;
    float ray_eps = 0.02f;

    // --- Per-pass GPU/CPU timers + optional per-dispatch work counters ---
    // t_frame spans the whole loop iteration (poll → present) and is CPU-only;
    // the GPU frame time is the sum of the individual GPU passes.
    PassTimer t_frame("frame", false);
    PassTimer t_poll("poll_events", false);    // poll_events + gui.begin_frame
    PassTimer t_input("camera_input", false);  // orbit/zoom + mouse pick + resize
    PassTimer t_geo("geometry");
    PassTimer t_light("lighting");
    PassTimer t_ray("ray");
    PassTimer t_ray_disp("ray_display");       // fullscreen quad, only when enabled
    PassTimer t_atlas("atlas_view");
    PassTimer t_bvh("bvh_overlay");
    PassTimer t_imgui("imgui", false);         // building the debug window
    PassTimer t_present("present", false);     // gui.render + swap_buffers

    GLuint perf_ssbo = 0;
    GLuint perf_acc[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    PhaseAccum phase_acc;
    int perf_frames = 0;
    if (perf_enabled) {
        glGenBuffers(1, &perf_ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, perf_ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, 14 * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    double prev_x, prev_y;
    window.cursor_position(prev_x, prev_y);

    gl::enable(GL_DEPTH_TEST);

    std::vector<float> frame_times;
    double last = window.time();
    int frame = 0;

    // 0.5 s display window: frame-period samples accumulate into period_* and
    // are averaged into disp_frame_period when the window elapses.
    double win_start = window.time();
    double period_acc = 0.0;
    int period_n = 0;
    double disp_frame_period = 0.0;

    // TEMP: scripted orbit sweep for grazing-angle diagnostics (SWEEP=<frames>).
    int sweep_total = 0;
    int sweep_frame = 0;
    if (const char* se = getenv("SWEEP")) sweep_total = std::max(1, atoi(se));

    auto print_perf = [&]() {
        double sum_gpu = t_geo.avg_gpu() + t_light.avg_gpu() + t_ray.avg_gpu() +
                         t_ray_disp.avg_gpu() + t_atlas.avg_gpu() + t_bvh.avg_gpu();
        printf("PERF avg over %d frames:\n", frame);
        auto row = [](const char* n, double g, double c) {
            printf("  %-11s gpu %7.2f ms  cpu %7.2f ms\n", n, g, c);
        };
        row("frame", sum_gpu, t_frame.avg_cpu());
        row("poll_events", 0.0, t_poll.avg_cpu());
        row("camera_input", 0.0, t_input.avg_cpu());
        row("geometry", t_geo.avg_gpu(), t_geo.avg_cpu());
        row("lighting", t_light.avg_gpu(), t_light.avg_cpu());
        row("ray", t_ray.avg_gpu(), t_ray.avg_cpu());
        row("ray_display", t_ray_disp.avg_gpu(), t_ray_disp.avg_cpu());
        row("atlas_view", t_atlas.avg_gpu(), t_atlas.avg_cpu());
        row("bvh_overlay", t_bvh.avg_gpu(), t_bvh.avg_cpu());
        row("imgui", 0.0, t_imgui.avg_cpu());
        row("present", 0.0, t_present.avg_cpu());
        if (perf_enabled && perf_frames > 0) {
            double rays = double(perf_acc[5]);
            printf("  ray work (accumulated over %d dispatches):\n", perf_frames);
            printf("    rays / frame         = %.0f\n", rays / double(perf_frames));
            if (rays > 0.0) {
                printf("    leaf visits / ray   = %.2f\n", double(perf_acc[0]) / rays);
                printf("    texel queries / ray = %.2f\n", double(perf_acc[1]) / rays);
                printf("    march iters / ray   = %.2f\n", double(perf_acc[2]) / rays);
                printf("    bisect iters / ray  = %.2f\n", double(perf_acc[3]) / rays);
                printf("    guard exits / ray   = %.4f\n", double(perf_acc[4]) / rays);
                printf("    max BVH stack depth = %u   (MAX_STACK=16)\n", perf_acc[12]);
                printf("    max quadtree level  = %u   (MAX_MIP=7)\n", perf_acc[13]);

                // Per-phase shader-clock cycle buckets (low 32 bits of the
                // GPU cycle counter, lifetime average), normalised to the
                // whole-dispatch GPU ms.
                const char* ph_name[6] = {"setup", "traversal", "leaf_setup",
                                          "march", "bisect", "finalize"};
                double ph_cyc[6];
                for (int k = 0; k < 6; ++k) ph_cyc[k] = phase_acc.avg_cyc(k);
                double tot_cyc = 0.0;
                for (int k = 0; k < 6; ++k) tot_cyc += ph_cyc[k];
                double gpu_ms = t_ray.avg_gpu();
                if (tot_cyc > 0.0) {
                    printf("    phase cycles/ray (low32 shader clock):\n");
                    for (int k = 0; k < 6; ++k)
                        printf("      %-10s %10.1f  %5.1f%%   ~%.4f ms\n",
                               ph_name[k], ph_cyc[k],
                               100.0 * ph_cyc[k] / tot_cyc,
                               gpu_ms * ph_cyc[k] / tot_cyc);
                    printf("      (sum %.1f cyc/ray; dispatch %.2f ms; shader clock counts issue cycles, not latency)\n",
                           tot_cyc, gpu_ms);
                }
            }
        }
    };

    while (!window.should_close()) {
        ++frame;
        double now = window.time();
        float dt = float(now - last);
        last = now;

        if (frame_times.size() >= 120) frame_times.erase(frame_times.begin());
        frame_times.push_back(dt * 1000.0f);
        period_acc += dt * 1000.0f;
        period_n++;

        // Whole-frame CPU timer: spans poll_events → present. The GPU frame
        // time is the sum of the individual GPU passes below.
        t_frame.begin();

        t_poll.begin();
        window.poll_events();
        gui.begin_frame();
        t_poll.end();

        if (sweep_total && sweep_frame >= sweep_total) break;
        if (sweep_total) {
            float ang = 2.0f * 3.14159265f * float(sweep_frame) / float(sweep_total);
            float r = 3.8f;
            glm::vec3 eye(r * std::cos(ang), 1.0f, r * std::sin(ang));
            cam.look_at(eye, glm::vec3(0.0f, 1.0f, 0.0f));
        }

        t_input.begin();
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

        // Left click in the 3D view: ray-cast against the BVH to pick the
        // patch under the cursor (highlighted in the atlas view). Clicks that
        // press or release over an ImGui window never pick.
        bool left_now = window.mouse_down(gfx::MouseButton::left);
        if (left_now && !left_prev && !gui.wants_mouse()) {
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
                    // Atlas highlight follows the exact surface triangle hit
                    // (the BVH pick above can grab a neighbouring patch whose
                    // AABB is entered first).
                    selected_patch = pick_patch_exact(atlas, ro, rd);
                }
            }
            press_in_view = false;
        }
        left_prev = left_now;

        cam.set_aspect(float(window.framebuffer_width()) / float(window.framebuffer_height()));

        if (window.framebuffer_width() != gbuf.width() || window.framebuffer_height() != gbuf.height())
            gbuf.create(window.framebuffer_width(), window.framebuffer_height());
        t_input.end();

        glm::mat4 vp = cam.view_projection();

        // --- Geometry pass ---
        t_geo.begin();
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
        t_geo.end();

        gl::disable(GL_DEPTH_TEST);

        // --- Lighting / patch pass ---
        t_light.begin();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        gl::viewport(0, 0, window.framebuffer_width(), window.framebuffer_height());
        gl::clear_color(0.03f, 0.03f, 0.05f, 1.0f);
        gl::clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (show_patches) {
            gl::enable(GL_DEPTH_TEST);
            patch_prog.use();
            loc = patch_prog.uniform_location("u_vp");
            if (loc >= 0) patch_prog.uniform_matrix4fv(loc, glm::value_ptr(vp));
            loc = patch_prog.uniform_location("u_sel_patch");
            if (loc >= 0) patch_prog.uniform1i(loc, selected_patch);
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
        t_light.end();

        // --- Primary-ray pass: BVH traversal + atlas mip-chain refinement ---
        if (show_ray_pass) {
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

                t_ray.begin();
                gl::Program* rp;
                if (ray_dbg != 0)
                    rp = perf_enabled ? &ray_debug_perf_prog : &ray_debug_prog;
                else
                    rp = perf_enabled ? &ray_lean_perf_prog : &ray_lean_prog;
                rp->use();
                auto r_loc = [&](const char* n) { return rp->uniform_location(n); };
                glm::mat4 inv_vp = glm::inverse(vp);
                glm::vec3 ro = cam.position();
                loc = r_loc("u_vp_inv");
                if (loc >= 0) rp->uniform_matrix4fv(loc, glm::value_ptr(inv_vp));
                loc = r_loc("u_res");        if (loc >= 0) rp->uniform2f(loc, float(ray_pass.w), float(ray_pass.h));
                loc = r_loc("u_ro");         if (loc >= 0) rp->uniform3fv(loc, glm::value_ptr(ro));
                {
                    // Exact per-pixel ray basis: the unprojected direction to a
                    // pixel is (col0*ndc.x + col1*ndc.y + col2 + col3).xyz scaled
                    // by a per-pixel scalar, which normalize() drops, so
                    // rd = normalize(u_rd0 + ndc.x*u_rd_dx + ndc.y*u_rd_dy) is
                    // identical to the old mat4 * ndc + perspective-divide.
                    glm::vec4 c0 = inv_vp[0], c1 = inv_vp[1],
                              c2 = inv_vp[2], c3 = inv_vp[3];
                    glm::vec3 rd0 = glm::vec3(c2 + c3) - ro * (c2.w + c3.w);
                    glm::vec3 rdx = glm::vec3(c0) - ro * c0.w;
                    glm::vec3 rdy = glm::vec3(c1) - ro * c1.w;
                    glm::vec2 inv_res = 1.0f / glm::vec2(float(ray_pass.w), float(ray_pass.h));
                    glm::vec2 ndc_scale = 2.0f * inv_res;
                    glm::vec2 ndc_bias = inv_res - 1.0f;
                    loc = r_loc("u_ndc_scale"); if (loc >= 0) rp->uniform2f(loc, ndc_scale.x, ndc_scale.y);
                    loc = r_loc("u_ndc_bias");  if (loc >= 0) rp->uniform2f(loc, ndc_bias.x, ndc_bias.y);
                    loc = r_loc("u_rd0");       if (loc >= 0) rp->uniform3fv(loc, glm::value_ptr(rd0));
                    loc = r_loc("u_rd_dx");     if (loc >= 0) rp->uniform3fv(loc, glm::value_ptr(rdx));
                    loc = r_loc("u_rd_dy");     if (loc >= 0) rp->uniform3fv(loc, glm::value_ptr(rdy));
                    const auto& bn = atlas.bvh_nodes();
                    if (!bn.empty()) {
                        loc = r_loc("u_root_amin");
                        if (loc >= 0) rp->uniform3f(loc, bn[0].aabb_min.x, bn[0].aabb_min.y, bn[0].aabb_min.z);
                        loc = r_loc("u_root_amax");
                        if (loc >= 0) rp->uniform3f(loc, bn[0].aabb_max.x, bn[0].aabb_max.y, bn[0].aabb_max.z);
                    }
                }
                loc = r_loc("u_depth_origin"); if (loc >= 0) rp->uniform3fv(loc, glm::value_ptr(ray_depth_origin));
                loc = r_loc("u_depth_lo");   if (loc >= 0) rp->uniform1f(loc, ray_depth_lo);
                loc = r_loc("u_depth_hi");   if (loc >= 0) rp->uniform1f(loc, ray_depth_hi);
                loc = r_loc("u_thick_max");  if (loc >= 0) rp->uniform1f(loc, ray_thick_max);
                loc = r_loc("u_eps");        if (loc >= 0) rp->uniform1f(loc, ray_eps);
                loc = r_loc("u_dbg");        if (loc >= 0) rp->uniform1i(loc, ray_dbg);
                loc = r_loc("u_perf");       if (loc >= 0) rp->uniform1i(loc, perf_enabled);
                loc = r_loc("u_patch_count"); if (loc >= 0) rp->uniform1i(loc, int(patch_gpu.patch_count));
                loc = r_loc("u_level_off");  if (loc >= 0) glUniform1iv(loc, 32, atlas_view.level_off);
                loc = r_loc("u_value_off_dt"); if (loc >= 0) glUniform1iv(loc, 32, atlas_view.value_off_dt);
                loc = r_loc("u_dtm_buf");    if (loc >= 0) rp->uniform1i(loc, 0);

                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_BUFFER, atlas_view.dtm_tex);

                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, atlas_view.rects_ssbo);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, atlas_view.meta_ssbo);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ray_pass.patch_info_ssbo);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ray_pass.bvh_ssbo);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, atlas_view.cover_ssbo);
                loc = r_loc("u_atlas_w"); if (loc >= 0) rp->uniform1i(loc, atlas_view.view_w);
                if (perf_enabled) {
                    GLuint z[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, perf_ssbo);
                    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(z), z);
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, perf_ssbo);
                }
                glBindImageTexture(0, ray_pass.ray_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
                gl::dispatch_compute((ray_pass.w + 7) / 8, (ray_pass.h + 7) / 8, 1);
                gl::memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                                   GL_TEXTURE_FETCH_BARRIER_BIT |
                                   GL_SHADER_STORAGE_BARRIER_BIT);
                glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
                if (perf_enabled) {
                    GLuint pv[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, perf_ssbo);
                    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(pv), pv);
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
                    for (int k = 0; k < 12; ++k) perf_acc[k] += pv[k];
                    // Max-tracked counters: keep the per-frame maximum.
                    perf_acc[12] = std::max(perf_acc[12], pv[12]);
                    perf_acc[13] = std::max(perf_acc[13], pv[13]);
                    phase_acc.readback(pv);
                    perf_frames++;
                }
                t_ray.end();

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
                if (loc2 >= 0) glGetUniformiv(rp->handle(), loc2, &upc);
                loc2 = r_loc("u_res");
                if (loc2 >= 0) glGetUniformiv(rp->handle(), loc2, ures);
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
                t_ray_disp.begin();
                    gl::disable(GL_DEPTH_TEST);
                    ray_disp_prog.use();
                    loc = ray_disp_prog.uniform_location("u_tex");
                    if (loc >= 0) ray_disp_prog.uniform1i(loc, 0);
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, ray_pass.ray_tex);
                    quad_vao.bind();
                    gl::draw_elements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
                    t_ray_disp.end();
                } else {
                    t_ray_disp.skip();
                }
        }

        // --- Atlas texture reconstruction (full MIP4 chain walk, into view FBO) ---
        if (show_atlas && atlas_view.view_tex) {
            t_atlas.begin();
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
            loc = a_loc("u_sel_patch");   if (loc >= 0) atlas_prog.uniform1i(loc, selected_patch);
            loc = a_loc("u_sel_on");      if (loc >= 0) atlas_prog.uniform1i(loc, highlight_atlas_patch ? 1 : 0);
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
            t_atlas.end();
        } else {
            t_atlas.skip();
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
            t_bvh.begin();
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
            t_bvh.end();
        } else {
            t_bvh.skip();
        }

        // Selected-patch box in 3D: always drawn (independent of show_bvh) so
        // the user sees exactly which region the atlas highlight corresponds to.
        if (selected_patch >= 0 && size_t(selected_patch) < atlas.patches().size()) {
            const auto& sp = atlas.patches()[size_t(selected_patch)];
            t_bvh.begin();
            bvh_draw.clear();
            glm::vec3 c = (sp.aabb_min + sp.aabb_max) * 0.5f;
            glm::vec3 half = (sp.aabb_max - sp.aabb_min) * 0.5f + glm::vec3(0.006f);
            bvh_draw.draw_box(c - half, c + half, {1.0f, 0.08f, 0.15f, 1.0f});
            gl::disable(GL_DEPTH_TEST);
            bvh_draw.render(vp);
            t_bvh.end();
        }

        t_imgui.begin();
        // --- ImGui debug window ---
        {
            ImGui::Begin("MDC Lighting - Debug");
            ImGui::Text("Frametime (avg 0.5s): %.3f ms", disp_frame_period);
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            ImGui::PlotLines("frame period ms (raw)", frame_times.data(), int(frame_times.size()),
                             0, nullptr, 0.0f, 40.0f, ImVec2(0, 50));
            ImGui::Separator();

            // ---- Frame timing breakdown ----
            // Order matches the order the segments run in each frame.
            struct PassRow { const char* name; const PassTimer* t; };
            const PassRow passes[] = {
                {"poll events", &t_poll},
                {"camera input", &t_input},
                {"geometry", &t_geo},
                {"lighting", &t_light},
                {"ray compute", &t_ray},
                {"ray display", &t_ray_disp},
                {"atlas view", &t_atlas},
                {"bvh overlay", &t_bvh},
                {"imgui build", &t_imgui},
                {"present", &t_present},
            };
            const int kPasses = int(sizeof(passes) / sizeof(passes[0]));
            const ImU32 pass_cols[kPasses] = {
                IM_COL32(236, 100, 75, 255),
                IM_COL32(243, 156, 18, 255),
                IM_COL32(52, 152, 219, 255),
                IM_COL32(46, 204, 113, 255),
                IM_COL32(155, 89, 182, 255),
                IM_COL32(52, 73, 94, 255),
                IM_COL32(241, 196, 15, 255),
                IM_COL32(26, 188, 156, 255),
                IM_COL32(230, 126, 34, 255),
                IM_COL32(192, 57, 43, 255),
            };

            double gpu_sum = 0.0;
            for (int i = 0; i < kPasses; ++i)
                if (passes[i].t->gpu()) gpu_sum += passes[i].t->disp_gpu();

            if (ImGui::BeginTable("ftable", 3,
                                  ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthFixed);
                ImGui::TableSetupColumn("GPU avg (ms)", ImGuiTableColumnFlags_WidthFixed);
                ImGui::TableSetupColumn("CPU avg (ms)", ImGuiTableColumnFlags_WidthFixed);
                ImGui::TableHeadersRow();
                for (int i = 0; i < kPasses; ++i) {
                    const PassTimer& t = *passes[i].t;
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", passes[i].name);
                    if (t.gpu()) {
                        ImGui::TableNextColumn(); ImGui::Text("%.3f", t.disp_gpu());
                    } else {
                        ImGui::TableNextColumn(); ImGui::Text("cpu-only");
                    }
                    ImGui::TableNextColumn(); ImGui::Text("%.3f", t.disp_cpu());
                }
                const ImVec4 total_col(1.0f, 0.82f, 0.3f, 1.0f);
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextColored(total_col, "frame total");
                ImGui::TableNextColumn(); ImGui::TextColored(total_col, "%.3f", gpu_sum);
                ImGui::TableNextColumn(); ImGui::TextColored(total_col, "%.3f", t_frame.disp_cpu());
                ImGui::EndTable();
            }

            // CPU time stacked bar across every segment of the frame.
            const char* cpu_names[kPasses];
            float cpu_vals[kPasses];
            for (int i = 0; i < kPasses; ++i) {
                cpu_names[i] = passes[i].name;
                cpu_vals[i] = float(passes[i].t->disp_cpu());
            }
            ImGui::Text("CPU time by segment (frame: %.3f ms):", t_frame.disp_cpu());
            float bar_w = ImGui::GetContentRegionAvail().x;
            if (bar_w < 64) bar_w = 256;
            imgui_stacked_bar(ImGui::GetCursorScreenPos(), ImVec2(bar_w, 16),
                              cpu_vals, pass_cols, kPasses);
            ImGui::Dummy(ImVec2(0, 16));
            imgui_stacked_legend("cpu_legend", cpu_names, cpu_vals, pass_cols, kPasses,
                                 float(t_frame.disp_cpu()));
            ImGui::Separator();

            // GPU time stacked bar across the GPU passes only.
            const char* gpu_names[kPasses];
            float gpu_vals[kPasses];
            ImU32 gpu_cols[kPasses];
            int gn = 0;
            for (int i = 0; i < kPasses; ++i)
                if (passes[i].t->gpu()) {
                    gpu_names[gn] = passes[i].name;
                    gpu_vals[gn] = float(passes[i].t->disp_gpu());
                    gpu_cols[gn] = pass_cols[i];
                    ++gn;
                }
            ImGui::Text("GPU time by pass (sum: %.3f ms):", gpu_sum);
            float gw = ImGui::GetContentRegionAvail().x;
            if (gw < 64) gw = 256;
            imgui_stacked_bar(ImGui::GetCursorScreenPos(), ImVec2(gw, 16),
                              gpu_vals, gpu_cols, gn);
            ImGui::Dummy(ImVec2(0, 16));
            imgui_stacked_legend("gpu_legend", gpu_names, gpu_vals, gpu_cols, gn,
                                 float(gpu_sum));
            ImGui::Separator();

            if (ImGui::CollapsingHeader("Pass history (120 windowed samples)")) {
                for (int i = 0; i < kPasses; ++i) {
                    const PassTimer& t = *passes[i].t;
                    if (!t.cpu_hist().empty()) {
                        ImGui::PlotLines(passes[i].name, t.cpu_hist().data(),
                                         int(t.cpu_hist().size()), 0, nullptr,
                                         0.0f, 40.0f, ImVec2(0, 26));
                    }
                }
            }

            if (perf_enabled && perf_frames > 0) {
                double rays = double(perf_acc[5]);
                if (rays > 0.0) {
                    ImGui::Text("Ray work / ray: leaf %.1f  texel %.1f  march %.1f  bisect %.1f",
                                double(perf_acc[0]) / rays, double(perf_acc[1]) / rays,
                                double(perf_acc[2]) / rays, double(perf_acc[3]) / rays);
                    ImGui::Text("Guard-limited rays: %.0f / %.0f",
                                double(perf_acc[4]), rays);

                    // Per-phase shader-clock cycles (0.5 s windowed average,
                    // mirrors the PERF console dump).
                    static const char* ph_names[6] = {"setup", "traversal",
                                                      "leaf_setup", "march",
                                                      "bisect", "finalize"};
                    const ImU32 ph_cols[6] = {
                        IM_COL32(52, 152, 219, 255),
                        IM_COL32(243, 156, 18, 255),
                        IM_COL32(46, 204, 113, 255),
                        IM_COL32(236, 100, 75, 255),
                        IM_COL32(155, 89, 182, 255),
                        IM_COL32(241, 196, 15, 255),
                    };
                    float ph_ms[6] = {0, 0, 0, 0, 0, 0};
                    double tot_cyc = 0.0;
                    for (int k = 0; k < 6; ++k) tot_cyc += phase_acc.disp_cyc(k);
                    if (tot_cyc > 0.0) {
                        double gpu_ms = t_ray.disp_gpu();
                        for (int k = 0; k < 6; ++k)
                            ph_ms[k] = float(gpu_ms * phase_acc.disp_cyc(k) / tot_cyc);
                        ImGui::Text("Ray shader cycles by phase (dispatch: %.3f ms):",
                                    gpu_ms);
                        float bw = ImGui::GetContentRegionAvail().x;
                        if (bw < 64) bw = 256;
                        imgui_stacked_bar(ImGui::GetCursorScreenPos(), ImVec2(bw, 16),
                                          ph_ms, ph_cols, 6);
                        ImGui::Dummy(ImVec2(0, 16));
                        imgui_stacked_legend("ray_phase_legend", ph_names, ph_ms, ph_cols,
                                             6, float(gpu_ms));
                    }
                }
            }
            ImGui::Separator();

            int cur_model_prev = cur_model;
            if (ImGui::Combo("Model", &cur_model,
                             "Cornell Box\0Stanford Bunny\0Stanford Dragon\0")) {
                if (load_scene(kModels[cur_model], model, st)) {
                    cam.look_at(st.cam_target + glm::vec3(0.0f, 0.0f, st.cam_dist), st.cam_target);
                    base_density = atlas.config().texel_density;
                    pick = RayHit{};
                    selected_patch = -1;
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

            ImGui::Checkbox("Show atlas reconstruction", &show_atlas);
            if (show_atlas)
                ImGui::Checkbox("Highlight picked patch in atlas", &highlight_atlas_patch);
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

            if (selected_patch >= 0 && size_t(selected_patch) < atlas.patches().size()) {
                const auto& sp = atlas.patches()[size_t(selected_patch)];
                ImGui::Text("Selected patch %d: atlas (%d,%d) %dx%d  axis %d  proj (%.4f, %.4f)",
                            selected_patch, sp.atlas_x, sp.atlas_y, sp.tex_w, sp.tex_h,
                            sp.axis, sp.proj_min.x, sp.proj_min.y);
                ImGui::Text("  aabb min (%.3f, %.3f, %.3f)  max (%.3f, %.3f, %.3f)",
                            sp.aabb_min.x, sp.aabb_min.y, sp.aabb_min.z,
                            sp.aabb_max.x, sp.aabb_max.y, sp.aabb_max.z);
            }
            ImGui::Text("Coverage atlas: %dx%d @ %.0f texels/unit",
                        atlas.atlas_width(), atlas.atlas_height(), atlas.final_density());
            ImGui::Combo("Atlas texture", &atlas_chain, "UV\0Thickness\0Depth\0Normal\0");
            ImGui::SliderFloat("Resolution scale", &density_scale, 0.25f, 4.0f, "%.2fx");
            // Each atlas texel is one cube cell of 1/texels_per_unit world
            // units, so the scale slider's real meaning is a voxel grid count
            // across the model's axes. Show it as such (prospective value the
            // slider would apply on "Recompute atlas").
            {
                glm::vec3 span = st.model_aabb_max - st.model_aabb_min;
                float longest = std::max(span.x, std::max(span.y, span.z));
                float est = base_density * density_scale;
                if (longest > 0.0f && est > 0.0f) {
                    ImGui::Text("Voxel grid: %.0f x %.0f x %.0f  (~%.0f^3, voxel %.4f u)",
                                span.x * est, span.y * est, span.z * est, longest * est,
                                1.0f / est);
                }
            }
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
                    selected_patch = -1;
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
                // Screen-space highlight of the picked patch: the shader tint is
                // diluted when the atlas is downscaled, so draw a thick border on
                // top of the image. The image is upright w.r.t. the atlas FBO:
                // image top-left = texture (s=0, t=1) = atlas row (0, view_h-1).
                if (highlight_atlas_patch && selected_patch >= 0 &&
                    size_t(selected_patch) < atlas.patches().size()) {
                    const auto& sp = atlas.patches()[size_t(selected_patch)];
                    const float vw = float(atlas_view.view_w), vh = float(atlas_view.view_h);
                    const ImVec2 mn = ImGui::GetItemRectMin();
                    const float ix0 = mn.x + (float(sp.atlas_x) / vw) * asz;
                    const float iy0 = mn.y + (1.0f - float(sp.atlas_y + sp.tex_h) / vh) * ah;
                    const float ix1 = mn.x + (float(sp.atlas_x + sp.tex_w) / vw) * asz;
                    const float iy1 = mn.y + (1.0f - float(sp.atlas_y) / vh) * ah;
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    dl->AddRect(ImVec2(ix0, iy0), ImVec2(ix1, iy1),
                                IM_COL32(255, 70, 55, 255), 0.0f, 0, 3.0f);
                    char lbl[64];
                    std::snprintf(lbl, sizeof(lbl), "patch %d", selected_patch);
                    dl->AddText(ImVec2(ix0 + 5, iy0 + 5), IM_COL32(255, 70, 55, 255), lbl);
                }
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
        t_imgui.end();

        t_present.begin();
        gui.render();
        window.swap_buffers();
        t_present.end();
        t_frame.end();

        t_frame.readback();
        t_poll.readback();
        t_input.readback();
        t_geo.readback();
        t_light.readback();
        t_ray.readback();
        t_ray_disp.readback();
        t_atlas.readback();
        t_bvh.readback();
        t_imgui.readback();
        t_present.readback();

        // Every ~0.5 s, average the accumulated samples into the values the
        // ImGui window displays so the readout is stable instead of jumping
        // on every frame.
        if (window.time() - win_start >= 0.5) {
            win_start = window.time();
            disp_frame_period = period_n ? period_acc / double(period_n) : 0.0;
            period_acc = 0.0;
            period_n = 0;
            t_frame.flush_window();
            t_poll.flush_window();
            t_input.flush_window();
            t_geo.flush_window();
            t_light.flush_window();
            t_ray.flush_window();
            t_ray_disp.flush_window();
            t_atlas.flush_window();
            t_bvh.flush_window();
            t_imgui.flush_window();
            t_present.flush_window();
            phase_acc.flush_window();
        }
        if (frame % 120 == 0 || (perf_enabled && frame % 5 == 0)) print_perf();
    }

    print_perf();

    return EXIT_SUCCESS;
}
